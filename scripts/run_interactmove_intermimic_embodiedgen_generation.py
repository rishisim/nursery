#!/usr/bin/env python3
"""Run EmbodiedGen's real prompt-to-layout and asset stages without a robot.

This wrapper intentionally reproduces the released ``gen_layout`` stages up
through BFS placement, but it does not call the released ``sim_cli`` tail.
That tail always constructs a Franka actor even when ``insert_robot`` is
false. The generated ``layout.json`` may retain EmbodiedGen's native robot
placement metadata; no robot asset or simulator actor is created here.
"""

from __future__ import annotations

import argparse
import base64
import copy
import gc
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


EMBODIEDGEN_COMMIT = "9b333554254af196bace88c1a171a3bf047fa09c"
EMBODIEDGEN_VERSION = "v2.0.1"
TRELLIS_SOURCE_COMMIT = "55a8e8164b195bbf927e0978f00e76c835e6011f"
TRELLIS_FLEXICUBES_COMMIT = "f97beb0dd3c6c68f3ab5696b6dcaf9af69f0514e"
TRELLIS_CHECKPOINT_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
DINOv2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
RECEIPT_SCHEMA = "InteractMoveInterMimicFreshEmbodiedGenReceipt"
RECEIPT_SCHEMA_VERSION = 1
OPENAI_API_MODE = "responses"
OPENAI_MODEL = "gpt-5.6-luna"
OPENAI_REASONING_EFFORT = "none"
OPENAI_SDK_VERSION = "3.1.0"
OPENAI_DEFAULT_MAX_OUTPUT_TOKENS = 8192
OPENAI_TIMEOUT_S = 120.0
MULTIVIEW_QUALITY_PREAMBLE = """
The image input contains four camera views of the same single generated 3D
asset. Repetition across views is not multiple object instances. Judge whether
each view consistently depicts that one asset, and only report duplicate
geometry when a duplicate exists within an individual view.
"""


class _ResponsesGPTClient:
    """Expose EmbodiedGen's GPTclient interface over OpenAI Responses."""

    _DEFAULT_SYSTEM_ROLE = (
        "You are a highly knowledgeable assistant specializing in physics, "
        "engineering, and object properties."
    )
    _IMAGE_MEDIA_TYPES = {
        ".gif": "image/gif",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    _IGNORED_CHAT_PARAMS = {
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "temperature",
        "top_p",
    }
    _TOKEN_PARAMS = {"max_tokens", "max_completion_tokens", "max_output_tokens"}

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model_name: str,
        reasoning_effort: str,
        timeout: float = OPENAI_TIMEOUT_S,
        client: Any | None = None,
    ) -> None:
        if endpoint.rstrip("/") != "https://api.openai.com/v1":
            raise ValueError("Responses adapter requires the public OpenAI v1 endpoint")
        if model_name != OPENAI_MODEL:
            raise ValueError(f"Responses adapter requires model {OPENAI_MODEL}")
        if reasoning_effort != OPENAI_REASONING_EFFORT:
            raise ValueError(
                "Responses adapter reasoning effort does not match the frozen protocol"
            )
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise ValueError("Responses adapter timeout must be numeric")
        if timeout <= 0:
            raise ValueError("Responses adapter timeout must be positive")

        if client is None:
            import openai
            from openai import OpenAI

            if openai.__version__ != OPENAI_SDK_VERSION:
                raise RuntimeError(
                    "OpenAI SDK mismatch: expected "
                    f"{OPENAI_SDK_VERSION}, got {openai.__version__}"
                )
            client = OpenAI(
                base_url=endpoint,
                api_key=api_key,
                timeout=float(timeout),
                max_retries=4,
            )

        self.client = client
        self.endpoint = endpoint
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.timeout = float(timeout)
        self.image_formats = set(self._IMAGE_MEDIA_TYPES)
        self.verbose = False

    @staticmethod
    def _require_output_tokens(value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("GPT maximum output tokens must be a positive integer")
        return value

    def _normalize_params(self, params: Mapping[str, Any] | None) -> int:
        if params is None:
            return OPENAI_DEFAULT_MAX_OUTPUT_TOKENS
        if not isinstance(params, Mapping):
            raise ValueError("GPT params must be a mapping")
        unknown = set(params) - self._IGNORED_CHAT_PARAMS - self._TOKEN_PARAMS
        if unknown:
            raise ValueError(
                "Unsupported GPT Responses params: " + ", ".join(sorted(unknown))
            )
        token_values = [
            self._require_output_tokens(params[name])
            for name in sorted(self._TOKEN_PARAMS)
            if name in params
        ]
        if not token_values:
            return OPENAI_DEFAULT_MAX_OUTPUT_TOKENS
        if len(set(token_values)) != 1:
            raise ValueError("Conflicting GPT maximum output token parameters")
        return token_values[0]

    @staticmethod
    def _base64_payload(value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Image input is neither a safe path nor valid base64") from exc
        return value

    def _image_data_url(self, image: Any) -> str:
        if isinstance(image, (str, os.PathLike)):
            value = os.fspath(image)
            if value.startswith("data:image/"):
                return value
            suffix = Path(value).suffix.lower()
            if suffix in self._IMAGE_MEDIA_TYPES:
                path = Path(value)
                if not path.is_file() or path.is_symlink():
                    raise FileNotFoundError(f"Image file not found or unsafe: {path}")
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                media_type = self._IMAGE_MEDIA_TYPES[suffix]
                return f"data:{media_type};base64,{encoded}"
            encoded = self._base64_payload(value)
            return f"data:image/png;base64,{encoded}"

        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("Image input must be a path, base64 string, or PIL image")
        image_format = (image.format or "PNG").upper()
        if image_format not in {"GIF", "JPEG", "JPG", "PNG", "WEBP"}:
            raise ValueError(f"Unsupported OpenAI image format: {image_format}")
        media_type = "image/jpeg" if image_format in {"JPG", "JPEG"} else (
            f"image/{image_format.lower()}"
        )
        buffer = BytesIO()
        image.save(buffer, format=image_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    @staticmethod
    def _output_text(response: Any) -> str | None:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
        return None

    def _create_response(
        self,
        *,
        text_prompt: str,
        image_base64: Any = None,
        system_role: str | None = None,
        max_output_tokens: int,
    ) -> Any:
        if not isinstance(text_prompt, str) or not text_prompt:
            raise ValueError("GPT text prompt must be a non-empty string")
        if system_role is None:
            system_role = self._DEFAULT_SYSTEM_ROLE
        if not isinstance(system_role, str) or not system_role:
            raise ValueError("GPT system role must be a non-empty string")

        user_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": text_prompt}
        ]
        if image_base64 is not None:
            images = image_base64 if isinstance(image_base64, list) else [image_base64]
            if not images:
                raise ValueError("GPT image list must not be empty")
            for image in images:
                user_content.append(
                    {
                        "type": "input_image",
                        "image_url": self._image_data_url(image),
                        "detail": "auto",
                    }
                )

        return self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_role}],
                },
                {"role": "user", "content": user_content},
            ],
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=max_output_tokens,
            store=False,
        )

    def query(
        self,
        text_prompt: str,
        image_base64: Any = None,
        system_role: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Run one Responses request while preserving EmbodiedGen's call shape."""

        try:
            response = self._create_response(
                text_prompt=text_prompt,
                image_base64=image_base64,
                system_role=system_role,
                max_output_tokens=self._normalize_params(params),
            )
            return self._output_text(response)
        except Exception:
            return None

    def check_connection(self) -> None:
        """Fail closed unless the configured model returns visible text."""

        try:
            response = self._create_response(
                text_prompt="Return exactly the word OK.",
                system_role="You are a test system.",
                max_output_tokens=64,
            )
            if self._output_text(response) is None:
                raise RuntimeError("Responses probe returned no output text")
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to GPT Responses API at {self.endpoint}"
            ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned EmbodiedGen GPT layout, SD3.5 image, TRELLIS asset, "
            "background retrieval, and BFS placement stages for one fresh "
            "prompt. This never calls sim_cli and never loads a robot actor."
        )
    )
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gpt-config", type=Path, required=True)
    parser.add_argument("--background-catalog", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--resume-layout", type=Path, required=True)
    parser.add_argument("--resume-scene-tree", type=Path, required=True)
    parser.add_argument("--resume-images", type=Path, required=True)
    parser.add_argument("--resume-receipt", type=Path, required=True)
    parser.add_argument("--resume-assets", type=Path, required=True)
    parser.add_argument("--resume-asset-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _command_output(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def _configure_gpt(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file() or path.is_symlink():
        raise ValueError("--gpt-config must be a regular non-symlink file")
    if path.stat().st_mode & 0o077:
        raise ValueError("--gpt-config must not be group/world accessible")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {
        "agent_type",
        "openai-platform",
    }:
        raise ValueError("GPT config has unexpected top-level fields")
    if config["agent_type"] != "openai-platform":
        raise ValueError("GPT config agent_type must be openai-platform")
    provider = config["openai-platform"]
    if not isinstance(provider, dict) or set(provider) != {
        "endpoint",
        "api_key",
        "api_version",
        "model_name",
    }:
        raise ValueError("GPT provider config has unexpected fields")
    for field in ("endpoint", "api_key", "model_name"):
        if not isinstance(provider[field], str) or not provider[field]:
            raise ValueError(f"GPT provider {field} must be non-empty")
    if provider["api_version"] is not None and not isinstance(
        provider["api_version"], str
    ):
        raise ValueError("GPT provider api_version must be string or null")
    endpoint = provider["endpoint"].rstrip("/") + "/"
    os.environ["ENDPOINT"] = endpoint
    os.environ["API_KEY"] = provider["api_key"]
    os.environ["MODEL_NAME"] = provider["model_name"]
    if provider["api_version"] is None:
        os.environ.pop("API_VERSION", None)
    else:
        os.environ["API_VERSION"] = provider["api_version"]
    return {
        "agent_type": config["agent_type"],
        "api_mode": OPENAI_API_MODE,
        "endpoint": endpoint,
        "api_version": provider["api_version"],
        "model_name": provider["model_name"],
        "reasoning_effort": OPENAI_REASONING_EFFORT,
        "sdk_version": OPENAI_SDK_VERSION,
        "api_key_present": True,
    }


def _install_openai_responses_client(
    module: Any, *, client: Any | None = None
) -> _ResponsesGPTClient:
    """Replace the upstream Chat/Azure singleton before consumers import it."""

    adapter = _ResponsesGPTClient(
        endpoint=os.environ["ENDPOINT"],
        api_key=os.environ["API_KEY"],
        model_name=os.environ["MODEL_NAME"],
        reasoning_effort=OPENAI_REASONING_EFFORT,
        client=client,
    )
    module.GPT_CLIENT = adapter
    return adapter


def _source_state(source_root: Path) -> dict[str, Any]:
    source = source_root.resolve(strict=True)
    actual = _command_output(["git", "-C", str(source), "rev-parse", "HEAD"])
    if actual != EMBODIEDGEN_COMMIT:
        raise ValueError(
            f"EmbodiedGen commit mismatch: expected {EMBODIEDGEN_COMMIT}, got {actual}"
        )
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise ValueError("EmbodiedGen tracked source checkout is dirty")
    trellis = source / "thirdparty" / "TRELLIS"
    flexicubes = trellis / "trellis" / "representations" / "mesh" / "flexicubes"
    for path, expected, label in (
        (trellis, TRELLIS_SOURCE_COMMIT, "TRELLIS"),
        (flexicubes, TRELLIS_FLEXICUBES_COMMIT, "TRELLIS FlexiCubes"),
    ):
        submodule_commit = _command_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"]
        )
        if submodule_commit != expected:
            raise ValueError(
                f"{label} commit mismatch: expected {expected}, got {submodule_commit}"
            )
        for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
            result = subprocess.run(
                ["git", "-C", str(path), *args],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                raise ValueError(f"{label} tracked source checkout is dirty")
    return {
        "path": str(source),
        "version": EMBODIEDGEN_VERSION,
        "commit": actual,
        "tracked_files_clean": True,
        "trellis": {
            "path": str(trellis),
            "commit": TRELLIS_SOURCE_COMMIT,
            "flexicubes_commit": TRELLIS_FLEXICUBES_COMMIT,
            "tracked_files_clean": True,
        },
    }


def _generation_config(protocol: Mapping[str, Any]) -> dict[str, Any]:
    generation = protocol.get("fresh_generation")
    if not isinstance(generation, dict):
        raise ValueError("protocol fresh_generation must be an object")
    required = {
        "activity_spec",
        "background_dataset",
        "gpt_api",
        "gpt_model",
        "gpt_reasoning_effort",
        "openai_sdk_version",
        "text_to_image_model",
        "text_to_image_backend",
        "image_to_3d_backend",
        "trellis",
        "sam3d_comparison",
        "resume_source",
        "asset_resume_source",
        "image_samples_per_prompt",
        "text_guidance_scale",
        "image_denoise_steps",
        "gpt_service_seed_supported",
        "invoke_upstream_sim_cli",
        "robot_actor_loaded",
        "slurm",
    }
    if set(generation) != required:
        raise ValueError("protocol fresh_generation fields do not match the schema")
    expected = {
        "gpt_api": OPENAI_API_MODE,
        "gpt_model": OPENAI_MODEL,
        "gpt_reasoning_effort": OPENAI_REASONING_EFFORT,
        "openai_sdk_version": OPENAI_SDK_VERSION,
        "text_to_image_model": "stabilityai/stable-diffusion-3.5-medium",
        "text_to_image_backend": "sd35",
        "image_to_3d_backend": "TRELLIS",
        "trellis": {
            "source_repository": "microsoft/TRELLIS",
            "source_commit": TRELLIS_SOURCE_COMMIT,
            "source_license": "MIT",
            "checkpoint_repository": "microsoft/TRELLIS-image-large",
            "checkpoint_revision": TRELLIS_CHECKPOINT_REVISION,
            "checkpoint_license": "MIT",
            "dinov2_repository": "facebookresearch/dinov2",
            "dinov2_commit": DINOv2_COMMIT,
            "dinov2_license": "Apache-2.0",
        },
        "sam3d_comparison": {
            "repository": "facebook/sam-3d-objects",
            "commit": "2e73555018d2741ccd486e56c24fac41155a1dc6",
            "access_status": "pending_not_admitted",
            "blocks_trellis_run": False,
        },
        "asset_resume_source": {
            "job_id": "328381",
            "generation_receipt_sha256": (
                "0842642b0e10a5ddee297d2188accf3ac3ac95042764551f9c3ce3dd90408d23"
            ),
            "requalification_policy": "four_separate_views_of_one_asset",
            "nodes": {
                "table": (
                    "standalone rectangular wooden table with a clean empty "
                    "tabletop and warm polished surface"
                ),
                "red mug": (
                    "glossy crimson ceramic mug with curved handle, thick rim, "
                    "and smooth reflective surface"
                ),
                "plate": (
                    "white ceramic dinner plate with subtle rim, clean glossy "
                    "finish, and circular form"
                ),
                "spoon": (
                    "small stainless-steel spoon with polished reflective bowl "
                    "and slender rounded handle"
                ),
            },
        },
        "image_samples_per_prompt": 1,
        "text_guidance_scale": 7.0,
        "image_denoise_steps": 25,
        "gpt_service_seed_supported": False,
        "invoke_upstream_sim_cli": False,
        "robot_actor_loaded": False,
    }
    for key, value in expected.items():
        if generation.get(key) != value:
            raise ValueError(f"protocol fresh_generation.{key} mismatch")
    return copy.deepcopy(generation)


def _resume_inputs(args: argparse.Namespace, generation: Mapping[str, Any]) -> dict[str, Any]:
    frozen = generation["resume_source"]
    layout_path = args.resume_layout.resolve(strict=True)
    scene_tree_path = args.resume_scene_tree.resolve(strict=True)
    images_root = args.resume_images.resolve(strict=True)
    receipt_path = args.resume_receipt.resolve(strict=True)
    if not images_root.is_dir() or images_root.is_symlink():
        raise ValueError("--resume-images must be a non-symlink directory")
    expected_hashes = {
        layout_path: frozen["layout_draft_sha256"],
        scene_tree_path: frozen["scene_tree_sha256"],
        receipt_path: frozen["generation_receipt_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise ValueError(f"resume artifact hash mismatch: {path.name}")
    prior_receipt = _read_json(receipt_path)
    if prior_receipt.get("status") != "failed":
        raise ValueError("resume receipt must be the admitted failed partial run")
    if prior_receipt.get("activity", {}).get("activity_id") != "red-mug-mouth-return":
        raise ValueError("resume receipt activity mismatch")
    if prior_receipt.get("seeds") != {
        "asset": 2026081502,
        "image": 2026081501,
        "layout": 2026081503,
    }:
        raise ValueError("resume receipt seeds mismatch")
    image_records: dict[str, dict[str, Any]] = {}
    for node, record in frozen["conditioning_images"].items():
        image = images_root / f"{node}.png"
        raw_image = images_root / f"{node}_raw.png"
        for path, key in (
            (image, "image_sha256"),
            (raw_image, "raw_image_sha256"),
        ):
            if not path.is_file() or path.is_symlink() or _sha256(path) != record[key]:
                raise ValueError(f"resume conditioning image hash mismatch: {path.name}")
        image_records[node] = {
            **copy.deepcopy(record),
            "image_path": image,
            "raw_image_path": raw_image,
        }
    return {
        "layout_path": layout_path,
        "scene_tree_path": scene_tree_path,
        "receipt_path": receipt_path,
        "prior_receipt": prior_receipt,
        "images": image_records,
    }


def _install_resume_conditioning_images(
    module: Any,
    records: Mapping[str, Mapping[str, Any]],
    *,
    initial_image_seed: int,
) -> set[str]:
    original = module.text_to_image
    reused: set[str] = set()
    attempts: dict[str, int] = {}

    def text_to_image(
        prompt: str,
        save_path: str,
        n_retry: int,
        img_denoise_step: int,
        text_guidance_scale: float,
        n_img_sample: int,
        image_hw: tuple[int, int] = (1024, 1024),
        seed: int | None = None,
    ) -> bool:
        node = Path(save_path).stem
        record = records.get(node)
        if record is None:
            return original(
                prompt,
                save_path,
                n_retry,
                img_denoise_step,
                text_guidance_scale,
                n_img_sample,
                image_hw=image_hw,
                seed=seed,
            )
        if prompt != record["prompt"]:
            raise ValueError(f"resume prompt mismatch for {node}")
        attempts[node] = attempts.get(node, 0) + 1
        if attempts[node] == 1 and seed != initial_image_seed:
            raise ValueError(f"resume initial image seed mismatch for {node}")
        destination = Path(save_path)
        raw_destination = destination.with_name(destination.stem + "_raw.png")
        if destination.exists() or raw_destination.exists():
            if not destination.is_file() or not raw_destination.is_file():
                raise ValueError(f"resume destination is incomplete for {node}")
            if _sha256(destination) != record["image_sha256"]:
                raise ValueError(f"resume destination image changed for {node}")
            if _sha256(raw_destination) != record["raw_image_sha256"]:
                raise ValueError(f"resume destination raw image changed for {node}")
        else:
            shutil.copy2(record["image_path"], destination)
            shutil.copy2(record["raw_image_path"], raw_destination)
        reused.add(node)
        return True

    module.text_to_image = text_to_image
    return reused


def _resume_assets(args: argparse.Namespace, generation: Mapping[str, Any]) -> dict[str, Any]:
    frozen = generation["asset_resume_source"]
    root = args.resume_assets.resolve(strict=True)
    receipt_path = args.resume_asset_receipt.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("--resume-assets must be a non-symlink directory")
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError("--resume-asset-receipt must be a regular non-symlink file")
    if _sha256(receipt_path) != frozen["generation_receipt_sha256"]:
        raise ValueError("resume asset receipt hash mismatch")
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "failed":
        raise ValueError("resume asset receipt must be a failed partial run")
    if receipt.get("models", {}).get("image_to_3d_backend") != "TRELLIS":
        raise ValueError("resume asset receipt was not generated by TRELLIS")
    if receipt.get("models", {}).get("trellis") != generation["trellis"]:
        raise ValueError("resume asset receipt TRELLIS pins mismatch")
    if receipt.get("error", {}).get("message") != (
        "resume initial image seed mismatch for table"
    ):
        raise ValueError("resume asset receipt failure boundary mismatch")

    manifest: dict[str, dict[str, Any]] = {}
    for record in receipt.get("files", []):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("resume asset receipt has malformed file records")
        relative = record["path"]
        if not isinstance(relative, str):
            raise ValueError("resume asset receipt has a non-string path")
        safe = PurePosixPath(relative)
        if safe.is_absolute() or not safe.parts or any(
            part in {"", ".", ".."} for part in safe.parts
        ):
            raise ValueError("resume asset receipt has an unsafe path")
        if relative in manifest:
            raise ValueError("resume asset receipt has a duplicate path")
        path = root.joinpath(*safe.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"resume asset file is missing or unsafe: {relative}")
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"resume asset file hash mismatch: {relative}")
        manifest[relative] = copy.deepcopy(record)

    nodes: dict[str, dict[str, Any]] = {}
    for node, prompt in frozen["nodes"].items():
        save_node = node.replace(" ", "_")
        result_relative = f"asset3d/{save_node}/result"
        render_paths = [
            f"{result_relative}/renders/image_color/{index:04d}.png"
            for index in range(4)
        ]
        required = [
            f"images/{save_node}.png",
            *render_paths,
        ]
        for relative in required:
            if relative not in manifest:
                raise ValueError(f"resume asset is missing required file: {relative}")
        result_records = [
            record
            for relative, record in sorted(manifest.items())
            if relative.startswith(result_relative + "/")
        ]
        if not result_records:
            raise ValueError(f"resume asset has no result package: {node}")
        raw_relative = f"images/{save_node}_raw.png"
        nodes[node] = {
            "prompt": prompt,
            "result_relative": result_relative,
            "result_path": root / result_relative,
            "image_path": root / f"images/{save_node}.png",
            "raw_image_path": (
                root / raw_relative if raw_relative in manifest else None
            ),
            "render_paths": [root / relative for relative in render_paths],
            "result_manifest_sha256": _canonical_sha256(result_records),
        }
    return {
        "root": root,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "nodes": nodes,
    }


def _manifest(root: Path, excluded: set[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def _load_background_catalog(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, description = line.split(":", 1)
        key = key.strip()
        description = description.strip()
        if not key or not description or key in values:
            raise ValueError("background catalog has an invalid or duplicate row")
        values[key] = description
    if not values:
        raise ValueError("background catalog is empty")
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    started_utc = _utc_now()
    output = args.output_dir.resolve(strict=False)
    receipt_path = args.receipt.resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise ValueError("--output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    if receipt_path.parent != output:
        raise ValueError("--receipt must be a direct child of --output-dir")

    activity = _read_json(args.activity.resolve(strict=True))
    request = _read_json(args.request.resolve(strict=True))
    protocol = _read_json(args.protocol.resolve(strict=True))
    generation = _generation_config(protocol)
    resume = _resume_inputs(args, generation)
    asset_resume = _resume_assets(args, generation)
    public_gpt = _configure_gpt(args.gpt_config.resolve(strict=True))
    if public_gpt["model_name"] != generation["gpt_model"]:
        raise ValueError("GPT config model does not match the frozen protocol")
    source = _source_state(args.source_root)

    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    if str(source["path"]) not in sys.path:
        sys.path.insert(0, str(source["path"]))
    from babyworld_lite.interactmove_intermimic.activity import (
        build_embodiedgen_request,
        validate_activity_spec,
    )

    validate_activity_spec(activity)
    expected_request = build_embodiedgen_request(
        activity,
        upstream_version=source["version"],
        upstream_commit=source["commit"],
        background_list=request.get("background_list", []),
    )
    if request != expected_request:
        raise ValueError("request is not the canonical ActivitySpec projection")
    catalog_path = args.background_catalog.resolve(strict=True)
    catalog = _load_background_catalog(catalog_path)
    if request["background_list"] != list(catalog):
        raise ValueError("request background_list must equal the frozen catalog order")
    dataset = generation["background_dataset"]
    if dataset.get("scene_id") not in catalog:
        raise ValueError("frozen background scene is absent from the catalog")

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "running",
        "started_utc": started_utc,
        "finished_utc": None,
        "duration_s": None,
        "activity": {
            "activity_id": activity["activity_id"],
            "prompt": activity["prompt"],
            "activity_spec_sha256": _canonical_sha256(activity),
            "embodiedgen_request_sha256": _canonical_sha256(request),
        },
        "seeds": copy.deepcopy(request["seeds"]),
        "source": source,
        "models": {
            "gpt": public_gpt,
            "text_to_image": generation["text_to_image_model"],
            "text_to_image_backend": generation["text_to_image_backend"],
            "image_to_3d_backend": generation["image_to_3d_backend"],
            "trellis": copy.deepcopy(generation["trellis"]),
            "sam3d_comparison": copy.deepcopy(generation["sam3d_comparison"]),
        },
        "generation_config": {
            "image_samples_per_prompt": generation["image_samples_per_prompt"],
            "text_guidance_scale": generation["text_guidance_scale"],
            "image_denoise_steps": generation["image_denoise_steps"],
            "retry_limits": copy.deepcopy(request["retry_limits"]),
            "gpt_service_seed_supported": generation[
                "gpt_service_seed_supported"
            ],
            "keep_intermediate": False,
            "resume_source": {
                "job_id": generation["resume_source"]["job_id"],
                "generation_receipt_sha256": generation["resume_source"][
                    "generation_receipt_sha256"
                ],
                "scene_tree_sha256": generation["resume_source"][
                    "scene_tree_sha256"
                ],
                "layout_draft_sha256": generation["resume_source"][
                    "layout_draft_sha256"
                ],
                "layout_recovery": generation["resume_source"]["layout_recovery"],
            },
            "asset_resume_source": {
                "job_id": generation["asset_resume_source"]["job_id"],
                "generation_receipt_sha256": generation["asset_resume_source"][
                    "generation_receipt_sha256"
                ],
                "requalification_policy": generation["asset_resume_source"][
                    "requalification_policy"
                ],
            },
        },
        "background": {
            "repository": dataset["repository"],
            "commit": dataset["commit"],
            "catalog_path": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "selected_scene_id": None,
            "retrieved_pre_generated_background": True,
            "fresh_background_generation": False,
        },
        "robot_policy": {
            "render_insert_robot": request["render_insert_robot"],
            "upstream_sim_cli_invoked": False,
            "robot_actor_loaded": False,
            "native_layout_robot_metadata_may_be_present": True,
        },
        "boundaries": {
            "fresh_embodiedgen_scene_generation": "running",
            "scene_bundle_validation": "not_run",
            "scene_only_sapien_physics_render": "not_run",
            "interactmove_motion_generation": "not_run",
            "intermimic_execution": "not_run",
        },
        "stages": [],
        "layout": None,
        "files": [],
        "error": None,
    }
    _write_json(receipt_path, receipt)

    secret = os.environ["API_KEY"]
    try:
        import embodied_gen.utils.gpt_clients as gpt_clients

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        GPT_CLIENT = _install_openai_responses_client(gpt_clients)
        GPT_CLIENT.check_connection()
        receipt["stages"].append(
            {"name": "gpt_connection", "status": "passed", "finished_utc": _utc_now()}
        )
        _write_json(receipt_path, receipt)

        import embodied_gen.scripts.textto3d as textto3d_module
        from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
        from embodied_gen.utils.geometry import bfs_placement
        from embodied_gen.utils.process_media import SceneTreeVisualizer
        from embodied_gen.validators.quality_checkers import SemanticMatcher

        textto3d_module.TXTGEN_CHECKER.prompt = (
            MULTIVIEW_QUALITY_PREAMBLE + textto3d_module.TXTGEN_CHECKER.prompt
        )

        scene_graph_path = output / "scene_tree.jpg"
        gpt_params = {
            "temperature": 1.0,
            "top_p": 0.95,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.5,
            "max_tokens": 500,
        }
        layout_info = LayoutInfo.from_dict(_read_json(resume["layout_path"]))
        verification_path = output / "scene_tree.verify.jpg"
        SceneTreeVisualizer(layout_info).render(save_path=str(verification_path))
        if _sha256(verification_path) != generation["resume_source"][
            "scene_tree_sha256"
        ]:
            raise RuntimeError("recovered GPT layout does not reproduce scene-tree hash")
        verification_path.unlink()
        shutil.copy2(resume["scene_tree_path"], scene_graph_path)
        provenance_root = output / "provenance" / (
            "resume_job_" + generation["resume_source"]["job_id"]
        )
        provenance_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(
            resume["layout_path"], provenance_root / "layout_draft.json"
        )
        shutil.copy2(
            resume["receipt_path"], provenance_root / "generation_receipt.json"
        )
        asset_provenance_root = output / "provenance" / (
            "resume_job_" + generation["asset_resume_source"]["job_id"]
        )
        asset_provenance_root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(
            asset_resume["receipt_path"],
            asset_provenance_root / "generation_receipt.json",
        )
        manipulated = list(
            layout_info.relation.get(Scene3DItemEnum.MANIPULATED_OBJS.value, [])
        )
        if len(manipulated) != 1:
            raise RuntimeError(
                "fresh layout must contain exactly one manipulated object; "
                f"got {len(manipulated)}"
            )
        receipt["stages"].append(
            {
                "name": "gpt_scene_layout",
                "status": "reused",
                "finished_utc": _utc_now(),
                "manipulated_source_keys": manipulated,
                "source_job_id": generation["resume_source"]["job_id"],
                "layout_draft_sha256": generation["resume_source"][
                    "layout_draft_sha256"
                ],
                "scene_tree_sha256": generation["resume_source"][
                    "scene_tree_sha256"
                ],
                "recovery_validation": generation["resume_source"][
                    "layout_recovery"
                ],
            }
        )
        _write_json(receipt_path, receipt)

        reused_images = _install_resume_conditioning_images(
            textto3d_module,
            resume["images"],
            initial_image_seed=request["seeds"]["image"],
        )

        prompts_mapping = {value: key for key, value in layout_info.objs_desc.items()}
        prompts = [
            value
            for key, value in layout_info.objs_desc.items()
            if layout_info.objs_mapping[key] != Scene3DItemEnum.BACKGROUND.value
        ]
        for index, prompt in enumerate(prompts):
            node = prompts_mapping[prompt]
            resumed_asset = asset_resume["nodes"].get(node)
            if resumed_asset is not None:
                if prompt != resumed_asset["prompt"]:
                    raise RuntimeError(f"resume asset prompt mismatch for {node}")
                qa_response = textto3d_module.TXTGEN_CHECKER.query(
                    node, [str(path) for path in resumed_asset["render_paths"]]
                )
                qa_result = qa_response.strip() if isinstance(qa_response, str) else None
                if qa_result != "YES":
                    raise RuntimeError(
                        f"resumed TRELLIS asset failed multiview requalification: "
                        f"{node}: {qa_result}"
                    )
                save_node = node.replace(" ", "_")
                destination = output / "asset3d" / save_node / "result"
                destination.parent.mkdir(parents=True, exist_ok=False)
                shutil.copytree(resumed_asset["result_path"], destination)
                images_destination = output / "images"
                images_destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    resumed_asset["image_path"],
                    images_destination / f"{save_node}.png",
                )
                if resumed_asset["raw_image_path"] is not None:
                    shutil.copy2(
                        resumed_asset["raw_image_path"],
                        images_destination / f"{save_node}_raw.png",
                    )
                generation_log = {
                    "assets": {node: resumed_asset["result_relative"]},
                    "quality": {node: qa_result},
                }
                stage_name = "trellis_asset_requalified"
                conditioning_image = {
                    "status": "reused",
                    "source_job_id": generation["asset_resume_source"]["job_id"],
                    "sha256": _sha256(resumed_asset["image_path"]),
                }
            else:
                generation_log = textto3d_module.text_to_3d(
                    prompts=[prompt],
                    output_root=str(output),
                    asset_names=[node],
                    n_img_sample=generation["image_samples_per_prompt"],
                    text_guidance_scale=generation["text_guidance_scale"],
                    img_denoise_step=generation["image_denoise_steps"],
                    n_image_retry=request["retry_limits"]["image"],
                    n_asset_retry=request["retry_limits"]["asset"],
                    n_pipe_retry=request["retry_limits"]["pipeline"],
                    seed_img=request["seeds"]["image"],
                    seed_3d=request["seeds"]["asset"],
                    keep_intermediate=False,
                    image3d_model=generation["image_to_3d_backend"],
                )
                qa_result = generation_log["quality"].get(node)
                if not isinstance(qa_result, str) or qa_result != "YES":
                    raise RuntimeError(
                        f"fresh TRELLIS asset failed final quality gate: "
                        f"{node}: {qa_result}"
                    )
                stage_name = "sd35_trellis_asset"
                conditioning_image = (
                    {
                        "status": "reused",
                        "source_job_id": generation["resume_source"]["job_id"],
                        "accepted_retry_seed": resume["images"][node][
                            "accepted_retry_seed"
                        ],
                        "sha256": resume["images"][node]["image_sha256"],
                    }
                    if node in reused_images
                    else {
                        "status": "generated",
                        "initial_seed": request["seeds"]["image"],
                    }
                )
            layout_info.assets.update(generation_log["assets"])
            layout_info.quality.update(generation_log["quality"])
            receipt["stages"].append(
                {
                    "name": stage_name,
                    "status": "passed",
                    "asset_index": index,
                    "source_node_key": node,
                    "prompt": prompt,
                    "quality": generation_log["quality"].get(node),
                    "conditioning_image": conditioning_image,
                    "result_manifest_sha256": (
                        resumed_asset["result_manifest_sha256"]
                        if resumed_asset is not None
                        else None
                    ),
                    "finished_utc": _utc_now(),
                }
            )
            _write_json(receipt_path, receipt)

        matcher = SemanticMatcher(GPT_CLIENT)
        match_key = matcher.query(
            layout_info.objs_desc[
                layout_info.relation[Scene3DItemEnum.BACKGROUND.value]
            ],
            str(catalog),
            params=gpt_params,
        )
        attempts = 1
        while match_key not in catalog and attempts < 10:
            match_key = matcher.query(
                layout_info.objs_desc[
                    layout_info.relation[Scene3DItemEnum.BACKGROUND.value]
                ],
                str(catalog),
                params=gpt_params,
            )
            attempts += 1
        if match_key not in catalog:
            raise RuntimeError("background semantic matcher returned no catalog key")
        background_source = catalog_path.parent / match_key
        if not background_source.is_dir() or background_source.is_symlink():
            raise RuntimeError("selected background is not a regular directory")
        background_save = output / "background"
        shutil.copytree(background_source, background_save, dirs_exist_ok=False)
        background_node = layout_info.relation[Scene3DItemEnum.BACKGROUND.value]
        layout_info.assets[background_node] = "background"
        receipt["background"]["selected_scene_id"] = match_key
        receipt["stages"].append(
            {
                "name": "background_retrieval",
                "status": "passed",
                "matcher_attempts": attempts,
                "selected_scene_id": match_key,
                "finished_utc": _utc_now(),
            }
        )
        _write_json(receipt_path, receipt)

        layout_path = output / "layout.json"
        _write_json(layout_path, layout_info.to_dict())
        placed = bfs_placement(
            str(layout_path), seed=request["seeds"]["layout"]
        )
        _write_json(layout_path, placed.to_dict())
        parsed = LayoutInfo.from_dict(_read_json(layout_path))
        native_robot = parsed.relation.get(Scene3DItemEnum.ROBOT.value)
        receipt["stages"].append(
            {
                "name": "bfs_placement",
                "status": "passed",
                "layout_seed": request["seeds"]["layout"],
                "finished_utc": _utc_now(),
            }
        )
        receipt["layout"] = {
            "path": "layout.json",
            "sha256": _sha256(layout_path),
            "native_robot_metadata_declared": native_robot is not None,
            "native_robot_source_key": native_robot,
            "native_robot_loaded": False,
            "non_background_asset_count": len(prompts),
        }
        torch.cuda.empty_cache()
        gc.collect()

        receipt["files"] = _manifest(output, {receipt_path})
        receipt["status"] = "passed"
        receipt["boundaries"]["fresh_embodiedgen_scene_generation"] = "passed"
        receipt["finished_utc"] = _utc_now()
        receipt["duration_s"] = round(time.monotonic() - started, 6)
        _write_json(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        message = str(exc).replace(secret, "<redacted>") if secret else str(exc)
        receipt["status"] = "failed"
        receipt["boundaries"]["fresh_embodiedgen_scene_generation"] = "failed"
        receipt["finished_utc"] = _utc_now()
        receipt["duration_s"] = round(time.monotonic() - started, 6)
        receipt["error"] = {"type": type(exc).__name__, "message": message}
        try:
            receipt["files"] = _manifest(output, {receipt_path})
            _write_json(receipt_path, receipt)
        finally:
            os.environ.pop("API_KEY", None)
        raise
    finally:
        os.environ.pop("API_KEY", None)


def main() -> int:
    args = _parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "activity_id": result["activity"]["activity_id"],
                "layout_sha256": result["layout"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
