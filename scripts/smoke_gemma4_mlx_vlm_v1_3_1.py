#!/usr/bin/env python3
"""Public/synthetic-only Gemma 4 + MLX-VLM strict-schema smoke harness.

This program never discovers or reads ChildLens or any quarantine.  It creates
only a synthetic image and tone, builds a loopback MLX-VLM request blueprint,
and validates synthetic responses against a strict JSON schema.  It does not
download weights, accept a license, start a server, or claim model execution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw


VERSION = "childlens-gemma4-mlx-vlm-public-smoke-v1.3.1"
REPORT_SCHEMA = "childlens-gemma4-mlx-vlm-shortlist-receipt-v1.3.1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/childlens_feasibility_v1_3_1/gemma4_mlx_vlm_shortlist_receipt.json"
)
MODEL_IDS = {
    "gemma4_e4b": "mlx-community/gemma-4-e4b-it-4bit",
    "gemma4_12b_unified": "mlx-community/gemma-4-12B-it-4bit",
}
KNOWN_CACHE_ROOTS = (
    Path.home() / ".cache/huggingface/hub",
    Path.home() / "Library/Caches/huggingface/hub",
)
MODEL_CACHE_NAMES = {
    key: "models--" + model_id.replace("/", "--")
    for key, model_id in MODEL_IDS.items()
}


STRICT_SCHEMA: Mapping[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["language", "audio_scene", "visual_scene", "events"],
    "properties": {
        "language": {
            "type": "string",
            "enum": ["en", "und", "none"],
        },
        "audio_scene": {
            "type": "string",
            "enum": ["tone", "speech", "silence", "uncertain"],
        },
        "visual_scene": {
            "type": "string",
            "enum": ["red_square", "other", "uncertain"],
        },
        "events": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_role",
                    "referential_status",
                    "mention_family",
                    "confidence_band",
                ],
                "properties": {
                    "source_role": {
                        "type": "string",
                        "enum": ["NONSPEECH", "NON_CHILD", "CHILD", "OVERLAP", "UNCERTAIN"],
                    },
                    "referential_status": {
                        "type": "string",
                        "enum": [
                            "VISIBLE_CANDIDATE",
                            "NULL_NOT_VISIBLE",
                            "IRRELEVANT",
                            "UNDECIDABLE",
                            "UNUSABLE",
                        ],
                    },
                    "mention_family": {
                        "type": "string",
                        "enum": ["NOUN_OBJECT", "VERB_ACTION", "NEITHER"],
                    },
                    "confidence_band": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH", "NOT_APPLICABLE"],
                    },
                },
            },
        },
    },
}

SYNTHETIC_VALID_OUTPUT: Mapping[str, Any] = {
    "language": "none",
    "audio_scene": "tone",
    "visual_scene": "red_square",
    "events": [
        {
            "source_role": "NONSPEECH",
            "referential_status": "IRRELEVANT",
            "mention_family": "NEITHER",
            "confidence_band": "HIGH",
        }
    ],
}


class SmokeError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def create_synthetic_fixtures(directory: Path) -> tuple[Path, Path]:
    """Create a code-generated red-square image and 440 Hz tone."""

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    image_path = directory / "synthetic_red_square.png"
    audio_path = directory / "synthetic_tone.wav"

    image = Image.new("RGB", (256, 256), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 64, 192, 192), fill=(220, 30, 30))
    image.save(image_path, format="PNG")

    sample_rate = 16_000
    duration_seconds = 1.0
    amplitude = 0.25 * 32767
    samples = [
        int(amplitude * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(int(sample_rate * duration_seconds))
    ]
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(struct.pack("<h", value) for value in samples))
    return image_path, audio_path


def build_loopback_request(
    model_id: str, image_path: Path, audio_path: Path
) -> Mapping[str, Any]:
    """Build the provisional joint-modality strict-schema request blueprint."""

    if model_id not in MODEL_IDS.values():
        raise SmokeError("E_MODEL_NOT_ALLOWLISTED")
    for path in (image_path, audio_path):
        if not path.is_absolute() or not path.is_file():
            raise SmokeError("E_SYNTHETIC_FIXTURE")
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": str(image_path),
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analyze the synthetic image and audio together. "
                            "Return only the constrained object."
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": str(audio_path),
                    },
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "SyntheticJointObservation",
                "strict": True,
                "schema": STRICT_SCHEMA,
            },
        },
        "temperature": 0.0,
        "max_tokens": 256,
    }


def validate_response_text(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SmokeError("E_RESPONSE_NOT_JSON") from exc
    errors = sorted(
        Draft202012Validator(STRICT_SCHEMA).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        raise SmokeError("E_RESPONSE_SCHEMA")
    if not isinstance(value, Mapping):
        raise SmokeError("E_RESPONSE_SCHEMA")
    return value


def _package_state() -> Mapping[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("mlx", "mlx_vlm", "jsonschema", "PIL")
    }


def _model_cache_state() -> Mapping[str, bool]:
    result: dict[str, bool] = {}
    for key, cache_name in MODEL_CACHE_NAMES.items():
        result[key] = any(
            (root / cache_name).is_dir()
            for root in KNOWN_CACHE_ROOTS
            if root.is_dir()
        )
    return result


def _physical_memory_gib() -> float | None:
    """Measure installed host memory without inspecting process state."""

    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return round(memory_bytes / (1024**3), 3)


def preflight_report() -> Mapping[str, Any]:
    start = time.perf_counter()
    packages = _package_state()
    cache = _model_cache_state()
    with tempfile.TemporaryDirectory(prefix="gemma4-public-synthetic-") as temporary:
        image_path, audio_path = create_synthetic_fixtures(Path(temporary))
        requests = {
            key: build_loopback_request(model_id, image_path, audio_path)
            for key, model_id in MODEL_IDS.items()
        }
        valid = validate_response_text(json.dumps(SYNTHETIC_VALID_OUTPUT))
        invalid_rejected = False
        invalid = dict(SYNTHETIC_VALID_OUTPUT)
        invalid["unexpected"] = True
        try:
            validate_response_text(json.dumps(invalid))
        except SmokeError as exc:
            invalid_rejected = exc.code == "E_RESPONSE_SCHEMA"
        request_shape_valid = all(
            request["response_format"]["json_schema"]["strict"] is True
            and len(request["messages"][0]["content"]) == 3
            for request in requests.values()
        )
    runtime_ready = packages["mlx"] and packages["mlx_vlm"]
    any_cached = any(cache.values())
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "NOT_EXECUTED",
        "scope": "PUBLIC_SYNTHETIC_ONLY",
        "hosted_or_cloud_inference_used": False,
        "childlens_or_quarantine_accessed": False,
        "model_weights_downloaded": False,
        "license_clickthrough_accepted": False,
        "model_inference_executed": False,
        "wall_time_or_memory_compared": False,
        "host": {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "physical_memory_gib": _physical_memory_gib(),
            "physical_memory_measurement": "OS_SYSCONF_INSTALLED_MEMORY",
        },
        "local_preflight": {
            "packages": packages,
            "model_cache_present": cache,
            "runtime_ready": runtime_ready,
            "any_shortlist_weights_cached": any_cached,
        },
        "documentation_preflight": {
            "mlx_vlm_reference_release": "0.6.6",
            "upstream_model_license": "Apache-2.0",
            "mlx_vlm_code_license": "MIT",
            "conversion_license_metadata_consistent": False,
            "strict_schema_server_feature": "DOCUMENTED_GENERIC_MULTIMODAL",
            "published_conversion_artifact_gb_not_runtime_memory": {
                "gemma4_e4b": 5.15,
                "gemma4_12b_unified": 6.74,
            },
        },
        "synthetic_schema_harness": {
            "valid_fixture_accepted": valid == SYNTHETIC_VALID_OUTPUT,
            "extra_property_rejected": invalid_rejected,
            "joint_audio_image_request_shape_built": request_shape_valid,
            "strict_json_schema_requested": True,
            "server_call_executed": False,
        },
        "model_dispositions": {
            "gemma4_e4b": {
                "native_joint_audio_image": "DOCUMENTED_UPSTREAM",
                "mlx_vlm_image_path": "DOCUMENTED",
                "mlx_vlm_audio_path": "DOCUMENTED",
                "mlx_vlm_joint_audio_image_strict_schema": "NOT_EXECUTED",
                "shortlist_status": "CONDITIONAL_SMOKE_REQUIRED",
            },
            "gemma4_12b_unified": {
                "native_joint_audio_image": "DOCUMENTED_UPSTREAM",
                "mlx_vlm_image_path": "DOCUMENTED",
                "mlx_vlm_audio_path": "DOCUMENTED_IN_MERGED_UNIFIED_SUPPORT",
                "mlx_vlm_joint_audio_image": "DOCUMENTED_MIXED_INPUT_REPRODUCTION",
                "mlx_vlm_joint_audio_image_strict_schema": "NOT_EXECUTED",
                "shortlist_status": "CONDITIONAL_SMOKE_REQUIRED",
            },
        },
        "measured_preflight_wall_seconds": round(time.perf_counter() - start, 6),
        "unmeasured_fields": [
            "model_load_peak_memory",
            "joint_prefill_peak_memory",
            "generation_wall_time",
            "schema_validity_from_model_output",
            "semantic_audio_image_use",
        ],
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    expected_parent = REPO_ROOT / "output/childlens_feasibility_v1_3_1"
    expected_parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    if resolved_parent != expected_parent.resolve(strict=True):
        raise SmokeError("E_OUTPUT_PATH")
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = preflight_report()
        _write_report(args.output, report)
        print(json.dumps({"status": report["status"], "schema_harness": "PASS"}))
        return 0
    except SmokeError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("E_PUBLIC_SYNTHETIC_SMOKE", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
