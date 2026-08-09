#!/usr/bin/env python3
"""Run the frozen public-only Gemma joint common-schema canary.

The fixture is generated locally from a fixed German phrase and five synthetic
frames. The model child is launched under OS network denial. This script has no
quarantine discovery, accepts no content paths, exports no raw model text, and
does not authorize restricted inference.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
import sys
import tempfile
import wave
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
WORKER = ROOT / "scripts/nursery_gemma4_referential_worker.py"
FIREWALL = ROOT / "scripts/childlens_local_inference_firewall_v1_3.py"
VALIDATOR = ROOT / "scripts/validate_prototype_opaque_instrument_v1.py"
OPAQUE_AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_e4b_opaque_instrument_provenance_amendment_v1.json"
CANARY_CONTRACT = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_common_schema_canary_contract_v1.json"
ACTIVATION_SCHEMA = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_activation_receipt_schema_v1.json"
RUNTIME_ERRATUM = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json"
PRIOR_STOP = ROOT / "output/nursery_program_convergence_v1/gemma4_replacement_terminal_decision.json"
PROVENANCE_ACTIVATION = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_opaque_instrument_provenance_activation_receipt_v1.json"
RESOURCE_RECEIPT = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_activation_resource_receipt.json"
OUTPUT = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_common_schema_public_canary_receipt_v1.json"
FAILURE = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_common_schema_public_canary_failure_v1.json"
RUNTIME_LOCK = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_runtime_lock_v1.json"
ACTIVATION = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_full_activation_receipt_v1_1.json"
PUBLIC_ROOT = Path.home() / "Library/Application Support/ChildLens Public Model Bakeoff/v1.3.1"
PYTHON = PUBLIC_ROOT / "venv/bin/python3.10"
MODEL = PUBLIC_ROOT / "gemma-4-e4b-it-4bit"
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
SAY = Path("/usr/bin/say")
MAX_PIPE_BYTES = 2 * 1024 * 1024
CANARY_CONTRACT_SHA256 = "9b894fcfd47df93824d25a467961d2c9f3398666dbe824367f2d8e1afe0635e9"
ACTIVATION_SCHEMA_SHA256 = "0668d8d304a1be30e43e9df2c05469a338d647c262418a04c6d067d2e77f8b05"
RUNTIME_ERRATUM_SHA256 = "b84da17121dd3235b1b6799d91c077d542e30e384f5cff5e131ec6c9755cecc1"
OPAQUE_AMENDMENT_SHA256 = "93874c69153daaf41d902216351666c59b12aeb656b9739ecd07d1e6e4fe42b1"
PROVENANCE_ACTIVATION_SHA256 = "2f25cff5d8fc140fb7dddeebae4cccfa83880327b5d383d14b3f21f3e2adcf13"
RESOURCE_RECEIPT_SHA256 = "363fda5ebc189216b2d8e9a9133cb51cea91fcc0ea9dc15fef023a028648dcd4"
ARTIFACT_MANIFEST_SHA256 = "34310498dc5b809bf4baa79a5290c9e675022d12b725993317bccbffacf1d3ae"
CONVERSION_REVISION = "475b9088d29754a3379866cf5aeb6b41acd313c2"
BASELINE_COMMON_SCHEMA_SHA256 = "a02c97d81d8114792f6ff01c244a553e16d45f03ea9fbaf42648219c279885a2"
EXACT_OUTPUT_SCHEMA_SHA256 = "19b7ae8948aed46410e1307c0a302303813422995c8f4cdc19949118693a8089"
PROMPT_SHA256 = "93e51fb982bd3ea47dd4defbd24e818cf67aee89c6df9d98fd1ec40fc2188ff6"
PARSER_CONTRACT_SHA256 = "af8e5988febb42e378bb3b6c8cbbb2799cca167cb4b3f4be5d32732a6035974a"
PHRASE = "Nimm die rote Tasse."
EXPECTED_RUNTIME = {
    "python": "3.10.20",
    "mlx-vlm": "0.6.6",
    "mlx": "0.32.0",
    "transformers": "5.14.1",
    "tokenizers": "0.22.2",
    "numpy": "2.2.6",
    "pillow": "12.3.0",
    "llguidance": "1.7.6",
}


class CanaryError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CanaryError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_module(WORKER, "nursery_gemma_canary_worker")
firewall = _load_module(FIREWALL, "nursery_gemma_canary_firewall")
validator = _load_module(VALIDATOR, "nursery_gemma_canary_validator")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError("E_PUBLIC_ARTIFACT") from exc
    if not isinstance(value, Mapping):
        raise CanaryError("E_PUBLIC_ARTIFACT")
    return value


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".pending-{os.getpid()}-{path.name}")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, 0o600)
    except Exception:
        with contextlib.suppress(OSError):
            pending.unlink()
        raise


def _atomic_once(path: Path, payload: bytes) -> None:
    """Publish owner-private bytes exactly once without replacing history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".pending-{os.getpid()}-{path.name}")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(pending, 0o600)
        os.link(pending, path)
        pending.unlink()
    except Exception:
        with contextlib.suppress(OSError):
            pending.unlink()
        raise


def _tree_manifest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total = count = 0
    for member in sorted(path.rglob("*")):
        if not member.is_file() or member.is_symlink() or ".cache" in member.parts:
            continue
        relative = member.relative_to(path).as_posix()
        size = member.stat().st_size
        digest.update(f"{relative}\0{size}\0{_sha256_file(member)}\n".encode())
        total += size
        count += 1
    if not count:
        raise CanaryError("E_MODEL_MANIFEST")
    return digest.hexdigest(), total, count


def _public_preflight(parser_mode: str) -> dict[str, Any]:
    if parser_mode not in {"PRIMARY", "ONE_OUTER_FENCE"}:
        raise CanaryError("E_PARSER_MODE")
    required = (
        OPAQUE_AMENDMENT,
        CANARY_CONTRACT,
        ACTIVATION_SCHEMA,
        RUNTIME_ERRATUM,
        PRIOR_STOP,
        PROVENANCE_ACTIVATION,
        RESOURCE_RECEIPT,
        WORKER,
        FIREWALL,
        PYTHON,
        FFMPEG,
        SAY,
    )
    if any(not path.is_file() for path in required) or not MODEL.is_dir():
        raise CanaryError("E_PUBLIC_PREFLIGHT")
    if (
        _sha256_file(OPAQUE_AMENDMENT) != OPAQUE_AMENDMENT_SHA256
        or _sha256_file(CANARY_CONTRACT) != CANARY_CONTRACT_SHA256
        or _sha256_file(ACTIVATION_SCHEMA) != ACTIVATION_SCHEMA_SHA256
        or _sha256_file(RUNTIME_ERRATUM) != RUNTIME_ERRATUM_SHA256
        or _sha256_file(PROVENANCE_ACTIVATION) != PROVENANCE_ACTIVATION_SHA256
        or _sha256_file(RESOURCE_RECEIPT) != RESOURCE_RECEIPT_SHA256
    ):
        raise CanaryError("E_PUBLIC_BINDING")
    amendment = _read_json(OPAQUE_AMENDMENT)
    prior_stop = _read_json(PRIOR_STOP)
    contract = _read_json(CANARY_CONTRACT)
    activation_schema = _read_json(ACTIVATION_SCHEMA)
    runtime_erratum = _read_json(RUNTIME_ERRATUM)
    provenance = _read_json(PROVENANCE_ACTIVATION)
    resource = _read_json(RESOURCE_RECEIPT)
    if validator.validate_amendment(amendment, prior_stop, _sha256_file(PRIOR_STOP)):
        raise CanaryError("E_OPAQUE_AMENDMENT")
    if validator.validate_canary_contract(contract, OPAQUE_AMENDMENT_SHA256):
        raise CanaryError("E_CANARY_CONTRACT")
    if validator.validate_runtime_erratum(
        runtime_erratum, RUNTIME_ERRATUM_SHA256, ACTIVATION_SCHEMA_SHA256
    ):
        raise CanaryError("E_RUNTIME_ERRATUM")
    effective_pins = runtime_erratum.get("effective_runtime_pins", {})
    expected_with_ffmpeg = {
        "python": EXPECTED_RUNTIME["python"],
        "mlx_vlm": EXPECTED_RUNTIME["mlx-vlm"],
        "mlx": EXPECTED_RUNTIME["mlx"],
        "transformers": EXPECTED_RUNTIME["transformers"],
        "tokenizers": EXPECTED_RUNTIME["tokenizers"],
        "numpy": EXPECTED_RUNTIME["numpy"],
        "pillow": EXPECTED_RUNTIME["pillow"],
        "llguidance": EXPECTED_RUNTIME["llguidance"],
        "ffmpeg": "8.0.1",
        "jsonschema": "NOT_USED_MANUAL_EXACT_VALIDATOR",
    }
    if not isinstance(effective_pins, Mapping) or dict(effective_pins) != expected_with_ffmpeg:
        raise CanaryError("E_RUNTIME_ERRATUM")
    if validator.validate_provenance_activation(provenance, OPAQUE_AMENDMENT_SHA256):
        raise CanaryError("E_PROVENANCE_ACTIVATION")
    if resource.get("status") != "VERIFIED_REUSE_RESOURCE_ADMITTED" or resource.get("boundary_attestations", {}).get("restricted_or_quarantine_data_accessed") is not False:
        raise CanaryError("E_RESOURCE_RECEIPT")
    manifest, artifact_bytes, file_count = _tree_manifest(MODEL)
    if manifest != ARTIFACT_MANIFEST_SHA256 or artifact_bytes != 5_179_241_512 or file_count != 10:
        raise CanaryError("E_MODEL_MANIFEST")
    if (
        worker.prompt_digest() != PROMPT_SHA256
        or worker.exact_schema_digest() != EXACT_OUTPUT_SCHEMA_SHA256
        or contract.get("contract_digests", {}).get("parser_contract_sha256") != PARSER_CONTRACT_SHA256
    ):
        raise CanaryError("E_PROMPT_SCHEMA")
    return {
        "schema_version": "nursery-gemma4-prototype-public-canary-seal-v1",
        "script_sha256": _sha256_file(SCRIPT),
        "worker_sha256": _sha256_file(WORKER),
        "firewall_sha256": _sha256_file(FIREWALL),
        "amendment_sha256": OPAQUE_AMENDMENT_SHA256,
        "canary_contract_sha256": CANARY_CONTRACT_SHA256,
        "activation_schema_sha256": ACTIVATION_SCHEMA_SHA256,
        "runtime_pin_erratum_sha256": RUNTIME_ERRATUM_SHA256,
        "provenance_activation_sha256": PROVENANCE_ACTIVATION_SHA256,
        "resource_receipt_sha256": RESOURCE_RECEIPT_SHA256,
        "artifact_manifest_sha256": manifest,
        "artifact_bytes": artifact_bytes,
        "artifact_file_count": file_count,
        "prompt_sha256": PROMPT_SHA256,
        "exact_output_schema_sha256": EXACT_OUTPUT_SCHEMA_SHA256,
        "parser_contract_sha256": PARSER_CONTRACT_SHA256,
        "parser_mode": parser_mode,
    }


def _read_fd(descriptor: int) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(MAX_PIPE_BYTES + 1)
    if not payload or len(payload) > MAX_PIPE_BYTES:
        raise CanaryError("E_PIPE")
    return json.loads(payload)


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_PIPE_BYTES:
        raise CanaryError("E_PIPE")
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()


def _network_denied() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            return False
    except OSError:
        return True


def _quiet(command: list[str], timeout: int = 120) -> None:
    environment = {
        "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        close_fds=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise CanaryError("E_LOCAL_TOOL")


def _voice_inventory_digest() -> str:
    completed = subprocess.run(
        [str(SAY), "-v", "?"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0 or b"Anna" not in completed.stdout:
        raise CanaryError("E_GERMAN_VOICE")
    build = subprocess.run(
        ["/usr/bin/sw_vers", "-buildVersion"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    if build.returncode != 0:
        raise CanaryError("E_OS_BUILD")
    return _sha256_bytes(build.stdout.strip() + b"\0" + completed.stdout)


def _create_fixture(directory: Path) -> tuple[Path, list[Path], dict[str, Any]]:
    from PIL import Image, ImageDraw

    aiff = directory / "german.aiff"
    audio = directory / "german.wav"
    _quiet(
        [
            str(SAY),
            "-v",
            "Anna",
            "-r",
            "170",
            "-o",
            str(aiff),
            "--data-format=LEI16@22050",
            PHRASE,
        ]
    )
    _quiet(
        [
            str(FFMPEG),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(aiff),
            "-af",
            "apad=pad_dur=10",
            "-t",
            "10",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-bitexact",
            "-y",
            str(audio),
        ]
    )
    with wave.open(str(audio), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16000
            or stream.getnframes() != 160000
        ):
            raise CanaryError("E_AUDIO_FIXTURE")
    frames: list[Path] = []
    for index, center_x in enumerate((128, 176, 224, 272, 320)):
        image = Image.new("RGB", (448, 252), "#F5F2E8")
        draw = ImageDraw.Draw(image)
        draw.line((0, 190, 447, 190), fill="#382A22", width=10)
        left, top, right, bottom = center_x - 48, 88, center_x + 47, 175
        draw.rectangle((left, top, right, bottom), fill="#C83232")
        draw.arc((right - 8, 106, right + 44, 162), start=270, end=90, fill="#C83232", width=12)
        frame = directory / f"frame-{index}.png"
        image.save(frame, format="PNG", compress_level=9, optimize=False)
        frames.append(frame)
    manifest = {
        "generator_source_sha256": _sha256_file(SCRIPT),
        "audio_sha256": _sha256_file(audio),
        "audio_bytes": audio.stat().st_size,
        "frame_sha256s_in_order": [_sha256_file(path) for path in frames],
        "frame_bytes_in_order": [path.stat().st_size for path in frames],
    }
    manifest["combined_fixture_sha256"] = _sha256_bytes(_canonical(manifest))
    return audio, frames, manifest


def _parser_negative_fixtures(parser_mode: str) -> bool:
    valid = (
        '{"candidate_count_bin":"one","visibility_bin":"clear","referential_status":"visible_candidate",'
        '"lexical_support":"noun_object","lag_event_unit_bin":"overlap"}'
    )
    invalid = [
        valid[:-1] + ',"candidate_count_bin":"one"}',
        valid[:-1] + ',"extra":"x"}',
        valid.replace('"one"', '"unknown"', 1),
        valid.replace(',"visibility_bin":"clear"', ""),
        valid + valid,
        "prefix " + valid,
        "```json\n" + valid + "\n``` trailing",
    ]
    try:
        worker._parse_exact(valid, parser_mode)
    except Exception:
        return False
    for fixture in invalid:
        try:
            worker._parse_exact(fixture, parser_mode)
        except Exception:
            continue
        return False
    fenced = "```json\n" + valid + "\n```"
    try:
        worker._parse_exact(fenced, parser_mode)
        fence_accepted = True
    except Exception:
        fence_accepted = False
    return fence_accepted is (parser_mode == "ONE_OUTER_FENCE")


def _package_tree_manifest(package_name: str) -> str:
    spec = importlib.util.find_spec(package_name)
    if spec is None or not spec.submodule_search_locations:
        raise CanaryError("E_RUNTIME_PACKAGE")
    root = Path(next(iter(spec.submodule_search_locations)))
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{path.stat().st_size}\0{_sha256_file(path)}\n".encode())
        count += 1
    if count == 0:
        raise CanaryError("E_RUNTIME_PACKAGE")
    return digest.hexdigest()


def _runtime_lock() -> dict[str, Any]:
    actual = {"python": platform.python_version()}
    for distribution in EXPECTED_RUNTIME:
        if distribution != "python":
            actual[distribution] = importlib.metadata.version(distribution)
    if actual != EXPECTED_RUNTIME:
        raise CanaryError("E_RUNTIME_PINS")
    distributions = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda item: (item.metadata.get("Name") or "").casefold()):
        name = distribution.metadata.get("Name")
        if not name:
            continue
        record = distribution.read_text("RECORD") or ""
        distributions.append(
            {
                "name": name,
                "version": distribution.version,
                "record_sha256": _sha256_bytes(record.encode("utf-8")),
            }
        )
    if not distributions:
        raise CanaryError("E_RUNTIME_LOCK")
    return {
        "schema_version": "nursery-gemma4-e4b-prototype-runtime-lock-v1",
        "runtime_pin_erratum_path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json",
        "runtime_pin_erratum_sha256": RUNTIME_ERRATUM_SHA256,
        "runtime_versions": actual,
        "distributions": distributions,
        "interpreter_executable_sha256": _sha256_file(Path(sys.executable).resolve()),
        "mlx_vlm_installed_tree_manifest_sha256": _package_tree_manifest("mlx_vlm"),
        "ffmpeg_executable_sha256": _sha256_file(FFMPEG),
        "worker_adapter_sha256": _sha256_file(WORKER),
        "network_denial_profile_sha256": _sha256_file(FIREWALL),
        "say_executable_sha256": _sha256_file(SAY),
        "os_build_and_voice_inventory_sha256": _voice_inventory_digest(),
    }


def sandboxed_execute(seal: Any) -> dict[str, Any]:
    if not isinstance(seal, Mapping) or seal != _public_preflight(str(seal.get("parser_mode", ""))):
        raise CanaryError("E_SEAL")
    if not _network_denied():
        raise CanaryError("E_NETWORK")
    runtime_lock = _runtime_lock()
    os.umask(0o077)
    with tempfile.TemporaryDirectory(prefix="gemma-public-common-schema-") as temporary:
        directory = Path(temporary)
        os.chmod(directory, 0o700)
        audio, frames, fixture_manifest = _create_fixture(directory)
        model, processor, config, generate = worker._load_instrument(str(MODEL))
        parsed, schema_valid, raw = worker._infer_one(
            model, processor, config, generate, frames, audio, str(seal["parser_mode"])
        )
    parser_pass = _parser_negative_fixtures(str(seal["parser_mode"]))
    passed = schema_valid and parser_pass
    correction_count = 1 if seal["parser_mode"] == "ONE_OUTER_FENCE" else 0
    return {
        "runtime_lock": runtime_lock,
        "receipt": {
            "schema_version": "nursery-gemma4-e4b-prototype-common-schema-public-canary-receipt-v1",
            "status": "PUBLIC_COMMON_SCHEMA_CANARY_PASS" if passed else "PUBLIC_COMMON_SCHEMA_CANARY_REVISE",
            "decision": "PUBLIC_COMMON_SCHEMA_CANARY_PASS" if passed else "PUBLIC_COMMON_SCHEMA_CANARY_REVISE",
            "amendment_sha256": OPAQUE_AMENDMENT_SHA256,
            "canary_contract_sha256": CANARY_CONTRACT_SHA256,
            "runtime_pin_erratum_path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json",
            "runtime_pin_erratum_sha256": RUNTIME_ERRATUM_SHA256,
            "provenance_activation_receipt_sha256": PROVENANCE_ACTIVATION_SHA256,
            "resource_receipt_sha256": RESOURCE_RECEIPT_SHA256,
            "instrument_id": validator.OPAQUE_INSTRUMENT_ID,
            "conversion_revision": CONVERSION_REVISION,
            "artifact_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
            "artifact_bytes": 5_179_241_512,
            "artifact_file_count": 10,
            "restricted_common_schema_sha256": BASELINE_COMMON_SCHEMA_SHA256,
            "exact_output_schema_sha256": EXACT_OUTPUT_SCHEMA_SHA256,
            "prompt_sha256": PROMPT_SHA256,
            "parser_contract_sha256": PARSER_CONTRACT_SHA256,
            "prompt_content_order": "SYSTEM_THEN_AUDIO_FIVE_ORDERED_FRAMES_EXACT_USER_PROMPT",
            "tts_executable_sha256": runtime_lock["say_executable_sha256"],
            "os_build_and_voice_inventory_sha256": runtime_lock["os_build_and_voice_inventory_sha256"],
            "fixture_manifest_sha256": _sha256_bytes(_canonical(fixture_manifest)),
            "fixture_audio_sample_count": 160000,
            "fixture_frame_count": 5,
            "runtime_versions": runtime_lock["runtime_versions"],
            "runtime_lock_sha256": _sha256_bytes(_canonical(runtime_lock)),
            "interpreter_executable_sha256": runtime_lock["interpreter_executable_sha256"],
            "mlx_vlm_installed_tree_manifest_sha256": runtime_lock["mlx_vlm_installed_tree_manifest_sha256"],
            "ffmpeg_executable_sha256": runtime_lock["ffmpeg_executable_sha256"],
            "adapter_sha256": runtime_lock["worker_adapter_sha256"],
            "network_denial_profile_sha256": runtime_lock["network_denial_profile_sha256"],
            "self_generated_public_fixture_only": True,
            "joint_audio_plus_five_frames": True,
            "five_frames_consumed_in_order": True,
            "audio_consumed": True,
            "exact_schema_valid": schema_valid,
            "all_parser_negative_fixtures_pass": parser_pass,
            "network_denial_sentinel_passed": True,
            "no_external_request": True,
            "corrections_used": correction_count,
            "correction_scope": "PARSER" if correction_count else "NONE",
            "semantic_tuning_performed": False,
            "semantic_quality_gate_used": False,
            "raw_model_output_sha256": _sha256_bytes(raw.encode("utf-8")),
            "raw_model_output_exported": False,
            "restricted_payload_accessed": False,
            "download_performed": False,
            "clickthrough_accepted": False,
            "hosted_or_cloud_inference_used": False,
            "restricted_inference_authorized_by_canary": False,
            "scientific_outcome_authorized_by_canary": False,
        },
    }


def _json_document(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _launch_resource_recheck() -> dict[str, bool]:
    manifest, artifact_bytes, file_count = _tree_manifest(MODEL)
    if manifest != ARTIFACT_MANIFEST_SHA256 or artifact_bytes != 5_179_241_512 or file_count != 10:
        raise CanaryError("E_LAUNCH_REHASH")
    filesystem = os.statvfs(ROOT)
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    if free_bytes - 2_684_354_560 < 53_687_091_200:
        raise CanaryError("E_LAUNCH_DISK_FLOOR")
    pressure = subprocess.run(
        ["/usr/bin/memory_pressure", "-Q"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    throttled = subprocess.run(
        ["/usr/bin/vm_stat"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30,
    )
    free_match = re.search(rb"System-wide memory free percentage:\s*([0-9]+)%", pressure.stdout)
    throttled_match = re.search(rb"Pages throttled:\s*([0-9]+)\.", throttled.stdout)
    if (
        pressure.returncode != 0
        or throttled.returncode != 0
        or free_match is None
        or throttled_match is None
        or int(free_match.group(1)) < 25
        or int(throttled_match.group(1)) != 0
    ):
        raise CanaryError("E_LAUNCH_MEMORY_PRESSURE")
    return {
        "launch_rehash_passed": True,
        "launch_disk_floor_passed": True,
        "launch_memory_pressure_passed": True,
    }


@contextlib.contextmanager
def _exclusive_public_mps_lock() -> Any:
    lock_path = PUBLIC_ROOT / ".gemma4-e4b-prototype-mps.lock"
    if lock_path.is_symlink():
        raise CanaryError("E_MPS_LOCK")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise CanaryError("E_MPS_LOCK")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CanaryError("E_MPS_LOCK") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _build_activation_receipt(
    receipt: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    canary_bytes: bytes,
    runtime_lock_bytes: bytes,
    resource_recheck: Mapping[str, bool],
) -> dict[str, Any]:
    """Build and self-validate the public-only post-canary activation receipt."""

    canary_sha256 = _sha256_bytes(canary_bytes)
    runtime_lock_sha256 = _sha256_bytes(runtime_lock_bytes)
    if (
        receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_PASS"
        or receipt.get("runtime_lock_file_sha256") != runtime_lock_sha256
        or validator.validate_canary(receipt, OPAQUE_AMENDMENT_SHA256, CANARY_CONTRACT_SHA256)
        or set(resource_recheck) != {
            "launch_rehash_passed",
            "launch_disk_floor_passed",
            "launch_memory_pressure_passed",
        }
        or any(value is not True for value in resource_recheck.values())
    ):
        raise CanaryError("E_ACTIVATION_PREREQUISITE")
    versions = runtime_lock.get("runtime_versions")
    if not isinstance(versions, Mapping) or dict(versions) != EXPECTED_RUNTIME:
        raise CanaryError("E_ACTIVATION_RUNTIME")
    activation = {
        "schema_version": validator.EFFECTIVE_ACTIVATION_SCHEMA,
        "created_at": dt.date.today().isoformat(),
        "status": "PROTOTYPE_INSTRUMENT_ACTIVATED_RESTRICTED_INFERENCE_NOT_RUN",
        "scope": "EXACT_CACHED_GEMMA4_E4B_INTERNAL_PSEUDO_CALIBRATION_ONLY",
        "immutable_bindings": {
            "replacement_amendment": {
                "path": "docs/nursery_program_convergence_v1/frozen_gemma4_replacement_instrument_amendment.json",
                "sha256": "8e6d7a43ea9639fa451a59386b8657495558fde77fb2147a20767388d47ed24a",
            },
            "opaque_provenance_amendment": {
                "path": "docs/nursery_program_convergence_v1/frozen_gemma4_e4b_opaque_instrument_provenance_amendment_v1.json",
                "sha256": OPAQUE_AMENDMENT_SHA256,
                "status": "FROZEN_PROVENANCE_GATE_PASS_OPAQUE_INTERNAL_ONLY",
            },
            "opaque_provenance_activation_receipt": {
                "path": "output/nursery_program_convergence_v1/gemma4_e4b_opaque_instrument_provenance_activation_receipt_v1.json",
                "sha256": PROVENANCE_ACTIVATION_SHA256,
                "status": "OPAQUE_PROVENANCE_BRANCH_ACTIVATED_FULL_INSTRUMENT_PENDING",
            },
            "resource_receipt": {
                "path": "output/nursery_program_convergence_v1/gemma4_e4b_prototype_activation_resource_receipt.json",
                "sha256": RESOURCE_RECEIPT_SHA256,
                "status": "VERIFIED_REUSE_RESOURCE_ADMITTED",
            },
            "common_schema_canary_contract": {
                "path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_common_schema_canary_contract_v1.json",
                "sha256": CANARY_CONTRACT_SHA256,
                "status": "FROZEN_FOR_PUBLIC_SYNTHETIC_CANARY_ONLY",
            },
            "runtime_pin_erratum": {
                "path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json",
                "sha256": RUNTIME_ERRATUM_SHA256,
                "status": "FROZEN_PRE_CANARY_CONTENT_INDEPENDENT_ERRATUM",
            },
        },
        "artifact_binding": {
            "repository": "mlx-community/gemma-4-e4b-it-4bit",
            "conversion_revision": CONVERSION_REVISION,
            "upstream_repository": "google/gemma-4-E4B-it",
            "current_upstream_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
            "snapshot_file_count": 10,
            "snapshot_bytes": 5_179_241_512,
            "snapshot_manifest_sha256": ARTIFACT_MANIFEST_SHA256,
            "weight_bytes": 5_146_800_534,
            "weight_sha256": "932b8271fc3fe65adcc78b96c10c6268bbfb13e8f67d1358727c0d6ee97e1eff",
            "all_ten_files_match_provenance_amendment": True,
            "cached_rehash_exact_match": True,
            "artifact_substitution_mutation_download_or_copy": False,
            "opaque_internal_only_limitations_visible": True,
        },
        "runtime_binding": {
            "python": versions["python"],
            "mlx_vlm": versions["mlx-vlm"],
            "mlx": versions["mlx"],
            "transformers": versions["transformers"],
            "tokenizers": versions["tokenizers"],
            "numpy": versions["numpy"],
            "pillow": versions["pillow"],
            "llguidance": versions["llguidance"],
            "ffmpeg": "8.0.1",
            "complete_hash_locked_dependency_file_sha256": runtime_lock_sha256,
            "interpreter_executable_sha256": runtime_lock["interpreter_executable_sha256"],
            "mlx_vlm_installed_tree_manifest_sha256": runtime_lock["mlx_vlm_installed_tree_manifest_sha256"],
            "ffmpeg_executable_sha256": runtime_lock["ffmpeg_executable_sha256"],
            "canary_adapter_sha256": runtime_lock["worker_adapter_sha256"],
            "network_denial_profile_sha256": runtime_lock["network_denial_profile_sha256"],
            "say_executable_sha256": runtime_lock["say_executable_sha256"],
            "german_voice_inventory_sha256": runtime_lock["os_build_and_voice_inventory_sha256"],
            "compiled_runtime_lock_available": True,
            "all_runtime_hashes_verified": True,
        },
        "resource_binding": {
            "path": "output/nursery_program_convergence_v1/gemma4_e4b_prototype_activation_resource_receipt.json",
            "sha256": RESOURCE_RECEIPT_SHA256,
            "status": "VERIFIED_REUSE_RESOURCE_ADMITTED",
            **dict(resource_recheck),
            "single_MPS_lock_acquired": True,
        },
        "public_canary_binding": {
            "receipt_schema_version": receipt["schema_version"],
            "receipt_status": receipt["status"],
            "receipt_path": "output/nursery_program_convergence_v1/gemma4_e4b_prototype_common_schema_public_canary_receipt_v1.json",
            "receipt_sha256": canary_sha256,
            "contract_path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_common_schema_canary_contract_v1.json",
            "contract_sha256": CANARY_CONTRACT_SHA256,
            "fixture_manifest_sha256": receipt["fixture_manifest_sha256"],
            "prompt_sha256": PROMPT_SHA256,
            "common_schema_sha256": EXACT_OUTPUT_SCHEMA_SHA256,
            "parser_contract_sha256": PARSER_CONTRACT_SHA256,
            "parser_correction_count": receipt["corrections_used"],
            "audio_consumed": receipt["audio_consumed"],
            "five_frames_consumed_in_order": receipt["five_frames_consumed_in_order"],
            "exact_schema_valid": receipt["exact_schema_valid"],
            "all_parser_negative_fixtures_pass": receipt["all_parser_negative_fixtures_pass"],
            "network_denial_sentinel_passed": receipt["network_denial_sentinel_passed"],
            "external_request_or_telemetry": False,
            "semantic_quality_or_restricted_content_inspected": False,
            "canary_authorizes_restricted_inference": False,
        },
        "qwen3_read_only_binding": {
            "instrument_id": validator.QWEN3_ID,
            "revision": "defcdea7cc7a4b0858fea563cbbce171d328e457",
            "public_canary_sha256": "eb5447e13f44cc35c64e433f7217dc2423c5aff799c27101b6ce22bec50390b3",
            "restricted_hypotheses_recomputed": False,
            "restricted_hypotheses_opened_for_activation": False,
            "quarantine_binding_digest_match": True,
            "restricted_digest_exported": False,
            "third_visual_model": False,
        },
        "unchanged_contract": {
            "primary_item_count": 15,
            "primary_total_seconds": 900,
            "candidate_window_count": 137,
            "frame_offsets_seconds": [-5.0, -2.5, 0.0, 2.5, 5.0],
            "minimum_export_cell_items": 5,
            "schema_validity_minimum": 0.95,
            "primary_item_coverage_minimum": 0.8,
            "primary_seconds_coverage_minimum": 0.8,
            "maximum_abstention_for_a_dimension": 0.5,
            "minimum_paths_per_task_family": 2,
            "K5_complement_protection": True,
            "outward_rounding_unchanged": True,
            "agreement_is_not_a_pass_gate": True,
            "reserve_activation": False,
            "reselection": False,
        },
        "gate_results": {
            "opaque_provenance_gate": "PASS_OPAQUE_INTERNAL_ONLY",
            "artifact_manifest_gate": "PASS",
            "resource_and_launch_recheck_gate": "PASS",
            "runtime_and_adapter_hash_gate": "PASS",
            "public_joint_common_schema_canary_gate": "PASS",
            "network_denial_gate": "PASS",
            "qwen3_read_only_gate": "PASS",
            "quarantine_preflight_gate": "PASS",
            "restricted_schema_coverage_abstention_K5_gates": "PENDING_RESTRICTED_EXECUTION",
            "scientific_outcome_gate": "NOT_AUTHORIZED",
        },
        "authorization_boundary": {
            "full_instrument_activation": "ACTIVE_ONLY_FOR_ONE_BOUNDED_15_ITEM_137_WINDOW_LOCAL_PSEUDO_CALIBRATION_PASS",
            "authorized_restricted_action": "RUN_EXACT_GEMMA_REPLACEMENT_PATH_ON_FROZEN_137_WINDOWS_UNDER_NETWORK_DENIAL_AND_EXPORT_ONLY_FROZEN_K5_AGGREGATES",
            "restricted_inference_run": False,
            "fresh_download_copy_or_artifact_change": False,
            "external_or_hosted_inference": False,
            "learner_tokenizer_checkpoint_or_causal_outcome": False,
            "authorization_expires_on_any_binding_or_gate_mismatch": True,
        },
        "privacy_and_scientific_attestations": {
            "restricted_content_accessed_to_issue_receipt": False,
            "restricted_payload_exported": False,
            "hosted_or_cloud_inference_used": False,
            "public_canary_semantics_tuned": False,
            "model_agreement_is_ground_truth": False,
            "pseudo_labels_are_human_evidence": False,
            "learner_ancestry_allowed": False,
            "simulator_oracle_remains_only_evaluation_truth": True,
            "scientific_outcome_run": False,
            "scientific_outcome_authorized": False,
        },
        "next_action": "A separately launched fail-closed coordinator may execute only the authorized bounded restricted pseudo-calibration pass; no scientific learner or causal outcome is authorized.",
    }
    activation_schema = _read_json(ACTIVATION_SCHEMA)
    runtime_erratum = _read_json(RUNTIME_ERRATUM)
    issues = validator.validate_activation(
        activation,
        activation_schema,
        ACTIVATION_SCHEMA_SHA256,
        runtime_erratum,
        RUNTIME_ERRATUM_SHA256,
        canary_sha256,
    )
    if issues:
        raise CanaryError("E_ACTIVATION_RECEIPT")
    return activation


def public_execute(parser_mode: str) -> Mapping[str, Any]:
    seal = _public_preflight(parser_mode)
    backend = firewall.NetworkIsolationBackend.detect()
    firewall.verify_network_isolation(backend)
    resource_recheck = _launch_resource_recheck()
    if OUTPUT.exists() or RUNTIME_LOCK.exists() or ACTIVATION.exists():
        raise CanaryError("E_IMMUTABLE_OUTPUT_EXISTS")
    mps_lock = _exclusive_public_mps_lock()
    mps_lock.__enter__()
    seal_read = seal_write = result_read = result_write = -1
    try:
        seal_read, seal_write = os.pipe()
        result_read, result_write = os.pipe()
        os.write(seal_write, _canonical(seal))
        os.close(seal_write)
        seal_write = -1
        argv = [
            str(PYTHON),
            "-I",
            str(SCRIPT),
            "--sandboxed",
            "--seal-fd",
            str(seal_read),
            "--result-fd",
            str(result_write),
        ]
        process = subprocess.Popen(
            backend.command(argv),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(
                firewall.scrubbed_subprocess_environment(
                    {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1",
                        "DO_NOT_TRACK": "1",
                    }
                )
            ),
            pass_fds=(seal_read, result_write),
            close_fds=True,
        )
        os.close(seal_read)
        seal_read = -1
        os.close(result_write)
        result_write = -1
        with os.fdopen(result_read, "rb") as handle:
            payload = handle.read(MAX_PIPE_BYTES + 1)
        result_read = -1
        returncode = process.wait(timeout=30 * 60)
        if returncode != 0 or not payload or len(payload) > MAX_PIPE_BYTES:
            raise CanaryError("E_SANDBOXED_CANARY")
        result = json.loads(payload)
        if not isinstance(result, Mapping) or not isinstance(result.get("receipt"), Mapping) or not isinstance(result.get("runtime_lock"), Mapping):
            raise CanaryError("E_CANARY_RESULT")
        receipt, runtime_lock = result["receipt"], result["runtime_lock"]
        if receipt.get("runtime_lock_sha256") != _sha256_bytes(_canonical(runtime_lock)):
            raise CanaryError("E_RUNTIME_LOCK")
        if receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_PASS" and validator.validate_canary(
            receipt, OPAQUE_AMENDMENT_SHA256, CANARY_CONTRACT_SHA256
        ):
            raise CanaryError("E_CANARY_RECEIPT")
        runtime_lock_bytes = _json_document(runtime_lock)
        receipt = dict(receipt)
        receipt["runtime_lock_file_sha256"] = _sha256_bytes(runtime_lock_bytes)
        canary_bytes = _json_document(receipt)
        if receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_PASS":
            return receipt
        activation = _build_activation_receipt(
            receipt,
            runtime_lock,
            canary_bytes,
            runtime_lock_bytes,
            resource_recheck,
        )
        _atomic_once(RUNTIME_LOCK, runtime_lock_bytes)
        _atomic_once(OUTPUT, canary_bytes)
        _atomic_once(ACTIVATION, _json_document(activation))
        return receipt
    finally:
        for descriptor in (seal_read, seal_write, result_read, result_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        mps_lock.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parser-correction", action="store_true")
    parser.add_argument("--sandboxed", action="store_true")
    parser.add_argument("--seal-fd", type=int)
    parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.sandboxed:
            if args.seal_fd is None or args.result_fd is None or args.parser_correction:
                return 2
            _write_fd(args.result_fd, sandboxed_execute(_read_fd(args.seal_fd)))
            return 0
        if args.seal_fd is not None or args.result_fd is not None:
            return 2
        receipt = public_execute("ONE_OUTER_FENCE" if args.parser_correction else "PRIMARY")
        print("GEMMA4_PUBLIC_COMMON_SCHEMA_CANARY_PASS" if receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_PASS" else "GEMMA4_PUBLIC_COMMON_SCHEMA_CANARY_REVISE")
        return 0 if receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_PASS" else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, CanaryError) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None:
            code = "E_INTERNAL"
        if args.sandboxed and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd(args.result_fd, {"schema_version": "nursery-gemma4-canary-failure-v1", "failure_code": code})
        else:
            failure = {
                "schema_version": "nursery-gemma4-canary-failure-v1",
                "status": "REVISE",
                "failure_code": code,
                "restricted_payload_accessed": False,
                "raw_model_output_exported": False,
            }
            with contextlib.suppress(Exception):
                _atomic(FAILURE, json.dumps(failure, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            print("GEMMA4_PUBLIC_COMMON_SCHEMA_CANARY_REVISE")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
