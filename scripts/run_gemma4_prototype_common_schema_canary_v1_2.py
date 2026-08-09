#!/usr/bin/env python3
"""Additive CAF-only public Gemma canary and activation runner v1.2.

This module preserves the failed AIFF runner and receipts. It applies only the
frozen public-fixture transport overlay (AIFF -> CAF), keeps the normalized WAV
and every model/scientific contract unchanged, and never discovers restricted
data. No output path from the prior attempt is writable here.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import wave
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
LEGACY_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary.py"
CAF_VALIDATOR = ROOT / "scripts/validate_gemma_caf_transport_amendment_v1.py"
CAF_AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json"
NO_FALLBACK_OVERRIDE = ROOT / "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json"
OUTPUT_ROOT = ROOT / "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1"
OUTPUT = OUTPUT_ROOT / "public_common_schema_canary_receipt_v1.json"
FAILURE = OUTPUT_ROOT / "public_common_schema_canary_failure_v1.json"
RUNTIME_LOCK = OUTPUT_ROOT / "runtime_lock_receipt_v1.json"
ACTIVATION = OUTPUT_ROOT / "full_activation_receipt_v1.json"

LEGACY_RUNNER_SHA256 = "d5f3402c21043a0bba480e9540db7bac2d2eb01f65887673538db5211d19e8ea"
CAF_AMENDMENT_SHA256 = "3ecf4bf6c62920d73177d029c18918f38366d34f7d7997642037696def37bbd0"
NO_FALLBACK_OVERRIDE_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
LEGACY_FAILURE_SHA256 = "ef5031b3b90a3611eb488ff9fa76a48e0b2f2a0af0ab5e2230316d3234a0eda6"
LEGACY_TERMINAL_SHA256 = "046b06209bd644ae5f3ebcb0f83bb37edca42f0fdb3a1dfff2bb492a0ffecb90"
LEGACY_VALIDATION_SHA256 = "c5a94a2098a7e07e0d43c77da9d018444773519a0e5adaca276fcaa0ff2340c3"
LEGACY_DISPOSITION_SHA256 = "ee04891eb010a5244ec5de2a813154c58d9e48a7ddfbcc6612106ed759b74a7d"
CANARY_SCHEMA = "nursery-gemma4-e4b-prototype-common-schema-public-canary-caf-v1"
RUNTIME_SCHEMA = "nursery-gemma4-e4b-prototype-runtime-lock-caf-v1"
ACTIVATION_SCHEMA = "nursery-gemma4-e4b-prototype-full-activation-receipt-caf-v1"
FAILURE_SCHEMA = "nursery-gemma4-public-canary-caf-failure-v1"
MAX_PIPE_BYTES = 2 * 1024 * 1024
LEGACY_FRAME_SHA256S = [
    "aa1e90e06d2e11a4f2d8c7a9180bb41e6e47d73bb1669bf6459716ecd5b841a4",
    "e2aa5524b382c1509e7cdc3ed1c980bbc81638ead240d89bc62fc79b31b4ad87",
    "f523715d795e5d562200c3468bcc95398c7cb6ea95bd0cc73f3219740b17bd01",
    "71d19b0d7378fd05dab38afa463a43dac031fa1e93a66f1a4fc62b5339d968ff",
    "20463f892bcbf2780d71e32964777f46fbe25acdab4bcf75d798c6c11fe8f19a",
]


class CafCanaryError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CafCanaryError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_module(LEGACY_RUNNER, "nursery_gemma_caf_legacy_runner")
caf_validator = _load_module(CAF_VALIDATOR, "nursery_gemma_caf_transport_validator")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise CafCanaryError("E_PUBLIC_ARTIFACT") from exc
    if not isinstance(value, Mapping):
        raise CafCanaryError("E_PUBLIC_ARTIFACT")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_document(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _prior_paths() -> dict[Path, str]:
    return {
        LEGACY_RUNNER: LEGACY_RUNNER_SHA256,
        legacy.FAILURE: LEGACY_FAILURE_SHA256,
        ROOT / "output/nursery_program_convergence_v1/prototype_gemma_activation_terminal_decision_v1.json": LEGACY_TERMINAL_SHA256,
        ROOT / "output/nursery_program_convergence_v1/prototype_gemma_activation_validation_receipt_v1.json": LEGACY_VALIDATION_SHA256,
        ROOT / "docs/nursery_program_convergence_v1/prototype_gemma_activation_technical_revision_disposition_v1.md": LEGACY_DISPOSITION_SHA256,
    }


def _validate_caf_amendment() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if _sha256_file(CAF_AMENDMENT) != CAF_AMENDMENT_SHA256 or _sha256_file(NO_FALLBACK_OVERRIDE) != NO_FALLBACK_OVERRIDE_SHA256:
        raise CafCanaryError("E_CAF_BINDING")
    if any(not path.is_file() or _sha256_file(path) != digest for path, digest in _prior_paths().items()):
        raise CafCanaryError("E_PRIOR_IMMUTABILITY")
    amendment = _read_json(CAF_AMENDMENT)
    override = _read_json(NO_FALLBACK_OVERRIDE)
    if (
        amendment.get("schema_version") != "nursery-gemma4-public-canary-caf-transport-amendment-v1"
        or amendment.get("status") != "FROZEN_PRE_CANARY_CAF_TRANSPORT_ONLY"
        or amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_TRANSPORT_OVERLAY"
        or amendment.get("historical_artifacts_modified") is not False
        or amendment.get("public_canary_run") is not False
        or amendment.get("restricted_content_accessed") is not False
        or amendment.get("restricted_inference_run") is not False
        or amendment.get("scientific_outcome_run") is not False
    ):
        raise CafCanaryError("E_CAF_AMENDMENT")
    changes = amendment.get("exact_transport_delta", {}).get("changes")
    if not isinstance(changes, list) or len(changes) != 2:
        raise CafCanaryError("E_CAF_DELTA")
    encoded = _canonical(changes).decode("utf-8")
    if encoded.count("german.aiff") != 2 or encoded.count("german.caf") != 2:
        raise CafCanaryError("E_CAF_DELTA")
    if legacy.validator.validate_no_fallback_override(override, NO_FALLBACK_OVERRIDE_SHA256):
        raise CafCanaryError("E_NO_FALLBACK")
    base_contract = _read_json(legacy.CANARY_CONTRACT)
    terminal = _read_json(ROOT / "output/nursery_program_convergence_v1/prototype_gemma_activation_terminal_decision_v1.json")
    validation = _read_json(ROOT / "output/nursery_program_convergence_v1/prototype_gemma_activation_validation_receipt_v1.json")
    prior_failure = _read_json(legacy.FAILURE)
    if (
        caf_validator.validate_prior_receipts(
            terminal, LEGACY_TERMINAL_SHA256,
            validation, LEGACY_VALIDATION_SHA256,
            prior_failure, LEGACY_FAILURE_SHA256,
            LEGACY_DISPOSITION_SHA256,
        )
        or caf_validator.validate_no_fallback(override, NO_FALLBACK_OVERRIDE_SHA256)
        or caf_validator.validate_amendment(
            amendment, CAF_AMENDMENT_SHA256,
            base_contract, legacy.CANARY_CONTRACT_SHA256,
        )
    ):
        raise CafCanaryError("E_CAF_AMENDMENT")
    legacy_outputs = (legacy.OUTPUT, legacy.RUNTIME_LOCK, legacy.ACTIVATION)
    if any(path.exists() for path in legacy_outputs):
        raise CafCanaryError("E_PRIOR_CANONICAL_OUTPUT_MUTATION")
    return amendment, override


def _caf_common_receipt() -> dict[str, Any]:
    return {
        "caf_transport_amendment_path": "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json",
        "caf_transport_amendment_sha256": CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": NO_FALLBACK_OVERRIDE_SHA256,
        "output_namespace": "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1",
        "historical_output_overwritten": False,
        "canonical_output_overwritten": False,
        "intermediate_container": "CAF",
        "final_wav_contract_unchanged": True,
        "parser_corrections_spent_before_revised_canary": 0,
        "parser_allowance_preserved": True,
        "restricted_content_accessed_before_activation": False,
        "restricted_inference_run": False,
        "scientific_outcome_run": False,
    }


def _ensure_new_namespace_absent() -> None:
    if OUTPUT_ROOT.exists():
        raise CafCanaryError("E_NEW_NAMESPACE_EXISTS")


def _validate_parser_correction_state() -> None:
    if not OUTPUT_ROOT.is_dir() or OUTPUT.exists() or RUNTIME_LOCK.exists() or ACTIVATION.exists():
        raise CafCanaryError("E_PARSER_CORRECTION_STATE")
    members = list(OUTPUT_ROOT.iterdir())
    if members != [FAILURE] or not FAILURE.is_file():
        raise CafCanaryError("E_PARSER_CORRECTION_STATE")
    failure = _read_json(FAILURE)
    if (
        failure.get("schema_version") != FAILURE_SCHEMA
        or failure.get("failure_code") != "E_SCHEMA_OUTPUT"
        or failure.get("parser_mode") != "PRIMARY"
        or failure.get("parser_correction_eligible") is not True
        or failure.get("caf_transport_amendment_sha256") != CAF_AMENDMENT_SHA256
        or failure.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_OVERRIDE_SHA256
    ):
        raise CafCanaryError("E_PARSER_CORRECTION_STATE")


def _ensure_output_root() -> None:
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(mode=0o700)
    info = OUTPUT_ROOT.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise CafCanaryError("E_OUTPUT_PRIVACY")


def _atomic_once(path: Path, payload: bytes) -> None:
    _ensure_output_root()
    legacy._atomic_once(path, payload)


def _public_preflight(parser_mode: str) -> dict[str, Any]:
    _validate_caf_amendment()
    if parser_mode not in {"PRIMARY", "ONE_OUTER_FENCE"}:
        raise CafCanaryError("E_PARSER_MODE")
    if parser_mode == "PRIMARY":
        _ensure_new_namespace_absent()
    else:
        _validate_parser_correction_state()
    legacy_seal = legacy._public_preflight(parser_mode)
    return {
        "schema_version": "nursery-gemma4-public-canary-caf-seal-v1",
        "runner_sha256": _sha256_file(SCRIPT),
        "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
        "caf_transport_amendment_sha256": CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": NO_FALLBACK_OVERRIDE_SHA256,
        "intermediate_container": "CAF",
        "final_container": "WAV",
        "parser_mode": parser_mode,
        "legacy_preflight": legacy_seal,
    }


_LAST_FIXTURE_MANIFEST: dict[str, Any] | None = None


def _create_fixture(directory: Path) -> tuple[Path, list[Path], dict[str, Any]]:
    from PIL import Image, ImageDraw

    global _LAST_FIXTURE_MANIFEST
    caf = directory / "german.caf"
    audio = directory / "german.wav"
    legacy._quiet([
        str(legacy.SAY), "-v", "Anna", "-r", "170", "-o", str(caf),
        "--data-format=LEI16@22050", legacy.PHRASE,
    ])
    legacy._quiet([
        str(legacy.FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(caf), "-af", "apad=pad_dur=10", "-t", "10", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-bitexact", "-y", str(audio),
    ])
    caf.unlink()
    raw_wav = audio.read_bytes()
    if len(raw_wav) < 44 or raw_wav[:4] != b"RIFF" or raw_wav[8:12] != b"WAVE":
        raise CafCanaryError("E_FINAL_WAV_BYTES")
    with wave.open(str(audio), "rb") as stream:
        if (
            stream.getcomptype() != "NONE"
            or stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16000
            or stream.getnframes() != 160000
            or len(stream.readframes(160000)) != 320000
        ):
            raise CafCanaryError("E_FINAL_WAV_SPEC")
    frames: list[Path] = []
    for index, center_x in enumerate((128, 176, 224, 272, 320)):
        image = Image.new("RGB", (448, 252), "#F5F2E8")
        draw = ImageDraw.Draw(image)
        draw.line((0, 190, 447, 190), fill="#382A22", width=10)
        left, top, right, bottom = center_x - 48, 88, center_x + 48, 176
        draw.rectangle((left, top, right, bottom), fill="#C83232")
        draw.arc((right - 8, 106, right + 44, 162), start=270, end=90, fill="#C83232", width=12)
        frame = directory / f"frame-{index}.png"
        image.save(frame, format="PNG", compress_level=9, optimize=False)
        frames.append(frame)
    frame_sha256s = [_sha256_file(path) for path in frames]
    if frame_sha256s != LEGACY_FRAME_SHA256S:
        raise CafCanaryError("E_LEGACY_FRAME_PROJECTION")
    manifest = {
        "generator_source_sha256": _sha256_file(SCRIPT),
        "caf_transport_amendment_sha256": CAF_AMENDMENT_SHA256,
        "intermediate_container": "CAF",
        "intermediate_deleted_before_inference": not caf.exists(),
        "audio_sha256": _sha256_bytes(raw_wav),
        "audio_bytes": len(raw_wav),
        "audio_contract": {"container": "WAV", "channels": 1, "sample_width_bytes": 2, "sample_rate_hz": 16000, "sample_count": 160000},
        "frame_sha256s_in_order": frame_sha256s,
        "frame_bytes_in_order": [path.stat().st_size for path in frames],
    }
    manifest["combined_fixture_sha256"] = _sha256_bytes(_canonical(manifest))
    _LAST_FIXTURE_MANIFEST = manifest
    return audio, frames, manifest


def _legacy_canary_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(receipt)
    projected["schema_version"] = legacy.validator.CANARY_SCHEMA
    projected["status"] = "PUBLIC_COMMON_SCHEMA_CANARY_PASS"
    return projected


def _validate_caf_canary(receipt: Mapping[str, Any]) -> None:
    if caf_validator.validate_revised_chain_receipt(receipt, CAF_AMENDMENT_SHA256):
        raise CafCanaryError("E_CAF_CANARY_RECEIPT")
    if (
        receipt.get("schema_version") != CANARY_SCHEMA
        or receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_CAF_PASS"
        or receipt.get("decision") != "PUBLIC_COMMON_SCHEMA_CANARY_CAF_PASS"
        or receipt.get("caf_transport_amendment_sha256") != CAF_AMENDMENT_SHA256
        or receipt.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_OVERRIDE_SHA256
        or receipt.get("legacy_runner_sha256") != LEGACY_RUNNER_SHA256
        or receipt.get("prior_aiff_failure_receipt_sha256") != LEGACY_FAILURE_SHA256
        or receipt.get("intermediate_container") != "CAF"
        or receipt.get("intermediate_deleted_before_inference") is not True
        or receipt.get("final_wav_sha256") != receipt.get("fixture_audio_sha256")
        or not isinstance(receipt.get("final_wav_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("final_wav_sha256")))
        or receipt.get("final_wav_contract")
        != {"container": "WAV", "channels": 1, "sample_width_bytes": 2, "sample_rate_hz": 16000, "sample_count": 160000}
        or receipt.get("historical_output_paths_overwritten") is not False
        or receipt.get("automatic_fallback_if_revised_canary_fails") is not False
    ):
        raise CafCanaryError("E_CAF_CANARY_RECEIPT")
    if legacy.validator.validate_canary(
        _legacy_canary_projection(receipt),
        legacy.OPAQUE_AMENDMENT_SHA256,
        legacy.CANARY_CONTRACT_SHA256,
    ):
        raise CafCanaryError("E_CAF_CANARY_RECEIPT")


def sandboxed_execute(seal: Any) -> dict[str, Any]:
    if not isinstance(seal, Mapping) or dict(seal) != _public_preflight(str(seal.get("parser_mode", ""))):
        raise CafCanaryError("E_SEAL")
    if not legacy._network_denied():
        raise CafCanaryError("E_NETWORK")
    runtime_lock = dict(legacy._runtime_lock())
    runtime_lock.update({
        **_caf_common_receipt(),
        "schema_version": RUNTIME_SCHEMA,
        "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
        "runner_sha256": _sha256_file(SCRIPT),
        "intermediate_container": "CAF",
        "final_container": "WAV",
    })
    os.umask(0o077)
    global _LAST_FIXTURE_MANIFEST
    _LAST_FIXTURE_MANIFEST = None
    with tempfile.TemporaryDirectory(prefix="gemma-public-common-schema-caf-") as temporary:
        directory = Path(temporary)
        os.chmod(directory, 0o700)
        audio, frames, fixture_manifest = _create_fixture(directory)
        model, processor, config, generate = legacy.worker._load_instrument(str(legacy.MODEL))
        parsed, schema_valid, raw = legacy.worker._infer_one(
            model, processor, config, generate, frames, audio, str(seal["parser_mode"])
        )
        del parsed
    if _LAST_FIXTURE_MANIFEST != fixture_manifest:
        raise CafCanaryError("E_FIXTURE_STATE")
    parser_pass = legacy._parser_negative_fixtures(str(seal["parser_mode"]))
    passed = schema_valid and parser_pass
    correction_count = 1 if seal["parser_mode"] == "ONE_OUTER_FENCE" else 0
    status = "PUBLIC_COMMON_SCHEMA_CANARY_CAF_PASS" if passed else "PUBLIC_COMMON_SCHEMA_CANARY_CAF_REVISE"
    receipt = {
        **_caf_common_receipt(),
        "schema_version": CANARY_SCHEMA,
        "status": status,
        "decision": status,
        "amendment_sha256": legacy.OPAQUE_AMENDMENT_SHA256,
        "canary_contract_sha256": legacy.CANARY_CONTRACT_SHA256,
        "runtime_pin_erratum_path": "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json",
        "runtime_pin_erratum_sha256": legacy.RUNTIME_ERRATUM_SHA256,
        "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
        "runner_sha256": _sha256_file(SCRIPT),
        "prior_aiff_failure_receipt_sha256": LEGACY_FAILURE_SHA256,
        "provenance_activation_receipt_sha256": legacy.PROVENANCE_ACTIVATION_SHA256,
        "resource_receipt_sha256": legacy.RESOURCE_RECEIPT_SHA256,
        "instrument_id": legacy.validator.OPAQUE_INSTRUMENT_ID,
        "conversion_revision": legacy.CONVERSION_REVISION,
        "artifact_manifest_sha256": legacy.ARTIFACT_MANIFEST_SHA256,
        "artifact_bytes": 5_179_241_512,
        "artifact_file_count": 10,
        "restricted_common_schema_sha256": legacy.BASELINE_COMMON_SCHEMA_SHA256,
        "exact_output_schema_sha256": legacy.EXACT_OUTPUT_SCHEMA_SHA256,
        "prompt_sha256": legacy.PROMPT_SHA256,
        "parser_contract_sha256": legacy.PARSER_CONTRACT_SHA256,
        "prompt_content_order": "SYSTEM_THEN_AUDIO_FIVE_ORDERED_FRAMES_EXACT_USER_PROMPT",
        "tts_executable_sha256": runtime_lock["say_executable_sha256"],
        "os_build_and_voice_inventory_sha256": runtime_lock["os_build_and_voice_inventory_sha256"],
        "fixture_manifest_sha256": _sha256_bytes(_canonical(fixture_manifest)),
        "fixture_audio_sha256": fixture_manifest["audio_sha256"],
        "fixture_audio_bytes": fixture_manifest["audio_bytes"],
        "final_wav_sha256": fixture_manifest["audio_sha256"],
        "final_wav_contract": fixture_manifest["audio_contract"],
        "intermediate_container": "CAF",
        "intermediate_deleted_before_inference": fixture_manifest["intermediate_deleted_before_inference"],
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
        "historical_output_paths_overwritten": False,
        "automatic_fallback_if_revised_canary_fails": False,
    }
    return {"runtime_lock": runtime_lock, "receipt": receipt}


def _legacy_activation_projection(activation: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(activation)
    projected["schema_version"] = legacy.validator.EFFECTIVE_ACTIVATION_SCHEMA
    projected["status"] = "PROTOTYPE_INSTRUMENT_ACTIVATED_RESTRICTED_INFERENCE_NOT_RUN"
    projected.pop("caf_transport_binding", None)
    projected.pop("no_automatic_fallback_override", None)
    for field in _caf_common_receipt():
        projected.pop(field, None)
    immutable = dict(projected["immutable_bindings"])
    immutable.pop("caf_transport_amendment", None)
    immutable.pop("no_automatic_fallback_override", None)
    projected["immutable_bindings"] = immutable
    canary = dict(projected["public_canary_binding"])
    canary["receipt_schema_version"] = legacy.validator.CANARY_SCHEMA
    canary["receipt_path"] = "output/nursery_program_convergence_v1/gemma4_e4b_prototype_common_schema_public_canary_receipt_v1.json"
    projected["public_canary_binding"] = canary
    return projected


def _validate_caf_activation(activation: Mapping[str, Any], canary_sha256: str) -> None:
    if caf_validator.validate_revised_chain_receipt(activation, CAF_AMENDMENT_SHA256):
        raise CafCanaryError("E_CAF_ACTIVATION")
    caf = activation.get("caf_transport_binding", {})
    fallback = activation.get("no_automatic_fallback_override", {})
    canary = activation.get("public_canary_binding", {})
    immutable = activation.get("immutable_bindings", {})
    if (
        activation.get("schema_version") != ACTIVATION_SCHEMA
        or activation.get("status") != "PROTOTYPE_INSTRUMENT_ACTIVATED_CAF_RESTRICTED_INFERENCE_NOT_RUN"
        or not isinstance(caf, Mapping)
        or caf.get("sha256") != CAF_AMENDMENT_SHA256
        or caf.get("prior_aiff_failure_receipt_sha256") != LEGACY_FAILURE_SHA256
        or caf.get("intermediate_container") != "CAF"
        or not isinstance(fallback, Mapping)
        or fallback.get("sha256") != NO_FALLBACK_OVERRIDE_SHA256
        or fallback.get("automatic_fallback_allowed") is not False
        or not isinstance(immutable, Mapping)
        or immutable.get("caf_transport_amendment", {}).get("sha256") != CAF_AMENDMENT_SHA256
        or immutable.get("no_automatic_fallback_override", {}).get("sha256") != NO_FALLBACK_OVERRIDE_SHA256
        or not isinstance(canary, Mapping)
        or canary.get("receipt_schema_version") != CANARY_SCHEMA
        or canary.get("receipt_path") != "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/public_common_schema_canary_receipt_v1.json"
        or canary.get("receipt_sha256") != canary_sha256
    ):
        raise CafCanaryError("E_CAF_ACTIVATION")
    legacy_schema = legacy._read_json(legacy.ACTIVATION_SCHEMA)
    erratum = legacy._read_json(legacy.RUNTIME_ERRATUM)
    if legacy.validator.validate_activation(
        _legacy_activation_projection(activation),
        legacy_schema,
        legacy.ACTIVATION_SCHEMA_SHA256,
        erratum,
        legacy.RUNTIME_ERRATUM_SHA256,
        canary_sha256,
    ):
        raise CafCanaryError("E_CAF_ACTIVATION")


def _build_activation_receipt(
    receipt: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    canary_bytes: bytes,
    runtime_lock_bytes: bytes,
    resource_recheck: Mapping[str, bool],
) -> dict[str, Any]:
    _validate_caf_canary(receipt)
    projected = _legacy_canary_projection(receipt)
    base_activation = legacy._build_activation_receipt(
        projected, runtime_lock, canary_bytes, runtime_lock_bytes, resource_recheck
    )
    activation = dict(base_activation)
    activation["schema_version"] = ACTIVATION_SCHEMA
    activation["status"] = "PROTOTYPE_INSTRUMENT_ACTIVATED_CAF_RESTRICTED_INFERENCE_NOT_RUN"
    immutable = dict(activation["immutable_bindings"])
    immutable["caf_transport_amendment"] = {
        "path": "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json",
        "sha256": CAF_AMENDMENT_SHA256,
        "status": "FROZEN_PRE_CANARY_CAF_TRANSPORT_ONLY",
    }
    immutable["no_automatic_fallback_override"] = {
        "path": "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json",
        "sha256": NO_FALLBACK_OVERRIDE_SHA256,
        "status": "FROZEN_BEFORE_REVISED_GEMMA_CANARY_OR_SCIENTIFIC_ENDPOINT",
    }
    activation["immutable_bindings"] = immutable
    binding = dict(activation["public_canary_binding"])
    binding["receipt_schema_version"] = CANARY_SCHEMA
    binding["receipt_path"] = "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/public_common_schema_canary_receipt_v1.json"
    activation["public_canary_binding"] = binding
    activation["caf_transport_binding"] = {
        "path": "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json",
        "sha256": CAF_AMENDMENT_SHA256,
        "status": "FROZEN_PRE_CANARY_CAF_TRANSPORT_ONLY",
        "prior_aiff_failure_receipt_sha256": LEGACY_FAILURE_SHA256,
        "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
        "runner_sha256": _sha256_file(SCRIPT),
        "intermediate_container": "CAF",
        "final_container": "WAV",
    }
    activation["no_automatic_fallback_override"] = {
        "path": "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json",
        "sha256": NO_FALLBACK_OVERRIDE_SHA256,
        "status": "FROZEN_BEFORE_REVISED_GEMMA_CANARY_OR_SCIENTIFIC_ENDPOINT",
        "automatic_fallback_allowed": False,
        "scientific_endpoint_on_failed_gemma_allowed": False,
    }
    activation.update(_caf_common_receipt())
    _validate_caf_activation(activation, _sha256_bytes(canary_bytes))
    return activation


def _read_fd(descriptor: int) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(MAX_PIPE_BYTES + 1)
    if not payload or len(payload) > MAX_PIPE_BYTES:
        raise CafCanaryError("E_PIPE")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CafCanaryError("E_PIPE") from exc


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_PIPE_BYTES:
        raise CafCanaryError("E_PIPE")
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()


def public_execute(parser_mode: str) -> Mapping[str, Any]:
    seal = _public_preflight(parser_mode)
    backend = legacy.firewall.NetworkIsolationBackend.detect()
    legacy.firewall.verify_network_isolation(backend)
    resource_recheck = legacy._launch_resource_recheck()
    mps_lock = legacy._exclusive_public_mps_lock()
    mps_lock.__enter__()
    seal_read = seal_write = result_read = result_write = -1
    try:
        seal_read, seal_write = os.pipe()
        result_read, result_write = os.pipe()
        os.write(seal_write, _canonical(seal))
        os.close(seal_write)
        seal_write = -1
        argv = [
            str(legacy.PYTHON), "-I", str(SCRIPT), "--sandboxed",
            "--seal-fd", str(seal_read), "--result-fd", str(result_write),
        ]
        process = subprocess.Popen(
            backend.command(argv),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(legacy.firewall.scrubbed_subprocess_environment({
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
            })),
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
            raise CafCanaryError("E_SANDBOXED_CANARY")
        result = json.loads(payload)
        if not isinstance(result, Mapping) or not isinstance(result.get("receipt"), Mapping) or not isinstance(result.get("runtime_lock"), Mapping):
            raise CafCanaryError("E_CANARY_RESULT")
        receipt, runtime_lock = dict(result["receipt"]), dict(result["runtime_lock"])
        if (
            runtime_lock.get("schema_version") != RUNTIME_SCHEMA
            or runtime_lock.get("caf_transport_amendment_sha256") != CAF_AMENDMENT_SHA256
            or runtime_lock.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_OVERRIDE_SHA256
            or receipt.get("runtime_lock_sha256") != _sha256_bytes(_canonical(runtime_lock))
            or caf_validator.validate_revised_chain_receipt(runtime_lock, CAF_AMENDMENT_SHA256)
        ):
            raise CafCanaryError("E_RUNTIME_LOCK")
        if receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_CAF_PASS":
            return receipt
        _validate_caf_canary(receipt)
        runtime_lock_bytes = _json_document(runtime_lock)
        receipt["runtime_lock_file_sha256"] = _sha256_bytes(runtime_lock_bytes)
        canary_bytes = _json_document(receipt)
        activation = _build_activation_receipt(
            receipt, runtime_lock, canary_bytes, runtime_lock_bytes, resource_recheck
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


def _write_failure(code: str, *, parser_mode: str, parser_eligible: bool) -> None:
    failure = {
        **_caf_common_receipt(),
        "schema_version": FAILURE_SCHEMA,
        "status": "REVISE",
        "failure_code": code,
        "parser_mode": parser_mode,
        "parser_correction_eligible": parser_eligible,
        "caf_transport_amendment_sha256": CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": NO_FALLBACK_OVERRIDE_SHA256,
        "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
        "prior_aiff_failure_receipt_sha256": LEGACY_FAILURE_SHA256,
        "restricted_payload_accessed": False,
        "restricted_inference_run": False,
        "raw_model_output_exported": False,
        "automatic_calibration_fallback_allowed": False,
        "scientific_endpoint_allowed": False,
        "historical_output_paths_overwritten": False,
    }
    if FAILURE.exists():
        return
    _atomic_once(FAILURE, _json_document(failure))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parser-correction", action="store_true")
    parser.add_argument("--sandboxed", action="store_true")
    parser.add_argument("--seal-fd", type=int)
    parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    parser_mode = "ONE_OUTER_FENCE" if args.parser_correction else "PRIMARY"
    try:
        if args.sandboxed:
            if args.seal_fd is None or args.result_fd is None or args.parser_correction:
                return 2
            _write_fd(args.result_fd, sandboxed_execute(_read_fd(args.seal_fd)))
            return 0
        if args.seal_fd is not None or args.result_fd is not None:
            return 2
        receipt = public_execute(parser_mode)
        passed = receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_CAF_PASS"
        if not passed:
            _write_failure("E_SCHEMA_OUTPUT", parser_mode=parser_mode, parser_eligible=parser_mode == "PRIMARY")
        print("GEMMA4_PUBLIC_COMMON_SCHEMA_CAF_CANARY_PASS" if passed else "GEMMA4_PUBLIC_COMMON_SCHEMA_CAF_CANARY_REVISE")
        return 0 if passed else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, (CafCanaryError, legacy.CanaryError)) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None:
            code = "E_INTERNAL"
        if args.sandboxed and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd(args.result_fd, {"schema_version": FAILURE_SCHEMA, "failure_code": code})
        else:
            with contextlib.suppress(Exception):
                _write_failure(code, parser_mode=parser_mode, parser_eligible=False)
            print("GEMMA4_PUBLIC_COMMON_SCHEMA_CAF_CANARY_REVISE")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
