#!/usr/bin/env python3
"""One-shot Gemma-for-Qwen2 pseudo-calibration substitution coordinator.

The public phase verifies the separate activation receipt, public joint canary,
exact cached artifact, frozen amendment, and historical immutability. Only then
may a network-denied private phase discover the quarantine. That phase reuses
the completed Qwen3 outputs byte-for-byte read-only and runs Gemma on exactly
the existing 137 ordered windows. Repository output is aggregate-only.
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
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = ROOT / "scripts/nursery_pseudo_calibration.py"
HISTORICAL_WORKER = ROOT / "scripts/nursery_local_challenger_worker.py"
WORKER = ROOT / "scripts/nursery_gemma4_referential_worker.py"
VALIDATOR_SCRIPT = ROOT / "scripts/validate_prototype_opaque_instrument_v1.py"
FIREWALL = ROOT / "scripts/childlens_local_inference_firewall_v1_3.py"
AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_e4b_opaque_instrument_provenance_amendment_v1.json"
PRIOR_STOP = ROOT / "output/nursery_program_convergence_v1/gemma4_replacement_terminal_decision.json"
PROVENANCE_ACTIVATION = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_opaque_instrument_provenance_activation_receipt_v1.json"
RESOURCE_RECEIPT = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_activation_resource_receipt.json"
CANARY_CONTRACT = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_common_schema_canary_contract_v1.json"
ACTIVATION_SCHEMA = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_activation_receipt_schema_v1.json"
RUNTIME_ERRATUM = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json"
BASE_PROTOCOL = ROOT / "docs/nursery_program_convergence_v1/frozen_childlens_pseudo_calibration_protocol.json"
HISTORICAL_ACTIVATION = ROOT / "output/nursery_program_convergence_v1/instrument_activation_receipt.json"
HISTORICAL_RECEIPT = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json"
ACTIVATION = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_full_activation_receipt_v1_1.json"
PUBLIC_CANARY = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_common_schema_public_canary_receipt_v1.json"
PUBLIC_RECEIPT = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
PUBLIC_FAILURE = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_failure.json"
PUBLIC_INSTRUMENT_ROOT = Path.home() / "Library/Application Support/ChildLens Public Model Bakeoff/v1.3.1"
GEMMA_PYTHON = PUBLIC_INSTRUMENT_ROOT / "venv/bin/python3.10"
GEMMA_MODEL = PUBLIC_INSTRUMENT_ROOT / "gemma-4-e4b-it-4bit"
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
GIB = 1024**3
WINDOW_COUNT = 137
MAX_PIPE_BYTES = 2 * 1024 * 1024
QWEN3_ASR_SCHEMA = "nursery-childlens-qwen3-asr-restricted-v1"
QWEN3_VLM_SCHEMA = "nursery-childlens-qwen3-vl-restricted-v1"
GEMMA_OUTPUT_SCHEMA = "nursery-childlens-gemma4-referential-restricted-v1"
PUBLIC_SCHEMA = "nursery-prototype-opaque-gemma-calibration-receipt-v1"
QWEN3_INSTRUMENT = "qwen3_vl_8b_instruct_4bit"
GEMMA_INSTRUMENT = "gemma4_e4b_it_4bit_joint_audio_five_frame"
EXPECTED_QWEN3_WORKER_SHA256 = "e785ea475aee62fd72ab9bc064f3c9c3dcbc189b4361e996963edf9f2ab14f88"
EXPECTED_BASE_SCRIPT_SHA256 = "810ba0caf4bf5cfbb94917ecd4747c328633d21a8fb5733be28e6fc6bc5faaed"
EXPECTED_PROTOCOL_SHA256 = "db5336726f4841e592cb9c73df5ab22a0887d1bd34c7cf673146f38de9e86961"
EXPECTED_HISTORICAL_ACTIVATION_SHA256 = "c926207bbb9fd02b7cc23316cd17bd1c87298e5f64593ef7ed0a80c32271edf5"
EXPECTED_HISTORICAL_RECEIPT_SHA256 = "c8da2349663f7e49291381c42e15f8972bb3b13b0b739dad4e1221ee937dfc59"
EXPECTED_OPAQUE_AMENDMENT_SHA256 = "93874c69153daaf41d902216351666c59b12aeb656b9739ecd07d1e6e4fe42b1"
EXPECTED_PROVENANCE_ACTIVATION_SHA256 = "2f25cff5d8fc140fb7dddeebae4cccfa83880327b5d383d14b3f21f3e2adcf13"
EXPECTED_RESOURCE_RECEIPT_SHA256 = "363fda5ebc189216b2d8e9a9133cb51cea91fcc0ea9dc15fef023a028648dcd4"
EXPECTED_CANARY_CONTRACT_SHA256 = "9b894fcfd47df93824d25a467961d2c9f3398666dbe824367f2d8e1afe0635e9"
EXPECTED_ACTIVATION_SCHEMA_SHA256 = "0668d8d304a1be30e43e9df2c05469a338d647c262418a04c6d067d2e77f8b05"
EXPECTED_RUNTIME_ERRATUM_SHA256 = "b84da17121dd3235b1b6799d91c077d542e30e384f5cff5e131ec6c9755cecc1"
EXPECTED_QWEN3_ASR_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
EXPECTED_QWEN3_ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
EXPECTED_QWEN3_VLM_REVISION = "defcdea7cc7a4b0858fea563cbbce171d328e457"
ALLOWED_QWEN3_PROTOCOL_HASHES = {
    EXPECTED_PROTOCOL_SHA256,
    "fe05708f9b0933406db933561e295eadd4f4debc0931cc7da3e7195af0248ff0",
}


class GemmaSubstitutionError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GemmaSubstitutionError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_module(BASE_SCRIPT, "nursery_gemma_substitution_base")
worker = _load_module(WORKER, "nursery_gemma_substitution_worker")
replacement_validator = _load_module(VALIDATOR_SCRIPT, "nursery_gemma_substitution_validator")
firewall = _load_module(FIREWALL, "nursery_gemma_substitution_firewall")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GemmaSubstitutionError("E_CANONICAL") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, maximum: int = 64 * 1024 * 1024) -> Any:
    try:
        payload = path.read_bytes()
        if not payload or len(payload) > maximum:
            raise GemmaSubstitutionError("E_JSON")
        return json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise GemmaSubstitutionError("E_JSON") from exc


def _read_private_json(path: Path, maximum: int = 64 * 1024 * 1024) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077 != 0
        ):
            raise GemmaSubstitutionError("E_PRIVATE_FILE")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(maximum + 1)
        if not payload or len(payload) > maximum:
            raise GemmaSubstitutionError("E_PRIVATE_FILE")
        return json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise GemmaSubstitutionError("E_PRIVATE_FILE") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _private_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _private_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _atomic(path: Path, payload: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".pending-{os.getpid()}-{path.name}")
    descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, mode)
    except Exception:
        with contextlib.suppress(OSError):
            pending.unlink()
        raise


def _write_once(path: Path, value: Any) -> None:
    payload = _canonical(value)
    if path.exists():
        if not _private_file(path) or _canonical(_read_private_json(path)) != payload:
            raise GemmaSubstitutionError("E_IMMUTABLE_CONFLICT")
        return
    _atomic(path, payload)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _tree_manifest(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total = count = 0
    for member in sorted(path.rglob("*")):
        if not member.is_file() or member.is_symlink() or ".cache" in member.parts:
            continue
        relative = member.relative_to(path).as_posix()
        size = member.stat().st_size
        member_digest = _sha256_file(member)
        digest.update(f"{relative}\0{size}\0{member_digest}\n".encode())
        total += size
        count += 1
    if count == 0:
        raise GemmaSubstitutionError("E_MODEL_EMPTY")
    return digest.hexdigest(), total, count


def _historical_immutability_check() -> None:
    expected = {
        BASE_SCRIPT: EXPECTED_BASE_SCRIPT_SHA256,
        HISTORICAL_WORKER: EXPECTED_QWEN3_WORKER_SHA256,
        BASE_PROTOCOL: EXPECTED_PROTOCOL_SHA256,
        HISTORICAL_ACTIVATION: EXPECTED_HISTORICAL_ACTIVATION_SHA256,
        HISTORICAL_RECEIPT: EXPECTED_HISTORICAL_RECEIPT_SHA256,
    }
    if any(not path.is_file() or _sha256_file(path) != digest for path, digest in expected.items()):
        raise GemmaSubstitutionError("E_HISTORICAL_IMMUTABILITY")


def _issues_present(issues: Sequence[Any]) -> bool:
    return bool(issues)


def _parser_mode(activation: Mapping[str, Any]) -> str:
    canary = activation.get("public_canary_binding", {})
    if not isinstance(canary, Mapping):
        raise GemmaSubstitutionError("E_CORRECTION")
    used = canary.get("parser_correction_count")
    if used == 0:
        return "PRIMARY"
    if used == 1:
        return "ONE_OUTER_FENCE"
    raise GemmaSubstitutionError("E_CORRECTION")


def _public_gemma_seal() -> dict[str, Any]:
    _historical_immutability_check()
    amendment = _read_json(AMENDMENT)
    prior_stop = _read_json(PRIOR_STOP)
    provenance_activation = _read_json(PROVENANCE_ACTIVATION)
    resource = _read_json(RESOURCE_RECEIPT)
    canary_contract = _read_json(CANARY_CONTRACT)
    activation_schema = _read_json(ACTIVATION_SCHEMA)
    runtime_erratum = _read_json(RUNTIME_ERRATUM)
    amendment_sha = _sha256_file(AMENDMENT)
    baseline_sha = _sha256_file(BASE_PROTOCOL)
    if (
        amendment_sha != EXPECTED_OPAQUE_AMENDMENT_SHA256
        or _sha256_file(PROVENANCE_ACTIVATION) != EXPECTED_PROVENANCE_ACTIVATION_SHA256
        or _sha256_file(RESOURCE_RECEIPT) != EXPECTED_RESOURCE_RECEIPT_SHA256
        or _sha256_file(CANARY_CONTRACT) != EXPECTED_CANARY_CONTRACT_SHA256
        or _sha256_file(ACTIVATION_SCHEMA) != EXPECTED_ACTIVATION_SCHEMA_SHA256
        or _sha256_file(RUNTIME_ERRATUM) != EXPECTED_RUNTIME_ERRATUM_SHA256
    ):
        raise GemmaSubstitutionError("E_PUBLIC_BINDING")
    if _issues_present(
        replacement_validator.validate_amendment(
            amendment, prior_stop, _sha256_file(PRIOR_STOP)
        )
    ):
        raise GemmaSubstitutionError("E_AMENDMENT")
    if _issues_present(
        replacement_validator.validate_provenance_activation(
            provenance_activation, amendment_sha
        )
    ):
        raise GemmaSubstitutionError("E_PROVENANCE_ACTIVATION")
    if _issues_present(
        replacement_validator.validate_canary_contract(canary_contract, amendment_sha)
    ):
        raise GemmaSubstitutionError("E_CANARY_CONTRACT")
    if _issues_present(
        replacement_validator.validate_runtime_erratum(
            runtime_erratum,
            EXPECTED_RUNTIME_ERRATUM_SHA256,
            EXPECTED_ACTIVATION_SCHEMA_SHA256,
        )
    ):
        raise GemmaSubstitutionError("E_RUNTIME_ERRATUM")
    if (
        resource.get("status") != "VERIFIED_REUSE_RESOURCE_ADMITTED"
        or resource.get("boundary_attestations", {}).get("restricted_or_quarantine_data_accessed") is not False
    ):
        raise GemmaSubstitutionError("E_RESOURCE_RECEIPT")
    if not ACTIVATION.is_file() or not PUBLIC_CANARY.is_file():
        raise GemmaSubstitutionError("E_ACTIVATION_MISSING")
    activation = _read_json(ACTIVATION)
    canary = _read_json(PUBLIC_CANARY)
    activation_sha = _sha256_file(ACTIVATION)
    canary_sha = _sha256_file(PUBLIC_CANARY)
    if _issues_present(
        replacement_validator.validate_canary(
            canary, amendment_sha, EXPECTED_CANARY_CONTRACT_SHA256
        )
    ):
        raise GemmaSubstitutionError("E_PUBLIC_CANARY")
    if _issues_present(
        replacement_validator.validate_activation(
            activation,
            activation_schema,
            EXPECTED_ACTIVATION_SCHEMA_SHA256,
            runtime_erratum,
            EXPECTED_RUNTIME_ERRATUM_SHA256,
            canary_sha,
        )
    ):
        raise GemmaSubstitutionError("E_ACTIVATION")
    if not GEMMA_PYTHON.is_file() or not FFMPEG.is_file() or not GEMMA_MODEL.is_dir():
        raise GemmaSubstitutionError("E_RUNTIME")
    manifest, artifact_bytes, artifact_files = _tree_manifest(GEMMA_MODEL)
    artifact = activation.get("artifact_binding", {})
    runtime = activation.get("runtime_binding", {})
    qwen = activation.get("qwen3_read_only_binding", {})
    canary_binding = activation.get("public_canary_binding", {})
    if not all(isinstance(value, Mapping) for value in (artifact, runtime, qwen, canary_binding)):
        raise GemmaSubstitutionError("E_ACTIVATION")
    if (
        artifact.get("snapshot_manifest_sha256") != manifest
        or artifact.get("snapshot_bytes") != artifact_bytes
        or artifact.get("snapshot_file_count") != artifact_files
        or runtime.get("canary_adapter_sha256") != _sha256_file(WORKER)
        or runtime.get("network_denial_profile_sha256") != _sha256_file(FIREWALL)
        or canary.get("prompt_sha256") != worker.prompt_digest()
        or canary.get("exact_output_schema_sha256") != worker.exact_schema_digest()
        or canary.get("runtime_pin_erratum_sha256") != EXPECTED_RUNTIME_ERRATUM_SHA256
        or canary_binding.get("parser_correction_count") != canary.get("corrections_used")
        or qwen.get("restricted_hypotheses_recomputed") is not False
        or qwen.get("restricted_digest_exported") is not False
        or qwen.get("third_visual_model") is not False
    ):
        raise GemmaSubstitutionError("E_MODEL_SEAL")
    return {
        "schema_version": "nursery-childlens-gemma4-substitution-seal-v1",
        "script_sha256": _sha256_file(SCRIPT),
        "worker_sha256": _sha256_file(WORKER),
        "amendment_sha256": amendment_sha,
        "activation_sha256": activation_sha,
        "canary_sha256": canary_sha,
        "canary_contract_sha256": EXPECTED_CANARY_CONTRACT_SHA256,
        "runtime_pin_erratum_sha256": EXPECTED_RUNTIME_ERRATUM_SHA256,
        "provenance_activation_sha256": EXPECTED_PROVENANCE_ACTIVATION_SHA256,
        "resource_receipt_sha256": EXPECTED_RESOURCE_RECEIPT_SHA256,
        "baseline_protocol_sha256": baseline_sha,
        "model_manifest_sha256": manifest,
        "model_artifact_bytes": artifact_bytes,
        "model_artifact_file_count": artifact_files,
        "prompt_and_schema_sha256": worker.prompt_and_schema_digest(),
        "parser_mode": _parser_mode(activation),
        "qwen3_private_digest_exported": False,
    }


def _validate_seal(seal: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(seal, Mapping) or seal.get("schema_version") != "nursery-childlens-gemma4-substitution-seal-v1":
        raise GemmaSubstitutionError("E_SEAL")
    expected = _public_gemma_seal()
    if dict(seal) != expected:
        raise GemmaSubstitutionError("E_SEAL")
    return _read_json(AMENDMENT), _read_json(ACTIVATION)


def _read_fd(descriptor: int, maximum: int = MAX_PIPE_BYTES) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise GemmaSubstitutionError("E_PIPE")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise GemmaSubstitutionError("E_PIPE") from exc


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_PIPE_BYTES:
        raise GemmaSubstitutionError("E_PIPE")
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()


def _expected_windows(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        for local_index, pair in enumerate(item["candidate_windows"]):
            start, end = float(pair[0]), float(pair[1])
            result.append(
                {
                    "opaque_key": item["opaque_key"],
                    "window_index": local_index,
                    "window_start_seconds": start,
                    "window_end_seconds": end,
                }
            )
    if len(result) != WINDOW_COUNT:
        raise GemmaSubstitutionError("E_WINDOW_COUNT")
    return result


def _validate_qwen3_documents(
    items: Sequence[Mapping[str, Any]], asr: Any, vlm: Any
) -> None:
    if (
        not isinstance(asr, Mapping)
        or asr.get("schema_version") != QWEN3_ASR_SCHEMA
        or asr.get("pseudo_labels_are_ground_truth") is not False
        or asr.get("network_disabled_during_inference") is not True
        or not isinstance(asr.get("items"), list)
        or not isinstance(vlm, Mapping)
        or vlm.get("schema_version") != QWEN3_VLM_SCHEMA
        or vlm.get("pseudo_labels_are_ground_truth") is not False
        or vlm.get("network_disabled_during_inference") is not True
        or not isinstance(vlm.get("items"), list)
    ):
        raise GemmaSubstitutionError("E_QWEN3_SCHEMA")
    expected_keys = [str(item["opaque_key"]) for item in items]
    asr_keys = [row.get("opaque_key") for row in asr["items"] if isinstance(row, Mapping)]
    vlm_keys = [row.get("opaque_key") for row in vlm["items"] if isinstance(row, Mapping)]
    if asr_keys != expected_keys or vlm_keys != expected_keys or len(set(expected_keys)) != 15:
        raise GemmaSubstitutionError("E_QWEN3_ALIGNMENT")
    total = 0
    by_key = {item["opaque_key"]: item for item in items}
    for row in vlm["items"]:
        item = by_key[row["opaque_key"]]
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(item["candidate_windows"]):
            raise GemmaSubstitutionError("E_QWEN3_ALIGNMENT")
        for index, (candidate, window_pair) in enumerate(zip(candidates, item["candidate_windows"])):
            if not isinstance(candidate, Mapping):
                raise GemmaSubstitutionError("E_QWEN3_SCHEMA")
            start, end = float(window_pair[0]), float(window_pair[1])
            if (
                candidate.get("window_index") != index
                or candidate.get("window_start_seconds") != start
                or candidate.get("window_end_seconds") != end
                or type(candidate.get("schema_valid")) is not bool
                or any(candidate.get(field) not in allowed for field, allowed in worker.ALLOWED.items())
            ):
                raise GemmaSubstitutionError("E_QWEN3_SCHEMA")
            total += 1
    if total != WINDOW_COUNT:
        raise GemmaSubstitutionError("E_WINDOW_COUNT")


def _load_qwen3_outputs_read_only(
    root: Path,
    items: Sequence[Mapping[str, Any]],
    sample_digest: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, str]]:
    namespace = root / "provisional_calibration_v1"
    binding_path = namespace / "execution_binding.json"
    asr_path = namespace / "qwen3_asr_restricted.json"
    vlm_path = namespace / "qwen3_vl_restricted.json"
    if any(not _private_file(path) for path in (binding_path, asr_path, vlm_path)):
        raise GemmaSubstitutionError("E_QWEN3_PRIVATE")
    binding = _read_private_json(binding_path)
    asr_sha, vlm_sha = _sha256_file(asr_path), _sha256_file(vlm_path)
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema_version") != "nursery-childlens-pseudo-calibration-binding-v1"
        or binding.get("sample_digest") != sample_digest
        or binding.get("worker_sha256") != EXPECTED_QWEN3_WORKER_SHA256
        or binding.get("protocol_sha256") not in ALLOWED_QWEN3_PROTOCOL_HASHES
        or binding.get("asr_revision") != EXPECTED_QWEN3_ASR_REVISION
        or binding.get("aligner_revision") != EXPECTED_QWEN3_ALIGNER_REVISION
        or binding.get("vlm_revision") != EXPECTED_QWEN3_VLM_REVISION
        or binding.get("asr_output_sha256") != asr_sha
        or binding.get("vlm_output_sha256") != vlm_sha
        or binding.get("network_disabled") is not True
        or binding.get("pseudo_labels_are_ground_truth") is not False
    ):
        raise GemmaSubstitutionError("E_QWEN3_BINDING")
    asr = _read_private_json(asr_path)
    vlm = _read_private_json(vlm_path)
    _validate_qwen3_documents(items, asr, vlm)
    return asr, vlm, {
        "binding_sha256": _sha256_file(binding_path),
        "asr_output_sha256": asr_sha,
        "vlm_output_sha256": vlm_sha,
    }


def _checkpoint_expected_row(index: int, expected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "global_window_index": index,
        "opaque_key": expected["opaque_key"],
        "window_index": expected["window_index"],
        "window_start_seconds": expected["window_start_seconds"],
        "window_end_seconds": expected["window_end_seconds"],
    }


def _validate_checkpoint_row(row: Any, index: int, expected: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "global_window_index",
        "opaque_key",
        "window_index",
        "window_start_seconds",
        "window_end_seconds",
        *worker.ALLOWED.keys(),
        "schema_valid",
        "raw_response_sha256",
        "raw_response",
        "pseudo_labels_are_ground_truth",
        "human_validation",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise GemmaSubstitutionError("E_CHECKPOINT_SCHEMA")
    fixed = _checkpoint_expected_row(index, expected)
    if any(row.get(field) != value for field, value in fixed.items()):
        raise GemmaSubstitutionError("E_CHECKPOINT_ORDER")
    raw = row.get("raw_response")
    if (
        row.get("schema_version") != worker.CHECKPOINT_SCHEMA
        or type(row.get("schema_valid")) is not bool
        or row.get("pseudo_labels_are_ground_truth") is not False
        or row.get("human_validation") is not False
        or not isinstance(raw, str)
        or row.get("raw_response_sha256") != _sha256_bytes(raw.encode("utf-8"))
        or any(row.get(field) not in allowed for field, allowed in worker.ALLOWED.items())
    ):
        raise GemmaSubstitutionError("E_CHECKPOINT_SCHEMA")
    return row


def _load_checkpoint_prefix(checkpoint: Path, expected: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not _private_directory(checkpoint):
        raise GemmaSubstitutionError("E_CHECKPOINT_DIRECTORY")
    names = sorted(member.name for member in checkpoint.iterdir())
    if any(not re.fullmatch(r"[0-9]{6}\.json", name) for name in names):
        raise GemmaSubstitutionError("E_CHECKPOINT_FILES")
    expected_names = [f"{index:06d}.json" for index in range(len(names))]
    if names != expected_names or len(names) > len(expected):
        raise GemmaSubstitutionError("E_CHECKPOINT_PREFIX")
    return [
        _validate_checkpoint_row(_read_private_json(checkpoint / name), index, expected[index])
        for index, name in enumerate(names)
    ]


def _worker_environment(work: Path) -> dict[str, str]:
    environment = dict(
        firewall.scrubbed_subprocess_environment(
            {
                "OMP_NUM_THREADS": "2",
                "VECLIB_MAXIMUM_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
            }
        )
    )
    environment["TMPDIR"] = str(work)
    return environment


def _make_job(items: Sequence[Mapping[str, Any]], resume_from: int, parser_mode: str) -> dict[str, Any]:
    media_fds = [int(item["media_fd"]) for item in items]
    if any(descriptor < 3 for descriptor in media_fds):
        raise GemmaSubstitutionError("E_MEDIA_DESCRIPTOR")
    return {
        "schema_version": worker.JOB_SCHEMA,
        "sample_item_count": 15,
        "sample_total_seconds": 900,
        "candidate_window_count": WINDOW_COUNT,
        "resume_from": resume_from,
        "parser_mode": parser_mode,
        "items": [
            {
                "opaque_key": item["opaque_key"],
                "media_fd": item["media_fd"],
                "intervals": [
                    {"start_seconds": float(start), "end_seconds": float(end)}
                    for start, end in item["intervals"]
                ],
                "candidate_windows": [
                    {"start_seconds": float(start), "end_seconds": float(end)}
                    for start, end in item["candidate_windows"]
                ],
            }
            for item in items
        ],
    }


def _invoke_gemma_worker(
    items: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    work: Path,
    resume_from: int,
    parser_mode: str,
    backend: Any,
) -> None:
    media_fds: list[int] = []
    job_fd = checkpoint_fd = -1
    job_path = work / "job.json"
    try:
        worker_items = []
        for item in items:
            descriptor = os.open(item["media"], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            media_fds.append(descriptor)
            worker_items.append({**item, "media_fd": descriptor})
        job = _make_job(worker_items, resume_from, parser_mode)
        _atomic(job_path, _canonical(job))
        job_fd = os.open(job_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        checkpoint_fd = os.open(checkpoint, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        argv = [
            str(GEMMA_PYTHON),
            "-I",
            str(WORKER),
            "--job-fd",
            str(job_fd),
            "--checkpoint-dir-fd",
            str(checkpoint_fd),
            "--ffmpeg",
            str(FFMPEG),
            "--model",
            str(GEMMA_MODEL),
        ]
        completed = subprocess.run(
            backend.command(argv),
            cwd=work,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_worker_environment(work),
            pass_fds=(job_fd, checkpoint_fd, *media_fds),
            close_fds=True,
            check=False,
            timeout=6 * 60 * 60,
        )
        if completed.returncode != 0:
            raise GemmaSubstitutionError("E_GEMMA_WORKER")
    finally:
        for descriptor in (job_fd, checkpoint_fd, *media_fds):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        job_path.unlink(missing_ok=True)


def _compact_gemma_output(
    items: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {str(item["opaque_key"]): [] for item in items}
    for row in rows:
        by_key[str(row["opaque_key"])].append(
            {
                "window_index": row["window_index"],
                "window_start_seconds": row["window_start_seconds"],
                "window_end_seconds": row["window_end_seconds"],
                **{field: row[field] for field in worker.ALLOWED},
                "schema_valid": row["schema_valid"],
            }
        )
    return {
        "schema_version": GEMMA_OUTPUT_SCHEMA,
        "pseudo_labels_are_ground_truth": False,
        "human_validation": False,
        "network_disabled_during_inference": True,
        "candidate_window_count": WINDOW_COUNT,
        "items": [
            {"opaque_key": item["opaque_key"], "candidates": by_key[str(item["opaque_key"])]}
            for item in items
        ],
    }


def _validate_gemma_output(items: Sequence[Mapping[str, Any]], document: Any) -> None:
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != GEMMA_OUTPUT_SCHEMA
        or document.get("pseudo_labels_are_ground_truth") is not False
        or document.get("human_validation") is not False
        or document.get("network_disabled_during_inference") is not True
        or document.get("candidate_window_count") != WINDOW_COUNT
        or not isinstance(document.get("items"), list)
        or len(document["items"]) != 15
    ):
        raise GemmaSubstitutionError("E_GEMMA_OUTPUT")
    expected_keys = [item["opaque_key"] for item in items]
    if [row.get("opaque_key") for row in document["items"] if isinstance(row, Mapping)] != expected_keys:
        raise GemmaSubstitutionError("E_GEMMA_OUTPUT")
    total = 0
    for item, output_item in zip(items, document["items"]):
        candidates = output_item.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(item["candidate_windows"]):
            raise GemmaSubstitutionError("E_GEMMA_OUTPUT")
        for index, (candidate, pair) in enumerate(zip(candidates, item["candidate_windows"])):
            if (
                not isinstance(candidate, Mapping)
                or candidate.get("window_index") != index
                or candidate.get("window_start_seconds") != float(pair[0])
                or candidate.get("window_end_seconds") != float(pair[1])
                or type(candidate.get("schema_valid")) is not bool
                or any(candidate.get(field) not in allowed for field, allowed in worker.ALLOWED.items())
            ):
                raise GemmaSubstitutionError("E_GEMMA_OUTPUT")
            total += 1
    if total != WINDOW_COUNT:
        raise GemmaSubstitutionError("E_GEMMA_OUTPUT")


def _gemma_checkpoint_output(
    root: Path,
    items: Sequence[Mapping[str, Any]],
    sample_digest: str,
    qwen_hashes: Mapping[str, str],
    seal: Mapping[str, Any],
    activation: Mapping[str, Any],
    backend: Any,
) -> Mapping[str, Any]:
    expected_windows = _expected_windows(items)
    private_window_digest = _sha256_bytes(_canonical(expected_windows))
    expected_binding = {
        "schema_version": "nursery-childlens-gemma4-substitution-binding-v1",
        "amendment_sha256": seal["amendment_sha256"],
        "activation_sha256": seal["activation_sha256"],
        "sample_digest": sample_digest,
        "window_set_digest": private_window_digest,
        "window_count": WINDOW_COUNT,
        "qwen3_binding_sha256": qwen_hashes["binding_sha256"],
        "qwen3_asr_output_sha256": qwen_hashes["asr_output_sha256"],
        "qwen3_vlm_output_sha256": qwen_hashes["vlm_output_sha256"],
        "worker_sha256": seal["worker_sha256"],
        "model_manifest_sha256": seal["model_manifest_sha256"],
        "model_artifact_bytes": seal["model_artifact_bytes"],
        "model_artifact_file_count": seal["model_artifact_file_count"],
        "upstream_revision": activation["artifact_binding"]["current_upstream_revision"],
        "conversion_revision": activation["artifact_binding"]["conversion_revision"],
        "opaque_provenance_amendment_sha256": seal["amendment_sha256"],
        "opaque_provenance_activation_sha256": seal["provenance_activation_sha256"],
        "runtime_pin_erratum_sha256": seal["runtime_pin_erratum_sha256"],
        "runtime_lock_sha256": activation["runtime_binding"]["complete_hash_locked_dependency_file_sha256"],
        "prompt_and_schema_sha256": seal["prompt_and_schema_sha256"],
        "parser_mode": seal["parser_mode"],
        "frame_offsets_seconds": list(worker.FRAME_OFFSETS),
        "audio_rule": "EXACT_FROZEN_HALF_OPEN_WINDOW_MONO_16KHZ_PCM_S16LE",
    }
    run_key = _sha256_bytes(_canonical(expected_binding))[:32]
    base_namespace = root / "provisional_calibration_v1"
    if not _private_directory(base_namespace) or not _inside(base_namespace.resolve(strict=True), root):
        raise GemmaSubstitutionError("E_QUARANTINE")
    substitution_namespace = base_namespace / "gemma_substitution_v1"
    if substitution_namespace.exists() and not _private_directory(substitution_namespace):
        raise GemmaSubstitutionError("E_QUARANTINE")
    substitution_namespace.mkdir(mode=0o700, exist_ok=True)
    if not _private_directory(substitution_namespace) or not _inside(substitution_namespace.resolve(strict=True), root):
        raise GemmaSubstitutionError("E_QUARANTINE")
    namespace = substitution_namespace / run_key
    if namespace.exists() and not _private_directory(namespace):
        raise GemmaSubstitutionError("E_QUARANTINE")
    namespace.mkdir(mode=0o700, exist_ok=True)
    if not _private_directory(namespace) or not _inside(namespace.resolve(strict=True), root):
        raise GemmaSubstitutionError("E_QUARANTINE")
    checkpoint = namespace / "checkpoint"
    if checkpoint.exists() and not _private_directory(checkpoint):
        raise GemmaSubstitutionError("E_QUARANTINE")
    checkpoint.mkdir(mode=0o700, exist_ok=True)
    if not _private_directory(checkpoint):
        raise GemmaSubstitutionError("E_QUARANTINE")
    output = namespace / "gemma_referential_restricted.json"
    binding_path = namespace / "execution_binding.json"
    if _private_file(output) and _private_file(binding_path):
        binding = _read_private_json(binding_path)
        if (
            isinstance(binding, Mapping)
            and all(binding.get(field) == value for field, value in expected_binding.items())
            and binding.get("output_sha256") == _sha256_file(output)
            and binding.get("network_disabled") is True
            and binding.get("pseudo_labels_are_ground_truth") is False
            and binding.get("evaluation_truth") is False
        ):
            document = _read_private_json(output)
            _validate_gemma_output(items, document)
            return document
        raise GemmaSubstitutionError("E_GEMMA_BINDING_CONFLICT")
    rows = _load_checkpoint_prefix(checkpoint, expected_windows)
    if len(rows) < WINDOW_COUNT:
        scratch = Path(tempfile.mkdtemp(prefix="run-", dir=namespace))
        os.chmod(scratch, 0o700)
        try:
            _invoke_gemma_worker(items, checkpoint, scratch, len(rows), str(seal["parser_mode"]), backend)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        rows = _load_checkpoint_prefix(checkpoint, expected_windows)
    if len(rows) != WINDOW_COUNT:
        raise GemmaSubstitutionError("E_GEMMA_INCOMPLETE")
    document = _compact_gemma_output(items, rows)
    _validate_gemma_output(items, document)
    _write_once(output, document)
    binding = {
        **expected_binding,
        "output_sha256": _sha256_file(output),
        "network_disabled": True,
        "pseudo_labels_are_ground_truth": False,
        "evaluation_truth": False,
    }
    _write_once(binding_path, binding)
    return document


def _rename_range_paths(value: Any) -> Any:
    replacements = {
        "existing_whisper_qwen2_path": "qwen3asr_plus_qwen3vl_path",
        "challenger_qwen3asr_qwen3vl_path": "qwen3asr_plus_gemma4_path",
    }
    if isinstance(value, Mapping):
        return {replacements.get(str(key), str(key)): _rename_range_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_rename_range_paths(child) for child in value]
    return value


def _public_export_policy() -> dict[str, Any]:
    return {
        "minimum_cluster_k": 5,
        "complementary_suppression": True,
        "rate_rounding_step": 0.1,
        "lag_rounding_step_event_units": 0.5,
        "outward_rounding": {"rate_step": 0.1, "lag_step_event_units": 0.5},
        "raw_counts": False,
        "item_level_results": False,
        "identifiers_or_paths": False,
        "exact_timestamps": False,
        "confidence_or_raw_payload": False,
        "paths": False,
        "identifiers": False,
        "filenames": False,
        "exact_timestamps_or_intervals": False,
        "transcript_or_lexical_content": False,
        "frames_or_audio": False,
        "item_level_predictions": False,
        "confidence_or_raw_model_payload": False,
        "free_form_errors": False,
    }


def restricted_execute(seal: Any) -> dict[str, Any]:
    _, activation = _validate_seal(seal)
    backend = firewall.NetworkIsolationBackend.detect()
    firewall.verify_network_isolation(backend)
    author = _load_module(base.AUTHOR, "nursery_gemma_substitution_author_discovery")
    root = firewall.validate_quarantine_root(Path(author.discover_runtime_root()), ROOT)
    if shutil.disk_usage(root).free < 50 * GIB:
        raise GemmaSubstitutionError("E_STORAGE_FLOOR")
    items, sample_digest = base._restricted_inputs(root)
    if len(items) != 15 or sum(len(item["candidate_windows"]) for item in items) != WINDOW_COUNT:
        raise GemmaSubstitutionError("E_FROZEN_SAMPLE")
    qwen_asr, qwen_vlm, qwen_hashes = _load_qwen3_outputs_read_only(
        root, items, sample_digest
    )
    gemma_vlm = _gemma_checkpoint_output(
        root, items, sample_digest, qwen_hashes, seal, activation, backend
    )
    ranges, visual = base._calibration_ranges(items, gemma_vlm, qwen_vlm)
    ranges = _rename_range_paths(ranges)
    speech = base._speech_diagnostics(items, qwen_asr)
    internal_status, failures = base._usable_gate(ranges, visual, speech)
    passed = internal_status == "CALIBRATION_USABLE"
    status = "CALIBRATION_PASS" if passed else "CALIBRATION_REVISE"
    correction_count = activation["public_canary_binding"]["parser_correction_count"]
    receipt = {
        "schema_version": PUBLIC_SCHEMA,
        "status": status,
        "decision": "CALIBRATION_PASS_INTERNAL_PROTOTYPE" if passed else "CALIBRATION_REVISE_INTERNAL_PROTOTYPE",
        "amendment_sha256": seal["amendment_sha256"],
        "activation_receipt_sha256": seal["activation_sha256"],
        "canary_contract_sha256": seal["canary_contract_sha256"],
        "runtime_pin_erratum_sha256": seal["runtime_pin_erratum_sha256"],
        "scope": "FROZEN_PREDICTION_INDEPENDENT_15_SPEECH_MINUTE_SAMPLE",
        "referential_instrument_ids": [QWEN3_INSTRUMENT, GEMMA_INSTRUMENT],
        "instrument_paths": ["qwen3asr_plus_qwen3vl_path", "qwen3asr_plus_gemma4_path"],
        "third_visual_model_used": False,
        "qwen3_reused_read_only": True,
        "primary_item_count": 15,
        "primary_total_seconds": 900,
        "candidate_window_count": WINDOW_COUNT,
        "sample_reselected": False,
        "unchanged_contract": True,
        "speech_model_model_diagnostics": speech,
        "visual_model_model_diagnostics": visual,
        "calibration_ranges": ranges,
        "gate_failures": failures,
        "corrections_used": correction_count,
        "correction_scope": "PARSER" if correction_count else "NONE",
        "semantic_tuning_performed": False,
        "all_frozen_calibration_gates_passed": passed,
        "aggregate_only": True,
        "public_export": _public_export_policy(),
        "security": {
            "restricted_inference_network_disabled": True,
            "external_api_used": False,
            "hosted_or_cloud_content_path": False,
            "quarantine_only_pseudo_payloads": True,
            "memory_heavy_inference_serialized": True,
        },
        "ancestry": {
            "sole_empirical_anchor": "ChildLens frozen pilot",
            "AEA_empirical_ancestry": False,
            "BabyView_empirical_ancestry": False,
            "cross_corpus_pooling": False,
            "learner_ancestry_from_instruments": False,
        },
        "pseudo_labels_are_ground_truth": False,
        "human_validation_claimed": False,
        "human_evidence_available": False,
        "model_model_agreement_is_human_reliability": False,
        "simulator_oracle_only_evaluation_truth": True,
        "scientific_outcome_authorized": False,
        "scientific_outcome_run": False,
        "scientific_endpoint_opened": False,
    }
    return receipt


def _validate_public_receipt(receipt: Any) -> None:
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != PUBLIC_SCHEMA or receipt.get("status") not in {"CALIBRATION_PASS", "CALIBRATION_REVISE"}:
        raise GemmaSubstitutionError("E_PUBLIC_RECEIPT")
    passed = receipt.get("status") == "CALIBRATION_PASS"
    if (
        receipt.get("decision")
        != ("CALIBRATION_PASS_INTERNAL_PROTOTYPE" if passed else "CALIBRATION_REVISE_INTERNAL_PROTOTYPE")
        or receipt.get("all_frozen_calibration_gates_passed") is not passed
        or receipt.get("aggregate_only") is not True
        or (passed and replacement_validator.validate_calibration(receipt, _sha256_file(ACTIVATION)))
        or (not passed and not receipt.get("gate_failures"))
    ):
        raise GemmaSubstitutionError("E_PUBLIC_RECEIPT")
    encoded = _canonical(receipt).decode("utf-8").casefold()
    forbidden = (
        "/users/",
        "transcript_text",
        "participant_id",
        "media_relpath",
        "expected_media_sha256",
        "prediction_join_key",
        "raw_response",
    )
    if any(token in encoded for token in forbidden):
        raise GemmaSubstitutionError("E_PUBLIC_PRIVACY")


def public_execute() -> Mapping[str, Any]:
    seal = _public_gemma_seal()
    backend = firewall.NetworkIsolationBackend.detect()
    firewall.verify_network_isolation(backend)
    seal_read, seal_write = os.pipe()
    result_read, result_write = os.pipe()
    try:
        os.write(seal_write, _canonical(seal))
        os.close(seal_write)
        seal_write = -1
        argv = [
            sys.executable,
            str(SCRIPT),
            "--restricted",
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
            env=dict(firewall.scrubbed_subprocess_environment()),
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
        returncode = process.wait(timeout=8 * 60 * 60)
        if not payload or len(payload) > MAX_PIPE_BYTES:
            raise GemmaSubstitutionError("E_RESTRICTED_EXECUTION")
        receipt = json.loads(payload)
        if returncode != 0:
            code = receipt.get("failure_code") if isinstance(receipt, Mapping) else None
            raise GemmaSubstitutionError(
                code if isinstance(code, str) and re.fullmatch(r"E_[A-Z0-9_]+", code) else "E_RESTRICTED_EXECUTION"
            )
        _validate_public_receipt(receipt)
        if PUBLIC_RECEIPT.exists():
            raise GemmaSubstitutionError("E_PUBLIC_RECEIPT_EXISTS")
        _atomic(
            PUBLIC_RECEIPT,
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n",
        )
        return receipt
    finally:
        for descriptor in (seal_read, seal_write, result_read, result_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restricted", action="store_true")
    parser.add_argument("--seal-fd", type=int)
    parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.restricted:
            if args.seal_fd is None or args.result_fd is None:
                return 2
            receipt = restricted_execute(_read_fd(args.seal_fd))
            _write_fd(args.result_fd, receipt)
            return 0
        if args.seal_fd is not None or args.result_fd is not None:
            return 2
        receipt = public_execute()
        print("CHILDLENS_GEMMA_SUBSTITUTION_COMPLETE" if receipt.get("status") == "CALIBRATION_PASS" else "CHILDLENS_GEMMA_SUBSTITUTION_REVISE")
        return 0 if receipt.get("status") == "CALIBRATION_PASS" else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, GemmaSubstitutionError) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None:
            code = "E_INTERNAL"
        if args.restricted and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd(
                    args.result_fd,
                    {"schema_version": "nursery-childlens-gemma-substitution-failure-v1", "failure_code": code},
                )
        else:
            failure = {
                "schema_version": "nursery-childlens-gemma-substitution-failure-v1",
                "status": "REVISE",
                "failure_code": code,
                "restricted_payload_exported": False,
            }
            with contextlib.suppress(Exception):
                _atomic(PUBLIC_FAILURE, json.dumps(failure, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            print("CHILDLENS_GEMMA_SUBSTITUTION_REVISE")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
