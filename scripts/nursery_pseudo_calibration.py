#!/usr/bin/env python3
"""Bounded ChildLens model-triangulation and aggregate-only exporter.

The zero-argument public phase seals public code/models and launches a private
phase under macOS network denial. The private phase binds the already frozen
15-minute sample, reuses the historical Whisper/Qwen2 hypotheses read-only,
runs the Qwen3 ASR/VL challenger sequentially, and returns only a K-suppressed,
outward-rounded aggregate receipt through an inherited descriptor.
"""

from __future__ import annotations

import argparse
import contextlib
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
WORKER = ROOT / "scripts/nursery_local_challenger_worker.py"
QWEN2_RETRY_WORKER = ROOT / "scripts/nursery_qwen2_schema_retry_worker.py"
FIREWALL = ROOT / "scripts/childlens_local_inference_firewall_v1_3.py"
AUTHOR = ROOT / "scripts/childlens_author_audit_v1_3.py"
INITIALIZER = ROOT / "scripts/initialize_childlens_author_audit_v1_3.py"
PROTOCOL = ROOT / "docs/nursery_program_convergence_v1/frozen_childlens_pseudo_calibration_protocol.json"
SENSITIVITY = ROOT / "docs/nursery_program_convergence_v1/frozen_calibration_conditioned_sensitivity_protocol.json"
ACTIVATION = ROOT / "output/nursery_program_convergence_v1/instrument_activation_receipt.json"
ASR_CANARY = ROOT / "output/nursery_program_convergence_v1/qwen3_asr_public_canary.json"
VLM_CANARY = ROOT / "output/nursery_program_convergence_v1/qwen3_vl_public_canary.json"
QWEN2_RETRY_AMENDMENT = ROOT / "output/nursery_program_convergence_v1/qwen2_schema_retry_amendment.json"
PUBLIC_RECEIPT = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json"
PUBLIC_FAILURE = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_failure.json"
INSTRUMENT_ROOT = Path("/Users/rishisim/Library/Application Support/ChildLens Instruments/provisional-calibration-v1")
ASR_PYTHON = INSTRUMENT_ROOT / "qwen-asr-venv/bin/python"
VLM_PYTHON = INSTRUMENT_ROOT / "mlx-vlm-venv/bin/python"
ASR_MODEL = INSTRUMENT_ROOT / "models/Qwen3-ASR-1.7B"
ALIGNER_MODEL = INSTRUMENT_ROOT / "models/Qwen3-ForcedAligner-0.6B"
VLM_MODEL = INSTRUMENT_ROOT / "models/Qwen3-VL-8B-Instruct-4bit"
QWEN2_PYTHON = Path("/Users/rishisim/Library/Application Support/ChildLens Instruments/v1.3/venv/bin/python3.10")
QWEN2_MODEL = Path("/Users/rishisim/Library/Application Support/ChildLens Instruments/v1.3/models/Qwen2-VL-2B-Instruct")
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
EXPECTED_ASR_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
EXPECTED_ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
EXPECTED_VLM_REVISION = "defcdea7cc7a4b0858fea563cbbce171d328e457"
ASR_SCHEMA = "nursery-childlens-qwen3-asr-restricted-v1"
VLM_SCHEMA = "nursery-childlens-qwen3-vl-restricted-v1"
QWEN2_RETRY_SCHEMA = "nursery-childlens-qwen2-vl-schema-retry-restricted-v1"
PUBLIC_SCHEMA = "nursery-childlens-pseudo-calibration-receipt-v1"
K = 5
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 730001
GIB = 1024**3
MAX_PIPE_BYTES = 2 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FIELDS = (
    "visible_candidate",
    "null_or_irrelevant",
    "ambiguous_or_undecidable",
    "partial_or_clear_visibility",
    "noun_object_support",
    "verb_action_support",
)
CATEGORY_FIELDS = {
    "candidate_multiplicity": ("zero", "one", "two", "three_or_more"),
    "lag_event_unit": ("lead_two_plus", "lead_one", "overlap", "lag_one", "lag_two_plus"),
}


class CalibrationError(RuntimeError):
    pass


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CalibrationError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


firewall = _load(FIREWALL, "nursery_calibration_firewall")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CalibrationError("E_CANONICAL") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, maximum: int = 64 * 1024 * 1024) -> Any:
    payload = path.read_bytes()
    if not payload or len(payload) > maximum:
        raise CalibrationError("E_JSON")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CalibrationError("E_JSON") from exc


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


def _private_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) and metadata.st_uid == os.getuid() and stat.S_IMODE(metadata.st_mode) & 0o077 == 0


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
        raise CalibrationError("E_MODEL_EMPTY")
    return digest.hexdigest(), total, count


def _public_seal() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL)
    sensitivity = _read_json(SENSITIVITY)
    activation = _read_json(ACTIVATION)
    asr_canary = _read_json(ASR_CANARY)
    vlm_canary = _read_json(VLM_CANARY)
    if (
        protocol.get("status") not in {
            "AMENDED_AND_REFROZEN_V2_BEFORE_RESTRICTED_CHALLENGER_INFERENCE_AND_AGREEMENT_AGGREGATES",
            "PREDECLARED_SCHEMA_ONLY_ENGINEERING_RETRY_ACTIVATED_AFTER_SCHEMA_GATE_FAILURE",
        }
        or sensitivity.get("execution", {}).get("outcome_authorized") is not False
        or activation.get("frozen_before_restricted_challenger_inference") is not True
        or asr_canary.get("status") != "PASS"
        or vlm_canary.get("status") != "PASS"
        or asr_canary.get("asr_revision") != EXPECTED_ASR_REVISION
        or asr_canary.get("aligner_revision") != EXPECTED_ALIGNER_REVISION
        or vlm_canary.get("revision") != EXPECTED_VLM_REVISION
    ):
        raise CalibrationError("E_PUBLIC_PREFLIGHT")
    asr_manifest, asr_bytes, _ = _tree_manifest(ASR_MODEL)
    aligner_manifest, aligner_bytes, _ = _tree_manifest(ALIGNER_MODEL)
    vlm_manifest, vlm_bytes, _ = _tree_manifest(VLM_MODEL)
    if (
        asr_manifest != asr_canary.get("asr_manifest_sha256")
        or aligner_manifest != asr_canary.get("aligner_manifest_sha256")
        or vlm_manifest != vlm_canary.get("artifact_manifest_sha256")
    ):
        raise CalibrationError("E_MODEL_SEAL")
    amendment = _read_json(QWEN2_RETRY_AMENDMENT)
    if amendment.get("status") != "ACTIVATED_ONCE" or amendment.get("maximum_attempts") != 1 or amendment.get("thresholds_changed") is not False:
        raise CalibrationError("E_RETRY_AMENDMENT")
    for executable in (ASR_PYTHON, VLM_PYTHON, QWEN2_PYTHON, FFMPEG):
        if not executable.is_file():
            raise CalibrationError("E_RUNTIME")
    if not QWEN2_MODEL.is_dir():
        raise CalibrationError("E_RUNTIME")
    return {
        "schema_version": "nursery-childlens-pseudo-calibration-seal-v1",
        "protocol_sha256": _sha256_file(PROTOCOL),
        "sensitivity_sha256": _sha256_file(SENSITIVITY),
        "script_sha256": _sha256_file(SCRIPT),
        "worker_sha256": _sha256_file(WORKER),
        "qwen2_retry_worker_sha256": _sha256_file(QWEN2_RETRY_WORKER),
        "qwen2_retry_amendment_sha256": _sha256_file(QWEN2_RETRY_AMENDMENT),
        "activation_sha256": _sha256_file(ACTIVATION),
        "models": {
            "asr": {"manifest_sha256": asr_manifest, "bytes": asr_bytes},
            "aligner": {"manifest_sha256": aligner_manifest, "bytes": aligner_bytes},
            "vlm": {"manifest_sha256": vlm_manifest, "bytes": vlm_bytes},
        },
    }


def _read_fd(descriptor: int, maximum: int = MAX_PIPE_BYTES) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise CalibrationError("E_PIPE")
    return json.loads(payload)


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_PIPE_BYTES:
        raise CalibrationError("E_PIPE")
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()


def _validate_seal(seal: Any) -> None:
    if not isinstance(seal, Mapping) or seal.get("schema_version") != "nursery-childlens-pseudo-calibration-seal-v1":
        raise CalibrationError("E_SEAL")
    expected = {
        "protocol_sha256": _sha256_file(PROTOCOL),
        "sensitivity_sha256": _sha256_file(SENSITIVITY),
        "script_sha256": _sha256_file(SCRIPT),
        "worker_sha256": _sha256_file(WORKER),
        "qwen2_retry_worker_sha256": _sha256_file(QWEN2_RETRY_WORKER),
        "qwen2_retry_amendment_sha256": _sha256_file(QWEN2_RETRY_AMENDMENT),
        "activation_sha256": _sha256_file(ACTIVATION),
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise CalibrationError("E_SEAL")


def _interval_overlap(intervals: Iterable[tuple[float, float]], start: float, end: float) -> bool:
    return any(left < end and right > start for left, right in intervals)


def _restricted_inputs(root: Path) -> tuple[list[dict[str, Any]], str]:
    author = _load(AUTHOR, "nursery_calibration_author")
    initializer = _load(INITIALIZER, "nursery_calibration_initializer")
    expected_digest = initializer._load_public_sample_digest()
    matches: list[Path] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        current = Path(directory)
        if not initializer._private_directory(current):
            directories[:] = []
            continue
        directories[:] = [
            name for name in directories
            if name not in initializer.SKIP_DIRECTORIES
            and not (current / name).is_symlink()
            and initializer._private_directory(current / name)
        ]
        for name in files:
            candidate = current / name
            if candidate.suffix.lower() == ".json" and initializer._canonical_json_digest(candidate) == expected_digest:
                matches.append(candidate.resolve(strict=True))
    # A later bounded-media workflow may retain a byte-identical copy of the
    # immutable packet. Multiple matches are acceptable only because every
    # candidate is bound to the same canonical digest; no content-based choice
    # or re-selection occurs.
    if not matches:
        raise CalibrationError("E_SAMPLE_NOT_FOUND")
    packet_path = sorted(set(matches), key=lambda value: os.fspath(value))[0]
    packet = author._load_json_file(packet_path, "E_PACKET")
    validated, sample_digest, _ = author._validate_packet(packet, root)
    if sample_digest != expected_digest or len(validated) != 15:
        raise CalibrationError("E_SAMPLE")
    items: list[dict[str, Any]] = []
    total = 0.0
    for row in validated:
        digest = row["expected_media_sha256"]
        media = (root / row["media_relpath"]).resolve(strict=True)
        if not _inside(media, root) or not _private_file(media) or _sha256_file(media) != digest:
            raise CalibrationError("E_MEDIA")
        intervals = [(segment["start_ms"] / 1000.0, segment["end_ms"] / 1000.0) for segment in row["segments"]]
        total += sum(end - start for start, end in intervals)
        audio_path = root / "model_assisted_v1_3/pseudo_annotations/audio" / f"{digest}.json"
        vlm_path = root / "model_assisted_v1_3/pseudo_annotations/referential" / f"{digest}.json"
        if not _private_file(audio_path) or not _private_file(vlm_path):
            raise CalibrationError("E_BASELINE")
        audio = _read_json(audio_path)
        vlm = _read_json(vlm_path)
        if audio.get("schema_version") != "childlens-restricted-audio-pseudo-labels-v1.3.0" or vlm.get("schema_version") != "childlens-restricted-referential-pseudo-labels-v1.3.0":
            raise CalibrationError("E_BASELINE")
        windows: list[tuple[float, float]] = []
        baseline_candidates: list[dict[str, Any]] = []
        for candidate in vlm.get("candidates", []):
            if not isinstance(candidate, Mapping):
                continue
            start, end = candidate.get("window_start_seconds"), candidate.get("window_end_seconds")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or float(end) <= float(start):
                continue
            center = (float(start) + float(end)) / 2.0
            if any(left <= center < right for left, right in intervals):
                windows.append((float(start), float(end)))
                baseline_candidates.append(_baseline_common(candidate))
        if len(windows) != len(baseline_candidates):
            raise CalibrationError("E_CROSSWALK")
        baseline_segments = []
        for segment in audio.get("asr_hypotheses", []):
            if not isinstance(segment, Mapping) or not isinstance(segment.get("text"), str):
                continue
            kept = []
            for interval in segment.get("source_intervals", []):
                if not isinstance(interval, Mapping):
                    continue
                start, end = interval.get("start_seconds"), interval.get("end_seconds")
                if isinstance(start, (int, float)) and isinstance(end, (int, float)) and _interval_overlap(intervals, float(start), float(end)):
                    kept.append({"start_seconds": float(start), "end_seconds": float(end)})
            if kept:
                baseline_segments.append({"text": segment["text"], "source_intervals": kept})
        items.append(
            {
                "opaque_key": digest,
                "media": media,
                "intervals": intervals,
                "candidate_windows": windows,
                "baseline_candidates": baseline_candidates,
                "baseline_segments": baseline_segments,
                "baseline_language": str(audio.get("language_hypothesis", "")),
            }
        )
    if abs(total - 900.0) > 0.01:
        raise CalibrationError("E_SAMPLE")
    return sorted(items, key=lambda value: value["opaque_key"]), sample_digest


def _baseline_common(row: Mapping[str, Any]) -> dict[str, Any]:
    status_raw = str(row.get("referential_status", "UNUSABLE"))
    status = {
        "VISIBLE_CANDIDATE": "visible_candidate",
        "NULL_NOT_VISIBLE": "null_or_irrelevant",
        "IRRELEVANT": "null_or_irrelevant",
        "UNDECIDABLE": "undecidable",
        "UNUSABLE": "undecidable",
    }.get(status_raw, "undecidable")
    nouns = row.get("noun_object_candidates") if isinstance(row.get("noun_object_candidates"), list) else []
    verbs = row.get("verb_action_candidates") if isinstance(row.get("verb_action_candidates"), list) else []
    unique = {unicodedata.normalize("NFKC", str(value)).casefold().strip() for value in [*nouns, *verbs] if str(value).strip()}
    count = "zero" if len(unique) == 0 else "one" if len(unique) == 1 else "two" if len(unique) == 2 else "three_or_more"
    lexical = "both" if nouns and verbs else "noun_object" if nouns else "verb_action" if verbs else "neither" if status != "undecidable" else "undecidable"
    return {
        "candidate_count_bin": count if status != "undecidable" else "abstain",
        "visibility_bin": "partial" if status == "visible_candidate" else "none" if status == "null_or_irrelevant" else "abstain",
        "referential_status": status,
        "lexical_support": lexical,
        "lag_event_unit_bin": "overlap" if status == "visible_candidate" else "undecidable",
        "schema_valid": bool(row.get("machine_parse_succeeded")) and status_raw != "UNUSABLE",
    }


def _worker_environment(work: Path) -> dict[str, str]:
    environment = dict(firewall.scrubbed_subprocess_environment({
        "OMP_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
    }))
    environment.update({
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "TMPDIR": str(work),
    })
    return environment


def _invoke_worker(
    mode: str,
    items: Sequence[Mapping[str, Any]],
    output: Path,
    work: Path,
    backend: Any,
    transcript_by_key: Mapping[str, list[dict[str, Any]]] | None = None,
) -> None:
    media_fds: list[int] = []
    job_fd = output_fd = -1
    job_path = work / f"{mode}-job.json"
    pending = work / f"{mode}-output.json"
    try:
        rows = []
        for item in items:
            descriptor = os.open(item["media"], os.O_RDONLY)
            media_fds.append(descriptor)
            row: dict[str, Any] = {
                "opaque_key": item["opaque_key"],
                "media_fd": descriptor,
                "intervals": [{"start_seconds": start, "end_seconds": end} for start, end in item["intervals"]],
            }
            if mode == "vlm":
                row["candidate_windows"] = [{"start_seconds": start, "end_seconds": end} for start, end in item["candidate_windows"]]
                row["transcript_hypotheses"] = list((transcript_by_key or {}).get(item["opaque_key"], []))
            rows.append(row)
        job = {
            "schema_version": f"nursery-childlens-{mode}-job-v1",
            "sample_item_count": 15,
            "sample_total_seconds": 900,
            "items": rows,
        }
        _atomic(job_path, _canonical(job))
        job_fd = os.open(job_path, os.O_RDONLY)
        output_fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        python = ASR_PYTHON if mode == "asr" else VLM_PYTHON
        argv = [
            str(python), "-I", str(WORKER), "--mode", mode, "--job-fd", str(job_fd),
            "--output-fd", str(output_fd), "--ffmpeg", str(FFMPEG),
        ]
        if mode == "asr":
            argv.extend(["--asr-model", str(ASR_MODEL), "--aligner-model", str(ALIGNER_MODEL)])
        else:
            argv.extend(["--vlm-model", str(VLM_MODEL)])
        process = subprocess.run(
            backend.command(argv), cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=_worker_environment(work), pass_fds=(job_fd, output_fd, *media_fds),
            close_fds=True, check=False, timeout=6 * 60 * 60,
        )
        os.close(output_fd)
        output_fd = -1
        if process.returncode != 0 or not pending.is_file():
            raise CalibrationError(f"E_{mode.upper()}_WORKER")
        document = _read_json(pending)
        expected = ASR_SCHEMA if mode == "asr" else VLM_SCHEMA
        if document.get("schema_version") != expected or document.get("pseudo_labels_are_ground_truth") is not False or document.get("network_disabled_during_inference") is not True or len(document.get("items", [])) != 15:
            raise CalibrationError(f"E_{mode.upper()}_SCHEMA")
        os.replace(pending, output)
        os.chmod(output, 0o600)
    finally:
        for descriptor in (output_fd, job_fd, *media_fds):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        job_path.unlink(missing_ok=True)
        pending.unlink(missing_ok=True)


def _invoke_qwen2_retry(items: Sequence[Mapping[str, Any]], output: Path, work: Path, backend: Any) -> None:
    media_fds: list[int] = []
    job_fd = output_fd = -1
    job_path = work / "qwen2-retry-job.json"
    pending = work / "qwen2-retry-output.json"
    try:
        rows = []
        for item in items:
            descriptor = os.open(item["media"], os.O_RDONLY)
            media_fds.append(descriptor)
            rows.append({
                "opaque_key": item["opaque_key"],
                "media_fd": descriptor,
                "intervals": [{"start_seconds": start, "end_seconds": end} for start, end in item["intervals"]],
                "candidate_windows": [{"start_seconds": start, "end_seconds": end} for start, end in item["candidate_windows"]],
                "transcript_hypotheses": list(item["baseline_segments"]),
            })
        job = {
            "schema_version": "nursery-childlens-qwen2-vlm-retry-job-v1",
            "sample_item_count": 15,
            "sample_total_seconds": 900,
            "items": rows,
        }
        _atomic(job_path, _canonical(job))
        job_fd = os.open(job_path, os.O_RDONLY)
        output_fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        argv = [
            str(QWEN2_PYTHON), "-I", str(QWEN2_RETRY_WORKER),
            "--job-fd", str(job_fd), "--output-fd", str(output_fd),
            "--qwen-model", str(QWEN2_MODEL),
        ]
        process = subprocess.run(
            backend.command(argv), cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=_worker_environment(work),
            pass_fds=(job_fd, output_fd, *media_fds), close_fds=True, check=False,
            timeout=6 * 60 * 60,
        )
        os.close(output_fd)
        output_fd = -1
        if process.returncode != 0 or not pending.is_file():
            failure: Any = None
            if pending.is_file():
                try:
                    failure = _read_json(pending)
                except Exception:
                    failure = None
            code = failure.get("failure_code") if isinstance(failure, Mapping) else None
            if isinstance(code, str) and re.fullmatch(r"E_[A-Z0-9_]+", code):
                raise CalibrationError(code)
            raise CalibrationError("E_QWEN2_RETRY_WORKER")
        document = _read_json(pending)
        if (
            document.get("schema_version") != QWEN2_RETRY_SCHEMA
            or document.get("pseudo_labels_are_ground_truth") is not False
            or document.get("network_disabled_during_inference") is not True
            or document.get("schema_only_engineering_retry") is not True
            or len(document.get("items", [])) != 15
        ):
            raise CalibrationError("E_QWEN2_RETRY_SCHEMA")
        os.replace(pending, output)
        os.chmod(output, 0o600)
    finally:
        for descriptor in (output_fd, job_fd, *media_fds):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        job_path.unlink(missing_ok=True)
        pending.unlink(missing_ok=True)


def _qwen2_retry_output(root: Path, items: Sequence[Mapping[str, Any]], sample_digest: str, backend: Any) -> Mapping[str, Any]:
    namespace = root / "provisional_calibration_v1"
    namespace.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(namespace, 0o700)
    output = namespace / "qwen2_schema_retry_restricted.json"
    binding_path = namespace / "qwen2_schema_retry_binding.json"
    expected = {
        "schema_version": "nursery-childlens-qwen2-schema-retry-binding-v1",
        "sample_digest": sample_digest,
        "worker_sha256": _sha256_file(QWEN2_RETRY_WORKER),
        "amendment_sha256": _sha256_file(QWEN2_RETRY_AMENDMENT),
        "maximum_attempts": 1,
    }
    reusable = False
    if _private_file(output) and _private_file(binding_path):
        binding = _read_json(binding_path)
        reusable = all(binding.get(key) == value for key, value in expected.items()) and binding.get("output_sha256") == _sha256_file(output)
    if not reusable:
        scratch = Path(tempfile.mkdtemp(prefix="qwen2-retry-", dir=namespace))
        os.chmod(scratch, 0o700)
        try:
            _invoke_qwen2_retry(items, output, scratch, backend)
            binding = dict(expected)
            binding.update({
                "output_sha256": _sha256_file(output),
                "network_disabled": True,
                "pseudo_labels_are_ground_truth": False,
            })
            _atomic(binding_path, _canonical(binding))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    return _read_json(output)


def _challenger_outputs(root: Path, items: Sequence[Mapping[str, Any]], sample_digest: str, backend: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    namespace = root / "provisional_calibration_v1"
    namespace.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(namespace, 0o700)
    if not _inside(namespace.resolve(strict=True), root) or namespace.is_symlink():
        raise CalibrationError("E_QUARANTINE")
    scratch = Path(tempfile.mkdtemp(prefix="run-", dir=namespace))
    os.chmod(scratch, 0o700)
    asr_output = namespace / "qwen3_asr_restricted.json"
    vlm_output = namespace / "qwen3_vl_restricted.json"
    binding_path = namespace / "execution_binding.json"
    expected_binding = {
        "schema_version": "nursery-childlens-pseudo-calibration-binding-v1",
        "sample_digest": sample_digest,
        "protocol_sha256": _sha256_file(PROTOCOL),
        "worker_sha256": _sha256_file(WORKER),
        "asr_revision": EXPECTED_ASR_REVISION,
        "aligner_revision": EXPECTED_ALIGNER_REVISION,
        "vlm_revision": EXPECTED_VLM_REVISION,
    }
    try:
        reusable = False
        if _private_file(binding_path) and _private_file(asr_output) and _private_file(vlm_output):
            existing = _read_json(binding_path)
            stable_keys = tuple(key for key in expected_binding if key != "protocol_sha256")
            allowed_protocols = {
                expected_binding["protocol_sha256"],
                "fe05708f9b0933406db933561e295eadd4f4debc0931cc7da3e7195af0248ff0",
            }
            reusable = (
                all(existing.get(key) == expected_binding[key] for key in stable_keys)
                and existing.get("protocol_sha256") in allowed_protocols
                and existing.get("asr_output_sha256") == _sha256_file(asr_output)
                and existing.get("vlm_output_sha256") == _sha256_file(vlm_output)
            )
        if not reusable:
            _invoke_worker("asr", items, asr_output, scratch, backend)
            asr_document = _read_json(asr_output)
            transcript_by_key = {
                row["opaque_key"]: row.get("utterance_hypotheses", [])
                for row in asr_document["items"]
                if isinstance(row, Mapping) and isinstance(row.get("opaque_key"), str)
            }
            _invoke_worker("vlm", items, vlm_output, scratch, backend, transcript_by_key)
            binding = dict(expected_binding)
            binding.update({
                "asr_output_sha256": _sha256_file(asr_output),
                "vlm_output_sha256": _sha256_file(vlm_output),
                "network_disabled": True,
                "pseudo_labels_are_ground_truth": False,
            })
            _atomic(binding_path, _canonical(binding))
        asr_document = _read_json(asr_output)
        vlm_document = _read_json(vlm_output)
        return asr_document, vlm_document
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(character if character.isalnum() or character.isspace() else " " for character in text)
    return " ".join(text.split())


def _edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _flatten_segments(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        text = str(row.get("text", ""))
        for interval in row.get("source_intervals", []):
            if isinstance(interval, Mapping) and isinstance(interval.get("start_seconds"), (int, float)) and isinstance(interval.get("end_seconds"), (int, float)):
                result.append({"start": float(interval["start_seconds"]), "end": float(interval["end_seconds"]), "text": text})
    return sorted(result, key=lambda value: (value["start"], value["end"], value["text"]))


def _match_boundaries(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    # Successive shortest augmenting paths: maximum cardinality first, then
    # minimum total collar error with stable index tie-breaking.
    n, m = len(left), len(right)
    source, left0, right0, sink = 0, 1, 1 + n, 1 + n + m
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]
    def add(u: int, v: int, cap: int, cost: int) -> None:
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])
    for i in range(n):
        add(source, left0 + i, 1, 0)
    for j in range(m):
        add(right0 + j, sink, 1, 0)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            onset = abs(float(a["start"]) - float(b["start"]))
            offset = abs(float(a["end"]) - float(b["end"]))
            if onset <= 0.5 and offset <= 0.5:
                cost = round((onset + offset) * 1_000_000) * 10_000 + i * 100 + j
                add(left0 + i, right0 + j, 1, cost)
    while True:
        distance = [10**30] * len(graph)
        parent: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for u, edges in enumerate(graph):
                if distance[u] >= 10**30:
                    continue
                for edge_index, edge in enumerate(edges):
                    v, cap, cost, _ = edge
                    if cap and distance[u] + cost < distance[v]:
                        distance[v] = distance[u] + cost
                        parent[v] = (u, edge_index)
                        changed = True
            if not changed:
                break
        if parent[sink] is None:
            break
        node = sink
        while node != source:
            u, edge_index = parent[node]  # type: ignore[misc]
            edge = graph[u][edge_index]
            edge[1] -= 1
            graph[node][edge[3]][1] += 1
            node = u
    matches = []
    for i in range(n):
        for edge in graph[left0 + i]:
            v, cap, _, _ = edge
            if right0 <= v < right0 + m and cap == 0:
                matches.append((i, v - right0))
    return sorted(matches)


def _round_range(value: float, step: float = 0.1, clamp: tuple[float, float] = (0.0, 1.0)) -> list[float]:
    decimal = Decimal(str(value))
    quantum = Decimal(str(step))
    lower = (decimal / quantum).to_integral_value(rounding=ROUND_FLOOR) * quantum
    upper = (decimal / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum
    return [float(max(lower, Decimal(str(clamp[0])))), float(min(upper, Decimal(str(clamp[1]))))]


def _cluster_interval(values: Mapping[str, Sequence[bool | None]]) -> dict[str, Any]:
    clusters = sorted(values)
    contributing = [key for key in clusters if any(value is not None for value in values[key])]
    positive_clusters = sum(any(value is True for value in values[key]) for key in contributing)
    negative_clusters = sum(any(value is False for value in values[key]) for key in contributing)
    if len(contributing) < K or positive_clusters < K or negative_clusters < K:
        return {"status": "SUPPRESSED_K5"}
    def rate(keys: Sequence[str]) -> float:
        flattened = [value for key in keys for value in values[key] if value is not None]
        return sum(value is True for value in flattened) / len(flattened) if flattened else 0.0
    rng = random.Random(BOOTSTRAP_SEED)
    boot = sorted(rate([rng.choice(contributing) for _ in contributing]) for _ in range(BOOTSTRAPS))
    low = boot[max(0, math.floor(0.025 * (BOOTSTRAPS - 1)))]
    high = boot[min(BOOTSTRAPS - 1, math.ceil(0.975 * (BOOTSTRAPS - 1)))]
    lower = math.floor(low * 10) / 10
    upper = math.ceil(high * 10) / 10
    return {"status": "PUBLISHED", "interval": [max(0.0, lower), min(1.0, upper)]}


def _binary(candidate: Mapping[str, Any], field: str) -> bool | None:
    if field == "visible_candidate":
        value = candidate.get("referential_status")
        return None if value == "undecidable" else value == "visible_candidate"
    if field == "null_or_irrelevant":
        value = candidate.get("referential_status")
        return None if value == "undecidable" else value == "null_or_irrelevant"
    if field == "ambiguous_or_undecidable":
        return candidate.get("referential_status") in {"ambiguous", "undecidable"}
    if field == "partial_or_clear_visibility":
        value = candidate.get("visibility_bin")
        return None if value == "abstain" else value in {"partial", "clear"}
    if field == "noun_object_support":
        value = candidate.get("lexical_support")
        return None if value == "undecidable" else value in {"noun_object", "both"}
    if field == "verb_action_support":
        value = candidate.get("lexical_support")
        return None if value == "undecidable" else value in {"verb_action", "both"}
    raise CalibrationError("E_FIELD")


def _calibration_ranges(
    items: Sequence[Mapping[str, Any]], challenger_vlm: Mapping[str, Any], baseline_vlm: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    challenger_map = {row["opaque_key"]: row.get("candidates", []) for row in challenger_vlm.get("items", []) if isinstance(row, Mapping)}
    baseline_map = (
        {row["opaque_key"]: row.get("candidates", []) for row in baseline_vlm.get("items", []) if isinstance(row, Mapping)}
        if isinstance(baseline_vlm, Mapping)
        else {item["opaque_key"]: list(item["baseline_candidates"]) for item in items}
    )
    paths: dict[str, dict[str, list[Mapping[str, Any]]]] = {"existing_whisper_qwen2_path": {}, "challenger_qwen3asr_qwen3vl_path": {}}
    for item in items:
        key = item["opaque_key"]
        challenger = challenger_map.get(key, [])
        baseline = baseline_map.get(key, [])
        if len(challenger) != len(baseline) or len(baseline) != len(item["candidate_windows"]):
            raise CalibrationError("E_VISUAL_ALIGNMENT")
        paths["existing_whisper_qwen2_path"][key] = list(baseline)
        paths["challenger_qwen3asr_qwen3vl_path"][key] = list(challenger)
    result: dict[str, Any] = {}
    for field in FIELDS:
        model_estimates: dict[str, Any] = {}
        by_model: dict[str, dict[str, list[bool | None]]] = {}
        for path_name, item_map in paths.items():
            values = {key: [_binary(candidate, field) for candidate in rows] for key, rows in item_map.items()}
            by_model[path_name] = values
            model_estimates[path_name] = _cluster_interval(values)
        intersection: dict[str, list[bool | None]] = {}
        union: dict[str, list[bool | None]] = {}
        for key in sorted(paths["existing_whisper_qwen2_path"]):
            left_values = by_model["existing_whisper_qwen2_path"][key]
            right_values = by_model["challenger_qwen3asr_qwen3vl_path"][key]
            intersection[key] = [None if a is None or b is None else a and b for a, b in zip(left_values, right_values)]
            union[key] = [None if a is None or b is None else a or b for a, b in zip(left_values, right_values)]
        components = {**model_estimates, "model_intersection": _cluster_interval(intersection), "model_union": _cluster_interval(union)}
        intervals = [value["interval"] for value in components.values() if value.get("status") == "PUBLISHED"]
        envelope = {"status": "SUPPRESSED_K5"} if not intervals else {"status": "PUBLISHED", "interval": [min(value[0] for value in intervals), max(value[1] for value in intervals)]}
        result[field] = {**components, "conservative_envelope": envelope}
    for field, bins in CATEGORY_FIELDS.items():
        source_key = "candidate_count_bin" if field == "candidate_multiplicity" else "lag_event_unit_bin"
        result[field] = {}
        for bin_value in bins:
            model_values: dict[str, dict[str, list[bool | None]]] = {}
            components: dict[str, Any] = {}
            for path_name, item_map in paths.items():
                values: dict[str, list[bool | None]] = {}
                for key, rows in item_map.items():
                    values[key] = [None if candidate.get(source_key) in {"abstain", "undecidable", None} else candidate.get(source_key) == bin_value for candidate in rows]
                model_values[path_name] = values
                components[path_name] = _cluster_interval(values)
            intersection: dict[str, list[bool | None]] = {}
            union: dict[str, list[bool | None]] = {}
            for key in sorted(model_values["existing_whisper_qwen2_path"]):
                left = model_values["existing_whisper_qwen2_path"][key]
                right = model_values["challenger_qwen3asr_qwen3vl_path"][key]
                intersection[key] = [None if a is None or b is None else a and b for a, b in zip(left, right)]
                union[key] = [None if a is None or b is None else a or b for a, b in zip(left, right)]
            components["model_intersection"] = _cluster_interval(intersection)
            components["model_union"] = _cluster_interval(union)
            intervals = [value["interval"] for value in components.values() if value.get("status") == "PUBLISHED"]
            components["conservative_envelope"] = {"status": "SUPPRESSED_K5"} if not intervals else {"status": "PUBLISHED", "interval": [min(value[0] for value in intervals), max(value[1] for value in intervals)]}
            result[field][bin_value] = components
    diagnostics: dict[str, Any] = {}
    total_windows = sum(len(item["baseline_candidates"]) for item in items)
    baseline_valid = sum(bool(row.get("schema_valid")) for rows in baseline_map.values() for row in rows)
    challenger_rows = [row for rows in challenger_map.values() for row in rows]
    challenger_valid = sum(bool(row.get("schema_valid")) for row in challenger_rows)
    diagnostics["candidate_window_count"] = total_windows if total_windows >= K else "SUPPRESSED_K5"
    diagnostics["visual_item_coverage_interval"] = _round_range(sum(bool(item["candidate_windows"]) for item in items) / 15)
    diagnostics["existing_schema_validity_interval"] = _round_range(baseline_valid / total_windows if total_windows else 0.0)
    diagnostics["challenger_schema_validity_interval"] = _round_range(challenger_valid / total_windows if total_windows else 0.0)
    field_agreement: dict[str, Any] = {}
    for field in ("referential_status", "candidate_count_bin", "visibility_bin", "lexical_support", "lag_event_unit_bin"):
        comparisons = []
        for item in items:
            for left, right in zip(baseline_map[item["opaque_key"]], challenger_map[item["opaque_key"]]):
                a, b = left.get(field), right.get(field)
                if a not in {None, "abstain", "undecidable"} and b not in {None, "abstain", "undecidable"}:
                    comparisons.append(a == b)
        field_agreement[field] = "SUPPRESSED_K5" if len(comparisons) < K else {"status": "MODEL_MODEL_DIAGNOSTIC_ONLY", "interval": _round_range(sum(comparisons) / len(comparisons))}
    diagnostics["agreement"] = field_agreement
    return result, diagnostics


def _speech_diagnostics(items: Sequence[Mapping[str, Any]], challenger_asr: Mapping[str, Any]) -> dict[str, Any]:
    challenger_map = {row["opaque_key"]: row for row in challenger_asr.get("items", []) if isinstance(row, Mapping)}
    totals = {"left": 0, "right": 0, "matched": 0}
    wers: list[float] = []
    cers: list[float] = []
    language_agreement: list[bool] = []
    nonempty_items = 0
    for item in items:
        key = item["opaque_key"]
        row = challenger_map.get(key)
        if not isinstance(row, Mapping):
            raise CalibrationError("E_ASR_ALIGNMENT")
        left = _flatten_segments(item["baseline_segments"])
        right = _flatten_segments(row.get("utterance_hypotheses", []))
        pairs = _match_boundaries(left, right)
        totals["left"] += len(left)
        totals["right"] += len(right)
        totals["matched"] += len(pairs)
        if str(row.get("transcript_hypothesis", "")).strip():
            nonempty_items += 1
        baseline_language = item["baseline_language"].casefold()
        challenger_language = str(row.get("language_hypothesis", "")).casefold()
        if baseline_language and challenger_language:
            german = {"de", "german", "deutsch", "german (de)"}
            language_agreement.append((baseline_language in german) == (challenger_language in german))
        for left_index, right_index in pairs:
            a = _normalize(left[left_index]["text"])
            b = _normalize(right[right_index]["text"])
            if not a or not b:
                continue
            a_words, b_words = a.split(), b.split()
            distance_words = _edit_distance(a_words, b_words)
            wers.append(0.5 * (distance_words / max(1, len(a_words)) + distance_words / max(1, len(b_words))))
            distance_chars = _edit_distance(list(a), list(b))
            cers.append(0.5 * (distance_chars / max(1, len(a)) + distance_chars / max(1, len(b))))
    precision = totals["matched"] / totals["right"] if totals["right"] else 0.0
    recall = totals["matched"] / totals["left"] if totals["left"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "status": "MODEL_MODEL_DIAGNOSTIC_NOT_HUMAN_VALIDATED",
        "boundary_precision_interval": _round_range(precision),
        "boundary_recall_interval": _round_range(recall),
        "boundary_f1_interval": _round_range(f1),
        "symmetric_wer_interval": "SUPPRESSED_K5" if len(wers) < K else _round_range(sum(wers) / len(wers), 0.1, (0.0, 5.0)),
        "symmetric_cer_interval": "SUPPRESSED_K5" if len(cers) < K else _round_range(sum(cers) / len(cers), 0.1, (0.0, 5.0)),
        "language_id_agreement_interval": "SUPPRESSED_K5" if len(language_agreement) < K else _round_range(sum(language_agreement) / len(language_agreement)),
        "challenger_nonempty_item_coverage_interval": _round_range(nonempty_items / 15),
        "reference_path_declared": False,
        "translation_used_as_truth": False,
    }


def _usable_gate(ranges: Mapping[str, Any], visual: Mapping[str, Any], speech: Mapping[str, Any]) -> tuple[str, list[str]]:
    failures = []
    if visual.get("visual_item_coverage_interval", [0])[0] < 0.8:
        failures.append("VISUAL_ITEM_COVERAGE")
    if visual.get("existing_schema_validity_interval", [0])[0] < 0.9:
        failures.append("EXISTING_SCHEMA_VALIDITY")
    if visual.get("challenger_schema_validity_interval", [0])[0] < 0.9:
        failures.append("CHALLENGER_SCHEMA_VALIDITY")
    if speech.get("challenger_nonempty_item_coverage_interval", [0])[0] < 0.8:
        failures.append("ASR_ITEM_COVERAGE")
    for field in FIELDS:
        if ranges.get(field, {}).get("conservative_envelope", {}).get("status") != "PUBLISHED":
            failures.append(f"ENVELOPE_{field.upper()}")
    for category in CATEGORY_FIELDS:
        published = sum(value.get("conservative_envelope", {}).get("status") == "PUBLISHED" for value in ranges.get(category, {}).values())
        if published == 0:
            failures.append(f"ENVELOPE_{category.upper()}")
    return ("CALIBRATION_USABLE" if not failures else "CALIBRATION_REVISE", sorted(set(failures)))


def restricted_execute(seal: Any) -> dict[str, Any]:
    _validate_seal(seal)
    backend = firewall.NetworkIsolationBackend.detect()
    firewall.verify_network_isolation(backend)
    author = _load(AUTHOR, "nursery_calibration_author_discovery")
    root = author.discover_runtime_root()
    root = firewall.validate_quarantine_root(Path(root), ROOT)
    if shutil.disk_usage(root).free < 50 * GIB:
        raise CalibrationError("E_STORAGE_FLOOR")
    items, sample_digest = _restricted_inputs(root)
    challenger_asr, challenger_vlm = _challenger_outputs(root, items, sample_digest, backend)
    qwen2_retry_vlm = _qwen2_retry_output(root, items, sample_digest, backend)
    ranges, visual = _calibration_ranges(items, challenger_vlm, qwen2_retry_vlm)
    speech = _speech_diagnostics(items, challenger_asr)
    status, failures = _usable_gate(ranges, visual, speech)
    receipt = {
        "schema_version": PUBLIC_SCHEMA,
        "status": status,
        "scope": "FROZEN_PREDICTION_INDEPENDENT_15_SPEECH_MINUTE_SAMPLE",
        "primary_item_count": 15,
        "primary_total_seconds": 900,
        "instrument_paths": ["existing_whisper_qwen2_path", "challenger_qwen3asr_qwen3vl_path"],
        "engineering_retry": {
            "qwen2_schema_only_retry_used": True,
            "attempts": 1,
            "same_checkpoint_windows_and_whisper_hypotheses": True,
            "semantic_or_agreement_directed_tuning": False
        },
        "speech_model_model_diagnostics": speech,
        "visual_model_model_diagnostics": visual,
        "calibration_ranges": ranges,
        "gate_failures": failures,
        "public_export": {
            "minimum_cluster_k": 5,
            "complementary_suppression": True,
            "outward_rounding": {"rate_step": 0.1, "lag_step_event_units": 0.5},
            "raw_counts": False,
            "paths": False,
            "identifiers": False,
            "filenames": False,
            "exact_timestamps_or_intervals": False,
            "transcript_or_lexical_content": False,
            "frames_or_audio": False,
            "item_level_predictions": False,
            "confidence_or_raw_model_payload": False,
            "free_form_errors": False,
        },
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
        "human_evidence_available": False,
        "model_model_agreement_is_human_reliability": False,
        "simulator_oracle_only_evaluation_truth": True,
        "scientific_outcome_run": False,
    }
    return receipt


def _validate_public_receipt(receipt: Any) -> None:
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != PUBLIC_SCHEMA or receipt.get("status") not in {"CALIBRATION_USABLE", "CALIBRATION_REVISE"}:
        raise CalibrationError("E_PUBLIC_RECEIPT")
    encoded = _canonical(receipt).decode("utf-8").casefold()
    # Policy keys such as ``exact_timestamps_or_intervals: false`` are required
    # and must not trip the content sentinel. Match only payload-bearing names
    # and absolute-path syntax here; the explicit policy checks below validate
    # every required false field.
    forbidden = ("/users/", "transcript_text", "participant_id", "media_relpath", "expected_media_sha256", "prediction_join_key")
    if any(token in encoded for token in forbidden):
        raise CalibrationError("E_PUBLIC_PRIVACY")
    public = receipt.get("public_export", {})
    if public.get("minimum_cluster_k") != 5 or public.get("complementary_suppression") is not True or any(public.get(key) is not False for key in ("raw_counts", "paths", "identifiers", "filenames", "exact_timestamps_or_intervals", "transcript_or_lexical_content", "frames_or_audio", "item_level_predictions", "confidence_or_raw_model_payload", "free_form_errors")):
        raise CalibrationError("E_PUBLIC_PRIVACY")
    if receipt.get("pseudo_labels_are_ground_truth") is not False or receipt.get("human_evidence_available") is not False or receipt.get("scientific_outcome_run") is not False:
        raise CalibrationError("E_PUBLIC_SEMANTICS")


def public_execute() -> Mapping[str, Any]:
    seal = _public_seal()
    backend = firewall.NetworkIsolationBackend.detect()
    firewall.verify_network_isolation(backend)
    seal_read, seal_write = os.pipe()
    result_read, result_write = os.pipe()
    try:
        os.write(seal_write, _canonical(seal))
        os.close(seal_write)
        seal_write = -1
        argv = [sys.executable, str(SCRIPT), "--restricted", "--seal-fd", str(seal_read), "--result-fd", str(result_write)]
        process = subprocess.Popen(
            backend.command(argv), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=dict(firewall.scrubbed_subprocess_environment()),
            pass_fds=(seal_read, result_write), close_fds=True,
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
            raise CalibrationError("E_RESTRICTED_EXECUTION")
        receipt = json.loads(payload)
        if returncode != 0:
            code = receipt.get("failure_code") if isinstance(receipt, Mapping) else None
            raise CalibrationError(code if isinstance(code, str) and re.fullmatch(r"E_[A-Z0-9_]+", code) else "E_RESTRICTED_EXECUTION")
        _validate_public_receipt(receipt)
        _atomic(PUBLIC_RECEIPT, json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
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
        print("CHILDLENS_PSEUDO_CALIBRATION_COMPLETE" if receipt.get("status") == "CALIBRATION_USABLE" else "CHILDLENS_PSEUDO_CALIBRATION_REVISE")
        return 0 if receipt.get("status") == "CALIBRATION_USABLE" else 3
    except Exception as exc:
        code = (
            exc.args[0]
            if isinstance(exc, CalibrationError) and exc.args
            else getattr(exc, "code", f"E_{type(exc).__name__.upper()}")
        )
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None:
            code = "E_INTERNAL"
        if args.restricted and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd(args.result_fd, {"schema_version": "nursery-childlens-pseudo-calibration-failure-v1", "failure_code": code})
        else:
            failure = {
                "schema_version": "nursery-childlens-pseudo-calibration-failure-v1",
                "status": "REVISE",
                "failure_code": code,
                "restricted_payload_exported": False,
            }
            with contextlib.suppress(Exception):
                _atomic(PUBLIC_FAILURE, json.dumps(failure, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            print("CHILDLENS_PSEUDO_CALIBRATION_REVISE")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
