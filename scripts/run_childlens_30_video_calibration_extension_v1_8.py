#!/usr/bin/env python3
"""Run the frozen 15-item ChildLens extension and aggregate all 30 items.

The public phase validates only public bindings and launches this same program
under OS network denial. The restricted phase discovers the owner-private
quarantine, runs Whisper, Qwen3-ASR/Qwen3-VL, and Gemma serially on only the
new 15-item extension, reuses the original 15-item Qwen3/Gemma outputs
read-only, and emits one aggregate-only K=5 receipt through an inherited file
descriptor. Restricted text, frames, audio, identifiers, paths, timestamps,
and item-level predictions never enter the repository result.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
AUTHOR = ROOT / "scripts/childlens_author_audit_v1_3.py"
HUMAN_V12 = ROOT / "scripts/childlens_human_validation_v1_2.py"
FIREWALL = ROOT / "scripts/childlens_local_inference_firewall_v1_3.py"
BASE = ROOT / "scripts/nursery_pseudo_calibration.py"
GEMMA_COORDINATOR = ROOT / "scripts/nursery_gemma_substitution_calibration.py"
QWEN_WORKER = ROOT / "scripts/nursery_local_challenger_worker_extension_v1_8.py"
GEMMA_WORKER = ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py"
GEMMA_EXTENSION_WORKER = (
    ROOT / "scripts/nursery_gemma4_referential_worker_extension_v1_8.py"
)
WHISPER_ADAPTER = ROOT / "scripts/childlens_local_pseudo_adapter_v1_3.py"
ACQUISITION_RECEIPT = (
    ROOT
    / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "acquisition_and_clipping_receipt.json"
)
SELECTION_RECEIPT = (
    ROOT
    / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "selection_and_admission_receipt.json"
)
PROTOCOL = (
    ROOT
    / "docs/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "frozen_extension_protocol_v1_8.json"
)
PREDECESSOR = (
    ROOT
    / "output/nursery_program_convergence_v1"
    / "childlens_pseudo_calibration_gemma_substitution_receipt.json"
)
QWEN_ACTIVATION = (
    ROOT / "output/nursery_program_convergence_v1/instrument_activation_receipt.json"
)
QWEN_ASR_CANARY = (
    ROOT / "output/nursery_program_convergence_v1/qwen3_asr_public_canary.json"
)
QWEN_VLM_CANARY = (
    ROOT / "output/nursery_program_convergence_v1/qwen3_vl_public_canary.json"
)
GEMMA_ACTIVATION = (
    ROOT
    / "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5"
    / "full_activation_receipt_v1_5.json"
)
PUBLIC_RECEIPT = (
    ROOT
    / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "combined_calibration_receipt.json"
)
PUBLIC_FAILURE = (
    ROOT
    / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "combined_calibration_failure_v1_8_9.json"
)

INSTRUMENT_V13 = Path(
    "/Users/rishisim/Library/Application Support/ChildLens Instruments/v1.3"
)
WHISPER_PYTHON = INSTRUMENT_V13 / "venv/bin/python3.10"
WHISPER_CLI = (
    INSTRUMENT_V13 / ".worktrees/whisper.cpp-v1.9.1/build/bin/whisper-cli"
)
WHISPER_MODEL = INSTRUMENT_V13 / "models/ggml-large-v3-turbo.bin"
SILERO_MODEL = INSTRUMENT_V13 / "models/silero-vad-6.2.1/silero_vad.onnx"
QWEN_ROOT = Path(
    "/Users/rishisim/Library/Application Support/ChildLens Instruments/provisional-calibration-v1"
)
QWEN_PYTHON = QWEN_ROOT / "mlx-vlm-venv/bin/python"
QWEN_ASR_PYTHON = QWEN_ROOT / "qwen-asr-venv/bin/python"
QWEN_ASR_MODEL = QWEN_ROOT / "models/Qwen3-ASR-1.7B"
QWEN_ALIGNER_MODEL = QWEN_ROOT / "models/Qwen3-ForcedAligner-0.6B"
QWEN_VLM_MODEL = QWEN_ROOT / "models/Qwen3-VL-8B-Instruct-4bit"
GEMMA_ROOT = Path(
    "/Users/rishisim/Library/Application Support/ChildLens Public Model Bakeoff/v1.3.1"
)
GEMMA_PYTHON = GEMMA_ROOT / "venv/bin/python3.10"
GEMMA_MODEL = GEMMA_ROOT / "gemma-4-e4b-it-4bit"
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")

VERSION = "nursery-childlens-30-video-calibration-extension-v1.8.0"
PLAN_SCHEMA = "nursery-childlens-30-video-extension-restricted-plan-v1.8.0"
CHECKPOINT_SCHEMA = "nursery-childlens-30-video-extension-transfer-checkpoint-v1.8.0"
PUBLIC_SCHEMA = "nursery-childlens-combined-30-video-calibration-receipt-v1.8.0"
PROTOCOL_SHA256 = "e8d7ca48ed888f990730861e8f5a97274743dfc0e08befcdf40a508356c2aa56"
SELECTION_RECEIPT_SHA256 = "85399959c18775c8f9588879075f9d91e166cf73c92697aa3eeeb830b774d9b1"
PREDECESSOR_SHA256 = "2918e5094071e003931f79286945ad515ad41a2c5987eee3e98b14ea398f5b56"
QWEN_ACTIVATION_SHA256 = "c926207bbb9fd02b7cc23316cd17bd1c87298e5f64593ef7ed0a80c32271edf5"
QWEN_ASR_CANARY_SHA256 = "bd98566b8ebe4f4c1bccd41d123c4bf675f635eecd8cd896913f1f97a0fb4502"
QWEN_VLM_CANARY_SHA256 = "eb5447e13f44cc35c64e433f7217dc2423c5aff799c27101b6ce22bec50390b3"
GEMMA_ACTIVATION_SHA256 = "10cb7d0151e2fff40a3f5bf006ea510a3f82a4223410dec51ebfb08a34f14d7f"
QWEN_WORKER_SHA256 = "83ab0bf22240ee15a2fa7065bf57452e5361da5c0b943e07c764ccfbff61118c"
GEMMA_WORKER_SHA256 = "c78fa8e9c0f995ed611f7fd8f7d2076f03a73d370574d3c92cf5e107f236c38f"
GEMMA_EXTENSION_WORKER_SHA256 = "a98ce153d6b368813efa22b6fd28f4ac338444b97f1c8bfc03a979b3841e8be3"
WHISPER_ADAPTER_SHA256 = "abad8f58c606ae0081b04fbf44373f3bccb19abceb720c7c3b5ffc0732c13c46"
WHISPER_MODEL_SHA256 = "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
WHISPER_CLI_SHA256 = "9613b31e5380c184ae29ccb1d4046953d7037e8eb55308c9f1a34f145143b892"
AUTHOR_SHA256 = "4701390b00de5306baf4b877db972570cd7c1d67b97e9e6801b4e83c5405350c"
HUMAN_V12_SHA256 = "8bb132a5e0e7ea3b6e9fdcc005c197de109361f9d2e99a088c167e6f23459661"
ITEM_COUNT_NEW = 15
ITEM_COUNT_COMBINED = 30
SECONDS_NEW = 900
SECONDS_COMBINED = 1800
WINDOWS_NEW = 135
WINDOWS_OLD = 137
WINDOWS_COMBINED = 272
FREE_FLOOR_BYTES = 50 * 1024**3
MAX_PIPE_BYTES = 2 * 1024**2
MAX_JSON_BYTES = 64 * 1024**2
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PATH_TOKEN = re.compile(r"(?i)(?:/users/|file://|\\\\users\\\\)")
MEDIA_TOKEN = re.compile(r"(?i)\b\S+\.(?:mp4|mov|mkv|avi|webm|wav|m4a)\b")


class CombinedCalibrationError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise CombinedCalibrationError(code)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        _fail("E_MODULE")
    return module


base = _load("childlens_combined_base_v18", BASE)
gemma_base = _load("childlens_combined_gemma_base_v18", GEMMA_COORDINATOR)
gemma_worker = _load("childlens_combined_gemma_worker_v18", GEMMA_WORKER)
firewall = _load("childlens_combined_firewall_v18", FIREWALL)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("E_CANONICAL")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        _fail("E_FILE")
    return digest.hexdigest()


def _private_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _private_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_json(path: Path, maximum: int = MAX_JSON_BYTES) -> Any:
    try:
        payload = path.read_bytes()
        if not payload or len(payload) > maximum:
            _fail("E_JSON")
        return json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_JSON")


def _write_once(path: Path, value: Any, *, private: bool) -> None:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        os.chmod(path.parent, 0o700)
    if path.exists():
        if path.read_bytes() != payload:
            _fail("E_IMMUTABLE_CONFLICT")
        return
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    mode = 0o600 if private else 0o644
    try:
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, mode)
    except OSError:
        _fail("E_WRITE")
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _read_fd(descriptor: int) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(MAX_PIPE_BYTES + 1)
    if not payload or len(payload) > MAX_PIPE_BYTES:
        _fail("E_PIPE")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        _fail("E_PIPE")


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_PIPE_BYTES:
        _fail("E_PIPE")
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()


def _environment(work: Path) -> dict[str, str]:
    value = dict(
        firewall.scrubbed_subprocess_environment(
            {
                "OMP_NUM_THREADS": "2",
                "VECLIB_MAXIMUM_THREADS": "2",
            }
        )
    )
    value["TMPDIR"] = str(work)
    return value


def _outer_environment() -> dict[str, str]:
    return dict(
        firewall.scrubbed_subprocess_environment(
            {
                "OMP_NUM_THREADS": "2",
                "VECLIB_MAXIMUM_THREADS": "2",
            }
        )
    )


def _public_preflight() -> Mapping[str, Any]:
    fixed = {
        PROTOCOL: PROTOCOL_SHA256,
        SELECTION_RECEIPT: SELECTION_RECEIPT_SHA256,
        PREDECESSOR: PREDECESSOR_SHA256,
        QWEN_ACTIVATION: QWEN_ACTIVATION_SHA256,
        QWEN_ASR_CANARY: QWEN_ASR_CANARY_SHA256,
        QWEN_VLM_CANARY: QWEN_VLM_CANARY_SHA256,
        GEMMA_ACTIVATION: GEMMA_ACTIVATION_SHA256,
        QWEN_WORKER: QWEN_WORKER_SHA256,
        GEMMA_WORKER: GEMMA_WORKER_SHA256,
        GEMMA_EXTENSION_WORKER: GEMMA_EXTENSION_WORKER_SHA256,
        WHISPER_ADAPTER: WHISPER_ADAPTER_SHA256,
        WHISPER_MODEL: WHISPER_MODEL_SHA256,
        WHISPER_CLI: WHISPER_CLI_SHA256,
        AUTHOR: AUTHOR_SHA256,
        HUMAN_V12: HUMAN_V12_SHA256,
    }
    if any(not path.is_file() or _sha256_file(path) != digest for path, digest in fixed.items()):
        _fail("E_PUBLIC_BINDING")
    if not ACQUISITION_RECEIPT.is_file():
        _fail("E_ACQUISITION_NOT_READY")
    acquisition = _read_json(ACQUISITION_RECEIPT)
    selection = _read_json(SELECTION_RECEIPT)
    if (
        acquisition.get("status") != "EXTENSION_ACQUISITION_AND_CLIPPING_COMPLETE"
        or acquisition.get("completed_item_count") != ITEM_COUNT_NEW
        or acquisition.get("free_space_floor_satisfied") is not True
        or acquisition.get("scientific_endpoint_opened") is not False
        or selection.get("combined_item_count") != ITEM_COUNT_COMBINED
        or selection.get("combined_candidate_window_count") != WINDOWS_COMBINED
    ):
        _fail("E_ACQUISITION_RECEIPT")
    required_runtime = (
        WHISPER_PYTHON,
        SILERO_MODEL,
        QWEN_PYTHON,
        QWEN_ASR_PYTHON,
        FFMPEG,
    )
    required_models = (
        QWEN_ASR_MODEL,
        QWEN_ALIGNER_MODEL,
        QWEN_VLM_MODEL,
        GEMMA_MODEL,
    )
    if any(not path.is_file() for path in required_runtime) or any(
        not path.is_dir() for path in required_models
    ) or not GEMMA_PYTHON.is_file():
        _fail("E_RUNTIME")
    gemma_activation = _read_json(GEMMA_ACTIVATION)
    qwen_asr_canary = _read_json(QWEN_ASR_CANARY)
    qwen_vlm_canary = _read_json(QWEN_VLM_CANARY)
    artifact = gemma_activation.get("artifact_binding", {})
    if (
        artifact.get("snapshot_manifest_sha256")
        != "34310498dc5b809bf4baa79a5290c9e675022d12b725993317bccbffacf1d3ae"
        or artifact.get("snapshot_bytes") != 5179241512
        or gemma_activation.get("status")
        != "PROTOTYPE_INSTRUMENT_ACTIVATED_PREFLIGHT_DELEGATION_RESTRICTED_INFERENCE_NOT_RUN"
        or qwen_asr_canary.get("status") != "PASS"
        or qwen_vlm_canary.get("status") != "PASS"
    ):
        _fail("E_MODEL_BINDING")
    asr_manifest, asr_bytes, _ = base._tree_manifest(QWEN_ASR_MODEL)
    aligner_manifest, aligner_bytes, _ = base._tree_manifest(QWEN_ALIGNER_MODEL)
    qwen_vlm_manifest, qwen_vlm_bytes, _ = base._tree_manifest(QWEN_VLM_MODEL)
    gemma_manifest, gemma_bytes, _ = gemma_base._tree_manifest(GEMMA_MODEL)
    if (
        asr_manifest != qwen_asr_canary.get("asr_manifest_sha256")
        or asr_bytes != qwen_asr_canary.get("asr_artifact_bytes")
        or aligner_manifest != qwen_asr_canary.get("aligner_manifest_sha256")
        or aligner_bytes != qwen_asr_canary.get("aligner_artifact_bytes")
        or qwen_vlm_manifest != qwen_vlm_canary.get("artifact_manifest_sha256")
        or qwen_vlm_bytes != qwen_vlm_canary.get("artifact_bytes")
        or gemma_manifest != artifact.get("snapshot_manifest_sha256")
        or gemma_bytes != artifact.get("snapshot_bytes")
    ):
        _fail("E_MODEL_HASH")
    backend = firewall.NetworkIsolationBackend.detect()
    firewall.verify_network_isolation(backend)
    return {
        "acquisition_receipt_sha256": _sha256_file(ACQUISITION_RECEIPT),
        "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "qwen_activation_sha256": QWEN_ACTIVATION_SHA256,
        "gemma_activation_sha256": GEMMA_ACTIVATION_SHA256,
    }


def _new_items(root: Path) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    namespace = root / "provisional_calibration_v1/extension_v1_8"
    plan = _read_json(namespace / "restricted_extension_plan.json")
    checkpoint = _read_json(namespace / "restricted_transfer_checkpoint.json")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("status") != "COMPLETE"
        or plan.get("additional_item_count") != ITEM_COUNT_NEW
        or len(plan.get("items", [])) != ITEM_COUNT_NEW
        or len(checkpoint.get("items", [])) != ITEM_COUNT_NEW
    ):
        _fail("E_RESTRICTED_BINDING")
    states = {
        row.get("selection_rank"): row
        for row in checkpoint["items"]
        if isinstance(row, Mapping)
    }
    items: list[dict[str, Any]] = []
    seconds = windows = 0
    for row in plan["items"]:
        state = states.get(row.get("selection_rank"))
        if not isinstance(state, Mapping) or state.get("status") != "COMPLETE":
            _fail("E_RESTRICTED_BINDING")
        relative = state.get("clip_relative_path")
        digest = state.get("clip_sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not HEX64.fullmatch(digest)
        ):
            _fail("E_RESTRICTED_BINDING")
        media = (root / relative).resolve(strict=True)
        if (
            not _inside(media, root)
            or not _private_file(media)
            or media.stat().st_size != state.get("clip_bytes")
            or _sha256_file(media) != digest
        ):
            _fail("E_RESTRICTED_MEDIA")
        intervals = [
            (float(span["start_ms"]) / 1000.0, float(span["end_ms"]) / 1000.0)
            for span in row["sample_segments_clip_ms"]
        ]
        candidate_windows = [
            (float(span["start_ms"]) / 1000.0, float(span["end_ms"]) / 1000.0)
            for span in row["candidate_windows_clip_ms"]
        ]
        seconds += sum(end - start for start, end in intervals)
        windows += len(candidate_windows)
        items.append(
            {
                "opaque_key": digest,
                "media": media,
                "intervals": intervals,
                "candidate_windows": candidate_windows,
                "baseline_candidates": [{} for _ in candidate_windows],
            }
        )
    items.sort(key=lambda row: row["opaque_key"])
    if (
        len({row["opaque_key"] for row in items}) != ITEM_COUNT_NEW
        or abs(seconds - SECONDS_NEW) > 0.01
        or windows != WINDOWS_NEW
    ):
        _fail("E_RESTRICTED_SAMPLE")
    return items, plan, checkpoint


def _invoke_whisper(
    items: Sequence[Mapping[str, Any]], namespace: Path, work: Path
) -> Mapping[str, Any]:
    output_dir = namespace / "whisper"
    output_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(output_dir, 0o700)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        output = output_dir / f"{item['opaque_key']}.json"
        if not _private_file(output):
            scratch = Path(tempfile.mkdtemp(prefix=f"whisper-{index:03d}-", dir=work))
            os.chmod(scratch, 0o700)
            media_fd = job_fd = output_fd = -1
            try:
                job = {
                    "schema_version": "childlens-restricted-audio-job-v1.3.0",
                    "windows": [
                        {"start_seconds": start, "end_seconds": end}
                        for start, end in item["intervals"]
                    ],
                }
                job_path = scratch / "job.json"
                pending = scratch / "output.json"
                _write_once(job_path, job, private=True)
                media_fd = os.open(item["media"], os.O_RDONLY)
                job_fd = os.open(job_path, os.O_RDONLY)
                output_fd = os.open(
                    pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                command = [
                    str(WHISPER_PYTHON),
                    "-I",
                    str(WHISPER_ADAPTER),
                    "--mode",
                    "audio",
                    "--job-fd",
                    str(job_fd),
                    "--media-fd",
                    str(media_fd),
                    "--output-fd",
                    str(output_fd),
                    "--ffmpeg",
                    str(FFMPEG),
                    "--whisper-cli",
                    str(WHISPER_CLI),
                    "--whisper-model",
                    str(WHISPER_MODEL),
                    "--silero-model",
                    str(SILERO_MODEL),
                ]
                completed = subprocess.run(
                    command,
                    cwd=scratch,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=_environment(scratch),
                    pass_fds=(media_fd, job_fd, output_fd),
                    close_fds=True,
                    check=False,
                    timeout=8 * 60 * 60,
                )
                if completed.returncode != 0:
                    _fail("E_WHISPER")
                document = _read_json(pending)
                _write_once(output, document, private=True)
            finally:
                for descriptor in (media_fd, job_fd, output_fd):
                    if descriptor >= 0:
                        with contextlib.suppress(OSError):
                            os.close(descriptor)
                shutil.rmtree(scratch, ignore_errors=True)
        document = _read_json(output)
        if (
            document.get("schema_version")
            != "childlens-restricted-audio-pseudo-labels-v1.3.0"
            or document.get("pseudo_labels_are_ground_truth") is not False
            or not isinstance(document.get("asr_hypotheses"), list)
        ):
            _fail("E_WHISPER_OUTPUT")
        rows.append({"opaque_key": item["opaque_key"], "document": document})
    return {
        "schema_version": "nursery-childlens-whisper-extension-restricted-v1.8.0",
        "pseudo_labels_are_ground_truth": False,
        "network_disabled_during_inference": True,
        "items": rows,
    }


def _invoke_qwen(
    mode: str,
    items: Sequence[Mapping[str, Any]],
    namespace: Path,
    work: Path,
    transcript_by_key: Mapping[str, list[dict[str, Any]]] | None = None,
) -> Mapping[str, Any]:
    output = namespace / f"qwen3_{mode}_restricted.json"
    if _private_file(output):
        return _read_json(output)
    media_fds: list[int] = []
    job_fd = output_fd = -1
    scratch = Path(tempfile.mkdtemp(prefix=f"qwen-{mode}-", dir=work))
    os.chmod(scratch, 0o700)
    try:
        rows = []
        for item in items:
            descriptor = os.open(item["media"], os.O_RDONLY)
            media_fds.append(descriptor)
            row: dict[str, Any] = {
                "opaque_key": item["opaque_key"],
                "media_fd": descriptor,
                "intervals": [
                    {"start_seconds": start, "end_seconds": end}
                    for start, end in item["intervals"]
                ],
            }
            if mode == "vlm":
                row["candidate_windows"] = [
                    {"start_seconds": start, "end_seconds": end}
                    for start, end in item["candidate_windows"]
                ]
                row["transcript_hypotheses"] = list(
                    (transcript_by_key or {}).get(item["opaque_key"], [])
                )
            rows.append(row)
        job = {
            "schema_version": f"nursery-childlens-{mode}-job-v1",
            "sample_item_count": ITEM_COUNT_NEW,
            "sample_total_seconds": SECONDS_NEW,
            "items": rows,
        }
        job_path = scratch / "job.json"
        pending = scratch / "output.json"
        _write_once(job_path, job, private=True)
        job_fd = os.open(job_path, os.O_RDONLY)
        output_fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        python = QWEN_ASR_PYTHON if mode == "asr" else QWEN_PYTHON
        command = [
            str(python),
            "-I",
            str(QWEN_WORKER),
            "--mode",
            mode,
            "--job-fd",
            str(job_fd),
            "--output-fd",
            str(output_fd),
            "--ffmpeg",
            str(FFMPEG),
        ]
        if mode == "asr":
            command.extend(
                [
                    "--asr-model",
                    str(QWEN_ASR_MODEL),
                    "--aligner-model",
                    str(QWEN_ALIGNER_MODEL),
                ]
            )
        else:
            command.extend(["--vlm-model", str(QWEN_VLM_MODEL)])
        completed = subprocess.run(
            command,
            cwd=scratch,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_environment(scratch),
            pass_fds=(job_fd, output_fd, *media_fds),
            close_fds=True,
            check=False,
            timeout=8 * 60 * 60,
        )
        if completed.returncode != 0:
            _fail(f"E_QWEN_{mode.upper()}")
        document = _read_json(pending)
        _write_once(output, document, private=True)
        return document
    finally:
        for descriptor in (job_fd, output_fd, *media_fds):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        shutil.rmtree(scratch, ignore_errors=True)


def _expected_windows(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for local_index, (start, end) in enumerate(item["candidate_windows"]):
            rows.append(
                {
                    "opaque_key": item["opaque_key"],
                    "window_index": local_index,
                    "window_start_seconds": float(start),
                    "window_end_seconds": float(end),
                }
            )
    if len(rows) != WINDOWS_NEW:
        _fail("E_WINDOW_COUNT")
    return rows


def _valid_checkpoint_row(row: Any, index: int, expected: Mapping[str, Any]) -> bool:
    return (
        isinstance(row, Mapping)
        and row.get("schema_version") == gemma_worker.CHECKPOINT_SCHEMA
        and row.get("global_window_index") == index
        and row.get("opaque_key") == expected["opaque_key"]
        and row.get("window_index") == expected["window_index"]
        and row.get("window_start_seconds") == expected["window_start_seconds"]
        and row.get("window_end_seconds") == expected["window_end_seconds"]
        and type(row.get("schema_valid")) is bool
        and row.get("pseudo_labels_are_ground_truth") is False
        and row.get("human_validation") is False
        and isinstance(row.get("raw_response"), str)
        and row.get("raw_response_sha256")
        == hashlib.sha256(row["raw_response"].encode("utf-8")).hexdigest()
        and all(
            row.get(field) in allowed
            for field, allowed in gemma_worker.ALLOWED.items()
        )
    )


def _gemma_output(
    items: Sequence[Mapping[str, Any]], namespace: Path, work: Path
) -> Mapping[str, Any]:
    output = namespace / "gemma_referential_restricted.json"
    checkpoint = namespace / "gemma_checkpoint"
    checkpoint.mkdir(mode=0o700, exist_ok=True)
    os.chmod(checkpoint, 0o700)
    expected = _expected_windows(items)
    names = sorted(member.name for member in checkpoint.iterdir())
    if any(not re.fullmatch(r"[0-9]{6}\.json", name) for name in names):
        _fail("E_GEMMA_CHECKPOINT")
    if names != [f"{index:06d}.json" for index in range(len(names))]:
        _fail("E_GEMMA_CHECKPOINT")
    rows = []
    for index, name in enumerate(names):
        row = _read_json(checkpoint / name)
        if not _valid_checkpoint_row(row, index, expected[index]):
            _fail("E_GEMMA_CHECKPOINT")
        rows.append(row)
    if len(rows) < WINDOWS_NEW:
        scratch = Path(tempfile.mkdtemp(prefix="gemma-", dir=work))
        os.chmod(scratch, 0o700)
        media_fds: list[int] = []
        job_fd = checkpoint_fd = -1
        try:
            worker_items = []
            for item in items:
                descriptor = os.open(item["media"], os.O_RDONLY)
                media_fds.append(descriptor)
                worker_items.append(
                    {
                        "opaque_key": item["opaque_key"],
                        "media_fd": descriptor,
                        "intervals": [
                            {"start_seconds": start, "end_seconds": end}
                            for start, end in item["intervals"]
                        ],
                        "candidate_windows": [
                            {"start_seconds": start, "end_seconds": end}
                            for start, end in item["candidate_windows"]
                        ],
                    }
                )
            job = {
                "schema_version": gemma_worker.JOB_SCHEMA,
                "sample_item_count": ITEM_COUNT_NEW,
                "sample_total_seconds": SECONDS_NEW,
                "candidate_window_count": WINDOWS_NEW,
                "resume_from": len(rows),
                "parser_mode": "PRIMARY",
                "items": worker_items,
            }
            job_path = scratch / "job.json"
            _write_once(job_path, job, private=True)
            job_fd = os.open(job_path, os.O_RDONLY)
            checkpoint_fd = os.open(checkpoint, os.O_RDONLY)
            completed = subprocess.run(
                [
                    str(GEMMA_PYTHON),
                    "-I",
                    str(GEMMA_EXTENSION_WORKER),
                    "--job-fd",
                    str(job_fd),
                    "--checkpoint-dir-fd",
                    str(checkpoint_fd),
                    "--ffmpeg",
                    str(FFMPEG),
                    "--model",
                    str(GEMMA_MODEL),
                ],
                cwd=scratch,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=_environment(scratch),
                pass_fds=(job_fd, checkpoint_fd, *media_fds),
                close_fds=True,
                check=False,
                timeout=12 * 60 * 60,
            )
            if completed.returncode != 0:
                _fail("E_GEMMA")
        finally:
            for descriptor in (job_fd, checkpoint_fd, *media_fds):
                if descriptor >= 0:
                    with contextlib.suppress(OSError):
                        os.close(descriptor)
            shutil.rmtree(scratch, ignore_errors=True)
        names = sorted(member.name for member in checkpoint.iterdir())
        if names != [f"{index:06d}.json" for index in range(WINDOWS_NEW)]:
            _fail("E_GEMMA_INCOMPLETE")
        rows = []
        for index, name in enumerate(names):
            row = _read_json(checkpoint / name)
            if not _valid_checkpoint_row(row, index, expected[index]):
                _fail("E_GEMMA_CHECKPOINT")
            rows.append(row)
    if len(rows) != WINDOWS_NEW:
        _fail("E_GEMMA_INCOMPLETE")
    by_key: dict[str, list[dict[str, Any]]] = {
        item["opaque_key"]: [] for item in items
    }
    for row in rows:
        by_key[row["opaque_key"]].append(
            {
                "window_index": row["window_index"],
                "window_start_seconds": row["window_start_seconds"],
                "window_end_seconds": row["window_end_seconds"],
                **{field: row[field] for field in gemma_worker.ALLOWED},
                "schema_valid": row["schema_valid"],
            }
        )
    document = {
        "schema_version": gemma_worker.legacy.OUTPUT_SCHEMA,
        "pseudo_labels_are_ground_truth": False,
        "human_validation": False,
        "network_disabled_during_inference": True,
        "candidate_window_count": WINDOWS_NEW,
        "items": [
            {"opaque_key": item["opaque_key"], "candidates": by_key[item["opaque_key"]]}
            for item in items
        ],
    }
    _write_once(output, document, private=True)
    return document


def _validate_qwen_new(
    items: Sequence[Mapping[str, Any]], asr: Any, vlm: Any
) -> None:
    expected = [item["opaque_key"] for item in items]
    if (
        not isinstance(asr, Mapping)
        or asr.get("schema_version") != base.ASR_SCHEMA
        or asr.get("pseudo_labels_are_ground_truth") is not False
        or asr.get("network_disabled_during_inference") is not True
        or [row.get("opaque_key") for row in asr.get("items", [])] != expected
        or not isinstance(vlm, Mapping)
        or vlm.get("schema_version") != base.VLM_SCHEMA
        or vlm.get("pseudo_labels_are_ground_truth") is not False
        or vlm.get("network_disabled_during_inference") is not True
        or [row.get("opaque_key") for row in vlm.get("items", [])] != expected
    ):
        _fail("E_QWEN_OUTPUT")
    total = 0
    by_key = {item["opaque_key"]: item for item in items}
    for row in vlm["items"]:
        candidates = row.get("candidates")
        item = by_key[row["opaque_key"]]
        if not isinstance(candidates, list) or len(candidates) != len(
            item["candidate_windows"]
        ):
            _fail("E_QWEN_OUTPUT")
        total += len(candidates)
    if total != WINDOWS_NEW:
        _fail("E_QWEN_OUTPUT")


def _old_gemma(
    root: Path, old_items: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    expected = [item["opaque_key"] for item in old_items]
    base_dir = root / "provisional_calibration_v1/gemma_substitution_v1"
    matches: list[Mapping[str, Any]] = []
    if not _private_directory(base_dir):
        _fail("E_OLD_GEMMA")
    for path in base_dir.glob("*/gemma_referential_restricted.json"):
        if not _private_file(path):
            continue
        document = _read_json(path)
        if (
            document.get("schema_version") == gemma_worker.legacy.OUTPUT_SCHEMA
            and document.get("pseudo_labels_are_ground_truth") is False
            and document.get("human_validation") is False
            and document.get("network_disabled_during_inference") is True
            and document.get("candidate_window_count") == WINDOWS_OLD
            and [row.get("opaque_key") for row in document.get("items", [])]
            == expected
        ):
            matches.append(document)
    unique = {_digest(document): document for document in matches}
    if len(unique) != 1:
        _fail("E_OLD_GEMMA")
    return next(iter(unique.values()))


def _combine_documents(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    schema: str,
    candidate_count: int | None = None,
) -> Mapping[str, Any]:
    items = [*old.get("items", []), *new.get("items", [])]
    if len(items) != ITEM_COUNT_COMBINED:
        _fail("E_COMBINE")
    result: dict[str, Any] = {
        "schema_version": schema,
        "pseudo_labels_are_ground_truth": False,
        "human_validation": False,
        "network_disabled_during_inference": True,
        "items": items,
    }
    if candidate_count is not None:
        result["candidate_window_count"] = candidate_count
    return result


def _rename_paths(value: Any) -> Any:
    names = {
        "existing_whisper_qwen2_path": "qwen3asr_plus_qwen3vl_path",
        "challenger_qwen3asr_qwen3vl_path": "qwen3asr_plus_gemma4_path",
    }
    if isinstance(value, Mapping):
        return {
            names.get(str(key), str(key)): _rename_paths(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_rename_paths(child) for child in value]
    return value


def _abstention_rates(
    items: Sequence[Mapping[str, Any]],
    qwen: Mapping[str, Any],
    gemma: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    maps = {
        "qwen3asr_plus_qwen3vl_path": {
            row["opaque_key"]: row["candidates"] for row in qwen["items"]
        },
        "qwen3asr_plus_gemma4_path": {
            row["opaque_key"]: row["candidates"] for row in gemma["items"]
        },
    }
    result: dict[str, dict[str, float]] = {}
    for field in base.FIELDS:
        result[field] = {}
        for path, by_key in maps.items():
            values = [
                base._binary(candidate, field)
                for item in items
                for candidate in by_key[item["opaque_key"]]
            ]
            result[field][path] = sum(value is None for value in values) / len(values)
    for category, bins in base.CATEGORY_FIELDS.items():
        source = (
            "candidate_count_bin"
            if category == "candidate_multiplicity"
            else "lag_event_unit_bin"
        )
        result[category] = {}
        for path, by_key in maps.items():
            values = [
                candidate.get(source)
                for item in items
                for candidate in by_key[item["opaque_key"]]
            ]
            result[category][path] = sum(
                value in {None, "abstain", "undecidable"} for value in values
            ) / len(values)
    return result


def _gate(
    ranges: Mapping[str, Any],
    visual: Mapping[str, Any],
    speech: Mapping[str, Any],
    abstention: Mapping[str, Mapping[str, float]],
    raw_schema: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    if visual["visual_item_coverage_interval"][0] < 0.8:
        failures.append("VISUAL_ITEM_COVERAGE")
    if raw_schema["qwen3asr_plus_qwen3vl_path"] < 0.95:
        failures.append("QWEN3_SCHEMA_VALIDITY")
    if raw_schema["qwen3asr_plus_gemma4_path"] < 0.95:
        failures.append("GEMMA_SCHEMA_VALIDITY")
    if speech["challenger_nonempty_item_coverage_interval"][0] < 0.8:
        failures.append("ASR_ITEM_COVERAGE")
    for dimension, paths in abstention.items():
        if any(value > 0.5 for value in paths.values()):
            failures.append(f"ABSTENTION_{dimension.upper()}")
    for field in base.FIELDS:
        if (
            ranges.get(field, {})
            .get("conservative_envelope", {})
            .get("status")
            != "PUBLISHED"
        ):
            failures.append(f"ENVELOPE_{field.upper()}")
    for category in base.CATEGORY_FIELDS:
        if not any(
            value.get("conservative_envelope", {}).get("status") == "PUBLISHED"
            for value in ranges.get(category, {}).values()
        ):
            failures.append(f"ENVELOPE_{category.upper()}")
    return sorted(set(failures))


def _public_guard(value: Any) -> None:
    text = _canonical(value).decode("utf-8")
    if PATH_TOKEN.search(text) or MEDIA_TOKEN.search(text):
        _fail("E_PUBLIC_PRIVACY")
    forbidden = {
        "opaque_key",
        "media_key",
        "participant_key",
        "session_key",
        "source_locator",
        "transcript_hypothesis",
        "utterance_hypotheses",
        "candidate_windows",
        "raw_response",
    }
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if forbidden.intersection(current):
                _fail("E_PUBLIC_PRIVACY")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _restricted_error_code(value: Any) -> str | None:
    if (
        isinstance(value, Mapping)
        and value.get("schema_version")
        == "nursery-childlens-restricted-error-envelope-v1.8.4"
        and value.get("status") == "ERROR"
        and isinstance(value.get("failure_code"), str)
        and re.fullmatch(r"E_[A-Z0-9_]+", value["failure_code"])
        and set(value) == {"schema_version", "status", "failure_code"}
    ):
        return value["failure_code"]
    return None


def _bind_import_alias(name: str, module: Any) -> None:
    existing = sys.modules.get(name)
    if existing is not None and existing is not module:
        _fail("E_AUTHOR_ALIAS")
    sys.modules[name] = module


def _bind_author_import_alias(author: Any) -> None:
    _bind_import_alias("childlens_author_audit_v1_3", author)


def restricted_execute() -> Mapping[str, Any]:
    bindings = _public_preflight()
    author = _load("childlens_combined_author_v18", AUTHOR)
    _bind_author_import_alias(author)
    human_v12 = _load("childlens_combined_human_v12_v18", HUMAN_V12)
    _bind_import_alias("childlens_human_validation_v1_2", human_v12)
    root = firewall.validate_quarantine_root(Path(author.discover_runtime_root()), ROOT)
    if shutil.disk_usage(root).free < FREE_FLOOR_BYTES:
        _fail("E_STORAGE_FLOOR")
    new_items, plan, checkpoint = _new_items(root)
    namespace = root / "provisional_calibration_v1/extension_v1_8/inference"
    namespace.mkdir(mode=0o700, exist_ok=True)
    os.chmod(namespace, 0o700)
    work = namespace / "scratch"
    work.mkdir(mode=0o700, exist_ok=True)
    os.chmod(work, 0o700)
    if any(work.iterdir()):
        _fail("E_SCRATCH_NOT_EMPTY")
    whisper = _invoke_whisper(new_items, namespace, work)
    for item, row in zip(new_items, whisper["items"]):
        document = row["document"]
        item["baseline_segments"] = list(document["asr_hypotheses"])
        item["baseline_language"] = str(document.get("language_hypothesis", ""))
    qwen_new_asr = _invoke_qwen("asr", new_items, namespace, work)
    transcript_by_key = {
        row["opaque_key"]: list(row.get("utterance_hypotheses", []))
        for row in qwen_new_asr.get("items", [])
        if isinstance(row, Mapping)
    }
    qwen_new_vlm = _invoke_qwen(
        "vlm", new_items, namespace, work, transcript_by_key=transcript_by_key
    )
    _validate_qwen_new(new_items, qwen_new_asr, qwen_new_vlm)
    gemma_new = _gemma_output(new_items, namespace, work)
    if any(work.iterdir()):
        _fail("E_SCRATCH_NOT_EMPTY")

    old_items, old_sample_digest = base._restricted_inputs(root)
    if len(old_items) != 15:
        _fail("E_OLD_SAMPLE")
    old_qwen_asr = _read_json(root / "provisional_calibration_v1/qwen3_asr_restricted.json")
    old_qwen_vlm = _read_json(root / "provisional_calibration_v1/qwen3_vl_restricted.json")
    old_gemma = _old_gemma(root, old_items)
    combined_items = sorted([*old_items, *new_items], key=lambda row: row["opaque_key"])
    qwen_asr = _combine_documents(
        old_qwen_asr, qwen_new_asr, schema=base.ASR_SCHEMA
    )
    qwen_vlm = _combine_documents(
        old_qwen_vlm,
        qwen_new_vlm,
        schema=base.VLM_SCHEMA,
        candidate_count=WINDOWS_COMBINED,
    )
    gemma_vlm = _combine_documents(
        old_gemma,
        gemma_new,
        schema=gemma_worker.legacy.OUTPUT_SCHEMA,
        candidate_count=WINDOWS_COMBINED,
    )
    ranges, visual = base._calibration_ranges(
        combined_items, gemma_vlm, qwen_vlm
    )
    ranges = _rename_paths(ranges)
    visual["visual_item_coverage_interval"] = base._round_range(
        sum(bool(item["candidate_windows"]) for item in combined_items)
        / ITEM_COUNT_COMBINED
    )
    speech = base._speech_diagnostics(combined_items, qwen_asr)
    nonempty = sum(
        bool(str(row.get("transcript_hypothesis", "")).strip())
        for row in qwen_asr["items"]
    )
    speech["challenger_nonempty_item_coverage_interval"] = base._round_range(
        nonempty / ITEM_COUNT_COMBINED
    )
    qwen_candidates = [
        candidate for row in qwen_vlm["items"] for candidate in row["candidates"]
    ]
    gemma_candidates = [
        candidate for row in gemma_vlm["items"] for candidate in row["candidates"]
    ]
    raw_schema = {
        "qwen3asr_plus_qwen3vl_path": sum(
            bool(row.get("schema_valid")) for row in qwen_candidates
        )
        / WINDOWS_COMBINED,
        "qwen3asr_plus_gemma4_path": sum(
            bool(row.get("schema_valid")) for row in gemma_candidates
        )
        / WINDOWS_COMBINED,
    }
    abstention = _abstention_rates(
        combined_items, qwen_vlm, gemma_vlm
    )
    failures = _gate(ranges, visual, speech, abstention, raw_schema)
    passed = not failures
    receipt = {
        "schema_version": PUBLIC_SCHEMA,
        "status": "CALIBRATION_PASS" if passed else "CALIBRATION_STOP",
        "decision": (
            "CALIBRATION_PASS_MINIMAL_OUTCOME_READY"
            if passed
            else "CALIBRATION_STOP_MODEL_TRIANGULATION"
        ),
        "scope": "ONE_TIME_POST_FAILURE_30_PARTICIPANT_DISTINCT_VIDEO_EXTENSION",
        "protocol_sha256": PROTOCOL_SHA256,
        "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
        "acquisition_receipt_sha256": bindings["acquisition_receipt_sha256"],
        "predecessor_receipt_sha256": PREDECESSOR_SHA256,
        "original_sample_digest_sha256": old_sample_digest,
        "combined_selection_sha256": plan["combined_selection_sha256"],
        "restricted_transfer_checkpoint_sha256": _digest(checkpoint),
        "original_item_count": 15,
        "additional_item_count": 15,
        "combined_item_count": ITEM_COUNT_COMBINED,
        "combined_participants_distinct": True,
        "combined_speech_seconds": SECONDS_COMBINED,
        "combined_candidate_window_count": WINDOWS_COMBINED,
        "post_failure_extension_disclosed": True,
        "further_expansion_allowed": False,
        "speech_model_model_diagnostics": speech,
        "visual_model_model_diagnostics": visual,
        "calibration_ranges": ranges,
        "schema_validity_intervals": {
            path: base._round_range(value)
            for path, value in raw_schema.items()
        },
        "abstention_intervals": {
            dimension: {
                path: base._round_range(value)
                for path, value in paths.items()
            }
            for dimension, paths in abstention.items()
        },
        "gate_failures": failures,
        "all_frozen_calibration_gates_passed": passed,
        "minimum_cluster_k": 5,
        "outward_rate_rounding_step": 0.1,
        "model_specific_intersection_union_and_envelope_preserved": True,
        "majority_vote_used": False,
        "third_visual_model_used": False,
        "semantic_prompt_tuning_performed": False,
        "new_parser_correction_used": False,
        "original_successful_outputs_reused_read_only": True,
        "restricted_inference_network_disabled": True,
        "memory_heavy_inference_serialized": True,
        "pseudo_labels_are_ground_truth": False,
        "human_validation_claimed": False,
        "model_model_agreement_is_human_reliability": False,
        "simulator_oracle_only_evaluation_truth": True,
        "AEA_empirical_ancestry": False,
        "BabyView_empirical_ancestry": False,
        "cross_corpus_pooling": False,
        "restricted_payload_exported": False,
        "scientific_endpoint_opened": False,
        "scientific_outcome_run": False,
    }
    _public_guard(receipt)
    return receipt


def public_execute() -> Mapping[str, Any]:
    _public_preflight()
    backend = firewall.NetworkIsolationBackend.detect()
    result_read, result_write = os.pipe()
    try:
        completed = subprocess.run(
            backend.command(
                [
                    sys.executable,
                    "-I",
                    str(SCRIPT),
                    "--restricted",
                    "--result-fd",
                    str(result_write),
                ]
            ),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_outer_environment(),
            pass_fds=(result_write,),
            close_fds=True,
            check=False,
            timeout=36 * 60 * 60,
        )
        os.close(result_write)
        result_write = -1
        if completed.returncode != 0:
            _fail("E_RESTRICTED_EXECUTION")
        receipt = _read_fd(result_read)
        restricted_error = _restricted_error_code(receipt)
        if restricted_error is not None:
            _fail(restricted_error)
    finally:
        for descriptor in (result_read, result_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
    _public_guard(receipt)
    _write_once(PUBLIC_RECEIPT, receipt, private=False)
    return receipt


def _write_failure(code: str) -> None:
    value = {
        "schema_version": "nursery-childlens-combined-calibration-failure-v1.8.0",
        "status": "TECHNICAL_REVISE_PRE_CALIBRATION",
        "failure_code": code,
        "protocol_sha256": PROTOCOL_SHA256,
        "restricted_payload_exported": False,
        "scientific_endpoint_opened": False,
        "scientific_outcome_run": False,
    }
    with contextlib.suppress(Exception):
        _write_once(PUBLIC_FAILURE, value, private=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restricted", action="store_true")
    parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(argv)
    old_umask = os.umask(0o077)
    try:
        if args.restricted:
            if args.result_fd is None:
                _fail("E_ARGUMENTS")
            _write_fd(args.result_fd, restricted_execute())
            return 0
        if args.result_fd is not None:
            _fail("E_ARGUMENTS")
        receipt = public_execute()
        print(receipt["decision"])
        return 0 if receipt["status"] == "CALIBRATION_PASS" else 3
    except CombinedCalibrationError as exc:
        if args.restricted and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd(
                    args.result_fd,
                    {
                        "schema_version": "nursery-childlens-restricted-error-envelope-v1.8.4",
                        "status": "ERROR",
                        "failure_code": exc.code,
                    },
                )
            return 0
        if not args.restricted:
            _write_failure(exc.code)
            print("CALIBRATION_TECHNICAL_REVISE")
        return 2
    except Exception:
        if args.restricted and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd(
                    args.result_fd,
                    {
                        "schema_version": "nursery-childlens-restricted-error-envelope-v1.8.4",
                        "status": "ERROR",
                        "failure_code": "E_INTERNAL",
                    },
                )
            return 0
        if not args.restricted:
            _write_failure("E_INTERNAL")
            print("CALIBRATION_TECHNICAL_REVISE")
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
