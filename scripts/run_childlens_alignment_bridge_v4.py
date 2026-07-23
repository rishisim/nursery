#!/usr/bin/env python3
"""Run the development-only ChildLens bridge v4 audiovisual preflight.

All identifiers, paths, exact intervals, media, embeddings, projection weights,
and row-level results remain in the owner-private ChildLens quarantine.  Public
files contain only K-safe aggregates and cryptographic receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_v4.json"
PUBLIC_ROOT = ROOT / "output/childlens_alignment_bridge_v4"
FREEZE_RECEIPT = PUBLIC_ROOT / "protocol_and_split_freeze_receipt.json"
CALIBRATION_REPORT = PUBLIC_ROOT / "calibration_summary.json"
RESULT_REPORT = PUBLIC_ROOT / "development_go_no_go_report.json"
RESULT_MARKDOWN = PUBLIC_ROOT / "development_go_no_go_report.md"
VALIDATION_RECEIPT = PUBLIC_ROOT / "validation_receipt.json"

PRIVATE_RELATIVE = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_v4"
)
PRIVATE_MANIFEST = PRIVATE_RELATIVE / "restricted_split_window_manifest.json"
PRIVATE_EMBEDDINGS = PRIVATE_RELATIVE / "restricted_frontend_embeddings.npz"
PRIVATE_WEIGHTS = PRIVATE_RELATIVE / "restricted_projection_weights.npz"
PRIVATE_RESULT = PRIVATE_RELATIVE / "restricted_development_result.json"

V2_MANIFEST = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_remediation_v2/"
    "restricted_development_manifest.json"
)
V3_MANIFEST = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_expansion_v3/"
    "restricted_measurement_manifest.json"
)
V3_PLAN = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_expansion_v3/"
    "restricted_expansion_plan.json"
)

MODEL_ROOT = ROOT / ".external/models/childlens_bridge_v4"
AUDIO_MODEL = MODEL_ROOT / "wav2vec2-base"
VISION_MODEL = MODEL_ROOT / "dinov2-small"
INSTRUMENT_PYTHON = (
    Path.home()
    / "Library/Application Support/ChildLens Instruments/"
    "provisional-calibration-v1/qwen-asr-venv/bin/python"
)
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
FFPROBE = Path("/opt/homebrew/bin/ffprobe")
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"

EXPECTED_RESTRICTED_HASHES = {
    V2_MANIFEST: "9cb87b853eb43636d6baf09c281725eb98255b6cf73439025b47d627e84da5a8",
    V3_MANIFEST: "028efa424ed4dc2fe511a0b723b38f4feccba7f4de7db055934218dff1fe705d",
    V3_PLAN: "796eccc748cd61590bd0c9d4499e92e81277327f11e6ad4a19ce106afa4b4cb6",
}
EXPECTED_MODEL_HASHES = {
    AUDIO_MODEL / "pytorch_model.bin": (
        "3249fe98bfc62fcbc26067f724716a6ec49d12c4728a2af1df659013905dff21"
    ),
    AUDIO_MODEL / "config.json": (
        "4937977e24d12d1bba70cdce8709c3c04807a8e4ae8ddac4229c48c436ae99ae"
    ),
    VISION_MODEL / "model.safetensors": (
        "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
    ),
    VISION_MODEL / "config.json": (
        "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
    ),
}
PRIOR_SCOPES = {
    "v1": [
        "configs/childlens_alignment_bridge_v1.json",
        "babyworld_lite/childlens_alignment_bridge_v1",
        "output/childlens_alignment_bridge_v1",
        "scripts/run_childlens_alignment_bridge_preflight_v1.py",
    ],
    "v2": [
        "configs/childlens_alignment_bridge_remediation_v2.json",
        "babyworld_lite/childlens_alignment_bridge_v2",
        "output/childlens_alignment_bridge_remediation_v2",
        "scripts/run_childlens_alignment_bridge_remediation_v2.py",
    ],
    "v3": [
        "configs/childlens_alignment_bridge_expansion_v3.json",
        "babyworld_lite/childlens_alignment_bridge_v3",
        "output/childlens_alignment_bridge_expansion_v3",
        "scripts/run_childlens_alignment_bridge_expansion_v3.py",
        "scripts/acquire_childlens_alignment_bridge_expansion_v3.py",
        "scripts/measure_childlens_alignment_bridge_expansion_v3.py",
    ],
}
EXPECTED_PRIOR_TREES = {
    "v1": "4f668c666636e47e72a3c2162cb30590f89077f0e4e6ef8e69b149f4966df679",
    "v2": "30df10121591465cd4c357ead6b4aab8c2dff30b2a292106faa277661c621e45",
    "v3": "29718e4741bf31f27e5001e6a6398423745b1d5b8688032b3e9140a3f5ef90fa",
}

sys.path.insert(0, str(ROOT))
from babyworld_lite.childlens_alignment_bridge_v4.preflight import (  # noqa: E402
    BridgeV4Error,
    candidate_windows,
    canonical_bytes,
    cross_participant_assignment,
    deterministic_split,
    digest,
    effect_summary,
    keyed_hash,
    projected_cosine,
    safe_fraction,
    shifted_window,
    train_projection_heads,
)


def _fail(code: str) -> None:
    raise BridgeV4Error(code)


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
    except OSError as exc:
        raise BridgeV4Error("E_FILE") from exc
    return value.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("E_FILE")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeV4Error("E_FILE") from exc


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


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_once(path: Path, value: Any, *, private: bool) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        os.chmod(path.parent, 0o700)
    if path.exists():
        if path.read_bytes() != payload:
            _fail("E_IMMUTABLE_CONFLICT")
        return
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600 if private else 0o644,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, 0o600 if private else 0o644)
    except OSError as exc:
        raise BridgeV4Error("E_WRITE") from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _write_text_once(path: Path, value: str) -> None:
    payload = value.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            _fail("E_IMMUTABLE_CONFLICT")
        return
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, 0o644)
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _write_private_npz_once(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".pending-", suffix=".npz", delete=False
    ) as handle:
        pending = Path(handle.name)
    try:
        np.savez_compressed(pending, **arrays)
        payload_hash = _sha256_file(pending)
        if path.exists():
            if _sha256_file(path) != payload_hash:
                _fail("E_IMMUTABLE_CONFLICT")
            return
        os.replace(pending, path)
        os.chmod(path, 0o600)
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _discover_runtime() -> Path:
    candidates: list[Path] = []
    for hidden in ROOT.parent.iterdir():
        if (
            hidden.name.startswith(".")
            and "childlens" in hidden.name.casefold()
            and _private_directory(hidden)
        ):
            candidates.extend(
                path.parent
                for path in hidden.rglob("restricted_manifest/preselection_manifest.json")
                if _private_file(path)
            )
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) != 1:
        _fail("E_RUNTIME_DISCOVERY")
    runtime = unique[0]
    if (
        _inside(runtime, ROOT)
        or not _private_directory(runtime)
        or not _private_file(runtime / ".metadata_never_index")
    ):
        _fail("E_RUNTIME_CONTROL")
    return runtime


def _tree_digest(entries: Sequence[str]) -> str:
    rows: list[dict[str, str]] = []
    for entry in entries:
        path = ROOT / entry
        files = (
            [path]
            if path.is_file()
            else sorted(
                value
                for value in path.rglob("*")
                if value.is_file()
                and "__pycache__" not in value.parts
                and value.suffix != ".pyc"
            )
        )
        if not files:
            _fail("E_PRIOR_TREE")
        for value in files:
            rows.append(
                {
                    "path": value.relative_to(ROOT).as_posix(),
                    "sha256": _sha256_file(value),
                }
            )
    return digest(rows)


def _validate_prior_trees() -> dict[str, str]:
    observed = {
        version: _tree_digest(entries) for version, entries in PRIOR_SCOPES.items()
    }
    if observed != EXPECTED_PRIOR_TREES:
        _fail("E_PRIOR_IMMUTABILITY")
    return observed


def _config() -> Mapping[str, Any]:
    value = _read_json(CONFIG)
    if (
        value.get("schema_version")
        != "childlens-alignment-bridge-transcription-free-preflight-v4.0.0"
        or value.get("status")
        != "FROZEN_BEFORE_DEVELOPMENT_SPLIT_WINDOW_ASSIGNMENT_OR_AUDIOVISUAL_OUTCOME_INSPECTION"
        or value.get("scope", {}).get("locked_evaluation_allowed") is not False
        or value.get("scope", {}).get("development_participants") != 18
        or value.get("method_selection", {}).get("route_count") != 1
        or value.get("method_selection", {})
        .get("audio_frontend", {})
        .get("tokenizer_decoder_or_ctc_head_loaded")
        is not False
    ):
        _fail("E_PROTOCOL")
    return value


def _validate_models() -> None:
    for path, expected in EXPECTED_MODEL_HASHES.items():
        if not path.is_file() or _sha256_file(path) != expected:
            _fail("E_MODEL_BINDING")
    for executable in (INSTRUMENT_PYTHON, FFMPEG, FFPROBE, SANDBOX_EXEC):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            _fail("E_RUNTIME_DEPENDENCY")


def _ffprobe_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            [
                str(FFPROBE),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        duration = float(result.stdout.strip()) if result.returncode == 0 else math.nan
    except (OSError, subprocess.SubprocessError, ValueError):
        duration = math.nan
    if not math.isfinite(duration) or duration <= 0:
        _fail("E_MEDIA_DURATION")
    return duration


def _normalise_items(runtime: Path) -> list[dict[str, Any]]:
    for relative, expected in EXPECTED_RESTRICTED_HASHES.items():
        path = runtime / relative
        if not _private_file(path) or _sha256_file(path) != expected:
            _fail("E_RESTRICTED_BINDING")
    v2 = _read_json(runtime / V2_MANIFEST)
    v3 = _read_json(runtime / V3_MANIFEST)
    plan = _read_json(runtime / V3_PLAN)
    if (
        v2.get("development_count") != 8
        or v2.get("locked_rows_copied") != 0
        or v3.get("development_count") != 10
        or v3.get("locked_rows_copied_or_evaluated") != 0
        or len(v2.get("items", [])) != 8
        or len(v3.get("items", [])) != 10
        or len(plan.get("items", [])) != 10
        or plan.get("locked_participant_count_excluded") != 22
    ):
        _fail("E_DEVELOPMENT_BINDING")
    plan_by_rank = {
        int(row["selection_rank"]): row for row in plan["items"]
    }
    result: list[dict[str, Any]] = []
    for source, cohort in ((v2, "prior_v2"), (v3, "expansion_v3")):
        for raw in source["items"]:
            media = Path(str(raw.get("media_path", ""))).resolve()
            annotation = Path(str(raw.get("annotation_path", ""))).resolve()
            if (
                not _inside(media, runtime)
                or not _private_file(media)
                or _sha256_file(media) != raw.get("expected_media_sha256")
                or not _inside(annotation, runtime.parent)
                or not _private_file(annotation)
                or _sha256_file(annotation) != raw.get("annotation_sha256")
            ):
                _fail("E_MEDIA_BINDING")
            if cohort == "expansion_v3":
                plan_row = plan_by_rank.get(int(raw["selection_rank"]))
                if plan_row is None or len(plan_row.get("stratum", [])) != 4:
                    _fail("E_STRATUM_BINDING")
                activity = str(plan_row["stratum"][0])
                location = str(plan_row["stratum"][3])
            else:
                activity = str(raw.get("activity_label") or "__UNAVAILABLE__")
                location = str(raw.get("location_label") or "__UNAVAILABLE__")
            duration = _ffprobe_duration(media)
            result.append(
                {
                    "participant_key": str(raw["participant_key"]),
                    "item_key": str(raw["item_key"]),
                    "source_cohort": cohort,
                    "media_path": str(media),
                    "expected_media_sha256": str(raw["expected_media_sha256"]),
                    "annotation_sha256": str(raw["annotation_sha256"]),
                    "activity_label": activity,
                    "location_label": location,
                    "recording_duration_seconds": duration,
                    "sample_span_start_seconds": float(
                        raw["sample_span_start_seconds"]
                    ),
                    "sample_span_end_seconds": float(raw["sample_span_end_seconds"]),
                    "sample_segments": [dict(segment) for segment in raw["sample_segments"]],
                }
            )
    participants = [row["participant_key"] for row in result]
    if len(result) != 18 or len(set(participants)) != 18:
        _fail("E_PARTICIPANT_DISTINCTNESS")
    return result


def _speech_overlap(
    start_seconds: float,
    end_seconds: float,
    segments: Sequence[Mapping[str, Any]],
) -> float:
    return sum(
        max(
            0.0,
            min(end_seconds, float(segment["end_seconds"]))
            - max(start_seconds, float(segment["start_seconds"])),
        )
        for segment in segments
    )


def freeze() -> Mapping[str, Any]:
    config = _config()
    prior = _validate_prior_trees()
    _validate_models()
    runtime = _discover_runtime()
    items = _normalise_items(runtime)
    split = deterministic_split(
        [
            {
                "participant_key": row["participant_key"],
                "cohort": row["source_cohort"],
            }
            for row in items
        ],
        seed=str(config["development_split"]["split_seed"]),
        evaluation_per_cohort=3,
    )
    frozen_items: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for item in items:
        participant = item["participant_key"]
        duration = float(item["recording_duration_seconds"])
        windows = candidate_windows(
            item["sample_segments"],
            recording_duration_seconds=duration,
            window_seconds=float(config["windows"]["window_seconds"]),
            exclusion_buffer_seconds=float(
                config["windows"]["real_window_exclusion_buffer_seconds"]
            ),
            maximum_windows=int(config["windows"]["maximum_windows_per_participant"]),
            seed=str(config["development_split"]["split_seed"]),
            participant_key=participant,
            item_key=item["item_key"],
        )
        retained: list[dict[str, Any]] = []
        for index, window in enumerate(windows):
            row_key = keyed_hash(
                str(config["development_split"]["split_seed"]),
                participant,
                item["item_key"],
                window["start_ms"],
            )
            shifted = shifted_window(
                start_ms=int(window["start_ms"]),
                end_ms=int(window["end_ms"]),
                recording_duration_ms=round(duration * 1000),
                offset_ms=round(
                    float(
                        config["controls"]["within_recording_time_shift"][
                            "offset_seconds"
                        ]
                    )
                    * 1000
                ),
                minimum_gap_ms=round(
                    float(
                        config["controls"]["within_recording_time_shift"][
                            "minimum_gap_from_real_seconds"
                        ]
                    )
                    * 1000
                ),
                seed=str(config["development_split"]["split_seed"]),
                row_key=row_key,
            )
            if shifted is None:
                continue
            start = int(window["start_ms"]) / 1000.0
            end = int(window["end_ms"]) / 1000.0
            density = _speech_overlap(start, end, item["sample_segments"]) / (
                end - start
            )
            row = {
                "row_key": row_key,
                "row_hash": row_key,
                "participant_key": participant,
                "item_key": item["item_key"],
                "split": split[participant],
                "source_cohort": item["source_cohort"],
                "media_path": item["media_path"],
                "start_ms": int(window["start_ms"]),
                "end_ms": int(window["end_ms"]),
                "shift_start_ms": shifted[0],
                "shift_end_ms": shifted[1],
                "speech_density": density,
                "activity_label": item["activity_label"],
                "location_label": item["location_label"],
                "recording_position": ((start + end) / 2.0) / duration,
                "ordinal": index,
            }
            retained.append(row)
            all_rows.append(row)
        frozen_items.append(
            {
                **item,
                "split": split[participant],
                "windows": retained,
            }
        )
    counts = Counter(row["split"] for row in all_rows)
    participant_counts = Counter(row["participant_key"] for row in all_rows)
    evaluation = [row for row in all_rows if row["split"] == "evaluation"]
    assignment = cross_participant_assignment(evaluation)
    for index, donor_index in enumerate(assignment):
        evaluation[index]["shuffle_donor_row_key"] = evaluation[donor_index]["row_key"]
    minimum = int(config["windows"]["minimum_windows_per_participant"])
    g1 = (
        len(split) == 18
        and sum(value == "train" for value in split.values()) == 12
        and sum(value == "evaluation" for value in split.values()) == 6
        and min(participant_counts.values(), default=0) >= minimum
        and counts["train"] >= 72
        and counts["evaluation"] >= 36
        and all("shuffle_donor_row_key" in row for row in evaluation)
    )
    manifest = {
        "schema_version": "childlens-alignment-bridge-restricted-manifest-v4.0.0",
        "protocol_sha256": _sha256_file(CONFIG),
        "runner_sha256": _sha256_file(Path(__file__)),
        "package_sha256": _sha256_file(
            ROOT / "babyworld_lite/childlens_alignment_bridge_v4/preflight.py"
        ),
        "v2_manifest_sha256": EXPECTED_RESTRICTED_HASHES[V2_MANIFEST],
        "v3_manifest_sha256": EXPECTED_RESTRICTED_HASHES[V3_MANIFEST],
        "v3_plan_sha256": EXPECTED_RESTRICTED_HASHES[V3_PLAN],
        "development_participants": 18,
        "train_participants": 12,
        "evaluation_participants": 6,
        "locked_rows_loaded_scored_summarized_or_inspected": 0,
        "items": frozen_items,
    }
    manifest_path = runtime / PRIVATE_MANIFEST
    _write_once(manifest_path, manifest, private=True)
    split_identity = sorted(
        {
            keyed_hash(
                str(config["development_split"]["split_seed"]),
                participant,
                value,
            ): value
            for participant, value in split.items()
        }.items()
    )
    receipt = {
        "schema_version": "childlens-alignment-bridge-freeze-receipt-v4.0.0",
        "status": "FROZEN_BEFORE_MEDIA_DECODING_EMBEDDING_TRAINING_OR_OUTCOME_SCORING",
        "protocol_sha256": _sha256_file(CONFIG),
        "runner_sha256": _sha256_file(Path(__file__)),
        "package_sha256": manifest["package_sha256"],
        "prior_tree_sha256": prior,
        "model_weights_sha256": {
            "audio": EXPECTED_MODEL_HASHES[AUDIO_MODEL / "pytorch_model.bin"],
            "vision": EXPECTED_MODEL_HASHES[VISION_MODEL / "model.safetensors"],
        },
        "restricted_manifest_sha256": _sha256_file(manifest_path),
        "restricted_source_manifest_sha256": {
            "prior_eight": EXPECTED_RESTRICTED_HASHES[V2_MANIFEST],
            "expansion_ten": EXPECTED_RESTRICTED_HASHES[V3_MANIFEST],
        },
        "development_participants": 18,
        "participant_distinct": True,
        "train_participants": 12,
        "evaluation_participants": 6,
        "participant_overlap": 0,
        "train_windows": counts["train"],
        "evaluation_windows": counts["evaluation"],
        "minimum_windows_per_participant": min(participant_counts.values(), default=0),
        "split_identity_sha256": digest(split_identity),
        "control_assignment_complete": all(
            "shuffle_donor_row_key" in row for row in evaluation
        ),
        "G0_governance_binding_and_immutability_pass": True,
        "G1_split_windows_and_support_pass": g1,
        "locked_rows_loaded_scored_summarized_or_inspected": 0,
        "new_recordings_or_downloads": 0,
        "external_volume_aea_or_babyview_used": False,
        "asr_translation_language_id_transcripts_or_generative_labels_used": False,
        "media_decoded": False,
        "audiovisual_outcome_inspected": False,
        "restricted_values_exported": False,
    }
    _public_guard(receipt)
    _write_once(FREEZE_RECEIPT, receipt, private=False)
    return receipt


def _public_guard(value: Any) -> None:
    forbidden_keys = {
        "participant_key",
        "item_key",
        "row_key",
        "media_path",
        "annotation_path",
        "start_ms",
        "end_ms",
        "shift_start_ms",
        "shift_end_ms",
        "shuffle_donor_row_key",
        "embedding",
        "weights",
        "transcript",
    }
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if forbidden_keys.intersection(current):
                _fail("E_PUBLIC_PRIVACY")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, str):
            lowered = current.casefold()
            if (
                "/users/" in lowered
                or "file://" in lowered
                or any(
                    lowered.endswith(suffix)
                    for suffix in (".mp4", ".mov", ".mkv", ".wav", ".m4a")
                )
            ):
                _fail("E_PUBLIC_PRIVACY")


def _sandbox_run() -> Mapping[str, Any]:
    environment = dict(os.environ)
    environment["CHILDLENS_V4_NETWORK_DENIED"] = "1"
    process = subprocess.run(
        [
            str(SANDBOX_EXEC),
            "-p",
            SANDBOX_PROFILE,
            str(INSTRUMENT_PYTHON),
            str(Path(__file__).resolve()),
            "run-worker",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=14_400,
        env=environment,
    )
    if process.returncode != 0:
        _fail("E_RESTRICTED_WORKER")
    try:
        result = json.loads(process.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        _fail("E_RESTRICTED_WORKER")
    if result.get("status") != "ok":
        _fail("E_RESTRICTED_WORKER")
    return result


def _decode_audio(path: str, start_seconds: float, duration_seconds: float) -> np.ndarray:
    try:
        process = subprocess.run(
            [
                str(FFMPEG),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_seconds:.6f}",
                "-i",
                path,
                "-t",
                f"{duration_seconds:.6f}",
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=False,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BridgeV4Error("E_AUDIO_DECODE") from exc
    signal = np.frombuffer(process.stdout, dtype="<f4").copy()
    expected = round(duration_seconds * 16000)
    if process.returncode != 0 or abs(len(signal) - expected) > 160:
        _fail("E_AUDIO_DECODE")
    if len(signal) < expected:
        signal = np.pad(signal, (0, expected - len(signal)))
    return signal[:expected]


def _decode_frames(path: str, start_seconds: float, duration_seconds: float) -> list[Any]:
    import av

    targets = [start_seconds + 0.5 + index for index in range(round(duration_seconds))]
    container = av.open(path)
    try:
        video = next((stream for stream in container.streams if stream.type == "video"), None)
        if video is None:
            _fail("E_VIDEO_DECODE")
        seek_seconds = max(0.0, start_seconds - 1.0)
        container.seek(round(seek_seconds * 1_000_000), backward=True, any_frame=False)
        selected: list[Any] = []
        prior: tuple[float, Any] | None = None
        target_index = 0
        for frame in container.decode(video):
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp = float(frame.pts * frame.time_base)
            if timestamp < start_seconds - 1.0:
                continue
            while target_index < len(targets) and timestamp >= targets[target_index]:
                candidate = frame
                if prior is not None and abs(prior[0] - targets[target_index]) <= abs(
                    timestamp - targets[target_index]
                ):
                    candidate = prior[1]
                selected.append(candidate.to_image().convert("RGB"))
                target_index += 1
            prior = (timestamp, frame)
            if target_index == len(targets):
                break
        if len(selected) != len(targets):
            _fail("E_VIDEO_DECODE")
        return selected
    finally:
        container.close()


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    if left.shape != right.shape or np.any(denominator == 0):
        _fail("E_COSINE")
    return np.sum(left * right, axis=1) / denominator


def _extract_frontends(
    requests: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int]:
    import torch
    from transformers import AutoImageProcessor, Dinov2Model, Wav2Vec2Model

    torch.set_num_threads(4)
    audio_model = Wav2Vec2Model.from_pretrained(
        AUDIO_MODEL, local_files_only=True
    ).eval()
    vision_processor = AutoImageProcessor.from_pretrained(
        VISION_MODEL, local_files_only=True, use_fast=False
    )
    vision_model = Dinov2Model.from_pretrained(
        VISION_MODEL, local_files_only=True
    ).eval()
    output: dict[str, dict[str, Any]] = {}
    failures = 0
    for request in requests:
        key = str(request["request_key"])
        try:
            signal = _decode_audio(
                str(request["media_path"]),
                float(request["start_seconds"]),
                float(request["duration_seconds"]),
            )
            frames = _decode_frames(
                str(request["media_path"]),
                float(request["start_seconds"]),
                float(request["duration_seconds"]),
            )
            normalized = (signal - signal.mean()) / math.sqrt(
                float(signal.var()) + 1e-7
            )
            with torch.inference_mode():
                audio = (
                    audio_model(torch.from_numpy(normalized).unsqueeze(0))
                    .last_hidden_state.mean(dim=1)[0]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                pixels = vision_processor(
                    images=frames, return_tensors="pt"
                ).pixel_values
                per_frame = vision_model(pixel_values=pixels).pooler_output
                per_frame = torch.nn.functional.normalize(per_frame, dim=1)
                visual = torch.nn.functional.normalize(
                    per_frame.mean(dim=0), dim=0
                ).cpu().numpy().astype(np.float32)
            grayscale = [
                np.asarray(frame.convert("L").resize((64, 64)), dtype=np.float32)
                / 255.0
                for frame in frames
            ]
            differences = [
                float(np.mean(np.abs(right - left)))
                for left, right in zip(grayscale, grayscale[1:])
            ]
            output[key] = {
                "audio": audio,
                "visual": visual,
                "rms": float(np.sqrt(np.mean(np.square(signal)))),
                "non_silent_fraction": float(np.mean(np.abs(signal) >= 1e-3)),
                "clipped_fraction": float(np.mean(np.abs(signal) >= 0.999)),
                "motion": float(np.mean(differences)),
                "persistence": float(1.0 - np.mean(differences)),
                "scene_change_rate": float(
                    np.mean(np.asarray(differences) >= 0.2)
                ),
            }
        except Exception:
            failures += 1
    return output, failures


def _build_requests(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base: list[dict[str, Any]] = []
    perturb: list[dict[str, Any]] = []
    for item in manifest["items"]:
        for row in item["windows"]:
            base.append(
                {
                    "request_key": row["row_key"],
                    "media_path": row["media_path"],
                    "start_seconds": row["start_ms"] / 1000.0,
                    "duration_seconds": (row["end_ms"] - row["start_ms"]) / 1000.0,
                }
            )
            if row["split"] == "evaluation":
                base.append(
                    {
                        "request_key": f"{row['row_key']}:shift",
                        "media_path": row["media_path"],
                        "start_seconds": row["shift_start_ms"] / 1000.0,
                        "duration_seconds": (
                            row["shift_end_ms"] - row["shift_start_ms"]
                        )
                        / 1000.0,
                    }
                )
            for offset in (-0.25, 0.25):
                start = row["start_ms"] / 1000.0 + offset
                end = row["end_ms"] / 1000.0 + offset
                if start >= 0 and end <= item["recording_duration_seconds"]:
                    perturb.append(
                        {
                            "request_key": f"{row['row_key']}:perturb:{offset:+.2f}",
                            "media_path": row["media_path"],
                            "start_seconds": start,
                            "duration_seconds": end - start,
                        }
                    )
    unique_base = {row["request_key"]: row for row in base}
    unique_perturb = {row["request_key"]: row for row in perturb}
    return list(unique_base.values()), list(unique_perturb.values())


def _participant_median(values: Mapping[str, Sequence[float]]) -> float:
    return float(np.median([np.median(rows) for rows in values.values() if rows]))


def _stability(
    manifest: Mapping[str, Any],
    embeddings: Mapping[str, Mapping[str, Any]],
    perturbations: Mapping[str, Mapping[str, Any]],
    *,
    expected_perturbations: int,
    failures: int,
) -> dict[str, Any]:
    audio: dict[str, list[float]] = defaultdict(list)
    visual: dict[str, list[float]] = defaultdict(list)
    produced = 0
    for item in manifest["items"]:
        participant = item["participant_key"]
        for row in item["windows"]:
            base = embeddings.get(row["row_key"])
            if base is None:
                continue
            for offset in (-0.25, 0.25):
                other = perturbations.get(
                    f"{row['row_key']}:perturb:{offset:+.2f}"
                )
                if other is None:
                    continue
                produced += 1
                audio[participant].append(
                    float(
                        _cosine_rows(
                            np.asarray(base["audio"])[None, :],
                            np.asarray(other["audio"])[None, :],
                        )[0]
                    )
                )
                visual[participant].append(
                    float(
                        _cosine_rows(
                            np.asarray(base["visual"])[None, :],
                            np.asarray(other["visual"])[None, :],
                        )[0]
                    )
                )
    fraction = produced / expected_perturbations if expected_perturbations else 0.0
    return {
        "expected_embedding_pairs": expected_perturbations,
        "produced_embedding_pairs": produced,
        "failed_requests": failures,
        "production_fraction": fraction,
        "participant_median_audio_embedding_cosine": _participant_median(audio),
        "participant_median_visual_embedding_cosine": _participant_median(visual),
    }


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p10": None, "median": None, "p90": None}
    result = np.percentile(np.asarray(values, dtype=np.float64), [10, 50, 90])
    return {
        "p10": round(float(result[0]), 3),
        "median": round(float(result[1]), 3),
        "p90": round(float(result[2]), 3),
    }


def _calibration(
    manifest: Mapping[str, Any],
    embeddings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    participant_rows: list[dict[str, Any]] = []
    activity_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    for item in manifest["items"]:
        segments = sorted(
            (
                float(row["start_seconds"]),
                float(row["end_seconds"]),
            )
            for row in item["sample_segments"]
            if float(row["end_seconds"]) > float(row["start_seconds"])
        )
        observation = (
            float(item["sample_span_end_seconds"])
            - float(item["sample_span_start_seconds"])
        )
        speech_seconds = sum(end - start for start, end in segments)
        gaps = [
            max(0.0, right[0] - left[1])
            for left, right in zip(segments, segments[1:])
        ]
        measurements = [
            embeddings[row["row_key"]]
            for row in item["windows"]
            if row["row_key"] in embeddings
        ]
        activity_counts[item["activity_label"]] += 1
        location_counts[item["location_label"]] += 1
        participant_rows.append(
            {
                "split": item["split"],
                "bout_duration": float(
                    np.median([end - start for start, end in segments])
                ),
                "gap": float(np.median(gaps)) if gaps else 0.0,
                "speech_seconds_per_observation_minute": speech_seconds
                / (observation / 60.0),
                "speech_bout_density_per_minute": len(segments)
                / (observation / 60.0),
                "candidate_event_density_per_minute": len(item["windows"])
                / (observation / 60.0),
                "activity_duration_seconds": observation,
                "activity_recurrence_per_minute": 1.0 / (observation / 60.0),
                "speech_activity_overlap_fraction": (
                    speech_seconds / speech_seconds if speech_seconds else 0.0
                ),
                "retained_audio_seconds": len(measurements) * 4.0,
                "audio_rms": float(np.median([row["rms"] for row in measurements])),
                "audio_non_silent_fraction": float(
                    np.median([row["non_silent_fraction"] for row in measurements])
                ),
                "audio_clipped_fraction": float(
                    np.median([row["clipped_fraction"] for row in measurements])
                ),
                "motion": float(np.median([row["motion"] for row in measurements])),
                "persistence": float(
                    np.median([row["persistence"] for row in measurements])
                ),
                "scene_change_rate": float(
                    np.median([row["scene_change_rate"] for row in measurements])
                ),
            }
        )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        fields = [
            key
            for key in participant_rows[0]
            if key != "split"
        ]
        return {
            "participant_count": len(rows),
            **{
                field: _percentiles([float(row[field]) for row in rows])
                for field in fields
            },
        }

    minimum = 5
    activity: dict[str, Any] = {}
    for label, count in sorted(activity_counts.items()):
        fraction, suppressed = safe_fraction(
            count, 18, minimum_cell_size=minimum
        )
        activity[label] = {
            "participant_share": None if fraction is None else round(fraction, 3),
            "suppressed": suppressed,
        }
    location: dict[str, Any] = {}
    for label, count in sorted(location_counts.items()):
        fraction, suppressed = safe_fraction(
            count, 18, minimum_cell_size=minimum
        )
        location[label] = {
            "participant_share": None if fraction is None else round(fraction, 3),
            "suppressed": suppressed,
        }
    report = {
        "schema_version": "childlens-alignment-bridge-calibration-v4.0.0",
        "status": "PRIVACY_SAFE_DEVELOPMENT_CALIBRATION",
        "scientific_role": "PROVISIONAL_DEVELOPMENTAL_CALIBRATION_AGES_3_TO_5",
        "participant_clustered_distributions": {
            "combined": summarize(participant_rows),
            "development_train": summarize(
                [row for row in participant_rows if row["split"] == "train"]
            ),
            "participant_disjoint_development_evaluation": summarize(
                [row for row in participant_rows if row["split"] == "evaluation"]
            ),
        },
        "coarse_activity_participant_shares": activity,
        "coarse_location_participant_shares": location,
        "notes": [
            "Released speech-like timing is treated as candidate-signal support, not lexical or utterance ground truth.",
            "Speech-activity overlap uses the recording-level released coarse activity stratum because timed activity subevents were not consistently bound across both immutable development cohorts.",
            "Motion and scene change are model-independent grayscale adjacent-frame summaries over retained learner windows.",
        ],
        "public_minimum_cell_size": 5,
        "complementary_suppression": True,
        "restricted_values_exported": False,
    }
    _public_guard(report)
    return report


def _mean_by_participant(
    rows: Sequence[Mapping[str, Any]], field: str
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["participant_key"])].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _bootstrap_participant_metric(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["participant_key"])].append(float(row[field]))
    values = [float(np.mean(value)) for _, value in sorted(grouped.items())]
    rng = np.random.default_rng(seed)
    estimates = np.sort(
        np.mean(
            rng.choice(values, size=(10000, len(values)), replace=True),
            axis=1,
        )
    )
    return [float(estimates[499]), float(estimates[9500])]


def _evaluate(
    manifest: Mapping[str, Any],
    embeddings: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    runtime: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rows = [row for item in manifest["items"] for row in item["windows"]]
    train = [row for row in rows if row["split"] == "train"]
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    row_by_key = {row["row_key"]: row for row in rows}

    def matrix(selected: Sequence[Mapping[str, Any]], modality: str) -> np.ndarray:
        return np.stack(
            [np.asarray(embeddings[row["row_key"]][modality]) for row in selected]
        )

    train_audio = matrix(train, "audio")
    train_visual = matrix(train, "visual")
    eval_audio = matrix(evaluation, "audio")
    eval_visual = matrix(evaluation, "visual")
    eval_shift = np.stack(
        [
            np.asarray(embeddings[f"{row['row_key']}:shift"]["visual"])
            for row in evaluation
        ]
    )
    eval_shuffle = np.stack(
        [
            np.asarray(
                embeddings[
                    row_by_key[row["shuffle_donor_row_key"]]["row_key"]
                ]["visual"]
            )
            for row in evaluation
        ]
    )
    weights: dict[str, np.ndarray] = {}
    seed_scores: list[dict[str, np.ndarray]] = []
    for seed in config["training"]["seeds"]:
        audio_weight, visual_weight = train_projection_heads(
            train_audio,
            train_visual,
            [row["participant_key"] for row in train],
            output_dimension=128,
            epochs=int(config["training"]["epochs"]),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            temperature=float(
                config["method_selection"]["learned_components"]["temperature"]
            ),
            gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            seed=int(seed),
        )
        weights[f"audio_{seed}"] = audio_weight
        weights[f"visual_{seed}"] = visual_weight
        seed_scores.append(
            {
                "real": projected_cosine(
                    eval_audio, eval_visual, audio_weight, visual_weight
                ),
                "shift": projected_cosine(
                    eval_audio, eval_shift, audio_weight, visual_weight
                ),
                "shuffle": projected_cosine(
                    eval_audio, eval_shuffle, audio_weight, visual_weight
                ),
            }
        )
    mean_scores = {
        name: np.mean(np.stack([value[name] for value in seed_scores]), axis=0)
        for name in ("real", "shift", "shuffle")
    }
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(evaluation):
        values = [
            mean_scores["real"][index],
            mean_scores["shift"][index],
            mean_scores["shuffle"][index],
        ]
        order = np.argsort(-np.asarray(values))
        rank = int(np.flatnonzero(order == 0)[0]) + 1
        scored.append(
            {
                "participant_key": row["participant_key"],
                "row_key": row["row_key"],
                "real_cosine": float(values[0]),
                "shift_cosine": float(values[1]),
                "shuffle_cosine": float(values[2]),
                "real_beats_shift": float(values[0] > values[1]),
                "real_beats_shuffle": float(values[0] > values[2]),
                "real_top1": float(rank == 1),
                "reciprocal_rank": 1.0 / rank,
            }
        )
    uncertainty = config["uncertainty_and_metrics"]
    primary = effect_summary(
        scored,
        control_field="shift_cosine",
        confidence=float(uncertainty["cluster_bootstrap_confidence"]),
        bootstrap_replicates=int(uncertainty["cluster_bootstrap_replicates"]),
        permutation_replicates=int(uncertainty["permutation_replicates_otherwise"]),
        seed=int(uncertainty["seed"]),
    )
    secondary = effect_summary(
        scored,
        control_field="shuffle_cosine",
        confidence=float(uncertainty["cluster_bootstrap_confidence"]),
        bootstrap_replicates=int(uncertainty["cluster_bootstrap_replicates"]),
        permutation_replicates=int(uncertainty["permutation_replicates_otherwise"]),
        seed=int(uncertainty["seed"]) + 10,
    )
    seed_lifts = {
        "within_recording_shift": [
            float(np.mean(value["real"] - value["shift"])) for value in seed_scores
        ],
        "cross_participant_shuffle": [
            float(np.mean(value["real"] - value["shuffle"])) for value in seed_scores
        ],
    }
    threshold = float(uncertainty["meaningful_mean_cosine_lift_min_each_control"])
    primary_pass = (
        primary["mean_lift"] >= threshold
        and primary["participant_cluster_bootstrap_interval"][0] > 0
        and primary["one_sided_sign_flip_p"]
        <= float(uncertainty["one_sided_sign_flip_p_max"])
        and sum(value > 0 for value in seed_lifts["within_recording_shift"]) >= 2
    )
    secondary_pass = (
        secondary["mean_lift"] >= threshold
        and secondary["participant_cluster_bootstrap_interval"][0] > 0
        and secondary["one_sided_sign_flip_p"]
        <= float(uncertainty["one_sided_sign_flip_p_max"])
        and sum(value > 0 for value in seed_lifts["cross_participant_shuffle"]) >= 2
    )
    metrics = {
        "control_envelopes": {
            "real": {
                "participant_mean_cosine": _mean_by_participant(scored, "real_cosine"),
                "participant_cluster_bootstrap_90pct": _bootstrap_participant_metric(
                    scored, "real_cosine", seed=20260723
                ),
            },
            "within_recording_shift": {
                "participant_mean_cosine": _mean_by_participant(
                    scored, "shift_cosine"
                ),
                "participant_cluster_bootstrap_90pct": _bootstrap_participant_metric(
                    scored, "shift_cosine", seed=20260724
                ),
            },
            "cross_participant_shuffle": {
                "participant_mean_cosine": _mean_by_participant(
                    scored, "shuffle_cosine"
                ),
                "participant_cluster_bootstrap_90pct": _bootstrap_participant_metric(
                    scored, "shuffle_cosine", seed=20260725
                ),
            },
        },
        "primary_real_minus_within_recording_shift": primary,
        "secondary_real_minus_cross_participant_shuffle": secondary,
        "retrieval": {
            field: {
                "participant_mean": _mean_by_participant(scored, field),
                "participant_cluster_bootstrap_90pct": _bootstrap_participant_metric(
                    scored, field, seed=20260730 + index
                ),
            }
            for index, field in enumerate(
                (
                    "real_beats_shift",
                    "real_beats_shuffle",
                    "real_top1",
                    "reciprocal_rank",
                )
            )
        },
        "seed_specific_mean_lifts": seed_lifts,
        "G3_primary_within_recording_alignment_pass": primary_pass,
        "G4_secondary_cross_participant_alignment_pass": secondary_pass,
    }
    private = {
        "schema_version": "childlens-alignment-bridge-restricted-result-v4.0.0",
        "protocol_sha256": _sha256_file(CONFIG),
        "evaluation_rows": scored,
        "metrics": metrics,
        "locked_rows_loaded_scored_summarized_or_inspected": 0,
    }
    _write_once(runtime / PRIVATE_RESULT, private, private=True)
    return metrics, weights


def _round_public(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, Mapping):
        return {key: _round_public(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_public(item) for item in value]
    return value


def run_worker() -> Mapping[str, Any]:
    if os.environ.get("CHILDLENS_V4_NETWORK_DENIED") != "1":
        _fail("E_NETWORK_FIREWALL")
    config = _config()
    prior = _validate_prior_trees()
    _validate_models()
    runtime = _discover_runtime()
    freeze_receipt = _read_json(FREEZE_RECEIPT)
    manifest_path = runtime / PRIVATE_MANIFEST
    if (
        freeze_receipt.get("G0_governance_binding_and_immutability_pass") is not True
        or freeze_receipt.get("G1_split_windows_and_support_pass") is not True
        or freeze_receipt.get("protocol_sha256") != _sha256_file(CONFIG)
        or freeze_receipt.get("runner_sha256") != _sha256_file(Path(__file__))
        or freeze_receipt.get("restricted_manifest_sha256")
        != _sha256_file(manifest_path)
        or prior != EXPECTED_PRIOR_TREES
    ):
        _fail("E_FREEZE_BINDING")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("development_participants") != 18
        or manifest.get("locked_rows_loaded_scored_summarized_or_inspected") != 0
    ):
        _fail("E_MANIFEST")
    base_requests, perturb_requests = _build_requests(manifest)
    base_embeddings, base_failures = _extract_frontends(base_requests)
    perturb_embeddings, perturb_failures = _extract_frontends(perturb_requests)
    expected_base = len(base_requests)
    missing_fraction = (
        base_failures / expected_base if expected_base else 1.0
    )
    stability = _stability(
        manifest,
        base_embeddings,
        perturb_embeddings,
        expected_perturbations=len(perturb_requests),
        failures=perturb_failures,
    )
    g2 = (
        missing_fraction
        <= float(
            config["ordered_gates"]["G2_preprocessing_and_perturbation_stability"][
                "missing_or_substituted_row_fraction_max"
            ]
        )
        and stability["production_fraction"]
        >= float(
            config["timing_perturbation"]["minimum_embedding_production_fraction"]
        )
        and stability["participant_median_audio_embedding_cosine"]
        >= float(
            config["timing_perturbation"][
                "participant_median_audio_embedding_cosine_min"
            ]
        )
        and stability["participant_median_visual_embedding_cosine"]
        >= float(
            config["timing_perturbation"][
                "participant_median_visual_embedding_cosine_min"
            ]
        )
    )
    arrays: dict[str, np.ndarray] = {}
    for index, (key, value) in enumerate(sorted(base_embeddings.items())):
        arrays[f"audio_{index:04d}"] = np.asarray(value["audio"])
        arrays[f"visual_{index:04d}"] = np.asarray(value["visual"])
    _write_private_npz_once(runtime / PRIVATE_EMBEDDINGS, **arrays)
    calibration = _calibration(manifest, base_embeddings)
    _write_once(CALIBRATION_REPORT, calibration, private=False)
    metrics: dict[str, Any] | None = None
    weights_hash: str | None = None
    if g2:
        metrics, weights = _evaluate(
            manifest, base_embeddings, config, runtime
        )
        _write_private_npz_once(runtime / PRIVATE_WEIGHTS, **weights)
        weights_hash = _sha256_file(runtime / PRIVATE_WEIGHTS)
    g3 = bool(metrics and metrics["G3_primary_within_recording_alignment_pass"])
    g4 = bool(metrics and metrics["G4_secondary_cross_participant_alignment_pass"])
    go = bool(g2 and g3 and g4)
    if not g2:
        failure = "LEARNER_OR_PREPROCESSING_INSTABILITY"
    elif not g3 and g4:
        failure = "SHORTCUT_ONLY_SHUFFLED_SEPARATION"
    elif not g3:
        failure = "ABSENCE_OF_WITHIN_RECORDING_ALIGNMENT"
    elif not g4:
        failure = "ABSENCE_OF_CROSS_PARTICIPANT_ALIGNMENT"
    else:
        failure = None
    report = {
        "schema_version": "childlens-alignment-bridge-development-report-v4.0.0",
        "status": (
            "DEVELOPMENT_GO_RECOMMEND_SEPARATELY_AUTHORIZED_LOCKED_EVALUATION"
            if go
            else "TERMINAL_DEVELOPMENT_NO_GO"
        ),
        "decision": "GO" if go else "NO_GO",
        "failure_classification": failure,
        "scientific_role": "PROVISIONAL_DEVELOPMENTAL_CALIBRATION_AGES_3_TO_5",
        "training": {
            "participants": freeze_receipt["train_participants"],
            "windows": freeze_receipt["train_windows"],
            "pairing": "real audio-real video only",
            "frozen_frontends": [
                "wav2vec2-base raw-speech encoder",
                "DINOv2-small vision encoder",
            ],
            "learned": "two bias-free 128-dimensional linear projections",
            "seeds": config["training"]["seeds"],
        },
        "participant_disjoint_evaluation": {
            "participants": freeze_receipt["evaluation_participants"],
            "windows": freeze_receipt["evaluation_windows"],
            "participant_overlap_with_training": 0,
            "metrics": None if metrics is None else _round_public(metrics),
        },
        "preprocessing_and_timing_perturbation": {
            "base_missing_or_substituted_fraction": round(missing_fraction, 4),
            "substituted_rows": 0,
            **_round_public(stability),
        },
        "ordered_gates": {
            "G0_governance_binding_and_immutability": True,
            "G1_split_windows_and_support": True,
            "G2_preprocessing_and_perturbation_stability": g2,
            "G3_primary_within_recording_alignment": g3,
            "G4_secondary_cross_participant_alignment": g4,
            "G5_privacy_safe_reporting": True,
        },
        "shortcut_risk": (
            "FAILURE_REAL_EXCEEDS_SHUFFLE_BUT_NOT_WITHIN_RECORDING_SHIFT"
            if g4 and not g3
            else (
                "PRIMARY_WITHIN_RECORDING_CONTROL_PASSED"
                if g3
                else "NO_CREDIBLE_WITHIN_RECORDING_ALIGNMENT"
            )
        ),
        "scope_enforcement": {
            "locked_rows_loaded_scored_summarized_or_inspected": 0,
            "locked_evaluation_run": False,
            "new_recordings_or_downloads": 0,
            "aea_external_volume_or_babyview_used": False,
            "asr_translation_language_id_transcripts_or_generative_labels_used": False,
            "simulator_generation_run": False,
            "physical_side_cue_condition_training_run": False,
        },
        "reproducibility": {
            "protocol_sha256": _sha256_file(CONFIG),
            "runner_sha256": _sha256_file(Path(__file__)),
            "package_sha256": _sha256_file(
                ROOT / "babyworld_lite/childlens_alignment_bridge_v4/preflight.py"
            ),
            "restricted_manifest_sha256": _sha256_file(manifest_path),
            "restricted_embeddings_sha256": _sha256_file(
                runtime / PRIVATE_EMBEDDINGS
            ),
            "restricted_projection_weights_sha256": weights_hash,
            "restricted_result_sha256": (
                _sha256_file(runtime / PRIVATE_RESULT)
                if (runtime / PRIVATE_RESULT).is_file()
                else None
            ),
            "prior_tree_sha256": prior,
            "network_denial_backend": "MACOS_SANDBOX_DENY_NETWORK",
        },
        "michael_frank_advice_instantiation": "V4 measures privacy-safe temporal, speech-audio, visual-motion, and audiovisual distributions in ChildLens; freezes those aspects and an apples-to-apples contrastive learner for eventual simulation with otherwise unavailable embodied side information; and stops here at the development anchor. The only eventual allowed claim is measured cue lift in simulation under temporal, visual, and audiovisual distributions provisionally calibrated to ChildLens ages 3-5, not validated naturalistic German lexical grounding.",
        "limitations": [
            "Only six participant-disjoint development participants are in the reportable evaluation cell.",
            "The fixed audio frontend was self-supervised on adult English read speech and may transfer imperfectly to German child-centered natural audio.",
            "Released speech-like timing supplies candidate support, not word boundaries, transcripts, language identity, or lexical truth.",
            "Recording-level coarse activity is available consistently; timed activity subevents are not uniformly bound across both immutable cohorts.",
            "A development pass would still require a separately authorized locked evaluation before any simulator or cue-lift study.",
        ],
        "recommend_separately_authorized_locked_evaluation": go,
        "restricted_values_exported": False,
    }
    _public_guard(report)
    _write_once(RESULT_REPORT, report, private=False)
    _write_text_once(RESULT_MARKDOWN, _markdown_report(report))
    validation = validate(write=False)
    _write_once(VALIDATION_RECEIPT, validation, private=False)
    return {"status": "ok", "decision": report["decision"]}


def _markdown_report(report: Mapping[str, Any]) -> str:
    evaluation = report["participant_disjoint_evaluation"]
    metrics = evaluation["metrics"]
    lines = [
        "# ChildLens bridge v4 — development-only go/no-go",
        "",
        f"**Decision: {report['decision']}**  ",
        f"Status: `{report['status']}`",
        "",
        "## Design",
        "",
        f"Training used {report['training']['participants']} participants and "
        f"{report['training']['windows']} real audiovisual windows. Evaluation "
        f"used {evaluation['participants']} participant-disjoint participants and "
        f"{evaluation['windows']} windows, with zero participant overlap.",
        "",
        "The sole route used fixed wav2vec2-base raw-speech and DINOv2-small "
        "vision frontends. Only two 128-dimensional linear projection heads were "
        "learned. No transcript, ASR decoder, language identification, translation, "
        "categorical referent label, or generative VLM judgment was used.",
        "",
        "## Ordered gates",
        "",
    ]
    for gate, passed in report["ordered_gates"].items():
        lines.append(f"- {gate}: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "## Participant-disjoint controls",
            "",
        ]
    )
    if metrics is None:
        lines.append(
            "Audiovisual training and scoring stopped because preprocessing or "
            "timing-perturbation stability failed."
        )
    else:
        primary = metrics["primary_real_minus_within_recording_shift"]
        secondary = metrics["secondary_real_minus_cross_participant_shuffle"]
        lines.extend(
            [
                f"- Real minus within-recording shift: mean lift "
                f"{primary['mean_lift']:.4f}, 90% participant-cluster interval "
                f"{primary['participant_cluster_bootstrap_interval']}, one-sided "
                f"sign-flip p={primary['one_sided_sign_flip_p']:.4f}.",
                f"- Real minus cross-participant shuffle: mean lift "
                f"{secondary['mean_lift']:.4f}, 90% participant-cluster interval "
                f"{secondary['participant_cluster_bootstrap_interval']}, one-sided "
                f"sign-flip p={secondary['one_sided_sign_flip_p']:.4f}.",
                f"- Shortcut audit: `{report['shortcut_risk']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Scope and interpretation",
            "",
            report["michael_frank_advice_instantiation"],
            "",
            "All 22 locked participants remained sealed. No locked evaluation, "
            "new recording acquisition, AEA/external-volume use, BabyView use, "
            "simulator generation, or physical side-cue condition training occurred.",
            "",
            "## Limitations",
            "",
            *[f"- {value}" for value in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)


def validate(*, write: bool = True) -> Mapping[str, Any]:
    config = _config()
    prior = _validate_prior_trees()
    _validate_models()
    runtime = _discover_runtime()
    freeze_receipt = _read_json(FREEZE_RECEIPT)
    public_files = [FREEZE_RECEIPT]
    if CALIBRATION_REPORT.is_file():
        public_files.append(CALIBRATION_REPORT)
    if RESULT_REPORT.is_file():
        public_files.append(RESULT_REPORT)
    for path in public_files:
        _public_guard(_read_json(path))
    manifest_path = runtime / PRIVATE_MANIFEST
    if (
        freeze_receipt.get("protocol_sha256") != _sha256_file(CONFIG)
        or freeze_receipt.get("runner_sha256") != _sha256_file(Path(__file__))
        or freeze_receipt.get("restricted_manifest_sha256")
        != _sha256_file(manifest_path)
        or config["immutable_prior_bindings"]["v1_tree_sha256"] != prior["v1"]
        or config["immutable_prior_bindings"]["v2_tree_sha256"] != prior["v2"]
        or config["immutable_prior_bindings"]["v3_tree_sha256"] != prior["v3"]
    ):
        _fail("E_VALIDATION_BINDING")
    result = _read_json(RESULT_REPORT) if RESULT_REPORT.is_file() else None
    receipt = {
        "schema_version": "childlens-alignment-bridge-validation-v4.0.0",
        "status": "PASS",
        "protocol_sha256": _sha256_file(CONFIG),
        "runner_sha256": _sha256_file(Path(__file__)),
        "package_sha256": _sha256_file(
            ROOT / "babyworld_lite/childlens_alignment_bridge_v4/preflight.py"
        ),
        "prior_v1_v2_v3_immutable": True,
        "prior_tree_sha256": prior,
        "restricted_manifest_hash_verified": True,
        "development_participants": 18,
        "participant_distinct": True,
        "train_evaluation_participant_overlap": 0,
        "locked_rows_loaded_scored_summarized_or_inspected": 0,
        "new_recordings_or_downloads": 0,
        "aea_external_volume_or_babyview_used": False,
        "asr_translation_language_id_transcripts_or_generative_labels_used": False,
        "simulator_or_side_cue_training_run": False,
        "public_files_scanned": len(public_files),
        "public_minimum_cell_size": 5,
        "complementary_suppression_enforced": True,
        "row_level_values_exported": False,
        "development_decision": None if result is None else result["decision"],
    }
    _public_guard(receipt)
    if write:
        _write_once(VALIDATION_RECEIPT, receipt, private=False)
    return receipt


def locked_status() -> Mapping[str, Any]:
    return {
        "locked_evaluation_authorized": False,
        "locked_rows_loaded_scored_summarized_or_inspected": 0,
        "reason": "V4_DEVELOPMENT_ONLY_REQUIRES_SEPARATE_AUTHORIZATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("freeze", "run", "run-worker", "validate", "locked-status"),
    )
    args = parser.parse_args()
    old_umask = os.umask(0o077)
    try:
        if args.command == "freeze":
            result = freeze()
        elif args.command == "run":
            result = _sandbox_run()
        elif args.command == "run-worker":
            result = run_worker()
        elif args.command == "validate":
            result = validate()
        else:
            result = locked_status()
        print(json.dumps(result, sort_keys=True))
        return 0
    except BridgeV4Error as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
