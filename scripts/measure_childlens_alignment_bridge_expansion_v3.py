#!/usr/bin/env python3
"""Measure and aggregate the frozen ChildLens ten-participant expansion v3."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_expansion_v3.json"
V2_CONFIG = ROOT / "configs/childlens_alignment_bridge_remediation_v2.json"
V2_RUNNER = ROOT / "scripts/run_childlens_alignment_bridge_remediation_v2.py"
SELECTION_RECEIPT = (
    ROOT
    / "output/childlens_alignment_bridge_expansion_v3/"
    "selection_freeze_receipt.json"
)
ACQUISITION_RECEIPT = (
    ROOT
    / "output/childlens_alignment_bridge_expansion_v3/"
    "acquisition_receipt.json"
)
PUBLIC_ROOT = ROOT / "output/childlens_alignment_bridge_expansion_v3"
MEASUREMENT_FREEZE_RECEIPT = PUBLIC_ROOT / "measurement_freeze_receipt.json"
SEGMENT_RECEIPT = PUBLIC_ROOT / "segment_freeze_receipt.json"
REPORT = PUBLIC_ROOT / "measurement_expansion_report_v3_0_1.json"

PRIVATE_RELATIVE = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_expansion_v3"
)
PRIVATE_PLAN = PRIVATE_RELATIVE / "restricted_expansion_plan.json"
PRIVATE_CHECKPOINT = PRIVATE_RELATIVE / "restricted_acquisition_checkpoint.json"
PRIVATE_MEASUREMENT = PRIVATE_RELATIVE / "restricted_measurement_manifest.json"
PRIVATE_SEGMENTS = PRIVATE_RELATIVE / "restricted_frozen_segment_manifest.json"
PRIVATE_WORKER_SEGMENT_RECEIPT = (
    PRIVATE_RELATIVE / "restricted_segment_worker_receipt.json"
)
PRIVATE_WORKER_FREEZE_RECEIPT = (
    PRIVATE_RELATIVE / "restricted_measurement_worker_receipt.json"
)
PRIVATE_NEW_RESULT = PRIVATE_RELATIVE / "restricted_new_cohort_result.json"
PRIVATE_COMBINED_RESULT = PRIVATE_RELATIVE / "restricted_combined_result.json"
V2_RESULT = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_remediation_v2/"
    "restricted_remediation_result.json"
)

PROTOCOL_SHA256 = "787f64eba92a6a2f206e09a447b2f595691230349ba8f17c800faa0e50108f02"
V2_CONFIG_SHA256 = "0fd819db4509d23abe2ea195e95a988307871a937aefaec9259f1e2dbd94af97"
V2_RUNNER_SHA256 = "96544bcb8253f60dfc86833ad4fac84ee80a1649fa872a60cb6b0df8d2243a16"
SELECTION_RECEIPT_SHA256 = (
    "9decd2b5ad832c29c24d97b416711032df0950aa39fcf82302a75be0cc91d081"
)
ACQUISITION_RECEIPT_SHA256 = (
    "bc46b97300230133254e8e866cb43a36e9652f0ba353ea40578c7f60770d9862"
)
RESTRICTED_PLAN_SHA256 = (
    "796eccc748cd61590bd0c9d4499e92e81277327f11e6ad4a19ce106afa4b4cb6"
)
RESTRICTED_CHECKPOINT_SHA256 = (
    "9acd8c71ac193bea74166a4b4025ac8c96b6bed9944101a6257848bd69ba635e"
)
V2_RESULT_SHA256 = "ab61da2ecdedaf33809591df427afd913bea45763ce65656938d2d463814c44f"
NEW_COUNT = 10
OLD_COUNT = 8
COMBINED_COUNT = 18
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"
INSTRUMENT_PYTHON = (
    Path.home()
    / "Library/Application Support/ChildLens Instruments/v1.3/venv/bin/python3.10"
)
PATH_TOKEN = __import__("re").compile(r"(?i)(?:/users/|file://|\\users\\)")
MEDIA_TOKEN = __import__("re").compile(
    r"(?i)\b\S+\.(?:mp4|mov|mkv|avi|webm|wav|m4a)\b"
)


class MeasurementError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise MeasurementError(code)


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


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
    except OSError:
        _fail("E_FILE")
    return value.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("E_FILE")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_FILE")


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
    payload = _canonical(value) + b"\n"
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
    except OSError:
        _fail("E_WRITE")
    finally:
        if pending.exists():
            pending.unlink()


def _discover_runtime() -> Path:
    candidates: list[Path] = []
    for hidden in ROOT.parent.iterdir():
        if (
            hidden.name.startswith(".")
            and "childlens" in hidden.name.casefold()
            and _private_directory(hidden)
        ):
            for manifest in hidden.rglob("restricted_manifest/preselection_manifest.json"):
                if _private_file(manifest):
                    candidates.append(manifest.parent)
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) != 1:
        _fail("E_RUNTIME_DISCOVERY")
    runtime = unique[0]
    if not _private_directory(runtime) or not _private_file(
        runtime / ".metadata_never_index"
    ):
        _fail("E_RUNTIME_CONTROL")
    return runtime


def _load_v2():
    if (
        _sha256_file(V2_RUNNER) != V2_RUNNER_SHA256
        or _sha256_file(V2_CONFIG) != V2_CONFIG_SHA256
    ):
        _fail("E_V2_CODE_BINDING")
    name = "childlens_alignment_bridge_v2_bound_for_v3"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V2_RUNNER)
    if spec is None or spec.loader is None:
        _fail("E_V2_IMPORT")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        _fail("E_V2_IMPORT")
    return module


def _patch_v2(module: Any, runtime: Path) -> None:
    v2_config = _read_json(V2_CONFIG)
    module.PRIVATE_RELATIVE = PRIVATE_RELATIVE
    module.PRIVATE_DEVELOPMENT = PRIVATE_MEASUREMENT
    module.PRIVATE_SEGMENTS = PRIVATE_SEGMENTS
    module.PRIVATE_RESULT = PRIVATE_NEW_RESULT
    module.FREEZE_RECEIPT = runtime / PRIVATE_WORKER_FREEZE_RECEIPT
    module.SEGMENT_RECEIPT = runtime / PRIVATE_WORKER_SEGMENT_RECEIPT
    module.REPORT_JSON = REPORT
    module._config = lambda: (v2_config, PROTOCOL_SHA256)


def _validate_public_bindings() -> None:
    if (
        _sha256_file(CONFIG) != PROTOCOL_SHA256
        or _sha256_file(SELECTION_RECEIPT) != SELECTION_RECEIPT_SHA256
        or _sha256_file(ACQUISITION_RECEIPT) != ACQUISITION_RECEIPT_SHA256
        or not SANDBOX_EXEC.is_file()
        or not INSTRUMENT_PYTHON.is_file()
    ):
        _fail("E_PUBLIC_BINDING")
    config = _read_json(CONFIG)
    if (
        config.get("status") != "FROZEN_BEFORE_EXPANSION_SELECTION_OR_MEDIA_OPEN"
        or config.get("scope", {}).get("combined_development_participants")
        != COMBINED_COUNT
        or config.get("scope", {}).get("locked_alignment_allowed") is not False
    ):
        _fail("E_PROTOCOL")


def freeze_measurement() -> Mapping[str, Any]:
    _validate_public_bindings()
    runtime = _discover_runtime()
    plan_path = runtime / PRIVATE_PLAN
    checkpoint_path = runtime / PRIVATE_CHECKPOINT
    old_result_path = runtime / V2_RESULT
    if (
        _sha256_file(plan_path) != RESTRICTED_PLAN_SHA256
        or _sha256_file(checkpoint_path) != RESTRICTED_CHECKPOINT_SHA256
        or _sha256_file(old_result_path) != V2_RESULT_SHA256
    ):
        _fail("E_RESTRICTED_BINDING")
    plan = _read_json(plan_path)
    checkpoint = _read_json(checkpoint_path)
    if (
        not isinstance(plan.get("items"), list)
        or not isinstance(checkpoint.get("items"), list)
        or len(plan["items"]) != NEW_COUNT
        or len(checkpoint["items"]) != NEW_COUNT
        or checkpoint.get("status") != "COMPLETE"
    ):
        _fail("E_ACQUISITION")
    module = _load_v2()
    module._validate_instruments()
    items: list[dict[str, Any]] = []
    participants: set[str] = set()
    for row, acquired in zip(plan["items"], checkpoint["items"]):
        if (
            row.get("selection_rank") != acquired.get("selection_rank")
            or acquired.get("status") != "COMPLETE"
            or not isinstance(row.get("participant_key"), str)
            or row["participant_key"] in participants
            or not isinstance(row.get("annotation_rows"), list)
            or len(row["annotation_rows"]) != 1
        ):
            _fail("E_MEASUREMENT_MANIFEST")
        media_path = (runtime / str(acquired["clip_relative_path"])).resolve()
        annotation_path = Path(row["annotation_rows"][0]["locator"]).resolve()
        if (
            not _inside(media_path, runtime)
            or not _private_file(media_path)
            or _sha256_file(media_path) != acquired["clip_sha256"]
            or not _inside(annotation_path, runtime.parent)
            or not _private_file(annotation_path)
            or _sha256_file(annotation_path)
            != row["annotation_rows"][0]["sha256"]
        ):
            _fail("E_MEASUREMENT_BINDING")
        clip_start = int(row["clip_source_start_ms"])
        segments = [
            {
                "start_seconds": (int(segment["start_ms"]) - clip_start) / 1000.0,
                "end_seconds": (int(segment["end_ms"]) - clip_start) / 1000.0,
            }
            for segment in row["sample_segments_source_ms"]
        ]
        span_start = min(segment["start_seconds"] for segment in segments)
        span_end = max(segment["end_seconds"] for segment in segments)
        if span_start < 0 or span_end <= span_start:
            _fail("E_MEASUREMENT_MANIFEST")
        items.append(
            {
                "cohort": "expansion_v3",
                "participant_key": row["participant_key"],
                "item_key": acquired["clip_sha256"],
                "media_path": str(media_path),
                "expected_media_sha256": acquired["clip_sha256"],
                "annotation_path": str(annotation_path),
                "annotation_sha256": row["annotation_rows"][0]["sha256"],
                "sample_segments": segments,
                "sample_span_start_seconds": span_start,
                "sample_span_end_seconds": span_end,
                "source_offset_seconds": clip_start / 1000.0,
                "selection_rank": row["selection_rank"],
            }
        )
        participants.add(row["participant_key"])
    if len(items) != NEW_COUNT or len(participants) != NEW_COUNT:
        _fail("E_MEASUREMENT_MANIFEST")
    manifest = {
        "schema_version": "childlens-alignment-bridge-expansion-measurement-v3.0.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "restricted_plan_sha256": RESTRICTED_PLAN_SHA256,
        "restricted_checkpoint_sha256": RESTRICTED_CHECKPOINT_SHA256,
        "v2_result_sha256": V2_RESULT_SHA256,
        "development_count": NEW_COUNT,
        "locked_rows_copied_or_evaluated": 0,
        "items": items,
    }
    manifest_path = runtime / PRIVATE_MEASUREMENT
    _write_once(manifest_path, manifest, private=True)
    _write_once(
        runtime / PRIVATE_WORKER_FREEZE_RECEIPT,
        {
            "schema_version": "childlens-alignment-bridge-expansion-v2-worker-binding-v3.0.0",
            "protocol_sha256": PROTOCOL_SHA256,
            "restricted_development_manifest_sha256": _sha256_file(manifest_path),
            "development_participants": NEW_COUNT,
            "locked_rows_copied_or_evaluated": 0,
        },
        private=True,
    )
    receipt = {
        "schema_version": "childlens-alignment-bridge-expansion-measurement-freeze-v3.0.0",
        "status": "FROZEN_BEFORE_SEGMENTATION_OR_ASR",
        "protocol_sha256": PROTOCOL_SHA256,
        "v2_config_sha256": V2_CONFIG_SHA256,
        "v2_runner_sha256": V2_RUNNER_SHA256,
        "restricted_measurement_manifest_sha256": _sha256_file(manifest_path),
        "restricted_v2_result_sha256": V2_RESULT_SHA256,
        "new_development_participants": NEW_COUNT,
        "prior_development_participants_reused_without_rerunning": OLD_COUNT,
        "combined_development_participants": COMBINED_COUNT,
        "all_instrument_hashes_verified": True,
        "locked_rows_copied_or_evaluated": 0,
        "alignment_scoring_authorized": False,
        "restricted_values_exported": False,
    }
    _write_once(MEASUREMENT_FREEZE_RECEIPT, receipt, private=False)
    return receipt


def _sandbox_reexec(command: str, timeout: int) -> Mapping[str, Any]:
    environment = {
        **os.environ,
        "CHILDLENS_V2_NETWORK_DENIED": "1",
        "CHILDLENS_V3_NETWORK_DENIED": "1",
    }
    process = subprocess.run(
        [
            str(SANDBOX_EXEC),
            "-p",
            SANDBOX_PROFILE,
            str(INSTRUMENT_PYTHON),
            str(Path(__file__).resolve()),
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    if process.returncode != 0:
        _fail("E_RESTRICTED_WORKER")
    try:
        value = json.loads(process.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        _fail("E_RESTRICTED_WORKER")
    if value.get("status") != "ok":
        _fail("E_RESTRICTED_WORKER")
    return value


def prepare_segments() -> Mapping[str, Any]:
    _validate_public_bindings()
    if not MEASUREMENT_FREEZE_RECEIPT.is_file():
        _fail("E_MEASUREMENT_NOT_FROZEN")
    return _sandbox_reexec("_prepare-worker", 14_400)


def _prepare_worker() -> Mapping[str, Any]:
    if os.environ.get("CHILDLENS_V3_NETWORK_DENIED") != "1":
        _fail("E_NETWORK_DENIAL")
    runtime = _discover_runtime()
    module = _load_v2()
    _patch_v2(module, runtime)
    module._prepare_segments_restricted()
    worker_receipt = _read_json(runtime / PRIVATE_WORKER_SEGMENT_RECEIPT)
    segment_path = runtime / PRIVATE_SEGMENTS
    if (
        worker_receipt.get("development_participants") != NEW_COUNT
        or worker_receipt.get("locked_rows_loaded") != 0
        or not _private_file(segment_path)
    ):
        _fail("E_SEGMENT_RESULT")
    receipt = {
        "schema_version": "childlens-alignment-bridge-expansion-segment-freeze-v3.0.0",
        "status": "NEW_TEN_SHARED_SEGMENTS_FROZEN_BEFORE_ASR",
        "protocol_sha256": PROTOCOL_SHA256,
        "restricted_segment_manifest_sha256": _sha256_file(segment_path),
        "new_development_participants": NEW_COUNT,
        "locked_rows_loaded": 0,
        "accepted_segment_count": worker_receipt["accepted_segment_count"],
        "accepted_speech_seconds": worker_receipt["accepted_speech_seconds"],
        "same_segments_for_both_asr_systems": True,
        "network_denial_backend": "MACOS_SANDBOX_DENY_NETWORK",
        "restricted_values_exported": False,
    }
    _write_once(SEGMENT_RECEIPT, receipt, private=False)
    return {"status": "ok", "state": receipt["status"]}


def _cohort_summary(
    items: Sequence[Mapping[str, Any]],
    module: Any,
    *,
    bootstrap: bool,
) -> dict[str, Any]:
    count = len(items)
    if count == 0:
        _fail("E_EMPTY_COHORT")
    timing_keys = (
        "annotation_precision",
        "annotation_recall",
        "boundary_f1",
        "median_duration_change",
        "coverage_ratio",
    )
    transcript_keys = (
        "matched_char_median",
        "matched_embedding_median",
        "primary_self_char_median",
        "sensitivity_self_char_median",
        "primary_self_embedding_median",
        "sensitivity_self_embedding_median",
    )
    timing = {
        key: [float(item["timing"][key]) for item in items]
        for key in timing_keys
    }
    transcript = {
        key: [float(item["transcript"][key]) for item in items]
        for key in transcript_keys
    }
    total_segments = sum(int(item["timing"]["segment_count"]) for item in items)
    total_speech = sum(float(item["timing"]["speech_seconds"]) for item in items)
    primary_nonempty = sum(
        int(item["transcript"]["primary_nonempty_count"]) > 0 for item in items
    )
    sensitivity_nonempty = sum(
        int(item["transcript"]["sensitivity_nonempty_count"]) > 0 for item in items
    )
    primary_usable = sum(
        int(item["transcript"]["primary_nonempty_count"]) for item in items
    )
    frozen_segments = sum(
        int(item["transcript"]["frozen_segment_count"]) for item in items
    )
    boundary_interval = (
        module._bootstrap(timing["boundary_f1"], 1) if bootstrap else (None, None)
    )
    char_interval = (
        module._bootstrap(transcript["matched_char_median"], 2)
        if bootstrap
        else (None, None)
    )
    embedding_interval = (
        module._bootstrap(transcript["matched_embedding_median"], 3)
        if bootstrap
        else (None, None)
    )
    return {
        "participant_count": count,
        "accepted_segment_count": total_segments,
        "accepted_speech_seconds": total_speech,
        "nonempty_segment_participant_fraction": sum(
            int(item["timing"]["segment_count"]) > 0 for item in items
        )
        / count,
        "annotation_precision_median": module._median(
            timing["annotation_precision"]
        ),
        "annotation_recall_median": module._median(timing["annotation_recall"]),
        "boundary_f1_median": module._median(timing["boundary_f1"]),
        "boundary_f1_bootstrap_90pct": boundary_interval,
        "duration_change_median": module._median(
            timing["median_duration_change"]
        ),
        "coverage_ratio_median": module._median(timing["coverage_ratio"]),
        "primary_nonempty_item_fraction": primary_nonempty / count,
        "sensitivity_nonempty_item_fraction": sensitivity_nonempty / count,
        "primary_usable_segment_count": primary_usable,
        "primary_usable_segment_fraction": (
            primary_usable / frozen_segments if frozen_segments else 0.0
        ),
        "primary_german_item_fraction": sum(
            item["primary_language"] == "de" for item in items
        )
        / count,
        "language_id_agreement_fraction": sum(
            item["primary_language"] == item["sensitivity_language"]
            for item in items
        )
        / count,
        "character_similarity_median": module._median(
            transcript["matched_char_median"]
        ),
        "character_similarity_bootstrap_90pct": char_interval,
        "embedding_cosine_median": module._median(
            transcript["matched_embedding_median"]
        ),
        "embedding_cosine_bootstrap_90pct": embedding_interval,
        "primary_character_self_median": module._median(
            transcript["primary_self_char_median"]
        ),
        "sensitivity_character_self_median": module._median(
            transcript["sensitivity_self_char_median"]
        ),
        "primary_embedding_self_median": module._median(
            transcript["primary_self_embedding_median"]
        ),
        "sensitivity_embedding_self_median": module._median(
            transcript["sensitivity_self_embedding_median"]
        ),
    }


def _checks(
    summary: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    require_bootstrap: bool,
) -> dict[str, bool]:
    g1 = gates["G1_signal_and_vad"]
    g2 = gates["G2_matched_asr"]
    g3 = gates["G3_timing_shift_robustness"]
    boundary_interval = summary["boundary_f1_bootstrap_90pct"]
    char_interval = summary["character_similarity_bootstrap_90pct"]
    embedding_interval = summary["embedding_cosine_bootstrap_90pct"]
    checks = {
        "G1_nonempty_participants": summary[
            "nonempty_segment_participant_fraction"
        ]
        >= g1["development_participant_nonempty_segment_fraction_min"],
        "G1_segment_count": summary["accepted_segment_count"]
        >= g1["accepted_segments_min"],
        "G1_speech_seconds": summary["accepted_speech_seconds"]
        >= g1["accepted_speech_seconds_min"],
        "G1_annotation_precision": summary["annotation_precision_median"]
        >= g1["participant_median_annotation_overlap_precision_min"],
        "G1_annotation_recall": summary["annotation_recall_median"]
        >= g1["participant_median_annotation_overlap_recall_min"],
        "G1_boundary_median": summary["boundary_f1_median"]
        >= g1["participant_median_vad_self_consistency_boundary_f1_min"],
        "G1_boundary_lower": (
            not require_bootstrap
            or float(boundary_interval[0])
            >= g1["participant_bootstrap_boundary_f1_90pct_lower_min"]
        ),
        "G1_duration_change": summary["duration_change_median"]
        <= g1["participant_median_absolute_duration_change_seconds_max"],
        "G1_coverage_ratio": g1[
            "participant_median_perturbed_to_base_speech_coverage_ratio_range"
        ][0]
        <= summary["coverage_ratio_median"]
        <= g1[
            "participant_median_perturbed_to_base_speech_coverage_ratio_range"
        ][1],
        "G2_primary_nonempty": summary["primary_nonempty_item_fraction"]
        >= g2["primary_nonempty_item_fraction_min"],
        "G2_sensitivity_nonempty": summary["sensitivity_nonempty_item_fraction"]
        >= g2["sensitivity_nonempty_item_fraction_min"],
        "G2_usable_count": summary["primary_usable_segment_count"]
        >= g2["primary_usable_utterances_min"],
        "G2_usable_fraction": summary["primary_usable_segment_fraction"]
        >= g2["usable_utterance_fraction_of_frozen_segments_min"],
        "G2_primary_german": summary["primary_german_item_fraction"]
        >= g2["primary_german_item_fraction_min"],
        "G2_language_agreement": summary["language_id_agreement_fraction"]
        >= g2["primary_sensitivity_language_agreement_min"],
        "G2_character_median": summary["character_similarity_median"]
        >= g2["participant_median_normalized_character_similarity_min"],
        "G2_character_lower": (
            not require_bootstrap
            or float(char_interval[0])
            >= g2["participant_cluster_bootstrap_character_90pct_lower_min"]
        ),
        "G2_embedding_median": summary["embedding_cosine_median"]
        >= g2["participant_median_embedding_cosine_min"],
        "G2_embedding_lower": (
            not require_bootstrap
            or float(embedding_interval[0])
            >= g2["participant_cluster_bootstrap_embedding_90pct_lower_min"]
        ),
        "G3_primary_character": summary["primary_character_self_median"]
        >= g3["participant_median_primary_character_self_similarity_min"],
        "G3_sensitivity_character": summary["sensitivity_character_self_median"]
        >= g3["participant_median_sensitivity_character_self_similarity_min"],
        "G3_primary_embedding": summary["primary_embedding_self_median"]
        >= g3["participant_median_primary_embedding_self_similarity_min"],
        "G3_sensitivity_embedding": summary["sensitivity_embedding_self_median"]
        >= g3["participant_median_sensitivity_embedding_self_similarity_min"],
    }
    return checks


def _gate(checks: Mapping[str, bool], prefix: str) -> bool:
    return all(value for key, value in checks.items() if key.startswith(prefix))


def _rounded_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    participant_fraction_keys = {
        "nonempty_segment_participant_fraction",
        "primary_nonempty_item_fraction",
        "sensitivity_nonempty_item_fraction",
        "primary_german_item_fraction",
        "language_id_agreement_fraction",
    }
    participant_count = int(summary["participant_count"])
    result: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, tuple):
            result[key] = [
                round(float(item), 3) if item is not None else None
                for item in value
            ]
        elif isinstance(value, float):
            if key in participant_fraction_keys:
                cell = round(value * participant_count)
                suppress = (
                    cell not in (0, participant_count)
                    and min(cell, participant_count - cell) < 5
                )
                result[key] = None if suppress else round(value, 3)
                result[f"{key}_suppressed"] = suppress
            else:
                result[key] = round(value, 3)
        else:
            result[key] = value
    return result


def evaluate() -> Mapping[str, Any]:
    _validate_public_bindings()
    if not SEGMENT_RECEIPT.is_file():
        _fail("E_SEGMENTS_NOT_FROZEN")
    return _sandbox_reexec("_evaluate-worker", 28_800)


def _evaluate_worker() -> Mapping[str, Any]:
    if os.environ.get("CHILDLENS_V3_NETWORK_DENIED") != "1":
        _fail("E_NETWORK_DENIAL")
    runtime = _discover_runtime()
    segment_receipt = _read_json(SEGMENT_RECEIPT)
    segment_path = runtime / PRIVATE_SEGMENTS
    old_result_path = runtime / V2_RESULT
    if (
        segment_receipt.get("new_development_participants") != NEW_COUNT
        or segment_receipt.get("locked_rows_loaded") != 0
        or _sha256_file(segment_path)
        != segment_receipt.get("restricted_segment_manifest_sha256")
        or _sha256_file(old_result_path) != V2_RESULT_SHA256
    ):
        _fail("E_SEGMENT_BINDING")
    manifest = _read_json(segment_path)
    if len(manifest.get("items", [])) != NEW_COUNT:
        _fail("E_SEGMENT_BINDING")
    module = _load_v2()
    _patch_v2(module, runtime)
    private_root = runtime / PRIVATE_RELATIVE
    asr_root = private_root / "asr"
    per_item: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    embedding_pairs: list[dict[str, str]] = []
    for item in manifest["items"]:
        waveform = module._read_waveform(Path(item["audio_path"]))
        item_key = str(item["item_key"])
        base_audio = asr_root / f"{item_key}-base.wav"
        expanded_audio = asr_root / f"{item_key}-expanded.wav"
        base_mapping = module._concatenate_segments(
            waveform,
            item["base_vad_segments"],
            base_audio,
            expanded=False,
        )
        expanded_mapping = module._concatenate_segments(
            waveform,
            item["base_vad_segments"],
            expanded_audio,
            expanded=True,
        )
        documents: dict[tuple[str, str], dict[str, Any]] = {}
        for model_name, model, dtw in (
            ("primary", module.PRIMARY_MODEL, "large.v3"),
            ("sensitivity", module.SENSITIVITY_MODEL, "large.v3.turbo"),
        ):
            for condition, audio, mapping in (
                ("base", base_audio, base_mapping),
                ("expanded", expanded_audio, expanded_mapping),
            ):
                prefix = asr_root / f"{item_key}-{model_name}-{condition}"
                output = prefix.with_suffix(".json")
                document = (
                    module._read_json(output)
                    if output.exists()
                    else module._run_whisper(
                        model=model,
                        dtw=dtw,
                        audio=audio,
                        prefix=prefix,
                    )
                )
                documents[(model_name, condition)] = {
                    "language": module._language(document),
                    "text": module._map_transcript(document, mapping),
                }
        for segment in item["base_vad_segments"]:
            key = str(segment["segment_key"])
            record = {
                "participant_key": item["participant_key"],
                "segment_key": key,
                "primary_base": documents[("primary", "base")]["text"].get(key, ""),
                "sensitivity_base": documents[("sensitivity", "base")]["text"].get(
                    key, ""
                ),
                "primary_expanded": documents[("primary", "expanded")]["text"].get(
                    key, ""
                ),
                "sensitivity_expanded": documents[
                    ("sensitivity", "expanded")
                ]["text"].get(key, ""),
            }
            segment_records.append(record)
            for label, left, right in (
                ("matched", record["primary_base"], record["sensitivity_base"]),
                ("primary_self", record["primary_base"], record["primary_expanded"]),
                (
                    "sensitivity_self",
                    record["sensitivity_base"],
                    record["sensitivity_expanded"],
                ),
            ):
                if left and right:
                    embedding_pairs.append(
                        {
                            "key": f"{key}:{label}",
                            "left": left,
                            "right": right,
                        }
                    )
        per_item.append(
            {
                "participant_key": item["participant_key"],
                "primary_language": documents[("primary", "base")]["language"],
                "sensitivity_language": documents[("sensitivity", "base")][
                    "language"
                ],
                "timing": module._timing_item(item),
            }
        )
    cosines = module._embed_cosines(embedding_pairs, private_root=private_root)
    by_participant: dict[str, list[Mapping[str, Any]]] = {}
    for record in segment_records:
        by_participant.setdefault(record["participant_key"], []).append(record)
    for item in per_item:
        records = by_participant.get(item["participant_key"], [])
        matched_char = [
            module.character_similarity(row["primary_base"], row["sensitivity_base"])
            for row in records
            if row["primary_base"] and row["sensitivity_base"]
        ]
        primary_self_char = [
            module.character_similarity(row["primary_base"], row["primary_expanded"])
            for row in records
            if row["primary_base"] and row["primary_expanded"]
        ]
        sensitivity_self_char = [
            module.character_similarity(
                row["sensitivity_base"],
                row["sensitivity_expanded"],
            )
            for row in records
            if row["sensitivity_base"] and row["sensitivity_expanded"]
        ]
        item["transcript"] = {
            "frozen_segment_count": len(records),
            "primary_nonempty_count": sum(
                bool(row["primary_base"]) for row in records
            ),
            "sensitivity_nonempty_count": sum(
                bool(row["sensitivity_base"]) for row in records
            ),
            "both_nonempty_count": len(matched_char),
            "matched_char_median": module._median(matched_char),
            "matched_embedding_median": module._median(
                [
                    cosines[f"{row['segment_key']}:matched"]
                    for row in records
                    if f"{row['segment_key']}:matched" in cosines
                ]
            ),
            "primary_self_char_median": module._median(primary_self_char),
            "sensitivity_self_char_median": module._median(
                sensitivity_self_char
            ),
            "primary_self_embedding_median": module._median(
                [
                    cosines[f"{row['segment_key']}:primary_self"]
                    for row in records
                    if f"{row['segment_key']}:primary_self" in cosines
                ]
            ),
            "sensitivity_self_embedding_median": module._median(
                [
                    cosines[f"{row['segment_key']}:sensitivity_self"]
                    for row in records
                    if f"{row['segment_key']}:sensitivity_self" in cosines
                ]
            ),
        }
    old_result = _read_json(old_result_path)
    old_items = old_result.get("per_item")
    if (
        not isinstance(old_items, list)
        or len(old_items) != OLD_COUNT
        or len(per_item) != NEW_COUNT
        or {
            item["participant_key"] for item in old_items
        }
        & {item["participant_key"] for item in per_item}
    ):
        _fail("E_COHORT_BINDING")
    combined_items = [*old_items, *per_item]
    module.BOOTSTRAP_REPLICATES = 10_000
    module.BOOTSTRAP_SEED = 20_260_706
    old_summary = _cohort_summary(old_items, module, bootstrap=False)
    new_summary = _cohort_summary(per_item, module, bootstrap=True)
    combined_summary = _cohort_summary(
        combined_items,
        module,
        bootstrap=True,
    )
    gates = _read_json(CONFIG)["gates"]
    new_checks = _checks(
        new_summary,
        gates,
        require_bootstrap=False,
    )
    combined_checks = _checks(
        combined_summary,
        gates,
        require_bootstrap=True,
    )
    new_point_pass = (
        _gate(new_checks, "G1_")
        and _gate(new_checks, "G2_")
        and _gate(new_checks, "G3_")
    )
    combined_g1 = _gate(combined_checks, "G1_")
    combined_g2 = _gate(combined_checks, "G2_")
    combined_g3 = _gate(combined_checks, "G3_")
    recommend_locked = (
        new_point_pass and combined_g1 and combined_g2 and combined_g3
    )
    private_new = {
        "schema_version": "childlens-alignment-bridge-expansion-new-result-v3.0.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "restricted_segment_manifest_sha256": _sha256_file(segment_path),
        "development_only": True,
        "locked_rows_loaded_or_evaluated": 0,
        "per_item": per_item,
        "segment_records": segment_records,
        "new_point_checks": new_checks,
    }
    new_path = runtime / PRIVATE_NEW_RESULT
    _write_once(new_path, private_new, private=True)
    private_combined = {
        "schema_version": "childlens-alignment-bridge-expansion-combined-result-v3.0.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "restricted_v2_result_sha256": V2_RESULT_SHA256,
        "restricted_new_result_sha256": _sha256_file(new_path),
        "development_only": True,
        "locked_rows_loaded_or_evaluated": 0,
        "old_items": old_items,
        "new_items": per_item,
        "new_point_checks": new_checks,
        "combined_checks": combined_checks,
        "recommend_separately_authorized_locked_evaluation": recommend_locked,
    }
    combined_path = runtime / PRIVATE_COMBINED_RESULT
    _write_once(combined_path, private_combined, private=True)
    public = {
        "schema_version": "childlens-alignment-bridge-expansion-report-v3.0.1",
        "status": (
            "MEASUREMENT_EXPANSION_PASS_RECOMMEND_SEPARATE_LOCKED_AUTHORIZATION"
            if recommend_locked
            else "MEASUREMENT_EXPANSION_STOP"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "restricted_segment_manifest_sha256": _sha256_file(segment_path),
        "restricted_new_result_sha256": _sha256_file(new_path),
        "restricted_combined_result_sha256": _sha256_file(combined_path),
        "cohorts": {
            "prior_eight": _rounded_summary(old_summary),
            "new_ten": _rounded_summary(new_summary),
            "combined_eighteen": _rounded_summary(combined_summary),
        },
        "new_ten_point_checks": new_checks,
        "combined_checks": combined_checks,
        "new_ten_replication_safeguard_pass": new_point_pass,
        "combined_gate_G1_signal_and_vad_pass": combined_g1,
        "combined_gate_G2_matched_asr_pass": combined_g2,
        "combined_gate_G3_timing_shift_robustness_pass": combined_g3,
        "recommend_separately_authorized_locked_evaluation": recommend_locked,
        "locked_rows_loaded_or_evaluated": 0,
        "locked_evaluation_run": False,
        "alignment_scoring_run": False,
        "simulator_or_cue_training_run": False,
        "human_validation": False,
        "ground_truth": False,
        "asr_accuracy_claimed": False,
        "restricted_values_exported": False,
        "privacy_correction": (
            "Participant-level fractions with a nonzero complementary cell "
            "smaller than five are suppressed."
        ),
    }
    _public_guard(public)
    _write_once(REPORT, public, private=False)
    return {"status": "ok", "state": public["status"]}


def _public_guard(value: Any) -> None:
    encoded = _canonical(value).decode("utf-8")
    if PATH_TOKEN.search(encoded) or MEDIA_TOKEN.search(encoded):
        _fail("E_PUBLIC_PRIVACY")
    forbidden = {
        "participant_key",
        "segment_key",
        "transcript",
        "media_path",
        "audio_path",
        "annotation_path",
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


def locked_status() -> Mapping[str, Any]:
    return {
        "locked_evaluation_authorized": False,
        "locked_rows_loaded_or_evaluated": 0,
        "recommendation_for_separate_authorization": (
            bool(
                _read_json(REPORT).get(
                    "recommend_separately_authorized_locked_evaluation"
                )
            )
            if REPORT.is_file()
            else False
        ),
        "reason": "SEPARATE_USER_AUTHORIZATION_REQUIRED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze-measurement",
            "prepare-segments",
            "_prepare-worker",
            "evaluate",
            "_evaluate-worker",
            "locked-status",
        ),
    )
    args = parser.parse_args()
    old_umask = os.umask(0o077)
    try:
        if args.command == "freeze-measurement":
            result = freeze_measurement()
        elif args.command == "prepare-segments":
            result = prepare_segments()
        elif args.command == "_prepare-worker":
            result = _prepare_worker()
        elif args.command == "evaluate":
            result = evaluate()
        elif args.command == "_evaluate-worker":
            result = _evaluate_worker()
        else:
            result = locked_status()
        print(json.dumps(result, sort_keys=True))
        return 0
    except MeasurementError as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
