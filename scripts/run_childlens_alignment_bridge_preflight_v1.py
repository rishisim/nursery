#!/usr/bin/env python3
"""Freeze and run the fail-closed ChildLens alignment-bridge preflight.

Restricted identifiers, paths, intervals, ASR text, and row-level measurements
remain under the existing owner-private ChildLens quarantine. Repository
outputs contain only K-safe aggregates and cryptographic receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_v1.json"
PUBLIC_ROOT = ROOT / "output/childlens_alignment_bridge_v1"
FREEZE_RECEIPT = PUBLIC_ROOT / "freeze_receipt.json"
DEV_REPORT = PUBLIC_ROOT / "development_gate_report_v1_0_1.json"
CORRECTION_RECEIPT = PUBLIC_ROOT / "mechanical_language_normalization_correction_v1_0_1.json"
PRIVATE_RELATIVE = Path("provisional_calibration_v1/childlens_alignment_bridge_v1")
PRIVATE_SPLIT = PRIVATE_RELATIVE / "restricted_split_manifest.json"
PRIVATE_DEV = PRIVATE_RELATIVE / "restricted_development_evaluation_v1_0_1.json"
EXPECTED_SAMPLE_COUNT = 30
MIN_PUBLIC_CELL = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_706

sys.path.insert(0, str(ROOT))
from babyworld_lite.childlens_alignment_bridge_v1.preflight import (  # noqa: E402
    BridgeError,
    canonical_text,
    character_similarity,
    interval_boundary_f1,
    participant_bootstrap_interval,
)


def _fail(code: str) -> None:
    raise BridgeError(code)


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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("E_FILE")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_FILE")


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


def _write_once(path: Path, value: Any, *, private: bool) -> None:
    payload = _canonical(value) + b"\n"
    mode = 0o600 if private else 0o644
    directory_mode = 0o700 if private else 0o755
    path.parent.mkdir(parents=True, exist_ok=True, mode=directory_mode)
    if private:
        os.chmod(path.parent, 0o700)
    if path.exists():
        if path.read_bytes() != payload:
            _fail("E_IMMUTABLE_CONFLICT")
        return
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    try:
        fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, mode)
    finally:
        if pending.exists():
            pending.unlink()


def _discover_runtime() -> Path:
    search = ROOT.parent
    candidates = []
    for hidden in search.iterdir():
        if (
            hidden.name.startswith(".")
            and "childlens" in hidden.name.casefold()
            and _private_directory(hidden)
        ):
            candidates.extend(
                path.parents[1]
                for path in hidden.rglob("post_acquisition_v1_2/restricted_measurement_manifest.json")
                if _private_file(path)
            )
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) != 1:
        _fail("E_RUNTIME_DISCOVERY")
    runtime = unique[0]
    if not _private_directory(runtime) or ROOT in runtime.parents or runtime == ROOT:
        _fail("E_RUNTIME_CONTROL")
    sentinel = runtime / ".metadata_never_index"
    if not _private_file(sentinel):
        _fail("E_RUNTIME_CONTROL")
    return runtime


def _config() -> tuple[Mapping[str, Any], str]:
    config = _read_json(CONFIG)
    if (
        config.get("schema_version") != "childlens-alignment-bridge-preflight-v1.0.0"
        or config.get("status")
        != "FROZEN_BEFORE_SPLIT_ASSIGNMENT_OR_DEVELOPMENT_OUTCOME_INSPECTION"
        or config.get("scope", {}).get("simulator_generation_allowed") is not False
        or config.get("scope", {}).get("learner_training_allowed") is not False
    ):
        _fail("E_PROTOCOL")
    return config, _sha256_file(CONFIG)


def _interval_bounds(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    starts = [float(row["start_seconds"]) for row in rows]
    ends = [float(row["end_seconds"]) for row in rows]
    if not starts or min(starts) < 0 or max(ends) <= min(starts):
        _fail("E_INTERVAL")
    return min(starts), max(ends)


def _load_sample_rows(runtime: Path) -> list[dict[str, Any]]:
    full = _read_json(runtime / "post_acquisition_v1_2/frozen_v1_1_restricted_input.json")
    measurement = _read_json(runtime / "post_acquisition_v1_2/restricted_measurement_manifest.json")
    snapshot = _read_json(runtime / "author_audit_v1_3/primary_sample_snapshot.json")
    extension = _read_json(runtime / "provisional_calibration_v1/extension_v1_8/restricted_extension_plan.json")
    checkpoint = _read_json(
        runtime / "provisional_calibration_v1/extension_v1_8/restricted_transfer_checkpoint.json"
    )
    media_by_key = {row["media_key"]: row for row in full["media"]}
    annotation_by_media = {row["linked_media_key"]: row for row in full["annotations"]}
    object_by_key = {row["object_key"]: row for row in full["objects"]}
    snapshot_by_hash = {row["expected_media_sha256"]: row for row in snapshot["items"]}
    rows: list[dict[str, Any]] = []

    for item in measurement["items"]:
        media = media_by_key.get(item["blinded_item_key"])
        sample = snapshot_by_hash.get(item["expected_media_sha256"])
        annotation = annotation_by_media.get(item["blinded_item_key"])
        if media is None or sample is None or annotation is None:
            _fail("E_ORIGINAL_LINK")
        annotation_object = object_by_key.get(annotation["object_key"])
        if annotation_object is None:
            _fail("E_ORIGINAL_LINK")
        segments = [
            {"start_seconds": row["start_ms"] / 1000.0, "end_seconds": row["end_ms"] / 1000.0}
            for row in sample["segments"]
        ]
        span_start, span_end = _interval_bounds(segments)
        media_path = runtime / item["media_relpath"]
        whisper_path = runtime / "model_assisted_v1_3/pseudo_annotations/audio" / (
            item["expected_media_sha256"] + ".json"
        )
        rows.append(
            {
                "cohort": "original",
                "participant_key": media["participant_key"],
                "media_key": media["media_key"],
                "item_key": item["expected_media_sha256"],
                "media_path": str(media_path),
                "expected_media_sha256": item["expected_media_sha256"],
                "annotation_path": str(runtime / annotation_object["source_locator"]),
                "annotation_sha256": annotation_object["local_sha256"],
                "primary_asr_container": str(runtime / "provisional_calibration_v1/qwen3_asr_restricted.json"),
                "sensitivity_asr_path": str(whisper_path),
                "sample_segments": segments,
                "sample_span_start_seconds": span_start,
                "sample_span_end_seconds": span_end,
                "activity_label": media.get("coarse_activity_label"),
                "location_label": media.get("location_label"),
            }
        )

    checkpoint_by_object = {row["object_key"]: row for row in checkpoint["items"]}
    for item in extension["items"]:
        transfer = checkpoint_by_object.get(item["object_key"])
        media = media_by_key.get(item["media_key"])
        annotation = annotation_by_media.get(item["media_key"])
        if transfer is None or media is None or annotation is None:
            _fail("E_EXTENSION_LINK")
        annotation_object = object_by_key.get(annotation["object_key"])
        if annotation_object is None:
            _fail("E_EXTENSION_LINK")
        segments = [
            {"start_seconds": row["start_ms"] / 1000.0, "end_seconds": row["end_ms"] / 1000.0}
            for row in item["sample_segments_clip_ms"]
        ]
        span_start, span_end = _interval_bounds(segments)
        clip_hash = transfer["clip_sha256"]
        rows.append(
            {
                "cohort": "extension",
                "participant_key": item["participant_key"],
                "media_key": item["media_key"],
                "item_key": clip_hash,
                "media_path": str(runtime / transfer["clip_relative_path"]),
                "expected_media_sha256": clip_hash,
                "annotation_path": str(runtime / annotation_object["source_locator"]),
                "annotation_sha256": annotation_object["local_sha256"],
                "primary_asr_container": str(
                    runtime / "provisional_calibration_v1/extension_v1_8/inference/qwen3_asr_restricted.json"
                ),
                "sensitivity_asr_path": str(
                    runtime / "provisional_calibration_v1/extension_v1_8/inference/whisper" / f"{clip_hash}.json"
                ),
                "sample_segments": segments,
                "sample_span_start_seconds": span_start,
                "sample_span_end_seconds": span_end,
                "source_offset_seconds": item["clip_source_start_ms"] / 1000.0,
                "activity_label": media.get("coarse_activity_label"),
                "location_label": media.get("location_label"),
            }
        )
    if len(rows) != EXPECTED_SAMPLE_COUNT or len({row["participant_key"] for row in rows}) != len(rows):
        _fail("E_SAMPLE")
    for row in rows:
        for key in ("media_path", "annotation_path", "sensitivity_asr_path"):
            path = Path(row[key])
            if not _private_file(path):
                _fail("E_RESTRICTED_FILE")
        if _sha256_file(Path(row["media_path"])) != row["expected_media_sha256"]:
            _fail("E_MEDIA_HASH")
        if _sha256_file(Path(row["annotation_path"])) != row["annotation_sha256"]:
            _fail("E_ANNOTATION_HASH")
        if not _private_file(Path(row["primary_asr_container"])):
            _fail("E_RESTRICTED_FILE")
    return rows


def freeze() -> Mapping[str, Any]:
    config, protocol_sha256 = _config()
    runtime = _discover_runtime()
    rows = _load_sample_rows(runtime)
    for cohort in ("original", "extension"):
        cohort_rows = [row for row in rows if row["cohort"] == cohort]
        cohort_rows.sort(
            key=lambda row: hashlib.sha256(
                f"{protocol_sha256}|{row['participant_key']}".encode("utf-8")
            ).hexdigest()
        )
        for index, row in enumerate(cohort_rows):
            row["split"] = "development" if index < 4 else "locked"
            row["split_rank"] = index
    rows.sort(key=lambda row: (row["split"], row["cohort"], row["split_rank"]))
    private = {
        "schema_version": "childlens-alignment-bridge-restricted-split-v1.0.0",
        "protocol_sha256": protocol_sha256,
        "combined_selection_sha256": config["source_bindings"]["combined_selection_sha256"],
        "participant_distinct": True,
        "locked_outcomes_inspected_during_split": False,
        "items": rows,
    }
    private_path = runtime / PRIVATE_SPLIT
    _write_once(private_path, private, private=True)
    split_counts = Counter(row["split"] for row in rows)
    cohort_counts = Counter((row["split"], row["cohort"]) for row in rows)
    receipt = {
        "schema_version": "childlens-alignment-bridge-freeze-receipt-v1.0.0",
        "status": "FROZEN",
        "protocol_sha256": protocol_sha256,
        "restricted_split_manifest_sha256": _sha256_file(private_path),
        "combined_selection_sha256": config["source_bindings"]["combined_selection_sha256"],
        "sample_count": len(rows),
        "participant_distinct": True,
        "development_count": split_counts["development"],
        "locked_count": split_counts["locked"],
        "development_original_count": cohort_counts[("development", "original")],
        "development_extension_count": cohort_counts[("development", "extension")],
        "locked_outcomes_inspected": False,
        "restricted_identifiers_exported": False,
        "restricted_paths_or_filenames_exported": False,
        "transcripts_or_exact_intervals_exported": False,
        "simulator_generated": False,
        "learner_trained": False,
    }
    _write_once(FREEZE_RECEIPT, receipt, private=False)
    return receipt


def _as_interval(row: Mapping[str, Any]) -> dict[str, float] | None:
    intervals = row.get("source_intervals")
    if not isinstance(intervals, list) or not intervals:
        return None
    try:
        start = min(float(value["start_seconds"]) for value in intervals)
        end = max(float(value["end_seconds"]) for value in intervals)
    except (KeyError, TypeError, ValueError):
        return None
    return {"start_seconds": start, "end_seconds": end} if end > start else None


def _primary_rows(container: Path, keys: set[str]) -> dict[str, Mapping[str, Any]]:
    # The historical ASR container is item-keyed but monolithic. Only requested
    # development rows are retained or analyzed; locked rows are never returned.
    document = _read_json(container)
    selected = {
        row["opaque_key"]: row
        for row in document.get("items", [])
        if row.get("opaque_key") in keys
    }
    if set(selected) != keys:
        _fail("E_PRIMARY_ASR_LINK")
    return selected


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        _fail("E_EMPTY_SUMMARY")
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def _safe_round(value: float) -> float:
    return round(float(value), 3)


def _gate(value: float, threshold: float) -> bool:
    return value >= threshold


def _language_code(value: Any) -> str:
    normalized = canonical_text(str(value))
    aliases = {
        "de": "de",
        "deu": "de",
        "deutsch": "de",
        "german": "de",
        "en": "en",
        "eng": "en",
        "english": "en",
    }
    return aliases.get(normalized, normalized)


def evaluate_development() -> Mapping[str, Any]:
    config, protocol_sha256 = _config()
    runtime = _discover_runtime()
    private_path = runtime / PRIVATE_SPLIT
    if not _private_file(private_path):
        _fail("E_NOT_FROZEN")
    split = _read_json(private_path)
    if (
        split.get("protocol_sha256") != protocol_sha256
        or _sha256_file(private_path) != _read_json(FREEZE_RECEIPT).get("restricted_split_manifest_sha256")
    ):
        _fail("E_FREEZE_BINDING")
    development = [row for row in split["items"] if row.get("split") == "development"]
    if len(development) != 8:
        _fail("E_DEVELOPMENT_SPLIT")

    by_container: dict[str, set[str]] = {}
    for row in development:
        by_container.setdefault(row["primary_asr_container"], set()).add(row["item_key"])
    primary: dict[str, Mapping[str, Any]] = {}
    for path, keys in by_container.items():
        primary.update(_primary_rows(Path(path), keys))

    per_item = []
    for row in development:
        item = primary[row["item_key"]]
        sensitivity = _read_json(Path(row["sensitivity_asr_path"]))
        primary_utterances = []
        for utterance in item.get("utterance_hypotheses", []):
            interval = _as_interval(utterance)
            text = canonical_text(str(utterance.get("text", "")))
            if interval is None:
                continue
            duration = interval["end_seconds"] - interval["start_seconds"]
            if 0.3 <= duration <= 12.0 and len(text) >= 2:
                primary_utterances.append({**interval, "text": text})
        sensitivity_utterances = []
        sensitivity_texts = []
        for utterance in sensitivity.get("asr_hypotheses", []):
            interval = _as_interval(utterance)
            text = canonical_text(str(utterance.get("text", "")))
            if interval is not None and text:
                sensitivity_utterances.append(interval)
                sensitivity_texts.append(text)
        primary_text = canonical_text(str(item.get("transcript_hypothesis", "")))
        sensitivity_text = " ".join(sensitivity_texts)
        per_item.append(
            {
                "participant_key": row["participant_key"],
                "cohort": row["cohort"],
                "primary_nonempty": bool(primary_text),
                "sensitivity_nonempty": bool(sensitivity_text),
                "primary_language": _language_code(item.get("language_hypothesis")),
                "sensitivity_language": _language_code(sensitivity.get("language_hypothesis")),
                "primary_utterance_count": len(primary_utterances),
                "primary_total_utterance_count": len(item.get("utterance_hypotheses", [])),
                "character_similarity": character_similarity(primary_text, sensitivity_text),
                "boundary_f1": interval_boundary_f1(primary_utterances, sensitivity_utterances),
            }
        )

    count = len(per_item)
    primary_nonempty = sum(row["primary_nonempty"] for row in per_item) / count
    sensitivity_nonempty = sum(row["sensitivity_nonempty"] for row in per_item) / count
    usable = sum(row["primary_utterance_count"] for row in per_item)
    total_primary = sum(row["primary_total_utterance_count"] for row in per_item)
    usable_fraction = usable / total_primary if total_primary else 0.0
    primary_german = sum(row["primary_language"] == "de" for row in per_item) / count
    language_agreement = sum(
        row["primary_language"] == row["sensitivity_language"] for row in per_item
    ) / count
    char_values = [row["character_similarity"] for row in per_item]
    boundary_values = [row["boundary_f1"] for row in per_item]
    char_interval = participant_bootstrap_interval(
        char_values,
        confidence=0.9,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    boundary_interval = participant_bootstrap_interval(
        boundary_values,
        confidence=0.9,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED + 1,
    )
    thresholds = config["ordered_gates"]
    g1 = thresholds["G1_development_asr_coverage"]
    g2 = thresholds["G2_development_german_and_model_model_stability"]
    checks = {
        "G1_primary_nonempty": _gate(primary_nonempty, g1["primary_nonempty_item_fraction_min"]),
        "G1_sensitivity_nonempty": _gate(
            sensitivity_nonempty, g1["sensitivity_nonempty_item_fraction_min"]
        ),
        "G1_usable_count": usable >= g1["primary_usable_utterances_min"],
        "G1_usable_fraction": _gate(
            usable_fraction, g1["usable_utterance_fraction_of_primary_min"]
        ),
        "G2_primary_german": _gate(
            primary_german, g2["primary_german_item_fraction_min"]
        ),
        "G2_language_agreement": _gate(
            language_agreement, g2["primary_sensitivity_language_agreement_min"]
        ),
        "G2_character_median": _gate(
            _median(char_values), g2["participant_median_normalized_character_similarity_min"]
        ),
        "G2_character_lower": _gate(
            char_interval[0], g2["participant_cluster_bootstrap_90pct_lower_min"]
        ),
        "G2_boundary_median": _gate(
            _median(boundary_values), g2["participant_median_boundary_f1_min"]
        ),
        "G2_boundary_lower": _gate(
            boundary_interval[0],
            g2["participant_cluster_bootstrap_boundary_f1_90pct_lower_min"],
        ),
    }
    g1_pass = all(value for key, value in checks.items() if key.startswith("G1_"))
    g2_pass = all(value for key, value in checks.items() if key.startswith("G2_"))
    stop = not (g1_pass and g2_pass)
    private_result = {
        "schema_version": "childlens-alignment-bridge-restricted-development-v1.0.1",
        "protocol_sha256": protocol_sha256,
        "restricted_split_manifest_sha256": _sha256_file(private_path),
        "development_only": True,
        "locked_rows_analyzed": 0,
        "per_item": per_item,
        "checks": checks,
        "gate_G1_pass": g1_pass,
        "gate_G2_pass": g2_pass,
        "stop_before_alignment": stop,
    }
    private_dev_path = runtime / PRIVATE_DEV
    _write_once(private_dev_path, private_result, private=True)
    public = {
        "schema_version": "childlens-alignment-bridge-development-gate-v1.0.1",
        "status": "STOP_ASR_STABILITY" if stop else "PASS_TO_DEVELOPMENT_ALIGNMENT",
        "protocol_sha256": protocol_sha256,
        "restricted_split_manifest_sha256": _sha256_file(private_path),
        "restricted_development_result_sha256": _sha256_file(private_dev_path),
        "development_participants": count,
        "locked_participants_analyzed": 0,
        "development_cohort_counts": {
            key: value for key, value in Counter(row["cohort"] for row in per_item).items()
            if value >= MIN_PUBLIC_CELL
        },
        "coverage": {
            "primary_nonempty_item_fraction": _safe_round(primary_nonempty),
            "sensitivity_nonempty_item_fraction": _safe_round(sensitivity_nonempty),
            "primary_usable_utterance_count": usable,
            "primary_usable_utterance_fraction": _safe_round(usable_fraction),
        },
        "language_model_model_diagnostic": {
            "primary_german_item_fraction": _safe_round(primary_german),
            "language_id_agreement_fraction": _safe_round(language_agreement),
            "character_similarity_participant_median": _safe_round(_median(char_values)),
            "character_similarity_bootstrap_90pct": [
                _safe_round(char_interval[0]),
                _safe_round(char_interval[1]),
            ],
            "boundary_f1_participant_median": _safe_round(_median(boundary_values)),
            "boundary_f1_bootstrap_90pct": [
                _safe_round(boundary_interval[0]),
                _safe_round(boundary_interval[1]),
            ],
            "human_validation": False,
            "ground_truth": False,
        },
        "checks": checks,
        "gate_G1_pass": g1_pass,
        "gate_G2_pass": g2_pass,
        "development_alignment_scored": False,
        "locked_alignment_scored": False,
        "model_independent_annotation_vector_status": (
            "DEFERRED_BY_ORDERED_ASR_STOP" if stop else "READY"
        ),
        "motion_persistence_scene_change_status": (
            "DEFERRED_BY_ORDERED_ASR_STOP" if stop else "READY"
        ),
        "simulator_generated": False,
        "learner_trained": False,
        "restricted_text_intervals_ids_paths_exported": False,
        "decision": (
            "STOP_BEFORE_ALIGNMENT_AND_LOCKED_EVALUATION"
            if stop
            else "CONTINUE_TO_DEVELOPMENT_ALIGNMENT_ONLY"
        ),
    }
    _write_once(DEV_REPORT, public, private=False)
    superseded_public = PUBLIC_ROOT / "development_gate_report.json"
    superseded_private = (
        runtime
        / PRIVATE_RELATIVE
        / "restricted_development_evaluation.json"
    )
    correction = {
        "schema_version": "childlens-alignment-bridge-mechanical-correction-v1.0.1",
        "status": "CORRECTED_LANGUAGE_LABEL_NORMALIZATION_ONLY",
        "protocol_sha256": protocol_sha256,
        "frozen_thresholds_changed": False,
        "split_changed": False,
        "asr_text_or_boundaries_changed": False,
        "old_public_result_sha256": (
            _sha256_file(superseded_public) if superseded_public.is_file() else None
        ),
        "old_restricted_result_sha256": (
            _sha256_file(superseded_private) if _private_file(superseded_private) else None
        ),
        "corrected_public_result_sha256": _sha256_file(DEV_REPORT),
        "corrected_restricted_result_sha256": _sha256_file(private_dev_path),
        "decision_changed": False,
        "superseded_result_should_not_be_used": True,
    }
    _write_once(CORRECTION_RECEIPT, correction, private=False)
    return public


def locked_status() -> Mapping[str, Any]:
    runtime = _discover_runtime()
    private_dev = runtime / PRIVATE_DEV
    if not _private_file(private_dev):
        _fail("E_DEVELOPMENT_NOT_RUN")
    result = _read_json(private_dev)
    allowed = (
        result.get("gate_G1_pass") is True
        and result.get("gate_G2_pass") is True
        and result.get("stop_before_alignment") is False
    )
    return {
        "locked_evaluation_allowed": allowed,
        "reason": "DEVELOPMENT_GATES_PASSED" if allowed else "DEVELOPMENT_STOP_RULE_ACTIVE",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "evaluate-development", "locked-status"))
    args = parser.parse_args()
    old_umask = os.umask(0o077)
    try:
        if args.command == "freeze":
            result = freeze()
        elif args.command == "evaluate-development":
            result = evaluate_development()
        else:
            result = locked_status()
        print(json.dumps(result, sort_keys=True))
        return 0
    except BridgeError as exc:
        print(json.dumps({"status": "error", "error_code": str(exc)}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
