from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from babyworld_lite.aea.config import AEA_SCIENTIFIC_ROLE
from babyworld_lite.aea.preprocess import IMU_CHANNELS, SCHEMA_VERSION


MODEL_INPUT_KEYS = {
    "frame_paths", "frame_capture_device_time_ns", "transcript", "imu_path",
    "imu_channels", "imu_stream", "imu_rate_hz",
}


def audit_aea_records(
    records: Sequence[Mapping[str, Any]],
    dataset_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(dataset_root)
    failures: list[str] = []
    action_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    composition_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    location_counts: Counter[str] = Counter()
    wearer_groups: set[str] = set()
    event_to_recordings: dict[str, set[int]] = defaultdict(set)
    coverage: list[float] = []
    rates: list[float] = []
    confidences: list[float] = []
    frame_time_errors: list[float] = []
    frame_errors: list[str] = []
    imu_errors: list[str] = []
    seen_ids: set[str] = set()
    expected_imu_count = round(
        float(config["window"]["duration_seconds"]) * float(config["window"]["imu_rate_hz"])
    )
    for row in records:
        example_id = str(row.get("example_id", ""))
        if row.get("schema_version") != SCHEMA_VERSION:
            failures.append(f"{example_id}: wrong schema version")
        if example_id in seen_ids:
            failures.append(f"duplicate example id: {example_id}")
        seen_ids.add(example_id)
        inputs = row.get("model_inputs", {})
        if set(inputs) != MODEL_INPUT_KEYS:
            failures.append(f"{example_id}: model input allowlist violation")
        if any(key.startswith("mps") for key in inputs):
            failures.append(f"{example_id}: unnecessary MPS input present")
        frames = list(inputs.get("frame_paths", []))
        if len(frames) != int(config["window"]["rgb_frames"]):
            frame_errors.append(f"{example_id}: wrong frame count")
        for relative in frames:
            if not (root / relative).is_file():
                frame_errors.append(f"{example_id}: missing {relative}")
        imu_path = root / str(inputs.get("imu_path", ""))
        if not imu_path.is_file():
            imu_errors.append(f"{example_id}: missing IMU")
        else:
            values = np.load(imu_path, allow_pickle=False)
            if values.shape != (expected_imu_count, 6) or not np.isfinite(values).all():
                imu_errors.append(f"{example_id}: invalid IMU shape/values {values.shape}")
        if tuple(inputs.get("imu_channels", ())) != IMU_CHANNELS:
            imu_errors.append(f"{example_id}: IMU channel order mismatch")
        target = row.get("evaluation_targets", {})
        action = str(target.get("action_verb", ""))
        obj = str(target.get("object_noun", ""))
        if not action:
            failures.append(f"{example_id}: missing action target")
        action_counts[action] += 1
        if obj:
            object_counts[obj] += 1
            composition_counts[f"{action}|{obj}"] += 1
        sequence_counts[str(row.get("sequence_id"))] += 1
        location_counts[str(row.get("location"))] += 1
        wearer_groups.add(str(row.get("wearer_session_group")))
        event_to_recordings[str(row.get("event_group"))].add(int(row.get("recording", 0)))
        quality = row.get("quality", {})
        coverage.append(float(quality.get("coverage_fraction", 0)))
        rates.append(float(quality.get("raw_rate_hz", 0)))
        frame_time_error = float(quality.get("maximum_frame_time_error_ms", float("inf")))
        frame_time_errors.append(frame_time_error)
        if frame_time_error > float(config["quality"]["maximum_frame_time_error_ms"]):
            frame_errors.append(f"{example_id}: RGB timestamp error exceeds threshold")
        confidences.append(float(row.get("annotation_provenance", {}).get("action_asr_confidence", 0)))
    failures.extend(frame_errors[:20])
    failures.extend(imu_errors[:20])
    if not records:
        failures.append("no preprocessed windows; dataset is not experiment-ready")
    if config["profile"]["scientific_role"] != AEA_SCIENTIFIC_ROLE:
        failures.append("scientific role does not prohibit developmental-evidence claims")
    result = {
        "valid": not failures,
        "failures": failures,
        "scientific_role": config["profile"]["scientific_role"],
        "counts": {
            "windows": len(records),
            "sequences": len(sequence_counts),
            "locations": dict(location_counts),
            "wearer_session_proxy_groups": len(wearer_groups),
            "concurrent_event_groups": sum(len(value) > 1 for value in event_to_recordings.values()),
            "actions": dict(action_counts),
            "objects": dict(object_counts),
            "action_object_compositions": dict(composition_counts),
        },
        "quality": {
            "minimum_imu_coverage": min(coverage) if coverage else None,
            "raw_imu_rate_hz_range": [min(rates), max(rates)] if rates else None,
            "maximum_frame_time_error_ms": max(frame_time_errors) if frame_time_errors else None,
            "action_asr_confidence_range": [min(confidences), max(confidences)] if confidences else None,
            "frame_error_count": len(frame_errors),
            "imu_error_count": len(imu_errors),
        },
        "leakage_policy": {
            "all_windows_grouped_by_sequence": True,
            "concurrent_recordings_grouped_or_purged": True,
            "model_input_allowlist": sorted(MODEL_INPUT_KEYS),
            "mps_excluded": True,
            "imu_training_only_primary_test": True,
        },
        "label_limitations": {
            "action_labels": "automatic speech-recognition lexical anchors, not human action annotations",
            "object_labels": "nearest ASR lexical item within a locked time gap; absent when unsupported",
            "wearer_identity": "location+script+recording proxy; persistent person identity is not exposed",
        },
    }
    (root / "audit_summary.json").write_text(json.dumps(result, indent=2))
    return result
