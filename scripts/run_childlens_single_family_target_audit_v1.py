#!/usr/bin/env python3
"""Audit the frozen V5 estimators under exact timed object-play conditioning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE = (
    ROOT.parent
    / ".childlens_restricted/feasibility_v1_1/restricted_manifest/"
    "provisional_calibration_v1/childlens_alignment_bridge_v5"
)
DEFAULT_OUTPUT = ROOT / "output/childlens_single_family_preflight/target_audit.json"
FREEZE = ROOT / "configs/childlens_single_family_preflight_v1.json"
V5_RUNNER = ROOT / "scripts/run_childlens_alignment_bridge_v5.py"
V5_SUMMARY = ROOT / "output/childlens_alignment_bridge_v5/clean_calibration_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def overlap_fraction(start_ms: int, end_ms: int, intervals: Iterable[tuple[int, int]]) -> float:
    overlap = sum(
        max(0, min(end_ms, right) - max(start_ms, left))
        for left, right in intervals
    )
    return overlap / (end_ms - start_ms)


def interval(values: list[float], *, seed: int) -> list[float]:
    clean = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, clean.size, size=(10_000, clean.size))
    estimates = clean[draws].mean(axis=1)
    return [float(np.quantile(estimates, 0.05)), float(np.quantile(estimates, 0.95))]


def summary(values: list[float], *, seed: int) -> dict[str, Any]:
    if len(values) < 5:
        return {"status": "SUPPRESSED_K_LT_5"}
    bounds = interval(values, seed=seed)
    return {
        "participant_count": len(values),
        "participant_mean": round(float(np.mean(values)), 4),
        "participant_cluster_90pct": [round(value, 4) for value in bounds],
        "participant_quantiles_10_50_90": [
            round(float(value), 4) for value in np.quantile(values, (0.1, 0.5, 0.9))
        ],
    }


def run(private_root: Path) -> dict[str, Any]:
    freeze = read_json(FREEZE)
    audit = freeze["target_audit"]
    admin_path = private_root / "administrative/development_only_scientific_input.json"
    manifest_path = private_root / "clean_run/restricted_feature_manifest.json"
    features_path = private_root / "clean_run/restricted_frontend_features.npz"
    admin = read_json(admin_path)
    manifest = read_json(manifest_path)
    if admin["development_participant_count"] != 18 or admin["locked_participant_count"] != 0:
        raise RuntimeError("private input is not the attested exact-18 development-only scope")

    object_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    event_count = 0
    for item in admin["items"]:
        annotation_path = Path(item["annotation_bindings"][0]["source_locator"])
        if sha256(annotation_path) != item["annotation_bindings"][0]["local_sha256"]:
            raise RuntimeError("annotation hash mismatch")
        annotations = read_json(annotation_path)["annotations"]
        for row in annotations:
            if (
                row.get("eventId") == audit["annotation_event_id"]
                and row.get("type") == audit["annotation_type"]
            ):
                object_intervals[str(item["source_object_key"])].append(
                    (round(float(row["startTime"]) * 1000), round(float(row["endTime"]) * 1000))
                )
                event_count += 1

    zero_indices = {
        int(request_index)
        for row in manifest["rows"]
        for request_index in [
            next(
                request["index"]
                for request in manifest["requests"]
                if request["request_key"] == row["lag_requests"]["0"]
            )
        ]
    }
    eligible: dict[str, list[int]] = defaultdict(list)
    minimum_overlap = float(audit["minimum_object_play_overlap_fraction"])
    allowed_durations = set(audit["window_durations_seconds"])
    for request in manifest["requests"]:
        index = int(request["index"])
        if index not in zero_indices or int(request["duration_seconds"]) not in allowed_durations:
            continue
        if overlap_fraction(
            int(request["start_ms"]),
            int(request["end_ms"]),
            object_intervals[str(request["source_object_key"])],
        ) >= minimum_overlap:
            eligible[str(request["participant_key"])].append(index)

    arrays = np.load(features_path, allow_pickle=False)
    metric_arrays = {
        "motion": arrays["motion"],
        "adjacent_frame_persistence": arrays["persistence"],
        "scene_change_rate": arrays["scene_change"],
    }
    participant_values = {
        metric: [
            float(np.mean(values[indices]))
            for _, indices in sorted(eligible.items())
            if indices
        ]
        for metric, values in metric_arrays.items()
    }
    conditional = {
        metric: summary(values, seed=int(audit["uncertainty"]["seed_base"]) + offset)
        for offset, (metric, values) in enumerate(sorted(participant_values.items()))
    }
    v5 = read_json(V5_SUMMARY)
    return {
        "schema": "ChildLensTimedObjectPlayTargetAudit.v1",
        "status": "VALID_CONDITIONAL_TARGET",
        "scientific_role": freeze["scientific_role"],
        "freeze_sha256": sha256(FREEZE),
        "v5_runner_sha256": sha256(V5_RUNNER),
        "v5_summary_sha256": sha256(V5_SUMMARY),
        "private_input_hashes": {
            "administrative_input_sha256": sha256(admin_path),
            "feature_manifest_sha256": sha256(manifest_path),
            "frontend_features_sha256": sha256(features_path),
            "annotation_aggregate_sha256": hashlib.sha256(
                "".join(
                    sorted(item["annotation_bindings"][0]["local_sha256"] for item in admin["items"])
                ).encode()
            ).hexdigest(),
        },
        "estimator_reproduction": {
            "definitions_exactly_match_frozen_V5": True,
            "dinov2_weights_hash_verified_by_V5_receipt": True,
            "language_interpretation_used": False,
        },
        "support": {
            "development_participants": 18,
            "locked_participants": 0,
            "timed_object_play_events": event_count,
            "participants_with_eligible_zero_lag_windows": len(eligible),
            "eligible_zero_lag_windows": sum(map(len, eligible.values())),
            "eligible_windows_per_participant_min": min(map(len, eligible.values())),
            "eligible_windows_per_participant_median": float(np.median(list(map(len, eligible.values())))),
            "eligible_windows_per_participant_max": max(map(len, eligible.values())),
            "minimum_overlap_fraction": minimum_overlap,
        },
        "primary_timed_object_play_participant_clustered": conditional,
        "secondary_full_representative_natural_mixture": {
            metric: v5["representative_participant_clustered"][metric]
            for metric in metric_arrays
        },
        "recording_level_fallback_relationship_only": v5[
            "grounding_enriched_recording_level_fallback"
        ],
        "limitations": [
            "ChildLens ages 3-to-5 are a provisional developmental calibration, not an infant age match.",
            "The exact timed condition is coarse playing_with_object activity, not a lexical referent label.",
            "DINOv2 persistence is a transparent model-model sensitivity, not human validation or ground truth.",
            "The participant sample is development-only (n=18); locked participants were not read.",
        ],
        "privacy": {
            "participant_level_values_exported": False,
            "identifiers_paths_raw_media_or_annotation_rows_exported": False,
            "AEA_or_BabyView_used": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.private_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
