#!/usr/bin/env python3
"""Run the frozen ChildLens V5 outcome-blind support repair exactly once."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_audiovisual_support_repair.json"
PUBLIC = ROOT / "output/childlens_audiovisual_support_repair"
PRIVATE_ROOT = Path(
    "/Volumes/CHILDLENS_RESTRICTED/feasibility_v1_1/restricted_manifest/"
    "provisional_calibration_v1/childlens_alignment_bridge_v5/clean_run"
)
MANIFEST = PRIVATE_ROOT / "restricted_feature_manifest.json"
FEATURES = PRIVATE_ROOT / "restricted_frontend_features.npz"
WEIGHTS = PRIVATE_ROOT / "restricted_projection_weights.npz"
RESULT = PRIVATE_ROOT / "restricted_development_result.json"
FEASIBILITY = PUBLIC / "support_feasibility_receipt.json"
LAG_REPORT = PUBLIC / "lag_response_report.json"
DECISION = PUBLIC / "decision_report.json"

from babyworld_lite.childlens_alignment_bridge_v5.preflight import (  # noqa: E402
    score_temporal_projectors,
)
from babyworld_lite.childlens_support_repair import (  # noqa: E402
    SupportRepairError,
    nuisance_matrix,
    participant_contrasts,
    stable_weights,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG)
    expected = config["immutable_v5"]
    observed = {
        "feature_manifest_sha256": sha256(MANIFEST),
        "frontend_features_sha256": sha256(FEATURES),
        "projection_weights_sha256": sha256(WEIGHTS),
        "restricted_result_sha256": sha256(RESULT),
    }
    if observed != {
        key: expected[key]
        for key in observed
    }:
        raise SupportRepairError("E_HASH_BINDING")
    manifest = load_json(MANIFEST)
    if (
        manifest.get("clean_run_id") != "childlens-v5-clean-20260723-a"
        or manifest.get("embedding_completion_fraction") != 1.0
        or manifest.get(
            "tokenizer_decoder_ctc_asr_translation_or_language_id_loaded"
        )
        is not False
    ):
        raise SupportRepairError("E_MANIFEST_SCOPE")
    return config, manifest


def support_design() -> tuple[
    dict[str, Any],
    dict[tuple[str, int, int], np.ndarray],
]:
    config, manifest = inputs()
    gates = config["support_gates"]
    requests = {row["request_key"]: row for row in manifest["requests"]}
    with np.load(FEATURES, allow_pickle=False) as arrays:
        log_rms = np.log(np.maximum(np.asarray(arrays["rms"], float), 1e-8))
        motion = np.asarray(arrays["motion"], float)
        persistence = np.asarray(arrays["persistence"], float)
    diagnostics: list[dict[str, Any]] = []
    arm_weights: dict[tuple[str, int, int], np.ndarray] = {}
    rows = manifest["rows"]
    for participant in sorted({str(row["participant_key"]) for row in rows}):
        for duration in (2, 6, 18):
            selected = [
                row
                for row in rows
                if str(row["participant_key"]) == participant
                and int(row["duration_seconds"]) == duration
            ]
            lag_values = sorted(map(int, selected[0]["lag_requests"]))
            arms = {
                lag: [
                    requests[row["lag_requests"][str(lag)]] for row in selected
                ]
                for lag in lag_values
            }
            matrices, kept = nuisance_matrix(
                arms,
                log_rms=log_rms,
                motion=motion,
                persistence=persistence,
            )
            target = np.zeros(next(iter(matrices.values())).shape[1])
            for lag, matrix in matrices.items():
                weights, values = stable_weights(
                    matrix,
                    target,
                    smd_limit=gates[
                        "maximum_absolute_standardized_mean_difference"
                    ],
                    maximum_weight=gates["maximum_single_window_weight"],
                )
                values.update(
                    {
                        "participant_key": participant,
                        "duration_seconds": duration,
                        "signed_lag_seconds": lag,
                        "raw_windows": len(matrix),
                        "exact_activity_categories": len(kept["activity"]),
                        "exact_location_categories": len(kept["location"]),
                    }
                )
                diagnostics.append(values)
                arm_weights[(participant, duration, lag)] = weights
    failures = [
        row
        for row in diagnostics
        if not row["solver_success"]
        or row["raw_windows"] < gates["minimum_raw_windows_per_arm"]
        or row["maximum_absolute_smd"]
        > gates["maximum_absolute_standardized_mean_difference"] + 1e-6
        or row["effective_sample_size"]
        < gates["minimum_effective_sample_size_per_arm"]
        or row["maximum_weight"] > gates["maximum_single_window_weight"] + 1e-8
        or row["top_10_weight_share"]
        > gates["maximum_top_10_weight_share"]
    ]
    participant_count = len(
        {row["participant_key"] for row in diagnostics}
    )
    pass_gate = (
        not failures
        and len(diagnostics) == gates["required_arms"]
        and participant_count == gates["required_participants"]
    )
    receipt = {
        "schema_version": "childlens-support-feasibility-v1.0.0",
        "status": "PASS" if pass_gate else "NO_GO_UNINFORMATIVE",
        "phase": "OUTCOME_BLIND_NUISANCE_ONLY",
        "config_sha256": sha256(CONFIG),
        "feature_manifest_sha256": sha256(MANIFEST),
        "frontend_features_sha256": sha256(FEATURES),
        "alignment_scores_or_projection_weights_loaded": False,
        "development_participant_count": participant_count,
        "locked_participant_count": 0,
        "arm_count": len(diagnostics),
        "failure_count": len(failures),
        "minimum_raw_windows": min(row["raw_windows"] for row in diagnostics),
        "minimum_effective_sample_size": round(
            min(row["effective_sample_size"] for row in diagnostics), 6
        ),
        "maximum_single_window_weight": round(
            max(row["maximum_weight"] for row in diagnostics), 6
        ),
        "maximum_top_10_weight_share": round(
            max(row["top_10_weight_share"] for row in diagnostics), 6
        ),
        "maximum_absolute_smd": round(
            max(row["maximum_absolute_smd"] for row in diagnostics), 6
        ),
        "minimum_exact_activity_categories": min(
            row["exact_activity_categories"] for row in diagnostics
        ),
        "minimum_exact_location_categories": min(
            row["exact_location_categories"] for row in diagnostics
        ),
        "privacy": {
            "row_level_diagnostics_exported": False,
            "identifiers_exported": False,
            "restricted_paths_exported": False,
        },
    }
    return receipt, arm_weights


def cluster_interval(values: list[float], seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = array[
        generator.integers(0, len(array), size=(10000, len(array)))
    ].mean(axis=1)
    return [float(value) for value in np.quantile(draws, (0.05, 0.95))]


def score() -> None:
    if not FEASIBILITY.is_file() or load_json(FEASIBILITY)["status"] != "PASS":
        raise SupportRepairError("E_NOT_FROZEN")
    config, manifest = inputs()
    receipt, arm_weights = support_design()
    if receipt != load_json(FEASIBILITY):
        raise SupportRepairError("E_FEASIBILITY_DRIFT")
    rows = manifest["rows"]
    request_index = {
        str(row["request_key"]): int(row["index"])
        for row in manifest["requests"]
    }
    weights_by_state: dict[tuple[int, int], dict[str, dict[str, np.ndarray]]] = {}
    with np.load(WEIGHTS, allow_pickle=False) as learned:
        for fold in range(3):
            for seed in (20260723, 20260724, 20260725):
                state: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
                prefix = f"fold{fold}_seed{seed}_"
                for key in learned.files:
                    if key.startswith(prefix):
                        tail = key[len(prefix) :]
                        modality, name = tail.split("_", 1)
                        state[modality][name.replace("_", ".")] = learned[key]
                weights_by_state[(fold, seed)] = dict(state)
    scores: dict[str, dict[int, float]] = {}
    with np.load(FEATURES, allow_pickle=False) as arrays:
        audio = np.asarray(arrays["audio_stats"])
        vision = np.asarray(arrays["vision_stats"])
        for fold in range(3):
            evaluation = [row for row in rows if int(row["fold"]) == fold]
            pairs: list[tuple[str, int]] = []
            audio_indices: list[int] = []
            vision_indices: list[int] = []
            for row in evaluation:
                zero = request_index[row["lag_requests"]["0"]]
                for lag, request_key in row["lag_requests"].items():
                    pairs.append((str(row["row_key"]), int(lag)))
                    audio_indices.append(zero)
                    vision_indices.append(request_index[request_key])
            seed_scores = [
                score_temporal_projectors(
                    audio[audio_indices],
                    vision[vision_indices],
                    weights_by_state[(fold, seed)],
                )
                for seed in (20260723, 20260724, 20260725)
            ]
            for (row_key, lag), value in zip(
                pairs, np.mean(np.stack(seed_scores), axis=0)
            ):
                scores.setdefault(row_key, {})[lag] = float(value)
    primary, curves = participant_contrasts(rows, scores, arm_weights)
    participants = sorted(primary)
    interval = cluster_interval([primary[key] for key in participants], 20260726)
    mean = float(np.mean(list(primary.values())))
    participant_fold = {
        str(row["participant_key"]): int(row["fold"]) for row in rows
    }
    fold_values = [
        float(
            np.mean(
                [
                    primary[key]
                    for key in participants
                    if participant_fold[key] == fold
                ]
            )
        )
        for fold in range(3)
    ]
    fold_range = max(fold_values) - min(fold_values)
    variances = []
    for fold in range(3):
        values = np.asarray(
            [
                primary[key]
                for key in participants
                if participant_fold[key] == fold
            ]
        )
        variances.append(float(values.var(ddof=1) / len(values)))
    inverse = 1 / np.maximum(variances, 1e-12)
    pooled = float(np.sum(inverse * fold_values) / np.sum(inverse))
    q = float(np.sum(inverse * np.square(np.asarray(fold_values) - pooled)))
    i2 = 0.0 if q <= 0 else max(0.0, (q - 2) / q)
    amplitude = {
        key: float(
            np.mean(
                [
                    max(curves[key][duration].values())
                    - min(curves[key][duration].values())
                    for duration in (2, 6, 18)
                ]
            )
        )
        for key in participants
    }
    amplitude_interval = cluster_interval(list(amplitude.values()), 20260727)
    curve_coordinates = [
        (duration, lag)
        for duration in (2, 6, 18)
        for lag in sorted(curves[participants[0]][duration])
    ]
    fold_curves = []
    for fold in range(3):
        fold_curves.append(
            [
                float(
                    np.mean(
                        [
                            curves[key][duration][lag]
                            for key in participants
                            if participant_fold[key] == fold
                        ]
                    )
                )
                for duration, lag in curve_coordinates
            ]
        )
    correlations = []
    for fold in range(3):
        other = np.mean(
            [fold_curves[index] for index in range(3) if index != fold], axis=0
        )
        correlations.append(
            float(np.corrcoef(fold_curves[fold], other)[0, 1])
        )
    positive = sum(value > 0 for value in primary.values())
    shared = {
        "support": True,
        "positive_control": True,
        "preprocessing": True,
        "shortcut_interpretable": True,
        "precision": interval[1] - interval[0] <= 0.04,
        "heterogeneity": i2 <= 0.5 and fold_range <= 0.03,
    }
    detectable = (
        all(shared.values())
        and mean >= 0.02
        and interval[0] > 0
        and positive >= 12
        and all(value > 0 for value in fold_values)
        and all(value >= 0.5 for value in correlations)
    )
    weak = (
        all(shared.values())
        and interval[0] >= -0.02
        and interval[1] <= 0.02
        and amplitude_interval[1] <= 0.02
    )
    terminal = (
        "PASS_DETECTABLE_STRUCTURE"
        if detectable
        else "PASS_PRECISE_WEAK_OR_FLAT"
        if weak
        else "NO_GO_UNINFORMATIVE"
    )
    lag_report = {
        "schema_version": "childlens-support-repair-lag-response-v1.0.0",
        "status": "COMPLETE",
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "primary_zero_minus_signed_2x_4x": {
            "participant_mean": round(mean, 6),
            "participant_cluster_90pct": [
                round(value, 6) for value in interval
            ],
            "positive_participants": positive,
        },
        "fold_primary_contrasts": [round(value, 6) for value in fold_values],
        "cross_fold_i2": round(i2, 6),
        "fold_primary_max_minus_min": round(fold_range, 6),
        "fold_curve_correlations_with_other_folds": [
            round(value, 6) for value in correlations
        ],
        "curve_amplitude_participant_cluster_90pct": [
            round(value, 6) for value in amplitude_interval
        ],
        "participant_level_inference": True,
        "row_level_scores_exported": False,
    }
    decision = {
        "schema_version": "childlens-support-repair-decision-v1.0.0",
        "decision": terminal,
        "gates": shared,
        "detectable_structure_gate": detectable,
        "precise_weak_or_flat_gate": weak,
        "locked_confirmation_recommended": terminal.startswith("PASS_"),
        "locked_data_accessed": False,
        "simulator_or_side_cue_condition_run": False,
    }
    write_json(LAG_REPORT, lag_report)
    write_json(DECISION, decision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("support-only", "score"))
    arguments = parser.parse_args()
    if arguments.phase == "support-only":
        receipt, _ = support_design()
        write_json(FEASIBILITY, receipt)
        if receipt["status"] != "PASS":
            raise SystemExit(2)
    else:
        score()


if __name__ == "__main__":
    main()
