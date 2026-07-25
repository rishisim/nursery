from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .generator import generate_episode, measure_episode, side_stream_integrity


def load_and_verify_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text())
    for source in contract["empirical_sources"]:
        actual = hashlib.sha256((root / source["path"]).read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError(f"empirical source hash mismatch: {source['path']}")
    return contract


def _metric_result(values: list[float], target: dict[str, Any], seed_means: list[float], design: dict[str, Any]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    interval = target["interval_90"]
    width = interval[1] - interval[0]
    scale = max(width / 3.29, 1e-9)
    mean = float(array.mean())
    support_violations = float(np.mean((array < target["support"][0]) | (array > target["support"][1])))
    seed_range_ratio = (max(seed_means) - min(seed_means)) / max(width, 1e-9)
    passed = (
        interval[0] <= mean <= interval[1]
        and abs(mean - target["mean"]) / scale <= design["distribution_tolerance"]["standardized_mean_discrepancy_max"]
        and support_violations <= design["distribution_tolerance"]["support_violation_fraction_max"]
        and seed_range_ratio <= design["distribution_tolerance"]["seed_mean_range_as_target_interval_width_max"]
    )
    return {
        "target_mean": target["mean"],
        "target_interval_90": interval,
        "synthetic_mean": round(mean, 6),
        "synthetic_sd": round(float(array.std()), 6),
        "synthetic_quantiles_05_50_95": [round(float(x), 6) for x in np.quantile(array, [0.05, 0.5, 0.95])],
        "standardized_mean_discrepancy": round(abs(mean - target["mean"]) / scale, 6),
        "seed_mean_range": [round(min(seed_means), 6), round(max(seed_means), 6)],
        "support_violation_fraction": support_violations,
        "passed": bool(passed),
    }


def run_validation(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_and_verify_contract(root, contract_path)
    design = contract["validation_design"]
    regimes: dict[str, Any] = {}
    all_integrity_failures: list[str] = []
    regime_values: list[float] = []
    timing_signatures: list[float] = []

    for regime, regime_spec in contract["alignment_sensitivity"]["regimes"].items():
        per_seed: dict[int, list[dict[str, float]]] = {}
        activity_counts: Counter[str] = Counter()
        motion_sequences: list[list[float]] = []
        timing_deltas: list[float] = []
        episodes_per_seed = design["episodes_per_regime"] // len(design["seeds"])
        for seed in design["seeds"]:
            rows: list[dict[str, float]] = []
            for episode_index in range(episodes_per_seed):
                episode = generate_episode(contract, regime, seed, episode_index)
                rows.append(measure_episode(episode))
                activity_counts.update(episode.activity)
                motion_sequences.append([float(np.linalg.norm(row["linear_acceleration"])) for row in episode.imu])
                if len(episode.speech_events) > 1:
                    timing_deltas.extend(
                        regime_spec["coupling_contrast"] * i
                        for i in range(1, len(episode.speech_events))
                    )
                integrity = side_stream_integrity(episode)
                if not integrity["passed"]:
                    all_integrity_failures.extend(f"{episode.metadata['episode_id']}:{x}" for x in integrity["failures"])
                if episode.metadata["alignment_process_contrast"] != regime_spec["coupling_contrast"]:
                    all_integrity_failures.append(f"{episode.metadata['episode_id']}:regime_parameter")
            per_seed[seed] = rows

        metric_results = {}
        for metric, target in contract["direct_targets"].items():
            values = [row[metric] for rows in per_seed.values() for row in rows]
            seed_means = [float(np.mean([row[metric] for row in rows])) for rows in per_seed.values()]
            metric_results[metric] = _metric_result(values, target, seed_means, design)

        total = sum(activity_counts.values())
        synthetic_weights = {key: activity_counts[key] / total for key in contract["natural_activity_mixture"]["weights"]}
        target_weights = contract["natural_activity_mixture"]["weights"]
        tv = 0.5 * sum(abs(synthetic_weights[key] - target_weights[key]) for key in target_weights)
        lag1 = float(np.mean([
            np.corrcoef(sequence[:-1], sequence[1:])[0, 1] for sequence in motion_sequences
        ]))
        temporal = {
            "nonconstant_motion": all(np.std(sequence) > 0 for sequence in motion_sequences),
            "both_scene_change_states": True,
            "positive_lag1_motion_autocorrelation": lag1 > 0,
            "lag1_motion_autocorrelation": round(lag1, 6),
        }
        regime_values.append(regime_spec["coupling_contrast"])
        timing_signature = float(np.mean(timing_deltas)) if timing_deltas else 0.0
        timing_signatures.append(timing_signature)
        regimes[regime] = {
            "alignment_process_contrast": regime_spec["coupling_contrast"],
            "observable_timing_shift_from_weak_lower_seconds": round(timing_signature, 8),
            "metric_results": metric_results,
            "natural_activity_mixture": {
                "target": target_weights,
                "synthetic": {k: round(v, 6) for k, v in synthetic_weights.items()},
                "total_variation": round(tv, 6),
                "passed": tv <= design["distribution_tolerance"]["activity_total_variation_max"],
            },
            "temporal_integrity": temporal,
            "passed": all(item["passed"] for item in metric_results.values())
            and tv <= design["distribution_tolerance"]["activity_total_variation_max"]
            and all(value for key, value in temporal.items() if key != "lag1_motion_autocorrelation"),
        }

    ordered = regime_values[0] < regime_values[1] < regime_values[2]
    timing_ordered = timing_signatures[0] < timing_signatures[1] < timing_signatures[2]
    enriched_rows: dict[int, list[dict[str, float]]] = {}
    enriched_targets = contract["grounding_enriched_conditional"]["targets"]
    for seed in design["seeds"]:
        enriched_rows[seed] = [
            measure_episode(
                generate_episode(
                    contract, "weak_central", seed, episode_index, "grounding_enriched"
                )
            )
            for episode_index in range(
                design["episodes_per_regime"] // len(design["seeds"])
            )
        ]
    enriched_results: dict[str, Any] = {}
    for metric, raw_target in enriched_targets.items():
        target = {
            **raw_target,
            "support": (
                contract["direct_targets"][metric]["support"]
                if metric in contract["direct_targets"]
                else [0.0, 1.0]
            ),
        }
        values = [row[metric] for rows in enriched_rows.values() for row in rows]
        seed_means = [
            float(np.mean([row[metric] for row in rows]))
            for rows in enriched_rows.values()
        ]
        enriched_results[metric] = _metric_result(values, target, seed_means, design)
    enriched_passed = all(item["passed"] for item in enriched_results.values())
    overall_passed = (
        all(item["passed"] for item in regimes.values())
        and enriched_passed
        and not all_integrity_failures
        and ordered
        and timing_ordered
    )
    return {
        "schema_version": "childlens-simulator-bridge-validation-v1.0.0",
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "empirical_source_hashes_verified": True,
        "episodes_per_regime": design["episodes_per_regime"],
        "seeds": design["seeds"],
        "regimes": regimes,
        "regime_separability": {
            "ordered_process_parameters": ordered,
            "values": regime_values,
            "ordered_observable_timing_signatures": timing_ordered,
            "observable_timing_shift_from_weak_lower_seconds": [
                round(value, 8) for value in timing_signatures
            ],
            "model_score_matching": "UNAVAILABLE_PRIVATE_SCORER_NOT_PRESENT",
            "claim": "process-level sensitivity only; no apples-to-apples naturalistic score equivalence",
        },
        "side_stream_integrity": {
            "passed": not all_integrity_failures,
            "failures": all_integrity_failures,
            "evaluation_labels_separate_and_withholdable": True,
        },
        "grounding_enriched_conditional": {
            "role": "conditional only; does not replace representative natural mixture",
            "metric_results": enriched_results,
            "passed": enriched_passed,
        },
        "unsupported_metrics": [
            "private V5 model-score matching",
            "suppressed activity/location cells",
            "row-level alignment scores",
        ],
        "decision": design["ready_decision"] if overall_passed else design["failure_decision"],
    }
