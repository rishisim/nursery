from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from .protocol import canonical_digest


def _unique_top(scores: Sequence[float], tolerance: float) -> tuple[int, bool, int]:
    values = np.asarray(scores, dtype=np.float64)
    maximum = float(values.max())
    tied = np.flatnonzero(np.abs(values - maximum) <= tolerance)
    return int(tied[0]), len(tied) == 1, int(len(tied))


def _rank_of_answer(scores: Sequence[float], answer: int, tolerance: float) -> int:
    values = np.asarray(scores, dtype=np.float64)
    truth = float(values[int(answer)])
    strictly_greater = int(np.sum(values > truth + tolerance))
    tied = int(np.sum(np.abs(values - truth) <= tolerance))
    return strictly_greater + tied


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "strict_top1": 0.0,
            "mean_rank": 0.0,
            "mrr": 0.0,
            "multiclass_log_loss": 0.0,
            "multiclass_brier": 0.0,
            "tie_rate": 0.0,
            "exact_chance": 0.0,
        }
    groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["concept_group"])].append(row)
    group_accuracies = [
        float(np.mean([float(row["strict_correct"]) for row in values]))
        for _, values in sorted(groups.items())
    ]
    return {
        "count": len(rows),
        "concept_group_count": len(groups),
        "strict_top1": float(np.mean(group_accuracies)),
        "item_strict_top1": float(
            np.mean([float(row["strict_correct"]) for row in rows])
        ),
        "mean_rank": float(np.mean([int(row["rank"]) for row in rows])),
        "mrr": float(np.mean([1.0 / int(row["rank"]) for row in rows])),
        "multiclass_log_loss": float(
            np.mean([-math.log(max(float(row["correct_probability"]), 1e-15)) for row in rows])
        ),
        "multiclass_brier": float(
            np.mean([float(row["brier"]) for row in rows])
        ),
        "tie_rate": float(np.mean([not bool(row["unique_top"]) for row in rows])),
        "mean_tied_maxima": float(np.mean([int(row["tied_maxima"]) for row in rows])),
        "exact_chance": float(np.mean([float(row["exact_chance"]) for row in rows])),
        "strict_unique_argmax_required": True,
        "ties_count_wrong": True,
    }


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    key_by_id = {str(row["prompt_id"]): row for row in keys}
    if len(key_by_id) != len(keys):
        raise ValueError("duplicate evaluation key prompt id")
    prediction_by_id = {str(row["prompt_id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("duplicate prediction prompt id")
    if set(key_by_id) != set(prediction_by_id):
        raise ValueError("prediction/key prompt set mismatch")
    tolerance = float(config["learner"]["exact_tie_tolerance"])
    rows = []
    for prompt_id in sorted(key_by_id):
        key = key_by_id[prompt_id]
        prediction = prediction_by_id[prompt_id]
        scores = list(map(float, prediction["scores"]))
        probabilities = np.asarray(prediction["probabilities"], dtype=np.float64)
        answer = int(key["answer_index"])
        expected_candidate_count = int(round(1.0 / float(key["exact_chance"])))
        if (
            len(scores) != len(probabilities)
            or len(scores) != expected_candidate_count
            or not 0 <= answer < len(scores)
            or not np.all(np.isfinite(probabilities))
            or not np.all((0.0 <= probabilities) & (probabilities <= 1.0))
        ):
            raise ValueError("malformed scored prediction")
        if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-12):
            raise ValueError("prediction probabilities do not sum to one")
        top, unique, tied = _unique_top(scores, tolerance)
        target = np.zeros(len(probabilities), dtype=float)
        target[answer] = 1.0
        rows.append(
            {
                "prompt_id": prompt_id,
                "kind": str(key["kind"]),
                "presence": key.get("presence"),
                "concept_group": str(key["concept_group"]),
                "answer_index": answer,
                "predicted_index": top,
                "unique_top": unique,
                "tied_maxima": tied,
                "strict_correct": bool(unique and top == answer),
                "rank": _rank_of_answer(scores, answer, tolerance),
                "correct_probability": float(probabilities[answer]),
                "brier": float(np.mean((probabilities - target) ** 2)),
                "exact_chance": float(key["exact_chance"]),
            }
        )
    by_kind = {
        kind: _summarize_rows([row for row in rows if row["kind"] == kind])
        for kind in ("lexical", "composition", "presence", "zero_exposure")
    }
    by_presence = {
        presence: _summarize_rows(
            [
                row
                for row in rows
                if row["kind"] == "presence" and row["presence"] == presence
            ]
        )
        for presence in ("present", "null")
    }
    return {
        "schema_version": "nursery-corrective-score-v1",
        "by_kind": by_kind,
        "by_presence": by_presence,
        "lexical_acquisition_top1": by_kind["lexical"]["strict_top1"],
        "heldout_composition_top1": by_kind["composition"]["strict_top1"],
        "presence_balanced_accuracy": float(
            np.mean(
                [
                    by_presence["present"]["strict_top1"],
                    by_presence["null"]["strict_top1"],
                ]
            )
        ),
        "item_rows": rows,
    }


def average_model_replicates(
    scored_units: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    purpose: str,
) -> dict[str, Any]:
    expected_models = list(map(int, config["resolved_registries"][purpose]["model"]))
    grouped: defaultdict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_units:
        grouped[(int(row["corpus_seed"]), str(row["condition"]))].append(row)
    output = []
    variation_rows = []
    metric_names = (
        "lexical_acquisition_top1",
        "heldout_composition_top1",
        "presence_balanced_accuracy",
    )
    for (corpus_seed, condition), values in sorted(grouped.items()):
        by_model = {int(row["model_seed"]): row for row in values}
        if len(by_model) != len(values):
            raise RuntimeError(f"duplicate model replicate: {corpus_seed}/{condition}")
        if list(sorted(by_model)) != list(sorted(expected_models)):
            raise RuntimeError(f"model replicate set mismatch: {corpus_seed}/{condition}")
        ordered = [by_model[seed] for seed in expected_models]
        metrics = {
            name: float(
                math.fsum(float(row["metrics"][name]) for row in ordered)
                / len(ordered)
            )
            for name in metric_names
        }
        for kind in ("lexical", "composition", "presence", "zero_exposure"):
            for field in (
                "strict_top1",
                "mean_rank",
                "mrr",
                "multiclass_log_loss",
                "multiclass_brier",
                "tie_rate",
                "exact_chance",
            ):
                metrics[f"{kind}.{field}"] = float(
                    math.fsum(
                        float(row["metrics"]["by_kind"][kind][field]) for row in ordered
                    )
                    / len(ordered)
                )
        for presence in ("present", "null"):
            metrics[f"presence.{presence}.strict_top1"] = float(
                math.fsum(
                    float(row["metrics"]["by_presence"][presence]["strict_top1"])
                    for row in ordered
                )
                / len(ordered)
            )
        output.append(
            {
                "corpus_seed": corpus_seed,
                "condition": condition,
                "model_seed_order": expected_models,
                "model_replicates_averaged": len(ordered),
                "metrics": metrics,
            }
        )
        for name in metric_names:
            observations = [float(row["metrics"][name]) for row in ordered]
            variation_rows.append(
                {
                    "corpus_seed": corpus_seed,
                    "condition": condition,
                    "metric": name,
                    "sample_sd": float(np.std(observations, ddof=1)),
                    "range": float(max(observations) - min(observations)),
                    "values": observations,
                }
            )
    return {
        "schema_version": "nursery-corrective-model-average-v1",
        "independent_unit": "corpus_seed",
        "model_seed_order": expected_models,
        "exact_model_seed_set_required": True,
        "floating_reduction": "math.fsum_in_frozen_model_seed_order",
        "rows": output,
        "within_corpus_model_variation": variation_rows,
    }


def _t_summary(
    values: Sequence[float],
    *,
    confidence: float,
) -> dict[str, Any]:
    array = np.asarray(list(map(float, values)), dtype=np.float64)
    if len(array) < 2:
        raise ValueError("at least two corpora required for t inference")
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1))
    standard_error = sample_sd / math.sqrt(len(array))
    critical = float(stats.t.ppf(confidence, df=len(array) - 1))
    lcb = mean - critical * standard_error
    ucb = mean + critical * standard_error
    return {
        "n": len(array),
        "mean": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "degrees_of_freedom": len(array) - 1,
        "one_sided_confidence_level": confidence,
        "t_critical": critical,
        "one_sided_lcb": lcb,
        "one_sided_ucb": ucb,
        "median": float(np.median(array)),
        "mad": float(np.median(np.abs(array - np.median(array)))),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "values": list(map(float, array)),
        "values_digest": canonical_digest(list(map(float, array))),
    }


def paired_t_summary(
    values: Sequence[float],
    *,
    confidence: float,
) -> dict[str, Any]:
    return _t_summary(values, confidence=confidence)


def equivalence_interval(
    values: Sequence[float],
    *,
    confidence: float,
) -> dict[str, Any]:
    array = np.asarray(list(map(float, values)), dtype=np.float64)
    if len(array) < 2:
        return {"status": "FAIL", "problem": "fewer_than_two_corpora", "n": len(array)}
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1))
    return {
        "status": "PASS",
        "n": len(array),
        "confidence_level": float(confidence),
        "mean": mean,
        "lower": mean - critical * standard_error,
        "upper": mean + critical * standard_error,
        "critical": critical,
        "standard_error": standard_error,
        "values": list(map(float, array)),
        "values_digest": canonical_digest(list(map(float, array))),
    }


def welch_difference_summary(
    high: Sequence[float],
    low: Sequence[float],
    *,
    confidence: float,
) -> dict[str, Any]:
    first = np.asarray(list(map(float, high)), dtype=np.float64)
    second = np.asarray(list(map(float, low)), dtype=np.float64)
    if len(first) < 2 or len(second) < 2:
        return {
            "status": "FAIL",
            "problem": "fewer_than_two_corpora_per_group",
            "high_n": len(first),
            "low_n": len(second),
        }
    first_variance = float(first.var(ddof=1))
    second_variance = float(second.var(ddof=1))
    variance = first_variance / len(first) + second_variance / len(second)
    standard_error = math.sqrt(max(variance, 0.0))
    if variance <= 1e-30:
        degrees = float(len(first) + len(second) - 2)
        critical = float(stats.t.ppf(confidence, degrees))
    else:
        numerator = variance**2
        denominator = (
            (first_variance / len(first)) ** 2 / (len(first) - 1)
            + (second_variance / len(second)) ** 2 / (len(second) - 1)
        )
        degrees = numerator / denominator if denominator > 0 else float("inf")
        critical = float(stats.t.ppf(confidence, degrees))
    difference = float(first.mean() - second.mean())
    return {
        "status": "PASS",
        "high_n": len(first),
        "low_n": len(second),
        "high_mean": float(first.mean()),
        "low_mean": float(second.mean()),
        "difference": difference,
        "standard_error": standard_error,
        "degrees_of_freedom": degrees,
        "one_sided_confidence_level": confidence,
        "one_sided_lcb": difference - critical * standard_error,
        "high_values_digest": canonical_digest(list(map(float, first))),
        "low_values_digest": canonical_digest(list(map(float, second))),
    }


def analyze_informativeness(
    averaged: Mapping[str, Any],
    mutation_summary: Mapping[str, Any],
    corpus_audits: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    purpose: str,
) -> dict[str, Any]:
    table = {
        (int(row["corpus_seed"]), str(row["condition"])): row["metrics"]
        for row in averaged["rows"]
    }
    audits = {int(row["corpus_seed"]): row for row in corpus_audits}
    corpora = list(map(int, config["resolved_registries"][purpose]["corpus"]))
    if set(corpora) != set(audits):
        raise RuntimeError("informativeness corpus audit set mismatch")
    confidence = float(config["analysis"]["primary_one_sided_confidence_level"])
    equivalence_confidence = float(
        config["qualification_gates"]["equivalence_confidence_level"]
    )
    margin = float(config["qualification_gates"]["uninformative_equivalence_margin"])
    endpoints = (
        "lexical_acquisition_top1",
        "heldout_composition_top1",
    )
    details: dict[str, Any] = {}
    gates: dict[str, bool] = {}

    negative_headroom = {}
    for endpoint in endpoints:
        rows = {}
        for condition in config["design"]["negative_controls"]:
            values = [float(table[(seed, condition)][endpoint]) for seed in corpora]
            rows[str(condition)] = {
                "mean": float(np.mean(values)),
                "maximum_allowed": float(
                    config["qualification_gates"]["maximum_negative_control_top1"]
                ),
                "passed": float(np.mean(values))
                <= float(config["qualification_gates"]["maximum_negative_control_top1"]),
            }
        negative_headroom[endpoint] = rows
    details["negative_control_headroom"] = negative_headroom
    gates["negative_controls_retain_headroom"] = all(
        row["passed"]
        for endpoint in negative_headroom.values()
        for row in endpoint.values()
    )

    oracle_rows = {}
    for endpoint in endpoints:
        oracle = [float(table[(seed, "oracle_alignment")][endpoint]) for seed in corpora]
        comparator = [
            max(
                float(table[(seed, "absent")][endpoint]),
                float(table[(seed, "shuffled")][endpoint]),
            )
            for seed in corpora
        ]
        accuracy = paired_t_summary(oracle, confidence=confidence)
        lift = paired_t_summary(
            [value - baseline for value, baseline in zip(oracle, comparator)],
            confidence=confidence,
        )
        passed = (
            accuracy["mean"]
            >= float(config["qualification_gates"]["minimum_oracle_top1"])
            and accuracy["one_sided_lcb"]
            > float(config["qualification_gates"]["minimum_oracle_lcb"])
            and lift["one_sided_lcb"]
            > float(config["qualification_gates"]["minimum_oracle_lift_lcb"])
        )
        oracle_rows[endpoint] = {"accuracy": accuracy, "lift": lift, "status": passed}
    details["oracle"] = oracle_rows
    gates["oracle_accuracy_and_lift"] = all(row["status"] for row in oracle_rows.values())

    equivalence_rows = {}
    disconnected_rows = {
        int(row["corpus_seed"]): row
        for row in mutation_summary["mutations"]["evidence_disconnected"]["corpus_rows"]
    }
    for endpoint, mutation_field in (
        ("lexical_acquisition_top1", "lexical_absolute"),
        ("heldout_composition_top1", "composition_absolute"),
    ):
        contrasts = {
            "uninformative_minus_absent": [
                float(table[(seed, "uninformative")][endpoint])
                - float(table[(seed, "absent")][endpoint])
                for seed in corpora
            ],
            "evidence_disconnected_minus_absent": [
                float(disconnected_rows[seed][mutation_field])
                - float(table[(seed, "absent")][endpoint])
                for seed in corpora
            ],
        }
        endpoint_rows = {}
        for name, values in contrasts.items():
            interval = equivalence_interval(values, confidence=equivalence_confidence)
            passed = (
                interval.get("status") == "PASS"
                and float(interval["lower"]) >= -margin
                and float(interval["upper"]) <= margin
            )
            endpoint_rows[name] = {**interval, "equivalence_margin": margin, "passed": passed}
        equivalence_rows[endpoint] = endpoint_rows
    details["negative_equivalence"] = equivalence_rows
    gates["uninformative_and_disconnected_equivalent_to_absent"] = all(
        contrast["passed"]
        for endpoint in equivalence_rows.values()
        for contrast in endpoint.values()
    )

    corrupted_rows = {}
    for endpoint in endpoints:
        summary = paired_t_summary(
            [
                float(table[(seed, "corrupted")][endpoint])
                - float(table[(seed, "absent")][endpoint])
                for seed in corpora
            ],
            confidence=confidence,
        )
        passed = summary["one_sided_ucb"] < float(
            config["qualification_gates"]["maximum_corrupted_gain_ucb"]
        )
        corrupted_rows[endpoint] = {**summary, "passed": passed}
    details["corrupted_vs_absent"] = corrupted_rows
    gates["corrupted_cannot_help"] = all(row["passed"] for row in corrupted_rows.values())

    shift_rows = {}
    for endpoint in endpoints:
        endpoint_rows = {}
        for condition in ("shift_minus", "shift_plus"):
            summary = paired_t_summary(
                [
                    float(table[(seed, "synchronized")][endpoint])
                    - float(table[(seed, condition)][endpoint])
                    for seed in corpora
                ],
                confidence=confidence,
            )
            passed = summary["one_sided_lcb"] > float(
                config["qualification_gates"]["minimum_shift_lcb"]
            )
            endpoint_rows[condition] = {**summary, "passed": passed}
        shift_rows[endpoint] = endpoint_rows
    details["signed_shifts"] = shift_rows
    gates["both_signed_shifts_disrupt"] = all(
        row["passed"] for endpoint in shift_rows.values() for row in endpoint.values()
    )

    midpoint = float(np.mean(config["design"]["side_informativeness_range"]))
    high_informativity = [
        seed for seed in corpora if float(audits[seed]["side_informativity_draw"]) >= midpoint
    ]
    low_informativity = [seed for seed in corpora if seed not in high_informativity]
    realized_detector_accuracy = {
        seed: float(
            audits[seed]["condition_audit"]["detector_manipulation"]["synchronized"][
                "strict_event_or_null_accuracy"
            ]
        )
        for seed in corpora
    }
    interaction_summary = welch_difference_summary(
        [realized_detector_accuracy[seed] for seed in high_informativity],
        [realized_detector_accuracy[seed] for seed in low_informativity],
        confidence=confidence,
    )
    interaction_passed = interaction_summary.get("status") == "PASS" and interaction_summary[
        "one_sided_lcb"
    ] > float(config["qualification_gates"]["minimum_informativeness_interaction_lcb"])
    details["side_informativeness_interaction"] = {
        "fixed_midpoint": midpoint,
        "high_corpora": high_informativity,
        "low_corpora": low_informativity,
        "estimand": "high_minus_low_synchronized_detector_event_or_null_accuracy",
        "summary": interaction_summary,
        "passed": interaction_passed,
    }
    gates["side_informativeness_interaction"] = interaction_passed

    lag_threshold = float(config["qualification_gates"]["high_lag_rms_threshold"])
    lag_rms = {
        seed: math.sqrt(
            float(audits[seed]["lag_mean_draw"]) ** 2
            + float(audits[seed]["lag_sd_draw"]) ** 2
        )
        for seed in corpora
    }
    high_lag = [seed for seed in corpora if lag_rms[seed] >= lag_threshold]
    low_lag = [seed for seed in corpora if seed not in high_lag]
    lag_rows = {}
    for endpoint in endpoints:
        exact_drop = welch_difference_summary(
            [float(table[(seed, "exact_window")][endpoint]) for seed in low_lag],
            [float(table[(seed, "exact_window")][endpoint]) for seed in high_lag],
            confidence=confidence,
        )
        latent_recovery = (
            paired_t_summary(
                [
                    float(table[(seed, "synchronized")][endpoint])
                    - float(table[(seed, "exact_window")][endpoint])
                    for seed in high_lag
                ],
                confidence=confidence,
            )
            if len(high_lag) >= 2
            else {"status": "FAIL", "problem": "fewer_than_two_high_lag_corpora"}
        )
        exact_pass = exact_drop.get("status") == "PASS" and exact_drop[
            "one_sided_lcb"
        ] > float(config["qualification_gates"]["minimum_exact_window_high_lag_drop"])
        recovery_pass = latent_recovery.get("status") != "FAIL" and latent_recovery[
            "one_sided_lcb"
        ] > float(config["qualification_gates"]["minimum_latent_high_lag_recovery"])
        lag_rows[endpoint] = {
            "exact_window_low_minus_high": {**exact_drop, "passed": exact_pass},
            "high_lag_synchronized_minus_exact": {
                **latent_recovery,
                "passed": recovery_pass,
            },
        }
    lag_counts_pass = (
        len(high_lag)
        >= int(config["qualification_gates"]["minimum_high_lag_corpus_count"])
        and len(low_lag)
        >= int(config["qualification_gates"]["minimum_low_lag_corpus_count"])
    )
    details["temporal_uncertainty"] = {
        "lag_rms_threshold": lag_threshold,
        "lag_rms": lag_rms,
        "high_lag_corpora": high_lag,
        "low_lag_corpora": low_lag,
        "count_gate": lag_counts_pass,
        "endpoints": lag_rows,
    }
    gates["exact_window_degrades_and_latent_recovers_at_high_lag"] = lag_counts_pass and all(
        row["passed"]
        for endpoint in lag_rows.values()
        for row in endpoint.values()
    )

    null_rows = mutation_summary["mutations"]["null_head_ablation"]["corpus_rows"]
    null_drop = paired_t_summary(
        [-float(row["null_change"]) for row in null_rows], confidence=confidence
    )
    present_change = paired_t_summary(
        [float(row["present_change"]) for row in null_rows], confidence=confidence
    )
    null_pass = (
        null_drop["mean"]
        >= float(config["qualification_gates"]["minimum_null_ablation_drop"])
        and null_drop["one_sided_lcb"]
        > float(config["qualification_gates"]["minimum_null_ablation_lcb"])
        and present_change["one_sided_lcb"]
        > -float(config["qualification_gates"]["maximum_null_ablation_present_harm"])
    )
    details["null_head_ablation"] = {
        "active_minus_ablated_null": null_drop,
        "ablated_minus_active_present": present_change,
        "passed": null_pass,
    }
    gates["null_head_is_selectively_causal"] = null_pass

    null_side_rows = mutation_summary["mutations"]["null_side_zero"][
        "corpus_rows"
    ]
    null_side_drop = paired_t_summary(
        [-float(row["null_change"]) for row in null_side_rows],
        confidence=confidence,
    )
    null_side_present = paired_t_summary(
        [float(row["present_change"]) for row in null_side_rows],
        confidence=confidence,
    )
    null_side_equivalence = {}
    null_side_margin = float(
        config["qualification_gates"]["null_side_semantic_equivalence_margin"]
    )
    for field in ("lexical_change", "composition_change"):
        interval = equivalence_interval(
            [float(row[field]) for row in null_side_rows],
            confidence=equivalence_confidence,
        )
        passed = (
            interval.get("status") == "PASS"
            and float(interval["lower"]) >= -null_side_margin
            and float(interval["upper"]) <= null_side_margin
        )
        null_side_equivalence[field] = {**interval, "passed": passed}
    null_side_pass = (
        null_side_drop["one_sided_lcb"]
        > float(
            config["qualification_gates"]["minimum_null_side_mutation_drop_lcb"]
        )
        and null_side_present["one_sided_lcb"]
        > -float(config["qualification_gates"]["maximum_null_ablation_present_harm"])
        and all(row["passed"] for row in null_side_equivalence.values())
    )
    details["null_side_zero"] = {
        "active_minus_null_side_zero_null": null_side_drop,
        "null_side_zero_minus_active_present": null_side_present,
        "semantic_endpoint_equivalence": null_side_equivalence,
        "equivalence_margin": null_side_margin,
        "passed": null_side_pass,
    }
    gates["null_side_path_is_selectively_causal"] = null_side_pass

    null_update_rows = mutation_summary["mutations"]["null_update_disconnected"][
        "corpus_rows"
    ]
    null_update_drop = paired_t_summary(
        [-float(row["null_change"]) for row in null_update_rows],
        confidence=confidence,
    )
    null_update_present = paired_t_summary(
        [float(row["present_change"]) for row in null_update_rows],
        confidence=confidence,
    )
    null_update_equivalence = {}
    null_update_margin = float(
        config["qualification_gates"]["null_update_semantic_equivalence_margin"]
    )
    for field in ("lexical_change", "composition_change"):
        interval = equivalence_interval(
            [float(row[field]) for row in null_update_rows],
            confidence=equivalence_confidence,
        )
        passed = (
            interval.get("status") == "PASS"
            and float(interval["lower"]) >= -null_update_margin
            and float(interval["upper"]) <= null_update_margin
        )
        null_update_equivalence[field] = {**interval, "passed": passed}
    null_update_pass = (
        null_update_drop["one_sided_lcb"]
        > float(
            config["qualification_gates"][
                "minimum_null_update_disconnect_drop_lcb"
            ]
        )
        and null_update_present["one_sided_lcb"]
        > -float(config["qualification_gates"]["maximum_null_ablation_present_harm"])
        and all(row["passed"] for row in null_update_equivalence.values())
    )
    details["null_update_disconnect"] = {
        "mutation": "updates_frozen_at_configured_initial_null_prior",
        "active_minus_disconnected_null": null_update_drop,
        "disconnected_minus_active_present": null_update_present,
        "semantic_endpoint_equivalence": null_update_equivalence,
        "equivalence_margin": null_update_margin,
        "passed": null_update_pass,
    }
    gates["null_update_path_is_selectively_causal"] = null_update_pass

    semantic_side_rows = mutation_summary["mutations"]["semantic_side_zero"][
        "corpus_rows"
    ]
    semantic_side_drops = {}
    for field in ("lexical_change", "composition_change"):
        summary = paired_t_summary(
            [-float(row[field]) for row in semantic_side_rows], confidence=confidence
        )
        semantic_side_drops[field] = {
            **summary,
            "passed": summary["one_sided_lcb"]
            > float(
                config["qualification_gates"][
                    "minimum_side_semantic_mutation_drop_lcb"
                ]
            ),
        }
    semantic_null_interval = equivalence_interval(
        [float(row["null_change"]) for row in semantic_side_rows],
        confidence=equivalence_confidence,
    )
    semantic_null_margin = float(
        config["qualification_gates"]["semantic_ablation_null_equivalence_margin"]
    )
    semantic_null_pass = (
        semantic_null_interval.get("status") == "PASS"
        and float(semantic_null_interval["lower"]) >= -semantic_null_margin
        and float(semantic_null_interval["upper"]) <= semantic_null_margin
    )
    semantic_side_pass = all(
        row["passed"] for row in semantic_side_drops.values()
    ) and semantic_null_pass
    details["semantic_side_zero"] = {
        "semantic_endpoint_drops": semantic_side_drops,
        "null_endpoint_equivalence": {
            **semantic_null_interval,
            "equivalence_margin": semantic_null_margin,
            "passed": semantic_null_pass,
        },
        "passed": semantic_side_pass,
    }
    gates["semantic_side_path_is_selectively_causal"] = semantic_side_pass

    alignment_mutation_details = {}
    for mutation in ("within_bag_permuted", "old_agreement_gate"):
        rows = mutation_summary["mutations"][mutation]["corpus_rows"]
        endpoint_rows = {}
        for field in ("lexical_change", "composition_change"):
            summary = paired_t_summary(
                [-float(row[field]) for row in rows], confidence=confidence
            )
            endpoint_rows[field] = {
                **summary,
                "passed": summary["one_sided_lcb"]
                > float(
                    config["qualification_gates"][
                        "minimum_alignment_mutation_drop_lcb"
                    ]
                ),
            }
        alignment_mutation_details[mutation] = endpoint_rows
    details["alignment_mechanism_mutations"] = alignment_mutation_details
    gates["permutation_and_old_gate_fail_positive_control"] = all(
        row["passed"]
        for mutation in alignment_mutation_details.values()
        for row in mutation.values()
    )

    semantic_rows = mutation_summary["mutations"]["semantic_update_disconnected"][
        "corpus_rows"
    ]
    semantic_details = {}
    for field in ("lexical_change", "composition_change"):
        summary = paired_t_summary(
            [-float(row[field]) for row in semantic_rows], confidence=confidence
        )
        passed = summary["one_sided_lcb"] > float(
            config["qualification_gates"]["minimum_semantic_disconnect_endpoint_drop_lcb"]
        )
        semantic_details[field] = {**summary, "passed": passed}
    details["semantic_update_disconnect"] = semantic_details
    gates["semantic_update_is_causal_for_acquisition"] = all(
        row["passed"] for row in semantic_details.values()
    )

    side_probe_rows = {}
    for label in ("primitive", "manner"):
        summary = paired_t_summary(
            [
                float(audits[seed]["side_only_probe"][f"{label}_accuracy"])
                - float(audits[seed]["side_only_probe"][f"{label}_chance"])
                for seed in corpora
            ],
            confidence=confidence,
        )
        passed = summary["one_sided_ucb"] < float(
            config["qualification_gates"]["maximum_side_only_concept_accuracy_above_chance"]
        )
        side_probe_rows[label] = {**summary, "passed": passed}
    details["side_only_leakage_probe"] = side_probe_rows
    gates["side_channel_does_not_encode_concept_identity"] = all(
        row["passed"] for row in side_probe_rows.values()
    )

    detector_effects = []
    detector_effects_by_state = {"present": [], "null": []}
    oracle_detector_exact = True
    corrupted_detector_zero = True
    matched_shuffle = True
    for seed in corpora:
        manipulation = audits[seed]["condition_audit"]["detector_manipulation"]
        sync = float(manipulation["synchronized"]["strict_event_or_null_accuracy"])
        disrupted = max(
            float(manipulation[name]["strict_event_or_null_accuracy"])
            for name in ("shuffled", "shift_minus", "shift_plus")
        )
        detector_effects.append(sync - disrupted)
        for state in detector_effects_by_state:
            sync_state = float(
                manipulation["synchronized"]["by_state"][state][
                    "strict_event_or_null_accuracy"
                ]
            )
            disrupted_state = max(
                float(
                    manipulation[name]["by_state"][state][
                        "strict_event_or_null_accuracy"
                    ]
                )
                for name in ("shuffled", "shift_minus", "shift_plus")
            )
            detector_effects_by_state[state].append(sync_state - disrupted_state)
        oracle_detector_exact &= (
            float(manipulation["oracle_alignment"]["strict_event_or_null_accuracy"])
            == 1.0
        )
        corrupted_detector_zero &= (
            float(manipulation["corrupted"]["strict_event_or_null_accuracy"])
            <= float(config["qualification_gates"]["maximum_corrupted_detector_accuracy"])
        )
        matched_shuffle &= bool(
            audits[seed]["condition_audit"][
                "shuffle_preserves_learner_visible_evidence_marginal_by_block"
            ]
        )
    detector_lift = paired_t_summary(detector_effects, confidence=confidence)
    detector_lift_by_state = {
        state: paired_t_summary(values, confidence=confidence)
        for state, values in detector_effects_by_state.items()
    }
    detector_pass = (
        detector_lift["one_sided_lcb"]
        > float(config["qualification_gates"]["minimum_synchronized_detector_lift_lcb"])
        and oracle_detector_exact
        and corrupted_detector_zero
        and matched_shuffle
        and all(
            row["one_sided_lcb"]
            > float(
                config["qualification_gates"][
                    "minimum_synchronized_detector_lift_lcb"
                ]
            )
            for row in detector_lift_by_state.values()
        )
    )
    details["detector_manipulation"] = {
        "synchronized_minus_best_disrupted": detector_lift,
        "synchronized_minus_best_disrupted_by_present_null_state": detector_lift_by_state,
        "oracle_detector_exact": oracle_detector_exact,
        "corrupted_detector_zero": corrupted_detector_zero,
        "shuffled_evidence_marginal_exact_by_block": matched_shuffle,
        "passed": detector_pass,
    }
    gates["detector_manipulations_valid"] = detector_pass

    structural_gate_names = {
        "negative_controls_retain_headroom",
        "oracle_accuracy_and_lift",
        "uninformative_and_disconnected_equivalent_to_absent",
        "corrupted_cannot_help",
        "side_informativeness_interaction",
        "null_head_is_selectively_causal",
        "null_side_path_is_selectively_causal",
        "null_update_path_is_selectively_causal",
        "side_channel_does_not_encode_concept_identity",
        "detector_manipulations_valid",
    }
    causal_attribution_gate_names = {
        "both_signed_shifts_disrupt",
        "exact_window_degrades_and_latent_recovers_at_high_lag",
        "semantic_update_is_causal_for_acquisition",
        "semantic_side_path_is_selectively_causal",
        "permutation_and_old_gate_fail_positive_control",
    }
    required_gate_names = sorted(
        structural_gate_names
        | (
            causal_attribution_gate_names
            if purpose == "construction_qualification"
            else set()
        )
    )
    status = "PASS" if all(gates[name] for name in required_gate_names) else "FAIL"
    causal_attribution_status = (
        "PASS"
        if all(gates[name] for name in causal_attribution_gate_names)
        else "FAIL"
    )
    return {
        "schema_version": "nursery-corrective-informativeness-v1",
        "status": status,
        "purpose": purpose,
        "independent_unit": "corpus_seed",
        "model_replicates_averaged_first": True,
        "publication_state": "UNPUBLISHED_CANDIDATE",
        "development_candidate": purpose == "development",
        "scientific_outcome": False,
        "gates": gates,
        "required_gate_names": required_gate_names,
        "causal_attribution_status": causal_attribution_status,
        "causal_attribution_gate_names": sorted(causal_attribution_gate_names),
        "positive_result_causal_attribution_gate_names": sorted(
            causal_attribution_gate_names
        ),
        "clean_scientific_null_can_remain_informative": True,
        "details": details,
    }


def _bootstrap_lcb(
    values: Sequence[float],
    *,
    seed: int,
    label: str,
    replicates: int,
    confidence: float,
) -> float:
    array = np.asarray(list(map(float, values)), dtype=np.float64)
    digest = hashlib.sha256(f"bootstrap|{seed}|{label}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 1.0 - confidence, method="linear"))


def analyze_primary(
    averaged: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    purpose: str,
) -> dict[str, Any]:
    table = {
        (int(row["corpus_seed"]), str(row["condition"])): row["metrics"]
        for row in averaged["rows"]
    }
    corpus_seeds = list(map(int, config["resolved_registries"][purpose]["corpus"]))
    analysis = config["analysis"]
    confidence = float(analysis["primary_one_sided_confidence_level"])
    practical = float(analysis["minimum_practical_effect"])
    bound = float(analysis["primary_lower_bound_must_exceed"])
    endpoints = {
        "lexical_acquisition_top1": "lexical.exact_chance",
        "heldout_composition_top1": "composition.exact_chance",
    }
    results = {}
    primary_comparators = list(map(str, config["design"]["primary_comparators"]))
    for endpoint, chance_field in endpoints.items():
        differences = []
        synchronized = []
        comparator_values = []
        chances = []
        comparator_choices = []
        for corpus_seed in corpus_seeds:
            sync = float(table[(corpus_seed, "synchronized")][endpoint])
            comparator_rows = {
                condition: float(table[(corpus_seed, condition)][endpoint])
                for condition in primary_comparators
            }
            comparator = max(comparator_rows.values())
            differences.append(sync - comparator)
            synchronized.append(sync)
            comparator_values.append(comparator)
            chances.append(float(table[(corpus_seed, "synchronized")][chance_field]))
            comparator_choices.append(
                next(
                    condition
                    for condition in primary_comparators
                    if comparator_rows[condition] == comparator
                )
            )
        summary = _t_summary(differences, confidence=confidence)
        sync_minus_chance = [
            value - chance for value, chance in zip(synchronized, chances)
        ]
        sync_summary = _t_summary(sync_minus_chance, confidence=confidence)
        positive_fraction = float(np.mean(np.asarray(differences) > 0.0))
        large_negative_fraction = float(
            np.mean(
                np.asarray(differences)
                <= float(analysis["large_negative_threshold"])
            )
        )
        all_identical = len(set(differences)) == 1
        bootstrap_lcb = _bootstrap_lcb(
            differences,
            seed=int(config["resolved_registries"][purpose]["inference"][0]),
            label=endpoint,
            replicates=int(analysis["bootstrap_sensitivity_replicates"]),
            confidence=confidence,
        )
        gates = {
            "mean_at_least_minimum_practical_effect": summary["mean"] >= practical,
            "one_sided_lcb_exceeds_practical_bound": summary["one_sided_lcb"] > bound,
            "synchronized_lcb_above_chance": sync_summary["one_sided_lcb"]
            > float(analysis["synchronized_lcb_above_exact_chance"]),
            "positive_corpus_fraction": positive_fraction
            >= float(analysis["minimum_positive_corpus_fraction"]),
            "large_negative_corpus_fraction": large_negative_fraction
            <= float(analysis["maximum_large_negative_corpus_fraction"]),
            "non_degenerate_effect_vector": (
                not all_identical
                if bool(analysis["all_identical_effects_fail"])
                else True
            ),
        }
        results[endpoint] = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "estimand": "synchronized_minus_per_corpus_max_absent_shuffled",
            "summary": summary,
            "synchronized_minus_exact_chance": sync_summary,
            "bootstrap_sensitivity_lcb": bootstrap_lcb,
            "minimum_practical_effect": practical,
            "required_lcb": bound,
            "positive_corpus_fraction": positive_fraction,
            "large_negative_corpus_fraction": large_negative_fraction,
            "all_effects_identical": all_identical,
            "comparator_choice_counts": dict(Counter(comparator_choices)),
            "comparator_values": comparator_values,
            "synchronized_values": synchronized,
            "gates": gates,
        }
    passed = all(value["status"] == "PASS" for value in results.values())
    return {
        "schema_version": "nursery-corrective-primary-inference-v1",
        "status": "PASS" if passed else "FAIL",
        "intersection_union_rule": True,
        "no_alpha_split_for_conjunctive_alternative": True,
        "independent_unit": "corpus_seed",
        "model_replicates_averaged_first": True,
        "co_primary": results,
        "publication_state": "UNPUBLISHED_CANDIDATE",
        "development_candidate": purpose == "development",
        "scientific_outcome": False,
        "confirmation_outcome": False,
    }


def dependence_diagnostics(
    averaged: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    table = {
        (int(row["corpus_seed"]), str(row["condition"])): row["metrics"]
        for row in averaged["rows"]
    }
    corpora = sorted({key[0] for key in table})
    comparators = list(map(str, config["design"]["primary_comparators"]))
    if comparators != ["absent", "shuffled"]:
        raise ValueError("presence/null diagnostics require frozen absent+shuffled comparators")
    equality_tolerance = float(
        config["qualification_gates"]["presence_null_effect_equality_tolerance"]
    )
    equality_forbidden = bool(
        config["qualification_gates"][
            "exact_present_null_effect_vector_equality_forbidden"
        ]
    )
    by_comparator = {}
    for comparator in comparators:
        present = [
            float(table[(seed, "synchronized")]["presence.present.strict_top1"])
            - float(table[(seed, comparator)]["presence.present.strict_top1"])
            for seed in corpora
        ]
        null = [
            float(table[(seed, "synchronized")]["presence.null.strict_top1"])
            - float(table[(seed, comparator)]["presence.null.strict_top1"])
            for seed in corpora
        ]
        lexical = [
            float(table[(seed, "synchronized")]["lexical_acquisition_top1"])
            - float(table[(seed, comparator)]["lexical_acquisition_top1"])
            for seed in corpora
        ]
        maximum_absolute_difference = float(
            np.max(np.abs(np.asarray(present) - np.asarray(null)))
        )
        equal_within_tolerance = (
            maximum_absolute_difference <= equality_tolerance
        )
        if len(set(present)) > 1 and len(set(null)) > 1:
            pearson = float(np.corrcoef(present, null)[0, 1])
            spearman = float(stats.spearmanr(present, null).statistic)
        else:
            pearson = None
            spearman = None
        by_comparator[comparator] = {
            "present_effects": present,
            "null_effects": null,
            "lexical_effects_secondary_diagnostic": lexical,
            "present_null_equal_within_tolerance": equal_within_tolerance,
            "present_null_maximum_absolute_effect_difference": (
                maximum_absolute_difference
            ),
            "present_null_pearson": pearson,
            "present_null_spearman": spearman,
            "present_null_covariance": float(
                np.cov(present, null, ddof=1)[0, 1]
            ),
            "status": (
                "FAIL"
                if equality_forbidden and equal_within_tolerance
                else "PASS"
            ),
        }
    passed = all(row["status"] == "PASS" for row in by_comparator.values())
    absent = by_comparator["absent"]
    return {
        "status": "PASS" if passed else "FAIL",
        "comparators": comparators,
        "by_comparator": by_comparator,
        "present_effects": absent["present_effects"],
        "null_effects": absent["null_effects"],
        "synchronized_minus_absent_lexical_effects_secondary_diagnostic": (
            absent["lexical_effects_secondary_diagnostic"]
        ),
        "present_null_exact_vector_equality": absent[
            "present_null_equal_within_tolerance"
        ],
        "exact_present_null_effect_vector_equality_forbidden": (
            equality_forbidden
        ),
        "present_null_effect_equality_tolerance": equality_tolerance,
        "present_null_maximum_absolute_effect_difference": absent[
            "present_null_maximum_absolute_effect_difference"
        ],
        "present_null_pearson": absent["present_null_pearson"],
        "present_null_spearman": absent["present_null_spearman"],
        "present_null_covariance": absent["present_null_covariance"],
        "high_correlation_is_diagnostic_not_decisive": True,
        "selective_mutation_required": True,
    }
