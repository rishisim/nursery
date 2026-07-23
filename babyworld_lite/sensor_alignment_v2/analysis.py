from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from .learners import evaluation_firewall_contract
from .protocol import (
    canonical_digest,
    guard_records,
    guard_seed_operation,
    verify_confirmation_manifest,
)


def _select_values(
    records: Sequence[Mapping[str, Any]],
    *,
    learner: str,
    metric: str,
    conditions: Sequence[str],
    analysis_kind: str,
    factor: str | None = None,
    level: Any | None = None,
) -> dict[tuple[int, int], dict[str, float]]:
    values: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in records:
        if row.get("analysis_kind") != analysis_kind:
            continue
        if row.get("learner") != learner or row.get("condition") not in set(conditions):
            continue
        if factor is not None and (
            row.get("factor") != factor or str(row.get("level")) != str(level)
        ):
            continue
        values[(int(row["corpus_seed"]), int(row["model_seed"]))][
            str(row["condition"])
        ] = float(row["metrics"][metric])
    return values


def _exhaustive_sign_flip_pvalue(differences: Sequence[float]) -> float:
    observed = float(np.mean(differences))
    sums = np.asarray([0.0], dtype=np.float64)
    for value in differences:
        sums = np.concatenate([sums - float(value), sums + float(value)])
    tolerance = 1e-12
    return float(np.mean(sums / len(differences) >= observed - tolerance))


def _effect_summary(
    differences: Mapping[int, float], *, exact_randomization: bool
) -> dict[str, Any]:
    seeds = sorted(differences)
    values = np.asarray([differences[seed] for seed in seeds], dtype=np.float64)
    n = len(values)
    if n < 2:
        raise ValueError("at least two corpus effects are required")
    point = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(n))
    if standard_error == 0.0:
        t_statistic = float("inf") if point > 0 else (float("-inf") if point < 0 else 0.0)
        one_sided_p = 0.0 if point > 0 else (1.0 if point < 0 else 0.5)
        ci_low = ci_high = point
        lower_one_sided = point
    else:
        t_statistic = point / standard_error
        one_sided_p = float(stats.t.sf(t_statistic, df=n - 1))
        critical = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = point - critical * standard_error
        ci_high = point + critical * standard_error
        lower_one_sided = point - float(stats.t.ppf(0.95, df=n - 1)) * standard_error
    positive = int(np.sum(values > 0))
    negative = int(np.sum(values < 0))
    nonzero = positive + negative
    sign_p = (
        1.0
        if nonzero == 0
        else float(stats.binomtest(positive, nonzero, 0.5, alternative="greater").pvalue)
    )
    return {
        "point_estimate": point,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "degrees_of_freedom": n - 1,
        "one_sided_t_pvalue": one_sided_p,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "one_sided_95_lower_bound": float(lower_one_sided),
        "positive_corpus_count": positive,
        "negative_corpus_count": negative,
        "zero_corpus_count": n - nonzero,
        "exact_sign_test_one_sided_pvalue": sign_p,
        "exhaustive_sign_flip_one_sided_pvalue": (
            _exhaustive_sign_flip_pvalue(values) if exact_randomization else None
        ),
        "n_independent_corpus_seeds": n,
        "independent_unit": "corpus_seed",
        "unit_effects": [
            {"corpus_seed": seed, "difference": float(differences[seed])}
            for seed in seeds
        ],
        "inference_method": (
            "Student t inference over one model-seed-averaged effect per corpus; "
            "exact sign and exhaustive sign-flip evidence are corroborating"
        ),
    }


def corpus_condition_contrast(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    learner: str,
    metric: str,
    left: str,
    right: str,
    analysis_kind: str = "main",
    factor: str | None = None,
    level: Any | None = None,
    exact_randomization: bool = False,
) -> dict[str, Any]:
    guard_records(records, config, operation="summarize")
    values = _select_values(
        records,
        learner=learner,
        metric=metric,
        conditions=[left, right],
        analysis_kind=analysis_kind,
        factor=factor,
        level=level,
    )
    corpora = sorted(map(int, config["seeds"]["development"]["corpus"]))
    models = sorted(map(int, config["seeds"]["development"]["model"]))
    differences: dict[int, float] = {}
    for corpus in corpora:
        left_values: list[float] = []
        right_values: list[float] = []
        for model in models:
            arms = values.get((corpus, model), {})
            if set(arms) != {left, right}:
                raise ValueError(
                    f"incomplete paired condition records for corpus={corpus}, model={model}, "
                    f"factor={factor}, level={level}: {arms.keys()}"
                )
            left_values.append(arms[left])
            right_values.append(arms[right])
        differences[corpus] = float(np.mean(left_values) - np.mean(right_values))
    result = _effect_summary(differences, exact_randomization=exact_randomization)
    result.update(
        {
            "estimand": f"{metric}({left}) - {metric}({right})",
            "learner": learner,
            "analysis_kind": analysis_kind,
            "factor": factor,
            "level": level,
            "left_condition": left,
            "right_condition": right,
            "model_seed_handling": (
                f"average {len(models)} algorithmic replicates within each corpus before subtraction"
            ),
        }
    )
    return result


def corpus_learner_contrast(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    metric: str,
    condition: str,
    left_learner: str,
    right_learner: str,
) -> dict[str, Any]:
    guard_records(records, config, operation="summarize")
    corpora = sorted(map(int, config["seeds"]["development"]["corpus"]))
    models = sorted(map(int, config["seeds"]["development"]["model"]))
    lookup = {
        (int(row["corpus_seed"]), int(row["model_seed"]), str(row["learner"])): float(
            row["metrics"][metric]
        )
        for row in records
        if row.get("analysis_kind") == "main" and row.get("condition") == condition
    }
    differences: dict[int, float] = {}
    for corpus in corpora:
        differences[corpus] = float(
            np.mean([lookup[(corpus, model, left_learner)] for model in models])
            - np.mean([lookup[(corpus, model, right_learner)] for model in models])
        )
    result = _effect_summary(differences, exact_randomization=False)
    result.update(
        {
            "estimand": f"{metric}({left_learner}) - {metric}({right_learner})",
            "condition": condition,
            "left_learner": left_learner,
            "right_learner": right_learner,
            "model_seed_handling": "algorithmic replicates averaged within corpus",
        }
    )
    return result


def condition_means(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    learner: str,
    metric: str,
) -> dict[str, float]:
    guard_records(records, config, operation="summarize")
    values: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row.get("analysis_kind") == "main" and row.get("learner") == learner:
            values[str(row["condition"])].append(float(row["metrics"][metric]))
    return {condition: float(np.mean(rows)) for condition, rows in sorted(values.items())}


def pairing_fairness_audit(
    records: Sequence[Mapping[str, Any]],
    generation_audits: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("analysis_kind") == "main":
            grouped[(str(row["learner"]), int(row["corpus_seed"]), int(row["model_seed"]))].append(row)
    checks_by_group: dict[str, dict[str, bool]] = {}
    expected_conditions = set(config["conditions"]["names"])
    for key, rows in sorted(grouped.items()):
        traces = [row["training_trace"] for row in rows]
        checks_by_group[f"{key[0]}|corpus={key[1]}|model={key[2]}"] = {
            "all_conditions_present": {str(row["condition"]) for row in rows}
            == expected_conditions,
            "matched_initialization": len(
                {trace["initialization_digest"] for trace in traces}
            )
            == 1,
            "matched_episode_order": len(
                {trace["episode_order_digest"] for trace in traces}
            )
            == 1,
            "matched_non_channel_inventory": len(
                {trace["non_channel_inventory_digest"] for trace in traces}
            )
            == 1,
            "matched_examples": len({trace["training_examples"] for trace in traces}) == 1,
            "matched_update_passes": len({trace["update_passes"] for trace in traces}) == 1,
        }
    corpus_checks = {
        str(index): {
            "generation_valid": bool(audit["valid"]),
            "condition_inventory_matched": bool(
                audit["condition_checks"]["matched_non_sensor_inventory"]
            ),
            "raw_rows_distribution_matched": bool(
                audit["condition_checks"]["present_raw_row_multisets_match"]
            ),
            "shuffled_group_safe": bool(audit["donor_checks"]["different_lexical_group"]),
            "shuffled_bijection": bool(audit["donor_checks"]["bijection"]),
        }
        for index, audit in enumerate(generation_audits)
    }
    valid = all(all(value.values()) for value in checks_by_group.values()) and all(
        all(value.values()) for value in corpus_checks.values()
    )
    return {
        "schema_version": "synthetic-sensor-event-pairing-audit-v2",
        "valid": valid,
        "paired_training_checks": checks_by_group,
        "corpus_generation_checks": corpus_checks,
        "compute_definition": (
            "Within learner/corpus/model blocks all conditions use the same non-channel inventory, "
            "order, initialization, example count, and update passes."
        ),
    }


def stochastic_replicate_audit(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    primary = str(config["learners"]["primary"])
    selected = [
        row
        for row in records
        if row.get("analysis_kind") == "main"
        and row.get("learner") == primary
        and row.get("condition") == "synchronized"
    ]
    by_corpus: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_corpus[int(row["corpus_seed"])].append(row)
    checks = {
        "two_model_seeds_configured": len(config["seeds"]["development"]["model"]) >= 2,
        "initializations_vary_within_every_corpus": all(
            len({row["training_trace"]["initialization_digest"] for row in rows}) > 1
            for rows in by_corpus.values()
        ),
        "serialized_models_vary_within_every_corpus": all(
            len({row["model_digest"] for row in rows}) > 1 for rows in by_corpus.values()
        ),
        "dropout_traces_show_algorithmic_sampling": any(
            len({row["training_trace"]["active_evidence_updates"] for row in rows}) > 1
            for rows in by_corpus.values()
        ),
        "inference_averages_models_within_corpus": config["inference"]["model_seed_role"]
        == "algorithmic_replicate_averaged_within_corpus",
    }
    return {
        "schema_version": "synthetic-sensor-event-model-replicate-audit-v2",
        "valid": all(checks.values()),
        "checks": checks,
        "corpus_count": len(by_corpus),
        "model_seeds": list(config["seeds"]["development"]["model"]),
        "independent_empirical_units": len(by_corpus),
    }


def leakage_audit(
    records: Sequence[Mapping[str, Any]],
    lexicons: Sequence[Mapping[str, Any]],
    generation_audits: Sequence[Mapping[str, Any]],
    calibration_provenance: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary = str(config["learners"]["primary"])
    zero_values = [
        float(row["metrics"]["zero_exposure_word_6way_accuracy"])
        for row in records
        if row.get("analysis_kind") == "main"
        and row.get("learner") == primary
        and row.get("condition") == "synchronized"
    ]
    mappings = [
        canonical_digest(
            {
                "primitive": lexicon["primitive_word_to_index"],
                "manner": lexicon["manner_word_to_index"],
                "noun": lexicon["noun_word_to_object"],
            }
        )
        for lexicon in lexicons
    ]
    chance = 1.0 / 6.0
    zero_mean = float(np.mean(zero_values))
    checks = {
        "randomized_mapping_changes_across_corpora": len(set(mappings)) >= 15,
        "raw_stream_semantic_key_audits_pass": all(
            audit["checks"]["raw_stream_contains_no_semantic_or_oracle_keys"]
            for audit in generation_audits
        ),
        "oracle_records_physically_separate": all(
            audit["checks"]["oracle_physically_separate"] for audit in generation_audits
        ),
        "calibration_has_no_lexical_or_referent_targets": all(
            provenance["lexical_targets_present"] is False
            and provenance["referent_targets_present"] is False
            and provenance["randomized_word_mappings_present"] is False
            for provenance in calibration_provenance
        ),
        "heldout_compositions_disjoint": all(
            audit["checks"]["heldout_object_action_compositions_disjoint"]
            for audit in generation_audits
        ),
        "structured_components_exposed_but_combination_held_out": all(
            audit["checks"]["structured_combination_absent_but_components_exposed"]
            for audit in generation_audits
        ),
        "zero_exposure_panel_at_chance": zero_mean
        <= chance + float(config["gates"]["transfer"]["maximum_zero_exposure_above_chance"]),
    }
    return {
        "schema_version": "synthetic-sensor-event-leakage-audit-v2",
        "valid": all(checks.values()),
        "checks": checks,
        "zero_exposure_accuracy": zero_mean,
        "chance": chance,
        "unique_mapping_count": len(set(mappings)),
    }


def learnability_audit(
    records: Sequence[Mapping[str, Any]],
    evaluation_items_by_corpus: Sequence[Sequence[Any]],
    evaluation_oracle_by_corpus: Sequence[Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    metric = str(config["evaluation"]["primary_metric"])
    oracle_values = [
        float(row["metrics"][metric])
        for row in records
        if row.get("analysis_kind") == "main"
        and row.get("learner") == "oracle_event_alignment_upper"
        and row.get("condition") == "absent"
    ]
    pointer_values = [
        float(row["metrics"][metric])
        for row in records
        if row.get("analysis_kind") == "main"
        and row.get("learner") == "v1_pointer_style_upper"
        and row.get("condition") == "synchronized"
    ]
    action_capacity: list[float] = []
    object_capacity: list[float] = []
    for items, oracle_rows in zip(evaluation_items_by_corpus, evaluation_oracle_by_corpus):
        oracle_by_id = {str(row["evaluation_id"]): row for row in oracle_rows}
        for item in items:
            truth = oracle_by_id[str(item.evaluation_id)]
            action = np.asarray(item.action_observation, dtype=float)
            primitive = int(np.argmax(action[:3]))
            manner = int(np.argmax(action[3:]))
            predicted_action = primitive * 2 + manner
            action_capacity.append(
                float(predicted_action == int(truth["true_action_index"]))
            )
            object_capacity.append(
                float(
                    int(np.argmax(item.object_observation))
                    == int(truth["true_object_index"])
                )
            )
    results = {
        "oracle_alignment_accuracy": float(np.mean(oracle_values)),
        "v1_pointer_upper_accuracy": float(np.mean(pointer_values)),
        "direct_action_observation_capacity": float(np.mean(action_capacity)),
        "direct_object_observation_capacity": float(np.mean(object_capacity)),
    }
    thresholds = config["gates"]["controls"]
    checks = {
        "oracle_alignment_positive_control": results["oracle_alignment_accuracy"]
        >= float(thresholds["minimum_oracle_alignment_accuracy"]),
        "v1_pointer_upper_positive_control": results["v1_pointer_upper_accuracy"]
        >= float(thresholds["minimum_v1_pointer_upper_accuracy"]),
        "direct_action_capacity": results["direct_action_observation_capacity"]
        >= float(thresholds["minimum_direct_action_observation_capacity"]),
        "direct_object_capacity": results["direct_object_observation_capacity"]
        >= float(thresholds["minimum_direct_object_observation_capacity"]),
    }
    return {
        "schema_version": "synthetic-sensor-event-learnability-audit-v2",
        "valid": all(checks.values()),
        "checks": checks,
        "results": results,
    }


def reserve_guard_audit(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_checks = verify_confirmation_manifest(manifest, config)
    reserve = config["seeds"]["confirmation_reserve"]
    values = [
        int(reserve["corpus"][0]),
        int(reserve["model"][0]),
        int(reserve["calibration"][0]),
    ]
    blocked: dict[str, bool] = {}
    for operation in ("generate", "calibrate", "train", "evaluate", "read", "summarize"):
        try:
            guard_seed_operation(config, operation=operation, seeds=values)
        except PermissionError:
            blocked[operation] = True
        else:
            blocked[operation] = False
    return {
        "schema_version": "synthetic-sensor-event-reserve-audit-v2",
        "valid": all(manifest_checks.values()) and all(blocked.values()),
        "manifest_checks": manifest_checks,
        "blocked_operations": blocked,
        "authorization_present": False,
        "reserved_outcomes_accessed": 0,
    }


def summarize_factor_sensitivity(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    learner = str(config["learners"]["primary"])
    metric = str(config["evaluation"]["primary_metric"])
    output: dict[str, Any] = {}
    for factor, levels in config["factors"].items():
        output[factor] = {}
        for level in levels:
            output[factor][str(level)] = {
                "synchronized_minus_absent": corpus_condition_contrast(
                    records,
                    config,
                    learner=learner,
                    metric=metric,
                    left="synchronized",
                    right="absent",
                    analysis_kind="sensitivity",
                    factor=factor,
                    level=level,
                ),
                "synchronized_minus_shuffled": corpus_condition_contrast(
                    records,
                    config,
                    learner=learner,
                    metric=metric,
                    left="synchronized",
                    right="shuffled",
                    analysis_kind="sensitivity",
                    factor=factor,
                    level=level,
                ),
            }
    return {
        "schema_version": "synthetic-sensor-event-factor-sensitivity-v2",
        "status": "secondary_descriptive_refit_within_each_level",
        "independent_unit": "corpus_seed_after_model_seed_averaging",
        "factors": output,
    }


def summarize_results(
    main_records: Sequence[Mapping[str, Any]],
    sensitivity_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = str(config["learners"]["primary"])
    metric = str(config["evaluation"]["primary_metric"])
    co_primary = {
        control: corpus_condition_contrast(
            main_records,
            config,
            learner=primary,
            metric=metric,
            left="synchronized",
            right=control,
            exact_randomization=True,
        )
        for control in ("absent", "shuffled")
    }
    means = condition_means(
        main_records, config, learner=primary, metric=metric
    )
    noun_contrasts = {
        control: corpus_condition_contrast(
            main_records,
            config,
            learner=primary,
            metric="matched_noun_6way_macro_accuracy",
            left="synchronized",
            right=control,
        )
        for control in ("absent", "shuffled")
    }
    ablations = {
        learner: corpus_learner_contrast(
            main_records,
            config,
            metric=metric,
            condition="synchronized",
            left_learner=primary,
            right_learner=learner,
        )
        for learner in (
            "exact_window_symbolic",
            "sensor_latent_single_occurrence",
            "cross_occurrence_no_sensor",
            "sensor_latent_cross_no_null",
            "structural_absence",
        )
    }
    endpoint_means = {
        endpoint: condition_means(
            main_records, config, learner=primary, metric=endpoint
        )
        for endpoint in config["evaluation"]["endpoints"]
    }
    learner_means = {
        learner: condition_means(
            main_records, config, learner=learner, metric=metric
        )
        for learner in config["learners"]["names"]
    }
    shifts = {
        condition: {
            "offset_samples": int(config["conditions"]["time_shift_offsets"][condition]),
            "accuracy": means[condition],
            "shifted_minus_absent": corpus_condition_contrast(
                main_records,
                config,
                learner=primary,
                metric=metric,
                left=condition,
                right="absent",
            ),
        }
        for condition in config["conditions"]["time_shift_offsets"]
    }
    factor = summarize_factor_sensitivity(sensitivity_records, config)
    aggregate = {
        "schema_version": "synthetic-sensor-event-aggregate-v2",
        "protocol_id": config["protocol"]["id"],
        "co_primary_estimands": co_primary,
        "primary_condition_means": means,
        "time_shift_characterization": shifts,
        "matched_noun_contrasts": noun_contrasts,
        "synchronized_ablation_contrasts": ablations,
        "primary_learner_endpoint_condition_means": endpoint_means,
        "learner_condition_means": learner_means,
        "multiplicity": {
            "rule": "intersection-union: both co-primary one-sided tests must pass at alpha .05",
            "alpha_splitting": False,
            "secondary_status": "descriptive gates, not additional discoveries",
        },
        "independent_unit": "20 corpus seeds",
        "model_seed_role": "two stochastic algorithmic replicates averaged within corpus",
        "primary_bootstrap_used": False,
    }
    return aggregate, factor


def terminal_decision(
    aggregate: Mapping[str, Any],
    factor: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    co_primary = aggregate["co_primary_estimands"]
    means = aggregate["primary_condition_means"]
    alpha = float(config["inference"]["primary_alpha_one_sided"])
    co_thresholds = config["gates"]["co_primary"]
    co_checks: dict[str, dict[str, bool]] = {}
    for control, result in co_primary.items():
        co_checks[control] = {
            "minimum_lift": result["point_estimate"]
            >= float(co_thresholds["minimum_absolute_lift"]),
            "one_sided_t_test": result["one_sided_t_pvalue"] < alpha,
            "two_sided_ci_above_zero": result["ci95_low"] > 0.0,
            "substantial_positive_majority": result["positive_corpus_count"]
            >= int(co_thresholds["minimum_positive_corpus_count"]),
        }
    noun = aggregate["matched_noun_contrasts"]
    zero_level = factor["factors"]["sensor_informativeness"]["0.0"]
    informative_levels = [
        value
        for level, value in factor["factors"]["sensor_informativeness"].items()
        if float(level) > 0
    ]
    ablations = aggregate["synchronized_ablation_contrasts"]
    endpoints = aggregate["primary_learner_endpoint_condition_means"]
    transfer = config["gates"]["transfer"]
    robustness_tolerance = float(
        config["gates"]["robustness"]["maximum_corrupted_drop_below_absence"]
    )
    selectivity_tolerance = float(
        config["gates"]["selectivity"]["maximum_absolute_matched_noun_lift"]
    )
    zero_tolerance = float(
        config["gates"]["selectivity"]["maximum_absolute_zero_information_lift"]
    )
    ablation_thresholds = config["gates"]["ablations"]
    all_audits = all(bool(value.get("valid")) for value in audits.values())
    go_checks = {
        "all_co_primary_components_pass": all(
            all(check.values()) for check in co_checks.values()
        ),
        "all_integrity_and_positive_control_audits_pass": all_audits,
        "corrupted_conditions_within_absence_tolerance": all(
            means[condition] >= means["absent"] - robustness_tolerance
            for condition in ("shuffled", "uninformative")
        ),
        "zero_information_has_no_synchronized_advantage": all(
            abs(zero_level[name]["point_estimate"]) <= zero_tolerance
            for name in ("synchronized_minus_absent", "synchronized_minus_shuffled")
        ),
        "informative_sensor_strata_help": all(
            value[name]["point_estimate"] > 0.0
            for value in informative_levels
            for name in ("synchronized_minus_absent", "synchronized_minus_shuffled")
        ),
        "matched_noun_lift_near_zero_empirically": all(
            abs(value["point_estimate"]) <= selectivity_tolerance
            for value in noun.values()
        ),
        "primary_accuracy_threshold": endpoints[
            "heldout_object_action_composition_action_6way_macro_accuracy"
        ]["synchronized"]
        >= float(transfer["minimum_synchronized_primary_accuracy"]),
        "seen_combination_threshold": endpoints[
            "seen_combination_action_6way_accuracy"
        ]["synchronized"]
        >= float(transfer["minimum_seen_combination_accuracy"]),
        "structured_concept_threshold": endpoints[
            "structured_heldout_concept_action_6way_accuracy"
        ]["synchronized"]
        >= float(transfer["minimum_structured_concept_accuracy"]),
        "seen_component_threshold": endpoints[
            "seen_lexical_component_accuracy_on_independent_tokens"
        ]["synchronized"]
        >= float(transfer["minimum_seen_component_accuracy"]),
        "beats_exact_window": ablations["exact_window_symbolic"]["point_estimate"]
        >= float(ablation_thresholds["minimum_over_exact_window"]),
        "beats_single_occurrence": ablations["sensor_latent_single_occurrence"][
            "point_estimate"
        ]
        >= float(ablation_thresholds["minimum_over_single_occurrence"]),
        "beats_no_sensor_cross_occurrence": ablations["cross_occurrence_no_sensor"][
            "point_estimate"
        ]
        >= float(ablation_thresholds["minimum_over_cross_occurrence_no_sensor"]),
        "beats_no_null": ablations["sensor_latent_cross_no_null"]["point_estimate"]
        >= float(ablation_thresholds["minimum_over_no_null"]),
    }
    critical_audits = (
        "protocol_freeze",
        "detector_validation",
        "pairing_fairness",
        "leakage",
        "learnability",
        "evaluation_firewall",
        "confirmation_reserve_guard",
    )
    stop_checks = {
        "critical_integrity_detector_or_learnability_failure": any(
            not bool(audits[name].get("valid")) for name in critical_audits
        ),
        "both_co_primary_intervals_entirely_below_zero": all(
            result["ci95_high"] < 0.0 for result in co_primary.values()
        ),
    }
    if all(go_checks.values()):
        recommendation = "GO"
        reason = "all frozen co-primary, robustness, selectivity, transfer, ablation, and audit gates passed"
    elif any(stop_checks.values()):
        recommendation = "STOP"
        reason = "a frozen critical validity/positive-control gate failed or both co-primary intervals showed clear harm"
    else:
        recommendation = "REVISE"
        reason = "the study remained interpretable, but at least one frozen GO gate did not pass"
    return {
        "schema_version": "synthetic-sensor-event-terminal-decision-v2",
        "protocol_id": config["protocol"]["id"],
        "study_phase": "development_only",
        "recommendation_for_later_v2_confirmation": recommendation,
        "reason": reason,
        "co_primary_component_checks": co_checks,
        "go_checks": go_checks,
        "stop_checks": stop_checks,
        "co_primary_effects": co_primary,
        "confirmation_authorized": False,
        "v1_confirmation_authorized": False,
        "confirmation_reserve_status": "untouched_and_guarded",
        "claim_boundary": (
            "Synthetic raw-sensor-assisted action lexical grounding only; no infant, ecological, "
            "raw-pixel/audio, arbitrary unseen-word, or timestamp-agreement claim."
        ),
    }
