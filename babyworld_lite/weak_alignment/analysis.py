from __future__ import annotations

from collections import Counter, defaultdict
import inspect
from typing import Any, Mapping, Sequence

import numpy as np

from babyworld_lite.weak_alignment.learners import modality_withholding_contract
from babyworld_lite.weak_alignment.protocol import (
    canonical_digest,
    guard_records_for_read_or_summary,
    guard_seed_operation,
    verify_confirmation_manifest,
)
from babyworld_lite.weak_alignment.synthetic import ACTIONS, OBJECTS, SyntheticCorpus


def paired_hierarchical_bootstrap(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    learner: str,
    metric: str,
    left_condition: str,
    right_condition: str,
    analysis_kind: str = "main",
    stratum_factor: str | None = None,
    stratum_level: Any | None = None,
) -> dict[str, Any]:
    guard_records_for_read_or_summary(records, config, operation="summarize")
    selected = [
        row for row in records
        if row.get("analysis_kind") == analysis_kind
        and row.get("learner") == learner
        and row.get("condition") in {left_condition, right_condition}
        and (stratum_factor is None or row.get("stratum_factor") == stratum_factor)
        and (stratum_factor is None or str(row.get("stratum_level")) == str(stratum_level))
    ]
    values: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in selected:
        key = (int(row["corpus_seed"]), int(row["model_seed"]))
        values[key][str(row["condition"])] = float(row["metrics"][metric])
    incomplete = [key for key, arms in values.items() if set(arms) != {left_condition, right_condition}]
    if incomplete:
        raise ValueError(f"incomplete paired records for bootstrap: {incomplete}")
    corpus_seeds = sorted({key[0] for key in values})
    expected_corpus = sorted(map(int, config["seeds"]["development"]["corpus"]))
    expected_models = sorted(map(int, config["seeds"]["development"]["model"]))
    if corpus_seeds != expected_corpus:
        raise ValueError("bootstrap corpus units do not match the frozen development seeds")
    if any(sorted(model for corpus, model in values if corpus == seed) != expected_models for seed in corpus_seeds):
        raise ValueError("bootstrap model units do not match the frozen crossed model seeds")
    differences = {
        key: arms[left_condition] - arms[right_condition]
        for key, arms in values.items()
    }
    point = float(np.mean(list(differences.values())))
    bootstrap_samples = int(config["inference"]["bootstrap_samples"])
    seed_namespace = canonical_digest({
        "learner": learner,
        "metric": metric,
        "left": left_condition,
        "right": right_condition,
        "kind": analysis_kind,
        "factor": stratum_factor,
        "level": stratum_level,
    })
    seed = int(config["inference"]["bootstrap_seed"]) ^ int(seed_namespace[:8], 16)
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled_corpora = rng.choice(corpus_seeds, size=len(corpus_seeds), replace=True)
        sampled_values: list[float] = []
        for corpus_seed in sampled_corpora:
            sampled_models = rng.choice(expected_models, size=len(expected_models), replace=True)
            sampled_values.extend(
                differences[(int(corpus_seed), int(model_seed))]
                for model_seed in sampled_models
            )
        samples[index] = np.mean(sampled_values)
    low, high = np.quantile(samples, [0.025, 0.975])
    unit_rows = [
        {
            "corpus_seed": corpus,
            "model_seed": model,
            "difference": difference,
        }
        for (corpus, model), difference in sorted(differences.items())
    ]
    return {
        "estimand": f"{metric}({left_condition}) - {metric}({right_condition})",
        "learner": learner,
        "analysis_kind": analysis_kind,
        "stratum_factor": stratum_factor,
        "stratum_level": stratum_level,
        "point_estimate": point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_probability_gt_zero": float(np.mean(samples > 0)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_method": (
            "paired hierarchical bootstrap: resample corpus seeds, then crossed model seeds "
            "within sampled corpus; no episode/window resampling"
        ),
        "independent_unit": "synthetic corpus seed crossed with model initialization seed",
        "n_corpus_seeds": len(corpus_seeds),
        "n_model_seeds_per_corpus": len(expected_models),
        "n_paired_units": len(differences),
        "positive_pair_count": int(sum(value > 0 for value in differences.values())),
        "zero_pair_count": int(sum(value == 0 for value in differences.values())),
        "unit_differences": unit_rows,
    }


def condition_means(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *, learner: str, metric: str
) -> dict[str, float]:
    guard_records_for_read_or_summary(records, config, operation="summarize")
    values: dict[str, list[float]] = defaultdict(list)
    for row in records:
        if row.get("analysis_kind") == "main" and row.get("learner") == learner:
            values[str(row["condition"])].append(float(row["metrics"][metric]))
    return {condition: float(np.mean(rows)) for condition, rows in sorted(values.items())}


def manipulation_checks(corpora: Sequence[SyntheticCorpus], config: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    chances: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_level: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_level_chance: dict[tuple[str, str], list[float]] = defaultdict(list)
    boundary_values: dict[str, list[float]] = defaultdict(list)
    for corpus in corpora:
        oracle = {str(row["episode_id"]): row for row in corpus.oracle_episodes}
        for condition, rows in corpus.condition_views.items():
            if condition == "absent":
                continue
            for row in rows:
                truth = oracle[str(row["episode_id"])]
                candidate_count = len(row["events"])
                target = truth["target_event_index"]
                answer = candidate_count if target is None else int(target)
                predicted = int(np.asarray(row["side_scores"]).argmax())
                information = float(truth["factors"]["side_informativeness"])
                category = "configured_uninformative" if information == 0.0 else "configured_informative"
                values[(condition, category)].append(float(predicted == answer))
                chances[(condition, category)].append(1.0 / (candidate_count + 1))
                level = f"{information:g}"
                per_level[(condition, level)].append(float(predicted == answer))
                per_level_chance[(condition, level)].append(1.0 / (candidate_count + 1))
                boundary_values[condition].append(float((predicted == candidate_count) == (target is None)))
    summary: dict[str, Any] = {}
    for condition in sorted({key[0] for key in values}):
        summary[condition] = {}
        for category in ("configured_uninformative", "configured_informative"):
            rows = values[(condition, category)]
            chance_rows = chances[(condition, category)]
            summary[condition][category] = {
                "referent_or_null_top1_accuracy": float(np.mean(rows)),
                "mean_chance": float(np.mean(chance_rows)),
                "accuracy_minus_chance": float(np.mean(rows) - np.mean(chance_rows)),
                "n_episodes": len(rows),
            }
        summary[condition]["event_boundary_or_null_binary_accuracy"] = float(
            np.mean(boundary_values[condition])
        )
        summary[condition]["by_informativeness_level"] = {
            level: {
                "referent_or_null_top1_accuracy": float(np.mean(per_level[(condition, level)])),
                "mean_chance": float(np.mean(per_level_chance[(condition, level)])),
                "n_episodes": len(per_level[(condition, level)]),
            }
            for level in sorted({key[1] for key in per_level if key[0] == condition}, key=float)
        }
    sync_info = summary["synchronized"]["configured_informative"]
    sync_zero = summary["synchronized"]["configured_uninformative"]
    control_info = [
        summary[condition]["configured_informative"]["referent_or_null_top1_accuracy"]
        for condition in ("shuffled", "time_shifted", "uninformative")
    ]
    thresholds = config["manipulation_gates"]
    checks = {
        "synchronized_informative_above_chance": (
            sync_info["accuracy_minus_chance"]
            >= float(thresholds["minimum_informative_accuracy_minus_chance"])
        ),
        "synchronized_uninformative_near_chance": (
            abs(sync_zero["accuracy_minus_chance"])
            <= float(thresholds["maximum_uninformative_absolute_accuracy_minus_chance"])
        ),
        "synchronized_informative_beats_each_matched_control": all(
            sync_info["referent_or_null_top1_accuracy"] - value
            >= float(thresholds["minimum_informative_control_gap"])
            for value in control_info
        ),
        "absent_is_structural": all(
            all("side_scores" not in row for row in corpus.condition_views["absent"])
            for corpus in corpora
        ),
        "present_conditions_distribution_matched": all(
            len({
                value for value in corpus.audits["side_distribution_hashes"].values()
                if value is not None
            }) == 1
            for corpus in corpora
        ),
    }
    return {"valid": all(checks.values()), "checks": checks, "results": summary}


def pairing_fairness_checks(
    records: Sequence[Mapping[str, Any]], corpora: Sequence[SyntheticCorpus], config: Mapping[str, Any]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("analysis_kind") == "main":
            grouped[(str(row["learner"]), int(row["corpus_seed"]), int(row["model_seed"]))].append(row)
    checks_by_group: dict[str, dict[str, bool]] = {}
    for key, rows in sorted(grouped.items()):
        name = f"{key[0]}|corpus={key[1]}|model={key[2]}"
        traces = [row["training_trace"] for row in rows]
        checks_by_group[name] = {
            "all_five_conditions": {str(row["condition"]) for row in rows}
            == set(config["conditions"]["training_side_modality"]),
            "matched_initialization": len({trace["initialization_digest"] for trace in traces}) == 1,
            "matched_data_order": len({trace["data_order_digest"] for trace in traces}) == 1,
            "matched_inventory": len({trace["inventory_digest"] for trace in traces}) == 1,
            "matched_optimizer_steps": len({trace["optimizer_steps"] for trace in traces}) == 1,
            "matched_training_examples": len({trace["training_examples"] for trace in traces}) == 1,
        }
    corpus_checks = {
        str(corpus.corpus_seed): {
            "generation_audit_valid": bool(corpus.audits["valid"]),
            "donor_bijection": bool(corpus.audits["donor_checks"]["bijection"]),
            "donor_no_self": bool(corpus.audits["donor_checks"]["no_self"]),
            "donor_different_group": bool(corpus.audits["donor_checks"]["different_episode_group"]),
            "donor_candidate_count_matched": bool(corpus.audits["donor_checks"]["candidate_count_matched"]),
            "condition_inventory_identity": len(
                set(corpus.audits["condition_inventory_hashes"].values())
            ) == 1,
        }
        for corpus in corpora
    }
    valid = all(all(checks.values()) for checks in checks_by_group.values()) and all(
        all(checks.values()) for checks in corpus_checks.values()
    )
    return {
        "valid": valid,
        "paired_training_checks": checks_by_group,
        "corpus_and_donor_checks": corpus_checks,
        "compute_definition": (
            "Within each learner/corpus/model block, conditions receive identical episode order, "
            "initial lexical state, update passes, and example count. Structural absence omits the "
            "auxiliary tensor but does not alter update-pass count."
        ),
    }


def leakage_shortcut_checks(
    records: Sequence[Mapping[str, Any]], corpora: Sequence[SyntheticCorpus], config: Mapping[str, Any]
) -> dict[str, Any]:
    mapping_by_corpus = {
        corpus.corpus_seed: dict(corpus.lexicon_oracle["action_word_to_action"])
        for corpus in corpora
    }
    corpus_seeds = sorted(mapping_by_corpus)
    surface_predictions: list[float] = []
    for heldout in corpus_seeds:
        other = [seed for seed in corpus_seeds if seed != heldout]
        if not other:
            continue
        for word, truth in mapping_by_corpus[heldout].items():
            counts = Counter(mapping_by_corpus[seed][word] for seed in other)
            maximum = max(counts.values())
            prediction = sorted(action for action, count in counts.items() if count == maximum)[0]
            surface_predictions.append(float(prediction == truth))
    zero_values = [
        float(row["metrics"]["zero_exposure_action_6way_macro_accuracy"])
        for row in records
        if row.get("analysis_kind") == "main"
        and row.get("learner") == config["evaluation"]["primary_learner"]
    ]
    chance = 1.0 / len(ACTIONS)
    surface_accuracy = (
        chance if not surface_predictions else float(np.mean(surface_predictions))
    )
    zero_accuracy = float(np.mean(zero_values))
    checks = {
        "all_model_visible_allowlists_pass": all(
            corpus.audits["checks"]["visible_allowlist"] for corpus in corpora
        ),
        "oracle_separate": all(
            corpus.audits["checks"]["oracle_physically_separate_in_memory"] for corpus in corpora
        ),
        "surface_mapping_changes_across_corpora": len({
            canonical_digest(corpus.lexicon_oracle["action_word_to_action"])
            for corpus in corpora
        }) == len(corpora),
        "leave_one_corpus_surface_shortcut_near_chance": (
            not surface_predictions
            or surface_accuracy
            <= float(config["shortcut_gates"]["maximum_surface_shortcut_accuracy"])
        ),
        "zero_exposure_types_at_chance": zero_accuracy
        <= chance + float(config["shortcut_gates"]["zero_exposure_chance_tolerance"]),
        "train_eval_compositions_disjoint": all(
            corpus.audits["checks"]["heldout_compositions_disjoint"] for corpus in corpora
        ),
        "train_eval_instances_independent": all(
            corpus.audits["checks"]["independent_evaluation_hashes"] for corpus in corpora
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "surface_leave_one_corpus_out_accuracy": surface_accuracy,
        "surface_leave_one_corpus_out_supported": bool(surface_predictions),
        "zero_exposure_action_accuracy": zero_accuracy,
        "chance": chance,
        "claim_boundary": (
            "Canonical action dimensions are controlled perceptual features; the learner receives "
            "neither their semantic names nor the randomized word-to-action mapping."
        ),
    }


def learnability_checks(
    records: Sequence[Mapping[str, Any]], corpora: Sequence[SyntheticCorpus], config: Mapping[str, Any]
) -> dict[str, Any]:
    metric = str(config["evaluation"]["primary_metric"])
    oracle_values = [
        float(row["metrics"][metric])
        for row in records
        if row.get("analysis_kind") == "main"
        and row.get("learner") == "oracle_alignment"
        and row.get("condition") == "absent"
    ]
    noun_values = [
        float(row["metrics"]["heldout_composition_noun_control_6way_macro_accuracy"])
        for row in records
        if row.get("analysis_kind") == "main"
        and row.get("learner") == config["evaluation"]["primary_learner"]
    ]
    action_capacity: list[float] = []
    scene_capacity: list[float] = []
    for corpus in corpora:
        oracle = {str(row["evaluation_id"]): row for row in corpus.evaluation_oracle}
        for item in corpus.evaluation_items:
            truth = oracle[item.evaluation_id]
            action_capacity.append(
                float(ACTIONS[int(np.asarray(item.action_observation).argmax())] == truth["true_action"])
            )
            scene_capacity.append(
                float(OBJECTS[int(np.asarray(item.scene_observation).argmax())] == truth["true_object"])
            )
    results = {
        "oracle_alignment_primary_accuracy": float(np.mean(oracle_values)),
        "direct_action_observation_capacity": float(np.mean(action_capacity)),
        "direct_scene_observation_capacity": float(np.mean(scene_capacity)),
        "noun_control_accuracy": float(np.mean(noun_values)),
    }
    thresholds = config["learnability_gates"]
    checks = {
        "oracle_alignment_positive_control": results["oracle_alignment_primary_accuracy"]
        >= float(thresholds["minimum_oracle_action_accuracy"]),
        "action_observation_capacity": results["direct_action_observation_capacity"]
        >= float(thresholds["minimum_observation_capacity"]),
        "scene_observation_capacity": results["direct_scene_observation_capacity"]
        >= float(thresholds["minimum_observation_capacity"]),
        "noun_non_action_control_learnable": results["noun_control_accuracy"]
        >= float(thresholds["minimum_noun_control_accuracy"]),
    }
    return {"valid": all(checks.values()), "checks": checks, "results": results}


def reserve_guard_audit(
    config: Mapping[str, Any], confirmation_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_checks = verify_confirmation_manifest(confirmation_manifest, config)
    corpus_seed = int(config["seeds"]["confirmation_reserve"]["corpus"][0])
    model_seed = int(config["seeds"]["confirmation_reserve"]["model"][0])
    blocked: dict[str, bool] = {}
    for operation in ("generate", "train", "evaluate", "read", "summarize"):
        try:
            guard_seed_operation(
                config, operation=operation, corpus_seed=corpus_seed, model_seed=model_seed
            )
        except PermissionError:
            blocked[operation] = True
        else:
            blocked[operation] = False
    return {
        "valid": all(manifest_checks.values()) and all(blocked.values()),
        "manifest_checks": manifest_checks,
        "blocked_without_future_explicit_authorization": blocked,
        "reserved_outcomes_read": 0,
        "reserved_outcomes_generated": 0,
        "authorization_present": False,
    }


def summarize_results(
    main_records: Sequence[Mapping[str, Any]],
    sensitivity_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary_learner = str(config["evaluation"]["primary_learner"])
    primary_metric = str(config["evaluation"]["primary_metric"])
    primary = paired_hierarchical_bootstrap(
        main_records, config, learner=primary_learner, metric=primary_metric,
        left_condition="synchronized", right_condition="shuffled",
    )
    means = condition_means(
        main_records, config, learner=primary_learner, metric=primary_metric
    )
    secondary_condition_contrasts = {
        control: paired_hierarchical_bootstrap(
            main_records, config, learner=primary_learner, metric=primary_metric,
            left_condition="synchronized", right_condition=control,
        )
        for control in ("time_shifted", "absent", "uninformative")
    }
    noun_contrast = paired_hierarchical_bootstrap(
        main_records, config, learner=primary_learner,
        metric="heldout_composition_noun_control_6way_macro_accuracy",
        left_condition="synchronized", right_condition="shuffled",
    )
    fixed_components: dict[str, Any] = {}
    for learner in (
        "exact_window",
        "latent_mil_single_occurrence",
        "cross_situational_uniform",
        "latent_mil_cross_no_null",
    ):
        transformed: list[dict[str, Any]] = []
        for row in main_records:
            if row["condition"] != "synchronized":
                continue
            if row["learner"] == primary_learner:
                transformed.append({**row, "learner": "component", "condition": "combined"})
            elif row["learner"] == learner:
                transformed.append({**row, "learner": "component", "condition": "ablation"})
        fixed_components[learner] = paired_hierarchical_bootstrap(
            transformed, config, learner="component", metric=primary_metric,
            left_condition="combined", right_condition="ablation",
        )
    sensitivity: dict[str, Any] = {}
    factors = sorted({str(row["stratum_factor"]) for row in sensitivity_records})
    for factor in factors:
        levels = sorted(
            {str(row["stratum_level"]) for row in sensitivity_records if row["stratum_factor"] == factor},
            key=lambda value: float(value),
        )
        sensitivity[factor] = {
            level: paired_hierarchical_bootstrap(
                sensitivity_records, config, learner=primary_learner,
                metric="supported_primary_lexical_mapping_accuracy",
                left_condition="synchronized", right_condition="shuffled",
                analysis_kind="sensitivity", stratum_factor=factor, stratum_level=level,
            )
            for level in levels
        }
    return {
        "schema_version": "synthetic-weak-alignment-aggregate-v1",
        "primary_estimand": primary,
        "primary_condition_means": means,
        "secondary_condition_contrasts": secondary_condition_contrasts,
        "noun_control_contrast": noun_contrast,
        "component_ablation_contrasts_within_synchronized": fixed_components,
        "factor_sensitivity": sensitivity,
        "multiplicity": {
            "primary_family": "one predeclared primary metric and one synchronized-minus-shuffled contrast",
            "primary_alpha": 0.05,
            "secondary_status": "descriptive_exploratory_no_multiplicity_adjusted_claims",
            "decision_uses_secondary_controls_as_directional_gates_not_additional discoveries": True,
        },
    }


def terminal_decision(
    aggregate: Mapping[str, Any], audits: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    primary = aggregate["primary_estimand"]
    means = aggregate["primary_condition_means"]
    noun = aggregate["noun_control_contrast"]
    sensitivity = aggregate["factor_sensitivity"]["side_informativeness"]
    informative_effects = [
        value["point_estimate"] for level, value in sensitivity.items() if float(level) > 0
    ]
    zero_effect = sensitivity[next(level for level in sensitivity if float(level) == 0.0)]["point_estimate"]
    validity = all(bool(value.get("valid")) for value in audits.values())
    thresholds = config["decision_rule"]
    go_checks = {
        "all_validity_and_control_audits_pass": validity,
        "minimum_primary_lift": primary["point_estimate"]
        >= float(thresholds["go_minimum_absolute_lift"]),
        "primary_ci_above_zero": primary["ci95_low"] > 0.0,
        "positive_pair_count": primary["positive_pair_count"]
        >= int(thresholds["go_minimum_positive_pairs"]),
        "synchronized_beats_all_matched_controls": all(
            means["synchronized"] > means[condition]
            for condition in ("shuffled", "time_shifted", "absent", "uninformative")
        ),
        "action_selective_over_noun_control": (
            primary["point_estimate"] - noun["point_estimate"]
            >= float(thresholds["go_minimum_action_minus_noun_lift"])
        ),
        "informative_strata_positive": all(value > 0 for value in informative_effects),
        "configured_uninformative_stratum_near_zero": abs(zero_effect)
        <= float(thresholds["go_maximum_uninformative_absolute_lift"]),
    }
    stop_checks = {
        "validity_or_positive_control_failure": not validity,
        "clear_harm": primary["ci95_high"] < 0.0,
    }
    if all(go_checks.values()):
        recommendation = "GO"
        reason = "all frozen development validity, positive-control, effect, selectivity, and sensitivity gates passed"
    elif any(stop_checks.values()):
        recommendation = "STOP"
        reason = "a frozen validity/learnability gate failed or synchronized training showed clear harm"
    else:
        recommendation = "REVISE"
        reason = "the study remained valid and learnable, but one or more frozen GO gates did not pass"
    return {
        "schema_version": "synthetic-weak-alignment-terminal-decision-v1",
        "protocol_id": config["protocol"]["id"],
        "study_phase": "development_only",
        "recommendation_for_later_frozen_synthetic_confirmation": recommendation,
        "reason": reason,
        "go_checks": go_checks,
        "stop_checks": stop_checks,
        "primary_effect": dict(primary),
        "confirmation_authorized": False,
        "confirmation_reserve_status": "untouched_and_guarded",
        "claim_boundary": (
            "This recommendation concerns a later synthetic lexical/action-grounding confirmation only; "
            "it is not evidence of complete language acquisition, timestamp agreement, infant learning, "
            "or BabyView-like ecological validity."
        ),
    }
