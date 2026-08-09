from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from . import PROTOCOL_ID
from .protocol import canonical_digest
from .statistics import analyze_informativeness, dependence_diagnostics as recompute_dependence


PACKAGE_GATES = (
    "old_line_closure_upheld",
    "original_brief_respected",
    "fresh_registries_disjoint",
    "old_ranges_never_operated",
    "confirmation_reserve_untouched",
    "construct_headroom",
    "corrective_disagreement_path",
    "mechanism_mutations",
    "acquisition_endpoints",
    "temporal_event_null_alignment",
    "matched_causal_controls",
    "test_time_side_withholding",
    "independent_generalization",
    "corpus_heterogeneity",
    "meaningful_model_stochasticity",
    "presence_null_dissociation",
    "frozen_statistics_and_decision_rule",
    "sample_size_frozen",
    "jobs_1_jobs_n_byte_identity",
    "official_micro_attempt_nonreplayable",
    "parallel_speedup_and_resource_ceiling",
    "isolated_shards_and_canonical_merge",
    "failure_atomic_no_outcome",
    "authorization_actual_paths_jobs_bound",
    "authorization_non_replayable",
    "manifest_exactness_and_path_safety",
    "authorization_negative_tests",
    "frozen_snapshot_integrity",
    "exactly_one_excluded_rehearsal",
    "rehearsal_scientific_inference_suppressed",
    "independent_rehearsal_recompute_byte_identity",
    "official_tests",
    "prior_evidence_preserved",
    "predecessor_v1_preserved",
    "predecessor_v2_preserved",
    "predecessor_v3_preserved",
    "development_outcome_count_zero",
    "confirmation_outcome_count_zero",
    "development_output_pristine",
    "atomic_launch_seal_preconditions",
    "prospective_prequalification_lock",
    "traceability_complete",
)


def _mean_by_condition_endpoint(
    averaged: Mapping[str, Any],
) -> dict[tuple[str, str], float]:
    values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in averaged["rows"]:
        for endpoint in (
            "lexical_acquisition_top1",
            "heldout_composition_top1",
        ):
            values[(str(row["condition"]), endpoint)].append(
                float(row["metrics"][endpoint])
            )
    return {
        key: float(np.mean(rows)) for key, rows in sorted(values.items())
    }


def mechanism_mutation_state_selectivity(
    merged_results: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the state-level causal selectivity of every frozen mutation."""
    expected_conditions = list(map(str, config["design"]["conditions"]))
    expected_mutations = list(
        map(str, config["design"]["mechanism_mutations"])
    )
    required_mutations = {
        "presence_side_disconnected",
        "null_update_disconnected",
        "evidence_disconnected",
        "semantic_update_disconnected",
        "semantic_side_zero",
        "within_bag_permuted",
        "old_agreement_gate",
        "null_head_ablation",
    }
    if set(expected_mutations) != required_mutations:
        raise RuntimeError("mechanism mutation inventory contract mismatch")

    rows = []
    seen_units: set[tuple[int, int]] = set()
    purposes = {str(unit.get("purpose")) for unit in merged_results}
    if len(purposes) != 1:
        raise RuntimeError("mechanism mutation state audit purpose mismatch")
    purpose = next(iter(purposes))
    if purpose not in {"construction_qualification", "development"}:
        raise RuntimeError("mechanism mutation state audit purpose is unsupported")
    expected_units = {
        (int(corpus), int(model))
        for corpus in config["resolved_registries"][purpose]["corpus"]
        for model in config["resolved_registries"][purpose]["model"]
    }
    for unit in merged_results:
        unit_key = (int(unit["corpus_seed"]), int(unit["model_seed"]))
        if unit_key in seen_units:
            raise RuntimeError("duplicate unit in mutation state audit")
        seen_units.add(unit_key)
        condition_values = list(unit["condition_results"])
        mutation_values = list(unit["mutation_results"])
        conditions = {
            str(row["condition"]): row for row in condition_values
        }
        mutations = {
            str(row["mutation"]): row for row in mutation_values
        }
        if not (
            len(condition_values) == len(expected_conditions)
            and len(conditions) == len(expected_conditions)
            and set(conditions) == set(expected_conditions)
            and len(mutation_values) == len(expected_mutations)
            and len(mutations) == len(expected_mutations)
            and set(mutations) == set(expected_mutations)
        ):
            raise RuntimeError("mechanism mutation state audit inventory mismatch")
        active = conditions["synchronized"]
        absent = conditions["absent"]

        def semantic_digest(row: Mapping[str, Any]) -> str:
            return canonical_digest(row["model"]["semantics"])

        def presence_digest(row: Mapping[str, Any]) -> str:
            return canonical_digest(
                {
                    "presence_intercept": row["model"]["presence_intercept"],
                    "presence_mismatch_weight": row["model"][
                        "presence_mismatch_weight"
                    ],
                }
            )

        presence_side = mutations["presence_side_disconnected"]
        null_update = mutations["null_update_disconnected"]
        evidence_disconnected = mutations["evidence_disconnected"]
        semantic_update = mutations["semantic_update_disconnected"]
        semantic_side = mutations["semantic_side_zero"]
        within_bag = mutations["within_bag_permuted"]
        old_gate = mutations["old_agreement_gate"]
        null_head = mutations["null_head_ablation"]
        checks = {
            "presence_side_semantics_exactly_preserved": semantic_digest(
                presence_side
            )
            == semantic_digest(active),
            "presence_side_parameters_changed": presence_digest(presence_side)
            != presence_digest(active),
            "presence_side_trace_disconnects_only_calibration_labels": (
                presence_side["trace"][
                    "null_side_to_presence_calibration_weight_effective"
                ]
                == 0.0
                and presence_side["trace"][
                    "null_side_to_semantic_event_mass_weight_effective"
                ]
                == active["trace"][
                    "null_side_to_semantic_event_mass_weight_effective"
                ]
                and presence_side["trace"][
                    "presence_side_disconnect_preserves_semantic_event_mass_gating"
                ]
                is True
                and presence_side["trace"][
                    "presence_calibration_uses_training_only_side_soft_labels"
                ]
                is False
            ),
            "null_update_semantics_exactly_preserved": semantic_digest(null_update)
            == semantic_digest(active),
            "null_update_parameters_changed": presence_digest(null_update)
            != presence_digest(active),
            "null_update_trace_is_frozen_and_side_label_inactive": (
                null_update["trace"]["null_update_frozen_at_initial_prior"]
                is True
                and null_update["trace"][
                    "null_side_to_presence_calibration_weight_effective"
                ]
                == 0.0
                and null_update["trace"][
                    "presence_calibration_uses_training_only_side_soft_labels"
                ]
                is False
            ),
            "evidence_disconnect_equals_absent_model": evidence_disconnected[
                "model_digest"
            ]
            == absent["model_digest"],
            "evidence_disconnect_equals_absent_predictions": evidence_disconnected[
                "prediction_digest"
            ]
            == absent["prediction_digest"],
            "evidence_disconnect_trace_has_no_side_path": (
                evidence_disconnected["trace"]["semantic_side_weight_effective"]
                == 0.0
                and evidence_disconnected["trace"][
                    "null_side_to_semantic_event_mass_weight_effective"
                ]
                == 0.0
                and evidence_disconnected["trace"][
                    "null_side_to_presence_calibration_weight_effective"
                ]
                == 0.0
                and evidence_disconnected["trace"][
                    "presence_calibration_uses_training_only_side_soft_labels"
                ]
                is False
            ),
            "semantic_update_disconnect_keeps_initial_semantics": semantic_update[
                "trace"
            ]["final_semantics_digest"]
            == semantic_update["trace"]["initial_semantics_digest"],
            "semantic_update_disconnect_trace_has_no_effective_semantic_side": (
                semantic_update["trace"]["semantic_side_weight_effective"]
                == 0.0
            ),
            "semantic_side_changes_semantics": semantic_digest(semantic_side)
            != semantic_digest(active),
            "semantic_side_trace_weight_is_zero": semantic_side["trace"][
                "semantic_side_weight_effective"
            ]
            == 0.0,
            "within_bag_permutation_changes_semantics": semantic_digest(within_bag)
            != semantic_digest(active),
            "within_bag_trace_names_mutation": within_bag["trace"]["mutation"]
            == "within_bag_permuted",
            "old_agreement_gate_changes_semantics": semantic_digest(old_gate)
            != semantic_digest(active),
            "old_gate_trace_activates_disagreement_gate": old_gate["trace"][
                "disagreement_gate"
            ]
            is True,
            "null_head_semantics_exactly_preserved": semantic_digest(null_head)
            == semantic_digest(active),
            "null_head_presence_parameters_changed": presence_digest(null_head)
            != presence_digest(active),
            "null_head_trace_matches_forced_ablation": null_head["trace"]
            == {
                "mutation": "null_head_ablation",
                "semantic_parameters_unchanged": True,
                "presence_intercept_forced_to": -20.0,
                "presence_mismatch_weight_forced_to": 0.0,
            },
        }
        rows.append(
            {
                "corpus_seed": unit_key[0],
                "model_seed": unit_key[1],
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )
    exact_unit_inventory = seen_units == expected_units
    return {
        "schema_version": "nursery-corrective-mutation-state-selectivity-v1",
        "status": (
            "PASS"
            if rows
            and exact_unit_inventory
            and all(row["status"] == "PASS" for row in rows)
            else "FAIL"
        ),
        "purpose": purpose,
        "unit_count": len(rows),
        "expected_unit_count": len(expected_units),
        "exact_unit_inventory": exact_unit_inventory,
        "rows": rows,
        "scientific_outcome": False,
    }


def construction_qualification_decision(
    *,
    averaged: Mapping[str, Any],
    mutation_summary: Mapping[str, Any],
    mechanism_qualification: Mapping[str, Any],
    dependence_diagnostics: Mapping[str, Any],
    merged_results: Sequence[Mapping[str, Any]],
    corpus_audits: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_corpora = list(
        map(
            int,
            config["resolved_registries"]["construction_qualification"][
                "corpus"
            ],
        )
    )
    expected_models = list(
        map(
            int,
            config["resolved_registries"]["construction_qualification"][
                "model"
            ],
        )
    )
    expected_conditions = list(map(str, config["design"]["conditions"]))
    averaged_rows = list(averaged.get("rows", []))
    averaged_keys = [
        (int(row["corpus_seed"]), str(row["condition"]))
        for row in averaged_rows
    ]
    expected_averaged_keys = {
        (corpus, condition)
        for corpus in expected_corpora
        for condition in expected_conditions
    }
    if not (
        averaged.get("schema_version")
        == "nursery-corrective-model-average-v3"
        and averaged.get("independent_unit") == "corpus_seed"
        and averaged.get("model_seed_order") == expected_models
        and averaged.get("exact_model_seed_set_required") is True
        and len(averaged_keys) == len(expected_averaged_keys)
        and len(set(averaged_keys)) == len(expected_averaged_keys)
        and set(averaged_keys) == expected_averaged_keys
        and all(
            row.get("model_seed_order") == expected_models
            and int(row.get("model_replicates_averaged", -1))
            == len(expected_models)
            and isinstance(row.get("metrics"), Mapping)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value))
                for value in row["metrics"].values()
            )
            for row in averaged_rows
        )
    ):
        raise RuntimeError("construction model-average inventory contract mismatch")
    recomputed_dependence = recompute_dependence(averaged, config)
    dependence_input_matches = canonical_digest(dict(dependence_diagnostics)) == canonical_digest(
        recomputed_dependence
    )
    means = _mean_by_condition_endpoint(averaged)
    negative_conditions = list(config["design"]["negative_controls"])
    endpoints = (
        "lexical_acquisition_top1",
        "heldout_composition_top1",
    )
    maximum_negative = float(
        config["qualification_gates"]["maximum_negative_control_top1"]
    )
    negative_rows = []
    for condition in negative_conditions:
        for endpoint in endpoints:
            value = means[(condition, endpoint)]
            negative_rows.append(
                {
                    "condition": condition,
                    "endpoint": endpoint,
                    "strict_top1": value,
                    "wrong_or_tied_fraction": 1.0 - value,
                    "status": (
                        "PASS"
                        if value <= maximum_negative
                        and 1.0 - value
                        >= float(
                            config["qualification_gates"][
                                "minimum_negative_control_wrong_or_tied_fraction"
                            ]
                        )
                        else "FAIL"
                    ),
                }
            )
    oracle_rows = []
    for endpoint in endpoints:
        value = means[(str(config["design"]["positive_control"]), endpoint)]
        oracle_rows.append(
            {
                "endpoint": endpoint,
                "strict_top1": value,
                "status": (
                    "PASS"
                    if value
                    >= float(config["qualification_gates"]["minimum_oracle_top1"])
                    else "FAIL"
                ),
            }
        )
    action_geometry_count = len(
        {str(row["action_geometry_digest"]) for row in corpus_audits}
    )
    episode_geometry_count = len(
        {str(row["episode_geometry_digest"]) for row in corpus_audits}
    )
    split_count = len(
        {str(row["heldout_composition_digest"]) for row in corpus_audits}
    )
    repetition_count = len(
        {str(row["repetition_count_digest"]) for row in corpus_audits}
    )
    ambiguity_strata = sorted(
        {str(row["ambiguity_stratum"]) for row in corpus_audits}
    )
    candidate_histogram_count = len(
        {str(row["candidate_count_histogram_digest"]) for row in corpus_audits}
    )
    grounded_count_count = len(
        {int(row["grounded_episode_count"]) for row in corpus_audits}
    )
    realized_ranges = {
        "primitive_action_geometry_mean_pairwise_distance": float(
            max(
                float(
                    row["action_geometry"]["slots"]["primitive"][
                        "mean_pairwise_distance"
                    ]
                )
                for row in corpus_audits
            )
            - min(
                float(
                    row["action_geometry"]["slots"]["primitive"][
                        "mean_pairwise_distance"
                    ]
                )
                for row in corpus_audits
            )
        ),
        "manner_action_geometry_mean_pairwise_distance": float(
            max(
                float(
                    row["action_geometry"]["slots"]["manner"][
                        "mean_pairwise_distance"
                    ]
                )
                for row in corpus_audits
            )
            - min(
                float(
                    row["action_geometry"]["slots"]["manner"][
                        "mean_pairwise_distance"
                    ]
                )
                for row in corpus_audits
            )
        ),
        "grounded_rate": float(
            max(
                float(row["grounded_episode_count"]) / float(row["episode_count"])
                for row in corpus_audits
            )
            - min(
                float(row["grounded_episode_count"]) / float(row["episode_count"])
                for row in corpus_audits
            )
        ),
        "foil_present_rate": float(
            max(float(row["realized_foil_present_rate"]) for row in corpus_audits)
            - min(float(row["realized_foil_present_rate"]) for row in corpus_audits)
        ),
        "observed_lag_mean": float(
            max(float(row["observed_lag_mean"]) for row in corpus_audits)
            - min(float(row["observed_lag_mean"]) for row in corpus_audits)
        ),
        "observation_peak": float(
            max(
                float(row["realized_mean_event_observation_peak"])
                for row in corpus_audits
            )
            - min(
                float(row["realized_mean_event_observation_peak"])
                for row in corpus_audits
            )
        ),
        "detector_accuracy": float(
            max(
                float(
                    row["condition_audit"]["detector_manipulation"]["synchronized"][
                        "strict_event_or_null_accuracy"
                    ]
                )
                for row in corpus_audits
            )
            - min(
                float(
                    row["condition_audit"]["detector_manipulation"]["synchronized"][
                        "strict_event_or_null_accuracy"
                    ]
                )
                for row in corpus_audits
            )
        ),
    }
    factor_names = (
        "grounded_rate_draw",
        "foil_rate_draw",
        "visibility_draw",
        "lag_mean_draw",
        "lag_sd_draw",
        "side_informativity_draw",
        "side_noise_draw",
    )
    factor_variation = {
        name: len({float(row[name]) for row in corpus_audits})
        for name in factor_names
    }
    variation_rows = averaged["within_corpus_model_variation"]
    average_metric_table = {
        (int(row["corpus_seed"]), str(row["condition"]), str(metric)): float(value)
        for row in averaged_rows
        for metric, value in row["metrics"].items()
    }
    expected_variation_fields = {
        "corpus_seed",
        "condition",
        "metric",
        "sample_sd",
        "range",
        "values",
    }
    for row in variation_rows:
        values = row.get("values") if isinstance(row, Mapping) else None
        if not (
            isinstance(row, Mapping)
            and set(row) == expected_variation_fields
            and isinstance(values, list)
            and len(values) == len(expected_models)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value))
                for value in values
            )
        ):
            raise RuntimeError("construction model-variation row contract mismatch")
        observations = np.asarray(values, dtype=float)
        key = (
            int(row["corpus_seed"]),
            str(row["condition"]),
            str(row["metric"]),
        )
        if not (
            key in average_metric_table
            and np.isclose(
                float(row["sample_sd"]),
                float(np.std(observations, ddof=1)),
                rtol=1e-12,
                atol=1e-15,
            )
            and np.isclose(
                float(row["range"]),
                float(np.ptp(observations)),
                rtol=1e-12,
                atol=1e-15,
            )
            and np.isclose(
                float(np.mean(observations)),
                average_metric_table[key],
                rtol=1e-12,
                atol=1e-15,
            )
        ):
            raise RuntimeError("construction model-variation binding mismatch")
    variation_gate_metrics = set(
        map(
            str,
            config["qualification_gates"]["model_variation_gate_metrics"],
        )
    )
    variation_excluded_conditions = set(
        map(
            str,
            config["qualification_gates"][
                "model_variation_excluded_conditions"
            ],
        )
    )
    variation_required_conditions = set(
        map(
            str,
            config["qualification_gates"][
                "model_variation_required_conditions"
            ],
        )
    )
    if (
        not variation_required_conditions
        or variation_required_conditions & variation_excluded_conditions
        or not variation_required_conditions.issubset(
            set(map(str, config["design"]["conditions"]))
        )
    ):
        raise RuntimeError("construction model-variation condition contract mismatch")
    eligible_variation_rows = [
        row
        for row in variation_rows
        if str(row["metric"]) in variation_gate_metrics
        and str(row["condition"]) in variation_required_conditions
    ]
    eligible_variation_keys = [
        (
            int(row["corpus_seed"]),
            str(row["condition"]),
            str(row["metric"]),
        )
        for row in eligible_variation_rows
    ]
    expected_eligible_variation_keys = {
        (corpus, condition, metric)
        for corpus in expected_corpora
        for condition in variation_required_conditions
        for metric in variation_gate_metrics
    }
    if not (
        len(eligible_variation_keys) == len(expected_eligible_variation_keys)
        and len(set(eligible_variation_keys))
        == len(expected_eligible_variation_keys)
        and set(eligible_variation_keys) == expected_eligible_variation_keys
    ):
        raise RuntimeError("construction model-variation inventory mismatch")
    nonzero_variation_fraction = float(
        np.mean(
            [
                float(row["sample_sd"])
                >= float(
                    config["qualification_gates"][
                        "minimum_nonzero_within_corpus_model_metric_sd"
                    ]
                )
                for row in eligible_variation_rows
            ]
        )
    )
    nonzero_variation_fraction_by_metric = {
        metric: float(
            np.mean(
                [
                    float(row["sample_sd"])
                    >= float(
                        config["qualification_gates"][
                            "minimum_nonzero_within_corpus_model_metric_sd"
                        ]
                    )
                    for row in eligible_variation_rows
                    if str(row["metric"]) == metric
                ]
            )
        )
        for metric in sorted(variation_gate_metrics)
    }
    nonzero_variation_fraction_by_metric_condition = {
        f"{metric}|{condition}": float(
            np.mean(
                [
                    float(row["sample_sd"])
                    >= float(
                        config["qualification_gates"][
                            "minimum_nonzero_within_corpus_model_metric_sd"
                        ]
                    )
                    for row in eligible_variation_rows
                    if str(row["metric"]) == metric
                    and str(row["condition"]) == condition
                ]
            )
        )
        for metric in sorted(variation_gate_metrics)
        for condition in sorted(variation_required_conditions)
    }
    audit_corpora = [int(row["corpus_seed"]) for row in corpus_audits]
    exact_corpus_audit_inventory = (
        len(audit_corpora) == len(expected_corpora)
        and len(set(audit_corpora)) == len(expected_corpora)
        and set(audit_corpora) == set(expected_corpora)
    )
    state_digests: defaultdict[tuple[int, str], list[str]] = defaultdict(list)
    prediction_digests: defaultdict[tuple[int, str], list[str]] = defaultdict(list)
    for unit in merged_results:
        corpus_seed = int(unit["corpus_seed"])
        for row in unit["condition_results"]:
            model = row["model"]
            key = (corpus_seed, str(row["condition"]))
            state_digests[key].append(
                canonical_digest(
                    {
                        "semantics": model["semantics"],
                        "presence_intercept": model["presence_intercept"],
                        "presence_mismatch_weight": model[
                            "presence_mismatch_weight"
                        ],
                    }
                )
            )
            prediction_digests[key].append(str(row["prediction_digest"]))
    expected_trace_key_count = len(corpus_audits) * len(
        config["design"]["conditions"]
    )
    state_and_prediction_variation_exact = (
        len(state_digests) == expected_trace_key_count
        and set(state_digests) == set(prediction_digests)
        and all(
            len(values) == len(expected_models)
            and len(set(values)) == len(expected_models)
            and len(prediction_digests[key]) == len(expected_models)
            and len(set(prediction_digests[key])) == len(expected_models)
            for key, values in state_digests.items()
        )
    )
    informativeness = analyze_informativeness(
        averaged,
        mutation_summary,
        corpus_audits,
        config,
        purpose="construction_qualification",
    )
    corpus_audit_checks = {
        "matched_shuffled_evidence_marginal": all(
            bool(
                row["condition_audit"][
                    "shuffle_preserves_learner_visible_evidence_marginal_by_block"
                ]
            )
            for row in corpus_audits
        ),
        "opaque_visible_identifiers": all(
            bool(row["visible_identifiers_are_opaque_and_truth_independent_in_format"])
            for row in corpus_audits
        ),
        "independent_train_eval_provenance": all(
            bool(row["train_evaluation_generator_namespaces_disjoint"])
            and int(row["provenance"]["train_evaluation_instance_id_overlap"]) == 0
            and bool(row["every_constituent_exposed_in_training"])
            for row in corpus_audits
        ),
        "heldout_compositions_absent_from_all_training_events": all(
            bool(row["all_training_candidate_events_exclude_heldout_compositions"])
            and int(row["heldout_candidate_event_count"]) == 0
            for row in corpus_audits
        ),
        "evaluation_schema_and_side_withholding": bool(
            config["design"]["evaluation"]["side_modality_withheld"]
        )
        and all(
            bool(row["evaluation_side_fields_absent"])
            and bool(row["evaluation_top_level_schema_exact_allowlist"])
            for row in corpus_audits
        ),
        "correction_challenge_structure": all(
            bool(row["correction_challenges_cover_every_training_composition"])
            and bool(row["all_grounded_episodes_are_correction_challenges"])
            and bool(row["every_training_composition_has_present_and_null_support"])
            and bool(row["foil_map_is_bijective"])
            and bool(row["foil_map_changes_both_action_dimensions"])
            for row in corpus_audits
        ),
        "matched_factorized_event_background_geometry": all(
            bool(row["matched_background_proposal_banks"])
            and bool(
                row[
                    "event_and_background_proposal_counts_and_widths_match"
                ]
            )
            and bool(row["signed_shift_guard_bands_valid"])
            and bool(row["realized_event_widths_within_frozen_range"])
            and bool(row["realized_inter_event_gaps_within_frozen_range"])
            and bool(row["all_condition_evidence_factorized"])
            and bool(
                row["condition_audit"][
                    "detector_scoring_factorizes_presence_from_conditional_event"
                ]
            )
            and bool(
                row["condition_audit"][
                    "event_and_null_logits_never_jointly_ranked"
                ]
            )
            for row in corpus_audits
        ),
    }
    zero_exposure_rows = [
        row["metrics"]
        for row in averaged["rows"]
    ]
    zero_exposure_passed = all(
        float(metrics["zero_exposure.strict_top1"]) == 0.0
        and float(metrics["zero_exposure.tie_rate"]) == 1.0
        for metrics in zero_exposure_rows
    )
    mutation_state_audit = mechanism_mutation_state_selectivity(
        merged_results, config
    )
    mutation_state_selectivity_passed = mutation_state_audit["status"] == "PASS"
    gates = {
        "headroom_every_negative_control": all(
            row["status"] == "PASS" for row in negative_rows
        ) and all(
            bool(row["passed"])
            for row in informativeness["details"][
                "primary_comparator_headroom"
            ].values()
        ),
        "oracle_positive_control": all(
            row["status"] == "PASS" for row in oracle_rows
        ),
        "prespecified_disagreement_mutations": mechanism_qualification["status"]
        == "PASS"
        and mutation_state_selectivity_passed,
        "action_geometry_varies_by_corpus": action_geometry_count
        >= int(
            config["qualification_gates"][
                "minimum_distinct_action_geometry_digests"
            ]
        ),
        "episode_geometry_varies_by_corpus": episode_geometry_count
        >= int(
            config["qualification_gates"][
                "minimum_distinct_episode_geometry_digests"
            ]
        ),
        "action_geometry_distance_structure_varies": min(
            realized_ranges["primitive_action_geometry_mean_pairwise_distance"],
            realized_ranges["manner_action_geometry_mean_pairwise_distance"],
        )
        >= float(
            config["qualification_gates"][
                "minimum_action_geometry_pairwise_distance_range"
            ]
        ),
        "action_geometry_is_separated_and_shared_across_train_eval": all(
            bool(row["action_geometry"]["all_rows_normalized"])
            and bool(row["action_geometry"]["all_entries_nonnegative"])
            and bool(row["action_geometry"]["all_margins_pass"])
            and bool(row["train_evaluation_use_same_action_prototypes"])
            and min(
                float(
                    row["action_geometry"]["slots"][slot][
                        "minimum_self_cross_dot_margin"
                    ]
                )
                for slot in ("primitive", "manner")
            )
            >= float(
                config["qualification_gates"][
                    "minimum_action_geometry_self_cross_dot_margin"
                ]
            )
            for row in corpus_audits
        ),
        "heldout_splits_vary_by_corpus": split_count
        >= int(
            config["qualification_gates"]["minimum_distinct_heldout_split_digests"]
        ),
        "repetition_patterns_vary_by_corpus": repetition_count
        >= int(
            config["qualification_gates"]["minimum_distinct_repetition_digests"]
        ),
        "ambiguity_strata_vary_by_corpus": len(ambiguity_strata)
        >= int(config["qualification_gates"]["minimum_distinct_ambiguity_strata"])
        and all(
            sum(str(row["ambiguity_stratum"]) == stratum for row in corpus_audits)
            >= int(
                config["qualification_gates"][
                    "minimum_corpora_per_ambiguity_stratum"
                ]
            )
            for stratum in ambiguity_strata
        ),
        "realized_candidate_counts_vary_by_corpus": candidate_histogram_count
        >= int(
            config["qualification_gates"][
                "minimum_distinct_candidate_count_histograms"
            ]
        ),
        "realized_grounded_rates_vary_by_corpus": grounded_count_count
        >= int(
            config["qualification_gates"][
                "minimum_distinct_grounded_episode_counts"
            ]
        )
        and realized_ranges["grounded_rate"]
        >= float(
            config["qualification_gates"][
                "minimum_realized_grounded_rate_range"
            ]
        ),
        "realized_ambiguity_varies_by_corpus": realized_ranges["foil_present_rate"]
        >= float(config["qualification_gates"]["minimum_realized_foil_rate_range"]),
        "realized_lag_varies_by_corpus": realized_ranges["observed_lag_mean"]
        >= float(config["qualification_gates"]["minimum_observed_lag_mean_range"]),
        "realized_visibility_varies_by_corpus": realized_ranges["observation_peak"]
        >= float(
            config["qualification_gates"][
                "minimum_realized_visibility_peak_range"
            ]
        ),
        "realized_side_informativeness_varies_by_corpus": realized_ranges[
            "detector_accuracy"
        ]
        >= float(
            config["qualification_gates"][
                "minimum_realized_detector_accuracy_range"
            ]
        ),
        "all_frozen_factors_vary": all(
            value
            >= int(
                config["qualification_gates"][
                    "minimum_distinct_construction_factor_draws"
                ]
            )
            for value in factor_variation.values()
        ),
        "model_training_stochasticity_is_numerically_relevant": all(
            value
            >= float(
                config["qualification_gates"][
                    "minimum_nonzero_continuous_model_variation_fraction"
                ]
            )
            for value in nonzero_variation_fraction_by_metric_condition.values()
        )
        and len(eligible_variation_rows)
        == len(expected_corpora)
        * len(variation_gate_metrics)
        * len(variation_required_conditions)
        and state_and_prediction_variation_exact,
        "all_informativeness_and_manipulation_gates": informativeness["status"]
        == "PASS",
        "presence_null_effects_not_identical": dependence_input_matches
        and recomputed_dependence["status"]
        == "PASS"
        and informativeness["gates"]["null_head_is_selectively_causal"]
        and informativeness["gates"]["presence_side_path_is_selectively_causal"]
        and informativeness["gates"]["null_update_path_is_selectively_causal"]
        and informativeness["gates"][
            "active_and_oracle_presence_states_are_informative"
        ],
        **corpus_audit_checks,
        "test_time_side_withheld": corpus_audit_checks[
            "evaluation_schema_and_side_withholding"
        ],
        "zero_exposure_leakage_control": zero_exposure_passed,
        "fixture_cohort_exact_registry_inventory_retained": (
            exact_corpus_audit_inventory
            and mutation_state_audit["exact_unit_inventory"] is True
        ),
    }
    return {
        "schema_version": "nursery-corrective-construction-decision-v3",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "negative_control_headroom": negative_rows,
        "oracle_positive_control": oracle_rows,
        "mechanism_qualification": mechanism_qualification,
        "action_geometry_digest_count": action_geometry_count,
        "episode_geometry_digest_count": episode_geometry_count,
        "heldout_split_digest_count": split_count,
        "repetition_digest_count": repetition_count,
        "ambiguity_strata": ambiguity_strata,
        "candidate_histogram_count": candidate_histogram_count,
        "grounded_episode_count_distinct_count": grounded_count_count,
        "realized_factor_ranges": realized_ranges,
        "factor_distinct_counts": factor_variation,
        "nonzero_model_variation_fraction": nonzero_variation_fraction,
        "nonzero_model_variation_fraction_by_metric": (
            nonzero_variation_fraction_by_metric
        ),
        "nonzero_model_variation_fraction_by_metric_condition": (
            nonzero_variation_fraction_by_metric_condition
        ),
        "eligible_model_variation_row_count": len(eligible_variation_rows),
        "model_variation_gate_metrics": sorted(variation_gate_metrics),
        "model_variation_excluded_conditions": sorted(
            variation_excluded_conditions
        ),
        "model_variation_required_conditions": sorted(
            variation_required_conditions
        ),
        "state_and_prediction_variation_exact": (
            state_and_prediction_variation_exact
        ),
        "informativeness": informativeness,
        "dependence_diagnostics": recomputed_dependence,
        "supplied_dependence_diagnostics_matches_recomputation": (
            dependence_input_matches
        ),
        "corpus_audit_checks": corpus_audit_checks,
        "zero_exposure_leakage_control": {
            "status": "PASS" if zero_exposure_passed else "FAIL",
            "required_strict_top1": 0.0,
            "required_tie_rate": 1.0,
        },
        "mechanism_mutation_state_selectivity": mutation_state_audit,
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }


def package_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    gates = value.get("gates", {})
    if not isinstance(gates, Mapping):
        gates = {}
    missing = sorted(set(PACKAGE_GATES) - set(gates))
    unexpected = sorted(set(gates) - set(PACKAGE_GATES))
    non_boolean = sorted(
        key for key, child in gates.items() if not isinstance(child, bool)
    )
    contradictions = list(value.get("contradictions", []))
    well_formed = not missing and not unexpected and not non_boolean and not contradictions
    passed = well_formed and all(gates.get(name) is True for name in PACKAGE_GATES)
    return {
        "schema_version": "nursery-corrective-package-terminal-v4",
        "protocol_id": PROTOCOL_ID,
        "terminal_state": (
            "CORRECTIVE_DEVELOPMENT_LAUNCH_READY" if passed else "REVISE"
        ),
        "status": "PASS" if passed else "FAIL",
        "development_authorized": passed,
        "confirmation_authorized": False,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "claim_scope": "synthetic lexical/action grounding and held-out transfer only",
        "infant_learning_claim_authorized": False,
        "ecological_validity_claim_authorized": False,
        "gates": dict(gates),
        "input_contract": {
            "well_formed": well_formed,
            "missing_gates": missing,
            "unexpected_gates": unexpected,
            "non_boolean_gates": non_boolean,
            "contradictions": contradictions,
        },
    }


AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "status",
        "development_authorized",
        "confirmation_authorized",
        "development_outcome_count",
        "confirmation_outcome_count",
        "authorization_scope",
        "authorization_non_replayable",
        "resolved_repository_root",
        "resolved_snapshot_root",
        "resolved_authorization_path",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "resolved_publication_seal_path",
        "resolved_claim_path",
        "resolved_capability_receipt_path",
        "resolved_consumption_root",
        "resolved_consumed_authorization_path",
        "required_cwd",
        "required_jobs",
        "required_thread_environment",
        "required_python_executable",
        "required_python_sha256",
        "required_runner_path",
        "required_runner_sha256",
        "required_config_sha256",
        "required_snapshot_manifest_sha256",
        "required_freeze_receipt_sha256",
        "required_prequalification_design_lock_sha256",
        "required_micro_attempt_sha256",
        "required_worker_launcher_preflight_sha256",
        "required_environment_sha256",
        "required_registries_sha256",
        "required_package_core_manifest_sha256",
        "required_package_terminal_sha256",
        "required_excluded_rehearsal_sha256",
        "required_recompute_comparison_sha256",
        "required_official_tests_sha256",
        "required_benchmark_sha256",
        "required_preservation_proof_sha256",
        "development_registry",
        "confirmation_reserve_digest",
        "exact_argv",
        "exact_shell_command",
        "authorization_digest",
    }
)


def authorization_payload(**values: Any) -> dict[str, Any]:
    payload = {
        "schema_version": "nursery-corrective-development-authorization-v4",
        "protocol_id": PROTOCOL_ID,
        "status": "CORRECTIVE_DEVELOPMENT_LAUNCH_READY",
        "development_authorized": True,
        "confirmation_authorized": False,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "authorization_scope": "exactly_one_later_development_attempt",
        "authorization_non_replayable": True,
        **values,
    }
    missing = AUTHORIZATION_FIELDS - set(payload) - {"authorization_digest"}
    unexpected = set(payload) - AUTHORIZATION_FIELDS
    if missing or unexpected:
        raise ValueError(
            f"authorization payload schema mismatch missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    return {**payload, "authorization_digest": canonical_digest(payload)}


def verify_authorization(value: Mapping[str, Any]) -> dict[str, Any]:
    problems = []
    if set(value) != AUTHORIZATION_FIELDS:
        problems.append("exact_schema")
    payload = {key: child for key, child in value.items() if key != "authorization_digest"}
    if value.get("authorization_digest") != canonical_digest(payload):
        problems.append("digest")
    if value.get("protocol_id") != PROTOCOL_ID:
        problems.append("protocol_id")
    if value.get("status") != "CORRECTIVE_DEVELOPMENT_LAUNCH_READY":
        problems.append("status")
    if value.get("development_authorized") is not True:
        problems.append("development_authorized")
    if value.get("confirmation_authorized") is not False:
        problems.append("confirmation_authorized")
    if int(value.get("development_outcome_count", -1)) != 0:
        problems.append("development_outcome_count")
    if int(value.get("confirmation_outcome_count", -1)) != 0:
        problems.append("confirmation_outcome_count")
    if value.get("authorization_non_replayable") is not True:
        problems.append("non_replayable")
    if value.get("schema_version") != "nursery-corrective-development-authorization-v4":
        problems.append("schema_version")
    if value.get("authorization_scope") != "exactly_one_later_development_attempt":
        problems.append("authorization_scope")
    if int(value.get("required_jobs", 0)) < 1:
        problems.append("required_jobs")
    for field in (
        "resolved_repository_root",
        "resolved_snapshot_root",
        "resolved_authorization_path",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "resolved_publication_seal_path",
        "resolved_claim_path",
        "resolved_capability_receipt_path",
        "resolved_consumption_root",
        "resolved_consumed_authorization_path",
        "required_cwd",
        "required_python_executable",
        "required_runner_path",
    ):
        child = value.get(field)
        pure = PurePosixPath(child) if isinstance(child, str) else None
        if (
            not isinstance(child, str)
            or not child.startswith("/")
            or "\\" in child
            or "//" in child
            or pure is None
            or not pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != child
        ):
            problems.append(f"absolute_path:{field}")
    if not isinstance(value.get("exact_argv"), list) or not value.get("exact_argv"):
        problems.append("exact_argv")
    if not isinstance(value.get("exact_shell_command"), str) or not value.get(
        "exact_shell_command"
    ):
        problems.append("exact_shell_command")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}
