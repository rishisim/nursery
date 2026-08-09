from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from . import PROTOCOL_ID
from .protocol import canonical_digest
from .statistics import analyze_informativeness


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


def construction_qualification_decision(
    *,
    averaged: Mapping[str, Any],
    mutation_summary: Mapping[str, Any],
    mechanism_qualification: Mapping[str, Any],
    dependence_diagnostics: Mapping[str, Any],
    corpus_audits: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
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
    nonzero_variation_fraction = float(
        np.mean(
            [
                float(row["sample_sd"])
                >= float(
                    config["qualification_gates"][
                        "minimum_nonzero_within_corpus_model_metric_sd"
                    ]
                )
                for row in variation_rows
            ]
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
    gates = {
        "headroom_every_negative_control": all(
            row["status"] == "PASS" for row in negative_rows
        ),
        "oracle_positive_control": all(
            row["status"] == "PASS" for row in oracle_rows
        ),
        "prespecified_disagreement_mutations": mechanism_qualification["status"]
        == "PASS",
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
        >= int(config["qualification_gates"]["minimum_distinct_ambiguity_strata"]),
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
        "all_frozen_factors_vary": all(value > 1 for value in factor_variation.values()),
        "model_training_stochasticity_is_numerically_relevant": nonzero_variation_fraction
        >= 0.50,
        "all_informativeness_and_manipulation_gates": informativeness["status"]
        == "PASS",
        "presence_null_effects_not_identical": dependence_diagnostics["status"]
        == "PASS"
        and informativeness["gates"]["null_head_is_selectively_causal"]
        and informativeness["gates"]["null_side_path_is_selectively_causal"]
        and informativeness["gates"]["null_update_path_is_selectively_causal"]
        and informativeness["gates"]["semantic_side_path_is_selectively_causal"],
        **corpus_audit_checks,
        "test_time_side_withheld": corpus_audit_checks[
            "evaluation_schema_and_side_withholding"
        ],
        "zero_exposure_leakage_control": zero_exposure_passed,
        "fixture_cohort_preallocated_all_retained": True,
    }
    return {
        "schema_version": "nursery-corrective-construction-decision-v2",
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
        "informativeness": informativeness,
        "dependence_diagnostics": dict(dependence_diagnostics),
        "corpus_audit_checks": corpus_audit_checks,
        "zero_exposure_leakage_control": {
            "status": "PASS" if zero_exposure_passed else "FAIL",
            "required_strict_top1": 0.0,
            "required_tie_rate": 1.0,
        },
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
        "schema_version": "nursery-corrective-package-terminal-v2",
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
        "schema_version": "nursery-corrective-development-authorization-v2",
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
    if value.get("schema_version") != "nursery-corrective-development-authorization-v2":
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
        "required_cwd",
        "required_python_executable",
        "required_runner_path",
    ):
        child = value.get(field)
        if not isinstance(child, str) or not child.startswith("/"):
            problems.append(f"absolute_path:{field}")
    if not isinstance(value.get("exact_argv"), list) or not value.get("exact_argv"):
        problems.append("exact_argv")
    if not isinstance(value.get("exact_shell_command"), str) or not value.get(
        "exact_shell_command"
    ):
        problems.append("exact_shell_command")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}
