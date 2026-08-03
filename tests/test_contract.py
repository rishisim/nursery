from __future__ import annotations

import json
from pathlib import Path

import pytest

from nursery_egobaby_preflight.contract import (
    canonical_json_sha256,
    lexical_macro_wiring,
    schedule_cycle,
    validate_phase_state,
)


def test_frozen_schedule_is_exact_4_1_1() -> None:
    config = json.loads(Path("configs/egobaby_cuda_preflight.json").read_text())
    assert schedule_cycle(config["learner"]["schedule"]) == [
        "contrastive",
        "contrastive",
        "contrastive",
        "contrastive",
        "mlm",
        "dinov2",
    ]


def test_config_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})


def test_lexical_wiring_requires_noun_then_adjective() -> None:
    result = lexical_macro_wiring({"noun": [1, 0], "adjective": [0, 1]})
    assert result == {"noun": 0.5, "adjective": 0.5, "lexical_macro": 0.5}
    with pytest.raises(ValueError, match="ordered exactly"):
        lexical_macro_wiring({"adjective": [0, 1], "noun": [1, 0]})


def test_phase4_preregistration_preserves_frozen_contract() -> None:
    config = json.loads(Path("configs/synthetic_video_preregistration.json").read_text())
    assert config["schema_version"] == 6
    assert config["status"] == "PHASE4_CORRECTED_ASSETS_PASS_AMBITIOUS_LEARNER_EFFECTIVE_H3_AMENDMENT_FROZEN_PRIOR_NO_GOS_PRESERVED_GENERATION_LICENSE_BLOCKED"
    validate_phase_state(config)
    premodel = config["mechanistic_training_tuple_premodel_result"]
    assert premodel["status"] == "PASS_ARTIFACTS_READY_LOCAL_RELOAD_PENDING_BLIND_SIZING"
    assert premodel["dependency_manifest_commitment_sha256"] == (
        "8c787a01f2e0f6224bc96989d3e3bd28ef6f6b03e0459599507636e41c85b527"
    )
    assert premodel["model_inference_executed"] is False
    fixture_protocol = config["mechanistic_training_tuple_public_fixture_protocol"]
    assert fixture_protocol["status"] == "FROZEN_BEFORE_PUBLIC_MODEL_OUTCOMES"
    assert fixture_protocol["protocol_commitment_sha256"] == (
        "506a1f41a3685ca777f3c9d23f6f9b884523acec2a78080d5de2547b3324251d"
    )
    assert fixture_protocol["action_direction_labels"] == 8
    assert fixture_protocol["public_outcome_opened"] is False
    fixture_preparation = config[
        "mechanistic_training_tuple_fixture_preparation_amendment"
    ]
    assert fixture_preparation["status"] == (
        "FROZEN_BEFORE_MANIFEST_CONSTRUCTION_OR_PUBLIC_MODEL_OUTCOMES"
    )
    assert fixture_preparation["preparation_amendment_commitment_sha256"] == (
        "1cc8d0e3498da5785a2c2105307bf6d5ab20dd10f839ec0f2b92b9def372ff1d"
    )
    assert fixture_preparation["items_per_partition"] == 312
    assert fixture_preparation["public_outcome_opened"] is False
    fixture_feasibility = config[
        "mechanistic_training_tuple_fixture_feasibility_repair_amendment"
    ]
    assert fixture_feasibility["status"] == (
        "FROZEN_AFTER_PREMODEL_FIXTURE_YIELD_STOP_BEFORE_ANY_PUBLIC_MODEL_OUTCOME"
    )
    assert fixture_feasibility["fixture_feasibility_repair_commitment_sha256"] == (
        "e5fd286e9b8140583a37b855fe7125d7c6a0a2e3b57589b53294f77d28e47048"
    )
    assert fixture_feasibility["scientific_thresholds_changed"] is False
    fixture_result = config["mechanistic_training_tuple_fixture_feasibility_result"]
    assert fixture_result["status"] == "NO_GO_ANNOTATION_ONLY_FIXTURE_SOURCE_YIELD"
    assert fixture_result["fixture_feasibility_commitment_sha256"] == (
        "dee0a37548d75fc29f829159ce3ad648a63288339c7bc173f8201483e51213e7"
    )
    assert fixture_result["blocking_stratum"] == "hand_no_contact"
    assert fixture_result["required_count"] == 12
    assert fixture_result["available_count"] == 0
    assert fixture_result["model_inference_executed"] is False
    assert fixture_result["governed_C_reopened"] is False
    assert fixture_result["LTX_or_synthetic_learner_run"] is False
    correction = config["mechanistic_training_tuple_visor_hos_correction_amendment"]
    assert correction["status"] == (
        "FROZEN_BEFORE_NEW_PUBLIC_SOURCE_INVENTORY_MODEL_C_GENERATOR_OR_LEARNER_OUTCOMES"
    )
    assert correction["amendment_commitment_sha256"] == (
        "31c1c26f76c5c7dc09e34aff9d5dde291d20631035b07a975d6b8ff5861bf8d4"
    )
    assert correction["prior_fixture_no_go_preserved"] == (
        "dee0a37548d75fc29f829159ce3ad648a63288339c7bc173f8201483e51213e7"
    )
    assert correction["per_partition_counts"] == [48, 48, 48]
    assert correction["five_critical_axes_must_pass"] is True
    assert correction["axes_required"] == 6
    assert correction["public_outcome_opened"] is False
    complete_source = config[
        "mechanistic_training_tuple_visor_hos_complete_source_result"
    ]
    assert complete_source["status"] == "NO_GO_COMPLETE_SOURCE_FEASIBILITY"
    assert complete_source["external_complete_source_record_commitment_sha256"] == (
        "5f4aeff25da36cde4c35699de7031b63ae427d1aee072370bb3844e3c4413b37"
    )
    assert complete_source["Charades_action_counts"] == [44, 44]
    assert complete_source["model_inference_executed"] is False
    assert complete_source["governed_C_reopened"] is False
    assert complete_source["LTX_or_synthetic_learner_run"] is False
    runtime_preparation = config[
        "mechanistic_training_tuple_runtime_preparation_result"
    ]
    assert runtime_preparation["status"] == (
        "PASS_55_DEPENDENCY_RUNTIME_READY_LOCAL_RELOAD_BLIND_SIZING_PENDING"
    )
    assert runtime_preparation["dependency_count"] == 55
    assert runtime_preparation["runtime_dependency_commitment_sha256"] == (
        "df15ff20c2f1e137530ec6f8a6f848ed676f2bcde0ec48148736c23ddd6fc0c4"
    )
    assert runtime_preparation["model_inference_executed"] is False
    sizing_validation = config[
        "mechanistic_training_tuple_sizing_validation_amendment"
    ]
    assert sizing_validation["status"] == (
        "FROZEN_BEFORE_SIZING_RERUN_OR_PUBLIC_FIXTURE_OUTCOMES"
    )
    assert sizing_validation["validation_commitment_sha256"] == (
        "afc936f742bd4313c35ff2e9a11a2389c589675c03309bbf09d8f8ab718ea2d5"
    )
    assert sizing_validation["padding_mask_exact_required"] is True
    assert sizing_validation["public_fixture_outcome_opened"] is False
    sizing_result = config["mechanistic_training_tuple_sizing_result"]
    assert sizing_result["status"] == "PASS_LABEL_BLIND_LOCAL_RELOAD_SIZING"
    assert sizing_result["module_count"] == 8
    assert sizing_result["finite_output_count"] == 8
    assert sizing_result["tuple_sizing_commitment_sha256"] == (
        "b627590518b2e54daeace3d1c52e6918b41c2e203b42742538135c6a4c63e029"
    )
    assert sizing_result["public_fixture_outcome_opened"] is False
    runtime = config["mechanistic_training_tuple_runtime_amendment"]
    assert runtime["runtime_amendment_commitment_sha256"] == (
        "eb878d8c68aa6f79b5115502beb4c8b64d84e9f495b7cec4513abc3e94effbea"
    )
    assert runtime["compatibility_adapter_count"] == 7
    assert schedule_cycle(config["learner"]["schedule"]) == [
        "contrastive",
        "contrastive",
        "contrastive",
        "contrastive",
        "mlm",
        "dinov2",
    ]
    assert config["comparisons"]["primary"] == (
        "Synthetic-full_minus_Real-full_two_sided_equivalence"
    )
    assert config["endpoint"]["common_asset_across_arms"] is True
    assert config["analysis"]["equivalence_margin_bounds_absolute"] == [0.02, 0.05]
    assert config["analysis"]["minimum_common_seeds"] >= 3
    assert sum(config["split"]["target_proportions"].values()) == 1.0
    assert sum(config["split"]["minimum_children"].values()) == 6
    assert config["split"]["calibration_C"]["children"] == 18
    assert config["split"]["eligible_confirmatory_catalog"]["children"] == 40
    assert config["split"]["eligible_confirmatory_catalog"]["duration_metadata_complete"] is True
    assert sum(
        config["split"]["assigned_child_counts_if_all_40_remain_eligible"].values()
    ) == 40
    assert config["unblinding"]["synthetic_scores_sealed_until_real_only_gate_passes"] is True
    assert config["cost"]["existing_childlens_cost_is_zero"] is False
    assert config["gates"]["childlens_academic_noncommercial_access"] == "ESTABLISHED"
    assert config["gates"]["childlens_aggregate_reporting"] == "ESTABLISHED"
    assert config["gates"]["current_childlens_storage_encryption"] == (
        "PASS_AES_256_ENCRYPTED_SPARSEBUNDLE"
    )
    assert config["gates"]["storage_migration_regular_files_verified"] == 67087
    assert config["gates"]["unencrypted_source_removed_after_verification"] is True
    assert config["gates"]["authorized_read_only_aggregate_inventory"] == "COMPLETE"
    assert config["gates"]["personnel_model"] == (
        "SINGLE_AUTHORIZED_APPLICANT_WITH_STAGED_PROCEDURAL_ROLES"
    )
    assert config["unblinding"]["personnel_independence"] is False
    assert config["gates"]["governed_cuda_qualification"] == (
        "PASS_PROPORTIONATE_INSTITUTIONAL_CONTROLS"
    )
    assert config["governed_compute"]["account_active"] is True
    assert config["governed_compute"]["login_node_egress_test"][
        "open_egress_observed"
    ] is True
    assert config["governed_compute"]["login_node_egress_test"][
        "phase3_blocking"
    ] is False
    assert config["governed_compute"]["orientation"]["status"] == (
        "OFFICIAL_SELF_STUDY_REVIEWED"
    )
    assert config["governed_compute"]["compute_node_egress_test"][
        "open_egress_observed"
    ] is True
    assert config["governed_compute"]["compute_node_egress_test"][
        "phase3_blocking"
    ] is False
    assert (
        config["governed_compute"]["compute_node_egress_test"]["gpu_requested"] is False
    )
    assert config["governed_compute"]["container_runtime"][
        "network_namespace_required"
    ] is False
    assert config["governed_compute"]["gpu_job_launched"] is False
    assert config["governed_compute"]["restricted_data_transfer_to_juno_authorized"] is True
    assert "no_restricted_artifact_API_hosted_GPU_cloud_Git_or_third_party_transfer" in (
        config["governed_compute"]["restricted_job_controls"]
    )
    assert "add_or_affirm_yding_slurm_fair_share_association" in config[
        "governed_compute"
    ]["administrative_followup_nonblocking"]
    assert config["gates"]["phase4_authorized"] is True
    assert config["gates"]["childlens_audio_processing_authorized"] is True
    assert config["gates"]["learner_effective_implementation_authorized"] is True
    assert config["gates"]["learner_effective_public_qualification_authorized"] is False
    assert config["language"]["identical_real_synthetic_pipeline_frozen"] is True
    assert config["gates"]["generator_work_authorized"] is False
    assert config["gates"]["real_only_training_authorized"] is False
    assert config["governed_compute"]["additional_slurm_account_name_committed_to_git"] is False


def test_phase_state_validator_rejects_contradictory_nested_state() -> None:
    config = json.loads(Path("configs/synthetic_video_preregistration.json").read_text())
    config["gates"]["phase4_status"] = "PASS_BOTH_COMMON_ASSET_FAMILIES_HASHED_AND_SEALED"
    with pytest.raises(ValueError, match="contradictory Phase 4 states"):
        validate_phase_state(config)


def test_one_hour_coverage_redesign_is_exploratory_and_exact_schedule() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    assert config["budgets_credited_hours"] == {
        "real": 1,
        "synthetic_accepted": 1,
        "synthetic_above_one_hour_prohibited": True,
    }
    assert len(config["learner"]["seeds"]) == 3
    assert schedule_cycle(config["learner"]["schedule"]) == ["contrastive"] * 4 + ["mlm", "dinov2"]
    assert config["learner"]["objective_steps"] == 4668
    assert config["learner"]["objective_counts"] == {"contrastive": 3112, "mlm": 778, "dinov2": 778}
    assert config["sealed_prior_570_step_pilot"]["status"] == "PRESERVED_NOT_REINTERPRETED"
    assert config["schema_version"] == 15
    amendment = config["ambitious_learner_effective_h3_amendment"]
    commitment = amendment.pop("amendment_commitment_sha256")
    assert commitment == canonical_json_sha256(amendment)
    assert commitment == "d907d2479ba88c7e51d25e935e429cb5860e550a82620313e502482383e2855d"
    assert amendment["order_action_role_amendment"]["new_role"] == (
        "SUPPORTING_DIAGNOSTIC_NONBLOCKING"
    )
    assert amendment["hierarchy"]["public_and_governed_C_combined_pass_rule"].startswith(
        "all five critical axes pass"
    )
    assert config["post_gate_descriptive_extension"]["preserved_gate_result"]["mean_gain"] == 0.017661900756938558
    assert config["post_gate_descriptive_extension"]["preserved_gate_result"]["required_mean_gain"] == 0.02
    assert config["post_gate_descriptive_extension"]["binary_success_gate"] == "NONE"
    assert config["governance_incident"]["status"].startswith("REVIEWED_NO_FURTHER_ACTION")
    assert config["governance_incident"]["containment_verification"]["restricted_asset_key_matches_in_all_git_tracked_content"] == 0
    assert "more_than_one_accepted_synthetic_hour" in config["prohibitions"]
    assert "confirmatory_phase5" in config["prohibitions"]


def test_active_tuple_contract_is_not_confused_with_legacy_broad_calibration() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    calibration = config["calibration_C"]
    assert calibration["axes_status"].startswith("LEGACY_")
    assert calibration["joint_distributions_status"].startswith("LEGACY_")
    assert calibration["episode_plan_status"].startswith("LEGACY_PROVISIONAL_")
    assert calibration["scale_up_evidence_status"].startswith("LEGACY_")

    for pointer in calibration["active_calibration_contract_sources"].values():
        value = config
        for token in pointer.lstrip("/").split("/"):
            value = value[token]
        assert value is not None

    results = json.loads(Path("results/synthetic_video_phase4.json").read_text())
    active = results["prospective_ambitious_h3_extension"]
    assert active["amendment_commitment_sha256"] == (
        "d907d2479ba88c7e51d25e935e429cb5860e550a82620313e502482383e2855d"
    )
    assert active["order_action_counts_reused_without_substitution"] == {
        "development": 44,
        "holdout": 44,
    }
    assert active["H3_weight_download_or_inference_run"] is False
