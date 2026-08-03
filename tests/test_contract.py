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
    assert config["schema_version"] == 4
    assert config["status"] == "PHASE4_CORRECTED_ASSETS_PASS_MECHANISTIC_TRAINING_TUPLE_RUNTIME_PREP_PASS_PENDING_BLIND_SIZING_AND_PUBLIC_QUALIFICATION_PRIOR_NO_GOS_PRESERVED"
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
    runtime_preparation = config[
        "mechanistic_training_tuple_runtime_preparation_result"
    ]
    assert runtime_preparation["status"] == (
        "PASS_RUNTIME_READY_LOCAL_RELOAD_BLIND_SIZING_PENDING"
    )
    assert runtime_preparation["dependency_count"] == 53
    assert runtime_preparation["runtime_dependency_commitment_sha256"] == (
        "03c15506b0ce9fcfe403ebd735e0025dd7685d34867425b385db5463b4542c15"
    )
    assert runtime_preparation["model_inference_executed"] is False
    runtime = config["mechanistic_training_tuple_runtime_amendment"]
    assert runtime["runtime_amendment_commitment_sha256"] == (
        "623225bf24f67743e1e8990e02cebe8364191bcd17f89859c27213488ea009e4"
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
    assert config["schema_version"] == 11
    assert config["post_gate_descriptive_extension"]["preserved_gate_result"]["mean_gain"] == 0.017661900756938558
    assert config["post_gate_descriptive_extension"]["preserved_gate_result"]["required_mean_gain"] == 0.02
    assert config["post_gate_descriptive_extension"]["binary_success_gate"] == "NONE"
    assert config["governance_incident"]["status"].startswith("REVIEWED_NO_FURTHER_ACTION")
    assert config["governance_incident"]["containment_verification"]["restricted_asset_key_matches_in_all_git_tracked_content"] == 0
    assert "more_than_one_accepted_synthetic_hour" in config["prohibitions"]
    assert "confirmatory_phase5" in config["prohibitions"]
