import json
import importlib.util
import hmac
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "phase4_assets", Path("scripts/run_synthetic_video_phase4_assets.py")
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
canonical = MODULE.canonical
digest = MODULE.digest
deal_children = MODULE.deal_children


def test_phase4_asset_config_freezes_shared_assets_and_split():
    cfg = json.loads(Path("configs/synthetic_video_phase4_assets.json").read_text())
    assert cfg["allocation"]["counts"] == {"training": 28, "evaluation": 8, "validation": 4}
    assert cfg["lexical"]["styles"] == ["realistic", "cartoon"]
    assert cfg["temporal"]["candidate_count"] == 8
    assert cfg["temporal"]["frame_decode_failure"] == "exclude_complete_query_row_without_substitution"
    assert cfg["lexical"]["upstream_filter"]["min_score"] == 0.15
    assert cfg["lexical"]["upstream_filter"]["require_contrastive"] is True
    assert cfg["lexical"]["upstream_filter"]["implementation_commit"] == "224621caf0628270b6115845ac75a65b984234a3"
    assert cfg["sealing"]["all_later_arms"] == ["Real-full", "Synthetic-full", "Real-small", "Mixed"]
    assert cfg["sealing"]["test_assets_may_steer_later_work"] is False


def test_canonical_digest_is_order_independent():
    assert canonical({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_allocation_uses_literal_hmac_and_repeating_deal():
    children = [f"child-{index}" for index in range(8)]
    allocation = {"study_id_utf8": "synthetic-video", "counts": {"training": 4, "evaluation": 2, "validation": 2}, "deal_order": ["evaluation", "validation", "training"]}
    key = b"fixed-test-key"
    ranked = sorted(children, key=lambda child: hmac.digest(key, b"synthetic-video" + child.encode(), "sha256"))
    result = deal_children(children, key, allocation)
    assert result["evaluation"] == [ranked[0], ranked[3]]
    assert result["validation"] == [ranked[1], ranked[4]]
    assert result["training"] == [ranked[2], ranked[5], ranked[6], ranked[7]]


def test_language_model_preparation_is_public_cpu_only_and_offline_validated():
    batch = Path("scripts/phase4_prepare_language_models.sbatch").read_text()
    assert "#SBATCH --partition=dev" in batch
    assert "#SBATCH --gpus" not in batch
    assert "openai-whisper==20250625" in batch
    assert "1a922f3b32a8e809e17a47d4b32142d8105924e5" in batch
    assert "HF_HUB_OFFLINE=1" in batch
    assert "local_files_only=True" in batch
    assert "LANGUAGE_ENVIRONMENT_READY" in batch


def test_lexical_filter_dummy_preflight_is_public_offline_and_bounded():
    batch = Path("scripts/phase4_filter_dummy_preflight.sbatch").read_text()
    assert "#SBATCH --partition=h200" in batch
    assert "#SBATCH --gpus-per-node=1" in batch
    assert "#SBATCH --time=00:30:00" in batch
    assert "HF_HUB_OFFLINE=1" in batch
    assert "score_image_text_matrix" in batch


def test_governed_build_freezes_final_topology_and_seals_common_assets():
    batch = Path("scripts/build_synthetic_video_phase4_assets.sbatch").read_text()
    source = Path("scripts/build_synthetic_video_phase4_assets.py").read_text()
    assert "#SBATCH --partition=h100" in batch
    assert "#SBATCH --ntasks-per-node=2" in batch
    assert "#SBATCH --gpus-per-node=2" in batch
    assert "#SBATCH --time=12:00:00" in batch
    assert "HF_HUB_OFFLINE=1" in batch
    assert "imageio_ffmpeg.get_ffmpeg_exe" in batch
    assert 'ln -s "$ffmpeg_exe" "$tmp/bin/ffmpeg"' in batch
    assert 'SINGULARITYENV_PREPEND_PATH="$tmp/bin"' in batch
    assert 'dist.init_process_group("nccl"' in source
    assert "calibration_C_only" in source
    assert '"Recall@1","MRR"' in source
    assert "common_asset_references" in source
    assert "temporal_rows" in source
    assert "candidate_strata" in source
    assert "temporary.replace(target)" in source
    assert "E_CALIBRATION_EVALUATION_CHILD_OVERLAP" in source
    assert "public_provenance" in source
    assert "test_assets_may_steer_later_work" in source
    assert "score_image_text_matrix" in source
    assert "MachineDevBenchLexicalDataset" in source
    assert "official_evaluator_structural_smoke" in source


def test_phase4_seal_contract_is_identical_for_every_later_arm():
    result = json.loads(Path("results/synthetic_video_phase4.json").read_text())
    references = result["common_asset_references"]
    assert result["status"] == "CORRECTED_COMMON_ASSETS_PASS_PRIOR_CALIBRATION_NO_GOS_PRESERVED_ACTIVITY_CHECKPOINT_SELECTION_FROZEN"
    assert result["scientifically_accepted"] is True
    assert result["contract_identical_all_arms"] is True
    assert set(references) == {"Real-full", "Synthetic-full", "Real-small", "Mixed"}
    assert len({canonical(value) for value in references.values()}) == 1
    assert references["Real-full"]["lexical"] == result["lexical_commitment"]
    assert references["Real-full"]["temporal"] == result["temporal_commitment"]
    assert result["lean_pilot"]["status"].startswith("STOP_REAL_1H_AND_REAL_3H")
    assert result["coverage_redesign"]["status"] == "STOP_REAL_1H_POSITIVE_CONTROL_FAILED_MEAN_GAIN"
    assert result["coverage_redesign"]["gate_pass"] is False
    assert result["coverage_redesign"]["gate_components"]["mean_gain_at_least_0_02"] is False
    assert result["post_gate_descriptive_extension"]["preserves_coverage_redesign_stop"] is True
    assert result["post_gate_descriptive_extension"]["status"] == "PENDING_FROZEN_ACTIVITY_CHECKPOINT_SELECTION_THEN_UNCHANGED_PUBLIC_AND_C_GATES"
    assert result["post_gate_descriptive_extension"]["prior_stop_preserved"] == "STOPPED_CALIBRATION_EXTRACTOR_REDESIGN_PREMODEL_FEASIBILITY_NO_GO"
    assert result["governed_C_calibration"]["critical_axis_failure"] == "audiovisual_grounding_opportunity_high_missingness"
    assert result["governed_C_calibration"]["provisional_episode_plan_executable"] is False
    repair = result["governed_C_calibration_extractor_repair"]
    assert repair["status"] == "NO_GO_PUBLIC_QUALIFICATION"
    assert repair["unchanged_missingness_max"] == 0.2
    assert repair["original_calibration_commitment_preserved"] == result["governed_C_calibration"]["calibration_commitment_sha256"]
    assert repair["counts"]["hand_negative_correct"] == 1
    assert repair["thresholds"]["hand_negative_correct_min"] == 2
    assert repair["failed_component"] == "hand_negative_specificity"
    redesign = result["governed_C_calibration_extractor_redesign"]
    assert redesign["status"] == "NO_GO_PREMODEL_FEASIBILITY"
    assert redesign["new_model_inference_executed"] is False
    assert redesign["public_holdout_opened"] is False
    assert redesign["governed_C_reopened"] is False
    assert redesign["blocking_component"] == "EgoVLPv2_activity_context"
    assert redesign["official_checkpoint_HEAD_http_status"] == 403
    assert redesign["official_checkpoint_ranged_GET_http_status"] == 403
    assert redesign["checkpoint_sha256_resolved"] is False
    assert redesign["original_calibration_commitment_preserved"] == result["governed_C_calibration"]["calibration_commitment_sha256"]
    assert redesign["first_repair_public_commitment_preserved"] == repair["public_qualification_commitment_sha256"]
    commitment = redesign.pop("feasibility_record_commitment_sha256")
    assert digest(redesign) == commitment
    selection = result["governed_C_activity_checkpoint_selection"]
    assert selection["status"] == "FROZEN_BEFORE_EMPIRICAL_CANDIDATE_OUTCOMES"
    assert selection["prior_failure_sequence_preserved"] == [
        result["governed_C_calibration"]["calibration_commitment_sha256"],
        repair["public_qualification_commitment_sha256"],
        commitment,
    ]
    assert selection["bounded_candidate_ids"] == [
        "egohod_egovideo_l_zero_shot",
        "videoprism_lvt_l_zero_shot",
        "vjepa2_vitl_public_probe",
    ]
    assert selection["development_outcomes_opened"] is False
    assert selection["winner_selected"] is False
    assert selection["public_holdout_opened"] is False
    assert selection["governed_C_reopened"] is False
    selection_commitment = selection.pop("selection_amendment_commitment_sha256")
    assert digest(selection) == selection_commitment
    preparation = result["governed_C_activity_checkpoint_preparation"]
    assert preparation["status"] == "PASS_PREPARED_NO_MODEL_INFERENCE"
    assert preparation["dependency_manifest_commitment_sha256"] == (
        "20bc4ad80b661eba4822630f157de09a3406fad50e5bb5ba0777b838a46d0bca"
    )
    assert preparation["scientific_outcome_observed"] is False
    assert preparation["restricted_mount_present"] is False
    assert preparation["valid_for_candidate_inference"] is False
    sizing = result["governed_C_activity_checkpoint_sizing_amendment"]
    assert sizing["status"] == (
        "FROZEN_CONTROLS_RETAINED_RESTARTING_AFTER_PREINFERENCE_SAFE_LOAD_REPAIR"
    )
    assert sizing["fixture_labels_used"] is False
    assert sizing["scores_predictions_or_scientific_metrics_retained"] is False
    assert sizing["aggregate_GPU_hours_through_C_including_sizing_max"] == 20.0
    safe_load = result["governed_C_activity_checkpoint_safe_load_repair"]
    assert safe_load["status"] == "FROZEN_DYNAMIC_FLOAT64_DTYPE_REPAIR_BEFORE_MODEL_OUTPUT"
    assert safe_load["unsafe_global_count"] == 13
    assert safe_load["dynamic_safe_type_count"] == 1
    assert safe_load["weights_only_remains_true"] is True
    assert safe_load["model_inference_executed"] is False
    assert result["governance_incident"]["restricted_execution_paused"] is False


def test_coverage_redesign_is_frozen_without_rewriting_prior_stop():
    proof = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    assert proof["status"] == "ACTIVITY_CHECKPOINT_SAFE_LOAD_DEPENDENCIES_RESEALED_PRIOR_NO_GOS_PRESERVED_PENDING_BLIND_SIZING"
    assert proof["budgets_credited_hours"] == {
        "real": 1,
        "synthetic_accepted": 1,
        "synthetic_above_one_hour_prohibited": True,
    }
    assert proof["sealed_prior_570_step_pilot"]["status"] == "PRESERVED_NOT_REINTERPRETED"
    assert proof["learner"]["schedule"] == {"contrastive": 4, "mlm": 1, "dinov2": 1}
    assert proof["learner"]["initialization"].startswith("byte_identical")
    assert proof["learner"]["batch_size"] == 2
    assert proof["learner"]["objective_steps"] == 4668
    assert proof["learner"]["objective_counts"] == {"contrastive": 3112, "mlm": 778, "dinov2": 778}
    assert proof["learner"]["complete_4_1_1_cycles"] * 6 == proof["learner"]["objective_steps"]
    assert proof["learner"]["seeds"] == [436034264, 1285938051, 151347827]
    assert proof["real_1h_positive_control_gate"]["realistic_lexical_macro_seed_mean_min"] == 0.52
    assert proof["real_1h_positive_control_gate"]["mean_improvement_over_seed_matched_initialization_min"] == 0.02
    assert proof["generator_gate"]["selected"] == "LTX-2.3-22B-Distilled-1.1"
    assert proof["schema_version"] == 10
    assert proof["generator_gate"]["status"] == "SELECTED_LOCAL_NOT_RUN_ACTIVITY_CHECKPOINT_SELECTION_PENDING"
    assert proof["generator_gate"]["implementation"]["commit"] == "9377758131b1ffde4b7f766804590a6617bf2ab9"
    assert proof["generator_gate"]["weights"]["revision"] == "4229404625088d21c4f112eb640fb04a0900ee25"
    assert proof["generator_gate"]["production_ceiling_provisional_until_preflight"]["accepted_credited_seconds_exact"] == 3600
    assert proof["generator_gate"]["production_ceiling_provisional_until_preflight"]["raw_generated_seconds_max"] == 5399.625
    assert proof["generator_gate"]["production_ceiling_provisional_until_preflight"]["attempts_max"] == 1071
    assert "LTX 19/28" in proof["generator_gate"]["bakeoff_interpretation"]
    assert proof["generator_gate"]["no_substitution"] is True
    assert proof["calibration_C"]["source"] == "development_set_C_only_never_training_validation_or_evaluation"
    assert proof["calibration_C"]["local_generator_gate"] == "PENDING_FROZEN_ACTIVITY_CHECKPOINT_SELECTION_THEN_UNCHANGED_PUBLIC_AND_C_GATES"
    repair_result = proof["calibration_C"]["extractor_repair_result"]
    assert repair_result["status"] == "NO_GO"
    assert repair_result["hand_negative_correct_count"] == 1
    assert repair_result["hand_negative_correct_min"] == 2
    assert repair_result["governed_C_rerun_status"] == "NOT_RUN_PUBLIC_NO_GO"
    assert proof["calibration_C"]["governed_result"]["measured_axis_count"] == 3
    assert proof["calibration_C"]["governed_result"]["episode_plan_status"].startswith("PROVISIONAL_NOT_EXECUTABLE")
    assert "governed Juno" in proof["calibration_C"]["local_generator_input"]
    assert "no full distributional calibration claim" in proof["calibration_C"]["limitations"]
    assert len(proof["calibration_C"]["axes"]) == 8
    assert proof["calibration_C"]["joint_distributions"] == [
        "naming_by_referent_visibility",
        "naming_by_hand_object_action",
        "clutter_by_occlusion",
        "motion_by_blur",
    ]
    assert proof["calibration_C"]["omnibus_score"] == "PROHIBITED"
    assert proof["post_gate_descriptive_extension"]["preserved_gate_result"]["status"] == "FAILED_NOT_REINTERPRETED"
    assert "directionally_competitive" in proof["post_gate_descriptive_extension"]["prohibited_interpretation"]
    assert "more_than_one_accepted_synthetic_hour" in proof["prohibitions"]


def test_lean_real_preparation_uses_shared_adapter_and_exact_credit():
    source = Path("scripts/run_synthetic_video_lean_pilot.py").read_text()
    batch = Path("scripts/prepare_synthetic_video_lean_pilot.sbatch").read_text()
    assert "translate_segments" in source
    assert "frozen_language_adapter_v1" in source
    assert "credited_target = float(credited_hours * 3600)" in source
    assert "reserve_target = credited_target * 1.1" in source
    assert "choices=(1, 3)" in source
    assert "E_EXACT_CREDIT" in source
    assert "#SBATCH --partition=h100" in batch
    assert "#SBATCH --gpus-per-node=1" in batch
    assert "#SBATCH --time=04:00:00" in batch
    assert "HF_HUB_OFFLINE=1" in batch


def test_lean_learner_freezes_matched_coverage_and_seed_contract():
    source = Path("scripts/run_synthetic_video_lean_learner.py").read_text()
    batch = Path("scripts/run_synthetic_video_lean_learner.sbatch").read_text()
    assert "_load_shared_prior" in source
    assert "strict_state_equality" in source
    assert 'checkpoints/"initialized.pt"' in source
    assert "objective_steps" in source
    assert 'trainer._save(f"{arm}_seed_{args.seed}_step_{trainer.global_step}")' in source
    assert "E_UNREGISTERED_SEED" in source
    assert "E_RECORD_COUNT" in source
    assert "ssl_iterator" in source
    assert 'mode == "contrastive"' in source
    assert 'mode == "dinov2"' in source
    assert "MachineDevBenchLexicalDataset" in source
    assert "temporal_recall_at_1" in source
    assert "initialization_state_hash" in source
    assert "E_MODE_COUNTS" in source
    assert "#SBATCH --partition=h100" in batch
    assert "#SBATCH --time=06:00:00" in batch
    assert "HF_HUB_OFFLINE=1" in batch
