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


def test_phase4_qualification_wrapper_has_fail_closed_health_topology() -> None:
    wrapper = Path("scripts/qualify_synthetic_video_calibration.sbatch").read_text()
    topology_guard = wrapper.split("require_health_topology() {", 1)[1].split("\n}", 1)[0]
    assert "#SBATCH --partition=a30" in wrapper
    assert "#SBATCH --nodes=1" in wrapper
    assert "#SBATCH --ntasks=1" in wrapper
    assert "#SBATCH --gpus-per-node=1" in wrapper
    assert "#SBATCH --cpus-per-task=8" in wrapper
    assert "#SBATCH --mem-per-cpu=4G" in wrapper
    assert "#SBATCH --time=03:00:00" in wrapper
    assert wrapper.count("#SBATCH --time=03:00:00") == 1
    assert 'tuple_run_mode" == "health"' in wrapper
    assert (
        'elif [[ "$tuple_run_mode" == "health" || "$tuple_run_mode" == "development" || "$tuple_run_mode" == "holdout" ]]; then'
        in wrapper
    )
    assert "mechanistic-tuples/construct-aligned-engineering-health" in wrapper
    assert 'health_attempt="${PHASE4_HEALTH_ATTEMPT:-}"' in wrapper
    assert "8) ;;" in wrapper
    assert "7) ;;" not in wrapper
    assert "6) ;;" not in wrapper
    assert "5) ;;" not in wrapper
    assert "4) ;;" not in wrapper
    assert "1|2|3) ;;" not in wrapper
    assert 'require_health_topology' in wrapper
    assert 'scontrol show job --oneliner "$SLURM_JOB_ID"' in wrapper
    assert 'Partition=a30' in wrapper
    assert 'NumNodes=1' in wrapper
    assert 'NumCPUs=8' in wrapper
    assert 'NumTasks=1' in wrapper
    assert 'TimeLimit=01:00:00' in wrapper
    assert 'time_limit_minutes":60' in wrapper
    assert 'MinMemoryCPU=4G' in wrapper
    assert 'TresPerNode=gres/gpu:nvidia_a30:1' in wrapper
    assert 'SLURM_JOB_GPUS' not in topology_guard
    assert 'nvidia-smi' not in topology_guard
    assert '"GPU_type":"NVIDIA_A30_24GB"' in wrapper
    assert "echo " not in topology_guard
    assert topology_guard.count("printf ") == 1
    assert 'topology-attestation.json' in topology_guard
    assert '"source":"WRAPPER_SCONTROL_BEFORE_CONTAINER"' in topology_guard
    assert '"predicate_count":7,"predicate_pass_count":7' in topology_guard
    assert topology_guard.count("2>/dev/null") == 1
    assert 'python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-health' in wrapper
    assert '--attempt "$health_attempt"' in wrapper
    assert '--partition "$tuple_run_mode"' in wrapper
    assert wrapper.count("singularity exec --nv --net --network none") >= 4
    assert '--bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}"' in wrapper
    assert 'phase4-language-pydeps.tar' in wrapper
    assert 'sha256sum --check --status' in wrapper
    assert 'runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source:' in wrapper
    assert '[[ -L "$container_image" ]] || exit 65' in wrapper
    assert '[[ -f "$container_image" ]] || exit 65' in wrapper
    assert 'container-attestation.json' in wrapper
    assert '"source":"WRAPPER_HOST_BEFORE_CONTAINER"' in wrapper
    assert '--container-attestation "$container_attestation"' in wrapper


def test_config_hash_is_order_independent() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})


def test_lexical_wiring_requires_noun_then_adjective() -> None:
    result = lexical_macro_wiring({"noun": [1, 0], "adjective": [0, 1]})
    assert result == {"noun": 0.5, "adjective": 0.5, "lexical_macro": 0.5}
    with pytest.raises(ValueError, match="ordered exactly"):
        lexical_macro_wiring({"adjective": [0, 1], "noun": [1, 0]})


def test_phase4_preregistration_preserves_frozen_contract() -> None:
    config = json.loads(Path("configs/synthetic_video_preregistration.json").read_text())
    assert config["schema_version"] == 21
    assert config["status"] == "PHASE4_CORRECTED_ASSETS_PASS_LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_7_TIMEOUT_PRESERVED_ZERO_RUNTIME_H100_ATTEMPT_8_CANCELED_A30_TOPOLOGY_FROZEN_PENDING_ATTEMPT_8_NO_NEW_OUTCOME"
    validate_phase_state(config)
    geometry_repair = config["public_fixture_geometry_rasterization_repair"]
    assert geometry_repair["fixture_schema_version"] == 3
    assert geometry_repair["declared_mask_dimensions"] == (
        "exact_positive_non_boolean_integers_required"
    )
    assert geometry_repair["repair_commitment_sha256"] == (
        "6084fd937c208feda00aa3dc1cf14d0ec56e8f13bd24b56e23e4a6a6553e61ef"
    )
    fixture_result = config["learner_effective_public_fixture_preparation_result"]
    assert fixture_result["status"] == "PASS_PUBLIC_FIXTURES_SEALED_NO_MODEL_INFERENCE"
    assert fixture_result["job_id"] == 315501
    assert fixture_result["public_fixture_manifest_commitment_sha256"] == (
        "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
    )
    assert fixture_result["source_overlap_counts"] == [0, 0, 0, 0]
    assert fixture_result["model_inference_executed"] is False
    assert fixture_result["public_development_authorized"] is True
    resource = config["learner_effective_public_qualification_resource_amendment"]
    assert resource["triggering_H100_job_id"] == 315509
    assert resource["triggering_H100_elapsed_seconds"] == 0
    assert resource["qualification_output_count"] == 0
    assert resource["active_GPU_type"] == "NVIDIA_A30_24GB"
    assert resource["development_and_holdout_GPU_hours_max"] == 6.0
    assert resource["amendment_commitment_sha256"] == (
        "5330cf582e46d1bf075ca97af7c8bfceb47cfcd09499786f4e366b6f8e283beb"
    )
    development = config["learner_effective_public_development_result"]
    assert development["status"] == "NO_GO_DEVELOPMENT_COMBINED_GATE"
    assert development["job_id"] == 315542
    assert development["module_counts"] == [7, 2, 5]
    assert development["critical_axis_pass_count"] == 1
    assert development["validated_axis_count"] == 2
    assert development["unaccounted_failure_count"] == 3
    assert development["holdout_authorized"] is False
    assert development["holdout_result_present"] is False
    assert development["public_qualification_commitment_sha256"] == (
        "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
    )
    proof = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    result = json.loads(Path("results/synthetic_video_phase4.json").read_text())
    assert proof["learner_effective_public_development_result"] == result[
        "learner_effective_public_development_result"
    ]
    assert canonical_json_sha256(
        proof["learner_effective_public_development_result"]
    ) == "c43c7a678e3a2eac10ed5a5ac75c8964520931ec180ab9306585c76d198fb8c8"
    health = config["learner_effective_engineering_health_amendment"]
    assert health["amendment_commitment_sha256"] == (
        "d447a7e165136032a1fba43605d3f81881b41ec030c82e9028e1a8f5cb2c6205"
    )
    assert health["microfixture_case_count"] == 28
    assert health["scientific_metric_count"] == 0
    assert health["submission_count_max"] == 3
    assert health["aggregate_GPU_hours_max"] == 0.75
    assert health["new_engineering_outcome_opened"] is False
    assert health["new_scientific_outcome_opened"] is False
    redirect = config["learner_effective_engineering_health_resource_redirect"]
    assert redirect["amendment_commitment_sha256"] == (
        "f7fc16f5c399c2a2d213b13a0d255a14b5b2f3ece41d62adaed17f61f186db6d"
    )
    assert redirect["canceled_A30_job_id"] == 316158
    assert redirect["canceled_A30_elapsed_seconds"] == 0
    assert redirect["canceled_A30_qualification_output_count"] == 0
    assert redirect["active_partition"] == "h100"
    assert redirect["active_GRES"] == "gpu:nvidia_h100_nvl_3g.47gb:1"
    assert redirect["active_GPU_type"] == "NVIDIA_H100_NVL_3G_47GB_MIG"
    assert redirect["scientific_contract_changed"] is False
    assert redirect == result["learner_effective_engineering_health_resource_redirect"]
    scheduler = config["learner_effective_engineering_health_scheduler_policy"]
    assert scheduler["amendment_commitment_sha256"] == (
        "2cd0b824e91b8bf228d06aae240f16e70f8ffc03fb2f204518f8ce5eeeab3fba"
    )
    assert scheduler["canceled_job_id"] == 316697
    assert scheduler["selected_partition"] == "a30"
    assert scheduler["selected_GPU_type"] == "NVIDIA_A30_24GB"
    assert scheduler == result[
        "learner_effective_engineering_health_scheduler_policy"
    ]
    restore = config["learner_effective_engineering_health_dependency_restore"]
    assert restore["repair_commitment_sha256"] == (
        "3c54503b4087fae1e993b0aa952823f988a088a5c6543760df1022e2dc046db4"
    )
    assert restore["trigger_job_id"] == 316325
    assert restore["trigger_model_module_inference_count"] == 0
    assert restore["trigger_scientific_metric_count"] == 0
    assert restore["classification"] == (
        "ENGINEERING_DEPENDENCY_CACHE_MISS_NOT_SCIENTIFIC_NO_GO"
    )
    assert restore["active_language_archive_sha256"] == (
        "97ef52ecaa8c99db017e598d8a63d0d2170affef14ef46e7df7a656abd3a1a07"
    )
    assert restore["offline_local_files_only_reload"] == "PASS"
    assert restore["submission_count_remaining"] == 2
    assert restore["new_scientific_outcome_opened"] is False
    assert restore == result["learner_effective_engineering_health_dependency_restore"]
    topology_repair = config[
        "learner_effective_engineering_health_topology_guard_repair"
    ]
    assert topology_repair["repair_commitment_sha256"] == (
        "8db2d8ae04ee702ab3c68ff7c243afed0d8e4710c01f3f7865faa4975fb9a5b8"
    )
    assert topology_repair["trigger_job_id"] == 316353
    assert topology_repair["trigger_model_module_inference_count"] == 0
    assert topology_repair["trigger_scientific_metric_count"] == 0
    assert topology_repair["authoritative_scontrol_predicate_pass_count"] == 7
    assert topology_repair["health_topology_changed"] is False
    assert topology_repair["submission_count_remaining"] == 1
    assert topology_repair == result[
        "learner_effective_engineering_health_topology_guard_repair"
    ]
    blocker = config["learner_effective_engineering_health_result"]
    assert blocker["status"].startswith("ENGINEERING_BLOCKER_ROUTE_EXHAUSTED")
    assert blocker["job_id"] == 316370
    assert blocker["module_counts"] == [7, 0, 7]
    assert blocker["scientific_metric_count"] == 0
    assert blocker["unaccounted_failure_count"] == 1
    assert blocker["private_trace_count"] == 7
    assert blocker["declared_preflight_blocked_trace_count"] == 6
    assert blocker["attempt_4_authorized"] is False
    assert blocker["public_development_authorized"] is False
    assert blocker["governed_C_authorized"] is False
    assert blocker["LTX_or_synthetic_learner_run"] is False
    assert blocker["blocker_commitment_sha256"] == (
        "644028babc768e881276fa078b95349ba77f8418cb76d722e8baf2588f9d0f81"
    )
    assert blocker == result["learner_effective_engineering_health_result"]
    reauthorization = config[
        "learner_effective_engineering_health_reauthorization"
    ]
    assert reauthorization["reauthorization_commitment_sha256"] == (
        "3271499c19a77ffab8c53e2cd2052ea14514682a3d4b73fd7c5179ceec4a7ff4"
    )
    assert reauthorization["sealed_engineering_blocker_sha256"] == (
        "644028babc768e881276fa078b95349ba77f8418cb76d722e8baf2588f9d0f81"
    )
    assert reauthorization["attempt"] == 4
    assert reauthorization["submission_count"] == 1
    assert reauthorization["reauthorized_GPU_hours_max"] == 0.25
    assert reauthorization["runner_scontrol_inside_container"] is False
    assert reauthorization["model_fixture_threshold_metric_seed_or_gate_changed"] is False
    assert reauthorization == result[
        "learner_effective_engineering_health_reauthorization"
    ]
    terminal_reauthorization = config[
        "learner_effective_engineering_health_reauthorization_result"
    ]
    assert terminal_reauthorization == result[
        "learner_effective_engineering_health_reauthorization_result"
    ]
    assert terminal_reauthorization["job_id"] == 316478
    assert terminal_reauthorization["topology_predicate_counts"] == [7, 7]
    assert terminal_reauthorization["model_module_inference_count"] == 0
    assert terminal_reauthorization["scientific_metric_count"] == 0
    assert terminal_reauthorization["attempt_5_authorized"] is False
    assert terminal_reauthorization["blocker_commitment_sha256"] == (
        "59b1778b35cedd1cb020177e41fe6887371a5480f7ee6bf6e57f55d4c90edde3"
    )
    parser_repair = config[
        "learner_effective_engineering_health_parser_repair_reauthorization"
    ]
    assert parser_repair == result[
        "learner_effective_engineering_health_parser_repair_reauthorization"
    ]
    assert parser_repair["reauthorization_commitment_sha256"] == (
        "d9cf3feaa0f5c4d65978ca796b722f31e75ce1078b918d114ca35b298a148c8b"
    )
    assert parser_repair["attempt"] == 5
    assert parser_repair["attempt_4_remains_rejected_and_sealed"] is True
    parser_result = config[
        "learner_effective_engineering_health_parser_repair_result"
    ]
    assert parser_result == result[
        "learner_effective_engineering_health_parser_repair_result"
    ]
    assert parser_result["blocker_commitment_sha256"] == (
        "b05dc8da3155561b182b3bcfa50c851f83828b34e063918306bdfb57fdedeb9c"
    )
    assert parser_result["attempt"] == 5
    assert parser_result["scientific_metric_count"] == 0
    assert parser_result["attempt_6_authorized"] is False
    attempt_6 = config[
        "learner_effective_engineering_health_iterative_attempt_6_result"
    ]
    assert attempt_6 == result[
        "learner_effective_engineering_health_iterative_attempt_6_result"
    ]
    assert attempt_6["blocker_commitment_sha256"] == (
        "e559cd535d2a6dd833d2588c75b180754260dd3ff68ea9f6731a0e4478a6d114"
    )
    assert attempt_6["scientific_metric_count"] == 0
    progress = config["learner_effective_engineering_health_progress_repair"]
    assert progress == result[
        "learner_effective_engineering_health_progress_repair"
    ]
    assert progress["reauthorization_commitment_sha256"] == (
        "a2d1347bef14848a5238f9a10c6e94da8eaa68593aa3d97e8a460dfbf8694d07"
    )
    assert progress["active_attempt"] == 7
    attempt_7 = config["learner_effective_engineering_health_attempt_7_result"]
    assert attempt_7 == result[
        "learner_effective_engineering_health_attempt_7_result"
    ]
    assert attempt_7["blocker_commitment_sha256"] == (
        "03c09a61cedb29e04cf465287db693cd1248c53d45d9a7a47a777e6cdf1d594d"
    )
    assert attempt_7["scientific_metric_count"] == 0
    extended = config[
        "learner_effective_engineering_health_extended_wall_repair"
    ]
    assert extended == result[
        "learner_effective_engineering_health_extended_wall_repair"
    ]
    assert extended["reauthorization_commitment_sha256"] == (
        "d2db51229719a0e64f84da9541a284d88c75a2fd32e2e186ba34e17ab5eed6e7"
    )
    assert extended["active_attempt"] == 8
    assert extended["wall_minutes_max"] == 60
    assert progress["model_fixture_threshold_metric_seed_or_gate_changed"] is False
    label_correction = config["public_no_hand_review_label_semantics_correction"]
    assert label_correction["coded_item_count_at_correction"] == 195
    assert label_correction["binary_yes_no_swapped_count"] == 193
    assert label_correction["abstention_unchanged_count"] == 2
    assert label_correction["before_label_record_sha256"] == (
        "7dd640e1bc8e02fe7ced62d1ad7cbe5ccb9e2503b1877840ccb9b093b54e4fb5"
    )
    assert label_correction["after_label_record_sha256"] == (
        "2ffae4d120fdda4d995cb9d9056655d83082d05607d1860ef75199ea7109eae8"
    )
    assert label_correction["review_sealed"] is False
    assert label_correction["model_inference_executed"] is False
    assert label_correction["public_model_or_scientific_gate_outcome_opened"] is False
    review_seal = config["public_no_hand_review_seal_result"]
    assert review_seal["status"].startswith("PASS_BLIND_NO_HAND_REVIEW_SEALED")
    assert review_seal["job_id"] == 315425
    assert review_seal["coded_count"] == 251
    assert review_seal["verified_no_hand_count"] == 96
    assert review_seal["deficit_partition_count"] == 0
    assert review_seal["verified_no_hand_seal_commitment_sha256"] == (
        "a58ca3f10fd72ba2a7bfc2faf9c8c65b22a22913fcf2c92786859401b8d21c97"
    )
    assert review_seal["fixture_preparation_authorized"] is True
    assert review_seal["model_inference_executed"] is False
    fixture_repair = config["public_fixture_preparation_engineering_repair"]
    assert fixture_repair["status"].startswith("PASS_PINNED_FFMPEG_PORTABILITY_DIAGNOSIS")
    assert fixture_repair["runtime_manifest_failure"]["job_id"] == 315430
    assert fixture_repair["video_encode_failure"]["job_id"] == 315445
    assert fixture_repair["current_runner_source_sha256"] == (
        "1897c40be4f19c54e18042bc97cedf4c895447963577c670d0f48947a7066768"
    )
    assert fixture_repair["suffix_fix_retry_failure"]["job_id"] == 315452
    assert fixture_repair["diagnostic_jobs"][-1] == {
        "job_id": 315458,
        "status": "PASS_PINNED_BINARY_SMOKE",
    }
    assert fixture_repair["portable_filter_mapping"] == {
        "silent_source": "anullsrc=r=22050:cl=mono,atrim=duration=<d>",
        "speech_delay": "adelay=2500",
        "unsupported_options_prohibited": ["d=", "all="],
    }
    assert fixture_repair["smoke"]["timing_within_one_AAC_frame"] is True
    assert fixture_repair["audit_status"] == "PASS"
    assert fixture_repair["fixture_development_threshold_holdout_outputs_present"] is False
    assert fixture_repair["model_inference_executed"] is False
    assert fixture_repair["fixture_retry_authorized"] is True
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
    assert config["gates"]["learner_effective_public_qualification_authorized"] is True
    assert config["gates"]["learner_effective_runner_implementation_status"] == (
        "ATTEMPT_7_TIMEOUT_PRESERVED_ZERO_RUNTIME_H100_ATTEMPT_8_CANCELED_A30_TOPOLOGY_FROZEN_BEFORE_ATTEMPT_8_OR_NEW_OUTCOME"
    )
    assert config["gates"]["public_model_inference_requires_blind_no_hand_review_seal"] is True
    assert config["gates"]["learner_effective_no_hand_review_authorized"] is True
    assert config["gates"]["learner_effective_no_hand_review_sealed"] is True
    assert config["gates"]["learner_effective_public_fixture_preparation_authorized"] is True
    assert config["gates"]["learner_effective_public_model_inference_authorized"] is True
    assert config["gates"]["learner_effective_public_model_inference_scope"] == (
        "ATTEMPT_8_EXTENDED_WALL_ENGINEERING_MICROHEALTH_ONLY_NO_SCIENTIFIC_METRICS"
    )
    assert config["gates"][
        "learner_effective_public_model_inference_conditionally_authorized_after_fixture_seal"
    ] is True
    assert config["gates"]["learner_effective_new_route_scientific_development_authorized"] is False
    assert config["gates"]["governed_C_calibration_authorized"] is False
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
    assert config["schema_version"] == 30
    health = dict(config["learner_effective_engineering_health_amendment"])
    health_commitment = health.pop("amendment_commitment_sha256")
    assert health_commitment == canonical_json_sha256(health)
    assert health_commitment == (
        "d447a7e165136032a1fba43605d3f81881b41ec030c82e9028e1a8f5cb2c6205"
    )
    assert health["engineering_microfixture_suite"]["total_case_count"] == 28
    assert health["metric_withholding"]["microhealth_scientific_metric_count"] == 0
    assert health["scientific_threshold_state"]["DINOv2_recurrence_cosine"] == 0.85
    assert health["scientific_threshold_state"]["threshold_change_or_relaxation"] is False
    assert health["bounded_resource_policy"]["per_submission_wall_minutes_max"] == 15
    extended = dict(
        config["learner_effective_engineering_health_extended_wall_repair"]
    )
    extended_commitment = extended.pop("reauthorization_commitment_sha256")
    assert extended_commitment == canonical_json_sha256(extended)
    assert extended_commitment == (
        "d2db51229719a0e64f84da9541a284d88c75a2fd32e2e186ba34e17ab5eed6e7"
    )
    assert extended["active_attempt_resource_policy"]["attempt"] == 8
    assert extended["active_attempt_resource_policy"]["wall_minutes_max"] == 60
    scheduler = dict(
        config["learner_effective_engineering_health_scheduler_policy"]
    )
    scheduler_commitment = scheduler.pop("amendment_commitment_sha256")
    assert scheduler_commitment == canonical_json_sha256(scheduler)
    assert scheduler_commitment == (
        "2cd0b824e91b8bf228d06aae240f16e70f8ffc03fb2f204518f8ce5eeeab3fba"
    )
    assert scheduler["canceled_submission"]["job_id"] == 316697
    assert scheduler["canceled_submission"]["elapsed_seconds"] == 0
    assert scheduler["active_attempt_resource_policy"]["GPU_type"] == (
        "NVIDIA_A30_24GB"
    )
    redirect = dict(config["learner_effective_engineering_health_resource_redirect"])
    redirect_commitment = redirect.pop("amendment_commitment_sha256")
    assert redirect_commitment == canonical_json_sha256(redirect)
    assert redirect_commitment == (
        "f7fc16f5c399c2a2d213b13a0d255a14b5b2f3ece41d62adaed17f61f186db6d"
    )
    assert redirect["canceled_A30_submission"]["elapsed_seconds"] == 0
    assert redirect["active_health_topology"] == {
        "partition": "h100",
        "nodes": 1,
        "tasks": 1,
        "GRES": "gpu:nvidia_h100_nvl_3g.47gb:1",
        "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
        "GPU_count": 1,
        "process_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "DDP": False,
        "per_submission_wall_minutes_max": 15,
    }
    resource = dict(config["learner_effective_public_qualification_resource_amendment"])
    commitment = resource.pop("amendment_commitment_sha256")
    assert commitment == canonical_json_sha256(resource)
    assert commitment == "5330cf582e46d1bf075ca97af7c8bfceb47cfcd09499786f4e366b6f8e283beb"
    assert resource["canonical_wrapper_sha256"] == (
        "2e4d6d7601aa35a85c6e9a12648bcf6319c54741b5cfc2c9808c797972988b03"
    )
    amendment = dict(config["ambitious_learner_effective_h3_amendment"])
    commitment = amendment.pop("amendment_commitment_sha256")
    assert commitment == canonical_json_sha256(amendment)
    assert commitment == "d907d2479ba88c7e51d25e935e429cb5860e550a82620313e502482383e2855d"
    assert amendment["order_action_role_amendment"]["new_role"] == (
        "SUPPORTING_DIAGNOSTIC_NONBLOCKING"
    )
    assert amendment["hierarchy"]["public_and_governed_C_combined_pass_rule"].startswith(
        "all five critical axes pass"
    )
    active = dict(config["construct_aligned_ltx_resume_amendment"])
    active_commitment = active.pop("amendment_commitment_sha256")
    assert active_commitment == canonical_json_sha256(active)
    assert active_commitment == (
        "842d5a16141d8b0a6bdc82d86fb405bcbb14bbbd4e6cfe645ffae328ad881a39"
    )
    assert active["combined_gate"]["order_action"].startswith(
        "SUPPORTING_DIAGNOSTIC_NONBLOCKING"
    )
    assert active["generator"]["selected"] == "LTX-2.3-22B-Distilled-1.1"
    generator_amendment = dict(config["ltx_sole_generator_prompt_compiler_amendment"])
    generator_commitment = generator_amendment.pop("amendment_commitment_sha256")
    assert generator_commitment == canonical_json_sha256(generator_amendment)
    assert generator_commitment == (
        "cb4a7cd2dd60fb14f3430aaf2544d79db95064d42a3f22ad6107206832319c62"
    )
    assert generator_amendment["generator_selection"]["role"].startswith(
        "SOLE_SELECTED_GENERATOR"
    )
    assert generator_amendment["compiler"]["fixed_frames"] == 121
    runner = config["construct_aligned_public_runner_implementation"]
    assert runner["status"].startswith("PASS_")
    assert runner["exact_source_reuse_guard"] is True
    assert runner["exact_action_fixture_counts"] == {
        "development": 44,
        "holdout": 44,
    }
    assert runner["action_performance_is_nonblocking_and_nonrescuing"] is True
    assert runner["action_integrity_privacy_provenance_and_external_call_failures_remain_blocking"] is True
    assert runner["public_model_or_extractor_outcome_opened"] is False
    assert runner["implementation_source_sha256"] == (
        "1897c40be4f19c54e18042bc97cedf4c895447963577c670d0f48947a7066768"
    )
    repair = config["public_no_hand_preparation_engineering_repair"]
    assert repair["failure_class"] == "ENGINEERING_FAILURE_NOT_SCIENTIFIC_NO_GO"
    assert repair["model_inference_executed"] is False
    assert repair["scientific_threshold_or_fixture_changed"] is False
    assert repair["archive_download_worker_count"] == 4
    prepared = config["public_no_hand_review_preparation_result"]
    assert prepared["status"] == "READY_FOR_AUTHORIZED_APPLICANT_BLIND_REVIEW"
    assert prepared["review_queue_count"] == 384
    assert prepared["contact_sheet_count"] == 48
    assert prepared["decode_failure_count"] == 0
    assert prepared["within_CPU_wall_ceiling"] is True
    assert prepared["model_inference_executed"] is False
    correction = config["public_no_hand_review_label_semantics_correction"]
    assert correction["status"].startswith(
        "USER_DIRECTED_PRE_SEAL_LABEL_SEMANTICS_CORRECTION_APPLIED"
    )
    assert correction["review_queue_commitment_sha256"] == (
        "5ba8ae3eaa1d7189aac0a9435f1d10200dfbdfa4555125ed32867bb22c82f2b7"
    )
    assert correction["coded_item_count_at_correction"] == 195
    assert correction["binary_yes_no_swapped_count"] == 193
    assert correction["abstention_unchanged_count"] == 2
    assert correction["before_partition_counts"] == {
        "development": {"yes": 6, "no": 184, "abstain": 2},
        "holdout": {"yes": 0, "no": 3, "abstain": 0},
    }
    assert correction["after_partition_counts"] == {
        "development": {"yes": 184, "no": 6, "abstain": 2},
        "holdout": {"yes": 3, "no": 0, "abstain": 0},
    }
    assert correction["before_label_record_sha256"] == (
        "7dd640e1bc8e02fe7ced62d1ad7cbe5ccb9e2503b1877840ccb9b093b54e4fb5"
    )
    assert correction["after_label_record_sha256"] == (
        "2ffae4d120fdda4d995cb9d9056655d83082d05607d1860ef75199ea7109eae8"
    )
    assert correction["review_sealed"] is False
    assert correction["model_inference_executed"] is False
    assert correction["public_model_or_scientific_gate_outcome_opened"] is False
    review_seal = config["public_no_hand_review_seal_result"]
    assert review_seal["status"].startswith("PASS_BLIND_NO_HAND_REVIEW_SEALED")
    assert review_seal["job_id"] == 315425
    assert review_seal["exit_code"] == "0:0"
    assert review_seal["elapsed_seconds"] == 13
    assert review_seal["allocated_CPU_count"] == 4
    assert review_seal["allocated_memory_GiB"] == 16
    assert review_seal["direct_monetary_cost_usd"] == 0
    assert review_seal["partition_count"] == 2
    assert review_seal["coded_count"] == 251
    assert review_seal["verified_no_hand_count"] == 96
    assert review_seal["visible_hand_count"] == 15
    assert review_seal["abstain_count"] == 2
    assert review_seal["unreviewed_count"] == 133
    assert review_seal["deficit_partition_count"] == 0
    assert review_seal["review_labels_commitment_sha256"] == (
        "723f218b3f3d06189949728d93bc04114bd54c9d405910bc31b0f1653f121edd"
    )
    assert review_seal["verified_no_hand_seal_commitment_sha256"] == (
        "a58ca3f10fd72ba2a7bfc2faf9c8c65b22a22913fcf2c92786859401b8d21c97"
    )
    assert review_seal["stderr_empty"] is True
    assert all(review_seal["attestations"].values())
    assert review_seal["fixture_preparation_authorized"] is True
    assert review_seal["fixture_preparation_executed"] is False
    assert review_seal["model_inference_executed"] is False
    assert review_seal["public_development_or_holdout_outcome_opened"] is False
    fixture_repair = config["public_fixture_preparation_engineering_repair"]
    assert fixture_repair["runtime_manifest_failure"] == {
        "job_id": 315430,
        "exit_code": "1:0",
        "elapsed_seconds": 5,
        "failure": "CURRENT_PUBLIC_ROOT_LACKED_THE_EARLIER_SEALED_RUNTIME_MANIFEST",
        "old_runtime_job": 313924,
        "current_root_job": 314974,
        "roots_distinct": True,
        "failed_before_fixture_manifest_or_model_inference": True,
    }
    assert fixture_repair["runtime_lineage_copy"]["copied_public_families"] == [
        "mechanistic-tuples",
        "activity-code",
        "activity-pydeps",
    ]
    assert fixture_repair["runtime_lineage_copy"]["copied_bytes"] == 23438221791
    assert fixture_repair["runtime_lineage_copy"]["copied_GiB"] == 21.828545
    assert fixture_repair["runtime_lineage_copy"]["free_GiB_after_copy"] == 535.877
    assert fixture_repair["runtime_lineage_copy"]["within_storage_ceiling"] is True
    assert fixture_repair["video_encode_failure"]["job_id"] == 315445
    assert fixture_repair["video_encode_failure"]["failure"] == (
        "E_TUPLE_FIXTURE_VIDEO_ENCODE"
    )
    assert fixture_repair["video_encode_failure"]["canonical_fix"].endswith(
        ".mp4.partial to .partial.mp4"
    )
    assert fixture_repair["suffix_fix_retry_failure"] == {
        "job_id": 315452,
        "exit_code": "1:0",
        "elapsed_seconds": 11,
        "failure": "E_TUPLE_FIXTURE_VIDEO_ENCODE",
        "suffix_fix_present": True,
        "failed_before_fixture_manifest_or_model_inference": True,
    }
    assert fixture_repair["audio_seed_inventory"] == {
        "file_count": 112,
        "channels": 1,
        "channel_layout": "mono",
        "sample_rate_hz": 22050,
    }
    smoke = fixture_repair["pinned_binary_smoke"]
    assert smoke["job_id"] == 315458
    assert smoke["silent_exit_code"] == smoke["speech_exit_code"] == 0
    assert smoke["decoded_duration_seconds"] == 7.012426
    assert smoke["frame_count"] == 63
    assert smoke["frames_per_second"] == 9
    assert smoke["silent_active_sample_count"] == 0
    assert smoke["speech_source_active_sample_range"] == [14, 21035]
    assert smoke["speech_output_active_sample_range"] == [55198, 76134]
    assert smoke["speech_timing_error_samples"] == {"start": 59, "end": -26}
    assert smoke["AAC_frame_samples"] == 1024
    assert smoke["timing_errors_within_one_AAC_frame"] is True
    assert fixture_repair["current_fixture_output_present"] is False
    assert fixture_repair["current_development_output_present"] is False
    assert fixture_repair["current_threshold_output_present"] is False
    assert fixture_repair["current_holdout_output_present"] is False
    assert fixture_repair["model_inference_executed"] is False
    assert fixture_repair["deterministic_retry_safe"] is True
    assert fixture_repair["no_further_engineering_diagnostic_required"] is True
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
    historical = results["prospective_ambitious_h3_extension"]
    assert historical["amendment_commitment_sha256"] == (
        "d907d2479ba88c7e51d25e935e429cb5860e550a82620313e502482383e2855d"
    )
    assert historical["order_action_counts_reused_without_substitution"] == {
        "development": 44,
        "holdout": 44,
    }
    assert historical["H3_weight_download_or_inference_run"] is False
    active = results["prospective_construct_aligned_ltx_resume_extension"]
    assert active["amendment_commitment_sha256"] == (
        "842d5a16141d8b0a6bdc82d86fb405bcbb14bbbd4e6cfe645ffae328ad881a39"
    )
    assert active["selected_generator"] == "LTX-2.3-22B-Distilled-1.1"
    generator = results["prospective_ltx_sole_generator_prompt_compiler_extension"]
    assert generator["amendment_commitment_sha256"] == (
        "cb4a7cd2dd60fb14f3430aaf2544d79db95064d42a3f22ad6107206832319c62"
    )
    assert generator["generator_role"].startswith("SOLE_SELECTED_GENERATOR")
