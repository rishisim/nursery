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
    assert result["schema_version"] == 19
    assert result["status"] == "CORRECTED_COMMON_ASSETS_PASS_LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_8_GIT_ABSENCE_BLOCKER_PRESERVED_CLEAN_TREE_FALLBACK_ATTEMPT_9_FROZEN_NO_NEW_OUTCOME"
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
    assert result["post_gate_descriptive_extension"]["status"] == "STOPPED_COMPLETE_PUBLIC_SOURCE_FEASIBILITY_NO_GO"
    assert result["post_gate_descriptive_extension"]["prior_stop_preserved"] == "REAL_1H_FORMAL_GATE_AND_ALL_FOUR_CALIBRATION_NO_GOS_PRESERVED"
    assert result["prospective_ambitious_h3_extension"]["all_prior_calibration_no_gos_preserved"] is True
    assert result["prospective_ambitious_h3_extension"]["H3_weight_download_or_inference_run"] is False
    assert result["prospective_construct_aligned_ltx_resume_extension"]["all_prior_calibration_no_gos_preserved"] is True
    assert result["prospective_construct_aligned_ltx_resume_extension"]["LTX_preflight_or_generation_run"] is False
    assert result["prospective_ltx_sole_generator_prompt_compiler_extension"]["H3_role"].startswith("HISTORY_ONLY_OUT_OF_SCOPE")
    health = result["learner_effective_engineering_health_amendment"]
    assert health["amendment_commitment_sha256"] == (
        "d447a7e165136032a1fba43605d3f81881b41ec030c82e9028e1a8f5cb2c6205"
    )
    assert health["prior_public_development_no_go_commitment_sha256"] == (
        "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
    )
    scheduler = result["learner_effective_engineering_health_scheduler_policy"]
    assert scheduler["amendment_commitment_sha256"] == (
        "8ef8e53c2754fe13b91518c02f419a1d3c4f3162aa18648f7044986854d327d6"
    )
    assert scheduler["canceled_job_id"] == 316697
    assert scheduler["canceled_elapsed_seconds"] == 0
    assert scheduler["selected_GPU_type"] == "NVIDIA_H100_NVL"
    assert scheduler["new_health_or_scientific_outcome_opened"] is False
    attempt_8 = result["learner_effective_engineering_health_attempt_8_result"]
    assert attempt_8["job_id"] == 316777
    assert attempt_8["scientific_metric_count"] == 0
    assert attempt_8["blocker_commitment_sha256"] == (
        "409c36d2c3ba4aefdd2f510c661ba363c000fc232dce2fcfba0151ba25f9aad7"
    )
    git_repair = result[
        "learner_effective_engineering_health_git_fallback_repair"
    ]
    assert git_repair["active_attempt"] == 9
    assert git_repair["repair_commitment_sha256"] == (
        "b6a93e3a0b0b716d8bdd8fdd47656e69f8ff5b66c0a3ec8f96973e565a9066f9"
    )
    assert health["new_engineering_outcome_opened"] is False
    assert health["new_scientific_outcome_opened"] is False
    assert health["LTX_or_synthetic_learner_run"] is False
    blocker = result["learner_effective_engineering_health_result"]
    assert blocker["job_id"] == 316370
    assert blocker["module_counts"] == [7, 0, 7]
    assert blocker["scientific_metric_count"] == 0
    assert blocker["attempt_4_authorized"] is False
    assert blocker["public_development_authorized"] is False
    assert blocker["governed_C_authorized"] is False
    assert blocker["LTX_or_synthetic_learner_run"] is False
    reauthorization = result[
        "learner_effective_engineering_health_reauthorization"
    ]
    assert reauthorization["attempt"] == 4
    assert reauthorization["submission_count"] == 1
    assert reauthorization["sealed_engineering_blocker_sha256"] == (
        blocker["blocker_commitment_sha256"]
    )
    assert reauthorization["new_health_or_scientific_outcome_opened"] is False
    terminal_reauthorization = result[
        "learner_effective_engineering_health_reauthorization_result"
    ]
    assert terminal_reauthorization["job_id"] == 316478
    assert terminal_reauthorization["attempt"] == 4
    assert terminal_reauthorization["full_result_count"] == 0
    assert terminal_reauthorization["scientific_metric_count"] == 0
    assert terminal_reauthorization["attempt_5_authorized"] is False
    parser_repair = result[
        "learner_effective_engineering_health_parser_repair_reauthorization"
    ]
    assert parser_repair["attempt"] == 5
    assert parser_repair["attempt_4_blocker_sha256"] == (
        terminal_reauthorization["blocker_commitment_sha256"]
    )
    parser_result = result[
        "learner_effective_engineering_health_parser_repair_result"
    ]
    assert parser_result["job_id"] == 316537
    assert parser_result["module_count"] == 7
    assert parser_result["completed_module_count"] == 0
    assert parser_result["scientific_metric_count"] == 0
    assert parser_result["blocker_commitment_sha256"] == (
        "b05dc8da3155561b182b3bcfa50c851f83828b34e063918306bdfb57fdedeb9c"
    )
    assert parser_result["attempt_6_authorized"] is False
    attempt_6 = result[
        "learner_effective_engineering_health_iterative_attempt_6_result"
    ]
    assert attempt_6["job_id"] == 316604
    assert attempt_6["scheduler_state"] == "TIMEOUT"
    assert attempt_6["scientific_metric_count"] == 0
    assert attempt_6["blocker_commitment_sha256"] == (
        "e559cd535d2a6dd833d2588c75b180754260dd3ff68ea9f6731a0e4478a6d114"
    )
    progress = result["learner_effective_engineering_health_progress_repair"]
    assert progress["active_attempt"] == 7
    assert progress["reauthorization_commitment_sha256"] == (
        "a2d1347bef14848a5238f9a10c6e94da8eaa68593aa3d97e8a460dfbf8694d07"
    )
    attempt_7 = result["learner_effective_engineering_health_attempt_7_result"]
    assert attempt_7["job_id"] == 316641
    assert attempt_7["scientific_metric_count"] == 0
    assert attempt_7["blocker_commitment_sha256"] == (
        "03c09a61cedb29e04cf465287db693cd1248c53d45d9a7a47a777e6cdf1d594d"
    )
    extended = result[
        "learner_effective_engineering_health_extended_wall_repair"
    ]
    assert extended["active_attempt"] == 8
    assert extended["wall_minutes_max"] == 60
    assert extended["reauthorization_commitment_sha256"] == (
        "d2db51229719a0e64f84da9541a284d88c75a2fd32e2e186ba34e17ab5eed6e7"
    )
    runner = result["construct_aligned_public_runner_implementation_result"]
    assert runner["status"].startswith("PASS_")
    assert runner["implementation_source_sha256"] == (
        "1897c40be4f19c54e18042bc97cedf4c895447963577c670d0f48947a7066768"
    )
    assert runner["action_performance_role"] == "NONBLOCKING_NONRESCUING_DIAGNOSTIC"
    assert runner["action_integrity_role"] == "BLOCKING"
    assert runner["public_development_or_holdout_outcome_opened"] is False
    repair = result["public_no_hand_preparation_engineering_repair"]
    assert repair["failure_class"] == "ENGINEERING_FAILURE_NOT_SCIENTIFIC_NO_GO"
    assert repair["model_inference_executed"] is False
    assert repair["scientific_contract_changed"] is False
    prepared = result["public_no_hand_review_preparation_result"]
    assert prepared["status"] == "READY_FOR_AUTHORIZED_APPLICANT_BLIND_REVIEW"
    assert prepared["review_queue_count"] == 384
    assert prepared["model_inference_executed"] is False
    assert prepared["public_model_outcome_opened"] is False
    correction = result["public_no_hand_review_label_semantics_correction"]
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
    assert correction["review_queue_commitment_sha256"] == (
        "5ba8ae3eaa1d7189aac0a9435f1d10200dfbdfa4555125ed32867bb22c82f2b7"
    )
    assert correction["before_label_record_sha256"] == (
        "7dd640e1bc8e02fe7ced62d1ad7cbe5ccb9e2503b1877840ccb9b093b54e4fb5"
    )
    assert correction["after_label_record_sha256"] == (
        "2ffae4d120fdda4d995cb9d9056655d83082d05607d1860ef75199ea7109eae8"
    )
    assert correction["queue_or_scientific_contract_changed"] is False
    assert correction["review_sealed"] is False
    assert correction["model_inference_executed"] is False
    assert correction["public_model_or_scientific_gate_outcome_opened"] is False
    review_seal = result["public_no_hand_review_seal_result"]
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
    assert review_seal["authorized_applicant_attested"] is True
    assert review_seal["blind_to_EgoHOS_output_attested"] is True
    assert review_seal["EgoHOS_inference_not_started_attested"] is True
    assert review_seal["fixture_preparation_authorized"] is True
    assert review_seal["fixture_preparation_executed"] is False
    assert review_seal["model_inference_executed"] is False
    assert review_seal["public_development_or_holdout_outcome_opened"] is False
    fixture_repair = result["public_fixture_preparation_engineering_repair"]
    assert fixture_repair["runtime_manifest_failure"]["job_id"] == 315430
    assert fixture_repair["runtime_manifest_failure"]["exit_code"] == "1:0"
    assert fixture_repair["runtime_lineage_copy"]["copied_bytes"] == 23438221791
    assert fixture_repair["runtime_lineage_copy"]["copied_GiB"] == 21.828545
    assert fixture_repair["runtime_lineage_copy"]["free_GiB_after_copy"] == 535.877
    assert fixture_repair["video_encode_failure"]["job_id"] == 315445
    assert fixture_repair["video_encode_failure"]["elapsed_seconds"] == 81
    assert fixture_repair["video_encode_failure"]["root_cause_suffix"] == ".mp4.partial"
    assert fixture_repair["video_encode_failure"]["fixed_suffix"] == ".partial.mp4"
    assert fixture_repair["current_runner_source_sha256"] == (
        "1897c40be4f19c54e18042bc97cedf4c895447963577c670d0f48947a7066768"
    )
    assert fixture_repair["suffix_fix_retry_failure"] == {
        "job_id": 315452,
        "exit_code": "1:0",
        "elapsed_seconds": 11,
        "failure": "E_TUPLE_FIXTURE_VIDEO_ENCODE",
        "failed_before_fixture_manifest_or_model_inference": True,
    }
    assert [row["job_id"] for row in fixture_repair["diagnostic_jobs"]] == [
        315453,
        315455,
        315457,
        315458,
    ]
    assert fixture_repair["audio_seed_inventory"] == {
        "file_count": 112,
        "channel_layout": "mono",
        "sample_rate_hz": 22050,
    }
    assert fixture_repair["portable_filter_mapping"]["silent_source"] == (
        "anullsrc=r=22050:cl=mono,atrim=duration=<d>"
    )
    assert fixture_repair["portable_filter_mapping"]["speech_delay"] == "adelay=2500"
    smoke = fixture_repair["pinned_binary_smoke"]
    assert smoke["silent_exit_code"] == smoke["speech_exit_code"] == 0
    assert smoke["silent_active_sample_count"] == 0
    assert smoke["speech_timing_error_samples"] == {"start": 59, "end": -26}
    assert smoke["timing_errors_within_one_AAC_frame"] is True
    assert fixture_repair["audit_status"] == "PASS"
    assert fixture_repair["fixture_development_threshold_holdout_outputs_present"] is False
    assert fixture_repair["model_inference_executed"] is False
    assert fixture_repair["fixture_retry_authorized"] is True
    assert fixture_repair["no_further_engineering_diagnostic_required"] is True
    tuple_amendment = result["governed_C_mechanistic_training_tuple_amendment"]
    assert tuple_amendment["status"] == "FROZEN_BEFORE_NEW_PUBLIC_C_GENERATOR_OR_SYNTHETIC_LEARNER_OUTCOMES"
    assert len(tuple_amendment["prior_no_go_commitments_preserved"]) == 4
    assert tuple_amendment["axis_count"] == 7
    assert tuple_amendment["critical_axis_count"] == 5
    assert tuple_amendment["public_outcomes_opened_after_amendment"] is False
    assert tuple_amendment["governed_C_reopened_after_amendment"] is False
    assert tuple_amendment["generator_or_synthetic_learner_outcome_opened"] is False
    assert tuple_amendment["prior_pre_clarification_amendment_commitment_sha256"] == "fed6a3dc573c1453d4c46a07c786805dd65aad774fb6ae6e386d11fc0f444222"
    assert tuple_amendment["amendment_commitment_sha256"] == "c9a48206d09e0a3e8f771c5ec96f03c02d244a18f60b226227eca8ccddd9adaf"
    fixture_protocol = result["mechanistic_training_tuple_public_fixture_protocol"]
    assert fixture_protocol["status"] == "FROZEN_BEFORE_PUBLIC_MODEL_OUTCOMES"
    assert fixture_protocol["protocol_commitment_sha256"] == (
        "506a1f41a3685ca777f3c9d23f6f9b884523acec2a78080d5de2547b3324251d"
    )
    assert fixture_protocol["order_action_label_count"] == 8
    assert fixture_protocol["task_matched_fixture_manifest_commitments"].startswith(
        "PENDING_PREPARATION"
    )
    assert fixture_protocol["public_model_outcome_opened"] is False
    fixture_preparation = result[
        "mechanistic_training_tuple_fixture_preparation_amendment"
    ]
    assert fixture_preparation["status"] == (
        "FROZEN_BEFORE_MANIFEST_CONSTRUCTION_OR_PUBLIC_MODEL_OUTCOMES"
    )
    assert fixture_preparation["preparation_amendment_commitment_sha256"] == (
        "1cc8d0e3498da5785a2c2105307bf6d5ab20dd10f839ec0f2b92b9def372ff1d"
    )
    assert fixture_preparation["items_per_partition"] == 312
    assert fixture_preparation["public_model_outcome_opened"] is False
    fixture_feasibility = result[
        "mechanistic_training_tuple_fixture_feasibility_repair_amendment"
    ]
    assert fixture_feasibility["status"] == (
        "FROZEN_AFTER_PREMODEL_FIXTURE_YIELD_STOP_BEFORE_ANY_PUBLIC_MODEL_OUTCOME"
    )
    assert fixture_feasibility["fixture_feasibility_repair_commitment_sha256"] == (
        "e5fd286e9b8140583a37b855fe7125d7c6a0a2e3b57589b53294f77d28e47048"
    )
    assert fixture_feasibility["public_model_outcome_opened"] is False
    assert fixture_feasibility["scientific_thresholds_changed"] is False
    fixture_result = result["mechanistic_training_tuple_fixture_feasibility_result"]
    assert fixture_result["status"] == "NO_GO_ANNOTATION_ONLY_FIXTURE_SOURCE_YIELD"
    assert fixture_result["fixture_feasibility_commitment_sha256"] == (
        "dee0a37548d75fc29f829159ce3ad648a63288339c7bc173f8201483e51213e7"
    )
    assert fixture_result["development_stratum_counts"] == {
        "hand_contact": 16,
        "hand_no_contact": 0,
        "true_no_hand": 12,
    }
    assert fixture_result["holdout_stratum_counts"] == {
        "hand_contact": 16,
        "hand_no_contact": 0,
        "true_no_hand": 12,
    }
    assert fixture_result["required_count"] == 12
    assert fixture_result["available_count"] == 0
    assert fixture_result["model_inference_executed"] is False
    assert fixture_result["media_rendering_executed"] is False
    assert fixture_result["public_development_opened"] is False
    assert fixture_result["public_holdout_opened"] is False
    assert fixture_result["governed_C_reopened"] is False
    assert fixture_result["LTX_preflight_or_generation_run"] is False
    assert fixture_result["synthetic_learner_run"] is False
    runtime_preparation = result[
        "mechanistic_training_tuple_runtime_preparation_result"
    ]
    assert runtime_preparation["status"] == (
        "PASS_55_DEPENDENCY_RUNTIME_READY_LOCAL_RELOAD_BLIND_SIZING_PENDING"
    )
    assert runtime_preparation["installed_distribution_count"] == 55
    assert runtime_preparation["runtime_dependency_commitment_sha256"] == (
        "df15ff20c2f1e137530ec6f8a6f848ed676f2bcde0ec48148736c23ddd6fc0c4"
    )
    assert runtime_preparation["model_inference_executed"] is False
    sizing_validation = result[
        "mechanistic_training_tuple_sizing_validation_amendment"
    ]
    assert sizing_validation["status"] == (
        "FROZEN_BEFORE_SIZING_RERUN_OR_PUBLIC_FIXTURE_OUTCOMES"
    )
    assert sizing_validation["validation_commitment_sha256"] == (
        "afc936f742bd4313c35ff2e9a11a2389c589675c03309bbf09d8f8ab718ea2d5"
    )
    assert sizing_validation["padding_mask_exact_required"] is True
    assert sizing_validation["score_or_prediction_retained"] is False
    sizing_result = result["mechanistic_training_tuple_sizing_result"]
    assert sizing_result["status"] == "PASS_LABEL_BLIND_LOCAL_RELOAD_SIZING"
    assert sizing_result["module_count"] == 8
    assert sizing_result["finite_output_count"] == 8
    assert sizing_result["failure_count"] == 0
    assert sizing_result["tuple_sizing_commitment_sha256"] == (
        "b627590518b2e54daeace3d1c52e6918b41c2e203b42742538135c6a4c63e029"
    )
    assert sizing_result["public_fixture_outcome_opened"] is False
    premodel = result["mechanistic_training_tuple_premodel_result"]
    assert premodel["status"] == "PASS_ARTIFACTS_READY_LOCAL_RELOAD_PENDING_BLIND_SIZING"
    assert premodel["dependency_manifest_commitment_sha256"] == (
        "8c787a01f2e0f6224bc96989d3e3bd28ef6f6b03e0459599507636e41c85b527"
    )
    assert premodel["engineering_failure_count_before_pass"] == 3
    assert premodel["model_inference_executed"] is False
    assert premodel["restricted_mount_present"] is False
    assert premodel["governed_C_reopened"] is False
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
    assert sizing["status"] == "PASS_ALL_THREE_CONTROLS_RETAINED"
    assert sizing["fixture_labels_used"] is False
    assert sizing["scores_predictions_or_scientific_metrics_retained"] is False
    assert sizing["aggregate_GPU_hours_through_C_including_sizing_max"] == 20.0
    safe_load = result["governed_C_activity_checkpoint_safe_load_repair"]
    assert safe_load["status"] == "PASS_SAFE_LOAD_AND_EGOHOD_BLIND_SIZING"
    assert safe_load["unsafe_global_count"] == 13
    assert safe_load["dynamic_safe_type_count"] == 1
    assert safe_load["weights_only_remains_true"] is True
    assert safe_load["model_inference_executed"] is True
    sizing_progress = result["governed_C_activity_checkpoint_sizing_progress"]
    assert sizing_progress["status"] == "PASS_ALL_THREE"
    assert sizing_progress["completed_candidate_count"] == 3
    assert sizing_progress["passing_candidate_count"] == 3
    assert sizing_progress["videoprism_record_commitment_verified"] is True
    assert sizing_progress["vjepa2_record_commitment_verified"] is True
    development = result["governed_C_activity_checkpoint_development_result"]
    assert development["status"] == "NO_GO_NO_ELIGIBLE_CANDIDATE"
    assert development["eligible_candidate_count"] == 0
    assert development["selection_commitment_verified"] is True
    assert development["winner_selected"] is False
    assert development["holdout_opened"] is False
    assert development["governed_C_reopened"] is False
    assert sizing_progress["record_commitment_verified"] is True
    assert sizing_progress["score_or_prediction_retained"] is False
    assert result["governance_incident"]["restricted_execution_paused"] is False


def test_coverage_redesign_is_frozen_without_rewriting_prior_stop():
    proof = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    assert proof["status"] == "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_8_GIT_ABSENCE_BLOCKER_PRESERVED_CLEAN_TREE_FALLBACK_ATTEMPT_9_FROZEN_NO_NEW_OUTCOME"
    geometry_repair = proof["public_fixture_geometry_rasterization_repair"]
    assert geometry_repair["repair_commitment_sha256"] == "6084fd937c208feda00aa3dc1cf14d0ec56e8f13bd24b56e23e4a6a6553e61ef"
    assert geometry_repair["triggering_attempt"]["public_model_inference_executed"] is False
    assert geometry_repair["scientific_thresholds_changed"] is False
    fixture_result = proof["learner_effective_public_fixture_preparation_result"]
    assert fixture_result["status"] == "PASS_PUBLIC_FIXTURES_SEALED_NO_MODEL_INFERENCE"
    assert fixture_result["public_fixture_manifest_commitment_sha256"] == (
        "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
    )
    assert fixture_result["public_development_authorized"] is True
    assert fixture_result["holdout_authorized_only_after_development_threshold_seal"] is True
    resource = proof["learner_effective_public_qualification_resource_amendment"]
    assert resource["active_topology"]["GPU_type"] == "NVIDIA_A30_24GB"
    assert resource["triggering_H100_job"]["qualification_output_count"] == 0
    assert resource["amendment_commitment_sha256"] == (
        "5330cf582e46d1bf075ca97af7c8bfceb47cfcd09499786f4e366b6f8e283beb"
    )
    development = proof["learner_effective_public_development_result"]
    assert development["status"] == "NO_GO_DEVELOPMENT_COMBINED_GATE"
    assert development["combined_gate"]["critical_axis_pass_count"] == 1
    assert development["combined_gate"]["validated_axis_count"] == 2
    assert development["integrity"]["unaccounted_failure_count"] == 3
    assert development["holdout_authorized"] is False
    assert development["governed_C_reopened"] is False
    assert development["LTX_or_synthetic_learner_run"] is False
    health = dict(proof["learner_effective_engineering_health_amendment"])
    commitment = health.pop("amendment_commitment_sha256")
    assert commitment == digest(health)
    assert commitment == "d447a7e165136032a1fba43605d3f81881b41ec030c82e9028e1a8f5cb2c6205"
    assert health["prior_public_development_result"]["canonical_subtree_sha256"] == (
        "c43c7a678e3a2eac10ed5a5ac75c8964520931ec180ab9306585c76d198fb8c8"
    )
    scheduler = dict(
        proof["learner_effective_engineering_health_scheduler_policy"]
    )
    scheduler_commitment = scheduler.pop("amendment_commitment_sha256")
    assert scheduler_commitment == digest(scheduler)
    assert scheduler_commitment == (
        "8ef8e53c2754fe13b91518c02f419a1d3c4f3162aa18648f7044986854d327d6"
    )
    assert scheduler["canceled_submission"]["attempt_file_count"] == 0
    assert scheduler["active_attempt_resource_policy"]["GPU_type"] == (
        "NVIDIA_H100_NVL"
    )
    attempt_8 = dict(
        proof["learner_effective_engineering_health_attempt_8_result"]
    )
    attempt_8_commitment = attempt_8.pop("blocker_commitment_sha256")
    assert attempt_8_commitment == digest(attempt_8)
    assert attempt_8_commitment == (
        "409c36d2c3ba4aefdd2f510c661ba363c000fc232dce2fcfba0151ba25f9aad7"
    )
    assert attempt_8["compact_aggregate"]["scientific_metric_count"] == 0
    repair = dict(
        proof["learner_effective_engineering_health_git_fallback_repair"]
    )
    repair_commitment = repair.pop("repair_commitment_sha256")
    assert repair_commitment == digest(repair)
    assert repair_commitment == (
        "b6a93e3a0b0b716d8bdd8fdd47656e69f8ff5b66c0a3ec8f96973e565a9066f9"
    )
    assert repair["active_attempt_resource_policy"]["attempt"] == 9
    assert repair["execution_and_stop_rule"][
        "no_repeated_scheduler_polling_or_unchanged_status_updates"
    ] is True
    assert health["engineering_microfixture_suite"]["total_case_count"] == 28
    assert health["bounded_resource_policy"][
        "initial_plus_repair_resmoke_submission_count_max"
    ] == 3
    redirect = dict(proof["learner_effective_engineering_health_resource_redirect"])
    redirect_commitment = redirect.pop("amendment_commitment_sha256")
    assert redirect_commitment == digest(redirect)
    assert redirect_commitment == (
        "f7fc16f5c399c2a2d213b13a0d255a14b5b2f3ece41d62adaed17f61f186db6d"
    )
    assert redirect["canceled_A30_submission"]["state"] == (
        "CANCELLED_BEFORE_ALLOCATION"
    )
    assert redirect["canceled_A30_submission"]["elapsed_seconds"] == 0
    assert redirect["canceled_A30_submission"]["compact_result_count"] == 0
    assert redirect["active_health_topology"]["partition"] == "h100"
    assert redirect["active_health_topology"]["GRES"] == (
        "gpu:nvidia_h100_nvl_3g.47gb:1"
    )
    assert redirect["bounded_resource_policy"]["aggregate_GPU_hours_max"] == 0.75
    restore = dict(proof["learner_effective_engineering_health_dependency_restore"])
    restore_commitment = restore.pop("repair_commitment_sha256")
    assert restore_commitment == digest(restore)
    assert restore_commitment == (
        "3c54503b4087fae1e993b0aa952823f988a088a5c6543760df1022e2dc046db4"
    )
    assert restore["triggering_attempt"]["job_id"] == 316325
    assert restore["triggering_attempt"]["scientific_metric_count"] == 0
    assert restore["active_language_dependency_archive"]["sha256"] == (
        "97ef52ecaa8c99db017e598d8a63d0d2170affef14ef46e7df7a656abd3a1a07"
    )
    assert restore["remaining_health_budget"]["submission_count_remaining"] == 2
    topology_repair = dict(
        proof["learner_effective_engineering_health_topology_guard_repair"]
    )
    topology_commitment = topology_repair.pop("repair_commitment_sha256")
    assert topology_commitment == digest(topology_repair)
    assert topology_commitment == (
        "8db2d8ae04ee702ab3c68ff7c243afed0d8e4710c01f3f7865faa4975fb9a5b8"
    )
    assert topology_repair["triggering_attempt"]["job_id"] == 316353
    assert topology_repair["triggering_attempt"]["scientific_metric_count"] == 0
    assert topology_repair["aggregate_read_only_diagnosis"][
        "authoritative_scontrol_predicate_pass_count"
    ] == 7
    assert topology_repair["remaining_health_budget"]["submission_count_remaining"] == 1
    blocker = dict(proof["learner_effective_engineering_health_result"])
    blocker_commitment = blocker.pop("blocker_commitment_sha256")
    assert blocker_commitment == digest(blocker)
    assert blocker_commitment == (
        "644028babc768e881276fa078b95349ba77f8418cb76d722e8baf2588f9d0f81"
    )
    assert blocker["final_attempt"]["job_id"] == 316370
    assert blocker["compact_aggregate"]["status"] == "ENGINEERING_BLOCKER"
    assert blocker["compact_aggregate"]["scientific_metric_count"] == 0
    assert blocker["stable_aggregate_diagnosis"]["unaccounted_exception_type"] == (
        "FileNotFoundError"
    )
    assert blocker["resource_accounting"]["submission_count_remaining"] == 0
    assert blocker["terminal_gate"]["attempt_4_authorized"] is False
    reauthorization = dict(
        proof["learner_effective_engineering_health_reauthorization"]
    )
    reauthorization_commitment = reauthorization.pop(
        "reauthorization_commitment_sha256"
    )
    assert reauthorization_commitment == digest(reauthorization)
    assert reauthorization_commitment == (
        "3271499c19a77ffab8c53e2cd2052ea14514682a3d4b73fd7c5179ceec4a7ff4"
    )
    assert reauthorization["preserved_without_change"][
        "sealed_engineering_blocker_sha256"
    ] == blocker_commitment
    assert reauthorization["effective_resource_policy"]["reauthorized_attempt"] == 4
    assert reauthorization["execution_and_stop_rule"][
        "repair_or_resmoke_cycles_after_attempt_4"
    ] == 0
    terminal_reauthorization = dict(
        proof["learner_effective_engineering_health_reauthorization_result"]
    )
    terminal_reauthorization_commitment = terminal_reauthorization.pop(
        "blocker_commitment_sha256"
    )
    assert terminal_reauthorization_commitment == digest(terminal_reauthorization)
    assert terminal_reauthorization_commitment == (
        "59b1778b35cedd1cb020177e41fe6887371a5480f7ee6bf6e57f55d4c90edde3"
    )
    assert terminal_reauthorization["submission_provenance"]["job_id"] == 316478
    assert terminal_reauthorization["submission_provenance"][
        "scientific_metric_count"
    ] == 0
    assert terminal_reauthorization["terminal_gate"]["attempt_5_authorized"] is False
    parser_repair = dict(
        proof[
            "learner_effective_engineering_health_parser_repair_reauthorization"
        ]
    )
    parser_repair_commitment = parser_repair.pop(
        "reauthorization_commitment_sha256"
    )
    assert parser_repair_commitment == digest(parser_repair)
    assert parser_repair_commitment == (
        "d9cf3feaa0f5c4d65978ca796b722f31e75ce1078b918d114ca35b298a148c8b"
    )
    assert parser_repair["preserved_without_change"][
        "attempt_4_blocker_sha256"
    ] == terminal_reauthorization_commitment
    assert parser_repair["failure_specific_repair"]["new_parser_choices"] == [
        1,
        2,
        3,
        5,
    ]
    assert parser_repair["failure_specific_repair"][
        "attempt_4_remains_rejected_and_sealed"
    ] is True
    assert parser_repair["effective_resource_policy"][
        "reauthorized_attempt"
    ] == 5
    assert parser_repair["execution_and_stop_rule"][
        "repair_or_resmoke_cycles_after_attempt_5"
    ] == 0
    parser_result = dict(
        proof["learner_effective_engineering_health_parser_repair_result"]
    )
    parser_result_commitment = parser_result.pop("blocker_commitment_sha256")
    assert parser_result_commitment == digest(parser_result)
    assert parser_result_commitment == (
        "b05dc8da3155561b182b3bcfa50c851f83828b34e063918306bdfb57fdedeb9c"
    )
    assert parser_result["submission_provenance"]["job_id"] == 316537
    assert parser_result["compact_aggregate"]["completed_module_count"] == 0
    assert parser_result["compact_aggregate"]["scientific_metric_count"] == 0
    assert parser_result["stable_aggregate_diagnosis"]["artifact_family"] == (
        "BASE_CONTAINER"
    )
    assert parser_result["terminal_gate"]["attempt_6_authorized"] is False
    iterative = dict(
        proof["learner_effective_engineering_health_iterative_reauthorization"]
    )
    iterative_commitment = iterative.pop("reauthorization_commitment_sha256")
    assert iterative_commitment == digest(iterative)
    assert iterative_commitment == (
        "3114e1763f65dbeb8b2f89bb2a0480c86f4266f888c1ac2ff740bee85d357ab9"
    )
    assert iterative["preserved_without_change"]["attempt_5_blocker_sha256"] == (
        parser_result_commitment
    )
    assert iterative["active_attempt_resource_policy"]["attempt"] == 6
    assert iterative["rolling_execution_policy"][
        "blanket_user_authorization_for_additional_ordinary_engineering_attempts"
    ] is True
    attempt_6 = dict(
        proof["learner_effective_engineering_health_iterative_attempt_6_result"]
    )
    attempt_6_commitment = attempt_6.pop("blocker_commitment_sha256")
    assert attempt_6_commitment == digest(attempt_6)
    assert attempt_6_commitment == (
        "e559cd535d2a6dd833d2588c75b180754260dd3ff68ea9f6731a0e4478a6d114"
    )
    assert attempt_6["compact_aggregate"]["model_module_inference_count"] == 0
    progress = dict(proof["learner_effective_engineering_health_progress_repair"])
    progress_commitment = progress.pop("reauthorization_commitment_sha256")
    assert progress_commitment == digest(progress)
    assert progress_commitment == (
        "a2d1347bef14848a5238f9a10c6e94da8eaa68593aa3d97e8a460dfbf8694d07"
    )
    assert progress["preserved_without_change"]["attempt_6_blocker_sha256"] == (
        attempt_6_commitment
    )
    assert progress["active_attempt_resource_policy"]["attempt"] == 7
    attempt_7 = dict(proof["learner_effective_engineering_health_attempt_7_result"])
    attempt_7_commitment = attempt_7.pop("blocker_commitment_sha256")
    assert attempt_7_commitment == digest(attempt_7)
    assert attempt_7_commitment == (
        "03c09a61cedb29e04cf465287db693cd1248c53d45d9a7a47a777e6cdf1d594d"
    )
    assert attempt_7["compact_aggregate"]["scientific_metric_count"] == 0
    extended = dict(
        proof["learner_effective_engineering_health_extended_wall_repair"]
    )
    extended_commitment = extended.pop("reauthorization_commitment_sha256")
    assert extended_commitment == digest(extended)
    assert extended_commitment == (
        "d2db51229719a0e64f84da9541a284d88c75a2fd32e2e186ba34e17ab5eed6e7"
    )
    assert extended["active_attempt_resource_policy"]["attempt"] == 8
    assert extended["active_attempt_resource_policy"]["wall_minutes_max"] == 60
    assert health["unchanged_downstream_contract"]["accepted_synthetic_seconds_exact"] == 3600
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
    assert proof["schema_version"] == 31
    premodel = proof["calibration_C"]["extractor"]["mechanistic_training_tuple_premodel_result"]
    assert premodel["dependency_manifest_commitment_sha256"] == (
        "8c787a01f2e0f6224bc96989d3e3bd28ef6f6b03e0459599507636e41c85b527"
    )
    assert premodel["counts"]["artifact_bytes"] == 14621041722
    assert premodel["counts"]["egohos_archive_license_file_count"] == 0
    assert premodel["model_inference_executed"] is False
    assert proof["generator_gate"]["status"].startswith("SELECTED_LOCAL_NOT_RUN")
    assert proof["generator_gate"]["implementation"]["commit"] == (
        "9377758131b1ffde4b7f766804590a6617bf2ab9"
    )
    assert proof["generator_gate"]["weights"]["revision"] == (
        "4229404625088d21c4f112eb640fb04a0900ee25"
    )
    assert proof["generator_gate"]["production_ceiling_provisional_until_preflight"]["accepted_credited_seconds_exact"] == 3600
    assert proof["generator_gate"]["production_ceiling_provisional_until_preflight"]["raw_generated_seconds_max"] == 5399.625
    assert proof["generator_gate"]["production_ceiling_provisional_until_preflight"]["attempts_max"] == 1071
    assert "LTX 19/28" in proof["generator_gate"]["bakeoff_interpretation"]
    assert proof["ambitious_learner_effective_h3_amendment"]["amendment_commitment_sha256"] == "d907d2479ba88c7e51d25e935e429cb5860e550a82620313e502482383e2855d"
    assert proof["generator_gate"]["no_substitution"] is True
    assert proof["calibration_C"]["source"] == "development_set_C_only_never_training_validation_or_evaluation"
    assert proof["calibration_C"]["local_generator_gate"] == (
        "CONDITIONAL_ON_CONSTRUCT_ALIGNED_COMBINED_PUBLIC_AND_C_PASS_PLUS_LTX_FINAL_TOPOLOGY_PREFLIGHT"
    )
    fixture_source_no_go = proof["calibration_C"]["extractor"][
        "mechanistic_training_tuple_fixture_feasibility_result"
    ]
    assert fixture_source_no_go["status"] == "NO_GO_ANNOTATION_ONLY_FIXTURE_SOURCE_YIELD"
    assert fixture_source_no_go["fixture_feasibility_commitment_sha256"] == (
        "dee0a37548d75fc29f829159ce3ad648a63288339c7bc173f8201483e51213e7"
    )
    assert fixture_source_no_go["required_count"] == 12
    assert fixture_source_no_go["available_count"] == 0
    assert fixture_source_no_go["public_development_opened"] is False
    assert fixture_source_no_go["governed_C_reopened"] is False
    assert fixture_source_no_go["LTX_preflight_or_generation_run"] is False
    assert fixture_source_no_go["synthetic_learner_run"] is False
    correction = proof["calibration_C"]["extractor"][
        "mechanistic_training_tuple_visor_hos_correction_amendment"
    ]
    assert correction["status"] == (
        "FROZEN_BEFORE_NEW_PUBLIC_SOURCE_INVENTORY_MODEL_C_GENERATOR_OR_LEARNER_OUTCOMES"
    )
    assert correction["prior_result_preserved"][
        "fixture_feasibility_commitment_sha256"
    ] == "dee0a37548d75fc29f829159ce3ad648a63288339c7bc173f8201483e51213e7"
    assert correction["official_annotation_artifact"]["combined_JSON_file_count"] == 158
    assert correction["partition_and_joint_sampler"][
        "quota_per_partition_per_stratum"
    ] == 48
    assert correction["public_execution_and_combined_gate"][
        "one_supporting_axis_may_be_unmeasured"
    ] is True
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
