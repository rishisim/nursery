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
    assert result["status"] == "CORRECTED_COMMON_ASSETS_PASS_LEAN_EQUAL_DURATION_PILOT_IN_PROGRESS"
    assert result["scientifically_accepted"] is True
    assert result["contract_identical_all_arms"] is True
    assert set(references) == {"Real-full", "Synthetic-full", "Real-small", "Mixed"}
    assert len({canonical(value) for value in references.values()}) == 1
    assert references["Real-full"]["lexical"] == result["lexical_commitment"]
    assert references["Real-full"]["temporal"] == result["temporal_commitment"]


def test_lean_equal_duration_proof_is_frozen_before_scores():
    proof = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    assert proof["status"] == "FROZEN_LEAN_EQUAL_DURATION_PILOT_PRE_SCORE"
    assert proof["budgets_credited_hours"] == {
        "primary": 1,
        "conditional_real_extension": 3,
        "automatic_six_hour_disabled": True,
    }
    assert proof["learner"]["schedule"] == {"contrastive": 4, "mlm": 1, "dinov2": 1}
    assert proof["learner"]["initialization"].startswith("byte_identical")
    assert proof["real_1h_gate"]["realistic_lexical_macro_min"] == 0.52
    assert proof["real_1h_gate"]["improvement_over_initialized_min"] == 0.02
    assert proof["generator_gate"]["selected"] == "gemini-omni-flash-preview"
    assert proof["generator_gate"]["status"].endswith("INSTITUTIONAL_TRAINING_USE_CLEARANCE")
    assert proof["generator_gate"]["request"]["store"] is False
    assert proof["generator_gate"]["no_substitution"] is True
    assert proof["calibration_C"]["source"] == "development_set_C_only_never_training_validation_or_evaluation"
    assert proof["calibration_C"]["hosted_provider_gate"].startswith("BLOCKED_PENDING_AFFIRMATIVE")
    assert "no full distributional calibration claim" in proof["calibration_C"]["limitations"]
    assert len(proof["calibration_C"]["axes"]) == 8
    assert proof["calibration_C"]["joint_distributions"] == [
        "naming_by_referent_visibility",
        "naming_by_hand_object_action",
        "clutter_by_occlusion",
        "motion_by_blur",
    ]
    assert proof["calibration_C"]["omnibus_score"] == "PROHIBITED"
    assert "automatic_1_3_6_curve" in proof["prohibitions"]
