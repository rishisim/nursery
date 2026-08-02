from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "calibration", Path("scripts/run_synthetic_video_calibration.py")
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_calibration_protocol_freezes_eight_axes_and_four_joints() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    calibration = config["calibration_C"]
    assert len(calibration["axes"]) == 8
    assert len(calibration["joint_distributions"]) == 4
    assert calibration["extractor"]["uncertainty"] == {
        "method": "whole-child_cluster_bootstrap",
        "replicates": 1000,
        "seed": 42,
        "interval": 0.95,
    }
    assert calibration["extractor"]["generated_comparison_tolerances"]["omnibus_score"] == "PROHIBITED"
    assert calibration["episode_plan"]["candidate_plan_count"] == 1428
    assert calibration["episode_plan"]["evaluation_steering"] == "PROHIBITED"
    repair = calibration["extractor"]["coverage_repair"]
    assert repair["status"] == "FROZEN_ACTIVE"
    assert len(repair["candidate_set"]) == 1
    assert repair["unchanged_gate"] == {
        "maximum_axis_missing_fraction": 0.2,
        "critical_axes_must_all_pass": True,
        "measured_axes_min": 6,
        "no_imputation": True,
    }
    assert repair["detector_model"]["revision"] == "cfd3195ba4ea9592eec887ded089f4c08eff231d"
    assert repair["detector_model"]["license"] == "Apache-2.0"
    assert repair["runtime_dependency"] == {
        "package": "scipy",
        "version": "1.16.1",
        "wheel": "scipy-1.16.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "wheel_sha256": "adccd93a2fa937a27aae826d33e3bfa5edf9aa672376a4852d23a7cd67a2e5b7",
        "license": "BSD-3-Clause",
        "role": "required by the pinned Transformers OWLv2 image processor; public local dependency only",
    }
    assert len(repair["public_qualification"]["fixtures"]) == 8
    assert repair["public_qualification"]["thresholds"] == {
        "activity_correct_min": 4,
        "expected_object_hits_min": 5,
        "hand_positive_hits_min": 2,
        "hand_negative_correct_min": 2,
        "proxy_complete_required": 8,
        "invalid_box_count_max": 0,
    }

    redesign = calibration["extractor"]["domain_appropriate_redesign"]
    assert redesign["status"] == "NO_GO_PREMODEL_FEASIBILITY"
    assert set(redesign["single_stack"]) == {
        "activity_context",
        "hand_object_action",
        "scene_and_referent_detection",
        "mask_tracking",
        "diversity_embeddings",
    }
    assert redesign["governed_C_gate_unchanged"] == {
        "maximum_axis_missing_fraction": 0.2,
        "critical_axes_must_all_pass": True,
        "measured_axes_min": 6,
        "axis_count": 8,
        "joint_count": 4,
        "omnibus_score": "PROHIBITED",
    }
    assert redesign["execution_controls"]["canonical_entry_point"] == "scripts/run_synthetic_video_calibration.py"
    assert redesign["execution_controls"]["no_substitution"] is True
    gates = redesign["public_holdout_gates"]
    assert gates["activity_context_macro_f1_min"] == 0.6
    assert gates["hand_visibility_sensitivity_min"] == 0.8
    assert gates["hand_visibility_specificity_min"] == 0.8
    assert gates["tracked_category_temporal_presence_f1_min"] == 0.7
    assert gates["audiovisual_referent_timing_macro_f1_min"] == 0.65
    assert gates["near_duplicate_balanced_accuracy_min"] == 0.9
    feasibility = redesign["premodel_feasibility_result"]
    assert feasibility["status"] == "NO_GO"
    assert feasibility["blocking_component"] == "EgoVLPv2_activity_context"
    assert feasibility["new_model_inference_executed"] is False
    assert feasibility["checkpoint_bytes_resolved"] is False
    assert feasibility["checkpoint_sha256_resolved"] is False
    assert feasibility["checkpoint_specific_weight_terms_found"] is False

    selection = calibration["extractor"]["activity_checkpoint_selection_amendment"]
    assert selection["status"] == "FROZEN_BEFORE_EMPIRICAL_CANDIDATE_OUTCOMES"
    assert [candidate["candidate_id"] for candidate in selection["bounded_candidates"]] == [
        "egohod_egovideo_l_zero_shot",
        "videoprism_lvt_l_zero_shot",
        "vjepa2_vitl_public_probe",
    ]
    assert all(len(candidate["weight_sha256"]) == 64 for candidate in selection["bounded_candidates"])
    assert selection["public_activity_fixture"]["manifest_commitment_sha256"] == "7a44e6cd72043e3720c98111e9d6e92b5a043ac43d66cdad5355bc01782441f8"
    assert selection["public_activity_fixture"]["subject_overlap_count"] == 0
    assert selection["public_activity_fixture"]["video_overlap_count"] == 0
    assert selection["public_activity_fixture"]["holdout_outcomes_opened"] is False
    assert selection["development_comparison"]["candidate_count"] == 3
    assert selection["development_comparison"]["eligibility_floors"]["macro_f1_min"] == 0.6
    assert selection["development_comparison"]["eligibility_floors"]["nonabstained_coverage_min"] == 0.8
    assert selection["independent_holdout"]["activity_context_macro_f1_min_unchanged"] == 0.6
    assert selection["resource_ceiling"]["GPU_count"] == 1
    assert selection["execution_controls"]["DDP"] is False
    runtime = selection["runtime_environment"]
    assert runtime["container"]["sha256"] == "f274f1ac3726376b762b557ff9a07203b2d42aac3157a7a354b998e589c35792"
    assert runtime["egohod"]["input_resolution"] == 336
    assert runtime["egohod"]["input_frames"] == 16
    assert runtime["videoprism"]["c4_en_sentencepiece_sha256"] == "1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec"
    assert runtime["vjepa2"]["config_sha256"] == "3dec96fe962e94e569182d3a7b9ef0dd74b6b8c89c337a428e43e10d593e70c9"
    assert runtime["shared"]["dependency_manifest_commitment_sha256"] == (
        "5fb4a9d3c4375621bc94b8d7c25a26f434c7c1de9226fadd9f50ffdd3023e81d"
    )
    preparation = selection["public_preparation_result"]
    assert preparation["status"] == "PASS_PREPARED_NO_MODEL_INFERENCE"
    assert preparation["candidate_count"] == 3
    assert preparation["model_inference_executed"] is False
    assert preparation["restricted_mount_present"] is False
    assert preparation["valid_for_candidate_inference"] is False
    safe_load = runtime["egohod"]["checkpoint_safe_load"]
    assert safe_load["weights_only"] is True
    assert safe_load["weights_only_false"] == "PROHIBITED"
    assert len(safe_load["exact_allowed_globals"]) == 13
    assert safe_load["additional_dynamic_safe_types_not_reported_by_static_scanner"] == [
        "numpy.dtypes.Float64DType"
    ]
    repreparation = selection["public_repreparation_result"]
    assert repreparation["status"] == "PASS_PREPARED_NO_MODEL_INFERENCE"
    assert repreparation["installed_distribution_count"] == 91
    assert repreparation["valid_for_candidate_inference"] is True
    assert repreparation["model_inference_executed"] is False
    assert [
        candidate["expected_sizing_output_width"]
        for candidate in selection["bounded_candidates"]
    ] == [8, 8, 1024]
    sizing = selection["resource_ceiling"]["blind_sizing"]
    assert sizing["fixture_labels_used"] is False
    assert sizing["score_or_prediction_retention"] == "PROHIBITED"
    assert sizing["scientific_metric_computation"] == "PROHIBITED"
    assert sizing["item_count_per_candidate"] == 1
    assert sizing["aggregate_GPU_hours_max"] == 1.5
    assert selection["resource_ceiling"]["aggregate_GPU_hours_through_C_max"] == 20.0


def test_activity_temporal_permutation_is_deterministic_and_not_identity() -> None:
    first = MODULE._deterministic_nonidentity_permutation(16, 20260802, "public-fixture")
    second = MODULE._deterministic_nonidentity_permutation(16, 20260802, "public-fixture")
    assert first == second
    assert sorted(first) == list(range(16))
    assert first != list(range(16))


def test_activity_threshold_and_abstention_are_conservative_and_explicit() -> None:
    threshold = MODULE._choose_label_threshold([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert 0.2 < threshold < 0.8
    labels = ["a", "b"]
    rows = [[0.7, 0.2], [0.1, 0.8], [0.51, 0.49]]
    predictions = MODULE._predict_score_rows(rows, labels, [0.5, 0.5], 0.05)
    assert predictions == [{"a"}, {"b"}, set()]
    margin = MODULE._choose_abstention_margin(
        rows,
        [{"a"}, {"b"}, set()],
        labels,
        [0.5, 0.5],
        [0.0, 0.05],
        2 / 3,
    )
    assert margin == 0.05


def test_activity_selection_order_uses_frozen_metrics_then_resources() -> None:
    base = {
        "eligible": True,
        "macro_f1": 0.7,
        "worst_class_recall": 0.6,
        "nonabstained_coverage": 0.9,
        "temporal": {
            "ordered_over_shuffled_positive_fraction": 0.7,
            "ordered_over_repeated_positive_fraction": 0.8,
        },
        "peak_vram_gib": 20.0,
        "median_item_runtime_seconds": 2.0,
    }
    slower = {**base, "candidate_id": "slower", "median_item_runtime_seconds": 3.0}
    faster = {**base, "candidate_id": "faster"}
    assert MODULE._select_activity_winner([slower, faster], 1e-6)["candidate_id"] == "faster"
    assert MODULE._select_activity_winner([{**faster, "eligible": False}], 1e-6) is None


def test_activity_vector_guard_rejects_nonfinite_or_wrong_width() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="E_TEST"):
        MODULE._finite_vector([0.1, float("nan")], 2, "E_TEST")
    with pytest.raises(RuntimeError, match="E_TEST"):
        MODULE._finite_vector([0.1], 2, "E_TEST")


def test_repository_commit_verification_works_inside_gitless_container(tmp_path, monkeypatch) -> None:
    expected = "a" * 40
    repository = tmp_path / "repo"
    (repository / ".git/refs/heads").mkdir(parents=True)
    (repository / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (repository / ".git/refs/heads/main").write_text(expected + "\n")
    monkeypatch.setattr(MODULE.shutil, "which", lambda _: None)
    MODULE._verify_repository_commit(repository, expected)


def test_activity_preparation_does_not_upgrade_pinned_container_torch() -> None:
    import inspect

    source = inspect.getsource(MODULE.prepare_activity_public)
    guard = inspect.getsource(MODULE._verify_container_torch_not_shadowed)
    assert '"--upgrade"' not in source
    assert source.count('"--no-deps"') == 1
    assert 'candidate_id == "egohod_egovideo_l_zero_shot"' in source
    assert "E_ACTIVITY_CONTAINER_TORCH_SHADOWED" in guard
    assert "E_ACTIVITY_CONTAINER_TORCH_VERSION" in guard


def test_activity_sizing_is_label_blind_and_retains_no_scores() -> None:
    import inspect

    source = inspect.getsource(MODULE.size_activity_candidate)
    assert "fixture['labels']" not in source
    assert '"rows"' not in source
    assert '"score_or_prediction_retained": False' in source
    assert '"scientific_metric_computed": False' in source
    assert '"fixture_manifest_ordinal": 0' in source
    assert '"external_call_count": 0' in source
    assert "candidate_id" not in MODULE.ACTIVITY_SIZING_FIELDS
    assert "candidate_id" not in MODULE.ACTIVITY_CANDIDATE_FIELDS
    assert "partition" not in MODULE.ACTIVITY_CANDIDATE_FIELDS
    assert "winner_candidate_id" not in MODULE.ACTIVITY_SELECTION_FIELDS


def test_egohod_optional_import_compatibility_is_inference_only() -> None:
    import sys

    import pytest

    torch = pytest.importorskip("torch")

    names = {
        "ipdb",
        "cv2",
        "timm",
        "timm.models",
        "timm.models.layers",
        "mmengine",
        "mmengine.model",
        "mmengine.model.weight_init",
    }
    previous = {name: sys.modules.get(name) for name in names}
    try:
        MODULE._install_egohod_optional_import_compatibility()
        from mmengine.model.weight_init import constant_init, trunc_normal_init
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_

        value = torch.tensor([1.0, 2.0])
        assert torch.equal(DropPath(0.0).eval()(value), value)
        with pytest.raises(RuntimeError, match="E_EGOHOD_UNEXPECTED_STOCHASTIC_DEPTH"):
            DropPath(0.1).train()(value)
        assert to_2tuple(4) == (4, 4)
        assert to_2tuple((3, 5)) == (3, 5)
        assert trunc_normal_ is torch.nn.init.trunc_normal_

        linear = torch.nn.Linear(2, 2)
        constant_init(linear, 0.25, bias=-0.5)
        assert torch.all(linear.weight == 0.25)
        assert torch.all(linear.bias == -0.5)
        trunc_normal_init(linear, mean=0.0, std=0.01, bias=0.75)
        assert torch.isfinite(linear.weight).all()
        assert torch.all(linear.bias == 0.75)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_egohod_checkpoint_load_keeps_weights_only_and_exact_global_gate() -> None:
    import inspect

    source = inspect.getsource(MODULE._load_egohod_activity_adapter)
    assert "get_unsafe_globals_in_checkpoint" in source
    assert "E_EGOHOD_UNEXPECTED_CHECKPOINT_GLOBAL" in source
    assert "torch.serialization.safe_globals" in source
    globals_source = inspect.getsource(MODULE._egohod_checkpoint_safe_globals)
    assert "numpy.dtypes.Float64DType" in globals_source
    assert "weights_only=True" in source
    assert "weights_only=False" not in source


def test_videoprism_import_compatibility_cannot_replace_local_tokenization(
    tmp_path, monkeypatch
) -> None:
    import sys

    code = tmp_path / "videoprism"
    code.mkdir()
    (code / "models.py").write_text(
        "x: tokenizers.Tokenizer\n"
        "def load(): return tokenizers.SentencePieceTokenizer('hosted')\n"
    )
    previous = sys.modules.get("videoprism.tokenizers")
    try:
        MODULE._install_videoprism_tokenizer_import_compatibility(tmp_path)
        tokenizers = sys.modules["videoprism.tokenizers"]
        import pytest

        with pytest.raises(
            RuntimeError, match="E_VIDEOPRISM_HOSTED_TOKENIZER_PATH_PROHIBITED"
        ):
            tokenizers.SentencePieceTokenizer("hosted")
        (code / "models.py").write_text("x = tokenizers.new_surface\n")
        with pytest.raises(
            RuntimeError, match="E_VIDEOPRISM_TOKENIZER_IMPORT_SURFACE"
        ):
            MODULE._install_videoprism_tokenizer_import_compatibility(tmp_path)
    finally:
        if previous is None:
            sys.modules.pop("videoprism.tokenizers", None)
        else:
            sys.modules["videoprism.tokenizers"] = previous


def test_bucket_and_union_duration_are_frozen_and_exact() -> None:
    assert MODULE.bucket(0.03, [0.0, 0.03, 0.07, 1.0]) == "bin_1"
    assert MODULE.bucket(1.0, [0.0, 0.03, 0.07, 1.0]) == "bin_2"
    assert MODULE.union_duration([(0, 2), (1, 3), (5, 6)]) == 4


def test_image_metric_gradients_share_the_same_interior_grid() -> None:
    import numpy as np
    from PIL import Image

    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, 4:, :] = 255
    metrics = MODULE._image_metrics(Image.fromarray(pixels), None)
    assert 0.0 <= metrics["clutter_edge_fraction"] <= 1.0
    assert metrics["blur_edge_strength"] > 0.0


def test_public_speech_act_rules_do_not_need_c_vocabulary() -> None:
    rules = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())["calibration_C"]["extractor"]["language_rules"]
    assert MODULE.classify_speech_act("This is a toy.", rules) == "naming"
    assert MODULE.classify_speech_act("Where is it?", rules) == "question"
    assert MODULE.classify_speech_act("Take it now.", rules) == "directive"
    assert MODULE.classify_speech_act("We are here.", rules) == "other"


def test_largest_remainder_allocation_is_deterministic_and_exact() -> None:
    first = MODULE._allocated_labels({"a": 0.6, "b": 0.4}, 11, 42, "fixture")
    second = MODULE._allocated_labels({"a": 0.6, "b": 0.4}, 11, 42, "fixture")
    assert first == second
    assert len(first) == 11
    assert first.count("a") == 7
    assert first.count("b") == 4


def test_detector_proxy_emits_explicit_negative_classes_without_imputation() -> None:
    repair = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )["calibration_C"]["extractor"]["coverage_repair"]
    proxies = MODULE._detector_proxies(
        {"detections": [], "invalid_box_count": 0, "width": 100, "height": 100},
        repair,
    )
    assert proxies == {
        "hand_visibility": "not_visible",
        "hand_action": "no_hand",
        "referent": "none",
        "distractors": "one",
        "occlusion": "clear",
        "framing": "distributed",
    }
    assert all(value is not None for value in proxies.values())


def test_detector_proxy_finds_contact_and_multiple_referents() -> None:
    repair = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )["calibration_C"]["extractor"]["coverage_repair"]
    record = {
        "invalid_box_count": 0,
        "width": 100,
        "height": 100,
        "detections": [
            {"kind": "hand", "label": "hand", "score": 0.9, "box": [20, 20, 50, 50]},
            {"kind": "object", "label": "toy", "score": 0.8, "box": [40, 30, 65, 60]},
            {"kind": "object", "label": "cup", "score": 0.7, "box": [75, 30, 95, 60]},
        ],
    }
    proxies = MODULE._detector_proxies(record, repair)
    assert proxies["hand_visibility"] == "visible"
    assert proxies["hand_action"] == "grasp_hold"
    assert proxies["referent"] == "ambiguous_many"
    assert proxies["distractors"] == "few"


def test_temporal_hand_completion_requires_all_three_positions() -> None:
    assert MODULE._temporal_hand_completion({}) is None
    assert MODULE._temporal_hand_completion(
        {
            "before": {"hand_action": "reach"},
            "during": {"hand_action": "grasp_hold"},
            "after": {"hand_action": "visible_no_contact"},
        }
    ) == "completed"


def test_batch_script_enforces_governed_offline_scratch_contract() -> None:
    source = Path("scripts/run_synthetic_video_calibration.sbatch").read_text()
    assert "#SBATCH --partition=h100" in source
    assert "#SBATCH --gpus-per-node=1" in source
    assert "#SBATCH --time=04:00:00" in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "WANDB_DISABLED=true" in source
    assert "scratch/media" in source
    assert "trap 'find" in source
    assert "--output=/dev/null" in source
    assert "calibration_repair" in source


def test_public_qualification_jobs_never_mount_restricted_data() -> None:
    prepare = Path("scripts/prepare_synthetic_video_calibration.sbatch").read_text()
    qualify = Path("scripts/qualify_synthetic_video_calibration.sbatch").read_text()
    assert "#SBATCH --partition=dev" in prepare
    assert "prepare-public" in prepare
    assert "activity-prepare" in prepare
    assert "PHASE4_ACTIVITY_CODE_CLEAN=1" in prepare
    assert "PHASE4_RESTRICTED_ROOT" not in prepare
    assert "#SBATCH --partition=h100" in qualify
    assert "#SBATCH --gpus-per-node=1" in qualify
    assert "#SBATCH --time=01:30:00" in qualify
    assert "qualify-public" in qualify
    assert "activity-candidate" in qualify
    assert "--net --network none" in qualify
    assert "HF_HUB_OFFLINE=1" in qualify
    assert "calibration-repair-pydeps" in qualify
    assert "PHASE4_RESTRICTED_ROOT" not in qualify


def test_terminal_report_is_flat_and_guarded() -> None:
    source = Path("scripts/run_synthetic_video_calibration.py").read_text()
    assert "compact_aggregate_json" in source
    assert "TERMINAL_FIELDS" in source
    assert "AXIS_STATUS_FIELDS" in source
    assert "report-axis-status" in source
    assert "restricted_calibration_features.json" in source
    assert "synthetic_one_hour/calibration_repair" in source
    assert "E_ORIGINAL_CALIBRATION_PROVENANCE" in source
    assert "ACTIVITY_CANDIDATE_FIELDS" in source
    assert "ACTIVITY_SELECTION_FIELDS" in source
    assert "E_ACTIVITY_HOLDOUT_BEFORE_WINNER_SEAL" in source
    assert 'print(json.dumps' not in source
