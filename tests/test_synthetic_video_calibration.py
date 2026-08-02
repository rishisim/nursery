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
    assert len(repair["public_qualification"]["fixtures"]) == 8
    assert repair["public_qualification"]["thresholds"] == {
        "activity_correct_min": 4,
        "expected_object_hits_min": 5,
        "hand_positive_hits_min": 2,
        "hand_negative_correct_min": 2,
        "proxy_complete_required": 8,
        "invalid_box_count_max": 0,
    }


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
    assert "PHASE4_RESTRICTED_ROOT" not in prepare
    assert "#SBATCH --partition=h100" in qualify
    assert "#SBATCH --gpus-per-node=1" in qualify
    assert "#SBATCH --time=01:00:00" in qualify
    assert "qualify-public" in qualify
    assert "HF_HUB_OFFLINE=1" in qualify
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
    assert 'print(json.dumps' not in source
