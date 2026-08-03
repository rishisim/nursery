from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile


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
    assert selection["status"] == "PUBLIC_DEVELOPMENT_NO_GO_NO_ELIGIBLE_CANDIDATE"
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
    development = selection["development_selection_result"]
    assert development["status"] == "NO_GO_NO_ELIGIBLE_CANDIDATE"
    assert development["eligible_candidate_count"] == 0
    assert development["winner_selected"] is False
    assert development["holdout_opened"] is False
    assert development["governed_C_reopened"] is False
    assert development["selection_commitment_verified"] is True
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

    tuple_amendment = calibration["extractor"]["mechanistic_training_tuple_amendment"]
    assert tuple_amendment["status"] == (
        "FROZEN_BEFORE_NEW_PUBLIC_C_GENERATOR_OR_SYNTHETIC_LEARNER_OUTCOMES"
    )
    assert len(tuple_amendment["prior_no_go_sequence_preserved"]) == 4
    assert [axis["id"] for axis in tuple_amendment["axes"]] == [
        "adapter_qualified_yield",
        "noun_adjective_exposure",
        "utterance_centered_referent_visibility_dominance_ambiguity",
        "cross_episode_recurrence",
        "adjective_attribute_contrast",
        "hand_action_coupling",
        "egocentric_sensor_regime",
    ]
    assert [axis["priority"] for axis in tuple_amendment["axes"]].count("critical") == 5
    assert tuple_amendment["broad_activity_context"]["status"] == "DESCRIPTIVE_NONBLOCKING"
    assert tuple_amendment["genuinely_order_dependent_action_control"]["public_action_pairs"] == [
        ["open", "close"],
        ["take", "put"],
        ["sit_down", "stand_up"],
        ["turn_on", "turn_off"],
    ]
    assert tuple_amendment["governed_C_measurement_gate"] == {
        "new_gate_not_retroactive_change": "the former eight-axis 0.20 missingness and six-of-eight no-go remains final for its broad estimator; this amendment prospectively evaluates seven different learner-effective axes",
        "maximum_axis_missing_fraction": 0.2,
        "public_ontology_eligible_fraction_of_all_noun_adjective_mentions_min": 0.6,
        "critical_axes": [
            "adapter_qualified_yield",
            "noun_adjective_exposure",
            "utterance_centered_referent_visibility_dominance_ambiguity",
            "cross_episode_recurrence",
            "adjective_attribute_contrast",
        ],
        "critical_axes_must_all_pass": True,
        "measured_axes_min": 6,
        "axis_count": 7,
        "human_transfer_audit_must_pass": True,
        "no_imputation": True,
        "valid_negative_requires_public_specificity_pass": True,
        "model_derived_proxies_not_human_ground_truth": True,
    }
    assert tuple_amendment["resource_ceiling"]["aggregate_GPU_hours_through_C_max"] == 15.0
    assert tuple_amendment["resource_ceiling"]["multi_process_or_DDP"] is False
    expected_commitment = tuple_amendment.pop("amendment_commitment_sha256")
    assert MODULE.digest(tuple_amendment) == expected_commitment


def test_activity_temporal_permutation_is_deterministic_and_not_identity() -> None:
    first = MODULE._deterministic_nonidentity_permutation(16, 20260802, "public-fixture")
    second = MODULE._deterministic_nonidentity_permutation(16, 20260802, "public-fixture")
    assert first == second
    assert sorted(first) == list(range(16))
    assert first != list(range(16))


def test_tuple_amendment_commitment_and_axis_guards_reject_mutation() -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    amendment = MODULE._tuple_amendment(config)
    assert amendment["amendment_commitment_sha256"] == (
        "c9a48206d09e0a3e8f771c5ec96f03c02d244a18f60b226227eca8ccddd9adaf"
    )
    config["calibration_C"]["extractor"]["mechanistic_training_tuple_amendment"][
        "axes"
    ][0]["priority"] = "important"
    with pytest.raises(RuntimeError, match="E_TUPLE_AMENDMENT_COMMITMENT"):
        MODULE._tuple_amendment(config)


def test_tuple_runtime_amendment_is_exact_and_rejects_mutation() -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    runtime = MODULE._tuple_runtime_amendment(config)
    assert runtime["runtime_amendment_commitment_sha256"] == (
        "eb878d8c68aa6f79b5115502beb4c8b64d84e9f495b7cec4513abc3e94effbea"
    )
    assert len(runtime["dependency_versions"]) == 55
    assert runtime["dependency_versions"]["einops"] == "0.8.0"
    assert runtime["dependency_versions"]["submitit"] == "1.5.3"
    assert runtime["dependency_versions"]["cloudpickle"] == "3.1.1"
    assert runtime["local_reload_gate"][
        "all_seven_axes_and_order_dependent_action_control_must_pass"
    ] is True
    assert runtime["local_reload_gate"]["module_count"] == 8
    assert len(runtime["compatibility_adapters"]) == 7
    assert runtime["prior_runtime_amendment_commitments_sha256"][-1] == (
        "623225bf24f67743e1e8990e02cebe8364191bcd17f89859c27213488ea009e4"
    )
    config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_runtime_amendment"
    ]["dependency_versions"]["numpy"] = "2.4.6"
    with pytest.raises(RuntimeError, match="E_TUPLE_RUNTIME_COMMITMENT"):
        MODULE._tuple_runtime_amendment(config)


def test_tuple_sizing_validation_freezes_exact_grounding_padding_semantics() -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    amendment = MODULE._tuple_sizing_validation(config)
    assert amendment["validation_commitment_sha256"] == (
        "afc936f742bd4313c35ff2e9a11a2389c589675c03309bbf09d8f8ab718ea2d5"
    )
    rule = amendment["grounding_dino_output_rule"]
    valid = {
        "raw_nan_count": 0,
        "raw_positive_infinity_count": 0,
        "raw_active_position_nonfinite_count": 0,
        "raw_padding_position_non_negative_infinity_count": 0,
        "post_sigmoid_nonfinite_count": 0,
        "pred_box_nonfinite_count": 0,
        "pred_box_min": 0.1,
        "pred_box_max": 0.9,
    }
    MODULE._validate_grounding_sizing_counts(valid, rule)
    for key, value in {
        "raw_active_position_nonfinite_count": 1,
        "raw_padding_position_non_negative_infinity_count": 1,
        "post_sigmoid_nonfinite_count": 1,
        "pred_box_nonfinite_count": 1,
        "pred_box_min": -0.01,
        "pred_box_max": 1.01,
    }.items():
        invalid = {**valid, key: value}
        with pytest.raises(RuntimeError, match="E_TUPLE_GROUNDING_NONFINITE"):
            MODULE._validate_grounding_sizing_counts(invalid, rule)
    config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_sizing_validation_amendment"
    ]["grounding_dino_output_rule"]["sizing_caption"] = "changed prompt."
    with pytest.raises(RuntimeError, match="E_TUPLE_SIZING_VALIDATION_COMMITMENT"):
        MODULE._tuple_sizing_validation(config)


def test_tuple_fixture_protocol_freezes_task_matched_action_semantics() -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    protocol = MODULE._tuple_fixture_protocol(config)
    assert protocol["protocol_commitment_sha256"] == (
        "506a1f41a3685ca777f3c9d23f6f9b884523acec2a78080d5de2547b3324251d"
    )
    action = protocol["order_dependent_action_control"]
    assert action["labels"] == [
        "open",
        "close",
        "take",
        "put",
        "sit_down",
        "stand_up",
        "turn_on",
        "turn_off",
    ]
    assert all(len(action["prompt_ensembles"][label]) == 3 for label in action["labels"])
    assert action["class_code_pairs"][0]["matched_codes"][0] == ["c008", "c006"]
    assert action["class_code_pairs"][2]["matched_codes"] == [["c151", "c154"]]
    assert action["class_code_pairs"][3]["matched_codes"] == [["c104", "c105"]]
    assert "prior 96 broad-context fixture" in action["eligible_interval"]
    config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_public_fixture_protocol"
    ]["order_dependent_action_control"]["labels"][0] = "opening"
    with pytest.raises(RuntimeError, match="E_TUPLE_FIXTURE_PROTOCOL_COMMITMENT"):
        MODULE._tuple_fixture_protocol(config)


def test_tuple_fixture_preparation_amendment_is_exact_and_preoutcome() -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    preparation = MODULE._tuple_fixture_preparation_amendment(config)
    assert preparation["preparation_amendment_commitment_sha256"] == (
        "1cc8d0e3498da5785a2c2105307bf6d5ab20dd10f839ec0f2b92b9def372ff1d"
    )
    assert preparation["counts_per_partition"] == {
        "language_lexical": 48,
        "referent_attribute": 64,
        "recurrence": 64,
        "hand_contact": 40,
        "sensor": 48,
        "order_action": 48,
    }
    assert preparation["source_archives"]["COCO_2017_instances"]["sha256"] == (
        "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
    )
    assert preparation["source_archives"]["COCO_2017_validation_images"]["sha256"] == (
        "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
    )
    assert preparation["execution"]["model_inference"] is False
    config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_fixture_preparation_amendment"
    ]["counts_per_partition"]["sensor"] = 47
    with pytest.raises(
        RuntimeError, match="E_TUPLE_FIXTURE_PREPARATION_COMMITMENT"
    ):
        MODULE._tuple_fixture_preparation_amendment(config)


def test_tuple_fixture_feasibility_repair_preserves_scientific_gates() -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    repair = MODULE._tuple_fixture_feasibility_repair(config)
    assert repair["fixture_feasibility_repair_commitment_sha256"] == (
        "e5fd286e9b8140583a37b855fe7125d7c6a0a2e3b57589b53294f77d28e47048"
    )
    assert repair["scientific_thresholds_changed"] is False
    assert repair["source_selection_repair"][
        "unchanged_target_bbox_minimum_pixels"
    ] == [48, 48]
    assert repair["source_selection_repair"][
        "active_COCO_area_fraction_range"
    ] == [0.0, 0.5]
    config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_fixture_feasibility_repair_amendment"
    ]["scientific_thresholds_changed"] = True
    with pytest.raises(
        RuntimeError, match="E_TUPLE_FIXTURE_FEASIBILITY_REPAIR_COMMITMENT"
    ):
        MODULE._tuple_fixture_feasibility_repair(config)


def test_tuple_visor_hos_correction_amendment_is_committed_and_guarded() -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    amendment = MODULE._tuple_visor_hos_correction_amendment(config)
    assert amendment["official_annotation_artifact"]["combined_JSON_file_count"] == 158
    assert amendment["official_annotation_artifact"]["combined_bytes"] == 868821446
    assert amendment["partition_and_joint_sampler"][
        "quota_per_partition_per_stratum"
    ] == 48
    assert amendment["partition_and_joint_sampler"]["per_video_per_stratum_cap"] == 4
    assert amendment["partition_and_joint_sampler"][
        "per_source_frame_cap_across_strata"
    ] == 1
    assert amendment["public_execution_and_combined_gate"]["critical_axes"] == list(
        MODULE.TUPLE_CRITICAL_AXIS_IDS
    )

    changed = json.loads(json.dumps(config))
    value = changed["calibration_C"]["extractor"][
        "mechanistic_training_tuple_visor_hos_correction_amendment"
    ]
    value["partition_and_joint_sampler"]["quota_per_partition_per_stratum"] = 47
    scope = json.loads(json.dumps(value))
    scope.pop("amendment_commitment_sha256")
    value["amendment_commitment_sha256"] = MODULE.digest(scope)
    with pytest.raises(RuntimeError, match="E_VISOR_HOS_CORRECTION_SCHEMA"):
        MODULE._tuple_visor_hos_correction_amendment(changed)


def test_task_matched_language_fixture_has_frozen_accept_and_abstention_mix() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    preparation = MODULE._tuple_fixture_preparation_amendment(config)
    for partition in preparation["partitions"]:
        rows = MODULE._language_lexical_fixture_rows(preparation, partition)
        assert len(rows) == 48
        assert sum(row["expected_pipeline_status"] == "ACCEPT" for row in rows) == 32
        assert sum(row["expected_pipeline_status"] == "ABSTAIN" for row in rows) == 16
        assert {
            row["expected_reason"]
            for row in rows
            if row["expected_pipeline_status"] == "ABSTAIN"
        } == {
            "LANGUAGE_MISMATCH",
            "EMPTY_ASR",
            "INVALID_TIMESTAMP",
            "LOW_CONFIDENCE",
            "EMPTY_TRANSLATION",
            "SILENT_TRUNCATION",
            "INSUFFICIENT_IN_BOUNDS_FRAMES",
            "ONTOLOGY_UNMATCHED",
        }


def test_public_fixture_partition_is_deterministic_and_namespace_scoped() -> None:
    first = MODULE._fixture_partition(20260802, "visor_partition", "P01")
    assert first == MODULE._fixture_partition(20260802, "visor_partition", "P01")
    assert first in {"development", "holdout"}
    assert MODULE._fixture_order(20260802, "a", "x") != MODULE._fixture_order(
        20260802, "b", "x"
    )


def test_charades_action_parser_rejects_invalid_or_reversed_intervals() -> None:
    import pytest

    assert MODULE._parse_charades_actions("c008 1.00 2.50;c006 3.0 4.0") == [
        {"code": "c008", "start": 1.0, "end": 2.5},
        {"code": "c006", "start": 3.0, "end": 4.0},
    ]
    with pytest.raises(RuntimeError, match="E_TUPLE_ACTION_ANNOTATION"):
        MODULE._parse_charades_actions("c008 2.0 1.0")


def test_visor_valid_negative_requires_valid_annotation_geometry() -> None:
    row = {
        "image": {
            "name": "frame.jpg",
            "image_path": "P01_01/frame.jpg",
            "video": "P01_01",
        },
        "annotations": [
            {
                "id": "object",
                "name": "cup",
                "segments": [[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0]]],
            }
        ],
    }
    truth = MODULE._visor_frame_truth(row)
    assert truth is not None
    assert truth["stratum"] == "true_no_hand"
    row["annotations"][0]["segments"] = [[[1.0, 1.0], [float("nan"), 1.0], [4.0, 4.0]]]
    assert MODULE._visor_frame_truth(row) is None


def test_visor_fixture_availability_reports_frozen_stratum_shortfall() -> None:
    participant = next(
        value
        for value in (f"P{index:02d}" for index in range(1, 100))
        if MODULE._fixture_partition(20260802, "visor_partition", value)
        == "development"
    )
    rows = []
    for index in range(6):
        rows.append(
            {
                "image": {
                    "name": f"frame-{index}.jpg",
                    "image_path": f"{participant}_01/frame-{index}.jpg",
                    "video": f"{participant}_01",
                },
                "annotations": [
                    {
                        "id": f"hand-{index}",
                        "name": "left hand",
                        "segments": [[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0]]],
                    }
                ],
            }
        )
    preparation = {
        "seed": 20260802,
        "partitions": ["development"],
        "visor_selection": {
            "strata_per_partition": {
                "hand_contact": 0,
                "hand_no_contact": 5,
                "true_no_hand": 0,
            }
        },
    }
    _selected, counts = MODULE._visor_fixture_availability(
        [{"video_annotations": rows}], preparation
    )
    assert counts["development"]["hand_no_contact"] == 4


def _visor_hos_row(
    participant: str,
    video_ordinal: int,
    frame_ordinal: int,
    relation: object,
    *,
    hand_side: str = "left hand",
) -> dict[str, object]:
    video = f"{participant}_{video_ordinal:02d}"
    annotations: list[dict[str, object]] = [
        {
            "id": f"hand-{frame_ordinal}",
            "name": hand_side,
            "segments": [[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0]]],
            "in_contact_object": relation,
        }
    ]
    if relation == "object":
        annotations.append(
            {
                "id": "object",
                "name": "cup",
                "segments": [[[5.0, 1.0], [8.0, 1.0], [8.0, 4.0]]],
            }
        )
    return {
        "image": {
            "name": f"frame-{frame_ordinal:04d}.jpg",
            "image_path": f"{video}/frame-{frame_ordinal:04d}.jpg",
            "video": video,
        },
        "annotations": annotations,
    }


def _visor_hos_no_hand_row(
    participant: str, video_ordinal: int, frame_ordinal: int
) -> dict[str, object]:
    video = f"{participant}_{video_ordinal:02d}"
    return {
        "image": {
            "name": f"frame-{frame_ordinal:04d}.jpg",
            "image_path": f"{video}/frame-{frame_ordinal:04d}.jpg",
            "video": video,
        },
        "annotations": [
            {
                "id": f"object-{frame_ordinal}",
                "name": "cup",
                "segments": [[[5.0, 1.0], [8.0, 1.0], [8.0, 4.0]]],
            }
        ],
    }


def test_visor_hos_contact_truth_uses_explicit_labels_and_abstains() -> None:
    contact_row = _visor_hos_row("P01", 1, 1, "object")
    hand = contact_row["annotations"][0]
    assert MODULE._visor_hos_contact_truth(hand, contact_row["annotations"]) == {
        "status": "MEASURED",
        "reason": None,
        "contact_state": "contact",
        "contact": True,
    }
    hand["in_contact_object"] = "hand-not-in-contact"
    assert MODULE._visor_hos_contact_truth(hand, contact_row["annotations"])[
        "contact_state"
    ] == "no_contact"
    for invalid in (None, True, False, "none-of-the-above", "inconclusive", "none"):
        hand["in_contact_object"] = invalid
        assert (
            MODULE._visor_hos_contact_truth(hand, contact_row["annotations"])[
                "status"
            ]
            == "ABSTAIN"
        )
    hand.pop("in_contact_object")
    assert MODULE._visor_hos_contact_truth(hand, contact_row["annotations"]) == {
        "status": "ABSTAIN",
        "reason": "MISSING_CONTACT_LABEL",
    }


def test_visor_hos_no_hand_is_separate_from_contact_state() -> None:
    parsed = MODULE._visor_hos_frame_candidates(_visor_hos_no_hand_row("P01", 1, 1))
    assert parsed["status"] == "NOMINEE"
    assert parsed["candidates"][0]["stratum"] == "no_hand_nominee"
    assert parsed["candidates"][0]["contact"] is None
    assert parsed["candidates"][0]["hand_visible"] is False
    glove = _visor_hos_no_hand_row("P01", 1, 2)
    glove["annotations"][0].update(
        {"name": "glove", "on_which_hand": ["left hand"]}
    )
    parsed_glove = MODULE._visor_hos_frame_candidates(glove)
    assert parsed_glove["status"] == "ABSTAIN"
    assert parsed_glove["reason"] == "ON_HAND_GLOVE_WITHOUT_VISIBLE_HAND"


def test_visor_hos_joint_sampler_is_order_invariant_and_collects_deficits() -> None:
    rows = []
    for participant_index in range(1, 9):
        participant = f"P{participant_index:02d}"
        for stratum_ordinal, relation in enumerate(
            ("object", "hand-not-in-contact", None)
        ):
            for video_offset in range(2):
                for frame_offset in range(4):
                    frame = stratum_ordinal * 100 + video_offset * 10 + frame_offset
                    if relation is None:
                        rows.append(
                            _visor_hos_no_hand_row(
                                participant, stratum_ordinal * 2 + video_offset, frame
                            )
                        )
                    else:
                        rows.append(
                            _visor_hos_row(
                                participant,
                                stratum_ordinal * 2 + video_offset,
                                frame,
                                relation,
                            )
                        )
    verified_no_hand_frames = {
        (row["image"]["video"], row["image"]["name"])
        for row in rows
        if not any(
            item["name"] in {"left hand", "right hand"}
            for item in row["annotations"]
        )
    }
    first, first_report = MODULE._visor_hos_joint_sampler(
        [{"video_annotations": rows}],
        seed=20260802,
        target_per_stratum=12,
        verified_no_hand_frames=verified_no_hand_frames,
    )
    second, second_report = MODULE._visor_hos_joint_sampler(
        [{"video_annotations": list(reversed(rows))}],
        seed=20260802,
        target_per_stratum=12,
        verified_no_hand_frames=verified_no_hand_frames,
    )
    compact = lambda value: [
        (row["video"], row["frame_name"], row["stratum"])
        for partition in ("development", "holdout")
        for row in value[partition]
    ]
    assert compact(first) == compact(second)
    assert first_report == second_report
    assert first_report["status"] == "PASS"
    assert first_report["participant_overlap_count"] == 0
    assert first_report["video_overlap_count"] == 0
    assert first_report["frame_overlap_count"] == 0
    for partition in ("development", "holdout"):
        assert first_report["final_counts"][partition] == {
            "contact": 12,
            "explicit_no_contact": 12,
            "verified_no_hand": 12,
        }
        frames = [
            (row["video"], row["frame_name"]) for row in first[partition]
        ]
        assert len(frames) == len(set(frames))
        for stratum in MODULE.VISOR_HOS_STRATA:
            video_counts = {
                video: sum(
                    row["video"] == video and row["stratum"] == stratum
                    for row in first[partition]
                )
                for video in {row["video"] for row in first[partition]}
            }
            assert max(video_counts.values()) <= 4

    _selected, deficient = MODULE._visor_hos_joint_sampler(
        [{"video_annotations": rows[:4]}], seed=20260802, target_per_stratum=3
    )
    assert deficient["status"] == "NO_GO"
    assert len(deficient["deficits"]) >= 3
    assert {item["partition"] for item in deficient["deficits"]} == {
        "development",
        "holdout",
    }
    _selected, empty = MODULE._visor_hos_joint_sampler(
        [], seed=20260802, target_per_stratum=3
    )
    assert len(empty["deficits"]) == 6


def test_visor_hos_no_hand_requires_external_verification_and_honors_exclusion() -> None:
    first = _visor_hos_no_hand_row("P01", 1, 1)
    second = _visor_hos_no_hand_row("P02", 1, 2)
    contact = _visor_hos_row("P01", 2, 3, "object")
    selected, report = MODULE._visor_hos_joint_sampler(
        [{"video_annotations": [first, second, contact]}],
        seed=20260802,
        target_per_stratum=1,
        verified_no_hand_frames={("P01_01", "frame-0001.jpg")},
        correction_excluded_frame_keys={("P01_02", "frame-0003.jpg")},
    )
    retained_no_hand = [
        row
        for partition in selected.values()
        for row in partition
        if row["stratum"] == "verified_no_hand"
    ]
    assert len(retained_no_hand) == 1
    assert retained_no_hand[0]["frame_name"] == "frame-0001.jpg"
    assert report["no_hand_nominee_count"] == 2
    assert report["verified_no_hand_input_count"] == 1
    assert report["matched_verified_no_hand_count"] == 1
    assert report["unverified_no_hand_nominee_count"] == 1
    assert report["correction_excluded_frame_count"] == 1
    assert all(
        row["frame_name"] != "frame-0003.jpg"
        for partition in selected.values()
        for row in partition
    )


def test_tuple_combined_public_gate_collects_all_failures() -> None:
    axes = {
        axis: {"status": "PASS"}
        for axis in (
            *MODULE.TUPLE_CRITICAL_AXIS_IDS,
            *MODULE.TUPLE_SUPPORTING_AXIS_IDS,
        )
    }
    axes[MODULE.TUPLE_CRITICAL_AXIS_IDS[0]] = {"status": "FAIL"}
    axes[MODULE.TUPLE_CRITICAL_AXIS_IDS[1]] = {"status": "UNMEASURED"}
    result = MODULE._tuple_combined_public_gate(
        axes,
        {"status": "FAIL"},
        {"status": "FAIL"},
    )
    assert result["status"] == "NO_GO"
    assert result["critical_axis_failures"] == [
        MODULE.TUPLE_CRITICAL_AXIS_IDS[0],
        MODULE.TUPLE_CRITICAL_AXIS_IDS[1],
    ]
    assert result["validated_axis_count"] == 5
    assert result["combined_gate_failures"] == [
        f"critical_axis:{MODULE.TUPLE_CRITICAL_AXIS_IDS[0]}",
        f"critical_axis:{MODULE.TUPLE_CRITICAL_AXIS_IDS[1]}",
        "validated_axes_minimum:6_of_7",
        "order_dependent_action_control",
    ]
    assert result["broad_activity_used_in_gate"] is False


def test_tuple_combined_public_gate_allows_one_supporting_axis_unmeasured() -> None:
    axes = {
        axis: {"status": "PASS"}
        for axis in (
            *MODULE.TUPLE_CRITICAL_AXIS_IDS,
            *MODULE.TUPLE_SUPPORTING_AXIS_IDS,
        )
    }
    axes[MODULE.TUPLE_SUPPORTING_AXIS_IDS[0]] = {"status": "UNMEASURED"}
    first = MODULE._tuple_combined_public_gate(
        axes, {"status": "PASS"}, {"status": "FAIL"}
    )
    second = MODULE._tuple_combined_public_gate(
        axes, {"status": "PASS"}, {"status": "PASS"}
    )
    assert first["status"] == second["status"] == "PASS"
    assert first["validated_axis_count"] == 6


def test_tuple_zip_extraction_blocks_path_traversal(tmp_path: Path) -> None:
    import pytest
    import zipfile

    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape", "blocked")
    with pytest.raises(RuntimeError, match="E_TUPLE_ARCHIVE_PATH"):
        MODULE._safe_extract_zip(archive, tmp_path / "output")


def test_tuple_nltk_namespace_is_scratch_only_and_hash_guarded(tmp_path: Path) -> None:
    import pytest

    public = tmp_path / "public"
    resource = public / "models/nltk_data"
    tagger = resource / "averaged_perceptron_tagger_eng/model.json"
    wordnet = resource / "wordnet/index.noun"
    tagger.parent.mkdir(parents=True)
    wordnet.parent.mkdir(parents=True)
    tagger.write_text("{}")
    wordnet.write_text("noun")
    payload = {
        "nltk_resource_files": [
            {
                "relative_path": "averaged_perceptron_tagger_eng/model.json",
                "sha256": MODULE.file_digest(tagger),
            },
            {
                "relative_path": "wordnet/index.noun",
                "sha256": MODULE.file_digest(wordnet),
            },
        ]
    }
    commitment = MODULE.digest(payload)
    manifest = {**payload, "tuple_dependency_commitment_sha256": commitment}
    manifest_path = MODULE._tuple_run_root(public) / "dependency_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "calibration_C": {
            "extractor": {
                "mechanistic_training_tuple_premodel_result": {
                    "dependency_manifest_commitment_sha256": commitment
                }
            }
        }
    }
    staged = MODULE._stage_tuple_nltk_resources(public, tmp_path / "scratch", config)
    assert (staged / "taggers/averaged_perceptron_tagger_eng").is_symlink()
    assert (staged / "corpora/wordnet").is_symlink()
    assert not (resource / "taggers").exists()
    wordnet.write_text("tampered")
    with pytest.raises(RuntimeError, match="E_TUPLE_NLTK_RESOURCE_HASH"):
        MODULE._stage_tuple_nltk_resources(public, tmp_path / "scratch-2", config)


def test_tuple_repository_archive_records_bytes_and_commit(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as handle:
        handle.writestr("repo-abc/LICENSE", "public license")
        handle.writestr("repo-abc/module.py", "value = 1\n")

    def copy_archive(_url: str, target: Path) -> None:
        shutil.copyfile(source, target)

    monkeypatch.setattr(MODULE, "_download_public_artifact", copy_archive)
    target = tmp_path / "repo"
    record = MODULE._clone_public_repository(
        "https://github.com/example/repo.git", "abc", target
    )
    assert record == {
        "archive_sha256": MODULE.file_digest(source),
        "archive_bytes": source.stat().st_size,
    }
    assert (target / ".source-commit").read_text().strip() == "abc"
    assert (target / "module.py").read_text() == "value = 1\n"


def test_tuple_download_resumes_only_on_partial_content(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"x" * (1024 * 1024 + 1)
    target = tmp_path / "artifact.bin"
    partial = tmp_path / "artifact.bin.partial"
    partial.write_bytes(b"prefix")

    class Response(BytesIO):
        status = 206

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        MODULE.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(payload),
    )
    MODULE._download_public_artifact("https://public.invalid/artifact", target)
    assert target.read_bytes() == b"prefix" + payload


def test_tuple_public_file_download_retries_transient_network_failure(
    tmp_path: Path, monkeypatch
) -> None:
    attempts = []

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def open_with_failure(_request, timeout):
        assert timeout == 300
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("temporary DNS failure")
        return Response(b"public fixture")

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", open_with_failure)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    target = tmp_path / "fixture.json"
    MODULE._download_public_file("https://example.test/fixture.json", target)
    assert len(attempts) == 2
    assert target.read_bytes() == b"public fixture"


def test_tuple_exact_download_accepts_small_hashed_public_files(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"public tokenizer configuration"
    expected = MODULE.hashlib.sha256(payload).hexdigest()

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        MODULE.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(payload),
    )
    target = tmp_path / "config.json"
    MODULE._download_exact_public_artifact(
        "https://public.invalid/config", target, expected
    )
    assert target.read_bytes() == payload


def test_tuple_public_prep_pins_language_and_vision_resources() -> None:
    import inspect

    source = inspect.getsource(MODULE.prepare_tuple_public)
    assert MODULE.NLTK_DATA_COMMIT == "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
    assert MODULE.NLTK_RESOURCE_ARCHIVES == {
        "wordnet.zip": {
            "relative_url": "packages/corpora/wordnet.zip",
            "sha256": "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59",
        },
        "averaged_perceptron_tagger_eng.zip": {
            "relative_url": "packages/taggers/averaged_perceptron_tagger_eng.zip",
            "sha256": "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b",
        },
    }
    assert 'weight_root / "PE-Core-L14-336.pt"' in source
    assert "huggingface.co/" in source
    assert "hf_hub_download" not in source
    assert '"nltk": "3.9.1"' in source
    assert '"wordfreq": "3.0.2"' in source


def test_tuple_runtime_prep_is_resource_only_and_hash_sealed() -> None:
    import inspect

    source = inspect.getsource(MODULE.prepare_tuple_runtime)
    assert "mechanistic_training_tuple_premodel_result" in source
    assert '"--no-deps"' in source
    assert '"--no-build-isolation"' in source
    assert '"MMCV_WITH_OPS": "0"' in source
    assert '"SAM2_BUILD_CUDA": "0"' in source
    assert '"fvcore", "iopath", "mmcv"' in source
    assert '"model_inference_executed": False' in source
    assert "runtime_dependency_commitment_sha256" in source
    assert "E_TUPLE_RUNTIME_ADDED_DEPENDENCY_HASH" in source
    assert 'runtime["added_dependency_wheels"]' in source
    assert "_apply_grounding_dino_fallback_patch" in source
    assert "restricted_root" not in source


def test_tuple_sizing_is_label_blind_and_retains_no_predictions() -> None:
    import inspect

    source = inspect.getsource(MODULE.size_tuple_runtime)
    assert '"fixture_labels_used": False' in source
    assert '"scientific_metric_computed": False' in source
    assert '"prediction_or_score_retained": False' in source
    assert '"external_call_count": 0' in source
    assert '"module_count"' in source
    assert "_verify_activity_dependency_manifest" in source
    assert MODULE.TUPLE_LANGUAGE_ADAPTER_SHA256 == (
        "005f368bef97dfc791f43e45da8bbfe01ea22e8790b2032e9580b14b1ea62ac8"
    )
    assert "E_TUPLE_LANGUAGE_ADAPTER_SOURCE" in source
    assert "_stage_tuple_nltk_resources" in source
    assert "_tuple_fixture_protocol" in source
    assert "_tuple_sizing_validation" in source
    assert "sizing_validation_commitment_sha256" in source
    assert "prompt_groups_override=action_protocol" in source
    assert "public_fixture_protocol_commitment_sha256" in source
    assert "activity['development_selection_result']" not in source
    assert "restricted_root" not in source


def test_grounding_fallback_patch_is_narrow_and_extension_conditional() -> None:
    import inspect

    source = inspect.getsource(MODULE._apply_grounding_dino_fallback_patch)
    assert MODULE.GROUNDING_DINO_DEFORM_ATTN_SOURCE_SHA256 == (
        "42aa71c7c47e6f930f48100924393adac95eb94aae0eef779bd7cad2d5bcc95d"
    )
    assert MODULE.GROUNDING_DINO_DEFORM_ATTN_PATCHED_SHA256 == (
        "778efabd5d875a4aa457ede6948979a4196844fbd14bf7a76bc4d4b1440122c6"
    )
    assert MODULE.GROUNDING_DINO_MODEL_SOURCE_SHA256 == (
        "cdfb48d5b15d6b98f3d2002f59ae4730740a1ecfbaeba324f6840c5e4666a5b8"
    )
    assert MODULE.GROUNDING_DINO_MODEL_NO_VISUALIZER_SHA256 == (
        "0da7cea7ddbaddced76432d7a8bc13844dc69d3bee3ce5ae674c46fd0339c671"
    )
    assert '"if torch.cuda.is_available() and value.is_cuda:"' in source
    assert "if '_C' in globals() and torch.cuda.is_available() and value.is_cuda:" in source
    assert "text.count(original) != 1" in source
    assert 'model_text.count("COCOVisualizer") != 1' in source
    assert "E_TUPLE_GROUNDING_VISUALIZER_PATCH_STATE" in source


def test_tuple_window_uses_segment_midpoint_and_never_fabricates_word_time() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    amendment = MODULE._tuple_amendment(config)
    segment = {
        "status": "ACCEPT",
        "start": 4.0,
        "end": 5.0,
        "en": "The red ball",
        "words": [{"word": "der", "start": 4.0, "end": 4.4}],
    }
    result = MODULE._tuple_segment_window(segment, 9.0, amendment)
    assert result["status"] == "ACCEPT"
    assert result["mention_anchor"] == 4.5
    assert result["samples"] == {
        "before": [1.5, 2.5, 3.5],
        "during": [4.166667, 4.5, 4.833333],
        "after": [5.5, 6.5, 7.5],
    }
    assert "words" not in result


def test_tuple_window_abstains_at_boundary_and_on_adapter_rejection() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    amendment = MODULE._tuple_amendment(config)
    edge = {"status": "ACCEPT", "start": 0.0, "end": 1.0, "en": "ball"}
    assert MODULE._tuple_segment_window(edge, 8.0, amendment)["reason"] == (
        "INSUFFICIENT_IN_BOUNDS_FRAMES"
    )
    assert MODULE._tuple_segment_window(
        {"status": "ABSTAIN", "reason": "LOW_CONFIDENCE"}, 8.0, amendment
    ) == {"status": "ABSTAIN", "reason": "ADAPTER_ABSTAIN"}


def test_lexical_mentions_preserve_roundtrip_lemma_pos_and_band() -> None:
    tags = {"Red": "JJ", "balls": "NNS", "roll": "VBP"}
    result = MODULE._lexical_mentions(
        "Red balls roll",
        lambda tokens: [(token, tags[token]) for token in tokens],
        lambda token, pos: "ball" if (token, pos) == ("balls", "n") else token,
        lambda lemma, language: {"red": 4.5, "ball": 3.5}[lemma],
        {"low": [0.0, 3.0], "mid": [3.0, 4.0], "high": [4.0, 8.0]},
    )
    assert [(row["lemma"], row["part_of_speech"], row["frequency_band"]) for row in result] == [
        ("red", "adjective", "high"),
        ("ball", "noun", "mid"),
    ]


def test_public_ontology_abstains_on_unmatched_or_ambiguous_lemma() -> None:
    mapping = {"toy": ["ball", "block"], "container": ["cup", "block"]}
    assert MODULE._map_public_ontology("ball", mapping) == {
        "status": "ACCEPT",
        "reason": None,
        "category": "toy",
    }
    assert MODULE._map_public_ontology("unknown", mapping)["reason"] == "ONTOLOGY_UNMATCHED"
    assert MODULE._map_public_ontology("block", mapping)["reason"] == "ONTOLOGY_AMBIGUOUS"


def test_referent_proxy_requires_qualified_specificity_for_valid_negative() -> None:
    definitions = {
        "visible_mask_fraction_min": 0.01,
        "dominant_target_mask_fraction_min": 0.05,
        "dominant_area_ratio_to_next_candidate_min": 2.0,
    }
    assert MODULE._referent_frame_proxy(
        [], "toy", definitions, inference_succeeded=True, negative_specificity_passed=False
    )["reason"] == "NEGATIVE_NOT_QUALIFIED"
    assert MODULE._referent_frame_proxy(
        [], "toy", definitions, inference_succeeded=True, negative_specificity_passed=True
    )["status"] == "MEASURED_NEGATIVE"


def test_referent_proxy_rejects_bad_geometry_and_measures_dominance() -> None:
    definitions = {
        "visible_mask_fraction_min": 0.01,
        "dominant_target_mask_fraction_min": 0.05,
        "dominant_area_ratio_to_next_candidate_min": 2.0,
    }
    invalid = [{"category": "toy", "mask_fraction": 0.2, "center_distance": 0.1, "box": [0, 0, 2, 1]}]
    assert MODULE._referent_frame_proxy(
        invalid, "toy", definitions, inference_succeeded=True, negative_specificity_passed=True
    )["reason"] == "INVALID_GEOMETRY"
    candidates = [
        {"category": "toy", "mask_fraction": 0.2, "center_distance": 0.1, "box": [0.2, 0.2, 0.6, 0.7]},
        {"category": "distractor", "mask_fraction": 0.05, "center_distance": 0.4, "box": [0.7, 0.2, 0.9, 0.5]},
    ]
    measured = MODULE._referent_frame_proxy(
        candidates, "toy", definitions, inference_succeeded=True, negative_specificity_passed=True
    )
    assert measured["status"] == "MEASURED_POSITIVE"
    assert measured["dominant"] is True
    assert measured["candidate_count_bin"] == "2plus"


def test_track_and_recurrence_guards_abstain_instead_of_imputing() -> None:
    assert MODULE._validate_monotonic_track(
        [
            {"time": 0.1, "box": [0.1, 0.1, 0.2, 0.2]},
            {"time": 0.2, "box": [0.2, 0.1, 0.3, 0.2]},
        ]
    )
    assert not MODULE._validate_monotonic_track(
        [
            {"time": 0.2, "box": [0.1, 0.1, 0.2, 0.2]},
            {"time": 0.1, "box": [0.2, 0.1, 0.3, 0.2]},
        ]
    )
    invalid = MODULE._recurrence_decision(
        float("nan"), 0.9, exact_duplicate=False, perceptual_duplicate=False
    )
    assert invalid == {"status": "ABSTAIN", "reason": "INVALID_SIMILARITY"}
    same = MODULE._recurrence_decision(
        0.95, 0.9, exact_duplicate=False, perceptual_duplicate=True
    )
    assert same["same_referent"] is True
    assert same["visual_variation"] is False


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
    (code / "utils.py").write_text("def load(): return gfile.GFile('hosted')\n")
    names = {
        "videoprism.tokenizers",
        "tensorflow",
        "tensorflow.io",
        "tensorflow.io.gfile",
    }
    previous = {name: sys.modules.get(name) for name in names}
    try:
        MODULE._install_videoprism_tokenizer_import_compatibility(tmp_path)
        tokenizers = sys.modules["videoprism.tokenizers"]
        import pytest

        with pytest.raises(
            RuntimeError, match="E_VIDEOPRISM_HOSTED_TOKENIZER_PATH_PROHIBITED"
        ):
            tokenizers.SentencePieceTokenizer("hosted")
        with pytest.raises(
            RuntimeError, match="E_VIDEOPRISM_HOSTED_GFILE_PATH_PROHIBITED"
        ):
            sys.modules["tensorflow.io.gfile"].GFile("hosted")
        MODULE._remove_videoprism_tensorflow_import_compatibility()
        assert "tensorflow" not in sys.modules
        assert "tensorflow.io" not in sys.modules
        assert "tensorflow.io.gfile" not in sys.modules
        (code / "models.py").write_text("x = tokenizers.new_surface\n")
        with pytest.raises(
            RuntimeError, match="E_VIDEOPRISM_TOKENIZER_IMPORT_SURFACE"
        ):
            MODULE._install_videoprism_tokenizer_import_compatibility(tmp_path)
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


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
    assert "tuple-prepare" in prepare
    assert "mechanistic-tuple" in prepare
    assert "tuple-runtime-prepare" in prepare
    assert "mechanistic-tuple-runtime" in prepare
    assert "tuple-fixtures-prepare" in prepare
    assert "mechanistic-tuple-fixtures" in prepare
    assert "tuple-fixtures-feasibility" in prepare
    assert "mechanistic-tuple-fixture-feasibility" in prepare
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
    assert "PHASE4_TUPLE_RUN_MODE" in qualify
    assert "tuple-size" in qualify


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
    assert "TUPLE_PREP_FIELDS" in source
    assert "TUPLE_RUNTIME_PREP_FIELDS" in source
    assert "TUPLE_SIZING_FIELDS" in source
    assert "TUPLE_FIXTURE_PREP_FIELDS" in source
    assert "tuple-prepare" in source
    assert "tuple-runtime-prepare" in source
    assert "tuple-size" in source
    assert "tuple-audio-seed" in source
    assert "tuple-fixtures-prepare" in source
    assert "tuple-fixtures-feasibility" in source
    assert "E_ACTIVITY_HOLDOUT_BEFORE_WINNER_SEAL" in source
    assert 'print(json.dumps' not in source
