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


def _current_audio_fixture_config() -> dict:
    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    amendment = config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_visor_hos_correction_amendment"
    ]
    scope = json.loads(json.dumps(amendment))
    scope.pop("amendment_commitment_sha256")
    amendment["amendment_commitment_sha256"] = MODULE.digest(scope)
    return config


def test_construct_aligned_resume_amendment_is_exact_and_schema23_compatible() -> None:
    import pytest

    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    assert config["schema_version"] == 26
    amendment = MODULE._construct_aligned_ltx_resume_amendment(config)
    assert (
        amendment["amendment_commitment_sha256"]
        == MODULE.CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256
    )
    assert amendment["source_reuse"]["action_fixture_counts"] == {
        "development": 44,
        "holdout": 44,
    }
    historical = MODULE._tuple_visor_hos_correction_amendment(config)
    assert historical["public_fixture_counts_per_partition"][
        "order_dependent_action_clips"
    ] == 48

    mutated = json.loads(json.dumps(config))
    active = mutated["construct_aligned_ltx_resume_amendment"]
    active["source_reuse"]["action_fixture_counts"]["development"] = 45
    payload = json.loads(json.dumps(active))
    payload.pop("amendment_commitment_sha256")
    active["amendment_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(RuntimeError, match="E_CONSTRUCT_ALIGNED_RESUME_COMMITMENT"):
        MODULE._construct_aligned_ltx_resume_amendment(mutated)


def test_engineering_health_amendment_and_geometry_lineage_are_exact() -> None:
    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    amendment = MODULE._engineering_health_amendment(config)
    assert amendment["amendment_commitment_sha256"] == (
        "d447a7e165136032a1fba43605d3f81881b41ec030c82e9028e1a8f5cb2c6205"
    )
    assert amendment["prior_public_development_result"][
        "public_qualification_commitment_sha256"
    ] == "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
    assert amendment["engineering_microfixture_suite"]["total_case_count"] == 28
    redirect = MODULE._engineering_health_resource_redirect(config)
    assert redirect["amendment_commitment_sha256"] == (
        "f7fc16f5c399c2a2d213b13a0d255a14b5b2f3ece41d62adaed17f61f186db6d"
    )
    assert redirect["canceled_A30_submission"]["job_id"] == 316158
    assert redirect["canceled_A30_submission"]["elapsed_seconds"] == 0
    assert redirect["active_health_topology"]["GRES"] == (
        "gpu:nvidia_h100_nvl_3g.47gb:1"
    )
    assert MODULE._engineering_health_resource_policy(config)["GPU_type"] == (
        "NVIDIA_H100_NVL_3G_47GB_MIG"
    )
    compatibility = amendment["historical_geometry_compatibility"]
    source_sha256, ast_sha256 = MODULE._geometry_function_bundle_digests(
        Path("scripts/run_synthetic_video_calibration.py"),
        compatibility["function_names"],
    )
    assert source_sha256 == compatibility["exact_function_source_bundle_sha256"]
    assert ast_sha256 == compatibility["canonical_AST_bundle_sha256"]
    assert MODULE._public_fixture_geometry_rasterization_repair(config)[
        "repair_commitment_sha256"
    ] == "6084fd937c208feda00aa3dc1cf14d0ec56e8f13bd24b56e23e4a6a6553e61ef"


def test_pe_core_resolver_accepts_identical_copies_and_prefers_canonical(
    tmp_path, monkeypatch
) -> None:
    import pytest

    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    frozen = config["calibration_C"]["extractor"]["vision_model"]
    canonical = (
        tmp_path / "models/mechanistic-tuples/weights/PE-Core-L14-336.pt"
    )
    cached = (
        tmp_path
        / "models/pe-hf-home/hub/models--facebook--PE-Core-L14-336/snapshots"
        / frozen["revision"]
        / "PE-Core-L14-336.pt"
    )
    canonical.parent.mkdir(parents=True)
    cached.parent.mkdir(parents=True)
    canonical.write_bytes(b"byte-identical frozen copy")
    cached.write_bytes(b"byte-identical frozen copy")
    observed = {
        canonical: frozen["weights_sha256"],
        cached: frozen["weights_sha256"],
    }
    monkeypatch.setattr(MODULE, "file_digest", lambda path: observed[path])

    assert MODULE._resolve_tuple_pe_core_checkpoint(tmp_path, config) == canonical

    observed[canonical] = "0" * 64
    with pytest.raises(RuntimeError, match="E_FROZEN_VISION_MODEL"):
        MODULE._resolve_tuple_pe_core_checkpoint(tmp_path, config)

    canonical.unlink()
    assert MODULE._resolve_tuple_pe_core_checkpoint(tmp_path, config) == cached

    altered = json.loads(json.dumps(config))
    altered["calibration_C"]["extractor"]["vision_model"]["revision"] = "0" * 40
    with pytest.raises(RuntimeError, match="E_FROZEN_VISION_MODEL"):
        MODULE._resolve_tuple_pe_core_checkpoint(tmp_path, altered)

    cached.unlink()
    with pytest.raises(RuntimeError, match="E_FROZEN_VISION_MODEL"):
        MODULE._resolve_tuple_pe_core_checkpoint(tmp_path, config)


def test_order_action_uses_exact_frozen_egohod_without_activity_selection() -> None:
    import inspect
    import pytest

    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    activity = config["calibration_C"]["extractor"][
        "activity_checkpoint_selection_amendment"
    ]
    assert activity["status"] == "PUBLIC_DEVELOPMENT_NO_GO_NO_ELIGIBLE_CANDIDATE"
    candidate, runtime = MODULE._tuple_order_action_egohod_wiring(config)
    assert candidate["candidate_id"] == "egohod_egovideo_l_zero_shot"
    assert candidate["weight_sha256"] == (
        "71faa0b6e5ebfb912238a099b16b1ff8b6b0a74cbb5b9eb43d5ad8bc92f880da"
    )
    assert runtime["input_frames"] == 16
    source = inspect.getsource(MODULE._tuple_order_action_module)
    assert "_tuple_order_action_egohod_wiring" in source
    assert "_activity_config" not in source

    altered = json.loads(json.dumps(config))
    candidates = altered["calibration_C"]["extractor"][
        "activity_checkpoint_selection_amendment"
    ]["bounded_candidates"]
    next(
        value
        for value in candidates
        if value["candidate_id"] == "egohod_egovideo_l_zero_shot"
    )["weight_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_ACTION_EGOHOD_IDENTITY"):
        MODULE._tuple_order_action_egohod_wiring(altered)


def test_public_fixture_geometry_rasterization_repair_is_frozen() -> None:
    import pytest

    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    repair = MODULE._public_fixture_geometry_rasterization_repair(config)
    assert (
        repair["repair_commitment_sha256"]
        == "6084fd937c208feda00aa3dc1cf14d0ec56e8f13bd24b56e23e4a6a6553e61ef"
    )
    assert repair["triggering_attempt"]["job_id"] == 315462
    assert repair["triggering_attempt"]["public_model_inference_executed"] is False
    assert repair["rasterization_semantics"]["manual_clip"] is False
    assert repair["rasterization_semantics"]["pixel_tolerance"] is None
    assert repair["scientific_thresholds_changed"] is False
    assert repair["source_selection_changed"] is False

    mutated = json.loads(json.dumps(config))
    mutated["public_fixture_geometry_rasterization_repair"][
        "scientific_thresholds_changed"
    ] = True
    with pytest.raises(
        RuntimeError, match="E_TUPLE_VISOR_RASTERIZATION_REPAIR_COMMITMENT"
    ):
        MODULE._public_fixture_geometry_rasterization_repair(mutated)


def _write_public_audio_seed(
    tmp_path: Path,
    config: dict,
    *,
    predicative: bool = False,
    stale: bool = False,
) -> Path:
    preparation = MODULE._tuple_fixture_preparation_amendment(config)
    correction = MODULE._tuple_visor_hos_correction_amendment(config)
    recipe = MODULE._tuple_referent_audio_fixture(config)
    root = tmp_path / recipe["active_external_location"]
    root.mkdir(parents=True)
    nouns = {
        "sports ball": "Ball",
        "cup": "Becher",
        "bottle": "Flasche",
        "bowl": "Schüssel",
        "book": "Buch",
        "chair": "Stuhl",
        "apple": "Apfel",
        "banana": "Banane",
    }
    scenarios = preparation["referent_attribute_rendering"][
        "scenarios_once_per_category"
    ]
    records = []
    for partition in preparation["partitions"]:
        for category in preparation["public_object_ontology"]:
            for ordinal, scenario in enumerate(scenarios):
                if scenario == "no_speech_visible_object":
                    continue
                slug = category.replace(" ", "-")
                relative = Path(partition) / f"{slug}-{ordinal:02d}.aiff"
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f"public-audio-{partition}-{slug}-{ordinal}".encode())
                phrase = MODULE._tuple_audio_phrase(
                    recipe, partition, category, ordinal, nouns[category]
                )
                if predicative:
                    phrase = f"{nouns[category]} ist rot."
                records.append(
                    {
                        "partition": partition,
                        "category": category,
                        "scenario": scenario,
                        "ordinal": ordinal,
                        "phrase_de": phrase,
                        "relative_path": str(relative),
                        "sha256": MODULE.file_digest(target),
                        "bytes": target.stat().st_size,
                    }
                )
    manifest = {
        "schema_version": 2,
        "status": "SEALED_SELF_AUTHORED_PUBLIC_AUDIO_SEED",
        "preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "visor_hos_correction_amendment_commitment_sha256": correction[
            "amendment_commitment_sha256"
        ],
        "referent_audio_fixture_recipe": recipe,
        "referent_audio_fixture_recipe_commitment_sha256": MODULE.digest(recipe),
        "active_external_location": recipe["active_external_location"],
        "license": "CC0-1.0 self-authored text and rendered fixture audio",
        "language": "de",
        "voice": "macOS Anna de_DE",
        "rate_words_per_minute": 175,
        "muxed_speech_delay_seconds": recipe["muxed_speech_delay_seconds"],
        "maximum_spoken_audio_seconds": recipe["maximum_spoken_audio_seconds"],
        "platform_version": "public-test",
        "say_binary_sha256": "0" * 64,
        "audio_file_count": len(records),
        "records": records,
    }
    if stale:
        manifest["schema_version"] = 1
        manifest.pop("visor_hos_correction_amendment_commitment_sha256")
        manifest.pop("referent_audio_fixture_recipe")
        manifest.pop("referent_audio_fixture_recipe_commitment_sha256")
        manifest.pop("active_external_location")
    manifest["audio_seed_commitment_sha256"] = MODULE.digest(manifest)
    (root / "audio-seed-manifest.json").write_text(json.dumps(manifest))
    return root


def test_active_tuple_protocol_blocks_legacy_broad_governed_run(tmp_path: Path) -> None:
    import pytest
    from types import SimpleNamespace

    config_path = tmp_path / "proof.json"
    config_path.write_text(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    args = SimpleNamespace(config=config_path)
    with pytest.raises(
        RuntimeError,
        match="E_LEGACY_BROAD_CALIBRATION_SUPERSEDED_BY_ACTIVE_TUPLE_PROTOCOL",
    ):
        MODULE.run(args)


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
    assert amendment["official_semantic_reference"]["gen_coco_format_py_sha256"] == (
        "686a052c8676c8378438efcf90e97e71cd6abca576381b0ca560e6cb07759cd7"
    )
    assert amendment["official_semantic_reference"][
        "gen_coco_format_handside_contact_py_sha256"
    ] == "44feea718164ed171ee6cb24eb90cde402e429eb920c4ce728c00492b79084f6"
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
    execution = amendment["qualification_execution_clarification"]
    assert execution["phase_aggregation"]["requested_samples"] == 9
    assert execution["phase_aggregation"]["minimum_valid_samples"] == 8
    assert execution["grid_selection"]["action"]["abstention_margin_grid"] == [
        0.0,
        0.005,
        0.01,
        0.02,
        0.05,
    ]
    assert execution["language_fixture_stage_semantics"] == {
        "item_count_per_partition": 48,
        "adapter_accept_count": 36,
        "adapter_abstain_count": 12,
        "learner_tuple_accept_count": 34,
        "learner_tuple_abstain_count": 14,
        "adapter_abstention_reasons": [
            "LANGUAGE_MISMATCH",
            "EMPTY_ASR",
            "INVALID_TIMESTAMP",
            "LOW_CONFIDENCE",
            "EMPTY_TRANSLATION",
            "SILENT_TRUNCATION",
        ],
        "tuple_only_abstention_reason": "INSUFFICIENT_IN_BOUNDS_FRAMES",
        "grounding_only_abstention_reason": "ONTOLOGY_UNMATCHED",
        "ontology_unmatched_rule": "the adapter and lexical learner tuple remain accepted and only the public-ontology grounding component abstains",
    }
    assert execution["pHash"]["near_duplicate_hamming_distance_max"] == 4
    assert execution["recurrence_pixels"]["full_canvas_unmasked_proxy"] == (
        "PROHIBITED"
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
    from collections import Counter

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    preparation = MODULE._tuple_fixture_preparation_amendment(config)
    for partition in preparation["partitions"]:
        rows = MODULE._language_lexical_fixture_rows(preparation, partition)
        assert len(rows) == 48
        assert Counter(row["expected_adapter_status"] for row in rows) == {
            "ACCEPT": 36,
            "ABSTAIN": 12,
        }
        assert Counter(row["expected_tuple_status"] for row in rows) == {
            "ACCEPT": 34,
            "ABSTAIN": 14,
        }
        assert Counter(row["expected_grounding_status"] for row in rows) == {
            "ACCEPT": 32,
            "ABSTAIN": 16,
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


def _visor_hos_review_fixture(tmp_path: Path) -> tuple[list[dict[str, object]], Path]:
    from PIL import Image

    frame_root = tmp_path / "public-frames"
    rows: list[dict[str, object]] = []
    # Twenty-six participant keys alternate into thirteen participants per
    # partition. Four frames per video yield 52 capped nominees per partition,
    # enough to test exact first-48 selection without large fixture media.
    for participant_index in range(1, 27):
        participant = f"P{participant_index:02d}"
        for frame_ordinal in range(1, 5):
            row = _visor_hos_no_hand_row(participant, 1, frame_ordinal)
            rows.append(row)
            target = frame_root / row["image"]["image_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (24, 18),
                (participant_index * 7 % 255, frame_ordinal * 31, 90),
            ).save(target)
    return rows, frame_root


def test_visor_hos_no_hand_review_queue_is_fixed_blind_and_order_invariant(
    tmp_path: Path,
) -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    rows, frame_root = _visor_hos_review_fixture(tmp_path)
    first, first_report = MODULE._visor_hos_no_hand_review_nominees(
        [{"video_annotations": rows}],
        seed=20260802,
        per_video_cap=4,
    )
    second, second_report = MODULE._visor_hos_no_hand_review_nominees(
        [{"video_annotations": list(reversed(rows))}],
        seed=20260802,
        per_video_cap=4,
    )
    assert first == second
    assert first_report == second_report
    assert first_report["queue_counts"] == {
        "development": 52,
        "holdout": 52,
    }
    assert first_report["participant_overlap_count"] == 0
    assert first_report["video_overlap_count"] == 0
    assert first_report["frame_overlap_count"] == 0

    review_root = tmp_path / "blind-review"
    compact = MODULE.prepare_visor_hos_no_hand_review(
        [{"video_annotations": rows}],
        cfg=config,
        frame_root=frame_root,
        review_root=review_root,
    )
    assert compact["status"] == "READY_FOR_AUTHORIZED_APPLICANT_BLIND_REVIEW"
    assert compact["review_queue_count"] == 104
    assert compact["decode_failure_count"] == 0
    assert compact["contact_sheet_count"] == 14
    assert set(compact) == {
        "status",
        "partition_count",
        "nominee_count",
        "review_queue_count",
        "decode_failure_count",
        "contact_sheet_count",
        "review_queue_commitment_sha256",
    }
    queue = json.loads((review_root / "review-queue.json").read_text())
    assert queue["model_inference_executed_before_review"] is False
    assert queue["model_output_fields_present"] is False
    assert all(
        len(row["review_token"]) == 24
        for partition in queue["partitions"].values()
        for row in partition
    )
    assert all(
        not any(key in row for key in ("prediction", "score", "egohos_output"))
        for partition in queue["partitions"].values()
        for row in partition
    )
    assert len(list(review_root.glob("contact-sheet-*.png"))) == 14


def test_visor_hos_no_hand_review_never_auto_accepts_annotation_absence(
    tmp_path: Path,
) -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    rows, frame_root = _visor_hos_review_fixture(tmp_path)
    review_root = tmp_path / "blind-review"
    MODULE.prepare_visor_hos_no_hand_review(
        [{"video_annotations": rows}],
        cfg=config,
        frame_root=frame_root,
        review_root=review_root,
    )
    incomplete = MODULE.seal_visor_hos_no_hand_review(
        review_root=review_root,
        authorized_applicant_attested=True,
        blind_to_egohos_output_attested=True,
        egohos_inference_not_started_attested=True,
    )
    assert incomplete["status"] == "INCOMPLETE_REVIEW"
    assert incomplete["verified_no_hand_count"] == 0
    assert incomplete["unreviewed_count"] == 104
    assert not (review_root / "verified-no-hand-seal.json").exists()
    with pytest.raises(RuntimeError, match="E_VISOR_HOS_NO_HAND_REVIEW_NOT_SEALED"):
        MODULE.load_visor_hos_verified_no_hand_frames(review_root)
    with pytest.raises(RuntimeError, match="E_VISOR_HOS_NO_HAND_REVIEW_ATTESTATION"):
        MODULE.seal_visor_hos_no_hand_review(
            review_root=review_root,
            authorized_applicant_attested=True,
            blind_to_egohos_output_attested=False,
            egohos_inference_not_started_attested=True,
        )
    next(review_root.glob("contact-sheet-*.png")).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="E_VISOR_HOS_NO_HAND_REVIEW_SHEET_COMMITMENT"):
        MODULE.seal_visor_hos_no_hand_review(
            review_root=review_root,
            authorized_applicant_attested=True,
            blind_to_egohos_output_attested=True,
            egohos_inference_not_started_attested=True,
        )


def test_visor_hos_no_hand_review_seals_exact_first_48_per_partition(
    tmp_path: Path,
) -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    rows, frame_root = _visor_hos_review_fixture(tmp_path)
    review_root = tmp_path / "blind-review"
    MODULE.prepare_visor_hos_no_hand_review(
        [{"video_annotations": rows}],
        cfg=config,
        frame_root=frame_root,
        review_root=review_root,
    )
    labels_path = review_root / "review-labels.json"
    labels = json.loads(labels_path.read_text())
    by_partition: dict[str, int] = {"development": 0, "holdout": 0}
    for row in labels["labels"]:
        ordinal = by_partition[row["partition"]]
        row["label"] = "no" if ordinal == 0 else "abstain" if ordinal == 1 else "yes"
        by_partition[row["partition"]] += 1
    MODULE.write_private(labels_path, labels)
    compact = MODULE.seal_visor_hos_no_hand_review(
        review_root=review_root,
        authorized_applicant_attested=True,
        blind_to_egohos_output_attested=True,
        egohos_inference_not_started_attested=True,
    )
    assert compact["status"] == "PASS"
    assert compact["coded_count"] == 104
    assert compact["verified_no_hand_count"] == 96
    assert compact["visible_hand_count"] == 2
    assert compact["abstain_count"] == 2
    assert compact["unreviewed_count"] == 0
    verified = MODULE.load_visor_hos_verified_no_hand_frames(review_root)
    assert len(verified) == 96
    seal = json.loads((review_root / "verified-no-hand-seal.json").read_text())
    for partition in ("development", "holdout"):
        queue_rows = json.loads((review_root / "review-queue.json").read_text())[
            "partitions"
        ][partition]
        first_48_yes = [row for row in queue_rows[2:]][:48]
        selected = seal["partitions"][partition]["selected"]
        assert [row["review_token"] for row in selected] == [
            row["review_token"] for row in first_48_yes
        ]


def test_visor_hos_annotation_commitments_match_frozen_serializations(
    tmp_path: Path,
) -> None:
    import hashlib

    root = tmp_path / "annotations"
    train = root / "train/P01_01.json"
    val = root / "val/P02_01.json"
    train.parent.mkdir(parents=True)
    val.parent.mkdir(parents=True)
    train.write_bytes(b'{"split":"train"}\n')
    val.write_bytes(b'{"split":"val"}\n')
    first, second, manifest, records = MODULE._visor_hos_annotation_commitments(
        root, [val, train]
    )
    expected_manifest = (
        f"{MODULE.file_digest(train)}  train/P01_01.json\n"
        f"{MODULE.file_digest(val)}  val/P02_01.json\n"
    ).encode("ascii")
    assert manifest == expected_manifest
    assert first == hashlib.sha256(expected_manifest).hexdigest()
    framed = hashlib.sha256()
    for relative, payload in (
        ("train/P01_01.json", train.read_bytes()),
        ("val/P02_01.json", val.read_bytes()),
    ):
        framed.update(relative.encode() + b"\0" + str(len(payload)).encode() + b"\0")
        framed.update(payload)
    assert second == framed.hexdigest()
    assert [row["relative_path"] for row in records] == [
        "train/P01_01.json",
        "val/P02_01.json",
    ]


def test_visor_hos_catalog_requires_exact_official_resource_semantics() -> None:
    import pytest

    frozen = {
        "dataset_id": "package-name",
        "revision_id": "revision",
        "JSON_file_count": 1,
        "bytes": 17,
    }
    resource = {
        "id": "resource",
        "name": "P01_01.json",
        "url": (
            "https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/"
            "GroundTruth-SparseAnnotations/annotations/train/P01_01.json"
        ),
        "size": 17,
        "hash": "",
    }
    package = {
        "success": True,
        "result": {
            "name": "package-name",
            "revision_id": "revision",
            "resources": [resource],
        },
    }
    assert MODULE._visor_hos_resource_rows(package, "train", frozen)[0][
        "name"
    ] == "P01_01.json"
    changed = json.loads(json.dumps(package))
    changed["result"]["resources"][0]["url"] = changed["result"]["resources"][
        0
    ]["url"].replace("/train/", "/val/")
    with pytest.raises(RuntimeError, match="E_VISOR_HOS_CATALOG_RESOURCE"):
        MODULE._visor_hos_resource_rows(changed, "train", frozen)
    changed = json.loads(json.dumps(package))
    changed["result"]["resources"][0]["hash"] = "author-digest"
    with pytest.raises(RuntimeError, match="E_VISOR_HOS_CATALOG_RESOURCE"):
        MODULE._visor_hos_resource_rows(changed, "train", frozen)


def test_visor_hos_source_inventory_keeps_no_hand_as_nominee_and_is_order_invariant() -> None:
    rows = []
    for participant_index in range(1, 9):
        participant = f"P{participant_index:02d}"
        for stratum_ordinal, relation in enumerate(
            ("object", "hand-not-in-contact", None)
        ):
            for video_offset in range(2):
                for frame_offset in range(4):
                    frame = stratum_ordinal * 100 + video_offset * 10 + frame_offset
                    rows.append(
                        _visor_hos_no_hand_row(
                            participant, stratum_ordinal * 2 + video_offset, frame
                        )
                        if relation is None
                        else _visor_hos_row(
                            participant,
                            stratum_ordinal * 2 + video_offset,
                            frame,
                            relation,
                        )
                    )
    first, first_report = MODULE._visor_hos_source_inventory(
        [{"video_annotations": rows}],
        seed=20260802,
        target_per_stratum=12,
        no_hand_review_queue_ceiling=16,
        per_video_stratum_cap=4,
        correction_excluded_frame_keys=set(),
    )
    second, second_report = MODULE._visor_hos_source_inventory(
        [{"video_annotations": list(reversed(rows))}],
        seed=20260802,
        target_per_stratum=12,
        no_hand_review_queue_ceiling=16,
        per_video_stratum_cap=4,
        correction_excluded_frame_keys=set(),
    )
    compact = lambda selected: [
        (row["video"], row["frame_name"], row["stratum"])
        for partition in ("development", "holdout")
        for row in selected[partition]
    ]
    assert compact(first) == compact(second)
    assert first_report == second_report
    assert first_report["status"] == "PASS_SOURCE_NOMINEES"
    assert first_report["no_hand_items_are_unverified_nominees"] is True
    assert not any(
        row["stratum"] == "verified_no_hand"
        for partition in first.values()
        for row in partition
    )
    assert first_report["participant_overlap_count"] == 0
    assert first_report["video_overlap_count"] == 0
    assert first_report["frame_overlap_count"] == 0


def test_charades_source_inventory_collects_every_label_deficit() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    action = MODULE._tuple_fixture_protocol(config)[
        "order_dependent_action_control"
    ]
    selected, report = MODULE._charades_action_source_inventory(
        [], action, 20260802, set(), set()
    )
    assert report["status"] == "NO_GO"
    assert len(report["deficits"]) == 16
    assert {row["partition"] for row in report["deficits"]} == {
        "development",
        "holdout",
    }
    assert selected == {"development": [], "holdout": []}


def _active_visor_source_selections() -> dict[str, list[dict[str, object]]]:
    output = {"development": [], "holdout": []}
    for partition_index, partition in enumerate(output, start=1):
        participant = f"P{partition_index:02d}"
        for stratum_index, stratum in enumerate(
            ("contact", "explicit_no_contact", "no_hand_nominee")
        ):
            for ordinal in range(48):
                video_ordinal = stratum_index * 20 + ordinal // 4 + 1
                video = f"{participant}_{video_ordinal:02d}"
                frame_name = f"{video}_frame_{ordinal:010d}.jpg"
                output[partition].append(
                    {
                        "participant": participant,
                        "video": video,
                        "frame_name": frame_name,
                        "image_path": f"{video}/{frame_name}",
                        "source_split": "train",
                        "stratum": stratum,
                        "hand_visible": stratum != "no_hand_nominee",
                        "contact": True if stratum == "contact" else False if stratum == "explicit_no_contact" else None,
                        "target_hand_side": (
                            "left hand" if stratum != "no_hand_nominee" else None
                        ),
                        "target_hand_ordinal": (
                            0 if stratum != "no_hand_nominee" else None
                        ),
                        "target_hand_segments": (
                            [[[1.0, 1.0], [4.0, 1.0], [4.0, 4.0]]]
                            if stratum != "no_hand_nominee"
                            else None
                        ),
                    }
                )
    return output


def _active_visor_source_record() -> dict[str, object]:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    correction = MODULE._tuple_visor_hos_correction_amendment(config)
    preparation = MODULE._tuple_fixture_preparation_amendment(config)
    repair = MODULE._tuple_fixture_feasibility_repair(config)
    protocol = MODULE._tuple_fixture_protocol(config)
    return {
        "schema_version": 2,
        "status": "PASS_SOURCE_FEASIBILITY_PENDING_NO_HAND_REVIEW",
        "visor_hos_correction_amendment_commitment_sha256": correction[
            "amendment_commitment_sha256"
        ],
        "fixture_preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "fixture_feasibility_repair_commitment_sha256": repair[
            "fixture_feasibility_repair_commitment_sha256"
        ],
        "public_fixture_protocol_commitment_sha256": protocol[
            "protocol_commitment_sha256"
        ],
        "families": {
            "visor_hos_contact": {"status": "PASS"},
            "visor_hos_explicit_no_contact": {"status": "PASS"},
            "visor_hos_no_hand_nominees": {
                "status": "PASS_NOMINEE_QUEUE_READY"
            },
            "visor_hos_integrity": {"status": "PASS"},
            "cross_partition_source_independence": {"status": "PASS"},
        },
        "failing_family_names": [],
        "selections": {
            "visor_hos_source_nominees": _active_visor_source_selections()
        },
        "audits": {
            "source_subject_overlap_count": 0,
            "source_video_overlap_count": 0,
            "source_object_overlap_count": 0,
        },
        "no_hand_truth_opened": False,
        "model_inference_executed": False,
        "media_rendering_executed": False,
        "restricted_mount_present": False,
    }


def test_active_visor_source_record_and_verified_merge_are_fail_closed(
    tmp_path: Path,
) -> None:
    import pytest

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    record = _active_visor_source_record()
    record["visor_hos_source_feasibility_commitment_sha256"] = MODULE.digest(record)
    MODULE.write_private(tmp_path / "visor-hos-source-feasibility.json", record)
    loaded = MODULE._load_active_visor_hos_source_feasibility(tmp_path, config)
    selected = loaded["selections"]["visor_hos_source_nominees"]
    verified = {"partitions": {}}
    for partition in ("development", "holdout"):
        nominees = [
            row for row in selected[partition] if row["stratum"] == "no_hand_nominee"
        ]
        verified["partitions"][partition] = [
            {
                "video": row["video"],
                "frame_name": row["frame_name"],
                "source_frame_sha256": "1" * 64,
                "review_token": f"token-{partition}-{ordinal}",
            }
            for ordinal, row in enumerate(nominees)
        ]
    merged, audits = MODULE._merge_active_visor_hos_selections(loaded, verified)
    assert audits == {
        "source_frame_overlap_count": 0,
        "source_frame_duplicate_count": 0,
    }
    for partition in ("development", "holdout"):
        assert len(merged[partition]) == 144
        assert sum(row["stratum"] == "contact" for row in merged[partition]) == 48
        assert sum(
            row["stratum"] == "explicit_no_contact" for row in merged[partition]
        ) == 48
        assert sum(
            row["stratum"] == "verified_no_hand" for row in merged[partition]
        ) == 48
    broken = json.loads(json.dumps(verified))
    broken["partitions"]["holdout"][0]["frame_name"] = "not-nominated.jpg"
    with pytest.raises(
        RuntimeError, match="E_VISOR_HOS_VERIFIED_NO_HAND_NOT_NOMINATED"
    ):
        MODULE._merge_active_visor_hos_selections(loaded, broken)
    tampered = json.loads(json.dumps(record))
    tampered["selections"]["visor_hos_source_nominees"]["holdout"][0][
        "participant"
    ] = "P01"
    tampered["visor_hos_source_feasibility_commitment_sha256"] = MODULE.digest(
        {
            key: value
            for key, value in tampered.items()
            if key != "visor_hos_source_feasibility_commitment_sha256"
        }
    )
    MODULE.write_private(tmp_path / "visor-hos-source-feasibility.json", tampered)
    with pytest.raises(RuntimeError, match="E_VISOR_HOS_SOURCE_FEASIBILITY_OVERLAP"):
        MODULE._load_active_visor_hos_source_feasibility(tmp_path, config)


def _construct_aligned_action_source_rows(config: dict) -> dict[str, list[dict]]:
    protocol = MODULE._tuple_fixture_protocol(config)[
        "order_dependent_action_control"
    ]
    preparation = MODULE._tuple_fixture_preparation_amendment(config)
    code_by_label = {
        label: pair["matched_codes"][0][index]
        for pair in protocol["class_code_pairs"]
        for index, label in enumerate(pair["pair"])
    }
    output: dict[str, list[dict]] = {}
    for partition, counts in MODULE.CONSTRUCT_ALIGNED_ACTION_CLASS_COUNTS.items():
        rows = []
        for label in protocol["labels"]:
            for ordinal in range(counts[label]):
                rows.append(
                    {
                        "video": f"{partition}-{label}-{ordinal:02d}",
                        "subject": f"{partition}-subject-{label}-{ordinal:02d}",
                        "label": label,
                        "code": code_by_label[label],
                        "start": 1.0,
                        "end": 3.0,
                        "source_duration": 10.0,
                    }
                )
        rows.sort(
            key=lambda row: (
                protocol["labels"].index(row["label"]),
                MODULE._fixture_order(
                    int(preparation["seed"]),
                    "mechanistic_action_final",
                    partition,
                    row["video"],
                    row["start"],
                ),
            )
        )
        output[partition] = rows
    return output


def _construct_aligned_source_record(config: dict) -> dict:
    record = _active_visor_source_record()
    statuses = {
        "official_visor_hos_artifact": "PASS",
        "visor_hos_semantic_reference": "PASS",
        "visor_hos_contact": "PASS",
        "visor_hos_explicit_no_contact": "PASS",
        "visor_hos_no_hand_nominees": "PASS_NOMINEE_QUEUE_READY",
        "visor_hos_integrity": "PASS",
        "coco_composite_sources": "PASS",
        "language_and_lexical": "PASS",
        "referent_attribute_composite": "PASS",
        "recurrence": "PASS",
        "sensor": "PASS",
        "charades_order_action": "NO_GO",
        "cross_partition_source_independence": "PASS",
    }
    record.update(
        {
            "status": "NO_GO_COMPLETE_SOURCE_FEASIBILITY",
            "families": {
                name: {"status": status} for name, status in statuses.items()
            },
            "failing_family_names": ["charades_order_action"],
            "action_inventory": {
                "status": "NO_GO",
                "final_counts": MODULE.CONSTRUCT_ALIGNED_ACTION_CLASS_COUNTS,
                "deficits": MODULE.CONSTRUCT_ALIGNED_ACTION_DEFICITS,
                "subject_overlap_count": 0,
                "video_overlap_count": 0,
            },
            "no_hand_review_required_before_public_model_inference": True,
            "large_Charades_video_archive_downloaded": False,
        }
    )
    record["selections"]["charades_order_action"] = (
        _construct_aligned_action_source_rows(config)
    )
    return record


def test_construct_aligned_source_reuse_accepts_only_exact_action_no_go(
    tmp_path: Path, monkeypatch,
) -> None:
    import inspect
    import pytest

    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    record = _construct_aligned_source_record(config)
    commitment = MODULE.digest(record)
    record["visor_hos_source_feasibility_commitment_sha256"] = commitment
    path = tmp_path / "visor-hos-source-feasibility.json"
    MODULE.write_private(path, record)
    monkeypatch.setattr(
        MODULE, "CONSTRUCT_ALIGNED_SOURCE_NO_GO_SHA256", commitment
    )
    monkeypatch.setattr(
        MODULE,
        "_construct_aligned_ltx_resume_amendment",
        lambda _cfg: {
            "amendment_commitment_sha256": (
                MODULE.CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256
            ),
            "prior_results_and_amendments_preserved": {
                "complete_source_no_go": commitment
            },
        },
    )
    before = path.read_bytes()
    loaded = MODULE._load_construct_aligned_visor_hos_source_reuse(
        tmp_path, config
    )
    assert path.read_bytes() == before
    assert {
        partition: len(rows)
        for partition, rows in loaded["selections"][
            "charades_order_action"
        ].items()
    } == MODULE.CONSTRUCT_ALIGNED_ACTION_COUNTS
    assert "_select_charades_action_fixtures" not in inspect.getsource(
        MODULE._prepare_action_fixtures
    )

    broken = json.loads(json.dumps(record))
    same_label = broken["selections"]["charades_order_action"]["development"]
    same_label[0], same_label[1] = same_label[1], same_label[0]
    broken_payload = json.loads(json.dumps(broken))
    broken_payload.pop("visor_hos_source_feasibility_commitment_sha256")
    broken_commitment = MODULE.digest(broken_payload)
    broken["visor_hos_source_feasibility_commitment_sha256"] = broken_commitment
    MODULE.write_private(path, broken)
    monkeypatch.setattr(
        MODULE, "CONSTRUCT_ALIGNED_SOURCE_NO_GO_SHA256", broken_commitment
    )
    monkeypatch.setattr(
        MODULE,
        "_construct_aligned_ltx_resume_amendment",
        lambda _cfg: {
            "prior_results_and_amendments_preserved": {
                "complete_source_no_go": broken_commitment
            }
        },
    )
    with pytest.raises(
        RuntimeError, match="E_CONSTRUCT_ALIGNED_ACTION_SOURCE_ORDER"
    ):
        MODULE._load_construct_aligned_visor_hos_source_reuse(tmp_path, config)


def test_construct_aligned_action_fixture_projection_is_exact() -> None:
    import pytest

    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    source = _construct_aligned_action_source_rows(config)["development"]
    fixtures = [
        {
            **row,
            "fixture_ordinal": ordinal,
            "media_relative_path": f"media/{ordinal:03d}.mp4",
            "media_sha256": f"{ordinal + 1:064x}",
            "media_bytes": ordinal + 1,
        }
        for ordinal, row in enumerate(source)
    ]
    MODULE._validate_construct_aligned_action_fixture_projection(
        fixtures, source, "development"
    )
    fixtures[0]["video"] = "reselected-video"
    with pytest.raises(
        RuntimeError, match="E_CONSTRUCT_ALIGNED_ACTION_FIXTURE_PROJECTION"
    ):
        MODULE._validate_construct_aligned_action_fixture_projection(
            fixtures, source, "development"
        )


def test_fixture_materialization_never_calls_legacy_sparse_visor_path() -> None:
    import inspect

    source = inspect.getsource(MODULE.prepare_tuple_fixtures)
    active_source = inspect.getsource(MODULE._prepare_active_visor_hos_fixtures)
    assert "_prepare_active_visor_hos_fixtures(" in source
    assert "_prepare_visor_fixtures(" not in source
    for field in (
        "visor_hos_correction_amendment_commitment_sha256",
        "visor_hos_source_feasibility_commitment_sha256",
        "verified_no_hand_seal_commitment_sha256",
        "source_frame_overlap_count",
    ):
        assert field in source
    assert '"target_hand_segments":' not in active_source
    assert "_write_visor_hos_target_hand_mask(" in active_source


class _FakeFixedCanvasCV2:
    __version__ = "4.10.0"
    error = RuntimeError

    def __init__(self) -> None:
        self.contour_dtypes = []
        self.fill_poly_contour_counts = []

    def fillPoly(self, canvas, contours, color) -> None:
        import numpy as np
        from PIL import Image, ImageDraw

        assert contours
        self.fill_poly_contour_counts.append(len(contours))
        self.contour_dtypes.extend(contour.dtype for contour in contours)
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for contour in contours:
            draw.polygon(
                [tuple(int(value) for value in point) for point in contour],
                fill=int(color[0]),
            )
        canvas[...] = np.asarray(image)


def test_visor_hos_target_hand_mask_uses_fixed_canvas_boundary_semantics(
    tmp_path: Path,
) -> None:
    import pytest
    import numpy as np
    from PIL import Image

    cv2 = _FakeFixedCanvasCV2()
    target = tmp_path / "target-hand.png"
    record = MODULE._write_visor_hos_target_hand_mask(
        [[[1.0, 1.0], [12.0, 1.0], [17.0, 17.0], [1.0, 10.0]]],
        12,
        10,
        target,
        cv2_module=cv2,
    )
    assert record["path"] == target
    assert record["sha256"] == MODULE.file_digest(target)
    assert record["bytes"] == target.stat().st_size
    assert record["width"] == 12
    assert record["height"] == 10
    assert record["boundary_vertex_count"] == 2
    assert record["outside_canvas_vertex_count"] == 1
    assert record["outside_canvas_component_count"] == 1
    assert cv2.contour_dtypes == [np.dtype(np.int32)]
    with Image.open(target) as mask:
        observed = np.asarray(mask)
        assert mask.format == "PNG"
        assert mask.mode == "L"
        assert mask.size == (12, 10)
        assert {
            index for index, count in enumerate(mask.histogram()) if count
        } == {0, 255}
        assert observed[:, -1].any()
        assert observed[-1, :].any()
    with pytest.raises(RuntimeError, match="E_TUPLE_VISOR_EMPTY_HAND_MASK"):
        MODULE._write_visor_hos_target_hand_mask(
            [[[13.0, 1.0], [18.0, 1.0], [18.0, 7.0], [13.0, 7.0]]],
            12,
            10,
            tmp_path / "out-of-bounds.png",
            cv2_module=cv2,
        )
    union = MODULE._write_visor_hos_target_hand_mask(
        [
            [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]],
            [[13.0, 1.0], [18.0, 1.0], [18.0, 7.0], [13.0, 7.0]],
        ],
        12,
        10,
        tmp_path / "mixed-in-and-outside-components.png",
        cv2_module=cv2,
    )
    assert union["outside_canvas_component_count"] == 1
    assert union["outside_canvas_vertex_count"] == 4
    assert cv2.fill_poly_contour_counts[-1] == 2


def test_visor_hos_target_hand_mask_rejects_invalid_geometry_before_rasterizer(
    tmp_path: Path,
) -> None:
    import pytest

    cv2 = _FakeFixedCanvasCV2()
    invalid = (
        [[[1.0, 1.0], [2.0, 2.0]]],
        [[[-1.0, 1.0], [2.0, 1.0], [2.0, 2.0]]],
        [[[float("nan"), 1.0], [2.0, 1.0], [2.0, 2.0]]],
        [[[float("inf"), 1.0], [2.0, 1.0], [2.0, 2.0]]],
        [[[2**31, 1.0], [2.0, 1.0], [2.0, 2.0]]],
    )
    for ordinal, segments in enumerate(invalid):
        with pytest.raises(RuntimeError, match="E_TUPLE_VISOR_GEOMETRY"):
            MODULE._write_visor_hos_target_hand_mask(
                segments,
                12,
                10,
                tmp_path / f"invalid-{ordinal}.png",
                cv2_module=cv2,
            )
    assert cv2.contour_dtypes == []


def test_visor_hos_rasterizer_version_and_target_mask_serialization_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib.metadata
    import sys
    import types

    import pytest
    from PIL import Image

    fake = types.SimpleNamespace(__version__="4.10.0", fillPoly=lambda *_: None)
    monkeypatch.setitem(sys.modules, "cv2", fake)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "4.10.0.83" if name == "opencv-python-headless" else "0",
    )
    with pytest.raises(RuntimeError, match="E_TUPLE_VISOR_RASTERIZER_VERSION"):
        MODULE._load_visor_hos_opencv()

    fixture_root = tmp_path / "fixtures"
    source = fixture_root / "source.png"
    mask = fixture_root / "mask.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (12, 10), (10, 20, 30)).save(source)
    Image.new("RGB", (12, 10), (255, 0, 0)).save(mask)
    row = {
        "media_relative_path": "source.png",
        "media_sha256": MODULE.file_digest(source),
        "media_bytes": source.stat().st_size,
        "target_hand_mask_relative_path": "mask.png",
        "target_hand_mask_sha256": MODULE.file_digest(mask),
        "target_hand_mask_bytes": mask.stat().st_size,
        "target_hand_mask_width": 12,
        "target_hand_mask_height": 10,
    }
    with pytest.raises(
        RuntimeError, match="E_TUPLE_EGOHOS_TARGET_MASK_SERIALIZATION"
    ):
        MODULE._read_tuple_egohos_target_mask(row, fixture_root)

    Image.new("L", (12, 10), 255).save(mask)
    row["target_hand_mask_sha256"] = MODULE.file_digest(mask)
    row["target_hand_mask_bytes"] = mask.stat().st_size
    row["target_hand_mask_width"] = True
    with pytest.raises(
        RuntimeError, match="E_TUPLE_EGOHOS_TARGET_MASK_SERIALIZATION"
    ):
        MODULE._read_tuple_egohos_target_mask(row, fixture_root)

    row.pop("target_hand_mask_width")
    with pytest.raises(
        RuntimeError, match="E_TUPLE_EGOHOS_TARGET_MASK_SERIALIZATION"
    ):
        MODULE._read_tuple_egohos_target_mask(row, fixture_root)


def test_active_visor_materialization_seals_masks_and_no_hand_lineage(
    tmp_path: Path, monkeypatch,
) -> None:
    from PIL import Image

    fixture_root = tmp_path / "fixtures"
    review_root = fixture_root / "no-hand-review"
    review_root.mkdir(parents=True)
    (fixture_root / "visor-hos-source-feasibility.json").write_text("{}")
    (review_root / "verified-no-hand-seal.json").write_text("{}")
    source_commitment = "1" * 64
    seal_commitment = "2" * 64
    amendment_commitment = "3" * 64
    selected: dict[str, list[dict[str, object]]] = {}
    for partition_index, partition in enumerate(("development", "holdout")):
        split = "train" if partition == "development" else "val"
        rows: list[dict[str, object]] = []
        for stratum_index, stratum in enumerate(
            ("contact", "explicit_no_contact", "verified_no_hand")
        ):
            participant = f"P{partition_index * 3 + stratum_index + 1:02d}"
            video = f"{participant}_fixture"
            frame_name = f"frame_{stratum_index}.jpg"
            buffer = BytesIO()
            Image.new(
                "RGB",
                (16, 12),
                (20 + partition_index * 80, 40 + stratum_index * 50, 60),
            ).save(buffer, format="JPEG")
            frame_bytes = buffer.getvalue()
            archive_path = (
                fixture_root
                / "sources/VISOR-HOS/frame-archives"
                / split
                / participant
                / f"{video}.zip"
            )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"nested/{frame_name}", frame_bytes)
            visible = stratum != "verified_no_hand"
            rows.append(
                {
                    "stratum": stratum,
                    "hand_visible": visible,
                    "contact": (
                        stratum == "contact" if visible else None
                    ),
                    "target_hand_side": "left hand" if visible else None,
                    "target_hand_segments": (
                        [[[2.0, 2.0], [10.0, 2.0], [10.0, 8.0], [2.0, 8.0]]]
                        if visible
                        else None
                    ),
                    "source_split": split,
                    "participant": participant,
                    "video": video,
                    "frame_name": frame_name,
                    "verified_source_frame_sha256": (
                        MODULE.hashlib.sha256(frame_bytes).hexdigest()
                        if not visible
                        else None
                    ),
                    "review_token": (
                        f"review-{partition}" if not visible else None
                    ),
                }
            )
        selected[partition] = rows

    source_record = {
        "visor_hos_source_feasibility_commitment_sha256": source_commitment
    }
    verified = {"commitment_sha256": seal_commitment}
    monkeypatch.setattr(
        MODULE, "_load_visor_hos_opencv", lambda: _FakeFixedCanvasCV2()
    )
    monkeypatch.setattr(
        MODULE,
        "_load_construct_aligned_visor_hos_source_reuse",
        lambda fixture_root, cfg: source_record,
    )
    monkeypatch.setattr(
        MODULE,
        "_load_visor_hos_verified_no_hand_lineage",
        lambda review_root, **kwargs: verified,
    )
    monkeypatch.setattr(
        MODULE,
        "_merge_active_visor_hos_selections",
        lambda source, seal: (
            selected,
            {"source_frame_overlap_count": 0, "source_frame_duplicate_count": 0},
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_visor_hos_correction_amendment",
        lambda cfg: {
            "amendment_commitment_sha256": amendment_commitment,
            "official_annotation_artifact": {
                "external_sorted_relative_path_and_SHA256_manifest_commitment_sha256": "4"
                * 64,
                "combined_bytes": 868821446,
            },
        },
    )
    output, provenance, lineage = MODULE._prepare_active_visor_hos_fixtures(
        fixture_root,
        {
            "partitions": ["development", "holdout"],
            "source_archives": {
                "EPIC_KITCHENS_VISOR_validation": {
                    "repository_root": "https://example.invalid/VISOR",
                    "license": "CC-BY-NC-4.0",
                }
            },
        },
        json.loads(
            Path("configs/synthetic_video_real_only_proof.json").read_text()
        ),
        review_root,
    )

    assert len(provenance) == 9
    assert lineage == {
        "visor_hos_correction_amendment_commitment_sha256": amendment_commitment,
        "visor_hos_source_feasibility_commitment_sha256": source_commitment,
        "verified_no_hand_seal_commitment_sha256": seal_commitment,
        "construct_aligned_ltx_resume_amendment_commitment_sha256": (
            MODULE.CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256
        ),
        "source_frame_overlap_count": 0,
        "source_frame_duplicate_count": 0,
    }
    for rows in output.values():
        assert len(rows) == 3
        for row in rows:
            assert "target_hand_segments" not in row
            if row["stratum"] == "verified_no_hand":
                assert row["verified_no_hand_seal_commitment_sha256"] == seal_commitment
                assert row["target_hand_mask_relative_path"] is None
                continue
            assert row["verified_no_hand_seal_commitment_sha256"] is None
            mask_path = fixture_root / row["target_hand_mask_relative_path"]
            assert MODULE.file_digest(mask_path) == row["target_hand_mask_sha256"]
            assert mask_path.stat().st_size == row["target_hand_mask_bytes"]
            with Image.open(mask_path) as mask:
                assert mask.mode == "L"
                assert mask.size == (16, 12)
                assert {
                    index for index, count in enumerate(mask.histogram()) if count
                } == {0, 255}


def test_active_no_hand_queue_is_exactly_recovered_from_sealed_source(
    monkeypatch,
) -> None:
    import pytest

    seed = 20260802
    amendment = {
        "partition_and_joint_sampler": {
            "seed": seed,
            "per_video_per_stratum_cap": 4,
        }
    }
    monkeypatch.setattr(
        MODULE,
        "_tuple_visor_hos_correction_amendment",
        lambda cfg: amendment,
    )
    selections = _active_visor_source_selections()
    for partition in ("development", "holdout"):
        nominees = [
            row for row in selections[partition] if row["stratum"] == "no_hand_nominee"
        ]
        for ordinal, row in enumerate(nominees, start=1):
            row["review_ordinal"] = ordinal
            row["review_token"] = MODULE._visor_hos_no_hand_review_token(
                seed, partition, row
            )
    inventory = {
        "queue_counts": {"development": 48, "holdout": 48},
        "participant_overlap_count": 0,
        "video_overlap_count": 0,
        "frame_overlap_count": 0,
    }
    record = {
        "selections": {"visor_hos_source_nominees": selections},
        "visor_hos_inventory": {
            "no_hand_review_queue_inventory": inventory
        },
    }
    queues, recovered_inventory = MODULE._active_visor_hos_no_hand_review_queues(
        record, {}
    )
    assert recovered_inventory == inventory
    assert {
        partition: len(rows) for partition, rows in queues.items()
    } == {"development": 48, "holdout": 48}
    assert all(
        row["review_ordinal"] == ordinal
        for rows in queues.values()
        for ordinal, row in enumerate(rows, start=1)
    )

    broken = json.loads(json.dumps(record))
    development_nominee = broken["selections"]["visor_hos_source_nominees"][
        "development"
    ][-48]
    holdout_nominee = broken["selections"]["visor_hos_source_nominees"][
        "holdout"
    ][-48]
    for key in (
        "participant",
        "video",
        "frame_name",
        "image_path",
        "source_split",
    ):
        holdout_nominee[key] = development_nominee[key]
    holdout_nominee["review_token"] = MODULE._visor_hos_no_hand_review_token(
        seed, "holdout", holdout_nominee
    )
    with pytest.raises(
        RuntimeError, match="E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_OVERLAP"
    ):
        MODULE._active_visor_hos_no_hand_review_queues(broken, {})


def test_no_hand_review_materializes_only_frozen_nominee_frames(
    tmp_path: Path,
) -> None:
    from PIL import Image

    fixture_root = tmp_path / "fixtures"
    review_root = fixture_root / "no-hand-review"
    queues: dict[str, list[dict[str, object]]] = {
        "development": [],
        "holdout": [],
    }
    for index, partition in enumerate(("development", "holdout"), start=1):
        participant = f"P{index:02d}"
        video = f"{participant}_01"
        frame_name = f"{video}_frame_0000000001.jpg"
        image_path = f"{video}/{frame_name}"
        split = "train" if partition == "development" else "val"
        buffer = BytesIO()
        Image.new("RGB", (18, 14), (index * 60, 40, 80)).save(
            buffer, format="JPEG"
        )
        archive_path = (
            fixture_root
            / "sources/VISOR-HOS/frame-archives"
            / split
            / participant
            / f"{video}.zip"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(f"nested/{frame_name}", buffer.getvalue())
            archive.writestr("nested/unselected.jpg", buffer.getvalue())
        queues[partition].append(
            {
                "source_split": split,
                "participant": participant,
                "video": video,
                "frame_name": frame_name,
                "image_path": image_path,
                "review_token": f"token-{partition}",
            }
        )
    source_commitment = "5" * 64
    frame_root, record = (
        MODULE._materialize_active_visor_hos_no_hand_review_frames(
            fixture_root,
            review_root,
            {
                "source_archives": {
                    "EPIC_KITCHENS_VISOR_validation": {
                        "repository_root": "https://example.invalid/VISOR",
                        "license": "CC-BY-NC-4.0",
                    }
                }
            },
            {
                "visor_hos_source_feasibility_commitment_sha256": source_commitment
            },
            queues,
            MODULE.CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256,
        )
    )
    assert record["archive_count"] == 2
    assert record["archive_download_worker_count"] == 2
    assert record["source_frame_count"] == 2
    assert record["model_inference_executed"] is False
    assert record["restricted_mount_present"] is False
    assert not list(frame_root.rglob("unselected.jpg"))
    for rows in queues.values():
        for row in rows:
            assert (frame_root / row["image_path"]).is_file()
    loaded = MODULE._load_visor_hos_no_hand_frame_materialization(
        review_root,
        source_commitment,
        MODULE.CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256,
    )
    assert loaded["source_frame_materialization_commitment_sha256"] == record[
        "source_frame_materialization_commitment_sha256"
    ]


def test_active_no_hand_cli_wrappers_are_canonical_and_aggregate_only(
    tmp_path: Path, monkeypatch,
) -> None:
    from types import SimpleNamespace

    config_path = tmp_path / "config.json"
    config_path.write_text(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    public_root = tmp_path / "public-root"
    fixture_root = MODULE._tuple_fixture_root(public_root)
    source_commitment = "6" * 64
    materialization_commitment = "7" * 64
    queue_commitment = "8" * 64
    label_commitment = "9" * 64
    source_record = {
        "visor_hos_source_feasibility_commitment_sha256": source_commitment
    }
    queues = {"development": [], "holdout": []}
    inventory = {"raw_nominee_count": 0}
    monkeypatch.setattr(
        MODULE,
        "_load_construct_aligned_visor_hos_source_reuse",
        lambda root, cfg: source_record,
    )
    monkeypatch.setattr(
        MODULE,
        "_active_visor_hos_no_hand_review_queues",
        lambda source, cfg: (queues, inventory),
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_fixture_preparation_amendment",
        lambda cfg: {"source_archives": {}},
    )
    monkeypatch.setattr(
        MODULE,
        "_materialize_active_visor_hos_no_hand_review_frames",
        lambda *args: (
            fixture_root / "no-hand-review/source-frames",
            {
                "source_frame_count": 96,
                "archive_count": 24,
                "source_frame_materialization_commitment_sha256": (
                    materialization_commitment
                ),
            },
        ),
    )

    def fake_prepare(annotation_documents, **kwargs):
        assert annotation_documents is None
        assert kwargs["review_root"] == fixture_root / "no-hand-review"
        assert kwargs["preselected_queues"] is queues
        assert kwargs["source_feasibility_commitment_sha256"] == source_commitment
        assert (
            kwargs["source_frame_materialization_commitment_sha256"]
            == materialization_commitment
        )
        assert (
            kwargs["construct_aligned_amendment_commitment_sha256"]
            == MODULE.CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256
        )
        return {
            "status": "READY_FOR_AUTHORIZED_APPLICANT_BLIND_REVIEW",
            "partition_count": 2,
            "nominee_count": 96,
            "review_queue_count": 96,
            "decode_failure_count": 0,
            "contact_sheet_count": 12,
            "review_queue_commitment_sha256": queue_commitment,
        }

    monkeypatch.setattr(MODULE, "prepare_visor_hos_no_hand_review", fake_prepare)
    args = SimpleNamespace(public_root=public_root, config=config_path)
    prepared = MODULE.prepare_active_visor_hos_no_hand_review(args)
    serialized = MODULE.compact_aggregate_json(
        prepared,
        allowed_fields=MODULE.TUPLE_NO_HAND_REVIEW_PREP_FIELDS,
        sha256_fields=MODULE.TUPLE_NO_HAND_REVIEW_PREP_HASH_FIELDS,
    )
    assert "source-frames" not in serialized
    assert "review_token" not in serialized

    queue = {
        "review_queue_commitment_sha256": queue_commitment,
        "visor_hos_source_feasibility_commitment_sha256": source_commitment,
        "source_frame_materialization_commitment_sha256": materialization_commitment,
        "visor_hos_correction_amendment_commitment_sha256": "a" * 64,
        "construct_aligned_ltx_resume_amendment_commitment_sha256": (
            MODULE.CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256
        ),
    }
    monkeypatch.setattr(
        MODULE, "_load_visor_hos_no_hand_review_queue", lambda root: queue
    )
    monkeypatch.setattr(
        MODULE,
        "_load_visor_hos_no_hand_frame_materialization",
        lambda root, expected, expected_active: {
            "source_frame_materialization_commitment_sha256": (
                materialization_commitment
            )
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_visor_hos_correction_amendment",
        lambda cfg: {"amendment_commitment_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        MODULE,
        "seal_visor_hos_no_hand_review",
        lambda **kwargs: {
            "status": "INCOMPLETE_REVIEW",
            "partition_count": 2,
            "coded_count": 0,
            "verified_no_hand_count": 0,
            "visible_hand_count": 0,
            "abstain_count": 0,
            "unreviewed_count": 96,
            "deficit_partition_count": 2,
            "review_labels_commitment_sha256": label_commitment,
            "verified_no_hand_seal_commitment_sha256": None,
        },
    )
    seal_args = SimpleNamespace(
        public_root=public_root,
        config=config_path,
        authorized_applicant_attested=True,
        blind_to_egohos_output_attested=True,
        egohos_inference_not_started_attested=True,
    )
    incomplete = MODULE.seal_active_visor_hos_no_hand_review(seal_args)
    assert "verified_no_hand_seal_commitment_sha256" not in incomplete
    MODULE.compact_aggregate_json(
        incomplete,
        allowed_fields=MODULE.TUPLE_NO_HAND_REVIEW_INCOMPLETE_SEAL_FIELDS,
        sha256_fields=MODULE.TUPLE_NO_HAND_REVIEW_INCOMPLETE_SEAL_HASH_FIELDS,
    )


def test_public_output_guard_allows_ignored_root_and_rejects_tracked_root() -> None:
    import pytest

    MODULE._require_external_or_ignored_output(Path("tmp/source-feasibility-test"))
    with pytest.raises(
        RuntimeError, match="E_TUPLE_FIXTURE_OUTPUT_NOT_EXTERNAL_OR_IGNORED"
    ):
        MODULE._require_external_or_ignored_output(Path("scripts/source-artifacts"))


def test_public_output_guard_without_git_allows_only_external_root(
    monkeypatch, tmp_path: Path
) -> None:
    import pytest

    external = tmp_path / "external-output"
    monkeypatch.setattr(MODULE.shutil, "which", lambda _name: None)
    MODULE._require_external_or_ignored_output(external)

    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="E_TUPLE_FIXTURE_OUTPUT_IN_GIT"):
        MODULE._require_external_or_ignored_output(repository / "ignored-output")


def test_charades_loader_uses_official_first_person_only_tables(
    tmp_path: Path,
) -> None:
    header = "id,subject,verified,length,actions,egocentric\n"
    for split in ("train", "test"):
        (tmp_path / f"CharadesEgo_v1_{split}.csv").write_text(
            header + f"third-{split},subject-{split},Yes,10,c000 1 3,paired\n"
        )
        (tmp_path / f"CharadesEgo_v1_{split}_only1st.csv").write_text(
            header + f"first-{split}EGO,subject-{split},Yes,10,c000 1 3,paired\n"
        )
    rows = MODULE._load_charades_rows(tmp_path)
    assert [row["id"] for row in rows] == ["first-trainEGO", "first-testEGO"]


def test_prior_activity_reconstruction_is_subject_partitioned_and_bounded() -> None:
    fixture = {
        "seed": 20260802,
        "development_items": 3,
        "holdout_items": 3,
        "label_code_map": {
            f"label_{index}": [f"c{index:03d}"] for index in range(8)
        },
    }
    subjects = {"development": [], "holdout": []}
    ordinal = 0
    while any(len(values) < 4 for values in subjects.values()):
        subject = f"public-subject-{ordinal}"
        partition = MODULE._fixture_partition(
            fixture["seed"], "partition", subject
        )
        # `_fixture_partition` hashes the same literal seed|partition|subject
        # tuple used by the frozen broad-context recipe.
        if len(subjects[partition]) < 4:
            subjects[partition].append(subject)
        ordinal += 1
    rows = []
    action_string = ";".join(
        f"c{index:03d} 1 3" for index in range(8)
    )
    for partition, values in subjects.items():
        for index, subject in enumerate(values):
            rows.append(
                {
                    "id": f"{partition}-{index}EGO",
                    "subject": subject,
                    "verified": "Yes",
                    "length": "10",
                    "actions": action_string,
                }
            )
    selected = MODULE._reconstruct_prior_activity_selection(rows, fixture)
    assert {key: len(value) for key, value in selected.items()} == {
        "development": 3,
        "holdout": 3,
    }
    assert not (
        {row["subject"] for row in selected["development"]}
        & {row["subject"] for row in selected["holdout"]}
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


def test_construct_aligned_combined_gate_ignores_only_action_performance() -> None:
    axes = {
        axis: {"status": "PASS"}
        for axis in (
            *MODULE.TUPLE_CRITICAL_AXIS_IDS,
            *MODULE.TUPLE_SUPPORTING_AXIS_IDS,
        )
    }
    axes[MODULE.TUPLE_SUPPORTING_AXIS_IDS[0]] = {"status": "UNMEASURED"}
    passed = MODULE._tuple_combined_public_gate(
        axes,
        {"status": "PASS"},
        action_control_blocks=False,
    )
    diagnostic_no_go = MODULE._tuple_combined_public_gate(
        axes,
        {"status": "NO_GO_DIAGNOSTIC"},
        action_control_blocks=False,
    )
    assert passed["status"] == diagnostic_no_go["status"] == "PASS"
    assert diagnostic_no_go["action_control_used_in_gate"] is False
    assert diagnostic_no_go["validated_axis_count"] == 6

    axes[MODULE.TUPLE_CRITICAL_AXIS_IDS[0]] = {"status": "NO_GO"}
    cannot_rescue = MODULE._tuple_combined_public_gate(
        axes,
        {"status": "PASS"},
        action_control_blocks=False,
    )
    assert cannot_rescue["status"] == "NO_GO"
    assert cannot_rescue["critical_axis_failures"] == [
        MODULE.TUPLE_CRITICAL_AXIS_IDS[0]
    ]

    integrity = {
        "failure_count": 1,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
        "error_module_count": 1,
        "unaccounted_failure_count": 0,
    }
    integrity_blocked = MODULE._apply_tuple_integrity_gate(
        diagnostic_no_go, integrity
    )
    assert integrity_blocked["status"] == "NO_GO"
    assert "integrity:failure_count" in integrity_blocked[
        "combined_gate_failures"
    ]


def test_tuple_integrity_failure_blocks_otherwise_passing_combined_gate() -> None:
    axes = {
        axis: {"status": "PASS"}
        for axis in (
            *MODULE.TUPLE_CRITICAL_AXIS_IDS,
            *MODULE.TUPLE_SUPPORTING_AXIS_IDS,
        )
    }
    combined = MODULE._tuple_combined_public_gate(axes, {"status": "PASS"})
    integrity = {
        "failure_count": 0,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
        "error_module_count": 1,
        "unaccounted_failure_count": 0,
    }
    result = MODULE._apply_tuple_integrity_gate(combined, integrity)
    assert result["status"] == "NO_GO"
    assert result["combined_gate_failures"] == ["integrity:error_module_count"]


def test_tuple_qualification_record_rejects_nonfinite_or_nonjson_values() -> None:
    import pytest

    MODULE._validate_tuple_qualification_record(
        {"status": "PASS", "metrics": {"coverage": 1.0}, "rows": []}
    )
    with pytest.raises(
        RuntimeError, match="E_TUPLE_QUALIFICATION_NONFINITE_RESULT"
    ):
        MODULE._validate_tuple_qualification_record({"metrics": {"score": float("nan")}})
    with pytest.raises(RuntimeError, match="E_TUPLE_QUALIFICATION_NONJSON_RESULT"):
        MODULE._validate_tuple_qualification_record({"rows": [Path("public.bin")]})


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


def test_tuple_public_file_download_retries_silent_content_truncation(
    tmp_path: Path, monkeypatch
) -> None:
    payload = b"complete public fixture"
    attempts = []

    class Response(BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def open_with_truncation(_request, timeout):
        assert timeout == 300
        attempts.append(True)
        return Response(payload[:5] if len(attempts) == 1 else payload)

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", open_with_truncation)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    target = tmp_path / "fixture.zip"
    MODULE._download_public_file("https://example.test/fixture.zip", target)
    assert len(attempts) == 2
    assert target.read_bytes() == payload


def test_tuple_public_file_download_fails_closed_after_silent_truncation(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    class Response(BytesIO):
        headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        MODULE.urllib.request,
        "urlopen",
        lambda _request, timeout: Response(b"short"),
    )
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    target = tmp_path / "fixture.zip"
    with pytest.raises(RuntimeError, match="E_TUPLE_PUBLIC_FILE_TRUNCATED"):
        MODULE._download_public_file("https://example.test/fixture.zip", target)
    assert not target.exists()
    assert not target.with_suffix(".zip.partial").exists()


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


def test_tuple_qualification_execution_is_nested_in_committed_correction() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    correction = MODULE._tuple_visor_hos_correction_amendment(config)
    execution = MODULE._tuple_qualification_execution(config)
    assert execution is correction["qualification_execution_clarification"]
    assert execution["grid_selection"]["action"]["abstention_margin_grid"] == [
        0.0,
        0.005,
        0.01,
        0.02,
        0.05,
    ]
    assert execution["pHash"]["near_duplicate_hamming_distance_max"] == 4


def test_tuple_binary_metrics_count_abstention_as_a_positive_miss() -> None:
    metrics = MODULE._binary_classification_metrics(
        [True, True, False, False], [True, None, False, True]
    )
    assert metrics == {
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.5,
        "f1": 0.5,
        "balanced_accuracy": 0.5,
        "coverage": 0.75,
    }


def test_tuple_weighted_kappa_excludes_abstentions_but_not_from_coverage() -> None:
    labels = ["0", "1", "2plus"]
    assert MODULE._weighted_kappa(
        ["0", "1", "2plus", "2plus"],
        ["0", None, "2plus", "1"],
        labels,
    ) < 1.0
    assert MODULE._weighted_kappa(labels, labels, labels) == 1.0


def test_referent_adapter_observation_uses_actual_accepted_lexical_span() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    adjudication = {
        "status": "ACCEPT",
        "reason": None,
        "text_en": "red ball",
        "words": [
            {"word": "rot", "start": 2.5, "end": 3.1, "probability": 0.9},
            {"word": "Ball", "start": 3.2, "end": 4.5, "probability": 0.9},
        ],
    }
    observed = MODULE._tuple_build_referent_adapter_observation(
        adjudication,
        7.0,
        MODULE._tuple_amendment(config),
        MODULE._tuple_public_ontology_mapping(
            MODULE._tuple_fixture_preparation_amendment(config)
        ),
        tagger=lambda tokens: [(tokens[0], "JJ"), (tokens[1], "NN")],
        lemmatize=lambda token, _part: token,
        zipf_frequency=lambda lemma, _language: {"red": 4.5, "ball": 3.5}[lemma],
        frequency_bands={"low": [0.0, 3.0], "mid": [3.0, 4.0], "high": [4.0, 8.0]},
    )
    assert observed["status"] == "ACCEPT"
    assert observed["abstention_reason"] is None
    assert observed["category"] == "sports ball"
    assert [(row["lemma"], row["part_of_speech"]) for row in observed["mentions"]] == [
        ("red", "adjective"),
        ("ball", "noun"),
    ]
    assert observed["samples"] == {
        "before": [0.0, 1.0, 2.0],
        "during": [2.833333, 3.5, 4.166667],
        "after": [5.0, 6.0, 7.0],
    }


def test_referent_adapter_observation_abstains_without_mapped_noun() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    observed = MODULE._tuple_build_referent_adapter_observation(
        {
            "status": "ACCEPT",
            "reason": None,
            "text_en": "quickly",
            "words": [
                {"word": "schnell", "start": 2.5, "end": 4.5, "probability": 0.9}
            ],
        },
        7.0,
        MODULE._tuple_amendment(config),
        MODULE._tuple_public_ontology_mapping(
            MODULE._tuple_fixture_preparation_amendment(config)
        ),
        tagger=lambda tokens: [(tokens[0], "RB")],
        lemmatize=lambda token, _part: token,
        zipf_frequency=lambda _lemma, _language: 4.0,
        frequency_bands={"low": [0.0, 3.0], "mid": [3.0, 4.0], "high": [4.0, 8.0]},
    )
    assert observed == {
        "status": "ABSTAIN",
        "reason": "ONTOLOGY_UNMATCHED",
        "abstention_reason": "ONTOLOGY_UNMATCHED",
        "mentions": [],
    }


def test_referent_language_artifacts_require_pinned_local_revision(
    tmp_path: Path, monkeypatch,
) -> None:
    whisper = tmp_path / "models/whisper/small.pt"
    whisper.parent.mkdir(parents=True)
    whisper.write_bytes(b"small")
    translation = tmp_path / "models/opus-mt-de-en"
    translation.mkdir(parents=True)
    for name in (
        "config.json",
        "source.spm",
        "target.spm",
        "tokenizer_config.json",
        "model.safetensors",
    ):
        (translation / name).write_bytes(name.encode())
    metadata = translation / ".cache/huggingface/download/model.safetensors.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{MODULE.TUPLE_OPUS_MT_DE_EN_REVISION}\netag\n0\n")
    monkeypatch.setattr(
        MODULE, "TUPLE_WHISPER_SMALL_SHA256", MODULE.file_digest(whisper)
    )
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_DISABLED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    record = MODULE._tuple_language_artifact_record(tmp_path)
    assert record["opus_mt_de_en_revision"] == MODULE.TUPLE_OPUS_MT_DE_EN_REVISION
    assert record["opus_mt_de_en_file_count"] == 5
    metadata.write_text("0" * 40 + "\netag\n0\n")
    import pytest

    with pytest.raises(RuntimeError, match="E_TUPLE_REFERENT_TRANSLATION_REVISION"):
        MODULE._tuple_language_artifact_record(tmp_path)


def _referent_truth_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    from PIL import Image, ImageDraw

    fixture_root = tmp_path / "fixtures"
    mask_root = fixture_root / "truth"
    mask_root.mkdir(parents=True)
    target = Image.new("L", (32, 24), 0)
    ImageDraw.Draw(target).rectangle((12, 6, 24, 18), fill=255)
    distractor = Image.new("L", (32, 24), 0)
    ImageDraw.Draw(distractor).rectangle((2, 7, 10, 17), fill=255)
    target_path = mask_root / "target.png"
    distractor_path = mask_root / "distractor.png"
    target.save(target_path)
    distractor.save(distractor_path)
    sampled = []
    for ordinal, phase in enumerate(
        ["before"] * 3 + ["during"] * 3 + ["after"] * 3
    ):
        sampled.append(
            {
                "sample_time": float(ordinal),
                "phase": phase,
                "target_mask_relative_path": "truth/target.png",
                "target_mask_sha256": MODULE.file_digest(target_path),
                "distractor_mask_relative_path": "truth/distractor.png",
                "distractor_mask_sha256": MODULE.file_digest(distractor_path),
            }
        )
    row: dict[str, object] = {
        "fixture_ordinal": 1,
        "category": "cup",
        "scenario": "persistent_ambiguous",
        "source_image_id": "target-image",
        "source_annotation_id": "target-annotation",
        "source_image_sha256": "1" * 64,
        "distractor_source_category": "cup",
        "distractor_source_image_id": "distractor-image",
        "distractor_source_annotation_id": "distractor-annotation",
        "distractor_source_image_sha256": "2" * 64,
        "distractor_source_distinct_from_target": True,
        "truth": {
            "visibility_by_phase": {phase: True for phase in ("before", "during", "after")},
            "dominance_by_phase": {phase: False for phase in ("before", "during", "after")},
            "candidate_count_by_phase": {phase: "2plus" for phase in ("before", "during", "after")},
            "sampled_mask_truth": sampled,
        },
    }
    return row, fixture_root


def test_referent_truth_requires_phase_counts_masks_and_distinct_same_category(
    tmp_path: Path,
) -> None:
    row, fixture_root = _referent_truth_fixture(tmp_path)
    truth = MODULE._tuple_referent_truth_record(row, fixture_root)
    assert truth["candidate_count_by_phase"] == {
        "before": "2plus",
        "during": "2plus",
        "after": "2plus",
    }
    changed = json.loads(json.dumps(row))
    changed["distractor_source_category"] = "bowl"
    import pytest

    with pytest.raises(RuntimeError, match="E_TUPLE_REFERENT_DISTRACTOR_PROVENANCE"):
        MODULE._tuple_referent_truth_record(changed, fixture_root)
    scalar = json.loads(json.dumps(row))
    scalar["truth"] = {"visibility": {"before": True}}
    with pytest.raises(RuntimeError, match="E_TUPLE_REFERENT_TRUTH_SCHEMA"):
        MODULE._tuple_referent_truth_record(scalar, fixture_root)


def test_grounding_phrase_box_and_nms_rules_are_exact() -> None:
    class Tokenizer:
        def __call__(self, value, add_special_tokens):
            tokens = {"sports ball.": [101, 2998, 3608, 1012, 102], "sports ball": [2998, 3608]}
            return {"input_ids": tokens[value]}

    assert MODULE._tuple_grounding_phrase_positions(
        Tokenizer(), "sports ball.", "sports ball"
    ) == [1, 2]
    assert MODULE._tuple_cxcywh_to_normalized_xyxy([0.5, 0.5, 0.4, 0.2]) == [
        0.3,
        0.4,
        0.7,
        0.6,
    ]
    assert MODULE._tuple_cxcywh_to_normalized_xyxy([0.5, 0.5, -1.0, 0.2]) is None
    retained = MODULE._tuple_same_category_nms(
        [
            {"box": [0.1, 0.1, 0.5, 0.5], "box_score": 0.9, "text_score": 0.8},
            {"box": [0.12, 0.12, 0.5, 0.5], "box_score": 0.8, "text_score": 0.8},
            {"box": [0.6, 0.6, 0.9, 0.9], "box_score": 0.7, "text_score": 0.7},
        ],
        0.5,
    )
    assert [row["box_score"] for row in retained] == [0.9, 0.7]


def _perfect_referent_track(
    ordinal: int, truth_count: str, dominant: bool | None
) -> dict[str, object]:
    visible = truth_count != "0"
    candidates = []
    if visible:
        candidates.append(
            {
                "category": "cup",
                "box": [0.35, 0.25, 0.65, 0.75],
                "box_score": 0.3,
                "text_score": 0.25,
                "valid": True,
                "mask_fraction": 0.2,
                "center_distance": 0.0,
            }
        )
    if truth_count == "2plus":
        candidates.append(
            {
                "category": "cup",
                "box": [0.05, 0.25, 0.25, 0.65],
                "box_score": 0.3,
                "text_score": 0.25,
                "valid": True,
                "mask_fraction": 0.2,
                "center_distance": 0.35,
            }
        )
    phases = ("before", "during", "after")
    return {
        "fixture_ordinal": ordinal,
        "category": "cup",
        "adapter_observation": {
            "status": "ACCEPT",
            "abstention_reason": None,
            "category": "cup",
        },
        "truth": {
            "visibility_by_phase": {phase: visible for phase in phases},
            "dominance_by_phase": {phase: dominant if visible else None for phase in phases},
            "candidate_count_by_phase": {phase: truth_count for phase in phases},
        },
        "samples": [
            {
                "sample_time": float(index),
                "phase": phase,
                "inference_succeeded": True,
                "candidates": json.loads(json.dumps(candidates)),
            }
            for index, phase in enumerate(["before"] * 3 + ["during"] * 3 + ["after"] * 3)
        ],
    }


def test_referent_metrics_score_visibility_dominance_ambiguity_and_nulls() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    definitions = MODULE._tuple_axis(
        config, "utterance_centered_referent_visibility_dominance_ambiguity"
    )["definitions"]
    tracks = [
        _perfect_referent_track(0, "1", True),
        _perfect_referent_track(1, "2plus", False),
        _perfect_referent_track(2, "0", None),
    ]
    metrics, rows, integrity = MODULE._tuple_referent_metrics(
        tracks, 0.3, 0.25, definitions, 8
    )
    assert metrics == {
        "event_coverage": 1.0,
        "visibility_timing_macro_f1": 1.0,
        "no_referent_specificity": 1.0,
        "dominance_weighted_kappa": 1.0,
        "ambiguity_macro_f1": 1.0,
        "valid_geometry_and_monotonic_track_fraction": 1.0,
    }
    assert len(rows) == 3
    assert integrity == {
        "invalid_retained_record_count": 0,
        "inference_failure_count": 0,
        "scientific_abstention_count": 0,
        "category_error_count": 0,
        "eligible_event_count": 3,
        "measured_event_count": 3,
    }


def test_referent_invalid_retained_mask_abstains_event() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    definitions = MODULE._tuple_axis(
        config, "utterance_centered_referent_visibility_dominance_ambiguity"
    )["definitions"]
    track = _perfect_referent_track(0, "1", True)
    track["samples"][0]["candidates"][0]["valid"] = False
    track["samples"][0]["candidates"][0]["reason"] = "INVALID_OR_EMPTY_MASK"
    prediction = MODULE._tuple_referent_track_prediction(
        track, 0.2, 0.15, definitions, 8
    )
    assert prediction["status"] == "ABSTAIN"
    assert prediction["reason"] == "INVALID_OR_EMPTY_RETAINED_MASK"
    assert prediction["invalid_retained_record_count"] == 1


def test_referent_valid_ten_percent_abstention_is_coverage_not_integrity_failure() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    axis = MODULE._tuple_axis(
        config, "utterance_centered_referent_visibility_dominance_ambiguity"
    )
    tracks = [
        *(_perfect_referent_track(index, "1", True) for index in range(3)),
        *(_perfect_referent_track(index + 3, "2plus", False) for index in range(3)),
        *(_perfect_referent_track(index + 6, "0", None) for index in range(3)),
    ]
    abstained = _perfect_referent_track(9, "1", True)
    abstained["samples"] = abstained["samples"][:7]
    tracks.append(abstained)
    metrics, _rows, integrity = MODULE._tuple_referent_metrics(
        tracks, 0.3, 0.25, axis["definitions"], 8
    )
    assert metrics["event_coverage"] == 0.9
    assert metrics["valid_geometry_and_monotonic_track_fraction"] == 1.0
    assert integrity["scientific_abstention_count"] == 1
    assert integrity["inference_failure_count"] == 0
    assert integrity["invalid_retained_record_count"] == 0
    assert MODULE._tuple_referent_gate_pass(metrics, axis["public_gate"])


def test_referent_low_confidence_no_detection_is_scientific_abstention() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    axis = MODULE._tuple_axis(
        config, "utterance_centered_referent_visibility_dominance_ambiguity"
    )
    prediction = MODULE._tuple_referent_track_prediction(
        _perfect_referent_track(0, "1", True),
        0.4,
        0.3,
        axis["definitions"],
        8,
    )
    assert prediction["status"] == "ABSTAIN"
    assert prediction["reason"] == "LOW_CONFIDENCE_OR_NO_DETECTION"
    assert prediction["scientific_abstention_count"] == 1
    assert prediction["inference_failure_count"] == 0
    assert prediction["invalid_retained_record_count"] == 0


def test_referent_inference_exception_remains_integrity_blocking() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    axis = MODULE._tuple_axis(
        config, "utterance_centered_referent_visibility_dominance_ambiguity"
    )
    tracks = [
        _perfect_referent_track(0, "1", True),
        _perfect_referent_track(1, "2plus", False),
        _perfect_referent_track(2, "0", None),
    ]
    tracks[0]["samples"][0] = {
        **tracks[0]["samples"][0],
        "inference_succeeded": False,
        "error_code": "E_TUPLE_GROUNDING_OUTPUT_SHAPE",
        "candidates": [],
    }
    metrics, _rows, integrity = MODULE._tuple_referent_metrics(
        tracks, 0.3, 0.25, axis["definitions"], 8
    )
    assert integrity["inference_failure_count"] == 1
    assert integrity["scientific_abstention_count"] == 0
    assert not (
        MODULE._tuple_referent_gate_pass(metrics, axis["public_gate"])
        and not integrity["inference_failure_count"]
        and not integrity["invalid_retained_record_count"]
    )


def test_referent_category_mismatch_abstains_before_model_prompt(monkeypatch) -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    observation = {
        "status": "ACCEPT",
        "abstention_reason": None,
        "category": "cup",
        "samples": {
            "before": [0.0, 1.0, 2.0],
            "during": [3.0, 4.0, 5.0],
            "after": [6.0, 7.0],
        },
    }
    row = {
        "fixture_ordinal": 0,
        "category": "bowl",
        "scenario": "persistent_clear",
        "truth": {"attribute": "red"},
    }
    monkeypatch.setattr(
        MODULE,
        "_tuple_qualification_execution",
        lambda _cfg: {
            "phase_aggregation": {"minimum_valid_samples": 8},
            "grounding_geometry": {"same_category_NMS_IoU": 0.5},
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_frozen_threshold_grids",
        lambda _cfg: {
            "Grounding_DINO_box_score": (0.2,),
            "Grounding_DINO_text_score": (0.15,),
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_referent_adapter_observations",
        lambda _context: {0: observation},
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_referent_truth_record",
        lambda _row, _root: {
            "visibility_by_phase": {phase: True for phase in ("before", "during", "after")},
            "dominance_by_phase": {phase: True for phase in ("before", "during", "after")},
            "candidate_count_by_phase": {phase: "1" for phase in ("before", "during", "after")},
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_load_tuple_grounding_stack",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("category mismatch must not load or prompt the model")
        ),
    )
    tracks = MODULE._tuple_grounding_sampled_tracks(
        {
            "cfg": config,
            "rows": {"referent_attribute": [row]},
            "fixture_root": Path("unused"),
            "public_root": Path("unused"),
            "device": "cpu",
            "module_cache": {},
        }
    )
    assert tracks[0]["category_mismatch"] is True
    assert tracks[0]["samples"] == []
    axis = MODULE._tuple_axis(
        config, "utterance_centered_referent_visibility_dominance_ambiguity"
    )
    prediction = MODULE._tuple_referent_track_prediction(
        tracks[0], 0.2, 0.15, axis["definitions"], 8
    )
    assert prediction["status"] == "ABSTAIN"
    assert prediction["reason"] == "PUBLIC_CATEGORY_MISMATCH"
    assert prediction["scientific_abstention_count"] == 1
    assert prediction["category_error_count"] == 1
    assert prediction["inference_failure_count"] == 0


def test_tuple_grid_selection_uses_conservative_threshold_tie_order() -> None:
    selected = MODULE._select_frozen_grid_result(
        [
            {"eligible": True, "score": 0.8, "box": 0.2, "text": 0.3},
            {"eligible": True, "score": 0.8, "box": 0.3, "text": 0.2},
            {"eligible": False, "score": 0.9, "box": 0.4, "text": 0.4},
        ],
        primary_metric="score",
        threshold_fields=("box", "text"),
    )
    assert selected == {
        "eligible": True,
        "score": 0.8,
        "box": 0.3,
        "text": 0.2,
    }


def test_tuple_module_collection_does_not_abort_after_independent_failure() -> None:
    called = []

    def runner(module_id):
        def execute(_context):
            called.append(module_id)
            if module_id in {"referent", "hand_contact"}:
                raise RuntimeError(f"E_TEST_{module_id.upper()}")
            return {"status": "PASS"}

        return execute

    results = MODULE._collect_tuple_module_results(
        {
            module_id: runner(module_id)
            for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
        },
        {},
    )
    assert called == list(MODULE.TUPLE_QUALIFICATION_MODULE_IDS)
    assert results["referent"]["status"] == "ERROR"
    assert results["referent"]["error_code"] == "E_TEST_REFERENT"
    assert results["hand_contact"]["status"] == "ERROR"
    assert sum(row["status"] == "PASS" for row in results.values()) == 5


def test_tuple_missing_registered_module_is_an_error_not_a_pass(monkeypatch) -> None:
    monkeypatch.setitem(
        MODULE.TUPLE_MODULE_RUNNER_NAMES,
        "referent",
        "_tuple_intentionally_absent_module",
    )
    results = MODULE._collect_tuple_module_results(MODULE._tuple_module_runners(), {})
    assert results["referent"] == {
        "status": "ERROR",
        "error_code": "E_TUPLE_QUALIFICATION_MODULE_UNIMPLEMENTED",
        "metrics": {},
        "row_count": 0,
        "failure_count": 1,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
    }


def test_tuple_diagnostic_status_is_only_valid_for_order_action() -> None:
    def runner(module_id):
        def execute(_context):
            return {
                "status": "NO_GO_DIAGNOSTIC",
                "metrics": {},
                "row_count": 0,
                "failure_count": 0,
                "invalid_retained_record_count": 0,
                "silent_truncation_count": 0,
                "external_call_count": 0,
            }

        return execute

    results = MODULE._collect_tuple_module_results(
        {
            module_id: runner(module_id)
            for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
        },
        {},
    )
    assert results["order_action"]["status"] == "NO_GO_DIAGNOSTIC"
    for module_id in set(MODULE.TUPLE_QUALIFICATION_MODULE_IDS) - {"order_action"}:
        assert results[module_id]["status"] == "ERROR"
        assert (
            results[module_id]["error_code"]
            == "E_TUPLE_QUALIFICATION_MODULE_RESULT"
        )


def test_tuple_qualification_development_seals_then_holdout_cannot_refit(
    tmp_path: Path, monkeypatch
) -> None:
    import argparse
    import pytest

    config_path = Path("configs/synthetic_video_real_only_proof.json").resolve()
    config = json.loads(config_path.read_text())
    correction = MODULE._tuple_visor_hos_correction_amendment(config)
    manifest = {
        "public_fixture_manifest_commitment_sha256": "b" * 64,
        "visor_hos_correction_amendment_commitment_sha256": correction[
            "amendment_commitment_sha256"
        ],
        "verified_no_hand_seal_commitment_sha256": "a" * 64,
        "construct_aligned_ltx_resume_amendment_commitment_sha256": (
            MODULE.CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256
        ),
        "partitions": {
            "development": {},
            "holdout": {},
        },
    }
    called = []
    thresholds = {
        "referent": {
            "Grounding_DINO_box_score": 0.2,
            "Grounding_DINO_text_score": 0.15,
        },
        "recurrence": {"DINOv2_recurrence_cosine": 0.8},
        "attribute": {"PE_Core_attribute_margin": 0.0},
        "hand_contact": {"EgoHOS_min_mask_fraction": 0.0025},
        "order_action": {"action_abstention_margin": 0.0},
    }

    def runner(module_id):
        def execute(context):
            called.append((context["partition"], module_id))
            axis_results = {
                axis: {"status": "PASS", "metrics": {}}
                for axis in MODULE.TUPLE_MODULE_AXIS_IDS[module_id]
            }
            return {
                "status": (
                    "NO_GO_DIAGNOSTIC"
                    if module_id == "order_action"
                    else "PASS"
                ),
                "axis_results": axis_results,
                "metrics": {},
                "selected_thresholds": (
                    thresholds.get(module_id, {})
                    if context["partition"] == "development"
                    else {
                        key: context["thresholds"][key]
                        for key in thresholds.get(module_id, {})
                    }
                ),
                "rows": [],
                "row_count": 0,
                "failure_count": 0,
                "invalid_retained_record_count": 0,
                "silent_truncation_count": 0,
                "external_call_count": 0,
            }

        return execute

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setattr(MODULE, "_tuple_health_topology", lambda *_: None)
    monkeypatch.setattr(MODULE, "_verify_tuple_runtime_manifest", lambda *_: {})
    monkeypatch.setattr(
        MODULE,
        "_verify_tuple_fixture_manifest",
        lambda public, _cfg: (manifest, public / "fixtures"),
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_module_runners",
        lambda: {
            module_id: runner(module_id)
            for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
        },
    )
    health_commitment = "c" * 64
    dependency_commitment = "d" * 64
    monkeypatch.setattr(
        MODULE,
        "_load_tuple_health_pass",
        lambda *_: {
            "engineering_health_commitment_sha256": health_commitment,
            "dependency_config_commitment_sha256": dependency_commitment,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_health_dependency_preflight",
        lambda *_: {
            "dependency_config_commitment_sha256": dependency_commitment
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_partition_engineering_integrity",
        lambda *_: [
            MODULE._tuple_health_pass_result(
                module_id,
                1,
                {"module_id": module_id, "partition_integrity": True},
            )
            for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
        ],
    )
    public = tmp_path / "public"
    scratch = tmp_path / "scratch"
    base = {
        "public_root": public,
        "scratch_root": scratch,
        "config": config_path,
        "device": "cpu",
    }
    development = MODULE.qualify_tuple_public(
        argparse.Namespace(**base, partition="development")
    )
    assert development["status"] == "PASS_DEVELOPMENT_THRESHOLDS_SEALED"
    assert development["action_control_status"] == "NO_GO_DIAGNOSTIC"
    assert called == [
        ("development", module_id)
        for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
    ]
    paths = MODULE._tuple_qualification_paths(public)
    assert paths["development_result"].is_file()
    assert paths["development_threshold_seal"].is_file()
    transaction_path = MODULE._tuple_qualification_transaction_path(
        public, "development"
    )
    assert transaction_path.is_file()
    sealed_threshold_bytes = paths["development_threshold_seal"].read_bytes()
    paths["development_threshold_seal"].unlink()
    recovered = MODULE.qualify_tuple_public(
        argparse.Namespace(**base, partition="development")
    )
    assert recovered == development
    assert paths["development_threshold_seal"].read_bytes() == sealed_threshold_bytes
    assert len(called) == len(MODULE.TUPLE_QUALIFICATION_MODULE_IDS)
    reused = MODULE.qualify_tuple_public(
        argparse.Namespace(**base, partition="development")
    )
    assert reused == development
    assert len(called) == len(MODULE.TUPLE_QUALIFICATION_MODULE_IDS)

    holdout = MODULE.qualify_tuple_public(
        argparse.Namespace(**base, partition="holdout")
    )
    assert holdout["status"] == "PASS_PUBLIC_COMBINED_GATE"
    assert holdout["action_control_status"] == "NO_GO_DIAGNOSTIC"
    assert called[-7:] == [
        ("holdout", module_id)
        for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
    ]
    with pytest.raises(RuntimeError, match="E_TUPLE_HOLDOUT_RESULT_ALREADY_EXISTS"):
        MODULE.qualify_tuple_public(
            argparse.Namespace(**base, partition="holdout")
        )


def test_tuple_fixture_preparation_lineage_and_development_pair_fail_closed(
    tmp_path: Path,
) -> None:
    import pytest

    expected = {
        "fixture_preparation_amendment_commitment_sha256": "a" * 64,
        "visor_hos_correction_amendment_commitment_sha256": "b" * 64,
    }
    manifest = dict(expected)
    MODULE._verify_tuple_fixture_commitments(manifest, expected)
    manifest["fixture_preparation_amendment_commitment_sha256"] = "c" * 64
    with pytest.raises(
        RuntimeError, match="E_TUPLE_QUALIFICATION_FIXTURE_PROVENANCE"
    ):
        MODULE._verify_tuple_fixture_commitments(manifest, expected)

    paths = MODULE._tuple_qualification_paths(tmp_path)
    MODULE.write_private(paths["development_result"], {"partial": True})
    with pytest.raises(RuntimeError, match="E_TUPLE_DEVELOPMENT_PAIR_PARTIAL"):
        MODULE._load_tuple_development_pair(
            tmp_path,
            {},
            {},
            missing_code="E_TEST_MISSING",
        )


def test_prepare_tuple_fixtures_refuses_any_postqualification_replacement(
    tmp_path: Path, monkeypatch
) -> None:
    import argparse
    import pytest

    config_path = Path("configs/synthetic_video_real_only_proof.json").resolve()
    public = tmp_path / "public"
    paths = MODULE._tuple_qualification_paths(public)
    MODULE.write_private(paths["holdout_result"], {"sealed": True})
    monkeypatch.setattr(MODULE, "_verify_tuple_runtime_manifest", lambda *_: {})
    with pytest.raises(
        RuntimeError, match="E_TUPLE_FIXTURE_REPLACEMENT_AFTER_QUALIFICATION"
    ):
        MODULE.prepare_tuple_fixtures(
            argparse.Namespace(
                config=config_path,
                public_root=public,
                audio_seed_root=tmp_path / "audio",
                no_hand_review_root=None,
            )
        )

    reuse_public = tmp_path / "reuse-public"
    fixture_root = MODULE._tuple_fixture_root(reuse_public)
    manifest_path = fixture_root / "fixture-manifest.json"
    partitions = {
        partition: {
            family: []
            for family in (
                "language_lexical",
                "referent_attribute",
                "recurrence",
                "hand_contact",
                "sensor",
                "order_action",
            )
        }
        for partition in ("development", "holdout")
    }
    manifest = {
        "source_provenance": [],
        "partitions": partitions,
        "audits": {
            "source_subject_overlap_count": 0,
            "source_video_overlap_count": 0,
            "source_frame_overlap_count": 0,
            "source_object_overlap_count": 0,
            "target_hand_boundary_vertex_count": 0,
            "target_hand_outside_canvas_vertex_count": 0,
            "target_hand_boundary_item_count": 0,
            "target_hand_outside_canvas_item_count": 0,
        },
        "public_fixture_manifest_commitment_sha256": "d" * 64,
    }
    MODULE.write_private(manifest_path, manifest)
    before = manifest_path.read_bytes()
    for name in (
        "_tuple_amendment",
        "_tuple_visor_hos_correction_amendment",
        "_tuple_fixture_protocol",
        "_tuple_fixture_preparation_amendment",
        "_tuple_fixture_feasibility_repair",
        "_public_fixture_geometry_rasterization_repair",
    ):
        monkeypatch.setattr(MODULE, name, lambda *_: {})
    monkeypatch.setattr(
        MODULE,
        "_verify_tuple_fixture_manifest",
        lambda *_: (manifest, fixture_root),
    )
    reused = MODULE.prepare_tuple_fixtures(
        argparse.Namespace(
            config=config_path,
            public_root=reuse_public,
            audio_seed_root=tmp_path / "unused-audio",
            no_hand_review_root=None,
        )
    )
    assert reused["public_fixture_manifest_commitment_sha256"] == "d" * 64
    assert manifest_path.read_bytes() == before


def test_audio_seed_verifier_requires_current_attributive_recipe(tmp_path: Path) -> None:
    import pytest

    config = _current_audio_fixture_config()
    current = _write_public_audio_seed(tmp_path / "current", config)
    manifest, records = MODULE._read_audio_seed_manifest(current, config)
    assert manifest["schema_version"] == 2
    assert len(records) == 112

    predicative = _write_public_audio_seed(
        tmp_path / "predicative", config, predicative=True
    )
    with pytest.raises(RuntimeError, match="E_TUPLE_AUDIO_SEED_RECIPE"):
        MODULE._read_audio_seed_manifest(predicative, config)

    stale = _write_public_audio_seed(tmp_path / "stale", config, stale=True)
    with pytest.raises(RuntimeError, match="E_TUPLE_AUDIO_SEED_MANIFEST"):
        MODULE._read_audio_seed_manifest(stale, config)


def test_audio_seed_verifier_rejects_prior_external_root(tmp_path: Path) -> None:
    import pytest

    config = _current_audio_fixture_config()
    current = _write_public_audio_seed(tmp_path, config)
    prior = tmp_path / "public/mechanistic-tuple-audio-seed"
    current.rename(prior)
    with pytest.raises(RuntimeError, match="E_TUPLE_AUDIO_SEED_CANONICAL_ROOT"):
        MODULE._read_audio_seed_manifest(prior, config)


def test_tuple_qualification_rejects_threshold_outside_frozen_grid() -> None:
    import pytest

    config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    results = {
        module_id: {"selected_thresholds": {}}
        for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
    }
    results["recurrence"]["selected_thresholds"] = {
        "DINOv2_recurrence_cosine": 0.83
    }
    with pytest.raises(
        RuntimeError, match="E_TUPLE_QUALIFICATION_THRESHOLD_NOT_FROZEN"
    ):
        MODULE._tuple_selected_thresholds(config, results)


def test_tuple_fixture_file_rejects_traversal_and_hash_mismatch(tmp_path: Path) -> None:
    import pytest

    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    media = fixture_root / "public.png"
    media.write_bytes(b"public fixture")
    expected = MODULE.file_digest(media)
    assert MODULE._tuple_fixture_file(
        fixture_root, "public.png", expected, len(b"public fixture")
    ) == media
    with pytest.raises(RuntimeError, match="E_TUPLE_QUALIFICATION_MEDIA_PATH"):
        MODULE._tuple_fixture_file(fixture_root, "../escape", expected)
    with pytest.raises(RuntimeError, match="E_TUPLE_QUALIFICATION_MEDIA_HASH"):
        MODULE._tuple_fixture_file(fixture_root, "public.png", "0" * 64)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (fixture_root / "escape.bin").symlink_to(outside)
    with pytest.raises(RuntimeError, match="E_TUPLE_QUALIFICATION_MEDIA_PATH"):
        MODULE._tuple_fixture_file(
            fixture_root, "escape.bin", MODULE.file_digest(outside)
        )


def test_write_fixture_video_uses_mp4_safe_atomic_temporary(
    tmp_path: Path, monkeypatch,
) -> None:
    import sys
    from types import ModuleType, SimpleNamespace

    target = tmp_path / "fixture.mp4"

    class Writer:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)

        def append_data(self, _frame: object) -> None:
            return None

        def close(self) -> None:
            self.path.write_bytes(b"video-only")

    imageio_package = ModuleType("imageio")
    imageio_package.__path__ = []
    imageio_v2 = ModuleType("imageio.v2")
    imageio_v2.get_writer = lambda path, **_kwargs: Writer(path)
    imageio_package.v2 = imageio_v2
    imageio_ffmpeg = ModuleType("imageio_ffmpeg")
    imageio_ffmpeg.get_ffmpeg_exe = lambda: "ffmpeg"
    monkeypatch.setitem(sys.modules, "imageio", imageio_package)
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio_v2)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", imageio_ffmpeg)
    filters: list[str] = []

    def encode(command, **_kwargs):
        output = Path(command[-1])
        assert output == tmp_path / "fixture.partial.mp4"
        assert output.suffix == ".mp4"
        assert command[command.index("-c:v") + 1] == "copy"
        assert command[command.index("-movflags") + 1] == "+faststart"
        audio_filter = command[command.index("-filter_complex") + 1]
        assert ":d=" not in audio_filter
        assert "all=" not in audio_filter
        assert "anullsrc=r=22050:cl=mono,atrim=duration=1.0" in audio_filter
        filters.append(audio_filter)
        output.write_bytes(b"encoded-mp4")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(MODULE.subprocess, "run", encode)
    MODULE._write_fixture_video(
        frames=[object()], fps=1, duration=1.0, audio=None, target=target
    )
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"public speech fixture")
    MODULE._write_fixture_video(
        frames=[object()], fps=1, duration=1.0, audio=audio, target=target
    )

    assert target.read_bytes() == b"encoded-mp4"
    assert not list(tmp_path.glob("*partial*"))
    assert filters == [
        "anullsrc=r=22050:cl=mono,atrim=duration=1.0[a]",
        (
            "anullsrc=r=22050:cl=mono,atrim=duration=1.0[base];"
            "[1:a]atrim=duration=2.0,adelay=2500[spoken];"
            "[base][spoken]amix=inputs=2:duration=first[a]"
        ),
    ]


def test_tuple_fixture_manifest_hashes_all_nested_files_before_inference(
    tmp_path: Path,
) -> None:
    import pytest

    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    media = fixture_root / "public.bin"
    media.write_bytes(b"public fixture")
    record = {
        "nested": [
            {
                "media_relative_path": "public.bin",
                "media_sha256": MODULE.file_digest(media),
                "media_bytes": media.stat().st_size,
                "optional_mask_relative_path": None,
                "optional_mask_sha256": None,
            }
        ]
    }
    assert MODULE._verify_tuple_fixture_files_recursive(record, fixture_root) == 1
    del record["nested"][0]["media_sha256"]
    with pytest.raises(
        RuntimeError, match="E_TUPLE_QUALIFICATION_MEDIA_HASH_FIELD"
    ):
        MODULE._verify_tuple_fixture_files_recursive(record, fixture_root)


def test_tuple_perceptual_hash_is_deterministic_and_sensitive() -> None:
    from PIL import Image, ImageDraw

    first = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(first).rectangle((4, 4, 28, 50), fill="black")
    second = first.copy()
    third = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(third).ellipse((25, 4, 60, 60), fill="black")
    assert MODULE._tuple_perceptual_hash(first) == MODULE._tuple_perceptual_hash(second)
    assert (
        MODULE._tuple_perceptual_hash(first)
        ^ MODULE._tuple_perceptual_hash(third)
    ).bit_count() > 4


def test_tuple_action_prediction_applies_pair_margin_and_controls() -> None:
    labels = ["open", "close"]
    rows = [
        {
            "label": "open",
            "ordered_scores": [0.9, 0.1],
            "reversed_scores": [0.2, 0.8],
            "repeated_center_scores": [0.4, 0.3],
        },
        {
            "label": "close",
            "ordered_scores": [0.2, 0.8],
            "reversed_scores": [0.7, 0.3],
            "repeated_center_scores": [0.4, 0.45],
        },
    ]
    predicted, metrics = MODULE._tuple_action_predictions(
        rows, 0.05, labels, {"open": "close", "close": "open"}
    )
    assert predicted == labels
    assert metrics == {
        "ordered_action_direction_macro_f1": 1.0,
        "opposite_pair_correct_margin_fraction": 1.0,
        "ordered_over_time_reversed_target_score_fraction": 1.0,
        "ordered_over_repeated_center_confidence_fraction": 1.0,
        "coverage": 1.0,
    }


def test_tuple_action_grid_rejects_higher_f1_that_fails_temporal_floor() -> None:
    gate = {
        "ordered_action_direction_macro_f1_min": 0.6,
        "opposite_pair_correct_margin_fraction_min": 0.7,
        "ordered_over_time_reversed_target_score_fraction_min": 0.7,
        "ordered_over_repeated_center_confidence_fraction_min": 0.7,
        "coverage_min": 0.8,
    }
    selected = MODULE._select_tuple_action_development(
        [
            {
                "abstention_margin": 0.0,
                "ordered_action_direction_macro_f1": 0.9,
                "opposite_pair_correct_margin_fraction": 0.9,
                "ordered_over_time_reversed_target_score_fraction": 0.6,
                "ordered_over_repeated_center_confidence_fraction": 0.9,
                "coverage": 1.0,
            },
            {
                "abstention_margin": 0.05,
                "ordered_action_direction_macro_f1": 0.8,
                "opposite_pair_correct_margin_fraction": 0.8,
                "ordered_over_time_reversed_target_score_fraction": 0.8,
                "ordered_over_repeated_center_confidence_fraction": 0.8,
                "coverage": 0.9,
            },
        ],
        gate,
    )
    assert selected is not None
    assert selected["abstention_margin"] == 0.05
    assert selected["ordered_action_direction_macro_f1"] == 0.8
    assert selected["eligible"] is True
    invalid = dict(selected)
    invalid["invalid_retained_record_count"] = 1
    assert MODULE._tuple_action_metrics_pass(invalid, gate) is False


def test_tuple_action_diagnostic_fallback_is_deterministic_and_nonpassing() -> None:
    gate = {
        "ordered_action_direction_macro_f1_min": 0.6,
        "opposite_pair_correct_margin_fraction_min": 0.7,
        "ordered_over_time_reversed_target_score_fraction_min": 0.7,
        "ordered_over_repeated_center_confidence_fraction_min": 0.7,
        "coverage_min": 0.8,
    }
    shared = {
        "opposite_pair_correct_margin_fraction": 0.9,
        "ordered_over_time_reversed_target_score_fraction": 0.6,
        "ordered_over_repeated_center_confidence_fraction": 0.9,
        "coverage": 1.0,
    }
    selected, passed = MODULE._select_tuple_action_diagnostic(
        [
            {
                "abstention_margin": 0.0,
                "ordered_action_direction_macro_f1": 0.55,
                **shared,
            },
            {
                "abstention_margin": 0.02,
                "ordered_action_direction_macro_f1": 0.58,
                **shared,
            },
            {
                "abstention_margin": 0.05,
                "ordered_action_direction_macro_f1": 0.58,
                **shared,
            },
        ],
        gate,
    )
    assert passed is False
    assert selected["abstention_margin"] == 0.05
    assert selected["ordered_action_direction_macro_f1"] == 0.58
    assert selected["eligible"] is False


def _egohos_test_masks(contact: bool | None):
    import numpy as np

    stage1 = np.zeros((20, 20), dtype=np.uint8)
    stage2 = np.zeros((20, 20), dtype=np.uint8)
    stage3 = np.zeros((20, 20), dtype=np.uint8)
    stage1[5:10, 5:10] = 1
    if contact is True:
        stage2[5:10, 10] = 1
        stage3[5:10, 11:15] = 1
    elif contact is None:
        stage3[14:18, 14:18] = 1
    return {"stage1": stage1, "stage2": stage2, "stage3": stage3}


def _egohos_official_test_config(additional_channel=None):
    config = {
        "data": {
            "test": {
                "pipeline": [
                    {"type": "LoadImageFromFile"},
                    {
                        "type": "MultiScaleFlipAug",
                        "img_scale": (360, 480),
                        "flip": False,
                        "transforms": [
                            {"type": "Resize", "keep_ratio": True},
                            {"type": "RandomFlip"},
                            {
                                "type": "Normalize",
                                "mean": [123.675, 116.28, 103.53],
                                "std": [58.395, 57.12, 57.375],
                                "to_rgb": True,
                            },
                            {"type": "ImageToTensor", "keys": ["img"]},
                            {"type": "Collect", "keys": ["img"]},
                        ],
                    },
                ]
            }
        }
    }
    if additional_channel is not None:
        config["additional_channel"] = additional_channel
    return config


def test_tuple_egohos_official_stage_uses_pinned_pipeline_and_original_shape(
    tmp_path: Path,
) -> None:
    import numpy as np
    import pytest
    from PIL import Image

    image_path = tmp_path / "images/public-frame.jpg"
    image_path.parent.mkdir()
    Image.new("RGB", (61, 37), "white").save(image_path)
    config = _egohos_official_test_config()

    class Model:
        cfg = None

    model = Model()
    calls = []

    def inference(instance, path):
        calls.append(path)
        assert instance is model
        assert instance.cfg is config
        assert instance.cfg["additional_channel"] == "twohands"
        return [np.zeros((37, 61), dtype=np.uint8)]

    predictions = MODULE._tuple_egohos_official_stage(
        model,
        config,
        [image_path],
        [(37, 61)],
        stage="stage2",
        run_root=tmp_path,
        output_folder="pred_cb",
        inference_segmentor=inference,
    )
    assert calls == [str(image_path)]
    assert predictions[0].shape == (37, 61)
    with Image.open(tmp_path / "pred_cb/public-frame.png") as prediction:
        assert prediction.size == (61, 37)
    changed = _egohos_official_test_config("twohands")
    changed["data"]["test"]["pipeline"][1]["transforms"][0][
        "keep_ratio"
    ] = False
    with pytest.raises(RuntimeError, match="E_TUPLE_EGOHOS_TEST_PIPELINE"):
        MODULE._tuple_egohos_official_stage(
            model,
            changed,
            [image_path],
            [(37, 61)],
            stage="stage2",
            run_root=tmp_path,
            output_folder="pred_cb",
            inference_segmentor=inference,
        )


def test_tuple_egohos_staging_preserves_source_bytes_and_never_force_resizes(
    tmp_path: Path, monkeypatch
) -> None:
    from PIL import Image

    fixture_root = tmp_path / "fixtures"
    source = fixture_root / "media/original.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (61, 37), "white").save(source, quality=91)
    source_hash = MODULE.file_digest(source)
    captured = {}

    def official(_public, _device, paths, shapes, _run_root):
        captured["shape"] = shapes[0]
        captured["hash"] = MODULE.file_digest(paths[0])
        captured["suffix"] = paths[0].suffix
        return [_egohos_test_masks(False)]

    monkeypatch.setattr(MODULE, "_tuple_egohos_run_official_pipeline", official)
    MODULE._tuple_egohos_predict_rows(
        tmp_path / "public",
        "cpu",
        [
            {
                "media_relative_path": "media/original.jpg",
                "media_sha256": source_hash,
                "media_bytes": source.stat().st_size,
            }
        ],
        fixture_root,
        tmp_path / "scratch",
    )
    assert captured == {
        "shape": (37, 61),
        "hash": source_hash,
        "suffix": ".jpg",
    }


def test_tuple_egohos_sizing_uses_same_official_pipeline_without_crop(
    tmp_path: Path, monkeypatch
) -> None:
    import numpy as np
    from PIL import Image

    captured = {}

    def official(_public, _device, paths, shapes, _run_root):
        captured["shape"] = shapes[0]
        with Image.open(paths[0]) as image:
            captured["image_size"] = image.size
        return [
            {
                "stage1": np.zeros(shapes[0], dtype=np.uint8),
                "stage2": np.zeros(shapes[0], dtype=np.uint8),
                "stage3": np.zeros(shapes[0], dtype=np.uint8),
            }
        ]

    monkeypatch.setattr(MODULE, "_tuple_egohos_run_official_pipeline", official)
    result = MODULE._size_tuple_egohos(
        tmp_path / "public",
        "cpu",
        np.zeros((37, 61, 3), dtype=np.uint8),
        tmp_path / "scratch",
    )
    assert captured == {"shape": (37, 61), "image_size": (61, 37)}
    assert result == {"finite": True, "output_width": 37 * 61 * 3}


def test_tuple_egohos_mapping_uses_target_side_object_and_dilated_boundary() -> None:
    import numpy as np
    import pytest

    target = np.zeros((20, 20), dtype=np.uint8)
    target[5:10, 5:10] = 255
    positive = MODULE._tuple_egohos_mask_observation(
        _egohos_test_masks(True),
        minimum_mask_fraction=0.01,
        target_hand_side="left hand",
        target_hand_mask=target,
    )
    assert positive["status"] == "MEASURED"
    assert positive["hand_visible"] is True
    assert positive["contact"] is True
    assert positive["adjacent_contact_boundary_present"] is True
    assert positive["object_boundary_overlap_present"] is True

    both_hands_object = _egohos_test_masks(True)
    both_hands_object["stage3"][both_hands_object["stage3"] == 1] = 3
    assert MODULE._tuple_egohos_mask_observation(
        both_hands_object,
        minimum_mask_fraction=0.01,
        target_hand_side="left hand",
        target_hand_mask=target,
    )["contact"] is True

    wrong_side_object = _egohos_test_masks(True)
    wrong_side_object["stage3"][wrong_side_object["stage3"] == 1] = 2
    wrong_side = MODULE._tuple_egohos_mask_observation(
        wrong_side_object,
        minimum_mask_fraction=0.01,
        target_hand_side="left hand",
        target_hand_mask=target,
    )
    assert wrong_side["status"] == "ABSTAIN"
    assert wrong_side["contact"] is None

    negative = MODULE._tuple_egohos_mask_observation(
        _egohos_test_masks(False),
        minimum_mask_fraction=0.01,
        target_hand_side="left hand",
        target_hand_mask=target,
    )
    assert negative["status"] == "MEASURED"
    assert negative["contact"] is False

    discordant = MODULE._tuple_egohos_mask_observation(
        _egohos_test_masks(None),
        minimum_mask_fraction=0.01,
        target_hand_side="left hand",
        target_hand_mask=target,
    )
    assert discordant["status"] == "ABSTAIN"
    assert discordant["reason"] == "CONTACT_EVIDENCE_DISCORDANT"
    assert discordant["contact"] is None

    invalid = _egohos_test_masks(True)
    invalid["stage3"][0, 0] = 4
    with pytest.raises(RuntimeError, match="E_TUPLE_EGOHOS_STAGE3_MASK"):
        MODULE._tuple_egohos_mask_observation(
            invalid,
            minimum_mask_fraction=0.01,
            target_hand_side="left hand",
            target_hand_mask=target,
        )


def test_tuple_egohos_verified_no_hand_requires_pass_seal_and_row_link() -> None:
    import pytest

    commitment = "a" * 64
    rows = [
        {
            "stratum": "contact",
            "contact": True,
            "target_hand_side": "left hand",
            "target_hand_mask_relative_path": "masks/left.png",
            "target_hand_mask_sha256": "b" * 64,
            "target_hand_mask_width": 20,
            "target_hand_mask_height": 20,
            "target_hand_boundary_vertex_count": 0,
            "target_hand_outside_canvas_vertex_count": 0,
            "target_hand_outside_canvas_component_count": 0,
            "source_polygon_finite_nonnegative": True,
            "target_hand_mask_exact_frame_binary_nonempty": True,
            "geometry_valid": True,
            "media_relative_path": "media/contact.png",
            "media_sha256": "c" * 64,
        },
        {
            "stratum": "verified_no_hand",
            "contact": None,
            "target_hand_side": None,
            "target_hand_mask_relative_path": None,
            "target_hand_mask_sha256": None,
            "target_hand_mask_bytes": None,
            "target_hand_mask_width": None,
            "target_hand_mask_height": None,
            "target_hand_boundary_vertex_count": 0,
            "target_hand_outside_canvas_vertex_count": 0,
            "target_hand_outside_canvas_component_count": 0,
            "target_hand_mask_exact_frame_binary_nonempty": False,
            "verified_no_hand_seal_commitment_sha256": commitment,
            "media_relative_path": "media/no-hand.png",
            "media_sha256": "d" * 64,
        },
    ]
    context = {
        "verified_no_hand_seal": {
            "status": "PASS",
            "verified_no_hand_seal_commitment_sha256": commitment,
        }
    }
    assert MODULE._validate_tuple_egohos_fixture_rows(context, rows) == commitment
    broken = json.loads(json.dumps(context))
    broken["verified_no_hand_seal"]["status"] = "NO_GO"
    with pytest.raises(RuntimeError, match="E_TUPLE_EGOHOS_NO_HAND_PASS_SEAL"):
        MODULE._validate_tuple_egohos_fixture_rows(broken, rows)
    broken = json.loads(json.dumps(rows))
    broken[0].pop("target_hand_mask_relative_path")
    with pytest.raises(
        RuntimeError, match="E_TUPLE_EGOHOS_VISIBLE_HAND_FIXTURE_TRUTH"
    ):
        MODULE._validate_tuple_egohos_fixture_rows(context, broken)
    broken = json.loads(json.dumps(rows))
    broken[0]["target_hand_mask_width"] = True
    with pytest.raises(
        RuntimeError, match="E_TUPLE_EGOHOS_VISIBLE_HAND_FIXTURE_TRUTH"
    ):
        MODULE._validate_tuple_egohos_fixture_rows(context, broken)


def test_tuple_egohos_metrics_separate_no_hand_from_contact_state(
    tmp_path: Path,
) -> None:
    import numpy as np
    from PIL import Image

    fixture_root = tmp_path / "fixtures"
    mask_path = fixture_root / "masks/left.png"
    mask_path.parent.mkdir(parents=True)
    target = np.zeros((20, 20), dtype=np.uint8)
    target[5:10, 5:10] = 255
    Image.fromarray(target).save(mask_path)
    mask_hash = MODULE.file_digest(mask_path)
    rows = [
        {
            "fixture_ordinal": 0,
            "stratum": "contact",
            "contact": True,
            "target_hand_side": "left hand",
            "target_hand_mask_relative_path": "masks/left.png",
            "target_hand_mask_sha256": mask_hash,
            "target_hand_mask_width": 20,
            "target_hand_mask_height": 20,
        },
        {
            "fixture_ordinal": 1,
            "stratum": "explicit_no_contact",
            "contact": False,
            "target_hand_side": "left hand",
            "target_hand_mask_relative_path": "masks/left.png",
            "target_hand_mask_sha256": mask_hash,
            "target_hand_mask_width": 20,
            "target_hand_mask_height": 20,
        },
        {
            "fixture_ordinal": 2,
            "stratum": "verified_no_hand",
            "contact": None,
            "target_hand_side": None,
        },
    ]
    no_hand = _egohos_test_masks(False)
    no_hand["stage1"][:] = 0
    metrics = MODULE._tuple_egohos_threshold_metrics(
        rows,
        [_egohos_test_masks(True), _egohos_test_masks(False), no_hand],
        fixture_root,
        0.01,
    )
    assert metrics["hand_sensitivity"] == 1.0
    assert metrics["hand_specificity"] == 1.0
    assert metrics["contact_no_contact_macro_f1"] == 1.0
    assert metrics["mention_contact_alignment_f1"] == 1.0
    assert metrics["coverage"] == 1.0
    assert metrics["visible_hand_item_count"] == 2
    assert metrics["verified_no_hand_item_count"] == 1
    assert metrics["contact_metric_item_count"] == 2


def test_tuple_egohos_invalid_stage_record_abstains_and_blocks_gate(
    tmp_path: Path,
) -> None:
    import numpy as np
    from PIL import Image

    fixture_root = tmp_path / "fixtures"
    mask_path = fixture_root / "masks/left.png"
    mask_path.parent.mkdir(parents=True)
    target = np.zeros((20, 20), dtype=np.uint8)
    target[5:10, 5:10] = 255
    Image.fromarray(target).save(mask_path)
    row = {
        "fixture_ordinal": 0,
        "stratum": "contact",
        "contact": True,
        "target_hand_side": "left hand",
        "target_hand_mask_relative_path": "masks/left.png",
        "target_hand_mask_sha256": MODULE.file_digest(mask_path),
        "target_hand_mask_width": 20,
        "target_hand_mask_height": 20,
    }
    invalid = _egohos_test_masks(True)
    invalid["stage2"][0, 0] = 2
    metrics = MODULE._tuple_egohos_threshold_metrics(
        [row], [invalid], fixture_root, 0.01
    )
    assert metrics["rows"][0]["status"] == "ABSTAIN"
    assert metrics["rows"][0]["reason"] == "E_TUPLE_EGOHOS_STAGE2_MASK"
    assert metrics["coverage"] == 0.0
    assert metrics["invalid_retained_record_count"] == 1
    gate = MODULE._tuple_axis(
        json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text()),
        "hand_action_coupling",
    )["public_gate"]
    assert MODULE._tuple_egohos_metrics_pass(metrics, gate) is False


def test_tuple_hand_module_selects_only_frozen_grid_and_reports_one_axis(
    tmp_path: Path, monkeypatch
) -> None:
    import numpy as np
    from PIL import Image

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    fixture_root = tmp_path / "fixtures"
    mask_path = fixture_root / "masks/left.png"
    mask_path.parent.mkdir(parents=True)
    target = np.zeros((20, 20), dtype=np.uint8)
    target[5:10, 5:10] = 255
    Image.fromarray(target).save(mask_path)
    mask_hash = MODULE.file_digest(mask_path)
    commitment = "a" * 64
    rows = [
        {
            "fixture_ordinal": 0,
            "stratum": "contact",
            "contact": True,
            "target_hand_side": "left hand",
            "target_hand_mask_relative_path": "masks/left.png",
            "target_hand_mask_sha256": mask_hash,
            "target_hand_mask_width": 20,
            "target_hand_mask_height": 20,
            "target_hand_boundary_vertex_count": 0,
            "target_hand_outside_canvas_vertex_count": 0,
            "target_hand_outside_canvas_component_count": 0,
            "source_polygon_finite_nonnegative": True,
            "target_hand_mask_exact_frame_binary_nonempty": True,
            "geometry_valid": True,
            "media_relative_path": "media/contact.png",
            "media_sha256": "b" * 64,
        },
        {
            "fixture_ordinal": 1,
            "stratum": "explicit_no_contact",
            "contact": False,
            "target_hand_side": "left hand",
            "target_hand_mask_relative_path": "masks/left.png",
            "target_hand_mask_sha256": mask_hash,
            "target_hand_mask_width": 20,
            "target_hand_mask_height": 20,
            "target_hand_boundary_vertex_count": 0,
            "target_hand_outside_canvas_vertex_count": 0,
            "target_hand_outside_canvas_component_count": 0,
            "source_polygon_finite_nonnegative": True,
            "target_hand_mask_exact_frame_binary_nonempty": True,
            "geometry_valid": True,
            "media_relative_path": "media/no-contact.png",
            "media_sha256": "c" * 64,
        },
        {
            "fixture_ordinal": 2,
            "stratum": "verified_no_hand",
            "contact": None,
            "target_hand_side": None,
            "target_hand_mask_relative_path": None,
            "target_hand_mask_sha256": None,
            "target_hand_mask_bytes": None,
            "target_hand_mask_width": None,
            "target_hand_mask_height": None,
            "target_hand_boundary_vertex_count": 0,
            "target_hand_outside_canvas_vertex_count": 0,
            "target_hand_outside_canvas_component_count": 0,
            "target_hand_mask_exact_frame_binary_nonempty": False,
            "verified_no_hand_seal_commitment_sha256": commitment,
            "media_relative_path": "media/no-hand.png",
            "media_sha256": "d" * 64,
        },
    ]
    no_hand = _egohos_test_masks(False)
    no_hand["stage1"][:] = 0
    monkeypatch.setattr(
        MODULE,
        "_tuple_egohos_predict_rows",
        lambda *_args, **_kwargs: [
            _egohos_test_masks(True),
            _egohos_test_masks(False),
            no_hand,
        ],
    )
    monkeypatch.setattr(
        MODULE,
        "_read_tuple_egohos_target_mask",
        lambda *_args, **_kwargs: target.copy(),
    )
    result = MODULE._tuple_hand_contact_module(
        {
            "cfg": config,
            "rows": {"hand_contact": rows},
            "fixture_root": fixture_root,
            "public_root": tmp_path / "public",
            "scratch_root": tmp_path / "scratch",
            "device": "cpu",
            "partition": "development",
            "verified_no_hand_seal": {
                "status": "PASS",
                "verified_no_hand_seal_commitment_sha256": commitment,
            },
        }
    )
    assert result["status"] == "PASS"
    assert result["selected_thresholds"] == {
        "EgoHOS_min_mask_fraction": 0.02
    }
    assert set(result["axis_results"]) == {"hand_action_coupling"}
    assert result["metrics"]["contact_metric_item_count"] == 2


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


def test_referent_attribute_size_pair_uses_one_source_and_exact_render_sizes() -> None:
    import numpy as np
    from PIL import Image

    preparation = {
        "seed": 20260802,
        "referent_attribute_rendering": {
            "geometry": {
                "width": 384,
                "height": 288,
                "fps": 9,
                "duration_seconds": 7.0,
                "frames": 63,
            },
            "utterance_interval_seconds": [2.5, 4.5],
        },
    }
    definitions = {
        "visible_mask_fraction_min": 0.01,
        "dominant_target_mask_fraction_min": 0.05,
        "dominant_area_ratio_to_next_candidate_min": 2.0,
    }
    target = Image.new("RGBA", (40, 30), (190, 90, 40, 255))
    distractor = Image.new("RGBA", (40, 30), (40, 180, 90, 255))
    assert MODULE._referent_attribute_source_index(4, 4) == 0
    assert MODULE._referent_attribute_source_index(5, 4) == 0
    assert MODULE._referent_attribute_episode_id(
        "development", "cup", "persistent_ambiguous", 4, 0
    ) == MODULE._referent_attribute_episode_id(
        "development",
        "cup",
        "persistent_dominant_with_small_distractor",
        5,
        0,
    )
    big = MODULE._render_referent_fixture(
        preparation,
        "development",
        "cup",
        "persistent_ambiguous",
        4,
        target,
        distractor,
        definitions,
    )
    small = MODULE._render_referent_fixture(
        preparation,
        "development",
        "cup",
        "persistent_dominant_with_small_distractor",
        5,
        target,
        distractor,
        definitions,
    )

    def longest_side(mask_stack) -> int:
        values = []
        for frame in mask_stack:
            indices = np.argwhere(frame > 0)
            if indices.size:
                values.append(
                    max(
                        int(indices[:, 1].max() - indices[:, 1].min() + 1),
                        int(indices[:, 0].max() - indices[:, 0].min() + 1),
                    )
                )
        return max(values)

    big_frames, big_target, big_distractor, big_truth = big
    small_frames, small_target, small_distractor, small_truth = small
    assert longest_side(big_target) == 130
    assert longest_side(small_target) == 72
    assert big_truth["target_longest_side_pixels"] == 130
    assert small_truth["target_longest_side_pixels"] == 72
    assert big_truth["background_identity_role"] == "shared_relative_size_pair"
    assert small_truth["background_identity_role"] == "shared_relative_size_pair"
    assert np.array_equal(big_frames[0][0, 0], small_frames[0][0, 0])
    assert big_distractor.shape == big_target.shape == (63, 288, 384)
    assert small_distractor.shape == small_target.shape == (63, 288, 384)
    assert big_truth["candidate_count_by_phase"] == {
        "before": "2plus",
        "during": "2plus",
        "after": "2plus",
    }
    assert set(big_truth["dominance_by_phase"]) == {
        "before",
        "during",
        "after",
    }
    assert big_truth["sample_count_by_phase"] == {
        "before": 3,
        "during": 3,
        "after": 2,
    }
    assert len(big_truth["sampled_mask_truth"]) == 8
    assert all(
        set(row) >= {"phase", "sample_time", "frame_index"}
        for row in big_truth["sampled_mask_truth"]
    )


def test_referent_attribute_distractor_is_distinct_same_category_source() -> None:
    import pytest

    records = [
        {
            "category": "cup",
            "image_id": index + 10,
            "annotation_id": index + 20,
            "source_image_sha256": f"sha-{index}",
        }
        for index in range(4)
    ]
    target, distractor = MODULE._referent_attribute_source_records(
        records, 4, "persistent_ambiguous"
    )
    assert target is records[0]
    assert distractor is records[1]
    paired_target, paired_distractor = MODULE._referent_attribute_source_records(
        records, 5, "persistent_dominant_with_small_distractor"
    )
    assert paired_target is target
    assert paired_distractor is distractor
    unused_target, unused_distractor = MODULE._referent_attribute_source_records(
        records, 0, "persistent_clear"
    )
    assert unused_target is records[0]
    assert unused_distractor is None

    wrong_category = [dict(records[0]), {**records[1], "category": "bottle"}]
    with pytest.raises(RuntimeError, match="E_TUPLE_DISTRACTOR_CATEGORY_MISMATCH"):
        MODULE._referent_attribute_source_records(
            wrong_category, 4, "persistent_ambiguous"
        )
    repeated_source = [dict(records[0]), dict(records[1])]
    repeated_source[1]["image_id"] = repeated_source[0]["image_id"]
    with pytest.raises(RuntimeError, match="E_TUPLE_DISTRACTOR_SOURCE_IDENTITY"):
        MODULE._referent_attribute_source_records(
            repeated_source, 4, "persistent_ambiguous"
        )


def test_referent_attribute_non_distractor_scenario_does_not_require_source() -> None:
    import numpy as np
    from PIL import Image

    preparation = {
        "seed": 20260802,
        "referent_attribute_rendering": {
            "geometry": {
                "width": 384,
                "height": 288,
                "fps": 9,
                "duration_seconds": 7.0,
                "frames": 63,
            },
            "utterance_interval_seconds": [2.5, 4.5],
        },
    }
    definitions = {
        "visible_mask_fraction_min": 0.01,
        "dominant_target_mask_fraction_min": 0.05,
        "dominant_area_ratio_to_next_candidate_min": 2.0,
    }
    target = Image.new("RGBA", (40, 30), (190, 90, 40, 255))
    _, target_masks, distractor_masks, truth = MODULE._render_referent_fixture(
        preparation,
        "development",
        "cup",
        "persistent_clear",
        0,
        target,
        None,
        definitions,
    )
    assert np.any(target_masks)
    assert not np.any(distractor_masks)
    assert truth["candidate_count_by_phase"] == {
        "before": "1",
        "during": "1",
        "after": "1",
    }


def test_attribute_deterministic_measurement_rejects_reference_mask() -> None:
    import numpy as np
    import pytest

    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[2:8, 5:15] = [210, 40, 30]
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 5:15] = 1
    with pytest.raises(
        RuntimeError, match="E_TUPLE_ATTRIBUTE_REFERENCE_MASK_PROHIBITED"
    ):
        MODULE._predicted_mask_attribute_measurements(
            image, mask, mask_role="reference_mask"
        )
    measured = MODULE._predicted_mask_attribute_measurements(
        image, mask, mask_role="predicted_SAM_mask"
    )
    assert measured == {
        "mask_fraction": 0.3,
        "median_rgb": [210.0, 40.0, 30.0],
        "bbox_width_fraction": 0.5,
        "bbox_height_fraction": 0.6,
        "bbox_longest_side_fraction": 0.6,
    }


def test_recurrence_render_preserves_full_canvas_binary_segmentation_masks() -> None:
    from PIL import Image, ImageDraw

    first_crop = Image.new("RGBA", (40, 30), (220, 30, 30, 0))
    first_draw = ImageDraw.Draw(first_crop)
    first_draw.rectangle((6, 5, 32, 25), fill=(220, 30, 30, 255))
    second_crop = Image.new("RGBA", (30, 40), (30, 40, 220, 0))
    second_draw = ImageDraw.Draw(second_crop)
    second_draw.ellipse((4, 7, 25, 34), fill=(30, 40, 220, 255))

    first, second, first_mask, second_mask = MODULE._render_recurrence_pair(
        first_crop,
        second_crop,
        "different_category",
        0,
    )
    assert first.size == second.size == (224, 224)
    assert first_mask.size == second_mask.size == (224, 224)
    assert first_mask.mode == second_mask.mode == "L"
    assert {
        value for value, count in enumerate(first_mask.histogram()) if count
    } == {0, 255}
    assert {
        value for value, count in enumerate(second_mask.histogram()) if count
    } == {0, 255}
    assert first_mask.getpixel((0, 0)) == 0
    assert first_mask.getpixel((112, 112)) == 255
    assert second_mask.getpixel((0, 0)) == 0
    assert second_mask.getpixel((112, 112)) == 255


def test_recurrence_fixture_mask_roundtrip_and_schema_fail_closed(
    tmp_path: Path,
) -> None:
    import pytest
    from PIL import Image, ImageDraw

    crop = Image.new("RGBA", (40, 30), (180, 80, 30, 0))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((5, 4, 34, 26), fill=(180, 80, 30, 255))
    first, second, first_mask, second_mask = MODULE._render_recurrence_pair(
        crop,
        crop,
        "same_instance_near_duplicate",
        0,
    )
    paths = {
        "first": tmp_path / "first.png",
        "second": tmp_path / "second.png",
        "first_mask": tmp_path / "first-mask.png",
        "second_mask": tmp_path / "second-mask.png",
    }
    first.save(paths["first"], format="PNG")
    second.save(paths["second"], format="PNG")
    first_mask.save(paths["first_mask"], format="PNG")
    second_mask.save(paths["second_mask"], format="PNG")
    row = {
        "fixture_ordinal": 0,
        "stratum": "same_instance_near_duplicate",
        "same_referent": True,
        "near_duplicate": True,
        "source_image_ids": [10, 10],
    }
    for prefix, path in paths.items():
        row[f"{prefix}_relative_path"] = path.name
        row[f"{prefix}_sha256"] = MODULE.file_digest(path)
        row[f"{prefix}_bytes"] = path.stat().st_size
    MODULE._validate_tuple_recurrence_fixture_rows([row], tmp_path)

    missing_bytes = dict(row)
    missing_bytes.pop("first_mask_bytes")
    with pytest.raises(RuntimeError, match="E_TUPLE_RECURRENCE_FIXTURE_SCHEMA"):
        MODULE._validate_tuple_recurrence_fixture_rows(
            [missing_bytes], tmp_path
        )

    Image.new("L", (224, 224), 128).save(paths["first_mask"], format="PNG")
    nonbinary = dict(row)
    nonbinary["first_mask_sha256"] = MODULE.file_digest(paths["first_mask"])
    nonbinary["first_mask_bytes"] = paths["first_mask"].stat().st_size
    with pytest.raises(RuntimeError, match="E_TUPLE_RECURRENCE_MASK"):
        MODULE._validate_tuple_recurrence_fixture_rows([nonbinary], tmp_path)


def test_language_fixture_freezes_staged_statuses_and_attribute_spans() -> None:
    from collections import Counter

    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    preparation = MODULE._tuple_fixture_preparation_amendment(config)
    for partition in ("development", "holdout"):
        rows = MODULE._language_lexical_fixture_rows(preparation, partition)
        assert len(rows) == 48
        assert Counter(row["expected_adapter_status"] for row in rows) == {
            "ACCEPT": 36,
            "ABSTAIN": 12,
        }
        assert Counter(row["expected_tuple_status"] for row in rows) == {
            "ACCEPT": 34,
            "ABSTAIN": 14,
        }
        assert Counter(row["expected_grounding_status"] for row in rows) == {
            "ACCEPT": 32,
            "ABSTAIN": 16,
        }
        assert all(
            "expected_pipeline_status" not in row and "expected_reason" not in row
            for row in rows
        )
        cloud = [
            row
            for row in rows
            if row["expected_grounding_reason"] == "ONTOLOGY_UNMATCHED"
        ]
        assert len(cloud) == 2
        assert all(row["expected_tuple_status"] == "ACCEPT" for row in cloud)
        assert all(
            row["expected_lexical_mentions"]
            == [
                {
                    "token": "cloud",
                    "part_of_speech": "noun",
                    "expected_lemma": "cloud",
                    "expected_frequency_band": "high",
                }
            ]
            for row in cloud
        )
        assert all(
            mention["expected_lemma"] == mention["token"]
            and mention["expected_frequency_band"] == "high"
            for row in rows
            for mention in row["expected_lexical_mentions"]
        )
        spans = [
            (row["case_id"], span["adjective"], span["noun"])
            for row in rows
            for span in row["expected_adjective_noun_spans"]
        ]
        assert len(spans) == 8
        assert {attribute for _case, attribute, _noun in spans} == {"red"}


def test_attribute_span_extractor_requires_actual_adjacent_order() -> None:
    mentions = [
        {"token_index": 1, "lemma": "red", "part_of_speech": "adjective"},
        {"token_index": 2, "lemma": "cup", "part_of_speech": "noun"},
        {"token_index": 4, "lemma": "bowl", "part_of_speech": "noun"},
        {"token_index": 5, "lemma": "blue", "part_of_speech": "adjective"},
    ]
    assert MODULE._adjacent_adjective_noun_spans(mentions) == [
        {"adjective": "red", "noun": "cup"}
    ]
    assert MODULE._span_f1(
        [("case", "red", "cup")], [("case", "red", "cup")]
    ) == 1.0
    assert MODULE._span_f1(
        [("case", "red", "cup")], [("case", "blue", "cup")]
    ) == 0.0


def test_lexical_truth_rejects_a_wrong_frozen_frequency_band() -> None:
    expected = [
        {
            "token": "red",
            "part_of_speech": "adjective",
            "expected_lemma": "red",
            "expected_frequency_band": "high",
        }
    ]
    observed = [
        {
            "token": "red",
            "part_of_speech": "adjective",
            "lemma": "red",
            "frequency_band": "high",
        }
    ]
    assert MODULE._tuple_lexical_truth_checks(expected, observed) == [True]
    expected[0]["expected_frequency_band"] = "middle"
    checks = MODULE._tuple_lexical_truth_checks(expected, observed)
    assert checks == [False]
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    gate = MODULE._tuple_axis(config, "noun_adjective_exposure")["public_gate"]
    assert sum(checks) / len(checks) < gate[
        "lemma_and_frequency_band_exact_fraction_min"
    ]


def test_attribute_opportunity_uses_strict_per_referent_adapter_event() -> None:
    import pytest

    row = {"category": "cup", "truth": {"attribute": "red"}}
    accepted = {
        "status": "ACCEPT",
        "abstention_reason": None,
        "mentions": [
            {"token_index": 2, "lemma": "red", "part_of_speech": "adjective"},
            {"token_index": 3, "lemma": "cup", "part_of_speech": "noun"},
        ],
    }
    assert MODULE._tuple_attribute_language_accepts(row, accepted) is True
    accepted["mentions"][0]["token_index"] = 4
    assert MODULE._tuple_attribute_language_accepts(row, accepted) is False
    assert MODULE._tuple_attribute_language_accepts(
        row,
        {"status": "ABSTAIN", "abstention_reason": "EMPTY_ASR", "mentions": []},
    ) is False
    with pytest.raises(RuntimeError, match="E_TUPLE_ATTRIBUTE_ADAPTER_EVENT_SCHEMA"):
        MODULE._tuple_attribute_language_accepts(row, None)


def test_attribute_prompts_use_only_nested_committed_public_values() -> None:
    config = json.loads(Path("configs/synthetic_video_real_only_proof.json").read_text())
    assert MODULE._tuple_attribute_prompt_groups(config) == {
        "color": {
            value: [
                f"a {value} object",
                f"an object that is {value}",
                f"the visible object is {value}",
            ]
            for value in ("red", "blue", "green", "yellow")
        },
        "relative_size": {
            value: [
                f"a {value} object",
                f"an object that is {value}",
                f"the visible object is {value}",
            ]
            for value in ("big", "small")
        },
    }


def test_attribute_palette_and_paired_predicted_mask_geometry() -> None:
    assert MODULE._tuple_attribute_palette_label([210.0, 45.0, 35.0]) == "red"
    assert MODULE._tuple_attribute_palette_label([40.0, 85.0, 210.0]) == "blue"
    common = {
        "attribute_family": "relative_size",
        "attribute_pair_id": "pair",
        "episode_id": "episode",
        "source_image_id": 10,
        "source_annotation_id": 20,
        "background_identity_role": "shared_relative_size_pair",
    }
    rows = [
        {
            **common,
            "attribute": "big",
            "target_longest_side_pixels": 130,
            "predicted_mask_bbox_longest_side_median": 0.52,
            "deterministic_label": None,
        },
        {
            **common,
            "attribute": "small",
            "target_longest_side_pixels": 72,
            "predicted_mask_bbox_longest_side_median": 0.30,
            "deterministic_label": None,
        },
    ]
    MODULE._tuple_attribute_apply_size_pairs(rows)
    assert [row["deterministic_label"] for row in rows] == ["big", "small"]
    rows[0]["predicted_mask_bbox_longest_side_median"] = 0.44
    MODULE._tuple_attribute_apply_size_pairs(rows)
    assert [row["deterministic_label"] for row in rows] == [None, None]


def test_attribute_metrics_abstain_on_disagreement_and_invalid_mask() -> None:
    values = ("red", "blue", "green", "yellow", "big", "small")
    rows = [
        {
            "attribute": value,
            "contrast_expected": True,
            "mask_measurement_expected": True,
            "language_span_accepted": True,
            "mask_measurement_valid": True,
            "pe_margin": 0.2,
            "pe_label": value,
            "deterministic_label": value,
        }
        for value in values
    ]
    rows.extend(
        {
            "attribute": "red",
            "contrast_expected": False,
            "mask_measurement_expected": mask,
            "language_span_accepted": language,
            "mask_measurement_valid": mask,
            "pe_margin": 0.2 if mask else None,
            "pe_label": "red" if mask else None,
            "deterministic_label": "red" if mask else None,
        }
        for language, mask in ((False, True), (True, False))
    )
    assert MODULE._tuple_attribute_metrics(rows, 0.05, 1.0) == {
        "adjective_noun_span_f1": 1.0,
        "eligible_visual_attribute_coverage": 1.0,
        "visible_contrast_macro_f1": 1.0,
        "null_contrast_specificity": 1.0,
        "valid_mask_measurement_fraction": 1.0,
    }
    rows[0]["deterministic_label"] = "blue"
    rows[1]["mask_measurement_valid"] = False
    metrics = MODULE._tuple_attribute_metrics(rows, 0.05, 1.0)
    assert metrics["eligible_visual_attribute_coverage"] == 4 / 6
    assert metrics["visible_contrast_macro_f1"] < 1.0
    assert metrics["valid_mask_measurement_fraction"] == 6 / 7
    assert metrics["null_contrast_specificity"] == 1.0


def test_attribute_truth_fails_closed_on_legacy_scalar_schema() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="E_TUPLE_ATTRIBUTE_FIXTURE_SCHEMA"):
        MODULE._tuple_attribute_truth(
            {
                "fixture_ordinal": 1,
                "category": "cup",
                "scenario": "persistent_clear",
                "truth": {"attribute": "red", "visible": True},
            }
        )


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
    assert "--no-hand-review-root" in prepare
    assert (
        '--audio-seed-root "${PHASE4_PUBLIC_ROOT}/public/'
        'mechanistic-training-tuple-fixtures/audio-seed"'
    ) in prepare
    assert (
        '--audio-seed-root "${PHASE4_PUBLIC_ROOT}/public/'
        'mechanistic-tuple-audio-seed"'
    ) not in prepare
    assert "tuple-fixtures-feasibility" in prepare
    assert "mechanistic-tuple-fixture-feasibility" in prepare
    assert "--recipe active-visor-hos" in prepare
    assert "tuple-no-hand-review-prepare" in prepare
    assert "mechanistic-tuple-no-hand-review-prepare" in prepare
    assert "tuple-no-hand-review-seal" in prepare
    assert "mechanistic-tuple-no-hand-review-seal" in prepare
    assert "PHASE4_AUTHORIZED_APPLICANT_ATTESTED" in prepare
    assert "PHASE4_BLIND_TO_EGOHOS_OUTPUT_ATTESTED" in prepare
    assert "PHASE4_EGOHOS_INFERENCE_NOT_STARTED_ATTESTED" in prepare
    assert "PHASE4_ACTIVITY_CODE_CLEAN=1" in prepare
    assert "PHASE4_RESTRICTED_ROOT" not in prepare
    assert "#SBATCH --partition=a30" in qualify
    assert "#SBATCH --time=03:00:00" in qualify
    assert "#SBATCH --gpus-per-node=1" in qualify
    assert "qualify-public" in qualify
    assert "activity-candidate" in qualify
    assert "--net --network none" in qualify
    assert "HF_HUB_OFFLINE=1" in qualify
    assert "calibration-repair-pydeps" in qualify
    assert "PHASE4_RESTRICTED_ROOT" not in qualify
    assert "PHASE4_TUPLE_RUN_MODE" in qualify
    assert "tuple-size" in qualify
    assert "tuple-qualify" in qualify
    assert '--partition "$tuple_run_mode"' in qualify
    assert "development" in qualify and "holdout" in qualify
    assert "97ef52ecaa8c99db017e598d8a63d0d2170affef14ef46e7df7a656abd3a1a07" in qualify
    assert "language_archive_bytes=542423040" in qualify
    assert "sha256sum --check --status" in qualify


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
    assert "tuple-qualify" in source
    assert "TUPLE_QUALIFICATION_FIELDS" in source
    assert "E_ACTIVITY_HOLDOUT_BEFORE_WINNER_SEAL" in source
    assert 'print(json.dumps' not in source
