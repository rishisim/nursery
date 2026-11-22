import json
from pathlib import Path

from babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate.contracts import (
    CONFIG_PATH,
    compile_contract_matrix,
    load_frozen_config,
    validate_frozen_config,
)


def test_frozen_contract_matrix_is_three_required_cells_with_three_garments():
    config = load_frozen_config()
    validate_frozen_config(config)
    matrix = compile_contract_matrix(config)
    assert len(matrix) == 3
    assert len({row["scene_spec"]["room_family"] for row in matrix}) == 3
    assert len({row["avatar_spec"]["garment_configuration_id"] for row in matrix}) == 3
    assert {row["episode_spec"]["cell_id"] for row in matrix} == {"A", "B", "C"}
    assert all(row["activity_plan"]["no_seed_specific_retuning"] for row in matrix)
    compiler = config["scene_compiler"]
    assert compiler["target_midpoint_reach_band_m"] == [0.33, 0.38]
    assert compiler["target_lateral_bias_toward_right_shoulder_m"] == 0.02
    assert compiler["same_band_all_room_seeds"]
    assert not compiler["seed_specific_retuning"]
    assert {
        row["scene_spec"]["reachability"]["compiled_requested_midpoint_m"]
        for row in matrix
    } == {0.34, 0.355, 0.37}
    assert all(
        row["scene_spec"]["reachability"]["compiled_midpoint_band_m"] == [0.33, 0.38]
        and not row["scene_spec"]["reachability"]["seed_specific_retuning"]
        for row in matrix
    )


def test_one_exact_clock_and_registered_capture_contract():
    config = json.loads(CONFIG_PATH.read_text())
    assert config["authority"]["physics_hz"] == 240
    assert config["authority"]["render_hz"] == 30
    assert config["authority"]["steps_per_render_frame"] == 8
    capture = config["qa_tolerances"]["capture"]
    assert capture["resolution_px"] == [1920, 1080]
    assert capture["streams"] == ["rgb", "metric_depth", "semantic", "persistent_instance"]
    assert capture["same_frozen_frame_required"]


def test_contact_qualification_shell_and_force_semantics_are_distinct():
    config = load_frozen_config()
    contact = config["qa_tolerances"]["contact"]
    interaction = config["qa_tolerances"]["interaction"]
    assert contact["qualification_max_measured_separation_m"] == 0.0005
    assert contact["eligible_positive_separation_is_proximity_not_force"]
    assert contact["rows_over_qualification_shell_preserved_as_raw_truth_only"]
    assert contact["simultaneous_nonzero_physx_impulse_required"]
    assert interaction["right_opposition_min_s"] == 0.30
    assert interaction["left_support_min_s"] == 0.25
    assert interaction["lift_min_m"] == 0.10
    assert interaction["turn_min_deg"] == 30.0


def test_trace_closes_corrected_bimanual_truth_gaps():
    config = load_frozen_config()
    trace = config["trace_contract"]
    assert trace["body_state"] == [
        "root", "pelvis", "torso", "neck", "head",
        "left_shoulder", "left_upper_arm", "left_elbow", "left_lower_arm", "left_forearm", "left_wrist", "left_palm",
        "right_shoulder", "right_upper_arm", "right_elbow", "right_lower_arm", "right_forearm", "right_wrist", "right_palm",
    ]
    assert "rotation_world_xyzw" in trace["pose_fields"]
    assert "segments" in trace["per_digit_fields"]
    assert "angular_velocity_world_rad_s" in trace["object_fields"]
    assert "clearance_m" in trace["camera_fields"]
    assert trace["assistance_ledger"] == []
    assert trace["recovery_ledger"] == []
    assert "target_force_counter" in trace["authority_audit"]


def test_preserved_visual_pass_and_integrated_no_go_are_distinct():
    config = load_frozen_config()
    evidence = config["preserved_evidence"]
    assert evidence["visual_feasibility"]["decision"] == "PASS_VISUAL_SHELL_ONLY"
    assert evidence["corrected_prior_gate"]["decision"] == "NO-GO"
    assert Path(evidence["visual_feasibility"]["record"]).is_file()
    assert Path(evidence["corrected_prior_gate"]["record"]).is_file()


def test_no_assistance_and_claim_boundaries_are_frozen():
    config = load_frozen_config()
    assert "FixedJoint grasp" in config["forbidden_assistance"]
    assert "free-object transform writes after initialization" in config["forbidden_assistance"]
    claims = config["privacy_and_claims"]
    assert not claims["restricted_childlens_access_permitted"]
    assert not claims["restricted_external_drive_access_permitted"]
    assert not claims["infant_trained"]
    assert not claims["biological_torque_valid"]


def test_scene_catalog_has_licenses_dimensions_collision_and_physics():
    config = load_frozen_config()
    catalog = config["assets"]["furniture_catalog"]
    assert catalog["license"] == "CC0"
    assert all(row["dimensions_m"] and row["semantic_class"] and row["collision_source"] for row in catalog["members"])
    for contract in compile_contract_matrix(config):
        target = contract["scene_spec"]["target"]
        assert target["persistent_id"] in {"red_toy_001", "blue_cup_001", "yellow_block_001"}
        assert target["collision_policy"] == "free_non_kinematic_physx_rigidbody"
        assert target["post_initialization_transform_writes"] == 0
        assert len(target["geometry_spec_sha256"]) == 64
        for instance in contract["scene_spec"]["instances"]:
            assert "static_friction" in instance
            assert "dynamic_friction" in instance
            catalog_row = next(
                row for row in catalog["members"] if row["id"] == instance["asset_id"]
            )
            assert len(catalog_row["sha256"]) == 64


def test_integrated_anti_clipping_and_rich_scene_contract_is_frozen():
    config = load_frozen_config()
    registration = config["qa_tolerances"]["registration"]
    assert registration["skin_collider_max_m"] == 0.005
    assert registration["garment_body_max_penetration_m"] == 0.002
    assert registration["garment_affected_vertex_fraction_max"] == 0.001
    assert registration["finger_object_max_penetration_m"] == 0.002
    assert registration["support_max_penetration_m"] == 0.0015
    assert config["scene_compiler"]["minimum_contextual_objects"] == 11
    compiled = compile_contract_matrix(config)
    assert all(len(row["scene_spec"]["instances"]) > 10 for row in compiled)
    assert all(row["scene_spec"]["minimum_contextual_objects"] == 11 for row in compiled)
    assert all({"persistent_id", "asset_id", "asset_dimensions_m", "semantic_class",
                "collision_source", "static_friction", "dynamic_friction"}
               <= set(instance)
               for row in compiled for instance in row["scene_spec"]["instances"])


def test_unity_interface_freezes_dynamic_finger_post_step_synchronization():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/GateContracts.cs"
    ).read_text()
    assert 'DurationSeconds = 24f' in source
    assert 'RightForceOppositionSeconds = 0.30f' in source
    assert 'LeftForceSupportSeconds = 0.25f' in source
    assert 'Dictionary<string, Rigidbody> FingerBodies' in source
    assert 'Dictionary<string, ConfigurableJoint> FingerJoints' in source
    assert 'SynchronizeCompletedPhysicsState' in source
    assert 'AuthorityAuditState' in source
