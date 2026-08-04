import json
from pathlib import Path

from babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate.contracts import (
    CONFIG_PATH,
    compile_contract_matrix,
    load_frozen_config,
    validate_frozen_config,
)


def test_frozen_contract_matrix_is_three_rooms_by_three_garments():
    config = load_frozen_config()
    validate_frozen_config(config)
    matrix = compile_contract_matrix(config)
    assert len(matrix) == 9
    assert len({row["scene_spec"]["room_family"] for row in matrix}) == 3
    assert len({row["avatar_spec"]["garment_configuration_id"] for row in matrix}) == 3
    assert all(row["activity_plan"]["no_seed_specific_retuning"] for row in matrix)


def test_one_exact_clock_and_registered_capture_contract():
    config = json.loads(CONFIG_PATH.read_text())
    assert config["authority"]["physics_hz"] == 240
    assert config["authority"]["render_hz"] == 30
    assert config["authority"]["steps_per_render_frame"] == 8
    capture = config["qa_tolerances"]["capture"]
    assert capture["resolution_px"] == [1920, 1080]
    assert capture["streams"] == ["rgb", "metric_depth", "semantic", "persistent_instance"]
    assert capture["same_frozen_frame_required"]


def test_trace_closes_corrected_bimanual_truth_gaps():
    config = load_frozen_config()
    trace = config["trace_contract"]
    assert trace["body_state"] == ["root", "torso", "neck", "head"]
    assert "rotation_world_xyzw" in trace["pose_fields"]
    assert "segments" in trace["per_digit_fields"]
    assert "angular_velocity_world_rad_s" in trace["object_fields"]
    assert "clearance_m" in trace["camera_fields"]
    assert trace["assistance_ledger"] == []


def test_preserved_visual_pass_and_integrated_no_go_are_distinct():
    config = load_frozen_config()
    evidence = config["preserved_evidence"]
    assert evidence["visual_feasibility"]["decision"] == "PASS_VISUAL_SHELL_ONLY"
    assert evidence["controller_registration"]["decision"] == "NO-GO_STAGE_C_AND_INTEGRATED"
    assert Path(evidence["visual_feasibility"]["record"]).is_file()
    assert Path(evidence["controller_registration"]["record"]).is_file()


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
        assert target["persistent_id"] == "target_001"
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
