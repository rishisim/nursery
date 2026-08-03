import json
import ast
from pathlib import Path


CONFIG = json.loads(Path("configs/embodied_simulation_unity_mujoco_gate.json").read_text())
SOURCE = Path("babyworld_lite/childlens_engine_bakeoff/unity_mujoco_gate.py")


def test_gate_clock_and_schedule_are_exact():
    clock = CONFIG["clock"]
    assert clock["physics_hz"] == 240
    assert clock["render_hz"] == 30
    assert clock["steps_per_frame"] == 8
    assert clock["physics_steps"] == clock["physics_hz"] * clock["duration_s"]
    assert clock["render_frames"] == clock["render_hz"] * clock["duration_s"]
    phases = CONFIG["phases"]
    assert phases[0]["start_s"] == 0
    assert phases[-1]["end_s"] == clock["duration_s"]
    assert all(a["end_s"] == b["start_s"] for a, b in zip(phases, phases[1:]))


def test_gate_has_single_authority_and_no_assistance():
    authority = CONFIG["authority"]
    assert authority["physics_and_side_streams"] == "MuJoCo"
    assert authority["appearance"] == "Unity only"
    assert authority["immutable_trace_replay"] is True
    assert authority["unity_physics_enabled"] is False
    assert authority["object_attachment_or_direct_pose_after_initialization"] is False
    assert authority["external_force_or_hidden_spring_assist"] is False
    assert authority["assistance_ledger_expected_entries"] == 0


def test_gate_freezes_anatomical_grasp_camera_and_robustness():
    tolerances = CONFIG["frozen_tolerances"]
    assert CONFIG["frozen_before_manipulation_outcomes"] is True
    assert set(tolerances["required_digits"]) == {"thumb", "index", "middle"}
    assert tolerances["minimum_distinct_support_digits"] >= 3
    assert tolerances["minimum_lift_m"] == 0.08
    assert tolerances["collider_skin_landmark_max_m"] == 0.01
    assert CONFIG["camera"]["neutral_optical_axis_max_deg"] <= 15
    assert len(CONFIG["robustness_cells"]) == 3
    assert tolerances["robustness_min_pass_cells"] == 2


def test_gate_privacy_and_visual_lineage_are_explicit():
    assert CONFIG["scope"]["restricted_childlens_access_permitted"] is False
    assert CONFIG["scope"]["public_or_synthetic_only"] is True
    assert CONFIG["lineage"]["required_git_ancestor"] == "537d30c"
    assert len(CONFIG["lineage"]["visual_audition_fbx_sha256"]) == 64


def test_tactile_controller_requires_dwell_and_actuator_only_recovery():
    controller = CONFIG["controller"]
    assert controller["lift_requires_qualified_thumb_index_middle_graph"] is True
    assert controller["contact_graph_dwell_s"] > 0
    assert controller["force_target_n_per_digit"] > 0
    assert controller["shoulder_recenter_limit_deg"] <= 3
    assert controller["retry_open_deg_s"] > 0


def test_manifest_gate_is_the_only_runnable_protocol():
    source = SOURCE.read_text()
    functions = {
        node.name for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "manifest_registration" in functions
    assert not {"model_xml", "controls", "execute", "run_gate"} & functions
    assert 'parser.add_argument("--rest-manifest", type=Path, required=True)' in source
    assert "add_mutually_exclusive_group(required=True)" in source
