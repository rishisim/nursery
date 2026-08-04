import json
from pathlib import Path


def test_episode_and_clock_are_single_continuous_authority():
    episode = json.loads(Path("configs/embodied_simulation_episode.json").read_text())
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    assert episode["activity"]["duration_s"] == 56.0
    assert episode["activity"]["continuous_trace_required"]
    assert not episode["activity"]["hidden_world_resets_permitted"]
    assert gate["authority"]["physics_hz"] == 240
    assert gate["authority"]["render_hz"] == 30
    assert gate["authority"]["steps_per_frame"] == 8
    assert gate["assistance_ledger"] == []


def test_hybrid_source_contains_no_object_assistance_api():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "HybridBimanualCellBuilder.cs"
    ).read_text()
    assert "AddForce" not in source
    assert "AddTorque" not in source
    assert "FixedJoint" not in source
    assert "SetPositionAndRotation" not in source
    assert "target.position =" not in source
    assert "blueCup.position =" not in source
    assert "assistance_ledger_entries=0" in source


def test_unavailable_truth_cannot_satisfy_an_integrated_pass():
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    provenance = gate["field_provenance"]
    assert provenance["metric_depth"].startswith("UNAVAILABLE")
    assert provenance["semantic_and_instance"].startswith("UNAVAILABLE")
    assert provenance["head_imu"].startswith("UNAVAILABLE")
    assert gate["scientific_decision"] == "NO-GO"
    decision = Path("docs/embodied_simulation_bimanual_gate_decision.md").read_text()
    assert "Decision: **NO-GO" in decision


def test_fixed_downward_camera_mount_is_disqualifying_not_near_neutral():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "WeightedBimanualBaselineBuilder.cs"
    ).read_text()
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    stage_c = gate["stage_decisions"]["c"]

    assert "Mathf.Sin(50*Mathf.Deg2Rad)" in source
    assert stage_c["decision"] == "NO-GO"
    assert not stage_c["frozen_gate_validation"]
    assert stage_c["camera"]["fixed_optical_pitch_deg"] == 50
    assert not stage_c["camera"]["near_neutral_mount_pass"]
    assert not gate["pass_predicate_policy"][
        "fixed_50deg_downward_optical_pitch_can_satisfy_near_neutral_mount"
    ]


def test_viewport_object_center_cannot_claim_contact_registration():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "WeightedBimanualBaselineBuilder.cs"
    ).read_text()
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    registration = gate["stage_decisions"]["c"]["registration"]

    assert "WorldToViewportPoint(row.object_position_m+Offset)" in source
    assert "object_viewport=vp" in source
    assert registration["contact_visibility_check"] == "object-center viewport proxy only"
    assert registration["measured_contact_to_visible_surface_status"] == "NOT_MEASURED"
    assert registration["visible_vs_physical_touch_timing_status"] == "NOT_MEASURED"
    assert not gate["pass_predicate_policy"][
        "object_center_viewport_visibility_can_satisfy_contact_registration"
    ]


def test_partial_render_cannot_be_promoted_to_stage_d_pass():
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    stage_d = gate["stage_decisions"]["d"]

    assert stage_d["decision"] == "PARTIAL_PHYSICAL_RENDER_DEMONSTRATION_NOT_PASS"
    assert not stage_d["frozen_gate_validation"]
    assert stage_d["render_resolution"] == [960, 540]
    assert stage_d["required_render_resolution"] == [1920, 1080]
    assert stage_d["render_resolution"] != stage_d["required_render_resolution"]


def test_missing_registration_streams_remain_unavailable():
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    provenance = gate["field_provenance"]

    assert provenance["palm_rotation"].startswith("UNAVAILABLE")
    assert provenance["per_digit_pose_velocity"].startswith("UNAVAILABLE")
    assert provenance["object_angular_velocity"].startswith("UNAVAILABLE")
    assert provenance["camera_extrinsics"].startswith("UNAVAILABLE")
    assert provenance["contact_to_visible_skin_registration"].startswith("NOT_MEASURED")
