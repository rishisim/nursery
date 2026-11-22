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


def test_historical_camera_failure_is_preserved_and_reconstruction_is_neutral():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "WeightedBimanualBaselineBuilder.cs"
    ).read_text()
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    historical_stage_c = gate["stage_decisions"]["c"]
    reconstruction = gate["stage_c_reconstruction"]

    assert historical_stage_c["decision"] == "NO-GO"
    assert not historical_stage_c["frozen_gate_validation"]
    assert historical_stage_c["camera"]["fixed_optical_pitch_deg"] == 50
    assert not historical_stage_c["camera"]["near_neutral_mount_pass"]
    assert not gate["pass_predicate_policy"][
        "fixed_50deg_downward_optical_pitch_can_satisfy_near_neutral_mount"
    ]
    assert "Mathf.Sin(50*Mathf.Deg2Rad)" not in source
    assert "bounds.max.z + .032f" in source
    assert "Quaternion.LookRotation(Vector3.forward, Vector3.up)" in source
    assert "ConfigureCamera(headCamera, 68" in source
    assert reconstruction["decision"] == "PASS"
    assert reconstruction["camera_contract"]["neutral_mount_angle_deg_max"] == 15.0
    assert reconstruction["result"]["render_registration"][
        "camera_minimum_skin_clearance_m"
    ] >= reconstruction["camera_contract"]["minimum_skin_clearance_m"]


def test_historical_viewport_proxy_is_preserved_but_reconstruction_measures_contact():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "WeightedBimanualBaselineBuilder.cs"
    ).read_text()
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    historical_registration = gate["stage_decisions"]["c"]["registration"]
    reconstruction = gate["stage_c_reconstruction"]

    assert historical_registration[
        "contact_visibility_check"
    ] == "object-center viewport proxy only"
    assert historical_registration[
        "measured_contact_to_visible_surface_status"
    ] == "NOT_MEASURED"
    assert historical_registration[
        "visible_vs_physical_touch_timing_status"
    ] == "NOT_MEASURED"
    assert not gate["pass_predicate_policy"][
        "object_center_viewport_visibility_can_satisfy_contact_registration"
    ]
    assert "WorldToViewportPoint(contact.point_m)" in source
    assert "digitVertices.TryGetValue" in source
    assert "first_touch_frame_difference" in source
    result = reconstruction["result"]["render_registration"]
    contract = reconstruction["registration_contract"]
    assert result["contact_digit_surface_p95_m"] <= contract[
        "contact_to_correct_visible_digit_p95_m"
    ]
    assert result["first_touch_frame_difference"] <= contract[
        "visible_vs_physical_first_touch_frames_max"
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

    reconstruction = gate["stage_c_reconstruction"]
    assert reconstruction["decision"] == "PASS"
    assert reconstruction["scope_limits"][0].startswith(
        "This PASS answers only the bounded registration question"
    )
    assert reconstruction["result"]["render_registration"][
        "proxy_pixels_in_head_or_clean"
    ] == 0


def test_stage_c_reconstruction_requires_continuous_opposition_and_free_object():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "HybridBimanualCellBuilder.cs"
    ).read_text()
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    result = gate["stage_c_reconstruction"]["result"]["physics"]

    assert "qualifiedOppositionFrames==requiredOppositionFrames" in source
    assert "pivot+orbit*(rLift-pivot)" in source
    assert result["qualified_opposition_render_frames"] == result[
        "required_opposition_render_frames"
    ]
    assert result["lift_m"] >= 0.08
    assert result["turn_deg"] >= 20
    assert result["maximum_finger_penetration_m"] <= 0.003
    assert result["maximum_palm_angular_speed_deg_s"] <= 45
    assert result["final_quarter_second_contact_unique_steps"] == 0
    assert result["object_pose_writes_after_initialization"] == 0
    assert result["object_external_forces"] == 0
    assert result["attachment_or_joint_count"] == 0
    assert result["assistance_ledger_entries"] == 0


def test_stage_c_reconstruction_has_complete_trace_and_separate_qa_overlay():
    physics = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "HybridBimanualCellBuilder.cs"
    ).read_text()
    renderer = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "WeightedBimanualBaselineBuilder.cs"
    ).read_text()

    assert 'schema="embodied.hybrid_bimanual_trace.v2"' in physics
    for field in (
        "right_palm_rotation",
        "right_digit_closures",
        "right_digit_segments",
        "object_angular_velocity_rad_s",
        "torso_local_delta",
        "neck_local_delta",
        "head_local_delta",
    ):
        assert field in physics
    assert "headCamera.cullingMask &= ~(1 << QaLayer)" in renderer
    assert "cleanCamera.cullingMask &= ~(1 << QaLayer)" in renderer
    assert "COLLIDERS, NOT A SECOND BODY" in renderer
