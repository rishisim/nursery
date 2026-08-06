import json
import hashlib
import math
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate.independent_qa import (
    FAIL,
    PASS,
    UNAVAILABLE,
    _capture_checks,
    _matrix_checks,
    _robustness_checks,
    audit_run,
)
from babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate.contracts import load_frozen_config


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def _write_rgb_png(path: Path, width: int, height: int, uint24: int):
    rgb = bytes((uint24 & 255, (uint24 >> 8) & 255, (uint24 >> 16) & 255))
    raw = b"".join(b"\x00" + rgb * width for _ in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _pose(position=(0.0, 1.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)):
    return {
        "position_world_m": list(position),
        "rotation_world_xyzw": list(rotation),
        "linear_velocity_world_m_s": [0.0, 0.0, 0.0],
        "angular_velocity_world_rad_s": [0.0, 0.0, 0.0],
    }


def _digit():
    return {
        "segments": [_pose(), _pose(), _pose()],
        "closure_commanded": 0.5,
        "closure_observed": 0.5,
    }


def _contact(hand, digit, normal):
    return {
        "collider_a": f"{hand}_{digit}",
        "collider_b": "target_001",
        "point_world_m": [0.0, 0.7, 0.0],
        "normal_world": normal,
        "separation_m": -0.001,
        "relative_velocity_world_m_s": [0.0, 0.0, 0.0],
        "available_impulse_n_s": 0.01,
        "provenance": "physx_measured",
    }


def _trace_row(step):
    if step < 70:
        height = 0.7
        angle = 0.0
    else:
        fraction = min(1.0, (step - 70) / 40.0)
        height = 0.7 + 0.12 * fraction
        angle = math.radians(35.0 * fraction)
    rotation = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
    hand_state = {"left_palm": _pose(), "right_palm": _pose()}
    for hand in ("left", "right"):
        for digit in ("thumb", "index", "middle", "ring", "little"):
            hand_state[f"{hand}_{digit}"] = _digit()
    contacts = []
    if 60 <= step <= 135:
        contacts.extend(
            [
                _contact("right", "thumb", [1.0, 0.0, 0.0]),
                _contact("right", "index", [-1.0, 0.0, 0.0]),
                _contact("right", "middle", [-1.0, 0.0, 0.0]),
            ]
        )
    if 70 <= step <= 135:
        contacts.extend(
            [
                _contact("left", "index", [-1.0, 0.0, 0.0]),
                _contact("left", "middle", [-1.0, 0.0, 0.0]),
            ]
        )
    return {
        "episode": {
            "episode_id": "episode_fixture",
            "target_id": "target_001",
            "destination_id": "table",
            "contact_strategy": "opposed_pinch",
            "final_gaze_zone": "window",
        },
        "clock": {
            "physics_step": step,
            "render_frame": step // 8,
            "time_s": step / 240.0,
            "phase_id": "capture" if step < 70 else "turn" if step < 136 else "release",
        },
        "body_state": {name: {"segment_id": name, **_pose()} for name in (
            "root", "pelvis", "torso", "neck", "head",
            "left_shoulder", "left_upper_arm", "left_elbow", "left_lower_arm", "left_forearm", "left_wrist",
            "right_shoulder", "right_upper_arm", "right_elbow", "right_lower_arm", "right_forearm", "right_wrist",
        )},
        "hand_state": hand_state,
        "controller_state": {
            "commanded_release": step == 136,
            "events": ["release_commanded"] if step == 136 else [],
            "targets": {},
            "errors": {},
            "recovery_state": "nominal",
            "speed_limits": {
                "palm_linear_max_m_s": 1.0,
                "palm_angular_max_rad_s": 2.0,
                "finger_segment_linear_max_m_s": 1.0,
                "finger_segment_angular_max_rad_s": 2.0,
                "provenance": "commanded",
            },
            "closure_limits": {
                "closure_min": 0.0,
                "closure_max": 1.0,
                "closure_rate_max_s": 10.0,
                "provenance": "commanded",
            },
        },
        "contacts": contacts,
        "objects": [
            {
                "persistent_id": "target_001",
                "semantic_id": "target",
                "instance_id": 17,
                "role": "target",
                **_pose((0.0, height, 0.0), rotation),
                "support_id": "table" if step < 70 or step >= 145 else None,
                "sleeping": step >= 145,
                "is_kinematic": False,
                "parent_id": None,
            }
        ],
        "camera_state": {
            **_pose((0.0, 1.1, 0.0)),
            "parent_id": "head",
            "intrinsics": {"fx": 1000.0},
            "clearance_m": 0.02,
            "optical_vs_face_forward_deg": 2.0,
        },
        "derived_state": {
            "joint_proprioception": {},
            "head_accelerometer_m_s2": [0.0, 9.81, 0.0],
            "head_gyroscope_rad_s": [0.0, 0.0, 0.0],
        },
        "assistance_ledger": [],
        "authority_counters": {
            "targetPoseWriteCounter": 0,
            "targetVelocityWriteCounter": 0,
            "targetForceCounter": 0,
            "targetTorqueCounter": 0,
            "targetJointCounter": 0,
            "targetParentingCounter": 0,
            "targetKinematicChangeCounter": 0,
        },
    }


def _find_check(report, check_id):
    return next(check for gate in report["gates"] for check in gate["checks"] if check["check_id"] == check_id)


def test_missing_evidence_is_unavailable_and_vetoes_promotion(tmp_path):
    report = audit_run(tmp_path)
    assert report["qa_decision"] == "PROMOTION_VETO"
    assert report["promotion_veto"]
    assert all(status == UNAVAILABLE for status in report["gate_summary"].values())
    assert not report["evidence_policy"]["visual_counters_accepted"]


def test_physx_contact_lift_turn_and_free_release_are_recomputed_from_trace(tmp_path):
    trace = tmp_path / "trace.json"
    _write_json(trace, [_trace_row(step) for step in range(160)])
    _write_json(
        tmp_path / "qa_evidence.json",
        {
            "schema": "embodied.independent_qa_evidence.v1",
            "trace": "trace.json",
            "target_object_id": "target_001",
        },
    )
    report = audit_run(tmp_path)
    assert _find_check(report, "trace.clock")["status"] == PASS
    assert _find_check(report, "trace.fields")["status"] == PASS
    assert _find_check(report, "interaction.right_capture")["status"] == PASS
    assert _find_check(report, "interaction.left_support")["status"] == PASS
    assert _find_check(report, "interaction.lift_turn")["status"] == PASS
    assert _find_check(report, "interaction.free_release")["status"] == PASS
    assert _find_check(report, "interaction.no_assistance")["status"] == PASS
    assert len(report["dense_timeline"]) == 20


def test_speculative_contacts_above_frozen_half_millimeter_never_increment_dwell(tmp_path):
    rows = [_trace_row(step) for step in range(160)]
    for row in rows:
        for contact in row["contacts"]:
            contact["separation_m"] = 0.0005001
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_json(
        tmp_path / "qa_evidence.json",
        {"schema": "embodied.independent_qa_evidence.v1", "trace": "episode_trace.json", "target_object_id": "target_001"},
    )
    report = audit_run(tmp_path)
    assert _find_check(report, "interaction.right_capture")["status"] == FAIL
    metrics = report["derived_metrics"]["interaction"]
    assert metrics["eligible_hand_target_rows"] == 0
    assert metrics["speculative_hand_target_rows_excluded"] > 0


def test_geometric_opposition_without_nonzero_impulse_cannot_qualify(tmp_path):
    rows = [_trace_row(step) for step in range(160)]
    for row in rows:
        for contact in row["contacts"]:
            contact["available_impulse_n_s"] = 0.0
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1",
        "trace": "episode_trace.json",
        "target_object_id": "target_001",
    })
    report = audit_run(tmp_path)
    assert _find_check(report, "interaction.right_capture")["status"] == FAIL
    assert _find_check(report, "interaction.left_support")["status"] == FAIL
    assert report["derived_metrics"]["interaction"]["right_measured_opposition_dwell_s"] > 0.0
    assert report["derived_metrics"]["interaction"]["right_required_digits_nonzero_impulse_dwell_s"] == 0.0


def test_supported_capture_instant_is_not_the_unsupported_qualified_lift_window(tmp_path):
    rows = [_trace_row(step) for step in range(160)]
    qualification_step = 72
    force_contacts = _trace_row(60)["contacts"]
    for step, row in enumerate(rows):
        row["contacts"] = force_contacts if step <= 135 else []
        row["controller_state"]["bimanual_qualification_step"] = (
            qualification_step if step >= qualification_step else -1
        )
        row["objects"][0]["support_id"] = "source_support" if step == qualification_step else None
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1",
        "trace": "episode_trace.json",
        "target_object_id": "target_001",
    })
    metrics = audit_run(tmp_path)["derived_metrics"]["interaction"]
    assert metrics["qualified_lift_manipulation_steps"]
    assert metrics["object_unsupported_during_qualified_manipulation"] is True


def test_dynamic_compliant_finger_truth_is_mandatory(tmp_path):
    row = _trace_row(0)
    for side in ("left", "right"):
        for digit in ("thumb", "index", "middle", "ring", "little"):
            state = row["hand_state"][f"{side}_{digit}"]
            body = {"is_kinematic": False, "mass_kg": 0.01, "provenance": "physx_measured", "pose": _pose()}
            joint = {
                "provenance": "engine_observed",
                "drive_provenance": "commanded ConfigurableJoint drive consumed by PhysX",
                "angular_x_drive_spring_n_m_rad": 1.0,
                "angular_x_drive_damper_n_m_s_rad": 1.0,
                "angular_x_drive_max_force_n_m": 1.0,
                "angular_yz_drive_spring_n_m_rad": 1.0,
                "angular_yz_drive_damper_n_m_s_rad": 1.0,
                "angular_yz_drive_max_force_n_m": 1.0,
            }
            state["dynamic_body_states"] = [body, body, body]
            state["compliant_joint_states"] = [joint, joint, joint]
    _write_json(tmp_path / "episode_trace.json", [row])
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1", "trace": "episode_trace.json",
        "target_object_id": "target_001",
    })
    assert _find_check(audit_run(tmp_path), "embodiment.compliant_dynamic_fingers")["status"] == PASS


def test_garment_affected_vertex_fraction_above_one_tenth_percent_fails(tmp_path):
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1",
        "registration_report": {
            "body_collider_provenance": "engine_observed",
            "garment_body_provenance": "engine_observed",
            "body_collider_registration": {"maximum_m": 0.004},
            "sampled_every_physics_step": True,
            "garments": [{
                "maximum_penetration_m": 0.001,
                "affected_fraction": 0.00101,
                "affected_vertex_distribution_counts": [1, 2],
            }],
        },
    })
    assert _find_check(audit_run(tmp_path), "registration.garment_body")["status"] == FAIL


def test_prerequisite_failure_lists_every_downstream_artifact_not_generated(tmp_path):
    report = audit_run(tmp_path)
    assert report["first_nonpassing_gate"] == "A"
    assert any("E three integrated episodes" in item for item in report["downstream_artifacts_not_generated"])
    assert any("F robustness" in item for item in report["downstream_artifacts_not_generated"])


def test_exact_integer_clock_does_not_false_fail_float32_microsecond_drift(tmp_path):
    rows = [_trace_row(step) for step in range(160)]
    for step, row in enumerate(rows):
        row["clock"]["time_s"] = step / 240.0 + 1.27e-6 * step / (len(rows) - 1)
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_json(
        tmp_path / "qa_evidence.json",
        {"schema": "embodied.independent_qa_evidence.v1", "trace": "episode_trace.json", "target_object_id": "target_001"},
    )
    report = audit_run(tmp_path)
    clock = _find_check(report, "trace.clock")
    assert clock["status"] == PASS
    assert clock["evidence"]["maximum_float_timestamp_drift_s"] > 1e-6
    assert clock["evidence"]["timestamp_drift_is_diagnostic_not_integer_mapping_evidence"]


def test_turn_is_measured_only_during_bimanual_turn_phase(tmp_path):
    rows = [_trace_row(step) for step in range(160)]
    rows[-1]["objects"][0]["rotation_world_xyzw"] = [0.0, 1.0, 0.0, 0.0]
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_json(
        tmp_path / "qa_evidence.json",
        {"schema": "embodied.independent_qa_evidence.v1", "trace": "episode_trace.json", "target_object_id": "target_001"},
    )
    report = audit_run(tmp_path)
    turn = report["derived_metrics"]["interaction"]["turn_during_bimanual_turn_deg"]
    assert turn == pytest.approx(35.0, abs=0.7)


def test_carry_telemetry_and_literal_same_row_postqualification_geometry_are_distinct(tmp_path):
    rows = [_trace_row(step) for step in range(40)]
    for step, row in enumerate(rows):
        row["contacts"] = []
        row["controller_state"]["bimanual_qualification_step"] = 10 if step >= 10 else -1
        row["controller_state"]["carry_contacts_maintained"] = 10 <= step <= 25
        if 10 <= step <= 25:
            row["contacts"] = [
                _contact("right", "thumb", [1.0, 0.0, 0.0]),
                _contact("right", "index", [-1.0, 0.0, 0.0]),
                _contact("right", "middle", [-1.0, 0.0, 0.0]),
                _contact("left", "index", [-1.0, 0.0, 0.0]),
            ]
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_json(
        tmp_path / "qa_evidence.json",
        {"schema": "embodied.independent_qa_evidence.v1", "trace": "episode_trace.json", "target_object_id": "target_001"},
    )
    metrics = audit_run(tmp_path)["derived_metrics"]["interaction"]
    assert metrics["controller_carry_telemetry_dwell_s"] == pytest.approx(16 / 240)
    assert metrics["literal_same_row_postqualification_opposing_geometry_dwell_s"] == pytest.approx(15 / 240)


def test_registered_capture_roll_overrides_coordinate_dependent_world_euler_roll(tmp_path):
    rows = [_trace_row(step) for step in range(16)]
    for row in rows:
        row["camera_state"]["rotation_world_xyzw"] = [0.0, 1.0, 0.0, 0.0]
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_jsonl(
        tmp_path / "capture_frame_ledger.jsonl",
        [{"render_frame": frame, "roll_deg": 7.91463} for frame in range(2)],
    )
    _write_json(
        tmp_path / "qa_evidence.json",
        {"schema": "embodied.independent_qa_evidence.v1", "trace": "episode_trace.json", "target_object_id": "target_001"},
    )
    camera = audit_run(tmp_path)["derived_metrics"]["camera"]
    assert camera["maximum_abs_roll_deg"] == pytest.approx(7.91463)
    assert camera["roll_source"] == "registered_capture_ledger.roll_deg"


def test_viewport_object_center_proxy_cannot_satisfy_visible_contact_gate(tmp_path):
    _write_json(
        tmp_path / "qa_evidence.json",
        {
            "schema": "embodied.independent_qa_evidence.v1",
            "contact_projection": {
                "records": [
                    {
                        "method": "dense viewport object_center proxy counter",
                        "hand": "right",
                        "digit": "thumb",
                        "physical_frame": 10,
                        "visible_frame": 10,
                        "physical_contact_point_world_m": [0.0, 0.0, 0.0],
                        "correct_visible_surface": True,
                        "visible_skin_surface_id": "right_thumb_skin",
                        "visible_object_surface_id": "target_001",
                        "skin_projection_error_m": 0.0,
                        "object_projection_error_m": 0.0,
                    }
                ]
            },
        },
    )
    report = audit_run(tmp_path)
    check = _find_check(report, "contact.visible_projection")
    assert check["status"] == FAIL
    assert report["promotion_veto"]


def test_registered_projection_adapter_cross_checks_embedded_ledger_and_physx_trace(tmp_path):
    rows = [_trace_row(step) for step in range(160)]
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_rgb_png(tmp_path / "head/semantic/labels.png", 2, 2, 100)
    _write_rgb_png(tmp_path / "head/instance/labels.png", 2, 2, 501)
    _write_json(tmp_path / "semantic_instance_manifest.json", {
        "schema": "embodied.semantic_instance_manifest.v1",
        "bindings": [{
            "renderer_path": "Authority/Avatar/WeightedBody",
            "semantic_name": "body_skin",
            "semantic_uint24": 100,
            "persistent_instance_name": "episode_fixture::weighted_body",
            "persistent_instance_uint24": 501,
        }],
    })
    semantic_hash = hashlib.sha256((tmp_path / "head/semantic/labels.png").read_bytes()).hexdigest()
    instance_hash = hashlib.sha256((tmp_path / "head/instance/labels.png").read_bytes()).hexdigest()
    state_hash = "d" * 64
    ledger_rows = []
    projection_records = []
    for step in (71, 79):
        visibility = []
        for contact in rows[step]["contacts"]:
            hand, digit = contact["collider_a"].split("_", 1)
            item = {
                "input_id": "physx_contact_point_input",
                "point_world_m": contact["point_world_m"],
                "pixel_xy": [1.0, 0.0],
                "rendered_label_pixel_xy": [1.0, 0.0],
                "rendered_semantic_uint24": 100,
                "rendered_persistent_instance_uint24": 501,
                "rendered_label_sample_complete": True,
                "expected_contact_collider_a": contact["collider_a"],
                "expected_contact_collider_b": contact["collider_b"],
                "first_visible_collider_id": contact["collider_a"],
                "first_visible_renderer_path": f"Avatar/{hand}/{digit}/Skin",
                "first_visible_semantic_uint24": 100,
                "first_visible_persistent_instance_uint24": 501,
                "contact_projects_to_expected_visible_surface": True,
                "contact_visible_in_registered_frame": True,
                "contact_projection_method": "actual semantic/instance uint24 values decoded at the projected contact pixel after both frozen replacement-shader renders; Physics.RaycastAll supplies only the expected collider surface check",
                "provenance": "rendered-pixel measured labels plus same-state PhysX contact",
            }
            visibility.append(item)
            projection_records.append({
                "render_frame": step // 8,
                "physics_step": step,
                "physical_contact_point_world_m": item["point_world_m"],
                **{key: value for key, value in item.items() if key not in {"input_id", "point_world_m", "contact_projection_method"}},
                "method": item["contact_projection_method"],
            })
        ledger_rows.append({
            "schema": "embodied.registered_capture_frame.v1",
            "render_frame": step // 8,
            "physics_step": step,
            "authority_state_sha256": state_hash,
            "event_visibility_inputs": visibility,
            "streams": [
                {"stream": "head_semantic_uint24", "relative_path": "head/semantic/labels.png",
                 "file_sha256": semantic_hash, "authority_state_sha256": state_hash},
                {"stream": "head_persistent_instance_uint24", "relative_path": "head/instance/labels.png",
                 "file_sha256": instance_hash, "authority_state_sha256": state_hash},
            ],
        })
    _write_jsonl(tmp_path / "capture_frame_ledger.jsonl", ledger_rows)
    ledger_hash = hashlib.sha256((tmp_path / "capture_frame_ledger.jsonl").read_bytes()).hexdigest()
    _write_json(tmp_path / "contact_projection.json", {
        "schema": "embodied.registered_contact_projection.v1",
        "source_capture_ledger": "capture_frame_ledger.jsonl",
        "source_capture_ledger_sha256": ledger_hash,
        "records": projection_records,
        "provenance": "direct adapter of embedded registered projection rows",
    })
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1",
        "trace": "episode_trace.json",
        "target_object_id": "target_001",
        "capture_frame_ledger": "capture_frame_ledger.jsonl",
        "contact_projection": "contact_projection.json",
    })
    check = _find_check(audit_run(tmp_path), "contact.visible_projection")
    assert check["status"] == PASS
    assert check["evidence"]["valid_records"] == len(projection_records)
    assert check["evidence"]["source_capture_ledger_sha256_matches"]

    _write_rgb_png(tmp_path / "head/instance/labels.png", 2, 2, 999)
    assert _find_check(audit_run(tmp_path), "contact.visible_projection")["status"] == FAIL


def test_free_release_must_settle_on_frozen_episode_destination(tmp_path):
    rows = [_trace_row(step) for step in range(160)]
    for row in rows[145:]:
        row["objects"][0]["support_id"] = "wrong_support"
    _write_json(tmp_path / "episode_trace.json", rows)
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1",
        "trace": "episode_trace.json",
        "target_object_id": "target_001",
    })
    check = _find_check(audit_run(tmp_path), "interaction.free_release")
    assert check["status"] == FAIL
    assert check["evidence"]["expected_destination_id"] == "table"
    assert check["evidence"]["observed_post_release_support_ids"] == ["wrong_support"]


def test_registered_visual_measurements_never_promote_geometric_receipts_to_visual_pass(tmp_path):
    _write_json(tmp_path / "visual_audit.json", {
        "schema": "embodied.registered_visual_measurements.v1",
        "measured_capture_evidence_only": True,
        "direct_decoded_frame_review_performed": False,
        "registered_projection_record_count": 12,
        "measurement_scope": "same-state first-visible-surface projections",
    })
    report = audit_run(tmp_path)
    for check_id in ("visual.garment_sweeps", "visual.motion_camera", "visual.event_visibility", "visual.episode_coherence"):
        assert _find_check(report, check_id)["status"] == UNAVAILABLE


def test_registered_capture_manifest_adapter_validates_real_stream_receipts(tmp_path):
    state_hash = "a" * 64
    streams = []
    for stream, filename in (
        ("head_rgb_hero", "head/rgb/frame_0000.png"),
        ("head_metric_depth_uint24_mm", "head/depth/frame_0000.png"),
        ("head_semantic_uint24", "head/semantic/frame_0000.png"),
        ("head_persistent_instance_uint24", "head/instance/frame_0000.png"),
    ):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((stream + "\n").encode())
        streams.append({
            "stream": stream,
            "relative_path": filename,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "authority_state_sha256": state_hash,
            "render_frame": 0,
            "physics_step": 7,
        })
    ledger = [{
        "schema": "embodied.registered_capture_frame.v1",
        "render_frame": 0,
        "physics_step": 7,
        "authority_state_sha256": state_hash,
        "authority_state_unchanged_across_modalities": True,
        "physics_advanced_between_modalities": False,
        "streams": streams,
        "intrinsics": {"fx_px": 1000.0, "fy_px": 1000.0},
        "camera_to_world_matrix_column_major": [0.0] * 16,
        "world_to_camera_matrix_column_major": [0.0] * 16,
        "projection_matrix_column_major": [0.0] * 16,
    }]
    manifest = {
        "schema": "embodied.registered_capture_manifest.v1",
        "width_px": 1920,
        "height_px": 1080,
        "fps": 30,
        "frames_captured": 1,
        "exact_integer_clock": True,
        "all_modalities_state_invariant": True,
        "physics_advanced_between_modalities": False,
        "hero_contains_proxy_pixels": False,
    }
    tolerances = load_frozen_config()["qa_tolerances"]
    check = _capture_checks(manifest, tmp_path, tolerances, 1 / 30, ledger)[0]
    assert check.status == PASS
    streams[2]["authority_state_sha256"] = "b" * 64
    assert _capture_checks(manifest, tmp_path, tolerances, 1 / 30, ledger)[0].status == FAIL


def test_registration_penetration_is_adapted_only_from_measured_trace_contacts(tmp_path):
    row = _trace_row(70)
    row["contacts"].append({
        "collider_a": "target_001",
        "collider_b": "table",
        "point_world_m": [0.0, 0.7, 0.0],
        "normal_world": [0.0, 1.0, 0.0],
        "separation_m": -0.001,
        "relative_velocity_world_m_s": [0.0, 0.0, 0.0],
        "available_impulse_n_s": 0.01,
        "provenance": "physx_measured",
    })
    _write_json(tmp_path / "episode_trace.json", [row])
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1",
        "trace": "episode_trace.json",
        "target_object_id": "target_001",
        "registration_report": {
            "schema": "embodied.embodiment_registration.v2",
            "body_collider_provenance": "EngineObserved",
            "garment_body_provenance": "EngineObserved",
            "passed": True,
            "anatomical_self_clearance_provenance": "PhysXMeasured",
            "self_clearance_sampled_every_physics_step": True,
            "non_adjacent_anatomy_clearance_passed": True,
            "non_adjacent_self_clearance_samples": 3,
            "non_adjacent_self_overlap_samples": 0,
            "self_clearance_steps": [{"swept_pair_pose_samples": 3, "overlap_samples": 0}],
            "solver_motion_self_clearance_provenance": "PhysXMeasured",
            "solver_motion_self_clearance_sampled_every_physics_step": True,
            "solver_motion_non_adjacent_anatomy_clearance_passed": True,
            "solver_motion_self_clearance_samples": 3,
            "solver_motion_self_overlap_samples": 0,
            "solver_motion_incomplete_sweep_intervals": 0,
            "expected_motion_sample_count": 1,
            "solver_motion_self_clearance_steps": [{"physics_step": 0, "swept_pair_pose_samples": 3, "overlap_samples": 0, "incomplete_sweep_intervals": 0}],
            "finger_object_penetration_provenance": "PhysXMeasured",
            "target_support_penetration_provenance": "PhysXMeasured",
            "finger_object_contact_samples": 5,
            "target_support_contact_samples": 1,
            "finger_object_max_penetration_m": 0.001,
            "target_support_max_penetration_m": 0.001,
            "body_collider_registration": {"maximum_m": 0.004},
            "sampled_every_physics_step": True,
            "garments": [{
                "maximum_penetration_m": 0.001,
                "affected_fraction": 0.001,
                "affected_vertex_distribution_counts": [9, 1],
            }],
        },
    })
    check = _find_check(audit_run(tmp_path), "registration.penetration")
    assert check["status"] == PASS
    assert check["evidence"]["finger_object_max_penetration_m"] == pytest.approx(0.001)
    assert check["evidence"]["target_support_max_penetration_m"] == pytest.approx(0.001)
    assert check["evidence"]["trace_contact_penetration_provenance"].startswith("physx_measured")


def test_trace_requires_all_nineteen_unique_body_segment_ids(tmp_path):
    row = _trace_row(0)
    row["body_state"]["right_wrist"]["segment_id"] = "left_wrist"
    _write_json(tmp_path / "episode_trace.json", [row])
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1", "trace": "episode_trace.json",
        "target_object_id": "target_001",
    })
    check = _find_check(audit_run(tmp_path), "trace.fields")
    assert check["status"] == FAIL
    assert check["evidence"]["missing_row_counts"]["body_state.duplicate_segment_ids"] == 1


def test_registration_pass_flag_and_swept_self_clearance_are_mandatory(tmp_path):
    report = {
        "schema": "embodied.embodiment_registration.v2",
        "body_collider_provenance": "EngineObserved",
        "garment_body_provenance": "EngineObserved",
        "passed": False,
        "body_collider_registration": {"maximum_m": 0.0},
        "sampled_every_physics_step": True,
        "garments": [{"maximum_penetration_m": 0.0, "affected_fraction": 0.0,
                      "affected_vertex_distribution_counts": [1]}],
        "anatomical_self_clearance_provenance": "PhysXMeasured",
        "self_clearance_sampled_every_physics_step": True,
        "non_adjacent_anatomy_clearance_passed": True,
        "non_adjacent_self_clearance_samples": 3,
        "non_adjacent_self_overlap_samples": 0,
        "self_clearance_steps": [{"swept_pair_pose_samples": 3, "overlap_samples": 0}],
        "solver_motion_self_clearance_provenance": "PhysXMeasured",
        "solver_motion_self_clearance_sampled_every_physics_step": True,
        "solver_motion_non_adjacent_anatomy_clearance_passed": True,
        "solver_motion_self_clearance_samples": 3,
        "solver_motion_self_overlap_samples": 0,
        "solver_motion_incomplete_sweep_intervals": 0,
        "expected_motion_sample_count": 1,
        "solver_motion_self_clearance_steps": [{"physics_step": 0, "swept_pair_pose_samples": 3, "overlap_samples": 0, "incomplete_sweep_intervals": 0}],
        "finger_object_max_penetration_m": 0.0,
        "target_support_max_penetration_m": 0.0,
    }
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1", "registration_report": report,
    })
    checks = [check for check_id in ("registration.skin_collider", "registration.garment_body", "registration.penetration")
              for check in [_find_check(audit_run(tmp_path), check_id)]]
    assert all(check["status"] == FAIL for check in checks)


def test_registration_rejects_missing_post_solver_swept_clearance(tmp_path):
    report = {
        "schema": "embodied.embodiment_registration.v2",
        "body_collider_provenance": "EngineObserved",
        "garment_body_provenance": "EngineObserved",
        "passed": True,
        "body_collider_registration": {"maximum_m": 0.0},
        "sampled_every_physics_step": True,
        "garments": [{"maximum_penetration_m": 0.0, "affected_fraction": 0.0,
                      "affected_vertex_distribution_counts": [1]}],
        "anatomical_self_clearance_provenance": "PhysXMeasured",
        "self_clearance_sampled_every_physics_step": True,
        "non_adjacent_anatomy_clearance_passed": True,
        "non_adjacent_self_clearance_samples": 3,
        "non_adjacent_self_overlap_samples": 0,
        "self_clearance_steps": [{"swept_pair_pose_samples": 3, "overlap_samples": 0}],
        "finger_object_max_penetration_m": 0.0,
        "target_support_max_penetration_m": 0.0,
    }
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1", "registration_report": report,
    })
    assert _find_check(audit_run(tmp_path), "registration.skin_collider")["status"] == FAIL


def test_matrix_and_robustness_ignore_completed_labels_without_independent_qa_receipts():
    qa_pass = {
        "schema": "embodied.independent_qa_report.v1",
        "qa_decision": "PROMOTION_VETO",
        "promotion_veto": True,
        "gate_summary": {**{gate: PASS for gate in "ABCD"}, "E": FAIL, "F": UNAVAILABLE},
    }
    episodes = []
    for index in range(3):
        episodes.append({
            "episode_id": f"episode_{index}", "room_family": f"room_{index}",
            "garment_configuration_id": f"garment_{index}", "compiler_id": "compiler",
            "catalog_id": "catalog", "controller_profile_id": "controller",
            "seed_specific_retuning": False, "assistance_entries": 0, "completed": True,
            "contextual_object_count": 11, "closed_finished_room": True,
            "no_visible_primitive_furniture": True, "all_reachable_objects_physics_backed": True,
            "compiler_generated": True, "destination_relation": f"relation_{index}",
            "contact_strategy": "pinch" if index < 2 else "envelopment",
        })
    visual = {"episodes": [{"episode_id": f"episode_{index}", "coherent": True,
                             "method": "dense decoded-frame visual review"} for index in range(3)]}
    assert _matrix_checks({"episodes": episodes}, visual)[0].status == FAIL
    for episode in episodes[:2]:
        episode["independent_qa"] = qa_pass
    assert _matrix_checks({"episodes": episodes}, visual)[0].status == PASS

    variants = [{"variant": name, "completed": True, "controller_profile_id": "controller",
                 "retuned": False, "assistance_entries": 0}
                for name in ("nominal", "lateral_target_shift", "mass_friction_change")]
    assert _robustness_checks({"primary_robustness": variants})[0].status == FAIL
    variants[0]["independent_qa"] = qa_pass
    variants[1]["independent_qa"] = qa_pass
    assert _robustness_checks({"primary_robustness": variants})[0].status == PASS


def test_source_audit_requires_python_coverage_and_explicit_rerender_exclusion(tmp_path):
    body = {
        "schema": "embodied.procedural_gate_source_audit.v1",
        "audit_policy": "fail_closed",
        "source_root": "procedural_scene_gate",
        "source_sha256": {name: "a" * 64 for name in (
            "__init__.py", "__main__.py", "runner.py", "contracts.py", "independent_qa.py", "cli.py",
            "ProceduralSceneGateBuilder.cs",
        )},
        "source_set_sha256": "b" * 64,
        "allowlist": [],
        "findings": [{
            "file": "ProceduralSceneGateBuilder.cs", "line": 10, "operation": "pose_write",
            "statement_sha256": "c" * 64, "target_related": False, "status": "ALLOWLISTED",
            "scope": "rerender_only",
            "reason": "explicit Physics-disabled render playback excluded from manipulation evidence",
        }],
        "forbidden_findings": [],
        "passed": True,
    }
    def receipt(value):
        result = dict(value)
        result["receipt_sha256"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return result
    source = receipt(body)
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1",
        "source_audit_receipt": source,
        "execution_receipt": {"source_audit_sha256": source["receipt_sha256"]},
        "authority_receipt": {"source_audit_sha256": source["receipt_sha256"]},
    })
    assert _find_check(audit_run(tmp_path), "authority.source_audit")["status"] == PASS

    body["source_sha256"].pop("cli.py")
    source = receipt(body)
    _write_json(tmp_path / "qa_evidence.json", {
        "schema": "embodied.independent_qa_evidence.v1",
        "source_audit_receipt": source,
        "execution_receipt": {"source_audit_sha256": source["receipt_sha256"]},
        "authority_receipt": {"source_audit_sha256": source["receipt_sha256"]},
    })
    check = _find_check(audit_run(tmp_path), "authority.source_audit")
    assert check["status"] == FAIL
    assert not check["evidence"]["configured_python_source_coverage"]


def test_unity_json_vector_and_digit_array_shape_is_accepted(tmp_path):
    row = _trace_row(0)

    def unity_vector(value):
        keys = ("x", "y", "z", "w")
        return {key: component for key, component in zip(keys, value)}

    def unity_pose(value):
        return {
            **value,
            "position_world_m": unity_vector(value["position_world_m"]),
            "rotation_world_xyzw": unity_vector(value["rotation_world_xyzw"]),
            "linear_velocity_world_m_s": unity_vector(value["linear_velocity_world_m_s"]),
            "angular_velocity_world_rad_s": unity_vector(value["angular_velocity_world_rad_s"]),
        }

    row["body_state"] = {name: unity_pose(value) for name, value in row["body_state"].items()}
    source_hands = row["hand_state"]
    hands = {
        "left_palm": unity_pose(source_hands["left_palm"]),
        "right_palm": unity_pose(source_hands["right_palm"]),
    }
    for side in ("left", "right"):
        hands[f"{side}_digits"] = []
        for digit in ("thumb", "index", "middle", "ring", "little"):
            source = source_hands[f"{side}_{digit}"]
            hands[f"{side}_digits"].append(
                {
                    "hand": side,
                    "digit": digit,
                    "segments": [
                        {"segment_index": index, "segment_id": f"{side}_{digit}_{index}", "pose": unity_pose(pose)}
                        for index, pose in enumerate(source["segments"])
                    ],
                    "closure_commanded": source["closure_commanded"],
                    "closure_observed": source["closure_observed"],
                }
            )
    row["hand_state"] = hands
    source_object = row["objects"][0]
    pose_keys = ("position_world_m", "rotation_world_xyzw", "linear_velocity_world_m_s", "angular_velocity_world_rad_s")
    row["objects"] = [{key: value for key, value in source_object.items() if key not in pose_keys} | {"pose": unity_pose(source_object)}]
    row["camera_state"] = unity_pose(row["camera_state"])
    _write_json(tmp_path / "episode_trace.json", [row])
    _write_json(
        tmp_path / "qa_evidence.json",
        {"schema": "embodied.independent_qa_evidence.v1", "trace": "episode_trace.json", "target_object_id": "target_001"},
    )
    report = audit_run(tmp_path)
    assert _find_check(report, "trace.fields")["status"] == PASS


def test_every_discovered_video_is_ffprobed_and_fully_decoded(tmp_path, monkeypatch):
    declared = []
    for name, role, clean, labeled in (
        ("primary_head.mp4", "clean_head", True, False),
        ("primary_external.mp4", "clean_external", True, False),
        ("primary_contact_overlay.mp4", "contact_overlay", False, True),
        ("extra_variant.mp4", "variant", True, False),
    ):
        (tmp_path / name).write_bytes(b"synthetic-video-placeholder")
        declared.append({"path": name, "role": role, "clean": clean, "labeled": labeled})
    _write_json(
        tmp_path / "qa_evidence.json",
        {"schema": "embodied.independent_qa_evidence.v1", "videos": declared},
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            payload = {
                "streams": [
                    {
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "pix_fmt": "yuv420p",
                        "avg_frame_rate": "30/1",
                        "nb_read_frames": "720",
                        "duration": "24.0",
                    }
                ],
                "format": {"duration": "24.0"},
            }
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/fake")
    monkeypatch.setattr("subprocess.run", fake_run)
    report = audit_run(tmp_path)
    assert len(report["video_audits"]) == 4
    assert all(item["full_decode_ok"] for item in report["video_audits"])
    assert _find_check(report, "video.every_file_decodes")["status"] == PASS
    assert _find_check(report, "video.separate_views")["status"] == PASS
    assert sum(command[0] == "ffprobe" for command in calls) == 4
    assert sum(command[0] == "ffmpeg" for command in calls) == 4


def test_evidence_paths_cannot_escape_run_root(tmp_path):
    outside = tmp_path.parent / "outside_trace.json"
    _write_json(outside, [])
    _write_json(
        tmp_path / "qa_evidence.json",
        {"schema": "embodied.independent_qa_evidence.v1", "trace": "../outside_trace.json"},
    )
    with pytest.raises(ValueError, match="escapes run root"):
        audit_run(tmp_path)


def test_canonical_unity_authority_and_expanded_provenance_receipts_are_accepted(tmp_path):
    authority = {
        "schema": "embodied.single_authority_receipt.v1",
        "physics_hz": 240,
        "render_hz": 30,
        "steps_per_render_frame": 8,
        "authority_root": "EpisodeAuthority",
        "avatar_root": "Avatar",
        "head": "Head",
        "camera_parent": "HeadMount",
        "target_rigidbody": "Target",
        "object_pose_writes_after_initialization": 999,
        "object_external_forces": 999,
        "attachment_or_joint_count": 999,
        "assistance_ledger_entries": 999,
        "independent_render_timeline": False,
        "single_state_drives_body_clothing_camera_truth": True,
    }
    _write_json(tmp_path / "authority_receipt.json", authority)
    provenance = [
        ("body_hand_camera_pose", "engine_observed"),
        ("body_hand_segment_velocity", "derived"),
        ("controller_targets", "commanded"),
        ("controller_errors", "derived"),
        ("contacts", "physx_measured"),
        ("free_object", "physx_measured"),
        ("joint_proprioception", "derived"),
        ("head_imu", "derived"),
        ("biological_torque", "unavailable"),
    ]
    _write_json(
        tmp_path / "episode_trace_manifest.json",
        {
            "provenance_registry": [
                {"field": field, "provenance": source, "formula_or_source": "frozen source", "units": "SI"}
                for field, source in provenance
            ]
        },
    )
    report = audit_run(tmp_path)
    authority_check = _find_check(report, "authority.single_state")
    assert authority_check["status"] == PASS
    assert "not independent runtime detectors" in authority_check["evidence"]["producer_counter_semantics"]
    assert _find_check(report, "truth.provenance")["status"] == PASS
