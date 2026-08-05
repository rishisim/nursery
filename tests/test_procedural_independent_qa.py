import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate.independent_qa import (
    FAIL,
    PASS,
    UNAVAILABLE,
    audit_run,
)


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


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
        height = 0.7 + 0.09 * fraction
        angle = math.radians(25.0 * fraction)
    rotation = (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))
    hand_state = {"left_palm": _pose(), "right_palm": _pose()}
    for hand in ("left", "right"):
        for digit in ("thumb", "index", "middle", "ring", "little"):
            hand_state[f"{hand}_{digit}"] = _digit()
    contacts = []
    if step <= 135:
        contacts.extend(
            [
                _contact("right", "thumb", [1.0, 0.0, 0.0]),
                _contact("right", "index", [1.0, 0.0, 0.0]),
                _contact("right", "middle", [1.0, 0.0, 0.0]),
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
        "clock": {
            "physics_step": step,
            "render_frame": step // 8,
            "time_s": step / 240.0,
            "phase_id": "capture" if step < 70 else "turn" if step < 136 else "release",
        },
        "body_state": {name: _pose() for name in ("root", "torso", "neck", "head")},
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
    assert turn == pytest.approx(25.0, abs=0.7)


def test_carry_telemetry_and_literal_same_row_postqualification_geometry_are_distinct(tmp_path):
    rows = [_trace_row(step) for step in range(40)]
    for step, row in enumerate(rows):
        row["contacts"] = []
        row["controller_state"]["bimanual_qualification_step"] = 10 if step >= 10 else -1
        row["controller_state"]["carry_contacts_maintained"] = 10 <= step <= 25
        if 10 <= step <= 25:
            row["contacts"] = [
                _contact("right", "thumb", [1.0, 0.0, 0.0]),
                _contact("right", "index", [1.0, 0.0, 0.0]),
                _contact("right", "middle", [1.0, 0.0, 0.0]),
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
                        "nb_read_frames": "480",
                        "duration": "16.0",
                    }
                ],
                "format": {"duration": "16.0"},
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
