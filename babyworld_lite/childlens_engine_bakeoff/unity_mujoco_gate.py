"""Canonical trace-first Unity--MuJoCo embodied episode gate.

MuJoCo is the only state authority.  Unity consumes the immutable render-rate
projection of the 240 Hz trace and never steps physics.  Complete run products
belong under the ignored ``runs/`` root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def unity_position_to_mj(row: dict[str, float]) -> np.ndarray:
    return np.asarray([row["x"], -row["z"], row["y"]], dtype=float)


def unity_quaternion_matrix(row: dict[str, float]) -> np.ndarray:
    x, y, z, w = (row[k] for k in ("x", "y", "z", "w"))
    unity = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    # B maps MuJoCo vectors to Unity: (x, y, z) -> (x, z, -y).
    bridge = np.asarray([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    return bridge.T @ unity @ bridge


def matrix_quaternion(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4); mujoco.mju_mat2Quat(quat, matrix.reshape(-1)); return quat


def manifest_registration(manifest_path: Path, output: Path, manipulation: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text()); bones = {row["name"]: row for row in manifest["bones"]}
    world_p = {name: unity_position_to_mj(row["world_position_unity"]) for name, row in bones.items()}
    world_r = {name: unity_quaternion_matrix(row["world_rotation_unity"]) for name, row in bones.items()}
    children: dict[str, list[str]] = {name: [] for name in bones}; roots = []
    for name, row in bones.items():
        parent = row["retained_parent"]
        if parent in bones: children[parent].append(name)
        else: roots.append(name)
    joint_names = {
        "spine03": "torso_pitch", "head": "head_yaw", "upperarm01.R": "shoulder_pitch",
        "lowerarm01.R": "elbow_flex", "wrist.R": "wrist_roll",
    }
    digit_labels = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "little"}
    for digit in range(1, 6):
        for part, label in ((1, "proximal"), (2, "middle"), (3, "distal")):
            joint_names[f"finger{digit}-{part}.R"] = f"{digit_labels[digit]}_{label}_flex"
    def body_xml(name: str, parent: str | None) -> str:
        if parent is None: pos, rotation = world_p[name], world_r[name]
        else: rotation = world_r[parent].T @ world_r[name]; pos = world_r[parent].T @ (world_p[name] - world_p[parent])
        quat = matrix_quaternion(rotation)
        nested = "".join(body_xml(child, name) for child in children[name])
        joint = f'<joint name="{joint_names[name]}" axis="1 0 0" range="-0.9 0.9" damping="0.08" armature="0.0002"/><inertial pos="0 0 0" mass="0.01" diaginertia="0.00002 0.00002 0.00002"/>' if name in joint_names else ""
        collider = ""
        if manipulation and name.startswith("finger"):
            finger_children = [child for child in children[name] if child.startswith("finger")]
            endpoint = world_r[name].T @ (world_p[finger_children[0]] - world_p[name]) if finger_children else np.asarray([0.0, 0.018, 0.0])
            part = int(name.split("-")[1].split(".")[0]); radius = {1: .0085, 2: .0075, 3: .0065}[part]
            collider = f'<geom name="collider_{name}" type="capsule" fromto="0 0 0 {endpoint[0]} {endpoint[1]} {endpoint[2]}" size="{radius}" mass="0.004" friction="1.8 0.02 0.002" contype="2" conaffinity="1" solref="0.008 1" solimp="0.95 0.99 0.001"/>'
        if manipulation and name == "wrist.R": collider += '<geom name="palm_collider" type="box" pos="0 0.025 0" size="0.045 0.035 0.025" mass="0.04" friction="1.6 0.02 0.002" contype="4" conaffinity="1"/>'
        return f'<body name="rest_{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">{joint}{collider}<site name="landmark_{name}" size="0.002"/>{nested}</body>'
    actuators = ''.join(f'<position name="{joint}_motor" joint="{joint}" kp="18" kv="1.0" forcerange="-7 7"/>' for joint in joint_names.values())
    scene = ''
    if manipulation:
        scene = '<geom name="table" type="box" pos="0.18 -0.315 0.39625" size="0.60 0.45 0.39625" friction="1.3 0.02 0.002" contype="8" conaffinity="1"/><body name="red_toy_001" pos="0.18 -0.315 0.8202"><freejoint name="red_toy_free"/><geom name="red_toy_001_geom" type="box" size="0.0275 0.0275 0.0275" mass="0.055" friction="1.35 0.02 0.002" solref="0.006 1" solimp="0.97 0.995 0.001"/></body>'
    gravity = '0 0 -9.81' if manipulation else '0 0 0'
    xml = f'<mujoco model="mpfb_manifest_rest"><compiler angle="radian"/><option timestep="0.004166666666666667" gravity="{gravity}" integrator="implicitfast" iterations="100" cone="elliptic"/><worldbody>' + ''.join(body_xml(root, None) for root in roots) + scene + f'</worldbody><actuator>{actuators}</actuator></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    errors = {}; body_pose = {}
    for name in bones:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"landmark_{name}")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"rest_{name}")
        errors[name] = float(np.linalg.norm(data.site_xpos[site] - world_p[name]))
        body_pose[name] = np.concatenate((data.xpos[body], data.xquat[body])).tolist()
    digit_names = [f"finger{digit}-{part}.R" for digit in range(1, 6) for part in range(1, 4)]
    actuator_ids = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
    body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"rest_{name}") for name in bones}
    model_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    frames, qpos_stream, qvel_stream, command_stream = [], [], [], []
    contacted_digits: set[str] = set(); maximum_contact_digits = 0; contact_samples = 0
    required_digits = {"thumb", "index", "middle"}
    finger_command_deg = {digit: -30.0 for digit in digit_labels.values()}
    previous_force_n = {digit: 0.0 for digit in digit_labels.values()}
    graph_dwell_steps = 0; graph_qualified = False; qualification_time_s = None
    recovery_state = "approach"; recovery_transitions: list[dict[str, Any]] = []
    shoulder_recenter_deg = 0.0
    # Measured outside-head offset from the verified audition, but with the
    # forbidden 50-degree optical pitch removed.  Identity camera axes are
    # neutral under the explicit coordinate bridge; head/torso joints gaze.
    camera_unity = np.asarray([bones["head"]["world_position_unity"][k] for k in ("x", "y", "z")]) + np.asarray([0.0, 0.07643974, -0.13252706])
    camera_mj = np.asarray([camera_unity[0], -camera_unity[2], camera_unity[1]])
    # MuJoCo camera columns are right/up/back.  The fixed mount preserves the
    # verified audition's outside-head face-forward direction; gaze comes from
    # the trace-driven head joint, not an optical pitch or target lock.
    camera_rotation_rest = np.asarray([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    head_rest_position = world_p["head"]
    head_rest_rotation = world_r["head"]
    camera_local_position = head_rest_rotation.T @ (camera_mj - head_rest_position)
    camera_local_rotation = head_rest_rotation.T @ camera_rotation_rest
    for step in range(5281):
        t = step / 240
        gesture = smooth(3.0, 7.0, t) * (1 - smooth(17.0, 20.5, t))
        close = smooth(6.5, 8.5, t) * (1 - smooth(17.5, 19.5, t))
        lift = smooth(0, 2.4, max(0.0, t - qualification_time_s)) * (1 - smooth(16.8, 18.5, t)) if manipulation and graph_qualified else 0
        targets = {"torso_pitch": np.deg2rad(-10) * gesture, "head_yaw": np.deg2rad(24) * gesture, "shoulder_pitch": np.deg2rad(22 + shoulder_recenter_deg + 13 * lift) * gesture, "elbow_flex": np.deg2rad(28 - 6 * lift) * gesture, "wrist_roll": np.deg2rad(24) * smooth(12,15,t) * (1-smooth(16,18,t))}
        for joint in joint_names.values():
            if "_flex" in joint and joint not in ("elbow_flex",):
                part_ratio = 1.0 if "proximal" in joint else .72 if "middle" in joint else .55
                digit = joint.split("_")[0]
                targets[joint] = np.deg2rad(finger_command_deg[digit]) * part_ratio
        for joint, value in targets.items(): data.ctrl[actuator_ids[f"{joint}_motor"]] = value
        if step: mujoco.mj_step(model, data)
        else: mujoco.mj_forward(model, data)
        step_digits: set[str] = set(); step_force_n = {digit: 0.0 for digit in digit_labels.values()}; signed_force_x = 0.0
        if manipulation:
            target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_toy_001_geom")
            for ci in range(data.ncon):
                con = data.contact[ci]
                if target_geom not in (con.geom1, con.geom2): continue
                other = con.geom2 if con.geom1 == target_geom else con.geom1
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                if geom_name.startswith("collider_finger"):
                    digit = digit_labels[int(geom_name.split("finger")[1][0])]; step_digits.add(digit)
                    wrench = np.zeros(6); mujoco.mj_contactForce(model, data, ci, wrench)
                    step_force_n[digit] += abs(float(wrench[0])); signed_force_x += float(con.frame[0]) * float(wrench[0])
                elif geom_name == "palm_collider": step_digits.add("palm")
            contacted_digits |= step_digits; maximum_contact_digits = max(maximum_contact_digits, len(step_digits)); contact_samples += bool(step_digits)
            if 6.5 <= t < 17.5:
                for digit in finger_command_deg:
                    if digit in step_digits:
                        # Contact-responsive impedance: hold at contact and make
                        # only a bounded force-error correction toward 1.0 N.
                        finger_command_deg[digit] += np.clip(1.0 - step_force_n[digit], -1.0, 1.0) * (4.0 / 240.0)
                    else:
                        finger_command_deg[digit] += 24.0 / 240.0
                    finger_command_deg[digit] = float(np.clip(finger_command_deg[digit], -30.0, 50.0))
                graph_dwell_steps = graph_dwell_steps + 1 if required_digits <= step_digits else 0
                if not graph_qualified and graph_dwell_steps >= 48:
                    graph_qualified = True; qualification_time_s = t
                    recovery_transitions.append({"time_s": t, "state": "qualified_lift"}); recovery_state = "qualified_lift"
                shoulder_recenter_deg = float(np.clip(shoulder_recenter_deg - np.clip(signed_force_x, -2.0, 2.0) * 0.002, -3.0, 3.0))
                if t > 11.0 and not graph_qualified and recovery_state != "retry_open":
                    recovery_state = "retry_open"; recovery_transitions.append({"time_s": t, "state": recovery_state})
                if recovery_state == "retry_open":
                    for digit in finger_command_deg: finger_command_deg[digit] = max(-30.0, finger_command_deg[digit] - 36.0 / 240.0)
            elif t >= 17.5:
                for digit in finger_command_deg: finger_command_deg[digit] = max(-30.0, finger_command_deg[digit] - 36.0 / 240.0)
                recovery_state = "commanded_release"
            previous_force_n = step_force_n
        qpos_stream.append(data.qpos.copy()); qvel_stream.append(data.qvel.copy()); command_stream.append(data.ctrl.copy())
        if step < 5280 and step % 8 == 0:
            frame = step // 8
            def pose(name: str) -> list[float]:
                body = body_ids[name]; return np.concatenate((data.xpos[body], data.xquat[body])).tolist()
            head_pose = pose("head"); wrist_pose = pose("wrist.R")
            head_body = body_ids["head"]
            head_rotation = data.xmat[head_body].reshape(3, 3)
            camera_position = data.xpos[head_body] + head_rotation @ camera_local_position
            camera_rotation = head_rotation @ camera_local_rotation
            target_pose = [5,5,5,1,0,0,0]
            if manipulation:
                target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_toy_001"); target_pose = np.concatenate((data.xpos[target_body], data.xquat[target_body])).tolist()
            phases = (("scan_reorient", 3), ("reach_touch", 7), ("preshape_grasp", 10), ("lift_inspect", 14), ("manipulate", 17), ("place_release_withdraw", 21), ("gaze_away", 22.1))
            phase = next(label for label, end in phases if t < end) if manipulation else "dynamic_registration_no_object"
            frames.append({"frame": frame, "truth_index": step, "time_s": step / 240, "phase": phase, "qpos": data.qpos.tolist(), "wrist_pose_mj": wrist_pose, "head_pose_mj": head_pose, "camera_pose_mj": [*camera_position, *camera_rotation.reshape(-1)], "target_pose_mj": target_pose, "contact_points_mj": [[0,0,0] for _ in range(6)], "digit_segment_pose_mj": sum((pose(name) for name in digit_names), []), "contact_digits": len(step_digits), "support_contact": False})
    projection = {"schema": "embodied.unity_trace_projection.v1", "physics_hz": 240, "render_hz": 30, "steps_per_frame": 8, "joint_names": model_joint_names, "digit_segment_names": digit_names, "frames": frames}
    output.mkdir(parents=True, exist_ok=True); (output / "model.xml").write_text(xml); (output / "render_trace.json").write_bytes(canonical_json(projection)); np.savez_compressed(output / "authoritative_trace.npz", qpos=np.asarray(qpos_stream), qvel=np.asarray(qvel_stream), actuator_command=np.asarray(command_stream), time_s=np.arange(5281) / 240)
    qa = {"schema": "embodied.manifest_registration.v1", "bones": len(bones), "maximum_roundtrip_landmark_error_m": max(errors.values()), "mean_roundtrip_landmark_error_m": float(np.mean(list(errors.values()))), "frozen_tolerance_m": 0.01, "passed": max(errors.values()) <= 0.01, "per_bone_error_m": errors, "manifest_sha256": sha256(manifest_path), "manipulation": manipulation}
    if manipulation:
        target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_toy_001"); target_z = np.asarray([frame["target_pose_mj"][2] for frame in frames]); qa["lift_m"] = float(target_z.max() - target_z[0]); qa["final_target_position_m"] = data.xpos[target_body].tolist(); qa["contacted_digits"] = sorted(contacted_digits); qa["maximum_simultaneous_contact_digits"] = maximum_contact_digits; qa["contact_samples"] = int(contact_samples)
        qa["required_contact_graph_met_over_episode"] = {"thumb", "index", "middle"} <= contacted_digits
        qa["stable_support_grasp_through_lift"] = False
        qa["stage_c_passed"] = False
        qa["stage_c_failure"] = "The free target escapes laterally during actuator-only closure; no stable opposed multi-digit support survives the commanded lift."
        qa["tactile_impedance_attempt"] = {"contact_graph_dwell_s": 0.2, "force_target_n_per_digit": 1.0, "force_adjustment_limit_deg_s": 4.0, "contact_free_closure_deg_s": 24.0, "shoulder_recenter_limit_deg": 3.0, "shoulder_recenter_gain_deg_per_n_step": 0.002, "retry_open_deg_s": 36.0, "qualification_time_s": qualification_time_s, "recovery_transitions": recovery_transitions, "final_recovery_state": recovery_state}
    (output / "manifest_registration_qa.json").write_bytes(canonical_json(qa))


def smooth(a: float, b: float, t: float) -> float:
    x = float(np.clip((t - a) / (b - a), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def validate_rest_manifest(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"rest manifest does not exist: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "embodied.mpfb_rest_manifest.v1":
        raise ValueError("rest manifest must use schema embodied.mpfb_rest_manifest.v1")
    names = {row.get("name") for row in manifest.get("bones", [])}
    required = {"root", "head", "wrist.R"} | {
        f"finger{digit}-{part}.R" for digit in range(1, 6) for part in range(1, 4)
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"rest manifest is missing required retained bones: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rest-manifest", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--registration", action="store_true", help="Run manifest-derived no-object registration")
    modes.add_argument("--manipulation", action="store_true", help="Run manifest-derived tactile manipulation gate")
    args = parser.parse_args()
    validate_rest_manifest(args.rest_manifest)
    manifest_registration(args.rest_manifest, args.output, manipulation=args.manipulation)


if __name__ == "__main__":
    main()
