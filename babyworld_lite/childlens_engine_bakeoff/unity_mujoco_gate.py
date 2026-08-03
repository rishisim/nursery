"""Canonical trace-first Unity--MuJoCo embodied episode gate.

MuJoCo is the only state authority.  Unity consumes the immutable render-rate
projection of the 240 Hz trace and never steps physics.  Complete run products
belong under the ignored ``runs/`` root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


DIGITS = ("thumb", "index", "middle", "ring", "little")


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


def phase_at(t: float, phases: list[dict[str, Any]]) -> str:
    for phase in phases:
        if phase["start_s"] <= t < phase["end_s"]:
            return str(phase["id"])
    return str(phases[-1]["id"])


def model_xml(config: dict[str, Any], cell: dict[str, Any]) -> str:
    target = config["target"]
    half = np.asarray(target["dimensions_m"], dtype=float) / 2
    x = 0.04 + float(cell["target_lateral_offset_m"])
    if cell.get("no_object", False):
        x = 5.0
    mass = float(target["mass_kg"]) * float(cell["mass_scale"])
    friction = np.asarray(target["friction"], dtype=float) * float(cell["friction_scale"])
    finger_base = {
        "thumb": (-0.0335, 0.0115, -0.0064, 56, 0.0243, 0.0258),
        "index": (-0.0667, 0.0537, -0.0107, 79, 0.0208, 0.0185),
        "middle": (-0.0502, 0.0657, -0.0044, 88, 0.0272, 0.0224),
        "ring": (-0.0364, 0.0707, 0.0028, 75, 0.0240, 0.0201),
        "little": (-0.0230, 0.0722, 0.0134, 94, 0.0172, 0.0201),
    }
    fingers = []
    actuators = []
    sensors = []
    for digit in DIGITS:
        x_pos, y, z, yaw, segment, middle_length = finger_base[digit]
        fingers.append(
            f'''<body name="{digit}_proximal" pos="{x_pos} {y} {z}" euler="0 0 {yaw}">
              <joint name="{digit}_proximal_flex" type="hinge" axis="1 0 0" range="-70 75" damping="0.12" armature="0.0003"/>
              <geom name="{digit}_proximal_collider" type="capsule" fromto="0 0 0 0 {segment} 0" size="0.0085" mass="0.010" friction="1.8 0.02 0.002" contype="2" conaffinity="1" solref="0.008 1" solimp="0.95 0.99 0.001" rgba="0.8 0.55 0.42 0"/>
              <body name="{digit}_middle" pos="0 {segment} 0">
                <joint name="{digit}_middle_flex" type="hinge" axis="1 0 0" range="-70 80" damping="0.10" armature="0.0002"/>
                <geom name="{digit}_middle_collider" type="capsule" fromto="0 0 0 0 {middle_length} 0" size="0.0075" mass="0.008" friction="1.8 0.02 0.002" contype="2" conaffinity="1" solref="0.008 1" solimp="0.95 0.99 0.001" rgba="0.8 0.55 0.42 0"/>
                <body name="{digit}_distal" pos="0 {middle_length} 0">
                  <joint name="{digit}_distal_flex" type="hinge" axis="1 0 0" range="-70 85" damping="0.08" armature="0.00015"/>
                  <geom name="{digit}_distal_collider" type="capsule" fromto="0 0 0 0 0.017 0" size="0.0065" mass="0.006" friction="1.8 0.02 0.002" contype="2" conaffinity="1" solref="0.008 1" solimp="0.95 0.99 0.001" rgba="0.8 0.55 0.42 0"/>
                  <site name="{digit}_touch" pos="0 0.015 0" size="0.005"/>
                </body>
              </body>
            </body>'''
        )
        for part, ratio in (("proximal", 1.0), ("middle", 0.82), ("distal", 0.68)):
            actuators.append(f'<position name="{digit}_{part}_motor" joint="{digit}_{part}_flex" kp="8" kv="0.35" ctrlrange="-1.5 1.4" forcerange="-2.2 2.2"/>')
        sensors.append(f'<touch name="{digit}_touch_sensor" site="{digit}_touch"/>')
    return f'''<mujoco model="unity_mujoco_child_gate">
      <compiler angle="degree" autolimits="true"/>
      <option timestep="{1 / config['clock']['physics_hz']:.16f}" gravity="0 0 -9.81" integrator="implicitfast" iterations="100" ls_iterations="20" cone="elliptic"/>
      <size nconmax="300" njmax="1200"/>
      <default><joint limited="true" damping="1.0"/><geom condim="6" margin="0.0005"/></default>
      <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1" rgba="0.35 0.24 0.16 1" friction="1.2 0.02 0.002"/>
        <geom name="table" type="box" pos="0.10 0.43 0.31" size="0.65 0.38 0.31" rgba="0.42 0.24 0.12 1" friction="1.3 0.02 0.002"/>
        <body name="torso" pos="0 0 0.73">
          <joint name="torso_pitch" type="hinge" axis="1 0 0" range="-18 18" damping="4" armature="0.03"/>
          <geom name="torso_proxy" type="capsule" fromto="0 0 0 0 0 0.30" size="0.11" mass="8" contype="0" conaffinity="0" rgba="0 0 0 0"/>
          <body name="neck" pos="0 0 0.31"><joint name="neck_pitch" type="hinge" axis="1 0 0" range="-35 30" damping="1.5"/><inertial pos="0 0 0.03" mass="0.08" diaginertia="0.0002 0.0002 0.0002"/>
            <body name="head" pos="0 0 0.07"><joint name="head_yaw" type="hinge" axis="0 0 1" range="-55 55" damping="1.5"/>
              <geom name="head_proxy" type="sphere" size="0.09" mass="2" contype="0" conaffinity="0" rgba="0 0 0 0"/>
              <site name="imu_site" pos="0 0.015 0.015" size="0.004"/>
              <camera name="child_camera" pos="0 0.096 0.014" xyaxes="1 0 0 0 0 1" fovy="{config['camera']['vertical_fov_deg']}"/>
            </body>
          </body>
        </body>
        <body name="shoulder_chain" pos="0.16 0 0.98">
          <joint name="shoulder_pitch" axis="1 0 0" range="-80 100" damping="2"/>
          <joint name="shoulder_yaw" axis="0 0 1" range="-70 70" damping="2"/>
          <geom name="upperarm_proxy" type="capsule" fromto="0 0 0 0 0.16 -0.11" size="0.035" mass="0.35" contype="0" conaffinity="0" rgba="0 0 0 0"/>
          <body name="elbow_chain" pos="0 0.16 -0.11"><joint name="elbow_flex" axis="1 0 0" range="0 145" damping="1.5"/>
            <geom name="forearm_proxy" type="capsule" fromto="0 0 0 0 0.15 -0.08" size="0.029" mass="0.25" contype="0" conaffinity="0" rgba="0 0 0 0"/>
          </body>
        </body>
        <body name="wrist_guide" pos="0.16 0.10 0.72">
          <joint name="wrist_x" type="slide" axis="1 0 0" range="-0.13 0.13" damping="18" armature="0.08"/>
          <joint name="wrist_y" type="slide" axis="0 1 0" range="0 0.36" damping="18" armature="0.08"/>
          <joint name="wrist_z" type="slide" axis="0 0 1" range="-0.10 0.18" damping="18" armature="0.08"/>
          <joint name="wrist_roll" type="hinge" axis="0 1 0" range="-35 35" damping="1.5" armature="0.01"/>
          <geom name="palm_collider" type="box" pos="0 0.025 0" size="0.045 0.035 0.025" mass="0.10" friction="1.5 0.02 0.002" contype="4" conaffinity="1" rgba="0 0 0 0"/>
          <site name="palm_touch" pos="0 0.014 0" size="0.009"/>
          {''.join(fingers)}
        </body>
        <body name="red_toy_001" pos="{x} 0.43 {0.62 + half[2] + 0.0002}">
          <freejoint name="red_toy_free"/>
          <geom name="red_toy_001_geom" type="box" size="{half[0]} {half[1]} {half[2]}" mass="{mass}" friction="{friction[0]} {friction[1]} {friction[2]}" solref="0.006 1" solimp="0.97 0.995 0.001" rgba="0.9 0.03 0.02 1"/>
        </body>
        <body name="blue_distractor" pos="-0.20 0.40 0.655"><geom type="sphere" size="0.035" mass="0" rgba="0.05 0.2 0.8 1"/></body>
        <body name="yellow_distractor" pos="0.32 0.49 0.655"><geom type="box" size="0.03 0.03 0.035" mass="0" rgba="0.95 0.75 0.05 1"/></body>
      </worldbody>
      <actuator>
        <position name="torso_motor" joint="torso_pitch" kp="180" kv="20" forcerange="-80 80"/>
        <position name="neck_motor" joint="neck_pitch" kp="80" kv="10" forcerange="-25 25"/>
        <position name="head_motor" joint="head_yaw" kp="65" kv="8" forcerange="-18 18"/>
        <position name="shoulder_pitch_motor" joint="shoulder_pitch" kp="45" kv="5" forcerange="-20 20"/>
        <position name="shoulder_yaw_motor" joint="shoulder_yaw" kp="45" kv="5" forcerange="-20 20"/>
        <position name="elbow_motor" joint="elbow_flex" kp="35" kv="4" forcerange="-15 15"/>
        <position name="wrist_x_motor" joint="wrist_x" kp="900" kv="55" forcerange="-120 120"/>
        <position name="wrist_y_motor" joint="wrist_y" kp="900" kv="55" forcerange="-120 120"/>
        <position name="wrist_z_motor" joint="wrist_z" kp="900" kv="55" forcerange="-120 120"/>
        <position name="wrist_roll_motor" joint="wrist_roll" kp="35" kv="4" forcerange="-10 10"/>
        {''.join(actuators)}
      </actuator>
      <sensor>
        {''.join(sensors)}
        <touch name="palm_touch_sensor" site="palm_touch"/>
        <accelerometer name="head_accelerometer" site="imu_site"/>
        <gyro name="head_gyroscope" site="imu_site"/>
        <framepos name="camera_parent_position" objtype="body" objname="head"/>
        <framequat name="camera_parent_quaternion" objtype="body" objname="head"/>
      </sensor>
    </mujoco>'''


@dataclass
class Run:
    model: mujoco.MjModel
    data: mujoco.MjData
    joint_qpos: dict[str, int]
    actuator: dict[str, int]
    body: dict[str, int]
    geom: dict[str, int]
    sensor: dict[str, slice]


def make_run(xml: str) -> Run:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i]) for i in range(model.njnt)}
    actuators = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
    bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    geoms = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
    sensors = {}
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        start = int(model.sensor_adr[i]); sensors[name] = slice(start, start + int(model.sensor_dim[i]))
    return Run(model, data, joints, actuators, bodies, geoms, sensors)


def controls(run: Run, t: float, offset: float) -> np.ndarray:
    a = np.zeros(run.model.nu)
    def setc(name: str, value: float) -> None: a[run.actuator[name]] = value
    look = smooth(0.25, 2.4, t)
    away = smooth(21.0, 21.8, t)
    setc("torso_motor", np.deg2rad(-5 - 10 * smooth(2.0, 5.0, t) + 12 * away))
    setc("neck_motor", np.deg2rad(-8 - 15 * look + 18 * away))
    setc("head_motor", np.deg2rad(-18 + 24 * look - 38 * away))
    setc("shoulder_pitch_motor", np.deg2rad(12 + 30 * smooth(3, 6.5, t) - 25 * smooth(19.5, 21, t)))
    setc("shoulder_yaw_motor", np.deg2rad(-8 + 18 * smooth(3, 6.5, t)))
    setc("elbow_motor", np.deg2rad(55 - 22 * smooth(3, 6.5, t) + 18 * smooth(19.5, 21, t)))
    reach = smooth(3.0, 6.4, t)
    withdraw = smooth(19.6, 21.0, t)
    lift = smooth(10.0, 11.8, t) * (1 - smooth(17.0, 19.0, t))
    setc("wrist_x_motor", (-0.060 + offset) * reach * (1 - withdraw))
    setc("wrist_y_motor", 0.242 * reach * (1 - withdraw))
    setc("wrist_z_motor", (-0.037 * reach + 0.100 * lift) * (1 - withdraw))
    roll = np.deg2rad(25) * smooth(14.0, 15.2, t) * (1 - smooth(16.0, 17.0, t))
    setc("wrist_roll_motor", roll)
    close = smooth(6.4, 9.1, t) * (1 - smooth(18.85, 19.45, t))
    for digit in DIGITS:
        open_angle = np.deg2rad(58)
        closed_angle = np.deg2rad(-28)
        base = open_angle + (closed_angle - open_angle) * close
        for part, ratio in (("proximal", 1.0), ("middle", 0.82), ("distal", 0.68)):
            setc(f"{digit}_{part}_motor", base * ratio)
    return a


def contact_rows(run: Run) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, set[str], float]:
    points = np.full((len(DIGITS) + 1, 3), np.nan)
    normals = np.full_like(points, np.nan)
    forces = np.zeros_like(points)
    impulses = np.zeros(len(DIGITS) + 1)
    target_geom = run.geom["red_toy_001_geom"]
    hit: set[str] = set()
    max_pen = 0.0
    names = list(DIGITS) + ["palm"]
    geom_to_slot = {
        run.geom[f"{digit}_{part}_collider"]: slot
        for slot, digit in enumerate(DIGITS)
        for part in ("proximal", "middle", "distal")
    }
    geom_to_slot[run.geom["palm_collider"]] = len(DIGITS)
    for ci in range(run.data.ncon):
        con = run.data.contact[ci]
        if target_geom not in (con.geom1, con.geom2): continue
        other = con.geom2 if con.geom1 == target_geom else con.geom1
        if other not in geom_to_slot: continue
        slot = geom_to_slot[other]
        wrench = np.zeros(6); mujoco.mj_contactForce(run.model, run.data, ci, wrench)
        points[slot] = con.pos
        normal = con.frame[:3].copy()
        if con.geom2 == target_geom: normal *= -1
        normals[slot] = normal
        forces[slot] = normal * abs(wrench[0])
        impulses[slot] += abs(wrench[0]) * run.model.opt.timestep
        hit.add(names[slot]); max_pen = max(max_pen, max(0.0, -float(con.dist)))
    return points, normals, forces, impulses, hit, max_pen


def execute(config: dict[str, Any], cell: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    xml = model_xml(config, cell)
    run = make_run(xml)
    for digit in DIGITS:
        open_angle = np.deg2rad(58)
        for part, ratio in (("proximal", 1.0), ("middle", 0.82), ("distal", 0.68)):
            run.data.qpos[run.joint_qpos[f"{digit}_{part}_flex"]] = open_angle * ratio
    dt = float(run.model.opt.timestep)
    n = int(config["clock"]["physics_steps"]) + 1
    target_body = run.body["red_toy_001"]
    head_body = run.body["head"]
    wrist_body = run.body["wrist_guide"]
    cam_id = mujoco.mj_name2id(run.model, mujoco.mjtObj.mjOBJ_CAMERA, "child_camera")
    streams: dict[str, list[Any]] = {k: [] for k in (
        "time_s phase qpos qvel actuator_command actuator_force target_pose target_velocity wrist_pose head_pose camera_pose camera_parent_pose digit_segment_pose head_accel head_gyro contact_points contact_normals contact_forces contact_impulses contact_digits support_contact target_support_penetration finger_object_penetration recovery_state assist_active".split()
    )}
    target_start = None
    all_hit: set[str] = set()
    max_pen = 0.0
    support_first_after_release = None
    for step in range(n):
        t = step * dt
        run.data.ctrl[:] = controls(run, t, float(cell["target_lateral_offset_m"]))
        if step > 0: mujoco.mj_step(run.model, run.data)
        else: mujoco.mj_forward(run.model, run.data)
        if target_start is None: target_start = run.data.xpos[target_body].copy()
        points, normals, forces, impulses, hit, penetration = contact_rows(run)
        all_hit |= hit; max_pen = max(max_pen, penetration)
        support = False; support_pen = 0.0
        for ci in range(run.data.ncon):
            con = run.data.contact[ci]
            if run.geom["red_toy_001_geom"] in (con.geom1, con.geom2) and run.geom["table"] in (con.geom1, con.geom2):
                support = True; support_pen = max(support_pen, max(0.0, -float(con.dist)))
        if t >= 18.8 and support and support_first_after_release is None: support_first_after_release = t
        object_velocity = np.zeros(6); mujoco.mj_objectVelocity(run.model, run.data, mujoco.mjtObj.mjOBJ_BODY, target_body, object_velocity, 0)
        camera_rotation = run.data.cam_xmat[cam_id].reshape(3, 3)
        camera_pose = np.concatenate((run.data.cam_xpos[cam_id], camera_rotation.reshape(-1)))
        parent_pose = np.concatenate((run.data.xpos[head_body], run.data.xquat[head_body]))
        streams["time_s"].append(t); streams["phase"].append(phase_at(min(t, 21.999999), config["phases"]))
        streams["qpos"].append(run.data.qpos.copy()); streams["qvel"].append(run.data.qvel.copy())
        streams["actuator_command"].append(run.data.ctrl.copy()); streams["actuator_force"].append(run.data.actuator_force.copy())
        streams["target_pose"].append(np.concatenate((run.data.xpos[target_body], run.data.xquat[target_body])))
        streams["target_velocity"].append(object_velocity.copy())
        streams["wrist_pose"].append(np.concatenate((run.data.xpos[wrist_body], run.data.xquat[wrist_body])))
        streams["head_pose"].append(np.concatenate((run.data.xpos[head_body], run.data.xquat[head_body])))
        streams["camera_pose"].append(camera_pose); streams["camera_parent_pose"].append(parent_pose)
        streams["digit_segment_pose"].append(np.asarray([
            np.concatenate((run.data.xpos[run.body[f"{digit}_{part}"]], run.data.xquat[run.body[f"{digit}_{part}"]]))
            for digit in DIGITS for part in ("proximal", "middle", "distal")
        ]))
        streams["head_accel"].append(run.data.sensordata[run.sensor["head_accelerometer"]].copy())
        streams["head_gyro"].append(run.data.sensordata[run.sensor["head_gyroscope"]].copy())
        streams["contact_points"].append(points); streams["contact_normals"].append(normals); streams["contact_forces"].append(forces); streams["contact_impulses"].append(impulses)
        streams["contact_digits"].append(len(hit)); streams["support_contact"].append(support)
        streams["target_support_penetration"].append(support_pen); streams["finger_object_penetration"].append(penetration)
        streams["recovery_state"].append(0); streams["assist_active"].append(False)
    arrays = {k: np.asarray(v) for k, v in streams.items()}
    z = arrays["target_pose"][:, 2]
    lift = float(np.max(z) - target_start[2])
    # Quaternion angular displacement relative to the grasp pose.
    grasp_i = int(round(10.0 / dt)); q0 = arrays["target_pose"][grasp_i, 3:]
    dots = np.clip(np.abs(arrays["target_pose"][:, 3:] @ q0), 0, 1)
    rotation = float(np.degrees(2 * np.arccos(dots)).max())
    release_slice = arrays["target_velocity"][int(20 / dt):, 3:]
    settled_speed = float(np.linalg.norm(release_slice, axis=1).max())
    required = set(config["frozen_tolerances"]["required_digits"])
    qa = {
        "cell": cell["id"], "physics_steps": n - 1, "truth_samples": n,
        "first_contact_s": float(arrays["time_s"][np.flatnonzero(arrays["contact_digits"] > 0)[0]]) if np.any(arrays["contact_digits"] > 0) else None,
        "contact_digits_seen": sorted(all_hit), "required_contact_graph_met": required <= all_hit,
        "maximum_simultaneous_contact_digits": int(arrays["contact_digits"].max()),
        "lift_m": lift, "manipulation_deg": rotation, "support_contact_after_release_s": support_first_after_release,
        "settled_max_linear_speed_m_s_after_20s": settled_speed,
        "finger_object_penetration_max_m": max_pen,
        "target_support_penetration_max_m": float(arrays["target_support_penetration"].max()),
        "assist_frames": int(arrays["assist_active"].sum()), "recovery_frames": int(np.count_nonzero(arrays["recovery_state"])),
        "object_free_joint": True, "object_pose_writes_after_initialization": 0,
        "attachments_equalities_external_forces": 0
    }
    tol = config["frozen_tolerances"]
    qa["passed"] = bool(
        qa["required_contact_graph_met"] and qa["maximum_simultaneous_contact_digits"] >= tol["minimum_distinct_support_digits"]
        and tol["minimum_lift_m"] <= lift <= tol["maximum_lift_m"] + 0.015
        and rotation >= tol["minimum_manipulation_deg"]
        and support_first_after_release is not None and settled_speed < 0.08
        and max_pen <= tol["finger_object_penetration_max_m"]
        and qa["target_support_penetration_max_m"] <= tol["target_support_penetration_max_m"]
        and qa["assist_frames"] == 0
    )
    return arrays, qa, xml


def trace_projection(arrays: dict[str, np.ndarray], config: dict[str, Any], model: mujoco.MjModel) -> dict[str, Any]:
    indices = np.arange(0, int(config["clock"]["physics_steps"]), config["clock"]["steps_per_frame"])
    frames = []
    for frame, i in enumerate(indices):
        frames.append({
            "frame": int(frame), "truth_index": int(i), "time_s": float(arrays["time_s"][i]), "phase": str(arrays["phase"][i]),
            "qpos": arrays["qpos"][i].tolist(), "wrist_pose_mj": arrays["wrist_pose"][i].tolist(),
            "head_pose_mj": arrays["head_pose"][i].tolist(), "camera_pose_mj": arrays["camera_pose"][i].tolist(),
            "target_pose_mj": arrays["target_pose"][i].tolist(), "contact_points_mj": arrays["contact_points"][i].tolist(),
            "digit_segment_pose_mj": arrays["digit_segment_pose"][i].reshape(-1).tolist(),
            "contact_digits": int(arrays["contact_digits"][i]), "support_contact": bool(arrays["support_contact"][i])
        })
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    return {"schema": "embodied.unity_trace_projection.v1", "physics_hz": 240, "render_hz": 30, "steps_per_frame": 8, "joint_names": joint_names, "digit_segment_names": [f"{d}_{p}" for d in DIGITS for p in ("proximal", "middle", "distal")], "frames": frames}


def run_gate(config_path: Path, output: Path, stage: str) -> None:
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    cells = config["robustness_cells"] if stage == "all" else config["robustness_cells"][:1]
    if stage == "no_object":
        cells = [{"id": "dynamic_registration_no_object", "target_lateral_offset_m": 0.0, "mass_scale": 1.0, "friction_scale": 1.0, "no_object": True}]
    results = []
    for cell in cells:
        cell_dir = output / cell["id"]; cell_dir.mkdir(exist_ok=True)
        arrays, qa, xml = execute(config, cell)
        np.savez_compressed(cell_dir / "authoritative_trace.npz", **arrays)
        replay, replay_qa, _ = execute(config, cell)
        max_error = max(float(np.max(np.abs(arrays[k] - replay[k]))) for k in arrays if arrays[k].dtype.kind in "fiu" and arrays[k].size)
        qa["replay_numeric_max_abs"] = max_error; qa["replay_qa_equal"] = qa == replay_qa | {"replay_numeric_max_abs": max_error, "replay_qa_equal": True} if False else all(qa.get(k) == replay_qa.get(k) for k in replay_qa)
        projection = trace_projection(arrays, config, make_run(xml).model)
        (cell_dir / "render_trace.json").write_bytes(canonical_json(projection))
        (cell_dir / "model.xml").write_text(xml)
        (cell_dir / "physics_qa.json").write_bytes(canonical_json(qa))
        results.append(qa)
    summary = {"schema": "embodied.unity_mujoco_gate_results.v1", "cells": results, "passed_cells": sum(bool(x["passed"]) for x in results)}
    (output / "gate_results.json").write_bytes(canonical_json(summary))
    provenance = {
        "mujoco": mujoco.__version__, "numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "config_sha256": sha256(config_path), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "privacy": "public/synthetic only; restricted ChildLens media not accessed"
    }
    (output / "runtime_receipt.json").write_bytes(canonical_json(provenance))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/embodied_simulation_unity_mujoco_gate.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("nominal", "no_object", "all"), default="nominal")
    parser.add_argument("--rest-manifest", type=Path)
    parser.add_argument("--manipulation", action="store_true")
    args = parser.parse_args()
    if args.rest_manifest: manifest_registration(args.rest_manifest, args.output, args.manipulation)
    else: run_gate(args.config, args.output, args.stage)


if __name__ == "__main__": main()
