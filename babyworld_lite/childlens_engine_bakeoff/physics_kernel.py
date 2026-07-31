"""Spec-driven causal MuJoCo kernel for object-centered reach/manipulation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import mujoco
import numpy as np

from .controllers import (
    ContactTriggeredGrasp,
    GraspState,
    activate_weld_preserving_current_pose,
    minimum_hand_target_distance,
)


def value(spec: dict[str, Any], key: str) -> Any:
    return spec[key]["value"]


def _hand_geom_names(mimo_assets: Path) -> tuple[str, ...]:
    root = ElementTree.parse(mimo_assets / "mimo" / "MIMo_modelv2.xml").getroot()
    tokens = ("hand", "ff", "mf", "rf", "lf", "th")
    return tuple(
        sorted(
            geom.get("name")
            for geom in root.iter("geom")
            if (geom.get("name") or "").startswith("geom:right_")
            and any(token in (geom.get("name") or "") for token in tokens)
        )
    )


def _component_xml(
    mimo_assets: Path,
    root_xy: tuple[float, float],
    target_definition: dict[str, Any],
) -> str:
    contact_pairs = "\n".join(
        f'    <pair geom1="{name}" geom2="target_geom" condim="3"/>'
        for name in _hand_geom_names(mimo_assets)
    )
    geometry = target_definition["geometry"]
    rgba = " ".join(str(value) for value in target_definition["rgba"])
    if geometry == "cylinder_with_three_capsule_handle":
        target_geoms = f"""
      <geom name="target_geom" type="cylinder" size="0.04 0.05"
            mass="0.30" rgba="{rgba}"
            solref="0.03 1" solimp="0.9 0.95 0.001"
            friction="1 0.005 0.0002" contype="16" conaffinity="16"/>
      <geom name="target_handle_upper" type="capsule" size="0.007"
            fromto="0 -0.025 0.025 0 -0.065 0.025" rgba="{rgba}"
            contype="0" conaffinity="0" mass="0.002"/>
      <geom name="target_handle_outer" type="capsule" size="0.007"
            fromto="0 -0.065 -0.025 0 -0.065 0.025" rgba="{rgba}"
            contype="0" conaffinity="0" mass="0.002"/>
      <geom name="target_handle_lower" type="capsule" size="0.007"
            fromto="0 -0.025 -0.025 0 -0.065 -0.025" rgba="{rgba}"
            contype="0" conaffinity="0" mass="0.002"/>"""
    elif geometry == "sphere":
        target_geoms = f"""
      <geom name="target_geom" type="sphere" size="0.05"
            mass="0.30" rgba="{rgba}"
            solref="0.03 1" solimp="0.9 0.95 0.001"
            friction="1 0.005 0.0002" contype="16" conaffinity="16"/>"""
    else:
        raise ValueError(f"unsupported target geometry: {geometry}")
    return f"""<mujoco model="SpecDrivenMIMoKernel">
  <compiler inertiafromgeom="true" angle="degree" assetdir="{mimo_assets}"/>
  <include file="{mimo_assets / "mimo" / "MIMo_metav2.xml"}"/>
  <worldbody>
    <body name="mimo_location" pos="{root_xy[0]} {root_xy[1]} 0.55" euler="0 0 180">
      <include file="{mimo_assets / "mimo" / "MIMo_modelv2.xml"}"/>
    </body>
    <body name="hand_mocap" mocap="true" pos="{root_xy[0]} {root_xy[1] + 0.13} 0.59"
          quat="0 0 0.996195 0.087156">
      <geom type="sphere" size="0.006" rgba="0 1 1 0.2"
            contype="0" conaffinity="0"/>
    </body>
    <body name="grasp_mocap" mocap="true"
          pos="{root_xy[0]} {root_xy[1] + 0.13} 0.59"
          quat="0 0 0.996195 0.087156"/>
    <body name="target" pos="{root_xy[0] + 0.23} {root_xy[1] + 0.13} 0.59">
      <joint name="target_free" type="free" damping="0.5" armature="0.001"/>
{target_geoms}
    </body>
    <body name="support" pos="{root_xy[0] + 0.28} {root_xy[1] + 0.13} 0.515">
      <geom name="support_geom" type="box" size="0.32 0.22 0.015"
            rgba="0.34 0.19 0.08 1" friction="1 0.005 0.0002"
            contype="16" conaffinity="16"/>
    </body>
    <camera name="chest_camera" pos="{root_xy[0] + 0.06} {root_xy[1] + 0.04} 0.88"
            xyaxes="0 -1 0 0.45 0 0.89" fovy="90"/>
  </worldbody>
  <equality>
    <weld name="right_hand_mocap_weld" body1="hand_mocap" body2="right_hand"
          solref="0.05 1"/>
    <weld name="target_grasp_weld" body1="grasp_mocap" body2="target"
          active="false" solref="0.05 1" torquescale="0.04"/>
  </equality>
  <contact>
{contact_pairs}
  </contact>
</mujoco>
"""


@dataclass
class KernelModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    hand_geom_ids: tuple[int, ...]
    target_geom_id: int
    target_body_id: int
    right_hand_body_id: int
    mocap_id: int
    grasp_mocap_id: int
    grasp_equality_id: int
    camera_id: int
    sensor_slices: dict[str, slice]
    hand_target_collision_bit: int
    target_support_collision_bit: int
    mimo_body_ids: tuple[int, ...]
    mimo_body_names: tuple[str, ...]


def build_kernel_model(
    scene_path: Path,
    mimo_assets: Path,
    component_path: Path,
    *,
    root_xy: tuple[float, float],
    target_definition: dict[str, Any],
) -> KernelModel:
    component_path.write_text(
        _component_xml(mimo_assets, root_xy, target_definition), encoding="utf-8"
    )
    parent = mujoco.MjSpec.from_file(str(scene_path.resolve()))
    child = mujoco.MjSpec.from_file(str(component_path.resolve()))
    frame = parent.worldbody.add_frame()
    frame.name = "spec_kernel_attach"
    parent.attach(child, prefix="kernel_", frame=frame)
    model = parent.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    hand_geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith("kernel_geom:right_")
        and any(
            token in (model.geom(geom_id).name or "")
            for token in ("hand", "ff", "mf", "rf", "lf", "th")
        )
    )
    if not hand_geom_ids:
        raise RuntimeError("MIMo right-hand collision geoms were not found")
    mimo_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith("kernel_geom:")
    ]
    model.geom_contype[mimo_geom_ids] = 0
    model.geom_conaffinity[mimo_geom_ids] = 0
    target_geom_id = model.geom("kernel_target_geom").id
    support_geom_id = model.geom("kernel_support_geom").id
    kernel_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if (model.body(int(model.geom_bodyid[geom_id])).name or "").startswith(
            "kernel_"
        )
    }
    used_bits = 0
    for geom_id in range(model.ngeom):
        if geom_id not in kernel_geom_ids:
            used_bits |= int(model.geom_contype[geom_id])
            used_bits |= int(model.geom_conaffinity[geom_id])
    free_bits = [1 << bit for bit in range(30) if not used_bits & (1 << bit)]
    if len(free_bits) < 2:
        raise RuntimeError("compiled scene has fewer than two free collision bits")
    hand_target_bit, target_support_bit = free_bits[:2]
    model.geom_contype[list(hand_geom_ids)] = hand_target_bit
    model.geom_conaffinity[list(hand_geom_ids)] = hand_target_bit
    model.geom_contype[target_geom_id] = hand_target_bit | target_support_bit
    model.geom_conaffinity[target_geom_id] = hand_target_bit | target_support_bit
    model.geom_contype[support_geom_id] = target_support_bit
    model.geom_conaffinity[support_geom_id] = target_support_bit
    for body_id in range(model.nbody):
        body_name = model.body(body_id).name or ""
        if body_name.startswith("kernel_") and body_name not in {
            "kernel_target",
            "kernel_support",
        }:
            model.body_gravcomp[body_id] = 1.0

    def sensor_slice(name: str, width: int) -> slice:
        sensor_id = model.sensor(name).id
        start = int(model.sensor_adr[sensor_id])
        return slice(start, start + width)

    mimo_body_ids = tuple(
        body_id
        for body_id in range(model.nbody)
        if (model.body(body_id).name or "").startswith("kernel_")
        and body_id
        not in {
            model.body("kernel_target").id,
            model.body("kernel_support").id,
            model.body("kernel_hand_mocap").id,
            model.body("kernel_grasp_mocap").id,
        }
    )
    return KernelModel(
        model=model,
        data=data,
        hand_geom_ids=hand_geom_ids,
        target_geom_id=target_geom_id,
        target_body_id=model.body("kernel_target").id,
        right_hand_body_id=model.body("kernel_right_hand").id,
        mocap_id=int(model.body_mocapid[model.body("kernel_hand_mocap").id]),
        grasp_mocap_id=int(
            model.body_mocapid[model.body("kernel_grasp_mocap").id]
        ),
        grasp_equality_id=model.equality("kernel_target_grasp_weld").id,
        camera_id=model.camera("kernel_chest_camera").id,
        sensor_slices={
            "vestibular_accelerometer": sensor_slice("kernel_vestibular_acc", 3),
            "vestibular_gyroscope": sensor_slice("kernel_vestibular_gyro", 3),
        },
        hand_target_collision_bit=hand_target_bit,
        target_support_collision_bit=target_support_bit,
        mimo_body_ids=mimo_body_ids,
        mimo_body_names=tuple(model.body(body_id).name for body_id in mimo_body_ids),
    )


def phase_at(spec: dict[str, Any], time_s: float) -> dict[str, Any]:
    phases = value(spec, "phase_timestamps")
    for phase in phases:
        if phase["start_s"] <= time_s < phase["end_s"]:
            return phase
    return phases[-1]


def interpolate(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return start + alpha * (end - start)


def contact_sample(kernel: KernelModel) -> tuple[np.ndarray, int]:
    wrench = np.zeros(6, dtype=np.float64)
    count = 0
    for contact_index in range(kernel.data.ncon):
        contact = kernel.data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if kernel.target_geom_id in pair and pair.intersection(kernel.hand_geom_ids):
            local = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(kernel.model, kernel.data, contact_index, local)
            wrench += local
            count += 1
    return wrench, count


def _aim_camera(
    model: mujoco.MjModel,
    camera_id: int,
    position: np.ndarray,
    target: np.ndarray,
) -> None:
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray([0.0, 0.0, 1.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(
        quaternion, np.column_stack([right, up, backward]).ravel()
    )
    model.cam_pos[camera_id] = position
    model.cam_quat[camera_id] = quaternion


def _conditioned_camera_offset(
    base_offset: np.ndarray, *, episode_time: float, seed: int
) -> np.ndarray:
    """Deterministic one-second head-motion nuisance with four larger turns."""
    second = min(29, max(0, int(episode_time)))
    turns = (0.0, 78.0, -68.0, 86.0, -76.0)
    turn_index = sum(second >= boundary for boundary in (6, 12, 18, 24))
    rng = np.random.default_rng(seed + second * 7919)
    angle = np.deg2rad(turns[turn_index] + float(rng.uniform(-28.0, 28.0)))
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    rotated = np.asarray(
        [
            cosine * base_offset[0] - sine * base_offset[1],
            sine * base_offset[0] + cosine * base_offset[1],
            base_offset[2] + float(rng.uniform(-0.025, 0.025)),
        ],
        dtype=np.float64,
    )
    return rotated


def _desired_hand_position(
    phase: dict[str, Any],
    *,
    initial_hand: np.ndarray,
    target_anchor: np.ndarray,
    grasp_origin: np.ndarray,
    controller: dict[str, Any],
) -> np.ndarray:
    duration = phase["end_s"] - phase["start_s"]
    alpha = 0.0 if duration <= 0 else (phase["_time_s"] - phase["start_s"]) / duration
    kind = phase["phase"]
    miss_start = target_anchor + np.asarray([-0.11, 0.12, 0.02])
    miss_end = target_anchor + np.asarray([0.11, 0.12, 0.02])
    if kind == "look_settle":
        return initial_hand
    if kind == "reach":
        lateral = initial_hand + np.asarray([0.0, 0.10, 0.0])
        if alpha < 0.45:
            return interpolate(initial_hand, lateral, alpha / 0.45)
        return interpolate(lateral, miss_start, (alpha - 0.45) / 0.55)
    if kind == "near_miss":
        return interpolate(miss_start, miss_end, alpha)
    if kind == "touch_push":
        if alpha < 0.35:
            return interpolate(
                miss_end,
                target_anchor + np.asarray([-0.10, 0.0, 0.02]),
                alpha / 0.35,
            )
        return interpolate(
            target_anchor + np.asarray([-0.10, 0.0, 0.02]),
            target_anchor + np.asarray([0.15, 0.0, -0.04]),
            (alpha - 0.35) / 0.65,
        )
    if kind == "post_contact_grasp":
        return interpolate(
            target_anchor + np.asarray([0.10, 0.0, -0.04]),
            target_anchor + np.asarray([0.18, 0.0, -0.04]),
            alpha,
        )
    if kind == "lift_place":
        return grasp_origin + np.asarray(
            [
                controller["transport_delta_m"] * alpha,
                0.0,
                controller["lift_delta_m"] * min(alpha / 0.55, 1.0),
            ]
        )
    if kind == "release_drop":
        if alpha < 0.35:
            return grasp_origin + np.asarray(
                [
                    controller["transport_delta_m"],
                    0.0,
                    controller["lift_delta_m"],
                ]
            )
        release_alpha = (alpha - 0.35) / 0.65
        return grasp_origin + np.asarray(
            [
                controller["transport_delta_m"],
                -0.12 * release_alpha,
                controller["lift_delta_m"] + 0.15 * release_alpha,
            ]
        )
    return grasp_origin + np.asarray(
        [
            controller["transport_delta_m"],
            -0.12,
            controller["lift_delta_m"] + 0.15,
        ]
    )


def run_physics_trace(
    kernel: KernelModel,
    spec: dict[str, Any],
    *,
    fps: int = 30,
    settle_steps: int = 1500,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run the resolved episode on a shared clock without appearance rendering."""
    model, data = kernel.model, kernel.data
    controller = value(spec, "controller")
    camera = value(spec, "camera")
    substeps = int(controller["control_substeps"])
    model.opt.timestep = 1.0 / (fps * substeps)
    data.eq_active[kernel.grasp_equality_id] = 0
    model.geom_contype[kernel.target_geom_id] = 0
    model.geom_conaffinity[kernel.target_geom_id] = 0
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    prestage_start = data.mocap_pos[kernel.mocap_id].copy()
    prestage_end = prestage_start + np.asarray([-0.10, -0.08, 0.0])
    for step in range(500):
        data.mocap_pos[kernel.mocap_id] = interpolate(
            prestage_start, prestage_end, (step + 1) / 500
        )
        mujoco.mj_step(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    data.time = 0.0
    initial_hand = data.xpos[kernel.right_hand_body_id].copy()
    initial_mocap = data.mocap_pos[kernel.mocap_id].copy()
    target_anchor = initial_hand + np.asarray(
        controller["target_anchor_from_hand_m"], dtype=np.float64
    )
    target_mocap = initial_mocap + (target_anchor - initial_hand)
    target_joint_id = model.joint("kernel_target_free").id
    target_qpos_adr = int(model.jnt_qposadr[target_joint_id])
    target_dof_adr = int(model.jnt_dofadr[target_joint_id])
    data.qpos[target_qpos_adr : target_qpos_adr + 3] = target_anchor
    data.qpos[target_qpos_adr + 3 : target_qpos_adr + 7] = (1, 0, 0, 0)
    data.qvel[target_dof_adr : target_dof_adr + 6] = 0
    support_body_id = model.body("kernel_support").id
    model.body_pos[support_body_id] = target_anchor + np.asarray(
        [controller["transport_delta_m"] * 0.5, 0.0, -0.065]
    )
    model.geom_contype[kernel.target_geom_id] = (
        kernel.hand_target_collision_bit | kernel.target_support_collision_bit
    )
    model.geom_conaffinity[kernel.target_geom_id] = (
        kernel.hand_target_collision_bit | kernel.target_support_collision_bit
    )
    camera_position = target_anchor + np.asarray(
        camera["position_offset_m"], dtype=np.float64
    )
    _aim_camera(
        model,
        kernel.camera_id,
        camera_position,
        target_anchor
        + np.asarray(
            [
                controller["transport_delta_m"]
                * camera["look_transport_fraction"],
                camera["look_offset_y_m"],
                camera["look_offset_z_m"],
            ]
        ),
    )
    mujoco.mj_forward(model, data)
    initial_target = data.xpos[kernel.target_body_id].copy()
    grasp = ContactTriggeredGrasp()
    grasp_origin = target_mocap + np.asarray([0.06, 0.0, 0.02])
    pose_jump_m = None
    release_done = False
    first_contact_time_s = None
    first_contact_pairs: list[list[str]] = []
    initial_separation_m = minimum_hand_target_distance(
        model, data, kernel.hand_geom_ids, kernel.target_geom_id
    ).signed_distance_m
    maximum_target_z = float(initial_target[2])
    maximum_contact_force_n = 0.0
    trace: dict[str, list[Any]] = {
        "time_s": [],
        "phase": [],
        "qpos": [],
        "qvel": [],
        "action": [],
        "touch_wrench": [],
        "touch_contact_count": [],
        "grasp_active": [],
        "vestibular_accelerometer": [],
        "vestibular_gyroscope": [],
        "target_pose": [],
        "hand_pose": [],
        "camera_pose": [],
        "body_pose": [],
    }
    substep_time: list[float] = []
    substep_separation: list[float] = []
    substep_contact_count: list[int] = []
    minimum_sample = None
    contacts_at_minimum: list[list[str]] = []

    def record(action: np.ndarray, wrench: np.ndarray, contacts: int) -> None:
        phase = phase_at(spec, float(data.time))["phase"]
        trace["time_s"].append(float(data.time))
        trace["phase"].append(phase)
        trace["qpos"].append(data.qpos.copy())
        trace["qvel"].append(data.qvel.copy())
        trace["action"].append(action.copy())
        trace["touch_wrench"].append(wrench.copy())
        trace["touch_contact_count"].append(contacts)
        trace["grasp_active"].append(bool(data.eq_active[kernel.grasp_equality_id]))
        for name, sensor_slice in kernel.sensor_slices.items():
            trace[name].append(data.sensordata[sensor_slice].copy())
        trace["target_pose"].append(
            np.concatenate(
                [data.xpos[kernel.target_body_id], data.xquat[kernel.target_body_id]]
            )
        )
        trace["hand_pose"].append(
            np.concatenate(
                [
                    data.xpos[kernel.right_hand_body_id],
                    data.xquat[kernel.right_hand_body_id],
                ]
            )
        )
        trace["camera_pose"].append(
            np.concatenate(
                [
                    data.cam_xpos[kernel.camera_id],
                    data.cam_xmat[kernel.camera_id].reshape(9),
                ]
            )
        )
        trace["body_pose"].append(
            np.concatenate(
                [
                    data.xpos[list(kernel.mimo_body_ids)],
                    data.xquat[list(kernel.mimo_body_ids)],
                ],
                axis=1,
            )
        )

    record(np.zeros(3), np.zeros(6), 0)
    frames = int(round(value(spec, "duration_s") * fps))
    started = time.perf_counter()
    for frame_index in range(frames):
        episode_time = frame_index / fps
        phase = dict(phase_at(spec, episode_time))
        phase["_time_s"] = episode_time
        desired = _desired_hand_position(
            phase,
            initial_hand=initial_mocap,
            target_anchor=target_mocap,
            grasp_origin=grasp_origin,
            controller=controller,
        )
        previous = data.mocap_pos[kernel.mocap_id].copy()
        if (
            phase["phase"] == "post_contact_grasp"
            and grasp.state is GraspState.APPROACH
        ):
            desired = previous + np.asarray([0.001, 0.0, 0.0])
        elif grasp.state is GraspState.CONTACT_SEEN:
            desired = previous
        frame_wrench = np.zeros(6)
        frame_contacts = 0
        for substep in range(substeps):
            fraction = (substep + 1) / substeps
            data.mocap_pos[kernel.mocap_id] = interpolate(
                previous, desired, fraction
            )
            if grasp.state is GraspState.GRASP_ACTIVE:
                data.mocap_pos[kernel.grasp_mocap_id] = data.xpos[
                    kernel.right_hand_body_id
                ]
                data.mocap_quat[kernel.grasp_mocap_id] = data.xquat[
                    kernel.right_hand_body_id
                ]
            mujoco.mj_step(model, data)
            wrench, contacts = contact_sample(kernel)
            maximum_contact_force_n = max(
                maximum_contact_force_n, float(np.linalg.norm(wrench[:3]))
            )
            frame_wrench += wrench
            frame_contacts += contacts
            separation = minimum_hand_target_distance(
                model,
                data,
                kernel.hand_geom_ids,
                kernel.target_geom_id,
            )
            if (
                minimum_sample is None
                or separation.signed_distance_m < minimum_sample.signed_distance_m
            ):
                minimum_sample = separation
                contacts_at_minimum = [
                    [
                        model.geom(int(data.contact[index].geom1)).name,
                        model.geom(int(data.contact[index].geom2)).name,
                    ]
                    for index in range(data.ncon)
                    if kernel.target_geom_id
                    in {
                        int(data.contact[index].geom1),
                        int(data.contact[index].geom2),
                    }
                ]
            substep_time.append(float(data.time))
            substep_separation.append(separation.signed_distance_m)
            substep_contact_count.append(contacts)
            maximum_target_z = max(
                maximum_target_z, float(data.xpos[kernel.target_body_id, 2])
            )
            if contacts and first_contact_time_s is None:
                first_contact_time_s = float(data.time)
                for contact_index in range(data.ncon):
                    contact = data.contact[contact_index]
                    pair = {int(contact.geom1), int(contact.geom2)}
                    if kernel.target_geom_id in pair and pair.intersection(
                        kernel.hand_geom_ids
                    ):
                        first_contact_pairs.append(
                            [
                                model.geom(int(contact.geom1)).name,
                                model.geom(int(contact.geom2)).name,
                            ]
                        )
            grasp.observe_contact(time_s=float(data.time), contact_count=contacts)
            if (
                phase["phase"] == "post_contact_grasp"
                and grasp.state is GraspState.CONTACT_SEEN
            ):
                data.mocap_pos[kernel.grasp_mocap_id] = data.xpos[
                    kernel.right_hand_body_id
                ]
                data.mocap_quat[kernel.grasp_mocap_id] = data.xquat[
                    kernel.right_hand_body_id
                ]
                pose_jump_m = activate_weld_preserving_current_pose(
                    model, data, kernel.grasp_equality_id
                )
                model.geom_contype[list(kernel.hand_geom_ids)] = 0
                model.geom_conaffinity[list(kernel.hand_geom_ids)] = 0
                grasp.engage(time_s=float(data.time))
                grasp_origin = data.mocap_pos[kernel.mocap_id].copy()
            if (
                phase["phase"] == "release_drop"
                and grasp.state is GraspState.GRASP_ACTIVE
                and not release_done
                and (episode_time - phase["start_s"])
                / (phase["end_s"] - phase["start_s"])
                >= 0.35
            ):
                data.eq_active[kernel.grasp_equality_id] = 0
                grasp.release(time_s=float(data.time))
                release_done = True
        if camera.get("follow_target", False):
            followed_target = data.xpos[kernel.target_body_id].copy()
            camera_offset = np.asarray(
                camera["position_offset_m"], dtype=np.float64
            )
            if camera.get("conditioned_head_motion", False):
                camera_offset = _conditioned_camera_offset(
                    camera_offset,
                    episode_time=episode_time,
                    seed=int(value(spec, "seed")),
                )
            _aim_camera(
                model,
                kernel.camera_id,
                followed_target + camera_offset,
                followed_target
                + np.asarray(
                    [
                        0.0,
                        camera["look_offset_y_m"],
                        camera["look_offset_z_m"],
                    ]
                ),
            )
            mujoco.mj_forward(model, data)
        record(desired - previous, frame_wrench, frame_contacts)
    wall_seconds = time.perf_counter() - started
    arrays = {key: np.asarray(items) for key, items in trace.items()}
    arrays["substep_time_s"] = np.asarray(substep_time)
    arrays["substep_signed_separation_m"] = np.asarray(substep_separation)
    arrays["substep_contact_count"] = np.asarray(substep_contact_count)
    target_xyz = arrays["target_pose"][:, :3]
    target_linear_speed = np.linalg.norm(
        arrays["qvel"][:, target_dof_adr : target_dof_adr + 3], axis=1
    )
    release_index = (
        int(np.searchsorted(arrays["time_s"], grasp.release_time_s))
        if grasp.release_time_s is not None
        else len(target_xyz) - 1
    )
    final_speed = float(np.linalg.norm(data.qvel[target_dof_adr : target_dof_adr + 3]))
    contact_frame = (
        int(np.searchsorted(arrays["time_s"], first_contact_time_s))
        if first_contact_time_s is not None
        else 0
    )
    engagement_frame = (
        int(np.searchsorted(arrays["time_s"], grasp.engagement_time_s))
        if grasp.engagement_time_s is not None
        else contact_frame
    )
    stability_start_s = max(
        float(arrays["time_s"][-1]) - float(controller["post_release_settle_s"]),
        float(grasp.release_time_s or 0.0),
    )
    stability_mask = arrays["time_s"] >= stability_start_s
    near_miss_phase = value(spec, "phase_timestamps")[2]
    near_mask = (
        (arrays["substep_time_s"] >= near_miss_phase["start_s"])
        & (arrays["substep_time_s"] < near_miss_phase["end_s"])
    )
    phase_minimum_separation = {}
    for phase_definition in value(spec, "phase_timestamps"):
        mask = (
            (arrays["substep_time_s"] >= phase_definition["start_s"])
            & (arrays["substep_time_s"] < phase_definition["end_s"])
        )
        phase_minimum_separation[phase_definition["phase"]] = (
            float(np.min(arrays["substep_signed_separation_m"][mask]))
            if np.any(mask)
            else None
        )
    receipt = {
        "schema": "QAReport.physics",
        "frames": frames,
        "substeps": int(len(substep_time)),
        "wall_seconds": wall_seconds,
        "simulation_frames_per_wall_second": frames / wall_seconds,
        "first_contact_time_s": first_contact_time_s,
        "first_contact_geom_pairs": first_contact_pairs,
        "maximum_hand_target_contact_force_n": maximum_contact_force_n,
        "initial_minimum_signed_separation_m": initial_separation_m,
        "grasp": grasp.receipt(),
        "grasp_activation_pose_jump_m": pose_jump_m,
        "near_miss_minimum_signed_separation_m": float(
            np.min(arrays["substep_signed_separation_m"][near_mask])
        ),
        "near_miss_contact_substeps": int(
            np.count_nonzero(arrays["substep_contact_count"][near_mask])
        ),
        "phase_minimum_signed_separation_m": phase_minimum_separation,
        "maximum_lift_from_initial_m": maximum_target_z - float(initial_target[2]),
        "contact_driven_pregrasp_displacement_m": float(
            np.linalg.norm(
                target_xyz[engagement_frame] - target_xyz[contact_frame]
            )
        ),
        "horizontal_transport_m": float(
            np.linalg.norm(target_xyz[-1, :2] - initial_target[:2])
        ),
        "released": release_done,
        "post_release_minimum_height_m": float(np.min(target_xyz[release_index:, 2])),
        "final_target_speed_m_s": final_speed,
        "post_release_stability": {
            "window_start_s": stability_start_s,
            "window_duration_s": float(
                arrays["time_s"][-1] - stability_start_s
            ),
            "maximum_speed_m_s": float(np.max(target_linear_speed[stability_mask])),
            "mean_speed_m_s": float(np.mean(target_linear_speed[stability_mask])),
            "threshold_m_s": float(controller["post_release_max_speed_m_s"]),
            "passes": bool(
                np.max(target_linear_speed[stability_mask])
                <= float(controller["post_release_max_speed_m_s"])
            ),
        },
        "target_initial_xyz": initial_target.tolist(),
        "target_final_xyz": target_xyz[-1].tolist(),
        "direct_target_transform_after_initialization": False,
        "collision_layers": {
            "hand_target_bit": kernel.hand_target_collision_bit,
            "target_support_bit": kernel.target_support_collision_bit,
            "selection": "lowest two bits unused by compiled furnished scene",
        },
        "global_minimum_separation": {
            "signed_distance_m": minimum_sample.signed_distance_m,
            "hand_geom": model.geom(minimum_sample.hand_geom_id).name,
            "target_geom": model.geom(minimum_sample.target_geom_id).name,
            "target_contacts_at_sample": contacts_at_minimum,
            "hand_contype": int(model.geom_contype[minimum_sample.hand_geom_id]),
            "hand_conaffinity": int(
                model.geom_conaffinity[minimum_sample.hand_geom_id]
            ),
            "target_contype": int(model.geom_contype[kernel.target_geom_id]),
            "target_conaffinity": int(model.geom_conaffinity[kernel.target_geom_id]),
        },
    }
    return arrays, receipt
