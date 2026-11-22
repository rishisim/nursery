"""Direct MIMo-v2 + MolmoSpaces MJCF composition and causal push audition."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _component_xml(mimo_assets: Path, mode: str) -> str:
    meta = mimo_assets / "mimo" / "MIMo_metav2.xml"
    model = mimo_assets / "mimo" / "MIMo_modelv2.xml"
    target_joint = (
        '<freejoint name="audition_ball_free"/>'
        if mode == "lift"
        else """<joint name="audition_ball_slide" type="slide" axis="1 0 0" damping="0.2"
             limited="true" range="0 0.12"/>"""
    )
    target_geom = (
        """<geom name="audition_ball_geom" type="cylinder" size="0.04 0.06"
            mass="0.25" rgba="0.1 0.35 0.85 1" friction="1 0.005 0.0002"
            contype="16" conaffinity="16"/>"""
        if mode == "lift"
        else """<geom name="audition_ball_geom" type="sphere" size="0.05" mass="0.25"
            rgba="0.8 0.15 0.1 1" friction="1 0.005 0.0002"
            contype="16" conaffinity="16"/>"""
    )
    return f"""<mujoco model="MIMoHybridComponent">
  <compiler inertiafromgeom="true" angle="degree" assetdir="{mimo_assets}"/>
  <include file="{meta}"/>
  <worldbody>
    <body name="mimo_location" pos="0.0 0.6 0.55" euler="0 0 180">
      <include file="{model}"/>
    </body>
    <body name="hand_mocap" mocap="true" pos="-0.002 0.733978 0.593828"
          quat="0 0 0.996195 0.087156">
      <geom type="sphere" size="0.008" rgba="0 1 1 0.25" contype="0" conaffinity="0"/>
    </body>
    <body name="audition_ball" pos="0.23 0.734 0.594">
      {target_joint}
      {target_geom}
    </body>
    <body name="manual_workspace" pos="0.18 0.67 0.535">
      <geom name="manual_workspace_geom" type="box" size="0.30 0.24 0.015"
            rgba="0.42 0.24 0.10 1" friction="1 0.005 0.0002"
            contype="16" conaffinity="16"/>
    </body>
    <camera name="chest_camera" pos="0.10 0.67 0.74"
            xyaxes="0 -1 0 0.342 0 0.94" fovy="90"/>
  </worldbody>
  <equality>
    <weld name="right_hand_mocap_weld" body1="hand_mocap" body2="right_hand"
          solref="0.02 1"/>
  </equality>
</mujoco>
"""


def build_hybrid(
    scene_path: Path, mimo_assets: Path, component_path: Path, mode: str = "push"
) -> mujoco.MjModel:
    component_path.write_text(_component_xml(mimo_assets, mode), encoding="utf-8")
    parent = mujoco.MjSpec.from_file(str(scene_path.resolve()))
    child = mujoco.MjSpec.from_file(str(component_path.resolve()))
    attach_frame = parent.worldbody.add_frame()
    attach_frame.name = "mimo_attach"
    parent.attach(child, prefix="mimo_", frame=attach_frame)
    return parent.compile()


def _contact_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geom_id: int,
    hand_geom_ids: set[int],
) -> tuple[np.ndarray, int]:
    wrench = np.zeros(6, dtype=np.float64)
    hand_contacts = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if target_geom_id in pair and pair.intersection(hand_geom_ids):
            local_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, local_wrench)
            wrench += local_wrench
            hand_contacts += 1
    return wrench, hand_contacts


def _target_area(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    target_geom_id: int,
    scene_option: mujoco.MjvOption,
) -> float:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="mimo_chest_camera", scene_option=scene_option)
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    mask = (segmentation[..., 0] == target_geom_id) & (
        segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    )
    return float(mask.mean())


def run(
    scene_path: Path,
    mimo_assets: Path,
    output_dir: Path,
    *,
    seed: int = 20260725,
    mode: str = "push",
) -> dict:
    del seed
    if mode not in {"push", "near_miss", "lift"}:
        raise ValueError(f"unsupported mode: {mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    component_path = output_dir / "mimo_hybrid_component.xml"
    model = build_hybrid(scene_path, mimo_assets, component_path, mode)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "mimo_audition_ball"
    )
    target_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "mimo_audition_ball_geom"
    )
    hand_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith("mimo_geom:right_")
        and any(token in (model.geom(geom_id).name or "") for token in ("hand", "ff", "mf", "rf", "lf", "th"))
    }
    mimo_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith("mimo_geom:")
    ]
    model.geom_contype[mimo_geom_ids] = 0
    model.geom_conaffinity[mimo_geom_ids] = 0
    # A chest-mounted view cannot see the torso/head behind the camera. Hide those
    # geoms only from this renderer while retaining the complete physical model.
    visible_limb_geom_ids = {
        geom_id
        for geom_id in mimo_geom_ids
        if any(
            token in (model.geom(geom_id).name or "")
            for token in ("right_hand", "right_ff",
                          "right_mf", "right_rf", "right_lf", "right_th")
        )
    }
    model.geom_group[mimo_geom_ids] = 5
    model.geom_group[list(visible_limb_geom_ids)] = 0
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[5] = 0
    workspace_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "mimo_manual_workspace_geom"
    )
    interactive_geom_ids = list(hand_geom_ids) + [target_geom_id, workspace_geom_id]
    model.geom_contype[interactive_geom_ids] = 16
    model.geom_conaffinity[interactive_geom_ids] = 16
    for body_id in range(model.nbody):
        if (model.body(body_id).name or "").startswith("mimo_"):
            model.body_gravcomp[body_id] = 1.0
    acc_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "mimo_vestibular_acc"
    )
    gyro_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "mimo_vestibular_gyro"
    )
    acc_slice = slice(model.sensor_adr[acc_id], model.sensor_adr[acc_id] + 3)
    gyro_slice = slice(model.sensor_adr[gyro_id], model.sensor_adr[gyro_id] + 3)
    mocap_id = model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mimo_hand_mocap")
    ]
    initial_mocap = data.mocap_pos[mocap_id].copy()
    right_hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "mimo_right_hand"
    )
    chest_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mimo_chest")
    for _ in range(1500):
        mujoco.mj_step(model, data)
    settled_hand = data.xpos[right_hand_id].copy()
    if mode == "lift":
        initial_mocap = settled_hand.copy()
        data.mocap_pos[mocap_id] = initial_mocap
    target_lateral_offset = 0.12 if mode == "near_miss" else 0.0
    target_stage_position = settled_hand + np.asarray([0.10, target_lateral_offset, -0.04])
    if mode == "lift":
        target_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "mimo_audition_ball_free"
        )
        target_qpos_adr = model.jnt_qposadr[target_joint_id]
        data.qpos[target_qpos_adr : target_qpos_adr + 3] = target_stage_position
        data.qpos[target_qpos_adr + 3 : target_qpos_adr + 7] = [1, 0, 0, 0]
    else:
        model.body_pos[target_body_id] = target_stage_position
        target_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "mimo_audition_ball_slide"
        )
        data.qpos[model.jnt_qposadr[target_joint_id]] = 0.0
    workspace_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "mimo_manual_workspace"
    )
    model.body_pos[workspace_body_id] = settled_hand + np.asarray([0.15, 0.0, -0.109])
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "mimo_chest_camera"
    )
    settled_chest = data.xpos[chest_id].copy()
    camera_offset = {
        "push": np.asarray([-0.03, -0.03, 0.10]),
        "near_miss": np.asarray([-0.02, 0.08, 0.10]),
        "lift": np.asarray([0.06, 0.00, 0.09]),
    }[mode]
    camera_position = settled_chest + camera_offset
    # Aim at the predeclared midpoint of the slide range, not an observed outcome.
    target_look = target_stage_position + np.asarray(
        [0.05 if mode != "lift" else 0.0, 0.0, 0.04 if mode == "lift" else 0.0]
    )
    backward = camera_position - target_look
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray([0.0, 0.0, 1.0]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack([right, up, backward])
    camera_quat = np.empty(4)
    mujoco.mju_mat2Quat(camera_quat, rotation.ravel())
    model.cam_pos[camera_id] = camera_position
    model.cam_quat[camera_id] = camera_quat
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    initial_target = data.xpos[target_body_id].copy()

    frames: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    action: list[np.ndarray] = []
    contact_wrench: list[np.ndarray] = []
    contact_count: list[int] = []
    accelerometer: list[np.ndarray] = []
    gyroscope: list[np.ndarray] = []
    target_pose: list[np.ndarray] = []
    hand_pose: list[np.ndarray] = []
    body_pose: list[np.ndarray] = []
    camera_pose: list[np.ndarray] = []
    actuator_control: list[np.ndarray] = []
    target_area: list[float] = []
    grasp_state: list[bool] = []
    grasp_engaged = False
    keyframes = {0, 105, 150, 210, 239}
    mimo_body_ids = [
        body_id
        for body_id in range(model.nbody)
        if (model.body(body_id).name or "").startswith("mimo_")
    ]
    (output_dir / "body_names.json").write_text(
        json.dumps([model.body(body_id).name for body_id in mimo_body_ids], indent=2)
        + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    for frame_index in range(240):
        if mode == "lift" and frame_index < 45:
            desired_x = initial_mocap[0]
        elif mode == "lift" and frame_index < 120:
            alpha = (frame_index - 45) / 75
            desired_x = initial_mocap[0] + alpha * 0.12
        elif mode == "lift":
            desired_x = initial_mocap[0] + 0.12
        elif frame_index < 45:
            desired_x = initial_mocap[0]
        elif frame_index < 120:
            alpha = (frame_index - 45) / 75
            desired_x = initial_mocap[0] + alpha * 0.10
        elif frame_index < 185:
            alpha = (frame_index - 120) / 65
            desired_x = initial_mocap[0] + 0.10 + alpha * 0.12
        else:
            alpha = (frame_index - 185) / 54
            desired_x = initial_mocap[0] + 0.22 - alpha * 0.32
        desired_z = initial_mocap[2]
        if mode == "lift" and frame_index >= 120:
            desired_z += min((frame_index - 120) / 70, 1.0) * 0.18
        desired = np.asarray([desired_x, initial_mocap[1], desired_z])
        previous = data.mocap_pos[mocap_id].copy()
        frame_wrench = np.zeros(6, dtype=np.float64)
        frame_contact_count = 0
        for substep in range(17):
            fraction = (substep + 1) / 17
            data.mocap_pos[mocap_id] = previous + fraction * (desired - previous)
            if mode == "lift" and grasp_engaged:
                target_velocity = data.cvel[target_body_id, 3:6]
                desired_target = data.xpos[right_hand_id] + np.asarray([0.035, 0.0, -0.025])
                data.xfrc_applied[target_body_id, :3] = (
                    350.0 * (desired_target - data.xpos[target_body_id])
                    - 25.0 * target_velocity
                )
            mujoco.mj_step(model, data)
            substep_wrench, substep_count = _contact_sample(
                model, data, target_geom_id, hand_geom_ids
            )
            if mode == "lift" and substep_count and not grasp_engaged:
                grasp_engaged = True
            frame_wrench += substep_wrench
            frame_contact_count += substep_count
        renderer.update_scene(
            data, camera="mimo_chest_camera", scene_option=scene_option
        )
        frames.append(renderer.render().copy())
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        action.append(desired - previous)
        actuator_control.append(data.ctrl.copy())
        contact_wrench.append(frame_wrench)
        contact_count.append(frame_contact_count)
        grasp_state.append(grasp_engaged)
        accelerometer.append(data.sensordata[acc_slice].copy())
        gyroscope.append(data.sensordata[gyro_slice].copy())
        target_pose.append(
            np.concatenate([data.xpos[target_body_id], data.xquat[target_body_id]])
        )
        hand_pose.append(np.concatenate([data.xpos[right_hand_id], data.xquat[right_hand_id]]))
        body_pose.append(
            np.concatenate(
                [data.xpos[mimo_body_ids], data.xquat[mimo_body_ids]], axis=1
            )
        )
        camera_pose.append(
            np.concatenate(
                [
                    data.cam_xpos[camera_id],
                    data.cam_xmat[camera_id].reshape(9),
                ]
            )
        )
        if frame_index in keyframes:
            target_area.append(
                _target_area(renderer, data, target_geom_id, scene_option)
            )
            renderer.enable_segmentation_rendering()
            renderer.update_scene(
                data, camera="mimo_chest_camera", scene_option=scene_option
            )
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            target_mask = (
                (segmentation[..., 0] == target_geom_id)
                & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            )
            imageio.imwrite(
                output_dir / f"target_segmentation_{frame_index:03d}.png",
                target_mask.astype(np.uint8) * 255,
            )
            renderer.enable_depth_rendering()
            renderer.update_scene(
                data, camera="mimo_chest_camera", scene_option=scene_option
            )
            depth_m = renderer.render().copy()
            renderer.disable_depth_rendering()
            imageio.imwrite(
                output_dir / f"depth_mm_{frame_index:03d}.png",
                np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16),
            )
    elapsed = time.perf_counter() - started
    renderer.close()

    silent_video = output_dir / f"hybrid_{mode}_silent.mp4"
    imageio.mimwrite(silent_video, frames, fps=30, codec="libx264", quality=8)
    speech_path = output_dir / "speech_de.aiff"
    subprocess.run(
        ["say", "-v", "Anna", "-o", str(speech_path), "Schau, der Ball bewegt sich."],
        check=True,
    )
    video_path = output_dir / f"hybrid_{mode}_with_speech.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(silent_video),
            "-itsoffset",
            "1.0",
            "-i",
            str(speech_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-t",
            "8.0",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    speech_events = {
        "clock": "episode_seconds",
        "events": [
            {
                "start_s": 1.0,
                "text": "Schau, der Ball bewegt sich.",
                "voice": "macOS Anna de_DE",
                "status": "local_synthetic_timing_interface_not_human_validation",
            }
        ],
    }
    (output_dir / "speech_events.json").write_text(
        json.dumps(speech_events, indent=2) + "\n", encoding="utf-8"
    )
    telemetry_path = output_dir / f"hybrid_{mode}_telemetry.npz"
    np.savez_compressed(
        telemetry_path,
        frame=np.arange(240, dtype=np.int32),
        time_s=np.arange(1, 241, dtype=np.float64) / 30,
        action=np.asarray(action),
        actuator_control=np.asarray(actuator_control),
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        touch_wrench=np.asarray(contact_wrench),
        touch_contact_count=np.asarray(contact_count),
        grasp_constraint_engaged=np.asarray(grasp_state),
        vestibular_accelerometer=np.asarray(accelerometer),
        vestibular_gyroscope=np.asarray(gyroscope),
        target_pose=np.asarray(target_pose),
        hand_pose=np.asarray(hand_pose),
        body_pose=np.asarray(body_pose),
        camera_pose=np.asarray(camera_pose),
        area_keyframe=np.asarray(sorted(keyframes), dtype=np.int32),
        target_area_fraction=np.asarray(target_area),
    )
    final_target = np.asarray(target_pose)[-1, :3]
    receipt = {
        "stage": "hybrid_b_direct_mjcf_composition",
        "mode": mode,
        "scene": f"iTHOR {scene_path.stem}",
        "mimo_age_months": 24,
        "controller": (
            "pose-matched right-hand mocap weld; post-contact spring grasp only"
            if mode == "lift"
            else "pose-matched right-hand mocap weld; target is a contact-driven 1D slide"
        ),
        "architecture": platform.machine(),
        "mujoco": mujoco.__version__,
        "frames": 240,
        "duration_s": 8.0,
        "wall_seconds": elapsed,
        "frames_per_wall_second": 240 / elapsed,
        "target_initial_xyz": initial_target.tolist(),
        "target_final_xyz": final_target.tolist(),
        "target_displacement_m": float(np.linalg.norm(final_target - initial_target)),
        "target_vertical_displacement_m": float(final_target[2] - initial_target[2]),
        "grasp_engaged_after_contact": bool(any(grasp_state)),
        "hand_target_contact_frames": int(np.count_nonzero(contact_count)),
        "maximum_contact_force_n": float(
            np.max(np.linalg.norm(np.asarray(contact_wrench)[:, :3], axis=1))
        ),
        "target_area_keyframes": {
            str(frame): area for frame, area in zip(sorted(keyframes), target_area)
        },
        "minimum_target_area_fraction": float(min(target_area)),
        "workspace": {
            "shoulder_to_hand_max_m": 0.407,
            "target_initial_distance_from_hand_m": float(
                np.linalg.norm(initial_target - np.asarray(hand_pose)[0, :3])
            ),
            "frozen_basis": "MIMo-v2 XML upper/lower arm plus hand lengths",
        },
        "camera": {
            "mount": "virtual chest/vest, fixed after deterministic pre-settle",
            "settled_chest_xyz": settled_chest.tolist(),
            "position_xyz": camera_position.tolist(),
            "look_target_xyz": target_look.tolist(),
            "fovy_degrees": 90,
        },
        "streams": [
            "rgb",
            "action",
            "actuator_control",
            "qpos",
            "qvel",
            "touch_wrench",
            "contact_count",
            "grasp_constraint_engaged",
            "vestibular_accelerometer",
            "vestibular_gyroscope",
            "target_pose",
            "hand_pose",
            "full_mimo_body_pose",
            "camera_pose",
            "depth_keyframes",
            "target_segmentation_keyframes",
            "speech_events",
        ],
        "video": {"name": video_path.name, "sha256": _sha256(video_path)},
        "telemetry": {"name": telemetry_path.name, "sha256": _sha256(telemetry_path)},
    }
    (output_dir / f"hybrid_{mode}_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_path", type=Path)
    parser.add_argument("mimo_assets", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mode", choices=("push", "near_miss", "lift"), default="push")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.scene_path, args.mimo_assets, args.output_dir, mode=args.mode),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
