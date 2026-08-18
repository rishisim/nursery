"""Embodied MIMo/MuJoCo kernel with an immutable articulated-head camera."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import mujoco
import numpy as np

from .controllers import minimum_hand_target_distance


TARGET_COLLISION_BIT = 1 << 20
ARM_JOINTS = (
    "robot:right_shoulder_horizontal",
    "robot:right_shoulder_ad_ab",
    "robot:right_shoulder_rotation",
    "robot:right_elbow",
    "robot:right_hand1",
    "robot:right_hand2",
    "robot:right_hand3",
)
POSTURE_JOINTS = (
    "robot:hip_rot1",
    "robot:hip_rot2",
    "robot:chest_rot",
    "robot:chest_lean",
)
HEAD_JOINTS = (
    "robot:head_swivel",
    "robot:head_tilt",
    "robot:head_tilt_side",
)


def value(spec: dict[str, Any], key: str) -> Any:
    item = spec[key]
    return item["value"] if isinstance(item, dict) and "value" in item else item


def _radians(text: str) -> str:
    return " ".join(f"{np.deg2rad(float(item)):.17g}" for item in text.split())


def _hand_geom_names(mimo_assets: Path) -> tuple[str, ...]:
    root = ElementTree.parse(mimo_assets / "mimo" / "MIMo_modelv2.xml").getroot()
    return tuple(
        sorted(
            geom.get("name")
            for geom in root.iter("geom")
            if (geom.get("name") or "").startswith("geom:right_")
            and any(
                (geom.get("name") or "").removeprefix("geom:right_").startswith(token)
                for token in ("hand", "ff", "mf", "rf", "lf", "th")
            )
        )
    )


def _write_embodied_mimo_model(
    mimo_assets: Path,
    destination: Path,
    *,
    camera_mount_position: tuple[float, float, float],
    camera_mount_quaternion: tuple[float, float, float, float],
) -> None:
    """Convert stock degree-authored MIMo for a radian MolmoSpaces parent."""
    tree = ElementTree.parse(mimo_assets / "mimo" / "MIMo_modelv2.xml")
    root = tree.getroot()
    for element in root.iter():
        if "euler" in element.attrib:
            element.set("euler", _radians(element.attrib["euler"]))
        if element.tag == "joint":
            for attribute in ("range", "ref", "springref"):
                if attribute in element.attrib:
                    element.set(attribute, _radians(element.attrib[attribute]))

    head = root.find(".//body[@name='head']")
    if head is None:
        raise RuntimeError("MIMo head body was not found")
    ElementTree.SubElement(
        head,
        "camera",
        {
            "name": "head_camera",
            "mode": "fixed",
            "fovy": "90",
            "pos": " ".join(str(item) for item in camera_mount_position),
            "quat": " ".join(str(item) for item in camera_mount_quaternion),
        },
    )

    # Separate physics and appearance layers. The duplicate has no collision or
    # meaningful mass and follows the exact same articulated bodies.
    right_arm = root.find(".//body[@name='right_upper_arm']")
    if right_arm is None:
        raise RuntimeError("MIMo right arm was not found")
    for body in right_arm.iter("body"):
        for geom in list(body.findall("geom")):
            name = geom.get("name")
            if not name:
                continue
            geom.set("group", "3")
            visual = ElementTree.Element("geom", dict(geom.attrib))
            visual.set("name", f"visual:{name}")
            visual.set("group", "2")
            visual.set("contype", "0")
            visual.set("conaffinity", "0")
            visual.set("mass", "1e-9")
            visual.set("material", "skin")
            body.append(visual)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")


def _target_xml(target_definition: dict[str, Any]) -> str:
    rgba = " ".join(str(item) for item in target_definition["rgba"])
    if target_definition["geometry"] == "sphere":
        return f"""      <geom name="target_geom" type="sphere" size="0.05"
            mass="0.30" rgba="{rgba}" friction="3.0 0.01 0.0005"
            solref="0.06 1" solimp="0.8 0.95 0.003"
            contype="{TARGET_COLLISION_BIT}" conaffinity="{TARGET_COLLISION_BIT | 8}"/>"""
    if target_definition["geometry"] != "cylinder_with_three_capsule_handle":
        raise ValueError(f"unsupported target geometry: {target_definition['geometry']}")
    return f"""      <geom name="target_geom" type="cylinder" size="0.04 0.05"
            mass="0.30" rgba="{rgba}" friction="3.0 0.01 0.0005"
            solref="0.06 1" solimp="0.8 0.95 0.003"
            contype="{TARGET_COLLISION_BIT}" conaffinity="{TARGET_COLLISION_BIT | 8}"/>
      <geom name="target_handle_upper" type="capsule" size="0.007"
            fromto="0 -0.025 0.025 0 -0.065 0.025" rgba="{rgba}"
            contype="{TARGET_COLLISION_BIT}" conaffinity="{TARGET_COLLISION_BIT | 8}" mass="0.002"/>
      <geom name="target_handle_outer" type="capsule" size="0.007"
            fromto="0 -0.065 -0.025 0 -0.065 0.025" rgba="{rgba}"
            contype="{TARGET_COLLISION_BIT}" conaffinity="{TARGET_COLLISION_BIT | 8}" mass="0.002"/>
      <geom name="target_handle_lower" type="capsule" size="0.007"
            fromto="0 -0.025 -0.025 0 -0.065 -0.025" rgba="{rgba}"
            contype="{TARGET_COLLISION_BIT}" conaffinity="{TARGET_COLLISION_BIT | 8}" mass="0.002"/>"""


def _target_geom_names(target_definition: dict[str, Any]) -> tuple[str, ...]:
    if target_definition["geometry"] == "sphere":
        return ("target_geom",)
    if target_definition["geometry"] == "cylinder_with_three_capsule_handle":
        return (
            "target_geom",
            "target_handle_upper",
            "target_handle_outer",
            "target_handle_lower",
        )
    raise ValueError(
        f"unsupported target geometry: {target_definition['geometry']}"
    )


def _clutter_xml(
    root_xy: tuple[float, float], clutter_layout: list[dict[str, Any]]
) -> str:
    rows = []
    for item in clutter_layout:
        identifier = item["id"]
        offset = np.asarray(item["offset_xyz_m"], dtype=np.float64)
        position = np.asarray([root_xy[0], root_xy[1], 0.0]) + offset
        size = " ".join(str(value_) for value_ in item["half_size_xyz_m"])
        rgba = " ".join(str(value_) for value_ in item["rgba"])
        rows.append(
            f'''    <body name="{identifier}" pos="{' '.join(str(value_) for value_ in position)}">
      <geom name="clutter_physical_{identifier}" type="box" size="{size}"
            group="4" rgba="0 1 0 1" friction="1.2 0.01 0.0005"
            contype="8" conaffinity="{TARGET_COLLISION_BIT | 7}"/>
      <geom name="clutter_visual_{identifier}" type="box" size="{size}"
            group="0" rgba="{rgba}" contype="0" conaffinity="0"/>
    </body>'''
        )
    return "\n".join(rows)


def _component_xml(
    mimo_assets: Path,
    embodied_model_path: Path,
    root_xy: tuple[float, float],
    target_definition: dict[str, Any],
    support_definition: dict[str, Any],
    clutter_layout: list[dict[str, Any]],
    target_offset_from_root: tuple[float, float, float],
    support_offset_from_root: tuple[float, float, float],
    reach_distractor_offset_from_root: tuple[float, float, float],
) -> str:
    contact_pairs = "\n".join(
        f'    <pair geom1="{name}" geom2="target_geom" condim="4" margin="0.006" gap="0.001" friction="2.0 0.01 0.0005 0.0001 0.0001" solref="0.03 1" solimp="0.9 0.98 0.002"/>'
        for name in _hand_geom_names(mimo_assets)
    )
    target_support_pairs = "\n".join(
        f'    <pair geom1="{target_name}" geom2="{support_name}" condim="4" margin="0.003" gap="0.001" friction="3.0 0.01 0.0005 0.0001 0.0001" solref="0.008 1" solimp="0.98 0.999 0.0005"/>'
        for target_name in _target_geom_names(target_definition)
        for support_name in (
            "support_catch_tray", "support_geom", "support_rim_back"
        )
    )
    target_floor_pairs = "\n".join(
        f'    <pair geom1="{target_name}" geom2="floor" condim="4" margin="0.003" gap="0.001" friction="2.0 0.01 0.0005 0.0001 0.0001" solref="0.002 1" solimp="0.98 0.999 0.0005"/>'
        for target_name in _target_geom_names(target_definition)
    )
    clutter_xml = _clutter_xml(root_xy, clutter_layout)
    catch_radius = float(support_definition["catch_tray_radius_m"])
    catch_half_height = float(
        support_definition["catch_tray_half_height_m"]
    )
    catch_center_below = float(
        support_definition["catch_tray_center_below_support_origin_m"]
    )
    pedestal_radius = float(support_definition["pedestal_radius_m"])
    pedestal_half_height = float(
        support_definition["pedestal_half_height_m"]
    )
    target_position = np.asarray(
        [root_xy[0], root_xy[1], 0.55], dtype=np.float64
    ) + np.asarray(target_offset_from_root, dtype=np.float64)
    support_position = np.asarray(
        [root_xy[0], root_xy[1], 0.55], dtype=np.float64
    ) + np.asarray(support_offset_from_root, dtype=np.float64)
    distractor_position = np.asarray(
        [root_xy[0], root_xy[1], 0.55], dtype=np.float64
    ) + np.asarray(reach_distractor_offset_from_root, dtype=np.float64)
    # Preserve the already-qualified decimal-authored base heights exactly;
    # binary addition of 0.55 and 0.025 otherwise serializes as
    # 0.5750000000000001 and needlessly changes the frozen base model.
    target_position[2] = round(float(target_position[2]), 12)
    support_position[2] = round(float(support_position[2]), 12)
    distractor_position[2] = round(float(distractor_position[2]), 12)
    return f"""<mujoco model="EmbodiedMIMoKernel">
  <compiler inertiafromgeom="true" angle="radian" assetdir="{mimo_assets}"/>
  <include file="{mimo_assets / 'mimo' / 'MIMo_metav2.xml'}"/>
  <worldbody>
    <body name="mimo_location" pos="{root_xy[0]} {root_xy[1]} 0.55" quat="1 0 0 0">
      <joint name="root_x" type="slide" axis="1 0 0" range="-0.02 0.14" damping="80" armature="0.2"/>
      <joint name="root_y" type="slide" axis="0 1 0" range="-0.02 0.02" damping="80" armature="0.2"/>
      <joint name="root_yaw" type="hinge" axis="0 0 1" range="-0.25 0.25" damping="30" armature="0.1"/>
      <include file="{embodied_model_path}"/>
    </body>
    <body name="target" pos="{' '.join(str(item) for item in target_position)}">
      <joint name="target_free" type="free" damping="3.0" armature="0.001"/>
{_target_xml(target_definition)}
    </body>
    <body name="support" pos="{' '.join(str(item) for item in support_position)}">
      <geom name="support_catch_tray" type="cylinder" pos="0 0 -{catch_center_below}"
            size="{catch_radius} {catch_half_height}" rgba="0.24 0.13 0.06 1" friction="3 0.005 0.0002"
            solref="0.008 1" solimp="0.98 0.999 0.0005" contype="8"
            conaffinity="{TARGET_COLLISION_BIT | 7}"/>
      <geom name="support_geom" type="cylinder" size="{pedestal_radius} {pedestal_half_height}"
            rgba="0.34 0.19 0.08 1" friction="3 0.005 0.0002"
            solref="0.008 1"
            solimp="0.98 0.999 0.0005" contype="8"
            conaffinity="{TARGET_COLLISION_BIT | 7}"/>
      <geom name="support_rim_back" type="box" pos="0.045 0 0.04"
            size="0.004 0.05 0.015" rgba="0.28 0.14 0.05 1"
            friction="3 0.005 0.0002" solref="0.008 1"
            solimp="0.98 0.999 0.0005" contype="8"
            conaffinity="{TARGET_COLLISION_BIT | 7}"/>
    </body>
    <body name="reach_distractor" pos="{' '.join(str(item) for item in distractor_position)}">
      <geom name="reach_distractor_geom" type="sphere" size="0.035"
            rgba="0.1 0.35 0.8 1" mass="0.08" contype="2"
            conaffinity="{TARGET_COLLISION_BIT | 12}"/>
    </body>
{clutter_xml}
    <camera name="external_qa" pos="{root_xy[0] - 0.7} {root_xy[1] - 0.9} 1.25"
            mode="targetbody" target="mimo_location" fovy="55"/>
  </worldbody>
  <contact>
{contact_pairs}
{target_support_pairs}
{target_floor_pairs}
  </contact>
</mujoco>
"""


@dataclass
class KernelModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    hand_geom_ids: tuple[int, ...]
    target_geom_id: int
    target_geom_ids: tuple[int, ...]
    support_geom_ids: tuple[int, ...]
    reach_distractor_geom_id: int
    clutter_geom_ids: tuple[int, ...]
    clutter_visual_geom_ids: tuple[int, ...]
    target_body_id: int
    target_qpos_adr: int
    target_dof_adr: int
    right_hand_body_id: int
    head_body_id: int
    vestibular_site_id: int
    camera_id: int
    external_camera_id: int
    sensor_slices: dict[str, slice]
    arm_joint_ids: tuple[int, ...]
    arm_qpos_ids: tuple[int, ...]
    arm_dof_ids: tuple[int, ...]
    posture_joint_ids: tuple[int, ...]
    head_joint_ids: tuple[int, ...]
    root_qpos_ids: tuple[int, ...]
    root_dof_ids: tuple[int, ...]
    finger_joint_ids: tuple[int, ...]
    fingertip_geom_ids: tuple[int, ...]
    relevant_collision_geom_ids: tuple[int, ...]
    collision_policy_sha256: str
    camera_mount_position: np.ndarray
    camera_mount_quaternion: np.ndarray
    mimo_body_ids: tuple[int, ...]
    mimo_body_names: tuple[str, ...]


def _ids_for_joints(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(model.joint(f"kernel_{name}").id for name in names)


def build_kernel_model(
    scene_path: Path,
    mimo_assets: Path,
    component_path: Path,
    *,
    root_xy: tuple[float, float],
    target_definition: dict[str, Any],
    support_definition: dict[str, Any] | None = None,
    clutter_layout: list[dict[str, Any]] | None = None,
    target_offset_from_root: tuple[float, float, float] = (0.34, -0.15, 0.10),
    support_offset_from_root: tuple[float, float, float] = (0.34, -0.15, 0.025),
    reach_distractor_offset_from_root: tuple[float, float, float] = (
        0.23,
        0.02,
        0.10,
    ),
    camera_mount_position: tuple[float, float, float] = (0.081, 0.0, 0.067375),
    camera_mount_quaternion: tuple[float, float, float, float] = (
        0.2642907803854001,
        -0.48706974071026582,
        0.4375916206381148,
        0.70811512103260366,
    ),
) -> KernelModel:
    scene_path = scene_path.resolve()
    mimo_assets = mimo_assets.resolve()
    component_path.parent.mkdir(parents=True, exist_ok=True)
    embodied_model_path = component_path.with_name("embodied_mimo_model.xml").resolve()
    _write_embodied_mimo_model(
        mimo_assets,
        embodied_model_path,
        camera_mount_position=camera_mount_position,
        camera_mount_quaternion=camera_mount_quaternion,
    )
    component_path.write_text(
        _component_xml(
            mimo_assets,
            embodied_model_path,
            root_xy,
            target_definition,
            support_definition
            or {
                "pedestal_radius_m": 0.025,
                "pedestal_half_height_m": 0.025,
                "catch_tray_radius_m": 0.18,
                "catch_tray_half_height_m": 0.01,
                "catch_tray_center_below_support_origin_m": 0.025,
            },
            clutter_layout or [],
            target_offset_from_root,
            support_offset_from_root,
            reach_distractor_offset_from_root,
        ),
        encoding="utf-8",
    )
    parent = mujoco.MjSpec.from_file(str(scene_path))
    child = mujoco.MjSpec.from_file(str(component_path.resolve()))
    frame = parent.worldbody.add_frame()
    frame.name = "embodied_kernel_attach"
    parent.attach(child, prefix="kernel_", frame=frame)
    model = parent.compile()
    data = mujoco.MjData(model)

    hand_geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith("kernel_geom:right_")
        and not (model.geom(geom_id).name or "").startswith("kernel_visual:")
        and any(
            (model.geom(geom_id).name or "")
            .removeprefix("kernel_geom:right_")
            .startswith(token)
            for token in ("hand", "ff", "mf", "rf", "lf", "th")
        )
    )
    if not hand_geom_ids:
        raise RuntimeError("MIMo right-hand collision geoms were not found")
    model.geom_conaffinity[list(hand_geom_ids)] |= TARGET_COLLISION_BIT
    target_geom_ids = tuple(
        geom_id for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith("kernel_target_")
        and model.geom(geom_id).name != "kernel_target_free"
    )
    target_geom_id = model.geom("kernel_target_geom").id
    support_geom_ids = tuple(
        model.geom(name).id
        for name in (
            "kernel_support_catch_tray",
            "kernel_support_geom",
            "kernel_support_rim_back",
        )
    )
    clutter_geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith(
            "kernel_clutter_physical_"
        )
    )
    clutter_visual_geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith(
            "kernel_clutter_visual_"
        )
    )
    hand_contype_union = int(np.bitwise_or.reduce(model.geom_contype[list(hand_geom_ids)]))
    model.geom_conaffinity[list(target_geom_ids)] = (
        hand_contype_union | 8 | TARGET_COLLISION_BIT
    )
    relevant_tokens = (
        "kernel_cb", "kernel_ub", "kernel_neck", "kernel_head",
        "kernel_right_uarm", "kernel_right_larm", "kernel_geom:right_",
        "kernel_target_", "kernel_support", "kernel_reach_distractor_geom",
        "kernel_clutter_physical_",
    )
    relevant_collision_geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if name.startswith("kernel_visual:") or name.startswith(
            "kernel_clutter_visual_"
        ):
            continue
        is_existing_scene_collider = (
            not name.startswith("kernel_")
            and bool(model.geom_contype[geom_id])
            and bool(model.geom_conaffinity[geom_id])
        )
        if any(token in name for token in relevant_tokens) or is_existing_scene_collider:
            relevant_collision_geom_ids.append(geom_id)
            if name.startswith("kernel_") and model.geom_contype[geom_id] == 0:
                model.geom_contype[geom_id] = 4
            if name.startswith("kernel_") and model.geom_conaffinity[geom_id] == 0:
                model.geom_conaffinity[geom_id] = 15
    for body_id in range(model.nbody):
        if (
            (model.body(body_id).name or "").startswith("kernel_")
            and model.body(body_id).name not in {
                "kernel_target", "kernel_support", "kernel_reach_distractor"
            }
            and not (model.body(body_id).name or "").startswith(
                "kernel_clutter_"
            )
        ):
            model.body_gravcomp[body_id] = 0.85

    def sensor_slice(name: str, width: int) -> slice:
        sensor_id = model.sensor(f"kernel_{name}").id
        start = int(model.sensor_adr[sensor_id])
        return slice(start, start + width)

    arm_joint_ids = _ids_for_joints(model, ARM_JOINTS)
    posture_joint_ids = _ids_for_joints(model, POSTURE_JOINTS)
    head_joint_ids = _ids_for_joints(model, HEAD_JOINTS)
    root_joint_ids = tuple(
        model.joint(name).id
        for name in ("kernel_root_x", "kernel_root_y", "kernel_root_yaw")
    )
    finger_joint_ids = tuple(
        joint_id for joint_id in range(model.njnt)
        if (model.joint(joint_id).name or "").startswith("kernel_robot:right_")
        and any(token in (model.joint(joint_id).name or "") for token in ("ff_", "mf_", "rf_", "lf_", "th_"))
    )
    for joint_id in (*arm_joint_ids, *head_joint_ids, *posture_joint_ids):
        dof_id = int(model.jnt_dofadr[joint_id])
        model.dof_damping[dof_id] = max(float(model.dof_damping[dof_id]), 0.10)
    fingertip_geom_ids = tuple(
        geom_id for geom_id in hand_geom_ids
        if any(token in (model.geom(geom_id).name or "") for token in ("ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal"))
    )
    mimo_body_ids = tuple(
        body_id for body_id in range(model.nbody)
        if (model.body(body_id).name or "").startswith("kernel_")
        and model.body(body_id).name not in {
            "kernel_target", "kernel_support", "kernel_reach_distractor"
        }
        and not (model.body(body_id).name or "").startswith("kernel_clutter_")
    )
    collision_rows = [
        [model.geom(i).name, int(model.geom_contype[i]), int(model.geom_conaffinity[i])]
        for i in relevant_collision_geom_ids
    ]
    collision_policy_sha256 = hashlib.sha256(
        json.dumps(collision_rows, separators=(",", ":")).encode()
    ).hexdigest()
    model.vis.map.znear = 0.02 / float(model.stat.extent)
    mujoco.mj_forward(model, data)
    target_joint = model.joint("kernel_target_free").id
    return KernelModel(
        model=model,
        data=data,
        hand_geom_ids=hand_geom_ids,
        target_geom_id=target_geom_id,
        target_geom_ids=target_geom_ids,
        support_geom_ids=support_geom_ids,
        reach_distractor_geom_id=model.geom("kernel_reach_distractor_geom").id,
        clutter_geom_ids=clutter_geom_ids,
        clutter_visual_geom_ids=clutter_visual_geom_ids,
        target_body_id=model.body("kernel_target").id,
        target_qpos_adr=int(model.jnt_qposadr[target_joint]),
        target_dof_adr=int(model.jnt_dofadr[target_joint]),
        right_hand_body_id=model.body("kernel_right_hand").id,
        head_body_id=model.body("kernel_head").id,
        vestibular_site_id=model.site("kernel_vestibular").id,
        camera_id=model.camera("kernel_head_camera").id,
        external_camera_id=model.camera("kernel_external_qa").id,
        sensor_slices={
            "vestibular_accelerometer": sensor_slice("vestibular_acc", 3),
            "vestibular_gyroscope": sensor_slice("vestibular_gyro", 3),
        },
        arm_joint_ids=arm_joint_ids,
        arm_qpos_ids=tuple(int(model.jnt_qposadr[j]) for j in arm_joint_ids),
        arm_dof_ids=tuple(int(model.jnt_dofadr[j]) for j in arm_joint_ids),
        posture_joint_ids=posture_joint_ids,
        head_joint_ids=head_joint_ids,
        root_qpos_ids=tuple(int(model.jnt_qposadr[j]) for j in root_joint_ids),
        root_dof_ids=tuple(int(model.jnt_dofadr[j]) for j in root_joint_ids),
        finger_joint_ids=finger_joint_ids,
        fingertip_geom_ids=fingertip_geom_ids,
        relevant_collision_geom_ids=tuple(relevant_collision_geom_ids),
        collision_policy_sha256=collision_policy_sha256,
        camera_mount_position=np.asarray(camera_mount_position, dtype=np.float64),
        camera_mount_quaternion=np.asarray(camera_mount_quaternion, dtype=np.float64),
        mimo_body_ids=mimo_body_ids,
        mimo_body_names=tuple(model.body(body_id).name for body_id in mimo_body_ids),
    )


def _phase_schedule(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "activity" in spec:
        return spec["activity"]["vertical_slice"]
    return value(spec, "phase_timestamps")


def phase_at(spec: dict[str, Any], time_s: float) -> dict[str, Any]:
    phases = _phase_schedule(spec)
    for phase in phases:
        if phase["start_s"] <= time_s < phase["end_s"]:
            return phase
    return phases[-1]


def _phase_action(phase: dict[str, Any]) -> str:
    """Return the bounded planner action for an internal behavior phase."""
    if "action" in phase:
        return str(phase["action"])
    aliases = {
        "reach_past_distractor": "reach",
        "fingertip_contact": "touch",
        "lift": "grasp",
        "inspect_rotate": "rotate",
        "head_turn_maintain_contact": "inspect",
        "release": "drop",
        "settle": "drop",
    }
    return aliases.get(str(phase["phase"]), str(phase["phase"]))


def _named_phase(spec: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next(
        (phase for phase in _phase_schedule(spec) if phase["phase"] == name),
        None,
    )


def _smooth(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _interpolate_waypoints(waypoints: np.ndarray, alpha: float) -> np.ndarray:
    position = float(np.clip(alpha, 0.0, 1.0)) * (len(waypoints) - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, len(waypoints) - 1)
    fraction = position - lower
    return waypoints[lower] * (1.0 - fraction) + waypoints[upper] * fraction


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    error = target @ current.T
    return 0.5 * np.asarray(
        [error[2, 1] - error[1, 2], error[0, 2] - error[2, 0], error[1, 0] - error[0, 1]]
    )


def solve_arm_ik(
    kernel: KernelModel,
    seed_qpos: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    iterations: int = 35,
) -> np.ndarray:
    model = kernel.model
    shadow = mujoco.MjData(model)
    shadow.qpos[:] = seed_qpos
    for _ in range(iterations):
        mujoco.mj_forward(model, shadow)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, shadow, jacp, jacr, kernel.right_hand_body_id)
        current_rotation = shadow.xmat[kernel.right_hand_body_id].reshape(3, 3)
        error = np.concatenate(
            [target_position - shadow.xpos[kernel.right_hand_body_id], 0.35 * _rotation_error(target_rotation, current_rotation)]
        )
        if np.linalg.norm(error[:3]) < 5e-4 and np.linalg.norm(error[3:]) < 2e-3:
            break
        jacobian = np.vstack([jacp[:, kernel.arm_dof_ids], 0.35 * jacr[:, kernel.arm_dof_ids]])
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 2e-4 * np.eye(6), error
        )
        delta = np.clip(delta, -0.06, 0.06)
        for joint_id, qpos_id, amount in zip(kernel.arm_joint_ids, kernel.arm_qpos_ids, delta):
            shadow.qpos[qpos_id] += amount
            if model.jnt_limited[joint_id]:
                shadow.qpos[qpos_id] = np.clip(shadow.qpos[qpos_id], *model.jnt_range[joint_id])
    return shadow.qpos[list(kernel.arm_qpos_ids)].copy()


def solve_head_attention(
    kernel: KernelModel,
    seed_qpos: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """Resolve a bounded neck pose; the fixed camera is never manipulated."""
    model = kernel.model
    shadow = mujoco.MjData(model)
    head_qpos = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in kernel.head_joint_ids]
    )
    candidate = seed_qpos.copy()

    def loss(qpos: np.ndarray) -> float:
        shadow.qpos[:] = qpos
        mujoco.mj_forward(model, shadow)
        forward = -shadow.cam_xmat[kernel.camera_id].reshape(3, 3)[:, 2]
        direction = target_position - shadow.cam_xpos[kernel.camera_id]
        norm = np.linalg.norm(direction)
        if norm <= 1e-9:
            return 0.0
        return float(1.0 - np.dot(forward, direction / norm))

    step = np.deg2rad(30.0)
    for _ in range(12):
        best = candidate
        best_loss = loss(candidate)
        for qpos_id, joint_id in zip(head_qpos, kernel.head_joint_ids):
            for direction in (-1.0, 1.0):
                trial = candidate.copy()
                trial[qpos_id] = np.clip(
                    trial[qpos_id] + direction * step,
                    model.jnt_range[joint_id, 0],
                    model.jnt_range[joint_id, 1],
                )
                trial_loss = loss(trial)
                if trial_loss < best_loss:
                    best, best_loss = trial, trial_loss
        candidate = best.copy()
        step *= 0.70
    return candidate[head_qpos].copy()


def _finger_targets(kernel: KernelModel, closure: float) -> dict[int, float]:
    closure = float(np.clip(closure, 0.0, 1.0))
    result: dict[int, float] = {}
    for joint_id in kernel.finger_joint_ids:
        name = kernel.model.joint(joint_id).name or ""
        if "_side" in name or "_pivot" in name:
            degrees = 0.0
        elif "th_swivel" in name:
            degrees = 35.0 + 45.0 * closure
        elif "th_adduction" in name:
            degrees = -40.0 + 30.0 * closure
        elif "_knuckle" in name:
            degrees = -5.0 + 80.0 * closure
        elif "_middle" in name:
            degrees = 10.0 + 65.0 * closure
        elif "_distal" in name:
            degrees = 8.0 + 55.0 * closure
        elif "lf_meta" in name:
            degrees = 25.0 * closure
        else:
            degrees = 0.0
        result[joint_id] = float(np.deg2rad(degrees))
    return result


def _target_contacts(kernel: KernelModel) -> tuple[np.ndarray, int, set[int], float]:
    wrench = np.zeros(6)
    count = 0
    finger_bodies: set[int] = set()
    minimum_distance = float("inf")
    hand_set = set(kernel.hand_geom_ids)
    target_set = set(kernel.target_geom_ids)
    for contact_index in range(kernel.data.ncon):
        contact = kernel.data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair.intersection(hand_set) and pair.intersection(target_set):
            local = np.zeros(6)
            mujoco.mj_contactForce(kernel.model, kernel.data, contact_index, local)
            wrench += local
            count += 1
            hand_geom = next(iter(pair.intersection(hand_set)))
            hand_name = kernel.model.geom(hand_geom).name or ""
            if any(token in hand_name for token in ("ff", "mf", "rf", "lf", "th")):
                finger_bodies.add(int(kernel.model.geom_bodyid[hand_geom]))
            minimum_distance = min(minimum_distance, float(contact.dist))
    return (
        wrench,
        count,
        finger_bodies,
        minimum_distance if count else float("nan"),
    )


def _contacts_between(
    kernel: KernelModel, first_ids: tuple[int, ...], second_ids: tuple[int, ...]
) -> tuple[int, float]:
    """Count contacts and return the maximum resultant force between sets."""
    first = set(first_ids)
    second = set(second_ids)
    count = 0
    maximum_force = 0.0
    for contact_index in range(kernel.data.ncon):
        contact = kernel.data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if not (
            (geom1 in first and geom2 in second)
            or (geom2 in first and geom1 in second)
        ):
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(
            kernel.model, kernel.data, contact_index, local
        )
        count += 1
        maximum_force = max(maximum_force, float(np.linalg.norm(local[:3])))
    return count, maximum_force


def _relevant_contact_distances(
    kernel: KernelModel,
) -> tuple[float, float, tuple[str, str] | None, tuple[str, str] | None]:
    """Return all moving-kernel and body/environment minimum distances."""
    moving_bodies = set(kernel.mimo_body_ids) | {kernel.target_body_id}
    mimo_bodies = set(kernel.mimo_body_ids)
    minimum_relevant = 0.0
    minimum_body_environment = 0.0
    relevant_pair = None
    body_environment_pair = None
    for contact_index in range(kernel.data.ncon):
        contact = kernel.data.contact[contact_index]
        body1 = int(kernel.model.geom_bodyid[int(contact.geom1)])
        body2 = int(kernel.model.geom_bodyid[int(contact.geom2)])
        names = (
            kernel.model.geom(int(contact.geom1)).name or "unnamed",
            kernel.model.geom(int(contact.geom2)).name or "unnamed",
        )
        if (
            (body1 in moving_bodies or body2 in moving_bodies)
            and not (body1 in mimo_bodies and body2 in mimo_bodies)
        ):
            if float(contact.dist) < minimum_relevant:
                minimum_relevant = float(contact.dist)
                relevant_pair = names
        if (body1 in mimo_bodies) != (body2 in mimo_bodies):
            if float(contact.dist) < minimum_body_environment:
                minimum_body_environment = float(contact.dist)
                body_environment_pair = names
    return (
        minimum_relevant,
        minimum_body_environment,
        relevant_pair,
        body_environment_pair,
    )


def _apply_pd(
    kernel: KernelModel,
    joint_targets: dict[int, float],
    root_targets: np.ndarray,
    *,
    gentle_arm: bool = False,
    grasped_arm: bool = False,
) -> np.ndarray:
    model, data = kernel.model, kernel.data
    data.qfrc_applied[:] = 0.0
    for index, (qpos_id, dof_id) in enumerate(zip(kernel.root_qpos_ids, kernel.root_dof_ids)):
        force = (700.0 if index < 2 else 260.0) * (root_targets[index] - data.qpos[qpos_id])
        force -= (95.0 if index < 2 else 35.0) * data.qvel[dof_id]
        data.qfrc_applied[dof_id] = np.clip(force, -450.0, 450.0)
    for joint_id, target in joint_targets.items():
        qpos_id = int(model.jnt_qposadr[joint_id])
        dof_id = int(model.jnt_dofadr[joint_id])
        name = model.joint(joint_id).name or ""
        if joint_id in kernel.finger_joint_ids:
            kp, kd, limit = 0.35, 0.025, 0.08
        elif joint_id in kernel.head_joint_ids:
            kp, kd, limit = 25.0, 0.60, 3.0
        elif joint_id in kernel.posture_joint_ids:
            kp, kd, limit = 80.0, 1.20, 8.0
        elif grasped_arm:
            kp, kd, limit = 80.0, 1.20, 8.0
        else:
            kp, kd, limit = (10.0, 1.0, 1.5) if gentle_arm else (25.0, 0.55, 3.0)
        torque = kp * (target - data.qpos[qpos_id]) - kd * data.qvel[dof_id]
        data.qfrc_applied[dof_id] += np.clip(torque, -limit, limit)
    return data.qfrc_applied.copy()


def _apply_hand_task_control(
    kernel: KernelModel,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    gentle: bool,
    grasped: bool,
    tracking_gain_scale: float = 1.0,
) -> None:
    """Apply bounded operational-space control through articulated arm joints."""
    model, data = kernel.model, kernel.data
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacBody(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        kernel.right_hand_body_id,
    )
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        kernel.right_hand_body_id,
        velocity,
        0,
    )
    current_rotation = data.xmat[kernel.right_hand_body_id].reshape(3, 3)
    if grasped:
        position_kp, position_kd, force_limit = 180.0, 10.0, 22.0
        # Wrist orientation is governed by the bounded joint waypoint.  A
        # simultaneous task-space orientation objective over-constrains this
        # seven-DoF chain near its wrist limit and can select a discontinuous
        # elbow posture, so the grasped task controller stabilizes translation.
        rotation_kp, rotation_kd, torque_limit = 0.0, 0.0, 0.0
        joint_limit = 6.0
    elif gentle:
        position_kp, position_kd, force_limit = 70.0, 8.0, 7.0
        rotation_kp, rotation_kd, torque_limit = 2.0, 0.20, 1.0
        joint_limit = 2.5
    else:
        position_kp, position_kd, force_limit = 120.0, 7.0, 14.0
        rotation_kp, rotation_kd, torque_limit = 3.0, 0.20, 1.5
        joint_limit = 4.0
    if tracking_gain_scale <= 0.0:
        raise ValueError("tracking_gain_scale must be positive")
    position_kp *= tracking_gain_scale
    position_kd *= np.sqrt(tracking_gain_scale)
    force_limit *= tracking_gain_scale
    joint_limit *= tracking_gain_scale
    task_force = np.clip(
        position_kp * (target_position - data.xpos[kernel.right_hand_body_id])
        - position_kd * velocity[3:],
        -force_limit,
        force_limit,
    )
    task_torque = np.clip(
        rotation_kp * _rotation_error(target_rotation, current_rotation)
        - rotation_kd * velocity[:3],
        -torque_limit,
        torque_limit,
    )
    arm_dofs = np.asarray(kernel.arm_dof_ids)
    task_joint_torque = (
        jacobian_position[:, arm_dofs].T @ task_force
        + jacobian_rotation[:, arm_dofs].T @ task_torque
    )
    data.qfrc_applied[arm_dofs] = np.clip(
        data.qfrc_applied[arm_dofs] + task_joint_torque,
        -joint_limit,
        joint_limit,
    )


def _desired_hand(
    phase: dict[str, Any],
    episode_time: float,
    initial_hand: np.ndarray,
    target: np.ndarray,
    spec: dict[str, Any],
) -> tuple[np.ndarray, float]:
    alpha = _smooth((episode_time - phase["start_s"]) / (phase["end_s"] - phase["start_s"]))
    name = phase["phase"]
    miss = target + np.asarray([-0.15, 0.08, 0.02])
    contact = target + np.asarray([-0.072, 0.0, 0.015])
    lifted = contact + np.asarray([0.015, 0.0, 0.23])
    if name in {"look", "reorient", "approach"}:
        return initial_hand, 0.0
    if name == "reach_past_distractor":
        return initial_hand * (1 - alpha) + miss * alpha, 0.0
    if name == "fingertip_contact":
        return miss * (1 - alpha) + contact * alpha, 0.15 * alpha
    if name == "grasp":
        if alpha < 0.32:
            cycle = 0.15 + 0.50 * alpha / 0.32
        elif alpha < 0.45:
            cycle = 0.65 - 0.40 * (alpha - 0.32) / 0.13
        elif alpha < 0.75:
            cycle = 0.25 + 0.40 * (alpha - 0.45) / 0.30
        else:
            cycle = 0.65
        return contact, cycle
    if name == "lift":
        return contact * (1 - alpha) + lifted * alpha, 0.20
    if name == "inspect":
        return lifted + np.asarray([0.010, -0.005, 0.0]), 0.20
    if name == "rotate":
        return lifted + np.asarray([0.010, -0.005, 0.0]), 0.20
    if name == "inspect_rotate":
        return lifted + np.asarray([0.010 * alpha, -0.005 * alpha, 0.005 * np.sin(np.pi * alpha)]), 0.20
    if name == "head_turn_maintain_contact":
        return lifted + np.asarray([0.010, -0.005, 0.0]), 0.20
    if name == "shake":
        controller = spec.get("controller", {})
        frequency = float(controller.get("shake_frequency_hz", 1.5))
        amplitude = float(controller.get("shake_vertical_amplitude_m", 0.018))
        return (
            lifted
            + np.asarray(
                [
                    0.010,
                    -0.005,
                    amplitude
                    * np.sin(
                        2.0
                        * np.pi
                        * frequency
                        * (episode_time - phase["start_s"])
                    ),
                ]
            ),
            0.20,
        )
    if name == "bang":
        controller = spec.get("controller", {})
        cycles = int(controller.get("bang_cycles", 2))
        amplitude = float(controller.get("bang_clearance_amplitude_m", 0.025))
        inspect_position = lifted + np.asarray([0.010, -0.005, 0.0])
        high_position = contact + np.asarray([0.010, -0.005, amplitude])
        if alpha < 0.25:
            descend = _smooth(alpha / 0.25)
            return (
                inspect_position * (1.0 - descend)
                + high_position * descend,
                0.20,
            )
        progress = (alpha - 0.25) / 0.75
        clearance = amplitude * (
            0.5 + 0.5 * np.cos(2.0 * np.pi * cycles * progress)
        )
        return contact + np.asarray([0.010, -0.005, clearance]), 0.20
    if name == "transfer":
        controller = spec.get("controller", {})
        distance = float(controller.get("transfer_lateral_distance_m", 0.055))
        amplitude = float(controller.get("bang_clearance_amplitude_m", 0.025))
        return (
            contact
            + np.asarray(
                [0.010, -0.005 + distance * np.sin(np.pi * alpha), amplitude]
            ),
            0.20,
        )
    if name == "release":
        retract = _smooth(max(0.0, (alpha - 0.55) / 0.45))
        if _duration(spec) > 30.0:
            amplitude = float(
                spec.get("controller", {}).get(
                    "bang_clearance_amplitude_m", 0.025
                )
            )
            return (
                contact
                + np.asarray(
                    [
                        0.010,
                        -0.005 - 0.10 * retract,
                        amplitude + 0.04 * retract,
                    ]
                ),
                0.20 * max(0.0, 1.0 - alpha / 0.45),
            )
        return lifted + np.asarray([0.010, -0.005 - 0.10 * retract, 0.04 * retract]), 0.20 * max(0.0, 1.0 - alpha / 0.45)
    if name in {"settle", "retrieve_reorient"}:
        if _duration(spec) > 30.0:
            return contact + np.asarray([0.010, -0.105, 0.065]), 0.0
    if name in {
        "retrieve",
        "retrieve_grasp",
        "retrieve_lift",
        "retrieve_inspect",
        "final_dwell",
    }:
        return initial_hand, 0.0
    return lifted + np.asarray([0.010, -0.105, 0.04]), 0.0


def _duration(spec: dict[str, Any]) -> float:
    return float(spec["activity"]["duration_s"] if "activity" in spec else value(spec, "duration_s"))


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    # acos(trace(R)) loses several orders of precision close to identity.  The
    # chordal form is stable at the immutable-mount tolerance (1e-9 radians).
    chord = np.linalg.norm(second - first, ord="fro") / (2.0 * np.sqrt(2.0))
    return float(2.0 * np.arcsin(np.clip(chord, 0.0, 1.0)))


def run_physics_trace(
    kernel: KernelModel,
    spec: dict[str, Any],
    *,
    truth_hz: int = 60,
    physics_hz: int = 240,
    settle_steps: int = 240,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model, data = kernel.model, kernel.data
    model.opt.timestep = 1.0 / physics_hz
    initial_collision_hash = kernel.collision_policy_sha256
    posture_targets = {
        joint_id: float(data.qpos[int(model.jnt_qposadr[joint_id])])
        for joint_id in kernel.posture_joint_ids
    }
    posture_targets[model.joint("kernel_robot:chest_lean").id] = np.deg2rad(4.0)
    base_head_target = np.zeros(len(kernel.head_joint_ids), dtype=np.float64)
    head_targets = dict(zip(kernel.head_joint_ids, base_head_target))
    arm_targets = {
        joint_id: float(data.qpos[int(model.jnt_qposadr[joint_id])])
        for joint_id in kernel.arm_joint_ids
    }
    for _ in range(settle_steps):
        joints = {**posture_targets, **head_targets, **arm_targets, **_finger_targets(kernel, 0.0)}
        _apply_pd(kernel, joints, np.zeros(3))
        mujoco.mj_step(model, data)
    data.qpos[list(kernel.root_qpos_ids)] = 0.0
    data.qvel[list(kernel.root_dof_ids)] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)

    initial_hand = data.xpos[kernel.right_hand_body_id].copy()
    target_center = data.xpos[kernel.target_body_id].copy()
    initial_target = target_center.copy()
    initial_target_rotation = data.xmat[kernel.target_body_id].reshape(3, 3).copy()
    initial_hand_rotation = data.xmat[kernel.right_hand_body_id].reshape(3, 3).copy()
    side_grasp_rotation = np.column_stack(
        [
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([0.0, -1.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
        ]
    )
    relative_grasp_rotation = side_grasp_rotation @ initial_hand_rotation.T
    relative_angle = _rotation_angle(np.eye(3), relative_grasp_rotation)
    relative_axis = np.asarray(
        [
            relative_grasp_rotation[2, 1] - relative_grasp_rotation[1, 2],
            relative_grasp_rotation[0, 2] - relative_grasp_rotation[2, 0],
            relative_grasp_rotation[1, 0] - relative_grasp_rotation[0, 1],
        ]
    ) / (2.0 * np.sin(relative_angle))
    desired_hand_rotation = (
        _axis_angle_rotation(relative_axis, 0.55 * relative_angle)
        @ initial_hand_rotation
    )
    initial_arm_target = data.qpos[list(kernel.arm_qpos_ids)].copy()
    # Calibrate the IK branch to the collision-settled articulated posture in
    # the selected furnished scene.  Using the XML rest pose here can select a
    # different elbow branch and miss the target even when joint tracking is
    # exact.
    waypoint_seed = data.qpos.copy()
    waypoint_seed[list(kernel.root_qpos_ids)] = np.asarray([0.11, 0.0, 0.0])
    miss_position = target_center + np.asarray([-0.15, 0.08, 0.02])
    contact_position = target_center + np.asarray([-0.072, 0.0, 0.015])
    lift_position = contact_position + np.asarray([0.015, 0.0, 0.23])
    miss_target = solve_arm_ik(
        kernel, waypoint_seed, miss_position, initial_hand_rotation, iterations=140
    )
    contact_target = solve_arm_ik(
        kernel, waypoint_seed, contact_position, desired_hand_rotation, iterations=140
    )
    waypoint_seed[list(kernel.arm_qpos_ids)] = contact_target
    lift_waypoints = []
    for alpha in np.linspace(0.0, 1.0, 31):
        position = contact_position * (1.0 - alpha) + lift_position * alpha
        arm = solve_arm_ik(
            kernel, waypoint_seed, position, desired_hand_rotation, iterations=60
        )
        waypoint_seed[list(kernel.arm_qpos_ids)] = arm
        lift_waypoints.append(arm)
    lift_waypoints = np.asarray(lift_waypoints)
    lift_target = lift_waypoints[-1]
    inspect_waypoints = np.repeat(lift_target[None, :], 31, axis=0)
    wrist_index = 4
    wrist_joint = kernel.arm_joint_ids[wrist_index]
    inspect_waypoints[:, wrist_index] = np.clip(
        lift_target[wrist_index] + np.deg2rad(35.0) * np.linspace(0.0, 1.0, 31),
        model.jnt_range[wrist_joint, 0],
        model.jnt_range[wrist_joint, 1],
    )
    inspect_target = inspect_waypoints[-1]
    ik_target = initial_arm_target.copy()
    control_stride = physics_hz // truth_hz
    total_steps = int(round(_duration(spec) * physics_hz))
    trace: dict[str, list[Any]] = {
        key: [] for key in (
            "time_s", "phase", "behavior_action", "qpos", "qvel", "action", "touch_wrench",
            "touch_contact_count", "distinct_finger_contacts", "assist_active",
            "locomotion_assist_active", "vestibular_accelerometer",
            "vestibular_gyroscope", "vestibular_kinematic_accelerometer",
            "vestibular_kinematic_gyroscope", "vestibular_position", "target_pose",
            "hand_pose", "head_pose", "camera_pose", "body_pose",
            "reach_distractor_pose",
            "touch_contact_position", "touch_minimum_distance_m",
            "support_contact_count", "support_contact_force_n",
            "minimum_relevant_contact_distance_m",
            "minimum_body_environment_contact_distance_m",
            "near_miss_clearance_m",
            "camera_mount_translation_error_m", "camera_mount_rotation_error_rad",
            "world_reset",
        )
    }
    local_camera_rotation = np.empty(9)
    mujoco.mju_quat2Mat(local_camera_rotation, kernel.camera_mount_quaternion)
    local_camera_rotation = local_camera_rotation.reshape(3, 3)
    assist_active = False
    assist_qualified_time = 0.0
    assist_contact_qualified = False
    maximum_multipoint_contact_duration = 0.0
    contact_window_finger_bodies: set[int] = set()
    assist_relative_position = np.zeros(3)
    assist_relative_rotation = np.eye(3)
    assist_engagement_time: float | None = None
    assist_engagement_times: list[float] = []
    assist_release_times: list[float] = []
    assist_pose_jump_m = 0.0
    assist_pose_jump_degrees = 0.0
    assist_before_position: np.ndarray | None = None
    assist_before_rotation: np.ndarray | None = None
    first_contact_time: float | None = None
    maximum_force = 0.0
    minimum_penetration = 0.0
    minimum_relevant_penetration = 0.0
    minimum_body_environment_penetration = 0.0
    minimum_relevant_pair: tuple[str, str] | None = None
    minimum_body_environment_pair: tuple[str, str] | None = None
    minimum_relevant_time_s: float | None = None
    minimum_body_environment_time_s: float | None = None
    maximum_lift = 0.0
    physical_lift_before_assist = 0.0
    current_grasp_physical_lift = 0.0
    current_grasp_start_height = float(initial_target[2])
    head_turn_contact_samples = 0
    head_turn_samples = 0
    near_miss_separations: list[float] = []
    near_miss_contacts = 0
    last_action = np.zeros(len(kernel.root_qpos_ids) + len(kernel.arm_joint_ids) + len(kernel.finger_joint_ids))
    attention_head_target = base_head_target.copy()
    previous_phase_key: tuple[str, float] | None = None
    retrieve_start_arm = initial_arm_target.copy()
    retrieve_contact_target = contact_target.copy()
    retrieve_contact_position = contact_position.copy()
    retrieve_start_hand_position = initial_hand.copy()
    retrieve_lift_waypoints = lift_waypoints.copy()
    retrieve_lift_start_height = float(initial_target[2])
    retrieve_inspect_target = inspect_target.copy()
    shake_anchor_position = lift_position.copy()
    shake_anchor_object_position = initial_target.copy()
    shake_anchor_arm = inspect_target.copy()
    shake_low_target = inspect_target.copy()
    shake_high_target = inspect_target.copy()
    bang_start_position = lift_position.copy()
    bang_start_object_position = initial_target.copy()
    bang_start_arm = inspect_target.copy()
    bang_low_position = contact_position.copy()
    bang_high_position = contact_position.copy()
    bang_low_target = contact_target.copy()
    bang_high_target = contact_target.copy()
    transfer_start_position = contact_position.copy()
    transfer_end_position = contact_position.copy()
    transfer_start_arm = contact_target.copy()
    transfer_out_target = contact_target.copy()
    transfer_end_target = contact_target.copy()
    release_start_position = contact_position.copy()
    release_start_arm = contact_target.copy()
    release_retract_target = miss_target.copy()
    approach_phase = _named_phase(spec, "approach")
    if approach_phase is None:
        raise ValueError("episode schedule must contain an approach phase")
    attention_stride = (
        control_stride
        if _duration(spec) <= 30.0
        else max(control_stride, physics_hz // 10)
    )

    def root_target_at(time_s: float) -> np.ndarray:
        active = phase_at(spec, time_s)
        if active["phase"] == "approach":
            alpha = _smooth(
                (time_s - active["start_s"])
                / (active["end_s"] - active["start_s"])
            )
            forward = 0.11 * alpha
        else:
            forward = 0.11 if time_s >= approach_phase["end_s"] else 0.0
        target = np.asarray([forward, 0.0, 0.0])
        if active["phase"] == "reorient":
            alpha = _smooth(
                (time_s - active["start_s"])
                / (active["end_s"] - active["start_s"])
            )
            target[2] = np.deg2rad(-8.0) * np.sin(np.pi * alpha)
        return target

    def record() -> None:
        nonlocal head_turn_contact_samples, head_turn_samples
        phase_definition = phase_at(spec, float(data.time))
        phase = phase_definition["phase"]
        head_rotation = data.xmat[kernel.head_body_id].reshape(3, 3)
        expected_position = data.xpos[kernel.head_body_id] + head_rotation @ kernel.camera_mount_position
        expected_rotation = head_rotation @ local_camera_rotation
        camera_rotation = data.cam_xmat[kernel.camera_id].reshape(3, 3)
        translation_error = float(np.linalg.norm(data.cam_xpos[kernel.camera_id] - expected_position))
        rotation_error = _rotation_angle(expected_rotation, camera_rotation)
        wrench, contacts, finger_bodies, contact_distance = _target_contacts(kernel)
        support_contacts, support_force = _contacts_between(
            kernel, kernel.target_geom_ids, kernel.support_geom_ids
        )
        relevant_distance, body_environment_distance, _, _ = (
            _relevant_contact_distances(kernel)
        )
        contact_position = np.full(3, np.nan)
        hand_set = set(kernel.hand_geom_ids)
        target_set = set(kernel.target_geom_ids)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair.intersection(hand_set) and pair.intersection(target_set):
                contact_position = contact.pos.copy()
                break
        if phase == "head_turn_maintain_contact":
            head_turn_samples += 1
            head_turn_contact_samples += int(contacts > 0 or assist_active)
        trace["time_s"].append(float(data.time))
        trace["phase"].append(phase)
        trace["behavior_action"].append(_phase_action(phase_definition))
        trace["qpos"].append(data.qpos.copy())
        trace["qvel"].append(data.qvel.copy())
        trace["action"].append(last_action.copy())
        trace["touch_wrench"].append(wrench.copy())
        trace["touch_contact_count"].append(contacts)
        trace["distinct_finger_contacts"].append(len(finger_bodies))
        trace["assist_active"].append(assist_active)
        trace["locomotion_assist_active"].append(phase == "approach")
        for sensor_name, sensor_slice in kernel.sensor_slices.items():
            trace[sensor_name].append(data.sensordata[sensor_slice].copy())
        site_velocity = np.zeros(6)
        site_acceleration = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_SITE,
            kernel.vestibular_site_id, site_velocity, 1,
        )
        mujoco.mj_objectAcceleration(
            model, data, mujoco.mjtObj.mjOBJ_SITE,
            kernel.vestibular_site_id, site_acceleration, 1,
        )
        trace["vestibular_kinematic_accelerometer"].append(site_acceleration[3:].copy())
        trace["vestibular_kinematic_gyroscope"].append(site_velocity[:3].copy())
        trace["vestibular_position"].append(data.site_xpos[kernel.vestibular_site_id].copy())
        trace["target_pose"].append(np.concatenate([data.xpos[kernel.target_body_id], data.xquat[kernel.target_body_id]]))
        trace["hand_pose"].append(np.concatenate([data.xpos[kernel.right_hand_body_id], data.xquat[kernel.right_hand_body_id]]))
        trace["head_pose"].append(np.concatenate([data.xpos[kernel.head_body_id], data.xquat[kernel.head_body_id]]))
        trace["camera_pose"].append(np.concatenate([data.cam_xpos[kernel.camera_id], camera_rotation.reshape(9)]))
        trace["body_pose"].append(np.concatenate([data.xpos[list(kernel.mimo_body_ids)], data.xquat[list(kernel.mimo_body_ids)]], axis=1))
        trace["reach_distractor_pose"].append(
            data.geom_xpos[kernel.reach_distractor_geom_id].copy()
        )
        trace["touch_contact_position"].append(contact_position)
        trace["touch_minimum_distance_m"].append(contact_distance)
        trace["support_contact_count"].append(support_contacts)
        trace["support_contact_force_n"].append(support_force)
        trace["minimum_relevant_contact_distance_m"].append(relevant_distance)
        trace["minimum_body_environment_contact_distance_m"].append(
            body_environment_distance
        )
        trace["near_miss_clearance_m"].append(
            minimum_hand_target_distance(
                model,
                data,
                kernel.hand_geom_ids,
                kernel.reach_distractor_geom_id,
            ).signed_distance_m
            if phase == "reach_past_distractor"
            else np.nan
        )
        trace["camera_mount_translation_error_m"].append(translation_error)
        trace["camera_mount_rotation_error_rad"].append(rotation_error)
        trace["world_reset"].append(False)

    record()
    started = time.perf_counter()
    for step in range(total_steps):
        episode_time = step / physics_hz
        phase = phase_at(spec, episode_time)
        phase_alpha = _smooth((episode_time - phase["start_s"]) / (phase["end_s"] - phase["start_s"]))
        root_target = root_target_at(episode_time)
        next_time = min(_duration(spec), episode_time + 1.0 / physics_hz)
        next_root_target = root_target_at(next_time)
        data.qpos[list(kernel.root_qpos_ids)] = root_target
        data.qvel[list(kernel.root_dof_ids)] = (
            next_root_target - root_target
        ) * physics_hz
        mujoco.mj_forward(model, data)
        phase_name = phase["phase"]
        phase_key = (phase_name, float(phase["start_s"]))
        if phase_key != previous_phase_key:
            action_rotation = desired_hand_rotation
            if phase_name in {"grasp", "retrieve_grasp"}:
                assist_qualified_time = 0.0
                assist_contact_qualified = False
                contact_window_finger_bodies = set()
                current_grasp_physical_lift = 0.0
                current_grasp_start_height = float(
                    data.xpos[kernel.target_body_id, 2]
                )
            if phase_name == "shake":
                shake_anchor_position = data.xpos[
                    kernel.right_hand_body_id
                ].copy()
                shake_anchor_object_position = data.xpos[
                    kernel.target_body_id
                ].copy()
                shake_anchor_arm = data.qpos[
                    list(kernel.arm_qpos_ids)
                ].copy()
                shake_rotation = data.xmat[
                    kernel.right_hand_body_id
                ].reshape(3, 3).copy()
                shake_amplitude = float(
                    spec.get("controller", {}).get(
                        "shake_vertical_amplitude_m", 0.018
                    )
                )
                shake_seed = data.qpos.copy()
                shake_high_target = solve_arm_ik(
                    kernel,
                    shake_seed,
                    shake_anchor_position
                    + np.asarray([0.0, 0.0, shake_amplitude]),
                    shake_rotation,
                    iterations=140,
                )
                shake_seed[list(kernel.arm_qpos_ids)] = shake_high_target
                shake_low_target = solve_arm_ik(
                    kernel,
                    shake_seed,
                    shake_anchor_position
                    - np.asarray([0.0, 0.0, shake_amplitude]),
                    shake_rotation,
                    iterations=140,
                )
            if phase_name == "bang":
                bang_start_position = data.xpos[
                    kernel.right_hand_body_id
                ].copy()
                bang_start_object_position = data.xpos[
                    kernel.target_body_id
                ].copy()
                bang_start_arm = data.qpos[list(kernel.arm_qpos_ids)].copy()
                held_offset = (
                    data.xpos[kernel.target_body_id]
                    - data.xpos[kernel.right_hand_body_id]
                )
                bang_low_position = initial_target - held_offset
                bang_high_position = bang_low_position + np.asarray(
                    [
                        0.0,
                        0.0,
                        float(
                            spec.get("controller", {}).get(
                                "bang_clearance_amplitude_m", 0.025
                            )
                        ),
                    ]
                )
                bang_seed = data.qpos.copy()
                bang_high_target = solve_arm_ik(
                    kernel,
                    bang_seed,
                    bang_high_position,
                    action_rotation,
                    iterations=180,
                )
                bang_seed[list(kernel.arm_qpos_ids)] = bang_high_target
                bang_low_target = solve_arm_ik(
                    kernel,
                    bang_seed,
                    bang_low_position,
                    action_rotation,
                    iterations=180,
                )
            if phase_name == "transfer":
                transfer_start_position = data.xpos[
                    kernel.right_hand_body_id
                ].copy()
                transfer_start_arm = data.qpos[
                    list(kernel.arm_qpos_ids)
                ].copy()
                held_offset = (
                    data.xpos[kernel.target_body_id]
                    - data.xpos[kernel.right_hand_body_id]
                )
                transfer_end_position = initial_target - held_offset
                transfer_end_position[2] += float(
                    spec.get("controller", {}).get(
                        "bang_clearance_amplitude_m", 0.025
                    )
                )
                transfer_out_position = transfer_end_position + np.asarray(
                    [
                        0.0,
                        float(
                            spec.get("controller", {}).get(
                                "transfer_lateral_distance_m", 0.055
                            )
                        ),
                        0.0,
                    ]
                )
                transfer_seed = data.qpos.copy()
                transfer_out_target = solve_arm_ik(
                    kernel,
                    transfer_seed,
                    transfer_out_position,
                    action_rotation,
                    iterations=180,
                )
                transfer_seed[list(kernel.arm_qpos_ids)] = transfer_out_target
                transfer_end_target = solve_arm_ik(
                    kernel,
                    transfer_seed,
                    transfer_end_position,
                    action_rotation,
                    iterations=180,
                )
            if phase_name == "release":
                release_start_position = data.xpos[
                    kernel.right_hand_body_id
                ].copy()
                release_start_arm = data.qpos[
                    list(kernel.arm_qpos_ids)
                ].copy()
                release_seed = data.qpos.copy()
                release_retract_target = solve_arm_ik(
                    kernel,
                    release_seed,
                    release_start_position + np.asarray([0.0, -0.10, 0.04]),
                    action_rotation,
                    iterations=180,
                )
            if phase_name == "retrieve":
                retrieve_start_arm = data.qpos[
                    list(kernel.arm_qpos_ids)
                ].copy()
                retrieve_start_hand_position = data.xpos[
                    kernel.right_hand_body_id
                ].copy()
                retrieve_contact_position = data.xpos[
                    kernel.target_body_id
                ].copy() + np.asarray([-0.065, 0.040, 0.050])
                retrieve_seed = data.qpos.copy()
                retrieve_contact_target = solve_arm_ik(
                    kernel,
                    retrieve_seed,
                    retrieve_contact_position,
                    desired_hand_rotation,
                    iterations=180,
                )
            if phase_name == "retrieve_lift":
                retrieve_lift_start_height = float(
                    data.xpos[kernel.target_body_id, 2]
                )
                retrieve_seed = data.qpos.copy()
                start_position = data.xpos[kernel.right_hand_body_id].copy()
                retrieve_lift_waypoints_rows = []
                for lift_alpha in np.linspace(0.0, 1.0, 31):
                    position = start_position + np.asarray(
                        [0.0, 0.0, 0.14 * lift_alpha]
                    )
                    arm = solve_arm_ik(
                        kernel,
                        retrieve_seed,
                        position,
                        desired_hand_rotation,
                        iterations=60,
                    )
                    retrieve_seed[list(kernel.arm_qpos_ids)] = arm
                    retrieve_lift_waypoints_rows.append(arm)
                retrieve_lift_waypoints = np.asarray(
                    retrieve_lift_waypoints_rows
                )
                retrieve_inspect_target = retrieve_lift_waypoints[-1].copy()
                retrieve_inspect_target[wrist_index] = np.clip(
                    retrieve_inspect_target[wrist_index] + np.deg2rad(20.0),
                    model.jnt_range[wrist_joint, 0],
                    model.jnt_range[wrist_joint, 1],
                )
            previous_phase_key = phase_key

        hand_position, closure = _desired_hand(
            phase, episode_time, initial_hand, target_center, spec
        )
        if phase_name == "shake":
            frequency = float(
                spec.get("controller", {}).get("shake_frequency_hz", 1.5)
            )
            amplitude = float(
                spec.get("controller", {}).get(
                    "shake_vertical_amplitude_m", 0.018
                )
            )
            desired_object_position = shake_anchor_object_position + np.asarray(
                [
                    0.0,
                    0.0,
                    amplitude
                    * np.sin(
                        2.0
                        * np.pi
                        * frequency
                        * (episode_time - phase["start_s"])
                    ),
                ]
            )
            hand_position = data.xpos[kernel.right_hand_body_id] + (
                desired_object_position - data.xpos[kernel.target_body_id]
            )
            closure = 0.20
        elif phase_name == "bang":
            amplitude = float(
                spec.get("controller", {}).get(
                    "bang_clearance_amplitude_m", 0.025
                )
            )
            cycles = int(spec.get("controller", {}).get("bang_cycles", 2))
            if phase_alpha < 0.25:
                descend = _smooth(phase_alpha / 0.25)
                desired_object_position = (
                    bang_start_object_position * (1.0 - descend)
                    + (initial_target + np.asarray([0.0, 0.0, amplitude]))
                    * descend
                )
            else:
                progress = (phase_alpha - 0.25) / 0.75
                clearance = amplitude * (
                    0.5
                    + 0.5 * np.cos(2.0 * np.pi * cycles * progress)
                )
                desired_object_position = initial_target + np.asarray(
                    [0.0, 0.0, clearance]
                )
            hand_position = data.xpos[kernel.right_hand_body_id] + (
                desired_object_position - data.xpos[kernel.target_body_id]
            )
            closure = 0.20
        elif phase_name == "transfer":
            distance = float(
                spec.get("controller", {}).get(
                    "transfer_lateral_distance_m", 0.055
                )
            )
            desired_object_position = initial_target + np.asarray(
                [
                    0.0,
                    distance * np.sin(np.pi * phase_alpha),
                    float(
                        spec.get("controller", {}).get(
                            "bang_clearance_amplitude_m", 0.025
                        )
                    ),
                ]
            )
            hand_position = data.xpos[kernel.right_hand_body_id] + (
                desired_object_position - data.xpos[kernel.target_body_id]
            )
            closure = 0.20
        elif phase_name == "release" and _duration(spec) > 30.0:
            retract = _smooth(max(0.0, (phase_alpha - 0.55) / 0.45))
            if phase_alpha < 0.30:
                desired_object_position = initial_target + np.asarray(
                    [
                        0.0,
                        0.0,
                        float(
                            spec.get("controller", {}).get(
                                "bang_clearance_amplitude_m", 0.025
                            )
                        ),
                    ]
                )
                hand_position = data.xpos[kernel.right_hand_body_id] + (
                    desired_object_position - data.xpos[kernel.target_body_id]
                )
            else:
                hand_position = release_start_position + np.asarray(
                    [0.0, -0.10 * retract, 0.04 * retract]
                )
            closure = 0.20 * max(0.0, 1.0 - phase_alpha / 0.45)
        elif phase_name == "retrieve":
            hand_position = (
                retrieve_start_hand_position * (1.0 - phase_alpha)
                + retrieve_contact_position * phase_alpha
            )
            closure = 0.15 * phase_alpha
        elif phase_name == "retrieve_grasp":
            hand_position = retrieve_contact_position
            if phase_alpha < 0.32:
                closure = 0.15 + 0.50 * phase_alpha / 0.32
            elif phase_alpha < 0.45:
                closure = 0.65 - 0.40 * (phase_alpha - 0.32) / 0.13
            elif phase_alpha < 0.75:
                closure = 0.25 + 0.40 * (phase_alpha - 0.45) / 0.30
            else:
                closure = 0.65
        elif phase_name == "retrieve_lift":
            hand_position = (
                retrieve_contact_position
                + np.asarray([0.0, 0.0, 0.14 * phase_alpha])
            )
            closure = 0.20
        elif phase_name in {"retrieve_inspect", "final_dwell"}:
            hand_position = retrieve_contact_position + np.asarray(
                [0.0, 0.0, 0.14]
            )
            closure = 0.20
        head_values = base_head_target.copy()
        if phase["phase"] == "look":
            head_values[0] += np.deg2rad(-8.0) * np.sin(np.pi * phase_alpha)
        elif phase["phase"] == "reorient":
            head_values[0] += np.deg2rad(10.0) * np.sin(np.pi * phase_alpha)
        elif phase["phase"] == "reach_past_distractor":
            if step % attention_stride == 0:
                attention_head_target = solve_head_attention(
                    kernel,
                    data.qpos.copy(),
                    0.5
                    * (
                        data.xpos[kernel.right_hand_body_id]
                        + data.geom_xpos[kernel.reach_distractor_geom_id]
                    ),
                )
            attention_alpha = _smooth(min(1.0, phase_alpha / 0.50))
            head_values = (
                base_head_target * (1.0 - attention_alpha)
                + attention_head_target * attention_alpha
            )
        elif phase["phase"] == "fingertip_contact":
            return_alpha = _smooth(min(1.0, phase_alpha / 0.45))
            head_values = (
                attention_head_target * (1.0 - return_alpha)
                + base_head_target * return_alpha
            )
        elif phase["phase"] == "head_turn_maintain_contact":
            turn = np.sin(np.pi * phase_alpha)
            head_values[0] += np.deg2rad(22.0) * turn
            head_values[1] += np.deg2rad(5.0) * turn
            head_values[2] -= np.deg2rad(5.0) * turn
        elif phase["phase"] in {
            "shake",
            "bang",
            "transfer",
            "retrieve_reorient",
            "retrieve",
            "retrieve_grasp",
            "retrieve_lift",
            "retrieve_inspect",
            "final_dwell",
        }:
            if step % attention_stride == 0:
                attention_head_target = solve_head_attention(
                    kernel,
                    data.qpos.copy(),
                    data.xpos[kernel.target_body_id].copy(),
                )
            attention_alpha = (
                _smooth(min(1.0, phase_alpha / 0.65))
                if phase["phase"]
                in {"retrieve_reorient", "retrieve", "retrieve_lift"}
                else 1.0
            )
            head_values = (
                base_head_target * (1.0 - attention_alpha)
                + attention_head_target * attention_alpha
            )
        elif phase["phase"] in {"release", "settle"}:
            if step % attention_stride == 0:
                attention_head_target = solve_head_attention(
                    kernel,
                    data.qpos.copy(),
                    data.xpos[kernel.target_body_id].copy(),
                )
            attention_alpha = (
                _smooth(min(1.0, phase_alpha / 0.65))
                if phase["phase"] == "release"
                else 1.0
            )
            head_values = (
                base_head_target * (1.0 - attention_alpha)
                + attention_head_target * attention_alpha
            )
        for index, joint_id in enumerate(kernel.head_joint_ids):
            head_targets[joint_id] = float(
                np.clip(
                    head_values[index],
                    model.jnt_range[joint_id, 0],
                    model.jnt_range[joint_id, 1],
                )
            )

        if phase_name in {"look", "reorient", "approach"}:
            ik_target = initial_arm_target
        elif phase_name == "reach_past_distractor":
            ik_target = initial_arm_target * (1.0 - phase_alpha) + miss_target * phase_alpha
        elif phase_name == "fingertip_contact":
            contact_alpha = _smooth(min(1.0, phase_alpha / 0.55))
            ik_target = miss_target * (1.0 - contact_alpha) + contact_target * contact_alpha
        elif phase_name == "grasp":
            ik_target = contact_target
        elif phase_name == "lift":
            ik_target = _interpolate_waypoints(lift_waypoints, phase_alpha)
        elif phase_name == "inspect":
            ik_target = lift_target
        elif phase_name == "rotate":
            ik_target = _interpolate_waypoints(inspect_waypoints, phase_alpha)
        elif phase_name == "inspect_rotate":
            ik_target = _interpolate_waypoints(inspect_waypoints, phase_alpha)
        elif phase_name == "head_turn_maintain_contact":
            ik_target = inspect_target
        elif phase_name == "shake":
            shake_fraction = 0.5 + 0.5 * np.sin(
                2.0
                * np.pi
                * float(
                    spec.get("controller", {}).get(
                        "shake_frequency_hz", 1.5
                    )
                )
                * (episode_time - phase["start_s"])
            )
            ik_target = (
                shake_low_target * (1.0 - shake_fraction)
                + shake_high_target * shake_fraction
            )
        elif phase_name == "bang":
            if phase_alpha < 0.25:
                descend = _smooth(phase_alpha / 0.25)
                ik_target = (
                    bang_start_arm * (1.0 - descend)
                    + bang_high_target * descend
                )
            else:
                progress = (phase_alpha - 0.25) / 0.75
                high_fraction = 0.5 + 0.5 * np.cos(
                    2.0
                    * np.pi
                    * int(spec.get("controller", {}).get("bang_cycles", 2))
                    * progress
                )
                ik_target = (
                    bang_low_target * (1.0 - high_fraction)
                    + bang_high_target * high_fraction
                )
        elif phase_name == "transfer":
            if phase_alpha < 0.5:
                fraction = _smooth(phase_alpha / 0.5)
                ik_target = (
                    transfer_start_arm * (1.0 - fraction)
                    + transfer_out_target * fraction
                )
            else:
                fraction = _smooth((phase_alpha - 0.5) / 0.5)
                ik_target = (
                    transfer_out_target * (1.0 - fraction)
                    + transfer_end_target * fraction
                )
        elif phase_name == "release":
            retract_alpha = _smooth(
                max(
                    0.0,
                    min(
                        1.0,
                        (
                            (phase_alpha - 0.55) / 0.45
                            if _duration(spec) > 30.0
                            else (phase_alpha - 0.55) / 0.45
                        ),
                    ),
                )
            )
            if _duration(spec) > 30.0:
                ik_target = (
                    release_start_arm * (1.0 - retract_alpha)
                    + release_retract_target * retract_alpha
                )
            else:
                ik_target = (
                    inspect_target * (1.0 - retract_alpha)
                    + miss_target * retract_alpha
                )
        elif phase_name in {"settle", "retrieve_reorient"} and _duration(
            spec
        ) > 30.0:
            ik_target = release_retract_target
        elif phase_name == "retrieve":
            ik_target = (
                retrieve_start_arm * (1.0 - phase_alpha)
                + retrieve_contact_target * phase_alpha
            )
        elif phase_name == "retrieve_grasp":
            ik_target = retrieve_contact_target
        elif phase_name == "retrieve_lift":
            ik_target = _interpolate_waypoints(
                retrieve_lift_waypoints, phase_alpha
            )
        elif phase_name in {"retrieve_inspect", "final_dwell"}:
            inspect_alpha = phase_alpha if phase_name == "retrieve_inspect" else 1.0
            ik_target = (
                retrieve_lift_waypoints[-1] * (1.0 - inspect_alpha)
                + retrieve_inspect_target * inspect_alpha
            )
        else:
            ik_target = miss_target
        arm_targets = dict(zip(kernel.arm_joint_ids, ik_target))
        joint_targets = {**posture_targets, **head_targets, **arm_targets, **_finger_targets(kernel, closure)}
        hand_target_center_distance = float(
            np.linalg.norm(
                data.xpos[kernel.right_hand_body_id]
                - data.xpos[kernel.target_body_id]
            )
        )
        applied = _apply_pd(
            kernel,
            joint_targets,
            root_target,
            gentle_arm=(
                phase_name
                in {
                    "fingertip_contact", "grasp", "lift", "inspect_rotate",
                    "inspect", "rotate", "head_turn_maintain_contact",
                    "shake", "bang", "transfer", "release", "retrieve",
                    "retrieve_grasp", "retrieve_lift", "retrieve_inspect",
                    "final_dwell",
                }
                and hand_target_center_distance < 0.11
                and not assist_active
            ),
            grasped_arm=assist_active,
        )
        desired_task_rotation = desired_hand_rotation
        if phase_name in {
            "rotate",
            "inspect_rotate",
            "head_turn_maintain_contact",
            "shake",
            "bang",
            "transfer",
            "release",
            "settle",
        }:
            rotation_alpha = (
                phase_alpha if phase_name in {"rotate", "inspect_rotate"} else 1.0
            )
            desired_task_rotation = desired_hand_rotation @ _axis_angle_rotation(
                np.asarray([1.0, 0.0, 0.0]),
                np.deg2rad(35.0) * rotation_alpha,
            )
        if assist_active or phase_name in {
            "fingertip_contact",
            "grasp",
            "retrieve",
            "retrieve_grasp",
        }:
            _apply_hand_task_control(
                kernel,
                hand_position,
                desired_task_rotation,
                gentle=False,
                grasped=assist_active,
                tracking_gain_scale=(
                    1.5
                    if _duration(spec) > 30.0
                    and phase_name
                    in {
                        "shake",
                        "transfer",
                        "release",
                        "retrieve",
                        "retrieve_grasp",
                        "retrieve_lift",
                        "retrieve_inspect",
                        "final_dwell",
                    }
                    else 1.0
                ),
            )
            applied = data.qfrc_applied.copy()

        wrench, contacts, finger_bodies, penetration = _target_contacts(kernel)
        (
            relevant_penetration,
            body_environment_penetration,
            relevant_pair,
            body_environment_pair,
        ) = _relevant_contact_distances(kernel)
        if np.isfinite(penetration):
            minimum_penetration = min(minimum_penetration, penetration)
        if relevant_penetration < minimum_relevant_penetration:
            minimum_relevant_penetration = relevant_penetration
            minimum_relevant_pair = relevant_pair
            minimum_relevant_time_s = episode_time
        if body_environment_penetration < minimum_body_environment_penetration:
            minimum_body_environment_penetration = body_environment_penetration
            minimum_body_environment_pair = body_environment_pair
            minimum_body_environment_time_s = episode_time
        maximum_force = max(maximum_force, float(np.linalg.norm(wrench[:3])))
        if contacts and first_contact_time is None:
            first_contact_time = episode_time
        if phase_name in {"grasp", "retrieve_grasp"}:
            if contacts > 0:
                assist_qualified_time += 1.0 / physics_hz
                contact_window_finger_bodies.update(finger_bodies)
                if len(contact_window_finger_bodies) >= 2:
                    maximum_multipoint_contact_duration = max(
                        maximum_multipoint_contact_duration, assist_qualified_time
                    )
                    assist_contact_qualified = (
                        assist_contact_qualified or assist_qualified_time >= 0.10
                    )
        two_attempts_complete = episode_time >= float(phase["start_s"]) + 1.15
        if not assist_active and phase_name in {"grasp", "retrieve_grasp"}:
            current_grasp_physical_lift = max(
                current_grasp_physical_lift,
                float(
                    data.xpos[kernel.target_body_id, 2]
                    - current_grasp_start_height
                ),
            )
            physical_lift_before_assist = max(
                physical_lift_before_assist, current_grasp_physical_lift
            )
        if (
            phase_name in {"grasp", "retrieve_grasp"}
            and two_attempts_complete
            and not assist_active
            and assist_contact_qualified
            and current_grasp_physical_lift < 0.02
        ):
            hand_rotation = data.xmat[kernel.right_hand_body_id].reshape(3, 3)
            assist_relative_position = hand_rotation.T @ (data.xpos[kernel.target_body_id] - data.xpos[kernel.right_hand_body_id])
            assist_relative_rotation = hand_rotation.T @ data.xmat[kernel.target_body_id].reshape(3, 3)
            assist_active = True
            if assist_engagement_time is None:
                assist_engagement_time = episode_time
            assist_engagement_times.append(episode_time)
            assist_before_position = data.xpos[kernel.target_body_id].copy()
            assist_before_rotation = data.xmat[kernel.target_body_id].reshape(3, 3).copy()
        was_assist_active = assist_active
        if phase_name == "settle" or (
            phase_name == "release" and phase_alpha >= 0.25
        ):
            assist_active = False
        if was_assist_active and not assist_active:
            assist_release_times.append(episode_time)
        if assist_active:
            hand_rotation = data.xmat[kernel.right_hand_body_id].reshape(3, 3)
            desired_object = data.xpos[kernel.right_hand_body_id] + hand_rotation @ assist_relative_position
            object_velocity = data.qvel[kernel.target_dof_adr : kernel.target_dof_adr + 3]
            hand_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY,
                kernel.right_hand_body_id, hand_velocity, 0,
            )
            force = (
                80.0 * (desired_object - data.xpos[kernel.target_body_id])
                + 9.0 * (hand_velocity[3:] - object_velocity)
                - float(model.body_subtreemass[kernel.target_body_id])
                * np.asarray(model.opt.gravity)
            )
            desired_rotation = hand_rotation @ assist_relative_rotation
            current_rotation = data.xmat[kernel.target_body_id].reshape(3, 3)
            angular_error = _rotation_error(desired_rotation, current_rotation)
            object_angular_velocity = data.qvel[
                kernel.target_dof_adr + 3 : kernel.target_dof_adr + 6
            ]
            torque = 1.5 * angular_error + 0.08 * (
                hand_velocity[:3] - object_angular_velocity
            )
            data.xfrc_applied[kernel.target_body_id, :3] = np.clip(force, -18.0, 18.0)
            data.xfrc_applied[kernel.target_body_id, 3:] = np.clip(torque, -1.5, 1.5)
        else:
            data.xfrc_applied[kernel.target_body_id] = 0.0
        last_action = np.concatenate([root_target, ik_target, np.fromiter(_finger_targets(kernel, closure).values(), dtype=float)])
        mujoco.mj_step(model, data)
        data.qpos[list(kernel.root_qpos_ids)] = next_root_target
        data.qvel[list(kernel.root_dof_ids)] = (
            next_root_target - root_target
        ) * physics_hz
        # mj_step integrates qpos after its dynamics pass; refresh all derived
        # kinematics so every recorded pose corresponds to the recorded qpos.
        mujoco.mj_forward(model, data)
        if assist_before_position is not None:
            assist_pose_jump_m = max(assist_pose_jump_m, float(np.linalg.norm(data.xpos[kernel.target_body_id] - assist_before_position)))
            if assist_before_rotation is not None:
                assist_pose_jump_degrees = max(
                    assist_pose_jump_degrees,
                    float(
                        np.rad2deg(
                            _rotation_angle(
                                assist_before_rotation,
                                data.xmat[kernel.target_body_id].reshape(3, 3),
                            )
                        )
                    ),
                )
            assist_before_position = None
            assist_before_rotation = None
        maximum_lift = max(maximum_lift, float(data.xpos[kernel.target_body_id, 2] - initial_target[2]))
        if phase["phase"] == "reach_past_distractor":
            separation = minimum_hand_target_distance(
                model, data, kernel.hand_geom_ids,
                kernel.reach_distractor_geom_id,
            ).signed_distance_m
            near_miss_separations.append(float(separation))
            hand_set = set(kernel.hand_geom_ids)
            near_miss_contacts += sum(
                1
                for contact_index in range(data.ncon)
                if {
                    int(data.contact[contact_index].geom1),
                    int(data.contact[contact_index].geom2),
                }.intersection(hand_set)
                and kernel.reach_distractor_geom_id
                in {
                    int(data.contact[contact_index].geom1),
                    int(data.contact[contact_index].geom2),
                }
            )
        if (step + 1) % control_stride == 0:
            record()
    wall_seconds = time.perf_counter() - started
    arrays = {key: np.asarray(items) for key, items in trace.items()}
    target_xyz = arrays["target_pose"][:, :3]
    target_speed = np.linalg.norm(arrays["qvel"][:, kernel.target_dof_adr : kernel.target_dof_adr + 3], axis=1)
    settle_mask = arrays["phase"] == "settle"
    head_turn_mask = arrays["phase"] == "head_turn_maintain_contact"
    head_rotations = []
    for quaternion in arrays["head_pose"][:, 3:7]:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        head_rotations.append(matrix.reshape(3, 3))
    head_turn_angle = 0.0
    indices = np.flatnonzero(head_turn_mask)
    if len(indices) > 1:
        reference = head_rotations[indices[0]]
        head_turn_angle = max(_rotation_angle(reference, head_rotations[index]) for index in indices)
    target_rotation_angle = 0.0
    for quaternion in arrays["target_pose"][:, 3:7]:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        target_rotation_angle = max(target_rotation_angle, _rotation_angle(initial_target_rotation, matrix.reshape(3, 3)))
    collision_rows_final = [
        [model.geom(i).name, int(model.geom_contype[i]), int(model.geom_conaffinity[i])]
        for i in kernel.relevant_collision_geom_ids
    ]
    collision_hash_final = hashlib.sha256(json.dumps(collision_rows_final, separators=(",", ":")).encode()).hexdigest()
    expected_acc = arrays["vestibular_kinematic_accelerometer"]
    expected_gyro = arrays["vestibular_kinematic_gyroscope"]
    accel_rmse = float(np.sqrt(np.mean((arrays["vestibular_accelerometer"] - expected_acc) ** 2)))
    gyro_rmse = float(np.sqrt(np.mean((arrays["vestibular_gyroscope"] - expected_gyro) ** 2)))
    settle_speeds = target_speed[settle_mask]
    settled_threshold = 0.08
    final_settled_samples = 0
    for speed in settle_speeds[::-1]:
        if speed > settled_threshold:
            break
        final_settled_samples += 1
    root_positions = arrays["qpos"][:, list(kernel.root_qpos_ids[:2])]
    root_velocity = np.gradient(root_positions, 1.0 / truth_hz, axis=0)
    root_acceleration = np.gradient(root_velocity, 1.0 / truth_hz, axis=0)
    penetration_limit = float(
        spec.get("frozen_gates", {}).get("penetration_depth_m_max", 0.002)
    )
    locomotion_mask = arrays["locomotion_assist_active"].astype(bool)
    locomotion_minimum_environment_distance = (
        float(
            arrays["minimum_body_environment_contact_distance_m"][
                locomotion_mask
            ].min()
        )
        if np.any(locomotion_mask)
        else 0.0
    )
    shake_mask = arrays["phase"] == "shake"
    bang_mask = arrays["phase"] == "bang"
    transfer_mask = arrays["phase"] == "transfer"
    retrieve_mask = np.isin(
        arrays["phase"],
        ("retrieve", "retrieve_grasp", "retrieve_lift", "retrieve_inspect"),
    )
    retrieve_lift_mask = arrays["phase"] == "retrieve_lift"
    target_steps = np.linalg.norm(np.diff(target_xyz, axis=0), axis=1)

    assist_intervals = []
    interval_start: float | None = None
    for index, active in enumerate(arrays["assist_active"].astype(bool)):
        if active and interval_start is None:
            interval_start = float(arrays["time_s"][index])
        if interval_start is not None and (
            not active or index == len(arrays["assist_active"]) - 1
        ):
            end_index = index if not active else index
            assist_intervals.append(
                {
                    "start_s": interval_start,
                    "end_s": float(arrays["time_s"][end_index]),
                }
            )
            interval_start = None

    unique_actions = sorted(set(arrays["behavior_action"].tolist()))
    grasp_episode_count = sum(
        phase["phase"] in {"grasp", "retrieve_grasp"}
        for phase in _phase_schedule(spec)
    )
    receipt = {
        "schema": "EmbodiedPhysicsQA",
        "duration_s": _duration(spec),
        "truth_samples": len(arrays["time_s"]),
        "physics_steps": total_steps,
        "wall_seconds": wall_seconds,
        "simulation_steps_per_wall_second": total_steps / wall_seconds,
        "camera_mount": {
            "immutable": True,
            "maximum_translation_error_m": float(arrays["camera_mount_translation_error_m"].max()),
            "maximum_rotation_error_rad": float(arrays["camera_mount_rotation_error_rad"].max()),
        },
        "collision_policy": {
            "initial_sha256": initial_collision_hash,
            "final_sha256": collision_hash_final,
            "unchanged": initial_collision_hash == collision_hash_final,
            "relevant_geom_count": len(kernel.relevant_collision_geom_ids),
            "all_relevant_enabled": all(model.geom_contype[i] and model.geom_conaffinity[i] for i in kernel.relevant_collision_geom_ids),
            "minimum_contact_distance_m": minimum_penetration,
            "minimum_relevant_contact_distance_m": minimum_relevant_penetration,
            "minimum_relevant_contact_pair": minimum_relevant_pair,
            "minimum_relevant_contact_time_s": minimum_relevant_time_s,
            "minimum_body_environment_contact_distance_m": minimum_body_environment_penetration,
            "minimum_body_environment_contact_pair": minimum_body_environment_pair,
            "minimum_body_environment_contact_time_s": minimum_body_environment_time_s,
            "persistent_penetration_frames": int(
                np.sum(
                    arrays["minimum_relevant_contact_distance_m"]
                    < -penetration_limit
                )
            ),
        },
        "near_miss": {
            "minimum_clearance_m": min(near_miss_separations) if near_miss_separations else None,
            "contact_substeps": near_miss_contacts,
        },
        "contact": {
            "first_contact_time_s": first_contact_time,
            "maximum_force_n": maximum_force,
            "maximum_distinct_finger_contacts": int(arrays["distinct_finger_contacts"].max()),
        },
        "grasp": {
            "physical_attempts": 2 * grasp_episode_count,
            "physical_attempts_per_grasp": 2,
            "grasp_episode_count": grasp_episode_count,
            "physical_lift_before_assist_m": physical_lift_before_assist,
            "assist_engaged": assist_engagement_time is not None,
            "assist_engagement_time_s": assist_engagement_time,
            "assist_engagement_times_s": assist_engagement_times,
            "assist_release_times_s": assist_release_times,
            "assist_intervals": assist_intervals,
            "assist_contact_gate_s": 0.10,
            "multipoint_contact_evidence_duration_s": maximum_multipoint_contact_duration,
            "multipoint_contact_duration_rule": "cumulative contacting substeps during the frozen grasp phase with at least two distinct finger bodies observed",
            "assist_pose_jump_m": assist_pose_jump_m,
            "assist_pose_jump_degrees": assist_pose_jump_degrees,
            "collisions_remained_enabled": initial_collision_hash == collision_hash_final,
        },
        "manipulation": {
            "maximum_lift_m": maximum_lift,
            "maximum_object_rotation_degrees": float(np.rad2deg(target_rotation_angle)),
            "maximum_head_turn_degrees": float(np.rad2deg(head_turn_angle)),
            "head_turn_contact_retention_fraction": head_turn_contact_samples / max(1, head_turn_samples),
            "released": bool(assist_release_times) or not bool(
                arrays["assist_active"][-1]
            ),
            "settle_peak_speed_m_s": float(settle_speeds.max()) if len(settle_speeds) else None,
            "final_settled_window_maximum_speed_m_s": (
                float(settle_speeds[-final_settled_samples:].max())
                if final_settled_samples else None
            ),
            "final_settled_duration_s": final_settled_samples / truth_hz,
        },
        "imu_consistency": {
            "accelerometer_rmse_m_s2": accel_rmse,
            "gyroscope_rmse_rad_s": gyro_rmse,
            "method": "MuJoCo site velocity/acceleration kinematics in the vestibular sensor frame on the shared truth clock",
        },
        "locomotion_assist": {
            "explicitly_labeled": True,
            "active_frames": int(arrays["locomotion_assist_active"].sum()),
            "maximum_root_speed_m_s": float(
                np.linalg.norm(root_velocity, axis=1).max()
            ),
            "maximum_root_acceleration_m_s2": float(
                np.linalg.norm(root_acceleration, axis=1).max()
            ),
            "minimum_environment_contact_distance_m": (
                locomotion_minimum_environment_distance
            ),
            "collision_checked": locomotion_minimum_environment_distance
            >= -penetration_limit,
        },
        "synchronization": {
            "truth_hz": truth_hz,
            "physics_hz": physics_hz,
            "all_frame_stream_lengths_equal": all(
                len(items) == len(arrays["time_s"])
                for items in arrays.values()
            ),
            "maximum_timestamp_error_s": float(np.max(np.abs(arrays["time_s"] - np.arange(len(arrays["time_s"])) / truth_hz))),
        },
        "object_identity": {
            "persistent_id": spec["scene_family"]["target"]["persistent_id"],
            "changes": 0,
        },
        "continuous_episode": {
            "single_physics_trace": True,
            "hidden_world_reset_count": int(arrays["world_reset"].sum()),
            "maximum_target_truth_step_m": (
                float(target_steps.max()) if len(target_steps) else 0.0
            ),
            "bounded_actions_observed": unique_actions,
            "shake_vertical_range_m": (
                float(np.ptp(target_xyz[shake_mask, 2]))
                if np.any(shake_mask)
                else 0.0
            ),
            "bang_support_contact_samples": int(
                np.sum(arrays["support_contact_count"][bang_mask] > 0)
            ),
            "bang_maximum_support_force_n": (
                float(arrays["support_contact_force_n"][bang_mask].max())
                if np.any(bang_mask)
                else 0.0
            ),
            "transfer_lateral_range_m": (
                float(np.ptp(target_xyz[transfer_mask, 1]))
                if np.any(transfer_mask)
                else 0.0
            ),
            "retrieve_contact_samples": int(
                np.sum(arrays["touch_contact_count"][retrieve_mask] > 0)
            ),
            "retrieve_lift_m": (
                float(
                    target_xyz[retrieve_lift_mask, 2].max()
                    - retrieve_lift_start_height
                )
                if np.any(retrieve_lift_mask)
                else 0.0
            ),
            "assist_interval_count": len(assist_intervals),
        },
        "direct_target_transform_after_initialization": False,
        "independent_camera_control": False,
        "hand_mocap_or_weld": False,
    }
    return arrays, receipt
