"""Minimal MuJoCo proof of a contact-triggered, pose-preserving weld grasp."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .controllers import activate_weld_preserving_current_pose

PROBE_XML = """
<mujoco model="contact_triggered_weld_probe">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <geom friction="1.0 0.02 0.002" solref="0.004 1"/>
  </default>
  <worldbody>
    <geom name="support" type="plane" size="1 1 0.01"/>
    <body name="hand" mocap="true" pos="-0.15 0 0.06">
      <geom name="hand_geom" type="sphere" size="0.025" mass="0.05"/>
    </body>
    <body name="target" pos="0 0 0.03">
      <freejoint name="target_free"/>
      <geom name="target_geom" type="box" size="0.03 0.03 0.03"
            mass="0.08" rgba="1 0.8 0 1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="grasp_weld" body1="hand" body2="target" active="false"
          solref="0.003 1"/>
  </equality>
</mujoco>
"""


@dataclass(frozen=True)
class WeldGraspProbeResult:
    mujoco_version: str
    timestep_s: float
    contact_time_s: float
    activation_time_s: float
    release_time_s: float
    pose_jump_at_activation_m: float
    maximum_target_height_m: float
    lift_m: float
    transport_m: float
    final_height_m: float
    final_speed_m_s: float
    weld_initially_inactive: bool
    triggering_contact_count: int
    contact_precedes_or_equals_activation: bool
    landed_stably_on_support: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _body_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def _pair_contact_count(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1_name: str,
    geom2_name: str,
) -> int:
    geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_name),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_name),
    }
    return sum(
        {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        == geom_ids
        for index in range(data.ncon)
    )


def run_weld_grasp_probe() -> WeldGraspProbeResult:
    """Approach, contact, weld, lift/transport, release, and settle a small box."""
    model = mujoco.MjModel.from_xml_string(PROBE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    equality_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld"
    )
    target_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "target_free"
    )
    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target"
    )
    target_qpos_address = int(model.jnt_qposadr[target_joint_id])
    target_qvel_address = int(model.jnt_dofadr[target_joint_id])
    initial_target_position = data.qpos[
        target_qpos_address : target_qpos_address + 3
    ].copy()

    contact_time = None
    activation_time = None
    release_time = None
    pose_jump = None
    maximum_height = float(initial_target_position[2])
    weld_initially_inactive = not bool(data.eq_active[equality_id])
    triggering_contact_count = 0

    # The deterministic approach ends with 1 mm proxy penetration, ensuring that
    # activation is causally gated by an actual hand-target contact.
    for step in range(125):
        alpha = (step + 1) / 125
        data.mocap_pos[0] = (-0.15 + alpha * 0.096, 0.0, 0.06)
        mujoco.mj_step(model, data)
        maximum_height = max(
            maximum_height, float(data.xpos[target_body_id, 2])
        )
        triggering_contact_count = _pair_contact_count(
            model, data, "hand_geom", "target_geom"
        )
        if triggering_contact_count:
            contact_time = float(data.time)
            pose_jump = activate_weld_preserving_current_pose(
                model, data, equality_id
            )
            activation_time = float(data.time)
            break
    if contact_time is None:
        raise RuntimeError("probe approach did not produce physical contact")

    grasp_start = data.mocap_pos[0].copy()
    for step in range(400):
        alpha = (step + 1) / 400
        data.mocap_pos[0] = grasp_start + np.array(
            [0.18 * alpha, 0.0, 0.14 * min(alpha / 0.5, 1.0)]
        )
        mujoco.mj_step(model, data)
        target_position, _ = _body_pose(model, data, "target")
        maximum_height = max(maximum_height, float(target_position[2]))

    # Stop the transport motion while the constraint remains active so release
    # does not impart a deliberate throw.
    for _ in range(200):
        mujoco.mj_step(model, data)
        target_position, _ = _body_pose(model, data, "target")
        maximum_height = max(maximum_height, float(target_position[2]))

    release_time = float(data.time)
    data.eq_active[equality_id] = 0
    # Pull the hand away immediately after release, then allow the target to settle.
    data.mocap_pos[0] += (0.0, 0.15, 0.08)
    for _ in range(1000):
        mujoco.mj_step(model, data)

    final_position = data.qpos[
        target_qpos_address : target_qpos_address + 3
    ].copy()
    final_speed = float(
        np.linalg.norm(data.qvel[target_qvel_address : target_qvel_address + 3])
    )
    transport = float(
        np.linalg.norm(final_position[:2] - initial_target_position[:2])
    )
    return WeldGraspProbeResult(
        mujoco_version=mujoco.__version__,
        timestep_s=float(model.opt.timestep),
        contact_time_s=contact_time,
        activation_time_s=activation_time,
        release_time_s=release_time,
        pose_jump_at_activation_m=float(pose_jump),
        maximum_target_height_m=maximum_height,
        lift_m=maximum_height - float(initial_target_position[2]),
        transport_m=transport,
        final_height_m=float(final_position[2]),
        final_speed_m_s=final_speed,
        weld_initially_inactive=weld_initially_inactive,
        triggering_contact_count=triggering_contact_count,
        contact_precedes_or_equals_activation=contact_time <= activation_time,
        landed_stably_on_support=(
            abs(float(final_position[2]) - 0.03) <= 0.003
            and final_speed <= 0.01
        ),
    )
