"""Physics controller primitives shared by spec-driven prototype episodes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import mujoco
import numpy as np


@dataclass(frozen=True)
class SeparationSample:
    signed_distance_m: float
    hand_geom_id: int
    target_geom_id: int
    hand_witness_xyz: tuple[float, float, float]
    target_witness_xyz: tuple[float, float, float]


def minimum_hand_target_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geom_ids: Iterable[int],
    target_geom_id: int,
    *,
    distmax_m: float = 1.0,
) -> SeparationSample:
    """Return the exact closest MuJoCo collision-proxy pair and witnesses."""
    best: SeparationSample | None = None
    for hand_geom_id in sorted(hand_geom_ids):
        fromto = np.zeros(6, dtype=np.float64)
        distance = float(
            mujoco.mj_geomDistance(
                model,
                data,
                int(hand_geom_id),
                int(target_geom_id),
                float(distmax_m),
                fromto,
            )
        )
        sample = SeparationSample(
            signed_distance_m=distance,
            hand_geom_id=int(hand_geom_id),
            target_geom_id=int(target_geom_id),
            hand_witness_xyz=tuple(float(value) for value in fromto[:3]),
            target_witness_xyz=tuple(float(value) for value in fromto[3:]),
        )
        if best is None or (distance, hand_geom_id) < (
            best.signed_distance_m,
            best.hand_geom_id,
        ):
            best = sample
    if best is None:
        raise ValueError("hand_geom_ids must be non-empty")
    return best


class GraspState(str, Enum):
    APPROACH = "approach"
    CONTACT_SEEN = "contact_seen"
    GRASP_ACTIVE = "grasp_active"
    RELEASED = "released"


@dataclass
class ContactTriggeredGrasp:
    """Transparent grasp state machine; the runner owns equality activation."""

    state: GraspState = GraspState.APPROACH
    contact_time_s: float | None = None
    engagement_time_s: float | None = None
    release_time_s: float | None = None

    def observe_contact(self, *, time_s: float, contact_count: int) -> bool:
        if self.state is GraspState.APPROACH and contact_count > 0:
            self.state = GraspState.CONTACT_SEEN
            self.contact_time_s = float(time_s)
            return True
        return False

    def engage(self, *, time_s: float) -> None:
        if self.state is not GraspState.CONTACT_SEEN:
            raise RuntimeError("grasp equality cannot engage before measured contact")
        self.state = GraspState.GRASP_ACTIVE
        self.engagement_time_s = float(time_s)

    def release(self, *, time_s: float) -> None:
        if self.state is not GraspState.GRASP_ACTIVE:
            raise RuntimeError("only an active grasp can release")
        self.state = GraspState.RELEASED
        self.release_time_s = float(time_s)

    def receipt(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "contact_time_s": self.contact_time_s,
            "engagement_time_s": self.engagement_time_s,
            "release_time_s": self.release_time_s,
            "contact_precedes_or_equals_engagement": (
                self.contact_time_s is not None
                and self.engagement_time_s is not None
                and self.contact_time_s <= self.engagement_time_s
            ),
        }


def activate_weld_preserving_current_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    equality_id: int,
) -> float:
    """Materialize the current body-relative pose, then activate a weld."""
    mujoco.mj_forward(model, data)
    body1_id = int(model.eq_obj1id[equality_id])
    body2_id = int(model.eq_obj2id[equality_id])
    rotation1 = data.xmat[body1_id].reshape(3, 3)
    relative_position = rotation1.T @ (
        data.xpos[body2_id] - data.xpos[body1_id]
    )
    inverse_quaternion1 = data.xquat[body1_id].copy()
    inverse_quaternion1[1:] *= -1
    relative_quaternion = np.empty(4)
    mujoco.mju_mulQuat(
        relative_quaternion, inverse_quaternion1, data.xquat[body2_id]
    )
    model.eq_data[equality_id, 3:6] = relative_position
    model.eq_data[equality_id, 6:10] = relative_quaternion
    position_before = data.xpos[body2_id].copy()
    data.eq_active[equality_id] = 1
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(data.xpos[body2_id] - position_before))
