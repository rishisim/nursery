"""Validation helpers for the canonical reduced Unity anatomical rig manifest."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class JacobianColumnAgreement:
    joint_id: str
    analytic_m_per_rad: tuple[float, float, float]
    finite_difference_m_per_rad: tuple[float, float, float]
    direction_error_deg: float
    relative_magnitude_error: float


def load_manifest(path: str | Path = "configs/embodied_simulation_anatomical_rig.json") -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    if manifest["schema"] != "embodied.anatomical_rig.v1":
        raise ValueError("unexpected anatomical rig schema")
    if manifest["source"]["license"] != "CC0":
        raise ValueError("the canonical avatar must retain its CC0 provenance")
    authority = manifest["authority"]
    if authority["object_attachment"] or authority["object_pose_writes_after_initialization"]:
        raise ValueError("object assistance is forbidden")
    if authority["independent_animation"]:
        raise ValueError("the visible skin may not have independent animation")
    clock = manifest["clock"]
    if clock["physics_hz"] // clock["render_hz"] != clock["steps_per_frame"] or clock["physics_hz"] % clock["render_hz"]:
        raise ValueError("physics/render clocks require an exact integer mapping")
    landmarks = manifest["landmarks_m"]
    required = {"shoulder", "elbow", "wrist"}
    required |= {f"{digit}_{part}" for digit in manifest["digits"] for part in range(1, 4)}
    if required - landmarks.keys():
        raise ValueError(f"missing measured landmarks: {sorted(required - landmarks.keys())}")
    for joint in manifest["joints"]:
        axis = np.asarray(joint["axis_parent"], dtype=float)
        if not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1) > 1e-3:
            raise ValueError(f"{joint['id']} axis is not a verified unit vector")
        if joint["anchor"] not in landmarks:
            raise ValueError(f"{joint['id']} has an unknown anchor")
        if joint["limits_deg"][0] >= joint["limits_deg"][1]:
            raise ValueError(f"{joint['id']} has invalid limits")
        for key in ("mass_kg", "damping", "stiffness", "force_limit"):
            if joint[key] <= 0:
                raise ValueError(f"{joint['id']} has invalid {key}")


def _rotation(axis: np.ndarray, radians: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(radians) * skew + (1 - math.cos(radians)) * (skew @ skew)


def arm_fk(manifest: dict, joint_radians: np.ndarray, rest_site: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Serial product-of-rotations FK from measured rest anchors.

    Axes are expressed in each joint's parent frame. The returned joint rows are
    world-space (anchor, axis) pairs and are the source for the analytic Jacobian.
    """
    joints = manifest["joints"]
    if len(joint_radians) != len(joints):
        raise ValueError("joint vector length does not match the arm manifest")
    rotation = np.eye(3)
    translation = np.zeros(3)
    previous_rest_anchor = np.asarray(manifest["landmarks_m"][joints[0]["anchor"]], dtype=float)
    world_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for joint, angle in zip(joints, joint_radians, strict=True):
        rest_anchor = np.asarray(manifest["landmarks_m"][joint["anchor"]], dtype=float)
        translation += rotation @ (rest_anchor - previous_rest_anchor)
        world_anchor = np.asarray(manifest["landmarks_m"][joints[0]["anchor"]], dtype=float) + translation
        world_axis = rotation @ np.asarray(joint["axis_parent"], dtype=float)
        world_rows.append((world_anchor, world_axis))
        rotation = rotation @ _rotation(np.asarray(joint["axis_parent"], dtype=float), float(angle))
        previous_rest_anchor = rest_anchor
    site = np.asarray(manifest["landmarks_m"][joints[0]["anchor"]], dtype=float) + translation + rotation @ (rest_site - previous_rest_anchor)
    return site, rotation, world_rows


def analytic_position_jacobian(manifest: dict, joint_radians: np.ndarray, rest_site: np.ndarray) -> np.ndarray:
    site, _, rows = arm_fk(manifest, joint_radians, rest_site)
    return np.column_stack([np.cross(axis, site - anchor) for anchor, axis in rows])


def compare_central_difference(manifest: dict, joint_radians: np.ndarray, rest_site: np.ndarray, epsilon: float = 1e-5) -> list[JacobianColumnAgreement]:
    analytic = analytic_position_jacobian(manifest, joint_radians, rest_site)
    finite = np.zeros_like(analytic)
    for column in range(len(joint_radians)):
        plus, minus = joint_radians.copy(), joint_radians.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        finite[:, column] = (arm_fk(manifest, plus, rest_site)[0] - arm_fk(manifest, minus, rest_site)[0]) / (2 * epsilon)
    result = []
    for index, joint in enumerate(manifest["joints"]):
        a, f = analytic[:, index], finite[:, index]
        denom = max(np.linalg.norm(a), np.linalg.norm(f), 1e-12)
        cosine = float(np.clip(np.dot(a, f) / max(np.linalg.norm(a) * np.linalg.norm(f), 1e-15), -1, 1))
        direction = 0.0 if np.linalg.norm(a) < 1e-10 and np.linalg.norm(f) < 1e-10 else math.degrees(math.acos(cosine))
        result.append(JacobianColumnAgreement(joint["id"], tuple(a), tuple(f), direction, abs(np.linalg.norm(a) - np.linalg.norm(f)) / denom))
    return result
