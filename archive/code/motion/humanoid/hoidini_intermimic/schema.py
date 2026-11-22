"""The upstream InterMimic SMPL-X reference tensor, not a new motion format."""

import numpy as np

FPS = 30
WIDTH = 591
FIELDS = {
    "root_pos": slice(0, 3), "root_rot": slice(3, 7),
    "dof_pos": slice(9, 162), "body_pos": slice(162, 318),
    "obj_pos": slice(318, 321), "obj_rot": slice(321, 325),
    "contact_obj": slice(330, 331), "contact_human": slice(331, 383),
    "body_rot": slice(383, 591),
}


def validate_source(arrays):
    shapes = {"poses": (156,), "trans": (3,), "joints": (52, 3),
              "poses_obj": (3,), "trans_obj": (3,), "contact": (60,)}
    frames = len(arrays.get("poses", []))
    if frames < 2:
        raise ValueError("At least two HOIDiNi motion frames are required")
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != (frames, *shape):
            raise ValueError(f"Invalid HOIDiNi {name} shape; expected {(frames, *shape)}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Non-finite HOIDiNi {name}")
    return frames


def validate_reference(data):
    if data.ndim != 2 or data.shape[1] != WIDTH or len(data) < 2:
        raise ValueError("InterMimic requires a frames × 591 reference tensor")
    if not np.isfinite(data).all():
        raise ValueError("Non-finite InterMimic reference")
    for key in ("root_rot", "obj_rot", "body_rot"):
        q = data[:, FIELDS[key]].reshape(len(data), -1, 4)
        if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=2e-6):
            raise ValueError(f"Non-unit {key} quaternion")
    if np.any(data[:, 7:9]) or np.any(data[:, 325:330]):
        raise ValueError("Upstream reserved columns must remain zero")
    if not np.isin(data[:, FIELDS["contact_human"]], [-1, 0, 1]).all():
        raise ValueError("Invalid upstream human contact labels")
    if not np.isin(data[:, FIELDS["contact_obj"]], [0, 1]).all():
        raise ValueError("Invalid upstream object contact labels")
    if not np.allclose(data[:, :3], data[:, 162:165], atol=1e-6):
        raise ValueError("Root and pelvis positions disagree")
