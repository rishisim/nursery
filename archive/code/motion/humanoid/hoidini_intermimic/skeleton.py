"""Transfer SMPL-X joint angles into an explicit upstream SMPL-X skeleton."""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .coordinates import continuous_quaternions

# HOIDiNi's 22 body + 30 finger order; jaw and eye joints are absent.
SOURCE_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
    "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck", "L_Thorax",
    "R_Thorax", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
] + [f"{side}_{finger}{joint}" for side in ("L", "R")
     for finger in ("Index", "Middle", "Pinky", "Ring", "Thumb") for joint in (1, 2, 3)]
SOURCE_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
for start, wrist in ((22, 20), (37, 21)):
    for joint in range(start, start + 15, 3):
        SOURCE_PARENTS.extend([wrist, joint, joint + 1])

# InterAct interact2mimic.py upright_start uses global_R @ inverse(B).
# This changes body-local axes, not the scene's world coordinates.
BASIS = Rotation.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()


def read_skeleton(path):
    root = ET.parse(path).getroot()
    bodies = root.findall("./worldbody/body")
    if len(bodies) != 1:
        raise ValueError("Expected one upstream SMPL-X humanoid root")
    names, parents, offsets = [], [], []

    def visit(body, parent):
        name = body.get("name")
        index = len(names)
        names.append(name)
        parents.append(parent)
        offset = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError(f"Invalid body offset: {name}")
        offsets.append(offset)
        if any(k in body.attrib for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
            raise ValueError("Rotated MJCF rest frames are not supported")
        joints = body.findall("joint")
        if parent == -1:
            if body.find("freejoint") is None:
                raise ValueError("Expected a free pelvis root")
        elif len(joints) != 3:
            raise ValueError(f"Expected three SMPL-X rotation DOFs: {name}")
        else:
            for axis, joint in enumerate(joints):
                if joint.get("name") != f"{name}_{'xyz'[axis]}" or joint.get("type", "hinge") != "hinge":
                    raise ValueError(f"Unsupported DOF order: {name}")
                if not np.array_equal(np.fromstring(joint.get("axis", ""), sep=" "), np.eye(3)[axis]):
                    raise ValueError(f"Unsupported joint axis: {name}")
                if joint.get("pos", "0 0 0") != "0 0 0":
                    raise ValueError("Offset joint pivots are not supported")
        for child in body.findall("body"):
            visit(child, index)

    visit(bodies[0], -1)
    if len(names) != 52 or set(names) != set(SOURCE_NAMES) or names[0] != "Pelvis":
        raise ValueError("Expected the complete upstream 52-body SMPL-X skeleton")
    order = np.array([SOURCE_NAMES.index(name) for name in names])
    for i in range(1, 52):
        if names[parents[i]] != SOURCE_NAMES[SOURCE_PARENTS[order[i]]]:
            raise ValueError(f"Unexpected skeleton parent: {names[i]}")
    return names, np.asarray(parents), np.asarray(offsets), order


def map_skeleton(local_source_rotations, root_positions, skeleton):
    names, parents, offsets, order = skeleton
    source_global = np.empty_like(local_source_rotations)
    for joint, parent in enumerate(SOURCE_PARENTS):
        source_global[:, joint] = (local_source_rotations[:, joint] if parent < 0 else
                                  source_global[:, parent] @ local_source_rotations[:, joint])
    global_rotations = source_global[:, order] @ BASIS.T
    local = np.empty_like(global_rotations)
    positions = np.empty((len(root_positions), 52, 3))
    for joint, parent in enumerate(parents):
        if parent < 0:
            local[:, joint] = global_rotations[:, joint]
            positions[:, joint] = root_positions
        else:
            local[:, joint] = global_rotations[:, parent].transpose(0, 2, 1) @ global_rotations[:, joint]
            positions[:, joint] = positions[:, parent] + (global_rotations[:, parent] @ offsets[joint])
    dofs = Rotation.from_matrix(local[:, 1:].reshape(-1, 3, 3)).as_rotvec().reshape(len(root_positions), 153)
    return dofs, positions, continuous_quaternions(global_rotations)
