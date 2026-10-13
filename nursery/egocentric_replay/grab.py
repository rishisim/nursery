"""Reconstruct GRAB using its subject template and PCA hand convention."""

from pathlib import Path

import numpy as np

from .sessions import contact_rule


def load_clip(path):
    with np.load(path, allow_pickle=True) as archive:
        return {key: value.item() if value.shape == () else value
                for key, value in archive.items()}


def mesh_path(root, stored):
    stored = str(stored)
    for prefix in ("tools/object_meshes/", "tools/subject_meshes/"):
        if stored.startswith(prefix):
            stored = stored.removeprefix(prefix)
            break
    path = (Path(root) / stored).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f"Invalid GRAB mesh reference: {stored}")
    return path


def rigid_matrix(params, frame):
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(params["global_orient"][frame]).as_matrix()
    matrix[:3, 3] = params["transl"][frame]
    return matrix


def reconstruct(data, indices, config, hand_ids):
    import smplx
    import torch
    import trimesh
    from smplx.lbs import batch_rodrigues

    torch.set_num_threads(1)
    template = trimesh.load(mesh_path(config["grab_root"], data["body"]["vtemp"]), process=False)
    model = smplx.create(config["body_models"], model_type="smplx", ext="npz",
                         gender=data["gender"], v_template=np.asarray(template.vertices),
                         use_pca=True, num_pca_comps=int(data["n_comps"]),
                         flat_hand_mean=True, batch_size=len(indices))
    params = {key: torch.tensor(value[indices], dtype=torch.float32)
              for key, value in data["body"]["params"].items()}
    with torch.no_grad():
        result = model(**params, return_full_pose=True)
        rotations = batch_rodrigues(result.full_pose.reshape(-1, 3)).reshape(len(indices), -1, 3, 3)
    head_rotation = np.tile(np.eye(3), (len(indices), 1, 1))
    # SMPL-X kinematic parents, rather than global orientation alone, carry head motion.
    ancestors, joint = [], 15
    while joint >= 0:
        ancestors.append(joint)
        joint = int(model.parents[joint])
    for joint in reversed(ancestors):
        head_rotation = head_rotation @ rotations[:, joint].numpy()
    vertices = result.vertices.numpy()
    cameras = np.tile(np.eye(4), (len(indices), 1, 1))
    # SMPL-X rest coordinates face +Z with +Y up; Blender looks along -Z.
    basis = np.diag([-1, 1, -1])
    cameras[:, :3, :3] = head_rotation @ basis
    cameras[:, :3, 3] = result.joints[:, 15].numpy() + np.einsum(
        "nij,j->ni", head_rotation, [0, 0.05, 0.13])
    left, right = contact_rule(data["contact"]["body"][indices], hand_ids)
    return vertices, model.faces, cameras, left, right


def contact_distances(data, indices, vertices, hand_ids, root):
    import trimesh

    obj = trimesh.load(mesh_path(root, data["object"]["object_mesh"]), process=False)
    labelled, whole = [], []
    for position, source in enumerate(indices):
        transform = rigid_matrix(data["object"]["params"], source)
        for ids in hand_ids.values():
            contact = data["contact"]["body"][source, ids] != 0
            if not contact.any():
                continue
            points = (vertices[position, ids] - transform[:3, 3]) @ transform[:3, :3]
            distances = trimesh.proximity.closest_point(obj, points)[1]
            labelled.extend(distances[contact].tolist())
            whole.extend(distances.tolist())
    return labelled, whole
