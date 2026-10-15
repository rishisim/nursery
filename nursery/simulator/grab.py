"""Reconstruct HOIDiNi neutral-body retargeted GRAB motion."""

import functools
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
    matrix[:3, :3] = Rotation.from_rotvec(params["global_orient"][frame]).as_matrix().T
    matrix[:3, 3] = params["transl"][frame]
    return matrix


def reconstruct(data, indices, config, hand_ids):
    import smplx
    import torch
    from smplx.lbs import batch_rodrigues

    torch.set_num_threads(1)
    model = smplx.create(config["body_models"], model_type="smplx", ext="npz",
                         gender="neutral", use_pca=False, flat_hand_mean=True,
                         batch_size=len(indices))
    source = {key: data["body"]["params"][key]
              for key in ("global_orient", "body_pose", "transl")}
    source.update(left_hand_pose=data["lhand"]["params"]["fullpose"],
                  right_hand_pose=data["rhand"]["params"]["fullpose"])
    params = {key: torch.tensor(value[indices], dtype=torch.float32)
              for key, value in source.items()}
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
    return vertices, model.faces, cameras, left, right, result.joints[:, :55].numpy()


def contact_distances(data, indices, vertices, hand_ids, root, vertex_ids=None, labelled_only=False, surface=None):
    triangles, surface = _object_surface(data, root) if surface is None else surface
    labelled, whole = [], []
    for position, source in enumerate(indices):
        transform = rigid_matrix(data["object"]["params"], source)
        for ids in hand_ids.values():
            contact = data["contact"]["body"][source, ids] != 0
            if not contact.any():
                continue
            local_ids = ids if vertex_ids is None else np.searchsorted(vertex_ids, ids)
            points = (vertices[position, local_ids] - transform[:3, 3]) @ transform[:3, :3]
            if labelled_only:
                points = points[contact]
            distances = np.concatenate([_surface_distances(triangles, surface, chunk)
                for chunk in np.array_split(points, max(1, len(points) // 64))])
            labelled.extend((distances if labelled_only else distances[contact]).tolist())
            if not labelled_only:
                whole.extend(distances.tolist())
    return labelled, whole


def contact_indices(data, rendered, hand_ids):
    ids = np.concatenate(list(hand_ids.values()))
    labelled = np.flatnonzero((data["contact"]["body"][:, ids] != 0).any(axis=1))
    selected = labelled[np.unique(np.linspace(0, len(labelled) - 1, min(200, len(labelled)), dtype=int))]
    one_hz = np.arange(0, len(data["contact"]["body"]), max(1, int(round(data["framerate"]))))
    return np.unique(np.r_[rendered, one_hz, selected])


def hand_geometry(data, hand_ids, gate_indices, config, model=None):
    import torch
    from smplx.lbs import batch_rigid_transform, batch_rodrigues, vertices2joints

    torch.set_num_threads(1)
    if model is None:
        model = _body_model(config)
    ids = np.unique(np.concatenate(list(hand_ids.values())))
    template = model.v_template[ids]
    posedirs = model.posedirs.reshape(-1, len(model.v_template), 3)[:, ids].reshape(-1, len(ids) * 3)
    joints = vertices2joints(model.J_regressor, model.v_template[None])
    weights = model.lbs_weights[ids]
    count = len(data["body"]["params"]["transl"])
    hands = np.empty((count, len(ids), 3), dtype=np.float32)
    # Skin only hand vertices to retain exact full-rate clearance without a dense body pass.
    with torch.no_grad():
        for start in range(0, count, 128):
            stop = min(start + 128, count)
            body = data["body"]["params"]
            pose = np.concatenate([body["global_orient"][start:stop], body["body_pose"][start:stop],
                                   np.zeros((stop - start, 9)), data["lhand"]["params"]["fullpose"][start:stop],
                                   data["rhand"]["params"]["fullpose"][start:stop]], axis=1)
            rotations = batch_rodrigues(torch.tensor(pose, dtype=torch.float32).reshape(-1, 3)).reshape(-1, 55, 3, 3)
            feature = (rotations[:, 1:] - torch.eye(3)).reshape(stop - start, -1)
            posed = template + (feature @ posedirs).reshape(stop - start, len(ids), 3)
            _, transforms = batch_rigid_transform(rotations, joints.expand(stop - start, -1, -1), model.parents)
            skin = (weights @ transforms.reshape(stop - start, 55, 16)).reshape(-1, len(ids), 4, 4)
            homogeneous = torch.cat([posed, torch.ones(stop - start, len(ids), 1)], dim=-1)
            vertices = (skin @ homogeneous[..., None])[:, :, :3, 0]
            vertices += torch.tensor(body["transl"][start:stop], dtype=torch.float32)[:, None]
            hands[start:stop] = vertices.numpy()
    bounds = []
    for hand in hand_ids.values():
        points = hands[:, np.searchsorted(ids, hand)]
        bounds.extend(np.stack([points.min(axis=1), points.max(axis=1)], axis=1).tolist())
    return hands[gate_indices], ids, bounds, 0.0



def _surface_index(triangles):
    from scipy.spatial import cKDTree

    tree = cKDTree(triangles.mean(axis=1), leafsize=16)
    bounds, children, leaves = [], [], []

    def visit(node):
        index = len(bounds)
        values = triangles[node.indices]
        bounds.append([values.min(axis=(0, 1)), values.max(axis=(0, 1))])
        children.append([-1, -1])
        leaves.append(np.resize(node.indices, 16) if node.split_dim < 0 else np.zeros(16, dtype=int))
        if node.split_dim >= 0:
            children[index] = [visit(node.lesser), visit(node.greater)]
        return index

    visit(tree.tree)
    return np.array(bounds), np.array(children), np.array(leaves), cKDTree(triangles.reshape(-1, 3))


def _surface_distances(triangles, surface, points):
    from trimesh.triangles import closest_point

    bounds, children, leaves, vertices = surface
    result = vertices.query(points)[0] ** 2
    nodes, rows = np.zeros(len(points), dtype=int), np.arange(len(points))
    while len(nodes):
        gap = np.maximum(np.maximum(bounds[nodes, 0] - points[rows], points[rows] - bounds[nodes, 1]), 0)
        keep = np.sum(gap * gap, axis=1) <= result[rows] + 1e-15
        nodes, rows = nodes[keep], rows[keep]
        leaf = children[nodes, 0] < 0
        if leaf.any():
            query = np.repeat(points[rows[leaf]], 16, axis=0)
            nearest = closest_point(triangles[leaves[nodes[leaf]]].reshape(-1, 3, 3), query)
            distance = np.sum((nearest - query) ** 2, axis=1).reshape(-1, 16).min(axis=1)
            np.minimum.at(result, rows[leaf], distance)
        # Node AABBs enclose complete triangles, so pruning preserves exact surface distances.
        rows = np.repeat(rows[~leaf], 2)
        nodes = children[nodes[~leaf]].ravel()
    return np.sqrt(result)


def _body_model(config):
    import smplx

    return smplx.create(config["body_models"], model_type="smplx", ext="npz",
                        gender="neutral", use_pca=False, flat_hand_mean=True, batch_size=1)


def _object_surface(data, root):
    return _surface_for(mesh_path(root, data["object"]["object_mesh"]))


@functools.lru_cache(maxsize=None)
def _surface_for(path):
    import trimesh

    mesh = trimesh.load(path, process=False)
    triangles = np.asarray(mesh.triangles)
    return triangles, _surface_index(triangles)
