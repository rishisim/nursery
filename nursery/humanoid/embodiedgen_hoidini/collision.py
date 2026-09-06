"""Dependency-light collision checks for prepared EmbodiedGen scenes.

This module is an offline validator, not a physics engine or a guarantee that a
soft optimization loss will find a collision-free motion.  Closed meshes use
ray-parity signs.  Open meshes are checked as two-sided clearance surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .prepare_scene import PreparedSceneBundle, TriangleMesh, _load_obj


@dataclass(frozen=True)
class CollisionGeometry:
    node_name: str
    mesh: TriangleMesh
    watertight: bool
    source_kind: str = "collision"


@dataclass(frozen=True)
class CollisionSample:
    frame: float
    moving_geometry: str
    obstacle: str
    minimum_distance_m: float
    maximum_penetration_m: float
    colliding_vertices: int


def is_watertight(mesh: TriangleMesh) -> bool:
    """Return whether every non-degenerate undirected edge has two faces."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0 or np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        return False
    edges = np.sort(
        np.concatenate((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(len(counts) and np.all(counts == 2))


def build_static_collision_scene(
    bundle: PreparedSceneBundle,
) -> tuple[CollisionGeometry, ...]:
    """Load transformed room collision meshes, excluding the moving target."""
    result = []
    target = bundle.active_interaction.target_name
    for name in sorted(bundle.nodes):
        if name == target:
            continue
        node = bundle.nodes[name]
        references = node.collision_geometry or node.visual_geometry
        for reference in references:
            mesh = _load_obj(reference.mesh_path).transformed(
                node.pose.matrix @ reference.mesh_to_asset
            )
            result.append(
                CollisionGeometry(name, mesh, is_watertight(mesh), reference.kind)
            )
    return tuple(result)


def signed_distance(
    points: np.ndarray,
    mesh: TriangleMesh,
    *,
    point_chunk_size: int = 256,
    face_chunk_size: int = 2048,
) -> np.ndarray:
    """Measure point-to-triangle distance; negative means inside a closed mesh."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if point_chunk_size < 1 or face_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    triangles = mesh.vertices[mesh.faces]
    distances = np.empty(len(points), dtype=np.float64)
    closed = is_watertight(mesh)
    for start in range(0, len(points), point_chunk_size):
        part = points[start : start + point_chunk_size]
        part_distances = np.full(len(part), np.inf)
        intersection_counts = np.zeros(len(part), dtype=np.int64)
        for face_start in range(0, len(triangles), face_chunk_size):
            faces = triangles[face_start : face_start + face_chunk_size]
            part_distances = np.minimum(
                part_distances, _point_triangle_distance(part, faces)
            )
            if closed:
                intersection_counts += _ray_intersection_counts(part, faces)
        if closed:
            part_distances[intersection_counts % 2 == 1] *= -1
        distances[start : start + len(part)] = part_distances
    return distances


def validate_trajectories(
    body_vertices: np.ndarray,
    object_vertices: np.ndarray,
    obstacles: Sequence[CollisionGeometry],
    *,
    object_faces: np.ndarray | None = None,
    object_name: str = "target_object",
    clearance_m: float = 0.0,
    subdivisions: int = 1,
) -> tuple[CollisionSample, ...]:
    """Check body-room, object-room, and optionally body-object collisions.

    ``object_vertices`` must already be in world space. Linear subframes make
    tunnelling less likely but are still discrete samples, not continuous CCD.
    """
    body = _trajectory(body_vertices, "body_vertices")
    obj = _trajectory(object_vertices, "object_vertices")
    if len(body) != len(obj):
        raise ValueError("body and object trajectories must have equal length")
    if clearance_m < 0 or subdivisions < 1:
        raise ValueError("clearance_m must be nonnegative and subdivisions positive")
    samples = []
    for frame, body_points, object_points in _subframes(body, obj, subdivisions):
        for moving_name, points in (("body", body_points), ("object", object_points)):
            for obstacle in obstacles:
                distance = signed_distance(points, obstacle.mesh)
                # Open/thin meshes have no meaningful inside. Treat them as a
                # two-sided clearance surface; closed meshes additionally report depth.
                colliding = distance < clearance_m
                penetration = (
                    np.maximum(-distance, 0.0)
                    if obstacle.watertight
                    else np.zeros_like(distance)
                )
                samples.append(
                    CollisionSample(
                        frame=frame,
                        moving_geometry=moving_name,
                        obstacle=obstacle.node_name,
                        minimum_distance_m=float(distance.min(initial=np.inf)),
                        maximum_penetration_m=float(penetration.max(initial=0.0)),
                        colliding_vertices=int(colliding.sum()),
                    )
                )
        if object_faces is not None:
            target = TriangleMesh(
                object_points, np.asarray(object_faces, dtype=np.int64)
            )
            distance = signed_distance(body_points, target)
            colliding = distance < clearance_m
            penetration = (
                np.maximum(-distance, 0.0)
                if is_watertight(target)
                else np.zeros_like(distance)
            )
            samples.append(
                CollisionSample(
                    frame=frame,
                    moving_geometry="body",
                    obstacle=object_name,
                    minimum_distance_m=float(distance.min(initial=np.inf)),
                    maximum_penetration_m=float(penetration.max(initial=0.0)),
                    colliding_vertices=int(colliding.sum()),
                )
            )
    return tuple(samples)


def _trajectory(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or result.shape[2] != 3 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite array with shape (T, V, 3)")
    return result


def _subframes(
    body: np.ndarray, obj: np.ndarray, subdivisions: int
) -> Iterable[tuple[float, np.ndarray, np.ndarray]]:
    if len(body) == 0:
        return
    yield 0.0, body[0], obj[0]
    for index in range(len(body) - 1):
        for step in range(1, subdivisions + 1):
            alpha = step / subdivisions
            yield (
                index + alpha,
                (1 - alpha) * body[index] + alpha * body[index + 1],
                (1 - alpha) * obj[index] + alpha * obj[index + 1],
            )


def _point_triangle_distance(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Exact unsigned distances using closest points on triangle regions."""
    # Real-Time Collision Detection, Christer Ericson, section 5.1.5.
    p = points[:, None, :]
    a, b, c = (triangles[None, :, i, :] for i in range(3))
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = np.sum(ab * ap, axis=-1), np.sum(ac * ap, axis=-1)
    closest = np.empty_like(p + a)
    assigned = np.zeros(d1.shape, dtype=bool)

    def put(mask, value):
        nonlocal assigned
        mask &= ~assigned
        closest[mask] = np.broadcast_to(value, closest.shape)[mask]
        assigned |= mask

    put((d1 <= 0) & (d2 <= 0), a)
    bp = p - b
    d3, d4 = np.sum(ab * bp, axis=-1), np.sum(ac * bp, axis=-1)
    put((d3 >= 0) & (d4 <= d3), b)
    vc = d1 * d4 - d3 * d2
    denom = d1 - d3
    v = np.divide(d1, denom, out=np.zeros_like(d1), where=denom != 0)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[..., None] * ab)
    cp = p - c
    d5, d6 = np.sum(ab * cp, axis=-1), np.sum(ac * cp, axis=-1)
    put((d6 >= 0) & (d5 <= d6), c)
    vb = d5 * d2 - d1 * d6
    denom = d2 - d6
    w = np.divide(d2, denom, out=np.zeros_like(d2), where=denom != 0)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[..., None] * ac)
    va = d3 * d6 - d5 * d4
    denom = (d4 - d3) + (d5 - d6)
    w = np.divide(d4 - d3, denom, out=np.zeros_like(d3), where=denom != 0)
    put(
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
        b + w[..., None] * (c - b),
    )
    denom = va + vb + vc
    v = np.divide(vb, denom, out=np.zeros_like(vb), where=denom != 0)
    w = np.divide(vc, denom, out=np.zeros_like(vc), where=denom != 0)
    put(~assigned, a + v[..., None] * ab + w[..., None] * ac)
    return np.linalg.norm(p - closest, axis=-1).min(axis=1)


def _ray_intersection_counts(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    # A non-axis-aligned ray reduces edge coincidences; two tiny deterministic
    # offsets make exact vertex/edge hits much less likely.
    direction = np.array([1.0, 0.37139068, 0.69474659])
    direction /= np.linalg.norm(direction)
    origins = points + np.array([0.0, 1e-10, -1e-10])
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    valid_face = np.abs(det) > 1e-12
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid_face)
    s = origins[:, None, :] - triangles[None, :, 0, :]
    u = np.einsum("nfi,fi->nf", s, h) * inv
    q = np.cross(s, edge1[None, :, :])
    v = np.einsum("i,nfi->nf", direction, q) * inv
    t = np.einsum("fi,nfi->nf", edge2, q) * inv
    hits = valid_face & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-10)
    return hits.sum(axis=1)
