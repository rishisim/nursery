from __future__ import annotations

import numpy as np

from nursery.humanoid.embodiedgen_hoidini.collision import (
    CollisionGeometry,
    is_watertight,
    signed_distance,
    validate_trajectories,
)
from nursery.humanoid.embodiedgen_hoidini.prepare_scene import TriangleMesh


def _box() -> TriangleMesh:
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return TriangleMesh(vertices, faces)


def test_watertight_classification_and_signed_distance() -> None:
    box = _box()
    open_box = TriangleMesh(box.vertices, box.faces[:-2])
    assert is_watertight(box)
    assert not is_watertight(open_box)
    np.testing.assert_allclose(
        signed_distance(np.array([[0.5, 0.5, 0.5], [1.2, 0.5, 0.5]]), box),
        [-0.5, 0.2], atol=1e-9,
    )
    assert signed_distance(np.array([[0.5, 0.5, 0.5]]), open_box)[0] >= 0


def test_subframes_detect_tunnelling_missed_by_endpoints() -> None:
    wall = TriangleMesh(
        np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], dtype=float),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    obstacle = CollisionGeometry("wall", wall, watertight=False)
    body = np.array([[[-1, 0, 0]], [[1, 0, 0]]], dtype=float)
    obj = np.array([[[-2, 0, 0]], [[-2, 0, 0]]], dtype=float)

    endpoints = validate_trajectories(body, obj, [obstacle], clearance_m=0.01)
    swept = validate_trajectories(body, obj, [obstacle], clearance_m=0.01, subdivisions=2)

    assert not any(row.colliding_vertices for row in endpoints)
    assert any(row.frame == 0.5 and row.colliding_vertices for row in swept)


def test_closed_mesh_reports_penetration_and_open_mesh_reports_clearance() -> None:
    box = CollisionGeometry("cabinet", _box(), watertight=True)
    point = np.array([[[0.5, 0.5, 0.5]]])
    far_object = np.array([[[2.0, 2.0, 2.0]]])
    rows = validate_trajectories(point, far_object, [box], clearance_m=0.02)
    body = next(row for row in rows if row.moving_geometry == "body")
    assert body.colliding_vertices == 1
    assert body.maximum_penetration_m == 0.5


def test_body_object_penetration_is_checked_without_adding_target_to_room() -> None:
    target = _box()
    body = np.array([[[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]])
    object_vertices = target.vertices[None]

    rows = validate_trajectories(
        body,
        object_vertices,
        [],
        object_faces=target.faces,
        object_name="mug",
    )

    assert len(rows) == 1
    assert rows[0].obstacle == "mug"
    assert rows[0].colliding_vertices == 1
    assert rows[0].maximum_penetration_m == 0.5
