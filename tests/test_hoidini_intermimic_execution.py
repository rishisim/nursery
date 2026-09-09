import copy

import numpy as np
import pytest

from nursery.humanoid.hoidini_intermimic.execute import room_collision_mesh


def scene(tmp_path):
    path = tmp_path / "triangle.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    mesh_transform = np.eye(4)
    mesh_transform[:3, :3] *= 2
    mesh_transform[0, 3] = 1
    node = {"pose": [10, 20, 3, 0, 0, 2**-0.5, 2**-0.5],
            "collision_geometry": [{"mesh_path": str(path), "mesh_to_asset": mesh_transform.tolist()}],
            "visual_geometry": []}
    return {"active_interaction": {"target_name": "mug"},
            "nodes": {"desk": node, "mug": copy.deepcopy(node),
                      "root": {"collision_geometry": [], "visual_geometry": []}}}


def test_room_geometry_preserves_scale_local_origin_and_world_pose(tmp_path):
    receipt = scene(tmp_path)
    mesh, sources = room_collision_mesh(receipt)
    np.testing.assert_allclose(mesh.vertices, [[10, 21, 3], [10, 23, 3], [8, 21, 3]])
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2]])
    assert [source["node"] for source in sources] == ["desk"]


def test_target_is_not_duplicated_as_static_collision(tmp_path):
    receipt = scene(tmp_path)
    receipt["nodes"]["mug"]["collision_geometry"][0]["mesh_path"] = "missing-target.obj"
    assert len(room_collision_mesh(receipt)[0].faces) == 1


def test_visible_room_geometry_without_collision_is_rejected(tmp_path):
    receipt = scene(tmp_path)
    desk = receipt["nodes"]["desk"]
    desk["visual_geometry"], desk["collision_geometry"] = desk["collision_geometry"], []
    with pytest.raises(ValueError, match="no collision geometry"):
        room_collision_mesh(receipt)
