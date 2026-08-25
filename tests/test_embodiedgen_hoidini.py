from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nursery.humanoid.embodiedgen_hoidini import (
    BridgeValidationError,
    prepare_scene,
)
from nursery.humanoid.embodiedgen_hoidini.prepare_scene import (
    TriangleMesh,
    derive_support_corners,
    quaternion_xyzw_to_axis_angle,
    quaternion_xyzw_to_matrix,
    sample_point_cloud,
    validate_embodiedgen_pose,
)


BOX_OBJ = """\
v -1 -1 0
v 1 -1 0
v 1 1 0
v -1 1 0
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""

TARGET_OBJ = """\
v 0 0 0
v 1 0 0
v 0 2 0
v 0 0 3
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""


def _write_asset(
    root: Path,
    name: str,
    obj_text: str,
    *,
    scale: str = "1 1 1",
    xyz: str = "0 0 0",
    rpy: str = "0 0 0",
    affordance: bool = False,
) -> Path:
    asset = root / "assets" / name
    mesh_dir = asset / "mesh"
    mesh_dir.mkdir(parents=True)
    mesh_path = mesh_dir / f"{name}.obj"
    collision_path = mesh_dir / f"{name}_collision.obj"
    mesh_path.write_text(obj_text, encoding="utf-8")
    collision_path.write_text(obj_text, encoding="utf-8")
    custom_data = ""
    if affordance:
        annotation = {"affordances": [{"part_name": "handle", "grasp_group": {}}]}
        (asset / "affordance.json").write_text(json.dumps(annotation), encoding="utf-8")
        custom_data = """
    <custom_data><affordance>
      <affordance_annot>affordance.json</affordance_annot>
    </affordance></custom_data>"""
    urdf = f"""<robot name="{name}">
  <link name="{name}">
    <visual>
      <origin xyz="{xyz}" rpy="{rpy}"/>
      <geometry><mesh filename="mesh/{name}.obj" scale="{scale}"/></geometry>
    </visual>
    <collision>
      <origin xyz="{xyz}" rpy="{rpy}"/>
      <geometry><mesh filename="mesh/{name}_collision.obj" scale="{scale}"/></geometry>
    </collision>{custom_data}
  </link>
</robot>"""
    (asset / f"{name}.urdf").write_text(urdf, encoding="utf-8")
    return asset


def _layout_fixture(tmp_path: Path) -> tuple[Path, dict]:
    (tmp_path / "background").mkdir()
    _write_asset(tmp_path, "table", BOX_OBJ)
    _write_asset(
        tmp_path,
        "red_mug",
        TARGET_OBJ,
        scale="2 1 1",
        xyz="0.1 0.2 0.3",
        rpy=f"0 0 {math.pi / 2}",
        affordance=True,
    )
    _write_asset(tmp_path, "book", BOX_OBJ, scale="0.2 0.3 0.05")
    layout = {
        "tree": {
            "kitchen": [["table", "FLOOR"], ["franka", "IN"]],
            "table": [["red_mug", "ON"], ["book", "ON"]],
        },
        "relation": {
            "task_desc": "Pick up the red mug from the table.",
            "task": "single-arm pick",
            "robot": "franka",
            "background": "kitchen",
            "context": "table",
            "manipulated_objs": ["red_mug"],
            "distractor_objs": ["book"],
        },
        "objs_desc": {
            "kitchen": "bright kitchen",
            "table": "wooden table",
            "red_mug": "glossy red mug",
            "book": "blue book",
        },
        "objs_mapping": {
            "kitchen": "background",
            "table": "context",
            "red_mug": "manipulated_objs",
            "book": "distractor_objs",
        },
        "assets": {
            "kitchen": "background",
            "table": "assets/table",
            "red_mug": "assets/red_mug",
            "book": "assets/book",
        },
        "quality": {"red_mug": {"semantic": "OK"}},
        "position": {
            "kitchen": [0, 0, 0, 0, 0, 0, 1],
            "table": [1, 2, 0.5, 0, 0, 0, 1],
            "red_mug": [1.5, 2.25, 1.5, 0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)],
            "book": [0.5, 2, 1.5, 0, 0, 0, 1],
            "franka": [0, 1, 1.5, 0, 0, 0, 1],
        },
        "extra_source_field": {"preserved": True},
    }
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path, layout


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))
    return values[order]


def test_complete_layout_is_preserved_and_paths_are_resolved(tmp_path: Path) -> None:
    layout_path, layout = _layout_fixture(tmp_path)

    bundle = prepare_scene(layout_path, point_count=32, seed=7)

    assert set(bundle.nodes) == {"kitchen", "table", "red_mug", "book", "franka"}
    assert bundle.raw_layout == layout
    assert bundle.nodes["franka"].role == "robot"
    assert bundle.nodes["red_mug"].asset_reference == "assets/red_mug"
    assert bundle.nodes["red_mug"].asset_path == (tmp_path / "assets/red_mug").resolve()
    assert len(bundle.nodes["red_mug"].visual_geometry) == 1
    assert len(bundle.nodes["red_mug"].collision_geometry) == 1
    assert bundle.nodes["red_mug"].quality == {"semantic": "OK"}
    assert bundle.nodes["red_mug"].affordance_metadata["payload"]["affordances"][0]["part_name"] == "handle"
    assert "robot gripper grasps are metadata only" in bundle.nodes["red_mug"].affordance_metadata["frame"]
    assert bundle.units == "metres"
    assert bundle.up_axis == "+Z"
    np.testing.assert_allclose(bundle.embodiedgen_to_hoidini_world, np.eye(4))
    json.dumps(bundle.to_manifest())


def test_ambiguous_target_requires_explicit_selection(tmp_path: Path) -> None:
    layout_path, layout = _layout_fixture(tmp_path)
    layout["relation"]["manipulated_objs"].append("bowl")
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    with pytest.raises(BridgeValidationError, match="explicit target"):
        prepare_scene(layout_path)


def test_urdf_scale_and_origin_rotation_are_applied(tmp_path: Path) -> None:
    layout_path, _ = _layout_fixture(tmp_path)
    bundle = prepare_scene(layout_path, point_count=8)
    reference = bundle.nodes["red_mug"].visual_geometry[0]

    transformed = reference.mesh_to_asset @ np.array([1.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(transformed[:3], [0.1, 2.2, 0.3], atol=1e-9)
    assert reference.scale == (2.0, 1.0, 1.0)
    assert reference.origin_xyz == (0.1, 0.2, 0.3)


def test_quaternion_validation_and_conversion() -> None:
    pose = validate_embodiedgen_pose([1, 2, 3, 0, 0, 0, 1.0005])
    assert pose.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    half = math.sqrt(0.5)
    rotation = quaternion_xyzw_to_matrix([0, 0, half, half])
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(
        quaternion_xyzw_to_axis_angle([0, 0, half, half]),
        [0, 0, math.pi / 2],
        atol=1e-12,
    )
    with pytest.raises(BridgeValidationError, match="zero length"):
        validate_embodiedgen_pose([0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(BridgeValidationError, match="not within"):
        validate_embodiedgen_pose([0, 0, 0, 0, 0, 0, 2])


def test_recentering_preserves_world_space_geometry(tmp_path: Path) -> None:
    layout_path, _ = _layout_fixture(tmp_path)
    bundle = prepare_scene(layout_path, point_count=8)
    node = bundle.nodes["red_mug"]
    reference = node.visual_geometry[0]
    raw_vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float64
    )
    source_asset = TriangleMesh(raw_vertices, np.array([[0, 2, 1]], dtype=np.int64)).transformed(reference.mesh_to_asset)
    source_world = source_asset.transformed(node.pose.matrix).vertices
    active = bundle.active_interaction
    canonical_world = active.canonical_target_mesh.transformed(
        active.canonical_target_pose.matrix
    ).vertices

    np.testing.assert_allclose(
        _sorted_rows(source_world), _sorted_rows(canonical_world), atol=1e-10
    )
    canonical_bounds = active.canonical_target_mesh.vertices
    np.testing.assert_allclose(
        canonical_bounds.min(axis=0) + canonical_bounds.max(axis=0), 0, atol=1e-12
    )


def test_point_sampling_is_deterministic_and_normals_are_finite(tmp_path: Path) -> None:
    layout_path, _ = _layout_fixture(tmp_path)
    mesh = prepare_scene(layout_path, point_count=8).active_interaction.canonical_target_mesh

    first = sample_point_cloud(mesh, point_count=64, seed=19)
    second = sample_point_cloud(mesh, point_count=64, seed=19)
    third = sample_point_cloud(mesh, point_count=64, seed=20)

    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.normals, second.normals)
    assert not np.array_equal(first.points, third.points)
    assert np.isfinite(first.points).all()
    assert np.isfinite(first.normals).all()
    np.testing.assert_allclose(np.linalg.norm(first.normals, axis=1), 1.0)


def test_support_corners_are_counter_clockwise_with_upward_normal(tmp_path: Path) -> None:
    layout_path, _ = _layout_fixture(tmp_path)

    corners = prepare_scene(layout_path, point_count=8).active_interaction.support_corners_world

    assert corners.shape == (4, 3)
    np.testing.assert_allclose(corners[:, 2], 1.5)
    assert np.cross(corners[1] - corners[0], corners[2] - corners[1])[2] > 0
    assert corners[:, 0].min() == pytest.approx(0.0)
    assert corners[:, 0].max() == pytest.approx(2.0)
    assert corners[:, 1].min() == pytest.approx(1.0)
    assert corners[:, 1].max() == pytest.approx(3.0)


def test_non_horizontal_and_ambiguous_supports_are_rejected() -> None:
    sloped = TriangleMesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0.2], [1, 1, 0.2], [0, 1, 0]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
    )
    with pytest.raises(BridgeValidationError, match="no horizontal"):
        derive_support_corners(sloped)

    vertices = np.array(
        [
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            [2, 0, 1], [3, 0, 1], [3, 1, 1], [2, 1, 1],
        ],
        dtype=np.float64,
    )
    disconnected = TriangleMesh(
        vertices=vertices,
        faces=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]),
    )
    with pytest.raises(BridgeValidationError, match="multiple disconnected"):
        derive_support_corners(disconnected)


def test_prompt_preservation_override_and_prefix_status(tmp_path: Path) -> None:
    layout_path, _ = _layout_fixture(tmp_path)

    default = prepare_scene(layout_path, point_count=8)
    overridden = prepare_scene(
        layout_path, prompt="The person lifts the mug.", point_count=8
    )

    assert default.active_interaction.prompt == "Pick up the red mug from the table."
    assert overridden.active_interaction.prompt == "The person lifts the mug."
    assert default.human_prefix.required is True
    assert default.human_prefix.provided is False
    assert default.human_prefix.frames == 15
    assert default.human_prefix.fps == 20
    assert "still required" in default.human_prefix.message
    assert default.ready_for_hoidini_inference is False


def test_bridge_has_no_grab_or_hoidini_runtime_dependency(tmp_path: Path) -> None:
    layout_path, _ = _layout_fixture(tmp_path)

    bundle = prepare_scene(layout_path, point_count=8)

    assert bundle.active_interaction.target_name == "red_mug"
    assert bundle.active_interaction.support_name == "table"
