from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

from babyworld_lite.interactmove_intermimic.environment import inspect_mesh


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_interactmove_intermimic_room_repair.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("room_repair_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_room_repair_help_describes_preserved_sam3d_composition() -> None:
    runner = _load_runner()

    help_text = runner._parser().format_help()

    assert "EmbodiedGen panorama-to-mesh" in help_text
    assert "existing accepted SAM3D object scene" in help_text


def test_authored_wall_collisions_are_separate_finite_convex_boxes(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    bounds = {
        "aabb_min_m": [-3.0, -2.5, 0.0],
        "aabb_max_m": [3.0, 2.5, 3.0],
    }

    collisions = runner._write_wall_collisions(tmp_path / "collision", bounds)

    assert [collision_id for collision_id, _ in collisions] == [
        "wall_x_min",
        "wall_x_max",
        "wall_y_min",
        "wall_y_max",
    ]
    reports = [
        inspect_mesh(
            path,
            f"background/collision/{path.name}",
            [1.0, 1.0, 1.0],
        )
        for _, path in collisions
    ]
    assert all(report["vertex_count"] == 8 for report in reports)
    assert all(report["face_count"] == 12 for report in reports)
    assert all(report["finite_vertices"] is True for report in reports)
    expected_extents = [
        [0.05, 5.1, 3.0],
        [0.05, 5.1, 3.0],
        [6.0, 0.05, 3.0],
        [6.0, 0.05, 3.0],
    ]
    for actual, expected in zip(
        sorted(report["extents_m"] for report in reports), expected_extents
    ):
        assert actual == pytest.approx(expected)


def test_preserved_activity_binding_projects_embodiedgen_seeds() -> None:
    runner = _load_runner()
    activity = {
        "activity_id": "red-mug-mouth-return",
        "prompt": "Pick up the red mug.",
        "seeds": {
            "master": 1,
            "embodiedgen_image": 2,
            "embodiedgen_asset": 3,
            "embodiedgen_layout": 4,
            "interactmove": 5,
            "intermimic": 6,
            "render": 7,
        },
    }
    receipt = {
        "activity": {
            "activity_id": "red-mug-mouth-return",
            "prompt": "Pick up the red mug.",
        },
        "seeds": {"image": 2, "asset": 3, "layout": 4},
    }

    runner._validate_preserved_activity_binding(activity, receipt)

    receipt["seeds"]["image"] = 99
    with pytest.raises(RuntimeError, match="seeds changed"):
        runner._validate_preserved_activity_binding(activity, receipt)


def test_torchvision_functional_tensor_compat_exposes_grayscale(monkeypatch) -> None:
    runner = _load_runner()
    grayscale = lambda value: value
    functional = types.ModuleType("torchvision.transforms.functional")
    functional.rgb_to_grayscale = grayscale
    transforms = types.ModuleType("torchvision.transforms")
    transforms.functional = functional
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = transforms
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)
    monkeypatch.setitem(sys.modules, "torchvision.transforms.functional", functional)
    monkeypatch.delitem(
        sys.modules, "torchvision.transforms.functional_tensor", raising=False
    )

    runner._install_torchvision_functional_tensor_compat()

    module = runner.sys.modules["torchvision.transforms.functional_tensor"]
    assert module.rgb_to_grayscale is grayscale


def test_equilib_patch_swaps_embodiedgen_height_width_order() -> None:
    runner = _load_runner()
    observed = []

    def cube2equi(cubemap, cube_format, width, height):
        observed.append((cubemap, cube_format, width, height))
        return "panorama"

    trainer = types.SimpleNamespace(cube2equi=cube2equi)
    runner._install_equilib_argument_order_patch(trainer)

    assert trainer.cube2equi("cube", "list", 1024, 2048) == "panorama"
    assert observed == [("cube", "list", 2048, 1024)]
