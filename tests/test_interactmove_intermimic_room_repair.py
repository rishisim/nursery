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


def test_authored_wall_collision_is_finite_room_scale(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "walls.obj"
    bounds = {
        "aabb_min_m": [-3.0, -2.5, 0.0],
        "aabb_max_m": [3.0, 2.5, 3.0],
    }

    runner._write_wall_collision(path, bounds)
    report = inspect_mesh(path, "background/collision/walls.obj", [1.0, 1.0, 1.0])

    assert report["vertex_count"] == 32
    assert report["face_count"] == 48
    assert report["finite_vertices"] is True
    assert report["extents_m"] == [6.1, 5.1, 3.0]


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
