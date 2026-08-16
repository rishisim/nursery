from __future__ import annotations

import importlib.util
from pathlib import Path

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
