from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_interactmove_intermimic_scene_canary.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("scene_canary_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canary_uses_exact_ten_second_half_open_clock() -> None:
    runner = _load_runner()

    steps = runner._sample_steps(frame_count=300, sim_hz=200, video_fps=30)

    assert len(steps) == 300
    assert steps[0] == 0
    assert steps[-1] == 1993
    assert steps == sorted(set(steps))


def test_canary_help_states_scientific_boundary() -> None:
    runner = _load_runner()
    help_text = runner._parser().format_help()

    assert "robot-free" in help_text
    assert "not an InterMimic or humanoid rollout" in help_text


def test_contact_summary_reports_maximum_penetration() -> None:
    runner = _load_runner()

    class Point:
        def __init__(self, separation: float) -> None:
            self.separation = separation

    class Contact:
        def __init__(self, *separations: float) -> None:
            self.points = [Point(value) for value in separations]

    class Scene:
        def get_contacts(self):
            return [Contact(-0.003, 0.001), Contact(-0.012)]

    metrics = runner._contact_metrics(Scene())

    assert metrics == {
        "contact_pair_count": 2,
        "contact_point_count": 3,
        "max_penetration_m": pytest.approx(0.012),
    }


def test_scene_bundle_file_verification_rejects_mutation(tmp_path: Path) -> None:
    runner = _load_runner()
    scene = tmp_path / "scene"
    scene.mkdir()
    artifact = scene / "mesh.obj"
    artifact.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    bundle = {
        "files": [
            {"path": "mesh.obj", "bytes": 8, "sha256": digest, "roles": ["mesh"]}
        ]
    }
    runner._verify_scene_bundle_files(scene, bundle)

    artifact.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        runner._verify_scene_bundle_files(scene, bundle)


def test_fixed_room_loader_keeps_every_parsed_object(tmp_path: Path) -> None:
    runner = _load_runner()
    urdf = tmp_path / "room.urdf"
    urdf.write_text("<robot name='room'/>", encoding="utf-8")

    class Loader:
        def load_multiple(self, path):
            assert path == str(urdf)
            return ["fixed-articulation"], ["visual", "walls"]

    articulations, entities, loaded = runner._load_fixed_room(Loader(), urdf)

    assert articulations == ["fixed-articulation"]
    assert entities == ["visual", "walls"]
    assert loaded == ["fixed-articulation", "visual", "walls"]
