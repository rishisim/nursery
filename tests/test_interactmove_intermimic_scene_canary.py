from __future__ import annotations

import importlib.util
from pathlib import Path


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
