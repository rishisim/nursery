import json
from pathlib import Path

import pytest

from babyworld_lite.childlens_engine_bakeoff.spec_kernel import (
    CalibrationPolicy,
    UnsupportedEpisodeIntent,
    compile_episode_intent,
)


PROTOCOL = json.loads(
    Path("configs/childlens_mimo_molmospaces_spec_kernel.json").read_text()
)
CALIBRATION = CalibrationPolicy.load(Path("configs/childlens_simulator_bridge.json"))
ASSETS = {"yellow_cup": {"sha256": "a" * 64, "scale_m": [0.08, 0.08, 0.10]}}


def _intent():
    return {
        "mode": "calibrated",
        "activity_family": "object_centered_reach_manipulate",
        "scene": {
            "id": "FloorPlan1",
            "version": "iTHOR-pinned",
            "asset_sha256": "b" * 64,
        },
        "target": {"asset_id": "yellow_cup"},
        "phases": ["look_settle", "near_miss", "touch_push", "object_naming"],
        "duration_s": 30,
        "speech": {"name_count": 2, "text": "die gelbe Tasse"},
    }


def test_compiler_is_deterministic_and_provenance_rich():
    first = compile_episode_intent(_intent(), PROTOCOL, CALIBRATION, asset_registry=ASSETS)
    second = compile_episode_intent(_intent(), PROTOCOL, CALIBRATION, asset_registry=ASSETS)
    assert first == second
    assert first["spec_sha256"] == second["spec_sha256"]
    assert first["speech_events"]["provenance"] == "childlens_calibration"
    assert first["canonical_scientific_batch_admissible"]["value"] is True
    assert first["calibration"]["record_sha256"]["value"] == CALIBRATION.source_sha256


def test_calibrated_mode_rejects_user_timing_override():
    intent = _intent()
    intent["speech_timing"] = {"start_s": 1.0, "gap_s": 1.0}
    with pytest.raises(UnsupportedEpisodeIntent, match="cannot override"):
        compile_episode_intent(intent, PROTOCOL, CALIBRATION, asset_registry=ASSETS)


def test_unsupported_activity_rejected_not_approximated():
    intent = _intent()
    intent["activity_family"] = "navigation"
    with pytest.raises(UnsupportedEpisodeIntent, match="unsupported activity_family"):
        compile_episode_intent(intent, PROTOCOL, CALIBRATION, asset_registry=ASSETS)
