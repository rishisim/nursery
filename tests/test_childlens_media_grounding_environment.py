from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_fail_closed_and_childlens_only() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "childlens_media_grounding_environment.json").read_text()
    )
    assert protocol["scope"]["sole_empirical_source"] == "ChildLens"
    assert protocol["scope"]["locked_participants"] == 22
    assert protocol["scope"]["locked_participants_must_remain_unread"] is True
    assert protocol["scope"]["final_cue_lift_experiment_authorized"] is False
    assert protocol["fallback_policy"]["blender_failure"].startswith("terminate")


def test_scaffold_diagnostic_uses_output_and_fails() -> None:
    diagnostic = json.loads(
        (
            ROOT
            / "output"
            / "childlens_media_grounding_environment"
            / "scaffold_output_diagnostic.json"
        ).read_text()
    )
    assert diagnostic["decision"] == "FAIL_OUTPUT_CALIBRATION"
    assert diagnostic["actual_output"]["motion_mean"] < 0.01
    assert diagnostic["actual_output"]["scene_change_mean"] == 0.0
    assert "metadata was not used" in diagnostic["claim"]


def test_engine_gate_is_terminal_no_go() -> None:
    receipt = json.loads(
        (
            ROOT
            / "output"
            / "childlens_media_grounding_environment"
            / "engine_gate_receipt.json"
        ).read_text()
    )
    assert receipt["tdw"]["compatibility_repairs"] == 1
    assert receipt["tdw"]["gate"] == "FAIL"
    assert receipt["blender_fallback"]["gate"] == "FAIL"
    assert receipt["selected_engine_gate"] == "FAIL"
    assert receipt["terminal_effect"] == "NO_GO_MEDIA_GROUNDING_ENV"
