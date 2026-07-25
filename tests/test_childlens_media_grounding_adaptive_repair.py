from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_historical_no_go_is_immutable() -> None:
    expected = {
        "configs/childlens_media_grounding_environment.json": "ac6542baa58d029780f56e6f36b0e82346ec32f80365f18310e45167c4ba8ecd",
        "docs/childlens_media_grounding_environment_protocol.md": "3582b688f6789a553c250a8232374a9b0d2787c6649686160c9e0f85ebc33dcd",
        "output/childlens_media_grounding_environment/engine_gate_receipt.json": "447fb3e353f14b11b58155f15f84f3d92837c65d35dbd02e6d7cde3128c95fdd",
        "output/childlens_media_grounding_environment/terminal_report.md": "13110b599111fffa697e32d33c1304fe2eeb9e21bb1968f3b8ef382ca47ace3c",
    }
    for relative, value in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == value


def test_amendment_is_childlens_only_and_cue_lift_sealed() -> None:
    config = json.loads(
        (ROOT / "configs/childlens_media_grounding_adaptive_repair.json").read_text()
    )
    assert config["scope"]["sole_empirical_source"] == "ChildLens"
    assert config["scope"]["development_participants"] == 18
    assert config["scope"]["locked_participants"] == 22
    assert config["scope"]["final_cue_lift_experiment_authorized"] is False
    assert config["calibration_freeze"]["cue_lift_arms_forbidden"] is True
    assert config["stop_condition"]["generator_fit_started"] is False
    assert config["stop_condition"]["cue_lift_arms_run"] is False


def test_engineering_pass_does_not_override_scientific_no_go() -> None:
    engineering = json.loads(
        (
            ROOT
            / "output/childlens_media_grounding_adaptive_repair/engineering_receipt.json"
        ).read_text()
    )
    assert engineering["blender"]["gate"] == "PASS"
    assert engineering["blender"]["positive_contact"] is True
    assert engineering["blender"]["negative_no_contact"] is True
    assert engineering["blender"]["deterministic_state_replay"] is True
    report = (
        ROOT
        / "output/childlens_media_grounding_adaptive_repair/terminal_report.md"
    ).read_text()
    assert "NO_GO_ADAPTIVE_MEDIA_REPAIR" in report
    assert "participant-disjoint" in report
