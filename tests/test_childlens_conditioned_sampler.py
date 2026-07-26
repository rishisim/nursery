import json
from pathlib import Path

import pytest

from babyworld_lite.childlens_engine_bakeoff.conditioned_sampler import (
    FrozenSampler,
    qualify_sampled_population,
)
from babyworld_lite.childlens_engine_bakeoff.spec_kernel import (
    CalibrationPolicy,
    UnsupportedEpisodeIntent,
    compile_episode_intent,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_conditioned_batch_v1.json"


def test_frozen_population_is_deterministic_and_schedule_qualified(monkeypatch):
    monkeypatch.chdir(ROOT)
    sampler = FrozenSampler.load(CONFIG)
    first = sampler.sample()
    second = sampler.sample()
    assert first == second
    assert len(first) == 24
    assert len({episode["seed"] for episode in first}) == 24
    result = qualify_sampled_population(first, sampler.config, sampler.calibration)
    assert result["all_direct_schedule_features_pass"]
    assert result["joint_coverage"]["passed"]
    assert {episode["intent"]["scene"]["id"] for episode in first} == {
        "FloorPlan1",
        "FloorPlan201",
    }
    assert {episode["intent"]["target"]["asset_id"] for episode in first} == {
        "yellow_cup_authored",
        "red_ball_authored",
    }


def test_conditioned_sample_fails_closed_on_calibration_hash(monkeypatch):
    monkeypatch.chdir(ROOT)
    sampler = FrozenSampler.load(CONFIG)
    episode = sampler.sample()[0]
    intent = json.loads(json.dumps(episode["intent"]))
    intent["conditioned_sample"]["calibration_sha256"] = "0" * 64
    calibration = CalibrationPolicy.load(ROOT / "configs/childlens_simulator_bridge.json")
    with pytest.raises(UnsupportedEpisodeIntent, match="hash mismatch"):
        compile_episode_intent(
            intent, sampler.protocol, calibration, asset_registry=sampler.assets
        )
