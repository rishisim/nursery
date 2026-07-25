from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from babyworld_lite.childlens_simulator_bridge.calibration import load_and_verify_contract
from babyworld_lite.childlens_simulator_bridge.generator import (
    generate_episode,
    measure_episode,
    side_stream_integrity,
    write_episode,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "childlens_simulator_bridge.json"


def _contract() -> dict:
    return load_and_verify_contract(ROOT, CONTRACT_PATH)


def test_episode_has_synchronized_withholdable_modalities(tmp_path: Path) -> None:
    episode = generate_episode(_contract(), "weak_central", 2026072501, 0)
    n = 300
    assert episode.rgb_frames.shape == (n, 16, 16, 3)
    assert episode.rgb_frames.dtype == np.uint8
    assert all(len(getattr(episode, name)) == n for name in ("vision", "audio", "activity", "action", "proprioception", "contact", "imu"))
    assert episode.evaluation["withholdable"] is True
    assert side_stream_integrity(episode)["passed"] is True

    manifest = write_episode(episode, tmp_path, render_rgb=True)
    record = json.loads(manifest.read_text())
    assert len(list((manifest.parent / "rgb").glob("*.png"))) == n
    assert record["metadata"]["synchronization_clock"] == "episode_monotonic_seconds"


def test_generation_is_reproducible() -> None:
    first = generate_episode(_contract(), "weak_upper", 2026072502, 3)
    second = generate_episode(_contract(), "weak_upper", 2026072502, 3)
    assert first.metadata == second.metadata
    assert first.vision == second.vision
    assert first.audio == second.audio
    assert np.array_equal(first.rgb_frames, second.rgb_frames)
    assert measure_episode(first) == measure_episode(second)


def test_alignment_regimes_are_frozen_nonnegative_process_parameters() -> None:
    contract = _contract()
    values = [
        generate_episode(contract, regime, 2026072503, 0).metadata["alignment_process_contrast"]
        for regime in ("weak_lower", "weak_central", "weak_upper")
    ]
    assert values == [0.0, 0.00177, 0.0062]


def test_enriched_stratum_is_explicit_and_separate() -> None:
    episode = generate_episode(
        _contract(), "weak_central", 2026072501, 0, "grounding_enriched"
    )
    assert episode.metadata["stratum"] == "grounding_enriched"
    assert measure_episode(episode)["released_speech_support_fraction"] > 0


def test_side_streams_do_not_contain_target_or_referent() -> None:
    episode = generate_episode(_contract(), "weak_lower", 2026072501, 2)
    result = side_stream_integrity(episode)
    assert result == {
        "passed": True,
        "failures": [],
        "evaluation_separate_and_withholdable": True,
        "side_streams": ["action", "proprioception", "contact", "imu"],
    }


def test_curated_example_contains_no_empirical_payload() -> None:
    example = json.loads(
        (ROOT / "output" / "childlens_simulator_bridge" / "schema_example.json").read_text()
    )
    assert example["sensitivity"] == "synthetic_non_sensitive_metadata_only"
    assert example["bulk_payload_committed"] is False
    assert example["empirical_content_present"] is False
    assert example["evaluation_only"]["withholdable"] is True
