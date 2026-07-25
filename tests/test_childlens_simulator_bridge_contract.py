from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "childlens_simulator_bridge.json"


def test_contract_sources_are_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text())
    for source in contract["empirical_sources"]:
        payload = (ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["sha256"]


def test_contract_freezes_scope_and_alignment_semantics() -> None:
    contract = json.loads(CONTRACT.read_text())
    scope = contract["scientific_scope"]
    assert scope["age_range_years"] == [3, 5]
    assert scope["locked_participants_accessed"] == 0
    assert scope["natural_audiovisual_calibration_passed"] is False
    assert scope["learner_training_authorized"] is False

    alignment = contract["alignment_sensitivity"]
    values = [item["coupling_contrast"] for item in alignment["regimes"].values()]
    assert values == [0.0, 0.00177, 0.0062]
    assert alignment["curve_amplitude_sensitivity_cap"] == 0.03713
    assert "anti-grounding" in alignment["negative_sampling_noise_policy"]


def test_natural_and_enriched_strata_are_separate() -> None:
    contract = json.loads(CONTRACT.read_text())
    weights = contract["natural_activity_mixture"]["weights"]
    assert abs(sum(weights.values()) - 0.9999) < 1e-9
    assert contract["natural_activity_mixture"]["role"].startswith("primary")
    assert "never replaces" in contract["grounding_enriched_conditional"]["role"]
