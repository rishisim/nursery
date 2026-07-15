from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_aea_dev_imu_diagnostic import (
    EXPECTED_FEATURE_DIMENSION,
    build_endpoints,
    extract_imu_features,
    group_safe_bijection,
    make_endpoint_folds,
    validate_development_corpus,
)


def _protocol() -> dict:
    return {
        "support": {"minimum_windows": 6, "minimum_event_groups": 3},
        "coarse_mapping": {"motion": ["get", "move"], "state": ["put"]},
        "semantic_high_motion_actions": ["get", "move"],
        "folds": {"count": 5, "shuffle": True, "random_state": 41031},
    }


def _row(example_id: str, group: str, action: str) -> dict:
    return {
        "example_id": example_id,
        "event_group": group,
        "sequence_id": f"sequence-{group}",
        "evaluation_targets": {"action_verb": action},
    }


def test_support_aware_endpoints_are_frozen_before_folds() -> None:
    rows = []
    for action, groups, per_group in (
        ("get", ["g0", "g1", "g2", "g3", "g4"], 2),
        ("move", ["h0", "h1"], 4),  # enough windows, too few groups
        ("put", ["p0", "p1", "p2"], 2),
    ):
        for group in groups:
            for number in range(per_group):
                rows.append(_row(f"{action}-{group}-{number}", group, action))
    endpoints = build_endpoints(rows, _protocol())
    assert endpoints["fine"]["eligible_labels"] == ["get", "put"]
    assert endpoints["semantic_high_motion_fine"]["eligible_labels"] == ["get"]
    # Coarse support is based on fixed category support, not post-hoc fine eligibility.
    assert endpoints["coarse"]["eligible_labels"] == ["motion", "state"]


def test_stratified_group_folds_are_deterministic_leakage_safe_and_complete() -> None:
    rows = []
    for action in ("get", "put"):
        for group_number in range(5):
            group = f"{action}-g{group_number}"
            for number in range(2):
                rows.append(_row(f"{group}-{number}", group, action))
    endpoint = build_endpoints(rows, _protocol())["fine"]
    first = make_endpoint_folds(endpoint, _protocol())
    second = make_endpoint_folds(endpoint, _protocol())
    assert first == second
    held_out = [index for fold in first for index in fold["test_indices"]]
    assert sorted(held_out) == list(range(len(endpoint["rows"])))
    assert all(not fold["group_overlap"] for fold in first)


def test_group_safe_derangement_reports_feasible_and_infeasible_sides() -> None:
    feasible_rows = [
        {"example_id": f"x{i}", "event_group": group}
        for i, group in enumerate(["a", "a", "b", "b"])
    ]
    feasible = group_safe_bijection(range(4), feasible_rows, seed=5101)
    assert feasible["feasible"]
    assert feasible["whole_window_bijection"]
    assert feasible["self_match_count"] == 0
    assert feasible["same_event_group_match_count"] == 0
    assert len(set(feasible["donor_map"].values())) == 4

    invalid_rows = [
        {"example_id": f"y{i}", "event_group": group}
        for i, group in enumerate(["a", "a", "a", "b"])
    ]
    invalid = group_safe_bijection(range(4), invalid_rows, seed=5101)
    assert not invalid["feasible"]
    assert not invalid["necessary_half_size_bound_passed"]
    assert invalid["donor_map"] is None


def test_fixed_features_have_expected_dimension_and_are_finite() -> None:
    time = np.arange(300, dtype=np.float64) / 50.0
    values = np.stack([
        np.sin(2 * np.pi * frequency * time) if frequency else np.ones_like(time)
        for frequency in (0, 0.25, 1.0, 3.0, 8.0, 12.0)
    ], axis=1)
    features = extract_imu_features(values, 50.0)
    assert features.shape == (EXPECTED_FEATURE_DIMENSION,)
    assert EXPECTED_FEATURE_DIMENSION == 129
    assert np.isfinite(features).all()


def test_development_validation_rejects_confirmation_membership(tmp_path: Path) -> None:
    development = tmp_path / "development_examples.jsonl"
    dev = _row("dev", "g-dev", "get")
    confirmation = _row("confirm", "g-confirm", "get")
    development.write_text(json.dumps(dev, sort_keys=True) + "\n")
    receipt = {
        "development_examples_sha256": hashlib.sha256(development.read_bytes()).hexdigest(),
    }
    development.with_suffix(".receipt.json").write_text(json.dumps(receipt))
    partition = {
        "confirmation_event_groups": ["g-confirm"],
        "entries": [
            {**dev, "action_verb": "get", "partition": "development"},
            {**confirmation, "action_verb": "get", "partition": "confirmation"},
        ],
    }
    checks = validate_development_corpus([dev], development, partition)
    assert checks["exact_development_manifest_membership"]
    assert checks["zero_confirmation_id_overlap"]
    assert checks["zero_confirmation_event_group_overlap"]

    development.write_text(json.dumps(confirmation, sort_keys=True) + "\n")
    receipt["development_examples_sha256"] = hashlib.sha256(development.read_bytes()).hexdigest()
    development.with_suffix(".receipt.json").write_text(json.dumps(receipt))
    try:
        validate_development_corpus([confirmation], development, partition)
    except ValueError as error:
        assert "development/reserve isolation" in str(error)
    else:
        raise AssertionError("confirmation row was not rejected")
