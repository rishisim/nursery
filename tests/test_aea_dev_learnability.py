from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.run_aea_dev_learnability import (
    VideoTextDataset,
    audit_summary,
    build_endpoints,
    cluster_bootstrap,
    group_derangement,
    imu_features,
    mechanical_decision,
    pairwise_credit,
    select_capacity_subset,
    validate_development_source,
    validate_folds,
)


def row(index: int, action: str, group: str) -> dict:
    return {
        "schema_version": "aea-grounding-v1",
        "example_id": f"e{index}",
        "sequence_id": f"{group}_rec{index % 2}",
        "event_group": group,
        "evaluation_targets": {"action_verb": action, "object_noun": ""},
        "model_inputs": {"transcript": action},
    }


def protocol() -> dict:
    return {
        "protocol_id": "p", "source_examples_sha256": "source",
        "confirmation_event_groups": ["reserve"],
        "counts": {"development_windows": 18},
        "support": {"minimum_windows": 6, "minimum_event_groups": 3},
        "coarse_mapping": {
            "locomotion_transport": ["walk"],
            "cleaning_grooming": ["wash"],
        },
        "semantic_high_motion_actions": ["walk", "wash"],
    }


def supported_rows() -> list[dict]:
    actions = ["walk", "wash", "read"]
    return [row(i, actions[i // 6], f"g{i % 3}") for i in range(18)]


def test_development_source_is_exact_and_reserve_excluded() -> None:
    rows = supported_rows()
    partition = {
        "protocol_id": "p", "source_examples_sha256": "source",
        "entries": [
            *[{"example_id": r["example_id"], "partition": "development"} for r in rows],
            {"example_id": "held", "partition": "confirmation"},
        ],
    }
    assert all(validate_development_source(rows, partition, protocol()).values())
    contaminated = copy.deepcopy(rows)
    contaminated[0]["event_group"] = "reserve"
    with pytest.raises(ValueError, match="integrity failure"):
        validate_development_source(contaminated, partition, protocol())


def test_fixed_endpoint_mappings_and_support() -> None:
    endpoints = build_endpoints(supported_rows(), protocol())
    assert endpoints["fine"]["labels"] == ("read", "walk", "wash")
    assert endpoints["coarse"]["labels"] == ("cleaning_grooming", "locomotion_transport")
    assert endpoints["coarse"]["prompt_members"]["locomotion_transport"] == ("walk",)
    assert endpoints["semantic_high_motion"]["labels"] == ("walk", "wash")


def test_real_coarse_support_does_not_inherit_fine_filter() -> None:
    root = Path(__file__).parents[1]
    rows = [
        json.loads(line)
        for line in (root / "output/aea_dev_learnability_v1/development_examples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    frozen = json.loads(
        (root / "output/aea_dev_learnability_v1/preregistered_protocol.json").read_text()
    )
    endpoints = build_endpoints(rows, frozen)
    assert endpoints["coarse"]["labels"] == (
        "cleaning_grooming",
        "food_drink_preparation",
        "locomotion_transport",
        "media_leisure",
        "object_state_manipulation",
    )
    assert len(endpoints["coarse"]["row_label"]) == 192


def test_fold_validator_rejects_group_leakage_and_requires_one_holdout() -> None:
    rows = [row(0, "walk", "g0"), row(1, "walk", "g0"), row(2, "wash", "g1")]
    valid = [{"train_indices": [2], "test_indices": [0, 1]},
             {"train_indices": [0, 1], "test_indices": [2]}]
    assert all(validate_folds(rows, [0, 1, 2], valid).values())
    invalid = [{"train_indices": [0, 2], "test_indices": [1]},
               {"train_indices": [0, 1], "test_indices": [2]}]
    with pytest.raises(ValueError, match="fold leakage"):
        validate_folds(rows, [0, 1, 2], invalid)


def test_group_derangement_is_deterministic_bijective_and_hard_fails() -> None:
    indices = list(range(6))
    groups = {0: "a", 1: "a", 2: "b", 3: "b", 4: "c", 5: "c"}
    first = group_derangement(indices, groups, 5101)
    assert first == group_derangement(indices, groups, 5101)
    assert set(first) == set(first.values()) == set(indices)
    assert all(groups[i] != groups[donor] for i, donor in first.items())
    impossible = {0: "a", 1: "a", 2: "a", 3: "b"}
    with pytest.raises(ValueError, match="exceeds half"):
        group_derangement(list(impossible), impossible, 5101)


def test_capacity_subset_is_deterministic_and_support_balanced() -> None:
    rows = supported_rows()
    first = select_capacity_subset(rows, ("walk", "wash", "read"), 9)
    assert first == select_capacity_subset(rows, ("walk", "wash", "read"), 9)
    counts = {action: sum(rows[i]["evaluation_targets"]["action_verb"] == action for i in first)
              for action in ("walk", "wash", "read")}
    assert counts == {"walk": 3, "wash": 3, "read": 3}


def test_pairwise_ties_and_cluster_interval_are_deterministic() -> None:
    assert pairwise_credit([0.2, 0.2, 0.1], 0) == 0.75
    rows = [
        {"event_group": f"g{i % 3}", "true_label": "a" if i % 2 else "b",
         "credit": float(i % 2)} for i in range(12)
    ]
    assert cluster_bootstrap(rows, 100, 42101) == cluster_bootstrap(rows, 100, 42101)


def test_fixed_imu_feature_extractor_is_finite_129_dimensions() -> None:
    t = np.arange(300) / 50.0
    values = np.stack([np.sin(t * (axis + 1)) for axis in range(6)], axis=1)
    features = imu_features(values, 50.0)
    assert features.shape == (129,)
    assert np.isfinite(features).all()


class NoMotorCorpus:
    def video(self, index: int):
        import torch
        return torch.zeros(2, 3, 4, 4, dtype=torch.uint8)

    def text(self, index: int) -> str:
        return "walk"

    def metadata(self, index: int):
        return object()

    def motor(self, index: int):  # pragma: no cover - any access must fail the test
        raise AssertionError("motor accessed")


def test_video_text_dataset_has_no_motor_access_or_field() -> None:
    from babyworld_lite.grounding.pilot_data import WordTokenizer
    item = VideoTextDataset(NoMotorCorpus(), [0], WordTokenizer.fit(["walk"]))[0]
    assert "motor" not in item
    assert item["text"] == "walk"


def audit(passed: bool) -> dict:
    return {"summaries": {
        name: {"gate_passed": passed}
        for name in ("overall", "fine", "coarse", "semantic_high_motion")
    }}


def endpoint_flags(fine: bool, coarse: bool = False, high: bool = False, imu: bool = False):
    grounding = {
        "fine": {"gate_passed": fine}, "coarse": {"gate_passed": coarse},
        "semantic_high_motion": {"gate_passed": high},
    }
    imu_results = {
        "fine": {"paired_gate_passed": imu and fine},
        "coarse": {"paired_gate_passed": imu and coarse},
        "semantic_high_motion": {"paired_gate_passed": imu and high},
    }
    return grounding, imu_results


def test_mechanical_decision_boundaries() -> None:
    grounding, imu = endpoint_flags(True, imu=True)
    assert mechanical_decision(audit(True), True, grounding, imu, True)["recommendation"] == "GO"
    assert mechanical_decision(audit(True), True, grounding, imu, False)["recommendation"] == "REVISE"
    assert mechanical_decision(audit(True), False, grounding, imu, True)["recommendation"] == "REVISE"
    grounding, imu = endpoint_flags(False)
    assert mechanical_decision(audit(True), True, grounding, imu, True)["recommendation"] == "STOP"


def test_audit_rates_use_all_rows_not_only_judgeable() -> None:
    labels = ([{"audit_label": "clear_match"}] * 6
              + [{"audit_label": "plausible_or_ambiguous"}] * 2
              + [{"audit_label": "mismatch"}]
              + [{"audit_label": "not_visually_judgeable"}])
    summary = audit_summary(labels)
    assert summary["metrics"]["judgeable"]["rate"] == 0.9
    assert summary["metrics"]["clear_or_plausible"]["rate"] == 0.8
    assert summary["metrics"]["mismatch"]["rate"] == 0.1
