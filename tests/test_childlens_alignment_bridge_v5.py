from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from babyworld_lite.childlens_alignment_bridge_v5.preflight import (
    BridgeV5Error,
    common_duration_target,
    deterministic_folds,
    lag_grid,
    public_guard,
    run_embedding_positive_control,
    terminal_decision,
    trainable_parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_childlens_alignment_bridge_v5.py"


def _runner():
    spec = importlib.util.spec_from_file_location("childlens_v5_runner_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _participants() -> list[dict[str, str]]:
    activities = ("book", "play", "craft")
    return [
        {
            "participant_key": f"p{index:02d}",
            "cohort": "original" if index < 8 else "expansion",
            "activity_bin": activities[index % len(activities)],
            "location_bin": "inside" if index % 2 else "mixed",
            "speech_support_bin": "high" if index % 3 else "medium",
        }
        for index in range(18)
    ]


def test_common_duration_prefers_fifteen_minutes_and_fails_below_ten() -> None:
    rows = [
        {"participant_key": f"p{index}", "available_released_seconds": 930 + index}
        for index in range(18)
    ]
    assert common_duration_target(rows) == 900
    rows[0]["available_released_seconds"] = 599
    with pytest.raises(BridgeV5Error, match="E_DURATION_SUPPORT"):
        common_duration_target(rows)


def test_three_folds_are_disjoint_complete_balanced_and_order_invariant() -> None:
    rows = _participants()
    first = deterministic_folds(rows, seed="frozen")
    second = deterministic_folds(list(reversed(rows)), seed="frozen")
    assert first == second
    assert set(first) == {row["participant_key"] for row in rows}
    assert sorted(list(first.values()).count(fold) for fold in range(3)) == [6, 6, 6]
    original_by_fold = [
        sum(first[row["participant_key"]] == fold for row in rows[:8])
        for fold in range(3)
    ]
    assert max(original_by_fold) - min(original_by_fold) <= 1


def test_lag_grid_is_signed_nonoverlapping_and_multiscale() -> None:
    assert lag_grid([2, 6, 18], [1, 2, 4]) == {
        2: [-8, -4, -2, 0, 2, 4, 8],
        6: [-24, -12, -6, 0, 6, 12, 24],
        18: [-72, -36, -18, 0, 18, 36, 72],
    }


def test_architecture_parameter_count_is_frozen() -> None:
    assert trainable_parameter_count(1024, 384, 256, 128) == 792_832


def test_synthetic_positive_control_recovers_injected_relationship() -> None:
    try:
        result = run_embedding_positive_control(bootstrap_replicates=1000)
    except BridgeV5Error as exc:
        if exc.code == "E_TORCH_UNAVAILABLE":
            pytest.skip("frozen learner runtime is not installed in system Python")
        raise
    assert result["pass"] is True
    assert result["mean_aligned_minus_lagged_cosine"] >= 0.1
    assert result["participant_cluster_interval"][0] >= 0.05
    assert result["empirical_evidence"] is False
    assert result["learner_trainable_parameter_count"] == 792_832


def test_exact_three_state_decision_and_governance_failure_dominates() -> None:
    shared = {
        "governance": True,
        "support": True,
        "positive_control": True,
        "preprocessing": True,
        "shortcut_interpretable": True,
        "precision": True,
        "heterogeneity": True,
        "detectable_structure": True,
        "precise_weak_or_flat": False,
    }
    assert terminal_decision(shared) == "PASS_DETECTABLE_STRUCTURE"
    shared["detectable_structure"] = False
    shared["precise_weak_or_flat"] = True
    assert terminal_decision(shared) == "PASS_PRECISE_WEAK_OR_FLAT"
    shared["governance"] = False
    assert terminal_decision(shared) == "NO_GO_UNINFORMATIVE"


def test_protocol_has_one_route_and_forbids_every_excluded_source() -> None:
    config = json.loads(
        (ROOT / "configs/childlens_alignment_bridge_v5.json").read_text()
    )
    assert config["learner"]["route_count"] == 1
    assert config["learner"]["alternative_search_after_childlens_outcomes"] is False
    assert config["scope"]["empirical_source"] == "CHILDLENS_ONLY"
    assert config["scope"]["aea_external_volume_or_babyview_allowed"] is False
    assert config["scope"]["locked_confirmation_allowed"] is False
    assert config["scope"]["simulator_generation_allowed"] is False
    assert config["scope"]["side_cue_training_or_causal_cue_lift_claims_allowed"] is False
    assert config["stage0"]["mixed_scope_manifest_may_be_opened_or_parsed"] is False


def test_public_guard_rejects_restricted_fields_and_terminal_record_validates() -> None:
    with pytest.raises(BridgeV5Error, match="E_PUBLIC_PRIVACY"):
        public_guard({"participant_key": "private"})
    with pytest.raises(BridgeV5Error, match="E_PUBLIC_PRIVACY"):
        public_guard({"note": "/Users/private/source"})
    runner = _runner()
    receipt = runner.validate(write=False)
    assert receipt["status"] == "PASS_FOR_TERMINAL_STAGE0_RECORD"
    assert receipt["development_decision"] == "NO_GO_UNINFORMATIVE"
    assert receipt["zero_locked_access_certifiable"] is False
