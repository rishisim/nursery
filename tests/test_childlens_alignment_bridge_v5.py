from __future__ import annotations

import importlib.util
import hashlib
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


def test_administrative_correction_preserves_frozen_method_and_clean_stage0_passes() -> None:
    config_path = ROOT / "configs/childlens_alignment_bridge_v5.json"
    assert (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        == "b048ca4f4950eaf37d8e751a88ee358d5eabeb83b5187302502d9e08d62b130d"
    )
    correction = json.loads(
        (
            ROOT
            / "output/childlens_alignment_bridge_v5/"
            "administrative_correction_receipt.json"
        ).read_text()
    )
    stage0 = json.loads(
        (
            ROOT
            / "output/childlens_alignment_bridge_v5/"
            "clean_stage0_freeze_receipt.json"
        ).read_text()
    )
    assert correction["status"] == "PASS"
    assert correction["development_participant_count"] == 18
    assert correction["locked_participant_count"] == 0
    assert correction["scientific_method_changed"] is False
    assert stage0["status"] == "FROZEN_BEFORE_SELECTIVE_ACQUISITION_OR_MEDIA_DECODING"
    assert stage0["support_gate_pass"] is True
    assert stage0["target_seconds_per_participant"] == 900
    assert stage0["fold_counts"] == [6, 6, 6]
    assert min(stage0["minimum_windows_per_participant_by_duration_seconds"].values()) >= 40


def test_clean_input_guard_rejects_attestation_with_locked_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    runtime = tmp_path / "restricted_manifest"
    private = (
        runtime
        / "provisional_calibration_v1/childlens_alignment_bridge_v5/administrative"
    )
    private.mkdir(parents=True, mode=0o700)
    input_path = private / "development_only_scientific_input.json"
    attestation_path = private / "development_only_attestation.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": "childlens-v5-development-only-scientific-input-v1.0.0",
                "clean_run_id": runner.CLEAN_RUN_ID,
                "frozen_v5_config_sha256": runner.FROZEN_CONFIG_SHA256,
                "development_participant_count": 18,
                "locked_participant_count": 1,
                "items": [],
            }
        )
    )
    input_path.chmod(0o600)
    attestation_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scientific_input_sha256": hashlib.sha256(
                    input_path.read_bytes()
                ).hexdigest(),
                "development_participant_count": 18,
                "locked_participant_count": 1,
                "legacy_mixed_scope_inputs_available_to_scientific_process": False,
            }
        )
    )
    attestation_path.chmod(0o600)
    public_path = tmp_path / "administrative_correction_receipt.json"
    public_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "clean_run_id": runner.CLEAN_RUN_ID,
                "frozen_v5_config_sha256": runner.FROZEN_CONFIG_SHA256,
                "incident_receipt_sha256": runner.ORIGINAL_INCIDENT_RECEIPT_SHA256,
                "attestation_sha256": hashlib.sha256(
                    attestation_path.read_bytes()
                ).hexdigest(),
                "scientific_input_sha256": hashlib.sha256(
                    input_path.read_bytes()
                ).hexdigest(),
                "locked_participant_count": 1,
                "legacy_mixed_scope_inputs_available_to_scientific_process": False,
            }
        )
    )
    monkeypatch.setattr(runner, "_discover_attested_runtime", lambda: runtime)
    monkeypatch.setattr(runner, "ADMIN_RECEIPT", public_path)
    with pytest.raises(BridgeV5Error, match="E_ATTESTED_INPUT"):
        runner._attested_input()


def test_original_incident_cannot_be_overwritten() -> None:
    runner = _runner()
    with pytest.raises(BridgeV5Error, match="E_IMMUTABLE_INCIDENT_RECORD"):
        runner.freeze_and_stop()


def test_generated_clip_is_secured_before_private_guard(tmp_path: Path) -> None:
    runner = _runner()
    generated = tmp_path / "derived.mp4"
    generated.write_bytes(b"synthetic-media-placeholder")
    generated.chmod(0o644)
    assert runner._private_file(generated) is False
    assert runner._secure_generated_file(generated) is True
    assert generated.stat().st_mode & 0o777 == 0o600

    symlink = tmp_path / "derived-link.mp4"
    symlink.symlink_to(generated)
    assert runner._secure_generated_file(symlink) is False


def test_temporal_stats_round_trip_through_two_step_sequence() -> None:
    runner = _runner()
    numpy = pytest.importorskip("numpy")
    rng = numpy.random.default_rng(17)
    mean = rng.normal(size=(5, 7)).astype("float32")
    standard = numpy.abs(rng.normal(size=(5, 7))).astype("float32")
    stats = numpy.stack((mean, standard), axis=1)
    sequence = runner._sequence_from_stats(stats)
    numpy.testing.assert_allclose(sequence.mean(axis=1), mean, atol=1e-6)
    numpy.testing.assert_allclose(sequence.std(axis=1), standard, atol=1e-6)


def test_nonfinite_private_values_are_safely_suppressed() -> None:
    runner = _runner()
    assert runner._finite_json(
        {"finite": 1.0, "missing": float("nan"), "nested": [float("inf")]}
    ) == {"finite": 1.0, "missing": None, "nested": [None]}


def test_clean_terminal_result_is_support_failure_not_threshold_rescue() -> None:
    public = ROOT / "output/childlens_alignment_bridge_v5"
    decision = json.loads((public / "clean_development_decision.json").read_text())
    validation = json.loads((public / "clean_validation_receipt.json").read_text())
    assert decision["decision"] == "NO_GO_UNINFORMATIVE"
    assert decision["gates"]["governance"] is True
    assert decision["gates"]["positive_control"] is True
    assert decision["gates"]["preprocessing"] is True
    assert decision["gates"]["support"] is False
    assert decision["support_gate_minimum_required"] == 40
    assert decision["support_gate_minimum_observed"] < 40
    assert decision["primary_mean_lift"] is None
    assert validation["development_participant_count"] == 18
    assert validation["locked_participant_count"] == 0
    assert validation["simulator_or_side_cue_training_run"] is False
