from __future__ import annotations

import json
from pathlib import Path

import pytest

from babyworld_lite.aea.coarse_action_v3 import (
    MODELED_ACTIONS,
    construct_constrained_folds,
    group_safe_bijection,
    load_json,
    summarize_annotations,
    validate_stage1_rows,
    validate_stage2_rows,
    verify_protocol_freeze,
)
from scripts.finalize_aea_coarse_action_v3 import _acquisition_gate, _decision


ROOT = Path(__file__).parents[1]
V3 = ROOT / "output/aea_coarse_action_v3"


def protocol() -> dict:
    return load_json(V3 / "preregistered_protocol.json")


def test_v3_protocol_freeze_and_fixed_sample_are_exact_and_reserve_free() -> None:
    frozen, checks = verify_protocol_freeze(
        V3 / "preregistered_protocol.json",
        V3 / "protocol_freeze_receipt.json",
        ROOT / "docs/aea_coarse_action_v3_preregistration.md",
        V3 / "annotation_codebook.md",
    )
    assert all(checks["checks"].values())
    assert frozen["sources"]["sample_size"] == 72
    manifest = load_json(V3 / "fixed_dense_manifest.json")
    assert manifest["sample_size"] == 72
    assert len({row["example_id"] for row in manifest["rows"]}) == 72
    assert len({row["event_group"] for row in manifest["rows"]}) == 18
    assert manifest["reserve_groups_present"] == []
    assert all(len(row["dense_frame_paths"]) == 31 for row in manifest["rows"])


def test_v3_visual_packets_hide_language_and_use_distinct_orders() -> None:
    packet_a = load_json(V3 / "annotation_packet_pass_a_stage1_visual.json")
    packet_b = load_json(V3 / "annotation_packet_pass_b_stage1_visual.json")
    assert len(packet_a["rows"]) == len(packet_b["rows"]) == 72
    assert packet_a["transcript_or_anchored_verb_present"] is False
    for packet in (packet_a, packet_b):
        assert all("transcript" not in row and "anchored_asr_verb" not in row for row in packet["rows"])
    ids_a = [row["blind_id"] for row in packet_a["rows"]]
    ids_b = [row["blind_id"] for row in packet_b["rows"]]
    assert ids_a != ids_b
    assert set(ids_a) == set(ids_b)


def synthetic_rows(language_aligned: int = 18) -> tuple[list[dict], list[dict], list[dict]]:
    manifest = []
    annotations = []
    for index in range(72):
        label = MODELED_ACTIONS[index % 2]
        blind_id = f"CA3-{index:012d}"
        manifest.append({
            "blind_id": blind_id,
            "example_id": f"example-{index}",
            "sequence_id": f"sequence-{index % 18}",
            "event_group": f"group-{index % 18}",
            "location": index % 5 + 1,
        })
        wearer = index < language_aligned
        annotations.append({
            "blind_id": blind_id,
            "observable_action": label,
            "visible_confidence": "high",
            "evidence_frame_start": 4,
            "evidence_frame_end": 24,
            "visible_rationale": "The wearer action is clear across the ordered visual evidence.",
            "asr_referent": "wearer_action" if wearer else "nonwearer_or_nonliteral",
            "temporal_relation": "aligned" if wearer else "none",
            "language_confidence": "high",
            "language_rationale": "The anchored expression has a clear literal referent and timing.",
        })
    return manifest, annotations, [dict(row) for row in reversed(annotations)]


def test_v3_stage_validators_enforce_sealed_interface() -> None:
    p = protocol()
    manifest, combined, _ = synthetic_rows()
    ids = {row["blind_id"] for row in manifest}
    stage1 = [{key: row[key] for key in (
        "blind_id", "observable_action", "visible_confidence",
        "evidence_frame_start", "evidence_frame_end", "visible_rationale",
    )} for row in combined]
    stage2 = [{key: row[key] for key in (
        "blind_id", "asr_referent", "temporal_relation",
        "language_confidence", "language_rationale",
    )} for row in combined]
    assert len(validate_stage1_rows(stage1, ids, p)) == 72
    assert len(validate_stage2_rows(stage2, ids, p)) == 72
    stage1[0]["transcript"] = "forbidden"
    with pytest.raises(ValueError):
        validate_stage1_rows(stage1, ids, p)


def test_v3_coarse_and_language_gates_use_all_72_rows() -> None:
    manifest, pass_a, pass_b = synthetic_rows(language_aligned=18)
    summary = summarize_annotations(pass_a, pass_b, manifest, protocol())
    assert summary["coarse_annotation_gate_passed"]
    assert summary["language_alignment_gate_passed"]
    assert summary["rates"]["language_aligned_consensus"]["successes"] == 18
    assert summary["rates"]["modeled_consensus_yield"]["successes"] == 72
    assert set(summary["support"]) == set(MODELED_ACTIONS)


def test_v3_language_gate_is_terminal_and_separate_from_coarse_gate() -> None:
    manifest, pass_a, pass_b = synthetic_rows(language_aligned=17)
    summary = summarize_annotations(pass_a, pass_b, manifest, protocol())
    assert summary["coarse_annotation_gate_passed"]
    assert not summary["language_alignment_gate_passed"]
    assert summary["language_conclusion"] == "STOP_LANGUAGE_ROUTE"


def test_v3_group_safe_bijection_proves_feasibility_and_failure() -> None:
    rows = [
        {"example_id": f"e{index}", "event_group": group}
        for index, group in enumerate(["a", "a", "b", "b", "c", "c"])
    ]
    valid = group_safe_bijection(range(6), rows, 6301, "v3-test")
    assert valid["feasible"] and valid["whole_window_bijection"]
    assert valid["self_match_count"] == valid["same_event_group_match_count"] == 0
    invalid_rows = [
        {"example_id": f"x{index}", "event_group": group}
        for index, group in enumerate(["a", "a", "a", "b"])
    ]
    invalid = group_safe_bijection(range(4), invalid_rows, 6301, "v3-test")
    assert not invalid["feasible"]
    assert not invalid["half_size_theorem_passed"]


def split_rows() -> tuple[list[dict], list[dict]]:
    sample = []
    consensus = []
    for group_number in range(18):
        group = f"group-{group_number:02d}"
        location = group_number % 5 + 1
        sample.append({"event_group": group, "location": location})
        for label in MODELED_ACTIONS:
            for repeat in range(2):
                index = len(consensus)
                consensus.append({
                    "example_id": f"example-{index}",
                    "sequence_id": f"sequence-{group_number:02d}-{repeat}",
                    "event_group": group,
                    "location": location,
                    "observable_action": label,
                })
    return consensus, sample


def test_v3_constrained_fold_is_deterministic_group_and_donor_safe() -> None:
    consensus, sample = split_rows()
    first = construct_constrained_folds(consensus, sample, protocol())
    second = construct_constrained_folds(consensus, sample, protocol())
    assert first["status"] == "feasible"
    assert first["group_assignments"] == second["group_assignments"]
    assert first["checks"]["each_endpoint_row_held_out_once"]
    assert first["checks"]["zero_event_group_leakage"]
    assert first["checks"]["all_training_donors_feasible"]
    assert first["checks"]["all_imu_test_donors_feasible"]


def test_v3_terminal_decision_keeps_language_and_sensor_conclusions_separate() -> None:
    integrity = {"ok": True}
    agreement = {"coarse_annotation_gate_passed": True}
    language = {"language_alignment_gate_passed": True}
    split = {"split_gate_passed": True}
    capacity = {"capacity_gate_passed": True}
    imu = {"imu_gate_passed": True}
    go = _decision(agreement, language, split, capacity, imu, integrity)
    assert go["formal_decision"] == "GO"
    assert go["scientific_conclusion"] == "GO_TO_TWO_HUMAN_ANNOTATORS"
    language["language_alignment_gate_passed"] = False
    sensor_only = _decision(agreement, language, split, capacity, imu, integrity)
    assert sensor_only["formal_decision"] == "REVISE"
    assert sensor_only["scientific_conclusion"] == "RETAIN_SENSOR_ANALOG_ONLY"
    assert sensor_only["language_conclusion"] == "STOP_LANGUAGE_ROUTE"
    imu["imu_gate_passed"] = False
    stopped = _decision(agreement, language, split, capacity, imu, integrity)
    assert stopped["formal_decision"] == "STOP"
    assert stopped["scientific_conclusion"] == "STOP_AEA_ROUTE"


def test_v3_acquisition_gate_rejects_valid_null_and_language_failure() -> None:
    agreement = {"coarse_annotation_gate_passed": True}
    language = {"language_alignment_gate_passed": False}
    split = {"status": "feasible"}
    capacity = {"capacity_gate_passed": True}
    imu = {
        "status": "complete",
        "imu_gate_passed": False,
        "gate_checks": {
            "synchronized_accuracy_above_chance_margin": False,
            "bootstrap_lower_above_chance": False,
            "chance_permutation_p": False,
            "synchronized_minus_donor_minimum": False,
            "paired_bootstrap_lower_above_zero": False,
            "paired_randomization_p": False,
            "train_minus_heldout_gap": True,
        },
        "synchronized_minus_donor": -0.01,
        "heldout_synchronized_balanced_accuracy": 0.48,
        "macro_chance": 0.5,
        "train_minus_heldout_gap": 0.2,
    }
    decision = {
        "aea_survives_as_language_analogue": False,
        "aea_survives_as_sensor_analogue": False,
    }
    gate = _acquisition_gate(
        agreement, language, split, capacity, imu, decision
    )
    assert not gate["passed"]
    assert gate["language_failure_forbids_language_grounding_acquisition"]


def test_v3_terminal_artifacts_stop_without_capacity_or_imu_access() -> None:
    results = load_json(V3 / "aea_coarse_action_v3_results.json")
    assert results["all_hard_integrity_checks_passed"]
    assert results["decision"]["formal_decision"] == "STOP"
    assert results["decision"]["scientific_conclusion"] == "STOP_AEA_ROUTE"
    assert results["decision"]["language_conclusion"] == "STOP_LANGUAGE_ROUTE"
    assert results["annotation"]["rates"]["modeled_consensus_yield"]["successes"] == 43
    assert results["language_alignment"]["rates"]["language_aligned_consensus"]["successes"] == 7
    assert results["capacity"]["status"] == "not_run_stage_gate_failed"
    assert results["imu"]["status"] == "not_run_stage_gate_failed"
    assert results["imu"]["development_imu_arrays_opened"] == 0
    assert results["imu"]["reserve_imu_arrays_opened"] == 0
    assert not results["acquisition"]["recommend_additional_recordings"]
