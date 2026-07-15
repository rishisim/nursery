from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import torch

from babyworld_lite.aea.visible_action_v2 import (
    apply_protocol_amendment,
    construct_constrained_folds,
    group_safe_bijection,
    packet_order,
    select_frozen_sample,
    summarize_annotations,
    validate_frozen_development_inputs,
)
from scripts.finalize_aea_visible_action_v2 import _acquisition_gate, _decision
from scripts.run_aea_visible_action_v2_capacity import _action_head_run, _transcript_run


ROOT = Path(__file__).parents[1]


def effective_protocol() -> dict:
    protocol = json.loads(
        (ROOT / "output/aea_visible_action_v2/preregistered_protocol.json").read_text()
    )
    amendment = json.loads(
        (ROOT / "output/aea_visible_action_v2/preregistered_protocol_amendment_1.json").read_text()
    )
    return apply_protocol_amendment(protocol, amendment)


def test_frozen_sample_is_deterministic_group_aware_and_reserve_free() -> None:
    protocol = effective_protocol()
    rows, prior, receipt = validate_frozen_development_inputs(
        ROOT / "output/aea_dev_learnability_v1/development_examples.jsonl",
        ROOT / "output/aea_dev_learnability_v1/partition_manifest.json",
        ROOT / "output/aea_dev_learnability_v1/audit_manifest_prelabel.json",
        protocol,
    )
    first = select_frozen_sample(rows, prior, protocol)
    second = select_frozen_sample(rows, prior, protocol)
    assert first == second
    assert len(first) == 72
    assert sum(row["prior_v1_audit_id"] for row in first) == 48
    assert len({row["event_group"] for row in first}) == 18
    assert not ({row["event_group"] for row in first} & set(protocol["sources"]["reserve_event_groups"]))
    source_support = Counter(row["event_group"] for row in rows)
    sample_support = Counter(row["event_group"] for row in first)
    assert all(sample_support[group] >= min(4, count) for group, count in source_support.items())
    assert receipt["reserve_rgb_files_opened"] == receipt["reserve_imu_files_opened"] == 0
    order_a = [row["blind_id"] for row in packet_order(first, protocol["sample"]["pass_order_salts"]["pass_a"])]
    order_b = [row["blind_id"] for row in packet_order(first, protocol["sample"]["pass_order_salts"]["pass_b"])]
    assert order_a != order_b
    assert set(order_a) == set(order_b)


def synthetic_annotations() -> tuple[list[dict], list[dict], list[dict]]:
    sample = []
    labels = []
    for index in range(72):
        action = "locomotion_posture" if index % 2 == 0 else "reach_grasp"
        blind_id = f"VA2-{index:012d}"
        sample.append({
            "blind_id": blind_id,
            "example_id": f"example-{index}",
            "sequence_id": f"sequence-{index % 18}",
            "event_group": f"group-{index % 18}",
            "location": index % 5 + 1,
        })
        labels.append({
            "blind_id": blind_id,
            "asr_refers_to_visible_wearer_action": "yes" if index % 2 == 0 else "no",
            "agency": "wearer" if index % 2 == 0 else "other_person",
            "temporal_alignment": "aligned_within_window" if index % 2 == 0 else "no_corresponding_action",
            "observable_action": action,
            "confidence": "high",
            "evidence_frame_start": 5,
            "evidence_frame_end": 20,
            "rationale": "Visible action is clear across the dense temporal sequence.",
        })
    return sample, labels, [dict(row) for row in reversed(labels)]


def test_agreement_gate_and_consensus_support_use_all_72_rows() -> None:
    protocol = effective_protocol()
    sample, pass_a, pass_b = synthetic_annotations()
    summary = summarize_annotations(pass_a, pass_b, sample, protocol)
    assert summary["annotation_gate_passed"]
    assert not summary["severe_annotation_failure"]
    assert summary["rates"]["modeled_consensus_yield"]["successes"] == 72
    assert set(summary["retained_support"]) == {"locomotion_posture", "reach_grasp"}
    assert summary["agreement"]["observable_action"]["cohens_kappa_unweighted"] == 1.0


def test_group_safe_bijection_uses_theorem_and_matching() -> None:
    rows = [
        {"example_id": f"e{index}", "event_group": group}
        for index, group in enumerate(["a", "a", "b", "b", "c", "c"])
    ]
    first = group_safe_bijection(range(6), rows, 6201, "test")
    second = group_safe_bijection(range(6), rows, 6201, "test")
    assert first == second
    assert first["feasible"] and first["whole_window_bijection"]
    assert first["self_match_count"] == first["same_event_group_match_count"] == 0
    invalid_rows = [
        {"example_id": f"x{index}", "event_group": group}
        for index, group in enumerate(["a", "a", "a", "b"])
    ]
    invalid = group_safe_bijection(range(4), invalid_rows, 6201, "test")
    assert not invalid["feasible"]
    assert not invalid["half_size_theorem_passed"]


def split_rows() -> tuple[list[dict], list[dict]]:
    all_rows = []
    consensus = []
    for group_number in range(18):
        group = f"group-{group_number:02d}"
        location = group_number % 5 + 1
        all_rows.append({"event_group": group, "location": location})
        for label in ("locomotion_posture", "reach_grasp"):
            index = len(consensus)
            consensus.append({
                "example_id": f"example-{index}",
                "sequence_id": f"sequence-{group_number:02d}",
                "event_group": group,
                "location": location,
                "observable_action": label,
            })
    return consensus, all_rows


def test_constrained_fold_solver_is_deterministic_leakage_and_donor_safe() -> None:
    protocol = effective_protocol()
    consensus, all_rows = split_rows()
    first = construct_constrained_folds(
        consensus, all_rows, ("locomotion_posture", "reach_grasp"), protocol
    )
    second = construct_constrained_folds(
        consensus, all_rows, ("locomotion_posture", "reach_grasp"), protocol
    )
    assert first["status"] == "feasible"
    assert first["group_assignments"] == second["group_assignments"]
    assert first["checks"]["each_endpoint_row_held_out_once"]
    assert first["checks"]["zero_event_group_leakage"]
    assert first["checks"]["all_training_donors_feasible"]
    assert first["checks"]["all_imu_test_donors_feasible"]
    assert all(len(fold["test_event_groups"]) == 6 for fold in first["folds"])


def test_constrained_fold_solver_reports_infeasibility_without_relaxation() -> None:
    protocol = effective_protocol()
    consensus, all_rows = split_rows()
    # Concentrate one otherwise globally retained label in four groups. It cannot
    # occupy at least two test groups in each of three disjoint folds.
    concentrated = [
        row for row in consensus
        if row["observable_action"] == "reach_grasp"
        or int(row["event_group"].split("-")[-1]) < 4
    ]
    result = construct_constrained_folds(
        concentrated, all_rows, ("locomotion_posture", "reach_grasp"), protocol
    )
    assert result["status"] == "infeasible_certified"
    assert result["solver"]["runs"] == 1
    assert not result["relaxation_or_retry_performed"]
    assert not result["split_gate_passed"]


def test_frozen_decision_and_acquisition_precedence() -> None:
    integrity = {"ok": True}
    annotation = {"severe_annotation_failure": False, "annotation_gate_passed": True}
    split = {"status": "feasible", "split_gate_passed": True}
    capacity = {"capacity_gate_passed": True}
    imu = {"imu_gate_passed": True}
    assert _decision(annotation, split, capacity, imu, integrity)["recommendation"] == "GO"
    imu["imu_gate_passed"] = False
    assert _decision(annotation, split, capacity, imu, integrity)["recommendation"] == "STOP"
    annotation["annotation_gate_passed"] = False
    assert _decision(annotation, split, capacity, imu, integrity)["recommendation"] == "REVISE"
    annotation["severe_annotation_failure"] = True
    assert _decision(annotation, split, capacity, imu, integrity)["recommendation"] == "STOP"

    gate = _acquisition_gate(
        {"annotation_gate_passed": True},
        {"status": "infeasible_certified"},
        {"capacity_gate_passed": True},
        {"status": "not_run"},
    )
    assert gate["passed"] and gate["certified_support_bottleneck"]
    assert not _acquisition_gate(
        {"annotation_gate_passed": False},
        {"status": "infeasible_certified"},
        {"capacity_gate_passed": True},
        {"status": "not_run"},
    )["passed"]


def test_action_head_and_transcript_controls_execute_without_motor() -> None:
    videos = torch.randint(0, 256, (4, 2, 3, 16, 16), dtype=torch.uint8)
    labels = ("locomotion_posture", "reach_grasp")
    indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    config = {
        "hidden_dim": 8,
        "learning_rate": 0.001,
        "batch_size": 2,
        "epochs": 2,
    }
    action = _action_head_run(videos, indices, labels, config, 7201, torch.device("cpu"))
    assert 0 <= action["training_balanced_accuracy"] <= 1
    rows = [
        {"transcript": f"example transcript {index}", "observable_action": labels[index // 2]}
        for index in range(4)
    ]
    transcript = _transcript_run(
        videos, rows, labels, {**config, "seed": 7299}, torch.device("cpu")
    )
    assert 0 <= transcript["training_action_balanced_2afc"] <= 1
    assert transcript["imu_files_opened"] == 0
    assert not transcript["motor_encoder_instantiated"]
