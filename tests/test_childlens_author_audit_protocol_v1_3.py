from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCRIPT = ROOT / "scripts" / "childlens_author_audit_protocol_v1_3.py"
SAMPLER_SCRIPT = ROOT / "scripts" / "prepare_childlens_author_audit_packet_v1_3.py"
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "childlens_feasibility_v1_3"
    / "frozen_model_assisted_author_audit_protocol_v1_3.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol_module = load_module("childlens_author_audit_protocol_v1_3", PROTOCOL_SCRIPT)
sampler_module = load_module("prepare_childlens_author_audit_packet_v1_3_contract", SAMPLER_SCRIPT)


def test_protocol_self_binding_and_exact_schema_validate() -> None:
    protocol = protocol_module.load_protocol(PROTOCOL_PATH)
    assert protocol["schema_version"] == protocol_module.SCHEMA_VERSION
    assert protocol["status"] == "FROZEN_BEFORE_AUTHOR_LABELS"
    assert protocol["supersedes_prior_protocols"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("audit_sampling", "primary_target_total_ms"), 899_000),
        (("audit_sampling", "reserve_target_total_ms"), 899_000),
        (("author_audit", "author_blinded_to_predictions"), False),
        (("agreement_gates", "source_role", "pass_thresholds", "non_child_precision_min"), 0.5),
        (("bounded_escalation", "maximum_activations"), 2),
        (("scientific_boundaries", "cloud_or_external_inference"), True),
    ],
)
def test_any_frozen_policy_or_threshold_mutation_is_rejected(
    path: tuple[str, ...], replacement: object
) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cursor = protocol
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(protocol_module.ProtocolError, match="E_PROTOCOL_MUTATED"):
        protocol_module.validate_protocol(protocol)


def test_runtime_sampler_and_frozen_protocol_have_identical_duration_semantics() -> None:
    protocol = protocol_module.load_protocol(PROTOCOL_PATH)
    sampling = protocol["audit_sampling"]
    assert sampler_module.EXPECTED_ITEM_COUNT == sampling["selected_item_count"] == 15
    assert sampler_module.TARGET_SAMPLE_MS == sampling["primary_target_total_ms"] == 900_000
    assert sampler_module.TARGET_SAMPLE_MS == sampling["reserve_target_total_ms"] == 900_000
    assert sampler_module.PREFERRED_ITEM_MS == sampling["preferred_per_item_ms"] == 60_000
    assert (
        sampler_module.MINIMUM_CLUSTER_MS_PER_SAMPLE
        == sampling["minimum_per_item_per_sample_ms"]
        == 1_000
    )
    assert set(sampling["permitted_manifest_top_level_fields"]) == sampler_module.TOP_LEVEL_FIELDS
    assert set(sampling["permitted_item_input_fields"]) == sampler_module.ITEM_FIELDS
    assert set(sampling["permitted_speech_window_fields"]) == sampler_module.WINDOW_FIELDS


def test_authoritative_sampler_allocates_exact_disjoint_15_minute_samples_synthetically() -> None:
    capacities = {f"synthetic-{index:02d}": 180_000 for index in range(15)}
    primary, reserve = sampler_module._allocate(capacities, "a" * 64)
    assert sum(primary.values()) == sum(reserve.values()) == 900_000
    assert all(value == 60_000 for value in primary.values())
    assert all(value == 60_000 for value in reserve.values())

    windows = [(0, 90_000), (120_000, 210_000)]
    first, second = sampler_module._place_samples(
        item_key="synthetic-item",
        selection="a" * 64,
        windows=windows,
        primary_duration=60_000,
        reserve_duration=60_000,
    )
    assert sum(row["end_ms"] - row["start_ms"] for row in first) == 60_000
    assert sum(row["end_ms"] - row["start_ms"] for row in second) == 60_000
    assert sampler_module._physical_overlap(first, second) is False


def test_authoritative_sampler_redistributes_short_item_but_preserves_every_cluster() -> None:
    capacities = {f"synthetic-{index:02d}": 180_000 for index in range(15)}
    capacities["synthetic-00"] = 30_000
    primary, reserve = sampler_module._allocate(capacities, "b" * 64)
    assert sum(primary.values()) == sum(reserve.values()) == 900_000
    assert all(value >= 1_000 for value in primary.values())
    assert all(value >= 1_000 for value in reserve.values())
    assert primary["synthetic-00"] == reserve["synthetic-00"] == 15_000


def test_runtime_sampler_policy_is_prediction_and_content_independent() -> None:
    policy = sampler_module.POLICY
    assert policy["model_predictions_used"] is False
    assert policy["confidence_used"] is False
    assert policy["lexical_content_used"] is False
    assert policy["visual_content_used"] is False
    assert policy["reserve_activation"] == "BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK"


def test_public_policy_receipt_is_aggregate_only_and_does_not_claim_observed_allocation() -> None:
    receipt = protocol_module.build_public_protocol_receipt()
    serialized = json.dumps(receipt, sort_keys=True)
    assert receipt["target_selected_item_count"] == 15
    assert receipt["target_selected_speech_minutes"] == 15.0
    assert receipt["target_reserve_speech_minutes"] == 15.0
    assert receipt["one_minute_per_item_preferred"] is True
    assert receipt["observed_sample_receipt_required_separately"] is True
    assert receipt["author_blinded_to_predictions"] is True
    assert receipt["author_lock_required_before_comparison"] is True
    assert receipt["second_human_required"] is False
    assert receipt["inter_human_reliability_available"] is False
    for forbidden in ("item_key", "segments", "timestamp", "filename", "transcript"):
        assert forbidden not in serialized.lower()


def test_all_required_task_gates_and_thresholds_are_frozen() -> None:
    protocol = protocol_module.load_protocol(PROTOCOL_PATH)
    gates = protocol["agreement_gates"]
    assert set(gates) == {
        "boundary_matching",
        "source_role",
        "non_child_transcription",
        "coarse_referential",
        "noun_object_candidates",
        "verb_action_candidates",
    }
    assert gates["boundary_matching"]["declared_tolerances_ms"] == {
        "strict_both_edges": 500,
        "segment_match": 1000,
    }
    assert gates["source_role"]["pass_thresholds"]["non_child_precision_min"] == 0.85
    assert gates["non_child_transcription"]["pass_thresholds"] == {
        "cer_max": 0.25,
        "wer_max_if_applicable": 0.4,
        "usable_transcript_coverage_min": 0.75,
    }
    assert gates["coarse_referential"]["pass_thresholds"]["visible_precision_min"] == 0.75
    assert gates["noun_object_candidates"]["pass_thresholds"] == {
        "candidate_precision_min": 0.75,
        "candidate_coverage_min": 0.5,
    }
    assert gates["verb_action_candidates"]["pass_thresholds"] == {
        "candidate_precision_min": 0.7,
        "candidate_coverage_min": 0.4,
    }


def test_cluster_uncertainty_and_small_sample_limits_are_explicit() -> None:
    protocol = protocol_module.load_protocol(PROTOCOL_PATH)
    uncertainty = protocol["cluster_aware_uncertainty"]
    assert uncertainty["cluster_unit"].startswith("FROZEN_SELECTED_ITEM")
    assert uncertainty["confidence_level"] == 0.9
    assert uncertainty["replicates"] == 10_000
    assert uncertainty["minimum_defined_replicate_fraction"] == 0.8
    assert uncertainty["maximum_single_item_support_share_for_PASS"] == 0.3
    assert "ONE_AUTHOR_CANNOT_ESTIMATE_INTER_HUMAN_RELIABILITY" in protocol["limitations"]


def test_exactly_one_borderline_only_escalation_and_no_outcome_adaptation() -> None:
    protocol = protocol_module.load_protocol(PROTOCOL_PATH)
    escalation = protocol["bounded_escalation"]
    assert escalation["maximum_activations"] == 1
    assert escalation["maximum_additional_speech_ms"] == 900_000
    assert escalation["activation_condition"] == "PRIMARY_BORDERLINE_ONLY"
    assert escalation["sample_source"] == "PREFROZEN_DISJOINT_BORDERLINE_RESERVE"
    assert "LEARNER_OUTCOME" in escalation["activation_prohibited_for"]


def test_author_blinding_lock_and_pseudo_label_separation_are_frozen() -> None:
    protocol = protocol_module.load_protocol(PROTOCOL_PATH)
    author = protocol["author_audit"]
    boundaries = protocol["scientific_boundaries"]
    assert author["prediction_store_mounted_in_author_app_before_lock"] is False
    assert author["comparison_before_author_lock"] is False
    assert author["author_lock_mutable"] is False
    assert author["model_human_agreement_only"] is True
    assert boundaries["hosted_model_content_access"] is False
    assert boundaries["cloud_or_external_inference"] is False
    assert boundaries["unaudited_pseudo_labels_are_primary_evaluation_truth"] is False
    assert boundaries["simulator_oracle_labels_are_primary_causal_evaluation_truth"] is True


def test_frozen_payload_digest_covers_every_scientific_component() -> None:
    protocol = protocol_module.load_protocol(PROTOCOL_PATH)
    frozen = set(protocol["freeze_binding"]["frozen_top_level_fields"])
    assert frozen == {
        "scientific_interpretation",
        "scientific_boundaries",
        "audit_sampling",
        "author_audit",
        "annotation_ontology",
        "comparison_rules",
        "agreement_gates",
        "cluster_aware_uncertainty",
        "bounded_escalation",
        "decision_logic",
        "prohibited_adaptations",
        "limitations",
    }


def test_protocol_validation_does_not_mutate_input() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    before = copy.deepcopy(protocol)
    protocol_module.validate_protocol(protocol)
    assert protocol == before
