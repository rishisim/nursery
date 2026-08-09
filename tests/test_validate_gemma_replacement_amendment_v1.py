from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/validate_gemma_replacement_amendment_v1.py"
AMENDMENT_PATH = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_replacement_instrument_amendment.json"
BASELINE_PATH = ROOT / "docs/nursery_program_convergence_v1/frozen_childlens_pseudo_calibration_protocol.json"
SPEC = importlib.util.spec_from_file_location("validate_gemma_replacement", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _sha(char: str) -> str:
    return char * 64


@pytest.fixture()
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def amendment() -> dict:
    return json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def baseline_sha() -> str:
    return validator._sha256_bytes(BASELINE_PATH.read_bytes())


@pytest.fixture()
def amendment_sha() -> str:
    return validator._sha256_bytes(AMENDMENT_PATH.read_bytes())


def _activation(amendment: dict, amendment_sha: str) -> dict:
    return {
        "schema_version": validator.ACTIVATION_SCHEMA,
        "status": "ACTIVE_ALL_GATES_PASS",
        "amendment_sha256": amendment_sha,
        "all_placeholders_replaced": True,
        "restricted_inference_before_activation": False,
        "qwen3_reuse": {
            "instrument_id": validator.QWEN3_RETAINED,
            "read_only": True,
            "predictions_recomputed": False,
            "restricted_output_digest_match": True,
            "restricted_output_sha256": _sha("a"),
        },
        "gemma_instrument": {
            "instrument_id": validator.GEMMA_ADDED,
            "upstream_repository": "google/gemma-4-E4B-it",
            "upstream_revision": "abcdef0123456789",
            "conversion_revision": "123456789abcdef0",
            "license_reconciliation_receipt_sha256": _sha("b"),
            "upstream_model_card_sha256": _sha("c"),
            "conversion_model_card_sha256": _sha("d"),
            "artifact_manifest_sha256": _sha("e"),
            "artifact_file_count": 12,
            "artifact_bytes": 5_000_000_000,
            "processor_config_sha256": _sha("f"),
            "tokenizer_config_sha256": _sha("1"),
            "chat_template_sha256": _sha("2"),
            "generation_config_sha256": _sha("3"),
            "runtime_lock_sha256": _sha("4"),
            "adapter_sha256": _sha("5"),
            "public_canary_receipt_sha256": _sha("6"),
            "upstream_license_identifier": "Apache-2.0",
            "conversion_permission_resolved": True,
            "permission_inferred_from_public_access": False,
            "agent_accepted_clickthrough": False,
            "restricted_outputs_quarantine_only": True,
        },
        "unchanged_contract": {
            "before_protocol_sha256": amendment["unchanged_contract"]["before_protocol_sha256"],
            "invariant_contract_id": validator.INVARIANT_CONTRACT_ID,
            "primary_item_count": 15,
            "primary_total_seconds": 900,
            "candidate_window_count": 137,
            "sample_reselected": False,
            "schema_or_gate_changed": False,
        },
        "bounded_correction": {
            "corrections_used": 0,
            "scope": "NONE",
            "semantic_prompt_changed": False,
            "common_schema_changed": False,
            "model_or_revision_changed": False,
            "input_or_window_changed": False,
            "threshold_or_gate_changed": False,
            "semantic_tuning_performed": False,
            "agreement_or_quality_triggered": False,
        },
        "hosted_or_cloud_inference_used": False,
        "external_request_made": False,
        "scientific_outcome_authorized": False,
        "scientific_outcome_run": False,
        "network_denial_verified": True,
        "aggregate_only_public_output": True,
    }


def _canary(activation: dict, amendment_sha: str) -> dict:
    gemma = activation["gemma_instrument"]
    return {
        "schema_version": validator.CANARY_SCHEMA,
        "status": "PASS",
        "scope": "SELF_GENERATED_PUBLIC_AUDIO_PLUS_FIVE_FRAMES_ONLY",
        "childlens_or_quarantine_accessed": False,
        "joint_single_request": True,
        "audio_consumed": True,
        "five_frames_consumed_in_order": True,
        "exact_schema_valid": True,
        "audio_nonce_recovered": True,
        "visual_nonce_recovered": True,
        "network_denial_sentinel_passed": True,
        "no_external_request": True,
        "hosted_or_cloud_inference_used": False,
        "restricted_inference_authorized_by_canary": False,
        "pseudo_output_is_human_evidence": False,
        "amendment_sha256": amendment_sha,
        "upstream_revision": gemma["upstream_revision"],
        "conversion_revision": gemma["conversion_revision"],
        "artifact_manifest_sha256": gemma["artifact_manifest_sha256"],
        "runtime_lock_sha256": gemma["runtime_lock_sha256"],
        "adapter_sha256": gemma["adapter_sha256"],
        "transport_or_parser_corrections_used": 0,
    }


def _receipt(amendment: dict, amendment_sha: str, activation_sha: str) -> dict:
    return {
        "schema_version": validator.RECEIPT_SCHEMA,
        "amendment_sha256": amendment_sha,
        "activation_receipt_sha256": activation_sha,
        "referential_instrument_ids": [validator.GEMMA_ADDED, validator.QWEN3_RETAINED],
        "third_visual_model_used": False,
        "qwen3_reused_read_only": True,
        "primary_item_count": 15,
        "primary_total_seconds": 900,
        "candidate_window_count": 137,
        "sample_reselected": False,
        "unchanged_contract": {
            "before_protocol_sha256": amendment["unchanged_contract"]["before_protocol_sha256"],
            "invariant_contract_id": validator.INVARIANT_CONTRACT_ID,
            "schema_or_gate_changed": False,
        },
        "public_export": {
            "raw_counts": False,
            "minimum_cluster_k": 5,
            "complementary_suppression": True,
            "paths": False,
            "identifiers": False,
            "filenames": False,
            "exact_timestamps_or_intervals": False,
            "transcript_or_lexical_content": False,
            "frames_or_audio": False,
            "item_level_predictions": False,
            "confidence_or_raw_model_payload": False,
            "free_form_errors": False,
            "outward_rounding": {"rate_step": 0.1, "lag_step_event_units": 0.5},
        },
        "calibration_ranges": {"visibility": {"envelope": [0.2, 0.9]}},
        "pseudo_labels_are_ground_truth": False,
        "human_evidence_available": False,
        "simulator_oracle_only_evaluation_truth": True,
        "scientific_outcome_authorized": False,
        "scientific_outcome_run": False,
        "scientific_endpoint_opened": False,
        "semantic_tuning_performed": False,
        "corrections_used": 0,
    }


def _codes(issues: list) -> set[str]:
    return {issue.code for issue in issues}


def test_live_frozen_amendment_passes(amendment: dict, baseline: dict, baseline_sha: str) -> None:
    assert validator.validate_amendment(amendment, baseline, baseline_sha) == []


def test_synthetic_activation_canary_and_receipt_pass(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    canary = _canary(activation, amendment_sha)
    canary_sha = validator.canonical_digest(canary)
    activation["gemma_instrument"]["public_canary_receipt_sha256"] = canary_sha
    activation_sha = validator.canonical_digest(activation)
    assert validator.validate_activation_receipt(activation, amendment, amendment_sha) == []
    assert validator.validate_joint_canary(canary, activation, amendment_sha, canary_sha) == []
    assert validator.validate_aggregate_receipt(_receipt(amendment, amendment_sha, activation_sha), amendment, amendment_sha, activation_sha) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("removed_path_id", validator.QWEN3_RETAINED),
        ("added_path_id", "third_model"),
        ("retained_path_id", validator.QWEN2_REMOVED),
        ("substitution_count", 2),
        ("no_third_model", False),
    ],
)
def test_exact_one_instrument_substitution(amendment: dict, baseline: dict, field: str, value: object) -> None:
    amendment["substitution"][field] = value
    assert "SUBSTITUTION_CARDINALITY" in _codes(validator.validate_amendment(amendment, baseline))


def test_before_after_sets_reject_third_model(amendment: dict, baseline: dict) -> None:
    amendment["substitution"]["after_active_referential_paths"].append("third_model")
    codes = _codes(validator.validate_amendment(amendment, baseline))
    assert {"SUBSTITUTION_SET", "SUBSTITUTION_DELTA"} <= codes


@pytest.mark.parametrize("contract", ["sample", "schema", "gates", "thresholds", "K5_suppression", "outward_rounding"])
def test_frozen_contract_cannot_change(amendment: dict, baseline: dict, contract: str) -> None:
    amendment["unchanged_contract"][contract] = False
    assert "CONTRACT_CHANGED" in _codes(validator.validate_amendment(amendment, baseline))


def test_baseline_byte_digest_is_bound(amendment: dict, baseline: dict) -> None:
    assert "BASELINE_DIGEST" in _codes(validator.validate_amendment(amendment, baseline, _sha("f")))


def test_projection_cannot_change(amendment: dict, baseline: dict) -> None:
    amendment["unchanged_contract"]["invariant_projection"]["candidate_window_count"] = 138
    assert "INVARIANT_PROJECTION" in _codes(validator.validate_amendment(amendment, baseline))


def test_qwen3_must_remain_read_only(amendment: dict, baseline: dict) -> None:
    for field, value in (("read_only", False), ("predictions_recomputed", True), ("digest_match", "PASS")):
        bad = copy.deepcopy(amendment)
        bad["qwen3_reuse"][field] = value
        assert "QWEN3_NOT_READ_ONLY" in _codes(validator.validate_amendment(bad, baseline))


def test_freeze_placeholders_cannot_claim_activation(amendment: dict, baseline: dict) -> None:
    amendment["gemma4_provenance_placeholders"]["upstream_revision"] = "abcdef0123456789"
    assert "PLACEHOLDER_ACTIVATED_IN_AMENDMENT" in _codes(validator.validate_amendment(amendment, baseline))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_corrections", 2),
        ("transport_or_parser_only", False),
        ("prompt_change", True),
        ("model_or_revision_change", True),
        ("input_or_window_change", True),
        ("semantic_bin_or_definition_change", True),
        ("threshold_or_gate_change", True),
        ("semantic_tuning", True),
        ("agreement_or_confidence_directed", True),
    ],
)
def test_only_one_nonsemantic_correction(amendment: dict, baseline: dict, field: str, value: object) -> None:
    amendment["correction_policy"][field] = value
    assert "CORRECTION_POLICY" in _codes(validator.validate_amendment(amendment, baseline))


def test_activation_is_byte_bound(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation["amendment_sha256"] = _sha("f")
    assert "ACTIVATION_AMENDMENT_BINDING" in _codes(validator.validate_activation_receipt(activation, amendment, amendment_sha))


def test_activation_requires_read_only_qwen3(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation["qwen3_reuse"]["predictions_recomputed"] = True
    assert "ACTIVATION_QWEN3" in _codes(validator.validate_activation_receipt(activation, amendment, amendment_sha))


@pytest.mark.parametrize(
    "field",
    [
        "license_reconciliation_receipt_sha256",
        "artifact_manifest_sha256",
        "processor_config_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
        "generation_config_sha256",
        "runtime_lock_sha256",
        "adapter_sha256",
        "public_canary_receipt_sha256",
    ],
)
def test_activation_requires_license_provenance_hashes(amendment: dict, amendment_sha: str, field: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation["gemma_instrument"][field] = "bad"
    assert "GEMMA_HASH_GATE" in _codes(validator.validate_activation_receipt(activation, amendment, amendment_sha))


def test_public_access_does_not_imply_permission(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation["gemma_instrument"]["conversion_permission_resolved"] = False
    activation["gemma_instrument"]["permission_inferred_from_public_access"] = True
    assert "GEMMA_LICENSE_PROVENANCE" in _codes(validator.validate_activation_receipt(activation, amendment, amendment_sha))


@pytest.mark.parametrize(
    "field",
    [
        "semantic_prompt_changed",
        "common_schema_changed",
        "model_or_revision_changed",
        "input_or_window_changed",
        "threshold_or_gate_changed",
        "semantic_tuning_performed",
        "agreement_or_quality_triggered",
    ],
)
def test_activation_cannot_semantically_tune(amendment: dict, amendment_sha: str, field: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation["bounded_correction"][field] = True
    assert "ACTIVATION_SEMANTIC_TUNING" in _codes(validator.validate_activation_receipt(activation, amendment, amendment_sha))


@pytest.mark.parametrize(
    "field",
    [
        "joint_single_request",
        "audio_consumed",
        "five_frames_consumed_in_order",
        "exact_schema_valid",
        "audio_nonce_recovered",
        "visual_nonce_recovered",
        "network_denial_sentinel_passed",
        "no_external_request",
    ],
)
def test_joint_canary_is_conjunctive(amendment: dict, amendment_sha: str, field: str) -> None:
    activation = _activation(amendment, amendment_sha)
    canary = _canary(activation, amendment_sha)
    canary[field] = False
    assert "CANARY_JOINT_MODALITY" in _codes(validator.validate_joint_canary(canary, activation, amendment_sha))


def test_canary_must_match_pinned_artifact(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    canary = _canary(activation, amendment_sha)
    canary["adapter_sha256"] = _sha("f")
    assert "CANARY_ARTIFACT_MISMATCH" in _codes(validator.validate_joint_canary(canary, activation, amendment_sha))


def test_activation_must_bind_exact_canary_bytes(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    canary = _canary(activation, amendment_sha)
    assert "CANARY_RECEIPT_BINDING" in _codes(
        validator.validate_joint_canary(canary, activation, amendment_sha, _sha("f"))
    )


def test_canary_allows_at_most_one_parser_correction(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    canary = _canary(activation, amendment_sha)
    canary["transport_or_parser_corrections_used"] = 2
    assert "CANARY_CORRECTION_COUNT" in _codes(validator.validate_joint_canary(canary, activation, amendment_sha))
    canary = _canary(activation, amendment_sha)
    canary["transport_or_parser_corrections_used"] = 1
    canary["correction"] = {"scope": "PARSER", "semantic_prompt_changed": True, "common_schema_changed": False}
    assert "CANARY_CORRECTION_SCOPE" in _codes(validator.validate_joint_canary(canary, activation, amendment_sha))


def test_receipt_rejects_third_model_and_recomputed_qwen3(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation_sha = validator.canonical_digest(activation)
    receipt = _receipt(amendment, amendment_sha, activation_sha)
    receipt["referential_instrument_ids"].append("third_model")
    assert "RECEIPT_INSTRUMENT_SET" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))
    receipt = _receipt(amendment, amendment_sha, activation_sha)
    receipt["qwen3_reused_read_only"] = False
    assert "RECEIPT_MODEL_BOUNDARY" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))


def test_receipt_is_bound_to_exact_artifacts(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation_sha = validator.canonical_digest(activation)
    receipt = _receipt(amendment, amendment_sha, activation_sha)
    receipt["activation_receipt_sha256"] = _sha("f")
    assert "RECEIPT_BINDING" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))


def test_receipt_rejects_raw_item_text_and_confidence(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation_sha = validator.canonical_digest(activation)
    for field, value in (("transcript_text", "restricted"), ("source_filename", "restricted.mp4"), ("participant_id", "person"), ("item_rows", [{"item_id": "item"}]), ("exact_interval", [1.0, 2.0]), ("confidence_score", 0.9), ("audio_content", "bytes")):
        receipt = _receipt(amendment, amendment_sha, activation_sha)
        receipt["nested"] = {field: value}
        assert "RESTRICTED_PUBLIC_FIELD" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))


def test_receipt_requires_aggregate_only_k5_and_rounding(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation_sha = validator.canonical_digest(activation)
    receipt = _receipt(amendment, amendment_sha, activation_sha)
    receipt["public_export"]["raw_counts"] = True
    assert "RECEIPT_EXPORT_POLICY" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))
    receipt = _receipt(amendment, amendment_sha, activation_sha)
    receipt["public_export"]["minimum_cluster_k"] = 4
    assert "RECEIPT_K_POLICY" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))
    receipt = _receipt(amendment, amendment_sha, activation_sha)
    receipt["public_export"]["outward_rounding"]["rate_step"] = 0.05
    assert "RECEIPT_ROUNDING" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))


def test_no_causal_outcome_or_human_truth(amendment: dict, amendment_sha: str) -> None:
    activation = _activation(amendment, amendment_sha)
    activation["scientific_outcome_run"] = True
    assert "ACTIVATION_BOUNDARY" in _codes(validator.validate_activation_receipt(activation, amendment, amendment_sha))
    activation = _activation(amendment, amendment_sha)
    activation_sha = validator.canonical_digest(activation)
    receipt = _receipt(amendment, amendment_sha, activation_sha)
    receipt["scientific_endpoint_opened"] = True
    assert "RECEIPT_OUTCOME" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))
    receipt = _receipt(amendment, amendment_sha, activation_sha)
    receipt["pseudo_labels_are_ground_truth"] = True
    assert "RECEIPT_SEMANTICS" in _codes(validator.validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))


def test_validator_contains_no_restricted_discovery_path() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in ("rglob(", "os.walk", "restricted_manifest", "quarantine_root", "output/aea_", "data/aea"):
        assert token not in source
