#!/usr/bin/env python3
"""Validate the public Gemma replacement contract and aggregate receipts.

This validator is deliberately schema-only. It accepts explicit public JSON
artifacts, never discovers a quarantine, and never opens media, transcripts,
restricted manifests, or model-label payloads. A frozen amendment can validate
while remaining NOT_ACTIVE; activation is a separate, stricter receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


QWEN2_REMOVED = "qwen2_vl_2b_existing"
QWEN3_RETAINED = "qwen3_vl_8b_instruct_4bit"
GEMMA_ADDED = "gemma4_e4b_it_4bit_joint_audio_five_frame"
EXPECTED_BEFORE = frozenset({QWEN2_REMOVED, QWEN3_RETAINED})
EXPECTED_AFTER = frozenset({GEMMA_ADDED, QWEN3_RETAINED})
AMENDMENT_SCHEMA = "nursery-gemma4-replacement-instrument-amendment-v1"
ACTIVATION_SCHEMA = "nursery-gemma4-e4b-activation-receipt-v1"
CANARY_SCHEMA = "nursery-gemma4-e4b-joint-public-canary-v1"
RECEIPT_SCHEMA = "nursery-childlens-pseudo-calibration-receipt-v2"
INVARIANT_CONTRACT_ID = "childlens-pseudo-calibration-referential-invariants-v1"


@dataclass(frozen=True)
class Issue:
    code: str
    location: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "location": self.location, "message": self.message}


def _issue(issues: list[Issue], code: str, location: str, message: str) -> None:
    issues.append(Issue(code, location, message))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _valid_revision(value: Any) -> bool:
    return isinstance(value, str) and len(value) >= 7 and all(c in "0123456789abcdef" for c in value.lower())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(payload)


def _instrument_set(value: Any) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return frozenset()
    return frozenset(value)


def validate_amendment(
    amendment: Mapping[str, Any],
    baseline_protocol: Mapping[str, Any],
    baseline_sha256: str | None = None,
) -> list[Issue]:
    """Validate the immutable, intentionally inactive freeze artifact."""

    issues: list[Issue] = []
    if amendment.get("schema_version") != AMENDMENT_SCHEMA:
        _issue(issues, "AMENDMENT_SCHEMA", "amendment.schema_version", "unexpected amendment schema")
    if amendment.get("status") != "FROZEN_NOT_ACTIVE_PENDING_PROVENANCE" or amendment.get("activation_state") != "NOT_ACTIVE":
        _issue(issues, "AMENDMENT_STATE", "amendment", "frozen amendment must remain NOT_ACTIVE pending separate activation")
    if (
        amendment.get("amendment_mode") != "ADDITIVE_REPLACEMENT_CONTRACT_ONLY"
        or amendment.get("historical_protocols_modified") is not False
        or amendment.get("restricted_payload_accessed_to_draft") is not False
        or amendment.get("scientific_outcome_authorized") is not False
    ):
        _issue(issues, "AMENDMENT_SCOPE", "amendment", "amendment changed history, used restricted payload, or authorized an outcome")

    substitution = _mapping(amendment.get("substitution"))
    before_set = _instrument_set(substitution.get("before_active_referential_paths"))
    after_set = _instrument_set(substitution.get("after_active_referential_paths"))
    if before_set != EXPECTED_BEFORE or after_set != EXPECTED_AFTER:
        _issue(issues, "SUBSTITUTION_SET", "amendment.substitution", "exact Qwen2-to-Gemma substitution with retained Qwen3 is required")
    if (
        substitution.get("removed_path_id") != QWEN2_REMOVED
        or substitution.get("added_path_id") != GEMMA_ADDED
        or substitution.get("retained_path_id") != QWEN3_RETAINED
        or substitution.get("substitution_count") != 1
        or substitution.get("no_third_model") is not True
    ):
        _issue(issues, "SUBSTITUTION_CARDINALITY", "amendment.substitution", "exactly one instrument may be replaced; no third model is allowed")
    if len(after_set - before_set) != 1 or len(before_set - after_set) != 1 or len(after_set & before_set) != 1:
        _issue(issues, "SUBSTITUTION_DELTA", "amendment.substitution", "set delta must be one add, one remove, one retain")

    unchanged = _mapping(amendment.get("unchanged_contract"))
    for field in ("sample", "schema", "gates", "thresholds", "K5_suppression", "outward_rounding"):
        if unchanged.get(field) is not True:
            _issue(issues, "CONTRACT_CHANGED", f"amendment.unchanged_contract.{field}", "frozen contract changed")
    if baseline_sha256 is not None and unchanged.get("before_protocol_sha256") != baseline_sha256:
        _issue(issues, "BASELINE_DIGEST", "amendment.unchanged_contract.before_protocol_sha256", "baseline protocol byte digest mismatch")
    if unchanged.get("before_invariant_contract_id") != INVARIANT_CONTRACT_ID or unchanged.get("after_invariant_contract_id") != INVARIANT_CONTRACT_ID:
        _issue(issues, "INVARIANT_ID", "amendment.unchanged_contract", "invariant contract identity changed")
    sample = _mapping(baseline_protocol.get("sample"))
    gates = _mapping(baseline_protocol.get("predeclared_gates"))
    projection = _mapping(unchanged.get("invariant_projection"))
    expected_projection = {
        "primary_item_count": sample.get("primary_item_count"),
        "primary_total_seconds": sample.get("primary_total_seconds"),
        "candidate_window_count": 137,
        "frame_offsets_seconds": [-5.0, -2.5, 0.0, 2.5, 5.0],
        "minimum_export_cell_items": gates.get("minimum_export_cell_items"),
        "schema_validity_minimum": gates.get("schema_validity_minimum"),
        "primary_item_coverage_minimum": gates.get("primary_item_coverage_minimum"),
        "primary_seconds_coverage_minimum": gates.get("primary_seconds_coverage_minimum"),
        "maximum_abstention_for_a_dimension": gates.get("maximum_abstention_for_a_dimension"),
        "minimum_paths_per_task_family": gates.get("minimum_paths_per_task_family"),
        "agreement_is_not_a_pass_gate": gates.get("agreement_is_not_a_pass_gate"),
    }
    if dict(projection) != expected_projection:
        _issue(issues, "INVARIANT_PROJECTION", "amendment.unchanged_contract.invariant_projection", "sample/schema/gate projection changed")

    crosswalk = _mapping(amendment.get("sample_and_crosswalk"))
    if (
        crosswalk.get("primary_item_count") != 15
        or crosswalk.get("primary_total_seconds") != 900
        or crosswalk.get("candidate_window_count") != 137
        or crosswalk.get("reserve_activation") is not False
        or crosswalk.get("reselection") is not False
        or crosswalk.get("content_model_confidence_or_agreement_dependent_selection") is not False
    ):
        _issue(issues, "SAMPLE_CROSSWALK", "amendment.sample_and_crosswalk", "sample was changed or made output-dependent")

    qwen = _mapping(amendment.get("qwen3_reuse"))
    if (
        qwen.get("instrument_id") != QWEN3_RETAINED
        or not _valid_revision(qwen.get("revision"))
        or qwen.get("read_only") is not True
        or qwen.get("predictions_recomputed") is not False
        or qwen.get("restricted_output_digest_match_required") is not True
        or qwen.get("digest_match") != "PENDING_ACTIVATION_RECEIPT_NOT_ACTIVE"
    ):
        _issue(issues, "QWEN3_NOT_READ_ONLY", "amendment.qwen3_reuse", "Qwen3 must remain read-only and digest-gated")

    placeholders = _mapping(amendment.get("gemma4_provenance_placeholders"))
    if placeholders.get("instrument_id") != GEMMA_ADDED or placeholders.get("placeholder_values_are_activation_evidence") is not False:
        _issue(issues, "PLACEHOLDER_BOUNDARY", "amendment.gemma4_provenance_placeholders", "placeholder identity/evidence boundary changed")
    placeholder_fields = (
        "upstream_revision",
        "conversion_revision",
        "upstream_license_identifier",
        "conversion_license_identifier",
        "license_reconciliation_receipt_sha256",
        "upstream_model_card_sha256",
        "conversion_model_card_sha256",
        "artifact_manifest_sha256",
        "artifact_file_count",
        "artifact_bytes",
        "processor_config_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
        "generation_config_sha256",
        "runtime_lock_sha256",
        "adapter_sha256",
        "public_canary_receipt_sha256",
    )
    for field in placeholder_fields:
        value = placeholders.get(field)
        if not isinstance(value, str) or not value.startswith("PLACEHOLDER_") or not value.endswith("NOT_ACTIVE"):
            _issue(issues, "PLACEHOLDER_ACTIVATED_IN_AMENDMENT", f"amendment.gemma4_provenance_placeholders.{field}", "freeze artifact must retain explicit non-evidentiary placeholder")

    correction = _mapping(amendment.get("correction_policy"))
    if (
        correction.get("maximum_corrections") != 1
        or correction.get("transport_or_parser_only") is not True
        or correction.get("prompt_change") is not False
        or correction.get("model_or_revision_change") is not False
        or correction.get("input_or_window_change") is not False
        or correction.get("semantic_bin_or_definition_change") is not False
        or correction.get("threshold_or_gate_change") is not False
        or correction.get("semantic_tuning") is not False
        or correction.get("agreement_or_confidence_directed") is not False
    ):
        _issue(issues, "CORRECTION_POLICY", "amendment.correction_policy", "only one output-independent transport/parser correction is allowed")

    canary_binding = _mapping(amendment.get("public_canary_binding"))
    required_booleans = _mapping(canary_binding.get("required_booleans"))
    if canary_binding.get("required_schema_version") != CANARY_SCHEMA or canary_binding.get("current_status") != "NOT_RUN_NOT_ACTIVE":
        _issue(issues, "CANARY_FREEZE", "amendment.public_canary_binding", "public canary must remain required and not yet active")
    for field in ("audio_consumed", "five_frames_consumed_in_order", "exact_schema_valid", "network_denial_sentinel_passed", "no_external_request"):
        if required_booleans.get(field) is not True:
            _issue(issues, "CANARY_REQUIREMENT", f"amendment.public_canary_binding.required_booleans.{field}", "joint-canary requirement was weakened")
    if required_booleans.get("restricted_inference_authorized_by_canary") is not False:
        _issue(issues, "CANARY_REQUIREMENT", "amendment.public_canary_binding.required_booleans.restricted_inference_authorized_by_canary", "canary cannot itself authorize restricted inference")

    privacy = _mapping(amendment.get("privacy_and_execution"))
    for field in ("hosted_or_external_model", "external_API", "telemetry", "stdout_or_stderr_content", "raw_or_item_level_repository_output"):
        if privacy.get(field) is not False:
            _issue(issues, "PRIVACY_BOUNDARY", f"amendment.privacy_and_execution.{field}", "external or raw-output path was enabled")
    if privacy.get("owner_private_quarantine") is not True or privacy.get("aggregate_only_tool_output") is not True or privacy.get("no_learner_tokenizer_checkpoint_or_causal_arm") is not True:
        _issue(issues, "PRIVACY_BOUNDARY", "amendment.privacy_and_execution", "quarantine/aggregate/no-outcome boundary was weakened")

    terminal = _mapping(amendment.get("terminal_state"))
    if terminal.get("amendment") != "FROZEN_NOT_ACTIVE_PENDING_PROVENANCE":
        _issue(issues, "TERMINAL_STATE", "amendment.terminal_state.amendment", "unexpected frozen terminal state")
    for field in ("restricted_gemma_inference_run", "replacement_comparison_run", "calibration_receipt_reissued", "scientific_outcome_run"):
        if terminal.get(field) is not False:
            _issue(issues, "OUTCOME_BOUNDARY", f"amendment.terminal_state.{field}", "freeze artifact cannot report inference/comparison/outcome execution")
    return issues


def validate_activation_receipt(
    activation: Mapping[str, Any], amendment: Mapping[str, Any], amendment_sha256: str
) -> list[Issue]:
    """Validate the separate receipt that may activate restricted Gemma work."""

    issues: list[Issue] = []
    if activation.get("schema_version") != ACTIVATION_SCHEMA or activation.get("status") != "ACTIVE_ALL_GATES_PASS":
        _issue(issues, "ACTIVATION_SCHEMA_OR_STATUS", "activation", "activation receipt must explicitly pass all gates")
    if activation.get("amendment_sha256") != amendment_sha256:
        _issue(issues, "ACTIVATION_AMENDMENT_BINDING", "activation.amendment_sha256", "activation is not byte-bound to the frozen amendment")
    if activation.get("all_placeholders_replaced") is not True or activation.get("restricted_inference_before_activation") is not False:
        _issue(issues, "ACTIVATION_ORDER", "activation", "all exact values must be resolved before restricted inference")

    qwen = _mapping(activation.get("qwen3_reuse"))
    if (
        qwen.get("instrument_id") != QWEN3_RETAINED
        or qwen.get("read_only") is not True
        or qwen.get("predictions_recomputed") is not False
        or qwen.get("restricted_output_digest_match") is not True
        or not _valid_sha256(qwen.get("restricted_output_sha256"))
    ):
        _issue(issues, "ACTIVATION_QWEN3", "activation.qwen3_reuse", "Qwen3 reuse is not immutable and read-only")

    gemma = _mapping(activation.get("gemma_instrument"))
    if gemma.get("instrument_id") != GEMMA_ADDED or gemma.get("upstream_repository") != "google/gemma-4-E4B-it":
        _issue(issues, "GEMMA_IDENTITY", "activation.gemma_instrument", "exact Gemma E4B identity is required")
    if not _valid_revision(gemma.get("upstream_revision")) or not _valid_revision(gemma.get("conversion_revision")):
        _issue(issues, "GEMMA_REVISION", "activation.gemma_instrument", "upstream and conversion revisions must be exact and pinned")
    hash_fields = (
        "license_reconciliation_receipt_sha256",
        "upstream_model_card_sha256",
        "conversion_model_card_sha256",
        "artifact_manifest_sha256",
        "processor_config_sha256",
        "tokenizer_config_sha256",
        "chat_template_sha256",
        "generation_config_sha256",
        "runtime_lock_sha256",
        "adapter_sha256",
        "public_canary_receipt_sha256",
    )
    for field in hash_fields:
        if not _valid_sha256(gemma.get(field)):
            _issue(issues, "GEMMA_HASH_GATE", f"activation.gemma_instrument.{field}", "required immutable hash is missing")
    if type(gemma.get("artifact_file_count")) is not int or gemma.get("artifact_file_count", 0) <= 0 or type(gemma.get("artifact_bytes")) is not int or gemma.get("artifact_bytes", 0) <= 0:
        _issue(issues, "GEMMA_ARTIFACT_SIZE", "activation.gemma_instrument", "verified positive artifact file count and byte size are required")
    if (
        gemma.get("upstream_license_identifier") != "Apache-2.0"
        or gemma.get("conversion_permission_resolved") is not True
        or gemma.get("permission_inferred_from_public_access") is not False
        or gemma.get("agent_accepted_clickthrough") is not False
        or gemma.get("restricted_outputs_quarantine_only") is not True
    ):
        _issue(issues, "GEMMA_LICENSE_PROVENANCE", "activation.gemma_instrument", "license/provenance/quarantine gates are incomplete")

    contract = _mapping(activation.get("unchanged_contract"))
    frozen = _mapping(amendment.get("unchanged_contract"))
    if (
        contract.get("before_protocol_sha256") != frozen.get("before_protocol_sha256")
        or contract.get("invariant_contract_id") != INVARIANT_CONTRACT_ID
        or contract.get("primary_item_count") != 15
        or contract.get("primary_total_seconds") != 900
        or contract.get("candidate_window_count") != 137
        or contract.get("sample_reselected") is not False
        or contract.get("schema_or_gate_changed") is not False
    ):
        _issue(issues, "ACTIVATION_CONTRACT", "activation.unchanged_contract", "activation changed the frozen sample/schema/gates")

    correction = _mapping(activation.get("bounded_correction"))
    used = correction.get("corrections_used")
    if type(used) is not int or used not in (0, 1) or correction.get("scope") not in {"NONE", "TRANSPORT", "PARSER"}:
        _issue(issues, "ACTIVATION_CORRECTION", "activation.bounded_correction", "at most one transport/parser correction may be used")
    for field in ("semantic_prompt_changed", "common_schema_changed", "model_or_revision_changed", "input_or_window_changed", "threshold_or_gate_changed", "semantic_tuning_performed", "agreement_or_quality_triggered"):
        if correction.get(field) is not False:
            _issue(issues, "ACTIVATION_SEMANTIC_TUNING", f"activation.bounded_correction.{field}", "semantic/output-driven tuning is forbidden")

    for field in ("hosted_or_cloud_inference_used", "external_request_made", "scientific_outcome_authorized", "scientific_outcome_run"):
        if activation.get(field) is not False:
            _issue(issues, "ACTIVATION_BOUNDARY", f"activation.{field}", "external processing or scientific outcome is forbidden")
    if activation.get("network_denial_verified") is not True or activation.get("aggregate_only_public_output") is not True:
        _issue(issues, "ACTIVATION_BOUNDARY", "activation", "network denial and aggregate-only output must be verified")
    return issues


def validate_joint_canary(
    canary: Mapping[str, Any],
    activation: Mapping[str, Any],
    amendment_sha256: str,
    canary_sha256: str | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    if canary.get("schema_version") != CANARY_SCHEMA or canary.get("status") != "PASS":
        _issue(issues, "CANARY_SCHEMA_OR_STATUS", "canary", "public joint canary must pass exact schema")
    if canary.get("scope") != "SELF_GENERATED_PUBLIC_AUDIO_PLUS_FIVE_FRAMES_ONLY" or canary.get("childlens_or_quarantine_accessed") is not False:
        _issue(issues, "CANARY_SCOPE", "canary", "canary must use self-generated public input only")
    for field in ("joint_single_request", "audio_consumed", "five_frames_consumed_in_order", "exact_schema_valid", "audio_nonce_recovered", "visual_nonce_recovered", "network_denial_sentinel_passed", "no_external_request"):
        if canary.get(field) is not True:
            _issue(issues, "CANARY_JOINT_MODALITY", f"canary.{field}", "joint audio/five-frame strict-schema evidence is incomplete")
    if canary.get("amendment_sha256") != amendment_sha256:
        _issue(issues, "CANARY_AMENDMENT_BINDING", "canary.amendment_sha256", "canary is not bound to the frozen amendment")
    gemma = _mapping(activation.get("gemma_instrument"))
    if canary_sha256 is not None and gemma.get("public_canary_receipt_sha256") != canary_sha256:
        _issue(issues, "CANARY_RECEIPT_BINDING", "activation.gemma_instrument.public_canary_receipt_sha256", "activation is not byte-bound to this canary")
    for field in ("upstream_revision", "conversion_revision", "artifact_manifest_sha256", "runtime_lock_sha256", "adapter_sha256"):
        if canary.get(field) != gemma.get(field):
            _issue(issues, "CANARY_ARTIFACT_MISMATCH", f"canary.{field}", "canary did not run the activation-bound artifact")
    used = canary.get("transport_or_parser_corrections_used")
    if type(used) is not int or used not in (0, 1):
        _issue(issues, "CANARY_CORRECTION_COUNT", "canary.transport_or_parser_corrections_used", "canary may use at most one bounded correction")
    if used == 1:
        correction = _mapping(canary.get("correction"))
        if correction.get("scope") not in {"TRANSPORT", "PARSER"} or correction.get("semantic_prompt_changed") is not False or correction.get("common_schema_changed") is not False:
            _issue(issues, "CANARY_CORRECTION_SCOPE", "canary.correction", "correction exceeded transport/parser-only scope")
    for field in ("hosted_or_cloud_inference_used", "restricted_inference_authorized_by_canary", "pseudo_output_is_human_evidence"):
        if canary.get(field) is not False:
            _issue(issues, "CANARY_BOUNDARY", f"canary.{field}", "canary cannot use cloud, authorize restricted inference, or become human evidence")
    return issues


_SENSITIVE_KEYS = {
    "transcript_text",
    "translated_text",
    "source_filename",
    "source_path",
    "participant_id",
    "item_id",
    "exact_timestamp",
    "exact_interval",
    "frame_content",
    "audio_content",
    "item_rows",
    "item_level_prediction",
    "confidence_score",
    "raw_model_payload",
}


def validate_aggregate_receipt(
    receipt: Mapping[str, Any],
    amendment: Mapping[str, Any],
    amendment_sha256: str,
    activation_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        _issue(issues, "RECEIPT_SCHEMA", "receipt.schema_version", "replacement receipt must use v2 schema")
    if receipt.get("amendment_sha256") != amendment_sha256 or receipt.get("activation_receipt_sha256") != activation_sha256:
        _issue(issues, "RECEIPT_BINDING", "receipt", "receipt is not byte-bound to amendment and activation")
    instruments = receipt.get("referential_instrument_ids")
    if not isinstance(instruments, list) or _instrument_set(instruments) != EXPECTED_AFTER or len(instruments) != 2:
        _issue(issues, "RECEIPT_INSTRUMENT_SET", "receipt.referential_instrument_ids", "receipt must contain Gemma+Qwen3 only")
    if receipt.get("third_visual_model_used") is not False or receipt.get("qwen3_reused_read_only") is not True:
        _issue(issues, "RECEIPT_MODEL_BOUNDARY", "receipt", "third model forbidden; Qwen3 must remain read-only")
    frozen = _mapping(amendment.get("unchanged_contract"))
    contract = _mapping(receipt.get("unchanged_contract"))
    if (
        receipt.get("primary_item_count") != 15
        or receipt.get("primary_total_seconds") != 900
        or receipt.get("candidate_window_count") != 137
        or receipt.get("sample_reselected") is not False
        or contract.get("before_protocol_sha256") != frozen.get("before_protocol_sha256")
        or contract.get("invariant_contract_id") != INVARIANT_CONTRACT_ID
        or contract.get("schema_or_gate_changed") is not False
    ):
        _issue(issues, "RECEIPT_CONTRACT", "receipt", "frozen sample/schema/gates changed")

    policy = _mapping(receipt.get("public_export"))
    if policy.get("minimum_cluster_k") != 5 or policy.get("complementary_suppression") is not True:
        _issue(issues, "RECEIPT_K_POLICY", "receipt.public_export", "K=5 complementary suppression is mandatory")
    for field in ("raw_counts", "paths", "identifiers", "filenames", "exact_timestamps_or_intervals", "transcript_or_lexical_content", "frames_or_audio", "item_level_predictions", "confidence_or_raw_model_payload", "free_form_errors"):
        if policy.get(field) is not False:
            _issue(issues, "RECEIPT_EXPORT_POLICY", f"receipt.public_export.{field}", "restricted export flag must be false")
    rounding = _mapping(policy.get("outward_rounding"))
    if rounding.get("rate_step") != 0.1 or rounding.get("lag_step_event_units") != 0.5:
        _issue(issues, "RECEIPT_ROUNDING", "receipt.public_export.outward_rounding", "outward rounding grid changed")

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_location = f"{location}.{key}"
                if str(key).lower() in _SENSITIVE_KEYS and child not in (False, None, "", [], {}):
                    _issue(issues, "RESTRICTED_PUBLIC_FIELD", child_location, "restricted or item-level field is nonempty")
                walk(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            _issue(issues, "NONFINITE_PUBLIC_VALUE", location, "NaN/infinity are forbidden")

    walk(receipt, "receipt")
    if receipt.get("pseudo_labels_are_ground_truth") is not False or receipt.get("human_evidence_available") is not False or receipt.get("simulator_oracle_only_evaluation_truth") is not True:
        _issue(issues, "RECEIPT_SEMANTICS", "receipt", "pseudo outputs cannot be human/ground-truth evidence")
    for field in ("scientific_outcome_authorized", "scientific_outcome_run", "scientific_endpoint_opened"):
        if receipt.get(field) is not False:
            _issue(issues, "RECEIPT_OUTCOME", f"receipt.{field}", "calibration receipt cannot authorize/open/run a causal outcome")
    if receipt.get("semantic_tuning_performed") is not False or receipt.get("corrections_used") not in (0, 1):
        _issue(issues, "RECEIPT_TUNING", "receipt", "semantic tuning or multiple corrections are forbidden")
    return issues


def _load_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, _sha256_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--baseline-protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path)
    parser.add_argument("--canary", type=Path)
    parser.add_argument("--aggregate-receipt", type=Path)
    args = parser.parse_args(argv)

    issues: list[Issue] = []
    try:
        amendment, amendment_sha = _load_with_sha(args.amendment)
        baseline, baseline_sha = _load_with_sha(args.baseline_protocol)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(Issue("PUBLIC_ARTIFACT_LOAD", "input", str(exc)))
        amendment, baseline, amendment_sha, baseline_sha = {}, {}, "", ""
    activation: Mapping[str, Any] | None = None
    activation_sha = ""
    if not issues:
        issues.extend(validate_amendment(amendment, baseline, baseline_sha))
        if args.activation_receipt:
            try:
                activation, activation_sha = _load_with_sha(args.activation_receipt)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append(Issue("ACTIVATION_LOAD", str(args.activation_receipt), str(exc)))
            else:
                issues.extend(validate_activation_receipt(activation, amendment, amendment_sha))
        if args.canary:
            if activation is None:
                issues.append(Issue("CANARY_WITHOUT_ACTIVATION", "--canary", "canary validation requires an activation receipt"))
            else:
                try:
                    canary, canary_sha = _load_with_sha(args.canary)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(Issue("CANARY_LOAD", str(args.canary), str(exc)))
                else:
                    issues.extend(validate_joint_canary(canary, activation, amendment_sha, canary_sha))
        if args.aggregate_receipt:
            if activation is None:
                issues.append(Issue("RECEIPT_WITHOUT_ACTIVATION", "--aggregate-receipt", "aggregate validation requires an activation receipt"))
            else:
                try:
                    receipt, _ = _load_with_sha(args.aggregate_receipt)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    issues.append(Issue("RECEIPT_LOAD", str(args.aggregate_receipt), str(exc)))
                else:
                    issues.extend(validate_aggregate_receipt(receipt, amendment, amendment_sha, activation_sha))

    result = {
        "schema_version": "nursery-gemma-replacement-validation-v1",
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": [issue.as_dict() for issue in issues],
        "restricted_or_quarantine_access": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
