#!/usr/bin/env python3
"""Fail-closed public validator for the prototype-only opaque instrument lane.

Only explicit public JSON artifacts are accepted. The module contains no
restricted-data discovery, media decoding, inference, or outcome execution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AMENDMENT_SCHEMA = "nursery-gemma4-e4b-opaque-instrument-provenance-amendment-v1"
ACTIVATION_SCHEMA = "nursery-gemma4-e4b-prototype-full-activation-receipt-v1"
ACTIVATION_SCHEMA_ID = "urn:nursery:gemma4-e4b:prototype-full-activation-receipt:v1"
ACTIVATION_RECEIPT_SCHEMA_SHA256 = "0668d8d304a1be30e43e9df2c05469a338d647c262418a04c6d067d2e77f8b05"
RUNTIME_ERRATUM_SCHEMA = "nursery-gemma4-prototype-runtime-pin-erratum-v1.1"
RUNTIME_ERRATUM_SHA256 = "b84da17121dd3235b1b6799d91c077d542e30e384f5cff5e131ec6c9755cecc1"
RUNTIME_ERRATUM_PATH = "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json"
EFFECTIVE_ACTIVATION_SCHEMA = "nursery-gemma4-e4b-prototype-full-activation-receipt-v1-plus-runtime-pin-erratum-v1.1"
NO_FALLBACK_OVERRIDE_SCHEMA = "nursery-prototype-no-automatic-fallback-override-v1"
NO_FALLBACK_OVERRIDE_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
PROVENANCE_ACTIVATION_SCHEMA = "nursery-gemma4-e4b-opaque-instrument-provenance-activation-receipt-v1"
CANARY_CONTRACT_SCHEMA = "nursery-gemma4-prototype-common-schema-canary-contract-v1"
CANARY_SCHEMA = "nursery-gemma4-e4b-prototype-common-schema-public-canary-receipt-v1"
CALIBRATION_SCHEMA = "nursery-prototype-opaque-gemma-calibration-receipt-v1"
OUTCOME_SEAL_SCHEMA = "nursery-minimal-outcome-authorization-seal-v1"
CONSTRUCTION_GATE_SCHEMA = "nursery-construction-falsification-gate-receipt-v1"

OPAQUE_INSTRUMENT_ID = "gemma4_e4b_it_4bit_joint_audio_five_frame"
QWEN3_ID = "qwen3_vl_8b_instruct_4bit"
QWEN2_ID = "qwen2_vl_2b_existing"
CACHED_ARTIFACT_MANIFEST_SHA256 = "34310498dc5b809bf4baa79a5290c9e675022d12b725993317bccbffacf1d3ae"
CACHED_CONVERSION_REVISION = "475b9088d29754a3379866cf5aeb6b41acd313c2"
CACHED_ARTIFACT_BYTES = 5_179_241_512
PRIOR_STOP_SCHEMA = "nursery-gemma4-replacement-terminal-decision-v1"
PRIOR_STOP_DECISION = "CALIBRATION_STOP_MODEL_TRIANGULATION"
PRIOR_STOP_SHA256 = "a9f955d53768e0295d61a58e8ecf73d7e8658890ec56a4408772cd6bb10612b6"
PROVENANCE_ACTIVATION_SHA256 = "2f25cff5d8fc140fb7dddeebae4cccfa83880327b5d383d14b3f21f3e2adcf13"
RESOURCE_RECEIPT_SHA256 = "363fda5ebc189216b2d8e9a9133cb51cea91fcc0ea9dc15fef023a028648dcd4"
CANARY_CONTRACT_SHA256 = "9b894fcfd47df93824d25a467961d2c9f3398666dbe824367f2d8e1afe0635e9"
BASELINE_PROTOCOL_SHA256 = "db5336726f4841e592cb9c73df5ab22a0887d1bd34c7cf673146f38de9e86961"
SENSITIVITY_PROTOCOL_SHA256 = "c6b7afab6c8914a968b0f06706a222d2771863c2a3fd2017184c17d431bfa7d6"
HISTORICAL_CALIBRATION_SHA256 = "c8da2349663f7e49291381c42e15f8972bb3b13b0b739dad4e1221ee937dfc59"
BASELINE_COMMON_SCHEMA_SHA256 = "a02c97d81d8114792f6ff01c244a553e16d45f03ea9fbaf42648219c279885a2"
EXACT_OUTPUT_SCHEMA_SHA256 = "19b7ae8948aed46410e1307c0a302303813422995c8f4cdc19949118693a8089"
EXPECTED_REFERENTIAL_SET = frozenset({OPAQUE_INSTRUMENT_ID, QWEN3_ID})


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _exact_pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(row, str) for row in value) and frozenset(value) == EXPECTED_REFERENTIAL_SET


def validate_prior_stop(stop: Mapping[str, Any], stop_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if stop_sha256 != PRIOR_STOP_SHA256 or stop.get("schema_version") != PRIOR_STOP_SCHEMA or stop.get("decision") != PRIOR_STOP_DECISION:
        _issue(issues, "PRIOR_STOP_IDENTITY", "prior_stop", "exact historical STOP bytes and decision are required")
    for field in ("restricted_gemma_inference_run", "restricted_gemma_output_opened", "qwen3_instruments_rerun", "calibration_receipt_replaced", "scientific_outcome_run", "third_visual_model_tried"):
        if stop.get(field) is not False:
            _issue(issues, "PRIOR_STOP_MUTATED", f"prior_stop.{field}", "historical STOP execution state changed")
    gates = _mapping(stop.get("gate_results"))
    if gates.get("G8_minimal_outcome_authorization") != "FAIL_CLOSED":
        _issue(issues, "PRIOR_STOP_OUTCOME", "prior_stop.gate_results.G8_minimal_outcome_authorization", "historical outcome gate must remain fail-closed")
    return issues


def validate_no_fallback_override(
    override: Mapping[str, Any], override_sha256: str
) -> list[Issue]:
    issues: list[Issue] = []
    if (
        override_sha256 != NO_FALLBACK_OVERRIDE_SHA256
        or override.get("schema_version") != NO_FALLBACK_OVERRIDE_SCHEMA
        or override.get("status") != "FROZEN_BEFORE_REVISED_GEMMA_CANARY_OR_SCIENTIFIC_ENDPOINT"
    ):
        _issue(issues, "NO_FALLBACK_IDENTITY", "no_fallback_override", "override bytes, schema, or freeze state changed")
    expected = {
        "QWEN3_ONLY_CALIBRATION",
        "CONSERVATIVE_UNBOUNDED_OR_FULL_DOMAIN_SYNTHETIC_SENSITIVITY",
        "EMPIRICAL_FREE_AUTOMATIC_OUTCOME",
        "THIRD_VISUAL_MODEL",
        "ALTERED_CLAIM_WORDING_TO_ENABLE_AN_OUTCOME",
    }
    observed = override.get("automatic_fallbacks_forbidden")
    if not isinstance(observed, list) or len(observed) != 5 or set(observed) != expected:
        _issue(issues, "NO_FALLBACK_SET", "no_fallback_override.automatic_fallbacks_forbidden", "exact five automatic fallback prohibitions are required")
    failure = _mapping(override.get("failure_behavior"))
    if failure.get("restricted_calibration_fallback_allowed") is not False or failure.get("scientific_endpoint_allowed") is not False or "Stop" not in str(failure.get("required_terminal_behavior", "")):
        _issue(issues, "NO_FALLBACK_FAILURE_BEHAVIOR", "no_fallback_override.failure_behavior", "failed Gemma must stop at a user decision point")
    invariants = _mapping(override.get("scientific_invariants"))
    for field in ("prompt_tuning_after_output", "range_seed_threshold_learner_or_control_tuning", "model_agreement_is_human_truth", "AEA_empirical_ancestry", "BabyView_empirical_ancestry", "restricted_payload_export"):
        if invariants.get(field) is not False:
            _issue(issues, "NO_FALLBACK_SCIENTIFIC_BOUNDARY", f"no_fallback_override.scientific_invariants.{field}", "scientific/privacy fallback boundary changed")
    if invariants.get("simulator_oracle_is_only_evaluation_truth") is not True:
        _issue(issues, "NO_FALLBACK_SCIENTIFIC_BOUNDARY", "no_fallback_override.scientific_invariants", "simulator oracle must remain sole evaluation truth")
    state = _mapping(override.get("state_at_freeze"))
    for field in ("CAF_revised_canary_run", "restricted_Gemma_inference_run", "scientific_endpoint_opened"):
        if state.get(field) is not False:
            _issue(issues, "NO_FALLBACK_FREEZE_STATE", f"no_fallback_override.state_at_freeze.{field}", "override was not frozen before revised canary/outcome")
    return issues


def validate_amendment(
    amendment: Mapping[str, Any],
    prior_stop: Mapping[str, Any],
    prior_stop_sha256: str,
) -> list[Issue]:
    issues = validate_prior_stop(prior_stop, prior_stop_sha256)
    if amendment.get("schema_version") != AMENDMENT_SCHEMA:
        _issue(issues, "AMENDMENT_SCHEMA", "amendment.schema_version", "unexpected opaque amendment schema")
    if (
        amendment.get("status") != "FROZEN_PROVENANCE_GATE_PASS_OPAQUE_INTERNAL_ONLY"
        or amendment.get("amendment_mode") != "ADDITIVE_NEW_BRANCH_NO_HISTORICAL_OVERWRITE"
    ):
        _issue(issues, "AMENDMENT_STATE", "amendment", "amendment must be a frozen additive internal-only branch")
    for field in ("restricted_content_accessed_to_draft", "new_model_downloaded", "license_or_clickthrough_accepted"):
        if amendment.get(field) is not False:
            _issue(issues, "AMENDMENT_ORDER", f"amendment.{field}", "freeze used restricted content, downloaded a model, or accepted terms")

    preserved = _mapping(amendment.get("historical_decisions_preserved"))
    history = _mapping(preserved.get("gemma_replacement_stop"))
    original = _mapping(preserved.get("original_replacement_amendment"))
    protocol = _mapping(preserved.get("original_calibration_protocol"))
    if (
        history.get("decision") != PRIOR_STOP_DECISION
        or history.get("sha256") != PRIOR_STOP_SHA256
        or history.get("unchanged_and_correct_under_original_gate") is not True
        or original.get("sha256") != "8e6d7a43ea9639fa451a59386b8657495558fde77fb2147a20767388d47ed24a"
        or original.get("unchanged") is not True
        or protocol.get("sha256") != BASELINE_PROTOCOL_SHA256
        or protocol.get("unchanged") is not True
    ):
        _issue(issues, "STOP_IMMUTABILITY", "amendment.historical_decisions_preserved", "new prototype lane must preserve the prior STOP and frozen artifacts")

    decision = _mapping(amendment.get("decision"))
    if (
        decision.get("exact_cached_artifact_provenance_gate") != "PASS_WITH_OPAQUE_INTERNAL_ONLY_LIMITATIONS"
        or decision.get("complete_conversion_reconstructibility") != "NOT_ESTABLISHED_AND_NOT_CLAIMED"
        or decision.get("legal_clearance") != "NOT_CLAIMED_NOT_LEGAL_ADVICE"
        or decision.get("full_instrument_activation") != "PENDING_UNCHANGED_NON_PROVENANCE_GATES"
        or decision.get("restricted_inference_authorized_by_this_amendment_alone") is not False
        or decision.get("scientific_outcome_authorized") is not False
    ):
        _issue(issues, "OPAQUE_DECISION_SCOPE", "amendment.decision", "opaque provenance pass was broadened into legal/full/scientific authorization")

    instrument = _mapping(amendment.get("exact_cached_artifact"))
    if (
        instrument.get("repository_label") != "mlx-community/gemma-4-e4b-it-4bit"
        or instrument.get("conversion_revision") != CACHED_CONVERSION_REVISION
        or instrument.get("artifact_manifest_sha256") != CACHED_ARTIFACT_MANIFEST_SHA256
        or instrument.get("artifact_bytes") != CACHED_ARTIFACT_BYTES
        or instrument.get("file_count") != 10
        or instrument.get("cached_rehash_matches_historical_receipt") is not True
        or instrument.get("artifact_substitution_or_mutation_permitted") is not False
    ):
        _issue(issues, "OPAQUE_ARTIFACT_BINDING", "amendment.opaque_instrument", "exact preexisting cached artifact binding is required")
    files = instrument.get("files")
    if not isinstance(files, list) or len(files) != 10 or sum(row.get("bytes", -1) for row in files if isinstance(row, Mapping)) != CACHED_ARTIFACT_BYTES or any(not _valid_sha256(row.get("sha256")) for row in files if isinstance(row, Mapping)):
        _issue(issues, "OPAQUE_FILE_MANIFEST", "amendment.exact_cached_artifact.files", "complete ten-file cached manifest is malformed")

    limits = _mapping(amendment.get("known_provenance_limitations"))
    if (
        limits.get("exact_converter_version_or_commit") is not None
        or limits.get("complete_transform_recipe") is not None
        or limits.get("reconstructible_from_upstream") is not False
        or limits.get("cryptographic_proof_of_derivation") is not False
        or limits.get("permission_inferred_from_public_access") is not False
        or limits.get("limitations_must_remain_visible_in_every_activation_and_report") is not True
    ):
        _issue(issues, "OPAQUE_LIMITATIONS", "amendment.known_provenance_limitations", "known opaque-provenance limitations were hidden or overstated")

    allowed = _mapping(amendment.get("allowed_scope"))
    allowed_text = " ".join(str(value).lower() for value in allowed.values())
    prohibited_text = " ".join(str(value).lower() for value in amendment.get("prohibited_scope", []) if isinstance(value, str))
    for token in ("owner-controlled local host only", "opaque immutable bytes", "quarantine-only", "k=5", "simulator oracle only", "no general or publication-grade approval"):
        if token not in allowed_text:
            _issue(issues, "OPAQUE_USE_BOUNDARY", "amendment.allowed_scope", f"missing internal-only boundary: {token}")
    for token in ("redistribute", "another revision", "learner ancestry", "ground truth", "change the frozen sample", "scientific learner"):
        if token not in prohibited_text:
            _issue(issues, "OPAQUE_USE_BOUNDARY", "amendment.prohibited_scope", f"missing prohibition: {token}")

    gates = _mapping(amendment.get("unchanged_non_provenance_activation_gates"))
    expected = {
        "exact_common_schema_public_joint_audio_five_frame_canary": "PENDING",
        "qwen3_read_only_digest_match": True,
        "same_15_items_900_seconds_137_windows": True,
        "schema_validity_minimum": 0.95,
        "primary_item_coverage_minimum": 0.8,
        "primary_seconds_coverage_minimum": 0.8,
        "maximum_abstention_for_a_dimension": 0.5,
        "minimum_export_cell_items": 5,
        "minimum_paths_per_task_family": 2,
        "agreement_is_not_a_pass_gate": True,
        "no_restricted_inference_before_separate_full_activation_receipt": True,
    }
    for field, value in expected.items():
        if gates.get(field) != value:
            _issue(issues, "CONTRACT_CHANGED", f"amendment.unchanged_non_provenance_activation_gates.{field}", "unchanged activation gate changed")

    next_state = _mapping(amendment.get("next_state"))
    if next_state.get("provenance_branch") != "OPAQUE_INTERNAL_ONLY_PASS" or next_state.get("instrument") != "NOT_FULLY_ACTIVE":
        _issue(issues, "AMENDMENT_ORDER", "amendment.next_state", "provenance-only amendment cannot fully activate the instrument")
    return issues


def validate_canary_contract(contract: Mapping[str, Any], amendment_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if contract.get("schema_version") != CANARY_CONTRACT_SCHEMA or contract.get("status") != "FROZEN_FOR_PUBLIC_SYNTHETIC_CANARY_ONLY":
        _issue(issues, "CANARY_CONTRACT_STATE", "canary_contract", "common-schema canary contract must be finalized and frozen")
    if contract.get("restricted_content_permitted") is not False or contract.get("scientific_outcome_permitted") is not False:
        _issue(issues, "CANARY_CONTRACT_BOUNDARY", "canary_contract", "canary contract cannot permit restricted content or outcome work")
    bindings = _mapping(contract.get("immutable_bindings"))
    prototype = _mapping(bindings.get("prototype_provenance_amendment"))
    original = _mapping(bindings.get("original_calibration_protocol"))
    if prototype.get("sha256") != amendment_sha256 or original.get("sha256") != BASELINE_PROTOCOL_SHA256:
        _issue(issues, "CANARY_CONTRACT_BINDING", "canary_contract.immutable_bindings", "canary contract is not byte-bound to provenance amendment and baseline")
    instrument = _mapping(contract.get("prototype_instrument"))
    if instrument.get("instrument_id") != OPAQUE_INSTRUMENT_ID or instrument.get("learner_ancestor") is not False or instrument.get("evaluation_truth") is not False or instrument.get("hosted_or_external_inference") is not False:
        _issue(issues, "CANARY_CONTRACT_INSTRUMENT", "canary_contract.prototype_instrument", "opaque measurement instrument crossed learner/truth/external boundary")
    frozen = _mapping(contract.get("unchanged_calibration_contract"))
    expected = {"primary_item_count": 15, "primary_total_seconds": 900, "candidate_window_count": 137, "minimum_export_cell_items": 5, "K5_complement_protection": True, "outward_rounding_unchanged": True, "reserve_activation": False, "reselection": False}
    for field, value in expected.items():
        if frozen.get(field) != value:
            _issue(issues, "CANARY_CONTRACT_FROZEN_GATE", f"canary_contract.unchanged_calibration_contract.{field}", "sample/K5/rounding contract changed")
    if canonical_digest(contract.get("exact_common_schema")) != EXACT_OUTPUT_SCHEMA_SHA256:
        _issue(issues, "CANARY_CONTRACT_SCHEMA", "canary_contract.exact_common_schema", "exact five-key common schema digest changed")
    fixture = _mapping(contract.get("public_fixture"))
    audio = _mapping(fixture.get("audio"))
    tts = _mapping(audio.get("local_TTS"))
    frames = _mapping(fixture.get("frames"))
    if (
        fixture.get("scope") != "SELF_GENERATED_PUBLIC_SYNTHETIC_ONLY"
        or fixture.get("fixture_count") != 1
        or audio.get("sample_rate_hz") != 16000
        or audio.get("sample_count") != 160000
        or audio.get("language") != "de-DE"
        or audio.get("phrase_utf8") != "Nimm die rote Tasse."
        or tts.get("executable") != "/usr/bin/say"
        or tts.get("voice") != "Anna"
        or tts.get("voice_locale") != "de_DE"
        or tts.get("rate_words_per_minute") != 170
        or tts.get("executable_sha256_required_in_receipt") is not True
        or tts.get("OS_build_and_voice_inventory_digest_required_in_receipt") is not True
        or frames.get("count") != 5
        or frames.get("offset_order_seconds") != [-5.0, -2.5, 0.0, 2.5, 5.0]
    ):
        _issue(issues, "CANARY_CONTRACT_FIXTURE", "canary_contract.public_fixture", "fixed local German TTS plus ordered five-frame fixture changed")
    artifact = _mapping(_mapping(contract.get("cached_artifact_and_runtime_requirements")).get("artifact"))
    if artifact.get("revision") != CACHED_CONVERSION_REVISION or artifact.get("snapshot_manifest_sha256") != CACHED_ARTIFACT_MANIFEST_SHA256 or artifact.get("snapshot_bytes") != CACHED_ARTIFACT_BYTES or artifact.get("snapshot_file_count") != 10:
        _issue(issues, "CANARY_CONTRACT_ARTIFACT", "canary_contract.cached_artifact_and_runtime_requirements.artifact", "canary contract changed cached artifact")
    qwen = _mapping(contract.get("qwen3_read_only_binding"))
    if qwen.get("instrument_id") != QWEN3_ID or qwen.get("restricted_hypotheses_recomputed") is not False or qwen.get("quarantine_binding_digest_match_required") is not True or qwen.get("third_visual_model") is not False:
        _issue(issues, "QWEN3_READ_ONLY", "canary_contract.qwen3_read_only_binding", "Qwen3 must remain read-only and no third model is allowed")
    correction = _mapping(_mapping(contract.get("parser_contract")).get("outer_fence_correction"))
    if correction.get("maximum_uses") != 1 or correction.get("semantic_prompt_or_schema_change") is not False or correction.get("model_revision_or_generation_change") is not False:
        _issue(issues, "CANARY_CONTRACT_CORRECTION", "canary_contract.parser_contract.outer_fence_correction", "only one nonsemantic parser correction is allowed")
    resource = _mapping(contract.get("resource_and_privacy_gates"))
    for field in ("restricted_or_item_level_output", "hosted_or_cloud_inference", "external_API", "telemetry", "restricted_inference", "causal_or_learner_outcome"):
        if resource.get(field) is not False:
            _issue(issues, "CANARY_CONTRACT_BOUNDARY", f"canary_contract.resource_and_privacy_gates.{field}", "public canary contract crossed privacy/outcome boundary")
    return issues


def validate_provenance_activation(
    receipt: Mapping[str, Any], amendment_sha256: str
) -> list[Issue]:
    """Validate the live provenance-only receipt; it must not fully activate."""

    issues: list[Issue] = []
    if (
        receipt.get("schema_version") != PROVENANCE_ACTIVATION_SCHEMA
        or receipt.get("status") != "OPAQUE_PROVENANCE_BRANCH_ACTIVATED_FULL_INSTRUMENT_PENDING"
        or receipt.get("scope") != "EXACT_CACHED_ARTIFACT_INTERNAL_PROTOTYPE_ONLY"
    ):
        _issue(issues, "PROVENANCE_ACTIVATION_STATE", "provenance_activation", "unexpected provenance-only activation state")
    frozen = _mapping(receipt.get("frozen_amendment"))
    if frozen.get("sha256") != amendment_sha256:
        _issue(issues, "PROVENANCE_ACTIVATION_BINDING", "provenance_activation.frozen_amendment", "receipt is not byte-bound to opaque amendment")
    history = _mapping(_mapping(receipt.get("historical_stop_binding")).get("gemma_replacement_terminal_decision"))
    history_root = _mapping(receipt.get("historical_stop_binding"))
    if history.get("decision") != PRIOR_STOP_DECISION or history.get("sha256") != PRIOR_STOP_SHA256 or history.get("preserved") is not True or history_root.get("historical_decisions_overwritten") is not False:
        _issue(issues, "PROVENANCE_STOP_IMMUTABILITY", "provenance_activation.historical_stop_binding", "prior STOP was not preserved")
    artifact = _mapping(receipt.get("artifact_binding"))
    if (
        artifact.get("conversion_revision") != CACHED_CONVERSION_REVISION
        or artifact.get("artifact_manifest_sha256") != CACHED_ARTIFACT_MANIFEST_SHA256
        or artifact.get("artifact_bytes") != CACHED_ARTIFACT_BYTES
        or artifact.get("artifact_file_count") != 10
        or artifact.get("cached_rehash_exact_match") is not True
        or artifact.get("artifact_mutation_or_substitution_allowed") is not False
    ):
        _issue(issues, "PROVENANCE_ARTIFACT_BINDING", "provenance_activation.artifact_binding", "provenance receipt changed exact cached bytes")
    limits = _mapping(receipt.get("unresolved_limitations"))
    for field in ("conversion_license_or_notice_present", "converter_version_or_commit_known", "complete_transform_recipe_known", "cryptographic_derivation_proven", "byte_reconstructibility_claimed", "official_google_conversion_claimed"):
        if limits.get(field) is not False:
            _issue(issues, "PROVENANCE_LIMITATION", f"provenance_activation.unresolved_limitations.{field}", "receipt overstated opaque provenance")
    if limits.get("limitations_accepted_only_for_fixed_internal_prototype_instrument") is not True:
        _issue(issues, "PROVENANCE_LIMITATION", "provenance_activation.unresolved_limitations", "limitation scope was broadened")
    covenants = _mapping(receipt.get("covenants"))
    for field in ("local_internal_use_only", "network_denied_inference_required", "quarantine_only_outputs", "simulator_oracle_remains_evaluation_truth"):
        if covenants.get(field) is not True:
            _issue(issues, "PROVENANCE_COVENANT", f"provenance_activation.covenants.{field}", "internal/privacy/oracle covenant failed")
    for field in ("redistribution_upload_publication_or_sharing", "new_download_or_clickthrough", "telemetry_or_external_service", "pseudo_labels_are_ground_truth", "model_agreement_is_human_evidence", "learner_ancestry_allowed"):
        if covenants.get(field) is not False:
            _issue(issues, "PROVENANCE_COVENANT", f"provenance_activation.covenants.{field}", "forbidden distribution/download/truth/ancestry path enabled")
    gates = _mapping(receipt.get("gate_result"))
    if (
        gates.get("new_opaque_internal_only_provenance_gate") != "PASS"
        or gates.get("full_instrument_activation") != "NOT_ACTIVE"
        or gates.get("restricted_inference_authorized_by_this_receipt") is not False
        or gates.get("learner_or_causal_outcome_authorized") is not False
    ):
        _issue(issues, "PROVENANCE_GATE_SCOPE", "provenance_activation.gate_result", "provenance-only receipt improperly activated inference or outcome")
    privacy = _mapping(receipt.get("privacy_and_ancestry"))
    if privacy.get("restricted_content_accessed") is not False or privacy.get("hosted_or_cloud_inference_used") is not False or privacy.get("new_model_bytes_downloaded") != 0 or privacy.get("license_or_clickthrough_accepted") is not False or privacy.get("restricted_payload_exported") is not False:
        _issue(issues, "PROVENANCE_PRIVACY", "provenance_activation.privacy_and_ancestry", "provenance receipt crossed restricted/download/cloud boundary")
    return issues


def validate_canary(
    canary: Mapping[str, Any], amendment_sha256: str, canary_contract_sha256: str
) -> list[Issue]:
    issues: list[Issue] = []
    if canary.get("schema_version") != CANARY_SCHEMA or canary.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_PASS":
        _issue(issues, "CANARY_STATUS", "canary", "exact common-schema public canary must pass")
    if canary.get("amendment_sha256") != amendment_sha256:
        _issue(issues, "CANARY_AMENDMENT_BINDING", "canary.amendment_sha256", "canary is not bound to amendment bytes")
    if canary.get("canary_contract_sha256") != canary_contract_sha256:
        _issue(issues, "CANARY_CONTRACT_BINDING", "canary.canary_contract_sha256", "canary is not byte-bound to its frozen contract")
    if canary.get("runtime_pin_erratum_path") != RUNTIME_ERRATUM_PATH or canary.get("runtime_pin_erratum_sha256") != RUNTIME_ERRATUM_SHA256:
        _issue(issues, "CANARY_ERRATUM_BINDING", "canary", "canary is not bound to the authoritative runtime-pin erratum")
    for field in ("tts_executable_sha256", "os_build_and_voice_inventory_sha256", "fixture_manifest_sha256"):
        if not _valid_sha256(canary.get(field)):
            _issue(issues, "CANARY_FIXTURE_PROVENANCE", f"canary.{field}", "fixed public fixture provenance hash is missing")
    if (
        canary.get("instrument_id") != OPAQUE_INSTRUMENT_ID
        or canary.get("conversion_revision") != CACHED_CONVERSION_REVISION
        or canary.get("artifact_manifest_sha256") != CACHED_ARTIFACT_MANIFEST_SHA256
        or canary.get("restricted_common_schema_sha256") != BASELINE_COMMON_SCHEMA_SHA256
        or canary.get("exact_output_schema_sha256") != EXACT_OUTPUT_SCHEMA_SHA256
    ):
        _issue(issues, "CANARY_ARTIFACT_OR_SCHEMA", "canary", "canary artifact or exact common schema is stale")
    for field in ("self_generated_public_fixture_only", "joint_audio_plus_five_frames", "five_frames_consumed_in_order", "exact_schema_valid", "network_denial_sentinel_passed", "no_external_request"):
        if canary.get(field) is not True:
            _issue(issues, "CANARY_REQUIREMENT", f"canary.{field}", "joint public canary requirement failed")
    for field in ("restricted_payload_accessed", "download_performed", "clickthrough_accepted", "hosted_or_cloud_inference_used", "restricted_inference_authorized_by_canary"):
        if canary.get(field) is not False:
            _issue(issues, "CANARY_BOUNDARY", f"canary.{field}", "public canary crossed a privacy/license/activation boundary")
    used = canary.get("corrections_used")
    if type(used) is not int or used not in (0, 1) or canary.get("correction_scope") not in {"NONE", "TRANSPORT", "PARSER"}:
        _issue(issues, "CANARY_CORRECTION", "canary", "canary may use at most one transport/parser correction")
    if canary.get("semantic_tuning_performed") is not False:
        _issue(issues, "CANARY_SEMANTIC_TUNING", "canary.semantic_tuning_performed", "semantic tuning is forbidden")
    if canary.get("semantic_quality_gate_used") is not False:
        _issue(issues, "CANARY_SEMANTIC_TUNING", "canary.semantic_quality_gate_used", "public canary is structural/schema evidence, not semantic quality evidence")
    return issues


def validate_runtime_erratum(
    erratum: Mapping[str, Any],
    erratum_sha256: str,
    activation_schema_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if (
        erratum_sha256 != RUNTIME_ERRATUM_SHA256
        or erratum.get("schema_version") != RUNTIME_ERRATUM_SCHEMA
        or erratum.get("status") != "FROZEN_PRE_CANARY_CONTENT_INDEPENDENT_ERRATUM"
    ):
        _issue(issues, "RUNTIME_ERRATUM_IDENTITY", "runtime_erratum", "runtime erratum bytes, schema, or status changed")
    timing = _mapping(erratum.get("timing_and_independence"))
    for field in ("public_canary_run_before_erratum", "public_model_output_existed_before_erratum", "restricted_content_accessed", "restricted_inference_run", "model_semantics_agreement_or_quality_inspected", "scientific_outcome_run"):
        if timing.get(field) is not False:
            _issue(issues, "RUNTIME_ERRATUM_TIMING", f"runtime_erratum.timing_and_independence.{field}", "erratum was not content-independent and pre-canary")
    if timing.get("content_independent_runtime_correction") is not True:
        _issue(issues, "RUNTIME_ERRATUM_TIMING", "runtime_erratum.timing_and_independence", "runtime-only correction was not attested")
    originals = _mapping(erratum.get("immutable_original_bindings"))
    activation_binding = _mapping(originals.get("activation_receipt_schema_v1"))
    canary_binding = _mapping(originals.get("common_schema_canary_contract_v1"))
    if activation_binding.get("sha256") != activation_schema_sha256 or canary_binding.get("sha256") != CANARY_CONTRACT_SHA256:
        _issue(issues, "RUNTIME_ERRATUM_ORIGINAL_BINDING", "runtime_erratum.immutable_original_bindings", "erratum is not bound to exact immutable originals")
    expected_corrections = [
        ("activation_receipt_schema_v1", "/properties/runtime_binding/properties/python/const", "3.12.13", "3.10.20"),
        ("activation_receipt_schema_v1", "/properties/runtime_binding/properties/numpy/const", "2.5.1", "2.2.6"),
        ("common_schema_canary_contract_v1", "/cached_artifact_and_runtime_requirements/runtime_exact_pins/python_version", "3.12.13", "3.10.20"),
        ("common_schema_canary_contract_v1", "/cached_artifact_and_runtime_requirements/runtime_exact_pins/numpy", "2.5.1", "2.2.6"),
    ]
    observed: list[tuple[Any, Any, Any, Any]] = []
    corrections = erratum.get("exact_corrections")
    if isinstance(corrections, list):
        for row in corrections:
            mapped = _mapping(row)
            observed.append((mapped.get("document"), mapped.get("JSON_pointer"), mapped.get("frozen_value"), mapped.get("corrected_value")))
    if observed != expected_corrections:
        _issue(issues, "RUNTIME_ERRATUM_CORRECTIONS", "runtime_erratum.exact_corrections", "erratum must contain exactly the four frozen runtime substitutions")
    expected_pins = {"python": "3.10.20", "mlx_vlm": "0.6.6", "mlx": "0.32.0", "transformers": "5.14.1", "tokenizers": "0.22.2", "numpy": "2.2.6", "pillow": "12.3.0", "llguidance": "1.7.6", "ffmpeg": "8.0.1", "jsonschema": "NOT_USED_MANUAL_EXACT_VALIDATOR"}
    if dict(_mapping(erratum.get("effective_runtime_pins"))) != expected_pins:
        _issue(issues, "RUNTIME_ERRATUM_PINS", "runtime_erratum.effective_runtime_pins", "effective runtime pin set changed")
    unchanged = _mapping(erratum.get("unchanged_contract"))
    for field, value in unchanged.items():
        if field == "frame_offsets":
            if value != [-5.0, -2.5, 0.0, 2.5, 5.0]:
                _issue(issues, "RUNTIME_ERRATUM_SCOPE", f"runtime_erratum.unchanged_contract.{field}", "frame offsets changed")
        elif value is not True:
            _issue(issues, "RUNTIME_ERRATUM_SCOPE", f"runtime_erratum.unchanged_contract.{field}", "non-runtime contract changed")
    rule = _mapping(erratum.get("fail_closed_application_rule"))
    if rule.get("maximum_corrected_semantic_or_scientific_fields") != 0 or rule.get("maximum_runtime_fields_changed_per_document") != 2 or rule.get("reject_if_any_unlisted_field_changes") is not True or rule.get("does_not_spend_outer_fence_parser_correction") is not True:
        _issue(issues, "RUNTIME_ERRATUM_SCOPE", "runtime_erratum.fail_closed_application_rule", "erratum scope or parser-correction accounting changed")
    boundary = _mapping(erratum.get("activation_boundary"))
    if boundary.get("future_activation_receipt_schema_version") != EFFECTIVE_ACTIVATION_SCHEMA:
        _issue(issues, "RUNTIME_ERRATUM_SCHEMA_VERSION", "runtime_erratum.activation_boundary", "effective receipt schema version changed")
    for field in ("erratum_is_activation_receipt", "public_canary_authorized_by_erratum_alone", "restricted_inference_authorized", "scientific_outcome_authorized"):
        if boundary.get(field) is not False:
            _issue(issues, "RUNTIME_ERRATUM_BOUNDARY", f"runtime_erratum.activation_boundary.{field}", "erratum improperly authorized work")
    return issues


def effective_activation_schema(
    activation_schema: Mapping[str, Any], runtime_erratum: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the exact additive runtime overlay in memory, changing nothing else."""

    effective = copy.deepcopy(dict(activation_schema))
    properties = effective["properties"]
    properties["runtime_binding"]["properties"]["python"]["const"] = "3.10.20"
    properties["runtime_binding"]["properties"]["numpy"]["const"] = "2.2.6"
    properties["schema_version"]["const"] = EFFECTIVE_ACTIVATION_SCHEMA
    immutable = properties["immutable_bindings"]
    immutable["required"].append("runtime_pin_erratum")
    immutable["properties"]["runtime_pin_erratum"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "sha256", "status"],
        "properties": {
            "path": {"const": RUNTIME_ERRATUM_PATH},
            "sha256": {"const": RUNTIME_ERRATUM_SHA256},
            "status": {"const": "FROZEN_PRE_CANARY_CONTENT_INDEPENDENT_ERRATUM"},
        },
    }
    return effective


def _resolve_local_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        return {}
    value: Any = root
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            return {}
        value = value[token]
    return _mapping(value)


def _schema_violations(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    location: str = "activation",
) -> list[tuple[str, str]]:
    """Validate the exact keyword subset used by the frozen activation schema."""

    if "$ref" in schema:
        resolved = _resolve_local_ref(root, str(schema.get("$ref")))
        if not resolved:
            return [(location, "unresolvable local schema reference")]
        return _schema_violations(value, resolved, root, location)
    violations: list[tuple[str, str]] = []
    def json_equal(left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
            return type(left) is type(right) and left == right
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(json_equal(left[key], right[key]) for key in left)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right))
        if type(left) in (int, float) and type(right) in (int, float):
            return math.isfinite(float(left)) and math.isfinite(float(right)) and left == right
        return type(left) is type(right) and left == right

    if "const" in schema and not json_equal(value, schema.get("const")):
        violations.append((location, "value differs from frozen const"))
        return violations
    if "enum" in schema and not any(json_equal(value, choice) for choice in schema.get("enum", [])):
        violations.append((location, "value is outside frozen enum"))
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(str(expected_type), True)
    if not type_ok:
        violations.append((location, f"expected JSON type {expected_type}"))
        return violations
    if isinstance(value, str) and "pattern" in schema and re.fullmatch(str(schema["pattern"]), value) is None:
        violations.append((location, "string does not match frozen pattern"))
    if isinstance(value, Mapping):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    violations.append((f"{location}.{key}", "required property is missing"))
        properties = _mapping(schema.get("properties"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    violations.append((f"{location}.{key}", "additional property is forbidden"))
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                violations.extend(_schema_violations(value[key], child_schema, root, f"{location}.{key}"))
    return violations


def validate_activation_schema_overlay(
    activation_schema: Mapping[str, Any],
    activation_schema_sha256: str,
    runtime_erratum: Mapping[str, Any],
    runtime_erratum_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    state = _mapping(activation_schema.get("x_contract_state"))
    if (
        activation_schema_sha256 != ACTIVATION_RECEIPT_SCHEMA_SHA256
        or activation_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or activation_schema.get("$id") != ACTIVATION_SCHEMA_ID
        or state.get("schema_frozen") is not True
        or state.get("activation_receipt_created") is not False
        or state.get("restricted_inference_run") is not False
        or state.get("scientific_outcome_run") is not False
        or state.get("schema_itself_authorizes_restricted_inference") is not False
    ):
        _issue(issues, "ACTIVATION_SCHEMA_IDENTITY", "activation_schema", "activation schema bytes, identity, or fail-closed state differ from frozen contract")
        return issues
    erratum_issues = validate_runtime_erratum(
        runtime_erratum, runtime_erratum_sha256, activation_schema_sha256
    )
    issues.extend(erratum_issues)
    if erratum_issues:
        return issues
    effective_schema = effective_activation_schema(activation_schema, runtime_erratum)
    effective_properties = _mapping(effective_schema.get("properties"))
    runtime = _mapping(_mapping(effective_properties.get("runtime_binding")).get("properties"))
    immutable = _mapping(effective_properties.get("immutable_bindings"))
    if (
        _mapping(effective_properties.get("schema_version")).get("const") != EFFECTIVE_ACTIVATION_SCHEMA
        or _mapping(runtime.get("python")).get("const") != "3.10.20"
        or _mapping(runtime.get("numpy")).get("const") != "2.2.6"
        or "runtime_pin_erratum" not in immutable.get("required", [])
    ):
        _issue(issues, "ACTIVATION_OVERLAY_APPLICATION", "activation_schema", "effective schema did not apply the exact runtime overlay and binding")
    return issues


def validate_activation(
    activation: Mapping[str, Any],
    activation_schema: Mapping[str, Any],
    activation_schema_sha256: str,
    runtime_erratum: Mapping[str, Any],
    runtime_erratum_sha256: str,
    canary_sha256: str,
) -> list[Issue]:
    """Validate only the exact frozen nested schema plus authoritative overlay."""

    issues = validate_activation_schema_overlay(
        activation_schema,
        activation_schema_sha256,
        runtime_erratum,
        runtime_erratum_sha256,
    )
    if issues:
        return issues
    effective_schema = effective_activation_schema(activation_schema, runtime_erratum)
    for location, message in _schema_violations(activation, effective_schema, effective_schema):
        _issue(issues, "ACTIVATION_SCHEMA_VALIDATION", location, message)
    canary = _mapping(activation.get("public_canary_binding"))
    if canary.get("receipt_sha256") != canary_sha256:
        _issue(issues, "ACTIVATION_CANARY_BINDING", "activation.public_canary_binding.receipt_sha256", "activation does not bind the exact validated public canary bytes")
    return issues


_SENSITIVE = {
    "transcript_text", "translated_text", "source_filename", "source_path", "participant_id", "item_id",
    "exact_timestamp", "exact_interval", "frame_content", "audio_content", "item_rows", "item_level_prediction",
    "confidence_score", "raw_model_payload",
}


def validate_calibration(
    receipt: Mapping[str, Any],
    activation_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if receipt.get("schema_version") != CALIBRATION_SCHEMA or receipt.get("decision") != "CALIBRATION_PASS_INTERNAL_PROTOTYPE":
        _issue(issues, "CALIBRATION_STATUS", "calibration", "future calibration receipt must explicitly pass frozen gates")
    if receipt.get("activation_receipt_sha256") != activation_sha256:
        _issue(issues, "CALIBRATION_BINDING", "calibration.activation_receipt_sha256", "calibration is not bound to activation bytes")
    if receipt.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_OVERRIDE_SHA256:
        _issue(issues, "CALIBRATION_FALLBACK_BINDING", "calibration.no_automatic_fallback_override_sha256", "calibration is not bound to no-automatic-fallback override")
    if not _exact_pair(receipt.get("referential_instrument_ids")) or receipt.get("qwen3_reused_read_only") is not True:
        _issue(issues, "CALIBRATION_INSTRUMENTS", "calibration", "exact opaque+Qwen3 pair with read-only Qwen3 is required")
    if receipt.get("primary_item_count") != 15 or receipt.get("primary_total_seconds") != 900 or receipt.get("candidate_window_count") != 137 or receipt.get("sample_reselected") is not False:
        _issue(issues, "CALIBRATION_SAMPLE", "calibration", "15/900/137 frozen sample changed")
    export = _mapping(receipt.get("public_export"))
    if export.get("minimum_cluster_k") != 5 or export.get("complementary_suppression") is not True or export.get("rate_rounding_step") != 0.1 or export.get("lag_rounding_step_event_units") != 0.5:
        _issue(issues, "CALIBRATION_EXPORT", "calibration.public_export", "K5/complementary suppression/outward rounding changed")
    for field in ("raw_counts", "item_level_results", "transcript_or_lexical_content", "frames_or_audio", "identifiers_or_paths", "exact_timestamps", "confidence_or_raw_payload"):
        if export.get(field) is not False:
            _issue(issues, "CALIBRATION_EXPORT", f"calibration.public_export.{field}", "restricted payload export is forbidden")

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_location = f"{location}.{key}"
                if str(key).lower() in _SENSITIVE and child not in (False, None, "", [], {}):
                    _issue(issues, "RESTRICTED_PUBLIC_FIELD", child_location, "restricted field is nonempty")
                walk(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            _issue(issues, "NONFINITE_PUBLIC_VALUE", location, "nonfinite public value is forbidden")

    walk(receipt, "calibration")
    for field in ("all_frozen_calibration_gates_passed", "aggregate_only", "simulator_oracle_only_evaluation_truth"):
        if receipt.get(field) is not True:
            _issue(issues, "CALIBRATION_GATE", f"calibration.{field}", "required calibration boundary/gate failed")
    for field in ("pseudo_labels_are_ground_truth", "human_validation_claimed", "scientific_outcome_run", "scientific_endpoint_opened", "semantic_tuning_performed"):
        if receipt.get(field) is not False:
            _issue(issues, "CALIBRATION_BOUNDARY", f"calibration.{field}", "calibration cannot claim truth, tune, or run/open outcome")
    return issues


def validate_outcome_seal(
    seal: Mapping[str, Any],
    calibration: Mapping[str, Any],
    calibration_sha256: str,
    construction: Mapping[str, Any],
    construction_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if seal.get("schema_version") != OUTCOME_SEAL_SCHEMA or seal.get("status") != "OUTCOME_AUTHORIZED_GATES_PASS":
        _issue(issues, "OUTCOME_SEAL_STATUS", "outcome_seal", "outcome authorization seal has not passed")
    if seal.get("calibration_receipt_sha256") != calibration_sha256 or seal.get("construction_falsification_receipt_sha256") != construction_sha256:
        _issue(issues, "OUTCOME_SEAL_BINDING", "outcome_seal", "authorization is not byte-bound to both prerequisite receipts")
    if seal.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_OVERRIDE_SHA256:
        _issue(issues, "OUTCOME_FALLBACK_BINDING", "outcome_seal", "outcome seal is not bound to no-automatic-fallback override")
    calibration_gate_issues = validate_calibration(
        calibration, str(calibration.get("activation_receipt_sha256", ""))
    )
    if calibration_gate_issues or calibration.get("decision") != "CALIBRATION_PASS_INTERNAL_PROTOTYPE" or calibration.get("all_frozen_calibration_gates_passed") is not True:
        _issue(issues, "OUTCOME_WITHOUT_CALIBRATION", "calibration", "passed calibration is required before outcome authorization")
    if construction.get("schema_version") != CONSTRUCTION_GATE_SCHEMA or construction.get("decision") != "CONSTRUCTION_AND_FALSIFICATION_PASS" or construction.get("construction_gate_passed") is not True or construction.get("falsification_gate_passed") is not True:
        _issue(issues, "OUTCOME_WITHOUT_CONSTRUCTION", "construction", "construction and falsification gates must both pass")
    if seal.get("calibration_gate_passed") is not True or seal.get("construction_gate_passed") is not True or seal.get("falsification_gate_passed") is not True:
        _issue(issues, "OUTCOME_SEAL_GATE", "outcome_seal", "seal must explicitly attest all prerequisite gates")
    if seal.get("authorization_used_restricted_item_payload") is not False or seal.get("scientific_outcome_run") is not False:
        _issue(issues, "OUTCOME_SEAL_BOUNDARY", "outcome_seal", "authorization may use aggregates only and may not itself run the outcome")
    return issues


def _load(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, _sha256(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--prior-stop", type=Path, required=True)
    parser.add_argument("--no-fallback-override", type=Path)
    parser.add_argument("--provenance-activation", type=Path)
    parser.add_argument("--canary-contract", type=Path)
    parser.add_argument("--canary", type=Path)
    parser.add_argument("--activation-schema", type=Path)
    parser.add_argument("--runtime-erratum", type=Path)
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--construction-gate", type=Path)
    parser.add_argument("--outcome-seal", type=Path)
    args = parser.parse_args(argv)

    issues: list[Issue] = []
    try:
        amendment, amendment_sha = _load(args.amendment)
        prior_stop, prior_stop_sha = _load(args.prior_stop)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(Issue("PUBLIC_ARTIFACT_LOAD", "input", str(exc)))
        amendment, prior_stop, amendment_sha, prior_stop_sha = {}, {}, "", ""
    canary_contract = canary = activation_schema = runtime_erratum = activation = calibration = construction = None
    canary_contract_sha = canary_sha = activation_schema_sha = runtime_erratum_sha = activation_sha = calibration_sha = construction_sha = ""
    if not issues:
        issues.extend(validate_amendment(amendment, prior_stop, prior_stop_sha))
        if args.no_fallback_override:
            no_fallback, no_fallback_sha = _load(args.no_fallback_override)
            issues.extend(validate_no_fallback_override(no_fallback, no_fallback_sha))
        if args.provenance_activation:
            provenance_activation, _ = _load(args.provenance_activation)
            issues.extend(validate_provenance_activation(provenance_activation, amendment_sha))
        if args.canary_contract:
            canary_contract, canary_contract_sha = _load(args.canary_contract)
            issues.extend(validate_canary_contract(canary_contract, amendment_sha))
        if args.canary:
            if canary_contract is None:
                issues.append(Issue("CANARY_WITHOUT_CONTRACT", "--canary", "canary validation requires its frozen contract"))
            canary, canary_sha = _load(args.canary)
            issues.extend(validate_canary(canary, amendment_sha, canary_contract_sha))
        if args.activation_schema or args.runtime_erratum:
            if not args.activation_schema or not args.runtime_erratum:
                issues.append(Issue("INCOMPLETE_ACTIVATION_OVERLAY", "input", "activation schema and runtime erratum must be supplied together"))
            else:
                activation_schema, activation_schema_sha = _load(args.activation_schema)
                runtime_erratum, runtime_erratum_sha = _load(args.runtime_erratum)
                if not args.activation:
                    issues.extend(
                        validate_activation_schema_overlay(
                            activation_schema,
                            activation_schema_sha,
                            runtime_erratum,
                            runtime_erratum_sha,
                        )
                    )
        if args.activation:
            if canary is None or not args.activation_schema or not args.runtime_erratum:
                issues.append(Issue("ACTIVATION_WITHOUT_PREREQUISITES", "--activation", "activation requires a validated public canary, exact frozen activation schema, and authoritative runtime erratum"))
            else:
                activation, activation_sha = _load(args.activation)
                issues.extend(validate_activation(activation, activation_schema, activation_schema_sha, runtime_erratum, runtime_erratum_sha, canary_sha))
        if args.calibration:
            if activation is None:
                issues.append(Issue("CALIBRATION_WITHOUT_ACTIVATION", "--calibration", "calibration requires a validated activation"))
            else:
                calibration, calibration_sha = _load(args.calibration)
                issues.extend(validate_calibration(calibration, activation_sha))
        if args.construction_gate:
            construction, construction_sha = _load(args.construction_gate)
        if args.outcome_seal:
            if calibration is None or construction is None:
                issues.append(Issue("OUTCOME_WITHOUT_PREREQUISITES", "--outcome-seal", "outcome seal requires calibration and construction/falsification receipts"))
            else:
                seal, _ = _load(args.outcome_seal)
                issues.extend(validate_outcome_seal(seal, calibration, calibration_sha, construction, construction_sha))

    result = {
        "schema_version": "nursery-prototype-opaque-instrument-validation-v1",
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": [issue.as_dict() for issue in issues],
        "restricted_access": False,
        "scientific_outcome_run": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
