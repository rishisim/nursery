#!/usr/bin/env python3
"""Validate the additive Gemma rendered-template placeholder-order amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AMENDMENT_SCHEMA = "nursery-gemma4-public-canary-rendered-template-placeholder-order-amendment-v1.3"
AMENDMENT_SHA256 = "2d117cf0f1619d43d6e71a90aa853241f0a5a7c331e84fc04af8231c5d6081fb"
CAF_AMENDMENT_SHA256 = "3ecf4bf6c62920d73177d029c18918f38366d34f7d7997642037696def37bbd0"
NO_FALLBACK_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
CAF_RUNNER_SHA256 = "4506897e0a115f0d119990a9434eb37a3093b1a0f08947bff8553372466a81e2"
CAF_FAILURE_SHA256 = "f84bf0c82136f4479258e6b4d9212c104487dc101deae67afdcc5a92d2a50ac4"
LEGACY_WORKER_SHA256 = "6f6ad2f2605c94aec8f92cd927248367eef74e16222768cced3a4cc25c050a2f"
BASE_CONTRACT_SHA256 = "9b894fcfd47df93824d25a467961d2c9f3398666dbe824367f2d8e1afe0635e9"
RUNTIME_ERRATUM_SHA256 = "b84da17121dd3235b1b6799d91c077d542e30e384f5cff5e131ec6c9755cecc1"
ACTIVATION_SCHEMA_SHA256 = "0668d8d304a1be30e43e9df2c05469a338d647c262418a04c6d067d2e77f8b05"
NEW_NAMESPACE = "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3"
AUDIO = "<|audio|>"
IMAGE = "<|image|>"
OLD_SEQUENCE = ["SYSTEM_PROMPT", IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, "USER_PROMPT", AUDIO]
NEW_SEQUENCE = ["SYSTEM_PROMPT", AUDIO, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, "USER_PROMPT"]
DECLARATIVE_ORDER = ["audio", "image_1", "image_2", "image_3", "image_4", "image_5", "text"]


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


def validate_bound_inputs(
    caf_amendment_sha256: str,
    no_fallback_sha256: str,
    caf_runner_sha256: str,
    caf_failure: Mapping[str, Any],
    caf_failure_sha256: str,
    legacy_worker_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if caf_amendment_sha256 != CAF_AMENDMENT_SHA256 or no_fallback_sha256 != NO_FALLBACK_SHA256 or caf_runner_sha256 != CAF_RUNNER_SHA256 or legacy_worker_sha256 != LEGACY_WORKER_SHA256:
        _issue(issues, "TEMPLATE_BOUND_INPUT", "bound_inputs", "CAF amendment, override, runner, or legacy worker bytes changed")
    if (
        caf_failure_sha256 != CAF_FAILURE_SHA256
        or caf_failure.get("schema_version") != "nursery-gemma4-public-canary-caf-failure-v1"
        or caf_failure.get("status") != "REVISE"
        or caf_failure.get("caf_transport_amendment_sha256") != CAF_AMENDMENT_SHA256
        or caf_failure.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_SHA256
        or caf_failure.get("legacy_runner_sha256") != "d5f3402c21043a0bba480e9540db7bac2d2eb01f65887673538db5211d19e8ea"
        or caf_failure.get("prior_aiff_failure_receipt_sha256") != "ef5031b3b90a3611eb488ff9fa76a48e0b2f2a0af0ab5e2230316d3234a0eda6"
        or caf_failure.get("parser_correction_eligible") is not False
        or caf_failure.get("restricted_payload_accessed") is not False
        or caf_failure.get("restricted_inference_run") is not False
        or caf_failure.get("scientific_endpoint_allowed") is not False
        or caf_failure.get("automatic_calibration_fallback_allowed") is not False
        or caf_failure.get("historical_output_paths_overwritten") is not False
    ):
        _issue(issues, "CAF_FAILURE_IMMUTABILITY", "caf_failure", "exact fail-closed CAF failure receipt is required")
    return issues


def validate_amendment(amendment: Mapping[str, Any], amendment_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if amendment_sha256 != AMENDMENT_SHA256 or amendment.get("schema_version") != AMENDMENT_SCHEMA or amendment.get("status") != "FROZEN_PRE_CANARY_TEMPLATE_PLACEHOLDER_ORDER_ONLY":
        _issue(issues, "TEMPLATE_AMENDMENT_IDENTITY", "amendment", "template-order amendment bytes/schema/status changed")
    if amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_POST_RENDER_PLACEHOLDER_ORDER_OVERLAY" or amendment.get("historical_artifacts_modified") is not False:
        _issue(issues, "TEMPLATE_ADDITIVE_BOUNDARY", "amendment", "amendment must be additive and preserve history")
    for field in ("public_canary_run", "restricted_content_accessed", "restricted_inference_run", "scientific_outcome_run"):
        if amendment.get(field) is not False:
            _issue(issues, "TEMPLATE_FREEZE_ORDER", f"amendment.{field}", "amendment was not frozen before canary/restricted/outcome work")

    bindings = _mapping(amendment.get("immutable_bindings"))
    expected = {
        "CAF_transport_amendment": CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override": NO_FALLBACK_SHA256,
        "CAF_public_canary_failure_receipt": CAF_FAILURE_SHA256,
        "CAF_public_canary_runner": CAF_RUNNER_SHA256,
        "bound_worker_at_failure": LEGACY_WORKER_SHA256,
        "prior_common_schema_canary_contract": BASE_CONTRACT_SHA256,
        "runtime_pin_erratum": RUNTIME_ERRATUM_SHA256,
        "activation_receipt_schema": ACTIVATION_SCHEMA_SHA256,
    }
    for field, digest in expected.items():
        if _mapping(bindings.get(field)).get("sha256") != digest:
            _issue(issues, "TEMPLATE_IMMUTABLE_BINDING", f"amendment.immutable_bindings.{field}", "bound predecessor digest changed")

    trigger = _mapping(amendment.get("trigger"))
    if (
        trigger.get("classification") != "PUBLIC_FIXTURE_TEMPLATE_PLACEHOLDER_ORDER_INCOMPATIBILITY"
        or trigger.get("stage") != "AFTER_CAF_FIXTURE_RUNTIME_AND_MODEL_LOAD_BEFORE_GENERATION"
        or trigger.get("failure_symbol") != "E_CONTENT_ORDER"
        or trigger.get("CAF_fixture_passed") is not True
        or trigger.get("model_generation_called") is not False
        or trigger.get("raw_model_output_existed") is not False
        or trigger.get("semantic_or_scientific_evidence_involved") is not False
        or trigger.get("outer_fence_parser_corrections_spent") != 0
        or trigger.get("public_token_diagnostic_anchor_sequence") != OLD_SEQUENCE
    ):
        _issue(issues, "TEMPLATE_TRIGGER", "amendment.trigger", "trigger was not the exact pre-generation public placeholder-order failure")

    delta = _mapping(amendment.get("exact_post_render_delta"))
    if (
        delta.get("maximum_relocated_substrings") != 1
        or delta.get("relocated_substring_utf8") != AUDIO
        or delta.get("relocation_stage") != "AFTER_EXACT_PROCESSOR_CHAT_TEMPLATE_RENDERING_AND_BEFORE_PROCESSOR_TOKENIZATION_OR_GENERATION"
        or delta.get("frozen_observed_anchor_sequence") != OLD_SEQUENCE
        or delta.get("amended_anchor_sequence") != NEW_SEQUENCE
        or delta.get("declarative_message_content_order_remains") != DECLARATIVE_ORDER
    ):
        _issue(issues, "TEMPLATE_EXACT_RELOCATION", "amendment.exact_post_render_delta", "exact single audio-placeholder relocation changed")
    for field in ("declarative_messages_changed", "system_prompt_text_changed", "user_prompt_text_changed", "template_arguments_changed", "placeholder_text_changed", "placeholder_counts_changed", "non_placeholder_rendered_bytes_changed", "media_argument_order_changed", "semantic_fields_changed", "model_runtime_or_artifact_fields_changed", "scientific_fields_changed", "parser_allowance_changed"):
        if delta.get(field) is not False:
            _issue(issues, "TEMPLATE_SCOPE_EXPANSION", f"amendment.exact_post_render_delta.{field}", "non-order field changed")
    if delta.get("does_not_spend_outer_fence_parser_correction") is not True:
        _issue(issues, "TEMPLATE_PARSER_BUDGET", "amendment.exact_post_render_delta", "template relocation spent parser correction")

    canary = _mapping(amendment.get("preserved_public_canary_contract"))
    for field in ("CAF_transport_overlay_unchanged", "public_fixture_phrase_voice_rate_unchanged", "final_WAV_contract_unchanged", "five_frame_generator_and_offsets_unchanged", "exact_system_and_user_prompt_text_unchanged", "exact_prompt_sha256_unchanged", "exact_common_schema_sha256_unchanged", "exact_parser_contract_sha256_unchanged", "schema_and_semantic_label_space_unchanged"):
        if canary.get(field) is not True:
            _issue(issues, "TEMPLATE_CANARY_CONTRACT", f"amendment.preserved_public_canary_contract.{field}", "public canary contract changed")
    if canary.get("outer_fence_parser_correction_maximum") != 1 or canary.get("outer_fence_parser_correction_used_before_amendment") != 0 or canary.get("parser_mode") != "PRIMARY":
        _issue(issues, "TEMPLATE_PARSER_BUDGET", "amendment.preserved_public_canary_contract", "parser allowance changed or was spent")

    scientific = _mapping(amendment.get("preserved_artifact_sample_and_scientific_contract"))
    for field in ("cached_Gemma_repository_revision_manifest_and_weight_hashes_unchanged", "Qwen3_read_only_digest_contract_unchanged", "sample_15_items_900_seconds_137_candidate_windows_unchanged", "K5_complement_protection_and_outward_rounding_unchanged", "coverage_abstention_path_and_schema_gates_unchanged", "agreement_is_not_a_pass_gate", "simulator_oracle_remains_only_evaluation_truth"):
        if scientific.get(field) is not True:
            _issue(issues, "TEMPLATE_SCIENTIFIC_CONTRACT", f"amendment.preserved_artifact_sample_and_scientific_contract.{field}", "artifact/sample/scientific gate changed")
    for field in ("causal_thresholds_changed", "calibration_range_or_scientific_protocol_changed", "automatic_fallback_allowed"):
        if scientific.get(field) is not False:
            _issue(issues, "TEMPLATE_SCIENTIFIC_CONTRACT", f"amendment.preserved_artifact_sample_and_scientific_contract.{field}", "threshold/protocol/fallback change enabled")

    rule = _mapping(amendment.get("overlay_application_rule"))
    for field in ("reject_if_any_bound_digest_differs", "reject_if_exact_render_preconditions_differ", "reject_if_more_or_less_than_one_audio_placeholder_is_relocated", "reject_if_any_non_audio_placeholder_byte_changes", "reject_if_any_prompt_message_schema_parser_artifact_sample_threshold_or_scientific_field_changes"):
        if rule.get(field) is not True:
            _issue(issues, "TEMPLATE_APPLICATION_RULE", f"amendment.overlay_application_rule.{field}", "fail-closed application rule weakened")
    if rule.get("reuse_prior_CAF_failure_or_success_receipt_as_template_order_receipt") is not False:
        _issue(issues, "TEMPLATE_APPLICATION_RULE", "amendment.overlay_application_rule", "prior CAF receipt reuse was enabled")

    namespace = _mapping(amendment.get("new_immutable_output_namespace"))
    if (
        namespace.get("path") != NEW_NAMESPACE
        or namespace.get("must_not_exist_before_execution") is not True
        or namespace.get("historical_output_paths_overwritten") is not False
        or namespace.get("future_public_canary_receipt") != "public_common_schema_canary_receipt_v1_3.json"
        or namespace.get("future_public_canary_failure") != "public_common_schema_canary_failure_v1_3.json"
        or namespace.get("future_runtime_lock_receipt") != "runtime_lock_receipt_v1_3.json"
        or namespace.get("future_activation_receipt") != "full_activation_receipt_v1_3.json"
    ):
        _issue(issues, "TEMPLATE_OUTPUT_NAMESPACE", "amendment.new_immutable_output_namespace", "new namespace or no-overwrite rule changed")
    auth = _mapping(amendment.get("authorization_boundary"))
    for field in ("amendment_is_canary_execution", "amendment_is_activation_receipt", "public_canary_authorized_by_amendment_alone", "restricted_inference_authorized", "scientific_outcome_authorized", "model_agreement_or_pseudo_labels_are_ground_truth"):
        if auth.get(field) is not False:
            _issue(issues, "TEMPLATE_AUTHORIZATION", f"amendment.authorization_boundary.{field}", "amendment improperly authorized execution/truth")
    return issues


def validate_revised_receipt(receipt: Mapping[str, Any], amendment_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if receipt.get("template_order_amendment_sha256") != amendment_sha256 or receipt.get("template_order_amendment_path") != "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json":
        _issue(issues, "TEMPLATE_RECEIPT_BINDING", "receipt", "revised receipt is not byte-bound to template-order amendment")
    if receipt.get("caf_transport_amendment_sha256") != CAF_AMENDMENT_SHA256 or receipt.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_SHA256 or receipt.get("prior_caf_runner_sha256") != CAF_RUNNER_SHA256 or receipt.get("prior_caf_failure_receipt_sha256") != CAF_FAILURE_SHA256:
        _issue(issues, "TEMPLATE_RECEIPT_CHAIN", "receipt", "revised receipt lost CAF/failure/no-fallback chain")
    if receipt.get("declarative_user_content_order_preserved") is not True or receipt.get("rendered_audio_placeholder_relocation_count") != 1 or receipt.get("rendered_placeholder_order") != "AUDIO_THEN_FIVE_CONTIGUOUS_IMAGES":
        _issue(issues, "TEMPLATE_RECEIPT_RELOCATION", "receipt", "revised receipt changed declarative content or relocation")
    for field in ("system_prompt_occurrences", "user_prompt_occurrences", "audio_placeholder_occurrences"):
        if receipt.get(field) != 1:
            _issue(issues, "TEMPLATE_RECEIPT_COUNTS", f"receipt.{field}", "prompt/audio occurrence count changed")
    if receipt.get("image_placeholder_occurrences") != 5 or receipt.get("five_image_placeholders_contiguous") is not True or receipt.get("non_audio_rendered_bytes_unchanged") is not True:
        _issue(issues, "TEMPLATE_RECEIPT_COUNTS", "receipt", "five-image/non-audio byte contract changed")
    if receipt.get("intermediate_container") != "CAF" or receipt.get("final_wav_contract_unchanged") is not True or receipt.get("five_frame_contract_unchanged") is not True:
        _issue(issues, "TEMPLATE_RECEIPT_FIXTURE", "receipt", "CAF/WAV/frame contract changed")
    if receipt.get("parser_corrections_spent_before_revised_canary") != 0 or receipt.get("parser_allowance_preserved") is not True:
        _issue(issues, "TEMPLATE_RECEIPT_PARSER", "receipt", "parser budget changed or was spent")
    if receipt.get("output_namespace") != NEW_NAMESPACE or receipt.get("historical_output_overwritten") is not False or receipt.get("canonical_output_overwritten") is not False:
        _issue(issues, "TEMPLATE_RECEIPT_OUTPUT", "receipt", "new namespace/no-overwrite boundary failed")
    for field in ("restricted_content_accessed_before_activation", "restricted_inference_run", "scientific_outcome_run"):
        if receipt.get(field) is not False:
            _issue(issues, "TEMPLATE_RECEIPT_ORDER", f"receipt.{field}", "revised receipt ran restricted/outcome work early")
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
    parser.add_argument("--caf-amendment", type=Path, required=True)
    parser.add_argument("--no-fallback-override", type=Path, required=True)
    parser.add_argument("--caf-runner", type=Path, required=True)
    parser.add_argument("--caf-failure", type=Path, required=True)
    parser.add_argument("--legacy-worker", type=Path, required=True)
    parser.add_argument("--revised-receipt", type=Path)
    args = parser.parse_args(argv)
    amendment, amendment_sha = _load(args.amendment)
    caf_failure, caf_failure_sha = _load(args.caf_failure)
    issues = validate_bound_inputs(_sha256(args.caf_amendment.read_bytes()), _sha256(args.no_fallback_override.read_bytes()), _sha256(args.caf_runner.read_bytes()), caf_failure, caf_failure_sha, _sha256(args.legacy_worker.read_bytes()))
    issues.extend(validate_amendment(amendment, amendment_sha))
    if args.revised_receipt:
        revised, _ = _load(args.revised_receipt)
        issues.extend(validate_revised_receipt(revised, amendment_sha))
    result = {"schema_version": "nursery-gemma-template-order-validation-v1.3", "status": "PASS" if not issues else "FAIL", "issue_count": len(issues), "issues": [issue.as_dict() for issue in issues], "restricted_access": False, "model_or_outcome_run": False}
    print(json.dumps(result, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
