#!/usr/bin/env python3
"""Validate the public-only Gemma v1.5 immutable-seal projection overlay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AMENDMENT_SCHEMA = "nursery-gemma4-public-canary-preflight-delegation-recursion-amendment-v1.5"
AMENDMENT_SHA256 = "16049ce3bcbeb9f3ae91c28695c7dafb10c9a013ff64b301d977b96f9943429a"
V14_ENV_AMENDMENT_SHA256 = "ac05205256bc1e949c0a70a9dc865cebc2ea50beb73770319e816cec0910ceb1"
V14_RUNNER_SHA256 = "76d040b88cf1f3eb205524bc82839c1851d2ab293b1f7bb633358fe55763204c"
V14_COORDINATOR_SHA256 = "7c62c302205c505d9aa01c6b5e9894cdc1b9ac3d9be335b6066362773b0448a2"
V14_FAILURE_SHA256 = "db0b1b9431d1e1eaa602d5e09f6ce7505b6bdc13adcfd492341ee7bd3fae1f2a"
V13_RUNNER_SHA256 = "dcbdf8714f12a114806ef4c903ee1a7f95b65647f58273b47fd184fb1143cf44"
V13_WORKER_SHA256 = "c78fa8e9c0f995ed611f7fd8f7d2076f03a73d370574d3c92cf5e107f236c38f"
TEMPLATE_SHA256 = "2d117cf0f1619d43d6e71a90aa853241f0a5a7c331e84fc04af8231c5d6081fb"
CAF_SHA256 = "3ecf4bf6c62920d73177d029c18918f38366d34f7d7997642037696def37bbd0"
NO_FALLBACK_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
V15_RUNNER_SHA256 = "4657eccaf04f08e0b689d14be5e3577ed483882e3f1212fcdab0f476ee28025b"
NEW_NAMESPACE = "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5"


@dataclass(frozen=True)
class Issue:
    code: str
    location: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "location": self.location, "message": self.message}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _add(issues: list[Issue], code: str, location: str, message: str) -> None:
    issues.append(Issue(code, location, message))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_bound_inputs(digests: Mapping[str, str], failure: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    expected = {
        "v1_4_subprocess_environment_amendment": V14_ENV_AMENDMENT_SHA256,
        "v1_4_public_canary_runner": V14_RUNNER_SHA256,
        "v1_4_calibration_coordinator": V14_COORDINATOR_SHA256,
        "v1_4_public_canary_failure_receipt": V14_FAILURE_SHA256,
        "v1_3_public_canary_runner": V13_RUNNER_SHA256,
        "v1_3_worker": V13_WORKER_SHA256,
        "template_placeholder_order_amendment": TEMPLATE_SHA256,
        "CAF_transport_amendment": CAF_SHA256,
        "no_automatic_fallback_override": NO_FALLBACK_SHA256,
    }
    if dict(digests) != expected:
        _add(issues, "PREFLIGHT_BOUND_INPUT", "bound_inputs", "one or more exact predecessor bytes changed")
    if (
        failure.get("schema_version") != "nursery-gemma4-public-canary-subprocess-environment-failure-v1.4"
        or failure.get("status") != "REVISE"
        or failure.get("failure_code") != "E_SANDBOXED_CANARY"
        or failure.get("runner_sha256") != V14_RUNNER_SHA256
        or failure.get("worker_sha256") != V13_WORKER_SHA256
        or failure.get("subprocess_environment_correction_amendment_sha256") != V14_ENV_AMENDMENT_SHA256
        or failure.get("template_placeholder_order_amendment_sha256") != TEMPLATE_SHA256
        or failure.get("caf_transport_amendment_sha256") != CAF_SHA256
        or failure.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_SHA256
        or failure.get("parser_correction_eligible") is not False
        or failure.get("restricted_payload_accessed") is not False
        or failure.get("restricted_inference_run") is not False
        or failure.get("scientific_outcome_run") is not False
        or failure.get("automatic_calibration_fallback_allowed") is not False
        or failure.get("historical_output_overwritten") is not False
        or failure.get("canonical_output_overwritten") is not False
    ):
        _add(issues, "PREFLIGHT_FAILURE_IMMUTABILITY", "v1_4_failure", "exact fail-closed v1.4 receipt is required")
    return issues


def validate_amendment(amendment: Mapping[str, Any], amendment_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if amendment_sha256 != AMENDMENT_SHA256 or amendment.get("schema_version") != AMENDMENT_SCHEMA or amendment.get("status") != "FROZEN_PRE_CANARY_PREFLIGHT_DELEGATION_ONLY":
        _add(issues, "PREFLIGHT_AMENDMENT_IDENTITY", "amendment", "amendment bytes/schema/status changed")
    if amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_IMMUTABLE_SEAL_PROJECTION_OVERLAY" or amendment.get("historical_artifacts_modified") is not False:
        _add(issues, "PREFLIGHT_ADDITIVE_BOUNDARY", "amendment", "amendment is not additive and history-preserving")
    for field in ("public_canary_run", "fixture_created", "model_loaded_by_failed_v1_4_attempt", "model_generation_called", "restricted_content_accessed", "restricted_inference_run", "scientific_endpoint_opened", "scientific_outcome_run"):
        if amendment.get(field) is not False:
            _add(issues, "PREFLIGHT_FREEZE_ORDER", f"amendment.{field}", "amendment was not frozen before execution")

    diagnosed = _mapping(amendment.get("diagnosed_failures"))
    first = _mapping(diagnosed.get("observed_v1_4_receipt"))
    second = _mapping(diagnosed.get("public_synthetic_followup"))
    if first.get("receipt_failure_code") != "E_SANDBOXED_CANARY" or "recursion" not in str(first.get("root_cause", "")).lower():
        _add(issues, "PREFLIGHT_DIAGNOSIS", "amendment.diagnosed_failures.observed_v1_4_receipt", "recursion defect is not exactly bound")
    if second.get("failure_symbol") != "E_TEMPLATE_BINDING" or second.get("scientific_or_restricted_content_involved") is not False or "insufficient" not in str(second.get("root_cause", "")).lower():
        _add(issues, "PREFLIGHT_DIAGNOSIS", "amendment.diagnosed_failures.public_synthetic_followup", "captured-original insufficiency is not bound")
    if diagnosed.get("model_generation_called") is not False or diagnosed.get("raw_model_output_existed") is not False or diagnosed.get("outer_fence_parser_corrections_spent") != 0:
        _add(issues, "PREFLIGHT_DIAGNOSIS", "amendment.diagnosed_failures", "diagnosis crossed model/parser boundary")

    bindings = _mapping(amendment.get("immutable_bindings"))
    expected_bindings = {
        "v1_4_subprocess_environment_amendment": V14_ENV_AMENDMENT_SHA256,
        "v1_4_public_canary_runner": V14_RUNNER_SHA256,
        "v1_4_calibration_coordinator": V14_COORDINATOR_SHA256,
        "v1_4_public_canary_failure_receipt": V14_FAILURE_SHA256,
        "v1_3_public_canary_runner": V13_RUNNER_SHA256,
        "v1_3_worker": V13_WORKER_SHA256,
        "template_placeholder_order_amendment": TEMPLATE_SHA256,
        "CAF_transport_amendment": CAF_SHA256,
        "no_automatic_fallback_override": NO_FALLBACK_SHA256,
    }
    for field, digest in expected_bindings.items():
        if _mapping(bindings.get(field)).get("sha256") != digest:
            _add(issues, "PREFLIGHT_IMMUTABLE_BINDING", f"amendment.immutable_bindings.{field}", "predecessor digest changed")

    forbidden = _mapping(amendment.get("insufficient_correction_explicitly_forbidden"))
    for field in ("capture_original_v1_3_preflight_then_rerun_it_under_mutated_historical_globals", "dynamic_self_call_through_v13_public_preflight", "rerun_any_v1_5_v1_4_v1_3_or_v1_2_amendment_preflight_during_nested_seal_equality_checks", "derive_or_modify_the_seal_after_historical_worker_fixture_or_preflight_bindings_are_overlaid"):
        if forbidden.get(field) is not True:
            _add(issues, "PREFLIGHT_FORBIDDEN_STRATEGY", f"amendment.insufficient_correction_explicitly_forbidden.{field}", "known-insufficient strategy not forbidden")

    delta = _mapping(amendment.get("exact_preflight_delegation_delta"))
    if delta.get("maximum_scientific_or_content_changes") != 0 or delta.get("precompute_count_per_sandboxed_execution") != 1 or delta.get("immutable_representation") != "Canonical compact sorted-key UTF-8 JSON bytes of the fully validated v1.5 public seal, retained in a local lexical closure together with its SHA-256 and exact parser_mode string.":
        _add(issues, "PREFLIGHT_PRECOMPUTE", "amendment.exact_preflight_delegation_delta", "single immutable precompute contract changed")
    required = delta.get("required_precompute_and_validation")
    if not isinstance(required, list) or len(required) != 5 or "exactly once" not in required[1] or "bytes are identical" not in required[3] or "Do not call any amendment preflight again" not in required[4]:
        _add(issues, "PREFLIGHT_PRECOMPUTE", "amendment.exact_preflight_delegation_delta.required_precompute_and_validation", "precompute/equality/no-revalidation algorithm changed")
    projection = _mapping(delta.get("pure_projection_callable_contract"))
    expected_projection = {
        "state_source": "Only the local immutable canonical seal bytes, their SHA-256, and the exact parser_mode captured before historical state mutation.",
        "module_global_reads": False, "filesystem_reads": False, "environment_reads": False,
        "network_access": False, "content_or_model_reads": False, "accepted_argument_count": 1,
        "accepted_argument": "parser_mode", "parser_mode_comparison": "EXACT_TYPE_AND_UTF8_BYTE_EQUALITY",
        "parser_mismatch_failure_symbol": "E_PARSER_MODE",
        "return_value": "A fresh JSON decoding of the immutable precomputed canonical seal bytes.",
        "return_canonical_bytes_must_equal_precomputed_bytes": True,
        "return_SHA256_must_equal_precomputed_SHA256": True, "other_return_or_fallback_allowed": False,
    }
    if dict(projection) != expected_projection:
        _add(issues, "PREFLIGHT_PURE_PROJECTION", "amendment.exact_preflight_delegation_delta.pure_projection_callable_contract", "pure parser-exact projection changed")
    restoration = delta.get("temporary_substitution_and_restoration")
    if not isinstance(restoration, list) or len(restoration) != 6 or "pure projection callable" not in restoration[1] or "finally" not in restoration[4] or "identities" not in restoration[5]:
        _add(issues, "PREFLIGHT_RESTORATION", "amendment.exact_preflight_delegation_delta.temporary_substitution_and_restoration", "substitution/restoration contract changed")
    if delta.get("direct_v1_5_mutated_module_targets") != ["v13._public_preflight"] or delta.get("v12_projection_propagation_is_only_via_bound_v1_3_executor") is not True:
        _add(issues, "PREFLIGHT_MUTATION_SCOPE", "amendment.exact_preflight_delegation_delta", "historical mutation scope expanded")
    for field in ("historical_preflight_source_changed", "historical_worker_or_fixture_binding_policy_changed", "subprocess_environment_contract_changed", "subprocess_argv_cwd_stdio_or_passed_FDs_changed", "network_or_firewall_policy_changed", "semantic_fields_changed", "model_runtime_or_artifact_fields_changed", "parser_allowance_changed"):
        if delta.get(field) is not False:
            _add(issues, "PREFLIGHT_SCOPE_EXPANSION", f"amendment.exact_preflight_delegation_delta.{field}", "unlisted runtime/content/scientific field changed")
    if delta.get("does_not_spend_outer_fence_parser_correction") is not True:
        _add(issues, "PREFLIGHT_PARSER_BUDGET", "amendment.exact_preflight_delegation_delta", "preflight correction spent parser allowance")

    canary = _mapping(amendment.get("preserved_public_canary_contract"))
    for field in ("v1_4_exact_14_key_scrubbed_environment_and_four_byte_assertions_unchanged", "CAF_transport_and_final_WAV_contract_unchanged", "template_placeholder_relocation_unchanged", "declarative_messages_and_exact_prompt_text_unchanged", "five_frame_generator_and_offsets_unchanged", "model_repository_revision_manifest_weights_and_runtime_pins_unchanged", "schema_parser_and_semantic_label_space_unchanged"):
        if canary.get(field) is not True:
            _add(issues, "PREFLIGHT_CANARY_CONTRACT", f"amendment.preserved_public_canary_contract.{field}", "public canary invariant changed")
    if canary.get("outer_fence_parser_correction_maximum") != 1 or canary.get("outer_fence_parser_corrections_spent_before_amendment") != 0 or canary.get("parser_mode") != "PRIMARY":
        _add(issues, "PREFLIGHT_PARSER_BUDGET", "amendment.preserved_public_canary_contract", "parser contract changed")

    scientific = _mapping(amendment.get("preserved_sample_export_and_scientific_contract"))
    for field in ("sample_15_items_900_seconds_137_candidate_windows_unchanged", "K5_complement_protection_and_outward_rounding_unchanged", "coverage_abstention_path_and_schema_gates_unchanged", "Qwen3_read_only_digest_contract_unchanged", "agreement_is_not_a_pass_gate", "simulator_oracle_remains_only_evaluation_truth"):
        if scientific.get(field) is not True:
            _add(issues, "PREFLIGHT_SCIENTIFIC_CONTRACT", f"amendment.preserved_sample_export_and_scientific_contract.{field}", "sample/export/scientific invariant changed")
    for field in ("calibration_range_or_scientific_protocol_changed", "causal_thresholds_changed", "automatic_fallback_allowed"):
        if scientific.get(field) is not False:
            _add(issues, "PREFLIGHT_SCIENTIFIC_CONTRACT", f"amendment.preserved_sample_export_and_scientific_contract.{field}", "protocol/fallback change enabled")

    rule = _mapping(amendment.get("overlay_application_rule"))
    for field in ("reject_if_any_bound_digest_differs", "reject_if_the_full_seal_is_computed_more_than_once_inside_sandboxed_execute", "reject_if_any_nested_historical_preflight_is_dynamically_recomputed", "reject_if_projection_parser_mode_is_not_exact", "reject_if_projection_output_differs_from_precomputed_canonical_seal", "reject_if_v13_or_v12_preflight_callable_is_not_restored_on_every_path", "reject_if_any_unlisted_runtime_content_or_contract_field_changes"):
        if rule.get(field) is not True:
            _add(issues, "PREFLIGHT_APPLICATION_RULE", f"amendment.overlay_application_rule.{field}", "fail-closed application rule weakened")
    if rule.get("reuse_prior_v1_4_failure_or_success_receipt_as_v1_5_receipt") is not False:
        _add(issues, "PREFLIGHT_APPLICATION_RULE", "amendment.overlay_application_rule", "prior receipt reuse enabled")

    namespace = _mapping(amendment.get("new_immutable_output_namespace"))
    if namespace != {"path": NEW_NAMESPACE, "must_not_exist_before_execution": True, "owner_private_during_public_execution": True, "future_public_canary_receipt": "public_common_schema_canary_receipt_v1_5.json", "future_public_canary_failure": "public_common_schema_canary_failure_v1_5.json", "future_runtime_lock_receipt": "runtime_lock_receipt_v1_5.json", "future_activation_receipt": "full_activation_receipt_v1_5.json", "historical_output_paths_overwritten": False}:
        _add(issues, "PREFLIGHT_OUTPUT_NAMESPACE", "amendment.new_immutable_output_namespace", "v1.5 namespace/no-overwrite contract changed")
    auth = _mapping(amendment.get("authorization_boundary"))
    for field in ("amendment_is_canary_execution", "amendment_is_activation_receipt", "public_canary_authorized_by_amendment_alone", "child_process_launch_authorized_by_amendment_alone", "restricted_inference_authorized", "scientific_endpoint_authorized", "scientific_outcome_authorized", "model_agreement_or_pseudo_labels_are_ground_truth"):
        if auth.get(field) is not False:
            _add(issues, "PREFLIGHT_AUTHORIZATION", f"amendment.authorization_boundary.{field}", "amendment improperly authorizes execution or truth")
    return issues


def validate_corrected_runner(source: str, runner_sha256: str) -> list[Issue]:
    """Static validation only: parse source without importing or executing it."""
    issues: list[Issue] = []
    if runner_sha256 != V15_RUNNER_SHA256:
        _add(issues, "PREFLIGHT_RUNNER_IDENTITY", "corrected_runner", "sealed v1.5 runner bytes changed")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues + [Issue("PREFLIGHT_RUNNER_SYNTAX", "corrected_runner", "runner is not valid Python")]
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    sandboxed = functions.get("sandboxed_execute")
    if sandboxed is None:
        return issues + [Issue("PREFLIGHT_RUNNER_STRUCTURE", "corrected_runner", "sandboxed_execute is missing")]
    segment = ast.get_source_segment(source, sandboxed) or ""
    required_tokens = (
        "expected_bytes", "expected_sha256", "parser_mode", "E_PARSER_MODE", "json.loads", "finally",
        "saved_v13_preflight", "saved_v12_preflight", "v13._public_preflight",
        "v13.sandboxed_execute", "is not saved_v13_preflight", "is not saved_v12_preflight",
    )
    for token in required_tokens:
        if token not in segment:
            _add(issues, "PREFLIGHT_RUNNER_STRUCTURE", f"corrected_runner.{token}", "immutable projection/restoration sentinel missing")
    if segment.count("v13._public_preflight =") != 2 or "v12._public_preflight =" in segment:
        _add(issues, "PREFLIGHT_RUNNER_MUTATION_SCOPE", "corrected_runner", "v13 preflight must be substituted/restored exactly once; v12 propagation remains delegated")
    # The old recursion and captured-original dynamic-revalidation forms are forbidden.
    if "return v13._public_preflight(parser_mode)" in segment or "return saved_v13_preflight(parser_mode)" in segment:
        _add(issues, "PREFLIGHT_RUNNER_DYNAMIC_REVALIDATION", "corrected_runner", "nested historical preflight is dynamically re-run")
    if segment.count("_public_preflight(") != 1:
        _add(issues, "PREFLIGHT_RUNNER_PRECOMPUTE_COUNT", "corrected_runner", "full v1.5 preflight must be called exactly once in sandboxed_execute")
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
    parser.add_argument("--v1-4-environment-amendment", type=Path, required=True)
    parser.add_argument("--v1-4-runner", type=Path, required=True)
    parser.add_argument("--v1-4-coordinator", type=Path, required=True)
    parser.add_argument("--v1-4-failure", type=Path, required=True)
    parser.add_argument("--v1-3-runner", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--template-amendment", type=Path, required=True)
    parser.add_argument("--caf-amendment", type=Path, required=True)
    parser.add_argument("--no-fallback-override", type=Path, required=True)
    parser.add_argument("--corrected-runner", type=Path)
    args = parser.parse_args(argv)
    amendment, amendment_sha = _load(args.amendment)
    failure, failure_sha = _load(args.v1_4_failure)
    digests = {
        "v1_4_subprocess_environment_amendment": _sha256(args.v1_4_environment_amendment.read_bytes()),
        "v1_4_public_canary_runner": _sha256(args.v1_4_runner.read_bytes()),
        "v1_4_calibration_coordinator": _sha256(args.v1_4_coordinator.read_bytes()),
        "v1_4_public_canary_failure_receipt": failure_sha,
        "v1_3_public_canary_runner": _sha256(args.v1_3_runner.read_bytes()),
        "v1_3_worker": _sha256(args.worker.read_bytes()),
        "template_placeholder_order_amendment": _sha256(args.template_amendment.read_bytes()),
        "CAF_transport_amendment": _sha256(args.caf_amendment.read_bytes()),
        "no_automatic_fallback_override": _sha256(args.no_fallback_override.read_bytes()),
    }
    issues = validate_bound_inputs(digests, failure)
    issues.extend(validate_amendment(amendment, amendment_sha))
    if args.corrected_runner:
        payload = args.corrected_runner.read_bytes()
        issues.extend(validate_corrected_runner(payload.decode("utf-8"), _sha256(payload)))
    result = {"schema_version": "nursery-gemma-preflight-delegation-validation-v1.5", "status": "PASS" if not issues else "FAIL", "issue_count": len(issues), "issues": [issue.as_dict() for issue in issues], "restricted_access": False, "model_or_outcome_run": False}
    print(json.dumps(result, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
