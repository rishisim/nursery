#!/usr/bin/env python3
"""Validate the public-only Gemma v1.4 subprocess-environment correction."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AMENDMENT_SCHEMA = "nursery-gemma4-public-canary-subprocess-environment-correction-amendment-v1.4"
AMENDMENT_SHA256 = "ac05205256bc1e949c0a70a9dc865cebc2ea50beb73770319e816cec0910ceb1"
V13_RUNNER_SHA256 = "dcbdf8714f12a114806ef4c903ee1a7f95b65647f58273b47fd184fb1143cf44"
V13_WORKER_SHA256 = "c78fa8e9c0f995ed611f7fd8f7d2076f03a73d370574d3c92cf5e107f236c38f"
V13_FAILURE_SHA256 = "e69abc985a7400d5305eab9f76217a4bdb9f60baaf78715a87f3abf63ca75492"
V14_RUNNER_SHA256 = "76d040b88cf1f3eb205524bc82839c1851d2ab293b1f7bb633358fe55763204c"
TEMPLATE_SHA256 = "2d117cf0f1619d43d6e71a90aa853241f0a5a7c331e84fc04af8231c5d6081fb"
CAF_SHA256 = "3ecf4bf6c62920d73177d029c18918f38366d34f7d7997642037696def37bbd0"
NO_FALLBACK_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
FIREWALL_SHA256 = "a10f8c2568d402259215afe18f6005ceb4b52431607b42b132a8db849ff32f7b"
ENV_SHA256 = "bf67ff8d311d901d39d8f996339e33f62c72b2f87a0c4ff9d997dce069cdf822"
NEW_NAMESPACE = "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4"
REQUIRED_FLAGS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}
EXACT_ENV = {
    "PATH": "/bin:/usr/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "WANDB_DISABLED": "true", "DO_NOT_TRACK": "1",
    "NO_PROXY": "*", "no_proxy": "*",
}


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


def validate_bound_inputs(
    v13_runner_sha256: str,
    worker_sha256: str,
    failure: Mapping[str, Any],
    failure_sha256: str,
    template_sha256: str,
    caf_sha256: str,
    no_fallback_sha256: str,
    firewall_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    actual = (v13_runner_sha256, worker_sha256, failure_sha256, template_sha256, caf_sha256, no_fallback_sha256, firewall_sha256)
    expected = (V13_RUNNER_SHA256, V13_WORKER_SHA256, V13_FAILURE_SHA256, TEMPLATE_SHA256, CAF_SHA256, NO_FALLBACK_SHA256, FIREWALL_SHA256)
    if actual != expected:
        _add(issues, "ENV_BOUND_INPUT", "bound_inputs", "a byte-bound predecessor changed")
    if (
        failure.get("schema_version") != "nursery-gemma4-public-canary-template-order-failure-v1.3"
        or failure.get("status") != "REVISE"
        or failure.get("failure_code") != "E_SUBPROCESS_ENV"
        or failure.get("runner_sha256") != V13_RUNNER_SHA256
        or failure.get("worker_sha256") != V13_WORKER_SHA256
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
        _add(issues, "ENV_FAILURE_IMMUTABILITY", "v1_3_failure", "exact fail-closed v1.3 failure receipt is required")
    return issues


def validate_amendment(amendment: Mapping[str, Any], amendment_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if amendment_sha256 != AMENDMENT_SHA256 or amendment.get("schema_version") != AMENDMENT_SCHEMA or amendment.get("status") != "FROZEN_PRE_CANARY_SUBPROCESS_ENVIRONMENT_ONLY":
        _add(issues, "ENV_AMENDMENT_IDENTITY", "amendment", "amendment bytes/schema/status changed")
    if amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_SUBPROCESS_ENVIRONMENT_CALL_CORRECTION" or amendment.get("historical_artifacts_modified") is not False:
        _add(issues, "ENV_ADDITIVE_BOUNDARY", "amendment", "amendment is not additive and history-preserving")
    for field in ("public_canary_run", "child_process_launched", "restricted_content_accessed", "restricted_inference_run", "model_generation_called", "scientific_outcome_run"):
        if amendment.get(field) is not False:
            _add(issues, "ENV_FREEZE_ORDER", f"amendment.{field}", "amendment was not frozen before execution")

    bindings = _mapping(amendment.get("immutable_bindings"))
    expected_bindings = {
        "v1_3_public_canary_runner": V13_RUNNER_SHA256,
        "v1_3_worker": V13_WORKER_SHA256,
        "v1_3_public_canary_failure_receipt": V13_FAILURE_SHA256,
        "template_placeholder_order_amendment": TEMPLATE_SHA256,
        "CAF_transport_amendment": CAF_SHA256,
        "no_automatic_fallback_override": NO_FALLBACK_SHA256,
        "firewall_scrubber_implementation_at_failure": FIREWALL_SHA256,
    }
    for field, digest in expected_bindings.items():
        if _mapping(bindings.get(field)).get("sha256") != digest:
            _add(issues, "ENV_IMMUTABLE_BINDING", f"amendment.immutable_bindings.{field}", "bound digest changed")

    trigger = _mapping(amendment.get("trigger"))
    if (
        trigger.get("classification") != "PUBLIC_RUNNER_REDUNDANT_FIREWALL_EXTRA_WHITELIST_REJECTION"
        or trigger.get("stage") != "BEFORE_CHILD_PROCESS_LAUNCH_AND_BEFORE_MODEL_GENERATION"
        or trigger.get("failure_symbol") != "E_SUBPROCESS_ENV"
        or trigger.get("child_process_launched") is not False
        or trigger.get("model_generation_called") is not False
        or trigger.get("raw_model_output_existed") is not False
        or trigger.get("semantic_or_scientific_evidence_involved") is not False
        or trigger.get("outer_fence_parser_corrections_spent") != 0
    ):
        _add(issues, "ENV_TRIGGER", "amendment.trigger", "trigger is not the exact pre-launch public environment failure")

    delta = _mapping(amendment.get("exact_subprocess_environment_delta"))
    old = _mapping(delta.get("frozen_call_semantics"))
    new = _mapping(delta.get("amended_call_semantics"))
    if delta.get("maximum_runner_call_site_changes") != 1 or old.get("function") != "scrubbed_subprocess_environment" or old.get("positional_argument_count") != 1 or old.get("extra_mapping") != REQUIRED_FLAGS:
        _add(issues, "ENV_EXACT_OLD_CALL", "amendment.exact_subprocess_environment_delta", "frozen one-extra-mapping call changed")
    if new.get("function") != "scrubbed_subprocess_environment" or new.get("positional_argument_count") != 0 or new.get("keyword_argument_count") != 0 or new.get("extra_mapping") is not None or new.get("returned_mapping_materialized_for_subprocess") is not True:
        _add(issues, "ENV_EXACT_NEW_CALL", "amendment.exact_subprocess_environment_delta", "amended call must have zero arguments and no extra mapping")

    contract = _mapping(delta.get("scrubber_mandatory_environment_contract"))
    canonical = json.dumps(EXACT_ENV, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if (
        contract.get("inherited_environment_keys_allowed") is not False
        or contract.get("extra_key_count") != 0
        or contract.get("exact_returned_key_count") != 14
        or contract.get("PATH_source") != "os.defpath"
        or contract.get("PATH_value_at_freeze") != "/bin:/usr/bin"
        or contract.get("exact_returned_mapping") != EXACT_ENV
        or contract.get("canonical_JSON_sort_keys_compact_UTF8_sha256") != ENV_SHA256
        or _sha256(canonical) != ENV_SHA256
        or contract.get("credential_authorization_cookie_secret_or_nonblocked_proxy_keys_allowed") is not False
    ):
        _add(issues, "ENV_EXACT_SCRUBBER", "amendment.exact_subprocess_environment_delta.scrubber_mandatory_environment_contract", "exact credential-free offline/no-proxy environment changed")

    assertions = _mapping(delta.get("mandatory_post_call_assertions"))
    if (
        assertions.get("comparison") != "EXACT_UTF8_BYTE_EQUALITY"
        or assertions.get("required_key_values") != REQUIRED_FLAGS
        or assertions.get("required_value_UTF8_hex") != {key: "31" for key in REQUIRED_FLAGS}
        or assertions.get("failure_symbol_on_missing_or_different_value") != "E_SUBPROCESS_ENV"
        or assertions.get("assert_before_subprocess_launch") is not True
    ):
        _add(issues, "ENV_POST_CALL_ASSERTION", "amendment.exact_subprocess_environment_delta.mandatory_post_call_assertions", "mandatory byte assertions changed")
    for field in ("firewall_implementation_changed", "firewall_extra_whitelist_changed", "mandatory_environment_defaults_changed", "credential_or_proxy_inheritance_allowed", "network_policy_changed", "subprocess_argv_changed", "subprocess_cwd_changed", "subprocess_stdio_or_passed_FDs_changed", "semantic_fields_changed", "model_runtime_or_artifact_fields_changed", "scientific_fields_changed", "parser_allowance_changed"):
        if delta.get(field) is not False:
            _add(issues, "ENV_SCOPE_EXPANSION", f"amendment.exact_subprocess_environment_delta.{field}", "non-environment-call field changed")
    if delta.get("does_not_spend_outer_fence_parser_correction") is not True:
        _add(issues, "ENV_PARSER_BUDGET", "amendment.exact_subprocess_environment_delta", "environment correction spent parser allowance")

    canary = _mapping(amendment.get("preserved_public_canary_contract"))
    for field in ("CAF_transport_and_final_WAV_contract_unchanged", "template_placeholder_relocation_unchanged", "declarative_messages_and_exact_prompt_text_unchanged", "five_frame_generator_and_offsets_unchanged", "model_repository_revision_manifest_weights_and_runtime_pins_unchanged", "schema_parser_and_semantic_label_space_unchanged"):
        if canary.get(field) is not True:
            _add(issues, "ENV_CANARY_CONTRACT", f"amendment.preserved_public_canary_contract.{field}", "public canary contract changed")
    if canary.get("outer_fence_parser_correction_maximum") != 1 or canary.get("outer_fence_parser_corrections_spent_before_amendment") != 0 or canary.get("parser_mode") != "PRIMARY":
        _add(issues, "ENV_PARSER_BUDGET", "amendment.preserved_public_canary_contract", "parser contract changed")

    scientific = _mapping(amendment.get("preserved_sample_export_and_scientific_contract"))
    for field in ("sample_15_items_900_seconds_137_candidate_windows_unchanged", "K5_complement_protection_and_outward_rounding_unchanged", "coverage_abstention_path_and_schema_gates_unchanged", "Qwen3_read_only_digest_contract_unchanged", "agreement_is_not_a_pass_gate", "simulator_oracle_remains_only_evaluation_truth"):
        if scientific.get(field) is not True:
            _add(issues, "ENV_SCIENTIFIC_CONTRACT", f"amendment.preserved_sample_export_and_scientific_contract.{field}", "sample/export/scientific contract changed")
    for field in ("calibration_range_or_scientific_protocol_changed", "causal_thresholds_changed", "automatic_fallback_allowed"):
        if scientific.get(field) is not False:
            _add(issues, "ENV_SCIENTIFIC_CONTRACT", f"amendment.preserved_sample_export_and_scientific_contract.{field}", "protocol/fallback change enabled")

    rule = _mapping(amendment.get("overlay_application_rule"))
    for field in ("reject_if_any_bound_digest_differs", "reject_if_scrubber_is_called_with_any_argument_or_extra_mapping", "reject_if_any_required_returned_flag_is_missing_or_not_exactly_UTF8_0x31", "reject_if_the_scrubbed_mapping_is_mutated_after_return", "reject_if_any_unlisted_runner_or_contract_field_changes"):
        if rule.get(field) is not True:
            _add(issues, "ENV_APPLICATION_RULE", f"amendment.overlay_application_rule.{field}", "fail-closed rule weakened")
    if rule.get("reuse_prior_v1_3_failure_or_success_receipt_as_v1_4_receipt") is not False:
        _add(issues, "ENV_APPLICATION_RULE", "amendment.overlay_application_rule", "prior receipt reuse enabled")

    namespace = _mapping(amendment.get("new_immutable_output_namespace"))
    if namespace != {
        "path": NEW_NAMESPACE, "must_not_exist_before_execution": True,
        "owner_private_during_public_execution": True,
        "future_public_canary_receipt": "public_common_schema_canary_receipt_v1_4.json",
        "future_public_canary_failure": "public_common_schema_canary_failure_v1_4.json",
        "future_runtime_lock_receipt": "runtime_lock_receipt_v1_4.json",
        "future_activation_receipt": "full_activation_receipt_v1_4.json",
        "historical_output_paths_overwritten": False,
    }:
        _add(issues, "ENV_OUTPUT_NAMESPACE", "amendment.new_immutable_output_namespace", "v1.4 namespace/no-overwrite contract changed")
    auth = _mapping(amendment.get("authorization_boundary"))
    for field in ("amendment_is_canary_execution", "amendment_is_activation_receipt", "public_canary_authorized_by_amendment_alone", "child_process_launch_authorized_by_amendment_alone", "restricted_inference_authorized", "scientific_outcome_authorized", "model_agreement_or_pseudo_labels_are_ground_truth"):
        if auth.get(field) is not False:
            _add(issues, "ENV_AUTHORIZATION", f"amendment.authorization_boundary.{field}", "amendment improperly authorizes execution or truth")
    return issues


def validate_corrected_runner(source: str, runner_sha256: str) -> list[Issue]:
    """Statically validate the sealed runner; never import or execute it."""
    issues: list[Issue] = []
    if runner_sha256 != V14_RUNNER_SHA256:
        _add(issues, "ENV_RUNNER_IDENTITY", "corrected_runner", "sealed v1.4 runner bytes changed")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues + [Issue("ENV_RUNNER_SYNTAX", "corrected_runner", "runner is not valid Python")]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "scrubbed_subprocess_environment"]
    if len(calls) != 1 or calls[0].args or calls[0].keywords:
        _add(issues, "ENV_RUNNER_ZERO_ARGUMENT_CALL", "corrected_runner", "runner must contain one zero-argument scrubber call")
    if "environment = dict(" not in source or "env=environment" not in source:
        _add(issues, "ENV_RUNNER_UNMUTATED_MAPPING", "corrected_runner", "same materialized scrubbed mapping must reach Popen")
    for key in REQUIRED_FLAGS:
        if f'"{key}"' not in source:
            _add(issues, "ENV_RUNNER_REQUIRED_ASSERTION", f"corrected_runner.{key}", "mandatory offline assertion missing")
    if '.encode("utf-8") != b"1"' not in source or 'raise EnvironmentCanaryError("E_SUBPROCESS_ENV")' not in source:
        _add(issues, "ENV_RUNNER_REQUIRED_ASSERTION", "corrected_runner", "exact UTF-8 byte assertion is missing")
    if "os.environ" in source or ".update(environment" in source or "environment.update(" in source:
        _add(issues, "ENV_RUNNER_INHERITANCE", "corrected_runner", "environment inheritance or mutation detected")
    return issues


def validate_receipt(receipt: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    expected = {
        "subprocess_environment_correction_amendment_sha256": AMENDMENT_SHA256,
        "prior_v1_3_runner_sha256": V13_RUNNER_SHA256,
        "prior_v1_3_failure_receipt_sha256": V13_FAILURE_SHA256,
        "worker_sha256": V13_WORKER_SHA256,
        "scrubber_called_with_extra_mapping": False,
        "mandatory_offline_defaults_asserted_before_launch": True,
        "scrubbed_environment_mutated_after_return": False,
        "parser_corrections_spent_before_environment_canary": 0,
        "output_namespace": NEW_NAMESPACE,
        "historical_output_overwritten": False,
        "canonical_output_overwritten": False,
        "restricted_content_accessed_before_activation": False,
        "restricted_inference_run": False,
        "scientific_outcome_run": False,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            _add(issues, "ENV_RECEIPT_CONTRACT", f"receipt.{field}", "v1.4 receipt chain/environment/privacy field changed")
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
    parser.add_argument("--v1-3-runner", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--v1-3-failure", type=Path, required=True)
    parser.add_argument("--template-amendment", type=Path, required=True)
    parser.add_argument("--caf-amendment", type=Path, required=True)
    parser.add_argument("--no-fallback-override", type=Path, required=True)
    parser.add_argument("--firewall", type=Path, required=True)
    parser.add_argument("--corrected-runner", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    amendment, amendment_sha = _load(args.amendment)
    failure, failure_sha = _load(args.v1_3_failure)
    issues = validate_bound_inputs(*[
        _sha256(args.v1_3_runner.read_bytes()), _sha256(args.worker.read_bytes()), failure, failure_sha,
        _sha256(args.template_amendment.read_bytes()), _sha256(args.caf_amendment.read_bytes()),
        _sha256(args.no_fallback_override.read_bytes()), _sha256(args.firewall.read_bytes()),
    ])
    issues.extend(validate_amendment(amendment, amendment_sha))
    if args.corrected_runner:
        payload = args.corrected_runner.read_bytes()
        issues.extend(validate_corrected_runner(payload.decode("utf-8"), _sha256(payload)))
    if args.receipt:
        receipt, _ = _load(args.receipt)
        issues.extend(validate_receipt(receipt))
    result = {"schema_version": "nursery-gemma-subprocess-environment-validation-v1.4", "status": "PASS" if not issues else "FAIL", "issue_count": len(issues), "issues": [issue.as_dict() for issue in issues], "restricted_access": False, "model_or_outcome_run": False}
    print(json.dumps(result, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
