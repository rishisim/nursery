#!/usr/bin/env python3
"""Validate Gemma v1.6 worker-environment amendment and v1.6.1 path erratum."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AMENDMENT_SCHEMA = "nursery-gemma4-restricted-worker-subprocess-environment-amendment-v1.6"
AMENDMENT_SHA256 = "316fd2f1f9e4c1b69936703f86eb6b72e0fff574e9115d2a5bb138e5fa7da577"
ERRATUM_SCHEMA = "nursery-gemma4-restricted-worker-success-receipt-path-compatibility-erratum-v1.6.1"
ERRATUM_SHA256 = "c85eb4034fbf7c02e4ddba8b48cb36c877ce27a968f22a619a3b4175ef6dd693"
EXECUTION_CONTRACT_SHA256 = "db760f4135f81f7f78184a80f8abae16578918ff10b44c952f83a76cf5daa810"
V15_AMENDMENT_SHA256 = "16049ce3bcbeb9f3ae91c28695c7dafb10c9a013ff64b301d977b96f9943429a"
V15_RUNNER_SHA256 = "4657eccaf04f08e0b689d14be5e3577ed483882e3f1212fcdab0f476ee28025b"
V15_COORDINATOR_SHA256 = "66e0454d854ecada6eddb1580312c63fd38656892b15bfb8b4ccf1339604958e"
V15_CANARY_SHA256 = "9843e812b605979641a6a777f870348c9aa3c7f10ec382a082bb87c84def4e3b"
V15_RUNTIME_SHA256 = "69e0ef3a47198956bc06a123d6bfbc6b448ebeef5c8e6ed5e248bb882ba92b1f"
V15_ACTIVATION_SHA256 = "10cb7d0151e2fff40a3f5bf006ea510a3f82a4223410dec51ebfb08a34f14d7f"
V15_FAILURE_SHA256 = "eb8dc3313c746bc6b6ca100ebab30853607191c625ed0bfc1a2f3fe292dc6b79"
BASE_COORDINATOR_SHA256 = "4fdacb4a0bf4e03627c4c036ec65107bb249bf0a4b35564f3cf7a5a58dbeeeed"
WORKER_SHA256 = "c78fa8e9c0f995ed611f7fd8f7d2076f03a73d370574d3c92cf5e107f236c38f"
FIREWALL_SHA256 = "a10f8c2568d402259215afe18f6005ceb4b52431607b42b132a8db849ff32f7b"
V14_ENV_AMENDMENT_SHA256 = "ac05205256bc1e949c0a70a9dc865cebc2ea50beb73770319e816cec0910ceb1"
NO_FALLBACK_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
MANDATORY_ENV_SHA256 = "bf67ff8d311d901d39d8f996339e33f62c72b2f87a0c4ff9d997dce069cdf822"
V16_COORDINATOR_SHA256 = "c532777f3492a4c5a407814f48e792270ca309d9c7d92e2be37af6ec2a4ccae4"
NAMESPACE = "output/nursery_program_convergence_v1/gemma4_restricted_worker_environment_v1_6"
SUCCESS_PATH = "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
FAILURE_PATH = f"{NAMESPACE}/childlens_pseudo_calibration_gemma_substitution_failure_v1_6.json"
THREAD_EXTRAS = {"OMP_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}
MANDATORY_FLAGS = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
OLD_EXTRAS = {**THREAD_EXTRAS, **MANDATORY_FLAGS}


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
        "v1_5_preflight_delegation_amendment": V15_AMENDMENT_SHA256,
        "v1_5_public_canary_runner": V15_RUNNER_SHA256,
        "v1_5_restricted_calibration_coordinator": V15_COORDINATOR_SHA256,
        "v1_5_public_canary_receipt": V15_CANARY_SHA256,
        "v1_5_runtime_lock_receipt": V15_RUNTIME_SHA256,
        "v1_5_full_activation_receipt": V15_ACTIVATION_SHA256,
        "v1_5_restricted_calibration_failure_receipt": V15_FAILURE_SHA256,
        "base_restricted_calibration_coordinator": BASE_COORDINATOR_SHA256,
        "restricted_Gemma_worker": WORKER_SHA256,
        "local_inference_firewall": FIREWALL_SHA256,
        "v1_4_public_subprocess_environment_amendment": V14_ENV_AMENDMENT_SHA256,
        "no_automatic_fallback_override": NO_FALLBACK_SHA256,
    }
    if dict(digests) != expected:
        _add(issues, "WORKER_ENV_BOUND_INPUT", "bound_inputs", "one or more byte-bound predecessors changed")
    if (
        failure.get("schema_version") != "nursery-childlens-gemma-substitution-preflight-delegation-failure-v1.5"
        or failure.get("status") != "REVISE"
        or failure.get("failure_code") != "E_INTERNAL"
        or failure.get("preflight_delegation_recursion_amendment_sha256") != V15_AMENDMENT_SHA256
        or failure.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_SHA256
        or failure.get("automatic_calibration_fallback_allowed") is not False
        or failure.get("scientific_endpoint_allowed") is not False
        or failure.get("restricted_payload_exported") is not False
    ):
        _add(issues, "WORKER_ENV_FAILURE_IMMUTABILITY", "v1_5_failure", "exact fail-closed v1.5 restricted failure receipt is required")
    return issues


def validate_amendment(amendment: Mapping[str, Any], amendment_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if amendment_sha256 != AMENDMENT_SHA256 or amendment.get("schema_version") != AMENDMENT_SCHEMA or amendment.get("status") != "FROZEN_PRE_RESTRICTED_RERUN_WORKER_ENVIRONMENT_ONLY":
        _add(issues, "WORKER_ENV_AMENDMENT_IDENTITY", "amendment", "amendment bytes/schema/status changed")
    if amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_RESTRICTED_WORKER_ENVIRONMENT_CALL_CORRECTION" or amendment.get("historical_artifacts_modified") is not False:
        _add(issues, "WORKER_ENV_ADDITIVE_BOUNDARY", "amendment", "amendment is not additive and history-preserving")
    for field in ("restricted_rerun_executed", "Gemma_worker_process_launched", "Gemma_worker_inference_run", "Gemma_model_output_existed", "scientific_endpoint_opened", "scientific_outcome_run"):
        if amendment.get(field) is not False:
            _add(issues, "WORKER_ENV_FREEZE_ORDER", f"amendment.{field}", "amendment was not frozen before worker/outcome execution")

    diagnosis = _mapping(amendment.get("diagnosed_failure_and_restricted_access_state"))
    if (
        diagnosis.get("v1_5_failure_receipt_code") != "E_INTERNAL"
        or diagnosis.get("typed_local_diagnostic_symbol") != "E_SUBPROCESS_ENV"
        or diagnosis.get("failure_stage") != "AFTER_BOUNDED_RESTRICTED_PREFLIGHT_AND_BEFORE_GEMMA_WORKER_SUBPROCESS_LAUNCH"
        or diagnosis.get("v1_5_public_seal_passed") is not True
        or diagnosis.get("restricted_inputs_locally_opened_or_read_during_bounded_preflight") is not True
        or diagnosis.get("bounded_media_file_descriptors_opened") is not True
        or diagnosis.get("Gemma_worker_process_launched") is not False
        or diagnosis.get("Gemma_worker_inference_run") is not False
        or diagnosis.get("Gemma_model_output_existed") is not False
        or diagnosis.get("restricted_payload_exported") is not False
        or diagnosis.get("restricted_payload_logged_or_serialized_to_repository") is not False
        or diagnosis.get("scientific_endpoint_opened") is not False
        or diagnosis.get("outer_fence_parser_corrections_spent") != 0
    ):
        _add(issues, "WORKER_ENV_DIAGNOSIS", "amendment.diagnosed_failure_and_restricted_access_state", "prior bounded failure stage or no-worker/no-output/no-endpoint state changed")

    bindings = _mapping(amendment.get("immutable_bindings"))
    expected_bindings = {
        "v1_5_preflight_delegation_amendment": V15_AMENDMENT_SHA256,
        "v1_5_public_canary_runner": V15_RUNNER_SHA256,
        "v1_5_restricted_calibration_coordinator": V15_COORDINATOR_SHA256,
        "v1_5_public_canary_receipt": V15_CANARY_SHA256,
        "v1_5_runtime_lock_receipt": V15_RUNTIME_SHA256,
        "v1_5_full_activation_receipt": V15_ACTIVATION_SHA256,
        "v1_5_restricted_calibration_failure_receipt": V15_FAILURE_SHA256,
        "base_restricted_calibration_coordinator": BASE_COORDINATOR_SHA256,
        "restricted_Gemma_worker": WORKER_SHA256,
        "local_inference_firewall": FIREWALL_SHA256,
        "v1_4_public_subprocess_environment_amendment": V14_ENV_AMENDMENT_SHA256,
        "no_automatic_fallback_override": NO_FALLBACK_SHA256,
    }
    for field, digest in expected_bindings.items():
        if _mapping(bindings.get(field)).get("sha256") != digest:
            _add(issues, "WORKER_ENV_IMMUTABLE_BINDING", f"amendment.immutable_bindings.{field}", "predecessor digest changed")

    delta = _mapping(amendment.get("exact_restricted_worker_environment_delta"))
    if delta.get("maximum_worker_environment_call_site_changes") != 1 or delta.get("frozen_scrubber_extra_mapping") != OLD_EXTRAS or delta.get("amended_scrubber_extra_mapping") != THREAD_EXTRAS:
        _add(issues, "WORKER_ENV_EXACT_DELTA", "amendment.exact_restricted_worker_environment_delta", "exact one-call old/new extra mapping changed")
    if delta.get("exact_deleted_extra_keys") != list(MANDATORY_FLAGS) or delta.get("deleted_extra_key_count") != 4 or delta.get("retained_thread_extra_key_count") != 2 or delta.get("retained_thread_extra_values_UTF8_hex") != {key: "32" for key in THREAD_EXTRAS}:
        _add(issues, "WORKER_ENV_EXACT_DELTA", "amendment.exact_restricted_worker_environment_delta", "exact four deletions/two retained thread extras changed")
    mandatory = _mapping(delta.get("firewall_mandatory_environment_contract"))
    if (
        mandatory.get("exact_mandatory_key_count") != 14
        or mandatory.get("canonical_JSON_sort_keys_compact_UTF8_sha256") != MANDATORY_ENV_SHA256
        or mandatory.get("removed_extra_keys_remain_mandatory_with_values") != MANDATORY_FLAGS
        or mandatory.get("removed_extra_values_UTF8_hex") != {key: "31" for key in MANDATORY_FLAGS}
        or mandatory.get("mandatory_defaults_changed") is not False
        or mandatory.get("extra_whitelist_changed") is not False
    ):
        _add(issues, "WORKER_ENV_MANDATORY_DEFAULTS", "amendment.exact_restricted_worker_environment_delta.firewall_mandatory_environment_contract", "mandatory offline defaults changed")
    tmpdir = _mapping(delta.get("TMPDIR_rule"))
    if (
        tmpdir.get("existing_addition_preserved") is not True
        or tmpdir.get("assignment") != "environment['TMPDIR'] = str(work)"
        or tmpdir.get("work_must_be_absolute_owner_private_quarantine_local_directory") is not True
        or tmpdir.get("TMPDIR_may_not_be_exported_logged_or_written_to_repository_artifacts") is not True
        or tmpdir.get("other_post_scrubber_environment_mutations_allowed") is not False
    ):
        _add(issues, "WORKER_ENV_TMPDIR", "amendment.exact_restricted_worker_environment_delta.TMPDIR_rule", "existing quarantine TMPDIR-only rule changed")
    shape = _mapping(delta.get("expected_final_environment_shape"))
    if shape != {"mandatory_key_count": 14, "allowed_thread_extra_key_count": 2, "quarantine_local_TMPDIR_key_count": 1, "exact_total_key_count": 17, "inherited_environment_keys_allowed": False, "credential_authorization_cookie_secret_or_nonblocked_proxy_keys_allowed": False}:
        _add(issues, "WORKER_ENV_FINAL_SHAPE", "amendment.exact_restricted_worker_environment_delta.expected_final_environment_shape", "exact 17-key credential-free environment shape changed")
    for field in ("worker_subprocess_argv_changed", "worker_subprocess_cwd_stdio_passed_FDs_or_checkpointing_changed", "bounded_media_FD_set_or_order_changed", "network_denial_or_firewall_policy_changed", "quarantine_or_privacy_policy_changed", "model_prompt_schema_or_parser_changed", "sample_selection_or_thresholds_changed"):
        if delta.get(field) is not False:
            _add(issues, "WORKER_ENV_SCOPE_EXPANSION", f"amendment.exact_restricted_worker_environment_delta.{field}", "worker/checkpoint/network/model/sample/schema contract changed")
    if delta.get("does_not_spend_outer_fence_parser_correction") is not True:
        _add(issues, "WORKER_ENV_PARSER_BUDGET", "amendment.exact_restricted_worker_environment_delta", "worker environment correction spent parser allowance")

    preserved = _mapping(amendment.get("preserved_calibration_and_scientific_contract"))
    for field in ("v1_5_public_canary_activation_and_runtime_receipts_reused_read_only", "v1_5_preflight_delegation_and_v1_4_public_environment_corrections_unchanged", "model_repository_revision_manifest_weights_and_runtime_pins_unchanged", "prompt_message_media_schema_and_parser_contracts_unchanged", "sample_15_items_900_seconds_137_candidate_windows_unchanged", "K5_complement_protection_and_outward_rounding_unchanged", "coverage_abstention_path_and_schema_gates_unchanged", "Qwen3_hypotheses_and_digest_remain_read_only", "agreement_is_not_a_pass_gate", "simulator_oracle_remains_only_evaluation_truth"):
        if preserved.get(field) is not True:
            _add(issues, "WORKER_ENV_SCIENTIFIC_CONTRACT", f"amendment.preserved_calibration_and_scientific_contract.{field}", "calibration/scientific invariant changed")
    for field in ("calibration_range_or_scientific_protocol_changed", "causal_thresholds_changed", "automatic_fallback_allowed"):
        if preserved.get(field) is not False:
            _add(issues, "WORKER_ENV_SCIENTIFIC_CONTRACT", f"amendment.preserved_calibration_and_scientific_contract.{field}", "protocol/fallback change enabled")

    rule = _mapping(amendment.get("overlay_application_rule"))
    for field in ("reject_if_any_bound_digest_differs", "reject_if_any_extra_key_other_than_the_two_exact_thread_controls_is_passed_to_the_scrubber", "reject_if_any_mandatory_default_or_its_digest_changes", "reject_if_TMPDIR_is_not_the_existing_owner_private_quarantine_local_work_directory", "reject_if_final_environment_has_any_key_beyond_14_mandatory_2_thread_and_TMPDIR", "reject_if_any_worker_argument_FD_network_model_sample_prompt_parser_threshold_or_scientific_field_changes"):
        if rule.get(field) is not True:
            _add(issues, "WORKER_ENV_APPLICATION_RULE", f"amendment.overlay_application_rule.{field}", "fail-closed application rule weakened")
    if rule.get("reuse_prior_v1_5_failure_or_success_receipt_as_v1_6_receipt") is not False:
        _add(issues, "WORKER_ENV_APPLICATION_RULE", "amendment.overlay_application_rule", "prior receipt reuse enabled")
    namespace = _mapping(amendment.get("new_immutable_output_namespace"))
    if namespace.get("path") != NAMESPACE or namespace.get("must_not_exist_before_execution") is not True or namespace.get("future_restricted_calibration_failure") != "childlens_pseudo_calibration_gemma_substitution_failure_v1_6.json" or namespace.get("prior_v1_5_failure_receipt_remains_immutable") is not True or namespace.get("historical_output_paths_overwritten") is not False:
        _add(issues, "WORKER_ENV_OUTPUT_NAMESPACE", "amendment.new_immutable_output_namespace", "v1.6 diagnostic namespace/no-overwrite contract changed")
    auth = _mapping(amendment.get("authorization_boundary"))
    for field in ("amendment_is_restricted_rerun", "amendment_is_calibration_receipt", "restricted_worker_launch_authorized_by_amendment_alone", "scientific_endpoint_authorized", "scientific_outcome_authorized", "automatic_fallback_authorized", "model_agreement_or_pseudo_labels_are_ground_truth"):
        if auth.get(field) is not False:
            _add(issues, "WORKER_ENV_AUTHORIZATION", f"amendment.authorization_boundary.{field}", "amendment improperly authorizes execution/truth")
    return issues


def validate_path_erratum(erratum: Mapping[str, Any], erratum_sha256: str, execution_contract_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if erratum_sha256 != ERRATUM_SHA256 or erratum.get("schema_version") != ERRATUM_SCHEMA or erratum.get("status") != "FROZEN_PRE_RESTRICTED_RERUN_SUCCESS_PATH_ONLY":
        _add(issues, "WORKER_ENV_ERRATUM_IDENTITY", "erratum", "path erratum bytes/schema/status changed")
    if erratum.get("erratum_mode") != "ADDITIVE_ONE_FIELD_OUTPUT_PATH_COMPATIBILITY_OVERRIDE" or erratum.get("v1_6_amendment_modified") is not False or erratum.get("historical_artifacts_modified") is not False:
        _add(issues, "WORKER_ENV_ERRATUM_SCOPE", "erratum", "erratum is not one-field additive override")
    for field in ("restricted_rerun_executed", "restricted_content_accessed", "Gemma_worker_process_launched", "Gemma_worker_inference_run", "scientific_endpoint_opened", "scientific_outcome_run"):
        if erratum.get(field) is not False:
            _add(issues, "WORKER_ENV_ERRATUM_FREEZE_ORDER", f"erratum.{field}", "erratum was not frozen before execution")
    bindings = _mapping(erratum.get("immutable_bindings"))
    if _mapping(bindings.get("v1_6_restricted_worker_environment_amendment")).get("sha256") != AMENDMENT_SHA256 or _mapping(bindings.get("frozen_causal_execution_contract_v2")).get("sha256") != EXECUTION_CONTRACT_SHA256 or execution_contract_sha256 != EXECUTION_CONTRACT_SHA256:
        _add(issues, "WORKER_ENV_ERRATUM_BINDING", "erratum.immutable_bindings", "v1.6 amendment or execution contract digest changed")
    field = _mapping(erratum.get("exact_field_supersession"))
    if (
        field.get("source_JSON_pointer") != "/new_immutable_output_namespace/future_restricted_calibration_receipt"
        or field.get("frozen_field_value") != "childlens_pseudo_calibration_gemma_substitution_receipt_v1_6.json"
        or field.get("frozen_namespaced_resolution_superseded") != f"{NAMESPACE}/childlens_pseudo_calibration_gemma_substitution_receipt_v1_6.json"
        or field.get("superseding_repo_relative_success_path") != SUCCESS_PATH
        or field.get("maximum_superseded_fields") != 1
        or field.get("success_receipt_schema_or_content_changed") is not False
    ):
        _add(issues, "WORKER_ENV_ERRATUM_EXACT_FIELD", "erratum.exact_field_supersession", "exact success-path-only supersession changed")
    creation = _mapping(erratum.get("success_path_creation_contract"))
    for name in ("path_must_be_absent_before_restricted_execution", "absence_verified_at_freeze", "exclusive_creation_required", "fail_if_path_exists"):
        if creation.get(name) is not True:
            _add(issues, "WORKER_ENV_SUCCESS_EXCLUSIVE_CREATE", f"erratum.success_path_creation_contract.{name}", "success receipt exclusive-create gate weakened")
    for name in ("overwrite_truncate_replace_rename_over_existing_or_reuse_allowed", "partial_or_unvalidated_success_receipt_allowed", "historical_file_overwritten"):
        if creation.get(name) is not False:
            _add(issues, "WORKER_ENV_SUCCESS_EXCLUSIVE_CREATE", f"erratum.success_path_creation_contract.{name}", "success receipt overwrite/partial output enabled")
    failure = _mapping(erratum.get("failure_and_diagnostic_path_preserved"))
    if failure.get("resolved_failure_path") != FAILURE_PATH or failure.get("failure_or_diagnostic_artifact_at_planned_success_path_allowed") is not False or failure.get("v1_5_failure_receipt_overwritten") is not False:
        _add(issues, "WORKER_ENV_FAILURE_PATH", "erratum.failure_and_diagnostic_path_preserved", "failure escaped v1.6 namespace or overwrote history")
    preserved = _mapping(erratum.get("all_other_v1_6_fields_preserved"))
    for name in ("restricted_worker_environment_delta_unchanged", "exact_14_mandatory_plus_2_thread_plus_TMPDIR_environment_unchanged", "quarantine_privacy_network_FD_and_subprocess_contracts_unchanged", "model_prompt_schema_parser_and_sample_unchanged", "calibration_thresholds_and_gates_unchanged", "no_automatic_fallback_rule_unchanged", "simulator_oracle_scientific_boundary_unchanged"):
        if preserved.get(name) is not True:
            _add(issues, "WORKER_ENV_ERRATUM_PRESERVATION", f"erratum.all_other_v1_6_fields_preserved.{name}", "non-path field changed")
    if preserved.get("scientific_or_content_fields_changed") is not False:
        _add(issues, "WORKER_ENV_ERRATUM_PRESERVATION", "erratum.all_other_v1_6_fields_preserved", "scientific/content change enabled")
    return issues


def validate_corrected_coordinator(source: str, coordinator_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if coordinator_sha256 != V16_COORDINATOR_SHA256:
        _add(issues, "WORKER_ENV_COORDINATOR_IDENTITY", "coordinator", "sealed v1.6 coordinator bytes changed")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return issues + [Issue("WORKER_ENV_COORDINATOR_SYNTAX", "coordinator", "coordinator is not valid Python")]
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    worker_env = functions.get("_worker_environment")
    if worker_env is None:
        return issues + [Issue("WORKER_ENV_COORDINATOR_STRUCTURE", "coordinator", "single worker environment override is missing")]
    calls = [node for node in ast.walk(worker_env) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "scrubbed_subprocess_environment"]
    try:
        extra = ast.literal_eval(calls[0].args[0]) if len(calls) == 1 and len(calls[0].args) == 1 and not calls[0].keywords else None
    except (ValueError, TypeError):
        extra = None
    if extra != THREAD_EXTRAS:
        _add(issues, "WORKER_ENV_COORDINATOR_EXTRAS", "coordinator._worker_environment", "scrubber extras are not exactly the two thread controls")
    segment = ast.get_source_segment(source, worker_env) or ""
    for token in ("work.is_absolute()", "legacy._private_directory(work)", "work.resolve(strict=True)", "MANDATORY_ENVIRONMENT_SHA256", "len(mandatory) != 14", "len(environment) != 17", 'environment["TMPDIR"] = str(work)'):
        if token not in segment:
            _add(issues, "WORKER_ENV_COORDINATOR_ASSERTION", f"coordinator.{token}", "mandatory environment/TMPDIR/final-shape assertion missing")
    if "os.environ" in segment or "environment.update(" in segment or "MKL_NUM_THREADS" in segment:
        _add(issues, "WORKER_ENV_COORDINATOR_INHERITANCE", "coordinator._worker_environment", "environment inheritance, mutation, or unapproved extra detected")
    for token in (SUCCESS_PATH, NAMESPACE, "childlens_pseudo_calibration_gemma_substitution_failure_v1_6.json", "O_WRONLY | os.O_CREAT | os.O_EXCL", "0o600"):
        if token not in source:
            _add(issues, "WORKER_ENV_COORDINATOR_OUTPUT_PATH", f"coordinator.{token}", "success exclusive-create or namespaced failure path missing")
    return issues


def _load(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes(); value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, _sha256(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True); parser.add_argument("--path-erratum", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True); parser.add_argument("--v1-5-failure", type=Path, required=True)
    parser.add_argument("--binding", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--coordinator", type=Path)
    args = parser.parse_args(argv)
    amendment, amendment_sha = _load(args.amendment); erratum, erratum_sha = _load(args.path_erratum); failure, _ = _load(args.v1_5_failure)
    digests = {name: _sha256(Path(path).read_bytes()) for name, path in args.binding}
    issues = validate_bound_inputs(digests, failure); issues.extend(validate_amendment(amendment, amendment_sha)); issues.extend(validate_path_erratum(erratum, erratum_sha, _sha256(args.execution_contract.read_bytes())))
    if args.coordinator:
        payload = args.coordinator.read_bytes(); issues.extend(validate_corrected_coordinator(payload.decode("utf-8"), _sha256(payload)))
    result = {"schema_version": "nursery-gemma-restricted-worker-environment-validation-v1.6.1", "status": "PASS" if not issues else "FAIL", "issue_count": len(issues), "issues": [issue.as_dict() for issue in issues], "restricted_access": False, "model_or_outcome_run": False}
    print(json.dumps(result, sort_keys=True)); return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
