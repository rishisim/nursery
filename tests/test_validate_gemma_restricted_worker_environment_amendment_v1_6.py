from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_gemma_restricted_worker_environment_amendment_v1_6.py"
SPEC = importlib.util.spec_from_file_location("validate_worker_env_v16", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)
AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_restricted_worker_environment_v1_6/frozen_restricted_worker_environment_amendment_v1_6.json"
ERRATUM = ROOT / "docs/nursery_program_convergence_v1/gemma4_restricted_worker_environment_v1_6_1/frozen_success_receipt_path_compatibility_erratum_v1_6_1.json"
EXECUTION = ROOT / "docs/nursery_program_convergence_v1/calibration_conditioned_symbolic_execution_contract_v2.json"
V15_FAILURE = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_preflight_delegation_failure_v1_5.json"
V16_COORDINATOR = ROOT / "scripts/nursery_gemma_substitution_calibration_v1_6.py"
BINDINGS = {
    "v1_5_preflight_delegation_amendment": ROOT / "docs/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/frozen_preflight_delegation_recursion_amendment_v1_5.json",
    "v1_5_public_canary_runner": ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_5.py",
    "v1_5_restricted_calibration_coordinator": ROOT / "scripts/nursery_gemma_substitution_calibration_v1_5.py",
    "v1_5_public_canary_receipt": ROOT / "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/public_common_schema_canary_receipt_v1_5.json",
    "v1_5_runtime_lock_receipt": ROOT / "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/runtime_lock_receipt_v1_5.json",
    "v1_5_full_activation_receipt": ROOT / "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/full_activation_receipt_v1_5.json",
    "v1_5_restricted_calibration_failure_receipt": V15_FAILURE,
    "base_restricted_calibration_coordinator": ROOT / "scripts/nursery_gemma_substitution_calibration.py",
    "restricted_Gemma_worker": ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py",
    "local_inference_firewall": ROOT / "scripts/childlens_local_inference_firewall_v1_3.py",
    "v1_4_public_subprocess_environment_amendment": ROOT / "docs/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/frozen_subprocess_environment_correction_amendment_v1_4.json",
    "no_automatic_fallback_override": ROOT / "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json",
}


def load(path: Path):
    payload = path.read_bytes(); return json.loads(payload), MODULE._sha256(payload)


def codes(issues):
    return {issue.code for issue in issues}


def test_live_amendment_erratum_and_bindings_pass() -> None:
    amendment, amendment_sha = load(AMENDMENT); erratum, erratum_sha = load(ERRATUM); failure, _ = load(V15_FAILURE)
    digests = {name: MODULE._sha256(path.read_bytes()) for name, path in BINDINGS.items()}
    assert MODULE.validate_bound_inputs(digests, failure) == []
    assert MODULE.validate_amendment(amendment, amendment_sha) == []
    assert MODULE.validate_path_erratum(erratum, erratum_sha, MODULE._sha256(EXECUTION.read_bytes())) == []


def test_sealed_v16_coordinator_passes_static_validation() -> None:
    payload = V16_COORDINATOR.read_bytes()
    assert MODULE.validate_corrected_coordinator(payload.decode(), MODULE._sha256(payload)) == []


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("status",), "PASS", "WORKER_ENV_AMENDMENT_IDENTITY"),
        (("restricted_rerun_executed",), True, "WORKER_ENV_FREEZE_ORDER"),
        (("Gemma_worker_process_launched",), True, "WORKER_ENV_FREEZE_ORDER"),
        (("diagnosed_failure_and_restricted_access_state", "typed_local_diagnostic_symbol"), "OTHER", "WORKER_ENV_DIAGNOSIS"),
        (("diagnosed_failure_and_restricted_access_state", "Gemma_worker_process_launched"), True, "WORKER_ENV_DIAGNOSIS"),
        (("diagnosed_failure_and_restricted_access_state", "Gemma_worker_inference_run"), True, "WORKER_ENV_DIAGNOSIS"),
        (("diagnosed_failure_and_restricted_access_state", "Gemma_model_output_existed"), True, "WORKER_ENV_DIAGNOSIS"),
        (("diagnosed_failure_and_restricted_access_state", "scientific_endpoint_opened"), True, "WORKER_ENV_DIAGNOSIS"),
        (("immutable_bindings", "v1_5_restricted_calibration_coordinator", "sha256"), "0" * 64, "WORKER_ENV_IMMUTABLE_BINDING"),
        (("exact_restricted_worker_environment_delta", "maximum_worker_environment_call_site_changes"), 2, "WORKER_ENV_EXACT_DELTA"),
        (("exact_restricted_worker_environment_delta", "amended_scrubber_extra_mapping", "OMP_NUM_THREADS"), "3", "WORKER_ENV_EXACT_DELTA"),
        (("exact_restricted_worker_environment_delta", "amended_scrubber_extra_mapping", "HF_HUB_OFFLINE"), "1", "WORKER_ENV_EXACT_DELTA"),
        (("exact_restricted_worker_environment_delta", "firewall_mandatory_environment_contract", "canonical_JSON_sort_keys_compact_UTF8_sha256"), "0" * 64, "WORKER_ENV_MANDATORY_DEFAULTS"),
        (("exact_restricted_worker_environment_delta", "TMPDIR_rule", "other_post_scrubber_environment_mutations_allowed"), True, "WORKER_ENV_TMPDIR"),
        (("exact_restricted_worker_environment_delta", "expected_final_environment_shape", "exact_total_key_count"), 18, "WORKER_ENV_FINAL_SHAPE"),
        (("exact_restricted_worker_environment_delta", "worker_subprocess_argv_changed"), True, "WORKER_ENV_SCOPE_EXPANSION"),
        (("exact_restricted_worker_environment_delta", "worker_subprocess_cwd_stdio_passed_FDs_or_checkpointing_changed"), True, "WORKER_ENV_SCOPE_EXPANSION"),
        (("exact_restricted_worker_environment_delta", "bounded_media_FD_set_or_order_changed"), True, "WORKER_ENV_SCOPE_EXPANSION"),
        (("exact_restricted_worker_environment_delta", "network_denial_or_firewall_policy_changed"), True, "WORKER_ENV_SCOPE_EXPANSION"),
        (("exact_restricted_worker_environment_delta", "model_prompt_schema_or_parser_changed"), True, "WORKER_ENV_SCOPE_EXPANSION"),
        (("exact_restricted_worker_environment_delta", "sample_selection_or_thresholds_changed"), True, "WORKER_ENV_SCOPE_EXPANSION"),
        (("preserved_calibration_and_scientific_contract", "Qwen3_hypotheses_and_digest_remain_read_only"), False, "WORKER_ENV_SCIENTIFIC_CONTRACT"),
        (("preserved_calibration_and_scientific_contract", "simulator_oracle_remains_only_evaluation_truth"), False, "WORKER_ENV_SCIENTIFIC_CONTRACT"),
        (("preserved_calibration_and_scientific_contract", "automatic_fallback_allowed"), True, "WORKER_ENV_SCIENTIFIC_CONTRACT"),
        (("overlay_application_rule", "reject_if_any_extra_key_other_than_the_two_exact_thread_controls_is_passed_to_the_scrubber"), False, "WORKER_ENV_APPLICATION_RULE"),
        (("new_immutable_output_namespace", "historical_output_paths_overwritten"), True, "WORKER_ENV_OUTPUT_NAMESPACE"),
        (("authorization_boundary", "scientific_endpoint_authorized"), True, "WORKER_ENV_AUTHORIZATION"),
    ],
)
def test_amendment_mutations_fail_closed(path, value, code) -> None:
    amendment, _ = load(AMENDMENT); mutated = copy.deepcopy(amendment); target = mutated
    for part in path[:-1]: target = target[part]
    target[path[-1]] = value
    assert code in codes(MODULE.validate_amendment(mutated, MODULE.AMENDMENT_SHA256))


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("status",), "PASS", "WORKER_ENV_ERRATUM_IDENTITY"),
        (("restricted_rerun_executed",), True, "WORKER_ENV_ERRATUM_FREEZE_ORDER"),
        (("immutable_bindings", "frozen_causal_execution_contract_v2", "sha256"), "0" * 64, "WORKER_ENV_ERRATUM_BINDING"),
        (("exact_field_supersession", "maximum_superseded_fields"), 2, "WORKER_ENV_ERRATUM_EXACT_FIELD"),
        (("exact_field_supersession", "superseding_repo_relative_success_path"), "output/wrong.json", "WORKER_ENV_ERRATUM_EXACT_FIELD"),
        (("success_path_creation_contract", "exclusive_creation_required"), False, "WORKER_ENV_SUCCESS_EXCLUSIVE_CREATE"),
        (("success_path_creation_contract", "overwrite_truncate_replace_rename_over_existing_or_reuse_allowed"), True, "WORKER_ENV_SUCCESS_EXCLUSIVE_CREATE"),
        (("failure_and_diagnostic_path_preserved", "resolved_failure_path"), "output/wrong.json", "WORKER_ENV_FAILURE_PATH"),
        (("failure_and_diagnostic_path_preserved", "failure_or_diagnostic_artifact_at_planned_success_path_allowed"), True, "WORKER_ENV_FAILURE_PATH"),
        (("all_other_v1_6_fields_preserved", "exact_14_mandatory_plus_2_thread_plus_TMPDIR_environment_unchanged"), False, "WORKER_ENV_ERRATUM_PRESERVATION"),
    ],
)
def test_erratum_mutations_fail_closed(path, value, code) -> None:
    erratum, _ = load(ERRATUM); mutated = copy.deepcopy(erratum); target = mutated
    for part in path[:-1]: target = target[part]
    target[path[-1]] = value
    assert code in codes(MODULE.validate_path_erratum(mutated, MODULE.ERRATUM_SHA256, MODULE.EXECUTION_CONTRACT_SHA256))


def test_unapproved_extra_and_environment_inheritance_fail_static_validation() -> None:
    source = '''
def _worker_environment(work):
    environment = dict(firewall.scrubbed_subprocess_environment({"OMP_NUM_THREADS":"2","VECLIB_MAXIMUM_THREADS":"2","MKL_NUM_THREADS":"2"}))
    environment["TMPDIR"] = str(work)
    return environment
'''
    found = codes(MODULE.validate_corrected_coordinator(source, MODULE.V16_COORDINATOR_SHA256))
    assert "WORKER_ENV_COORDINATOR_EXTRAS" in found and "WORKER_ENV_COORDINATOR_INHERITANCE" in found


def test_validator_has_no_execution_or_restricted_discovery_path() -> None:
    source = SCRIPT.read_text()
    assert "import subprocess" not in source and "subprocess.run" not in source and "subprocess.Popen" not in source
    for forbidden in ("keychain", "urlopen", "requests"):
        assert forbidden not in source.lower()
