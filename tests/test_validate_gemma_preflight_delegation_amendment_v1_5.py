from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_gemma_preflight_delegation_amendment_v1_5.py"
SPEC = importlib.util.spec_from_file_location("validate_preflight_v15", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

AMENDMENT_PATH = ROOT / "docs/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/frozen_preflight_delegation_recursion_amendment_v1_5.json"
FAILURE_PATH = ROOT / "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/public_common_schema_canary_failure_v1_4.json"
V15_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_5.py"
PATHS = {
    "v1_4_subprocess_environment_amendment": ROOT / "docs/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/frozen_subprocess_environment_correction_amendment_v1_4.json",
    "v1_4_public_canary_runner": ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_4.py",
    "v1_4_calibration_coordinator": ROOT / "scripts/nursery_gemma_substitution_calibration_v1_4.py",
    "v1_4_public_canary_failure_receipt": FAILURE_PATH,
    "v1_3_public_canary_runner": ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_3.py",
    "v1_3_worker": ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py",
    "template_placeholder_order_amendment": ROOT / "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json",
    "CAF_transport_amendment": ROOT / "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json",
    "no_automatic_fallback_override": ROOT / "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json",
}


def load(path: Path):
    payload = path.read_bytes()
    return json.loads(payload), MODULE._sha256(payload)


def codes(issues):
    return {issue.code for issue in issues}


def test_live_amendment_and_bound_inputs_pass() -> None:
    amendment, digest = load(AMENDMENT_PATH)
    failure, _ = load(FAILURE_PATH)
    digests = {key: MODULE._sha256(path.read_bytes()) for key, path in PATHS.items()}
    assert MODULE.validate_amendment(amendment, digest) == []
    assert MODULE.validate_bound_inputs(digests, failure) == []


def test_sealed_v15_runner_passes_static_validation() -> None:
    payload = V15_RUNNER.read_bytes()
    assert MODULE.validate_corrected_runner(payload.decode(), MODULE._sha256(payload)) == []


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("status",), "PASS", "PREFLIGHT_AMENDMENT_IDENTITY"),
        (("public_canary_run",), True, "PREFLIGHT_FREEZE_ORDER"),
        (("fixture_created",), True, "PREFLIGHT_FREEZE_ORDER"),
        (("diagnosed_failures", "observed_v1_4_receipt", "receipt_failure_code"), "OTHER", "PREFLIGHT_DIAGNOSIS"),
        (("diagnosed_failures", "public_synthetic_followup", "failure_symbol"), "OTHER", "PREFLIGHT_DIAGNOSIS"),
        (("immutable_bindings", "v1_4_public_canary_runner", "sha256"), "0" * 64, "PREFLIGHT_IMMUTABLE_BINDING"),
        (("insufficient_correction_explicitly_forbidden", "dynamic_self_call_through_v13_public_preflight"), False, "PREFLIGHT_FORBIDDEN_STRATEGY"),
        (("insufficient_correction_explicitly_forbidden", "capture_original_v1_3_preflight_then_rerun_it_under_mutated_historical_globals"), False, "PREFLIGHT_FORBIDDEN_STRATEGY"),
        (("exact_preflight_delegation_delta", "precompute_count_per_sandboxed_execution"), 2, "PREFLIGHT_PRECOMPUTE"),
        (("exact_preflight_delegation_delta", "pure_projection_callable_contract", "module_global_reads"), True, "PREFLIGHT_PURE_PROJECTION"),
        (("exact_preflight_delegation_delta", "pure_projection_callable_contract", "parser_mode_comparison"), "EQUAL", "PREFLIGHT_PURE_PROJECTION"),
        (("exact_preflight_delegation_delta", "pure_projection_callable_contract", "other_return_or_fallback_allowed"), True, "PREFLIGHT_PURE_PROJECTION"),
        (("exact_preflight_delegation_delta", "direct_v1_5_mutated_module_targets"), ["v13._public_preflight", "v12._public_preflight"], "PREFLIGHT_MUTATION_SCOPE"),
        (("exact_preflight_delegation_delta", "historical_preflight_source_changed"), True, "PREFLIGHT_SCOPE_EXPANSION"),
        (("exact_preflight_delegation_delta", "subprocess_environment_contract_changed"), True, "PREFLIGHT_SCOPE_EXPANSION"),
        (("exact_preflight_delegation_delta", "parser_allowance_changed"), True, "PREFLIGHT_SCOPE_EXPANSION"),
        (("preserved_public_canary_contract", "v1_4_exact_14_key_scrubbed_environment_and_four_byte_assertions_unchanged"), False, "PREFLIGHT_CANARY_CONTRACT"),
        (("preserved_public_canary_contract", "declarative_messages_and_exact_prompt_text_unchanged"), False, "PREFLIGHT_CANARY_CONTRACT"),
        (("preserved_sample_export_and_scientific_contract", "simulator_oracle_remains_only_evaluation_truth"), False, "PREFLIGHT_SCIENTIFIC_CONTRACT"),
        (("preserved_sample_export_and_scientific_contract", "automatic_fallback_allowed"), True, "PREFLIGHT_SCIENTIFIC_CONTRACT"),
        (("overlay_application_rule", "reject_if_any_nested_historical_preflight_is_dynamically_recomputed"), False, "PREFLIGHT_APPLICATION_RULE"),
        (("overlay_application_rule", "reject_if_v13_or_v12_preflight_callable_is_not_restored_on_every_path"), False, "PREFLIGHT_APPLICATION_RULE"),
        (("new_immutable_output_namespace", "path"), "output/wrong", "PREFLIGHT_OUTPUT_NAMESPACE"),
        (("new_immutable_output_namespace", "historical_output_paths_overwritten"), True, "PREFLIGHT_OUTPUT_NAMESPACE"),
        (("authorization_boundary", "restricted_inference_authorized"), True, "PREFLIGHT_AUTHORIZATION"),
        (("authorization_boundary", "scientific_outcome_authorized"), True, "PREFLIGHT_AUTHORIZATION"),
    ],
)
def test_amendment_mutations_fail_closed(path, value, code) -> None:
    amendment, _ = load(AMENDMENT_PATH)
    mutated = copy.deepcopy(amendment)
    target = mutated
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    assert code in codes(MODULE.validate_amendment(mutated, MODULE.AMENDMENT_SHA256))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failure_code", "OTHER"),
        ("runner_sha256", "0" * 64),
        ("restricted_inference_run", True),
        ("scientific_outcome_run", True),
        ("automatic_calibration_fallback_allowed", True),
    ],
)
def test_v14_failure_mutations_fail_closed(field, value) -> None:
    failure, _ = load(FAILURE_PATH)
    failure[field] = value
    digests = {key: MODULE._sha256(path.read_bytes()) for key, path in PATHS.items()}
    assert "PREFLIGHT_FAILURE_IMMUTABILITY" in codes(MODULE.validate_bound_inputs(digests, failure))


def test_dynamic_recursion_and_captured_original_are_rejected() -> None:
    source = '''
def sandboxed_execute(seal):
    precomputed = _public_preflight(seal["parser_mode"])
    saved_v13_preflight = v13._public_preflight
    def projection(parser_mode):
        return v13._public_preflight(parser_mode)
    try:
        v13._public_preflight = projection
        return v13.sandboxed_execute(seal)
    finally:
        v13._public_preflight = saved_v13_preflight
'''
    assert "PREFLIGHT_RUNNER_DYNAMIC_REVALIDATION" in codes(MODULE.validate_corrected_runner(source, MODULE.V15_RUNNER_SHA256))
    source = source.replace("return v13._public_preflight(parser_mode)", "return saved_v13_preflight(parser_mode)")
    assert "PREFLIGHT_RUNNER_DYNAMIC_REVALIDATION" in codes(MODULE.validate_corrected_runner(source, MODULE.V15_RUNNER_SHA256))


def test_validator_has_no_execution_or_restricted_discovery_path() -> None:
    source = SCRIPT.read_text()
    assert "import subprocess" not in source
    assert "subprocess.run" not in source
    assert "subprocess.Popen" not in source
    for forbidden in ("quarantine", "keychain", "urlopen", "requests"):
        assert forbidden not in source.lower()
