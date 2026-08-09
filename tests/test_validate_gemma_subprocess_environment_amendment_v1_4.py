from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_gemma_subprocess_environment_amendment_v1_4.py"
SPEC = importlib.util.spec_from_file_location("validate_env_v14", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

AMENDMENT_PATH = ROOT / "docs/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/frozen_subprocess_environment_correction_amendment_v1_4.json"
FAILURE_PATH = ROOT / "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/public_common_schema_canary_failure_v1_3.json"
V13_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_3.py"
V14_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_4.py"
WORKER = ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py"
TEMPLATE = ROOT / "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json"
CAF = ROOT / "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json"
NO_FALLBACK = ROOT / "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json"
FIREWALL = ROOT / "scripts/childlens_local_inference_firewall_v1_3.py"


def load(path: Path):
    payload = path.read_bytes()
    return json.loads(payload), MODULE._sha256(payload)


def codes(issues):
    return {issue.code for issue in issues}


def test_live_amendment_passes() -> None:
    amendment, digest = load(AMENDMENT_PATH)
    assert MODULE.validate_amendment(amendment, digest) == []


def test_live_bound_inputs_pass() -> None:
    failure, failure_digest = load(FAILURE_PATH)
    assert MODULE.validate_bound_inputs(
        MODULE._sha256(V13_RUNNER.read_bytes()), MODULE._sha256(WORKER.read_bytes()),
        failure, failure_digest, MODULE._sha256(TEMPLATE.read_bytes()),
        MODULE._sha256(CAF.read_bytes()), MODULE._sha256(NO_FALLBACK.read_bytes()),
        MODULE._sha256(FIREWALL.read_bytes()),
    ) == []


def test_sealed_corrected_runner_passes_static_validation() -> None:
    payload = V14_RUNNER.read_bytes()
    assert MODULE.validate_corrected_runner(payload.decode(), MODULE._sha256(payload)) == []


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (("status",), "PASS", "ENV_AMENDMENT_IDENTITY"),
        (("public_canary_run",), True, "ENV_FREEZE_ORDER"),
        (("immutable_bindings", "v1_3_public_canary_runner", "sha256"), "0" * 64, "ENV_IMMUTABLE_BINDING"),
        (("trigger", "child_process_launched"), True, "ENV_TRIGGER"),
        (("exact_subprocess_environment_delta", "maximum_runner_call_site_changes"), 2, "ENV_EXACT_OLD_CALL"),
        (("exact_subprocess_environment_delta", "frozen_call_semantics", "extra_mapping", "HF_HUB_OFFLINE"), "0", "ENV_EXACT_OLD_CALL"),
        (("exact_subprocess_environment_delta", "amended_call_semantics", "positional_argument_count"), 1, "ENV_EXACT_NEW_CALL"),
        (("exact_subprocess_environment_delta", "amended_call_semantics", "extra_mapping"), {}, "ENV_EXACT_NEW_CALL"),
        (("exact_subprocess_environment_delta", "scrubber_mandatory_environment_contract", "inherited_environment_keys_allowed"), True, "ENV_EXACT_SCRUBBER"),
        (("exact_subprocess_environment_delta", "scrubber_mandatory_environment_contract", "exact_returned_mapping", "NO_PROXY"), "localhost", "ENV_EXACT_SCRUBBER"),
        (("exact_subprocess_environment_delta", "scrubber_mandatory_environment_contract", "exact_returned_mapping", "HF_HUB_OFFLINE"), "0", "ENV_EXACT_SCRUBBER"),
        (("exact_subprocess_environment_delta", "mandatory_post_call_assertions", "required_value_UTF8_hex", "DO_NOT_TRACK"), "30", "ENV_POST_CALL_ASSERTION"),
        (("exact_subprocess_environment_delta", "firewall_extra_whitelist_changed"), True, "ENV_SCOPE_EXPANSION"),
        (("exact_subprocess_environment_delta", "network_policy_changed"), True, "ENV_SCOPE_EXPANSION"),
        (("exact_subprocess_environment_delta", "subprocess_argv_changed"), True, "ENV_SCOPE_EXPANSION"),
        (("exact_subprocess_environment_delta", "parser_allowance_changed"), True, "ENV_SCOPE_EXPANSION"),
        (("preserved_public_canary_contract", "template_placeholder_relocation_unchanged"), False, "ENV_CANARY_CONTRACT"),
        (("preserved_sample_export_and_scientific_contract", "simulator_oracle_remains_only_evaluation_truth"), False, "ENV_SCIENTIFIC_CONTRACT"),
        (("preserved_sample_export_and_scientific_contract", "automatic_fallback_allowed"), True, "ENV_SCIENTIFIC_CONTRACT"),
        (("overlay_application_rule", "reject_if_the_scrubbed_mapping_is_mutated_after_return"), False, "ENV_APPLICATION_RULE"),
        (("new_immutable_output_namespace", "path"), "output/wrong", "ENV_OUTPUT_NAMESPACE"),
        (("new_immutable_output_namespace", "historical_output_paths_overwritten"), True, "ENV_OUTPUT_NAMESPACE"),
        (("authorization_boundary", "restricted_inference_authorized"), True, "ENV_AUTHORIZATION"),
        (("authorization_boundary", "scientific_outcome_authorized"), True, "ENV_AUTHORIZATION"),
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
        ("runner_sha256", "0" * 64),
        ("failure_code", "OTHER"),
        ("restricted_inference_run", True),
        ("scientific_outcome_run", True),
        ("automatic_calibration_fallback_allowed", True),
    ],
)
def test_failure_receipt_mutations_fail_closed(field, value) -> None:
    failure, _ = load(FAILURE_PATH)
    failure[field] = value
    issues = MODULE.validate_bound_inputs(
        MODULE.V13_RUNNER_SHA256, MODULE.V13_WORKER_SHA256, failure, MODULE.V13_FAILURE_SHA256,
        MODULE.TEMPLATE_SHA256, MODULE.CAF_SHA256, MODULE.NO_FALLBACK_SHA256, MODULE.FIREWALL_SHA256,
    )
    assert "ENV_FAILURE_IMMUTABILITY" in codes(issues)


def test_runner_extra_mapping_is_rejected() -> None:
    source = V14_RUNNER.read_text().replace(
        "scrubbed_subprocess_environment()",
        'scrubbed_subprocess_environment({"HF_HUB_OFFLINE": "1"})',
        1,
    )
    assert "ENV_RUNNER_ZERO_ARGUMENT_CALL" in codes(MODULE.validate_corrected_runner(source, MODULE.V14_RUNNER_SHA256))


def test_runner_environment_inheritance_is_rejected() -> None:
    source = V14_RUNNER.read_text() + "\n# os.environ\n"
    assert "ENV_RUNNER_INHERITANCE" in codes(MODULE.validate_corrected_runner(source, MODULE.V14_RUNNER_SHA256))


def test_future_receipt_contract() -> None:
    receipt = {
        "subprocess_environment_correction_amendment_sha256": MODULE.AMENDMENT_SHA256,
        "prior_v1_3_runner_sha256": MODULE.V13_RUNNER_SHA256,
        "prior_v1_3_failure_receipt_sha256": MODULE.V13_FAILURE_SHA256,
        "worker_sha256": MODULE.V13_WORKER_SHA256,
        "scrubber_called_with_extra_mapping": False,
        "mandatory_offline_defaults_asserted_before_launch": True,
        "scrubbed_environment_mutated_after_return": False,
        "parser_corrections_spent_before_environment_canary": 0,
        "output_namespace": MODULE.NEW_NAMESPACE,
        "historical_output_overwritten": False,
        "canonical_output_overwritten": False,
        "restricted_content_accessed_before_activation": False,
        "restricted_inference_run": False,
        "scientific_outcome_run": False,
    }
    assert MODULE.validate_receipt(receipt) == []
    receipt["scrubber_called_with_extra_mapping"] = True
    assert "ENV_RECEIPT_CONTRACT" in codes(MODULE.validate_receipt(receipt))


def test_validator_has_no_execution_or_restricted_discovery_path() -> None:
    source = SCRIPT.read_text()
    for forbidden in ("subprocess", "quarantine", "keychain", "requests", "urlopen", "model.generate"):
        if forbidden == "subprocess":
            # The validator names the contract but never imports or invokes subprocess.
            assert "import subprocess" not in source and "subprocess.run" not in source and "subprocess.Popen" not in source
        else:
            assert forbidden not in source.lower()
