#!/usr/bin/env python3
"""Additive v1.4 subprocess-environment public canary and activation runner."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
V13_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_3.py"
WORKER = ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py"
ENVIRONMENT_AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/frozen_subprocess_environment_correction_amendment_v1_4.json"
V13_FAILURE = ROOT / "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/public_common_schema_canary_failure_v1_3.json"
OUTPUT_ROOT = ROOT / "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4"
OUTPUT = OUTPUT_ROOT / "public_common_schema_canary_receipt_v1_4.json"
FAILURE = OUTPUT_ROOT / "public_common_schema_canary_failure_v1_4.json"
RUNTIME_LOCK = OUTPUT_ROOT / "runtime_lock_receipt_v1_4.json"
ACTIVATION = OUTPUT_ROOT / "full_activation_receipt_v1_4.json"

V13_RUNNER_SHA256 = "dcbdf8714f12a114806ef4c903ee1a7f95b65647f58273b47fd184fb1143cf44"
WORKER_SHA256 = "c78fa8e9c0f995ed611f7fd8f7d2076f03a73d370574d3c92cf5e107f236c38f"
V13_FAILURE_SHA256 = "e69abc985a7400d5305eab9f76217a4bdb9f60baaf78715a87f3abf63ca75492"
ENVIRONMENT_AMENDMENT_SHA256 = "ac05205256bc1e949c0a70a9dc865cebc2ea50beb73770319e816cec0910ceb1"
FIREWALL_SHA256 = "a10f8c2568d402259215afe18f6005ceb4b52431607b42b132a8db849ff32f7b"
ENVIRONMENT_CONTRACT_SHA256 = "bf67ff8d311d901d39d8f996339e33f62c72b2f87a0c4ff9d997dce069cdf822"
CANARY_SCHEMA = "nursery-gemma4-e4b-prototype-common-schema-public-canary-subprocess-environment-v1.4"
RUNTIME_SCHEMA = "nursery-gemma4-e4b-prototype-runtime-lock-subprocess-environment-v1.4"
ACTIVATION_SCHEMA = "nursery-gemma4-e4b-prototype-full-activation-receipt-subprocess-environment-v1.4"
FAILURE_SCHEMA = "nursery-gemma4-public-canary-subprocess-environment-failure-v1.4"
MAX_PIPE_BYTES = 2 * 1024 * 1024

EXPECTED_ENVIRONMENT = {
    "PATH": "/bin:/usr/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_DISABLED": "true",
    "DO_NOT_TRACK": "1",
    "NO_PROXY": "*",
    "no_proxy": "*",
}
BYTE_ASSERTED_KEYS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "DO_NOT_TRACK",
)


class EnvironmentCanaryError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EnvironmentCanaryError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v13 = _load_module(V13_RUNNER, "nursery_gemma_environment_v13_runner")
worker = v13.worker


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_document(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentCanaryError("E_PUBLIC_ARTIFACT") from exc
    if not isinstance(value, Mapping):
        raise EnvironmentCanaryError("E_PUBLIC_ARTIFACT")
    return value


def _environment_common_receipt() -> dict[str, Any]:
    base = dict(v13._template_common_receipt())
    base.update({
        "output_namespace": "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4",
        "subprocess_environment_correction_amendment_path": "docs/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/frozen_subprocess_environment_correction_amendment_v1_4.json",
        "subprocess_environment_correction_amendment_sha256": ENVIRONMENT_AMENDMENT_SHA256,
        "subprocess_environment_contract_sha256": ENVIRONMENT_CONTRACT_SHA256,
        "prior_v1_3_runner_sha256": V13_RUNNER_SHA256,
        "prior_v1_3_failure_receipt_sha256": V13_FAILURE_SHA256,
        "scrubber_called_with_extra_mapping": False,
        "mandatory_offline_defaults_asserted_before_launch": True,
        "scrubbed_environment_mutated_after_return": False,
        "parser_corrections_spent_before_environment_canary": 0,
    })
    return base


def _validate_environment_amendment() -> Mapping[str, Any]:
    if (
        _sha256_file(V13_RUNNER) != V13_RUNNER_SHA256
        or _sha256_file(WORKER) != WORKER_SHA256
        or _sha256_file(V13_FAILURE) != V13_FAILURE_SHA256
        or _sha256_file(ENVIRONMENT_AMENDMENT) != ENVIRONMENT_AMENDMENT_SHA256
        or _sha256_file(v13.v12.legacy.FIREWALL) != FIREWALL_SHA256
        or _sha256_bytes(_canonical(EXPECTED_ENVIRONMENT)) != ENVIRONMENT_CONTRACT_SHA256
    ):
        raise EnvironmentCanaryError("E_ENVIRONMENT_BINDING")
    v13._validate_template_amendment()
    amendment = _read_json(ENVIRONMENT_AMENDMENT)
    failure = _read_json(V13_FAILURE)
    if (
        amendment.get("schema_version") != "nursery-gemma4-public-canary-subprocess-environment-correction-amendment-v1.4"
        or amendment.get("status") != "FROZEN_PRE_CANARY_SUBPROCESS_ENVIRONMENT_ONLY"
        or amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_SUBPROCESS_ENVIRONMENT_CALL_CORRECTION"
        or amendment.get("public_canary_run") is not False
        or amendment.get("child_process_launched") is not False
        or amendment.get("restricted_content_accessed") is not False
        or amendment.get("restricted_inference_run") is not False
        or amendment.get("model_generation_called") is not False
        or amendment.get("scientific_outcome_run") is not False
        or failure.get("failure_code") != "E_SUBPROCESS_ENV"
        or failure.get("restricted_payload_accessed") is not False
        or failure.get("restricted_inference_run") is not False
        or failure.get("raw_model_output_exported") is not False
    ):
        raise EnvironmentCanaryError("E_ENVIRONMENT_AMENDMENT")
    delta = amendment.get("exact_subprocess_environment_delta", {})
    frozen = delta.get("frozen_call_semantics", {}) if isinstance(delta, Mapping) else {}
    amended = delta.get("amended_call_semantics", {}) if isinstance(delta, Mapping) else {}
    contract = delta.get("scrubber_mandatory_environment_contract", {}) if isinstance(delta, Mapping) else {}
    assertions = delta.get("mandatory_post_call_assertions", {}) if isinstance(delta, Mapping) else {}
    if (
        delta.get("maximum_runner_call_site_changes") != 1
        or frozen.get("positional_argument_count") != 1
        or frozen.get("extra_mapping") != {key: "1" for key in BYTE_ASSERTED_KEYS}
        or amended.get("positional_argument_count") != 0
        or amended.get("keyword_argument_count") != 0
        or amended.get("extra_mapping") is not None
        or amended.get("returned_mapping_materialized_for_subprocess") is not True
        or contract.get("inherited_environment_keys_allowed") is not False
        or contract.get("extra_key_count") != 0
        or contract.get("exact_returned_key_count") != len(EXPECTED_ENVIRONMENT)
        or contract.get("exact_returned_mapping") != EXPECTED_ENVIRONMENT
        or contract.get("canonical_JSON_sort_keys_compact_UTF8_sha256") != ENVIRONMENT_CONTRACT_SHA256
        or contract.get("credential_authorization_cookie_secret_or_nonblocked_proxy_keys_allowed") is not False
        or assertions.get("required_key_values") != {key: "1" for key in BYTE_ASSERTED_KEYS}
        or assertions.get("required_value_UTF8_hex") != {key: "31" for key in BYTE_ASSERTED_KEYS}
        or assertions.get("failure_symbol_on_missing_or_different_value") != "E_SUBPROCESS_ENV"
        or assertions.get("assert_before_subprocess_launch") is not True
        or delta.get("firewall_implementation_changed") is not False
        or delta.get("firewall_extra_whitelist_changed") is not False
        or delta.get("mandatory_environment_defaults_changed") is not False
        or delta.get("credential_or_proxy_inheritance_allowed") is not False
        or delta.get("network_policy_changed") is not False
        or delta.get("subprocess_argv_changed") is not False
        or delta.get("subprocess_cwd_changed") is not False
        or delta.get("subprocess_stdio_or_passed_FDs_changed") is not False
        or delta.get("semantic_fields_changed") is not False
        or delta.get("model_runtime_or_artifact_fields_changed") is not False
        or delta.get("scientific_fields_changed") is not False
        or delta.get("parser_allowance_changed") is not False
        or delta.get("does_not_spend_outer_fence_parser_correction") is not True
    ):
        raise EnvironmentCanaryError("E_ENVIRONMENT_DELTA")
    if any(path.exists() for path in (v13.OUTPUT, v13.RUNTIME_LOCK, v13.ACTIVATION)):
        raise EnvironmentCanaryError("E_V13_SUCCESS_OUTPUT_PRESENT")
    return amendment


def _validate_correction_state(parser_mode: str) -> None:
    if parser_mode == "PRIMARY":
        if OUTPUT_ROOT.exists():
            raise EnvironmentCanaryError("E_NEW_NAMESPACE_EXISTS")
        return
    if parser_mode != "ONE_OUTER_FENCE" or not OUTPUT_ROOT.is_dir():
        raise EnvironmentCanaryError("E_PARSER_CORRECTION_STATE")
    members = list(OUTPUT_ROOT.iterdir())
    failure = _read_json(FAILURE) if members == [FAILURE] and FAILURE.is_file() else {}
    if (
        failure.get("failure_code") != "E_SCHEMA_OUTPUT"
        or failure.get("parser_mode") != "PRIMARY"
        or failure.get("parser_correction_eligible") is not True
        or failure.get("subprocess_environment_correction_amendment_sha256") != ENVIRONMENT_AMENDMENT_SHA256
    ):
        raise EnvironmentCanaryError("E_PARSER_CORRECTION_STATE")


def _v13_public_preflight(parser_mode: str) -> Mapping[str, Any]:
    original = v13._validate_correction_state
    try:
        v13._validate_correction_state = lambda _mode: None
        return v13._public_preflight(parser_mode)
    finally:
        v13._validate_correction_state = original


def _public_preflight(parser_mode: str) -> dict[str, Any]:
    _validate_environment_amendment()
    _validate_correction_state(parser_mode)
    return {
        "schema_version": "nursery-gemma4-public-canary-subprocess-environment-seal-v1.4",
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_sha256": WORKER_SHA256,
        "v1_3_runner_sha256": V13_RUNNER_SHA256,
        "v1_3_failure_receipt_sha256": V13_FAILURE_SHA256,
        "subprocess_environment_correction_amendment_sha256": ENVIRONMENT_AMENDMENT_SHA256,
        "subprocess_environment_contract_sha256": ENVIRONMENT_CONTRACT_SHA256,
        "template_placeholder_order_amendment_sha256": v13.TEMPLATE_AMENDMENT_SHA256,
        "caf_transport_amendment_sha256": v13.v12.CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": v13.v12.NO_FALLBACK_OVERRIDE_SHA256,
        "template_order_correction_digest": worker.template_order_correction_digest(),
        "parser_mode": parser_mode,
        "v1_3_preflight": _v13_public_preflight(parser_mode),
    }


def _v13_canary_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(receipt)
    for key in _environment_common_receipt():
        if key not in v13._template_common_receipt():
            projected.pop(key, None)
    projected.update({
        "schema_version": v13.CANARY_SCHEMA,
        "status": "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS",
        "decision": "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS",
        "output_namespace": "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3",
        "runner_sha256": V13_RUNNER_SHA256,
    })
    return projected


def _validate_environment_canary(receipt: Mapping[str, Any]) -> None:
    if (
        receipt.get("schema_version") != CANARY_SCHEMA
        or receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS"
        or receipt.get("decision") != "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS"
        or receipt.get("subprocess_environment_correction_amendment_sha256") != ENVIRONMENT_AMENDMENT_SHA256
        or receipt.get("subprocess_environment_contract_sha256") != ENVIRONMENT_CONTRACT_SHA256
        or receipt.get("prior_v1_3_runner_sha256") != V13_RUNNER_SHA256
        or receipt.get("prior_v1_3_failure_receipt_sha256") != V13_FAILURE_SHA256
        or receipt.get("scrubber_called_with_extra_mapping") is not False
        or receipt.get("mandatory_offline_defaults_asserted_before_launch") is not True
        or receipt.get("scrubbed_environment_mutated_after_return") is not False
        or receipt.get("parser_corrections_spent_before_environment_canary") != 0
        or receipt.get("template_order_correction_digest") != worker.template_order_correction_digest()
    ):
        raise EnvironmentCanaryError("E_ENVIRONMENT_CANARY_RECEIPT")
    v13._validate_template_canary(_v13_canary_projection(receipt))


def sandboxed_execute(seal: Any) -> dict[str, Any]:
    if not isinstance(seal, Mapping) or dict(seal) != _public_preflight(str(seal.get("parser_mode", ""))):
        raise EnvironmentCanaryError("E_SEAL")
    original_preflight = v13._public_preflight
    try:
        v13._public_preflight = _public_preflight
        result = v13.sandboxed_execute(seal)
    finally:
        v13._public_preflight = original_preflight
    runtime = dict(result["runtime_lock"])
    runtime.update(_environment_common_receipt())
    runtime.update({
        "schema_version": RUNTIME_SCHEMA,
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_adapter_sha256": WORKER_SHA256,
    })
    receipt = dict(result["receipt"])
    passed = receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS"
    status = "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS" if passed else "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_REVISE"
    receipt.update(_environment_common_receipt())
    receipt.update({
        "schema_version": CANARY_SCHEMA,
        "status": status,
        "decision": status,
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_adapter_sha256": WORKER_SHA256,
        "runtime_lock_sha256": _sha256_bytes(_canonical(runtime)),
    })
    return {"runtime_lock": runtime, "receipt": receipt}


def _v13_activation_projection(activation: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(activation)
    projected.pop("subprocess_environment_correction_binding", None)
    for key in _environment_common_receipt():
        if key not in v13._template_common_receipt():
            projected.pop(key, None)
    projected.update({
        "schema_version": v13.ACTIVATION_SCHEMA,
        "status": "PROTOTYPE_INSTRUMENT_ACTIVATED_TEMPLATE_ORDER_RESTRICTED_INFERENCE_NOT_RUN",
        "output_namespace": "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3",
    })
    immutable = dict(projected["immutable_bindings"])
    immutable.pop("subprocess_environment_correction_amendment", None)
    projected["immutable_bindings"] = immutable
    canary = dict(projected["public_canary_binding"])
    canary["receipt_schema_version"] = v13.CANARY_SCHEMA
    canary["receipt_path"] = "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/public_common_schema_canary_receipt_v1_3.json"
    projected["public_canary_binding"] = canary
    return projected


def _validate_environment_activation(activation: Mapping[str, Any], canary_sha256: str) -> None:
    binding = activation.get("subprocess_environment_correction_binding", {})
    immutable = activation.get("immutable_bindings", {})
    canary = activation.get("public_canary_binding", {})
    if (
        activation.get("schema_version") != ACTIVATION_SCHEMA
        or activation.get("status") != "PROTOTYPE_INSTRUMENT_ACTIVATED_SUBPROCESS_ENVIRONMENT_RESTRICTED_INFERENCE_NOT_RUN"
        or not isinstance(binding, Mapping)
        or binding.get("sha256") != ENVIRONMENT_AMENDMENT_SHA256
        or binding.get("prior_v1_3_runner_sha256") != V13_RUNNER_SHA256
        or binding.get("prior_v1_3_failure_receipt_sha256") != V13_FAILURE_SHA256
        or binding.get("scrubber_positional_argument_count") != 0
        or binding.get("scrubber_keyword_argument_count") != 0
        or binding.get("scrubber_extra_mapping") is not None
        or binding.get("exact_returned_key_count") != len(EXPECTED_ENVIRONMENT)
        or binding.get("exact_returned_mapping_sha256") != ENVIRONMENT_CONTRACT_SHA256
        or binding.get("mandatory_offline_defaults_asserted_before_launch") is not True
        or binding.get("scrubbed_environment_mutated_after_return") is not False
        or binding.get("parser_correction_spent") is not False
        or not isinstance(immutable, Mapping)
        or immutable.get("subprocess_environment_correction_amendment", {}).get("sha256") != ENVIRONMENT_AMENDMENT_SHA256
        or not isinstance(canary, Mapping)
        or canary.get("receipt_schema_version") != CANARY_SCHEMA
        or canary.get("receipt_path") != "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/public_common_schema_canary_receipt_v1_4.json"
        or canary.get("receipt_sha256") != canary_sha256
    ):
        raise EnvironmentCanaryError("E_ENVIRONMENT_ACTIVATION")
    v13._validate_template_activation(_v13_activation_projection(activation), canary_sha256)


def _build_activation_receipt(
    receipt: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    canary_bytes: bytes,
    runtime_lock_bytes: bytes,
    resource_recheck: Mapping[str, bool],
) -> dict[str, Any]:
    _validate_environment_canary(receipt)
    activation = dict(v13._build_activation_receipt(
        _v13_canary_projection(receipt), runtime_lock, canary_bytes, runtime_lock_bytes, resource_recheck
    ))
    activation.update(_environment_common_receipt())
    activation["schema_version"] = ACTIVATION_SCHEMA
    activation["status"] = "PROTOTYPE_INSTRUMENT_ACTIVATED_SUBPROCESS_ENVIRONMENT_RESTRICTED_INFERENCE_NOT_RUN"
    immutable = dict(activation["immutable_bindings"])
    immutable["subprocess_environment_correction_amendment"] = {
        "path": "docs/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/frozen_subprocess_environment_correction_amendment_v1_4.json",
        "sha256": ENVIRONMENT_AMENDMENT_SHA256,
        "status": "FROZEN_PRE_CANARY_SUBPROCESS_ENVIRONMENT_ONLY",
    }
    activation["immutable_bindings"] = immutable
    canary = dict(activation["public_canary_binding"])
    canary["receipt_schema_version"] = CANARY_SCHEMA
    canary["receipt_path"] = "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/public_common_schema_canary_receipt_v1_4.json"
    activation["public_canary_binding"] = canary
    activation["subprocess_environment_correction_binding"] = {
        "path": "docs/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/frozen_subprocess_environment_correction_amendment_v1_4.json",
        "sha256": ENVIRONMENT_AMENDMENT_SHA256,
        "status": "FROZEN_PRE_CANARY_SUBPROCESS_ENVIRONMENT_ONLY",
        "prior_v1_3_runner_sha256": V13_RUNNER_SHA256,
        "prior_v1_3_failure_receipt_sha256": V13_FAILURE_SHA256,
        "worker_sha256": WORKER_SHA256,
        "firewall_sha256": FIREWALL_SHA256,
        "scrubber_positional_argument_count": 0,
        "scrubber_keyword_argument_count": 0,
        "scrubber_extra_mapping": None,
        "exact_returned_key_count": len(EXPECTED_ENVIRONMENT),
        "exact_returned_mapping_sha256": ENVIRONMENT_CONTRACT_SHA256,
        "mandatory_offline_defaults_asserted_before_launch": True,
        "scrubbed_environment_mutated_after_return": False,
        "parser_correction_spent": False,
    }
    _validate_environment_activation(activation, _sha256_bytes(canary_bytes))
    return activation


def _ensure_output_root() -> None:
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(mode=0o700)
    metadata = OUTPUT_ROOT.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise EnvironmentCanaryError("E_OUTPUT_PRIVACY")


def _atomic_once(path: Path, payload: bytes) -> None:
    _ensure_output_root()
    v13.v12.legacy._atomic_once(path, payload)


def _read_fd(descriptor: int) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(MAX_PIPE_BYTES + 1)
    if not payload or len(payload) > MAX_PIPE_BYTES:
        raise EnvironmentCanaryError("E_PIPE")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise EnvironmentCanaryError("E_PIPE") from exc


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_PIPE_BYTES:
        raise EnvironmentCanaryError("E_PIPE")
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()


def public_execute(parser_mode: str) -> Mapping[str, Any]:
    seal = _public_preflight(parser_mode)
    backend = v13.v12.legacy.firewall.NetworkIsolationBackend.detect()
    v13.v12.legacy.firewall.verify_network_isolation(backend)
    resource_recheck = v13.v12.legacy._launch_resource_recheck()
    mps_lock = v13.v12.legacy._exclusive_public_mps_lock()
    mps_lock.__enter__()
    seal_read = seal_write = result_read = result_write = -1
    try:
        seal_read, seal_write = os.pipe()
        result_read, result_write = os.pipe()
        os.write(seal_write, _canonical(seal))
        os.close(seal_write)
        seal_write = -1
        argv = [
            str(v13.v12.legacy.PYTHON), "-I", str(SCRIPT), "--sandboxed",
            "--seal-fd", str(seal_read), "--result-fd", str(result_write),
        ]
        environment = dict(v13.v12.legacy.firewall.scrubbed_subprocess_environment())
        if (
            environment != EXPECTED_ENVIRONMENT
            or _sha256_bytes(_canonical(environment)) != ENVIRONMENT_CONTRACT_SHA256
            or any(environment.get(key, "").encode("utf-8") != b"1" for key in BYTE_ASSERTED_KEYS)
        ):
            raise EnvironmentCanaryError("E_SUBPROCESS_ENV")
        process = subprocess.Popen(
            backend.command(argv),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            pass_fds=(seal_read, result_write),
            close_fds=True,
        )
        os.close(seal_read)
        seal_read = -1
        os.close(result_write)
        result_write = -1
        with os.fdopen(result_read, "rb") as handle:
            payload = handle.read(MAX_PIPE_BYTES + 1)
        result_read = -1
        returncode = process.wait(timeout=30 * 60)
        if returncode != 0 or not payload or len(payload) > MAX_PIPE_BYTES:
            raise EnvironmentCanaryError("E_SANDBOXED_CANARY")
        result = json.loads(payload)
        if not isinstance(result, Mapping) or not isinstance(result.get("receipt"), Mapping) or not isinstance(result.get("runtime_lock"), Mapping):
            raise EnvironmentCanaryError("E_CANARY_RESULT")
        receipt, runtime = dict(result["receipt"]), dict(result["runtime_lock"])
        if (
            runtime.get("schema_version") != RUNTIME_SCHEMA
            or runtime.get("subprocess_environment_correction_amendment_sha256") != ENVIRONMENT_AMENDMENT_SHA256
            or runtime.get("subprocess_environment_contract_sha256") != ENVIRONMENT_CONTRACT_SHA256
            or runtime.get("worker_adapter_sha256") != WORKER_SHA256
            or receipt.get("runtime_lock_sha256") != _sha256_bytes(_canonical(runtime))
        ):
            raise EnvironmentCanaryError("E_RUNTIME_LOCK")
        if receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS":
            return receipt
        _validate_environment_canary(receipt)
        runtime_bytes = _json_document(runtime)
        receipt["runtime_lock_file_sha256"] = _sha256_bytes(runtime_bytes)
        canary_bytes = _json_document(receipt)
        activation = _build_activation_receipt(receipt, runtime, canary_bytes, runtime_bytes, resource_recheck)
        _atomic_once(RUNTIME_LOCK, runtime_bytes)
        _atomic_once(OUTPUT, canary_bytes)
        _atomic_once(ACTIVATION, _json_document(activation))
        return receipt
    finally:
        for descriptor in (seal_read, seal_write, result_read, result_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        mps_lock.__exit__(None, None, None)


def _write_failure(code: str, parser_mode: str, parser_eligible: bool) -> None:
    if FAILURE.exists():
        return
    failure = {
        **_environment_common_receipt(),
        "schema_version": FAILURE_SCHEMA,
        "status": "REVISE",
        "failure_code": code,
        "parser_mode": parser_mode,
        "parser_correction_eligible": parser_eligible,
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_sha256": WORKER_SHA256,
        "restricted_payload_accessed": False,
        "restricted_inference_run": False,
        "raw_model_output_exported": False,
        "automatic_calibration_fallback_allowed": False,
        "scientific_endpoint_allowed": False,
    }
    _atomic_once(FAILURE, _json_document(failure))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parser-correction", action="store_true")
    parser.add_argument("--sandboxed", action="store_true")
    parser.add_argument("--seal-fd", type=int)
    parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    parser_mode = "ONE_OUTER_FENCE" if args.parser_correction else "PRIMARY"
    try:
        if args.sandboxed:
            if args.seal_fd is None or args.result_fd is None or args.parser_correction:
                return 2
            _write_fd(args.result_fd, sandboxed_execute(_read_fd(args.seal_fd)))
            return 0
        if args.seal_fd is not None or args.result_fd is not None:
            return 2
        receipt = public_execute(parser_mode)
        passed = receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS"
        if not passed:
            _write_failure("E_SCHEMA_OUTPUT", parser_mode, parser_mode == "PRIMARY")
        print("GEMMA4_PUBLIC_COMMON_SCHEMA_SUBPROCESS_ENVIRONMENT_CANARY_PASS" if passed else "GEMMA4_PUBLIC_COMMON_SCHEMA_SUBPROCESS_ENVIRONMENT_CANARY_REVISE")
        return 0 if passed else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, (EnvironmentCanaryError, v13.TemplateCanaryError, v13.v12.CafCanaryError, v13.v12.legacy.CanaryError, RuntimeError)) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None:
            code = "E_INTERNAL"
        if args.sandboxed and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd(args.result_fd, {"schema_version": FAILURE_SCHEMA, "failure_code": code})
        else:
            with contextlib.suppress(Exception):
                _write_failure(code, parser_mode, False)
            print("GEMMA4_PUBLIC_COMMON_SCHEMA_SUBPROCESS_ENVIRONMENT_CANARY_REVISE")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
