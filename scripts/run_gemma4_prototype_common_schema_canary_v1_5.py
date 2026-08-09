#!/usr/bin/env python3
"""Additive v1.5 immutable preflight-seal delegation canary runner."""

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
V14_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_4.py"
V14_COORDINATOR = ROOT / "scripts/nursery_gemma_substitution_calibration_v1_4.py"
WORKER = ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py"
DELEGATION_AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/frozen_preflight_delegation_recursion_amendment_v1_5.json"
V14_FAILURE = ROOT / "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/public_common_schema_canary_failure_v1_4.json"
OUTPUT_ROOT = ROOT / "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5"
OUTPUT = OUTPUT_ROOT / "public_common_schema_canary_receipt_v1_5.json"
FAILURE = OUTPUT_ROOT / "public_common_schema_canary_failure_v1_5.json"
RUNTIME_LOCK = OUTPUT_ROOT / "runtime_lock_receipt_v1_5.json"
ACTIVATION = OUTPUT_ROOT / "full_activation_receipt_v1_5.json"

V14_RUNNER_SHA256 = "76d040b88cf1f3eb205524bc82839c1851d2ab293b1f7bb633358fe55763204c"
V14_COORDINATOR_SHA256 = "7c62c302205c505d9aa01c6b5e9894cdc1b9ac3d9be335b6066362773b0448a2"
V14_FAILURE_SHA256 = "db0b1b9431d1e1eaa602d5e09f6ce7505b6bdc13adcfd492341ee7bd3fae1f2a"
WORKER_SHA256 = "c78fa8e9c0f995ed611f7fd8f7d2076f03a73d370574d3c92cf5e107f236c38f"
DELEGATION_AMENDMENT_SHA256 = "16049ce3bcbeb9f3ae91c28695c7dafb10c9a013ff64b301d977b96f9943429a"
CANARY_SCHEMA = "nursery-gemma4-e4b-prototype-common-schema-public-canary-preflight-delegation-v1.5"
RUNTIME_SCHEMA = "nursery-gemma4-e4b-prototype-runtime-lock-preflight-delegation-v1.5"
ACTIVATION_SCHEMA = "nursery-gemma4-e4b-prototype-full-activation-receipt-preflight-delegation-v1.5"
FAILURE_SCHEMA = "nursery-gemma4-public-canary-preflight-delegation-failure-v1.5"
MAX_PIPE_BYTES = 2 * 1024 * 1024


class DelegationCanaryError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DelegationCanaryError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v14 = _load_module(V14_RUNNER, "nursery_gemma_delegation_v14_runner")
worker = v14.worker


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
        raise DelegationCanaryError("E_PUBLIC_ARTIFACT") from exc
    if not isinstance(value, Mapping):
        raise DelegationCanaryError("E_PUBLIC_ARTIFACT")
    return value


def _delegation_common_receipt() -> dict[str, Any]:
    value = dict(v14._environment_common_receipt())
    value.update({
        "output_namespace": "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5",
        "preflight_delegation_recursion_amendment_path": "docs/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/frozen_preflight_delegation_recursion_amendment_v1_5.json",
        "preflight_delegation_recursion_amendment_sha256": DELEGATION_AMENDMENT_SHA256,
        "prior_v1_4_runner_sha256": V14_RUNNER_SHA256,
        "prior_v1_4_coordinator_sha256": V14_COORDINATOR_SHA256,
        "prior_v1_4_failure_receipt_sha256": V14_FAILURE_SHA256,
        "preflight_seal_precompute_count": 1,
        "preflight_seal_projection_pure": True,
        "nested_amendment_preflight_recomputed": False,
        "historical_preflight_callables_restored": True,
        "parser_corrections_spent_before_delegation_canary": 0,
    })
    return value


def _validate_delegation_amendment() -> Mapping[str, Any]:
    if (
        _sha256_file(V14_RUNNER) != V14_RUNNER_SHA256
        or _sha256_file(V14_COORDINATOR) != V14_COORDINATOR_SHA256
        or _sha256_file(V14_FAILURE) != V14_FAILURE_SHA256
        or _sha256_file(WORKER) != WORKER_SHA256
        or _sha256_file(DELEGATION_AMENDMENT) != DELEGATION_AMENDMENT_SHA256
    ):
        raise DelegationCanaryError("E_DELEGATION_BINDING")
    v14._validate_environment_amendment()
    amendment = _read_json(DELEGATION_AMENDMENT)
    failure = _read_json(V14_FAILURE)
    if (
        amendment.get("schema_version") != "nursery-gemma4-public-canary-preflight-delegation-recursion-amendment-v1.5"
        or amendment.get("status") != "FROZEN_PRE_CANARY_PREFLIGHT_DELEGATION_ONLY"
        or amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_IMMUTABLE_SEAL_PROJECTION_OVERLAY"
        or amendment.get("public_canary_run") is not False
        or amendment.get("fixture_created") is not False
        or amendment.get("model_loaded_by_failed_v1_4_attempt") is not False
        or amendment.get("model_generation_called") is not False
        or amendment.get("restricted_content_accessed") is not False
        or amendment.get("restricted_inference_run") is not False
        or amendment.get("scientific_endpoint_opened") is not False
        or amendment.get("scientific_outcome_run") is not False
        or failure.get("failure_code") != "E_SANDBOXED_CANARY"
        or failure.get("restricted_payload_accessed") is not False
        or failure.get("restricted_inference_run") is not False
        or failure.get("raw_model_output_exported") is not False
    ):
        raise DelegationCanaryError("E_DELEGATION_AMENDMENT")
    forbidden = amendment.get("insufficient_correction_explicitly_forbidden", {})
    delta = amendment.get("exact_preflight_delegation_delta", {})
    projection = delta.get("pure_projection_callable_contract", {}) if isinstance(delta, Mapping) else {}
    if (
        not isinstance(forbidden, Mapping)
        or not forbidden
        or any(value is not True for value in forbidden.values())
        or delta.get("maximum_scientific_or_content_changes") != 0
        or delta.get("precompute_count_per_sandboxed_execution") != 1
        or projection.get("module_global_reads") is not False
        or projection.get("filesystem_reads") is not False
        or projection.get("environment_reads") is not False
        or projection.get("network_access") is not False
        or projection.get("content_or_model_reads") is not False
        or projection.get("accepted_argument_count") != 1
        or projection.get("parser_mode_comparison") != "EXACT_TYPE_AND_UTF8_BYTE_EQUALITY"
        or projection.get("parser_mismatch_failure_symbol") != "E_PARSER_MODE"
        or delta.get("direct_v1_5_mutated_module_targets") != ["v13._public_preflight"]
        or delta.get("v12_projection_propagation_is_only_via_bound_v1_3_executor") is not True
        or delta.get("historical_preflight_source_changed") is not False
        or delta.get("historical_worker_or_fixture_binding_policy_changed") is not False
        or delta.get("subprocess_environment_contract_changed") is not False
        or delta.get("semantic_fields_changed") is not False
        or delta.get("model_runtime_or_artifact_fields_changed") is not False
        or delta.get("parser_allowance_changed") is not False
        or delta.get("does_not_spend_outer_fence_parser_correction") is not True
    ):
        raise DelegationCanaryError("E_DELEGATION_DELTA")
    if any(path.exists() for path in (v14.OUTPUT, v14.RUNTIME_LOCK, v14.ACTIVATION)):
        raise DelegationCanaryError("E_V14_SUCCESS_OUTPUT_PRESENT")
    return amendment


def _validate_correction_state(parser_mode: str) -> None:
    if parser_mode == "PRIMARY":
        if OUTPUT_ROOT.exists():
            raise DelegationCanaryError("E_NEW_NAMESPACE_EXISTS")
        return
    if parser_mode != "ONE_OUTER_FENCE" or not OUTPUT_ROOT.is_dir():
        raise DelegationCanaryError("E_PARSER_CORRECTION_STATE")
    members = list(OUTPUT_ROOT.iterdir())
    failure = _read_json(FAILURE) if members == [FAILURE] and FAILURE.is_file() else {}
    if (
        failure.get("failure_code") != "E_SCHEMA_OUTPUT"
        or failure.get("parser_mode") != "PRIMARY"
        or failure.get("parser_correction_eligible") is not True
        or failure.get("preflight_delegation_recursion_amendment_sha256") != DELEGATION_AMENDMENT_SHA256
    ):
        raise DelegationCanaryError("E_PARSER_CORRECTION_STATE")


def _v14_preflight_projection(parser_mode: str) -> dict[str, Any]:
    v14._validate_environment_amendment()
    return {
        "schema_version": "nursery-gemma4-public-canary-subprocess-environment-seal-v1.4",
        "runner_sha256": V14_RUNNER_SHA256,
        "worker_sha256": WORKER_SHA256,
        "v1_3_runner_sha256": v14.V13_RUNNER_SHA256,
        "v1_3_failure_receipt_sha256": v14.V13_FAILURE_SHA256,
        "subprocess_environment_correction_amendment_sha256": v14.ENVIRONMENT_AMENDMENT_SHA256,
        "subprocess_environment_contract_sha256": v14.ENVIRONMENT_CONTRACT_SHA256,
        "template_placeholder_order_amendment_sha256": v14.v13.TEMPLATE_AMENDMENT_SHA256,
        "caf_transport_amendment_sha256": v14.v13.v12.CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": v14.v13.v12.NO_FALLBACK_OVERRIDE_SHA256,
        "template_order_correction_digest": worker.template_order_correction_digest(),
        "parser_mode": parser_mode,
        "v1_3_preflight": v14._v13_public_preflight(parser_mode),
    }


def _public_preflight(parser_mode: str) -> dict[str, Any]:
    _validate_delegation_amendment()
    _validate_correction_state(parser_mode)
    return {
        "schema_version": "nursery-gemma4-public-canary-preflight-delegation-seal-v1.5",
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_sha256": WORKER_SHA256,
        "v1_4_runner_sha256": V14_RUNNER_SHA256,
        "v1_4_coordinator_sha256": V14_COORDINATOR_SHA256,
        "v1_4_failure_receipt_sha256": V14_FAILURE_SHA256,
        "preflight_delegation_recursion_amendment_sha256": DELEGATION_AMENDMENT_SHA256,
        "subprocess_environment_correction_amendment_sha256": v14.ENVIRONMENT_AMENDMENT_SHA256,
        "subprocess_environment_contract_sha256": v14.ENVIRONMENT_CONTRACT_SHA256,
        "template_placeholder_order_amendment_sha256": v14.v13.TEMPLATE_AMENDMENT_SHA256,
        "caf_transport_amendment_sha256": v14.v13.v12.CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": v14.v13.v12.NO_FALLBACK_OVERRIDE_SHA256,
        "template_order_correction_digest": worker.template_order_correction_digest(),
        "parser_mode": parser_mode,
        "v1_4_preflight": _v14_preflight_projection(parser_mode),
    }


def _v14_canary_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(receipt)
    for key in _delegation_common_receipt():
        if key not in v14._environment_common_receipt():
            projected.pop(key, None)
    projected.update({
        "schema_version": v14.CANARY_SCHEMA,
        "status": "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS",
        "decision": "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS",
        "output_namespace": "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4",
        "runner_sha256": V14_RUNNER_SHA256,
    })
    return projected


def _validate_delegation_canary(receipt: Mapping[str, Any]) -> None:
    if (
        receipt.get("schema_version") != CANARY_SCHEMA
        or receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_PREFLIGHT_DELEGATION_PASS"
        or receipt.get("decision") != "PUBLIC_COMMON_SCHEMA_CANARY_PREFLIGHT_DELEGATION_PASS"
        or receipt.get("preflight_delegation_recursion_amendment_sha256") != DELEGATION_AMENDMENT_SHA256
        or receipt.get("prior_v1_4_runner_sha256") != V14_RUNNER_SHA256
        or receipt.get("prior_v1_4_coordinator_sha256") != V14_COORDINATOR_SHA256
        or receipt.get("prior_v1_4_failure_receipt_sha256") != V14_FAILURE_SHA256
        or receipt.get("preflight_seal_precompute_count") != 1
        or receipt.get("preflight_seal_projection_pure") is not True
        or receipt.get("nested_amendment_preflight_recomputed") is not False
        or receipt.get("historical_preflight_callables_restored") is not True
        or receipt.get("parser_corrections_spent_before_delegation_canary") != 0
    ):
        raise DelegationCanaryError("E_DELEGATION_CANARY_RECEIPT")
    v14._validate_environment_canary(_v14_canary_projection(receipt))


def sandboxed_execute(seal: Any) -> dict[str, Any]:
    if not isinstance(seal, Mapping) or not isinstance(seal.get("parser_mode"), str):
        raise DelegationCanaryError("E_SEAL")
    parser_mode = seal["parser_mode"]
    expected_bytes = _canonical(_public_preflight(parser_mode))
    expected_sha256 = _sha256_bytes(expected_bytes)
    if _canonical(seal) != expected_bytes:
        raise DelegationCanaryError("E_SEAL")

    def pure_projection(candidate_mode: str) -> Mapping[str, Any]:
        if not isinstance(candidate_mode, str) or candidate_mode.encode("utf-8") != parser_mode.encode("utf-8"):
            raise DelegationCanaryError("E_PARSER_MODE")
        projected = json.loads(expected_bytes)
        if _canonical(projected) != expected_bytes or _sha256_bytes(_canonical(projected)) != expected_sha256:
            raise DelegationCanaryError("E_SEAL")
        return projected

    v13 = v14.v13
    v12 = v13.v12
    saved_v13_preflight = v13._public_preflight
    saved_v12_preflight = v12._public_preflight
    try:
        v13._public_preflight = pure_projection
        nested = v13.sandboxed_execute(seal)
    finally:
        v13._public_preflight = saved_v13_preflight
    if v13._public_preflight is not saved_v13_preflight or v12._public_preflight is not saved_v12_preflight:
        raise DelegationCanaryError("E_PREFLIGHT_RESTORATION")

    runtime = dict(nested["runtime_lock"])
    runtime.update(v14._environment_common_receipt())
    runtime.update({"schema_version": v14.RUNTIME_SCHEMA, "runner_sha256": V14_RUNNER_SHA256, "worker_adapter_sha256": WORKER_SHA256})
    receipt = dict(nested["receipt"])
    v14_pass = receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS"
    v14_status = "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS" if v14_pass else "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_REVISE"
    receipt.update(v14._environment_common_receipt())
    receipt.update({
        "schema_version": v14.CANARY_SCHEMA,
        "status": v14_status,
        "decision": v14_status,
        "runner_sha256": V14_RUNNER_SHA256,
        "worker_adapter_sha256": WORKER_SHA256,
        "runtime_lock_sha256": _sha256_bytes(_canonical(runtime)),
    })
    runtime.update(_delegation_common_receipt())
    runtime.update({"schema_version": RUNTIME_SCHEMA, "runner_sha256": _sha256_file(SCRIPT), "worker_adapter_sha256": WORKER_SHA256})
    passed = receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_SUBPROCESS_ENVIRONMENT_PASS"
    status = "PUBLIC_COMMON_SCHEMA_CANARY_PREFLIGHT_DELEGATION_PASS" if passed else "PUBLIC_COMMON_SCHEMA_CANARY_PREFLIGHT_DELEGATION_REVISE"
    receipt.update(_delegation_common_receipt())
    receipt.update({
        "schema_version": CANARY_SCHEMA,
        "status": status,
        "decision": status,
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_adapter_sha256": WORKER_SHA256,
        "runtime_lock_sha256": _sha256_bytes(_canonical(runtime)),
    })
    return {"runtime_lock": runtime, "receipt": receipt}


def _v14_activation_projection(activation: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(activation)
    projected.pop("preflight_delegation_binding", None)
    for key in _delegation_common_receipt():
        if key not in v14._environment_common_receipt():
            projected.pop(key, None)
    projected.update({
        "schema_version": v14.ACTIVATION_SCHEMA,
        "status": "PROTOTYPE_INSTRUMENT_ACTIVATED_SUBPROCESS_ENVIRONMENT_RESTRICTED_INFERENCE_NOT_RUN",
        "output_namespace": "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4",
    })
    immutable = dict(projected["immutable_bindings"])
    immutable.pop("preflight_delegation_recursion_amendment", None)
    projected["immutable_bindings"] = immutable
    binding = dict(projected["public_canary_binding"])
    binding["receipt_schema_version"] = v14.CANARY_SCHEMA
    binding["receipt_path"] = "output/nursery_program_convergence_v1/gemma4_subprocess_environment_v1_4/public_common_schema_canary_receipt_v1_4.json"
    projected["public_canary_binding"] = binding
    return projected


def _validate_delegation_activation(activation: Mapping[str, Any], canary_sha256: str) -> None:
    binding = activation.get("preflight_delegation_binding", {})
    immutable = activation.get("immutable_bindings", {})
    canary = activation.get("public_canary_binding", {})
    if (
        activation.get("schema_version") != ACTIVATION_SCHEMA
        or activation.get("status") != "PROTOTYPE_INSTRUMENT_ACTIVATED_PREFLIGHT_DELEGATION_RESTRICTED_INFERENCE_NOT_RUN"
        or not isinstance(binding, Mapping)
        or binding.get("sha256") != DELEGATION_AMENDMENT_SHA256
        or binding.get("prior_v1_4_runner_sha256") != V14_RUNNER_SHA256
        or binding.get("prior_v1_4_failure_receipt_sha256") != V14_FAILURE_SHA256
        or binding.get("precompute_count_per_sandboxed_execution") != 1
        or binding.get("nested_amendment_preflight_recomputed") is not False
        or binding.get("historical_preflight_callables_restored") is not True
        or binding.get("parser_correction_spent") is not False
        or not isinstance(immutable, Mapping)
        or immutable.get("preflight_delegation_recursion_amendment", {}).get("sha256") != DELEGATION_AMENDMENT_SHA256
        or not isinstance(canary, Mapping)
        or canary.get("receipt_schema_version") != CANARY_SCHEMA
        or canary.get("receipt_path") != "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/public_common_schema_canary_receipt_v1_5.json"
        or canary.get("receipt_sha256") != canary_sha256
    ):
        raise DelegationCanaryError("E_DELEGATION_ACTIVATION")
    v14._validate_environment_activation(_v14_activation_projection(activation), canary_sha256)


def _build_activation_receipt(receipt: Mapping[str, Any], runtime: Mapping[str, Any], canary_bytes: bytes, runtime_bytes: bytes, resource: Mapping[str, bool]) -> dict[str, Any]:
    _validate_delegation_canary(receipt)
    activation = dict(v14._build_activation_receipt(_v14_canary_projection(receipt), runtime, canary_bytes, runtime_bytes, resource))
    activation.update(_delegation_common_receipt())
    activation["schema_version"] = ACTIVATION_SCHEMA
    activation["status"] = "PROTOTYPE_INSTRUMENT_ACTIVATED_PREFLIGHT_DELEGATION_RESTRICTED_INFERENCE_NOT_RUN"
    immutable = dict(activation["immutable_bindings"])
    immutable["preflight_delegation_recursion_amendment"] = {
        "path": "docs/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/frozen_preflight_delegation_recursion_amendment_v1_5.json",
        "sha256": DELEGATION_AMENDMENT_SHA256,
        "status": "FROZEN_PRE_CANARY_PREFLIGHT_DELEGATION_ONLY",
    }
    activation["immutable_bindings"] = immutable
    canary = dict(activation["public_canary_binding"])
    canary["receipt_schema_version"] = CANARY_SCHEMA
    canary["receipt_path"] = "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/public_common_schema_canary_receipt_v1_5.json"
    activation["public_canary_binding"] = canary
    activation["preflight_delegation_binding"] = {
        "path": "docs/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/frozen_preflight_delegation_recursion_amendment_v1_5.json",
        "sha256": DELEGATION_AMENDMENT_SHA256,
        "prior_v1_4_runner_sha256": V14_RUNNER_SHA256,
        "prior_v1_4_failure_receipt_sha256": V14_FAILURE_SHA256,
        "precompute_count_per_sandboxed_execution": 1,
        "projection_state_source": "IMMUTABLE_CANONICAL_SEAL_BYTES_SHA256_AND_EXACT_PARSER_MODE",
        "nested_amendment_preflight_recomputed": False,
        "historical_preflight_callables_restored": True,
        "parser_correction_spent": False,
    }
    _validate_delegation_activation(activation, _sha256_bytes(canary_bytes))
    return activation


def _ensure_output_root() -> None:
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(mode=0o700)
    metadata = OUTPUT_ROOT.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise DelegationCanaryError("E_OUTPUT_PRIVACY")


def _atomic_once(path: Path, payload: bytes) -> None:
    _ensure_output_root()
    v14.v13.v12.legacy._atomic_once(path, payload)


def _read_fd(descriptor: int) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(MAX_PIPE_BYTES + 1)
    if not payload or len(payload) > MAX_PIPE_BYTES:
        raise DelegationCanaryError("E_PIPE")
    return json.loads(payload)


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_PIPE_BYTES:
        raise DelegationCanaryError("E_PIPE")
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()


def public_execute(parser_mode: str) -> Mapping[str, Any]:
    seal = _public_preflight(parser_mode)
    legacy = v14.v13.v12.legacy
    backend = legacy.firewall.NetworkIsolationBackend.detect()
    legacy.firewall.verify_network_isolation(backend)
    resource = legacy._launch_resource_recheck()
    lock = legacy._exclusive_public_mps_lock()
    lock.__enter__()
    seal_read = seal_write = result_read = result_write = -1
    try:
        seal_read, seal_write = os.pipe()
        result_read, result_write = os.pipe()
        os.write(seal_write, _canonical(seal)); os.close(seal_write); seal_write = -1
        argv = [str(legacy.PYTHON), "-I", str(SCRIPT), "--sandboxed", "--seal-fd", str(seal_read), "--result-fd", str(result_write)]
        environment = dict(legacy.firewall.scrubbed_subprocess_environment())
        if environment != v14.EXPECTED_ENVIRONMENT or _sha256_bytes(_canonical(environment)) != v14.ENVIRONMENT_CONTRACT_SHA256 or any(environment.get(key, "").encode("utf-8") != b"1" for key in v14.BYTE_ASSERTED_KEYS):
            raise DelegationCanaryError("E_SUBPROCESS_ENV")
        process = subprocess.Popen(backend.command(argv), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment, pass_fds=(seal_read, result_write), close_fds=True)
        os.close(seal_read); seal_read = -1
        os.close(result_write); result_write = -1
        with os.fdopen(result_read, "rb") as handle:
            payload = handle.read(MAX_PIPE_BYTES + 1)
        result_read = -1
        if process.wait(timeout=30 * 60) != 0 or not payload or len(payload) > MAX_PIPE_BYTES:
            raise DelegationCanaryError("E_SANDBOXED_CANARY")
        result = json.loads(payload)
        if not isinstance(result, Mapping) or not isinstance(result.get("receipt"), Mapping) or not isinstance(result.get("runtime_lock"), Mapping):
            raise DelegationCanaryError("E_CANARY_RESULT")
        receipt, runtime = dict(result["receipt"]), dict(result["runtime_lock"])
        if runtime.get("schema_version") != RUNTIME_SCHEMA or runtime.get("preflight_delegation_recursion_amendment_sha256") != DELEGATION_AMENDMENT_SHA256 or runtime.get("worker_adapter_sha256") != WORKER_SHA256 or receipt.get("runtime_lock_sha256") != _sha256_bytes(_canonical(runtime)):
            raise DelegationCanaryError("E_RUNTIME_LOCK")
        if receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_PREFLIGHT_DELEGATION_PASS":
            return receipt
        _validate_delegation_canary(receipt)
        runtime_bytes = _json_document(runtime)
        receipt["runtime_lock_file_sha256"] = _sha256_bytes(runtime_bytes)
        canary_bytes = _json_document(receipt)
        activation = _build_activation_receipt(receipt, runtime, canary_bytes, runtime_bytes, resource)
        _atomic_once(RUNTIME_LOCK, runtime_bytes); _atomic_once(OUTPUT, canary_bytes); _atomic_once(ACTIVATION, _json_document(activation))
        return receipt
    finally:
        for descriptor in (seal_read, seal_write, result_read, result_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError): os.close(descriptor)
        lock.__exit__(None, None, None)


def _write_failure(code: str, parser_mode: str, parser_eligible: bool) -> None:
    if FAILURE.exists():
        return
    failure = {**_delegation_common_receipt(), "schema_version": FAILURE_SCHEMA, "status": "REVISE", "failure_code": code, "parser_mode": parser_mode, "parser_correction_eligible": parser_eligible, "runner_sha256": _sha256_file(SCRIPT), "worker_sha256": WORKER_SHA256, "restricted_payload_accessed": False, "restricted_inference_run": False, "raw_model_output_exported": False, "automatic_calibration_fallback_allowed": False, "scientific_endpoint_allowed": False}
    _atomic_once(FAILURE, _json_document(failure))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parser-correction", action="store_true"); parser.add_argument("--sandboxed", action="store_true"); parser.add_argument("--seal-fd", type=int); parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv)); parser_mode = "ONE_OUTER_FENCE" if args.parser_correction else "PRIMARY"
    try:
        if args.sandboxed:
            if args.seal_fd is None or args.result_fd is None or args.parser_correction: return 2
            _write_fd(args.result_fd, sandboxed_execute(_read_fd(args.seal_fd))); return 0
        if args.seal_fd is not None or args.result_fd is not None: return 2
        receipt = public_execute(parser_mode); passed = receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_PREFLIGHT_DELEGATION_PASS"
        if not passed: _write_failure("E_SCHEMA_OUTPUT", parser_mode, parser_mode == "PRIMARY")
        print("GEMMA4_PUBLIC_COMMON_SCHEMA_PREFLIGHT_DELEGATION_CANARY_PASS" if passed else "GEMMA4_PUBLIC_COMMON_SCHEMA_PREFLIGHT_DELEGATION_CANARY_REVISE"); return 0 if passed else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, (DelegationCanaryError, v14.EnvironmentCanaryError, RuntimeError)) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None: code = "E_INTERNAL"
        if args.sandboxed and args.result_fd is not None:
            with contextlib.suppress(Exception): _write_fd(args.result_fd, {"schema_version": FAILURE_SCHEMA, "failure_code": code})
        else:
            with contextlib.suppress(Exception): _write_failure(code, parser_mode, False)
            print("GEMMA4_PUBLIC_COMMON_SCHEMA_PREFLIGHT_DELEGATION_CANARY_REVISE")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
