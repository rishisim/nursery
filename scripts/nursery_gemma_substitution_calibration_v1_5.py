#!/usr/bin/env python3
"""Additive v1.5 coordinator gated by immutable preflight delegation."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
V14_COORDINATOR = ROOT / "scripts/nursery_gemma_substitution_calibration_v1_4.py"
RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_5.py"
WORKER = ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py"
PUBLIC_RECEIPT = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
PUBLIC_FAILURE = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_preflight_delegation_failure_v1_5.json"
V14_COORDINATOR_SHA256 = "7c62c302205c505d9aa01c6b5e9894cdc1b9ac3d9be335b6066362773b0448a2"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("E_MODULE")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


v14 = _load_module(V14_COORDINATOR, "nursery_gemma_delegation_v14_coordinator")
runner = _load_module(RUNNER, "nursery_gemma_delegation_public_runner")
worker = runner.worker
legacy = v14.legacy
GemmaSubstitutionError = legacy.GemmaSubstitutionError
ACTIVATION = runner.ACTIVATION
PUBLIC_CANARY = runner.OUTPUT


def _public_gemma_seal() -> dict[str, Any]:
    legacy._historical_immutability_check()
    if legacy._sha256_file(V14_COORDINATOR) != V14_COORDINATOR_SHA256:
        raise GemmaSubstitutionError("E_V14_COORDINATOR")
    runner._validate_delegation_amendment()
    if not ACTIVATION.is_file() or not PUBLIC_CANARY.is_file():
        raise GemmaSubstitutionError("E_DELEGATION_ACTIVATION_MISSING")
    canary = legacy._read_json(PUBLIC_CANARY); activation = legacy._read_json(ACTIVATION)
    canary_sha = legacy._sha256_file(PUBLIC_CANARY); activation_sha = legacy._sha256_file(ACTIVATION)
    try:
        runner._validate_delegation_canary(canary); runner._validate_delegation_activation(activation, canary_sha)
    except runner.DelegationCanaryError as exc:
        raise GemmaSubstitutionError("E_DELEGATION_CHAIN") from exc
    amendment = legacy._read_json(legacy.AMENDMENT); prior_stop = legacy._read_json(legacy.PRIOR_STOP)
    provenance = legacy._read_json(legacy.PROVENANCE_ACTIVATION); contract = legacy._read_json(legacy.CANARY_CONTRACT)
    erratum = legacy._read_json(legacy.RUNTIME_ERRATUM); override = legacy._read_json(v14.v13.v12.NO_FALLBACK_OVERRIDE)
    amendment_sha = legacy._sha256_file(legacy.AMENDMENT)
    if (
        amendment_sha != legacy.EXPECTED_OPAQUE_AMENDMENT_SHA256
        or legacy._sha256_file(legacy.PROVENANCE_ACTIVATION) != legacy.EXPECTED_PROVENANCE_ACTIVATION_SHA256
        or legacy._sha256_file(legacy.RESOURCE_RECEIPT) != legacy.EXPECTED_RESOURCE_RECEIPT_SHA256
        or legacy._sha256_file(legacy.CANARY_CONTRACT) != legacy.EXPECTED_CANARY_CONTRACT_SHA256
        or legacy._sha256_file(legacy.ACTIVATION_SCHEMA) != legacy.EXPECTED_ACTIVATION_SCHEMA_SHA256
        or legacy._sha256_file(legacy.RUNTIME_ERRATUM) != legacy.EXPECTED_RUNTIME_ERRATUM_SHA256
        or legacy.replacement_validator.validate_amendment(amendment, prior_stop, legacy._sha256_file(legacy.PRIOR_STOP))
        or legacy.replacement_validator.validate_provenance_activation(provenance, amendment_sha)
        or legacy.replacement_validator.validate_canary_contract(contract, amendment_sha)
        or legacy.replacement_validator.validate_runtime_erratum(erratum, legacy.EXPECTED_RUNTIME_ERRATUM_SHA256, legacy.EXPECTED_ACTIVATION_SCHEMA_SHA256)
        or legacy.replacement_validator.validate_no_fallback_override(override, runner.v14.v13.v12.NO_FALLBACK_OVERRIDE_SHA256)
    ):
        raise GemmaSubstitutionError("E_PUBLIC_BINDING")
    if not legacy.GEMMA_PYTHON.is_file() or not legacy.FFMPEG.is_file() or not legacy.GEMMA_MODEL.is_dir():
        raise GemmaSubstitutionError("E_RUNTIME")
    manifest, artifact_bytes, artifact_files = legacy._tree_manifest(legacy.GEMMA_MODEL)
    artifact = activation.get("artifact_binding", {}); runtime = activation.get("runtime_binding", {})
    qwen = activation.get("qwen3_read_only_binding", {}); canary_binding = activation.get("public_canary_binding", {})
    if not all(isinstance(value, Mapping) for value in (artifact, runtime, qwen, canary_binding)):
        raise GemmaSubstitutionError("E_DELEGATION_CHAIN")
    if (
        artifact.get("snapshot_manifest_sha256") != manifest or artifact.get("snapshot_bytes") != artifact_bytes or artifact.get("snapshot_file_count") != artifact_files
        or runtime.get("canary_adapter_sha256") != legacy._sha256_file(WORKER)
        or runtime.get("network_denial_profile_sha256") != legacy._sha256_file(legacy.FIREWALL)
        or canary.get("prompt_sha256") != worker.prompt_digest() or canary.get("exact_output_schema_sha256") != worker.exact_schema_digest()
        or canary.get("preflight_delegation_recursion_amendment_sha256") != runner.DELEGATION_AMENDMENT_SHA256
        or canary.get("subprocess_environment_correction_amendment_sha256") != runner.v14.ENVIRONMENT_AMENDMENT_SHA256
        or canary.get("template_placeholder_order_amendment_sha256") != runner.v14.v13.TEMPLATE_AMENDMENT_SHA256
        or canary_binding.get("parser_correction_count") != canary.get("corrections_used")
        or qwen.get("restricted_hypotheses_recomputed") is not False or qwen.get("restricted_digest_exported") is not False or qwen.get("third_visual_model") is not False
    ):
        raise GemmaSubstitutionError("E_MODEL_SEAL")
    return {
        "schema_version": "nursery-childlens-gemma4-substitution-preflight-delegation-seal-v1.5",
        "script_sha256": legacy._sha256_file(SCRIPT), "v14_coordinator_sha256": V14_COORDINATOR_SHA256,
        "runner_sha256": legacy._sha256_file(RUNNER), "worker_sha256": legacy._sha256_file(WORKER),
        "amendment_sha256": amendment_sha, "caf_transport_amendment_sha256": runner.v14.v13.v12.CAF_AMENDMENT_SHA256,
        "template_placeholder_order_amendment_sha256": runner.v14.v13.TEMPLATE_AMENDMENT_SHA256,
        "subprocess_environment_correction_amendment_sha256": runner.v14.ENVIRONMENT_AMENDMENT_SHA256,
        "subprocess_environment_contract_sha256": runner.v14.ENVIRONMENT_CONTRACT_SHA256,
        "preflight_delegation_recursion_amendment_sha256": runner.DELEGATION_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": runner.v14.v13.v12.NO_FALLBACK_OVERRIDE_SHA256,
        "activation_sha256": activation_sha, "canary_sha256": canary_sha,
        "canary_contract_sha256": legacy.EXPECTED_CANARY_CONTRACT_SHA256, "runtime_pin_erratum_sha256": legacy.EXPECTED_RUNTIME_ERRATUM_SHA256,
        "provenance_activation_sha256": legacy.EXPECTED_PROVENANCE_ACTIVATION_SHA256, "resource_receipt_sha256": legacy.EXPECTED_RESOURCE_RECEIPT_SHA256,
        "baseline_protocol_sha256": legacy._sha256_file(legacy.BASE_PROTOCOL), "model_manifest_sha256": manifest,
        "model_artifact_bytes": artifact_bytes, "model_artifact_file_count": artifact_files,
        "prompt_and_schema_sha256": worker.prompt_and_schema_digest(), "template_order_correction_digest": worker.template_order_correction_digest(),
        "parser_mode": legacy._parser_mode(activation), "qwen3_private_digest_exported": False,
        "automatic_fallback_allowed": False, "scientific_endpoint_if_gemma_fails": False,
    }


def _validate_seal(seal: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(seal, Mapping) or seal.get("schema_version") != "nursery-childlens-gemma4-substitution-preflight-delegation-seal-v1.5" or dict(seal) != _public_gemma_seal():
        raise GemmaSubstitutionError("E_SEAL")
    return legacy._read_json(legacy.AMENDMENT), legacy._read_json(ACTIVATION)


_v14_restricted_execute = v14.restricted_execute
_v14_validate_public_receipt = v14._validate_public_receipt


def restricted_execute(seal: Any) -> dict[str, Any]:
    receipt = dict(_v14_restricted_execute(seal))
    receipt.update({"preflight_delegation_recursion_amendment_sha256": runner.DELEGATION_AMENDMENT_SHA256, "automatic_fallback_allowed": False, "scientific_endpoint_if_gemma_fails": False})
    return receipt


def _validate_public_receipt(receipt: Any) -> None:
    _v14_validate_public_receipt(receipt)
    if not isinstance(receipt, Mapping) or receipt.get("preflight_delegation_recursion_amendment_sha256") != runner.DELEGATION_AMENDMENT_SHA256 or receipt.get("automatic_fallback_allowed") is not False or receipt.get("scientific_endpoint_if_gemma_fails") is not False:
        raise GemmaSubstitutionError("E_DELEGATION_CALIBRATION_RECEIPT")


def _configure() -> None:
    for module in (v14, legacy):
        module.ACTIVATION = ACTIVATION; module.PUBLIC_CANARY = PUBLIC_CANARY; module.PUBLIC_RECEIPT = PUBLIC_RECEIPT; module.PUBLIC_FAILURE = PUBLIC_FAILURE
        module._public_gemma_seal = _public_gemma_seal; module._validate_seal = _validate_seal; module.restricted_execute = restricted_execute; module._validate_public_receipt = _validate_public_receipt
    legacy.SCRIPT = SCRIPT; legacy.WORKER = WORKER; legacy.worker = worker


def _write_failure(code: str) -> None:
    if PUBLIC_FAILURE.exists(): return
    value = {"schema_version": "nursery-childlens-gemma-substitution-preflight-delegation-failure-v1.5", "status": "REVISE", "failure_code": code, "preflight_delegation_recursion_amendment_sha256": runner.DELEGATION_AMENDMENT_SHA256, "no_automatic_fallback_override_sha256": runner.v14.v13.v12.NO_FALLBACK_OVERRIDE_SHA256, "automatic_calibration_fallback_allowed": False, "scientific_endpoint_allowed": False, "restricted_payload_exported": False}
    descriptor = os.open(PUBLIC_FAILURE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle: handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    _configure(); parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--restricted", action="store_true"); parser.add_argument("--seal-fd", type=int); parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.restricted:
            if args.seal_fd is None or args.result_fd is None: return 2
            legacy._write_fd(args.result_fd, restricted_execute(legacy._read_fd(args.seal_fd))); return 0
        if args.seal_fd is not None or args.result_fd is not None: return 2
        receipt = legacy.public_execute(); passed = receipt.get("status") == "CALIBRATION_PASS"
        print("CHILDLENS_GEMMA_PREFLIGHT_DELEGATION_SUBSTITUTION_COMPLETE" if passed else "CHILDLENS_GEMMA_PREFLIGHT_DELEGATION_SUBSTITUTION_REVISE"); return 0 if passed else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, GemmaSubstitutionError) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None: code = "E_INTERNAL"
        if args.restricted and args.result_fd is not None:
            with contextlib.suppress(Exception): legacy._write_fd(args.result_fd, {"schema_version": "nursery-childlens-gemma-substitution-preflight-delegation-failure-v1.5", "failure_code": code})
        else:
            with contextlib.suppress(Exception): _write_failure(code)
            print("CHILDLENS_GEMMA_PREFLIGHT_DELEGATION_SUBSTITUTION_REVISE")
        return 2


if __name__ == "__main__": raise SystemExit(main())
