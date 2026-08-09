#!/usr/bin/env python3
"""Additive CAF-v1.2 gate for the unchanged Gemma/Qwen3 calibration.

Only the fresh CAF canary and activation chain can open the legacy restricted
coordinator. The 15/900/137 sample, exact common schema, private Qwen3
read-only checks, thresholds, and K=5 export policy are delegated unchanged.
"""

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
LEGACY_COORDINATOR = ROOT / "scripts/nursery_gemma_substitution_calibration.py"
CAF_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_2.py"
NO_FALLBACK_OVERRIDE = ROOT / "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json"
PUBLIC_RECEIPT = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
PUBLIC_FAILURE = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_caf_failure_v1.json"
LEGACY_COORDINATOR_SHA256 = "4fdacb4a0bf4e03627c4c036ec65107bb249bf0a4b35564f3cf7a5a58dbeeeed"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_module(LEGACY_COORDINATOR, "nursery_gemma_caf_legacy_coordinator")
caf = _load_module(CAF_RUNNER, "nursery_gemma_caf_public_runner")
GemmaSubstitutionError = legacy.GemmaSubstitutionError

ACTIVATION = caf.ACTIVATION
PUBLIC_CANARY = caf.OUTPUT


def _public_gemma_seal() -> dict[str, Any]:
    legacy._historical_immutability_check()
    if legacy._sha256_file(LEGACY_COORDINATOR) != LEGACY_COORDINATOR_SHA256:
        raise GemmaSubstitutionError("E_LEGACY_COORDINATOR")
    caf._validate_caf_amendment()
    if not ACTIVATION.is_file() or not PUBLIC_CANARY.is_file():
        raise GemmaSubstitutionError("E_CAF_ACTIVATION_MISSING")
    canary = legacy._read_json(PUBLIC_CANARY)
    activation = legacy._read_json(ACTIVATION)
    canary_sha = legacy._sha256_file(PUBLIC_CANARY)
    activation_sha = legacy._sha256_file(ACTIVATION)
    try:
        caf._validate_caf_canary(canary)
        caf._validate_caf_activation(activation, canary_sha)
    except caf.CafCanaryError as exc:
        raise GemmaSubstitutionError("E_CAF_CHAIN") from exc
    override = legacy._read_json(NO_FALLBACK_OVERRIDE)
    if legacy.replacement_validator.validate_no_fallback_override(
        override, caf.NO_FALLBACK_OVERRIDE_SHA256
    ):
        raise GemmaSubstitutionError("E_NO_FALLBACK")
    amendment = legacy._read_json(legacy.AMENDMENT)
    prior_stop = legacy._read_json(legacy.PRIOR_STOP)
    provenance = legacy._read_json(legacy.PROVENANCE_ACTIVATION)
    contract = legacy._read_json(legacy.CANARY_CONTRACT)
    erratum = legacy._read_json(legacy.RUNTIME_ERRATUM)
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
        or legacy.replacement_validator.validate_runtime_erratum(
            erratum, legacy.EXPECTED_RUNTIME_ERRATUM_SHA256, legacy.EXPECTED_ACTIVATION_SCHEMA_SHA256
        )
    ):
        raise GemmaSubstitutionError("E_PUBLIC_BINDING")
    if not legacy.GEMMA_PYTHON.is_file() or not legacy.FFMPEG.is_file() or not legacy.GEMMA_MODEL.is_dir():
        raise GemmaSubstitutionError("E_RUNTIME")
    manifest, artifact_bytes, artifact_files = legacy._tree_manifest(legacy.GEMMA_MODEL)
    artifact = activation.get("artifact_binding", {})
    runtime = activation.get("runtime_binding", {})
    qwen = activation.get("qwen3_read_only_binding", {})
    binding = activation.get("public_canary_binding", {})
    if not all(isinstance(value, Mapping) for value in (artifact, runtime, qwen, binding)):
        raise GemmaSubstitutionError("E_CAF_CHAIN")
    if (
        artifact.get("snapshot_manifest_sha256") != manifest
        or artifact.get("snapshot_bytes") != artifact_bytes
        or artifact.get("snapshot_file_count") != artifact_files
        or runtime.get("canary_adapter_sha256") != legacy._sha256_file(legacy.WORKER)
        or runtime.get("network_denial_profile_sha256") != legacy._sha256_file(legacy.FIREWALL)
        or canary.get("prompt_sha256") != legacy.worker.prompt_digest()
        or canary.get("exact_output_schema_sha256") != legacy.worker.exact_schema_digest()
        or canary.get("caf_transport_amendment_sha256") != caf.CAF_AMENDMENT_SHA256
        or canary.get("no_automatic_fallback_override_sha256") != caf.NO_FALLBACK_OVERRIDE_SHA256
        or binding.get("parser_correction_count") != canary.get("corrections_used")
        or qwen.get("restricted_hypotheses_recomputed") is not False
        or qwen.get("restricted_digest_exported") is not False
        or qwen.get("third_visual_model") is not False
    ):
        raise GemmaSubstitutionError("E_MODEL_SEAL")
    return {
        "schema_version": "nursery-childlens-gemma4-substitution-caf-seal-v1",
        "script_sha256": legacy._sha256_file(SCRIPT),
        "legacy_coordinator_sha256": LEGACY_COORDINATOR_SHA256,
        "worker_sha256": legacy._sha256_file(legacy.WORKER),
        "amendment_sha256": amendment_sha,
        "caf_transport_amendment_sha256": caf.CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": caf.NO_FALLBACK_OVERRIDE_SHA256,
        "caf_runner_sha256": legacy._sha256_file(CAF_RUNNER),
        "activation_sha256": activation_sha,
        "canary_sha256": canary_sha,
        "canary_contract_sha256": legacy.EXPECTED_CANARY_CONTRACT_SHA256,
        "runtime_pin_erratum_sha256": legacy.EXPECTED_RUNTIME_ERRATUM_SHA256,
        "provenance_activation_sha256": legacy.EXPECTED_PROVENANCE_ACTIVATION_SHA256,
        "resource_receipt_sha256": legacy.EXPECTED_RESOURCE_RECEIPT_SHA256,
        "baseline_protocol_sha256": legacy._sha256_file(legacy.BASE_PROTOCOL),
        "model_manifest_sha256": manifest,
        "model_artifact_bytes": artifact_bytes,
        "model_artifact_file_count": artifact_files,
        "prompt_and_schema_sha256": legacy.worker.prompt_and_schema_digest(),
        "parser_mode": legacy._parser_mode(activation),
        "qwen3_private_digest_exported": False,
        "automatic_fallback_allowed": False,
        "scientific_endpoint_if_gemma_fails": False,
    }


def _validate_seal(seal: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(seal, Mapping) or seal.get("schema_version") != "nursery-childlens-gemma4-substitution-caf-seal-v1":
        raise GemmaSubstitutionError("E_SEAL")
    if dict(seal) != _public_gemma_seal():
        raise GemmaSubstitutionError("E_SEAL")
    return legacy._read_json(legacy.AMENDMENT), legacy._read_json(ACTIVATION)


_legacy_restricted_execute = legacy.restricted_execute
_legacy_validate_public_receipt = legacy._validate_public_receipt


def restricted_execute(seal: Any) -> dict[str, Any]:
    receipt = dict(_legacy_restricted_execute(seal))
    receipt["caf_transport_amendment_sha256"] = caf.CAF_AMENDMENT_SHA256
    receipt["no_automatic_fallback_override_sha256"] = caf.NO_FALLBACK_OVERRIDE_SHA256
    receipt["automatic_fallback_allowed"] = False
    receipt["scientific_endpoint_if_gemma_fails"] = False
    return receipt


def _validate_public_receipt(receipt: Any) -> None:
    _legacy_validate_public_receipt(receipt)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("caf_transport_amendment_sha256") != caf.CAF_AMENDMENT_SHA256
        or receipt.get("no_automatic_fallback_override_sha256") != caf.NO_FALLBACK_OVERRIDE_SHA256
        or receipt.get("automatic_fallback_allowed") is not False
        or receipt.get("scientific_endpoint_if_gemma_fails") is not False
    ):
        raise GemmaSubstitutionError("E_CAF_CALIBRATION_RECEIPT")


def _configure_legacy() -> None:
    legacy.SCRIPT = SCRIPT
    legacy.ACTIVATION = ACTIVATION
    legacy.PUBLIC_CANARY = PUBLIC_CANARY
    legacy.PUBLIC_RECEIPT = PUBLIC_RECEIPT
    legacy.PUBLIC_FAILURE = PUBLIC_FAILURE
    legacy._public_gemma_seal = _public_gemma_seal
    legacy._validate_seal = _validate_seal
    legacy.restricted_execute = restricted_execute
    legacy._validate_public_receipt = _validate_public_receipt


def _write_failure(code: str) -> None:
    if PUBLIC_FAILURE.exists():
        return
    failure = {
        "schema_version": "nursery-childlens-gemma-substitution-caf-failure-v1",
        "status": "REVISE",
        "failure_code": code,
        "caf_transport_amendment_sha256": caf.CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": caf.NO_FALLBACK_OVERRIDE_SHA256,
        "automatic_calibration_fallback_allowed": False,
        "scientific_endpoint_allowed": False,
        "restricted_payload_exported": False,
    }
    descriptor = os.open(PUBLIC_FAILURE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(failure, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_legacy()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restricted", action="store_true")
    parser.add_argument("--seal-fd", type=int)
    parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.restricted:
            if args.seal_fd is None or args.result_fd is None:
                return 2
            legacy._write_fd(args.result_fd, restricted_execute(legacy._read_fd(args.seal_fd)))
            return 0
        if args.seal_fd is not None or args.result_fd is not None:
            return 2
        receipt = legacy.public_execute()
        passed = receipt.get("status") == "CALIBRATION_PASS"
        print("CHILDLENS_GEMMA_CAF_SUBSTITUTION_COMPLETE" if passed else "CHILDLENS_GEMMA_CAF_SUBSTITUTION_REVISE")
        return 0 if passed else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, GemmaSubstitutionError) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None:
            code = "E_INTERNAL"
        if args.restricted and args.result_fd is not None:
            with contextlib.suppress(Exception):
                legacy._write_fd(args.result_fd, {"schema_version": "nursery-childlens-gemma-substitution-caf-failure-v1", "failure_code": code})
        else:
            with contextlib.suppress(Exception):
                _write_failure(code)
            print("CHILDLENS_GEMMA_CAF_SUBSTITUTION_REVISE")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
