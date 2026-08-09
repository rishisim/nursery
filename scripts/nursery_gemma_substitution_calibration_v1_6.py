#!/usr/bin/env python3
"""Additive v1.6 restricted worker-environment calibration coordinator."""

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
V15_COORDINATOR = ROOT / "scripts/nursery_gemma_substitution_calibration_v1_5.py"
BASE_COORDINATOR = ROOT / "scripts/nursery_gemma_substitution_calibration.py"
WORKER = ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py"
FIREWALL = ROOT / "scripts/childlens_local_inference_firewall_v1_3.py"
WORKER_ENV_AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_restricted_worker_environment_v1_6/frozen_restricted_worker_environment_amendment_v1_6.json"
SUCCESS_PATH_ERRATUM = ROOT / "docs/nursery_program_convergence_v1/gemma4_restricted_worker_environment_v1_6_1/frozen_success_receipt_path_compatibility_erratum_v1_6_1.json"
EXECUTION_CONTRACT = ROOT / "docs/nursery_program_convergence_v1/calibration_conditioned_symbolic_execution_contract_v2.json"
V15_FAILURE = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_preflight_delegation_failure_v1_5.json"
PUBLIC_CANARY = ROOT / "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/public_common_schema_canary_receipt_v1_5.json"
RUNTIME_LOCK = ROOT / "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/runtime_lock_receipt_v1_5.json"
ACTIVATION = ROOT / "output/nursery_program_convergence_v1/gemma4_preflight_delegation_v1_5/full_activation_receipt_v1_5.json"
OUTPUT_ROOT = ROOT / "output/nursery_program_convergence_v1/gemma4_restricted_worker_environment_v1_6"
PUBLIC_RECEIPT = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
PUBLIC_FAILURE = ROOT / "output/nursery_program_convergence_v1/gemma4_restricted_worker_environment_v1_6/childlens_pseudo_calibration_gemma_substitution_failure_v1_6.json"

V15_COORDINATOR_SHA256 = "66e0454d854ecada6eddb1580312c63fd38656892b15bfb8b4ccf1339604958e"
BASE_COORDINATOR_SHA256 = "4fdacb4a0bf4e03627c4c036ec65107bb249bf0a4b35564f3cf7a5a58dbeeeed"
WORKER_SHA256 = "c78fa8e9c0f995ed611f7fd8f7d2076f03a73d370574d3c92cf5e107f236c38f"
FIREWALL_SHA256 = "a10f8c2568d402259215afe18f6005ceb4b52431607b42b132a8db849ff32f7b"
WORKER_ENV_AMENDMENT_SHA256 = "316fd2f1f9e4c1b69936703f86eb6b72e0fff574e9115d2a5bb138e5fa7da577"
SUCCESS_PATH_ERRATUM_SHA256 = "c85eb4034fbf7c02e4ddba8b48cb36c877ce27a968f22a619a3b4175ef6dd693"
EXECUTION_CONTRACT_SHA256 = "db760f4135f81f7f78184a80f8abae16578918ff10b44c952f83a76cf5daa810"
V15_FAILURE_SHA256 = "eb8dc3313c746bc6b6ca100ebab30853607191c625ed0bfc1a2f3fe292dc6b79"
PUBLIC_CANARY_SHA256 = "9843e812b605979641a6a777f870348c9aa3c7f10ec382a082bb87c84def4e3b"
RUNTIME_LOCK_SHA256 = "69e0ef3a47198956bc06a123d6bfbc6b448ebeef5c8e6ed5e248bb882ba92b1f"
ACTIVATION_SHA256 = "10cb7d0151e2fff40a3f5bf006ea510a3f82a4223410dec51ebfb08a34f14d7f"
MANDATORY_ENVIRONMENT_SHA256 = "bf67ff8d311d901d39d8f996339e33f62c72b2f87a0c4ff9d997dce069cdf822"
THREAD_EXTRAS = {"OMP_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2"}
OFFLINE_FLAGS = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError("E_MODULE")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


v15 = _load_module(V15_COORDINATOR, "nursery_gemma_worker_env_v15_coordinator")
legacy = v15.legacy
runner = v15.runner
worker = v15.worker
firewall = legacy.firewall
GemmaSubstitutionError = legacy.GemmaSubstitutionError


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json(path: Path) -> Mapping[str, Any]:
    try: value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc: raise GemmaSubstitutionError("E_PUBLIC_ARTIFACT") from exc
    if not isinstance(value, Mapping): raise GemmaSubstitutionError("E_PUBLIC_ARTIFACT")
    return value


def _validate_worker_environment_amendment() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    exact = {
        V15_COORDINATOR: V15_COORDINATOR_SHA256, BASE_COORDINATOR: BASE_COORDINATOR_SHA256,
        WORKER: WORKER_SHA256, FIREWALL: FIREWALL_SHA256, WORKER_ENV_AMENDMENT: WORKER_ENV_AMENDMENT_SHA256,
        SUCCESS_PATH_ERRATUM: SUCCESS_PATH_ERRATUM_SHA256, EXECUTION_CONTRACT: EXECUTION_CONTRACT_SHA256,
        V15_FAILURE: V15_FAILURE_SHA256, PUBLIC_CANARY: PUBLIC_CANARY_SHA256,
        RUNTIME_LOCK: RUNTIME_LOCK_SHA256, ACTIVATION: ACTIVATION_SHA256,
    }
    if any(_sha256_file(path) != digest for path, digest in exact.items()): raise GemmaSubstitutionError("E_WORKER_ENV_BINDING")
    amendment = _read_json(WORKER_ENV_AMENDMENT); erratum = _read_json(SUCCESS_PATH_ERRATUM)
    if (
        amendment.get("schema_version") != "nursery-gemma4-restricted-worker-subprocess-environment-amendment-v1.6"
        or amendment.get("status") != "FROZEN_PRE_RESTRICTED_RERUN_WORKER_ENVIRONMENT_ONLY"
        or amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_RESTRICTED_WORKER_ENVIRONMENT_CALL_CORRECTION"
        or amendment.get("restricted_rerun_executed") is not False or amendment.get("Gemma_worker_process_launched") is not False
        or amendment.get("Gemma_worker_inference_run") is not False or amendment.get("scientific_endpoint_opened") is not False
        or amendment.get("scientific_outcome_run") is not False
        or erratum.get("schema_version") != "nursery-gemma4-restricted-worker-success-receipt-path-compatibility-erratum-v1.6.1"
        or erratum.get("status") != "FROZEN_PRE_RESTRICTED_RERUN_SUCCESS_PATH_ONLY"
        or erratum.get("erratum_mode") != "ADDITIVE_ONE_FIELD_OUTPUT_PATH_COMPATIBILITY_OVERRIDE"
        or erratum.get("restricted_rerun_executed") is not False or erratum.get("restricted_content_accessed") is not False
    ): raise GemmaSubstitutionError("E_WORKER_ENV_AMENDMENT")
    delta = amendment.get("exact_restricted_worker_environment_delta", {}); tmp = delta.get("TMPDIR_rule", {}) if isinstance(delta, Mapping) else {}
    shape = delta.get("expected_final_environment_shape", {}) if isinstance(delta, Mapping) else {}
    if (
        delta.get("maximum_worker_environment_call_site_changes") != 1
        or delta.get("amended_scrubber_extra_mapping") != THREAD_EXTRAS
        or delta.get("exact_deleted_extra_keys") != list(OFFLINE_FLAGS)
        or delta.get("deleted_extra_key_count") != 4 or delta.get("retained_thread_extra_key_count") != 2
        or tmp.get("assignment") != "environment['TMPDIR'] = str(work)"
        or tmp.get("work_must_be_absolute_owner_private_quarantine_local_directory") is not True
        or tmp.get("other_post_scrubber_environment_mutations_allowed") is not False
        or shape.get("exact_total_key_count") != 17 or shape.get("inherited_environment_keys_allowed") is not False
        or delta.get("worker_subprocess_argv_changed") is not False
        or delta.get("worker_subprocess_cwd_stdio_passed_FDs_or_checkpointing_changed") is not False
        or delta.get("network_denial_or_firewall_policy_changed") is not False
        or delta.get("model_prompt_schema_or_parser_changed") is not False
        or delta.get("sample_selection_or_thresholds_changed") is not False
        or delta.get("does_not_spend_outer_fence_parser_correction") is not True
    ): raise GemmaSubstitutionError("E_WORKER_ENV_DELTA")
    supersession = erratum.get("exact_field_supersession", {}); creation = erratum.get("success_path_creation_contract", {})
    if (
        supersession.get("superseding_repo_relative_success_path") != str(PUBLIC_RECEIPT.relative_to(ROOT))
        or supersession.get("maximum_superseded_fields") != 1
        or creation.get("exclusive_creation_required") is not True or creation.get("fail_if_path_exists") is not True
        or erratum.get("failure_and_diagnostic_path_preserved", {}).get("resolved_failure_path") != str(PUBLIC_FAILURE.relative_to(ROOT))
    ): raise GemmaSubstitutionError("E_SUCCESS_PATH_ERRATUM")
    runner._validate_delegation_canary(_read_json(PUBLIC_CANARY))
    runner._validate_delegation_activation(_read_json(ACTIVATION), PUBLIC_CANARY_SHA256)
    if PUBLIC_RECEIPT.exists(): raise GemmaSubstitutionError("E_PUBLIC_RECEIPT_EXISTS")
    return amendment, erratum


def _worker_environment(work: Path) -> dict[str, str]:
    if not isinstance(work, Path) or not work.is_absolute() or not legacy._private_directory(work):
        raise GemmaSubstitutionError("E_WORKER_TMPDIR")
    try: resolved = work.resolve(strict=True)
    except OSError as exc: raise GemmaSubstitutionError("E_WORKER_TMPDIR") from exc
    if resolved != work or not legacy._private_directory(resolved) or not legacy._private_directory(resolved.parent):
        raise GemmaSubstitutionError("E_WORKER_TMPDIR")
    environment = dict(firewall.scrubbed_subprocess_environment({
        "OMP_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
    }))
    mandatory = {key: value for key, value in environment.items() if key not in THREAD_EXTRAS}
    if (
        len(mandatory) != 14 or hashlib.sha256(_canonical(mandatory)).hexdigest() != MANDATORY_ENVIRONMENT_SHA256
        or any(environment.get(key, "").encode("utf-8") != value.encode("utf-8") for key, value in {**OFFLINE_FLAGS, **THREAD_EXTRAS}.items())
        or set(environment) != set(mandatory) | set(THREAD_EXTRAS)
    ): raise GemmaSubstitutionError("E_SUBPROCESS_ENV")
    environment["TMPDIR"] = str(work)
    if len(environment) != 17 or environment.get("TMPDIR") != str(work): raise GemmaSubstitutionError("E_SUBPROCESS_ENV")
    return environment


def _public_gemma_seal() -> dict[str, Any]:
    _validate_worker_environment_amendment()
    seal = dict(v15._public_gemma_seal())
    seal.update({
        "schema_version": "nursery-childlens-gemma4-substitution-restricted-worker-environment-seal-v1.6",
        "script_sha256": _sha256_file(SCRIPT), "v15_coordinator_sha256": V15_COORDINATOR_SHA256,
        "restricted_worker_environment_amendment_sha256": WORKER_ENV_AMENDMENT_SHA256,
        "success_receipt_path_compatibility_erratum_sha256": SUCCESS_PATH_ERRATUM_SHA256,
        "execution_contract_sha256": EXECUTION_CONTRACT_SHA256,
        "automatic_fallback_allowed": False, "scientific_endpoint_if_gemma_fails": False,
    })
    return seal


def _validate_seal(seal: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(seal, Mapping) or seal.get("schema_version") != "nursery-childlens-gemma4-substitution-restricted-worker-environment-seal-v1.6" or dict(seal) != _public_gemma_seal():
        raise GemmaSubstitutionError("E_SEAL")
    return legacy._read_json(legacy.AMENDMENT), legacy._read_json(ACTIVATION)


_v15_restricted_execute = v15.restricted_execute
_v15_validate_public_receipt = v15._validate_public_receipt


def restricted_execute(seal: Any) -> dict[str, Any]:
    receipt = dict(_v15_restricted_execute(seal))
    receipt.update({
        "restricted_worker_environment_amendment_sha256": WORKER_ENV_AMENDMENT_SHA256,
        "success_receipt_path_compatibility_erratum_sha256": SUCCESS_PATH_ERRATUM_SHA256,
        "worker_environment_mandatory_key_count": 14, "worker_environment_thread_extra_key_count": 2,
        "worker_environment_quarantine_tmpdir_key_count": 1, "checkpoint_resume_preserved": True,
        "automatic_fallback_allowed": False, "scientific_endpoint_if_gemma_fails": False,
    })
    return receipt


def _validate_public_receipt(receipt: Any) -> None:
    _v15_validate_public_receipt(receipt)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("restricted_worker_environment_amendment_sha256") != WORKER_ENV_AMENDMENT_SHA256
        or receipt.get("success_receipt_path_compatibility_erratum_sha256") != SUCCESS_PATH_ERRATUM_SHA256
        or receipt.get("worker_environment_mandatory_key_count") != 14
        or receipt.get("worker_environment_thread_extra_key_count") != 2
        or receipt.get("worker_environment_quarantine_tmpdir_key_count") != 1
        or receipt.get("checkpoint_resume_preserved") is not True
        or receipt.get("automatic_fallback_allowed") is not False or receipt.get("scientific_endpoint_if_gemma_fails") is not False
    ): raise GemmaSubstitutionError("E_WORKER_ENV_CALIBRATION_RECEIPT")


def _configure() -> None:
    legacy.SCRIPT = SCRIPT; legacy.ACTIVATION = ACTIVATION; legacy.PUBLIC_CANARY = PUBLIC_CANARY
    legacy.PUBLIC_RECEIPT = PUBLIC_RECEIPT; legacy.PUBLIC_FAILURE = PUBLIC_FAILURE
    legacy.WORKER = WORKER; legacy.worker = worker; legacy._worker_environment = _worker_environment
    legacy._public_gemma_seal = _public_gemma_seal; legacy._validate_seal = _validate_seal
    legacy.restricted_execute = restricted_execute; legacy._validate_public_receipt = _validate_public_receipt


def _ensure_output_root() -> None:
    OUTPUT_ROOT.mkdir(mode=0o700, parents=False, exist_ok=True)
    metadata = OUTPUT_ROOT.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise GemmaSubstitutionError("E_OUTPUT_PRIVACY")


def _write_failure(code: str) -> None:
    _ensure_output_root()
    if PUBLIC_FAILURE.exists(): return
    value = {
        "schema_version": "nursery-childlens-gemma-substitution-restricted-worker-environment-failure-v1.6",
        "status": "REVISE", "failure_code": code,
        "restricted_worker_environment_amendment_sha256": WORKER_ENV_AMENDMENT_SHA256,
        "success_receipt_path_compatibility_erratum_sha256": SUCCESS_PATH_ERRATUM_SHA256,
        "no_automatic_fallback_override_sha256": runner.v14.v13.v12.NO_FALLBACK_OVERRIDE_SHA256,
        "automatic_calibration_fallback_allowed": False, "scientific_endpoint_allowed": False,
        "restricted_payload_exported": False,
    }
    descriptor = os.open(PUBLIC_FAILURE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle: handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def public_execute() -> Mapping[str, Any]:
    seal = _public_gemma_seal()
    backend = firewall.NetworkIsolationBackend.detect()
    firewall.verify_network_isolation(backend)
    seal_read, seal_write = os.pipe(); result_read, result_write = os.pipe()
    try:
        os.write(seal_write, _canonical(seal)); os.close(seal_write); seal_write = -1
        argv = [sys.executable, str(SCRIPT), "--restricted", "--seal-fd", str(seal_read), "--result-fd", str(result_write)]
        process = subprocess.Popen(
            backend.command(argv), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=dict(firewall.scrubbed_subprocess_environment()),
            pass_fds=(seal_read, result_write), close_fds=True,
        )
        os.close(seal_read); seal_read = -1; os.close(result_write); result_write = -1
        with os.fdopen(result_read, "rb") as handle: payload = handle.read(legacy.MAX_PIPE_BYTES + 1)
        result_read = -1; returncode = process.wait(timeout=8 * 60 * 60)
        if not payload or len(payload) > legacy.MAX_PIPE_BYTES: raise GemmaSubstitutionError("E_RESTRICTED_EXECUTION")
        receipt = json.loads(payload)
        if returncode != 0:
            code = receipt.get("failure_code") if isinstance(receipt, Mapping) else None
            raise GemmaSubstitutionError(code if isinstance(code, str) and re.fullmatch(r"E_[A-Z0-9_]+", code) else "E_RESTRICTED_EXECUTION")
        _validate_public_receipt(receipt)
        descriptor = os.open(PUBLIC_RECEIPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
                handle.flush(); os.fsync(handle.fileno())
        except Exception:
            with contextlib.suppress(OSError): PUBLIC_RECEIPT.unlink()
            raise
        return receipt
    finally:
        for descriptor in (seal_read, seal_write, result_read, result_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError): os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    _configure(); parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--restricted", action="store_true"); parser.add_argument("--seal-fd", type=int); parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        if args.restricted:
            if args.seal_fd is None or args.result_fd is None: return 2
            legacy._write_fd(args.result_fd, restricted_execute(legacy._read_fd(args.seal_fd))); return 0
        if args.seal_fd is not None or args.result_fd is not None: return 2
        _validate_worker_environment_amendment(); _ensure_output_root(); receipt = public_execute()
        passed = receipt.get("status") == "CALIBRATION_PASS"
        print("CHILDLENS_GEMMA_WORKER_ENVIRONMENT_SUBSTITUTION_COMPLETE" if passed else "CHILDLENS_GEMMA_WORKER_ENVIRONMENT_SUBSTITUTION_REVISE"); return 0 if passed else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, GemmaSubstitutionError) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None: code = "E_INTERNAL"
        if args.restricted and args.result_fd is not None:
            with contextlib.suppress(Exception): legacy._write_fd(args.result_fd, {"schema_version": "nursery-childlens-gemma-substitution-restricted-worker-environment-failure-v1.6", "failure_code": code})
        else:
            with contextlib.suppress(Exception): _write_failure(code)
            print("CHILDLENS_GEMMA_WORKER_ENVIRONMENT_SUBSTITUTION_REVISE")
        return 2


if __name__ == "__main__": raise SystemExit(main())
