#!/usr/bin/env python3
"""Additive v1.3 rendered-template-order public canary and activation runner."""

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
V12_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_2.py"
WORKER = ROOT / "scripts/nursery_gemma4_referential_worker_v1_3.py"
TEMPLATE_AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json"
OUTPUT_ROOT = ROOT / "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3"
OUTPUT = OUTPUT_ROOT / "public_common_schema_canary_receipt_v1_3.json"
FAILURE = OUTPUT_ROOT / "public_common_schema_canary_failure_v1_3.json"
RUNTIME_LOCK = OUTPUT_ROOT / "runtime_lock_receipt_v1_3.json"
ACTIVATION = OUTPUT_ROOT / "full_activation_receipt_v1_3.json"
V12_FAILURE = ROOT / "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/public_common_schema_canary_failure_v1.json"

V12_RUNNER_SHA256 = "4506897e0a115f0d119990a9434eb37a3093b1a0f08947bff8553372466a81e2"
V12_FAILURE_SHA256 = "f84bf0c82136f4479258e6b4d9212c104487dc101deae67afdcc5a92d2a50ac4"
TEMPLATE_AMENDMENT_SHA256 = "2d117cf0f1619d43d6e71a90aa853241f0a5a7c331e84fc04af8231c5d6081fb"
CANARY_SCHEMA = "nursery-gemma4-e4b-prototype-common-schema-public-canary-template-order-v1.3"
RUNTIME_SCHEMA = "nursery-gemma4-e4b-prototype-runtime-lock-template-order-v1.3"
ACTIVATION_SCHEMA = "nursery-gemma4-e4b-prototype-full-activation-receipt-template-order-v1.3"
FAILURE_SCHEMA = "nursery-gemma4-public-canary-template-order-failure-v1.3"
MAX_PIPE_BYTES = 2 * 1024 * 1024


class TemplateCanaryError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TemplateCanaryError("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v12 = _load_module(V12_RUNNER, "nursery_gemma_template_v12_runner")
worker = _load_module(WORKER, "nursery_gemma_template_worker")
_v12_create_fixture = v12._create_fixture


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
        raise TemplateCanaryError("E_PUBLIC_ARTIFACT") from exc
    if not isinstance(value, Mapping):
        raise TemplateCanaryError("E_PUBLIC_ARTIFACT")
    return value


def _template_common_receipt() -> dict[str, Any]:
    base = dict(v12._caf_common_receipt())
    base.update({
        "output_namespace": "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3",
        "template_placeholder_order_amendment_path": "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json",
        "template_placeholder_order_amendment_sha256": TEMPLATE_AMENDMENT_SHA256,
        "prior_caf_runner_sha256": V12_RUNNER_SHA256,
        "prior_caf_failure_receipt_sha256": V12_FAILURE_SHA256,
        "declarative_user_content_order_preserved": True,
        "rendered_audio_placeholder_relocation_count": 1,
        "rendered_placeholder_order": "SYSTEM_AUDIO_FIVE_CONTIGUOUS_IMAGES_USER",
        "non_audio_rendered_bytes_changed": False,
        "parser_corrections_spent_before_template_canary": 0,
        "template_order_correction_digest": worker.template_order_correction_digest(),
    })
    return base


def _validate_template_amendment() -> Mapping[str, Any]:
    if (
        _sha256_file(V12_RUNNER) != V12_RUNNER_SHA256
        or _sha256_file(V12_FAILURE) != V12_FAILURE_SHA256
        or _sha256_file(TEMPLATE_AMENDMENT) != TEMPLATE_AMENDMENT_SHA256
        or _sha256_file(WORKER) == v12.legacy._sha256_file(v12.legacy.WORKER)
    ):
        raise TemplateCanaryError("E_TEMPLATE_BINDING")
    v12._validate_caf_amendment()
    amendment = _read_json(TEMPLATE_AMENDMENT)
    if (
        amendment.get("schema_version") != "nursery-gemma4-public-canary-rendered-template-placeholder-order-amendment-v1.3"
        or amendment.get("status") != "FROZEN_PRE_CANARY_TEMPLATE_PLACEHOLDER_ORDER_ONLY"
        or amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_POST_RENDER_PLACEHOLDER_ORDER_OVERLAY"
        or amendment.get("public_canary_run") is not False
        or amendment.get("restricted_content_accessed") is not False
        or amendment.get("restricted_inference_run") is not False
        or amendment.get("scientific_outcome_run") is not False
    ):
        raise TemplateCanaryError("E_TEMPLATE_AMENDMENT")
    delta = amendment.get("exact_post_render_delta", {})
    if (
        delta.get("maximum_relocated_substrings") != 1
        or delta.get("relocated_substring_utf8") != "<|audio|>"
        or len("<|audio|>".encode("utf-8")) != 9
        or delta.get("declarative_messages_changed") is not False
        or delta.get("non_placeholder_rendered_bytes_changed") is not False
        or delta.get("parser_allowance_changed") is not False
        or delta.get("does_not_spend_outer_fence_parser_correction") is not True
    ):
        raise TemplateCanaryError("E_TEMPLATE_DELTA")
    if any(path.exists() for path in (v12.OUTPUT, v12.RUNTIME_LOCK, v12.ACTIVATION)):
        raise TemplateCanaryError("E_V12_SUCCESS_OUTPUT_PRESENT")
    return amendment


def _validate_correction_state(parser_mode: str) -> None:
    if parser_mode == "PRIMARY":
        if OUTPUT_ROOT.exists():
            raise TemplateCanaryError("E_NEW_NAMESPACE_EXISTS")
        return
    if parser_mode != "ONE_OUTER_FENCE" or not OUTPUT_ROOT.is_dir():
        raise TemplateCanaryError("E_PARSER_CORRECTION_STATE")
    members = list(OUTPUT_ROOT.iterdir())
    failure = _read_json(FAILURE) if members == [FAILURE] and FAILURE.is_file() else {}
    if (
        failure.get("failure_code") != "E_SCHEMA_OUTPUT"
        or failure.get("parser_mode") != "PRIMARY"
        or failure.get("parser_correction_eligible") is not True
        or failure.get("template_placeholder_order_amendment_sha256") != TEMPLATE_AMENDMENT_SHA256
    ):
        raise TemplateCanaryError("E_PARSER_CORRECTION_STATE")


def _public_preflight(parser_mode: str) -> dict[str, Any]:
    _validate_template_amendment()
    _validate_correction_state(parser_mode)
    legacy_seal = v12.legacy._public_preflight(parser_mode)
    return {
        "schema_version": "nursery-gemma4-public-canary-template-order-seal-v1.3",
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_sha256": _sha256_file(WORKER),
        "v12_runner_sha256": V12_RUNNER_SHA256,
        "v12_failure_receipt_sha256": V12_FAILURE_SHA256,
        "template_placeholder_order_amendment_sha256": TEMPLATE_AMENDMENT_SHA256,
        "caf_transport_amendment_sha256": v12.CAF_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": v12.NO_FALLBACK_OVERRIDE_SHA256,
        "template_order_correction_digest": worker.template_order_correction_digest(),
        "parser_mode": parser_mode,
        "legacy_preflight": legacy_seal,
    }


def _create_fixture(directory: Path) -> tuple[Path, list[Path], dict[str, Any]]:
    audio, frames, original = _v12_create_fixture(directory)
    manifest = dict(original)
    manifest["generator_source_sha256"] = _sha256_file(SCRIPT)
    manifest["template_placeholder_order_amendment_sha256"] = TEMPLATE_AMENDMENT_SHA256
    manifest["template_order_correction_digest"] = worker.template_order_correction_digest()
    manifest["combined_fixture_sha256"] = _sha256_bytes(_canonical({key: value for key, value in manifest.items() if key != "combined_fixture_sha256"}))
    v12._LAST_FIXTURE_MANIFEST = manifest
    return audio, frames, manifest


def _v12_canary_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(receipt)
    projected["schema_version"] = v12.CANARY_SCHEMA
    projected["status"] = "PUBLIC_COMMON_SCHEMA_CANARY_CAF_PASS"
    projected["decision"] = "PUBLIC_COMMON_SCHEMA_CANARY_CAF_PASS"
    projected["output_namespace"] = "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1"
    projected["runner_sha256"] = V12_RUNNER_SHA256
    return projected


def _validate_template_canary(receipt: Mapping[str, Any]) -> None:
    if (
        receipt.get("schema_version") != CANARY_SCHEMA
        or receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS"
        or receipt.get("decision") != "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS"
        or receipt.get("template_placeholder_order_amendment_sha256") != TEMPLATE_AMENDMENT_SHA256
        or receipt.get("prior_caf_runner_sha256") != V12_RUNNER_SHA256
        or receipt.get("prior_caf_failure_receipt_sha256") != V12_FAILURE_SHA256
        or receipt.get("declarative_user_content_order_preserved") is not True
        or receipt.get("rendered_audio_placeholder_relocation_count") != 1
        or receipt.get("rendered_placeholder_order") != "SYSTEM_AUDIO_FIVE_CONTIGUOUS_IMAGES_USER"
        or receipt.get("non_audio_rendered_bytes_changed") is not False
        or receipt.get("parser_corrections_spent_before_template_canary") != 0
        or receipt.get("template_order_correction_digest") != worker.template_order_correction_digest()
    ):
        raise TemplateCanaryError("E_TEMPLATE_CANARY_RECEIPT")
    v12._validate_caf_canary(_v12_canary_projection(receipt))


def sandboxed_execute(seal: Any) -> dict[str, Any]:
    if not isinstance(seal, Mapping) or dict(seal) != _public_preflight(str(seal.get("parser_mode", ""))):
        raise TemplateCanaryError("E_SEAL")
    original_preflight = v12._public_preflight
    original_fixture = v12._create_fixture
    original_worker = v12.legacy.worker
    original_worker_path = v12.legacy.WORKER
    try:
        v12._public_preflight = _public_preflight
        v12._create_fixture = _create_fixture
        v12.legacy.worker = worker
        v12.legacy.WORKER = WORKER
        result = v12.sandboxed_execute(seal)
    finally:
        v12._public_preflight = original_preflight
        v12._create_fixture = original_fixture
        v12.legacy.worker = original_worker
        v12.legacy.WORKER = original_worker_path
    runtime = dict(result["runtime_lock"])
    runtime.update(_template_common_receipt())
    runtime.update({
        "schema_version": RUNTIME_SCHEMA,
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_adapter_sha256": _sha256_file(WORKER),
    })
    receipt = dict(result["receipt"])
    passed = receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_CAF_PASS"
    status = "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS" if passed else "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_REVISE"
    receipt.update(_template_common_receipt())
    receipt.update({
        "schema_version": CANARY_SCHEMA,
        "status": status,
        "decision": status,
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_adapter_sha256": _sha256_file(WORKER),
        "runtime_lock_sha256": _sha256_bytes(_canonical(runtime)),
    })
    return {"runtime_lock": runtime, "receipt": receipt}


def _v12_activation_projection(activation: Mapping[str, Any]) -> dict[str, Any]:
    projected = dict(activation)
    projected["schema_version"] = v12.ACTIVATION_SCHEMA
    projected["status"] = "PROTOTYPE_INSTRUMENT_ACTIVATED_CAF_RESTRICTED_INFERENCE_NOT_RUN"
    projected.pop("template_placeholder_order_binding", None)
    for key in _template_common_receipt():
        if key not in v12._caf_common_receipt():
            projected.pop(key, None)
    projected["output_namespace"] = "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1"
    immutable = dict(projected["immutable_bindings"])
    immutable.pop("template_placeholder_order_amendment", None)
    projected["immutable_bindings"] = immutable
    binding = dict(projected["public_canary_binding"])
    binding["receipt_schema_version"] = v12.CANARY_SCHEMA
    binding["receipt_path"] = "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/public_common_schema_canary_receipt_v1.json"
    projected["public_canary_binding"] = binding
    return projected


def _validate_template_activation(activation: Mapping[str, Any], canary_sha256: str) -> None:
    binding = activation.get("template_placeholder_order_binding", {})
    canary = activation.get("public_canary_binding", {})
    immutable = activation.get("immutable_bindings", {})
    if (
        activation.get("schema_version") != ACTIVATION_SCHEMA
        or activation.get("status") != "PROTOTYPE_INSTRUMENT_ACTIVATED_TEMPLATE_ORDER_RESTRICTED_INFERENCE_NOT_RUN"
        or not isinstance(binding, Mapping)
        or binding.get("sha256") != TEMPLATE_AMENDMENT_SHA256
        or binding.get("prior_caf_runner_sha256") != V12_RUNNER_SHA256
        or binding.get("prior_caf_failure_receipt_sha256") != V12_FAILURE_SHA256
        or binding.get("relocated_substring_utf8") != "<|audio|>"
        or binding.get("relocated_substring_bytes") != 9
        or binding.get("relocation_count") != 1
        or binding.get("non_audio_rendered_bytes_changed") is not False
        or not isinstance(immutable, Mapping)
        or immutable.get("template_placeholder_order_amendment", {}).get("sha256") != TEMPLATE_AMENDMENT_SHA256
        or not isinstance(canary, Mapping)
        or canary.get("receipt_schema_version") != CANARY_SCHEMA
        or canary.get("receipt_path") != "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/public_common_schema_canary_receipt_v1_3.json"
        or canary.get("receipt_sha256") != canary_sha256
    ):
        raise TemplateCanaryError("E_TEMPLATE_ACTIVATION")
    v12._validate_caf_activation(_v12_activation_projection(activation), canary_sha256)


def _build_activation_receipt(
    receipt: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    canary_bytes: bytes,
    runtime_lock_bytes: bytes,
    resource_recheck: Mapping[str, bool],
) -> dict[str, Any]:
    _validate_template_canary(receipt)
    base = v12._build_activation_receipt(
        _v12_canary_projection(receipt),
        runtime_lock,
        canary_bytes,
        runtime_lock_bytes,
        resource_recheck,
    )
    activation = dict(base)
    activation.update(_template_common_receipt())
    activation["schema_version"] = ACTIVATION_SCHEMA
    activation["status"] = "PROTOTYPE_INSTRUMENT_ACTIVATED_TEMPLATE_ORDER_RESTRICTED_INFERENCE_NOT_RUN"
    immutable = dict(activation["immutable_bindings"])
    immutable["template_placeholder_order_amendment"] = {
        "path": "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json",
        "sha256": TEMPLATE_AMENDMENT_SHA256,
        "status": "FROZEN_PRE_CANARY_TEMPLATE_PLACEHOLDER_ORDER_ONLY",
    }
    activation["immutable_bindings"] = immutable
    canary_binding = dict(activation["public_canary_binding"])
    canary_binding["receipt_schema_version"] = CANARY_SCHEMA
    canary_binding["receipt_path"] = "output/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/public_common_schema_canary_receipt_v1_3.json"
    activation["public_canary_binding"] = canary_binding
    activation["template_placeholder_order_binding"] = {
        "path": "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json",
        "sha256": TEMPLATE_AMENDMENT_SHA256,
        "status": "FROZEN_PRE_CANARY_TEMPLATE_PLACEHOLDER_ORDER_ONLY",
        "prior_caf_runner_sha256": V12_RUNNER_SHA256,
        "prior_caf_failure_receipt_sha256": V12_FAILURE_SHA256,
        "worker_sha256": _sha256_file(WORKER),
        "template_order_correction_digest": worker.template_order_correction_digest(),
        "relocated_substring_utf8": "<|audio|>",
        "relocated_substring_bytes": 9,
        "relocation_count": 1,
        "declarative_messages_changed": False,
        "non_audio_rendered_bytes_changed": False,
        "parser_correction_spent": False,
    }
    _validate_template_activation(activation, _sha256_bytes(canary_bytes))
    return activation


def _ensure_output_root() -> None:
    if not OUTPUT_ROOT.exists():
        OUTPUT_ROOT.mkdir(mode=0o700)
    metadata = OUTPUT_ROOT.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise TemplateCanaryError("E_OUTPUT_PRIVACY")


def _atomic_once(path: Path, payload: bytes) -> None:
    _ensure_output_root()
    v12.legacy._atomic_once(path, payload)


def _read_fd(descriptor: int) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(MAX_PIPE_BYTES + 1)
    if not payload or len(payload) > MAX_PIPE_BYTES:
        raise TemplateCanaryError("E_PIPE")
    return json.loads(payload)


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_PIPE_BYTES:
        raise TemplateCanaryError("E_PIPE")
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()


def public_execute(parser_mode: str) -> Mapping[str, Any]:
    seal = _public_preflight(parser_mode)
    backend = v12.legacy.firewall.NetworkIsolationBackend.detect()
    v12.legacy.firewall.verify_network_isolation(backend)
    resource_recheck = v12.legacy._launch_resource_recheck()
    mps_lock = v12.legacy._exclusive_public_mps_lock()
    mps_lock.__enter__()
    seal_read = seal_write = result_read = result_write = -1
    try:
        seal_read, seal_write = os.pipe()
        result_read, result_write = os.pipe()
        os.write(seal_write, _canonical(seal))
        os.close(seal_write)
        seal_write = -1
        argv = [
            str(v12.legacy.PYTHON), "-I", str(SCRIPT), "--sandboxed",
            "--seal-fd", str(seal_read), "--result-fd", str(result_write),
        ]
        process = subprocess.Popen(
            backend.command(argv),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(v12.legacy.firewall.scrubbed_subprocess_environment({
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
            })),
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
            raise TemplateCanaryError("E_SANDBOXED_CANARY")
        result = json.loads(payload)
        if not isinstance(result, Mapping) or not isinstance(result.get("receipt"), Mapping) or not isinstance(result.get("runtime_lock"), Mapping):
            raise TemplateCanaryError("E_CANARY_RESULT")
        receipt, runtime = dict(result["receipt"]), dict(result["runtime_lock"])
        if (
            runtime.get("schema_version") != RUNTIME_SCHEMA
            or runtime.get("template_placeholder_order_amendment_sha256") != TEMPLATE_AMENDMENT_SHA256
            or runtime.get("worker_adapter_sha256") != _sha256_file(WORKER)
            or receipt.get("runtime_lock_sha256") != _sha256_bytes(_canonical(runtime))
        ):
            raise TemplateCanaryError("E_RUNTIME_LOCK")
        if receipt.get("status") != "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS":
            return receipt
        _validate_template_canary(receipt)
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
        **_template_common_receipt(),
        "schema_version": FAILURE_SCHEMA,
        "status": "REVISE",
        "failure_code": code,
        "parser_mode": parser_mode,
        "parser_correction_eligible": parser_eligible,
        "runner_sha256": _sha256_file(SCRIPT),
        "worker_sha256": _sha256_file(WORKER),
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
        passed = receipt.get("status") == "PUBLIC_COMMON_SCHEMA_CANARY_TEMPLATE_ORDER_PASS"
        if not passed:
            _write_failure("E_SCHEMA_OUTPUT", parser_mode, parser_mode == "PRIMARY")
        print("GEMMA4_PUBLIC_COMMON_SCHEMA_TEMPLATE_ORDER_CANARY_PASS" if passed else "GEMMA4_PUBLIC_COMMON_SCHEMA_TEMPLATE_ORDER_CANARY_REVISE")
        return 0 if passed else 3
    except Exception as exc:
        code = exc.args[0] if isinstance(exc, (TemplateCanaryError, v12.CafCanaryError, v12.legacy.CanaryError, RuntimeError)) and exc.args else "E_INTERNAL"
        if not isinstance(code, str) or re.fullmatch(r"E_[A-Z0-9_]+", code) is None:
            code = "E_INTERNAL"
        if args.sandboxed and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd(args.result_fd, {"schema_version": FAILURE_SCHEMA, "failure_code": code})
        else:
            with contextlib.suppress(Exception):
                _write_failure(code, parser_mode, False)
            print("GEMMA4_PUBLIC_COMMON_SCHEMA_TEMPLATE_ORDER_CANARY_REVISE")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
