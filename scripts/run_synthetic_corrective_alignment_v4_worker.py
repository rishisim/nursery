from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
while str(SOURCE_ROOT) in sys.path:
    sys.path.remove(str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

EXPECTED_PARALLEL = SOURCE_ROOT / "babyworld_lite/corrective_alignment_v4/parallel.py"
FAILURE_SCHEMA = "nursery-corrective-subprocess-failure-v4"
REQUEST_SCHEMA = "nursery-corrective-subprocess-request-v4"
INVOCATION_TOKEN_BYTES = 32


def _stable_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError("bootstrap evidence must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PermissionError("bootstrap evidence changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(_stable_regular_bytes(path)).hexdigest()


def _require_real_chain(path: Path, boundary: Path) -> None:
    candidate = path.absolute()
    root = boundary.absolute()
    if candidate != root and root not in candidate.parents:
        raise PermissionError("bootstrap source path escapes source root")
    current = candidate
    while True:
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError("bootstrap source symlink forbidden")
        if current == root:
            break
        current = current.parent


def _bootstrap_request(
    request_value: str,
    request_sha256: str,
    request_manifest_sha256: str,
    invocation_token: bytes,
) -> tuple[str, Path]:
    request_path = Path(request_value).absolute()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", request_sha256)
        or not re.fullmatch(r"[0-9a-f]{64}", request_manifest_sha256)
        or request_path.parent.name != "requests"
        or _sha256(request_path) != request_sha256
    ):
        raise PermissionError("bootstrap request commitment mismatch")
    scheduler = request_path.parent.parent
    if _sha256(request_path.parent / "manifest.json") != request_manifest_sha256:
        raise PermissionError("bootstrap request-manifest commitment mismatch")
    request = json.loads(_stable_regular_bytes(request_path))
    if set(request) != {
        "schema_version",
        "unit_id",
        "invocation_token_sha256",
        "arguments",
        "success_receipt_path",
        "failure_receipt_path",
        "bootstrap_tmp_root",
    } or request.get("schema_version") != REQUEST_SCHEMA:
        raise PermissionError("bootstrap request schema mismatch")
    unit_id = str(request["unit_id"])
    if (
        re.fullmatch(r"corpus-[0-9]+__model-[0-9]+", unit_id) is None
        or len(invocation_token) != INVOCATION_TOKEN_BYTES
        or request.get("invocation_token_sha256")
        != hashlib.sha256(invocation_token).hexdigest()
        or request_path != scheduler / "requests" / f"{unit_id}.json"
        or str(request.get("arguments", {}).get("unit", {}).get("unit_id", ""))
        != unit_id
        or int(request.get("arguments", {}).get("scheduler_parent_pid", -1))
        != os.getppid()
    ):
        raise PermissionError("bootstrap request unit mismatch")
    failure_path = Path(str(request["failure_receipt_path"])).absolute()
    success_path = Path(str(request["success_receipt_path"])).absolute()
    bootstrap_tmp = Path(str(request["bootstrap_tmp_root"])).absolute()
    if (
        failure_path != scheduler / "failures" / f"{unit_id}.json"
        or success_path != scheduler / "receipts" / f"{unit_id}.json"
        or bootstrap_tmp != scheduler / "bootstrap_tmp" / unit_id
        or os.environ.get("TMPDIR") != str(bootstrap_tmp)
    ):
        raise PermissionError("bootstrap receipt/temp path mismatch")
    for parent in (
        scheduler,
        scheduler / "requests",
        scheduler / "failures",
        scheduler / "receipts",
        bootstrap_tmp,
    ):
        metadata = parent.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError("bootstrap parent is not a real directory")
    return unit_id, failure_path


def _read_invocation_token(fd_value: str) -> bytes:
    try:
        descriptor = int(fd_value)
    except (TypeError, ValueError) as error:
        raise PermissionError("invocation capability descriptor is invalid") from error
    if descriptor < 3:
        raise PermissionError("invocation capability descriptor is unsafe")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISFIFO(metadata.st_mode):
            raise PermissionError("invocation capability is not an inherited pipe")
        token = os.read(descriptor, INVOCATION_TOKEN_BYTES + 1)
        if os.read(descriptor, 1):
            raise PermissionError("invocation capability contains trailing bytes")
    finally:
        os.close(descriptor)
    if len(token) != INVOCATION_TOKEN_BYTES:
        raise PermissionError("invocation capability length mismatch")
    return token


def _write_bootstrap_failure(
    failure_path: Path,
    *,
    unit_id: str,
    request_sha256: str,
    request_manifest_sha256: str,
    error: BaseException,
) -> None:
    payload = {
        "schema_version": FAILURE_SCHEMA,
        "status": "FAILED_NO_OUTCOME",
        "unit_id": unit_id,
        "request_sha256": request_sha256,
        "request_manifest_sha256": request_manifest_sha256,
        "exception": type(error).__name__,
        "message": str(error),
        "scientific_outcome": False,
    }
    data = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        failure_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _import_parallel():
    _require_real_chain(EXPECTED_PARALLEL, SOURCE_ROOT)
    metadata = EXPECTED_PARALLEL.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise PermissionError("corrective parallel source must be a regular file")
    from babyworld_lite.corrective_alignment_v4 import parallel as parallel

    if Path(str(parallel.__file__)).resolve() != EXPECTED_PARALLEL.resolve():
        raise PermissionError("corrective worker imported parallel code outside SOURCE_ROOT")
    return parallel


def _bootstrap_probe() -> int:
    parallel = _import_parallel()
    required = {}
    for name in ("numpy", "scipy", "yaml", "pytest", "threadpoolctl"):
        module = importlib.import_module(name)
        required[name] = str(Path(str(module.__file__)).resolve())
    launcher = Path(sys.executable).absolute()
    pyvenv_cfg = launcher.parent.parent / "pyvenv.cfg"
    result: dict[str, Any] = {
        "schema_version": "nursery-corrective-worker-bootstrap-probe-v4",
        "status": "PASS",
        "source_root": str(SOURCE_ROOT),
        "parallel_module": str(EXPECTED_PARALLEL),
        "python_executable": str(launcher),
        "python_executable_resolved": str(launcher.resolve()),
        "python_sha256": _sha256(launcher.resolve()),
        "pyvenv_cfg_path": str(pyvenv_cfg),
        "pyvenv_cfg_sha256": _sha256(pyvenv_cfg),
        "sys_prefix": str(Path(sys.prefix).absolute()),
        "sys_base_prefix": str(Path(sys.base_prefix).absolute()),
        "required_module_origins": required,
        "python_startup_flags": parallel.python_startup_flags_projection(),
        "scientific_outcome": False,
        "seed_identifiers_used": False,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bootstrap-probe", action="store_true")
    parser.add_argument("--request")
    parser.add_argument("--request-sha256")
    parser.add_argument("--request-manifest-sha256")
    parser.add_argument("--invocation-fd")
    arguments = parser.parse_args()
    if arguments.bootstrap_probe:
        if any(
            value is not None
            for value in (
                arguments.request,
                arguments.request_sha256,
                arguments.request_manifest_sha256,
                arguments.invocation_fd,
            )
        ):
            return 2
        try:
            return _bootstrap_probe()
        except BaseException as error:
            print(f"bootstrap probe failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
    if any(
        value is None
        for value in (
            arguments.request,
            arguments.request_sha256,
            arguments.request_manifest_sha256,
            arguments.invocation_fd,
        )
    ):
        return 2
    failure_path: Path | None = None
    unit_id = "unknown"
    try:
        invocation_token = _read_invocation_token(str(arguments.invocation_fd))
        unit_id, failure_path = _bootstrap_request(
            str(arguments.request),
            str(arguments.request_sha256),
            str(arguments.request_manifest_sha256),
            invocation_token,
        )
        parallel = _import_parallel()
    except BaseException as error:
        if failure_path is not None:
            try:
                _write_bootstrap_failure(
                    failure_path,
                    unit_id=unit_id,
                    request_sha256=str(arguments.request_sha256),
                    request_manifest_sha256=str(arguments.request_manifest_sha256),
                    error=error,
                )
            except BaseException:
                pass
        return 1
    return parallel._subprocess_worker_main(
        str(arguments.request),
        str(arguments.request_sha256),
        str(arguments.request_manifest_sha256),
        invocation_token,
    )


if __name__ == "__main__":
    raise SystemExit(main())
