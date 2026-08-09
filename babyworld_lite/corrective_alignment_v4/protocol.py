from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import PROTOCOL_ID, SCHEMA_VERSION


PURPOSES = (
    "construction_micro",
    "construction_qualification",
    "construction_mechanism",
    "excluded_rehearsal",
    "development",
)
ROLES = ("corpus", "model", "inference")
OPERATIONS = (
    "generate",
    "condition",
    "fit",
    "predict",
    "score",
    "control",
    "inference",
    "recompute",
)
ADJUDICATION_RELEVANT_FILES = frozenset(
    {
        "work_plan.json",
        "execution_contract.json",
        "merged_worker_results.json",
        "merged_operation_ledgers.json",
        "parent_operation_ledger.json",
        "parent_ledger_verification.json",
        "scored_condition_units.json",
        "scored_mutation_units.json",
        "model_averages.json",
        "primary_inference.json",
        "present_null_dependence.json",
        "mechanism_mutations.json",
        "informativeness.json",
        "development_validity.json",
        "scientific_summary.json",
        "completeness.json",
        "manifest.json",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def python_startup_flags_projection() -> dict[str, Any]:
    """Return every scalar interpreter-startup flag exposed by this Python."""
    return {
        name: getattr(sys.flags, name)
        for name in sorted(dir(sys.flags))
        if not name.startswith("_")
        and isinstance(getattr(sys.flags, name), (bool, int, str, type(None)))
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lstat_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def require_lstat_absent(path: str | Path) -> None:
    value = Path(path)
    if _lstat_exists(value):
        raise FileExistsError(f"path must be pristine and lstat-absent: {value}")


def atomic_rename_directory_noreplace(
    source: str | Path,
    destination: str | Path,
) -> None:
    """Atomically publish one real directory without replacing any destination."""
    source_path = Path(source).absolute()
    destination_path = Path(destination).absolute()
    source_metadata = source_path.lstat()
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISDIR(
        source_metadata.st_mode
    ):
        raise RuntimeError("atomic directory source must be a real directory")
    _reject_symlink_ancestors(source_path.parent)
    _reject_symlink_ancestors(destination_path.parent)
    destination_parent = destination_path.parent
    parent_metadata = destination_parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
        parent_metadata.st_mode
    ):
        raise RuntimeError("atomic directory destination parent must be real")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_path)
    destination_bytes = os.fsencode(destination_path)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise RuntimeError("kernel no-replace directory rename is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                f"atomic directory destination already exists: {destination_path}"
            )
        raise OSError(error, os.strerror(error), str(destination_path))
    for parent in {source_path.parent, destination_path.parent}:
        _fsync_directory(parent)


def _reject_symlink_ancestors(path: Path) -> None:
    current = path.absolute()
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"symlink ancestor forbidden: {current}")
        if current.parent == current:
            break
        current = current.parent


def _fsync_directory(path: str | Path) -> None:
    directory = Path(path).absolute()
    _reject_symlink_ancestors(directory)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("directory fsync target is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _real_directory_root(root: str | Path) -> Path:
    supplied = Path(root).absolute()
    _reject_symlink_ancestors(supplied)
    metadata = supplied.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"real directory root required: {supplied}")
    return supplied.resolve()


def atomic_write_bytes(
    path: str | Path,
    value: bytes,
    *,
    overwrite: bool = False,
) -> None:
    destination = Path(path)
    _reject_symlink_ancestors(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(destination.parent)
    if not overwrite:
        require_lstat_absent(destination)
    temporary = destination.with_name(
        f".{destination.name}.atomic-{os.getpid()}-{hashlib.sha256(value).hexdigest()[:12]}"
    )
    require_lstat_absent(temporary)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if overwrite:
        os.replace(temporary, destination)
    else:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise
        finally:
            temporary.unlink(missing_ok=True)
    _fsync_directory(destination.parent)


def write_json(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> None:
    atomic_write_bytes(path, canonical_bytes(value), overwrite=overwrite)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_relative(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError(f"unsafe manifest path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe manifest path: {value!r}")
    normalized = pure.as_posix()
    if normalized != value:
        raise ValueError(f"non-canonical manifest path: {value!r}")
    return normalized


def ensure_confined_regular_file(root: str | Path, relative: str) -> Path:
    base = _real_directory_root(root)
    safe = _safe_relative(relative)
    candidate = base / safe
    current = base
    for part in PurePosixPath(safe).parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink forbidden: {current}")
    resolved_parent = candidate.parent.resolve(strict=True)
    if resolved_parent != base and base not in resolved_parent.parents:
        raise RuntimeError(f"path escapes root: {relative}")
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"regular file required: {candidate}")
    return candidate


def read_confined_stable_regular_bytes(
    root: str | Path,
    relative: str,
) -> tuple[bytes, os.stat_result]:
    """Read a confined regular file through pinned, no-follow descriptors.

    Every path component is opened relative to the already-open parent
    directory.  This prevents an ancestor from being exchanged for a symlink
    between a pathname check and the file read.
    """
    base = _real_directory_root(root)
    safe = _safe_relative(relative)
    parts = PurePosixPath(safe).parts
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    try:
        current = os.open(base, directory_flags)
        descriptors.append(current)
        root_metadata = os.fstat(current)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise PermissionError("confined root descriptor is not a directory")
        for part in parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
            metadata = os.fstat(current)
            if not stat.S_ISDIR(metadata.st_mode):
                raise PermissionError("confined ancestor is not a directory")
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError("confined member is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, name) != getattr(after, name)
            for name in stable_fields
        ):
            raise InterruptedError("confined member changed during descriptor read")
        data = b"".join(chunks)
        if len(data) != int(before.st_size):
            raise InterruptedError("confined member size changed during read")
        return data, before
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _file_row(root: Path, relative: str) -> dict[str, Any]:
    data, metadata = read_confined_stable_regular_bytes(root, relative)
    return {
        "path": relative,
        "bytes": int(metadata.st_size),
        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def manifest_for_paths(root: str | Path, relatives: Iterable[str]) -> dict[str, Any]:
    base = _real_directory_root(root)
    values = list(map(str, relatives))
    if len(values) != len(set(values)):
        raise ValueError("duplicate manifest paths")
    paths = sorted(_safe_relative(value) for value in values)
    rows = [_file_row(base, relative) for relative in paths]
    return {
        "schema_version": "nursery-exact-file-manifest-v1",
        "file_count": len(rows),
        "files": rows,
        "digest_scheme": "canonical-json-of-sorted-file-rows",
        "digest": canonical_digest(rows),
    }


def tree_file_paths(
    root: str | Path,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    base = _real_directory_root(root)
    excluded = set(map(str, exclude))
    output: list[str] = []
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink forbidden in manifested tree: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"special file forbidden in manifested tree: {relative}")
        if relative not in excluded:
            output.append(relative)
    return output


def tree_directory_paths(root: str | Path) -> list[str]:
    base = _real_directory_root(root)
    output: list[str] = []
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink forbidden in manifested tree: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            output.append(relative)
        elif not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"special file forbidden in manifested tree: {relative}")
    return output


def manifest_for_tree(
    root: str | Path,
    *,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    return manifest_for_paths(root, tree_file_paths(root, exclude=exclude))


def verify_exact_manifest(
    root: str | Path,
    manifest: Mapping[str, Any],
    *,
    manifest_filename: str | None = None,
    allowed_extra_files: Iterable[str] = (),
) -> dict[str, Any]:
    base = _real_directory_root(root)
    problems: list[str] = []
    if set(manifest) != {
        "schema_version",
        "file_count",
        "files",
        "digest_scheme",
        "digest",
    }:
        problems.append("top_level_schema")
    if manifest.get("schema_version") != "nursery-exact-file-manifest-v1":
        problems.append("schema_version")
    if manifest.get("digest_scheme") != "canonical-json-of-sorted-file-rows":
        problems.append("digest_scheme")
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list):
        return {"status": "FAIL", "problems": ["files_not_list"]}
    paths: list[str] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "mode", "sha256"}:
            problems.append(f"row_schema:{index}")
            continue
        try:
            paths.append(_safe_relative(str(row["path"])))
        except ValueError:
            problems.append(f"unsafe_path:{index}")
    if paths != sorted(paths):
        problems.append("rows_not_sorted")
    if len(paths) != len(set(paths)):
        problems.append("duplicate_paths")
    if int(manifest.get("file_count", -1)) != len(raw_rows):
        problems.append("file_count")
    if manifest.get("digest") != canonical_digest(raw_rows):
        problems.append("manifest_digest")
    observed_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping) or "path" not in row:
            continue
        try:
            observed = _file_row(base, _safe_relative(str(row["path"])))
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
            problems.append(f"unreadable:{row.get('path')}:{type(error).__name__}")
            continue
        observed_rows.append(observed)
        if observed != dict(row):
            problems.append(f"row_mismatch:{row['path']}")
    excluded = set(map(str, allowed_extra_files))
    if manifest_filename is not None:
        excluded.add(manifest_filename)
    try:
        actual = set(tree_file_paths(base, exclude=excluded))
    except (RuntimeError, OSError) as error:
        problems.append(f"tree:{type(error).__name__}:{error}")
        actual = set()
    listed = set(paths)
    missing = sorted(listed - actual)
    unexpected = sorted(actual - listed)
    if missing:
        problems.append("missing_files")
    if unexpected:
        problems.append("unexpected_files")
    expected_directories = {
        PurePosixPath(path).parent.as_posix()
        for path in [*paths, *excluded]
        if PurePosixPath(path).parent.as_posix() != "."
    }
    expected_directories |= {
        parent.as_posix()
        for path in [*paths, *excluded]
        for parent in PurePosixPath(path).parents
        if parent.as_posix() not in {".", ""}
    }
    try:
        actual_directories = set(tree_directory_paths(base))
    except (RuntimeError, OSError) as error:
        problems.append(f"directories:{type(error).__name__}:{error}")
        actual_directories = set()
    unexpected_directories = sorted(actual_directories - expected_directories)
    if unexpected_directories:
        problems.append("unexpected_directories")
    return {
        "status": "PASS" if not problems else "FAIL",
        "file_count": len(raw_rows),
        "unique_path_count": len(set(paths)),
        "observed_row_count": len(observed_rows),
        "missing": missing,
        "unexpected": unexpected,
        "unexpected_directories": unexpected_directories,
        "problems": problems,
        "digest": manifest.get("digest"),
    }


def verify_consumed_package_transition(
    root: str | Path,
    pristine_manifest: Mapping[str, Any],
    *,
    pristine_authorization_relative: str,
    consumed_authorization_relative: str,
    claim_relative: str,
    receipt_relative: str,
    manifest_filename: str,
) -> dict[str, Any]:
    """Verify the only permitted post-launch-authority package transition.

    The frozen package manifest remains immutable.  Verification derives a
    transition manifest from it by (1) relocating the byte-identical
    authorization, and (2) changing the exact zero-state outcome registry's
    consumption bit from false to true.  The claim and issuance receipt are
    the only additional files.  All other frozen rows and directories remain
    exact.
    """
    base = _real_directory_root(root)
    original = _safe_relative(pristine_authorization_relative)
    consumed = _safe_relative(consumed_authorization_relative)
    claim = _safe_relative(claim_relative)
    receipt = _safe_relative(receipt_relative)
    manifest_name = _safe_relative(manifest_filename)
    if len({original, consumed, claim, receipt, manifest_name}) != 5:
        return {"status": "FAIL", "problems": ["transition_paths_not_unique"]}
    if set(pristine_manifest) != {
        "schema_version",
        "file_count",
        "files",
        "digest_scheme",
        "digest",
    }:
        return {"status": "FAIL", "problems": ["pristine_manifest_schema"]}
    raw_rows = pristine_manifest.get("files")
    if not isinstance(raw_rows, list):
        return {"status": "FAIL", "problems": ["pristine_rows_not_list"]}
    pristine_paths: list[str] = []
    try:
        for row in raw_rows:
            if not isinstance(row, Mapping) or set(row) != {
                "path",
                "bytes",
                "mode",
                "sha256",
            }:
                raise ValueError("row_schema")
            pristine_paths.append(_safe_relative(str(row["path"])))
    except (TypeError, ValueError) as error:
        return {
            "status": "FAIL",
            "problems": [f"pristine_row_validation:{error}"],
        }
    pristine_problems = []
    if pristine_manifest.get("schema_version") != "nursery-exact-file-manifest-v1":
        pristine_problems.append("pristine_schema_version")
    if (
        pristine_manifest.get("digest_scheme")
        != "canonical-json-of-sorted-file-rows"
    ):
        pristine_problems.append("pristine_digest_scheme")
    if int(pristine_manifest.get("file_count", -1)) != len(raw_rows):
        pristine_problems.append("pristine_file_count")
    if pristine_manifest.get("digest") != canonical_digest(raw_rows):
        pristine_problems.append("pristine_digest")
    if pristine_paths != sorted(pristine_paths):
        pristine_problems.append("pristine_rows_not_sorted")
    if len(pristine_paths) != len(set(pristine_paths)):
        pristine_problems.append("pristine_duplicate_paths")
    if original not in pristine_paths or consumed in pristine_paths:
        pristine_problems.append("pristine_authorization_inventory")
    if pristine_problems:
        return {"status": "FAIL", "problems": pristine_problems}
    expected_registry = {
        "schema_version": "nursery-corrective-outcome-registry-v4",
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "development_output_exists": False,
        "confirmation_output_exists": False,
        "authorization_consumed": True,
    }
    try:
        if read_json(base / "outcome_registry.json") != expected_registry:
            return {
                "status": "FAIL",
                "problems": ["consumed_outcome_registry_not_exact"],
            }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "FAIL",
            "problems": [f"consumed_outcome_registry:{type(error).__name__}"],
        }
    registry_bytes = canonical_bytes(expected_registry)
    transitioned_rows: list[dict[str, Any]] = []
    original_count = 0
    registry_count = 0
    for raw_row in raw_rows:
        row = dict(raw_row)
        path = str(row["path"])
        if path == original:
            original_count += 1
            row["path"] = consumed
        elif path == "outcome_registry.json":
            registry_count += 1
            row["bytes"] = len(registry_bytes)
            row["sha256"] = hashlib.sha256(registry_bytes).hexdigest()
        transitioned_rows.append(row)
    if original_count != 1 or registry_count != 1:
        return {
            "status": "FAIL",
            "problems": ["transition_source_rows_not_unique"],
        }
    transitioned_rows.sort(key=lambda row: str(row["path"]))
    transitioned_manifest = {
        "schema_version": "nursery-exact-file-manifest-v1",
        "file_count": len(transitioned_rows),
        "files": transitioned_rows,
        "digest_scheme": "canonical-json-of-sorted-file-rows",
        "digest": canonical_digest(transitioned_rows),
    }
    verification = verify_exact_manifest(
        base,
        transitioned_manifest,
        manifest_filename=manifest_name,
        allowed_extra_files=[claim, receipt],
    )
    return {
        **verification,
        "transition_manifest_digest": transitioned_manifest["digest"],
        "pristine_manifest_digest": pristine_manifest.get("digest"),
        "transition_kind": "authorization_relocated_and_consumption_bit_set",
    }


def reject_forbidden_fields(
    value: Any,
    forbidden: set[str] | frozenset[str],
    *,
    path: str = "visible",
) -> None:
    if isinstance(value, Mapping):
        overlap = set(value) & set(forbidden)
        if overlap:
            raise ValueError(f"forbidden fields at {path}: {sorted(overlap)}")
        for key, child in value.items():
            reject_forbidden_fields(child, forbidden, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            reject_forbidden_fields(child, forbidden, path=f"{path}[{index}]")


def _flatten_registry(registry: Mapping[str, Sequence[int]]) -> set[int]:
    return {int(value) for role in ROLES for value in registry[role]}


def _in_ranges(value: int, ranges: Sequence[Sequence[int]]) -> bool:
    return any(int(low) <= int(value) <= int(high) for low, high in ranges)


def load_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if value.get("protocol", {}).get("id") != PROTOCOL_ID:
        raise ValueError("wrong corrective protocol id")
    if value["protocol"].get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong corrective schema version")
    if int(value["protocol"].get("package_version", -1)) != 4:
        raise ValueError("wrong corrective package version")
    if value["protocol"].get("status") not in {"pre_freeze", "frozen"}:
        raise ValueError("status must be pre_freeze or frozen")
    protocol_false_flags = (
        "infant_learning_claim_authorized",
        "ecological_validity_claim_authorized",
        "development_execution_authorized",
        "confirmation_authorized",
        "prior_outcomes_authorized",
    )
    if any(value["protocol"].get(name) is not False for name in protocol_false_flags):
        raise ValueError("protocol authorization flags must remain false")
    expected_path_values = {
        "package_root": "output/synthetic_corrective_development_launch_package_v4",
        "frozen_snapshot": "output/synthetic_corrective_development_launch_package_v4/frozen_source_snapshot",
        "excluded_rehearsal": "output/synthetic_corrective_development_launch_package_v4/excluded_rehearsal",
        "excluded_rehearsal_attempt_receipt": "output/synthetic_corrective_development_launch_package_v4/excluded_rehearsal_attempt_consumed.json",
        "excluded_rehearsal_completion": "output/synthetic_corrective_development_launch_package_v4/excluded_rehearsal_completion.json",
        "independent_recompute": "output/synthetic_corrective_development_launch_package_v4/independent_recompute",
        "launch_seal": "output/synthetic_corrective_development_launch_package_v4/launch_seal",
        "launch_seal_staging": "output/.synthetic_corrective_development_launch_seal_v4.staging",
        "authorization": "output/synthetic_corrective_development_launch_package_v4/launch_seal/CORRECTIVE_DEVELOPMENT_LAUNCH_READY.json",
        "authorization_consumption_root": "output/synthetic_corrective_development_launch_package_v4/development_authorization_consumed",
        "consumed_authorization": "output/synthetic_corrective_development_launch_package_v4/development_authorization_consumed/authorization.json",
        "authorization_claim": "output/synthetic_corrective_development_launch_package_v4/development_authorization_consumed/claim.json",
        "authorization_capability_receipt": "output/synthetic_corrective_development_launch_package_v4/development_authorization_consumed/capability.json",
        "development_output": "output/synthetic_corrective_development_v4_one_shot",
        "development_staging": "output/.synthetic_corrective_development_v4.staging",
        "development_inner_staging": "output/..synthetic_corrective_development_v4.staging.cohort-staging",
        "development_publication_seal": "output/synthetic_corrective_development_v4_one_shot/SCIENTIFIC_OUTCOME.json",
        "preservation_baseline": "output/synthetic_confirmation_adversarial_audit_v1/preservation/baseline.sha256",
        "preservation_stat": "output/synthetic_confirmation_adversarial_audit_v1/preservation/baseline.stat",
        "preservation_symlinks": "output/synthetic_confirmation_adversarial_audit_v1/preservation/baseline.symlinks",
    }
    path_values = value.get("paths")
    if not isinstance(path_values, Mapping) or any(
        path_values.get(name) != expected
        for name, expected in expected_path_values.items()
    ):
        raise ValueError("corrective path layout differs from the exact v4 contract")
    for name, expected in expected_path_values.items():
        if _safe_relative(str(path_values[name])) != expected:
            raise ValueError(f"unsafe or non-canonical corrective path: {name}")
    if len(set(expected_path_values.values())) != len(expected_path_values):
        raise RuntimeError("internal corrective path contract contains aliases")
    design = value["design"]
    primitives = int(design["primitive_concepts"])
    manners = int(design["manner_concepts"])
    if (
        int(design["action_compositions"]) != primitives * manners
        or int(design["held_out_compositions_per_corpus"]) != primitives
        or list(design["primary_comparators"]) != ["absent", "shuffled"]
        or design["exact_window_comparator"] != "exact_window"
        or design["evaluation"].get("side_modality_withheld") is not True
        or design["evaluation"].get("lexical_concept_holdout_definition")
        != "independently_generated_unseen_instances_for_every_corpus_specific_exposed_concept"
        or design["evaluation"].get("atomic_concept_category_withheld") is not False
        or design["evaluation"].get("zero_shot_atomic_concept_claim_authorized")
        is not False
    ):
        raise ValueError("frozen design-count/comparator contract mismatch")
    analysis = value["analysis"]
    if (
        analysis.get("comparator_definition")
        != "per_corpus_max_of_absent_and_shuffled"
        or analysis.get("all_identical_effects_fail") is not True
        or analysis.get("independent_unit") != "corpus_seed"
    ):
        raise ValueError("frozen analysis contract mismatch")
    parallel = value["parallel"]
    required_child_allowlist = [
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "__CF_USER_TEXT_ENCODING",
        "KMP_DUPLICATE_LIB_OK",
    ]
    required_forbidden_environment = [
        "NPY_DISABLE_CPU_FEATURES",
        "NPY_ENABLE_CPU_FEATURES",
        "OPENBLAS_CORETYPE",
        "GOTO_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_SCHEDULE",
        "KMP_AFFINITY",
        "KMP_BLOCKTIME",
        "MKL_CBWR",
        "BLIS_ARCH_TYPE",
        "PYTHONOPTIMIZE",
        "PYTHONWARNINGS",
        "PYTHONMALLOC",
        "PYTHONPATH",
        "PYTHONHOME",
    ]
    if (
        parallel.get("backend") != "direct_subprocess_v1"
        or parallel.get("wave_key") != "model_seed"
        or parallel.get("worker_entrypoint")
        != "scripts/run_synthetic_corrective_alignment_v4_worker.py"
        or parallel.get("subprocess_new_session") is not True
        or parallel.get("worker_requests_manifested") is not True
        or parallel.get("child_inherited_environment_allowlist")
        != required_child_allowlist
        or parallel.get("forbidden_unbound_environment")
        != required_forbidden_environment
        or parallel.get("thread_environment", {}).get("OMP_DYNAMIC") != "FALSE"
        or parallel.get("thread_environment", {}).get("MKL_DYNAMIC") != "FALSE"
        or float(parallel.get("wall_time_contingency_multiplier", -1.0)) != 1.20
        or float(
            parallel.get("minimum_capability_publication_allowance_seconds", -1.0)
        )
        != 60.0
        or parallel.get("parent_memory_projection_method")
        != "all_results_full_micro_peak_scaled_by_unit_ratio"
        or float(parallel.get("maximum_current_available_ram_fraction", -1.0))
        != 0.85
        or float(parallel.get("minimum_benchmark_speedup", -1.0)) != 1.20
        or float(parallel.get("resource_contingency_multiplier", -1.0)) != 1.50
        or float(parallel.get("maximum_host_ram_fraction", -1.0)) != 0.50
        or float(parallel.get("maximum_current_free_disk_fraction", -1.0))
        != 0.50
        or int(parallel.get("minimum_logical_cpu_count", -1)) != 4
        or "start_method" in parallel
    ):
        raise ValueError("frozen direct-subprocess backend contract mismatch")
    if any(
        parallel.get(name) is not True
        for name in (
            "one_inflight_model_per_corpus",
            "worker_isolated_shards",
            "worker_isolated_tmp",
            "atomic_final_publish",
            "require_jobs_1_jobs_n_byte_identity",
        )
    ):
        raise ValueError("frozen parallel safety flags must be true")
    expected_parent_flags = {
        "bytes_warning": 0,
        "debug": 0,
        "dev_mode": False,
        "dont_write_bytecode": 1,
        "hash_randomization": 0,
        "ignore_environment": 0,
        "inspect": 0,
        "int_max_str_digits": 4300,
        "interactive": 0,
        "isolated": 0,
        "n_fields": 18,
        "n_sequence_fields": 18,
        "n_unnamed_fields": 0,
        "no_site": 0,
        "no_user_site": 0,
        "optimize": 0,
        "quiet": 0,
        "safe_path": False,
        "utf8_mode": 0,
        "verbose": 0,
        "warn_default_encoding": 0,
    }
    expected_worker_flags = {
        **expected_parent_flags,
        "no_user_site": 1,
        "safe_path": True,
    }
    environment = value["environment"]
    if (
        environment.get("parent_python_startup_flags") != expected_parent_flags
        or environment.get("worker_python_startup_flags") != expected_worker_flags
        or float(environment.get("worker_launcher_probe_timeout_seconds", -1.0))
        != 30.0
        or environment.get("worker_launcher_contract")
        != {
            "invoke_configured_lexical_path": True,
            "bind_symlink_chain": True,
            "bind_resolved_target_sha256": True,
            "bind_pyvenv_cfg_sha256": True,
            "bind_sys_prefixes": True,
            "real_import_probe_required_before_micro": True,
            "standard_library_bootstrap_failure_receipt": True,
        }
    ):
        raise ValueError("frozen Python launcher/startup contract mismatch")
    firewall = value["firewalls"]
    if (
        firewall.get("confirmation_command_exists") is not False
        or firewall.get("confirmation_execution_supported") is not False
        or any(
            firewall.get(name) is not True
            for name in (
                "development_output_must_be_pristine",
                "authorization_consumption_required",
                "authorization_replay_forbidden",
                "exact_manifest_file_set_required",
                "symlinks_forbidden_in_package_snapshot_shards_and_outputs",
            )
        )
    ):
        raise ValueError("frozen firewall contract mismatch")
    if value["environment"].get("bytecode_writes_disabled") is not True:
        raise ValueError("bytecode writes must be disabled")
    preservation_hash_fields = (
        "preservation_baseline_sha256",
        "preservation_stat_sha256",
        "preservation_symlinks_sha256",
        "preservation_audit_manifest_sha256",
    )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value["paths"].get(name, "")))
        for name in preservation_hash_fields
    ):
        raise ValueError("preservation anchors must be lowercase SHA-256 digests")
    qualification = value["qualification_gates"]
    if (
        qualification.get("exact_present_null_effect_vector_equality_forbidden")
        is not True
        or float(qualification["presence_null_effect_equality_tolerance"]) <= 0.0
    ):
        raise ValueError("present/null dependence contract mismatch")
    development_validity = value.get("development_validity_gates")
    if development_validity != {
        "minimum_distinct_factor_draws": 75,
        "minimum_corpora_per_ambiguity_stratum": 25,
        "factor_draw_reference_fraction": 0.75,
        "require_factor_draws_within_frozen_ranges": True,
        "require_trace_state_digest_binding": True,
        "require_variation_values_bound_to_averages": True,
    }:
        raise ValueError("operational development-validity contract mismatch")
    if int(development_validity["minimum_distinct_factor_draws"]) != math.ceil(
        float(development_validity["factor_draw_reference_fraction"])
        * int(value["analysis"]["development_corpus_count"])
    ):
        raise ValueError("development factor-draw threshold scaling mismatch")
    registries = {
        name: {
            role: list(map(int, value["registries"][name][role]))
            for role in ROLES
        }
        for name in PURPOSES + ("confirmation_reserve",)
    }
    for name, registry in registries.items():
        for role, identifiers in registry.items():
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate identifiers within {name}/{role}")
        flattened = [value for role in ROLES for value in registry[role]]
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"identifier reused across roles within {name}")
    sets = {name: _flatten_registry(registry) for name, registry in registries.items()}
    names = list(sets)
    overlaps = {
        f"{left}|{right}": sorted(sets[left] & sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if sets[left] & sets[right]
    }
    if overlaps:
        raise ValueError(f"registry overlap: {overlaps}")
    family_low, family_high = map(int, value["registries"]["namespace_family"])
    all_active = set().union(*sets.values())
    if any(not family_low <= seed <= family_high for seed in all_active):
        raise ValueError("active identifier outside corrective namespace")
    quarantine = value["registries"]["permanently_quarantined_ranges"]
    if any(_in_ranges(seed, quarantine) for seed in all_active):
        raise ValueError("quarantined identifier allocated")
    if len(registries["development"]["corpus"]) != int(
        value["analysis"]["development_corpus_count"]
    ):
        raise ValueError("development corpus count mismatch")
    if len(registries["development"]["model"]) != int(
        value["analysis"]["paired_model_replicates"]
    ):
        raise ValueError("development model count mismatch")
    if len(registries["construction_mechanism"]["model"]) != int(
        value["qualification_gates"]["disagreement_case_count"]
    ):
        raise ValueError("construction mechanism model count mismatch")
    if len(registries["confirmation_reserve"]["corpus"]) != int(
        value["analysis"]["development_corpus_count"]
    ):
        raise ValueError("confirmation corpus count mismatch")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.parents[1]
    )
    scientific_core = value.get("scientific_core")
    expected_scientific_sources = {
        "babyworld_lite/corrective_alignment_v4/generator.py": (
            "d3dbeb18ba04d7ff6e3f29c8dc27b425fbe45483f6df64b95444916d9fd413a5"
        ),
        "babyworld_lite/corrective_alignment_v4/learner.py": (
            "6fd754d220b10c3e39d64a72b817217537a2abf8ffd839418cb27b4ed7e1d67e"
        ),
        "babyworld_lite/corrective_alignment_v4/statistics.py": (
            "dd33d270cf9b1ad0eb5a0388509a6779492ce00b603806a09cbdc06ecced5a1b"
        ),
        "babyworld_lite/corrective_alignment_v4/adjudicator.py": (
            "dba87d6776ed81438de8c869052adfe1be5008857b0cdb6f41a45d84d86d56a8"
        ),
        "babyworld_lite/corrective_alignment_v4/parallel.py": (
            "fa6cce5caf52b620c124b35137402a4ee92216dc4522414855d11d1959fe0adc"
        ),
    }
    scientific_sections = [
        "design",
        "learner",
        "analysis",
        "qualification_gates",
        "development_validity_gates",
    ]
    expected_scientific_projection = (
        "393d225c848de186d6eb2a8f84761172a9ec683b3277b25ba1c37311161a8819"
    )
    expected_claim_and_leakage = (
        "56c814c398fe98eac81849f117c72765997424f5618e971605218c3d9761c39c"
    )
    if scientific_core != {
        "scientific_protocol_version": 3,
        "immediate_predecessor_operational_package_version": 3,
        "predecessor_scientific_protocol_version": 2,
        "vendored_byte_identically": False,
        "revision_status": "prospective_construct_corrective_revision",
        "projection_sections": scientific_sections,
        "projection_sha256": expected_scientific_projection,
        "claim_and_leakage_sha256": expected_claim_and_leakage,
        "source_sha256": expected_scientific_sources,
    }:
        raise ValueError("scientific-v3 revision contract mismatch")
    projection = {name: value[name] for name in scientific_sections}
    if canonical_digest(projection) != expected_scientific_projection:
        raise ValueError("scientific-v3 config projection changed")
    claim_and_leakage = {
        "protocol": {
            name: value["protocol"][name]
            for name in (
                "scientific_claim",
                "infant_learning_claim_authorized",
                "ecological_validity_claim_authorized",
                "prior_outcomes_authorized",
            )
        },
        "firewalls": value["firewalls"],
    }
    if canonical_digest(claim_and_leakage) != expected_claim_and_leakage:
        raise ValueError("scientific-v3 claim or leakage firewall changed")
    for relative, expected_digest in expected_scientific_sources.items():
        if sha256_file(ensure_confined_regular_file(root, relative)) != expected_digest:
            raise ValueError(f"scientific-v3 source changed: {relative}")
    prior = root / str(value["registries"]["canonical_prior_registry"])
    if sha256_file(prior) != str(
        value["registries"]["canonical_prior_registry_sha256"]
    ):
        raise ValueError("canonical prior registry hash mismatch")
    prior_value = read_json(prior)
    prior_identifiers: set[int] = set()

    def collect_identifiers(child: Any) -> None:
        if isinstance(child, Mapping):
            for grandchild in child.values():
                collect_identifiers(grandchild)
        elif isinstance(child, list):
            for grandchild in child:
                if isinstance(grandchild, int) and not isinstance(grandchild, bool):
                    prior_identifiers.add(int(grandchild))
                else:
                    collect_identifiers(grandchild)

    collect_identifiers(prior_value.get("registries", {}))
    if all_active & prior_identifiers:
        raise ValueError("corrective identifiers overlap canonical prior registry")
    value["prior_identifier_count"] = len(prior_identifiers)
    value["prior_identifier_digest"] = canonical_digest(sorted(prior_identifiers))
    value["resolved_registries"] = registries
    value["repository_root"] = str(root)
    return value


def require_frozen(config: Mapping[str, Any]) -> None:
    if config["protocol"]["status"] != "frozen":
        raise RuntimeError("corrective package is not frozen")


_CAPABILITY_FACTORY_SEAL = object()
_PREFLIGHT_PROOF_FACTORY_SEAL = object()


class DevelopmentPreflightProof:
    __slots__ = (
        "authorization_digest",
        "authorization_path",
        "authorization_sha256",
        "snapshot_root",
        "snapshot_manifest_sha256",
        "package_complete_manifest_path",
        "package_complete_manifest_sha256",
        "parsed_contract_digest",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "claim_path",
        "issuance_receipt_path",
        "consumption_root",
        "consumed_authorization_path",
        "outcome_registry_path",
        "required_jobs",
        "argv_digest",
        "environment_digest",
        "_identity_digest",
        "_seal",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("development preflight proofs are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        authorization_digest: str,
        authorization_path: str | Path,
        authorization_sha256: str,
        snapshot_root: str | Path,
        snapshot_manifest_sha256: str,
        package_complete_manifest_path: str | Path,
        package_complete_manifest_sha256: str,
        parsed_contract_digest: str,
        resolved_output_root: str | Path,
        resolved_staging_root: str | Path,
        resolved_inner_staging_root: str | Path,
        claim_path: str | Path,
        issuance_receipt_path: str | Path,
        consumption_root: str | Path,
        consumed_authorization_path: str | Path,
        outcome_registry_path: str | Path,
        required_jobs: int,
        argv_digest: str,
        environment_digest: str,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _PREFLIGHT_PROOF_FACTORY_SEAL:
            raise PermissionError("preflight proof requires strict verified issuance")
        values = {
            "authorization_digest": str(authorization_digest),
            "authorization_path": str(Path(authorization_path).resolve()),
            "authorization_sha256": str(authorization_sha256),
            "snapshot_root": str(Path(snapshot_root).resolve()),
            "snapshot_manifest_sha256": str(snapshot_manifest_sha256),
            "package_complete_manifest_path": str(
                Path(package_complete_manifest_path).resolve()
            ),
            "package_complete_manifest_sha256": str(
                package_complete_manifest_sha256
            ),
            "parsed_contract_digest": str(parsed_contract_digest),
            "resolved_output_root": str(Path(resolved_output_root).resolve()),
            "resolved_staging_root": str(Path(resolved_staging_root).resolve()),
            "resolved_inner_staging_root": str(
                Path(resolved_inner_staging_root).resolve()
            ),
            "claim_path": str(Path(claim_path).resolve()),
            "issuance_receipt_path": str(Path(issuance_receipt_path).resolve()),
            "consumption_root": str(Path(consumption_root).resolve()),
            "consumed_authorization_path": str(
                Path(consumed_authorization_path).resolve()
            ),
            "outcome_registry_path": str(Path(outcome_registry_path).resolve()),
            "required_jobs": int(required_jobs),
            "argv_digest": str(argv_digest),
            "environment_digest": str(environment_digest),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_identity_digest", canonical_digest(values))
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_frozen", True)


def _issue_development_preflight_proof(**values: Any) -> DevelopmentPreflightProof:
    return DevelopmentPreflightProof(
        **values,
        _seal=_PREFLIGHT_PROOF_FACTORY_SEAL,
    )


def _verify_development_preflight_proof(
    proof: DevelopmentPreflightProof,
) -> None:
    if (
        not isinstance(proof, DevelopmentPreflightProof)
        or proof._seal is not _PREFLIGHT_PROOF_FACTORY_SEAL
        or proof._frozen is not True
    ):
        raise PermissionError("development capability requires a strict preflight proof")
    values = {
        name: getattr(proof, name)
        for name in DevelopmentPreflightProof.__slots__
        if not name.startswith("_")
    }
    if canonical_digest(values) != proof._identity_digest:
        raise PermissionError("development preflight proof identity changed")
    if _lstat_exists(Path(proof.authorization_path)):
        raise PermissionError("original authorization still exists after consumption")
    consumption_root = Path(proof.consumption_root)
    metadata = consumption_root.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError("authorization consumption root is not a real directory")
    if (
        sha256_file(proof.consumed_authorization_path)
        != proof.authorization_sha256
    ):
        raise PermissionError("consumed authorization differs from preflight authority")
    registry = read_json(proof.outcome_registry_path)
    if (
        registry.get("authorization_consumed") is not True
        or int(registry.get("development_outcome_count", -1)) != 0
        or int(registry.get("confirmation_outcome_count", -1)) != 0
    ):
        raise PermissionError("authorization consumption registry transition missing")
    snapshot_manifest = Path(proof.snapshot_root) / "snapshot_manifest.json"
    if sha256_file(snapshot_manifest) != proof.snapshot_manifest_sha256:
        raise PermissionError("preflight-bound snapshot manifest changed")
    if (
        sha256_file(proof.package_complete_manifest_path)
        != proof.package_complete_manifest_sha256
    ):
        raise PermissionError("preflight-bound package manifest changed")


class AuthorizationCapability:
    __slots__ = (
        "purpose",
        "scope",
        "unit_id",
        "authorization_digest",
        "parsed_contract_digest",
        "claim_path",
        "claim_sha256",
        "issuance_receipt_path",
        "issuance_receipt_sha256",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "required_jobs",
        "worker_launch_secret_sha256",
        "_worker_launch_secret",
        "_identity_digest",
        "_seal",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("authorization capabilities are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        purpose: str,
        scope: str,
        unit_id: str | None,
        authorization_digest: str,
        parsed_contract_digest: str,
        claim_path: str,
        claim_sha256: str,
        issuance_receipt_path: str,
        issuance_receipt_sha256: str,
        resolved_output_root: str,
        resolved_staging_root: str,
        resolved_inner_staging_root: str,
        required_jobs: int,
        worker_launch_secret: bytes,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _CAPABILITY_FACTORY_SEAL:
            raise PermissionError(
                "development capability can only be issued from a consumed claim"
            )
        self.purpose = str(purpose)
        self.scope = str(scope)
        self.unit_id = None if unit_id is None else str(unit_id)
        self.authorization_digest = str(authorization_digest)
        self.parsed_contract_digest = str(parsed_contract_digest)
        self.claim_path = str(claim_path)
        self.claim_sha256 = str(claim_sha256)
        self.issuance_receipt_path = str(issuance_receipt_path)
        self.issuance_receipt_sha256 = str(issuance_receipt_sha256)
        self.resolved_output_root = str(resolved_output_root)
        self.resolved_staging_root = str(resolved_staging_root)
        self.resolved_inner_staging_root = str(resolved_inner_staging_root)
        self.required_jobs = int(required_jobs)
        if (
            not isinstance(worker_launch_secret, bytes)
            or len(worker_launch_secret) != 32
        ):
            raise PermissionError("worker launch capability secret is invalid")
        self.worker_launch_secret_sha256 = hashlib.sha256(
            worker_launch_secret
        ).hexdigest()
        self._worker_launch_secret = worker_launch_secret
        identity = {
            name: getattr(self, name)
            for name in AuthorizationCapability.__slots__
            if not name.startswith("_")
        }
        self._identity_digest = canonical_digest(identity)
        self._seal = _seal
        self._frozen = True


def issue_development_capability(
    *,
    authorization_digest: str,
    parsed_contract_digest: str,
    claim_path: str | Path,
    claim_sha256: str,
    resolved_output_root: str | Path,
    resolved_staging_root: str | Path,
    resolved_inner_staging_root: str | Path,
    issuance_receipt_path: str | Path,
    required_jobs: int,
    preflight_proof: DevelopmentPreflightProof,
) -> AuthorizationCapability:
    _verify_development_preflight_proof(preflight_proof)
    path = Path(claim_path).resolve()
    if sha256_file(path) != str(claim_sha256):
        raise PermissionError("authorization-consumption claim digest mismatch")
    claim = read_json(path)
    expected_fields = {
        "schema_version",
        "status",
        "authorization_digest",
        "original_authorization_path",
        "resolved_authorization_path",
        "authorization_sha256",
        "resolved_snapshot_root",
        "parsed_contract_digest",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "resolved_capability_receipt_path",
        "resolved_consumption_root",
        "pristine_package_complete_manifest_sha256",
        "required_jobs",
        "non_replayable",
        "created_before_any_guarded_development_operation",
        "development_outcome_count",
        "confirmation_outcome_count",
    }
    if set(claim) != expected_fields:
        raise PermissionError("authorization-consumption claim schema mismatch")
    if (
        claim.get("schema_version")
        != "nursery-corrective-authorization-consumption-v4"
        or claim.get("status") != "ATTEMPT_CONSUMED"
        or claim.get("authorization_digest") != str(authorization_digest)
        or claim.get("parsed_contract_digest") != str(parsed_contract_digest)
        or Path(str(claim.get("resolved_output_root"))).resolve()
        != Path(resolved_output_root).resolve()
        or Path(str(claim.get("resolved_staging_root"))).resolve()
        != Path(resolved_staging_root).resolve()
        or Path(str(claim.get("resolved_inner_staging_root"))).resolve()
        != Path(resolved_inner_staging_root).resolve()
        or Path(str(claim.get("resolved_capability_receipt_path"))).resolve()
        != Path(issuance_receipt_path).resolve()
        or Path(str(claim.get("resolved_consumption_root"))).resolve()
        != Path(preflight_proof.consumption_root)
        or claim.get("pristine_package_complete_manifest_sha256")
        != preflight_proof.package_complete_manifest_sha256
        or int(claim.get("required_jobs", -1)) != int(required_jobs)
        or claim.get("non_replayable") is not True
        or claim.get("created_before_any_guarded_development_operation") is not True
        or int(claim.get("development_outcome_count", -1)) != 0
        or int(claim.get("confirmation_outcome_count", -1)) != 0
    ):
        raise PermissionError("authorization-consumption claim content mismatch")
    if (
        preflight_proof.authorization_digest != str(authorization_digest)
        or preflight_proof.parsed_contract_digest != str(parsed_contract_digest)
        or Path(preflight_proof.claim_path) != path
        or Path(preflight_proof.issuance_receipt_path)
        != Path(issuance_receipt_path).resolve()
        or Path(preflight_proof.consumed_authorization_path)
        != Path(str(claim.get("resolved_authorization_path"))).resolve()
        or Path(preflight_proof.consumption_root)
        != Path(str(claim.get("resolved_consumption_root"))).resolve()
        or Path(preflight_proof.resolved_output_root)
        != Path(resolved_output_root).resolve()
        or Path(preflight_proof.resolved_staging_root)
        != Path(resolved_staging_root).resolve()
        or Path(preflight_proof.resolved_inner_staging_root)
        != Path(resolved_inner_staging_root).resolve()
        or preflight_proof.required_jobs != int(required_jobs)
    ):
        raise PermissionError("capability request differs from strict preflight proof")
    authorization_path = Path(str(claim["resolved_authorization_path"])).resolve()
    snapshot = Path(str(claim["resolved_snapshot_root"])).resolve()
    authorization = read_json(authorization_path)
    authorization_payload = {
        key: value for key, value in authorization.items() if key != "authorization_digest"
    }
    if (
        sha256_file(authorization_path) != str(claim["authorization_sha256"])
        or authorization.get("authorization_digest") != canonical_digest(authorization_payload)
        or authorization.get("authorization_digest") != str(authorization_digest)
        or authorization.get("status") != "CORRECTIVE_DEVELOPMENT_LAUNCH_READY"
        or authorization.get("development_authorized") is not True
        or authorization.get("confirmation_authorized") is not False
        or Path(str(claim.get("original_authorization_path"))).resolve()
        != Path(str(authorization.get("resolved_authorization_path"))).resolve()
        or authorization_path
        != Path(
            str(authorization.get("resolved_consumed_authorization_path"))
        ).resolve()
        or path != Path(str(authorization.get("resolved_claim_path"))).resolve()
        or Path(str(authorization.get("resolved_snapshot_root"))).resolve() != snapshot
        or Path(str(authorization.get("resolved_output_root"))).resolve()
        != Path(resolved_output_root).resolve()
        or Path(str(authorization.get("resolved_staging_root"))).resolve()
        != Path(resolved_staging_root).resolve()
        or Path(str(authorization.get("resolved_inner_staging_root"))).resolve()
        != Path(resolved_inner_staging_root).resolve()
        or Path(str(authorization.get("resolved_capability_receipt_path"))).resolve()
        != Path(issuance_receipt_path).resolve()
        or Path(str(authorization.get("resolved_consumption_root"))).resolve()
        != Path(str(claim.get("resolved_consumption_root"))).resolve()
        or int(authorization.get("required_jobs", -1)) != int(required_jobs)
    ):
        raise PermissionError("authorization artifact does not bind capability request")
    package = snapshot.parent
    if (
        snapshot.name != "frozen_source_snapshot"
        or package.name != "synthetic_corrective_development_launch_package_v4"
        or authorization_path.parent
        != Path(str(authorization.get("resolved_consumption_root"))).resolve()
        or authorization_path.parent.parent != package
    ):
        raise PermissionError("capability package/snapshot layout mismatch")
    config_path = snapshot / "configs/synthetic_corrective_alignment_v4.yaml"
    config = load_config(config_path, repository_root=snapshot)
    if (
        config["protocol"]["status"] != "frozen"
        or sha256_file(config_path) != authorization.get("required_config_sha256")
        or config["resolved_registries"]["development"]
        != authorization.get("development_registry")
    ):
        raise PermissionError("capability frozen configuration mismatch")
    receipt = Path(issuance_receipt_path).resolve()
    consumption_root = authorization_path.parent
    if (
        path.parent != consumption_root
        or receipt.parent != consumption_root
        or tree_directory_paths(consumption_root)
        or tree_file_paths(consumption_root)
        != sorted([authorization_path.name, path.name])
    ):
        raise PermissionError("authorization consumption tree is not pristine and exact")
    require_lstat_absent(receipt)
    worker_launch_secret = secrets.token_bytes(32)
    receipt_payload = {
        "schema_version": "nursery-corrective-capability-issuance-v4",
        "status": "PARENT_CAPABILITY_ISSUED",
        "authorization_digest": str(authorization_digest),
        "parsed_contract_digest": str(parsed_contract_digest),
        "claim_path": str(path),
        "claim_sha256": str(claim_sha256),
        "resolved_output_root": str(Path(resolved_output_root).resolve()),
        "resolved_staging_root": str(Path(resolved_staging_root).resolve()),
        "resolved_inner_staging_root": str(Path(resolved_inner_staging_root).resolve()),
        "required_jobs": int(required_jobs),
        "scope": "parent",
        "unit_id": None,
        "preflight_proof_identity_digest": preflight_proof._identity_digest,
        "parent_issuance_non_replayable": True,
        "worker_launch_secret_sha256": hashlib.sha256(
            worker_launch_secret
        ).hexdigest(),
    }
    receipt_value = {**receipt_payload, "receipt_digest": canonical_digest(receipt_payload)}
    write_json(receipt, receipt_value)
    if (
        tree_directory_paths(consumption_root)
        or tree_file_paths(consumption_root)
        != sorted([authorization_path.name, path.name, receipt.name])
    ):
        raise PermissionError("authorization consumption tree changed during issuance")
    receipt_sha256 = sha256_file(receipt)
    return AuthorizationCapability(
        purpose="development",
        scope="parent",
        unit_id=None,
        authorization_digest=str(authorization_digest),
        parsed_contract_digest=str(parsed_contract_digest),
        claim_path=str(path),
        claim_sha256=str(claim_sha256),
        issuance_receipt_path=str(receipt),
        issuance_receipt_sha256=receipt_sha256,
        resolved_output_root=str(Path(resolved_output_root).resolve()),
        resolved_staging_root=str(Path(resolved_staging_root).resolve()),
        resolved_inner_staging_root=str(Path(resolved_inner_staging_root).resolve()),
        required_jobs=int(required_jobs),
        worker_launch_secret=worker_launch_secret,
        _seal=_CAPABILITY_FACTORY_SEAL,
    )


def _issue_worker_development_capability(
    *,
    authorization_digest: str,
    parsed_contract_digest: str,
    claim_path: str | Path,
    claim_sha256: str,
    issuance_receipt_path: str | Path,
    issuance_receipt_sha256: str,
    resolved_output_root: str | Path,
    resolved_staging_root: str | Path,
    resolved_inner_staging_root: str | Path,
    required_jobs: int,
    unit_id: str,
    worker_launch_secret: bytes,
) -> AuthorizationCapability:
    claim = Path(claim_path).resolve()
    receipt = Path(issuance_receipt_path).resolve()
    if sha256_file(claim) != str(claim_sha256) or sha256_file(receipt) != str(
        issuance_receipt_sha256
    ):
        raise PermissionError("worker capability parent evidence changed")
    value = read_json(receipt)
    payload = {key: child for key, child in value.items() if key != "receipt_digest"}
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "authorization_digest",
            "parsed_contract_digest",
            "claim_path",
            "claim_sha256",
            "resolved_output_root",
            "resolved_staging_root",
            "resolved_inner_staging_root",
            "required_jobs",
            "scope",
            "unit_id",
            "preflight_proof_identity_digest",
            "parent_issuance_non_replayable",
            "worker_launch_secret_sha256",
            "receipt_digest",
        }
        or value.get("receipt_digest") != canonical_digest(payload)
        or value.get("status") != "PARENT_CAPABILITY_ISSUED"
        or value.get("authorization_digest") != str(authorization_digest)
        or value.get("parsed_contract_digest") != str(parsed_contract_digest)
        or Path(str(value.get("claim_path"))).resolve() != claim
        or value.get("claim_sha256") != str(claim_sha256)
        or Path(str(value.get("resolved_output_root"))).resolve()
        != Path(resolved_output_root).resolve()
        or Path(str(value.get("resolved_staging_root"))).resolve()
        != Path(resolved_staging_root).resolve()
        or Path(str(value.get("resolved_inner_staging_root"))).resolve()
        != Path(resolved_inner_staging_root).resolve()
        or int(value.get("required_jobs", -1)) != int(required_jobs)
        or value.get("scope") != "parent"
        or value.get("unit_id") is not None
        or not isinstance(value.get("preflight_proof_identity_digest"), str)
        or len(str(value.get("preflight_proof_identity_digest"))) != 64
        or value.get("parent_issuance_non_replayable") is not True
        or not isinstance(worker_launch_secret, bytes)
        or len(worker_launch_secret) != 32
        or value.get("worker_launch_secret_sha256")
        != hashlib.sha256(worker_launch_secret).hexdigest()
    ):
        raise PermissionError("worker capability issuance receipt mismatch")
    parent = AuthorizationCapability(
        purpose="development",
        scope="parent",
        unit_id=None,
        authorization_digest=str(authorization_digest),
        parsed_contract_digest=str(parsed_contract_digest),
        claim_path=str(claim),
        claim_sha256=str(claim_sha256),
        issuance_receipt_path=str(receipt),
        issuance_receipt_sha256=str(issuance_receipt_sha256),
        resolved_output_root=str(Path(resolved_output_root).resolve()),
        resolved_staging_root=str(Path(resolved_staging_root).resolve()),
        resolved_inner_staging_root=str(Path(resolved_inner_staging_root).resolve()),
        required_jobs=int(required_jobs),
        worker_launch_secret=worker_launch_secret,
        _seal=_CAPABILITY_FACTORY_SEAL,
    )
    verify_development_capability(
        parent,
        required_scope="parent",
        resolved_output_root=resolved_output_root,
        resolved_staging_root=resolved_staging_root,
        resolved_inner_staging_root=resolved_inner_staging_root,
        required_jobs=required_jobs,
        _full_integrity=False,
    )
    return AuthorizationCapability(
        purpose="development",
        scope="worker",
        unit_id=str(unit_id),
        authorization_digest=str(authorization_digest),
        parsed_contract_digest=str(parsed_contract_digest),
        claim_path=str(claim),
        claim_sha256=str(claim_sha256),
        issuance_receipt_path=str(receipt),
        issuance_receipt_sha256=str(issuance_receipt_sha256),
        resolved_output_root=str(Path(resolved_output_root).resolve()),
        resolved_staging_root=str(Path(resolved_staging_root).resolve()),
        resolved_inner_staging_root=str(Path(resolved_inner_staging_root).resolve()),
        required_jobs=int(required_jobs),
        worker_launch_secret=worker_launch_secret,
        _seal=_CAPABILITY_FACTORY_SEAL,
    )


def verify_development_capability(
    capability: AuthorizationCapability,
    *,
    resolved_output_root: str | Path | None = None,
    resolved_staging_root: str | Path | None = None,
    resolved_inner_staging_root: str | Path | None = None,
    required_jobs: int | None = None,
    required_scope: str | None = None,
    required_unit_id: str | None = None,
    _full_integrity: bool = True,
) -> None:
    if (
        not isinstance(capability, AuthorizationCapability)
        or capability._seal is not _CAPABILITY_FACTORY_SEAL
        or capability._frozen is not True
        or capability.purpose != "development"
    ):
        raise PermissionError("development requires an issued capability")
    identity = {
        name: getattr(capability, name)
        for name in AuthorizationCapability.__slots__
        if not name.startswith("_")
    }
    if canonical_digest(identity) != capability._identity_digest:
        raise PermissionError("development capability identity changed")
    if (
        not isinstance(capability._worker_launch_secret, bytes)
        or len(capability._worker_launch_secret) != 32
        or hashlib.sha256(capability._worker_launch_secret).hexdigest()
        != capability.worker_launch_secret_sha256
    ):
        raise PermissionError("development worker launch capability changed")
    if sha256_file(capability.claim_path) != capability.claim_sha256:
        raise PermissionError("consumed claim changed after capability issuance")
    if (
        sha256_file(capability.issuance_receipt_path)
        != capability.issuance_receipt_sha256
    ):
        raise PermissionError("capability issuance receipt changed")
    claim = read_json(capability.claim_path)
    expected_claim_fields = {
        "schema_version",
        "status",
        "authorization_digest",
        "original_authorization_path",
        "resolved_authorization_path",
        "authorization_sha256",
        "resolved_snapshot_root",
        "parsed_contract_digest",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "resolved_capability_receipt_path",
        "resolved_consumption_root",
        "pristine_package_complete_manifest_sha256",
        "required_jobs",
        "non_replayable",
        "created_before_any_guarded_development_operation",
        "development_outcome_count",
        "confirmation_outcome_count",
    }
    if (
        set(claim) != expected_claim_fields
        or claim.get("schema_version")
        != "nursery-corrective-authorization-consumption-v4"
        or claim.get("status") != "ATTEMPT_CONSUMED"
        or claim.get("authorization_digest") != capability.authorization_digest
        or claim.get("parsed_contract_digest") != capability.parsed_contract_digest
        or Path(str(claim.get("resolved_output_root"))).resolve()
        != Path(capability.resolved_output_root)
        or Path(str(claim.get("resolved_staging_root"))).resolve()
        != Path(capability.resolved_staging_root)
        or Path(str(claim.get("resolved_inner_staging_root"))).resolve()
        != Path(capability.resolved_inner_staging_root)
        or Path(str(claim.get("resolved_capability_receipt_path"))).resolve()
        != Path(capability.issuance_receipt_path)
        or Path(str(claim.get("resolved_consumption_root"))).resolve()
        != Path(capability.claim_path).parent
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(claim.get("pristine_package_complete_manifest_sha256", "")),
        )
        or int(claim.get("required_jobs", -1)) != capability.required_jobs
        or claim.get("non_replayable") is not True
        or claim.get("created_before_any_guarded_development_operation") is not True
        or int(claim.get("development_outcome_count", -1)) != 0
        or int(claim.get("confirmation_outcome_count", -1)) != 0
    ):
        raise PermissionError("capability differs from consumed claim")
    authorization_path = Path(str(claim["resolved_authorization_path"])).resolve()
    snapshot = Path(str(claim["resolved_snapshot_root"])).resolve()
    if (
        sha256_file(authorization_path) != str(claim["authorization_sha256"])
        or snapshot.name != "frozen_source_snapshot"
        or snapshot.parent.name
        != "synthetic_corrective_development_launch_package_v4"
    ):
        raise PermissionError("capability authority-chain path or digest mismatch")
    authorization = read_json(authorization_path)
    authorization_payload = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_digest"
    }
    if (
        authorization.get("authorization_digest")
        != canonical_digest(authorization_payload)
        or authorization.get("authorization_digest")
        != capability.authorization_digest
        or authorization.get("status")
        != "CORRECTIVE_DEVELOPMENT_LAUNCH_READY"
        or authorization.get("development_authorized") is not True
        or authorization.get("confirmation_authorized") is not False
        or Path(str(authorization.get("resolved_authorization_path"))).resolve()
        != Path(str(claim.get("original_authorization_path"))).resolve()
        or Path(
            str(authorization.get("resolved_consumed_authorization_path"))
        ).resolve()
        != authorization_path
        or Path(str(authorization.get("resolved_snapshot_root"))).resolve()
        != snapshot
        or Path(str(authorization.get("resolved_claim_path"))).resolve()
        != Path(capability.claim_path)
        or Path(str(authorization.get("resolved_capability_receipt_path"))).resolve()
        != Path(capability.issuance_receipt_path)
        or Path(str(authorization.get("resolved_consumption_root"))).resolve()
        != Path(capability.claim_path).parent
        or Path(str(authorization.get("resolved_output_root"))).resolve()
        != Path(capability.resolved_output_root)
        or Path(str(authorization.get("resolved_staging_root"))).resolve()
        != Path(capability.resolved_staging_root)
        or Path(str(authorization.get("resolved_inner_staging_root"))).resolve()
        != Path(capability.resolved_inner_staging_root)
        or int(authorization.get("required_jobs", -1))
        != capability.required_jobs
    ):
        raise PermissionError("capability authorization chain mismatch")
    consumption_root = Path(capability.claim_path).parent
    expected_consumption_files = sorted(
        [
            authorization_path.name,
            Path(capability.claim_path).name,
            Path(capability.issuance_receipt_path).name,
        ]
    )
    if (
        tree_directory_paths(consumption_root)
        or tree_file_paths(consumption_root) != expected_consumption_files
    ):
        raise PermissionError("capability consumption tree inventory mismatch")
    outcome_registry = read_json(snapshot.parent / "outcome_registry.json")
    expected_consumed_registry = {
        "schema_version": "nursery-corrective-outcome-registry-v4",
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "development_output_exists": False,
        "confirmation_output_exists": False,
        "authorization_consumed": True,
    }
    original_authorization_path = Path(
        str(claim["original_authorization_path"])
    ).resolve()
    complete_manifest_path = (
        snapshot.parent / "launch_seal/complete_file_manifest.json"
    )
    if (
        outcome_registry != expected_consumed_registry
        or _lstat_exists(original_authorization_path)
        or sha256_file(complete_manifest_path)
        != str(claim["pristine_package_complete_manifest_sha256"])
    ):
        raise PermissionError("capability outcome registry is not consumed-zero")
    config_path = snapshot / "configs/synthetic_corrective_alignment_v4.yaml"
    config = load_config(config_path, repository_root=snapshot)
    if (
        config["protocol"]["status"] != "frozen"
        or sha256_file(config_path)
        != str(authorization.get("required_config_sha256"))
        or config["resolved_registries"]["development"]
        != authorization.get("development_registry")
        or sha256_file(snapshot / "snapshot_manifest.json")
        != str(authorization.get("required_snapshot_manifest_sha256"))
    ):
        raise PermissionError("capability frozen snapshot chain mismatch")
    if _full_integrity:
        snapshot_manifest = read_json(snapshot / "snapshot_manifest.json")
        if verify_exact_manifest(
            snapshot,
            snapshot_manifest,
            manifest_filename="snapshot_manifest.json",
        )["status"] != "PASS":
            raise PermissionError("capability snapshot manifest mismatch")
    receipt = read_json(capability.issuance_receipt_path)
    expected_receipt_fields = {
        "schema_version",
        "status",
        "authorization_digest",
        "parsed_contract_digest",
        "claim_path",
        "claim_sha256",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "required_jobs",
        "scope",
        "unit_id",
        "preflight_proof_identity_digest",
        "parent_issuance_non_replayable",
        "worker_launch_secret_sha256",
        "receipt_digest",
    }
    receipt_payload = {
        key: value for key, value in receipt.items() if key != "receipt_digest"
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema_version")
        != "nursery-corrective-capability-issuance-v4"
        or receipt.get("status") != "PARENT_CAPABILITY_ISSUED"
        or receipt.get("receipt_digest") != canonical_digest(receipt_payload)
        or receipt.get("authorization_digest") != capability.authorization_digest
        or receipt.get("parsed_contract_digest") != capability.parsed_contract_digest
        or Path(str(receipt.get("claim_path"))).resolve()
        != Path(capability.claim_path)
        or receipt.get("claim_sha256") != capability.claim_sha256
        or Path(str(receipt.get("resolved_output_root"))).resolve()
        != Path(capability.resolved_output_root)
        or Path(str(receipt.get("resolved_staging_root"))).resolve()
        != Path(capability.resolved_staging_root)
        or Path(str(receipt.get("resolved_inner_staging_root"))).resolve()
        != Path(capability.resolved_inner_staging_root)
        or int(receipt.get("required_jobs", -1)) != capability.required_jobs
        or receipt.get("scope") != "parent"
        or receipt.get("unit_id") is not None
        or not isinstance(receipt.get("preflight_proof_identity_digest"), str)
        or len(str(receipt.get("preflight_proof_identity_digest"))) != 64
        or receipt.get("parent_issuance_non_replayable") is not True
        or receipt.get("worker_launch_secret_sha256")
        != capability.worker_launch_secret_sha256
    ):
        raise PermissionError("capability differs from issuance receipt")
    package = snapshot.parent
    claim_relative = Path(capability.claim_path).relative_to(package).as_posix()
    receipt_relative = Path(capability.issuance_receipt_path).relative_to(
        package
    ).as_posix()
    if _full_integrity:
        complete_manifest = read_json(complete_manifest_path)
        try:
            original_authorization_relative = Path(
                str(claim["original_authorization_path"])
            ).relative_to(package).as_posix()
            consumed_authorization_relative = authorization_path.relative_to(
                package
            ).as_posix()
        except ValueError as error:
            raise PermissionError(
                "capability authorization transition escapes package"
            ) from error
        transition = verify_consumed_package_transition(
            package,
            complete_manifest,
            pristine_authorization_relative=original_authorization_relative,
            consumed_authorization_relative=consumed_authorization_relative,
            claim_relative=claim_relative,
            receipt_relative=receipt_relative,
            manifest_filename="launch_seal/complete_file_manifest.json",
        )
        if transition["status"] != "PASS":
            raise PermissionError(
                f"capability package transition changed: {transition}"
            )
    if (
        capability.scope not in {"parent", "worker"}
        or (capability.scope == "parent" and capability.unit_id is not None)
        or (capability.scope == "worker" and not capability.unit_id)
    ):
        raise PermissionError("capability scope/unit invariant failed")
    if resolved_output_root is not None and Path(capability.resolved_output_root) != Path(
        resolved_output_root
    ).resolve():
        raise PermissionError("development capability output root mismatch")
    if resolved_staging_root is not None and Path(
        capability.resolved_staging_root
    ) != Path(resolved_staging_root).resolve():
        raise PermissionError("development capability staging root mismatch")
    if resolved_inner_staging_root is not None and Path(
        capability.resolved_inner_staging_root
    ) != Path(resolved_inner_staging_root).resolve():
        raise PermissionError("development capability inner staging root mismatch")
    if required_jobs is not None and capability.required_jobs != int(required_jobs):
        raise PermissionError("development capability job count mismatch")
    if required_scope is not None and capability.scope != str(required_scope):
        raise PermissionError("development capability scope mismatch")
    if required_unit_id is not None and capability.unit_id != str(required_unit_id):
        raise PermissionError("development capability unit mismatch")


_EXCLUDED_REHEARSAL_CAPABILITY_FACTORY_SEAL = object()


class ExcludedRehearsalCapability:
    __slots__ = (
        "purpose",
        "scope",
        "unit_id",
        "snapshot_root",
        "output_root",
        "persisted_input_root",
        "parallel_output_root",
        "attempt_receipt_path",
        "attempt_receipt_sha256",
        "required_jobs",
        "input_manifest_sha256",
        "scientific_contract_digest",
        "authorization_digest",
        "parsed_contract_digest",
        "_seal",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("excluded-rehearsal capabilities are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        scope: str,
        unit_id: str | None,
        snapshot_root: str | Path,
        output_root: str | Path,
        persisted_input_root: str | Path,
        parallel_output_root: str | Path,
        attempt_receipt_path: str | Path,
        attempt_receipt_sha256: str,
        required_jobs: int,
        input_manifest_sha256: str | None,
        scientific_contract_digest: str | None,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _EXCLUDED_REHEARSAL_CAPABILITY_FACTORY_SEAL:
            raise PermissionError(
                "excluded-rehearsal capability requires the consumed receipt"
            )
        self.purpose = "excluded_rehearsal"
        self.scope = str(scope)
        self.unit_id = None if unit_id is None else str(unit_id)
        self.snapshot_root = str(Path(snapshot_root).resolve())
        self.output_root = str(Path(output_root).resolve())
        self.persisted_input_root = str(Path(persisted_input_root).resolve())
        self.parallel_output_root = str(Path(parallel_output_root).resolve())
        self.attempt_receipt_path = str(Path(attempt_receipt_path).resolve())
        self.attempt_receipt_sha256 = str(attempt_receipt_sha256)
        self.required_jobs = int(required_jobs)
        self.input_manifest_sha256 = (
            None if input_manifest_sha256 is None else str(input_manifest_sha256)
        )
        self.scientific_contract_digest = (
            None
            if scientific_contract_digest is None
            else str(scientific_contract_digest)
        )
        self.authorization_digest = None
        self.parsed_contract_digest = None
        self._seal = _seal
        self._frozen = True


def _issue_excluded_rehearsal_capability(
    *,
    config: Mapping[str, Any],
    scope: str,
    unit_id: str | None,
    snapshot_root: str | Path,
    output_root: str | Path,
    persisted_input_root: str | Path,
    parallel_output_root: str | Path,
    attempt_receipt_path: str | Path,
    attempt_receipt_sha256: str,
    required_jobs: int,
    input_manifest_sha256: str | None = None,
    scientific_contract_digest: str | None = None,
) -> ExcludedRehearsalCapability:
    snapshot = Path(snapshot_root).resolve()
    repository = snapshot.parents[2]
    expected_output = (
        repository / str(config["paths"]["excluded_rehearsal"])
    ).resolve()
    expected_receipt = (
        repository
        / str(config["paths"]["excluded_rehearsal_attempt_receipt"])
    ).resolve()
    if (
        config["protocol"]["status"] != "frozen"
        or snapshot.name != "frozen_source_snapshot"
        or snapshot.parent.name
        != "synthetic_corrective_development_launch_package_v4"
        or Path(output_root).resolve() != expected_output
        or Path(attempt_receipt_path).resolve() != expected_receipt
        or int(required_jobs) != int(config["parallel"]["frozen_jobs"])
    ):
        raise PermissionError(
            "excluded-rehearsal capability differs from frozen configured paths"
        )
    capability = ExcludedRehearsalCapability(
        scope=scope,
        unit_id=unit_id,
        snapshot_root=snapshot_root,
        output_root=output_root,
        persisted_input_root=persisted_input_root,
        parallel_output_root=parallel_output_root,
        attempt_receipt_path=attempt_receipt_path,
        attempt_receipt_sha256=attempt_receipt_sha256,
        required_jobs=required_jobs,
        input_manifest_sha256=input_manifest_sha256,
        scientific_contract_digest=scientific_contract_digest,
        _seal=_EXCLUDED_REHEARSAL_CAPABILITY_FACTORY_SEAL,
    )
    verify_excluded_rehearsal_capability(capability, required_scope=scope)
    return capability


def verify_excluded_rehearsal_capability(
    capability: ExcludedRehearsalCapability,
    *,
    required_scope: str | None = None,
    required_unit_id: str | None = None,
    persisted_input_root: str | Path | None = None,
    parallel_output_root: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    scientific_contract_digest: str | None = None,
) -> None:
    if (
        not isinstance(capability, ExcludedRehearsalCapability)
        or capability._seal is not _EXCLUDED_REHEARSAL_CAPABILITY_FACTORY_SEAL
        or capability.purpose != "excluded_rehearsal"
    ):
        raise PermissionError(
            "excluded rehearsal requires a consumed attempt capability"
        )
    receipt_path = Path(capability.attempt_receipt_path)
    if sha256_file(receipt_path) != capability.attempt_receipt_sha256:
        raise PermissionError("excluded rehearsal attempt receipt changed")
    snapshot = Path(capability.snapshot_root)
    receipt = read_json(receipt_path)
    expected_receipt = {
        "schema_version": "nursery-corrective-excluded-rehearsal-attempt-v4",
        "status": "ONE_EXCLUDED_ATTEMPT_CONSUMED",
        "snapshot_root": str(snapshot),
        "snapshot_manifest_sha256": sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        "freeze_receipt_sha256": sha256_file(snapshot.parent / "freeze_receipt.json"),
        "output_root": capability.output_root,
        "jobs": capability.required_jobs,
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "non_replayable": True,
    }
    inner = Path(capability.output_root).with_name(
        f".{Path(capability.output_root).name}.cohort-staging"
    )
    if (
        receipt != expected_receipt
        or Path(capability.persisted_input_root) != (inner / "persisted_inputs").resolve()
        or Path(capability.parallel_output_root)
        != (inner / "parallel_compute").resolve()
        or capability.scope not in {"parent", "worker"}
        or (capability.scope == "parent" and capability.unit_id is not None)
        or (capability.scope == "worker" and not capability.unit_id)
        or (
            capability.scope == "parent"
            and (
                capability.input_manifest_sha256 is not None
                or capability.scientific_contract_digest is not None
            )
        )
        or (
            capability.scope == "worker"
            and (
                capability.input_manifest_sha256 is None
                or capability.scientific_contract_digest is None
            )
        )
    ):
        raise PermissionError("excluded rehearsal capability content mismatch")
    if required_scope is not None and capability.scope != str(required_scope):
        raise PermissionError("excluded rehearsal capability scope mismatch")
    if required_unit_id is not None and capability.unit_id != str(required_unit_id):
        raise PermissionError("excluded rehearsal capability unit mismatch")
    if persisted_input_root is not None and Path(
        capability.persisted_input_root
    ) != Path(persisted_input_root).resolve():
        raise PermissionError("excluded rehearsal persisted-input path mismatch")
    if parallel_output_root is not None and Path(
        capability.parallel_output_root
    ) != Path(parallel_output_root).resolve():
        raise PermissionError("excluded rehearsal parallel-output path mismatch")
    if (
        input_manifest_sha256 is not None
        and capability.input_manifest_sha256 != str(input_manifest_sha256)
    ):
        raise PermissionError("excluded rehearsal input commitment mismatch")
    if (
        scientific_contract_digest is not None
        and capability.scientific_contract_digest
        != str(scientific_contract_digest)
    ):
        raise PermissionError("excluded rehearsal scientific contract mismatch")


@dataclass(frozen=True, slots=True)
class IdentifierReference:
    role: str
    value: int


class IdentifierFirewall:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        purpose: str,
        capability: AuthorizationCapability | ExcludedRehearsalCapability | None = None,
        operation_contract_digest: str | None = None,
    ):
        if purpose not in PURPOSES:
            raise PermissionError(f"unsupported execution purpose: {purpose}")
        if purpose == "development":
            if not isinstance(capability, AuthorizationCapability):
                raise PermissionError("development requires verified capability")
            verify_development_capability(
                capability,
                _full_integrity=capability.scope != "worker",
            )
        elif purpose == "excluded_rehearsal":
            if not isinstance(capability, ExcludedRehearsalCapability):
                raise PermissionError(
                    "excluded rehearsal requires verified capability"
                )
            verify_excluded_rehearsal_capability(capability)
        elif capability is not None:
            raise PermissionError(
                "capability only valid for development or excluded rehearsal"
            )
        self.config = config
        self.purpose = purpose
        self.registry = config["resolved_registries"][purpose]
        self.capability = capability
        self.operation_contract_digest = operation_contract_digest
        self.operations: list[dict[str, Any]] = []
        self._previous_digest = "0" * 64

    def authorize(
        self,
        operation: str,
        references: Sequence[IdentifierReference],
        *,
        unit_id: str,
        local_sequence: int,
    ) -> str:
        if operation not in OPERATIONS:
            raise ValueError(f"unknown operation: {operation}")
        if (
            self.purpose in {"development", "excluded_rehearsal"}
            and self.capability is not None
            and self.capability.scope == "worker"
            and str(unit_id) != self.capability.unit_id
        ):
            raise PermissionError("worker capability cannot authorize another unit")
        if local_sequence != len(self.operations):
            raise ValueError("operation sequence must be contiguous")
        normalized = []
        quarantine = self.config["registries"]["permanently_quarantined_ranges"]
        for reference in references:
            if reference.role not in ROLES:
                raise ValueError(f"unknown identifier role: {reference.role}")
            value = int(reference.value)
            if _in_ranges(value, quarantine):
                raise PermissionError("permanently quarantined identifier")
            if value not in set(map(int, self.registry[reference.role])):
                raise PermissionError(
                    f"identifier {value} is not registered for {self.purpose}/{reference.role}"
                )
            normalized.append({"role": reference.role, "value": value})
        payload = {
            "purpose": self.purpose,
            "unit_id": str(unit_id),
            "local_sequence": int(local_sequence),
            "operation": operation,
            "references": normalized,
            "previous_digest": self._previous_digest,
        }
        entry_digest = canonical_digest(payload)
        self.operations.append({**payload, "entry_digest": entry_digest})
        self._previous_digest = entry_digest
        return entry_digest

    def ledger(self) -> dict[str, Any]:
        return {
            "schema_version": "nursery-operation-ledger-v1",
            "purpose": self.purpose,
            "authorization_digest": (
                self.capability.authorization_digest if self.capability else None
            ),
            "parsed_contract_digest": (
                self.capability.parsed_contract_digest if self.capability else None
            ),
            "operation_contract_digest": self.operation_contract_digest,
            "entry_count": len(self.operations),
            "terminal_digest": self._previous_digest,
            "entries": list(self.operations),
        }


def registry_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "nursery-corrective-registry-v4",
        "protocol_id": PROTOCOL_ID,
        "registries": config["resolved_registries"],
        "namespace_family": list(config["registries"]["namespace_family"]),
        "deliberately_unused_package_ranges": list(
            config["registries"]["deliberately_unused_package_ranges"]
        ),
        "permanently_quarantined_ranges": list(
            config["registries"]["permanently_quarantined_ranges"]
        ),
        "canonical_prior_registry": config["registries"]["canonical_prior_registry"],
        "canonical_prior_registry_sha256": config["registries"][
            "canonical_prior_registry_sha256"
        ],
    }


def verify_ledger(
    ledger: Mapping[str, Any],
    *,
    purpose: str,
    unit_id: str,
    expected_operations: Sequence[str],
    corpus_seed: int,
    model_seed: int,
    authorization_digest: str | None = None,
    parsed_contract_digest: str | None = None,
    operation_contract_digest: str | None = None,
) -> dict[str, Any]:
    problems: list[str] = []
    if set(ledger) != {
        "schema_version",
        "purpose",
        "authorization_digest",
        "parsed_contract_digest",
        "operation_contract_digest",
        "entry_count",
        "terminal_digest",
        "entries",
    }:
        problems.append("top_level_schema")
    entries = ledger.get("entries", [])
    if ledger.get("schema_version") != "nursery-operation-ledger-v1":
        problems.append("schema")
    if ledger.get("purpose") != purpose:
        problems.append("purpose")
    if ledger.get("authorization_digest") != authorization_digest:
        problems.append("authorization_digest")
    if ledger.get("parsed_contract_digest") != parsed_contract_digest:
        problems.append("parsed_contract_digest")
    if ledger.get("operation_contract_digest") != operation_contract_digest:
        problems.append("operation_contract_digest")
    if int(ledger.get("entry_count", -1)) != len(entries):
        problems.append("entry_count")
    if len(entries) != len(expected_operations):
        problems.append("expected_count")
    previous = "0" * 64
    for index, (entry, expected) in enumerate(zip(entries, expected_operations)):
        if set(entry) != {
            "purpose",
            "unit_id",
            "local_sequence",
            "operation",
            "references",
            "previous_digest",
            "entry_digest",
        }:
            problems.append(f"entry_schema:{index}")
        payload = {key: value for key, value in entry.items() if key != "entry_digest"}
        if entry.get("purpose") != purpose:
            problems.append(f"entry_purpose:{index}")
        if entry.get("unit_id") != unit_id:
            problems.append(f"unit:{index}")
        if int(entry.get("local_sequence", -1)) != index:
            problems.append(f"sequence:{index}")
        if entry.get("operation") != expected:
            problems.append(f"operation:{index}")
        if entry.get("previous_digest") != previous:
            problems.append(f"chain:{index}")
        if entry.get("entry_digest") != canonical_digest(payload):
            problems.append(f"digest:{index}")
        raw_references = list(entry.get("references", []))
        references = {
            (str(row.get("role")), int(row.get("value", -1)))
            for row in entry.get("references", [])
        }
        if references != {
            ("corpus", int(corpus_seed)),
            ("model", int(model_seed)),
        } or len(raw_references) != 2:
            problems.append(f"references:{index}")
        previous = str(entry.get("entry_digest", ""))
    if ledger.get("terminal_digest") != previous:
        problems.append("terminal_digest")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}
