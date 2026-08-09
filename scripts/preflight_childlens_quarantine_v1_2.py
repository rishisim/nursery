#!/usr/bin/env python3
"""Fail-closed, aggregate-only ChildLens quarantine re-admission preflight.

The script never opens restricted files and never serializes the quarantine
path, descendant names, identifiers, or payload.  It discovers the one
quarantine candidate from the existing no-index sentinel, inspects filesystem
metadata only, and emits a nonidentifying resource/control receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import shutil
import stat
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


GIB = 1_073_741_824
RAW_CAP_BYTES = 20 * GIB
NAMESPACE_CAP_BYTES = 73 * GIB
FREE_FLOOR_BYTES = 50 * GIB
RETENTION_DEADLINE = date(2027, 7, 31)


@dataclass(frozen=True)
class TreeAggregate:
    directory_count: int
    file_count: int
    symlink_count: int
    allocated_bytes: int
    apparent_bytes: int
    directory_mode_violation_count: int
    file_mode_violation_count: int
    owner_mismatch_count: int
    acl_entry_path_count: int
    acl_check_failure_count: int


def _captured(argv: list[str]) -> tuple[int, str]:
    """Run a local control probe and retain its output only in memory."""

    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 127, ""
    return result.returncode, (result.stdout + result.stderr)


def discover_quarantine(research_root: Path, repository_root: Path) -> list[Path]:
    """Discover candidate roots without exporting any candidate string."""

    repository_root = repository_root.resolve()
    candidates: list[Path] = []
    for sentinel in research_root.glob(".*/*/.metadata_never_index"):
        try:
            root = sentinel.parent.resolve(strict=True)
            root.relative_to(repository_root)
        except ValueError:
            candidates.append(root)
        except OSError:
            continue
    return candidates


def _iter_metadata(root: Path) -> Iterable[tuple[Path, os.stat_result]]:
    """Yield lstat metadata without opening restricted file content."""

    stack = [root]
    while stack:
        current = stack.pop()
        metadata = current.lstat()
        yield current, metadata
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            continue
        with os.scandir(current) as entries:
            for entry in entries:
                child = Path(entry.path)
                child_meta = child.lstat()
                if stat.S_ISDIR(child_meta.st_mode) and not stat.S_ISLNK(child_meta.st_mode):
                    stack.append(child)
                else:
                    yield child, child_meta


def _has_acl(path: Path) -> tuple[bool, bool]:
    """Return (has_acl, check_failed) without exposing ``ls`` output."""

    code, captured = _captured(["/bin/ls", "-lde", str(path)])
    if code != 0 or not captured:
        return False, True
    first_line = captured.splitlines()[0]
    mode_token = first_line.split(maxsplit=1)[0] if first_line else ""
    return mode_token.endswith("+"), False


def aggregate_tree(root: Path, expected_uid: int | None = None) -> TreeAggregate:
    expected_uid = os.getuid() if expected_uid is None else expected_uid
    directories = files = symlinks = 0
    allocated = apparent = 0
    directory_mode_violations = file_mode_violations = owner_mismatches = 0
    acl_paths = acl_failures = 0

    for path, metadata in _iter_metadata(root):
        mode = stat.S_IMODE(metadata.st_mode)
        allocated += metadata.st_blocks * 512
        apparent += metadata.st_size
        if stat.S_ISLNK(metadata.st_mode):
            symlinks += 1
        elif stat.S_ISDIR(metadata.st_mode):
            directories += 1
            if mode != 0o700:
                directory_mode_violations += 1
        elif stat.S_ISREG(metadata.st_mode):
            files += 1
            if mode != 0o600:
                file_mode_violations += 1
        else:
            file_mode_violations += 1
        if metadata.st_uid != expected_uid:
            owner_mismatches += 1
        has_acl, failed = _has_acl(path)
        acl_paths += int(has_acl)
        acl_failures += int(failed)

    return TreeAggregate(
        directory_count=directories,
        file_count=files,
        symlink_count=symlinks,
        allocated_bytes=allocated,
        apparent_bytes=apparent,
        directory_mode_violation_count=directory_mode_violations,
        file_mode_violation_count=file_mode_violations,
        owner_mismatch_count=owner_mismatches,
        acl_entry_path_count=acl_paths,
        acl_check_failure_count=acl_failures,
    )


def repair_owner_only_modes(root: Path) -> tuple[int, int]:
    """Normalize modes from metadata only; never follow links or open files."""

    repaired_directories = repaired_files = 0
    for path, metadata in _iter_metadata(root):
        if stat.S_ISLNK(metadata.st_mode):
            continue
        current = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode) and current != 0o700:
            os.chmod(path, 0o700, follow_symlinks=False)
            repaired_directories += 1
        elif stat.S_ISREG(metadata.st_mode) and current != 0o600:
            os.chmod(path, 0o600, follow_symlinks=False)
            repaired_files += 1
    return repaired_directories, repaired_files


def _filevault_on() -> bool | None:
    code, captured = _captured(["/usr/bin/fdesetup", "status"])
    if code != 0:
        return None
    normalized = captured.casefold()
    if "filevault is on" in normalized:
        return True
    if "filevault is off" in normalized:
        return False
    return None


def _time_machine_excluded(root: Path) -> bool | None:
    code, captured = _captured(["/usr/bin/tmutil", "isexcluded", "-X", str(root)])
    if code != 0:
        return None
    try:
        parsed = plistlib.loads(captured.encode("utf-8"))
    except (ValueError, plistlib.InvalidFileException):
        parsed = None
    if (
        isinstance(parsed, list)
        and len(parsed) == 1
        and isinstance(parsed[0], dict)
        and "IsExcluded" in parsed[0]
    ):
        return bool(parsed[0]["IsExcluded"])
    marker = captured.casefold()
    if "[excluded]" in marker:
        return True
    if "[included]" in marker:
        return False
    return None


def _cloud_sync_path(root: Path) -> bool:
    forbidden_components = {
        "cloudstorage",
        "mobile documents",
        "onedrive",
        "dropbox",
        "google drive",
    }
    return any(part.casefold() in forbidden_components for part in root.parts)


def build_receipt(
    repository_root: Path,
    research_root: Path,
    observed_date: date,
    *,
    control_sentinel_mode_repaired: bool = False,
    repaired_directory_mode_count: int = 0,
    repaired_file_mode_count: int = 0,
) -> dict:
    candidates = discover_quarantine(research_root, repository_root)
    receipt: dict = {
        "schema_version": "childlens-quarantine-resource-readmission-v1.2.0",
        "observed_date": observed_date.isoformat(),
        "time_scoped": True,
        "candidate_count": len(candidates),
        "restricted_root_exported": False,
        "restricted_names_or_payload_opened": False,
        "control_sentinel_mode_repaired": control_sentinel_mode_repaired,
        "repaired_directory_mode_count": repaired_directory_mode_count,
        "repaired_file_mode_count": repaired_file_mode_count,
        "retention_deadline": RETENTION_DEADLINE.isoformat(),
        "retention_days_remaining_at_observation": (RETENTION_DEADLINE - observed_date).days,
        "capacity_controls": {
            "raw_cap_bytes": RAW_CAP_BYTES,
            "namespace_peak_cap_bytes": NAMESPACE_CAP_BYTES,
            "minimum_free_after_peak_bytes": FREE_FLOOR_BYTES,
            "full_archive_permitted": False,
            "simultaneous_full_object_downloads": 1,
            "raw_cap_enforcement": "HARD_CUMULATIVE_BYTE_COUNTER_ABORT_BEFORE_CAP",
        },
    }
    if len(candidates) != 1:
        receipt.update(
            {
                "status": "FAIL_CLOSED_QUARANTINE_CANDIDATE_NOT_UNIQUE",
                "quarantine_controls_pass": False,
                "resource_snapshot_pass": False,
            }
        )
        return receipt

    root = candidates[0]
    tree = aggregate_tree(root)
    repo_resolved = repository_root.resolve()
    root_resolved = root.resolve(strict=True)
    try:
        root_resolved.relative_to(repo_resolved)
        outside_repository = False
    except ValueError:
        outside_repository = True

    root_metadata = root_resolved.lstat()
    same_device_as_repository = root_metadata.st_dev == repo_resolved.lstat().st_dev
    filevault_on = _filevault_on()
    tm_excluded = _time_machine_excluded(root_resolved)
    sentinel_present = (root_resolved / ".metadata_never_index").is_file()
    free_bytes = shutil.disk_usage(root_resolved).free
    namespace_headroom_bytes = max(0, NAMESPACE_CAP_BYTES - tree.allocated_bytes)
    free_floor_headroom_bytes = max(0, free_bytes - FREE_FLOOR_BYTES)
    maximum_additional_bytes = min(namespace_headroom_bytes, free_floor_headroom_bytes)

    controls_pass = all(
        (
            outside_repository,
            same_device_as_repository,
            stat.S_IMODE(root_metadata.st_mode) == 0o700,
            tree.directory_mode_violation_count == 0,
            tree.file_mode_violation_count == 0,
            tree.owner_mismatch_count == 0,
            tree.symlink_count == 0,
            tree.acl_entry_path_count == 0,
            tree.acl_check_failure_count == 0,
            filevault_on is True,
            tm_excluded is True,
            sentinel_present,
            not _cloud_sync_path(root_resolved),
            (RETENTION_DEADLINE - observed_date).days > 0,
        )
    )
    resource_snapshot_pass = maximum_additional_bytes >= RAW_CAP_BYTES

    receipt.update(
        {
            "status": (
                "PASS_QUARANTINE_AND_RESOURCE_SNAPSHOT"
                if controls_pass and resource_snapshot_pass
                else "FAIL_CLOSED_QUARANTINE_OR_RESOURCE_CONTROL"
            ),
            "quarantine_controls_pass": controls_pass,
            "resource_snapshot_pass": resource_snapshot_pass,
            "quarantine": {
                "exists": True,
                "outside_repository": outside_repository,
                "same_device_as_repository": same_device_as_repository,
                "same_encrypted_local_volume": same_device_as_repository and filevault_on is True,
                "filevault_on": filevault_on,
                "root_mode": format(stat.S_IMODE(root_metadata.st_mode), "04o"),
                "directory_count": tree.directory_count,
                "file_count": tree.file_count,
                "symlink_count": tree.symlink_count,
                "directory_mode_violation_count": tree.directory_mode_violation_count,
                "file_mode_violation_count": tree.file_mode_violation_count,
                "owner_mismatch_count": tree.owner_mismatch_count,
                "acl_entry_path_count": tree.acl_entry_path_count,
                "acl_check_failure_count": tree.acl_check_failure_count,
                "time_machine_excluded": tm_excluded,
                "spotlight_no_index_sentinel_present": sentinel_present,
                "cloud_sync_path_detected": _cloud_sync_path(root_resolved),
                "git_tracking_possible": not outside_repository,
                "allocated_bytes_at_observation": tree.allocated_bytes,
                "apparent_bytes_at_observation": tree.apparent_bytes,
            },
            "resource_snapshot": {
                "shared_volume_free_bytes": free_bytes,
                "current_namespace_allocated_bytes": tree.allocated_bytes,
                "namespace_headroom_bytes": namespace_headroom_bytes,
                "free_floor_headroom_bytes": free_floor_headroom_bytes,
                "maximum_additional_bytes_now": maximum_additional_bytes,
                "full_raw_cap_fits_current_controls": maximum_additional_bytes >= RAW_CAP_BYTES,
                "fresh_recheck_required_before_every_object": True,
            },
            "admission_scope": {
                "quarantine_control_state": "ADMITTED" if controls_pass else "NOT_ADMITTED",
                "host_capacity_state": "TIME_SCOPED_PASS" if resource_snapshot_pass else "NOT_ADMITTED",
                "media_transfer_authorized_by_this_receipt": False,
                "remaining_external_gates": [
                    "FROZEN_SELECTION_AND_RELEASE_RECEIPT_MATCH",
                    "NATIVE_TRANSFER_OR_PREDECLARED_CONSERVATIVE_BOUND_CONTROLLER_PASS",
                    "OBJECT_LEVEL_CUMULATIVE_BYTE_COUNTER_ACTIVE",
                ],
            },
            "privacy": {
                "restricted_path_serialized": False,
                "descendant_names_serialized": False,
                "restricted_file_content_opened": False,
                "identifiers_or_exact_media_times_serialized": False,
                "external_service_used": False,
            },
        }
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repair-control-sentinel-mode", action="store_true")
    parser.add_argument("--repair-owner-only-modes", action="store_true")
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    research_root = (args.research_root or repository_root.parent).resolve()
    repaired = False
    repaired_directories = repaired_files = 0
    if args.repair_control_sentinel_mode:
        candidates = discover_quarantine(research_root, repository_root)
        if len(candidates) == 1:
            sentinel = candidates[0] / ".metadata_never_index"
            if sentinel.is_file() and stat.S_IMODE(sentinel.lstat().st_mode) != 0o600:
                sentinel.chmod(0o600)
                repaired = True
    if args.repair_owner_only_modes:
        candidates = discover_quarantine(research_root, repository_root)
        if len(candidates) == 1:
            repaired_directories, repaired_files = repair_owner_only_modes(candidates[0])
    receipt = build_receipt(
        repository_root,
        research_root,
        date.today(),
        control_sentinel_mode_repaired=repaired,
        repaired_directory_mode_count=repaired_directories,
        repaired_file_mode_count=repaired_files,
    )
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if receipt.get("status") == "PASS_QUARANTINE_AND_RESOURCE_SNAPSHOT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
