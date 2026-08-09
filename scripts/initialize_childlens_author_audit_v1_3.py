#!/usr/bin/env python3
"""Zero-argument initializer for the ChildLens v1.3 author audit.

The initializer inherits the already verified v1.2 owner-private quarantine,
finds exactly one primary sample by its public canonical digest, and initializes
the blinded author store without accepting or mounting predictions.  It emits
only fixed, aggregate-safe status.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import childlens_author_audit_v1_3 as v13
import childlens_human_validation_v1_2 as v12


SAMPLING_RECEIPT = (
    v13.REPO_ROOT
    / "output/childlens_feasibility_v1_3/author_audit_sampling_receipt.json"
)
MAX_CANDIDATE_BYTES = 16 * 1024 * 1024
SKIP_DIRECTORIES = {
    "raw_v1_2",
    "browser_profile",
    "browser_cache",
    "browser_downloads",
    "pseudo_annotations_v1_3",
}


def _load_public_sample_digest() -> str:
    try:
        receipt = json.loads(SAMPLING_RECEIPT.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise v13.WorkflowError("E_PUBLIC_SAMPLE_RECEIPT")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version")
        != "childlens-author-audit-sampling-receipt-v1.3.0"
        or receipt.get("status") != "AUTHOR_AUDIT_SAMPLES_READY"
        or receipt.get("selected_item_count") != 15
        or receipt.get("primary_total_seconds") != 900
        or receipt.get("audit_sample_frozen_before_predictions") is not True
        or receipt.get("model_prediction_independent") is not True
    ):
        raise v13.WorkflowError("E_PUBLIC_SAMPLE_RECEIPT")
    return v13._require_hex(
        receipt.get("primary_packet_sha256"), "E_PUBLIC_SAMPLE_RECEIPT"
    )


def _private_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _canonical_json_digest(path: Path) -> str | None:
    if not v13._private_file(path):
        return None
    try:
        if path.stat().st_size > MAX_CANDIDATE_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    return v13._sha256(v13._canonical(value))


def locate_primary_sample(root: Path, expected_digest: str) -> Path:
    matches: list[Path] = []
    try:
        for directory, directories, files in os.walk(root, followlinks=False):
            current = Path(directory)
            if not _private_directory(current):
                directories[:] = []
                continue
            directories[:] = [
                name
                for name in directories
                if name not in SKIP_DIRECTORIES
                and not (current / name).is_symlink()
                and _private_directory(current / name)
            ]
            for name in files:
                candidate = current / name
                if candidate.suffix.lower() != ".json":
                    continue
                if _canonical_json_digest(candidate) == expected_digest:
                    matches.append(candidate.resolve(strict=True))
    except OSError:
        raise v13.WorkflowError("E_SAMPLE_DISCOVERY")
    unique = sorted(set(matches), key=str)
    if len(unique) != 1:
        raise v13.WorkflowError(
            "E_SAMPLE_DISCOVERY_AMBIGUOUS" if unique else "E_SAMPLE_DISCOVERY_NOT_FOUND"
        )
    return unique[0]


def inherit_v1_3_policy(root: Path) -> None:
    source_policy = v12._load_policy(root)
    source_digest = v13._sha256(v13._canonical(source_policy))
    payload: Mapping[str, Any] = {
        "schema_version": v13.VERSION,
        "created_at_utc": v13._utc_now(),
        "retention_deadline": v13.RETENTION_DEADLINE,
        "owner_only_verified": True,
        "git_exclusion_verified": True,
        "indexing_exclusion_verified": True,
        "backup_exclusion_verified_or_encrypted_local_only": True,
        "local_only": True,
        "signed_agreement_controls_inherited": True,
        "inherited_v1_2_policy_sha256": source_digest,
        "prediction_store_mounted_before_author_lock": False,
    }
    path = root / v13.POLICY_FILE
    if path.exists():
        existing = v13._load_json_file(path, "E_QUARANTINE_POLICY_INVALID")
        stable_existing = dict(existing)
        stable_payload = dict(payload)
        stable_existing.pop("created_at_utc", None)
        stable_payload.pop("created_at_utc", None)
        if stable_existing != stable_payload:
            raise v13.WorkflowError("E_QUARANTINE_POLICY_IMMUTABLE")
    else:
        v13._atomic_write(path, v13._canonical(payload) + b"\n")


def initialize_existing_quarantine() -> Mapping[str, Any]:
    try:
        existing = v13.discover_runtime_root()
    except v13.WorkflowError as exc:
        if exc.code != "E_RUNTIME_ROOT_NOT_FOUND":
            raise
    else:
        v13.validate_workflow(existing)
        receipt = v13.aggregate_receipt(existing)
        return {
            "status": "AUTHOR_AUDIT_ALREADY_INITIALIZED",
            "route": v13.ROUTE,
            "audit_item_count": receipt["audit_item_count"],
            "audit_speech_minutes": receipt["audit_speech_minutes"],
            "predictions_mounted": receipt["prediction_join_enabled"],
        }

    root = v12.discover_runtime_root()
    root = v12._validate_root_controls(root)
    expected_digest = _load_public_sample_digest()
    sample = locate_primary_sample(root, expected_digest)
    inherit_v1_3_policy(root)
    v13._DISCOVERED_RUNTIME_ROOT = root
    result = v13.initialize(root, sample)
    if result.get("audit_item_count") != 15:
        raise v13.WorkflowError("E_INITIALIZATION_RESULT")
    return {
        "status": "AUTHOR_AUDIT_INITIALIZED",
        "route": v13.ROUTE,
        "audit_item_count": 15,
        "audit_speech_minutes": 15.0,
        "predictions_mounted": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if supplied:
        print("E_ARGUMENTS", file=sys.stderr)
        return 2
    try:
        result = initialize_existing_quarantine()
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (v13.WorkflowError, v12.WorkflowError) as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("E_INITIALIZER_INTERNAL", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
