#!/usr/bin/env python3
"""Finalize the append-only STOP audit after correcting octal mode parsing.

The original preservation proof is retained as evidence of the validator bug.
This second pass rehashes every baseline-listed file and writes only new files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRESERVATION = OUT / "preservation"
BASELINE_SHA = PRESERVATION / "baseline.sha256"
BASELINE_STAT = PRESERVATION / "baseline.stat"
BASELINE_LINKS = PRESERVATION / "baseline.symlinks"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    if isinstance(value, (dict, list)):
        path.write_bytes(canonical_bytes(value))
    else:
        path.write_text(str(value))


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def verify_preservation() -> dict[str, Any]:
    expected_hashes = []
    for line in BASELINE_SHA.read_text().splitlines():
        expected_hashes.append((line[66:], line[:64]))
    expected_stats = {}
    for line in BASELINE_STAT.read_text().splitlines():
        size_text, mode_text, relative = line.split(" ", 2)
        expected_stats[relative] = (int(size_text), int(mode_text, 8))

    missing = []
    changed = []
    after_lines = []
    for relative, expected_hash in expected_hashes:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        observed_hash = sha256_file(path)
        observed_size = path.stat().st_size
        observed_mode = path.stat().st_mode & 0o7777
        expected_size, expected_mode = expected_stats[relative]
        after_lines.append(f"{observed_hash}  {relative}")
        if (observed_hash, observed_size, observed_mode) != (expected_hash, expected_size, expected_mode):
            changed.append(
                {
                    "path": relative,
                    "expected_sha256": expected_hash,
                    "observed_sha256": observed_hash,
                    "expected_bytes": expected_size,
                    "observed_bytes": observed_size,
                    "expected_mode_octal": format(expected_mode, "o"),
                    "observed_mode_octal": format(observed_mode, "o"),
                }
            )

    links_before = BASELINE_LINKS.read_text().splitlines()
    links_after = []
    link_changes = []
    for line in links_before:
        relative, target = line.split("\t", 1)
        path = ROOT / relative
        observed = path.readlink().as_posix() if path.is_symlink() else None
        links_after.append(f"{relative}\t{observed}")
        if observed != target:
            link_changes.append({"path": relative, "expected": target, "observed": observed})

    write_new(PRESERVATION / "after_v2.sha256", "\n".join(after_lines) + "\n")
    write_new(PRESERVATION / "after_v2.symlinks", "\n".join(links_after) + "\n")
    proof = {
        "status": "PASS" if not missing and not changed and not link_changes and len(after_lines) == len(expected_hashes) else "FAIL",
        "baseline_file_count": len(expected_hashes),
        "verified_file_count": len(after_lines),
        "baseline_symlink_count": len(links_before),
        "verified_symlink_count": len(links_after),
        "missing_paths": missing,
        "changed_paths": changed,
        "changed_symlinks": link_changes,
        "all_preexisting_bytes_unchanged": not missing and not changed and len(after_lines) == len(expected_hashes),
        "baseline_sha256_manifest_sha256": sha256_file(BASELINE_SHA),
        "after_v2_sha256_manifest_sha256": sha256_file(PRESERVATION / "after_v2.sha256"),
        "baseline_and_after_sha256_manifests_byte_identical": BASELINE_SHA.read_bytes() == (PRESERVATION / "after_v2.sha256").read_bytes(),
        "baseline_stat_manifest_sha256": sha256_file(BASELINE_STAT),
        "baseline_symlink_manifest_sha256": sha256_file(BASELINE_LINKS),
        "validator_note": "The retained v1 proof parsed octal mode text as decimal. It found zero hash/size changes but falsely flagged modes. This v2 pass uses base-8 parsing and rehashes every file.",
        "new_audit_files_are_not_part_of_the_preedit_baseline": True,
    }
    write_new(OUT / "preservation_proof_v2.json", proof)
    return proof


def write_final_summary(proof: dict[str, Any]) -> None:
    integrity = load(OUT / "development_integrity_audit.json")
    science = load(OUT / "scientific_diagnostics.json")
    registry = load(OUT / "registry_audit.json")
    terminal = science["terminal_decision"]
    if any(value != "PASS" for value in (integrity["status"], science["status"], registry["status"], proof["status"])):
        raise RuntimeError("cannot finalize a failed evidence audit")
    text = f"""# CONFIRMATION_STOP

The untouched 289xxx confirmation must not run.

## Decisive reason

{terminal['reason']}

## Closed evidence

- Development integrity and supplied hashes: PASS.
- Exact persisted-input set: 1,082/1,082; canonical digest matches.
- Retained persisted-input recomputation: 13/13 byte-identical.
- Closed development inventory: 1,104 files, 9,576,760,450 bytes, digest matches.
- Frozen inference reconstruction: all four effects, 20,000-bootstrap bounds, sign tests, and value digests match.
- Development operations: 8,688 total, 8,128 compute; all allowed and exactly registered.
- All 51 prior operation logs contain zero 289xxx references.
- Confirmation outcome count remains 0; no confirmation output existed before this audit.

## Adversarial construct finding

All 1,680 unaveraged action present/null cells across all seven conditions already have fractional accuracy exactly 1.0 with no ties. The frozen detector-selected semantic update has weight zero. Its operative path accepts only detector/learner agreement and power-sharpens the learner's existing semantic ordering; disagreement cannot correct it. The measured present/null probability gains correlate at {science['present_null_dependence']['sync_minus_absent_correlation']:.12f} and {science['present_null_dependence']['sync_minus_randomized_correlation']:.12f} because both are coupled to the same sharpened distribution.

The result is therefore alignment-conditioned confidence sharpening of already-correct synthetic mappings. It is not a test that sensor evidence supplied or corrected lexical/action grounding. Running the seed-invariant 289xxx replication would be scientifically uninformative for the frozen claim. Calling it confidence calibration would rename/broaden the claim; making sensors corrective would alter the learner/design. Both are prohibited after development.

## Preservation

Every one of the {proof['baseline_file_count']:,} pre-existing files and all {proof['baseline_symlink_count']} symlinks were checked again after the audit. SHA-256, byte size, mode, and link target all match the pre-edit baseline. The initial false preservation proof is retained: it found zero hash/size changes but parsed mode strings incorrectly. `preservation_proof_v2.json` is the corrected full second pass.

No confirmation package, authorization, or launch command was created, and confirmation was not executed.
"""
    write_new(OUT / "CONFIRMATION_STOP_FINAL.md", text)


def write_manifest() -> dict[str, Any]:
    rows = []
    for path in sorted(
        (path for path in OUT.rglob("*") if path.is_file() and path.name != "complete_file_manifest.json"),
        key=lambda value: value.relative_to(OUT).as_posix(),
    ):
        rows.append(
            {
                "path": path.relative_to(OUT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "protocol_id": "synthetic-confirmation-adversarial-audit-v1",
        "terminal": "CONFIRMATION_STOP",
        "self_excluded_by_definition": True,
        "file_count": len(rows),
        "files": rows,
        "digest": canonical_digest(rows),
    }
    write_new(OUT / "complete_file_manifest.json", manifest)
    return manifest


def main() -> int:
    for path in (OUT / "preservation_proof_v2.json", OUT / "CONFIRMATION_STOP_FINAL.md", OUT / "complete_file_manifest.json"):
        if path.exists():
            raise FileExistsError(path)
    proof = verify_preservation()
    if proof["status"] != "PASS":
        raise RuntimeError("corrected preservation verification failed")
    write_final_summary(proof)
    manifest = write_manifest()
    print(json.dumps({"terminal": "CONFIRMATION_STOP", "preservation": proof["status"], "package_files": manifest["file_count"], "package_digest": manifest["digest"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
