#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_sample(
    development: list[dict[str, Any]], sample_size: int, salt: str
) -> list[dict[str, Any]]:
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in development:
        by_action[str(row["evaluation_targets"]["action_verb"])].append(row)
    for action in by_action:
        by_action[action].sort(
            key=lambda row: (
                hashlib.sha256(f"{salt}|{row['example_id']}".encode()).hexdigest(),
                str(row["example_id"]),
            )
        )
    action_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < min(sample_size, len(development)):
        available = [action for action, rows in by_action.items() if rows]
        if not available:
            break
        action = min(available, key=lambda item: (action_counts[item], len(by_action[item]), item))
        candidates = by_action[action]
        chosen = min(
            candidates,
            key=lambda row: (
                group_counts[str(row["event_group"])],
                hashlib.sha256(f"{salt}|{row['example_id']}".encode()).hexdigest(),
                str(row["example_id"]),
            ),
        )
        candidates.remove(chosen)
        action_counts[action] += 1
        group_counts[str(chosen["event_group"])] += 1
        selected.append(chosen)
    return selected


def build_manifests(examples: Path, protocol_path: Path, out_dir: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    expected_hash = str(protocol["source_examples_sha256"])
    observed_hash = sha256_file(examples)
    if observed_hash != expected_hash:
        raise ValueError(f"source hash mismatch: expected {expected_hash}, got {observed_hash}")
    rows = load_jsonl(examples)
    reserve = set(map(str, protocol["confirmation_event_groups"]))
    entries = []
    for row in rows:
        partition = "confirmation" if str(row["event_group"]) in reserve else "development"
        entries.append({
            "example_id": str(row["example_id"]),
            "sequence_id": str(row["sequence_id"]),
            "event_group": str(row["event_group"]),
            "location": int(row["location"]),
            "action_verb": str(row["evaluation_targets"]["action_verb"]),
            "partition": partition,
        })
    development = [row for row in rows if str(row["event_group"]) not in reserve]
    confirmation = [row for row in rows if str(row["event_group"]) in reserve]
    observed_counts = {
        "all_windows": len(rows),
        "development_windows": len(development),
        "confirmation_windows": len(confirmation),
        "development_event_groups": len({str(row["event_group"]) for row in development}),
        "confirmation_event_groups": len({str(row["event_group"]) for row in confirmation}),
    }
    if observed_counts != protocol["counts"]:
        raise ValueError(f"partition count mismatch: {observed_counts} != {protocol['counts']}")
    development_groups = {str(row["event_group"]) for row in development}
    confirmation_groups = {str(row["event_group"]) for row in confirmation}
    if development_groups & confirmation_groups:
        raise AssertionError("event group crosses development and confirmation")
    sample = audit_sample(
        development,
        int(protocol["audit"]["sample_size"]),
        str(protocol["audit"]["sampling_salt"]),
    )
    root = examples.parent
    audit_rows = []
    for rank, row in enumerate(sample, start=1):
        audit_rows.append({
            "audit_rank": rank,
            "example_id": str(row["example_id"]),
            "event_group": str(row["event_group"]),
            "sequence_id": str(row["sequence_id"]),
            "location": int(row["location"]),
            "action_verb": str(row["evaluation_targets"]["action_verb"]),
            "transcript": str(row["model_inputs"]["transcript"]),
            "frame_paths": [str(root / item) for item in row["model_inputs"]["frame_paths"]],
            "audit_label": None,
            "rationale": None,
        })
    if any(row["event_group"] in reserve for row in audit_rows):
        raise AssertionError("confirmation row entered audit manifest")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "partition_manifest.json").write_text(json.dumps({
        "schema_version": "aea-dev-partition-v1",
        "protocol_id": protocol["protocol_id"],
        "source_examples_sha256": observed_hash,
        "algorithm": "one_event_group_per_location_metadata_support_optimizer_frozen_in_preregistration",
        "counts": observed_counts,
        "confirmation_event_groups": sorted(confirmation_groups),
        "development_event_groups": sorted(development_groups),
        "entries": entries,
    }, indent=2) + "\n")
    (out_dir / "audit_manifest_prelabel.json").write_text(json.dumps({
        "schema_version": "aea-dev-audit-manifest-prelabel-v1",
        "protocol_id": protocol["protocol_id"],
        "sampling_salt": protocol["audit"]["sampling_salt"],
        "sample_size": len(audit_rows),
        "labels_inspected_at_creation": False,
        "rows": audit_rows,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_manifests(args.examples, args.protocol, args.out)


if __name__ == "__main__":
    main()
