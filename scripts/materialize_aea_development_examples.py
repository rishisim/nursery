#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.partition.read_text())
    partition = {row["example_id"]: row["partition"] for row in manifest["entries"]}
    development_ids = {key for key, value in partition.items() if value == "development"}
    confirmation_ids = {key for key, value in partition.items() if value == "confirmation"}
    selected = []
    for line in args.examples.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = str(row["example_id"])
        if example_id in development_ids:
            selected.append(row)
        elif example_id not in confirmation_ids:
            raise ValueError(f"example absent from frozen partition: {example_id}")
    if {str(row["example_id"]) for row in selected} != development_ids:
        raise ValueError("development materialization does not match frozen manifest")
    confirmation_groups = set(manifest["confirmation_event_groups"])
    if any(str(row["event_group"]) in confirmation_groups for row in selected):
        raise AssertionError("confirmation group entered development corpus")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    args.out.write_text(payload)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    receipt = {
        "schema_version": "aea-development-materialization-v1",
        "partition_manifest_sha256": hashlib.sha256(args.partition.read_bytes()).hexdigest(),
        "development_examples_sha256": digest,
        "development_windows": len(selected),
        "development_event_groups": len({str(row["event_group"]) for row in selected}),
        "confirmation_paths_opened": False,
    }
    args.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
