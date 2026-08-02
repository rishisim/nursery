#!/usr/bin/env python3
"""Emit only whitelisted aggregate readiness facts from the governed store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nursery_egobaby_preflight.contract import compact_aggregate_json


FIELDS = frozenset(
    {
        "status",
        "calibration_record_count",
        "calibration_checkpoint_item_count",
        "calibration_checkpoint_complete",
    }
)


def inspect(restricted_root: Path) -> dict[str, object]:
    stage = json.loads((restricted_root / "restricted_stage_manifest.json").read_text())
    checkpoint = json.loads(
        (restricted_root / "asr/restricted_calibration.json").read_text()
    )
    records = stage.get("calibration")
    items = checkpoint.get("items")
    if not isinstance(records, list) or not isinstance(items, dict):
        raise RuntimeError("E_GOVERNED_SCHEMA")
    return {
        "status": "PASS",
        "calibration_record_count": len(records),
        "calibration_checkpoint_item_count": len(items),
        "calibration_checkpoint_complete": len(records) == len(items),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restricted-root", type=Path, required=True)
    args = parser.parse_args()
    print(compact_aggregate_json(inspect(args.restricted_root), allowed_fields=FIELDS))


if __name__ == "__main__":
    main()
