#!/usr/bin/env python3
"""Privacy-safe doctor for the ChildLens fold-evidence recovery amendment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "output/childlens_fold_evidence_recovery/recovery_receipt.json"
SUMMARY = ROOT / "output/childlens_alignment_bridge_v5/clean_calibration_summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    checks = {
        "exact_recovery": value["recovery"]["all_bound_hashes_match"],
        "zero_locked_access": value["recovery"]["locked_participant_count"] == 0,
        "pooled_byte_reproduction": (
            value["reproduction"]["byte_identical"]
            and sha256(SUMMARY) == value["reproduction"]["pooled_summary_sha256"]
        ),
        "blender_contract": value["blender_replay"]["physical_contract_pass"],
        "crossfit_pass": value["crossfit"]["gate"] == "PASS",
        "readiness_controls": value["readiness_controls_run"],
        "cue_lift_sealed": value["cue_lift_arms_run"] is False,
    }
    result = {
        "ready": all(checks.values()),
        "decision": value["decision"],
        "checks": checks,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
