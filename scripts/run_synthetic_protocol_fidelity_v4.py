#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.sensor_alignment_v4.study import (
    compare_core_artifacts,
    execute_qualification,
    finalize_package,
    freeze_package,
    run_test_suites,
    verify_freeze,
)

CONFIG = ROOT / "configs/synthetic_protocol_fidelity_v4.yaml"
PROTOCOL = ROOT / "docs/synthetic_protocol_fidelity_v4_protocol.md"
SOURCES = ROOT / "docs/synthetic_protocol_fidelity_v4_primary_sources.json"
TRACEABILITY = ROOT / "docs/synthetic_protocol_fidelity_v4_traceability.md"
LIMITATIONS = ROOT / "docs/synthetic_protocol_fidelity_v4_scientific_limitations.md"
OUTPUT = ROOT / "output/synthetic_protocol_fidelity_v4"
TRACKED = (
    "babyworld_lite/sensor_alignment_v4/__init__.py",
    "babyworld_lite/sensor_alignment_v4/protocol.py",
    "babyworld_lite/sensor_alignment_v4/corpus.py",
    "babyworld_lite/sensor_alignment_v4/detector_runtime.py",
    "babyworld_lite/sensor_alignment_v4/learner.py",
    "babyworld_lite/sensor_alignment_v4/controls.py",
    "babyworld_lite/sensor_alignment_v4/analysis.py",
    "babyworld_lite/sensor_alignment_v4/study.py",
    "scripts/run_synthetic_protocol_fidelity_v4.py",
    "tests/test_synthetic_protocol_fidelity_v4.py",
    "docs/synthetic_component_robustness_v3_forensic_audit_v4.md",
    "docs/synthetic_protocol_fidelity_v4_scientific_limitations.md",
    "docs/synthetic_protocol_fidelity_v4_fixture_repair_log.md",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("fixture", "freeze", "smoke", "reproduce", "tests", "finalize"),
    )
    args = parser.parse_args()
    if args.command == "fixture":
        result = execute_qualification(
            ROOT, CONFIG, OUTPUT / "fixture_validation_final", purpose="fixture_only"
        )
    elif args.command == "freeze":
        result = freeze_package(
            ROOT,
            CONFIG,
            PROTOCOL,
            SOURCES,
            TRACEABILITY,
            OUTPUT,
            TRACKED,
        )
    elif args.command == "smoke":
        integrity = verify_freeze(ROOT, OUTPUT)
        if integrity["status"] != "PASS":
            raise RuntimeError(f"freeze integrity failed before smoke: {integrity}")
        result = execute_qualification(
            ROOT, CONFIG, OUTPUT / "qualification_run", purpose="excluded_smoke"
        )
    elif args.command == "reproduce":
        if not (OUTPUT / "qualification_run/qualification_summary.json").exists():
            raise RuntimeError("primary excluded-smoke qualification was not executed")
        result = execute_qualification(
            ROOT, CONFIG, OUTPUT / "isolated_reproduction", purpose="excluded_smoke"
        )
        result = compare_core_artifacts(
            OUTPUT / "qualification_run",
            OUTPUT / "isolated_reproduction",
            OUTPUT / "reproduction_comparison.json",
        )
    elif args.command == "tests":
        result = run_test_suites(ROOT, OUTPUT / "test_execution_report.json")
    else:
        result = finalize_package(ROOT, OUTPUT, LIMITATIONS)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
