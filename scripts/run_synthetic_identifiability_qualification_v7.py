#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.sensor_alignment_v7.adjudicator import adjudicate_file
from babyworld_lite.sensor_alignment_v7.integrity import (
    collect_test_nodeids,
    compare_recomputation,
    finalize_decision,
    freeze_package,
    run_independent_recompute,
    run_negative_recompute_tests,
    run_official_tests,
    verify_freeze,
    verify_prior_preservation,
    write_prior_preservation_baseline,
)
from babyworld_lite.sensor_alignment_v7.protocol import (
    load_config,
    require_frozen_config,
    verify_file_manifest,
    write_json,
)
from babyworld_lite.sensor_alignment_v7.study import compute_from_persisted, execute_qualification

CONFIG = "configs/synthetic_identifiability_qualification_v7.yaml"
PROTOCOL = "docs/synthetic_identifiability_qualification_v7_protocol.md"
SOURCES = "docs/synthetic_identifiability_qualification_v7_primary_sources.json"
TRACEABILITY = "docs/synthetic_identifiability_qualification_v7_traceability.json"
AUDIT = "docs/synthetic_identifiability_qualification_v7_v5_v6_audit.json"
REPAIR_LEDGER = "docs/synthetic_identifiability_qualification_v7_repair_ledger.json"
OUTPUT = ROOT / "output/synthetic_identifiability_qualification_v7"
TRACKED = (
    "babyworld_lite/__init__.py",
    "babyworld_lite/sensor_alignment_v2/__init__.py",
    "babyworld_lite/sensor_alignment_v2/detector.py",
    "babyworld_lite/sensor_alignment_v2/protocol.py",
    "babyworld_lite/sensor_alignment_v2/synthetic.py",
    "babyworld_lite/sensor_alignment_v7/__init__.py",
    "babyworld_lite/sensor_alignment_v7/protocol.py",
    "babyworld_lite/sensor_alignment_v7/benchmark.py",
    "babyworld_lite/sensor_alignment_v7/detector_runtime.py",
    "babyworld_lite/sensor_alignment_v7/learner.py",
    "babyworld_lite/sensor_alignment_v7/controls.py",
    "babyworld_lite/sensor_alignment_v7/study.py",
    "babyworld_lite/sensor_alignment_v7/adjudicator.py",
    "babyworld_lite/sensor_alignment_v7/integrity.py",
    "scripts/run_synthetic_identifiability_qualification_v7.py",
    "tests/conftest.py",
    "tests/test_synthetic_identifiability_qualification_v7.py",
    "configs/synthetic_weak_alignment_recovery_v1.yaml",
    "configs/synthetic_sensor_event_robustness_v2.yaml",
    "configs/synthetic_component_robustness_v3.yaml",
    "configs/synthetic_protocol_fidelity_v4.yaml",
    "configs/synthetic_identifiability_qualification_v5.yaml",
    "configs/synthetic_identifiability_qualification_v6.yaml",
    "output/synthetic_identifiability_qualification_v6/prefreeze_invalidation.json",
    CONFIG,
    PROTOCOL,
    SOURCES,
    TRACEABILITY,
    AUDIT,
    REPAIR_LEDGER,
    "docs/synthetic_identifiability_qualification_v7_limitations.md",
)


def _verify_snapshot(snapshot_root: Path) -> None:
    manifest_path = snapshot_root / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("snapshot manifest missing")
    result = verify_file_manifest(snapshot_root, json.loads(manifest_path.read_text()))
    if result["status"] != "PASS":
        raise RuntimeError(f"frozen source verification failed: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "collect",
            "baseline",
            "fixture",
            "freeze",
            "smoke",
            "recompute",
            "reproduce",
            "tests",
            "preservation",
            "finalize",
            "adjudicate",
        ),
    )
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    args = parser.parse_args()
    if args.command == "recompute":
        snapshot = Path(args.snapshot_root).resolve()
        _verify_snapshot(snapshot)
        result = compute_from_persisted(args.input_root, args.output_root)
    elif args.command == "adjudicate":
        snapshot = Path(args.snapshot_root).resolve()
        _verify_snapshot(snapshot)
        result = adjudicate_file(args.input_path, args.output_path)
    else:
        config = load_config(ROOT / CONFIG, repository_root=ROOT)
        if args.command == "baseline":
            result = write_prior_preservation_baseline(
                ROOT,
                OUTPUT / "preserved_v1_v6_hashes_before.tsv",
            )
        elif args.command == "collect":
            executable = ROOT / str(config["environment"]["python"])
            result = collect_test_nodeids(
                ROOT, executable, str(config["environment"]["version_test_root"])
            )
            write_json(
                OUTPUT / "collected_tests_pre_freeze.json",
                result,
                overwrite=(OUTPUT / "collected_tests_pre_freeze.json").exists(),
            )
        elif args.command == "fixture":
            result = execute_qualification(
                ROOT, config, OUTPUT / "fixture_validation_official", purpose="fixture_only"
            )
        elif args.command == "freeze":
            result = freeze_package(
                ROOT,
                OUTPUT,
                config,
                tracked_files=TRACKED,
                config_path=CONFIG,
                protocol_path=PROTOCOL,
                sources_path=SOURCES,
                traceability_path=TRACEABILITY,
            )
        elif args.command == "smoke":
            require_frozen_config(config)
            integrity = verify_freeze(ROOT, OUTPUT)
            if integrity["status"] != "PASS":
                raise RuntimeError(f"freeze failed before excluded smoke: {integrity}")
            result = execute_qualification(
                ROOT, config, OUTPUT / "qualification_run", purpose="excluded_smoke"
            )
        elif args.command == "reproduce":
            if not (OUTPUT / "qualification_run/qualification_summary.json").is_file():
                raise RuntimeError("excluded smoke has not run")
            process = run_independent_recompute(OUTPUT)
            comparison = (
                compare_recomputation(OUTPUT)
                if process["exit_code"] == 0
                else {"status": "FAIL", "process": process}
            )
            negative = run_negative_recompute_tests(OUTPUT)
            result = {"process": process, "comparison": comparison, "negative": negative}
        elif args.command == "tests":
            result = run_official_tests(ROOT, OUTPUT)
        elif args.command == "preservation":
            result = verify_prior_preservation(
                ROOT,
                OUTPUT / "preserved_v1_v6_hashes_before.tsv",
                OUTPUT / "preserved_v1_v6_hashes_after.tsv",
                OUTPUT / "preservation_proof.json",
            )
        else:
            specification = json.loads((OUTPUT / "frozen_traceability.json").read_text())
            result = finalize_decision(ROOT, OUTPUT, specification)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
