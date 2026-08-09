#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.development_launch_v3.adjudicator import (
    adjudicate_file,
    verify_authorization,
)
from babyworld_lite.development_launch_v3.integrity import (
    collect_test_nodeids,
    compare_recomputation,
    finalize_package,
    freeze_package,
    run_independent_recompute,
    run_negative_recompute_tests,
    run_official_tests,
    verify_freeze,
    verify_prior_preservation,
    write_prior_preservation_baseline,
)
from babyworld_lite.development_launch_v3.protocol import (
    load_config,
    require_frozen,
    sha256_file,
    verify_file_manifest,
    write_json,
)
from babyworld_lite.development_launch_v3.study import (
    compute_from_persisted,
    execute_cohort,
)

CONFIG = "configs/synthetic_development_launch_v3.yaml"
PROTOCOL = "docs/synthetic_development_launch_v3_protocol.md"
SOURCES = "docs/synthetic_development_launch_v3_primary_sources.json"
TRACEABILITY = "docs/synthetic_development_launch_v3_traceability.json"
LIMITATIONS = "docs/synthetic_development_launch_v3_limitations.md"
SAMPLE_SIZE = "docs/synthetic_development_launch_v3_sample_size.md"
REPAIR_LEDGER = "docs/synthetic_development_launch_v3_repair_ledger.json"
OUTPUT = ROOT / "output/synthetic_development_launch_package_v3"
TRACKED = (
    "babyworld_lite/__init__.py",
    "babyworld_lite/sensor_alignment_v2/__init__.py",
    "babyworld_lite/sensor_alignment_v2/detector.py",
    "babyworld_lite/sensor_alignment_v2/protocol.py",
    "babyworld_lite/sensor_alignment_v2/synthetic.py",
    "babyworld_lite/sensor_alignment_v8/__init__.py",
    "babyworld_lite/sensor_alignment_v8/benchmark.py",
    "babyworld_lite/sensor_alignment_v8/controls.py",
    "babyworld_lite/sensor_alignment_v8/detector_runtime.py",
    "babyworld_lite/sensor_alignment_v8/learner.py",
    "babyworld_lite/sensor_alignment_v8/protocol.py",
    "babyworld_lite/development_launch_v3/__init__.py",
    "babyworld_lite/development_launch_v3/protocol.py",
    "babyworld_lite/development_launch_v3/statistics.py",
    "babyworld_lite/development_launch_v3/study.py",
    "babyworld_lite/development_launch_v3/adjudicator.py",
    "babyworld_lite/development_launch_v3/integrity.py",
    "scripts/run_synthetic_development_launch_v3.py",
    "tests/conftest.py",
    "tests/test_synthetic_development_launch_v3.py",
    "configs/synthetic_weak_alignment_recovery_v1.yaml",
    "configs/synthetic_sensor_event_robustness_v2.yaml",
    "configs/synthetic_component_robustness_v3.yaml",
    "configs/synthetic_protocol_fidelity_v4.yaml",
    "configs/synthetic_identifiability_qualification_v5.yaml",
    "configs/synthetic_identifiability_qualification_v6.yaml",
    "configs/synthetic_identifiability_qualification_v7.yaml",
    "configs/synthetic_identifiability_qualification_v8.yaml",
    "configs/synthetic_development_launch_v1.yaml",
    "configs/synthetic_development_launch_v2.yaml",
    "output/synthetic_identifiability_qualification_v8/terminal_decision.json",
    "output/synthetic_identifiability_qualification_v8/complete_file_manifest.json",
    "output/synthetic_identifiability_adversarial_audit_v8/run_2.json",
    "output/synthetic_identifiability_adversarial_audit_v8/complete_manifest.json",
    "output/synthetic_development_launch_package_v1/post_freeze_cycle_audit.json",
    "output/synthetic_development_launch_package_v2/post_freeze_cycle_audit.json",
    "output/synthetic_development_launch_package_v3/outcome_registry.json",
    "output/synthetic_development_launch_package_v3/fixture_rehearsal/cohort_summary.json",
    "output/synthetic_development_launch_package_v3/fixture_rehearsal/identifier_reference_audit.json",
    "output/synthetic_development_launch_package_v3/fixture_rehearsal/recomputable/development_controls.json",
    "output/synthetic_development_launch_package_v3/fixture_rehearsal/recomputable/development_inference.json",
    CONFIG,
    PROTOCOL,
    SOURCES,
    TRACEABILITY,
    LIMITATIONS,
    SAMPLE_SIZE,
    REPAIR_LEDGER,
)


def _verify_snapshot(snapshot_root: Path) -> None:
    manifest_path = snapshot_root / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("snapshot manifest missing")
    result = verify_file_manifest(
        snapshot_root, json.loads(manifest_path.read_text())
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"frozen source verification failed: {result}")


def _verify_development_authorization(
    authorization_path: Path,
    snapshot_root: Path,
    config: dict,
) -> dict:
    value = json.loads(authorization_path.read_text())
    if not verify_authorization(value):
        raise PermissionError("invalid development authorization payload")
    package_root = authorization_path.parent
    repository_root = package_root.parents[1]
    checks = {
        "package_terminal_sha256": sha256_file(
            package_root / "package_terminal.json"
        ),
        "freeze_receipt_sha256": sha256_file(
            package_root / "freeze_receipt.json"
        ),
        "snapshot_manifest_sha256": sha256_file(
            snapshot_root / "snapshot_manifest.json"
        ),
        "excluded_rehearsal_sha256": sha256_file(
            package_root / "excluded_rehearsal/cohort_summary.json"
        ),
        "adversarial_audit_sha256": sha256_file(
            repository_root
            / "output/synthetic_identifiability_adversarial_audit_v8/run_2.json"
        ),
        "official_tests_sha256": sha256_file(
            package_root / "official_test_execution_report.json"
        ),
    }
    mismatches = {
        key: {"expected": value[key], "observed": observed}
        for key, observed in checks.items()
        if str(value.get(key)) != observed
    }
    if mismatches:
        raise PermissionError(f"authorization provenance mismatch: {mismatches}")
    if value["development_registry"] != config["resolved_registries"]["development"]:
        raise PermissionError("authorization development registry mismatch")
    if value["confirmation_reserve"] != config["resolved_registries"][
        "confirmation_reserve"
    ]:
        raise PermissionError("authorization confirmation reserve mismatch")
    if value["exact_one_shot_command"] != config["commands"][
        "development_one_shot"
    ]:
        raise PermissionError("authorization command mismatch")
    package_manifest = json.loads(
        (package_root / "complete_file_manifest.json").read_text()
    )
    manifest_verification = verify_file_manifest(
        package_root, package_manifest
    )
    listed = {str(row["path"]) for row in package_manifest["files"]}
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and path != package_root / "complete_file_manifest.json"
    }
    if (
        manifest_verification["status"] != "PASS"
        or listed != actual
        or package_manifest["independent_second_pass_verification"]["status"]
        != "PASS"
    ):
        raise PermissionError("launch package complete manifest failed")
    outcome_registry = json.loads(
        (package_root / "outcome_registry.json").read_text()
    )
    if (
        int(outcome_registry["development_outcome_count"]) != 0
        or int(outcome_registry["confirmation_outcome_count"]) != 0
    ):
        raise PermissionError("launch package outcome registry is no longer zero")
    terminal = json.loads((package_root / "package_terminal.json").read_text())
    traceability = json.loads(
        (package_root / "traceability_validation.json").read_text()
    )
    tests = json.loads(
        (package_root / "official_test_execution_report.json").read_text()
    )
    if terminal.get("decision") != "DEVELOPMENT_PACKAGE_SEALED":
        raise PermissionError("launch package terminal is not sealed")
    if traceability.get("status") != "PASS":
        raise PermissionError("launch package traceability failed")
    if tests.get("status") != "PASS":
        raise PermissionError("official frozen tests failed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "baseline",
            "collect",
            "fixture",
            "freeze",
            "excluded-rehearsal",
            "recompute",
            "reproduce",
            "tests",
            "preservation",
            "finalize",
            "adjudicate",
            "run-development",
        ),
    )
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    if args.command == "recompute":
        snapshot = Path(args.snapshot_root).resolve()
        _verify_snapshot(snapshot)
        result = compute_from_persisted(args.input_root, args.output_root)
    elif args.command == "adjudicate":
        snapshot = Path(args.snapshot_root).resolve()
        _verify_snapshot(snapshot)
        result = adjudicate_file(args.input_path, args.output_path)
    elif args.command == "run-development":
        snapshot = Path(args.snapshot_root).resolve()
        _verify_snapshot(snapshot)
        config = load_config(
            snapshot / CONFIG,
            repository_root=snapshot,
        )
        require_frozen(config)
        authorization_path = Path(args.authorization).resolve()
        authorization = _verify_development_authorization(
            authorization_path,
            snapshot,
            config,
        )
        destination = Path(args.output_root).resolve()
        if destination.exists():
            raise FileExistsError(
                "one-shot development output already exists; rerun prohibited"
            )
        result = execute_cohort(
            snapshot,
            config,
            destination,
            purpose="development",
            development_authorized=True,
        )
        write_json(
            destination / "ONE_SHOT_EXECUTION.json",
            {
                "status": "COMPLETE",
                "protocol_id": config["protocol"]["id"],
                "authorization_digest": authorization["authorization_digest"],
                "development_outcome_count": 1,
                "confirmation_outcome_count": 0,
                "confirmation_authorized": False,
                "cohort_summary_sha256": sha256_file(
                    destination / "cohort_summary.json"
                ),
            },
        )
    else:
        config = load_config(ROOT / CONFIG, repository_root=ROOT)
        if args.command == "baseline":
            OUTPUT.mkdir(parents=True, exist_ok=True)
            if not (OUTPUT / "outcome_registry.json").exists():
                write_json(
                    OUTPUT / "outcome_registry.json",
                    {
                        "development_outcome_count": 0,
                        "confirmation_outcome_count": 0,
                        "development_output_exists": False,
                        "confirmation_output_exists": False,
                    },
                )
            result = write_prior_preservation_baseline(
                ROOT,
                OUTPUT / "preserved_prior_evidence_before.tsv",
            )
        elif args.command == "collect":
            executable = ROOT / str(config["environment"]["python"])
            result = collect_test_nodeids(
                ROOT,
                executable,
                str(config["environment"]["version_test_root"]),
            )
            write_json(
                OUTPUT / "collected_tests_pre_freeze.json",
                result,
                overwrite=(OUTPUT / "collected_tests_pre_freeze.json").exists(),
            )
        elif args.command == "fixture":
            result = execute_cohort(
                ROOT,
                config,
                OUTPUT / "fixture_rehearsal",
                purpose="fixture_rehearsal",
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
        elif args.command == "excluded-rehearsal":
            require_frozen(config)
            freeze = verify_freeze(ROOT, OUTPUT)
            if freeze["status"] != "PASS":
                raise RuntimeError(f"freeze failed before excluded rehearsal: {freeze}")
            result = execute_cohort(
                ROOT,
                config,
                OUTPUT / "excluded_rehearsal",
                purpose="excluded_rehearsal",
            )
        elif args.command == "reproduce":
            process = run_independent_recompute(OUTPUT)
            comparison = (
                compare_recomputation(OUTPUT)
                if process["exit_code"] == 0
                else {"status": "FAIL", "process": process}
            )
            if process["exit_code"] != 0:
                write_json(
                    OUTPUT / "independent_recompute_comparison.json",
                    comparison,
                )
            negative = run_negative_recompute_tests(OUTPUT)
            result = {
                "process": process,
                "comparison": comparison,
                "negative": negative,
            }
        elif args.command == "tests":
            result = run_official_tests(ROOT, OUTPUT)
        elif args.command == "preservation":
            result = verify_prior_preservation(
                ROOT,
                OUTPUT / "preserved_prior_evidence_before.tsv",
                OUTPUT / "preserved_prior_evidence_after.tsv",
                OUTPUT / "preservation_proof.json",
            )
        else:
            specification = json.loads(
                (OUTPUT / "frozen_traceability.json").read_text()
            )
            result = finalize_package(ROOT, OUTPUT, config, specification)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
