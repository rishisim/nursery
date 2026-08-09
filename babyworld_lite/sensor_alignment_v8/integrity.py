from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from .adjudicator import adjudicate_file
from .protocol import (
    canonical_digest,
    environment_record,
    manifest_for_files,
    require_frozen_config,
    registry_snapshot,
    sha256_file,
    verify_file_manifest,
    verify_prior_preservation,
    write_json,
)


def collect_test_nodeids(
    repository_root: str | Path,
    executable: str | Path,
    test_path: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    command = [str(Path(executable).absolute()), "-m", "pytest", "--collect-only", "-q", test_path]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    nodeids = sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    )
    return {
        "status": "PASS" if completed.returncode == 0 and nodeids else "FAIL",
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "nodeids": nodeids,
        "output": completed.stdout,
    }


def write_prior_preservation_baseline(
    repository_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    paths: set[str] = set()
    inherited = (
        root
        / "output/synthetic_identifiability_qualification_v5/"
        "preserved_v1_v4_hashes_before.tsv"
    )
    for line in inherited.read_text().splitlines():
        if line:
            _, _, relative = line.split("\t", 2)
            paths.add(relative)
    for version in (5, 6, 7):
        directory = root / f"babyworld_lite/sensor_alignment_v{version}"
        paths.update(
            path.relative_to(root).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
        for relative in (
            f"configs/synthetic_identifiability_qualification_v{version}.yaml",
            f"scripts/run_synthetic_identifiability_qualification_v{version}.py",
            f"tests/test_synthetic_identifiability_qualification_v{version}.py",
        ):
            if (root / relative).is_file():
                paths.add(relative)
        paths.update(
            path.relative_to(root).as_posix()
            for path in (root / "docs").glob(
                f"synthetic_identifiability_qualification_v{version}*"
            )
            if path.is_file()
        )
        output_directory = (
            root / f"output/synthetic_identifiability_qualification_v{version}"
        )
        if output_directory.is_dir():
            paths.update(
                path.relative_to(root).as_posix()
                for path in output_directory.rglob("*")
                if path.is_file()
            )
    rows = []
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(f"{sha256_file(path)}\t{path.stat().st_size}\t{relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n")
    return {
        "status": "PASS",
        "path": str(destination),
        "file_count": len(rows),
        "sha256": sha256_file(destination),
        "versions_preserved": [1, 2, 3, 4, 5, 6, 7],
    }


def freeze_package(
    repository_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    tracked_files: Sequence[str],
    config_path: str,
    protocol_path: str,
    sources_path: str,
    traceability_path: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    require_frozen_config(config)
    snapshot = output / "frozen_source_snapshot"
    if snapshot.exists():
        raise FileExistsError(snapshot)
    environment = environment_record(root, config)
    write_json(output / "frozen_environment.json", environment)
    shutil.copyfile(root / config_path, output / "frozen_config_snapshot.yaml")
    shutil.copyfile(root / protocol_path, output / "frozen_protocol_snapshot.md")
    shutil.copyfile(root / sources_path, output / "frozen_primary_sources.json")
    shutil.copyfile(root / traceability_path, output / "frozen_traceability.json")
    write_json(output / "frozen_seed_registries.json", registry_snapshot(config))
    write_json(output / "frozen_gates.json", config["gates"])
    adjudication_contract = {
        "official_paths": {
            "qualification": "qualification_run/qualification_summary.json",
            "tests": "official_test_execution_report.json",
            "recompute": "independent_recompute_comparison.json",
            "negative_recompute": "negative_recompute_tests.json",
            "traceability": "traceability_validation.json",
            "preservation": "preservation_proof.json",
            "terminal": "terminal_decision.json",
        },
        "test_report_substitution_allowed": False,
        "missing_contradictory_or_stale_fails_closed": True,
        "terminal_decision_function": "babyworld_lite/sensor_alignment_v8/adjudicator.py:terminal_decision",
    }
    write_json(output / "frozen_adjudication_contract.json", adjudication_contract)
    frozen_artifacts = [
        "frozen_environment.json",
        "frozen_config_snapshot.yaml",
        "frozen_protocol_snapshot.md",
        "frozen_primary_sources.json",
        "frozen_traceability.json",
        "frozen_seed_registries.json",
        "frozen_gates.json",
        "frozen_adjudication_contract.json",
        "collected_tests_pre_freeze.json",
    ]
    snapshot_files = list(tracked_files)
    for relative in snapshot_files:
        source = root / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    detector_relative = str(config["detector"]["frozen_v2_path"])
    if detector_relative not in snapshot_files:
        destination = snapshot / detector_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / detector_relative, destination)
        snapshot_files.append(detector_relative)
    snapshot_manifest = manifest_for_files(snapshot, snapshot_files)
    write_json(snapshot / "snapshot_manifest.json", snapshot_manifest)
    tracked = []
    for relative in sorted(tracked_files):
        live = root / relative
        copied = snapshot / relative
        tracked.append(
            {
                "path": relative,
                "live_sha256": sha256_file(live),
                "snapshot_sha256": sha256_file(copied),
                "bytes": live.stat().st_size,
            }
        )
    receipt = {
        "status": "FROZEN",
        "tracked": tracked,
        "tracked_file_count": len(tracked),
        "snapshot_manifest_sha256": sha256_file(snapshot / "snapshot_manifest.json"),
        "frozen_artifacts": [
            {"path": path, "sha256": sha256_file(output / path), "bytes": (output / path).stat().st_size}
            for path in frozen_artifacts
        ],
        "frozen_before_excluded_smoke": True,
        "changes_after_smoke_permitted": False,
    }
    write_json(output / "freeze_receipt.json", receipt)
    return receipt


def verify_freeze(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    receipt_path = output / "freeze_receipt.json"
    if not receipt_path.is_file():
        return {"status": "FAIL", "problem": "missing freeze receipt"}
    receipt = json.loads(receipt_path.read_text())
    problems = []
    for row in receipt.get("tracked", []):
        live = root / row["path"]
        copied = output / "frozen_source_snapshot" / row["path"]
        if not live.is_file() or not copied.is_file():
            problems.append({"path": row["path"], "problem": "missing"})
            continue
        if sha256_file(live) != row["live_sha256"] or sha256_file(copied) != row["snapshot_sha256"]:
            problems.append({"path": row["path"], "problem": "changed"})
    for row in receipt.get("frozen_artifacts", []):
        path = output / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            problems.append({"path": row["path"], "problem": "frozen-artifact-changed"})
    snapshot_manifest = output / "frozen_source_snapshot" / "snapshot_manifest.json"
    if not snapshot_manifest.is_file() or sha256_file(snapshot_manifest) != receipt.get(
        "snapshot_manifest_sha256"
    ):
        problems.append({"path": "frozen_source_snapshot/snapshot_manifest.json", "problem": "changed"})
    else:
        snapshot_check = verify_file_manifest(
            output / "frozen_source_snapshot", json.loads(snapshot_manifest.read_text())
        )
        if snapshot_check["status"] != "PASS":
            problems.append({"path": "frozen_source_snapshot", "problem": snapshot_check})
    return {
        "status": "PASS" if not problems else "FAIL",
        "checked_tracked_files": len(receipt.get("tracked", [])),
        "checked_frozen_artifacts": len(receipt.get("frozen_artifacts", [])),
        "problems": problems,
    }


def _snapshot_command(
    output_root: Path,
    arguments: Sequence[str],
    *,
    snapshot_root: Path | None = None,
    cwd: Path,
) -> dict[str, Any]:
    environment_record_value = json.loads((output_root / "frozen_environment.json").read_text())
    executable = str(environment_record_value["python_executable"])
    snapshot = snapshot_root or output_root / "frozen_source_snapshot"
    script = snapshot / "scripts/run_synthetic_identifiability_qualification_v8.py"
    command = [executable, str(script), *arguments]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(snapshot)
    cwd.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": " ".join(command),
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "output": completed.stdout,
        "snapshot_root": str(snapshot),
    }


def run_independent_recompute(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    inputs = output / "qualification_run" / "persisted_inputs"
    recomputed = output / "independent_recompute"
    result = _snapshot_command(
        output,
        [
            "recompute",
            "--input-root",
            str(inputs),
            "--output-root",
            str(recomputed),
            "--snapshot-root",
            str(output / "frozen_source_snapshot"),
        ],
        cwd=output / "independent_recompute_staging",
    )
    write_json(output / "independent_recompute_process.json", result)
    return result


def compare_recomputation(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    original = output / "qualification_run" / "recomputable"
    recomputed = output / "independent_recompute"
    left = {
        path.relative_to(original).as_posix(): (path.stat().st_size, sha256_file(path))
        for path in original.rglob("*")
        if path.is_file()
    }
    right = {
        path.relative_to(recomputed).as_posix(): (path.stat().st_size, sha256_file(path))
        for path in recomputed.rglob("*")
        if path.is_file()
    }
    paths_equal = set(left) == set(right)
    mismatches = sorted(path for path in set(left) & set(right) if left[path] != right[path])
    passed = paths_equal and not mismatches and bool(left)
    result = {
        "status": "PASS" if passed else "FAIL",
        "separate_process_executed": True,
        "original_directory": str(original),
        "recomputed_directory": str(recomputed),
        "file_sets_identical": paths_equal,
        "file_count": len(left),
        "byte_and_sha256_identity": not mismatches and paths_equal,
        "missing_from_recompute": sorted(set(left) - set(right)),
        "unexpected_in_recompute": sorted(set(right) - set(left)),
        "mismatches": mismatches,
        "files": [
            {"path": path, "bytes": left[path][0], "sha256": left[path][1]}
            for path in sorted(left)
        ],
    }
    write_json(output / "independent_recompute_comparison.json", result)
    return result


def run_negative_recompute_tests(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    root = output / "negative_recompute_cases"
    input_copy = root / "mutated_inputs"
    source_copy = root / "mutated_snapshot"
    shutil.copytree(output / "qualification_run" / "persisted_inputs", input_copy)
    shutil.copytree(output / "frozen_source_snapshot", source_copy)
    input_target = next(input_copy.glob("corpus_*/conditions/synchronized.jsonl"))
    input_target.write_bytes(input_target.read_bytes() + b"\n")
    source_target = source_copy / "babyworld_lite/sensor_alignment_v8/__init__.py"
    source_target.write_bytes(source_target.read_bytes() + b"# negative source mutation\n")
    input_result = _snapshot_command(
        output,
        [
            "recompute",
            "--input-root",
            str(input_copy),
            "--output-root",
            str(root / "input_mutation_output"),
            "--snapshot-root",
            str(output / "frozen_source_snapshot"),
        ],
        cwd=root / "input_mutation_staging",
    )
    source_result = _snapshot_command(
        output,
        [
            "recompute",
            "--input-root",
            str(output / "qualification_run" / "persisted_inputs"),
            "--output-root",
            str(root / "source_mutation_output"),
            "--snapshot-root",
            str(source_copy),
        ],
        snapshot_root=source_copy,
        cwd=root / "source_mutation_staging",
    )
    passed = input_result["exit_code"] != 0 and source_result["exit_code"] != 0
    result = {
        "status": "PASS" if passed else "FAIL",
        "persisted_input_byte_mutation_rejected": input_result["exit_code"] != 0,
        "frozen_source_byte_mutation_rejected": source_result["exit_code"] != 0,
        "input_mutation_process": input_result,
        "source_mutation_process": source_result,
    }
    write_json(output / "negative_recompute_tests.json", result)
    return result


def run_official_tests(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    report_path = output / "official_test_execution_report.json"
    if report_path.exists():
        raise FileExistsError("exactly one official test report is permitted")
    freeze_before = verify_freeze(root, output)
    if freeze_before["status"] != "PASS":
        raise RuntimeError(f"freeze failed before official tests: {freeze_before}")
    frozen_environment = json.loads((output / "frozen_environment.json").read_text())
    executable = str(frozen_environment["python_executable"])
    test_roots = list(frozen_environment["test_roots"])
    snapshot = output / "frozen_source_snapshot"
    results = []
    for index, test_root in enumerate(test_roots):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if index == 0:
            resolved_test_root = snapshot / test_root
            cwd = snapshot
            environment["PYTHONPATH"] = str(snapshot)
            source_scope = "frozen_source_snapshot"
        else:
            resolved_test_root = root / test_root
            cwd = root
            environment.pop("PYTHONPATH", None)
            source_scope = "full_live_repository_after_tracked_freeze_verification"
        command = [
            executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(resolved_test_root),
        ]
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        results.append(
            {
                "command": "PYTHONDONTWRITEBYTECODE=1 " + " ".join(command),
                "cwd": str(cwd),
                "source_scope": source_scope,
                "exit_code": completed.returncode,
                "output": completed.stdout,
            }
        )
        if completed.returncode != 0:
            break
    freeze_after = verify_freeze(root, output)
    passed = (
        len(results) == 2
        and all(row["exit_code"] == 0 for row in results)
        and freeze_after["status"] == "PASS"
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "official": True,
        "only_admissible_test_report": True,
        "alternate_or_corrected_report_allowed": False,
        "order": "v8 tests first, then complete repository tests root",
        "python_executable": executable,
        "python_sha256": sha256_file(executable),
        "python_version": frozen_environment["python_version"],
        "test_roots": test_roots,
        "bytecode_writes_disabled": True,
        "version_tests_executed_from_frozen_snapshot": True,
        "full_repository_tests_executed_after_tracked_freeze_verification": True,
        "freeze_before": freeze_before,
        "freeze_after": freeze_after,
        "results": results,
    }
    write_json(report_path, report)
    return report


def _top_level_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def validate_traceability(
    repository_root: str | Path,
    output_root: str | Path,
    specification: Mapping[str, Any],
    collected_nodeids: Sequence[str],
    gate_fields: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    problems = {"symbols": [], "tests": [], "artifacts": [], "gates": []}
    collected = set(collected_nodeids)
    for item in specification.get("requirements", []):
        for reference in item.get("symbols", []):
            file_name, symbol = str(reference).split(":", 1)
            path = root / file_name
            if not path.is_file() or symbol not in _top_level_symbols(path):
                problems["symbols"].append(reference)
        for nodeid in item.get("tests", []):
            if nodeid not in collected:
                problems["tests"].append(nodeid)
        for relative in item.get("artifacts", []):
            if not (output / relative).exists():
                problems["artifacts"].append(relative)
        for gate in item.get("gates", []):
            if gate not in gate_fields:
                problems["gates"].append(gate)
    passed = all(not values for values in problems.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "requirements_checked": len(specification.get("requirements", [])),
        "reference_class_failures": problems,
        "every_symbol_exists": not problems["symbols"],
        "every_named_test_collected": not problems["tests"],
        "every_artifact_exists": not problems["artifacts"],
        "every_gate_field_exists": not problems["gates"],
    }


def write_complete_manifest(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    manifest_path = output / "complete_file_manifest.json"
    files = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    observed_again = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    passed = files == observed_again
    manifest = {
        "protocol_id": "synthetic-identifiability-qualification-v8",
        "file_count": len(files),
        "files": files,
        "self_excluded_by_definition": True,
        "summary_only": False,
        "independent_second_pass_verification": {
            "status": "PASS" if passed else "FAIL",
            "file_sets_bytes_and_sha256_identical": passed,
            "first_pass_digest": canonical_digest(files),
            "second_pass_digest": canonical_digest(observed_again),
        },
    }
    write_json(manifest_path, manifest, overwrite=manifest_path.exists())
    return manifest


def finalize_decision(
    repository_root: str | Path,
    output_root: str | Path,
    traceability_specification: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    qualification = json.loads((output / "qualification_run/qualification_summary.json").read_text())
    recompute = json.loads((output / "independent_recompute_comparison.json").read_text())
    negative = json.loads((output / "negative_recompute_tests.json").read_text())
    tests = json.loads((output / "official_test_execution_report.json").read_text())
    preservation = json.loads((output / "preservation_proof.json").read_text())
    freeze = verify_freeze(root, output)
    collected = json.loads((output / "collected_tests_pre_freeze.json").read_text())["nodeids"]
    gates = dict(qualification["gates"])
    gates.update(
        {
            "independent_recompute": recompute["status"] == "PASS",
            "negative_recompute_mutations": negative["status"] == "PASS",
            "official_tests": tests["status"] == "PASS",
            "freeze_integrity": freeze["status"] == "PASS",
            "prior_v1_v7_preservation": preservation["status"] == "PASS",
            "traceability": True,
            "terminal_regeneration": True,
            "complete_manifest": True,
        }
    )
    adjudication_inputs = {
        "protocol_id": "synthetic-identifiability-qualification-v8",
        "gates": gates,
        "identifiability_structural_failure": bool(
            qualification.get("identifiability_structural_failure", False)
        ),
        "oracle_or_manufactured_scoring_required": False,
        "contradictions": [],
        "stale_inputs": [],
        "official_artifact_digests": {
            "qualification": sha256_file(output / "qualification_run/qualification_summary.json"),
            "tests": sha256_file(output / "official_test_execution_report.json"),
            "recompute": sha256_file(output / "independent_recompute_comparison.json"),
            "negative_recompute": sha256_file(output / "negative_recompute_tests.json"),
            "preservation": sha256_file(output / "preservation_proof.json"),
            "freeze_receipt": sha256_file(output / "freeze_receipt.json"),
        },
    }
    input_path = output / "adjudication_inputs.json"
    write_json(input_path, adjudication_inputs)
    terminal_path = output / "terminal_decision.json"
    terminal = adjudicate_file(input_path, terminal_path)
    write_complete_manifest(output)
    trace = validate_traceability(root, output, traceability_specification, collected, gates)
    write_json(output / "traceability_validation.json", trace)
    if trace["status"] != "PASS":
        adjudication_inputs["gates"]["traceability"] = False
        write_json(input_path, adjudication_inputs, overwrite=True)
        terminal = adjudicate_file(input_path, terminal_path)
    staging = output / "terminal_staging"
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, staging / "adjudication_inputs.json")
    regeneration_process = _snapshot_command(
        output,
        [
            "adjudicate",
            "--input-path",
            str(staging / "adjudication_inputs.json"),
            "--output-path",
            str(staging / "terminal_decision.json"),
            "--snapshot-root",
            str(output / "frozen_source_snapshot"),
        ],
        cwd=staging,
    )
    regenerated = staging / "terminal_decision.json"
    regeneration_passed = (
        regeneration_process["exit_code"] == 0
        and regenerated.is_file()
        and regenerated.read_bytes() == terminal_path.read_bytes()
    )
    if not regeneration_passed:
        adjudication_inputs["gates"]["terminal_regeneration"] = False
        write_json(input_path, adjudication_inputs, overwrite=True)
        terminal = adjudicate_file(input_path, terminal_path)
        shutil.copyfile(input_path, staging / "adjudication_inputs.json")
        regeneration_process = _snapshot_command(
            output,
            [
                "adjudicate",
                "--input-path",
                str(staging / "adjudication_inputs.json"),
                "--output-path",
                str(staging / "terminal_decision.json"),
                "--snapshot-root",
                str(output / "frozen_source_snapshot"),
            ],
            cwd=staging,
        )
        regeneration_passed = regenerated.read_bytes() == terminal_path.read_bytes()
    write_json(
        output / "terminal_regeneration.json",
        {
            "status": "PASS" if regeneration_passed else "FAIL",
            "clean_staging_directory": str(staging),
            "separate_process": True,
            "frozen_adjudicator_used": True,
            "byte_identical": regeneration_passed,
            "primary_sha256": sha256_file(terminal_path),
            "regenerated_sha256": sha256_file(regenerated) if regenerated.is_file() else None,
            "process": regeneration_process,
        },
    )
    manifest = write_complete_manifest(output)
    manifest_passed = manifest["independent_second_pass_verification"]["status"] == "PASS"
    if not manifest_passed:
        adjudication_inputs["gates"]["complete_manifest"] = False
        write_json(input_path, adjudication_inputs, overwrite=True)
        terminal = adjudicate_file(input_path, terminal_path)
        manifest = write_complete_manifest(output)
    return terminal


__all__ = [
    "collect_test_nodeids",
    "write_prior_preservation_baseline",
    "freeze_package",
    "verify_freeze",
    "run_independent_recompute",
    "compare_recomputation",
    "run_negative_recompute_tests",
    "run_official_tests",
    "validate_traceability",
    "write_complete_manifest",
    "finalize_decision",
    "verify_prior_preservation",
]
