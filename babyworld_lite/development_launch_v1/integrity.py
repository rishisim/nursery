from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .adjudicator import (
    adjudicate_file,
    authorization_payload,
    package_decision,
)
from .protocol import (
    canonical_digest,
    manifest_for_files,
    registry_snapshot,
    require_frozen,
    sha256_file,
    verify_file_manifest,
    write_json,
)


def collect_test_nodeids(
    repository_root: str | Path,
    executable: str | Path,
    test_path: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    command = [
        str(Path(executable).absolute()),
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        test_path,
    ]
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


def environment_record(
    repository_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    executable = (root / str(config["environment"]["python"])).resolve()
    return {
        "python_executable": str(executable),
        "python_sha256": sha256_file(executable),
        "python_version": sys.version,
        "bytecode_writes_disabled": True,
        "test_roots": [
            str(config["environment"]["version_test_root"]),
            str(config["environment"]["repository_test_root"]),
        ],
        "official_commands": [
            (
                f"PYTHONDONTWRITEBYTECODE=1 {executable} -m pytest -q "
                f"-p no:cacheprovider {config['environment']['version_test_root']}"
            ),
            (
                f"PYTHONDONTWRITEBYTECODE=1 {executable} -m pytest -q "
                f"-p no:cacheprovider {config['environment']['repository_test_root']}"
            ),
        ],
    }


def write_prior_preservation_baseline(
    repository_root: str | Path, output_path: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    paths: set[str] = set()
    inherited = (
        root
        / "output/synthetic_identifiability_qualification_v8/"
        "preserved_v1_v7_hashes_before.tsv"
    )
    for line in inherited.read_text().splitlines():
        if line:
            _, _, relative = line.split("\t", 2)
            paths.add(relative)
    for directory in (
        root / "babyworld_lite/sensor_alignment_v8",
        root / "output/synthetic_identifiability_qualification_v8",
        root / "output/synthetic_identifiability_adversarial_audit_v8",
    ):
        paths.update(
            path.relative_to(root).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    for relative in (
        "configs/synthetic_identifiability_qualification_v8.yaml",
        "scripts/run_synthetic_identifiability_qualification_v8.py",
        "tests/test_synthetic_identifiability_qualification_v8.py",
    ):
        paths.add(relative)
    paths.update(
        path.relative_to(root).as_posix()
        for path in (root / "docs").glob(
            "synthetic_identifiability_qualification_v8*"
        )
        if path.is_file()
    )
    rows = []
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            f"{sha256_file(path)}\t{path.stat().st_size}\t{relative}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n")
    return {
        "status": "PASS",
        "path": str(destination),
        "file_count": len(rows),
        "sha256": sha256_file(destination),
        "versions_preserved": [1, 2, 3, 4, 5, 6, 7, 8],
        "adversarial_audit_preserved": True,
    }


def verify_prior_preservation(
    repository_root: str | Path,
    baseline_path: str | Path,
    after_path: str | Path,
    proof_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    baseline = Path(baseline_path).resolve()
    after = Path(after_path).resolve()
    if after.exists() or Path(proof_path).exists():
        raise FileExistsError("preservation proof already exists")
    before_rows = {}
    for line in baseline.read_text().splitlines():
        if line:
            digest, size, relative = line.split("\t", 2)
            before_rows[relative] = (digest, int(size))
    after_lines = []
    missing = []
    changed = []
    for relative, expected in sorted(before_rows.items()):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        observed = (sha256_file(path), path.stat().st_size)
        after_lines.append(f"{observed[0]}\t{observed[1]}\t{relative}")
        if observed != expected:
            changed.append(relative)
    after.write_text("\n".join(after_lines) + "\n")
    proof = {
        "status": "PASS" if not missing and not changed else "FAIL",
        "baseline_file_count": len(before_rows),
        "after_file_count": len(after_lines),
        "before_manifest_sha256": sha256_file(baseline),
        "after_manifest_sha256": sha256_file(after),
        "byte_for_byte_preserved": not missing and not changed,
        "missing_paths": missing,
        "changed_paths": changed,
    }
    write_json(proof_path, proof)
    return proof


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
    require_frozen(config)
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
    write_json(output / "frozen_analysis.json", config["analysis"])
    write_json(output / "frozen_qualification_gates.json", config["gates"])
    write_json(
        output / "frozen_output_contract.json",
        {
            "development_output_root": "output/synthetic_development_v1_one_shot",
            "development_output_must_not_exist_before_run": True,
            "one_shot_marker": "ONE_SHOT_EXECUTION.json",
            "confirmation_output_supported": False,
            "development_outcome_count_before_run": 0,
            "confirmation_outcome_count_before_run": 0,
            "exact_one_shot_command": config["commands"]["development_one_shot"],
        },
    )
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
    frozen_artifacts = [
        "frozen_environment.json",
        "frozen_config_snapshot.yaml",
        "frozen_protocol_snapshot.md",
        "frozen_primary_sources.json",
        "frozen_traceability.json",
        "frozen_seed_registries.json",
        "frozen_analysis.json",
        "frozen_qualification_gates.json",
        "frozen_output_contract.json",
        "collected_tests_pre_freeze.json",
    ]
    tracked = [
        {
            "path": relative,
            "live_sha256": sha256_file(root / relative),
            "snapshot_sha256": sha256_file(snapshot / relative),
            "bytes": (root / relative).stat().st_size,
        }
        for relative in sorted(tracked_files)
    ]
    receipt = {
        "status": "FROZEN",
        "tracked": tracked,
        "tracked_file_count": len(tracked),
        "snapshot_manifest_sha256": sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        "frozen_artifacts": [
            {
                "path": relative,
                "sha256": sha256_file(output / relative),
                "bytes": (output / relative).stat().st_size,
            }
            for relative in frozen_artifacts
        ],
        "frozen_before_excluded_rehearsal": True,
        "changes_after_excluded_rehearsal_permitted": False,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }
    write_json(output / "freeze_receipt.json", receipt)
    return receipt


def verify_freeze(
    repository_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    receipt_path = output / "freeze_receipt.json"
    if not receipt_path.is_file():
        return {"status": "FAIL", "problem": "missing freeze receipt"}
    receipt = json.loads(receipt_path.read_text())
    problems = []
    for row in receipt.get("tracked", []):
        live = root / str(row["path"])
        copied = output / "frozen_source_snapshot" / str(row["path"])
        if not live.is_file() or not copied.is_file():
            problems.append({"path": row["path"], "problem": "missing"})
            continue
        if (
            sha256_file(live) != row["live_sha256"]
            or sha256_file(copied) != row["snapshot_sha256"]
        ):
            problems.append({"path": row["path"], "problem": "changed"})
    for row in receipt.get("frozen_artifacts", []):
        path = output / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            problems.append(
                {"path": row["path"], "problem": "frozen-artifact-changed"}
            )
    snapshot_manifest = (
        output / "frozen_source_snapshot" / "snapshot_manifest.json"
    )
    if (
        not snapshot_manifest.is_file()
        or sha256_file(snapshot_manifest)
        != receipt.get("snapshot_manifest_sha256")
    ):
        problems.append(
            {
                "path": "frozen_source_snapshot/snapshot_manifest.json",
                "problem": "changed",
            }
        )
    else:
        verification = verify_file_manifest(
            output / "frozen_source_snapshot",
            json.loads(snapshot_manifest.read_text()),
        )
        if verification["status"] != "PASS":
            problems.append(
                {"path": "frozen_source_snapshot", "problem": verification}
            )
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
    cwd: Path,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    environment_value = json.loads(
        (output_root / "frozen_environment.json").read_text()
    )
    executable = str(environment_value["python_executable"])
    snapshot = snapshot_root or output_root / "frozen_source_snapshot"
    script = snapshot / "scripts/run_synthetic_development_launch_v1.py"
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
    result = _snapshot_command(
        output,
        [
            "recompute",
            "--input-root",
            str(output / "excluded_rehearsal/persisted_inputs"),
            "--output-root",
            str(output / "independent_recompute"),
            "--snapshot-root",
            str(output / "frozen_source_snapshot"),
        ],
        cwd=output / "independent_recompute_staging",
    )
    write_json(output / "independent_recompute_process.json", result)
    return result


def compare_recomputation(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    original = output / "excluded_rehearsal/recomputable"
    recomputed = output / "independent_recompute"
    left = {
        path.relative_to(original).as_posix(): (
            path.stat().st_size,
            sha256_file(path),
        )
        for path in original.rglob("*")
        if path.is_file()
    }
    right = {
        path.relative_to(recomputed).as_posix(): (
            path.stat().st_size,
            sha256_file(path),
        )
        for path in recomputed.rglob("*")
        if path.is_file()
    }
    mismatches = sorted(
        path for path in set(left) & set(right) if left[path] != right[path]
    )
    result = {
        "status": (
            "PASS"
            if left and set(left) == set(right) and not mismatches
            else "FAIL"
        ),
        "separate_process_executed": True,
        "file_count": len(left),
        "file_sets_identical": set(left) == set(right),
        "byte_and_sha256_identity": set(left) == set(right) and not mismatches,
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
    shutil.copytree(output / "excluded_rehearsal/persisted_inputs", input_copy)
    shutil.copytree(output / "frozen_source_snapshot", source_copy)
    input_target = next(
        input_copy.glob("corpus_*/conditions/synchronized.jsonl")
    )
    input_target.write_bytes(input_target.read_bytes() + b"\n")
    source_target = (
        source_copy / "babyworld_lite/development_launch_v1/__init__.py"
    )
    source_target.write_bytes(
        source_target.read_bytes() + b"# negative mutation\n"
    )
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
            str(output / "excluded_rehearsal/persisted_inputs"),
            "--output-root",
            str(root / "source_mutation_output"),
            "--snapshot-root",
            str(source_copy),
        ],
        cwd=root / "source_mutation_staging",
        snapshot_root=source_copy,
    )
    result = {
        "status": (
            "PASS"
            if input_result["exit_code"] != 0
            and source_result["exit_code"] != 0
            else "FAIL"
        ),
        "persisted_input_byte_mutation_rejected": input_result["exit_code"] != 0,
        "frozen_source_byte_mutation_rejected": source_result["exit_code"] != 0,
        "input_mutation_process": input_result,
        "source_mutation_process": source_result,
    }
    write_json(output / "negative_recompute_tests.json", result)
    return result


def run_official_tests(
    repository_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    report_path = output / "official_test_execution_report.json"
    if report_path.exists():
        raise FileExistsError("exactly one official report is permitted")
    before = verify_freeze(root, output)
    if before["status"] != "PASS":
        raise RuntimeError(f"freeze failed before tests: {before}")
    frozen_environment = json.loads(
        (output / "frozen_environment.json").read_text()
    )
    executable = str(frozen_environment["python_executable"])
    snapshot = output / "frozen_source_snapshot"
    results = []
    for index, test_root in enumerate(frozen_environment["test_roots"]):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if index == 0:
            resolved = snapshot / test_root
            cwd = snapshot
            environment["PYTHONPATH"] = str(snapshot)
            scope = "frozen_source_snapshot"
        else:
            resolved = root / test_root
            cwd = root
            environment.pop("PYTHONPATH", None)
            scope = "full_live_repository_after_tracked_freeze_verification"
        command = [
            executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            str(resolved),
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
                "source_scope": scope,
                "exit_code": completed.returncode,
                "output": completed.stdout,
            }
        )
        if completed.returncode != 0:
            break
    after = verify_freeze(root, output)
    report = {
        "status": (
            "PASS"
            if len(results) == 2
            and all(row["exit_code"] == 0 for row in results)
            and after["status"] == "PASS"
            else "FAIL"
        ),
        "official": True,
        "only_admissible_test_report": True,
        "alternate_or_corrected_report_allowed": False,
        "order": "development launch tests first, then complete repository tests",
        "python_executable": executable,
        "python_sha256": sha256_file(executable),
        "python_version": frozen_environment["python_version"],
        "bytecode_writes_disabled": True,
        "freeze_before": before,
        "freeze_after": after,
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
        "protocol_id": "synthetic-development-launch-v1",
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


def finalize_package(
    repository_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    traceability_specification: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    excluded = json.loads(
        (output / "excluded_rehearsal/cohort_summary.json").read_text()
    )
    fixture = json.loads(
        (output / "fixture_rehearsal_4/cohort_summary.json").read_text()
    )
    recompute = json.loads(
        (output / "independent_recompute_comparison.json").read_text()
    )
    negative = json.loads((output / "negative_recompute_tests.json").read_text())
    tests = json.loads(
        (output / "official_test_execution_report.json").read_text()
    )
    preservation = json.loads((output / "preservation_proof.json").read_text())
    freeze = verify_freeze(root, output)
    v8_terminal = json.loads(
        (
            root
            / "output/synthetic_identifiability_qualification_v8/"
            "terminal_decision.json"
        ).read_text()
    )
    adversarial = json.loads(
        (
            root
            / "output/synthetic_identifiability_adversarial_audit_v8/run_2.json"
        ).read_text()
    )
    identifier = json.loads(
        (output / "excluded_rehearsal/identifier_reference_audit.json").read_text()
    )
    registries = config["resolved_registries"]
    one_shot_root = root / "output/synthetic_development_v1_one_shot"
    outcome_registry = json.loads((output / "outcome_registry.json").read_text())
    gates = {
        "v8_version_ready": v8_terminal["decision"] == "VERSION_READY",
        "v8_adversarial_audit_pass": adversarial["status"] == "PASS",
        "fixture_rehearsal_pass": fixture["status"] == "PASS",
        "excluded_rehearsal_pass": excluded["status"] == "PASS",
        "exact_runner_independent_reproduction": recompute["status"] == "PASS",
        "negative_source_and_input_mutations": negative["status"] == "PASS",
        "official_tests": tests["status"] == "PASS",
        "freeze_integrity": freeze["status"] == "PASS",
        "prior_v1_v8_preservation": preservation["status"] == "PASS",
        "identifier_registry_reconciliation": identifier["status"] == "PASS",
        "development_registry_has_40_corpora": len(
            registries["development"]["corpus"]
        )
        == 40,
        "three_stochastic_model_replicates": len(
            registries["development"]["model"]
        )
        == 3,
        "inference_contract_frozen": (
            config["analysis"]["independent_unit"] == "corpus_seed"
            and config["analysis"]["model_replicate_handling"]
            == "average_within_corpus_before_inference"
        ),
        "cue_free_endpoints_frozen": all(
            config["design"]["held_out_endpoints"].values()
        ),
        "side_modality_firewall": bool(
            config["firewalls"]["side_modality_evaluation_fields_forbidden"]
        ),
        "confirmation_command_absent": not bool(
            config["firewalls"]["confirmation_command_exists"]
        ),
        "development_outcome_count_zero": int(
            outcome_registry["development_outcome_count"]
        )
        == 0,
        "confirmation_outcome_count_zero": int(
            outcome_registry["confirmation_outcome_count"]
        )
        == 0,
        "one_shot_output_absent": not one_shot_root.exists(),
        "one_shot_command_frozen": bool(
            config["commands"]["development_one_shot"]
        ),
        "traceability": True,
        "complete_manifest": True,
        "terminal_regeneration": True,
    }
    collected = json.loads(
        (output / "collected_tests_pre_freeze.json").read_text()
    )["nodeids"]
    trace = validate_traceability(
        root,
        output,
        traceability_specification,
        collected,
        gates,
    )
    gates["traceability"] = trace["status"] == "PASS"
    write_json(output / "traceability_validation.json", trace)
    adjudication_inputs = {
        "protocol_id": "synthetic-development-launch-v1",
        "gates": gates,
        "contradictions": [],
        "stale_inputs": [],
        "official_artifact_digests": {
            "fixture_rehearsal": sha256_file(
                output / "fixture_rehearsal_4/cohort_summary.json"
            ),
            "excluded_rehearsal": sha256_file(
                output / "excluded_rehearsal/cohort_summary.json"
            ),
            "tests": sha256_file(
                output / "official_test_execution_report.json"
            ),
            "recompute": sha256_file(
                output / "independent_recompute_comparison.json"
            ),
            "negative_recompute": sha256_file(
                output / "negative_recompute_tests.json"
            ),
            "preservation": sha256_file(output / "preservation_proof.json"),
            "freeze_receipt": sha256_file(output / "freeze_receipt.json"),
            "v8_terminal": sha256_file(
                root
                / "output/synthetic_identifiability_qualification_v8/"
                "terminal_decision.json"
            ),
            "v8_adversarial": sha256_file(
                root
                / "output/synthetic_identifiability_adversarial_audit_v8/"
                "run_2.json"
            ),
        },
    }
    input_path = output / "adjudication_inputs.json"
    write_json(input_path, adjudication_inputs)
    terminal_path = output / "package_terminal.json"
    terminal = adjudicate_file(input_path, terminal_path)
    staging = output / "terminal_staging"
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, staging / "adjudication_inputs.json")
    regeneration = _snapshot_command(
        output,
        [
            "adjudicate",
            "--input-path",
            str(staging / "adjudication_inputs.json"),
            "--output-path",
            str(staging / "package_terminal.json"),
            "--snapshot-root",
            str(output / "frozen_source_snapshot"),
        ],
        cwd=staging,
    )
    regenerated = staging / "package_terminal.json"
    regeneration_passed = (
        regeneration["exit_code"] == 0
        and regenerated.is_file()
        and regenerated.read_bytes() == terminal_path.read_bytes()
    )
    if not regeneration_passed:
        adjudication_inputs["gates"]["terminal_regeneration"] = False
        write_json(input_path, adjudication_inputs, overwrite=True)
        terminal = adjudicate_file(input_path, terminal_path)
    write_json(
        output / "terminal_regeneration.json",
        {
            "status": "PASS" if regeneration_passed else "FAIL",
            "separate_process": True,
            "frozen_adjudicator_used": True,
            "byte_identical": regeneration_passed,
            "primary_sha256": sha256_file(terminal_path),
            "regenerated_sha256": (
                sha256_file(regenerated) if regenerated.is_file() else None
            ),
            "process": regeneration,
        },
    )
    if terminal["decision"] == "DEVELOPMENT_PACKAGE_SEALED":
        authorization = authorization_payload(
            terminal_sha256=sha256_file(terminal_path),
            freeze_receipt_sha256=sha256_file(output / "freeze_receipt.json"),
            snapshot_manifest_sha256=sha256_file(
                output / "frozen_source_snapshot/snapshot_manifest.json"
            ),
            excluded_rehearsal_sha256=sha256_file(
                output / "excluded_rehearsal/cohort_summary.json"
            ),
            adversarial_audit_sha256=sha256_file(
                root
                / "output/synthetic_identifiability_adversarial_audit_v8/"
                "run_2.json"
            ),
            official_tests_sha256=sha256_file(
                output / "official_test_execution_report.json"
            ),
            exact_command=str(config["commands"]["development_one_shot"]),
            development_registry=registries["development"],
            confirmation_registry=registries["confirmation_reserve"],
        )
        write_json(output / "DEVELOPMENT_LAUNCH_READY.json", authorization)
    manifest = write_complete_manifest(output)
    if manifest["independent_second_pass_verification"]["status"] != "PASS":
        adjudication_inputs["gates"]["complete_manifest"] = False
        write_json(input_path, adjudication_inputs, overwrite=True)
        terminal = adjudicate_file(input_path, terminal_path)
        write_complete_manifest(output)
    return terminal
