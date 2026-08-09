from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import stat
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from threadpoolctl import threadpool_info

from .generator import generate_corpus
from .learner import (
    ablate_null_head,
    fit_corrective_mil,
    predict_without_side,
)
from .protocol import (
    AuthorizationCapability,
    ExcludedRehearsalCapability,
    IdentifierFirewall,
    IdentifierReference,
    atomic_rename_directory_noreplace,
    canonical_digest,
    canonical_bytes,
    _issue_worker_development_capability,
    _issue_excluded_rehearsal_capability,
    load_config,
    manifest_for_paths,
    manifest_for_tree,
    read_json,
    require_lstat_absent,
    sha256_file,
    tree_file_paths,
    verify_exact_manifest,
    verify_ledger,
    verify_development_capability,
    verify_excluded_rehearsal_capability,
    write_json,
)
from .statistics import (
    analyze_informativeness,
    analyze_primary,
    average_model_replicates,
    dependence_diagnostics,
    score_predictions,
)


_DEVELOPMENT_INTERNAL_SEAL = object()
_REHEARSAL_INTERNAL_SEAL = object()


def _require_frozen_project_module_origins(snapshot_root: str | Path) -> None:
    snapshot = Path(snapshot_root).resolve()
    expected = {
        "babyworld_lite": "babyworld_lite/__init__.py",
        "babyworld_lite.corrective_alignment_v1": "babyworld_lite/corrective_alignment_v1/__init__.py",
        "babyworld_lite.corrective_alignment_v1.protocol": "babyworld_lite/corrective_alignment_v1/protocol.py",
        "babyworld_lite.corrective_alignment_v1.generator": "babyworld_lite/corrective_alignment_v1/generator.py",
        "babyworld_lite.corrective_alignment_v1.learner": "babyworld_lite/corrective_alignment_v1/learner.py",
        "babyworld_lite.corrective_alignment_v1.statistics": "babyworld_lite/corrective_alignment_v1/statistics.py",
        "babyworld_lite.corrective_alignment_v1.parallel": "babyworld_lite/corrective_alignment_v1/parallel.py",
        "babyworld_lite.corrective_alignment_v1.adjudicator": "babyworld_lite/corrective_alignment_v1/adjudicator.py",
        "babyworld_lite.corrective_alignment_v1.integrity": "babyworld_lite/corrective_alignment_v1/integrity.py",
    }
    mismatches = {}
    for name, relative in expected.items():
        module = sys.modules.get(name)
        observed = Path(str(getattr(module, "__file__", ""))).resolve()
        required = (snapshot / relative).resolve()
        if module is None or observed != required:
            mismatches[name] = {"observed": str(observed), "required": str(required)}
    if mismatches:
        raise PermissionError(f"frozen project module origin mismatch: {mismatches}")


def _verify_frozen_execution_boundary(
    snapshot_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = Path(snapshot_root).resolve()
    if (
        snapshot.name != "frozen_source_snapshot"
        or snapshot.parent.name
        != "synthetic_corrective_development_launch_package_v1"
        or snapshot.parent.parent.name != "output"
    ):
        raise PermissionError("frozen execution snapshot layout mismatch")
    from .integrity import environment_record, verify_snapshot

    _require_frozen_project_module_origins(snapshot)

    verification = verify_snapshot(snapshot)
    if verification.get("status") != "PASS":
        raise PermissionError(f"frozen execution snapshot failed: {verification}")
    frozen_config_path = snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
    loaded = load_config(frozen_config_path, repository_root=snapshot)
    if (
        loaded["protocol"]["status"] != "frozen"
        or canonical_digest(loaded) != canonical_digest(config)
    ):
        raise PermissionError("in-memory config differs from verified frozen config")
    _require_thread_environment(loaded)
    frozen_environment = read_json(snapshot.parent / "frozen_environment.json")
    if environment_record(snapshot.parents[2], loaded) != frozen_environment:
        raise PermissionError("frozen execution environment lock mismatch")
    return verification


def _verified_prequalification_lock(
    repository_root: str | Path,
    config: Mapping[str, Any],
) -> tuple[Path, str]:
    repository = Path(repository_root).resolve()
    lock_path = (
        repository.parent / "prequalification_design_lock.json"
        if repository.name == "frozen_source_snapshot"
        else repository
        / str(config["paths"]["package_root"])
        / "prequalification_design_lock.json"
    )
    metadata = lock_path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError("prequalification design lock must be a regular file")
    lock = read_json(lock_path)
    if (
        lock.get("schema_version")
        != "nursery-corrective-prequalification-design-lock-v1"
        or lock.get("status") != "LOCKED_BEFORE_OFFICIAL_FIXTURES"
        or int(lock.get("development_outcome_count", -1)) != 0
        or int(lock.get("confirmation_outcome_count", -1)) != 0
    ):
        raise PermissionError("prequalification design lock content is invalid")
    return lock_path, sha256_file(lock_path)


def _verify_rehearsal_capability(
    capability: ExcludedRehearsalCapability | None,
    *,
    config: Mapping[str, Any],
    jobs: int | None = None,
    output_root: str | Path | None = None,
    persisted_input_root: str | Path | None = None,
    parallel_output_root: str | Path | None = None,
    snapshot_root: str | Path | None = None,
) -> None:
    if capability is None:
        raise PermissionError("excluded rehearsal requires a consumed attempt capability")
    verify_excluded_rehearsal_capability(capability)
    receipt_path = Path(capability.attempt_receipt_path)
    snapshot = Path(capability.snapshot_root)
    configured_receipt = (
        snapshot.parents[2]
        / str(config["paths"]["excluded_rehearsal_attempt_receipt"])
    ).resolve()
    if receipt_path.resolve() != configured_receipt:
        raise PermissionError("excluded rehearsal receipt path mismatch")
    if capability.required_jobs != int(config["parallel"]["frozen_jobs"]):
        raise PermissionError("excluded rehearsal capability changed frozen job count")
    if jobs is not None and int(jobs) != capability.required_jobs:
        raise PermissionError("excluded rehearsal job count mismatch")
    if output_root is not None and Path(output_root).resolve() != Path(
        capability.output_root
    ):
        raise PermissionError("excluded rehearsal output path mismatch")
    if snapshot_root is not None and Path(snapshot_root).resolve() != snapshot:
        raise PermissionError("excluded rehearsal snapshot path mismatch")
    verify_excluded_rehearsal_capability(
        capability,
        persisted_input_root=persisted_input_root,
        parallel_output_root=parallel_output_root,
    )


def _require_thread_environment(config: Mapping[str, Any]) -> None:
    mismatches = {
        str(name): {"expected": str(value), "observed": os.environ.get(str(name))}
        for name, value in config["parallel"]["thread_environment"].items()
        if os.environ.get(str(name)) != str(value)
    }
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        mismatches["PYTHONDONTWRITEBYTECODE"] = {
            "expected": "1",
            "observed": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        }
    if mismatches:
        raise RuntimeError(f"thread/startup environment mismatch: {mismatches}")


def _terminate_process_pool(executor: ProcessPoolExecutor) -> None:
    for process in list(getattr(executor, "_processes", {}).values()):
        if process.is_alive():
            process.terminate()
    for process in list(getattr(executor, "_processes", {}).values()):
        process.join(timeout=5.0)
    executor.shutdown(wait=True, cancel_futures=True)


def _execute_worker_batches(
    *,
    plan: Mapping[str, Any],
    arguments_by_id: Mapping[str, Mapping[str, Any]],
    jobs: int,
    context: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run model waves and terminate the whole pool on any BaseException."""
    telemetry: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    executor = ProcessPoolExecutor(max_workers=int(jobs), mp_context=context)
    pool_terminated = False
    try:
        for model_seed in plan["model_seed_order"]:
            if failures:
                break
            batch = [
                unit
                for unit in plan["units"]
                if int(unit["model_seed"]) == int(model_seed)
            ]
            futures = {
                executor.submit(
                    _worker, arguments_by_id[str(unit["unit_id"])]
                ): str(unit["unit_id"])
                for unit in batch
            }
            for future in as_completed(futures):
                unit_id = futures[future]
                try:
                    telemetry.append(future.result())
                except BaseException as error:
                    failures.append(
                        {
                            "unit_id": unit_id,
                            "exception": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    for pending in futures:
                        pending.cancel()
                    _terminate_process_pool(executor)
                    pool_terminated = True
                    break
    finally:
        if not pool_terminated:
            executor.shutdown(wait=True, cancel_futures=True)
    return telemetry, failures


def _validate_shard_inventory(
    shards: str | Path,
    expected_unit_ids: Sequence[str],
) -> list[str]:
    root = Path(shards)
    metadata = root.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("shard root must be a real directory")
    expected = list(map(str, expected_unit_ids))
    if len(expected) != len(set(expected)):
        raise RuntimeError("duplicate expected work-unit identifiers")
    actual: list[str] = []
    invalid_entries = []
    for path in root.iterdir():
        child_metadata = path.lstat()
        if not stat.S_ISDIR(child_metadata.st_mode) or stat.S_ISLNK(
            child_metadata.st_mode
        ):
            invalid_entries.append(path.name)
        else:
            actual.append(path.name)
    if invalid_entries:
        raise RuntimeError(
            f"non-directory or symlink shard entries: {sorted(invalid_entries)}"
        )
    actual_set = set(actual)
    expected_set = set(expected)
    if len(actual) != len(actual_set):
        raise RuntimeError("duplicate observed work-unit identifiers")
    if actual_set != expected_set:
        raise RuntimeError(
            f"work-unit completeness mismatch: missing={sorted(expected_set-actual_set)} "
            f"extra={sorted(actual_set-expected_set)}"
        )
    return sorted(actual)


def _config_contract(config: Mapping[str, Any], *, purpose: str) -> str:
    return canonical_digest(
        {
            "protocol": config["protocol"],
            "registry": config["resolved_registries"][purpose],
            "design": config["design"],
            "learner": config["learner"],
            "analysis": config["analysis"],
            "qualification_gates": config["qualification_gates"],
            "parallel": config["parallel"],
        }
    )


def _unit_id(corpus_seed: int, model_seed: int) -> str:
    return f"corpus-{int(corpus_seed)}__model-{int(model_seed)}"


def _expected_operations(config: Mapping[str, Any]) -> list[str]:
    output: list[str] = []
    for _condition in config["design"]["conditions"]:
        output.extend(["fit", "predict"])
    for mutation in config["design"]["mechanism_mutations"]:
        if mutation == "null_head_ablation":
            output.extend(["control", "predict"])
        else:
            output.extend(["control", "fit", "predict"])
    return output


def _verify_global_ledger(
    ledger: Mapping[str, Any],
    expected: Sequence[Mapping[str, Any]],
    *,
    purpose: str,
    authorization_digest: str | None,
    parsed_contract_digest: str | None,
    operation_contract_digest: str | None,
) -> dict[str, Any]:
    problems: list[str] = []
    expected_top = {
        "schema_version",
        "purpose",
        "authorization_digest",
        "parsed_contract_digest",
        "operation_contract_digest",
        "entry_count",
        "terminal_digest",
        "entries",
    }
    if set(ledger) != expected_top:
        problems.append("top_level_schema")
    if ledger.get("schema_version") != "nursery-operation-ledger-v1":
        problems.append("schema_version")
    if ledger.get("purpose") != purpose:
        problems.append("purpose")
    if ledger.get("authorization_digest") != authorization_digest:
        problems.append("authorization_digest")
    if ledger.get("parsed_contract_digest") != parsed_contract_digest:
        problems.append("parsed_contract_digest")
    if ledger.get("operation_contract_digest") != operation_contract_digest:
        problems.append("operation_contract_digest")
    entries = ledger.get("entries", [])
    if not isinstance(entries, list) or len(entries) != len(expected):
        problems.append("entry_count")
        entries = entries if isinstance(entries, list) else []
    if int(ledger.get("entry_count", -1)) != len(entries):
        problems.append("declared_entry_count")
    previous = "0" * 64
    for index, (entry, wanted) in enumerate(zip(entries, expected)):
        if set(entry) != {
            "purpose",
            "unit_id",
            "local_sequence",
            "operation",
            "references",
            "previous_digest",
            "entry_digest",
        }:
            problems.append(f"entry_schema:{index}")
            continue
        payload = {key: value for key, value in entry.items() if key != "entry_digest"}
        if entry.get("purpose") != purpose:
            problems.append(f"entry_purpose:{index}")
        if entry.get("unit_id") != wanted["unit_id"]:
            problems.append(f"unit:{index}")
        if int(entry.get("local_sequence", -1)) != index:
            problems.append(f"sequence:{index}")
        if entry.get("operation") != wanted["operation"]:
            problems.append(f"operation:{index}")
        if entry.get("references") != wanted["references"]:
            problems.append(f"references:{index}")
        if entry.get("previous_digest") != previous:
            problems.append(f"chain:{index}")
        if entry.get("entry_digest") != canonical_digest(payload):
            problems.append(f"digest:{index}")
        previous = str(entry.get("entry_digest", ""))
    if ledger.get("terminal_digest") != previous:
        problems.append("terminal_digest")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}


def prepare_persisted_inputs(
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    purpose: str,
    capability: AuthorizationCapability | None = None,
    _development_seal: object | None = None,
    _rehearsal_seal: object | None = None,
    _rehearsal_capability: ExcludedRehearsalCapability | None = None,
) -> dict[str, Any]:
    if purpose == "development":
        if _development_seal is not _DEVELOPMENT_INTERNAL_SEAL:
            raise PermissionError(
                "development input generation is unavailable through the generic API"
            )
        if capability is None:
            raise PermissionError("development input generation requires a capability")
        verify_development_capability(capability, required_scope="parent")
        snapshot = Path(
            str(read_json(capability.claim_path)["resolved_snapshot_root"])
        ).resolve()
        _verify_frozen_execution_boundary(snapshot, config)
        expected_input = (
            Path(capability.resolved_inner_staging_root) / "persisted_inputs"
        ).resolve()
        if Path(output_root).resolve() != expected_input:
            raise PermissionError("development persisted-input path mismatch")
    elif capability is not None:
        raise PermissionError("capability cannot be used for non-development generation")
    if purpose == "excluded_rehearsal":
        if _rehearsal_seal is not _REHEARSAL_INTERNAL_SEAL:
            raise PermissionError("excluded rehearsal generation requires the one-shot runner")
        _verify_rehearsal_capability(
            _rehearsal_capability,
            config=config,
            persisted_input_root=output_root,
        )
        _verify_frozen_execution_boundary(
            _rehearsal_capability.snapshot_root,
            config,
        )
    root = Path(output_root).resolve()
    require_lstat_absent(root)
    root.mkdir(parents=True)
    firewall = IdentifierFirewall(
        config,
        purpose=purpose,
        capability=(
            capability
            if purpose == "development"
            else _rehearsal_capability
            if purpose == "excluded_rehearsal"
            else None
        ),
        operation_contract_digest=_config_contract(config, purpose=purpose),
    )
    audits = []
    for corpus_seed in config["resolved_registries"][purpose]["corpus"]:
        corpus = generate_corpus(int(corpus_seed), config, firewall)
        directory = root / f"corpus-{int(corpus_seed)}"
        directory.mkdir()
        write_json(directory / "learner_input.json", corpus["learner_input"])
        write_json(
            directory / "protected_adjudication.json",
            corpus["protected_adjudication"],
        )
        write_json(directory / "corpus_audit.json", corpus["audit"])
        corpus_manifest = manifest_for_paths(
            directory,
            ["learner_input.json", "protected_adjudication.json", "corpus_audit.json"],
        )
        write_json(directory / "manifest.json", corpus_manifest)
        audits.append(corpus["audit"])
    generation_ledger = firewall.ledger()
    expected_generation = [
        {
            "unit_id": f"corpus-{int(corpus_seed)}",
            "operation": operation,
            "references": [{"role": "corpus", "value": int(corpus_seed)}],
        }
        for corpus_seed in config["resolved_registries"][purpose]["corpus"]
        for operation in ("generate", "condition")
    ]
    generation_verification = _verify_global_ledger(
        generation_ledger,
        expected_generation,
        purpose=purpose,
        authorization_digest=(capability.authorization_digest if capability else None),
        parsed_contract_digest=(capability.parsed_contract_digest if capability else None),
        operation_contract_digest=_config_contract(config, purpose=purpose),
    )
    if generation_verification["status"] != "PASS":
        raise RuntimeError(f"generation ledger failed: {generation_verification}")
    write_json(root / "generation_operation_ledger.json", generation_ledger)
    write_json(root / "generation_ledger_verification.json", generation_verification)
    write_json(root / "corpus_audits.json", {"rows": audits})
    manifest = manifest_for_tree(root, exclude=["input_manifest.json"])
    write_json(root / "input_manifest.json", manifest)
    verification = verify_exact_manifest(
        root,
        manifest,
        manifest_filename="input_manifest.json",
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"persisted input manifest failed: {verification}")
    return {
        "status": "PASS",
        "corpus_count": len(audits),
        "input_manifest_sha256": sha256_file(root / "input_manifest.json"),
        "input_manifest_digest": manifest["digest"],
        "operation_count": len(firewall.operations),
    }


def verify_persisted_inputs(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    manifest = read_json(base / "input_manifest.json")
    return verify_exact_manifest(
        base,
        manifest,
        manifest_filename="input_manifest.json",
    )


def build_work_plan(config: Mapping[str, Any], *, purpose: str) -> dict[str, Any]:
    corpus_seeds = list(map(int, config["resolved_registries"][purpose]["corpus"]))
    model_seeds = list(map(int, config["resolved_registries"][purpose]["model"]))
    units = [
        {
            "unit_id": _unit_id(corpus_seed, model_seed),
            "corpus_seed": corpus_seed,
            "model_seed": model_seed,
            "condition_order": list(config["design"]["conditions"]),
            "mutation_order": list(config["design"]["mechanism_mutations"]),
        }
        for corpus_seed in corpus_seeds
        for model_seed in model_seeds
    ]
    return {
        "schema_version": "nursery-corrective-work-plan-v1",
        "purpose": purpose,
        "bundle_unit": "corpus_seed_x_model_seed_all_conditions",
        "canonical_order": "corpus_then_model",
        "corpus_seed_order": corpus_seeds,
        "model_seed_order": model_seeds,
        "unit_count": len(units),
        "units": units,
        "digest": canonical_digest(units),
    }


def _read_input_manifest_committed_json(
    input_root: Path,
    input_manifest: Mapping[str, Any],
    relative: str,
) -> Any:
    rows = {
        str(row["path"]): row for row in input_manifest.get("files", [])
    }
    row = rows.get(str(relative))
    if row is None:
        raise RuntimeError(f"persisted input is not committed by root manifest: {relative}")
    path = input_root / relative
    payload = path.read_bytes()
    if (
        len(payload) != int(row.get("bytes", -1))
        or hashlib.sha256(payload).hexdigest() != str(row.get("sha256"))
    ):
        raise RuntimeError(f"persisted input changed during committed read: {relative}")
    return json.loads(payload)


def _worker(
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    for name, value in arguments["thread_environment"].items():
        os.environ[str(name)] = str(value)
    purpose = str(arguments["purpose"])
    if purpose == "development" and any(
        not str(arguments.get(field, ""))
        for field in (
            "claim_path",
            "claim_sha256",
            "issuance_receipt_path",
            "issuance_receipt_sha256",
        )
    ):
        raise PermissionError("development worker capability envelope is missing")
    if purpose == "excluded_rehearsal" and any(
        not str(arguments.get(field, ""))
        for field in (
            "rehearsal_snapshot_root",
            "rehearsal_output_root",
            "rehearsal_parallel_output_root",
            "rehearsal_receipt_path",
            "rehearsal_receipt_sha256",
        )
    ):
        raise PermissionError("excluded rehearsal worker receipt envelope is missing")
    input_root = Path(arguments["input_root"]).resolve()
    shard_base = Path(arguments["shard_base"]).resolve()
    unit = dict(arguments["unit"])
    unit_id = str(unit["unit_id"])
    config = load_config(
        arguments["config_path"],
        repository_root=arguments["repository_root"],
    )
    _require_thread_environment(config)
    plan = build_work_plan(config, purpose=purpose)
    planned_units = {str(row["unit_id"]): row for row in plan["units"]}
    if unit_id not in planned_units or unit != planned_units[unit_id]:
        raise PermissionError("worker unit/order differs from frozen work plan")
    input_verification = verify_persisted_inputs(input_root)
    input_manifest_sha256 = sha256_file(input_root / "input_manifest.json")
    if (
        input_verification.get("status") != "PASS"
        or input_manifest_sha256 != str(arguments["input_manifest_sha256"])
    ):
        raise PermissionError("worker persisted-input commitment mismatch")
    input_manifest = read_json(input_root / "input_manifest.json")
    if purpose in {"development", "excluded_rehearsal"}:
        snapshot = Path(str(arguments["repository_root"])).resolve()
        _verify_frozen_execution_boundary(snapshot, config)
        if (
            config["protocol"]["status"] != "frozen"
            or Path(str(arguments["config_path"])).resolve()
            != (snapshot / "configs/synthetic_corrective_alignment_v1.yaml").resolve()
        ):
            raise PermissionError("worker is not executing the frozen guarded source")
    firewall_capability = None
    if purpose == "development":
        development_capability = _issue_worker_development_capability(
            authorization_digest=str(arguments["authorization_digest"]),
            parsed_contract_digest=str(arguments["parsed_contract_digest"]),
            claim_path=str(arguments["claim_path"]),
            claim_sha256=str(arguments["claim_sha256"]),
            issuance_receipt_path=str(arguments["issuance_receipt_path"]),
            issuance_receipt_sha256=str(arguments["issuance_receipt_sha256"]),
            resolved_output_root=str(arguments["development_output_root"]),
            resolved_staging_root=str(arguments["development_staging_root"]),
            resolved_inner_staging_root=str(
                arguments["development_inner_staging_root"]
            ),
            required_jobs=int(arguments["required_jobs"]),
            unit_id=unit_id,
        )
        verify_development_capability(
            development_capability,
            required_scope="worker",
            required_unit_id=unit_id,
        )
        expected_input = (
            Path(development_capability.resolved_inner_staging_root)
            / "persisted_inputs"
        ).resolve()
        expected_shards = (
            Path(development_capability.resolved_inner_staging_root)
            / "parallel_compute/shards"
        ).resolve()
        if input_root != expected_input or shard_base != expected_shards:
            raise PermissionError("development worker path envelope mismatch")
        firewall_capability = development_capability
    elif purpose == "excluded_rehearsal":
        rehearsal_capability = _issue_excluded_rehearsal_capability(
            config=config,
            scope="worker",
            unit_id=unit_id,
            snapshot_root=str(arguments["rehearsal_snapshot_root"]),
            output_root=str(arguments["rehearsal_output_root"]),
            persisted_input_root=input_root,
            parallel_output_root=str(arguments["rehearsal_parallel_output_root"]),
            attempt_receipt_path=str(arguments["rehearsal_receipt_path"]),
            attempt_receipt_sha256=str(arguments["rehearsal_receipt_sha256"]),
            required_jobs=int(arguments["required_jobs"]),
            input_manifest_sha256=input_manifest_sha256,
            scientific_contract_digest=str(
                arguments["scientific_contract_digest"]
            ),
        )
        _verify_rehearsal_capability(
            rehearsal_capability,
            config=config,
            jobs=int(arguments["required_jobs"]),
            output_root=str(arguments["rehearsal_output_root"]),
            persisted_input_root=input_root,
            parallel_output_root=shard_base.parent,
            snapshot_root=str(arguments["rehearsal_snapshot_root"]),
        )
        verify_excluded_rehearsal_capability(
            rehearsal_capability,
            required_scope="worker",
            required_unit_id=unit_id,
            input_manifest_sha256=input_manifest_sha256,
            scientific_contract_digest=str(arguments["scientific_contract_digest"]),
        )
        if shard_base != Path(rehearsal_capability.parallel_output_root) / "shards":
            raise PermissionError("excluded rehearsal worker shard path mismatch")
        firewall_capability = rehearsal_capability
    firewall = IdentifierFirewall(
        config,
        purpose=purpose,
        capability=firewall_capability,
        operation_contract_digest=str(arguments["scientific_contract_digest"]),
    )
    temporary = shard_base / f".{unit_id}.tmp"
    complete = shard_base / unit_id
    require_lstat_absent(temporary)
    require_lstat_absent(complete)
    temporary.mkdir()
    temporary_tmp = temporary / "tmp"
    temporary_tmp.mkdir()
    os.environ["TMPDIR"] = str(temporary_tmp)
    tempfile.tempdir = str(temporary_tmp)
    learner_relative = (
        f"corpus-{int(unit['corpus_seed'])}/learner_input.json"
    )
    learner_input = _read_input_manifest_committed_json(
        input_root,
        input_manifest,
        learner_relative,
    )
    episodes = learner_input["training_episodes"]
    prompts = learner_input["evaluation_prompts"]
    condition_results = []
    active_model = None
    for condition in unit["condition_order"]:
        references = [
            IdentifierReference("corpus", int(unit["corpus_seed"])),
            IdentifierReference("model", int(unit["model_seed"])),
        ]
        firewall.authorize(
            "fit",
            references,
            unit_id=unit_id,
            local_sequence=len(firewall.operations),
        )
        model, trace = fit_corrective_mil(
            episodes,
            learner_input["condition_evidence"][condition],
            config,
            model_seed=int(unit["model_seed"]),
            condition=str(condition),
        )
        firewall.authorize(
            "predict",
            references,
            unit_id=unit_id,
            local_sequence=len(firewall.operations),
        )
        predictions = predict_without_side(model, prompts, config)
        condition_results.append(
            {
                "condition": condition,
                "model": model.serializable(),
                "model_digest": canonical_digest(model.serializable()),
                "trace": trace,
                "predictions": predictions,
                "prediction_digest": canonical_digest(predictions),
            }
        )
        if condition == "synchronized":
            active_model = model
    if active_model is None:
        raise RuntimeError("synchronized condition missing")
    mutation_results = []
    for mutation in unit["mutation_order"]:
        references = [
            IdentifierReference("corpus", int(unit["corpus_seed"])),
            IdentifierReference("model", int(unit["model_seed"])),
        ]
        firewall.authorize(
            "control",
            references,
            unit_id=unit_id,
            local_sequence=len(firewall.operations),
        )
        if mutation == "null_head_ablation":
            model = ablate_null_head(active_model)
            trace = {
                "mutation": mutation,
                "semantic_parameters_unchanged": True,
                "null_rates_forced_to": 1e-6,
            }
        else:
            firewall.authorize(
                "fit",
                references,
                unit_id=unit_id,
                local_sequence=len(firewall.operations),
            )
            model, trace = fit_corrective_mil(
                episodes,
                learner_input["condition_evidence"]["synchronized"],
                config,
                model_seed=int(unit["model_seed"]),
                condition="synchronized",
                mutation=str(mutation),
            )
        firewall.authorize(
            "predict",
            references,
            unit_id=unit_id,
            local_sequence=len(firewall.operations),
        )
        predictions = predict_without_side(model, prompts, config)
        mutation_results.append(
            {
                "mutation": mutation,
                "model": model.serializable(),
                "model_digest": canonical_digest(model.serializable()),
                "trace": trace,
                "predictions": predictions,
                "prediction_digest": canonical_digest(predictions),
            }
        )
    result = {
        "schema_version": "nursery-corrective-worker-result-v1",
        "purpose": arguments["purpose"],
        "unit_id": unit_id,
        "corpus_seed": int(unit["corpus_seed"]),
        "model_seed": int(unit["model_seed"]),
        "input_manifest_sha256": str(arguments["input_manifest_sha256"]),
        "condition_results": condition_results,
        "mutation_results": mutation_results,
        "scientific_contract_digest": str(arguments["scientific_contract_digest"]),
        "publication_state": "UNPUBLISHED_CANDIDATE",
        "scientific_outcome": False,
        "development_outcome_count": 0,
        "confirmation_outcome": False,
    }
    write_json(temporary / "result.json", result)
    write_json(temporary / "operation_ledger.json", firewall.ledger())
    manifest = manifest_for_paths(
        temporary, ["result.json", "operation_ledger.json"]
    )
    write_json(temporary / "manifest.json", manifest)
    write_json(
        temporary / "COMPLETE.json",
        {
            "schema_version": "nursery-corrective-shard-complete-v1",
            "unit_id": unit_id,
            "result_sha256": sha256_file(temporary / "result.json"),
            "ledger_sha256": sha256_file(temporary / "operation_ledger.json"),
            "manifest_sha256": sha256_file(temporary / "manifest.json"),
            "status": "COMPLETE",
        },
    )
    temporary_tmp.rmdir()
    atomic_rename_directory_noreplace(temporary, complete)
    elapsed = time.perf_counter() - started
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    native_threadpools = [
        {
            "internal_api": str(row.get("internal_api")),
            "prefix": str(row.get("prefix")),
            "num_threads": int(row.get("num_threads", -1)),
        }
        for row in threadpool_info()
    ]
    return {
        "unit_id": unit_id,
        "shard_manifest_sha256": sha256_file(complete / "manifest.json"),
        "wall_seconds": elapsed,
        "worker_peak_rss_native_units": peak_rss,
        "thread_environment_observed": {
            str(name): os.environ.get(str(name))
            for name in arguments["thread_environment"]
        },
        "native_threadpools": native_threadpools,
    }


def _validate_shard(
    shard: Path,
    unit: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    purpose: str,
    input_manifest_sha256: str,
    scientific_contract_digest: str,
    authorization_digest: str | None,
    parsed_contract_digest: str | None,
) -> dict[str, Any]:
    actual = set(tree_file_paths(shard))
    expected = {"result.json", "operation_ledger.json", "manifest.json", "COMPLETE.json"}
    if actual != expected:
        raise RuntimeError(f"shard exact file set mismatch: {shard}: {actual}")
    complete = read_json(shard / "COMPLETE.json")
    if complete != {
        "schema_version": "nursery-corrective-shard-complete-v1",
        "unit_id": unit["unit_id"],
        "result_sha256": sha256_file(shard / "result.json"),
        "ledger_sha256": sha256_file(shard / "operation_ledger.json"),
        "manifest_sha256": sha256_file(shard / "manifest.json"),
        "status": "COMPLETE",
    }:
        raise RuntimeError(f"shard completion seal mismatch: {shard}")
    manifest = read_json(shard / "manifest.json")
    verification = verify_exact_manifest(
        shard,
        manifest,
        manifest_filename="manifest.json",
        allowed_extra_files=["COMPLETE.json"],
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"shard manifest failed: {verification}")
    result = read_json(shard / "result.json")
    expected_result_fields = {
        "schema_version",
        "purpose",
        "unit_id",
        "corpus_seed",
        "model_seed",
        "input_manifest_sha256",
        "condition_results",
        "mutation_results",
        "scientific_contract_digest",
        "publication_state",
        "scientific_outcome",
        "development_outcome_count",
        "confirmation_outcome",
    }
    if (
        set(result) != expected_result_fields
        or result.get("schema_version") != "nursery-corrective-worker-result-v1"
        or result.get("unit_id") != unit["unit_id"]
        or int(result.get("corpus_seed", -1)) != int(unit["corpus_seed"])
        or int(result.get("model_seed", -1)) != int(unit["model_seed"])
        or result.get("purpose") != purpose
        or result.get("input_manifest_sha256") != input_manifest_sha256
        or result.get("scientific_contract_digest") != scientific_contract_digest
        or result.get("scientific_outcome") is not False
        or int(result.get("development_outcome_count", -1)) != 0
        or result.get("confirmation_outcome") is not False
        or result.get("publication_state") != "UNPUBLISHED_CANDIDATE"
    ):
        raise RuntimeError(f"shard unit/input metadata mismatch: {shard}")
    if [row["condition"] for row in result["condition_results"]] != list(
        unit["condition_order"]
    ):
        raise RuntimeError("condition result order mismatch")
    if [row["mutation"] for row in result["mutation_results"]] != list(
        unit["mutation_order"]
    ):
        raise RuntimeError("mutation result order mismatch")
    seen_prompt_sets = []
    for kind, rows, label_field in (
        ("condition", result["condition_results"], "condition"),
        ("mutation", result["mutation_results"], "mutation"),
    ):
        expected_fields = {
            label_field,
            "model",
            "model_digest",
            "trace",
            "predictions",
            "prediction_digest",
        }
        for index, row in enumerate(rows):
            if set(row) != expected_fields:
                raise RuntimeError(f"{kind} result schema mismatch: {index}")
            model = row["model"]
            if (
                not isinstance(model, Mapping)
                or model.get("schema_version")
                != "nursery-corrective-lexicon-model-v1"
                or canonical_digest(model) != row["model_digest"]
                or int(model.get("model_seed", -1)) != int(unit["model_seed"])
                or model.get("training_side_state_serialized") is not False
                or model.get("detector_serialized") is not False
                or model.get("oracle_serialized") is not False
                or model.get("evaluation_key_serialized") is not False
            ):
                raise RuntimeError(f"{kind} embedded model/digest mismatch: {index}")
            predictions = row["predictions"]
            if not isinstance(predictions, list) or canonical_digest(predictions) != row[
                "prediction_digest"
            ]:
                raise RuntimeError(f"{kind} prediction digest mismatch: {index}")
            prompt_ids = []
            for prediction in predictions:
                if set(prediction) != {
                    "prompt_id",
                    "kind",
                    "scores",
                    "probabilities",
                    "prediction_side_fields_consumed",
                } or prediction.get("prediction_side_fields_consumed") is not False:
                    raise RuntimeError(f"{kind} prediction schema mismatch: {index}")
                scores = list(prediction["scores"])
                probabilities = list(prediction["probabilities"])
                if (
                    len(scores) != len(probabilities)
                    or not scores
                    or not all(math.isfinite(float(value)) for value in [*scores, *probabilities])
                    or not all(0.0 <= float(value) <= 1.0 for value in probabilities)
                    or not math.isclose(
                        math.fsum(map(float, probabilities)), 1.0, abs_tol=1e-12
                    )
                ):
                    raise RuntimeError(f"{kind} prediction numeric mismatch: {index}")
                prompt_ids.append(str(prediction["prompt_id"]))
            if len(prompt_ids) != len(set(prompt_ids)):
                raise RuntimeError(f"{kind} duplicate prompt ids: {index}")
            seen_prompt_sets.append(set(prompt_ids))
    if seen_prompt_sets and any(values != seen_prompt_sets[0] for values in seen_prompt_sets):
        raise RuntimeError("condition/mutation prediction prompt sets differ")
    ledger_verification = verify_ledger(
        read_json(shard / "operation_ledger.json"),
        purpose=purpose,
        unit_id=str(unit["unit_id"]),
        expected_operations=_expected_operations(config),
        corpus_seed=int(unit["corpus_seed"]),
        model_seed=int(unit["model_seed"]),
        authorization_digest=authorization_digest,
        parsed_contract_digest=parsed_contract_digest,
        operation_contract_digest=scientific_contract_digest,
    )
    if ledger_verification["status"] != "PASS":
        raise RuntimeError(f"shard ledger failed: {ledger_verification}")
    return result


def _mutation_summary(
    scored_conditions: Sequence[Mapping[str, Any]],
    scored_mutations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    active = {
        (int(row["corpus_seed"]), int(row["model_seed"])): row["metrics"]
        for row in scored_conditions
        if row["condition"] == "synchronized"
    }
    by_mutation: dict[str, list[Mapping[str, Any]]] = {}
    for row in scored_mutations:
        by_mutation.setdefault(str(row["mutation"]), []).append(row)
    output = {}
    for mutation, rows in sorted(by_mutation.items()):
        lexical_changes = []
        composition_changes = []
        present_changes = []
        null_changes = []
        corpus_values: dict[int, dict[str, list[float]]] = {}
        for row in sorted(
            rows, key=lambda value: (int(value["corpus_seed"]), int(value["model_seed"]))
        ):
            key = (int(row["corpus_seed"]), int(row["model_seed"]))
            base = active[key]
            metrics = row["metrics"]
            lexical_changes.append(
                float(metrics["lexical_acquisition_top1"])
                - float(base["lexical_acquisition_top1"])
            )
            composition_changes.append(
                float(metrics["heldout_composition_top1"])
                - float(base["heldout_composition_top1"])
            )
            present_changes.append(
                float(metrics["by_presence"]["present"]["strict_top1"])
                - float(base["by_presence"]["present"]["strict_top1"])
            )
            null_changes.append(
                float(metrics["by_presence"]["null"]["strict_top1"])
                - float(base["by_presence"]["null"]["strict_top1"])
            )
            corpus = int(row["corpus_seed"])
            values = corpus_values.setdefault(
                corpus,
                {
                    "lexical_change": [],
                    "composition_change": [],
                    "present_change": [],
                    "null_change": [],
                    "lexical_absolute": [],
                    "composition_absolute": [],
                    "present_absolute": [],
                    "null_absolute": [],
                },
            )
            values["lexical_change"].append(lexical_changes[-1])
            values["composition_change"].append(composition_changes[-1])
            values["present_change"].append(present_changes[-1])
            values["null_change"].append(null_changes[-1])
            values["lexical_absolute"].append(
                float(metrics["lexical_acquisition_top1"])
            )
            values["composition_absolute"].append(
                float(metrics["heldout_composition_top1"])
            )
            values["present_absolute"].append(
                float(metrics["by_presence"]["present"]["strict_top1"])
            )
            values["null_absolute"].append(
                float(metrics["by_presence"]["null"]["strict_top1"])
            )
        corpus_rows = [
            {
                "corpus_seed": corpus,
                **{
                    name: float(math.fsum(values[name]) / len(values[name]))
                    for name in sorted(values)
                },
            }
            for corpus, values in sorted(corpus_values.items())
        ]
        output[mutation] = {
            "lexical_change": float(sum(lexical_changes) / len(lexical_changes)),
            "composition_change": float(
                sum(composition_changes) / len(composition_changes)
            ),
            "present_change": float(sum(present_changes) / len(present_changes)),
            "null_change": float(sum(null_changes) / len(null_changes)),
            "unit_count": len(rows),
            "lexical_changes": lexical_changes,
            "composition_changes": composition_changes,
            "present_changes": present_changes,
            "null_changes": null_changes,
            "corpus_rows": corpus_rows,
            "corpus_count": len(corpus_rows),
            "model_replicates_averaged_within_corpus_first": True,
        }
    return {
        "schema_version": "nursery-corrective-mutation-summary-v1",
        "mutations": output,
        "separate_semantic_and_null_heads": True,
    }


def _candidate_scientific_decision(
    *,
    purpose: str,
    primary_status: str,
    informativeness_status: str,
    dependence_status: str,
    causal_attribution_status: str,
) -> str:
    if purpose != "development":
        return "SCIENTIFIC_INFERENCE_SUPPRESSED"
    if informativeness_status != "PASS" or dependence_status != "PASS":
        return "REVISE_UNINFORMATIVE"
    if primary_status == "PASS" and causal_attribution_status == "PASS":
        return "GO"
    if primary_status == "PASS":
        return "REVISE_CAUSAL_ATTRIBUTION_FAILED"
    return "CORRECTIVE_STUDY_STOP_NO_SUPPORT"


def run_parallel_from_persisted(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    purpose: str,
    jobs: int,
    expected_input_manifest_sha256: str,
    capability: AuthorizationCapability | None = None,
    _development_seal: object | None = None,
    _rehearsal_seal: object | None = None,
    _rehearsal_capability: ExcludedRehearsalCapability | None = None,
) -> dict[str, Any]:
    if int(jobs) < 1:
        raise ValueError("jobs must be positive")
    if purpose == "development":
        if _development_seal is not _DEVELOPMENT_INTERNAL_SEAL:
            raise PermissionError("development is unavailable through the generic parallel API")
        if capability is None:
            raise PermissionError("development requires an issued capability")
        verify_development_capability(
            capability, required_jobs=jobs, required_scope="parent"
        )
        snapshot = Path(
            str(read_json(capability.claim_path)["resolved_snapshot_root"])
        ).resolve()
        _verify_frozen_execution_boundary(snapshot, config)
        if (
            Path(repository_root).resolve() != snapshot
            or Path(config_path).resolve()
            != (snapshot / "configs/synthetic_corrective_alignment_v1.yaml").resolve()
            or Path(input_root).resolve()
            != (Path(capability.resolved_inner_staging_root) / "persisted_inputs").resolve()
            or Path(output_root).resolve()
            != (Path(capability.resolved_inner_staging_root) / "parallel_compute").resolve()
        ):
            raise PermissionError("development parallel roots differ from authorization")
    elif capability is not None:
        raise PermissionError("development capability cannot be used for a non-development purpose")
    if purpose == "excluded_rehearsal":
        if _rehearsal_seal is not _REHEARSAL_INTERNAL_SEAL:
            raise PermissionError("excluded rehearsal execution requires the one-shot runner")
        _verify_rehearsal_capability(
            _rehearsal_capability,
            config=config,
            jobs=jobs,
            persisted_input_root=input_root,
            parallel_output_root=output_root,
            snapshot_root=repository_root,
        )
        _verify_frozen_execution_boundary(
            _rehearsal_capability.snapshot_root,
            config,
        )
        if Path(config_path).resolve() != (
            Path(_rehearsal_capability.snapshot_root)
            / "configs/synthetic_corrective_alignment_v1.yaml"
        ).resolve():
            raise PermissionError("excluded rehearsal config path mismatch")
    _require_thread_environment(config)
    output = Path(output_root).resolve()
    require_lstat_absent(output)
    input_base = Path(input_root).resolve()
    input_verification = verify_persisted_inputs(input_base)
    if input_verification["status"] != "PASS":
        raise RuntimeError(f"persisted inputs failed exact verification: {input_verification}")
    input_manifest_sha256 = sha256_file(input_base / "input_manifest.json")
    if input_manifest_sha256 != str(expected_input_manifest_sha256):
        raise RuntimeError("persisted input manifest changed after preparation")
    input_manifest = read_json(input_base / "input_manifest.json")
    generation_ledger = read_json(input_base / "generation_operation_ledger.json")
    expected_generation = [
        {
            "unit_id": f"corpus-{int(corpus_seed)}",
            "operation": operation,
            "references": [{"role": "corpus", "value": int(corpus_seed)}],
        }
        for corpus_seed in config["resolved_registries"][purpose]["corpus"]
        for operation in ("generate", "condition")
    ]
    generation_verification = _verify_global_ledger(
        generation_ledger,
        expected_generation,
        purpose=purpose,
        authorization_digest=(capability.authorization_digest if capability else None),
        parsed_contract_digest=(capability.parsed_contract_digest if capability else None),
        operation_contract_digest=_config_contract(config, purpose=purpose),
    )
    if generation_verification["status"] != "PASS":
        raise RuntimeError(f"persisted generation ledger failed: {generation_verification}")
    prequalification_lock, prequalification_lock_sha256 = (
        _verified_prequalification_lock(repository_root, config)
    )
    require_lstat_absent(output)
    output.mkdir(parents=True)
    plan = build_work_plan(config, purpose=purpose)
    adjudication = output / "adjudication"
    shards = output / "shards"
    telemetry_root = output / "runtime"
    adjudication.mkdir()
    shards.mkdir()
    telemetry_root.mkdir()
    write_json(adjudication / "work_plan.json", plan)
    capability_digest = capability.authorization_digest if capability else ""
    parsed_contract_digest = capability.parsed_contract_digest if capability else ""
    repository = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    runner_path = repository / "scripts/run_synthetic_corrective_alignment_v1.py"
    snapshot_manifest = repository / "snapshot_manifest.json"
    freeze_receipt = repository.parent / "freeze_receipt.json"
    scientific_contract = {
        "schema_version": "nursery-corrective-scientific-execution-contract-v1",
        "purpose": purpose,
        "protocol_id": config["protocol"]["id"],
        "config_sha256": sha256_file(config_file),
        "runner_sha256": sha256_file(runner_path),
        "snapshot_manifest_sha256": (
            sha256_file(snapshot_manifest) if snapshot_manifest.is_file() else None
        ),
        "freeze_receipt_sha256": (
            sha256_file(freeze_receipt) if freeze_receipt.is_file() else None
        ),
        "prequalification_design_lock_sha256": prequalification_lock_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "work_plan_digest": plan["digest"],
        "authorization_digest": capability_digest or None,
        "parsed_contract_digest": parsed_contract_digest or None,
    }
    scientific_contract_digest = canonical_digest(scientific_contract)
    runtime_contract = {
        "schema_version": "nursery-corrective-runtime-execution-contract-v1",
        "scientific_contract_digest": scientific_contract_digest,
        "jobs": int(jobs),
        "thread_environment": dict(config["parallel"]["thread_environment"]),
        "start_method": str(config["parallel"]["start_method"]),
    }
    runtime_contract_digest = canonical_digest(runtime_contract)
    write_json(
        adjudication / "execution_contract.json",
        {**scientific_contract, "scientific_contract_digest": scientific_contract_digest},
    )
    write_json(
        telemetry_root / "runtime_contract.json",
        {**runtime_contract, "runtime_contract_digest": runtime_contract_digest},
    )
    arguments_by_id = {
        str(unit["unit_id"]): {
            "unit": unit,
            "purpose": purpose,
            "repository_root": str(Path(repository_root).resolve()),
            "config_path": str(Path(config_path).resolve()),
            "input_root": str(input_base),
            "input_manifest_sha256": input_manifest_sha256,
            "shard_base": str(shards),
            "thread_environment": config["parallel"]["thread_environment"],
            "authorization_digest": capability_digest,
            "parsed_contract_digest": parsed_contract_digest,
            "scientific_contract_digest": scientific_contract_digest,
            "claim_path": capability.claim_path if capability else "",
            "claim_sha256": capability.claim_sha256 if capability else "",
            "issuance_receipt_path": (
                capability.issuance_receipt_path if capability else ""
            ),
            "issuance_receipt_sha256": (
                capability.issuance_receipt_sha256 if capability else ""
            ),
            "development_output_root": (
                capability.resolved_output_root if capability else ""
            ),
            "development_staging_root": (
                capability.resolved_staging_root if capability else ""
            ),
            "development_inner_staging_root": (
                capability.resolved_inner_staging_root if capability else ""
            ),
            "rehearsal_snapshot_root": (
                _rehearsal_capability.snapshot_root
                if _rehearsal_capability is not None
                else ""
            ),
            "rehearsal_output_root": (
                _rehearsal_capability.output_root
                if _rehearsal_capability is not None
                else ""
            ),
            "rehearsal_parallel_output_root": (
                _rehearsal_capability.parallel_output_root
                if _rehearsal_capability is not None
                else ""
            ),
            "rehearsal_receipt_path": (
                _rehearsal_capability.attempt_receipt_path
                if _rehearsal_capability is not None
                else ""
            ),
            "rehearsal_receipt_sha256": (
                _rehearsal_capability.attempt_receipt_sha256
                if _rehearsal_capability is not None
                else ""
            ),
            "required_jobs": int(jobs),
        }
        for unit in plan["units"]
    }
    context = multiprocessing.get_context(str(config["parallel"]["start_method"]))
    telemetry, failures = _execute_worker_batches(
        plan=plan,
        arguments_by_id=arguments_by_id,
        jobs=jobs,
        context=context,
    )
    if failures:
        write_json(
            telemetry_root / "FAILED_NO_OUTCOME.json",
            {
                "status": "FAILED_NO_OUTCOME",
                "scientific_outcome": False,
                "failures": failures,
            },
        )
        raise RuntimeError(f"parallel workers failed closed: {failures}")
    post_worker_input_verification = verify_persisted_inputs(input_base)
    if (
        post_worker_input_verification.get("status") != "PASS"
        or sha256_file(input_base / "input_manifest.json")
        != input_manifest_sha256
        or sha256_file(prequalification_lock) != prequalification_lock_sha256
    ):
        raise RuntimeError("persisted inputs or design lock changed during workers")
    if purpose in {"development", "excluded_rehearsal"}:
        _verify_frozen_execution_boundary(repository_root, config)
    _validate_shard_inventory(
        shards,
        [str(unit["unit_id"]) for unit in plan["units"]],
    )
    expected_unit_ids = [str(unit["unit_id"]) for unit in plan["units"]]
    telemetry_ids = [str(row.get("unit_id")) for row in telemetry]
    expected_telemetry_fields = {
        "unit_id",
        "shard_manifest_sha256",
        "wall_seconds",
        "worker_peak_rss_native_units",
        "thread_environment_observed",
        "native_threadpools",
    }
    if (
        len(telemetry_ids) != len(set(telemetry_ids))
        or set(telemetry_ids) != set(expected_unit_ids)
        or any(set(row) != expected_telemetry_fields for row in telemetry)
    ):
        raise RuntimeError("worker telemetry/unit commitment mismatch")
    telemetry_by_id = {str(row["unit_id"]): row for row in telemetry}
    for unit_id in expected_unit_ids:
        if telemetry_by_id[unit_id]["shard_manifest_sha256"] != sha256_file(
            shards / unit_id / "manifest.json"
        ):
            raise RuntimeError(f"worker shard commitment changed before merge: {unit_id}")
    merged_results = []
    merged_ledgers = []
    scored_conditions = []
    scored_mutations = []
    parent_firewall = IdentifierFirewall(
        config,
        purpose=purpose,
        capability=(
            capability
            if purpose == "development"
            else _rehearsal_capability
            if purpose == "excluded_rehearsal"
            else None
        ),
        operation_contract_digest=scientific_contract_digest,
    )
    expected_parent_operations = []
    for unit in plan["units"]:
        unit_id = str(unit["unit_id"])
        result = _validate_shard(
            shards / unit_id,
            unit,
            config,
            purpose=purpose,
            input_manifest_sha256=input_manifest_sha256,
            scientific_contract_digest=scientific_contract_digest,
            authorization_digest=capability_digest or None,
            parsed_contract_digest=parsed_contract_digest or None,
        )
        merged_results.append(result)
        merged_ledgers.append(read_json(shards / unit_id / "operation_ledger.json"))
        protected = _read_input_manifest_committed_json(
            input_base,
            input_manifest,
            f"corpus-{int(unit['corpus_seed'])}/protected_adjudication.json",
        )
        keys = protected["evaluation_keys"]
        for condition_result in result["condition_results"]:
            metrics = score_predictions(condition_result["predictions"], keys, config)
            scored_conditions.append(
                {
                    "corpus_seed": int(unit["corpus_seed"]),
                    "model_seed": int(unit["model_seed"]),
                    "condition": str(condition_result["condition"]),
                    "model_digest": str(condition_result["model_digest"]),
                    "prediction_digest": str(condition_result["prediction_digest"]),
                    "metrics": metrics,
                }
            )
        for mutation_result in result["mutation_results"]:
            metrics = score_predictions(mutation_result["predictions"], keys, config)
            scored_mutations.append(
                {
                    "corpus_seed": int(unit["corpus_seed"]),
                    "model_seed": int(unit["model_seed"]),
                    "mutation": str(mutation_result["mutation"]),
                    "model_digest": str(mutation_result["model_digest"]),
                    "prediction_digest": str(mutation_result["prediction_digest"]),
                    "metrics": metrics,
                }
            )
        score_references = [
            IdentifierReference("corpus", int(unit["corpus_seed"])),
            IdentifierReference("model", int(unit["model_seed"])),
        ]
        parent_firewall.authorize(
            "score",
            score_references,
            unit_id=unit_id,
            local_sequence=len(parent_firewall.operations),
        )
        expected_parent_operations.append(
            {
                "unit_id": unit_id,
                "operation": "score",
                "references": [
                    {"role": "corpus", "value": int(unit["corpus_seed"])},
                    {"role": "model", "value": int(unit["model_seed"])},
                ],
            }
        )
    inference_references = [
        *[
            IdentifierReference("corpus", int(value))
            for value in plan["corpus_seed_order"]
        ],
        *[
            IdentifierReference("model", int(value))
            for value in plan["model_seed_order"]
        ],
        IdentifierReference(
            "inference", int(config["resolved_registries"][purpose]["inference"][0])
        ),
    ]
    parent_firewall.authorize(
        "inference",
        inference_references,
        unit_id="parent-inference",
        local_sequence=len(parent_firewall.operations),
    )
    expected_parent_operations.append(
        {
            "unit_id": "parent-inference",
            "operation": "inference",
            "references": [
                *[
                    {"role": "corpus", "value": int(value)}
                    for value in plan["corpus_seed_order"]
                ],
                *[
                    {"role": "model", "value": int(value)}
                    for value in plan["model_seed_order"]
                ],
                {
                    "role": "inference",
                    "value": int(
                        config["resolved_registries"][purpose]["inference"][0]
                    ),
                },
            ],
        }
    )
    parent_ledger = parent_firewall.ledger()
    parent_ledger_verification = _verify_global_ledger(
        parent_ledger,
        expected_parent_operations,
        purpose=purpose,
        authorization_digest=capability_digest or None,
        parsed_contract_digest=parsed_contract_digest or None,
        operation_contract_digest=scientific_contract_digest,
    )
    if parent_ledger_verification["status"] != "PASS":
        raise RuntimeError(f"parent operation ledger failed: {parent_ledger_verification}")
    averaged = average_model_replicates(
        scored_conditions, config, purpose=purpose
    )
    primary = analyze_primary(averaged, config, purpose=purpose)
    dependence = dependence_diagnostics(averaged, config)
    mutations = _mutation_summary(scored_conditions, scored_mutations)
    corpus_audits = _read_input_manifest_committed_json(
        input_base,
        input_manifest,
        "corpus_audits.json",
    )["rows"]
    informativeness = analyze_informativeness(
        averaged,
        mutations,
        corpus_audits,
        config,
        purpose=purpose,
    )
    candidate_decision = _candidate_scientific_decision(
        purpose=purpose,
        primary_status=str(primary["status"]),
        informativeness_status=str(informativeness["status"]),
        dependence_status=str(dependence["status"]),
        causal_attribution_status=str(
            informativeness["causal_attribution_status"]
        ),
    )
    write_json(adjudication / "merged_worker_results.json", merged_results)
    write_json(adjudication / "merged_operation_ledgers.json", merged_ledgers)
    write_json(adjudication / "parent_operation_ledger.json", parent_ledger)
    write_json(
        adjudication / "parent_ledger_verification.json",
        parent_ledger_verification,
    )
    write_json(adjudication / "scored_condition_units.json", scored_conditions)
    write_json(adjudication / "scored_mutation_units.json", scored_mutations)
    write_json(adjudication / "model_averages.json", averaged)
    write_json(adjudication / "primary_inference.json", primary)
    write_json(adjudication / "present_null_dependence.json", dependence)
    write_json(adjudication / "mechanism_mutations.json", mutations)
    write_json(adjudication / "informativeness.json", informativeness)
    write_json(
        adjudication / "scientific_summary.json",
        {
            "schema_version": "nursery-corrective-scientific-summary-v1",
            "purpose": purpose,
            "primary_status": primary["status"],
            "informativeness_status": informativeness["status"],
            "dependence_status": dependence["status"],
            "causal_attribution_status": informativeness[
                "causal_attribution_status"
            ],
            "candidate_decision": candidate_decision,
            "publication_state": "UNPUBLISHED_CANDIDATE",
            "scientific_outcome": False,
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
        },
    )
    completeness = {
        "schema_version": "nursery-corrective-completeness-v1",
        "status": "PASS",
        "purpose": purpose,
        "expected_unit_count": int(plan["unit_count"]),
        "observed_unit_count": len(merged_results),
        "unique_unit_count": len({row["unit_id"] for row in merged_results}),
        "condition_count_per_unit": len(config["design"]["conditions"]),
        "mutation_count_per_unit": len(config["design"]["mechanism_mutations"]),
        "canonical_merge_order_followed": True,
        "missing_units": [],
        "duplicate_units": [],
        "unexpected_units": [],
        "publication_state": "UNPUBLISHED_CANDIDATE",
        "scientific_outcome": False,
        "development_outcome_count": 0,
        "confirmation_outcome": False,
    }
    write_json(adjudication / "completeness.json", completeness)
    adjudication_manifest = manifest_for_tree(
        adjudication, exclude=["manifest.json"]
    )
    write_json(adjudication / "manifest.json", adjudication_manifest)
    adjudication_verification = verify_exact_manifest(
        adjudication,
        adjudication_manifest,
        manifest_filename="manifest.json",
    )
    if adjudication_verification["status"] != "PASS":
        raise RuntimeError(
            f"adjudication manifest failed: {adjudication_verification}"
        )
    telemetry_sorted = sorted(telemetry, key=lambda row: row["unit_id"])
    write_json(
        telemetry_root / "runtime_telemetry.json",
        {
            "schema_version": "nursery-corrective-runtime-v1",
            "jobs": int(jobs),
            "units": telemetry_sorted,
            "wall_seconds_sum": float(
                sum(float(row["wall_seconds"]) for row in telemetry_sorted)
            ),
            "maximum_worker_peak_rss_native_units": max(
                int(row["worker_peak_rss_native_units"]) for row in telemetry_sorted
            ),
            "excluded_from_adjudication": True,
        },
    )
    final_input_verification = verify_persisted_inputs(input_base)
    if (
        final_input_verification.get("status") != "PASS"
        or sha256_file(input_base / "input_manifest.json")
        != input_manifest_sha256
        or sha256_file(prequalification_lock) != prequalification_lock_sha256
    ):
        raise RuntimeError("persisted inputs or design lock changed before sealing")
    if purpose in {"development", "excluded_rehearsal"}:
        _verify_frozen_execution_boundary(repository_root, config)
    summary = {
        "schema_version": "nursery-corrective-cohort-summary-v1",
        "status": "PASS",
        "purpose": purpose,
        "integrity_status": "PASS",
        "scientific_decision": (
            "SCIENTIFIC_INFERENCE_SUPPRESSED"
            if purpose != "development"
            else "PENDING_ATOMIC_PUBLICATION"
        ),
        "candidate_decision": candidate_decision,
        "publication_state": "UNPUBLISHED_CANDIDATE",
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "confirmation_authorized": False,
        "adjudication_manifest_sha256": sha256_file(
            adjudication / "manifest.json"
        ),
        "input_manifest_sha256": input_manifest_sha256,
        "scientific_contract_digest": scientific_contract_digest,
        "runtime_contract_digest": runtime_contract_digest,
    }
    write_json(output / "cohort_summary.json", summary)
    return summary


def _recompute_adjudication_from_rehearsal(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    rehearsal_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    rehearsal = Path(rehearsal_root).resolve()
    output = Path(output_root).resolve()
    _verify_frozen_execution_boundary(snapshot, config)
    repository = snapshot.parents[2]
    expected_rehearsal = (
        repository / str(config["paths"]["excluded_rehearsal"])
    ).resolve()
    expected_output = (
        repository / str(config["paths"]["independent_recompute"])
    ).resolve()
    if (
        snapshot.name != "frozen_source_snapshot"
        or config_file != snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
        or config["protocol"]["status"] != "frozen"
        or rehearsal != expected_rehearsal
        or output != expected_output
    ):
        raise PermissionError("recompute requires the frozen snapshot and config")
    completion_path = snapshot.parent / "excluded_rehearsal_completion.json"
    receipt_path = snapshot.parent / "excluded_rehearsal_attempt_consumed.json"
    completion = read_json(completion_path)
    expected_completion_fields = {
        "schema_version",
        "status",
        "attempt_receipt_sha256",
        "cohort_summary_sha256",
        "complete_manifest_sha256",
        "scientific_decision",
        "scientific_inference_suppressed",
        "development_outcome_count",
        "confirmation_outcome_count",
    }
    if (
        set(completion) != expected_completion_fields
        or completion.get("schema_version")
        != "nursery-corrective-excluded-rehearsal-completion-v1"
        or completion.get("status") != "PASS"
        or completion.get("attempt_receipt_sha256") != sha256_file(receipt_path)
        or completion.get("cohort_summary_sha256")
        != sha256_file(rehearsal / "cohort_summary.json")
        or completion.get("complete_manifest_sha256")
        != sha256_file(rehearsal / "complete_manifest.json")
        or completion.get("scientific_decision")
        != "SCIENTIFIC_INFERENCE_SUPPRESSED"
        or completion.get("scientific_inference_suppressed") is not True
        or int(completion.get("development_outcome_count", -1)) != 0
        or int(completion.get("confirmation_outcome_count", -1)) != 0
    ):
        raise PermissionError("recompute rehearsal completion hash chain mismatch")
    require_lstat_absent(output)
    rehearsal_manifest = read_json(rehearsal / "complete_manifest.json")
    rehearsal_verification = verify_exact_manifest(
        rehearsal,
        rehearsal_manifest,
        manifest_filename="complete_manifest.json",
    )
    if rehearsal_verification["status"] != "PASS":
        raise RuntimeError(f"rehearsal tree changed before recompute: {rehearsal_verification}")
    original_adjudication = rehearsal / "adjudication"
    original_manifest = read_json(original_adjudication / "manifest.json")
    original_verification = verify_exact_manifest(
        original_adjudication,
        original_manifest,
        manifest_filename="manifest.json",
    )
    if original_verification["status"] != "PASS":
        raise RuntimeError(f"rehearsal adjudication changed: {original_verification}")
    input_root = rehearsal / "persisted_inputs"
    input_verification = verify_persisted_inputs(input_root)
    if input_verification["status"] != "PASS":
        raise RuntimeError(f"rehearsal persisted inputs changed: {input_verification}")
    plan = build_work_plan(config, purpose="excluded_rehearsal")
    if read_json(original_adjudication / "work_plan.json") != plan:
        raise RuntimeError("rehearsal work plan differs from frozen reconstruction")
    execution_contract = read_json(original_adjudication / "execution_contract.json")
    scientific_contract = {
        key: value
        for key, value in execution_contract.items()
        if key != "scientific_contract_digest"
    }
    scientific_contract_digest = canonical_digest(scientific_contract)
    if (
        execution_contract.get("scientific_contract_digest")
        != scientific_contract_digest
        or scientific_contract.get("purpose") != "excluded_rehearsal"
        or scientific_contract.get("protocol_id") != config["protocol"]["id"]
        or scientific_contract.get("config_sha256") != sha256_file(config_file)
        or scientific_contract.get("runner_sha256")
        != sha256_file(snapshot / "scripts/run_synthetic_corrective_alignment_v1.py")
        or scientific_contract.get("snapshot_manifest_sha256")
        != sha256_file(snapshot / "snapshot_manifest.json")
        or scientific_contract.get("freeze_receipt_sha256")
        != sha256_file(snapshot.parent / "freeze_receipt.json")
        or scientific_contract.get("prequalification_design_lock_sha256")
        != sha256_file(snapshot.parent / "prequalification_design_lock.json")
        or scientific_contract.get("input_manifest_sha256")
        != sha256_file(input_root / "input_manifest.json")
        or scientific_contract.get("work_plan_digest") != plan["digest"]
        or scientific_contract.get("authorization_digest") is not None
        or scientific_contract.get("parsed_contract_digest") is not None
    ):
        raise RuntimeError("rehearsal scientific execution contract mismatch")

    output.mkdir()
    adjudication = output / "adjudication"
    adjudication.mkdir()
    write_json(adjudication / "work_plan.json", plan)
    write_json(adjudication / "execution_contract.json", execution_contract)
    merged_results = []
    merged_ledgers = []
    scored_conditions = []
    scored_mutations = []
    for unit in plan["units"]:
        unit_id = str(unit["unit_id"])
        result = _validate_shard(
            rehearsal / "shards" / unit_id,
            unit,
            config,
            purpose="excluded_rehearsal",
            input_manifest_sha256=sha256_file(input_root / "input_manifest.json"),
            scientific_contract_digest=scientific_contract_digest,
            authorization_digest=None,
            parsed_contract_digest=None,
        )
        merged_results.append(result)
        merged_ledgers.append(
            read_json(rehearsal / "shards" / unit_id / "operation_ledger.json")
        )
        protected = read_json(
            input_root
            / f"corpus-{int(unit['corpus_seed'])}"
            / "protected_adjudication.json"
        )
        keys = protected["evaluation_keys"]
        for condition_result in result["condition_results"]:
            scored_conditions.append(
                {
                    "corpus_seed": int(unit["corpus_seed"]),
                    "model_seed": int(unit["model_seed"]),
                    "condition": str(condition_result["condition"]),
                    "model_digest": str(condition_result["model_digest"]),
                    "prediction_digest": str(condition_result["prediction_digest"]),
                    "metrics": score_predictions(
                        condition_result["predictions"], keys, config
                    ),
                }
            )
        for mutation_result in result["mutation_results"]:
            scored_mutations.append(
                {
                    "corpus_seed": int(unit["corpus_seed"]),
                    "model_seed": int(unit["model_seed"]),
                    "mutation": str(mutation_result["mutation"]),
                    "model_digest": str(mutation_result["model_digest"]),
                    "prediction_digest": str(mutation_result["prediction_digest"]),
                    "metrics": score_predictions(
                        mutation_result["predictions"], keys, config
                    ),
                }
            )
    parent_ledger = read_json(original_adjudication / "parent_operation_ledger.json")
    expected_parent_operations = [
        {
            "unit_id": str(unit["unit_id"]),
            "operation": "score",
            "references": [
                {"role": "corpus", "value": int(unit["corpus_seed"])},
                {"role": "model", "value": int(unit["model_seed"])},
            ],
        }
        for unit in plan["units"]
    ]
    expected_parent_operations.append(
        {
            "unit_id": "parent-inference",
            "operation": "inference",
            "references": [
                *[
                    {"role": "corpus", "value": int(value)}
                    for value in plan["corpus_seed_order"]
                ],
                *[
                    {"role": "model", "value": int(value)}
                    for value in plan["model_seed_order"]
                ],
                {
                    "role": "inference",
                    "value": int(
                        config["resolved_registries"]["excluded_rehearsal"][
                            "inference"
                        ][0]
                    ),
                },
            ],
        }
    )
    parent_verification = _verify_global_ledger(
        parent_ledger,
        expected_parent_operations,
        purpose="excluded_rehearsal",
        authorization_digest=None,
        parsed_contract_digest=None,
        operation_contract_digest=scientific_contract_digest,
    )
    if parent_verification["status"] != "PASS":
        raise RuntimeError(f"rehearsal parent ledger mismatch: {parent_verification}")
    averaged = average_model_replicates(
        scored_conditions, config, purpose="excluded_rehearsal"
    )
    primary = analyze_primary(averaged, config, purpose="excluded_rehearsal")
    dependence = dependence_diagnostics(averaged, config)
    mutations = _mutation_summary(scored_conditions, scored_mutations)
    corpus_audits = read_json(input_root / "corpus_audits.json")["rows"]
    informativeness = analyze_informativeness(
        averaged,
        mutations,
        corpus_audits,
        config,
        purpose="excluded_rehearsal",
    )
    candidate_decision = "SCIENTIFIC_INFERENCE_SUPPRESSED"
    write_json(adjudication / "merged_worker_results.json", merged_results)
    write_json(adjudication / "merged_operation_ledgers.json", merged_ledgers)
    write_json(adjudication / "parent_operation_ledger.json", parent_ledger)
    write_json(adjudication / "parent_ledger_verification.json", parent_verification)
    write_json(adjudication / "scored_condition_units.json", scored_conditions)
    write_json(adjudication / "scored_mutation_units.json", scored_mutations)
    write_json(adjudication / "model_averages.json", averaged)
    write_json(adjudication / "primary_inference.json", primary)
    write_json(adjudication / "present_null_dependence.json", dependence)
    write_json(adjudication / "mechanism_mutations.json", mutations)
    write_json(adjudication / "informativeness.json", informativeness)
    write_json(
        adjudication / "scientific_summary.json",
        {
            "schema_version": "nursery-corrective-scientific-summary-v1",
            "purpose": "excluded_rehearsal",
            "primary_status": primary["status"],
            "informativeness_status": informativeness["status"],
            "dependence_status": dependence["status"],
            "causal_attribution_status": informativeness[
                "causal_attribution_status"
            ],
            "candidate_decision": candidate_decision,
            "publication_state": "UNPUBLISHED_CANDIDATE",
            "scientific_outcome": False,
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
        },
    )
    completeness = {
        "schema_version": "nursery-corrective-completeness-v1",
        "status": "PASS",
        "purpose": "excluded_rehearsal",
        "expected_unit_count": int(plan["unit_count"]),
        "observed_unit_count": len(merged_results),
        "unique_unit_count": len({row["unit_id"] for row in merged_results}),
        "condition_count_per_unit": len(config["design"]["conditions"]),
        "mutation_count_per_unit": len(config["design"]["mechanism_mutations"]),
        "canonical_merge_order_followed": True,
        "missing_units": [],
        "duplicate_units": [],
        "unexpected_units": [],
        "publication_state": "UNPUBLISHED_CANDIDATE",
        "scientific_outcome": False,
        "development_outcome_count": 0,
        "confirmation_outcome": False,
    }
    write_json(adjudication / "completeness.json", completeness)
    manifest = manifest_for_tree(adjudication, exclude=["manifest.json"])
    write_json(adjudication / "manifest.json", manifest)
    verification = verify_exact_manifest(
        adjudication, manifest, manifest_filename="manifest.json"
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"recomputed adjudication manifest failed: {verification}")

    rehearsal_inner = rehearsal.with_name(f".{rehearsal.name}.cohort-staging")
    recompute_capability = _issue_excluded_rehearsal_capability(
        config=config,
        scope="parent",
        unit_id=None,
        snapshot_root=snapshot,
        output_root=rehearsal,
        persisted_input_root=rehearsal_inner / "persisted_inputs",
        parallel_output_root=rehearsal_inner / "parallel_compute",
        attempt_receipt_path=receipt_path,
        attempt_receipt_sha256=sha256_file(receipt_path),
        required_jobs=int(config["parallel"]["frozen_jobs"]),
    )
    recompute_firewall = IdentifierFirewall(
        config,
        purpose="excluded_rehearsal",
        capability=recompute_capability,
        operation_contract_digest=scientific_contract_digest,
    )
    recompute_references = [
        *[
            IdentifierReference("corpus", int(value))
            for value in plan["corpus_seed_order"]
        ],
        *[
            IdentifierReference("model", int(value))
            for value in plan["model_seed_order"]
        ],
        IdentifierReference(
            "inference",
            int(config["resolved_registries"]["excluded_rehearsal"]["inference"][0]),
        ),
    ]
    recompute_firewall.authorize(
        "recompute",
        recompute_references,
        unit_id="independent-adjudication-recompute",
        local_sequence=0,
    )
    recompute_ledger = recompute_firewall.ledger()
    recompute_expected = [
        {
            "unit_id": "independent-adjudication-recompute",
            "operation": "recompute",
            "references": [
                *[
                    {"role": "corpus", "value": int(value)}
                    for value in plan["corpus_seed_order"]
                ],
                *[
                    {"role": "model", "value": int(value)}
                    for value in plan["model_seed_order"]
                ],
                {
                    "role": "inference",
                    "value": int(
                        config["resolved_registries"]["excluded_rehearsal"][
                            "inference"
                        ][0]
                    ),
                },
            ],
        }
    ]
    recompute_ledger_verification = _verify_global_ledger(
        recompute_ledger,
        recompute_expected,
        purpose="excluded_rehearsal",
        authorization_digest=None,
        parsed_contract_digest=None,
        operation_contract_digest=scientific_contract_digest,
    )
    write_json(output / "recompute_operation_ledger.json", recompute_ledger)
    write_json(
        output / "recompute_ledger_verification.json",
        recompute_ledger_verification,
    )
    complete_manifest = manifest_for_tree(output, exclude=["complete_manifest.json"])
    write_json(output / "complete_manifest.json", complete_manifest)
    complete_verification = verify_exact_manifest(
        output, complete_manifest, manifest_filename="complete_manifest.json"
    )
    if recompute_ledger_verification["status"] != "PASS" or complete_verification[
        "status"
    ] != "PASS":
        raise RuntimeError("recompute ledger or complete manifest failed")
    return {
        "status": "PASS",
        "source_worker_outputs_reused_without_refit": True,
        "models_refit": False,
        "adjudication_manifest_sha256": sha256_file(adjudication / "manifest.json"),
        "complete_manifest_sha256": sha256_file(output / "complete_manifest.json"),
    }


def execute_cohort(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    purpose: str,
    jobs: int,
    capability: AuthorizationCapability | None = None,
    _development_seal: object | None = None,
    _rehearsal_seal: object | None = None,
    _rehearsal_capability: ExcludedRehearsalCapability | None = None,
) -> dict[str, Any]:
    if purpose == "development":
        if _development_seal is not _DEVELOPMENT_INTERNAL_SEAL:
            raise PermissionError("development is unavailable through the generic cohort API")
        if capability is None:
            raise PermissionError("development requires an issued capability")
        output = Path(output_root).resolve()
        inner_staging = output.with_name(f".{output.name}.cohort-staging")
        verify_development_capability(
            capability,
            required_jobs=jobs,
            required_scope="parent",
            resolved_staging_root=output,
            resolved_inner_staging_root=inner_staging,
        )
        snapshot = Path(
            str(read_json(capability.claim_path)["resolved_snapshot_root"])
        ).resolve()
        _verify_frozen_execution_boundary(snapshot, config)
        if (
            Path(repository_root).resolve() != snapshot
            or Path(config_path).resolve()
            != (snapshot / "configs/synthetic_corrective_alignment_v1.yaml").resolve()
        ):
            raise PermissionError("development cohort source roots mismatch")
    elif capability is not None:
        raise PermissionError("capability cannot be used outside development")
    if purpose == "excluded_rehearsal":
        if _rehearsal_seal is not _REHEARSAL_INTERNAL_SEAL:
            raise PermissionError("excluded rehearsal cohort requires the one-shot runner")
        _verify_rehearsal_capability(
            _rehearsal_capability,
            config=config,
            jobs=jobs,
            output_root=output_root,
            snapshot_root=repository_root,
        )
        _verify_frozen_execution_boundary(
            _rehearsal_capability.snapshot_root,
            config,
        )
        if Path(config_path).resolve() != (
            Path(_rehearsal_capability.snapshot_root)
            / "configs/synthetic_corrective_alignment_v1.yaml"
        ).resolve():
            raise PermissionError("excluded rehearsal cohort config path mismatch")
    output = Path(output_root).resolve()
    require_lstat_absent(output)
    staging = output.with_name(f".{output.name}.cohort-staging")
    require_lstat_absent(staging)
    staging.mkdir()
    inputs = staging / "persisted_inputs"
    compute = staging / "parallel_compute"
    prepared = prepare_persisted_inputs(
        inputs,
        config,
        purpose=purpose,
        capability=capability,
        _development_seal=_development_seal,
        _rehearsal_seal=_rehearsal_seal,
        _rehearsal_capability=_rehearsal_capability,
    )
    summary = run_parallel_from_persisted(
        repository_root=repository_root,
        config_path=config_path,
        input_root=inputs,
        output_root=compute,
        config=config,
        purpose=purpose,
        jobs=jobs,
        capability=capability,
        _development_seal=_development_seal,
        _rehearsal_seal=_rehearsal_seal,
        _rehearsal_capability=_rehearsal_capability,
        expected_input_manifest_sha256=str(prepared["input_manifest_sha256"]),
    )
    shutil.move(str(compute / "adjudication"), str(staging / "adjudication"))
    shutil.move(str(compute / "runtime"), str(staging / "runtime"))
    shutil.move(str(compute / "shards"), str(staging / "shards"))
    shutil.move(
        str(compute / "cohort_summary.json"),
        str(staging / "cohort_summary.json"),
    )
    if read_json(staging / "cohort_summary.json") != summary:
        raise RuntimeError("persisted cohort summary differs from returned summary")
    compute.rmdir()
    manifest = manifest_for_tree(staging, exclude=["complete_manifest.json"])
    write_json(staging / "complete_manifest.json", manifest)
    verification = verify_exact_manifest(
        staging,
        manifest,
        manifest_filename="complete_manifest.json",
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"cohort complete manifest failed: {verification}")
    atomic_rename_directory_noreplace(staging, output)
    return summary


def _execute_development_cohort(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    jobs: int,
    capability: AuthorizationCapability,
) -> dict[str, Any]:
    claim = read_json(capability.claim_path)
    snapshot = Path(str(claim["resolved_snapshot_root"])).resolve()
    _verify_frozen_execution_boundary(snapshot, config)
    expected_config_path = snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
    if Path(repository_root).resolve() != snapshot:
        raise PermissionError("development repository must be the authorized snapshot")
    if Path(config_path).resolve() != expected_config_path.resolve():
        raise PermissionError("development config path must be the authorized snapshot config")
    loaded = load_config(expected_config_path, repository_root=snapshot)
    if loaded["protocol"]["status"] != "frozen" or canonical_digest(loaded) != canonical_digest(
        config
    ):
        raise PermissionError("development in-memory config differs from frozen config")
    verify_development_capability(
        capability,
        required_jobs=jobs,
        required_scope="parent",
        resolved_staging_root=output_root,
    )
    return execute_cohort(
        repository_root=repository_root,
        config_path=config_path,
        output_root=output_root,
        config=config,
        purpose="development",
        jobs=jobs,
        capability=capability,
        _development_seal=_DEVELOPMENT_INTERNAL_SEAL,
    )


def _execute_excluded_rehearsal_cohort(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    jobs: int,
) -> dict[str, Any]:
    snapshot = Path(repository_root).resolve()
    _verify_frozen_execution_boundary(snapshot, config)
    expected_config = snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
    loaded = load_config(expected_config, repository_root=snapshot)
    repository = snapshot.parents[2]
    expected_output = repository / str(loaded["paths"]["excluded_rehearsal"])
    if (
        snapshot.name != "frozen_source_snapshot"
        or snapshot.parent.name != "synthetic_corrective_development_launch_package_v1"
        or Path(config_path).resolve() != expected_config.resolve()
        or Path(output_root).resolve() != expected_output.resolve()
        or loaded["protocol"]["status"] != "frozen"
        or canonical_digest(loaded) != canonical_digest(config)
        or int(jobs) != int(loaded["parallel"]["frozen_jobs"])
    ):
        raise PermissionError("excluded rehearsal differs from the frozen one-shot contract")
    receipt_path = repository / str(
        loaded["paths"]["excluded_rehearsal_attempt_receipt"]
    )
    require_lstat_absent(receipt_path)
    receipt = {
        "schema_version": "nursery-corrective-excluded-rehearsal-attempt-v1",
        "status": "ONE_EXCLUDED_ATTEMPT_CONSUMED",
        "snapshot_root": str(snapshot),
        "snapshot_manifest_sha256": sha256_file(snapshot / "snapshot_manifest.json"),
        "freeze_receipt_sha256": sha256_file(snapshot.parent / "freeze_receipt.json"),
        "output_root": str(expected_output.resolve()),
        "jobs": int(jobs),
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "non_replayable": True,
    }
    write_json(receipt_path, receipt)
    rehearsal_inner = expected_output.with_name(
        f".{expected_output.name}.cohort-staging"
    )
    rehearsal_capability = _issue_excluded_rehearsal_capability(
        config=loaded,
        scope="parent",
        unit_id=None,
        snapshot_root=snapshot,
        output_root=expected_output,
        persisted_input_root=rehearsal_inner / "persisted_inputs",
        parallel_output_root=rehearsal_inner / "parallel_compute",
        attempt_receipt_path=receipt_path,
        attempt_receipt_sha256=sha256_file(receipt_path),
        required_jobs=int(jobs),
    )
    return execute_cohort(
        repository_root=snapshot,
        config_path=expected_config,
        output_root=expected_output,
        config=loaded,
        purpose="excluded_rehearsal",
        jobs=int(jobs),
        _rehearsal_seal=_REHEARSAL_INTERNAL_SEAL,
        _rehearsal_capability=rehearsal_capability,
    )
