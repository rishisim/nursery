from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from babyworld_lite.corrective_alignment_v1.adjudicator import (  # noqa: E402
    PACKAGE_GATES,
    construction_qualification_decision,
    verify_authorization,
)
from babyworld_lite.corrective_alignment_v1.integrity import (  # noqa: E402
    benchmark_report,
    _issue_finalization_proof,
    compare_adjudication_trees,
    compare_recompute,
    recompute_comparison_value,
    environment_record,
    finalize_package,
    freeze_snapshot,
    operation_identifier_audit,
    repository_operation_census,
    reverify_prior_closed_line,
    run_tests,
    verify_prior_preservation,
    verify_prefreeze_evidence,
    verify_snapshot,
    validate_benchmark_report,
    validate_official_test_report,
)
from babyworld_lite.corrective_alignment_v1.learner import (  # noqa: E402
    disagreement_mechanism_qualification,
)
from babyworld_lite.corrective_alignment_v1.parallel import (  # noqa: E402
    _verify_global_ledger,
    _execute_development_cohort,
    _execute_excluded_rehearsal_cohort,
    _recompute_adjudication_from_rehearsal,
    build_work_plan,
    execute_cohort,
    prepare_persisted_inputs,
    run_parallel_from_persisted,
    verify_persisted_inputs,
)
from babyworld_lite.corrective_alignment_v1.protocol import (  # noqa: E402
    AuthorizationCapability,
    IdentifierFirewall,
    atomic_rename_directory_noreplace,
    _issue_development_preflight_proof,
    atomic_write_bytes,
    canonical_digest,
    load_config,
    issue_development_capability,
    manifest_for_paths,
    manifest_for_tree,
    read_json,
    registry_snapshot,
    require_frozen,
    require_lstat_absent,
    sha256_file,
    tree_directory_paths,
    tree_file_paths,
    verify_development_capability,
    verify_exact_manifest,
    write_json,
)


CONFIG_RELATIVE = "configs/synthetic_corrective_alignment_v1.yaml"
PACKAGE_RELATIVE = "output/synthetic_corrective_development_launch_package_v1"
TRACKED = (
    "babyworld_lite/__init__.py",
    "babyworld_lite/corrective_alignment_v1/__init__.py",
    "babyworld_lite/corrective_alignment_v1/protocol.py",
    "babyworld_lite/corrective_alignment_v1/generator.py",
    "babyworld_lite/corrective_alignment_v1/learner.py",
    "babyworld_lite/corrective_alignment_v1/statistics.py",
    "babyworld_lite/corrective_alignment_v1/parallel.py",
    "babyworld_lite/corrective_alignment_v1/adjudicator.py",
    "babyworld_lite/corrective_alignment_v1/integrity.py",
    "scripts/run_synthetic_corrective_alignment_v1.py",
    "configs/synthetic_corrective_alignment_v1.yaml",
    "tests/test_synthetic_corrective_alignment_v1.py",
    "docs/synthetic_corrective_alignment_v1_protocol.md",
    "docs/synthetic_corrective_alignment_v1_primary_sources.json",
    "docs/synthetic_corrective_alignment_v1_sample_size.md",
    "docs/synthetic_corrective_alignment_v1_traceability.json",
    "docs/synthetic_corrective_alignment_v1_research_rationale.md",
    "docs/synthetic_corrective_alignment_v1_threshold_rationale.md",
    "docs/synthetic_corrective_alignment_v1_brief_trace.json",
)


def _qualification_config_projection(config: dict) -> dict:
    value = json.loads(json.dumps(config))
    value.pop("repository_root", None)
    return value


def _anticipated_frozen_config_bytes(config_path: Path) -> bytes:
    payload = config_path.read_bytes()
    marker = b"  status: pre_freeze\n"
    if payload.count(marker) != 1:
        raise RuntimeError("config must contain one exact pre_freeze status line")
    return payload.replace(marker, b"  status: frozen\n", 1)


def _zero_outcome_registry(package: Path) -> dict:
    registry = read_json(package / "outcome_registry.json")
    expected_fields = {
        "schema_version",
        "development_outcome_count",
        "confirmation_outcome_count",
        "development_output_exists",
        "confirmation_output_exists",
        "authorization_consumed",
    }
    if not (
        set(registry) == expected_fields
        and registry.get("schema_version")
        == "nursery-corrective-outcome-registry-v1"
        and int(registry.get("development_outcome_count", -1)) == 0
        and int(registry.get("confirmation_outcome_count", -1)) == 0
        and registry.get("development_output_exists") is False
        and registry.get("confirmation_output_exists") is False
        and registry.get("authorization_consumed") is False
    ):
        raise RuntimeError("prequalification requires the exact zero-outcome registry")
    return registry


def _prequalification_pristine_paths(root: Path, config: dict) -> dict[str, str]:
    fields = {
        "launch_seal": "launch_seal",
        "launch_seal_staging": "launch_seal_staging",
        "authorization_claim": "authorization_claim",
        "authorization_capability_receipt": "authorization_capability_receipt",
        "development_output": "development_output",
        "development_staging": "development_staging",
        "development_inner_staging": "development_inner_staging",
        "development_publication_seal": "development_publication_seal",
    }
    return {
        name: str((root / str(config["paths"][field])).absolute())
        for name, field in fields.items()
    }


def _require_prequalification_pristine_paths(
    root: Path,
    contract: dict[str, str],
) -> None:
    for name, value in contract.items():
        path = Path(value)
        require_lstat_absent(path)
        _no_symlink_ancestry(path, stop=root)


def _official_prefreeze_prefixes() -> tuple[str, ...]:
    return (
        "parallel_micro_attempt_4",
        "parallel_equivalence_attempt_4",
        "parallel_benchmark_attempt_4",
        ".construction_qualification",
        "construction_qualification",
        "mechanism_qualification",
        "prequalification_transition_verification",
        "prefreeze_evidence",
        "frozen_source_snapshot",
        "freeze_receipt",
        "freeze_attempt_consumed",
        "frozen_environment",
        "frozen_seed_registries",
        "excluded_rehearsal",
        "independent_recompute",
        "official_test_execution_report",
        "official_tests_attempt_consumed",
        "preservation",
        "launch_seal",
        "package_core_manifest",
        "PACKAGE_NOT_READY",
        "SCIENTIFIC_REPORT",
        "prior_line_reverification",
        "identifier_operation_audit",
        "repository_operation_census",
        "design_contract_validation",
        "rehearsal_contract_validation",
        "pristine_state_validation",
        "traceability_validation",
        "adjudication_inputs",
        "finalization_attempt_consumed",
    )


_LIFECYCLE_ATTEMPTS = {
    "freeze": (
        "freeze_attempt_consumed.json",
        "ONE_FREEZE_ATTEMPT_CONSUMED",
        "construction_qualification_attempt_consumed.json",
    ),
    "independent_recompute": (
        "independent_recompute_attempt_consumed.json",
        "ONE_INDEPENDENT_RECOMPUTE_ATTEMPT_CONSUMED",
        "excluded_rehearsal_completion.json",
    ),
    "official_tests": (
        "official_tests_attempt_consumed.json",
        "ONE_OFFICIAL_TEST_ATTEMPT_CONSUMED",
        "independent_recompute_comparison.json",
    ),
    "preservation": (
        "preservation_attempt_consumed.json",
        "ONE_PRESERVATION_ATTEMPT_CONSUMED",
        "official_test_execution_report.json",
    ),
    "finalization": (
        "finalization_attempt_consumed.json",
        "ONE_FINALIZATION_ATTEMPT_CONSUMED",
        "preservation/preservation_proof.json",
    ),
}


def _lifecycle_attempt_payload(
    package: Path,
    source_root: Path,
    config: dict,
    operation: str,
) -> dict:
    if operation not in _LIFECYCLE_ATTEMPTS:
        raise ValueError(f"unknown lifecycle attempt: {operation}")
    _filename, status, prerequisite_relative = _LIFECYCLE_ATTEMPTS[operation]
    prerequisite = package / prerequisite_relative
    config_path = source_root / CONFIG_RELATIVE
    runner_path = source_root / "scripts/run_synthetic_corrective_alignment_v1.py"
    snapshot_manifest = package / "frozen_source_snapshot/snapshot_manifest.json"
    freeze_receipt = package / "freeze_receipt.json"
    return {
        "schema_version": "nursery-corrective-lifecycle-attempt-v1",
        "status": status,
        "operation": operation,
        "source_root": str(source_root.resolve()),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner_path),
        "prequalification_design_lock_sha256": sha256_file(
            package / "prequalification_design_lock.json"
        ),
        "snapshot_manifest_sha256": (
            None
            if operation == "freeze"
            else sha256_file(snapshot_manifest)
        ),
        "freeze_receipt_sha256": (
            None if operation == "freeze" else sha256_file(freeze_receipt)
        ),
        "prerequisite_path": prerequisite_relative,
        "prerequisite_sha256": sha256_file(prerequisite),
        "jobs": int(config["parallel"]["frozen_jobs"]),
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "non_replayable": True,
    }


def _consume_lifecycle_attempt(
    package: Path,
    source_root: Path,
    config: dict,
    operation: str,
) -> dict:
    filename = _LIFECYCLE_ATTEMPTS[operation][0]
    receipt_path = package / filename
    require_lstat_absent(receipt_path)
    payload = _lifecycle_attempt_payload(package, source_root, config, operation)
    write_json(receipt_path, payload)
    return payload


def _validate_lifecycle_attempt_receipts(
    root: Path,
    package: Path,
    snapshot: Path,
    config: dict,
) -> dict:
    checks = {}
    for operation, (filename, _status, _prerequisite) in _LIFECYCLE_ATTEMPTS.items():
        source_root = root if operation == "freeze" else snapshot
        try:
            checks[operation] = read_json(package / filename) == (
                _lifecycle_attempt_payload(
                    package,
                    source_root,
                    config,
                    operation,
                )
            )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            checks[operation] = False
    return {
        "schema_version": "nursery-corrective-lifecycle-attempt-validation-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def _create_prequalification_design_lock(
    root: Path,
    package: Path,
    config: dict,
) -> dict:
    if config["protocol"]["status"] != "pre_freeze":
        raise RuntimeError("prequalification design lock requires pre_freeze status")
    lock_path = package / "prequalification_design_lock.json"
    require_lstat_absent(lock_path)
    prefixes = _official_prefreeze_prefixes()
    forbidden_entries = sorted(
        child.name
        for child in package.iterdir()
        if any(child.name.startswith(prefix) for prefix in prefixes)
    )
    if forbidden_entries:
        raise RuntimeError(
            f"official fixture/freeze paths predate design lock: {forbidden_entries}"
        )
    registry = _zero_outcome_registry(package)
    pristine_paths = _prequalification_pristine_paths(root, config)
    _require_prequalification_pristine_paths(root, pristine_paths)
    preexisting_paths = tree_file_paths(package)
    preexisting_manifest = manifest_for_paths(package, preexisting_paths)
    preexisting_directories = tree_directory_paths(package)
    preexisting_bound_top_level_entries = sorted(
        {
            Path(path).parts[0]
            for path in [*preexisting_paths, *preexisting_directories]
        }
    )
    prefreeze_config = _qualification_config_projection(config)
    anticipated_frozen_config = json.loads(json.dumps(prefreeze_config))
    anticipated_frozen_config["protocol"]["status"] = "frozen"
    config_path = root / CONFIG_RELATIVE
    anticipated_frozen_bytes = _anticipated_frozen_config_bytes(config_path)
    nonconfig_tracked = sorted(
        (set(TRACKED) - {CONFIG_RELATIVE})
        | {str(config["registries"]["canonical_prior_registry"])}
    )
    source_manifest = manifest_for_paths(root, nonconfig_tracked)
    qualification_environment = environment_record(root, config)
    expected_thread_environment = dict(config["parallel"]["thread_environment"])
    observed_thread_environment = {
        name: os.environ.get(str(name)) for name in expected_thread_environment
    }
    if (
        observed_thread_environment != expected_thread_environment
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise RuntimeError("official qualification requires the frozen thread environment")
    lock = {
        "schema_version": "nursery-corrective-prequalification-design-lock-v1",
        "status": "LOCKED_BEFORE_OFFICIAL_FIXTURES",
        "tracked_nonconfig_manifest": source_manifest,
        "preexisting_package_manifest": preexisting_manifest,
        "preexisting_package_directories": preexisting_directories,
        "preexisting_bound_top_level_entries": (
            preexisting_bound_top_level_entries
        ),
        "prefreeze_config": prefreeze_config,
        "prefreeze_config_digest": canonical_digest(prefreeze_config),
        "prefreeze_config_sha256": sha256_file(config_path),
        "anticipated_frozen_config_digest": canonical_digest(
            anticipated_frozen_config
        ),
        "anticipated_frozen_config_sha256": hashlib.sha256(
            anticipated_frozen_bytes
        ).hexdigest(),
        "allowed_postqualification_delta": {
            "path": "protocol.status",
            "before": "pre_freeze",
            "after": "frozen",
        },
        "official_micro_attempt": 4,
        "construction_fixture_selection": config["qualification_gates"][
            "construction_fixture_selection"
        ],
        "official_paths_absent_at_lock": True,
        "official_forbidden_prefixes": list(prefixes),
        "qualification_environment": qualification_environment,
        "required_thread_environment": expected_thread_environment,
        "pristine_path_contract": pristine_paths,
        "outcome_registry": registry,
        "outcome_registry_sha256": sha256_file(package / "outcome_registry.json"),
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }
    write_json(lock_path, lock)
    return lock


def _verify_prequalification_design_lock(
    root: Path,
    package: Path,
    config: dict,
    *,
    require_fixture_contracts: bool,
    environment_root: Path | None = None,
    fixture_contract_names: tuple[str, ...] | None = None,
) -> dict:
    lock_path = package / "prequalification_design_lock.json"
    lock = read_json(lock_path)
    expected_fields = {
        "schema_version",
        "status",
        "tracked_nonconfig_manifest",
        "preexisting_package_manifest",
        "preexisting_package_directories",
        "preexisting_bound_top_level_entries",
        "prefreeze_config",
        "prefreeze_config_digest",
        "prefreeze_config_sha256",
        "anticipated_frozen_config_digest",
        "anticipated_frozen_config_sha256",
        "allowed_postqualification_delta",
        "official_micro_attempt",
        "construction_fixture_selection",
        "official_paths_absent_at_lock",
        "official_forbidden_prefixes",
        "qualification_environment",
        "required_thread_environment",
        "pristine_path_contract",
        "outcome_registry",
        "outcome_registry_sha256",
        "scientific_inference_suppressed",
        "development_outcome_count",
        "confirmation_outcome_count",
    }
    nonconfig_tracked = sorted(
        (set(TRACKED) - {CONFIG_RELATIVE})
        | {str(config["registries"]["canonical_prior_registry"])}
    )
    observed_source = manifest_for_paths(root, nonconfig_tracked)
    current_status = str(config["protocol"]["status"])
    prefreeze_config = lock.get("prefreeze_config", {})
    anticipated = json.loads(json.dumps(prefreeze_config))
    if isinstance(anticipated, dict) and isinstance(anticipated.get("protocol"), dict):
        anticipated["protocol"]["status"] = "frozen"
    current_projection = _qualification_config_projection(config)
    preexisting_paths = [
        str(row["path"])
        for row in lock.get("preexisting_package_manifest", {}).get("files", [])
    ]
    bound_top_level_entries = list(
        map(str, lock.get("preexisting_bound_top_level_entries", []))
    )
    try:
        all_current_files = tree_file_paths(package)
        all_current_directories = tree_directory_paths(package)
        observed_preexisting_paths = sorted(
            path
            for path in all_current_files
            if Path(path).parts[0] in set(bound_top_level_entries)
        )
        observed_preexisting_directories = sorted(
            path
            for path in all_current_directories
            if Path(path).parts[0] in set(bound_top_level_entries)
        )
        observed_preexisting = manifest_for_paths(
            package, observed_preexisting_paths
        )
    except (OSError, RuntimeError, ValueError, FileNotFoundError):
        observed_preexisting = None
        observed_preexisting_paths = []
        observed_preexisting_directories = []
    expected_pristine_paths = _prequalification_pristine_paths(
        environment_root or root,
        config,
    )
    pristine_paths_pass = lock.get("pristine_path_contract") == expected_pristine_paths
    if pristine_paths_pass:
        try:
            _require_prequalification_pristine_paths(
                environment_root or root,
                expected_pristine_paths,
            )
        except (OSError, RuntimeError, ValueError, FileNotFoundError):
            pristine_paths_pass = False
    try:
        observed_registry = _zero_outcome_registry(package)
    except (OSError, RuntimeError, ValueError, FileNotFoundError):
        observed_registry = None
    environment_base = (environment_root or root).resolve()
    try:
        observed_environment = environment_record(environment_base, config)
    except (OSError, RuntimeError, ValueError, FileNotFoundError):
        observed_environment = None
    checks = {
        "lock_schema": set(lock) == expected_fields
        and lock.get("schema_version")
        == "nursery-corrective-prequalification-design-lock-v1"
        and lock.get("status") == "LOCKED_BEFORE_OFFICIAL_FIXTURES",
        "tracked_nonconfig_unchanged": observed_source
        == lock.get("tracked_nonconfig_manifest"),
        "preexisting_package_evidence_unchanged": observed_preexisting
        == lock.get("preexisting_package_manifest")
        and observed_preexisting_paths == preexisting_paths
        and observed_preexisting_directories
        == lock.get("preexisting_package_directories")
        and bound_top_level_entries
        == sorted(
            {
                Path(path).parts[0]
                for path in [
                    *preexisting_paths,
                    *lock.get("preexisting_package_directories", []),
                ]
            }
        ),
        "prefreeze_config_self_consistent": lock.get("prefreeze_config_digest")
        == canonical_digest(prefreeze_config),
        "anticipated_config_self_consistent": lock.get(
            "anticipated_frozen_config_digest"
        )
        == canonical_digest(anticipated),
        "only_status_transition_allowed": lock.get(
            "allowed_postqualification_delta"
        )
        == {"path": "protocol.status", "before": "pre_freeze", "after": "frozen"},
        "current_config_is_locked_state": (
            current_status == "pre_freeze"
            and canonical_digest(current_projection)
            == lock.get("prefreeze_config_digest")
            and sha256_file(root / CONFIG_RELATIVE)
            == lock.get("prefreeze_config_sha256")
        )
        or (
            current_status == "frozen"
            and canonical_digest(current_projection)
            == lock.get("anticipated_frozen_config_digest")
            == canonical_digest(anticipated)
            and sha256_file(root / CONFIG_RELATIVE)
            == lock.get("anticipated_frozen_config_sha256")
        ),
        "official_attempt_and_selection_bound": int(
            lock.get("official_micro_attempt", -1)
        )
        == 4
        and lock.get("construction_fixture_selection")
        == prefreeze_config.get("qualification_gates", {}).get(
            "construction_fixture_selection"
        ),
        "official_paths_were_absent_at_lock": lock.get(
            "official_paths_absent_at_lock"
        )
        is True
        and lock.get("official_forbidden_prefixes") == list(
            _official_prefreeze_prefixes()
        ),
        "qualification_environment_unchanged": observed_environment
        == lock.get("qualification_environment"),
        "thread_environment_bound": lock.get("required_thread_environment")
        == prefreeze_config.get("parallel", {}).get("thread_environment")
        and {
            name: os.environ.get(str(name))
            for name in lock.get("required_thread_environment", {})
        }
        == lock.get("required_thread_environment")
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "scientific_transaction_paths_pristine": pristine_paths_pass,
        "outcome_registry_exact_zero": observed_registry
        == lock.get("outcome_registry")
        and lock.get("outcome_registry_sha256")
        == sha256_file(package / "outcome_registry.json"),
        "zero_outcome_lock": int(lock.get("development_outcome_count", -1)) == 0
        and int(lock.get("confirmation_outcome_count", -1)) == 0
        and lock.get("scientific_inference_suppressed") is True,
    }
    fixture_contracts = {}
    if require_fixture_contracts:
        runner_row = next(
            row
            for row in lock["tracked_nonconfig_manifest"]["files"]
            if row["path"] == "scripts/run_synthetic_corrective_alignment_v1.py"
        )
        expected_runner_sha = runner_row["sha256"]
        contracts = {
            "micro_jobs_1": {
                "root": package / "parallel_micro_attempt_4_jobs_1",
                "input": package / "parallel_micro_attempt_4_persisted_inputs",
                "purpose": "construction_micro",
                "jobs": 1,
                "complete_manifest": True,
            },
            "micro_jobs_n": {
                "root": package / "parallel_micro_attempt_4_jobs_n",
                "input": package / "parallel_micro_attempt_4_persisted_inputs",
                "purpose": "construction_micro",
                "jobs": int(prefreeze_config["parallel"]["frozen_jobs"]),
                "complete_manifest": True,
            },
            "construction": {
                "root": package / "construction_qualification",
                "input": package / "construction_qualification/persisted_inputs",
                "purpose": "construction_qualification",
                "jobs": int(prefreeze_config["parallel"]["frozen_jobs"]),
                "complete_manifest": True,
            },
        }
        selected_names = fixture_contract_names or tuple(contracts)
        if not selected_names or not set(selected_names) <= set(contracts):
            raise ValueError("unknown or empty fixture contract selection")
        expected_execution_fields = {
            "schema_version",
            "purpose",
            "protocol_id",
            "config_sha256",
            "runner_sha256",
            "snapshot_manifest_sha256",
            "freeze_receipt_sha256",
            "prequalification_design_lock_sha256",
            "input_manifest_sha256",
            "work_plan_digest",
            "authorization_digest",
            "parsed_contract_digest",
            "scientific_contract_digest",
        }
        expected_runtime_fields = {
            "schema_version",
            "scientific_contract_digest",
            "jobs",
            "thread_environment",
            "start_method",
            "runtime_contract_digest",
        }
        for name in selected_names:
            specification = contracts[name]
            fixture_root = specification["root"]
            input_root = specification["input"]
            try:
                execution_path = fixture_root / "adjudication/execution_contract.json"
                runtime_path = fixture_root / "runtime/runtime_contract.json"
                value = read_json(execution_path)
                runtime = read_json(runtime_path)
                scientific_payload = {
                    key: child
                    for key, child in value.items()
                    if key != "scientific_contract_digest"
                }
                runtime_payload = {
                    key: child
                    for key, child in runtime.items()
                    if key != "runtime_contract_digest"
                }
                input_manifest = read_json(input_root / "input_manifest.json")
                input_exact = verify_exact_manifest(
                    input_root,
                    input_manifest,
                    manifest_filename="input_manifest.json",
                )["status"] == "PASS"
                adjudication_manifest = read_json(
                    fixture_root / "adjudication/manifest.json"
                )
                adjudication_exact = verify_exact_manifest(
                    fixture_root / "adjudication",
                    adjudication_manifest,
                    manifest_filename="manifest.json",
                )["status"] == "PASS"
                complete_exact = True
                if specification["complete_manifest"]:
                    complete_manifest = read_json(
                        fixture_root / "complete_manifest.json"
                    )
                    complete_exact = verify_exact_manifest(
                        fixture_root,
                        complete_manifest,
                        manifest_filename="complete_manifest.json",
                    )["status"] == "PASS"
                summary = read_json(fixture_root / "cohort_summary.json")
                expected_plan = build_work_plan(
                    config,
                    purpose=str(specification["purpose"]),
                )
                contract_pass = (
                    set(value) == expected_execution_fields
                    and value.get("schema_version")
                    == "nursery-corrective-scientific-execution-contract-v1"
                    and value.get("purpose") == specification["purpose"]
                    and value.get("protocol_id")
                    == prefreeze_config["protocol"]["id"]
                    and value.get("config_sha256")
                    == lock.get("prefreeze_config_sha256")
                    and value.get("runner_sha256") == expected_runner_sha
                    and value.get("snapshot_manifest_sha256") is None
                    and value.get("freeze_receipt_sha256") is None
                    and value.get("prequalification_design_lock_sha256")
                    == sha256_file(lock_path)
                    and value.get("input_manifest_sha256")
                    == sha256_file(input_root / "input_manifest.json")
                    and value.get("work_plan_digest") == expected_plan["digest"]
                    and value.get("authorization_digest") is None
                    and value.get("parsed_contract_digest") is None
                    and value.get("scientific_contract_digest")
                    == canonical_digest(scientific_payload)
                    and set(runtime) == expected_runtime_fields
                    and runtime.get("schema_version")
                    == "nursery-corrective-runtime-execution-contract-v1"
                    and runtime.get("scientific_contract_digest")
                    == value.get("scientific_contract_digest")
                    and int(runtime.get("jobs", -1))
                    == int(specification["jobs"])
                    and runtime.get("thread_environment")
                    == prefreeze_config["parallel"]["thread_environment"]
                    and runtime.get("start_method")
                    == prefreeze_config["parallel"]["start_method"]
                    and runtime.get("runtime_contract_digest")
                    == canonical_digest(runtime_payload)
                    and input_exact
                    and adjudication_exact
                    and complete_exact
                    and summary.get("status") == "PASS"
                    and summary.get("purpose") == specification["purpose"]
                    and summary.get("input_manifest_sha256")
                    == value.get("input_manifest_sha256")
                    and summary.get("scientific_contract_digest")
                    == value.get("scientific_contract_digest")
                    and int(summary.get("development_outcome_count", -1)) == 0
                    and int(summary.get("confirmation_outcome_count", -1)) == 0
                )
                fixture_contracts[name] = {
                    "execution_contract_sha256": sha256_file(execution_path),
                    "runtime_contract_sha256": sha256_file(runtime_path),
                    "input_manifest_sha256": sha256_file(
                        input_root / "input_manifest.json"
                    ),
                    "adjudication_manifest_sha256": sha256_file(
                        fixture_root / "adjudication/manifest.json"
                    ),
                    "complete_manifest_sha256": (
                        sha256_file(fixture_root / "complete_manifest.json")
                        if specification["complete_manifest"]
                        else None
                    ),
                    "status": "PASS" if contract_pass else "FAIL",
                }
            except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
                fixture_contracts[name] = {
                    "status": "FAIL",
                    "error": type(error).__name__,
                }
        checks["official_fixture_contracts_use_locked_design"] = all(
            row["status"] == "PASS" for row in fixture_contracts.values()
        )
    return {
        "schema_version": "nursery-corrective-prequalification-lock-verification-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "fixture_contracts": fixture_contracts,
        "lock_sha256": sha256_file(lock_path),
    }


def _require_frozen_module_origins(snapshot: Path) -> None:
    expected = {
        "babyworld_lite.corrective_alignment_v1": "babyworld_lite/corrective_alignment_v1/__init__.py",
        "babyworld_lite.corrective_alignment_v1.protocol": "babyworld_lite/corrective_alignment_v1/protocol.py",
        "babyworld_lite.corrective_alignment_v1.generator": "babyworld_lite/corrective_alignment_v1/generator.py",
        "babyworld_lite.corrective_alignment_v1.learner": "babyworld_lite/corrective_alignment_v1/learner.py",
        "babyworld_lite.corrective_alignment_v1.statistics": "babyworld_lite/corrective_alignment_v1/statistics.py",
        "babyworld_lite.corrective_alignment_v1.parallel": "babyworld_lite/corrective_alignment_v1/parallel.py",
        "babyworld_lite.corrective_alignment_v1.adjudicator": "babyworld_lite/corrective_alignment_v1/adjudicator.py",
        "babyworld_lite.corrective_alignment_v1.integrity": "babyworld_lite/corrective_alignment_v1/integrity.py",
    }
    problems = {}
    for module_name, relative in expected.items():
        module = sys.modules.get(module_name)
        observed = Path(str(getattr(module, "__file__", ""))).resolve()
        required = (snapshot / relative).resolve()
        if module is None or observed != required:
            problems[module_name] = {"observed": str(observed), "required": str(required)}
    if problems:
        raise RuntimeError(f"frozen module origin mismatch: {problems}")


def _live_root() -> Path:
    return SOURCE_ROOT


def _repository_from_snapshot(snapshot: Path) -> Path:
    resolved = snapshot.resolve()
    if resolved.name != "frozen_source_snapshot":
        raise ValueError("snapshot root must be the frozen_source_snapshot directory")
    if resolved.parent.name != "synthetic_corrective_development_launch_package_v1":
        raise ValueError("snapshot package directory mismatch")
    if resolved.parent.parent.name != "output":
        raise ValueError("snapshot must be inside repository output directory")
    return resolved.parents[2]


def _initialize_package(root: Path) -> Path:
    package = root / PACKAGE_RELATIVE
    package.mkdir(parents=True, exist_ok=True)
    registry = package / "outcome_registry.json"
    if not registry.exists():
        write_json(
            registry,
            {
                "schema_version": "nursery-corrective-outcome-registry-v1",
                "development_outcome_count": 0,
                "confirmation_outcome_count": 0,
                "development_output_exists": False,
                "confirmation_output_exists": False,
                "authorization_consumed": False,
            },
        )
    return package


def _micro(root: Path, config: dict) -> dict:
    package = _initialize_package(root)
    lock = _create_prequalification_design_lock(root, package, config)
    lock_verification = _verify_prequalification_design_lock(
        root,
        package,
        config,
        require_fixture_contracts=False,
    )
    if lock_verification["status"] != "PASS":
        raise RuntimeError(
            f"prequalification design lock failed before micro: {lock_verification}"
        )
    inputs = package / "parallel_micro_attempt_4_persisted_inputs"
    jobs_one = package / "parallel_micro_attempt_4_jobs_1"
    jobs_n = package / "parallel_micro_attempt_4_jobs_n"
    start = time.perf_counter()
    prepared = prepare_persisted_inputs(
        inputs,
        config,
        purpose="construction_micro",
    )
    input_generation_wall = time.perf_counter() - start
    start = time.perf_counter()
    run_parallel_from_persisted(
        repository_root=root,
        config_path=root / CONFIG_RELATIVE,
        input_root=inputs,
        output_root=jobs_one,
        config=config,
        purpose="construction_micro",
        jobs=1,
        expected_input_manifest_sha256=str(prepared["input_manifest_sha256"]),
    )
    wall_one = time.perf_counter() - start
    start = time.perf_counter()
    run_parallel_from_persisted(
        repository_root=root,
        config_path=root / CONFIG_RELATIVE,
        input_root=inputs,
        output_root=jobs_n,
        config=config,
        purpose="construction_micro",
        jobs=int(config["parallel"]["frozen_jobs"]),
        expected_input_manifest_sha256=str(prepared["input_manifest_sha256"]),
    )
    wall_n = time.perf_counter() - start
    start = time.perf_counter()
    jobs_n_manifest = manifest_for_tree(
        jobs_n, exclude=["complete_manifest.json"]
    )
    write_json(jobs_n / "complete_manifest.json", jobs_n_manifest)
    if verify_exact_manifest(
        jobs_n,
        jobs_n_manifest,
        manifest_filename="complete_manifest.json",
    )["status"] != "PASS":
        raise RuntimeError("jobs=N micro root manifest failed")
    final_assembly_probe_wall = time.perf_counter() - start
    jobs_one_manifest = manifest_for_tree(
        jobs_one, exclude=["complete_manifest.json"]
    )
    write_json(jobs_one / "complete_manifest.json", jobs_one_manifest)
    if verify_exact_manifest(
        jobs_one,
        jobs_one_manifest,
        manifest_filename="complete_manifest.json",
    )["status"] != "PASS":
        raise RuntimeError("jobs=1 micro root manifest failed")
    equivalence = compare_adjudication_trees(jobs_one, jobs_n)
    write_json(package / "parallel_equivalence_attempt_4.json", equivalence)
    micro_plan = build_work_plan(config, purpose="construction_micro")
    development_plan = build_work_plan(config, purpose="development")
    benchmark = benchmark_report(
        jobs_one_root=jobs_one,
        jobs_n_root=jobs_n,
        input_root=inputs,
        wall_jobs_one=wall_one,
        wall_jobs_n=wall_n,
        input_generation_wall=input_generation_wall,
        final_assembly_probe_wall=final_assembly_probe_wall,
        jobs_n=int(config["parallel"]["frozen_jobs"]),
        development_corpus_count=len(
            config["resolved_registries"]["development"]["corpus"]
        ),
        micro_corpus_count=len(
            config["resolved_registries"]["construction_micro"]["corpus"]
        ),
        development_unit_count=int(development_plan["unit_count"]),
        micro_unit_count=int(micro_plan["unit_count"]),
        minimum_speedup=float(config["parallel"]["minimum_benchmark_speedup"]),
        resource_contingency_multiplier=float(
            config["parallel"]["resource_contingency_multiplier"]
        ),
        maximum_host_ram_fraction=float(
            config["parallel"]["maximum_host_ram_fraction"]
        ),
        maximum_current_free_disk_fraction=float(
            config["parallel"]["maximum_current_free_disk_fraction"]
        ),
        minimum_logical_cpu_count=int(
            config["parallel"]["minimum_logical_cpu_count"]
        ),
    )
    write_json(package / "parallel_benchmark_attempt_4.json", benchmark)
    benchmark_validation = validate_benchmark_report(
        benchmark,
        jobs_one_root=jobs_one,
        jobs_n_root=jobs_n,
        input_root=inputs,
        config=config,
    )
    completion_path = package / "parallel_micro_attempt_4_completion.json"
    require_lstat_absent(completion_path)
    completion_passed = (
        equivalence.get("status") == "PASS"
        and equivalence.get("byte_identical") is True
        and benchmark_validation["status"] == "PASS"
    )
    write_json(
        completion_path,
        {
            "schema_version": "nursery-corrective-parallel-micro-completion-v1",
            "status": "PASS" if completion_passed else "FAIL",
            "prequalification_design_lock_sha256": sha256_file(
                package / "prequalification_design_lock.json"
            ),
            "input_manifest_sha256": sha256_file(inputs / "input_manifest.json"),
            "jobs_1_complete_manifest_sha256": sha256_file(
                jobs_one / "complete_manifest.json"
            ),
            "jobs_n_complete_manifest_sha256": sha256_file(
                jobs_n / "complete_manifest.json"
            ),
            "parallel_equivalence_sha256": sha256_file(
                package / "parallel_equivalence_attempt_4.json"
            ),
            "parallel_benchmark_sha256": sha256_file(
                package / "parallel_benchmark_attempt_4.json"
            ),
            "frozen_jobs": int(config["parallel"]["frozen_jobs"]),
            "scientific_inference_suppressed": True,
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
            "non_replayable": True,
        },
    )
    if not completion_passed:
        raise RuntimeError("parallel micro attempt 4 failed its sealed completion gates")
    return {
        "design_lock_sha256": sha256_file(
            package / "prequalification_design_lock.json"
        ),
        "design_lock_status": lock_verification["status"],
        "equivalence": equivalence,
        "benchmark": benchmark,
        "completion": read_json(completion_path),
    }


def _validate_parallel_micro_attempt_4(
    root: Path,
    package: Path,
    config: dict,
) -> dict:
    inputs = package / "parallel_micro_attempt_4_persisted_inputs"
    jobs_one = package / "parallel_micro_attempt_4_jobs_1"
    jobs_n = package / "parallel_micro_attempt_4_jobs_n"
    equivalence_path = package / "parallel_equivalence_attempt_4.json"
    benchmark_path = package / "parallel_benchmark_attempt_4.json"
    completion_path = package / "parallel_micro_attempt_4_completion.json"
    persisted_equivalence = read_json(equivalence_path)
    recomputed_equivalence = compare_adjudication_trees(jobs_one, jobs_n)
    benchmark = read_json(benchmark_path)
    benchmark_validation = validate_benchmark_report(
        benchmark,
        jobs_one_root=jobs_one,
        jobs_n_root=jobs_n,
        input_root=inputs,
        config=config,
    )
    completion = read_json(completion_path)
    expected_completion_fields = {
        "schema_version",
        "status",
        "prequalification_design_lock_sha256",
        "input_manifest_sha256",
        "jobs_1_complete_manifest_sha256",
        "jobs_n_complete_manifest_sha256",
        "parallel_equivalence_sha256",
        "parallel_benchmark_sha256",
        "frozen_jobs",
        "scientific_inference_suppressed",
        "development_outcome_count",
        "confirmation_outcome_count",
        "non_replayable",
    }
    input_verification = verify_persisted_inputs(inputs)
    checks = {
        "input_manifest": input_verification.get("status") == "PASS"
        and completion.get("input_manifest_sha256")
        == sha256_file(inputs / "input_manifest.json"),
        "equivalence_recomputed_exactly": persisted_equivalence
        == recomputed_equivalence
        and recomputed_equivalence.get("status") == "PASS"
        and recomputed_equivalence.get("byte_identical") is True,
        "benchmark_recomputed": benchmark_validation["status"] == "PASS",
        "completion_schema": set(completion) == expected_completion_fields
        and completion.get("schema_version")
        == "nursery-corrective-parallel-micro-completion-v1",
        "completion_status": completion.get("status") == "PASS"
        and completion.get("scientific_inference_suppressed") is True
        and completion.get("non_replayable") is True,
        "completion_hash_chain": completion.get(
            "prequalification_design_lock_sha256"
        )
        == sha256_file(package / "prequalification_design_lock.json")
        and completion.get("jobs_1_complete_manifest_sha256")
        == sha256_file(jobs_one / "complete_manifest.json")
        and completion.get("jobs_n_complete_manifest_sha256")
        == sha256_file(jobs_n / "complete_manifest.json")
        and completion.get("parallel_equivalence_sha256")
        == sha256_file(equivalence_path)
        and completion.get("parallel_benchmark_sha256")
        == sha256_file(benchmark_path),
        "frozen_jobs": int(completion.get("frozen_jobs", -1))
        == int(config["parallel"]["frozen_jobs"]),
        "zero_outcomes": int(completion.get("development_outcome_count", -1))
        == 0
        and int(completion.get("confirmation_outcome_count", -1)) == 0,
    }
    return {
        "schema_version": "nursery-corrective-parallel-micro-validation-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed_equivalence": recomputed_equivalence,
        "benchmark_validation": benchmark_validation,
        "completion_sha256": sha256_file(completion_path),
    }


def _validate_construction_attempt_receipt(
    package: Path,
    *,
    micro_lock_verification: dict,
    micro_validation: dict,
) -> dict:
    receipt_path = package / "construction_qualification_attempt_consumed.json"
    receipt = read_json(receipt_path)
    expected_fields = {
        "schema_version",
        "status",
        "prequalification_design_lock_sha256",
        "micro_lock_verification_digest",
        "parallel_equivalence_sha256",
        "parallel_benchmark_sha256",
        "parallel_micro_completion_sha256",
        "parallel_micro_validation_digest",
        "scientific_inference_suppressed",
        "development_outcome_count",
        "confirmation_outcome_count",
        "non_replayable",
    }
    checks = {
        "schema": set(receipt) == expected_fields
        and receipt.get("schema_version")
        == "nursery-corrective-construction-attempt-v1",
        "status": receipt.get("status") == "ONE_CONSTRUCTION_ATTEMPT_CONSUMED",
        "design_lock": receipt.get("prequalification_design_lock_sha256")
        == sha256_file(package / "prequalification_design_lock.json"),
        "micro_lock_verification": receipt.get("micro_lock_verification_digest")
        == canonical_digest(micro_lock_verification)
        and micro_lock_verification.get("status") == "PASS",
        "micro_completion": receipt.get("parallel_micro_completion_sha256")
        == sha256_file(package / "parallel_micro_attempt_4_completion.json"),
        "micro_validation": receipt.get("parallel_micro_validation_digest")
        == canonical_digest(micro_validation)
        and micro_validation.get("status") == "PASS",
        "equivalence": receipt.get("parallel_equivalence_sha256")
        == sha256_file(package / "parallel_equivalence_attempt_4.json"),
        "benchmark": receipt.get("parallel_benchmark_sha256")
        == sha256_file(package / "parallel_benchmark_attempt_4.json"),
        "suppressed_nonreplayable": receipt.get("scientific_inference_suppressed")
        is True
        and receipt.get("non_replayable") is True,
        "zero_outcomes": int(receipt.get("development_outcome_count", -1)) == 0
        and int(receipt.get("confirmation_outcome_count", -1)) == 0,
    }
    return {
        "schema_version": "nursery-corrective-construction-attempt-validation-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "receipt_sha256": sha256_file(receipt_path),
    }


def _construction(root: Path, config: dict) -> dict:
    package = _initialize_package(root)
    lock_verification = _verify_prequalification_design_lock(
        root,
        package,
        config,
        require_fixture_contracts=True,
        fixture_contract_names=("micro_jobs_1", "micro_jobs_n"),
    )
    equivalence = read_json(package / "parallel_equivalence_attempt_4.json")
    benchmark = read_json(package / "parallel_benchmark_attempt_4.json")
    micro_validation = _validate_parallel_micro_attempt_4(root, package, config)
    micro_gates = {
        "design_lock": lock_verification.get("status") == "PASS",
        "parallel_equivalence": equivalence.get("status") == "PASS"
        and equivalence.get("byte_identical") is True,
        "parallel_benchmark": benchmark.get("status") == "PASS"
        and all(benchmark.get("gates", {}).values()),
        "sealed_micro_completion": micro_validation.get("status") == "PASS",
    }
    if not all(micro_gates.values()):
        raise RuntimeError(f"construction requires passing locked micro proof: {micro_gates}")
    attempt_receipt_path = package / "construction_qualification_attempt_consumed.json"
    require_lstat_absent(attempt_receipt_path)
    write_json(
        attempt_receipt_path,
        {
            "schema_version": "nursery-corrective-construction-attempt-v1",
            "status": "ONE_CONSTRUCTION_ATTEMPT_CONSUMED",
            "prequalification_design_lock_sha256": sha256_file(
                package / "prequalification_design_lock.json"
            ),
            "micro_lock_verification_digest": canonical_digest(lock_verification),
            "parallel_equivalence_sha256": sha256_file(
                package / "parallel_equivalence_attempt_4.json"
            ),
            "parallel_benchmark_sha256": sha256_file(
                package / "parallel_benchmark_attempt_4.json"
            ),
            "parallel_micro_completion_sha256": sha256_file(
                package / "parallel_micro_attempt_4_completion.json"
            ),
            "parallel_micro_validation_digest": canonical_digest(
                micro_validation
            ),
            "scientific_inference_suppressed": True,
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
            "non_replayable": True,
        },
    )
    cohort_root = package / "construction_qualification"
    summary = execute_cohort(
        repository_root=root,
        config_path=root / CONFIG_RELATIVE,
        output_root=cohort_root,
        config=config,
        purpose="construction_qualification",
        jobs=int(config["parallel"]["frozen_jobs"]),
    )
    mechanism_contract_digest = canonical_digest(
        {
            "purpose": "construction_mechanism",
            "config_sha256": sha256_file(root / CONFIG_RELATIVE),
            "registry": config["resolved_registries"]["construction_mechanism"],
        }
    )
    mechanism_firewall = IdentifierFirewall(
        config,
        purpose="construction_mechanism",
        operation_contract_digest=mechanism_contract_digest,
    )
    mechanism = disagreement_mechanism_qualification(config, mechanism_firewall)
    write_json(package / "mechanism_qualification.json", mechanism)
    mechanism_ledger = mechanism_firewall.ledger()
    write_json(package / "mechanism_qualification_operation_ledger.json", mechanism_ledger)
    mechanism_registry = config["resolved_registries"]["construction_mechanism"]
    mechanism_mutations = (
        "active",
        "semantic_side_zero",
        "evidence_disconnected",
        "within_bag_permuted",
        "old_agreement_gate",
        "semantic_update_disconnected",
    )
    expected_mechanism_operations = []
    corpus_seed = int(mechanism_registry["corpus"][0])
    for case_index, model_seed in enumerate(mechanism_registry["model"]):
        references = [
            {"role": "corpus", "value": corpus_seed},
            {"role": "model", "value": int(model_seed)},
        ]
        for _mutation in mechanism_mutations:
            for operation in ("control", "fit"):
                expected_mechanism_operations.append(
                    {
                        "unit_id": f"mechanism-case-{case_index:02d}",
                        "operation": operation,
                        "references": references,
                    }
                )
        for operation in ("control", "fit"):
            expected_mechanism_operations.append(
                {
                    "unit_id": f"mechanism-null-{case_index:02d}",
                    "operation": operation,
                    "references": references,
                }
            )
    expected_mechanism_operations.append(
        {
            "unit_id": "mechanism-qualification",
            "operation": "inference",
            "references": [
                {"role": "corpus", "value": corpus_seed},
                {
                    "role": "inference",
                    "value": int(mechanism_registry["inference"][0]),
                },
            ],
        }
    )
    mechanism_ledger_verification = _verify_global_ledger(
        mechanism_ledger,
        expected_mechanism_operations,
        purpose="construction_mechanism",
        authorization_digest=None,
        parsed_contract_digest=None,
        operation_contract_digest=mechanism_contract_digest,
    )
    if mechanism_ledger_verification["status"] != "PASS":
        raise RuntimeError(
            f"mechanism operation ledger failed: {mechanism_ledger_verification}"
        )
    write_json(
        package / "mechanism_qualification_ledger_verification.json",
        mechanism_ledger_verification,
    )
    averaged = read_json(cohort_root / "adjudication/model_averages.json")
    mutations = read_json(cohort_root / "adjudication/mechanism_mutations.json")
    audits = read_json(cohort_root / "persisted_inputs/corpus_audits.json")["rows"]
    dependence = read_json(
        cohort_root / "adjudication/present_null_dependence.json"
    )
    decision = construction_qualification_decision(
        averaged=averaged,
        mutation_summary=mutations,
        mechanism_qualification=mechanism,
        dependence_diagnostics=dependence,
        corpus_audits=audits,
        config=config,
    )
    write_json(package / "construction_qualification_decision.json", decision)
    return {"summary": summary, "decision": decision}


def _recompute_construction_qualification_decision(
    package: Path,
    config: dict,
) -> dict:
    cohort_root = package / "construction_qualification"
    return construction_qualification_decision(
        averaged=read_json(cohort_root / "adjudication/model_averages.json"),
        mutation_summary=read_json(
            cohort_root / "adjudication/mechanism_mutations.json"
        ),
        mechanism_qualification=read_json(package / "mechanism_qualification.json"),
        dependence_diagnostics=read_json(
            cohort_root / "adjudication/present_null_dependence.json"
        ),
        corpus_audits=read_json(
            cohort_root / "persisted_inputs/corpus_audits.json"
        )["rows"],
        config=config,
    )


def _frozen_context(snapshot_value: str) -> tuple[Path, Path, dict]:
    snapshot = Path(snapshot_value).resolve()
    if Path(__file__).resolve() != (
        snapshot / "scripts/run_synthetic_corrective_alignment_v1.py"
    ).resolve():
        raise RuntimeError("post-freeze command must run from the selected frozen snapshot")
    _require_frozen_module_origins(snapshot)
    verification = verify_snapshot(snapshot)
    if verification["status"] != "PASS":
        raise RuntimeError(f"snapshot exact verification failed: {verification}")
    repository = _repository_from_snapshot(snapshot)
    config_path = snapshot / CONFIG_RELATIVE
    config = load_config(config_path, repository_root=snapshot)
    require_frozen(config)
    required_environment = {
        str(name): str(value)
        for name, value in config["parallel"]["thread_environment"].items()
    }
    mismatches = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in required_environment.items()
        if os.environ.get(name) != expected
    }
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        mismatches["PYTHONDONTWRITEBYTECODE"] = {
            "expected": "1",
            "observed": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        }
    if mismatches:
        raise RuntimeError(f"post-freeze thread environment mismatch: {mismatches}")
    frozen_environment = read_json(snapshot.parent / "frozen_environment.json")
    if environment_record(repository, config) != frozen_environment:
        raise RuntimeError("post-freeze runtime differs from frozen environment lock")
    _require_frozen_design_transition(repository, snapshot, config)
    return repository, snapshot, config


def _require_frozen_design_transition(
    repository: Path,
    snapshot: Path,
    config: dict,
) -> dict:
    package = snapshot.parent
    recomputed = _verify_prequalification_design_lock(
        snapshot,
        package,
        config,
        require_fixture_contracts=True,
        environment_root=repository,
    )
    persisted = read_json(
        package / "prequalification_transition_verification.json"
    )
    if recomputed.get("status") != "PASS" or recomputed != persisted:
        raise RuntimeError("frozen design transition no longer verifies exactly")
    return recomputed


def _excluded_rehearsal(snapshot_value: str, jobs: int) -> dict:
    repository, snapshot, config = _frozen_context(snapshot_value)
    if int(jobs) != int(config["parallel"]["frozen_jobs"]):
        raise ValueError("excluded rehearsal must use frozen job count")
    output = repository / str(config["paths"]["excluded_rehearsal"])
    receipt_path = repository / str(
        config["paths"]["excluded_rehearsal_attempt_receipt"]
    )
    completion_path = repository / str(
        config["paths"]["excluded_rehearsal_completion"]
    )
    require_lstat_absent(receipt_path)
    require_lstat_absent(completion_path)
    require_lstat_absent(output)
    summary = _execute_excluded_rehearsal_cohort(
        repository_root=snapshot,
        config_path=snapshot / CONFIG_RELATIVE,
        output_root=output,
        config=config,
        jobs=int(jobs),
    )
    completion = {
        "schema_version": "nursery-corrective-excluded-rehearsal-completion-v1",
        "status": "PASS",
        "attempt_receipt_sha256": sha256_file(receipt_path),
        "cohort_summary_sha256": sha256_file(output / "cohort_summary.json"),
        "complete_manifest_sha256": sha256_file(output / "complete_manifest.json"),
        "scientific_decision": summary["scientific_decision"],
        "scientific_inference_suppressed": summary["scientific_decision"]
        == "SCIENTIFIC_INFERENCE_SUPPRESSED",
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }
    write_json(completion_path, completion)
    return {"summary": summary, "completion": completion}


def _recompute_rehearsal(snapshot_value: str) -> dict:
    repository, snapshot, config = _frozen_context(snapshot_value)
    package = repository / PACKAGE_RELATIVE
    rehearsal = package / "excluded_rehearsal"
    output = package / "independent_recompute"
    require_lstat_absent(output)
    require_lstat_absent(package / "independent_recompute_comparison.json")
    _consume_lifecycle_attempt(
        package,
        snapshot,
        config,
        "independent_recompute",
    )
    completion = read_json(package / "excluded_rehearsal_completion.json")
    if (
        completion.get("status") != "PASS"
        or completion.get("scientific_inference_suppressed") is not True
        or int(completion.get("development_outcome_count", -1)) != 0
        or int(completion.get("confirmation_outcome_count", -1)) != 0
    ):
        raise RuntimeError("excluded rehearsal completion seal is invalid")
    result = _recompute_adjudication_from_rehearsal(
        repository_root=snapshot,
        config_path=snapshot / CONFIG_RELATIVE,
        rehearsal_root=rehearsal,
        output_root=output,
        config=config,
    )
    comparison = compare_recompute(
        rehearsal,
        output,
        package / "independent_recompute_comparison.json",
    )
    return {"result": result, "comparison": comparison}


def _no_symlink_ancestry(path: Path, *, stop: Path) -> None:
    stop = stop.resolve()
    current = path.absolute()
    chain = []
    while current != stop:
        chain.append(current)
        if current.parent == current:
            raise RuntimeError(f"path is not beneath repository root: {path}")
        current = current.parent
    for value in reversed(chain):
        try:
            metadata = value.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"symlink ancestry forbidden: {value}")


def _strict_development_preflight(
    *,
    snapshot: Path,
    authorization_path: Path,
    output_root: Path,
    jobs: int,
):
    actual_argv = list(sys.argv[1:])
    repository = _repository_from_snapshot(snapshot)
    snapshot_verification = verify_snapshot(snapshot)
    if snapshot_verification["status"] != "PASS":
        raise PermissionError(f"receipt-bound snapshot verification failed: {snapshot_verification}")
    expected_runner = snapshot / "scripts/run_synthetic_corrective_alignment_v1.py"
    if Path(__file__).resolve() != expected_runner.resolve():
        raise PermissionError("development must execute the exact frozen runner path")
    _require_frozen_module_origins(snapshot)
    config = load_config(snapshot / CONFIG_RELATIVE, repository_root=snapshot)
    require_frozen(config)
    package = snapshot.parent
    try:
        _require_frozen_design_transition(repository, snapshot, config)
    except RuntimeError as error:
        raise PermissionError(str(error)) from error
    expected_authorization = repository / str(config["paths"]["authorization"])
    if authorization_path.resolve() != expected_authorization.resolve():
        raise PermissionError("actual authorization path differs from frozen config")
    authorization = read_json(authorization_path)
    verification = verify_authorization(authorization)
    if verification["status"] != "PASS":
        raise PermissionError(f"authorization schema/digest failed: {verification}")

    expected_paths = {
        "resolved_repository_root": repository,
        "resolved_snapshot_root": snapshot,
        "resolved_authorization_path": expected_authorization,
        "resolved_output_root": repository / str(config["paths"]["development_output"]),
        "resolved_staging_root": repository / str(config["paths"]["development_staging"]),
        "resolved_inner_staging_root": repository
        / str(config["paths"]["development_inner_staging"]),
        "resolved_publication_seal_path": repository
        / str(config["paths"]["development_publication_seal"]),
        "resolved_claim_path": repository / str(config["paths"]["authorization_claim"]),
        "resolved_capability_receipt_path": repository
        / str(config["paths"]["authorization_capability_receipt"]),
    }
    for field, expected in expected_paths.items():
        if Path(str(authorization[field])).resolve() != expected.resolve():
            raise PermissionError(f"authorization path differs from frozen config: {field}")
    if output_root.resolve() != expected_paths["resolved_output_root"].resolve():
        raise PermissionError("actual parsed path differs from authorization: resolved_output_root")
    if Path.cwd().resolve() != repository.resolve() or Path(
        str(authorization["required_cwd"])
    ).resolve() != repository.resolve():
        raise PermissionError("working directory differs from frozen repository root")
    if int(jobs) != int(config["parallel"]["frozen_jobs"]) or int(jobs) != int(
        authorization["required_jobs"]
    ):
        raise PermissionError("worker count differs from frozen config/authorization")
    if actual_argv != list(config["commands"]["development_argv"]) or actual_argv != list(
        authorization["exact_argv"]
    ):
        raise PermissionError("actual parsed argv differs from frozen config/authorization")
    if authorization["exact_shell_command"] != str(
        config["commands"]["development_one_shot"]
    ):
        raise PermissionError("shell command differs from frozen config")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise PermissionError("PYTHONDONTWRITEBYTECODE=1 required")
    frozen_thread_environment = dict(config["parallel"]["thread_environment"])
    if authorization["required_thread_environment"] != frozen_thread_environment:
        raise PermissionError("authorization thread map differs from frozen config")
    for name, expected in frozen_thread_environment.items():
        if os.environ.get(str(name)) != str(expected):
            raise PermissionError(f"thread environment mismatch: {name}")

    frozen_environment = read_json(package / "frozen_environment.json")
    current_environment = environment_record(repository, config)
    if current_environment != frozen_environment:
        raise PermissionError("runtime environment differs from frozen environment lock")
    if Path(sys.executable).resolve() != Path(
        str(frozen_environment["python_executable_resolved"])
    ).resolve():
        raise PermissionError("Python executable differs from frozen environment")
    if (
        authorization["required_python_executable"]
        != frozen_environment["python_executable"]
        or authorization["required_python_sha256"]
        != frozen_environment["python_sha256"]
    ):
        raise PermissionError("authorization Python binding differs from environment lock")
    if Path(str(authorization["required_runner_path"])).resolve() != expected_runner.resolve():
        raise PermissionError("authorization runner path differs from frozen runner")
    anchored_hashes = {
        "required_runner_sha256": expected_runner,
        "required_config_sha256": snapshot / CONFIG_RELATIVE,
        "required_snapshot_manifest_sha256": snapshot / "snapshot_manifest.json",
        "required_freeze_receipt_sha256": package / "freeze_receipt.json",
        "required_environment_sha256": package / "frozen_environment.json",
        "required_registries_sha256": package / "frozen_seed_registries.json",
        "required_package_core_manifest_sha256": package / "package_core_manifest.json",
        "required_package_terminal_sha256": package
        / "launch_seal/package_terminal.json",
        "required_excluded_rehearsal_sha256": package
        / "excluded_rehearsal/cohort_summary.json",
        "required_recompute_comparison_sha256": package
        / "independent_recompute_comparison.json",
        "required_official_tests_sha256": package / "official_test_execution_report.json",
        "required_benchmark_sha256": package / "parallel_benchmark_attempt_4.json",
        "required_preservation_proof_sha256": package
        / "preservation/preservation_proof.json",
    }
    for field, path in anchored_hashes.items():
        if sha256_file(path) != authorization[field]:
            raise PermissionError(f"authorization-bound artifact changed: {field}")

    for value in (
        expected_paths["resolved_claim_path"],
        expected_paths["resolved_capability_receipt_path"],
        expected_paths["resolved_staging_root"],
        expected_paths["resolved_inner_staging_root"],
        expected_paths["resolved_output_root"],
        expected_paths["resolved_publication_seal_path"],
    ):
        require_lstat_absent(value)
        _no_symlink_ancestry(value, stop=repository)

    core_manifest = read_json(package / "package_core_manifest.json")
    core = verify_exact_manifest(
        package,
        core_manifest,
        manifest_filename="package_core_manifest.json",
        allowed_extra_files=[
            "launch_seal/package_terminal.json",
            "launch_seal/CORRECTIVE_DEVELOPMENT_LAUNCH_READY.json",
            "launch_seal/complete_file_manifest.json",
        ],
    )
    if core["status"] != "PASS":
        raise PermissionError(f"package core manifest failed: {core}")
    complete_manifest = read_json(
        package / "launch_seal/complete_file_manifest.json"
    )
    complete = verify_exact_manifest(
        package,
        complete_manifest,
        manifest_filename="launch_seal/complete_file_manifest.json",
    )
    if complete["status"] != "PASS":
        raise PermissionError(f"package complete manifest failed: {complete}")
    terminal = read_json(package / "launch_seal/package_terminal.json")
    if terminal.get("terminal_state") != "CORRECTIVE_DEVELOPMENT_LAUNCH_READY":
        raise PermissionError("package terminal is not launch-ready")
    registry = read_json(package / "outcome_registry.json")
    if (
        int(registry.get("development_outcome_count", -1)) != 0
        or int(registry.get("confirmation_outcome_count", -1)) != 0
        or registry.get("authorization_consumed") is not False
    ):
        raise PermissionError("outcome registry is not pristine zero state")
    if config["resolved_registries"]["development"] != authorization[
        "development_registry"
    ]:
        raise PermissionError("development registry differs from authorization")
    if canonical_digest(config["resolved_registries"]["confirmation_reserve"]) != authorization[
        "confirmation_reserve_digest"
    ]:
        raise PermissionError("confirmation reserve digest mismatch")
    parsed_contract = {
        "snapshot": str(snapshot.resolve()),
        "authorization": str(authorization_path.resolve()),
        "output": str(output_root.resolve()),
        "staging": str(expected_paths["resolved_staging_root"].resolve()),
        "inner_staging": str(expected_paths["resolved_inner_staging_root"].resolve()),
        "publication_seal": str(
            expected_paths["resolved_publication_seal_path"].resolve()
        ),
        "claim": str(expected_paths["resolved_claim_path"].resolve()),
        "capability_receipt": str(
            expected_paths["resolved_capability_receipt_path"].resolve()
        ),
        "jobs": int(jobs),
        "thread_environment": frozen_thread_environment,
        "argv": actual_argv,
        "snapshot_verification_digest": canonical_digest(snapshot_verification),
    }
    parsed_contract_digest = canonical_digest(parsed_contract)
    proof = _issue_development_preflight_proof(
        authorization_digest=str(authorization["authorization_digest"]),
        authorization_path=authorization_path,
        authorization_sha256=sha256_file(authorization_path),
        snapshot_root=snapshot,
        snapshot_manifest_sha256=sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        parsed_contract_digest=parsed_contract_digest,
        resolved_output_root=output_root,
        resolved_staging_root=expected_paths["resolved_staging_root"],
        resolved_inner_staging_root=expected_paths[
            "resolved_inner_staging_root"
        ],
        claim_path=expected_paths["resolved_claim_path"],
        issuance_receipt_path=expected_paths[
            "resolved_capability_receipt_path"
        ],
        required_jobs=int(jobs),
        argv_digest=canonical_digest(actual_argv),
        environment_digest=canonical_digest(current_environment),
    )
    return repository, config, authorization, parsed_contract, proof


_DEVELOPMENT_PUBLICATION_FACTORY_SEAL = object()


class _DevelopmentPublicationProof:
    __slots__ = (
        "capability_identity_digest",
        "authorization_digest",
        "parsed_contract_digest",
        "staging_root",
        "output_root",
        "publication_path",
        "complete_manifest_sha256",
        "cohort_summary_sha256",
        "_identity_digest",
        "_seal",
        "_frozen",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("development publication proofs are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, _seal=None, **values):
        if _seal is not _DEVELOPMENT_PUBLICATION_FACTORY_SEAL:
            raise PermissionError("publication proof requires verified parent authority")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        identity = {
            name: getattr(self, name)
            for name in self.__slots__
            if not name.startswith("_")
        }
        object.__setattr__(self, "_identity_digest", canonical_digest(identity))
        object.__setattr__(self, "_seal", _seal)
        object.__setattr__(self, "_frozen", True)


def _issue_development_publication_proof(
    *,
    staging: Path,
    output_root: Path,
    authorization: dict,
    summary: dict,
    capability: AuthorizationCapability,
) -> _DevelopmentPublicationProof:
    staging = staging.resolve()
    output_root = output_root.resolve()
    verify_development_capability(
        capability,
        required_scope="parent",
        resolved_output_root=output_root,
        resolved_staging_root=staging,
        required_jobs=int(authorization.get("required_jobs", -1)),
    )
    claim = read_json(capability.claim_path)
    persisted_authorization = read_json(claim["resolved_authorization_path"])
    persisted_summary = read_json(staging / "cohort_summary.json")
    if authorization != persisted_authorization:
        raise PermissionError("publication authorization differs from consumed authority")
    if summary != persisted_summary:
        raise PermissionError("publication summary differs from persisted cohort summary")
    if (
        authorization.get("authorization_digest")
        != capability.authorization_digest
        or claim.get("parsed_contract_digest")
        != capability.parsed_contract_digest
        or Path(str(authorization.get("resolved_output_root"))).resolve()
        != output_root
        or Path(str(authorization.get("resolved_staging_root"))).resolve()
        != staging
    ):
        raise PermissionError("publication paths or contract differ from capability")
    publication_path = Path(
        str(authorization["resolved_publication_seal_path"])
    ).resolve()
    if publication_path != output_root / "SCIENTIFIC_OUTCOME.json":
        raise PermissionError("publication seal path differs from authorized output")
    complete_manifest = read_json(staging / "complete_manifest.json")
    verification = verify_exact_manifest(
        staging,
        complete_manifest,
        manifest_filename="complete_manifest.json",
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"staged scientific output failed exact manifest: {verification}")
    require_lstat_absent(output_root)
    require_lstat_absent(publication_path)
    return _DevelopmentPublicationProof(
        capability_identity_digest=capability._identity_digest,
        authorization_digest=capability.authorization_digest,
        parsed_contract_digest=capability.parsed_contract_digest,
        staging_root=str(staging),
        output_root=str(output_root),
        publication_path=str(publication_path),
        complete_manifest_sha256=sha256_file(staging / "complete_manifest.json"),
        cohort_summary_sha256=sha256_file(staging / "cohort_summary.json"),
        _seal=_DEVELOPMENT_PUBLICATION_FACTORY_SEAL,
    )


def _verify_development_publication_proof(
    proof: _DevelopmentPublicationProof,
    *,
    capability: AuthorizationCapability,
    staging: Path,
    output_root: Path,
) -> None:
    if (
        not isinstance(proof, _DevelopmentPublicationProof)
        or proof._seal is not _DEVELOPMENT_PUBLICATION_FACTORY_SEAL
        or proof._frozen is not True
    ):
        raise PermissionError("scientific publication requires an issued proof")
    identity = {
        name: getattr(proof, name)
        for name in proof.__slots__
        if not name.startswith("_")
    }
    if canonical_digest(identity) != proof._identity_digest:
        raise PermissionError("publication proof identity changed")
    verify_development_capability(
        capability,
        required_scope="parent",
        resolved_output_root=output_root,
        resolved_staging_root=staging,
    )
    if (
        proof.capability_identity_digest != capability._identity_digest
        or proof.authorization_digest != capability.authorization_digest
        or proof.parsed_contract_digest != capability.parsed_contract_digest
        or Path(proof.staging_root) != staging.resolve()
        or Path(proof.output_root) != output_root.resolve()
        or Path(proof.publication_path)
        != output_root.resolve() / "SCIENTIFIC_OUTCOME.json"
        or sha256_file(staging / "complete_manifest.json")
        != proof.complete_manifest_sha256
        or sha256_file(staging / "cohort_summary.json")
        != proof.cohort_summary_sha256
    ):
        raise PermissionError("publication proof no longer matches staged evidence")


def _publish_development_staging(
    *,
    staging: Path,
    output_root: Path,
    authorization: dict,
    summary: dict,
    capability: AuthorizationCapability,
    publication_proof: _DevelopmentPublicationProof,
) -> dict:
    _verify_development_publication_proof(
        publication_proof,
        capability=capability,
        staging=staging,
        output_root=output_root,
    )
    if summary != read_json(staging / "cohort_summary.json"):
        raise PermissionError("publication summary changed after proof issuance")
    complete_manifest = read_json(staging / "complete_manifest.json")
    verification = verify_exact_manifest(
        staging,
        complete_manifest,
        manifest_filename="complete_manifest.json",
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"staged scientific output failed exact manifest: {verification}")
    publication_path = Path(str(authorization["resolved_publication_seal_path"]))
    expected_publication_path = output_root / "SCIENTIFIC_OUTCOME.json"
    if publication_path != expected_publication_path:
        raise RuntimeError("publication seal path changed after preflight")
    staged_publication_path = staging / "SCIENTIFIC_OUTCOME.json"
    publication = {
        "schema_version": "nursery-corrective-scientific-publication-v1",
        "status": "COMPLETE",
        "publication_state": "EFFECTIVE_ONLY_AT_AUTHORIZED_FINAL_ROOT",
        "authorized_final_root": str(output_root),
        "atomic_publication_operation": "single_directory_rename",
        "scientific_outcome": True,
        "scientific_decision": summary["candidate_decision"],
        "authorization_digest": authorization["authorization_digest"],
        "parsed_contract_digest": capability.parsed_contract_digest,
        "scientific_contract_digest": summary["scientific_contract_digest"],
        "prepublication_complete_manifest_sha256": sha256_file(
            staging / "complete_manifest.json"
        ),
        "cohort_summary_sha256": sha256_file(staging / "cohort_summary.json"),
        "development_outcome_count": 1,
        "confirmation_outcome_count": 0,
        "confirmation_authorized": False,
    }
    write_json(staged_publication_path, publication)
    published_manifest_path = staging / "published_complete_manifest.json"
    published_manifest = manifest_for_tree(
        staging,
        exclude=["published_complete_manifest.json"],
    )
    write_json(published_manifest_path, published_manifest)
    published_verification = verify_exact_manifest(
        staging,
        published_manifest,
        manifest_filename="published_complete_manifest.json",
    )
    if published_verification["status"] != "PASS":
        raise RuntimeError(
            f"staged published output failed exact manifest: {published_verification}"
        )
    require_lstat_absent(output_root)
    atomic_rename_directory_noreplace(staging, output_root)
    if not publication_path.is_file() or publication_path.is_symlink():
        raise RuntimeError("atomic publication seal missing after directory rename")
    return publication


def _run_development(
    snapshot_value: str,
    authorization_value: str,
    output_value: str,
    jobs: int,
) -> dict:
    snapshot = Path(snapshot_value).resolve()
    authorization_path = Path(authorization_value).resolve()
    output_root = Path(output_value).resolve()
    repository, config, authorization, parsed_contract, preflight_proof = _strict_development_preflight(
        snapshot=snapshot,
        authorization_path=authorization_path,
        output_root=output_root,
        jobs=jobs,
    )
    claim_path = Path(str(authorization["resolved_claim_path"]))
    parsed_contract_digest = canonical_digest(parsed_contract)
    write_json(
        claim_path,
        {
            "schema_version": "nursery-corrective-authorization-consumption-v1",
            "status": "ATTEMPT_CONSUMED",
            "authorization_digest": authorization["authorization_digest"],
            "resolved_authorization_path": str(authorization_path),
            "authorization_sha256": sha256_file(authorization_path),
            "resolved_snapshot_root": str(snapshot),
            "parsed_contract_digest": parsed_contract_digest,
            "resolved_output_root": str(output_root),
            "resolved_staging_root": str(authorization["resolved_staging_root"]),
            "resolved_inner_staging_root": str(
                authorization["resolved_inner_staging_root"]
            ),
            "resolved_capability_receipt_path": str(
                authorization["resolved_capability_receipt_path"]
            ),
            "required_jobs": int(jobs),
            "non_replayable": True,
            "created_before_any_guarded_development_operation": True,
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
        },
    )
    capability = issue_development_capability(
        authorization_digest=str(authorization["authorization_digest"]),
        parsed_contract_digest=parsed_contract_digest,
        claim_path=claim_path,
        claim_sha256=sha256_file(claim_path),
        resolved_output_root=output_root,
        resolved_staging_root=authorization["resolved_staging_root"],
        resolved_inner_staging_root=authorization["resolved_inner_staging_root"],
        issuance_receipt_path=authorization["resolved_capability_receipt_path"],
        required_jobs=int(jobs),
        preflight_proof=preflight_proof,
    )
    staging = Path(str(authorization["resolved_staging_root"]))
    summary = _execute_development_cohort(
        repository_root=snapshot,
        config_path=snapshot / CONFIG_RELATIVE,
        output_root=staging,
        config=config,
        jobs=int(jobs),
        capability=capability,
    )
    publication_proof = _issue_development_publication_proof(
        staging=staging,
        output_root=output_root,
        authorization=authorization,
        summary=summary,
        capability=capability,
    )
    return _publish_development_staging(
        staging=staging,
        output_root=output_root,
        authorization=authorization,
        summary=summary,
        capability=capability,
        publication_proof=publication_proof,
    )


def _all_int_identifiers(value: object) -> set[int]:
    output: set[int] = set()
    if isinstance(value, dict):
        for child in value.values():
            output |= _all_int_identifiers(child)
    elif isinstance(value, list):
        for child in value:
            output |= _all_int_identifiers(child)
    elif isinstance(value, int) and not isinstance(value, bool):
        output.add(int(value))
    return output


def _design_contract_validation(
    root: Path,
    package: Path,
    snapshot: Path,
    config: dict,
    construction: dict,
) -> dict:
    construction_gates = construction["gates"]
    recomputed_construction = _recompute_construction_qualification_decision(
        package, config
    )
    micro_validation = _validate_parallel_micro_attempt_4(root, package, config)
    required_controls = {
        "synchronized",
        "shuffled",
        "shift_minus",
        "shift_plus",
        "absent",
        "uninformative",
        "corrupted",
        "oracle_alignment",
        "exact_window",
    }
    required_mutations = {
        "semantic_side_zero",
        "evidence_disconnected",
        "within_bag_permuted",
        "old_agreement_gate",
        "semantic_update_disconnected",
        "null_side_zero",
        "null_update_disconnected",
        "null_head_ablation",
    }
    heterogeneity_gate_names = {
        "action_geometry_varies_by_corpus",
        "episode_geometry_varies_by_corpus",
        "action_geometry_distance_structure_varies",
        "action_geometry_is_separated_and_shared_across_train_eval",
        "heldout_splits_vary_by_corpus",
        "repetition_patterns_vary_by_corpus",
        "ambiguity_strata_vary_by_corpus",
        "realized_candidate_counts_vary_by_corpus",
        "realized_grounded_rates_vary_by_corpus",
        "realized_ambiguity_varies_by_corpus",
        "realized_lag_varies_by_corpus",
        "realized_visibility_varies_by_corpus",
        "realized_side_informativeness_varies_by_corpus",
        "all_frozen_factors_vary",
    }
    brief = read_json(
        snapshot / "docs/synthetic_corrective_alignment_v1_brief_trace.json"
    )
    brief_source = Path(str(brief["source_path"]))
    expected_brief_requirements = {
        "temporal_episode_bags_and_null",
        "cross_occurrence_learning",
        "training_only_side_information",
        "matched_controls",
        "independent_transfer",
        "confirmation_reserve_uninspected",
        "no_new_external_dataset",
    }
    sources = read_json(
        snapshot / "docs/synthetic_corrective_alignment_v1_primary_sources.json"
    )
    threshold_text = (
        snapshot / "docs/synthetic_corrective_alignment_v1_threshold_rationale.md"
    ).read_text(encoding="utf-8")
    frozen_registry = read_json(package / "frozen_seed_registries.json")
    registries = config["resolved_registries"]
    registry_sets = {
        name: {int(seed) for values in registry.values() for seed in values}
        for name, registry in registries.items()
    }
    active_identifiers = set().union(*registry_sets.values())
    prior = read_json(
        snapshot / str(config["registries"]["canonical_prior_registry"])
    )
    prior_identifiers = _all_int_identifiers(prior.get("registries", {}))
    family_low, family_high = map(int, config["registries"]["namespace_family"])
    quarantine = [
        tuple(map(int, row))
        for row in config["registries"]["permanently_quarantined_ranges"]
    ]
    persisted_transition = read_json(
        package / "prequalification_transition_verification.json"
    )
    recomputed_transition = _verify_prequalification_design_lock(
        snapshot,
        package,
        config,
        require_fixture_contracts=True,
        environment_root=root,
    )
    micro_lock_verification = _verify_prequalification_design_lock(
        snapshot,
        package,
        config,
        require_fixture_contracts=True,
        fixture_contract_names=("micro_jobs_1", "micro_jobs_n"),
        environment_root=root,
    )
    prefreeze_manifest = read_json(package / "prefreeze_evidence_manifest.json")
    prefreeze_rows = {
        str(row["path"]): row for row in prefreeze_manifest.get("files", [])
    }
    prospective_artifacts = {
        "prequalification_design_lock.json": package
        / "prequalification_design_lock.json",
        "construction_qualification_attempt_consumed.json": package
        / "construction_qualification_attempt_consumed.json",
        "prequalification_transition_verification.json": package
        / "prequalification_transition_verification.json",
    }
    prospective_rows_bound = all(
        relative in prefreeze_rows
        and prefreeze_rows[relative].get("sha256") == sha256_file(path)
        for relative, path in prospective_artifacts.items()
    )
    construction_attempt = read_json(
        package / "construction_qualification_attempt_consumed.json"
    )
    construction_attempt_validation = _validate_construction_attempt_receipt(
        package,
        micro_lock_verification=micro_lock_verification,
        micro_validation=micro_validation,
    )
    prospective_lock_pass = (
        persisted_transition.get("status") == "PASS"
        and recomputed_transition == persisted_transition
        and prospective_rows_bound
        and read_json(package / "frozen_environment.json")
        == read_json(package / "prequalification_design_lock.json").get(
            "qualification_environment"
        )
        and construction_attempt_validation.get("status") == "PASS"
    )
    checks = {
        "construction_decision_recomputed_exactly": construction
        == recomputed_construction,
        "parallel_micro_completion_recomputed": micro_validation["status"]
        == "PASS",
        "brief_source_exact_hash": brief_source.is_file()
        and sha256_file(brief_source) == brief["source_sha256"]
        and brief_source.stat().st_size == int(brief["source_bytes"]),
        "brief_requirement_map_complete": set(brief["mapped_requirements"])
        == expected_brief_requirements
        and all(brief["mapped_requirements"].values()),
        "claim_scope_narrow": config["protocol"]["scientific_claim"]
        == "synthetic_training_only_event_cues_can_correct_transferable_lexical_action_meanings"
        and config["protocol"]["infant_learning_claim_authorized"] is False
        and config["protocol"]["ecological_validity_claim_authorized"] is False,
        "primary_sources_recorded": sources["schema_version"]
        == "nursery-corrective-primary-sources-v1"
        and len(sources["sources"]) >= 12
        and all(str(row.get("url", "")).startswith("http") for row in sources["sources"]),
        "threshold_rationale_frozen": "+0.10" in threshold_text
        and "headroom" in threshold_text.lower(),
        "registry_snapshot_exact": frozen_registry == registry_snapshot(config),
        "registries_pairwise_disjoint": all(
            not registry_sets[left] & registry_sets[right]
            for index, left in enumerate(registry_sets)
            for right in list(registry_sets)[index + 1 :]
        ),
        "registry_namespace_and_quarantine": all(
            family_low <= seed <= family_high
            and not any(low <= seed <= high for low, high in quarantine)
            for seed in active_identifiers
        ),
        "registry_disjoint_from_prior": not active_identifiers & prior_identifiers,
        "headroom_predeclared_and_passed": construction_gates[
            "headroom_every_negative_control"
        ]
        and config["qualification_gates"]["maximum_negative_control_top1"] < 1.0,
        "corrective_path_active_and_mutated": construction_gates[
            "prespecified_disagreement_mutations"
        ]
        and construction_gates["all_informativeness_and_manipulation_gates"]
        and set(config["design"]["mechanism_mutations"]) == required_mutations,
        "acquisition_transfer_endpoints_frozen": config["analysis"][
            "co_primary_endpoints"
        ]
        == ["lexical_acquisition_top1", "heldout_composition_top1"]
        and {"multiclass_log_loss", "multiclass_brier"}
        <= set(config["analysis"]["secondary_metrics"])
        and config["analysis"]["secondary_multiplicity"]
        == "descriptive_only_no_p_values_or_confirmatory_claims",
        "temporal_event_null_alignment": int(
            config["design"]["candidate_event_count"][0]
        )
        >= 2
        and config["learner"]["null_prior"] > 0
        and {"shift_minus", "shift_plus", "exact_window"}
        <= set(config["design"]["conditions"])
        and construction_gates["all_informativeness_and_manipulation_gates"],
        "matched_causal_controls": set(config["design"]["conditions"])
        == required_controls
        and construction["corpus_audit_checks"][
            "matched_shuffled_evidence_marginal"
        ],
        "test_time_side_withholding": construction_gates[
            "test_time_side_withheld"
        ]
        and construction_gates["evaluation_schema_and_side_withholding"]
        and config["design"]["evaluation"]["side_modality_withheld"] is True,
        "independent_generalization": construction_gates[
            "independent_train_eval_provenance"
        ]
        and construction_gates[
            "heldout_compositions_absent_from_all_training_events"
        ]
        and construction_gates["zero_exposure_leakage_control"],
        "corpus_heterogeneity": all(
            construction_gates[name] for name in heterogeneity_gate_names
        ),
        "model_replicates_meaningful": construction_gates[
            "model_training_stochasticity_is_numerically_relevant"
        ]
        and int(config["analysis"]["paired_model_replicates"]) >= 3
        and config["analysis"]["model_replicate_handling"]
        == "average_within_corpus_in_frozen_seed_order",
        "presence_null_causally_distinct": construction_gates[
            "presence_null_effects_not_identical"
        ]
        and {
            "semantic_update_disconnected",
            "null_side_zero",
            "null_update_disconnected",
            "null_head_ablation",
        }
        <= set(config["design"]["mechanism_mutations"]),
        "statistics_and_iut_frozen": config["analysis"]["independent_unit"]
        == "corpus_seed"
        and config["analysis"]["intersection_union_rule"]
        == "both_co_primary_estimands_must_pass_every_gate"
        and config["analysis"]["primary_uncertainty"]
        == "one_sided_student_t_over_corpus_means"
        and float(config["analysis"]["minimum_practical_effect"])
        == float(config["analysis"]["primary_lower_bound_must_exceed"])
        and config["analysis"]["planning"]["adaptive_extension"] is False,
        "sample_size_and_seeds_frozen": len(registries["development"]["corpus"])
        == int(config["analysis"]["development_corpus_count"])
        == 100
        and len(registries["development"]["model"])
        == int(config["analysis"]["paired_model_replicates"]),
        "prospective_prequalification_lock": prospective_lock_pass,
        "no_post_outcome_adaptation": config["protocol"]["status"] == "frozen"
        and prospective_lock_pass
        and construction_gates["fixture_cohort_preallocated_all_retained"]
        and construction["scientific_inference_suppressed"] is True
        and config["protocol"]["development_execution_authorized"] is False
        and config["protocol"]["confirmation_authorized"] is False,
        "parallel_contract_frozen": config["parallel"]["start_method"] == "spawn"
        and config["parallel"]["worker_isolated_shards"] is True
        and config["parallel"]["worker_isolated_tmp"] is True
        and config["parallel"]["atomic_final_publish"] is True
        and config["parallel"]["require_jobs_1_jobs_n_byte_identity"] is True,
        "confirmation_unavailable": config["firewalls"][
            "confirmation_command_exists"
        ]
        is False
        and config["firewalls"]["confirmation_execution_supported"] is False,
    }
    categories = {
        "original_brief": [
            "brief_source_exact_hash",
            "brief_requirement_map_complete",
            "claim_scope_narrow",
            "primary_sources_recorded",
            "threshold_rationale_frozen",
        ],
        "registries": [
            "registry_snapshot_exact",
            "registries_pairwise_disjoint",
            "registry_namespace_and_quarantine",
            "registry_disjoint_from_prior",
            "confirmation_unavailable",
        ],
        "construct": ["headroom_predeclared_and_passed"],
        "corrective_path": ["corrective_path_active_and_mutated"],
        "acquisition": ["acquisition_transfer_endpoints_frozen"],
        "temporal": ["temporal_event_null_alignment"],
        "controls": ["matched_causal_controls"],
        "withholding": ["test_time_side_withholding"],
        "generalization": ["independent_generalization"],
        "heterogeneity": ["corpus_heterogeneity"],
        "model_variation": ["model_replicates_meaningful"],
        "presence_null": ["presence_null_causally_distinct"],
        "statistics": [
            "statistics_and_iut_frozen",
            "sample_size_and_seeds_frozen",
            "no_post_outcome_adaptation",
        ],
        "prospective_freeze": ["prospective_prequalification_lock"],
        "parallel": ["parallel_contract_frozen"],
    }
    category_status = {
        name: "PASS" if all(checks[field] for field in fields) else "FAIL"
        for name, fields in categories.items()
    }
    return {
        "schema_version": "nursery-corrective-design-contract-validation-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "categories": {
            name: {"checks": fields, "status": category_status[name]}
            for name, fields in categories.items()
        },
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }


def _rehearsal_contract_validation(
    package: Path,
    snapshot: Path,
    config: dict,
) -> dict:
    rehearsal = package / "excluded_rehearsal"
    receipt_path = package / "excluded_rehearsal_attempt_consumed.json"
    completion_path = package / "excluded_rehearsal_completion.json"
    receipt = read_json(receipt_path)
    completion = read_json(completion_path)
    summary = read_json(rehearsal / "cohort_summary.json")
    runtime_contract = read_json(rehearsal / "runtime/runtime_contract.json")
    execution_contract = read_json(
        rehearsal / "adjudication/execution_contract.json"
    )
    rehearsal_manifest = read_json(rehearsal / "complete_manifest.json")
    recompute = package / "independent_recompute"
    recompute_manifest = read_json(recompute / "complete_manifest.json")
    comparison_path = package / "independent_recompute_comparison.json"
    comparison = read_json(comparison_path)
    recomputed_comparison = recompute_comparison_value(rehearsal, recompute)
    expected_receipt = {
        "schema_version": "nursery-corrective-excluded-rehearsal-attempt-v1",
        "status": "ONE_EXCLUDED_ATTEMPT_CONSUMED",
        "snapshot_root": str(snapshot),
        "snapshot_manifest_sha256": sha256_file(snapshot / "snapshot_manifest.json"),
        "freeze_receipt_sha256": sha256_file(package / "freeze_receipt.json"),
        "output_root": str(rehearsal.resolve()),
        "jobs": int(config["parallel"]["frozen_jobs"]),
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "non_replayable": True,
    }
    observed_rehearsal_summaries = []
    for path in package.rglob("cohort_summary.json"):
        value = read_json(path)
        if value.get("purpose") == "excluded_rehearsal":
            observed_rehearsal_summaries.append(path.relative_to(package).as_posix())
    observed_rehearsal_summaries.sort()
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
    expected_summary_fields = {
        "schema_version",
        "status",
        "purpose",
        "integrity_status",
        "scientific_decision",
        "candidate_decision",
        "publication_state",
        "development_outcome_count",
        "confirmation_outcome_count",
        "confirmation_authorized",
        "adjudication_manifest_sha256",
        "input_manifest_sha256",
        "scientific_contract_digest",
        "runtime_contract_digest",
    }
    expected_execution_fields = {
        "schema_version",
        "purpose",
        "protocol_id",
        "config_sha256",
        "runner_sha256",
        "snapshot_manifest_sha256",
        "freeze_receipt_sha256",
        "prequalification_design_lock_sha256",
        "input_manifest_sha256",
        "work_plan_digest",
        "authorization_digest",
        "parsed_contract_digest",
        "scientific_contract_digest",
    }
    expected_runtime_fields = {
        "schema_version",
        "scientific_contract_digest",
        "jobs",
        "thread_environment",
        "start_method",
        "runtime_contract_digest",
    }
    execution_payload = {
        key: value
        for key, value in execution_contract.items()
        if key != "scientific_contract_digest"
    }
    runtime_payload = {
        key: value
        for key, value in runtime_contract.items()
        if key != "runtime_contract_digest"
    }
    expected_work_plan = build_work_plan(config, purpose="excluded_rehearsal")
    observed_work_plan = read_json(rehearsal / "adjudication/work_plan.json")
    input_root = rehearsal / "persisted_inputs"
    input_manifest_sha256 = sha256_file(input_root / "input_manifest.json")
    checks = {
        "attempt_receipt_exact": receipt == expected_receipt,
        "single_rehearsal_summary": observed_rehearsal_summaries
        == ["excluded_rehearsal/cohort_summary.json"],
        "completion_hash_chain": set(completion) == expected_completion_fields
        and completion.get("schema_version")
        == "nursery-corrective-excluded-rehearsal-completion-v1"
        and completion.get("status") == "PASS"
        and completion.get("attempt_receipt_sha256") == sha256_file(receipt_path)
        and completion.get("cohort_summary_sha256")
        == sha256_file(rehearsal / "cohort_summary.json")
        and completion.get("complete_manifest_sha256")
        == sha256_file(rehearsal / "complete_manifest.json")
        and completion.get("scientific_decision")
        == "SCIENTIFIC_INFERENCE_SUPPRESSED"
        and completion.get("scientific_inference_suppressed") is True
        and int(completion.get("development_outcome_count", -1)) == 0
        and int(completion.get("confirmation_outcome_count", -1)) == 0,
        "rehearsal_exact_manifest": verify_exact_manifest(
            rehearsal,
            rehearsal_manifest,
            manifest_filename="complete_manifest.json",
        )["status"]
        == "PASS",
        "execution_contract_exact": set(execution_contract)
        == expected_execution_fields
        and execution_contract.get("schema_version")
        == "nursery-corrective-scientific-execution-contract-v1"
        and execution_contract.get("purpose") == "excluded_rehearsal"
        and execution_contract.get("protocol_id") == config["protocol"]["id"]
        and execution_contract.get("config_sha256")
        == sha256_file(snapshot / CONFIG_RELATIVE)
        and execution_contract.get("runner_sha256")
        == sha256_file(
            snapshot / "scripts/run_synthetic_corrective_alignment_v1.py"
        )
        and execution_contract.get("snapshot_manifest_sha256")
        == sha256_file(snapshot / "snapshot_manifest.json")
        and execution_contract.get("freeze_receipt_sha256")
        == sha256_file(package / "freeze_receipt.json")
        and execution_contract.get("prequalification_design_lock_sha256")
        == sha256_file(package / "prequalification_design_lock.json")
        and execution_contract.get("input_manifest_sha256")
        == input_manifest_sha256
        and execution_contract.get("work_plan_digest")
        == expected_work_plan["digest"]
        and execution_contract.get("authorization_digest") is None
        and execution_contract.get("parsed_contract_digest") is None
        and execution_contract.get("scientific_contract_digest")
        == canonical_digest(execution_payload),
        "work_plan_and_inputs_exact": observed_work_plan == expected_work_plan
        and verify_persisted_inputs(input_root).get("status") == "PASS",
        "fixed_runtime_contract": set(runtime_contract)
        == expected_runtime_fields
        and runtime_contract.get("schema_version")
        == "nursery-corrective-runtime-execution-contract-v1"
        and int(runtime_contract.get("jobs", -1))
        == int(config["parallel"]["frozen_jobs"])
        and runtime_contract.get("thread_environment")
        == config["parallel"]["thread_environment"]
        and runtime_contract.get("start_method")
        == config["parallel"]["start_method"]
        and runtime_contract.get("scientific_contract_digest")
        == execution_contract.get("scientific_contract_digest")
        and runtime_contract.get("runtime_contract_digest")
        == canonical_digest(runtime_payload),
        "scientific_inference_suppressed": set(summary)
        == expected_summary_fields
        and summary.get("schema_version")
        == "nursery-corrective-cohort-summary-v1"
        and summary.get("status") == "PASS"
        and summary.get("purpose") == "excluded_rehearsal"
        and summary.get("integrity_status") == "PASS"
        and summary.get("scientific_decision")
        == "SCIENTIFIC_INFERENCE_SUPPRESSED"
        and summary.get("publication_state") == "UNPUBLISHED_CANDIDATE"
        and summary.get("confirmation_authorized") is False
        and summary.get("adjudication_manifest_sha256")
        == sha256_file(rehearsal / "adjudication/manifest.json")
        and summary.get("input_manifest_sha256") == input_manifest_sha256
        and summary.get("scientific_contract_digest")
        == execution_contract.get("scientific_contract_digest")
        and summary.get("runtime_contract_digest")
        == runtime_contract.get("runtime_contract_digest")
        and int(summary.get("development_outcome_count", -1)) == 0
        and int(summary.get("confirmation_outcome_count", -1)) == 0,
        "independent_recompute_exact": verify_exact_manifest(
            recompute,
            recompute_manifest,
            manifest_filename="complete_manifest.json",
        )["status"]
        == "PASS"
        and comparison == recomputed_comparison
        and recomputed_comparison.get("status") == "PASS"
        and recomputed_comparison.get("byte_identical") is True,
    }
    return {
        "schema_version": "nursery-corrective-rehearsal-contract-validation-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed_rehearsal_summaries": observed_rehearsal_summaries,
        "recomputed_comparison": recomputed_comparison,
        "comparison_sha256": sha256_file(comparison_path),
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }


def _pristine_state_validation(root: Path, package: Path, config: dict) -> dict:
    paths = {
        name: (root / str(config["paths"][field])).absolute()
        for name, field in {
            "development_output": "development_output",
            "development_staging": "development_staging",
            "development_inner_staging": "development_inner_staging",
            "authorization_claim": "authorization_claim",
            "capability_receipt": "authorization_capability_receipt",
            "publication_seal": "development_publication_seal",
        }.items()
    }
    absent = {}
    confined = {}
    for name, path in paths.items():
        try:
            path.lstat()
            absent[name] = False
        except FileNotFoundError:
            absent[name] = True
        try:
            _no_symlink_ancestry(path, stop=root)
            confined[name] = True
        except RuntimeError:
            confined[name] = False
    registry = read_json(package / "outcome_registry.json")
    expected_fields = {
        "schema_version",
        "development_outcome_count",
        "confirmation_outcome_count",
        "development_output_exists",
        "confirmation_output_exists",
        "authorization_consumed",
    }
    launch_paths = {
        "launch_seal": root / str(config["paths"]["launch_seal"]),
        "launch_seal_staging": root / str(config["paths"]["launch_seal_staging"]),
        "package_core_manifest": package / "package_core_manifest.json",
        "package_not_ready": package / "PACKAGE_NOT_READY.json",
    }
    launch_absent = {}
    for name, path in launch_paths.items():
        try:
            path.lstat()
            launch_absent[name] = False
        except FileNotFoundError:
            launch_absent[name] = True
    checks = {
        "all_transaction_paths_lstat_absent": all(absent.values()),
        "all_transaction_paths_confined_without_symlink_ancestry": all(
            confined.values()
        ),
        "outcome_registry_exact_zero_state": set(registry) == expected_fields
        and registry.get("schema_version") == "nursery-corrective-outcome-registry-v1"
        and int(registry.get("development_outcome_count", -1)) == 0
        and int(registry.get("confirmation_outcome_count", -1)) == 0
        and registry.get("development_output_exists") is False
        and registry.get("confirmation_output_exists") is False
        and registry.get("authorization_consumed") is False,
        "atomic_launch_paths_pristine": all(launch_absent.values()),
    }
    return {
        "schema_version": "nursery-corrective-pristine-state-validation-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "transaction_paths": {name: str(path) for name, path in paths.items()},
        "lstat_absent": absent,
        "path_confinement": confined,
        "launch_lstat_absent": launch_absent,
        "development_outcome_count": int(registry.get("development_outcome_count", -1)),
        "confirmation_outcome_count": int(
            registry.get("confirmation_outcome_count", -1)
        ),
    }


def _traceability(package: Path, snapshot: Path) -> dict:
    specification = read_json(
        snapshot / "docs/synthetic_corrective_alignment_v1_traceability.json"
    )
    expected_status = {
        "prequalification_design_lock.json": "LOCKED_BEFORE_OFFICIAL_FIXTURES",
        "construction_qualification_attempt_consumed.json": "ONE_CONSTRUCTION_ATTEMPT_CONSUMED",
        "prequalification_transition_verification.json": "PASS",
        "freeze_attempt_consumed.json": "ONE_FREEZE_ATTEMPT_CONSUMED",
        "prefreeze_evidence_contract.json": "FROZEN_BEFORE_REHEARSAL",
        "PREFREEZE_MICRO_ATTEMPT_1_FAILED.json": "FAILED_NO_OUTCOME",
        "PREFREEZE_MICRO_ATTEMPT_2_FAILED.json": "FAILED_NO_OUTCOME",
        "PREFREEZE_MICRO_ATTEMPT_3_SUPERSEDED.json": "SUPERSEDED_NO_OUTCOME",
        "construction_qualification_decision.json": "PASS",
        "mechanism_qualification.json": "PASS",
        "parallel_equivalence_attempt_4.json": "PASS",
        "parallel_benchmark_attempt_4.json": "PASS",
        "parallel_micro_attempt_4_completion.json": "PASS",
        "freeze_receipt.json": "FROZEN",
        "excluded_rehearsal_completion.json": "PASS",
        "independent_recompute_attempt_consumed.json": "ONE_INDEPENDENT_RECOMPUTE_ATTEMPT_CONSUMED",
        "independent_recompute_comparison.json": "PASS",
        "official_tests_attempt_consumed.json": "ONE_OFFICIAL_TEST_ATTEMPT_CONSUMED",
        "official_test_execution_report.json": "PASS",
        "preservation_attempt_consumed.json": "ONE_PRESERVATION_ATTEMPT_CONSUMED",
        "preservation/preservation_proof.json": "PASS",
        "finalization_attempt_consumed.json": "ONE_FINALIZATION_ATTEMPT_CONSUMED",
        "identifier_operation_audit.json": "PASS",
        "repository_operation_census.json": "PASS",
        "prior_line_reverification.json": "PASS",
        "design_contract_validation.json": "PASS",
        "rehearsal_contract_validation.json": "PASS",
        "pristine_state_validation.json": "PASS",
    }
    requirement_rows = []
    requirement_ids = [
        str(row.get("id", "")) for row in specification.get("requirements", [])
    ]
    artifact_lists = [
        list(map(str, row.get("artifacts", [])))
        for row in specification.get("requirements", [])
    ]
    specification_checks = {
        "schema": specification.get("schema_version")
        == "nursery-corrective-traceability-v1",
        "unique_nonempty_requirement_ids": bool(requirement_ids)
        and all(requirement_ids)
        and len(requirement_ids) == len(set(requirement_ids)),
        "nonempty_unique_safe_artifact_lists": all(
            artifacts
            and len(artifacts) == len(set(artifacts))
            and all(
                not Path(relative).is_absolute()
                and ".." not in Path(relative).parts
                and "\\" not in relative
                for relative in artifacts
            )
            for artifacts in artifact_lists
        ),
    }
    for requirement in specification["requirements"]:
        artifact_rows = []
        for relative in requirement["artifacts"]:
            path = package / str(relative)
            regular = False
            digest = None
            schema = None
            status = None
            try:
                metadata = path.lstat()
                regular = stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(
                    metadata.st_mode
                )
                if regular:
                    digest = sha256_file(path)
                    if path.suffix == ".json":
                        value = read_json(path)
                        schema = value.get("schema_version")
                        status = value.get("status")
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                regular = False
            expected = expected_status.get(str(relative))
            passed = regular and (expected is None or status == expected)
            artifact_rows.append(
                {
                    "path": str(relative),
                    "regular_non_symlink": regular,
                    "sha256": digest,
                    "schema_version": schema,
                    "observed_status": status,
                    "expected_status": expected,
                    "status": "PASS" if passed else "FAIL",
                }
            )
        requirement_rows.append(
            {
                "requirement": requirement["id"],
                "artifacts": artifact_rows,
                "status": (
                    "PASS"
                    if all(row["status"] == "PASS" for row in artifact_rows)
                    else "FAIL"
                ),
            }
        )
    passed = (
        all(specification_checks.values())
        and all(row["status"] == "PASS" for row in requirement_rows)
    )
    return {
        "schema_version": "nursery-corrective-traceability-validation-v1",
        "status": "PASS" if passed else "FAIL",
        "specification_checks": specification_checks,
        "requirements": requirement_rows,
    }


def _scientific_report(
    config: dict,
    construction: dict,
    equivalence: dict,
    benchmark: dict,
    comparison: dict,
    preservation: dict,
    prior: dict,
) -> bytes:
    command = str(config["commands"]["development_one_shot"])
    report = f"""# Corrective synthetic weak-alignment study v1

This is candidate launch-readiness evidence, not an effective authorization and not a scientific result. It freezes a narrow synthetic test of whether synchronized training-only event cues can correct an incorrect or ambiguous lexical/action ranking and transfer to independently generated instances and never-trained compositions. It makes no infant-learning or ecological-validity claim. Readiness becomes effective only if the atomic launch seal is subsequently and successfully published.

The prior line remains closed: the independent reverification recovered all {prior['unaveraged_action_cell_count']} perfect, tie-free action-present/null cells, zero detector-selected semantic-update weight, and present/null contrast correlations {prior['sync_minus_absent_present_null_correlation']:.15f} and {prior['sync_minus_randomized_present_null_correlation']:.15f}. No 281xxx or 289xxx identifier is authorized.

Construction qualification is {construction['status']}. Jobs=1 and jobs={int(config['parallel']['frozen_jobs'])} adjudication are {equivalence['status']} and byte-identical; measured speedup is {benchmark['measured_speedup']:.3f}x, with an estimated development wall time of {benchmark['estimated_development_wall_hours']:.3f} hours including the frozen contingency. The single excluded rehearsal and independent recompute are byte-identical ({comparison['status']}). Preservation is {preservation['status']}. Development outcomes: 0. Confirmation outcomes: 0.

The full repository test run makes exactly one disclosed historical deselection: `tests/test_synthetic_development_launch_v3.py::test_exact_one_shot_command_and_output_are_frozen`. That preserved old test asserts a pre-execution absence condition that became stale when the now-authoritative closed-protocol v3 development output was produced; no other test is deselected.

The co-primary endpoints are strict unique-top-1 lexical acquisition and strict unique-top-1 held-out-composition transfer. Both must satisfy the frozen intersection-union rule, including a mean effect and one-sided corpus-level lower bound strictly beyond +{float(config['analysis']['minimum_practical_effect']):.2f}. Proper scores are descriptive secondary endpoints and cannot rescue a failed acquisition claim. A clean informative null produces STOP; broken controls or mechanisms fail the informativeness contract.

Method choices follow primary work on cross-situational learning ([Yu & Smith, 2007](https://doi.org/10.1111/j.1467-9280.2007.01915.x)), latent noisy alignment ([Fazly et al., 2010](https://doi.org/10.1111/j.1551-6709.2010.01104.x)), multiple-instance bags ([Dietterich et al., 1997](https://doi.org/10.1016/S0004-3702(96)00034-3)), training-only privileged information ([Karlsson et al., 2022](https://proceedings.mlr.press/v151/k-a-karlsson22a.html)), proper scores ([Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437)), and prospective simulation discipline ([Morris et al., 2019](https://doi.org/10.1002/sim.8086)). Full source roles and threshold rationales are frozen in the package snapshot.

Exact later development command (not executed by this task):

This command is effective only with the exact non-replayable authorization inside the valid atomic launch seal; the text alone grants no authority.

```sh
{command}
```
"""
    return report.encode("utf-8")


def _finalize(root: Path, config: dict) -> dict:
    package = root / PACKAGE_RELATIVE
    snapshot = package / "frozen_source_snapshot"
    construction = read_json(package / "construction_qualification_decision.json")
    equivalence = read_json(package / "parallel_equivalence_attempt_4.json")
    benchmark = read_json(package / "parallel_benchmark_attempt_4.json")
    comparison = read_json(package / "independent_recompute_comparison.json")
    tests = read_json(package / "official_test_execution_report.json")
    preservation = read_json(package / "preservation/preservation_proof.json")
    snapshot_verification = verify_snapshot(snapshot)
    prefreeze_verification = verify_prefreeze_evidence(package)

    prior = reverify_prior_closed_line(root)
    write_json(package / "prior_line_reverification.json", prior)
    operation_audit = operation_identifier_audit(package, config)
    write_json(package / "identifier_operation_audit.json", operation_audit)
    census = repository_operation_census(root, package, config)
    write_json(package / "repository_operation_census.json", census)
    design = _design_contract_validation(
        root, package, snapshot, config, construction
    )
    write_json(package / "design_contract_validation.json", design)
    rehearsal_validation = _rehearsal_contract_validation(package, snapshot, config)
    write_json(
        package / "rehearsal_contract_validation.json", rehearsal_validation
    )
    pristine = _pristine_state_validation(root, package, config)
    write_json(package / "pristine_state_validation.json", pristine)
    lifecycle_attempts = _validate_lifecycle_attempt_receipts(
        root,
        package,
        snapshot,
        config,
    )
    official_validation = validate_official_test_report(
        tests,
        snapshot_root=snapshot,
        full_repository_root=root,
        executable=root / str(config["environment"]["python"]),
        package_root=package,
    )
    fresh_comparison = recompute_comparison_value(
        package / "excluded_rehearsal",
        package / "independent_recompute",
    )
    preliminary = {
        "snapshot": snapshot_verification["status"] == "PASS",
        "prefreeze": prefreeze_verification["status"] == "PASS",
        "prior": prior["status"] == "PASS",
        "operations": operation_audit["status"] == "PASS"
        and census["status"] == "PASS",
        "design": design["status"] == "PASS",
        "rehearsal": rehearsal_validation["status"] == "PASS",
        "comparison": comparison == fresh_comparison
        and fresh_comparison["status"] == "PASS",
        "official_tests": official_validation["status"] == "PASS",
        "preservation_report": preservation.get("status") == "PASS"
        and preservation.get("old_artifacts_preserved_byte_for_byte") is True,
        "pristine": pristine["status"] == "PASS",
        "lifecycle_attempts": lifecycle_attempts["status"] == "PASS",
    }
    if not all(preliminary.values()):
        raise RuntimeError(
            f"pre-report finalization evidence failed: {preliminary}"
        )
    atomic_write_bytes(
        package / "SCIENTIFIC_REPORT.md",
        _scientific_report(
            config,
            construction,
            equivalence,
            benchmark,
            comparison,
            preservation,
            prior,
        ),
    )
    traceability = _traceability(package, snapshot)
    write_json(package / "traceability_validation.json", traceability)

    test_groups = tests.get("required_test_groups", {})
    test_group_pass = {
        name: test_groups.get(name, {}).get("status") == "PASS"
        for name in (
            "construct_mechanism_and_leakage",
            "registry_and_operation_firewall",
            "manifest_and_frozen_evidence_mutations",
            "parallel_shards_inventory_and_interruption",
            "authorization_paths_jobs_and_replay",
            "atomic_failure_no_outcome",
            "prospective_lock_and_input_commitments",
            "rehearsal_one_shot_and_no_confirmation",
            "frozen_module_origins",
        )
    }
    category = {
        name: row["status"] == "PASS"
        for name, row in design["categories"].items()
    }
    outcome_registry = read_json(package / "outcome_registry.json")
    gates = {
        "old_line_closure_upheld": prior["status"] == "PASS",
        "original_brief_respected": category["original_brief"],
        "fresh_registries_disjoint": category["registries"]
        and test_group_pass["registry_and_operation_firewall"],
        "old_ranges_never_operated": census["status"] == "PASS"
        and operation_audit["status"] == "PASS"
        and not operation_audit["old_quarantined_references"],
        "confirmation_reserve_untouched": census["status"] == "PASS"
        and not operation_audit["confirmation_reserve_references"],
        "construct_headroom": category["construct"],
        "corrective_disagreement_path": category["corrective_path"],
        "mechanism_mutations": construction["status"] == "PASS"
        and test_group_pass["construct_mechanism_and_leakage"],
        "acquisition_endpoints": category["acquisition"],
        "temporal_event_null_alignment": category["temporal"],
        "matched_causal_controls": category["controls"],
        "test_time_side_withholding": category["withholding"],
        "independent_generalization": category["generalization"],
        "corpus_heterogeneity": category["heterogeneity"],
        "meaningful_model_stochasticity": category["model_variation"],
        "presence_null_dissociation": category["presence_null"],
        "frozen_statistics_and_decision_rule": category["statistics"],
        "prospective_prequalification_lock": category["prospective_freeze"]
        and test_group_pass["prospective_lock_and_input_commitments"],
        "sample_size_frozen": design["checks"]["sample_size_and_seeds_frozen"],
        "jobs_1_jobs_n_byte_identity": equivalence.get("status") == "PASS"
        and equivalence.get("byte_identical") is True,
        "parallel_speedup_and_resource_ceiling": benchmark.get("status") == "PASS"
        and all(benchmark.get("gates", {}).values()),
        "isolated_shards_and_canonical_merge": category["parallel"]
        and test_group_pass["parallel_shards_inventory_and_interruption"]
        and rehearsal_validation["checks"]["rehearsal_exact_manifest"],
        "failure_atomic_no_outcome": test_group_pass["atomic_failure_no_outcome"]
        and pristine["checks"]["all_transaction_paths_lstat_absent"],
        "authorization_actual_paths_jobs_bound": test_group_pass[
            "authorization_paths_jobs_and_replay"
        ],
        "authorization_non_replayable": test_group_pass[
            "authorization_paths_jobs_and_replay"
        ]
        and test_group_pass["rehearsal_one_shot_and_no_confirmation"],
        "manifest_exactness_and_path_safety": test_group_pass[
            "manifest_and_frozen_evidence_mutations"
        ]
        and prefreeze_verification["status"] == "PASS",
        "authorization_negative_tests": test_group_pass[
            "authorization_paths_jobs_and_replay"
        ]
        and test_group_pass["atomic_failure_no_outcome"],
        "frozen_snapshot_integrity": snapshot_verification["status"] == "PASS"
        and test_group_pass["frozen_module_origins"],
        "exactly_one_excluded_rehearsal": rehearsal_validation["status"] == "PASS"
        and census["status"] == "PASS",
        "rehearsal_scientific_inference_suppressed": rehearsal_validation[
            "checks"
        ]["scientific_inference_suppressed"],
        "independent_rehearsal_recompute_byte_identity": rehearsal_validation[
            "checks"
        ]["independent_recompute_exact"],
        "official_tests": tests.get("status") == "PASS"
        and tests.get("required_inventory_status") == "PASS"
        and tests.get("deselection_contract_status") == "PASS"
        and int(tests.get("observed_deselected_count", -1)) == 1,
        "prior_evidence_preserved": preservation.get("status") == "PASS"
        and preservation.get("old_artifacts_preserved_byte_for_byte") is True,
        "development_outcome_count_zero": int(
            outcome_registry.get("development_outcome_count", -1)
        )
        == 0,
        "confirmation_outcome_count_zero": int(
            outcome_registry.get("confirmation_outcome_count", -1)
        )
        == 0,
        "development_output_pristine": pristine["status"] == "PASS",
        "atomic_launch_seal_preconditions": pristine["checks"][
            "atomic_launch_paths_pristine"
        ]
        and test_group_pass["atomic_failure_no_outcome"],
        "traceability_complete": traceability["status"] == "PASS",
    }
    adjudication_input = {
        "schema_version": "nursery-corrective-package-adjudication-input-v1",
        "gates": gates,
        "contradictions": [],
        "development_outcome_count": int(
            outcome_registry["development_outcome_count"]
        ),
        "confirmation_outcome_count": int(
            outcome_registry["confirmation_outcome_count"]
        ),
    }
    if set(gates) != set(PACKAGE_GATES):
        raise RuntimeError("runner final gate set differs from adjudicator contract")
    write_json(package / "adjudication_inputs.json", adjudication_input)
    if not all(gates.values()):
        raise RuntimeError(
            "corrective package gate failure; preserve this version and do not authorize"
        )
    finalization_proof = _issue_finalization_proof(
        root,
        package,
        config,
        gates=gates,
    )
    return finalize_package(
        root,
        package,
        config,
        gates=gates,
        finalization_proof=finalization_proof,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "micro",
            "construction",
            "freeze",
            "excluded-rehearsal",
            "recompute-rehearsal",
            "tests",
            "preservation",
            "finalize",
            "run-development",
        ),
    )
    parser.add_argument("--snapshot-root")
    parser.add_argument("--authorization")
    parser.add_argument("--output-root")
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    is_snapshot_runner = (
        SOURCE_ROOT.name == "frozen_source_snapshot"
        and SOURCE_ROOT.parent.name
        == "synthetic_corrective_development_launch_package_v1"
    )
    prefreeze_commands = {"micro", "construction", "freeze"}
    postfreeze_commands = {
        "excluded-rehearsal",
        "recompute-rehearsal",
        "tests",
        "preservation",
        "finalize",
        "run-development",
    }
    if args.command in prefreeze_commands and is_snapshot_runner:
        raise PermissionError("frozen runner cannot execute pre-freeze commands")
    if args.command in postfreeze_commands and not is_snapshot_runner:
        raise PermissionError("post-freeze commands require the frozen snapshot runner")
    if args.command == "run-development":
        result = _run_development(
            args.snapshot_root,
            args.authorization,
            args.output_root,
            int(args.jobs),
        )
    elif args.command == "excluded-rehearsal":
        result = _excluded_rehearsal(args.snapshot_root, int(args.jobs))
    elif args.command == "recompute-rehearsal":
        result = _recompute_rehearsal(args.snapshot_root)
    else:
        if args.command in {"tests", "preservation", "finalize"}:
            root, _snapshot, config = _frozen_context(str(SOURCE_ROOT))
        else:
            root = _live_root()
            config = load_config(root / CONFIG_RELATIVE, repository_root=root)
        package = _initialize_package(root)
        if args.command == "micro":
            if config["protocol"]["status"] != "pre_freeze":
                raise RuntimeError("micro proof is forbidden after freeze status")
            result = _micro(root, config)
        elif args.command == "construction":
            if config["protocol"]["status"] != "pre_freeze":
                raise RuntimeError("construction qualification is forbidden after freeze status")
            result = _construction(root, config)
        elif args.command == "freeze":
            require_frozen(config)
            for path in (
                package / "frozen_source_snapshot",
                package / "freeze_receipt.json",
                package / "prefreeze_evidence_manifest.json",
                package / "prefreeze_evidence_contract.json",
                package / "frozen_environment.json",
                package / "frozen_seed_registries.json",
                package / "prequalification_transition_verification.json",
            ):
                require_lstat_absent(path)
            _consume_lifecycle_attempt(package, root, config, "freeze")
            transition = _verify_prequalification_design_lock(
                root,
                package,
                config,
                require_fixture_contracts=True,
                environment_root=root,
            )
            micro_validation = _validate_parallel_micro_attempt_4(
                root, package, config
            )
            persisted_construction = read_json(
                package / "construction_qualification_decision.json"
            )
            recomputed_construction = (
                _recompute_construction_qualification_decision(package, config)
            )
            required_prefreeze = {
                "prequalification_transition": transition["status"],
                "construction": persisted_construction["status"],
                "construction_recomputed_exactly": (
                    "PASS"
                    if persisted_construction == recomputed_construction
                    else "FAIL"
                ),
                "sealed_micro_completion": micro_validation["status"],
                "parallel_equivalence": read_json(
                    package / "parallel_equivalence_attempt_4.json"
                )["status"],
                "parallel_benchmark": read_json(
                    package / "parallel_benchmark_attempt_4.json"
                )["status"],
            }
            if any(value != "PASS" for value in required_prefreeze.values()):
                raise RuntimeError(f"pre-freeze gates failed: {required_prefreeze}")
            micro_lock_verification = _verify_prequalification_design_lock(
                root,
                package,
                config,
                require_fixture_contracts=True,
                fixture_contract_names=("micro_jobs_1", "micro_jobs_n"),
                environment_root=root,
            )
            construction_attempt_validation = (
                _validate_construction_attempt_receipt(
                    package,
                    micro_lock_verification=micro_lock_verification,
                    micro_validation=micro_validation,
                )
            )
            if construction_attempt_validation["status"] != "PASS":
                raise RuntimeError("construction attempt receipt is invalid")
            transition_path = package / "prequalification_transition_verification.json"
            require_lstat_absent(transition_path)
            write_json(transition_path, transition)
            receipt = freeze_snapshot(
                root,
                package,
                config,
                tracked_files=TRACKED,
            )
            if read_json(package / "frozen_environment.json") != read_json(
                package / "prequalification_design_lock.json"
            )["qualification_environment"]:
                raise RuntimeError("frozen environment differs from qualification lock")
            snapshot = package / "frozen_source_snapshot"
            snapshot_transition = _verify_prequalification_design_lock(
                snapshot,
                package,
                load_config(snapshot / CONFIG_RELATIVE, repository_root=snapshot),
                require_fixture_contracts=True,
                environment_root=root,
            )
            if snapshot_transition != transition:
                raise RuntimeError(
                    "frozen snapshot does not reproduce prequalification transition"
                )
            result = {
                "freeze_receipt": receipt,
                "prequalification_transition": transition,
                "snapshot_transition_byte_equivalent": True,
            }
        elif args.command == "tests":
            require_frozen(config)
            snapshot = package / "frozen_source_snapshot"
            require_lstat_absent(
                package / "official_test_execution_report.json"
            )
            _consume_lifecycle_attempt(
                package,
                snapshot,
                config,
                "official_tests",
            )
            comparison = read_json(package / "independent_recompute_comparison.json")
            if comparison.get("status") != "PASS" or comparison.get(
                "byte_identical"
            ) is not True:
                raise RuntimeError("official tests require a byte-identical rehearsal recompute")
            result = run_tests(
                snapshot,
                package / "official_test_execution_report.json",
                root / str(config["environment"]["python"]),
                full_repository_root=root,
            )
        elif args.command == "preservation":
            require_frozen(config)
            snapshot = package / "frozen_source_snapshot"
            require_lstat_absent(package / "preservation")
            _consume_lifecycle_attempt(
                package,
                snapshot,
                config,
                "preservation",
            )
            official_tests = read_json(package / "official_test_execution_report.json")
            if official_tests.get("status") != "PASS":
                raise RuntimeError("preservation requires passing official tests")
            result = verify_prior_preservation(
                root,
                package / "preservation",
                config,
                jobs=int(config["parallel"]["frozen_jobs"]),
            )
        elif args.command == "finalize":
            require_frozen(config)
            snapshot = package / "frozen_source_snapshot"
            for path in (
                package / "prior_line_reverification.json",
                package / "identifier_operation_audit.json",
                package / "repository_operation_census.json",
                package / "design_contract_validation.json",
                package / "rehearsal_contract_validation.json",
                package / "pristine_state_validation.json",
                package / "SCIENTIFIC_REPORT.md",
                package / "traceability_validation.json",
                package / "adjudication_inputs.json",
                package / "package_core_manifest.json",
                package / "PACKAGE_NOT_READY.json",
                root / str(config["paths"]["launch_seal"]),
                root / str(config["paths"]["launch_seal_staging"]),
            ):
                require_lstat_absent(path)
            _consume_lifecycle_attempt(
                package,
                snapshot,
                config,
                "finalization",
            )
            preservation = read_json(package / "preservation/preservation_proof.json")
            if preservation.get("status") != "PASS":
                raise RuntimeError("finalization requires a passing preservation proof")
            result = _finalize(root, config)
        else:
            raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
