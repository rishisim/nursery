from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_info

from . import PROTOCOL_ID
from .adjudicator import (
    PACKAGE_GATES,
    authorization_payload,
    construction_qualification_decision,
    package_decision,
)
from .protocol import (
    atomic_rename_directory_noreplace,
    atomic_write_bytes,
    canonical_bytes,
    canonical_digest,
    manifest_for_paths,
    manifest_for_tree,
    load_config,
    read_json,
    registry_snapshot,
    require_frozen,
    require_lstat_absent,
    sha256_file,
    tree_directory_paths,
    tree_file_paths,
    verify_exact_manifest,
    write_json,
    python_startup_flags_projection,
)


CLOSED_PROTOCOL_DESELECTION = (
    "tests/test_synthetic_development_launch_v3.py::"
    "test_exact_one_shot_command_and_output_are_frozen"
)
CLOSED_PROTOCOL_DESELECTION_RATIONALE = (
    "historical pre-execution invariant is stale because the authoritative closed-protocol "
    "development output now exists; the old test is preserved unchanged and the prior line "
    "is independently reverified and byte-preserved"
)


REQUIRED_CORRECTIVE_TEST_GROUPS = {
    "construct_mechanism_and_leakage": [
        "test_generated_corpus_separates_worker_inputs_keys_and_side",
        "test_action_geometry_varies_beyond_coordinate_permutation_and_is_shared",
        "test_train_evaluation_provenance_is_computed_and_detects_namespace_collision",
        "test_corrective_disagreement_microcases_and_mutations",
        "test_model_seed_changes_initialization_order_and_fitted_state",
        "test_prediction_is_side_free_and_rejects_sentinel",
        "test_scientific_decision_maps_controls_dependence_and_primary_status",
        "test_present_null_dependence_rejects_equal_vectors_even_if_lexical_differs",
    ],
    "registry_and_operation_firewall": [
        "test_registry_allocation_is_fresh_disjoint_and_confirmation_sealed",
        "test_firewall_and_operation_audit_reject_old_reserve_and_wrong_purpose_ids",
    ],
    "manifest_and_frozen_evidence_mutations": [
        "test_exact_manifest_rejects_structural_mutations",
        "test_exact_manifest_rejects_symlink_and_broken_symlink",
        "test_exact_manifest_rejects_empty_hidden_directory_and_root_symlink",
        "test_anticipated_launch_manifest_requires_exact_atomic_file_set",
        "test_prefreeze_evidence_rejects_every_bound_tree_mutation",
        "test_snapshot_receipt_rejects_schema_protocol_and_tracked_file_mutations",
        "test_parallel_micro_completion_rejects_mutated_or_partial_state",
        "test_benchmark_validator_recomputes_gates_and_telemetry",
        "test_construction_attempt_receipt_rejects_mutated_hash_chain",
        "test_recompute_comparison_rejects_changed_adjudication_bytes",
        "test_official_test_report_validator_rejects_mutable_groups_and_bindings",
        "test_prior_preservation_reverification_detects_postproof_mutation",
        "test_frozen_context_rejects_environment_drift",
    ],
    "parallel_shards_inventory_and_interruption": [
        "test_work_plan_is_exact_unique_cross_product",
        "test_shard_inventory_rejects_duplicate_missing_extra_and_symlink_units",
        "test_self_consistently_resealed_shard_outcome_mutations_fail_closed",
        "test_partial_shards_fail_before_adjudication",
        "test_ledger_requires_exact_count_chain_unit_and_references",
        "test_worker_nonzero_exit_terminates_siblings_and_returns_closed_failure",
        "test_direct_subprocess_backend_has_no_multiprocessing_primitives",
        "test_real_construction_work_plan_unit_ids_are_accepted_by_request_preparation",
        "test_real_subprocess_bootstrap_failure_is_semaphore_free_and_closed",
        "test_subprocess_scheduler_bounds_concurrency_and_enforces_wave_barriers",
        "test_subprocess_worker_rejects_resealed_alternate_receipt_path",
        "test_scheduler_evidence_is_independently_recomputed_and_mutations_fail",
        "test_worker_boundary_and_member_validation_are_bounded_and_mutation_safe",
        "test_child_environment_is_allowlisted_and_forbidden_overrides_fail",
        "test_parent_and_worker_python_startup_flags_are_exactly_bound",
        "test_historical_worker_boundary_allows_only_exact_status_transition",
        "test_scheduler_signal_handler_is_one_shot_and_restores_handlers",
        "test_parallel_artifacts_are_byte_identical_when_available",
    ],
    "authorization_paths_jobs_and_replay": [
        "test_development_has_no_boolean_authorization_bypass",
        "test_authorization_exact_schema_and_digest_mutations",
        "test_capability_issuance_is_nonreplayable_and_path_job_bound",
        "test_capability_rechecks_authorization_and_worker_cannot_forge_chain",
        "test_preflight_binds_actual_paths_jobs_argv_and_replay",
        "test_preflight_rejects_every_preexisting_output_type",
        "test_preflight_requires_every_transaction_path_pristine",
        "test_preflight_rejects_symlinked_transaction_ancestry",
        "test_publication_rechecks_persisted_authorization_after_proof",
    ],
    "atomic_failure_no_outcome": [
        "test_failure_before_publish_leaves_final_output_absent",
        "test_successful_cohort_assembly_moves_summary_and_publishes_exact_tree",
        "test_cohort_summary_mismatch_fails_before_atomic_publication",
        "test_atomic_publication_failpoints_leave_authorized_final_root_absent",
        "test_atomic_publication_success_is_one_exact_directory_tree",
        "test_publication_proof_rejects_summary_or_scope_substitution",
        "test_direct_scientific_publication_without_capability_is_rejected",
        "test_atomic_directory_publication_never_replaces_existing_destination",
        "test_finalize_package_requires_sealed_evidence_proof",
        "test_finalize_package_revalidates_forged_factory_proof_and_config",
        "test_runner_final_gate_keys_exactly_match_adjudicator_contract",
    ],
    "prospective_lock_and_input_commitments": [
        "test_predecessor_v1_tree_is_fully_rehashed_and_failure_is_detected",
        "test_prequalification_lock_rejects_source_config_and_registry_mutations",
        "test_prequalification_lock_refuses_any_prior_official_attempt_path",
        "test_prequalification_lock_binds_prior_attempt_directory_inventory",
        "test_parallel_requires_preparation_manifest_commitment_and_leaves_no_output",
        "test_excluded_rehearsal_deep_apis_require_receipt_capability",
        "test_official_test_contract_has_exact_one_historical_deselection",
    ],
    "rehearsal_one_shot_and_no_confirmation": [
        "test_runner_has_no_generic_scientific_recompute_or_confirmation_bypass",
        "test_generic_rehearsal_apis_cannot_bypass_one_shot_capability",
        "test_direct_live_rehearsal_execution_cannot_consume_frozen_attempt",
        "test_postfreeze_lifecycle_receipts_are_nonreplayable_and_exact",
        "test_rehearsal_validator_binds_worker_boundary_and_suppressed_candidate",
    ],
    "frozen_module_origins": [
        "test_project_module_origins_are_confined_to_the_selected_source_root",
    ],
}


def environment_record(
    repository_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    executable = (root / str(config["environment"]["python"])).absolute()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if Path(sys.executable).resolve() != executable.resolve():
        raise RuntimeError("freeze must run under the configured Python interpreter")
    required_thread_environment = {
        str(name): str(value)
        for name, value in config["parallel"]["thread_environment"].items()
    }
    inherited_allowlist = [
        str(name)
        for name in config["parallel"]["child_inherited_environment_allowlist"]
    ]
    forbidden_unbound = [
        str(name)
        for name in config["parallel"]["forbidden_unbound_environment"]
    ]
    observed_startup_flags = python_startup_flags_projection()
    expected_startup_flags = dict(
        config["environment"]["parent_python_startup_flags"]
    )
    mismatches = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in required_thread_environment.items()
        if os.environ.get(name) != expected
    }
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        mismatches["PYTHONDONTWRITEBYTECODE"] = {
            "expected": "1",
            "observed": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        }
    if observed_startup_flags != expected_startup_flags:
        mismatches["python_startup_flags"] = {
            "expected": expected_startup_flags,
            "observed": observed_startup_flags,
        }
    for name in forbidden_unbound:
        if name in os.environ:
            mismatches[name] = {
                "expected": None,
                "observed": os.environ.get(name),
            }
    if mismatches:
        raise RuntimeError(f"environment lock thread variables differ: {mismatches}")
    modules = {}
    distribution_names = {
        "numpy": "numpy",
        "scipy": "scipy",
        "yaml": "PyYAML",
        "pytest": "pytest",
        "threadpoolctl": "threadpoolctl",
    }
    for module_name in config["environment"]["required_modules"]:
        module = importlib.import_module(str(module_name))
        module_path = Path(module.__file__).resolve()
        distribution = importlib.metadata.distribution(
            distribution_names[str(module_name)]
        )
        distribution_rows = []
        for relative in sorted(map(str, distribution.files or [])):
            path = Path(distribution.locate_file(relative))
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                path = path.resolve(strict=True)
                metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            distribution_rows.append(
                {
                    "path": relative,
                    "bytes": int(metadata.st_size),
                    "sha256": sha256_file(path),
                }
            )
        modules[str(module_name)] = {
            "distribution": distribution_names[str(module_name)],
            "version": importlib.metadata.version(distribution_names[str(module_name)]),
            "module_file": str(module_path),
            "module_file_sha256": sha256_file(module_path),
            "distribution_regular_file_count": len(distribution_rows),
            "distribution_regular_file_bytes": sum(
                int(row["bytes"]) for row in distribution_rows
            ),
            "distribution_regular_files_digest": canonical_digest(
                distribution_rows
            ),
        }
    native_libraries = []
    for row in threadpool_info():
        filepath = Path(str(row.get("filepath", "")))
        if not filepath.is_file():
            raise RuntimeError("threadpool library path is not a regular file")
        native_libraries.append(
            {
                "internal_api": str(row.get("internal_api")),
                "prefix": str(row.get("prefix")),
                "filepath": str(filepath.resolve()),
                "sha256": sha256_file(filepath),
            }
        )
    native_libraries.sort(key=lambda row: (row["filepath"], row["internal_api"]))
    return {
        "schema_version": "nursery-corrective-environment-lock-v2",
        "python_executable": str(executable),
        "python_executable_resolved": str(executable.resolve()),
        "python_sha256": sha256_file(executable),
        "python_version": sys.version,
        "python_startup_flags": observed_startup_flags,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_count": os.cpu_count(),
        "modules": modules,
        "thread_environment": required_thread_environment,
        "child_inherited_environment_allowlist": inherited_allowlist,
        "child_inherited_environment_observed": {
            name: os.environ.get(name) for name in inherited_allowlist
        },
        "forbidden_unbound_environment": forbidden_unbound,
        "forbidden_unbound_environment_observed": {
            name: os.environ.get(name)
            for name in forbidden_unbound
            if name in os.environ
        },
        "native_threadpool_libraries": native_libraries,
        "bytecode_writes_disabled": os.environ.get("PYTHONDONTWRITEBYTECODE")
        == "1",
    }


def freeze_prefreeze_evidence(package_root: str | Path) -> dict[str, Any]:
    package = Path(package_root).resolve()
    manifest_path = package / "prefreeze_evidence_manifest.json"
    contract_path = package / "prefreeze_evidence_contract.json"
    require_lstat_absent(manifest_path)
    require_lstat_absent(contract_path)
    paths = tree_file_paths(package)
    directories = tree_directory_paths(package)
    top_level_entries = sorted(
        {Path(path).parts[0] for path in [*paths, *directories]}
    )
    manifest = manifest_for_paths(package, paths)
    write_json(manifest_path, manifest)
    contract = {
        "schema_version": "nursery-corrective-prefreeze-evidence-contract-v2",
        "status": "FROZEN_BEFORE_REHEARSAL",
        "manifest_file": "prefreeze_evidence_manifest.json",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_digest": manifest["digest"],
        "file_count": manifest["file_count"],
        "bound_top_level_entries": top_level_entries,
        "bound_directory_paths": directories,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }
    write_json(contract_path, contract)
    return contract


def verify_prefreeze_evidence(package_root: str | Path) -> dict[str, Any]:
    package = Path(package_root).resolve()
    manifest_path = package / "prefreeze_evidence_manifest.json"
    contract_path = package / "prefreeze_evidence_contract.json"
    manifest = read_json(manifest_path)
    contract = read_json(contract_path)
    problems = []
    expected_contract_fields = {
        "schema_version",
        "status",
        "manifest_file",
        "manifest_sha256",
        "manifest_digest",
        "file_count",
        "bound_top_level_entries",
        "bound_directory_paths",
        "development_outcome_count",
        "confirmation_outcome_count",
    }
    if set(contract) != expected_contract_fields:
        problems.append("contract_schema")
    if (
        contract.get("schema_version")
        != "nursery-corrective-prefreeze-evidence-contract-v2"
        or contract.get("status") != "FROZEN_BEFORE_REHEARSAL"
        or contract.get("manifest_file") != "prefreeze_evidence_manifest.json"
        or contract.get("manifest_sha256") != sha256_file(manifest_path)
        or contract.get("manifest_digest") != manifest.get("digest")
        or int(contract.get("file_count", -1)) != int(manifest.get("file_count", -2))
        or int(contract.get("development_outcome_count", -1)) != 0
        or int(contract.get("confirmation_outcome_count", -1)) != 0
    ):
        problems.append("contract_values")
    expected_paths = [str(row["path"]) for row in manifest.get("files", [])]
    try:
        observed_manifest = manifest_for_paths(package, expected_paths)
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as error:
        observed_manifest = None
        problems.append(f"manifest_rows:{type(error).__name__}")
    if observed_manifest != manifest:
        problems.append("manifest_rows_changed")
    bound = set(map(str, contract.get("bound_top_level_entries", [])))
    expected_bound = {
        Path(path).parts[0]
        for path in [
            *expected_paths,
            *map(str, contract.get("bound_directory_paths", [])),
        ]
    }
    if bound != expected_bound:
        problems.append("bound_top_level_entries_not_derived")
    try:
        actual_bound_paths = sorted(
            path for path in tree_file_paths(package) if Path(path).parts[0] in bound
        )
    except (OSError, RuntimeError, ValueError) as error:
        actual_bound_paths = []
        problems.append(f"bound_tree:{type(error).__name__}")
    if actual_bound_paths != sorted(expected_paths):
        problems.append("bound_file_set_changed")
    try:
        actual_bound_directories = sorted(
            path
            for path in tree_directory_paths(package)
            if Path(path).parts[0] in bound
        )
    except (OSError, RuntimeError, ValueError) as error:
        actual_bound_directories = []
        problems.append(f"bound_directories:{type(error).__name__}")
    if actual_bound_directories != sorted(
        map(str, contract.get("bound_directory_paths", []))
    ):
        problems.append("bound_directory_set_changed")
    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "file_count": len(expected_paths),
        "bound_top_level_entries": sorted(bound),
        "manifest_sha256": sha256_file(manifest_path),
        "contract_sha256": sha256_file(contract_path),
    }


def freeze_snapshot(
    repository_root: str | Path,
    package_root: str | Path,
    config: Mapping[str, Any],
    *,
    tracked_files: Sequence[str],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    package = Path(package_root).resolve()
    snapshot = package / "frozen_source_snapshot"
    require_lstat_absent(snapshot)
    prefreeze_contract = freeze_prefreeze_evidence(package)
    snapshot.mkdir(parents=True)
    for relative in sorted(set(map(str, tracked_files))):
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"tracked snapshot source must be regular: {source}")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    prior_relative = str(config["registries"]["canonical_prior_registry"])
    if prior_relative not in tracked_files:
        source = root / prior_relative
        destination = snapshot / prior_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    manifest = manifest_for_tree(snapshot, exclude=["snapshot_manifest.json"])
    write_json(snapshot / "snapshot_manifest.json", manifest)
    verification = verify_exact_manifest(
        snapshot,
        manifest,
        manifest_filename="snapshot_manifest.json",
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"frozen snapshot failed: {verification}")
    environment = environment_record(root, config)
    write_json(package / "frozen_environment.json", environment)
    write_json(package / "frozen_seed_registries.json", registry_snapshot(config))
    config_path = snapshot / "configs/synthetic_corrective_alignment_v2.yaml"
    runner_path = snapshot / "scripts/run_synthetic_corrective_alignment_v2.py"
    receipt = {
        "schema_version": "nursery-corrective-freeze-receipt-v2",
        "status": "FROZEN",
        "protocol_id": config["protocol"]["id"],
        "config_status": config["protocol"]["status"],
        "snapshot_manifest_sha256": sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        "snapshot_file_count": manifest["file_count"],
        "snapshot_digest": manifest["digest"],
        "frozen_config_sha256": sha256_file(config_path),
        "frozen_runner_sha256": sha256_file(runner_path),
        "frozen_environment_sha256": sha256_file(
            package / "frozen_environment.json"
        ),
        "frozen_registries_sha256": sha256_file(
            package / "frozen_seed_registries.json"
        ),
        "prequalification_design_lock_sha256": sha256_file(
            package / "prequalification_design_lock.json"
        ),
        "prefreeze_evidence_manifest_sha256": sha256_file(
            package / "prefreeze_evidence_manifest.json"
        ),
        "prefreeze_evidence_contract_sha256": sha256_file(
            package / "prefreeze_evidence_contract.json"
        ),
        "prefreeze_evidence_digest": prefreeze_contract["manifest_digest"],
        "prefreeze_evidence_file_count": prefreeze_contract["file_count"],
        "tracked_files": sorted(
            set(map(str, tracked_files)) | {prior_relative}
        ),
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "confirmation_authorized": False,
    }
    write_json(package / "freeze_receipt.json", receipt)
    return receipt


def verify_snapshot(snapshot_root: str | Path) -> dict[str, Any]:
    supplied = Path(snapshot_root)
    supplied_metadata = supplied.lstat()
    if stat.S_ISLNK(supplied_metadata.st_mode) or not stat.S_ISDIR(
        supplied_metadata.st_mode
    ):
        return {"status": "FAIL", "problems": ["snapshot_root_not_real_directory"]}
    snapshot = supplied.resolve()
    manifest = read_json(snapshot / "snapshot_manifest.json")
    manifest_verification = verify_exact_manifest(
        snapshot,
        manifest,
        manifest_filename="snapshot_manifest.json",
    )
    package = snapshot.parent
    receipt_path = package / "freeze_receipt.json"
    problems = []
    if manifest_verification["status"] != "PASS":
        problems.append("snapshot_manifest")
    try:
        receipt = read_json(receipt_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "FAIL", "problems": [*problems, "freeze_receipt"]}
    expected_receipt_fields = {
        "schema_version",
        "status",
        "protocol_id",
        "config_status",
        "snapshot_manifest_sha256",
        "snapshot_file_count",
        "snapshot_digest",
        "frozen_config_sha256",
        "frozen_runner_sha256",
        "frozen_environment_sha256",
        "frozen_registries_sha256",
        "prequalification_design_lock_sha256",
        "prefreeze_evidence_manifest_sha256",
        "prefreeze_evidence_contract_sha256",
        "prefreeze_evidence_digest",
        "prefreeze_evidence_file_count",
        "tracked_files",
        "development_outcome_count",
        "confirmation_outcome_count",
        "confirmation_authorized",
    }
    if set(receipt) != expected_receipt_fields:
        problems.append("freeze_receipt_schema")
    checks = {
        "receipt_schema_version": receipt.get("schema_version")
        == "nursery-corrective-freeze-receipt-v2",
        "receipt_status": receipt.get("status") == "FROZEN",
        "protocol_id": receipt.get("protocol_id") == PROTOCOL_ID,
        "config_status": receipt.get("config_status") == "frozen",
        "snapshot_manifest_sha256": receipt.get("snapshot_manifest_sha256")
        == sha256_file(snapshot / "snapshot_manifest.json"),
        "snapshot_file_count": int(receipt.get("snapshot_file_count", -1))
        == int(manifest.get("file_count", -2)),
        "snapshot_digest": receipt.get("snapshot_digest") == manifest.get("digest"),
        "frozen_config_sha256": receipt.get("frozen_config_sha256")
        == sha256_file(snapshot / "configs/synthetic_corrective_alignment_v2.yaml"),
        "frozen_runner_sha256": receipt.get("frozen_runner_sha256")
        == sha256_file(snapshot / "scripts/run_synthetic_corrective_alignment_v2.py"),
        "frozen_environment_sha256": receipt.get("frozen_environment_sha256")
        == sha256_file(package / "frozen_environment.json"),
        "frozen_registries_sha256": receipt.get("frozen_registries_sha256")
        == sha256_file(package / "frozen_seed_registries.json"),
        "prequalification_design_lock_sha256": receipt.get(
            "prequalification_design_lock_sha256"
        )
        == sha256_file(package / "prequalification_design_lock.json"),
        "prefreeze_evidence_manifest_sha256": receipt.get(
            "prefreeze_evidence_manifest_sha256"
        )
        == sha256_file(package / "prefreeze_evidence_manifest.json"),
        "prefreeze_evidence_contract_sha256": receipt.get(
            "prefreeze_evidence_contract_sha256"
        )
        == sha256_file(package / "prefreeze_evidence_contract.json"),
        "prefreeze_evidence_digest": receipt.get("prefreeze_evidence_digest")
        == read_json(package / "prefreeze_evidence_contract.json").get(
            "manifest_digest"
        ),
        "prefreeze_evidence_file_count": int(
            receipt.get("prefreeze_evidence_file_count", -1)
        )
        == int(
            read_json(package / "prefreeze_evidence_contract.json").get(
                "file_count", -2
            )
        ),
        "zero_outcomes": int(receipt.get("development_outcome_count", -1)) == 0
        and int(receipt.get("confirmation_outcome_count", -1)) == 0,
        "confirmation_forbidden": receipt.get("confirmation_authorized") is False,
        "tracked_files_exact": list(receipt.get("tracked_files", []))
        == sorted(str(row["path"]) for row in manifest.get("files", [])),
    }
    try:
        design_lock = read_json(package / "prequalification_design_lock.json")
        source_rows = {
            str(row["path"]): {
                "path": str(row["path"]),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in design_lock.get("tracked_nonconfig_manifest", {}).get(
                "files", []
            )
        }
        snapshot_rows = {
            str(row["path"]): {
                "path": str(row["path"]),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for row in manifest.get("files", [])
        }
        config_relative = "configs/synthetic_corrective_alignment_v2.yaml"
        config_row = snapshot_rows.get(config_relative, {})
        checks["prospective_source_anchor"] = (
            design_lock.get("schema_version")
            == "nursery-corrective-prequalification-design-lock-v2"
            and design_lock.get("status")
            == "LOCKED_BEFORE_OFFICIAL_FIXTURES"
            and set(snapshot_rows) == set(source_rows) | {config_relative}
            and all(snapshot_rows.get(path) == row for path, row in source_rows.items())
            and config_row.get("sha256")
            == design_lock.get("anticipated_frozen_config_sha256")
        )
        checks["prospective_environment_anchor"] = read_json(
            package / "frozen_environment.json"
        ) == design_lock.get("qualification_environment")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        checks["prospective_source_anchor"] = False
        checks["prospective_environment_anchor"] = False
    try:
        parsed_config = load_config(
            snapshot / "configs/synthetic_corrective_alignment_v2.yaml",
            repository_root=snapshot,
        )
        require_frozen(parsed_config)
        checks["parsed_frozen_config"] = (
            parsed_config.get("protocol", {}).get("id") == PROTOCOL_ID
            and receipt.get("protocol_id")
            == parsed_config.get("protocol", {}).get("id")
            and receipt.get("config_status")
            == parsed_config.get("protocol", {}).get("status")
        )
        checks["frozen_registries_semantic"] = read_json(
            package / "frozen_seed_registries.json"
        ) == registry_snapshot(parsed_config)
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError):
        checks["parsed_frozen_config"] = False
        checks["frozen_registries_semantic"] = False
    problems.extend(name for name, passed in checks.items() if not passed)
    prefreeze = verify_prefreeze_evidence(package)
    if prefreeze["status"] != "PASS":
        problems.append("prefreeze_evidence")
    return {
        **manifest_verification,
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "receipt_checks": checks,
        "prefreeze_evidence_verification": prefreeze,
        "freeze_receipt_sha256": sha256_file(receipt_path),
    }


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(root / relative)
        for relative in tree_file_paths(root)
    }


def compare_adjudication_trees(
    jobs_one_root: str | Path,
    jobs_n_root: str | Path,
) -> dict[str, Any]:
    left = Path(jobs_one_root).resolve() / "adjudication"
    right = Path(jobs_n_root).resolve() / "adjudication"
    left_hashes = _tree_hashes(left)
    right_hashes = _tree_hashes(right)
    shared = sorted(set(left_hashes) & set(right_hashes))
    mismatched = [
        relative
        for relative in shared
        if left_hashes[relative] != right_hashes[relative]
        or (left / relative).read_bytes() != (right / relative).read_bytes()
    ]
    missing_left = sorted(set(right_hashes) - set(left_hashes))
    missing_right = sorted(set(left_hashes) - set(right_hashes))
    passed = not mismatched and not missing_left and not missing_right
    return {
        "schema_version": "nursery-corrective-parallel-equivalence-v2",
        "status": "PASS" if passed else "FAIL",
        "adjudication_relevant_file_count": len(shared),
        "byte_identical": passed,
        "mismatched": mismatched,
        "missing_from_jobs_1": missing_left,
        "missing_from_jobs_n": missing_right,
        "jobs_and_runtime_telemetry_excluded": True,
        "left_digest": canonical_digest(left_hashes),
        "right_digest": canonical_digest(right_hashes),
    }


def _directory_bytes(root: Path) -> int:
    return sum((root / relative).stat().st_size for relative in tree_file_paths(root))


def _available_memory_bytes() -> int:
    if sys.platform == "darwin":
        output = subprocess.run(
            ["vm_stat"],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        first_line = output.splitlines()[0]
        match = re.search(r"page size of (\d+) bytes", first_line)
        page_size = int(match.group(1)) if match else int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = 0
        for line in output.splitlines():
            if line.startswith(
                ("Pages free:", "Pages inactive:", "Pages speculative:", "Pages purgeable:")
            ):
                available_pages += int(line.split(":", 1)[1].strip().rstrip("."))
        return available_pages * page_size
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError):
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


_PARENT_MEMORY_PROJECTION_METHOD = (
    "all_results_full_micro_peak_scaled_by_unit_ratio"
)


def benchmark_report(
    *,
    jobs_one_root: str | Path,
    jobs_n_root: str | Path,
    input_root: str | Path,
    wall_jobs_one: float,
    wall_jobs_n: float,
    input_generation_wall: float,
    final_assembly_probe_wall: float,
    jobs_n: int,
    development_corpus_count: int,
    micro_corpus_count: int,
    development_unit_count: int,
    micro_unit_count: int,
    minimum_speedup: float,
    wall_time_contingency_multiplier: float,
    resource_contingency_multiplier: float,
    parent_memory_projection_method: str,
    maximum_host_ram_fraction: float,
    maximum_current_available_ram_fraction: float,
    maximum_current_free_disk_fraction: float,
    minimum_logical_cpu_count: int,
) -> dict[str, Any]:
    if parent_memory_projection_method != _PARENT_MEMORY_PROJECTION_METHOD:
        raise ValueError("unsupported parent-memory projection method")
    left = Path(jobs_one_root).resolve()
    right = Path(jobs_n_root).resolve()
    inputs = Path(input_root).resolve()
    telemetry_one = read_json(left / "runtime/runtime_telemetry.json")
    telemetry_n = read_json(right / "runtime/runtime_telemetry.json")
    scheduler_one = read_json(left / "runtime/scheduler_validation.json")
    scheduler_n = read_json(right / "runtime/scheduler_validation.json")
    speedup = float(wall_jobs_one / wall_jobs_n)
    unit_scaling = development_unit_count / micro_unit_count
    corpus_scaling = development_corpus_count / micro_corpus_count
    micro_model_count = int(micro_unit_count // micro_corpus_count)
    development_model_count = int(development_unit_count // development_corpus_count)
    micro_parallel_batch_count = int(
        micro_model_count * math.ceil(micro_corpus_count / int(jobs_n))
    )
    development_parallel_batch_count = int(
        development_model_count
        * math.ceil(development_corpus_count / int(jobs_n))
    )
    maximum_worker_wall = max(
        float(row["wall_seconds"]) for row in telemetry_n["units"]
    )
    projected_worker_wave_seconds = float(
        development_parallel_batch_count * maximum_worker_wall
    )
    projected_full_cohort_seconds = float(wall_jobs_n * unit_scaling)
    additional_worker_stress_seconds = projected_worker_wave_seconds
    projected_input_seconds = float(input_generation_wall * corpus_scaling)
    projected_assembly_seconds = float(final_assembly_probe_wall * unit_scaling)
    estimated_seconds = float(
        (
            projected_input_seconds
            + projected_full_cohort_seconds
            + additional_worker_stress_seconds
            + projected_assembly_seconds
        )
        * float(wall_time_contingency_multiplier)
    )
    max_rss_native = max(
        int(telemetry_one["maximum_worker_peak_rss_native_units"]),
        int(telemetry_n["maximum_worker_peak_rss_native_units"]),
    )
    max_rss_bytes = (
        max_rss_native if sys.platform == "darwin" else max_rss_native * 1024
    )
    parent_rss_native = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    parent_rss_bytes = (
        parent_rss_native if sys.platform == "darwin" else parent_rss_native * 1024
    )
    projected_parent_rss_bytes = int(
        math.ceil(parent_rss_bytes * development_unit_count / micro_unit_count)
    )
    concurrent_rss_bytes = int(
        math.ceil(
            (max_rss_bytes * int(jobs_n) + projected_parent_rss_bytes)
        * float(resource_contingency_multiplier)
        )
    )
    physical_memory_bytes = int(
        os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    )
    available_memory_bytes = _available_memory_bytes()
    disk_free_bytes = int(shutil.disk_usage(right).free)
    jobs_n_disk_bytes = _directory_bytes(right)
    input_disk_bytes = _directory_bytes(inputs)
    projected_development_disk_bytes = int(
        math.ceil(
            (
                jobs_n_disk_bytes * unit_scaling
                + input_disk_bytes * corpus_scaling
            )
            * float(resource_contingency_multiplier)
        )
    )
    telemetry_rows = [
        *telemetry_one.get("units", []),
        *telemetry_n.get("units", []),
    ]
    thread_environment_measured = all(
        row.get("thread_environment_observed")
        == {
            name: os.environ.get(name)
            for name in row.get("thread_environment_observed", {})
        }
        for row in telemetry_rows
    )
    native_thread_counts = [
        int(pool["num_threads"])
        for row in telemetry_rows
        for pool in row.get("native_threadpools", [])
    ]
    native_threads_measured_one = all(value == 1 for value in native_thread_counts)
    worker_boundary_digests = {
        str(row.get("worker_boundary_digest")) for row in telemetry_rows
    }
    logical_cpu_count = int(os.cpu_count() or 0)
    gates = {
        "speedup": speedup >= minimum_speedup,
        "logical_cpu_count": logical_cpu_count >= int(minimum_logical_cpu_count),
        "ram_headroom": concurrent_rss_bytes
        <= physical_memory_bytes * float(maximum_host_ram_fraction)
        and concurrent_rss_bytes
        <= available_memory_bytes
        * float(maximum_current_available_ram_fraction),
        "disk_headroom": projected_development_disk_bytes
        <= disk_free_bytes * float(maximum_current_free_disk_fraction),
        "native_threadpools_single_or_undetected": native_threads_measured_one,
        "thread_environment_measured": thread_environment_measured,
        "jobs_n_concurrency_reached": int(
            scheduler_n.get("maximum_observed_concurrency", -1)
        )
        == int(jobs_n),
        "worker_boundary_equivalent_and_bounded": len(worker_boundary_digests) == 1
        and all(
            int(row.get("worker_full_tree_verifications", -1)) == 0
            and int(row.get("worker_input_regular_files_verified", -1)) == 1
            for row in telemetry_rows
        ),
    }
    return {
        "schema_version": "nursery-corrective-parallel-benchmark-v2",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "jobs_1_wall_seconds": float(wall_jobs_one),
        "jobs_n_wall_seconds": float(wall_jobs_n),
        "jobs_n": int(jobs_n),
        "measured_speedup": speedup,
        "minimum_speedup": float(minimum_speedup),
        "wall_time_contingency_multiplier": float(
            wall_time_contingency_multiplier
        ),
        "micro_unit_count": int(micro_unit_count),
        "development_unit_count": int(development_unit_count),
        "micro_corpus_count": int(micro_corpus_count),
        "development_corpus_count": int(development_corpus_count),
        "input_generation_micro_wall_seconds": float(input_generation_wall),
        "final_assembly_probe_micro_wall_seconds": float(final_assembly_probe_wall),
        "micro_parallel_batch_count": micro_parallel_batch_count,
        "development_parallel_batch_count": development_parallel_batch_count,
        "maximum_jobs_n_worker_wall_seconds": maximum_worker_wall,
        "projected_development_input_generation_seconds": projected_input_seconds,
        "projected_development_worker_wave_seconds": projected_worker_wave_seconds,
        "projected_development_full_cohort_by_unit_scaling_seconds": (
            projected_full_cohort_seconds
        ),
        "additional_max_worker_wave_stress_allowance_seconds": (
            additional_worker_stress_seconds
        ),
        "full_cohort_projection_includes_parent_merge_scoring_adjudication": True,
        "projected_development_final_assembly_seconds": projected_assembly_seconds,
        "estimated_development_compute_and_cohort_seconds_with_contingency": estimated_seconds,
        "estimated_development_compute_and_cohort_hours": estimated_seconds / 3600.0,
        "maximum_worker_peak_rss_native_units": max_rss_native,
        "maximum_worker_peak_rss_bytes": max_rss_bytes,
        "conservative_concurrent_peak_rss_bytes": concurrent_rss_bytes,
        "parent_peak_rss_bytes": parent_rss_bytes,
        "parent_memory_projection_method": parent_memory_projection_method,
        "parent_memory_unit_scaling_factor": unit_scaling,
        "projected_development_parent_peak_rss_bytes": (
            projected_parent_rss_bytes
        ),
        "resource_contingency_multiplier": float(resource_contingency_multiplier),
        "host_physical_memory_bytes": physical_memory_bytes,
        "host_available_memory_bytes_at_benchmark": available_memory_bytes,
        "maximum_host_ram_fraction": float(maximum_host_ram_fraction),
        "maximum_current_available_ram_fraction": float(
            maximum_current_available_ram_fraction
        ),
        "logical_cpu_count": logical_cpu_count,
        "minimum_logical_cpu_count": int(minimum_logical_cpu_count),
        "jobs_1_disk_bytes": _directory_bytes(left),
        "jobs_n_disk_bytes": jobs_n_disk_bytes,
        "micro_persisted_input_disk_bytes": input_disk_bytes,
        "projected_development_disk_bytes": projected_development_disk_bytes,
        "current_free_disk_bytes": disk_free_bytes,
        "maximum_current_free_disk_fraction": float(maximum_current_free_disk_fraction),
        "native_threadpool_counts_observed": native_thread_counts,
        "native_threadpool_detection_count": len(native_thread_counts),
        "nested_blas_openmp_threads": 1 if native_threads_measured_one else None,
        "resource_telemetry_excluded_from_equivalence": True,
        "scheduling": "one_model_wave_across_distinct_corpora",
        "jobs_1_complete_manifest_sha256": sha256_file(
            left / "complete_manifest.json"
        ),
        "jobs_n_complete_manifest_sha256": sha256_file(
            right / "complete_manifest.json"
        ),
        "input_manifest_sha256": sha256_file(inputs / "input_manifest.json"),
        "jobs_1_runtime_telemetry_sha256": sha256_file(
            left / "runtime/runtime_telemetry.json"
        ),
        "jobs_n_runtime_telemetry_sha256": sha256_file(
            right / "runtime/runtime_telemetry.json"
        ),
        "jobs_1_scheduler_validation_sha256": sha256_file(
            left / "runtime/scheduler_validation.json"
        ),
        "jobs_n_scheduler_validation_sha256": sha256_file(
            right / "runtime/scheduler_validation.json"
        ),
    }


def validate_benchmark_report(
    report: Mapping[str, Any],
    *,
    jobs_one_root: str | Path,
    jobs_n_root: str | Path,
    input_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    left = Path(jobs_one_root).resolve()
    right = Path(jobs_n_root).resolve()
    inputs = Path(input_root).resolve()
    expected_fields = {
        "schema_version",
        "status",
        "gates",
        "jobs_1_wall_seconds",
        "jobs_n_wall_seconds",
        "jobs_n",
        "measured_speedup",
        "minimum_speedup",
        "wall_time_contingency_multiplier",
        "micro_unit_count",
        "development_unit_count",
        "micro_corpus_count",
        "development_corpus_count",
        "input_generation_micro_wall_seconds",
        "final_assembly_probe_micro_wall_seconds",
        "micro_parallel_batch_count",
        "development_parallel_batch_count",
        "maximum_jobs_n_worker_wall_seconds",
        "projected_development_input_generation_seconds",
        "projected_development_worker_wave_seconds",
        "projected_development_full_cohort_by_unit_scaling_seconds",
        "additional_max_worker_wave_stress_allowance_seconds",
        "full_cohort_projection_includes_parent_merge_scoring_adjudication",
        "projected_development_final_assembly_seconds",
        "estimated_development_compute_and_cohort_seconds_with_contingency",
        "estimated_development_compute_and_cohort_hours",
        "maximum_worker_peak_rss_native_units",
        "maximum_worker_peak_rss_bytes",
        "conservative_concurrent_peak_rss_bytes",
        "parent_peak_rss_bytes",
        "parent_memory_projection_method",
        "parent_memory_unit_scaling_factor",
        "projected_development_parent_peak_rss_bytes",
        "resource_contingency_multiplier",
        "host_physical_memory_bytes",
        "host_available_memory_bytes_at_benchmark",
        "maximum_host_ram_fraction",
        "maximum_current_available_ram_fraction",
        "logical_cpu_count",
        "minimum_logical_cpu_count",
        "jobs_1_disk_bytes",
        "jobs_n_disk_bytes",
        "micro_persisted_input_disk_bytes",
        "projected_development_disk_bytes",
        "current_free_disk_bytes",
        "maximum_current_free_disk_fraction",
        "native_threadpool_counts_observed",
        "native_threadpool_detection_count",
        "nested_blas_openmp_threads",
        "resource_telemetry_excluded_from_equivalence",
        "scheduling",
        "jobs_1_complete_manifest_sha256",
        "jobs_n_complete_manifest_sha256",
        "input_manifest_sha256",
        "jobs_1_runtime_telemetry_sha256",
        "jobs_n_runtime_telemetry_sha256",
        "jobs_1_scheduler_validation_sha256",
        "jobs_n_scheduler_validation_sha256",
    }
    micro_units = (
        len(config["resolved_registries"]["construction_micro"]["corpus"])
        * len(config["resolved_registries"]["construction_micro"]["model"])
    )
    development_units = (
        len(config["resolved_registries"]["development"]["corpus"])
        * len(config["resolved_registries"]["development"]["model"])
    )
    speedup = float(report.get("jobs_1_wall_seconds", 0.0)) / float(
        report.get("jobs_n_wall_seconds", float("inf"))
    )
    unit_scaling = development_units / micro_units
    corpus_scaling = len(
        config["resolved_registries"]["development"]["corpus"]
    ) / len(config["resolved_registries"]["construction_micro"]["corpus"])
    micro_model_count = len(
        config["resolved_registries"]["construction_micro"]["model"]
    )
    development_model_count = len(
        config["resolved_registries"]["development"]["model"]
    )
    jobs_n = int(config["parallel"]["frozen_jobs"])
    micro_parallel_batch_count = int(
        micro_model_count
        * math.ceil(
            len(config["resolved_registries"]["construction_micro"]["corpus"])
            / jobs_n
        )
    )
    development_parallel_batch_count = int(
        development_model_count
        * math.ceil(
            len(config["resolved_registries"]["development"]["corpus"])
            / jobs_n
        )
    )
    telemetry_one = read_json(left / "runtime/runtime_telemetry.json")
    telemetry_n = read_json(right / "runtime/runtime_telemetry.json")
    scheduler_one = read_json(left / "runtime/scheduler_validation.json")
    scheduler_n = read_json(right / "runtime/scheduler_validation.json")
    maximum_worker_wall = max(
        float(row["wall_seconds"]) for row in telemetry_n["units"]
    )
    projected_input_seconds = float(
        report.get("input_generation_micro_wall_seconds", 0.0)
    ) * corpus_scaling
    projected_worker_wave_seconds = (
        development_parallel_batch_count * maximum_worker_wall
    )
    projected_full_cohort_seconds = float(
        report.get("jobs_n_wall_seconds", 0.0)
    ) * unit_scaling
    additional_worker_stress_seconds = projected_worker_wave_seconds
    projected_assembly_seconds = float(
        report.get("final_assembly_probe_micro_wall_seconds", 0.0)
    ) * unit_scaling
    estimated_seconds = (
        projected_input_seconds
        + projected_full_cohort_seconds
        + additional_worker_stress_seconds
        + projected_assembly_seconds
    ) * float(config["parallel"]["wall_time_contingency_multiplier"])
    from .parallel import build_work_plan

    expected_plan = build_work_plan(config, purpose="construction_micro")
    expected_unit_ids = [str(row["unit_id"]) for row in expected_plan["units"]]
    expected_unit_fields = {
        "unit_id",
        "shard_manifest_sha256",
        "wall_seconds",
        "worker_peak_rss_native_units",
        "bootstrap_environment_digest",
        "worker_boundary_digest",
        "worker_boundary_operational_file_count",
        "worker_input_regular_files_verified",
        "worker_full_tree_verifications",
        "thread_environment_observed",
        "native_threadpools",
        "python_startup_flags_observed",
    }
    expected_telemetry_fields = {
        "schema_version",
        "jobs",
        "units",
        "wall_seconds_sum",
        "maximum_worker_peak_rss_native_units",
        "excluded_from_adjudication",
    }
    def validate_telemetry(
        telemetry: Mapping[str, Any],
        root: Path,
        expected_jobs: int,
    ) -> bool:
        units = list(telemetry.get("units", []))
        ids = [str(row.get("unit_id")) for row in units]
        return (
            set(telemetry) == expected_telemetry_fields
            and telemetry.get("schema_version")
            == "nursery-corrective-runtime-v2"
            and int(telemetry.get("jobs", -1)) == int(expected_jobs)
            and telemetry.get("excluded_from_adjudication") is True
            and ids == sorted(expected_unit_ids)
            and len(ids) == len(set(ids))
            and all(set(row) == expected_unit_fields for row in units)
            and all(
                math.isfinite(float(row["wall_seconds"]))
                and float(row["wall_seconds"]) > 0.0
                and int(row["worker_peak_rss_native_units"]) > 0
                and isinstance(row["bootstrap_environment_digest"], str)
                and len(row["bootstrap_environment_digest"]) == 64
                and isinstance(row["worker_boundary_digest"], str)
                and len(row["worker_boundary_digest"]) == 64
                and int(row["worker_boundary_operational_file_count"]) > 0
                and int(row["worker_input_regular_files_verified"]) == 1
                and int(row["worker_full_tree_verifications"]) == 0
                and row["thread_environment_observed"]
                == {
                    str(name): str(value)
                    for name, value in config["parallel"][
                        "thread_environment"
                    ].items()
                }
                and isinstance(row["native_threadpools"], list)
                and row["python_startup_flags_observed"]
                == config["environment"]["worker_python_startup_flags"]
                and all(
                    set(pool) == {"internal_api", "prefix", "num_threads"}
                    and int(pool["num_threads"]) > 0
                    for pool in row["native_threadpools"]
                )
                and row["shard_manifest_sha256"]
                == sha256_file(
                    root / "shards" / str(row["unit_id"]) / "manifest.json"
                )
                for row in units
            )
            and float(telemetry.get("wall_seconds_sum", -1.0))
            == float(sum(float(row["wall_seconds"]) for row in units))
            and int(telemetry.get("maximum_worker_peak_rss_native_units", -1))
            == max(int(row["worker_peak_rss_native_units"]) for row in units)
        )
    telemetry_validations = {
        "jobs_1": validate_telemetry(telemetry_one, left, 1),
        "jobs_n": validate_telemetry(
            telemetry_n,
            right,
            int(config["parallel"]["frozen_jobs"]),
        ),
    }
    telemetry_rows = [
        *telemetry_one.get("units", []),
        *telemetry_n.get("units", []),
    ]
    max_rss_native = max(
        int(telemetry_one["maximum_worker_peak_rss_native_units"]),
        int(telemetry_n["maximum_worker_peak_rss_native_units"]),
    )
    max_rss_bytes = max_rss_native if sys.platform == "darwin" else max_rss_native * 1024
    native_thread_counts = [
        int(pool["num_threads"])
        for row in telemetry_rows
        for pool in row.get("native_threadpools", [])
    ]
    native_threads_measured_one = all(value == 1 for value in native_thread_counts)
    worker_boundary_digests = {
        str(row.get("worker_boundary_digest")) for row in telemetry_rows
    }
    expected_thread_environment = {
        str(name): str(value)
        for name, value in config["parallel"]["thread_environment"].items()
    }
    thread_environment_measured = bool(telemetry_rows) and all(
        row.get("thread_environment_observed") == expected_thread_environment
        for row in telemetry_rows
    )
    jobs_one_disk_bytes = _directory_bytes(left)
    jobs_n_disk_bytes = _directory_bytes(right)
    input_disk_bytes = _directory_bytes(inputs)
    projected_development_disk_bytes = int(
        math.ceil(
            (
                jobs_n_disk_bytes * unit_scaling
                + input_disk_bytes * corpus_scaling
            )
            * float(config["parallel"]["resource_contingency_multiplier"])
        )
    )
    parent_peak_rss_bytes = int(report.get("parent_peak_rss_bytes", -1))
    projected_parent_rss_bytes = int(
        math.ceil(parent_peak_rss_bytes * development_units / micro_units)
    )
    conservative_rss = int(
        math.ceil(
            (
                max_rss_bytes * int(config["parallel"]["frozen_jobs"])
                + projected_parent_rss_bytes
            )
            * float(config["parallel"]["resource_contingency_multiplier"])
        )
    )
    expected_gates = {
        "speedup": speedup
        >= float(config["parallel"]["minimum_benchmark_speedup"]),
        "logical_cpu_count": int(report.get("logical_cpu_count", -1))
        >= int(config["parallel"]["minimum_logical_cpu_count"]),
        "ram_headroom": conservative_rss
        <= int(report.get("host_physical_memory_bytes", -1))
        * float(config["parallel"]["maximum_host_ram_fraction"])
        and conservative_rss
        <= int(report.get("host_available_memory_bytes_at_benchmark", -1))
        * float(config["parallel"]["maximum_current_available_ram_fraction"]),
        "disk_headroom": projected_development_disk_bytes
        <= int(report.get("current_free_disk_bytes", -1))
        * float(config["parallel"]["maximum_current_free_disk_fraction"]),
        "native_threadpools_single_or_undetected": native_threads_measured_one,
        "thread_environment_measured": thread_environment_measured,
        "jobs_n_concurrency_reached": int(
            scheduler_n.get("maximum_observed_concurrency", -1)
        )
        == jobs_n,
        "worker_boundary_equivalent_and_bounded": len(worker_boundary_digests) == 1
        and all(
            int(row.get("worker_full_tree_verifications", -1)) == 0
            and int(row.get("worker_input_regular_files_verified", -1)) == 1
            for row in telemetry_rows
        ),
    }
    hashes = {
        "jobs_1_complete_manifest_sha256": sha256_file(
            left / "complete_manifest.json"
        ),
        "jobs_n_complete_manifest_sha256": sha256_file(
            right / "complete_manifest.json"
        ),
        "input_manifest_sha256": sha256_file(inputs / "input_manifest.json"),
        "jobs_1_runtime_telemetry_sha256": sha256_file(
            left / "runtime/runtime_telemetry.json"
        ),
        "jobs_n_runtime_telemetry_sha256": sha256_file(
            right / "runtime/runtime_telemetry.json"
        ),
        "jobs_1_scheduler_validation_sha256": sha256_file(
            left / "runtime/scheduler_validation.json"
        ),
        "jobs_n_scheduler_validation_sha256": sha256_file(
            right / "runtime/scheduler_validation.json"
        ),
    }
    checks = {
        "schema": set(report) == expected_fields
        and report.get("schema_version")
        == "nursery-corrective-parallel-benchmark-v2",
        "jobs": int(report.get("jobs_n", -1))
        == int(config["parallel"]["frozen_jobs"]),
        "counts": int(report.get("micro_unit_count", -1)) == micro_units
        and int(report.get("development_unit_count", -1)) == development_units
        and int(report.get("micro_corpus_count", -1))
        == len(config["resolved_registries"]["construction_micro"]["corpus"])
        and int(report.get("development_corpus_count", -1))
        == len(config["resolved_registries"]["development"]["corpus"]),
        "speedup_formula": float(report.get("measured_speedup", -1.0))
        == speedup
        and all(
            math.isfinite(float(report.get(name, -1.0)))
            and float(report.get(name, -1.0)) > 0.0
            for name in (
                "jobs_1_wall_seconds",
                "jobs_n_wall_seconds",
                "input_generation_micro_wall_seconds",
                "final_assembly_probe_micro_wall_seconds",
            )
        )
        and float(report.get("jobs_1_wall_seconds", 0.0))
        >= max(float(row["wall_seconds"]) for row in telemetry_one["units"])
        and float(report.get("jobs_n_wall_seconds", 0.0))
        >= max(float(row["wall_seconds"]) for row in telemetry_n["units"]),
        "estimate_formula": int(report.get("micro_parallel_batch_count", -1))
        == micro_parallel_batch_count
        and int(report.get("development_parallel_batch_count", -1))
        == development_parallel_batch_count
        and float(report.get("maximum_jobs_n_worker_wall_seconds", -1.0))
        == maximum_worker_wall
        and float(
            report.get("projected_development_input_generation_seconds", -1.0)
        )
        == projected_input_seconds
        and float(report.get("projected_development_worker_wave_seconds", -1.0))
        == projected_worker_wave_seconds
        and float(
            report.get(
                "projected_development_full_cohort_by_unit_scaling_seconds",
                -1.0,
            )
        )
        == projected_full_cohort_seconds
        and float(
            report.get("additional_max_worker_wave_stress_allowance_seconds", -1.0)
        )
        == additional_worker_stress_seconds
        and report.get(
            "full_cohort_projection_includes_parent_merge_scoring_adjudication"
        )
        is True
        and float(
            report.get("projected_development_final_assembly_seconds", -1.0)
        )
        == projected_assembly_seconds
        and float(
            report.get(
                "estimated_development_compute_and_cohort_seconds_with_contingency",
                -1.0,
            )
        )
        == estimated_seconds
        and float(
            report.get("estimated_development_compute_and_cohort_hours", -1.0)
        )
        == estimated_seconds / 3600.0,
        "frozen_thresholds": float(report.get("minimum_speedup", -1.0))
        == float(config["parallel"]["minimum_benchmark_speedup"])
        and float(report.get("wall_time_contingency_multiplier", -1.0))
        == float(config["parallel"]["wall_time_contingency_multiplier"])
        and float(report.get("resource_contingency_multiplier", -1.0))
        == float(config["parallel"]["resource_contingency_multiplier"])
        and float(report.get("maximum_host_ram_fraction", -1.0))
        == float(config["parallel"]["maximum_host_ram_fraction"])
        and float(report.get("maximum_current_available_ram_fraction", -1.0))
        == float(config["parallel"]["maximum_current_available_ram_fraction"])
        and float(report.get("maximum_current_free_disk_fraction", -1.0))
        == float(config["parallel"]["maximum_current_free_disk_fraction"])
        and int(report.get("minimum_logical_cpu_count", -1))
        == int(config["parallel"]["minimum_logical_cpu_count"]),
        "telemetry_derived_resources": int(
            report.get("maximum_worker_peak_rss_native_units", -1)
        )
        == max_rss_native
        and int(report.get("maximum_worker_peak_rss_bytes", -1)) == max_rss_bytes
        and parent_peak_rss_bytes > 0
        and report.get("parent_memory_projection_method")
        == config["parallel"]["parent_memory_projection_method"]
        == _PARENT_MEMORY_PROJECTION_METHOD
        and float(report.get("parent_memory_unit_scaling_factor", -1.0))
        == unit_scaling
        and int(
            report.get("projected_development_parent_peak_rss_bytes", -1)
        )
        == projected_parent_rss_bytes
        and int(report.get("conservative_concurrent_peak_rss_bytes", -1))
        == conservative_rss
        and report.get("native_threadpool_counts_observed")
        == native_thread_counts
        and int(report.get("native_threadpool_detection_count", -1))
        == len(native_thread_counts)
        and report.get("nested_blas_openmp_threads")
        == (1 if native_threads_measured_one else None),
        "telemetry_schema_and_units": all(telemetry_validations.values()),
        "disk_formulas": int(report.get("jobs_1_disk_bytes", -1))
        == jobs_one_disk_bytes
        and int(report.get("jobs_n_disk_bytes", -1)) == jobs_n_disk_bytes
        and int(report.get("micro_persisted_input_disk_bytes", -1))
        == input_disk_bytes
        and int(report.get("projected_development_disk_bytes", -1))
        == projected_development_disk_bytes,
        "host_snapshots_plausible": int(
            report.get("host_physical_memory_bytes", -1)
        )
        > 0
        and int(report.get("host_available_memory_bytes_at_benchmark", -1)) > 0
        and int(report.get("current_free_disk_bytes", -1)) > 0
        and int(report.get("logical_cpu_count", -1)) > 0,
        "artifact_hashes": all(report.get(name) == value for name, value in hashes.items()),
        "scheduler_evidence": scheduler_one.get("status") == "PASS"
        and scheduler_n.get("status") == "PASS"
        and int(scheduler_one.get("jobs", -1)) == 1
        and int(scheduler_n.get("jobs", -1)) == jobs_n
        and int(scheduler_n.get("maximum_observed_concurrency", -1)) == jobs_n,
        "root_manifests": verify_exact_manifest(
            left,
            read_json(left / "complete_manifest.json"),
            manifest_filename="complete_manifest.json",
        )["status"]
        == "PASS"
        and verify_exact_manifest(
            right,
            read_json(right / "complete_manifest.json"),
            manifest_filename="complete_manifest.json",
        )["status"]
        == "PASS",
        "gates_and_status": report.get("gates") == expected_gates
        and report.get("status")
        == ("PASS" if all(expected_gates.values()) else "FAIL")
        and all(expected_gates.values()),
        "telemetry_exclusion": report.get(
            "resource_telemetry_excluded_from_equivalence"
        )
        is True,
        "scheduling": report.get("scheduling")
        == "one_model_wave_across_distinct_corpora",
    }
    return {
        "schema_version": "nursery-corrective-benchmark-validation-v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "artifact_hashes": hashes,
    }


def _hash_preserved_row(arguments: tuple[Path, str, str, int, int]) -> dict[str, Any]:
    root, relative, expected_hash, expected_size, expected_mode = arguments
    path = root / relative.removeprefix("./")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"relative": relative, "problem": "missing"}
    if not stat.S_ISREG(metadata.st_mode):
        return {"relative": relative, "problem": "not_regular"}
    observed_hash = sha256_file(path)
    observed_size = int(metadata.st_size)
    observed_mode = stat.S_IMODE(metadata.st_mode)
    problems = []
    if observed_hash != expected_hash:
        problems.append("sha256")
    if observed_size != expected_size:
        problems.append("size")
    if observed_mode != expected_mode:
        problems.append("mode")
    return {
        "relative": relative,
        "sha256": observed_hash,
        "size": observed_size,
        "mode": observed_mode,
        "problems": problems,
    }


def _preservation_anchor_checks(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, bool]:
    audit_manifest = (
        root
        / "output/synthetic_confirmation_adversarial_audit_v1/complete_file_manifest.json"
    )
    return {
        "baseline": sha256_file(
            root / str(config["paths"]["preservation_baseline"])
        )
        == str(config["paths"]["preservation_baseline_sha256"]),
        "stat": sha256_file(root / str(config["paths"]["preservation_stat"]))
        == str(config["paths"]["preservation_stat_sha256"]),
        "symlinks": sha256_file(
            root / str(config["paths"]["preservation_symlinks"])
        )
        == str(config["paths"]["preservation_symlinks_sha256"]),
        "audit_manifest": sha256_file(audit_manifest)
        == str(config["paths"]["preservation_audit_manifest_sha256"]),
    }


def verify_prior_preservation(
    repository_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    jobs: int = 4,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    require_lstat_absent(output)
    output.mkdir(parents=True)
    baseline_hash_path = root / str(config["paths"]["preservation_baseline"])
    baseline_stat_path = root / str(config["paths"]["preservation_stat"])
    baseline_symlink_path = root / str(config["paths"]["preservation_symlinks"])
    anchor_checks = _preservation_anchor_checks(root, config)
    hash_rows = []
    for line in baseline_hash_path.read_text(encoding="utf-8").splitlines():
        if line:
            digest, relative = line.split("  ", 1)
            hash_rows.append((digest, relative))
    stat_rows = {}
    for line in baseline_stat_path.read_text(encoding="utf-8").splitlines():
        if line:
            size, mode, relative = line.split(" ", 2)
            stat_rows[relative] = (int(size), int(mode, 8))
    if set(relative for _, relative in hash_rows) != set(stat_rows):
        raise RuntimeError("preservation hash/stat path sets differ")
    work = [
        (root, relative, digest, stat_rows[relative][0], stat_rows[relative][1])
        for digest, relative in hash_rows
    ]
    with ThreadPoolExecutor(max_workers=int(jobs)) as executor:
        observed = list(executor.map(_hash_preserved_row, work))
    problems = [row for row in observed if row.get("problem") or row.get("problems")]
    after_hash_lines = [
        f"{row.get('sha256', '')}  {row['relative']}" for row in observed
    ]
    after_stat_by_relative = {
        row["relative"]: f"{row.get('size', -1)} {format(int(row.get('mode', 0)), 'o')} {row['relative']}"
        for row in observed
    }
    after_stat_lines = [
        after_stat_by_relative[line.split(" ", 2)[2]]
        for line in baseline_stat_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    after_hash_bytes = ("\n".join(after_hash_lines) + "\n").encode()
    after_stat_bytes = ("\n".join(after_stat_lines) + "\n").encode()
    atomic_write_bytes(output / "preservation_after.sha256", after_hash_bytes)
    atomic_write_bytes(output / "preservation_after.stat", after_stat_bytes)
    symlink_lines = []
    symlink_problems = []
    for line in baseline_symlink_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        relative, expected_target = line.split("\t", 1)
        path = root / relative.removeprefix("./")
        try:
            metadata = path.lstat()
            target = os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else ""
        except FileNotFoundError:
            target = ""
        if target != expected_target:
            symlink_problems.append(relative)
        symlink_lines.append(f"{relative}\t{target}")
    after_symlink_bytes = ("\n".join(symlink_lines) + "\n").encode()
    atomic_write_bytes(output / "preservation_after.symlinks", after_symlink_bytes)
    audit_root = root / "output/synthetic_confirmation_adversarial_audit_v1"
    audit_manifest = read_json(audit_root / "complete_file_manifest.json")
    audit_verification = verify_exact_manifest(
        audit_root,
        audit_manifest,
        manifest_filename="complete_file_manifest.json",
    )
    audit_problems = list(audit_verification.get("problems", []))
    passed = (
        all(anchor_checks.values())
        and
        not problems
        and not symlink_problems
        and not audit_problems
        and after_hash_bytes == baseline_hash_path.read_bytes()
        and after_stat_bytes == baseline_stat_path.read_bytes()
        and after_symlink_bytes == baseline_symlink_path.read_bytes()
    )
    proof = {
        "schema_version": "nursery-corrective-preservation-proof-v2",
        "status": "PASS" if passed else "FAIL",
        "baseline_regular_file_count": len(hash_rows),
        "verified_regular_file_count": len(observed) - len(problems),
        "baseline_symlink_count": len(symlink_lines),
        "regular_file_problems": problems,
        "symlink_problems": symlink_problems,
        "baseline_hash_manifest_sha256": sha256_file(baseline_hash_path),
        "after_hash_manifest_sha256": sha256_file(
            output / "preservation_after.sha256"
        ),
        "baseline_stat_manifest_sha256": sha256_file(baseline_stat_path),
        "after_stat_manifest_sha256": sha256_file(
            output / "preservation_after.stat"
        ),
        "baseline_symlink_manifest_sha256": sha256_file(baseline_symlink_path),
        "after_symlink_manifest_sha256": sha256_file(
            output / "preservation_after.symlinks"
        ),
        "baseline_manifests_byte_identical": (
            after_hash_bytes == baseline_hash_path.read_bytes()
            and after_stat_bytes == baseline_stat_path.read_bytes()
            and after_symlink_bytes == baseline_symlink_path.read_bytes()
        ),
        "adversarial_audit_manifest_file_count": int(
            audit_manifest["file_count"]
        ),
        "adversarial_audit_manifest_digest": audit_manifest["digest"],
        "adversarial_audit_changed_rows": audit_problems,
        "old_artifacts_preserved_byte_for_byte": passed,
        "baseline_hash_anchor_match": anchor_checks["baseline"],
        "baseline_stat_anchor_match": anchor_checks["stat"],
        "baseline_symlink_anchor_match": anchor_checks["symlinks"],
        "adversarial_audit_manifest_anchor_match": anchor_checks[
            "audit_manifest"
        ],
    }
    write_json(output / "preservation_proof.json", proof)
    return proof


def reverify_prior_preservation(
    repository_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    jobs: int = 4,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    baseline_hash_path = root / str(config["paths"]["preservation_baseline"])
    baseline_stat_path = root / str(config["paths"]["preservation_stat"])
    baseline_symlink_path = root / str(config["paths"]["preservation_symlinks"])
    anchor_checks = _preservation_anchor_checks(root, config)
    hash_rows = []
    for line in baseline_hash_path.read_text(encoding="utf-8").splitlines():
        if line:
            digest, relative = line.split("  ", 1)
            hash_rows.append((digest, relative))
    stat_rows = {}
    for line in baseline_stat_path.read_text(encoding="utf-8").splitlines():
        if line:
            size, mode, relative = line.split(" ", 2)
            stat_rows[relative] = (int(size), int(mode, 8))
    if {relative for _, relative in hash_rows} != set(stat_rows):
        return {"status": "FAIL", "problems": ["baseline_path_set_mismatch"]}
    work = [
        (root, relative, digest, stat_rows[relative][0], stat_rows[relative][1])
        for digest, relative in hash_rows
    ]
    with ThreadPoolExecutor(max_workers=int(jobs)) as executor:
        observed = list(executor.map(_hash_preserved_row, work))
    problems = [row for row in observed if row.get("problem") or row.get("problems")]
    after_hash_bytes = (
        "\n".join(
            f"{row.get('sha256', '')}  {row['relative']}" for row in observed
        )
        + "\n"
    ).encode()
    after_stat_by_relative = {
        row["relative"]: (
            f"{row.get('size', -1)} {format(int(row.get('mode', 0)), 'o')} "
            f"{row['relative']}"
        )
        for row in observed
    }
    after_stat_bytes = (
        "\n".join(
            after_stat_by_relative[line.split(" ", 2)[2]]
            for line in baseline_stat_path.read_text(encoding="utf-8").splitlines()
            if line
        )
        + "\n"
    ).encode()
    symlink_lines = []
    symlink_problems = []
    for line in baseline_symlink_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        relative, expected_target = line.split("\t", 1)
        path = root / relative.removeprefix("./")
        try:
            metadata = path.lstat()
            target = os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else ""
        except FileNotFoundError:
            target = ""
        if target != expected_target:
            symlink_problems.append(relative)
        symlink_lines.append(f"{relative}\t{target}")
    after_symlink_bytes = ("\n".join(symlink_lines) + "\n").encode()
    audit_root = root / "output/synthetic_confirmation_adversarial_audit_v1"
    audit_manifest = read_json(audit_root / "complete_file_manifest.json")
    audit_verification = verify_exact_manifest(
        audit_root,
        audit_manifest,
        manifest_filename="complete_file_manifest.json",
    )
    audit_problems = list(audit_verification.get("problems", []))
    after_hash_path = output / "preservation_after.sha256"
    after_stat_path = output / "preservation_after.stat"
    after_symlink_path = output / "preservation_after.symlinks"
    persisted_proof = read_json(output / "preservation_proof.json")
    expected_proof = {
        "schema_version": "nursery-corrective-preservation-proof-v2",
        "status": "PASS",
        "baseline_regular_file_count": len(hash_rows),
        "verified_regular_file_count": len(observed) - len(problems),
        "baseline_symlink_count": len(symlink_lines),
        "regular_file_problems": problems,
        "symlink_problems": symlink_problems,
        "baseline_hash_manifest_sha256": sha256_file(baseline_hash_path),
        "after_hash_manifest_sha256": sha256_file(after_hash_path),
        "baseline_stat_manifest_sha256": sha256_file(baseline_stat_path),
        "after_stat_manifest_sha256": sha256_file(after_stat_path),
        "baseline_symlink_manifest_sha256": sha256_file(baseline_symlink_path),
        "after_symlink_manifest_sha256": sha256_file(after_symlink_path),
        "baseline_manifests_byte_identical": True,
        "adversarial_audit_manifest_file_count": int(audit_manifest["file_count"]),
        "adversarial_audit_manifest_digest": audit_manifest["digest"],
        "adversarial_audit_changed_rows": audit_problems,
        "old_artifacts_preserved_byte_for_byte": True,
        "baseline_hash_anchor_match": True,
        "baseline_stat_anchor_match": True,
        "baseline_symlink_anchor_match": True,
        "adversarial_audit_manifest_anchor_match": True,
    }
    checks = {
        "prospective_anchors": all(anchor_checks.values()),
        "regular_files": not problems
        and after_hash_bytes == baseline_hash_path.read_bytes()
        and after_hash_path.read_bytes() == after_hash_bytes,
        "stat_rows": after_stat_bytes == baseline_stat_path.read_bytes()
        and after_stat_path.read_bytes() == after_stat_bytes,
        "symlinks": not symlink_problems
        and after_symlink_bytes == baseline_symlink_path.read_bytes()
        and after_symlink_path.read_bytes() == after_symlink_bytes,
        "adversarial_audit_manifest": audit_verification["status"] == "PASS",
        "persisted_proof_exact": persisted_proof == expected_proof,
    }
    return {
        "schema_version": "nursery-corrective-preservation-reverification-v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_proof_digest": canonical_digest(expected_proof),
        "persisted_proof_digest": canonical_digest(persisted_proof),
    }


def run_tests(
    repository_root: str | Path,
    output_path: str | Path,
    executable: str | Path,
    *,
    full_repository_root: str | Path | None = None,
    persist: bool = True,
    binding_package_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    full_root = (
        Path(full_repository_root).resolve()
        if full_repository_root is not None
        else root
    )
    evidence_package = (
        Path(binding_package_root).resolve()
        if binding_package_root is not None
        else Path(output_path).resolve().parent
    )
    if persist:
        require_lstat_absent(output_path)
    commands = [
        (
            root,
            [
            str(executable),
            "-m",
            "pytest",
            "-q",
            "-rA",
            "-p",
            "no:cacheprovider",
            "tests/test_synthetic_corrective_alignment_v2.py",
            ],
        ),
        (
            full_root,
            [
            str(executable),
            "-m",
            "pytest",
            "-q",
            "-rA",
            "-p",
            "no:cacheprovider",
            "--deselect",
            CLOSED_PROTOCOL_DESELECTION,
            "tests",
            ],
        ),
    ]
    results = []
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
        }
    )
    for cwd, command in commands:
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
                "command": command,
                "cwd": str(cwd),
                "exit_code": completed.returncode,
                "output": completed.stdout,
            }
        )
    passed_nodeids = sorted(
        {
            line.removeprefix("PASSED ").strip()
            for line in results[0]["output"].splitlines()
            if line.startswith("PASSED ")
        }
    )
    passed_names = {
        node.split("::", 1)[1].split("[", 1)[0]
        for node in passed_nodeids
        if "::" in node
    }
    required_groups = REQUIRED_CORRECTIVE_TEST_GROUPS
    group_rows = {
        name: {
            "required_tests": required,
            "passed_tests": sorted(set(required) & passed_names),
            "missing_or_not_passed": sorted(set(required) - passed_names),
            "status": "PASS" if set(required) <= passed_names else "FAIL",
        }
        for name, required in required_groups.items()
    }
    execution_passed = all(row["exit_code"] == 0 for row in results)
    classified_names = {
        name for required in required_groups.values() for name in required
    }
    unclassified_passed_tests = sorted(passed_names - classified_names)
    inventory_passed = all(
        row["status"] == "PASS" for row in group_rows.values()
    ) and not unclassified_passed_tests
    deselection_counts = [
        int(value)
        for value in re.findall(r"(?<!\d)(\d+) deselected", results[1]["output"])
    ]
    observed_deselected_count = max(deselection_counts, default=0)
    deselected_source = full_root / CLOSED_PROTOCOL_DESELECTION.split("::", 1)[0]
    deselection_contract = [
        {
            "nodeid": CLOSED_PROTOCOL_DESELECTION,
            "reason": CLOSED_PROTOCOL_DESELECTION_RATIONALE,
            "source_sha256": sha256_file(deselected_source),
            "authoritative_old_output_exists": (
                full_root / "output/synthetic_development_v3_one_shot"
            ).is_dir(),
        }
    ]
    deselection_passed = (
        observed_deselected_count == 1
        and len(deselection_contract) == 1
        and deselection_contract[0]["authoritative_old_output_exists"] is True
    )
    report = {
        "schema_version": "nursery-corrective-official-tests-v2",
        "status": (
            "PASS"
            if execution_passed and inventory_passed and deselection_passed
            else "FAIL"
        ),
        "executions": results,
        "passed_nodeids": passed_nodeids,
        "passed_test_count": len(passed_nodeids),
        "required_test_groups": group_rows,
        "required_inventory_status": "PASS" if inventory_passed else "FAIL",
        "closed_protocol_deselections": deselection_contract,
        "observed_deselected_count": observed_deselected_count,
        "deselection_contract_status": "PASS" if deselection_passed else "FAIL",
        "unclassified_passed_tests": unclassified_passed_tests,
        "evidence_bindings": {
            "snapshot_root": str(root),
            "full_repository_root": str(full_root),
            "snapshot_manifest_sha256": sha256_file(
                root / "snapshot_manifest.json"
            ),
            "config_sha256": sha256_file(
                root / "configs/synthetic_corrective_alignment_v2.yaml"
            ),
            "runner_sha256": sha256_file(
                root / "scripts/run_synthetic_corrective_alignment_v2.py"
            ),
            "freeze_receipt_sha256": sha256_file(
                evidence_package / "freeze_receipt.json"
            ),
            "frozen_environment_sha256": sha256_file(
                evidence_package / "frozen_environment.json"
            ),
            "python_executable": str(Path(executable).resolve()),
            "python_sha256": sha256_file(executable),
        },
    }
    if persist:
        write_json(output_path, report)
    return report


def validate_official_test_report(
    report: Mapping[str, Any],
    *,
    snapshot_root: str | Path,
    full_repository_root: str | Path,
    executable: str | Path,
    package_root: str | Path,
) -> dict[str, Any]:
    snapshot = Path(snapshot_root).resolve()
    full_root = Path(full_repository_root).resolve()
    package = Path(package_root).resolve()
    python = Path(executable).resolve()
    expected_fields = {
        "schema_version",
        "status",
        "executions",
        "passed_nodeids",
        "passed_test_count",
        "required_test_groups",
        "required_inventory_status",
        "closed_protocol_deselections",
        "observed_deselected_count",
        "deselection_contract_status",
        "unclassified_passed_tests",
        "evidence_bindings",
    }
    expected_commands = [
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "-rA",
            "-p",
            "no:cacheprovider",
            "tests/test_synthetic_corrective_alignment_v2.py",
        ],
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "-rA",
            "-p",
            "no:cacheprovider",
            "--deselect",
            CLOSED_PROTOCOL_DESELECTION,
            "tests",
        ],
    ]
    executions = list(report.get("executions", []))
    reparsed_nodeids = sorted(
        {
            line.removeprefix("PASSED ").strip()
            for line in (
                executions[0].get("output", "").splitlines()
                if len(executions) == 2
                else []
            )
            if line.startswith("PASSED ")
        }
    )
    passed_names = {
        node.split("::", 1)[1].split("[", 1)[0]
        for node in reparsed_nodeids
        if "::" in node
    }
    groups = report.get("required_test_groups", {})
    expected_group_rows = {
        name: {
            "required_tests": required,
            "passed_tests": sorted(set(required) & passed_names),
            "missing_or_not_passed": sorted(set(required) - passed_names),
            "status": "PASS" if set(required) <= passed_names else "FAIL",
        }
        for name, required in REQUIRED_CORRECTIVE_TEST_GROUPS.items()
    }
    classified_names = {
        test
        for required in REQUIRED_CORRECTIVE_TEST_GROUPS.values()
        for test in required
    }
    unclassified = sorted(passed_names - classified_names)
    deselection_counts = [
        int(value)
        for value in re.findall(
            r"(?<!\d)(\d+) deselected",
            executions[1].get("output", "") if len(executions) == 2 else "",
        )
    ]
    observed_deselected = max(deselection_counts, default=0)
    deselected_source = full_root / CLOSED_PROTOCOL_DESELECTION.split("::", 1)[0]
    expected_deselection = [
        {
            "nodeid": CLOSED_PROTOCOL_DESELECTION,
            "reason": CLOSED_PROTOCOL_DESELECTION_RATIONALE,
            "source_sha256": sha256_file(deselected_source),
            "authoritative_old_output_exists": (
                full_root / "output/synthetic_development_v3_one_shot"
            ).is_dir(),
        }
    ]
    expected_bindings = {
        "snapshot_root": str(snapshot),
        "full_repository_root": str(full_root),
        "snapshot_manifest_sha256": sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        "config_sha256": sha256_file(
            snapshot / "configs/synthetic_corrective_alignment_v2.yaml"
        ),
        "runner_sha256": sha256_file(
            snapshot / "scripts/run_synthetic_corrective_alignment_v2.py"
        ),
        "freeze_receipt_sha256": sha256_file(package / "freeze_receipt.json"),
        "frozen_environment_sha256": sha256_file(
            package / "frozen_environment.json"
        ),
        "python_executable": str(python),
        "python_sha256": sha256_file(python),
    }
    checks = {
        "schema": set(report) == expected_fields
        and report.get("schema_version") == "nursery-corrective-official-tests-v2",
        "executions": len(executions) == 2
        and [row.get("command") for row in executions] == expected_commands
        and [row.get("cwd") for row in executions]
        == [str(snapshot), str(full_root)]
        and all(int(row.get("exit_code", -1)) == 0 for row in executions)
        and all(set(row) == {"command", "cwd", "exit_code", "output"} for row in executions),
        "passed_nodeids": report.get("passed_nodeids") == reparsed_nodeids
        and int(report.get("passed_test_count", -1)) == len(reparsed_nodeids),
        "group_inventory": groups == expected_group_rows
        and expected_group_rows
        and all(row["status"] == "PASS" for row in expected_group_rows.values())
        and report.get("required_inventory_status") == "PASS"
        and report.get("unclassified_passed_tests") == unclassified
        and not unclassified,
        "deselection": report.get("closed_protocol_deselections")
        == expected_deselection
        and observed_deselected == 1
        and int(report.get("observed_deselected_count", -1)) == 1
        and report.get("deselection_contract_status") == "PASS",
        "bindings": report.get("evidence_bindings") == expected_bindings,
        "status": report.get("status") == "PASS",
    }
    return {
        "schema_version": "nursery-corrective-official-tests-validation-v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reparsed_passed_test_count": len(reparsed_nodeids),
        "unclassified_passed_tests": unclassified,
    }


def recompute_comparison_value(
    rehearsal_root: str | Path,
    recompute_root: str | Path,
) -> dict[str, Any]:
    left = Path(rehearsal_root).resolve() / "adjudication"
    right = Path(recompute_root).resolve() / "adjudication"
    left_paths = tree_file_paths(left)
    right_paths = tree_file_paths(right)
    mismatches = []
    if left_paths == right_paths:
        mismatches = [
            relative
            for relative in left_paths
            if (left / relative).read_bytes() != (right / relative).read_bytes()
        ]
    expected_files = {
        "work_plan.json",
        "execution_contract.json",
        "merged_worker_results.json",
        "merged_operation_ledgers.json",
        "parent_operation_ledger.json",
        "parent_ledger_verification.json",
        "scored_condition_units.json",
        "scored_mutation_units.json",
        "model_averages.json",
        "primary_inference.json",
        "present_null_dependence.json",
        "mechanism_mutations.json",
        "informativeness.json",
        "scientific_summary.json",
        "completeness.json",
        "manifest.json",
    }
    expected_file_set = set(left_paths) == expected_files == set(right_paths)
    passed = expected_file_set and left_paths == right_paths and not mismatches
    return {
        "schema_version": "nursery-corrective-recompute-comparison-v2",
        "status": "PASS" if passed else "FAIL",
        "file_sets_identical": left_paths == right_paths,
        "expected_file_set_complete": expected_file_set,
        "adjudication_relevant_files_compared": len(left_paths),
        "byte_identical": passed,
        "mismatches": mismatches,
        "recompute_scientific_outcome": False,
        "recompute_confirmation_outcome": False,
    }


def compare_recompute(
    rehearsal_root: str | Path,
    recompute_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    result = recompute_comparison_value(rehearsal_root, recompute_root)
    write_json(output_path, result)
    return result


def operation_identifier_audit(
    package_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    package = Path(package_root).resolve()
    old_ranges = [tuple(map(int, value)) for value in config["registries"]["permanently_quarantined_ranges"]]
    confirmation = {
        int(value)
        for values in config["resolved_registries"]["confirmation_reserve"].values()
        for value in values
    }
    allowed_by_purpose = {
        purpose: {
            int(value)
            for values in registry.values()
            for value in values
        }
        for purpose, registry in config["resolved_registries"].items()
        if purpose != "confirmation_reserve"
    }
    references = []
    unknown_references = []
    log_paths = []
    ledger_names = {
        "operation_ledger.json",
        "generation_operation_ledger.json",
        "parent_operation_ledger.json",
        "mechanism_qualification_operation_ledger.json",
        "recompute_operation_ledger.json",
    }
    for path in sorted(
        child for child in package.rglob("*.json") if child.name in ledger_names
    ):
        log_paths.append(path.relative_to(package).as_posix())
        ledger = read_json(path)
        purpose = str(ledger.get("purpose"))
        for entry in ledger.get("entries", []):
            for row in entry.get("references", []):
                value = int(row["value"])
                references.append(value)
                if purpose not in allowed_by_purpose or value not in allowed_by_purpose[purpose]:
                    unknown_references.append(
                        {
                            "ledger": path.relative_to(package).as_posix(),
                            "purpose": purpose,
                            "value": value,
                        }
                    )
    old = sorted(
        {value for value in references if any(low <= value <= high for low, high in old_ranges)}
    )
    reserve = sorted(set(references) & confirmation)
    passed = not old and not reserve and not unknown_references
    return {
        "status": "PASS" if passed else "FAIL",
        "operation_log_count": len(log_paths),
        "operation_reference_count": len(references),
        "old_quarantined_references": old,
        "confirmation_reserve_references": reserve,
        "unknown_or_wrong_purpose_references": unknown_references,
        "operation_log_paths": log_paths,
    }


def repository_operation_census(
    repository_root: str | Path,
    package_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    package = Path(package_root).resolve()
    baseline_path = root / str(config["paths"]["preservation_baseline"])
    baseline_files = {
        relative.removeprefix("./")
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line
        for _digest, relative in [line.split("  ", 1)]
    }
    ledger_names = {
        "operation_ledger.json",
        "generation_operation_ledger.json",
        "parent_operation_ledger.json",
        "mechanism_qualification_operation_ledger.json",
        "recompute_operation_ledger.json",
    }
    observed = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "output").rglob("*.json")
        if path.name in ledger_names
    )
    new_ledgers = sorted(set(observed) - baseline_files)
    predecessor_relative = "output/synthetic_corrective_development_launch_package_v1"
    predecessor_manifest_path = package / "PREDECESSOR_V1_FILE_MANIFEST.json"
    predecessor_manifest = read_json(predecessor_manifest_path)
    predecessor_root = root / predecessor_relative
    predecessor_current_manifest = manifest_for_tree(predecessor_root)
    predecessor_manifest_exact = predecessor_current_manifest == predecessor_manifest
    retired_predecessor_ledgers = sorted(
        f"{predecessor_relative}/{row['path']}"
        for row in predecessor_manifest.get("files", [])
        if Path(str(row.get("path", ""))).name in ledger_names
    )
    observed_retired_predecessor_ledgers = sorted(
        set(new_ledgers) & set(retired_predecessor_ledgers)
    )
    active_new_ledgers = sorted(set(new_ledgers) - set(retired_predecessor_ledgers))
    package_relative = package.relative_to(root).as_posix()
    outside_package = [
        path
        for path in active_new_ledgers
        if not path.startswith(f"{package_relative}/")
    ]
    prefreeze_manifest = read_json(package / "prefreeze_evidence_manifest.json")
    prefreeze_ledgers = {
        f"{package_relative}/{row['path']}"
        for row in prefreeze_manifest["files"]
        if Path(str(row["path"])).name in ledger_names
    }
    rehearsal_registry = config["resolved_registries"]["excluded_rehearsal"]
    postfreeze_relative = {
        "excluded_rehearsal/persisted_inputs/generation_operation_ledger.json",
        "excluded_rehearsal/adjudication/parent_operation_ledger.json",
        "independent_recompute/adjudication/parent_operation_ledger.json",
        "independent_recompute/recompute_operation_ledger.json",
        *{
            "excluded_rehearsal/shards/"
            f"corpus-{int(corpus)}__model-{int(model)}/operation_ledger.json"
            for corpus in rehearsal_registry["corpus"]
            for model in rehearsal_registry["model"]
        },
    }
    expected = prefreeze_ledgers | {
        f"{package_relative}/{path}" for path in postfreeze_relative
    }
    missing = sorted(expected - set(active_new_ledgers))
    unexpected = sorted(set(active_new_ledgers) - expected)
    package_audit = operation_identifier_audit(package, config)
    passed = (
        not outside_package
        and not missing
        and not unexpected
        and predecessor_manifest_exact
        and len(retired_predecessor_ledgers) == 19
        and observed_retired_predecessor_ledgers == retired_predecessor_ledgers
        and package_audit["status"] == "PASS"
    )
    return {
        "schema_version": "nursery-corrective-repository-operation-census-v2",
        "status": "PASS" if passed else "FAIL",
        "baseline_manifest_sha256": sha256_file(baseline_path),
        "baseline_file_count": len(baseline_files),
        "new_ledger_count": len(new_ledgers),
        "active_new_ledger_count": len(active_new_ledgers),
        "expected_new_ledger_count": len(expected),
        "new_ledger_paths": new_ledgers,
        "active_new_ledger_paths": active_new_ledgers,
        "retired_predecessor_manifest_sha256": sha256_file(
            predecessor_manifest_path
        ),
        "retired_predecessor_manifest_exact": predecessor_manifest_exact,
        "retired_predecessor_ledger_count": len(retired_predecessor_ledgers),
        "retired_predecessor_ledger_paths": retired_predecessor_ledgers,
        "observed_retired_predecessor_ledger_paths": (
            observed_retired_predecessor_ledgers
        ),
        "outside_package_new_ledgers": outside_package,
        "missing_expected_ledgers": missing,
        "unexpected_new_ledgers": unexpected,
        "package_identifier_audit": package_audit,
        "old_and_prior_ledgers_covered_by_byte_preservation": True,
    }


def reverify_prior_closed_line(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    stop = root / "output/synthetic_confirmation_adversarial_audit_v1/CONFIRMATION_STOP_FINAL.md"
    diagnostics_path = (
        root / "output/synthetic_confirmation_adversarial_audit_v1/scientific_diagnostics.json"
    )
    development = root / "output/synthetic_development_v3_one_shot/recomputable"
    results_path = development / "model_results.json"
    averages_path = development / "model_averages.json"
    config_path = root / "output/synthetic_development_v3_one_shot/persisted_inputs/config.json"
    frozen_learner_path = (
        root
        / "output/synthetic_development_launch_package_v3/frozen_source_snapshot/babyworld_lite/sensor_alignment_v8/learner.py"
    )
    results = read_json(results_path)
    averaged = read_json(averages_path)
    old_config = read_json(config_path)
    diagnostics = read_json(diagnostics_path)
    cells = []
    for row in results:
        for presence in ("present", "null"):
            metrics = row["metrics"]["by_kind_presence"]["action"][presence]
            cells.append(
                {
                    "fractional_accuracy": float(metrics["fractional_accuracy"]),
                    "tie_frequency": float(metrics["tie_frequency"]),
                }
            )
    table = {
        (
            int(row["corpus_seed"]),
            str(row["condition"]),
            str(row["kind"]),
            str(row["presence"]),
        ): row["metrics"]
        for row in averaged["rows"]
    }
    seeds = list(
        map(int, old_config["resolved_registries"]["development"]["corpus"])
    )

    def contrast(presence: str, comparator: str) -> list[float]:
        return [
            float(
                table[(seed, "synchronized", "action", presence)][
                    "mean_correct_probability"
                ]
            )
            - float(
                table[(seed, comparator, "action", presence)][
                    "mean_correct_probability"
                ]
            )
            for seed in seeds
        ]

    present_absent = contrast("present", "absent_channel")
    null_absent = contrast("null", "absent_channel")
    present_random = contrast("present", "randomized_shuffle")
    null_random = contrast("null", "randomized_shuffle")
    correlation_absent = float(np.corrcoef(present_absent, null_absent)[0, 1])
    correlation_random = float(np.corrcoef(present_random, null_random)[0, 1])
    learner_source = frozen_learner_path.read_text(encoding="utf-8")
    weights = old_config["learner"]
    checks = {
        "authoritative_stop_exists": stop.is_file(),
        "authoritative_stop_reason_present": "CONFIRMATION_STOP" in stop.read_text(
            encoding="utf-8"
        ),
        "unaveraged_action_cell_count_1680": len(cells) == 1680,
        "all_action_cells_top_accuracy_one": all(
            row["fractional_accuracy"] == 1.0 for row in cells
        ),
        "all_action_cells_tie_free": all(
            row["tie_frequency"] == 0.0 for row in cells
        ),
        "semantic_joint_update_weight_zero": float(
            weights["sensor_joint_update_weight"]
        )
        == 0.0,
        "active_sharpening_weight_positive": float(
            weights["sensor_semantic_sharpening_weight"]
        )
        == 0.5,
        "agreement_gate_in_frozen_source": "agreement = detector_top_index == language_top_index"
        in learner_source,
        "zero_weight_scales_detector_selected_update": "* sensor_joint_update_weight"
        in learner_source,
        "power_sharpening_in_frozen_source": "np.power(" in learner_source
        and "sensor_semantic_sharpening_weight" in learner_source,
        "absent_correlation_reproduced": correlation_absent
        == float(
            diagnostics["present_null_dependence"][
                "sync_minus_absent_correlation"
            ]
        ),
        "randomized_correlation_reproduced": correlation_random
        == float(
            diagnostics["present_null_dependence"][
                "sync_minus_randomized_correlation"
            ]
        ),
        "old_terminal_decision_is_stop": diagnostics["terminal_decision"]["terminal"]
        == "CONFIRMATION_STOP",
    }
    return {
        "schema_version": "nursery-corrective-prior-line-reverification-v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "unaveraged_action_cell_count": len(cells),
        "sync_minus_absent_present_null_correlation": correlation_absent,
        "sync_minus_randomized_present_null_correlation": correlation_random,
        "frozen_weights": {
            "sensor_joint_update_weight": float(weights["sensor_joint_update_weight"]),
            "sensor_semantic_sharpening_weight": float(
                weights["sensor_semantic_sharpening_weight"]
            ),
        },
        "evidence_sha256": {
            "stop": sha256_file(stop),
            "diagnostics": sha256_file(diagnostics_path),
            "model_results": sha256_file(results_path),
            "model_averages": sha256_file(averages_path),
            "config": sha256_file(config_path),
            "frozen_learner": sha256_file(frozen_learner_path),
        },
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "old_confirmation_execution_authorized": False,
    }


def _verify_anticipated_launch_manifest(
    package: Path,
    staging: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    problems = []
    rows = list(manifest.get("files", []))
    paths = [str(row.get("path")) for row in rows]
    if (
        set(manifest)
        != {"schema_version", "file_count", "files", "digest_scheme", "digest"}
        or manifest.get("schema_version") != "nursery-exact-file-manifest-v1"
        or manifest.get("digest_scheme") != "canonical-json-of-sorted-file-rows"
        or int(manifest.get("file_count", -1)) != len(rows)
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
        or manifest.get("digest") != canonical_digest(rows)
    ):
        problems.append("manifest_contract")
    expected_paths = sorted(
        [
            *tree_file_paths(package),
            *[
                f"launch_seal/{path}"
                for path in tree_file_paths(
                    staging, exclude=["complete_file_manifest.json"]
                )
            ],
        ]
    )
    if paths != expected_paths:
        problems.append("anticipated_file_set")
    local_rows = [
        {**row, "path": str(row["path"]).removeprefix("launch_seal/")}
        for row in rows
        if str(row.get("path", "")).startswith("launch_seal/")
    ]
    local_manifest = {
        "schema_version": "nursery-exact-file-manifest-v1",
        "file_count": len(local_rows),
        "files": local_rows,
        "digest_scheme": "canonical-json-of-sorted-file-rows",
        "digest": canonical_digest(local_rows),
    }
    local_verification = verify_exact_manifest(
        staging,
        local_manifest,
        manifest_filename="complete_file_manifest.json",
    )
    if local_verification["status"] != "PASS":
        problems.append("staging_exact_tree")
    for row in rows:
        relative = str(row.get("path"))
        if relative.startswith("launch_seal/"):
            source = staging / relative.removeprefix("launch_seal/")
        else:
            source = package / relative
        try:
            metadata = source.lstat()
            observed = {
                "path": relative,
                "bytes": int(metadata.st_size),
                "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
                "sha256": sha256_file(source),
            }
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                problems.append(f"not_regular:{relative}")
            elif observed != dict(row):
                problems.append(f"row_mismatch:{relative}")
        except (FileNotFoundError, OSError) as error:
            problems.append(f"unreadable:{relative}:{type(error).__name__}")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems}


_FINALIZATION_PROOF_FACTORY_SEAL = object()


class FinalizationProof:
    __slots__ = (
        "repository_root",
        "package_root",
        "config_digest",
        "snapshot_config_sha256",
        "gates_digest",
        "adjudication_inputs_sha256",
        "evidence_hashes",
        "validation_digest",
        "_identity_digest",
        "_seal",
        "_frozen",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("finalization proofs are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, _seal: object | None = None, **values: Any) -> None:
        if _seal is not _FINALIZATION_PROOF_FACTORY_SEAL:
            raise PermissionError("launch finalization requires verified proof issuance")
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


def _issue_finalization_proof(
    repository_root: str | Path,
    package_root: str | Path,
    config: Mapping[str, Any],
    *,
    gates: Mapping[str, bool],
) -> FinalizationProof:
    root = Path(repository_root).resolve()
    package = Path(package_root).resolve()
    snapshot = package / "frozen_source_snapshot"
    if config["protocol"]["status"] != "frozen":
        raise PermissionError("finalization proof requires frozen configuration")
    parsed_config = load_config(
        snapshot / "configs/synthetic_corrective_alignment_v2.yaml",
        repository_root=snapshot,
    )
    require_frozen(parsed_config)
    if canonical_digest(config) != canonical_digest(parsed_config):
        raise PermissionError("finalization config differs from frozen snapshot config")
    if set(gates) != set(PACKAGE_GATES) or not all(
        value is True for value in gates.values()
    ):
        raise PermissionError("finalization proof requires the exact passing gate set")
    adjudication_path = package / "adjudication_inputs.json"
    adjudication = read_json(adjudication_path)
    expected_adjudication = {
        "schema_version": "nursery-corrective-package-adjudication-input-v2",
        "gates": dict(gates),
        "contradictions": [],
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }
    snapshot_verification = verify_snapshot(snapshot)
    environment_matches = environment_record(root, config) == read_json(
        package / "frozen_environment.json"
    )
    construction_root = package / "construction_qualification"
    persisted_construction = read_json(
        package / "construction_qualification_decision.json"
    )
    recomputed_construction = construction_qualification_decision(
        averaged=read_json(
            construction_root / "adjudication/model_averages.json"
        ),
        mutation_summary=read_json(
            construction_root / "adjudication/mechanism_mutations.json"
        ),
        mechanism_qualification=read_json(package / "mechanism_qualification.json"),
        dependence_diagnostics=read_json(
            construction_root / "adjudication/present_null_dependence.json"
        ),
        corpus_audits=read_json(
            construction_root / "persisted_inputs/corpus_audits.json"
        )["rows"],
        config=config,
    )
    comparison_path = package / "independent_recompute_comparison.json"
    persisted_comparison = read_json(comparison_path)
    recomputed_comparison = recompute_comparison_value(
        package / "excluded_rehearsal",
        package / "independent_recompute",
    )
    benchmark_path = package / "parallel_benchmark_attempt_1.json"
    benchmark_validation = validate_benchmark_report(
        read_json(benchmark_path),
        jobs_one_root=package / "parallel_micro_attempt_1_jobs_1",
        jobs_n_root=package / "parallel_micro_attempt_1_jobs_n",
        input_root=package / "parallel_micro_attempt_1_persisted_inputs",
        config=config,
    )
    official_path = package / "official_test_execution_report.json"
    official_validation = validate_official_test_report(
        read_json(official_path),
        snapshot_root=snapshot,
        full_repository_root=root,
        executable=root / str(config["environment"]["python"]),
        package_root=package,
    )
    fresh_official_report = run_tests(
        snapshot,
        official_path,
        root / str(config["environment"]["python"]),
        full_repository_root=root,
        persist=False,
        binding_package_root=package,
    )
    fresh_official_validation = validate_official_test_report(
        fresh_official_report,
        snapshot_root=snapshot,
        full_repository_root=root,
        executable=root / str(config["environment"]["python"]),
        package_root=package,
    )
    def official_semantic_attestation(report: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": report.get("status"),
            "commands": [row.get("command") for row in report.get("executions", [])],
            "cwds": [row.get("cwd") for row in report.get("executions", [])],
            "exit_codes": [
                row.get("exit_code") for row in report.get("executions", [])
            ],
            "passed_nodeids": report.get("passed_nodeids"),
            "passed_test_count": report.get("passed_test_count"),
            "required_test_groups": report.get("required_test_groups"),
            "required_inventory_status": report.get("required_inventory_status"),
            "closed_protocol_deselections": report.get(
                "closed_protocol_deselections"
            ),
            "observed_deselected_count": report.get("observed_deselected_count"),
            "deselection_contract_status": report.get(
                "deselection_contract_status"
            ),
            "unclassified_passed_tests": report.get(
                "unclassified_passed_tests"
            ),
            "evidence_bindings": report.get("evidence_bindings"),
        }
    persisted_official_attestation = official_semantic_attestation(
        read_json(official_path)
    )
    fresh_official_attestation = official_semantic_attestation(
        fresh_official_report
    )
    preservation_validation = reverify_prior_preservation(
        root,
        package / "preservation",
        config,
        jobs=int(config["parallel"]["frozen_jobs"]),
    )
    outcome_registry = read_json(package / "outcome_registry.json")
    frozen_runner_path = (
        snapshot / "scripts/run_synthetic_corrective_alignment_v2.py"
    )
    runner_spec = importlib.util.spec_from_file_location(
        "_nursery_corrective_frozen_finalization_runner",
        frozen_runner_path,
    )
    if runner_spec is None or runner_spec.loader is None:
        raise PermissionError("cannot load exact frozen finalization runner")
    frozen_runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(frozen_runner)
    if Path(frozen_runner.__file__).resolve() != frozen_runner_path.resolve():
        raise PermissionError("finalization validator runner origin mismatch")
    fresh_reports = {
        "PREDECESSOR_V1_FINAL_REVERIFICATION.json": (
            frozen_runner._verify_or_create_predecessor_v1_evidence(
                root,
                package,
                create=False,
            )
        ),
        "design_contract_validation.json": frozen_runner._design_contract_validation(
            root,
            package,
            snapshot,
            config,
            persisted_construction,
        ),
        "rehearsal_contract_validation.json": frozen_runner._rehearsal_contract_validation(
            package,
            snapshot,
            config,
        ),
        "walltime_estimate_validation.json": (
            frozen_runner._validate_development_command_walltime_estimate(
                package,
                config,
            )
        ),
        "preservation_completion_validation.json": (
            frozen_runner._validate_preservation_completion(package, config)
        ),
        "pristine_state_validation.json": frozen_runner._pristine_state_validation(
            root,
            package,
            config,
        ),
        "traceability_validation.json": frozen_runner._traceability(
            package,
            snapshot,
        ),
        "prior_line_reverification.json": reverify_prior_closed_line(root),
        "identifier_operation_audit.json": operation_identifier_audit(
            package,
            config,
        ),
        "repository_operation_census.json": repository_operation_census(
            root,
            package,
            config,
        ),
    }
    reports_pass = all(
        fresh.get("status") == "PASS"
        and read_json(package / relative) == fresh
        for relative, fresh in fresh_reports.items()
    )
    lifecycle_attempt_validation = (
        frozen_runner._validate_lifecycle_attempt_receipts(
            root,
            package,
            snapshot,
            config,
        )
    )
    transaction_paths = [
        root / str(config["paths"][field])
        for field in (
            "launch_seal",
            "launch_seal_staging",
            "development_output",
            "development_staging",
            "development_inner_staging",
            "authorization_claim",
            "authorization_capability_receipt",
            "development_publication_seal",
        )
    ]
    transaction_paths.append(package / "package_core_manifest.json")
    transaction_pristine = True
    for path in transaction_paths:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        transaction_pristine = False
    checks = {
        "adjudication_exact": adjudication == expected_adjudication
        and package_decision(adjudication)["terminal_state"]
        == "CORRECTIVE_DEVELOPMENT_LAUNCH_READY",
        "snapshot": snapshot_verification.get("status") == "PASS",
        "environment": environment_matches,
        "construction": persisted_construction == recomputed_construction
        and recomputed_construction.get("status") == "PASS",
        "rehearsal_comparison": persisted_comparison == recomputed_comparison
        and recomputed_comparison.get("status") == "PASS",
        "benchmark": benchmark_validation.get("status") == "PASS",
        "official_tests": official_validation.get("status") == "PASS"
        and fresh_official_validation.get("status") == "PASS"
        and persisted_official_attestation == fresh_official_attestation,
        "preservation": preservation_validation.get("status") == "PASS",
        "preservation_completion": fresh_reports[
            "preservation_completion_validation.json"
        ].get("status")
        == "PASS",
        "predecessor_v1": fresh_reports[
            "PREDECESSOR_V1_FINAL_REVERIFICATION.json"
        ].get("status")
        == "PASS",
        "zero_outcomes": int(outcome_registry.get("development_outcome_count", -1))
        == 0
        and int(outcome_registry.get("confirmation_outcome_count", -1)) == 0
        and outcome_registry.get("authorization_consumed") is False,
        "reports": reports_pass,
        "lifecycle_attempts": lifecycle_attempt_validation.get("status")
        == "PASS",
        "transaction_pristine": transaction_pristine,
    }
    if not all(checks.values()):
        raise PermissionError(f"finalization evidence failed: {checks}")
    evidence_paths = [
        adjudication_path,
        package / "freeze_receipt.json",
        package / "frozen_environment.json",
        package / "construction_qualification_decision.json",
        comparison_path,
        benchmark_path,
        official_path,
        package / "preservation/preservation_proof.json",
        package / "preservation_timing.json",
        package / "preservation_completion.json",
        package / "outcome_registry.json",
        *[package / relative for relative in fresh_reports],
        *[
            package / filename
            for filename, _status, _prerequisite in frozen_runner._LIFECYCLE_ATTEMPTS.values()
        ],
    ]
    evidence_hashes = {
        path.relative_to(package).as_posix(): sha256_file(path)
        for path in evidence_paths
    }
    validation = {
        "checks": checks,
        "snapshot_verification_digest": canonical_digest(snapshot_verification),
        "recomputed_construction_digest": canonical_digest(
            recomputed_construction
        ),
        "recomputed_comparison_digest": canonical_digest(recomputed_comparison),
        "benchmark_validation_digest": canonical_digest(benchmark_validation),
        "official_validation_digest": canonical_digest(official_validation),
        "fresh_official_validation_digest": canonical_digest(
            fresh_official_validation
        ),
        "fresh_official_semantic_attestation_digest": canonical_digest(
            fresh_official_attestation
        ),
        "preservation_validation_digest": canonical_digest(
            preservation_validation
        ),
        "fresh_report_digests": {
            relative: canonical_digest(report)
            for relative, report in fresh_reports.items()
        },
        "lifecycle_attempt_validation_digest": canonical_digest(
            lifecycle_attempt_validation
        ),
    }
    return FinalizationProof(
        repository_root=str(root),
        package_root=str(package),
        config_digest=canonical_digest(config),
        snapshot_config_sha256=sha256_file(
            snapshot / "configs/synthetic_corrective_alignment_v2.yaml"
        ),
        gates_digest=canonical_digest(dict(gates)),
        adjudication_inputs_sha256=sha256_file(adjudication_path),
        evidence_hashes=evidence_hashes,
        validation_digest=canonical_digest(validation),
        _seal=_FINALIZATION_PROOF_FACTORY_SEAL,
    )


def _verify_finalization_proof(
    proof: FinalizationProof,
    *,
    repository_root: Path,
    package_root: Path,
    config: Mapping[str, Any],
    gates: Mapping[str, bool],
) -> None:
    if (
        not isinstance(proof, FinalizationProof)
        or proof._seal is not _FINALIZATION_PROOF_FACTORY_SEAL
        or proof._frozen is not True
    ):
        raise PermissionError("launch seal requires an issued finalization proof")
    identity = {
        name: getattr(proof, name)
        for name in proof.__slots__
        if not name.startswith("_")
    }
    if canonical_digest(identity) != proof._identity_digest:
        raise PermissionError("finalization proof identity changed")
    if (
        Path(proof.repository_root) != repository_root
        or Path(proof.package_root) != package_root
        or proof.config_digest != canonical_digest(config)
        or proof.snapshot_config_sha256
        != sha256_file(
            package_root
            / "frozen_source_snapshot/configs/synthetic_corrective_alignment_v2.yaml"
        )
        or proof.gates_digest != canonical_digest(dict(gates))
        or proof.adjudication_inputs_sha256
        != sha256_file(package_root / "adjudication_inputs.json")
        or any(
            sha256_file(package_root / relative) != digest
            for relative, digest in proof.evidence_hashes.items()
        )
    ):
        raise PermissionError("finalization proof evidence changed")


def finalize_package(
    repository_root: str | Path,
    package_root: str | Path,
    config: Mapping[str, Any],
    *,
    gates: Mapping[str, bool],
    finalization_proof: FinalizationProof,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    package = Path(package_root).resolve()
    _verify_finalization_proof(
        finalization_proof,
        repository_root=root,
        package_root=package,
        config=config,
        gates=gates,
    )
    freshly_issued_proof = _issue_finalization_proof(
        root,
        package,
        config,
        gates=gates,
    )
    supplied_identity = {
        name: getattr(finalization_proof, name)
        for name in finalization_proof.__slots__
        if not name.startswith("_")
    }
    fresh_identity = {
        name: getattr(freshly_issued_proof, name)
        for name in freshly_issued_proof.__slots__
        if not name.startswith("_")
    }
    if supplied_identity != fresh_identity:
        raise PermissionError("finalization proof is not the freshly verified evidence proof")
    seal = (root / str(config["paths"]["launch_seal"])).resolve()
    staging = (root / str(config["paths"]["launch_seal_staging"])).resolve()
    if seal.parent != package:
        raise RuntimeError("launch seal must be a direct child of the package")
    expected_authorization = (root / str(config["paths"]["authorization"])).resolve()
    if expected_authorization != seal / "CORRECTIVE_DEVELOPMENT_LAUNCH_READY.json":
        raise RuntimeError("authorization path must be inside the atomic launch seal")
    if staging.parent != (root / "output").resolve():
        raise RuntimeError("launch seal staging must be a direct child of output")
    terminal_path = seal / "package_terminal.json"
    authorization_path = seal / "CORRECTIVE_DEVELOPMENT_LAUNCH_READY.json"
    core_manifest_path = package / "package_core_manifest.json"
    complete_manifest_path = seal / "complete_file_manifest.json"
    for path in (
        seal,
        staging,
        core_manifest_path,
    ):
        require_lstat_absent(path)
    decision = package_decision({"gates": dict(gates), "contradictions": []})
    if decision["terminal_state"] != "CORRECTIVE_DEVELOPMENT_LAUNCH_READY":
        write_json(package / "PACKAGE_NOT_READY.json", decision)
        return decision
    excluded_from_core = {"package_core_manifest.json", "development_authorization_consumed.json"}
    core_manifest = manifest_for_tree(package, exclude=excluded_from_core)
    write_json(core_manifest_path, core_manifest)
    future_seal_files = [
        "launch_seal/package_terminal.json",
        "launch_seal/CORRECTIVE_DEVELOPMENT_LAUNCH_READY.json",
        "launch_seal/complete_file_manifest.json",
    ]
    core_verification = verify_exact_manifest(
        package,
        core_manifest,
        manifest_filename="package_core_manifest.json",
        allowed_extra_files=future_seal_files,
    )
    if core_verification["status"] != "PASS":
        raise RuntimeError(f"package core manifest failed: {core_verification}")
    staging.mkdir()
    staged_terminal = staging / "package_terminal.json"
    staged_authorization = staging / "CORRECTIVE_DEVELOPMENT_LAUNCH_READY.json"
    staged_complete_manifest = staging / "complete_file_manifest.json"
    write_json(staged_terminal, decision)
    environment = read_json(package / "frozen_environment.json")
    snapshot = package / "frozen_source_snapshot"
    runner = snapshot / "scripts/run_synthetic_corrective_alignment_v2.py"
    values = {
        "resolved_repository_root": str(root),
        "resolved_snapshot_root": str(snapshot.resolve()),
        "resolved_authorization_path": str(authorization_path.resolve()),
        "resolved_output_root": str((root / config["paths"]["development_output"]).resolve()),
        "resolved_staging_root": str((root / config["paths"]["development_staging"]).resolve()),
        "resolved_inner_staging_root": str(
            (root / config["paths"]["development_inner_staging"]).resolve()
        ),
        "resolved_publication_seal_path": str(
            (root / config["paths"]["development_publication_seal"]).resolve()
        ),
        "resolved_claim_path": str((root / config["paths"]["authorization_claim"]).resolve()),
        "resolved_capability_receipt_path": str(
            (root / config["paths"]["authorization_capability_receipt"]).resolve()
        ),
        "required_cwd": str(root),
        "required_jobs": int(config["parallel"]["frozen_jobs"]),
        "required_thread_environment": dict(config["parallel"]["thread_environment"]),
        "required_python_executable": str(environment["python_executable"]),
        "required_python_sha256": str(environment["python_sha256"]),
        "required_runner_path": str(runner.resolve()),
        "required_runner_sha256": sha256_file(runner),
        "required_config_sha256": sha256_file(
            snapshot / "configs/synthetic_corrective_alignment_v2.yaml"
        ),
        "required_snapshot_manifest_sha256": sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        "required_freeze_receipt_sha256": sha256_file(
            package / "freeze_receipt.json"
        ),
        "required_environment_sha256": sha256_file(
            package / "frozen_environment.json"
        ),
        "required_registries_sha256": sha256_file(
            package / "frozen_seed_registries.json"
        ),
        "required_package_core_manifest_sha256": sha256_file(core_manifest_path),
        "required_package_terminal_sha256": sha256_file(staged_terminal),
        "required_excluded_rehearsal_sha256": sha256_file(
            package / "excluded_rehearsal/cohort_summary.json"
        ),
        "required_recompute_comparison_sha256": sha256_file(
            package / "independent_recompute_comparison.json"
        ),
        "required_official_tests_sha256": sha256_file(
            package / "official_test_execution_report.json"
        ),
        "required_benchmark_sha256": sha256_file(
            package / "parallel_benchmark_attempt_1.json"
        ),
        "required_preservation_proof_sha256": sha256_file(
            package / "preservation/preservation_proof.json"
        ),
        "development_registry": config["resolved_registries"]["development"],
        "confirmation_reserve_digest": canonical_digest(
            config["resolved_registries"]["confirmation_reserve"]
        ),
        "exact_argv": list(config["commands"]["development_argv"]),
        "exact_shell_command": str(config["commands"]["development_one_shot"]),
    }
    authorization = authorization_payload(**values)
    write_json(staged_authorization, authorization)
    package_rows = manifest_for_tree(package)["files"]
    staged_rows = manifest_for_tree(staging)["files"]
    seal_rows = [
        {**row, "path": f"launch_seal/{row['path']}"}
        for row in staged_rows
    ]
    rows = sorted([*package_rows, *seal_rows], key=lambda row: str(row["path"]))
    manifest = {
        "schema_version": "nursery-exact-file-manifest-v1",
        "file_count": len(rows),
        "files": rows,
        "digest_scheme": "canonical-json-of-sorted-file-rows",
        "digest": canonical_digest(rows),
    }
    write_json(staged_complete_manifest, manifest)
    anticipated = _verify_anticipated_launch_manifest(package, staging, manifest)
    if anticipated["status"] != "PASS":
        raise RuntimeError(f"anticipated launch manifest failed: {anticipated}")
    require_lstat_absent(seal)
    atomic_rename_directory_noreplace(staging, seal)
    verification = verify_exact_manifest(
        package,
        manifest,
        manifest_filename="launch_seal/complete_file_manifest.json",
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"complete launch package manifest failed: {verification}")
    return {
        **decision,
        "authorization_sha256": sha256_file(authorization_path),
        "complete_manifest_sha256": sha256_file(complete_manifest_path),
        "complete_manifest_verification": verification,
        "anticipated_manifest_verification": anticipated,
    }
