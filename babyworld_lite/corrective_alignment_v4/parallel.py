from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import resource
import secrets
import signal
import shutil
import stat
import subprocess
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
    ADJUDICATION_RELEVANT_FILES,
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
    read_confined_stable_regular_bytes,
    require_lstat_absent,
    sha256_file,
    tree_file_paths,
    verify_exact_manifest,
    verify_ledger,
    verify_development_capability,
    verify_excluded_rehearsal_capability,
    write_json,
    python_startup_flags_projection,
)
from .statistics import (
    analyze_informativeness,
    analyze_primary,
    average_model_replicates,
    dependence_diagnostics,
    score_predictions,
)
from .adjudicator import mechanism_mutation_state_selectivity


_DEVELOPMENT_INTERNAL_SEAL = object()
_REHEARSAL_INTERNAL_SEAL = object()


def _require_project_module_origins(
    source_root: str | Path,
    *,
    include_parent_only_modules: bool = False,
) -> None:
    source = Path(source_root).resolve()
    expected = {
        "babyworld_lite": "babyworld_lite/__init__.py",
        "babyworld_lite.corrective_alignment_v4": "babyworld_lite/corrective_alignment_v4/__init__.py",
        "babyworld_lite.corrective_alignment_v4.protocol": "babyworld_lite/corrective_alignment_v4/protocol.py",
        "babyworld_lite.corrective_alignment_v4.generator": "babyworld_lite/corrective_alignment_v4/generator.py",
        "babyworld_lite.corrective_alignment_v4.learner": "babyworld_lite/corrective_alignment_v4/learner.py",
        "babyworld_lite.corrective_alignment_v4.statistics": "babyworld_lite/corrective_alignment_v4/statistics.py",
        "babyworld_lite.corrective_alignment_v4.parallel": "babyworld_lite/corrective_alignment_v4/parallel.py",
    }
    if include_parent_only_modules:
        expected.update(
            {
                "babyworld_lite.corrective_alignment_v4.adjudicator": "babyworld_lite/corrective_alignment_v4/adjudicator.py",
                "babyworld_lite.corrective_alignment_v4.integrity": "babyworld_lite/corrective_alignment_v4/integrity.py",
            }
        )
    mismatches = {}
    for name, relative in expected.items():
        module = sys.modules.get(name)
        observed = Path(str(getattr(module, "__file__", ""))).resolve()
        required = (source / relative).resolve()
        if module is None or observed != required:
            mismatches[name] = {"observed": str(observed), "required": str(required)}
    if mismatches:
        raise PermissionError(f"project module origin mismatch: {mismatches}")


def _require_frozen_project_module_origins(snapshot_root: str | Path) -> None:
    _require_project_module_origins(
        snapshot_root,
        include_parent_only_modules=True,
    )


def _verify_frozen_execution_boundary(
    snapshot_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = Path(snapshot_root).resolve()
    if (
        snapshot.name != "frozen_source_snapshot"
        or snapshot.parent.name
        != "synthetic_corrective_development_launch_package_v4"
        or snapshot.parent.parent.name != "output"
    ):
        raise PermissionError("frozen execution snapshot layout mismatch")
    from .integrity import environment_record, verify_snapshot

    _require_frozen_project_module_origins(snapshot)

    verification = verify_snapshot(snapshot)
    if verification.get("status") != "PASS":
        raise PermissionError(f"frozen execution snapshot failed: {verification}")
    frozen_config_path = snapshot / "configs/synthetic_corrective_alignment_v4.yaml"
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
        != "nursery-corrective-prequalification-design-lock-v4"
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
    for name in config["parallel"]["forbidden_unbound_environment"]:
        if str(name) in os.environ:
            mismatches[str(name)] = {
                "expected": None,
                "observed": os.environ.get(str(name)),
            }
    if mismatches:
        raise RuntimeError(f"thread/startup environment mismatch: {mismatches}")


def _child_environment_base(
    *,
    inherited_allowlist: Sequence[str],
    thread_environment: Mapping[str, Any],
) -> dict[str, str]:
    """Construct the complete non-temporary worker environment."""
    allowed = list(map(str, inherited_allowlist))
    if len(allowed) != len(set(allowed)):
        raise RuntimeError("duplicate child inherited-environment entry")
    overlap = set(allowed) & {
        *map(str, thread_environment),
        "PYTHONDONTWRITEBYTECODE",
        "TMPDIR",
    }
    if overlap:
        raise RuntimeError(f"child environment roles overlap: {sorted(overlap)}")
    child = {
        name: str(os.environ[name])
        for name in allowed
        if name in os.environ
    }
    child.update({str(name): str(value) for name, value in thread_environment.items()})
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    return child


_WORKER_OPERATIONAL_RELATIVES = (
    "babyworld_lite/__init__.py",
    "babyworld_lite/corrective_alignment_v4/__init__.py",
    "babyworld_lite/corrective_alignment_v4/protocol.py",
    "babyworld_lite/corrective_alignment_v4/generator.py",
    "babyworld_lite/corrective_alignment_v4/learner.py",
    "babyworld_lite/corrective_alignment_v4/statistics.py",
    "babyworld_lite/corrective_alignment_v4/parallel.py",
    "scripts/run_synthetic_corrective_alignment_v4.py",
    "scripts/run_synthetic_corrective_alignment_v4_worker.py",
    "configs/synthetic_corrective_alignment_v4.yaml",
)


def _native_threadpool_projection() -> list[dict[str, Any]]:
    rows = [
        {
            "internal_api": str(row.get("internal_api")),
            "prefix": str(row.get("prefix")),
            "filepath": str(Path(str(row.get("filepath"))).resolve()),
            "num_threads": int(row.get("num_threads", -1)),
        }
        for row in threadpool_info()
    ]
    rows.sort(key=lambda row: (row["filepath"], row["internal_api"], row["prefix"]))
    return rows


def _python_launcher_evidence(launcher_path: str | Path) -> dict[str, Any]:
    launcher = Path(launcher_path).absolute()
    current = launcher
    seen: set[str] = set()
    chain: list[dict[str, Any]] = []
    for _depth in range(8):
        key = str(current)
        if key in seen:
            raise RuntimeError("Python launcher symlink cycle")
        seen.add(key)
        metadata = current.lstat()
        row: dict[str, Any] = {
            "path": key,
            "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
        }
        if stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(current)
            row.update({"type": "symlink", "target": target})
            chain.append(row)
            target_path = Path(target)
            current = (
                target_path
                if target_path.is_absolute()
                else current.parent / target_path
            ).absolute()
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("Python launcher chain must end at a regular file")
        row.update(
            {
                "type": "regular_file",
                "bytes": int(metadata.st_size),
                "sha256": sha256_file(current),
            }
        )
        chain.append(row)
        break
    else:
        raise RuntimeError("Python launcher symlink chain exceeds eight nodes")
    pyvenv_cfg = launcher.parent.parent / "pyvenv.cfg"
    pyvenv_metadata = pyvenv_cfg.lstat()
    if (
        len(chain) < 2
        or chain[0].get("type") != "symlink"
        or chain[-1].get("type") != "regular_file"
        or stat.S_ISLNK(pyvenv_metadata.st_mode)
        or not stat.S_ISREG(pyvenv_metadata.st_mode)
    ):
        raise RuntimeError("configured worker launcher is not a bound venv launcher")
    return {
        "chain": chain,
        "chain_digest": canonical_digest(chain),
        "pyvenv_cfg_path": str(pyvenv_cfg),
        "pyvenv_cfg_sha256": sha256_file(pyvenv_cfg),
    }


def _worker_boundary_special_inventory(
    repository: Path,
    config: Mapping[str, Any],
) -> list[tuple[Path, Path]]:
    """Derive the exact no-follow evidence inventory for the source mode."""
    package_raw = str(config["paths"]["package_root"])
    snapshot_raw = str(config["paths"]["frozen_snapshot"])
    package_relative = Path(package_raw)
    snapshot_relative = Path(snapshot_raw)
    if any(
        any(
            (
                not raw,
                "\\" in raw,
                path.is_absolute(),
                os.path.normpath(raw) != raw,
                any(part in {"", ".", ".."} for part in path.parts),
            )
        )
        for raw, path in (
            (package_raw, package_relative),
            (snapshot_raw, snapshot_relative),
        )
    ) or snapshot_relative != package_relative / "frozen_source_snapshot":
        raise PermissionError("worker-boundary package layout contract mismatch")
    if repository.name == "frozen_source_snapshot":
        live_root = repository.parents[2]
        package = (live_root / package_relative).absolute()
        expected_snapshot = (live_root / snapshot_relative).absolute()
        if repository != expected_snapshot or repository.parent != package:
            raise PermissionError("worker-boundary frozen snapshot layout mismatch")
        inventory = [
            (repository / "snapshot_manifest.json", repository),
            (package / "freeze_receipt.json", package),
            (package / "frozen_environment.json", package),
            (
                package / "prequalification_design_lock.json",
                package,
            ),
        ]
    else:
        package = repository / package_relative
        inventory = [
            (package / "prequalification_design_lock.json", repository),
        ]
    normalized = [(path.absolute(), root.absolute()) for path, root in inventory]
    normalized.sort(key=lambda item: str(item[0]))
    if len({str(path) for path, _root in normalized}) != len(normalized):
        raise PermissionError("worker-boundary special-evidence inventory is not unique")
    return normalized


def create_worker_boundary_commitment(
    *,
    repository_root: str | Path,
    config: Mapping[str, Any],
    prequalification_lock: str | Path,
) -> dict[str, Any]:
    """Commit the bounded operational surface each child revalidates."""
    repository = Path(repository_root).resolve()
    lock = Path(prequalification_lock).absolute()
    prior = str(config["registries"]["canonical_prior_registry"])
    relatives = sorted({*_WORKER_OPERATIONAL_RELATIVES, prior})
    operational_manifest = manifest_for_paths(repository, relatives)
    mode = "locked_prefreeze_source"
    if repository.name == "frozen_source_snapshot":
        mode = "frozen_snapshot"
    special_inventory = _worker_boundary_special_inventory(repository, config)
    expected_lock = next(
        path
        for path, _root in special_inventory
        if path.name == "prequalification_design_lock.json"
    )
    if lock != expected_lock:
        raise PermissionError("worker-boundary prequalification lock path mismatch")
    special_evidence = []
    for path, confinement_root in special_inventory:
        data, metadata = _read_confined_stable_regular_bytes(path, confinement_root)
        special_evidence.append(
            {
                "path": str(path),
                "bytes": int(metadata.st_size),
                "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    special_evidence.sort(key=lambda row: row["path"])
    launcher = Path(sys.executable).absolute()
    launcher_evidence = _python_launcher_evidence(launcher)
    payload = {
        "schema_version": "nursery-corrective-worker-boundary-v4",
        "mode": mode,
        "repository_root": str(repository),
        "operational_manifest": operational_manifest,
        "special_evidence": special_evidence,
        "python_executable": str(launcher),
        "python_executable_resolved": str(launcher.resolve()),
        "python_sha256": sha256_file(launcher),
        "python_launcher_chain": launcher_evidence["chain"],
        "python_launcher_chain_digest": launcher_evidence["chain_digest"],
        "pyvenv_cfg_path": launcher_evidence["pyvenv_cfg_path"],
        "pyvenv_cfg_sha256": launcher_evidence["pyvenv_cfg_sha256"],
        "sys_prefix": str(Path(sys.prefix).absolute()),
        "sys_base_prefix": str(Path(sys.base_prefix).absolute()),
        "python_version": sys.version,
        "worker_python_startup_flags": dict(
            config["environment"]["worker_python_startup_flags"]
        ),
        "thread_environment": {
            str(name): str(value)
            for name, value in config["parallel"]["thread_environment"].items()
        },
        "native_threadpools": _native_threadpool_projection(),
        "full_tree_verification_delegated_to_parent": True,
        "scientific_outcome": False,
    }
    return {**payload, "boundary_digest": canonical_digest(payload)}


def verify_worker_boundary_commitment(
    commitment: Mapping[str, Any],
    *,
    repository_root: str | Path,
    config: Mapping[str, Any],
    allow_historical_prefreeze_transition: bool = False,
    require_loaded_module_origins: bool = True,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    expected_fields = {
        "schema_version",
        "mode",
        "repository_root",
        "operational_manifest",
        "special_evidence",
        "python_executable",
        "python_executable_resolved",
        "python_sha256",
        "python_launcher_chain",
        "python_launcher_chain_digest",
        "pyvenv_cfg_path",
        "pyvenv_cfg_sha256",
        "sys_prefix",
        "sys_base_prefix",
        "python_version",
        "worker_python_startup_flags",
        "thread_environment",
        "native_threadpools",
        "full_tree_verification_delegated_to_parent",
        "scientific_outcome",
        "boundary_digest",
    }
    payload = {
        key: value for key, value in commitment.items() if key != "boundary_digest"
    }
    expected_mode = (
        "frozen_snapshot"
        if repository.name == "frozen_source_snapshot"
        else "locked_prefreeze_source"
    )
    special = commitment.get("special_evidence")
    expected_special_inventory = _worker_boundary_special_inventory(
        repository, config
    )
    expected_special_paths = [str(path) for path, _root in expected_special_inventory]
    if (
        not isinstance(special, list)
        or len(special) != len(expected_special_paths)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "mode", "sha256"}
            for row in special
        )
        or [str(row.get("path")) for row in special] != expected_special_paths
    ):
        raise PermissionError("worker boundary special-evidence inventory mismatch")
    special_bytes: dict[str, bytes] = {}
    for row, (path, confinement_root) in zip(
        special, expected_special_inventory, strict=True
    ):
        try:
            data, metadata = _read_confined_stable_regular_bytes(
                path, confinement_root
            )
        except (FileNotFoundError, OSError, RuntimeError) as error:
            raise PermissionError(
                "worker boundary special evidence is not a stable regular file"
            ) from error
        if (
            int(metadata.st_size) != int(row["bytes"])
            or format(stat.S_IMODE(metadata.st_mode), "04o") != str(row["mode"])
            or hashlib.sha256(data).hexdigest() != str(row["sha256"])
        ):
            raise PermissionError("worker boundary special evidence changed")
        special_bytes[str(path)] = data
    manifest = commitment.get("operational_manifest", {})
    paths = [str(row.get("path")) for row in manifest.get("files", [])]
    manifest_matches = False
    if allow_historical_prefreeze_transition and commitment.get("mode") == (
        "locked_prefreeze_source"
    ):
        committed_rows = {
            str(row.get("path")): dict(row) for row in manifest.get("files", [])
        }
        nonconfig_paths = [path for path in paths if path != "configs/synthetic_corrective_alignment_v4.yaml"]
        observed_nonconfig = manifest_for_paths(repository, nonconfig_paths)
        lock_path = next(
            path
            for path, _root in expected_special_inventory
            if path.name == "prequalification_design_lock.json"
        )
        lock_value = json.loads(special_bytes[str(lock_path)])
        prefreeze_projection = json.loads(
            json.dumps(lock_value.get("prefreeze_config", {}))
        )
        anticipated_projection = json.loads(json.dumps(prefreeze_projection))
        if isinstance(anticipated_projection.get("protocol"), dict):
            anticipated_projection["protocol"]["status"] = "frozen"
        current_projection = json.loads(json.dumps(config))
        current_projection.pop("repository_root", None)
        live_config_path = (
            repository / "configs/synthetic_corrective_alignment_v4.yaml"
        )
        manifest_matches = (
            all(
                committed_rows.get(str(row["path"])) == dict(row)
                for row in observed_nonconfig["files"]
            )
            and committed_rows.get(
                "configs/synthetic_corrective_alignment_v4.yaml", {}
            ).get("sha256")
            == lock_value.get("prefreeze_config_sha256")
            and lock_value.get("allowed_postqualification_delta")
            == {"path": "protocol.status", "before": "pre_freeze", "after": "frozen"}
            and canonical_digest(prefreeze_projection)
            == lock_value.get("prefreeze_config_digest")
            and canonical_digest(anticipated_projection)
            == lock_value.get("anticipated_frozen_config_digest")
            and canonical_digest(current_projection)
            == lock_value.get("anticipated_frozen_config_digest")
            and sha256_file(live_config_path)
            == lock_value.get("anticipated_frozen_config_sha256")
        )
    else:
        manifest_matches = manifest_for_paths(repository, paths) == manifest
    if (
        set(commitment) != expected_fields
        or commitment.get("schema_version")
        != "nursery-corrective-worker-boundary-v4"
        or commitment.get("mode") != expected_mode
        or Path(str(commitment.get("repository_root"))).resolve() != repository
        or commitment.get("boundary_digest") != canonical_digest(payload)
        or paths
        != sorted(
            {
                *_WORKER_OPERATIONAL_RELATIVES,
                str(config["registries"]["canonical_prior_registry"]),
            }
        )
        or not manifest_matches
        or Path(str(commitment.get("python_executable"))).absolute()
        != Path(sys.executable).absolute()
        or commitment.get("python_executable_resolved")
        != str(Path(sys.executable).resolve())
        or commitment.get("python_sha256")
        != sha256_file(Path(sys.executable).absolute())
        or commitment.get("python_launcher_chain")
        != _python_launcher_evidence(Path(sys.executable).absolute())["chain"]
        or commitment.get("python_launcher_chain_digest")
        != _python_launcher_evidence(Path(sys.executable).absolute())["chain_digest"]
        or commitment.get("pyvenv_cfg_path")
        != _python_launcher_evidence(Path(sys.executable).absolute())["pyvenv_cfg_path"]
        or commitment.get("pyvenv_cfg_sha256")
        != _python_launcher_evidence(Path(sys.executable).absolute())["pyvenv_cfg_sha256"]
        or commitment.get("sys_prefix") != str(Path(sys.prefix).absolute())
        or commitment.get("sys_base_prefix")
        != str(Path(sys.base_prefix).absolute())
        or commitment.get("python_version") != sys.version
        or commitment.get("worker_python_startup_flags")
        != dict(config["environment"]["worker_python_startup_flags"])
        or commitment.get("thread_environment")
        != {
            str(name): str(value)
            for name, value in config["parallel"]["thread_environment"].items()
        }
        or commitment.get("native_threadpools") != _native_threadpool_projection()
        or any(
            int(row.get("num_threads", -1)) != 1
            for row in commitment.get("native_threadpools", [])
        )
        or commitment.get("full_tree_verification_delegated_to_parent") is not True
        or commitment.get("scientific_outcome") is not False
    ):
        raise PermissionError("worker operational-boundary commitment mismatch")
    if require_loaded_module_origins:
        _require_project_module_origins(repository)
    _require_thread_environment(config)
    return {
        "status": "PASS",
        "boundary_digest": str(commitment["boundary_digest"]),
        "operational_file_count": len(paths),
        "special_evidence_count": len(special),
        "full_tree_verifications": 0,
    }


_SUBPROCESS_REQUEST_SCHEMA = "nursery-corrective-subprocess-request-v4"
_SUBPROCESS_SUCCESS_SCHEMA = "nursery-corrective-subprocess-success-v4"
_SUBPROCESS_FAILURE_SCHEMA = "nursery-corrective-subprocess-failure-v4"
_SUBPROCESS_INVOCATION_TOKEN_BYTES = 32


def _require_real_confined_path(path: Path, root: Path) -> None:
    """Reject symlinks in an existing path chain and require lexical confinement."""
    candidate = path.absolute()
    boundary = root.absolute()
    if (
        candidate != Path(os.path.normpath(str(candidate)))
        or boundary != Path(os.path.normpath(str(boundary)))
        or ".." in candidate.parts
        or ".." in boundary.parts
    ):
        raise RuntimeError(f"subprocess control path is not canonical: {candidate}")
    if candidate != boundary and boundary not in candidate.parents:
        raise RuntimeError(f"subprocess control path escapes root: {candidate}")
    current = candidate
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"subprocess control symlink forbidden: {current}")
        if current == boundary:
            break
        current = current.parent


def _prepare_subprocess_requests(
    *,
    plan: Mapping[str, Any],
    arguments_by_id: Mapping[str, Mapping[str, Any]],
    runtime_root: str | Path,
    child_environment_base: Mapping[str, str],
    parent_launch_secret: bytes | None = None,
) -> dict[str, Any]:
    runtime = Path(runtime_root).resolve()
    scheduler = runtime / "subprocess"
    require_lstat_absent(scheduler)
    scheduler.mkdir()
    children = {
        name: scheduler / name
        for name in ("requests", "receipts", "failures", "logs", "bootstrap_tmp")
    }
    for path in children.values():
        path.mkdir()
    expected_ids = [str(unit["unit_id"]) for unit in plan["units"]]
    if len(expected_ids) != len(set(expected_ids)) or set(arguments_by_id) != set(
        expected_ids
    ):
        raise RuntimeError("subprocess request unit inventory mismatch")
    request_paths: dict[str, Path] = {}
    request_sha256_by_id: dict[str, str] = {}
    invocation_tokens_by_id: dict[str, bytes] = {}
    launch_secret = (
        secrets.token_bytes(_SUBPROCESS_INVOCATION_TOKEN_BYTES)
        if parent_launch_secret is None
        else parent_launch_secret
    )
    if not isinstance(launch_secret, bytes) or len(
        launch_secret
    ) != _SUBPROCESS_INVOCATION_TOKEN_BYTES:
        raise PermissionError("parent worker launch secret is invalid")
    launch_secret_sha256 = hashlib.sha256(launch_secret).hexdigest()
    for unit_id in expected_ids:
        if re.fullmatch(r"corpus-[0-9]+__model-[0-9]+", unit_id) is None:
            raise RuntimeError(f"unsafe subprocess unit identifier: {unit_id!r}")
        bootstrap = children["bootstrap_tmp"] / unit_id
        bootstrap.mkdir()
        request_path = children["requests"] / f"{unit_id}.json"
        worker_arguments = dict(arguments_by_id[unit_id])
        worker_arguments["scheduler_parent_pid"] = int(os.getpid())
        worker_arguments["expected_bootstrap_environment"] = {
            **{str(name): str(value) for name, value in child_environment_base.items()},
            "TMPDIR": str(bootstrap.resolve()),
        }
        request_core = {
            "schema_version": _SUBPROCESS_REQUEST_SCHEMA,
            "unit_id": unit_id,
            "parent_launch_secret_sha256": launch_secret_sha256,
            "arguments": worker_arguments,
            "success_receipt_path": str(
                (children["receipts"] / f"{unit_id}.json").resolve()
            ),
            "failure_receipt_path": str(
                (children["failures"] / f"{unit_id}.json").resolve()
            ),
            "bootstrap_tmp_root": str(bootstrap.resolve()),
        }
        request = {
            **request_core,
            "request_authenticator_sha256": hmac.new(
                launch_secret,
                canonical_bytes(request_core),
                hashlib.sha256,
            ).hexdigest(),
        }
        write_json(request_path, request)
        request_paths[unit_id] = request_path
        request_sha256_by_id[unit_id] = sha256_file(request_path)
        invocation_tokens_by_id[unit_id] = launch_secret
    request_manifest = manifest_for_paths(
        children["requests"],
        [f"{unit_id}.json" for unit_id in expected_ids],
    )
    write_json(children["requests"] / "manifest.json", request_manifest)
    verification = verify_exact_manifest(
        children["requests"],
        request_manifest,
        manifest_filename="manifest.json",
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"subprocess request manifest failed: {verification}")
    return {
        "root": scheduler,
        **children,
        "request_paths": request_paths,
        "request_sha256_by_id": request_sha256_by_id,
        "invocation_tokens_by_id": invocation_tokens_by_id,
        "request_manifest_sha256": sha256_file(
            children["requests"] / "manifest.json"
        ),
    }


def _subprocess_worker_main(
    request_value: str | Path,
    request_sha256: str,
    request_manifest_sha256: str,
    invocation_token: bytes,
) -> int:
    """Tracked child entrypoint; it publishes exactly one receipt and one shard."""
    request_path = Path(request_value).absolute()
    failure_path: Path | None = None
    unit_id = "unknown"
    try:
        if (
            len(str(request_sha256)) != 64
            or len(str(request_manifest_sha256)) != 64
            or sha256_file(request_path) != str(request_sha256)
        ):
            raise PermissionError("subprocess request digest mismatch")
        if request_path.parent.name != "requests":
            raise PermissionError("subprocess request layout mismatch")
        scheduler = request_path.parent.parent
        _require_real_confined_path(request_path, scheduler)
        metadata = request_path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PermissionError("subprocess request must be a regular file")
        request_manifest_path = request_path.parent / "manifest.json"
        if sha256_file(request_manifest_path) != str(request_manifest_sha256):
            raise PermissionError("subprocess request-manifest digest mismatch")
        request = read_json(request_path)
        expected_fields = {
            "schema_version",
            "unit_id",
            "parent_launch_secret_sha256",
            "request_authenticator_sha256",
            "arguments",
            "success_receipt_path",
            "failure_receipt_path",
            "bootstrap_tmp_root",
        }
        if set(request) != expected_fields or request.get("schema_version") != (
            _SUBPROCESS_REQUEST_SCHEMA
        ):
            raise PermissionError("subprocess request schema mismatch")
        unit_id = str(request["unit_id"])
        authenticated_request = {
            key: value
            for key, value in request.items()
            if key != "request_authenticator_sha256"
        }
        if (
            not isinstance(invocation_token, bytes)
            or len(invocation_token) != _SUBPROCESS_INVOCATION_TOKEN_BYTES
            or request.get("parent_launch_secret_sha256")
            != hashlib.sha256(invocation_token).hexdigest()
            or request.get("request_authenticator_sha256")
            != hmac.new(
                invocation_token,
                canonical_bytes(authenticated_request),
                hashlib.sha256,
            ).hexdigest()
            or request_path.name != f"{unit_id}.json"
            or str(
                request.get("arguments", {}).get("unit", {}).get("unit_id", "")
            )
            != unit_id
            or int(request.get("arguments", {}).get("scheduler_parent_pid", -1))
            != os.getppid()
        ):
            raise PermissionError("subprocess request unit mismatch")
        success_path = Path(str(request["success_receipt_path"])).absolute()
        failure_path = Path(str(request["failure_receipt_path"])).absolute()
        bootstrap = Path(str(request["bootstrap_tmp_root"])).absolute()
        if (
            success_path != scheduler / "receipts" / f"{unit_id}.json"
            or failure_path != scheduler / "failures" / f"{unit_id}.json"
            or bootstrap != scheduler / "bootstrap_tmp" / unit_id
        ):
            raise PermissionError("subprocess receipt/temp path substitution")
        for path in (success_path, failure_path, bootstrap):
            _require_real_confined_path(path, scheduler)
        bootstrap_metadata = bootstrap.lstat()
        if (
            stat.S_ISLNK(bootstrap_metadata.st_mode)
            or not stat.S_ISDIR(bootstrap_metadata.st_mode)
            or any(bootstrap.iterdir())
            or os.environ.get("TMPDIR") != str(bootstrap)
        ):
            raise PermissionError("subprocess bootstrap temp contract mismatch")
        require_lstat_absent(success_path)
        require_lstat_absent(failure_path)
        telemetry = _worker(request["arguments"])
        expected_telemetry_fields = {
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
        if set(telemetry) != expected_telemetry_fields or telemetry.get(
            "unit_id"
        ) != unit_id:
            raise RuntimeError("worker telemetry schema/unit mismatch")
        write_json(
            success_path,
            {
                "schema_version": _SUBPROCESS_SUCCESS_SCHEMA,
                "status": "PASS",
                "unit_id": unit_id,
                "request_sha256": str(request_sha256),
                "request_manifest_sha256": str(request_manifest_sha256),
                "telemetry": telemetry,
                "scientific_outcome": False,
            },
        )
        return 0
    except BaseException as error:
        if failure_path is not None:
            try:
                require_lstat_absent(failure_path)
                write_json(
                    failure_path,
                    {
                        "schema_version": _SUBPROCESS_FAILURE_SCHEMA,
                        "status": "FAILED_NO_OUTCOME",
                        "unit_id": unit_id,
                        "request_sha256": str(request_sha256),
                        "request_manifest_sha256": str(request_manifest_sha256),
                        "exception": type(error).__name__,
                        "message": str(error),
                        "scientific_outcome": False,
                    },
                )
            except BaseException:
                pass
        return 1


def _terminate_subprocess_workers(
    active: Mapping[str, Mapping[str, Any]],
    *,
    grace_seconds: float = 5.0,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for unit_id, child in active.items():
        process = child["process"]
        row = {
            "unit_id": unit_id,
            "pid": int(process.pid),
            "term_requested": False,
            "kill_requested": False,
            "reaped": False,
            "cleanup_errors": [],
        }
        records[unit_id] = row
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                row["term_requested"] = True
            except BaseException as group_error:
                row["cleanup_errors"].append(
                    f"killpg_term:{type(group_error).__name__}:{group_error}"
                )
                try:
                    process.terminate()
                    row["term_requested"] = True
                except BaseException as terminate_error:
                    row["cleanup_errors"].append(
                        f"terminate:{type(terminate_error).__name__}:{terminate_error}"
                    )
    deadline = time.monotonic() + float(grace_seconds)
    while time.monotonic() < deadline and any(
        child["process"].poll() is None for child in active.values()
    ):
        time.sleep(0.02)
    for unit_id, child in active.items():
        process = child["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                records[unit_id]["kill_requested"] = True
            except BaseException as group_error:
                records[unit_id]["cleanup_errors"].append(
                    f"killpg_kill:{type(group_error).__name__}:{group_error}"
                )
                try:
                    process.kill()
                    records[unit_id]["kill_requested"] = True
                except BaseException as kill_error:
                    records[unit_id]["cleanup_errors"].append(
                        f"kill:{type(kill_error).__name__}:{kill_error}"
                    )
    for unit_id, child in active.items():
        process = child["process"]
        try:
            process.wait(timeout=5.0)
            records[unit_id]["reaped"] = True
        except BaseException as wait_error:
            records[unit_id]["cleanup_errors"].append(
                f"wait:{type(wait_error).__name__}:{wait_error}"
            )
        try:
            child["stdout"].close()
        except BaseException as close_error:
            records[unit_id]["cleanup_errors"].append(
                f"stdout_close:{type(close_error).__name__}:{close_error}"
            )
        try:
            child["stderr"].close()
        except BaseException as close_error:
            records[unit_id]["cleanup_errors"].append(
                f"stderr_close:{type(close_error).__name__}:{close_error}"
            )
    return [records[unit_id] for unit_id in active]


def _validate_subprocess_success(
    *,
    unit_id: str,
    request_path: Path,
    receipt_path: Path,
    failure_path: Path,
    request_manifest_sha256: str,
) -> dict[str, Any]:
    require_lstat_absent(failure_path)
    metadata = receipt_path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("subprocess success receipt must be a regular file")
    receipt = read_json(receipt_path)
    expected = {
        "schema_version",
        "status",
        "unit_id",
        "request_sha256",
        "request_manifest_sha256",
        "telemetry",
        "scientific_outcome",
    }
    if (
        set(receipt) != expected
        or receipt.get("schema_version") != _SUBPROCESS_SUCCESS_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("unit_id") != unit_id
        or receipt.get("request_sha256") != sha256_file(request_path)
        or receipt.get("request_manifest_sha256")
        != str(request_manifest_sha256)
        or receipt.get("scientific_outcome") is not False
    ):
        raise RuntimeError("subprocess success receipt commitment mismatch")
    return dict(receipt["telemetry"])


def _validate_subprocess_failure(
    *,
    unit_id: str,
    request_path: Path,
    failure_path: Path,
    success_path: Path,
    request_manifest_sha256: str,
) -> str:
    require_lstat_absent(success_path)
    metadata = failure_path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("subprocess failure receipt must be a regular file")
    receipt = read_json(failure_path)
    expected = {
        "schema_version",
        "status",
        "unit_id",
        "request_sha256",
        "request_manifest_sha256",
        "exception",
        "message",
        "scientific_outcome",
    }
    if (
        set(receipt) != expected
        or receipt.get("schema_version") != _SUBPROCESS_FAILURE_SCHEMA
        or receipt.get("status") != "FAILED_NO_OUTCOME"
        or receipt.get("unit_id") != unit_id
        or receipt.get("request_sha256") != sha256_file(request_path)
        or receipt.get("request_manifest_sha256")
        != str(request_manifest_sha256)
        or not isinstance(receipt.get("exception"), str)
        or not receipt.get("exception")
        or not isinstance(receipt.get("message"), str)
        or not receipt.get("message")
        or receipt.get("scientific_outcome") is not False
    ):
        raise RuntimeError("subprocess failure receipt commitment mismatch")
    return sha256_file(failure_path)


def _execute_worker_batches_impl(
    *,
    plan: Mapping[str, Any],
    jobs: int,
    scheduler: Mapping[str, Any],
    repository_root: str | Path,
    worker_entrypoint: str | Path,
    python_executable: str | Path,
    child_environment_base: Mapping[str, str],
    expected_launcher_evidence: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run bounded model waves with direct subprocesses and fail closed."""
    if int(jobs) < 1:
        raise ValueError("jobs must be positive")
    repository = Path(repository_root).resolve()
    entrypoint = Path(worker_entrypoint).absolute()
    executable = Path(python_executable).absolute()
    expected_entrypoint = repository / "scripts/run_synthetic_corrective_alignment_v4_worker.py"
    _require_real_confined_path(entrypoint, repository)
    entrypoint_metadata = entrypoint.lstat()
    executable_metadata = executable.lstat()
    resolved_executable = executable.resolve(strict=True)
    resolved_metadata = resolved_executable.lstat()
    launcher_evidence = _python_launcher_evidence(executable)
    frozen_launcher_evidence = dict(
        expected_launcher_evidence
        if expected_launcher_evidence is not None
        else launcher_evidence
    )
    if (
        entrypoint != expected_entrypoint
        or stat.S_ISLNK(entrypoint_metadata.st_mode)
        or not stat.S_ISREG(entrypoint_metadata.st_mode)
        or executable != Path(sys.executable).absolute()
        or not stat.S_ISLNK(executable_metadata.st_mode)
        or not stat.S_ISREG(resolved_metadata.st_mode)
        or launcher_evidence != frozen_launcher_evidence
    ):
        raise PermissionError("subprocess worker entrypoint path mismatch")
    scheduler_root = Path(scheduler["root"]).resolve()
    request_paths = scheduler["request_paths"]
    request_sha256_by_id = scheduler["request_sha256_by_id"]
    invocation_tokens_by_id = scheduler["invocation_tokens_by_id"]
    telemetry_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    launch_order: list[str] = []
    wave_rows: list[dict[str, Any]] = []
    maximum_active = 0
    active: dict[str, dict[str, Any]] = {}

    def launch(unit: Mapping[str, Any]) -> None:
        nonlocal maximum_active
        unit_id = str(unit["unit_id"])
        request_path = Path(request_paths[unit_id])
        if (
            _python_launcher_evidence(executable) != frozen_launcher_evidence
            or
            sha256_file(Path(scheduler["requests"]) / "manifest.json")
            != scheduler["request_manifest_sha256"]
            or sha256_file(request_path) != request_sha256_by_id[unit_id]
        ):
            raise PermissionError("prepared subprocess request commitment changed")
        stdout_path = Path(scheduler["logs"]) / f"{unit_id}.stdout"
        stderr_path = Path(scheduler["logs"]) / f"{unit_id}.stderr"
        stdout_handle = stdout_path.open("xb", buffering=0)
        try:
            stderr_handle = stderr_path.open("xb", buffering=0)
        except BaseException:
            stdout_handle.close()
            raise
        child_environment = {
            str(name): str(value) for name, value in child_environment_base.items()
        }
        child_environment["TMPDIR"] = str(
            (Path(scheduler["bootstrap_tmp"]) / unit_id).resolve()
        )
        expected_environment = read_json(request_path)["arguments"].get(
            "expected_bootstrap_environment"
        )
        if child_environment != expected_environment:
            raise PermissionError("subprocess child environment commitment changed")
        invocation_token = invocation_tokens_by_id.get(unit_id)
        request_value = read_json(request_path)
        authenticated_request = {
            key: value
            for key, value in request_value.items()
            if key != "request_authenticator_sha256"
        }
        if (
            not isinstance(invocation_token, bytes)
            or len(invocation_token) != _SUBPROCESS_INVOCATION_TOKEN_BYTES
            or request_value.get("parent_launch_secret_sha256")
            != hashlib.sha256(invocation_token).hexdigest()
            or request_value.get("request_authenticator_sha256")
            != hmac.new(
                invocation_token,
                canonical_bytes(authenticated_request),
                hashlib.sha256,
            ).hexdigest()
        ):
            raise PermissionError("subprocess invocation capability changed")
        invocation_read_fd, invocation_write_fd = os.pipe()
        try:
            if os.write(invocation_write_fd, invocation_token) != len(
                invocation_token
            ):
                raise RuntimeError("subprocess invocation capability write was short")
        finally:
            os.close(invocation_write_fd)
        argv = [
            str(executable),
            "-B",
            "-s",
            "-P",
            str(entrypoint),
            "--request",
            str(request_path),
            "--request-sha256",
            request_sha256_by_id[unit_id],
            "--request-manifest-sha256",
            str(scheduler["request_manifest_sha256"]),
            "--invocation-fd",
            str(invocation_read_fd),
        ]
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(repository),
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                close_fds=True,
                pass_fds=(invocation_read_fd,),
                shell=False,
                start_new_session=True,
            )
        except BaseException:
            stdout_handle.close()
            stderr_handle.close()
            raise
        finally:
            os.close(invocation_read_fd)
        active[unit_id] = {
            "process": process,
            "stdout": stdout_handle,
            "stderr": stderr_handle,
            "request": request_path,
        }
        launch_order.append(unit_id)
        maximum_active = max(maximum_active, len(active))

    try:
        for model_seed in plan["model_seed_order"]:
            batch = [
                unit
                for unit in plan["units"]
                if int(unit["model_seed"]) == int(model_seed)
            ]
            pending = list(batch)
            wave_launch_start = len(launch_order)
            while pending or active:
                while pending and len(active) < int(jobs) and not failures:
                    unit = pending.pop(0)
                    try:
                        launch(unit)
                    except BaseException as error:
                        failures.append(
                            {
                                "unit_id": str(unit["unit_id"]),
                                "exception": type(error).__name__,
                                "message": str(error),
                                "stage": "subprocess_launch",
                            }
                        )
                        break
                if failures:
                    break
                progressed = False
                for unit_id in list(active):
                    child = active[unit_id]
                    returncode = child["process"].poll()
                    if returncode is None:
                        continue
                    progressed = True
                    child["stdout"].close()
                    child["stderr"].close()
                    active.pop(unit_id)
                    if int(returncode) != 0:
                        failure_path = Path(scheduler["failures"]) / f"{unit_id}.json"
                        try:
                            failure_receipt_sha256 = _validate_subprocess_failure(
                                unit_id=unit_id,
                                request_path=child["request"],
                                failure_path=failure_path,
                                success_path=Path(scheduler["receipts"])
                                / f"{unit_id}.json",
                                request_manifest_sha256=str(
                                    scheduler["request_manifest_sha256"]
                                ),
                            )
                        except BaseException as receipt_error:
                            failures.append(
                                {
                                    "unit_id": unit_id,
                                    "exception": type(receipt_error).__name__,
                                    "message": str(receipt_error),
                                    "stage": "subprocess_failure_receipt_validation",
                                    "failure_receipt_sha256": None,
                                }
                            )
                            break
                        failures.append(
                            {
                                "unit_id": unit_id,
                                "exception": "SubprocessExit",
                                "message": f"worker exited {returncode}",
                                "stage": "subprocess_worker",
                                "failure_receipt_sha256": failure_receipt_sha256,
                            }
                        )
                        break
                    try:
                        telemetry_by_id[unit_id] = _validate_subprocess_success(
                            unit_id=unit_id,
                            request_path=child["request"],
                            receipt_path=Path(scheduler["receipts"])
                            / f"{unit_id}.json",
                            failure_path=Path(scheduler["failures"])
                            / f"{unit_id}.json",
                            request_manifest_sha256=str(
                                scheduler["request_manifest_sha256"]
                            ),
                        )
                    except BaseException as error:
                        failures.append(
                            {
                                "unit_id": unit_id,
                                "exception": type(error).__name__,
                                "message": str(error),
                                "stage": "subprocess_receipt_validation",
                            }
                        )
                        break
                if failures:
                    break
                if not progressed:
                    time.sleep(0.01)
            if failures:
                break
            wave_ids = [str(unit["unit_id"]) for unit in batch]
            if active or any(unit_id not in telemetry_by_id for unit_id in wave_ids):
                raise RuntimeError("subprocess model wave ended incompletely")
            wave_rows.append(
                {
                    "model_seed": int(model_seed),
                    "unit_ids": wave_ids,
                    "launch_order_slice": launch_order[wave_launch_start:],
                    "complete_before_next_wave": True,
                }
            )
    except BaseException as error:
        terminations = _terminate_subprocess_workers(active)
        write_json(
            scheduler_root / "PARENT_FAILED_NO_OUTCOME.json",
            {
                "schema_version": "nursery-corrective-subprocess-parent-failure-v4",
                "status": "FAILED_NO_OUTCOME",
                "exception": type(error).__name__,
                "message": str(error),
                "terminated_workers": terminations,
                "scientific_outcome": False,
            },
        )
        raise
    if failures:
        terminations = _terminate_subprocess_workers(active)
        write_json(
            scheduler_root / "FAILED_NO_OUTCOME.json",
            {
                "schema_version": "nursery-corrective-subprocess-scheduler-v4",
                "status": "FAILED_NO_OUTCOME",
                "failures": failures,
                "terminated_workers": terminations,
                "scientific_outcome": False,
            },
        )
        return [telemetry_by_id[key] for key in sorted(telemetry_by_id)], failures
    expected_ids = [str(unit["unit_id"]) for unit in plan["units"]]
    receipt_names = sorted(path.name for path in Path(scheduler["receipts"]).iterdir())
    log_names = sorted(path.name for path in Path(scheduler["logs"]).iterdir())
    if (
        set(telemetry_by_id) != set(expected_ids)
        or receipt_names != sorted(f"{unit_id}.json" for unit_id in expected_ids)
        or any(Path(scheduler["failures"]).iterdir())
        or sorted(
            path.name for path in Path(scheduler["bootstrap_tmp"]).iterdir()
        )
        != sorted(expected_ids)
        or log_names
        != sorted(
            name
            for unit_id in expected_ids
            for name in (f"{unit_id}.stdout", f"{unit_id}.stderr")
        )
        or launch_order != [
            str(unit["unit_id"])
            for model_seed in plan["model_seed_order"]
            for unit in plan["units"]
            if int(unit["model_seed"]) == int(model_seed)
        ]
        or maximum_active > int(jobs)
        or any(
            (lambda metadata: stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode))(
                (Path(scheduler["bootstrap_tmp"]) / unit_id).lstat()
            )
            or any((Path(scheduler["bootstrap_tmp"]) / unit_id).iterdir())
            for unit_id in expected_ids
        )
    ):
        raise RuntimeError("subprocess scheduler exact inventory/order mismatch")
    request_manifest = read_json(Path(scheduler["requests"]) / "manifest.json")
    if verify_exact_manifest(
        scheduler["requests"],
        request_manifest,
        manifest_filename="manifest.json",
    )["status"] != "PASS":
        raise RuntimeError("subprocess request manifest changed after execution")
    if (
        sha256_file(Path(scheduler["requests"]) / "manifest.json")
        != scheduler["request_manifest_sha256"]
        or any(
            sha256_file(Path(request_paths[unit_id]))
            != request_sha256_by_id[unit_id]
            for unit_id in expected_ids
        )
    ):
        raise RuntimeError("prepared subprocess request hash changed after execution")
    receipt_manifest = manifest_for_paths(
        scheduler["receipts"],
        [f"{unit_id}.json" for unit_id in expected_ids],
    )
    write_json(Path(scheduler["receipts"]) / "manifest.json", receipt_manifest)
    log_manifest = manifest_for_paths(scheduler["logs"], log_names)
    write_json(Path(scheduler["logs"]) / "manifest.json", log_manifest)
    if (
        verify_exact_manifest(
            scheduler["receipts"],
            receipt_manifest,
            manifest_filename="manifest.json",
        )["status"]
        != "PASS"
        or verify_exact_manifest(
            scheduler["logs"],
            log_manifest,
            manifest_filename="manifest.json",
        )["status"]
        != "PASS"
    ):
        raise RuntimeError("subprocess receipt/log manifest failed")
    for unit_id in expected_ids:
        (Path(scheduler["bootstrap_tmp"]) / unit_id).rmdir()
    Path(scheduler["bootstrap_tmp"]).rmdir()
    Path(scheduler["failures"]).rmdir()
    write_json(
        scheduler_root / "scheduler_summary.json",
        {
            "schema_version": "nursery-corrective-subprocess-scheduler-v4",
            "status": "PASS",
            "backend": "direct_subprocess_v1",
            "jobs": int(jobs),
            "launch_order": launch_order,
            "maximum_observed_concurrency": maximum_active,
            "wave_key": "model_seed",
            "waves": wave_rows,
            "request_manifest_sha256": str(
                scheduler["request_manifest_sha256"]
            ),
            "receipt_manifest_sha256": sha256_file(
                Path(scheduler["receipts"]) / "manifest.json"
            ),
            "log_manifest_sha256": sha256_file(
                Path(scheduler["logs"]) / "manifest.json"
            ),
            "validated_empty_failure_and_bootstrap_roots_removed": True,
            "scheduler_parent_pid": int(os.getpid()),
            "scientific_outcome": False,
        },
    )
    return [telemetry_by_id[unit_id] for unit_id in expected_ids], []


def _execute_worker_batches(
    *,
    plan: Mapping[str, Any],
    jobs: int,
    scheduler: Mapping[str, Any],
    repository_root: str | Path,
    worker_entrypoint: str | Path,
    python_executable: str | Path,
    child_environment_base: Mapping[str, str],
    expected_launcher_evidence: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Install scoped parent-interruption handlers around the fail-closed scheduler."""
    previous_handlers: dict[int, Any] = {}

    interrupted = False

    def interrupt(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        for signal_value in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(signal_value, signal.SIG_IGN)
        raise InterruptedError(f"subprocess scheduler received signal {signum}")

    for signal_value in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous_handlers[int(signal_value)] = signal.getsignal(signal_value)
            signal.signal(signal_value, interrupt)
        except ValueError as error:
            raise RuntimeError(
                "direct-subprocess scheduler must run in the main thread"
            ) from error
    try:
        return _execute_worker_batches_impl(
            plan=plan,
            jobs=jobs,
            scheduler=scheduler,
            repository_root=repository_root,
            worker_entrypoint=worker_entrypoint,
            python_executable=python_executable,
            child_environment_base=child_environment_base,
            expected_launcher_evidence=expected_launcher_evidence,
        )
    finally:
        for signal_number, previous in previous_handlers.items():
            signal.signal(signal_number, previous)


def validate_subprocess_scheduler_artifacts(
    *,
    runtime_root: str | Path,
    shard_root: str | Path,
    plan: Mapping[str, Any],
    jobs: int,
    config: Mapping[str, Any],
    expected_repository_root: str | Path,
    expected_config_path: str | Path,
    child_environment_base: Mapping[str, str],
    input_root: str | Path,
    expected_input_manifest_sha256: str,
    expected_scientific_contract_digest: str,
    persist: bool,
    allow_historical_prefreeze_transition: bool = False,
    relocated_from_cohort_staging: str | Path | None = None,
) -> dict[str, Any]:
    """Independently revalidate every persisted scheduler commitment."""
    runtime = Path(runtime_root).resolve()
    scheduler = runtime / "subprocess"
    shards = Path(shard_root).resolve()
    inputs = Path(input_root).resolve()
    recorded_runtime = runtime
    recorded_shards = shards
    recorded_inputs = inputs
    repository = Path(expected_repository_root).resolve()
    config_path = Path(expected_config_path).resolve()
    purpose = str(plan.get("purpose"))
    if relocated_from_cohort_staging is not None:
        published_root = runtime.parent
        expected_staging = published_root.with_name(
            f".{published_root.name}.cohort-staging"
        )
        supplied_staging = Path(relocated_from_cohort_staging).resolve()
        expected_parallel = expected_staging / "parallel_compute"
        if (
            supplied_staging != expected_staging
            or runtime != published_root / "runtime"
            or shards != published_root / "shards"
            or inputs != published_root / "persisted_inputs"
        ):
            raise RuntimeError("scheduler validator relocation layout mismatch")
        recorded_runtime = expected_parallel / "runtime"
        recorded_shards = expected_parallel / "shards"
        recorded_inputs = expected_staging / "persisted_inputs"
        try:
            expected_staging.lstat()
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError(
                "scheduler validator transient staging survived publication"
            )
    if (
        len(str(expected_input_manifest_sha256)) != 64
        or len(str(expected_scientific_contract_digest)) != 64
        or sha256_file(inputs / "input_manifest.json")
        != str(expected_input_manifest_sha256)
        or shards != runtime.parent / "shards"
    ):
        raise RuntimeError("scheduler validator input/scientific anchor mismatch")
    scientific_contract = read_json(runtime.parent / "adjudication/execution_contract.json")
    if (
        scientific_contract.get("purpose") != purpose
        or scientific_contract.get("input_manifest_sha256")
        != str(expected_input_manifest_sha256)
        or scientific_contract.get("scientific_contract_digest")
        != str(expected_scientific_contract_digest)
    ):
        raise RuntimeError("scheduler validator execution-contract anchor mismatch")
    host_repository = (
        repository.parents[2]
        if repository.name == "frozen_source_snapshot"
        else repository
    )
    empty_capability_envelope = {
        "claim_path": "",
        "claim_sha256": "",
        "issuance_receipt_path": "",
        "issuance_receipt_sha256": "",
        "development_output_root": "",
        "development_staging_root": "",
        "development_inner_staging_root": "",
        "rehearsal_snapshot_root": "",
        "rehearsal_output_root": "",
        "rehearsal_parallel_output_root": "",
        "rehearsal_receipt_path": "",
        "rehearsal_receipt_sha256": "",
    }
    expected_capability_envelope = dict(empty_capability_envelope)
    if purpose == "excluded_rehearsal":
        rehearsal_receipt = host_repository / str(
            config["paths"]["excluded_rehearsal_attempt_receipt"]
        )
        expected_capability_envelope.update(
            {
                "rehearsal_snapshot_root": str(repository),
                "rehearsal_output_root": str(
                    (
                        host_repository
                        / str(config["paths"]["excluded_rehearsal"])
                    ).resolve()
                ),
                "rehearsal_parallel_output_root": str(recorded_runtime.parent),
                "rehearsal_receipt_path": str(rehearsal_receipt.resolve()),
                "rehearsal_receipt_sha256": sha256_file(rehearsal_receipt),
            }
        )
    elif purpose == "development":
        claim = host_repository / str(config["paths"]["authorization_claim"])
        issuance = host_repository / str(
            config["paths"]["authorization_capability_receipt"]
        )
        expected_capability_envelope.update(
            {
                "claim_path": str(claim.resolve()),
                "claim_sha256": sha256_file(claim),
                "issuance_receipt_path": str(issuance.resolve()),
                "issuance_receipt_sha256": sha256_file(issuance),
                "development_output_root": str(
                    (host_repository / str(config["paths"]["development_output"])).resolve()
                ),
                "development_staging_root": str(
                    (host_repository / str(config["paths"]["development_staging"])).resolve()
                ),
                "development_inner_staging_root": str(
                    (
                        host_repository
                        / str(config["paths"]["development_inner_staging"])
                    ).resolve()
                ),
            }
        )
    elif purpose not in {
        "construction_micro",
        "construction_qualification",
        "construction_mechanism",
    }:
        raise RuntimeError("scheduler validator purpose is unsupported")
    expected_bound_arguments = {
        "purpose": purpose,
        "repository_root": str(repository),
        "config_path": str(config_path),
        "input_root": str(recorded_inputs),
        "input_manifest_sha256": str(expected_input_manifest_sha256),
        "shard_base": str(recorded_shards),
        "thread_environment": config["parallel"]["thread_environment"],
        "authorization_digest": scientific_contract.get("authorization_digest") or "",
        "parsed_contract_digest": scientific_contract.get("parsed_contract_digest") or "",
        "scientific_contract_digest": str(expected_scientific_contract_digest),
        **expected_capability_envelope,
        "required_jobs": int(jobs),
        "worker_python_startup_flags": dict(
            config["environment"]["worker_python_startup_flags"]
        ),
        "expected_launcher_evidence": scientific_contract.get(
            "python_launcher_evidence"
        ),
    }
    expected_scheduler_entries = {
        "requests",
        "receipts",
        "logs",
        "scheduler_summary.json",
    }
    observed_scheduler_entries = {path.name for path in scheduler.iterdir()}
    if observed_scheduler_entries != expected_scheduler_entries:
        raise RuntimeError("persisted subprocess scheduler file set mismatch")
    for name in ("requests", "receipts", "logs"):
        metadata = (scheduler / name).lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(f"subprocess scheduler {name} is not a real directory")
    summary_path = scheduler / "scheduler_summary.json"
    summary_metadata = summary_path.lstat()
    if stat.S_ISLNK(summary_metadata.st_mode) or not stat.S_ISREG(
        summary_metadata.st_mode
    ):
        raise RuntimeError("subprocess scheduler summary is not a regular file")

    expected_ids = [str(unit["unit_id"]) for unit in plan["units"]]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("persisted scheduler plan has duplicate unit identifiers")
    units_by_id = {str(unit["unit_id"]): unit for unit in plan["units"]}
    expected_launch_order = [
        str(unit["unit_id"])
        for model_seed in plan["model_seed_order"]
        for unit in plan["units"]
        if int(unit["model_seed"]) == int(model_seed)
    ]
    if set(expected_launch_order) != set(expected_ids):
        raise RuntimeError("persisted scheduler plan wave partition mismatch")

    request_names = [f"{unit_id}.json" for unit_id in expected_ids]
    receipt_names = [f"{unit_id}.json" for unit_id in expected_ids]
    log_names = [
        name
        for unit_id in expected_ids
        for name in (f"{unit_id}.stdout", f"{unit_id}.stderr")
    ]
    manifests: dict[str, dict[str, Any]] = {}
    for name, expected_names in (
        ("requests", request_names),
        ("receipts", receipt_names),
        ("logs", log_names),
    ):
        directory = scheduler / name
        manifest_path = directory / "manifest.json"
        manifest = read_json(manifest_path)
        if sorted(str(row.get("path")) for row in manifest.get("files", [])) != (
            sorted(expected_names)
        ):
            raise RuntimeError(f"subprocess {name} manifest member set mismatch")
        if verify_exact_manifest(
            directory,
            manifest,
            manifest_filename="manifest.json",
        )["status"] != "PASS":
            raise RuntimeError(f"subprocess {name} manifest failed revalidation")
        manifests[name] = manifest
    if any((scheduler / "logs" / name).stat().st_size != 0 for name in log_names):
        raise RuntimeError("subprocess worker stdout/stderr must be empty")

    summary = read_json(summary_path)
    expected_summary_fields = {
        "schema_version",
        "status",
        "backend",
        "jobs",
        "launch_order",
        "maximum_observed_concurrency",
        "wave_key",
        "waves",
        "request_manifest_sha256",
        "receipt_manifest_sha256",
        "log_manifest_sha256",
        "validated_empty_failure_and_bootstrap_roots_removed",
        "scheduler_parent_pid",
        "scientific_outcome",
    }
    wave_ids = [
        [
            str(unit["unit_id"])
            for unit in plan["units"]
            if int(unit["model_seed"]) == int(model_seed)
        ]
        for model_seed in plan["model_seed_order"]
    ]
    expected_waves = [
        {
            "model_seed": int(model_seed),
            "unit_ids": ids,
            "launch_order_slice": ids,
            "complete_before_next_wave": True,
        }
        for model_seed, ids in zip(plan["model_seed_order"], wave_ids, strict=True)
    ]
    expected_maximum_active = min(int(jobs), max(map(len, wave_ids), default=0))
    if (
        set(summary) != expected_summary_fields
        or summary.get("schema_version")
        != "nursery-corrective-subprocess-scheduler-v4"
        or summary.get("status") != "PASS"
        or summary.get("backend") != config["parallel"]["backend"]
        or int(summary.get("jobs", -1)) != int(jobs)
        or summary.get("launch_order") != expected_launch_order
        or int(summary.get("maximum_observed_concurrency", -1))
        != expected_maximum_active
        or summary.get("wave_key") != config["parallel"]["wave_key"]
        or summary.get("waves") != expected_waves
        or summary.get("request_manifest_sha256")
        != sha256_file(scheduler / "requests/manifest.json")
        or summary.get("receipt_manifest_sha256")
        != sha256_file(scheduler / "receipts/manifest.json")
        or summary.get("log_manifest_sha256")
        != sha256_file(scheduler / "logs/manifest.json")
        or summary.get("validated_empty_failure_and_bootstrap_roots_removed")
        is not True
        or int(summary.get("scheduler_parent_pid", -1)) <= 1
        or summary.get("scientific_outcome") is not False
    ):
        raise RuntimeError("subprocess scheduler summary contract mismatch")

    expected_argument_fields = {
        "unit",
        "purpose",
        "repository_root",
        "config_path",
        "input_root",
        "input_manifest_sha256",
        "shard_base",
        "thread_environment",
        "authorization_digest",
        "parsed_contract_digest",
        "scientific_contract_digest",
        "claim_path",
        "claim_sha256",
        "issuance_receipt_path",
        "issuance_receipt_sha256",
        "development_output_root",
        "development_staging_root",
        "development_inner_staging_root",
        "rehearsal_snapshot_root",
        "rehearsal_output_root",
        "rehearsal_parallel_output_root",
        "rehearsal_receipt_path",
        "rehearsal_receipt_sha256",
        "required_jobs",
        "worker_boundary_commitment",
        "expected_launcher_evidence",
        "worker_python_startup_flags",
        "scheduler_parent_pid",
        "expected_bootstrap_environment",
    }
    expected_receipt_fields = {
        "schema_version",
        "status",
        "unit_id",
        "request_sha256",
        "request_manifest_sha256",
        "telemetry",
        "scientific_outcome",
    }
    expected_telemetry_fields = {
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
    _validate_shard_inventory(shards, expected_ids)
    request_manifest_sha256 = sha256_file(scheduler / "requests/manifest.json")
    common_worker_boundary: dict[str, Any] | None = None
    common_worker_boundary_verification: dict[str, Any] | None = None
    for unit_id in expected_ids:
        request_path = scheduler / "requests" / f"{unit_id}.json"
        receipt_path = scheduler / "receipts" / f"{unit_id}.json"
        request = read_json(request_path)
        arguments = request.get("arguments", {})
        expected_bootstrap = {
            **{str(name): str(value) for name, value in child_environment_base.items()},
            "TMPDIR": str(
                (
                    recorded_runtime
                    / "subprocess/bootstrap_tmp"
                    / unit_id
                ).resolve()
            ),
        }
        if common_worker_boundary is None:
            common_worker_boundary = dict(
                arguments.get("worker_boundary_commitment", {})
            )
            common_worker_boundary_verification = verify_worker_boundary_commitment(
                common_worker_boundary,
                repository_root=arguments.get("repository_root", ""),
                config=config,
                allow_historical_prefreeze_transition=(
                    allow_historical_prefreeze_transition
                ),
                require_loaded_module_origins=(
                    not allow_historical_prefreeze_transition
                ),
            )
        elif arguments.get("worker_boundary_commitment") != common_worker_boundary:
            raise RuntimeError("subprocess requests use different worker boundaries")
        if (
            set(request)
            != {
                "schema_version",
                "unit_id",
                "parent_launch_secret_sha256",
                "request_authenticator_sha256",
                "arguments",
                "success_receipt_path",
                "failure_receipt_path",
                "bootstrap_tmp_root",
            }
            or request.get("schema_version") != _SUBPROCESS_REQUEST_SCHEMA
            or request.get("unit_id") != unit_id
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(request.get("parent_launch_secret_sha256", "")),
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(request.get("request_authenticator_sha256", "")),
            )
            is None
            or set(arguments) != expected_argument_fields
            or arguments.get("unit") != units_by_id[unit_id]
            or arguments.get("purpose") != plan.get("purpose")
            or {
                key: arguments.get(key) for key in expected_bound_arguments
            }
            != expected_bound_arguments
            or int(arguments.get("required_jobs", -1)) != int(jobs)
            or int(arguments.get("scheduler_parent_pid", -1))
            != int(summary["scheduler_parent_pid"])
            or arguments.get("thread_environment")
            != config["parallel"]["thread_environment"]
            or arguments.get("expected_bootstrap_environment")
            != expected_bootstrap
            or request.get("success_receipt_path")
            != str(
                (
                    recorded_runtime
                    / "subprocess/receipts"
                    / f"{unit_id}.json"
                ).resolve()
            )
            or request.get("failure_receipt_path")
            != str(
                (
                    recorded_runtime
                    / "subprocess/failures"
                    / f"{unit_id}.json"
                ).resolve()
            )
            or request.get("bootstrap_tmp_root")
            != str(
                (
                    recorded_runtime
                    / "subprocess/bootstrap_tmp"
                    / unit_id
                ).resolve()
            )
        ):
            raise RuntimeError(f"subprocess request contract mismatch: {unit_id}")
        receipt = read_json(receipt_path)
        telemetry = receipt.get("telemetry", {})
        shard = shards / unit_id
        if (
            set(receipt) != expected_receipt_fields
            or receipt.get("schema_version") != _SUBPROCESS_SUCCESS_SCHEMA
            or receipt.get("status") != "PASS"
            or receipt.get("unit_id") != unit_id
            or receipt.get("request_sha256") != sha256_file(request_path)
            or receipt.get("request_manifest_sha256")
            != request_manifest_sha256
            or receipt.get("scientific_outcome") is not False
            or set(telemetry) != expected_telemetry_fields
            or telemetry.get("unit_id") != unit_id
            or telemetry.get("shard_manifest_sha256")
            != sha256_file(shard / "manifest.json")
            or telemetry.get("bootstrap_environment_digest")
            != canonical_digest(expected_bootstrap)
            or telemetry.get("worker_boundary_digest")
            != arguments.get("worker_boundary_commitment", {}).get(
                "boundary_digest"
            )
            or int(
                telemetry.get("worker_boundary_operational_file_count", -1)
            )
            != len(
                arguments.get("worker_boundary_commitment", {})
                .get("operational_manifest", {})
                .get("files", [])
            )
            or int(telemetry.get("worker_input_regular_files_verified", -1)) != 1
            or int(telemetry.get("worker_full_tree_verifications", -1)) != 0
            or telemetry.get("thread_environment_observed")
            != {
                str(name): str(value)
                for name, value in config["parallel"]["thread_environment"].items()
            }
            or telemetry.get("native_threadpools")
            != [
                {
                    "internal_api": row["internal_api"],
                    "prefix": row["prefix"],
                    "num_threads": row["num_threads"],
                }
                for row in arguments["worker_boundary_commitment"][
                    "native_threadpools"
                ]
            ]
            or telemetry.get("python_startup_flags_observed")
            != config["environment"]["worker_python_startup_flags"]
            or any(
                int(pool.get("num_threads", -1)) != 1
                for pool in telemetry.get("native_threadpools", [])
            )
            or not math.isfinite(float(telemetry.get("wall_seconds", math.nan)))
            or float(telemetry.get("wall_seconds", -1.0)) < 0.0
            or int(telemetry.get("worker_peak_rss_native_units", -1)) < 0
        ):
            raise RuntimeError(f"subprocess receipt/telemetry mismatch: {unit_id}")
        complete = read_json(shard / "COMPLETE.json")
        manifest = read_json(shard / "manifest.json")
        if (
            complete
            != {
                "schema_version": "nursery-corrective-shard-complete-v4",
                "unit_id": unit_id,
                "result_sha256": sha256_file(shard / "result.json"),
                "ledger_sha256": sha256_file(shard / "operation_ledger.json"),
                "manifest_sha256": sha256_file(shard / "manifest.json"),
                "status": "COMPLETE",
            }
            or verify_exact_manifest(
                shard,
                manifest,
                manifest_filename="manifest.json",
                allowed_extra_files=["COMPLETE.json"],
            )["status"]
            != "PASS"
        ):
            raise RuntimeError(f"subprocess shard seal mismatch: {unit_id}")

    report = {
        "schema_version": "nursery-corrective-scheduler-validation-v4",
        "status": "PASS",
        "unit_count": len(expected_ids),
        "jobs": int(jobs),
        "maximum_observed_concurrency": expected_maximum_active,
        "wave_count": len(expected_waves),
        "request_manifest_sha256": request_manifest_sha256,
        "receipt_manifest_sha256": sha256_file(
            scheduler / "receipts/manifest.json"
        ),
        "log_manifest_sha256": sha256_file(scheduler / "logs/manifest.json"),
        "scheduler_summary_sha256": sha256_file(summary_path),
        "shard_manifest_set_digest": canonical_digest(
            [
                {
                    "unit_id": unit_id,
                    "manifest_sha256": sha256_file(
                        shards / unit_id / "manifest.json"
                    ),
                }
                for unit_id in expected_ids
            ]
        ),
        "worker_boundary_digest": str(
            (common_worker_boundary_verification or {}).get("boundary_digest", "")
        ),
        "empty_failure_and_bootstrap_roots_absent": True,
        "all_worker_logs_empty": True,
        "scientific_outcome": False,
    }
    report_path = runtime / "scheduler_validation.json"
    if persist:
        require_lstat_absent(report_path)
        write_json(report_path, report)
    return report


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


def _read_confined_stable_regular_bytes(path: Path, root: Path) -> tuple[bytes, os.stat_result]:
    base = Path(root).absolute()
    candidate = Path(path).absolute()
    try:
        relative = candidate.relative_to(base).as_posix()
    except ValueError:
        raise PermissionError("path escapes committed root") from None
    return read_confined_stable_regular_bytes(base, relative)


def _verify_worker_persisted_input_commitment(
    root: str | Path,
    *,
    expected_manifest_sha256: str,
    required_relative: str,
) -> dict[str, Any]:
    """Verify the root commitment and the one corpus member a worker may read."""
    base = Path(root).resolve()
    _require_real_confined_path(base, base)
    manifest_path = base / "input_manifest.json"
    metadata = manifest_path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError("worker input manifest must be a regular file")
    if sha256_file(manifest_path) != str(expected_manifest_sha256):
        raise PermissionError("worker input-manifest digest mismatch")
    manifest = read_json(manifest_path)
    rows = manifest.get("files")
    if (
        set(manifest)
        != {"schema_version", "file_count", "files", "digest_scheme", "digest"}
        or manifest.get("schema_version") != "nursery-exact-file-manifest-v1"
        or manifest.get("digest_scheme")
        != "canonical-json-of-sorted-file-rows"
        or not isinstance(rows, list)
        or int(manifest.get("file_count", -1)) != len(rows)
        or manifest.get("digest") != canonical_digest(rows)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "mode", "sha256"}
            for row in rows
        )
    ):
        raise PermissionError("worker input-manifest schema/digest mismatch")
    paths = [str(row["path"]) for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PermissionError("worker input-manifest path ordering mismatch")
    matching = [row for row in rows if str(row["path"]) == str(required_relative)]
    if len(matching) != 1:
        raise PermissionError("worker learner input is not uniquely committed")
    member = base / str(required_relative)
    member_bytes, member_metadata = _read_confined_stable_regular_bytes(
        member,
        base,
    )
    row = matching[0]
    if (
        not stat.S_ISREG(member_metadata.st_mode)
        or int(member_metadata.st_size) != int(row["bytes"])
        or format(stat.S_IMODE(member_metadata.st_mode), "04o") != str(row["mode"])
        or hashlib.sha256(member_bytes).hexdigest() != str(row["sha256"])
    ):
        raise PermissionError("worker learner input member changed")
    return manifest


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
        "schema_version": "nursery-corrective-work-plan-v4",
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
    payload, metadata = _read_confined_stable_regular_bytes(path, input_root)
    if (
        len(payload) != int(row.get("bytes", -1))
        or format(stat.S_IMODE(metadata.st_mode), "04o") != str(row.get("mode"))
        or hashlib.sha256(payload).hexdigest() != str(row.get("sha256"))
    ):
        raise RuntimeError(f"persisted input changed during committed read: {relative}")
    return json.loads(payload)


def _worker(
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    expected_bootstrap_environment = {
        str(name): str(value)
        for name, value in arguments.get("expected_bootstrap_environment", {}).items()
    }
    if not expected_bootstrap_environment or dict(os.environ) != (
        expected_bootstrap_environment
    ):
        raise PermissionError("worker bootstrap environment differs from request")
    observed_worker_startup_flags = python_startup_flags_projection()
    if observed_worker_startup_flags != dict(
        arguments.get("worker_python_startup_flags", {})
    ):
        raise PermissionError("worker Python startup flags differ from request")
    bootstrap_environment_digest = canonical_digest(expected_bootstrap_environment)
    scheduler_parent_pid = int(arguments.get("scheduler_parent_pid", -1))
    if scheduler_parent_pid <= 1 or os.getppid() != scheduler_parent_pid:
        raise PermissionError("worker scheduler-parent liveness mismatch")
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
    if _python_launcher_evidence(Path(sys.executable).absolute()) != dict(
        arguments.get("expected_launcher_evidence", {})
    ):
        raise PermissionError("worker launcher differs from locked request")
    input_root = Path(arguments["input_root"]).resolve()
    shard_base = Path(arguments["shard_base"]).resolve()
    unit = dict(arguments["unit"])
    unit_id = str(unit["unit_id"])
    config = load_config(
        arguments["config_path"],
        repository_root=arguments["repository_root"],
    )
    _require_thread_environment(config)
    worker_boundary_verification = verify_worker_boundary_commitment(
        arguments["worker_boundary_commitment"],
        repository_root=arguments["repository_root"],
        config=config,
    )
    registry = config["resolved_registries"].get(purpose)
    corpus_seed = int(unit.get("corpus_seed", -1))
    model_seed = int(unit.get("model_seed", -1))
    expected_unit = {
        "unit_id": _unit_id(corpus_seed, model_seed),
        "corpus_seed": corpus_seed,
        "model_seed": model_seed,
        "condition_order": list(config["design"]["conditions"]),
        "mutation_order": list(config["design"]["mechanism_mutations"]),
    }
    if (
        registry is None
        or corpus_seed not in map(int, registry["corpus"])
        or model_seed not in map(int, registry["model"])
        or unit != expected_unit
    ):
        raise PermissionError("worker unit/order differs from frozen work plan")
    learner_relative = f"corpus-{corpus_seed}/learner_input.json"
    input_manifest_sha256 = sha256_file(input_root / "input_manifest.json")
    input_manifest = _verify_worker_persisted_input_commitment(
        input_root,
        expected_manifest_sha256=str(arguments["input_manifest_sha256"]),
        required_relative=learner_relative,
    )
    if input_manifest_sha256 != str(arguments["input_manifest_sha256"]):
        raise PermissionError("worker persisted-input commitment mismatch")
    if purpose in {"development", "excluded_rehearsal"}:
        snapshot = Path(str(arguments["repository_root"])).resolve()
        if (
            snapshot.name != "frozen_source_snapshot"
            or
            config["protocol"]["status"] != "frozen"
            or Path(str(arguments["config_path"])).resolve()
            != (snapshot / "configs/synthetic_corrective_alignment_v4.yaml").resolve()
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
            _full_integrity=False,
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
                "presence_intercept_forced_to": -20.0,
                "presence_mismatch_weight_forced_to": 0.0,
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
        "schema_version": "nursery-corrective-worker-result-v4",
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
            "schema_version": "nursery-corrective-shard-complete-v4",
            "unit_id": unit_id,
            "result_sha256": sha256_file(temporary / "result.json"),
            "ledger_sha256": sha256_file(temporary / "operation_ledger.json"),
            "manifest_sha256": sha256_file(temporary / "manifest.json"),
            "status": "COMPLETE",
        },
    )
    temporary_tmp.rmdir()
    if os.getppid() != scheduler_parent_pid:
        raise InterruptedError("scheduler parent disappeared before shard publication")
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
        "bootstrap_environment_digest": bootstrap_environment_digest,
        "worker_boundary_digest": worker_boundary_verification[
            "boundary_digest"
        ],
        "worker_boundary_operational_file_count": worker_boundary_verification[
            "operational_file_count"
        ],
        "worker_input_regular_files_verified": 1,
        "worker_full_tree_verifications": worker_boundary_verification[
            "full_tree_verifications"
        ],
        "thread_environment_observed": {
            str(name): os.environ.get(str(name))
            for name in arguments["thread_environment"]
        },
        "native_threadpools": native_threadpools,
        "python_startup_flags_observed": observed_worker_startup_flags,
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
        "schema_version": "nursery-corrective-shard-complete-v4",
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
        or result.get("schema_version") != "nursery-corrective-worker-result-v4"
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
                != "nursery-corrective-lexicon-model-v3"
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
        "schema_version": "nursery-corrective-mutation-summary-v4",
        "mutations": output,
        "separate_semantic_and_null_heads": True,
    }


def _candidate_scientific_decision(
    *,
    purpose: str,
    development_validity_status: str,
    primary_status: str,
    informativeness_status: str,
    dependence_status: str,
    causal_attribution_status: str,
) -> str:
    if purpose != "development":
        return "SCIENTIFIC_INFERENCE_SUPPRESSED"
    if development_validity_status != "PASS":
        return "REVISE_INVALID_DEVELOPMENT_COHORT"
    if informativeness_status != "PASS" or dependence_status != "PASS":
        return "REVISE_UNINFORMATIVE"
    if primary_status == "PASS" and causal_attribution_status == "PASS":
        return "GO"
    if primary_status == "PASS":
        return "REVISE_CAUSAL_ATTRIBUTION_FAILED"
    return "CORRECTIVE_STUDY_STOP_NO_SUPPORT"


def _development_validity_audit(
    *,
    purpose: str,
    averaged: Mapping[str, Any],
    merged_results: Sequence[Mapping[str, Any]],
    corpus_audits: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed if the realized development cohort loses the frozen construct."""
    if purpose != "development":
        return {
            "schema_version": "nursery-corrective-development-validity-v4",
            "status": "NOT_APPLICABLE_SCIENTIFIC_INFERENCE_SUPPRESSED",
            "purpose": purpose,
            "scientific_inference_suppressed": True,
            "scientific_outcome": False,
        }
    expected_corpora = list(
        map(int, config["resolved_registries"]["development"]["corpus"])
    )
    expected_models = list(
        map(int, config["resolved_registries"]["development"]["model"])
    )
    expected_conditions = list(map(str, config["design"]["conditions"]))
    variation_metric_names = (
        "lexical_acquisition_top1",
        "heldout_composition_top1",
        "presence_balanced_accuracy",
        "lexical.multiclass_log_loss",
        "composition.multiclass_log_loss",
        "presence.multiclass_log_loss",
    )
    averaged_metric_names = set(variation_metric_names)
    for kind in ("lexical", "composition", "presence", "zero_exposure"):
        for field in (
            "strict_top1",
            "mean_rank",
            "mrr",
            "multiclass_log_loss",
            "multiclass_brier",
            "tie_rate",
            "exact_chance",
        ):
            averaged_metric_names.add(f"{kind}.{field}")
    averaged_metric_names.update(
        {"presence.present.strict_top1", "presence.null.strict_top1"}
    )

    def is_exact_int(value: Any, *, minimum: int | None = None) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and (minimum is None or int(value) >= minimum)
        )

    def is_finite_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def is_sha256(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    if not expected_corpora or len(expected_models) < 2 or not expected_conditions:
        raise RuntimeError("development validity requires nonempty frozen registries")
    factor_names = (
        "grounded_rate_draw",
        "foil_rate_draw",
        "visibility_draw",
        "lag_mean_draw",
        "lag_sd_draw",
        "side_informativity_draw",
        "side_noise_draw",
    )
    digest_fields = (
        "action_geometry_digest",
        "episode_geometry_digest",
        "heldout_composition_digest",
        "repetition_count_digest",
        "candidate_count_histogram_digest",
    )
    for row in corpus_audits:
        if not isinstance(row, Mapping):
            raise RuntimeError("development corpus audit must be a mapping")
        if not all(is_sha256(row.get(name)) for name in digest_fields):
            raise RuntimeError("development corpus audit digest contract mismatch")
        if not all(is_finite_number(row.get(name)) for name in factor_names):
            raise RuntimeError("development factor draws must all be finite")
        stratum = str(row.get("ambiguity_stratum"))
        if stratum not in set(map(str, config["design"]["ambiguity_strata"])):
            raise RuntimeError("development ambiguity stratum is outside the frozen design")
        factor_ranges = {
            "grounded_rate_draw": config["design"]["grounded_rate_range"],
            "foil_rate_draw": config["design"]["foil_rate_by_stratum"][stratum],
            "visibility_draw": config["design"]["visibility_range"],
            "lag_mean_draw": config["design"]["lag_mean_range"],
            "lag_sd_draw": config["design"]["lag_sd_range"],
            "side_informativity_draw": config["design"]["side_informativeness_range"],
            "side_noise_draw": config["design"]["side_noise_range"],
        }
        if any(
            not float(bounds[0]) <= float(row[name]) <= float(bounds[1])
            for name, bounds in factor_ranges.items()
        ):
            raise RuntimeError("development factor draw is outside its frozen range")
        if not (
            is_exact_int(row.get("corpus_seed"))
            and is_exact_int(row.get("episode_count"), minimum=1)
            and is_exact_int(row.get("grounded_episode_count"), minimum=0)
            and int(row["grounded_episode_count"]) <= int(row["episode_count"])
            and is_finite_number(row.get("realized_foil_present_rate"))
            and 0.0 <= float(row["realized_foil_present_rate"]) <= 1.0
            and is_finite_number(row.get("observed_lag_mean"))
            and is_finite_number(row.get("realized_mean_event_observation_peak"))
            and float(row["realized_mean_event_observation_peak"]) >= 0.0
            and row.get("correction_challenges_cover_every_training_composition")
            is True
            and row.get("all_grounded_episodes_are_correction_challenges") is True
            and row.get("every_training_composition_has_present_and_null_support")
            is True
            and row.get("foil_map_is_bijective") is True
            and row.get("foil_map_changes_both_action_dimensions") is True
            and row.get("matched_background_proposal_banks") is True
            and row.get(
                "event_and_background_proposal_counts_and_widths_match"
            )
            is True
            and row.get("signed_shift_guard_bands_valid") is True
            and row.get("realized_event_widths_within_frozen_range") is True
            and row.get("realized_inter_event_gaps_within_frozen_range") is True
            and row.get("all_condition_evidence_factorized") is True
        ):
            raise RuntimeError("development corpus audit numeric contract mismatch")
        try:
            primitive = row["action_geometry"]["slots"]["primitive"]
            manner = row["action_geometry"]["slots"]["manner"]
            detector_accuracy = row["condition_audit"]["detector_manipulation"][
                "synchronized"
            ]["strict_event_or_null_accuracy"]
        except (KeyError, TypeError):
            raise RuntimeError("development corpus audit nested contract mismatch") from None
        for slot in (primitive, manner):
            if not (
                is_finite_number(slot.get("mean_pairwise_distance"))
                and float(slot["mean_pairwise_distance"]) >= 0.0
                and is_finite_number(slot.get("minimum_self_cross_dot_margin"))
            ):
                raise RuntimeError("development action geometry must be finite")
        if not (
            is_finite_number(detector_accuracy)
            and 0.0 <= float(detector_accuracy) <= 1.0
            and row["condition_audit"].get(
                "detector_scoring_factorizes_presence_from_conditional_event"
            )
            is True
            and row["condition_audit"].get(
                "event_and_null_logits_never_jointly_ranked"
            )
            is True
        ):
            raise RuntimeError("development detector accuracy must be finite and bounded")

    expected_average_fields = {
        "schema_version",
        "independent_unit",
        "model_seed_order",
        "exact_model_seed_set_required",
        "floating_reduction",
        "rows",
        "within_corpus_model_variation",
    }
    if not (
        isinstance(averaged, Mapping)
        and set(averaged) == expected_average_fields
        and averaged.get("schema_version") == "nursery-corrective-model-average-v3"
        and averaged.get("independent_unit") == "corpus_seed"
        and averaged.get("model_seed_order") == expected_models
        and averaged.get("exact_model_seed_set_required") is True
        and averaged.get("floating_reduction")
        == "math.fsum_in_frozen_model_seed_order"
        and isinstance(averaged.get("rows"), list)
        and isinstance(averaged.get("within_corpus_model_variation"), list)
    ):
        raise RuntimeError("development model-average contract mismatch")
    audits_by_seed = {int(row["corpus_seed"]): row for row in corpus_audits}
    results_by_unit = {
        (int(row["corpus_seed"]), int(row["model_seed"])): row
        for row in merged_results
    }
    if not (
        len(corpus_audits) == len(expected_corpora)
        and sorted(audits_by_seed) == sorted(expected_corpora)
    ):
        raise RuntimeError("development corpus-audit inventory mismatch")
    expected_unit_keys = {
        (corpus_seed, model_seed)
        for corpus_seed in expected_corpora
        for model_seed in expected_models
    }
    if not (
        len(merged_results) == len(expected_unit_keys)
        and set(results_by_unit) == expected_unit_keys
    ):
        raise RuntimeError("development model-unit inventory mismatch")
    action_geometry_count = len(
        {str(row["action_geometry_digest"]) for row in corpus_audits}
    )
    episode_geometry_count = len(
        {str(row["episode_geometry_digest"]) for row in corpus_audits}
    )
    heldout_split_count = len(
        {str(row["heldout_composition_digest"]) for row in corpus_audits}
    )
    repetition_count = len(
        {str(row["repetition_count_digest"]) for row in corpus_audits}
    )
    candidate_histogram_count = len(
        {str(row["candidate_count_histogram_digest"]) for row in corpus_audits}
    )
    grounded_count_count = len(
        {int(row["grounded_episode_count"]) for row in corpus_audits}
    )
    ambiguity_strata = sorted(
        {str(row["ambiguity_stratum"]) for row in corpus_audits}
    )
    ambiguity_stratum_counts = {
        stratum: sum(
            str(row["ambiguity_stratum"]) == stratum for row in corpus_audits
        )
        for stratum in map(str, config["design"]["ambiguity_strata"])
    }
    factor_distinct_counts = {
        name: len({float(row[name]) for row in corpus_audits})
        for name in factor_names
    }
    construction_corpus_count = len(
        config["resolved_registries"]["construction_qualification"]["corpus"]
    )

    def scaled_distinct_minimum(name: str) -> int:
        return int(
            math.ceil(
                int(config["qualification_gates"][name])
                * len(expected_corpora)
                / construction_corpus_count
            )
        )

    def value_range(values: Sequence[float]) -> float:
        if not values or not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("development realized range requires finite observations")
        return float(max(values) - min(values))

    realized_ranges = {
        "primitive_action_geometry_mean_pairwise_distance": value_range(
            [
                float(
                    row["action_geometry"]["slots"]["primitive"][
                        "mean_pairwise_distance"
                    ]
                )
                for row in corpus_audits
            ]
        ),
        "manner_action_geometry_mean_pairwise_distance": value_range(
            [
                float(
                    row["action_geometry"]["slots"]["manner"][
                        "mean_pairwise_distance"
                    ]
                )
                for row in corpus_audits
            ]
        ),
        "grounded_rate": value_range(
            [
                float(row["grounded_episode_count"]) / float(row["episode_count"])
                for row in corpus_audits
            ]
        ),
        "foil_present_rate": value_range(
            [float(row["realized_foil_present_rate"]) for row in corpus_audits]
        ),
        "observed_lag_mean": value_range(
            [float(row["observed_lag_mean"]) for row in corpus_audits]
        ),
        "observation_peak": value_range(
            [
                float(row["realized_mean_event_observation_peak"])
                for row in corpus_audits
            ]
        ),
        "detector_accuracy": value_range(
            [
                float(
                    row["condition_audit"]["detector_manipulation"][
                        "synchronized"
                    ]["strict_event_or_null_accuracy"]
                )
                for row in corpus_audits
            ]
        ),
    }
    model_trace_rows = []
    expected_trace_fields = {
        "schema_version",
        "condition",
        "mutation",
        "model_seed",
        "epochs",
        "initial_parameter_digest",
        "initial_semantics_digest",
        "epoch_parameter_digests",
        "distinct_epoch_parameter_digests",
        "epoch_order_digests",
        "distinct_epoch_order_digests",
        "semantic_side_weight_effective",
        "null_side_to_semantic_event_mass_weight_effective",
        "null_side_to_presence_calibration_weight_effective",
        "presence_side_disconnect_preserves_semantic_event_mass_gating",
        "null_update_frozen_at_initial_prior",
        "disagreement_gate",
        "argmax_prefilter",
        "all_candidate_events_enter_posterior",
        "factorized_presence_and_conditional_event_posteriors",
        "presence_calibration_uses_training_only_side_soft_labels",
        "presence_calibration_test_feature_is_side_free_semantic_mismatch",
        "semantic_disagreement_count",
        "side_changed_event_top_count",
        "mean_null_posterior",
        "mean_event_posterior_mass",
        "minimum_event_posterior_mass",
        "semantic_update_scaled_by_event_mass",
        "mean_semantic_update_l1",
        "mean_null_update_absolute",
        "presence_intercept",
        "presence_mismatch_weight",
        "final_semantics_digest",
        "exact_window_null_selection_count",
        "model_digest",
        "oracle_fields_consumed",
        "evaluation_keys_consumed",
    }
    for corpus_seed in expected_corpora:
        synchronized_rows = []
        for model_seed in expected_models:
            unit = results_by_unit[(corpus_seed, model_seed)]
            condition_values = list(unit["condition_results"])
            condition_rows = {str(row["condition"]): row for row in condition_values}
            if not (
                len(condition_values) == len(expected_conditions)
                and len(condition_rows) == len(expected_conditions)
                and sorted(condition_rows) == sorted(expected_conditions)
            ):
                raise RuntimeError("development condition-result inventory mismatch")
            row = condition_rows["synchronized"]
            trace = row["trace"]
            model = row["model"]
            epochs = int(config["learner"]["epochs"])
            epoch_parameter_digests = trace.get("epoch_parameter_digests", [])
            epoch_order_digests = trace.get("epoch_order_digests", [])
            expected_model_fields = {
                "schema_version",
                "semantics",
                "presence_intercept",
                "presence_mismatch_weight",
                "model_seed",
                "epochs",
                "learner",
                "training_side_state_serialized",
                "detector_serialized",
                "oracle_serialized",
                "evaluation_key_serialized",
            }
            model_state = {
                "semantics": model.get("semantics") if isinstance(model, Mapping) else None,
                "presence_intercept": model.get("presence_intercept") if isinstance(model, Mapping) else None,
                "presence_mismatch_weight": model.get("presence_mismatch_weight") if isinstance(model, Mapping) else None,
            }
            model_state_numeric = (
                isinstance(model_state["semantics"], Mapping)
                and bool(model_state["semantics"])
                and all(
                    isinstance(values, list)
                    and bool(values)
                    and all(is_finite_number(value) and float(value) >= 0.0 for value in values)
                    and math.isclose(
                        math.fsum(map(float, values)), 1.0, rel_tol=0.0, abs_tol=1e-12
                    )
                    for values in model_state["semantics"].values()
                )
                and is_finite_number(model_state["presence_intercept"])
                and is_finite_number(model_state["presence_mismatch_weight"])
                and 0.0 <= float(model_state["presence_mismatch_weight"])
                <= float(config["learner"]["maximum_presence_mismatch_weight"])
            )
            primitive_keys = sorted(
                key for key in model_state["semantics"] if str(key).startswith("primitive|")
            ) if isinstance(model_state["semantics"], Mapping) else []
            manner_keys = sorted(
                key for key in model_state["semantics"] if str(key).startswith("manner|")
            ) if isinstance(model_state["semantics"], Mapping) else []
            model_token_inventory_exact = (
                len(primitive_keys) == int(config["design"]["primitive_concepts"])
                and len(manner_keys) == int(config["design"]["manner_concepts"])
                and set(primitive_keys) | set(manner_keys)
                == set(model_state["semantics"])
                and all(
                    len(model_state["semantics"][key])
                    == int(config["design"]["primitive_concepts"])
                    for key in primitive_keys
                )
                and all(
                    len(model_state["semantics"][key])
                    == int(config["design"]["manner_concepts"])
                    for key in manner_keys
                )
            )
            posterior_values = (
                trace.get("mean_null_posterior"),
                trace.get("mean_event_posterior_mass"),
                trace.get("minimum_event_posterior_mass"),
                trace.get("mean_semantic_update_l1"),
                trace.get("mean_null_update_absolute"),
            )
            posterior_numeric = all(is_finite_number(value) for value in posterior_values)
            maximum_training_updates = epochs * int(audits_by_seed[corpus_seed]["episode_count"])
            if not (
                set(trace) == expected_trace_fields
                and trace.get("schema_version")
                == "nursery-corrective-training-trace-v3"
                and trace.get("condition") == "synchronized"
                and trace.get("mutation") == "active"
                and int(trace.get("model_seed", -1)) == model_seed
                and int(trace.get("epochs", -1)) == epochs
                and len(epoch_parameter_digests) == epochs
                and len(epoch_order_digests) == epochs
                and is_sha256(trace.get("initial_parameter_digest"))
                and is_sha256(trace.get("initial_semantics_digest"))
                and all(is_sha256(value) for value in epoch_parameter_digests)
                and all(is_sha256(value) for value in epoch_order_digests)
                and int(trace.get("distinct_epoch_parameter_digests", -1))
                == len(set(epoch_parameter_digests))
                and int(trace.get("distinct_epoch_order_digests", -1))
                == len(set(epoch_order_digests))
                and int(trace.get("distinct_epoch_parameter_digests", 0)) > 1
                and int(trace.get("distinct_epoch_order_digests", 0)) > 1
                and isinstance(model, Mapping)
                and set(model) == expected_model_fields
                and model.get("schema_version") == "nursery-corrective-lexicon-model-v3"
                and model.get("learner") == config["learner"]["name"]
                and int(model.get("epochs", -1)) == epochs
                and model.get("training_side_state_serialized") is False
                and model.get("detector_serialized") is False
                and model.get("oracle_serialized") is False
                and model.get("evaluation_key_serialized") is False
                and model_state_numeric
                and model_token_inventory_exact
                and epoch_parameter_digests[-1] == canonical_digest(model_state)
                and trace.get("semantic_side_weight_effective")
                == float(config["learner"]["semantic_side_weight"])
                and trace.get(
                    "null_side_to_semantic_event_mass_weight_effective"
                )
                == float(config["learner"]["null_side_weight"])
                and trace.get(
                    "null_side_to_presence_calibration_weight_effective"
                )
                == float(config["learner"]["null_side_weight"])
                and trace.get(
                    "presence_side_disconnect_preserves_semantic_event_mass_gating"
                )
                is False
                and trace.get("null_update_frozen_at_initial_prior") is False
                and trace.get("disagreement_gate") is False
                and trace.get("argmax_prefilter") is False
                and trace.get("all_candidate_events_enter_posterior") is True
                and trace.get("factorized_presence_and_conditional_event_posteriors") is True
                and trace.get("presence_calibration_uses_training_only_side_soft_labels") is True
                and trace.get("presence_calibration_test_feature_is_side_free_semantic_mismatch") is True
                and trace.get("semantic_update_scaled_by_event_mass") is True
                and is_exact_int(trace.get("semantic_disagreement_count"), minimum=1)
                and int(trace["semantic_disagreement_count"]) <= maximum_training_updates
                and is_exact_int(trace.get("side_changed_event_top_count"), minimum=1)
                and int(trace["side_changed_event_top_count"]) <= maximum_training_updates
                and posterior_numeric
                and 0.0 < float(trace["mean_null_posterior"]) < 1.0
                and 0.0 < float(trace["mean_event_posterior_mass"]) < 1.0
                and math.isclose(
                    float(trace["mean_null_posterior"])
                    + float(trace["mean_event_posterior_mass"]),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and 0.0 < float(trace["minimum_event_posterior_mass"])
                <= float(trace["mean_event_posterior_mass"])
                and float(trace["mean_semantic_update_l1"]) > 0.0
                and float(trace["mean_null_update_absolute"]) > 0.0
                and trace.get("presence_intercept") == model.get("presence_intercept")
                and trace.get("presence_mismatch_weight") == model.get("presence_mismatch_weight")
                and trace.get("final_semantics_digest")
                == canonical_digest(model.get("semantics"))
                and is_exact_int(trace.get("exact_window_null_selection_count"), minimum=0)
                and trace.get("oracle_fields_consumed") is False
                and trace.get("evaluation_keys_consumed") is False
                and canonical_digest(model) == row.get("model_digest")
                and trace.get("model_digest") == row.get("model_digest")
                and is_sha256(row.get("model_digest"))
                and is_sha256(row.get("prediction_digest"))
                and int(model.get("model_seed", -1)) == model_seed
            ):
                raise RuntimeError(
                    "development synchronized training trace contract mismatch"
                )
            synchronized_rows.append(row)
        initial_digests = [
            str(row["trace"]["initial_parameter_digest"])
            for row in synchronized_rows
        ]
        order_digests = [
            canonical_digest(row["trace"]["epoch_order_digests"])
            for row in synchronized_rows
        ]
        fitted_digests = [
            canonical_digest(
                {
                    "semantics": row["model"]["semantics"],
                    "presence_intercept": row["model"]["presence_intercept"],
                    "presence_mismatch_weight": row["model"]["presence_mismatch_weight"],
                }
            )
            for row in synchronized_rows
        ]
        prediction_digests = [
            str(row["prediction_digest"]) for row in synchronized_rows
        ]
        model_trace_rows.append(
            {
                "corpus_seed": corpus_seed,
                "model_seed_order": expected_models,
                "distinct_initial_parameter_digests": len(set(initial_digests)),
                "distinct_epoch_order_sequences": len(set(order_digests)),
                "distinct_fitted_model_digests": len(set(fitted_digests)),
                "distinct_prediction_digests": len(set(prediction_digests)),
            }
        )
    averaged_rows = list(averaged["rows"])
    expected_average_row_fields = {
        "corpus_seed",
        "condition",
        "model_seed_order",
        "model_replicates_averaged",
        "metrics",
    }
    average_row_keys = set()
    average_metric_by_key: dict[tuple[int, str, str], float] = {}
    for row in averaged_rows:
        if not (
            isinstance(row, Mapping)
            and set(row) == expected_average_row_fields
            and row.get("model_seed_order") == expected_models
            and row.get("model_replicates_averaged") == len(expected_models)
            and isinstance(row.get("metrics"), Mapping)
            and set(row["metrics"]) == averaged_metric_names
            and all(is_finite_number(value) for value in row["metrics"].values())
        ):
            raise RuntimeError("development averaged row contract mismatch")
        average_row_keys.add((int(row["corpus_seed"]), str(row["condition"])))
        for metric in variation_metric_names:
            average_metric_by_key[
                (int(row["corpus_seed"]), str(row["condition"]), metric)
            ] = float(row["metrics"][metric])
    expected_average_row_keys = {
        (corpus_seed, condition)
        for corpus_seed in expected_corpora
        for condition in expected_conditions
    }
    if len(averaged_rows) != len(expected_average_row_keys) or average_row_keys != expected_average_row_keys:
        raise RuntimeError("development averaged row inventory mismatch")

    variation_rows = list(averaged["within_corpus_model_variation"])
    expected_variation_count = (
        len(expected_corpora) * len(expected_conditions) * len(variation_metric_names)
    )
    expected_variation_fields = {
        "corpus_seed",
        "condition",
        "metric",
        "sample_sd",
        "range",
        "values",
    }
    variation_keys = set()
    variation_mean_bindings = []
    for row in variation_rows:
        values = row.get("values") if isinstance(row, Mapping) else None
        if not (
            isinstance(row, Mapping)
            and set(row) == expected_variation_fields
            and isinstance(values, list)
            and len(values) == len(expected_models)
            and all(is_finite_number(value) for value in values)
            and is_finite_number(row.get("sample_sd"))
            and float(row["sample_sd"]) >= 0.0
            and is_finite_number(row.get("range"))
            and float(row["range"]) >= 0.0
        ):
            raise RuntimeError("development model-variation row contract mismatch")
        values_float = list(map(float, values))
        mean = math.fsum(values_float) / len(values_float)
        recomputed_sd = math.sqrt(
            math.fsum((value - mean) ** 2 for value in values_float)
            / (len(values_float) - 1)
        )
        recomputed_range = max(values_float) - min(values_float)
        if not (
            math.isclose(
                float(row["sample_sd"]), recomputed_sd, rel_tol=1e-12, abs_tol=1e-15
            )
            and math.isclose(
                float(row["range"]), recomputed_range, rel_tol=1e-12, abs_tol=1e-15
            )
        ):
            raise RuntimeError("development model-variation statistics mismatch")
        variation_key = (
            int(row["corpus_seed"]),
            str(row["condition"]),
            str(row["metric"]),
        )
        variation_keys.add(variation_key)
        variation_mean_bindings.append(
            variation_key in average_metric_by_key
            and math.isclose(
                mean,
                average_metric_by_key.get(variation_key, math.nan),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
    expected_variation_keys = {
        (corpus_seed, condition, metric)
        for corpus_seed in expected_corpora
        for condition in expected_conditions
        for metric in variation_metric_names
    }
    if (
        len(variation_rows) != expected_variation_count
        or variation_keys != expected_variation_keys
    ):
        raise RuntimeError("development model-variation inventory mismatch")
    if not all(variation_mean_bindings):
        raise RuntimeError("development model-variation mean binding mismatch")
    eligible_variation_metrics = set(
        map(
            str,
            config["qualification_gates"]["model_variation_gate_metrics"],
        )
    )
    excluded_variation_conditions = set(
        map(
            str,
            config["qualification_gates"][
                "model_variation_excluded_conditions"
            ],
        )
    )
    required_variation_conditions = set(
        map(
            str,
            config["qualification_gates"][
                "model_variation_required_conditions"
            ],
        )
    )
    if (
        not required_variation_conditions
        or required_variation_conditions & excluded_variation_conditions
        or not required_variation_conditions.issubset(set(expected_conditions))
    ):
        raise RuntimeError("development model-variation condition contract mismatch")
    eligible_variation_rows = [
        row
        for row in variation_rows
        if str(row["metric"]) in eligible_variation_metrics
        and str(row["condition"]) in required_variation_conditions
    ]
    expected_eligible_variation_count = (
        len(expected_corpora)
        * len(required_variation_conditions)
        * len(eligible_variation_metrics)
    )
    nonzero_flags = [
        float(row["sample_sd"])
        >= float(
            config["qualification_gates"][
                "minimum_nonzero_within_corpus_model_metric_sd"
            ]
        )
        for row in eligible_variation_rows
    ]
    nonzero_variation_fraction = float(
        math.fsum(1.0 if value else 0.0 for value in nonzero_flags)
        / len(nonzero_flags)
    )
    nonzero_variation_fraction_by_metric = {
        metric: float(
            math.fsum(
                1.0
                if float(row["sample_sd"])
                >= float(
                    config["qualification_gates"][
                        "minimum_nonzero_within_corpus_model_metric_sd"
                    ]
                )
                else 0.0
                for row in eligible_variation_rows
                if str(row["metric"]) == metric
            )
            / sum(
                1
                for row in eligible_variation_rows
                if str(row["metric"]) == metric
            )
        )
        for metric in sorted(eligible_variation_metrics)
    }
    nonzero_variation_fraction_by_metric_condition = {
        f"{metric}|{condition}": float(
            math.fsum(
                1.0
                if float(row["sample_sd"])
                >= float(
                    config["qualification_gates"][
                        "minimum_nonzero_within_corpus_model_metric_sd"
                    ]
                )
                else 0.0
                for row in eligible_variation_rows
                if str(row["metric"]) == metric
                and str(row["condition"]) == condition
            )
            / sum(
                1
                for row in eligible_variation_rows
                if str(row["metric"]) == metric
                and str(row["condition"]) == condition
            )
        )
        for metric in sorted(eligible_variation_metrics)
        for condition in sorted(required_variation_conditions)
    }
    average_table = {
        (int(row["corpus_seed"]), str(row["condition"])): row["metrics"]
        for row in averaged_rows
    }
    primary_effect_vectors = {
        endpoint: [
            float(average_table[(seed, "synchronized")][endpoint])
            - max(
                float(average_table[(seed, "absent")][endpoint]),
                float(average_table[(seed, "shuffled")][endpoint]),
            )
            for seed in expected_corpora
        ]
        for endpoint in config["analysis"]["co_primary_endpoints"]
    }
    zero_exposure_passed = all(
        float(row["metrics"]["zero_exposure.strict_top1"]) == 0.0
        and float(row["metrics"]["zero_exposure.tie_rate"]) == 1.0
        for row in averaged_rows
    )
    mutation_state_audit = mechanism_mutation_state_selectivity(
        merged_results, config
    )
    gates = config["qualification_gates"]
    factor_draw_distinct_minimum = int(
        config["development_validity_gates"]["minimum_distinct_factor_draws"]
    )
    checks = {
        "exact_complete_corpus_audit_set": len(corpus_audits)
        == len(expected_corpora)
        and sorted(audits_by_seed) == sorted(expected_corpora),
        "exact_complete_model_unit_set": len(merged_results)
        == len(expected_corpora) * len(expected_models)
        and set(results_by_unit) == expected_unit_keys,
        "action_geometry_varies": action_geometry_count
        >= scaled_distinct_minimum("minimum_distinct_action_geometry_digests"),
        "episode_geometry_varies": episode_geometry_count
        >= scaled_distinct_minimum("minimum_distinct_episode_geometry_digests"),
        "heldout_splits_vary": heldout_split_count
        >= scaled_distinct_minimum("minimum_distinct_heldout_split_digests"),
        "repetition_patterns_vary": repetition_count
        >= scaled_distinct_minimum("minimum_distinct_repetition_digests"),
        "ambiguity_strata_vary": len(ambiguity_strata)
        >= int(gates["minimum_distinct_ambiguity_strata"])
        and all(
            count
            >= int(
                config["development_validity_gates"][
                    "minimum_corpora_per_ambiguity_stratum"
                ]
            )
            for count in ambiguity_stratum_counts.values()
        ),
        "realized_candidate_counts_vary": candidate_histogram_count
        >= scaled_distinct_minimum("minimum_distinct_candidate_count_histograms"),
        "realized_grounded_counts_vary": grounded_count_count
        >= scaled_distinct_minimum("minimum_distinct_grounded_episode_counts"),
        "all_frozen_factor_draws_vary": all(
            count >= factor_draw_distinct_minimum
            for count in factor_distinct_counts.values()
        ),
        "action_geometry_distance_varies": min(
            realized_ranges[
                "primitive_action_geometry_mean_pairwise_distance"
            ],
            realized_ranges["manner_action_geometry_mean_pairwise_distance"],
        )
        >= float(gates["minimum_action_geometry_pairwise_distance_range"]),
        "realized_grounded_rate_varies": realized_ranges["grounded_rate"]
        >= float(gates["minimum_realized_grounded_rate_range"]),
        "realized_ambiguity_varies": realized_ranges["foil_present_rate"]
        >= float(gates["minimum_realized_foil_rate_range"]),
        "realized_lag_varies": realized_ranges["observed_lag_mean"]
        >= float(gates["minimum_observed_lag_mean_range"]),
        "realized_visibility_varies": realized_ranges["observation_peak"]
        >= float(gates["minimum_realized_visibility_peak_range"]),
        "realized_side_informativeness_varies": realized_ranges[
            "detector_accuracy"
        ]
        >= float(gates["minimum_realized_detector_accuracy_range"]),
        "action_geometry_separated_and_shared": all(
            row["action_geometry"]["all_rows_normalized"] is True
            and row["action_geometry"]["all_entries_nonnegative"] is True
            and row["action_geometry"]["all_margins_pass"] is True
            and row["train_evaluation_use_same_action_prototypes"] is True
            and min(
                float(
                    row["action_geometry"]["slots"][slot][
                        "minimum_self_cross_dot_margin"
                    ]
                )
                for slot in ("primitive", "manner")
            )
            >= float(gates["minimum_action_geometry_self_cross_dot_margin"])
            for row in corpus_audits
        ),
        "matched_shuffle_and_leakage_guards": all(
            row["condition_audit"][
                "shuffle_preserves_learner_visible_evidence_marginal_by_block"
            ] is True
            and row["visible_identifiers_are_opaque_and_truth_independent_in_format"]
            is True
            and row["train_evaluation_generator_namespaces_disjoint"] is True
            and is_exact_int(
                row["provenance"]["train_evaluation_instance_id_overlap"],
                minimum=0,
            )
            and row["provenance"]["train_evaluation_instance_id_overlap"] == 0
            and row["every_constituent_exposed_in_training"] is True
            and row["all_training_candidate_events_exclude_heldout_compositions"]
            is True
            and is_exact_int(row["heldout_candidate_event_count"], minimum=0)
            and row["heldout_candidate_event_count"] == 0
            and row["evaluation_side_fields_absent"] is True
            and row["evaluation_top_level_schema_exact_allowlist"] is True
            for row in corpus_audits
        ),
        "zero_exposure_controls_remain_unlearned": zero_exposure_passed,
        "initialization_order_and_fitted_states_all_vary": all(
            row["distinct_initial_parameter_digests"] == len(expected_models)
            and row["distinct_epoch_order_sequences"] == len(expected_models)
            and row["distinct_fitted_model_digests"] == len(expected_models)
            and row["distinct_prediction_digests"] == len(expected_models)
            for row in model_trace_rows
        ),
        "model_metric_variation_inventory_exact": len(variation_rows)
        == expected_variation_count
        and variation_keys == expected_variation_keys,
        "eligible_model_metric_variation_inventory_exact": len(
            eligible_variation_rows
        )
        == expected_eligible_variation_count,
        "model_metric_variation_values_bound_to_averages": all(
            variation_mean_bindings
        ),
        "model_metric_variation_is_operational": all(
            fraction
            >= float(
                config["qualification_gates"][
                    "minimum_nonzero_continuous_model_variation_fraction"
                ]
            )
            for fraction in nonzero_variation_fraction_by_metric_condition.values()
        ),
        "mechanism_mutation_state_selectivity": mutation_state_audit["status"]
        == "PASS",
        "primary_effect_vectors_are_non_degenerate": (
            not bool(config["analysis"]["all_identical_effects_fail"])
            or all(
                len(set(values)) > 1
                or values[0]
                <= float(
                    config["analysis"]["primary_lower_bound_must_exceed"]
                )
                for values in primary_effect_vectors.values()
            )
        ),
    }
    return {
        "schema_version": "nursery-corrective-development-validity-v4",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "purpose": purpose,
        "checks": checks,
        "action_geometry_digest_count": action_geometry_count,
        "episode_geometry_digest_count": episode_geometry_count,
        "heldout_split_digest_count": heldout_split_count,
        "repetition_digest_count": repetition_count,
        "candidate_histogram_count": candidate_histogram_count,
        "grounded_episode_count_distinct_count": grounded_count_count,
        "ambiguity_strata": ambiguity_strata,
        "ambiguity_stratum_counts": ambiguity_stratum_counts,
        "factor_distinct_counts": factor_distinct_counts,
        "factor_draw_distinct_minimum": factor_draw_distinct_minimum,
        "scaled_distinct_minima": {
            name: scaled_distinct_minimum(name)
            for name in (
                "minimum_distinct_action_geometry_digests",
                "minimum_distinct_episode_geometry_digests",
                "minimum_distinct_heldout_split_digests",
                "minimum_distinct_repetition_digests",
                "minimum_distinct_candidate_count_histograms",
                "minimum_distinct_grounded_episode_counts",
            )
        },
        "realized_factor_ranges": realized_ranges,
        "model_trace_rows": model_trace_rows,
        "nonzero_model_metric_variation_fraction": nonzero_variation_fraction,
        "nonzero_model_metric_variation_fraction_by_metric": (
            nonzero_variation_fraction_by_metric
        ),
        "nonzero_model_metric_variation_fraction_by_metric_condition": (
            nonzero_variation_fraction_by_metric_condition
        ),
        "primary_effect_vectors": primary_effect_vectors,
        "mechanism_mutation_state_selectivity": mutation_state_audit,
        "eligible_model_metric_variation_row_count": len(
            eligible_variation_rows
        ),
        "expected_eligible_model_metric_variation_row_count": (
            expected_eligible_variation_count
        ),
        "eligible_model_metric_variation_metrics": sorted(
            eligible_variation_metrics
        ),
        "model_metric_variation_excluded_conditions": sorted(
            excluded_variation_conditions
        ),
        "model_metric_variation_required_conditions": sorted(
            required_variation_conditions
        ),
        "model_variation_is_descriptive_not_an_efficacy_estimand": True,
        "no_unit_exclusion_or_replacement": True,
        "scientific_outcome": False,
    }


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
            != (snapshot / "configs/synthetic_corrective_alignment_v4.yaml").resolve()
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
            / "configs/synthetic_corrective_alignment_v4.yaml"
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
    runner_path = repository / "scripts/run_synthetic_corrective_alignment_v4.py"
    worker_entrypoint = repository / str(config["parallel"]["worker_entrypoint"])
    python_executable = Path(sys.executable).absolute()
    if (
        config["parallel"].get("backend") != "direct_subprocess_v1"
        or config["parallel"].get("wave_key") != "model_seed"
        or config["parallel"].get("subprocess_new_session") is not True
        or config["parallel"].get("worker_requests_manifested") is not True
        or not worker_entrypoint.is_file()
        or not python_executable.is_file()
    ):
        raise RuntimeError("direct-subprocess execution contract mismatch")
    snapshot_manifest = repository / "snapshot_manifest.json"
    freeze_receipt = repository.parent / "freeze_receipt.json"
    worker_boundary = create_worker_boundary_commitment(
        repository_root=repository,
        config=config,
        prequalification_lock=prequalification_lock,
    )
    locked_environment = read_json(prequalification_lock)["qualification_environment"]
    locked_launcher_evidence = {
        "chain": locked_environment["python_launcher_chain"],
        "chain_digest": locked_environment["python_launcher_chain_digest"],
        "pyvenv_cfg_path": locked_environment["pyvenv_cfg_path"],
        "pyvenv_cfg_sha256": locked_environment["pyvenv_cfg_sha256"],
    }
    if (
        _python_launcher_evidence(python_executable) != locked_launcher_evidence
        or worker_boundary.get("python_launcher_chain")
        != locked_launcher_evidence["chain"]
        or worker_boundary.get("python_launcher_chain_digest")
        != locked_launcher_evidence["chain_digest"]
        or worker_boundary.get("pyvenv_cfg_path")
        != locked_launcher_evidence["pyvenv_cfg_path"]
        or worker_boundary.get("pyvenv_cfg_sha256")
        != locked_launcher_evidence["pyvenv_cfg_sha256"]
    ):
        raise RuntimeError("worker launcher differs from prospective environment lock")
    worker_boundary_verification = verify_worker_boundary_commitment(
        worker_boundary,
        repository_root=repository,
        config=config,
    )
    if worker_boundary_verification.get("status") != "PASS":
        raise RuntimeError("worker operational boundary failed before scheduling")
    scientific_contract = {
        "schema_version": "nursery-corrective-scientific-execution-contract-v4",
        "purpose": purpose,
        "protocol_id": config["protocol"]["id"],
        "config_sha256": sha256_file(config_file),
        "runner_sha256": sha256_file(runner_path),
        "worker_entrypoint_sha256": sha256_file(worker_entrypoint),
        "parallel_backend": str(config["parallel"]["backend"]),
        "snapshot_manifest_sha256": (
            sha256_file(snapshot_manifest) if snapshot_manifest.is_file() else None
        ),
        "freeze_receipt_sha256": (
            sha256_file(freeze_receipt) if freeze_receipt.is_file() else None
        ),
        "prequalification_design_lock_sha256": prequalification_lock_sha256,
        "worker_boundary_digest": worker_boundary["boundary_digest"],
        "python_launcher_evidence": locked_launcher_evidence,
        "input_manifest_sha256": input_manifest_sha256,
        "work_plan_digest": plan["digest"],
        "authorization_digest": capability_digest or None,
        "parsed_contract_digest": parsed_contract_digest or None,
    }
    scientific_contract_digest = canonical_digest(scientific_contract)
    write_json(
        adjudication / "execution_contract.json",
        {**scientific_contract, "scientific_contract_digest": scientific_contract_digest},
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
            "worker_boundary_commitment": worker_boundary,
            "expected_launcher_evidence": locked_launcher_evidence,
            "worker_python_startup_flags": dict(
                config["environment"]["worker_python_startup_flags"]
            ),
        }
        for unit in plan["units"]
    }
    child_environment_base = _child_environment_base(
        inherited_allowlist=config["parallel"][
            "child_inherited_environment_allowlist"
        ],
        thread_environment=config["parallel"]["thread_environment"],
    )
    scheduler = _prepare_subprocess_requests(
        plan=plan,
        arguments_by_id=arguments_by_id,
        runtime_root=telemetry_root,
        child_environment_base=child_environment_base,
    )
    runtime_contract = {
        "schema_version": "nursery-corrective-runtime-execution-contract-v4",
        "scientific_contract_digest": scientific_contract_digest,
        "jobs": int(jobs),
        "thread_environment": dict(config["parallel"]["thread_environment"]),
        "parent_python_startup_flags": dict(
            config["environment"]["parent_python_startup_flags"]
        ),
        "worker_python_startup_flags": dict(
            config["environment"]["worker_python_startup_flags"]
        ),
        "child_inherited_environment_allowlist": list(
            config["parallel"]["child_inherited_environment_allowlist"]
        ),
        "child_environment_base": child_environment_base,
        "forbidden_unbound_environment": list(
            config["parallel"]["forbidden_unbound_environment"]
        ),
        "backend": str(config["parallel"]["backend"]),
        "wave_key": str(config["parallel"]["wave_key"]),
        "python_executable": str(python_executable),
        "python_launcher_evidence": locked_launcher_evidence,
        "worker_entrypoint": str(worker_entrypoint.resolve()),
        "worker_entrypoint_sha256": sha256_file(worker_entrypoint),
        "request_manifest_sha256": str(scheduler["request_manifest_sha256"]),
        "subprocess_new_session": True,
    }
    runtime_contract_digest = canonical_digest(runtime_contract)
    write_json(
        telemetry_root / "runtime_contract.json",
        {**runtime_contract, "runtime_contract_digest": runtime_contract_digest},
    )
    telemetry, failures = _execute_worker_batches(
        plan=plan,
        jobs=jobs,
        scheduler=scheduler,
        repository_root=repository,
        worker_entrypoint=worker_entrypoint,
        python_executable=python_executable,
        child_environment_base=child_environment_base,
        expected_launcher_evidence=locked_launcher_evidence,
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
    scheduler_validation = validate_subprocess_scheduler_artifacts(
        runtime_root=telemetry_root,
        shard_root=shards,
        plan=plan,
        jobs=jobs,
        config=config,
        expected_repository_root=repository,
        expected_config_path=config_file,
        child_environment_base=child_environment_base,
        input_root=input_base,
        expected_input_manifest_sha256=input_manifest_sha256,
        expected_scientific_contract_digest=scientific_contract_digest,
        persist=True,
    )
    if scheduler_validation.get("status") != "PASS":
        raise RuntimeError("persisted subprocess scheduler validation failed")
    post_worker_input_verification = verify_persisted_inputs(input_base)
    post_worker_boundary_verification = verify_worker_boundary_commitment(
        worker_boundary,
        repository_root=repository,
        config=config,
    )
    if (
        post_worker_input_verification.get("status") != "PASS"
        or post_worker_boundary_verification.get("status") != "PASS"
        or sha256_file(input_base / "input_manifest.json")
        != input_manifest_sha256
        or sha256_file(prequalification_lock) != prequalification_lock_sha256
        or sha256_file(worker_entrypoint)
        != scientific_contract["worker_entrypoint_sha256"]
        or sha256_file(
            Path(scheduler["requests"]) / "manifest.json"
        )
        != scheduler["request_manifest_sha256"]
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
        "bootstrap_environment_digest",
        "worker_boundary_digest",
        "worker_boundary_operational_file_count",
        "worker_input_regular_files_verified",
        "worker_full_tree_verifications",
        "thread_environment_observed",
        "native_threadpools",
        "python_startup_flags_observed",
    }
    if (
        len(telemetry_ids) != len(set(telemetry_ids))
        or set(telemetry_ids) != set(expected_unit_ids)
        or any(set(row) != expected_telemetry_fields for row in telemetry)
        or any(
            row.get("thread_environment_observed")
            != {
                str(name): str(value)
                for name, value in config["parallel"]["thread_environment"].items()
            }
            for row in telemetry
        )
        or any(
            row.get("python_startup_flags_observed")
            != config["environment"]["worker_python_startup_flags"]
            for row in telemetry
        )
        or any(
            row.get("bootstrap_environment_digest")
            != canonical_digest(
                {
                    **child_environment_base,
                    "TMPDIR": str(
                        (Path(scheduler["bootstrap_tmp"]) / str(row.get("unit_id"))).resolve()
                    ),
                }
            )
            for row in telemetry
        )
        or any(
            row.get("worker_boundary_digest")
            != worker_boundary["boundary_digest"]
            or int(row.get("worker_boundary_operational_file_count", -1))
            != int(worker_boundary_verification["operational_file_count"])
            or int(row.get("worker_input_regular_files_verified", -1)) != 1
            or int(row.get("worker_full_tree_verifications", -1)) != 0
            for row in telemetry
        )
        or any(
            int(pool.get("num_threads", -1)) != 1
            for row in telemetry
            for pool in row.get("native_threadpools", [])
        )
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
    development_validity = _development_validity_audit(
        purpose=purpose,
        averaged=averaged,
        merged_results=merged_results,
        corpus_audits=corpus_audits,
        config=config,
    )
    candidate_decision = _candidate_scientific_decision(
        purpose=purpose,
        development_validity_status=str(development_validity["status"]),
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
    write_json(adjudication / "development_validity.json", development_validity)
    write_json(
        adjudication / "scientific_summary.json",
        {
            "schema_version": "nursery-corrective-scientific-summary-v4",
            "purpose": purpose,
            "development_validity_status": development_validity["status"],
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
        "schema_version": "nursery-corrective-completeness-v4",
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
    expected_pre_manifest_files = ADJUDICATION_RELEVANT_FILES - {"manifest.json"}
    if set(tree_file_paths(adjudication)) != expected_pre_manifest_files:
        raise RuntimeError("adjudication file inventory incomplete before manifest")
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
    if set(tree_file_paths(adjudication)) != ADJUDICATION_RELEVANT_FILES:
        raise RuntimeError("adjudication file inventory changed after manifest")
    telemetry_sorted = sorted(telemetry, key=lambda row: row["unit_id"])
    write_json(
        telemetry_root / "runtime_telemetry.json",
        {
            "schema_version": "nursery-corrective-runtime-v4",
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
    final_worker_boundary_verification = verify_worker_boundary_commitment(
        worker_boundary,
        repository_root=repository,
        config=config,
    )
    if (
        final_input_verification.get("status") != "PASS"
        or final_worker_boundary_verification.get("status") != "PASS"
        or sha256_file(input_base / "input_manifest.json")
        != input_manifest_sha256
        or sha256_file(prequalification_lock) != prequalification_lock_sha256
    ):
        raise RuntimeError("persisted inputs or design lock changed before sealing")
    if purpose in {"development", "excluded_rehearsal"}:
        _verify_frozen_execution_boundary(repository_root, config)
    summary = {
        "schema_version": "nursery-corrective-cohort-summary-v4",
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
        or config_file != snapshot / "configs/synthetic_corrective_alignment_v4.yaml"
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
        != "nursery-corrective-excluded-rehearsal-completion-v4"
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
        != sha256_file(snapshot / "scripts/run_synthetic_corrective_alignment_v4.py")
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
    development_validity = _development_validity_audit(
        purpose="excluded_rehearsal",
        averaged=averaged,
        merged_results=merged_results,
        corpus_audits=corpus_audits,
        config=config,
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
    write_json(adjudication / "development_validity.json", development_validity)
    write_json(
        adjudication / "scientific_summary.json",
        {
            "schema_version": "nursery-corrective-scientific-summary-v4",
            "purpose": "excluded_rehearsal",
            "development_validity_status": development_validity["status"],
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
        "schema_version": "nursery-corrective-completeness-v4",
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
    expected_pre_manifest_files = ADJUDICATION_RELEVANT_FILES - {"manifest.json"}
    if set(tree_file_paths(adjudication)) != expected_pre_manifest_files:
        raise RuntimeError("recomputed adjudication inventory incomplete before manifest")
    manifest = manifest_for_tree(adjudication, exclude=["manifest.json"])
    write_json(adjudication / "manifest.json", manifest)
    verification = verify_exact_manifest(
        adjudication, manifest, manifest_filename="manifest.json"
    )
    if verification["status"] != "PASS":
        raise RuntimeError(f"recomputed adjudication manifest failed: {verification}")
    if set(tree_file_paths(adjudication)) != ADJUDICATION_RELEVANT_FILES:
        raise RuntimeError("recomputed adjudication inventory changed after manifest")

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
            != (snapshot / "configs/synthetic_corrective_alignment_v4.yaml").resolve()
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
            / "configs/synthetic_corrective_alignment_v4.yaml"
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
    execution_contract = read_json(
        output / "adjudication/execution_contract.json"
    )
    expected_child_environment = _child_environment_base(
        inherited_allowlist=config["parallel"][
            "child_inherited_environment_allowlist"
        ],
        thread_environment=config["parallel"]["thread_environment"],
    )
    relocated_scheduler_validation = validate_subprocess_scheduler_artifacts(
        runtime_root=output / "runtime",
        shard_root=output / "shards",
        plan=build_work_plan(config, purpose=purpose),
        jobs=int(jobs),
        config=config,
        expected_repository_root=repository_root,
        expected_config_path=config_path,
        child_environment_base=expected_child_environment,
        input_root=output / "persisted_inputs",
        expected_input_manifest_sha256=sha256_file(
            output / "persisted_inputs/input_manifest.json"
        ),
        expected_scientific_contract_digest=str(
            execution_contract["scientific_contract_digest"]
        ),
        persist=False,
        relocated_from_cohort_staging=staging,
    )
    persisted_scheduler_validation = read_json(
        output / "runtime/scheduler_validation.json"
    )
    if relocated_scheduler_validation != persisted_scheduler_validation:
        raise RuntimeError(
            "cohort scheduler evidence changed across atomic publication"
        )
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
    expected_config_path = snapshot / "configs/synthetic_corrective_alignment_v4.yaml"
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
    expected_config = snapshot / "configs/synthetic_corrective_alignment_v4.yaml"
    loaded = load_config(expected_config, repository_root=snapshot)
    repository = snapshot.parents[2]
    expected_output = repository / str(loaded["paths"]["excluded_rehearsal"])
    if (
        snapshot.name != "frozen_source_snapshot"
        or snapshot.parent.name != "synthetic_corrective_development_launch_package_v4"
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
        "schema_version": "nursery-corrective-excluded-rehearsal-attempt-v4",
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
