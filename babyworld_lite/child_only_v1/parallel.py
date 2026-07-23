"""Device-aware future bundle planning and fail-closed canonical merge.

This module never trains a learner.  It freezes whole five-condition bundle
assignments for a later, separately authorized outcome task.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from .policy import CONSTRUCTION_PROFILE, PolicyViolation, canonical_digest, canonical_json_bytes
from .protocol import CONDITION_ORDER, condition_bundle_id


THREAD_CAP_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "TORCH_NUM_INTEROP_THREADS": "1",
}


def build_parallel_plan(
    *,
    protocol_digest: str,
    corpus_instance_id: str,
    corpus_seeds: Sequence[int],
    model_seeds: Sequence[int],
    backend: str,
    max_trainers: int,
    cpu_workers: int,
    profile_label: str,
    outcome_task_authorized: bool,
) -> dict[str, Any]:
    """Build canonical whole-bundle shards without launching any process."""

    if profile_label != CONSTRUCTION_PROFILE and not outcome_task_authorized:
        raise PolicyViolation("scientific parallel plan requires a separate authorized outcome task")
    if outcome_task_authorized and profile_label == CONSTRUCTION_PROFILE:
        raise PolicyViolation("construction fixtures can never be authorized as scientific outcomes")
    if backend not in {"mps", "cuda", "slurm_cuda"}:
        raise ValueError("backend must be mps, cuda, or slurm_cuda")
    if max_trainers <= 0 or cpu_workers <= 0:
        raise ValueError("worker limits must be positive")
    if backend == "mps" and max_trainers != 1:
        raise PolicyViolation("a local MPS plan must cap concurrent training at one")
    if len(set(corpus_seeds)) != len(corpus_seeds) or len(set(model_seeds)) != len(model_seeds):
        raise PolicyViolation("seed grids cannot contain duplicates")
    if not corpus_seeds or not model_seeds:
        raise PolicyViolation("seed grids must be nonempty")

    bundles: list[dict[str, Any]] = []
    for corpus_seed in sorted(corpus_seeds):
        for model_seed in sorted(model_seeds):
            bundle_id = condition_bundle_id(protocol_digest, corpus_instance_id, corpus_seed, model_seed)
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "corpus_instance_id": corpus_instance_id,
                    "corpus_seed": corpus_seed,
                    "model_seed": model_seed,
                    "conditions_serial_on_one_device": list(CONDITION_ORDER),
                    "worker_creates_child_pool": False,
                }
            )
    plan_core = {
        "plan_version": "child-only-parallel-plan-v1",
        "profile_label": profile_label,
        "outcome_task_authorized": outcome_task_authorized,
        "protocol_digest": protocol_digest,
        "backend": backend,
        "max_concurrent_trainers": max_trainers,
        "cpu_generation_evaluation_workers": cpu_workers,
        "thread_cap_environment": THREAD_CAP_ENV,
        "atomic_unit": "COMPLETE_CORPUS_SEED_BY_MODEL_SEED_FIVE_CONDITION_BUNDLE",
        "condition_sharding": "FORBIDDEN",
        "bundle_execution_order": "CONDITIONS_SERIAL_ON_ASSIGNED_DEVICE",
        "worker_output_policy": "PRIVATE_TEMP_DIRECTORY_THEN_ATOMIC_RENAME_TO_BUNDLE_ID",
        "resume_policy": "ONLY_MISSING_EXPECTED_BUNDLES_UNDER_IDENTICAL_PLAN_DIGEST",
        "slurm_array_index_to_bundle_id": {
            str(index): bundle["bundle_id"] for index, bundle in enumerate(bundles)
        },
        "bundles": bundles,
    }
    return {**plan_core, "plan_digest": canonical_digest(plan_core)}


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "plan_version",
        "profile_label",
        "outcome_task_authorized",
        "protocol_digest",
        "backend",
        "max_concurrent_trainers",
        "cpu_generation_evaluation_workers",
        "thread_cap_environment",
        "atomic_unit",
        "condition_sharding",
        "bundle_execution_order",
        "worker_output_policy",
        "resume_policy",
        "slurm_array_index_to_bundle_id",
        "bundles",
        "plan_digest",
    }
    if set(plan) != expected or plan["plan_version"] != "child-only-parallel-plan-v1":
        raise PolicyViolation("parallel plan schema/version mismatch")
    if plan["thread_cap_environment"] != THREAD_CAP_ENV:
        raise PolicyViolation("parallel plan does not prevent nested thread oversubscription")
    if plan["backend"] == "mps" and plan["max_concurrent_trainers"] != 1:
        raise PolicyViolation("MPS plan exceeds one concurrent trainer")
    bundles = plan["bundles"]
    ids = [bundle["bundle_id"] for bundle in bundles]
    seed_order = [(bundle["corpus_seed"], bundle["model_seed"]) for bundle in bundles]
    if seed_order != sorted(seed_order):
        raise PolicyViolation("bundle enumeration is not canonical")
    if len(ids) != len(set(ids)):
        raise PolicyViolation("parallel plan contains duplicate bundle IDs")
    for bundle in bundles:
        if bundle["conditions_serial_on_one_device"] != list(CONDITION_ORDER):
            raise PolicyViolation("conditions were split or reordered inside a coupled bundle")
        expected_id = condition_bundle_id(
            plan["protocol_digest"],
            bundle["corpus_instance_id"],
            bundle["corpus_seed"],
            bundle["model_seed"],
        )
        if bundle["bundle_id"] != expected_id:
            raise PolicyViolation("bundle ID does not match the frozen plan inputs")
    expected_array_map = {str(index): bundle["bundle_id"] for index, bundle in enumerate(bundles)}
    if plan["slurm_array_index_to_bundle_id"] != expected_array_map:
        raise PolicyViolation("Slurm array mapping differs from canonical bundle enumeration")
    if plan["condition_sharding"] != "FORBIDDEN" or plan["bundle_execution_order"] != "CONDITIONS_SERIAL_ON_ASSIGNED_DEVICE":
        raise PolicyViolation("parallel plan permits unsafe condition sharding")
    core = {key: plan[key] for key in plan if key != "plan_digest"}
    if plan["plan_digest"] != canonical_digest(core):
        raise PolicyViolation("parallel plan digest mismatch")


def canonical_merge(plan: Mapping[str, Any], shard_root: str | Path) -> dict[str, Any]:
    """Merge complete shard commitments; never emit a partial aggregate."""

    validate_plan(plan)
    root = Path(shard_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise PolicyViolation("shard root must be a regular directory")
    expected_ids = {bundle["bundle_id"] for bundle in plan["bundles"]}
    observed_dirs = {path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()}
    unexpected_entries = [path.name for path in root.iterdir() if not path.is_dir() or path.is_symlink()]
    if unexpected_entries or observed_dirs != expected_ids:
        raise PolicyViolation(
            f"shard inventory incomplete or unexpected: missing={sorted(expected_ids-observed_dirs)}, "
            f"extra={sorted(observed_dirs-expected_ids)}, invalid={sorted(unexpected_entries)}"
        )
    records: list[dict[str, Any]] = []
    for bundle in plan["bundles"]:
        bundle_dir = root / bundle["bundle_id"]
        manifest_path = bundle_dir / "result_manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise PolicyViolation("bundle result manifest is missing or unsafe")
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_record_keys = {
            "result_manifest_version",
            "bundle_id",
            "plan_digest",
            "protocol_digest",
            "corpus_instance_id",
            "corpus_seed",
            "model_seed",
            "condition_order",
            "condition_status",
            "shared_digest_receipts",
            "payload_path",
            "payload_sha256",
            "final_status",
        }
        if set(record) != expected_record_keys:
            raise PolicyViolation("bundle result manifest has unknown or missing fields")
        if record["result_manifest_version"] != "child-only-result-manifest-v1":
            raise PolicyViolation("wrong bundle result-manifest version")
        for key in ("bundle_id", "corpus_instance_id", "corpus_seed", "model_seed"):
            if record[key] != bundle[key]:
                raise PolicyViolation("bundle result identity differs from plan")
        if record["plan_digest"] != plan["plan_digest"] or record["protocol_digest"] != plan["protocol_digest"]:
            raise PolicyViolation("bundle result uses a foreign plan or protocol")
        if record["condition_order"] != list(CONDITION_ORDER) or set(record["condition_status"]) != set(CONDITION_ORDER):
            raise PolicyViolation("bundle result is condition-incomplete")
        if any(record["condition_status"][condition] != "COMPLETE" for condition in CONDITION_ORDER):
            raise PolicyViolation("bundle contains a failed or nonfinal condition")
        if record["final_status"] != "COMPLETE":
            raise PolicyViolation("bundle is not final")
        if set(record["shared_digest_receipts"]) != {
            "data",
            "tokenizer",
            "architecture",
            "initialization",
            "optimizer",
            "example_order",
            "updates",
            "compute",
        }:
            raise PolicyViolation("bundle lacks matched-condition digest receipts")
        payload_rel = Path(record["payload_path"])
        if payload_rel.is_absolute() or ".." in payload_rel.parts or len(payload_rel.parts) != 1:
            raise PolicyViolation("bundle payload path must be a safe direct child")
        payload_path = bundle_dir / payload_rel
        if not payload_path.is_file() or payload_path.is_symlink():
            raise PolicyViolation("bundle payload is missing or unsafe")
        payload_digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        if payload_digest != record["payload_sha256"]:
            raise PolicyViolation("bundle payload checksum mismatch")
        observed_files = {path.name for path in bundle_dir.iterdir() if path.is_file() and not path.is_symlink()}
        if observed_files != {"result_manifest.json", payload_rel.name}:
            raise PolicyViolation("bundle directory contains missing or unexpected files")
        records.append(record)
    records.sort(key=lambda item: (item["corpus_instance_id"], item["corpus_seed"], item["model_seed"]))
    core = {
        "merge_version": "child-only-canonical-merge-v1",
        "plan_digest": plan["plan_digest"],
        "protocol_digest": plan["protocol_digest"],
        "complete": True,
        "bundle_count": len(records),
        "bundle_manifests": records,
    }
    return {**core, "merge_digest": canonical_digest(core)}


def write_plan(plan: Mapping[str, Any], destination: str | Path) -> None:
    """Write a validated plan only; launching is outside this construction task."""

    validate_plan(plan)
    path = Path(destination)
    if path.exists():
        raise FileExistsError("parallel plan writer refuses to overwrite existing evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(plan) + b"\n")
