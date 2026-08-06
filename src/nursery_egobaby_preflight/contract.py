"""Small deterministic protocol and lineage helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_COMPACT_ENUM = re.compile(r"[A-Z][A-Z0-9_]{0,95}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def compact_aggregate_json(
    record: Mapping[str, Any],
    *,
    allowed_fields: set[str] | frozenset[str],
    sha256_fields: set[str] | frozenset[str] = frozenset(),
) -> str:
    """Serialize one terminal-safe, flat aggregate record.

    Governed commands must explicitly whitelist every output field. Values may
    only be finite numbers, booleans, short uppercase enum strings, or a SHA-256
    commitment in an explicitly designated field. Containers and arbitrary
    strings are rejected so identifiers, row lists, text, and paths cannot be
    printed accidentally.
    """
    unknown = set(record) - set(allowed_fields)
    missing = set(allowed_fields) - set(record)
    if unknown or missing:
        raise ValueError("compact aggregate fields do not match the whitelist")
    if not set(sha256_fields) <= set(allowed_fields):
        raise ValueError("SHA-256 fields must be output-whitelisted")
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            clean[key] = value
        elif key in sha256_fields and isinstance(value, str) and _SHA256.fullmatch(value):
            clean[key] = value
        elif isinstance(value, str) and _COMPACT_ENUM.fullmatch(value):
            clean[key] = value
        else:
            raise ValueError(f"unsafe compact aggregate value for {key!r}")
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def schedule_cycle(schedule: Mapping[str, int]) -> list[str]:
    cycle: list[str] = []
    for objective, count in schedule.items():
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"schedule count for {objective!r} must be a positive integer")
        cycle.extend([objective] * count)
    if not cycle:
        raise ValueError("schedule must contain at least one objective")
    return cycle


def lexical_macro_wiring(task_results: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Exercise noun/adjective aggregation without presenting a scientific score."""
    required = ("noun", "adjective")
    if tuple(task_results) != required:
        raise ValueError(f"lexical tasks must be ordered exactly as {required!r}")
    means: dict[str, float] = {}
    for task in required:
        values = tuple(float(value) for value in task_results[task])
        if not values:
            raise ValueError(f"{task} fixture is empty")
        if any(value not in (0.0, 1.0) for value in values):
            raise ValueError(f"{task} fixture must contain binary correctness wiring values")
        means[task] = sum(values) / len(values)
    return {**means, "lexical_macro": (means["noun"] + means["adjective"]) / 2}


def validate_phase_state(config: Mapping[str, Any]) -> None:
    """Reject scientifically contradictory top-level and nested Phase 4 states."""
    top = str(config.get("status", ""))
    nested = str(config.get("gates", {}).get("phase4_status", ""))
    if not top or not nested:
        raise ValueError("phase status and gates.phase4_status are required")
    top_provisional = "REOPENED" in top or "PROVISIONAL" in top
    nested_provisional = "PROVISIONAL" in nested or "SUPERSEDED" in nested
    top_pass = top.startswith("PASS") or "CORRECTED_ASSETS_PASS" in top
    nested_pass = nested.startswith("PASS") or "CORRECTED_COMMON_ASSETS_PASS" in nested
    top_no_go = "NO_GO" in top
    nested_no_go = "NO_GO" in nested
    active_markers = (
        "VISOR_HOS_CORRECTION_AMENDMENT_FROZEN_PENDING_COMPLETE_PUBLIC_COMBINED_GATE",
        "MECHANISTIC_TRAINING_TUPLE_FIXTURE_SOURCE_NO_GO",
        "LEARNER_EFFECTIVE_COMPLETE_PUBLIC_SOURCE_FEASIBILITY_NO_GO",
        "AMBITIOUS_LEARNER_EFFECTIVE_H3_AMENDMENT_FROZEN",
        "CONSTRUCT_ALIGNED_LEARNER_EFFECTIVE_LTX_RESUME_AMENDMENT_FROZEN",
        "CONSTRUCT_ALIGNED_LEARNER_EFFECTIVE_LTX_SOLE_GENERATOR_COMPILER_AMENDMENT_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_AMENDMENT_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_H100_RESOURCE_REDIRECT_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_DEPENDENCY_RESTORE_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_TOPOLOGY_GUARD_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_BLOCKER_SEALED",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_BLOCKER_PRESERVED_SINGLE_ATTEMPT_REAUTHORIZATION_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_REAUTHORIZED_ATTEMPT_4_BLOCKER_SEALED",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_4_BLOCKER_PRESERVED_PARSER_REPAIR_REAUTHORIZATION_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_POST_BLOCKER_ATTEMPT_5_ENGINEERING_BLOCKER_SEALED",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_5_BLOCKER_PRESERVED_HOST_CONTAINER_ATTESTATION_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_6_TIMEOUT_PRESERVED_PROGRESS_INSTRUMENTATION_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_7_TIMEOUT_PRESERVED_EXTENDED_WALL_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_7_TIMEOUT_PRESERVED_ZERO_RUNTIME_H100_MIG_ATTEMPT_8_CANCELED_FULL_H100_TOPOLOGY_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_8_GIT_ABSENCE_BLOCKER_PRESERVED_CLEAN_TREE_FALLBACK_ATTEMPT_9_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_9_HISTORICAL_LINEAGE_BLOCKER_PRESERVED_LINEAGE_REPAIR_ATTEMPT_10_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_10_CROSS_PYTHON_AST_BLOCKER_PRESERVED_PORTABLE_AST_REPAIR_ATTEMPT_11_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_11_FIXTURE_ROOT_BLOCKER_PRESERVED_READ_ONLY_FIXTURE_BIND_REPAIR_ATTEMPT_12_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_12_SUBMISSION_EXPORT_BLOCKER_PRESERVED_EXPORT_CONTRACT_REPAIR_ATTEMPT_13_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_13_RUNTIME_BLOCKERS_PRESERVED_NLTK_MATPLOTLIB_REPAIR_ATTEMPT_14_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_14_SLURM_GRES_SERIALIZATION_BLOCKER_PRESERVED_ATTEMPT_15_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_15_ACTIVE_DISPATCH_BLOCKER_PRESERVED_ATTEMPT_16_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_16_RESOURCE_DISPATCH_BLOCKER_PRESERVED_ATTEMPT_17_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_17_RUNTIME_EXECUTION_BLOCKER_PRESERVED_ATTEMPT_18_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_18_HISTORICAL_FULL_RESULT_LINEAGE_BLOCKER_PRESERVED_ATTEMPT_19_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_ATTEMPT_19_GROUNDING_STATE_BLOCKER_PRESERVED_ATTEMPT_20_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_ENGINEERING_HEALTH_PASS_SEALED_PUBLIC_DEVELOPMENT_AUTHORIZED",
        "LEARNER_EFFECTIVE_PUBLIC_DEVELOPMENT_ENGINEERING_ATTEMPT_1_BLOCKER_PRESERVED_MASK_ROUNDTRIP_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_PUBLIC_DEVELOPMENT_ENGINEERING_ATTEMPT_2_BLOCKER_PRESERVED_ATTRIBUTE_DEPENDENCY_REPAIR_FROZEN",
        "LEARNER_EFFECTIVE_PUBLIC_DEVELOPMENT_COMPLETE_SCIENTIFIC_NO_GO_DOWNSTREAM_STOPPED",
        "PHASE4_PRIOR_RESULTS_PRESERVED_PUBLIC_ONLY_CALIBRATION_READINESS_AMENDMENT_FROZEN",
        "PHASE4_PRIOR_RESULTS_PRESERVED_PUBLIC_ONLY_CALIBRATION_READINESS_TOPOLOGY_FROZEN",
        "PHASE4_PRIOR_RESULTS_PRESERVED_PUBLIC_ONLY_CALIBRATION_READINESS_RENTED_RESOURCE_FROZEN",
        "PHASE4_PRIOR_RESULTS_PRESERVED_PUBLIC_ONLY_CALIBRATION_READINESS_ENGINEERING_BLOCKER",
    )
    marker_mismatch = any((marker in top) != (marker in nested) for marker in active_markers)
    if (
        top_provisional != nested_provisional
        or top_pass != nested_pass
        or top_no_go != nested_no_go
        or marker_mismatch
    ):
        raise ValueError(f"contradictory Phase 4 states: {top!r} vs {nested!r}")
