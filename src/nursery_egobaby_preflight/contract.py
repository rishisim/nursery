"""Small deterministic protocol and lineage helpers."""

from __future__ import annotations

import hashlib
import json
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
