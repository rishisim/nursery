"""Episode bundle hashing, manifest construction, and QA invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shared_clock(
    time_s: np.ndarray, streams: dict[str, np.ndarray]
) -> dict[str, Any]:
    time_s = np.asarray(time_s, dtype=np.float64)
    if time_s.ndim != 1 or not len(time_s):
        raise ValueError("time_s must be a non-empty 1-D array")
    if time_s[0] != 0.0:
        raise ValueError("shared clock must include t=0")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("shared clock must be strictly increasing")
    mismatches = {
        name: int(len(values))
        for name, values in streams.items()
        if len(values) != len(time_s)
    }
    if mismatches:
        raise ValueError(f"stream length mismatch: {mismatches}")
    return {
        "passed": True,
        "samples": int(len(time_s)),
        "start_s": float(time_s[0]),
        "end_s": float(time_s[-1]),
        "stream_names": sorted(streams),
    }


def trace_equivalence(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    if set(first) != set(second):
        raise ValueError("trace keys differ")
    maximum_error = 0.0
    exact_keys: list[str] = []
    for key in sorted(first):
        left, right = np.asarray(first[key]), np.asarray(second[key])
        if left.shape != right.shape:
            raise ValueError(f"trace shape differs for {key}")
        if np.issubdtype(left.dtype, np.number):
            error = float(np.max(np.abs(left.astype(float) - right.astype(float))))
            maximum_error = max(maximum_error, error)
        else:
            if not np.array_equal(left, right):
                raise ValueError(f"nonnumeric trace values differ for {key}")
            exact_keys.append(key)
    return {
        "passed": maximum_error <= absolute_tolerance,
        "maximum_absolute_error": maximum_error,
        "absolute_tolerance": absolute_tolerance,
        "exact_nonnumeric_keys": exact_keys,
    }


def build_manifest(
    bundle_dir: Path,
    relative_files: Iterable[str],
    *,
    spec_sha256: str,
    provenance: dict[str, Any],
    regeneration_command: list[str],
) -> dict[str, Any]:
    files = []
    for relative in sorted(relative_files):
        path = bundle_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "EpisodeBundleManifest",
        "spec_sha256": spec_sha256,
        "files": files,
        "provenance": provenance,
        "regeneration_command": regeneration_command,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
