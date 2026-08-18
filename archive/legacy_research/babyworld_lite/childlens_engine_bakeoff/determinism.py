"""Compare two EpisodeTrace archives under a declared numeric tolerance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(first_path: Path, second_path: Path, *, atol: float = 1e-9) -> dict:
    first = np.load(first_path, allow_pickle=False)
    second = np.load(second_path, allow_pickle=False)
    if set(first.files) != set(second.files):
        raise ValueError("EpisodeTrace stream sets differ")
    rows = []
    for name in sorted(first.files):
        left, right = first[name], second[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"EpisodeTrace stream schema differs: {name}")
        if np.issubdtype(left.dtype, np.number):
            nan_pattern_equal = bool(
                np.array_equal(np.isnan(left), np.isnan(right))
            )
            finite = np.isfinite(left) & np.isfinite(right)
            maximum = (
                float(np.max(np.abs(left[finite] - right[finite])))
                if np.any(finite)
                else 0.0
            )
            infinity_equal = bool(
                np.array_equal(np.isposinf(left), np.isposinf(right))
                and np.array_equal(np.isneginf(left), np.isneginf(right))
            )
            passed = bool(
                nan_pattern_equal and infinity_equal and maximum <= atol
            )
        else:
            maximum = None
            passed = bool(np.array_equal(left, right))
        rows.append(
            {"stream": name, "maximum_absolute_error": maximum, "passes": passed}
        )
    return {
        "schema": "DeterministicReplayReceipt.v1",
        "absolute_tolerance": atol,
        "first_trace_sha256": _sha256(first_path),
        "second_trace_sha256": _sha256(second_path),
        "rows": rows,
        "maximum_numeric_absolute_error": max(
            row["maximum_absolute_error"] or 0.0 for row in rows
        ),
        "all_pass": all(row["passes"] for row in rows),
    }


def write_receipt(
    first_path: Path, second_path: Path, output_path: Path, *, atol: float = 1e-9
) -> dict:
    receipt = compare(first_path, second_path, atol=atol)
    output_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
