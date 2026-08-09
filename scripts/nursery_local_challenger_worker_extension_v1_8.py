#!/usr/bin/env python3
"""V1.8 input-validation adapter for the immutable local challenger worker.

Only candidate-window validation differs: candidate windows may overlap but
must retain exact keys, finite nonnegative endpoints, positive duration, and
monotonic start order. Speech extraction intervals remain non-overlapping.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Mapping


BASE = Path(__file__).with_name("nursery_local_challenger_worker.py")
SPEC = importlib.util.spec_from_file_location(
    "nursery_local_challenger_worker_extension_base_v18", BASE
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("E_MODULE")
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)
_original_validate_items = worker._validate_items


def _validate_candidate_windows(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        raise RuntimeError("E_INTERVALS")
    result: list[tuple[float, float]] = []
    previous_start = -1.0
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"start_seconds", "end_seconds"}:
            raise RuntimeError("E_INTERVALS")
        start, end = row["start_seconds"], row["end_seconds"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < previous_start
            or float(start) < 0
            or float(end) <= float(start)
        ):
            raise RuntimeError("E_INTERVALS")
        result.append((float(start), float(end)))
        previous_start = float(start)
    return result


def _validate_items(job: Any, mode: str) -> list[dict[str, Any]]:
    if mode == "asr":
        return _original_validate_items(job, mode)
    if (
        not isinstance(job, Mapping)
        or job.get("schema_version") != "nursery-childlens-vlm-job-v1"
        or job.get("sample_item_count") != 15
        or job.get("sample_total_seconds") != 900
        or not isinstance(job.get("items"), list)
        or len(job["items"]) != 15
    ):
        raise RuntimeError("E_JOB")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    seconds = 0.0
    for row in job["items"]:
        if not isinstance(row, Mapping):
            raise RuntimeError("E_JOB")
        key = row.get("opaque_key")
        media_fd = row.get("media_fd")
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(ch not in "0123456789abcdef" for ch in key)
            or key in seen
            or not isinstance(media_fd, int)
            or isinstance(media_fd, bool)
            or media_fd < 3
        ):
            raise RuntimeError("E_JOB")
        intervals = worker._validate_intervals(row.get("intervals"))
        seconds += sum(end - start for start, end in intervals)
        transcript = row.get("transcript_hypotheses")
        if not isinstance(transcript, list):
            raise RuntimeError("E_JOB")
        result.append(
            {
                "opaque_key": key,
                "media_fd": media_fd,
                "intervals": intervals,
                "candidate_windows": _validate_candidate_windows(
                    row.get("candidate_windows")
                ),
                "transcript_hypotheses": transcript,
            }
        )
        seen.add(key)
    if abs(seconds - 900.0) > 0.01:
        raise RuntimeError("E_JOB")
    return result


worker._validate_items = _validate_items


if __name__ == "__main__":
    raise SystemExit(worker.main())
