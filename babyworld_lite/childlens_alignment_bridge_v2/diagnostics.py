"""Pure timing and transcript diagnostic helpers for remediation v2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import unicodedata
from typing import Any


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\wäöüß]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def interval_overlap_seconds(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    return max(
        0.0,
        min(float(left["end_seconds"]), float(right["end_seconds"]))
        - max(float(left["start_seconds"]), float(right["start_seconds"])),
    )


def interval_set_precision_recall(
    predicted: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    """Duration-weighted set precision/recall after exact interval union."""

    def union(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
        ordered = sorted(
            (float(row["start_seconds"]), float(row["end_seconds"]))
            for row in rows
            if float(row["end_seconds"]) > float(row["start_seconds"])
        )
        merged: list[list[float]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    predicted_union = union(predicted)
    reference_union = union(reference)
    predicted_seconds = sum(end - start for start, end in predicted_union)
    reference_seconds = sum(end - start for start, end in reference_union)
    overlap = sum(
        max(0.0, min(p_end, r_end) - max(p_start, r_start))
        for p_start, p_end in predicted_union
        for r_start, r_end in reference_union
    )
    precision = overlap / predicted_seconds if predicted_seconds else 0.0
    recall = overlap / reference_seconds if reference_seconds else 0.0
    return precision, recall


def matched_boundary_f1(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    tolerance_seconds: float,
) -> float:
    """One-to-one start-and-end boundary agreement under a fixed tolerance."""

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    unmatched = set(range(len(right)))
    matches = 0
    for left_row in sorted(left, key=lambda row: float(row["start_seconds"])):
        candidates = []
        for index in unmatched:
            right_row = right[index]
            start_error = abs(
                float(left_row["start_seconds"]) - float(right_row["start_seconds"])
            )
            end_error = abs(
                float(left_row["end_seconds"]) - float(right_row["end_seconds"])
            )
            if start_error <= tolerance_seconds and end_error <= tolerance_seconds:
                candidates.append((start_error + end_error, index))
        if candidates:
            _, index = min(candidates)
            unmatched.remove(index)
            matches += 1
    precision = matches / len(left)
    recall = matches / len(right)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
