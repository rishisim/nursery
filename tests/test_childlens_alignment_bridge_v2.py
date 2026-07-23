from __future__ import annotations

import pytest

from babyworld_lite.childlens_alignment_bridge_v2.diagnostics import (
    interval_overlap_seconds,
    interval_set_precision_recall,
    matched_boundary_f1,
    normalize_text,
)


def test_normalize_text_preserves_german_letters() -> None:
    assert normalize_text("  GRÜNE—Bälle! ") == "grüne bälle"


def test_interval_overlap_and_union_precision_recall() -> None:
    left = {"start_seconds": 0.0, "end_seconds": 2.0}
    right = {"start_seconds": 1.0, "end_seconds": 3.0}
    assert interval_overlap_seconds(left, right) == 1.0
    precision, recall = interval_set_precision_recall(
        [left, {"start_seconds": 1.5, "end_seconds": 2.5}],
        [right],
    )
    assert precision == pytest.approx(1.5 / 2.5)
    assert recall == pytest.approx(1.5 / 2.0)


def test_matched_boundary_f1_requires_both_boundaries() -> None:
    base = [
        {"start_seconds": 0.0, "end_seconds": 1.0},
        {"start_seconds": 2.0, "end_seconds": 3.0},
    ]
    stable = [
        {"start_seconds": 0.1, "end_seconds": 1.1},
        {"start_seconds": 1.9, "end_seconds": 3.1},
    ]
    shifted_end = [{"start_seconds": 0.1, "end_seconds": 2.0}]
    assert matched_boundary_f1(base, stable, tolerance_seconds=0.5) == 1.0
    assert matched_boundary_f1(base, shifted_end, tolerance_seconds=0.5) == 0.0
