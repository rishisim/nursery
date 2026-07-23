from __future__ import annotations

import pytest
import numpy as np

from babyworld_lite.childlens_alignment_bridge_v1.alignment import (
    normalized_diagonal_cosine,
    shifted_window,
    sign_flip_pvalue,
)
from babyworld_lite.childlens_alignment_bridge_v1.preflight import (
    BridgeError,
    build_control_assignment,
    canonical_text,
    character_similarity,
    interval_boundary_f1,
    participant_bootstrap_interval,
)


def test_character_similarity_is_bounded_and_normalized() -> None:
    assert canonical_text("  Das—AUTO! ") == "das auto"
    assert character_similarity("Das Auto", "das auto") == 1.0
    assert 0.0 <= character_similarity("Ball", "Puppe") <= 1.0
    assert character_similarity("", "") == 1.0


def test_boundary_f1_is_one_to_one_and_tolerance_bounded() -> None:
    primary = [
        {"start_seconds": 0.0, "end_seconds": 1.0},
        {"start_seconds": 2.0, "end_seconds": 3.0},
    ]
    close = [
        {"start_seconds": 0.2, "end_seconds": 1.2},
        {"start_seconds": 2.1, "end_seconds": 3.1},
    ]
    far = [{"start_seconds": 10.0, "end_seconds": 11.0}]
    assert interval_boundary_f1(primary, close) == 1.0
    assert interval_boundary_f1(primary, far) == 0.0


def test_participant_bootstrap_is_deterministic() -> None:
    left = participant_bootstrap_interval(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        confidence=0.9,
        replicates=500,
        seed=7,
    )
    right = participant_bootstrap_interval(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        confidence=0.9,
        replicates=500,
        seed=7,
    )
    assert left == right
    assert left[0] <= 0.3 <= left[1]


def test_control_assignment_deranges_participants() -> None:
    rows = [
        {"utterance_key": f"u{index}", "participant_key": f"p{index}", "stratum": "a"}
        for index in range(5)
    ]
    assignments = build_control_assignment(rows, protocol_sha256="a" * 64)
    donors = {row["utterance_key"]: row["donor_utterance_key"] for row in assignments}
    assert set(donors) == {f"u{index}" for index in range(5)}
    assert all(receiver != donor for receiver, donor in donors.items())


def test_control_assignment_fails_without_cross_participant_support() -> None:
    rows = [
        {"utterance_key": "u1", "participant_key": "p1", "stratum": "a"},
        {"utterance_key": "u2", "participant_key": "p1", "stratum": "a"},
    ]
    with pytest.raises(BridgeError, match="E_CONTROL_SUPPORT"):
        build_control_assignment(rows, protocol_sha256="b" * 64)


def test_cosine_keeps_row_ids_and_normalizes() -> None:
    scores = normalized_diagonal_cosine(
        np.array([[2.0, 0.0], [0.0, 3.0]]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        ["a", "b"],
    )
    assert scores == [{"row_id": "a", "cosine": 1.0}, {"row_id": "b", "cosine": 0.0}]


def test_shift_is_nonwrapping_and_falls_back_to_other_direction() -> None:
    shifted = shifted_window(
        start_seconds=2.0,
        end_seconds=4.0,
        recording_start_seconds=0.0,
        recording_end_seconds=30.0,
        offset_seconds=15.0,
        protocol_sha256="c" * 64,
        utterance_key="u",
    )
    assert shifted == (17.0, 19.0)
    unavailable = shifted_window(
        start_seconds=10.0,
        end_seconds=20.0,
        recording_start_seconds=0.0,
        recording_end_seconds=25.0,
        offset_seconds=15.0,
        protocol_sha256="c" * 64,
        utterance_key="u",
    )
    assert unavailable is None


def test_exact_sign_flip_detects_consistent_positive_lift() -> None:
    assert sign_flip_pvalue([0.1] * 5, replicates=1000, seed=4) == 1 / 32
