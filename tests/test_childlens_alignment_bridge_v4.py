from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from babyworld_lite.childlens_alignment_bridge_v4.preflight import (
    BridgeV4Error,
    candidate_windows,
    cross_participant_assignment,
    deterministic_split,
    effect_summary,
    projected_cosine,
    safe_fraction,
    shifted_window,
    train_projection_heads,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_childlens_alignment_bridge_v4.py"


def _runner():
    spec = importlib.util.spec_from_file_location(
        "childlens_alignment_bridge_v4_runner_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_is_cohort_balanced_participant_disjoint_and_order_invariant() -> None:
    rows = [
        {"participant_key": f"p{index}", "cohort": "old" if index < 8 else "new"}
        for index in range(18)
    ]
    first = deterministic_split(rows, seed="frozen", evaluation_per_cohort=3)
    second = deterministic_split(
        list(reversed(rows)), seed="frozen", evaluation_per_cohort=3
    )
    assert first == second
    assert sum(value == "evaluation" for value in first.values()) == 6
    assert sum(
        value == "evaluation"
        for row, value in ((row, first[row["participant_key"]]) for row in rows[:8])
    ) == 3
    assert set(first) == {row["participant_key"] for row in rows}


def test_windows_are_fixed_nonoverlapping_capped_and_shift_is_disjoint() -> None:
    windows = candidate_windows(
        [
            {"start_seconds": 1.0, "end_seconds": 2.0},
            {"start_seconds": 2.1, "end_seconds": 8.1},
            {"start_seconds": 20.0, "end_seconds": 30.0},
        ],
        recording_duration_seconds=40.0,
        window_seconds=4.0,
        exclusion_buffer_seconds=0.5,
        maximum_windows=4,
        seed="frozen",
        participant_key="p",
        item_key="i",
    )
    assert len(windows) <= 4
    assert all(row["end_ms"] - row["start_ms"] == 4000 for row in windows)
    assert all(
        right["start_ms"] >= left["end_ms"] + 500
        for left, right in zip(windows, windows[1:])
    )
    control = shifted_window(
        start_ms=10_000,
        end_ms=14_000,
        recording_duration_ms=30_000,
        offset_ms=8_000,
        minimum_gap_ms=1_000,
        seed="frozen",
        row_key="row",
    )
    assert control in {(2_000, 6_000), (18_000, 22_000)}


def test_shuffle_is_one_to_one_and_never_same_participant() -> None:
    rows = [
        {
            "participant_key": f"p{index // 2}",
            "activity_label": "play" if index % 2 else "meal",
            "location_label": "home",
            "speech_density": index / 10,
            "recording_position": index / 12,
            "row_hash": f"{index:064x}",
        }
        for index in range(12)
    ]
    assignment = cross_participant_assignment(rows)
    assert sorted(assignment) == list(range(len(rows)))
    assert all(
        rows[receiver]["participant_key"] != rows[donor]["participant_key"]
        for receiver, donor in enumerate(assignment)
    )


def test_projection_training_is_deterministic_and_scores_are_bounded() -> None:
    rng = np.random.default_rng(9)
    latent = rng.normal(size=(24, 6)).astype(np.float32)
    audio = np.concatenate([latent, latent[:, :2]], axis=1)
    visual = np.concatenate([latent, -latent[:, :2]], axis=1)
    participants = [f"p{index % 6}" for index in range(24)]
    first = train_projection_heads(
        audio,
        visual,
        participants,
        output_dimension=4,
        epochs=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        temperature=0.07,
        gradient_clip_norm=1.0,
        seed=12,
    )
    second = train_projection_heads(
        audio,
        visual,
        participants,
        output_dimension=4,
        epochs=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        temperature=0.07,
        gradient_clip_norm=1.0,
        seed=12,
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    scores = projected_cosine(audio, visual, *first)
    assert np.all((-1.0 <= scores) & (scores <= 1.0))


def test_effect_inference_uses_participant_clusters() -> None:
    rows = [
        {
            "participant_key": f"p{participant}",
            "real_cosine": 0.5 + participant / 100,
            "shift_cosine": 0.3,
        }
        for participant in range(6)
        for _ in range(participant + 1)
    ]
    result = effect_summary(
        rows,
        control_field="shift_cosine",
        confidence=0.9,
        bootstrap_replicates=1000,
        permutation_replicates=1000,
        seed=1,
    )
    assert result["participant_count"] == 6
    assert result["mean_lift"] > 0
    assert result["one_sided_sign_flip_p"] == pytest.approx(1 / 64)


def test_complementary_suppression_and_public_guard_fail_closed() -> None:
    assert safe_fraction(18, 18, minimum_cell_size=5) == (1.0, False)
    assert safe_fraction(16, 18, minimum_cell_size=5) == (None, True)
    runner = _runner()
    with pytest.raises(BridgeV4Error, match="E_PUBLIC_PRIVACY"):
        runner._public_guard({"participant_key": "private"})
    with pytest.raises(BridgeV4Error, match="E_PUBLIC_PRIVACY"):
        runner._public_guard({"note": "/Users/private/source"})


def test_protocol_forbids_every_out_of_scope_route_and_locked_status_is_sealed() -> None:
    config = json.loads(
        (ROOT / "configs/childlens_alignment_bridge_v4.json").read_text()
    )
    scope = config["scope"]
    assert scope["locked_rows_may_be_loaded_scored_summarized_or_inspected"] is False
    assert scope["locked_evaluation_allowed"] is False
    assert scope["new_recordings_or_downloads_allowed"] is False
    assert scope["aea_or_external_volume_allowed"] is False
    assert scope["babyview_allowed"] is False
    assert (
        scope[
            "asr_translation_language_id_transcripts_or_generative_labels_allowed"
        ]
        is False
    )
    assert config["method_selection"]["route_count"] == 1
    assert (
        config["method_selection"]["automatic_alternative_or_model_shopping_allowed"]
        is False
    )
    status = _runner().locked_status()
    assert status["locked_evaluation_authorized"] is False
    assert status["locked_rows_loaded_scored_summarized_or_inspected"] == 0
