import numpy as np

from babyworld_lite.childlens_support_repair import (
    nuisance_matrix,
    participant_contrasts,
    stable_weights,
)


def test_stable_weights_balance_and_remain_effective():
    values = np.asarray([[-1.0], [-0.5], [0.5], [1.0]])
    weights, diagnostics = stable_weights(
        values, np.asarray([0.0]), smd_limit=0.01, maximum_weight=0.5
    )
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert diagnostics["solver_success"]
    assert diagnostics["maximum_absolute_smd"] <= 0.01
    assert diagnostics["effective_sample_size"] >= 3.9


def test_nuisance_coarsening_is_support_only():
    arms = {
        lag: [
            {
                "index": index,
                "released_speech_support_fraction": index / 10,
                "activity": "common" if index < 5 else f"rare-{lag}",
                "location": "room",
            }
            for index in range(6)
        ]
        for lag in (-1, 0, 1)
    }
    values, kept = nuisance_matrix(
        arms,
        log_rms=np.arange(6.0),
        motion=np.arange(6.0),
        persistence=np.arange(6.0),
    )
    assert kept["activity"] == ["common"]
    assert kept["location"] == ["room"]
    assert {matrix.shape[0] for matrix in values.values()} == {6}


def test_participant_contrast_uses_weighted_arm_means():
    rows = []
    scores = {}
    weights = {}
    for duration in (2, 6, 18):
        lags = (-4 * duration, -2 * duration, -duration, 0, duration, 2 * duration, 4 * duration)
        for index in range(2):
            key = f"{duration}-{index}"
            rows.append(
                {
                    "participant_key": "p",
                    "duration_seconds": duration,
                    "row_key": key,
                }
            )
            scores[key] = {lag: (1.0 if lag == 0 else 0.0) for lag in lags}
        for lag in lags:
            weights[("p", duration, lag)] = np.asarray([0.5, 0.5])
    primary, _ = participant_contrasts(rows, scores, weights)
    assert primary == {"p": 1.0}
