"""Outcome-blind stable balancing mechanics for the ChildLens V5 repair."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize


class SupportRepairError(RuntimeError):
    """Fail-closed support repair error."""


def nuisance_matrix(
    arm_requests: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    log_rms: np.ndarray,
    motion: np.ndarray,
    persistence: np.ndarray,
    category_minimum: int = 5,
) -> tuple[dict[int, np.ndarray], dict[str, list[str]]]:
    """Build outcome-free, deterministically coarsened nuisance matrices."""

    if not arm_requests or category_minimum <= 0:
        raise SupportRepairError("E_NUISANCE_INPUT")
    lengths = {len(rows) for rows in arm_requests.values()}
    if len(lengths) != 1 or min(lengths) == 0:
        raise SupportRepairError("E_ARM_INVENTORY")
    kept: dict[str, list[str]] = {}
    for field in ("activity", "location"):
        levels = sorted(
            {str(row[field]) for rows in arm_requests.values() for row in rows}
        )
        kept[field] = [
            level
            for level in levels
            if min(
                sum(str(row[field]) == level for row in rows)
                for rows in arm_requests.values()
            )
            >= category_minimum
        ]

    def encode(row: Mapping[str, Any]) -> list[float]:
        index = int(row["index"])
        values = [
            float(row["released_speech_support_fraction"]),
            float(log_rms[index]),
            float(motion[index]),
            float(persistence[index]),
        ]
        for field in ("activity", "location"):
            level = (
                str(row[field])
                if str(row[field]) in kept[field]
                else "__OTHER__"
            )
            levels = sorted(set(kept[field]) | {"__OTHER__"})
            values.extend(float(level == candidate) for candidate in levels[:-1])
        return values

    raw = {
        lag: np.asarray([encode(row) for row in rows], dtype=np.float64)
        for lag, rows in arm_requests.items()
    }
    pooled = np.concatenate(list(raw.values()), axis=0)
    means = pooled.mean(axis=0)
    scales = pooled.std(axis=0)
    active = scales > 1e-10
    standardized = {
        lag: ((values - means) / np.where(scales > 0, scales, 1))[:, active]
        for lag, values in raw.items()
    }
    return standardized, kept


def stable_weights(
    values: np.ndarray,
    target: np.ndarray,
    *,
    smd_limit: float,
    maximum_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find minimum-variance nonnegative weights with approximate balance."""

    matrix = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        matrix.ndim != 2
        or target.shape != (matrix.shape[1],)
        or matrix.shape[0] == 0
        or smd_limit <= 0
        or maximum_weight <= 0
    ):
        raise SupportRepairError("E_WEIGHT_INPUT")
    count = matrix.shape[0]
    solution = minimize(
        lambda weights: 0.5 * np.square(count * weights - 1).sum(),
        np.full(count, 1 / count),
        method="SLSQP",
        bounds=[(0.0, maximum_weight)] * count,
        constraints=[
            {"type": "eq", "fun": lambda weights: weights.sum() - 1},
            {
                "type": "ineq",
                "fun": lambda weights: smd_limit
                - np.abs(weights @ matrix - target),
            },
        ],
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    weights = np.asarray(solution.x, dtype=np.float64)
    imbalance = (
        float(np.max(np.abs(weights @ matrix - target)))
        if matrix.shape[1]
        else 0.0
    )
    diagnostics = {
        "solver_success": bool(solution.success),
        "maximum_absolute_smd": imbalance,
        "effective_sample_size": float(1 / np.square(weights).sum()),
        "maximum_weight": float(weights.max()),
        "top_10_weight_share": float(
            np.sort(weights)[-min(10, count) :].sum()
        ),
    }
    return weights, diagnostics


def participant_contrasts(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[int, float]],
    arm_weights: Mapping[tuple[str, int, int], np.ndarray],
) -> tuple[dict[str, float], dict[str, dict[int, dict[int, float]]]]:
    """Aggregate weighted arm means and V5 primary contrasts by participant."""

    participants = sorted({str(row["participant_key"]) for row in rows})
    lags = {
        2: (-8, -4, -2, 0, 2, 4, 8),
        6: (-24, -12, -6, 0, 6, 12, 24),
        18: (-72, -36, -18, 0, 18, 36, 72),
    }
    curves: dict[str, dict[int, dict[int, float]]] = {}
    primary: dict[str, float] = {}
    for participant in participants:
        curves[participant] = {}
        scale_values: list[float] = []
        for duration, duration_lags in lags.items():
            selected = [
                row
                for row in rows
                if str(row["participant_key"]) == participant
                and int(row["duration_seconds"]) == duration
            ]
            curve: dict[int, float] = {}
            for lag in duration_lags:
                weights = arm_weights[(participant, duration, lag)]
                values = np.asarray(
                    [scores[str(row["row_key"])][lag] for row in selected],
                    dtype=np.float64,
                )
                curve[lag] = float(weights @ values)
            curves[participant][duration] = curve
            scale_values.append(
                curve[0]
                - float(
                    np.mean(
                        [
                            curve[-4 * duration],
                            curve[-2 * duration],
                            curve[2 * duration],
                            curve[4 * duration],
                        ]
                    )
                )
            )
        primary[participant] = float(np.mean(scale_values))
    if any(not math.isfinite(value) for value in primary.values()):
        raise SupportRepairError("E_NONFINITE_CONTRAST")
    return primary, curves
