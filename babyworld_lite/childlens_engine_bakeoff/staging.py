"""Deterministic occupancy-first scene staging for the bounded kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StageCandidate:
    row: int
    column: int
    world_x_m: float
    world_y_m: float
    rank: int
    selection_rule: str


def select_free_candidate(
    occupancy: np.ndarray,
    map_to_world: np.ndarray,
    *,
    anchor_xy_m: tuple[float, float],
) -> StageCandidate:
    """Select the closest free pixel, breaking ties lexicographically."""
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a 2-D array")
    free_rc = np.argwhere(occupancy.astype(bool))
    if free_rc.size == 0:
        raise ValueError("occupancy contains no free candidates")
    rc1 = np.column_stack([free_rc[:, 0], free_rc[:, 1], np.ones(len(free_rc))])
    world = (np.asarray(map_to_world, dtype=np.float64) @ rc1.T).T
    squared_distance = (
        (world[:, 0] - anchor_xy_m[0]) ** 2
        + (world[:, 1] - anchor_xy_m[1]) ** 2
    )
    order = np.lexsort((free_rc[:, 1], free_rc[:, 0], squared_distance))
    selected_index = int(order[0])
    row, column = (int(value) for value in free_rc[selected_index])
    return StageCandidate(
        row=row,
        column=column,
        world_x_m=float(world[selected_index, 0]),
        world_y_m=float(world[selected_index, 1]),
        rank=0,
        selection_rule=(
            "minimum squared world distance to preregistered anchor; "
            "ties by ascending map row then column"
        ),
    )
