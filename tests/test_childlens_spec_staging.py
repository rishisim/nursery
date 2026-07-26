import numpy as np

from babyworld_lite.childlens_engine_bakeoff.staging import select_free_candidate


def test_staging_is_deterministic_and_uses_lexicographic_tie_break():
    occupancy = np.zeros((3, 3), dtype=bool)
    occupancy[0, 1] = True
    occupancy[1, 0] = True
    candidate = select_free_candidate(
        occupancy, np.eye(3), anchor_xy_m=(0.0, 0.0)
    )
    assert (candidate.row, candidate.column) == (0, 1)
    assert (candidate.world_x_m, candidate.world_y_m) == (0.0, 1.0)


def test_staging_rejects_empty_map():
    try:
        select_free_candidate(np.zeros((2, 2)), np.eye(3), anchor_xy_m=(0, 0))
    except ValueError as error:
        assert "no free candidates" in str(error)
    else:
        raise AssertionError("empty occupancy map was accepted")
