from __future__ import annotations

import pytest

from babyworld_lite.childlens_media_repair import EpisodeSpec, derive_imu


def test_clock_is_exact() -> None:
    spec = EpisodeSpec()
    assert spec.ticks_per_frame == 400
    assert 47 * spec.ticks_per_frame == 18_800


def test_imu_is_derived_from_physics_velocity() -> None:
    samples = [
        {"tick": 0, "velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0]},
        {"tick": 400, "velocity": [1.0, -1.0, 0.0], "angular_velocity": [0.0, 0.2, 0.0]},
        {"tick": 800, "velocity": [2.0, -2.0, 0.0], "angular_velocity": [0.0, 0.4, 0.0]},
    ]
    imu = derive_imu(samples, 0.5)
    assert imu[1]["linear_acceleration_m_s2"] == pytest.approx([2.0, -2.0, 0.0])
    assert imu[1]["angular_velocity_rad_s"] == [0.0, 0.2, 0.0]
    assert all(row["method"] == "finite_difference_physics_velocity" for row in imu)


def test_imu_rejects_underspecified_input() -> None:
    with pytest.raises(ValueError):
        derive_imu([], 1 / 60)
