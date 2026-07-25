from __future__ import annotations

import math

import pytest

from babyworld_lite.childlens_media_repair.blender_adapter import (
    BlenderEpisodeSpec,
    derive_pose_signals,
    sphere_box_signed_distance,
)


def test_canonical_clock_is_exact() -> None:
    spec = BlenderEpisodeSpec(duration_seconds=1.0)
    assert spec.ticks_per_physics_step == 100
    assert spec.physics_steps == 240
    with pytest.raises(ValueError):
        BlenderEpisodeSpec(physics_hz=239)


def test_exact_contact_positive_and_negative_fixtures() -> None:
    positive = sphere_box_signed_distance((1.1, 0, 0), 0.2, (0, 0, 0), (1, 1, 1))
    negative = sphere_box_signed_distance((1.3, 0, 0), 0.2, (0, 0, 0), (1, 1, 1))
    assert positive["active"] is True
    assert positive["signed_separation_m"] == pytest.approx(-0.1)
    assert negative["active"] is False
    assert negative["signed_separation_m"] == pytest.approx(0.1)
    assert positive["method"] == "exact_oriented_sphere_box_sdf"


def test_joint_and_imu_dynamic_and_static_controls() -> None:
    dt = 0.01
    dynamic = [
        {
            "position_m": [0.5 * (i * dt) ** 2, 0.0, 0.0],
            "joint_angle_rad": 0.5 * i * dt,
        }
        for i in range(20)
    ]
    signals = derive_pose_signals(dynamic, dt)
    assert max(abs(v[0]) for v in signals["linear_acceleration_world_m_s2"][2:-2]) == pytest.approx(1.0)
    assert max(abs(v[2]) for v in signals["angular_velocity_world_rad_s"]) == pytest.approx(0.5)

    static = [{"position_m": [0, 0, 0], "joint_angle_rad": 0.0} for _ in range(5)]
    quiet = derive_pose_signals(static, dt)
    assert all(math.isclose(x, 0.0) for x in quiet["joint_velocity_rad_s"])
    assert all(math.isclose(x, 0.0) for row in quiet["linear_acceleration_world_m_s2"] for x in row)
