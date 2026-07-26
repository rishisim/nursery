import pytest

from babyworld_lite.childlens_engine_bakeoff.weld_grasp_probe import (
    run_weld_grasp_probe,
)


def test_contact_triggered_weld_lifts_transports_releases_and_settles():
    result = run_weld_grasp_probe()

    assert result.mujoco_version == "3.5.0"
    assert result.weld_initially_inactive
    assert result.triggering_contact_count >= 1
    assert result.contact_precedes_or_equals_activation
    assert result.pose_jump_at_activation_m <= 1e-9
    assert result.lift_m >= 0.10
    assert result.transport_m >= 0.15
    assert result.release_time_s > result.activation_time_s
    assert result.landed_stably_on_support
    assert result.final_height_m == pytest.approx(0.03, abs=0.003)
    assert result.final_speed_m_s <= 0.01
