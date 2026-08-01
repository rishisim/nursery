import json
from pathlib import Path

import pytest

from babyworld_lite.childlens_engine_bakeoff.compile_episode import (
    compose_embodied_prompt_plan,
)
from babyworld_lite.childlens_engine_bakeoff.run_continuous_episode import (
    _execution_spec,
)
from babyworld_lite.childlens_engine_bakeoff.speech_audio import _word_alignment


EPISODE = json.loads(
    Path("configs/embodied_simulation_episode.json").read_text()
)
VERTICAL = json.loads(
    Path("configs/embodied_simulation_vertical_slice.json").read_text()
)


def test_continuous_episode_contract_is_frozen_and_contiguous():
    assert EPISODE["frozen_before_phase_4_outcomes_at_utc"]
    assert EPISODE["activity"]["duration_s"] == 60.0
    schedule = EPISODE["activity"]["vertical_slice"]
    assert schedule[0]["start_s"] == 0.0
    assert schedule[-1]["end_s"] == 60.0
    assert all(
        left["end_s"] == right["start_s"]
        for left, right in zip(schedule, schedule[1:])
    )
    assert EPISODE["activity"]["hidden_world_resets_permitted"] is False
    assert 4 <= len(EPISODE["language"]["utterances"]) <= 6


def test_prompt_planner_emits_only_bounded_actions_and_language():
    plan = compose_embodied_prompt_plan(EPISODE, VERTICAL)
    allowed = set(EPISODE["planner"]["allowed_actions"])
    assert {row["action"] for row in plan["actions"]} <= allowed
    assert plan["scene_resolution"]["status"] == "resolved_before_execution"
    assert (
        plan["scene_resolution"]["resolved_persistent_id"]
        == "yellow_cup_authored"
    )
    assert plan["planner_emitted_joint_trajectories"] is False
    assert plan["planner_emitted_camera_trajectory"] is False
    assert plan["planner_emitted_object_trajectory"] is False
    forbidden = {"qpos", "qvel", "joint_targets", "camera_pose", "object_pose"}
    assert not forbidden.intersection(plan)


def test_scene_validator_rejects_an_impossible_target():
    episode = json.loads(json.dumps(EPISODE))
    episode["planner"]["target_reference"]["required_persistent_id"] = (
        "missing_object"
    )
    with pytest.raises(ValueError, match="does not resolve"):
        compose_embodied_prompt_plan(episode, VERTICAL)


def test_phase_4_reuses_hard_gates_and_forbids_neural_audio():
    assert (
        EPISODE["qualification"][
            "reuse_vertical_slice_hard_gates_without_change"
        ]
        is True
    )
    assert EPISODE["language"]["speech"]["neural_render_audio_permitted"] is False
    assert EPISODE["selection_policy"]["authoritative_rgb"] == (
        "deterministic_native_mimo_baseline"
    )
    assert EPISODE["bounded_repairs"][-1][
        "frozen_plan_timeline_seeds_and_thresholds_changed"
    ] is False
    assert VERTICAL["scene_family"]["support"]["collision_enabled"] is True
    assert VERTICAL["scene_family"]["support"]["catch_tray_radius_m"] == 0.18


def test_execution_spec_preserves_frozen_physics_gates_and_mount():
    plan = compose_embodied_prompt_plan(EPISODE, VERTICAL)
    resolved = _execution_spec(EPISODE, VERTICAL, plan)
    assert resolved["frozen_gates"] == VERTICAL["frozen_gates"]
    assert (
        resolved["embodiment"]["camera_mount"]
        == VERTICAL["embodiment"]["camera_mount"]
    )
    assert resolved["collision_policy"] == VERTICAL["collision_policy"]
    assert resolved["continuous_episode"]["vertical_slice_hard_gates_changed"] is False


def test_recorded_word_alignment_is_monotonic_and_bounded():
    rows = _word_alignment("Shake, tap, and move it.", 28.2, 29.7)
    assert [row["word"] for row in rows] == [
        "Shake",
        "tap",
        "and",
        "move",
        "it",
    ]
    assert rows[0]["start_s"] == 28.2
    assert rows[-1]["end_s"] == 29.7
    assert all(left["end_s"] == right["start_s"] for left, right in zip(rows, rows[1:]))
