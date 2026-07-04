from __future__ import annotations

import copy

from babyworld_lite.eval.features import (
    ARM_MODALITIES,
    extract_windowed_feature_groups,
    flatten_feature_groups,
    mutate_future_frames_for_test,
    prediction_frame,
    regression_targets,
)
from babyworld_lite.sim import simulate_episode


def test_windowed_features_are_invariant_to_future_frame_mutation() -> None:
    episode = simulate_episode(0, 123).to_jsonable()
    k = prediction_frame(episode)
    before = extract_windowed_feature_groups(episode, k, max_frames=len(episode["frames"]))

    mutated = copy.deepcopy(episode)
    mutate_future_frames_for_test(mutated, k)
    after = extract_windowed_feature_groups(mutated, k, max_frames=len(mutated["frames"]))

    assert before == after


def test_windowed_touch_summaries_do_not_reuse_episode_wide_tactile_summary() -> None:
    episode = simulate_episode(0, 123).to_jsonable()
    k = prediction_frame(episode)
    groups = extract_windowed_feature_groups(episode, k, max_frames=len(episode["frames"]))

    assert groups["touch"]["touch_contact_count"] < episode["tactile_summary"]["contact_count"]
    assert groups["touch"]["touch_max_force"] <= episode["tactile_summary"]["max_force"]


def test_vision_snapshot_at_first_contact_does_not_include_post_contact_motion() -> None:
    episode = simulate_episode(0, 123).to_jsonable()
    k = prediction_frame(episode)
    groups = extract_windowed_feature_groups(episode, k, max_frames=len(episode["frames"]))

    assert episode["frames"][k]["touch"]["contact"] > 0
    assert episode["frames"][k]["object"]["vx"] != episode["frames"][k - 1]["object"]["vx"]
    assert groups["vision"]["vision_object_state_frame"] == float(k - 1)
    assert groups["vision"]["vision_obj_vx_k"] == episode["frames"][k - 1]["object"]["vx"]
    assert groups["vision"]["vision_obj_vy_k"] == episode["frames"][k - 1]["object"]["vy"]


def test_hardness_proxy_is_not_emitted_by_any_modality_arm() -> None:
    episode = simulate_episode(1, 1132).to_jsonable()
    groups = extract_windowed_feature_groups(episode, prediction_frame(episode), max_frames=len(episode["frames"]))

    for features in groups.values():
        assert "hardness_proxy" not in features
        assert all("hardness_proxy" not in name for name in features)

    for modalities in ARM_MODALITIES.values():
        flattened = flatten_feature_groups(groups, modalities)
        assert "hardness_proxy" not in flattened
        assert all("hardness_proxy" not in name for name in flattened)


def test_regression_targets_are_forward_rollout_targets() -> None:
    episode = simulate_episode(0, 123).to_jsonable()
    k = prediction_frame(episode)
    targets = regression_targets(episode, k)
    final_state = episode["frames"][-1]["object"]
    state_k = episode["frames"][k]["object"]

    assert set(targets) == {
        "target_delta_x",
        "target_delta_y",
        "target_displacement",
        "target_final_x",
        "target_final_y",
        "target_topple_angle",
    }
    assert targets["target_final_x"] == final_state["x"]
    assert targets["target_final_y"] == final_state["y"]
    assert targets["target_delta_x"] == final_state["x"] - state_k["x"]
    assert targets["target_delta_y"] == final_state["y"] - state_k["y"]
