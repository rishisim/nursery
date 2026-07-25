import json
from pathlib import Path

from babyworld_lite.childlens_asset_rich import load_episode_specs, validate_bakeoff_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_camera_is_an_uncertainty_set_and_seals_learner_work():
    config = json.loads((ROOT / "configs/childlens_room_hand_camera_bakeoff.json").read_text())
    assert config["scope"]["sole_empirical_child_source"] == "ChildLens"
    assert config["scope"]["excluded_empirical_ancestry"] == ["AEA", "BabyView"]
    assert config["scope"]["learner_or_cue_lift_authorized"] is False
    assert config["camera"]["frozen_result"]["status"] == "uncertainty_set"
    assert config["camera"]["learner_outcome_tuning_forbidden"] is True


def test_episode_matrix_has_rooms_actions_controls_and_instance_diversity():
    episodes = load_episode_specs(ROOT / "configs/childlens_room_hand_episodes.json")
    matrix = validate_bakeoff_matrix(episodes)
    assert matrix["episode_count"] == 6
    assert len(matrix["rooms"]) == 3
    assert {"push", "roll", "touch", "lift_and_place", "near_miss"} <= set(matrix["actions"])
    assert matrix["has_near_miss"] is True
    assert matrix["instances_per_category"] == {"ball": 3, "cup": 3}
    assert {item["target"]["split"] for item in episodes} == {"development", "evaluation"}


def test_scored_renderer_forbids_direct_target_location_keyframes():
    source = (ROOT / "babyworld_lite/childlens_asset_rich/pilot_scene.py").read_text()
    assert 'target.keyframe_insert("location"' not in source
    assert "logged_grasp_after_contact" in source
    assert "rigid_body_collision" in source
