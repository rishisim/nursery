import json
from pathlib import Path

import pytest

from babyworld_lite.childlens_engine_bakeoff.unity_native_gate.compiler import compile_episode, compile_scene, validate_reachability, validate_scene_geometry


CONFIG = json.loads(Path("configs/embodied_simulation_unity_native_gate.json").read_text())


def test_single_authority_and_exact_clock():
    assert CONFIG["authority"]["physics"] == "Unity ArticulationBody/PhysX"
    assert CONFIG["authority"]["object_attachment"] is False
    assert CONFIG["unity"]["physics_hz"] // CONFIG["unity"]["render_hz"] == CONFIG["unity"]["steps_per_frame"] == 8


def test_three_prompts_compile_through_one_framework():
    outputs = []
    for row in CONFIG["prompts"]:
        episode = compile_episode(row["text"], row["seed"], row["room_family"], row["duration_s"], "nominal")
        scene = compile_scene(episode)
        outputs.append((episode, scene))
        assert episode.backend == "unity_physx"
        assert episode.contact_regions["required_digits"] == ["thumb", "index", "middle"]
        assert scene["compiled_validation"]["collision_free_initialization_scopes_pass"] is True
        assert scene["compiled_validation"]["approach_volume_clear"] is True
        assert scene["support_relations"]
        assert all("transform" in x and "position_m" in x["transform"] and "scale_m" in x["transform"] for x in scene["instances"])
        assert "transform" in scene["target"] and "transform" in scene["destination"]
    assert {x[0].target["geometry"] for x in outputs} >= {"cup", "rounded_cube"}
    assert len({x[1]["room"]["family"] for x in outputs}) == 3
    assert len({x[1]["variation"]["layout_variant"] for x in outputs}) == 3


def test_prompt_and_seed_change_structured_outputs_without_trajectories():
    a = compile_episode(CONFIG["prompts"][0]["text"], 1, "playroom", 15, "nominal")
    b = compile_episode(CONFIG["prompts"][1]["text"], 2, "living_area", 15, "nominal")
    assert a.target != b.target
    scene_a, scene_b = compile_scene(a), compile_scene(b)
    assert scene_a["variation"] != scene_b["variation"]
    assert scene_a["target"]["transform"]["position_m"] != scene_b["target"]["transform"]["position_m"]
    encoded = json.dumps(a.__dict__)
    assert "joint_trajectory" not in encoded and "camera_keyframe" not in encoded


def test_scene_spec_contains_computed_geometry_receipts():
    row = CONFIG["prompts"][0]
    scene = compile_scene(compile_episode(row["text"], row["seed"], row["room_family"], row["duration_s"], "nominal"))
    validation = scene["compiled_validation"]
    assert validation["structural_checked_non_overlap_pairs"]
    assert validation["tabletop_checked_non_overlap_pairs"]
    assert validation["minimum_approach_clearance_m"] >= scene["approach_corridor"]["radius_m"]
    assert validation["event_sightline"]["unoccluded_by_compiled_aabbs"] is True
    assert {x["child_id"] for x in scene["support_relations"]} >= {scene["target"]["persistent_id"], scene["destination"]["persistent_id"]}
    assert all(abs(x["computed_gap_m"]) <= .001 for x in validation["support_receipts"])
    assert all("intersects" in x and "minimum_sampled_clearance_m" in x for x in validation["event_sightline"]["obstacle_receipts"])


def test_scene_rejects_blocked_sightline():
    row = CONFIG["prompts"][0]
    scene = compile_scene(compile_episode(row["text"], row["seed"], row["room_family"], row["duration_s"], "nominal"))
    obstacle = next(x for x in scene["instances"] if x["persistent_id"] != "primary_support")
    origin, target = scene["compiled_validation"]["event_sightline"]["origin_m"], scene["target"]["transform"]["position_m"]
    obstacle["transform"] = {"position_m": [(origin[i] + target[i]) * .5 for i in range(3)], "rotation_y_deg": 23, "scale_m": [.4, .4, .4]}
    with pytest.raises(ValueError, match="blocked event sightline"):
        validate_scene_geometry(scene)


def test_scene_rejects_overlapping_tabletop_objects():
    row = CONFIG["prompts"][0]
    scene = compile_scene(compile_episode(row["text"], row["seed"], row["room_family"], row["duration_s"], "nominal"))
    scene["distractors"][0]["transform"] = dict(scene["target"]["transform"])
    with pytest.raises(ValueError, match="unintended overlap"):
        validate_scene_geometry(scene)


def test_scene_rejects_out_of_reach_target():
    row = CONFIG["prompts"][0]
    scene = compile_scene(compile_episode(row["text"], row["seed"], row["room_family"], row["duration_s"], "nominal"))
    scene["target"]["transform"]["position_m"] = [-.40, .90, 1.20]
    with pytest.raises(ValueError, match="outside embodiment reach envelope"):
        validate_reachability(scene)


def test_frozen_tolerances_are_not_weaker_than_gate():
    t = CONFIG["frozen_tolerances"]
    assert t["collider_skin_max_m"] <= 0.0075
    assert t["finger_object_penetration_max_m"] <= 0.003
    assert t["minimum_lift_m"] >= 0.08
    assert t["minimum_digits"] >= 3
