import json
from pathlib import Path


CONTRACT = json.loads(
    Path("configs/embodied_simulation_vertical_slice.json").read_text()
)


def test_vertical_slice_is_frozen_before_outcomes():
    assert CONTRACT["frozen_before_phase_1_outcomes"] is True
    assert CONTRACT["activity"]["duration_s"] == 19.5
    phases = CONTRACT["activity"]["vertical_slice"]
    assert phases[0]["start_s"] == 0.0
    assert phases[-1]["end_s"] == CONTRACT["activity"]["duration_s"]
    assert all(a["end_s"] == b["start_s"] for a, b in zip(phases, phases[1:]))


def test_camera_and_collision_authority_are_immutable():
    mount = CONTRACT["embodiment"]["camera_mount"]
    assert mount["immutable"] is True
    assert mount["world_pose_equation"] == (
        "T_world_camera(t) = T_world_head(t) * T_head_camera"
    )
    collision = CONTRACT["collision_policy"]
    assert collision["static_for_entire_episode"] is True
    assert collision["runtime_collision_disabling_permitted"] is False
    assert collision["object_direct_pose_keyframes_permitted"] is False
    appearance = CONTRACT["appearance_policy"]
    assert appearance["mpfb_role"] == "diagnostic_only_not_phase_1_gate"
    assert appearance["physical_collision_geometry_enabled"] is True
    assert appearance["physical_collision_geometry_visible_in_rgb"] is False
    assert (
        appearance[
            "frozen_physics_camera_contact_sync_determinism_thresholds_changed"
        ]
        is False
    )


def test_assist_requires_contact_and_never_disables_collision():
    grasp = CONTRACT["grasp"]
    assert grasp["minimum_distinct_finger_contacts_for_assist"] >= 2
    assert grasp["minimum_contact_duration_s_for_assist"] > 0
    assert grasp["assist_must_preserve_collisions"] is True
    assert grasp["assist_must_be_flagged_per_frame"] is True


def test_phase_2_required_variants_and_empirical_boundary():
    variants = {row["id"]: row for row in CONTRACT["scene_family"]["variants"]}
    assert variants["sparse"]["required_to_pass_phase_2"] is True
    assert variants["household"]["required_to_pass_phase_2"] is True
    assert variants["messy"]["required_to_pass_phase_2"] is False
    assert CONTRACT["empirical_boundary"]["sole_child_source"] == "ChildLens"
    assert CONTRACT["empirical_boundary"]["restricted_media_permitted"] is False


def test_all_code_pins_are_complete_hashes():
    provenance = CONTRACT["provenance"]
    for component in ("MIMo", "MolmoSpaces", "MPFB"):
        commit = provenance[component]["commit"]
        assert len(commit) == 40
        int(commit, 16)
    blender = provenance["Blender"]
    assert blender["build_hash"] == "396f546c9d82"
    for key in (
        "macos_arm64_distribution_sha256",
        "executable_sha256",
    ):
        assert len(blender[key]) == 64
        int(blender[key], 16)
