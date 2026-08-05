from pathlib import Path

import pytest

from babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate.runner import (
    AVATAR_SHA256,
    EDITOR_SOURCE_FILES,
    contract_by_episode_id,
    discover_public_asset_project,
    discover_unity_editor,
)


def test_verified_local_unity_and_public_assets_are_discoverable():
    unity = discover_unity_editor()
    assert unity.is_file()
    project = discover_public_asset_project()
    assert project.joinpath("Assets/Avatar/child.fbx").is_file()
    assert len(AVATAR_SHA256) == 64


def test_runner_uses_one_canonical_unity_orchestrator():
    assert EDITOR_SOURCE_FILES.count("ProceduralSceneGateBuilder.cs") == 1
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/ProceduralSceneGateBuilder.cs"
    ).read_text()
    assert "Physics.Simulate(Dt)" in source
    assert "CaptureFrozenFrame(context)" in source
    assert source.index("Physics.Simulate(Dt)") < source.index("CaptureFrozenFrame(context)")
    assert "object_pose_writes_after_initialization = 0" in source
    assert "independent_render_timeline = false" in source


def test_contract_lookup_rejects_nonfrozen_episode():
    contract = contract_by_episode_id("warm_playroom__sunset_play")
    assert contract["scene_spec"]["room_family"] == "warm_playroom"
    with pytest.raises(ValueError):
        contract_by_episode_id("ad_hoc_seed_specific_episode")


def test_cli_exposes_one_entry_point_for_ordered_stage_runs():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/cli.py"
    ).read_text()
    assert "run-episode" in source
    assert "garment_sweep" in source
    assert "motion_camera" in source
    assert "bimanual_cell" in source
    assert "integrated" in source
