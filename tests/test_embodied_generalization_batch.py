import copy
import json
from pathlib import Path

import numpy as np

from babyworld_lite.childlens_engine_bakeoff.generalization_batch import (
    _grayscale_metrics,
    build_cells,
    materialize_cell,
)
from babyworld_lite.childlens_engine_bakeoff.physics_kernel import (
    _target_geom_names,
)


CONFIG = json.loads(
    Path("configs/embodied_simulation_generalization.json").read_text()
)
EPISODE = json.loads(
    Path("configs/embodied_simulation_episode.json").read_text()
)
VERTICAL = json.loads(
    Path("configs/embodied_simulation_vertical_slice.json").read_text()
)
REGISTRY = json.loads(
    Path("configs/childlens_mimo_molmospaces_asset_registry.json").read_text()
)


def test_generalization_matrix_is_frozen_and_complete():
    cells = build_cells(CONFIG)
    assert CONFIG["frozen_before_phase_5_outcomes_at_utc"]
    assert len(cells) == 12
    assert len({cell["cell_id"] for cell in cells}) == 12
    assert {cell["scene_variant"] for cell in cells} == {
        "sparse",
        "household",
        "messy",
    }
    assert {cell["target"]["persistent_id"] for cell in cells} == {
        "yellow_cup_authored",
        "red_ball_authored",
    }
    assert {cell["seed"]["seed"] for cell in cells} == {
        20260731,
        20260801,
    }
    assert CONFIG["matrix"]["camera_mount_changed_between_cells"] is False
    assert (
        CONFIG["matrix"][
            "physics_or_qualification_threshold_changed_between_cells"
        ]
        is False
    )


def test_materialized_ball_cell_changes_identity_language_and_placement_only():
    cell = next(
        cell
        for cell in build_cells(CONFIG)
        if cell["cell_id"] == "sparse__red_ball_authored__20260801"
    )
    episode, vertical = materialize_cell(EPISODE, VERTICAL, REGISTRY, cell)
    assert episode["scene_variant"] == "sparse"
    assert episode["controller"]["seed"] == 20260801
    assert episode["planner"]["target_reference"] == {
        "requested_label": "red ball",
        "required_persistent_id": "red_ball_authored",
    }
    assert episode["language"]["utterances"][0]["text"] == (
        "Look, a red ball."
    )
    assert episode["language"]["utterances"][-1]["text"] == "Red ball."
    assert vertical["scene_family"]["target"]["geometry"] == "sphere"
    assert vertical["scene_family"]["target"]["persistent_id"] == (
        "red_ball_authored"
    )
    placement = vertical["scene_family"]["qualified_placement"]
    assert placement["target_offset_from_root_m"] == [0.33, -0.142, 0.1]
    assert placement["support_offset_from_root_m"] == [0.33, -0.142, 0.025]
    assert np.allclose(
        placement["reach_distractor_offset_from_root_m"],
        [0.235, 0.015, 0.1],
    )
    assert episode["activity"] == EPISODE["activity"]
    assert vertical["frozen_gates"] == VERTICAL["frozen_gates"]
    assert vertical["embodiment"]["camera_mount"] == VERTICAL["embodiment"][
        "camera_mount"
    ]


def test_materialization_does_not_mutate_canonical_inputs():
    episode = copy.deepcopy(EPISODE)
    vertical = copy.deepcopy(VERTICAL)
    cell = build_cells(CONFIG)[-1]
    materialize_cell(episode, vertical, REGISTRY, cell)
    assert episode == EPISODE
    assert vertical == VERTICAL


def test_target_contact_pair_names_follow_geometry():
    assert _target_geom_names(REGISTRY["red_ball_authored"]) == (
        "target_geom",
    )
    assert _target_geom_names(REGISTRY["yellow_cup_authored"]) == (
        "target_geom",
        "target_handle_upper",
        "target_handle_outer",
        "target_handle_lower",
    )


def test_visual_motion_uses_frozen_grayscale_definition():
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    sampling = {
        "grayscale_size_px": [64, 64],
        "scene_change_motion_threshold": 0.2,
    }
    metrics, images = _grayscale_metrics([black, white, black], sampling)
    assert len(images) == 3
    assert metrics["motion"] == 1.0
    assert metrics["scene_change_rate"] == 1.0
