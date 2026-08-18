import json
from pathlib import Path

import numpy as np

from babyworld_lite.childlens_engine_bakeoff.determinism import compare
from babyworld_lite.childlens_engine_bakeoff.physics_kernel import _clutter_xml
from babyworld_lite.childlens_engine_bakeoff.run_kernel_episode import _scene_variant


def test_engine_bakeoff_protocol_is_frozen_and_childlens_only():
    protocol = json.loads(
        Path("configs/childlens_mimo_molmospaces_engine_bakeoff.json").read_text()
    )
    assert protocol["frozen_before_outcome_rendering"] is True
    assert protocol["empirical_source"] == "ChildLens only"
    assert protocol["appearance_gates"]["minimum_target_projected_area_fraction"] == 0.015
    assert protocol["developmental_scope"]["mimo_age_months"] == 24
    assert len(protocol["terminal_ladder"]) == 6
    assert protocol["camera_uncertainty_set"]["canonical"] == "equisolid_140_diagonal"
    assert protocol["camera_uncertainty_set"]["learner_outcome_tuning_forbidden"] is True


def test_historical_protocol_does_not_authorize_private_media_or_causal_claims():
    protocol = json.loads(
        Path("configs/childlens_mimo_molmospaces_spec_kernel.json").read_text()
    )
    boundaries = protocol["claim_boundaries"]
    assert boundaries["learner_or_cue_lift_run"] is False
    assert boundaries["private_childlens_material_permitted"] is False
    assert boundaries["developmental_calibration"] == "provisional_not_age_matched"


def test_episode_trace_determinism_treats_matching_nan_samples_as_equal(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    streams = {
        "numeric": np.asarray([0.0, np.nan, np.inf, -np.inf]),
        "phase": np.asarray(["look", "look", "reach", "settle"]),
    }
    np.savez_compressed(first, **streams)
    np.savez_compressed(second, **streams)
    receipt = compare(first, second, atol=1e-9)
    assert receipt["all_pass"] is True
    assert receipt["maximum_numeric_absolute_error"] == 0.0


def test_repaired_kernel_has_fixed_head_mount_and_no_weld_or_mocap_path():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/physics_kernel.py"
    ).read_text()
    assert '"mode": "fixed"' in source
    assert 'visual.set("group", "2")' in source
    assert 'geom.set("group", "3")' in source
    assert "<weld" not in source
    assert "mocap=" not in source
    assert "reach_distractor_pose" in source


def test_mpfb_replay_is_trace_driven_and_depth_occluded():
    overlay = Path(
        "babyworld_lite/childlens_engine_bakeoff/mpfb_overlay_renderer.py"
    ).read_text()
    composite = Path(
        "babyworld_lite/childlens_engine_bakeoff/depth_composite.py"
    ).read_text()
    assert "read-only EpisodeTrace replay" in overlay
    assert "_pose_mimo_chains" in overlay
    assert "foreground_depth" in composite
    assert "+ 0.001" in composite


def test_scene_variants_use_frozen_ordered_clutter_prefixes():
    contract = json.loads(
        Path("configs/embodied_simulation_vertical_slice.json").read_text()
    )
    expected = {"sparse": 3, "household": 12, "messy": 24}
    for variant, count in expected.items():
        resolved, placements, resolved_count = _scene_variant(contract, variant)
        assert resolved == variant
        assert resolved_count == count
        assert len(placements) == count - 1
        assert [item["id"] for item in placements] == [
            f"clutter_{index:02d}" for index in range(1, count)
        ]


def test_authored_clutter_has_separate_hidden_physics_and_visible_layers():
    contract = json.loads(
        Path("configs/embodied_simulation_vertical_slice.json").read_text()
    )
    _, placements, _ = _scene_variant(contract, "household")
    xml = _clutter_xml((0.0, 0.0), placements)
    assert xml.count('name="clutter_physical_') == 11
    assert xml.count('name="clutter_visual_') == 11
    assert xml.count('group="4"') == 11
    assert xml.count('group="0"') == 11
    assert xml.count('contype="0" conaffinity="0"') == 11
