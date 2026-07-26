from pathlib import Path

import pytest

from babyworld_lite.childlens_engine_bakeoff.physics_kernel import (
    build_kernel_model,
    phase_at,
)


EXTERNAL_ROOT = Path(".external/engine_bakeoff")


def test_phase_schedule_is_driven_by_resolved_spec():
    spec = {
        "phase_timestamps": {
            "value": [
                {"phase": "look_settle", "start_s": 0.0, "end_s": 1.0},
                {"phase": "near_miss", "start_s": 1.0, "end_s": 2.0},
            ]
        }
    }
    assert phase_at(spec, 0.5)["phase"] == "look_settle"
    assert phase_at(spec, 1.5)["phase"] == "near_miss"


@pytest.mark.skipif(
    not EXTERNAL_ROOT.exists(), reason="pinned external engine assets unavailable"
)
def test_compiled_kernel_has_explicit_all_hand_pairs_and_free_scene_bits(tmp_path):
    scene = (
        EXTERNAL_ROOT
        / "assets/molmospaces/scenes/ithor/FloorPlan201_physics.xml"
    )
    mimo_assets = Path(
        "/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/repos/"
        "MIMo/mimoEnv/assets"
    )
    if not scene.exists() or not mimo_assets.exists():
        pytest.skip("pinned external scene or MIMo assets unavailable")
    kernel = build_kernel_model(
        scene,
        mimo_assets,
        tmp_path / "component.xml",
        root_xy=(-0.49, 0.29),
        target_definition={
            "geometry": "cylinder_with_three_capsule_handle",
            "rgba": [0.92, 0.72, 0.05, 1.0],
        },
    )
    model = kernel.model
    paired_hand_geoms = {
        int(model.pair_geom1[pair_id])
        if int(model.pair_geom2[pair_id]) == kernel.target_geom_id
        else int(model.pair_geom2[pair_id])
        for pair_id in range(model.npair)
        if kernel.target_geom_id
        in {int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id])}
    }
    assert set(kernel.hand_geom_ids) <= paired_hand_geoms
    assert kernel.hand_target_collision_bit != kernel.target_support_collision_bit
    assert kernel.hand_target_collision_bit == 16
    assert kernel.target_support_collision_bit == 32
