import json
from pathlib import Path

import numpy as np

from babyworld_lite.childlens_engine_bakeoff.appearance_experiment import (
    _depth_rgb,
    _edge_rgb,
    _frame_rate,
    _protected_masks,
)


def test_appearance_windows_and_seeds_are_frozen_before_outcomes():
    config = json.loads(
        Path("configs/embodied_simulation_appearance.json").read_text()
    )
    assert config["frozen_before_phase_3_outcomes_at_utc"]
    assert config["seeds"] == [20260734, 20260735, 20260736]
    assert len(config["methods"]["cosmos3_nano"]["model_revision"]) == 40
    assert len(config["methods"]["cosmos3_nano"]["tokenizer_revision"]) == 40
    assert len(config["methods"]["cosmos3_nano"]["vision_vae_revision"]) == 40
    assert len(config["methods"]["oscar_2b"]["model_revision"]) == 40
    assert len(config["methods"]["oscar_2b"]["text_encoder_revision"]) == 40
    assert (
        len(config["methods"]["oscar_2b"]["text_encoder_processor_revision"])
        == 40
    )
    assert len(config["methods"]["oscar_2b"]["vision_vae_revision"]) == 40
    assert [window["id"] for window in config["windows"]] == [
        "near_miss",
        "contact_grasp",
        "inspect_head_turn",
        "release_settle",
    ]
    for window in config["windows"]:
        assert 3.0 <= window["encoded_duration_s"] <= 5.0
        assert np.isclose(
            window["encoded_duration_s"], window["frames"] / window["fps"]
        )
        assert np.isclose(
            window["sample_span_s"], (window["frames"] - 1) / window["fps"]
        )


def test_juno_run_uses_bundled_cosmos_tokenizer_but_keeps_samples_silent():
    run_script = Path(
        "babyworld_lite/childlens_engine_bakeoff/juno_appearance_run.sh"
    ).read_text()
    assert 'sound["from_checkpoint"] = True' in run_script
    assert '"sound_generation_enabled_in_samples": False' in run_script
    assert "--experiment-overrides" not in run_script
    assert 'cosmos_site_packages}/nvidia' in run_script


def test_depth_edge_and_protected_mask_encodings_are_bounded():
    depth = np.asarray([[0.02, 5.0, np.nan]], dtype=np.float32)
    encoded = _depth_rgb(depth, 0.02, 5.0)
    assert encoded.shape == (1, 3, 3)
    assert encoded[0, 0].tolist() == [255, 255, 255]
    assert encoded[0, 1].tolist() == [0, 0, 0]
    assert encoded[0, 2].tolist() == [0, 0, 0]

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, 4:] = 255
    edge = _edge_rgb(rgb, 0.12)
    assert edge.shape == rgb.shape
    assert edge.max() == 255

    segmentation = np.zeros((1, 9, 9, 2), dtype=np.int32)
    segmentation[0, 4, 4] = [7, 5]
    core, alpha = _protected_masks(
        segmentation, {7}, dilation_px=1, feather_px=1
    )
    assert core.shape == (1, 9, 9)
    assert int(core.sum()) == 9
    assert np.all(alpha[core] == 255)
    assert _frame_rate("30/2") == 15.0
