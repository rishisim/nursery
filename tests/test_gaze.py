"""Scene-only gaze and causal retargeting of a fixed caregiver schedule."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_frames
from nursery.common import load_config
from nursery.simulator.gaze import generate_gaze, gaze_summary
from nursery.speech.transcripts import couple_gaze, generate_transcripts


def scene(n=10000, videos=2):
    frames = synthetic_frames(n, videos)
    frames["clip"] = "long_clip"
    frames["held_category"] = "mug"
    objects = [{"id": i + 1, "category": c, "visible_px": 100,
                "image_xy": [40 + 60 * i, 80]} for i, c in enumerate(["mug", "bowl", "apple", "banana"])]
    frames["objects"] = json.dumps(objects)
    return frames


def test_gaze_scene_only_determinism_stationary_share_and_bouts():
    frames = scene()
    gaze = generate_gaze(frames, 5)
    pd.testing.assert_frame_equal(gaze, generate_gaze(frames, 5))
    # Arbitrary speech-like columns cannot influence gaze.
    frames["referent"] = "banana"
    frames["text"] = "look at the banana"
    pd.testing.assert_frame_equal(gaze, generate_gaze(frames, 5))
    report = gaze_summary(gaze)
    assert report["held_fraction_when_available"] == pytest.approx(.60, abs=.025)
    assert report["observed_bout_mean_s"] == pytest.approx(2.334, abs=.10)
    assert report["duration_limited_frames"] == 0
    assert ((gaze.gaze_x >= 0) & (gaze.gaze_x < 1)).all()
    assert ((gaze.gaze_y >= 0) & (gaze.gaze_y < 1)).all()


def test_gaze_missing_visibility_cuts_and_duplicate_categories():
    frames = scene(12, 1)
    frames.loc[2:4, "objects"] = "[]"
    frames.loc[6:, "clip"] = "second"
    objects = [{"id": 1, "category": "mug", "visible_px": 10, "image_xy": [10, 20]},
               {"id": 2, "category": "mug", "visible_px": 10, "image_xy": [220, 100]}]
    frames.loc[7:, "objects"] = json.dumps(objects)
    gaze = generate_gaze(frames, 0)
    assert gaze.loc[2:4, "gaze_object_id"].isna().all()
    assert gaze.loc[2:4, ["gaze_x", "gaze_y"]].isna().all().all()
    assert gaze.loc[6, "gaze_bout"] != gaze.loc[5, "gaze_bout"]
    for row in gaze.iloc[7:].itertuples():
        assert row.gaze_x == objects[row.gaze_object_id - 1]["image_xy"][0] / 320
        assert row.gaze_held == (row.gaze_object_id == 1)
    invalid = scene(2, 1)
    invalid["objects"] = '[{"id": 1, "category": "mug", "visible_px": 1, "image_xy": [-1, 10]}]'
    with pytest.raises(ValueError, match="in-frame"):
        generate_gaze(invalid)


def test_coupling_preserves_schedule_and_matches_feasible_channels():
    frames = scene(12000, 2)
    # A fifth object occurs briefly in each clip so it is an unseen candidate
    # elsewhere, while the held and gazed visible channels remain available.
    objects = json.loads(frames.objects.iloc[0])
    frames.loc[frames.frame_idx < 10, "objects"] = json.dumps(objects + [
        {"id": 5, "category": "cup", "visible_px": 1, "image_xy": [120, 100]}])
    config = load_config()
    config["vocabulary"]["cup"] = "cup"
    config["generator"].update(rate_hz=1, p_ref=1, p_name=1)
    old, _ = generate_transcripts(frames, config, 17)
    gaze = generate_gaze(frames, 4)
    changed, report = couple_gaze(old, frames, gaze, config, 17)
    fixed = ["video_id", "frame_idx", "source_frame_idx", "utterance_time_s", "named",
             "referential", "jitter_draw_frames", "jitter_frames", "weight_none", "weight_contact"]
    pd.testing.assert_frame_equal(old[fixed], changed[fixed])
    assert report["named"]["source_gazed_given_seen"] == pytest.approx(.55, abs=.025)
    assert report["named"]["channel_shares"]["held"] == pytest.approx(.35, abs=.02)
    assert report["named"]["channel_shares"]["unseen"] == pytest.approx(.24, abs=.02)
    assert not old.text.equals(changed.text)
    pd.testing.assert_frame_equal(gaze, generate_gaze(frames, 4))
    assert changed.weight_oracle.mean() == pytest.approx(1)


def test_coupling_stops_when_gaze_opportunity_unavailable():
    frames = scene(60, 2)
    config = load_config()
    old, _ = generate_transcripts(frames, config, 0)
    gaze = generate_gaze(frames, 0, p_held=1)
    with pytest.raises(ValueError, match="no eligible|infeasible"):
        couple_gaze(old, frames, gaze, config, 0)
