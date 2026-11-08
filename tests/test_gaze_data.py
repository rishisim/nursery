"""Gaze geometry, compact-cache and partition-safe window contracts."""

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.learner.gaze_data import GazeData, dino_crop_points


def fixture_data(tmp_path, full=True):
    gaze = pd.DataFrame({"video_id": ["v"] * 5, "frame_idx": range(5),
                         "gaze_x": [0.3, 0.4, np.nan, 0.6, 0.7],
                         "gaze_y": [0.5, 0.4, np.nan, 0.6, 0.5],
                         "gaze_category": ["DO NOT READ"] * 5})
    manifest = gaze.iloc[[0, 1, 3, 4]][["video_id", "frame_idx"]].copy()
    lut = {f"v|{i}": (0, i) for i in range(5)}
    partitions = {k: "train" for k in lut}
    cache = tmp_path / "cache"
    cache.mkdir()
    index = gaze.iloc[[4, 0, 3, 1]][["video_id", "frame_idx"]].copy()
    index["row"] = range(len(index))
    index.to_parquet(cache / "index.parquet")
    values = np.repeat(index.frame_idx.to_numpy()[:, None], 3, axis=1).astype(np.float16)
    if full:
        values = np.repeat(values[:, None], 257, axis=1)
        values[:, 0] = 100  # CLS is excluded from the compact summary.
    np.save(cache / "emb.f16.npy", values)
    return manifest, lut, gaze, partitions, cache


def test_crop_matches_pinned_processor_integer_geometry():
    from transformers.image_transforms import get_resize_output_image_size, center_crop

    # Check the installed encoder processor's integer resize and crop offsets.
    image = np.zeros((240, 320, 3), np.uint8)
    rh, rw = get_resize_output_image_size(image, 256, default_to_square=False,
                                          input_data_format="channels_last")
    assert (rh, rw) == (256, 341)
    pixels = np.stack(np.meshgrid(np.arange(rw), np.arange(rh)), axis=-1)
    cropped = center_crop(pixels, (224, 224), input_data_format="channels_last")
    np.testing.assert_array_equal(cropped[0, 0], [58, 16])
    points = dino_crop_points([[0.5, 0.5], [0, 0], [0.99, 0.99]])
    np.testing.assert_allclose(points[0], [(170.5 - 58) / 224, 0.5])
    assert (points[1] < 0).all() and (points[2] > 1).all()


@pytest.mark.parametrize("full", [True, False])
def test_compact_features_framepairs_order_nulls_padding_and_no_labels(tmp_path, full):
    manifest, lut, gaze, partitions, cache = fixture_data(tmp_path, full)
    data = GazeData(manifest, lut, gaze, partitions=partitions, closeup_cache=cache, window=1)
    ctx = data.batch([0, 1])
    assert data.frame_keys[0] == ["v|0", "v|1"]
    assert ctx["gaze_valid"].tolist() == [[True, True, False], [True, True, False]]
    np.testing.assert_array_equal(ctx["closeup"][0, :, 0], [0, 1, 0])
    assert set(ctx) == {"gaze_point", "gaze_valid", "closeup", "keep"}
    # Audit labels cannot affect model inputs.
    gaze["gaze_category"] = "different"
    other = GazeData(manifest, lut, gaze, partitions=partitions, closeup_cache=cache, window=1)
    for key in ctx:
        torch.testing.assert_close(ctx[key], other.batch([0, 1])[key])


def test_point_only_and_removed_need_no_closeup(tmp_path):
    manifest, lut, gaze, partitions, _ = fixture_data(tmp_path)
    data = GazeData(manifest, lut, gaze, partitions=partitions, point_only=True)
    assert "closeup" not in data.batch([0])
    assert data.batch([0], mode="removed") is None
    assert not data.batch([0], dropout=1)["keep"].any()
    with pytest.raises(ValueError, match="Close-up cache"):
        GazeData(manifest, lut, gaze, partitions=partitions)


def test_shuffle_uses_linked_donor_and_never_crosses_partition(tmp_path):
    manifest, lut, gaze, partitions, cache = fixture_data(tmp_path)
    partitions.update({"v|3": "dev", "v|4": "dev"})
    data = GazeData(manifest, lut, gaze, partitions=partitions, closeup_cache=cache, window=0)
    assert data.donor_indices.tolist() == [1, 0, 3, 2]
    real, shuffled = data.batch(range(4)), data.batch(range(4), mode="shuffled")
    for key in ("gaze_point", "gaze_valid", "closeup"):
        torch.testing.assert_close(shuffled[key], real[key][data.donor_indices])


def test_rejects_partition_leakage_and_unselected_experience(tmp_path):
    manifest, lut, gaze, partitions, _ = fixture_data(tmp_path)
    partitions["v|1"] = "test"
    with pytest.raises(ValueError, match="partition"):
        GazeData(manifest, lut, gaze, partitions=partitions, point_only=True)
    partitions["v|1"] = "train"
    with pytest.raises(ValueError, match="selected experience"):
        GazeData(manifest, lut, gaze, partitions=partitions, point_only=True,
                 allowed_keys={"v|0"})


def test_missing_and_corrupt_inputs_fail_closed(tmp_path):
    manifest, lut, gaze, partitions, cache = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="cover"):
        GazeData(manifest, lut, gaze.iloc[1:], partitions=partitions, point_only=True)
    gaze.loc[0, "gaze_x"] = np.nan
    with pytest.raises(ValueError, match="wholly null"):
        GazeData(manifest, lut, gaze, partitions=partitions, point_only=True)
    gaze.loc[0, "gaze_x"] = 0.3
    values = np.load(cache / "emb.f16.npy")
    values[0, 1, 0] = np.nan
    np.save(cache / "emb.f16.npy", values)
    with pytest.raises(ValueError, match="Nonfinite"):
        GazeData(manifest, lut, gaze, partitions=partitions, closeup_cache=cache)


def test_shuffle_rejects_single_unique_moment(tmp_path):
    manifest, lut, gaze, partitions, _ = fixture_data(tmp_path)
    manifest = pd.concat([manifest.iloc[:1], manifest.iloc[:1]])
    data = GazeData(manifest, lut, gaze, partitions=partitions, point_only=True)
    with pytest.raises(ValueError, match="another moment"):
        data.batch([0], mode="shuffled")
