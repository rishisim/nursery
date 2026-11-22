"""Close-up encoder contract: pinned offline model, no labels, complete finite cache."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nursery.learner import _upstream_module
from nursery.learner.embed import embed_closeups


pytestmark = pytest.mark.usefixtures("offline")


@pytest.fixture
def closeups(episodes, upstream, monkeypatch):
    import torch

    root, output = episodes
    frames = pd.read_parquet(root / "frames.parquet").iloc[[2, 0, 4]].reset_index(drop=True)
    frames.to_parquet(root / "frames.parquet", index=False)
    snapshot = root.parent / "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    snapshot.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (snapshot / name).write_text("synthetic pinned model fixture")
    module = _upstream_module("embed_regions")
    calls = []

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model, *, local_files_only):
            assert Path(model) == snapshot and local_files_only is True
            calls.append("processor")
            return cls()

        def __call__(self, images, return_tensors):
            return Inputs(batch=len(images))

    class Encoder:
        config = type("Config", (), {"hidden_size": 768, "num_register_tokens": 0})()

        @classmethod
        def from_pretrained(cls, model, *, local_files_only):
            assert Path(model) == snapshot and local_files_only is True
            calls.append("model")
            return cls()

        def eval(self):
            return self

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, batch):
            tokens = torch.arange(257).reshape(1, 257, 1).expand(batch, 257, 768).float()
            return type("Result", (), {"last_hidden_state": tokens})()

    monkeypatch.setattr(module, "AutoImageProcessor", Processor)
    monkeypatch.setattr(module, "AutoModel", Encoder)
    return root, output, snapshot, frames, calls


def test_closeup_encoding_pins_offline_model_and_preserves_order(closeups):
    root, output, snapshot, frames, calls = closeups
    embed_closeups(root, output, snapshot, device="cpu", batch=2)
    values = np.load(output / "closeup_cache/emb.f16.npy")
    np.testing.assert_array_equal(values[0, :, 0], np.arange(257))
    pd.testing.assert_frame_equal(pd.read_parquet(output / "closeup_cache/index.parquet")
                                  [["video_id", "frame_idx"]], frames[["video_id", "frame_idx"]])
    assert list(pd.read_parquet(output / "closeup_frames.parquet")) == ["video_id", "frame_idx", "image"]
    record = json.loads((output / "embedding_closeup.json").read_text())
    assert record["complete"] and record["shape"] == [3, 257, 768]
    assert len(record["provenance"]["snapshot_sha256"]) == 3
    assert set(record["cache_sha256"]) == {"index.parquet", "emb.f16.npy"}
    embed_closeups(root, output, snapshot, device="cpu", batch=2)
    assert calls == ["processor", "model"]
    image = root / frames.iloc[0].image
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="provenance changed"):
        embed_closeups(root, output, snapshot, device="cpu")


@pytest.mark.parametrize("failure", ["missing", "escape", "duplicate"])
def test_closeup_encoding_rejects_invalid_inputs(closeups, failure):
    root, output, snapshot, frames, calls = closeups
    if failure == "missing":
        (root / frames.iloc[0].image).unlink()
    elif failure == "escape":
        frames.loc[0, "image"] = "../outside.jpg"
    else:
        frames.loc[1, ["video_id", "frame_idx"]] = frames.loc[0, ["video_id", "frame_idx"]]
    frames.to_parquet(root / "frames.parquet", index=False)
    with pytest.raises(ValueError):
        embed_closeups(root, output, snapshot, device="cpu")
    assert not calls and not (output / "embedding_closeup.json").exists()


def test_closeup_encoding_rejects_nonfinite_cache(closeups):
    root, output, snapshot, _, _ = closeups
    embed_closeups(root, output, snapshot, device="cpu")
    values = np.load(output / "closeup_cache/emb.f16.npy", mmap_mode="r+")
    values[0, 0, 0] = np.nan
    values.flush()
    with pytest.raises(ValueError, match="Nonfinite"):
        embed_closeups(root, output, snapshot, device="cpu")
    assert not json.loads((output / "embedding_closeup.json").read_text())["complete"]


def test_closeup_encoding_rejects_finite_cache_corruption_on_every_retry(closeups):
    root, output, snapshot, _, _ = closeups
    embed_closeups(root, output, snapshot, device="cpu")
    values = np.load(output / "closeup_cache/emb.f16.npy", mmap_mode="r+")
    values[0, 0, 0] = 17
    values.flush()
    for _ in range(2):
        with pytest.raises(ValueError, match="checksum changed"):
            embed_closeups(root, output, snapshot, device="cpu")
