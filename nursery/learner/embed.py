"""Adapt episode and probe paths to upstream's unchanged region encoder."""

import json
from pathlib import Path
import sys

from nursery.common import REPO, _output_directory, load_config
from nursery.learner import _upstream_module


def _encode_bounded(upstream, argv, table, cache, shard_rows=4096):
    """Keep upstream encoding unchanged while bounding its in-memory output.

    A full 16x16 training cache exceeds local RAM. Upstream's deterministic
    shards are merged into a disk-backed array in the original frame order.
    Publish the index only after the complete array is written.
    """
    import math
    import shutil
    import numpy as np
    import pandas as pd
    from unittest.mock import patch

    expected = table[["video_id", "frame_idx"]].reset_index(drop=True).copy()
    expected["frame_idx"] = expected.frame_idx.astype(int)
    index_path = cache / "index.parquet"
    if index_path.exists():
        saved = pd.read_parquet(index_path)
        pd.testing.assert_frame_equal(saved[["video_id", "frame_idx"]], expected)
        if not np.array_equal(saved.row, np.arange(len(expected))):
            raise ValueError("Existing cache row ordering disagrees with frame table")
        if len(np.load(cache / "emb.f16.npy", mmap_mode="r")) != len(expected):
            raise ValueError("Existing cache length disagrees with frame table")
        return
    if len(table) <= shard_rows:
        with patch.object(sys, "argv", argv):
            upstream.main()
        return
    cache.mkdir(parents=True, exist_ok=True)
    shards = cache / ".embedding_shards"
    nshards = math.ceil(len(table) / shard_rows)
    merged = None
    for shard in range(nshards):
        part = shards / str(shard)
        part_argv = list(argv)
        part_argv[part_argv.index("--out") + 1] = str(part)
        part_argv += ["--shard", str(shard), "--nshards", str(nshards)]
        with patch.object(sys, "argv", part_argv):
            upstream.main()
        index = pd.read_parquet(part / "index.parquet")
        pd.testing.assert_frame_equal(index[["video_id", "frame_idx"]],
                                      expected.iloc[shard::nshards].reset_index(drop=True))
        values = np.load(part / "emb.f16.npy", mmap_mode="r")
        if len(values) != len(index) or not np.array_equal(index.row, np.arange(len(index))):
            raise ValueError("Incomplete or misordered embedding shard")
        if merged is None:
            merged = np.lib.format.open_memmap(cache / "emb.partial.npy", mode="w+",
                                               dtype=values.dtype, shape=(len(expected), *values.shape[1:]))
        if values.shape[1:] != merged.shape[1:] or values.dtype != merged.dtype:
            raise ValueError("Embedding shards disagree on dimensions or dtype")
        merged[shard::nshards] = values
        merged.flush()
        print(f"Embedded shard {shard + 1}/{nshards}", flush=True)
    del merged, values
    (cache / "emb.partial.npy").replace(cache / "emb.f16.npy")
    expected["row"] = np.arange(len(expected))
    expected.to_parquet(cache / "index.partial.parquet")
    (cache / "index.partial.parquet").replace(index_path)
    shutil.rmtree(shards)


def probe_frames(probes, config):
    table = probes.copy()
    if table.probe_id.duplicated().any() or not table.split.isin(["dev", "test"]).all():
        raise ValueError("Probe IDs must be unique and splits dev/test")
    table["video_id"] = "probe/" + table.probe_id.astype(str)
    table["frame_idx"] = 0
    table["category"] = table.category.map(config["vocabulary"])
    if table.category.isna().any():
        raise ValueError("Probe category missing from vocabulary")
    return table


def konkle_frames(config):
    """Konkle test-60: 17 photos per category, keyed like frames.

    The public set keeps 15 photos per category folder and 2 more under TestItems/; Vong's split
    (Frank's konkle/<category>/*.jpg) is all 17, so both are included, sorted by file name.
    """
    import pandas as pd

    root = REPO / config["konkle"]["root"]  # relative to the repo root unless absolute
    rows = []
    for category in sorted(config["konkle"]["categories"]):
        photos = sorted((p for p in (root / category).rglob("*") if p.suffix.lower() == ".jpg"
                         and not p.name.startswith(".")), key=lambda p: p.name)
        if len(photos) != 17:
            raise ValueError(f"Konkle {category}: expected 17 photos, found {len(photos)}")
        rows.extend({"video_id": f"konkle/{category}", "frame_idx": index, "category": category, "path": str(path)}
                    for index, path in enumerate(photos))
    return pd.DataFrame(rows)


def embed_episodes(episodes, output, device=None, batch=64, grid=16):
    from unittest.mock import patch

    import pandas as pd
    import numpy as np
    import torch

    device = str(torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")))
    if batch < 1:
        raise ValueError("Embedding batch must be positive")
    if grid not in (4, 16):
        raise ValueError("Embedding grid must be 4 (historical) or 16 (full patch grid)")
    episodes = Path(episodes).resolve()
    output = _output_directory(output)
    record_path = output / "generator.json"
    config = load_config()
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if Path(record["episodes"]) != episodes:
            raise ValueError("Embedding episodes disagree with the transcript run")
        config = record["config"]
    embedding_path = output / "embedding.json"
    provenance = {"episodes": str(episodes), "upstream": config["upstream"],
                  "encoder": "facebook/dinov2-base", "grid": grid, "cls": True}
    if embedding_path.exists():
        saved = json.loads(embedding_path.read_text())
        if {k: v for k, v in saved.items() if k not in ("device", "batch")} != provenance:
            raise ValueError("Embedding provenance disagrees with this episode root, encoder or grid; use a fresh output root")
    for name in ("train", "probes", "konkle"):
        cache_path = output / f"{name}_cache" / "emb.f16.npy"
        if cache_path.exists():
            cache = np.load(cache_path, mmap_mode="r")
            if cache.ndim != 3 or cache.shape[1] != 1 + grid * grid:
                raise ValueError(f"Existing {name} cache disagrees with the requested grid; use a fresh output root")
    frames = pd.read_parquet(episodes / "frames.parquet")
    sessions = pd.read_parquet(episodes / "sessions.parquet")
    frames = frames[frames.video_id.isin(sessions.loc[sessions.split == "train", "video_id"])]
    probes = pd.read_parquet(episodes / "probes.parquet")
    if set(probes.room) & set(sessions.loc[sessions.split == "train", "room"]):
        raise ValueError("Probe rooms must be disjoint from training rooms")
    tables = {"train": frames, "probes": probe_frames(probes, config)}
    if "konkle" in config:
        tables["konkle"] = konkle_frames(config)
    embedding_path.write_text(json.dumps({**provenance, "device": device, "batch": batch}, indent=2) + "\n")
    # HF downloads also belong to the ignored run; tests replace the encoder loaders.
    with patch.dict("os.environ", {"HF_HOME": str(output / "encoder_cache")}):
        upstream = _upstream_module("embed_regions")
        model_loader = upstream.AutoModel.from_pretrained
        processor_loader = upstream.AutoImageProcessor.from_pretrained

        # Upstream has no device flag; route its two transfers without changing CUDA detection.
        def _load_model(model_id):
            model = model_loader(model_id)
            move = model.to
            model.to = lambda _device: move(device)
            return model

        def _load_processor(model_id):
            processor = processor_loader(model_id)

            def _process(**kwargs):
                inputs = processor(**kwargs)
                move = inputs.to
                inputs.to = lambda _device: move(device)
                return inputs

            return _process

        for name, table in tables.items():
            if table.empty or table.duplicated(["video_id", "frame_idx"]).any():
                raise ValueError(f"Empty or duplicate {name} frame keys")
            if name == "konkle":
                paths = {(r.video_id, int(r.frame_idx)): Path(r.path) for r in table.itertuples()}
                table = table.drop(columns="path")
            else:
                paths = {(r.video_id, int(r.frame_idx)): episodes / r.image for r in table.itertuples()}
            if name != "konkle" and not all(p.is_file() and p.resolve().is_relative_to(episodes) for p in paths.values()):
                raise ValueError(f"Missing images or paths outside episode root: {name}")
            table_path = output / f"{name}_frames.parquet"
            table.to_parquet(table_path, index=False)
            argv = ["embed_regions", "--frames", str(table_path), "--out", str(output / f"{name}_cache"),
                    "--grid", str(grid), "--batch", str(batch)]
            with patch.object(sys, "argv", argv), \
                    patch.object(upstream.AutoModel, "from_pretrained", side_effect=_load_model), \
                    patch.object(upstream.AutoImageProcessor, "from_pretrained", side_effect=_load_processor), \
                    patch.object(upstream, "frame_path", side_effect=lambda v, f: paths[(v, int(f))]):
                _encode_bounded(upstream, argv, table, output / f"{name}_cache")
    return output
