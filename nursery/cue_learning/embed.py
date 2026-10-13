"""Adapt episode and probe paths to upstream's unchanged region encoder."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

from . import _output_directory, _upstream_module, load_config


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


def embed_episodes(episodes, output):
    import torch

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
                  "encoder": "facebook/dinov2-base", "grid": 4, "cls": True, "device": "cpu"}
    if embedding_path.exists() and json.loads(embedding_path.read_text()) != provenance:
        raise ValueError("Embedding provenance disagrees with this episode root or encoder")
    frames = pd.read_parquet(episodes / "frames.parquet")
    sessions = pd.read_parquet(episodes / "sessions.parquet")
    frames = frames[frames.video_id.isin(sessions.loc[sessions.split == "train", "video_id"])]
    probes = pd.read_parquet(episodes / "probes.parquet")
    if set(probes.room) & set(sessions.loc[sessions.split == "train", "room"]):
        raise ValueError("Probe rooms must be disjoint from training rooms")
    tables = {"train": frames, "probes": probe_frames(probes, config)}
    embedding_path.write_text(json.dumps(provenance, indent=2) + "\n")
    # HF downloads also belong to the ignored run; tests replace the encoder loaders.
    with patch.dict("os.environ", {"HF_HOME": str(output / "encoder_cache")}):
        upstream = _upstream_module("embed_regions")
        for name, table in tables.items():
            if table.empty or table.duplicated(["video_id", "frame_idx"]).any():
                raise ValueError(f"Empty or duplicate {name} frame keys")
            paths = {(r.video_id, int(r.frame_idx)): episodes / r.image for r in table.itertuples()}
            if not all(p.is_file() and p.resolve().is_relative_to(episodes) for p in paths.values()):
                raise ValueError(f"Missing images or paths outside episode root: {name}")
            table_path = output / f"{name}_frames.parquet"
            table.to_parquet(table_path, index=False)
            argv = ["embed_regions", "--frames", str(table_path), "--out", str(output / f"{name}_cache"),
                    "--grid", "4", "--batch", "16"]
            with patch.object(sys, "argv", argv), patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(upstream, "frame_path", side_effect=lambda v, f: paths[(v, int(f))]):
                upstream.main()
    return output
