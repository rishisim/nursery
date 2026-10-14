"""Adapt episode and probe paths to upstream's unchanged region encoder."""

import json
from pathlib import Path
import sys

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


def embed_episodes(episodes, output, device=None, batch=64):
    from unittest.mock import patch

    import pandas as pd
    import torch

    device = str(torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")))
    if batch < 1:
        raise ValueError("Embedding batch must be positive")
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
                  "encoder": "facebook/dinov2-base", "grid": 4, "cls": True}
    if embedding_path.exists():
        saved = json.loads(embedding_path.read_text())
        if {k: v for k, v in saved.items() if k not in ("device", "batch")} != provenance:
            raise ValueError("Embedding provenance disagrees with this episode root or encoder")
    frames = pd.read_parquet(episodes / "frames.parquet")
    sessions = pd.read_parquet(episodes / "sessions.parquet")
    frames = frames[frames.video_id.isin(sessions.loc[sessions.split == "train", "video_id"])]
    probes = pd.read_parquet(episodes / "probes.parquet")
    if set(probes.room) & set(sessions.loc[sessions.split == "train", "room"]):
        raise ValueError("Probe rooms must be disjoint from training rooms")
    tables = {"train": frames, "probes": probe_frames(probes, config)}
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
            paths = {(r.video_id, int(r.frame_idx)): episodes / r.image for r in table.itertuples()}
            if not all(p.is_file() and p.resolve().is_relative_to(episodes) for p in paths.values()):
                raise ValueError(f"Missing images or paths outside episode root: {name}")
            table_path = output / f"{name}_frames.parquet"
            table.to_parquet(table_path, index=False)
            argv = ["embed_regions", "--frames", str(table_path), "--out", str(output / f"{name}_cache"),
                    "--grid", "4", "--batch", str(batch)]
            with patch.object(sys, "argv", argv), \
                    patch.object(upstream.AutoModel, "from_pretrained", side_effect=_load_model), \
                    patch.object(upstream.AutoImageProcessor, "from_pretrained", side_effect=_load_processor), \
                    patch.object(upstream, "frame_path", side_effect=lambda v, f: paths[(v, int(f))]):
                upstream.main()
    return output
