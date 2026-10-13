"""Train the upstream two-tower with pair weights on CUDA when available."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _output_directory, _upstream_module


def train_run(run, condition, seed, epochs=None):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    upstream = _upstream_module("train_frame_mil")
    region = _upstream_module("train_region_mil")
    run = Path(run).resolve()
    generator = json.loads((run / "generator.json").read_text())
    config = generator["config"]
    settings = config["training"]
    epochs = settings["epochs"] if epochs is None else epochs
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifest_path = run / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    column = f"weight_{condition}"
    if column not in manifest:
        raise ValueError(f"Unknown cue condition: {condition}")
    vocab = upstream.build_vocab(manifest.text, settings["min_freq"])
    caches, lut = upstream.load_multi([run / "train_cache"])
    emb_dim = caches[0].shape[-1]
    if caches[0].ndim != 3 or caches[0].shape[1] != 17:
        raise ValueError("Expected CLS + 4x4 region caches")
    # FramePairs drops vocab-empty rows, so filter first to keep w aligned with its indices.
    kept = manifest.text.map(lambda text: bool(upstream.encode(text, vocab)))
    manifest = manifest[kept].reset_index(drop=True)
    if any(region.frame_key(r.video_id, r.frame_idx) not in lut for r in manifest.itertuples()):
        raise ValueError("Training cache does not cover the manifest")
    ds = upstream.FramePairs(caches, lut, manifest, vocab, settings["window"], False)
    if len(ds) < 2 or ds.n_noframe or ds.n_novocab:
        raise ValueError("Need at least two fully cached, vocabulary-bearing training pairs")
    weights = manifest[column].to_numpy(dtype=np.float32)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Pair weights must be finite and positive")
    weights /= weights.mean()
    weights = torch.from_numpy(weights)
    dl = torch.utils.data.DataLoader(ds, batch_size=settings["batch"], shuffle=True,
                                     drop_last=False, collate_fn=upstream.collate, num_workers=0)
    model = upstream.RegionMIL(len(vocab), emb_dim=emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.1)
    probes = pd.read_parquet(run / "probes_frames.parquet")
    probe_cache, probe_lut = region.load_region_cache(run / "probes_cache")
    if probe_cache.shape[1:] != (17, emb_dim):
        raise ValueError("Probe and training cache dimensions disagree")
    if any(region.frame_key(r.video_id, r.frame_idx) not in probe_lut for r in probes.itertuples()):
        raise ValueError("Probe cache does not cover all probes")
    tables = {split: probes[probes.split == split] for split in ("dev", "test")}
    for split, table in tables.items():
        counts = table.groupby("category").size()
        if len(counts) < 4 or (counts < 4).any():
            raise ValueError(f"Upstream 4AFC requires at least four probes/category and four categories: {split}")
        if sum(bool(upstream.encode(cat, vocab)) for cat in counts.index) < 4:
            raise ValueError(f"Upstream 4AFC needs four in-vocabulary categories: {split}")
    output = _output_directory(run / f"{condition}_s{seed}")
    if any(output.iterdir()):
        raise FileExistsError(f"Training output already contains files: {output}")
    history, best_sel, best_state, best_ep = [], -1.0, None, -1
    for epoch in range(epochs):
        model.train()
        losses = []
        for idx, v, t, n in dl:
            loss = model.forward_loss(v.to(device), t.to(device), n.to(device), weights[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        # Evaluation sees only image regions and word tokens; no cues or oracle labels.
        scores, details = {}, {}
        for split, table in tables.items():
            scores[split], details[split] = region.eval_4afc_region(
                model, probe_cache, probe_lut, table, vocab, device, return_detail=True,
            )
        if not all(np.isfinite(list(scores.values()))) or not np.isfinite(losses).all():
            raise ValueError("Non-finite training loss or probe score")
        if scores["dev"] > best_sel:
            best_sel, best_ep = scores["dev"], epoch
            best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        record = {"ep": epoch, "acc": round(scores["test"], 4), "dev": round(scores["dev"], 4),
                  "train_loss": float(np.mean(losses))}
        history.append(record)
        print(record, flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
    metrics = {
        "run": output.name, "seed": seed, "manifest": str(manifest_path),
        "caches": [str(run / "train_cache")], "eval_frames": str(run / "probes_frames.parquet"),
        "window": settings["window"], "cls_only": False, "center": False, "emb_dim": int(emb_dim),
        "epochs": epochs, "n_pairs": len(ds), "vocab": len(vocab), "selection": "dev",
        "selected_epoch": best_ep, "reported_test_acc": round(100 * history[best_ep]["acc"], 3),
        "best_test_acc": round(100 * max(h["acc"] for h in history), 3),
        "final_test_acc": round(100 * history[-1]["acc"], 3), "history": history,
        "condition": condition, "generator_parameters": config["generator"],
        "generator_seed": generator["seed"], "cue_parameters": config["cues"],
        "upstream": config["upstream"], "device": device, "n_manifest": int(len(kept)),
        "n_novocab": int((~kept).sum()), "frame_coverage": 1.0,
        "mean_weight": float(weights.mean()),
        "probe_counts": {s: {k: d[k] for k in ("n_scored", "n_oov", "n_total")} for s, d in details.items()},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"DONE {output} selected_ep {best_ep} reported_test_acc {metrics['reported_test_acc']}", flush=True)
    return metrics
