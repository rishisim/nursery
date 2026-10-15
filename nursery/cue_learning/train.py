"""Train the upstream two-tower with pair weights on the available accelerator."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _output_directory, _upstream_module


def _groups(details):
    """Accuracy at the selected epoch by word group: held (touched) vs seen-only vs GRAB-only extras."""
    from nursery.egocentric_replay.assets import load_vocabulary

    vocabulary = load_vocabulary()
    held = {w for sources in vocabulary["held"].values() for words in sources.values() for w in words}
    seen = set(vocabulary["words"]) - held
    extras = set(vocabulary["extras"].values())
    result = {}
    for split in ("dev", "test", "konkle"):
        if "per_cat" not in details.get(split, {}):
            continue
        per_cat = details[split]["per_cat"]
        for name, words in (("held", held), ("seen_only", seen), ("extras", extras)):
            values = [per_cat[w] for w in words if w in per_cat]
            if values:
                result[f"{split}_{name}"] = {"acc": round(100 * float(np.mean(values)), 3), "words": len(values)}
    return result


def _body_signals(body):
    from .body import SIGNALS

    if body in ("all", None):
        return SIGNALS
    names = () if body == "geometry" else tuple(body.split(","))
    if not set(names) <= set(SIGNALS):
        raise ValueError(f"Unknown body signals: {body}")
    return names


def _gate_stats(model, bodyset, device, signals):
    """Gate over all training pairs with every signal present, plus the pairs whose window has a hand in contact."""
    import torch

    model.eval()
    with torch.no_grad():
        n = len(bodyset["dist"])
        feats = {k: bodyset[k].to(device) for k in signals}
        g = model.body(feats, torch.ones(n, max(len(signals), 1), device=device), None).cpu().numpy()
    model.train()
    contact = ((bodyset["contact"][:, 0] > 0) | (bodyset["contact"][:, 2] > 0)).numpy()
    return {"mean": round(float(g.mean()), 4), "frac_below_0.1": round(float((g < 0.1).mean()), 4),
            "mean_contact": round(float(g[contact].mean()), 4) if contact.any() else None,
            "frac_contact": round(float(contact.mean()), 4)}


def train_run(run, condition, seed, epochs=None, body=None, body_control="none", body_drop=0.5, name=None, pool="max", tau_p=0.1):
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    upstream = _upstream_module("train_frame_mil")
    region = _upstream_module("train_region_mil")
    run = Path(run).resolve()
    generator = json.loads((run / "generator.json").read_text())
    config = generator["config"]
    settings = config["training"]
    epochs = settings["epochs"] if epochs is None else epochs
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if body is None and body_control != "none":
        raise ValueError("--body-control requires --body")
    if pool not in ("max", "soft"):
        raise ValueError("--pool must be max or soft")
    if body is not None and pool != "soft":
        raise ValueError("--body requires --pool soft")
    if not 0 <= body_drop <= 1:
        raise ValueError("--body-drop must be in [0, 1]")
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
    bodyset = None
    if body is not None:
        from . import body as bodylib

        signals = _body_signals(body)
        window = settings["window"]
        lut_has = lambda video_id, frame: region.frame_key(video_id, frame) in lut
        arrays = bodylib.pair_arrays(bodylib.load_body(generator["episodes"], manifest.video_id.unique()),
                                     manifest, lut_has, window)
        if body_control == "oracle":
            frames = pd.read_parquet(Path(generator["episodes"]) / "frames.parquet")
            arrays["oracle"] = bodylib.oracle_distances(manifest, frames, lut_has, window)
        bodyset = {k: torch.from_numpy(a) for k, a in arrays.items()}
        gate_generator = torch.Generator().manual_seed(seed + 1000)  # keeps the default RNG stream identical to video-only
        model = bodylib.BodyRegionMIL(len(vocab), signals, pool=pool, tau_p=tau_p, emb_dim=emb_dim)
    elif pool == "soft":
        from . import body as bodylib

        model = bodylib.BodyRegionMIL(len(vocab), None, pool=pool, tau_p=tau_p, emb_dim=emb_dim)
    else:
        model = upstream.RegionMIL(len(vocab), emb_dim=emb_dim)
    if device == "mps":
        model = model.float()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.1)
    probes = pd.read_parquet(run / "probes_frames.parquet")
    probe_cache, probe_lut = region.load_region_cache(run / "probes_cache")
    if probe_cache.shape[1:] != (17, emb_dim):
        raise ValueError("Probe and training cache dimensions disagree")
    if any(region.frame_key(r.video_id, r.frame_idx) not in probe_lut for r in probes.itertuples()):
        raise ValueError("Probe cache does not cover all probes")
    tables = {split: probes[probes.split == split] for split in ("dev", "test")}
    konkle = None
    if (run / "konkle_frames.parquet").exists():
        konkle_cache, konkle_lut = region.load_region_cache(run / "konkle_cache")
        konkle = (konkle_cache, konkle_lut, pd.read_parquet(run / "konkle_frames.parquet"))
    for split, table in tables.items():
        counts = table.groupby("category").size()
        if len(counts) < 4 or (counts < 4).any():
            raise ValueError(f"Upstream 4AFC requires at least four probes/category and four categories: {split}")
        if sum(bool(upstream.encode(cat, vocab)) for cat in counts.index) < 4:
            raise ValueError(f"Upstream 4AFC needs four in-vocabulary categories: {split}")
    output = _output_directory(run / f"{name or condition}_s{seed}")
    if any(output.iterdir()):
        raise FileExistsError(f"Training output already contains files: {output}")
    history, best_sel, best_state, best_ep, best_details = [], -1.0, None, -1, None
    for epoch in range(epochs):
        model.train()
        losses = []
        if bodyset is not None and body_control == "shuffle":
            others = torch.randperm(len(ds), generator=gate_generator)  # each pair gets another pair's body this epoch
        for idx, v, t, n in dl:
            if bodyset is None:
                loss = model.forward_loss(v.to(device), t.to(device), n.to(device), weights[idx].to(device))
            else:
                ids = others[idx] if body_control == "shuffle" else idx
                on, keep, noise = bodylib.drop_masks(len(idx), signals, body_drop, gate_generator)
                dist = bodyset["oracle" if body_control == "oracle" else "dist"][ids]
                loss = model.forward_loss(
                    v.to(device), t.to(device), n.to(device), weights[idx].to(device), dist.to(device),
                    {k: bodyset[k][ids].to(device) for k in signals}, on.to(device), keep.to(device), noise.to(device))
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
        if konkle is not None:
            # Frank's Konkle test-60 protocol: 100 trials/category, rng(0), macro mean, OOV at chance.
            scores["konkle"], details["konkle"] = region.eval_4afc_region(
                model, konkle[0], konkle[1], konkle[2], vocab, device, return_detail=True)
        if not all(np.isfinite(list(scores.values()))) or not np.isfinite(losses).all():
            raise ValueError("Non-finite training loss or probe score")
        if scores["dev"] > best_sel:
            best_sel, best_ep, best_details = scores["dev"], epoch, details
            best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        record = {"ep": epoch, "acc": round(scores["test"], 4), "dev": round(scores["dev"], 4),
                  "train_loss": float(np.mean(losses))}
        if konkle is not None:
            record["konkle"] = round(scores["konkle"], 4)
        if bodyset is not None:
            record["gate_mean"] = _gate_stats(model, bodyset, device, signals)["mean"]
        history.append(record)
        print(record, flush=True)
    model.load_state_dict(best_state)
    if bodyset is not None:
        body_info = {"signals": list(signals), "control": body_control, "drop": body_drop,
                     "gate": _gate_stats(model, bodyset, device, signals)}
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
        **({"reported_konkle_acc": round(100 * history[best_ep]["konkle"], 3),
            "best_konkle_acc": round(100 * max(h["konkle"] for h in history), 3)} if konkle is not None else {}),
        "pool": pool, **({"tau_p": tau_p} if pool == "soft" else {}), "condition": condition, "generator_parameters": config["generator"],
        "generator_seed": generator["seed"], "cue_parameters": config["cues"],
        "upstream": config["upstream"], "device": device, "n_manifest": int(len(kept)),
        "n_novocab": int((~kept).sum()), "frame_coverage": 1.0,
        "mean_weight": float(weights.mean()),
        "probe_counts": {s: {k: d[k] for k in ("n_scored", "n_oov", "n_total")} for s, d in details.items()},
    }
    if bodyset is not None:
        metrics["body"] = body_info
    metrics["groups"] = _groups(best_details)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"DONE {output} selected_ep {best_ep} reported_test_acc {metrics['reported_test_acc']}", flush=True)
    return metrics
