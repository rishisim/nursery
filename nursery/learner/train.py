"""Train the upstream two-tower with pair weights on the available accelerator, optionally with the body losses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nursery.common import _output_directory
from nursery.learner import _upstream_module
from nursery.learner.model import LOSSES

CONFIG = Path(__file__).with_name("config.json")
BODY_MODES = ("none", "real", "shuffled", "oracle")
EPISODE_MODES = ("contact", "temporal", "none")


def _groups(details):
    """Accuracy at the selected epoch by word group: held (touched) vs seen-only vs GRAB-only extras."""
    from nursery.simulator.assets import load_vocabulary

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


def _finite(value):
    """Probe metrics on tiny splits can be NaN; store them as null."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def _pair_meta(manifest, store, lut, window):
    """Per kept pair: session row, centre frame, and the frames of its image window (-1 pad), as FramePairs builds them."""
    ps = np.array([store["row"][v] for v in manifest.video_id])
    pt = manifest.frame_idx.to_numpy().astype(int)
    frames = np.full((len(manifest), 2 * window + 1), -1, np.int64)
    for i, (v, f) in enumerate(zip(manifest.video_id, pt)):
        here = [g for g in range(f - window, f + window + 1) if f"{v}|{g}" in lut]
        frames[i, :len(here)] = here
    return ps, pt, frames


def train_run(run, condition, seed, epochs=None, name=None, pool="max", tau_p=0.1, body="none", episodes="none",
              weights=None, heldout_subject=None, body_dropout=None):
    import torch

    from . import body as bodylib
    from . import evaluate
    from .model import BodyData, LearnerMIL

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    upstream = _upstream_module("train_frame_mil")
    region = _upstream_module("train_region_mil")
    run = Path(run).resolve()
    generator = json.loads((run / "generator.json").read_text())
    config = generator["config"]
    settings = config["training"]
    learner = json.loads(CONFIG.read_text())["body"]
    epochs = settings["epochs"] if epochs is None else epochs
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if pool not in ("max", "soft"):
        raise ValueError("--pool must be max or soft")
    if body not in BODY_MODES:
        raise ValueError(f"--body must be one of {BODY_MODES}")
    if episodes not in EPISODE_MODES:
        raise ValueError(f"--episodes must be one of {EPISODE_MODES}")
    if body != "none" and pool != "soft":
        raise ValueError("--body requires --pool soft")
    weights = {**learner["weights"], **{k: v for k, v in (weights or {}).items() if v is not None}}
    if set(weights) != set(LOSSES) or min(weights.values()) < 0:
        raise ValueError(f"Loss weights must be non-negative: {LOSSES}")
    if body == "none":
        weights = {k: 0.0 for k in LOSSES}
    if episodes == "none":
        weights["episode"] = weights["consist"] = 0.0
    dropout = learner["dropout"] if body_dropout is None else body_dropout
    if not 0 <= dropout <= 1:
        raise ValueError("--body-dropout must be in [0, 1]")
    heldout = learner["heldout_subject"] if heldout_subject is None else heldout_subject
    heldout = None if heldout == "none" else heldout
    torch.manual_seed(seed)
    np.random.seed(seed)
    manifest_path = run / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    column = f"weight_{condition}"
    if column not in manifest:
        raise ValueError(f"Unknown cue condition: {condition}")
    episode_root = Path(generator["episodes"])
    sessions = pd.read_parquet(episode_root / "sessions.parquet")
    subject_of = dict(zip(sessions.video_id, sessions.subject))
    is_heldout = manifest.video_id.map(subject_of).eq(heldout).to_numpy() if heldout else np.zeros(len(manifest), bool)
    heldout_manifest = manifest[is_heldout].reset_index(drop=True)
    manifest = manifest[~is_heldout]
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
    weight_values = manifest[column].to_numpy(dtype=np.float32)
    if not np.isfinite(weight_values).all() or (weight_values <= 0).any():
        raise ValueError("Pair weights must be finite and positive")
    weight_values /= weight_values.mean()
    pair_weights = torch.from_numpy(weight_values)
    dl = torch.utils.data.DataLoader(ds, batch_size=settings["batch"], shuffle=True,
                                     drop_last=False, collate_fn=upstream.collate, num_workers=0)
    window = settings["window"]
    body_info = None
    data = eval_data = held_ds = held_pairs = None
    if body == "none":
        model = LearnerMIL(len(vocab), emb_dim, tau_p) if pool == "soft" else upstream.RegionMIL(len(vocab), emb_dim=emb_dim)
    else:
        frames = pd.read_parquet(episode_root / "frames.parquet")
        video_ids = sorted(set(manifest.video_id) | set(heldout_manifest.video_id))
        heldout_videos = set(heldout_manifest.video_id)
        real = bodylib.load_store(episode_root, frames, video_ids, heldout_videos)
        real["subject"] = np.array([subject_of[v] for v in video_ids])
        shuffle_info = None
        train_store = real
        if body == "shuffled":
            train_store, shuffle_info = bodylib.shuffle_store(real, seed, heldout_videos)
        elif body == "oracle":
            train_store = real = bodylib.oracle_store(real)
        episode_ids = bodylib.build_episodes(train_store, episodes, learner["min_episode_s"], seed)
        bags = bodylib.bag_frames(episode_ids, learner["bag_frames"])
        row_of = np.full(real["clip"].shape, -1, np.int64)
        for key, (_, r) in lut.items():
            v, f = key.rsplit("|", 1)
            if v in real["row"] and int(f) < row_of.shape[1]:
                row_of[real["row"][v], int(f)] = r
        need_images = episodes != "none" and (weights["episode"] > 0 or weights["consist"] > 0)
        cache = np.ascontiguousarray(caches[0]) if need_images else None
        ps, pt, win = _pair_meta(manifest, train_store, lut, window)
        data = BodyData(train_store, ps, pt, win, bags, episode_ids, cache, row_of, device, learner["window_s"],
                        learner["mask_prob"] if weights["mask"] > 0 else 0.0, dropout)
        oracle_classes = len(real["categories"]) if body == "oracle" else 0
        model = LearnerMIL(len(vocab), emb_dim, tau_p, data.dims, oracle_classes)
        body_info = {"mode": body, "dropout": dropout, "episodes": episodes, "weights": weights, "shuffle": shuffle_info,
                     "n_episodes": int(len(np.unique(episode_ids[episode_ids >= 0]))),
                     "episode_frames": float((episode_ids >= 0).sum() / max(real["present"].sum(), 1))}
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
    soft = pool == "soft"
    history, best_sel, best_state, best_ep, best_details = [], -1.0, None, -1, None
    mask_generator = torch.Generator().manual_seed(seed + 1000)  # keeps the default RNG stream identical to video-only
    for epoch in range(epochs):
        model.train()
        losses, terms_sum = [], {}
        for idx, v, t, n in dl:
            w = pair_weights[idx].to(device)
            if soft:
                ctx = data.batch(idx.numpy(), mask_generator) if data is not None else None
                terms = model.losses(v.to(device), t.to(device), n.to(device), w, ctx, weights)
                loss = terms["match"] + sum(weights[k] * terms[k] for k in terms if k != "match")
                for k, x in terms.items():
                    terms_sum.setdefault(k, []).append(float(x.detach()))
            else:
                loss = model.forward_loss(v.to(device), t.to(device), n.to(device), w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        # Evaluation sees only image regions and word tokens; no body, cues or oracle labels.
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
        if terms_sum:
            record["losses"] = {k: round(float(np.mean(x)), 5) for k, x in terms_sum.items()}
        history.append(record)
        print(record, flush=True)
    model.load_state_dict(best_state)
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
        "mean_weight": float(weight_values.mean()),
        "probe_counts": {s: {k: d[k] for k in ("n_scored", "n_oov", "n_total")} for s, d in details.items()},
        "heldout_subject": heldout, "n_heldout_pairs": int(is_heldout.sum()),
    }
    if "losses" in history[best_ep]:
        metrics["losses_selected_epoch"], metrics["losses_final_epoch"] = history[best_ep]["losses"], history[-1]["losses"]
    # Held-out subject: named utterances with a visible referent (picture-only for every condition).
    test_rows = None
    if soft and heldout and len(heldout_manifest):
        test_rows = heldout_manifest[heldout_manifest.named.astype(bool) & heldout_manifest.referent_visible.astype(bool)]
        test_rows = test_rows[test_rows.text.map(lambda text: bool(upstream.encode(text, vocab)))].reset_index(drop=True)
        if body != "none":
            ps, pt, win = _pair_meta(test_rows, real, lut, window)
            eval_data = BodyData(real, ps, pt, win, None, None, None, None, device, learner["window_s"], 0.0, 0.0)
        if len(test_rows) >= 8:
            held_ds = upstream.FramePairs(caches, lut, test_rows, vocab, window, False)
            metrics["body_present"] = evaluate.body_present_4afc(model, upstream, held_ds, eval_data, test_rows, device, seed)
    if eval_data is not None and len(eval_data.ps):
        metrics["probe"] = evaluate.body_probe(model, eval_data, real, heldout, device)
    if body_info is not None:
        metrics["body"] = body_info
    metrics["groups"] = _groups(best_details)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
    text = json.dumps(_finite(metrics), indent=2, allow_nan=False)
    (output / "metrics.json").write_text(text + "\n")
    print(f"DONE {output} selected_ep {best_ep} reported_test_acc {metrics['reported_test_acc']}", flush=True)
    return metrics

