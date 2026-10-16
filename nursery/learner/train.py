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


def clip_partition(frames, heldout, cfg, steps=120):
    """Per-session partition of every second: 0 train, 1 dev, 2 test, -1 no frame. Within each subject (the held-out one
    stays all-train, unused) the sorted unique source clips are assigned at random by clip count; all replays of a
    clip share a partition. Seconds without a clip count as train. Also returns the clip counts per subject."""
    part = {}
    for v, g in frames.groupby("video_id"):
        part[v] = np.full(steps, -1, np.int64)
        part[v][g.frame_idx.to_numpy()] = 0
    rng = np.random.default_rng(cfg["seed"])
    counts = {}
    for subject, g in frames.groupby("subject"):
        if subject == heldout:
            continue
        g = g[g["clip"].map(lambda c: isinstance(c, str) and c != "")]
        clips = sorted(g["clip"].unique())
        n_dev, n_test = round(cfg["dev"] * len(clips)), round(cfg["test"] * len(clips))
        order = rng.permutation(len(clips))
        label = np.zeros(len(clips), np.int64)
        label[order[:n_dev]], label[order[n_dev:n_dev + n_test]] = 1, 2
        lab = dict(zip(clips, label))
        for v, gv in g.groupby("video_id"):
            part[v][gv.frame_idx.to_numpy()] = gv["clip"].map(lab).to_numpy()
        counts[subject] = {n: int((label == k).sum()) for k, n in enumerate(("train", "dev", "test"))}
    return part, counts


def split_pairs(manifest, part, window):
    """Partition of each pair = that of its centre frame; -1 if its image window (frames that exist) touches another one."""
    out = np.zeros(len(manifest), np.int64)
    for i, (v, f) in enumerate(zip(manifest.video_id, manifest.frame_idx.astype(int))):
        a = part[v]
        c = max(int(a[f]), 0)
        win = a[max(f - window, 0):f + window + 1]
        out[i] = c if (win[win >= 0] == c).all() else -1
    return out


def _load_teacher(path, vocab, manifest, heldout_manifest, subject_of, episode_root, lut, window, learner, device, tau_p):
    """Frozen oracle learner from a finished run directory, and its oracle-body data over `manifest`.
    Refuses a teacher whose vocabulary, pooling or body mode differs from this run's."""
    import torch

    from . import body as bodylib
    from .model import BodyData, LearnerMIL

    path = Path(path)
    info = json.loads((path / "metrics.json").read_text())
    if json.loads((path / "vocab.json").read_text()) != vocab:
        raise ValueError("The teacher's vocabulary differs from this run's")
    if info.get("pool") != "soft" or info.get("tau_p") != tau_p or info.get("body", {}).get("mode") != "oracle":
        raise ValueError("The teacher must be an oracle-body soft-pooling run with this run's tau_p")
    frames = pd.read_parquet(episode_root / "frames.parquet")
    video_ids = sorted(set(manifest.video_id) | set(heldout_manifest.video_id))
    store = bodylib.oracle_store(bodylib.load_store(episode_root, frames, video_ids, set(heldout_manifest.video_id)))
    store["subject"] = np.array([subject_of[v] for v in video_ids])

    def data(rows):
        ps, pt, win = _pair_meta(rows, store, lut, window)
        return BodyData(store, ps, pt, win, None, None, None, None, device, learner["window_s"], 0.0, 0.0)

    with torch.random.fork_rng(devices=[]):  # building the teacher must not move the student's RNG stream
        dims = {k: v.shape[-1] for k, v in store["groups"].items()}
        model = LearnerMIL(len(vocab), info["emb_dim"], tau_p, dims, len(store["categories"]))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model = model.float().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, data


def train_run(run, condition, seed, epochs=None, name=None, pool="max", tau_p=0.1, body="none", episodes="none",
              weights=None, heldout_subject=None, body_dropout=None, distill_from=None, distill_weight=1.0,
              distill_temp=2.0, inspect_epoch=None, labels=False, split="subject"):
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
    if distill_from and (body != "none" or pool != "soft"):
        raise ValueError("--distill-from needs a picture-only student: --body none --pool soft")
    if labels and (body != "none" or distill_from or pool != "soft"):
        raise ValueError("--labels is a picture-only soft-pooling diagnostic: use --pool soft, no --body or --distill-from")
    if split not in ("subject", "clips"):
        raise ValueError("--split must be subject or clips")
    if split == "clips" and (pool != "soft" or body in ("shuffled", "oracle") or distill_from or labels):
        raise ValueError("--split clips needs --pool soft, --body none or real, and no --distill-from or --labels")
    by_clips = split == "clips"  # `split` is reused as a loop name below
    if distill_weight < 0 or distill_temp <= 0:
        raise ValueError("--distill-weight must be non-negative and --distill-temp positive")
    weights = {**learner["weights"], **{k: v for k, v in (weights or {}).items() if v is not None}}
    if set(weights) != set(LOSSES) or min(weights.values()) < 0:
        raise ValueError(f"Loss weights must be non-negative: {LOSSES}")
    if body == "none" and weights["bodymatch"] > 0:
        raise ValueError("--w-bodymatch needs --body")
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
    vocab = upstream.build_vocab(manifest.text, settings["min_freq"])  # always from the speech, so labels share the vocabulary
    label_info = None
    if labels:  # LABEL DIAGNOSTIC: replace the speech by the word of one object visible in the frame
        manifest, label_info = evaluate.label_manifest(manifest, episode_root, seed)
    clip_cfg = learner["clip_split"]
    caches, lut = upstream.load_multi([run / "train_cache"])
    emb_dim = caches[0].shape[-1]
    if caches[0].ndim != 3 or caches[0].shape[1] != 17:
        raise ValueError("Expected CLS + 4x4 region caches")
    # FramePairs drops vocab-empty rows, so filter first to keep w aligned with its indices.
    kept = manifest.text.map(lambda text: bool(upstream.encode(text, vocab)))
    manifest = manifest[kept].reset_index(drop=True)
    if any(region.frame_key(r.video_id, r.frame_idx) not in lut for r in manifest.itertuples()):
        raise ValueError("Training cache does not cover the manifest")
    all_manifest = manifest  # every s1-s7 pair; the body store covers all of them
    split_info = part = None
    if by_clips:
        part, clip_counts = clip_partition(pd.read_parquet(episode_root / "frames.parquet"), heldout, clip_cfg)
        pair_part = split_pairs(manifest, part, settings["window"])
        subjects = manifest.video_id.map(subject_of).to_numpy()
        names = ("train", "dev", "test")
        split_info = {"mode": "clips", "dev": clip_cfg["dev"], "test": clip_cfg["test"], "seed": clip_cfg["seed"],
                      "clips": clip_counts, "n_dropped_crossing": int((pair_part < 0).sum()),
                      "pairs": {s: {n: int(((subjects == s) & (pair_part == k)).sum()) for k, n in enumerate(names)} for s in clip_counts}}
        manifest = manifest[pair_part == 0].reset_index(drop=True)
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
        video_ids = sorted(set(all_manifest.video_id) | set(heldout_manifest.video_id))
        heldout_videos = set(heldout_manifest.video_id)
        exclude = np.array([np.isin(part[v], (1, 2)) for v in video_ids]) if part else None  # dev/test clips stay out of the stats
        real = bodylib.load_store(episode_root, frames, video_ids, heldout_videos, exclude=exclude)
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
    teacher = None
    if distill_from:
        teacher, make_data = _load_teacher(distill_from, vocab, manifest, heldout_manifest, subject_of, episode_root, lut,
                                           window, learner, device, tau_p)
        teacher_train = make_data(manifest)
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
    # Held-out subject: named utterances with a visible referent.
    test_rows = None
    if pool == "soft" and heldout and len(heldout_manifest):
        test_rows = heldout_manifest[heldout_manifest.named.astype(bool) & heldout_manifest.referent_visible.astype(bool)]
        test_rows = test_rows[test_rows.text.map(lambda text: bool(upstream.encode(text, vocab)))].reset_index(drop=True)
        if len(test_rows) >= 8:
            held_ds = upstream.FramePairs(caches, lut, test_rows, vocab, window, False)
    teacher_held = make_data(test_rows) if teacher is not None and held_ds is not None else None
    bp_sets = {}
    if by_clips:  # body-present 4AFC sets: named, visible-referent pairs of each held-out partition (+ s8)
        def make_set(rows):
            rows = rows[rows.named.astype(bool) & rows.referent_visible.astype(bool)].reset_index(drop=True)
            if len(rows) < 8:
                raise ValueError("Too few body-present trials in a split")
            ps, pt, win = _pair_meta(rows, real, lut, window) if body != "none" else (None,) * 3
            edata = BodyData(real, ps, pt, win, None, None, None, None, device, learner["window_s"], 0.0, 0.0) if body != "none" else None
            return upstream.FramePairs(caches, lut, rows, vocab, window, False), edata, rows, rows.video_id.map(subject_of).to_numpy()

        for k, n in ((1, "clips_dev"), (2, "clips_test")):
            bp_sets[n] = make_set(all_manifest[pair_part == k])
        if held_ds is not None:
            bp_sets["s8"] = make_set(test_rows)
        held_words = set(pd.read_parquet(episode_root / "frames.parquet", columns=["held_category"]).held_category.dropna())
        train_rows = manifest[manifest.named.astype(bool) & manifest.referent_visible.astype(bool)]
        words = {"train": set(train_rows.referent), **{n: set(v[2].referent) for n, v in bp_sets.items()}}
        split_info["trials"] = {n: {s: int((v[3] == s).sum()) for s in sorted(set(v[3]))} for n, v in bp_sets.items()}
        split_info["coverage"] = {n: {"n_words": len(w), "words": sorted(w), "held_missing": sorted(held_words - w)} for n, w in words.items()}

    def bp_run(trials=False):
        return {n: evaluate.body_present_4afc(model, upstream, d, e, r, device, seed, subject=subj, trials=trials)
                for n, (d, e, r, subj) in bp_sets.items()}

    checkpoints = {}

    def checkpoint(ep, scores, details):
        """Inspection record of the model as it is now: probe scores, word-group accuracy and teacher agreement."""
        record = {"ep": ep, "dev": round(100 * scores["dev"], 3), "test": round(100 * scores["test"], 3),
                  "konkle": round(100 * scores["konkle"], 3) if "konkle" in scores else None, "groups": _groups(details)}
        if teacher is not None:
            record["agreement"] = {"train": evaluate.distill_agreement(model, teacher, upstream, ds, teacher_train, device, distill_temp, seed)}
            if teacher_held is not None:
                record["agreement"]["heldout"] = evaluate.distill_agreement(model, teacher, upstream, held_ds, teacher_held, device, distill_temp, seed)
        return record

    soft = pool == "soft"
    lw = {**weights, "distill": distill_weight}
    history, best_sel, best_state, best_ep, best_details = [], -1.0, None, -1, None
    best_bp, best_bp_state, best_bp_ep = -1.0, None, -1
    bp_key = "fused_soft" if body != "none" else "picture_soft"
    mask_generator = torch.Generator().manual_seed(seed + 1000)  # keeps the default RNG stream identical to video-only
    for epoch in range(epochs):
        model.train()
        losses, terms_sum = [], {}
        for idx, v, t, n in dl:
            w = pair_weights[idx].to(device)
            if soft:
                ctx = data.batch(idx.numpy(), mask_generator) if data is not None else None
                target = None
                if teacher is not None and distill_weight > 0:
                    target = teacher.teacher_logits(v.to(device), t.to(device), n.to(device), teacher_train.batch(idx.numpy(), None, training=False))
                terms = model.losses(v.to(device), t.to(device), n.to(device), w, ctx, lw, target, distill_temp)
                loss = terms["match"] + sum(lw[k] * terms[k] for k in terms if k != "match")
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
        if bp_sets:  # test curves are logged only; the body-present epoch is picked on clips_dev
            record["bp"] = {n: {k: r[k]["acc"] for k in ("picture_soft", "body", "fused_soft") if k in r} for n, r in bp_run().items()}
            if record["bp"]["clips_dev"][bp_key] > best_bp:
                best_bp, best_bp_ep = record["bp"]["clips_dev"][bp_key], epoch
                best_bp_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        if terms_sum:
            record["losses"] = {k: round(float(np.mean(x)), 5) for k, x in terms_sum.items()}
        history.append(record)
        print(record, flush=True)
        if inspect_epoch == epoch + 1:
            checkpoints[f"epoch_{inspect_epoch}"] = checkpoint(epoch, scores, details)
    model.load_state_dict(best_state)
    if bp_sets:
        split_info["body_present_at_picture_epoch"] = bp_run(trials=True)
    if teacher is not None or inspect_epoch:
        checkpoints["selected"] = checkpoint(best_ep, {k: history[best_ep][k2] for k, k2 in (("dev", "dev"), ("test", "acc"), ("konkle", "konkle")) if k2 in history[best_ep]}, best_details)
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
    if soft and heldout and len(heldout_manifest):
        if body != "none":
            ps, pt, win = _pair_meta(test_rows, real, lut, window)
            eval_data = BodyData(real, ps, pt, win, None, None, None, None, device, learner["window_s"], 0.0, 0.0)
        if held_ds is not None:
            metrics["body_present"] = evaluate.body_present_4afc(model, upstream, held_ds, eval_data, test_rows, device, seed)
    if eval_data is not None and len(eval_data.ps):
        metrics["probe"] = evaluate.body_probe(model, eval_data, real, heldout, device)
    if split_info is not None:
        metrics["body_present_at_picture_epoch"] = split_info.pop("body_present_at_picture_epoch")
        metrics["split"] = split_info
    if body_info is not None:
        metrics["body"] = body_info
    if teacher is not None:
        metrics["distill"] = {"from": str(Path(distill_from).resolve()), "weight": distill_weight, "temp": distill_temp}
        if held_ds is not None:  # ORACLE DIAGNOSTIC: the teacher's own body-present numbers (hard and soft pooling)
            metrics["distill"]["teacher_body_present"] = evaluate.body_present_4afc(teacher, upstream, held_ds, teacher_held, test_rows, device, seed)
    if label_info is not None:  # fit of the selected model on its own training labels
        metrics["labels"] = {**label_info, "fit": evaluate.label_fit(model, upstream, ds, manifest, vocab, device, seed)}
    if checkpoints:
        metrics["checkpoints"] = checkpoints
    metrics["groups"] = _groups(best_details)
    torch.save(model.state_dict(), output / "model.pt")
    if bp_sets:  # the body-present-selected state is evaluated last and saved beside the picture-selected one
        model.load_state_dict(best_bp_state)
        metrics["selection_body_present"] = {"epoch": best_bp_ep, "dev_score": best_bp, "rule": f"clips_dev {bp_key}"}
        metrics["body_present_selected"] = bp_run(trials=True)
        torch.save(model.state_dict(), output / "model_bodypresent.pt")
    (output / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
    text = json.dumps(_finite(metrics), indent=2, allow_nan=False)
    (output / "metrics.json").write_text(text + "\n")
    print(f"DONE {output} selected_ep {best_ep} reported_test_acc {metrics['reported_test_acc']}", flush=True)
    return metrics

