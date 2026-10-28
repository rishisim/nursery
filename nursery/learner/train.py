"""Train the upstream two-tower with pair weights on the available accelerator, optionally with the body losses."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from nursery.common import _output_directory
from nursery.learner import _upstream_module
from nursery.learner.losses import LOSSES

CONFIG = Path(__file__).with_name("config.json")
BODY_MODES = ("none", "real", "shuffled", "oracle")
EPISODE_MODES = ("contact", "temporal", "none")


def _study_center(ds):
    """Fit the visual mean on precisely the selected training observations."""
    total, count = None, 0
    for i in range(len(ds)):
        v = ds[i][1].numpy()
        value = v.sum(axis=0, dtype=np.float64)
        total = value if total is None else total + value
        count += len(v)
    return (total / count).astype(np.float32)


def _study_body_exclusion(manifest, frames, video_ids, half, steps=120):
    """Only seconds in selected, clip-local training histories fit body statistics."""
    exclude = np.ones((len(video_ids), steps), bool)
    row = {v: i for i, v in enumerate(video_ids)}
    clips = {(r.video_id, int(r.frame_idx)): r.clip for r in frames.itertuples()}
    for r in manifest.itertuples():
        centre = int(r.frame_idx)
        clip = clips[r.video_id, centre]
        exclude[row[r.video_id], centre] = False
        for direction in (-1, 1):
            for delta in range(1, half + 1):
                t = centre + direction * delta
                if not 0 <= t < steps or clips.get((r.video_id, t)) != clip:
                    break
                exclude[row[r.video_id], t] = False
    return exclude


def _study_evaluation(study, manifest, caches, lut, vocab, upstream, store, device, window, half):
    from . import evaluate
    from .windows import BodyData

    panels = json.loads((study["root"] / "panels.json").read_text())
    frames = pd.read_parquet(Path(study["episodes"]) / "frames.parquet")
    sets = {}
    for partition in ("dev", "test"):
        rows = manifest[manifest.partition == partition].reset_index(drop=True)
        ds = upstream.FramePairs(caches, lut, rows, vocab, window, False)
        if len(ds) != len(rows) or ds.n_novocab or ds.n_noframe:
            raise ValueError("Frozen study encounters must all survive FramePairs")
        context = evaluate.body_present_context(rows, frames, lut, window, half)
        data = None
        if store is not None:
            ps, pt, win = _pair_meta(rows, store, lut, window)
            data = BodyData(store, ps, pt, win, None, None, None, None, device, half, 0.0, 0.0)
            if set(data.dims) != {"contact", "fingers", "arm", "load", "skin"}:
                raise ValueError("Cross-situational body condition needs all five streams")
            if partition == "test":
                evaluate.ShuffledBodyHistory(data, panels["test_donors"], rows)
        # Validate frozen candidate identities and foil eligibility before training.
        evaluate._body_trials(rows, context, 0, None, panels[partition])
        sets[partition] = ds, data, rows, context
    return sets, panels


def _study_score(model, upstream, data, panel, device, trials=False):
    from .evaluate import body_present_4afc

    ds, body, rows, context = data
    return body_present_4afc(model, upstream, ds, body, rows, device, seed=0,
                             trials=trials, context=context, fixed_trials=panel)


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


def _window_layout(manifest, lut, window):
    """Per pair, in FramePairs order: seconds from the centre frame of each window frame that exists [N,Fmax] and a
    valid mask (False = the repeated padding FramePairs adds). Independent of the body store."""
    offset = np.zeros((len(manifest), 2 * window + 1), np.int64)
    valid = np.zeros(offset.shape, bool)
    for i, (v, f) in enumerate(zip(manifest.video_id, manifest.frame_idx.astype(int))):
        here = [g - f for g in range(f - window, f + window + 1) if f"{v}|{g}" in lut]
        offset[i, :len(here)], valid[i, :len(here)] = here, True
    return offset, valid


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
    from .windows import BodyData
    from .additive import AdditiveMIL

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
        model = AdditiveMIL(len(vocab), info["emb_dim"], tau_p, dims, len(store["categories"]))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model = model.float().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, data


def train_run(run, condition, seed, epochs=None, name=None, pool="max", tau_p=0.1, body="none", episodes="none",
              weights=None, heldout_subject=None, body_dropout=None, distill_from=None, distill_weight=1.0,
              distill_temp=2.0, inspect_epoch=None, labels=False, split="subject", arch="additive", physical=None, physical_shuffle=False, study_protocol=None, study_timing=False):
    import torch

    from . import body as bodylib
    from . import evaluate
    from .windows import BodyData
    from .additive import AdditiveMIL
    from .video_only import VideoOnlyMIL

    started = time.perf_counter()
    if study_timing and study_protocol is None:
        raise ValueError("Timing-only mode requires a cross-situational protocol")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    upstream = _upstream_module("train_frame_mil")
    region = _upstream_module("train_region_mil")
    run = Path(run).resolve()
    study = None
    if study_protocol is not None:
        from nursery.common import load_config
        protocol_path = Path(study_protocol).resolve()
        study = json.loads(protocol_path.read_text())
        study["protocol_path"] = str(protocol_path)
        study["root"] = protocol_path.parent
        study["hashes"] = {f"study_{key}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                           for key, path in (("protocol", protocol_path),
                                             ("manifest", protocol_path.parent / "manifest.parquet"),
                                             ("panels", protocol_path.parent / "panels.json"))}
        if arch != "additive" or body not in ("none", "real") or distill_from or labels or physical_shuffle or episodes != "none" or inspect_epoch:
            raise ValueError("Cross-situational study requires additive architecture, body none/real and no diagnostics")
        if seed not in (0, 1, 2) or epochs not in (None, 20):
            raise ValueError("Cross-situational study freezes seeds 0,1,2 and 20 epochs")
        if any(v is not None and v != 0 for v in (weights or {}).values()):
            raise ValueError("Cross-situational study has no auxiliary losses")
        if body_dropout not in (None, 0.5):
            raise ValueError("Cross-situational study freezes body dropout at 0.5")
        pool, tau_p, split, heldout_subject = "soft", 0.1, "subject", "none"
        weights = {k: 0.0 for k in LOSSES}
        body_dropout = 0.5
        physical = study["physical"] if body == "real" else None
        generator = {"config": load_config(), "episodes": study["episodes"], "seed": 0}
        generator["config"]["training"] = {"epochs": 20, "batch": 32, "lr": 0.0003, "window": 2, "min_freq": 1}
    else:
        generator = json.loads((run / "generator.json").read_text())
    config = generator["config"]
    settings = config["training"]
    learner = json.loads(CONFIG.read_text())["body"]
    if study:
        learner["window_s"] = 5
    epochs = 1 if study_timing else settings["epochs"] if epochs is None else epochs
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
    if arch not in ("additive", "joint"):
        raise ValueError("--arch must be additive or joint")
    joint = arch == "joint"
    if joint:  # Reproduction only; the proposed joint architecture is not implemented.
        from archive.code.learner.joint_pilot import JointMIL
    if physical and body != "real":
        raise ValueError("--physical needs --body real")
    if physical_shuffle and not physical:
        raise ValueError("--physical-shuffle needs --physical")
    if joint:
        if pool != "soft" or body not in ("none", "real") or distill_from or labels or episodes != "none":
            raise ValueError("--arch joint needs --pool soft, --body none or real, --episodes none, and no --distill-from or --labels")
        if any(v is not None and v > 0 for v in (weights or {}).values()):
            raise ValueError("--arch joint has no auxiliary losses: do not pass a positive --w-*")
    weights = {**learner["weights"], **{k: v for k, v in (weights or {}).items() if v is not None}}
    if set(weights) != set(LOSSES) or min(weights.values()) < 0:
        raise ValueError(f"Loss weights must be non-negative: {LOSSES}")
    if body == "none" and weights["bodymatch"] > 0:
        raise ValueError("--w-bodymatch needs --body")
    if body == "none" or joint:
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
    manifest_path = (study["root"] if study else run) / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    if study:
        if not {"train", "dev", "test"} == set(manifest.partition):
            raise ValueError("Cross-situational manifest needs train/dev/test partitions")
        study_manifest = manifest.copy()
        manifest = manifest[manifest.partition == "train"].reset_index(drop=True)
        manifest["weight_" + condition] = 1.0
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
    caches, lut = upstream.load_multi([Path(study["cache"]) if study else run / "train_cache"])
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
                                     **({"generator": torch.Generator().manual_seed(seed)} if study else {}),
                                     drop_last=False, collate_fn=upstream.collate, num_workers=0)
    window = settings["window"]
    body_info = None
    data = eval_data = held_ds = held_pairs = None
    if joint:  # window offsets and padding of every training pair, for control and body runs alike
        offset, valid = (torch.as_tensor(a, device=device) for a in _window_layout(manifest, lut, window))
    if body == "none":
        model = (JointMIL(len(vocab), emb_dim, tau_p) if joint else
                 VideoOnlyMIL(len(vocab), emb_dim, tau_p, pool=pool))
    else:
        frames = pd.read_parquet(episode_root / "frames.parquet")
        video_ids = sorted(set(study_manifest.video_id) if study else set(all_manifest.video_id) | set(heldout_manifest.video_id))
        heldout_videos = set(heldout_manifest.video_id)
        exclude = np.array([np.isin(part[v], (1, 2)) for v in video_ids]) if part else None  # dev/test clips stay out of the stats
        if study:
            exclude = _study_body_exclusion(manifest, frames, video_ids, learner["window_s"])
        real = bodylib.load_store(episode_root, frames, video_ids, heldout_videos, exclude=exclude, physical=physical)
        real["subject"] = np.array([subject_of[v] for v in video_ids])
        shuffle_info = physical_info = None
        train_store = real
        if physical_shuffle:  # training sees load/skin from another clip; evaluation keeps the real store
            same = np.array([part[v] for v in video_ids]) if part else np.zeros(real["clip"].shape, np.int64)
            train_store, physical_info = bodylib.physical_shuffle_store(real, same, seed)
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
        model = JointMIL(len(vocab), emb_dim, tau_p, data.dims) if joint else AdditiveMIL(len(vocab), emb_dim, tau_p, data.dims, oracle_classes)
        body_info = {"mode": body, "dropout": dropout, "episodes": episodes, "weights": weights, "shuffle": shuffle_info,
                     "physical": physical, "physical_shuffle": physical_info if physical_shuffle else False,
                     "n_episodes": int(len(np.unique(episode_ids[episode_ids >= 0]))),
                     "episode_frames": float((episode_ids >= 0).sum() / max(real["present"].sum(), 1))}
    if study:
        model.center_mu.copy_(torch.from_numpy(_study_center(ds)).to(model.center_mu))
    if device == "mps":
        model = model.float()
    model = model.to(device)
    teacher = None
    if distill_from:
        teacher, make_data = _load_teacher(distill_from, vocab, manifest, heldout_manifest, subject_of, episode_root, lut,
                                           window, learner, device, tau_p)
        teacher_train = make_data(manifest)
    opt = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=0.1)
    if study:
        output = _output_directory(run / "timing" / (name or condition) if study_timing else run / f"{name or condition}_s{seed}")
        if any(output.iterdir()):
            raise FileExistsError(f"Training output already contains files: {output}")
        study_sets, study_panels = _study_evaluation(study, study_manifest, caches, lut, vocab, upstream,
                                                   real if body != "none" else None, device, settings["window"], learner["window_s"])
        tables, konkle, bp_sets = {}, None, {}
    else:
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
        scoring_frames = pd.read_parquet(episode_root / "frames.parquet") if held_ds is not None or by_clips else None
        def scoring_context(rows, subject=None):
            return evaluate.body_present_context(rows, scoring_frames, lut, window, learner["window_s"], subject)
        held_context = scoring_context(test_rows) if held_ds is not None else None
        if held_context is not None and not any(len(p) >= 3 for p in held_context["pools"]):
            raise ValueError("Body-present heldout: insufficient valid foils for every trial")
        bp_contexts = {}
        bp_sets = {}
        if by_clips:  # body-present 4AFC sets: named, visible-referent pairs of each held-out partition (+ s8)
            def make_set(rows):
                rows = rows[rows.named.astype(bool) & rows.referent_visible.astype(bool)].reset_index(drop=True)
                if len(rows) < 8:
                    raise ValueError("Too few body-present trials in a split")
                ps, pt, win = _pair_meta(rows, real, lut, window) if body != "none" else (None,) * 3
                edata = BodyData(real, ps, pt, win, None, None, None, None, device, learner["window_s"], 0.0, 0.0) if body != "none" else None
                lay = tuple(torch.as_tensor(a, device=device) for a in _window_layout(rows, lut, window)) if joint else None
                return upstream.FramePairs(caches, lut, rows, vocab, window, False), edata, rows, rows.video_id.map(subject_of).to_numpy(), lay

            for k, n in ((1, "clips_dev"), (2, "clips_test")):
                bp_sets[n] = make_set(all_manifest[pair_part == k])
            if held_ds is not None:
                bp_sets["s8"] = make_set(test_rows)
            for n, (_, _, rows, subj, _) in bp_sets.items():
                bp_contexts[n] = scoring_context(rows, subj)
                if not any(len(p) >= 3 for p in bp_contexts[n]["pools"]):
                    raise ValueError(f"Body-present split {n}: insufficient valid foils for every trial (different session, same subject, target absent)")
            held_words = set(pd.read_parquet(episode_root / "frames.parquet", columns=["held_category"]).held_category.dropna())
            train_rows = manifest[manifest.named.astype(bool) & manifest.referent_visible.astype(bool)]
            words = {"train": set(train_rows.referent), **{n: set(v[2].referent) for n, v in bp_sets.items()}}
            split_info["trials"] = {n: {s: int((v[3] == s).sum()) for s in sorted(set(v[3]))} for n, v in bp_sets.items()}
            split_info["coverage"] = {n: {"n_words": len(w), "words": sorted(w), "held_missing": sorted(held_words - w)} for n, w in words.items()}

        def bp_run(trials=False):
            if joint:
                return {n: evaluate.joint_body_present_4afc(model, upstream, d, e, r, lay, device, seed, subject=subj, trials=trials, context=bp_contexts[n])
                        for n, (d, e, r, subj, lay) in bp_sets.items()}
            return {n: evaluate.body_present_4afc(model, upstream, d, e, r, device, seed, subject=subj, trials=trials, context=bp_contexts[n])
                    for n, (d, e, r, subj, _) in bp_sets.items()}

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
    loop_started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses, terms_sum = [], {}
        for idx, v, t, n in dl:
            w = pair_weights[idx].to(device)
            if soft:
                ctx = data.batch(idx.numpy(), mask_generator) if data is not None else None
                if joint:
                    ix = idx.to(device)
                    ctx = {**(ctx or {}), "vis_offset": offset[ix], "vis_valid": valid[ix]}
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
        if study:
            dev_result = _study_score(model, upstream, study_sets["dev"], study_panels["dev"], device)
            scores, details = {"dev": dev_result[bp_key]["acc"] / 100}, {}
        else:
            # Evaluation sees only image regions and word tokens; no body, cues or oracle labels.
            scores, details = {}, {}
            hard = {}
            for split, table in tables.items():
                if joint:  # soft score drives selection; upstream's hard max is logged for reference
                    scores[split], details[split] = evaluate.probe_4afc_soft(model, region, probe_cache, probe_lut, table, vocab, device)
                    hard[split] = region.eval_4afc_region(model, probe_cache, probe_lut, table, vocab, device)
                    continue
                scores[split], details[split] = region.eval_4afc_region(
                    model, probe_cache, probe_lut, table, vocab, device, return_detail=True,
                )
            if konkle is not None:
                # Frank's Konkle test-60 protocol: 100 trials/category, rng(0), macro mean, OOV at chance.
                if joint:
                    scores["konkle"], details["konkle"] = evaluate.probe_4afc_soft(model, region, konkle[0], konkle[1], konkle[2], vocab, device)
                    hard["konkle"] = region.eval_4afc_region(model, konkle[0], konkle[1], konkle[2], vocab, device)
                else:
                    scores["konkle"], details["konkle"] = region.eval_4afc_region(
                        model, konkle[0], konkle[1], konkle[2], vocab, device, return_detail=True)
        if not all(np.isfinite(list(scores.values()))) or not np.isfinite(losses).all():
            raise ValueError("Non-finite training loss or probe score")
        if scores["dev"] > best_sel:
            best_sel, best_ep, best_details = scores["dev"], epoch, details
            best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        record = {"ep": epoch, **({"acc": round(scores["test"], 4)} if not study else {}), "dev": round(scores["dev"], 4),
                  "train_loss": float(np.mean(losses))}
        if konkle is not None:
            record["konkle"] = round(scores["konkle"], 4)
        if joint:
            record.update({f"hard_{k}": round(x, 4) for k, x in hard.items()})
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
    if study_timing:
        timing = {"study_protocol": study["protocol_path"], **study["hashes"], "seed": seed, "condition": condition,
                  "body": body, "device": device, "epochs": 1, "full_epochs": settings["epochs"],
                  "startup_seconds": loop_started - started,
                  "train_dev_seconds": time.perf_counter() - loop_started,
                  "total_seconds": time.perf_counter() - started,
                  "n_pairs": len(ds), "history": history, "weights_discarded": True,
                  "test_evaluated": False}
        (output / "timing.json").write_text(json.dumps(timing, indent=2, allow_nan=False) + "\n")
        print(f"TIMING {output} {timing['train_dev_seconds']:.3f}s train+dev; weights discarded", flush=True)
        return timing
    model.load_state_dict(best_state)
    if study:
        test = _study_score(model, upstream, study_sets["test"], study_panels["test"], device, trials=True)
        shuffled = None
        if body != "none":
            from .evaluate import ShuffledBodyHistory
            d, e, rows, context = study_sets["test"]
            donors = study_panels["test_donors"]
            shuffled = _study_score(model, upstream, (d, ShuffledBodyHistory(e, donors, rows), rows, context),
                                    study_panels["test"], device, trials=True)
        metrics = {"study_protocol": study["protocol_path"], **study["hashes"], "seed": seed, "condition": condition,
                   "body": body, "selected_epoch": best_ep, "selection": "own_dev_situation_accuracy_earliest_tie",
                   "history": history, "test": test, "shuffled_test": shuffled, "n_pairs": len(ds),
                   "vocab": len(vocab), "emb_dim": int(model.word.embedding_dim), "settings": settings,
                   "pool": pool, "tau_p": tau_p, "body_dropout": dropout, "weights": weights,
                   "center": True, "body_stats": real["stats"] if body != "none" else None,
                   "evaluation_counts": {n: {"targets": len(study_panels[n]), "situations": len(study_sets[n][2])} for n in ("dev", "test")},
                   "reported_test_acc": test[bp_key]["acc"], "device": device}
        torch.save(model.state_dict(), output / "model.pt")
        (output / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")
        (output / "metrics.json").write_text(json.dumps(_finite(metrics), indent=2, allow_nan=False) + "\n")
        print(f"DONE {output} selected_ep {best_ep} reported_test_acc {metrics['reported_test_acc']}", flush=True)
        return metrics
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
        if held_ds is not None and joint:
            lay = tuple(torch.as_tensor(a, device=device) for a in _window_layout(test_rows, lut, window))
            metrics["body_present"] = evaluate.joint_body_present_4afc(model, upstream, held_ds, eval_data, test_rows, lay, device, seed, context=held_context)
        elif held_ds is not None:
            metrics["body_present"] = evaluate.body_present_4afc(model, upstream, held_ds, eval_data, test_rows, device, seed, context=held_context)
    if eval_data is not None and len(eval_data.ps) and not joint:
        metrics["probe"] = evaluate.body_probe(model, eval_data, real, heldout, device)
    if split_info is not None:
        metrics["body_present_at_picture_epoch"] = split_info.pop("body_present_at_picture_epoch")
        metrics["split"] = split_info
    if body_info is not None:
        metrics["body"] = body_info
    if joint:
        metrics["arch"], metrics["joint"] = arch, {**model.settings, "params": model.param_counts()}
        if epochs >= 30:
            metrics["readout_epoch_30"] = history[29]
    if teacher is not None:
        metrics["distill"] = {"from": str(Path(distill_from).resolve()), "weight": distill_weight, "temp": distill_temp}
        if held_ds is not None:  # ORACLE DIAGNOSTIC: the teacher's own body-present numbers (hard and soft pooling)
            metrics["distill"]["teacher_body_present"] = evaluate.body_present_4afc(teacher, upstream, held_ds, teacher_held, test_rows, device, seed, context=held_context)
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

