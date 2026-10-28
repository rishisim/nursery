"""Bounded heavy/light word-grounding diagnostic over existing analytic load signals.

This deliberately aligned two-word task is separate from ordinary weak noun
learning. Sampled mass does not affect the recorded motion or rendered video.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from nursery.common import REPO, _output_directory
from . import _upstream_module
from .body import clip_index
from .windows import BodyData
from .additive import AdditiveMIL as LearnerMIL
from .interaction import JointInteractionMIL
from .losses import weighted_nce
from .train import _pair_meta


WORDS = ("heavy", "light")  # Also the score-column order; ties receive half credit.
VOCAB = {"<pad>": 0, "heavy": 1, "light": 2}
CATEGORIES = {
    "train": ["doll", "rock", "tube", "bottle", "trumpet", "bowl"],
    "dev": ["clock", "train"],
    "test": ["pyramid", "stapler"],
}
SETTINGS = {"seeds": [0, 1, 2], "epochs": 100, "batch": 16, "lr": 0.0003,
            "weight_decay": 0.1, "tau_p": 0.1, "dim": 512, "body_dropout": 0.5,
            "visual_half_window": 2, "body_half_window": 5, "budget_s": 1800,
            "bootstrap_draws": 10000, "bootstrap_seed": 0, "shuffle_draws": 10,
            "shuffle_seed": 0, "light_max_kg": 0.15, "heavy_min_kg": 0.3,
            "visible_min_px": 100}
GATE = {"body_accuracy_min": 0.75, "gain_min": 0.15,
        "each_seed_gain_positive": True, "paired_ci_lower_positive": True,
        "shuffle_drop_min": 0.10}
LIMITS = [
    "Post-hoc analytic wrist load m(g-a), not a contact-force solve; sampled mass does not affect motion or RGB.",
    "Per-clip mass is not a persistent visual-asset property. Labels describe the existing synthetic draw.",
    "Explicit aligned property-word utterances, not ordinary weak speech or broad language acquisition.",
    "Novelty applies to diagnostic training exposure; the frozen DINO encoder has prior visual knowledge.",
    "Test has only two native GRAB categories; intervals condition on those categories and three fitted seeds.",
    "Full-frame visibility exclusion does not prove center-crop recognizability of the positive target.",
]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _revision():
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def select_manifest(frames, physical, categories=CATEGORIES, settings=SETTINGS):
    """One admissible center/source clip, then equal heavy/light clip counts within each category.

    Visibility checks apply to every RGB frame actually presented. Label-dependent
    balancing is intentional in this scaffolded diagnostic, not ordinary speech.
    """
    category_split = {cat: split for split, cats in categories.items() for cat in cats}
    if len(category_split) != sum(map(len, categories.values())):
        raise ValueError("Property categories must be disjoint")
    rows = []
    visual_half = settings["visual_half_window"]
    for video, group in frames.groupby("video_id", sort=True):
        group = group.sort_values("frame_idx").set_index("frame_idx")
        objects = {int(t): json.loads(value) for t, value in group.objects.items()}
        with np.load(Path(physical) / f"{video}.npz", allow_pickle=False) as signal:
            index = {int(t): i for i, t in enumerate(signal["frame_idx"])}
            load, mass = signal["load"], signal["mass_kg"]
            if load.shape != (len(index), 2, 5) or not np.isfinite(load).all() or not np.isfinite(mass).all():
                raise ValueError(f"Invalid physical arrays: {video}")
            for t, row in group.iterrows():
                category = row.held_category
                if category not in category_split or not (row.contact_left or row.contact_right):
                    continue
                seconds = list(range(int(t) - visual_half, int(t) + visual_half + 1))
                if any(s not in group.index for s in seconds):
                    continue
                if not group.loc[seconds, "clip"].eq(row["clip"]).all():
                    continue
                visible = max((obj["visible_px"] for obj in objects[t] if obj["category"] == category), default=0)
                if visible < settings["visible_min_px"] or t not in index or load[index[t], :, 3].sum() <= 0:
                    continue
                kg = float(mass[index[t]])
                word = ("light" if 0 < kg <= settings["light_max_kg"] else
                        "heavy" if kg >= settings["heavy_min_kg"] else None)
                if word is None:
                    continue
                split = category_split[category]
                forbidden = (set(categories["dev"] + categories["test"]) if split == "train" else
                             set(categories["test"]) if split == "dev" else set())
                if any(obj["category"] in forbidden and obj["visible_px"] > 0
                       for second in seconds for obj in objects[second]):
                    continue
                rows.append({"video_id": video, "frame_idx": int(t), "clip": row["clip"],
                             "category": category, "split": split, "text": word, "mass_kg": kg,
                             "target_visible_px": int(visible), "subject": row.subject})
    if not rows:
        raise ValueError("No eligible property-word experiences")
    unique = pd.DataFrame(rows).sort_values(["clip", "video_id", "frame_idx"]).drop_duplicates("clip")
    balanced = []
    for category in category_split:
        group = unique[unique.category == category]
        count = min(int(group.text.eq(word).sum()) for word in WORDS)
        if count < 2:
            raise ValueError(f"Need at least two independent clips per word: {category}")
        balanced.extend(group[group.text == word].head(count) for word in WORDS)
    manifest = pd.concat(balanced).sort_values(["split", "category", "text", "clip"]).reset_index(drop=True)
    manifest["example_id"] = np.arange(len(manifest))
    audit = {"candidate_centers": len(rows), "unique_source_clips": len(unique),
             "eligible_counts": unique.groupby("split").size().to_dict(),
             "balanced_counts": manifest.groupby("split").size().to_dict(),
             "per_category_word": {f"{s}/{c}/{w}": int(n) for (s, c, w), n in
                                   manifest.groupby(["split", "category", "text"]).size().items()},
             "rule": "First eligible center in clip/video/frame lexicographic order; first n clips per word/category, n=min(word counts)."}
    return manifest, audit


def load_store(frames, manifest, physical, half=5):
    """Load-only BodyData store; normalization fits UNIQUE accessible training seconds.

    Clip identifiers are used solely for recurrent resets. No category, mass,
    word, or asset identifier enters a model input group.
    """
    videos = sorted(manifest.video_id.unique())
    subset = frames[frames.video_id.isin(videos)]
    steps = int(subset.frame_idx.max()) + 1
    clips, _, _, _ = clip_index(subset, videos, steps)
    lookup = {video: i for i, video in enumerate(videos)}
    raw = np.zeros((len(videos), steps, 10), np.float32)
    available = np.zeros(clips.shape, bool)
    for video in videos:
        with np.load(Path(physical) / f"{video}.npz", allow_pickle=False) as signals:
            idx = signals["frame_idx"].astype(int)
            raw[lookup[video], idx] = signals["load"].reshape(len(idx), 10)
            available[lookup[video], idx] = True
    accessible = np.zeros(clips.shape, bool)
    fit = np.zeros(clips.shape, bool)
    for row in manifest.itertuples():
        s, t = lookup[row.video_id], int(row.frame_idx)
        win = np.arange(max(0, t - half), min(steps, t + half + 1))
        win = win[(clips[s, win] == clips[s, t]) & available[s, win] & (clips[s, win] >= 0)]
        accessible[s, win] = True
        if row.split == "train":
            fit[s, win] = True
    if not fit.any() or not np.isfinite(raw[accessible]).all():
        raise ValueError("Invalid training normalization support")
    mean, std = raw[fit].mean(0), raw[fit].std(0) + 1e-6
    normalized = ((raw - mean) / std * accessible[..., None]).astype(np.float32)
    store = {"groups": {"load": normalized}, "targets": {}, "clip": clips,
             "present": accessible, "next_valid": np.zeros_like(accessible),
             "video_ids": videos, "row": lookup}
    seconds = pd.DataFrame([(videos[s], int(t)) for s, t in np.argwhere(fit)], columns=["video_id", "frame_idx"])
    stats = {"mean": mean.tolist(), "std": std.tolist(), "unique_training_seconds": len(seconds)}
    return store, stats, seconds


def donor_derangements(rows, draws=10, seed=0):
    """Whole coherent donor windows, permuted within test category without consulting labels."""
    rng = np.random.default_rng(seed)
    donors = []
    for _ in range(draws):
        donor = np.arange(len(rows))
        for _, group in rows.groupby("category", sort=True):
            ids = group.index.to_numpy()
            if len(ids) < 2:
                raise ValueError("Derangements require two examples per category")
            for attempt in range(10000):
                permutation = rng.permutation(ids)
                if (permutation != ids).all():
                    donor[ids] = permutation
                    break
            else:
                raise ValueError("Could not draw a donor derangement")
        donors.append(donor.tolist())
    return donors


def _code_hashes():
    return {str(path.relative_to(REPO)): _sha(path) for path in
            [Path(__file__), Path(__file__).with_name("__main__.py"),
             Path(__file__).with_name("additive.py"), Path(__file__).with_name("interaction.py"),
             Path(__file__).with_name("video_only.py"), Path(__file__).with_name("windows.py"), Path(__file__).with_name("losses.py"), Path(__file__).with_name("body.py"),
             Path(__file__).with_name("train.py"), Path(__file__).with_name("config.json")]}


def prepare(output, episodes=None, cache=None, physical=None):
    """Freeze the manifest, normalization, donor controls, protocol and input hashes; never train."""
    output = _output_directory(output)
    if any(output.iterdir()):
        raise FileExistsError(f"Property preparation output must be empty: {output}")
    episodes = Path(episodes or REPO / "outputs/renders/clean_856").resolve()
    cache = Path(cache or REPO / "outputs/runs/clean_body/train_cache").resolve()
    physical = Path(physical or episodes / "physical_by_clip").resolve()
    frames = pd.read_parquet(episodes / "frames.parquet")
    manifest, audit = select_manifest(frames, physical)
    index = pd.read_parquet(cache / "index.parquet")
    keys = set(zip(index.video_id, index.frame_idx))
    if any((row.video_id, t) not in keys for row in manifest.itertuples()
           for t in range(row.frame_idx - SETTINGS["visual_half_window"], row.frame_idx + SETTINGS["visual_half_window"] + 1)):
        raise ValueError("Visual cache does not cover every selected window")
    embeddings = np.load(cache / "emb.f16.npy", mmap_mode="r")
    if embeddings.ndim != 3 or embeddings.shape[:2] != (len(index), 17):
        raise ValueError("Expected one CLS plus 4x4 visual grid per cached frame")
    _, stats, seconds = load_store(frames, manifest, physical, SETTINGS["body_half_window"])
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    paths = [episodes / "frames.parquet", cache / "index.parquet", cache / "emb.f16.npy"]
    # Every session participates in eligibility inventory, including unselected ones.
    paths.extend(physical / f"{video}.npz" for video in sorted(frames.video_id.unique()))
    protocol = {"task": "scaffolded_synthetic_heavy_light_word_grounding", "settings": SETTINGS,
                "categories": CATEGORIES, "vocab": VOCAB, "word_score_columns": WORDS, "gate": GATE,
                "limits": LIMITS, "episodes": str(episodes), "cache": str(cache), "physical": str(physical),
                "prepared_code_revision": _revision(), "code_sha256": _code_hashes(),
                "source_sha256": {str(path): _sha(path) for path in paths},
                "normalization": stats, "audit": audit,
                "donor_indices": donor_derangements(test, SETTINGS["shuffle_draws"], SETTINGS["shuffle_seed"]),
                "selection": "First maximum dev word accuracy, with half credit for exact ties; no test epoch selection.",
                "body_inputs": "Only normalized 10-dimensional wrist load, missing mask and clip-reset boundaries.",
                "exposure": "One five-frame RGB window and clip-local eleven-second body window per source clip; one aligned property word.",
                "visual_centering": "None (existing zero center_mu). Frozen DINO features; no encoder fitting.",
                "budget": "All nine models share one 1800-second wall-clock deadline. Incomplete models abort the comparison; no automatic retry."}
    manifest.to_parquet(output / "manifest.parquet", index=False)
    seconds.to_parquet(output / "normalization_seconds.parquet", index=False)
    protocol["manifest_sha256"] = _sha(output / "manifest.parquet")
    protocol["normalization_seconds_sha256"] = _sha(output / "normalization_seconds.parquet")
    _json(output / "protocol.json", protocol)
    return protocol


def _verify_frozen(output, protocol):
    if protocol["settings"] != SETTINGS or protocol["categories"] != CATEGORIES or protocol["vocab"] != VOCAB or protocol["gate"] != GATE:
        raise ValueError("Frozen diagnostic settings differ from the canonical protocol")
    if _code_hashes() != protocol["code_sha256"]:
        raise ValueError("Diagnostic source changed after preparation; review and prepare a new empty run root")
    for path, expected in protocol["source_sha256"].items():
        if _sha(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for filename, field in (("manifest.parquet", "manifest_sha256"),
                            ("normalization_seconds.parquet", "normalization_seconds_sha256")):
        if _sha(output / filename) != protocol[field]:
            raise ValueError(f"Frozen prepared artifact changed: {filename}")


def _body_data(rows, store, lut, device, settings):
    session, center, win = _pair_meta(rows, store, lut, settings["visual_half_window"])
    return BodyData(store, session, center, win, None, None, None, None, device,
                    settings["body_half_window"], 0.0, settings["body_dropout"])


def _common_hash(model):
    body_prefix = ("encoder.", "region_in.", "region_head.", "decoder.", "fuse.")
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key != "beta" and not key.startswith(body_prefix + ("interaction_",)):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def word_credit(scores, texts):
    """Binary word decisions; exact ties get 0.5, independent of candidate order."""
    labels = np.array([WORDS.index(text) for text in texts])
    scores = np.asarray(scores)
    if scores.shape != (len(labels), 2) or not np.isfinite(scores).all():
        raise ValueError("Expected finite B-by-2 word scores")
    correct = scores[np.arange(len(labels)), labels]
    other = scores[np.arange(len(labels)), 1 - labels]
    return (correct > other).astype(float) + 0.5 * (correct == other)


@torch.no_grad()
def score_words(model, dataset, data, device, upstream, batch=16, donors=None):
    """Score both learned words. Donors replace the complete coherent body context."""
    model.eval()
    candidates = torch.tensor([[VOCAB[word]] for word in WORDS], device=device)
    lengths = torch.ones(2, dtype=torch.long, device=device)
    scores = []
    for start in range(0, len(dataset), batch):
        ids = np.arange(start, min(start + batch, len(dataset)))
        _, visual, _, _ = upstream.collate([dataset[int(i)] for i in ids])
        ctx = data.batch(ids if donors is None else np.asarray(donors)[ids], None, training=False) if data is not None else None
        scores.append(model.pair_scores(visual.to(device), candidates, lengths, ctx)[2].cpu().numpy())
    return np.concatenate(scores)


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("The fixed nine-model training budget expired; comparison is incomplete")


def _train_arm(seed, body, datasets, rows, data, emb_dim, device, upstream, settings, output, deadline, arch="additive", timing=False):
    """One fixed run; common initialization and actual minibatch orders are fingerprinted."""
    _deadline(deadline)
    if not timing:
        output.mkdir()
    torch.manual_seed(seed)
    np.random.seed(seed)
    factory = JointInteractionMIL if arch == "joint" else LearnerMIL
    model = factory(len(VOCAB), emb_dim, settings["tau_p"], {"load": 10} if body else None).float().to(device)
    if model.word.embedding_dim != settings["dim"]:
        raise ValueError("The existing learner no longer matches the frozen 512-dimensional representation")
    common_hash = _common_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    order_rng = torch.Generator().manual_seed(seed)
    body_rng = torch.Generator().manual_seed(seed + 1000)
    order_hash = hashlib.sha256()
    best, best_epoch, state, history = -1.0, -1, None, []
    for epoch in range(settings["epochs"]):
        model.train()
        order = torch.randperm(len(datasets["train"]), generator=order_rng).numpy()
        order_hash.update(order.tobytes())
        losses = []
        for start in range(0, len(order), settings["batch"]):
            _deadline(deadline)
            ids = order[start:start + settings["batch"]]
            _, visual, tokens, lengths = upstream.collate([datasets["train"][int(i)] for i in ids])
            ctx = data["train"].batch(ids, body_rng) if body else None
            _, _, scores = model.pair_scores(visual.to(device), tokens.to(device), lengths.to(device), ctx)
            # Repeated words remain in the canonical square symmetric matching loss.
            loss = weighted_nce(model.scale() * scores, torch.ones(len(ids), device=device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite diagnostic language loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scores = score_words(model, datasets["dev"], data["dev"] if body else None, device, upstream, settings["batch"])
        dev = float(word_credit(scores, rows["dev"].text).mean())
        _deadline(deadline)
        history.append({"epoch": epoch, "dev_accuracy": dev, "language_loss": float(np.mean(losses))})
        if dev > best:  # First maximum, identical for both arms.
            best, best_epoch = dev, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if not timing:
                torch.save(state, output / "model.pt")
        if not timing:
            _json(output / "history.json", history)
    model.load_state_dict(state)
    return model, {"seed": seed, "body": body, "arch": arch, "selected_epoch": best_epoch,
                   "dev_accuracy": best, "epochs_completed": len(history),
                   "common_initialization_sha256": common_hash, "minibatch_order_sha256": order_hash.hexdigest()}


def paired_summary(video, body, shuffled, settings=SETTINGS, gate=GATE):
    """Pair by source clip, average the fixed three seeds, then bootstrap clips."""
    video, body, shuffled = map(np.asarray, (video, body, shuffled))
    gain = body - video
    rng = np.random.default_rng(settings["bootstrap_seed"])
    idx = rng.integers(0, gain.shape[1], (settings["bootstrap_draws"], gain.shape[1]))
    interval = np.percentile(gain.mean(0)[idx].mean(1), [2.5, 97.5])
    seed_gains = gain.mean(1)
    shuffle_drop = float(body.mean() - shuffled.mean())
    criteria = {"body_accuracy": bool(body.mean() >= gate["body_accuracy_min"]),
                "gain": bool(gain.mean() >= gate["gain_min"]),
                "every_seed_positive": bool((seed_gains > 0).all()),
                "paired_ci_positive": bool(interval[0] > 0),
                "shuffled_load_drop": bool(shuffle_drop >= gate["shuffle_drop_min"])}
    return {"video_accuracy": float(video.mean()), "body_accuracy": float(body.mean()),
            "gain": float(gain.mean()), "seed_gains": seed_gains.tolist(),
            "paired_clip_ci95": interval.tolist(), "shuffled_accuracy": float(shuffled.mean()),
            "shuffle_drop": shuffle_drop, "criteria": criteria, "capability_pass": all(criteria.values()),
            "interval_scope": "10000 paired source-clip draws after averaging three fixed seeds; conditional on models and two test categories."}


def train_prepared(output):
    """Run exactly the frozen nine models once, preserving every completed/partial output."""
    output = Path(output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    expected = {"protocol.json", "manifest.parquet", "normalization_seconds.parquet"}
    if {path.name for path in output.iterdir()} != expected:
        raise FileExistsError("Training requires exactly the prepared artifacts; existing runs are never overwritten or retried")
    _output_directory(output)  # Recheck the ignored-root guard before any writes.
    _verify_frozen(output, protocol)
    manifest = pd.read_parquet(output / "manifest.parquet")
    frames = pd.read_parquet(Path(protocol["episodes"]) / "frames.parquet")
    store, stats, seconds = load_store(frames, manifest, protocol["physical"], SETTINGS["body_half_window"])
    if stats != protocol["normalization"] or not seconds.equals(pd.read_parquet(output / "normalization_seconds.parquet")):
        raise ValueError("Prepared training-only normalization does not reproduce")
    upstream = _upstream_module("train_frame_mil")
    caches, lut = upstream.load_multi([protocol["cache"]])
    torch.set_num_threads(1)
    device = "cpu"
    rows = {split: manifest[manifest.split == split].reset_index(drop=True) for split in CATEGORIES}
    datasets = {split: upstream.FramePairs(caches, lut, table, VOCAB, SETTINGS["visual_half_window"], False)
                for split, table in rows.items()}
    if any(len(datasets[split]) != len(table) or datasets[split].n_novocab or datasets[split].n_noframe
           for split, table in rows.items()):
        raise ValueError("Property dataset dropped a frozen example")
    data = {split: _body_data(table, store, lut, device, SETTINGS) for split, table in rows.items()}
    metadata = {"code_revision": _revision(), "protocol_sha256": _sha(output / "protocol.json"),
                "device": device, "cpu_threads": 1, "status": "running", "completed_models": [],
                "settings": SETTINGS, "vocab": VOCAB, "word_score_columns": WORDS}
    _json(output / "run.json", metadata)
    started = time.monotonic()
    deadline = started + SETTINGS["budget_s"]
    all_video, all_body, all_shuffled, all_picture = [], [], [], []
    all_joint, all_joint_shuffled = [], []
    try:
        for seed in SETTINGS["seeds"]:
            matched = []
            for use_body, arch in ((False, "additive"), (True, "additive"), (True, "joint")):
                name = f"{'joint' if arch == 'joint' else 'load' if use_body else 'video'}_s{seed}"
                arm_dir = output / name
                model, metrics = _train_arm(seed, use_body, datasets, rows, data, caches[0].shape[-1], device,
                                            upstream, SETTINGS, arm_dir, deadline, arch=arch)
                scores = score_words(model, datasets["test"], data["test"] if use_body else None, device, upstream)
                credit = word_credit(scores, rows["test"].text)
                predictions = rows["test"].copy()
                predictions["score_heavy"], predictions["score_light"] = scores.T
                predictions["credit"] = credit
                predictions.to_parquet(arm_dir / "predictions.parquet", index=False)
                metrics["test_accuracy"] = float(credit.mean())
                if use_body:
                    picture = score_words(model, datasets["test"], None, device, upstream)
                    predictions["picture_score_heavy"], predictions["picture_score_light"] = picture.T
                    predictions["picture_credit"] = word_credit(picture, rows["test"].text)
                    predictions.to_parquet(arm_dir / "predictions.parquet", index=False)
                    if arch == "joint":
                        all_joint.append(credit)
                    else:
                        all_body.append(credit)
                        all_picture.append(predictions.picture_credit.to_numpy())
                    shuffle_records, shuffle_credit = [], []
                    for draw, donor in enumerate(protocol["donor_indices"]):
                        _deadline(deadline)
                        score = score_words(model, datasets["test"], data["test"], device, upstream, donors=donor)
                        credit_draw = word_credit(score, rows["test"].text)
                        shuffle_credit.append(credit_draw)
                        shuffle_records.extend({"draw": draw, "example_id": int(rows["test"].iloc[i].example_id),
                                                "donor_example_id": int(rows["test"].iloc[d].example_id),
                                                "score_heavy": float(score[i, 0]), "score_light": float(score[i, 1]),
                                                "credit": float(credit_draw[i])} for i, d in enumerate(donor))
                    pd.DataFrame(shuffle_records).to_parquet(arm_dir / "shuffled_predictions.parquet", index=False)
                    if arch == "joint":
                        all_joint_shuffled.append(np.mean(shuffle_credit, axis=0))
                    else:
                        all_shuffled.append(np.mean(shuffle_credit, axis=0))
                    metrics["picture_only_accuracy"] = float(predictions.picture_credit.mean())
                    metrics["shuffled_accuracy"] = float(np.mean(shuffle_credit))
                else:
                    all_video.append(credit)
                _deadline(deadline)
                _json(arm_dir / "metrics.json", metrics)
                metadata["completed_models"].append(name)
                _json(output / "run.json", metadata)
                matched.append(metrics)
                del model
            if any(matched[0][key] != m[key] for m in matched[1:] for key in ("common_initialization_sha256", "minibatch_order_sha256")):
                raise ValueError(f"Common initialization or minibatch exposure differs for seed {seed}")
        _verify_frozen(output, protocol)
        _deadline(deadline)
        result = paired_summary(all_video, all_body, all_shuffled)
        result["joint"] = paired_summary(all_video, all_joint, all_joint_shuffled)
        result.update({"completed": True, "n_test_source_clips": len(rows["test"]), "seeds": SETTINGS["seeds"],
                       "picture_only_load_model_accuracy": float(np.mean(all_picture)),
                       "picture_only_minus_independent_video": float(np.mean(all_picture) - np.mean(all_video)),
                       "limits": LIMITS, "runtime_s": time.monotonic() - started,
                       "final_frozen_source_and_code_verification": "passed"})
        _json(output / "comparison.json", result)
        metadata.update(status="complete", runtime_s=result["runtime_s"])
        _json(output / "run.json", metadata)
        return result
    except Exception as error:
        metadata.update(status="aborted_partial", error=f"{type(error).__name__}: {error}", runtime_s=time.monotonic() - started)
        _json(output / "run.json", metadata)
        raise


def prepare_existing(original, output):
    """Copy the retained, previously frozen panel without resampling any encounter."""
    import shutil

    original, output = Path(original).resolve(), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Property preparation output must be empty: {output}")
    old = json.loads((original / "protocol.json").read_text())
    run = json.loads((original / "run.json").read_text())
    if run["status"] != "complete" or _sha(original / "protocol.json") != run["protocol_sha256"]:
        raise ValueError("Original heavy/light protocol does not match completed run")
    for name, key in (("manifest.parquet", "manifest_sha256"),
                      ("normalization_seconds.parquet", "normalization_seconds_sha256")):
        if _sha(original / name) != old[key]:
            raise ValueError(f"Original frozen artifact changed: {name}")
    if old["settings"] != SETTINGS or old["categories"] != CATEGORIES or old["vocab"] != VOCAB:
        raise ValueError("Original diagnostic settings differ")
    for path, expected in old["source_sha256"].items():
        if _sha(path) != expected:
            raise ValueError(f"Original source changed: {path}")
    output = _output_directory(output)
    for name in ("manifest.parquet", "normalization_seconds.parquet"):
        shutil.copyfile(original / name, output / name)
    protocol = dict(old, code_sha256=_code_hashes(), prepared_code_revision=_revision(),
                    original_protocol=str(original / "protocol.json"),
                    original_protocol_sha256=run["protocol_sha256"],
                    joint_interaction_source_commit="08ebb82c",
                    joint_interaction_sha256=_sha(Path(__file__).with_name("interaction.py")),
                    architecture_conditions=["video", "load", "joint"], device="cpu", cpu_threads=1,
                    budget="Nine models share one 1800-second wall-clock deadline; incomplete comparison aborts.")
    _json(output / "protocol.json", protocol)
    _verify_frozen(output, protocol)
    return protocol


def timing_prepared(output, timing_output):
    """One joint training/dev epoch on CPU; discard weights and never score test."""
    output = Path(output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    _verify_frozen(output, protocol)
    manifest = pd.read_parquet(output / "manifest.parquet")
    frames = pd.read_parquet(Path(protocol["episodes"]) / "frames.parquet")
    store, stats, seconds = load_store(frames, manifest, protocol["physical"])
    if stats != protocol["normalization"] or not seconds.equals(pd.read_parquet(output / "normalization_seconds.parquet")):
        raise ValueError("Prepared training-only normalization does not reproduce")
    torch.set_num_threads(1)
    upstream = _upstream_module("train_frame_mil")
    caches, lut = upstream.load_multi([protocol["cache"]])
    rows = {split: manifest[manifest.split == split].reset_index(drop=True) for split in ("train", "dev")}
    datasets = {split: upstream.FramePairs(caches, lut, table, VOCAB, 2, False) for split, table in rows.items()}
    if any(len(datasets[split]) != len(rows[split]) for split in rows):
        raise ValueError("Timing dataset dropped a frozen example")
    data = {split: _body_data(table, store, lut, "cpu", SETTINGS) for split, table in rows.items()}
    timing_output = Path(timing_output)
    if timing_output.exists() and any(timing_output.iterdir()):
        raise FileExistsError("Timing output must be empty")
    timing_output = _output_directory(timing_output)
    started = time.monotonic()
    _, metrics = _train_arm(0, True, datasets, rows, data, caches[0].shape[-1], "cpu", upstream,
                            {**SETTINGS, "epochs": 1}, timing_output, started + SETTINGS["budget_s"],
                            arch="joint", timing=True)
    result = {"seconds": time.monotonic() - started, "epochs": 1, "condition": "joint", "device": "cpu",
              "cpu_threads": 1, "training_examples": len(rows["train"]), "dev_examples": len(rows["dev"]),
              "protocol_sha256": _sha(output / "protocol.json"), "weights_saved": False,
              "test_scored": False, "metrics": metrics}
    _json(timing_output / "timing.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--original", required=True)
    setup.add_argument("--output", required=True)
    timing = commands.add_parser("timing")
    timing.add_argument("--run", required=True)
    timing.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_existing(args.original, args.output)
    elif args.command == "timing":
        print(json.dumps(timing_prepared(args.run, args.output), indent=2))
    else:
        print(json.dumps(train_prepared(args.run), indent=2))
