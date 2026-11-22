"""Frozen Step 3: temporal/sensory pretraining followed by video-only weak naming.

The staged entry point freezes exposures and controls before any model is fit.
It reuses the existing learner, consistency/prediction losses and picture tests.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from nursery.common import REPO, _output_directory
from . import UPSTREAM, _upstream_module
from . import body, evaluate
from .model import LearnerMIL
from .property import _sha, _json, _revision
from .train import CONFIG, clip_partition, split_pairs


SETTINGS = {"seeds": [0, 1, 2], "arms": ["V", "A", "S"], "window": 2,
            "pretrain_passes": 50, "pretrain_batch": 64, "pretrain_updates": 500,
            "epochs": 30, "batch": 256, "lr": 0.0003, "weight_decay": 0.1,
            "tau_p": 0.1, "dim": 512, "predict_weight": 0.5,
            "budget_s": 7200, "bootstrap_draws": 10000, "bootstrap_seed": 0,
            "sampling_seed": 0, "donor_seed": 0, "evaluation_seed": 0, "condition": "none"}
COUNTS = {"pretrain_clips": 631, "train_pairs": 18613,
          "clips_dev": [486, 119], "clips_test": [570, 115], "s8": [787, 158],
          "sensory_dev_clips": 134, "sensory_dev_source_clips": 135}
DEV_EXCLUSIONS = {"s1/stamp_stamp_1.npz": "Only three contiguous corpus frames (117–119); five required. No padding or synthetic frames."}
GATE = {"gain_min": 0.05, "each_seed_positive": True, "ci_lower_positive": True}
LIMITS = ["Exploratory gate: the test sets were previously consulted.",
          "Target-clip bootstrap does not account for dependence from reused candidate foils.",
          "The +5 percentage-point pilot threshold is not a power claim.",
          "All readouts are body-absent; this does not test body-present understanding or a higher learning ceiling.",
          "Clip-test gain alone does not prove object representation; reused picture probes are corroboration only."]
PREPARED = {"protocol.json", "windows.parquet", "dev_windows.parquet", "normalization_seconds.parquet",
            "pairs.parquet", "vocab.json", "trials.json"}


def select_windows(frames, part, available, partition=0, seed=0):
    """First eligible contiguous occurrence per sorted source clip; uniform five-frame start.

    Only boundaries, subject/split and observed availability participate. The
    result has no words, identities, masses or other oracle metadata.
    """
    metadata = frames[["video_id", "frame_idx", "clip", "subject"]].copy()
    metadata = metadata[metadata.subject.ne("s8") & metadata["clip"].map(lambda c: isinstance(c, str) and bool(c))]
    metadata = metadata[[part[r.video_id][int(r.frame_idx)] == partition for r in metadata.itertuples()]]
    rng, selected = np.random.default_rng(seed), []
    for name, group in metadata.groupby("clip", sort=True):
        found = False
        for video, occurrence in group.groupby("video_id", sort=True):
            occurrence = occurrence.sort_values("frame_idx")
            times = occurrence.frame_idx.to_numpy(dtype=int)
            for chunk in np.split(times, np.flatnonzero(np.diff(times) != 1) + 1):
                starts = [int(t) for t in chunk[:-4] if all((video, int(f)) in available for f in range(int(t), int(t) + 5))]
                if starts:
                    start = int(rng.choice(starts))
                    subject = occurrence.subject.iloc[0]
                    selected.append({"clip": name, "video_id": video, "subject": subject,
                                     "start": start, "frame_idx": start + 2, "mask": "11111"})
                    found = True
                    break
            if found:
                break
        if not found:
            # This predeclared exception is descriptive-dev-only. Training and
            # any unexpected ineligible source clip remain strict failures.
            if partition == 1 and name in DEV_EXCLUSIONS:
                continue
            raise ValueError(f"No eligible five-frame occurrence for source clip: {name}")
    return pd.DataFrame(selected)


def donor_derangement(windows, seed=0):
    """Bijective full-window derangement in subject/availability-mask strata."""
    rng, donors = np.random.default_rng(seed), np.arange(len(windows))
    for _, group in windows.groupby(["subject", "mask"], sort=True):
        ids = group.index.to_numpy()
        if len(ids) < 2:
            raise ValueError("Every subject/mask stratum needs two different source clips")
        order = rng.permutation(ids)
        donors[order] = np.roll(order, 1)
    if not windows["clip"].is_unique or (windows["clip"].to_numpy() == windows.iloc[donors]["clip"].to_numpy()).any():
        raise ValueError("Donors must have different training-source clips")
    return donors


def sensor_arrays(episodes, videos):
    """Existing sensor target definitions only; no encoder inputs or identities."""
    arrays, available = {}, set()
    for video in videos:
        with np.load(Path(episodes) / "body" / f"{video}.npz", allow_pickle=False) as signal:
            features = body.session_features(signal)
            times = signal["frame_idx"].astype(int)
        arrays[video] = {int(t): {key: features[key][i] for key in ("contact", "aperture", "in_view")}
                         for i, t in enumerate(times)}
        available.update((video, int(t)) for t in times)
    return arrays, available


def window_targets(windows, arrays, stats=None):
    """Fit aperture on selected training frames only; changes use four internal edges."""
    values = {key: np.stack([[arrays[r.video_id][t][key] for t in range(r.start, r.start + 5)]
                             for r in windows.itertuples()]).astype(np.float32)
              for key in ("contact", "aperture", "in_view")}
    seconds = windows[["video_id", "start"]].assign(frame_idx=lambda x: x.start.map(lambda t: list(range(t, t + 5)))).explode("frame_idx")
    seconds = seconds[["video_id", "frame_idx"]].drop_duplicates().sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    seconds["frame_idx"] = seconds.frame_idx.astype(int)
    if stats is None:
        fit = np.stack([arrays[r.video_id][r.frame_idx]["aperture"] for r in seconds.itertuples()])
        stats = {"mean": fit.mean(0).tolist(), "std": (fit.std(0) + 1e-6).tolist(),
                 "unique_training_seconds": len(seconds)}
    values["aperture"] = ((values["aperture"] - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)
    values["contact"] = (values["contact"] > 0).astype(np.float32)
    values["change"] = np.zeros_like(values["contact"])
    values["change"][:, :-1] = np.diff(values["contact"], axis=1)
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Non-finite sensory targets")
    if not np.isin(values["in_view"], [0, 1]).all():
        raise ValueError("View BCE targets must be binary")
    return values, stats, seconds


def language_rows(run, frames, part, upstream):
    """Mirror the canonical corrected-speech split, retaining every ordinary pair."""
    generator = json.loads((run / "generator.json").read_text())
    manifest = pd.read_parquet(run / "manifest.parquet")
    sessions = pd.read_parquet(Path(generator["episodes"]) / "sessions.parquet")
    subjects = dict(zip(sessions.video_id, sessions.subject))
    heldout = manifest.video_id.map(subjects).eq("s8")
    vocab = upstream.build_vocab(manifest[~heldout].text, generator["config"]["training"]["min_freq"])
    manifest = manifest[manifest.text.map(lambda text: bool(upstream.encode(text, vocab)))].copy()
    heldout = manifest.video_id.map(subjects).eq("s8")
    split = split_pairs(manifest[~heldout], part, SETTINGS["window"])
    manifest["partition"] = "s8"
    manifest.loc[~heldout, "partition"] = [{-1: "crossing", 0: "train", 1: "clips_dev", 2: "clips_test"}[int(k)] for k in split]
    manifest = manifest[manifest.partition != "crossing"]
    manifest = manifest[(manifest.partition == "train") | (manifest.named.astype(bool) & manifest.referent_visible.astype(bool))].copy()
    lookup = frames.set_index(["video_id", "frame_idx"])["clip"]
    manifest["target_clip"] = [lookup.loc[r.video_id, int(r.frame_idx)] for r in manifest.itertuples()]
    manifest["subject"] = manifest.video_id.map(subjects)
    return manifest.reset_index(drop=True), vocab


def _code_hashes():
    files = [Path(__file__), Path(__file__).with_name("__main__.py"), Path(__file__).with_name("model.py"),
             Path(__file__).with_name("train.py"), Path(__file__).with_name("evaluate.py"),
             Path(__file__).with_name("body.py"), Path(__file__).with_name("property.py"), CONFIG,
             Path(__file__).with_name("hand_patches.json"),
             REPO / "nursery/common.py", REPO / "PLAN.md"]
    files += [UPSTREAM / "src" / name for name in ("train_frame_mil.py", "train_region_mil.py", "train.py", "common.py")]
    return {str(path.relative_to(REPO)): _sha(path) for path in files}


def prepare(output, run=None):
    """Freeze exposures, complete-window donors, splits, normalization and hashes; never train."""
    output = _output_directory(output)
    if any(output.iterdir()):
        raise FileExistsError("Sensory-pretraining preparation requires an empty ignored root")
    run = Path(run or REPO / "outputs/runs/step1_matched").resolve()
    generator = json.loads((run / "generator.json").read_text())
    episodes = Path(generator["episodes"])
    frames = pd.read_parquet(episodes / "frames.parquet")
    upstream = _upstream_module("train_frame_mil")
    _, lut = upstream.load_multi([run / "train_cache"])
    cfg = json.loads(CONFIG.read_text())["body"]["clip_split"]
    part, counts = clip_partition(frames, "s8", cfg, int(frames.frame_idx.max()) + 1)
    arrays, available = sensor_arrays(episodes, sorted(frames.video_id.unique()))
    available &= {(key.rsplit("|", 1)[0], int(key.rsplit("|", 1)[1])) for key in lut}
    windows = select_windows(frames, part, available)
    dev_windows = select_windows(frames, part, available, partition=1)
    dev_source_clips = {r.clip for r in frames.itertuples() if r.subject != "s8" and
                        isinstance(r.clip, str) and r.clip and part[r.video_id][int(r.frame_idx)] == 1}
    dev_excluded = dev_source_clips - set(dev_windows["clip"])
    if (len(dev_windows) != COUNTS["sensory_dev_clips"] or
            len(dev_source_clips) != COUNTS["sensory_dev_source_clips"] or
            dev_excluded != set(DEV_EXCLUSIONS)):
        raise ValueError("Descriptive sensory-dev eligibility differs from the frozen 134/135 audit")
    targets, stats, seconds = window_targets(windows, arrays)
    windows["donor"] = donor_derangement(windows, SETTINGS["donor_seed"])
    pairs, vocab = language_rows(run, frames, part, upstream)
    if len(windows) != COUNTS["pretrain_clips"] or int(pairs.partition.eq("train").sum()) != COUNTS["train_pairs"]:
        raise ValueError("Frozen pretraining/language exposure counts differ from PLAN.md")
    trials = {}
    corrected_trials = json.loads((run / "trial_manifest.json").read_text())
    for split in ("clips_dev", "clips_test", "s8"):
        rows = pairs[pairs.partition == split].reset_index(drop=True)
        context = evaluate.body_present_context(rows, frames, lut, SETTINGS["window"], 5, rows.subject.to_numpy())
        trials[split] = {}
        for seed in SETTINGS["seeds"]:
            sampled = evaluate._body_trials(rows, context, SETTINGS["evaluation_seed"], rows.subject.to_numpy())
            query = [int(i) for i, _, _ in sampled]
            if [len(query), rows.iloc[query].target_clip.nunique()] != COUNTS[split]:
                raise ValueError(f"Corrected {split} query/target-clip counts changed")
            trials[split][str(seed)] = [{"row": int(i), "candidates": c.tolist(), "answer": int(a)} for i, c, a in sampled]
            identities = [[list(context["identity"][j]) for j in c] for _, c, _ in sampled]
            if (identities != corrected_trials[split]["candidate_identities"] or
                    [int(a) for _, _, a in sampled] != corrected_trials[split]["correct_slots"] or
                    query != corrected_trials[split]["trial_rows"]):
                raise ValueError(f"{split} differs from frozen corrected Step 1 trials")
    paths = [episodes / "frames.parquet", episodes / "sessions.parquet", run / "generator.json", run / "manifest.parquet", run / "trial_manifest.json"]
    for cache in ("train_cache", "probes_cache", "konkle_cache"):
        paths.extend([run / cache / "index.parquet", run / cache / "emb.f16.npy"])
    paths.extend(run / filename for filename in ("probes_frames.parquet", "konkle_frames.parquet"))
    paths.extend(episodes / "body" / f"{v}.npz" for v in sorted(frames.video_id.unique()))
    protocol = {"task": "step3_sensory_pretraining_transfer", "settings": SETTINGS, "counts": COUNTS, "gate": GATE,
                "limits": LIMITS, "run": str(run), "episodes": str(episodes), "normalization": stats,
                "clip_counts": counts, "prepared_code_revision": _revision(), "code_sha256": _code_hashes(),
                "sensory_dev_audit": {"source_clips": len(dev_source_clips), "eligible_clips": len(dev_windows),
                                      "excluded_clips": DEV_EXCLUSIONS, "target_frames": 5 * len(dev_windows),
                                      "internal_change_edges": 4 * len(dev_windows)},
                "source_sha256": {str(path): _sha(path) for path in paths},
                "selection": "First maximum corrected clips_dev picture_soft for clip/s8; first maximum rendered dev for picture probes.",
                "pretraining": "Mean of existing consistency_loss over four adjacent pairs; A/S add 0.5 existing predict_loss on all five frames and four internal change edges.",
                "transfer": "Only vproj copied into fresh video-only learner; fresh words/logit scale/optimizer and reset per-seed orders.",
                "sampling": "Sorted source clips; first eligible contiguous occurrence in video/frame order; uniformly seeded valid five-frame start.",
                "sensory_diagnostic": "Eligible dev source clips only (134/135; excluded clip and reason in sensory_dev_audit), one same-rule window per eligible clip with independent seed 0; final A/S on true aligned targets. Means fit selected train frames (change: four internal edges). Persistence uses previous true sensors, train mean at first frame, zero contact change; privileged reference. BCE probabilities clamp 1e-6..1-1e-6. No selection or gate.",
                "budget": "One 7200-second wall-clock budget for all nine runs; abort incomplete without a comparison; no retries or shortened runs."}
    windows.to_parquet(output / "windows.parquet", index=False)
    dev_windows.to_parquet(output / "dev_windows.parquet", index=False)
    seconds.to_parquet(output / "normalization_seconds.parquet", index=False)
    pairs.to_parquet(output / "pairs.parquet", index=False)
    _json(output / "vocab.json", vocab)
    _json(output / "trials.json", trials)
    protocol["prepared_sha256"] = {name: _sha(output / name) for name in sorted(PREPARED - {"protocol.json"})}
    _json(output / "protocol.json", protocol)
    return protocol


def _verify_frozen(output, protocol):
    if any(protocol[key] != value for key, value in (("settings", SETTINGS), ("counts", COUNTS), ("gate", GATE))):
        raise ValueError("Prepared settings differ from canonical frozen Step 3")
    if protocol["code_sha256"] != _code_hashes():
        raise ValueError("Source changed since preparation")
    for path, expected in protocol["source_sha256"].items():
        if _sha(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for name, expected in protocol["prepared_sha256"].items():
        if _sha(output / name) != expected:
            raise ValueError(f"Frozen prepared artifact changed: {name}")


def _hash_state(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("The fixed nine-run two-hour budget expired; comparison is incomplete")


def make_model(seed, vocab, emb_dim, device, sensory=False, projection=None):
    """Identical common initialization; only vproj crosses the phase boundary."""
    torch.manual_seed(seed)
    model = LearnerMIL(vocab, emb_dim, SETTINGS["tau_p"], {"contact": 22} if sensory else None).float().to(device)
    if model.word.embedding_dim != SETTINGS["dim"] or model.center_mu.any():
        raise ValueError("Step 3 requires the existing 512-dimensional learner and zero center_mu")
    if projection is not None:
        model.vproj.load_state_dict(projection)
    return model


def temporal_loss(model, images, episode):
    """Existing loss, averaged equally over all four internal adjacencies."""
    if images.ndim != 4 or images.shape[1:3] != (5, 17):
        raise ValueError("Temporal pretraining requires five full region frames")
    B = len(images)
    ctx = {"bag_sel": torch.arange(B, device=images.device), "bag_images": images, "bag_episode": episode}
    terms = []
    for t in range(4):
        ctx["bag_ab"] = torch.tensor([t, t + 1], device=images.device).expand(B, 2)
        terms.append(model.consistency_loss(torch.ones(B, device=images.device), ctx, model.scale()))
    return torch.stack(terms).mean()


def sensory_context(targets, ids, device):
    batch = {key: torch.as_tensor(value[ids], device=device) for key, value in targets.items()}
    valid = torch.ones((len(ids), 5), dtype=torch.bool, device=device)
    nxt = valid.clone()
    nxt[:, -1] = False  # No sixth frame; donor changes remain internal to the donor window.
    return {"frame_targets": batch, "frame_valid": valid, "frame_next_valid": nxt}


def sensory_components(prediction, ctx):
    """Descriptive component readout, with exactly the existing predict_loss reductions."""
    contact, change, aperture, view = prediction.split([22, 22, 2, 2], dim=-1)
    target, valid, nxt = ctx["frame_targets"], ctx["frame_valid"], ctx["frame_next_valid"]
    terms = {"contact": F.binary_cross_entropy_with_logits(contact[valid], target["contact"][valid]),
             "change": F.mse_loss(change[nxt], target["change"][nxt]),
             "aperture": F.mse_loss(aperture[valid], target["aperture"][valid]),
             "in_view": F.binary_cross_entropy_with_logits(view[valid], target["in_view"][valid])}
    return {**{k: float(v.detach()) for k, v in terms.items()}, "total": float(sum(terms.values()).detach())}


def sensory_references(train_targets, dev_targets):
    """Frozen train mean and privileged previous-true-sensor reference predictions."""
    means = {key: value[:, :-1].mean((0, 1)) if key == "change" else value.mean((0, 1))
             for key, value in train_targets.items()}
    mean = {key: np.broadcast_to(means[key], value.shape).copy() for key, value in dev_targets.items()}
    persistence = {key: np.concatenate([mean[key][:, :1], value[:, :-1]], axis=1)
                   for key, value in dev_targets.items()}
    persistence["change"] = np.zeros_like(dev_targets["change"])
    def raw(values):
        pieces = []
        for key in ("contact", "change", "aperture", "in_view"):
            value = torch.as_tensor(values[key])
            pieces.append(torch.logit(value.clamp(1e-6, 1 - 1e-6)) if key in ("contact", "in_view") else value)
        return torch.cat(pieces, dim=-1)
    ctx = sensory_context(dev_targets, np.arange(len(dev_targets["contact"])), "cpu")
    return {"mean": sensory_components(raw(mean), ctx),
            "persistence": sensory_components(raw(persistence), ctx),
            "persistence_scope": "Privileged reference: previous true contact/aperture/view; first frame train mean; change zero."}


@torch.no_grad()
def sensory_readout(model, images, targets, device, deadline, batch=64):
    """Final fixed checkpoint only, true aligned dev targets; never used for selection."""
    model.eval()
    predictions = []
    for start in range(0, len(images), batch):
        _deadline(deadline)
        regions = model.enc_regions(torch.as_tensor(images[start:start + batch], device=device).flatten(1, 2))
        regions = regions.view(len(regions), 5, 17, -1)
        predictions.append(model.region_head(F.relu(model.region_in(regions)).flatten(2)).cpu())
    _deadline(deadline)
    return sensory_components(torch.cat(predictions), sensory_context(targets, np.arange(len(images)), "cpu"))


def pretrain_arm(seed, arm, images, targets, donors, vocab, device, settings, output, deadline):
    _deadline(deadline)
    model = make_model(seed, vocab, images.shape[-1], device, sensory=True)
    initial = _hash_state(model.vproj.state_dict())
    # Body encoder, word vectors and fusion are not part of either pretraining objective.
    names = ("vproj.", "logit_scale") + (("region_in.", "region_head.") if arm != "V" else ())
    optimizer = torch.optim.AdamW([p for name, p in model.named_parameters() if name.startswith(names)],
                                  lr=settings["lr"], weight_decay=settings["weight_decay"])
    generator, order_hash = torch.Generator().manual_seed(seed), hashlib.sha256()
    history, updates = [], 0
    for epoch in range(settings["pretrain_passes"]):
        model.train()
        order = torch.randperm(len(images), generator=generator).numpy()
        order_hash.update(order.tobytes())
        sums = []
        for start in range(0, len(order), settings["pretrain_batch"]):
            _deadline(deadline)
            # Keep the shared temporal objective's dropout draws aligned even
            # though A/S also execute a sensory-prediction forward pass.
            torch.manual_seed(seed * 100000 + updates)
            ids = order[start:start + settings["pretrain_batch"]]
            visual = torch.as_tensor(images[ids], device=device)
            loss = temporal_loss(model, visual, torch.as_tensor(ids, device=device))
            if arm != "V":
                target_ids = donors[ids] if arm == "S" else ids
                ctx = sensory_context(targets, target_ids, device)
                prediction = model.predict_loss(model.enc_regions(visual.flatten(1, 2)), ctx)
                loss = loss + settings["predict_weight"] * prediction
            if not torch.isfinite(loss):
                raise ValueError("Non-finite pretraining loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            sums.append(float(loss.detach()))
        _deadline(deadline)
        history.append({"pass": epoch, "loss": float(np.mean(sums)), "updates": updates})
        _json(output / "pretrain_history.json", history)
    if updates != settings["pretrain_updates"]:
        raise ValueError("Pretraining did not complete exactly the frozen update count")
    state = {k: v.detach().cpu().clone() for k, v in model.vproj.state_dict().items()}
    torch.save(model.state_dict(), output / "pretrain_final.pt")
    return model, state, {"pretrain_initial_projection_sha256": initial,
                          "pretrain_order_sha256": order_hash.hexdigest(), "pretrain_updates": updates}


def _snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def transfer_arm(seed, projection, datasets, pairs, vocab, device, upstream, settings, output, deadline, score):
    model = make_model(seed, len(vocab), datasets["train"].caches[0].shape[-1], device, projection=projection)
    fresh_hash = _hash_state({k: v for k, v in model.state_dict().items() if not k.startswith("vproj.")})
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    generator, order_hash = torch.Generator().manual_seed(seed), hashlib.sha256()
    weights = torch.as_tensor(pairs["train"]["weight_none"].to_numpy(np.float32), device=device)
    weights = weights / weights.mean()
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid frozen ordinary speech weights")
    best_clip, best_picture, clip_state, picture_state = -1.0, -1.0, None, None
    clip_epoch, picture_epoch, history = -1, -1, []
    for epoch in range(settings["epochs"]):
        model.train()
        order = torch.randperm(len(datasets["train"]), generator=generator).numpy()
        order_hash.update(order.tobytes())
        losses = []
        for start in range(0, len(order), settings["batch"]):
            _deadline(deadline)
            ids = order[start:start + settings["batch"]]
            _, visual, tokens, lengths = upstream.collate([datasets["train"][int(i)] for i in ids])
            loss = model.forward_loss(visual.to(device), tokens.to(device), lengths.to(device), weights[ids])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite weak-noun language loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        clip_dev, picture_dev = score(model, "clips_dev"), score(model, "rendered_dev")
        _deadline(deadline)
        if clip_dev > best_clip:
            best_clip, clip_epoch, clip_state = clip_dev, epoch, _snapshot(model)
            torch.save(clip_state, output / "language_clip_selected.pt")
        if picture_dev > best_picture:
            best_picture, picture_epoch, picture_state = picture_dev, epoch, _snapshot(model)
            torch.save(picture_state, output / "language_picture_selected.pt")
        history.append({"epoch": epoch, "language_loss": float(np.mean(losses)),
                        "clips_dev_picture_soft": clip_dev, "rendered_dev": picture_dev})
        _json(output / "language_history.json", history)
    return model, clip_state, picture_state, {"fresh_language_state_sha256": fresh_hash,
                "language_order_sha256": order_hash.hexdigest(), "epochs_completed": len(history),
                "clip_selected_epoch": clip_epoch, "picture_selected_epoch": picture_epoch,
                "clip_dev_accuracy": best_clip, "rendered_dev_accuracy": best_picture}


def paired_summary(credits, clips, settings=SETTINGS):
    """Resample whole target clips; trial-weight within seed, then equally average seeds."""
    arrays = {arm: np.asarray(credits[arm], dtype=float) for arm in ("V", "A", "S")}
    if any(a.shape != arrays["V"].shape for a in arrays.values()) or arrays["V"].ndim != 2:
        raise ValueError("Expected matched seed-by-trial credit arrays")
    clips = np.asarray(clips)
    if len(clips) != arrays["V"].shape[1] or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Invalid matched target-clip credits")
    names, cluster = np.unique(clips, return_inverse=True)
    count = np.bincount(cluster)
    rng = np.random.default_rng(settings["bootstrap_seed"])
    draws = rng.integers(len(names), size=(settings["bootstrap_draws"], len(names)))
    result = {"accuracy": {arm: float(a.mean(1).mean()) for arm, a in arrays.items()}, "contrasts": {}}
    for control in ("V", "S"):
        delta = arrays["A"] - arrays[control]
        sums = np.stack([np.bincount(cluster, weights=row, minlength=len(names)) for row in delta])
        boot = (sums[:, draws].sum(2) / count[draws].sum(1)).mean(0)
        ci = np.percentile(boot, [2.5, 97.5])
        seed_gain = delta.mean(1)
        criteria = {"gain_at_least_5pp": bool(seed_gain.mean() >= GATE["gain_min"]),
                    "every_seed_positive": bool((seed_gain > 0).all()), "ci_lower_positive": bool(ci[0] > 0)}
        result["contrasts"][f"A-{control}"] = {"gain": float(seed_gain.mean()), "seed_gains": seed_gain.tolist(),
                   "target_clip_ci95": ci.tolist(), "criteria": criteria, "pass": all(criteria.values())}
    result["transfer_gate_pass"] = all(c["pass"] for c in result["contrasts"].values())
    return result


def _training_device():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Frozen Step 3 requires local MPS")
    return "mps"


def train_prepared(output):
    """Run the frozen nine experiments once on local MPS; preserve partial outputs on failure."""
    output = Path(output).resolve()
    if {p.name for p in output.iterdir()} != PREPARED:
        raise FileExistsError("Training requires exactly prepared artifacts; existing outputs are never overwritten/retried")
    _output_directory(output)
    protocol = json.loads((output / "protocol.json").read_text())
    _verify_frozen(output, protocol)
    device = _training_device()
    run, episodes = Path(protocol["run"]), Path(protocol["episodes"])
    upstream, region = _upstream_module("train_frame_mil"), _upstream_module("train_region_mil")
    caches, lut = upstream.load_multi([run / "train_cache"])
    windows = pd.read_parquet(output / "windows.parquet")
    dev_windows = pd.read_parquet(output / "dev_windows.parquet")
    pairs_all = pd.read_parquet(output / "pairs.parquet")
    vocab = json.loads((output / "vocab.json").read_text())
    frames = pd.read_parquet(episodes / "frames.parquet")
    arrays, _ = sensor_arrays(episodes, sorted(set(windows.video_id) | set(dev_windows.video_id)))
    targets, stats, seconds = window_targets(windows, arrays)
    if stats != protocol["normalization"] or not seconds.equals(pd.read_parquet(output / "normalization_seconds.parquet")):
        raise ValueError("Selected training-frame normalization no longer reproduces")
    images = np.stack([caches[0][[lut[f"{r.video_id}|{t}"][1] for t in range(r.start, r.start + 5)]].astype(np.float32)
                       for r in windows.itertuples()])
    dev_images = np.stack([caches[0][[lut[f"{r.video_id}|{t}"][1] for t in range(r.start, r.start + 5)]].astype(np.float32)
                           for r in dev_windows.itertuples()])
    dev_targets, _, _ = window_targets(dev_windows, arrays, stats)
    pairs = {split: pairs_all[pairs_all.partition == split].reset_index(drop=True)
             for split in ("train", "clips_dev", "clips_test", "s8")}
    datasets = {split: upstream.FramePairs(caches, lut, rows, vocab, SETTINGS["window"], False) for split, rows in pairs.items()}
    if any(len(datasets[s]) != len(rows) or datasets[s].n_noframe or datasets[s].n_novocab for s, rows in pairs.items()):
        raise ValueError("Frozen ordinary speech dataset lost pairs")
    contexts = {split: evaluate.body_present_context(rows, frames, lut, SETTINGS["window"], 5, rows.subject.to_numpy())
                for split, rows in pairs.items() if split != "train"}
    frozen_trials = json.loads((output / "trials.json").read_text())
    probe_cache, probe_lut = region.load_region_cache(run / "probes_cache")
    probes = pd.read_parquet(run / "probes_frames.parquet")
    konkle_cache, konkle_lut = region.load_region_cache(run / "konkle_cache")
    konkle = pd.read_parquet(run / "konkle_frames.parquet")
    metadata = {"status": "running", "completed_runs": [], "device": device, "code_revision": _revision(),
                "protocol_sha256": _sha(output / "protocol.json"), "settings": SETTINGS}
    started, credits, matched = time.monotonic(), {arm: [] for arm in SETTINGS["arms"]}, {}
    deadline = started + SETTINGS["budget_s"]
    _json(output / "run.json", metadata)
    try:
        _deadline(deadline)
        references = sensory_references(targets, dev_targets)
        for seed in SETTINGS["seeds"]:
            def clip_score(model, split, trials=False):
                result = evaluate.body_present_4afc(model, upstream, datasets[split], None, pairs[split], device, SETTINGS["evaluation_seed"],
                         subject=pairs[split].subject.to_numpy(), trials=True, context=contexts[split])
                sampled = [{"row": i, "candidates": c, "answer": a} for i, c, a in
                           zip(result["trial_rows"], result["candidate_rows"], result["correct_slots"])]
                if sampled != frozen_trials[split][str(seed)]:
                    raise ValueError("Evaluation trials changed after preparation")
                return result if trials else float(np.mean(result["trials"]["picture_soft"]))

            def score(model, split):
                if split == "rendered_dev":
                    return float(region.eval_4afc_region(model, probe_cache, probe_lut, probes[probes.split == "dev"], vocab, device))
                return clip_score(model, split)

            for arm in SETTINGS["arms"]:
                _deadline(deadline)
                name = f"{arm}_s{seed}"
                arm_dir = output / name
                arm_dir.mkdir()
                pretrained, projection, metrics = pretrain_arm(seed, arm, images, targets, windows.donor.to_numpy(), len(vocab),
                                                               device, SETTINGS, arm_dir, deadline)
                if arm in ("A", "S"):
                    metrics["sensory_dev"] = {"model": sensory_readout(pretrained, dev_images, dev_targets, device, deadline),
                                               "references": references, "n_source_clips": len(dev_windows)}
                del pretrained
                model, clip_state, picture_state, language_metrics = transfer_arm(seed, projection, datasets, pairs, vocab,
                                      device, upstream, SETTINGS, arm_dir, deadline, score)
                metrics.update(language_metrics)
                keys = ("pretrain_initial_projection_sha256", "pretrain_order_sha256", "fresh_language_state_sha256", "language_order_sha256")
                hashes = {key: metrics[key] for key in keys}
                if seed in matched and hashes != matched[seed]:
                    raise ValueError("Common initialization or exposure order differs across arms")
                matched[seed] = hashes
                model.load_state_dict(clip_state)
                readouts = {}
                for split in ("clips_dev", "clips_test", "s8"):
                    _deadline(deadline)
                    readouts[split] = clip_score(model, split, True)
                credits[arm].append(readouts["clips_test"]["trials"]["picture_soft"])
                _json(arm_dir / "clip_readouts.json", readouts)
                model.load_state_dict(picture_state)
                _deadline(deadline)
                metrics["picture_readouts"] = {"rendered_test": float(region.eval_4afc_region(model, probe_cache, probe_lut,
                    probes[probes.split == "test"], vocab, device))}
                _deadline(deadline)
                metrics["picture_readouts"]["konkle"] = float(region.eval_4afc_region(model,
                    konkle_cache, konkle_lut, konkle, vocab, device))
                metrics.update(seed=seed, arm=arm, body_absent=True)
                _deadline(deadline)
                _json(arm_dir / "metrics.json", metrics)
                metadata["completed_runs"].append(name)
                _json(output / "run.json", metadata)
                del model
        _deadline(deadline)
        query = [r["row"] for r in frozen_trials["clips_test"]["0"]]
        if any([r["row"] for r in frozen_trials["clips_test"][str(s)]] != query for s in SETTINGS["seeds"]):
            raise ValueError("Seed trial rows do not align for paired cluster bootstrap")
        result = paired_summary(credits, pairs["clips_test"].iloc[query].target_clip.to_numpy())
        _verify_frozen(output, protocol)
        _deadline(deadline)
        result.update(completed=True, runtime_s=time.monotonic() - started, limits=LIMITS,
                      n_test_queries=len(query), n_target_clips=len(set(pairs["clips_test"].iloc[query].target_clip)),
                      final_frozen_verification="passed")
        _json(output / "comparison.json", result)
        metadata.update(status="complete", runtime_s=result["runtime_s"])
        _json(output / "run.json", metadata)
        return result
    except Exception as error:
        metadata.update(status="aborted_partial", error=f"{type(error).__name__}: {error}", runtime_s=time.monotonic() - started)
        try:
            _verify_frozen(output, protocol)
            metadata["final_frozen_verification"] = "passed"
        except Exception as verification_error:
            metadata["final_frozen_verification"] = f"failed: {type(verification_error).__name__}: {verification_error}"
        _json(output / "run.json", metadata)
        raise
