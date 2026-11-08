"""Generate weakly aligned caregiver utterances and training-pair weights."""

import json
import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from nursery.common import _output_directory, load_config


def cue_timelines(frames, config, rng):
    groups = {v: g.sort_values("frame_idx").reset_index(drop=True)
              for v, g in frames.groupby("video_id", sort=True)}
    cohorts = {}
    for video, group in groups.items():
        if not np.array_equal(group.frame_idx, np.arange(len(group))):
            raise ValueError("Cue timelines require contiguous 1-Hz frames starting at zero")
        cohorts.setdefault(len(group), []).append(video)
    contact = {v: (g.contact_left | g.contact_right).to_numpy(dtype=bool) for v, g in groups.items()}
    donors = {}
    for videos in cohorts.values():
        if len(videos) < 2:
            raise ValueError("Shuffled cues need two train sessions of each length to preserve marginals")
        order = rng.permutation(videos).tolist()
        donors.update({v: order[(i + 1) % len(order)] for i, v in enumerate(order)})
    cues = {}
    for video, values in contact.items():
        indices = np.arange(len(values))
        cues[video] = {
            "contact": values,
            "shuffled": contact[donors[video]][indices % len(contact[donors[video]])],
            "shifted": values[(indices + config["shift_s"]) % len(values)],
        }
        for eps in config["noise_eps"]:
            cues[video][f"noisy_{eps:g}"] = values ^ (rng.random(len(values)) < eps)
    return cues, donors


def clip_bounds(group):
    """Inclusive bounds of each contiguous source clip; repeated IDs start new segments."""
    if "clip" not in group or not group["clip"].map(lambda c: isinstance(c, str) and bool(c.strip())).all():
        raise ValueError("Speech generation requires a nonempty string clip identity on every frame")
    starts = np.flatnonzero(np.r_[True, group["clip"].to_numpy()[1:] != group["clip"].to_numpy()[:-1]])
    ends = np.r_[starts[1:] - 1, len(group) - 1]
    lengths = ends - starts + 1
    return np.repeat(starts, lengths), np.repeat(ends, lengths)


def session_scene(group, vocabulary, window):
    """Per-frame visible and clip-local held-window categories, plus session categories."""
    starts, ends = clip_bounds(group)
    visible = [sorted({o["category"] for o in json.loads(objects) if o["visible_px"] > 0})
               for objects in group.objects]
    held = [None if pd.isna(c) else c for c in group.held_category]
    categories = set().union(*map(set, visible)) | {c for c in held if c is not None}
    missing = categories - vocabulary.keys()
    if missing:
        raise ValueError(f"Categories missing from vocabulary: {sorted(missing)}")
    n = len(group)
    windows = [sorted({c for c in held[max(starts[t], t - window):min(ends[t], t + window) + 1]
                       if c is not None})
               for t in range(n)]
    return visible, held, windows, categories


def channel_candidates(visible, windows, categories, t):
    near = set(windows[t])
    return {"held": windows[t],
            "unseen": sorted(categories - set(visible[t]) - near),
            "seen": [c for c in visible[t] if c not in near]}


def calibrate_p_held(scenes, parameters):
    """Internal held share so the achieved share of referents equals p_held after redraws."""
    availability = []
    for visible, _, windows, categories in scenes:
        for t in range(len(visible)):
            cand = channel_candidates(visible, windows, categories, t)
            availability.append([bool(cand[k]) for k in ("held", "unseen", "seen")])
    available = np.array(availability, dtype=bool)
    unseen = parameters["p_unseen"]

    def achieved(h):
        shares = np.array([h, unseen, 1 - h - unseen])
        total = (available * shares).sum(axis=1)
        ok = total > 0
        return float((available[ok, 0] * h / total[ok]).sum() / ok.sum()) if ok.any() else 0.0

    target, lo, hi = parameters["p_held"], 0.0, 1 - unseen
    if achieved(hi) < target:
        print(f"warning: p_held {target} unreachable; using internal {target:.4f}", flush=True)
        return target
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if achieved(mid) < target else (lo, mid)
    return (lo + hi) / 2


def generate_transcripts(frames, config, seed):
    """Sample weak speech without crossing source-clip cuts.

    source_frame_idx and utterance_time_s retain the original speech time.
    jitter_draw_frames is the rounded Gaussian draw; jitter_frames is the effective
    aligned-minus-source offset. Clamping puts boundary mass at clip endpoints.
    held_referent describes the aligned frame; source_held_referent describes the
    source frame. Neither flag filters speech or forces names to be held objects.
    """
    parameters = config["generator"]
    for key in ("p_ref", "p_name", "p_held", "p_unseen"):
        if not 0 <= parameters[key] <= 1:
            raise ValueError(f"{key} must be a probability")
    if parameters["p_held"] + parameters["p_unseen"] > 1:
        raise ValueError("p_held + p_unseen must be at most 1")
    if not isinstance(parameters["hold_window_s"], int) or parameters["hold_window_s"] < 0:
        raise ValueError("hold_window_s must be a nonnegative integer")
    if parameters["rate_hz"] <= 0 or parameters["sigma"] < 0:
        raise ValueError("Utterance rate must be positive and jitter nonnegative")
    if (config["cues"]["w_lo"] <= 0 or config["cues"]["w_hi"] < config["cues"]["w_lo"] or
            any(not 0 <= eps <= 1 for eps in config["cues"]["noise_eps"]) or
            not isinstance(config["cues"]["shift_s"], int)):
        raise ValueError("Invalid cue weights, noise probabilities, or integer shift")
    pool = json.loads(Path(__file__).with_name("frames.json").read_text())
    # Separate random streams keep the speech identical when cue settings change.
    speech_seed, cue_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(speech_seed)
    cues, donors = cue_timelines(frames, config["cues"], np.random.default_rng(cue_seed))
    scenes = {}
    for video, group in frames.groupby("video_id", sort=True):
        group = group.sort_values("frame_idx").reset_index(drop=True)
        scenes[video] = (group, *session_scene(group, config["vocabulary"], parameters["hold_window_s"]))
    p_held = calibrate_p_held([s[1:] for s in scenes.values()], parameters)
    shares = {"held": p_held, "unseen": parameters["p_unseen"],
              "seen": 1 - p_held - parameters["p_unseen"]}
    rows = []
    for video, (group, visible, held_all, windows, categories) in scenes.items():
        starts, ends = clip_bounds(group)
        time = rng.exponential(1 / parameters["rate_hz"])
        while time < len(group):
            source = min(int(np.floor(time + 0.5)), len(group) - 1)
            referential = bool(rng.random() < parameters["p_ref"])
            referent, named, channel = None, False, None
            held = held_all[source]
            if referential:
                cand = channel_candidates(visible, windows, categories, source)
                open_channels = [k for k in shares if cand[k]]
                total = sum(shares[k] for k in open_channels)
                if total > 0:
                    draw = rng.random() * total
                    channel = open_channels[-1]
                    for k in open_channels:
                        if draw < shares[k]:
                            channel = k
                            break
                        draw -= shares[k]
                    referent = str(rng.choice(cand[channel]))
                else:
                    referential = False
            if referential:
                named = bool(rng.random() < parameters["p_name"])
                templates = pool["named"] if named else pool["pronoun"]
                text = str(rng.choice(templates))
                if named:
                    text = text.format(noun=config["vocabulary"][referent])
            else:
                text = str(rng.choice(pool["nonreferential"]))
            jitter = int(np.rint(rng.normal(0, parameters["sigma"])))
            aligned = int(np.clip(source + jitter, starts[source], ends[source]))
            row = {"video_id": video, "frame_idx": aligned, "text": text,
                   "referential": referential, "referent": referent, "named": named,
                   "source_frame_idx": source, "utterance_time_s": float(time),
                   "jitter_frames": aligned - source, "channel": channel,
                   "referent_visible": referent in visible[aligned],
                   "held_referent": referent is not None and referent == held_all[aligned],
                   "source_held_referent": referent is not None and referent == held,
                   "jitter_draw_frames": jitter,
                   "weight_none": 1.0}
            indicators = {name: values[aligned] for name, values in cues[video].items()}
            indicators["oracle"] = referential and named and row["referent_visible"]
            for name, high in indicators.items():
                row[f"weight_{name}"] = config["cues"]["w_hi" if high else "w_lo"]
            rows.append(row)
            time += rng.exponential(1 / parameters["rate_hz"])
    if not rows:
        raise ValueError("No utterances sampled; increase session duration or rate")
    manifest = pd.DataFrame(rows)
    manifest.attrs["p_held_internal"] = p_held
    for column in manifest.filter(like="weight_"):
        manifest[column] /= manifest[column].mean()
    return manifest, donors


def couple_gaze(manifest, frames, gaze, config, seed, p_gazed_seen=0.55):
    """Retarget an existing speech schedule after scene-only gaze is frozen.

    Calibrate four channel weights against candidate availability at the fixed
    source times. This allows naming-channel choice to respond to attention;
    gaze and utterance times never respond to naming. The held channel retains
    the original clip-local +/- hold_window_s semantics, not instant holding.
    """
    if not 0 <= p_gazed_seen <= 1:
        raise ValueError("p_gazed_seen must be a probability")
    if gaze.duplicated(["video_id", "frame_idx"]).any():
        raise ValueError("Duplicate gaze keys")
    scenes = {}
    gazes = gaze.set_index(["video_id", "frame_idx"])
    for video, group in frames.groupby("video_id", sort=True):
        group = group.sort_values("frame_idx").reset_index(drop=True)
        scenes[video] = session_scene(group, config["vocabulary"], config["generator"]["hold_window_s"])
    result = manifest.copy(deep=True)
    # A third independent seed stream cannot perturb either original speech or cues.
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
    candidates, indices = [], []
    for index, row in result.iterrows():
        if not row.referential:
            continue
        visible, held, windows, categories = scenes[row.video_id]
        source = int(row.source_frame_idx)
        cand = channel_candidates(visible, windows, categories, source)
        looked = gazes.loc[(row.video_id, source), "gaze_category"]
        gazed = [looked] if pd.notna(looked) and looked in cand["seen"] else []
        candidates.append([cand["held"], cand["unseen"], gazed,
                           [c for c in cand["seen"] if c not in gazed]])
        indices.append(index)
    if not candidates:
        raise ValueError("No referential utterances to couple")
    available = np.asarray([[bool(c) for c in row] for row in candidates], dtype=bool)
    parameters = config["generator"]
    seen = 1 - parameters["p_held"] - parameters["p_unseen"]
    target = np.array([parameters["p_held"], parameters["p_unseen"],
                       seen * p_gazed_seen, seen * (1 - p_gazed_seen)])
    if (target < 0).any() or not available.any(axis=1).all():
        raise ValueError("Invalid gaze speech targets or empty candidate rows")
    weights = target.copy()
    for _ in range(2000):
        probabilities = available * weights
        totals = probabilities.sum(axis=1)
        if (totals <= 0).any():
            raise ValueError("Gaze speech shares are infeasible at fixed utterance times")
        probabilities /= totals[:, None]
        achieved = probabilities.mean(axis=0)
        if np.max(np.abs(achieved - target)) < 1e-8:
            break
        if ((achieved == 0) & (target > 0)).any():
            raise ValueError("Gaze speech target has no eligible source moments")
        weights *= np.divide(target, achieved, out=np.ones(4), where=achieved > 0)
        weights /= weights.sum()
    else:
        raise ValueError(f"Gaze speech shares infeasible: target {target.tolist()}, achieved {achieved.tolist()}")
    choices = ["held", "unseen", "seen", "seen"]
    for position, index in enumerate(indices):
        row = result.loc[index]
        draw = int(rng.choice(4, p=probabilities[position]))
        referent = str(rng.choice(candidates[position][draw]))
        if row.named:
            old = config["vocabulary"][row.referent]
            text, count = re.subn(r"\b" + re.escape(old) + r"\b", config["vocabulary"][referent], row.text, count=1)
            if count != 1:
                raise ValueError("Named speech template does not contain its recorded noun")
            result.at[index, "text"] = text
        result.at[index, "referent"] = referent
        result.at[index, "channel"] = choices[draw]
    source_matches, aligned_matches = [], []
    for index, row in result.iterrows():
        visible, held, _, _ = scenes[row.video_id]
        source, aligned = int(row.source_frame_idx), int(row.frame_idx)
        result.at[index, "referent_visible"] = row.referent in visible[aligned]
        result.at[index, "held_referent"] = row.referent is not None and row.referent == held[aligned]
        result.at[index, "source_held_referent"] = row.referent is not None and row.referent == held[source]
        source_gaze = gazes.loc[(row.video_id, source), "gaze_category"]
        aligned_gaze = gazes.loc[(row.video_id, aligned), "gaze_category"]
        source_matches.append(bool(pd.notna(source_gaze) and row.referent == source_gaze))
        aligned_matches.append(bool(pd.notna(aligned_gaze) and row.referent == aligned_gaze))
    result["source_gazed_referent"] = source_matches
    result["gazed_referent"] = aligned_matches
    high = result.referential & result.named & result.referent_visible
    oracle = np.where(high, config["cues"]["w_hi"], config["cues"]["w_lo"])
    result["weight_oracle"] = oracle / oracle.mean()
    report = {"p_gazed_seen_target": p_gazed_seen,
              "four_channel_targets": dict(zip(["held", "unseen", "gazed_seen", "other_seen"], target.tolist())),
              "four_channel_internal_weights": weights.tolist(),
              "four_channel_expected_shares": achieved.tolist(),
              "source_gazed_seen_opportunity": float(available[:, 2].mean()),
              "schedule_and_named_flags_preserved": True}
    for label, selected in (("referential", result[result.referential]), ("named", result[result.named])):
        seen_rows = selected[selected.channel == "seen"]
        report[label] = {"n": len(selected),
                         "channel_shares": selected.channel.value_counts(normalize=True).to_dict(),
                         "source_gazed_given_seen": float(seen_rows.source_gazed_referent.mean()),
                         "aligned_gazed_given_seen": float(seen_rows.gazed_referent.mean()),
                         "source_physically_held": float(selected.source_held_referent.mean()),
                         "aligned_physically_held": float(selected.held_referent.mean())}
    return result, report


def write_transcripts(episodes, output, seed, gaze=None, source_manifest=None, **overrides):
    episodes = Path(episodes).resolve()
    config = load_config()
    config["generator"].update({k: v for k, v in overrides.items() if v is not None})
    frames = pd.read_parquet(episodes / "frames.parquet")
    sessions = pd.read_parquet(episodes / "sessions.parquet")
    train_ids = sessions.loc[sessions.split == "train", "video_id"]
    frames = frames[frames.video_id.isin(train_ids)]
    manifest, donors = generate_transcripts(frames, config, seed)
    coupling = None
    if source_manifest is not None:
        if gaze is None:
            raise ValueError("A source manifest is only used for gaze coupling")
        original = pd.read_parquet(source_manifest)
        try:
            pd.testing.assert_frame_equal(original[manifest.columns].reset_index(drop=True),
                                          manifest.reset_index(drop=True), check_dtype=False)
        except (AssertionError, KeyError) as exc:
            raise ValueError("Source speech does not match this generator seed/config/frame input") from exc
        # The supplied old speech is authoritative for the exact retained schedule.
        manifest = original
    if gaze is not None:
        manifest, coupling = couple_gaze(manifest, frames, pd.read_parquet(gaze), config, seed)
        achieved = coupling["named"]
        if not .50 <= achieved["source_gazed_given_seen"] <= .60:
            raise ValueError("Achieved named visible-channel gaze coupling misses 50-60% gate")
        if any(abs(achieved["channel_shares"].get(channel, 0) - config["generator"][parameter]) > .02
               for channel, parameter in (("held", "p_held"), ("unseen", "p_unseen"))):
            raise ValueError("Achieved named held/unseen channel shares miss two-percentage-point gate")
    output = _output_directory(output)
    if (output / "manifest.parquet").exists():
        raise FileExistsError("Transcript manifest already exists; use a new ignored run directory")
    manifest.to_parquet(output / "manifest.parquet", index=False)
    record = {"episodes": str(episodes), "seed": seed, "config": config, "shuffled_donors": donors,
              "n_frames": len(frames), "n_utterances": len(manifest),
              "p_held_internal": manifest.attrs.get("p_held_internal"), "gaze_coupling": coupling}
    if gaze is not None:
        record.update(execution_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      speech_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name, path in (("gaze", gaze), ("source_manifest", source_manifest)):
        if path is not None:
            record[name] = {"path": str(Path(path).resolve()),
                            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    (output / "generator.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"transcripts: {len(manifest)} utterances / {len(frames)} train frames; "
          f"internal p_held {record['p_held_internal']}", flush=True)
    names = manifest[manifest.named].referent.value_counts().reindex(sorted(config["vocabulary"]), fill_value=0)
    rare = names[names < 5]
    print(f"named mentions per vocabulary word: min {names.min()}; under 5: "
          f"{rare.to_dict() if len(rare) else 'none'}", flush=True)
    return manifest
