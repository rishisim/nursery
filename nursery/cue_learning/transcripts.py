"""Generate weakly aligned caregiver utterances and training-pair weights."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _output_directory, load_config


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


def session_scene(group, vocabulary, window):
    """Per-frame visible categories and held-window categories, plus session categories."""
    visible = [sorted({o["category"] for o in json.loads(objects) if o["visible_px"] > 0})
               for objects in group.objects]
    held = [None if pd.isna(c) else c for c in group.held_category]
    categories = set().union(*map(set, visible)) | {c for c in held if c is not None}
    missing = categories - vocabulary.keys()
    if missing:
        raise ValueError(f"Categories missing from vocabulary: {sorted(missing)}")
    n = len(group)
    windows = [sorted({c for c in held[max(0, t - window):t + window + 1] if c is not None})
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
            aligned = int(np.clip(source + jitter, 0, len(group) - 1))
            row = {"video_id": video, "frame_idx": aligned, "text": text,
                   "referential": referential, "referent": referent, "named": named,
                   "source_frame_idx": source, "utterance_time_s": float(time),
                   "jitter_frames": jitter, "channel": channel, "referent_visible": referent in visible[aligned],
                   "held_referent": referent is not None and referent == held,
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


def write_transcripts(episodes, output, seed, **overrides):
    episodes = Path(episodes).resolve()
    config = load_config()
    config["generator"].update({k: v for k, v in overrides.items() if v is not None})
    frames = pd.read_parquet(episodes / "frames.parquet")
    sessions = pd.read_parquet(episodes / "sessions.parquet")
    train_ids = sessions.loc[sessions.split == "train", "video_id"]
    frames = frames[frames.video_id.isin(train_ids)]
    manifest, donors = generate_transcripts(frames, config, seed)
    output = _output_directory(output)
    if (output / "manifest.parquet").exists():
        raise FileExistsError("Transcript manifest already exists; use a new ignored run directory")
    manifest.to_parquet(output / "manifest.parquet", index=False)
    record = {"episodes": str(episodes), "seed": seed, "config": config, "shuffled_donors": donors,
              "n_frames": len(frames), "n_utterances": len(manifest),
              "p_held_internal": manifest.attrs["p_held_internal"]}
    (output / "generator.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"transcripts: {len(manifest)} utterances / {len(frames)} train frames; "
          f"internal p_held {record['p_held_internal']:.4f}", flush=True)
    names = manifest[manifest.named].referent.value_counts().reindex(sorted(config["vocabulary"]), fill_value=0)
    rare = names[names < 5]
    print(f"named mentions per vocabulary word: min {names.min()}; under 5: "
          f"{rare.to_dict() if len(rare) else 'none'}", flush=True)
    return manifest
