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


def generate_transcripts(frames, config, seed):
    parameters = config["generator"]
    for key in ("p_ref", "q_held", "p_name"):
        if not 0 <= parameters[key] <= 1:
            raise ValueError(f"{key} must be a probability")
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
    rows = []
    for video, group in frames.groupby("video_id", sort=True):
        group = group.sort_values("frame_idx").reset_index(drop=True)
        visible = [sorted({o["category"] for o in json.loads(objects) if o["visible_px"] > 0})
                   for objects in group.objects]
        categories = set().union(*map(set, visible)) | set(group.held_category.dropna())
        missing = categories - config["vocabulary"].keys()
        if missing:
            raise ValueError(f"Categories missing from vocabulary: {sorted(missing)}")
        time = rng.exponential(1 / parameters["rate_hz"])
        while time < len(group):
            source = min(int(np.floor(time + 0.5)), len(group) - 1)
            referential = bool(rng.random() < parameters["p_ref"])
            referent, named = None, False
            held = group.iloc[source].held_category
            held = None if pd.isna(held) else held
            if referential:
                other = [c for c in visible[source] if c != held]
                if held is not None and rng.random() < parameters["q_held"]:
                    referent = held
                elif other:
                    referent = str(rng.choice(other))
                else:
                    raise ValueError("Referential sampling needs a visible non-held candidate on this frame")
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
                   "jitter_frames": jitter, "referent_visible": referent in visible[aligned],
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
              "n_frames": len(frames), "n_utterances": len(manifest)}
    (output / "generator.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"transcripts: {len(manifest)} utterances / {len(frames)} train frames", flush=True)
    return manifest
