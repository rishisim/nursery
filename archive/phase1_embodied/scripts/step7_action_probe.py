"""CPU-only Step 7 action probe on the frozen synthetic train/dev clip split."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nursery.learner.body import BodyEncoder, load_store, window_steps
from nursery.learner.video_only import TemporalVideoEncoder

FROZEN = REPO / "outputs/runs/step6_acquisition_curves/frozen"
OUT = REPO / "outputs/runs/step7_word_types/probe"
ACTIONS = ("lift", "pass", "inspect", "offhand", "drink")
SEEDS = range(5)
EPOCHS = 30


def read(path):
    return json.loads(path.read_text())


def load_inputs():
    protocol = read(FROZEN / "protocol.json")
    episodes = Path(protocol["episodes"])
    physical = episodes / "physical_instance"
    cache = Path(protocol["inputs"]) / "train_cache"
    splits = read(FROZEN / "splits.json")
    pools = {"train": splits["eligible_order_seed_0"], "dev": splits["source_keys"]["dev"]}
    partition = {clip: split for split, clips in pools.items() for clip in clips}
    assert len(partition) == sum(map(len, pools.values()))
    grab = Path(read(REPO / "nursery/simulator/config.json")["grab_root"])
    actions = {}
    for clip in sorted(partition):
        with np.load(grab / clip, allow_pickle=True) as source:
            action = str(source["motion_intent"])
        if action in ACTIONS:
            actions[clip] = ACTIONS.index(action)

    timeline = pd.read_parquet(episodes / "frames.parquet",
                               columns=["video_id", "frame_idx", "clip", "held_category",
                                        "contact_left", "contact_right"])
    timeline = timeline.sort_values(["video_id", "frame_idx"])
    video_ids = sorted(timeline.loc[timeline["clip"].isin(actions), "video_id"].unique())
    timeline = timeline[timeline.video_id.isin(video_ids)]
    fit = timeline[timeline["clip"].isin(pools["train"])]
    session = {video: i for i, video in enumerate(video_ids)}
    exclude = np.ones((len(video_ids), 120), dtype=bool)
    exclude[fit.video_id.map(session), fit.frame_idx] = False
    store = load_store(episodes, timeline, video_ids, set(), exclude=exclude, physical=physical)
    assert sum(group.shape[-1] for group in store["groups"].values()) == 153

    # One center per replay occurrence. Prefer the middle contact sample;
    # very short replays can have no contact at the recorded 1 Hz samples.
    run = (timeline.video_id.ne(timeline.video_id.shift()) |
           timeline["clip"].ne(timeline["clip"].shift()) |
           timeline.frame_idx.diff().ne(1)).cumsum()
    records = []
    for _, rows in timeline[timeline["clip"].isin(actions)].groupby(run, sort=False):
        contact = rows[rows.contact_left | rows.contact_right]
        eligible = contact if len(contact) else rows
        center = eligible.iloc[(len(eligible) - 1) // 2]
        records.append((center.video_id, center.frame_idx, center["clip"], bool(len(contact))))
    rows = pd.DataFrame(records, columns=["video_id", "frame_idx", "clip", "contact"])
    rows["partition"] = rows["clip"].map(partition)
    rows["label"] = rows["clip"].map(actions)
    assert set(rows["clip"]) == set(actions)
    s, t = rows.video_id.map(session).to_numpy(), rows.frame_idx.to_numpy()
    steps, seg = window_steps(store["clip"], s, t, 5)
    same = (seg == seg[:, 5:6]) & (seg >= 0)
    grid = {name: torch.from_numpy(group[s[:, None], steps]) for name, group in store["groups"].items()}
    seg = torch.from_numpy(np.where(same, seg, -2))
    same = torch.from_numpy(same)

    index = pd.read_parquet(cache / "index.parquet")
    lookup = {(row.video_id, row.frame_idx): row.row for row in index.itertuples()}
    locations = np.full(steps.shape, -1)
    for i, j in zip(*np.where(same.numpy())):
        locations[i, j] = lookup[(video_ids[s[i]], steps[i, j])]
    bank = np.load(cache / "emb.f16.npy", mmap_mode="r")
    assert bank.shape[1:] == (257, 768)
    unique = np.unique(locations[locations >= 0])
    pooled = torch.empty((len(unique), 768))
    for start in range(0, len(unique), 64):
        ids = unique[start:start + 64]
        pooled[start:start + len(ids)] = torch.from_numpy(bank[ids].astype(np.float32)).mean(dim=1)
    video = torch.zeros((*steps.shape, 768))
    video[same] = pooled[np.searchsorted(unique, locations[same.numpy()])]
    return rows, grid, seg, same, video


def probe(rows, groups, seg, same, video, seed, modality):
    torch.manual_seed(seed)
    order = torch.Generator().manual_seed(seed)
    train = torch.from_numpy(np.flatnonzero(rows.partition.eq("train")))
    dev = torch.from_numpy(np.flatnonzero(rows.partition.eq("dev")))
    y = torch.tensor(rows.label.to_numpy())
    counts = rows.groupby("clip")["clip"].transform("size").to_numpy()
    sample_weight = torch.from_numpy(1 / counts).float()
    if modality == "video":
        encoder = TemporalVideoEncoder(video.shape[-1])

        def encode(ids):
            return encoder(video[ids], same[ids])
    else:
        encoder = BodyEncoder({name: group.shape[-1] for name, group in groups.items()})

        def encode(ids):
            return encoder({name: group[ids] for name, group in groups.items()},
                           (~same[ids]).float(), seg[ids])[:, 5]

    head = nn.Linear(128, len(ACTIONS))
    optimizer = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=1e-3, weight_decay=.1)
    for _ in range(EPOCHS):
        for ids in train[torch.randperm(len(train), generator=order)].split(128):
            loss = F.cross_entropy(head(encode(ids)), y[ids], reduction="none")
            loss = (loss * sample_weight[ids]).sum() / sample_weight[ids].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = torch.cat([head(encode(ids)) for ids in dev.split(128)]).numpy()
    clips = rows.iloc[dev.numpy()]["clip"].to_numpy()
    correct = []
    for clip in np.unique(clips):
        picks = clips == clip
        correct.append(logits[picks].mean(0).argmax() == y[dev[picks]][0].item())
    return np.array(correct)


def main():
    torch.set_num_threads(4)
    rows, groups, seg, same, video = load_inputs()
    print(rows.groupby("partition").agg(replays=("clip", "size"), clips=("clip", "nunique")), flush=True)
    scores = {name: [] for name in ("video", "body")}
    for seed in SEEDS:
        for name in scores:
            correct = probe(rows, groups, seg, same, video, seed, name)
            scores[name].append(correct)
            print(f"seed={seed} {name} dev={100 * correct.mean():.2f}%", flush=True)

    # Paired bootstrap resamples training seeds and held-out source clips.
    rng = np.random.default_rng(7)
    arrays = {name: np.array(value, dtype=float) for name, value in scores.items()}
    arrays["body_minus_video"] = arrays["body"] - arrays["video"]
    seeds, clips = arrays["video"].shape
    seed_draws = rng.integers(seeds, size=(10000, seeds))
    clip_draws = rng.integers(clips, size=(10000, clips))
    results = {}
    for name, values in arrays.items():
        bootstrap = values[seed_draws[:, :, None], clip_draws[:, None, :]].mean((1, 2)) * 100
        results[name] = {"mean_pct": 100 * values.mean(),
                         "ci95_pct": np.quantile(bootstrap, [.025, .975]).tolist(),
                         "per_seed_pct": (100 * values.mean(1)).tolist()}
    passed = results["body_minus_video"]["mean_pct"] <= 10
    report = {"results": results, "gate_passed": bool(passed), "epochs": EPOCHS,
              "seeds": list(SEEDS), "train_clips": int(rows[rows.partition.eq("train")]["clip"].nunique()),
              "dev_clips": clips, "replays_without_sampled_contact": int((~rows.contact).sum()),
              "mean_valid_steps": same.sum(1).float().mean().item(),
              "physical": "outputs/renders/clean_856/physical_instance"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
