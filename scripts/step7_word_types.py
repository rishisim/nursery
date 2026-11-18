"""Build the CPU-only data for Step 7 from the frozen Step 6 inputs."""

import argparse
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd

from nursery.common import _output_directory
from nursery.learner import body, evaluate
from nursery.learner import _upstream_module, acquisition as run
from nursery.learner.acquisition_data import _panel_donors, _sha, _source_keys, _write
from nursery.simulator.build import DEFAULT_CONFIG
from nursery.simulator.physical import LIFT_M, build_physical
from nursery.simulator.sessions import contact_rule, sample_chain
from nursery.speech.transcripts import add_event_words

RENDER = Path("outputs/renders/clean_856")
PHYSICAL = RENDER / "physical_instance"
FROZEN = Path("outputs/runs/step6_acquisition_curves/frozen")
MASS_CSV = Path("outputs/step7_scouting/instance_draw_mass.csv")
OUTPUT = Path("outputs/runs/step7_word_types")
ACTIONS = ("pass", "lift", "inspect", "offhand", "drink")
PROPERTIES = ("heavy", "light")
SPLITS = ("train", "dev", "test", "s8")
KEY = ["video_id", "frame_idx"]
ARMS = {
    "video": {"body_representation": None},
    "video_gaze": {"body_representation": None},
    "video_gaze_body_current": {"body_representation": "current", "body_mode": "real"},
    run.SHUFFLED_BODY: {"body_representation": "current", "body_mode": "shuffled"},
}
WORD_TYPE_PANELS = {
    "noun": {"dev": "clips_dev", "held_out": {"test": "clips_test"}},
    "action": {"dev": "action_dev", "held_out": {"test": "action_test", "s8": "action_s8"},
               "subset": "object_matched", "cluster": "clip"},
    "property": {"dev": "property_dev", "held_out": {"test": "property_test", "s8": "property_s8"},
                 "subset": "straddling_noun", "cluster": "instance_key"},
}
EXPECTED_TRIALS = {"clips_test": 556, "action_test": 1049, "action_s8": 876,
                   "property_test": 859, "property_s8": 769}


def prepare_learner(destination, *, clips=None, baseline=False):
    """Reuse the Step 6 split, fit mask and cache; retokenize the selected speech."""
    from nursery.learner.train import split_pairs

    parent = run._read(FROZEN / "protocol.json")
    parent_subset = FROZEN / "subsets/1.0"
    split_path = OUTPUT / "splits.json"
    if split_path.read_bytes() != (FROZEN / "splits.json").read_bytes():
        raise ValueError("Step 7 split differs from Step 6")
    split = run._read(split_path)
    ordered = split["eligible_order_seed_0"]
    if len(ordered) != 630 or (clips is not None and not 2 <= clips < 630):
        raise ValueError("Expected 630 frozen training clips or an explicit tiny smoke subset")
    chosen = set(ordered[:clips])
    frames = pd.read_parquet(RENDER / "frames.parquet")
    frames["clip"] = _source_keys(frames)
    with np.load(parent_subset / "fit_mask.npz", allow_pickle=False) as fit:
        ids, parent_include = fit["video_ids"], fit["include"]
    subject_clip = frames.set_index(KEY)[["subject", "clip"]]
    positions = {video: i for i, video in enumerate(ids)}
    include = np.zeros_like(parent_include)
    for frame in frames[frames["clip"].isin(chosen)].itertuples():
        include[positions[frame.video_id], frame.frame_idx] = True
    if not np.all(~include | parent_include) or (clips is None and not np.array_equal(include, parent_include)):
        raise ValueError("Fit mask differs from the frozen Step 6 training clips")
    # Same five-frame selection rule as Step 6, with the frozen fit mask
    # distinguishing other training clips from the selected ones.
    partition = np.full(parent_include.shape, -1, dtype=int)
    for frame in frames.itertuples():
        partition[positions[frame.video_id], frame.frame_idx] = 1
    partition[include] = 0
    transcript = (Path(parent["inputs"]) if baseline else OUTPUT) / "manifest.parquet"
    speech = pd.read_parquet(transcript)
    labels = split_pairs(speech, dict(zip(ids, partition)), parent["window"])
    raw = speech[(labels == 0) & speech.text.fillna("").str.strip().ne("")].reset_index(drop=True)
    metadata = subject_clip.reindex(pd.MultiIndex.from_frame(raw[KEY]))
    if not baseline and (not raw["split"].eq("train").all() or not np.array_equal(raw["clip"], metadata["clip"])):
        raise ValueError("Step 7 utterance metadata disagrees with the frozen training split")
    upstream = _upstream_module("train_frame_mil")
    vocab = upstream.build_vocab(raw.text, 5)
    manifest = raw[raw.text.map(lambda text: bool(upstream.encode(text, vocab)))].reset_index(drop=True)
    manifest["subject"] = subject_clip.reindex(pd.MultiIndex.from_frame(manifest[KEY]))["subject"].to_numpy()
    if len(manifest) < 2:
        raise ValueError("Too few vocabulary-bearing examples")
    if clips is None and not set((*ACTIONS, *PROPERTIES)) <= vocab.keys():
        raise ValueError("Full training vocabulary lost an approved word")
    destination = _output_directory(destination, exist_ok=False)
    panels = FROZEN / "panels" if baseline else OUTPUT / "panels"
    shutil.copytree(panels, destination / "panels")
    if not baseline:
        for panel, count in EXPECTED_TRIALS.items():
            trial_path = destination / "panels" / panel / "trials.json"
            if len(run._read(trial_path)["trial_rows"]) != count:
                raise ValueError(f"Unexpected trial count: {panel}")
        for panel in ("clips_dev", "clips_test"):
            for path in (panels / panel).iterdir():
                if _sha(path) != _sha(FROZEN / "panels" / panel / path.name):
                    raise ValueError(f"Step 6 noun panel changed: {path}")
    shutil.copyfile(split_path, destination / "splits.json")
    subset = destination / "subsets/1.0"
    subset.mkdir(parents=True)
    manifest.to_parquet(subset / "manifest.parquet", index=False)
    np.savez_compressed(subset / "fit_mask.npz", video_ids=ids, include=include)
    run._write(subset / "vocab.json", vocab)
    stats = dict(selected_clips=sorted(chosen), selected_clip_count=len(chosen),
                 utterances=len(manifest), utterances_before_vocab=len(raw), replay_exposure_seconds=int(include.sum()))
    run._write(subset / "stats.json", stats)
    if baseline:
        protocol = parent
    else:
        protocol = {k: v for k, v in parent.items() if k not in
                    ("rich_body_manifest", "picture_panels", "revision", "code_hashes")}
        protocol.update(physical=str(PHYSICAL.resolve()), temporal_video=True, word_type_panels=WORD_TYPE_PANELS,
                        arms=ARMS, conditions=list(ARMS), seeds=list(range(5)), fractions=[1.0], source_commit=None,
                        transcripts=str(transcript.resolve()), expected_trial_counts=EXPECTED_TRIALS,
                        body_input_dims={"current": 153}, approval="Draft protocol — 17 November 2026")
        protocol["selection"] = dict(panel="equal mean of noun, action and property dev accuracies",
                                     body_weight_grid=list(run.BODY_WEIGHT_GRID),
                                     body_rule=parent["selection"]["body_rule"],
                                     non_body_rule="earliest epoch at maximum mean dev accuracy",
                                     test_panels_select=False)
        protocol["rng"] = dict(parent["rng"], body_shuffle="training seed, within subject and across source clips; "
                               "within each dev/test panel, fixed seed 0 from frozen donor rows")
        inputs = Path(parent["inputs"])
        gaze = parent["gaze"]
        keep = [inputs / "train_cache" / name for name in ("emb.f16.npy", "index.parquet")]
        keep += [RENDER / name for name in ("frames.parquet", "sessions.parquet")]
        keep += [Path(gaze[k]) for k in ("points", "partitions")]
        keep += [Path(gaze["closeup_cache"]) / name for name in ("emb.f16.npy", "index.parquet")]
        protocol["hashes"] = {str(p.absolute()): parent["hashes"][str(p.absolute())] for p in keep}
        protocol["hashes"][str(transcript.resolve())] = _sha(transcript)
        protocol["hashes"].update({str(p.resolve()): _sha(p) for p in PHYSICAL.glob("*.npz")})
        protocol["panel_hashes"] = {str(p.relative_to(destination)): _sha(p)
                                    for p in sorted((destination / "panels").rglob("*")) if p.is_file()}
        protocol["code_hashes"] = run._source_snapshot()["code_hashes"]
        for path in (Path(__file__), Path("scripts/step4_rental.py"), Path("scripts/child_gaze_rental.py")):
            protocol["code_hashes"][str(path.resolve())] = _sha(path)
    protocol.update(subsets={"1.0": stats}, fractions=[1.0])
    if clips is not None:
        protocol.update(epochs=1, seeds=[0], smoke_evaluate_all_panels=True, device="cpu")
    run._write(destination / "protocol.json", protocol)
    return protocol


def freeze_learner():
    from scripts.step4_rental import jobs

    destination = OUTPUT / "frozen"
    protocol = prepare_learner(destination)
    gaps = closeup_gaps(destination, protocol)
    missing = set().union(*(set().union(*rows) for rows in gaps.values()))
    coverage = {"ready": not missing, "missing_closeup_frames": len(missing),
                "missing_by_panel_or_training": {name: len(set().union(*rows)) for name, rows in gaps.items()}}
    protocol["gaze_coverage"] = coverage
    run._write(destination / "protocol.json", protocol)
    run._write(destination / "closeup_coverage.json", dict(coverage, missing_frame_keys=sorted(missing)))
    signals = run._read(FROZEN / "signal_hashes.json")
    body_dir = (RENDER / "body").resolve()
    run._write(destination / "signal_hashes.json", {p: h for p, h in signals.items() if Path(p).parent == body_dir})
    run._write(destination / "queue.json", sorted(jobs(protocol), key=lambda job: (job[1], protocol["conditions"].index(job[2]))))
    run._freeze_inputs(destination.resolve(), audit_preparation=False)


def closeup_gaps(root, protocol):
    """Find gaze-valid frames missing their cached sharp view in either window."""
    from nursery.learner.gaze_data import TemporalVideoData, partition_window_keys

    gaze = protocol["gaze"]
    points = pd.read_parquet(gaze["points"]).dropna(subset=["gaze_x"])
    sharp = pd.read_parquet(Path(gaze["closeup_cache"]) / "index.parquet")
    frame_keys = lambda table: set(table.video_id + "|" + table.frame_idx.astype(str))
    missing = frame_keys(points) - frame_keys(sharp)
    index = pd.read_parquet(Path(protocol["inputs"]) / "train_cache/index.parquet")
    lut = {f"{r.video_id}|{r.frame_idx}": (0, int(r.row)) for r in index.itertuples()}
    partitions = run._read(gaze["partitions"])
    with np.load(root / "subsets/1.0/fit_mask.npz") as fit:
        ids, include = fit["video_ids"], fit["include"]
    frames = pd.read_parquet(RENDER / "frames.parquet", columns=["video_id", "frame_idx", "clip", "held_category"])
    clips, *_ = body.clip_index(frames, ids, include.shape[1])
    timeline = {vid: i for i, vid in enumerate(ids)}, clips
    paths = {"training": root / "subsets/1.0/manifest.parquet"}
    for spec in WORD_TYPE_PANELS.values():
        for name in (spec["dev"], *spec["held_out"].values()):
            paths[name] = root / "panels" / name / "manifest.parquet"
    result = {}
    for name, path in paths.items():
        rows = pd.read_parquet(path, columns=KEY)
        temporal = TemporalVideoData(rows, [], lut, timeline)
        ordinary = partition_window_keys(rows, lut, partitions, protocol["window"])
        result[name] = [missing.intersection(near + long) for near, long in zip(ordinary, temporal.frame_keys)]
    return result


def smoke_learner(destination, clips, *, baseline=False):
    import torch

    torch.set_num_threads(2)
    protocol = prepare_learner(destination, clips=clips, baseline=baseline)
    for arm in protocol["conditions"]:
        print(f"Running {arm}", flush=True)
        metrics = run.train_one(destination, 1.0, 0, arm, smoke=True, device="cpu")
        print(f"{arm}: {metrics['epoch_seconds']:.2f} s/epoch", flush=True)


def load_events(frames, masses):
    config = json.loads(Path(DEFAULT_CONFIG).read_text())
    with Path(config["hand_vertex_ids"]).open("rb") as handle:
        hand_ids = pickle.load(handle)
    clips = {}
    seen = {}
    for video, group in frames.groupby("video_id", sort=True):
        with np.load(PHYSICAL / f"{video}.npz") as data, np.load(
                RENDER / "physical_by_clip" / f"{video}.npz") as old:
            if set(data.files) != set(old.files) or "contact_area" not in data:
                raise ValueError(f"Physical store fields differ for {video}")
            np.testing.assert_array_equal(data["frame_idx"], group.frame_idx)
            keys = data["instance_key"]
            values = data["mass_kg"]
            np.testing.assert_array_equal(values, (masses.loc[keys, "mass_g"] / 1000).to_numpy(np.float32))
            np.testing.assert_array_equal(data["material"], masses.loc[keys, "material"])
            np.testing.assert_array_equal(data["contact_area"], old["contact_area"])
        for clip, key in zip(group["clip"], keys):
            if seen.setdefault(clip, key) != key:
                raise ValueError(f"Instance changes between replays of {clip}")
    if set(seen.values()) != set(masses.index):
        raise ValueError("Rendered instances differ from the mass audit")
    for clip, key in sorted(seen.items()):
        with np.load(Path(config["grab_root"]) / clip, allow_pickle=True) as raw:
            obj, action = str(raw["obj_name"]), str(raw["motion_intent"])
            fps = float(raw["framerate"])
            z = raw["object"].item()["params"]["transl"][:, 2]
            left, right = contact_rule(raw["contact"].item()["body"], hand_ids)
        contact = left | right
        lifted = contact & (z > z[0] + LIFT_M)
        record = masses.loc[key]
        clips[clip] = {"object": obj, "intent": action, "fps": fps, "n_frames": len(z),
                       "instance_key": key, "rendered_noun": record.rendered_noun,
                       "straddling_noun": bool(record.straddling_noun),
                       "action": action if action in ACTIONS else None,
                       "property": record.tertile if lifted.any() and record.tertile in PROPERTIES else None,
                       "action_event": lifted if action == "lift" else contact,
                       "property_event": lifted}
    return clips


def replay_events(frames, sessions, clips, split_by_clip):
    """Reconstruct raw source times using the renderer's integer-second sampling."""
    tables = []
    for video, group in frames.groupby("video_id", sort=True):
        duration = int(sessions.at[video, "duration_s"])
        sequence = json.loads(sessions.at[video, "clips"])
        segments = []
        start = 0.
        for clip in sequence:
            info = clips[clip]
            stop = min(duration, start + info["n_frames"] / info["fps"])
            segments.append(dict(clip=clip, start=start, stop=stop, fps=info["fps"],
                                 source_frames=info["n_frames"]))
            start = stop
        sampled = sample_chain(segments, duration)
        np.testing.assert_array_equal(group["clip"], [clip for clip, _ in sampled])
        t = group.frame_idx.to_numpy()
        occurrence = np.searchsorted([seg["stop"] for seg in segments], t, side="right")
        metadata = [{k: info[k] for k in ("object", "intent", "instance_key", "rendered_noun",
                                         "straddling_noun", "action", "property")}
                    for clip, _ in sampled for info in (clips[clip],)]
        table = pd.DataFrame(metadata, index=group.index)
        table = pd.concat([group, table], axis=1)
        table["split"] = table["clip"].map(split_by_clip)
        table["occurrence"] = occurrence
        table["replay_start_s"] = [segments[i]["start"] for i in occurrence]
        table["raw_frame_idx"] = [idx for _, idx in sampled]
        for rung in ("action", "property"):
            table[f"{rung}_event"] = [clips[clip][f"{rung}_event"][idx] for clip, idx in sampled]
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def speech_context(manifest, timeline, clips):
    lookup = timeline.set_index(KEY)
    context = lookup.loc[list(zip(manifest.video_id, manifest.source_frame_idx))].reset_index()
    aligned = lookup.loc[list(zip(manifest.video_id, manifest.frame_idx))]
    np.testing.assert_array_equal(context["clip"], aligned["clip"])
    np.testing.assert_array_equal(context.occurrence, aligned.occurrence)
    other_visible, own_visible = [], []
    for row in context.itertuples():
        visible = {obj["category"] for obj in json.loads(row.objects) if obj["visible_px"] > 0}
        other_visible.append(bool(visible - {row.rendered_noun}))
        own_visible.append(row.rendered_noun in visible)
    # Outside an event, treat an absent object as an unseen referent, as in noun speech.
    local_time = manifest.utterance_time_s - context.replay_start_s
    for rung in ("action", "property"):
        matching = []
        for clip, sec in zip(context["clip"], local_time):
            info = clips[clip]
            idx = int(np.floor(sec * info["fps"]))
            matching.append(0 <= idx < info["n_frames"] and info[f"{rung}_event"][idx])
        context[f"{rung}_alignment"] = np.where(
            matching, "matching", np.where(np.asarray(other_visible) & own_visible, "other_visible", "no_referent"))
    return context


def event_panel(timeline, rung, split):
    # Scouting counted any other GRAB action as a potential foil. Only the
    # five approved actions can be queries.
    label = "intent" if rung == "action" else "property"
    possible = timeline[timeline.split.eq(split) & timeline[label].notna()]
    keys = ["video_id", "occurrence"]
    moments = possible[possible[f"{rung}_event"]]
    groups = moments.groupby(keys, sort=True)
    # The middle available frame keeps the target away from contact boundaries.
    indices = [g.index[(len(g) - 1) // 2] for _, g in groups]
    columns = [*KEY, "clip", "subject", "occurrence", "raw_frame_idx", "instance_key",
               "rendered_noun", "object", "intent", "straddling_noun"]
    rows = moments.loc[indices, columns].reset_index(drop=True)
    rows["referent"] = moments.loc[indices, label].to_numpy()
    rows["text"] = rows.referent
    rows["object_matched"] = False
    rng = np.random.default_rng(0)
    by_clip = rows.groupby("clip").indices
    catalogue = rows.drop_duplicates("clip").set_index("clip")
    pools = {}
    for clip, row in catalogue.iterrows():
        foils = catalogue[catalogue.referent.ne(row.referent)]
        if rung == "action":
            ranks = foils.object.ne(row.object).astype(int)
        else:
            ranks = 2 * foils.rendered_noun.ne(row.rendered_noun).astype(int) + foils.intent.ne(row.intent)
        order = rng.permutation(len(foils))
        pools[clip] = foils.index[order[np.argsort(ranks.iloc[order], kind="stable")[:3]]]
    selected = []
    for i, row in enumerate(rows.itertuples()):
        if rung == "action" and row.referent not in ACTIONS:
            selected.append(np.array([], dtype=int))
            continue
        candidates = pools[row.clip]
        selected.append(np.array([rng.choice(by_clip[c]) for c in candidates], dtype=int))
        if rung == "action":
            rows.at[i, "object_matched"] = len(candidates) == 3 and catalogue.loc[candidates, "object"].eq(row.object).all()
    identity = list(zip(rows.video_id, rows.frame_idx.astype(int), rows.referent))
    trials = evaluate._body_trials(rows, {"pools": selected, "identity": identity}, 0, None)
    is_target = rows.referent.isin(ACTIONS if rung == "action" else PROPERTIES)
    if len(trials) != is_target.sum():
        raise ValueError(f"Insufficient distinct foil clips for {rung} {split}")
    for target, candidates, correct in trials:
        assert candidates[correct] == target
        assert rows.iloc[candidates]["clip"].nunique() == 4
        assert rows.iloc[candidates].referent.eq(rows.at[target, "referent"]).sum() == 1
    return rows, {
        "trial_rows": [int(i) for i, _, _ in trials],
        "candidate_rows": [c.tolist() for _, c, _ in trials],
        "correct_slots": [correct for _, _, correct in trials],
        "donor_rows": _panel_donors(rows.subject.tolist(), rows["clip"].tolist()),
    }, int(possible[possible[label].isin(ACTIONS if rung == "action" else PROPERTIES)]
           .groupby(keys).ngroups - len(trials))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("build", "freeze", "smoke", "baseline-smoke"), default="build")
    parser.add_argument("--clips", type=int, default=40)
    parser.add_argument("--use-existing-physical", action="store_true", help="Verify and reuse a completed Step 7 store")
    args = parser.parse_args()
    if args.action == "freeze":
        return freeze_learner()
    if args.action in ("smoke", "baseline-smoke"):
        return smoke_learner(OUTPUT / args.action, args.clips, baseline=args.action == "baseline-smoke")
    for name in ("manifest.parquet", "splits.json", "panels", "audit.json"):
        if (OUTPUT / name).exists():
            raise FileExistsError(OUTPUT / name)
    if not args.use_existing_physical:
        _output_directory(PHYSICAL.parent)
        build_physical(RENDER, key_by="instance", output=PHYSICAL)

    protocol = json.loads((FROZEN / "protocol.json").read_text())
    inputs = Path(protocol["inputs"])
    config = json.loads((inputs / "generator.json").read_text())
    manifest_path = inputs / "manifest.parquet"
    if _sha(manifest_path) != protocol["hashes"][str(manifest_path)]:
        raise ValueError("Step 6 input speech no longer matches its frozen hash")
    manifest = pd.read_parquet(manifest_path)
    splits = json.loads((FROZEN / "splits.json").read_text())
    source_keys = {**splits["source_keys"], "train": splits["eligible_order_seed_0"]}
    split_by_clip = {clip: split for split, keys in source_keys.items() for clip in keys}
    if len(split_by_clip) != sum(map(len, source_keys.values())):
        raise ValueError("Source clip crosses partitions")

    frames = pd.read_parquet(RENDER / "frames.parquet").sort_values(KEY).reset_index(drop=True)
    frames["clip"] = _source_keys(frames)
    sessions = pd.read_parquet(RENDER / "sessions.parquet").set_index("video_id")
    index = pd.read_parquet(inputs / "train_cache/index.parquet", columns=KEY).sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(frames[KEY], index)
    masses = pd.read_csv(MASS_CSV).set_index("instance_key")
    clips = load_events(frames, masses)
    timeline = replay_events(frames, sessions, clips, split_by_clip)
    assert timeline.loc[timeline.split.eq("s8"), "subject"].eq("s8").all()
    assert timeline.loc[timeline.subject.eq("s8"), "split"].eq("s8").all()
    context = speech_context(manifest, timeline, clips)
    if context.split.isna().any():
        raise ValueError("Step 6 speech contains a source outside the frozen splits")
    own_name = manifest.named & manifest.referent.eq(context.rendered_noun)
    own_rate = int((own_name & context.split.eq("train")).sum()) / len(source_keys["train"])
    speech, counts = add_event_words(manifest, context, config["config"]["vocabulary"], config["seed"], own_rate)
    pd.testing.assert_frame_equal(speech[manifest.columns.drop("text")], manifest.drop(columns="text"))
    speech = speech.assign(split=context.split, clip=context["clip"])
    noun_counts = speech[speech.named].groupby("split").size().to_dict()
    # Verify the current-B loader against real files using a small train-only fit.
    ids = [frames.video_id.iloc[0]]
    group = frames[frames.video_id.isin(ids)]
    fit = ~group["clip"].map(split_by_clip).eq("train").to_numpy()[None, :]
    store = body.load_store(RENDER, group, ids, set(), exclude=fit, physical=PHYSICAL)
    if sum(g.shape[-1] for g in store["groups"].values()) != 153:
        raise ValueError("The instance store did not load into 153-dimensional current B")

    panels, summary = {}, {}
    for rung in ("action", "property"):
        summary[rung] = {}
        for split in SPLITS[1:]:
            rows, trials, missing = event_panel(timeline, rung, split)
            panels[f"{rung}_{split}"] = (rows, trials)
            targets = rows.iloc[trials["trial_rows"]]
            summary[rung][split] = {"trials": len(targets), "clips": targets["clip"].nunique(),
                                    "instances": targets.instance_key.nunique(), "replays_without_event_frame": missing,
                                    "trials_by_word": targets.referent.value_counts().to_dict()}
        held = pd.concat([rows.iloc[trials["trial_rows"]] for name, (rows, trials) in panels.items()
                          if name in (f"{rung}_test", f"{rung}_s8")])
        flag = "object_matched" if rung == "action" else "straddling_noun"
        summary[rung]["held_out"] = {"trials": len(held), "clips": held["clip"].nunique(),
                                    "instances": held.instance_key.nunique(),
                                    f"{flag}_clips": held.loc[held[flag], "clip"].nunique(),
                                    f"{flag}_trials": int(held[flag].sum())}
    tests = json.loads((FROZEN / "panels/clips_test/trials.json").read_text())
    summary["noun_test_trials"] = len(tests["trial_rows"])

    _output_directory(OUTPUT)
    speech.to_parquet(OUTPUT / "manifest.parquet", index=False)
    shutil.copyfile(FROZEN / "splits.json", OUTPUT / "splits.json")
    for name in ("clips_dev", "clips_test", "s8"):
        shutil.copytree(FROZEN / "panels" / name, OUTPUT / "panels" / name)
    for name, (rows, trials) in panels.items():
        directory = OUTPUT / "panels" / name
        directory.mkdir()
        rows.to_parquet(directory / "manifest.parquet", index=False)
        _write(directory / "trials.json", trials)
    audit = {"physical": str(PHYSICAL.resolve()), "physical_sessions": frames.video_id.nunique(),
             "physical_frames": len(frames), "physical_instances": len(masses),
             "noun_utterances": noun_counts, "own_name_rate": own_rate,
             "speech": counts, "panels": summary, "source_splits_disjoint": True,
             "speech_source_sha256": _sha(manifest_path), "instance_mass_sha256": _sha(MASS_CSV)}
    _write(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
