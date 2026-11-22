"""Freeze and run Step 9 using the Step 6 ladder and frozen Step 7 speech."""

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from nursery.common import _output_directory, load_config
from nursery.learner import _upstream_module, acquisition as run
from nursery.learner.train import split_pairs
from scripts.step7_word_types import ACTIONS, ARMS, FROZEN, KEY, OUTPUT

ROOT = OUTPUT.with_name("step9_acquisition_curves")
STEP7 = OUTPUT / "frozen"
LADDER = dict(zip(run.FRACTIONS, (63, 158, 315, 473, 630)))
PICTURES = ("picture_test", "konkle_test")
MIN_COUNT = 5


def prepare():
    """Verify the source freezes, bind the new ladder and write Check A."""
    step6, step7 = FROZEN.resolve(), STEP7.resolve()
    step6_hashes, step7_hashes = (run._read(p / "frozen_inputs.json") for p in (step6, step7))

    def frozen(source, relative):
        path = source / relative
        hashes = step6_hashes if source == step6 else step7_hashes
        if run._sha(path) != hashes["prepared_hashes"][str(path)]:
            raise ValueError(f"Changed frozen input: {path}")
        return path

    parent = run._read(frozen(step6, "protocol.json"))
    protocol = run._read(frozen(step7, "protocol.json"))
    split_path = frozen(step7, "splits.json")
    if split_path.read_bytes() != frozen(step6, "splits.json").read_bytes():
        raise ValueError("Step 6 and Step 7 splits differ")
    splits = run._read(split_path)
    ordered = splits["eligible_order_seed_0"]
    held_out = set().union(*(splits["source_keys"][s] for s in ("dev", "test", "s8")))
    if len(ordered) != 630 or len(set(ordered)) != 630 or set(ordered) & held_out:
        raise ValueError("Invalid frozen training ladder")

    frames_path = Path(protocol["episodes"]) / "frames.parquet"
    speech_path = Path(protocol["transcripts"])
    for path in (frames_path, speech_path):
        if run._sha(path) != step7_hashes["external_hashes"][str(path)]:
            raise ValueError(f"Changed Step 7 source input: {path}")
    frames = pd.read_parquet(frames_path, columns=KEY + ["subject", "clip"])
    metadata = frames.set_index(KEY)
    speech = pd.read_parquet(speech_path)
    upstream = _upstream_module("train_frame_mil")
    tokenize = _upstream_module("train").tokenize
    noun_words = set(load_config()["vocabulary"].values())
    full = {path.name: frozen(step7, path.relative_to(step7))
            for path in sorted((step7 / "subsets/1.0").iterdir()) if path.is_file()}

    panel_files = [frozen(step7, path.relative_to(step7))
                   for path in sorted((step7 / "panels").rglob("*")) if path.is_file()]
    picture_specs = {panel: {"cache": parent["picture_panels"][panel]["cache"]} for panel in PICTURES}
    for panel, spec in picture_specs.items():
        panel_files.append(frozen(step6, f"panels/{panel}/trials.json"))
        for name in ("emb.f16.npy", "index.parquet"):
            path = str(Path(spec["cache"]) / name)
            protocol["hashes"][path] = step6_hashes["external_hashes"][path]

    root = _output_directory(ROOT / "frozen", exist_ok=False)
    for path in [split_path, *panel_files]:
        source = step7 if path.is_relative_to(step7) else step6
        target = root / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    counts, subsets = {}, {}
    for fraction, amount in LADDER.items():
        relative = Path("subsets") / str(fraction)
        selected = sorted(ordered[:amount])
        parent_stats = run._read(frozen(step6, relative / "stats.json"))
        if selected != parent_stats["selected_clips"]:
            raise ValueError(f"Step 6 subset differs at {amount} clips")
        mask = frozen(step6, relative / "fit_mask.npz")
        with np.load(mask, allow_pickle=False) as fit:
            ids, include = fit["video_ids"], fit["include"]
        positions = frames.video_id.map(dict(zip(ids, range(len(ids))))).to_numpy()
        frame_indices = frames.frame_idx.to_numpy()
        if not np.array_equal(include[positions, frame_indices], frames["clip"].isin(selected)):
            raise ValueError(f"Step 6 fit mask differs from the clip prefix at {amount} clips")
        partition = np.full(include.shape, -1, dtype=int)
        partition[positions, frame_indices] = 1
        partition[include] = 0
        labels = split_pairs(speech, dict(zip(ids, partition)), protocol["window"])
        raw = speech[(labels == 0) & speech.text.fillna("").str.strip().ne("")].reset_index(drop=True)
        rows = metadata.reindex(pd.MultiIndex.from_frame(raw[KEY]))
        if not raw["split"].eq("train").all() or not np.array_equal(raw["clip"], rows["clip"]):
            raise ValueError("Step 7 utterances disagree with the frozen training split")
        vocab = upstream.build_vocab(raw.text, MIN_COUNT)
        manifest = raw[raw.text.map(lambda words: bool(upstream.encode(words, vocab)))].reset_index(drop=True)
        manifest["subject"] = metadata.reindex(pd.MultiIndex.from_frame(manifest[KEY])).subject.to_numpy()
        stats = dict(selected_clips=selected, selected_clip_count=amount, utterances=len(manifest),
                     utterances_before_vocab=len(raw), replay_exposure_seconds=int(include.sum()))

        subset = root / relative
        subset.mkdir(parents=True)
        if fraction == 1.0:
            pd.testing.assert_frame_equal(manifest, pd.read_parquet(full["manifest.parquet"]))
            if stats != run._read(full["stats.json"]) or vocab != run._read(full["vocab.json"]):
                raise ValueError("The 630-clip subset differs from Step 7")
            with np.load(full["fit_mask.npz"], allow_pickle=False) as fit:
                if not (np.array_equal(fit["video_ids"], ids) and np.array_equal(fit["include"], include)):
                    raise ValueError("The full fit mask differs from Step 7")
            for path in full.values():
                shutil.copyfile(path, subset / path.name)
        else:
            manifest.to_parquet(subset / "manifest.parquet", index=False)
            shutil.copyfile(mask, subset / "fit_mask.npz")
            run._write(subset / "vocab.json", vocab)
            run._write(subset / "stats.json", stats)
        subsets[str(fraction)] = stats
        utterances = Counter(word for words in raw.text for word in set(tokenize(words)))
        counts[str(amount)] = {
            "noun_vocab_count": len(noun_words & vocab.keys()),
            "action_words": {word: {"utterances": utterances[word], "in_vocab": word in vocab} for word in ACTIONS},
        }

    protocol.update(run._source_snapshot())
    protocol.update(fractions=list(LADDER), conditions=list(ARMS), seeds=list(range(5)), arms=ARMS,
                    subsets=subsets, epochs=60, temporal_video=True, temporal_weight_init="per_word",
                    approval="Step 9 acquisition curves", picture_panels=picture_specs,
                    smoke_evaluate_all_panels=True)
    protocol["code_hashes"][str(Path(__file__).resolve())] = run._sha(__file__)
    protocol["selection"].update(panel="equal mean of noun and action dev accuracies",
                                 word_type_rungs=["noun", "action"])
    # Step 7 scored nouns on clips_test only. Its frozen close-up cache
    # does not cover the separate noun s8 panel.
    protocol["panel_hashes"] = {str(path.relative_to(root)): run._sha(path)
                                for path in sorted((root / "panels").rglob("*")) if path.is_file()}
    queue = [[fraction, seed, arm] for fraction, amount in LADDER.items()
             for seed in protocol["seeds"] for arm in ARMS
             if arm != run.SHUFFLED_BODY or amount in (158, 630)]
    run._write(root / "protocol.json", protocol)
    run._write(root / "signal_hashes.json", run._read(frozen(step7, "signal_hashes.json")))
    run._write(root / "queue.json", queue)
    run._freeze_inputs(root, audit_preparation=False)
    run._write(root.parent / "action_word_counts.json", counts)
    print(f"Froze {len(queue)} jobs and wrote action_word_counts.json", flush=True)


def smoke():
    import torch

    torch.set_num_threads(4)
    root = (ROOT / "frozen").resolve()
    run._verify_frozen(root, external=False)
    for arm in ARMS:
        print(f"Running {arm} on CPU", flush=True)
        metrics = run.train_one(root, 0.1, 0, arm, smoke=True, device="cpu")
        dev = metrics["history"][0]["dev_by_rung"]
        if set(dev) != {"noun", "action", "property"} or not all(map(math.isfinite, dev.values())):
            raise ValueError(f"Missing or non-finite development panels: {arm}")
        if not math.isclose(metrics["selection"]["body_present"]["dev_score"], (dev["noun"] + dev["action"]) / 2):
            raise ValueError(f"Incorrect selection score: {arm}")
        for key in ("primary", "all_extra_inputs_removed"):
            for rung, panels in metrics[key].items():
                for name, panel in panels.items():
                    if panel["n_trials"] == 0 or not math.isfinite(panel["accuracy"]):
                        raise ValueError(f"Empty or non-finite readout: {arm}/{key}/{rung}/{name}")
            for rung, names in {
                "noun": ("test",),
                "action": ("test", "s8", "pooled", "object_matched_test", "object_matched_s8", "object_matched"),
                "property": ("test", "s8", "pooled", "straddling_noun_test", "straddling_noun_s8", "straddling_noun"),
            }.items():
                if not set(names) <= metrics[key].get(rung, {}).keys():
                    raise ValueError(f"Missing readout: {arm}/{key}/{rung}")
        for name in PICTURES:
            panel = metrics["picture"][name]
            if panel["n_trials"] == 0 or not math.isfinite(panel["accuracy"]):
                raise ValueError(f"Empty or non-finite picture readout: {arm}/{name}")
        print(f"{arm}: {metrics['epoch_seconds']:.2f} s/epoch; all panels and readouts verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "smoke", "run"))
    parser.add_argument("--task", type=int, choices=range(85), metavar="0..84")
    args = parser.parse_args()
    if args.action == "prepare":
        return prepare()
    if args.action == "smoke":
        return smoke()
    if args.task is None:
        parser.error("run requires --task")
    root = ROOT / "frozen"
    fraction, seed, arm = run._read(root / "queue.json")[args.task]
    return run.train_one(root, fraction, seed, arm, device="cuda")


if __name__ == "__main__":
    main()
