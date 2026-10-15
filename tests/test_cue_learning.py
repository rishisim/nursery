"""CPU-only synthetic validation of transcript cues and upstream adapters."""

import json
from collections import Counter
import os
from pathlib import Path
import re
import socket
import subprocess

import numpy as np
import pandas as pd
import pytest

from nursery.cue_learning import UPSTREAM, _output_directory, _upstream_module, load_config
from nursery.cue_learning.embed import embed_episodes, probe_frames
from nursery.cue_learning.transcripts import cue_timelines, generate_transcripts, write_transcripts


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def upstream():
    if not (UPSTREAM / "src" / "train_frame_mil.py").exists():
        pytest.skip(f"Pinned upstream checkout absent: {UPSTREAM}")
    import torch

    torch.set_num_threads(1)
    return _upstream_module("train_frame_mil")


def synthetic_frames(n=120, videos=2):
    objects = json.dumps([{"category": c, "visible_px": 40, "cells": [0, 1]}
                          for c in ("mug", "bowl", "apple", "banana")])
    rows = []
    for video in range(videos):
        for frame in range(n):
            contact = ((frame * 17 + video * 23) % 101) < (20 + video * 10)
            rows.append({"video_id": f"v{video}", "frame_idx": frame,
                         "image": f"frames/v{video}/{frame:05d}.jpg", "subject": f"s{video + 1}",
                         "room": "train_room", "clip": "synthetic", "contact_left": contact,
                         "contact_right": False, "held_category": "mug" if contact else None,
                         "objects": objects})
    return pd.DataFrame(rows)


@pytest.fixture
def episodes(tmp_path):
    from PIL import Image

    # Exercise the production git check-ignore guard rather than bypassing it.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("outputs/\n")
    root = _output_directory(tmp_path / "outputs" / "episodes")
    frames = synthetic_frames()
    sessions = pd.DataFrame([
        {"video_id": v, "subject": f"s{i + 1}", "room": "train_room", "split": "train",
         "clips": '[]', "distractors": '[]', "duration_s": 120}
        for i, v in enumerate(("v0", "v1"))
    ] + [{"video_id": "v_dev", "subject": "s9", "room": "dev_room", "split": "dev",
          "clips": '[]', "distractors": '[]', "duration_s": 1},
         {"video_id": "v_test", "subject": "s10", "room": "test_room", "split": "test",
          "clips": '[]', "distractors": '[]', "duration_s": 1}])
    extra = frames.iloc[:2].copy()
    extra["video_id"] = ["v_dev", "v_test"]
    extra["frame_idx"] = 0
    extra["image"] = ["frames/v_dev/00000.jpg", "frames/v_test/00000.jpg"]
    extra["subject"] = ["s9", "s10"]
    extra["room"] = ["dev_room", "test_room"]
    frames = pd.concat([frames, extra], ignore_index=True)
    probes = pd.DataFrame([
        {"probe_id": f"{split}_{cat}_{i}", "image": f"probes/{split}_{cat}_{i}.jpg",
         "category": cat, "room": f"probe_{split}", "split": split}
        for split in ("dev", "test") for cat in ("mug", "bowl", "apple", "banana") for i in range(4)
    ])
    for image in pd.concat([frames.image, probes.image]):
        path = root / image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (80, 100, 120)).save(path)
    for name, table in (("frames", frames), ("sessions", sessions), ("probes", probes)):
        table.to_parquet(root / f"{name}.parquet", index=False)
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "synthetic": True}))
    return root, tmp_path / "outputs" / "run"


def test_transcript_statistics_and_determinism():
    config = load_config()
    frames = synthetic_frames(n=15000, videos=4)
    frames["held_category"] = "mug"
    manifest, _ = generate_transcripts(frames, config, 1701)
    again, _ = generate_transcripts(frames, config, 1701)
    pd.testing.assert_frame_equal(manifest, again)
    different, _ = generate_transcripts(frames, config, 1702)
    assert not manifest.equals(different)
    refs = manifest[manifest.referential]
    measured = {"utterances": len(manifest), "rate_hz": len(manifest) / len(frames),
                "p_ref": float(manifest.referential.mean()), "p_held": float((refs.channel == "held").mean()),
                "p_name": float(refs.named.mean()), "jitter_mean": float(manifest.jitter_frames.mean()),
                "jitter_std": float(manifest.jitter_frames.std())}
    print("statistics:", json.dumps(measured, sort_keys=True))
    assert measured["rate_hz"] == pytest.approx(1 / 3, abs=0.01)
    # Always-held scene: no unseen channel, so held share is p_held after redraws.
    for key in ("p_ref", "p_held", "p_name"):
        assert measured[key] == pytest.approx(config["generator"][key], abs=0.035)
    assert abs(measured["jitter_mean"]) < 0.04
    assert measured["jitter_std"] == pytest.approx(np.sqrt(1 + 1 / 12), abs=0.04)
    assert ((manifest.frame_idx >= 0) & (manifest.frame_idx < 15000)).all()
    altered = load_config()
    altered["cues"]["noise_eps"] = [0, 1]
    changed_cues, _ = generate_transcripts(frames, altered, 1701)
    pd.testing.assert_frame_equal(manifest.iloc[:, :12], changed_cues.iloc[:, :12])


def test_mean_weights_and_oracle():
    frames = synthetic_frames(1000, 4)
    # Visibility changes so the oracle must use the shifted frame, not the source.
    frames.loc[frames.frame_idx % 2 == 0, "objects"] = '[]'
    config = load_config()
    config["generator"].update(p_ref=1, p_name=0.5)
    frames["held_category"] = "mug"
    manifest, _ = generate_transcripts(frames, config, 34)
    means = manifest.filter(like="weight_").mean()
    np.testing.assert_allclose(means, 1, atol=1e-12)
    expected_visible = (manifest.frame_idx % 2 != 0).to_numpy()
    np.testing.assert_array_equal(manifest.referent_visible, expected_visible)
    high = manifest.referential & manifest.named & manifest.referent_visible
    oracle = np.where(high, 1.0, 0.2)
    np.testing.assert_allclose(manifest.weight_oracle, oracle / oracle.mean())
    cue_seed = np.random.SeedSequence(34).spawn(2)[1]
    cues, _ = cue_timelines(frames, config["cues"], np.random.default_rng(cue_seed))
    for name in cues["v0"]:
        values = [cues[r.video_id][name][r.frame_idx] for r in manifest.itertuples()]
        weights = np.where(values, 1.0, 0.2)
        np.testing.assert_allclose(manifest[f"weight_{name}"], weights / weights.mean())
    print("weight means:", means.to_dict(), "oracle positives:", int(high.sum()))


def test_shuffled_marginals_shift_and_noise():
    frames = synthetic_frames(1000, 4)
    config = load_config()["cues"]
    config["noise_eps"] = [0, 0.25, 1]
    cues, donors = cue_timelines(frames, config, np.random.default_rng(3))
    contact = np.concatenate([c["contact"] for c in cues.values()])
    shuffled = np.concatenate([c["shuffled"] for c in cues.values()])
    assert contact.sum() == shuffled.sum()
    for video, timeline in cues.items():
        assert donors[video] != video
        np.testing.assert_array_equal(timeline["shuffled"], cues[donors[video]]["contact"])
        np.testing.assert_array_equal(timeline["shifted"], np.roll(timeline["contact"], -10))
        np.testing.assert_array_equal(timeline["noisy_0"], timeline["contact"])
        np.testing.assert_array_equal(timeline["noisy_1"], ~timeline["contact"])
    flip_rate = np.concatenate([c["noisy_0.25"] ^ c["contact"] for c in cues.values()]).mean()
    assert flip_rate == pytest.approx(0.25, abs=0.03)
    disagreement = float((contact != shuffled).mean())
    assert disagreement > 0.3
    print(f"cue marginals: {contact.sum()}/{len(contact)} = {shuffled.sum()}/{len(shuffled)}; "
          f"shuffled disagreement={disagreement:.4f}; noisy_0.25 flip rate={flip_rate:.4f}")


def test_no_held_referents_and_category_independent_frames():
    frames = synthetic_frames(1000, 2)
    frames["held_category"] = pd.Series(pd.NA, index=frames.index, dtype="string")
    config = load_config()
    config["generator"].update(p_ref=1, p_name=1)
    manifest, _ = generate_transcripts(frames, config, 10)
    frequencies = manifest.referent.value_counts(normalize=True)
    assert set(frequencies.index) == {"mug", "bowl", "apple", "banana"}
    assert not manifest.held_referent.any() and not manifest.held_referent.isna().any()
    np.testing.assert_allclose(frequencies, 0.25, atol=0.06)
    swapped = frames.copy()
    swapped["objects"] = swapped.objects.str.replace('"mug"', '"plane"').str.replace('"apple"', '"glass"')
    changed_config = load_config()
    changed_config["generator"] = config["generator"]
    changed_config["vocabulary"].update(plane="plane", glass="glass")
    # Template draws cannot depend on object category; compare a one-object scene.
    for table, category in ((frames, "mug"), (swapped, "plane")):
        table["held_category"] = category
        table["objects"] = json.dumps([{"category": category, "visible_px": 5, "cells": [0]}])
    original, _ = generate_transcripts(frames, config, 10)
    changed, _ = generate_transcripts(swapped, changed_config, 10)
    assert original.text.str.replace("mug", "{noun}").tolist() == changed.text.str.replace("plane", "{noun}").tolist()
    pool = json.loads((Path(__file__).parents[1] / "nursery/cue_learning/frames.json").read_text())
    assert sum(map(len, pool.values())) == 150
    nouns = set(config["vocabulary"].values())
    for kind, templates in pool.items():
        assert len(set(templates)) == len(templates)
        for template in templates:
            assert ("{noun}" in template) == (kind == "named")
            assert not nouns.intersection(template.replace("{noun}", "").strip(".!?").split())


def test_channels_follow_window_and_visibility_and_calibrate():
    cats = ["mug", "bowl", "apple", "banana", "cup", "pan"]
    frames = synthetic_frames(n=6000, videos=2)
    rng = np.random.default_rng(0)
    for video in ("v0", "v1"):
        mask = frames.video_id == video
        shown = [sorted(rng.choice(cats, 3, replace=False)) for _ in range(6000)]
        frames.loc[mask, "objects"] = [json.dumps([{"category": c, "visible_px": 9, "cells": [0]} for c in cs])
                                       for cs in shown]
        held = np.full(6000, None, dtype=object)
        for start in rng.choice(6000, 300, replace=False):
            held[start:start + 3] = cats[int(rng.integers(len(cats)))]
        frames.loc[mask, "held_category"] = held
    config = load_config()
    config["vocabulary"].update(cup="cup", pan="pan")
    manifest, _ = generate_transcripts(frames, config, 5)
    window = config["generator"]["hold_window_s"]
    refs = manifest[manifest.referential]
    assert set(refs.channel) == {"held", "seen", "unseen"} and manifest.channel[~manifest.referential].isna().all()
    for r in refs.itertuples():
        group = frames[frames.video_id == r.video_id].reset_index(drop=True)
        near = set(group.held_category.iloc[max(0, r.source_frame_idx - window):r.source_frame_idx + window + 1].dropna())
        visible = {o["category"] for o in json.loads(group.objects[r.source_frame_idx])}
        assert (r.referent in near) == (r.channel == "held")
        assert (r.referent in visible - near) == (r.channel == "seen")
        if r.channel == "unseen":
            assert r.referent not in visible | near
    shares = refs.channel.value_counts(normalize=True)
    assert shares["held"] == pytest.approx(0.35, abs=0.02)
    assert manifest.attrs["p_held_internal"] > 0.35


def test_referential_draw_without_candidate_becomes_nonreferential():
    frames = synthetic_frames(1000, 2)
    frames["objects"] = '[]'
    frames["held_category"] = None
    config = load_config()
    config["generator"].update(p_ref=1, p_name=1)
    manifest, _ = generate_transcripts(frames, config, 10)
    pool = json.loads((Path(__file__).parents[1] / "nursery/cue_learning/frames.json").read_text())
    assert len(manifest) > 0
    assert not manifest.referential.any() and manifest.referent.isna().all()
    assert not manifest.named.any() and not manifest.held_referent.any()
    assert set(manifest.text) <= set(pool["nonreferential"])


def test_frames_have_no_frequent_pool_exclusive_words():
    pool = json.loads((Path(__file__).parents[1] / "nursery/cue_learning/frames.json").read_text())
    counts = [Counter(re.findall(r"[a-z]+(?:'[a-z]+)?", " ".join(templates).lower().replace("{noun}", "")))
              for templates in pool.values()]
    total = sum(counts, Counter())
    exclusive = {word for word, count in total.items() if count >= 3 and sum(word in c for c in counts) == 1}
    assert not exclusive
    for word in ("the", "where's", "what's", "it", "that", "this", "your", "yours"):
        rates = [c[word] for c in counts]
        assert min(rates) > 0 and max(rates) <= 3 * min(rates)


def test_reject_nonignored_outputs_and_unmatched_sessions(tmp_path):
    with pytest.raises(ValueError, match="ignored"):
        _output_directory(tmp_path / "unignored")
    with pytest.raises(ValueError, match="two train sessions"):
        cue_timelines(synthetic_frames(5, 1), load_config()["cues"], np.random.default_rng(0))


def test_transcript_writer_filters_train(episodes):
    root, run = episodes
    manifest = write_transcripts(root, run, 4)
    assert set(manifest.video_id) == {"v0", "v1"}
    saved = pd.read_parquet(run / "manifest.parquet")
    pd.testing.assert_frame_equal(manifest, saved)
    record = json.loads((run / "generator.json").read_text())
    assert record["n_frames"] == 240
    with pytest.raises(FileExistsError):
        write_transcripts(root, run, 4)


@pytest.mark.parametrize("available,mps,device", [(False, False, None), (False, True, None), (True, True, None), (True, True, "cpu"), (False, False, "cuda:0")])
def test_embedding_adapter_with_fake_encoder(episodes, upstream, monkeypatch, available, mps, device):
    import torch

    root, run = episodes
    module = _upstream_module("embed_regions")
    requested = []
    batches = []
    expected_device = device or ("cuda" if available else "mps" if mps else "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    class Inputs(dict):
        def to(self, device):
            assert device == expected_device
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model):
            requested.append(model)
            return cls()

        def __call__(self, images, return_tensors):
            batches.append(len(images))
            return Inputs(batch=len(images))

    class Encoder:
        config = type("Config", (), {"hidden_size": 8, "num_register_tokens": 0})()

        @classmethod
        def from_pretrained(cls, model):
            requested.append(model)
            return cls()

        def eval(self):
            return self

        def to(self, device):
            assert device == expected_device
            return self

        def __call__(self, batch):
            tokens = torch.arange(65).reshape(1, 65, 1).expand(batch, 65, 8).float()
            return type("Result", (), {"last_hidden_state": tokens})()

    monkeypatch.setattr(module, "AutoImageProcessor", Processor)
    monkeypatch.setattr(module, "AutoModel", Encoder)
    from PIL import Image
    from nursery.cue_learning import embed as embed_module

    # Konkle photos live outside the episode root; embed them from their own paths.
    photos = []
    for category in ("apple", "bowl"):
        for index in range(4):
            path = root.parent / "konkle" / category / f"{index}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
            photos.append({"video_id": f"konkle/{category}", "frame_idx": index, "category": category, "path": str(path)})
    monkeypatch.setattr(embed_module, "konkle_frames", lambda config: pd.DataFrame(photos))
    embed_episodes(root, run, device=device, batch=31)
    assert max(batches) == 31
    assert "path" not in pd.read_parquet(run / "konkle_frames.parquet").columns
    for name, count in (("train", 240), ("probes", 32), ("konkle", 8)):
        emb = np.load(run / f"{name}_cache" / "emb.f16.npy")
        index = pd.read_parquet(run / f"{name}_cache" / "index.parquet")
        assert emb.shape == (count, 17, 8) and emb.dtype == np.float16
        assert set(index.columns) == {"video_id", "frame_idx", "row"}
        expected = np.r_[0, np.array([[1 + 16 * y + 2 * x + 4.5 for x in range(4)] for y in range(4)]).ravel()]
        np.testing.assert_array_equal(emb[0, :, 0], expected)
    assert requested == ["facebook/dinov2-base"] * 6
    assert set(pd.read_parquet(run / "train_frames.parquet").video_id) == {"v0", "v1"}
    # Resume uses upstream's own cache logic and performs no additional model loads.
    embed_episodes(root, run, device="cpu")
    assert len(requested) == 6
    assert json.loads((run / "embedding.json").read_text())["device"] == "cpu"
    assert json.loads((run / "embedding.json").read_text())["batch"] == 64
    provenance = json.loads((run / "embedding.json").read_text())
    provenance["episodes"] = "other"
    (run / "embedding.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance"):
        embed_episodes(root, run)


def _fake_caches(root, run):
    rng = np.random.default_rng(12)
    frames = pd.read_parquet(root / "frames.parquet")
    frames = frames[frames.video_id.isin(["v0", "v1"])].copy()
    probes = probe_frames(pd.read_parquet(root / "probes.parquet"), load_config())
    probes.to_parquet(run / "probes_frames.parquet", index=False)
    for name, table in (("train", frames), ("probes", probes)):
        cache = run / f"{name}_cache"
        cache.mkdir()
        index = table[["video_id", "frame_idx"]].copy()
        index["row"] = np.arange(len(index))
        index.to_parquet(cache / "index.parquet", index=False)
        np.save(cache / "emb.f16.npy", rng.normal(size=(len(index), 17, 8)).astype(np.float16))


def test_cpu_train_cli_one_epoch(episodes, upstream):
    root, run = episodes
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    command = [os.sys.executable, "-m", "nursery.cue_learning", "train", "--run", str(run),
               "--condition", "contact", "--seed", "8", "--epochs", "1"]
    result = subprocess.run(command, check=True, capture_output=True, text=True,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    print(result.stdout)
    metrics = json.loads((run / "contact_s8" / "metrics.json").read_text())
    assert metrics["condition"] == "contact" and metrics["device"] in ("cpu", "mps")
    assert metrics["selection"] == "dev" and metrics["selected_epoch"] == 0
    assert metrics["epochs"] == 1 and metrics["n_pairs"] > 300 and metrics["emb_dim"] == 8
    assert metrics["generator_parameters"]["p_ref"] == 1
    assert metrics["frame_coverage"] == 1 and metrics["mean_weight"] == pytest.approx(1)
    assert 0 <= metrics["reported_test_acc"] <= 100
    import torch

    state = torch.load(run / "contact_s8" / "model.pt", map_location="cpu", weights_only=True)
    model = upstream.RegionMIL(metrics["vocab"], emb_dim=8)
    model.load_state_dict(state, strict=True)
    assert model.region_prior is None and all(value.device.type == "cpu" for value in state.values())


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_weights_follow_upstream_filtered_pair_indices(episodes, upstream, monkeypatch, device):
    import torch
    from nursery.cue_learning.train import train_run

    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: device == "mps")
    move = torch.Tensor.to
    transfers = []

    def to(tensor, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]) in ("cuda", "mps"):
            transfers.append(tensor.shape)
            args = ("cpu", *args[1:])
        return move(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    root, run = episodes
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    manifest = pd.read_parquet(run / "manifest.parquet")
    manifest.loc[1, "text"] = "quux"
    manifest.to_parquet(run / "manifest.parquet", index=False)
    vocab = upstream.build_vocab(manifest.text, 5)
    retained = manifest[manifest.text.map(lambda t: bool(upstream.encode(t, vocab)))].reset_index(drop=True)
    expected = retained.weight_contact.to_numpy(dtype=np.float32)
    expected /= expected.mean()
    original = upstream.RegionMIL.forward_loss
    observed = []

    def loss(model, v, t, n, w):
        observed.extend(w.tolist())
        return original(model, v, t, n, w)

    original_collate = upstream.collate
    batches = []

    def collate(batch):
        result = original_collate(batch)
        batches.extend(expected[result[0].numpy()].tolist())
        return result

    monkeypatch.setattr(upstream.RegionMIL, "forward_loss", loss)
    monkeypatch.setattr(upstream, "collate", collate)
    region = _upstream_module("train_region_mil")
    selection_scores = iter([0.4, 0.9, 0.7, 0.1])

    def evaluate(*args, **kwargs):
        assert args[5] == device
        return next(selection_scores), {"n_scored": 4, "n_oov": 0, "n_total": 4}

    monkeypatch.setattr(region, "eval_4afc_region", evaluate)
    metrics = train_run(run, "contact", 9, epochs=2)
    assert metrics["device"] == device
    assert bool(transfers) == (device != "cpu")
    np.testing.assert_allclose(observed, batches, atol=1e-7)
    assert metrics["n_novocab"] == 1 and metrics["n_pairs"] == len(manifest) - 1
    assert metrics["selected_epoch"] == 1 and metrics["reported_test_acc"] == 10
    assert metrics["best_test_acc"] == 90


@pytest.mark.parametrize("failure", ["missing_train", "missing_probe", "small_probe", "oov_probe"])
def test_training_rejects_incomplete_inputs(episodes, upstream, failure):
    from nursery.cue_learning.train import train_run

    root, run = episodes
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    if failure in ("missing_train", "missing_probe"):
        name = "train" if failure == "missing_train" else "probes"
        path = run / f"{name}_cache" / "index.parquet"
        table = pd.read_parquet(path).iloc[1:]
        table.to_parquet(path, index=False)
        reason = "does not cover"
    else:
        path = run / "probes_frames.parquet"
        table = pd.read_parquet(path)
        if failure == "small_probe":
            table = table.iloc[1:]
            reason = "four probes/category"
        else:
            table.loc[table.category == "mug", "category"] = "quux"
            reason = "four in-vocabulary"
        table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=reason):
        train_run(run, "contact", 8, epochs=1)
