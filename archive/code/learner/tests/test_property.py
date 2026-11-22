"""Synthetic checks for the bounded property-word diagnostic; no real training runs."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.common import _output_directory
from nursery.learner import property as diagnostic
from nursery.learner.model import LearnerMIL, weighted_nce


pytestmark = pytest.mark.usefixtures("offline")


@pytest.fixture
def property_data(episodes):
    root, run = episodes
    physical = root / "physical_by_clip"
    physical.mkdir()
    cache = _output_directory(run / "train_cache")
    rows = []
    number = 0
    for cats in diagnostic.CATEGORIES.values():
        for category in cats:
            for word, kg in (("light", 0.1), ("heavy", 0.6)):
                for repeat in range(3):
                    video = f"p{number:03d}"
                    for t in range(9):
                        rows.append({"video_id": video, "frame_idx": t, "clip": f"{category}/{word}_{repeat}.npz",
                                     "category": category, "held_category": category, "subject": "s1",
                                     "contact_left": True, "contact_right": False,
                                     "objects": json.dumps([{"category": category, "visible_px": 200}])})
                    load = np.zeros((9, 2, 5), np.float32)
                    load[:, 0, :4] = [0, 0, -9.81 * kg, 9.81 * kg]
                    np.savez_compressed(physical / f"{video}.npz", frame_idx=np.arange(9), load=load,
                                        mass_kg=np.full(9, kg, np.float32))
                    number += 1
    frames = pd.DataFrame(rows)
    frames.to_parquet(root / "frames.parquet", index=False)
    index = frames[["video_id", "frame_idx"]].copy()
    index["row"] = np.arange(len(index))
    index.to_parquet(cache / "index.parquet", index=False)
    features = np.random.default_rng(0).normal(size=(len(index), 17, 8)).astype(np.float16)
    np.save(cache / "emb.f16.npy", features)
    output = root.parent / "property"
    return frames, physical, cache, output, root


def _prepared_inputs(property_data, upstream):
    frames, physical, cache, output, root = property_data
    manifest, _ = diagnostic.select_manifest(frames, physical)
    store, _, _ = diagnostic.load_store(frames, manifest, physical)
    caches, lut = upstream.load_multi([cache])
    rows = {split: manifest[manifest.split == split].reset_index(drop=True) for split in diagnostic.CATEGORIES}
    datasets = {split: upstream.FramePairs(caches, lut, table, diagnostic.VOCAB, 2, False)
                for split, table in rows.items()}
    data = {split: diagnostic._body_data(table, store, lut, "cpu", diagnostic.SETTINGS)
            for split, table in rows.items()}
    return rows, datasets, data, store


def test_selection_is_source_clip_independent_and_balanced(property_data):
    frames, physical, *_ = property_data
    manifest, audit = diagnostic.select_manifest(frames, physical)
    assert manifest["clip"].is_unique
    assert (manifest.frame_idx == 2).all()  # first admissible full five-frame window
    assert audit["balanced_counts"] == {"dev": 12, "test": 12, "train": 36}
    assert set(manifest.text) == {"heavy", "light"}
    assert all(group.text.value_counts().nunique() == 1 for _, group in manifest.groupby("category"))
    assert all(set(manifest[manifest.split == split].category) == set(cats)
               for split, cats in diagnostic.CATEGORIES.items())
    # Another rendered exposure of the same source clip never creates independent data.
    replay = frames[frames.video_id == "p000"].copy()
    replay["video_id"] = "replay"
    with np.load(physical / "p000.npz") as signals:
        np.savez_compressed(physical / "replay.npz", **dict(signals))
    repeated, _ = diagnostic.select_manifest(pd.concat([frames, replay]), physical)
    pd.testing.assert_frame_equal(manifest, repeated)


@pytest.mark.parametrize("fault", ["background", "boundary", "visibility", "contact", "load", "mass"])
def test_selection_enforces_actual_visual_and_physical_exposure(property_data, fault):
    frames, physical, *_ = property_data
    selected = frames.video_id.eq("p000")
    if fault == "background":
        frames.loc[selected, "objects"] = json.dumps([{"category": "doll", "visible_px": 200},
                                                      {"category": "pyramid", "visible_px": 1}])
    elif fault == "boundary":
        frames.loc[selected & frames.frame_idx.eq(4), "clip"] = "other/clip.npz"
    elif fault == "visibility":
        frames.loc[selected, "objects"] = json.dumps([{"category": "doll", "visible_px": 99}])
    elif fault == "contact":
        frames.loc[selected, "contact_left"] = False
    else:
        with np.load(physical / "p000.npz") as signals:
            values = dict(signals)
        values["load" if fault == "load" else "mass_kg"][:] = 0 if fault == "load" else 0.2
        np.savez_compressed(physical / "p000.npz", **values)
    manifest, _ = diagnostic.select_manifest(frames, physical)
    assert "p000" not in set(manifest.video_id)
    assert manifest[manifest.category == "doll"].text.value_counts().to_dict() == {"heavy": 2, "light": 2}


def test_zero_pixel_background_is_not_exposure_and_dev_excludes_test(property_data):
    frames, physical, *_ = property_data
    train = frames.video_id.eq("p000")
    frames.loc[train, "objects"] = json.dumps([{"category": "doll", "visible_px": 200},
                                               {"category": "pyramid", "visible_px": 0}])
    manifest, _ = diagnostic.select_manifest(frames, physical)
    assert "p000" in set(manifest.video_id)
    dev_video = frames.loc[frames.category.eq("clock"), "video_id"].iloc[0]
    frames.loc[frames.video_id.eq(dev_video), "objects"] = json.dumps([
        {"category": "clock", "visible_px": 200}, {"category": "stapler", "visible_px": 1}])
    manifest, _ = diagnostic.select_manifest(frames, physical)
    assert dev_video not in set(manifest.video_id)


def test_normalization_uses_only_unique_accessible_training_seconds(property_data):
    frames, physical, *_ = property_data
    manifest, _ = diagnostic.select_manifest(frames, physical)
    store, stats, seconds = diagnostic.load_store(frames, manifest, physical)
    assert stats["unique_training_seconds"] == 36 * 8  # center 2 plus/minus 5, within 0..8
    assert not seconds.duplicated().any()
    assert set(seconds.video_id) == set(manifest[manifest.split == "train"].video_id)
    assert seconds.frame_idx.max() == 7
    # Neither evaluation observations nor an unexposed training second enter normalization.
    for video in manifest.video_id:
        with np.load(physical / f"{video}.npz") as signals:
            values = dict(signals)
        if video in set(manifest[manifest.split != "train"].video_id):
            values["load"] *= 10000
        values["load"][8] = 1e8
        np.savez_compressed(physical / f"{video}.npz", **values)
    changed, changed_stats, changed_seconds = diagnostic.load_store(frames, manifest, physical)
    assert changed_stats == stats
    pd.testing.assert_frame_equal(seconds, changed_seconds)
    train_ids = [store["row"][video] for video in seconds.video_id.unique()]
    np.testing.assert_array_equal(store["groups"]["load"][train_ids], changed["groups"]["load"][train_ids])


def test_mass_category_and_identity_metadata_never_become_body_inputs(property_data, upstream):
    frames, physical, *_ = property_data
    manifest, _ = diagnostic.select_manifest(frames, physical)
    original, stats, _ = diagnostic.load_store(frames, manifest, physical)
    changed = manifest.copy()
    changed["mass_kg"] = 999
    changed["category"] = "unrelated"
    changed["clip"] = "unrelated"
    changed["text"] = "unrelated"
    altered, altered_stats, _ = diagnostic.load_store(frames, changed, physical)
    assert original["groups"].keys() == {"load"}
    assert stats == altered_stats
    np.testing.assert_array_equal(original["groups"]["load"], altered["groups"]["load"])
    assert diagnostic.VOCAB == {"<pad>": 0, "heavy": 1, "light": 2}


def test_language_loss_gradients_and_binary_scoring(property_data, upstream):
    rows, datasets, data, _ = _prepared_inputs(property_data, upstream)
    model = LearnerMIL(len(diagnostic.VOCAB), 8, 0.1, {"load": 10}).float()
    ids = np.array([0, 1, 3, 4])
    _, visual, tokens, lengths = upstream.collate([datasets["train"][int(i)] for i in ids])
    data["train"].dropout = 0
    ctx = data["train"].batch(ids, torch.Generator().manual_seed(0))
    _, _, scores = model.pair_scores(visual, tokens, lengths, ctx)
    assert scores.shape == (4, 4)  # square matching loss, even when word labels repeat
    loss = weighted_nce(model.scale() * scores, torch.ones(4))
    loss.backward()
    assert model.word.weight.grad.abs().sum() > 0
    assert model.vproj[2].weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.encoder.parameters())
    assert sum(parameter.grad.abs().sum() for parameter in model.encoder.parameters()) > 0
    binary = diagnostic.score_words(model, datasets["test"], data["test"], "cpu", upstream)
    assert binary.shape == (len(rows["test"]), 2)
    assert np.isfinite(binary).all()


def test_ties_receive_half_credit_in_either_word_order():
    scores = np.array([[0, 0], [1, 0], [0, 1], [-1, -2]])
    texts = ["heavy", "heavy", "light", "light"]
    np.testing.assert_array_equal(diagnostic.word_credit(scores, texts), [0.5, 1, 1, 0])


def test_donor_controls_preserve_category_and_swap_entire_context(property_data, upstream):
    rows, datasets, data, store = _prepared_inputs(property_data, upstream)
    donors = diagnostic.donor_derangements(rows["test"])
    assert donors == diagnostic.donor_derangements(rows["test"])
    for donor in donors:
        assert sorted(donor) == list(range(len(rows["test"])))
        assert (np.array(donor) != np.arange(len(donor))).all()
        assert rows["test"].category.tolist() == rows["test"].iloc[donor].category.tolist()
    captured = []

    class Capture(torch.nn.Module):
        def pair_scores(self, visual, tokens, lengths, ctx):
            captured.append(ctx)
            return None, None, torch.zeros(len(visual), 2)

    # A missing donor second must move with the load and clip-reset segmentation.
    session = int(data["test"].ps[donors[0][0]])
    center = int(data["test"].pt[donors[0][0]])
    data["test"].present[session, center] = False
    diagnostic.score_words(Capture(), datasets["test"], data["test"], "cpu", upstream, donors=donors[0])
    expected = data["test"].batch(donors[0], None, training=False)
    for key in ("missing", "seg", "same"):
        torch.testing.assert_close(captured[0][key], expected[key])
    torch.testing.assert_close(captured[0]["groups"]["load"], expected["groups"]["load"])


def test_matched_one_epoch_smoke_and_first_dev_maximum(property_data, upstream, monkeypatch):
    rows, datasets, data, _ = _prepared_inputs(property_data, upstream)
    output = _output_directory(property_data[3])
    settings = {**diagnostic.SETTINGS, "epochs": 2}
    monkeypatch.setattr(diagnostic, "score_words", lambda model, dataset, *a, **kw: np.zeros((len(dataset), 2)))
    metrics = []
    for body in (False, True):
        model, record = diagnostic._train_arm(0, body, datasets, rows, data, 8, "cpu", upstream, settings,
                                               output / str(body), time.monotonic() + 30)
        metrics.append(record)
        assert record["selected_epoch"] == 0
        assert record["epochs_completed"] == 2
        assert (output / str(body) / "model.pt").exists()
        assert model.word.embedding_dim == 512
    for key in ("common_initialization_sha256", "minibatch_order_sha256"):
        assert metrics[0][key] == metrics[1][key]


def test_prepare_freezes_inputs_and_refuses_overwrite(property_data):
    frames, physical, cache, output, root = property_data
    protocol = diagnostic.prepare(output, root, cache, physical)
    assert protocol["audit"]["balanced_counts"] == {"train": 36, "dev": 12, "test": 12}
    assert {path.name for path in output.iterdir()} == {"protocol.json", "manifest.parquet", "normalization_seconds.parquet"}
    diagnostic._verify_frozen(output, protocol)
    with pytest.raises(FileExistsError, match="empty"):
        diagnostic.prepare(output, root, cache, physical)
    changed = json.loads((output / "protocol.json").read_text())
    changed["settings"]["epochs"] = 1
    with pytest.raises(ValueError, match="canonical"):
        diagnostic._verify_frozen(output, changed)
    (output / "manifest.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prepared artifact"):
        diagnostic._verify_frozen(output, protocol)


def test_paired_gate_requires_every_frozen_criterion():
    video = np.zeros((3, 22))
    body = np.ones((3, 22))
    shuffled = np.full((3, 22), 0.5)
    result = diagnostic.paired_summary(video, body, shuffled)
    assert result["capability_pass"]
    assert result["paired_clip_ci95"] == [1, 1]
    assert not diagnostic.paired_summary(video, body, body)["capability_pass"]
    body[0] = 0
    assert not diagnostic.paired_summary(video, body, shuffled)["criteria"]["every_seed_positive"]


def test_budget_expiry_is_explicit_before_model_creation(property_data, upstream):
    rows, datasets, data, _ = _prepared_inputs(property_data, upstream)
    output = _output_directory(property_data[3])
    with pytest.raises(TimeoutError, match="incomplete"):
        diagnostic._train_arm(0, True, datasets, rows, data, 8, "cpu", upstream,
                               diagnostic.SETTINGS, output / "expired", time.monotonic() - 1)
    assert not (output / "expired").exists()


def test_synthetic_comparison_smoke_preserves_predictions_and_refuses_retry(property_data, upstream, monkeypatch):
    _, physical, cache, output, root = property_data
    monkeypatch.setitem(diagnostic.SETTINGS, "epochs", 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    diagnostic.prepare(output, root, cache, physical)
    result = diagnostic.train_prepared(output)
    assert result["completed"]
    assert result["final_frozen_source_and_code_verification"] == "passed"
    assert result["n_test_source_clips"] == 12
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete" and len(run["completed_models"]) == 6
    for seed in (0, 1, 2):
        video = json.loads((output / f"video_s{seed}/metrics.json").read_text())
        body = json.loads((output / f"load_s{seed}/metrics.json").read_text())
        for key in ("common_initialization_sha256", "minibatch_order_sha256"):
            assert video[key] == body[key]
        predictions = pd.read_parquet(output / f"load_s{seed}/predictions.parquet")
        shuffled = pd.read_parquet(output / f"load_s{seed}/shuffled_predictions.parquet")
        assert len(predictions) == 12 and len(shuffled) == 120
        assert {"picture_credit", "score_heavy", "score_light", "credit"} <= set(predictions.columns)
    with pytest.raises(FileExistsError, match="never overwritten"):
        diagnostic.train_prepared(output)


def test_partial_abort_never_reports_a_completed_comparison(property_data, upstream, monkeypatch):
    _, physical, cache, output, root = property_data
    diagnostic.prepare(output, root, cache, physical)

    def expired(*args, **kwargs):
        raise TimeoutError("incomplete synthetic interruption")

    monkeypatch.setattr(diagnostic, "_train_arm", expired)
    with pytest.raises(TimeoutError, match="incomplete"):
        diagnostic.train_prepared(output)
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "aborted_partial"
    assert run["completed_models"] == []
    assert not (output / "comparison.json").exists()
