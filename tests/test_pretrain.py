"""Synthetic, CPU-only checks of the frozen sensory-pretraining protocol."""

import hashlib
import json
import time

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.common import _output_directory
from nursery.learner import pretrain as step3


pytestmark = pytest.mark.usefixtures("offline")


@pytest.fixture
def small_exposure(episodes, upstream):
    root, run = episodes
    output = _output_directory(run / "pretrain")
    rng = np.random.default_rng(4)
    images = rng.normal(size=(8, 5, 17, 8)).astype(np.float32)
    contact = rng.integers(0, 2, size=(8, 5, 22)).astype(np.float32)
    targets = {"contact": contact, "change": np.concatenate([np.diff(contact, axis=1), np.zeros((8, 1, 22))], 1).astype(np.float32),
               "aperture": rng.normal(size=(8, 5, 2)).astype(np.float32),
               "in_view": rng.integers(0, 2, size=(8, 5, 2)).astype(np.float32)}
    rows = pd.DataFrame({"video_id": [f"v{i}" for i in range(8)], "frame_idx": 2,
                         "text": ["mug", "bowl"] * 4, "weight_none": 1.0})
    vocab = {"<pad>": 0, "mug": 1, "bowl": 2}
    caches = [images.reshape(-1, 17, 8)]
    lut = {f"v{i}|{f}": (0, 5 * i + f) for i in range(8) for f in range(5)}
    datasets = {"train": upstream.FramePairs(caches, lut, rows, vocab, 2, False)}
    return output, images, targets, rows, vocab, datasets


def test_window_sampling_uses_first_eligible_occurrence_and_no_hidden_metadata():
    rows = []
    for video, clip, length, subject in [("z", "a", 9, "s1"), ("a", "a", 3, "s1"),
                                         ("b", "a", 8, "s1"), ("c", "b", 7, "s1"),
                                         ("d", "dev", 9, "s1"), ("s8", "held", 9, "s8")]:
        rows.extend({"video_id": video, "clip": clip, "frame_idx": t, "subject": subject,
                     "mass": 1000, "held_category": "oracle", "text": "hidden"} for t in range(length))
    frames = pd.DataFrame(rows)
    part = {v: np.full(10, 1 if v == "d" else 0) for v in frames.video_id.unique()}
    available = set(zip(frames.video_id, frames.frame_idx))
    result = step3.select_windows(frames, part, available)
    assert result["clip"].tolist() == ["a", "b"]
    assert result.video_id.tolist() == ["b", "c"]
    expected_rng = np.random.default_rng(0)
    assert result.start.tolist() == [expected_rng.choice(np.arange(4)), expected_rng.choice(np.arange(3))]
    assert set(result) == {"clip", "video_id", "subject", "start", "frame_idx", "mask"}
    changed = frames.copy()
    changed[["mass", "held_category", "text"]] = [0, "different", "other"]
    pd.testing.assert_frame_equal(result, step3.select_windows(changed, part, available))
    # A missing second invalidates every overlapping start; no stitching across gaps.
    available -= {("b", 3)}
    changed = step3.select_windows(frames, part, available)
    assert changed.iloc[0].video_id == "z"


def test_normalization_fits_selected_training_seconds_and_targets_have_four_edges():
    windows = pd.DataFrame({"video_id": ["train", "other"], "start": [1, 2]})
    arrays = {video: {t: {"contact": np.full(22, t % 2, np.float32),
                            "aperture": np.array([t, 2 * t], np.float32),
                            "in_view": np.array([t % 2, 1], np.float32)} for t in range(10)}
              for video in windows.video_id}
    train = windows.iloc[:1]
    values, stats, seconds = step3.window_targets(train, arrays)
    assert seconds.frame_idx.tolist() == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(values["aperture"].mean((0, 1)), 0, atol=1e-6)
    np.testing.assert_array_equal(values["change"][:, :4], np.diff(values["contact"], axis=1))
    assert not values["change"][:, -1].any()
    arrays["train"][6]["aperture"][:] = 1e20
    arrays["other"][3]["aperture"][:] = 1e20
    changed, changed_stats, _ = step3.window_targets(train, arrays)
    assert changed_stats == stats
    for key in values:
        np.testing.assert_array_equal(changed[key], values[key])
    held, held_stats, _ = step3.window_targets(windows.iloc[1:], arrays, stats)
    assert held_stats == stats and held["aperture"].max() > 1e10
    ctx = step3.sensory_context(values, [0], "cpu")
    assert ctx["frame_valid"].all()
    assert ctx["frame_next_valid"].tolist() == [[True, True, True, True, False]]


def test_only_declared_short_dev_clip_is_excluded_while_training_stays_strict():
    short = "s1/stamp_stamp_1.npz"
    frames = pd.DataFrame([{"video_id": "short", "frame_idx": t, "clip": short, "subject": "s1"}
                           for t in range(117, 120)] +
                          [{"video_id": "full", "frame_idx": t, "clip": "s1/eligible.npz", "subject": "s1"}
                           for t in range(7)])
    available = set(zip(frames.video_id, frames.frame_idx))
    dev_part = {video: np.ones(120, np.int64) for video in ("short", "full")}
    selected = step3.select_windows(frames, dev_part, available, partition=1)
    assert selected["clip"].tolist() == ["s1/eligible.npz"]
    assert selected["mask"].tolist() == ["11111"]
    train_part = {video: np.zeros(120, np.int64) for video in ("short", "full")}
    with pytest.raises(ValueError, match="stamp_stamp_1"):
        step3.select_windows(frames, train_part, available)
    unexpected = frames.copy()
    unexpected.loc[unexpected.video_id.eq("short"), "clip"] = "s1/unexpected_short.npz"
    with pytest.raises(ValueError, match="unexpected_short"):
        step3.select_windows(unexpected, dev_part, available, partition=1)


def test_derangement_is_full_coverage_same_subject_and_coherent(small_exposure):
    _, _, targets, *_ = small_exposure
    windows = pd.DataFrame({"clip": [f"c{i}" for i in range(8)], "subject": ["s1"] * 4 + ["s2"] * 4, "mask": "11111"})
    donors = step3.donor_derangement(windows)
    assert sorted(donors) == list(range(8))
    assert (donors != np.arange(8)).all()
    assert windows.subject.tolist() == windows.iloc[donors].subject.tolist()
    np.testing.assert_array_equal(donors, step3.donor_derangement(windows))
    ctx = step3.sensory_context(targets, donors, "cpu")
    for key, value in targets.items():
        np.testing.assert_array_equal(ctx["frame_targets"][key], value[donors])
    assert not ctx["frame_next_valid"][:, -1].any()


def test_temporal_loss_is_four_pair_mean_and_all_five_frames_receive_gradient(upstream):
    model = step3.make_model(0, 3, 8, "cpu", sensory=True).eval()
    images = torch.randn(3, 5, 17, 8, requires_grad=True)
    episode = torch.arange(3)
    loss = step3.temporal_loss(model, images, episode)
    expected = []
    for i in range(4):
        ctx = {"bag_sel": torch.arange(3), "bag_images": images, "bag_episode": episode,
               "bag_ab": torch.tensor([i, i + 1]).expand(3, 2)}
        expected.append(model.consistency_loss(torch.ones(3), ctx, model.scale()))
    torch.testing.assert_close(loss, torch.stack(expected).mean())
    loss.backward()
    assert (images.grad.abs().sum((0, 2, 3)) > 0).all()
    assert model.vproj[2].weight.grad.abs().sum() > 0
    assert model.word.weight.grad is None
    assert all(p.grad is None for p in model.encoder.parameters())


def test_prediction_loss_uses_all_five_frames_but_not_sixth_change(small_exposure):
    _, images, targets, *_ = small_exposure
    model = step3.make_model(0, 3, 8, "cpu", sensory=True).eval()
    visual = torch.tensor(images[:3], requires_grad=True)
    ctx = step3.sensory_context(targets, np.arange(3), "cpu")
    loss = model.predict_loss(model.enc_regions(visual.flatten(1, 2)), ctx)
    changed = {**ctx, "frame_targets": {k: v.clone() for k, v in ctx["frame_targets"].items()}}
    changed["frame_targets"]["change"][:, -1] = 1e10
    torch.testing.assert_close(loss, model.predict_loss(model.enc_regions(visual.flatten(1, 2)), changed))
    loss.backward()
    assert (visual.grad.abs().sum((0, 2, 3)) > 0).all()


def test_descriptive_components_equal_existing_loss_and_references_follow_frozen_rules(small_exposure):
    _, images, targets, *_ = small_exposure
    model = step3.make_model(0, 3, 8, "cpu", sensory=True).eval()
    ctx = step3.sensory_context(targets, np.arange(8), "cpu")
    regions = model.enc_regions(torch.tensor(images).flatten(1, 2))
    output = model.region_head(torch.relu(model.region_in(regions.view(8, 5, 17, -1))).flatten(2))
    components = step3.sensory_components(output, ctx)
    assert components["total"] == pytest.approx(float(model.predict_loss(regions, ctx).detach()))
    actual = step3.sensory_readout(model, images, targets, "cpu", time.monotonic() + 30, batch=3)
    assert actual == pytest.approx(components, abs=2e-6)
    refs = step3.sensory_references(targets, targets)
    expected_change = np.mean(targets["change"][:, :4] ** 2)
    assert refs["persistence"]["change"] == pytest.approx(expected_change)
    expected_mean_change = np.mean((targets["change"][:, :4] - targets["change"][:, :4].mean((0, 1))) ** 2)
    assert refs["mean"]["change"] == pytest.approx(expected_mean_change)
    mean = targets["aperture"].mean((0, 1))
    previous = np.concatenate([np.broadcast_to(mean, (8, 1, 2)), targets["aperture"][:, :-1]], 1)
    assert refs["persistence"]["aperture"] == pytest.approx(np.mean((previous - targets["aperture"]) ** 2))


def test_three_pretraining_arms_have_common_initialization_and_orders(small_exposure):
    output, images, targets, _, vocab, _ = small_exposure
    settings = {**step3.SETTINGS, "pretrain_passes": 2, "pretrain_batch": 4, "pretrain_updates": 4}
    metrics = []
    for arm in ("V", "A", "S"):
        directory = output / arm
        directory.mkdir()
        model, state, record = step3.pretrain_arm(1, arm, images, targets, np.roll(np.arange(8), 1), len(vocab),
                                           "cpu", settings, directory, time.monotonic() + 60)
        assert record["pretrain_updates"] == 4
        assert (directory / "pretrain_final.pt").is_file()
        assert all(not key.startswith("word") for key in state)
        metrics.append(record)
    for key in ("pretrain_initial_projection_sha256", "pretrain_order_sha256"):
        assert len({m[key] for m in metrics}) == 1


def test_transfer_copies_only_projection_and_uses_fresh_words_scale_and_optimizer(small_exposure, monkeypatch, upstream):
    output, _, _, rows, vocab, datasets = small_exposure
    source = step3.make_model(2, len(vocab), 8, "cpu", sensory=True)
    with torch.no_grad():
        source.word.weight.fill_(999)
        source.logit_scale.fill_(99)
        source.vproj[2].weight.add_(1)
    projection = {k: v.detach().clone() for k, v in source.vproj.state_dict().items()}
    target = step3.make_model(2, len(vocab), 8, "cpu", projection=projection)
    reference = step3.make_model(2, len(vocab), 8, "cpu")
    assert target.encoder is None
    torch.testing.assert_close(target.word.weight, reference.word.weight)
    torch.testing.assert_close(target.logit_scale, reference.logit_scale)
    for key, value in projection.items():
        torch.testing.assert_close(target.vproj.state_dict()[key], value)
    assert not torch.equal(target.vproj[2].weight, reference.vproj[2].weight)
    optimizers, original = [], torch.optim.AdamW

    def optimizer(*args, **kwargs):
        opt = original(*args, **kwargs)
        assert not opt.state
        optimizers.append(opt)
        return opt

    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    settings = {**step3.SETTINGS, "epochs": 3, "batch": 4}
    results = []
    for arm in ("V", "A"):
        directory = output / f"language_{arm}"
        directory.mkdir()
        dev = iter([0.3, 0.4, 0.3])
        picture = iter([0.4, 0.4, 0.2])
        def score(model, split):
            return next(dev if split == "clips_dev" else picture)
        model, clip, picture_state, metrics = step3.transfer_arm(2, projection, datasets, {"train": rows}, vocab,
                                         "cpu", upstream, settings, directory, time.monotonic() + 60, score)
        assert metrics["clip_selected_epoch"] == 1
        assert metrics["picture_selected_epoch"] == 0
        assert metrics["epochs_completed"] == 3
        results.append(metrics)
    assert optimizers[0] is not optimizers[1]
    for key in ("fresh_language_state_sha256", "language_order_sha256"):
        assert results[0][key] == results[1][key]


def test_bootstrap_is_trial_weighted_with_equal_seed_averages():
    # One large clip and two single-query clips distinguish trial and clip weighting.
    clips = ["large"] * 8 + ["small", "tiny"]
    video = np.zeros((3, 10))
    aligned = np.array([[1] * 8 + [0, 0], [1] * 8 + [1, 0], [1] * 8 + [0, 1]])
    shuffled = np.zeros((3, 10))
    settings = {**step3.SETTINGS, "bootstrap_draws": 100}
    result = step3.paired_summary({"V": video, "A": aligned, "S": shuffled}, clips, settings)
    expected = aligned.mean(1).mean()
    assert result["accuracy"]["A"] == expected
    assert result["contrasts"]["A-V"]["gain"] == expected
    rng = np.random.default_rng(0)
    draws = rng.integers(3, size=(100, 3))
    totals = np.array([[8, 0, 0], [8, 1, 0], [8, 0, 1]])
    # np.unique sorts large,small,tiny in this example.
    expected_boot = (totals[:, draws].sum(2) / np.array([8, 1, 1])[draws].sum(1)).mean(0)
    np.testing.assert_allclose(result["contrasts"]["A-V"]["target_clip_ci95"], np.percentile(expected_boot, [2.5, 97.5]))
    failed = aligned.copy()
    failed[0] = 0
    assert not step3.paired_summary({"V": video, "A": failed, "S": shuffled}, clips, settings)["transfer_gate_pass"]


def test_deadline_aborts_before_update_and_preserves_existing_artifacts(small_exposure):
    output, images, targets, _, vocab, _ = small_exposure
    previous = output / "keep.txt"
    previous.write_text("prior user artifact")
    before = hashlib.sha256(previous.read_bytes()).hexdigest()
    with pytest.raises(TimeoutError, match="incomplete"):
        step3.pretrain_arm(0, "V", images, targets, np.arange(8), len(vocab), "cpu", step3.SETTINGS,
                           output, time.monotonic() - 1)
    assert hashlib.sha256(previous.read_bytes()).hexdigest() == before
    assert not (output / "pretrain_final.pt").exists()


def test_training_refuses_nonempty_roots_without_modifying_prior_files(episodes):
    _, run = episodes
    output = _output_directory(run / "prepared")
    (output / "prior.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="overwritten"):
        step3.train_prepared(output)
    assert (output / "prior.txt").read_text() == "preserve"


def test_frozen_verification_detects_source_and_prepared_mutation(episodes, monkeypatch):
    _, run = episodes
    output = _output_directory(run / "prepared")
    source = output / "source.txt"
    source.write_text("original")
    prepared = output / "windows.parquet"
    prepared.write_bytes(b"frozen")
    monkeypatch.setattr(step3, "_code_hashes", lambda: {"code": "hash"})
    protocol = {"settings": step3.SETTINGS, "counts": step3.COUNTS, "gate": step3.GATE,
                "code_sha256": {"code": "hash"}, "source_sha256": {str(source): step3._sha(source)},
                "prepared_sha256": {prepared.name: step3._sha(prepared)}}
    step3._verify_frozen(output, protocol)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        step3._verify_frozen(output, protocol)
    source.write_text("original")
    prepared.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        step3._verify_frozen(output, protocol)


def test_new_cli_preserves_original_training_dispatch(monkeypatch):
    from nursery.learner import __main__ as cli
    import sys

    observed = []
    monkeypatch.setattr(cli, "train_run", lambda *args: observed.append(args))
    monkeypatch.setattr(sys, "argv", ["learner", "train", "--run", "existing", "--condition", "none", "--seed", "2"])
    cli.main()
    assert observed[0][:3] == ("existing", "none", 2)
    assert observed[0][5] == "max"
    monkeypatch.setattr(step3, "prepare", lambda output, run: {"counts": observed.append((output, run))})
    monkeypatch.setattr(sys, "argv", ["learner", "pretrain", "--phase", "prepare", "--output", "ignored", "--run", "corrected"])
    cli.main()
    assert observed[-1] == ("ignored", "corrected")


@pytest.fixture
def frozen_small_run(episodes, upstream, monkeypatch):
    """Complete miniature corrected corpus, using actual prepare/train entry points."""
    root, run = episodes
    run = _output_directory(run)
    body_root = root / "body"
    body_root.mkdir()
    rng, frames, pairs, sessions = np.random.default_rng(9), [], [], []
    words = ["mug", "bowl", "apple", "banana"]
    for partition in ("train", "clips_dev", "clips_test", "s8"):
        for i, word in enumerate(words):
            video = f"{partition}_{i}"
            subject = "s8" if partition == "s8" else "s1"
            sessions.append({"video_id": video, "subject": subject})
            for t in range(7):
                frames.append({"video_id": video, "frame_idx": t, "clip": f"{partition}/c{i}", "subject": subject,
                               "held_category": word, "objects": json.dumps([{"category": word, "visible_px": 200}])})
            for t in (2, 3, 4):
                pairs.append({"video_id": video, "frame_idx": t, "text": word, "referent": word,
                              "named": True, "referent_visible": True, "weight_none": 1.0})
            joints = np.zeros((7, 55, 3), np.float32)
            joints[..., 2] = -2
            np.savez(body_root / f"{video}.npz", frame_idx=np.arange(7),
                     camera=np.broadcast_to(np.eye(4), (7, 4, 4)), joints=joints,
                     left_hand_pose=rng.normal(size=(7, 45)), right_hand_pose=rng.normal(size=(7, 45)),
                     hand_contact=np.packbits(rng.integers(0, 2, size=(7, 1556), dtype=np.uint8), axis=1))
    frames = pd.DataFrame(frames)
    frames.to_parquet(root / "frames.parquet", index=False)
    pd.DataFrame(sessions).to_parquet(root / "sessions.parquet", index=False)
    pd.DataFrame(pairs).to_parquet(run / "manifest.parquet", index=False)
    step3._json(run / "generator.json", {"episodes": str(root), "config": {"training": {"min_freq": 1}}})
    for cache in ("train_cache", "probes_cache", "konkle_cache"):
        directory = _output_directory(run / cache)
        index = frames[["video_id", "frame_idx"]].copy()
        index["row"] = np.arange(len(index))
        index.to_parquet(directory / "index.parquet", index=False)
        np.save(directory / "emb.f16.npy", rng.normal(size=(len(index), 17, 8)).astype(np.float16))
    probes = pd.DataFrame([{"video_id": f"train_{i}", "frame_idx": t, "category": word, "split": split}
                           for split in ("dev", "test") for i, word in enumerate(words) for t in range(4)])
    probes.to_parquet(run / "probes_frames.parquet", index=False)
    probes[probes.split == "test"].to_parquet(run / "konkle_frames.parquet", index=False)
    def partition(frame_table, *args):
        labels = {"train": 0, "clips_dev": 1, "clips_test": 2, "s8": 0}
        result = {v: np.full(7, labels[v.rsplit("_", 1)[0]], np.int64) for v in frame_table.video_id.unique()}
        return result, {"s1": {"train": 4, "dev": 4, "test": 4}}
    monkeypatch.setattr(step3, "clip_partition", partition)
    monkeypatch.setattr(step3, "SETTINGS", {**step3.SETTINGS, "epochs": 1, "batch": 12,
                        "pretrain_passes": 1, "pretrain_batch": 4, "pretrain_updates": 1})
    monkeypatch.setattr(step3, "COUNTS", {"pretrain_clips": 4, "train_pairs": 12,
                                       "clips_dev": [12, 4], "clips_test": [12, 4], "s8": [12, 4],
                                       "sensory_dev_clips": 4, "sensory_dev_source_clips": 4})
    monkeypatch.setattr(step3, "DEV_EXCLUSIONS", {})
    part, _ = partition(frames)
    rows, _ = step3.language_rows(run, frames, part, upstream)
    _, lut = upstream.load_multi([run / "train_cache"])
    trial_manifest = {}
    for split in ("clips_dev", "clips_test", "s8"):
        table = rows[rows.partition == split].reset_index(drop=True)
        context = step3.evaluate.body_present_context(table, frames, lut, 2, 5, table.subject.to_numpy())
        sampled = step3.evaluate._body_trials(table, context, 0, table.subject.to_numpy())
        trial_manifest[split] = {"trial_rows": [i for i, _, _ in sampled], "correct_slots": [a for _, _, a in sampled],
                                "candidate_identities": [[list(context["identity"][j]) for j in c] for _, c, _ in sampled]}
    step3._json(run / "trial_manifest.json", trial_manifest)
    output = root.parent / "step3"
    protocol = step3.prepare(output, run)
    return output, protocol


def test_prepare_freezes_exposures_donors_trials_and_refuses_overwrite(frozen_small_run):
    output, protocol = frozen_small_run
    assert {p.name for p in output.iterdir()} == step3.PREPARED
    assert protocol["normalization"]["unique_training_seconds"] == 20
    assert protocol["sensory_dev_audit"] == {"source_clips": 4, "eligible_clips": 4, "excluded_clips": {},
                                            "target_frames": 20, "internal_change_edges": 16}
    assert "nursery/learner/hand_patches.json" in protocol["code_sha256"]
    windows = pd.read_parquet(output / "windows.parquet")
    dev = pd.read_parquet(output / "dev_windows.parquet")
    assert set(windows["clip"]).isdisjoint(dev["clip"])
    assert (windows.donor.to_numpy() != np.arange(4)).all()
    trials = json.loads((output / "trials.json").read_text())
    assert all(t["0"] == t["1"] == t["2"] for t in trials.values())
    step3._verify_frozen(output, protocol)
    with pytest.raises(FileExistsError, match="empty"):
        step3.prepare(output, protocol["run"])


def test_end_to_end_nine_synthetic_runs_preserve_inputs_and_stop_at_fixed_budget(frozen_small_run, monkeypatch):
    output, protocol = frozen_small_run
    monkeypatch.setattr(step3, "_training_device", lambda: "cpu")
    # Exercise real phase 1, transfer, model selection, all readouts and summaries.
    result = step3.train_prepared(output)
    assert result["completed"] and result["n_test_queries"] == 12
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete" and len(run["completed_runs"]) == 9
    for seed in (0, 1, 2):
        records = [json.loads((output / f"{arm}_s{seed}" / "metrics.json").read_text()) for arm in ("V", "A", "S")]
        assert all(record["pretrain_updates"] == 1 and record["epochs_completed"] == 1 for record in records)
        assert "sensory_dev" not in records[0] and all("sensory_dev" in r for r in records[1:])
    step3._verify_frozen(output, protocol)
    with pytest.raises(FileExistsError, match="overwritten"):
        step3.train_prepared(output)


def test_timeout_records_incomplete_preserves_partial_and_never_publishes_comparison(frozen_small_run, monkeypatch):
    output, protocol = frozen_small_run
    monkeypatch.setattr(step3, "_training_device", lambda: "cpu")
    def timeout(*args):
        directory = args[-2]
        (directory / "partial.txt").write_text("preserve partial evidence")
        raise TimeoutError("fixed budget expired")
    monkeypatch.setattr(step3, "pretrain_arm", timeout)
    with pytest.raises(TimeoutError, match="fixed budget"):
        step3.train_prepared(output)
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "aborted_partial" and run["completed_runs"] == []
    assert run["final_frozen_verification"] == "passed"
    assert (output / "V_s0" / "partial.txt").read_text() == "preserve partial evidence"
    assert not (output / "comparison.json").exists()
    step3._verify_frozen(output, protocol)
