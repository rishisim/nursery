"""Synthetic checks for the fixed study through the shared learner runner."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from nursery.learner.evaluate import ShuffledBodyHistory, _body_trials
from nursery.learner.train import _study_body_exclusion, _study_center

pytestmark = pytest.mark.usefixtures("offline")


def test_training_statistics_only_use_selected_experience():
    import torch

    class Selected:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return i, torch.full((3, 2), float(i + 1))

    np.testing.assert_allclose(_study_center(Selected()), [1.5, 1.5])
    frames = pd.DataFrame({"video_id": ["v"] * 10, "frame_idx": range(10), "clip": ["a"] * 4 + ["b"] * 6})
    chosen = pd.DataFrame({"video_id": ["v"], "frame_idx": [3]})
    excluded = _study_body_exclusion(chosen, frames, ["v"], 2, steps=10)
    np.testing.assert_array_equal(np.flatnonzero(~excluded[0]), [1, 2, 3])


def test_frozen_panel_checks_and_complete_history_shuffle():
    import torch

    rows = pd.DataFrame({"video_id": list("abcd"), "frame_idx": [0] * 4,
                         "referent": list("wxyz"), "subject": ["s1"] * 4, "clip": list("abcd")})
    context = {"identity": list(zip(rows.video_id, rows.frame_idx, rows.referent)), "subjects": rows.subject.tolist(),
               "pools": [np.array([j for j in range(4) if i != j]) for i in range(4)]}
    panel = [{"target": i, "candidates": [0, 1, 2, 3], "correct": i} for i in range(4)]
    a = _body_trials(rows, context, 0, None, panel)
    b = _body_trials(rows, context, 2, None, panel)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    with pytest.raises(ValueError, match="every target encounter"):
        _body_trials(rows, context, 0, None, panel[:1])
    rows["role"] = ["target", "foil", "foil", "foil"]
    assert len(_body_trials(rows, context, 0, None, panel[:1])) == 1

    class Histories:
        def batch(self, idx, generator, training=False):
            ix = torch.tensor(idx)
            return {"groups": {k: ix[:, None, None] + n for n, k in enumerate(("contact", "fingers", "arm", "load", "skin"))},
                    "missing": ix[:, None], "seg": ix[:, None] * 10, "same": ix[:, None] == 0}

    donors = [1, 2, 3, 0]
    shuffled = ShuffledBodyHistory(Histories(), donors, rows)
    got = shuffled.batch(np.array([0, 3]), None)
    expected = Histories().batch([1, 0], None)
    for key in ("missing", "seg", "same"):
        torch.testing.assert_close(got[key], expected[key])
    for key in expected["groups"]:
        torch.testing.assert_close(got["groups"][key], expected["groups"][key])
    with pytest.raises(ValueError, match="evaluation only"):
        shuffled.batch([0], None, training=True)
    with pytest.raises(ValueError, match="different source clip"):
        ShuffledBodyHistory(Histories(), [0, 1, 2, 3], rows)
    with pytest.raises(ValueError, match="permutation"):
        ShuffledBodyHistory(Histories(), [1, 2, 3, 1], rows)


def test_shared_study_runner_pairs_training_and_selects_before_test(episodes, upstream, monkeypatch):
    import torch
    import torch._dynamo  # Load capability introspection before replacing device availability.
    from test_learner import _write_body, _write_physical
    from nursery.learner import train
    from nursery.learner.video_only import _video_class
    from nursery.learner.additive import _model_class

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    root, run = episodes
    run.mkdir()
    cache = run / "cache"
    cache.mkdir()
    frames, manifest = [], []
    words = ["dax", "blicket", "wug", "fep"]
    for part_no, partition in enumerate(("train", "dev", "test")):
        for i, word in enumerate(words):
            video = f"{partition}_{i}"
            frames.append({"video_id": video, "frame_idx": 0, "clip": video, "subject": "s1",
                           "held_category": word, "objects": json.dumps([{"category": word, "visible_px": 100}])})
            manifest.append({**frames[-1], "partition": partition,
                             "text": f"Look, {word}." + (" forbidden" if partition != "train" else "")})
            manifest[-1]["referent"] = word
    frames = pd.DataFrame(frames)
    frames.to_parquet(root / "frames.parquet", index=False)
    pd.DataFrame({"video_id": frames.video_id, "subject": "s1"}).to_parquet(root / "sessions.parquet", index=False)
    pd.DataFrame(manifest).to_parquet(run / "manifest.parquet", index=False)
    index = frames[["video_id", "frame_idx"]].copy()
    index["row"] = np.arange(12)
    index.to_parquet(cache / "index.parquet", index=False)
    values = np.random.default_rng(0).normal(size=(12, 17, 8)).astype(np.float16)
    np.save(cache / "emb.f16.npy", values)
    _write_body(root, frames.video_id)
    physical = _write_physical(root, frames.video_id)
    protocol = run / "protocol.json"
    protocol.write_text(json.dumps({"episodes": str(root), "cache": str(cache), "physical": str(physical)}))
    panel = [{"target": i, "candidates": [0, 1, 2, 3], "correct": i} for i in range(4)]
    (run / "panels.json").write_text(json.dumps({"dev": panel, "test": panel, "test_donors": [1, 2, 3, 0]}))
    hashes = {f"study_{key}_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
              for key, path in (("protocol", protocol), ("manifest", run / "manifest.parquet"),
                                ("panels", run / "panels.json"))}
    original_score = train._study_score
    original_loss = _video_class().losses
    original_body_loss = _model_class().losses
    traces, snapshots, calls, dropout_rng = {}, {}, [], {}
    condition = None

    def loss(model, v, *args, **kwargs):
        traces[condition].append(v[:, 0, :].detach().clone())
        dropout_rng.setdefault(condition, []).append(torch.get_rng_state().clone())
        if hasattr(model, "interaction_enabled"):
            # Every fixture window has one real frame followed by four synthetic repeats.
            mask = args[3]["visual_valid"]
            assert mask.shape == v.shape[:2] and mask[:, :17].all() and not mask[:, 17:].any()
        if len(traces[condition]) == 1:
            snapshots[condition] = {k: x.detach().clone() for k, x in model.state_dict().items() if not k.startswith("encoder.")}
        return (original_body_loss if model.encoder is not None else original_loss)(model, v, *args, **kwargs)

    def score(model, module, data, panel, device, trials=False):
        calls.append((condition, "test" if trials else "dev"))
        result = original_score(model, module, data, panel, device, trials)
        if not trials:
            epoch = sum(c == condition and kind == "dev" for c, kind in calls)
            for key in ("picture_soft", "fused_soft"):
                if key in result:
                    result[key]["acc"] = 50 if epoch == 1 else 75
            if epoch == 2:
                snapshots[condition + "_selected"] = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            for k, value in snapshots[condition + "_selected"].items():
                torch.testing.assert_close(model.state_dict()[k], value)
        return result

    monkeypatch.setattr(_video_class(), "losses", loss)
    monkeypatch.setattr(_model_class(), "losses", loss)
    monkeypatch.setattr(train, "_study_score", score)
    for condition, body, arch in (("video", "none", "additive"), ("additive", "real", "additive"),
                                  ("interaction", "real", "interaction")):
        traces[condition] = []
        result = train.train_run(run, condition, 0, body=body, arch=arch, device="cpu", study_protocol=protocol)
        assert result["device"] == "cpu" and result["arch"] == arch
        if arch == "interaction":
            assert "additive_soft" in result["test"]
        assert result["selected_epoch"] == 1 and len(result["history"]) == 20
        assert {key: result[key] for key in hashes} == hashes
        assert all("acc" not in epoch for epoch in result["history"])
        assert result["settings"]["batch"] == 32 and result["emb_dim"] == 512
        assert "forbidden" not in json.loads((run / f"{condition}_s0" / "vocab.json").read_text())
        np.testing.assert_allclose(torch.load(run / f"{condition}_s0" / "model.pt", weights_only=True)["center_mu"], values[:4].astype(np.float32).mean((0, 1)), atol=1e-6)
        assert (result["shuffled_test"] is not None) == (body == "real")
        if body == "real":
            from test_learner import _body_session
            from nursery.learner.body import session_features
            expected = np.stack([session_features(_body_session(k))["fingers"][0] for k in range(4)]).mean(0)
            np.testing.assert_allclose(result["body_stats"]["fingers"][0], expected, atol=1e-6)
    for name in ("additive", "interaction"):
        for a, b in zip(traces["video"], traces[name]):
            torch.testing.assert_close(a, b)
        for a, b in zip(dropout_rng["video"], dropout_rng[name]):
            assert torch.equal(a, b)
        for key in ("word.weight", "vproj.2.weight", "vproj.2.bias"):
            torch.testing.assert_close(snapshots["video"][key], snapshots[name][key])
    assert [kind for c, kind in calls if c == "video"] == ["dev"] * 20 + ["test"]
    assert [kind for c, kind in calls if c == "additive"] == ["dev"] * 20 + ["test", "test"]
    assert [kind for c, kind in calls if c == "interaction"] == ["dev"] * 20 + ["test", "test"]
    condition = "timing"
    traces[condition] = []
    timing = train.train_run(run, condition, 0, study_protocol=protocol, study_timing=True)
    assert timing["epochs"] == 1 and timing["weights_discarded"] and not timing["test_evaluated"]
    assert {key: timing[key] for key in hashes} == hashes
    assert timing["train_dev_seconds"] > 0
    assert [p.name for p in (run / "timing" / condition).iterdir()] == ["timing.json"]
    assert [kind for c, kind in calls if c == "timing"] == ["dev"]


def test_actual_situation_evaluator_uses_complete_interaction_score(upstream):
    import torch
    from nursery.learner.interaction import JointInteractionMIL
    from nursery.learner.evaluate import body_present_4afc

    torch.manual_seed(31)
    model = JointInteractionMIL(6, 8, dims={k: 3 for k in ("contact", "fingers", "arm", "load", "skin")}).eval()
    visual = torch.randn(4, 85, 8)
    tokens = torch.arange(1, 5)[:, None]
    ctx = {"groups": {k: torch.randn(4, 7, 3) for k in model.encoder.names},
           "missing": torch.zeros(4, 7), "seg": torch.zeros(4, 7, dtype=torch.long), "keep": torch.ones(4)}

    class Body:
        def batch(self, idx, generator, training=False):
            return {k: {g: value[idx] for g, value in v.items()} if k == "groups" else v[idx] for k, v in ctx.items()}

    class Dataset:
        locs = [[None] * count for count in (1, 2, 3, 4)]

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return i, visual[i], tokens[i], 1

    rows = pd.DataFrame({"video_id": list("abcd"), "frame_idx": [0]*4, "referent": list("wxyz")})
    context = {"identity": list(zip(rows.video_id, rows.frame_idx, rows.referent)), "subjects": ["s1"]*4,
               "pools": [np.array([j for j in range(4) if i != j]) for i in range(4)], "held": [True]*4,
               "definition": {"n_insufficient_foils": 0}}
    panel = [{"target": i, "candidates": [0, 1, 2, 3], "correct": i} for i in range(4)]
    initial = body_present_4afc(model, upstream, Dataset(), Body(), rows, "cpu", context=context, fixed_trials=panel, trials=True)
    assert initial["trials"]["fused_soft"] == initial["trials"]["additive_soft"]
    with torch.no_grad():
        torch.manual_seed(100)
        model.interaction_out.weight.normal_(std=10)
    result = body_present_4afc(model, upstream, Dataset(), Body(), rows, "cpu", context=context, fixed_trials=panel, trials=True)
    with torch.no_grad():
        R = model.enc_regions(visual).half().float()
        T = model.enc_text(tokens, torch.ones(4))
        visual_valid = torch.arange(85)[None, :] < torch.tensor([1, 2, 3, 4])[:, None] * 17
        complete = model.score_encoded(R, T, dict(ctx, visual_valid=visual_valid))
        predictions = complete.argmax(0)
        expected = [int(predictions[i] == i) for i in range(4)]
    assert result["trials"]["fused_soft"] == expected
    assert result["trials"]["fused_soft"] != result["trials"]["additive_soft"]
    # Shuffling moves body histories, while each destination retains its own visual validity.
    rows["subject"], rows["clip"] = "s1", list("abcd")
    donors = [1, 2, 3, 0]
    shuffled = ShuffledBodyHistory(Body(), donors, rows)
    result = body_present_4afc(model, upstream, Dataset(), shuffled, rows, "cpu", context=context, fixed_trials=panel, trials=True)
    with torch.no_grad():
        donor_ctx = shuffled.batch(np.arange(4), None)
        complete = model.score_encoded(R, T, dict(donor_ctx, visual_valid=visual_valid))
        expected = [int(complete[:, i].argmax() == i) for i in range(4)]
    assert result["trials"]["fused_soft"] == expected
