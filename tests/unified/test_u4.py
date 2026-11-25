from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.learner import _upstream_module
from nursery.unified import evaluate
from nursery.unified.baselines import BabyLLaVANursery, CLIPGlobal
from nursery.unified.model import UnifiedEmbodiedLearner
from nursery.unified.u4_eval import (
    CATEGORIES, FAMILIAR, UNSEEN, build_blimp, greedy_decode, lm_logits,
    perplexity_batch, sensory_prefix,
)
from nursery.unified.u2_data import FROZEN


def test_blimp_is_deterministic_in_vocab_and_disjoint_from_u2(upstream):
    manifest = FROZEN / "manifest.parquet"
    if not manifest.is_file():
        pytest.skip("Frozen U2 speech is required")
    pairs, vocab = build_blimp()
    assert (pairs, vocab) == build_blimp()
    tokenize = _upstream_module("train").tokenize
    utterances = {tuple(tokenize(s)) for s in pd.read_parquet(manifest).text}
    test = {tuple(tokenize(p[side])) for p in pairs for side in ("good", "bad")}
    lengths = {len(seq) for seq in test}
    train = {u[i:i + length] for u in utterances for length in lengths
             for i in range(len(u) - length + 1)}
    assert {p["category"] for p in pairs} == set(CATEGORIES)
    assert {p["tier"] for p in pairs} == {FAMILIAR, UNSEEN}
    assert not any("these" in seq for seq in test)  # Absent in the frozen speech.
    assert any(p["good"].startswith("those ") for p in pairs)
    assert {p["good"].split()[0] for p in pairs if p["category"] == "verb_argument"} == {
        "drink", "look", "lift", "pass",
    }
    assert train.isdisjoint(test)
    assert all(word in vocab for seq in test for word in seq)
    assert all(len(tokenize(p["good"])) == len(tokenize(p["bad"])) for p in pairs)


@pytest.mark.parametrize("cls", [UnifiedEmbodiedLearner, CLIPGlobal, BabyLLaVANursery])
@torch.no_grad()
def test_greedy_decode_is_deterministic_and_matches_training_prefix(cls):
    torch.manual_seed(3)
    model = cls(8, input_dim=8).eval()
    video = torch.randn(2, 2, 257, 8)
    valid = torch.tensor([[True, True], [True, False]])
    sensory = model.encode_sensory(video, valid)
    prefix = sensory_prefix(model, sensory)
    words, counts = torch.tensor([[1, 2], [3, 0]]), torch.tensor([2, 1])
    inputs = model.language._tokens(words, counts)[:, :-1]
    embedded = cls is BabyLLaVANursery
    logits = lm_logits(model.language, inputs, prefix, embedded=embedded)
    expected_loss = model.language.lm_loss(words, counts, *prefix) if not embedded else (
        model(words, counts, video, valid)["loss"])
    vocab = {"one": 1, "two": 2, "three": 3}
    nll, n_tokens = perplexity_batch(model.language, ["one two", "three"], vocab, "cpu",
                                     prefix, embedded=embedded)
    torch.testing.assert_close(torch.tensor(nll / n_tokens), expected_loss)
    if embedded:
        assert prefix[1].sum(1).tolist() == [18, 17]
        torch.testing.assert_close(logits, model(words, counts, video, valid)["logits"])
    elif cls is CLIPGlobal:
        assert prefix[0].shape == (2, 1, 512)
        assert prefix[1].all()
    else:
        assert prefix[0].shape == (2, 3, 512)
        assert prefix[1].tolist() == [[True, False, False]] * 2

    first = logits[:, 0].clone()
    first[:, [0, model.language.bos]] = -torch.inf
    expected = [[] if word == model.language.eos else [word] for word in first.argmax(-1).tolist()]
    assert greedy_decode(model, sensory, max_len=1) == expected
    assert greedy_decode(model, sensory) == greedy_decode(model, sensory)
    model.language.lm_head.weight.zero_()
    model.language.lm_head.bias[model.language.eos] = 100
    assert greedy_decode(model, sensory) == [[], []]


@torch.no_grad()
def test_compositional_masks_body_and_both_gaze_paths(tmp_path, monkeypatch, upstream):
    torch.manual_seed(7)
    groups = dict.fromkeys(("contact", "fingers", "arm", "load", "skin"), 1)
    model = UnifiedEmbodiedLearner(5, condition="V+G+B", body_dims=groups).eval()
    study = evaluate.UnifiedEvaluation.__new__(evaluate.UnifiedEvaluation)
    study.device, study.condition, study.body_mode = "cpu", "V+G+B", "real"
    study.upstream, study.vocab, study.store = upstream, {
        "<pad>": 0, "lift": 1, "the": 2, "mug": 3, "ball": 4,
    }, {}
    study.protocol = {"gaze": {"dropout": 0.0}}
    study.lut = {"v|0": (0, 0), "v|1": (0, 1)}
    study.partitions = {"v|0": 0, "v|1": 1}
    study.caches = [np.random.default_rng(7).normal(size=(2, 257, 768)).astype("float32")]
    rows = pd.DataFrame(dict(video_id=["v", "v"], frame_idx=[0, 1], clip=["a", "b"],
                             split=["test", "s8"], straddling=True,
                             choices=[["lift the mug", "lift the ball"]] * 2))
    for panel in evaluate.U2_PANELS:
        rows.to_parquet(tmp_path / f"{panel}.parquet", index=False)

    signal = {"body": 1., "gaze": 1.}
    gaze_modes = []

    def gaze_batch(ids, *args, **kwargs):
        return dict(keep=torch.ones(len(ids), dtype=torch.bool),
                    gaze_valid=torch.ones(len(ids), 1, dtype=torch.bool),
                    gaze_point=torch.full((len(ids), 1, 2), .2 * signal["gaze"]),
                    closeup=torch.ones(len(ids), 1, 768) * signal["gaze"])

    def temporal_batch(model, ids, *, device, gaze_mode, keep):
        gaze_modes.append(gaze_mode)
        same = torch.ones(len(ids), 11, dtype=torch.bool)
        return dict(same=same, frames=torch.full((len(ids), 11, 768),
                                                signal["gaze"] if gaze_mode == "real" else 0.))

    def body_batch(ids, *args, **kwargs):
        return dict(groups={key: torch.full((len(ids), 11, 1), signal["body"]) for key in groups},
                    keep=torch.ones(len(ids)), missing=torch.zeros(len(ids), 11, dtype=torch.bool),
                    seg=torch.zeros(len(ids), 11, dtype=torch.long))

    study.gaze_data = lambda *args, **kwargs: SimpleNamespace(batch=gaze_batch)
    study.temporal_data = lambda *args: SimpleNamespace(batch=temporal_batch)
    monkeypatch.setattr(evaluate, "_pair_meta", lambda *args: (None, None, None))
    monkeypatch.setattr(evaluate, "BodyData", lambda *args: SimpleNamespace(batch=body_batch))

    def run(**kwargs):
        gaze_modes.clear()
        result, routing = study.compositional(model, tmp_path, **kwargs)
        assert set(result) == set(evaluate.U2_PANELS)
        mode = "removed" if kwargs.get("gaze_mode") == "removed" else "real"
        assert gaze_modes == [mode] * len(evaluate.U2_PANELS)
        if kwargs.get("body_mode") == "none":
            assert all(r["p_body"] == 0 for r in routing)
        else:
            assert any(r["p_body"] > 0 for r in routing)
        return result, routing

    as_trained = run()
    assert run(body_mode="real", gaze_mode=None) == as_trained
    body_removed = run(body_mode="none")
    extras_removed = run(body_mode="none", gaze_mode="removed")
    signal["body"] = -4.
    assert run() != as_trained
    assert run(body_mode="none") == body_removed
    signal["gaze"] = 4.
    assert run(body_mode="none") != body_removed
    assert run(body_mode="none", gaze_mode="removed") == extras_removed
