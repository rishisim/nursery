import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.unified import evaluate, train
from nursery.unified.baselines import BASELINE_CONDITIONS, BabyLLaVANursery, CLIPGlobal
from nursery.unified.model import UnifiedEmbodiedLearner


def examples():
    words = torch.tensor([[1, 2], [3, 0], [2, 4]])
    counts = words.ne(0).sum(1)
    video = torch.randn(3, 2, 257, 8)
    valid = torch.tensor([[True, False], [True, True], [False, True]])
    return words, counts, video, valid


def test_language_capacity():
    models = [cls(7, input_dim=8) for cls in (UnifiedEmbodiedLearner, CLIPGlobal, BabyLLaVANursery)]
    reference = models[0].language
    for model in models[1:]:
        assert repr(model.language) == repr(reference)
        assert {k: p.shape for k, p in model.language.named_parameters()} == {
            k: p.shape for k, p in reference.named_parameters()}


def test_clip_text_is_unconditional_and_trainable_gradients_finite():
    model = CLIPGlobal(7, input_dim=8).eval()
    words, counts, video, valid = examples()
    original = model(words, counts, video, valid)
    altered = model(words, counts, -video, valid)
    assert torch.equal(original["text"], altered["text"])
    assert not torch.equal(original["logits"], altered["logits"])
    hidden = model.language._hidden(model.language._tokens(words, counts))
    expected = model.language.project(hidden[torch.arange(len(words)), counts + 1])
    torch.testing.assert_close(original["text"], torch.nn.functional.normalize(expected))
    # Eval disables stochastic unconditional dropout so the visual LM branch is exercised.
    original["loss"].backward()
    for name, p in model.named_parameters():
        if name.startswith("language.ctx_proj."):
            assert not p.requires_grad
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


@torch.no_grad()
def test_babyllava_masks_frames_and_scores_autoregressively():
    model = BabyLLaVANursery(7, input_dim=8).eval()
    words, counts, video, valid = examples()
    changed = video.clone()
    changed[~valid] = 1e4 * torch.randn_like(changed[~valid])
    original = model(words, counts, video, valid)
    assert torch.equal(original["logits"], model(words, counts, changed, valid)["logits"])
    prefix = model.encode_sensory(video, valid)
    queries = torch.tensor([1, 2, 3])
    for prompt in ([], [4], [4, 5]):
        fast = model.score_words(prefix, queries, prompt)
        for i, query in enumerate(queries.tolist()):
            sequence = torch.tensor([model.language.bos, *prompt, query]).expand(len(video), -1)
            hidden = model.language.hidden_with_prefix(sequence, *prefix)
            expected = model.language.lm_head(hidden[:, -2]).log_softmax(-1)[:, query]
            torch.testing.assert_close(fast[:, i], expected, atol=1e-6, rtol=0)

    batches = words.repeat_interleave(2, 0)
    lengths = counts.repeat_interleave(2)
    scores, _ = evaluate.score_sentence_choices(model, prefix, batches, lengths, 2)
    targets = model.language._tokens(words, counts)[:, 1:]
    logp = original["logits"].log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    expected = logp.masked_fill(targets.eq(0), 0).sum(1) / (counts + 1)
    torch.testing.assert_close(scores, expected[:, None].expand_as(scores))


def test_babyllava_unconditional_dropout_has_finite_gradients_and_is_train_only():
    model = BabyLLaVANursery(7, input_dim=8, p_uncond=0.25)
    inputs = examples()
    torch.manual_seed(0)  # Two conditioned rows and one unconditioned row.
    loss = model(*inputs)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.visual[1].weight.grad is not None
    assert model.language.lm_head.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    unconditional_dropout_off = BabyLLaVANursery(7, input_dim=8, p_uncond=0).eval()
    unconditional_dropout_off.load_state_dict(model.state_dict())
    with torch.no_grad():
        with_dropout = model.eval()(*inputs)
        without_dropout = unconditional_dropout_off(*inputs)
        for key in with_dropout:
            assert torch.equal(with_dropout[key], without_dropout[key])


@torch.no_grad()
def test_babyllava_sentence_pmi_subtracts_unconditional_logprob():
    model = BabyLLaVANursery(7, input_dim=8).eval()
    words, counts, video, valid = examples()
    prefix = model.encode_sensory(video[:1], valid[:1])
    tokens = model.language._tokens(words, counts)
    expanded = tuple(t.expand(len(words), *t.shape[1:]) for t in prefix)
    hidden = model.language.hidden_with_prefix(tokens[:, :-1], *expanded)
    targets = tokens[:, 1:]
    logp = model.language.lm_head(hidden).log_softmax(-1)
    cond = logp.gather(-1, targets[..., None]).squeeze(-1).masked_fill(targets.eq(0), 0).sum(1)
    expected = cond - model.language.sentence_logprob(words, counts)
    scores = model.score_sentences(prefix, words, counts, len(words), pmi=True)
    assert torch.equal(scores, expected[None])


@torch.no_grad()
def test_babyllava_pmi_choice_margin_cancels_bias_with_shared_histories():
    torch.manual_seed(0)
    model = BabyLLaVANursery(7, input_dim=8).eval()
    _, _, video, valid = examples()
    prefix = model.encode_sensory(video[:1], valid[:1])
    words, counts = torch.tensor([[1], [2]]), torch.tensor([1, 1])
    # Tied input embeddings give both choices identical hidden histories, so
    # any change to each log-normalizer is shared across choices in that branch.
    model.language.word.weight[2].copy_(model.language.word.weight[1])
    before = model.score_sentences(prefix, words, counts, 2)[0]
    model.language.lm_head.bias[1].add_(2)
    after = model.score_sentences(prefix, words, counts, 2)[0]
    # The target bias cancels between branches. The shared log-normalizer
    # changes cancel in the margin even though the individual PMIs can move.
    assert not torch.allclose(before, after)
    torch.testing.assert_close(after[0] - after[1], before[0] - before[1], atol=1e-6, rtol=0)


@torch.no_grad()
def test_babyllava_patch_pool_matches_masked_mean_with_padded_garbage():
    model = BabyLLaVANursery(7, input_dim=8).eval()
    video = torch.randn(3, 4, 257, 8)
    valid = torch.tensor([[True, False, True, True], [False, True, False, False], [False] * 4])
    video[~valid] *= 1e4
    patches = video[:, :, 1:].masked_fill(~valid[:, :, None, None], 0).sum(1)
    patches = patches / valid.sum(1)[:, None, None].clamp_min(1)
    grid = patches.transpose(1, 2).reshape(3, 8, 16, 16)
    spatial = torch.nn.functional.avg_pool2d(grid, 4, 4).flatten(2).transpose(1, 2)
    cls = video[:, :, 0].masked_fill(~valid[..., None], 0)
    expected = model.visual(torch.cat((spatial, cls), dim=1)) + model.prefix_position[:20]
    prefix, _ = model.encode_sensory(video, valid)
    torch.testing.assert_close(prefix, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("cls", [CLIPGlobal, BabyLLaVANursery])
def test_checkpoint_bit_exact(tmp_path, cls):
    args = dict(vocab_size=7, input_dim=8, max_words=8)
    model = cls(**args).eval()
    path = tmp_path / "best.pt"
    torch.save(dict(model_args=args, run=dict(condition=model.condition), state_dict=model.state_dict()), path)
    restored, _ = evaluate.load_checkpoint(path, "cpu")
    inputs = examples()
    with torch.no_grad():
        assert torch.equal(model(*inputs)["logits"], restored(*inputs)["logits"])


@pytest.mark.parametrize("condition", BASELINE_CONDITIONS)
def test_train_eval_frozen_synthetic(tmp_path, upstream, monkeypatch, condition):
    root = tmp_path / "study"
    subset = root / "subsets/1.0"
    cache = root / "train_cache"
    subset.mkdir(parents=True)
    cache.mkdir()
    monkeypatch.setattr(evaluate, "STUDY", root)
    rows = pd.DataFrame(dict(video_id=["v"] * 4, frame_idx=range(4), row=range(4),
                             clip=list("abcd"), referent=["cup", "cup", "lift", "lift"],
                             held_category="cup", text=["the cup", "lift cup"] * 2))
    rows.to_parquet(cache / "index.parquet")
    np.save(cache / "emb.f16.npy", np.random.default_rng(2).normal(size=(4, 257, 768)).astype("float16"))
    rows.to_parquet(root / "frames.parquet")
    rows.to_parquet(root / "manifest.parquet")
    np.savez(subset / "fit_mask.npz", video_ids=["v"], include=np.ones((1, 4), dtype=bool))
    evaluate.write_json(subset / "stats.json", dict(selected_clip_count=4, selected_clips=list("abcd")))
    partition = root / "partitions.json"
    evaluate.write_json(partition, {f"v|{i}": i for i in range(4)})
    panels = dict(noun=dict(dev="clips_dev", held_out=dict(test="clips_test")),
                  action=dict(dev="action_dev", cluster="clip", held_out=dict(test="action_test", s8="action_s8")))
    evaluate.write_json(root / "protocol.json", dict(inputs=str(root), episodes=str(root), fractions=[1.0],
                        window=2, gaze=dict(partitions=str(partition)), word_type_panels=panels,
                        lr=.0003, weight_decay=.1, batch=4))
    for spec in panels.values():
        for name in (spec["dev"], *spec["held_out"].values()):
            path = root / "panels" / name
            path.mkdir(parents=True)
            rows.to_parquet(path / "manifest.parquet")
            evaluate.write_json(path / "trials.json", dict(trial_rows=list(range(4)), correct_slots=list(range(4)),
                                candidate_rows=[list(range(4))] * 4))
    sentences = rows.assign(choices=[["lift cup", "the cup"]] * 4,
                            split=["s8", "test"] * 2, straddling=True)
    for name in evaluate.U2_PANELS:
        sentences.to_parquet(root / f"{name}.parquet")
    pd.DataFrame([dict(split="seen", true_heavy="lift cup", true_light="the cup",
                       swap_heavy="the cup", swap_light="lift cup", heavy_noun="cup",
                       light_noun="cup")]).to_parquet(root / "vp_swap.parquet")
    out = tmp_path / "run"
    out.mkdir()
    monkeypatch.setattr(train, "_output_directory", lambda path, **kwargs: path)
    train.train(SimpleNamespace(condition=condition, epochs=1, seed=1, n_clips=4, device="cpu",
                                out_dir=out, u2=root, shuffle_body=False, lambda_gen=.2,
                                aux_weight=.25, tau_m=.1))
    metrics = json.loads((out / "metrics.json").read_text())
    summary = metrics["summary"]
    assert summary["reload_identical"]
    assert set(summary["compositional"]) == set(evaluate.U2_PANELS)
    assert summary["vp_swap"]["seen"]["quad_coverage"] == 1
    assert not list(out.glob("*routing*"))
    if condition == BabyLLaVANursery.condition:
        restored, checkpoint = evaluate.load_checkpoint(out / "best.pt", "cpu")
        assert checkpoint["model_args"]["p_uncond"] == restored.p_uncond == 0.25
        raw = summary["compositional_raw_ll"]
        assert raw.keys() == summary["compositional"].keys()
        for panel, raw_metrics in raw.items():
            assert raw_metrics.keys() == summary["compositional"][panel].keys()
            assert "trials" not in raw_metrics
        assert set(summary["u1_bare_diagnostic"]) == {"noun", "action"}
        # This vocabulary omits the action prompt. Prompted trials receive chance credit.
        panel = metrics["held_out"]["as_trained"]["action"]["pooled"]
        assert panel["coverage"] == 0 and panel["accuracy"] == 25
    else:
        assert "compositional_raw_ll" not in summary
