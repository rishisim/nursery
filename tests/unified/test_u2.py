import numpy as np
import pandas as pd
import pytest
import torch

from nursery.unified.evaluate import (
    UnifiedEvaluation, choice_credits, evaluate_vp_swap, load_checkpoint, score_sentence_choices,
    sentence_inputs,
)
from nursery.unified.language import CausalLanguage
from nursery.unified.model import UnifiedEmbodiedLearner
from nursery.unified.u2_data import TEST_NEGATION, vp_swap


def test_negation_bow_gets_exactly_chance():
    sentences = [
        TEST_NEGATION.format(noun="cup", other="bowl"),
        TEST_NEGATION.format(noun="bowl", other="cup"),
    ]
    vocab = {"holding": 1, "the": 2, "cup": 3, "not": 4, "bowl": 5}
    words, counts, known = sentence_inputs(sentences, vocab, "cpu")
    model = UnifiedEmbodiedLearner(6, input_dim=8, grid=2, bow_only=True).eval()
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in UnifiedEmbodiedLearner(6, input_dim=8, grid=2).parameters())
    with torch.no_grad():
        model.language.ctx_proj.weight.fill_(1)
        sensory = model.encode_sensory(torch.randn(1, 3, 5, 8), torch.ones(1, 3, dtype=torch.bool))
        scores, _ = score_sentence_choices(model, sensory, words, counts, 2)
    assert scores[0, 0].item() == scores[0, 1].item()
    assert choice_credits(scores.numpy(), [known.all()]).tolist() == [.5]


def test_u1_single_word_byte_parity_with_legacy_checkpoint(tmp_path):
    from nursery.learner import _upstream_module

    study = UnifiedEvaluation.__new__(UnifiedEvaluation)
    study.vocab = {"<pad>": 0, "lift": 1, "cup": 2}
    study.device = "cpu"
    study.upstream = _upstream_module("train_frame_mil")
    legacy = UnifiedEmbodiedLearner(3, input_dim=8, grid=2, max_words=16).eval()
    path = tmp_path / "u1.pt"
    with torch.no_grad():
        legacy.language.ctx_proj.weight.normal_()
        sensory = legacy.encode_sensory(torch.randn(1, 2, 5, 8), torch.ones(1, 2, dtype=torch.bool))
        expected = legacy.score(sensory, legacy.enc_text(torch.tensor([[2], [1]]), torch.tensor([1, 1])))
        torch.save(dict(state_dict=legacy.state_dict(), model_args=dict(vocab_size=3, input_dim=8, grid=2)),
                   path)
        model, _ = load_checkpoint(path, "cpu")
        mapping, vectors, _ = study._query_vectors(model, ["lift", "absent", "cup", "lift"])
        scores = model.score(sensory, vectors)
    assert mapping == {"cup": 0, "lift": 1}
    assert scores.numpy().tobytes() == expected.numpy().tobytes()


def test_bow_word_queries_use_lexical_vectors():
    model = UnifiedEmbodiedLearner(3, input_dim=8, grid=2, bow_only=True).eval()
    words, counts = torch.tensor([[1], [2]]), torch.tensor([1, 1])
    with torch.no_grad():
        model.language.ctx_proj.weight.normal_()
        vectors, content = model.language.encode_tokens(words, counts, bow_only=True)
        torch.testing.assert_close(model.enc_text(words, counts), vectors[content])


def test_unconditional_logprob_matches_token_loss_and_prefix_dropout():
    language = CausalLanguage(6).eval()
    words = torch.tensor([[1, 2, 3], [4, 0, 0]])
    counts = torch.tensor([3, 1])
    with torch.no_grad():
        logp = language.sentence_logprob(words, counts)
        loss = language.lm_loss(words, counts)
        torch.testing.assert_close(loss, -logp.sum() / (counts + 1).sum())

    language.train()
    torch.manual_seed(42)
    sensory = torch.randn(2, 3, 512, requires_grad=True)
    active = torch.ones(2, 3, dtype=torch.bool)
    loss = language.lm_loss(words, counts, sensory, active, p_uncond=1.0)
    loss.backward()
    assert sensory.grad is None
    assert torch.isfinite(language.word.weight.grad).all()
    assert language.position.num_embeddings == 34


def test_vp_swap_split_is_disjoint_and_cartesian():
    quads, noun_splits = vp_swap(np.random.default_rng(0))
    seen, held = noun_splits["seen"], noun_splits["held_out"]
    assert quads.groupby("split").size().to_dict() == {"seen": 64, "held_out": 16}
    assert set(sum(seen.values(), [])).isdisjoint(sum(held.values(), []))
    assert not quads.duplicated(["heavy_noun", "light_noun"]).any()
    assert all(quads.true_heavy.str.endswith(" heavy")) and all(quads.swap_heavy.str.endswith(" light"))


@pytest.mark.parametrize(
    "split, noun_differences, expected_quad, expected_margin",
    [
        ("held_out", (2, 2), 100.0, 4.0),
        ("seen", (3, -2), 0.0, 1.0),
        ("seen", (1, -2), 0.0, -1.0),
        ("held_out", (0, 0), 0.0, 0.0),
    ],
)
def test_vp_swap_uses_within_noun_differences(
    tmp_path, split, noun_differences, expected_quad, expected_margin,
):
    pd.DataFrame([dict(split=split, heavy_noun="pan", light_noun="pen",
                       true_heavy="the pan is heavy", true_light="the pen is light",
                       swap_heavy="the pan is light", swap_light="the pen is heavy")]).to_parquet(
        tmp_path / "vp_swap.parquet", index=False)
    vocab = {"the": 1, "pan": 2, "pen": 3, "is": 4, "heavy": 5, "light": 6}

    class FixedLanguage:
        def __init__(self, heavy_bias):
            self.heavy_bias = heavy_bias

        def eval(self):
            return self

        def sentence_logprob(self, words, counts):
            heavy_difference, light_difference = noun_differences
            scores = {(2, 5): -10 + heavy_difference, (2, 6): -10,
                      (3, 6): -20 + light_difference, (3, 5): -20}
            return torch.tensor([
                scores.get((noun, adjective), 0) + (self.heavy_bias if adjective == vocab["heavy"] else 0)
                for _, noun, _, adjective in words.tolist()
            ])

    # A pure adjective bias creates opposite preferences and a zero margin.
    for heavy_bias in (0, 1):
        result = evaluate_vp_swap(FixedLanguage(heavy_bias), vocab, tmp_path, "cpu")[split]
        assert "pair_accuracy" not in result
        assert result["quad_accuracy"] == expected_quad
        assert result["swap_margin_accuracy"] == (100.0 if expected_margin > 0 else 0.0)
        assert result["trials"][0]["swap_margin"] == expected_margin

    missing_noun = {word: token for word, token in vocab.items() if word != "pen"}
    result = evaluate_vp_swap(FixedLanguage(0), missing_noun, tmp_path, "cpu")[split]
    assert result["quad_accuracy"] == 25.0
    assert result["swap_margin_accuracy"] == 50.0
    assert result["quad_coverage"] == 0.0
    assert result["trials"][0]["swap_margin"] is None
