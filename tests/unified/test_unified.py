import pytest
import torch
from torch.nn import functional as F

from nursery.learner.gaze import _gaze_class
from nursery.learner.losses import soft_pool, weighted_nce
from nursery.unified import model as unified_model
from nursery.unified.model import UnifiedEmbodiedLearner


DIMS = dict(contact=2, fingers=3, arm=4, load=2, skin=1)


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(11)

    def noise(*shape):
        return torch.randn(*shape, generator=generator)

    frame_valid = torch.tensor([[True, True], [True, False]])
    return dict(
        words=torch.tensor([[1, 2, 3], [3, 2, 0]]),
        counts=torch.tensor([3, 2]),
        video=noise(2, 2, 5, 8),
        frame_valid=frame_valid,
        temporal=dict(frames=noise(2, 11, 8), same=torch.ones(2, 11, dtype=torch.bool)),
        gaze=dict(gaze_point=torch.full((2, 2, 2), 0.25), closeup=noise(2, 2, 8),
                  gaze_valid=frame_valid.clone(), keep=torch.ones(2, dtype=torch.bool)),
        body=dict(groups={name: noise(2, 11, dim) for name, dim in DIMS.items()},
                  missing=torch.zeros(2, 11), seg=torch.zeros(2, 11, dtype=torch.long),
                  keep=torch.ones(2)),
    )


def learner(condition="V+G+B", **kwargs):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        return UnifiedEmbodiedLearner(8, input_dim=8, grid=2, condition=condition,
                                      body_dims=DIMS if condition == "V+G+B" else None, **kwargs)


@torch.no_grad()
def test_masking_equivalence(inputs):
    full, gaze_only, reduced = [learner(c).eval() for c in ("V+G+B", "V+G", "V")]
    base = dict(inputs, gaze=None, body=None)
    expected = reduced(**base)

    gaze, body = inputs["gaze"], inputs["body"]
    gaze_masked = dict(gaze, keep=torch.zeros_like(gaze["keep"]),
                       gaze_point=torch.full_like(gaze["gaze_point"], torch.nan),
                       closeup=torch.full_like(gaze["closeup"], torch.nan))
    body_masked = dict(body, keep=torch.zeros_like(body["keep"]),
                       groups={key: torch.full_like(x, torch.nan) for key, x in body["groups"].items()})
    for actual in (reduced(**inputs), gaze_only(**dict(inputs, gaze=gaze_masked)),
                   full(**base), full(**dict(inputs, gaze=gaze_masked, body=body_masked))):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    expected_gaze = gaze_only(**inputs)
    actual_gaze = full(**dict(inputs, body=body_masked))
    for key in expected_gaze:
        torch.testing.assert_close(actual_gaze[key], expected_gaze[key], atol=0, rtol=0)

    args = [inputs[key] for key in ("video", "frame_valid", "temporal", "gaze", "body")]
    removed = full.score(full.encode_sensory(*args, remove_extras=True), expected["text"])
    assert torch.equal(removed, reduced.score(reduced.encode_sensory(*args), expected["text"]))


@torch.no_grad()
def test_static_score_is_pure_soft_pool(inputs):
    model = learner().eval()
    args = [inputs[key] for key in ("video", "frame_valid", "temporal", "gaze", "body")]
    text = model.enc_text(inputs["words"], inputs["counts"])
    sensory = model.encode_sensory(*args)
    _, _, regions, log_prior = sensory
    expected = soft_pool(regions @ text.T, log_prior, 0.1)
    assert torch.equal(model._scores(sensory, text)[1][..., 0], expected)


@torch.no_grad()
def test_sensory_tokens_are_masked_and_unit_normalized(inputs):
    model = learner().eval()
    inputs["temporal"]["same"][1] = False
    inputs["body"]["keep"][0] = 0
    args = [inputs[key] for key in ("video", "frame_valid", "temporal", "gaze", "body")]
    tokens, active, *_ = model.encode_sensory(*args)
    assert active.tolist() == [[True, True, False], [True, False, True]]
    torch.testing.assert_close(tokens.norm(dim=-1), active.to(tokens.dtype))
    assert tokens[~active].count_nonzero() == 0


@torch.no_grad()
def test_infonce_has_no_sensory_prefix(inputs):
    model = learner().eval()
    model.language.ctx_proj.weight.copy_(torch.eye(256))
    original = model(**inputs)
    model.language.prefix_position.normal_()
    changed_prefix = model(**inputs)
    assert torch.equal(original["logits"], changed_prefix["logits"])
    assert not torch.equal(original["loss"], changed_prefix["loss"])
    altered = model(**dict(inputs, video=-inputs["video"], gaze=None, body=None))
    torch.testing.assert_close(original["text"], altered["text"], rtol=0, atol=0)
    torch.testing.assert_close(original["text"], model.language.encode_tokens(
        inputs["words"], inputs["counts"]), rtol=0, atol=0)
    assert not torch.equal(original["logits"], altered["logits"])


@torch.no_grad()
def test_text_encoding_starts_at_lexical_mean():
    language = learner().eval().language
    words, counts = torch.tensor([[2, 0, 0], [3, 4, 0]]), torch.tensor([1, 2])
    lexical_mean = torch.stack((language.word.weight[2], language.word.weight[[3, 4]].mean(0)))
    assert language.ctx_proj.weight.count_nonzero() == 0
    assert language.ctx_proj.bias.count_nonzero() == 0
    expected = F.normalize(language.project(lexical_mean), dim=-1)
    torch.testing.assert_close(language.encode(words, counts), expected, atol=0, rtol=0)

    language.ctx_proj.weight.copy_(torch.eye(256))
    hidden = language._hidden(language._tokens(words, counts))
    ctx_mean = torch.stack((hidden[0, 1], hidden[1, 1:3].mean(0)))
    expected = F.normalize(language.project(lexical_mean + ctx_mean), dim=-1)
    torch.testing.assert_close(language.encode(words, counts), expected)
    torch.testing.assert_close(language.encode(words[:, :2], counts), expected)


@torch.no_grad()
def test_temporal_frames_match_phase_one_gaze_pooling(inputs):
    phase_one = _gaze_class()(8, 8, option="steering", num_regions=5, temporal_video=True)
    video = inputs["video"]
    ctx = dict(inputs["gaze"], same=inputs["frame_valid"].clone())
    ctx["keep"][1] = False
    ctx["gaze_point"][0, 1] = torch.tensor([2., 2.])
    expected = phase_one.temporal_frames(video, ctx)
    assert not torch.equal(expected[0], video.mean(dim=-2)[0])
    for condition in ("V+G", "V+G+B"):
        model = learner(condition)
        torch.testing.assert_close(model.temporal_frames(video, ctx), expected)
        assert torch.equal(model.temporal_frames(video, {"same": ctx["same"]}), video.mean(dim=-2))
    assert torch.equal(learner("V").temporal_frames(video, ctx), video.mean(dim=-2))


@torch.no_grad()
def test_temporal_stream_stays_active_in_training(inputs, monkeypatch):
    model = learner()
    args = [inputs[key] for key in ("video", "frame_valid", "temporal", "gaze", "body")]
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.tensor([0.2, 0.8], device=device))
    tokens, active, *_ = model.train().encode_sensory(*args)
    assert active[:, 1].all()
    torch.testing.assert_close(tokens[:, 1].norm(dim=-1), torch.ones(2))


def test_infonce_gradients_reach_all_active_streams(inputs):
    model = learner(lambda_gen=0.0).eval()
    model(**inputs)["loss"].backward()
    assert model.language.ctx_proj.weight.grad.count_nonzero() > 0
    for projection in (model.static.project[-1], model.temporal.project, model.body.project):
        grad = projection.weight.grad
        assert torch.isfinite(grad).all() and grad.count_nonzero() > 0
    for encoder in (model.temporal.encoder, *model.body.encoder.mlps.values(),
                    model.body.encoder.fwd, model.body.encoder.bwd):
        assert any(p.grad is not None and p.grad.count_nonzero() > 0 for p in encoder.parameters())


def test_uniform_active_prior_and_routing_simplex():
    model = learner()
    assert model.tau_m == model.static.tau == 0.1
    text = F.one_hot(torch.tensor([0, 1]), num_classes=512).float()
    # The first query sees equal active scores for every clip. The second
    # query sees different scores and must route to them in proportion.
    vectors = torch.tensor([[0.4, 0.1], [0.4, 0.3], [0.4, -0.2]])
    vectors = F.pad(vectors, (0, 510))
    tokens = vectors.expand(3, -1, -1)
    active = torch.arange(3)[None, :] <= torch.arange(3)[:, None]
    sensory = tokens, active, tokens[:, :1], torch.zeros(3, 1)
    scores, streams = model._scores(sensory, text)
    torch.testing.assert_close(scores[:, 0], torch.full((3,), 0.4), atol=1e-6, rtol=0)

    posterior = model.posterior_routing(sensory, text)
    torch.testing.assert_close(posterior.sum(-1), torch.ones(3, 2))
    assert (posterior >= 0).all()
    assert posterior.masked_select(~active[:, None, :]).count_nonzero() == 0
    torch.testing.assert_close(posterior[:, 0], active.float() / active.sum(1, keepdim=True))
    expected = (streams[:, 1] / model.tau_m).masked_fill(~active, -torch.inf).softmax(-1)
    torch.testing.assert_close(posterior[:, 1], expected)


@torch.no_grad()
def test_single_word_2d_queries_equal_content_token_queries(inputs):
    model = learner().eval()
    model.language.ctx_proj.weight.copy_(torch.eye(256))
    words, counts = torch.tensor([[1, 0], [2, 0]]), torch.ones(2, dtype=torch.long)
    text = model.enc_text(words, counts)
    tokens, content = model.language.encode_tokens(words, counts)
    assert content.tolist() == [[False, True, False, False]] * 2
    torch.testing.assert_close(tokens[content], text)
    sensory = model.encode_sensory(*(inputs[key] for key in
                                     ("video", "frame_valid", "temporal", "gaze", "body")))
    actual = model._scores(sensory, (tokens, content))
    torch.testing.assert_close(actual, model._scores(sensory, text))


@torch.no_grad()
def test_contextual_tokens_are_pooled_before_averaging(inputs):
    model = learner().eval()
    model.language.ctx_proj.weight.copy_(torch.eye(256))
    words, counts = inputs["words"], inputs["counts"]
    language = model.language
    tokens, content = language.encode_tokens(words, counts)
    assert content.tolist() == [[False, True, True, True, False], [False, True, True, False, False]]
    ids = language._tokens(words, counts)
    expected = F.normalize(language.project(language.word(ids) + language._hidden(ids)), dim=-1)
    torch.testing.assert_close(tokens, expected)
    sensory = model.encode_sensory(*(inputs[key] for key in
                                     ("video", "frame_valid", "temporal", "gaze", "body")))
    tokens = tokens.masked_fill(~content[..., None], torch.nan)
    scores, streams = model._scores(sensory, (tokens, content))
    for q in range(len(words)):
        token_scores, token_streams = model._scores(sensory, tokens[q, content[q]])
        torch.testing.assert_close(scores[:, q], token_scores.mean(1))
        torch.testing.assert_close(streams[:, q], token_streams.mean(1))


@pytest.mark.parametrize("body_rows", [[True, False, True], [False, False, False]])
@torch.no_grad()
def test_aux_nce_uses_only_active_pairs(body_rows, monkeypatch):
    model = learner(lambda_gen=0.0)
    model.logit_scale.fill_(10)  # Exercise the same clamp for all logits.
    active = torch.tensor([[True, True, body] for body in body_rows])
    streams = torch.randn(3, 3, 3, generator=torch.Generator().manual_seed(3))
    scores = torch.randn(3, 3, generator=torch.Generator().manual_seed(4))
    monkeypatch.setattr(model, "encode_sensory", lambda *args: (None, active, None, None))
    monkeypatch.setattr(model, "_scores", lambda *args: (scores, streams))

    calls = []

    def record_nce(logits, weights):
        calls.append(logits)
        return weighted_nce(logits, weights)

    monkeypatch.setattr(unified_model, "weighted_nce", record_nce)
    words = torch.tensor([[1], [2], [3]])
    output = model(words, torch.ones(3, dtype=torch.long), None, None)
    main_logits = 100 * scores
    expected_aux = [100 * streams[:, :, stream] for stream in (0, 1)]
    if any(body_rows):
        expected_aux.append(100 * streams[:, :, 2][[0, 2]][:, [0, 2]])
    assert len(calls) == 1 + len(expected_aux)
    for actual, expected in zip(calls, [main_logits, *expected_aux]):
        torch.testing.assert_close(actual, expected)
    base = weighted_nce(main_logits, torch.ones(3))
    aux = torch.stack([weighted_nce(term, torch.ones(len(term))) for term in expected_aux]).mean()
    torch.testing.assert_close(output["loss"], base + 0.25 * aux)
    assert torch.isfinite(output["loss"])
    torch.testing.assert_close(output["logits"], main_logits)

    calls.clear()
    evaluation = model.eval()(words, torch.ones(3, dtype=torch.long), None, None)
    assert len(calls) == 1
    torch.testing.assert_close(evaluation["loss"], base)


@torch.no_grad()
def test_zero_aux_weight_preserves_base_loss(inputs):
    model = learner(aux_weight=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        actual = model(**inputs)
        torch.manual_seed(7)
        words, counts = inputs["words"], inputs["counts"]
        text = model.language.encode_tokens(words, counts)
        sensory = model.encode_sensory(*(inputs[key] for key in
                                         ("video", "frame_valid", "temporal", "gaze", "body")))
        logits = model.logit_scale.exp() * model.score(sensory, text)
        tokens, active, *_ = sensory
        expected = weighted_nce(logits, logits.new_ones(len(logits)))
        expected += model.lambda_gen * model.language.lm_loss(words, counts, tokens, active)
    torch.testing.assert_close(actual["loss"], expected, atol=0, rtol=0)
