import pytest
import torch
from torch.nn import functional as F

from nursery.learner.gaze import _gaze_class
from nursery.learner.losses import weighted_nce
from nursery.unified import model as unified_model
from nursery.unified.gate import ModalityGate
from nursery.unified.model import UnifiedEmbodiedLearner
from nursery.unified.sensory import SensoryFusion


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
    # Check masking after attention has started mixing modalities.
    for model in (full, gaze_only, reduced):
        model.fusion.out_proj.weight.copy_(torch.eye(512))
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
            assert torch.equal(actual[key], expected[key]), key

    expected_gaze = gaze_only(**inputs)
    actual_gaze = full(**dict(inputs, body=body_masked))
    for key in expected_gaze:
        assert torch.equal(actual_gaze[key], expected_gaze[key]), key

    args = [inputs[key] for key in ("video", "frame_valid", "temporal", "gaze", "body")]
    removed = full.score(full.encode_sensory(*args, remove_extras=True), expected["text"])
    assert torch.equal(removed, reduced.score(reduced.encode_sensory(*args), expected["text"]))


def test_fusion_starts_as_exact_identity():
    fusion = SensoryFusion()
    tokens = torch.randn(2, 3, 512)
    active = torch.tensor([[True, True, True], [True, False, True]])
    expected = tokens.masked_fill(~active[..., None], 0)
    assert torch.equal(fusion(tokens, active), expected)

    with torch.no_grad():
        fusion.out_proj.weight.copy_(torch.eye(512))
    garbage = tokens.masked_fill(~active[..., None], torch.nan)
    assert torch.equal(fusion(garbage, active), fusion(tokens, active))


@pytest.mark.parametrize("mhsa", [True, False])
@torch.no_grad()
def test_static_fusion_residual_starts_at_exact_zero(inputs, mhsa):
    model = learner(mhsa=mhsa).eval()
    args = [inputs[key] for key in ("video", "frame_valid", "temporal", "gaze", "body")]
    *_, residual = model.encode_sensory(*args)
    assert torch.equal(residual, torch.zeros_like(residual))


@torch.no_grad()
def test_fusion_receives_unit_tokens_only_for_active_rows(inputs):
    model = learner().eval()
    inputs["temporal"]["same"][1] = False
    inputs["body"]["keep"][0] = 0
    args = [inputs[key] for key in ("video", "frame_valid", "temporal", "gaze", "body")]
    fusion_inputs = []
    with model.fusion.register_forward_pre_hook(lambda _fusion, args: fusion_inputs.append(args)):
        model.encode_sensory(*args)
    (tokens, active), = fusion_inputs
    assert active.tolist() == [[True, True, False], [True, False, True]]
    torch.testing.assert_close(tokens.norm(dim=-1), active.to(tokens.dtype))
    assert tokens[~active].count_nonzero() == 0


@torch.no_grad()
def test_infonce_has_no_sensory_prefix(inputs):
    model = learner().eval()
    original = model(**inputs)
    altered = model(**dict(inputs, video=-inputs["video"], gaze=None, body=None))
    assert torch.equal(original["text"], altered["text"])
    assert torch.equal(original["text"], model.enc_text(inputs["words"], inputs["counts"]))
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
    fused, active, *_ = model.train().encode_sensory(*args)
    assert active[:, 1].all()
    torch.testing.assert_close(fused[:, 1].norm(dim=-1), torch.ones(2))


def test_infonce_gradients_reach_all_active_streams(inputs):
    model = learner(lambda_gen=0.0).eval()
    model(**inputs)["loss"].backward()
    for projection in (model.static.project[-1], model.temporal.project, model.body.project):
        grad = projection.weight.grad
        assert torch.isfinite(grad).all() and grad.count_nonzero() > 0
    for encoder in (model.temporal.encoder, *model.body.encoder.mlps.values(),
                    model.body.encoder.fwd, model.body.encoder.bwd):
        assert any(p.grad is not None and p.grad.count_nonzero() > 0 for p in encoder.parameters())


def test_gate_simplex_and_initial_bias():
    gate = ModalityGate()
    assert isinstance(gate.mlp[0], torch.nn.LayerNorm)
    assert gate.mlp[0].normalized_shape == (512,)
    assert all(param.count_nonzero() == 0 for param in gate.mlp[-1].parameters())
    text = torch.randn(3, 512)
    active = torch.tensor([[True, True, True], [True, True, False], [True, False, False]])
    values = gate(text, active)
    torch.testing.assert_close(values.sum(-1), torch.ones(3))
    assert (values >= 0).all()
    assert torch.equal(values[~active], torch.zeros(3))
    bias = torch.tensor([1.0, -0.5, -0.5])
    assert torch.equal(gate.b_init, bias)
    expected = bias.expand_as(active)
    torch.testing.assert_close(values, expected.masked_fill(~active, -torch.inf).softmax(-1))


def test_gate_path_does_not_backpropagate_to_text_encoder(inputs):
    model = learner().eval()
    with torch.no_grad():
        model.gate.mlp[-1].weight[0, 0] = 1.0
    text = model.enc_text(inputs["words"], inputs["counts"])
    sensory = model.encode_sensory(*(inputs[key] for key in
                                     ("video", "frame_valid", "temporal", "gaze", "body")))
    gate_outputs = []
    with model.gate.register_forward_hook(lambda _gate, _args, output: gate_outputs.append(output)):
        model.score(sensory, text)

    gate_outputs[0][0, 0, 0].backward()
    assert model.gate.mlp[0].weight.grad.count_nonzero() > 0
    assert all(param.grad is None for param in model.language.parameters())


@pytest.mark.parametrize("body_rows", [[True, False, True], [False, False, False]])
@torch.no_grad()
def test_aux_nce_uses_only_active_pairs(body_rows, monkeypatch):
    model = learner(lambda_gen=0.0)
    model.logit_scale.fill_(10)  # Exercise the same clamp for all logits.
    active = torch.tensor([[True, True, body] for body in body_rows])
    streams = torch.randn(3, 3, 3, generator=torch.Generator().manual_seed(3))
    scores = torch.randn(3, 3, generator=torch.Generator().manual_seed(4))
    monkeypatch.setattr(model, "encode_sensory", lambda *args: (None, active))
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
        text = model.enc_text(words, counts)
        sensory = model.encode_sensory(*(inputs[key] for key in
                                         ("video", "frame_valid", "temporal", "gaze", "body")))
        logits = model.logit_scale.exp() * model.score(sensory, text)
        fused, active, *_ = sensory
        expected = weighted_nce(logits, logits.new_ones(len(logits)))
        expected += model.lambda_gen * model.language.lm_loss(words, counts, fused, active)
    torch.testing.assert_close(actual["loss"], expected, atol=0, rtol=0)
