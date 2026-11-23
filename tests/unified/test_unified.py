import pytest
import torch
from torch.nn import functional as F

from nursery.learner.gaze import _gaze_class
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


@torch.no_grad()
def test_infonce_has_no_sensory_prefix(inputs):
    model = learner().eval()
    original = model(**inputs)
    altered = model(**dict(inputs, video=-inputs["video"], gaze=None, body=None))
    assert torch.equal(original["text"], altered["text"])
    assert torch.equal(original["text"], model.enc_text(inputs["words"], inputs["counts"]))
    assert not torch.equal(original["logits"], altered["logits"])


@torch.no_grad()
def test_single_word_query_pools_only_its_content_token():
    language = learner().eval().language
    words, counts = torch.tensor([[2, 0, 0]]), torch.tensor([1])
    hidden = language._hidden(language._tokens(words, counts))
    expected = F.normalize(language.project(hidden[:, 1]), dim=-1)
    torch.testing.assert_close(language.encode(words, counts), expected)
    torch.testing.assert_close(language.encode(words[:, :1], counts), expected)


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
def test_temporal_dropout_masks_whole_stream_only_in_training(inputs, monkeypatch):
    model = learner()
    args = [inputs[key] for key in ("video", "frame_valid", "temporal", "gaze", "body")]
    monkeypatch.setattr(torch, "rand", lambda n, *, device: torch.tensor([0.2, 0.8], device=device))
    fused, active, *_ = model.train().encode_sensory(*args)
    assert active.tolist() == [[True, False, True], [True, True, True]]
    assert fused[0, 1].count_nonzero() == 0
    weights = model.gate(model.enc_text(inputs["words"], inputs["counts"]), active)
    assert weights[0, 1] == 0
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))
    _, active, *_ = model.eval().encode_sensory(*args)
    assert active.all()


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
    text = torch.randn(3, 512)
    active = torch.tensor([[True, True, True], [True, True, False], [True, False, False]])
    values = gate(text, active)
    torch.testing.assert_close(values.sum(-1), torch.ones(3))
    assert (values >= 0).all()
    assert torch.equal(values[~active], torch.zeros(3))
    torch.testing.assert_close(values[0], torch.tensor([0.96, 0.02, 0.02]), atol=0.005, rtol=0)
