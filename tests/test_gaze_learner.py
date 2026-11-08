"""Synthetic software checks only; no pilot, optimization run or score selection."""

import pytest
import torch

from nursery.learner.gaze import GazeMIL, compact_closeup
from nursery.learner.video_only import VideoOnlyMIL

pytestmark = pytest.mark.usefixtures("offline", "upstream")


def inputs():
    generator = torch.Generator().manual_seed(12)
    video = torch.randn(3, 2 * 257, 8, generator=generator)
    text = torch.tensor([[1, 2], [2, 3], [3, 1]])
    lengths = torch.tensor([2, 2, 2])
    ctx = {"gaze_point": torch.tensor([[[.1, .1], [.2, .2]], [[.7, .8], [.8, .7]], [[.4, .4], [.5, .5]]]),
           "gaze_valid": torch.ones(3, 2, dtype=torch.bool),
           "closeup": torch.randn(3, 2, 8, generator=generator),
           "keep": torch.ones(3, dtype=torch.bool)}
    return video, text, lengths, ctx


@pytest.mark.parametrize("option", ["additive", "steering"])
@pytest.mark.parametrize("point_only", [False, True])
def test_removed_is_exact_baseline_and_keeps_initialization(option, point_only):
    torch.manual_seed(4)
    baseline = VideoOnlyMIL(5, 8).eval()
    rng = torch.random.get_rng_state()
    torch.manual_seed(4)
    model = GazeMIL(5, 8, option=option, point_only=point_only).eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    video, text, lengths, ctx = inputs()
    expected = baseline.pair_scores(video, text, lengths)[2]
    assert torch.equal(model.pair_scores(video, text, lengths, None)[2], expected)
    ctx["keep"][:] = False
    ctx["gaze_point"][:] = float("nan")
    ctx["closeup"][:] = float("nan")
    assert torch.equal(model.pair_scores(video, text, lengths, ctx)[2], expected)


@pytest.mark.parametrize("option", ["additive", "steering"])
@pytest.mark.parametrize("point_only", [False, True])
def test_gaze_reaches_language_loss_and_removed_row_is_exact(option, point_only):
    model = GazeMIL(5, 8, option=option, point_only=point_only).eval()
    video, text, lengths, ctx = inputs()
    ctx["gaze_point"].requires_grad_(True)
    ctx["closeup"].requires_grad_(True)
    ctx["keep"][2] = False
    expected = model.pair_scores(video, text, lengths)[2]
    actual = model.pair_scores(video, text, lengths, ctx)[2]
    assert torch.equal(expected[2], actual[2])
    assert not torch.allclose(expected[:2], actual[:2])
    loss = model.forward_loss(video, text, lengths, torch.ones(3), ctx)
    assert torch.isfinite(loss)
    loss.backward()
    assert ctx["gaze_point"].grad[:2].abs().sum() > 0
    assert ctx["gaze_point"].grad[2].abs().sum() == 0
    if point_only:
        assert ctx["closeup"].grad is None
    else:
        assert ctx["closeup"].grad[:2].abs().sum() > 0
        assert ctx["closeup"].grad[2].abs().sum() == 0
    assert model.vproj[-1].weight.grad.abs().sum() > 0
    assert model.word.weight.grad.abs().sum() > 0
    if option == "additive":
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gaze_encoder.parameters())
        assert model.beta.grad is not None and model.beta.grad.abs() > 0


def test_steering_selects_correct_xy_patch_and_outside_crop_is_uniform():
    model = GazeMIL(3, 2, option="steering", point_only=True, num_regions=5,
                    steering_sigma=.05).eval()
    # CLS, top left, top right, bottom left, bottom right.
    regions = torch.tensor([[[0., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.]]])
    words = torch.eye(2)
    ctx = {"gaze_point": torch.tensor([[[.25, .25]]]), "gaze_valid": torch.ones(1, 1, dtype=torch.bool)}
    top_left = model.score_encoded(regions, words, ctx)
    assert top_left[0, 0] > top_left[0, 1]
    ctx["gaze_point"] = torch.tensor([[[.75, .25]]])
    top_right = model.score_encoded(regions, words, ctx)
    assert top_right[0, 1] > top_right[0, 0]
    ctx["gaze_point"] = torch.tensor([[[-.1, .25]]])
    torch.testing.assert_close(model.score_encoded(regions, words, ctx), model.score_encoded(regions, words))


@pytest.mark.parametrize("option", ["additive", "steering"])
def test_invalid_frame_values_cannot_affect_scores(option):
    model = GazeMIL(5, 8, option=option).eval()
    video, text, lengths, ctx = inputs()
    ctx["gaze_valid"][:, 1] = False
    expected = model.pair_scores(video, text, lengths, ctx)[2]
    ctx["gaze_point"][:, 1] = float("nan")
    ctx["closeup"][:, 1] = float("nan")
    assert torch.equal(expected, model.pair_scores(video, text, lengths, ctx)[2])


@pytest.mark.parametrize("option", ["additive", "steering"])
def test_point_only_never_requires_or_reads_closeup(option):
    model = GazeMIL(5, 8, option=option, point_only=True).eval()
    video, text, lengths, ctx = inputs()
    expected = model.pair_scores(video, text, lengths, ctx)[2]
    del ctx["closeup"]
    assert torch.equal(expected, model.pair_scores(video, text, lengths, ctx)[2])


def test_compact_summary_excludes_cls_and_uses_float_accumulation():
    values = torch.arange(257).half().reshape(1, 257, 1)
    result = compact_closeup(values)
    assert result.shape == (1, 1)
    assert result.item() == 128.5
    values[:, 0] = 999
    assert torch.equal(result, compact_closeup(values))
    assert result.dtype == torch.float32
    with pytest.raises(ValueError, match="square"):
        compact_closeup(torch.zeros(2, 6, 8))


def test_shape_and_finiteness_guards():
    model = GazeMIL(5, 8).eval()
    video, text, lengths, ctx = inputs()
    ctx["closeup"] = torch.zeros(3, 2, 257, 8)
    with pytest.raises(ValueError, match="compact"):
        model.pair_scores(video, text, lengths, ctx)
    ctx["closeup"] = torch.zeros(3, 2, 8)
    ctx["gaze_point"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.pair_scores(video, text, lengths, ctx)
    with pytest.raises(ValueError, match="square"):
        GazeMIL(5, 8, num_regions=256)


@pytest.mark.parametrize("missing_all", [False, True])
def test_steering_missing_frames_and_narrow_prior_have_finite_backward(missing_all):
    model = GazeMIL(5, 8, option="steering", steering_sigma=.005).eval()
    video, text, lengths, ctx = inputs()
    ctx["gaze_valid"][:, 1] = False
    ctx["gaze_valid"][2] = False
    if missing_all:
        ctx["gaze_valid"][:] = False
    ctx["gaze_point"][~ctx["gaze_valid"]] = float("nan")
    ctx["closeup"][~ctx["gaze_valid"]] = float("nan")
    ctx["gaze_point"].requires_grad_(True)
    loss = model.forward_loss(video, text, lengths, torch.ones(3), ctx)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if not missing_all:
        assert torch.isfinite(ctx["gaze_point"].grad).all()
