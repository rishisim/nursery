"""Synthetic checks for the shared joint learner; no experiment is launched."""

import pytest
import torch
import torch.nn.functional as F

from nursery.learner.additive import AdditiveMIL
from nursery.learner.interaction import JointInteractionMIL, centre_clip_valid

pytestmark = pytest.mark.usefixtures("offline", "upstream")
DIMS = {name: 3 for name in ("contact", "finger", "arm", "load", "skin")}


def inputs():
    torch.manual_seed(31)
    return (torch.randn(4, 6, 8), torch.arange(1, 5)[:, None], torch.ones(4),
            {"groups": {k: torch.randn(4, 7, d) for k, d in DIMS.items()},
             "missing": torch.zeros(4, 7), "seg": torch.zeros(4, 7, dtype=torch.long),
             "keep": torch.ones(4, dtype=torch.bool)})


def model():
    return JointInteractionMIL(12, 8, dims=DIMS).eval()


def test_zero_equivalence_initialization_and_rng():
    v, t, n, ctx = inputs()
    torch.manual_seed(19)
    additive = AdditiveMIL(12, 8, dims=DIMS).eval()
    after_additive = torch.random.get_rng_state()
    torch.manual_seed(19)
    joint = model()
    assert torch.equal(after_additive, torch.random.get_rng_state())
    for key, value in additive.state_dict().items():
        assert torch.equal(value, joint.state_dict()[key]), key
    torch.testing.assert_close(additive.pair_scores(v, t, n, ctx)[2], joint.pair_scores(v, t, n, ctx)[2], rtol=0, atol=0)
    with torch.no_grad():
        joint.interaction_out.weight.normal_()
    joint.interaction_enabled = False
    torch.testing.assert_close(additive.pair_scores(v, t, n, ctx)[2], joint.pair_scores(v, t, n, ctx)[2], rtol=0, atol=0)


def test_output_then_upstream_gradients_and_all_body_groups():
    v, t, n, ctx = inputs()
    joint = model()
    # Isolate the interaction's route into the existing body encoder.
    with torch.no_grad():
        joint.beta.zero_()
    joint.beta.requires_grad_(False)
    optimizer = torch.optim.SGD(joint.parameters(), lr=.1)
    for step in range(2):
        optimizer.zero_grad()
        scores = joint.pair_scores(v, t, n, ctx)[2]
        F.cross_entropy(scores, torch.arange(4)).backward()
        assert joint.interaction_out.weight.grad.abs().sum() > 0
        for layer in (joint.interaction_query, joint.interaction_key, joint.interaction_value):
            assert bool(layer.weight.grad.abs().sum() > 0) == bool(step)
        for name in DIMS:
            assert bool(joint.encoder.mlps[name][0].weight.grad.abs().sum() > 0) == bool(step)
        optimizer.step()


def test_contiguous_clip_missing_and_dropout_masks():
    v, t, n, ctx = inputs()
    ctx["seg"][0] = torch.tensor([0, 1, 0, 0, 0, 1, 0])
    ctx["missing"][0, 4] = 1
    assert centre_clip_valid(ctx)[0].tolist() == [False, False, True, True, False, False, False]
    joint = model()
    with torch.no_grad():
        joint.interaction_out.weight.normal_()
    R, T, _ = joint.pair_scores(v, t, n, ctx)
    original = joint.interaction_similarity(R, T, ctx)
    changed = dict(ctx, groups={k: g.clone() for k, g in ctx["groups"].items()})
    for group in changed["groups"].values():
        group[0, [0, 1, 5, 6]] += 1000
    torch.testing.assert_close(original[0], joint.interaction_similarity(R, T, changed)[0])
    for field in ("missing", "seg", "keep"):
        masked = dict(ctx)
        masked[field] = torch.ones_like(ctx[field]) if field == "missing" else torch.full_like(ctx[field], -2) if field == "seg" else torch.zeros_like(ctx[field])
        torch.testing.assert_close(joint.score_encoded(R, T, masked), joint.score_encoded(R, T))


def test_visual_masks_and_shared_training_evaluation_scoring():
    v, t, n, ctx = inputs()
    joint = model()
    with torch.no_grad():
        joint.interaction_out.weight.normal_()
    R, T, scores = joint.pair_scores(v, t, n, ctx)
    torch.testing.assert_close(scores, joint.score_encoded(R, T, ctx), rtol=0, atol=0)
    joint.train()
    torch.testing.assert_close(scores, joint.score_encoded(R, T, ctx), rtol=0, atol=0)
    ctx["visual_valid"] = torch.ones(4, 6, dtype=torch.bool)
    ctx["visual_valid"][:, -1] = False
    expected = joint.interaction_similarity(R, T, ctx)
    changed = R.clone()
    changed[:, -1] += 1000
    torch.testing.assert_close(expected, joint.interaction_similarity(changed, T, ctx))
    ctx["visual_valid"].zero_()
    result = joint.interaction_similarity(R, T, ctx)
    assert torch.isfinite(result).all() and torch.count_nonzero(result) == 0
