"""Small fixed-panel evaluation checks with synthetic cache and body histories."""

import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

from nursery.learner.acquisition_evaluate import StudyEvaluation, _metrics
from nursery.learner.additive import AdditiveMIL
from nursery.learner.interaction import JointInteractionMIL
from nursery.learner.video_only import VideoOnlyMIL
from nursery.learner.windows import BodyData

pytestmark = pytest.mark.usefixtures("offline", "upstream")
DIMS = {k: 2 for k in ("contact", "finger", "arm", "load", "skin")}


@pytest.fixture
def evaluation(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps({"picture_panels": {}}))
    panel = tmp_path / "panels" / "dev"
    panel.mkdir(parents=True)
    rows = pd.DataFrame({"video_id": [f"v{i}" for i in range(4)], "frame_idx": [2] * 4,
                         "text": ["a completely misleading sentence"] * 4,
                         "referent": ["mug", "unknown", "mug", "unknown"]})
    rows.to_parquet(panel / "manifest.parquet")
    trials = {"trial_rows": [0, 1], "candidate_rows": [[0, 1, 2, 3], [3, 2, 1, 0]],
              "correct_slots": [0, 3], "donor_rows": [2, 3, 0, 1]}
    (panel / "trials.json").write_text(json.dumps(trials))
    cache = np.random.default_rng(9).normal(size=(4, 17, 8)).astype(np.float32)
    lut = {f"v{i}|2": (0, i) for i in range(4)}
    ev = StudyEvaluation(tmp_path, tmp_path, [cache], lut, {"<pad>": 0, "mug": 1}, None, "cpu")
    rows, trials, ds, _, valid = ev._panel("dev")
    store = {"groups": {k: np.random.default_rng(i).normal(size=(4, 7, d)).astype(np.float32)
                        for i, (k, d) in enumerate(DIMS.items())},
             "targets": {}, "clip": np.tile(np.arange(4)[:, None], (1, 7)),
             "present": np.ones((4, 7), dtype=bool), "next_valid": np.zeros((4, 7), dtype=bool)}
    store["present"][1, 1] = False
    store["present"][2, 3:5] = False
    data = BodyData(store, np.arange(4), np.array([2, 3, 4, 5]), np.full((4, 5), 2),
                    None, None, None, None, "cpu", half=5, mask_prob=0., dropout=0.)
    ev.panels["dev"] = rows, trials, ds, data, valid
    return ev


def test_unknown_query_keeps_fixed_denominator():
    result = _metrics([[4, 0, 0, 0], [0, 0, 0, 4]], [True, False], [[0, 1, 2, 3]] * 2,
                      [0, 0], ["mug", "unknown"])
    assert result["accuracy"] == 62.5 and result["coverage"] == .5
    assert result["n_trials"] == 2 and result["credits"] == [1., .25]
    assert result["predicted_slots"][1] is None and result["scores"][1] is None


@pytest.mark.parametrize("factory", [VideoOnlyMIL, AdditiveMIL, JointInteractionMIL])
def test_noun_query_placeholder_independence_and_score_identity(evaluation, factory):
    ev = evaluation
    kwargs = {} if factory is VideoOnlyMIL else {"dims": DIMS}
    model = factory(2, 8, **kwargs).eval()
    result = ev.evaluate(model, "dev")
    assert result["queries"] == ["mug", "unknown"] and result["n_known"] == 1
    rows, trials, ds, data, valid = ev._panel("dev")
    _, images, _, _ = ev.upstream.collate([ds[i] for i in range(4)])
    query = torch.ones(1, 1, dtype=torch.long)
    ctx = None if model.encoder is None else data.batch(np.arange(4), None, training=False)
    if ctx is not None:
        from nursery.learner.interaction import centre_clip_valid
        ctx["keep"] *= centre_clip_valid(ctx).any(1)
        ctx["visual_valid"] = torch.as_tensor(valid).repeat_interleave(17, dim=1)
    expected = model.pair_scores(images, query, torch.ones(1), ctx)[2][:, 0]
    np.testing.assert_allclose(result["scores"][0], expected.detach().numpy(), rtol=1e-6)
    # Dataset placeholder tokens and original sentence cannot become the noun query.
    ds.ids = [[0]] * 4
    rows["text"] = "unknown"
    assert ev.evaluate(model, "dev")["scores"] == result["scores"]


def test_shuffled_complete_histories_and_fixed_candidates(evaluation, monkeypatch):
    ev = evaluation
    model = JointInteractionMIL(2, 8, dims=DIMS).eval()
    observed = []
    original = model.score_encoded
    def record(R, T, ctx=None):
        observed.append(ctx)
        return original(R, T, ctx)
    monkeypatch.setattr(model, "score_encoded", record)
    real = ev.evaluate(model, "dev", "real")
    shuffled = ev.evaluate(model, "dev", "shuffled")
    absent = ev.evaluate(model, "dev", "none")
    assert real["candidate_rows"] == shuffled["candidate_rows"] == absent["candidate_rows"]
    donor = torch.tensor(shuffled["donor_rows"])
    for key in ("missing", "seg", "same", "keep"):
        torch.testing.assert_close(observed[1][key], observed[0][key][donor])
    for key in DIMS:
        torch.testing.assert_close(observed[1]["groups"][key], observed[0]["groups"][key][donor])
    assert observed[2] is None


def test_repeated_dev_evaluation_preserves_model_and_rng(evaluation):
    # Training caller explicitly enters evaluation mode before invoking this API.
    model = JointInteractionMIL(2, 8, dims=DIMS).eval()
    parameters = {k: v.clone() for k, v in model.state_dict().items()}
    torch_state, numpy_state, python_state = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
    before_modes = [m.training for m in model.modules()]
    first = evaluation.evaluate(model, "dev")
    assert evaluation.evaluate(model, "dev") == first
    assert [m.training for m in model.modules()] == before_modes
    assert torch.equal(torch_state, torch.random.get_rng_state())
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0] and numpy_state[2:] == after_numpy[2:]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    assert python_state == random.getstate()
    for key, value in model.state_dict().items():
        assert torch.equal(parameters[key], value), key
