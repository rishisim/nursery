"""Check Step 7 temporal video against the approved protocol revision on CPU."""

from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nursery.learner.gaze import GazeMIL
from nursery.learner.gaze_data import GazeData, TemporalVideoData
from nursery.learner.video_only import VideoOnlyMIL
from nursery.learner.windows import BodyData

DIMS = dict(contact=22, fingers=90, arm=27, load=10, skin=4)


def reference(name):
    module = ModuleType(f"nursery.learner._reference_{name}")
    source = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"bb368b9e:nursery/learner/{name}.py"])
    exec(compile(source, module.__name__, "exec"), module.__dict__)
    return module


def check():
    torch.set_num_threads(2)
    old_video, old_gaze = reference("video_only"), reference("gaze")
    old_gaze._video_class = old_video._video_class
    torch.manual_seed(8)
    v = torch.randn(2, 5 * 257, 768)
    t = torch.tensor([[1], [2]])
    n, w = torch.ones(2, dtype=torch.long), torch.ones(2)
    gazectx = {"gaze_point": torch.rand(2, 5, 2), "gaze_valid": torch.rand(2, 5) > .2,
               "closeup": torch.randn(2, 5, 768), "keep": torch.tensor([True, False])}
    bodyctx = {"groups": {name: torch.randn(2, 11, dim) for name, dim in DIMS.items()},
               "missing": torch.zeros(2, 11), "seg": torch.zeros(2, 11, dtype=torch.long),
               "keep": torch.ones(2)}
    arms = (
        (old_video.VideoOnlyMIL, VideoOnlyMIL, {}, None),
        (old_gaze.GazeMIL, GazeMIL, {"option": "steering"}, gazectx),
        (old_gaze.GazeMIL, GazeMIL, {"option": "steering", "dims": DIMS}, dict(gazectx, body=bodyctx)),
    )

    same = torch.tensor([[False] * 2 + [True] * 7 + [False] * 2, [True] * 11])
    tokens = torch.randn(2, 11, 257, 768)
    temporal_gaze = {"same": same, "gaze_valid": torch.ones_like(same),
                     "gaze_point": torch.full((2, 11, 2), .25),
                     "closeup": torch.randn(2, 11, 768)}
    moved_gaze = dict(temporal_gaze, gaze_point=torch.full((2, 11, 2), .75))
    changed_counts = []
    for previous, current, kwargs, ctx in arms:
        torch.manual_seed(4)
        before = previous(5, 768, **kwargs)
        state = torch.random.get_rng_state()
        torch.manual_seed(4)
        off = current(5, 768, **kwargs, temporal_video=False)
        assert torch.equal(state, torch.random.get_rng_state())
        assert before.state_dict().keys() == off.state_dict().keys()
        expected = before.pair_scores(v, t, n, ctx)
        torch.random.set_rng_state(state)
        actual = off.pair_scores(v, t, n, ctx)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        state = torch.random.get_rng_state()
        expected_loss = before.forward_loss(v, t, n, w, ctx)
        torch.random.set_rng_state(state)
        assert torch.equal(expected_loss, off.forward_loss(v, t, n, w, ctx))

        enabled = current(5, 768, **kwargs, temporal_video=True).eval()
        delta = sum(p.numel() for p in enabled.parameters()) - sum(p.numel() for p in off.parameters())
        changed_counts.append(delta)
        frames = enabled.temporal_frames(tokens, temporal_gaze)
        moved = enabled.temporal_frames(tokens, moved_gaze)
        plain = enabled.temporal_frames(tokens, {"same": same})
        assert torch.equal(plain, tokens.mean(dim=-2))
        assert torch.equal(frames, moved) == (current is VideoOnlyMIL)
        if current is GazeMIL:
            no_sharp = dict(temporal_gaze, closeup=torch.zeros_like(temporal_gaze["closeup"]))
            assert not torch.allclose(frames[:, 5], enabled.temporal_frames(tokens, no_sharp)[:, 5])
            removed = dict(temporal_gaze, keep=torch.zeros(2, dtype=torch.bool))
            assert torch.equal(enabled.temporal_frames(tokens, removed), plain)

        encoded_words = enabled.enc_text(t, n)
        temporal = {"frames": frames, "same": same}
        score = enabled.temporal_score(encoded_words, temporal)
        changed = frames.masked_fill(~same[..., None], torch.nan)
        full_ctx = dict(ctx or {}, temporal=temporal)
        changed_ctx = dict(full_ctx, temporal=dict(temporal, frames=changed))
        expected = enabled.pair_scores(v, t, n, full_ctx)[2]
        assert torch.equal(expected, enabled.pair_scores(v, t, n, changed_ctx)[2])
        assert not torch.allclose(score, enabled.temporal_score(encoded_words,
                                  dict(temporal, frames=frames.flip(1))))
        base = enabled.pair_scores(v, t, n, ctx)[2]
        torch.testing.assert_close(expected, base + enabled.temporal_weight * score)
        loss = enabled.forward_loss(v, t, n, w, full_ctx)
        loss.backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in enabled.temporal_video.parameters())
        assert enabled.temporal_weight.grad.abs() > 0

        for mode in ("zero", "per_word"):
            zero = current(5, 768, **kwargs, temporal_video=True, temporal_weight_init=mode).eval()
            shared = {name: value for name, value in enabled.state_dict().items()
                      if name != "temporal_weight"}
            assert zero.load_state_dict(shared, strict=False).missing_keys == ["temporal_weight"]
            V, U, scores = zero.pair_scores(v, t, n, full_ctx)
            torch.testing.assert_close(scores, base, rtol=0, atol=0)
            zero.forward_loss(v, t, n, w, full_ctx).backward()
            assert zero.temporal_weight.grad.abs().sum() > 0
            if mode == "per_word":
                assert not zero.temporal_weight.grad[[0, 3, 4]].any()
                with torch.no_grad():
                    zero.temporal_weight[1], zero.temporal_weight[2] = 0.5, -0.25
                panel_ctx = dict(full_ctx, temporal_tokens=t.flip(0))
                actual = zero.score_encoded(V, U.flip(0), panel_ctx)
                expected = base.flip(1) + score.flip(1) * torch.tensor([-0.25, 0.5])
                torch.testing.assert_close(actual, expected)
                phrase = torch.tensor([[1, 2, 0]])
                phrase_score = zero.temporal_score(zero.enc_text(phrase, torch.tensor([2])), temporal, phrase)
                vectors = zero.word.weight[1:3, :128]
                weighted = (0.5 * vectors[0] - 0.25 * vectors[1]) / vectors.sum(dim=0).norm()
                latent = torch.nn.functional.normalize(zero.temporal_video(frames, same), dim=-1)
                torch.testing.assert_close(phrase_score[:, 0], latent @ weighted)

    # Two adjacent clips in the same partition. Align to the last second of
    # the first clip and first second of the second clip without crossing a cut.
    clip = np.array([[0, 0, 0, 1, 1]])
    rows = pd.DataFrame({"video_id": ["v", "v"], "frame_idx": [2, 3]})
    lut = {f"v|{i}": (0, i) for i in range(5)}
    cache = tokens[0, :5].numpy()
    data = TemporalVideoData(rows, [cache], lut, ({"v": 0}, clip))
    store = {"clip": clip, "present": clip >= 0, "next_valid": clip >= 0, "groups": {}, "targets": {}}
    body = BodyData(store, [0, 0], [2, 3], [[2], [3]], None, None, None, None, "cpu")
    ids = np.arange(2)
    body_same = body.batch(ids, None, training=False)["same"]
    model = VideoOnlyMIL(5, 768, temporal_video=True)
    context = data.batch(model, ids, device="cpu")
    assert torch.equal(context["same"], body_same)
    assert torch.equal(context["frames"][0, 3:6], torch.from_numpy(cache[:3]).mean(1))
    assert torch.equal(context["frames"][1, 5:7], torch.from_numpy(cache[3:]).mean(1))
    assert not context["frames"][~body_same].any()

    points = pd.DataFrame({"video_id": ["v"] * 5, "frame_idx": range(5), "gaze_x": .5, "gaze_y": .5})
    data.gaze = GazeData(rows, lut, points, partitions=dict.fromkeys(lut, "train"),
                         point_only=True, window=5, frame_keys=data.frame_keys)
    model = GazeMIL(5, 768, option="steering", point_only=True, temporal_video=True)
    dropped = data.batch(model, ids, device="cpu", gaze_mode="real", keep=torch.tensor([False, True]))
    assert torch.equal(dropped["frames"][0], context["frames"][0])
    assert not torch.allclose(dropped["frames"][1], context["frames"][1])

    assert changed_counts == [172929] * len(arms)
    print("PASS: flag-off scores and training loss bit-identical to bb368b9e in all three arms")
    print("PASS: same mask as body, masked frames invariant, gaze and sharp readout, frame-order sensitivity")
    print("PASS: temporal score and weight connected to loss, 172929 added parameters in every arm")
    print("PASS: both zero initializations, gradient routing, reordered queries and multi-token weighting")


if __name__ == "__main__":
    check()
