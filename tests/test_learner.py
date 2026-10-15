"""CPU-only synthetic validation of the embedding adapter, the body input and training."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_frames

from nursery.common import load_config
from nursery.learner import UPSTREAM, _upstream_module
from nursery.learner.embed import embed_episodes, probe_frames
from nursery.speech.transcripts import write_transcripts

pytestmark = pytest.mark.usefixtures("offline")


@pytest.mark.parametrize("available,mps,device", [(False, False, None), (False, True, None), (True, True, None), (True, True, "cpu"), (False, False, "cuda:0")])
def test_embedding_adapter_with_fake_encoder(episodes, upstream, monkeypatch, available, mps, device):
    import torch

    root, run = episodes
    module = _upstream_module("embed_regions")
    requested = []
    batches = []
    expected_device = device or ("cuda" if available else "mps" if mps else "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    class Inputs(dict):
        def to(self, device):
            assert device == expected_device
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, model):
            requested.append(model)
            return cls()

        def __call__(self, images, return_tensors):
            batches.append(len(images))
            return Inputs(batch=len(images))

    class Encoder:
        config = type("Config", (), {"hidden_size": 8, "num_register_tokens": 0})()

        @classmethod
        def from_pretrained(cls, model):
            requested.append(model)
            return cls()

        def eval(self):
            return self

        def to(self, device):
            assert device == expected_device
            return self

        def __call__(self, batch):
            tokens = torch.arange(65).reshape(1, 65, 1).expand(batch, 65, 8).float()
            return type("Result", (), {"last_hidden_state": tokens})()

    monkeypatch.setattr(module, "AutoImageProcessor", Processor)
    monkeypatch.setattr(module, "AutoModel", Encoder)
    from PIL import Image
    from nursery.learner import embed as embed_module

    # Konkle photos live outside the episode root; embed them from their own paths.
    photos = []
    for category in ("apple", "bowl"):
        for index in range(4):
            path = root.parent / "konkle" / category / f"{index}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
            photos.append({"video_id": f"konkle/{category}", "frame_idx": index, "category": category, "path": str(path)})
    monkeypatch.setattr(embed_module, "konkle_frames", lambda config: pd.DataFrame(photos))
    embed_episodes(root, run, device=device, batch=31)
    assert max(batches) == 31
    assert "path" not in pd.read_parquet(run / "konkle_frames.parquet").columns
    for name, count in (("train", 240), ("probes", 32), ("konkle", 8)):
        emb = np.load(run / f"{name}_cache" / "emb.f16.npy")
        index = pd.read_parquet(run / f"{name}_cache" / "index.parquet")
        assert emb.shape == (count, 17, 8) and emb.dtype == np.float16
        assert set(index.columns) == {"video_id", "frame_idx", "row"}
        expected = np.r_[0, np.array([[1 + 16 * y + 2 * x + 4.5 for x in range(4)] for y in range(4)]).ravel()]
        np.testing.assert_array_equal(emb[0, :, 0], expected)
    assert requested == ["facebook/dinov2-base"] * 6
    assert set(pd.read_parquet(run / "train_frames.parquet").video_id) == {"v0", "v1"}
    # Resume uses upstream's own cache logic and performs no additional model loads.
    embed_episodes(root, run, device="cpu")
    assert len(requested) == 6
    assert json.loads((run / "embedding.json").read_text())["device"] == "cpu"
    assert json.loads((run / "embedding.json").read_text())["batch"] == 64
    provenance = json.loads((run / "embedding.json").read_text())
    provenance["episodes"] = "other"
    (run / "embedding.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance"):
        embed_episodes(root, run)


def _fake_caches(root, run):
    rng = np.random.default_rng(12)
    frames = pd.read_parquet(root / "frames.parquet")
    frames = frames[frames.video_id.isin(["v0", "v1"])].copy()
    probes = probe_frames(pd.read_parquet(root / "probes.parquet"), load_config())
    probes.to_parquet(run / "probes_frames.parquet", index=False)
    for name, table in (("train", frames), ("probes", probes)):
        cache = run / f"{name}_cache"
        cache.mkdir()
        index = table[["video_id", "frame_idx"]].copy()
        index["row"] = np.arange(len(index))
        index.to_parquet(cache / "index.parquet", index=False)
        np.save(cache / "emb.f16.npy", rng.normal(size=(len(index), 17, 8)).astype(np.float16))


def test_cpu_train_cli_one_epoch(episodes, upstream):
    root, run = episodes
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    command = [os.sys.executable, "-m", "nursery.learner", "train", "--run", str(run),
               "--condition", "contact", "--seed", "8", "--epochs", "1"]
    result = subprocess.run(command, check=True, capture_output=True, text=True,
                            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    print(result.stdout)
    metrics = json.loads((run / "contact_s8" / "metrics.json").read_text())
    assert metrics["condition"] == "contact" and metrics["device"] in ("cpu", "mps")
    assert metrics["selection"] == "dev" and metrics["selected_epoch"] == 0
    assert metrics["epochs"] == 1 and metrics["n_pairs"] > 300 and metrics["emb_dim"] == 8
    assert metrics["generator_parameters"]["p_ref"] == 1
    assert metrics["frame_coverage"] == 1 and metrics["mean_weight"] == pytest.approx(1)
    assert 0 <= metrics["reported_test_acc"] <= 100
    import torch

    state = torch.load(run / "contact_s8" / "model.pt", map_location="cpu", weights_only=True)
    model = upstream.RegionMIL(metrics["vocab"], emb_dim=8)
    model.load_state_dict(state, strict=True)
    assert model.region_prior is None and all(value.device.type == "cpu" for value in state.values())


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_weights_follow_upstream_filtered_pair_indices(episodes, upstream, monkeypatch, device):
    import torch
    from nursery.learner.train import train_run

    monkeypatch.setattr(torch.cuda, "is_available", lambda: device == "cuda")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: device == "mps")
    move = torch.Tensor.to
    transfers = []

    def to(tensor, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)) and str(args[0]) in ("cuda", "mps"):
            transfers.append(tensor.shape)
            args = ("cpu", *args[1:])
        return move(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    root, run = episodes
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    manifest = pd.read_parquet(run / "manifest.parquet")
    manifest.loc[1, "text"] = "quux"
    manifest.to_parquet(run / "manifest.parquet", index=False)
    vocab = upstream.build_vocab(manifest.text, 5)
    retained = manifest[manifest.text.map(lambda t: bool(upstream.encode(t, vocab)))].reset_index(drop=True)
    expected = retained.weight_contact.to_numpy(dtype=np.float32)
    expected /= expected.mean()
    original = upstream.RegionMIL.forward_loss
    observed = []

    def loss(model, v, t, n, w):
        observed.extend(w.tolist())
        return original(model, v, t, n, w)

    original_collate = upstream.collate
    batches = []

    def collate(batch):
        result = original_collate(batch)
        batches.extend(expected[result[0].numpy()].tolist())
        return result

    monkeypatch.setattr(upstream.RegionMIL, "forward_loss", loss)
    monkeypatch.setattr(upstream, "collate", collate)
    region = _upstream_module("train_region_mil")
    selection_scores = iter([0.4, 0.9, 0.7, 0.1])

    def evaluate(*args, **kwargs):
        assert args[5] == device
        return next(selection_scores), {"n_scored": 4, "n_oov": 0, "n_total": 4}

    monkeypatch.setattr(region, "eval_4afc_region", evaluate)
    metrics = train_run(run, "contact", 9, epochs=2)
    assert metrics["device"] == device
    assert bool(transfers) == (device != "cpu")
    np.testing.assert_allclose(observed, batches, atol=1e-7)
    assert metrics["n_novocab"] == 1 and metrics["n_pairs"] == len(manifest) - 1
    assert metrics["selected_epoch"] == 1 and metrics["reported_test_acc"] == 10
    assert metrics["best_test_acc"] == 90


@pytest.mark.parametrize("failure", ["missing_train", "missing_probe", "small_probe", "oov_probe"])
def test_training_rejects_incomplete_inputs(episodes, upstream, failure):
    from nursery.learner.train import train_run

    root, run = episodes
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    if failure in ("missing_train", "missing_probe"):
        name = "train" if failure == "missing_train" else "probes"
        path = run / f"{name}_cache" / "index.parquet"
        table = pd.read_parquet(path).iloc[1:]
        table.to_parquet(path, index=False)
        reason = "does not cover"
    else:
        path = run / "probes_frames.parquet"
        table = pd.read_parquet(path)
        if failure == "small_probe":
            table = table.iloc[1:]
            reason = "four probes/category"
        else:
            table.loc[table.category == "mug", "category"] = "quux"
            reason = "four in-vocabulary"
        table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=reason):
        train_run(run, "contact", 8, epochs=1)


# ---- step 3: body input ----

def _body_session(seed, frames=120):
    rng = np.random.default_rng(seed)
    bits = np.zeros((frames, 1556), dtype=bool)
    bits[::3, :40] = True                                   # left hand touches every third second
    joints = rng.normal(size=(frames, 55, 3)).astype(np.float32) * 0.1
    joints[:, :, 2] -= 1.0                                  # everything in front of an identity camera
    camera = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    return dict(frame_idx=np.arange(frames), joints=joints, camera=camera,
                left_hand_pose=rng.normal(size=(frames, 45)).astype(np.float32),
                right_hand_pose=rng.normal(size=(frames, 45)).astype(np.float32),
                hand_contact=np.packbits(bits, axis=1), hand_contact_vertices=np.int32(1556))


def _write_body(root, video_ids=("v0", "v1")):
    (root / "body").mkdir(exist_ok=True)
    for k, video_id in enumerate(video_ids):
        np.savez(root / "body" / f"{video_id}.npz", **_body_session(k))


def test_projection_convention_and_regions():
    from nursery.learner import body

    camera = np.tile(np.eye(4), (3, 1, 1))
    camera[1, :3, 3] = (0.5, 0, 0)                           # camera-to-world: moved right by 0.5 m
    points = np.array([[0, 0, -1.0], [0.5, 0, -1.0], [0, 0, 1.0]])
    pixels, front = body.project_points(points, camera)
    np.testing.assert_allclose(pixels[0], (160, 120), atol=1e-4)
    np.testing.assert_allclose(pixels[1], (160, 120), atol=1e-4)  # same point relative to the moved camera
    assert front.tolist() == [True, True, False]
    right, _ = body.project_points(np.array([[0.2, 0.1, -1.0]]), camera[:1])
    assert right[0, 0] > 160 and right[0, 1] < 120            # +x is right, +y is up (smaller row)
    centre = body.GRID_CENTRES[5]                             # region (1, 1)
    d = body.region_distance(centre[None], np.array([True]))[0]
    assert d[6] == pytest.approx(0, abs=1e-5) and d[1:].argmin() == 5 and d.shape == (17,)
    assert body.region_distance(centre[None], np.array([False]))[0].tolist() == [body.D_NEUTRAL] * 17
    joints = np.zeros((1, 55, 3), np.float32)
    joints[0, body.RIGHT_HAND] = (0, 0, -1)
    dist, _ = body.project_hand(joints, camera[:1], np.array([-1]))
    assert (dist == body.D_NEUTRAL).all()                     # no hand in contact: no preference
    dist, pix = body.project_hand(joints, camera[:1], np.array([1]))
    np.testing.assert_allclose(pix[0], (160, 120), atol=1e-4)


def test_contact_is_coarse_and_joints_are_pose_invariant():
    from nursery.learner import body

    bits = np.zeros((3, 1556), dtype=bool)
    bits[1, :5] = True
    bits[2, 778:778 + 200] = True
    scalars, active = body._contact(np.packbits(bits, axis=1))
    assert scalars.shape == (3, 4) and active.tolist() == [-1, 0, 1]
    np.testing.assert_allclose(scalars[1], [1, 1 / 3, 0, 0])
    np.testing.assert_allclose(scalars[2], [0, 0, 1, 1])
    joints = np.random.default_rng(0).normal(size=(2, 55, 3)).astype(np.float32)
    theta = 0.7
    turn = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], np.float32)
    moved = joints @ turn.T + np.array([3, -2, 0], np.float32)
    np.testing.assert_allclose(body._joints(joints), body._joints(moved), atol=1e-4)
    assert body._joints(joints).shape == (2, 6)


def test_body_windows_match_framepairs(upstream):
    from nursery.learner import body

    region = _upstream_module("train_region_mil")
    present = {("v0", f) for f in range(120) if f not in (5, 6, 7, 8)} | {("v1", f) for f in range(10, 120)}
    lut = {region.frame_key(v, f): (0, k) for k, (v, f) in enumerate(sorted(present))}
    manifest = pd.DataFrame([{"video_id": v, "frame_idx": f, "text": "mug"}
                             for v, f in (("v0", 0), ("v0", 1), ("v0", 4), ("v0", 9), ("v0", 119), ("v1", 10), ("v1", 12))])
    vocab = {"<pad>": 0, "mug": 1}
    ds = upstream.FramePairs([np.zeros((len(lut), 17, 4), np.float16)], lut, manifest, vocab, 2, False)
    has = lambda v, f: region.frame_key(v, f) in lut
    for i, row in enumerate(manifest.itertuples()):
        frames = body.window_frames(row.video_id, row.frame_idx, has, 2)
        assert len(frames) == 5
        unpadded = [lut[region.frame_key(row.video_id, f)] for f in dict.fromkeys(frames)]
        assert unpadded == ds.locs[i]
    sessions = {"video_ids": ["v0", "v1"], "row": {"v0": 0, "v1": 1}, "frame_row": np.tile(np.arange(120), (2, 1)),
                "dist": np.arange(2 * 120 * 17, dtype=np.float32).reshape(2, 120, 17),
                **{k: np.arange(2 * 120, dtype=np.float32).reshape(2, 120, 1).repeat(n, 2)
                   for k, n in (("contact", 4), ("hand", 2), ("joints", 6))}}
    pairs = body.pair_arrays(sessions, manifest, has, 2)
    first = pairs["dist"][0].reshape(5, 17)                   # v0 t=0: frames 0,1,2 then padding with frame 0
    np.testing.assert_array_equal(first[:, 0], sessions["dist"][0, [0, 1, 2, 0, 0], 0])
    assert pairs["contact"][0, 0] == pytest.approx(1.0)       # mean of frames 0, 1, 2
    assert pairs["dist"].shape == (len(manifest), 85)


def test_drop_masks_and_isolated_rng():
    import torch
    from nursery.learner import body

    state = torch.get_rng_state()
    generator = torch.Generator().manual_seed(0)
    on, keep, noise = body.drop_masks(20000, body.SIGNALS, 0.5, generator)
    assert on.mean().item() == pytest.approx(0.5, abs=0.02) and keep.mean().item() == pytest.approx(0.7, abs=0.02)
    assert keep.shape == (20000, 3) and noise.shape == (20000, 12) and noise.std().item() == pytest.approx(body.INPUT_NOISE, abs=0.02)
    assert body.drop_masks(100, body.SIGNALS, 1.0, generator)[0].sum() == 0
    assert body.drop_masks(100, body.SIGNALS, 0.0, generator)[0].sum() == 100
    assert torch.equal(torch.get_rng_state(), state)           # default stream (shuffling, dropout) untouched
    again = body.drop_masks(20000, body.SIGNALS, 0.5, torch.Generator().manual_seed(0))
    assert torch.equal(again[0], on) and torch.equal(again[1], keep)


def test_body_model_matches_upstream_without_body_and_gate_learns(upstream):
    import torch
    from nursery.learner import body

    region = _upstream_module("train_region_mil")
    torch.manual_seed(0)
    plain = region.RegionMIL(12, emb_dim=8)
    state = torch.get_rng_state()
    torch.manual_seed(0)
    model = body.BodyRegionMIL(12, emb_dim=8)
    # Building the body encoders leaves the default RNG stream where video-only training has it.
    assert torch.equal(torch.get_rng_state(), state) and model.region_prior is None
    model.load_state_dict({**plain.state_dict(), **{k: v for k, v in model.state_dict().items() if k.startswith("body.")}})
    v, t, n, w = torch.randn(6, 85, 8), torch.randint(1, 12, (6, 4)), torch.full((6,), 4), torch.ones(6)
    model.eval(); plain.eval()
    reference = plain.forward_loss(v, t, n, w)
    assert torch.equal(model.forward_loss(v, t, n, w), reference)  # pool=max, no body: upstream path, byte-identical
    with pytest.raises(ValueError, match="pool='soft'"):
        model.forward_loss(v, t, n, w, torch.rand(6, 85), {}, torch.ones(6), torch.ones(6, 3), torch.zeros(6, 12))
    soft = body.BodyRegionMIL(12, SIGNALS_ALL, pool="soft", tau_p=0.1, emb_dim=8)
    soft.load_state_dict(model.state_dict())
    soft.eval()
    feats = {k: torch.randn(6, d) for k, d in (("contact", 4), ("hand", 2), ("joints", 6))}
    dist = torch.rand(6, 85) * 3
    keep, noise = torch.ones(6, 3), torch.zeros(6, 12)
    plain_soft = soft.forward_loss(v, t, n, w)                  # soft pooling, uniform prior
    off = soft.forward_loss(v, t, n, w, dist, feats, torch.zeros(6), keep, noise)
    torch.testing.assert_close(off, plain_soft)                 # whole body dropped: gate 0, uniform prior
    soft.train()
    loss = soft.forward_loss(v, t, n, w, dist, feats, torch.ones(6), keep, noise)
    loss.backward()
    assert soft.body.head.weight.grad.abs().sum() > 0          # the gate is trained through the prior
    assert body.BodyRegionMIL(12, None, pool="soft", emb_dim=8).body is None


SIGNALS_ALL = ("contact", "hand", "joints")


def test_soft_pool_shape_and_uniform_prior_at_zero_gate():
    import torch
    from nursery.learner import body

    dist = torch.rand(5, 85) * 3
    uniform = body.region_log_prior(dist, torch.zeros(5))
    torch.testing.assert_close(uniform, torch.full((5, 85), -np.log(85)))     # g = 0: uniform prior
    sharp = body.region_log_prior(dist, torch.full((5,), 3.0))
    torch.testing.assert_close(sharp.exp().sum(1), torch.ones(5))
    assert (sharp.argmax(1) == dist.argmin(1)).all()                          # mass goes to the region nearest the hand
    s = torch.randn(5, 85, 7)
    pooled = body.soft_pool(s, uniform, 0.1)
    assert pooled.shape == (5, 7)
    torch.testing.assert_close(pooled, 0.1 * (torch.logsumexp(s / 0.1, 1) - np.log(85)))
    assert (pooled <= s.max(1).values + 1e-6).all()
    near_max = body.soft_pool(s, uniform, 1e-3)                               # tau -> 0 recovers the hard max (up to tau * log R)
    torch.testing.assert_close(near_max, s.max(1).values, atol=1e-2, rtol=0)


def test_upstream_checkout_is_unmodified(upstream):
    assert subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--quiet"]).returncode == 0


@pytest.mark.parametrize("options", [
    {"body": "all"}, {"body": "geometry"}, {"body": "contact,joints", "body_control": "shuffle"},
    {"body": "all", "body_control": "oracle", "body_drop": 1.0}, {"body": None}])
def test_body_training_runs(episodes, upstream, options):
    from nursery.learner.train import train_run

    root, run = episodes
    _write_body(root)
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    metrics = train_run(run, "none", 3, epochs=2, name="body", pool="soft", tau_p=0.1, **options)
    assert metrics["pool"] == "soft" and metrics["tau_p"] == 0.1
    if options["body"] is None:
        assert "body" not in metrics
    else:
        assert metrics["body"]["control"] == options.get("body_control", "none")
        gate = metrics["body"]["gate"]
        assert 0 <= gate["mean"] <= 3 and 0 <= gate["frac_below_0.1"] <= 1
        assert "gate_mean" in metrics["history"][0]
    assert (run / "body_s3" / "metrics.json").exists()
    with pytest.raises(ValueError, match="requires --body"):
        train_run(run, "none", 4, epochs=1, body_control="shuffle")
    with pytest.raises(ValueError, match="requires --pool soft"):
        train_run(run, "none", 5, epochs=1, body="all")
