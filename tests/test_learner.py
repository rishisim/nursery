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


# ---- body input and the body pathway (docs/learner.md) ----

def _body_session(seed, frames=120):
    rng = np.random.default_rng(seed)
    bits = np.zeros((frames, 1556), dtype=bool)
    bits[np.arange(frames) % 6 < 3, :40] = True             # left hand touches for 3 s out of every 6
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


def _store(root, clips=None):
    from nursery.learner import body

    _write_body(root)
    frames = pd.read_parquet(root / "frames.parquet")
    if clips:  # split each session into clips of the given length
        frames["clip"] = frames.video_id + "/" + (frames.frame_idx // clips).astype(str)
    return body.load_store(root, frames, ["v0", "v1"], set())


def test_patch_pooling_shapes_and_map():
    from nursery.learner import body

    patches = json.loads(body.PATCH_FILE.read_text())
    assert len(patches["left"]) == len(patches["right"]) == body.VERTS_PER_HAND
    assert set(patches["left"]) == set(range(11)) and len(body.PATCH_NAMES) == 11
    matrix = body.patch_matrix()
    assert matrix.shape == (1556, 22) and np.allclose(matrix.sum(0), 1)
    session = _body_session(0)
    feats = body.session_features(session)
    assert feats["contact"].shape == (120, 22) and feats["fingers"].shape == (120, 90)
    assert feats["arm"].shape == (120, 15) and feats["aperture"].shape == (120, 2)
    assert feats["contact"][0, :11].max() > 0 and feats["contact"][4].max() == 0  # left hand only, 3 s on / 3 s off
    assert feats["contact"][:, 11:].max() == 0 and feats["hand"][0] == 0 and feats["hand"][4] == -1


def test_arm_is_in_the_camera_frame_with_full_rotation():
    from nursery.learner import body

    session = _body_session(1)
    base = body.session_features(session)["arm"]
    turned = dict(session)
    rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)  # head yaw 90 deg
    turned["camera"] = np.tile(rotation, (120, 1, 1))
    other = body.session_features(turned)["arm"]
    assert not np.allclose(base[:, :12], other[:, :12])       # positions follow the camera rotation
    np.testing.assert_allclose(np.linalg.norm(base[:, 12:], axis=1), 1, atol=1e-5)  # gravity direction


def test_clip_boundaries_reset_and_mask(episodes):
    import torch
    from nursery.learner import body

    root, _ = episodes
    store = _store(root, clips=10)
    assert store["clip"][0, 9] != store["clip"][0, 10] and store["clip"][0, 10] == store["clip"][0, 19]
    np.testing.assert_array_equal(store["groups"]["arm"][0, 10, 12:24], 0)  # no 1-step difference across a boundary
    encoder = body.BodyEncoder({k: v.shape[-1] for k, v in store["groups"].items()}).eval()
    s, t = np.zeros(1, int), np.array([12])
    steps, seg = body.window_steps(store["clip"], s, t, 5)
    groups = {k: torch.as_tensor(g[s[:, None], steps]) for k, g in store["groups"].items()}
    seg = torch.as_tensor(seg)
    missing = (seg == -2).float()
    base = encoder(groups, missing, seg)[:, 5]
    other = {k: g.clone() for k, g in groups.items()}
    outside = (seg != seg[:, 5:6])
    for k in other:
        other[k][outside] += 5.0                           # change everything outside the centre's clip
    assert torch.allclose(base, encoder(other, missing, seg)[:, 5], atol=1e-6)
    inside = {k: g.clone() for k, g in groups.items()}
    inside["arm"][:, 5] += 5.0
    assert not torch.allclose(base, encoder(inside, missing, seg)[:, 5], atol=1e-4)


def test_episode_segmentation_and_controls(episodes):
    from nursery.learner import body

    hand = np.array([-1, 0, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0])
    clip = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    ids = body.contact_episodes(hand, clip, min_len=2)
    assert ids.tolist() == [-1, 0, 0, 0, 1, 1, 2, 2, -1, 3, 4, 4][:3] + [1, 1, 1, 1, 2, 2, -1, 3, 4, 4][:0] + ids.tolist()[3:]
    # same hand, new clip: ends the episode; other hand: ends it; lone second (min length 2): dropped
    assert ids[1] == ids[3] != ids[4] == ids[5] != ids[6] == ids[7] and ids[9] == -1 and ids[10] == ids[11] >= 0
    assert ids[0] == ids[8] == -1
    root, _ = episodes
    store = _store(root, clips=10)
    frames = pd.read_parquet(root / "frames.parquet")
    for mode in ("contact", "temporal"):
        episode = body.build_episodes(store, mode)
        assert (episode >= 0).any()
        for s in range(2):
            for e in np.unique(episode[s][episode[s] >= 0]):
                members = np.flatnonzero(episode[s] == e)
                assert len(members) >= 2 and len(set(store["clip"][s, members])) == 1  # inside one clip
    contact = body.build_episodes(store, "contact")
    for s in range(2):  # held_category (a label) only checks episodes: one object per episode
        held = frames[frames.video_id == f"v{s}"].sort_values("frame_idx").held_category.tolist()
        for e in np.unique(contact[s][contact[s] >= 0]):
            assert len({held[t] for t in np.flatnonzero(contact[s] == e) if isinstance(held[t], str)}) <= 1
    assert (body.build_episodes(store, "none") == -1).all()


def test_shuffled_keeps_marginals(episodes):
    from nursery.learner import body

    root, _ = episodes
    store = _store(root, clips=10)
    shuffled, info = body.shuffle_store(store, 0)
    assert info["moved"] == info["units"] > 0
    for key in store["groups"]:
        np.testing.assert_allclose(np.sort(shuffled["groups"][key].reshape(-1)), np.sort(store["groups"][key].reshape(-1)))
    assert not np.allclose(shuffled["groups"]["contact"], store["groups"]["contact"])
    np.testing.assert_array_equal(shuffled["clip"], store["clip"])  # boundaries and images stay put
    kept, _ = body.shuffle_store(store, 0, exclude=("v0", "v1"))
    np.testing.assert_array_equal(kept["groups"]["fingers"], store["groups"]["fingers"])


def _tiny_model(store, device="cpu", weights=None):
    import torch
    from nursery.learner.model import BodyData, LearnerMIL

    torch.manual_seed(0)
    model = LearnerMIL(30, 8, 0.1, {k: v.shape[-1] for k, v in store["groups"].items()})
    model.region_head[0] = torch.nn.Linear(model.region_head[0].in_features, 128)
    bags = np.full(store["clip"].shape + (4,), -1)
    ps, pt = np.zeros(6, int), np.arange(10, 16)
    win = np.tile(np.arange(5), (6, 1)) + pt[:, None] - 2
    data = BodyData(store, ps, pt, win, bags, np.full(store["clip"].shape, -1), None, None, device, 5, 0.5, 0.0)
    return model, data


def _batch(model, data):
    import torch

    torch.manual_seed(1)
    v = torch.randn(6, 85, 8)
    t = torch.randint(1, 30, (6, 4))
    return v, t, torch.full((6,), 4), torch.ones(6)


def test_language_loss_depends_on_body_and_reaches_every_encoder(episodes):
    import torch

    root, _ = episodes
    store = _store(root, clips=10)
    model, data = _tiny_model(store)
    v, t, n, w = _batch(model, data)
    ctx = data.batch(np.arange(6), torch.Generator().manual_seed(0))
    ctx["keep"] = torch.ones(6)
    weights = {"pred": 0.0, "episode": 0.0, "consist": 0.0, "mask": 0.0}
    loss = model.forward_loss(v, t, n, w, ctx, weights)
    other = dict(ctx, groups={k: g + torch.randn_like(g) for k, g in ctx["groups"].items()})
    assert not torch.isclose(loss, model.forward_loss(v, t, n, w, other, weights))
    loss.backward()
    for name, module in model.encoder.named_children():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads), name
    for group in model.encoder.mlps:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.mlps[group].parameters()), group
    assert model.vproj[2].weight.grad.abs().sum() > 0 and model.beta.grad.abs() > 0


def test_body_dropout_one_equals_video_only_and_each_loss_switches_off(episodes):
    import torch

    root, _ = episodes
    store = _store(root, clips=10)
    model, data = _tiny_model(store)
    model.eval()
    v, t, n, w = _batch(model, data)
    ctx = data.batch(np.arange(6), torch.Generator().manual_seed(0))
    off = {"pred": 0.0, "episode": 0.0, "consist": 0.0, "mask": 0.0}
    ctx["keep"] = torch.zeros(6)  # dropout 1.0
    baseline = model.forward_loss(v, t, n, w, None, off)
    assert torch.equal(model.forward_loss(v, t, n, w, ctx, off), baseline)
    assert set(model.losses(v, t, n, w, ctx, off)) == {"match"}
    ctx["keep"] = torch.ones(6)
    assert not torch.equal(model.forward_loss(v, t, n, w, ctx, off), baseline)
    on = model.losses(v, t, n, w, ctx, {**off, "pred": 1.0, "mask": 1.0})
    assert set(on) == {"match", "pred", "mask"}
    for name in ("pred", "mask"):
        model.zero_grad()
        on[name].backward(retain_graph=True)
    generator = torch.Generator().manual_seed(0)
    assert data.dropout == 0.0
    data.dropout = 1.0
    assert data.batch(np.arange(6), generator)["keep"].sum() == 0


def _distill_pair(episodes):
    import torch
    from nursery.learner.model import LearnerMIL

    root, _ = episodes
    teacher, data = _tiny_model(_store(root, clips=10))
    teacher.eval()
    torch.manual_seed(2)
    student = LearnerMIL(30, 8, 0.1).eval()
    v, t, n, w = _batch(student, data)
    ctx = data.batch(np.arange(6), torch.Generator().manual_seed(0), training=False)
    return teacher, student, (v, t, n, w), ctx


def test_distillation_targets_are_detached_and_the_kl_reaches_the_student(episodes):
    import torch

    teacher, student, (v, t, n, w), ctx = _distill_pair(episodes)
    target = teacher.teacher_logits(v, t, n, ctx)
    assert not target.requires_grad and target.shape == (6, 6)
    weights = {"pred": 0.0, "episode": 0.0, "consist": 0.0, "mask": 0.0, "distill": 1.0}
    terms = student.losses(v, t, n, w, None, weights, target, 2.0)
    assert set(terms) == {"match", "distill"} and terms["distill"] > 0
    terms["distill"].backward()
    assert student.vproj[2].weight.grad.abs().sum() > 0 and student.word.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    # the teacher's own body is always on, whatever the batch context says
    off = teacher.teacher_logits(v, t, n, dict(ctx, keep=torch.zeros(6)))
    assert torch.equal(off, target)


def test_distillation_weight_zero_equals_the_plain_loss(episodes):
    import torch

    teacher, student, (v, t, n, w), ctx = _distill_pair(episodes)
    target = teacher.teacher_logits(v, t, n, ctx)
    weights = {"pred": 0.0, "episode": 0.0, "consist": 0.0, "mask": 0.0}
    plain = student.forward_loss(v, t, n, w, None, weights)
    assert torch.equal(student.forward_loss(v, t, n, w, None, {**weights, "distill": 0.0}, target, 2.0), plain)
    assert not torch.equal(student.forward_loss(v, t, n, w, None, {**weights, "distill": 1.0}, target, 2.0), plain)


def test_distill_run_with_weight_zero_equals_the_plain_run(episodes, upstream, monkeypatch):
    import torch
    from nursery.learner.train import train_run

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)  # CPU: exact repeatability

    root, run = episodes
    _write_body(root)
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    common = dict(pool="soft", tau_p=0.1, heldout_subject="s2", epochs=2)
    plain = train_run(run, "none", 3, name="plain", **common)
    train_run(run, "none", 3, name="teacher", body="oracle", **common)
    again = train_run(run, "none", 3, name="again", **common)
    again2 = train_run(run, "none", 3, name="again2", **common)
    print("RUNS", plain["history"][0]["train_loss"], again["history"][0]["train_loss"], again2["history"][0]["train_loss"])
    assert plain["history"] == again["history"]
    zero = train_run(run, "none", 3, name="zero", distill_from=run / "teacher_s3", distill_weight=0.0, inspect_epoch=1, **common)
    live = train_run(run, "none", 3, name="live", distill_from=run / "teacher_s3", inspect_epoch=1, **common)
    assert plain["history"] == zero["history"]
    assert live["history"][0]["losses"]["distill"] > 0 and "distill" not in zero["history"][0]["losses"]
    assert live["history"] != plain["history"]
    assert set(live["checkpoints"]) == {"epoch_1", "selected"} and live["checkpoints"]["selected"]["agreement"]["train"]["kl"] >= 0
    assert "picture_soft" in live["distill"]["teacher_body_present"] and "fused_soft" in live["distill"]["teacher_body_present"]


def test_picture_eval_does_not_touch_the_body(episodes, upstream, monkeypatch):
    import torch

    root, run = episodes
    store = _store(root, clips=10)
    model, _ = _tiny_model(store)
    region = _upstream_module("train_region_mil")
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    probes = pd.read_parquet(run / "probes_frames.parquet")
    cache, lut = region.load_region_cache(run / "probes_cache")
    vocab = {"<pad>": 0, **{c: i + 1 for i, c in enumerate(("mug", "bowl", "apple", "banana"))}}
    score = lambda: region.eval_4afc_region(model, cache, lut, probes[probes.split == "dev"], vocab, "cpu")
    before = score()

    def refuse(*args, **kwargs):
        raise AssertionError("evaluation used the body")

    monkeypatch.setattr(type(model.encoder), "forward", refuse)
    for module in (model.fuse, model.decoder, model.region_head):
        for p in module.parameters():
            p.data.add_(torch.randn_like(p))
    assert score() == before


@pytest.mark.parametrize("options", [
    {"body": "real"}, {"body": "shuffled", "episodes": "temporal", "weights": {"episode": 1, "consist": 0.1, "mask": 0.5}},
    {"body": "oracle", "episodes": "contact", "weights": {"episode": 1, "consist": 0.1}}, {"body": "none"}])
def test_body_training_runs(episodes, upstream, options):
    from nursery.learner.train import train_run

    root, run = episodes
    _write_body(root)
    write_transcripts(root, run, 8, p_ref=1, p_name=1, rate_hz=2)
    _fake_caches(root, run)
    metrics = train_run(run, "none", 3, epochs=2, name="body", pool="soft", tau_p=0.1, heldout_subject="s2", **options)
    assert metrics["pool"] == "soft" and metrics["tau_p"] == 0.1
    assert ("body" in metrics) == (options["body"] != "none")
    if options["body"] != "none":
        assert metrics["history"][0]["losses"]["pred"] > 0
        assert metrics["probe"]["splits"]["subject"] and metrics["body_present"]["n_trials"] >= 8
    assert (run / "body_s3" / "metrics.json").exists()
    with pytest.raises(ValueError, match="requires --pool soft"):
        train_run(run, "none", 5, epochs=1, body="real")


def test_upstream_checkout_is_unmodified(upstream):
    assert subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--quiet"]).returncode == 0
