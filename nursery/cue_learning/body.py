"""Body input for the word learner: training-only, body-guided region selection (step 3)."""

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _upstream_module

SIGNALS = ("contact", "hand", "joints")
WIDTH, HEIGHT, HFOV_DEG = 320, 240, 100.0
FOCAL = (WIDTH / 2) / np.tan(np.radians(HFOV_DEG) / 2)
# The pinned embedder uses DINOv2's processor: resize the short side to 256, centre-crop 224.
# The 4x4 region grid therefore covers only the central square of the render.
_SCALE = 256 / min(WIDTH, HEIGHT)
CROP_SIDE = 224 / _SCALE
CROP_X0, CROP_Y0 = (WIDTH - CROP_SIDE) / 2, (HEIGHT - CROP_SIDE) / 2
CELL = CROP_SIDE / 4
GRID_CENTRES = np.array([[CROP_X0 + (c + 0.5) * CELL, CROP_Y0 + (r + 0.5) * CELL]
                         for r in range(4) for c in range(4)], dtype=np.float32)  # row-major, pixels
D_MAX, D_NEUTRAL = 3.0, 1.5  # distances in cell units; no-hand frames get the neutral value
LEFT_HAND = [20] + list(range(25, 40))
RIGHT_HAND = [21] + list(range(40, 55))
VERTS_PER_HAND = 778
AREA_EDGES = (0.058, 0.129)  # tertiles of nonzero contact fractions
GATE_MAX, SIGMA, SELECT_TAU = 3.0, 1.0, 0.05
INPUT_NOISE, SIGNAL_DROP = 1.0, 0.3
SIGNAL_DIMS = {"contact": 4, "hand": 2, "joints": 6}  # standardised inputs; noise is added to all in training


def project_points(points, camera):
    """World points [N,3] -> pixels [N,2] and in-front mask [N]. `camera` is camera-to-world (Blender: -Z forward, +Y up)."""
    inverse = np.linalg.inv(camera)
    local = np.einsum("nij,nj->ni", inverse[:, :3, :3], points) + inverse[:, :3, 3]
    front = local[:, 2] < -1e-3
    depth = np.where(front, -local[:, 2], 1.0)
    pixels = np.stack([WIDTH / 2 + FOCAL * local[:, 0] / depth, HEIGHT / 2 - FOCAL * local[:, 1] / depth], 1)
    return pixels, front


def region_distance(pixels, valid):
    """Pixels [N,2] -> [N,17] distances (cell units) to CLS + 16 region centres; invalid rows get the neutral value."""
    d = np.linalg.norm(pixels[:, None, :] - GRID_CENTRES[None], axis=-1) / CELL
    d = np.minimum(d, D_MAX)
    full = np.concatenate([d.mean(1, keepdims=True), d], 1)  # CLS: the mean grid distance
    return np.where(valid[:, None], full, D_NEUTRAL).astype(np.float32)


def project_hand(joints, camera, active):
    """Joints [N,55,3], camera [N,4,4], active [N] in {-1 none, 0 left, 1 right} -> distances [N,17], hand pixel [N,2]."""
    points = np.stack([joints[:, LEFT_HAND].mean(1), joints[:, RIGHT_HAND].mean(1)], 1)
    point = points[np.arange(len(points)), np.maximum(active, 0)]
    pixels, front = project_points(point, camera)
    valid = (active >= 0) & front
    return region_distance(pixels, valid), pixels


def cell_distance(cells, count):
    """Held-object cells (indices on the full-image 4x4 grid) -> [17] region distances, 0 inside the object's cells."""
    if not len(cells):
        return np.full(count, D_NEUTRAL, dtype=np.float32)
    boxes = [(c % 4 * WIDTH / 4, c // 4 * HEIGHT / 4) for c in cells]
    d = np.zeros(16, dtype=np.float32)
    for k, (cx, cy) in enumerate(GRID_CENTRES):
        gaps = [np.hypot(max(x0 - cx, 0, cx - (x0 + WIDTH / 4)), max(y0 - cy, 0, cy - (y0 + HEIGHT / 4))) for x0, y0 in boxes]
        d[k] = min(gaps) / CELL
    d = np.minimum(d, D_MAX)
    return np.concatenate([[d.mean()], d]).astype(np.float32)


def _contact(bits):
    """Packed vertex bits [T,195] -> 4 coarse scalars (any, area level) per hand, and the active hand [T]."""
    bits = np.unpackbits(bits, axis=1)[:, :2 * VERTS_PER_HAND]
    counts = np.stack([bits[:, :VERTS_PER_HAND].sum(1), bits[:, VERTS_PER_HAND:].sum(1)], 1)
    fraction = counts / VERTS_PER_HAND
    level = (counts > 0).astype(int) + (fraction >= AREA_EDGES[0]) + (fraction >= AREA_EDGES[1])
    scalars = np.stack([counts[:, 0] > 0, level[:, 0] / 3, counts[:, 1] > 0, level[:, 1] / 3], 1).astype(np.float32)
    active = np.where(counts.max(1) > 0, counts.argmax(1), -1)
    return scalars, active


def _hand_closure(left, right):
    """Pose [T,45] per hand -> mean finger-joint rotation per hand [T,2] (grip closure; the full pose leaks object identity)."""
    return np.stack([np.linalg.norm(p.reshape(-1, 15, 3), axis=-1).mean(1) for p in (left, right)], 1).astype(np.float32)


def _joints(joints):
    """Wrists relative to the pelvis, yaw-aligned (hip axis -> +x) -> [T,6]; other joints leak the clip."""
    j = joints[:, :22] - joints[:, :1]
    lateral = j[:, 1, :2] - j[:, 2, :2]
    angle = -np.arctan2(lateral[:, 1], lateral[:, 0])
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = cos[:, None] * j[..., 0] - sin[:, None] * j[..., 1], sin[:, None] * j[..., 0] + cos[:, None] * j[..., 1]
    return np.stack([x, y, j[..., 2]], -1)[:, [20, 21]].reshape(len(j), 6).astype(np.float32)


def load_body(episodes, video_ids):
    """Per-session body arrays from `<episodes>/body/<video_id>.npz`, indexed by frame_idx.

    Returns {"video_ids", "row": {video_id: s}, "frame_row": [S, max_frame+1] (-1 = absent), and
    per-row arrays contact [S,T,4], hand [S,T,2], joints [S,T,6], dist [S,T,17], active [S,T]}.
    Hand closure and wrist positions are standardised with statistics over the sessions loaded.
    """
    from pathlib import Path

    video_ids = list(video_ids)
    parts = {key: [] for key in ("contact", "hand", "joints", "dist", "active", "frame_idx")}
    for video_id in video_ids:
        with np.load(Path(episodes) / "body" / f"{video_id}.npz") as data:
            contact, active = _contact(data["hand_contact"])
            parts["contact"].append(contact)
            parts["hand"].append(_hand_closure(data["left_hand_pose"], data["right_hand_pose"]))
            parts["joints"].append(_joints(data["joints"]))
            parts["dist"].append(project_hand(data["joints"], data["camera"], active)[0])
            parts["active"].append(active)
            parts["frame_idx"].append(data["frame_idx"])
    body = {key: np.stack(value) for key, value in parts.items() if key != "frame_idx"}
    for key in ("hand", "joints"):
        mean, std = body[key].mean((0, 1)), body[key].std((0, 1)) + 1e-6
        body[key] = ((body[key] - mean) / std).astype(np.float32)
    frame_row = np.full((len(video_ids), max(int(f.max()) for f in parts["frame_idx"]) + 1), -1, dtype=np.int64)
    for s, frames in enumerate(parts["frame_idx"]):
        frame_row[s, frames] = np.arange(len(frames))
    return {**body, "video_ids": video_ids, "row": {v: s for s, v in enumerate(video_ids)}, "frame_row": frame_row}


def window_frames(video_id, frame_idx, lut_has, window):
    """Frames FramePairs uses for a pair: fi-window..fi+window present in the cache, padded by repeating the first."""
    frames = [f for f in range(frame_idx - window, frame_idx + window + 1) if lut_has(video_id, f)]
    return frames + [frames[0]] * (2 * window + 1 - len(frames))


def pair_arrays(body, manifest, lut_has, window, regions=17):
    """Per-pair body inputs aligned to FramePairs: dist [N, Fmax*regions] and window-mean contact/hand/joints."""
    n, fmax = len(manifest), 2 * window + 1
    out = {"dist": np.zeros((n, fmax * regions), np.float32), **{k: np.zeros((n, body[k].shape[-1]), np.float32)
                                                               for k in SIGNALS}}
    for i, (video_id, frame_idx) in enumerate(zip(manifest.video_id, manifest.frame_idx)):
        frames = window_frames(video_id, int(frame_idx), lut_has, window)
        s = body["row"][video_id]
        rows = body["frame_row"][s, frames]
        out["dist"][i] = body["dist"][s, rows].reshape(-1)
        for k in SIGNALS:
            out[k][i] = body[k][s, np.unique(rows)].mean(0)
    return out


def oracle_distances(manifest, frames, lut_has, window, regions=17):
    """[N, Fmax*regions] distances to the true held object's visible cells (neutral when not held or not visible)."""
    import json

    held = {(r.video_id, r.frame_idx): r for r in frames.itertuples() if isinstance(r.held_category, str)}
    out = np.zeros((len(manifest), (2 * window + 1) * regions), np.float32)
    for i, (video_id, frame_idx) in enumerate(zip(manifest.video_id, manifest.frame_idx)):
        parts = []
        for f in window_frames(video_id, int(frame_idx), lut_has, window):
            row = held.get((video_id, f))
            cells = []
            if row is not None:
                cells = next((o["cells"] for o in json.loads(row.objects) if o["category"] == row.held_category), [])
            parts.append(cell_distance(cells, regions))
        out[i] = np.concatenate(parts)
    return out


def drop_masks(batch, signals, drop, generator, signal_drop=SIGNAL_DROP, noise_std=INPUT_NOISE):
    """Per-pair whole-body mask [B], per-signal keep mask [B,S] and input noise [B,12] from a CPU generator."""
    on = (torch.rand(batch, generator=generator) >= drop).float()
    keep = (torch.rand(batch, max(len(signals), 1), generator=generator) >= signal_drop).float()
    return on, keep, torch.randn(batch, sum(SIGNAL_DIMS.values()), generator=generator) * noise_std


class BodyEncoders(nn.Module):
    """Sub-encoders (contact, hand pose, joints) summed into a trust gate g in [0, GATE_MAX]; no signals -> g = 1."""

    def __init__(self, signals, width=16):
        super().__init__()
        self.signals = tuple(signals)
        mlp = lambda inp, hidden: nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, width))
        self.enc = nn.ModuleDict()
        if "contact" in self.signals:
            self.enc["contact"] = mlp(4, 16)
        if "hand" in self.signals:
            self.enc["hand"] = mlp(2, 16)
        if "joints" in self.signals:
            self.enc["joints"] = mlp(6, 16)
        if self.signals:
            self.head = nn.Linear(width, 1)

    def forward(self, feats, keep, noise):
        if not self.signals:
            return torch.ones(len(keep), device=keep.device)
        total = 0
        for k, name in enumerate(self.signals):
            x = feats[name]
            if self.training:
                start = sum(SIGNAL_DIMS[n] for n in SIGNALS[:SIGNALS.index(name)])
                x = x + noise[:, start:start + SIGNAL_DIMS[name]]
            total = total + self.enc[name](x) * keep[:, k:k + 1]
        return GATE_MAX * torch.sigmoid(self.head(total)).squeeze(-1)


@lru_cache(maxsize=None)
def _model_class():
    region = _upstream_module("train_region_mil")

    class BodyRegionMIL(region.RegionMIL):
        """RegionMIL whose training selection is biased toward the active hand; evaluation is unchanged."""

        def __init__(self, vocab, signals=SIGNALS, **kwargs):
            super().__init__(vocab, **kwargs)
            with torch.random.fork_rng(devices=[]):  # leave the default RNG stream exactly as video-only training sees it
                torch.manual_seed(0)
                self.body = BodyEncoders(signals)

        def sims(self, R, T, bias=None):
            if bias is None:
                return super().sims(R, T)
            s = torch.einsum("brd,md->brm", R, T)
            biased = s.detach() + bias.unsqueeze(-1)
            hard = s.gather(1, biased.argmax(1, keepdim=True)).squeeze(1)
            # Straight-through: the forward value is the raw sim at the selected region; the gate
            # (and nothing else) gets gradient through a soft selection.
            weights = torch.softmax(biased / SELECT_TAU, dim=1)
            soft = (weights * s.detach()).sum(1)
            return hard + (soft - soft.detach())

        def gate(self, feats, on, keep, noise):
            return self.body(feats, keep, noise) * on

        def forward_loss(self, v, t, n, w, dist=None, feats=None, on=None, keep=None, noise=None):
            if dist is None:
                return super().forward_loss(v, t, n, w)
            R, T = self.enc_regions(v), self.enc_text(t, n)
            bias = -self.gate(feats, on, keep, noise).unsqueeze(1) * dist / SIGMA
            logits = self.logit_scale.clamp(max=np.log(100)).exp() * self.sims(R, T, bias)
            lab = torch.arange(len(v), device=v.device)
            li = F.cross_entropy(logits, lab, reduction="none")
            lt = F.cross_entropy(logits.t(), lab, reduction="none")
            return (0.5 * w * (li + lt)).sum() / w.sum().clamp(min=1e-6)

    return BodyRegionMIL


def BodyRegionMIL(vocab, signals=SIGNALS, **kwargs):
    """Build the model (the class subclasses the pinned upstream RegionMIL, imported lazily)."""
    return _model_class()(vocab, signals, **kwargs)
