"""Body input for the word learner (docs/learner.md): features, clip boundaries, contact episodes, encoder.

Camera convention: `camera` in body/<video>.npz is camera-to-world (Blender: -Z forward, +Y up), 320x240,
100 deg horizontal FOV. The pinned embedder centre-crops 224 of a 256-short-side resize, so the 4x4 region
grid covers only the central square of the render; "hand in view" uses that square.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WIDTH, HEIGHT, HFOV_DEG = 320, 240, 100.0
FOCAL = (WIDTH / 2) / np.tan(np.radians(HFOV_DEG) / 2)
_SCALE = 256 / min(WIDTH, HEIGHT)
CROP_SIDE = 224 / _SCALE
CROP_X0, CROP_Y0 = (WIDTH - CROP_SIDE) / 2, (HEIGHT - CROP_SIDE) / 2

# SMPL-X joints: 15 head, 18/19 elbows, 20/21 wrists, 25-39 left fingers, 40-54 right fingers.
ARM_JOINTS = (18, 19, 20, 21)
LEFT_HAND = [20] + list(range(25, 40))
RIGHT_HAND = [21] + list(range(40, 55))
UP = np.array([0.0, 0.0, 1.0])  # world up (GRAB is z-up; the head sits +0.6 m above the pelvis along z)

# Hand contact layout (nursery/simulator/build.py): `hand_contact` is np.packbits over [T, 1556] bits =
# the 778 left-hand SMPL-X vertices, then the 778 right-hand ones, in the order of MANO_SMPLX_vertex_ids.pkl
# (hand_contact_vertices = 1556; the last 4 bits are padding).
# Patches: each hand vertex goes to the SMPL-X joint with its largest skinning weight among the wrist and 15
# finger joints (joint order per finger: 1 proximal, 2 middle, 3 distal; fingers index, middle, pinky, ring,
# thumb). Patch 0-4: fingertips (distal joint, per finger); 5-9: finger segments (proximal + middle, per
# finger); 10: palm (wrist joint). 11 patches per hand, left then right = 22. hand_patches.json holds the map.
FINGERS = ("index", "middle", "pinky", "ring", "thumb")
PATCH_NAMES = [f"{f}_tip" for f in FINGERS] + [f"{f}_segments" for f in FINGERS] + ["palm"]
N_PATCH = 2 * len(PATCH_NAMES)
VERTS_PER_HAND = 778
PATCH_FILE = Path(__file__).with_name("hand_patches.json")


def build_hand_patches(model_file, vertex_ids_file):
    """Derive the vertex -> patch map from SMPL-X skinning weights (run once; the result is hand_patches.json)."""
    import pickle

    weights = np.load(model_file)["weights"]
    with open(vertex_ids_file, "rb") as stream:
        ids = pickle.load(stream)
    out = {"names": PATCH_NAMES}
    for side, wrist, first in (("left", 20, 25), ("right", 21, 40)):
        joint = weights[ids[f"{side}_hand"]][:, [wrist] + list(range(first, first + 15))].argmax(1)
        finger, segment = (joint - 1) // 3, (joint - 1) % 3
        out[side] = np.where(joint == 0, 10, np.where(segment == 2, finger, 5 + finger)).tolist()
    return out


def patch_matrix():
    """[1556, 22] matrix averaging the contact bits of each patch's vertices."""
    patches = json.loads(PATCH_FILE.read_text())
    matrix = np.zeros((2 * VERTS_PER_HAND, N_PATCH), np.float32)
    for h, side in enumerate(("left", "right")):
        for v, p in enumerate(patches[side]):
            matrix[h * VERTS_PER_HAND + v, h * len(PATCH_NAMES) + p] = 1
    return matrix / matrix.sum(0, keepdims=True).clip(min=1)


def project_points(points, camera):
    """World points [N,3] -> pixels [N,2] and in-front mask [N]. `camera` [N,4,4] is camera-to-world."""
    inverse = np.linalg.inv(camera)
    local = np.einsum("nij,nj->ni", inverse[:, :3, :3], points) + inverse[:, :3, 3]
    front = local[:, 2] < -1e-3
    depth = np.where(front, -local[:, 2], 1.0)
    pixels = np.stack([WIDTH / 2 + FOCAL * local[:, 0] / depth, HEIGHT / 2 - FOCAL * local[:, 1] / depth], 1)
    return pixels, front


def session_features(data, matrix=None):
    """One body npz -> raw per-second arrays (no normalisation; clip-boundary diffs are made in `load_store`).

    contact [T,22] fraction of each patch's vertices touching; fingers [T,90] local (palm-relative) finger
    rotations, both hands; arm [T,15] elbow/wrist positions (left/right elbow, left/right wrist) in the head
    camera frame (full camera rotation) plus gravity direction in that frame; aperture [T,2] mean finger
    flexion per hand; in_view [T,2] hand centre inside the DINO crop; hand [T] contacting hand (-1 none, 0 left,
    1 right = the one with more contact vertices).
    Not used: object_pose, transl, global_orient, body_pose, labels.
    """
    matrix = patch_matrix() if matrix is None else matrix
    bits = np.unpackbits(data["hand_contact"], axis=1)[:, :2 * VERTS_PER_HAND].astype(np.float32)
    counts = np.stack([bits[:, :VERTS_PER_HAND].sum(1), bits[:, VERTS_PER_HAND:].sum(1)], 1)
    camera, joints = data["camera"].astype(np.float64), data["joints"].astype(np.float64)
    inverse = np.linalg.inv(camera)
    in_camera = np.einsum("tij,tkj->tki", inverse[:, :3, :3], joints[:, list(ARM_JOINTS)]) + inverse[:, None, :3, 3]
    gravity = np.einsum("tji,j->ti", camera[:, :3, :3], UP)  # world up in the camera frame
    pose = [data["left_hand_pose"], data["right_hand_pose"]]
    aperture = np.stack([np.linalg.norm(p.reshape(-1, 15, 3), axis=-1).mean(1) for p in pose], 1)
    in_view = []
    for hand in (LEFT_HAND, RIGHT_HAND):
        pixels, front = project_points(joints[:, hand].mean(1), camera)
        inside = ((pixels >= [CROP_X0, CROP_Y0]) & (pixels <= [CROP_X0 + CROP_SIDE, CROP_Y0 + CROP_SIDE])).all(1)
        in_view.append(front & inside)
    return {"contact": (bits @ matrix).astype(np.float32),
            "fingers": np.concatenate(pose, 1).astype(np.float32),
            "arm": np.concatenate([in_camera.reshape(len(joints), -1), gravity], 1).astype(np.float32),
            "aperture": aperture.astype(np.float32), "in_view": np.stack(in_view, 1).astype(np.float32),
            "hand": np.where(counts.max(1) > 0, counts.argmax(1), -1)}


def clip_index(frames, video_ids, steps):
    """frames.parquet -> clip id [S,steps] (-1 = no frame), held category index [S,steps] (-1 none), clip names, categories.

    Clip ids count the clip changes inside each session, so two consecutive runs of the same clip file are
    still two clips. This is the only clip-boundary source: the simulator writes one `clip` per second.
    """
    frames = frames[frames.video_id.isin(video_ids)].sort_values(["video_id", "frame_idx"])
    categories = sorted(frames.held_category.dropna().unique())
    row = {v: i for i, v in enumerate(video_ids)}
    clip = np.full((len(video_ids), steps), -1, np.int64)
    held = np.full((len(video_ids), steps), -1, np.int64)
    names = [[] for _ in video_ids]
    for video_id, group in frames.groupby("video_id", sort=False):
        s, count, previous = row[video_id], -1, None
        for frame, name, category in zip(group.frame_idx, group["clip"], group.held_category):
            if previous is None or name != previous or frame != last + 1:
                count, previous = count + 1, name
                names[s].append(name)
            clip[s, frame], last = count, frame
            if isinstance(category, str):
                held[s, frame] = categories.index(category)
    return clip, held, names, categories


def contact_episodes(hand, clip, min_len=2, first_id=0):
    """Contact episodes of one session. hand [T] in {-1,0,1}, clip [T] (-1 absent) -> episode id [T] (-1 none).

    An episode is a maximal run of consecutive frames, inside one clip, in which the same hand stays in
    contact; it ends at a clip change (which includes any change of the held object) or a missing frame.
    Runs shorter than `min_len` seconds are dropped (1 Hz makes a 1 s run indistinguishable from a flicker).
    """
    out = np.full(len(hand), -1, np.int64)
    start, next_id = 0, first_id
    for t in range(1, len(hand) + 1):
        same = t < len(hand) and hand[t] == hand[start] and clip[t] == clip[start]
        if not same:
            if hand[start] >= 0 and clip[start] >= 0 and t - start >= min_len:
                out[start:t] = next_id
                next_id += 1
            start = t
    return out


def temporal_blocks(clip, lengths, rng, min_len=2, first_id=0):
    """Control for contact episodes: cut each clip run into consecutive blocks whose durations are drawn from
    `lengths` (the contact-episode length distribution). A remainder shorter than `min_len` joins the last block."""
    out = np.full(len(clip), -1, np.int64)
    next_id, t = first_id, 0
    while t < len(clip):
        if clip[t] < 0:
            t += 1
            continue
        end = t
        while end < len(clip) and clip[end] == clip[t]:
            end += 1
        if end - t >= min_len:
            cuts, at = [], t
            while at < end:
                at = min(at + int(rng.choice(lengths)), end)
                cuts.append(at)
            if len(cuts) > 1 and cuts[-1] - cuts[-2] < min_len:
                cuts.pop(-2)
            at = t
            for cut in cuts:
                out[at:cut] = next_id
                next_id, at = next_id + 1, cut
        t = end
    return out


def load_store(episodes, frames, video_ids, heldout_videos, steps=120, exclude=None, physical=None):
    """Per-session body arrays [S,steps,...] indexed by frame_idx, normalised on sessions not in `heldout_videos`.

    heldout_videos: video ids whose sessions are excluded from the normalisation fit (the held-out subject).
    exclude: optional bool [S,steps] of further seconds kept out of the fit (dev/test clips).
    Groups (encoder inputs): contact [22], fingers [90], arm [27] = positions 12 + 1-step differences 12 +
    gravity 3. Differences are zero at the first second of a clip. `present` [S,T] marks seconds that exist.
    physical: optional directory of per-clip `<video_id>.npz` (nursery/simulator/physical.py). Adds groups load [10]
    (per hand mean force xyz, |F|, spread) and skin [4] (per hand skin dT, contact area), standardised on the same
    training seconds as fingers/arm, zero on absent seconds.
    """
    video_ids = list(video_ids)
    matrix = patch_matrix()
    clip, held, names, categories = clip_index(frames, video_ids, steps)
    raw = {}
    for s, video_id in enumerate(video_ids):
        with np.load(Path(episodes) / "body" / f"{video_id}.npz") as data:
            feats = session_features(data, matrix)
            index = data["frame_idx"]
        for key, value in feats.items():
            raw.setdefault(key, np.zeros((len(video_ids), steps) + value.shape[1:], value.dtype if key != "hand" else np.int64))
            raw[key][s, index] = value
    raw["hand"][clip < 0] = -1
    present = clip >= 0
    pos = raw["arm"][..., :12]
    first = np.ones_like(present)
    first[:, 1:] = clip[:, 1:] != clip[:, :-1]
    diff = np.zeros_like(pos)
    diff[:, 1:] = pos[:, 1:] - pos[:, :-1]
    diff[first | ~present] = 0
    arm = np.concatenate([pos, diff, raw["arm"][..., 12:]], -1)
    train = ~np.isin(video_ids, list(heldout_videos))[:, None] & present
    if exclude is not None:
        train &= ~exclude
    stats = {}
    for key, value in (("fingers", raw["fingers"]), ("arm", arm), ("aperture", raw["aperture"])):
        stats[key] = (value[train].mean(0), value[train].std(0) + 1e-6)
    norm = lambda key, value: ((value - stats[key][0]) / stats[key][1] * present[..., None]).astype(np.float32)
    if physical is not None:
        phys = {"load": np.zeros((len(video_ids), steps, 10), np.float32), "skin": np.zeros((len(video_ids), steps, 4), np.float32)}
        for s, video_id in enumerate(video_ids):
            with np.load(Path(physical) / f"{video_id}.npz") as data:
                index = data["frame_idx"]
                phys["load"][s, index] = data["load"].reshape(len(index), 10)
                phys["skin"][s, index] = np.concatenate([data["skin_dT"], data["contact_area"]], 1)
        for key, value in phys.items():
            stats[key] = (value[train].mean(0), value[train].std(0) + 1e-6)
        phys = {key: norm(key, value) for key, value in phys.items()}
    arm_norm = norm("arm", arm)
    arm_norm[first | ~present, 12:24] = 0  # no difference at the first second of a clip
    contact = (raw["contact"] * present[..., None]).astype(np.float32)
    binary = (contact > 0).astype(np.float32)
    nxt = np.zeros_like(binary)
    nxt[:, :-1] = binary[:, 1:] - binary[:, :-1]
    next_valid = np.zeros(present.shape, bool)
    next_valid[:, :-1] = present[:, :-1] & present[:, 1:] & (clip[:, :-1] == clip[:, 1:])
    nxt[~next_valid] = 0
    store = {
        "video_ids": video_ids, "row": {v: s for s, v in enumerate(video_ids)}, "steps": steps,
        "present": present, "clip": clip, "held": held, "categories": categories, "clip_names": names,
        "hand": raw["hand"], "next_valid": next_valid,
        "groups": {"contact": contact, "fingers": norm("fingers", raw["fingers"]), "arm": arm_norm},
        "targets": {"contact": binary, "change": nxt, "aperture": norm("aperture", raw["aperture"]),
                    "in_view": (raw["in_view"] * present[..., None]).astype(np.float32)},
        "stats": {k: [v[0].tolist(), v[1].tolist()] for k, v in stats.items()},
    }
    if physical is not None:
        store["groups"].update(phys)
    return store


def physical_shuffle_store(store, part, seed):
    """Control: swap the physical groups (load, skin) between source clips of the same subject and partition.

    Each source clip name inside a (subject, partition) group gets another clip of that group (a cyclic derangement
    after a random permutation; a group with one clip keeps its own). Each run of seconds of a clip takes a random
    run of its assigned clip; position k takes position k mod the run's length. Contact, fingers, arm, images and
    clip structure stay put. part [S,T] is the clip partition per second. Returns the new store and counts.
    """
    rng = np.random.default_rng(seed)
    runs = {}  # (subject, partition, clip name) -> [(session, start, length)]
    for s in range(len(store["video_ids"])):
        clip = store["clip"][s]
        t = 0
        while t < store["steps"]:
            end = t
            while end < store["steps"] and clip[end] == clip[t] and clip[t] >= 0:
                end += 1
            if end > t:
                runs.setdefault((store["subject"][s], int(part[s, t]), store["clip_names"][s][clip[t]]), []).append((s, t, end - t))
                t = end
            else:
                t += 1
    assigned = {}
    for subject, partition in sorted({k[:2] for k in runs}):
        names = sorted(k[2] for k in runs if k[:2] == (subject, partition))
        order = rng.permutation(len(names))
        shifted = np.roll(order, 1) if len(names) > 1 else order
        for dest, src in zip(order, shifted):
            assigned[(subject, partition, names[dest])] = (subject, partition, names[src])
    groups = {k: v.copy() for k, v in store["groups"].items()}
    for key, source in assigned.items():
        pool = runs[source]
        for s, t, length in runs[key]:
            s2, t2, length2 = pool[int(rng.integers(len(pool)))]
            for name in ("load", "skin"):
                groups[name][s, t:t + length] = store["groups"][name][s2, t2 + np.arange(length) % length2]
    own = sum(k == v for k, v in assigned.items())
    return {**store, "groups": groups}, {"clips": len(assigned), "kept_own": int(own)}


def oracle_store(store):
    """Same pipeline, but the 'body' is the true held-object identity (one-hot per second; zeros when nothing is held)."""
    held = store["held"]
    onehot = np.zeros(held.shape + (len(store["categories"]),), np.float32)
    s, t = np.nonzero(held >= 0)
    onehot[s, t, held[s, t]] = 1
    return {**store, "groups": {"identity": onehot}, "targets": {"identity": held}}


def shuffle_store(store, seed, exclude=()):
    """Swap whole clip-length body sequences between sessions/clips, keeping marginals and boundaries.

    Each run of seconds from one clip is a unit; units of equal length are cyclically permuted (a derangement
    after a random shuffle), so every swapped unit lands in a different clip. Images, speech, clip structure and
    held labels stay in place. Sessions in `exclude` (the held-out subject) are untouched.
    Returns the new store and {"units", "moved"}.
    """
    rng = np.random.default_rng(seed)
    units = {}
    for s, video_id in enumerate(store["video_ids"]):
        if video_id in exclude:
            continue
        clip = store["clip"][s]
        t = 0
        while t < store["steps"]:
            end = t
            while end < store["steps"] and clip[end] == clip[t] and clip[t] >= 0:
                end += 1
            if end > t:
                units.setdefault(end - t, []).append((s, t))
                t = end
            else:
                t += 1
    source = {}  # destination (s, t, length) -> source (s, t)
    for length, group in units.items():
        order = rng.permutation(len(group))
        if len(group) > 1:
            shifted = np.roll(order, int(rng.integers(1, len(group))))
            for dest, src in zip(order, shifted):
                source[group[dest] + (length,)] = group[src]
    def move(array):
        out = array.copy()
        for (s, t, length), (s2, t2) in source.items():
            out[s, t:t + length] = array[s2, t2:t2 + length]
        return out
    moved = {
        **store, "hand": move(store["hand"]),
        "groups": {k: move(v) for k, v in store["groups"].items()},
        "targets": {k: move(v) for k, v in store["targets"].items()},
        "next_valid": move(store["next_valid"]),
    }
    return moved, {"units": sum(len(g) for g in units.values()), "moved": len(source)}


def build_episodes(store, mode, min_len=2, seed=0):
    """Episode ids [S,T] (-1 none) for the pairs' bags: contact episodes, temporal control blocks, or none."""
    ids = np.full(store["clip"].shape, -1, np.int64)
    if mode == "none":
        return ids
    next_id = 0
    real = [contact_episodes(store["hand"][s], store["clip"][s], min_len) for s in range(len(ids))]
    lengths = np.array([n for e in real for n in np.bincount(e[e >= 0])[np.unique(e[e >= 0])]])
    if mode == "contact":
        for s, episode in enumerate(real):
            ids[s] = np.where(episode >= 0, episode + next_id, -1)
            next_id += episode.max() + 1 if (episode >= 0).any() else 0
        return ids
    if mode != "temporal":
        raise ValueError("episodes must be contact, temporal or none")
    rng = np.random.default_rng(seed)
    for s in range(len(ids)):
        block = temporal_blocks(store["clip"][s], lengths, rng, min_len, next_id)
        ids[s] = block
        next_id = max(next_id, block.max() + 1)
    # Match the share of seconds that sit in a bag to the contact-episode share, by dropping random blocks.
    target = sum(int((e >= 0).sum()) for e in real)
    blocks = np.unique(ids[ids >= 0])
    covered = int((ids >= 0).sum())
    for block in rng.permutation(blocks):
        if covered <= target:
            break
        covered -= int((ids == block).sum())
        ids[ids == block] = -1
    return ids


def bag_frames(episode_ids, cap):
    """Per (session, frame): the frames of its episode, at most `cap`, subsampled evenly. -> [S,T,cap] (-1 pad)."""
    out = np.full(episode_ids.shape + (cap,), -1, np.int64)
    for s in range(len(episode_ids)):
        for e in np.unique(episode_ids[s]):
            if e < 0:
                continue
            frames = np.flatnonzero(episode_ids[s] == e)
            if len(frames) > cap:
                frames = frames[np.linspace(0, len(frames) - 1, cap).round().astype(int)]
            out[s, episode_ids[s] == e, :len(frames)] = frames
    return out


def window_steps(store_clip, session, centre, half):
    """Window of 2*half+1 seconds around `centre` -> step index [B,K], clip id [B,K] (-2 outside the session/missing).

    The centre's clip id is returned too, so callers can see which steps belong to the centre's clip.
    """
    steps = centre[:, None] + np.arange(-half, half + 1)
    inside = (steps >= 0) & (steps < store_clip.shape[1])
    clipped = np.clip(steps, 0, store_clip.shape[1] - 1)
    clip = np.where(inside, store_clip[session[:, None], clipped], -2)
    clip = np.where(clip < 0, -2, clip)
    return clipped, clip


class BodyEncoder(nn.Module):
    """One MLP per group, then a bidirectional GRU (2 x 64 = a 128-d latent at the window centre).

    The recurrent state is reset at every clip boundary, in both directions, so nothing from another clip (or
    from outside the session) reaches the centre. Steps listed in `missing` carry zeros plus a missing flag.
    """

    def __init__(self, dims, width=64, hidden=64):
        super().__init__()
        self.names = tuple(dims)
        self.mlps = nn.ModuleDict({k: nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, width))
                                   for k, d in dims.items()})
        self.fwd = nn.GRUCell(width * len(dims) + 1, hidden)
        self.bwd = nn.GRUCell(width * len(dims) + 1, hidden)
        self.hidden = hidden

    def forward(self, groups, missing, seg):
        """groups {name: [B,K,d]}, missing [B,K] float (1 = no input), seg [B,K] clip ids (-2 = outside).
        Returns per-step states [B,K,2*hidden]."""
        keep = (1 - missing).unsqueeze(-1)
        x = torch.cat([self.mlps[k](groups[k] * keep) * keep for k in self.names] + [missing.unsqueeze(-1)], -1)
        B, K, _ = x.shape
        h = x.new_zeros(B, self.hidden)
        fwd = []
        for k in range(K):
            if k:
                h = h * (seg[:, k] == seg[:, k - 1]).float().unsqueeze(-1)
            h = self.fwd(x[:, k], h)
            fwd.append(h)
        h = x.new_zeros(B, self.hidden)
        bwd = [None] * K
        for k in reversed(range(K)):
            if k < K - 1:
                h = h * (seg[:, k] == seg[:, k + 1]).float().unsqueeze(-1)
            h = self.bwd(x[:, k], h)
            bwd[k] = h
        return torch.cat([torch.stack(fwd, 1), torch.stack(bwd, 1)], -1)
