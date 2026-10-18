"""Simulated wrist load and skin temperature change per second per hand (step 4.11).

Per held object (instance id, or "grab:<obj_name>" for native GRAB objects) one material and one mass are drawn
from `materials.json`, seeded by a stable hash of the key. They never reach the learner; only these two signals do.

Simplifications:
- Wrist load is felt force F = m (g - a) from the GRAB object trajectory (120 Hz), not a contact-force solve.
  The trajectory is smoothed with a Savitzky-Golay filter (15 frames, cubic) before the second derivative.
  Mean/std are over the raw frames within +-0.5 s of the second's source frame. Frames where the object is not
  lifted (z <= resting z + 1 cm) count as zero force. The force is split between hands by contacted vertices.
- Skin temperature change is a one-line contact-temperature formula (effusivity weighted) at the touched skin,
  not scaled by area, with first-order dynamics at 1 s steps. Contact area (contacted vertices / 100, capped at
  1) is a separate channel. It is not a heat-transfer simulation.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

MATERIALS = Path(__file__).parent / "materials.json"
GRAVITY = np.array([0.0, 0.0, -9.81])
LIFT_M = 0.01
SG_WINDOW, SG_ORDER = 15, 3
WEBER_SIGMA = 0.10
SKIN_NOISE_C = 0.3
TAU_CONTACT_S, TAU_RELEASE_S = 1.0, 5.0
FULL_CONTACT_VERTICES = 100
HAND_VERTICES = 778


def load_materials(path=MATERIALS):
    return json.loads(Path(path).read_text())


def stable_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def draw_properties(table, word, key):
    """(material, mass in kg) for one instance key; deterministic, log-uniform mass in the word's range."""
    entry = table["words"][word]
    rng = np.random.default_rng(stable_seed(f"physical:{key}"))
    names = sorted(entry["materials"])
    probs = np.array([entry["materials"][n] for n in names], float)
    material = names[rng.choice(len(names), p=probs / probs.sum())]
    low, high = entry["mass_g"]
    grams = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return material, grams / 1000.0


def object_acceleration(transl, fps):
    """Smoothed object acceleration [n,3] in m/s^2: Savitzky-Golay (15 frames, cubic) second derivative.

    Interior frames use the usual convolution; the first/last half-window use a local polynomial fit of the
    edge window. (scipy is not installed in the project environment, so this is plain numpy.)
    """
    n = len(transl)
    window = min(SG_WINDOW, n - (1 - n % 2))
    half, order = window // 2, min(SG_ORDER, window - 1)
    offsets = np.arange(-half, half + 1) / fps
    # second derivative at the window centre = 2 * coefficient of t^2 in the least-squares polynomial
    weights = 2.0 * np.linalg.pinv(np.vander(offsets, order + 1, increasing=True))[2]
    out = np.zeros((n, 3))
    for axis in range(3):
        out[half:n - half, axis] = np.convolve(transl[:, axis], weights[::-1], mode="valid")
        for edge, centre in ((slice(0, half), np.arange(half)), (slice(n - half, n), np.arange(n - half, n))):
            lo = 0 if centre[0] == 0 else n - window
            t = np.arange(lo, lo + window) / fps
            coefficients = np.polyfit(t, transl[lo:lo + window, axis], order)
            out[edge, axis] = np.polyval(np.polyder(coefficients, 2), centre / fps)
    return out


def window_force(acc, z, rest_z, mass):
    """Felt force per raw frame, world frame [n,3]: m (g - a) where the object is lifted, else zero."""
    force = mass * (GRAVITY[None] - acc)
    return force * (z > rest_z + LIFT_M)[:, None]


def load_stats(force_world, rotation):
    """Window of world-frame forces [k,3] + camera-to-world rotation [3,3] -> (Fx,Fy,Fz in camera, |F| mean, |F| std)."""
    magnitude = np.linalg.norm(force_world, axis=1)
    return np.concatenate([rotation.T @ force_world.mean(0), [magnitude.mean(), magnitude.std()]])


def contact_area(vertices):
    """Contact area value in [0, 1]: contacted vertices / 100, capped."""
    return min(1.0, vertices / FULL_CONTACT_VERTICES)


def contact_target(table, material):
    """Target skin temperature change (C) at the touched skin of a hand holding a `material` object."""
    skin, room = table["skin"], table["room_temperature_c"]
    e_skin, e_obj = skin["effusivity"], table["materials"][material]
    contact = (e_skin * skin["temperature_c"] + e_obj * room) / (e_skin + e_obj)
    return contact - skin["temperature_c"]


def skin_dynamics(target, touching, new_clip):
    """First-order skin dT [T] for one hand; 1 s steps; reset to 0 where `new_clip`."""
    out, state = np.zeros(len(target)), 0.0
    for t in range(len(target)):
        if new_clip[t]:
            state = 0.0
        goal = target[t] if touching[t] else 0.0
        tau = TAU_CONTACT_S if touching[t] else TAU_RELEASE_S
        state += (goal - state) * (1.0 - np.exp(-1.0 / tau))
        out[t] = state
    return out


class _Clips:
    """Raw GRAB clip data needed here, cached per clip file."""

    def __init__(self, root, swaps, table, vocabulary, key_by="instance"):
        from .swaps import assignment

        self.root, self.swaps, self.table, self.vocabulary = Path(root), swaps, table, vocabulary
        self.assignment, self.cache, self.key_by = assignment, {}, key_by

    def get(self, clip):
        if clip not in self.cache:
            from .grab import load_clip

            data = load_clip(self.root / clip)
            transl = np.asarray(data["object"]["params"]["transl"], np.float64)
            fps = float(data["framerate"])
            obj = str(data["obj_name"])
            given = self.assignment(self.swaps, clip)
            if given:
                word, key = given["word"], given["instance"]
            else:
                word, key = self.vocabulary["extras"][obj], f"grab:{obj}"
            key = f"clip:{clip}" if self.key_by == "clip" else key
            material, mass = draw_properties(self.table, word, key)
            self.cache[clip] = dict(transl=transl, fps=fps, acc=object_acceleration(transl, fps), rest_z=transl[0, 2],
                                    word=word, key=key, material=material, mass=mass)
        return self.cache[clip]


def _match_frames(transl, stored, fps):
    """Raw frame index per stored second: nearest object translation; ties go to the previous match + fps."""
    sources, errors, previous = [], [], None
    for position in stored:
        distance = np.linalg.norm(transl - position[None], axis=1)
        near = np.flatnonzero(distance <= distance.min() + 1e-6)
        pick = near[np.argmin(np.abs(near - (previous + fps)))] if previous is not None else near[0]
        sources.append(int(pick))
        errors.append(float(distance[pick]))
        previous = pick
    return np.array(sources), np.array(errors)


def session_signals(video_id, group, body, clips, table):
    """One session -> dict of [T,...] arrays. `group` = that session's frames.parquet rows."""
    steps = len(body["frame_idx"])
    bits = np.unpackbits(body["hand_contact"], axis=1)[:, :2 * HAND_VERTICES]
    counts = np.stack([bits[:, :HAND_VERTICES].sum(1), bits[:, HAND_VERTICES:].sum(1)], 1).astype(float)
    pose, camera = body["object_pose"].astype(np.float64), body["camera"].astype(np.float64)
    rng = np.random.default_rng(stable_seed(f"physical:{video_id}"))
    weber = np.exp(rng.normal(0.0, WEBER_SIGMA, (steps, 2)))
    noise = rng.normal(0.0, SKIN_NOISE_C, (steps, 2))
    name = {int(t): c for t, c in zip(group.frame_idx, group["clip"])}
    present = np.zeros(steps, bool)
    present[list(name)] = True
    run = np.full(steps, -1)  # clip run id; a run ends at a clip change or a missing frame
    count = -1
    for t in range(steps):
        if t in name:
            if t == 0 or (t - 1) not in name or name[t - 1] != name[t]:
                count += 1
            run[t] = count
    load = np.zeros((steps, 2, 5))
    target = np.zeros((steps, 2))
    area = (np.minimum(1.0, counts / FULL_CONTACT_VERTICES) * present[:, None]).astype(np.float32)
    names = sorted(table["materials"])
    material_idx = np.full(steps, -1)
    key, material, mass, errors = np.full(steps, "", dtype=object), np.full(steps, "", dtype=object), np.zeros(steps), []
    for r in range(count + 1):
        seconds = np.flatnonzero(run == r)
        info = clips.get(name[seconds[0]])
        sources, err = _match_frames(info["transl"], pose[seconds, :3, 3], info["fps"])
        errors.extend(err.tolist())
        key[seconds], material[seconds], mass[seconds] = info["key"], info["material"], info["mass"]
        half = int(round(info["fps"] / 2))
        force = window_force(info["acc"], info["transl"][:, 2], info["rest_z"], info["mass"])
        for t, source in zip(seconds, sources):
            total = counts[t].sum()
            if total > 0:
                stats = load_stats(force[max(0, source - half):source + half + 1], camera[t, :3, :3])
                for hand in (0, 1):  # Weber noise on the vector and magnitude, not the spread
                    scale = np.r_[[weber[t, hand]] * 4, 1.0]
                    load[t, hand] = stats * (counts[t, hand] / total) * scale
            for hand in (0, 1):
                if counts[t, hand] > 0:
                    target[t, hand] = contact_target(table, info["material"])
            if counts[t].sum() > 0:
                material_idx[t] = names.index(info["material"])
    new_clip = np.r_[True, run[1:] != run[:-1]]
    skin = np.stack([skin_dynamics(target[:, h], counts[:, h] > 0, new_clip) for h in (0, 1)], 1)
    skin = (skin + noise) * present[:, None]
    return dict(frame_idx=body["frame_idx"], load=load.astype(np.float32), skin_dT=skin.astype(np.float32), contact_area=area,
                material_idx=material_idx.astype(np.int8),
                instance_key=key.astype(str), material=material.astype(str), mass_kg=mass.astype(np.float32),
                clip_run=run.astype(np.int16), match_error_m=np.array(errors, np.float32)), errors


def build_physical(episodes, config_path=None, swaps_path="outputs/assets/objaverse/swaps_v2.json", key_by="instance"):
    """key_by 'clip' draws material and mass per GRAB clip (every clip a new object) into physical_by_clip/."""
    import pandas as pd

    from .build import DEFAULT_CONFIG
    from .assets import load_vocabulary

    config = json.loads(Path(config_path or DEFAULT_CONFIG).read_text())
    episodes = Path(episodes)
    table = load_materials()
    swaps = json.loads(Path(swaps_path).read_text())
    clips = _Clips(config["grab_root"], swaps, table, load_vocabulary(), key_by)
    frames = pd.read_parquet(episodes / "frames.parquet")
    output = episodes / ("physical" if key_by == "instance" else "physical_by_clip")
    output.mkdir(exist_ok=True)
    started, worst = time.monotonic(), 0.0
    groups = list(frames.groupby("video_id", sort=True))
    for number, (video_id, group) in enumerate(groups):
        with np.load(episodes / "body" / f"{video_id}.npz") as body:
            body = {k: body[k] for k in ("frame_idx", "object_pose", "camera", "hand_contact")}
        signals, errors = session_signals(video_id, group.sort_values("frame_idx"), body, clips, table)
        assert max(errors) < 1e-4, (video_id, max(errors))
        worst = max(worst, max(errors))
        np.savez_compressed(output / f"{video_id}.npz", **signals)
        if number % 100 == 0:
            print(f"{number}/{len(groups)} {time.monotonic() - started:.0f}s", flush=True)
    print(f"sessions {len(groups)} runtime {time.monotonic() - started:.0f}s max pose-match error {worst:.2e} m")
    return worst
