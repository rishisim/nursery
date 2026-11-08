"""Scene-only object looks on the saved 1-Hz timeline.

Object identities and categories are provenance for speech/audits, never learner
inputs. The learner-facing signal is the normalized image point. This module
does not import or read speech, utterances, words, or naming outcomes.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from nursery.common import _output_directory


DEFAULTS = {"mean_look_s": 2.334, "p_held": 0.60,
            "size_exponent": 0.5, "motion_gain": 2.0,
            "width": 320, "height": 240}


def generate_gaze(frames, seed=0, **overrides):
    """Sample looks from visible instance pixels, holding and image motion only.

    Geometry must supply id and image_xy (a visible pixel centre) per object.
    A sticky categorical Markov chain has stationary held preference 0.60 and
    mean observed object bout 2.334 s in a fixed feasible scene. Dynamic size,
    motion, disappearance and source-clip cuts alter achieved durations, which
    must be reported. A bout is an actual contiguous target run, not a hidden
    resampling event. No speech information determines any transition.
    """
    settings = {**DEFAULTS, **overrides}
    if settings["mean_look_s"] < 1 or not 0 <= settings["p_held"] <= 1:
        raise ValueError("Invalid gaze duration or held probability")
    if settings["size_exponent"] < 0 or settings["motion_gain"] < 0:
        raise ValueError("Size and motion weights must be nonnegative")
    if frames.duplicated(["video_id", "frame_idx"]).any():
        raise ValueError("Duplicate gaze frame keys")
    rows = []
    for video, group in frames.groupby("video_id", sort=True):
        group = group.sort_values("frame_idx")
        if not np.array_equal(group.frame_idx, np.arange(len(group))):
            raise ValueError("Gaze requires contiguous 1-Hz frames starting at zero")
        digest = hashlib.sha256(f"{seed}:{video}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        previous, target, clip, bout = {}, None, None, -1
        for row in group.itertuples():
            if not isinstance(row.clip, str) or not row.clip.strip():
                raise ValueError("Gaze requires source-clip identities")
            cut = row.clip != clip
            if cut:
                previous, target = {}, None
                clip = row.clip
            objects = json.loads(row.objects)
            visible = {}
            for obj in objects:
                if obj["visible_px"] <= 0:
                    continue
                x, y = obj["image_xy"]
                if (not np.isfinite([x, y]).all() or
                        not 0 <= x < settings["width"] or not 0 <= y < settings["height"]):
                    raise ValueError("Gaze geometry must contain an in-frame visible pixel")
                if obj["id"] in visible:
                    raise ValueError("Duplicate visible object identity")
                visible[obj["id"]] = obj
            candidates = list(visible)
            held = [i for i in candidates if i == 1 and
                    pd.notna(row.held_category) and visible[i]["category"] == row.held_category]
            other = [i for i in candidates if i not in held]
            masses = {}
            if held:
                masses[held[0]] = settings["p_held"] if other else 1.0
            if other:
                weights = []
                for i in other:
                    obj = visible[i]
                    xy = np.asarray(obj["image_xy"]) / [settings["width"], settings["height"]]
                    motion = np.linalg.norm(xy - previous[i]) if i in previous else 0.0
                    weights.append(obj["visible_px"] ** settings["size_exponent"] *
                                   (1 + settings["motion_gain"] * motion))
                weights = np.asarray(weights, dtype=float)
                weights = weights / weights.sum() * (1 - settings["p_held"] if held else 1.0)
                masses.update(zip(other, weights))
            pi = np.array([masses[i] for i in candidates])
            diversity = 1 - float(np.square(pi).sum())
            alpha = min(1.0, 1 / (settings["mean_look_s"] * diversity)) if diversity > 0 else 1.0
            old_target = target
            if not candidates:
                target = None  # Missing signal, never an invented point.
            elif target not in visible or rng.random() < alpha:
                target = int(rng.choice(candidates, p=pi))
            if cut or target != old_target:
                bout += 1
            obj = visible.get(target)
            rows.append({"video_id": video, "frame_idx": row.frame_idx,
                         "gaze_object_id": target, "gaze_category": obj["category"] if obj else None,
                         "gaze_x": obj["image_xy"][0] / settings["width"] if obj else np.nan,
                         "gaze_y": obj["image_xy"][1] / settings["height"] if obj else np.nan,
                         "gaze_held": bool(obj and target == 1 and pd.notna(row.held_category)
                                           and obj["category"] == row.held_category),
                         "gaze_bout": bout,
                         "gaze_held_available": bool(held),
                         "gaze_duration_limited": bool(candidates and
                             settings["mean_look_s"] * diversity < 1)})
            previous = {i: np.asarray(o["image_xy"]) / [settings["width"], settings["height"]]
                        for i, o in visible.items()}
    result = pd.DataFrame(rows)
    result["gaze_object_id"] = result.gaze_object_id.astype("Int64")
    result.attrs.update(seed=seed, settings=settings)
    return result


def gaze_summary(gaze):
    valid = gaze.gaze_object_id.notna()
    durations = gaze[valid].groupby(["video_id", "gaze_bout"]).size()
    return {"n_frames": len(gaze), "n_valid": int(valid.sum()),
            "held_fraction_valid": float(gaze.loc[valid, "gaze_held"].mean()),
            "held_fraction_when_available": float(gaze.loc[gaze.gaze_held_available, "gaze_held"].mean()),
            "duration_limited_frames": int(gaze.gaze_duration_limited.sum()),
            "observed_bout_mean_s": float(durations.mean()),
            "observed_bout_median_s": float(durations.median()),
            "observed_bout_p95_s": float(durations.quantile(.95)),
            "note": "Observed bouts include clip-end and disappearance censoring; timeline is 1 Hz."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source = Path(args.frames)
    output = _output_directory(args.output)
    if (output / "gaze.parquet").exists():
        raise FileExistsError("Gaze exists; preserve the saved world")
    gaze = generate_gaze(pd.read_parquet(source), args.seed)
    gaze.to_parquet(output / "gaze.parquet", index=False)
    record = {"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "execution_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "gaze_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              **gaze.attrs, **gaze_summary(gaze)}
    (output / "gaze.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
