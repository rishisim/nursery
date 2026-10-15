"""Deterministic equal-duration sessions with hard cuts and subject splits."""

import numpy as np


def subject_split(subject):
    number = int(subject.removeprefix("s"))
    if subject != f"s{number}" or number not in range(1, 11):
        raise ValueError(f"Unknown GRAB subject: {subject}")
    return "train" if number <= 8 else "dev" if number == 9 else "test"


def chain_clips(clips, duration_s, seed):
    if not isinstance(duration_s, int) or duration_s <= 0 or not clips:
        raise ValueError("Positive integer duration and nonempty clips required")
    if any(length <= 0 or fps <= 0 for _, length, fps in clips):
        raise ValueError("Invalid clip timebase")
    rng = np.random.default_rng(seed)
    segments, elapsed = [], 0.0
    while elapsed < duration_s:
        for index in rng.permutation(len(clips)):
            name, length, fps = clips[index]
            stop = min(duration_s, elapsed + length / fps)
            segments.append({"clip": name, "start": elapsed, "stop": stop, "fps": fps,
                             "source_frames": length})
            elapsed = stop
            if elapsed == duration_s:
                break
    return segments


def sample_chain(segments, duration_s):
    samples = []
    for second in range(duration_s):
        segment = next(item for item in segments if item["start"] <= second < item["stop"])
        source = min(int((second - segment["start"]) * segment["fps"]),
                     segment["source_frames"] - 1)
        samples.append((segment["clip"], source))
    return samples


def contact_rule(labels, hand_ids):
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("Contact labels must be frames by SMPL-X vertices")
    return tuple((labels[:, hand_ids[hand]] != 0).any(axis=1)
                 for hand in ("left_hand", "right_hand"))
