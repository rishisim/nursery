"""Shared fixtures for the speech and learner tests."""

import json
import socket
import subprocess

import pytest

from nursery.common import _output_directory
from nursery.learner import UPSTREAM, _upstream_module


@pytest.fixture
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def upstream():
    if not (UPSTREAM / "src" / "train_frame_mil.py").exists():
        pytest.skip(f"Pinned upstream checkout absent: {UPSTREAM}")
    import torch

    torch.set_num_threads(1)
    return _upstream_module("train_frame_mil")


def synthetic_frames(n=120, videos=2):
    import pandas as pd

    objects = json.dumps([{"category": c, "visible_px": 40, "cells": [0, 1]}
                          for c in ("mug", "bowl", "apple", "banana")])
    rows = []
    for video in range(videos):
        for frame in range(n):
            contact = ((frame * 17 + video * 23) % 101) < (20 + video * 10)
            rows.append({"video_id": f"v{video}", "frame_idx": frame,
                         "image": f"frames/v{video}/{frame:05d}.jpg", "subject": f"s{video + 1}",
                         "room": "train_room", "clip": "synthetic", "contact_left": contact,
                         "contact_right": False, "held_category": "mug" if contact else None,
                         "objects": objects})
    return pd.DataFrame(rows)



@pytest.fixture
def episodes(tmp_path):
    import pandas as pd
    from PIL import Image

    # Exercise the production git check-ignore guard rather than bypassing it.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("outputs/\n")
    root = _output_directory(tmp_path / "outputs" / "episodes")
    frames = synthetic_frames()
    sessions = pd.DataFrame([
        {"video_id": v, "subject": f"s{i + 1}", "room": "train_room", "split": "train",
         "clips": '[]', "distractors": '[]', "duration_s": 120}
        for i, v in enumerate(("v0", "v1"))
    ] + [{"video_id": "v_dev", "subject": "s9", "room": "dev_room", "split": "dev",
          "clips": '[]', "distractors": '[]', "duration_s": 1},
         {"video_id": "v_test", "subject": "s10", "room": "test_room", "split": "test",
          "clips": '[]', "distractors": '[]', "duration_s": 1}])
    extra = frames.iloc[:2].copy()
    extra["video_id"] = ["v_dev", "v_test"]
    extra["frame_idx"] = 0
    extra["image"] = ["frames/v_dev/00000.jpg", "frames/v_test/00000.jpg"]
    extra["subject"] = ["s9", "s10"]
    extra["room"] = ["dev_room", "test_room"]
    frames = pd.concat([frames, extra], ignore_index=True)
    probes = pd.DataFrame([
        {"probe_id": f"{split}_{cat}_{i}", "image": f"probes/{split}_{cat}_{i}.jpg",
         "category": cat, "room": f"probe_{split}", "split": split}
        for split in ("dev", "test") for cat in ("mug", "bowl", "apple", "banana") for i in range(4)
    ])
    for image in pd.concat([frames.image, probes.image]):
        path = root / image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (80, 100, 120)).save(path)
    for name, table in (("frames", frames), ("sessions", sessions), ("probes", probes)):
        table.to_parquet(root / f"{name}.parquet", index=False)
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "synthetic": True}))
    return root, tmp_path / "outputs" / "run"
