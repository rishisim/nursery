"""Extract the six preregistered short regression cells from two episode bundles."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


CELL_PHASES = {
    "near_miss": ("near_miss",),
    "causal_touch_push": ("touch_push",),
    "contact_grasp_lift_release": (
        "post_contact_grasp",
        "lift_place",
        "release_drop",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(bundle: Path, output_dir: Path, *, room: str) -> list[dict]:
    trace_path = bundle / "episode_trace.npz"
    video_path = bundle / "episode_with_speech.mp4"
    archive = np.load(trace_path, allow_pickle=False)
    times = archive["time_s"]
    phases = archive["phase"].astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell_name, included_phases in CELL_PHASES.items():
        mask = np.isin(phases, included_phases)
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError(f"trace has no samples for {cell_name}")
        start_s = float(times[indices[0]])
        end_s = float(times[indices[-1]])
        cell_stem = f"{room}_{cell_name}"
        cell_trace = output_dir / f"{cell_stem}.npz"
        streams = {}
        for name in archive.files:
            values = archive[name]
            if name.startswith("substep_"):
                submask = (
                    (archive["substep_time_s"] >= start_s)
                    & (archive["substep_time_s"] <= end_s)
                )
                streams[name] = values[submask]
            elif values.shape and values.shape[0] == len(times):
                streams[name] = values[mask]
        np.savez_compressed(cell_trace, **streams)
        cell_video = output_dir / f"{cell_stem}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_s),
                "-to",
                str(end_s),
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(cell_video),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rows.append(
            {
                "room": room,
                "cell": cell_name,
                "phases": list(included_phases),
                "start_s": start_s,
                "end_s": end_s,
                "trace": cell_trace.name,
                "trace_sha256": _sha256(cell_trace),
                "video": cell_video.name,
                "video_sha256": _sha256(cell_video),
            }
        )
    return rows


def extract_two_rooms(
    floorplan1_bundle: Path, floorplan201_bundle: Path, output_dir: Path
) -> dict:
    rows = extract(floorplan1_bundle, output_dir, room="FloorPlan1")
    rows += extract(floorplan201_bundle, output_dir, room="FloorPlan201")
    receipt = {
        "schema": "FocusedRegressionCells.v1",
        "count": len(rows),
        "cells": rows,
        "claim_boundary": "kernel qualification cells; not a generated dataset",
    }
    (output_dir / "regression_cells_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt
