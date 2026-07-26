#!/usr/bin/env python3
"""Render an existing causal trace through the canonical furnished/MPFB path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.childlens_engine_bakeoff.depth_composite import compose
from babyworld_lite.childlens_engine_bakeoff.physics_kernel import build_kernel_model
from babyworld_lite.childlens_engine_bakeoff.trace_render import render_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("scene_path", type=Path)
    parser.add_argument("mimo_assets", type=Path)
    parser.add_argument("blend_file", type=Path)
    parser.add_argument("blender", type=Path)
    parser.add_argument("--render-samples", type=int, default=1)
    args = parser.parse_args()
    episode = args.episode_dir
    spec = json.loads((episode / "resolved_episode_spec.json").read_text())
    staging = json.loads((episode / "staging_receipt.json").read_text())
    target = spec["target"]
    kernel = build_kernel_model(
        args.scene_path,
        args.mimo_assets,
        episode / "canonical_kernel_component.xml",
        root_xy=(staging["world_x_m"], staging["world_y_m"]),
        target_definition={
            "geometry": target["geometry"]["value"],
            "rgba": target["rgba"]["value"],
        },
    )
    archive = np.load(episode / "episode_trace.npz", allow_pickle=False)
    trace = {key: archive[key] for key in archive.files}
    background_qa = render_trace(
        kernel, trace, episode, show_collision_hand=False
    )
    overlay = episode / "mpfb_overlay"
    subprocess.run(
        [
            str(args.blender),
            str(args.blend_file),
            "--background",
            "--python",
            str(
                ROOT
                / "babyworld_lite/childlens_engine_bakeoff/mpfb_overlay_renderer.py"
            ),
            "--",
            "--trace",
            str(episode / "episode_trace.npz"),
            "--body-names",
            str(episode / "body_names.json"),
            "--spec",
            str(episode / "resolved_episode_spec.json"),
            "--output-dir",
            str(overlay),
            "--all-frames",
            "--render-samples",
            str(args.render_samples),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    canonical = episode / "canonical"
    composition = compose(episode, overlay, canonical)
    receipt = {
        "schema": "CanonicalEpisodeRenderReceipt.v1",
        "appearance": "actual furnished scene plus depth-composed MPFB hand/forearm",
        "background_qa": background_qa,
        "composition": composition,
        "canonical_video": "canonical/canonical_composed_episode.mp4",
        "inspection_sheet": "canonical/composed_inspection_sheet.png",
    }
    (episode / "canonical_render_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
