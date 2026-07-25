#!/usr/bin/env python3
"""Build privacy-safe aggregate evidence for the room/hand/camera bake-off."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from babyworld_lite.childlens_asset_rich import load_episode_specs, validate_bakeoff_matrix


def image_metrics(path: Path) -> dict:
    image = Image.open(path).convert("L")
    stat = ImageStat.Stat(image)
    edges = image.filter(ImageFilter.FIND_EDGES)
    edge_values = list(edges.getdata())
    return {
        "mean_luminance_0_255": round(stat.mean[0], 4),
        "luminance_sd_0_255": round(stat.stddev[0], 4),
        "edge_density_fraction": round(sum(value > 28 for value in edge_values) / len(edge_values), 6),
    }


def projected_target_fraction(row: dict, category: str) -> float:
    target = row["target_position_m"]
    camera = row["camera_position_m"]
    distance = math.dist(target, camera)
    radius_m = .15 if category == "ball" else .14
    focal_pixels = 13.5 / 36.0 * 960
    radius_pixels = focal_pixels * radius_m / max(distance, .001)
    return math.pi * radius_pixels ** 2 / (960 * 540)


def main(spec_path: Path, audition: Path, baseline: Path, destination: Path) -> None:
    specs = load_episode_specs(spec_path)
    matrix = validate_bakeoff_matrix(specs)
    episodes = []
    for spec in specs:
        root = audition / spec["episode_id"]
        summary = json.loads((root / "summary.json").read_text())
        rows = [json.loads(line) for line in (root / "telemetry.jsonl").read_text().splitlines()]
        fractions = [
            projected_target_fraction(rows[index], spec["target"]["category"])
            for index in (0, 89, 239)
        ]
        episodes.append({
            "episode_id": spec["episode_id"],
            "room_id": spec["room_id"],
            "action": spec["action_primitive"],
            "target_projected_fraction_begin_contact_end": [round(x, 6) for x in fractions],
            "target_fraction_minimum": round(min(fractions), 6),
            "target_fraction_gate_0_015": min(fractions) >= .015,
            "causal_contract": summary["causal_contract"],
            "image_metrics": {
                phase: image_metrics(root / f"{phase}.png")
                for phase in ("begin", "contact", "end")
            },
            "decoded_video_bytes": (root / "rgb.mp4").stat().st_size,
        })
    prior = {
        path.parent.name: image_metrics(path)
        for path in sorted(Path("tmp/childlens_asset_rich/pilots").glob("*/representative.png"))
    }
    aggregate = {
        "schema_version": "childlens-room-hand-bakeoff-evidence-1.0.0",
        "privacy": "aggregate simulator and prior synthetic-pilot evidence only; no ChildLens frame, path, audio, or transcript",
        "matrix": matrix,
        "episodes": episodes,
        "prior_pilot_representatives": prior,
        "aggregate": {
            "all_videos_present": all(item["decoded_video_bytes"] > 0 for item in episodes),
            "causal_positive_count": sum(item["causal_contract"]["target_displacement_m"] > .05 for item in episodes),
            "valid_near_miss_count": sum(
                item["causal_contract"]["near_miss"]
                and item["causal_contract"]["active_contact_frames"] == 0
                and item["causal_contract"]["target_displacement_m"] == 0
                for item in episodes
            ),
            "target_fraction_gate_pass_count": sum(item["target_fraction_gate_0_015"] for item in episodes),
            "median_render_seconds": round(statistics.median(
                json.loads((audition / spec["episode_id"] / "summary.json").read_text())["render"]["elapsed_seconds"]
                for spec in specs
            ), 3),
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
