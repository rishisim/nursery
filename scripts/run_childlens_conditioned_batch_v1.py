#!/usr/bin/env python3
"""Prepare or execute the frozen ChildLens-conditioned pilot population."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.childlens_engine_bakeoff.conditioned_sampler import (
    FrozenSampler,
    qualify_sampled_population,
    sha256_file,
)
from babyworld_lite.childlens_engine_bakeoff.run_kernel_episode import run


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(config_path: Path, output_root: Path) -> tuple[FrozenSampler, list[dict]]:
    sampler = FrozenSampler.load(config_path)
    episodes = sampler.sample()
    output_root.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        episode_dir = output_root / "episodes" / episode["episode_id"]
        write_json(episode_dir / "episode_intent.json", episode["intent"])
        write_json(episode_dir / "resolved_episode_spec.json", episode["resolved_spec"])
        write_json(
            episode_dir / "sampling_receipt.json",
            {
                "episode_id": episode["episode_id"],
                "seed": episode["seed"],
                "sampled_measurements": episode["sampled_measurements"],
                "calibration_sha256": sampler.config["calibration"]["sha256"],
                "spec_sha256": episode["resolved_spec"]["spec_sha256"],
            },
        )
    qualification = qualify_sampled_population(
        episodes, sampler.config, sampler.calibration
    )
    write_json(output_root / "planned_population_qualification.json", qualification)
    write_json(
        output_root / "frozen_batch_manifest.json",
        {
            "schema": "ChildLensConditionedPilotManifest.v1",
            "config_sha256": sha256_file(config_path),
            "calibration": sampler.config["calibration"],
            "episode_count": len(episodes),
            "episode_ids": [episode["episode_id"] for episode in episodes],
            "seeds": [episode["seed"] for episode in episodes],
            "scientific_boundary": (
                "ChildLens ages-3-to-5 provisional distribution calibration; "
                "not infant calibration, lexical-grounding evidence, or a cue-lift result"
            ),
        },
    )
    return sampler, episodes


def execute_episode(
    episode: dict,
    output_root: Path,
    scene_root: Path,
    mimo_assets: Path,
    *,
    render: bool,
    attempt: int,
) -> dict:
    episode_dir = output_root / "episodes" / episode["episode_id"]
    scene_id = episode["intent"]["scene"]["id"]
    started = time.perf_counter()
    try:
        qa = run(
            episode_dir / "resolved_episode_spec.json",
            episode_dir / "episode_intent.json",
            scene_root / f"{scene_id}_physics.xml",
            scene_root / f"{scene_id}_physics_map.png",
            mimo_assets,
            episode_dir,
            render=render,
        )
        return {
            "episode_id": episode["episode_id"],
            "attempt": attempt,
            "status": "PASS_EXECUTION",
            "wall_seconds": time.perf_counter() - started,
            "rendered": render,
            "physics_qa": qa["physics_qa"],
            "shared_clock_qa": qa["shared_clock_qa"],
        }
    except Exception as error:
        return {
            "episode_id": episode["episode_id"],
            "attempt": attempt,
            "status": "FAIL_EXECUTION",
            "wall_seconds": time.perf_counter() - started,
            "rendered": render,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/childlens_conditioned_batch_v1.json",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=ROOT / ".external/engine_bakeoff/assets/molmospaces/scenes/ithor",
    )
    parser.add_argument(
        "--mimo-assets",
        required=False,
        type=Path,
        default=Path(
            "/Volumes/EOS_202603/RESEARCH/nursery/engine_bakeoff/repos/"
            "MIMo/mimoEnv/assets"
        ),
    )
    parser.add_argument("--indices", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.render and not args.execute:
        parser.error("--render requires --execute")
    sampler, episodes = prepare(args.config, args.output_root)
    del sampler
    if not args.execute:
        print(json.dumps({"prepared": len(episodes)}, indent=2))
        return
    indices = (
        [int(token) for token in args.indices.split(",") if token]
        if args.indices
        else list(range(len(episodes)))
    )
    audit_path = args.output_root / "attempt_audit.json"
    prior = json.loads(audit_path.read_text()) if audit_path.exists() else []
    prior_counts = {
        episode["episode_id"]: sum(
            item["episode_id"] == episode["episode_id"] for item in prior
        )
        for episode in episodes
    }
    call_args = [
        (
            episodes[index],
            args.output_root,
            args.scene_root,
            args.mimo_assets,
            args.render,
            prior_counts[episodes[index]["episode_id"]] + 1,
        )
        for index in indices
    ]
    if args.workers == 1:
        attempts = [
            execute_episode(
                episode,
                output_root,
                scene_root,
                mimo_assets,
                render=render,
                attempt=attempt,
            )
            for episode, output_root, scene_root, mimo_assets, render, attempt in call_args
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = [
                executor.submit(
                    execute_episode,
                    episode,
                    output_root,
                    scene_root,
                    mimo_assets,
                    render=render,
                    attempt=attempt,
                )
                for episode, output_root, scene_root, mimo_assets, render, attempt in call_args
            ]
            attempts = [future.result() for future in futures]
    write_json(audit_path, [*prior, *attempts])
    print(json.dumps(attempts, indent=2))
    if any(item["status"] != "PASS_EXECUTION" for item in attempts):
        sys.exit(1)


if __name__ == "__main__":
    main()
