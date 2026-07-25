#!/usr/bin/env python3
"""Run the bounded TDW adaptive-repair evidence episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.childlens_media_repair import EpisodeSpec, run_episode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--port", type=int, default=0, help="0 selects an ephemeral local port")
    parser.add_argument("--launch-arch", choices=("x86_64", "native"), default="x86_64")
    args = parser.parse_args()
    receipt = run_episode(
        args.build, args.output,
        EpisodeSpec(frames=args.frames, port=args.port, launch_arch=args.launch_arch),
    )
    result = json.loads(receipt.read_text())
    print(receipt)
    return 0 if all(
        [
            result["controls"]["positive_collision_observed"],
            result["controls"]["negative_no_collision"],
            all(result["channels"]["pass_counts"][name] > 0 for name in ("img", "id", "depth")),
        ]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
