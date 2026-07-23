#!/usr/bin/env python3
"""Plan future child-only whole-bundle shards; never launch training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babyworld_lite.child_only_v1.parallel import build_parallel_plan, write_plan
from babyworld_lite.child_only_v1.policy import CONSTRUCTION_PROFILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-digest", required=True)
    parser.add_argument("--corpus-instance-id", default="construction-fixture-instance-v1")
    parser.add_argument("--corpus-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--model-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--backend", choices=("mps", "cuda", "slurm_cuda"), required=True)
    parser.add_argument("--max-trainers", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan = build_parallel_plan(
        protocol_digest=args.protocol_digest,
        corpus_instance_id=args.corpus_instance_id,
        corpus_seeds=args.corpus_seeds,
        model_seeds=args.model_seeds,
        backend=args.backend,
        max_trainers=args.max_trainers,
        cpu_workers=args.cpu_workers,
        profile_label=CONSTRUCTION_PROFILE,
        outcome_task_authorized=False,
    )
    write_plan(plan, args.out)


if __name__ == "__main__":
    main()
