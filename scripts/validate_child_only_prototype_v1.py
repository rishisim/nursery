#!/usr/bin/env python3
"""Run child-only construction gates without an acquisition outcome."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from babyworld_lite.child_only_v1.validation import run_construction_validation, write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--benchmark-device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run_construction_validation(args.repo_root, benchmark_device=args.benchmark_device)
    write_validation_report(report, args.out)


if __name__ == "__main__":
    main()
