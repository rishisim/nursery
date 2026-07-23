from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.weak_alignment.study import (
    freeze_study,
    run_development_study,
    verify_study,
)


DEFAULT_OUT = ROOT / "output" / "synthetic_weak_alignment_recovery_v1"
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_weak_alignment_recovery_v1.yaml"
DEFAULT_PROTOCOL = ROOT / "docs" / "synthetic_weak_alignment_recovery_v1_protocol.md"
DEFAULT_SOURCES = ROOT / "docs" / "synthetic_weak_alignment_recovery_v1_primary_sources.json"
TRACKED_PATHS = (
    "babyworld_lite/weak_alignment/__init__.py",
    "babyworld_lite/weak_alignment/protocol.py",
    "babyworld_lite/weak_alignment/synthetic.py",
    "babyworld_lite/weak_alignment/learners.py",
    "babyworld_lite/weak_alignment/analysis.py",
    "babyworld_lite/weak_alignment/study.py",
    "scripts/run_synthetic_weak_alignment_v1.py",
    "tests/test_synthetic_weak_alignment_v1.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen development-only synthetic weak-alignment recovery study"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze protocol before any development outcome")
    freeze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    freeze.add_argument("--out", type=Path, default=DEFAULT_OUT)

    run = subparsers.add_parser("run-development", help="run only the frozen development seeds")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    run.add_argument("--freeze-receipt", type=Path, default=DEFAULT_OUT / "freeze_receipt.json")
    run.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=DEFAULT_OUT / "confirmation_reserve_manifest.json",
    )
    run.add_argument("--out", type=Path, default=DEFAULT_OUT)

    verify = subparsers.add_parser("verify", help="verify freeze, guards, audits, and artifact hashes")
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify.add_argument("--freeze-receipt", type=Path, default=DEFAULT_OUT / "freeze_receipt.json")
    verify.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=DEFAULT_OUT / "confirmation_reserve_manifest.json",
    )
    verify.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "freeze":
        result = freeze_study(
            repository_root=ROOT,
            config_path=args.config,
            protocol_path=args.protocol,
            sources_path=args.sources,
            output_dir=args.out,
            tracked_paths=TRACKED_PATHS,
        )
    elif args.command == "run-development":
        command = " ".join(shlex.quote(value) for value in sys.argv)
        result = run_development_study(
            repository_root=ROOT,
            config_path=args.config,
            freeze_receipt_path=args.freeze_receipt,
            confirmation_manifest_path=args.confirmation_manifest,
            sources_path=args.sources,
            output_dir=args.out,
            command=command,
        )
    elif args.command == "verify":
        result = verify_study(
            repository_root=ROOT,
            config_path=args.config,
            freeze_receipt_path=args.freeze_receipt,
            confirmation_manifest_path=args.confirmation_manifest,
            output_dir=args.out,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
