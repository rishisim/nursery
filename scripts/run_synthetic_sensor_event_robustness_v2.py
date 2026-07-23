from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.sensor_alignment_v2.study import (
    compare_reproductions,
    freeze_study,
    run_development_study,
    verify_study,
)


DEFAULT_OUT = ROOT / "output" / "synthetic_sensor_event_robustness_v2"
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_sensor_event_robustness_v2.yaml"
DEFAULT_PROTOCOL = ROOT / "docs" / "synthetic_sensor_event_robustness_v2_protocol.md"
DEFAULT_SOURCES = ROOT / "docs" / "synthetic_sensor_event_robustness_v2_primary_sources.json"
TRACKED_PATHS = (
    "babyworld_lite/sensor_alignment_v2/__init__.py",
    "babyworld_lite/sensor_alignment_v2/protocol.py",
    "babyworld_lite/sensor_alignment_v2/synthetic.py",
    "babyworld_lite/sensor_alignment_v2/detector.py",
    "babyworld_lite/sensor_alignment_v2/learners.py",
    "babyworld_lite/sensor_alignment_v2/analysis.py",
    "babyworld_lite/sensor_alignment_v2/study.py",
    "scripts/run_synthetic_sensor_event_robustness_v2.py",
    "tests/test_synthetic_sensor_event_robustness_v2.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen development-only synthetic raw-sensor/event robustness study v2"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze", help="freeze protocol and code before any development outcome"
    )
    freeze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    freeze.add_argument("--out", type=Path, default=DEFAULT_OUT)

    run = subparsers.add_parser(
        "run-development", help="run only the frozen v2 development protocol"
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    run.add_argument(
        "--freeze-receipt",
        type=Path,
        default=DEFAULT_OUT / "freeze_receipt.json",
    )
    run.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=DEFAULT_OUT / "confirmation_reserve_manifest.json",
    )
    run.add_argument("--out", type=Path, default=DEFAULT_OUT)

    verify = subparsers.add_parser(
        "verify", help="verify freeze, reserve guards, audits, and deterministic hashes"
    )
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify.add_argument(
        "--freeze-receipt",
        type=Path,
        default=DEFAULT_OUT / "freeze_receipt.json",
    )
    verify.add_argument(
        "--confirmation-manifest",
        type=Path,
        default=DEFAULT_OUT / "confirmation_reserve_manifest.json",
    )
    verify.add_argument("--out", type=Path, default=DEFAULT_OUT)

    compare = subparsers.add_parser(
        "compare", help="compare every deterministic artifact in two completed runs"
    )
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
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
        result = run_development_study(
            repository_root=ROOT,
            config_path=args.config,
            freeze_receipt_path=args.freeze_receipt,
            confirmation_manifest_path=args.confirmation_manifest,
            sources_path=args.sources,
            output_dir=args.out,
            command=" ".join(shlex.quote(value) for value in sys.argv),
        )
    elif args.command == "verify":
        result = verify_study(
            repository_root=ROOT,
            config_path=args.config,
            freeze_receipt_path=args.freeze_receipt,
            confirmation_manifest_path=args.confirmation_manifest,
            output_dir=args.out,
        )
    elif args.command == "compare":
        result = compare_reproductions(left_dir=args.left, right_dir=args.right)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
