from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import OUTPUT_ROOT, load_frozen_config, validate_frozen_config, write_frozen_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical Unity procedural clothed embodiment scene gate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-contract", help="validate the frozen A-gate contracts")
    validate.set_defaults(handler=_validate)
    freeze = subparsers.add_parser("freeze", help="write the frozen 3x3 contract matrix to an ignored run root")
    freeze.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    freeze.set_defaults(handler=_freeze)
    return parser


def _validate(_: argparse.Namespace) -> int:
    config = load_frozen_config()
    validate_frozen_config(config)
    print(json.dumps({"schema": config["schema"], "valid": True}, sort_keys=True))
    return 0


def _freeze(args: argparse.Namespace) -> int:
    print(json.dumps(write_frozen_bundle(args.output_root), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
