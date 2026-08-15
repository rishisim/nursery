from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babyworld_lite.hoidini_intermimic.stage12 import (  # noqa: E402
    ContractError,
    load_task_contract,
    resolve_scene_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the Stage 1 task contract or resolve a robot-free EmbodiedGenV2 Stage 2 manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-task")
    validate.add_argument("--task-contract", type=Path, required=True)

    resolve = subparsers.add_parser("resolve-scene")
    resolve.add_argument("--task-contract", type=Path, required=True)
    resolve.add_argument("--layout", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-task":
            contract = load_task_contract(args.task_contract)
            print(json.dumps({"status": "valid", "task_id": contract["task_id"]}, sort_keys=True))
            return 0
        manifest = resolve_scene_manifest(args.task_contract, args.layout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "resolved", "scene_id": manifest["scene_id"], "output": str(args.output)}, sort_keys=True))
        return 0
    except ContractError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
