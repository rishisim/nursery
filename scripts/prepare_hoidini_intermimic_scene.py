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
    bind_shared_scene,
    load_task_contract,
    resolve_scene_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Stage 1, inspect a layout diagnostically, or bind the exact shared Stage 2 scene."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-task")
    validate.add_argument("--task-contract", type=Path, required=True)

    inspect = subparsers.add_parser("inspect-layout")
    inspect.add_argument("--task-contract", type=Path, required=True)
    inspect.add_argument("--layout", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)

    bind = subparsers.add_parser("bind-shared-scene")
    bind.add_argument("--task-contract", type=Path, required=True)
    bind.add_argument("--handoff", type=Path)
    bind.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-task":
            contract = load_task_contract(args.task_contract)
            print(json.dumps({"status": "valid", "task_id": contract["task_id"]}, sort_keys=True))
            return 0
        if args.command == "inspect-layout":
            result = resolve_scene_manifest(args.task_contract, args.layout)
            status = "inspected_diagnostic_only"
            identity = {"scene_id": result["scene_id"]}
        else:
            result = bind_shared_scene(args.task_contract, args.handoff)
            status = "bound"
            identity = {"task_id": result["task"]["task_id"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, **identity, "output": str(args.output)}, sort_keys=True))
        return 0
    except (ContractError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
