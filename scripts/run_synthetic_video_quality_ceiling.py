#!/usr/bin/env python3
"""Compile, run, and review the public-only LTX/Seedance comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nursery_egobaby_preflight.contract import canonical_json_sha256
from nursery_egobaby_preflight.synthetic_video_quality_ceiling import (
    compile_comparison_work_order,
    execute_comparison,
    load_default_configs,
    planned_cost_usd,
    render_blinded_gallery,
    verify_ltx_baseline,
    write_comparison_work_order,
)

DEFAULT_QUALITY_CONFIG = (
    REPOSITORY_ROOT / "configs" / "synthetic_video_quality_ceiling.json"
)
DEFAULT_PUBLIC_PILOT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "synthetic_video_public_pilot.json"
)


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _load_work_order(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _configs(args: argparse.Namespace) -> tuple[dict, dict]:
    return load_default_configs(args.config, args.public_pilot_config)


def _run_root(args: argparse.Namespace) -> Path:
    return Path(
        args.run_root
        or REPOSITORY_ROOT
        / "runs"
        / "synthetic_video_quality_ceiling"
        / args.run_id
    ).resolve()


def command_validate(args: argparse.Namespace) -> None:
    quality, public = _configs(args)
    _json(
        {
            "status": "PASS",
            "comparison_id": quality["comparison_id"],
            "quality_protocol_sha256": canonical_json_sha256(quality),
            "public_pilot_protocol_sha256": canonical_json_sha256(public),
            "candidate_endpoint": quality["candidate"]["endpoint"],
            "planned_generation_charge_usd": float(planned_cost_usd(quality)),
            "scientific_training_use_authorized": quality["authorization"][
                "scientific_training_use_authorized"
            ],
            "privacy_boundary": "public-only; ChildLens and BabyView derivatives prohibited",
        }
    )


def command_compile(args: argparse.Namespace) -> None:
    quality, public = _configs(args)
    order = compile_comparison_work_order(quality, public, args.run_id)
    run_root = _run_root(args)
    path = write_comparison_work_order(run_root, order, public)
    _json(
        {
            "status": "COMPILED_NO_API_CALL",
            "path": str(path),
            "candidate_attempt_count": len(order["attempts"]),
            "planned_generation_charge_usd": order["planned_cost_usd"],
            "quality_protocol_sha256": order["quality_protocol_sha256"],
            "work_order_sha256": order["work_order_sha256"],
        }
    )


def command_plan(args: argparse.Namespace) -> None:
    quality, public = _configs(args)
    order = compile_comparison_work_order(quality, public, args.run_id)
    baseline = verify_ltx_baseline(quality, public, REPOSITORY_ROOT)
    credential_name = quality["candidate"]["credential_environment_variable"]
    _json(
        {
            "status": "READY_REQUIRES_CREDENTIAL_AND_EXPLICIT_SPEND_CONFIRMATION",
            "run_id": args.run_id,
            "candidate_endpoint": quality["candidate"]["endpoint"],
            "candidate_attempt_ids": [
                attempt["attempt_id"] for attempt in order["attempts"]
            ],
            "request_settings": quality["candidate"]["request"],
            "automatic_retries": 0,
            "baseline_verified_attempt_ids": [
                baseline[scene_id]["attempt_id"]
                for scene_id in quality["comparison"]["scene_ids"]
            ],
            "credential_environment_variable": credential_name,
            "credential_present": bool(os.environ.get(credential_name)),
            "maximum_generation_charge_usd": float(planned_cost_usd(quality)),
            "scientific_training_use_authorized": False,
            "next_gate": "provide FAL_KEY securely and explicitly approve the frozen API charge",
        }
    )


def command_run(args: argparse.Namespace) -> None:
    quality, public = _configs(args)
    run_root = _run_root(args)
    work_order = _load_work_order(args.work_order or run_root / "work_order.json")
    credential_name = quality["candidate"]["credential_environment_variable"]
    api_key = os.environ.get(credential_name, "")
    status = execute_comparison(
        quality,
        public,
        work_order,
        repository_root=REPOSITORY_ROOT,
        run_root=run_root,
        approved_spend_usd=Decimal(args.approved_spend_usd),
        api_key=api_key,
        poll_interval_seconds=args.poll_interval_seconds,
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    _json(status)


def command_gallery(args: argparse.Namespace) -> None:
    quality, public = _configs(args)
    run_root = _run_root(args)
    work_order = _load_work_order(args.work_order or run_root / "work_order.json")
    gallery = render_blinded_gallery(
        quality,
        public,
        work_order,
        repository_root=REPOSITORY_ROOT,
        run_root=run_root,
        output_path=args.output,
    )
    _json(
        {
            "status": "READY",
            "gallery": str(gallery),
            "pair_count": len(work_order["pairs"]),
            "blinding_key": str(run_root / "blinding_key.json"),
        }
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", default=str(DEFAULT_QUALITY_CONFIG))
    root.add_argument(
        "--public-pilot-config",
        default=str(DEFAULT_PUBLIC_PILOT_CONFIG),
    )
    commands = root.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate both frozen protocols")
    validate.set_defaults(func=command_validate)

    compile_parser = commands.add_parser(
        "compile",
        help="write a four-request work order without making an API call",
    )
    compile_parser.add_argument("--run-id", required=True)
    compile_parser.add_argument("--run-root")
    compile_parser.set_defaults(func=command_compile)

    plan = commands.add_parser(
        "plan",
        help="verify the LTX baseline and print the exact no-launch cost gate",
    )
    plan.add_argument("--run-id", required=True)
    plan.set_defaults(func=command_plan)

    run = commands.add_parser(
        "run",
        help="run the four paid Seedance requests sequentially and resumably",
    )
    run.add_argument("--run-id", required=True)
    run.add_argument("--run-root")
    run.add_argument("--work-order")
    run.add_argument("--approved-spend-usd", required=True)
    run.add_argument("--poll-interval-seconds", type=float, default=10.0)
    run.add_argument("--ffmpeg", default="ffmpeg")
    run.add_argument("--ffprobe", default="ffprobe")
    run.set_defaults(func=command_run)

    gallery = commands.add_parser(
        "gallery",
        help="build the family-blinded LTX/Seedance gallery",
    )
    gallery.add_argument("--run-id", required=True)
    gallery.add_argument("--run-root")
    gallery.add_argument("--work-order")
    gallery.add_argument("--output")
    gallery.set_defaults(func=command_gallery)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
