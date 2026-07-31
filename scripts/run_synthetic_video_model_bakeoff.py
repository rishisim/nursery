#!/usr/bin/env python3
"""Compile, run, and review the public-only four-family video bakeoff."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nursery_egobaby_preflight.contract import canonical_json_sha256
from nursery_egobaby_preflight.synthetic_video_model_bakeoff import (
    compile_bakeoff_work_order,
    execute_bakeoff,
    finalize_blinded_review,
    inspect_blinded_review_status,
    load_default_configs,
    planned_cost_usd,
    render_blinded_gallery,
    verify_reference_media,
    write_bakeoff_work_order,
)

DEFAULT_BAKEOFF_CONFIG = (
    REPOSITORY_ROOT / "configs" / "synthetic_video_model_bakeoff.json"
)
DEFAULT_PUBLIC_CONFIG = (
    REPOSITORY_ROOT / "configs" / "synthetic_video_public_pilot.json"
)
DEFAULT_COMPLETED_QUALITY_CONFIG = (
    REPOSITORY_ROOT / "configs" / "synthetic_video_quality_ceiling.json"
)


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _configs(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    return load_default_configs(
        args.config,
        args.public_pilot_config,
        args.completed_quality_config,
        REPOSITORY_ROOT,
    )


def _run_root(args: argparse.Namespace) -> Path:
    return Path(
        args.run_root
        or REPOSITORY_ROOT
        / "runs"
        / "synthetic_video_model_bakeoff"
        / args.run_id
    ).resolve()


def _load_work_order(args: argparse.Namespace) -> dict:
    path = Path(args.work_order or _run_root(args) / "work_order.json")
    return json.loads(path.read_text())


def _ensure_ignored_output(path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", str(path)],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generated output path is not ignored: {path}")


def _clean_execution_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("paid execution requires a clean Git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def command_validate(args: argparse.Namespace) -> None:
    bakeoff, public, completed = _configs(args)
    _json(
        {
            "status": "PASS",
            "comparison_id": bakeoff["comparison_id"],
            "protocol_status": bakeoff["status"],
            "bakeoff_protocol_sha256": canonical_json_sha256(bakeoff),
            "public_pilot_protocol_sha256": canonical_json_sha256(public),
            "completed_quality_protocol_sha256": canonical_json_sha256(completed),
            "new_provider_submission_count": 8,
            "maximum_expected_new_charge_usd": float(planned_cost_usd(bakeoff)),
            "paid_execution_authorized": bakeoff["authorization"][
                "paid_execution_authorized"
            ],
            "scientific_training_use_authorized": False,
            "privacy_boundary": (
                "public-only; ChildLens and BabyView inputs and derivatives prohibited"
            ),
        }
    )


def command_compile(args: argparse.Namespace) -> None:
    bakeoff, public, completed = _configs(args)
    root = _run_root(args)
    _ensure_ignored_output(root)
    order = compile_bakeoff_work_order(
        bakeoff,
        public,
        completed,
        REPOSITORY_ROOT,
        args.run_id,
    )
    path = write_bakeoff_work_order(root, order, bakeoff, public)
    _json(
        {
            "status": "COMPILED_NO_API_CALL",
            "path": str(path),
            "new_attempt_count": len(order["attempts"]),
            "blinded_clip_count": sum(
                len(scene["presentation"]) for scene in order["scenes"]
            ),
            "maximum_expected_new_charge_usd": order["planned_cost_usd"],
            "bakeoff_protocol_sha256": order["bakeoff_protocol_sha256"],
            "work_order_sha256": order["work_order_sha256"],
        }
    )


def command_plan(args: argparse.Namespace) -> None:
    bakeoff, public, completed = _configs(args)
    order = compile_bakeoff_work_order(
        bakeoff,
        public,
        completed,
        REPOSITORY_ROOT,
        args.run_id,
    )
    references = verify_reference_media(
        bakeoff,
        public,
        completed,
        REPOSITORY_ROOT,
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    credentials = {
        family: {
            "environment_variable": settings[
                "credential_environment_variable"
            ],
            "present": bool(
                os.environ.get(settings["credential_environment_variable"])
            ),
        }
        for family, settings in bakeoff["new_families"].items()
    }
    credential_ready = all(value["present"] for value in credentials.values())
    spend_ready = (
        bakeoff["status"] == "FROZEN_EXECUTION_AUTHORIZED"
        and bakeoff["authorization"]["paid_execution_authorized"] is True
        and Decimal(
            str(bakeoff["provider_cost"]["user_authorized_new_spend_usd"])
        )
        == planned_cost_usd(bakeoff)
    )
    _json(
        {
            "status": (
                "READY_FOR_AUTHORIZED_PAID_EXECUTION"
                if credential_ready and spend_ready
                else "BLOCKED_AT_CREDENTIAL_AND_SPEND_GATE"
            ),
            "run_id": args.run_id,
            "new_attempt_ids": [
                attempt["attempt_id"] for attempt in order["attempts"]
            ],
            "reference_media_valid_count": {
                family: len(records) for family, records in references.items()
            },
            "credentials": credentials,
            "spend_authorized_in_protocol": spend_ready,
            "maximum_expected_gemini_charge_usd": bakeoff["provider_cost"][
                "gemini"
            ]["maximum_expected_charge_usd"],
            "maximum_expected_minimax_charge_usd": bakeoff["provider_cost"][
                "minimax"
            ]["maximum_expected_charge_usd"],
            "maximum_expected_new_charge_usd": float(planned_cost_usd(bakeoff)),
            "automatic_retries": 0,
            "scientific_training_use_authorized": False,
            "next_gate": (
                "run exactly eight calls with --approved-spend-usd 4.66"
                if credential_ready and spend_ready
                else (
                    "provide GEMINI_API_KEY and MINIMAX_API_KEY, then explicitly "
                    "approve the new $4.66 maximum expected charge"
                )
            ),
        }
    )


def command_run(args: argparse.Namespace) -> None:
    bakeoff, public, completed = _configs(args)
    root = _run_root(args)
    _ensure_ignored_output(root)
    status = execute_bakeoff(
        bakeoff,
        public,
        completed,
        _load_work_order(args),
        repository_root=REPOSITORY_ROOT,
        run_root=root,
        approved_spend_usd=Decimal(args.approved_spend_usd),
        gemini_api_key=os.environ.get(
            bakeoff["new_families"]["gemini_omni_flash"][
                "credential_environment_variable"
            ],
            "",
        ),
        minimax_api_key=os.environ.get(
            bakeoff["new_families"]["minimax_h3"][
                "credential_environment_variable"
            ],
            "",
        ),
        implementation_commit=_clean_execution_commit(),
        poll_interval_seconds=args.poll_interval_seconds,
        ffmpeg_executable=args.ffmpeg,
        ffprobe_executable=args.ffprobe,
    )
    _json(status)


def command_gallery(args: argparse.Namespace) -> None:
    bakeoff, public, completed = _configs(args)
    root = _run_root(args)
    _ensure_ignored_output(root)
    gallery = render_blinded_gallery(
        bakeoff,
        public,
        completed,
        _load_work_order(args),
        repository_root=REPOSITORY_ROOT,
        run_root=root,
        output_path=args.output,
    )
    _json(
        {
            "status": "READY",
            "gallery": str(gallery),
            "blinded_clip_count": 16,
            "blinding_key": str(root / "blinding_key.json"),
        }
    )


def command_review_status(args: argparse.Namespace) -> None:
    bakeoff, public, completed = _configs(args)
    _json(
        inspect_blinded_review_status(
            bakeoff,
            public,
            completed,
            _load_work_order(args),
            repository_root=REPOSITORY_ROOT,
            run_root=_run_root(args),
        )
    )


def command_finalize_review(args: argparse.Namespace) -> None:
    bakeoff, public, completed = _configs(args)
    _json(
        finalize_blinded_review(
            bakeoff,
            public,
            completed,
            _load_work_order(args),
            repository_root=REPOSITORY_ROOT,
            run_root=_run_root(args),
            ffmpeg_executable=args.ffmpeg,
            ffprobe_executable=args.ffprobe,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", default=str(DEFAULT_BAKEOFF_CONFIG))
    root.add_argument(
        "--public-pilot-config",
        default=str(DEFAULT_PUBLIC_CONFIG),
    )
    root.add_argument(
        "--completed-quality-config",
        default=str(DEFAULT_COMPLETED_QUALITY_CONFIG),
    )
    commands = root.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate all frozen inputs")
    validate.set_defaults(func=command_validate)

    compile_parser = commands.add_parser(
        "compile",
        help="write the eight-call work order without contacting either provider",
    )
    compile_parser.add_argument("--run-id", required=True)
    compile_parser.add_argument("--run-root")
    compile_parser.set_defaults(func=command_compile)

    plan = commands.add_parser(
        "plan",
        help="verify all eight reference clips and report the no-launch gate",
    )
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--ffmpeg", default="ffmpeg")
    plan.add_argument("--ffprobe", default="ffprobe")
    plan.set_defaults(func=command_plan)

    run = commands.add_parser(
        "run",
        help="run the eight paid Gemini and MiniMax calls sequentially",
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
        help="build the four-family blinded gallery",
    )
    gallery.add_argument("--run-id", required=True)
    gallery.add_argument("--run-root")
    gallery.add_argument("--work-order")
    gallery.add_argument("--output")
    gallery.set_defaults(func=command_gallery)

    review_status = commands.add_parser(
        "review-status",
        help="validate all 16 QA records without opening the family key",
    )
    review_status.add_argument("--run-id", required=True)
    review_status.add_argument("--run-root")
    review_status.add_argument("--work-order")
    review_status.set_defaults(func=command_review_status)

    finalize = commands.add_parser(
        "finalize-review",
        help="freeze QA, unblind, score, and write the recommendation",
    )
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--run-root")
    finalize.add_argument("--work-order")
    finalize.add_argument("--ffmpeg", default="ffmpeg")
    finalize.add_argument("--ffprobe", default="ffprobe")
    finalize.set_defaults(func=command_finalize_review)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
