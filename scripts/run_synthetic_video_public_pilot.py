#!/usr/bin/env python3
"""Compile, inspect, and review the public-only synthetic-video pilot."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nursery_egobaby_preflight.contract import (
    canonical_json_bytes,
    canonical_json_sha256,
)
from nursery_egobaby_preflight.synthetic_video_pilot import (
    build_model_command,
    compile_retry,
    compile_work_order,
    load_config,
    media_summary,
    mux_modular_audio,
    render_gallery,
    validate_final_media,
    write_work_order,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "synthetic_video_public_pilot.json"
DEFAULT_PREFLIGHT_RESULT = (
    REPOSITORY_ROOT / "results" / "synthetic_video_public_pilot_preflight.json"
)


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _load_work_order(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _merge_family_work_orders(run_root: Path) -> dict:
    paths = sorted((run_root / "families").glob("*/work_order.json"))
    if not paths:
        raise FileNotFoundError(f"no family work orders found under {run_root / 'families'}")
    orders = [_load_work_order(path) for path in paths]
    protocol_hashes = {order["protocol_sha256"] for order in orders}
    profiles = {order["profile"] for order in orders}
    run_ids = {order["run_id"] for order in orders}
    if len(protocol_hashes) != 1 or len(profiles) != 1 or len(run_ids) != 1:
        raise ValueError("family work orders do not share protocol, profile, and run ID")
    merged = {
        **{key: value for key, value in orders[0].items() if key not in {"families", "attempts", "work_order_sha256"}},
        "families": [family for family in ("wan", "ltx") if any(family in order["families"] for order in orders)],
        "attempts": [],
    }
    for path, order in zip(paths, orders, strict=True):
        family_root = path.parent
        prefix = family_root.relative_to(run_root)
        for attempt in order["attempts"]:
            merged_attempt = json.loads(json.dumps(attempt))
            merged_attempt["paths"] = {
                key: str(prefix / value) for key, value in merged_attempt["paths"].items()
            }
            merged["attempts"].append(merged_attempt)
    family_rank = {"wan": 0, "ltx": 1}
    scene_rank = {
        scene_id: index
        for index, scene_id in enumerate(
            ["pick_up", "transfer", "occlusion_persistence", "action_transition"]
        )
    }
    merged["attempts"].sort(
        key=lambda item: (
            scene_rank[item["scene_id"]],
            family_rank[item["family"]],
            item["seed"],
            item["attempt_number"],
        )
    )
    merged["work_order_sha256"] = canonical_json_sha256(merged)
    (run_root / "work_order.json").write_bytes(canonical_json_bytes(merged))
    return merged


def command_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    profile_counts = {}
    for profile_id, profile in config["profiles"].items():
        profile_counts[profile_id] = (
            len(profile["families"]) * len(profile["scene_ids"]) * len(profile["seeds"])
            if profile["execute_generation"]
            else 0
        )
    _json(
        {
            "status": "PASS",
            "pilot_id": config["pilot_id"],
            "protocol_sha256": canonical_json_sha256(config),
            "profiles": profile_counts,
            "privacy_boundary": "public-only; ChildLens and BabyView derivatives prohibited",
        }
    )


def command_compile(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    work_order = compile_work_order(config, args.profile, args.run_id, family=args.family)
    run_root = Path(args.run_root or REPOSITORY_ROOT / "runs" / "synthetic_video_public_pilot" / args.run_id)
    path = write_work_order(run_root, config, work_order)
    _json(
        {
            "status": "COMPILED",
            "path": str(path),
            "attempt_count": len(work_order["attempts"]),
            "protocol_sha256": work_order["protocol_sha256"],
            "work_order_sha256": work_order["work_order_sha256"],
        }
    )


def command_show_model_command(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    order = _load_work_order(args.work_order)
    attempt = next(
        (item for item in order["attempts"] if item["attempt_id"] == args.attempt_id),
        None,
    )
    if attempt is None:
        raise ValueError(f"unknown attempt ID {args.attempt_id!r}")
    spec = build_model_command(
        config,
        attempt,
        source_root=args.source_root,
        weights_root=args.weights_root,
        raw_output=args.raw_output,
        python_executable=args.python,
        uv_executable=args.uv,
    )
    _json({"cwd": spec.cwd, "argv": list(spec.argv), "shell_preview": spec.shell_preview()})


def command_retry(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    source = json.loads(Path(args.initial_attempt).read_text())
    retry = compile_retry(config, source, args.corrective_category)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(retry))
    _json({"status": "COMPILED", "attempt_id": retry["attempt_id"], "path": str(output)})


def command_mux(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    mux_modular_audio(
        config,
        args.scene_id,
        raw_video=args.raw_video,
        tts_wav=args.tts_wav,
        video_only=args.video_only,
        final_video=args.final_video,
        ffmpeg_executable=args.ffmpeg,
    )
    summary = media_summary(args.final_video, args.ffprobe)
    _json({"summary": summary, "conformance_errors": validate_final_media(config, summary)})


def command_inspect(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    summary = media_summary(args.video, args.ffprobe)
    _json({"summary": summary, "conformance_errors": validate_final_media(config, summary)})


def command_gallery(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_root = Path(args.run_root).resolve()
    work_order_path = run_root / "work_order.json"
    work_order = (
        _load_work_order(work_order_path)
        if work_order_path.exists()
        else _merge_family_work_orders(run_root)
    )
    gallery = render_gallery(config, work_order, run_root, output_path=args.output)
    _json({"status": "READY", "gallery": str(gallery), "attempt_count": len(work_order["attempts"])})


def command_cloud_plan(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    preflight = json.loads(Path(args.preflight_result).read_text())
    protocol_sha256 = canonical_json_sha256(config)
    if preflight["status"] != "PASS" or preflight["protocol_sha256"] != protocol_sha256:
        raise ValueError("cloud plan requires a PASS preflight for the current protocol hash")
    revision = preflight["nursery_commit"]
    script_url = (
        "https://raw.githubusercontent.com/rishisim/nursery/"
        f"{revision}/scripts/run_synthetic_video_hf_job.py"
    )
    cloud = config["cloud"]
    specs = []
    for family in ("wan", "ltx"):
        script_args = [
            "--nursery-revision",
            revision,
            "--expected-protocol-sha256",
            protocol_sha256,
            "--family",
            family,
            "--profile",
            "preview",
            "--run-id",
            args.run_id,
            "--output-repository",
            args.output_repository or cloud["output_repository_default"],
        ]
        shell = [
            "hf",
            "jobs",
            "uv",
            "run",
            "--detach",
            "--flavor",
            cloud["hardware_flavor"],
            "--timeout",
            cloud["gpu_timeout_per_family"],
            "--secrets",
            "HF_TOKEN",
            script_url,
            *script_args,
        ]
        specs.append(
            {
                "family": family,
                "operation": "uv",
                "script": script_url,
                "script_args": script_args,
                "flavor": cloud["hardware_flavor"],
                "timeout": cloud["gpu_timeout_per_family"],
                "secrets": ["HF_TOKEN"],
                "maximum_charge_usd": cloud["maximum_gpu_charge_per_family_usd"],
                "shell_preview": shlex.join(shell),
            }
        )
    _json(
        {
            "status": "READY_REQUIRES_EXPLICIT_SPEND_CONFIRMATION",
            "run_id": args.run_id,
            "protocol_sha256": protocol_sha256,
            "validated_nursery_commit": revision,
            "output_repository": args.output_repository
            or cloud["output_repository_default"],
            "jobs": specs,
            "maximum_preview_gpu_charge_usd": cloud[
                "maximum_preview_gpu_charge_usd"
            ],
            "launch_order": ["wan", "ltx"],
        }
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = root.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate the frozen public-only contract")
    validate.set_defaults(func=command_validate)

    compile_parser = commands.add_parser("compile", help="compile a prospective work order")
    compile_parser.add_argument("--profile", choices=("cloud_preflight", "preview", "formal"), default="preview")
    compile_parser.add_argument("--run-id", required=True)
    compile_parser.add_argument("--family", choices=("wan", "ltx"))
    compile_parser.add_argument("--run-root")
    compile_parser.set_defaults(func=command_compile)

    model_command = commands.add_parser("show-model-command", help="print, but do not execute, one model command")
    model_command.add_argument("--work-order", required=True)
    model_command.add_argument("--attempt-id", required=True)
    model_command.add_argument("--source-root", required=True)
    model_command.add_argument("--weights-root", required=True)
    model_command.add_argument("--raw-output", required=True)
    model_command.add_argument("--python", default="python")
    model_command.add_argument("--uv", default="uv")
    model_command.set_defaults(func=command_show_model_command)

    retry = commands.add_parser("retry", help="compile the one allowed formal corrective retry")
    retry.add_argument("--initial-attempt", required=True)
    retry.add_argument(
        "--corrective-category",
        required=True,
        choices=("hand", "identity", "camera", "transition", "referent_timing", "safety"),
    )
    retry.add_argument("--output", required=True)
    retry.set_defaults(func=command_retry)

    mux = commands.add_parser("mux", help="strip native audio and mux the frozen modular TTS track")
    mux.add_argument("--scene-id", required=True)
    mux.add_argument("--raw-video", required=True)
    mux.add_argument("--tts-wav", required=True)
    mux.add_argument("--video-only", required=True)
    mux.add_argument("--final-video", required=True)
    mux.add_argument("--ffmpeg", default="ffmpeg")
    mux.add_argument("--ffprobe", default="ffprobe")
    mux.set_defaults(func=command_mux)

    inspect = commands.add_parser("inspect", help="inspect final media conformance")
    inspect.add_argument("--video", required=True)
    inspect.add_argument("--ffprobe", default="ffprobe")
    inspect.set_defaults(func=command_inspect)

    gallery = commands.add_parser("gallery", help="build a side-by-side local review gallery")
    gallery.add_argument("--run-root", required=True)
    gallery.add_argument("--output")
    gallery.set_defaults(func=command_gallery)

    cloud_plan = commands.add_parser(
        "cloud-plan",
        help="emit exact paid job specs without launching or spending money",
    )
    cloud_plan.add_argument("--run-id", required=True)
    cloud_plan.add_argument(
        "--preflight-result",
        default=str(DEFAULT_PREFLIGHT_RESULT),
    )
    cloud_plan.add_argument("--output-repository")
    cloud_plan.set_defaults(func=command_cloud_plan)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
