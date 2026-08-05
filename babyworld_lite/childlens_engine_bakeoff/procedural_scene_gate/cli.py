from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import OUTPUT_ROOT, load_frozen_config, validate_frozen_config, write_frozen_bundle
from .independent_qa import audit_run, write_report
from .runner import encode_videos, prepare_project, run_unity_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical Unity procedural clothed embodiment scene gate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-contract", help="validate the frozen A-gate contracts")
    validate.set_defaults(handler=_validate)
    freeze = subparsers.add_parser("freeze", help="write the frozen 3x3 contract matrix to an ignored run root")
    freeze.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    freeze.set_defaults(handler=_freeze)
    prepare = subparsers.add_parser("prepare-project", help="stage verified public assets and canonical sources under the ignored run root")
    prepare.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    prepare.set_defaults(handler=_prepare)
    run = subparsers.add_parser("run-episode", help="run one frozen episode in a fresh Unity process")
    run.add_argument("episode_id")
    run.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    run.add_argument("--capture-mode", choices=("all", "qualification", "none"), default="all")
    run.add_argument(
        "--stage",
        choices=("garment_sweep", "motion_camera", "bimanual_cell", "integrated"),
        default="integrated",
    )
    run.set_defaults(handler=_run_episode)
    encode = subparsers.add_parser("encode", help="encode registered frame sequences for one frozen episode")
    encode.add_argument("episode_id")
    encode.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    encode.add_argument(
        "--stage",
        choices=("garment_sweep", "motion_camera", "bimanual_cell", "integrated"),
        default="integrated",
    )
    encode.set_defaults(handler=_encode)
    qa = subparsers.add_parser("qa", help="run independent fail-closed A-to-F QA on one stage run root")
    qa.add_argument("run_root", type=Path)
    qa.add_argument("--prior-unity-audition", type=Path)
    qa.add_argument("--prior-corrected-bimanual", type=Path)
    qa.add_argument("--output", type=Path)
    qa.set_defaults(handler=_qa)
    return parser


def _validate(_: argparse.Namespace) -> int:
    config = load_frozen_config()
    validate_frozen_config(config)
    print(json.dumps({"schema": config["schema"], "valid": True}, sort_keys=True))
    return 0


def _freeze(args: argparse.Namespace) -> int:
    print(json.dumps(write_frozen_bundle(args.output_root), indent=2, sort_keys=True))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    print(json.dumps(prepare_project(args.output_root), indent=2, sort_keys=True))
    return 0


def _run_episode(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            run_unity_episode(
                args.episode_id,
                output_root=args.output_root,
                capture_mode=args.capture_mode,
                stage=args.stage,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _encode(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            encode_videos(args.episode_id, args.output_root, stage=args.stage),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _qa(args: argparse.Namespace) -> int:
    report = audit_run(
        args.run_root,
        prior_unity_audition=args.prior_unity_audition,
        prior_corrected_bimanual=args.prior_corrected_bimanual,
    )
    destination = args.output or (args.run_root / "independent_qa_report.json")
    write_report(report, destination)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qa_decision"] == "PASS" else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)
