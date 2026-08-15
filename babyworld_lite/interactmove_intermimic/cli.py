"""Thin JSON CLI for the InteractMove--InterMimic Stage 1--3 contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .activity import (
    activity_spec_sha256,
    build_embodiedgen_request,
    load_activity_spec,
)
from .common import ContractError, content_sha256, read_json, write_json_atomic
from .embodiedgen import compile_scene_bundle, validate_scene_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = REPOSITORY_ROOT / "configs/interactmove_intermimic.json"
IN_REPOSITORY_RUN_ROOT = REPOSITORY_ROOT / "runs/interactmove_intermimic"


def _emit(value: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=stream,
    )


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        _emit(
            {"error": "ArgumentError", "message": message, "status": "error"},
            stream=sys.stderr,
        )
        raise SystemExit(2)


def _require_output_policy(path: Path) -> None:
    resolved = path.resolve(strict=False)
    repository = REPOSITORY_ROOT.resolve(strict=True)
    try:
        resolved.relative_to(repository)
    except ValueError:
        return
    allowed = IN_REPOSITORY_RUN_ROOT.resolve(strict=False)
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ContractError(
            "outputs inside the repository must be under "
            f"{IN_REPOSITORY_RUN_ROOT}"
        ) from exc


def _require_output_target(path: Path, *, overwrite: bool) -> None:
    _require_output_policy(path)
    if path.exists() and not overwrite:
        raise ContractError(f"refusing to overwrite existing output: {path}")


def _upstream(protocol_path: Path) -> tuple[str, str, str]:
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") != "interactmove_intermimic":
        raise ContractError(
            "protocol_id must be 'interactmove_intermimic'"
        )
    try:
        upstream = protocol["upstream"]["embodiedgen"]
        version = upstream["version"]
        commit = upstream["commit"]
        native_profile = upstream["native_profile"]
    except (KeyError, TypeError) as exc:
        raise ContractError(
            "protocol must define upstream.embodiedgen version, commit, and "
            "native_profile"
        ) from exc
    for name, value in (
        ("version", version),
        ("commit", commit),
        ("native_profile", native_profile),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ContractError(
                f"protocol upstream.embodiedgen.{name} must be a non-empty string"
            )
    return version, commit, native_profile


def _backgrounds(value: str) -> list[str]:
    if value.lstrip().startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid --background-list JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ContractError("--background-list JSON must be an array")
        backgrounds = parsed
    else:
        backgrounds = [item.strip() for item in value.split(",")]
    if not backgrounds or any(
        not isinstance(item, str) or not item for item in backgrounds
    ):
        raise ContractError(
            "--background-list must contain non-empty strings (comma-separated "
            "or a JSON array)"
        )
    return backgrounds


def _validate_activity(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_activity_spec(args.spec)
    result: dict[str, Any] = {
        "activity_id": spec["activity_id"],
        "command": "validate-activity",
        "status": "ok",
    }
    if args.print_digest:
        result["activity_spec_sha256"] = activity_spec_sha256(spec)
    return result


def _prepare_embodiedgen_request(args: argparse.Namespace) -> dict[str, Any]:
    _require_output_target(args.output, overwrite=args.overwrite)
    spec = load_activity_spec(args.spec)
    version, commit, native_profile = _upstream(args.protocol)
    request = build_embodiedgen_request(
        spec,
        upstream_version=version,
        upstream_commit=commit,
        background_list=_backgrounds(args.background_list),
    )
    if request.get("native_profile") != native_profile:
        raise ContractError(
            "activity request native_profile does not match the protocol config"
        )
    write_json_atomic(args.output, request, overwrite=args.overwrite)
    return {
        "artifact_sha256": content_sha256(request),
        "command": "prepare-embodiedgen-request",
        "output": str(args.output.resolve(strict=False)),
        "status": "ok",
    }


def _compile_scene_bundle(args: argparse.Namespace) -> dict[str, Any]:
    _require_output_target(args.output, overwrite=args.overwrite)
    spec = load_activity_spec(args.spec)
    request = read_json(args.request)
    version, commit, native_profile = _upstream(args.protocol)
    bundle = compile_scene_bundle(
        args.scene_root,
        args.layout,
        spec,
        upstream_version=version,
        upstream_commit=commit,
        native_profile=native_profile,
        embodiedgen_request=request,
    )
    validate_scene_bundle(bundle)
    write_json_atomic(args.output, bundle, overwrite=args.overwrite)
    return {
        "artifact_sha256": content_sha256(bundle),
        "command": "compile-scene-bundle",
        "output": str(args.output.resolve(strict=False)),
        "status": "ok",
    }


def _validate_scene_bundle(args: argparse.Namespace) -> dict[str, Any]:
    bundle = read_json(args.bundle)
    validate_scene_bundle(bundle)
    result: dict[str, Any] = {
        "command": "validate-scene-bundle",
        "status": "ok",
    }
    if args.print_digest:
        result["scene_bundle_sha256"] = content_sha256(bundle)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(prog="interactmove-intermimic")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_activity = commands.add_parser(
        "validate-activity", help="validate one Stage 1 activity specification"
    )
    validate_activity.add_argument("spec", type=Path, metavar="SPEC")
    validate_activity.add_argument("--print-digest", action="store_true")
    validate_activity.set_defaults(handler=_validate_activity)

    prepare_request = commands.add_parser(
        "prepare-embodiedgen-request",
        help="prepare one deterministic EmbodiedGen request",
    )
    prepare_request.add_argument("spec", type=Path, metavar="SPEC")
    prepare_request.add_argument("--background-list", required=True)
    prepare_request.add_argument("--output", required=True, type=Path)
    prepare_request.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    prepare_request.add_argument("--overwrite", action="store_true")
    prepare_request.set_defaults(handler=_prepare_embodiedgen_request)

    compile_bundle = commands.add_parser(
        "compile-scene-bundle", help="compile a canonical Stage 3 scene bundle"
    )
    compile_bundle.add_argument("spec", type=Path, metavar="SPEC")
    compile_bundle.add_argument("--request", required=True, type=Path)
    compile_bundle.add_argument("--scene-root", required=True, type=Path)
    compile_bundle.add_argument("--layout", default="layout.json")
    compile_bundle.add_argument("--output", required=True, type=Path)
    compile_bundle.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    compile_bundle.add_argument("--overwrite", action="store_true")
    compile_bundle.set_defaults(handler=_compile_scene_bundle)

    validate_bundle = commands.add_parser(
        "validate-scene-bundle", help="validate one canonical scene bundle"
    )
    validate_bundle.add_argument("bundle", type=Path, metavar="BUNDLE")
    validate_bundle.add_argument("--print-digest", action="store_true")
    validate_bundle.set_defaults(handler=_validate_scene_bundle)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
    except (ContractError, OSError) as exc:
        _emit(
            {
                "error": type(exc).__name__,
                "message": str(exc),
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 2
    _emit(result)
    return 0


__all__ = ["build_parser", "main"]
