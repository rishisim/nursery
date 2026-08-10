#!/usr/bin/env python3
"""Resolve Juno storage and protect disposable Codex worktrees."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
SPEC_PATH = WORKSPACE / "spec.json"
MARKER_NAME = ".egobabyvlm_clip_plus_storage.json"
DISPOSABLE_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "cache", "caches", "logs", "htmlcov",
}
DISPOSABLE_NAMES = {".DS_Store", ".coverage"}
DISPOSABLE_SUFFIXES = {".pyc", ".pyo", ".log"}


class StorageError(RuntimeError):
    """A storage or worktree invariant is not satisfied."""


def git(*args: str, cwd: Path = WORKSPACE, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def policy() -> dict[str, object]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))["storage_policy"]


def tier_config(tier: str) -> dict[str, object]:
    value = policy()[tier]
    if not isinstance(value, dict):
        raise StorageError(f"Invalid {tier} storage configuration")
    return value


def storage_root(tier: str) -> Path:
    config = tier_config(tier)
    override_name = str(config["environment_override"])
    value = os.environ.get(override_name, str(config["root"]))
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise StorageError(f"{override_name} must be absolute: {root}")
    return root


def tier_directories(tier: str) -> tuple[str, ...]:
    return tuple(str(value) for value in tier_config(tier)["directories"])


def ssh_target() -> str:
    remote = policy()["remote"]
    if not isinstance(remote, dict):
        raise StorageError("Invalid Juno remote configuration")
    return f"{remote['user']}@{remote['host']}"


def ssh_run(*parts: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    command = " ".join(shlex.quote(part) for part in parts)
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ssh_target(), command],
        check=True, text=True, input=input_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def expected_marker(tier: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "project": "egobabyvlm_clip_plus",
        "tier": tier,
        "root": str(storage_root(tier)),
        "directories": list(tier_directories(tier)),
        "owner_only_required": True,
    }


def initialize_tier(tier: str) -> None:
    root = storage_root(tier)
    directories = tier_directories(tier)
    marker = json.dumps(expected_marker(tier), sort_keys=True)
    remote_script = (
        "import json, os, pathlib, sys; "
        "root=pathlib.Path(sys.argv[1]); dirs=json.loads(sys.argv[2]); marker=json.loads(sys.argv[3]); "
        "root.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(root, 0o700); "
        "[(root/name).mkdir(exist_ok=True, mode=0o700) for name in dirs]; "
        "[os.chmod(root/name, 0o700) for name in dirs]; "
        "target=root/sys.argv[4]; target.write_text(json.dumps(marker, indent=2)+'\\n'); "
        "os.chmod(target, 0o600)"
    )
    ssh_run(
        "python3", "-c", remote_script, str(root), json.dumps(directories), marker, MARKER_NAME
    )


def check_tier(tier: str) -> None:
    root = storage_root(tier)
    directories = tier_directories(tier)
    remote_script = (
        "import json, os, pathlib, stat, sys; "
        "root=pathlib.Path(sys.argv[1]); dirs=json.loads(sys.argv[2]); marker=root/sys.argv[3]; "
        "result={'root_exists':root.is_dir(),'root_mode':stat.S_IMODE(root.stat().st_mode) if root.exists() else None,"
        "'missing':[name for name in dirs if not (root/name).is_dir()],"
        "'marker':json.loads(marker.read_text()) if marker.is_file() else None}; "
        "print(json.dumps(result))"
    )
    result = ssh_run("python3", "-c", remote_script, str(root), json.dumps(directories), MARKER_NAME)
    observed = json.loads(result.stdout)
    if not observed["root_exists"] or observed["root_mode"] != 0o700:
        raise StorageError(f"Juno {tier} root is missing or not mode 0700: {root}")
    if observed["missing"]:
        raise StorageError(f"Juno {tier} directories are missing: {observed['missing']}")
    if observed["marker"] != expected_marker(tier):
        raise StorageError(f"Juno {tier} marker does not match the frozen policy: {root}")


def git_path(flag: str) -> Path:
    value = git("rev-parse", "--path-format=absolute", flag).stdout.strip()
    if not value:
        raise StorageError(f"Git did not return a path for {flag}")
    return Path(value).resolve()


def is_disposable(path: Path) -> bool:
    return (
        path.name in DISPOSABLE_NAMES
        or path.suffix in DISPOSABLE_SUFFIXES
        or any(part in DISPOSABLE_PARTS for part in path.parts)
    )


def safe_to_remove() -> None:
    check_tier("durable")
    check_tier("scratch")
    current = git_path("--show-toplevel")
    common_dir = git_path("--git-common-dir")
    if common_dir.name != ".git":
        raise StorageError(f"Expected a non-bare primary .git directory, got {common_dir}")
    if current == common_dir.parent.resolve():
        raise StorageError("Refusing to approve removal of the primary checkout")

    status = git("status", "--porcelain=v1", "--untracked-files=all", cwd=current).stdout
    if status.strip():
        raise StorageError(f"Linked worktree has tracked or untracked changes:\n{status.rstrip()}")

    ignored_raw = git(
        "ls-files", "--others", "--ignored", "--exclude-standard", "-z", cwd=current
    ).stdout
    unexpected = [Path(item) for item in ignored_raw.split("\0") if item and not is_disposable(Path(item))]
    if unexpected:
        preview = "\n".join(f"  {path}" for path in unexpected[:20])
        raise StorageError(f"Linked worktree has non-disposable ignored files:\n{preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    root_parser = subparsers.add_parser("root", help="print a frozen Juno root")
    root_parser.add_argument("tier", choices=("durable", "scratch"))
    path_parser = subparsers.add_parser("path", help="print a Juno subdirectory")
    path_parser.add_argument("tier", choices=("durable", "scratch"))
    path_parser.add_argument("name")
    subparsers.add_parser("init-juno", help="initialize both owner-only Juno roots")
    subparsers.add_parser("check-juno", help="validate both owner-only Juno roots")
    subparsers.add_parser("safe-to-remove", help="verify that a linked worktree is disposable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "root":
            print(storage_root(args.tier))
        elif args.command == "path":
            if args.name not in tier_directories(args.tier):
                raise StorageError(f"Unknown {args.tier} directory: {args.name}")
            print(storage_root(args.tier) / args.name)
        elif args.command == "init-juno":
            for tier in ("durable", "scratch"):
                initialize_tier(tier)
            print(f"Initialized owner-only Juno storage on {ssh_target()}")
        elif args.command == "check-juno":
            for tier in ("durable", "scratch"):
                check_tier(tier)
            print(f"Juno storage is valid on {ssh_target()}")
        elif args.command == "safe-to-remove":
            safe_to_remove()
            print("Linked worktree is safe to remove")
        return 0
    except (StorageError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as error:
        print(f"storage error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
