"""Prepare the exact pinned upstream checkout in a disposable directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    parser.add_argument("--nursery-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    nursery_root = Path(args.nursery_root).resolve()
    destination = Path(args.destination).resolve()
    if destination.exists():
        raise FileExistsError(f"destination must not exist: {destination}")
    frozen = json.loads((nursery_root / "configs/egobaby_cuda_preflight.json").read_text())
    run(["git", "clone", "--filter=blob:none", frozen["upstream"]["repository"], str(destination)])
    run(["git", "checkout", "--detach", frozen["upstream"]["commit"]], cwd=destination)
    run(["git", "apply", "--check", str(nursery_root / "patches/egobabyvlm_shared_prior.patch")], cwd=destination)
    run(["git", "apply", str(nursery_root / "patches/egobabyvlm_shared_prior.patch")], cwd=destination)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
    print(json.dumps({"upstream_commit": actual, "patch_applied": True}, sort_keys=True))


if __name__ == "__main__":
    main()
