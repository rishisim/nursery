"""Stage the frozen Step 9 caches for one U1 training process on Juno."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time


STUDY = Path("outputs/runs/step9_acquisition_curves/frozen")
LOCAL = Path("/lwork")


def readable_file(path):
    if not path.is_file() or not os.access(path, os.R_OK):
        raise OSError(f"Unreadable frozen input: {path}")
    return path.stat().st_size


def local_free():
    if not LOCAL.is_dir() or not os.access(LOCAL, os.W_OK | os.X_OK):
        raise OSError(f"{LOCAL} is missing or unwritable")
    source = subprocess.check_output(
        ["findmnt", "--noheadings", "--first-only", "--output", "SOURCE", "--target", str(LOCAL)],
        text=True,
    ).strip()
    if not source.startswith("/dev/") or not stat.S_ISBLK(Path(source).stat().st_mode):
        raise OSError(f"{LOCAL} is not backed by a local block device ({source})")
    return shutil.disk_usage(LOCAL).free


def stage(study, protocol, caches, destination):
    started = time.monotonic()
    for source, relative, size in caches:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if target.stat().st_size != size:
            raise OSError(f"Cache size changed during staging: {source}")

    mirror = destination / "frozen"
    mirror.mkdir()
    for entry in study.iterdir():
        if entry.name != "protocol.json":
            (mirror / entry.name).symlink_to(entry)
    protocol["inputs"] = str(destination / "inputs")
    if (destination / "closeup_cache").is_dir():
        protocol["gaze"]["closeup_cache"] = str(destination / "closeup_cache")
    (mirror / "protocol.json").write_text(json.dumps(protocol) + "\n")
    print(f"Staged U1 caches in {time.monotonic() - started:.1f}s; using {mirror}")
    return mirror


def run_training(study=None):
    from nursery.unified import evaluate, train

    if study is not None:
        evaluate.STUDY = study
    train.main()


def stop(signum, _frame):
    # Let the first signal unwind the staging context, even if Slurm or the
    # parent shell sends another signal during cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--condition", required=True, choices=("V", "V+G", "V+G+B"))
    args, train_args = parser.parse_known_args()
    dry_run = args.dry_run or os.environ.get("STAGE_DRY_RUN") == "1"
    if not dry_run and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use the Slurm template or --dry-run")

    study = STUDY.resolve(strict=True)
    protocol = json.loads((study / "protocol.json").read_text())
    cache_dirs = [(Path(protocol["inputs"]) / "train_cache", Path("inputs/train_cache"))]
    if args.condition != "V":
        cache_dirs.append((Path(protocol["gaze"]["closeup_cache"]), Path("closeup_cache")))
        readable_file(Path(protocol["gaze"]["points"]))
    readable_file(Path(protocol["gaze"]["partitions"]))
    readable_file(Path(protocol["episodes"]) / "frames.parquet")
    if args.condition == "V+G+B" and not Path(protocol["physical"]).is_dir():
        raise OSError(f"Missing physical inputs: {protocol['physical']}")

    caches = []
    for source, relative in cache_dirs:
        for name in ("emb.f16.npy", "index.parquet"):
            path = (source / name).resolve(strict=True)
            size = readable_file(path)
            caches.append((path, relative / name, size))
            print(f"{path} ({size:,} bytes) -> {LOCAL}/nursery-u1-*/{relative / name}")
    total = sum(size for _, _, size in caches)
    required = (total * 11 + 9) // 10
    print(f"{args.condition}: {total:,} bytes total; {required:,} bytes required with 10% margin")
    sys.argv = [sys.argv[0], "--condition", args.condition, *train_args]

    try:
        free = local_free()
        print(f"{LOCAL}: {free:,} bytes available on this host")
        if free < required:
            raise OSError(f"{LOCAL} has insufficient free space")
    except (OSError, subprocess.CalledProcessError) as error:
        if dry_run:
            print(f"Dry run: {error}; the job checks its compute node at startup.")
            return
        print(f"WARNING: {error}; using original /work cache paths.", file=sys.stderr)
    else:
        if dry_run:
            print("Dry run: would stage these files and use a job-private protocol.")
            return
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        with ExitStack() as cleanup:
            try:
                destination = cleanup.enter_context(tempfile.TemporaryDirectory(
                    prefix=f"nursery-u1-{os.environ['SLURM_JOB_ID']}-", dir=LOCAL,
                ))
                mirror = stage(study, protocol, caches, Path(destination))
            except OSError as error:
                print(f"WARNING: staging failed ({error}); using original /work cache paths.",
                      file=sys.stderr)
            else:
                run_training(mirror)
                return

    run_training()


if __name__ == "__main__":
    main()
