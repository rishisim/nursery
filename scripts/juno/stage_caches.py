"""Stage the frozen Step 9 caches for training on Juno."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shlex
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
    # Prefer node-local NVMe (/lwork); some H200 nodes only have RAM-backed /tmp,
    # whose pages count toward the job's --mem.
    global LOCAL
    errors = []
    for candidate, kinds in ((Path("/lwork"), ("block",)), (Path("/tmp"), ("tmpfs",))):
        if not candidate.is_dir() or not os.access(candidate, os.W_OK | os.X_OK):
            errors.append(f"{candidate} is missing or unwritable")
            continue
        source, fstype = subprocess.check_output(
            ["findmnt", "--noheadings", "--first-only", "--output", "SOURCE,FSTYPE",
             "--target", str(candidate)], text=True,
        ).split()[:2]
        is_block = source.startswith("/dev/") and stat.S_ISBLK(Path(source).stat().st_mode)
        if ("block" in kinds and is_block) or fstype in kinds:
            LOCAL = candidate
            return shutil.disk_usage(LOCAL).free
        errors.append(f"{candidate} is not local ({source} {fstype})")
    raise OSError("; ".join(errors))


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


def run_packed(runs, mirror=None):
    wrapper = (
        "import sys; from pathlib import Path; "
        "from scripts.juno.stage_caches import run_training; "
        "mirror = sys.argv.pop(1); run_training(Path(mirror) if mirror else None)"
    )
    command = [sys.executable, "-B", "-u", "-c", wrapper, str(mirror or "")]
    children = []
    interrupted = 0

    def stop_children(signum, _frame):
        nonlocal interrupted
        interrupted = interrupted or signum
        for _, child in children:
            child.terminate()

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    job_id = os.environ["SLURM_JOB_ID"]
    try:
        for task, train_args in runs:
            if interrupted:
                break
            log = Path(f"outputs/runs/u3-packed-{job_id}-task{task}.out")
            print(f"U3 task {task} log: {log}")
            with log.open("w") as output:
                child = subprocess.Popen([*command, *train_args], stdout=output,
                                         stderr=subprocess.STDOUT)
                children.append((task, child))
            if interrupted:
                # A signal may have arrived before Popen returned the new child.
                child.terminate()

        status = 0
        for task, child in children:
            code = child.wait()
            if code:
                print(f"U3 task {task} exited with status {code}", file=sys.stderr)
                status = 1
        return 128 + interrupted if interrupted else status
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            child.wait()


def stop(signum, _frame):
    # Let the first signal unwind the staging context, even if Slurm or the
    # parent shell sends another signal during cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    starts = [index for index, arg in enumerate(argv) if arg == "--task"]
    packed = bool(starts)
    dry_run = os.environ.get("STAGE_DRY_RUN") == "1"
    if packed:
        parser.parse_args(argv[:starts[0]])
        dry_run = dry_run or "--dry-run" in argv
        parser.add_argument("--task", type=int, required=True)
        parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--condition", required=True,
                        choices=("V", "V+G", "V+G+B", "CLIP-Global", "BabyLLaVA-Nursery"))
    runs = []
    conditions = set()
    if packed:
        seen = set()
        for start, end in zip(starts, [*starts[1:], len(argv)]):
            args, train_args = parser.parse_known_args(argv[start:end])
            if not 0 <= args.task <= 134 or args.task in seen:
                parser.error(f"Invalid or duplicate U3 task ID: {args.task}")
            seen.add(args.task)
            train_args = ["--condition", args.condition, *train_args, "--out-dir", str(args.out_dir)]
            if dry_run:
                print(f"U3 task {args.task}: {shlex.join(train_args)}")
            if (args.out_dir / "metrics.json").is_file():
                print(f"Skipping completed run {args.out_dir}")
                continue
            if args.out_dir.exists() or args.out_dir.is_symlink():
                parser.error(f"Output path exists without metrics.json: {args.out_dir}")
            runs.append((args.task, train_args))
            conditions.add(args.condition)
        if not runs:
            return
    else:
        args, train_args = parser.parse_known_args(argv)
        dry_run = dry_run or args.dry_run
        conditions.add(args.condition)
        sys.argv = [sys.argv[0], "--condition", args.condition, *train_args]
    if not dry_run and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use the Slurm template or --dry-run")

    study = STUDY.resolve(strict=True)
    protocol = json.loads((study / "protocol.json").read_text())
    cache_dirs = [(Path(protocol["inputs"]) / "train_cache", Path("inputs/train_cache"))]
    if conditions & {"V+G", "V+G+B"}:
        cache_dirs.append((Path(protocol["gaze"]["closeup_cache"]), Path("closeup_cache")))
        readable_file(Path(protocol["gaze"]["points"]))
    readable_file(Path(protocol["gaze"]["partitions"]))
    readable_file(Path(protocol["episodes"]) / "frames.parquet")
    if "V+G+B" in conditions and not Path(protocol["physical"]).is_dir():
        raise OSError(f"Missing physical inputs: {protocol['physical']}")

    caches = []
    staging_root = "<node-local>" if packed else str(LOCAL)
    for source, relative in cache_dirs:
        for name in ("emb.f16.npy", "index.parquet"):
            path = (source / name).resolve(strict=True)
            size = readable_file(path)
            caches.append((path, relative / name, size))
            print(f"{path} ({size:,} bytes) -> {staging_root}/nursery-u1-*/{relative / name}")
    total = sum(size for _, _, size in caches)
    required = (total * 11 + 9) // 10
    print(f"{', '.join(sorted(conditions))}: {total:,} bytes total; "
          f"{required:,} bytes required with 10% margin")

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
                if packed:
                    return run_packed(runs, mirror)
                run_training(mirror)
                return

    if packed:
        return run_packed(runs)
    run_training()


if __name__ == "__main__":
    sys.exit(main())
