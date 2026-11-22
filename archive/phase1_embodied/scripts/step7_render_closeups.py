"""Stage the missing Step 7 gaze views and assemble an append-only cache."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import monotonic

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nursery.common import REPO, _output_directory
from nursery.learner.acquisition import _read, _sha, _write
from nursery.learner.embed import embed_closeups
from nursery.simulator.closeup import _configuration
from nursery.simulator.render import render_job


RUN = REPO / "outputs/runs/step7_word_types"
FROZEN = RUN / "frozen"
REPLAYS = REPO / "outputs/renders/child_gaze_closeup/sessions"
EPISODES = REPO / "outputs/renders/clean_856"
FILES = ("index.parquet", "emb.f16.npy")
KEY = ["video_id", "frame_idx"]


def index_keys(table):
    return list(table[KEY].itertuples(index=False, name=None))


def missing_frames(cached, installed=None):
    if installed is None:
        coverage = _read(FROZEN / "closeup_coverage.json")
        frames = [(video, int(frame)) for video, frame in
                  (key.rsplit("|", 1) for key in coverage["missing_frame_keys"])]
        count = coverage["missing_closeup_frames"]
    else:
        index = pd.read_parquet(installed / FILES[0])
        pd.testing.assert_frame_equal(index.iloc[:len(cached)], cached)
        frames = index_keys(index.iloc[len(cached):])
        count = _read(installed / "compact.json")["added_rows"]
    needed = set(frames)
    if not needed or len(needed) != count or len(needed) != len(frames):
        raise ValueError("Expected a nonempty frozen list of unique missing frames")
    if needed.intersection(index_keys(cached)):
        raise ValueError("The frozen missing-frame list overlaps the original cache")
    return sorted(frames)


def batches(frames):
    sample = frames[:50]
    if len(sample) != 50:
        raise ValueError("The frozen list has fewer than 50 missing frames")
    yield "sample", sample
    sessions = [index_keys(group) for _, group in frame_table(frames[50:]).groupby("video_id", sort=True)]
    for offset in range(0, len(sessions), 32):
        yield f"batch_{offset // 32:03d}", [key for group in sessions[offset:offset + 32] for key in group]


def equivalence_frames(cached):
    groups = [group.sort_values("frame_idx") for _, group in cached.groupby("video_id", sort=True)
              if len(group) >= 4]
    if len(groups) < 5:
        raise ValueError("Need at least five cached sessions with four close-ups each")
    positions = np.linspace(0, len(groups) - 1, 5, dtype=int)
    frames = []
    for position in positions:
        group = groups[position]
        frames.extend(index_keys(group.iloc[np.linspace(0, len(group) - 1, 4, dtype=int)]))
    return frames


def frame_table(frames):
    return pd.DataFrame(frames, columns=KEY)


def image_name(video, frame):
    return f"frames/{video}/{frame:05d}.jpg"


def render_session(destination, video, points, config):
    gaze = {image_name(video, r.frame_idx): [r.gaze_x, r.gaze_y] for r in points.itertuples()}
    expected = {Path("closeups") / image for image in gaze}
    if destination.exists():
        if {p.relative_to(destination) for p in destination.rglob("*.jpg")} != expected:
            raise ValueError(f"Unexpected images in completed session {destination}")
        return
    job = _read(REPLAYS / video / "job.json")
    job.update(mode="closeup", gaze=gaze, frame_images=sorted(gaze))
    with tempfile.TemporaryDirectory(prefix=".render-", dir=destination.parent) as attempt:
        output = Path(attempt) / "session"
        output.mkdir()
        render_job(job, config, output)
        if {p.relative_to(output) for p in output.rglob("*.jpg")} != expected:
            raise ValueError(f"Rendered image list differs from requested keys for {video}")
        output.rename(destination)


def cache_contents(path, *, compact):
    index = pd.read_parquet(path / FILES[0])
    values = np.load(path / FILES[1], mmap_mode="r", allow_pickle=False)
    shape = (len(index), 768) if compact else (len(index), 257, 768)
    dtype = np.float32 if compact else np.float16
    if (values.shape != shape or values.dtype != dtype or index.duplicated(KEY).any()
            or not np.array_equal(index.row, np.arange(len(index)))):
        raise ValueError(f"Invalid close-up cache layout at {path}")
    return index, values


def staged_values(directory, frames, *, verify=False):
    features = directory / "features"
    cache = features / "closeup_cache"
    record = _read(features / "embedding_closeup.json")
    index, values = cache_contents(cache, compact=False)
    if not record["complete"] or index_keys(index) != frames:
        raise ValueError(f"Incomplete or unexpected staged keys at {cache}")
    if verify and any(_sha(cache / name) != record["cache_sha256"][name] for name in FILES):
        raise ValueError(f"Staged embeddings changed at {cache}")
    return values


def stage(directory, frames, points, config):
    features = directory / "features"
    if features.exists():
        staged_values(directory, frames)
        return _read(features / "timing.json")

    start = monotonic()
    table = frame_table(frames).merge(points, on=KEY, how="left", validate="1:1")
    xy = table[["gaze_x", "gaze_y"]].to_numpy()
    if table.gaze_object_id.isna().any() or not ((xy >= 0) & (xy < 1)).all():
        raise ValueError("Requested view lacks valid frozen gaze coordinates")
    sessions = directory / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    reused = sum((sessions / video).exists() for video in table.video_id.unique())
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = [pool.submit(render_session, sessions / video, video, group, config)
                 for video, group in table.groupby("video_id", sort=True)]
        for task in tasks:
            task.result()
    table["image"] = [f"sessions/{v}/closeups/{image_name(v, f)}" for v, f in frames]
    table[KEY + ["image"]].to_parquet(directory / "frames.parquet", index=False)
    with tempfile.TemporaryDirectory(prefix=".embed-", dir=directory) as attempt:
        encoded = Path(attempt) / "features"
        embed_closeups(directory, encoded, device="mps", batch=32)
        timing = {"seconds": monotonic() - start, "reused_sessions": reused}
        _write(encoded / "timing.json", timing)
        encoded.rename(features)
    return timing


def compact_tokens(values):
    # Match nursery.learner.child_gaze._compact: float32 patch mean, excluding CLS.
    return np.asarray(values[:, 1:], dtype=np.float32).mean(axis=1)


def check_equivalence(frames, base, points, config):
    index, cached = cache_contents(base, compact=True)
    cached_rows = index.set_index(KEY).loc[frames, "row"]
    with tempfile.TemporaryDirectory(prefix="equivalence-", dir=_output_directory(RUN / "closeup_check")) as tmp:
        directory = Path(tmp)
        stage(directory, frames, points, config)
        actual = compact_tokens(staged_values(directory, frames)).astype(np.float64)
        expected = cached[cached_rows.to_numpy()].astype(np.float64)
        cosine = (actual * expected).sum(axis=1) / (
            np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
        print(f"{len(cosine)} compact embeddings: max cosine={cosine.max():.8f}, "
              f"mean={cosine.mean():.8f}, min={cosine.min():.8f}", flush=True)
        if not np.isfinite(cosine).all() or cosine.min() < 0.99:
            raise ValueError("A matched close-up is below the 0.99 cosine threshold")


@contextmanager
def stage_lock(staging):
    with (_output_directory(staging) / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another sample, full or install command owns the staging directory") from exc
        yield


def install_cache(base, destination, staging, plan, protocol):
    original, old_values = cache_contents(base, compact=True)
    source_hashes = {name: _sha(base / name) for name in FILES}
    provenance = protocol.get("render_provenance")
    expected = (provenance["original_cache_sha256"] if provenance is not None else
                {name: protocol["hashes"][str(base / name)] for name in FILES})
    if source_hashes != expected:
        raise ValueError("Original compact cache differs from its frozen hashes")
    additions = frame_table([key for _, frames in plan for key in frames])
    additions["row"] = np.arange(len(original), len(original) + len(additions))
    combined = pd.concat([original, additions], ignore_index=True)
    if combined.duplicated(KEY).any():
        raise ValueError("Refusing to install duplicate close-up keys")
    if destination.exists():
        index, _ = cache_contents(destination, compact=True)
        pd.testing.assert_frame_equal(index, combined)
        receipt = _read(destination / "compact.json")
        if (receipt["source_sha256"] != source_hashes
                or any(_sha(destination / name) != receipt["files_sha256"][name] for name in FILES)):
            raise ValueError("Installed cache checksum mismatch")
        return
    for name, frames in plan:
        staged_values(staging / name, frames)

    records = _output_directory(RUN / "closeup_install")
    backup = records / "index.before.parquet"
    if not backup.exists():
        with tempfile.TemporaryDirectory(prefix=".backup-", dir=records) as attempt:
            shutil.copyfile(base / FILES[0], Path(attempt) / backup.name)
            (Path(attempt) / backup.name).rename(backup)
    if _sha(backup) != source_hashes[FILES[0]]:
        raise ValueError("Saved original index backup does not match the original cache")
    with tempfile.TemporaryDirectory(prefix=".step7-compact-", dir=destination.parent) as tmp:
        pending = Path(tmp) / "cache"
        pending.mkdir()
        values = np.lib.format.open_memmap(pending / FILES[1], mode="w+",
                                           dtype=np.float32, shape=(len(combined), 768))
        values[:len(original)] = old_values
        offset = len(original)
        for name, frames in plan:
            new = staged_values(staging / name, frames, verify=True)
            for start in range(0, len(new), 128):
                block = compact_tokens(new[start:start + 128])
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite compact features in {name}")
                values[offset + start:offset + start + len(block)] = block
            offset += len(new)
        values.flush()
        del values
        combined.to_parquet(pending / FILES[0], index=False)
        _write(pending / "compact.json", {
            "source": str(base), "source_sha256": source_hashes, "staging": str(staging),
            "original_rows": len(original), "added_rows": len(additions),
            "files_sha256": {name: _sha(pending / name) for name in FILES},
        })
        pending.rename(destination)


def check_coverage(destination, protocol):
    from scripts.step7_word_types import closeup_gaps

    proposed = dict(protocol, gaze=dict(protocol["gaze"], closeup_cache=str(destination)))
    gaps = {name: set().union(*rows) for name, rows in closeup_gaps(FROZEN, proposed).items()}
    missing = set().union(*gaps.values())
    coverage = {
        "closeup_cache": str(destination), "ready": not missing,
        "missing_closeup_frames": len(missing), "missing_frame_keys": sorted(missing),
        "missing_by_panel_or_training": {name: len(keys) for name, keys in gaps.items()},
    }
    records = _output_directory(RUN / "closeup_install")
    with tempfile.TemporaryDirectory(prefix=".coverage-", dir=records) as tmp:
        pending = Path(tmp) / "closeup_coverage.json"
        _write(pending, coverage)
        pending.replace(records / pending.name)
    print(f"ready={str(coverage['ready']).lower()} missing_closeup_frames={len(missing)}", flush=True)
    if missing:
        raise ValueError("Installed cache does not cover all frozen windows")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("equivalence", "sample", "full", "install"))
    parser.add_argument("--blender", type=Path, help="Blender executable (defaults to the clean_856 manifest)")
    parser.add_argument("--dry-run", action="store_true", help="List selected keys without writing or using a GPU")
    args = parser.parse_args(argv)

    os.chdir(REPO)
    protocol = _read(FROZEN / "protocol.json")
    installed = None
    if "render_provenance" in protocol:
        installed = Path(protocol["gaze"]["closeup_cache"])
        base = Path(protocol["render_provenance"]["original_cache"])
    else:
        base = Path(protocol["gaze"]["closeup_cache"])
    staging, destination = base.with_name("step7_staging"), base.with_name("step7_compact")
    cached = pd.read_parquet(base / FILES[0])
    if args.command == "equivalence":
        plan = [("equivalence", equivalence_frames(cached))]
    else:
        plan = list(batches(missing_frames(cached, installed)))
        if args.command == "sample":
            plan = plan[:1]
    frames = [key for _, batch in plan for key in batch]

    if args.dry_run:
        print(f"{args.command}: {len(frames)} frames across {len({v for v, _ in frames})} sessions")
        for name, selected in plan:
            state = "staged" if (staging / name / "features").is_dir() else "pending"
            for video, frame in selected:
                print(f"{name} {state} {video}|{frame}")
        return

    if args.command != "install":
        _, config = _configuration(EPISODES)
        if args.blender is not None:
            config["blender"] = str(args.blender.expanduser().absolute())
        if not os.access(config["blender"], os.X_OK):
            raise ValueError(f"Blender executable is unavailable: {config['blender']}")
        if _sha(protocol["gaze"]["points"]) != protocol["hashes"][protocol["gaze"]["points"]]:
            raise ValueError("Gaze points differ from the frozen hash")
        points = pd.read_parquet(protocol["gaze"]["points"],
                                 columns=KEY + ["gaze_x", "gaze_y", "gaze_object_id"])
    if args.command == "equivalence":
        check_equivalence(frames, base, points, config)
        return
    with stage_lock(staging):
        if args.command == "install":
            install_cache(base, destination, staging, plan, protocol)
            check_coverage(destination, protocol)
            return
        complete = 0
        for name, selected in plan:
            timing = stage(staging / name, selected, points, config)
            complete += len(selected)
            print(f"{name}: {complete}/{len(frames)} frames staged", flush=True)
        if args.command == "sample":
            rate = 60 * len(frames) / timing["seconds"]
            print(f"{rate:.1f} frames/min over {timing['seconds']:.2f}s "
                  f"(previously rendered sessions: {timing['reused_sessions']})", flush=True)


if __name__ == "__main__":
    main()
