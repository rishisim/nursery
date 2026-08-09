#!/usr/bin/env python3
"""Physically materialize audit-only clips; contains no runtime discovery."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Iterable


class ClipError(RuntimeError):
    pass


def _private_regular(path: Path) -> bool:
    metadata = path.lstat()
    return stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) and metadata.st_uid == os.getuid() and stat.S_IMODE(metadata.st_mode) & 0o077 == 0


def probe_duration_seconds(path: Path, *, ffprobe: str = "/opt/homebrew/bin/ffprobe") -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        check=False, timeout=60, env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "LANG": "C", "LC_ALL": "C"},
    )
    try:
        value = float(json.loads(result.stdout).get("format", {}).get("duration"))
    except Exception as exc:
        raise ClipError("E_CLIP_PROBE") from exc
    if result.returncode != 0 or value <= 0:
        raise ClipError("E_CLIP_PROBE")
    return value


def materialize_bounded_clips(
    source: Path,
    segments_ms: Iterable[tuple[int, int]],
    destination: Path,
    *,
    ffmpeg: str = "/opt/homebrew/bin/ffmpeg",
    ffprobe: str = "/opt/homebrew/bin/ffprobe",
) -> list[dict[str, object]]:
    source = source.resolve(strict=True)
    if not _private_regular(source):
        raise ClipError("E_SOURCE_PRIVATE")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination, 0o700)
    results: list[dict[str, object]] = []
    for index, (start_ms, end_ms) in enumerate(segments_ms):
        if not isinstance(start_ms, int) or not isinstance(end_ms, int) or start_ms < 0 or end_ms <= start_ms:
            raise ClipError("E_SEGMENT")
        expected = (end_ms - start_ms) / 1000.0
        output = destination / f"window-{index:03d}.mp4"
        pending = destination / f".window-{index:03d}.pending.mp4"
        command = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start_ms / 1000:.3f}", "-t", f"{expected:.3f}", "-i", str(source),
            "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(pending),
        ]
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=max(120, int(expected * 20)),
            env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "LANG": "C", "LC_ALL": "C"},
        )
        if completed.returncode != 0 or not pending.is_file():
            raise ClipError("E_CLIP_TRANSCODE")
        observed = probe_duration_seconds(pending, ffprobe=ffprobe)
        tolerance = max(0.15, min(0.50, expected * 0.02))
        if abs(observed - expected) > tolerance:
            pending.unlink(missing_ok=True)
            raise ClipError("E_CLIP_DURATION")
        os.chmod(pending, 0o600)
        os.replace(pending, output)
        results.append({"path": output, "duration_seconds": round(observed, 3), "window_number": index + 1})
    return results
