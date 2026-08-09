#!/usr/bin/env python3
"""Future corrected audit view: only physically bounded clips may be rendered.

This module is intentionally not launched for the language-unqualified author.
The v1.3 attempt has been invalidated; a clean future pass requires a qualified,
authorized German annotator and a separately initialized private workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol


class VideoUI(Protocol):
    def caption(self, value: str) -> None: ...
    def video(self, value: str) -> None: ...


def render_bounded_windows(
    ui: VideoUI,
    clips: Iterable[dict[str, object]],
    *,
    item_number: int,
    item_count: int,
    completed_seconds: float,
    total_seconds: float = 900.0,
) -> None:
    clip_list = list(clips)
    if not clip_list:
        raise ValueError("E_NO_BOUNDED_WINDOWS")
    ui.caption(
        f"Frozen audit item {item_number} of {item_count} · "
        f"{completed_seconds / 60:.1f} of {total_seconds / 60:.1f} audit minutes completed"
    )
    for index, clip in enumerate(clip_list, start=1):
        path = clip.get("path")
        duration = clip.get("duration_seconds")
        if not isinstance(path, Path) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("E_BOUNDED_CLIP_SCHEMA")
        ui.caption(
            f"Audit window {index} of {len(clip_list)} · {float(duration):.1f} seconds. "
            "Playback is physically limited to this exact frozen window; do not inspect the source outside it."
        )
        # Deliberately no start_time/end_time: the object itself is the bounded clip.
        ui.video(str(path))
