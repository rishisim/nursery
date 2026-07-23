"""Outcome-blind selection helpers for ChildLens alignment-bridge v3."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def coalesce_milliseconds(
    windows: Sequence[tuple[float, float]],
    duration_milliseconds: int,
) -> list[tuple[int, int]]:
    """Clip, integerize, and union explicit released timing intervals."""

    cleaned: list[tuple[int, int]] = []
    for start, end in windows:
        left = max(0, math.ceil(float(start) * 1000.0))
        right = min(duration_milliseconds, math.floor(float(end) * 1000.0))
        if right > left:
            cleaned.append((left, right))
    merged: list[list[int]] = []
    for start, end in sorted(cleaned):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def map_union_range(
    windows: Sequence[tuple[int, int]],
    offset: int,
    duration: int,
) -> list[tuple[int, int]]:
    """Map a contiguous range in union-time back to source-time intervals."""

    if offset < 0 or duration <= 0:
        raise ValueError("E_SAMPLE_RANGE")
    cursor = 0
    remaining = duration
    result: list[tuple[int, int]] = []
    for start, end in windows:
        length = end - start
        if length <= 0:
            raise ValueError("E_SAMPLE_WINDOWS")
        if offset >= cursor + length:
            cursor += length
            continue
        local = max(0, offset - cursor)
        source_start = start + local
        take = min(remaining, end - source_start)
        if take > 0:
            result.append((source_start, source_start + take))
            offset += take
            remaining -= take
        cursor += length
        if remaining == 0:
            break
    if remaining:
        raise ValueError("E_SAMPLE_RANGE")
    return result


def _duration_tertiles(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["duration_milliseconds"]),
            str(row["media_key"]),
        ),
    )
    if not ordered:
        return {}
    labels = ("LOW", "MIDDLE", "HIGH")
    return {
        str(row["media_key"]): labels[min(2, (index * 3) // len(ordered))]
        for index, row in enumerate(ordered)
    }


def deterministic_participant_selection(
    candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    release_binding: str,
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    """Balance metadata strata with deterministic hash ties and no participant reuse."""

    if count <= 0:
        raise ValueError("E_SELECTION_COUNT")
    tertiles = _duration_tertiles(candidates)
    by_stratum: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        row = dict(raw)
        media_key = str(row["media_key"])
        stratum = (
            str(row["coarse_activity_label"]),
            str(row["speech_presence_bin"]),
            tertiles[media_key],
            (
                str(row["location_label"])
                if row.get("location_label") is not None
                else "__UNAVAILABLE__"
            ),
        )
        row["stratum"] = stratum
        row["selection_hash"] = hashlib.sha256(
            (
                release_binding
                + "|"
                + protocol_sha256
                + "|"
                + media_key
            ).encode("utf-8")
        ).hexdigest()
        by_stratum[stratum].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: (row["selection_hash"], row["media_key"]))
    stratum_hash = {
        stratum: hashlib.sha256(
            release_binding.encode("utf-8")
            + protocol_sha256.encode("utf-8")
            + canonical_bytes(list(stratum))
        ).hexdigest()
        for stratum in by_stratum
    }
    selected: list[dict[str, Any]] = []
    participants: set[str] = set()
    media_keys: set[str] = set()
    counts: Counter[tuple[str, str, str, str]] = Counter()
    while len(selected) < count:
        options: list[tuple[int, str, str, dict[str, Any]]] = []
        for stratum, rows in by_stratum.items():
            candidate = next(
                (
                    row
                    for row in rows
                    if row["media_key"] not in media_keys
                    and row["participant_key"] not in participants
                ),
                None,
            )
            if candidate is not None:
                options.append(
                    (
                        counts[stratum],
                        stratum_hash[stratum],
                        candidate["selection_hash"],
                        candidate,
                    )
                )
        if not options:
            raise ValueError("E_SELECTION_INSUFFICIENT")
        chosen = min(options, key=lambda row: row[:3])[3]
        selected.append(chosen)
        participants.add(str(chosen["participant_key"]))
        media_keys.add(str(chosen["media_key"]))
        counts[chosen["stratum"]] += 1
    return selected
