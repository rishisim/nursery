from __future__ import annotations

from babyworld_lite.childlens_alignment_bridge_v3.selection import (
    coalesce_milliseconds,
    deterministic_participant_selection,
    map_union_range,
)


def test_coalesce_clips_and_unions_explicit_windows() -> None:
    assert coalesce_milliseconds(
        [(-1.0, 1.0), (0.9, 2.0), (3.0, 8.0)],
        5_000,
    ) == [(0, 2_000), (3_000, 5_000)]


def test_map_union_range_crosses_source_gap_without_filling_it() -> None:
    assert map_union_range(
        [(0, 1_000), (5_000, 7_000)],
        offset=500,
        duration=1_500,
    ) == [(500, 1_000), (5_000, 6_000)]


def test_selection_is_deterministic_and_participant_distinct() -> None:
    rows = [
        {
            "media_key": f"m{index}",
            "participant_key": f"p{index // 2}",
            "duration_milliseconds": 60_000 + index,
            "coarse_activity_label": "PLAY" if index % 2 else "MEAL",
            "speech_presence_bin": "PRESENT",
            "location_label": None,
        }
        for index in range(12)
    ]
    first = deterministic_participant_selection(
        rows,
        count=5,
        release_binding="r",
        protocol_sha256="p",
    )
    second = deterministic_participant_selection(
        list(reversed(rows)),
        count=5,
        release_binding="r",
        protocol_sha256="p",
    )
    assert [row["media_key"] for row in first] == [
        row["media_key"] for row in second
    ]
    assert len({row["participant_key"] for row in first}) == 5
