"""Prospective ChildLens room/hand/camera qualification utilities.

This package never constructs or launches side-cue treatment arms.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_episode_specs(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != "childlens-room-hand-episodes-1.0.0":
        raise ValueError("unsupported episode specification")
    episodes = payload.get("episodes", [])
    ids = [item["episode_id"] for item in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episode_id values must be unique")
    return episodes


def validate_bakeoff_matrix(episodes: list[dict]) -> dict:
    rooms = {item["room_id"] for item in episodes}
    actions = {item["action_primitive"] for item in episodes}
    instances: dict[str, set[str]] = {}
    for item in episodes:
        target = item["target"]
        instances.setdefault(target["category"], set()).add(target["asset_id"])
    return {
        "episode_count": len(episodes),
        "rooms": sorted(rooms),
        "actions": sorted(actions),
        "instances_per_category": {key: len(value) for key, value in sorted(instances.items())},
        "has_near_miss": "near_miss" in actions,
    }
