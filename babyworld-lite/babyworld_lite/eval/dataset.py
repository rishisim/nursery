from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd

from babyworld_lite.eval.features import (
    ARM_MODALITIES,
    extract_windowed_feature_groups,
    hidden_impulse,
    prediction_frame,
    regression_targets,
)


def load_episodes(path: str | Path) -> List[Mapping[str, Any]]:
    episode_path = Path(path)
    with episode_path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_eval_frame(episodes: Iterable[Mapping[str, Any]], k_offset: int = 0) -> pd.DataFrame:
    episode_list = list(episodes)
    if not episode_list:
        raise ValueError("no episodes loaded")
    max_frames = max(len(ep["frames"]) for ep in episode_list)
    rows: list[Dict[str, Any]] = []
    for episode in episode_list:
        base_k = prediction_frame(episode)
        k = min(base_k + k_offset, len(episode["frames"]) - 1)
        groups = extract_windowed_feature_groups(episode, k, max_frames=max_frames)
        targets = regression_targets(episode, k)
        obj = episode["object"]
        impulse = hidden_impulse(episode)
        mass = float(obj["mass"])
        row: Dict[str, Any] = {
            "episode_id": episode["episode_id"],
            "seed": episode.get("seed"),
            "event_label": episode["event_label"],
            "shape": obj["shape"],
            "action": episode["action"],
            "material": obj["material"],
            "mass": mass,
            "hidden_impulse": impulse,
            "impulse_per_mass": impulse / max(0.2, mass),
            "prediction_frame": k,
            "base_contact_frame": base_k,
        }
        row.update(targets)
        for modality in ("vision", "proprio", "touch", "oracle"):
            row.update(groups[modality])
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns_for_modalities(df: pd.DataFrame, modalities: Iterable[str]) -> List[str]:
    prefixes = tuple(f"{modality}_" for modality in modalities)
    return [column for column in df.columns if column.startswith(prefixes)]


def feature_columns_for_arm(df: pd.DataFrame, arm: str, extra_columns: Optional[Iterable[str]] = None) -> List[str]:
    if arm not in ARM_MODALITIES:
        raise KeyError(f"unknown arm: {arm}")
    columns = feature_columns_for_modalities(df, ARM_MODALITIES[arm])
    if extra_columns:
        columns.extend(extra_columns)
    return columns


def categorical_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> tuple[List[str], List[str]]:
    categorical: list[str] = []
    numeric: list[str] = []
    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return categorical, numeric


def touch_feature_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in df.columns if column.startswith("touch_")]
