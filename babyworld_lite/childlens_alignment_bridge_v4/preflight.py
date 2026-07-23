"""Pure, ID-preserving mechanics for ChildLens alignment bridge v4.

Restricted filesystem access and media decoding intentionally live in the
versioned runner.  This module contains deterministic split, window, control,
training, inference, and privacy mechanics that can be regression-tested on
synthetic inputs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import random
from typing import Any

import numpy as np


class BridgeV4Error(RuntimeError):
    """Fail-closed error carrying a stable public code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BridgeV4Error("E_CANONICAL") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def keyed_hash(seed: str, *parts: Any) -> str:
    material = "|".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def deterministic_split(
    participants: Sequence[Mapping[str, str]],
    *,
    seed: str,
    evaluation_per_cohort: int,
) -> dict[str, str]:
    """Freeze a cohort-balanced participant split without outcome fields."""

    if evaluation_per_cohort <= 0:
        raise BridgeV4Error("E_SPLIT_CONFIG")
    by_cohort: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in participants:
        participant = str(row.get("participant_key", ""))
        cohort = str(row.get("cohort", ""))
        if not participant or not cohort or participant in seen:
            raise BridgeV4Error("E_SPLIT_PARTICIPANTS")
        by_cohort[cohort].append(participant)
        seen.add(participant)
    if len(by_cohort) < 2:
        raise BridgeV4Error("E_SPLIT_COHORTS")
    result: dict[str, str] = {}
    for cohort, values in sorted(by_cohort.items()):
        ordered = sorted(values, key=lambda value: keyed_hash(seed, cohort, value))
        if len(ordered) <= evaluation_per_cohort:
            raise BridgeV4Error("E_SPLIT_SUPPORT")
        for index, participant in enumerate(ordered):
            result[participant] = (
                "evaluation" if index < evaluation_per_cohort else "train"
            )
    return result


def _quantized_milliseconds(value: float) -> int:
    if not math.isfinite(float(value)):
        raise BridgeV4Error("E_WINDOW_NUMBER")
    return round(float(value) * 1000.0)


def candidate_windows(
    segments: Sequence[Mapping[str, Any]],
    *,
    recording_duration_seconds: float,
    window_seconds: float,
    exclusion_buffer_seconds: float,
    maximum_windows: int,
    seed: str,
    participant_key: str,
    item_key: str,
) -> list[dict[str, int | str]]:
    """Create fixed, nonoverlapping windows from released timed segments."""

    duration_ms = _quantized_milliseconds(recording_duration_seconds)
    width_ms = _quantized_milliseconds(window_seconds)
    buffer_ms = _quantized_milliseconds(exclusion_buffer_seconds)
    if duration_ms < width_ms or width_ms <= 0 or maximum_windows <= 0:
        raise BridgeV4Error("E_WINDOW_CONFIG")
    proposed: dict[tuple[int, int], str] = {}
    for raw in segments:
        try:
            start_ms = _quantized_milliseconds(float(raw["start_seconds"]))
            end_ms = _quantized_milliseconds(float(raw["end_seconds"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeV4Error("E_WINDOW_SEGMENT") from exc
        start_ms = max(0, start_ms)
        end_ms = min(duration_ms, end_ms)
        if end_ms <= start_ms:
            continue
        if end_ms - start_ms < width_ms:
            center = (start_ms + end_ms) // 2
            left = max(0, min(duration_ms - width_ms, center - width_ms // 2))
            starts = [left]
        else:
            starts = list(range(start_ms, end_ms - width_ms + 1, width_ms))
        for left in starts:
            right = left + width_ms
            if 0 <= left < right <= duration_ms:
                proposed[(left, right)] = keyed_hash(
                    seed, participant_key, item_key, left
                )
    retained: list[tuple[int, int, str]] = []
    for (left, right), row_hash in sorted(
        proposed.items(), key=lambda row: (row[0][0], row[0][1], row[1])
    ):
        if retained and left < retained[-1][1] + buffer_ms:
            continue
        retained.append((left, right, row_hash))
    retained = sorted(retained, key=lambda row: row[2])[:maximum_windows]
    retained.sort(key=lambda row: (row[0], row[1]))
    return [
        {"start_ms": left, "end_ms": right, "selection_hash": row_hash}
        for left, right, row_hash in retained
    ]


def shifted_window(
    *,
    start_ms: int,
    end_ms: int,
    recording_duration_ms: int,
    offset_ms: int,
    minimum_gap_ms: int,
    seed: str,
    row_key: str,
) -> tuple[int, int] | None:
    """Return the frozen non-wrapping within-recording video control."""

    if not (0 <= start_ms < end_ms <= recording_duration_ms):
        raise BridgeV4Error("E_SHIFT_BOUNDS")
    if offset_ms <= 0 or minimum_gap_ms < 0:
        raise BridgeV4Error("E_SHIFT_CONFIG")
    positive_first = int(keyed_hash(seed, "shift", row_key), 16) & 1
    directions = (1, -1) if positive_first else (-1, 1)
    for direction in directions:
        left = start_ms + direction * offset_ms
        right = end_ms + direction * offset_ms
        inside = 0 <= left < right <= recording_duration_ms
        separated = right + minimum_gap_ms <= start_ms or end_ms + minimum_gap_ms <= left
        if inside and separated:
            return left, right
    return None


def cross_participant_assignment(
    rows: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Minimum-cost one-to-one donor indices with same-participant edges forbidden."""

    from scipy.optimize import linear_sum_assignment

    count = len(rows)
    if count < 2:
        raise BridgeV4Error("E_SHUFFLE_SUPPORT")
    cost = np.full((count, count), 1e9, dtype=np.float64)
    for receiver_index, receiver in enumerate(rows):
        for donor_index, donor in enumerate(rows):
            if receiver["participant_key"] == donor["participant_key"]:
                continue
            activity = float(receiver["activity_label"] != donor["activity_label"])
            location = float(receiver["location_label"] != donor["location_label"])
            density = abs(
                float(receiver["speech_density"]) - float(donor["speech_density"])
            )
            position = abs(
                float(receiver["recording_position"])
                - float(donor["recording_position"])
            )
            # Stable fractional tie break is far below any declared nuisance term.
            tie = int(str(donor["row_hash"])[:8], 16) / (16**8) * 1e-9
            cost[receiver_index, donor_index] = (
                4.0 * activity + 2.0 * location + density + 0.1 * position + tie
            )
    receiver_indices, donor_indices = linear_sum_assignment(cost)
    if (
        len(receiver_indices) != count
        or set(receiver_indices.tolist()) != set(range(count))
        or any(cost[left, right] >= 1e8 for left, right in zip(receiver_indices, donor_indices))
    ):
        raise BridgeV4Error("E_SHUFFLE_INFEASIBLE")
    assignment = [-1] * count
    for receiver, donor in zip(receiver_indices.tolist(), donor_indices.tolist()):
        assignment[receiver] = donor
    if sorted(assignment) != list(range(count)):
        raise BridgeV4Error("E_SHUFFLE_INVENTORY")
    return assignment


def train_projection_heads(
    audio_features: np.ndarray,
    visual_features: np.ndarray,
    participant_keys: Sequence[str],
    *,
    output_dimension: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    gradient_clip_norm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train only two bias-free linear heads using one row per participant."""

    import torch

    audio = np.asarray(audio_features, dtype=np.float32)
    visual = np.asarray(visual_features, dtype=np.float32)
    if (
        audio.ndim != 2
        or visual.ndim != 2
        or audio.shape[0] != visual.shape[0]
        or audio.shape[0] != len(participant_keys)
        or output_dimension <= 0
        or epochs <= 0
        or temperature <= 0
    ):
        raise BridgeV4Error("E_TRAIN_SHAPE")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, participant in enumerate(participant_keys):
        grouped[str(participant)].append(index)
    if len(grouped) < 2 or any(not values for values in grouped.values()):
        raise BridgeV4Error("E_TRAIN_SUPPORT")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    audio_head = torch.nn.Linear(audio.shape[1], output_dimension, bias=False)
    visual_head = torch.nn.Linear(visual.shape[1], output_dimension, bias=False)
    optimizer = torch.optim.AdamW(
        [*audio_head.parameters(), *visual_head.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    audio_tensor = torch.from_numpy(audio)
    visual_tensor = torch.from_numpy(visual)
    participants = sorted(grouped)
    for epoch in range(epochs):
        indices = [
            grouped[participant][
                int(keyed_hash(str(seed), epoch, participant), 16)
                % len(grouped[participant])
            ]
            for participant in participants
        ]
        selected_audio = torch.nn.functional.normalize(
            audio_head(audio_tensor[indices]), dim=1
        )
        selected_visual = torch.nn.functional.normalize(
            visual_head(visual_tensor[indices]), dim=1
        )
        logits = selected_audio @ selected_visual.T / temperature
        labels = torch.arange(len(indices))
        loss = 0.5 * (
            torch.nn.functional.cross_entropy(logits, labels)
            + torch.nn.functional.cross_entropy(logits.T, labels)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*audio_head.parameters(), *visual_head.parameters()],
            gradient_clip_norm,
        )
        optimizer.step()
    return (
        audio_head.weight.detach().cpu().numpy().copy(),
        visual_head.weight.detach().cpu().numpy().copy(),
    )


def projected_cosine(
    audio_features: np.ndarray,
    visual_features: np.ndarray,
    audio_weight: np.ndarray,
    visual_weight: np.ndarray,
) -> np.ndarray:
    audio = np.asarray(audio_features, dtype=np.float64) @ np.asarray(
        audio_weight, dtype=np.float64
    ).T
    visual = np.asarray(visual_features, dtype=np.float64) @ np.asarray(
        visual_weight, dtype=np.float64
    ).T
    audio_norm = np.linalg.norm(audio, axis=1, keepdims=True)
    visual_norm = np.linalg.norm(visual, axis=1, keepdims=True)
    if (
        audio.shape != visual.shape
        or audio.ndim != 2
        or np.any(audio_norm == 0)
        or np.any(visual_norm == 0)
    ):
        raise BridgeV4Error("E_SCORE_SHAPE")
    return np.sum((audio / audio_norm) * (visual / visual_norm), axis=1)


def _participant_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_field: str,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            grouped[str(row["participant_key"])].append(
                float(row["real_cosine"]) - float(row[control_field])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeV4Error("E_EFFECT_ROW") from exc
    if not grouped:
        raise BridgeV4Error("E_EFFECT_EMPTY")
    return [float(np.mean(values)) for _, values in sorted(grouped.items())]


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if not values or replicates < 100 or not 0 < confidence < 1:
        raise BridgeV4Error("E_BOOTSTRAP_CONFIG")
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    )
    alpha = (1 - confidence) / 2
    low = math.floor(alpha * (replicates - 1))
    high = math.ceil((1 - alpha) * (replicates - 1))
    return estimates[low], estimates[high]


def _one_sided_sign_flip(values: Sequence[float], *, seed: int, replicates: int) -> float:
    observed = float(np.mean(values))
    count = len(values)
    if count <= 16:
        total = 1 << count
        exceed = 0
        for mask in range(total):
            estimate = sum(
                value if (mask >> index) & 1 else -value
                for index, value in enumerate(values)
            ) / count
            exceed += estimate >= observed - 1e-15
        return exceed / total
    rng = random.Random(seed)
    exceed = 1
    for _ in range(replicates):
        estimate = sum(
            value if rng.getrandbits(1) else -value for value in values
        ) / count
        exceed += estimate >= observed
    return exceed / (replicates + 1)


def effect_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_field: str,
    confidence: float,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = _participant_values(rows, control_field=control_field)
    interval = _bootstrap_mean_interval(
        values,
        confidence=confidence,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    return {
        "participant_count": len(values),
        "mean_lift": float(np.mean(values)),
        "participant_cluster_bootstrap_interval": [interval[0], interval[1]],
        "one_sided_sign_flip_p": _one_sided_sign_flip(
            values, seed=seed + 1, replicates=permutation_replicates
        ),
    }


def safe_fraction(
    numerator: int,
    denominator: int,
    *,
    minimum_cell_size: int,
) -> tuple[float | None, bool]:
    """Suppress when either nonzero complementary count is smaller than K."""

    if denominator < minimum_cell_size or not 0 <= numerator <= denominator:
        return None, True
    complement = denominator - numerator
    if (0 < numerator < minimum_cell_size) or (
        0 < complement < minimum_cell_size
    ):
        return None, True
    return numerator / denominator, False
