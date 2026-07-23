"""ID-preserving continuous alignment and matched-control inference mechanics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math
import random
from typing import Any

import numpy as np

from .preflight import BridgeError, participant_bootstrap_interval


def normalized_diagonal_cosine(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    row_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Reproduce EgoBabyVLM's normalized diagonal cosine without dropping IDs."""

    images = np.asarray(image_embeddings, dtype=np.float64)
    texts = np.asarray(text_embeddings, dtype=np.float64)
    if (
        images.ndim != 2
        or texts.ndim != 2
        or images.shape != texts.shape
        or images.shape[0] != len(row_ids)
        or len(set(row_ids)) != len(row_ids)
    ):
        raise BridgeError("E_EMBEDDING_SHAPE")
    image_norm = np.linalg.norm(images, axis=1, keepdims=True)
    text_norm = np.linalg.norm(texts, axis=1, keepdims=True)
    if np.any(image_norm == 0) or np.any(text_norm == 0):
        raise BridgeError("E_ZERO_EMBEDDING")
    scores = np.sum((images / image_norm) * (texts / text_norm), axis=1)
    return [
        {"row_id": row_id, "cosine": float(score)}
        for row_id, score in zip(row_ids, scores)
    ]


def shifted_window(
    *,
    start_seconds: float,
    end_seconds: float,
    recording_start_seconds: float,
    recording_end_seconds: float,
    offset_seconds: float,
    protocol_sha256: str,
    utterance_key: str,
) -> tuple[float, float] | None:
    """Return a deterministic non-wrapping, nonoverlapping time-shift control."""

    if not (
        recording_start_seconds <= start_seconds < end_seconds <= recording_end_seconds
        and offset_seconds > 0
    ):
        raise BridgeError("E_SHIFT_BOUNDS")
    bit = int(hashlib.sha256(f"{protocol_sha256}|{utterance_key}".encode()).hexdigest(), 16) & 1
    directions = (1.0, -1.0) if bit else (-1.0, 1.0)
    for direction in directions:
        shifted_start = start_seconds + direction * offset_seconds
        shifted_end = end_seconds + direction * offset_seconds
        inside = (
            recording_start_seconds <= shifted_start
            and shifted_end <= recording_end_seconds
        )
        disjoint = shifted_end <= start_seconds or shifted_start >= end_seconds
        if inside and disjoint:
            return shifted_start, shifted_end
    return None


def participant_lifts(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_field: str,
) -> list[float]:
    """Aggregate paired real-minus-control scores at the participant level."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            grouped[str(row["participant_key"])].append(
                float(row["real_cosine"]) - float(row[control_field])
            )
        except (KeyError, TypeError, ValueError):
            raise BridgeError("E_EFFECT_ROW") from None
    if not grouped:
        raise BridgeError("E_EFFECT_EMPTY")
    return [sum(values) / len(values) for _, values in sorted(grouped.items())]


def sign_flip_pvalue(
    participant_values: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> float:
    """One-sided paired sign-flip randomization p-value."""

    values = [float(value) for value in participant_values]
    if not values or replicates < 100:
        raise BridgeError("E_PERMUTATION_CONFIG")
    observed = sum(values) / len(values)
    n = len(values)
    if n <= 16:
        exceed = 0
        total = 1 << n
        for mask in range(total):
            estimate = sum(
                value if (mask >> index) & 1 else -value
                for index, value in enumerate(values)
            ) / n
            exceed += estimate >= observed
        return exceed / total
    rng = random.Random(seed)
    exceed = 1
    for _ in range(replicates):
        estimate = sum(value if rng.getrandbits(1) else -value for value in values) / n
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
) -> dict[str, float | list[float] | int]:
    values = participant_lifts(rows, control_field=control_field)
    interval = participant_bootstrap_interval(
        values,
        confidence=confidence,
        replicates=bootstrap_replicates,
        seed=seed,
        statistic="mean",
    )
    return {
        "participant_count": len(values),
        "mean_lift": sum(values) / len(values),
        "bootstrap_interval": [interval[0], interval[1]],
        "one_sided_sign_flip_p": sign_flip_pvalue(
            values, replicates=permutation_replicates, seed=seed + 1
        ),
    }
