"""Pure mechanics for the prospective ChildLens calibration recovery v5.

This module never discovers or reads restricted storage.  It contains only
outcome-blind selection, fold, lag-grid, learner, decision, and privacy
mechanics that can be exercised with synthetic inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import random
import re
from typing import Any

import numpy as np


class BridgeV5Error(RuntimeError):
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
        raise BridgeV5Error("E_CANONICAL") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def keyed_hash(seed: str, *parts: Any) -> str:
    material = "|".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def common_duration_target(
    rows: Sequence[Mapping[str, Any]],
    *,
    participant_count: int = 18,
    minimum_seconds: int = 600,
    preferred_seconds: int = 900,
    maximum_seconds: int = 1200,
    quantum_seconds: int = 60,
    minimum_total_hours: float = 3.0,
) -> int:
    """Choose one uniform target from development-only availability."""

    if (
        participant_count <= 0
        or not 0 < minimum_seconds <= preferred_seconds <= maximum_seconds
        or quantum_seconds <= 0
        or minimum_total_hours <= 0
    ):
        raise BridgeV5Error("E_DURATION_CONFIG")
    availability: dict[str, int] = {}
    for row in rows:
        participant = str(row.get("participant_key", ""))
        try:
            seconds = math.floor(float(row["available_released_seconds"]))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise BridgeV5Error("E_DURATION_ROW") from exc
        if not participant or participant in availability or seconds < 0:
            raise BridgeV5Error("E_DURATION_ROW")
        availability[participant] = seconds
    if len(availability) != participant_count:
        raise BridgeV5Error("E_DURATION_PARTICIPANTS")
    common = min(availability.values())
    target = min(preferred_seconds, maximum_seconds, common)
    target = (target // quantum_seconds) * quantum_seconds
    if (
        target < minimum_seconds
        or participant_count * target < minimum_total_hours * 3600
    ):
        raise BridgeV5Error("E_DURATION_SUPPORT")
    return target


def deterministic_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    fold_count: int = 3,
    fold_size: int = 6,
) -> dict[str, int]:
    """Assign metadata-stratified, participant-disjoint folds.

    Greedy assignment minimizes fold size first, then cohort and released
    metadata-stratum imbalance.  Only explicitly supplied development rows are
    accepted; the caller must bind their provenance before calling.
    """

    if fold_count <= 1 or fold_size <= 0:
        raise BridgeV5Error("E_FOLD_CONFIG")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in rows:
        participant = str(raw.get("participant_key", ""))
        cohort = str(raw.get("cohort", ""))
        activity = str(raw.get("activity_bin", "__UNAVAILABLE__"))
        location = str(raw.get("location_bin", "__UNAVAILABLE__"))
        speech = str(raw.get("speech_support_bin", "__UNAVAILABLE__"))
        if not participant or not cohort or participant in seen:
            raise BridgeV5Error("E_FOLD_ROW")
        seen.add(participant)
        normalized.append(
            {
                "participant_key": participant,
                "cohort": cohort,
                "stratum": "|".join((cohort, activity, location, speech)),
            }
        )
    if len(normalized) != fold_count * fold_size:
        raise BridgeV5Error("E_FOLD_SUPPORT")

    global_stratum = Counter(row["stratum"] for row in normalized)
    fold_rows: list[list[dict[str, str]]] = [[] for _ in range(fold_count)]
    fold_cohort: list[Counter[str]] = [Counter() for _ in range(fold_count)]
    fold_stratum: list[Counter[str]] = [Counter() for _ in range(fold_count)]
    by_cohort: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in normalized:
        by_cohort[row["cohort"]].append(row)
    for cohort, cohort_rows in sorted(
        by_cohort.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        ordered = sorted(
            cohort_rows,
            key=lambda row: (
                global_stratum[row["stratum"]],
                keyed_hash(seed, cohort, row["participant_key"]),
            ),
        )
        for row in ordered:
            options: list[tuple[int, int, int, str, int]] = []
            for fold in range(fold_count):
                if len(fold_rows[fold]) >= fold_size:
                    continue
                options.append(
                    (
                        fold_cohort[fold][cohort],
                        len(fold_rows[fold]),
                        fold_stratum[fold][row["stratum"]],
                        keyed_hash(seed, "fold", fold, row["participant_key"]),
                        fold,
                    )
                )
            if not options:
                raise BridgeV5Error("E_FOLD_CAPACITY")
            chosen = min(options)[-1]
            fold_rows[chosen].append(row)
            fold_cohort[chosen][cohort] += 1
            fold_stratum[chosen][row["stratum"]] += 1

    if any(len(values) != fold_size for values in fold_rows):
        raise BridgeV5Error("E_FOLD_CAPACITY")
    return {
        row["participant_key"]: fold
        for fold, values in enumerate(fold_rows)
        for row in values
    }


def lag_grid(window_seconds: Sequence[int], multiples: Sequence[int]) -> dict[int, list[int]]:
    """Return nonoverlapping signed onset lags for each temporal scale."""

    if not window_seconds or not multiples:
        raise BridgeV5Error("E_LAG_CONFIG")
    if any(value <= 0 for value in window_seconds) or any(
        value <= 0 for value in multiples
    ):
        raise BridgeV5Error("E_LAG_CONFIG")
    if len(set(window_seconds)) != len(window_seconds) or len(set(multiples)) != len(
        multiples
    ):
        raise BridgeV5Error("E_LAG_CONFIG")
    return {
        duration: [
            *[-duration * value for value in sorted(multiples, reverse=True)],
            0,
            *[duration * value for value in sorted(multiples)],
        ]
        for duration in sorted(window_seconds)
    }


def trainable_parameter_count(
    audio_dimension: int,
    vision_dimension: int,
    hidden_dimension: int,
    output_dimension: int,
) -> int:
    """Count two LayerNorm + two-layer statistic-pooling projectors."""

    if min(
        audio_dimension, vision_dimension, hidden_dimension, output_dimension
    ) <= 0:
        raise BridgeV5Error("E_ARCHITECTURE")

    def one(input_dimension: int) -> int:
        pooled = 2 * input_dimension
        layer_norm = 2 * pooled
        first = pooled * hidden_dimension + hidden_dimension
        second = hidden_dimension * output_dimension + output_dimension
        return layer_norm + first + second

    return one(audio_dimension) + one(vision_dimension)


def train_temporal_projectors(
    audio_sequences: np.ndarray,
    vision_sequences: np.ndarray,
    participant_keys: Sequence[str],
    *,
    hidden_dimension: int,
    output_dimension: int,
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    dropout: float,
    gradient_clip_norm: float,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Train the frozen statistic-pooling two-layer projectors."""

    try:
        import torch
    except ImportError as exc:
        raise BridgeV5Error("E_TORCH_UNAVAILABLE") from exc

    audio = np.asarray(audio_sequences, dtype=np.float32)
    vision = np.asarray(vision_sequences, dtype=np.float32)
    if (
        audio.ndim != 3
        or vision.ndim != 3
        or audio.shape[0] != vision.shape[0]
        or audio.shape[0] != len(participant_keys)
        or min(hidden_dimension, output_dimension, epochs, steps_per_epoch) <= 0
        or min(learning_rate, temperature, gradient_clip_norm) <= 0
        or weight_decay < 0
        or not 0 <= dropout < 1
    ):
        raise BridgeV5Error("E_TRAIN_SHAPE")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, participant in enumerate(participant_keys):
        grouped[str(participant)].append(index)
    if len(grouped) < 2:
        raise BridgeV5Error("E_TRAIN_SUPPORT")

    class Projector(torch.nn.Module):
        def __init__(self, input_dimension: int):
            super().__init__()
            pooled = 2 * input_dimension
            self.norm = torch.nn.LayerNorm(pooled)
            self.first = torch.nn.Linear(pooled, hidden_dimension)
            self.dropout = torch.nn.Dropout(dropout)
            self.second = torch.nn.Linear(hidden_dimension, output_dimension)

        def forward(self, values):
            pooled = torch.cat(
                (values.mean(dim=1), values.std(dim=1, unbiased=False)), dim=1
            )
            hidden = torch.nn.functional.gelu(self.first(self.norm(pooled)))
            return torch.nn.functional.normalize(
                self.second(self.dropout(hidden)), dim=1
            )

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    audio_projector = Projector(audio.shape[2])
    vision_projector = Projector(vision.shape[2])
    optimizer = torch.optim.AdamW(
        [*audio_projector.parameters(), *vision_projector.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    audio_tensor = torch.from_numpy(audio)
    vision_tensor = torch.from_numpy(vision)
    participants = sorted(grouped)
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            indices = [
                grouped[participant][
                    int(keyed_hash(str(seed), epoch, step, participant), 16)
                    % len(grouped[participant])
                ]
                for participant in participants
            ]
            projected_audio = audio_projector(audio_tensor[indices])
            projected_vision = vision_projector(vision_tensor[indices])
            logits = projected_audio @ projected_vision.T / temperature
            labels = torch.arange(len(indices))
            loss = 0.5 * (
                torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*audio_projector.parameters(), *vision_projector.parameters()],
                gradient_clip_norm,
            )
            optimizer.step()

    def export(module) -> dict[str, np.ndarray]:
        return {
            name: value.detach().cpu().numpy().copy()
            for name, value in module.state_dict().items()
        }

    return {"audio": export(audio_projector), "vision": export(vision_projector)}


def score_temporal_projectors(
    audio_sequences: np.ndarray,
    vision_sequences: np.ndarray,
    state: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    """Score aligned rows using exported frozen projector parameters."""

    try:
        import torch
    except ImportError as exc:
        raise BridgeV5Error("E_TORCH_UNAVAILABLE") from exc

    def project(values: np.ndarray, parameters: Mapping[str, np.ndarray]):
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
        if tensor.ndim != 3:
            raise BridgeV5Error("E_SCORE_SHAPE")
        pooled = torch.cat(
            (tensor.mean(dim=1), tensor.std(dim=1, unbiased=False)), dim=1
        )
        normalized = torch.nn.functional.layer_norm(
            pooled,
            (pooled.shape[1],),
            torch.from_numpy(np.asarray(parameters["norm.weight"])),
            torch.from_numpy(np.asarray(parameters["norm.bias"])),
        )
        hidden = torch.nn.functional.gelu(
            torch.nn.functional.linear(
                normalized,
                torch.from_numpy(np.asarray(parameters["first.weight"])),
                torch.from_numpy(np.asarray(parameters["first.bias"])),
            )
        )
        projected = torch.nn.functional.linear(
            hidden,
            torch.from_numpy(np.asarray(parameters["second.weight"])),
            torch.from_numpy(np.asarray(parameters["second.bias"])),
        )
        return torch.nn.functional.normalize(projected, dim=1)

    audio = project(audio_sequences, state["audio"])
    vision = project(vision_sequences, state["vision"])
    if audio.shape != vision.shape:
        raise BridgeV5Error("E_SCORE_SHAPE")
    return torch.sum(audio * vision, dim=1).detach().cpu().numpy()


def _cluster_interval(
    values: Sequence[float],
    *,
    confidence: float,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if not values or not 0 < confidence < 1 or replicates < 100:
        raise BridgeV5Error("E_INTERVAL_CONFIG")
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    )
    alpha = (1 - confidence) / 2
    return (
        estimates[math.floor(alpha * (replicates - 1))],
        estimates[math.ceil((1 - alpha) * (replicates - 1))],
    )


def run_embedding_positive_control(
    *,
    seed: int = 20260723,
    participants: int = 18,
    windows_per_participant: int = 24,
    confidence: float = 0.9,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Outcome-blind scoring sensitivity check with an injected shared latent.

    The first 12 synthetic participants train the exact frozen projector
    architecture and the final 6 evaluate it. A within-participant cyclic lag
    destroys the injected row-level relation while preserving inventory.
    """

    if participants < 5 or windows_per_participant < 4:
        raise BridgeV5Error("E_POSITIVE_SUPPORT")
    if participants != 18:
        raise BridgeV5Error("E_POSITIVE_SUPPORT")
    rng = np.random.default_rng(seed)
    latent_dimension = 32
    audio_dimension = 1024
    vision_dimension = 384
    audio_time = 3
    vision_time = 3
    total = participants * windows_per_participant
    latent = rng.normal(size=(total, latent_dimension)).astype(np.float32)
    audio = rng.normal(
        scale=0.08, size=(total, audio_time, audio_dimension)
    ).astype(np.float32)
    vision = rng.normal(
        scale=0.08, size=(total, vision_time, vision_dimension)
    ).astype(np.float32)
    audio[:, :, :latent_dimension] += latent[:, None, :]
    vision[:, :, :latent_dimension] += latent[:, None, :]
    participant_keys = [
        f"synthetic-{index // windows_per_participant:02d}" for index in range(total)
    ]
    training_rows = (participants - 6) * windows_per_participant
    state = train_temporal_projectors(
        audio[:training_rows],
        vision[:training_rows],
        participant_keys[:training_rows],
        hidden_dimension=256,
        output_dimension=128,
        epochs=80,
        steps_per_epoch=12,
        learning_rate=0.0003,
        weight_decay=0.01,
        temperature=0.07,
        dropout=0.1,
        gradient_clip_norm=1.0,
        seed=seed,
    )
    evaluation_audio = audio[training_rows:]
    evaluation_vision = vision[training_rows:]
    aligned = score_temporal_projectors(evaluation_audio, evaluation_vision, state)
    participant_effects: list[float] = []
    for participant in range(6):
        left = participant * windows_per_participant
        right = left + windows_per_participant
        lagged_vision = np.roll(evaluation_vision[left:right], shift=3, axis=0)
        lagged = score_temporal_projectors(
            evaluation_audio[left:right], lagged_vision, state
        )
        participant_effects.append(float(np.mean(aligned[left:right] - lagged)))
    interval = _cluster_interval(
        participant_effects,
        confidence=confidence,
        replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    mean_lift = float(np.mean(participant_effects))
    return {
        "synthetic_training_participants": 12,
        "synthetic_evaluation_participants": 6,
        "windows_per_participant": windows_per_participant,
        "injected_shared_latent_dimensions": latent_dimension,
        "nonshared_noise_standard_deviation": 0.08,
        "cyclic_lag_rows": 3,
        "learner_trainable_parameter_count": trainable_parameter_count(
            audio_dimension, vision_dimension, 256, 128
        ),
        "mean_aligned_minus_lagged_cosine": mean_lift,
        "participant_cluster_interval": [interval[0], interval[1]],
        "gate_thresholds": {
            "mean_lift_min": 0.1,
            "interval_lower_min": 0.05,
        },
        "pass": mean_lift >= 0.1 and interval[0] >= 0.05,
        "empirical_evidence": False,
    }


def terminal_decision(gates: Mapping[str, bool]) -> str:
    """Apply the frozen three-state development decision without rescue."""

    required = {
        "governance",
        "support",
        "positive_control",
        "preprocessing",
        "shortcut_interpretable",
        "precision",
        "heterogeneity",
        "detectable_structure",
        "precise_weak_or_flat",
    }
    if set(gates) != required or any(type(value) is not bool for value in gates.values()):
        raise BridgeV5Error("E_DECISION_INPUT")
    shared = all(
        gates[name]
        for name in (
            "governance",
            "support",
            "positive_control",
            "preprocessing",
            "shortcut_interpretable",
            "precision",
            "heterogeneity",
        )
    )
    if shared and gates["detectable_structure"]:
        return "PASS_DETECTABLE_STRUCTURE"
    if shared and gates["precise_weak_or_flat"]:
        return "PASS_PRECISE_WEAK_OR_FLAT"
    return "NO_GO_UNINFORMATIVE"


_PATH_OR_MEDIA = re.compile(
    r"(?i)(?:/users/|file://|\\users\\|"
    r"\b\S+\.(?:mp4|mov|mkv|avi|webm|wav|m4a)\b)"
)
_FORBIDDEN_KEYS = re.compile(
    r"(?i)^(participant_key|item_key|media_key|session_key|object_key|"
    r"source_locator|restricted_path|filename|exact_intervals|"
    r"embedding_values|row_scores|learned_weights)$"
)


def public_guard(value: Any) -> None:
    """Reject public artifacts containing restricted keys or path/media text."""

    def walk(current: Any) -> None:
        if isinstance(current, Mapping):
            for key, child in current.items():
                if _FORBIDDEN_KEYS.search(str(key)):
                    raise BridgeV5Error("E_PUBLIC_PRIVACY")
                walk(child)
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            for child in current:
                walk(child)
        elif isinstance(current, str) and _PATH_OR_MEDIA.search(current):
            raise BridgeV5Error("E_PUBLIC_PRIVACY")

    walk(value)
