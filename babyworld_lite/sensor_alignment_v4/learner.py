from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .corpus import EvalKey, EvalPrompt
from .protocol import (
    SeedFirewall,
    SeedReference,
    canonical_digest,
    reject_oracle_fields,
)

SLOT_DIMENSIONS = {"primitive": 3, "manner": 2, "noun": 6}


@dataclass(frozen=True)
class LexicalModel:
    prototypes: dict[str, tuple[float, ...]]
    learner: str
    model_seed: int
    iterations: int
    competition_weight: float

    def serializable(self) -> dict[str, Any]:
        return {
            "prototypes": {key: list(value) for key, value in sorted(self.prototypes.items())},
            "learner": self.learner,
            "model_seed": self.model_seed,
            "iterations": self.iterations,
            "competition_weight": self.competition_weight,
        }


def _rng(seed: int, label: str) -> np.random.Generator:
    value = int.from_bytes(
        hashlib.sha256(f"v4-learner|{seed}|{label}".encode()).digest()[:8], "little"
    )
    return np.random.default_rng(value)


def _normalize(value: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(value, dtype=float), 1e-12)
    return clipped / clipped.sum()


def _softmax_with_null(logits: np.ndarray, null_logit: float, temperature: float) -> tuple[np.ndarray, float]:
    values = np.append(np.asarray(logits, dtype=float), float(null_logit)) / temperature
    values -= values.max()
    probability = np.exp(values)
    probability /= probability.sum()
    return probability[:-1], float(probability[-1])


def _event_observations(row: Mapping[str, Any], slot: str) -> np.ndarray:
    if slot == "primitive":
        return np.asarray([event["action_observation"][:3] for event in row["events"]], dtype=float)
    if slot == "manner":
        return np.asarray([event["action_observation"][3:] for event in row["events"]], dtype=float)
    if slot == "noun":
        return np.asarray([event["object_observation"] for event in row["events"]], dtype=float)
    raise ValueError(slot)


def _lag_weight(row: Mapping[str, Any], lag_scale: float) -> float:
    scene_midpoint = float(
        np.mean(
            [
                (float(event["start"]) + float(event["end"])) / 2.0
                for event in row["events"]
            ]
        )
    )
    observed_lag = abs(float(row["utterance"]["speech_time"]) - scene_midpoint)
    return float(math.exp(-observed_lag / lag_scale))


def fit_joint_competitive(
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: SeedFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
    use_sensor: bool,
    iterations_override: int | None = None,
    competition_override: float | None = None,
) -> tuple[LexicalModel, dict[str, Any]]:
    firewall.authorize(
        "fit",
        [SeedReference("corpus", corpus_seed), SeedReference("model", model_seed)],
        purpose=purpose,
    )
    if not rows:
        raise ValueError("cannot fit an empty corpus")
    for row in rows:
        reject_oracle_fields(row)
    learner_config = config["learner"]
    iterations = int(
        learner_config["iterations"] if iterations_override is None else iterations_override
    )
    competition_weight = float(
        learner_config["competition_weight"]
        if competition_override is None
        else competition_override
    )
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0.0 <= competition_weight <= 1.0:
        raise ValueError("competition weight must be in [0, 1]")
    temperature = float(learner_config["temperature"])
    sensor_weight = float(learner_config["sensor_weight"])
    noun_sensor_weight = float(learner_config["noun_sensor_weight"])
    if noun_sensor_weight != 0.0:
        raise ValueError("v4 action-selectivity contract requires zero noun sensor weight")
    lag_scale = float(learner_config["lag_scale"])
    update_rate = float(learner_config["update_rate"])
    smoothing = float(learner_config["smoothing"])
    null_logit = float(learner_config["null_logit"])
    if learner_config["sensor_free_unidentified_manner_policy"] != "symmetric_projection":
        raise ValueError("unsupported sensor-free manner non-identifiability policy")
    rng = _rng(model_seed, "initialization")
    token_slots: dict[str, str] = {}
    for row in rows:
        for item in row["utterance"]["items"]:
            key = f"{item['slot']}|{item['token']}"
            token_slots[key] = str(item["slot"])
    prototypes = {}
    for key, slot in sorted(token_slots.items()):
        if slot == "manner":
            initialization = np.ones(SLOT_DIMENSIONS[slot])
        else:
            initialization = np.ones(SLOT_DIMENSIONS[slot]) + rng.uniform(
                0.0, 0.08, SLOT_DIMENSIONS[slot]
            )
        prototypes[key] = _normalize(initialization)
    objectives: list[float] = []
    mean_nulls: list[float] = []
    competition_penalties: list[float] = []
    sensor_applications = {"primitive": 0, "manner": 0, "noun": 0}
    lag_weights_observed: list[float] = []
    for _ in range(iterations):
        accumulators = {
            key: np.ones(SLOT_DIMENSIONS[slot]) * smoothing
            for key, slot in token_slots.items()
        }
        nulls: list[float] = []
        objective = 0.0
        penalty_total = 0.0
        for row in rows:
            items = list(row["utterance"]["items"])
            lag_weight = _lag_weight(row, lag_scale)
            lag_weights_observed.append(lag_weight)
            episode_evidence = evidence.get(str(row["episode_id"]), {})
            raw_sensor = np.asarray(
                episode_evidence.get("event_logits", [0.0] * len(row["events"])),
                dtype=float,
            )
            if raw_sensor.size:
                raw_sensor = raw_sensor - raw_sensor.mean()
                scale = raw_sensor.std()
                if scale > 1e-12:
                    raw_sensor = raw_sensor / scale
            reliability = float(episode_evidence.get("quality", 0.0)) * float(
                episode_evidence.get("availability", 0.0)
            )
            base_logits: list[np.ndarray] = []
            observations: list[np.ndarray] = []
            keys: list[str] = []
            for item in items:
                slot = str(item["slot"])
                key = f"{slot}|{item['token']}"
                obs = _event_observations(row, slot)
                logits = obs @ prototypes[key]
                active_sensor_weight = (
                    sensor_weight
                    if use_sensor and slot in {"primitive", "manner"}
                    else noun_sensor_weight
                )
                if active_sensor_weight:
                    logits = logits + active_sensor_weight * reliability * raw_sensor
                    sensor_applications[slot] += 1
                base_logits.append(np.asarray(logits, dtype=float))
                observations.append(obs)
                keys.append(key)
            individual_responsibilities = np.stack(
                [
                    _softmax_with_null(logits, null_logit, temperature)[0]
                    for logits in base_logits
                ]
            )
            joint_logits = np.sum(np.stack(base_logits), axis=0) / math.sqrt(len(base_logits))
            shared_responsibility, shared_null = _softmax_with_null(
                joint_logits, null_logit, temperature
            )
            responsibilities = (
                (1.0 - competition_weight) * individual_responsibilities
                + competition_weight * shared_responsibility[None, :]
            )
            for word_index, logits in enumerate(base_logits):
                _, individual_null = _softmax_with_null(
                    logits, null_logit, temperature
                )
                nulls.append(
                    (1.0 - competition_weight) * individual_null
                    + competition_weight * shared_null
                )
                disagreement = individual_responsibilities[word_index] - shared_responsibility
                penalty_total += float(np.sum(disagreement * disagreement))
                objective += float(np.sum(responsibilities[word_index] * logits))
            for word_index, key in enumerate(keys):
                accumulators[key] += lag_weight * (
                    responsibilities[word_index, :, None] * observations[word_index]
                ).sum(axis=0)
        estimates = {key: _normalize(value) for key, value in accumulators.items()}
        for slot, dimension in SLOT_DIMENSIONS.items():
            slot_keys = sorted(key for key, value in token_slots.items() if value == slot)
            if not slot_keys:
                continue
            matrix = np.stack([estimates[key] for key in slot_keys])
            column_mass = np.maximum(matrix.sum(axis=0), 1e-12)
            exclusive = matrix / np.power(column_mass[None, :], competition_weight)
            for key, row_value in zip(slot_keys, exclusive):
                estimates[key] = _normalize(row_value)
        prototypes = {
            key: _normalize(
                (1.0 - update_rate) * prototypes[key] + update_rate * estimates[key]
            )
            for key in prototypes
        }
        objectives.append(float(objective - competition_weight * penalty_total))
        mean_nulls.append(float(np.mean(nulls)))
        competition_penalties.append(float(penalty_total))
    symmetric_projection_applied = False
    if not use_sensor:
        for key, slot in token_slots.items():
            if slot == "manner":
                prototypes[key] = np.ones(SLOT_DIMENSIONS[slot]) / SLOT_DIMENSIONS[slot]
                symmetric_projection_applied = True
    model = LexicalModel(
        prototypes={key: tuple(map(float, value)) for key, value in prototypes.items()},
        learner="joint_competitive_cross_situational_v4_sensor"
        if use_sensor
        else "joint_competitive_cross_situational_v4_no_sensor",
        model_seed=int(model_seed),
        iterations=iterations,
        competition_weight=competition_weight,
    )
    trace = {
        "objective": "candidate-normalized shared assignment for co-referring component words plus vocabulary-column exclusivity normalization",
        "iterations_configured": iterations,
        "iterations_executed": len(objectives),
        "competition_weight": competition_weight,
        "iteration_objectives": objectives,
        "iteration_competition_penalties": competition_penalties,
        "mean_null_posterior_by_iteration": mean_nulls,
        "sensor_applications_by_slot": sensor_applications,
        "noun_sensor_weight": noun_sensor_weight,
        "lag_weight_min": float(min(lag_weights_observed)),
        "lag_weight_max": float(max(lag_weights_observed)),
        "model_digest": canonical_digest(model.serializable()),
        "oracle_fields_consumed": False,
        "manner_initialization": "symmetric_no_random_label_preference",
        "sensor_free_unidentified_manner_policy": learner_config[
            "sensor_free_unidentified_manner_policy"
        ],
        "sensor_free_symmetric_projection_applied": symmetric_projection_applied,
    }
    return model, trace


def predict_prompts(
    model: LexicalModel,
    prompts: Sequence[EvalPrompt],
    firewall: SeedFirewall,
    *,
    corpus_seed: int,
    purpose: str,
) -> list[dict[str, Any]]:
    firewall.authorize(
        "predict",
        [SeedReference("corpus", corpus_seed), SeedReference("model", model.model_seed)],
        purpose=purpose,
    )
    predictions: list[dict[str, Any]] = []
    for prompt in prompts:
        observation = np.asarray(prompt.observation, dtype=float)
        scores: list[float] = []
        if prompt.kind == "action":
            for primitive_word, manner_word in prompt.candidate_words:
                primitive = np.asarray(
                    model.prototypes.get(f"primitive|{primitive_word}", (1 / 3,) * 3)
                )
                manner = np.asarray(
                    model.prototypes.get(f"manner|{manner_word}", (1 / 2,) * 2)
                )
                scores.append(
                    float(primitive @ observation[:3] + manner @ observation[3:])
                )
        elif prompt.kind == "noun":
            for noun_word in prompt.candidate_words:
                noun = np.asarray(
                    model.prototypes.get(f"noun|{noun_word}", (1 / 6,) * 6)
                )
                scores.append(float(noun @ observation))
        else:
            raise ValueError(prompt.kind)
        predictions.append(
            {
                "prompt_id": prompt.prompt_id,
                "kind": prompt.kind,
                "predicted_index": int(np.argmax(scores)),
                "scores": scores,
            }
        )
    return predictions


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    keys: Sequence[EvalKey],
    firewall: SeedFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> dict[str, Any]:
    firewall.authorize(
        "score",
        [SeedReference("corpus", corpus_seed), SeedReference("model", model_seed)],
        purpose=purpose,
    )
    key_by_id = {key.prompt_id: int(key.answer_index) for key in keys}
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        prompt_id = str(prediction["prompt_id"])
        correct = int(prediction["predicted_index"]) == key_by_id[prompt_id]
        rows.append(
            {
                "prompt_id": prompt_id,
                "kind": str(prediction["kind"]),
                "correct": bool(correct),
            }
        )
    action = [row["correct"] for row in rows if row["kind"] == "action"]
    noun = [row["correct"] for row in rows if row["kind"] == "noun"]
    return {
        "action_accuracy": float(np.mean(action)),
        "noun_accuracy": float(np.mean(noun)),
        "overall_accuracy": float(np.mean([row["correct"] for row in rows])),
        "scored_rows": rows,
    }
