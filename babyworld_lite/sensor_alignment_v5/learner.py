from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import IdentifierFirewall, IdentifierReference, canonical_digest, reject_oracle_fields

SLOT_DIMENSIONS = {"primitive": 3, "manner": 2, "noun": 6}


def _normalize(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=float), 1e-12)
    return clipped / clipped.sum()


def _softmax(values: Sequence[float], temperature: float) -> np.ndarray:
    array = np.asarray(values, dtype=float) / temperature
    array -= array.max()
    exponential = np.exp(np.clip(array, -60.0, 60.0))
    return exponential / exponential.sum()


def _initial_distribution(token: str, slot: str, model_seed: int, jitter: float) -> np.ndarray:
    dimension = SLOT_DIMENSIONS[slot] + 1
    values = np.ones(dimension)
    for index in range(dimension):
        digest = hashlib.sha256(f"v5-init|{model_seed}|{slot}|{token}|{index}".encode()).digest()
        values[index] += jitter * (int.from_bytes(digest[:4], "little") / (2**32 - 1) - 0.5)
    return _normalize(values)


@dataclass(frozen=True, slots=True)
class LexicalModel:
    distributions: dict[str, tuple[float, ...]]
    learner: str
    model_seed: int
    iterations: int
    competition_weight: float

    def serializable(self) -> dict[str, Any]:
        return {
            "schema_version": "nursery-v5-lexical-model",
            "distributions": {key: list(value) for key, value in sorted(self.distributions.items())},
            "learner": self.learner,
            "model_seed": self.model_seed,
            "iterations": self.iterations,
            "competition_weight": self.competition_weight,
            "post_fit_projection": False,
            "truth_conditioned_canonicalization": False,
            "answer_key_alignment": False,
        }


def _token_key(item: Mapping[str, Any]) -> str:
    return f"{item['slot']}|{item['token']}"


def _event_observation(event: Mapping[str, Any], slot: str) -> np.ndarray:
    field = {
        "primitive": "primitive_observation",
        "manner": "manner_observation",
        "noun": "object_observation",
    }[slot]
    return np.asarray(event[field], dtype=float)


def _lexical_log_score(distribution: np.ndarray, observation: np.ndarray) -> float:
    return float(np.log(np.clip(distribution[:-1] @ observation, 1e-12, 1.0)))


def fit_joint_cross_situational(
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
    iterations_override: int | None = None,
    competition_override: float | None = None,
) -> tuple[LexicalModel, dict[str, Any]]:
    firewall.authorize(
        "fit",
        [IdentifierReference("corpus", corpus_seed), IdentifierReference("model", model_seed)],
        purpose=purpose,
    )
    for row in rows:
        reject_oracle_fields(row)
    learner_config = config["learner"]
    iterations = int(iterations_override or learner_config["iterations"])
    competition = float(
        learner_config["competition_weight"] if competition_override is None else competition_override
    )
    jitter = float(learner_config["initialization_jitter"])
    token_slots = {
        _token_key(item): str(item["slot"])
        for row in rows
        for item in row["utterance"]["items"]
    }
    distributions = {
        key: _initial_distribution(key.split("|", 1)[1], slot, model_seed, jitter)
        for key, slot in sorted(token_slots.items())
    }
    temperature = float(learner_config["temperature"])
    sensor_weight = float(learner_config["sensor_weight"])
    update_rate = float(learner_config["update_rate"])
    smoothing = float(learner_config["smoothing"])
    lag_scale = float(learner_config["lag_scale"])
    objectives: list[float] = []
    competition_penalties: list[float] = []
    iteration_digests: list[str] = []
    sensor_applications = {"primitive": 0, "manner": 0, "noun": 0}
    mean_nulls: list[float] = []
    ordered_rows = sorted(rows, key=lambda row: str(row["episode_id"]))
    for _ in range(iterations):
        accumulators = {
            key: np.full(SLOT_DIMENSIONS[slot] + 1, smoothing, dtype=float)
            for key, slot in token_slots.items()
        }
        objective = 0.0
        null_responsibilities: list[float] = []
        for row in ordered_rows:
            items = list(row["utterance"]["items"])
            events = list(row["events"])
            keys = [_token_key(item) for item in items]
            midpoints = [(float(event["start"]) + float(event["end"])) / 2 for event in events]
            speech_time = float(row["utterance"]["speech_time"])
            lag_weight = math.exp(-min(abs(speech_time - midpoint) for midpoint in midpoints) / lag_scale)
            scores = []
            for event in events:
                score = sum(
                    _lexical_log_score(distributions[key], _event_observation(event, str(item["slot"])))
                    for key, item in zip(keys, items)
                )
                scores.append(score)
            null_score = sum(float(np.log(np.clip(distributions[key][-1], 1e-12, 1.0))) for key in keys)
            detector_value = evidence.get(str(row["episode_id"]))
            if row["family"] == "action" and detector_value is not None:
                quality = float(detector_value["quality"])
                detector_logits = list(map(float, detector_value["event_logits"]))
                if len(detector_logits) != len(scores):
                    raise ValueError("detector/event count mismatch")
                for index in range(len(scores)):
                    scores[index] += sensor_weight * quality * float(np.clip(detector_logits[index], -5, 5))
                null_score += sensor_weight * quality * float(np.clip(detector_value["null_logit"], -5, 5))
                sensor_applications["primitive"] += 1
                sensor_applications["manner"] += 1
            responsibilities = _softmax([*scores, null_score], temperature)
            objective += lag_weight * float(np.log(np.exp(np.asarray([*scores, null_score]) - max([*scores, null_score])).sum()) + max([*scores, null_score]))
            null_responsibilities.append(float(responsibilities[-1]))
            for key, item in zip(keys, items):
                slot = str(item["slot"])
                for index, event in enumerate(events):
                    accumulators[key][:-1] += (
                        lag_weight * responsibilities[index] * _event_observation(event, slot)
                    )
                accumulators[key][-1] += lag_weight * responsibilities[-1]
        estimates = {key: _normalize(values) for key, values in accumulators.items()}
        penalty = 0.0
        for slot in SLOT_DIMENSIONS:
            keys = sorted(key for key, value in token_slots.items() if value == slot)
            if not keys:
                continue
            matrix = np.stack([estimates[key] for key in keys])
            mass = np.maximum(matrix.sum(axis=0), 1e-12)
            exclusive = matrix / np.power(mass[None, :], competition)
            penalty += float(np.sum((matrix - exclusive) ** 2))
            for key, row_value in zip(keys, exclusive):
                estimates[key] = _normalize(row_value)
        distributions = {
            key: _normalize((1.0 - update_rate) * distributions[key] + update_rate * estimates[key])
            for key in distributions
        }
        objectives.append(float(objective - competition * penalty))
        competition_penalties.append(penalty)
        mean_nulls.append(float(np.mean(null_responsibilities)))
        iteration_digests.append(
            canonical_digest({key: value.tolist() for key, value in sorted(distributions.items())})
        )
    model = LexicalModel(
        {key: tuple(map(float, value)) for key, value in distributions.items()},
        str(learner_config["name"]),
        int(model_seed),
        iterations,
        competition,
    )
    trace = {
        "iterations_configured": iterations,
        "iterations_executed": len(objectives),
        "competition_weight": competition,
        "iteration_objectives": objectives,
        "iteration_competition_penalties": competition_penalties,
        "iteration_model_digests": iteration_digests,
        "distinct_iteration_model_digests": len(set(iteration_digests)),
        "mean_null_posterior_by_iteration": mean_nulls,
        "sensor_applications_by_slot": sensor_applications,
        "noun_sensor_weight": float(learner_config["noun_sensor_weight"]),
        "model_digest": canonical_digest(model.serializable()),
        "oracle_fields_consumed": False,
        "evaluation_keys_consumed": False,
        "post_fit_projection_applied": False,
        "condition_specific_parameter_replacement": False,
    }
    return model, trace


def _candidate_score(model: LexicalModel, tokens: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]) -> float:
    if bool(candidate.get("null_option")):
        return sum(
            float(np.log(np.clip(model.distributions[_token_key(item)][-1], 1e-12, 1.0)))
            for item in tokens
        )
    return sum(
        _lexical_log_score(
            np.asarray(model.distributions[_token_key(item)], dtype=float),
            _event_observation(candidate, str(item["slot"])),
        )
        for item in tokens
    )


def predict_prompts(
    model: LexicalModel,
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    purpose: str,
) -> list[dict[str, Any]]:
    firewall.authorize(
        "predict", [IdentifierReference("corpus", corpus_seed), IdentifierReference("model", model.model_seed)], purpose=purpose
    )
    temperature = float(config["learner"]["temperature"])
    output = []
    for prompt in prompts:
        reject_oracle_fields(prompt)
        scores = [_candidate_score(model, prompt["tokens"], candidate) for candidate in prompt["candidates"]]
        stable_order = sorted(
            range(len(scores)), key=lambda index: str(prompt["candidates"][index]["candidate_id"])
        )
        stable_probabilities = _softmax([scores[index] for index in stable_order], temperature)
        probabilities = np.empty(len(scores), dtype=float)
        for stable_index, original_index in enumerate(stable_order):
            probabilities[original_index] = stable_probabilities[stable_index]
        output.append(
            {
                "prompt_id": str(prompt["prompt_id"]),
                "kind": str(prompt["kind"]),
                "candidate_ids": [str(candidate["candidate_id"]) for candidate in prompt["candidates"]],
                "scores": list(map(float, scores)),
                "probabilities": list(map(float, probabilities)),
            }
        )
    return output


def _best_global_permutation(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    predicted = sorted({left for left, _ in pairs})
    truth = sorted({right for _, right in pairs})
    if len(predicted) != len(truth) or len(predicted) > 8:
        return {"accuracy": 0.0, "mapping": {}, "identified": False}
    best_accuracy = -1.0
    best_mapping: dict[str, str] = {}
    for ordering in itertools.permutations(truth):
        mapping = dict(zip(predicted, ordering))
        accuracy = float(np.mean([mapping[left] == right for left, right in pairs]))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping
    return {"accuracy": best_accuracy, "mapping": best_mapping, "identified": best_accuracy == 1.0}


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> dict[str, Any]:
    firewall.authorize(
        "score",
        [IdentifierReference("corpus", corpus_seed), IdentifierReference("model", model_seed)],
        purpose=purpose,
    )
    key_by_id = {str(row["prompt_id"]): int(row["answer_index"]) for row in keys}
    tolerance = float(config["gates"]["exact_tie_tolerance"])
    rows = []
    permutation_pairs: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for prediction in sorted(predictions, key=lambda row: str(row["prompt_id"])):
        prompt_id = str(prediction["prompt_id"])
        answer_index = key_by_id[prompt_id]
        scores = np.asarray(prediction["scores"], dtype=float)
        probabilities = np.asarray(prediction["probabilities"], dtype=float)
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
            raise ValueError("probabilities do not sum to one")
        maxima = np.flatnonzero(np.isclose(scores, scores.max(), atol=tolerance, rtol=0.0))
        fractional = (1.0 / len(maxima)) if answer_index in set(map(int, maxima)) else 0.0
        correct_probability = float(probabilities[answer_index])
        stable_order = sorted(
            range(len(probabilities)), key=lambda index: str(prediction["candidate_ids"][index])
        )
        stable_probabilities = probabilities[stable_order]
        stable_answer = stable_order.index(answer_index)
        one_hot = np.zeros(len(stable_probabilities))
        one_hot[stable_answer] = 1.0
        kind = str(prediction["kind"])
        correct_id = str(prediction["candidate_ids"][answer_index])
        top_id = sorted(str(prediction["candidate_ids"][int(index)]) for index in maxima)[0]
        presence = "null" if correct_id.endswith("-null") else "present"
        if presence == "present":
            permutation_pairs[kind].append((top_id, correct_id))
        rows.append(
            {
                "prompt_id": prompt_id,
                "kind": kind,
                "presence": presence,
                "fractional_credit": float(fractional),
                "correct_probability": correct_probability,
                "brier": float(np.mean((stable_probabilities - one_hot) ** 2)),
                "log_loss": float(-np.log(np.clip(correct_probability, 1e-15, 1.0))),
                "tied_maxima": len(maxima),
            }
        )

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"count": 0}
        confidences = np.asarray([float(row["correct_probability"]) for row in selected])
        correctness = np.asarray([float(row["fractional_credit"]) for row in selected])
        bins = np.linspace(0.0, 1.0, 6)
        ece = 0.0
        for low, high in zip(bins[:-1], bins[1:]):
            chosen = (confidences >= low) & (confidences <= high if high == 1.0 else confidences < high)
            if chosen.any():
                ece += float(chosen.mean()) * abs(float(confidences[chosen].mean() - correctness[chosen].mean()))
        return {
            "count": len(selected),
            "fractional_accuracy": float(np.mean(correctness)),
            "mean_correct_probability": float(np.mean(confidences)),
            "brier": float(np.mean([row["brier"] for row in selected])),
            "log_loss": float(np.mean([row["log_loss"] for row in selected])),
            "tie_frequency": float(np.mean([int(row["tied_maxima"] > 1) for row in selected])),
            "mean_tied_maxima": float(np.mean([row["tied_maxima"] for row in selected])),
            "calibration_ece_5_bin": ece,
        }

    by_kind = {
        kind: summarize([row for row in rows if row["kind"] == kind])
        for kind in ("primitive", "manner", "action", "noun")
    }
    by_presence = {
        presence: summarize([row for row in rows if row["presence"] == presence])
        for presence in ("present", "null")
    }
    permutation = {
        kind: {
            "raw_label_accuracy": float(np.mean([left == right for left, right in pairs])),
            "best_global_permutation": _best_global_permutation(pairs),
        }
        for kind, pairs in sorted(permutation_pairs.items())
    }
    return {
        "tie_policy": "fractional_credit_across_exact_maxima",
        "proper_probability_rule": "softmax_log_loss_and_multiclass_brier",
        "by_kind": by_kind,
        "by_presence": by_presence,
        "overall": summarize(rows),
        "global_label_permutation_diagnostics": permutation,
        "scored_rows": rows,
    }


def reorder_prompts_and_keys(
    prompts: Sequence[Mapping[str, Any]], keys: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    key_by_id = {str(row["prompt_id"]): int(row["answer_index"]) for row in keys}
    reordered_prompts = []
    reordered_keys = []
    for prompt in prompts:
        copy = {**prompt, "tokens": [dict(item) for item in prompt["tokens"]]}
        candidates = list(reversed([dict(candidate) for candidate in prompt["candidates"]]))
        original_answer = key_by_id[str(prompt["prompt_id"])]
        correct_id = str(prompt["candidates"][original_answer]["candidate_id"])
        copy["candidates"] = candidates
        reordered_prompts.append(copy)
        reordered_keys.append(
            {
                "schema_version": "nursery-v5-sealed-key",
                "prompt_id": prompt["prompt_id"],
                "answer_index": next(
                    index for index, candidate in enumerate(candidates) if candidate["candidate_id"] == correct_id
                ),
            }
        )
    return reordered_prompts, reordered_keys
