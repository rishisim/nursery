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


def _initial_distribution(
    token: str,
    slot: str,
    model_seed: int,
    jitter: float,
    initial_presence_probability: float,
) -> np.ndarray:
    dimension = SLOT_DIMENSIONS[slot]
    semantic = np.ones(dimension)
    for index in range(dimension):
        digest = hashlib.sha256(
            f"v8-init|{model_seed}|{slot}|{token}|{index}".encode()
        ).digest()
        semantic[index] += jitter * (
            int.from_bytes(digest[:4], "little") / (2**32 - 1) - 0.5
        )
    semantic = _normalize(semantic)
    presence = float(np.clip(initial_presence_probability, 1e-6, 1.0 - 1e-6))
    return np.concatenate([presence * semantic, np.asarray([1.0 - presence])])


@dataclass(frozen=True, slots=True)
class LexicalModel:
    distributions: dict[str, tuple[float, ...]]
    learner: str
    model_seed: int
    iterations: int
    competition_weight: float

    def serializable(self) -> dict[str, Any]:
        return {
            "schema_version": "nursery-v8-lexical-model",
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


def _visible_cooccurrence_initialization(
    rows: Sequence[Mapping[str, Any]],
    token_slots: Mapping[str, str],
    *,
    presence_probability: float,
    competition: float,
    smoothing: float,
    passes: int,
    model_seed: int,
    jitter: float,
) -> dict[str, np.ndarray]:
    counts = {
        key: np.full(SLOT_DIMENSIONS[slot], smoothing, dtype=float)
        for key, slot in token_slots.items()
    }
    for row in rows:
        events = list(row["events"])
        for item in row["utterance"]["items"]:
            key = _token_key(item)
            slot = str(item["slot"])
            counts[key] += np.mean(
                [_event_observation(event, slot) for event in events],
                axis=0,
            )
    for key, values in counts.items():
        for index in range(len(values)):
            digest = hashlib.sha256(
                f"v8-cooccurrence-jitter|{model_seed}|{key}|{index}".encode()
            ).digest()
            centered = int.from_bytes(digest[:4], "little") / (2**32 - 1) - 0.5
            values[index] *= 1.0 + jitter * centered
    semantics = {key: _normalize(value) for key, value in counts.items()}
    for _ in range(passes):
        for slot in SLOT_DIMENSIONS:
            keys = sorted(key for key, value in token_slots.items() if value == slot)
            if not keys:
                continue
            matrix = np.stack([semantics[key] for key in keys])
            mass = np.maximum(matrix.sum(axis=0), 1e-12)
            exclusive = matrix / np.power(mass[None, :], competition)
            for key, row_value in zip(keys, exclusive):
                semantics[key] = _normalize(row_value)
    presence = float(np.clip(presence_probability, 1e-6, 1.0 - 1e-6))
    return {
        key: np.concatenate(
            [presence * semantics[key], np.asarray([1.0 - presence])]
        )
        for key in semantics
    }


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
    if bool(learner_config["visible_cooccurrence_initialization"]):
        distributions = _visible_cooccurrence_initialization(
            rows,
            token_slots,
            presence_probability=float(
                learner_config["initial_presence_probability"]
            ),
            competition=competition,
            smoothing=float(learner_config["smoothing"]),
            passes=int(learner_config["cooccurrence_initialization_passes"]),
            model_seed=model_seed,
            jitter=jitter,
        )
    else:
        distributions = {
            key: _initial_distribution(
                key.split("|", 1)[1],
                slot,
                model_seed,
                jitter,
                float(learner_config["initial_presence_probability"]),
            )
            for key, slot in sorted(token_slots.items())
        }
    temperature = float(learner_config["temperature"])
    sensor_weight = float(learner_config["sensor_weight"])
    sensor_null_calibration_weight = float(
        learner_config["sensor_null_calibration_weight"]
    )
    sensor_calibration_temperature = float(
        learner_config["sensor_calibration_temperature"]
    )
    sensor_joint_update_weight = float(
        learner_config["sensor_joint_update_weight"]
    )
    sensor_semantic_sharpening_weight = float(
        learner_config["sensor_semantic_sharpening_weight"]
    )
    update_rate = float(learner_config["update_rate"])
    smoothing = float(learner_config["smoothing"])
    presence_smoothing = float(learner_config["presence_smoothing"])
    lag_scale = float(learner_config["lag_scale"])
    null_margin_alpha = float(learner_config["null_margin_alpha"])
    no_absence_null_mass = float(learner_config["no_absence_null_mass"])
    objectives: list[float] = []
    competition_penalties: list[float] = []
    iteration_digests: list[str] = []
    sensor_applications = {"primitive": 0, "manner": 0, "noun": 0}
    mean_nulls: list[float] = []
    inferred_presence_counts_by_iteration: list[dict[str, dict[str, int]]] = []
    sensor_null_logit_adjustments_by_iteration: list[dict[str, float]] = []
    sensor_semantic_corroboration_by_iteration: list[dict[str, float]] = []
    ordered_rows = sorted(rows, key=lambda row: str(row["episode_id"]))
    for _ in range(iterations):
        semantic_accumulators = {
            key: np.full(SLOT_DIMENSIONS[slot], smoothing, dtype=float)
            for key, slot in token_slots.items()
        }
        inferred_counts = {
            key: {"present": 0, "null": 0}
            for key in token_slots
        }
        sensor_null_logit_adjustments: defaultdict[str, list[float]] = defaultdict(
            list
        )
        sensor_semantic_corroboration: defaultdict[str, list[float]] = defaultdict(
            list
        )
        objective = 0.0
        null_assignments: list[float] = []
        for row in ordered_rows:
            items = list(row["utterance"]["items"])
            events = list(row["events"])
            midpoints = [(float(event["start"]) + float(event["end"])) / 2 for event in events]
            speech_time = float(row["utterance"]["speech_time"])
            lag_weight = math.exp(-min(abs(speech_time - midpoint) for midpoint in midpoints) / lag_scale)
            detector_value = evidence.get(str(row["episode_id"]))
            joint_responsibilities: np.ndarray | None = None
            joint_update_scale = 0.0
            if (
                row["family"] == "action"
                and len(items) > 1
                and detector_value is not None
            ):
                joint_scores = []
                for event in events:
                    joint_scores.append(
                        sum(
                            float(
                                np.log(
                                    np.clip(
                                        _normalize(
                                            np.asarray(
                                                distributions[_token_key(item)],
                                                dtype=float,
                                            )[:-1]
                                        )
                                        @ _event_observation(
                                            event,
                                            str(item["slot"]),
                                        ),
                                        1e-12,
                                        1.0,
                                    )
                                )
                            )
                            for item in items
                        )
                        / len(items)
                    )
                quality = float(detector_value["quality"])
                language_top_index = int(np.argmax(joint_scores))
                detector_logits = list(
                    map(float, detector_value["event_logits"])
                )
                if len(detector_logits) != len(events):
                    raise ValueError("detector/event count mismatch")
                detector_probabilities_for_joint_update = _softmax(
                    [
                        *detector_logits,
                        float(detector_value["null_logit"]),
                    ],
                    sensor_calibration_temperature,
                )
                detector_top_index = int(
                    np.argmax(detector_probabilities_for_joint_update)
                )
                detector_top_probability = float(
                    detector_probabilities_for_joint_update[detector_top_index]
                )
                exact_chance = 1.0 / (len(events) + 1)
                agreement = detector_top_index == language_top_index
                joint_update_scale = (
                    quality
                    * float(agreement)
                    * max(
                        0.0,
                        (detector_top_probability - exact_chance)
                        / (1.0 - exact_chance),
                    )
                )
                for item in items:
                    sensor_semantic_corroboration[_token_key(item)].append(
                        joint_update_scale
                    )
                joint_scores = [
                    score
                    + sensor_weight
                    * joint_update_scale
                    * float(np.clip(detector_logits[index], -5, 5))
                    for index, score in enumerate(joint_scores)
                ]
                for item in items:
                    sensor_applications[str(item["slot"])] += 1
                joint_responsibilities = _softmax(
                    joint_scores,
                    temperature,
                )
                maximum = max(joint_scores)
                objective += lag_weight * float(
                    np.log(
                        np.exp(np.asarray(joint_scores) - maximum).sum()
                    )
                    + maximum
                )
            for item in items:
                key = _token_key(item)
                slot = str(item["slot"])
                distribution = np.asarray(distributions[key], dtype=float)
                semantic = _normalize(distribution[:-1])
                inferred_concept = int(np.argmax(semantic))
                observations = [
                    _event_observation(event, slot)
                    for event in events
                ]
                matching = [
                    index
                    for index, observation in enumerate(observations)
                    if int(np.argmax(observation)) == inferred_concept
                ]
                if row["family"] == "action" and detector_value is not None:
                    quality = float(detector_value["quality"])
                    detector_probabilities = _softmax(
                        [
                            *map(float, detector_value["event_logits"]),
                            float(detector_value["null_logit"]),
                        ],
                        sensor_calibration_temperature,
                    )
                    if matching:
                        observed_support = float(
                            detector_probabilities[matching].sum()
                        )
                        chance_support = len(matching) / (len(events) + 1)
                        signed_null_adjustment = -quality * (
                            observed_support - chance_support
                        )
                    else:
                        observed_support = float(detector_probabilities[-1])
                        chance_support = 1.0 / (len(events) + 1)
                        signed_null_adjustment = quality * (
                            observed_support - chance_support
                        )
                    sensor_null_logit_adjustments[key].append(
                        signed_null_adjustment
                    )
                if not matching:
                    inferred_counts[key]["null"] += 1
                    null_assignments.append(1.0)
                    objective += lag_weight * float(
                        np.log(np.clip(distribution[-1], 1e-12, 1.0))
                    )
                    continue
                inferred_counts[key]["present"] += 1
                null_assignments.append(0.0)
                if joint_responsibilities is not None:
                    consensus_index = int(np.argmax(joint_responsibilities))
                    if joint_update_scale > 0.0:
                        semantic_accumulators[key] += (
                            lag_weight
                            * sensor_joint_update_weight
                            * joint_update_scale
                            * observations[consensus_index]
                        )
                scores = [
                    float(
                        np.log(
                            np.clip(
                                semantic @ observations[index],
                                1e-12,
                                1.0,
                            )
                        )
                    )
                    for index in matching
                ]
                if row["family"] == "action" and detector_value is not None:
                    quality = float(detector_value["quality"])
                    detector_logits = list(map(float, detector_value["event_logits"]))
                    if len(detector_logits) != len(events):
                        raise ValueError("detector/event count mismatch")
                    scores = [
                        score
                        + sensor_weight
                        * quality
                        * float(np.clip(detector_logits[index], -5, 5))
                        for score, index in zip(scores, matching)
                    ]
                    sensor_applications[slot] += 1
                responsibilities = _softmax(scores, temperature)
                maximum = max(scores)
                objective += lag_weight * float(
                    np.log(np.exp(np.asarray(scores) - maximum).sum()) + maximum
                )
                for responsibility, index in zip(responsibilities, matching):
                    semantic_accumulators[key] += (
                        lag_weight * responsibility * observations[index]
                    )
        conditional_semantics = {
            key: _normalize(values) for key, values in semantic_accumulators.items()
        }
        penalty = 0.0
        for slot in SLOT_DIMENSIONS:
            keys = sorted(key for key, value in token_slots.items() if value == slot)
            if not keys:
                continue
            matrix = np.stack([conditional_semantics[key] for key in keys])
            mass = np.maximum(matrix.sum(axis=0), 1e-12)
            exclusive = matrix / np.power(mass[None, :], competition)
            penalty += float(np.sum((matrix - exclusive) ** 2))
            for key, row_value in zip(keys, exclusive):
                conditional_semantics[key] = _normalize(row_value)
        updated_distributions = {}
        for key, semantic_estimate in conditional_semantics.items():
            old_distribution = np.asarray(distributions[key], dtype=float)
            old_semantic = _normalize(old_distribution[:-1])
            semantic = _normalize(
                (1.0 - update_rate) * old_semantic
                + update_rate * semantic_estimate
            )
            mean_sensor_corroboration = float(
                np.mean(sensor_semantic_corroboration.get(key, [0.0]))
            )
            semantic = _normalize(
                np.power(
                    semantic,
                    1.0
                    + sensor_semantic_sharpening_weight
                    * mean_sensor_corroboration,
                )
            )
            ordered = np.sort(semantic)[::-1]
            top = float(ordered[0])
            runner_up = float(ordered[1]) if len(ordered) > 1 else 1e-12
            if (
                inferred_counts[key]["present"] > 0
                and inferred_counts[key]["null"] > 0
            ):
                null_odds = math.exp(
                    null_margin_alpha * math.log(np.clip(top, 1e-12, 1.0))
                    + (1.0 - null_margin_alpha)
                    * math.log(np.clip(runner_up, 1e-12, 1.0))
                )
                null_mass = null_odds / (1.0 + null_odds)
            else:
                null_mass = no_absence_null_mass
            mean_sensor_adjustment = float(
                np.mean(sensor_null_logit_adjustments.get(key, [0.0]))
            )
            base_logit = math.log(
                np.clip(null_mass, 1e-12, 1.0 - 1e-12)
                / np.clip(1.0 - null_mass, 1e-12, 1.0)
            )
            adjusted_logit = (
                base_logit
                + sensor_null_calibration_weight * mean_sensor_adjustment
            )
            null_mass = 1.0 / (1.0 + math.exp(-adjusted_logit))
            updated_distributions[key] = np.concatenate(
                [
                    (1.0 - null_mass) * semantic,
                    np.asarray([null_mass]),
                ]
            )
        distributions = updated_distributions
        objectives.append(float(objective - competition * penalty))
        competition_penalties.append(penalty)
        mean_nulls.append(float(np.mean(null_assignments)))
        inferred_presence_counts_by_iteration.append(
            {
                key: dict(value)
                for key, value in sorted(inferred_counts.items())
            }
        )
        sensor_null_logit_adjustments_by_iteration.append(
            {
                key: float(np.mean(sensor_null_logit_adjustments.get(key, [0.0])))
                for key in sorted(token_slots)
            }
        )
        sensor_semantic_corroboration_by_iteration.append(
            {
                key: float(
                    np.mean(sensor_semantic_corroboration.get(key, [0.0]))
                )
                for key in sorted(token_slots)
            }
        )
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
        "mean_model_inferred_null_rate_by_iteration": mean_nulls,
        "inferred_presence_counts_by_iteration": inferred_presence_counts_by_iteration,
        "sensor_null_logit_adjustments_by_iteration": sensor_null_logit_adjustments_by_iteration,
        "sensor_null_calibration_weight": sensor_null_calibration_weight,
        "sensor_calibration_temperature": sensor_calibration_temperature,
        "sensor_joint_update_weight": sensor_joint_update_weight,
        "sensor_semantic_sharpening_weight": sensor_semantic_sharpening_weight,
        "sensor_semantic_corroboration_by_iteration": sensor_semantic_corroboration_by_iteration,
        "sensor_applications_by_slot": sensor_applications,
        "noun_sensor_weight": float(learner_config["noun_sensor_weight"]),
        "null_parameterization": (
            "model_inferred_visible_candidate_absence_with_semantic_margin_calibration"
        ),
        "candidate_count_normalization": bool(
            learner_config["candidate_count_normalization"]
        ),
        "initial_presence_probability": float(
            learner_config["initial_presence_probability"]
        ),
        "presence_smoothing": presence_smoothing,
        "null_margin_alpha": null_margin_alpha,
        "no_absence_null_mass": no_absence_null_mass,
        "visible_cooccurrence_initialization": bool(
            learner_config["visible_cooccurrence_initialization"]
        ),
        "cooccurrence_initialization_passes": int(
            learner_config["cooccurrence_initialization_passes"]
        ),
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
        ) / len(tokens)
    return sum(
        _lexical_log_score(
            np.asarray(model.distributions[_token_key(item)], dtype=float),
            _event_observation(candidate, str(item["slot"])),
        )
        for item in tokens
    ) / len(tokens)


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
                "candidate_count": len(scores),
                "exact_candidate_set_chance": 1.0 / len(scores),
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
        chances = np.asarray(
            [float(row["exact_candidate_set_chance"]) for row in selected]
        )
        chance_log_loss = float(np.mean(-np.log(chances)))
        chance_brier = float(
            np.mean(
                [
                    (int(row["candidate_count"]) - 1)
                    / (int(row["candidate_count"]) ** 2)
                    for row in selected
                ]
            )
        )
        observed_brier = float(np.mean([row["brier"] for row in selected]))
        observed_log_loss = float(np.mean([row["log_loss"] for row in selected]))
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
            "mean_exact_candidate_set_chance": float(
                np.mean(chances)
            ),
            "fractional_margin_over_chance": float(
                np.mean(correctness)
                - np.mean(chances)
            ),
            "probability_margin_over_chance": float(
                np.mean(confidences)
                - np.mean(chances)
            ),
            "brier": observed_brier,
            "exact_chance_brier": chance_brier,
            "brier_improvement_over_exact_chance": chance_brier - observed_brier,
            "log_loss": observed_log_loss,
            "exact_chance_log_loss": chance_log_loss,
            "log_loss_improvement_over_exact_chance": (
                chance_log_loss - observed_log_loss
            ),
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
    by_kind_presence = {
        kind: {
            presence: summarize(
                [
                    row
                    for row in rows
                    if row["kind"] == kind and row["presence"] == presence
                ]
            )
            for presence in ("present", "null")
        }
        for kind in ("primitive", "manner", "action", "noun")
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
        "by_kind_presence": by_kind_presence,
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
                "schema_version": "nursery-v8-sealed-key",
                "prompt_id": prompt["prompt_id"],
                "answer_index": next(
                    index for index, candidate in enumerate(candidates) if candidate["candidate_id"] == correct_id
                ),
            }
        )
    return reordered_prompts, reordered_keys
