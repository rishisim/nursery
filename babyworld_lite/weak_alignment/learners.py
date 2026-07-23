from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, fields
import hashlib
import inspect
import math
from typing import Any, Mapping, Sequence

import numpy as np

from babyworld_lite.weak_alignment.protocol import (
    canonical_digest,
    guard_seed_operation,
)
from babyworld_lite.weak_alignment.synthetic import ACTIONS, OBJECTS, EvaluationItem


LEARNERS = (
    "exact_window",
    "latent_mil_single_occurrence",
    "cross_situational_uniform",
    "latent_mil_cross_occurrence",
    "latent_mil_cross_no_null",
    "oracle_alignment",
)


@dataclass(frozen=True, slots=True)
class LexiconModel:
    """Motor/cue-free lexical state allowed to cross the evaluation boundary."""

    action_prototypes: Mapping[str, tuple[float, ...]]
    scene_prototypes: Mapping[str, tuple[float, ...]]
    learner: str
    model_seed: int

    def serializable(self) -> dict[str, Any]:
        return {
            "schema_version": "synthetic-lexicon-model-v1",
            "action_prototypes": {
                key: list(value) for key, value in sorted(self.action_prototypes.items())
            },
            "scene_prototypes": {
                key: list(value) for key, value in sorted(self.scene_prototypes.items())
            },
            "learner": self.learner,
            "model_seed": self.model_seed,
        }


def _rng_for(seed: int, namespace: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{namespace}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    value = np.clip(value, 1e-9, None)
    return value / value.sum()


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / temperature
    scaled -= scaled.max()
    output = np.exp(scaled)
    return output / output.sum()


def _initial_action_prototypes(
    words: Sequence[str], model_seed: int, size: int
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for word in sorted(set(words)):
        rng = _rng_for(model_seed, f"action-initialization-{word}")
        output[word] = rng.dirichlet(np.ones(size) * 4.0)
    return output


def _scene_prototypes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in episodes:
        word = str(row["utterance"]["scene_word"])
        values[word].append(np.asarray(row["scene_observation"], dtype=np.float64))
    return {word: _normalize(np.mean(rows, axis=0)) for word, rows in values.items()}


def _exact_window_prototypes(
    episodes: Sequence[Mapping[str, Any]], initial: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in episodes:
        word = str(row["utterance"]["action_word"])
        speech_time = float(row["utterance"]["speech_time"])
        index = min(
            range(len(row["events"])),
            key=lambda position: abs(float(row["events"][position]["time"]) - speech_time),
        )
        values[word].append(
            np.asarray(row["events"][index]["action_observation"], dtype=np.float64)
        )
    return {
        word: _normalize(np.mean(values[word], axis=0)) if values[word] else initial[word]
        for word in initial
    }


def _uniform_cross_situational_prototypes(
    episodes: Sequence[Mapping[str, Any]], initial: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in episodes:
        word = str(row["utterance"]["action_word"])
        event_mean = np.mean(
            [np.asarray(event["action_observation"], dtype=np.float64) for event in row["events"]],
            axis=0,
        )
        values[word].append(event_mean)
    return {
        word: _normalize(np.mean(values[word], axis=0)) if values[word] else initial[word]
        for word in initial
    }


def _latent_mil_prototypes(
    episodes: Sequence[Mapping[str, Any]],
    initial: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    single_occurrence: bool,
    allow_null: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    learner_config = config["learners"]
    iterations = int(learner_config["iterations"])
    temperature = float(learner_config["posterior_temperature"])
    visual_weight = float(learner_config["visual_weight"])
    temporal_weight = float(learner_config["temporal_weight"])
    side_weight = float(learner_config["side_weight"])
    temporal_scale = float(learner_config["temporal_scale"])
    null_logit = float(learner_config["null_logit"])
    smoothing = float(learner_config["prototype_smoothing"])

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[str(row["utterance"]["action_word"])].append(row)
    if single_occurrence:
        grouped = {word: [rows[0]] for word, rows in grouped.items()}
    prototypes = {word: value.copy() for word, value in initial.items()}
    posterior_null: list[float] = []
    posterior_entropy: list[float] = []
    side_consumed = False
    for _iteration in range(iterations):
        next_prototypes: dict[str, np.ndarray] = {}
        for word in sorted(prototypes):
            accumulator = smoothing * prototypes[word]
            total_weight = smoothing
            for row in grouped.get(word, []):
                observations = np.stack([
                    np.asarray(event["action_observation"], dtype=np.float64)
                    for event in row["events"]
                ])
                times = np.asarray([float(event["time"]) for event in row["events"]])
                speech_time = float(row["utterance"]["speech_time"])
                visual_scores = observations @ prototypes[word]
                temporal_scores = -np.abs(times - speech_time) / temporal_scale
                logits = visual_weight * visual_scores + temporal_weight * temporal_scores
                if allow_null:
                    logits = np.concatenate([logits, np.asarray([null_logit])])
                if "side_scores" in row:
                    side_consumed = True
                    side_scores = np.asarray(row["side_scores"], dtype=np.float64)
                    if allow_null:
                        if len(side_scores) != len(logits):
                            raise ValueError("side score count does not match MIL candidates plus null")
                        logits = logits + side_weight * side_scores
                    else:
                        if len(side_scores) != len(logits) + 1:
                            raise ValueError("side score count does not match no-null MIL candidates")
                        logits = logits + side_weight * side_scores[:-1]
                posterior = _softmax(logits, temperature)
                candidate_posterior = posterior[:-1] if allow_null else posterior
                accumulator += (candidate_posterior[:, None] * observations).sum(axis=0)
                total_weight += float(candidate_posterior.sum())
                if allow_null:
                    posterior_null.append(float(posterior[-1]))
                posterior_entropy.append(
                    float(-(posterior * np.log(np.clip(posterior, 1e-12, None))).sum())
                )
            next_prototypes[word] = _normalize(accumulator / max(total_weight, 1e-9))
        prototypes = next_prototypes
    return prototypes, {
        "iterations": iterations,
        "single_occurrence": single_occurrence,
        "null_option_available": allow_null,
        "training_auxiliary_input_consumed": side_consumed,
        "mean_null_posterior": (
            None if not posterior_null else float(np.mean(posterior_null))
        ),
        "mean_assignment_entropy": (
            None if not posterior_entropy else float(np.mean(posterior_entropy))
        ),
    }


def _oracle_prototypes(
    episodes: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    initial: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    oracle_by_id = {str(row["episode_id"]): row for row in oracle_rows}
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in episodes:
        oracle = oracle_by_id[str(row["episode_id"])]
        target = oracle["target_event_index"]
        if target is None:
            continue
        word = str(row["utterance"]["action_word"])
        values[word].append(
            np.asarray(row["events"][int(target)]["action_observation"], dtype=np.float64)
        )
    return {
        word: _normalize(np.mean(values[word], axis=0)) if values[word] else initial[word]
        for word in initial
    }


def fit_learner(
    *,
    episodes: Sequence[Mapping[str, Any]],
    learner: str,
    corpus_seed: int,
    model_seed: int,
    config: Mapping[str, Any],
    oracle_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[LexiconModel, dict[str, Any]]:
    guard_seed_operation(
        config, operation="train", corpus_seed=int(corpus_seed), model_seed=int(model_seed)
    )
    if learner not in LEARNERS:
        raise KeyError(f"unknown learner: {learner}")
    if int(model_seed) not in set(map(int, config["seeds"]["development"]["model"])):
        raise PermissionError("only frozen development model seeds are authorized in this task")
    words = [str(row["utterance"]["action_word"]) for row in episodes]
    initial = _initial_action_prototypes(words, int(model_seed), len(ACTIONS))
    initial_digest = canonical_digest(
        {word: list(map(float, value)) for word, value in sorted(initial.items())}
    )
    extra: dict[str, Any] = {
        "iterations": int(config["learners"]["iterations"]),
        "single_occurrence": False,
        "null_option_available": False,
        "training_auxiliary_input_consumed": False,
        "mean_null_posterior": None,
        "mean_assignment_entropy": None,
    }
    if learner == "exact_window":
        action = _exact_window_prototypes(episodes, initial)
    elif learner == "cross_situational_uniform":
        action = _uniform_cross_situational_prototypes(episodes, initial)
    elif learner == "latent_mil_single_occurrence":
        action, extra = _latent_mil_prototypes(
            episodes, initial, config, single_occurrence=True, allow_null=True
        )
    elif learner == "latent_mil_cross_occurrence":
        action, extra = _latent_mil_prototypes(
            episodes, initial, config, single_occurrence=False, allow_null=True
        )
    elif learner == "latent_mil_cross_no_null":
        action, extra = _latent_mil_prototypes(
            episodes, initial, config, single_occurrence=False, allow_null=False
        )
    elif learner == "oracle_alignment":
        if oracle_rows is None:
            raise ValueError("oracle_alignment positive control requires separate oracle rows")
        action = _oracle_prototypes(episodes, oracle_rows, initial)
    else:
        raise AssertionError(learner)
    scenes = _scene_prototypes(episodes)
    model = LexiconModel(
        action_prototypes={word: tuple(map(float, value)) for word, value in action.items()},
        scene_prototypes={word: tuple(map(float, value)) for word, value in scenes.items()},
        learner=learner,
        model_seed=int(model_seed),
    )
    trace = {
        "learner": learner,
        "corpus_seed": int(corpus_seed),
        "model_seed": int(model_seed),
        "initialization_digest": initial_digest,
        "data_order_digest": canonical_digest([str(row["episode_id"]) for row in episodes]),
        "inventory_digest": canonical_digest([
            {key: value for key, value in row.items() if key != "side_scores"}
            for row in episodes
        ]),
        "optimizer_steps": int(config["learners"]["iterations"]),
        "training_examples": len(episodes),
        "serialized_model_digest": canonical_digest(model.serializable()),
        **extra,
    }
    return model, trace


def _macro_accuracy(correct: Sequence[bool], groups: Sequence[str]) -> float:
    by_group: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(correct, groups):
        by_group[str(group)].append(float(value))
    return float(np.mean([np.mean(values) for values in by_group.values()]))


def _prototype_or_uniform(
    prototypes: Mapping[str, Sequence[float]], word: str, size: int
) -> np.ndarray:
    if word not in prototypes:
        return np.full(size, 1.0 / size, dtype=np.float64)
    return np.asarray(prototypes[word], dtype=np.float64)


def evaluate_without_auxiliary_modality(
    model: LexiconModel,
    evaluation_items: Sequence[EvaluationItem],
    *,
    corpus_seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Final evaluation accepts only lexical state and cue-free evaluation items."""
    guard_seed_operation(
        config, operation="evaluate", corpus_seed=int(corpus_seed), model_seed=model.model_seed
    )
    if any(not isinstance(item, EvaluationItem) for item in evaluation_items):
        raise TypeError("final evaluation requires the fail-closed EvaluationItem schema")
    forbidden = {"side", "cue", "motor", "imu", "touch"}
    item_fields = {field.name for field in fields(EvaluationItem)}
    model_fields = {field.name for field in fields(LexiconModel)}
    if any(any(token in name.lower() for token in forbidden) for name in item_fields | model_fields):
        raise RuntimeError("auxiliary-modality field crossed the final evaluation boundary")

    action_correct: list[bool] = []
    action_groups: list[str] = []
    scene_correct: list[bool] = []
    scene_groups: list[str] = []
    roles: list[str] = []
    compositions: list[str] = []
    for item in evaluation_items:
        action_observation = np.asarray(item.action_observation, dtype=np.float64)
        action_scores = np.asarray([
            float(action_observation @ _prototype_or_uniform(
                model.action_prototypes, word, len(ACTIONS)
            ))
            for word in item.action_candidate_words
        ])
        scene_observation = np.asarray(item.scene_observation, dtype=np.float64)
        scene_scores = np.asarray([
            float(scene_observation @ _prototype_or_uniform(
                model.scene_prototypes, word, len(OBJECTS)
            ))
            for word in item.scene_candidate_words
        ])
        action_correct.append(int(action_scores.argmax()) == item.action_answer_index)
        scene_correct.append(int(scene_scores.argmax()) == item.scene_answer_index)
        action_groups.append(item.action_candidate_words[item.action_answer_index])
        scene_groups.append(item.scene_candidate_words[item.scene_answer_index])
        roles.append(item.evaluation_role)
        compositions.append(item.composition_id)

    primary_indices = [index for index, role in enumerate(roles) if role == "primary_heldout_lexemes"]
    anchor_indices = [index for index, role in enumerate(roles) if role == "anchor_lexemes"]
    zero_indices = [
        index for index, role in enumerate(roles)
        if role == "zero_exposure_lexical_type_negative_control"
    ]

    def subset_macro(values: Sequence[bool], groups: Sequence[str], indices: Sequence[int]) -> float:
        return _macro_accuracy([values[index] for index in indices], [groups[index] for index in indices])

    return {
        "heldout_composition_action_6way_macro_accuracy": subset_macro(
            action_correct, action_groups, primary_indices
        ),
        "heldout_composition_noun_control_6way_macro_accuracy": subset_macro(
            scene_correct, scene_groups, primary_indices
        ),
        "anchor_action_6way_macro_accuracy": subset_macro(
            action_correct, action_groups, anchor_indices
        ),
        "zero_exposure_action_6way_macro_accuracy": subset_macro(
            action_correct, action_groups, zero_indices
        ),
        "n_primary_evaluation_items": len(primary_indices),
        "n_anchor_evaluation_items": len(anchor_indices),
        "n_zero_exposure_items": len(zero_indices),
        "n_primary_heldout_compositions": len({compositions[index] for index in primary_indices}),
        "evaluation_modalities": ["action_observation", "scene_observation", "lexical_prototypes"],
        "training_auxiliary_modality_structurally_unavailable": True,
        "evaluation_item_fields": sorted(item_fields),
        "model_fields": sorted(model_fields),
    }


def lexical_mapping_accuracy(
    model: LexiconModel,
    action_word_to_action: Mapping[str, str],
    *,
    eligible_words: Sequence[str] | None = None,
) -> float:
    words = sorted(
        set(action_word_to_action)
        if eligible_words is None
        else set(map(str, eligible_words)) & set(action_word_to_action)
    )
    if not words:
        return float("nan")
    return float(np.mean([
        ACTIONS[int(np.asarray(model.action_prototypes.get(
            word, tuple([1.0 / len(ACTIONS)] * len(ACTIONS))
        )).argmax())] == action_word_to_action[word]
        for word in words
    ]))


def modality_withholding_contract() -> dict[str, Any]:
    signature = inspect.signature(evaluate_without_auxiliary_modality)
    item_fields = [field.name for field in fields(EvaluationItem)]
    model_fields = [field.name for field in fields(LexiconModel)]
    forbidden = ("side", "cue", "motor", "imu", "touch")
    checks = {
        "evaluator_signature_has_no_auxiliary_argument": not any(
            any(token in name.lower() for token in forbidden)
            for name in signature.parameters
        ),
        "evaluation_item_has_no_auxiliary_field": not any(
            any(token in name.lower() for token in forbidden) for name in item_fields
        ),
        "serialized_model_has_no_auxiliary_field_or_encoder": not any(
            any(token in name.lower() for token in forbidden) for name in model_fields
        ),
        "evaluation_source_has_no_encoder_call": "encode_" not in inspect.getsource(
            evaluate_without_auxiliary_modality
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "evaluator_signature": str(signature),
        "evaluation_item_fields": item_fields,
        "lexicon_model_fields": model_fields,
    }
