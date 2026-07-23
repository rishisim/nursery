from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
import inspect
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import canonical_digest, guard_seed_operation
from .synthetic import EvaluationItem, FINAL_FORBIDDEN_FRAGMENTS, rng_for


LEARNERS = (
    "exact_window_symbolic",
    "sensor_latent_single_occurrence",
    "cross_occurrence_no_sensor",
    "sensor_latent_cross_occurrence",
    "sensor_latent_cross_no_null",
    "structural_absence",
    "oracle_event_alignment_upper",
    "v1_pointer_style_upper",
)


@dataclass(frozen=True, slots=True)
class LexicalModel:
    token_prototypes: Mapping[str, tuple[float, ...]]
    learner: str
    model_seed: int

    def serializable(self) -> dict[str, Any]:
        return {
            "schema_version": "synthetic-lexical-model-v2",
            "token_prototypes": {
                key: list(value) for key, value in sorted(self.token_prototypes.items())
            },
            "learner": self.learner,
            "model_seed": self.model_seed,
        }


@dataclass(frozen=True, slots=True)
class EvaluationPolicy:
    allowed_corpora: tuple[int, ...]
    allowed_models: tuple[int, ...]
    blocked_values: tuple[int, ...]


def make_evaluation_policy(config: Mapping[str, Any]) -> EvaluationPolicy:
    reserve = config["seeds"]["confirmation_reserve"]
    return EvaluationPolicy(
        allowed_corpora=tuple(map(int, config["seeds"]["development"]["corpus"])),
        allowed_models=tuple(map(int, config["seeds"]["development"]["model"])),
        blocked_values=tuple(
            sorted(
                {
                    *map(int, reserve["corpus"]),
                    *map(int, reserve["model"]),
                    *map(int, reserve["calibration"]),
                }
            )
        ),
    )


def _normalize(value: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(value, dtype=np.float64), 1e-9, None)
    return result / result.sum()


def _softmax(value: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(value, dtype=np.float64) / temperature
    scaled -= scaled.max()
    output = np.exp(scaled)
    return output / output.sum()


def _slot_dimension(slot: str) -> int:
    return {"primitive": 3, "manner": 2, "noun": 6}[slot]


def _observation(event: Mapping[str, Any], slot: str) -> np.ndarray:
    if slot == "primitive":
        return np.asarray(event["action_observation"][:3], dtype=np.float64)
    if slot == "manner":
        return np.asarray(event["action_observation"][3:], dtype=np.float64)
    if slot == "noun":
        return np.asarray(event["object_observation"], dtype=np.float64)
    raise KeyError(slot)


def _key(slot: str, token: str) -> str:
    return f"{slot}|{token}"


def _occurrences(
    episodes: Sequence[Mapping[str, Any]], *, single_occurrence: bool
) -> dict[str, list[tuple[str, str, Mapping[str, Any]]]]:
    grouped: dict[str, list[tuple[str, str, Mapping[str, Any]]]] = defaultdict(list)
    for row in episodes:
        for item in row["utterance"]["items"]:
            slot = str(item["slot"])
            if slot == "ignore":
                continue
            token = str(item["token"])
            grouped[_key(slot, token)].append((slot, token, row))
    if single_occurrence:
        return {key: [values[0]] for key, values in grouped.items()}
    return dict(grouped)


def _initial_prototypes(
    grouped: Mapping[str, Sequence[tuple[str, str, Mapping[str, Any]]]], model_seed: int
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, rows in sorted(grouped.items()):
        slot = rows[0][0]
        dimension = _slot_dimension(slot)
        observations = [
            _observation(event, slot)
            for _, _, row in rows
            for event in row["events"]
        ]
        empirical = _normalize(np.mean(observations, axis=0))
        random = rng_for(model_seed, f"initial-{key}").dirichlet(
            np.ones(dimension) * 2.5
        )
        output[key] = _normalize(0.62 * empirical + 0.38 * random)
    return output


def _concentration(value: np.ndarray) -> float:
    value = _normalize(value)
    dimension = len(value)
    return float((value.max() - 1.0 / dimension) / (1.0 - 1.0 / dimension))


def _trust_by_key(
    grouped: Mapping[str, Sequence[tuple[str, str, Mapping[str, Any]]]],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    single_occurrence: bool,
) -> dict[str, dict[str, float]]:
    learner_config = config["learners"]
    output: dict[str, dict[str, float]] = {}
    for key, rows in sorted(grouped.items()):
        slot = rows[0][0]
        uniform_observations: list[np.ndarray] = []
        weighted_sum = np.zeros(_slot_dimension(slot), dtype=np.float64)
        weighted_total = 0.0
        quality_values: list[float] = []
        supported = 0
        for _, _, row in rows:
            episode_id = str(row["episode_id"])
            if episode_id not in evidence:
                continue
            row_evidence = evidence[episode_id]
            logits = np.asarray(row_evidence["event_logits"], dtype=np.float64)
            posterior = _softmax(logits, float(learner_config["posterior_temperature"]))
            observations = np.stack([_observation(event, slot) for event in row["events"]])
            weighted_sum += (posterior[:, None] * observations).sum(axis=0)
            weighted_total += float(posterior.sum())
            uniform_observations.extend(list(observations))
            quality_values.append(
                float(row_evidence["quality"]) * float(row_evidence["availability"])
            )
            supported += 1
        if supported == 0:
            output[key] = {
                "trust": 0.0,
                "concentration_gain": 0.0,
                "mean_quality": 0.0,
                "supported_occurrences": 0.0,
            }
            continue
        weighted = _normalize(weighted_sum / max(weighted_total, 1e-9))
        uniform = _normalize(np.mean(uniform_observations, axis=0))
        gain = _concentration(weighted) - _concentration(uniform)
        quality = float(np.mean(quality_values))
        if single_occurrence:
            raw = 0.55 * quality
        else:
            threshold = float(learner_config["reliability_gain_threshold"])
            scale = float(learner_config["reliability_gain_scale"])
            raw = 1.0 / (1.0 + math.exp(-((gain - threshold) / scale)))
            raw *= quality
        trust = min(float(learner_config["maximum_sensor_trust"]), raw)
        if trust < float(learner_config["minimum_active_trust"]):
            trust = 0.0
        output[key] = {
            "trust": float(trust),
            "concentration_gain": float(gain),
            "mean_quality": quality,
            "supported_occurrences": float(supported),
        }
    return output


def _exact_window(
    grouped: Mapping[str, Sequence[tuple[str, str, Mapping[str, Any]]]],
    initial: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, rows in grouped.items():
        selected: list[np.ndarray] = []
        for slot, _, row in rows:
            speech = int(row["utterance"]["speech_time"])
            index = min(
                range(len(row["events"])),
                key=lambda position: abs(
                    (int(row["events"][position]["start"]) + int(row["events"][position]["end"]))
                    / 2.0
                    - speech
                ),
            )
            selected.append(_observation(row["events"][index], slot))
        output[key] = _normalize(np.mean(selected, axis=0)) if selected else initial[key]
    return output


def _oracle_alignment(
    grouped: Mapping[str, Sequence[tuple[str, str, Mapping[str, Any]]]],
    oracle_rows: Sequence[Mapping[str, Any]],
    initial: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    oracle_by_id = {str(row["episode_id"]): row for row in oracle_rows}
    output: dict[str, np.ndarray] = {}
    for key, rows in grouped.items():
        selected: list[np.ndarray] = []
        for slot, _, row in rows:
            target = oracle_by_id[str(row["episode_id"])]["target_event_index"]
            if target is not None:
                selected.append(_observation(row["events"][int(target)], slot))
        output[key] = _normalize(np.mean(selected, axis=0)) if selected else initial[key]
    return output


def _pointer_evidence(
    episodes: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    corpus_seed: int,
) -> dict[str, dict[str, Any]]:
    oracle_by_id = {str(row["episode_id"]): row for row in oracle_rows}
    probability = float(config["learners"]["v1_pointer_correct_probability"])
    output: dict[str, dict[str, Any]] = {}
    for row in episodes:
        episode_id = str(row["episode_id"])
        target = oracle_by_id[episode_id]["target_event_index"]
        correct = len(row["events"]) if target is None else int(target)
        rng = rng_for(corpus_seed, f"pointer-{episode_id}")
        selected = correct if rng.random() < probability else int(
            rng.integers(0, len(row["events"]) + 1)
        )
        logits = np.linspace(-1.0, 1.0, len(row["events"]) + 1)
        rng.shuffle(logits)
        logits[selected] = 2.0
        output[episode_id] = {
            "event_logits": tuple(map(float, logits[:-1])),
            "null_logit": float(logits[-1]),
            "quality": 1.0,
            "availability": 1.0,
        }
    return output


def _latent_fit(
    grouped: Mapping[str, Sequence[tuple[str, str, Mapping[str, Any]]]],
    initial: Mapping[str, np.ndarray],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    model_seed: int,
    allow_null: bool,
    use_evidence: bool,
    single_occurrence: bool,
    iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    learner_config = config["learners"]
    temperature = float(learner_config["posterior_temperature"])
    visual_weight = float(learner_config["visual_weight"])
    temporal_weight = float(learner_config["temporal_weight"])
    temporal_scale = float(learner_config["temporal_scale_samples"])
    null_logit = float(learner_config["null_logit"])
    smoothing = float(learner_config["prototype_smoothing"])
    channel_weight = float(learner_config["sensor_logit_weight"])
    dropout = float(learner_config["cue_dropout_probability"])
    trust_info = (
        _trust_by_key(
            grouped,
            evidence,
            config,
            single_occurrence=single_occurrence,
        )
        if use_evidence
        else {
            key: {
                "trust": 0.0,
                "concentration_gain": 0.0,
                "mean_quality": 0.0,
                "supported_occurrences": 0.0,
            }
            for key in grouped
        }
    )
    prototypes = {key: value.copy() for key, value in initial.items()}
    null_posteriors: list[float] = []
    entropy_values: list[float] = []
    active_channel_updates = 0
    possible_channel_updates = 0
    for iteration in range(iterations):
        next_prototypes: dict[str, np.ndarray] = {}
        for key, rows in sorted(grouped.items()):
            accumulator = smoothing * prototypes[key]
            total = smoothing
            dimension = len(prototypes[key])
            trust = float(trust_info[key]["trust"])
            for occurrence_index, (slot, _, row) in enumerate(rows):
                observations = np.stack(
                    [_observation(event, slot) for event in row["events"]]
                )
                centers = np.asarray(
                    [
                        (int(event["start"]) + int(event["end"])) / 2.0
                        for event in row["events"]
                    ],
                    dtype=np.float64,
                )
                speech = float(row["utterance"]["speech_time"])
                visual = (observations @ prototypes[key] - 1.0 / dimension) * dimension
                temporal = -np.abs(centers - speech) / temporal_scale
                logits = visual_weight * visual + temporal_weight * temporal
                if allow_null:
                    logits = np.concatenate([logits, np.asarray([null_logit])])
                episode_id = str(row["episode_id"])
                if use_evidence and episode_id in evidence:
                    possible_channel_updates += 1
                    keep_rng = rng_for(
                        model_seed,
                        f"drop-{key}-{episode_id}-{iteration}-{occurrence_index}",
                    )
                    effective = trust if keep_rng.random() >= dropout else 0.0
                    if effective > 0:
                        row_evidence = evidence[episode_id]
                        extra = np.asarray(row_evidence["event_logits"], dtype=np.float64)
                        if allow_null:
                            extra = np.concatenate(
                                [extra, np.asarray([float(row_evidence["null_logit"])])]
                            )
                        logits += channel_weight * effective * extra
                        active_channel_updates += 1
                posterior = _softmax(logits, temperature)
                candidate_posterior = posterior[:-1] if allow_null else posterior
                accumulator += (candidate_posterior[:, None] * observations).sum(axis=0)
                total += float(candidate_posterior.sum())
                if allow_null:
                    null_posteriors.append(float(posterior[-1]))
                entropy_values.append(
                    float(-np.sum(posterior * np.log(np.clip(posterior, 1e-12, None))))
                )
            next_prototypes[key] = _normalize(accumulator / max(total, 1e-9))
        prototypes = next_prototypes
    return prototypes, {
        "iterations": iterations,
        "single_occurrence": single_occurrence,
        "null_option_available": allow_null,
        "derived_training_evidence_available": bool(evidence),
        "derived_training_evidence_used": use_evidence and active_channel_updates > 0,
        "active_evidence_updates": active_channel_updates,
        "possible_evidence_updates": possible_channel_updates,
        "mean_null_posterior": (
            None if not null_posteriors else float(np.mean(null_posteriors))
        ),
        "mean_assignment_entropy": (
            None if not entropy_values else float(np.mean(entropy_values))
        ),
        "reliability_by_token": trust_info,
    }


def _sensor_fused_fit(
    grouped: Mapping[str, Sequence[tuple[str, str, Mapping[str, Any]]]],
    initial: Mapping[str, np.ndarray],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    model_seed: int,
    allow_null: bool,
    single_occurrence: bool,
    iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    learner_config = config["learners"]
    baseline, _ = _latent_fit(
        grouped,
        initial,
        {},
        config,
        model_seed=model_seed,
        allow_null=True,
        use_evidence=False,
        single_occurrence=single_occurrence,
        iterations=iterations,
    )
    threshold = float(learner_config["reliability_mean_quality_threshold"])
    scale = float(learner_config["reliability_mean_quality_scale"])
    change_threshold = float(
        learner_config["reliability_concentration_change_threshold"]
    )
    change_scale = float(learner_config["reliability_concentration_change_scale"])
    minimum = float(learner_config["minimum_active_trust"])
    maximum = float(learner_config["maximum_sensor_trust"])
    temperature = float(learner_config["posterior_temperature"])
    channel_weight = float(learner_config["sensor_logit_weight"])
    dropout = float(learner_config["cue_dropout_probability"])
    output: dict[str, np.ndarray] = {}
    reliability: dict[str, dict[str, float]] = {}
    episode_ids_by_family: dict[str, set[str]] = defaultdict(set)
    for key, rows in grouped.items():
        family = "noun" if key.startswith("noun|") else "action"
        episode_ids_by_family[family].update(
            str(row["episode_id"]) for _, _, row in rows
        )
    family_quality: dict[str, float] = {}
    family_concentration_change: dict[str, float] = {}
    family_trust: dict[str, float] = {}
    token_diagnostics = _trust_by_key(
        grouped,
        evidence,
        config,
        single_occurrence=single_occurrence,
    )
    for family, episode_ids in episode_ids_by_family.items():
        represented = [
            evidence[episode_id]
            for episode_id in sorted(episode_ids)
            if episode_id in evidence
        ]
        mean_quality = (
            0.0
            if not represented
            else float(
                np.mean(
                    [
                        float(value["quality"]) * float(value["availability"])
                        for value in represented
                    ]
                )
            )
        )
        family_keys = [
            key
            for key in grouped
            if ("noun" if key.startswith("noun|") else "action") == family
        ]
        concentration_change = float(
            np.mean(
                [
                    abs(float(token_diagnostics[key]["concentration_gain"]))
                    for key in family_keys
                ]
            )
        )
        quality_gate = 1.0 / (
            1.0 + math.exp(-((mean_quality - threshold) / scale))
        )
        change_gate = 1.0 / (
            1.0
            + math.exp(
                -((concentration_change - change_threshold) / change_scale)
            )
        )
        raw_trust = maximum * quality_gate * change_gate
        family_quality[family] = mean_quality
        family_concentration_change[family] = concentration_change
        family_trust[family] = maximum if raw_trust >= minimum else 0.0
    active_updates = 0
    possible_updates = 0
    null_posteriors: list[float] = []
    entropy_values: list[float] = []
    for key, rows in sorted(grouped.items()):
        key_active_updates = 0
        supported = [
            evidence[str(row["episode_id"])]
            for _, _, row in rows
            if str(row["episode_id"]) in evidence
        ]
        mean_quality = (
            0.0
            if not supported
            else float(
                np.mean(
                    [
                        float(value["quality"]) * float(value["availability"])
                        for value in supported
                    ]
                )
            )
        )
        family = "noun" if key.startswith("noun|") else "action"
        trust = family_trust[family]
        accumulator = np.zeros_like(initial[key])
        total = 0.0
        for iteration in range(iterations):
            for occurrence_index, (slot, _, row) in enumerate(rows):
                episode_id = str(row["episode_id"])
                if episode_id not in evidence:
                    continue
                possible_updates += 1
                keep_rng = rng_for(
                    model_seed,
                    f"fused-drop-{key}-{episode_id}-{iteration}-{occurrence_index}",
                )
                if trust <= 0.0 or keep_rng.random() < dropout:
                    continue
                row_evidence = evidence[episode_id]
                logits = channel_weight * np.asarray(
                    row_evidence["event_logits"], dtype=np.float64
                )
                if allow_null:
                    logits = np.concatenate(
                        [
                            logits,
                            np.asarray(
                                [channel_weight * float(row_evidence["null_logit"])]
                            ),
                        ]
                    )
                posterior = _softmax(logits, temperature)
                candidate_posterior = posterior[:-1] if allow_null else posterior
                episode_weight = (
                    float(row_evidence["quality"])
                    * float(row_evidence["availability"])
                )
                observations = np.stack(
                    [_observation(event, slot) for event in row["events"]]
                )
                accumulator += (
                    episode_weight * candidate_posterior[:, None] * observations
                ).sum(axis=0)
                total += episode_weight * float(candidate_posterior.sum())
                active_updates += 1
                key_active_updates += 1
                if allow_null:
                    null_posteriors.append(float(posterior[-1]))
                entropy_values.append(
                    float(
                        -np.sum(
                            posterior
                            * np.log(np.clip(posterior, 1e-12, None))
                        )
                    )
                )
        sensor_prototype = (
            baseline[key]
            if total <= 1e-9
            else _normalize(accumulator / total)
        )
        output[key] = _normalize(
            (1.0 - trust) * baseline[key] + trust * sensor_prototype
        )
        reliability[key] = {
            "trust": float(trust),
            "mean_quality": mean_quality,
            "family_mean_quality": family_quality[family],
            "family_mean_absolute_concentration_change": family_concentration_change[
                family
            ],
            "quality_threshold": threshold,
            "supported_occurrences": float(len(supported)),
            "sensor_updates": float(key_active_updates),
        }
    return output, {
        "iterations": iterations,
        "single_occurrence": single_occurrence,
        "null_option_available": allow_null,
        "derived_training_evidence_available": bool(evidence),
        "derived_training_evidence_used": active_updates > 0,
        "active_evidence_updates": active_updates,
        "possible_evidence_updates": possible_updates,
        "mean_null_posterior": (
            None if not null_posteriors else float(np.mean(null_posteriors))
        ),
        "mean_assignment_entropy": (
            None if not entropy_values else float(np.mean(entropy_values))
        ),
        "reliability_by_token": reliability,
        "fusion_rule": (
            "channel-level sigmoid trust from mean detector overlap quality; unreliable channels "
            "fall back exactly to the sensor-free latent cross-occurrence prototype"
        ),
    }


def fit_learner(
    *,
    episodes: Sequence[Mapping[str, Any]],
    learner: str,
    condition: str,
    corpus_seed: int,
    model_seed: int,
    config: Mapping[str, Any],
    derived_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    oracle_rows: Sequence[Mapping[str, Any]] | None = None,
    sensitivity: bool = False,
) -> tuple[LexicalModel, dict[str, Any]]:
    guard_seed_operation(
        config, operation="train", seeds=[int(corpus_seed), int(model_seed)]
    )
    if learner not in LEARNERS:
        raise KeyError(learner)
    if int(model_seed) not in set(map(int, config["seeds"]["development"]["model"])):
        raise PermissionError("only configured v2 development model seeds may train")
    if int(corpus_seed) not in set(map(int, config["seeds"]["development"]["corpus"])):
        raise PermissionError("only configured v2 development corpus seeds may train")
    single = learner == "sensor_latent_single_occurrence"
    grouped = _occurrences(episodes, single_occurrence=single)
    initial = _initial_prototypes(grouped, int(model_seed))
    evidence = dict(derived_evidence or {})
    extra: dict[str, Any] = {
        "iterations": 1,
        "single_occurrence": single,
        "null_option_available": False,
        "derived_training_evidence_available": bool(evidence),
        "derived_training_evidence_used": False,
        "active_evidence_updates": 0,
        "possible_evidence_updates": 0,
        "mean_null_posterior": None,
        "mean_assignment_entropy": None,
        "reliability_by_token": {},
    }
    iterations = int(
        config["learners"]["sensitivity_iterations" if sensitivity else "iterations"]
    )
    if learner == "exact_window_symbolic":
        prototypes = _exact_window(grouped, initial)
    elif learner == "oracle_event_alignment_upper":
        if oracle_rows is None:
            raise ValueError("oracle event alignment requires physically separate oracle rows")
        prototypes = _oracle_alignment(grouped, oracle_rows, initial)
    else:
        if learner == "v1_pointer_style_upper":
            if oracle_rows is None:
                raise ValueError("pointer upper control requires physically separate oracle rows")
            evidence = _pointer_evidence(
                episodes,
                oracle_rows,
                config,
                corpus_seed=int(corpus_seed),
            )
        use_evidence = learner in {
            "sensor_latent_single_occurrence",
            "sensor_latent_cross_occurrence",
            "sensor_latent_cross_no_null",
            "v1_pointer_style_upper",
        }
        allow_null = learner != "sensor_latent_cross_no_null"
        if learner in {"cross_occurrence_no_sensor", "structural_absence"}:
            use_evidence = False
        if use_evidence:
            prototypes, extra = _sensor_fused_fit(
                grouped,
                initial,
                evidence,
                config,
                model_seed=int(model_seed),
                allow_null=allow_null,
                single_occurrence=single,
                iterations=iterations,
            )
        else:
            prototypes, extra = _latent_fit(
                grouped,
                initial,
                evidence,
                config,
                model_seed=int(model_seed),
                allow_null=allow_null,
                use_evidence=False,
                single_occurrence=single,
                iterations=iterations,
            )
    model = LexicalModel(
        token_prototypes={
            key: tuple(map(float, value)) for key, value in sorted(prototypes.items())
        },
        learner=learner,
        model_seed=int(model_seed),
    )
    trace = {
        "schema_version": "synthetic-sensor-event-training-trace-v2",
        "learner": learner,
        "condition": condition,
        "corpus_seed": int(corpus_seed),
        "model_seed": int(model_seed),
        "initialization_digest": canonical_digest(
            {key: list(map(float, value)) for key, value in sorted(initial.items())}
        ),
        "episode_order_digest": canonical_digest(
            [str(row["episode_id"]) for row in episodes]
        ),
        "non_channel_inventory_digest": canonical_digest(
            [
                {key: value for key, value in row.items() if key != "raw_stream"}
                for row in episodes
            ]
        ),
        "training_examples": len(episodes),
        "update_passes": iterations if learner not in {
            "exact_window_symbolic", "oracle_event_alignment_upper"
        } else 1,
        "serialized_model_digest": canonical_digest(model.serializable()),
        "sensitivity_refit": sensitivity,
        **extra,
    }
    return model, trace


def _prototype(
    model: LexicalModel, slot: str, token: str, dimension: int
) -> np.ndarray:
    value = model.token_prototypes.get(_key(slot, token))
    if value is None:
        return np.full(dimension, 1.0 / dimension, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _macro_accuracy(correct: Sequence[bool], groups: Sequence[str]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(correct, groups):
        grouped[str(group)].append(float(value))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def evaluate_final(
    model: LexicalModel,
    evaluation_items: Sequence[EvaluationItem],
    *,
    corpus_seed: int,
    policy: EvaluationPolicy,
) -> dict[str, Any]:
    if int(corpus_seed) in set(policy.blocked_values) or model.model_seed in set(
        policy.blocked_values
    ):
        raise PermissionError("evaluation policy blocked a confirmation reserve seed")
    if int(corpus_seed) not in set(policy.allowed_corpora) or model.model_seed not in set(
        policy.allowed_models
    ):
        raise PermissionError("evaluation policy permits only configured development seeds")
    if any(not isinstance(item, EvaluationItem) for item in evaluation_items):
        raise TypeError("final evaluation requires the fail-closed EvaluationItem schema")
    item_fields = {field.name for field in fields(EvaluationItem)}
    model_fields = {field.name for field in fields(LexicalModel)}
    if any(
        any(fragment in name.lower() for fragment in FINAL_FORBIDDEN_FRAGMENTS)
        for name in item_fields | model_fields
    ):
        raise RuntimeError("a training-only channel field crossed the final evaluation boundary")
    action_correct: list[bool] = []
    action_groups: list[str] = []
    noun_correct: list[bool] = []
    noun_groups: list[str] = []
    zero_correct: list[bool] = []
    zero_groups: list[str] = []
    primitive_correct: list[bool] = []
    manner_correct: list[bool] = []
    roles: list[str] = []
    compositions: list[str] = []
    for item in evaluation_items:
        action = np.asarray(item.action_observation, dtype=np.float64)
        action_scores = []
        for primitive_word, manner_word in item.action_candidate_phrases:
            action_scores.append(
                float(action[:3] @ _prototype(model, "primitive", primitive_word, 3))
                + float(action[3:] @ _prototype(model, "manner", manner_word, 2))
            )
        action_prediction = int(np.argmax(action_scores))
        primitive_candidates = tuple(
            dict.fromkeys(phrase[0] for phrase in item.action_candidate_phrases)
        )
        manner_candidates = tuple(
            dict.fromkeys(phrase[1] for phrase in item.action_candidate_phrases)
        )
        true_phrase = item.action_candidate_phrases[item.action_answer_index]
        primitive_prediction = int(
            np.argmax(
                [
                    float(action[:3] @ _prototype(model, "primitive", word, 3))
                    for word in primitive_candidates
                ]
            )
        )
        manner_prediction = int(
            np.argmax(
                [
                    float(action[3:] @ _prototype(model, "manner", word, 2))
                    for word in manner_candidates
                ]
            )
        )
        obj = np.asarray(item.object_observation, dtype=np.float64)
        noun_scores = [
            float(obj @ _prototype(model, "noun", word, 6))
            for word in item.noun_candidate_words
        ]
        zero_scores = [
            float(action[:3] @ _prototype(model, "zero", word, 3))
            for word in item.zero_candidate_words
        ]
        action_correct.append(action_prediction == item.action_answer_index)
        action_groups.append(str(item.action_answer_index))
        noun_correct.append(int(np.argmax(noun_scores)) == item.noun_answer_index)
        noun_groups.append(item.noun_candidate_words[item.noun_answer_index])
        zero_correct.append(int(np.argmax(zero_scores)) == item.zero_answer_index)
        zero_groups.append(item.zero_candidate_words[item.zero_answer_index])
        primitive_correct.append(
            primitive_candidates[primitive_prediction] == true_phrase[0]
        )
        manner_correct.append(manner_candidates[manner_prediction] == true_phrase[1])
        roles.append(item.exposure_role)
        compositions.append(item.composition_id)
    structured = [
        index for index, role in enumerate(roles) if role == "structured_heldout_concept"
    ]
    seen = [index for index, role in enumerate(roles) if role == "seen_combination"]

    def mean_subset(values: Sequence[bool], indices: Sequence[int]) -> float:
        return float(np.mean([values[index] for index in indices]))

    return {
        "heldout_object_action_composition_action_6way_macro_accuracy": _macro_accuracy(
            action_correct, action_groups
        ),
        "new_action_instance_accuracy": float(np.mean(action_correct)),
        "seen_combination_action_6way_accuracy": mean_subset(action_correct, seen),
        "structured_heldout_concept_action_6way_accuracy": mean_subset(
            action_correct, structured
        ),
        "seen_lexical_component_accuracy_on_independent_tokens": float(
            0.5 * (np.mean(primitive_correct) + np.mean(manner_correct))
        ),
        "primitive_component_3way_accuracy": float(np.mean(primitive_correct)),
        "manner_component_2way_accuracy": float(np.mean(manner_correct)),
        "matched_noun_6way_macro_accuracy": _macro_accuracy(noun_correct, noun_groups),
        "zero_exposure_word_6way_accuracy": _macro_accuracy(zero_correct, zero_groups),
        "n_evaluation_items": len(evaluation_items),
        "n_structured_items": len(structured),
        "n_seen_combination_items": len(seen),
        "n_heldout_object_action_compositions": len(set(compositions)),
        "new_action_instances": True,
        "component_word_exposure_for_structured_concept": True,
        "arbitrary_zero_word_exposure": 0,
        "evaluation_inputs": [
            "learned_token_prototypes",
            "action_observation",
            "object_observation",
        ],
        "training_only_channels_structurally_unavailable": True,
        "evaluation_item_fields": sorted(item_fields),
        "model_fields": sorted(model_fields),
    }


def evaluation_firewall_contract() -> dict[str, Any]:
    signature = inspect.signature(evaluate_final)
    item_fields = [field.name for field in fields(EvaluationItem)]
    model_fields = [field.name for field in fields(LexicalModel)]
    source = inspect.getsource(evaluate_final)
    checks = {
        "signature_has_no_training_channel_argument": not any(
            any(fragment in name.lower() for fragment in FINAL_FORBIDDEN_FRAGMENTS)
            for name in signature.parameters
        ),
        "evaluation_item_has_no_training_channel_field": not any(
            any(fragment in name.lower() for fragment in FINAL_FORBIDDEN_FRAGMENTS)
            for name in item_fields
        ),
        "serialized_model_has_no_training_channel_field": not any(
            any(fragment in name.lower() for fragment in FINAL_FORBIDDEN_FRAGMENTS)
            for name in model_fields
        ),
        "evaluator_has_no_encoding_or_feature_extraction_call": "raw_features" not in source
        and "candidate_evidence" not in source
        and "encode_" not in source,
        "evaluator_rejects_untyped_items": "isinstance(item, EvaluationItem)" in source,
    }
    return {
        "schema_version": "synthetic-sensor-event-evaluation-firewall-v2",
        "valid": all(checks.values()),
        "checks": checks,
        "signature": str(signature),
        "evaluation_item_fields": item_fields,
        "lexical_model_fields": model_fields,
    }
