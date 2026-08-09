from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import (
    IdentifierFirewall,
    IdentifierReference,
    canonical_digest,
    reject_forbidden_fields,
)


def _rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"corrective-learner-v1|{label}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    return array / array.sum()


def _softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64) / float(temperature)
    array -= float(array.max())
    exponential = np.exp(np.clip(array, -60.0, 60.0))
    return exponential / exponential.sum()


def _token_key(item: Mapping[str, Any]) -> str:
    return f"{item['slot']}|{item['token']}"


def _slot_dimension(slot: str, config: Mapping[str, Any]) -> int:
    return int(
        config["design"][
            "primitive_concepts" if slot == "primitive" else "manner_concepts"
        ]
    )


def _observation(event: Mapping[str, Any], slot: str) -> np.ndarray:
    return np.asarray(event[f"{slot}_observation"], dtype=np.float64)


def _logit(probability: float) -> float:
    value = float(np.clip(probability, 1e-8, 1.0 - 1e-8))
    return math.log(value / (1.0 - value))


@dataclass(frozen=True, slots=True)
class CorrectiveLexiconModel:
    semantics: dict[str, tuple[float, ...]]
    null_rates: dict[str, float]
    model_seed: int
    epochs: int
    learner: str

    def serializable(self) -> dict[str, Any]:
        return {
            "schema_version": "nursery-corrective-lexicon-model-v1",
            "semantics": {
                key: list(value) for key, value in sorted(self.semantics.items())
            },
            "null_rates": {
                key: float(value) for key, value in sorted(self.null_rates.items())
            },
            "model_seed": int(self.model_seed),
            "epochs": int(self.epochs),
            "learner": self.learner,
            "training_side_state_serialized": False,
            "detector_serialized": False,
            "oracle_serialized": False,
            "evaluation_key_serialized": False,
        }


def _initial_state(
    episodes: Sequence[Mapping[str, Any]],
    model_seed: int,
    config: Mapping[str, Any],
    initial_override: Mapping[str, Sequence[float]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    token_slots = {
        _token_key(item): str(item["slot"])
        for row in episodes
        for item in row["utterance"]["items"]
    }
    concentration = float(config["learner"]["initialization_concentration"])
    semantics = {}
    for key, slot in sorted(token_slots.items()):
        if initial_override is not None and key in initial_override:
            semantics[key] = _normalize(initial_override[key])
        else:
            semantics[key] = _rng(model_seed, f"initial-{key}").dirichlet(
                np.ones(_slot_dimension(slot, config)) * concentration
            )
    return semantics, {
        key: float(config["learner"]["null_prior"]) for key in semantics
    }


def fit_corrective_mil(
    episodes: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    model_seed: int,
    condition: str,
    mutation: str = "active",
    initial_override: Mapping[str, Sequence[float]] | None = None,
) -> tuple[CorrectiveLexiconModel, dict[str, Any]]:
    allowed_mutations = {
        "active",
        "semantic_side_zero",
        "evidence_disconnected",
        "within_bag_permuted",
        "old_agreement_gate",
        "semantic_update_disconnected",
        "null_side_zero",
        "null_update_disconnected",
    }
    if mutation not in allowed_mutations:
        raise ValueError(f"unknown mechanism mutation: {mutation}")
    training_forbidden = {
        "answer_index",
        "concept_id",
        "intended_concept",
        "oracle",
        "target_event_index",
    }
    for row in episodes:
        reject_forbidden_fields(row, training_forbidden, path="training")
    semantics, null_rates = _initial_state(
        episodes, model_seed, config, initial_override
    )
    learner = config["learner"]
    epochs = int(learner["epochs"])
    temperature = float(learner["posterior_temperature"])
    language_weight = float(learner["language_weight"])
    temporal_weight = float(learner["temporal_weight"])
    temporal_scale = float(learner["temporal_scale"])
    semantic_side_weight = float(learner["semantic_side_weight"])
    null_side_weight = float(learner["null_side_weight"])
    smoothing = float(learner["smoothing"])
    initial_digest = canonical_digest(
        {key: list(map(float, value)) for key, value in sorted(semantics.items())}
    )
    epoch_digests: list[str] = []
    epoch_orders: list[str] = []
    semantic_disagreement_count = 0
    side_changed_event_top_count = 0
    null_posterior_values: list[float] = []
    event_mass_values: list[float] = []
    semantic_update_norms: list[float] = []
    null_update_norms: list[float] = []
    ordered = sorted(episodes, key=lambda row: str(row["episode_id"]))
    for epoch in range(epochs):
        order_rng = _rng(model_seed, f"epoch-order-{epoch}")
        order = list(map(int, order_rng.permutation(len(ordered))))
        epoch_orders.append(canonical_digest(order))
        semantic_lr = float(learner["semantic_learning_rate"]) / (
            1.0 + float(learner["semantic_learning_rate_decay"]) * epoch
        )
        null_lr = float(learner["null_learning_rate"]) / (
            1.0 + float(learner["semantic_learning_rate_decay"]) * epoch
        )
        for row_index in order:
            row = ordered[row_index]
            items = list(row["utterance"]["items"])
            events = list(row["events"])
            episode_id = str(row["episode_id"])
            detector = dict(evidence.get(episode_id, {}))
            event_side = np.asarray(
                detector.get("event_logits", [0.0] * len(events)), dtype=float
            )
            null_side = float(detector.get("null_logit", 0.0))
            if len(event_side) != len(events):
                raise ValueError("side event count differs from candidate-event count")
            if mutation == "evidence_disconnected":
                event_side = np.zeros_like(event_side)
                null_side = 0.0
            elif mutation == "within_bag_permuted":
                event_side = event_side[::-1].copy()
            language_scores = []
            for event in events:
                language_scores.append(
                    float(
                        np.mean(
                            [
                                math.log(
                                    float(
                                        np.clip(
                                            semantics[_token_key(item)]
                                            @ _observation(event, str(item["slot"])),
                                            1e-12,
                                            1.0,
                                        )
                                    )
                                )
                                for item in items
                            ]
                        )
                    )
                )
            speech_time = float(row["utterance"]["speech_time"])
            temporal_scores = np.asarray(
                [
                    -abs(
                        speech_time
                        - (float(event["start"]) + float(event["end"])) / 2.0
                    )
                    / temporal_scale
                    for event in events
                ],
                dtype=float,
            )
            base_event_logits = (
                language_weight * np.asarray(language_scores)
                + temporal_weight * temporal_scores
            )
            language_top = int(np.argmax(base_event_logits))
            side_top = int(np.argmax(np.concatenate([event_side, [null_side]])))
            if side_top < len(events) and side_top != language_top:
                semantic_disagreement_count += 1
            event_weight = semantic_side_weight
            null_weight = null_side_weight
            if mutation in {"semantic_side_zero", "semantic_update_disconnected"}:
                event_weight = 0.0
            if mutation == "null_side_zero":
                null_weight = 0.0
            if mutation == "old_agreement_gate" and side_top != language_top:
                event_weight = 0.0
                null_weight = 0.0
            if condition == "exact_window" or bool(detector.get("exact_window")):
                midpoints = np.asarray(
                    [
                        (float(event["start"]) + float(event["end"])) / 2.0
                        for event in events
                    ]
                )
                chosen = int(np.argmin(np.abs(midpoints - speech_time)))
                event_posterior = np.zeros(len(events), dtype=float)
                event_posterior[chosen] = 1.0
                null_posterior = 0.0
                event_mass = 1.0
            else:
                full_logits = np.concatenate(
                    [
                        base_event_logits + event_weight * event_side,
                        [
                            float(np.mean([_logit(null_rates[_token_key(item)]) for item in items]))
                            + null_weight * null_side
                        ],
                    ]
                )
                full_posterior = _softmax(full_logits, temperature)
                null_posterior = float(full_posterior[-1])
                event_mass = float(full_posterior[:-1].sum())
                event_posterior = (
                    full_posterior[:-1] / event_mass
                    if event_mass > 1e-12
                    else np.full(len(events), 1.0 / len(events))
                )
                if int(np.argmax(event_posterior)) != language_top:
                    side_changed_event_top_count += 1
            null_posterior_values.append(null_posterior)
            event_mass_values.append(event_mass)
            for item in items:
                key = _token_key(item)
                slot = str(item["slot"])
                target = _normalize(
                    smoothing
                    + sum(
                        float(weight) * _observation(event, slot)
                        for weight, event in zip(event_posterior, events)
                    )
                )
                old = semantics[key].copy()
                if mutation != "semantic_update_disconnected":
                    effective_semantic_lr = semantic_lr * event_mass
                    semantics[key] = _normalize(
                        (1.0 - effective_semantic_lr) * semantics[key]
                        + effective_semantic_lr * target
                    )
                semantic_update_norms.append(
                    float(np.linalg.norm(semantics[key] - old, ord=1))
                )
                old_null = float(null_rates[key])
                if mutation != "null_update_disconnected":
                    null_rates[key] = float(
                        np.clip(
                            (1.0 - null_lr) * old_null
                            + null_lr * null_posterior,
                            1e-5,
                            1.0 - 1e-5,
                        )
                    )
                null_update_norms.append(abs(null_rates[key] - old_null))
        epoch_digests.append(
            canonical_digest(
                {
                    "semantics": {
                        key: list(map(float, value))
                        for key, value in sorted(semantics.items())
                    },
                    "null_rates": dict(sorted(null_rates.items())),
                }
            )
        )
    model = CorrectiveLexiconModel(
        semantics={
            key: tuple(map(float, value)) for key, value in sorted(semantics.items())
        },
        null_rates={key: float(value) for key, value in sorted(null_rates.items())},
        model_seed=int(model_seed),
        epochs=epochs,
        learner=str(learner["name"]),
    )
    trace = {
        "schema_version": "nursery-corrective-training-trace-v1",
        "condition": condition,
        "mutation": mutation,
        "model_seed": int(model_seed),
        "epochs": epochs,
        "initial_parameter_digest": initial_digest,
        "epoch_parameter_digests": epoch_digests,
        "distinct_epoch_parameter_digests": len(set(epoch_digests)),
        "epoch_order_digests": epoch_orders,
        "distinct_epoch_order_digests": len(set(epoch_orders)),
        "semantic_side_weight_effective": (
            0.0
            if mutation
            in {"semantic_side_zero", "semantic_update_disconnected", "evidence_disconnected"}
            else semantic_side_weight
        ),
        "null_side_weight_effective": (
            0.0
            if mutation in {"null_side_zero", "evidence_disconnected"}
            else null_side_weight
        ),
        "null_update_frozen_at_initial_prior": mutation
        == "null_update_disconnected",
        "disagreement_gate": mutation == "old_agreement_gate",
        "argmax_prefilter": False,
        "all_candidate_events_enter_posterior": True,
        "semantic_disagreement_count": semantic_disagreement_count,
        "side_changed_event_top_count": side_changed_event_top_count,
        "mean_null_posterior": float(np.mean(null_posterior_values)),
        "mean_event_posterior_mass": float(np.mean(event_mass_values)),
        "minimum_event_posterior_mass": float(np.min(event_mass_values)),
        "semantic_update_scaled_by_event_mass": True,
        "mean_semantic_update_l1": float(np.mean(semantic_update_norms)),
        "mean_null_update_absolute": float(np.mean(null_update_norms)),
        "model_digest": canonical_digest(model.serializable()),
        "oracle_fields_consumed": False,
        "evaluation_keys_consumed": False,
    }
    return model, trace


def _semantic(model: CorrectiveLexiconModel, item: Mapping[str, Any], dimension: int) -> np.ndarray:
    key = _token_key(item)
    if key not in model.semantics:
        return np.full(dimension, 1.0 / dimension)
    return np.asarray(model.semantics[key], dtype=np.float64)


def _event_score(
    model: CorrectiveLexiconModel,
    items: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[float, float]:
    likelihoods = []
    for item in items:
        slot = str(item["slot"])
        semantic = _semantic(model, item, _slot_dimension(slot, config))
        if "observation" in event:
            observation = np.asarray(event["observation"], dtype=float)
        else:
            observation = _observation(event, slot)
        likelihoods.append(float(np.clip(semantic @ observation, 1e-12, 1.0)))
    log_score = float(np.mean(np.log(likelihoods)))
    compatibility = float(np.exp(log_score))
    return log_score, compatibility


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"evaluation allowlist mismatch at {path}: "
            f"missing={sorted(expected - set(value))} unexpected={sorted(set(value) - expected)}"
        )


def _validate_side_free_evaluation_prompt(prompt: Mapping[str, Any]) -> None:
    kind = str(prompt.get("kind"))
    common = {
        "schema_version",
        "prompt_id",
        "kind",
        "tokens",
        "candidates",
        "generator_provenance",
    }
    if kind in {"lexical", "zero_exposure"}:
        _require_exact_fields(prompt, common | {"slot"}, f"{kind}.prompt")
        candidate_fields = {"candidate_id", "observation"}
    elif kind == "composition":
        _require_exact_fields(prompt, common, "composition.prompt")
        candidate_fields = {
            "candidate_id",
            "primitive_observation",
            "manner_observation",
        }
    elif kind == "presence":
        _require_exact_fields(prompt, common | {"null_option"}, "presence.prompt")
        if prompt["null_option"] is not True:
            raise ValueError("presence prompt must expose exactly one null option")
        candidate_fields = {
            "candidate_id",
            "primitive_observation",
            "manner_observation",
        }
    else:
        raise ValueError(f"unknown evaluation prompt kind: {kind}")
    if prompt.get("schema_version") != "nursery-corrective-eval-visible-v1":
        raise ValueError("evaluation prompt schema version mismatch")
    tokens = prompt.get("tokens")
    candidates = prompt.get("candidates")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("evaluation prompt tokens must be a nonempty list")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("evaluation prompt candidates must be a nonempty list")
    for index, item in enumerate(tokens):
        if not isinstance(item, Mapping):
            raise ValueError(f"evaluation token is not a mapping: {index}")
        _require_exact_fields(item, {"slot", "token"}, f"{kind}.tokens[{index}]")
        if item["slot"] not in {"primitive", "manner"}:
            raise ValueError("unknown evaluation token slot")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"evaluation candidate is not a mapping: {index}")
        _require_exact_fields(candidate, candidate_fields, f"{kind}.candidates[{index}]")


def predict_without_side(
    model: CorrectiveLexiconModel,
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    forbidden = set(config["firewalls"]["forbidden_visible_fields"])
    reject_forbidden_fields(model.serializable(), forbidden, path="model")
    output = []
    for prompt in prompts:
        _validate_side_free_evaluation_prompt(prompt)
        reject_forbidden_fields(prompt, forbidden, path="evaluation")
        items = list(prompt["tokens"])
        scores = []
        compatibilities = []
        for candidate in prompt["candidates"]:
            score, compatibility = _event_score(
                model, items, candidate, config
            )
            scores.append(score)
            compatibilities.append(compatibility)
        if bool(prompt.get("null_option")):
            mean_null = float(
                np.mean(
                    [
                        model.null_rates.get(
                            _token_key(item), float(config["learner"]["null_prior"])
                        )
                        for item in items
                    ]
                )
            )
            mismatch = 1.0 - max(compatibilities, default=0.0)
            scores.append(
                _logit(mean_null)
                + float(config["learner"]["null_mismatch_weight"]) * mismatch
            )
        probabilities = _softmax(scores)
        output.append(
            {
                "prompt_id": str(prompt["prompt_id"]),
                "kind": str(prompt["kind"]),
                "scores": list(map(float, scores)),
                "probabilities": list(map(float, probabilities)),
                "prediction_side_fields_consumed": False,
            }
        )
    return output


def ablate_null_head(model: CorrectiveLexiconModel) -> CorrectiveLexiconModel:
    return replace(
        model,
        null_rates={key: 1e-6 for key in model.null_rates},
        learner=f"{model.learner}+null_head_ablation",
    )


def _micro_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config)


def disagreement_mechanism_qualification(
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
) -> dict[str, Any]:
    mutations = (
        "active",
        "semantic_side_zero",
        "evidence_disconnected",
        "within_bag_permuted",
        "old_agreement_gate",
        "semantic_update_disconnected",
    )
    cases = []
    flip_counts = {mutation: 0 for mutation in mutations}
    case_count = int(config["qualification_gates"]["disagreement_case_count"])
    mechanism_registry = config["resolved_registries"]["construction_mechanism"]
    model_seeds = list(map(int, mechanism_registry["model"]))
    if len(model_seeds) != case_count:
        raise RuntimeError("mechanism registry must provide one model seed per case")
    corpus_seed = int(mechanism_registry["corpus"][0])
    null_rows = []
    for case_index in range(case_count):
        target = case_index % int(config["design"]["primitive_concepts"])
        foil = (target + 1 + (case_index // 8) % 2) % int(
            config["design"]["primitive_concepts"]
        )
        if foil == target:
            foil = (foil + 1) % int(config["design"]["primitive_concepts"])
        token = f"micro-{case_index}"
        episodes = []
        evidence = {}
        for repetition in range(6):
            episode_id = f"micro-case-{case_index}-{repetition}"
            target_observation = np.full(
                int(config["design"]["primitive_concepts"]), 0.01
            )
            target_observation[target] = 0.97
            foil_observation = np.full(
                int(config["design"]["primitive_concepts"]), 0.01
            )
            foil_observation[foil] = 0.97
            episodes.append(
                {
                    "episode_id": episode_id,
                    "utterance": {
                        "items": [{"slot": "primitive", "token": token}],
                        "speech_time": 10.0,
                    },
                    "events": [
                        {
                            "start": 4,
                            "end": 10,
                            "primitive_observation": _normalize(target_observation).tolist(),
                            "manner_observation": [1 / 3, 1 / 3, 1 / 3],
                        },
                        {
                            "start": 11,
                            "end": 17,
                            "primitive_observation": _normalize(foil_observation).tolist(),
                            "manner_observation": [1 / 3, 1 / 3, 1 / 3],
                        },
                    ],
                }
            )
            evidence[episode_id] = {
                "event_logits": [4.0, -4.0],
                "null_logit": -5.0,
            }
        initial = np.full(int(config["design"]["primitive_concepts"]), 0.02)
        initial[foil] = 0.92
        initial[target] = 0.02
        initial = _normalize(initial)
        row = {
            "case_id": f"case-{case_index:02d}",
            "initial_top": int(np.argmax(initial)),
            "target": target,
            "foil": foil,
            "mutation_results": {},
        }
        for mutation in mutations:
            references = [
                IdentifierReference("corpus", corpus_seed),
                IdentifierReference("model", model_seeds[case_index]),
            ]
            firewall.authorize(
                "control",
                references,
                unit_id=f"mechanism-case-{case_index:02d}",
                local_sequence=len(firewall.operations),
            )
            firewall.authorize(
                "fit",
                references,
                unit_id=f"mechanism-case-{case_index:02d}",
                local_sequence=len(firewall.operations),
            )
            model, trace = fit_corrective_mil(
                episodes,
                evidence,
                config,
                model_seed=model_seeds[case_index],
                condition="micro",
                mutation=mutation,
                initial_override={f"primitive|{token}": initial},
            )
            distribution = np.asarray(model.semantics[f"primitive|{token}"])
            ordered = np.sort(distribution)[::-1]
            unique = bool(ordered[0] - ordered[1] > 1e-6)
            flipped = int(np.argmax(distribution)) == target and unique
            flip_counts[mutation] += int(flipped)
            row["mutation_results"][mutation] = {
                "top": int(np.argmax(distribution)),
                "target_probability": float(distribution[target]),
                "foil_probability": float(distribution[foil]),
                "unique_correct_flip": flipped,
                "semantic_disagreement_count": trace["semantic_disagreement_count"],
            }
        null_evidence = {
            episode_id: {"event_logits": [-8.0, -8.0], "null_logit": 8.0}
            for episode_id in evidence
        }
        references = [
            IdentifierReference("corpus", corpus_seed),
            IdentifierReference("model", model_seeds[case_index]),
        ]
        firewall.authorize(
            "control",
            references,
            unit_id=f"mechanism-null-{case_index:02d}",
            local_sequence=len(firewall.operations),
        )
        firewall.authorize(
            "fit",
            references,
            unit_id=f"mechanism-null-{case_index:02d}",
            local_sequence=len(firewall.operations),
        )
        _null_model, null_trace = fit_corrective_mil(
            episodes,
            null_evidence,
            config,
            model_seed=model_seeds[case_index],
            condition="micro-null",
            mutation="active",
            initial_override={f"primitive|{token}": initial},
        )
        null_rows.append(
            {
                "case_id": f"case-{case_index:02d}",
                "mean_null_posterior": null_trace["mean_null_posterior"],
                "mean_event_posterior_mass": null_trace[
                    "mean_event_posterior_mass"
                ],
                "mean_semantic_update_l1": null_trace["mean_semantic_update_l1"],
            }
        )
        cases.append(row)
    fractions = {
        mutation: count / case_count for mutation, count in flip_counts.items()
    }
    active_minimum = float(
        config["qualification_gates"]["minimum_active_disagreement_flip_fraction"]
    )
    mutation_maximum = float(
        config["qualification_gates"]["maximum_mutated_disagreement_flip_fraction"]
    )
    minimum_null = float(
        config["qualification_gates"]["minimum_hand_case_null_posterior"]
    )
    maximum_null_update = float(
        config["qualification_gates"]["maximum_hand_case_null_semantic_update_l1"]
    )
    null_suppression_passed = all(
        float(row["mean_null_posterior"]) >= minimum_null
        and float(row["mean_semantic_update_l1"]) <= maximum_null_update
        for row in null_rows
    )
    firewall.authorize(
        "inference",
        [
            IdentifierReference("corpus", corpus_seed),
            IdentifierReference(
                "inference", int(mechanism_registry["inference"][0])
            ),
        ],
        unit_id="mechanism-qualification",
        local_sequence=len(firewall.operations),
    )
    passed = fractions["active"] >= active_minimum and all(
        fractions[mutation] <= mutation_maximum
        for mutation in mutations
        if mutation != "active"
    ) and null_suppression_passed
    return {
        "status": "PASS" if passed else "FAIL",
        "case_count": case_count,
        "prespecified_cases": True,
        "active_minimum": active_minimum,
        "mutation_maximum": mutation_maximum,
        "flip_fractions": fractions,
        "null_suppression": {
            "status": "PASS" if null_suppression_passed else "FAIL",
            "minimum_mean_null_posterior": minimum_null,
            "maximum_mean_semantic_update_l1": maximum_null_update,
            "cases": null_rows,
        },
        "cases": cases,
    }
