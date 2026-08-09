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
    digest = hashlib.sha256(f"corrective-learner-v3|{label}|{seed}".encode()).digest()
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


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-clipped))


def _logsumexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(array.max())
    return maximum + math.log(float(np.exp(array - maximum).sum()))


@dataclass(frozen=True, slots=True)
class CorrectiveLexiconModel:
    semantics: dict[str, tuple[float, ...]]
    presence_intercept: float
    presence_mismatch_weight: float
    model_seed: int
    epochs: int
    learner: str

    def serializable(self) -> dict[str, Any]:
        return {
            "schema_version": "nursery-corrective-lexicon-model-v3",
            "semantics": {
                key: list(value) for key, value in sorted(self.semantics.items())
            },
            "presence_intercept": float(self.presence_intercept),
            "presence_mismatch_weight": float(self.presence_mismatch_weight),
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
) -> dict[str, np.ndarray]:
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
    return semantics


def _training_mismatch(
    semantics: Mapping[str, np.ndarray],
    items: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> float:
    compatibilities = []
    for event in events:
        likelihoods = []
        for item in items:
            key = _token_key(item)
            slot = str(item["slot"])
            likelihoods.append(
                float(
                    np.clip(
                        semantics[key] @ _observation(event, slot),
                        1e-12,
                        1.0,
                    )
                )
            )
        compatibilities.append(
            float(math.exp(float(np.mean(np.log(likelihoods)))))
        )
    return 1.0 - max(compatibilities, default=0.0)


def _fit_presence_calibration(
    rows: Sequence[tuple[float, float]],
    config: Mapping[str, Any],
) -> tuple[float, float]:
    """Fit a monotone side-free null-rejection model to training-only soft labels."""
    if not rows:
        raise ValueError("presence calibration requires training rows")
    design = np.asarray([[1.0, float(row[0])] for row in rows], dtype=np.float64)
    targets = np.asarray([float(row[1]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(targets)):
        raise ValueError("presence calibration rows must be finite")
    if np.any(targets < 0.0) or np.any(targets > 1.0):
        raise ValueError("presence calibration targets must be probabilities")
    learner = config["learner"]
    center = np.asarray(
        [_logit(float(learner["null_prior"])), 0.0], dtype=np.float64
    )
    parameters = center.copy()
    ridge = float(learner["presence_calibration_ridge"])
    iterations = int(learner["presence_calibration_iterations"])
    maximum_weight = float(learner["maximum_presence_mismatch_weight"])
    for _ in range(iterations):
        logits = np.clip(design @ parameters, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - targets) / len(rows)
        gradient += ridge * (parameters - center)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
        hessian = (design.T * weights) @ design / len(rows)
        hessian += np.eye(2, dtype=np.float64) * ridge
        step = np.linalg.solve(hessian, gradient)
        parameters -= step
        parameters[0] = float(np.clip(parameters[0], -12.0, 12.0))
        parameters[1] = float(np.clip(parameters[1], 0.0, maximum_weight))
    return float(parameters[0]), float(parameters[1])


def _exact_window_event_or_null(
    events: Sequence[Mapping[str, Any]],
    speech_time: float,
) -> tuple[np.ndarray, float]:
    """Return the exact-window comparator's conditional event and null states."""
    if not events:
        raise ValueError("exact-window comparator requires event proposals")
    inside = [
        index
        for index, event in enumerate(events)
        if float(event["start"]) <= float(speech_time) <= float(event["end"])
    ]
    if not inside:
        return np.full(len(events), 1.0 / len(events)), 1.0
    chosen = min(
        inside,
        key=lambda index: abs(
            float(speech_time)
            - (float(events[index]["start"]) + float(events[index]["end"]))
            / 2.0
        ),
    )
    posterior = np.zeros(len(events), dtype=float)
    posterior[int(chosen)] = 1.0
    return posterior, 0.0


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
    if prompt.get("schema_version") != "nursery-corrective-eval-visible-v3":
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
            mismatch = 1.0 - max(compatibilities, default=0.0)
            null_log_odds = (
                float(model.presence_intercept)
                + float(model.presence_mismatch_weight) * mismatch
            )
            scores.append(
                _logsumexp(scores) + null_log_odds
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
        presence_intercept=-20.0,
        presence_mismatch_weight=0.0,
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
                "detector_input_present": True,
                "factorized_presence_event_logits": True,
                "matched_background_proposal_count": 2,
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
            episode_id: {
                "event_logits": [-8.0, -8.0],
                "null_logit": 8.0,
                "detector_input_present": True,
                "factorized_presence_event_logits": True,
                "matched_background_proposal_count": 2,
            }
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
    calibration_rows = [
        *((0.05, 0.02) for _ in range(8)),
        *((0.90, 0.98) for _ in range(8)),
    ]
    calibration_intercept, calibration_weight = _fit_presence_calibration(
        calibration_rows, config
    )
    low_mismatch_null_probability = _sigmoid(
        calibration_intercept + calibration_weight * 0.05
    )
    high_mismatch_null_probability = _sigmoid(
        calibration_intercept + calibration_weight * 0.90
    )
    disconnected_intercept, disconnected_weight = _fit_presence_calibration(
        [(mismatch, float(config["learner"]["null_prior"])) for mismatch, _ in calibration_rows],
        config,
    )
    disconnected_separation = abs(
        _sigmoid(disconnected_intercept + disconnected_weight * 0.90)
        - _sigmoid(disconnected_intercept + disconnected_weight * 0.05)
    )
    minimum_presence_separation = float(
        config["qualification_gates"][
            "minimum_hand_case_presence_probability_separation"
        ]
    )
    maximum_disconnected_separation = float(
        config["qualification_gates"][
            "maximum_disconnected_presence_probability_separation"
        ]
    )
    maximum_present_probability = float(
        config["qualification_gates"][
            "maximum_hand_case_present_null_probability"
        ]
    )
    minimum_null_probability = float(
        config["qualification_gates"][
            "minimum_hand_case_null_null_probability"
        ]
    )
    presence_calibration_passed = (
        calibration_weight > 0.0
        and high_mismatch_null_probability - low_mismatch_null_probability
        >= minimum_presence_separation
        and low_mismatch_null_probability <= maximum_present_probability
        and high_mismatch_null_probability >= minimum_null_probability
        and disconnected_separation <= maximum_disconnected_separation
    )
    exact_events = [
        {"start": 4, "end": 10},
        {"start": 20, "end": 26},
    ]
    exact_present, exact_present_null = _exact_window_event_or_null(
        exact_events, 7.0
    )
    exact_null, exact_null_probability = _exact_window_event_or_null(
        exact_events, 15.0
    )
    exact_window_passed = (
        int(np.argmax(exact_present)) == 0
        and exact_present_null == 0.0
        and np.allclose(exact_null, [0.5, 0.5])
        and exact_null_probability == 1.0
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
    passed = (
        fractions["active"] >= active_minimum
        and all(
            fractions[mutation] <= mutation_maximum
            for mutation in mutations
            if mutation != "active"
        )
        and null_suppression_passed
        and presence_calibration_passed
        and exact_window_passed
    )
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
        "presence_calibration": {
            "status": "PASS" if presence_calibration_passed else "FAIL",
            "training_targets": "prespecified_training_only_soft_presence_labels",
            "test_feature": "side_free_semantic_mismatch",
            "intercept": calibration_intercept,
            "mismatch_weight": calibration_weight,
            "low_mismatch_null_probability": low_mismatch_null_probability,
            "high_mismatch_null_probability": high_mismatch_null_probability,
            "probability_separation": (
                high_mismatch_null_probability - low_mismatch_null_probability
            ),
            "minimum_probability_separation": minimum_presence_separation,
            "maximum_present_case_null_probability": maximum_present_probability,
            "minimum_null_case_null_probability": minimum_null_probability,
            "disconnected_probability_separation": disconnected_separation,
            "maximum_disconnected_probability_separation": (
                maximum_disconnected_separation
            ),
        },
        "exact_window_event_or_null": {
            "status": "PASS" if exact_window_passed else "FAIL",
            "inside_window_event_posterior": list(map(float, exact_present)),
            "inside_window_null_posterior": exact_present_null,
            "outside_all_windows_event_posterior": list(map(float, exact_null)),
            "outside_all_windows_null_posterior": exact_null_probability,
        },
        "cases": cases,
    }


def _fit_corrective_mil_v3(
    episodes: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    model_seed: int,
    condition: str,
    mutation: str = "active",
    initial_override: Mapping[str, Sequence[float]] | None = None,
) -> tuple[CorrectiveLexiconModel, dict[str, Any]]:
    """Factorized event alignment and training-only presence calibration."""
    allowed_mutations = {
        "active",
        "semantic_side_zero",
        "evidence_disconnected",
        "within_bag_permuted",
        "old_agreement_gate",
        "semantic_update_disconnected",
        "presence_side_disconnected",
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
    episode_ids = [str(row["episode_id"]) for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("duplicate training episode identifiers")
    if set(map(str, evidence)) != set(episode_ids):
        raise ValueError("training evidence episode set mismatch")
    for row in episodes:
        reject_forbidden_fields(row, training_forbidden, path="training")
        detector = evidence[str(row["episode_id"])]
        required_evidence_fields = {
            "event_logits",
            "null_logit",
            "detector_input_present",
            "factorized_presence_event_logits",
            "matched_background_proposal_count",
        }
        optional_evidence_fields = {"positive_control", "exact_window", "corruption"}
        if not isinstance(detector, Mapping) or not required_evidence_fields <= set(
            detector
        ) or set(detector) - required_evidence_fields - optional_evidence_fields:
            raise ValueError("training evidence schema mismatch")
        event_logits = detector["event_logits"]
        if (
            not isinstance(event_logits, list)
            or len(event_logits) != len(row["events"])
            or int(detector["matched_background_proposal_count"])
            != len(row["events"])
            or detector["factorized_presence_event_logits"] is not True
            or not isinstance(detector["detector_input_present"], bool)
            or not all(math.isfinite(float(value)) for value in event_logits)
            or not math.isfinite(float(detector["null_logit"]))
        ):
            raise ValueError("malformed factorized training evidence")
        if (
            detector["detector_input_present"] is False
            and detector.get("positive_control") is not True
            and detector.get("exact_window") is not True
            and (
                any(abs(float(value)) > 0.0 for value in event_logits)
                or abs(float(detector["null_logit"])) > 0.0
            )
        ):
            raise ValueError("unavailable detector cannot carry active logits")
    semantics = _initial_state(episodes, model_seed, config, initial_override)
    learner = config["learner"]
    epochs = int(learner["epochs"])
    temperature = float(learner["posterior_temperature"])
    language_weight = float(learner["language_weight"])
    temporal_weight = float(learner["temporal_weight"])
    temporal_scale = float(learner["temporal_scale"])
    semantic_side_weight = float(learner["semantic_side_weight"])
    null_side_weight = float(learner["null_side_weight"])
    smoothing = float(learner["smoothing"])
    null_prior = float(learner["null_prior"])
    presence_intercept = _logit(null_prior)
    presence_mismatch_weight = 0.0
    initial_digest = canonical_digest(
        {
            "semantics": {
                key: list(map(float, value))
                for key, value in sorted(semantics.items())
            },
            "presence_intercept": presence_intercept,
            "presence_mismatch_weight": presence_mismatch_weight,
        }
    )
    initial_semantics_digest = canonical_digest(
        {
            key: list(map(float, value))
            for key, value in sorted(semantics.items())
        }
    )
    epoch_digests: list[str] = []
    epoch_orders: list[str] = []
    semantic_disagreement_count = 0
    side_changed_event_top_count = 0
    null_posterior_values: list[float] = []
    event_mass_values: list[float] = []
    semantic_update_norms: list[float] = []
    presence_update_norms: list[float] = []
    exact_window_null_selection_count = 0
    ordered = sorted(episodes, key=lambda row: str(row["episode_id"]))

    def components(
        row: Mapping[str, Any],
    ) -> tuple[np.ndarray, float, int, int]:
        items = list(row["utterance"]["items"])
        events = list(row["events"])
        detector = dict(evidence.get(str(row["episode_id"]), {}))
        if detector and detector.get("factorized_presence_event_logits") is not True:
            raise ValueError(
                "v4 evidence must declare factorized presence and event logits"
            )
        event_side = np.asarray(
            detector.get("event_logits", [0.0] * len(events)), dtype=float
        )
        null_logit = float(detector.get("null_logit", 0.0))
        if len(event_side) != len(events):
            raise ValueError("side event count differs from candidate-event count")
        detector_available = bool(detector.get("detector_input_present")) or bool(
            detector.get("positive_control")
        )
        if mutation == "evidence_disconnected":
            event_side = np.zeros_like(event_side)
            null_logit = 0.0
            detector_available = False
        elif mutation == "within_bag_permuted":
            event_side = np.roll(event_side, 1)
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
        side_top = int(np.argmax(event_side))
        if condition == "exact_window" or bool(detector.get("exact_window")):
            event_posterior, null_posterior = _exact_window_event_or_null(
                events, speech_time
            )
            return event_posterior, null_posterior, language_top, side_top
        event_weight = semantic_side_weight
        if mutation == "semantic_side_zero":
            event_weight = 0.0
        if mutation == "old_agreement_gate" and side_top != language_top:
            event_weight = 0.0
        event_posterior = _softmax(
            base_event_logits + event_weight * event_side,
            temperature,
        )
        if detector_available:
            null_posterior = _sigmoid(
                _logit(null_prior) + null_side_weight * null_logit
            )
        else:
            null_posterior = null_prior
        return event_posterior, null_posterior, language_top, side_top

    for epoch in range(epochs):
        order_rng = _rng(model_seed, f"epoch-order-{epoch}")
        order = list(map(int, order_rng.permutation(len(ordered))))
        epoch_orders.append(canonical_digest(order))
        semantic_lr = float(learner["semantic_learning_rate"]) / (
            1.0 + float(learner["semantic_learning_rate_decay"]) * epoch
        )
        for row_index in order:
            row = ordered[row_index]
            items = list(row["utterance"]["items"])
            events = list(row["events"])
            event_posterior, null_posterior, language_top, side_top = components(row)
            if side_top != language_top:
                semantic_disagreement_count += 1
            if int(np.argmax(event_posterior)) != language_top:
                side_changed_event_top_count += 1
            if null_posterior >= 1.0 - 1e-12 and (
                condition == "exact_window"
                or bool(evidence.get(str(row["episode_id"]), {}).get("exact_window"))
            ):
                exact_window_null_selection_count += 1
            event_mass = 1.0 - null_posterior
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
        calibration_rows = []
        for row in ordered:
            _events = list(row["events"])
            _items = list(row["utterance"]["items"])
            _event_posterior, null_posterior, _language_top, _side_top = components(row)
            calibration_target = (
                null_prior
                if mutation == "presence_side_disconnected"
                else null_posterior
            )
            calibration_rows.append(
                (
                    _training_mismatch(semantics, _items, _events, config),
                    calibration_target,
                )
            )
        old_presence = (presence_intercept, presence_mismatch_weight)
        if mutation != "null_update_disconnected":
            presence_intercept, presence_mismatch_weight = _fit_presence_calibration(
                calibration_rows,
                config,
            )
        presence_update_norms.append(
            abs(presence_intercept - old_presence[0])
            + abs(presence_mismatch_weight - old_presence[1])
        )
        epoch_digests.append(
            canonical_digest(
                {
                    "semantics": {
                        key: list(map(float, value))
                        for key, value in sorted(semantics.items())
                    },
                    "presence_intercept": presence_intercept,
                    "presence_mismatch_weight": presence_mismatch_weight,
                }
            )
        )
    model = CorrectiveLexiconModel(
        semantics={
            key: tuple(map(float, value))
            for key, value in sorted(semantics.items())
        },
        presence_intercept=float(presence_intercept),
        presence_mismatch_weight=float(presence_mismatch_weight),
        model_seed=int(model_seed),
        epochs=epochs,
        learner=str(learner["name"]),
    )
    trace = {
        "schema_version": "nursery-corrective-training-trace-v3",
        "condition": condition,
        "mutation": mutation,
        "model_seed": int(model_seed),
        "epochs": epochs,
        "initial_parameter_digest": initial_digest,
        "initial_semantics_digest": initial_semantics_digest,
        "epoch_parameter_digests": epoch_digests,
        "distinct_epoch_parameter_digests": len(set(epoch_digests)),
        "epoch_order_digests": epoch_orders,
        "distinct_epoch_order_digests": len(set(epoch_orders)),
        "semantic_side_weight_effective": (
            0.0
            if mutation
            in {
                "semantic_side_zero",
                "evidence_disconnected",
                "semantic_update_disconnected",
            }
            else semantic_side_weight
        ),
        "null_side_to_semantic_event_mass_weight_effective": (
            0.0
            if mutation == "evidence_disconnected"
            else null_side_weight
        ),
        "null_side_to_presence_calibration_weight_effective": (
            0.0
            if mutation
            in {
                "presence_side_disconnected",
                "evidence_disconnected",
                "null_update_disconnected",
            }
            else null_side_weight
        ),
        "presence_side_disconnect_preserves_semantic_event_mass_gating": mutation
        == "presence_side_disconnected",
        "null_update_frozen_at_initial_prior": mutation
        == "null_update_disconnected",
        "disagreement_gate": mutation == "old_agreement_gate",
        "argmax_prefilter": False,
        "all_candidate_events_enter_posterior": True,
        "factorized_presence_and_conditional_event_posteriors": True,
        "presence_calibration_uses_training_only_side_soft_labels": mutation
        not in {
            "presence_side_disconnected",
            "evidence_disconnected",
            "null_update_disconnected",
        },
        "presence_calibration_test_feature_is_side_free_semantic_mismatch": True,
        "semantic_disagreement_count": semantic_disagreement_count,
        "side_changed_event_top_count": side_changed_event_top_count,
        "mean_null_posterior": float(np.mean(null_posterior_values)),
        "mean_event_posterior_mass": float(np.mean(event_mass_values)),
        "minimum_event_posterior_mass": float(np.min(event_mass_values)),
        "semantic_update_scaled_by_event_mass": True,
        "mean_semantic_update_l1": float(np.mean(semantic_update_norms)),
        "mean_null_update_absolute": float(np.mean(presence_update_norms)),
        "presence_intercept": float(presence_intercept),
        "presence_mismatch_weight": float(presence_mismatch_weight),
        "final_semantics_digest": canonical_digest(
            {
                key: list(value)
                for key, value in sorted(model.semantics.items())
            }
        ),
        "exact_window_null_selection_count": int(
            exact_window_null_selection_count
        ),
        "model_digest": canonical_digest(model.serializable()),
        "oracle_fields_consumed": False,
        "evaluation_keys_consumed": False,
    }
    return model, trace


# The v4 scientific core intentionally supersedes the retired v3 implementation
# above while retaining its public API for the runner and mutation harness.
fit_corrective_mil = _fit_corrective_mil_v3
