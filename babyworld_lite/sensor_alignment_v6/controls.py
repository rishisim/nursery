from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .learner import LexicalModel, SLOT_DIMENSIONS
from .protocol import IdentifierFirewall, IdentifierReference, canonical_digest


def _key(item: Mapping[str, Any]) -> str:
    return f"{item['slot']}|{item['token']}"


def _observation(value: Mapping[str, Any], slot: str) -> np.ndarray:
    field = {
        "primitive": "primitive_observation",
        "manner": "manner_observation",
        "noun": "object_observation",
    }[slot]
    return np.asarray(value[field], dtype=float)


def _model_from_counts(
    counts: Mapping[str, np.ndarray], *, model_seed: int, learner: str
) -> LexicalModel:
    distributions = {
        key: tuple(map(float, value / value.sum())) for key, value in sorted(counts.items())
    }
    return LexicalModel(distributions, learner, model_seed, 1, 0.0)


def fit_oracle_control(
    visible: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> tuple[LexicalModel, dict[str, Any]]:
    firewall.authorize(
        "control",
        [IdentifierReference("corpus", corpus_seed), IdentifierReference("model", model_seed)],
        purpose=purpose,
    )
    oracle_by_id = {str(row["episode_id"]): row for row in oracle}
    counts: dict[str, np.ndarray] = {}
    observation_updates = 0
    null_updates = 0
    for row in visible:
        key_row = oracle_by_id[str(row["episode_id"])]
        for item in row["utterance"]["items"]:
            token_key = _key(item)
            slot = str(item["slot"])
            counts.setdefault(token_key, np.full(SLOT_DIMENSIONS[slot] + 1, 0.03))
            if key_row["grounded"]:
                event = row["events"][int(key_row["target_event_index"])]
                counts[token_key][:-1] += _observation(event, slot)
                observation_updates += 1
            else:
                counts[token_key][-1] += 1.0
                null_updates += 1
    model = _model_from_counts(counts, model_seed=model_seed, learner="v6_oracle_alignment_control")
    return model, {
        "control": "oracle_alignment_upper",
        "assigned_score_constant": False,
        "fitted_from_observed_target_events": observation_updates,
        "fitted_null_events": null_updates,
        "model_digest": canonical_digest(model.serializable()),
        "primary_learner": False,
    }


def fit_direct_capacity_control(
    prompts: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> tuple[LexicalModel, dict[str, Any]]:
    firewall.authorize(
        "control",
        [IdentifierReference("corpus", corpus_seed), IdentifierReference("model", model_seed)],
        purpose=purpose,
    )
    key_by_id = {str(row["prompt_id"]): int(row["answer_index"]) for row in keys}
    counts: dict[str, np.ndarray] = {}
    observation_updates = 0
    null_updates = 0
    for prompt in prompts:
        candidate = prompt["candidates"][key_by_id[str(prompt["prompt_id"])]]
        for item in prompt["tokens"]:
            token_key = _key(item)
            slot = str(item["slot"])
            counts.setdefault(token_key, np.full(SLOT_DIMENSIONS[slot] + 1, 0.01))
            if candidate.get("null_option"):
                counts[token_key][-1] += 1.0
                null_updates += 1
            else:
                counts[token_key][:-1] += 3.0 * _observation(candidate, slot)
                observation_updates += 1
    model = _model_from_counts(counts, model_seed=model_seed, learner="v6_direct_capacity_control")
    return model, {
        "control": "direct_capacity_upper",
        "assigned_score_constant": False,
        "supervised_observation_updates": observation_updates,
        "supervised_null_updates": null_updates,
        "model_digest": canonical_digest(model.serializable()),
        "primary_learner": False,
    }


def fit_corrupted_null_supervision_control(
    prompts: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> tuple[LexicalModel, dict[str, Any]]:
    firewall.authorize(
        "control",
        [IdentifierReference("corpus", corpus_seed), IdentifierReference("model", model_seed)],
        purpose=purpose,
    )
    key_by_id = {str(row["prompt_id"]): int(row["answer_index"]) for row in keys}
    counts: dict[str, np.ndarray] = {}
    corrupted_null_updates = 0
    retained_present_updates = 0
    for prompt in prompts:
        correct = prompt["candidates"][key_by_id[str(prompt["prompt_id"])]]
        if correct.get("null_option"):
            candidate = sorted(
                (
                    row
                    for row in prompt["candidates"]
                    if not bool(row.get("null_option"))
                ),
                key=lambda row: str(row["candidate_id"]),
            )[0]
            corrupted_null_updates += len(prompt["tokens"])
        else:
            candidate = correct
            retained_present_updates += len(prompt["tokens"])
        for item in prompt["tokens"]:
            token_key = _key(item)
            slot = str(item["slot"])
            counts.setdefault(token_key, np.full(SLOT_DIMENSIONS[slot] + 1, 0.01))
            counts[token_key][:-1] += 3.0 * _observation(candidate, slot)
    model = _model_from_counts(
        counts,
        model_seed=model_seed,
        learner="v6_corrupted_null_supervision_control",
    )
    return model, {
        "control": "corrupted_null_supervision",
        "assigned_score_constant": False,
        "corrupted_null_updates": corrupted_null_updates,
        "retained_present_updates": retained_present_updates,
        "model_digest": canonical_digest(model.serializable()),
        "primary_learner": False,
        "uses_corrupted_evaluation_supervision_only_as_negative_capacity_diagnostic": True,
    }


__all__ = [
    "fit_oracle_control",
    "fit_direct_capacity_control",
    "fit_corrupted_null_supervision_control",
]
