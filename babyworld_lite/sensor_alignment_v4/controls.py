from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .corpus import EvalKey, EvalPrompt
from .learner import LexicalModel, SLOT_DIMENSIONS
from .protocol import SeedFirewall, SeedReference, canonical_digest


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=float), 1e-12)
    return value / value.sum()


def fit_oracle_alignment(
    visible: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    firewall: SeedFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> tuple[LexicalModel, dict[str, Any]]:
    firewall.authorize(
        "control",
        [SeedReference("corpus", corpus_seed), SeedReference("model", model_seed)],
        purpose=purpose,
    )
    oracle_by_id = {str(row["episode_id"]): row for row in oracle}
    accumulators: dict[str, np.ndarray] = {}
    target_reads = 0
    for row in visible:
        truth = oracle_by_id[str(row["episode_id"])]
        target = int(truth["target_event_index"])
        event = row["events"][target]
        for item in row["utterance"]["items"]:
            slot = str(item["slot"])
            key = f"{slot}|{item['token']}"
            if key not in accumulators:
                accumulators[key] = np.ones(SLOT_DIMENSIONS[slot]) * 0.01
            if slot == "primitive":
                accumulators[key] += np.asarray(event["action_observation"][:3])
            elif slot == "manner":
                accumulators[key] += np.asarray(event["action_observation"][3:])
            elif slot == "noun":
                accumulators[key] += np.asarray(event["object_observation"])
            target_reads += 1
    model = LexicalModel(
        prototypes={key: tuple(map(float, _normalize(value))) for key, value in accumulators.items()},
        learner="executed_oracle_alignment_control_v4",
        model_seed=int(model_seed),
        iterations=1,
        competition_weight=0.0,
    )
    return model, {
        "implementation": "oracle target positions select actual candidate observations",
        "target_index_reads": target_reads,
        "assigned_score_constant": False,
        "model_digest": canonical_digest(model.serializable()),
    }


def fit_direct_capacity(
    prompts: Sequence[EvalPrompt],
    keys: Sequence[EvalKey],
    firewall: SeedFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> tuple[LexicalModel, dict[str, Any]]:
    firewall.authorize(
        "control",
        [SeedReference("corpus", corpus_seed), SeedReference("model", model_seed)],
        purpose=purpose,
    )
    key_by_id = {key.prompt_id: int(key.answer_index) for key in keys}
    accumulators: dict[str, np.ndarray] = {}
    supervised_rows = 0
    for prompt in prompts:
        answer = key_by_id[prompt.prompt_id]
        observation = np.asarray(prompt.observation, dtype=float)
        if prompt.kind == "action":
            primitive_word, manner_word = prompt.candidate_words[answer]
            for slot, word, value in (
                ("primitive", primitive_word, observation[:3]),
                ("manner", manner_word, observation[3:]),
            ):
                key = f"{slot}|{word}"
                accumulators.setdefault(key, np.ones(SLOT_DIMENSIONS[slot]) * 0.01)
                accumulators[key] += value
                supervised_rows += 1
        elif prompt.kind == "noun":
            word = prompt.candidate_words[answer]
            key = f"noun|{word}"
            accumulators.setdefault(key, np.ones(6) * 0.01)
            accumulators[key] += observation
            supervised_rows += 1
        else:
            raise ValueError(prompt.kind)
    model = LexicalModel(
        prototypes={key: tuple(map(float, _normalize(value))) for key, value in accumulators.items()},
        learner="executed_direct_capacity_control_v4",
        model_seed=int(model_seed),
        iterations=1,
        competition_weight=0.0,
    )
    return model, {
        "implementation": "supervised prototype fit over actual evaluation observations and sealed keys",
        "supervised_updates": supervised_rows,
        "assigned_score_constant": False,
        "model_digest": canonical_digest(model.serializable()),
    }
