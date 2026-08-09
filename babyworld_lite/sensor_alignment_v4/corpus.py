from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import SeedFirewall, SeedReference, reject_oracle_fields


@dataclass(frozen=True)
class EvalPrompt:
    prompt_id: str
    kind: str
    observation: tuple[float, ...]
    candidate_words: tuple[Any, ...]


@dataclass(frozen=True)
class EvalKey:
    prompt_id: str
    answer_index: int


@dataclass(frozen=True)
class Corpus:
    corpus_seed: int
    visible_episodes: tuple[dict[str, Any], ...]
    oracle_episodes: tuple[dict[str, Any], ...]
    evaluation_prompts: tuple[EvalPrompt, ...]
    evaluation_keys: tuple[EvalKey, ...]
    lexicon_oracle: dict[str, Any]
    audit: dict[str, Any]


def _rng(seed: int, label: str) -> np.random.Generator:
    value = int.from_bytes(
        hashlib.sha256(f"v4|{seed}|{label}".encode()).digest()[:8], "little"
    )
    return np.random.default_rng(value)


def _components(action: int) -> tuple[int, int]:
    return action // 2, action % 2


def _lexicon(seed: int) -> dict[str, Any]:
    rng = _rng(seed, "lexicon")
    primitive = [f"vp{seed:x}{int(value)}" for value in rng.permutation(3)]
    manner = [f"vm{seed:x}{int(value)}" for value in rng.permutation(2)]
    noun = [f"vn{seed:x}{int(value)}" for value in rng.permutation(6)]
    return {"primitive": primitive, "manner": manner, "noun": noun}


def _observation(index: int, dimension: int, visibility: float) -> list[float]:
    uniform = np.full(dimension, 1.0 / dimension)
    ideal = np.zeros(dimension)
    ideal[index] = 1.0
    value = visibility * ideal + (1.0 - visibility) * uniform
    return np.round(value, 8).tolist()


def _relation_candidates(target: int, repetition: int) -> list[int]:
    target_primitive, target_manner = _components(target)
    cells: dict[str, list[int]] = {
        "primitive_same=1|manner_same=0": [],
        "primitive_same=0|manner_same=1": [],
        "primitive_same=0|manner_same=0": [],
    }
    for candidate in range(6):
        if candidate == target:
            continue
        primitive, manner = _components(candidate)
        key = (
            f"primitive_same={int(primitive == target_primitive)}|"
            f"manner_same={int(manner == target_manner)}"
        )
        cells[key].append(candidate)
    return [
        cells["primitive_same=1|manner_same=0"][0],
        cells["primitive_same=0|manner_same=1"][repetition % 2],
        cells["primitive_same=0|manner_same=0"][(repetition // 2) % 2],
    ]


def _noun_distractors(target: int, repetition: int) -> list[int]:
    others = [value for value in range(6) if value != target]
    start = repetition % len(others)
    return [others[(start + offset) % len(others)] for offset in range(3)]


def _rotate_target_first(values: Sequence[int], target_position: int) -> list[int]:
    output = list(values[1:])
    output.insert(target_position, int(values[0]))
    return output


def _raw_stream(
    rng: np.random.Generator,
    intervals: Sequence[tuple[int, int]],
    owners: Sequence[bool],
) -> dict[str, Any]:
    samples = 64
    base = rng.normal(0.0, 0.055, (samples, 9))
    activity = np.zeros(samples)
    for (start, end), owner in zip(intervals, owners):
        if owner:
            activity[start : end + 1] += 2.0
    base[:, :6] += activity[:, None] * np.asarray([0.72, -0.42, 0.61, 0.34, -0.24, 0.51])
    base[:, 6:8] += activity[:, None] * np.asarray([0.52, 0.81])
    base[:, 8] += activity
    availability = np.ones((samples, 9), dtype=int)
    return {
        "timestamps": list(range(samples)),
        "imu": np.round(base[:, :6], 6).tolist(),
        "proprio": np.round(base[:, 6:8], 6).tolist(),
        "contact": np.round(base[:, 8:], 6).tolist(),
        "availability": availability.tolist(),
    }


def generate_corpus(
    corpus_seed: int,
    config: Mapping[str, Any],
    firewall: SeedFirewall,
    *,
    purpose: str,
) -> Corpus:
    firewall.authorize(
        "generate", [SeedReference("corpus", corpus_seed)], purpose=purpose
    )
    design = config["design"]
    repetitions = int(design["episodes_per_concept"])
    if repetitions != 16:
        raise ValueError("v4 exact noun factorial requires 16 episodes per concept")
    visibilities = list(map(float, design["visibility_levels"]))
    lags = list(map(int, design["lag_levels"]))
    if len(visibilities) != 2 or len(lags) != 2:
        raise ValueError("v4 fixture predeclares exactly two visibility and lag levels")
    lexicon = _lexicon(corpus_seed)
    visible: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    episode_number = 0
    for family in ("action", "noun"):
        for concept in range(6):
            for repetition in range(repetitions):
                target_position = repetition % 4
                if family == "action":
                    candidates = _rotate_target_first(
                        [concept, *_relation_candidates(concept, repetition)],
                        target_position,
                    )
                    owner_position = target_position
                else:
                    candidates = _rotate_target_first(
                        [concept, *_noun_distractors(concept, repetition)],
                        target_position,
                    )
                    owner_position = (repetition // 4) % 4
                visibility = visibilities[repetition % 2]
                lag = lags[(repetition // 2) % 2]
                intervals = [(4 + 14 * index, 10 + 14 * index) for index in range(4)]
                events: list[dict[str, Any]] = []
                for index, candidate in enumerate(candidates):
                    if family == "action":
                        action = candidate
                        obj = (concept + repetition + index) % 6
                    else:
                        action = (concept + repetition + index) % 6
                        obj = candidate
                    primitive, manner = _components(action)
                    events.append(
                        {
                            "event_id": f"e{index}",
                            "start": intervals[index][0],
                            "end": intervals[index][1],
                            "action_observation": [
                                *_observation(primitive, 3, visibility),
                                *_observation(manner, 2, visibility),
                            ],
                            "object_observation": _observation(obj, 6, visibility),
                        }
                    )
                owners = [index == owner_position for index in range(4)]
                raw = _raw_stream(
                    _rng(corpus_seed, f"raw-{family}-{concept}-{repetition}"),
                    intervals,
                    owners,
                )
                items = (
                    [
                        {"slot": "primitive", "token": lexicon["primitive"][_components(concept)[0]]},
                        {"slot": "manner", "token": lexicon["manner"][_components(concept)[1]]},
                    ]
                    if family == "action"
                    else [{"slot": "noun", "token": lexicon["noun"][concept]}]
                )
                scene_midpoint = float(
                    np.mean([(event["start"] + event["end"]) / 2 for event in events])
                )
                episode_id = f"v4-{corpus_seed}-{family[0]}-{concept}-{repetition:02d}"
                visible_row = {
                    "schema_version": "nursery-v4-visible",
                    "episode_id": episode_id,
                    "family": family,
                    "utterance": {"items": items, "speech_time": scene_midpoint + lag},
                    "events": events,
                    "factor_values": {"visibility": visibility, "lag": lag},
                    "raw_stream": raw,
                }
                oracle_row = {
                    "schema_version": "nursery-v4-oracle",
                    "episode_id": episode_id,
                    "corpus_seed": int(corpus_seed),
                    "family": family,
                    "intended_concept": concept,
                    "target_event_index": target_position,
                    "event_concepts": candidates,
                    "event_owners": owners,
                    "grounded": True,
                    "factor_values": {"visibility": visibility, "lag": lag},
                }
                reject_oracle_fields(visible_row)
                visible.append(visible_row)
                oracle.append(oracle_row)
                episode_number += 1
    prompts: list[EvalPrompt] = []
    keys: list[EvalKey] = []
    action_phrases = tuple(
        (
            lexicon["primitive"][_components(action)[0]],
            lexicon["manner"][_components(action)[1]],
        )
        for action in range(6)
    )
    noun_words = tuple(lexicon["noun"])
    for action in range(6):
        primitive, manner = _components(action)
        prompt_id = f"v4-{corpus_seed}-eval-action-{action}"
        prompts.append(
            EvalPrompt(
                prompt_id,
                "action",
                tuple([*_observation(primitive, 3, 1.0), *_observation(manner, 2, 1.0)]),
                action_phrases,
            )
        )
        keys.append(EvalKey(prompt_id, action))
    for noun in range(6):
        prompt_id = f"v4-{corpus_seed}-eval-noun-{noun}"
        prompts.append(
            EvalPrompt(prompt_id, "noun", tuple(_observation(noun, 6, 1.0)), noun_words)
        )
        keys.append(EvalKey(prompt_id, noun))
    audit = audit_corpus(visible, oracle)
    return Corpus(
        corpus_seed=int(corpus_seed),
        visible_episodes=tuple(visible),
        oracle_episodes=tuple(oracle),
        evaluation_prompts=tuple(prompts),
        evaluation_keys=tuple(keys),
        lexicon_oracle=lexicon,
        audit=audit,
    )


def audit_corpus(
    visible: Sequence[Mapping[str, Any]], oracle: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(visible) != len(oracle):
        raise ValueError("visible/oracle length mismatch")
    relation_by_concept: dict[str, Counter[str]] = {
        str(concept): Counter() for concept in range(6)
    }
    target_positions: dict[str, Counter[int]] = {
        f"{family}:{concept}": Counter()
        for family in ("action", "noun")
        for concept in range(6)
    }
    noun_joint: dict[str, Counter[str]] = {str(concept): Counter() for concept in range(6)}
    noun_target_owner = 0
    noun_distractor_owner = 0
    noun_distractor_opportunities = 0
    unique_candidates = True
    for visible_row, oracle_row in zip(visible, oracle):
        reject_oracle_fields(visible_row)
        if visible_row["episode_id"] != oracle_row["episode_id"]:
            raise ValueError("visible/oracle episode misalignment")
        concept = int(oracle_row["intended_concept"])
        target = int(oracle_row["target_event_index"])
        candidates = list(map(int, oracle_row["event_concepts"]))
        owners = list(map(bool, oracle_row["event_owners"]))
        unique_candidates &= len(candidates) == len(set(candidates)) == 4
        family = str(oracle_row["family"])
        target_positions[f"{family}:{concept}"][target] += 1
        owner_position = int(np.argmax(owners))
        if family == "action":
            target_primitive, target_manner = _components(concept)
            for index, candidate in enumerate(candidates):
                if index == target:
                    continue
                primitive, manner = _components(candidate)
                relation_by_concept[str(concept)][
                    f"primitive_same={int(primitive == target_primitive)}|"
                    f"manner_same={int(manner == target_manner)}"
                ] += 1
        else:
            noun_joint[str(concept)][f"target={target}|owner={owner_position}"] += 1
            noun_target_owner += int(owners[target])
            noun_distractor_owner += sum(
                int(owner) for index, owner in enumerate(owners) if index != target
            )
            noun_distractor_opportunities += len(owners) - 1
    expected_relations = {
        "primitive_same=0|manner_same=0": 16,
        "primitive_same=0|manner_same=1": 16,
        "primitive_same=1|manner_same=0": 16,
    }
    exact_relation = all(
        dict(sorted(counts.items())) == expected_relations
        for counts in relation_by_concept.values()
    )
    exact_joint = all(
        len(counts) == 16 and set(counts.values()) == {1}
        for counts in noun_joint.values()
    )
    target_rate = noun_target_owner / 96
    distractor_rate = noun_distractor_owner / noun_distractor_opportunities
    exact_positions = all(
        dict(sorted(counts.items())) == {0: 4, 1: 4, 2: 4, 3: 4}
        for counts in target_positions.values()
    )
    return {
        "status": "PASS"
        if exact_relation
        and exact_joint
        and target_rate == distractor_rate == 0.25
        and exact_positions
        and unique_candidates
        else "FAIL",
        "visible_oracle_separation": True,
        "unique_candidates": bool(unique_candidates),
        "action_relation_counts_by_concept": {
            key: dict(sorted(value.items())) for key, value in relation_by_concept.items()
        },
        "exact_action_relation_cell_balance": bool(exact_relation),
        "target_position_counts": {
            key: dict(sorted(value.items())) for key, value in target_positions.items()
        },
        "exact_target_position_balance": bool(exact_positions),
        "noun_target_owner_position_joint_counts": {
            key: dict(sorted(value.items())) for key, value in noun_joint.items()
        },
        "exact_noun_target_owner_factorial": bool(exact_joint),
        "noun_target_owner_rate": float(target_rate),
        "noun_distractor_owner_rate_per_opportunity": float(distractor_rate),
        "noun_owner_independent_of_identity_and_target": bool(
            exact_joint and target_rate == distractor_rate == 0.25
        ),
        "grounding_design": "always_present_fixed_no_variation",
    }


def condition_view(
    visible: Sequence[Mapping[str, Any]], condition: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if condition not in {"synchronized", "shuffled", "absent"}:
        raise ValueError(condition)
    rows = json.loads(json.dumps(list(visible)))
    donor_map: dict[str, str] = {}
    if condition == "absent":
        for row in rows:
            row.pop("raw_stream", None)
    elif condition == "shuffled":
        donors = [row["raw_stream"] for row in rows]
        donor_ids = [str(row["episode_id"]) for row in rows]
        shift = len(rows) // 2 + 1
        for index, row in enumerate(rows):
            donor_index = (index + shift) % len(rows)
            row["raw_stream"] = donors[donor_index]
            donor_map[str(row["episode_id"])] = donor_ids[donor_index]
    else:
        donor_map = {str(row["episode_id"]): str(row["episode_id"]) for row in rows}
    for row in rows:
        reject_oracle_fields(row)
    return rows, donor_map


def serializable_corpus(corpus: Corpus) -> dict[str, Any]:
    return {
        "corpus_seed": corpus.corpus_seed,
        "visible_episodes": list(corpus.visible_episodes),
        "oracle_episodes": list(corpus.oracle_episodes),
        "evaluation_prompts": [asdict(value) for value in corpus.evaluation_prompts],
        "evaluation_keys": [asdict(value) for value in corpus.evaluation_keys],
        "lexicon_oracle": corpus.lexicon_oracle,
        "audit": corpus.audit,
    }
