from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import IdentifierFirewall, IdentifierReference, canonical_digest, reject_oracle_fields


@dataclass(frozen=True, slots=True)
class Corpus:
    corpus_seed: int
    visible_episodes: tuple[dict[str, Any], ...]
    oracle_episodes: tuple[dict[str, Any], ...]
    evaluation_prompts: tuple[dict[str, Any], ...]
    evaluation_keys: tuple[dict[str, Any], ...]
    lexicon_oracle: dict[str, Any]
    audit: dict[str, Any]


def _rng(seed: int, label: str) -> np.random.Generator:
    integer = int.from_bytes(
        hashlib.sha256(f"nursery-v7|{seed}|{label}".encode()).digest()[:8], "little"
    )
    return np.random.default_rng(integer)


def _one_hot(index: int, dimension: int, visibility: float) -> list[float]:
    uniform = np.full(dimension, 1.0 / dimension)
    ideal = np.zeros(dimension)
    ideal[index] = 1.0
    return np.round(visibility * ideal + (1.0 - visibility) * uniform, 8).tolist()


def _components(action: int) -> tuple[int, int]:
    return action // 2, action % 2


def _lexicon(seed: int) -> dict[str, list[str]]:
    rng = _rng(seed, "lexicon")
    return {
        "primitive": [f"q7p{seed:x}{value}" for value in rng.permutation(3)],
        "manner": [f"q7m{seed:x}{value}" for value in rng.permutation(2)],
        "noun": [f"q7n{seed:x}{value}" for value in rng.permutation(6)],
    }


def _balanced_schedule(seed: int, label: str, levels: Sequence[Any], count: int) -> list[Any]:
    values = [levels[index % len(levels)] for index in range(count)]
    permutation = _rng(seed, f"schedule-{label}").permutation(count)
    return [values[int(index)] for index in permutation]


def _raw_stream(
    seed: int,
    intervals: Sequence[tuple[int, int]],
    *,
    target_index: int | None,
    owner_index: int | None,
    stratum: str,
    dropout: float,
    noise: float,
    false_positive: int,
    samples: int,
) -> dict[str, Any]:
    rng = _rng(seed, "raw")
    base = rng.normal(0.0, noise, (samples, 9))
    activity = np.zeros(samples)
    if stratum == "informative" and target_index is not None:
        active_index, amplitude = target_index, 2.0
    elif stratum == "weak" and target_index is not None:
        active_index, amplitude = target_index, 0.58
    elif stratum == "corrupted" and intervals:
        active_index = ((target_index if target_index is not None else 0) + 1) % len(intervals)
        amplitude = 2.0
    else:
        active_index, amplitude = None, 0.0
    if owner_index is not None and target_index is None and stratum in {"informative", "weak"}:
        active_index = owner_index
        amplitude = 0.45 if stratum == "weak" else 0.85
    if active_index is not None:
        start, end = intervals[active_index]
        activity[start : end + 1] += amplitude
    if false_positive and intervals:
        false_index = ((active_index if active_index is not None else 0) + 2) % len(intervals)
        start, end = intervals[false_index]
        activity[start : end + 1] += 0.72
    base[:, :6] += activity[:, None] * np.asarray([0.72, -0.42, 0.61, 0.34, -0.24, 0.51])
    base[:, 6:8] += activity[:, None] * np.asarray([0.52, 0.81])
    base[:, 8] += activity
    availability = np.ones((samples, 9), dtype=int)
    if dropout:
        dropout_count = int(round(samples * dropout))
        indices = _rng(seed, "dropout").choice(samples, dropout_count, replace=False)
        availability[indices] = 0
        base[indices] = 0.0
    return {
        "timestamps": list(range(samples)),
        "imu": np.round(base[:, :6], 6).tolist(),
        "proprio": np.round(base[:, 6:8], 6).tolist(),
        "contact": np.round(base[:, 8:], 6).tolist(),
        "availability": availability.tolist(),
    }


def _prompt_candidate(
    candidate_id: str,
    *,
    action: int | None = None,
    noun: int | None = None,
    primitive: int | None = None,
    manner: int | None = None,
    null: bool = False,
) -> dict[str, Any]:
    if null:
        return {"candidate_id": candidate_id, "null_option": True}
    if action is not None:
        primitive_value, manner_value = _components(action)
    else:
        primitive_value = primitive
        manner_value = manner
    candidate: dict[str, Any] = {"candidate_id": candidate_id, "null_option": False}
    if primitive_value is not None:
        candidate["primitive_observation"] = _one_hot(int(primitive_value), 3, 1.0)
    if manner_value is not None:
        candidate["manner_observation"] = _one_hot(int(manner_value), 2, 1.0)
    if noun is not None:
        candidate["object_observation"] = _one_hot(noun, 6, 1.0)
    return candidate


def _evaluation(seed: int, lexicon: Mapping[str, Sequence[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    specifications: list[tuple[str, int, list[dict[str, str]], list[dict[str, Any]]]] = []
    for primitive in range(3):
        specifications.append(
            (
                "primitive",
                primitive,
                [{"slot": "primitive", "token": lexicon["primitive"][primitive]}],
                [_prompt_candidate(f"primitive-{value}", primitive=value) for value in range(3)],
            )
        )
    for manner in range(2):
        specifications.append(
            (
                "manner",
                manner,
                [{"slot": "manner", "token": lexicon["manner"][manner]}],
                [_prompt_candidate(f"manner-{value}", manner=value) for value in range(2)],
            )
        )
    for action in range(6):
        primitive, manner = _components(action)
        specifications.append(
            (
                "action",
                action,
                [
                    {"slot": "primitive", "token": lexicon["primitive"][primitive]},
                    {"slot": "manner", "token": lexicon["manner"][manner]},
                ],
                [_prompt_candidate(f"action-{value}", action=value) for value in range(6)],
            )
        )
    for noun in range(6):
        specifications.append(
            (
                "noun",
                noun,
                [{"slot": "noun", "token": lexicon["noun"][noun]}],
                [_prompt_candidate(f"noun-{value}", noun=value) for value in range(6)],
            )
        )
    for kind, concept, tokens, full_candidates in specifications:
        for presence in ("present", "absent"):
            candidates = json.loads(json.dumps(full_candidates))
            correct_id = f"{kind}-{concept}"
            if presence == "absent":
                candidates = [row for row in candidates if row["candidate_id"] != correct_id]
                correct_id = f"{kind}-null"
            candidates.append(_prompt_candidate(f"{kind}-null", null=True))
            order = _rng(seed, f"eval-{kind}-{concept}-{presence}").permutation(len(candidates))
            candidates = [candidates[int(index)] for index in order]
            prompt_id = f"v7-{seed}-eval-{kind}-{concept}-{presence}"
            prompt = {
                "schema_version": "nursery-v7-language-prompt",
                "prompt_id": prompt_id,
                "kind": kind,
                "tokens": tokens,
                "candidates": candidates,
            }
            reject_oracle_fields(prompt)
            answer_index = next(
                index for index, candidate in enumerate(candidates) if candidate["candidate_id"] == correct_id
            )
            prompts.append(prompt)
            keys.append(
                {
                    "schema_version": "nursery-v7-sealed-key",
                    "prompt_id": prompt_id,
                    "answer_index": answer_index,
                }
            )
    return prompts, keys


def _rotate(values: Sequence[int], amount: int) -> list[int]:
    data = list(values)
    if not data:
        return data
    shift = amount % len(data)
    return data[shift:] + data[:shift]


def _structural_candidates(
    *,
    family: str,
    concept: int,
    count: int,
    grounded: bool,
    target_position: int | None,
    cycle: int,
    corpus_seed: int,
    sensor_stratum: str,
) -> tuple[list[int], int | None]:
    others = [
        int(value)
        for value in _rng(
            corpus_seed,
            f"structural-candidates|{family}|{concept}|{count}|{sensor_stratum}",
        ).permutation([value for value in range(6) if value != concept])
    ]
    if family == "action":
        selected = _rotate(others, cycle)[: count - 1 if grounded else count]
    else:
        selected = [
            int(value)
            for value in _rng(
                corpus_seed,
                (
                    f"noun-candidates|{concept}|{count}|{sensor_stratum}|"
                    f"{int(grounded)}|{target_position}|{cycle}"
                ),
            ).permutation(others)
        ][: count - 1 if grounded else count]
    if not grounded:
        return selected, None
    if target_position is None or not 0 <= target_position < count:
        raise ValueError("grounded structural row requires an in-range target position")
    selected.insert(int(target_position), int(concept))
    return selected, int(target_position)


def _structural_specs(
    corpus_seed: int,
    family: str,
    concept: int,
    design: Mapping[str, Any],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    strata = list(map(str, design["detector_strata"]))
    action_cycles = int(design["action_distractor_cycles"])
    noun_repetitions = int(
        design["noun_target_owner_pair_repetitions_per_sensor_stratum"]
    )
    if action_cycles != 5:
        raise ValueError("six-way action balance requires exactly five distractor cycles")
    if noun_repetitions < 1:
        raise ValueError("noun structural repetition count must be positive")
    for count in map(int, design["candidate_counts"]):
        for sensor_stratum in strata:
            if family == "noun":
                for repetition in range(noun_repetitions):
                    for target_position in range(count):
                        for owner_position in range(count):
                            specs.append(
                                {
                                    "candidate_count": count,
                                    "grounded": True,
                                    "target_position": target_position,
                                    "owner_position": owner_position,
                                    "candidate_cycle": repetition * count + owner_position,
                                    "sensor_stratum": sensor_stratum,
                                }
                            )
                    for owner_position in range(count):
                        specs.append(
                            {
                                "candidate_count": count,
                                "grounded": False,
                                "target_position": None,
                                "owner_position": owner_position,
                                "candidate_cycle": repetition * count + owner_position,
                                "sensor_stratum": sensor_stratum,
                            }
                        )
            else:
                # Five cyclic distractor windows make every one of the five
                # non-target actions appear exactly count-1 times per target
                # position. The resulting primitive/manner relation inventory
                # is therefore exactly the available 1:2:2 lattice ratio.
                for target_position in range(count):
                    for cycle in range(action_cycles):
                        specs.append(
                            {
                                "candidate_count": count,
                                "grounded": True,
                                "target_position": target_position,
                                "owner_position": target_position,
                                "candidate_cycle": cycle,
                                "sensor_stratum": sensor_stratum,
                            }
                        )
                for cycle in range(action_cycles):
                    specs.append(
                        {
                            "candidate_count": count,
                            "grounded": False,
                            "target_position": None,
                            "owner_position": None,
                            "candidate_cycle": cycle,
                            "sensor_stratum": sensor_stratum,
                        }
                    )
    return specs


def _assign_balanced_factor(
    family_specs: list[dict[str, Any]],
    indices: Sequence[int],
    *,
    family: str,
    factor: str,
    levels: Sequence[Any],
    seed: int,
    previously_assigned: Sequence[str],
) -> None:
    """Choose a balanced factor permutation with low structural association.

    This is a pre-outcome construction search over deterministic permutations.
    It uses only declared design-cell labels, never learner outputs or evaluation
    keys.  The resulting actual associations are audited again from persisted
    rows.
    """

    base = [levels[index % len(levels)] for index in range(len(indices))]
    control_names = (
        "_concept",
        "grounded",
        "sensor_stratum",
        "target_position",
        "owner_position",
        "_target_is_owner",
        "candidate_cycle",
        *previously_assigned,
    )
    best_values: list[Any] | None = None
    best_score: tuple[float, str] | None = None
    for trial in range(256):
        permutation = _rng(seed, f"factor-balance|{factor}|trial={trial}").permutation(
            len(base)
        )
        values = [base[int(index)] for index in permutation]
        associations = []
        for control in control_names:
            selected_pairs = [
                (family_specs[index].get(control), values[position])
                for position, index in enumerate(indices)
                if family_specs[index].get(control) is not None
            ]
            if (
                len({str(left) for left, _ in selected_pairs}) < 2
                or len({str(right) for _, right in selected_pairs}) < 2
            ):
                continue
            associations.append(
                _cramers_v(
                    [left for left, _ in selected_pairs],
                    [right for _, right in selected_pairs],
                )
            )
        score = (
            max(associations, default=0.0),
            canonical_digest(values),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_values = values
    if best_values is None:
        raise RuntimeError(f"no factor assignment for {family}/{factor}")
    for position, index in enumerate(indices):
        family_specs[index][factor] = best_values[position]


def generate_corpus(
    corpus_seed: int,
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    purpose: str,
) -> Corpus:
    firewall.authorize(
        "generate", [IdentifierReference("corpus", int(corpus_seed))], purpose=purpose
    )
    design = config["design"]
    lexicon = _lexicon(corpus_seed)
    visible: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for family_index, family in enumerate(("action", "noun")):
        family_specs: list[dict[str, Any]] = []
        for concept in range(6):
            for repetition, factors in enumerate(
                _structural_specs(corpus_seed, family, concept, design)
            ):
                family_specs.append(
                    {
                        **factors,
                        "_concept": concept,
                        "_concept_repetition": repetition,
                        "_target_is_owner": (
                            int(
                                int(factors["target_position"])
                                == int(factors["owner_position"])
                            )
                            if factors["target_position"] is not None
                            and factors["owner_position"] is not None
                            else None
                        ),
                    }
                )
        indices_by_count: defaultdict[int, list[int]] = defaultdict(list)
        for index, factors in enumerate(family_specs):
            indices_by_count[int(factors["candidate_count"])].append(index)
        factor_levels = {
            "visibility": design["action_visibility_levels"],
            "lag": design["lags"],
            "dropout": design["dropout_levels"],
            "noise": design["noise_levels"],
            "false_positive": design["false_positive_levels"],
            "repetition_phase": [0, 1, 2, 3],
        }
        for count, indices in sorted(indices_by_count.items()):
            schedule_seed = (
                corpus_seed
                + family_index * 100_000
                + count * 1_000
            )
            assigned: list[str] = []
            for factor, levels in factor_levels.items():
                _assign_balanced_factor(
                    family_specs,
                    indices,
                    family=family,
                    factor=factor,
                    levels=levels,
                    seed=schedule_seed,
                    previously_assigned=assigned,
                )
                assigned.append(factor)
        for factors in family_specs:
                concept = int(factors["_concept"])
                repetition = int(factors["_concept_repetition"])
                count = int(factors["candidate_count"])
                grounded = bool(factors["grounded"])
                episode_seed = (
                    corpus_seed * 10_000_000
                    + family_index * 1_000_000
                    + concept * 100_000
                    + repetition
                )
                candidates, target_index = _structural_candidates(
                    family=family,
                    concept=concept,
                    count=count,
                    grounded=grounded,
                    target_position=factors["target_position"],
                    cycle=int(factors["candidate_cycle"]),
                    corpus_seed=corpus_seed,
                    sensor_stratum=str(factors["sensor_stratum"]),
                )
                if family == "noun":
                    owner_index = int(factors["owner_position"])
                else:
                    owner_index = target_index
                intervals = [(3 + 13 * index, 3 + 13 * index + int(design["interval_width"]) - 1) for index in range(count)]
                events = []
                visibility = float(factors["visibility"])
                for event_index, candidate in enumerate(candidates):
                    action = candidate if family == "action" else (concept + repetition + event_index) % 6
                    noun = candidate if family == "noun" else (concept * 3 + repetition + event_index) % 6
                    primitive, manner = _components(action)
                    events.append(
                        {
                            "event_id": f"event-{event_index}",
                            "start": intervals[event_index][0],
                            "end": intervals[event_index][1],
                            "primitive_observation": _one_hot(primitive, 3, visibility),
                            "manner_observation": _one_hot(manner, 2, visibility),
                            "object_observation": _one_hot(noun, 6, visibility),
                        }
                    )
                raw = _raw_stream(
                    episode_seed,
                    intervals,
                    target_index=target_index if family == "action" else None,
                    owner_index=owner_index,
                    stratum=str(factors["sensor_stratum"]),
                    dropout=float(factors["dropout"]),
                    noise=float(factors["noise"]),
                    false_positive=int(factors["false_positive"]),
                    samples=int(design["raw_samples"]),
                )
                if family == "action":
                    primitive, manner = _components(concept)
                    tokens = [
                        {"slot": "primitive", "token": lexicon["primitive"][primitive]},
                        {"slot": "manner", "token": lexicon["manner"][manner]},
                    ]
                else:
                    tokens = [{"slot": "noun", "token": lexicon["noun"][concept]}]
                midpoint = float(np.mean([(start + end) / 2 for start, end in intervals]))
                episode_id = f"v7-{corpus_seed}-{family[0]}-{concept}-{repetition:02d}"
                visible_row = {
                    "schema_version": "nursery-v7-visible",
                    "episode_id": episode_id,
                    "family": family,
                    "utterance": {"items": tokens, "speech_time": midpoint + int(factors["lag"])},
                    "events": events,
                    "factor_values": {
                        **{
                            key: value
                            for key, value in factors.items()
                            if key
                            not in {
                                "_concept",
                                "_concept_repetition",
                                "_target_is_owner",
                                "grounded",
                                "target_position",
                                "owner_position",
                                "candidate_cycle",
                            }
                        },
                        "candidate_count": count,
                        "repetition": repetition,
                    },
                    "raw_stream": raw,
                }
                oracle_row = {
                    "schema_version": "nursery-v7-oracle",
                    "episode_id": episode_id,
                    "corpus_seed": int(corpus_seed),
                    "family": family,
                    "intended_concept": concept,
                    "target_event_index": target_index,
                    "event_concepts": candidates,
                    "event_owners": [index == owner_index for index in range(count)],
                    "grounded": grounded,
                    "factor_values": {
                        **visible_row["factor_values"],
                        "grounded_rate_stratum": "grounded" if grounded else "target_absent",
                        "target_position": target_index,
                        "owner_position": owner_index,
                        "candidate_cycle": int(factors["candidate_cycle"]),
                    },
                }
                reject_oracle_fields(visible_row)
                visible.append(visible_row)
                oracle.append(oracle_row)
    prompts, keys = _evaluation(corpus_seed, lexicon)
    audit = audit_corpus(visible, oracle, config)
    return Corpus(
        int(corpus_seed),
        tuple(visible),
        tuple(oracle),
        tuple(prompts),
        tuple(keys),
        lexicon,
        audit,
    )


def _cramers_v(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_levels = {value: index for index, value in enumerate(sorted(set(map(str, left))))}
    right_levels = {value: index for index, value in enumerate(sorted(set(map(str, right))))}
    table = np.zeros((len(left_levels), len(right_levels)), dtype=float)
    for a, b in zip(map(str, left), map(str, right)):
        table[left_levels[a], right_levels[b]] += 1
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    valid = expected > 0
    chi = float(np.sum(((table - expected) ** 2)[valid] / expected[valid]))
    denominator = table.sum() * max(1, min(table.shape[0] - 1, table.shape[1] - 1))
    return math.sqrt(chi / denominator)


def audit_corpus(
    visible: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(visible) != len(oracle):
        raise ValueError("visible/oracle count mismatch")
    noun_owner_counts: defaultdict[tuple[int, int, str], Counter[str]] = defaultdict(
        Counter
    )
    noun_joint_counts: defaultdict[
        tuple[int, int, str], Counter[tuple[int, int]]
    ] = defaultdict(Counter)
    noun_absent_owner_counts: defaultdict[
        tuple[int, int, str], Counter[int]
    ] = defaultdict(Counter)
    action_relation: Counter[str] = Counter()
    action_relation_by_group: defaultdict[
        tuple[int, int, str], Counter[str]
    ] = defaultdict(Counter)
    action_target_positions: defaultdict[
        tuple[int, int, str], Counter[int]
    ] = defaultdict(Counter)
    control_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    factors: defaultdict[str, list[Any]] = defaultdict(list)
    candidate_counts: Counter[int] = Counter()
    grounded: Counter[bool] = Counter()
    for row, key in zip(visible, oracle):
        reject_oracle_fields(row)
        if row["episode_id"] != key["episode_id"]:
            raise ValueError("ledger misalignment")
        values = row["factor_values"]
        for factor in (
            "candidate_count",
            "visibility",
            "lag",
            "grounded_rate_stratum",
            "sensor_stratum",
            "dropout",
            "noise",
            "false_positive",
            "repetition_phase",
        ):
            factors[factor].append(
                key["factor_values"][factor] if factor == "grounded_rate_stratum" else values[factor]
            )
        candidate_counts[int(values["candidate_count"])] += 1
        grounded[bool(key["grounded"])] += 1
        if key["family"] == "noun":
            owners = list(map(bool, key["event_owners"]))
            target = key["target_event_index"]
            owner_positions = [index for index, owner in enumerate(owners) if owner]
            if len(owner_positions) != 1:
                raise ValueError("noun structural construction requires exactly one owner")
            owner_position = int(owner_positions[0])
            group = (
                int(key["intended_concept"]),
                len(owners),
                str(values["sensor_stratum"]),
            )
            if target is not None:
                counts = noun_owner_counts[group]
                counts["target_owner"] += int(owners[int(target)])
                counts["target_opportunity"] += 1
                counts["distractor_owner"] += sum(
                    int(owner) for index, owner in enumerate(owners) if index != int(target)
                )
                counts["distractor_opportunity"] += len(owners) - 1
                noun_joint_counts[group][(int(target), owner_position)] += 1
                control_rows["noun"].append(
                    {
                        "candidate_count": len(owners),
                        "target_position": int(target),
                        "owner_position": owner_position,
                        "target_is_owner": int(int(target) == owner_position),
                        **{
                            factor: values[factor]
                            for factor in (
                                "visibility",
                                "lag",
                                "sensor_stratum",
                                "dropout",
                                "noise",
                                "false_positive",
                                "repetition_phase",
                            )
                        },
                    }
                )
            else:
                noun_absent_owner_counts[group][owner_position] += 1
        elif key["grounded"]:
            concept = int(key["intended_concept"])
            target_primitive, target_manner = _components(concept)
            group = (
                concept,
                int(values["candidate_count"]),
                str(values["sensor_stratum"]),
            )
            action_target_positions[group][int(key["target_event_index"])] += 1
            for index, candidate in enumerate(map(int, key["event_concepts"])):
                if index == key["target_event_index"]:
                    continue
                primitive, manner = _components(candidate)
                relation = (
                    f"primitive_same={int(primitive == target_primitive)}|"
                    f"manner_same={int(manner == target_manner)}"
                )
                action_relation[relation] += 1
                action_relation_by_group[group][relation] += 1
                control_rows["action"].append(
                    {
                        "candidate_count": int(values["candidate_count"]),
                        "target_position": int(key["target_event_index"]),
                        "distractor_relation": relation,
                        **{
                            factor: values[factor]
                            for factor in (
                                "visibility",
                                "lag",
                                "sensor_stratum",
                                "dropout",
                                "noise",
                                "false_positive",
                                "repetition_phase",
                            )
                        },
                    }
                )
    factor_names = sorted(factors)
    pairwise = {
        f"{left}|{right}": _cramers_v(factors[left], factors[right])
        for index, left in enumerate(factor_names)
        for right in factor_names[index + 1 :]
    }
    noun_conditional = {}
    conditional_gaps = []
    noun_joint_audits = {}
    for group, counts in sorted(noun_owner_counts.items()):
        concept, count, sensor_stratum = group
        target_rate = counts["target_owner"] / counts["target_opportunity"]
        distractor_rate = counts["distractor_owner"] / counts["distractor_opportunity"]
        gap = abs(target_rate - distractor_rate)
        conditional_gaps.append(gap)
        group_name = (
            f"noun={concept}|candidate_count={count}|grounding=grounded|"
            f"sensor_stratum={sensor_stratum}"
        )
        joint = noun_joint_counts[group]
        cell_counts = [
            int(joint[(target_position, owner_position)])
            for target_position in range(count)
            for owner_position in range(count)
        ]
        exact_joint = bool(cell_counts) and len(set(cell_counts)) == 1 and min(cell_counts) > 0
        noun_conditional[group_name] = {
            "target_owner_rate": target_rate,
            "distractor_owner_rate_per_opportunity": distractor_rate,
            "absolute_gap": gap,
            "expected_independent_rate": 1.0 / count,
        }
        noun_joint_audits[group_name] = {
            "target_by_owner_cell_counts": {
                f"target={target}|owner={owner}": int(joint[(target, owner)])
                for target in range(count)
                for owner in range(count)
            },
            "all_cells_equal_and_positive": exact_joint,
        }
    noun_absent_audits = {}
    for group, counts in sorted(noun_absent_owner_counts.items()):
        concept, count, sensor_stratum = group
        group_name = (
            f"noun={concept}|candidate_count={count}|grounding=target_absent|"
            f"sensor_stratum={sensor_stratum}"
        )
        values_by_position = [int(counts[position]) for position in range(count)]
        noun_absent_audits[group_name] = {
            "owner_position_counts": {
                str(position): int(counts[position]) for position in range(count)
            },
            "all_owner_positions_equal_and_positive": bool(values_by_position)
            and len(set(values_by_position)) == 1
            and min(values_by_position) > 0,
        }
    action_group_audits = {}
    relation_names = (
        "primitive_same=1|manner_same=0",
        "primitive_same=0|manner_same=1",
        "primitive_same=0|manner_same=0",
    )
    for group, relations in sorted(action_relation_by_group.items()):
        concept, count, sensor_stratum = group
        targets = action_target_positions[group]
        target_counts = [int(targets[position]) for position in range(count)]
        first, second, third = (int(relations[name]) for name in relation_names)
        group_name = (
            f"action={concept}|candidate_count={count}|grounding=grounded|"
            f"sensor_stratum={sensor_stratum}"
        )
        action_group_audits[group_name] = {
            "target_position_counts": {
                str(position): int(targets[position]) for position in range(count)
            },
            "target_positions_exactly_balanced": bool(target_counts)
            and len(set(target_counts)) == 1
            and min(target_counts) > 0,
            "distractor_relation_counts": {
                name: int(relations[name]) for name in relation_names
            },
            "distractor_relations_exact_available_lattice_ratio_1_2_2": (
                first > 0 and second == 2 * first and third == 2 * first
            ),
        }
    control_correlations = {}
    nuisance_factors = (
        "visibility",
        "lag",
        "sensor_stratum",
        "dropout",
        "noise",
        "false_positive",
        "repetition_phase",
    )
    controls_by_family = {
        "noun": ("target_position", "owner_position", "target_is_owner"),
        "action": ("target_position", "distractor_relation"),
    }
    for family, rows in sorted(control_rows.items()):
        for count in sorted({int(row["candidate_count"]) for row in rows}):
            selected = [row for row in rows if int(row["candidate_count"]) == count]
            for control in controls_by_family[family]:
                for factor in nuisance_factors:
                    left = [row[control] for row in selected]
                    right = [row[factor] for row in selected]
                    if len(set(map(str, left))) < 2 or len(set(map(str, right))) < 2:
                        continue
                    control_correlations[
                        f"{family}|candidate_count={count}|{control}|{factor}"
                    ] = _cramers_v(left, right)
    maximum_noun_gap = max(conditional_gaps, default=1.0)
    maximum_v = max(pairwise.values(), default=0.0)
    maximum_control_v = max(control_correlations.values(), default=0.0)
    exact_noun = (
        bool(noun_joint_audits)
        and all(
            value["all_cells_equal_and_positive"]
            for value in noun_joint_audits.values()
        )
        and bool(noun_absent_audits)
        and all(
            value["all_owner_positions_equal_and_positive"]
            for value in noun_absent_audits.values()
        )
        and maximum_noun_gap == 0.0
    )
    exact_action = bool(action_group_audits) and all(
        value["target_positions_exactly_balanced"]
        and value["distractor_relations_exact_available_lattice_ratio_1_2_2"]
        for value in action_group_audits.values()
    )
    passed = (
        set(candidate_counts) == set(map(int, config["design"]["candidate_counts"]))
        and set(grounded) == {False, True}
        and exact_noun
        and maximum_noun_gap
        <= float(config["gates"]["maximum_noun_target_owner_gap"])
        and maximum_v <= float(config["gates"]["maximum_factor_pair_cramers_v"])
        and maximum_control_v
        <= float(config["gates"]["maximum_factor_control_cramers_v"])
        and exact_action
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "visible_oracle_key_ledgers_separate": True,
        "episode_count": len(visible),
        "candidate_count_distribution": dict(sorted(candidate_counts.items())),
        "grounded_distribution": {str(key).lower(): value for key, value in sorted(grounded.items())},
        "target_absent_episodes": grounded[False],
        "learnable_null_option": True,
        "noun_target_owner_conditional_by_candidate_count": noun_conditional,
        "noun_target_by_owner_joint_audits": noun_joint_audits,
        "noun_absent_owner_position_audits": noun_absent_audits,
        "maximum_conditional_noun_target_owner_gap": maximum_noun_gap,
        "noun_target_owner_independent": exact_noun,
        "action_relation_counts": dict(sorted(action_relation.items())),
        "action_structural_audits": action_group_audits,
        "all_action_relation_cells_present": len(action_relation) == 3,
        "action_relation_and_target_position_exact": exact_action,
        "factor_marginals": {
            name: dict(sorted(Counter(map(str, values)).items())) for name, values in factors.items()
        },
        "pairwise_cramers_v": pairwise,
        "maximum_pairwise_cramers_v": maximum_v,
        "factor_order_correlation_guard": maximum_v
        <= float(config["gates"]["maximum_factor_pair_cramers_v"]),
        "factor_control_cramers_v": control_correlations,
        "maximum_factor_control_cramers_v": maximum_control_v,
        "factor_control_correlation_guard": maximum_control_v
        <= float(config["gates"]["maximum_factor_control_cramers_v"]),
    }


def _shift_array(values: Sequence[Any], amount: int, fill: Any) -> list[Any]:
    data = list(values)
    if amount == 0:
        return data
    count = min(abs(amount), len(data))
    return ([fill] * count + data[: len(data) - count]) if amount > 0 else (data[count:] + [fill] * count)


def _shift_stream(stream: Mapping[str, Any], amount: int) -> dict[str, Any]:
    output = json.loads(json.dumps(stream))
    output["imu"] = _shift_array(stream["imu"], amount, [0.0] * 6)
    output["proprio"] = _shift_array(stream["proprio"], amount, [0.0] * 2)
    output["contact"] = _shift_array(stream["contact"], amount, [0.0])
    output["availability"] = _shift_array(stream["availability"], amount, [0] * 9)
    return output


def condition_view(
    visible: Sequence[Mapping[str, Any]],
    condition: str,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if condition not in set(config["design"]["conditions"]):
        raise ValueError(condition)
    rows = json.loads(json.dumps(list(visible)))
    donor_map: dict[str, Any] = {}
    if condition == "randomized_shuffle":
        groups: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            factors = row["factor_values"]
            groups[
                (
                    row["family"],
                    factors["sensor_stratum"],
                    factors["dropout"],
                    factors["noise"],
                    factors["false_positive"],
                )
            ].append(index)
        for group_key, indices in sorted(groups.items(), key=lambda item: str(item[0])):
            if len(indices) < 2:
                raise RuntimeError(f"unmatchable shuffle stratum {group_key}")
            ordered = sorted(
                indices,
                key=lambda index: hashlib.sha256(
                    f"v7-shuffle|{rows[index]['episode_id']}".encode()
                ).hexdigest(),
            )
            shift = 1 + int(hashlib.sha256(str(group_key).encode()).hexdigest(), 16) % (len(ordered) - 1)
            donors = ordered[shift:] + ordered[:shift]
            streams = [json.loads(json.dumps(rows[index]["raw_stream"])) for index in donors]
            donor_ids = [str(rows[index]["episode_id"]) for index in donors]
            for receiver, stream, donor_id in zip(ordered, streams, donor_ids):
                if donor_id == rows[receiver]["episode_id"]:
                    raise RuntimeError("shuffle failed to derange")
                rows[receiver]["raw_stream"] = stream
                donor_map[str(rows[receiver]["episode_id"])] = donor_id
    elif condition in {"shift_minus_8", "shift_plus_8"}:
        amount = -8 if condition == "shift_minus_8" else 8
        for row in rows:
            row["raw_stream"] = _shift_stream(row["raw_stream"], amount)
            donor_map[str(row["episode_id"])] = {"source": row["episode_id"], "shift": amount}
    elif condition == "uninformative":
        for row in rows:
            samples = len(row["raw_stream"]["timestamps"])
            row["raw_stream"] = {
                "timestamps": list(range(samples)),
                "imu": [[0.0] * 6 for _ in range(samples)],
                "proprio": [[0.0] * 2 for _ in range(samples)],
                "contact": [[0.0] for _ in range(samples)],
                "availability": [[1] * 9 for _ in range(samples)],
            }
            donor_map[str(row["episode_id"])] = "constant-zero-visible-stream"
    elif condition == "absent_channel":
        for row in rows:
            row.pop("raw_stream", None)
            donor_map[str(row["episode_id"])] = "channel-absent"
    elif condition == "sensor_corrupted":
        for row in rows:
            row["raw_stream"] = _shift_stream(row["raw_stream"], 13)
            donor_map[str(row["episode_id"])] = {"source": row["episode_id"], "shift": 13}
    else:
        donor_map = {str(row["episode_id"]): str(row["episode_id"]) for row in rows}
    for row in rows:
        reject_oracle_fields(row)
    return rows, donor_map


def condition_audit(
    base: Sequence[Mapping[str, Any]],
    conditioned: Mapping[str, Sequence[Mapping[str, Any]]],
    donor_maps: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    base_ids = [str(row["episode_id"]) for row in base]
    same_ids = all([str(row["episode_id"]) for row in rows] == base_ids for rows in conditioned.values())
    matched_shuffle = True
    base_by_id = {str(row["episode_id"]): row for row in base}
    for receiver, donor in donor_maps["randomized_shuffle"].items():
        left = base_by_id[receiver]["factor_values"]
        right = base_by_id[str(donor)]["factor_values"]
        matched_shuffle &= receiver != donor and all(
            left[name] == right[name]
            for name in ("sensor_stratum", "dropout", "noise", "false_positive")
        )
    uninformative_unique = {
        canonical_digest(row.get("raw_stream")) for row in conditioned["uninformative"]
    }
    passed = (
        same_ids
        and matched_shuffle
        and len(donor_maps["randomized_shuffle"]) == len(base)
        and len(uninformative_unique) == 1
        and all("raw_stream" not in row for row in conditioned["absent_channel"])
        and set(conditioned) == set(config["design"]["conditions"])
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "conditions": sorted(conditioned),
        "episode_identity_exactly_matched": same_ids,
        "randomized_shuffle_deranged_and_factor_matched": matched_shuffle,
        "randomized_shuffle_rows": len(donor_maps["randomized_shuffle"]),
        "time_shifts": {"shift_minus_8": -8, "shift_plus_8": 8},
        "uninformative_distinct_streams": len(uninformative_unique),
        "absent_channel_rows": sum("raw_stream" not in row for row in conditioned["absent_channel"]),
    }
