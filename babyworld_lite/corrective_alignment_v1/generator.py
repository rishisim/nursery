from __future__ import annotations

from collections import Counter, defaultdict
import copy
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
    digest = hashlib.sha256(f"corrective-v1|{label}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _opaque_id(seed: int, namespace: str, serial: int) -> str:
    digest = hashlib.sha256(
        f"corrective-opaque-id-v1|{seed}|{namespace}|{serial}".encode()
    ).hexdigest()
    return f"{namespace}-{digest[:20]}"


def _opaque_child_id(parent: str, namespace: str, serial: int) -> str:
    digest = hashlib.sha256(
        f"corrective-opaque-child-v1|{parent}|{namespace}|{serial}".encode()
    ).hexdigest()
    return f"{namespace}-{digest[:20]}"


def _normalized(values: Sequence[float]) -> list[float]:
    array = np.maximum(np.asarray(values, dtype=np.float64), 1e-9)
    return list(map(float, array / array.sum()))


def _noisy_observation(
    prototype: Sequence[float],
    visibility: float,
    rng: np.random.Generator,
) -> list[float]:
    dimension = len(prototype)
    base = np.full(dimension, (1.0 - visibility) / dimension, dtype=float)
    base += visibility * np.asarray(prototype, dtype=float)
    noise = rng.dirichlet(np.ones(dimension) * 2.0)
    return _normalized(0.94 * base + 0.06 * noise)


def _action_geometry(
    corpus_seed: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, list[list[float]]], dict[str, Any]]:
    design = config["design"]
    geometry_config = design["action_geometry"]
    minimum_margin = float(geometry_config["minimum_self_cross_dot_margin"])
    maximum_attempts = int(geometry_config["maximum_generation_attempts"])
    mass_low, mass_high = map(
        float, geometry_config["off_diagonal_mass_range"]
    )
    prototypes: dict[str, list[list[float]]] = {}
    slot_audits: dict[str, Any] = {}
    for slot, dimension in (
        ("primitive", int(design["primitive_concepts"])),
        ("manner", int(design["manner_concepts"])),
    ):
        rng = _rng(
            corpus_seed,
            f"{geometry_config['rng_namespace']}|{slot}",
        )
        accepted = None
        for attempt in range(maximum_attempts):
            matrix = np.zeros((dimension, dimension), dtype=float)
            for concept in range(dimension):
                off_mass = float(rng.uniform(mass_low, mass_high))
                off = rng.dirichlet(np.ones(dimension - 1) * 1.5) * off_mass
                matrix[concept, concept] = 1.0 - off_mass
                matrix[concept, np.arange(dimension) != concept] = off
            permutation = rng.permutation(dimension)
            matrix = matrix[:, permutation]
            gram = matrix @ matrix.T
            margins = [
                float(gram[index, index] - np.max(np.delete(gram[index], index)))
                for index in range(dimension)
            ]
            if min(margins) >= minimum_margin:
                accepted = (matrix, gram, margins, attempt + 1)
                break
        if accepted is None:
            raise RuntimeError(f"could not construct separated {slot} action geometry")
        matrix, gram, margins, attempts = accepted
        distances = sorted(
            float(np.linalg.norm(matrix[left] - matrix[right]))
            for left in range(dimension)
            for right in range(left + 1, dimension)
        )
        prototypes[slot] = [list(map(float, row)) for row in matrix]
        slot_audits[slot] = {
            "gram_matrix": [list(map(float, row)) for row in gram],
            "sorted_pairwise_distances": distances,
            "minimum_self_cross_dot_margin": min(margins),
            "mean_pairwise_distance": float(np.mean(distances)),
            "pairwise_distance_sd": float(np.std(distances, ddof=1)),
            "row_sums": list(map(float, matrix.sum(axis=1))),
            "minimum_entry": float(matrix.min()),
            "generation_attempts": int(attempts),
        }
    audit = {
        "schema_version": "nursery-corrective-action-geometry-audit-v1",
        "prototype_digest": canonical_digest(prototypes),
        "action_geometry_digest": canonical_digest(
            {slot: value["gram_matrix"] for slot, value in slot_audits.items()}
        ),
        "slots": slot_audits,
        "all_rows_normalized": all(
            abs(value - 1.0) <= 1e-12
            for row in slot_audits.values()
            for value in row["row_sums"]
        ),
        "all_entries_nonnegative": all(
            row["minimum_entry"] >= 0.0 for row in slot_audits.values()
        ),
        "all_margins_pass": all(
            row["minimum_self_cross_dot_margin"] >= minimum_margin
            for row in slot_audits.values()
        ),
    }
    return prototypes, audit


def _lexicon(seed: int, primitives: int, manners: int) -> dict[str, Any]:
    primitive_forms = ["dax", "kiv", "mep", "zot", "wug", "nib"][:primitives]
    manner_forms = ["luma", "tavi", "seno", "raku", "pima"][:manners]
    rng = _rng(seed, "opaque-lexicon")
    rng.shuffle(primitive_forms)
    rng.shuffle(manner_forms)
    return {
        "primitive": primitive_forms,
        "manner": manner_forms,
        "oracle": {
            **{f"primitive|{token}": index for index, token in enumerate(primitive_forms)},
            **{f"manner|{token}": index for index, token in enumerate(manner_forms)},
        },
    }


def _heldout_compositions(seed: int, primitives: int, manners: int) -> list[tuple[int, int]]:
    rng = _rng(seed, "heldout-compositions")
    while True:
        manner_choices = list(map(int, rng.integers(0, manners, size=primitives)))
        if len(set(manner_choices)) > 1:
            return list(zip(range(primitives), manner_choices))


def _foil_map(
    seed: int,
    compositions: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, int]]:
    output = {}
    for value in compositions:
        eligible = [
            candidate
            for candidate in compositions
            if candidate[0] != value[0] and candidate[1] != value[1]
        ]
        if not eligible:
            raise RuntimeError("training-only foil set is empty")
        ordered = sorted(
            eligible,
            key=lambda candidate: hashlib.sha256(
                f"foil|{seed}|{value}|{candidate}".encode()
            ).hexdigest(),
        )
        output[value] = ordered[0]
    return output


def _event_intervals(
    count: int,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    design = config["design"]
    widths = rng.integers(
        int(design["event_width"][0]),
        int(design["event_width"][1]) + 1,
        size=count,
    )
    gaps = rng.integers(
        int(design["event_gap"][0]),
        int(design["event_gap"][1]) + 1,
        size=count + 1,
    )
    cursor = int(gaps[0])
    output = []
    for width, gap in zip(widths, gaps[1:]):
        end = cursor + int(width) - 1
        output.append((cursor, end))
        cursor = end + 1 + int(gap)
    samples = int(design["raw_sensor_samples"])
    if output[-1][1] >= samples - 2:
        scale = (samples - 4) / max(1, output[-1][1])
        output = [
            (max(1, int(start * scale)), max(2, int(end * scale)))
            for start, end in output
        ]
    return output


def _outside_indices(samples: int, intervals: Sequence[tuple[int, int]]) -> list[int]:
    occupied = {
        index
        for start, end in intervals
        for index in range(max(0, start), min(samples - 1, end) + 1)
    }
    return [index for index in range(samples) if index not in occupied]


def _make_side_stream(
    *,
    samples: int,
    intervals: Sequence[tuple[int, int]],
    target_index: int | None,
    informativity: float,
    noise: float,
    peak_range: Sequence[float],
    rng: np.random.Generator,
) -> list[float]:
    stream = np.abs(rng.normal(0.0, noise, size=samples))
    correct = bool(rng.random() < informativity)
    outside = _outside_indices(samples, intervals)
    if target_index is not None and correct:
        start, end = intervals[int(target_index)]
        positions = list(range(start, end + 1))
    elif target_index is None and correct and outside:
        center = int(rng.choice(outside))
        positions = [index for index in range(center - 2, center + 3) if index in outside]
    elif target_index is None:
        start, end = intervals[int(rng.integers(0, len(intervals)))]
        positions = list(range(start, end + 1))
    else:
        alternatives = [index for index in range(len(intervals)) if index != target_index]
        if alternatives:
            start, end = intervals[int(rng.choice(alternatives))]
            positions = list(range(start, end + 1))
        else:
            positions = outside[:5]
    amplitude = float(rng.uniform(float(peak_range[0]), float(peak_range[1])))
    if positions:
        center = float(np.mean(positions))
        for index in positions:
            stream[index] += amplitude * math.exp(-0.5 * ((index - center) / 2.2) ** 2)
    return list(map(float, stream))


def detect_event_or_null(
    stream: Sequence[float] | None,
    intervals: Sequence[Sequence[int]],
) -> dict[str, Any]:
    count = len(intervals)
    if stream is None:
        return {
            "event_logits": [0.0] * count,
            "null_logit": 0.0,
            "detector_input_present": False,
        }
    array = np.asarray(stream, dtype=np.float64)
    event_energy = [
        float(np.max(array[max(0, int(start)) : min(len(array), int(end) + 1)]))
        for start, end in intervals
    ]
    outside = _outside_indices(
        len(array), [(int(start), int(end)) for start, end in intervals]
    )
    null_energy = float(np.max(array[outside])) if outside else float(np.median(array))
    all_values = np.asarray([*event_energy, null_energy], dtype=np.float64)
    center = float(np.median(all_values))
    scale = float(np.std(all_values))
    if scale < 1e-8:
        logits = np.zeros_like(all_values)
    else:
        logits = np.clip((all_values - center) / scale, -4.0, 4.0)
    return {
        "event_logits": list(map(float, logits[:-1])),
        "null_logit": float(logits[-1]),
        "detector_input_present": True,
    }


def _shift_stream(values: Sequence[float], amount: int) -> list[float]:
    source = list(map(float, values))
    if amount == 0:
        return source
    count = min(abs(int(amount)), len(source))
    if amount > 0:
        return [0.0] * count + source[: len(source) - count]
    return source[count:] + [0.0] * count


def _corrupt_evidence(
    synchronized: Mapping[str, Any],
    target_index: int | None,
) -> dict[str, Any]:
    events = list(map(float, synchronized["event_logits"]))
    values = [*events, float(synchronized["null_logit"])]
    correct = len(events) if target_index is None else int(target_index)
    wrong = 0 if target_index is None else (int(target_index) + 1) % len(events)
    ranked = sorted(values, reverse=True)
    corrupted = [0.0] * len(values)
    corrupted[wrong] = ranked[0] + 1e-6
    remaining_positions = [index for index in range(len(values)) if index != wrong]
    for rank, (position, value) in enumerate(zip(remaining_positions, ranked[1:]), start=1):
        corrupted[position] = float(value) - rank * 1e-6
    if int(np.argmax(corrupted)) == correct:
        raise RuntimeError("corrupted detector did not force a wrong state")
    return {
        "event_logits": corrupted[:-1],
        "null_logit": corrupted[-1],
        "detector_input_present": True,
        "corruption": "deterministic_unique_wrong_state_rank_permutation",
    }


def _condition_evidence(
    visible: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    corpus_seed: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    keys = {str(row["episode_id"]): row for row in oracle}
    synchronized: dict[str, dict[str, Any]] = {}
    for row in visible:
        synchronized[str(row["episode_id"])] = detect_event_or_null(
            row["side_stream"],
            [(event["start"], event["end"]) for event in row["events"]],
        )
    groups: defaultdict[tuple[int], list[str]] = defaultdict(list)
    for row in visible:
        groups[(len(row["events"]),)].append(str(row["episode_id"]))
    donor_map: dict[str, str] = {}
    for group, identifiers in sorted(groups.items()):
        ordered = sorted(
            identifiers,
            key=lambda identifier: hashlib.sha256(
                f"shuffle|{corpus_seed}|{group}|{identifier}".encode()
            ).hexdigest(),
        )
        adjacency = {
            receiver: sorted(
                [
                    donor
                    for donor in ordered
                    if donor != receiver
                    and tuple(keys[donor]["composition"])
                    != tuple(keys[receiver]["composition"])
                ],
                key=lambda donor: hashlib.sha256(
                    f"shuffle-edge|{corpus_seed}|{receiver}|{donor}".encode()
                ).hexdigest(),
            )
            for receiver in ordered
        }
        donor_to_receiver: dict[str, str] = {}

        def assign(receiver: str, seen: set[str]) -> bool:
            for donor in adjacency[receiver]:
                if donor in seen:
                    continue
                seen.add(donor)
                if donor not in donor_to_receiver or assign(
                    donor_to_receiver[donor], seen
                ):
                    donor_to_receiver[donor] = receiver
                    return True
            return False

        receiver_order = sorted(ordered, key=lambda value: (len(adjacency[value]), value))
        if not all(assign(receiver, set()) for receiver in receiver_order):
            raise RuntimeError(f"no exact different-composition shuffle matching: {group}")
        group_map = {
            receiver: donor for donor, receiver in donor_to_receiver.items()
        }
        if set(group_map) != set(ordered) or len(set(group_map.values())) != len(ordered):
            raise RuntimeError(f"shuffle matching is not a bijection: {group}")
        donor_map.update(group_map)
    shift = int(config["design"]["signed_shift_samples"])
    output = {condition: {} for condition in config["design"]["conditions"]}
    by_id = {str(row["episode_id"]): row for row in visible}
    for episode_id, row in by_id.items():
        intervals = [(event["start"], event["end"]) for event in row["events"]]
        key = keys[episode_id]
        output["synchronized"][episode_id] = synchronized[episode_id]
        output["shuffled"][episode_id] = copy.deepcopy(
            synchronized[donor_map[episode_id]]
        )
        output["shift_minus"][episode_id] = detect_event_or_null(
            _shift_stream(row["side_stream"], -shift), intervals
        )
        output["shift_plus"][episode_id] = detect_event_or_null(
            _shift_stream(row["side_stream"], shift), intervals
        )
        output["absent"][episode_id] = detect_event_or_null(None, intervals)
        output["uninformative"][episode_id] = detect_event_or_null(
            [0.0] * len(row["side_stream"]), intervals
        )
        output["corrupted"][episode_id] = _corrupt_evidence(
            synchronized[episode_id], key["target_event_index"]
        )
        target = key["target_event_index"]
        output["oracle_alignment"][episode_id] = {
            "event_logits": [
                6.0 if target is not None and index == int(target) else -6.0
                for index in range(len(row["events"]))
            ],
            "null_logit": 6.0 if target is None else -6.0,
            "detector_input_present": False,
            "positive_control": True,
        }
        output["exact_window"][episode_id] = {
            "event_logits": [0.0] * len(row["events"]),
            "null_logit": -12.0,
            "detector_input_present": False,
            "exact_window": True,
        }
    synchronized_by_count = {
        str(count): sorted(
            canonical_digest(synchronized[identifier]) for identifier in identifiers
        )
        for (count,), identifiers in sorted(groups.items())
    }
    shuffled_by_count = {
        str(count): sorted(
            canonical_digest(output["shuffled"][identifier])
            for identifier in identifiers
        )
        for (count,), identifiers in sorted(groups.items())
    }
    detector_manipulation = {}
    for condition, rows in output.items():
        correct = 0
        log_losses = []
        by_state = {
            "present": {"correct": 0, "log_losses": []},
            "null": {"correct": 0, "log_losses": []},
        }
        for episode_id, detector in rows.items():
            values = np.asarray(
                [*detector["event_logits"], detector["null_logit"]], dtype=float
            )
            expected = (
                len(values) - 1
                if keys[episode_id]["target_event_index"] is None
                else int(keys[episode_id]["target_event_index"])
            )
            maximum = float(values.max())
            tied = np.flatnonzero(np.abs(values - maximum) <= 1e-12)
            is_correct = int(len(tied) == 1 and int(tied[0]) == expected)
            correct += is_correct
            centered = values - maximum
            probabilities = np.exp(np.clip(centered, -60.0, 60.0))
            probabilities /= probabilities.sum()
            loss = -math.log(max(float(probabilities[expected]), 1e-15))
            log_losses.append(loss)
            state = "null" if keys[episode_id]["target_event_index"] is None else "present"
            by_state[state]["correct"] += is_correct
            by_state[state]["log_losses"].append(loss)
        detector_manipulation[condition] = {
            "strict_event_or_null_accuracy": correct / len(rows),
            "multiclass_log_loss": float(np.mean(log_losses)),
            "episode_count": len(rows),
            "by_state": {
                state: {
                    "strict_event_or_null_accuracy": values["correct"]
                    / len(values["log_losses"]),
                    "multiclass_log_loss": float(np.mean(values["log_losses"])),
                    "episode_count": len(values["log_losses"]),
                }
                for state, values in by_state.items()
            },
        }
    audit = {
        "language_visual_records_identical": True,
        "condition_count": len(output),
        "condition_names": list(output),
        "shuffle_exact_bijection": len(set(donor_map.values())) == len(donor_map),
        "shuffle_blocking": "candidate_event_count",
        "shuffle_preserves_learner_visible_evidence_marginal_by_block": (
            synchronized_by_count == shuffled_by_count
        ),
        "shuffle_no_self_donor": all(key != value for key, value in donor_map.items()),
        "shuffle_no_same_composition": all(
            tuple(keys[receiver]["composition"])
            != tuple(keys[donor]["composition"])
            for receiver, donor in donor_map.items()
        ),
        "shift_non_circular": True,
        "signed_shift_samples": shift,
        "uninformative_constant_zero": True,
        "donor_map_digest": canonical_digest(donor_map),
        "synchronized_evidence_marginal_digest_by_block": canonical_digest(
            synchronized_by_count
        ),
        "shuffled_evidence_marginal_digest_by_block": canonical_digest(
            shuffled_by_count
        ),
        "detector_manipulation": detector_manipulation,
        "corrupted_detector_strict_accuracy_zero": (
            detector_manipulation["corrupted"]["strict_event_or_null_accuracy"] == 0.0
        ),
    }
    return output, {"audit": audit, "donor_map": donor_map}


def _tokens(lexicon: Mapping[str, Any], composition: tuple[int, int]) -> list[dict[str, str]]:
    return [
        {"slot": "primitive", "token": str(lexicon["primitive"][composition[0]])},
        {"slot": "manner", "token": str(lexicon["manner"][composition[1]])},
    ]


def _side_only_probe(
    visible: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    synchronized: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    keys = {str(row["episode_id"]): row for row in oracle}
    features = []
    primitive_labels = []
    manner_labels = []
    maximum_events = int(config["design"]["candidate_event_count"][1])
    for row in visible:
        episode_id = str(row["episode_id"])
        detector = synchronized[episode_id]
        event_logits = list(map(float, detector["event_logits"]))
        padded = event_logits + [-5.0] * (maximum_events - len(event_logits))
        features.append(
            [
                *padded,
                float(detector["null_logit"]),
                len(event_logits) / maximum_events,
                float(row["utterance"]["speech_time"])
                / float(config["design"]["raw_sensor_samples"]),
            ]
        )
        primitive_labels.append(int(keys[episode_id]["composition"][0]))
        manner_labels.append(int(keys[episode_id]["composition"][1]))
    matrix = np.asarray(features, dtype=float)

    def leave_one_out_accuracy(labels: Sequence[int], class_count: int) -> float:
        labels_array = np.asarray(labels, dtype=int)
        correct = 0
        for heldout in range(len(matrix)):
            retained = np.arange(len(matrix)) != heldout
            training = matrix[retained]
            mean = training.mean(axis=0)
            scale = training.std(axis=0)
            scale[scale < 1e-8] = 1.0
            standardized = (training - mean) / scale
            target = (matrix[heldout] - mean) / scale
            centroids = []
            for label in range(class_count):
                members = standardized[labels_array[retained] == label]
                if not len(members):
                    raise RuntimeError("side-only probe class absent")
                centroids.append(members.mean(axis=0))
            distances = [
                float(np.sum((target - centroid) ** 2)) for centroid in centroids
            ]
            correct += int(int(np.argmin(distances)) == int(labels_array[heldout]))
        return correct / len(matrix)

    return {
        "schema_version": "nursery-corrective-side-only-probe-v1",
        "features": "detector_logits_event_count_and_speech_time_only",
        "event_observations_consumed": False,
        "tokens_consumed": False,
        "identifiers_consumed": False,
        "cross_validation": "leave_one_episode_out_nearest_centroid",
        "primitive_accuracy": leave_one_out_accuracy(
            primitive_labels, int(config["design"]["primitive_concepts"])
        ),
        "primitive_chance": 1.0 / int(config["design"]["primitive_concepts"]),
        "manner_accuracy": leave_one_out_accuracy(
            manner_labels, int(config["design"]["manner_concepts"])
        ),
        "manner_chance": 1.0 / int(config["design"]["manner_concepts"]),
        "episode_count": len(matrix),
    }


def _evaluation(
    corpus_seed: int,
    config: Mapping[str, Any],
    lexicon: Mapping[str, Any],
    heldout: Sequence[tuple[int, int]],
    action_prototypes: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    design = config["design"]
    primitives = int(design["primitive_concepts"])
    manners = int(design["manner_concepts"])
    visible: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    provenance: dict[str, list[str]] = defaultdict(list)
    prompt_serial = 0

    def next_prompt_id(kind: str) -> str:
        nonlocal prompt_serial
        prompt_id = _opaque_id(corpus_seed, f"eval-{kind}", prompt_serial)
        prompt_serial += 1
        return prompt_id

    lexical_rng = _rng(corpus_seed, design["evaluation"]["lexical_eval_rng_namespace"])
    lexical_count = int(design["evaluation"]["held_out_instances_per_lexical_concept"])
    for slot, dimension in (("primitive", primitives), ("manner", manners)):
        for concept in range(dimension):
            token = str(lexicon[slot][concept])
            for repetition in range(lexical_count):
                candidate_pairs = [
                    (
                        index,
                        _noisy_observation(
                            action_prototypes[slot][index], 0.87, lexical_rng
                        ),
                    )
                    for index in range(dimension)
                ]
                lexical_rng.shuffle(candidate_pairs)
                prompt_id = next_prompt_id("lexical")
                candidates = [
                    {
                        "candidate_id": _opaque_child_id(prompt_id, "candidate", index),
                        "observation": observation,
                    }
                    for index, (_concept, observation) in enumerate(candidate_pairs)
                ]
                answer = [value for value, _ in candidate_pairs].index(concept)
                visible.append(
                    {
                        "schema_version": "nursery-corrective-eval-visible-v1",
                        "prompt_id": prompt_id,
                        "kind": "lexical",
                        "slot": slot,
                        "tokens": [{"slot": slot, "token": token}],
                        "candidates": candidates,
                        "generator_provenance": "lexical-eval-v1",
                    }
                )
                keys.append(
                    {
                        "prompt_id": prompt_id,
                        "kind": "lexical",
                        "answer_index": answer,
                        "concept_group": f"{slot}-{concept}",
                        "exact_chance": 1.0 / dimension,
                    }
                )
                provenance["lexical"].append(prompt_id)
    composition_rng = _rng(
        corpus_seed, design["evaluation"]["composition_eval_rng_namespace"]
    )
    comp_count = int(design["evaluation"]["held_out_instances_per_composition"])
    all_compositions = [
        (primitive, manner)
        for primitive in range(primitives)
        for manner in range(manners)
    ]
    for composition in heldout:
        for repetition in range(comp_count):
            candidate_pairs = [
                (
                    candidate,
                    _noisy_observation(
                        action_prototypes["primitive"][candidate[0]],
                        0.80,
                        composition_rng,
                    ),
                    _noisy_observation(
                        action_prototypes["manner"][candidate[1]],
                        0.80,
                        composition_rng,
                    ),
                )
                for candidate in all_compositions
            ]
            composition_rng.shuffle(candidate_pairs)
            prompt_id = next_prompt_id("composition")
            candidates = [
                {
                    "candidate_id": _opaque_child_id(prompt_id, "candidate", index),
                    "primitive_observation": primitive_observation,
                    "manner_observation": manner_observation,
                }
                for index, (
                    _candidate,
                    primitive_observation,
                    manner_observation,
                ) in enumerate(candidate_pairs)
            ]
            answer = [value for value, _, _ in candidate_pairs].index(
                tuple(composition)
            )
            visible.append(
                {
                    "schema_version": "nursery-corrective-eval-visible-v1",
                    "prompt_id": prompt_id,
                    "kind": "composition",
                    "tokens": _tokens(lexicon, tuple(composition)),
                    "candidates": candidates,
                    "generator_provenance": "composition-eval-v1",
                }
            )
            keys.append(
                {
                    "prompt_id": prompt_id,
                    "kind": "composition",
                    "answer_index": answer,
                    "concept_group": f"composition-{composition[0]}-{composition[1]}",
                    "exact_chance": 1.0 / len(all_compositions),
                }
            )
            provenance["composition"].append(prompt_id)
    presence_rng = _rng(corpus_seed, design["evaluation"]["presence_eval_rng_namespace"])
    presence_per = int(design["evaluation"]["presence_bags_per_composition"])
    bag_count = int(design["evaluation"]["presence_bag_candidates"])
    for composition in heldout:
        for repetition in range(presence_per):
            present = repetition % 2 == 0
            candidates = [tuple(composition)] if present else []
            pool = [value for value in all_compositions if value != tuple(composition)]
            presence_rng.shuffle(pool)
            candidates.extend(pool[: bag_count - len(candidates)])
            presence_rng.shuffle(candidates)
            answer = candidates.index(tuple(composition)) if present else len(candidates)
            prompt_id = next_prompt_id("presence")
            event_rows = [
                {
                    "candidate_id": _opaque_child_id(prompt_id, "candidate", index),
                    "primitive_observation": _noisy_observation(
                        action_prototypes["primitive"][value[0]],
                        0.78,
                        presence_rng,
                    ),
                    "manner_observation": _noisy_observation(
                        action_prototypes["manner"][value[1]],
                        0.78,
                        presence_rng,
                    ),
                }
                for index, value in enumerate(candidates)
            ]
            visible.append(
                {
                    "schema_version": "nursery-corrective-eval-visible-v1",
                    "prompt_id": prompt_id,
                    "kind": "presence",
                    "tokens": _tokens(lexicon, tuple(composition)),
                    "candidates": event_rows,
                    "null_option": True,
                    "generator_provenance": "presence-eval-v1",
                }
            )
            keys.append(
                {
                    "prompt_id": prompt_id,
                    "kind": "presence",
                    "answer_index": answer,
                    "presence": "present" if present else "null",
                    "concept_group": f"presence-{composition[0]}-{composition[1]}",
                    "exact_chance": 1.0 / (len(candidates) + 1),
                }
            )
            provenance["presence"].append(prompt_id)
    zero_rng = _rng(corpus_seed, "zero-exposure-eval-v1")
    for index in range(int(design["evaluation"]["zero_exposure_tokens"])):
        target = int(zero_rng.integers(0, primitives))
        candidate_pairs = [
            (
                candidate,
                _noisy_observation(
                    action_prototypes["primitive"][candidate], 0.85, zero_rng
                ),
            )
            for candidate in range(primitives)
        ]
        zero_rng.shuffle(candidate_pairs)
        prompt_id = next_prompt_id("zero")
        candidates = [
            {
                "candidate_id": _opaque_child_id(prompt_id, "candidate", candidate_index),
                "observation": observation,
            }
            for candidate_index, (_candidate, observation) in enumerate(candidate_pairs)
        ]
        visible.append(
            {
                "schema_version": "nursery-corrective-eval-visible-v1",
                "prompt_id": prompt_id,
                "kind": "zero_exposure",
                "slot": "primitive",
                "tokens": [
                    {
                        "slot": "primitive",
                        "token": _opaque_child_id(prompt_id, "unseen", 0),
                    }
                ],
                "candidates": candidates,
                "generator_provenance": "zero-exposure-eval-v1",
            }
        )
        keys.append(
            {
                "prompt_id": prompt_id,
                "kind": "zero_exposure",
                "answer_index": [value for value, _ in candidate_pairs].index(target),
                "concept_group": f"zero-{index}",
                "exact_chance": 1.0 / primitives,
            }
        )
        provenance["zero_exposure"].append(prompt_id)
    return visible, keys, {
        "train_namespace": design["evaluation"]["train_rng_namespace"],
        "evaluation_namespaces": [
            design["evaluation"]["lexical_eval_rng_namespace"],
            design["evaluation"]["composition_eval_rng_namespace"],
            design["evaluation"]["presence_eval_rng_namespace"],
        ],
        "prompt_ids_digest_by_kind": {
            key: canonical_digest(sorted(value)) for key, value in sorted(provenance.items())
        },
        "evaluation_prompt_ids": sorted(
            prompt_id for values in provenance.values() for prompt_id in values
        ),
        "evaluation_action_prototype_digest": canonical_digest(
            action_prototypes
        ),
    }


def generate_corpus(
    corpus_seed: int,
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
) -> dict[str, Any]:
    firewall.authorize(
        "generate",
        [IdentifierReference("corpus", int(corpus_seed))],
        unit_id=f"corpus-{corpus_seed}",
        local_sequence=len(firewall.operations),
    )
    design = config["design"]
    primitives = int(design["primitive_concepts"])
    manners = int(design["manner_concepts"])
    all_compositions = [
        (primitive, manner)
        for primitive in range(primitives)
        for manner in range(manners)
    ]
    heldout = _heldout_compositions(corpus_seed, primitives, manners)
    training = [value for value in all_compositions if value not in set(heldout)]
    if {value[0] for value in training} != set(range(primitives)) or {
        value[1] for value in training
    } != set(range(manners)):
        raise RuntimeError("heldout split removed a constituent")
    lexicon = _lexicon(corpus_seed, primitives, manners)
    action_prototypes, action_geometry_audit = _action_geometry(
        corpus_seed,
        config,
    )
    foil = _foil_map(corpus_seed, training)
    factor_rng = _rng(corpus_seed, "corpus-factors")
    stratum = str(factor_rng.choice(design["ambiguity_strata"]))
    grounded_rate = float(factor_rng.uniform(*map(float, design["grounded_rate_range"])))
    foil_rate = float(
        factor_rng.uniform(*map(float, design["foil_rate_by_stratum"][stratum]))
    )
    visibility = float(factor_rng.uniform(*map(float, design["visibility_range"])))
    lag_mean = float(factor_rng.uniform(*map(float, design["lag_mean_range"])))
    lag_sd = float(factor_rng.uniform(*map(float, design["lag_sd_range"])))
    side_informativity = float(
        factor_rng.uniform(*map(float, design["side_informativeness_range"]))
    )
    side_noise = float(factor_rng.uniform(*map(float, design["side_noise_range"])))
    repetitions_low, repetitions_high = map(
        int, design["training_repetitions_per_composition"]
    )
    visible: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    repetition_counts: dict[str, int] = {}
    episode_serial = 0
    train_rng = _rng(corpus_seed, design["evaluation"]["train_rng_namespace"])
    for composition in training:
        repetitions = int(train_rng.integers(repetitions_low, repetitions_high + 1))
        repetition_counts[f"{composition[0]}|{composition[1]}"] = repetitions
        for repetition in range(repetitions):
            episode_rng = _rng(
                corpus_seed,
                f"episode-{composition[0]}-{composition[1]}-{repetition}",
            )
            count_low, count_high = map(int, design["candidate_event_count"])
            count_span = count_high - count_low + 1
            composition_serial = all_compositions.index(tuple(composition))
            count_offset = int(
                _rng(corpus_seed, "candidate-count-offset").integers(0, count_span)
            )
            count = count_low + (
                repetition + composition_serial + count_offset
            ) % count_span
            grounded = bool(episode_rng.random() < grounded_rate)
            candidates: list[tuple[int, int]] = [tuple(composition)] if grounded else []
            if episode_rng.random() < foil_rate and foil[tuple(composition)] not in candidates:
                candidates.append(foil[tuple(composition)])
            pool = [value for value in training if value not in candidates]
            episode_rng.shuffle(pool)
            candidates.extend(pool[: count - len(candidates)])
            if len(candidates) < count:
                raise RuntimeError("candidate construction underflow")
            episode_rng.shuffle(candidates)
            target_index = candidates.index(tuple(composition)) if grounded else None
            intervals = _event_intervals(count, config, episode_rng)
            episode_id = _opaque_id(corpus_seed, "train", episode_serial)
            episode_serial += 1
            events = []
            for index, (candidate, interval) in enumerate(zip(candidates, intervals)):
                events.append(
                    {
                        "event_id": _opaque_child_id(episode_id, "event", index),
                        "start": int(interval[0]),
                        "end": int(interval[1]),
                        "primitive_observation": _noisy_observation(
                            action_prototypes["primitive"][candidate[0]],
                            visibility,
                            episode_rng,
                        ),
                        "manner_observation": _noisy_observation(
                            action_prototypes["manner"][candidate[1]],
                            visibility,
                            episode_rng,
                        ),
                    }
                )
            samples = int(design["raw_sensor_samples"])
            peak_range = (
                design["grounded_sensor_peak_range"]
                if target_index is not None
                else design["null_sensor_peak_range"]
            )
            stream = _make_side_stream(
                samples=samples,
                intervals=intervals,
                target_index=target_index,
                informativity=side_informativity,
                noise=side_noise,
                peak_range=peak_range,
                rng=episode_rng,
            )
            if target_index is None:
                anchor = float(np.mean([(start + end) / 2 for start, end in intervals]))
            else:
                start, end = intervals[int(target_index)]
                anchor = (start + end) / 2
            lag = float(episode_rng.normal(lag_mean, lag_sd))
            visible.append(
                {
                    "schema_version": "nursery-corrective-train-visible-v1",
                    "episode_id": episode_id,
                    "utterance": {
                        "items": _tokens(lexicon, tuple(composition)),
                        "speech_time": anchor + lag,
                    },
                    "events": events,
                    "side_stream": stream,
                }
            )
            oracle.append(
                {
                    "episode_id": episode_id,
                    "composition": list(composition),
                    "candidate_compositions": [list(value) for value in candidates],
                    "target_event_index": target_index,
                    "grounded": grounded,
                    "lag": lag,
                }
            )
    conditions, condition_metadata = _condition_evidence(
        visible, oracle, config, corpus_seed
    )
    firewall.authorize(
        "condition",
        [IdentifierReference("corpus", int(corpus_seed))],
        unit_id=f"corpus-{corpus_seed}",
        local_sequence=len(firewall.operations),
    )
    training_visible = []
    for row in visible:
        copied = copy.deepcopy(row)
        copied.pop("side_stream")
        training_visible.append(copied)
    eval_visible, eval_keys, provenance = _evaluation(
        corpus_seed,
        config,
        lexicon,
        heldout,
        action_prototypes,
    )
    training_instance_ids = sorted(str(row["episode_id"]) for row in training_visible)
    evaluation_instance_ids = list(map(str, provenance["evaluation_prompt_ids"]))
    provenance["training_instance_ids_digest"] = canonical_digest(
        training_instance_ids
    )
    provenance["evaluation_instance_ids_digest"] = canonical_digest(
        evaluation_instance_ids
    )
    provenance["train_evaluation_instance_id_overlap"] = len(
        set(training_instance_ids) & set(evaluation_instance_ids)
    )
    train_namespace = str(design["evaluation"]["train_rng_namespace"])
    evaluation_namespaces = list(map(str, provenance["evaluation_namespaces"]))
    generator_namespaces_disjoint = (
        train_namespace not in evaluation_namespaces
        and len(evaluation_namespaces) == len(set(evaluation_namespaces))
    )
    evaluation_forbidden = set(config["firewalls"]["forbidden_visible_fields"])
    for row in eval_visible:
        reject_forbidden_fields(row, evaluation_forbidden, path="evaluation")
    evaluation_schema_exact = all(
        set(row)
        == (
            {
                "schema_version",
                "prompt_id",
                "kind",
                "tokens",
                "candidates",
                "generator_provenance",
                "slot",
            }
            if row["kind"] in {"lexical", "zero_exposure"}
            else {
                "schema_version",
                "prompt_id",
                "kind",
                "tokens",
                "candidates",
                "generator_provenance",
                "null_option",
            }
            if row["kind"] == "presence"
            else {
                "schema_version",
                "prompt_id",
                "kind",
                "tokens",
                "candidates",
                "generator_provenance",
            }
        )
        for row in eval_visible
    )
    episode_geometry = [
        {
            "intervals": [[event["start"], event["end"]] for event in row["events"]],
            "candidate_compositions": key["candidate_compositions"],
            "target_event_index": key["target_event_index"],
        }
        for row, key in zip(visible, oracle)
    ]
    target_counts = Counter(key["target_event_index"] is not None for key in oracle)
    candidate_count_histogram = dict(
        sorted(Counter(len(key["candidate_compositions"]) for key in oracle).items())
    )
    heldout_set = set(map(tuple, heldout))
    heldout_candidate_event_count = sum(
        tuple(candidate) in heldout_set
        for key in oracle
        for candidate in key["candidate_compositions"]
    )
    foil_present_rate = float(
        np.mean(
            [
                list(foil[tuple(key["composition"])])
                in key["candidate_compositions"]
                for key in oracle
            ]
        )
    )
    realized_observation_peaks = [
        max(
            max(map(float, event["primitive_observation"])),
            max(map(float, event["manner_observation"])),
        )
        for row in training_visible
        for event in row["events"]
    ]
    lag_values = np.asarray([float(key["lag"]) for key in oracle], dtype=float)
    side_only_probe = _side_only_probe(
        visible,
        oracle,
        conditions["synchronized"],
        config,
    )
    audit = {
        "status": "PASS",
        "corpus_seed": int(corpus_seed),
        "ambiguity_stratum": stratum,
        "grounded_rate_draw": grounded_rate,
        "foil_rate_draw": foil_rate,
        "visibility_draw": visibility,
        "lag_mean_draw": lag_mean,
        "lag_sd_draw": lag_sd,
        "side_informativity_draw": side_informativity,
        "side_noise_draw": side_noise,
        "episode_count": len(training_visible),
        "grounded_episode_count": int(target_counts[True]),
        "null_episode_count": int(target_counts[False]),
        "observed_lag_mean": float(lag_values.mean()),
        "observed_lag_sd": float(lag_values.std(ddof=1)),
        "episode_geometry_digest": canonical_digest(episode_geometry),
        "action_geometry_digest": action_geometry_audit[
            "action_geometry_digest"
        ],
        "action_geometry": action_geometry_audit,
        "train_evaluation_shared_action_prototype_digest": (
            action_geometry_audit["prototype_digest"]
        ),
        "train_evaluation_use_same_action_prototypes": (
            action_geometry_audit["prototype_digest"]
            == provenance["evaluation_action_prototype_digest"]
        ),
        "heldout_composition_digest": canonical_digest([list(value) for value in heldout]),
        "heldout_compositions": [list(value) for value in heldout],
        "heldout_candidate_event_count": int(heldout_candidate_event_count),
        "all_training_candidate_events_exclude_heldout_compositions": (
            heldout_candidate_event_count == 0
        ),
        "heldout_split_space": "one_per_primitive_nonconstant_manner_assignment",
        "every_constituent_exposed_in_training": True,
        "train_evaluation_generator_namespaces_disjoint": (
            generator_namespaces_disjoint
        ),
        "visible_identifiers_are_opaque_and_truth_independent_in_format": True,
        "evaluation_side_fields_absent": True,
        "evaluation_top_level_schema_exact_allowlist": evaluation_schema_exact,
        "repetition_count_digest": canonical_digest(repetition_counts),
        "repetition_counts": repetition_counts,
        "candidate_count_histogram": candidate_count_histogram,
        "candidate_count_histogram_digest": canonical_digest(candidate_count_histogram),
        "realized_foil_present_rate": foil_present_rate,
        "realized_mean_event_observation_peak": float(
            np.mean(realized_observation_peaks)
        ),
        "side_only_probe": side_only_probe,
        "condition_audit": condition_metadata["audit"],
        "provenance": provenance,
        "lexicon_oracle_digest": canonical_digest(lexicon["oracle"]),
        "action_geometry_rng_namespace": design["action_geometry"][
            "rng_namespace"
        ],
    }
    return {
        "schema_version": "nursery-corrective-corpus-v1",
        "corpus_seed": int(corpus_seed),
        "learner_input": {
            "schema_version": "nursery-corrective-learner-input-v1",
            "training_episodes": training_visible,
            "condition_evidence": conditions,
            "evaluation_prompts": eval_visible,
        },
        "protected_adjudication": {
            "schema_version": "nursery-corrective-protected-v1",
            "corpus_seed": int(corpus_seed),
            "training_oracle": oracle,
            "evaluation_keys": eval_keys,
            "lexicon_oracle": lexicon["oracle"],
            "condition_metadata": condition_metadata,
        },
        "audit": audit,
    }
