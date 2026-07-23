from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict, dataclass
import hashlib
import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from babyworld_lite.weak_alignment.protocol import canonical_digest, guard_seed_operation


ACTIONS = ("push", "grasp", "tap", "lift", "shake", "roll")
OBJECTS = ("ball", "cup", "block", "plush", "ring", "spoon")
SIDE_CONDITIONS = (
    "synchronized",
    "shuffled",
    "time_shifted",
    "absent",
    "uninformative",
)
MODEL_VISIBLE_KEYS = frozenset(
    {"schema_version", "episode_id", "split", "episode_group", "utterance", "events", "scene_observation"}
)
EVENT_VISIBLE_KEYS = frozenset({"time", "action_observation"})
UTTERANCE_VISIBLE_KEYS = frozenset({"action_word", "scene_word", "speech_time"})
FORBIDDEN_VISIBLE_TOKENS = frozenset(
    {"true_action", "true_object", "target_event_index", "grounded", "oracle", "event_actions"}
)


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    evaluation_id: str
    evaluation_role: str
    lexical_panel: int
    action_observation: tuple[float, ...]
    scene_observation: tuple[float, ...]
    action_candidate_words: tuple[str, ...]
    scene_candidate_words: tuple[str, ...]
    action_answer_index: int
    scene_answer_index: int
    composition_id: str


@dataclass(frozen=True)
class SyntheticCorpus:
    corpus_seed: int
    visible_episodes: tuple[dict[str, Any], ...]
    oracle_episodes: tuple[dict[str, Any], ...]
    synchronized_side: Mapping[str, tuple[float, ...]]
    condition_views: Mapping[str, tuple[dict[str, Any], ...]]
    donor_map: Mapping[str, str]
    evaluation_items: tuple[EvaluationItem, ...]
    evaluation_oracle: tuple[dict[str, Any], ...]
    lexicon_oracle: Mapping[str, Any]
    audits: Mapping[str, Any]


def _rng_for(seed: int, namespace: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{namespace}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _noisy_category(
    index: int, size: int, *, reliable: bool, noise: float, rng: np.random.Generator
) -> list[float]:
    background = rng.dirichlet(np.ones(size))
    if reliable:
        value = noise * background
        value[index] += 1.0 - noise
    else:
        value = background
    value /= value.sum()
    return [float(item) for item in value]


def _lexicon(seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    rng = _rng_for(seed, "lexicon")
    action_word_to_action: dict[str, str] = {}
    action_panels: list[list[str]] = []
    for panel in range(int(config["design"]["lexical_panels"])):
        words = [f"verb_{panel}_{slot}" for slot in range(len(ACTIONS))]
        permutation = rng.permutation(ACTIONS)
        for word, action in zip(words, permutation):
            action_word_to_action[word] = str(action)
        action_panels.append(words)
    all_words = [word for panel in action_panels for word in panel]
    shuffled_words = list(np.asarray(all_words)[rng.permutation(len(all_words))])
    low, high = sorted(map(int, config["factors"]["word_occurrence_count"]))
    repetition_by_word = {
        str(word): (low if index < len(all_words) // 2 else high)
        for index, word in enumerate(shuffled_words)
    }

    scene_words = [f"scene_{slot}" for slot in range(len(OBJECTS))]
    scene_word_to_object = {
        word: str(obj) for word, obj in zip(scene_words, rng.permutation(OBJECTS))
    }
    object_to_scene_word = {obj: word for word, obj in scene_word_to_object.items()}

    zero_words = [f"zero_verb_{slot}" for slot in range(len(ACTIONS))]
    zero_word_to_action = {
        word: str(action) for word, action in zip(zero_words, rng.permutation(ACTIONS))
    }
    heldout_object_by_action = {
        action: str(obj) for action, obj in zip(ACTIONS, rng.permutation(OBJECTS))
    }
    return {
        "action_word_to_action": action_word_to_action,
        "action_panels": action_panels,
        "primary_scoring_panels": [0, 1],
        "anchor_panel": 2,
        "repetition_by_word": repetition_by_word,
        "scene_word_to_object": scene_word_to_object,
        "object_to_scene_word": object_to_scene_word,
        "scene_words": scene_words,
        "zero_word_to_action": zero_word_to_action,
        "zero_words": zero_words,
        "heldout_object_by_action": heldout_object_by_action,
    }


def _factor_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    factors = config["factors"]
    keys = (
        "speech_action_lag",
        "grounded_utterance_rate",
        "candidate_event_count",
        "action_visibility_rate",
        "side_informativeness",
    )
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(factors[key] for key in keys))
    ]


def _assign_factor_cells(
    lexicon: Mapping[str, Any], config: Mapping[str, Any], seed: int
) -> dict[str, list[dict[str, Any]]]:
    base = _factor_cells(config)
    low, high = sorted(map(int, config["factors"]["word_occurrence_count"]))
    by_repetition: dict[int, list[str]] = defaultdict(list)
    for word, count in lexicon["repetition_by_word"].items():
        by_repetition[int(count)].append(str(word))
    if len(by_repetition[low]) * low != len(base):
        raise ValueError("low-repetition lexemes must cover one complete weak-factor grid")
    if len(by_repetition[high]) * high != 2 * len(base):
        raise ValueError("high-repetition lexemes must cover two complete weak-factor grids")
    output: dict[str, list[dict[str, Any]]] = {}
    for count, repeats in ((low, 1), (high, 2)):
        rng = _rng_for(seed, f"factor-assignment-{count}")
        cells = [copy.deepcopy(cell) for _ in range(repeats) for cell in base]
        cells = list(np.asarray(cells, dtype=object)[rng.permutation(len(cells))])
        offset = 0
        for word in sorted(by_repetition[count]):
            output[word] = [dict(item) for item in cells[offset : offset + count]]
            offset += count
        if offset != len(cells):
            raise AssertionError("factor-cell allocation did not consume the balanced grid")
    return output


def _side_scores(
    *,
    correct_position: int,
    size: int,
    informativeness: float,
    rng: np.random.Generator,
) -> tuple[float, ...]:
    selected = (
        correct_position
        if rng.random() < informativeness
        else int(rng.integers(0, size))
    )
    ranks = np.linspace(-1.0, 1.0, size)
    ranks += rng.normal(0.0, 0.015, size=size)
    ranks.sort()
    values = np.empty(size, dtype=np.float64)
    values[selected] = ranks[-1]
    remaining_positions = [index for index in range(size) if index != selected]
    remaining_values = ranks[:-1][rng.permutation(size - 1)]
    for position, value in zip(remaining_positions, remaining_values):
        values[position] = value
    return tuple(float(value) for value in values)


def _make_episode(
    *,
    corpus_seed: int,
    episode_number: int,
    word: str,
    factors: Mapping[str, Any],
    lexicon: Mapping[str, Any],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any], tuple[float, ...]]:
    true_action = str(lexicon["action_word_to_action"][word])
    allowed_objects = [
        obj for obj in OBJECTS
        if obj != lexicon["heldout_object_by_action"][true_action]
    ]
    true_object = str(rng.choice(allowed_objects))
    scene_word = str(lexicon["object_to_scene_word"][true_object])
    candidate_count = int(factors["candidate_event_count"])
    grounded = bool(rng.random() < float(factors["grounded_utterance_rate"]))
    target_index = int(rng.integers(candidate_count)) if grounded else None
    times = np.linspace(0.14, 0.86, candidate_count)
    times += rng.normal(0.0, 0.015, size=candidate_count)
    times = np.clip(np.sort(times), 0.05, 0.95)
    distractor_actions = [action for action in ACTIONS if action != true_action]
    event_actions = [str(rng.choice(distractor_actions)) for _ in range(candidate_count)]
    if target_index is not None:
        event_actions[target_index] = true_action
    event_visibility = [
        bool(rng.random() < float(factors["action_visibility_rate"]))
        for _ in range(candidate_count)
    ]
    noise = float(config["design"]["action_observation_noise"])
    events = [
        {
            "time": float(times[index]),
            "action_observation": _noisy_category(
                ACTIONS.index(action),
                len(ACTIONS),
                reliable=event_visibility[index],
                noise=noise,
                rng=rng,
            ),
        }
        for index, action in enumerate(event_actions)
    ]
    if target_index is None:
        base_speech_time = float(rng.uniform(0.08, 0.92))
    else:
        base_speech_time = float(times[target_index])
    speech_time = float(
        np.clip(base_speech_time + float(factors["speech_action_lag"]), 0.0, 1.0)
    )
    scene_observation = _noisy_category(
        OBJECTS.index(true_object),
        len(OBJECTS),
        reliable=True,
        noise=float(config["design"]["scene_observation_noise"]),
        rng=rng,
    )
    episode_id = f"dev-c{corpus_seed}-e{episode_number:04d}"
    visible = {
        "schema_version": "synthetic-weak-alignment-visible-v1",
        "episode_id": episode_id,
        "split": "train",
        "episode_group": word,
        "utterance": {
            "action_word": word,
            "scene_word": scene_word,
            "speech_time": speech_time,
        },
        "events": events,
        "scene_observation": scene_observation,
    }
    oracle = {
        "schema_version": "synthetic-weak-alignment-oracle-v1",
        "episode_id": episode_id,
        "corpus_seed": corpus_seed,
        "true_action": true_action,
        "true_object": true_object,
        "target_event_index": target_index,
        "grounded": grounded,
        "event_actions": event_actions,
        "event_action_visible": event_visibility,
        "lexical_panel": int(word.split("_")[1]),
        "held_out_for_primary_scoring": int(word.split("_")[1]) in {0, 1},
        "factors": {
            **{key: float(value) for key, value in factors.items()},
            "candidate_event_count": candidate_count,
            "word_occurrence_count": int(lexicon["repetition_by_word"][word]),
        },
        "realized_speech_action_lag": (
            None if target_index is None else speech_time - float(times[target_index])
        ),
        "target_action_visible": (
            None if target_index is None else event_visibility[target_index]
        ),
    }
    correct_position = candidate_count if target_index is None else target_index
    side = _side_scores(
        correct_position=correct_position,
        size=candidate_count + 1,
        informativeness=float(factors["side_informativeness"]),
        rng=rng,
    )
    return visible, oracle, side


def validate_visible_episode(record: Mapping[str, Any], *, side_expected: bool | None = None) -> None:
    allowed = set(MODEL_VISIBLE_KEYS)
    if side_expected is True:
        allowed.add("side_scores")
    if set(record) != allowed:
        raise ValueError(f"model-visible episode keys violate allowlist: {sorted(set(record) ^ allowed)}")
    if set(record["utterance"]) != UTTERANCE_VISIBLE_KEYS:
        raise ValueError("utterance keys violate model-visible allowlist")
    if any(set(event) != EVENT_VISIBLE_KEYS for event in record["events"]):
        raise ValueError("event keys violate model-visible allowlist")
    flattened = repr(record).lower()
    leaked = sorted(token for token in FORBIDDEN_VISIBLE_TOKENS if token in flattened)
    if leaked:
        raise ValueError(f"ground-truth leakage tokens in model-visible episode: {leaked}")
    if side_expected is False and "side_scores" in record:
        raise ValueError("absent condition structurally contains side scores")


def _group_safe_donors(
    episodes: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, str]:
    output: dict[str, str] = {}
    by_size: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_size[len(episode["events"])].append(episode)
    for size, rows in sorted(by_size.items()):
        ordered = sorted(rows, key=lambda row: (str(row["episode_group"]), str(row["episode_id"])))
        rng = _rng_for(seed, f"donors-{size}")
        # Randomize within groups while preserving contiguous group blocks.
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in ordered:
            grouped[str(row["episode_group"])].append(row)
        ordered = []
        for group in sorted(grouped):
            values = grouped[group]
            order = rng.permutation(len(values))
            ordered.extend(values[int(index)] for index in order)
        valid_shift = None
        for shift in range(1, len(ordered)):
            if all(
                str(row["episode_group"])
                != str(ordered[(index + shift) % len(ordered)]["episode_group"])
                for index, row in enumerate(ordered)
            ):
                valid_shift = shift
                break
        if valid_shift is None:
            raise ValueError(f"group-safe cue donor bijection is impossible for candidate count {size}")
        for index, row in enumerate(ordered):
            donor = ordered[(index + valid_shift) % len(ordered)]
            output[str(row["episode_id"])] = str(donor["episode_id"])
    return output


def _uninformative_permutation(values: Sequence[float], episode_id: str, seed: int) -> list[float]:
    rng = _rng_for(seed, f"uninformative-{episode_id}")
    order = rng.permutation(len(values))
    return [float(values[int(index)]) for index in order]


def _condition_views(
    episodes: Sequence[Mapping[str, Any]],
    synchronized_side: Mapping[str, Sequence[float]],
    donors: Mapping[str, str],
    seed: int,
) -> dict[str, tuple[dict[str, Any], ...]]:
    views: dict[str, tuple[dict[str, Any], ...]] = {}
    for condition in SIDE_CONDITIONS:
        rows: list[dict[str, Any]] = []
        for base in episodes:
            row = copy.deepcopy(dict(base))
            episode_id = str(row["episode_id"])
            own = list(map(float, synchronized_side[episode_id]))
            if condition == "synchronized":
                row["side_scores"] = own
            elif condition == "shuffled":
                row["side_scores"] = list(map(float, synchronized_side[donors[episode_id]]))
            elif condition == "time_shifted":
                row["side_scores"] = list(np.roll(np.asarray(own), 1).astype(float))
            elif condition == "uninformative":
                row["side_scores"] = _uninformative_permutation(own, episode_id, seed)
            elif condition != "absent":
                raise KeyError(condition)
            validate_visible_episode(row, side_expected=condition != "absent")
            rows.append(row)
        views[condition] = tuple(rows)
    return views


def _evaluation_data(
    *, corpus_seed: int, lexicon: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[tuple[EvaluationItem, ...], tuple[dict[str, Any], ...]]:
    rng = _rng_for(corpus_seed, "independent-evaluation")
    items: list[EvaluationItem] = []
    oracle: list[dict[str, Any]] = []
    instances = int(config["evaluation"]["new_instances_per_lexeme"])
    action_noise = float(config["design"]["evaluation_action_observation_noise"])
    scene_noise = float(config["design"]["scene_observation_noise"])
    for panel_index, panel in enumerate(lexicon["action_panels"]):
        role = "primary_heldout_lexemes" if panel_index in {0, 1} else "anchor_lexemes"
        for word in panel:
            true_action = str(lexicon["action_word_to_action"][word])
            true_object = str(lexicon["heldout_object_by_action"][true_action])
            scene_word = str(lexicon["object_to_scene_word"][true_object])
            for repeat in range(instances):
                evaluation_id = f"eval-c{corpus_seed}-p{panel_index}-{word}-r{repeat:02d}"
                action_observation = tuple(
                    _noisy_category(
                        ACTIONS.index(true_action), len(ACTIONS), reliable=True,
                        noise=action_noise, rng=rng,
                    )
                )
                scene_observation = tuple(
                    _noisy_category(
                        OBJECTS.index(true_object), len(OBJECTS), reliable=True,
                        noise=scene_noise, rng=rng,
                    )
                )
                action_candidates = tuple(panel)
                scene_candidates = tuple(lexicon["scene_words"])
                item = EvaluationItem(
                    evaluation_id=evaluation_id,
                    evaluation_role=role,
                    lexical_panel=panel_index,
                    action_observation=action_observation,
                    scene_observation=scene_observation,
                    action_candidate_words=action_candidates,
                    scene_candidate_words=scene_candidates,
                    action_answer_index=action_candidates.index(word),
                    scene_answer_index=scene_candidates.index(scene_word),
                    composition_id=f"{true_object}:{true_action}",
                )
                items.append(item)
                oracle.append({
                    "evaluation_id": evaluation_id,
                    "corpus_seed": corpus_seed,
                    "true_action": true_action,
                    "true_object": true_object,
                    "action_word": word,
                    "scene_word": scene_word,
                    "new_action_instance": True,
                    "object_action_composition_held_out_from_training": True,
                    "held_out_for_primary_scoring": panel_index in {0, 1},
                })
    # A zero-exposure lexical-type negative control; no learner sees these words.
    zero_panel = tuple(lexicon["zero_words"])
    for word in zero_panel:
        true_action = str(lexicon["zero_word_to_action"][word])
        true_object = str(lexicon["heldout_object_by_action"][true_action])
        scene_word = str(lexicon["object_to_scene_word"][true_object])
        evaluation_id = f"eval-c{corpus_seed}-zero-{word}"
        item = EvaluationItem(
            evaluation_id=evaluation_id,
            evaluation_role="zero_exposure_lexical_type_negative_control",
            lexical_panel=3,
            action_observation=tuple(
                _noisy_category(
                    ACTIONS.index(true_action), len(ACTIONS), reliable=True,
                    noise=action_noise, rng=rng,
                )
            ),
            scene_observation=tuple(
                _noisy_category(
                    OBJECTS.index(true_object), len(OBJECTS), reliable=True,
                    noise=scene_noise, rng=rng,
                )
            ),
            action_candidate_words=zero_panel,
            scene_candidate_words=tuple(lexicon["scene_words"]),
            action_answer_index=zero_panel.index(word),
            scene_answer_index=tuple(lexicon["scene_words"]).index(scene_word),
            composition_id=f"{true_object}:{true_action}",
        )
        items.append(item)
        oracle.append({
            "evaluation_id": evaluation_id,
            "corpus_seed": corpus_seed,
            "true_action": true_action,
            "true_object": true_object,
            "action_word": word,
            "scene_word": scene_word,
            "new_action_instance": True,
            "object_action_composition_held_out_from_training": True,
            "held_out_for_primary_scoring": False,
            "zero_training_exposure": True,
        })
    return tuple(items), tuple(oracle)


def _generation_audits(
    visible: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    side: Mapping[str, Sequence[float]],
    views: Mapping[str, Sequence[Mapping[str, Any]]],
    donors: Mapping[str, str],
    eval_items: Sequence[EvaluationItem],
    eval_oracle: Sequence[Mapping[str, Any]],
    lexicon: Mapping[str, Any],
) -> dict[str, Any]:
    oracle_by_id = {str(row["episode_id"]): row for row in oracle}
    factor_combo_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in oracle:
        factors = row["factors"]
        repeat = str(int(factors["word_occurrence_count"]))
        combo = canonical_digest({
            key: factors[key]
            for key in (
                "speech_action_lag", "grounded_utterance_rate",
                "candidate_event_count", "action_visibility_rate", "side_informativeness",
            )
        })
        factor_combo_counts[repeat][combo] += 1
    donor_checks = {
        "bijection": len(set(donors.values())) == len(donors) == len(visible),
        "no_self": all(target != donor for target, donor in donors.items()),
        "same_split": True,
        "different_episode_group": all(
            next(row for row in visible if row["episode_id"] == target)["episode_group"]
            != next(row for row in visible if row["episode_id"] == donor)["episode_group"]
            for target, donor in donors.items()
        ),
        "candidate_count_matched": all(
            len(next(row for row in visible if row["episode_id"] == target)["events"])
            == len(next(row for row in visible if row["episode_id"] == donor)["events"])
            for target, donor in donors.items()
        ),
    }
    base_hashes = {
        condition: canonical_digest([
            {key: value for key, value in row.items() if key != "side_scores"}
            for row in rows
        ])
        for condition, rows in views.items()
    }
    side_distribution_hashes: dict[str, str | None] = {}
    for condition, rows in views.items():
        if condition == "absent":
            side_distribution_hashes[condition] = None
            continue
        sorted_rows = sorted(
            [sorted(map(float, row["side_scores"])) for row in rows],
            key=canonical_digest,
        )
        side_distribution_hashes[condition] = canonical_digest(sorted_rows)
    training_compositions = {
        (str(row["true_object"]), str(row["true_action"])) for row in oracle
    }
    evaluation_compositions = {
        (str(row["true_object"]), str(row["true_action"])) for row in eval_oracle
        if row.get("held_out_for_primary_scoring")
    }
    visible_hashes = {canonical_digest(row) for row in visible}
    eval_hashes = {canonical_digest(asdict(item)) for item in eval_items}
    checks = {
        "visible_allowlist": all(
            set(row) == MODEL_VISIBLE_KEYS
            and set(row["utterance"]) == UTTERANCE_VISIBLE_KEYS
            and all(set(event) == EVENT_VISIBLE_KEYS for event in row["events"])
            for row in visible
        ),
        "oracle_physically_separate_in_memory": all(
            not any(token in repr(row).lower() for token in FORBIDDEN_VISIBLE_TOKENS)
            for row in visible
        ),
        "all_conditions_same_inventory": len(set(base_hashes.values())) == 1,
        "absent_structural": all("side_scores" not in row for row in views["absent"]),
        "present_conditions_have_side": all(
            "side_scores" in row
            for condition, rows in views.items() if condition != "absent"
            for row in rows
        ),
        "distribution_matched_present_controls": len({
            value for value in side_distribution_hashes.values() if value is not None
        }) == 1,
        "donors_valid": all(donor_checks.values()),
        "balanced_factor_grid_within_repetition": all(
            len(set(counter.values())) == 1 for counter in factor_combo_counts.values()
        ),
        "heldout_compositions_disjoint": not (training_compositions & evaluation_compositions),
        "independent_evaluation_hashes": not (visible_hashes & eval_hashes),
        "evaluation_schema_has_no_side": all(
            not any("side" in key or "cue" in key or "motor" in key for key in asdict(item))
            for item in eval_items
        ),
        "randomized_surface_mapping": all(
            action not in word for word, action in lexicon["action_word_to_action"].items()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"synthetic corpus generation audit failed: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "donor_checks": donor_checks,
        "condition_inventory_hashes": base_hashes,
        "side_distribution_hashes": side_distribution_hashes,
        "factor_combo_counts_by_repetition": {
            repeat: dict(sorted(counter.items()))
            for repeat, counter in sorted(factor_combo_counts.items())
        },
        "n_training_episodes": len(visible),
        "n_evaluation_items": len(eval_items),
        "n_primary_heldout_compositions": len(evaluation_compositions),
        "synchronized_side_rows": len(side),
        "oracle_rows": len(oracle_by_id),
    }


def generate_corpus(config: Mapping[str, Any], corpus_seed: int) -> SyntheticCorpus:
    guard_seed_operation(config, operation="generate", corpus_seed=int(corpus_seed))
    if int(corpus_seed) not in set(map(int, config["seeds"]["development"]["corpus"])):
        raise PermissionError("only frozen development corpus seeds are authorized in this task")
    lexicon = _lexicon(int(corpus_seed), config)
    assigned = _assign_factor_cells(lexicon, config, int(corpus_seed))
    rng = _rng_for(int(corpus_seed), "training-corpus")
    visible: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    side: dict[str, tuple[float, ...]] = {}
    episode_number = 0
    for word in sorted(assigned):
        for factors in assigned[word]:
            model_row, oracle_row, scores = _make_episode(
                corpus_seed=int(corpus_seed), episode_number=episode_number,
                word=word, factors=factors, lexicon=lexicon, config=config, rng=rng,
            )
            validate_visible_episode(model_row, side_expected=None)
            visible.append(model_row)
            oracle.append(oracle_row)
            side[str(model_row["episode_id"])] = scores
            episode_number += 1
    donors = _group_safe_donors(visible, int(corpus_seed))
    views = _condition_views(visible, side, donors, int(corpus_seed))
    eval_items, eval_oracle = _evaluation_data(
        corpus_seed=int(corpus_seed), lexicon=lexicon, config=config
    )
    audits = _generation_audits(
        visible, oracle, side, views, donors, eval_items, eval_oracle, lexicon
    )
    return SyntheticCorpus(
        corpus_seed=int(corpus_seed),
        visible_episodes=tuple(visible),
        oracle_episodes=tuple(oracle),
        synchronized_side=side,
        condition_views=views,
        donor_map=donors,
        evaluation_items=eval_items,
        evaluation_oracle=eval_oracle,
        lexicon_oracle=lexicon,
        audits=audits,
    )
