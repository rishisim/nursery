from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import canonical_digest, guard_seed_operation


MODEL_VISIBLE_KEYS = frozenset(
    {"schema_version", "episode_id", "split", "episode_group", "utterance", "events", "raw_stream"}
)
EVENT_VISIBLE_KEYS = frozenset(
    {"start", "end", "action_observation", "object_observation"}
)
UTTERANCE_VISIBLE_KEYS = frozenset({"items", "speech_time"})
ITEM_VISIBLE_KEYS = frozenset({"token", "slot"})
RAW_STREAM_KEYS = frozenset(
    {"timestamps", "imu", "proprio", "contact", "availability"}
)
FORBIDDEN_RAW_KEYS = frozenset(
    {
        "action",
        "word",
        "token",
        "target",
        "referent",
        "grounded",
        "owner",
        "oracle",
        "salience",
        "score",
        "mapping",
        "label",
    }
)
FINAL_FORBIDDEN_FRAGMENTS = (
    "sensor",
    "detector",
    "imu",
    "proprio",
    "contact",
    "touch",
    "cue",
    "side",
    "encoder",
    "trust",
)


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    evaluation_id: str
    exposure_role: str
    action_observation: tuple[float, ...]
    object_observation: tuple[float, ...]
    action_candidate_phrases: tuple[tuple[str, str], ...]
    noun_candidate_words: tuple[str, ...]
    zero_candidate_words: tuple[str, ...]
    action_answer_index: int
    noun_answer_index: int
    zero_answer_index: int
    composition_id: str


@dataclass(frozen=True)
class SyntheticCorpus:
    corpus_seed: int
    visible_episodes: tuple[dict[str, Any], ...]
    oracle_episodes: tuple[dict[str, Any], ...]
    donor_map: Mapping[str, str]
    evaluation_items: tuple[EvaluationItem, ...]
    evaluation_oracle: tuple[dict[str, Any], ...]
    lexicon_oracle: Mapping[str, Any]
    audits: Mapping[str, Any]


@dataclass(frozen=True)
class CalibrationData:
    split: str
    seeds: tuple[int, ...]
    visible_records: tuple[dict[str, Any], ...]
    oracle_records: tuple[dict[str, Any], ...]
    provenance: Mapping[str, Any]


def rng_for(seed: int, namespace: str) -> np.random.Generator:
    digest = hashlib.sha256(f"sensor-v2|{namespace}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _noisy_category(
    index: int,
    size: int,
    *,
    reliable: bool,
    noise: float,
    rng: np.random.Generator,
) -> list[float]:
    background = rng.dirichlet(np.ones(size))
    if reliable:
        value = noise * background
        value[index] += 1.0 - noise
    else:
        value = background
    value = value / value.sum()
    return [float(item) for item in value]


def _lexicon(seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    rng = rng_for(seed, "lexicon")
    primitive_words = [f"ptok_{index}" for index in range(3)]
    manner_words = [f"mtok_{index}" for index in range(2)]
    noun_words = [f"ntok_{index}" for index in range(6)]
    primitive_word_to_index = {
        word: int(value) for word, value in zip(primitive_words, rng.permutation(3))
    }
    manner_word_to_index = {
        word: int(value) for word, value in zip(manner_words, rng.permutation(2))
    }
    noun_word_to_object = {
        word: int(value) for word, value in zip(noun_words, rng.permutation(6))
    }
    primitive_index_to_word = {value: word for word, value in primitive_word_to_index.items()}
    manner_index_to_word = {value: word for word, value in manner_word_to_index.items()}
    object_to_noun_word = {value: word for word, value in noun_word_to_object.items()}
    action_phrases = [
        [primitive_index_to_word[primitive], manner_index_to_word[manner]]
        for primitive in range(3)
        for manner in range(2)
    ]
    structured_holdout = int(rng.integers(0, 6))
    heldout_object_by_action = {
        str(action): int(obj) for action, obj in enumerate(rng.permutation(6))
    }
    zero_words = [f"ztok_{index}" for index in range(6)]
    zero_word_to_action = {
        word: int(value) for word, value in zip(zero_words, rng.permutation(6))
    }
    return {
        "primitive_words": primitive_words,
        "manner_words": manner_words,
        "noun_words": noun_words,
        "primitive_word_to_index": primitive_word_to_index,
        "manner_word_to_index": manner_word_to_index,
        "noun_word_to_object": noun_word_to_object,
        "primitive_index_to_word": primitive_index_to_word,
        "manner_index_to_word": manner_index_to_word,
        "object_to_noun_word": object_to_noun_word,
        "action_phrases": action_phrases,
        "structured_holdout_action": structured_holdout,
        "heldout_object_by_action": heldout_object_by_action,
        "zero_words": zero_words,
        "zero_word_to_action": zero_word_to_action,
    }


def _balanced_factor_schedule(
    config: Mapping[str, Any], *, count: int, seed: int, namespace: str
) -> list[dict[str, Any]]:
    schedules: dict[str, list[Any]] = {}
    for factor, levels in config["factors"].items():
        if count % len(levels) != 0:
            raise ValueError(f"{namespace} count {count} is not divisible by {factor} levels")
        values = list(levels) * (count // len(levels))
        rng = rng_for(seed, f"factor-{namespace}-{factor}")
        order = rng.permutation(count)
        schedules[factor] = [values[int(index)] for index in order]
    return [
        {factor: values[index] for factor, values in schedules.items()}
        for index in range(count)
    ]


def _event_windows(
    candidate_count: int, stream_samples: int, rng: np.random.Generator, config: Mapping[str, Any]
) -> list[tuple[int, int]]:
    minimum_separation = int(config["stream"]["maximum_event_width"]) + 1
    first_center = 5
    eligible_count = stream_samples - 10
    compressed_count = eligible_count - (candidate_count - 1) * (
        minimum_separation - 1
    )
    if compressed_count < candidate_count:
        raise RuntimeError("could not place nonoverlapping random candidate windows")
    base = np.sort(
        rng.choice(compressed_count, size=candidate_count, replace=False)
    )
    centers = (
        first_center
        + base
        + np.arange(candidate_count) * (minimum_separation - 1)
    )
    windows: list[tuple[int, int]] = []
    for center in centers:
        width = int(
            rng.integers(
                int(config["stream"]["minimum_event_width"]),
                int(config["stream"]["maximum_event_width"]) + 1,
            )
        )
        start = max(1, int(round(center)) - width // 2)
        end = min(stream_samples - 2, start + width - 1)
        start = max(1, end - width + 1)
        windows.append((start, end))
    return windows


def _add_pulse(
    imu: np.ndarray,
    proprio: np.ndarray,
    contact: np.ndarray,
    *,
    start: int,
    end: int,
    amplitude: float,
    rng: np.random.Generator,
) -> None:
    start = max(0, int(start))
    end = min(len(imu) - 1, int(end))
    if end <= start:
        return
    length = end - start + 1
    envelope = np.hanning(length + 2)[1:-1]
    direction = rng.normal(size=6)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    oscillation = 0.75 + 0.25 * np.sin(np.linspace(0.0, 3.0 * np.pi, length))
    imu[start : end + 1] += (
        amplitude * envelope[:, None] * oscillation[:, None] * direction[None, :]
    )
    proprio[start : end + 1, 0] += amplitude * 0.8 * envelope
    proprio[start : end + 1, 1] += amplitude * 0.45 * np.abs(np.gradient(envelope))
    contact[start : end + 1, 0] += amplitude * 0.65 * (envelope > 0.45)


def _raw_stream(
    *,
    windows: Sequence[tuple[int, int]],
    owners: Sequence[str],
    factors: Mapping[str, Any],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = int(config["stream"]["samples"])
    baseline = float(config["stream"]["baseline_noise_sd"])
    imu = rng.normal(0.0, baseline, size=(samples, 6))
    proprio = rng.normal(0.0, baseline * 0.55, size=(samples, 2))
    contact = np.abs(rng.normal(0.0, baseline * 0.18, size=(samples, 1)))
    availability = np.ones((samples, 9), dtype=np.float64)
    active = np.zeros(samples, dtype=np.int8)
    boundary = np.zeros(samples, dtype=np.int8)
    information = float(factors["sensor_informativeness"])
    amplitude = float(factors["sensor_snr"])
    aligned_pulses = 0
    randomized_pulses = 0
    false_pulses = 0
    for (start, end), owner in zip(windows, owners):
        if owner == "wearer":
            active[start : end + 1] = 1
            boundary[start] = 1
            boundary[end] = 1
    if information == 0.0:
        pulse_count = max(1, (len(windows) + 1) // 2)
        if rng.random() < float(factors["false_positive_sensor_rate"]):
            pulse_count += 1
            false_pulses += 1
        for pulse_index in range(pulse_count):
            reference_start, reference_end = windows[pulse_index % len(windows)]
            width = reference_end - reference_start + 1
            center = int(rng.integers(4, samples - 4))
            pulse_start = center - width // 2
            _add_pulse(
                imu,
                proprio,
                contact,
                start=pulse_start,
                end=pulse_start + width - 1,
                amplitude=amplitude,
                rng=rng,
            )
            randomized_pulses += 1
    else:
        for (start, end), owner in zip(windows, owners):
            if owner == "wearer":
                if rng.random() < information:
                    pulse_start, pulse_end = start, end
                    aligned_pulses += 1
                else:
                    width = end - start + 1
                    center = int(rng.integers(4, samples - 4))
                    pulse_start = center - width // 2
                    pulse_end = pulse_start + width - 1
                    randomized_pulses += 1
                _add_pulse(
                    imu,
                    proprio,
                    contact,
                    start=pulse_start,
                    end=pulse_end,
                    amplitude=amplitude,
                    rng=rng,
                )
            elif rng.random() < float(factors["false_positive_sensor_rate"]):
                _add_pulse(
                    imu,
                    proprio,
                    contact,
                    start=start,
                    end=end,
                    amplitude=amplitude,
                    rng=rng,
                )
                false_pulses += 1

    dropout = float(factors["sensor_dropout_rate"])
    if dropout > 0:
        length = max(1, int(round(samples * dropout)))
        for start_axis, stop_axis in ((0, 6), (6, 8), (8, 9)):
            start = int(rng.integers(0, samples - length + 1))
            availability[start : start + length, start_axis:stop_axis] = 0.0
    imu *= availability[:, :6]
    proprio *= availability[:, 6:8]
    contact *= availability[:, 8:9]
    raw = {
        "timestamps": [int(value) for value in range(samples)],
        "imu": imu.astype(float).tolist(),
        "proprio": proprio.astype(float).tolist(),
        "contact": contact.astype(float).tolist(),
        "availability": availability.astype(float).tolist(),
    }
    oracle = {
        "wearer_active": active.astype(int).tolist(),
        "wearer_boundary": boundary.astype(int).tolist(),
        "aligned_pulse_count": aligned_pulses,
        "randomized_pulse_count": randomized_pulses,
        "false_positive_pulse_count": false_pulses,
    }
    return raw, oracle


def _action_components(action_index: int) -> tuple[int, int]:
    return int(action_index) // 2, int(action_index) % 2


def _action_observation(
    action_index: int,
    *,
    reliable: bool,
    noise: float,
    rng: np.random.Generator,
) -> list[float]:
    primitive, manner = _action_components(action_index)
    return [
        *_noisy_category(primitive, 3, reliable=reliable, noise=noise, rng=rng),
        *_noisy_category(manner, 2, reliable=reliable, noise=noise, rng=rng),
    ]


def _make_episode(
    *,
    corpus_seed: int,
    episode_number: int,
    family: str,
    intended_index: int,
    factors: Mapping[str, Any],
    lexicon: Mapping[str, Any],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = int(config["stream"]["samples"])
    candidate_count = int(factors["candidate_event_count"])
    windows = _event_windows(candidate_count, samples, rng, config)
    irrelevant = bool(rng.random() < float(config["design"]["irrelevant_utterance_rate"]))
    grounded = bool(
        not irrelevant and rng.random() < float(factors["grounded_utterance_rate"])
    )
    preferred_distractor_index: int | None = None
    if grounded and family == "action" and int(factors["speech_action_lag_samples"]) != 0:
        lag = int(factors["speech_action_lag_samples"])
        centers = [0.5 * (start + end) for start, end in windows]
        ordered_pairs = [
            (abs((centers[distractor] - centers[target]) - lag), target, distractor)
            for target in range(candidate_count)
            for distractor in range(candidate_count)
            if target != distractor
        ]
        minimum_error = min(value[0] for value in ordered_pairs)
        best = [value for value in ordered_pairs if value[0] == minimum_error]
        _, target_index, preferred_distractor_index = best[
            int(rng.integers(len(best)))
        ]
    else:
        target_index = int(rng.integers(candidate_count)) if grounded else None
    event_actions = [int(rng.integers(0, 6)) for _ in range(candidate_count)]
    event_objects = [int(rng.integers(0, 6)) for _ in range(candidate_count)]
    distractor_candidates = [
        index for index in range(candidate_count) if index != target_index
    ]
    structured_distractor_index = (
        int(preferred_distractor_index)
        if preferred_distractor_index is not None
        else int(rng.choice(distractor_candidates))
    )
    structured_distractor_concept = (
        (intended_index + 3) % 6 if family == "action" else (intended_index + 1) % 6
    )
    structured_distractor_active = bool(
        target_index is None
        or rng.random()
        < float(config["design"]["structured_distractor_probability_when_grounded"])
    )
    if target_index is not None:
        if family == "action":
            event_actions[target_index] = intended_index
            allowed = [
                value
                for value in range(6)
                if value != int(lexicon["heldout_object_by_action"][str(intended_index)])
            ]
            event_objects[target_index] = int(rng.choice(allowed))
        else:
            event_objects[target_index] = intended_index
            allowed = [
                value
                for value in range(6)
                if int(lexicon["heldout_object_by_action"][str(value)]) != intended_index
            ]
            event_actions[target_index] = int(rng.choice(allowed))
    else:
        if family == "action":
            event_actions = [
                value if value != intended_index else (value + 1) % 6
                for value in event_actions
            ]
        else:
            event_objects = [
                value if value != intended_index else (value + 1) % 6
                for value in event_objects
            ]
    if family == "action":
        if target_index is None:
            event_actions = [structured_distractor_concept] * candidate_count
        else:
            if structured_distractor_active:
                event_actions[structured_distractor_index] = structured_distractor_concept
            event_actions = [
                value
                if index == target_index or value != intended_index
                else (value + 1) % 6
                for index, value in enumerate(event_actions)
            ]
    elif structured_distractor_active:
        event_objects[structured_distractor_index] = structured_distractor_concept
    if family == "noun":
        event_objects = [
            value
            if index == target_index or value != intended_index
            else (value + 1) % 6
            for index, value in enumerate(event_objects)
        ]
    for index, (action_value, object_value) in enumerate(
        zip(event_actions, event_objects)
    ):
        heldout = int(lexicon["heldout_object_by_action"][str(action_value)])
        if object_value == heldout:
            event_objects[index] = (object_value + 1) % 6
    visible_flags = [
        bool(rng.random() < float(factors["action_visibility_rate"]))
        for _ in range(candidate_count)
    ]
    if family == "action":
        owners = [
            "wearer"
            if rng.random() < 0.15 * float(factors["wearer_event_prevalence"])
            else "environment"
            for _ in range(candidate_count)
        ]
        if target_index is not None:
            owners[target_index] = "wearer"
            owners[structured_distractor_index] = "environment"
        else:
            owners = ["environment" for _ in owners]
    else:
        owners = [
            "wearer"
            if rng.random() < float(factors["wearer_event_prevalence"])
            else "environment"
            for _ in range(candidate_count)
        ]
    events = [
        {
            "start": int(windows[index][0]),
            "end": int(windows[index][1]),
            "action_observation": _action_observation(
                event_actions[index],
                reliable=visible_flags[index],
                noise=float(config["design"]["action_observation_noise"]),
                rng=rng,
            ),
            "object_observation": _noisy_category(
                event_objects[index],
                6,
                reliable=visible_flags[index],
                noise=float(config["design"]["object_observation_noise"]),
                rng=rng,
            ),
        }
        for index in range(candidate_count)
    ]
    if irrelevant:
        base_items = [{"token": "filler", "slot": "ignore"}]
        episode_group = "filler"
        relation = "irrelevant"
    elif family == "action":
        primitive, manner = _action_components(intended_index)
        primitive_word = str(lexicon["primitive_index_to_word"][primitive])
        manner_word = str(lexicon["manner_index_to_word"][manner])
        base_items = [
            {"token": primitive_word, "slot": "primitive"},
            {"token": manner_word, "slot": "manner"},
        ]
        episode_group = f"{primitive_word}|{manner_word}"
        relation = "grounded_action" if grounded else "no_visible_referent"
    else:
        noun_word = str(lexicon["object_to_noun_word"][intended_index])
        base_items = [{"token": noun_word, "slot": "noun"}]
        episode_group = noun_word
        relation = "grounded_noun" if grounded else "no_visible_referent"
    repetition = int(factors["lexical_repetition_count"])
    items = [copy.deepcopy(item) for _ in range(repetition) for item in base_items]
    if target_index is None:
        speech_time = int(rng.integers(2, samples - 2))
    else:
        start, end = windows[target_index]
        center = (start + end) // 2
        speech_time = int(
            np.clip(center + int(factors["speech_action_lag_samples"]), 0, samples - 1)
        )
    raw_stream, raw_oracle = _raw_stream(
        windows=windows,
        owners=owners,
        factors=factors,
        config=config,
        rng=rng,
    )
    episode_id = f"v2-dev-c{corpus_seed}-e{episode_number:04d}"
    visible = {
        "schema_version": "synthetic-sensor-event-visible-v2",
        "episode_id": episode_id,
        "split": "train",
        "episode_group": episode_group,
        "utterance": {"items": items, "speech_time": speech_time},
        "events": events,
        "raw_stream": raw_stream,
    }
    oracle = {
        "schema_version": "synthetic-sensor-event-oracle-v2",
        "episode_id": episode_id,
        "corpus_seed": int(corpus_seed),
        "family": family,
        "intended_index": int(intended_index),
        "target_event_index": target_index,
        "grounded": grounded,
        "relation": relation,
        "true_event_boundaries": [list(value) for value in windows],
        "event_owners": owners,
        "event_action_indices": event_actions,
        "event_object_indices": event_objects,
        "event_visible": visible_flags,
        "structured_distractor_event_index": structured_distractor_index,
        "structured_distractor_concept_index": structured_distractor_concept,
        "structured_distractor_active": structured_distractor_active,
        "selectivity_sensor_temporal_coupling": (
            "target_relevant_for_action"
            if family == "action"
            else "event_ownership_independent_of_noun_target"
        ),
        "factors": {
            key: (int(value) if isinstance(value, (int, np.integer)) else float(value))
            for key, value in factors.items()
        },
        **raw_oracle,
    }
    return visible, oracle


def validate_visible_episode(record: Mapping[str, Any], *, raw_expected: bool) -> None:
    expected = set(MODEL_VISIBLE_KEYS)
    if not raw_expected:
        expected.remove("raw_stream")
    if set(record) != expected:
        raise ValueError(f"visible episode keys violate allowlist: {sorted(set(record) ^ expected)}")
    if set(record["utterance"]) != UTTERANCE_VISIBLE_KEYS:
        raise ValueError("utterance keys violate allowlist")
    if any(set(item) != ITEM_VISIBLE_KEYS for item in record["utterance"]["items"]):
        raise ValueError("utterance item keys violate allowlist")
    if any(set(event) != EVENT_VISIBLE_KEYS for event in record["events"]):
        raise ValueError("event keys violate allowlist")
    if raw_expected:
        raw = record["raw_stream"]
        if set(raw) != RAW_STREAM_KEYS:
            raise ValueError("raw stream keys violate allowlist")
        lower_keys = {str(key).lower() for key in raw}
        if any(any(token in key for token in FORBIDDEN_RAW_KEYS) for key in lower_keys):
            raise ValueError("semantic or oracle key leaked into raw stream")
        samples = len(raw["timestamps"])
        if not (
            len(raw["imu"]) == samples
            and len(raw["proprio"]) == samples
            and len(raw["contact"]) == samples
            and len(raw["availability"]) == samples
            and all(len(row) == 6 for row in raw["imu"])
            and all(len(row) == 2 for row in raw["proprio"])
            and all(len(row) == 1 for row in raw["contact"])
            and all(len(row) == 9 for row in raw["availability"])
        ):
            raise ValueError("raw stream shapes violate the six-axis + state/contact contract")


def _group_safe_donor_map(
    episodes: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, str]:
    output: dict[str, str] = {}
    by_count: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_count[len(row["events"])].append(row)
    for count, rows in sorted(by_count.items()):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["episode_group"])].append(row)
        ordered: list[Mapping[str, Any]] = []
        rng = rng_for(seed, f"donor-{count}")
        for group in sorted(grouped):
            values = grouped[group]
            order = rng.permutation(len(values))
            ordered.extend(values[int(index)] for index in order)
        shift = next(
            (
                candidate
                for candidate in range(1, len(ordered))
                if all(
                    str(row["episode_group"])
                    != str(ordered[(index + candidate) % len(ordered)]["episode_group"])
                    for index, row in enumerate(ordered)
                )
            ),
            None,
        )
        if shift is None:
            raise ValueError(f"group-safe donor bijection impossible for {count} candidates")
        for index, row in enumerate(ordered):
            output[str(row["episode_id"])] = str(
                ordered[(index + shift) % len(ordered)]["episode_id"]
            )
    return output


def _shift_raw(raw: Mapping[str, Any], offset: int) -> dict[str, Any]:
    output = {"timestamps": list(raw["timestamps"])}
    for key in ("imu", "proprio", "contact", "availability"):
        output[key] = np.roll(np.asarray(raw[key], dtype=float), int(offset), axis=0).tolist()
    return output


def _uninformative_raw(raw: Mapping[str, Any], episode_id: str, seed: int) -> dict[str, Any]:
    samples = len(raw["timestamps"])
    block = 4
    blocks = list(range(samples // block))
    rng = rng_for(seed, f"uninformative-{episode_id}")
    order = list(map(int, rng.permutation(blocks)))
    indices = [index for block_index in order for index in range(block_index * block, block_index * block + block)]
    indices.extend(range((samples // block) * block, samples))
    output = {"timestamps": list(raw["timestamps"])}
    for key in ("imu", "proprio", "contact", "availability"):
        values = np.asarray(raw[key], dtype=float)
        output[key] = values[indices].tolist()
    return output


def condition_view(
    corpus: SyntheticCorpus, condition: str, config: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    if condition not in set(config["conditions"]["names"]):
        raise KeyError(condition)
    by_id = {str(row["episode_id"]): row for row in corpus.visible_episodes}
    output: list[dict[str, Any]] = []
    for base in corpus.visible_episodes:
        row = copy.deepcopy(base)
        episode_id = str(row["episode_id"])
        if condition == "absent":
            row.pop("raw_stream")
        elif condition == "shuffled":
            donor = by_id[str(corpus.donor_map[episode_id])]
            row["raw_stream"] = copy.deepcopy(donor["raw_stream"])
        elif condition in config["conditions"]["time_shift_offsets"]:
            row["raw_stream"] = _shift_raw(
                row["raw_stream"], int(config["conditions"]["time_shift_offsets"][condition])
            )
        elif condition == "uninformative":
            row["raw_stream"] = _uninformative_raw(
                row["raw_stream"], episode_id, corpus.corpus_seed
            )
        elif condition != "synchronized":
            raise AssertionError(condition)
        validate_visible_episode(row, raw_expected=condition != "absent")
        output.append(row)
    return tuple(output)


def _raw_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    joint_rows: list[Any] = []
    for row in rows:
        raw = row.get("raw_stream")
        if raw is None:
            continue
        for index in range(len(raw["timestamps"])):
            joint_rows.append(
                [
                    *raw["imu"][index],
                    *raw["proprio"][index],
                    *raw["contact"][index],
                    *raw["availability"][index],
                ]
            )
    return canonical_digest(sorted(joint_rows, key=canonical_digest))


def _evaluation_data(
    *, corpus_seed: int, lexicon: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[tuple[EvaluationItem, ...], tuple[dict[str, Any], ...]]:
    rng = rng_for(corpus_seed, "evaluation")
    instances = int(config["design"]["evaluation_instances_per_concept"])
    noise = float(config["design"]["evaluation_observation_noise"])
    action_candidates = tuple(tuple(value) for value in lexicon["action_phrases"])
    noun_candidates = tuple(map(str, lexicon["noun_words"]))
    zero_candidates = tuple(map(str, lexicon["zero_words"]))
    zero_by_action = {value: word for word, value in lexicon["zero_word_to_action"].items()}
    items: list[EvaluationItem] = []
    oracle: list[dict[str, Any]] = []
    for action_index in range(6):
        true_object = int(lexicon["heldout_object_by_action"][str(action_index)])
        noun_word = str(lexicon["object_to_noun_word"][true_object])
        zero_word = str(zero_by_action[action_index])
        role = (
            "structured_heldout_concept"
            if action_index == int(lexicon["structured_holdout_action"])
            else "seen_combination"
        )
        for repeat in range(instances):
            evaluation_id = f"v2-eval-c{corpus_seed}-a{action_index}-r{repeat:02d}"
            item = EvaluationItem(
                evaluation_id=evaluation_id,
                exposure_role=role,
                action_observation=tuple(
                    _action_observation(
                        action_index, reliable=True, noise=noise, rng=rng
                    )
                ),
                object_observation=tuple(
                    _noisy_category(
                        true_object, 6, reliable=True, noise=noise, rng=rng
                    )
                ),
                action_candidate_phrases=action_candidates,
                noun_candidate_words=noun_candidates,
                zero_candidate_words=zero_candidates,
                action_answer_index=action_index,
                noun_answer_index=noun_candidates.index(noun_word),
                zero_answer_index=zero_candidates.index(zero_word),
                composition_id=f"object={true_object}|action={action_index}",
            )
            items.append(item)
            oracle.append(
                {
                    "evaluation_id": evaluation_id,
                    "corpus_seed": int(corpus_seed),
                    "true_action_index": action_index,
                    "true_primitive_index": _action_components(action_index)[0],
                    "true_manner_index": _action_components(action_index)[1],
                    "true_object_index": true_object,
                    "new_action_instance": True,
                    "object_action_composition_held_out_from_training": True,
                    "structured_combination_held_out_from_training": role
                    == "structured_heldout_concept",
                    "component_words_seen_in_training": True,
                    "zero_panel_training_exposure": 0,
                }
            )
    return tuple(items), tuple(oracle)


def _generation_audits(
    visible: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    donor_map: Mapping[str, str],
    evaluation_items: Sequence[EvaluationItem],
    evaluation_oracle: Sequence[Mapping[str, Any]],
    lexicon: Mapping[str, Any],
    config: Mapping[str, Any],
    corpus: SyntheticCorpus | None = None,
) -> dict[str, Any]:
    by_id = {str(row["episode_id"]): row for row in visible}
    oracle_by_id = {str(row["episode_id"]): row for row in oracle}
    factor_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for row in oracle:
        for factor, value in row["factors"].items():
            factor_counts[str(row["family"])][factor][str(value)] += 1
    expected_per_family = len(visible) // 2
    marginal_balance = all(
        len(set(counter.values())) == 1 and sum(counter.values()) == expected_per_family
        for family in ("action", "noun")
        for counter in factor_counts[family].values()
    )
    training_action_compositions = {
        (int(obj), int(action))
        for row in oracle
        if row["family"] == "action"
        for obj, action in zip(row["event_object_indices"], row["event_action_indices"])
        if row["target_event_index"] is not None
        and int(action) == int(row["intended_index"])
    }
    evaluation_compositions = {
        (int(row["true_object_index"]), int(row["true_action_index"]))
        for row in evaluation_oracle
    }
    structured = int(lexicon["structured_holdout_action"])
    action_training_intended = {
        int(row["intended_index"])
        for row in oracle
        if row["family"] == "action" and row["relation"] != "irrelevant"
    }
    action_target_rows = [
        row
        for row in oracle
        if row["family"] == "action" and row["target_event_index"] is not None
    ]
    noun_target_rows = [
        row
        for row in oracle
        if row["family"] == "noun" and row["target_event_index"] is not None
    ]
    noun_target_wearer = [
        float(row["event_owners"][int(row["target_event_index"])] == "wearer")
        for row in noun_target_rows
    ]
    noun_all_wearer = [
        float(owner == "wearer")
        for row in oracle
        if row["family"] == "noun"
        for owner in row["event_owners"]
    ]
    donor_checks = {
        "bijection": len(donor_map) == len(visible)
        and len(set(donor_map.values())) == len(visible),
        "no_self": all(target != donor for target, donor in donor_map.items()),
        "different_lexical_group": all(
            by_id[target]["episode_group"] != by_id[donor]["episode_group"]
            for target, donor in donor_map.items()
        ),
        "candidate_count_matched": all(
            len(by_id[target]["events"]) == len(by_id[donor]["events"])
            for target, donor in donor_map.items()
        ),
    }
    checks = {
        "visible_schema_allowlist": all(
            set(row) == MODEL_VISIBLE_KEYS
            and set(row["utterance"]) == UTTERANCE_VISIBLE_KEYS
            and all(set(event) == EVENT_VISIBLE_KEYS for event in row["events"])
            and set(row["raw_stream"]) == RAW_STREAM_KEYS
            for row in visible
        ),
        "raw_stream_has_six_axis_imu_and_state_contact": all(
            all(len(value) == 6 for value in row["raw_stream"]["imu"])
            and all(len(value) == 2 for value in row["raw_stream"]["proprio"])
            and all(len(value) == 1 for value in row["raw_stream"]["contact"])
            for row in visible
        ),
        "raw_stream_contains_no_semantic_or_oracle_keys": all(
            not any(
                any(token in str(key).lower() for token in FORBIDDEN_RAW_KEYS)
                for key in row["raw_stream"]
            )
            for row in visible
        ),
        "oracle_physically_separate": all(
            not any(key in row for key in ("target_event_index", "event_owners", "grounded"))
            for row in visible
        ),
        "factor_marginals_balanced_within_both_families": marginal_balance,
        "all_action_targets_are_wearer_caused": all(
            row["event_owners"][int(row["target_event_index"])] == "wearer"
            for row in action_target_rows
        ),
        "noun_target_ownership_matches_bag_prevalence_empirically": (
            not noun_target_wearer
            or abs(float(np.mean(noun_target_wearer)) - float(np.mean(noun_all_wearer)))
            <= 0.20
        ),
        "structured_combination_absent_but_components_exposed": structured
        not in action_training_intended
        and all(
            any(_action_components(value)[position] == _action_components(structured)[position]
                for value in action_training_intended)
            for position in (0, 1)
        ),
        "heldout_object_action_compositions_disjoint": not (
            training_action_compositions & evaluation_compositions
        ),
        "evaluation_items_are_new_instances": all(
            row["new_action_instance"] for row in evaluation_oracle
        ),
        "zero_exposure_words_absent_from_training": not (
            set(lexicon["zero_words"])
            & {
                str(item["token"])
                for row in visible
                for item in row["utterance"]["items"]
            }
        ),
        "evaluation_schema_rejects_training_channels": all(
            not any(fragment in field.lower() for fragment in FINAL_FORBIDDEN_FRAGMENTS)
            for field in asdict(evaluation_items[0])
        ),
        "donor_map_valid": all(donor_checks.values()),
        "randomized_surface_mappings": len(
            {
                canonical_digest(lexicon["primitive_word_to_index"]),
                canonical_digest(lexicon["manner_word_to_index"]),
                canonical_digest(lexicon["noun_word_to_object"]),
            }
        )
        == 3,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v2 synthetic generation audit failed: {checks}")
    return {
        "valid": True,
        "checks": checks,
        "donor_checks": donor_checks,
        "factor_counts_by_family": {
            family: {
                factor: dict(sorted(counter.items()))
                for factor, counter in sorted(values.items())
            }
            for family, values in sorted(factor_counts.items())
        },
        "n_training_episodes": len(visible),
        "n_evaluation_items": len(evaluation_items),
        "n_action_target_rows": len(action_target_rows),
        "n_noun_target_rows": len(noun_target_rows),
        "noun_target_wearer_rate": float(np.mean(noun_target_wearer)),
        "noun_bag_wearer_rate": float(np.mean(noun_all_wearer)),
        "structured_holdout_action": structured,
        "oracle_rows": len(oracle_by_id),
    }


def generate_corpus(config: Mapping[str, Any], corpus_seed: int) -> SyntheticCorpus:
    guard_seed_operation(config, operation="generate", seeds=[int(corpus_seed)])
    if int(corpus_seed) not in set(map(int, config["seeds"]["development"]["corpus"])):
        raise PermissionError("only configured v2 development corpus seeds may be generated")
    lexicon = _lexicon(int(corpus_seed), config)
    action_seen = [
        value for value in range(6) if value != int(lexicon["structured_holdout_action"])
    ]
    action_count = len(action_seen) * int(
        config["design"]["training_action_episodes_per_seen_concept"]
    )
    noun_count = 6 * int(config["design"]["training_noun_episodes_per_object"])
    action_factors = _balanced_factor_schedule(
        config, count=action_count, seed=int(corpus_seed), namespace="action"
    )
    noun_factors = _balanced_factor_schedule(
        config, count=noun_count, seed=int(corpus_seed), namespace="noun"
    )
    inventory: list[tuple[str, int, Mapping[str, Any]]] = []
    index = 0
    for action in action_seen:
        for _ in range(int(config["design"]["training_action_episodes_per_seen_concept"])):
            inventory.append(("action", action, action_factors[index]))
            index += 1
    index = 0
    for obj in range(6):
        for _ in range(int(config["design"]["training_noun_episodes_per_object"])):
            inventory.append(("noun", obj, noun_factors[index]))
            index += 1
    rng = rng_for(int(corpus_seed), "training-episodes")
    visible: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for episode_number, (family, intended, factors) in enumerate(inventory):
        model_row, oracle_row = _make_episode(
            corpus_seed=int(corpus_seed),
            episode_number=episode_number,
            family=family,
            intended_index=intended,
            factors=factors,
            lexicon=lexicon,
            config=config,
            rng=rng,
        )
        validate_visible_episode(model_row, raw_expected=True)
        visible.append(model_row)
        oracle.append(oracle_row)
    donor_map = _group_safe_donor_map(visible, int(corpus_seed))
    evaluation_items, evaluation_oracle = _evaluation_data(
        corpus_seed=int(corpus_seed), lexicon=lexicon, config=config
    )
    provisional = SyntheticCorpus(
        corpus_seed=int(corpus_seed),
        visible_episodes=tuple(visible),
        oracle_episodes=tuple(oracle),
        donor_map=donor_map,
        evaluation_items=evaluation_items,
        evaluation_oracle=evaluation_oracle,
        lexicon_oracle=lexicon,
        audits={},
    )
    audits = _generation_audits(
        visible,
        oracle,
        donor_map,
        evaluation_items,
        evaluation_oracle,
        lexicon,
        config,
    )
    views = {
        condition: condition_view(provisional, condition, config)
        for condition in config["conditions"]["names"]
    }
    inventory_hashes = {
        condition: canonical_digest(
            [
                {key: value for key, value in row.items() if key != "raw_stream"}
                for row in rows
            ]
        )
        for condition, rows in views.items()
    }
    raw_hashes = {
        condition: None if condition == "absent" else _raw_rows_digest(rows)
        for condition, rows in views.items()
    }
    condition_checks = {
        "matched_non_sensor_inventory": len(set(inventory_hashes.values())) == 1,
        "absent_is_structural": all("raw_stream" not in row for row in views["absent"]),
        "all_present_conditions_have_raw_stream": all(
            "raw_stream" in row
            for condition, rows in views.items()
            if condition != "absent"
            for row in rows
        ),
        "present_raw_row_multisets_match": len(
            {value for value in raw_hashes.values() if value is not None}
        )
        == 1,
    }
    if not all(condition_checks.values()):
        raise RuntimeError(f"v2 condition construction audit failed: {condition_checks}")
    audits = {
        **audits,
        "condition_checks": condition_checks,
        "condition_inventory_hashes": inventory_hashes,
        "condition_raw_row_multiset_hashes": raw_hashes,
    }
    return SyntheticCorpus(
        corpus_seed=int(corpus_seed),
        visible_episodes=tuple(visible),
        oracle_episodes=tuple(oracle),
        donor_map=donor_map,
        evaluation_items=evaluation_items,
        evaluation_oracle=evaluation_oracle,
        lexicon_oracle=lexicon,
        audits=audits,
    )


def _calibration_factors(
    config: Mapping[str, Any], *, count: int, seed: int, split: str
) -> list[dict[str, Any]]:
    selected = {
        key: value
        for key, value in config["factors"].items()
        if key
        in {
            "candidate_event_count",
            "sensor_informativeness",
            "sensor_snr",
            "sensor_dropout_rate",
            "false_positive_sensor_rate",
            "wearer_event_prevalence",
        }
    }
    temporary = {"factors": selected}
    return _balanced_factor_schedule(
        temporary, count=count, seed=seed, namespace=f"calibration-{split}"
    )


def generate_calibration_data(
    config: Mapping[str, Any], *, split: str
) -> CalibrationData:
    if split not in {"train", "validation"}:
        raise ValueError("calibration split must be train or validation")
    seeds = tuple(map(int, config["seeds"]["generic_calibration"][split]))
    count_key = "episodes_per_train_seed" if split == "train" else "episodes_per_validation_seed"
    per_seed = int(config["calibration"][count_key])
    visible: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for seed in seeds:
        guard_seed_operation(config, operation="calibrate", seeds=[seed])
        factors = _calibration_factors(config, count=per_seed, seed=seed, split=split)
        rng = rng_for(seed, f"calibration-{split}")
        for index, factor_row in enumerate(factors):
            candidate_count = int(factor_row["candidate_event_count"])
            windows = _event_windows(
                candidate_count, int(config["stream"]["samples"]), rng, config
            )
            owners = [
                "wearer"
                if rng.random() < float(factor_row["wearer_event_prevalence"])
                else "environment"
                for _ in windows
            ]
            raw, raw_oracle = _raw_stream(
                windows=windows,
                owners=owners,
                factors=factor_row,
                config=config,
                rng=rng,
            )
            calibration_id = f"v2-cal-{split}-s{seed}-e{index:04d}"
            visible.append(
                {
                    "schema_version": "synthetic-sensor-calibration-visible-v2",
                    "calibration_id": calibration_id,
                    "split": split,
                    "candidate_intervals": [
                        {"start": int(start), "end": int(end)} for start, end in windows
                    ],
                    "raw_stream": raw,
                }
            )
            oracle.append(
                {
                    "schema_version": "synthetic-sensor-calibration-oracle-v2",
                    "calibration_id": calibration_id,
                    "calibration_seed": seed,
                    "event_owners": owners,
                    "true_event_boundaries": [list(value) for value in windows],
                    "sensor_informativeness": float(factor_row["sensor_informativeness"]),
                    "factors": {
                        key: float(value) if not isinstance(value, int) else int(value)
                        for key, value in factor_row.items()
                    },
                    **raw_oracle,
                }
            )
    forbidden_visible_tokens = ("word", "lexical", "referent", "target", "mapping", "grounded")
    checks = {
        "no_lexical_or_referent_fields": all(
            not any(token in canonical_json_lower for token in forbidden_visible_tokens)
            for canonical_json_lower in (
                str(sorted(record.keys())).lower() for record in visible
            )
        ),
        "raw_stream_shape_valid": all(
            len(record["raw_stream"]["imu"]) == int(config["stream"]["samples"])
            and all(len(row) == 6 for row in record["raw_stream"]["imu"])
            for record in visible
        ),
        "oracle_separate": all("event_owners" not in record for record in visible),
        "all_seeds_expected": {row["calibration_seed"] for row in oracle} == set(seeds),
    }
    if not all(checks.values()):
        raise RuntimeError(f"calibration generation audit failed: {checks}")
    provenance = {
        "schema_version": "synthetic-sensor-calibration-provenance-v2",
        "protocol_id": config["protocol"]["id"],
        "split": split,
        "seeds": list(seeds),
        "episodes_per_seed": per_seed,
        "record_count": len(visible),
        "visible_digest": canonical_digest(visible),
        "oracle_digest": canonical_digest(oracle),
        "lexical_targets_present": False,
        "referent_targets_present": False,
        "randomized_word_mappings_present": False,
        "checks": checks,
        "valid": all(checks.values()),
    }
    return CalibrationData(
        split=split,
        seeds=seeds,
        visible_records=tuple(visible),
        oracle_records=tuple(oracle),
        provenance=provenance,
    )
