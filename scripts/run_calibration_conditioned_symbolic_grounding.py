#!/usr/bin/env python3
"""Fixture-only calibration-conditioned symbolic grounding runner.

This is deliberately a small nonempirical construction harness.  ``smoke``
binds aggregate uncertainty ranges, builds matched synthetic episodes, and
exercises the dual noun/object and verb/action corrective mechanism.  ``run``
accepts the exact planned scientific command line but fails closed: the
scientific outcome is neither implemented nor authorized in this version.

No ChildLens row, lexical item, identifier, model payload, checkpoint, or
effect estimate is accepted by the smoke path.  Evaluation truth is generated
by the simulator and side cues are rejected at evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT
    / "docs/nursery_program_convergence_v1/"
    "frozen_calibration_conditioned_sensitivity_protocol.json"
)
DEFAULT_CALIBRATION = (
    ROOT
    / "output/nursery_program_convergence_v1/"
    "childlens_pseudo_calibration_receipt.json"
)
DEFAULT_FIXTURE_RANGES = (
    ROOT / "tests/fixtures/calibration_conditioned_symbolic_ranges_v1.json"
)
DEFAULT_OUTCOME_ROOT = (
    ROOT / "output/nursery_calibration_conditioned_symbolic_grounding"
)

VERSION = "nursery-calibration-conditioned-symbolic-fixture-v1"
FIXTURE_RANGE_SCHEMA = "nursery-calibration-range-fixture-v1"
PUBLIC_CALIBRATION_SCHEMA = "nursery-childlens-pseudo-calibration-receipt-v1"
PROTOCOL_SCHEMA = "nursery-calibration-conditioned-sensitivity-v1"

ARMS = (
    "weak_vl_absent_side",
    "weak_vl_synchronized_side",
    "weak_vl_group_shuffled_side",
    "weak_vl_balanced_time_shifted_side",
    "weak_vl_corrupted_uninformative_side",
)
SCALAR_DIMENSIONS = (
    "visible_candidate",
    "null_or_irrelevant",
    "ambiguous_or_undecidable",
    "partial_or_clear_visibility",
    "noun_object_support",
    "verb_action_support",
)
CATEGORY_DIMENSIONS = {
    "candidate_multiplicity": ("zero", "one", "two", "three_or_more"),
    "lag_event_unit": (
        "lead_two_plus",
        "lead_one",
        "overlap",
        "lag_one",
        "lag_two_plus",
    ),
}
FORBIDDEN_FIXTURE_KEYS = re.compile(
    r"(?i)(participant|session|filename|media|transcript|utterance|timestamp|"
    r"model_prediction|model_score|confidence|tokenizer|checkpoint|effect_size)"
)
ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'])(?:/Users/|/home/|file://)")
PLANNED_COMMAND = (
    "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
    "scripts/run_calibration_conditioned_symbolic_grounding.py run "
    "--protocol docs/nursery_program_convergence_v1/"
    "frozen_calibration_conditioned_sensitivity_protocol.json "
    "--calibration-receipt output/nursery_program_convergence_v1/"
    "childlens_pseudo_calibration_receipt.json "
    "--output-root output/nursery_calibration_conditioned_symbolic_grounding "
    "--jobs 4"
)


class SymbolicRunnerError(RuntimeError):
    """Fixed diagnostic safe for a public fixture invocation."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise SymbolicRunnerError(code)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("E_CANONICAL_JSON")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path, maximum: int = 16 * 1024 * 1024) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        if not payload or len(payload) > maximum:
            _fail("E_INPUT_JSON")
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_INPUT_JSON")
    if not isinstance(value, Mapping):
        _fail("E_INPUT_JSON")
    return value


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _interval(value: Any, code: str) -> tuple[float, float]:
    if (
        not isinstance(value, Mapping)
        or value.get("status") != "PUBLISHED"
        or not isinstance(value.get("interval"), list)
        or len(value["interval"]) != 2
    ):
        _fail(code)
    lower, upper = value["interval"]
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, (int, float))
        or not isinstance(upper, (int, float))
        or not math.isfinite(float(lower))
        or not math.isfinite(float(upper))
        or not 0.0 <= float(lower) <= float(upper) <= 1.0
    ):
        _fail(code)
    return float(lower), float(upper)


@dataclass(frozen=True)
class RangeFamily:
    name: str
    scalar: Mapping[str, tuple[float, float]]
    categories: Mapping[str, Mapping[str, tuple[float, float]]]


@dataclass(frozen=True)
class CalibrationBinding:
    schema_version: str
    status: str
    digest: str
    fixture_only: bool
    gate_passed: bool
    families: tuple[RangeFamily, ...]
    suppressed_dimensions: tuple[str, ...]


def bind_calibration_ranges(
    document: Mapping[str, Any], *, fixture_only: bool
) -> CalibrationBinding:
    """Bind aggregate ranges without accepting row-level or lexical payloads."""

    schema = document.get("schema_version")
    if fixture_only:
        if (
            schema != FIXTURE_RANGE_SCHEMA
            or document.get("scope") != "SYNTHETIC_FIXTURE_ONLY"
            or document.get("empirical_ancestry") is not False
            or document.get("scientific_outcome_allowed") is not False
        ):
            _fail("E_FIXTURE_BOUNDARY")
        if any(FORBIDDEN_FIXTURE_KEYS.search(key) for key in _walk_keys(document)):
            _fail("E_FIXTURE_RESTRICTED_FIELD")
        encoded = _canonical(document).decode("utf-8")
        if ABSOLUTE_PATH.search(encoded):
            _fail("E_FIXTURE_RESTRICTED_FIELD")
    elif schema != PUBLIC_CALIBRATION_SCHEMA:
        _fail("E_CALIBRATION_SCHEMA")

    ranges = document.get("calibration_ranges")
    if not isinstance(ranges, Mapping):
        _fail("E_CALIBRATION_SCHEMA")
    visible = ranges.get("visible_candidate")
    if not isinstance(visible, Mapping) or not visible:
        _fail("E_CALIBRATION_SCHEMA")
    family_names = tuple(sorted(str(key) for key in visible))
    families: list[RangeFamily] = []
    suppressed: set[str] = set()
    for family_name in family_names:
        scalar: dict[str, tuple[float, float]] = {}
        categories: dict[str, dict[str, tuple[float, float]]] = {}
        for dimension in SCALAR_DIMENSIONS:
            family_values = ranges.get(dimension)
            cell = family_values.get(family_name) if isinstance(family_values, Mapping) else None
            if isinstance(cell, Mapping) and cell.get("status") == "PUBLISHED":
                scalar[dimension] = _interval(cell, "E_CALIBRATION_INTERVAL")
            elif not fixture_only and isinstance(cell, Mapping) and cell.get("status") == "SUPPRESSED_K5":
                scalar[dimension] = (0.0, 1.0)
                suppressed.add(f"{family_name}:{dimension}")
            else:
                _fail("E_CALIBRATION_INTERVAL")
        for dimension, labels in CATEGORY_DIMENSIONS.items():
            category_value = ranges.get(dimension)
            if not isinstance(category_value, Mapping):
                _fail("E_CALIBRATION_INTERVAL")
            category_cells: dict[str, tuple[float, float]] = {}
            for label in labels:
                label_value = category_value.get(label)
                cell = label_value.get(family_name) if isinstance(label_value, Mapping) else None
                if isinstance(cell, Mapping) and cell.get("status") == "PUBLISHED":
                    category_cells[label] = _interval(cell, "E_CALIBRATION_INTERVAL")
                elif not fixture_only and isinstance(cell, Mapping) and cell.get("status") == "SUPPRESSED_K5":
                    category_cells[label] = (0.0, 1.0)
                    suppressed.add(f"{family_name}:{dimension}:{label}")
                else:
                    _fail("E_CALIBRATION_INTERVAL")
            categories[dimension] = category_cells
        families.append(RangeFamily(family_name, scalar, categories))

    status = str(document.get("status"))
    if fixture_only:
        gate_passed = True
    else:
        public = document.get("public_export")
        ancestry = document.get("ancestry")
        gate_passed = (
            status == "CALIBRATION_PASS"
            and not suppressed
            and isinstance(public, Mapping)
            and public.get("raw_counts") is False
            and public.get("paths") is False
            and public.get("identifiers") is False
            and public.get("transcript_or_lexical_content") is False
            and isinstance(ancestry, Mapping)
            and ancestry.get("AEA_empirical_ancestry") is False
            and ancestry.get("BabyView_empirical_ancestry") is False
            and ancestry.get("cross_corpus_pooling") is False
            and document.get("pseudo_labels_are_ground_truth") is False
            and document.get("simulator_oracle_only_evaluation_truth") is True
            and document.get("scientific_outcome_run") is False
        )
    return CalibrationBinding(
        schema_version=str(schema),
        status=status,
        digest=_digest(document),
        fixture_only=fixture_only,
        gate_passed=gate_passed,
        families=tuple(families),
        suppressed_dimensions=tuple(sorted(suppressed)),
    )


def _midpoint(interval: tuple[float, float]) -> float:
    return (interval[0] + interval[1]) / 2.0


def _category_distribution(values: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    midpoints = {key: _midpoint(value) for key, value in values.items()}
    total = sum(midpoints.values())
    if total <= 0:
        return {key: 1.0 / len(midpoints) for key in midpoints}
    return {key: value / total for key, value in midpoints.items()}


def parameter_point(family: RangeFamily) -> dict[str, Any]:
    """Fixture midpoint only; the planned outcome's seven-point design is not run."""

    candidate = _category_distribution(family.categories["candidate_multiplicity"])
    lag = _category_distribution(family.categories["lag_event_unit"])
    return {
        **{key: _midpoint(value) for key, value in family.scalar.items()},
        "candidate_multiplicity": candidate,
        "lag_event_unit": lag,
        "range_point_policy": "FIXTURE_MIDPOINT_NOT_SCIENTIFIC_MAXIMIN_DRAW",
    }


@dataclass(frozen=True)
class Episode:
    episode_id: str
    noun_token: str | None
    verb_token: str | None
    noun_options: tuple[str, ...]
    verb_options: tuple[str, ...]
    noun_truth: str
    verb_truth: str
    noun_side: tuple[float, ...]
    verb_side: tuple[float, ...]


def _fraction(seed: int, *parts: object) -> float:
    payload = "\0".join([str(seed), *(str(value) for value in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _candidate_count(distribution: Mapping[str, float]) -> int:
    expected = (
        distribution.get("one", 0.0)
        + 2.0 * distribution.get("two", 0.0)
        + 3.0 * distribution.get("three_or_more", 0.0)
    )
    return max(2, min(3, round(expected)))


def _options(
    meanings: Sequence[str], truth_index: int, candidate_count: int, position: int, visible: bool
) -> tuple[str, ...]:
    truth = meanings[truth_index]
    distractors = [meanings[(truth_index + offset) % len(meanings)] for offset in range(1, len(meanings))]
    values = ([truth] if visible else []) + distractors
    values = values[:candidate_count]
    if visible:
        values.remove(truth)
        values.insert(position % candidate_count, truth)
    return tuple(values)


def _side(options: Sequence[str], truth: str, peak_position: int) -> tuple[float, ...]:
    result = [0.15 + 0.05 * index for index in range(len(options))]
    target = options.index(truth) if truth in options else peak_position % len(options)
    result[target] = 1.0
    return tuple(result)


def generate_fixture_episodes(
    parameters: Mapping[str, Any], corpus_seed: int, *, concepts: int = 4, repetitions: int = 6
) -> tuple[Episode, ...]:
    if concepts < 3 or repetitions < 3:
        _fail("E_FIXTURE_SIZE")
    nouns = tuple(f"OBJ_{index}" for index in range(concepts))
    verbs = tuple(f"ACT_{index}" for index in range(concepts))
    candidate_count = _candidate_count(parameters["candidate_multiplicity"])
    episodes: list[Episode] = []
    for repetition in range(repetitions):
        for noun_index in range(concepts):
            for verb_index in range(concepts):
                episode_index = len(episodes)
                noun_truth = nouns[noun_index]
                verb_truth = verbs[verb_index]
                null = _fraction(corpus_seed, "null", episode_index) < float(parameters["null_or_irrelevant"])
                ambiguous = _fraction(corpus_seed, "ambiguous", episode_index) < float(parameters["ambiguous_or_undecidable"])
                noun_supported = _fraction(corpus_seed, "noun", episode_index) < float(parameters["noun_object_support"])
                verb_supported = _fraction(corpus_seed, "verb", episode_index) < float(parameters["verb_action_support"])
                visible = _fraction(corpus_seed, "visible", episode_index) < float(parameters["visible_candidate"])
                # Ambiguous/null cases remain in the matched primary episode but
                # contribute no linguistic update; no model prediction controls this.
                noun_token = None if null or ambiguous or not noun_supported else f"N_{noun_index}"
                verb_token = None if null or ambiguous or not verb_supported else f"V_{verb_index}"
                noun_position = (noun_index + repetition) % candidate_count
                verb_position = (verb_index + 2 * repetition) % candidate_count
                noun_options = _options(nouns, noun_index, candidate_count, noun_position, visible)
                verb_options = _options(verbs, verb_index, candidate_count, verb_position, visible)
                episodes.append(
                    Episode(
                        episode_id=f"E{episode_index:05d}",
                        noun_token=noun_token,
                        verb_token=verb_token,
                        noun_options=noun_options,
                        verb_options=verb_options,
                        noun_truth=noun_truth,
                        verb_truth=verb_truth,
                        noun_side=_side(noun_options, noun_truth, noun_position),
                        verb_side=_side(verb_options, verb_truth, verb_position),
                    )
                )
    return tuple(episodes)


PRIMARY_KEYS = (
    "episode_id",
    "noun_token",
    "verb_token",
    "noun_options",
    "verb_options",
)


def _primary_row(episode: Episode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "noun_token": episode.noun_token,
        "verb_token": episode.verb_token,
        "noun_options": list(episode.noun_options),
        "verb_options": list(episode.verb_options),
    }


def condition_views(episodes: Sequence[Episode]) -> Mapping[str, tuple[dict[str, Any], ...]]:
    primary = [_primary_row(episode) for episode in episodes]
    count = len(episodes)
    if count < 4:
        _fail("E_FIXTURE_SIZE")
    donor_maps = {
        "weak_vl_synchronized_side": list(range(count)),
        "weak_vl_group_shuffled_side": [(index + count // 2) % count for index in range(count)],
        "weak_vl_balanced_time_shifted_side": [index + 1 if index % 2 == 0 else index - 1 for index in range(count)],
    }
    result: dict[str, tuple[dict[str, Any], ...]] = {
        "weak_vl_absent_side": tuple(dict(row) for row in primary)
    }
    for arm, donors in donor_maps.items():
        rows: list[dict[str, Any]] = []
        for index, donor in enumerate(donors):
            row = dict(primary[index])
            row["noun_side"] = list(episodes[donor].noun_side)
            row["verb_side"] = list(episodes[donor].verb_side)
            rows.append(row)
        result[arm] = tuple(rows)
    corrupted: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        row = dict(primary[index])
        row["noun_side"] = sorted(episode.noun_side)
        row["verb_side"] = sorted(episode.verb_side, reverse=index % 2 == 0)
        corrupted.append(row)
    result["weak_vl_corrupted_uninformative_side"] = tuple(corrupted)
    if tuple(result) != ARMS:
        _fail("E_ARM_SET")
    return result


def primary_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _digest([{key: row[key] for key in PRIMARY_KEYS} for row in rows])


def side_multiset_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    values: list[float] = []
    for row in rows:
        for key in ("noun_side", "verb_side"):
            values.extend(float(value) for value in row.get(key, []))
    return _digest(sorted(values))


def _initial_scores(tokens: Iterable[str], meanings: Iterable[str], seed: int) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for token in sorted(set(tokens)):
        result[token] = {
            meaning: (_fraction(seed, "initial", token, meaning) - 0.5) * 1e-6
            for meaning in sorted(set(meanings))
        }
    return result


def fit_dual_corrective(
    rows: Sequence[Mapping[str, Any]], *, model_seed: int, corrective_weight: float = 2.5
) -> dict[str, Any]:
    """Fit factorized noun and verb maps without oracle fields or stored side state."""

    allowed = set(PRIMARY_KEYS) | {"noun_side", "verb_side"}
    if any(set(row) - allowed for row in rows):
        _fail("E_LEARNER_ORACLE_OR_EXTRA_FIELD")
    noun_tokens = [str(row["noun_token"]) for row in rows if row.get("noun_token") is not None]
    verb_tokens = [str(row["verb_token"]) for row in rows if row.get("verb_token") is not None]
    noun_meanings = [str(value) for row in rows for value in row["noun_options"]]
    verb_meanings = [str(value) for row in rows for value in row["verb_options"]]
    noun_scores = _initial_scores(noun_tokens, noun_meanings, model_seed)
    verb_scores = _initial_scores(verb_tokens, verb_meanings, model_seed + 1)
    update_count = 0
    for row in rows:
        for family, scores in (("noun", noun_scores), ("verb", verb_scores)):
            token_value = row.get(f"{family}_token")
            options = [str(value) for value in row[f"{family}_options"]]
            if token_value is None:
                # Matched compute: null/ambiguous rows still execute the same
                # candidate loop but cannot update a lexical map.
                update_count += len(options)
                continue
            token = str(token_value)
            side_raw = row.get(f"{family}_side")
            side = [0.0] * len(options) if side_raw is None else [float(value) for value in side_raw]
            if len(side) != len(options):
                _fail("E_SIDE_BUNDLE")
            mean_side = sum(side) / len(side)
            for option, signal in zip(options, side):
                scores[token][option] += 1.0 + corrective_weight * (signal - mean_side)
                update_count += 1
    return {
        "schema_version": "nursery-dual-symbolic-map-fixture-v1",
        "noun_scores": noun_scores,
        "verb_scores": verb_scores,
        "model_seed": model_seed,
        "candidate_update_count": update_count,
        "side_state_serialized": False,
        "oracle_fields_seen": False,
    }


def _top(scores: Mapping[str, float]) -> tuple[str, bool]:
    best = max(scores.values())
    winners = sorted(key for key, value in scores.items() if abs(value - best) <= 1e-12)
    return winners[0], len(winners) == 1


def evaluation_items(concepts: int = 4) -> tuple[dict[str, Any], ...]:
    nouns = [f"OBJ_{index}" for index in range(concepts)]
    verbs = [f"ACT_{index}" for index in range(concepts)]
    return tuple(
        {
            "noun_token": f"N_{index}",
            "verb_token": f"V_{index}",
            "noun_options": nouns,
            "verb_options": verbs,
            "noun_truth": nouns[index],
            "verb_truth": verbs[index],
        }
        for index in range(concepts)
    )


EVALUATION_KEYS = frozenset(
    {
        "noun_token",
        "verb_token",
        "noun_options",
        "verb_options",
        "noun_truth",
        "verb_truth",
    }
)


def evaluate_without_side(
    model: Mapping[str, Any], items: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    if any(set(item) != EVALUATION_KEYS for item in items):
        _fail("E_EVALUATION_SIDE_OR_EXTRA_FIELD")
    if model.get("side_state_serialized") is not False:
        _fail("E_MODEL_SIDE_STATE")
    noun_correct = verb_correct = 0
    for item in items:
        noun, _ = _top(model["noun_scores"][str(item["noun_token"])])
        verb, _ = _top(model["verb_scores"][str(item["verb_token"])])
        noun_correct += noun == item["noun_truth"]
        verb_correct += verb == item["verb_truth"]
    count = len(items)
    return {
        "cue_free_heldout_noun_object_macro_top1": noun_correct / count,
        "cue_free_heldout_verb_action_macro_top1": verb_correct / count,
    }


def _side_only_accuracy(episodes: Sequence[Episode], family: str) -> float:
    groups: dict[int, Counter[str]] = defaultdict(Counter)
    for episode in episodes:
        values = episode.noun_side if family == "noun" else episode.verb_side
        truth = episode.noun_truth if family == "noun" else episode.verb_truth
        signature = max(range(len(values)), key=lambda index: values[index])
        groups[signature][truth] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(episodes)


def corrective_microcase() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _ in range(2):
        rows.append(
            {
                "episode_id": f"M{len(rows)}",
                "noun_token": "N_0",
                "verb_token": "V_0",
                "noun_options": ["OBJ_0", "OBJ_1"],
                "verb_options": ["ACT_0", "ACT_1"],
                "noun_side": [1.0, 0.0],
                "verb_side": [1.0, 0.0],
            }
        )
    rows.append(
        {
            "episode_id": "M2",
            "noun_token": "N_0",
            "verb_token": "V_0",
            "noun_options": ["OBJ_1", "OBJ_2"],
            "verb_options": ["ACT_1", "ACT_2"],
            "noun_side": [0.0, 1.0],
            "verb_side": [0.0, 1.0],
        }
    )
    absent = [{key: value for key, value in row.items() if not key.endswith("_side")} for row in rows]
    baseline = fit_dual_corrective(absent, model_seed=101, corrective_weight=2.5)
    synchronized = fit_dual_corrective(rows, model_seed=101, corrective_weight=2.5)
    disconnected = fit_dual_corrective(rows, model_seed=101, corrective_weight=0.0)
    result: dict[str, Any] = {}
    for family, correct in (("noun", "OBJ_0"), ("verb", "ACT_0")):
        token = "N_0" if family == "noun" else "V_0"
        before, before_unique = _top(baseline[f"{family}_scores"][token])
        after, after_unique = _top(synchronized[f"{family}_scores"][token])
        disconnected_top, _ = _top(disconnected[f"{family}_scores"][token])
        result[family] = {
            "baseline_top": before,
            "baseline_unique": before_unique,
            "synchronized_top": after,
            "synchronized_unique": after_unique,
            "wrong_or_tied_to_correct": (before != correct or not before_unique)
            and after == correct
            and after_unique,
            "disconnected_equals_baseline_top": disconnected_top == before,
        }
    return result


def _strong_alignment_ceiling(model_seed: int) -> dict[str, float]:
    rows = []
    for index in range(4):
        rows.append(
            {
                "episode_id": f"C{index}",
                "noun_token": f"N_{index}",
                "verb_token": f"V_{index}",
                "noun_options": [f"OBJ_{index}"],
                "verb_options": [f"ACT_{index}"],
            }
        )
    model = fit_dual_corrective(rows, model_seed=model_seed)
    return evaluate_without_side(model, evaluation_items())


def execute_fixture_smoke(
    binding: CalibrationBinding,
    *,
    corpus_seeds: Sequence[int] = (910001, 910003),
    model_seeds: Sequence[int] = (920001, 920003),
) -> dict[str, Any]:
    if not binding.fixture_only or not binding.gate_passed:
        _fail("E_SMOKE_REQUIRES_SYNTHETIC_RANGES")
    bundles: list[dict[str, Any]] = []
    for family in binding.families:
        parameters = parameter_point(family)
        for corpus_seed in corpus_seeds:
            episodes = generate_fixture_episodes(parameters, corpus_seed)
            views = condition_views(episodes)
            digests = {arm: primary_digest(rows) for arm, rows in views.items()}
            side_digests = {
                arm: side_multiset_digest(rows)
                for arm, rows in views.items()
                if arm != "weak_vl_absent_side"
            }
            for model_seed in model_seeds:
                arm_results: dict[str, Any] = {}
                update_counts: set[int] = set()
                for arm in ARMS:
                    model = fit_dual_corrective(views[arm], model_seed=model_seed)
                    update_counts.add(int(model["candidate_update_count"]))
                    arm_results[arm] = evaluate_without_side(model, evaluation_items())
                microcase = corrective_microcase()
                chance = 0.25
                side_probe = {
                    "noun_accuracy": _side_only_accuracy(episodes, "noun"),
                    "verb_accuracy": _side_only_accuracy(episodes, "verb"),
                    "chance": chance,
                    "maximum_allowed": chance + 0.02,
                }
                bundle_gates = {
                    "exact_five_arms": tuple(arm_results) == ARMS,
                    "matched_primary_episode_digest": len(set(digests.values())) == 1,
                    "matched_side_value_multiset": len(set(side_digests.values())) == 1,
                    "matched_update_count": len(update_counts) == 1,
                    "evaluation_side_fields_absent": True,
                    "model_side_state_absent": True,
                    "noun_corrective_wrong_or_tied_mapping_changed": microcase["noun"]["wrong_or_tied_to_correct"],
                    "verb_corrective_wrong_or_tied_mapping_changed": microcase["verb"]["wrong_or_tied_to_correct"],
                    "event_selection_disconnection_removes_correction": microcase["noun"]["disconnected_equals_baseline_top"]
                    and microcase["verb"]["disconnected_equals_baseline_top"],
                    "side_only_noun_at_most_chance_plus_002": side_probe["noun_accuracy"] <= chance + 0.02,
                    "side_only_verb_at_most_chance_plus_002": side_probe["verb_accuracy"] <= chance + 0.02,
                    "no_AEA_empirical_ancestry": True,
                    "no_BabyView_empirical_ancestry": True,
                    "no_ChildLens_empirical_ancestry": True,
                    "simulator_oracle_only_evaluation": True,
                }
                bundles.append(
                    {
                        "range_family": family.name,
                        "corpus_seed": corpus_seed,
                        "model_seed": model_seed,
                        "paired_arm_names": list(ARMS),
                        "arm_results": arm_results,
                        "side_only_probe": side_probe,
                        "strong_alignment_ceiling": _strong_alignment_ceiling(model_seed),
                        "falsification_gates": bundle_gates,
                    }
                )
    all_gates = all(
        all(value is True for value in bundle["falsification_gates"].values())
        and all(value == 1.0 for value in bundle["strong_alignment_ceiling"].values())
        for bundle in bundles
    )
    return {
        "schema_version": VERSION,
        "status": "FIXTURE_SMOKE_PASS" if all_gates else "FIXTURE_SMOKE_FAIL",
        "scope": "NON_STUDY_SYNTHETIC_FIXTURE_ONLY",
        "calibration_binding_sha256": binding.digest,
        "range_family_count": len(binding.families),
        "paired_bundle_count": len(bundles),
        "five_matched_arms": list(ARMS),
        "dual_corrective_learning": ["NOUN_OBJECT", "VERB_ACTION"],
        "cue_withheld_at_evaluation": True,
        "side_only_leakage_falsification": True,
        "confidence_only_falsification": "REQUIRES_WRONG_OR_TIED_TO_CORRECT_TOP_MAPPING",
        "ancestry": {
            "empirical_input": False,
            "ChildLens": False,
            "AEA": False,
            "BabyView": False,
            "learner_initialization": "SCRATCH_SYMBOLIC_FIXTURE",
            "evaluation_truth": "SIMULATOR_ORACLE_ONLY",
        },
        "scientific_outcome_run": False,
        "scientific_claim_allowed": False,
        "bundles": bundles,
        "all_fixture_gates_passed": all_gates,
        "planned_outcome_command": PLANNED_COMMAND,
        "planned_outcome_command_executed": False,
    }


def validate_protocol_and_calibration(
    protocol_path: Path, calibration_path: Path
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    calibration = _read_json(calibration_path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        _fail("E_PROTOCOL_SCHEMA")
    binding = bind_calibration_ranges(calibration, fixture_only=False)
    execution = protocol.get("execution")
    if not isinstance(execution, Mapping):
        _fail("E_PROTOCOL_SCHEMA")
    return {
        "schema_version": VERSION,
        "status": "OUTCOME_BLOCKED",
        "protocol_outcome_authorized": execution.get("outcome_authorized") is True,
        "calibration_gate_passed": binding.gate_passed,
        "calibration_status": binding.status,
        "suppressed_dimension_count": len(binding.suppressed_dimensions),
        "planned_outcome_command": PLANNED_COMMAND,
        "planned_command_matches_protocol": execution.get("planned_launch_command") == PLANNED_COMMAND,
        "scientific_outcome_run": False,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run non-study synthetic fixture only")
    smoke.add_argument("--ranges", type=Path, default=DEFAULT_FIXTURE_RANGES)
    smoke.add_argument("--output", type=Path)
    validate = subparsers.add_parser("validate", help="validate planned outcome blockers only")
    validate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    validate.add_argument("--calibration-receipt", type=Path, default=DEFAULT_CALIBRATION)
    run = subparsers.add_parser("run", help="exact planned outcome interface; fail closed")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--calibration-receipt", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--jobs", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "smoke":
            binding = bind_calibration_ranges(_read_json(args.ranges), fixture_only=True)
            receipt = execute_fixture_smoke(binding)
            if args.output is not None:
                _atomic_json(args.output, receipt)
            print(json.dumps({"status": receipt["status"], "scientific_outcome_run": False}))
            return 0 if receipt["status"] == "FIXTURE_SMOKE_PASS" else 1
        validation = validate_protocol_and_calibration(
            args.protocol, args.calibration_receipt
        )
        if args.command == "validate":
            print(json.dumps(validation, sort_keys=True))
            return 0
        if args.jobs < 1 or args.jobs > 64:
            _fail("E_JOBS")
        # The interface exists so the frozen planned command is exact and
        # testable.  No output directory is created and no endpoints are run.
        if not validation["planned_command_matches_protocol"]:
            _fail("E_PLANNED_COMMAND_DRIFT")
        if not validation["protocol_outcome_authorized"]:
            _fail("E_OUTCOME_NOT_AUTHORIZED")
        if not validation["calibration_gate_passed"]:
            _fail("E_CALIBRATION_GATE")
        _fail("E_SCIENTIFIC_EXECUTION_NOT_IMPLEMENTED")
    except SymbolicRunnerError as exc:
        print(exc.code, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
