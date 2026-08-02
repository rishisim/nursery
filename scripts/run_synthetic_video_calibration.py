#!/usr/bin/env python3
"""Governed bounded C calibration and aggregate-conditioned episode planning."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.request

from nursery_egobaby_preflight.contract import compact_aggregate_json


TERMINAL_FIELDS = frozenset(
    {
        "status",
        "axis_count",
        "joint_count",
        "sampled_frame_count",
        "grounding_event_count",
        "missing_axis_count",
        "suppressed_cell_count",
        "episode_plan_count",
        "scheduled_frame_count",
        "decode_failure_count",
        "calibration_commitment_sha256",
        "episode_plan_commitment_sha256",
        "extractor_repair_commitment_sha256",
    }
)
TERMINAL_HASH_FIELDS = frozenset(
    {
        "calibration_commitment_sha256",
        "episode_plan_commitment_sha256",
        "extractor_repair_commitment_sha256",
    }
)
PUBLIC_PREP_FIELDS = frozenset(
    {
        "status",
        "model_file_count",
        "fixture_file_count",
        "extractor_repair_commitment_sha256",
    }
)
PUBLIC_QUALIFICATION_FIELDS = frozenset(
    {
        "status",
        "fixture_count",
        "activity_correct_count",
        "expected_object_hit_count",
        "hand_positive_hit_count",
        "hand_positive_count",
        "hand_negative_correct_count",
        "hand_negative_count",
        "proxy_complete_count",
        "invalid_box_count",
        "model_file_count",
        "fixture_file_count",
        "extractor_repair_commitment_sha256",
        "public_qualification_commitment_sha256",
    }
)
PUBLIC_PREP_HASH_FIELDS = frozenset({"extractor_repair_commitment_sha256"})
PUBLIC_QUALIFICATION_HASH_FIELDS = frozenset(
    {
        "extractor_repair_commitment_sha256",
        "public_qualification_commitment_sha256",
    }
)
ACTIVITY_CANDIDATE_FIELDS = frozenset(
    {
        "status",
        "candidate_id",
        "partition",
        "item_count",
        "failure_count",
        "invalid_record_count",
        "silent_truncation_count",
        "external_call_count",
        "peak_vram_gib",
        "median_item_runtime_seconds",
        "candidate_output_commitment_sha256",
    }
)
ACTIVITY_CANDIDATE_HASH_FIELDS = frozenset(
    {"candidate_output_commitment_sha256"}
)
ACTIVITY_PREP_FIELDS = frozenset(
    {
        "status",
        "candidate_count",
        "weight_file_count",
        "code_repository_count",
        "public_item_count",
        "installed_distribution_count",
        "dependency_manifest_commitment_sha256",
    }
)
ACTIVITY_PREP_HASH_FIELDS = frozenset(
    {"dependency_manifest_commitment_sha256"}
)
ACTIVITY_SELECTION_FIELDS = frozenset(
    {
        "status",
        "candidate_count",
        "eligible_candidate_count",
        "winner_candidate_id",
        "winner_macro_f1",
        "winner_worst_class_recall",
        "winner_nonabstained_coverage",
        "winner_temporal_shuffled_positive_fraction",
        "winner_temporal_repeated_positive_fraction",
        "activity_selection_commitment_sha256",
    }
)
ACTIVITY_SELECTION_HASH_FIELDS = frozenset(
    {"activity_selection_commitment_sha256"}
)
ACTIVITY_AXIS = "activity_context_mixture"
VISUAL_AXIS = "egocentric_visual_regime"
SCENE_AXIS = "scene_complexity"
HAND_AXIS = "hand_object_action_structure"
TEMPORAL_AXIS = "temporal_continuity_and_recurrence"
LANGUAGE_AXIS = "language_environment"
GROUNDING_AXIS = "audiovisual_grounding_opportunity"
DIVERSITY_AXIS = "diversity_and_heterogeneity"
AXES = (
    ACTIVITY_AXIS,
    VISUAL_AXIS,
    SCENE_AXIS,
    HAND_AXIS,
    TEMPORAL_AXIS,
    LANGUAGE_AXIS,
    GROUNDING_AXIS,
    DIVERSITY_AXIS,
)
AXIS_STATUS_FIELDS = frozenset(
    {"status"}
    | {f"axis_{index}_status" for index in range(1, len(AXES) + 1)}
    | {f"axis_{index}_missing_fraction" for index in range(1, len(AXES) + 1)}
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _activity_config(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        value = cfg["calibration_C"]["extractor"][
            "activity_checkpoint_selection_amendment"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_ACTIVITY_SELECTION_NOT_FROZEN") from error
    if value.get("status") != "FROZEN_BEFORE_EMPIRICAL_CANDIDATE_OUTCOMES":
        raise RuntimeError("E_ACTIVITY_SELECTION_NOT_FROZEN")
    candidates = value.get("bounded_candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 4:
        raise RuntimeError("E_ACTIVITY_CANDIDATE_SET")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if any(not isinstance(candidate_id, str) for candidate_id in candidate_ids):
        raise RuntimeError("E_ACTIVITY_CANDIDATE_ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("E_ACTIVITY_CANDIDATE_DUPLICATE")
    if value["development_comparison"]["candidate_count"] != len(candidates):
        raise RuntimeError("E_ACTIVITY_CANDIDATE_COUNT")
    return value


def _activity_candidate(
    amendment: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    matches = [
        value
        for value in amendment["bounded_candidates"]
        if value["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError("E_UNREGISTERED_ACTIVITY_CANDIDATE")
    return matches[0]


def _activity_labels(cfg: dict[str, Any]) -> list[str]:
    prompts = cfg["calibration_C"]["extractor"]["coverage_repair"][
        "prompt_ensembles"
    ]["activity"]
    labels = list(prompts)
    if len(labels) != 8 or len(labels) != len(set(labels)):
        raise RuntimeError("E_ACTIVITY_LABEL_SET")
    return labels


def _verify_activity_manifest(
    path: Path, amendment: dict[str, Any], media_root: Path | None = None
) -> dict[str, Any]:
    if not path.is_file() or file_digest(path) != amendment[
        "public_activity_fixture"
    ]["manifest_commitment_sha256"]:
        raise RuntimeError("E_ACTIVITY_MANIFEST_COMMITMENT")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("E_ACTIVITY_MANIFEST_SCHEMA")
    if manifest.get("seed") != amendment["public_activity_fixture"]["seed"]:
        raise RuntimeError("E_ACTIVITY_MANIFEST_SEED")
    partitions = manifest.get("partitions")
    if set(partitions or {}) != {"development", "holdout"}:
        raise RuntimeError("E_ACTIVITY_MANIFEST_PARTITIONS")
    expected_counts = {
        "development": amendment["public_activity_fixture"]["development_items"],
        "holdout": amendment["public_activity_fixture"]["holdout_items"],
    }
    seen_ids: set[str] = set()
    subjects: dict[str, set[str]] = {}
    labels = set(manifest.get("label_code_map", {}))
    if len(labels) != 8:
        raise RuntimeError("E_ACTIVITY_MANIFEST_LABELS")
    for partition, expected_count in expected_counts.items():
        rows = partitions[partition]
        if len(rows) != expected_count:
            raise RuntimeError("E_ACTIVITY_MANIFEST_COUNT")
        subjects[partition] = set()
        for row in rows:
            if set(row) != {
                "id",
                "labels",
                "length_seconds",
                "media_bytes",
                "media_sha256",
                "source",
                "subject",
            }:
                raise RuntimeError("E_ACTIVITY_MANIFEST_ROW_SCHEMA")
            if row["id"] in seen_ids or not set(row["labels"]).issubset(labels):
                raise RuntimeError("E_ACTIVITY_MANIFEST_ROW_VALUE")
            if not row["labels"] or not math.isfinite(float(row["length_seconds"])):
                raise RuntimeError("E_ACTIVITY_MANIFEST_ROW_VALUE")
            seen_ids.add(row["id"])
            subjects[partition].add(row["subject"])
            if media_root is not None:
                media = media_root / "CharadesEgo_v1_480" / f"{row['id']}.mp4"
                if (
                    not media.is_file()
                    or media.stat().st_size != row["media_bytes"]
                    or file_digest(media) != row["media_sha256"]
                ):
                    raise RuntimeError("E_ACTIVITY_MEDIA_COMMITMENT")
    if subjects["development"].intersection(subjects["holdout"]):
        raise RuntimeError("E_ACTIVITY_SUBJECT_OVERLAP")
    return manifest


def _activity_fold(subject: str, seed: int) -> int:
    return int(digest([seed, "fold", subject])[:16], 16) % 4


def _deterministic_nonidentity_permutation(
    count: int, seed: int, public_id: str
) -> list[int]:
    if count < 2:
        raise ValueError("temporal control needs at least two frames")
    order = list(range(count))
    generator = random.Random(int(digest([seed, "temporal_shuffle", public_id])[:16], 16))
    generator.shuffle(order)
    if order == list(range(count)):
        order = order[1:] + order[:1]
    return order


def _finite_vector(value: Any, width: int, error_code: str) -> list[float]:
    if not isinstance(value, list) or len(value) != width:
        raise RuntimeError(error_code)
    output = [float(item) for item in value]
    if not all(math.isfinite(item) for item in output):
        raise RuntimeError(error_code)
    return output


def _binary_f1(y_true: list[bool], y_pred: list[bool]) -> float:
    true_positive = sum(expected and predicted for expected, predicted in zip(y_true, y_pred, strict=True))
    false_positive = sum(not expected and predicted for expected, predicted in zip(y_true, y_pred, strict=True))
    false_negative = sum(expected and not predicted for expected, predicted in zip(y_true, y_pred, strict=True))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _classification_metrics(
    expected: list[set[str]], predicted: list[set[str]], labels: list[str]
) -> dict[str, float]:
    f1_values = []
    recall_values = []
    for label in labels:
        truth = [label in row for row in expected]
        guesses = [label in row for row in predicted]
        f1_values.append(_binary_f1(truth, guesses))
        positive_count = sum(truth)
        recall_values.append(
            sum(want and got for want, got in zip(truth, guesses, strict=True))
            / positive_count
            if positive_count
            else 0.0
        )
    return {
        "macro_f1": sum(f1_values) / len(f1_values),
        "worst_class_recall": min(recall_values),
        "nonabstained_coverage": sum(bool(row) for row in predicted)
        / len(predicted),
    }


def _threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        raise RuntimeError("E_ACTIVITY_THRESHOLD_VALUES")
    scale = max(1.0, max(abs(value) for value in unique))
    epsilon = scale * 1e-9
    return [unique[0] - epsilon] + [
        (first + second) / 2 for first, second in zip(unique, unique[1:])
    ] + [unique[-1] + epsilon]


def _choose_label_threshold(values: list[float], truth: list[bool]) -> float:
    ranked = []
    for threshold in _threshold_candidates(values):
        score = _binary_f1(truth, [value >= threshold for value in values])
        ranked.append((score, threshold))
    return max(ranked, key=lambda item: (item[0], item[1]))[1]


def _predict_score_rows(
    rows: list[list[float]], labels: list[str], thresholds: list[float], margin: float
) -> list[set[str]]:
    return [
        {
            label
            for label, score, threshold in zip(labels, row, thresholds, strict=True)
            if score >= threshold + margin
        }
        for row in rows
    ]


def _choose_abstention_margin(
    rows: list[list[float]],
    expected: list[set[str]],
    labels: list[str],
    thresholds: list[float],
    margins: list[float],
    coverage_min: float,
) -> float | None:
    eligible = []
    for margin in margins:
        predictions = _predict_score_rows(rows, labels, thresholds, margin)
        metrics = _classification_metrics(expected, predictions, labels)
        if metrics["nonabstained_coverage"] >= coverage_min:
            eligible.append((metrics["macro_f1"], margin))
    return max(eligible, key=lambda item: (item[0], item[1]))[1] if eligible else None


def _robust_parameters(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(rows[0])
    centers = []
    scales = []
    for index in range(width):
        values = [row[index] for row in rows]
        center = statistics.median(values)
        mad = statistics.median(abs(value - center) for value in values)
        centers.append(center)
        scales.append(1.4826 * mad + 1e-6)
    return centers, scales


def _robust_transform(
    rows: list[list[float]], centers: list[float], scales: list[float]
) -> list[list[float]]:
    return [
        [
            (value - center) / scale
            for value, center, scale in zip(row, centers, scales, strict=True)
        ]
        for row in rows
    ]


def _fit_probe_scores(
    train_embeddings: list[list[float]],
    train_expected: list[set[str]],
    score_embeddings: dict[str, list[list[float]]],
    labels: list[str],
    recipe: dict[str, Any],
) -> tuple[dict[str, list[list[float]]], list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression

    outputs = {name: [[] for _ in values] for name, values in score_embeddings.items()}
    models = []
    for label in labels:
        truth = [int(label in row) for row in train_expected]
        if len(set(truth)) != 2:
            raise RuntimeError("E_ACTIVITY_PROBE_CLASS_SUPPORT")
        model = LogisticRegression(
            solver=recipe["solver"],
            C=float(recipe["C"]),
            class_weight=recipe["class_weight"],
            max_iter=int(recipe["max_iter"]),
            tol=float(recipe["tolerance"]),
            random_state=int(recipe["seed"]),
        )
        model.fit(train_embeddings, truth)
        for name, values in score_embeddings.items():
            probabilities = model.predict_proba(values)[:, 1].tolist()
            for row, probability in zip(outputs[name], probabilities, strict=True):
                row.append(float(probability))
        models.append(
            {
                "label": label,
                "coefficient": [float(value) for value in model.coef_[0]],
                "intercept": float(model.intercept_[0]),
                "classes": [int(value) for value in model.classes_],
            }
        )
    return outputs, models


def _candidate_output_rows(
    output: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    candidate_id: str,
    labels: list[str],
) -> list[dict[str, Any]]:
    if (
        output.get("schema_version") != 1
        or output.get("status") != "COMPLETE"
        or output.get("candidate_id") != candidate_id
        or output.get("partition") != "development"
        or output.get("failure_count") != 0
        or output.get("invalid_record_count") != 0
        or output.get("silent_truncation_count") != 0
        or output.get("external_call_count") != 0
    ):
        raise RuntimeError("E_ACTIVITY_CANDIDATE_OUTPUT_STATUS")
    commitment = output.get("candidate_output_commitment_sha256")
    payload = dict(output)
    payload.pop("candidate_output_commitment_sha256", None)
    if not isinstance(commitment, str) or digest(payload) != commitment:
        raise RuntimeError("E_ACTIVITY_CANDIDATE_OUTPUT_COMMITMENT")
    rows = output.get("rows")
    if not isinstance(rows, list) or len(rows) != len(manifest_rows):
        raise RuntimeError("E_ACTIVITY_CANDIDATE_OUTPUT_ROWS")
    expected_by_id = {row["id"]: row for row in manifest_rows}
    if len(expected_by_id) != len(manifest_rows):
        raise RuntimeError("E_ACTIVITY_CANDIDATE_OUTPUT_ROWS")
    seen: set[str] = set()
    is_probe = candidate_id == "vjepa2_vitl_public_probe"
    for row in rows:
        public_id = row.get("id")
        expected = expected_by_id.get(public_id)
        if expected is None or public_id in seen:
            raise RuntimeError("E_ACTIVITY_CANDIDATE_OUTPUT_ID")
        seen.add(public_id)
        if row.get("subject") != expected["subject"] or row.get("labels") != expected["labels"]:
            raise RuntimeError("E_ACTIVITY_CANDIDATE_OUTPUT_ALIGNMENT")
        key = "embedding" if is_probe else "scores"
        width = None if is_probe else len(labels)
        for control in ("ordered", "shuffled", "repeated_center"):
            value = row.get(control, {}).get(key)
            if is_probe:
                if not isinstance(value, list) or not value:
                    raise RuntimeError("E_ACTIVITY_CANDIDATE_OUTPUT_VECTOR")
                _finite_vector(value, len(value), "E_ACTIVITY_CANDIDATE_OUTPUT_VECTOR")
            else:
                _finite_vector(value, int(width), "E_ACTIVITY_CANDIDATE_OUTPUT_VECTOR")
        if int(row.get("decoded_frame_count", -1)) != int(row.get("required_frame_count", -2)):
            raise RuntimeError("E_ACTIVITY_CANDIDATE_SILENT_TRUNCATION")
    return rows


def _crossfit_activity_candidate(
    output: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    amendment: dict[str, Any],
    labels: list[str],
) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    rows = _candidate_output_rows(output, manifest_rows, candidate_id, labels)
    seed = amendment["public_activity_fixture"]["seed"]
    folds = [_activity_fold(row["subject"], seed) for row in rows]
    expected = [set(row["labels"]) for row in rows]
    prediction_by_index: dict[int, set[str]] = {}
    temporal_ordered: dict[int, list[float]] = {}
    temporal_shuffled: dict[int, list[float]] = {}
    temporal_repeated: dict[int, list[float]] = {}
    calibration_records = []
    calibration_failure_count = 0
    is_probe = candidate_id == "vjepa2_vitl_public_probe"
    for fold in range(4):
        train_indices = [index for index, value in enumerate(folds) if value != fold]
        test_indices = [index for index, value in enumerate(folds) if value == fold]
        if not train_indices or not test_indices:
            raise RuntimeError("E_ACTIVITY_FOLD_EMPTY")
        train_expected = [expected[index] for index in train_indices]
        if is_probe:
            embedding_keys = {
                "train": [rows[index]["ordered"]["embedding"] for index in train_indices],
                "test_ordered": [rows[index]["ordered"]["embedding"] for index in test_indices],
                "test_shuffled": [rows[index]["shuffled"]["embedding"] for index in test_indices],
                "test_repeated": [rows[index]["repeated_center"]["embedding"] for index in test_indices],
            }
            scored, probe_models = _fit_probe_scores(
                embedding_keys["train"],
                train_expected,
                embedding_keys,
                labels,
                candidate["probe_recipe"],
            )
            train_scores = scored["train"]
            test_ordered = scored["test_ordered"]
            test_shuffled = scored["test_shuffled"]
            test_repeated = scored["test_repeated"]
            robust = None
        else:
            raw_train = [rows[index]["ordered"]["scores"] for index in train_indices]
            centers, scales = _robust_parameters(raw_train)
            train_scores = _robust_transform(raw_train, centers, scales)
            test_ordered = _robust_transform(
                [rows[index]["ordered"]["scores"] for index in test_indices], centers, scales
            )
            test_shuffled = _robust_transform(
                [rows[index]["shuffled"]["scores"] for index in test_indices], centers, scales
            )
            test_repeated = _robust_transform(
                [rows[index]["repeated_center"]["scores"] for index in test_indices], centers, scales
            )
            robust = {"centers": centers, "scales": scales}
            probe_models = None
        thresholds = [
            _choose_label_threshold(
                [row[label_index] for row in train_scores],
                [label in truth for truth in train_expected],
            )
            for label_index, label in enumerate(labels)
        ]
        comparison = amendment["development_comparison"]
        margin = _choose_abstention_margin(
            train_scores,
            train_expected,
            labels,
            thresholds,
            [float(value) for value in comparison["abstention_margin_grid"]],
            float(comparison["eligibility_floors"]["nonabstained_coverage_min"]),
        )
        if margin is None:
            calibration_failure_count += 1
            margin = 0.0
        fold_predictions = _predict_score_rows(test_ordered, labels, thresholds, margin)
        for index, prediction, ordered, shuffled, repeated in zip(
            test_indices,
            fold_predictions,
            test_ordered,
            test_shuffled,
            test_repeated,
            strict=True,
        ):
            prediction_by_index[index] = prediction
            temporal_ordered[index] = ordered
            temporal_shuffled[index] = shuffled
            temporal_repeated[index] = repeated
        calibration_records.append(
            {
                "fold": fold,
                "train_item_count": len(train_indices),
                "test_item_count": len(test_indices),
                "thresholds": dict(zip(labels, thresholds, strict=True)),
                "margin": margin,
                "robust_standardization": robust,
                "probe_models": probe_models,
            }
        )
    predictions = [prediction_by_index[index] for index in range(len(rows))]
    metrics = _classification_metrics(expected, predictions, labels)
    temporal_labels = set(amendment["temporal_controls"]["eligible_public_labels"])
    shuffled_differences = []
    repeated_differences = []
    for index, truth in enumerate(expected):
        target_indices = [labels.index(label) for label in labels if label in truth & temporal_labels]
        if not target_indices:
            continue
        ordered_score = statistics.mean(temporal_ordered[index][position] for position in target_indices)
        shuffled_score = statistics.mean(temporal_shuffled[index][position] for position in target_indices)
        repeated_score = statistics.mean(temporal_repeated[index][position] for position in target_indices)
        shuffled_differences.append(ordered_score - shuffled_score)
        repeated_differences.append(ordered_score - repeated_score)
    if not shuffled_differences or not repeated_differences:
        raise RuntimeError("E_ACTIVITY_TEMPORAL_CONTROL_SUPPORT")
    temporal = {
        "item_count": len(shuffled_differences),
        "ordered_minus_shuffled_mean": statistics.mean(shuffled_differences),
        "ordered_minus_repeated_mean": statistics.mean(repeated_differences),
        "ordered_over_shuffled_positive_fraction": sum(value > 0 for value in shuffled_differences) / len(shuffled_differences),
        "ordered_over_repeated_positive_fraction": sum(value > 0 for value in repeated_differences) / len(repeated_differences),
    }
    floors = amendment["development_comparison"]["eligibility_floors"]
    failures = (
        int(output["failure_count"])
        + int(output["invalid_record_count"])
        + int(output["silent_truncation_count"])
        + int(output["external_call_count"])
        + calibration_failure_count
    )
    eligible = all(
        [
            metrics["macro_f1"] >= floors["macro_f1_min"],
            metrics["worst_class_recall"] >= floors["worst_class_recall_min"],
            metrics["nonabstained_coverage"] >= floors["nonabstained_coverage_min"],
            temporal["ordered_minus_shuffled_mean"]
            > floors["ordered_target_score_mean_advantage_over_shuffled_strictly_greater_than"],
            temporal["ordered_minus_repeated_mean"]
            > floors["ordered_target_score_mean_advantage_over_repeated_center_strictly_greater_than"],
            temporal["ordered_over_shuffled_positive_fraction"]
            >= floors["ordered_target_score_positive_fraction_over_shuffled_min"],
            temporal["ordered_over_repeated_positive_fraction"]
            >= floors["ordered_target_score_positive_fraction_over_repeated_center_min"],
            failures
            <= floors["crashes_silent_truncations_invalid_records_external_calls_unaccounted_failures_max"],
        ]
    )
    return {
        "candidate_id": candidate_id,
        **metrics,
        "temporal": temporal,
        "peak_vram_gib": float(output["peak_vram_gib"]),
        "median_item_runtime_seconds": float(output["median_item_runtime_seconds"]),
        "failure_count": failures,
        "eligible": eligible,
        "cross_fitted_calibration": calibration_records,
    }


def _activity_selection_key(value: dict[str, Any], precision: float) -> tuple[Any, ...]:
    digits = max(0, int(round(-math.log10(precision))))
    temporal_floor = min(
        value["temporal"]["ordered_over_shuffled_positive_fraction"],
        value["temporal"]["ordered_over_repeated_positive_fraction"],
    )
    return (
        -round(value["macro_f1"], digits),
        -round(value["worst_class_recall"], digits),
        -round(value["nonabstained_coverage"], digits),
        -round(temporal_floor, digits),
        round(value["peak_vram_gib"], digits),
        round(value["median_item_runtime_seconds"], digits),
        value["candidate_id"],
    )


def _select_activity_winner(
    candidates: list[dict[str, Any]], precision: float
) -> dict[str, Any] | None:
    eligible = [value for value in candidates if value["eligible"]]
    return min(eligible, key=lambda value: _activity_selection_key(value, precision)) if eligible else None


def _activity_run_root(public: Path) -> Path:
    return public / "runs/synthetic-video-calibration/extractor-redesign/activity-checkpoint-selection"


def _activity_code_root(public: Path, candidate_id: str) -> Path:
    names = {
        "egohod_egovideo_l_zero_shot": "EgoHOD",
        "videoprism_lvt_l_zero_shot": "videoprism",
        "vjepa2_vitl_public_probe": "vjepa2",
    }
    try:
        return public / "models/activity-code" / names[candidate_id]
    except KeyError as error:
        raise RuntimeError("E_UNREGISTERED_ACTIVITY_CANDIDATE") from error


def _activity_checkpoint_root(public: Path, candidate_id: str) -> Path:
    return public / "models/activity-checkpoints" / candidate_id


def _link_public_artifact(source: Path, target: Path, expected_sha256: str) -> None:
    if target.is_file():
        if file_digest(target) != expected_sha256:
            raise RuntimeError("E_ACTIVITY_STAGED_ARTIFACT_CONFLICT")
        return
    if not source.is_file() or file_digest(source) != expected_sha256:
        raise RuntimeError("E_ACTIVITY_STAGED_ARTIFACT")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.link(source, target)
    os.chmod(target, 0o600)


def _installed_distributions(target: Path) -> list[dict[str, str]]:
    import importlib.metadata

    values = {}
    for distribution in importlib.metadata.distributions(path=[str(target)]):
        name = str(distribution.metadata.get("Name", distribution.name)).lower()
        values[name] = str(distribution.version)
    return [
        {"name": name, "version": values[name]} for name in sorted(values)
    ]


def prepare_activity_public(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    amendment = _activity_config(cfg)
    runtime = amendment["runtime_environment"]
    manifest = _verify_activity_manifest(
        args.manifest,
        amendment,
        args.public_root / "public/charades",
    )
    staged = args.public_root / "models/activity-checkpoints"
    staged_names = {
        "egohod_egovideo_l_zero_shot": "egovideo_large_best.pt",
        "videoprism_lvt_l_zero_shot": "videoprism_lvt_large.npz",
        "vjepa2_vitl_public_probe": "vjepa2_vitl.safetensors",
    }
    for candidate in amendment["bounded_candidates"]:
        _verify_repository_commit(
            _activity_code_root(args.public_root, candidate["candidate_id"]),
            candidate["code_commit"],
        )
        _link_public_artifact(
            staged / staged_names[candidate["candidate_id"]],
            _activity_checkpoint_root(args.public_root, candidate["candidate_id"])
            / candidate["weight_file"],
            candidate["weight_sha256"],
        )
    videoprism_root = _activity_checkpoint_root(
        args.public_root, "videoprism_lvt_l_zero_shot"
    )
    _link_public_artifact(
        staged / "c4_en_sentencepiece.model",
        videoprism_root / "c4_en_sentencepiece.model",
        runtime["videoprism"]["c4_en_sentencepiece_sha256"],
    )
    vjepa_root = _activity_checkpoint_root(
        args.public_root, "vjepa2_vitl_public_probe"
    )
    _link_public_artifact(
        staged / "vjepa2_config.json",
        vjepa_root / "config.json",
        runtime["vjepa2"]["config_sha256"],
    )
    _link_public_artifact(
        staged / "vjepa2_video_preprocessor_config.json",
        vjepa_root / "video_preprocessor_config.json",
        runtime["vjepa2"]["video_preprocessor_config_sha256"],
    )
    clip_root = args.public_root / "models/activity-code/CLIP"
    if not clip_root.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                runtime["egohod"]["openai_CLIP_repository"],
                str(clip_root),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(clip_root),
                "checkout",
                "--detach",
                runtime["egohod"]["openai_CLIP_commit"],
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    _verify_repository_commit(
        clip_root, runtime["egohod"]["openai_CLIP_commit"]
    )
    dependency_root = args.public_root / "models/activity-pydeps"
    logs = _activity_run_root(args.public_root) / "logs"
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime_keys = {
        "egohod_egovideo_l_zero_shot": "egohod",
        "videoprism_lvt_l_zero_shot": "videoprism",
        "vjepa2_vitl_public_probe": "vjepa2",
    }
    environment_records = []
    for candidate in amendment["bounded_candidates"]:
        candidate_id = candidate["candidate_id"]
        target = dependency_root / candidate_id
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        report = _activity_run_root(args.public_root) / f"{candidate_id}-pip-report.json"
        log = logs / f"{candidate_id}-prepare.log"
        requirements = runtime[runtime_keys[candidate_id]][
            "direct_python_requirements"
        ]
        with log.open("w") as handle:
            os.chmod(log, 0o600)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--upgrade",
                    "--target",
                    str(target),
                    "--report",
                    str(report),
                    *requirements,
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        os.chmod(report, 0o600)
        environment_records.append(
            {
                "candidate_id": candidate_id,
                "direct_requirements": requirements,
                "pip_report_sha256": file_digest(report),
                "installed_distributions": _installed_distributions(target),
            }
        )
    dependency_manifest = {
        "schema_version": 1,
        "status": "PASS_PREPARED_NO_MODEL_INFERENCE",
        "container": runtime["container"],
        "public_fixture_commitment_sha256": amendment[
            "public_activity_fixture"
        ]["manifest_commitment_sha256"],
        "public_item_count": sum(
            len(values) for values in manifest["partitions"].values()
        ),
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "code_commit": candidate["code_commit"],
                "weight_sha256": candidate["weight_sha256"],
            }
            for candidate in amendment["bounded_candidates"]
        ],
        "environments": environment_records,
        "network_required_after_preparation": False,
        "restricted_mount_present": False,
        "model_inference_executed": False,
    }
    dependency_manifest["dependency_manifest_commitment_sha256"] = digest(
        dependency_manifest
    )
    write_private(
        _activity_run_root(args.public_root) / "dependency_manifest.json",
        dependency_manifest,
    )
    return {
        "status": "PASS_PREPARED_NO_MODEL_INFERENCE",
        "candidate_count": len(amendment["bounded_candidates"]),
        "weight_file_count": len(amendment["bounded_candidates"]),
        "code_repository_count": len(amendment["bounded_candidates"]) + 1,
        "public_item_count": dependency_manifest["public_item_count"],
        "installed_distribution_count": sum(
            len(value["installed_distributions"])
            for value in environment_records
        ),
        "dependency_manifest_commitment_sha256": dependency_manifest[
            "dependency_manifest_commitment_sha256"
        ],
    }


def _verify_activity_dependency_manifest(
    public: Path, amendment: dict[str, Any]
) -> dict[str, Any]:
    path = _activity_run_root(public) / "dependency_manifest.json"
    value = json.loads(path.read_text())
    commitment = value.pop("dependency_manifest_commitment_sha256", None)
    expected = amendment["runtime_environment"]["shared"][
        "dependency_manifest_commitment_sha256"
    ]
    if (
        not isinstance(commitment, str)
        or commitment != expected
        or digest(value) != commitment
        or value.get("status") != "PASS_PREPARED_NO_MODEL_INFERENCE"
        or value.get("restricted_mount_present") is not False
        or value.get("model_inference_executed") is not False
    ):
        raise RuntimeError("E_ACTIVITY_DEPENDENCY_MANIFEST")
    value["dependency_manifest_commitment_sha256"] = commitment
    return value


def _verify_repository_commit(path: Path, expected: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    if completed.stdout.strip() != expected:
        raise RuntimeError("E_ACTIVITY_CODE_COMMIT")
    dirty = subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet"], check=False
    )
    if dirty.returncode != 0:
        raise RuntimeError("E_ACTIVITY_CODE_DIRTY")


def _decode_uniform_activity_frames(
    media: Path, frame_count: int
) -> tuple[Any, int]:
    import decord
    import numpy as np

    reader = decord.VideoReader(str(media), ctx=decord.cpu(0), num_threads=1)
    source_count = len(reader)
    if source_count < 1:
        raise RuntimeError("E_ACTIVITY_DECODE_EMPTY")
    indices = np.linspace(0, source_count - 1, frame_count, dtype=np.int64)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    if frames.shape[0] != frame_count or frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError("E_ACTIVITY_SILENT_TRUNCATION")
    return frames, source_count


def _activity_frame_controls(frames, seed: int, public_id: str):
    import numpy as np

    order = _deterministic_nonidentity_permutation(len(frames), seed, public_id)
    shuffled = frames[np.asarray(order, dtype=np.int64)]
    repeated = np.repeat(frames[len(frames) // 2 : len(frames) // 2 + 1], len(frames), axis=0)
    return {"ordered": frames, "shuffled": shuffled, "repeated_center": repeated}


def _resize_center_crop_torch(frames, size: int):
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    tensor = transform.resize(
        tensor, size, interpolation=InterpolationMode.BICUBIC, antialias=True
    )
    return transform.center_crop(tensor, [size, size])


def _convert_egohod_flash_state(state: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    replacements = {
        ".attn.Wqkv.weight": ".attn.in_proj_weight",
        ".attn.Wqkv.bias": ".attn.in_proj_bias",
        ".mlp.fc1.weight": ".mlp.c_fc.weight",
        ".mlp.fc1.bias": ".mlp.c_fc.bias",
        ".mlp.fc2.weight": ".mlp.c_proj.weight",
        ".mlp.fc2.bias": ".mlp.c_proj.bias",
    }
    for key, value in state.items():
        target = key
        for source, replacement in replacements.items():
            if source in target:
                target = target.replace(source, replacement)
        if target in converted:
            raise RuntimeError("E_EGOHOD_STATE_KEY_COLLISION")
        converted[target] = value
    return converted


def _load_egohod_activity_adapter(
    public: Path,
    candidate: dict[str, Any],
    cfg: dict[str, Any],
    labels: list[str],
    device: str,
):
    import torch
    import torch.nn.functional as functional

    code_root = _activity_code_root(public, candidate["candidate_id"])
    clip_root = public / "models/activity-code/CLIP"
    runtime = _activity_config(cfg)["runtime_environment"]["egohod"]
    _verify_repository_commit(code_root, candidate["code_commit"])
    _verify_repository_commit(clip_root, runtime["openai_CLIP_commit"])
    sys.path.insert(0, str(clip_root))
    sys.path.insert(0, str(code_root))
    import clip
    from model.clip import CLIP
    from model.transformer import TextTransformer, VisionTransformer

    checkpoint = _activity_checkpoint_root(public, candidate["candidate_id"]) / candidate["weight_file"]
    if not checkpoint.is_file() or file_digest(checkpoint) != candidate["weight_sha256"]:
        raise RuntimeError("E_ACTIVITY_WEIGHT_COMMITMENT")
    loaded = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    state = loaded.get("state_dict", loaded)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    state = _convert_egohod_flash_state(state)
    temporal = state.get("visual.temporal_embedding")
    if temporal is None or temporal.ndim != 2:
        raise RuntimeError("E_EGOHOD_TEMPORAL_STATE")
    native_frames = int(temporal.shape[0])
    convolution = state.get("visual.conv1.weight")
    if convolution is None:
        raise RuntimeError("E_EGOHOD_CONV_STATE")
    use_fast_conv1 = convolution.ndim == 2
    vision = VisionTransformer(
        336,
        14,
        1024,
        24,
        16,
        4,
        output_dim=512,
        num_frames=native_frames,
        patch_dropout=0.0,
        drop_path_rate=0.0,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=False,
    )
    text_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        width=768,
        heads=12,
        layers=12,
        output_dim=512,
        causal_mask=True,
    )
    model = CLIP(
        embed_dim=512,
        vision_model=vision,
        text_model=text_model,
        freeze_temperature=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"E_EGOHOD_STRICT_STATE missing={len(missing)} unexpected={len(unexpected)}"
        )
    model = model.to(device).eval().half()
    prompt_groups = cfg["calibration_C"]["extractor"]["coverage_repair"][
        "prompt_ensembles"
    ]["activity"]
    flattened = [prompt for label in labels for prompt in prompt_groups[label]]
    with torch.inference_mode():
        tokens = clip.tokenize(flattened, context_length=77, truncate=True).to(device)
        text = functional.normalize(model.encode_text(tokens), dim=-1)
        text = text.reshape(len(labels), 3, -1).mean(dim=1)
        text = functional.normalize(text, dim=-1)

    def score(frames):
        tensor = _resize_center_crop_torch(frames, 336)
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
        standard = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / standard).permute(1, 0, 2, 3).unsqueeze(0)
        tensor = tensor.to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            video = model.encode_image(tensor)[0]
            video = functional.normalize(video, dim=-1)
            values = (video @ text.T).float().cpu()[0].tolist()
        return _finite_vector(values, len(labels), "E_ACTIVITY_NONFINITE_SCORE")

    return score, int(runtime["input_frames"]), "scores"


def _videoprism_token_ids(model_path: Path, prompts: list[str]):
    import numpy as np
    import sentencepiece
    from videoprism import utils

    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    rows = []
    paddings = []
    for prompt in prompts:
        ids = list(processor.encode(utils.canonicalize_text(prompt), out_type=int))
        if processor.bos_id() >= 0:
            ids = [processor.bos_id()] + ids
        ids = ids[:64]
        padding = [0.0] * len(ids) + [1.0] * (64 - len(ids))
        ids = ids + [0] * (64 - len(ids))
        rows.append(ids)
        paddings.append(padding)
    return np.asarray(rows, dtype=np.int32), np.asarray(paddings, dtype=np.float32)


def _load_videoprism_activity_adapter(
    public: Path,
    candidate: dict[str, Any],
    cfg: dict[str, Any],
    labels: list[str],
    device: str,
):
    del device
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    code_root = _activity_code_root(public, candidate["candidate_id"])
    _verify_repository_commit(code_root, candidate["code_commit"])
    sys.path.insert(0, str(code_root))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from videoprism import models as videoprism

    checkpoint_root = _activity_checkpoint_root(public, candidate["candidate_id"])
    checkpoint = checkpoint_root / candidate["weight_file"]
    runtime = _activity_config(cfg)["runtime_environment"]["videoprism"]
    tokenizer_path = checkpoint_root / "c4_en_sentencepiece.model"
    if not checkpoint.is_file() or file_digest(checkpoint) != candidate["weight_sha256"]:
        raise RuntimeError("E_ACTIVITY_WEIGHT_COMMITMENT")
    if not tokenizer_path.is_file() or file_digest(tokenizer_path) != runtime["c4_en_sentencepiece_sha256"]:
        raise RuntimeError("E_VIDEOPRISM_TOKENIZER_COMMITMENT")
    model = videoprism.get_model("videoprism_lvt_public_v1_large")
    state = videoprism.load_pretrained_weights(
        "videoprism_lvt_public_v1_large", checkpoint_path=str(checkpoint)
    )
    prompt_groups = cfg["calibration_C"]["extractor"]["coverage_repair"][
        "prompt_ensembles"
    ]["activity"]
    flattened = [prompt for label in labels for prompt in prompt_groups[label]]
    text_ids, text_paddings = _videoprism_token_ids(tokenizer_path, flattened)
    text_ids = jnp.asarray(text_ids)
    text_paddings = jnp.asarray(text_paddings)

    @jax.jit
    def forward(video):
        return model.apply(state, video, text_ids, text_paddings, train=False)[:2]

    def score(frames):
        tensor = _resize_center_crop_torch(frames, 288)
        video = jnp.asarray(tensor.permute(0, 2, 3, 1).numpy()[None, ...], dtype=jnp.float32)
        video_embedding, text_embedding = forward(video)
        values = np.asarray(video_embedding @ text_embedding.T)[0]
        values = values.reshape(len(labels), 3).mean(axis=1).tolist()
        return _finite_vector(values, len(labels), "E_ACTIVITY_NONFINITE_SCORE")

    return score, 8, "scores"


def _load_vjepa2_activity_adapter(
    public: Path,
    candidate: dict[str, Any],
    cfg: dict[str, Any],
    labels: list[str],
    device: str,
):
    del labels
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoVideoProcessor

    code_root = _activity_code_root(public, candidate["candidate_id"])
    _verify_repository_commit(code_root, candidate["code_commit"])
    checkpoint_root = _activity_checkpoint_root(public, candidate["candidate_id"])
    checkpoint = checkpoint_root / candidate["weight_file"]
    runtime = _activity_config(cfg)["runtime_environment"]["vjepa2"]
    if not checkpoint.is_file() or file_digest(checkpoint) != candidate["weight_sha256"]:
        raise RuntimeError("E_ACTIVITY_WEIGHT_COMMITMENT")
    if file_digest(checkpoint_root / "config.json") != runtime["config_sha256"]:
        raise RuntimeError("E_VJEPA_CONFIG_COMMITMENT")
    if file_digest(checkpoint_root / "video_preprocessor_config.json") != runtime["video_preprocessor_config_sha256"]:
        raise RuntimeError("E_VJEPA_PROCESSOR_COMMITMENT")
    model = AutoModel.from_pretrained(
        checkpoint_root, local_files_only=True, use_safetensors=True
    ).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(
        checkpoint_root, local_files_only=True
    )

    def embed(frames):
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)
        values = processor(video, return_tensors="pt")["pixel_values_videos"].to(device)
        with torch.inference_mode():
            features = model.get_vision_features(values)
            pooled = functional.normalize(features.float().mean(dim=1), dim=-1)
        return _finite_vector(
            pooled.cpu()[0].tolist(), pooled.shape[-1], "E_ACTIVITY_NONFINITE_EMBEDDING"
        )

    return embed, 64, "embedding"


def _load_activity_adapter(
    public: Path,
    candidate: dict[str, Any],
    cfg: dict[str, Any],
    labels: list[str],
    device: str,
):
    loaders = {
        "egohod_egovideo_l_zero_shot": _load_egohod_activity_adapter,
        "videoprism_lvt_l_zero_shot": _load_videoprism_activity_adapter,
        "vjepa2_vitl_public_probe": _load_vjepa2_activity_adapter,
    }
    return loaders[candidate["candidate_id"]](public, candidate, cfg, labels, device)


def _gpu_peak_gib(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    values = []
    for line in completed.stdout.splitlines():
        try:
            values.append(float(line.strip()) / 1024)
        except ValueError:
            continue
    return max(values, default=0.0)


def run_activity_candidate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    amendment = _activity_config(cfg)
    _verify_activity_dependency_manifest(args.public_root, amendment)
    candidate = _activity_candidate(amendment, args.candidate_id)
    labels = _activity_labels(cfg)
    manifest = _verify_activity_manifest(
        args.manifest,
        amendment,
        args.public_root / "public/charades",
    )
    if args.partition != "development":
        raise RuntimeError("E_ACTIVITY_HOLDOUT_BEFORE_WINNER_SEAL")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("E_ACTIVITY_OFFLINE_ENVIRONMENT")
    infer, frame_count, output_key = _load_activity_adapter(
        args.public_root, candidate, cfg, labels, args.device
    )
    rows = []
    runtimes = []
    failure_count = invalid_count = truncation_count = 0
    peak_vram = _gpu_peak_gib(args.device)
    for fixture in manifest["partitions"][args.partition]:
        media = args.public_root / "public/charades/CharadesEgo_v1_480" / f"{fixture['id']}.mp4"
        started = time.monotonic()
        try:
            frames, source_frame_count = _decode_uniform_activity_frames(media, frame_count)
            controls = _activity_frame_controls(
                frames, amendment["public_activity_fixture"]["seed"], fixture["id"]
            )
            outputs = {control: infer(values) for control, values in controls.items()}
            if any(len(value) == 0 or not all(math.isfinite(float(item)) for item in value) for value in outputs.values()):
                invalid_count += 1
                raise RuntimeError("E_ACTIVITY_INVALID_OUTPUT")
            row = {
                "id": fixture["id"],
                "subject": fixture["subject"],
                "labels": fixture["labels"],
                "required_frame_count": frame_count,
                "decoded_frame_count": len(frames),
                "source_frame_count": source_frame_count,
            }
            for control, values in outputs.items():
                row[control] = {output_key: values}
            rows.append(row)
        except RuntimeError as error:
            failure_count += 1
            if str(error) == "E_ACTIVITY_SILENT_TRUNCATION":
                truncation_count += 1
            rows.append(
                {
                    "id": fixture["id"],
                    "subject": fixture["subject"],
                    "labels": fixture["labels"],
                    "status": "FAILED",
                    "error_code": str(error).split()[0],
                    "required_frame_count": frame_count,
                    "decoded_frame_count": 0,
                }
            )
        runtimes.append(time.monotonic() - started)
        peak_vram = max(peak_vram, _gpu_peak_gib(args.device))
    result = {
        "schema_version": 1,
        "status": "COMPLETE" if failure_count == 0 else "FAILED",
        "candidate_id": candidate["candidate_id"],
        "partition": args.partition,
        "candidate_weight_sha256": candidate["weight_sha256"],
        "candidate_code_commit": candidate["code_commit"],
        "manifest_commitment_sha256": amendment["public_activity_fixture"]["manifest_commitment_sha256"],
        "item_count": len(rows),
        "failure_count": failure_count,
        "invalid_record_count": invalid_count,
        "silent_truncation_count": truncation_count,
        "external_call_count": 0,
        "peak_vram_gib": peak_vram,
        "median_item_runtime_seconds": statistics.median(runtimes),
        "local_files_only_reload": True,
        "telemetry_tracking_disabled": True,
        "rows": rows,
    }
    result["candidate_output_commitment_sha256"] = digest(result)
    output = _activity_run_root(args.public_root) / "predictions" / f"{candidate['candidate_id']}-{args.partition}.json"
    write_private(output, result)
    return {key: result[key] for key in ACTIVITY_CANDIDATE_FIELDS}


def _fit_final_activity_calibration(
    output: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    candidate: dict[str, Any],
    amendment: dict[str, Any],
    labels: list[str],
) -> dict[str, Any]:
    rows = _candidate_output_rows(
        output, manifest_rows, candidate["candidate_id"], labels
    )
    expected = [set(row["labels"]) for row in rows]
    if candidate["candidate_id"] == "vjepa2_vitl_public_probe":
        embeddings = [row["ordered"]["embedding"] for row in rows]
        scored, models = _fit_probe_scores(
            embeddings,
            expected,
            {"ordered": embeddings},
            labels,
            candidate["probe_recipe"],
        )
        scores = scored["ordered"]
        transform = {"kind": "public_logistic_probe", "models": models}
    else:
        raw = [row["ordered"]["scores"] for row in rows]
        centers, scales = _robust_parameters(raw)
        scores = _robust_transform(raw, centers, scales)
        transform = {
            "kind": "robust_standardization",
            "centers": dict(zip(labels, centers, strict=True)),
            "scales": dict(zip(labels, scales, strict=True)),
        }
    thresholds = [
        _choose_label_threshold(
            [row[index] for row in scores],
            [label in truth for truth in expected],
        )
        for index, label in enumerate(labels)
    ]
    comparison = amendment["development_comparison"]
    margin = _choose_abstention_margin(
        scores,
        expected,
        labels,
        thresholds,
        [float(value) for value in comparison["abstention_margin_grid"]],
        float(comparison["eligibility_floors"]["nonabstained_coverage_min"]),
    )
    if margin is None:
        raise RuntimeError("E_ACTIVITY_FINAL_MARGIN")
    return {
        "transform": transform,
        "thresholds": dict(zip(labels, thresholds, strict=True)),
        "abstention_margin": margin,
        "training_partition": "development_only",
        "training_item_count": len(rows),
    }


def select_activity_public(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    amendment = _activity_config(cfg)
    labels = _activity_labels(cfg)
    manifest = _verify_activity_manifest(args.manifest, amendment)
    manifest_rows = manifest["partitions"]["development"]
    candidate_outputs = []
    metrics = []
    for candidate in amendment["bounded_candidates"]:
        path = _activity_run_root(args.public_root) / "predictions" / f"{candidate['candidate_id']}-development.json"
        output = json.loads(path.read_text())
        candidate_outputs.append(output)
        metrics.append(
            _crossfit_activity_candidate(
                output, manifest_rows, candidate, amendment, labels
            )
        )
    precision = float(amendment["development_comparison"]["metric_tie_precision"])
    winner = _select_activity_winner(metrics, precision)
    result = {
        "schema_version": 1,
        "status": "PASS_WINNER_SEALED" if winner is not None else "NO_GO_NO_ELIGIBLE_CANDIDATE",
        "candidate_count": len(metrics),
        "eligible_candidate_count": sum(value["eligible"] for value in metrics),
        "manifest_commitment_sha256": amendment["public_activity_fixture"]["manifest_commitment_sha256"],
        "thresholds_and_selection_rule_frozen_before_outcomes": True,
        "candidate_metrics": metrics,
        "winner_candidate_id": winner["candidate_id"] if winner else "NONE",
        "winner_macro_f1": winner["macro_f1"] if winner else 0.0,
        "winner_worst_class_recall": winner["worst_class_recall"] if winner else 0.0,
        "winner_nonabstained_coverage": winner["nonabstained_coverage"] if winner else 0.0,
        "winner_temporal_shuffled_positive_fraction": winner["temporal"]["ordered_over_shuffled_positive_fraction"] if winner else 0.0,
        "winner_temporal_repeated_positive_fraction": winner["temporal"]["ordered_over_repeated_positive_fraction"] if winner else 0.0,
        "holdout_opened": False,
        "governed_C_opened": False,
    }
    if winner is not None:
        winner_index = [
            candidate["candidate_id"]
            for candidate in amendment["bounded_candidates"]
        ].index(winner["candidate_id"])
        candidate = amendment["bounded_candidates"][winner_index]
        final_calibration = _fit_final_activity_calibration(
            candidate_outputs[winner_index],
            manifest_rows,
            candidate,
            amendment,
            labels,
        )
        seal = {
            "schema_version": 1,
            "status": "SEALED_BEFORE_HOLDOUT",
            "candidate_id": winner["candidate_id"],
            "candidate_weight_sha256": candidate["weight_sha256"],
            "candidate_code_commit": candidate["code_commit"],
            "manifest_commitment_sha256": amendment["public_activity_fixture"]["manifest_commitment_sha256"],
            "development_metrics": {
                key: winner[key]
                for key in (
                    "macro_f1",
                    "worst_class_recall",
                    "nonabstained_coverage",
                    "temporal",
                    "peak_vram_gib",
                    "median_item_runtime_seconds",
                )
            },
            "final_development_calibration": final_calibration,
            "holdout_outcomes_opened": False,
        }
        seal["winner_seal_commitment_sha256"] = digest(seal)
        result["winner_seal_commitment_sha256"] = seal[
            "winner_seal_commitment_sha256"
        ]
        write_private(_activity_run_root(args.public_root) / "winner_seal.json", seal)
    result["activity_selection_commitment_sha256"] = digest(result)
    write_private(
        _activity_run_root(args.public_root) / "development_selection.json", result
    )
    return {key: result[key] for key in ACTIVITY_SELECTION_FIELDS}


def write_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical(value) + b"\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def bucket(value: float, edges: list[float]) -> str:
    if not math.isfinite(value):
        raise ValueError("bucket value must be finite")
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return f"bin_{index}"
    if value == edges[-1]:
        return f"bin_{len(edges) - 2}"
    raise ValueError("bucket value is outside frozen edges")


def union_duration(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def classify_speech_act(text: str, rules: dict[str, Any]) -> str:
    normalized = " ".join(text.lower().strip().split())
    stripped = normalized.rstrip(".?!")
    first = stripped.split(maxsplit=1)[0] if stripped else ""
    if any(stripped.startswith(prefix) for prefix in rules["naming_prefixes"]):
        return "naming"
    if "?" in normalized or first in rules["question_mark_or_public_initial_words"]:
        return "question"
    if first in rules["directive_first_words"]:
        return "directive"
    return "other"


def _allocated_labels(
    proportions: dict[str, float], count: int, seed: int, namespace: str
) -> list[str]:
    usable = {key: max(0.0, float(value)) for key, value in proportions.items()}
    total = sum(usable.values())
    if not usable or total <= 0:
        raise ValueError(f"no allocation support for {namespace}")
    exact = {key: value / total * count for key, value in usable.items()}
    allocated = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = count - sum(allocated.values())
    order = sorted(
        usable,
        key=lambda key: (-(exact[key] - allocated[key]), digest([seed, namespace, key])),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    tagged = [
        (key, ordinal)
        for key in sorted(allocated)
        for ordinal in range(allocated[key])
    ]
    tagged.sort(key=lambda pair: digest([seed, namespace, pair[0], pair[1]]))
    return [key for key, _ in tagged]


def _duration(ffmpeg: str, source: Path) -> float:
    completed = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-i", str(source)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    match = re.search(
        rb"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr
    )
    if match is None:
        raise RuntimeError("E_MEDIA_DURATION")
    return int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])


def _decode_frame(ffmpeg: str, source: Path, timestamp: float, size: int):
    from PIL import Image

    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp):.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            f"scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    if completed.returncode or not completed.stdout:
        return None
    try:
        with Image.open(BytesIO(completed.stdout)) as image:
            return image.convert("RGB").copy()
    except Exception:
        return None


def _image_metrics(image, previous) -> dict[str, float | None]:
    import numpy as np

    array = np.asarray(image, dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_strength = float((dx.mean() + dy.mean()) / 2)
    edge_fraction = float(
        ((dx[:-1, :] + dy[:, :-1]) / 2 > 0.12).mean()
    )
    motion = None
    if previous is not None:
        prior = np.asarray(previous, dtype=np.float32).mean(axis=2) / 255.0
        motion = float(np.abs(gray - prior).mean())
    return {
        "brightness": float(gray.mean()),
        "blur_edge_strength": edge_strength,
        "clutter_edge_fraction": edge_fraction,
        "motion_mean_absolute_luma": motion,
    }


def _load_vision(public: Path, cfg: dict[str, Any], device: str):
    import torch
    from apps.alignment_scoring.third_party.perception_models.core.vision_encoder import (
        pe,
        transforms,
    )

    frozen = cfg["calibration_C"]["extractor"]["vision_model"]
    candidates = list(
        (public / "models/pe-hf-home/hub").glob(
            f"models--facebook--PE-Core-L14-336/snapshots/{frozen['revision']}/PE-Core-L14-336.pt"
        )
    )
    if len(candidates) != 1 or file_digest(candidates[0]) != frozen["weights_sha256"]:
        raise RuntimeError("E_FROZEN_VISION_MODEL")
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
    model.load_ckpt(str(candidates[0]))
    model = model.to(device).eval()
    transform = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    extractor = cfg["calibration_C"]["extractor"]
    repair = extractor.get("coverage_repair")
    prompt_groups = (
        repair["prompt_ensembles"]
        if repair and repair.get("status") == "FROZEN_ACTIVE"
        else extractor["regular_frame_prompt_groups"]
    )
    flattened: list[str] = []
    prompt_ranges: dict[str, list[tuple[str, int, int]]] = {}
    for group, prompts in prompt_groups.items():
        prompt_ranges[group] = []
        for label, values in prompts.items():
            values = values if isinstance(values, list) else [values]
            start = len(flattened)
            flattened.extend(values)
            prompt_ranges[group].append((label, start, len(flattened)))
    with torch.inference_mode():
        encoded = model.encode_text(
            tokenizer(flattened).to(device), normalize=True
        ).cpu()
        prototypes = []
        slices: dict[str, tuple[int, int, list[str]]] = {}
        for group, ranges in prompt_ranges.items():
            start = len(prototypes)
            labels = []
            for label, first, last in ranges:
                prototype = encoded[first:last].mean(dim=0)
                prototype = prototype / prototype.norm().clamp_min(1e-12)
                prototypes.append(prototype)
                labels.append(label)
            slices[group] = (start, len(prototypes), labels)
        text = torch.stack(prototypes)
    return model, transform, text, slices


def _vision_batch(
    model,
    transform,
    text,
    slices,
    images,
    margin: float | None,
    device: str,
):
    import torch

    with torch.inference_mode():
        batch = torch.stack([transform(image) for image in images]).to(device)
        features = model.encode_image(batch, normalize=True).cpu()
        scores = features @ text.T
    labels: list[dict[str, str | None]] = []
    margins: list[dict[str, float]] = []
    for row in scores:
        values: dict[str, str | None] = {}
        row_margins: dict[str, float] = {}
        for group, (start, stop, names) in slices.items():
            part = row[start:stop]
            order = torch.argsort(part, descending=True)
            top = int(order[0])
            gap = float(part[top] - part[int(order[1])]) if len(order) > 1 else 1.0
            values[group] = names[top] if margin is None or gap >= margin else None
            row_margins[group] = gap
        labels.append(values)
        margins.append(row_margins)
    return features, labels, margins


def _repair_config(cfg: dict[str, Any]) -> dict[str, Any]:
    repair = cfg["calibration_C"]["extractor"].get("coverage_repair")
    if not isinstance(repair, dict) or repair.get("status") != "FROZEN_ACTIVE":
        raise RuntimeError("E_EXTRACTOR_REPAIR_NOT_FROZEN")
    return repair


def _detector_model_root(public: Path, repair: dict[str, Any]) -> Path:
    model = repair["detector_model"]
    repository = model["repository"].replace("/", "--")
    return (
        public
        / "models/owlv2-hf-home/hub"
        / f"models--{repository}"
        / "snapshots"
        / model["revision"]
    )


def _verify_detector_files(root: Path, repair: dict[str, Any]) -> None:
    required = repair["detector_model"]["required_files_sha256"]
    for name, expected in required.items():
        path = root / name
        if not path.is_file() or file_digest(path) != expected:
            raise RuntimeError("E_FROZEN_DETECTOR_MODEL")


def _load_detector(public: Path, cfg: dict[str, Any], device: str):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    repair = _repair_config(cfg)
    root = _detector_model_root(public, repair)
    _verify_detector_files(root, repair)
    processor = Owlv2Processor.from_pretrained(root, local_files_only=True)
    model = Owlv2ForObjectDetection.from_pretrained(
        root, local_files_only=True, use_safetensors=True
    )
    return model.to(device).eval(), processor


def _box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _nms_detections(
    detections: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda value: -value["score"]):
        if all(
            detection["kind"] != prior["kind"]
            or _box_iou(detection["box"], prior["box"]) < threshold
            for prior in retained
        ):
            retained.append(detection)
    return retained


def _detector_batch(
    model,
    processor,
    images,
    repair: dict[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    import torch

    queries = repair["detector_queries"]
    texts = [value["text"] for value in queries]
    minimum_threshold = min(
        repair["detector_score_thresholds"][value["kind"]] for value in queries
    )
    output: list[dict[str, Any]] = []
    batch_size = repair["detector_batch_size"]
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        inputs = processor(
            text=[texts for _ in chunk], images=chunk, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            raw = model(**inputs)
        target_sizes = torch.tensor(
            [[image.height, image.width] for image in chunk], device=device
        )
        rows = processor.post_process_object_detection(
            outputs=raw,
            target_sizes=target_sizes,
            threshold=minimum_threshold,
        )
        for image, row in zip(chunk, rows, strict=True):
            detections = []
            invalid = 0
            for score, label, box in zip(
                row["scores"], row["labels"], row["boxes"], strict=True
            ):
                query = queries[int(label)]
                threshold = repair["detector_score_thresholds"][query["kind"]]
                score_value = float(score)
                if score_value < threshold:
                    continue
                values = [float(value) for value in box]
                if not all(math.isfinite(value) for value in values):
                    invalid += 1
                    continue
                values = [
                    min(max(values[0], 0.0), image.width),
                    min(max(values[1], 0.0), image.height),
                    min(max(values[2], 0.0), image.width),
                    min(max(values[3], 0.0), image.height),
                ]
                if values[2] <= values[0] or values[3] <= values[1]:
                    invalid += 1
                    continue
                detections.append(
                    {
                        "kind": query["kind"],
                        "label": query["label"],
                        "score": score_value,
                        "box": values,
                    }
                )
            detections = _nms_detections(
                detections, repair["detector_nms_iou"]
            )
            output.append(
                {
                    "detections": detections,
                    "invalid_box_count": invalid,
                    "width": image.width,
                    "height": image.height,
                }
            )
    return output


def _box_gap(first: list[float], second: list[float]) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def _detector_proxies(
    record: dict[str, Any], repair: dict[str, Any]
) -> dict[str, str]:
    detections = record["detections"]
    hands = [value for value in detections if value["kind"] == "hand"]
    objects = [value for value in detections if value["kind"] == "object"]
    width = float(record["width"])
    height = float(record["height"])
    diagonal = max(math.hypot(width, height), 1.0)
    contact = any(_box_iou(hand["box"], obj["box"]) > 0 for hand in hands for obj in objects)
    near = any(
        _box_gap(hand["box"], obj["box"]) / diagonal
        <= repair["hand_object_near_gap_fraction"]
        for hand in hands
        for obj in objects
    )
    if not hands:
        hand_action = "no_hand"
    elif contact:
        hand_action = "grasp_hold"
    elif near:
        hand_action = "reach"
    else:
        hand_action = "visible_no_contact"
    count = len(objects)
    if count == 0:
        referent = "none"
    elif count == 1:
        referent = "clear_one"
    else:
        referent = "ambiguous_many"
    distractors = "one" if count <= 1 else "few" if count <= 4 else "many"
    edge_margin = repair["box_edge_margin_fraction"]
    edge_count = sum(
        box["box"][0] <= width * edge_margin
        or box["box"][1] <= height * edge_margin
        or box["box"][2] >= width * (1 - edge_margin)
        or box["box"][3] >= height * (1 - edge_margin)
        for box in objects
    )
    overlap = max(
        (
            _box_iou(objects[first]["box"], objects[second]["box"])
            for first in range(len(objects))
            for second in range(first)
        ),
        default=0.0,
    )
    if objects and (
        edge_count / len(objects) >= repair["heavy_occlusion_edge_fraction"]
        or overlap >= repair["heavy_occlusion_pair_iou"]
    ):
        occlusion = "heavy"
    elif edge_count or overlap >= repair["partial_occlusion_pair_iou"]:
        occlusion = "partial"
    else:
        occlusion = "clear"
    if not objects:
        framing = "distributed"
    elif len(objects) > 1:
        centers = [
            ((value["box"][0] + value["box"][2]) / (2 * width))
            for value in objects
        ]
        if max(centers) - min(centers) >= repair["distributed_center_span"]:
            framing = "distributed"
        else:
            main = max(objects, key=lambda value: value["score"])
            x = (main["box"][0] + main["box"][2]) / (2 * width)
            y = (main["box"][1] + main["box"][3]) / (2 * height)
            framing = "peripheral" if min(x, y, 1 - x, 1 - y) < repair["peripheral_center_margin"] else "centered"
    else:
        main = objects[0]
        x = (main["box"][0] + main["box"][2]) / (2 * width)
        y = (main["box"][1] + main["box"][3]) / (2 * height)
        framing = "peripheral" if min(x, y, 1 - x, 1 - y) < repair["peripheral_center_margin"] else "centered"
    return {
        "hand_visibility": "visible" if hands else "not_visible",
        "hand_action": hand_action,
        "referent": referent,
        "distractors": distractors,
        "occlusion": occlusion,
        "framing": framing,
    }


def _temporal_hand_completion(by_position: dict[str, dict[str, str]]) -> str | None:
    if set(by_position) != {"before", "during", "after"}:
        return None
    before = by_position["before"]["hand_action"]
    during = by_position["during"]["hand_action"]
    after = by_position["after"]["hand_action"]
    contact = {"grasp_hold"}
    if during in contact and after not in contact:
        return "completed"
    if before not in contact and during in contact:
        return "contact_onset"
    if before in contact and during in contact and after in contact:
        return "persistent_contact"
    if during == "reach" or after == "reach":
        return "reach_without_contact"
    return "no_detected_transition"


def _public_repair_root(public: Path) -> Path:
    return public / "runs/synthetic-video-calibration/extractor-repair"


def prepare_public(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    cfg = json.loads(args.config.read_text())
    repair = _repair_config(cfg)
    model_cfg = repair["detector_model"]
    cache = args.public_root / "models/owlv2-hf-home/hub"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ["HF_HOME"] = str(args.public_root / "models/owlv2-hf-home")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    snapshot_download(
        repo_id=model_cfg["repository"],
        revision=model_cfg["revision"],
        cache_dir=cache,
        allow_patterns=sorted(model_cfg["required_files_sha256"]),
    )
    model_root = _detector_model_root(args.public_root, repair)
    _verify_detector_files(model_root, repair)
    dependency = repair["runtime_dependency"]
    wheel_root = args.public_root / "models/calibration-repair-wheels"
    wheel_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    wheel = wheel_root / dependency["wheel"]
    if not wheel.is_file() or file_digest(wheel) != dependency["wheel_sha256"]:
        with tempfile.TemporaryDirectory(prefix="calibration-repair-wheel-") as temp:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--dest",
                    temp,
                    f"{dependency['package']}=={dependency['version']}",
                ],
                check=True,
            )
            downloaded = Path(temp) / dependency["wheel"]
            if not downloaded.is_file() or file_digest(downloaded) != dependency["wheel_sha256"]:
                raise RuntimeError("E_RUNTIME_DEPENDENCY_HASH")
            shutil.copy2(downloaded, wheel)
            os.chmod(wheel, 0o600)
    dependency_root = args.public_root / "models/calibration-repair-pydeps"
    dependency_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--upgrade",
            "--target",
            str(dependency_root),
            str(wheel),
        ],
        check=True,
    )
    dependency_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scipy; print(scipy.__version__)",
        ],
        env={**os.environ, "PYTHONPATH": str(dependency_root)},
        text=True,
        capture_output=True,
        check=True,
    )
    if dependency_check.stdout.strip() != dependency["version"]:
        raise RuntimeError("E_RUNTIME_DEPENDENCY_VERSION")
    fixture_root = args.public_root / "models/calibration-public-fixtures"
    fixture_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for fixture in repair["public_qualification"]["fixtures"]:
        target = fixture_root / fixture["file"]
        if not target.is_file() or file_digest(target) != fixture["sha256"]:
            partial = target.with_suffix(target.suffix + ".partial")
            request = urllib.request.Request(
                fixture["url"], headers={"User-Agent": "synthetic-video-research/1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                partial.write_bytes(response.read())
            os.chmod(partial, 0o600)
            if file_digest(partial) != fixture["sha256"]:
                partial.unlink(missing_ok=True)
                raise RuntimeError("E_PUBLIC_FIXTURE_HASH")
            partial.replace(target)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "extractor_repair_commitment_sha256": digest(repair),
        "model": {
            "repository": model_cfg["repository"],
            "revision": model_cfg["revision"],
            "license": model_cfg["license"],
            "files_sha256": {
                name: file_digest(model_root / name)
                for name in sorted(model_cfg["required_files_sha256"])
            },
        },
        "runtime_dependency": {
            "package": dependency["package"],
            "version": dependency["version"],
            "license": dependency["license"],
            "wheel_sha256": file_digest(wheel),
        },
        "fixtures": [
            {
                "file": fixture["file"],
                "sha256": file_digest(fixture_root / fixture["file"]),
                "license": fixture["license"],
            }
            for fixture in repair["public_qualification"]["fixtures"]
        ],
        "local_files_only_required_after_preparation": True,
    }
    manifest["public_dependency_commitment_sha256"] = digest(manifest)
    write_private(_public_repair_root(args.public_root) / "dependency_manifest.json", manifest)
    return {
        "status": "PASS",
        "fixture_file_count": len(manifest["fixtures"]),
        "model_file_count": len(manifest["model"]["files_sha256"]),
        "extractor_repair_commitment_sha256": digest(repair),
    }


def qualify_public(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    cfg = json.loads(args.config.read_text())
    repair = _repair_config(cfg)
    repair_commitment = digest(repair)
    manifest_path = _public_repair_root(args.public_root) / "dependency_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "PASS"
        or manifest.get("extractor_repair_commitment_sha256") != repair_commitment
    ):
        raise RuntimeError("E_PUBLIC_DEPENDENCY_MANIFEST")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model, transform, text_features, slices = _load_vision(
        args.public_root, cfg, args.device
    )
    detector, processor = _load_detector(args.public_root, cfg, args.device)
    fixture_root = args.public_root / "models/calibration-public-fixtures"
    fixtures = repair["public_qualification"]["fixtures"]
    images = []
    for fixture in fixtures:
        path = fixture_root / fixture["file"]
        if file_digest(path) != fixture["sha256"]:
            raise RuntimeError("E_PUBLIC_FIXTURE_HASH")
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    _, labels, _ = _vision_batch(
        model, transform, text_features, slices, images, None, args.device
    )
    detection_rows = _detector_batch(
        detector, processor, images, repair, args.device
    )
    activity_correct = expected_object_hits = 0
    hand_positive_hits = hand_positive_count = 0
    hand_negative_correct = hand_negative_count = 0
    proxy_complete = invalid_boxes = 0
    proxy_fields = {
        "hand_visibility",
        "hand_action",
        "referent",
        "distractors",
        "occlusion",
        "framing",
    }
    for fixture, label, row in zip(fixtures, labels, detection_rows, strict=True):
        activity_correct += label["activity"] == fixture["expected_activity"]
        detected_labels = {
            value["label"] for value in row["detections"] if value["kind"] == "object"
        }
        expected_object_hits += bool(
            detected_labels.intersection(fixture["expected_object_labels"])
        )
        hand_visible = any(
            value["kind"] == "hand" for value in row["detections"]
        )
        if fixture["expected_hand_visible"]:
            hand_positive_count += 1
            hand_positive_hits += hand_visible
        else:
            hand_negative_count += 1
            hand_negative_correct += not hand_visible
        proxies = _detector_proxies(row, repair)
        proxy_complete += set(proxies) == proxy_fields and all(proxies.values())
        invalid_boxes += row["invalid_box_count"]
    thresholds = repair["public_qualification"]["thresholds"]
    passed = all(
        [
            activity_correct >= thresholds["activity_correct_min"],
            expected_object_hits >= thresholds["expected_object_hits_min"],
            hand_positive_hits >= thresholds["hand_positive_hits_min"],
            hand_negative_correct >= thresholds["hand_negative_correct_min"],
            proxy_complete == len(fixtures),
            invalid_boxes == 0,
        ]
    )
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "NO_GO",
        "fixture_count": len(fixtures),
        "activity_correct_count": int(activity_correct),
        "expected_object_hit_count": int(expected_object_hits),
        "hand_positive_hit_count": int(hand_positive_hits),
        "hand_positive_count": hand_positive_count,
        "hand_negative_correct_count": int(hand_negative_correct),
        "hand_negative_count": hand_negative_count,
        "proxy_complete_count": int(proxy_complete),
        "invalid_box_count": invalid_boxes,
        "model_file_count": len(repair["detector_model"]["required_files_sha256"]),
        "fixture_file_count": len(fixtures),
        "thresholds": thresholds,
        "extractor_repair_commitment_sha256": repair_commitment,
        "local_files_only_reload": True,
        "telemetry_disabled": True,
        "selection_rule": "single_frozen_combined_extractor_no_candidate_cycling",
    }
    result["public_qualification_commitment_sha256"] = digest(result)
    write_private(
        _public_repair_root(args.public_root) / "public_qualification.json", result
    )
    return {key: result[key] for key in PUBLIC_QUALIFICATION_FIELDS}


def _bootstrap_categorical(
    observations: list[tuple[str, str | None]], replicates: int, seed: int
) -> dict[str, Any]:
    import numpy as np

    present = [(child, value) for child, value in observations if value is not None]
    counts = Counter(value for _, value in present)
    support = sum(counts.values())
    proportions = {
        label: count / support for label, count in sorted(counts.items())
    } if support else {}
    by_child: dict[str, list[str]] = defaultdict(list)
    for child, value in present:
        by_child[child].append(value)
    children = sorted(by_child)
    intervals: dict[str, list[float]] = {}
    if children and counts:
        rng = np.random.default_rng(seed)
        draws: dict[str, list[float]] = {label: [] for label in counts}
        for _ in range(replicates):
            chosen = rng.choice(children, size=len(children), replace=True)
            sampled = Counter(
                value for child in chosen for value in by_child[str(child)]
            )
            denominator = sum(sampled.values())
            for label in draws:
                draws[label].append(sampled[label] / denominator if denominator else 0.0)
        intervals = {
            label: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for label, values in draws.items()
        }
    return {
        "support": support,
        "missing": len(observations) - support,
        "counts": dict(sorted(counts.items())),
        "proportions": proportions,
        "child_cluster_bootstrap_95ci": intervals,
    }


def _add(observations, axis: str, feature: str, child: str, value: str | None):
    observations[axis][feature].append((child, value))


def _summarize(
    observations: dict[str, dict[str, list[tuple[str, str | None]]]], cfg: dict[str, Any]
) -> tuple[dict[str, Any], int, int]:
    uncertainty = cfg["calibration_C"]["extractor"]["uncertainty"]
    missing_cfg = cfg["calibration_C"]["extractor"]["missingness"]
    sparse_min = cfg["calibration_C"]["extractor"]["disclosure_review"][
        "minimum_cell_count_for_external_numeric_export"
    ]
    targets: dict[str, Any] = {}
    missing_axes = 0
    suppressed = 0
    for axis in AXES:
        features = {}
        axis_observed = axis_total = 0
        for feature, values in sorted(observations[axis].items()):
            summary = _bootstrap_categorical(
                values, uncertainty["replicates"], uncertainty["seed"]
            )
            features[feature] = summary
            axis_observed += summary["support"]
            axis_total += summary["support"] + summary["missing"]
            suppressed += sum(
                count < sparse_min for count in summary["counts"].values()
            )
        missing_fraction = (
            1.0 - axis_observed / axis_total if axis_total else 1.0
        )
        status = "MEASURED"
        feature_support = [feature["support"] for feature in features.values()]
        insufficient_features = sum(
            support < missing_cfg["minimum_observations_per_feature"]
            for support in feature_support
        )
        if not feature_support or max(feature_support) < missing_cfg["minimum_observations_per_feature"]:
            status = "INSUFFICIENT_SUPPORT"
        elif missing_fraction > missing_cfg["maximum_axis_missing_fraction"]:
            status = "HIGH_MISSINGNESS"
        elif insufficient_features:
            status = "MEASURED_WITH_DECLARED_PARTIAL_FEATURES"
        if status in {"INSUFFICIENT_SUPPORT", "HIGH_MISSINGNESS"}:
            missing_axes += 1
        targets[axis] = {
            "status": status,
            "model_derived": axis in {ACTIVITY_AXIS, SCENE_AXIS, HAND_AXIS, GROUNDING_AXIS},
            "missing_fraction": missing_fraction,
            "insufficient_feature_count": insufficient_features,
            "features": features,
        }
    return targets, missing_axes, suppressed


def _proportions(targets: dict[str, Any], axis: str, feature: str, fallback):
    values = targets[axis]["features"].get(feature, {}).get("proportions", {})
    values = {key: value for key, value in values.items() if key != "ABSTAIN"}
    return values or fallback


def _speech_line(kind: str, noun: dict[str, str]) -> str:
    if kind == "naming":
        return f"Schau mal, das ist {noun['article_nom']} {noun['de']}."
    if kind == "directive":
        return f"Bitte nimm {noun['article_acc']} {noun['de']} und halte den Gegenstand gut fest."
    if kind == "question":
        return f"Wo ist {noun['article_nom']} {noun['de']}? Zeig mir den Gegenstand."
    return f"Wir schauen jetzt gemeinsam auf {noun['article_acc']} {noun['de']}."


def _build_episode_plans(
    cfg: dict[str, Any],
    targets: dict[str, Any],
    calibration_commitment: str,
    executable: bool = True,
) -> dict[str, Any]:
    plan_cfg = cfg["calibration_C"]["episode_plan"]
    count = plan_cfg["candidate_plan_count"]
    seed = plan_cfg["plan_seed"]
    activity = _allocated_labels(
        _proportions(targets, ACTIVITY_AXIS, "activity", {"other": 1.0}),
        count,
        seed,
        "activity",
    )
    hand = _allocated_labels(
        _proportions(targets, HAND_AXIS, "hand_action", {"manipulate": 1.0}),
        count,
        seed,
        "hand_action",
    )
    framing = _allocated_labels(
        _proportions(targets, VISUAL_AXIS, "framing", {"centered": 1.0}),
        count,
        seed,
        "framing",
    )
    occlusion = _allocated_labels(
        _proportions(targets, SCENE_AXIS, "occlusion", {"clear": 1.0}),
        count,
        seed,
        "occlusion",
    )
    distractors = _allocated_labels(
        _proportions(targets, SCENE_AXIS, "distractors", {"few": 1.0}),
        count,
        seed,
        "distractors",
    )
    acts = _allocated_labels(
        _proportions(targets, LANGUAGE_AXIS, "speech_act", {"naming": 1.0}),
        count * plan_cfg["utterances_per_plan"],
        seed,
        "speech_act",
    )
    contexts = plan_cfg["public_contexts"]
    actions = plan_cfg["public_actions"]
    lexicon = plan_cfg["public_lexicon"]
    adjectives = plan_cfg["public_adjectives"]
    rows = []
    for index in range(count):
        noun = lexicon[index % len(lexicon)]
        adjective = adjectives[(index // len(lexicon)) % len(adjectives)]
        context = contexts[(index // (len(lexicon) * len(adjectives))) % len(contexts)]
        action = actions[index % len(actions)]
        visual_prompt = (
            "Photorealistic naturalistic home-video footage in one continuous uncut "
            "first-person egocentric child-height head-camera shot. "
            f"An ordinary {context} during {activity[index].replace('_', ' ')}. "
            f"A {adjective['en']} {noun['en']} remains the same persistent object. "
            f"The child-view action is {action}; hand-action regime {hand[index].replace('_', ' ')}. "
            f"Framing is {framing[index]}, occlusion is {occlusion[index]}, and there are {distractors[index]} distractors. "
            "Use plausible head motion, ordinary lighting, stable object identity, realistic contact and chronological action order. "
            "No cuts, captions, text, logos, mirrors, identifiable faces, unsafe behavior, dialogue, lyrics, or music. "
            "Quiet room ambience only; generated audio will be discarded."
        )
        utterance_kinds = acts[index * 2 : index * 2 + 2]
        utterances = [_speech_line(kind, noun) for kind in utterance_kinds]
        rows.append(
            {
                "plan_key": digest([seed, "episode", index]),
                "seed": int(digest([seed, "ltx", index])[:8], 16) % 2147483647,
                "duration_seconds": plan_cfg["clip_seconds"],
                "visual_prompt_en": visual_prompt,
                "utterances_de": utterances,
                "utterance_kinds": utterance_kinds,
                "utterance_onsets_seconds": plan_cfg["utterance_onsets_seconds"],
                "public_provenance": {
                    "wordnet": noun["wordnet"],
                    "public_pool_only": True,
                },
                "target_labels": {
                    "activity": activity[index],
                    "hand_action": hand[index],
                    "framing": framing[index],
                    "occlusion": occlusion[index],
                    "distractors": distractors[index],
                },
            }
        )
    value = {
        "schema_version": 1,
        "status": "FROZEN_EXECUTABLE" if executable else "PROVISIONAL_NO_GO",
        "calibration_commitment_sha256": calibration_commitment,
        "plan_seed": seed,
        "selection": plan_cfg["selection"],
        "evaluation_steering": False,
        "plans": rows,
    }
    value["episode_plan_commitment_sha256"] = digest(value)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    import imageio_ffmpeg
    import numpy as np

    cfg = json.loads(args.config.read_text())
    extractor = cfg["calibration_C"]["extractor"]
    repair = _repair_config(cfg)
    repair_commitment = digest(repair)
    qualification_path = (
        _public_repair_root(args.public_root) / "public_qualification.json"
    )
    qualification = json.loads(qualification_path.read_text())
    if (
        qualification.get("status") != "PASS"
        or qualification.get("extractor_repair_commitment_sha256")
        != repair_commitment
    ):
        raise RuntimeError("E_PUBLIC_EXTRACTOR_QUALIFICATION")
    original_path = (
        args.restricted_root
        / "synthetic_one_hour/calibration/restricted_calibration_targets.json"
    )
    original = json.loads(original_path.read_text())
    expected_original_commitment = cfg["calibration_C"]["governed_result"][
        "calibration_commitment_sha256"
    ]
    if original.get("calibration_commitment_sha256") != expected_original_commitment:
        raise RuntimeError("E_ORIGINAL_CALIBRATION_PROVENANCE")
    stage = json.loads((args.restricted_root / "restricted_stage_manifest.json").read_text())
    checkpoint = json.loads(
        (args.restricted_root / "asr/restricted_calibration.json").read_text()
    )
    rows = stage.get("calibration")
    items = checkpoint.get("items")
    if not isinstance(rows, list) or not isinstance(items, dict) or len(rows) != len(items):
        raise RuntimeError("E_CALIBRATION_INPUT")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    model, transform, text_features, slices = _load_vision(
        args.public_root, cfg, args.device
    )
    detector, detector_processor = _load_detector(
        args.public_root, cfg, args.device
    )
    observations: dict[str, dict[str, list[tuple[str, str | None]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    restricted_features = []
    all_features = []
    all_recording_indices = []
    sampled_frames = grounding_events = 0
    scheduled_frames = decode_failures = 0
    recording_index = 0
    bins = extractor["fixed_numeric_bins"]
    margin = None
    scratch = args.scratch_root
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    for row in sorted(rows, key=lambda value: value["asset_key"]):
        asset_key = row["asset_key"]
        item = items.get(asset_key)
        if not isinstance(item, dict):
            raise RuntimeError("E_CALIBRATION_CHECKPOINT")
        source = args.restricted_root / row["file"]
        local = scratch / f"{digest([asset_key, 'media'])}{source.suffix.lower()}"
        shutil.copy2(source, local)
        try:
            duration = _duration(ffmpeg, local)
            regular_times = []
            timestamp = extractor["frame_sample_interval_seconds"] / 2
            while timestamp < duration and len(regular_times) < extractor["max_regular_frames_per_recording"]:
                regular_times.append(timestamp)
                timestamp += extractor["frame_sample_interval_seconds"]
            images = []
            decoded_times = []
            metrics = []
            previous = None
            child = row["child_key"]
            for timestamp in regular_times:
                scheduled_frames += 1
                image = _decode_frame(ffmpeg, local, timestamp, extractor["decoded_image_size"])
                if image is None:
                    decode_failures += 1
                    for axis, feature in (
                        (ACTIVITY_AXIS, "activity"),
                        (ACTIVITY_AXIS, "activity_confidence"),
                        (VISUAL_AXIS, "framing"),
                        (VISUAL_AXIS, "brightness"),
                        (VISUAL_AXIS, "blur_edge_strength"),
                        (VISUAL_AXIS, "motion"),
                        (SCENE_AXIS, "occlusion"),
                        (SCENE_AXIS, "distractors"),
                        (SCENE_AXIS, "clutter_edge_fraction"),
                        (HAND_AXIS, "hand_visibility"),
                        (HAND_AXIS, "hand_action"),
                        (TEMPORAL_AXIS, "idle_transition"),
                        (GROUNDING_AXIS, "regular_referent"),
                        (GROUNDING_AXIS, "speech_referent_null"),
                        (DIVERSITY_AXIS, "cross_recording_near_duplicate"),
                        (DIVERSITY_AXIS, "activity_long_tail"),
                    ):
                        _add(observations, axis, feature, child, None)
                    continue
                images.append(image)
                decoded_times.append(timestamp)
                metrics.append(_image_metrics(image, previous))
                previous = image
            if images:
                features, labels, vision_margins = _vision_batch(
                    model, transform, text_features, slices, images, margin, args.device
                )
                detector_rows = _detector_batch(
                    detector,
                    detector_processor,
                    images,
                    repair,
                    args.device,
                )
                detector_proxies = [
                    _detector_proxies(value, repair) for value in detector_rows
                ]
            else:
                features = np.empty((0, 1), dtype=np.float32)
                labels = []
                vision_margins = []
                detector_rows = []
                detector_proxies = []
            for index, (timestamp, metric, label, vision_margin, proxy, detector_row) in enumerate(
                zip(
                    decoded_times,
                    metrics,
                    labels,
                    vision_margins,
                    detector_proxies,
                    detector_rows,
                    strict=True,
                )
            ):
                _add(observations, ACTIVITY_AXIS, "activity", child, label["activity"])
                activity_margin = vision_margin["activity"]
                activity_confidence = (
                    "low"
                    if activity_margin < repair["activity_margin_bands"][0]
                    else "medium"
                    if activity_margin < repair["activity_margin_bands"][1]
                    else "high"
                )
                _add(
                    observations,
                    ACTIVITY_AXIS,
                    "activity_confidence",
                    child,
                    activity_confidence,
                )
                _add(observations, VISUAL_AXIS, "framing", child, proxy["framing"])
                _add(observations, SCENE_AXIS, "occlusion", child, proxy["occlusion"])
                _add(observations, SCENE_AXIS, "distractors", child, proxy["distractors"])
                _add(observations, HAND_AXIS, "hand_visibility", child, proxy["hand_visibility"])
                _add(observations, HAND_AXIS, "hand_action", child, proxy["hand_action"])
                _add(observations, GROUNDING_AXIS, "regular_referent", child, proxy["referent"])
                for feature in ("brightness", "blur_edge_strength", "clutter_edge_fraction"):
                    axis = SCENE_AXIS if feature == "clutter_edge_fraction" else VISUAL_AXIS
                    _add(observations, axis, feature, child, bucket(float(metric[feature]), bins[feature]))
                motion = metric["motion_mean_absolute_luma"]
                motion_bin = bucket(float(motion), bins["motion_mean_absolute_luma"]) if motion is not None else None
                _add(observations, VISUAL_AXIS, "motion", child, motion_bin)
                transition = None
                if motion is not None:
                    transition = "idle" if motion < 0.03 else "transition" if motion < 0.25 else "shot_discontinuity"
                _add(observations, TEMPORAL_AXIS, "idle_transition", child, transition)
                speech_here = any(
                    segment.get("status") == "ACCEPT" and segment["start"] <= timestamp <= segment["end"]
                    for segment in item.get("segments", [])
                )
                null_label = f"{'speech' if speech_here else 'silence'}|{proxy['referent']}"
                _add(observations, GROUNDING_AXIS, "speech_referent_null", child, null_label)
                combined_labels = {
                    "activity": label["activity"],
                    "activity_confidence": activity_confidence,
                    **proxy,
                }
                restricted_features.append({
                    "asset_key": asset_key,
                    "child_key": child,
                    "session_key": row["session_key"],
                    "timestamp": timestamp,
                    "metrics": metric,
                    "labels": combined_labels,
                    "detector_invalid_box_count": detector_row["invalid_box_count"],
                })
                sampled_frames += 1
            if len(features):
                features_np = features.numpy()
                all_features.extend(features_np)
                all_recording_indices.extend([recording_index] * len(features_np))
                for index in range(1, len(features_np)):
                    similarity = float(features_np[index] @ features_np[index - 1])
                    _add(observations, TEMPORAL_AXIS, "adjacent_similarity", row["child_key"], bucket(similarity, bins["adjacent_feature_similarity"]))
                for index in range(2, len(features_np)):
                    recurrence = float(np.max(features_np[index] @ features_np[: index - 1].T))
                    _add(observations, TEMPORAL_AXIS, "recurrence", row["child_key"], bucket(recurrence, bins["adjacent_feature_similarity"]))
            accepted = [segment for segment in item.get("segments", []) if segment.get("status") == "ACCEPT"]
            intervals = [(max(0.0, float(s["start"])), min(duration, float(s["end"]))) for s in accepted]
            speech_fraction = union_duration(intervals) / duration if duration > 0 else 0.0
            _add(observations, LANGUAGE_AXIS, "speech_fraction", row["child_key"], bucket(speech_fraction, bins["speech_fraction"]))
            normalized_counts = Counter(" ".join(str(s.get("en", "")).lower().split()) for s in accepted)
            previous_end = None
            speech_rows = []
            for index, segment in enumerate(accepted):
                start, stop = float(segment["start"]), float(segment["end"])
                text = str(segment.get("en", ""))
                act = classify_speech_act(text, extractor["language_rules"])
                _add(observations, LANGUAGE_AXIS, "utterance_duration", row["child_key"], bucket(max(0.0, stop - start), bins["utterance_duration_seconds"]))
                _add(observations, LANGUAGE_AXIS, "utterance_word_count", row["child_key"], bucket(float(len(text.split())), bins["utterance_word_count"]))
                _add(observations, LANGUAGE_AXIS, "speech_act", row["child_key"], act)
                repeated = "repeated" if normalized_counts[" ".join(text.lower().split())] > 1 else "unique"
                _add(observations, LANGUAGE_AXIS, "repetition", row["child_key"], repeated)
                if previous_end is not None:
                    _add(observations, LANGUAGE_AXIS, "pause_proxy", row["child_key"], bucket(max(0.0, start - previous_end), bins["pause_seconds"]))
                previous_end = max(previous_end or stop, stop)
                speech_rows.append((index, segment, act))
            ordered_events = sorted(speech_rows, key=lambda value: digest([asset_key, value[0], 42]))[: extractor["max_naming_events_per_recording"]]
            grounding_images = []
            grounding_meta = []
            grounding_metrics = []
            for event_ordinal, (index, segment, act) in enumerate(ordered_events):
                midpoint = (float(segment["start"]) + float(segment["end"])) / 2
                for position, offset in zip(("before", "during", "after"), extractor["naming_offsets_seconds"], strict=True):
                    scheduled_frames += 1
                    image = _decode_frame(ffmpeg, local, min(max(0.0, midpoint + offset), max(0.0, duration - 0.01)), extractor["decoded_image_size"])
                    if image is not None:
                        grounding_images.append(image)
                        grounding_meta.append((event_ordinal, position, act))
                        grounding_metrics.append(_image_metrics(image, None))
                    else:
                        decode_failures += 1
                        if act == "naming":
                            _add(
                                observations,
                                GROUNDING_AXIS,
                                f"naming_referent_{position}",
                                row["child_key"],
                                None,
                            )
                        if position == "during":
                            for axis, feature in (
                                (GROUNDING_AXIS, "naming_by_referent_visibility"),
                                (GROUNDING_AXIS, "naming_by_hand_action"),
                                (SCENE_AXIS, "clutter_by_occlusion"),
                                (HAND_AXIS, "action_completion"),
                            ):
                                _add(
                                    observations,
                                    axis,
                                    feature,
                                    row["child_key"],
                                    None,
                                )
            if grounding_images:
                _, grounding_labels, _ = _vision_batch(
                    model,
                    transform,
                    text_features,
                    slices,
                    grounding_images,
                    margin,
                    args.device,
                )
                grounding_detector_rows = _detector_batch(
                    detector,
                    detector_processor,
                    grounding_images,
                    repair,
                    args.device,
                )
                grounding_proxies = [
                    _detector_proxies(value, repair)
                    for value in grounding_detector_rows
                ]
                grouped: dict[
                    int,
                    list[
                        tuple[
                            str,
                            str,
                            dict[str, str | None],
                            dict[str, float | None],
                            dict[str, str],
                        ]
                    ],
                ] = defaultdict(list)
                for meta, label, metric, proxy in zip(
                    grounding_meta,
                    grounding_labels,
                    grounding_metrics,
                    grounding_proxies,
                    strict=True,
                ):
                    grouped[meta[0]].append((meta[1], meta[2], label, metric, proxy))
                for event_index, group in grouped.items():
                    position_proxies = {
                        position: proxy for position, _, _, _, proxy in group
                    }
                    completion = _temporal_hand_completion(position_proxies)
                    for position, act, label, metric, proxy in group:
                        if act == "naming":
                            _add(observations, GROUNDING_AXIS, f"naming_referent_{position}", row["child_key"], proxy["referent"])
                        if position == "during":
                            _add(observations, GROUNDING_AXIS, "naming_by_referent_visibility", row["child_key"], f"{act == 'naming'}|{proxy['referent']}")
                            _add(observations, GROUNDING_AXIS, "naming_by_hand_action", row["child_key"], f"{act == 'naming'}|{proxy['hand_action']}")
                            _add(
                                observations,
                                HAND_AXIS,
                                "action_completion",
                                row["child_key"],
                                completion,
                            )
                            clutter = bucket(float(metric["clutter_edge_fraction"]), bins["clutter_edge_fraction"])
                            _add(observations, SCENE_AXIS, "clutter_by_occlusion", row["child_key"], f"{clutter}|{proxy['occlusion']}")
                            grounding_events += 1
            _add(observations, LANGUAGE_AXIS, "overlap", row["child_key"], "present" if sum(stop-start for start,stop in intervals) - union_duration(intervals) > 0.01 else "absent")
        finally:
            local.unlink(missing_ok=True)
        recording_index += 1
    if all_features:
        feature_matrix = np.asarray(all_features, dtype=np.float32)
        record_ids = np.asarray(all_recording_indices)
        near = np.zeros(len(feature_matrix), dtype=bool)
        for start in range(0, len(feature_matrix), 256):
            scores = feature_matrix[start : start + 256] @ feature_matrix.T
            same = record_ids[start : start + 256, None] == record_ids[None, :]
            scores[same] = -1.0
            near[start : start + len(scores)] = scores.max(axis=1) >= 0.98
        for value, row in zip(near.tolist(), restricted_features, strict=True):
            _add(observations, DIVERSITY_AXIS, "cross_recording_near_duplicate", row["child_key"], "near_duplicate" if value else "unique")
        for row in restricted_features:
            _add(observations, DIVERSITY_AXIS, "activity_long_tail", row["child_key"], row["labels"]["activity"])
        activity_counts = Counter(
            row["labels"]["activity"]
            for row in restricted_features
            if row["labels"]["activity"] is not None
        )
        activity_total = sum(activity_counts.values())
        global_activity = {
            label: count / activity_total for label, count in activity_counts.items()
        } if activity_total else {}
        by_child = defaultdict(list)
        by_session = defaultdict(list)
        session_child = {}
        for row in restricted_features:
            label = row["labels"]["activity"]
            if label is None:
                continue
            by_child[row["child_key"]].append(label)
            by_session[row["session_key"]].append(label)
            session_child[row["session_key"]] = row["child_key"]
            tail = "long_tail" if global_activity.get(label, 0.0) < 0.05 else "common"
            _add(observations, DIVERSITY_AXIS, "long_tail_coverage", row["child_key"], tail)
        dispersion_edges = [0.0, 0.1, 0.2, 0.4, 1.000001]
        for child, labels in by_child.items():
            local_counts = Counter(labels)
            local_total = sum(local_counts.values())
            tv = 0.5 * sum(
                abs(local_counts[label] / local_total - global_activity.get(label, 0.0))
                for label in set(local_counts) | set(global_activity)
            )
            _add(observations, DIVERSITY_AXIS, "between_child_dispersion", child, bucket(tv, dispersion_edges))
        for session, labels in by_session.items():
            local_counts = Counter(labels)
            local_total = sum(local_counts.values())
            tv = 0.5 * sum(
                abs(local_counts[label] / local_total - global_activity.get(label, 0.0))
                for label in set(local_counts) | set(global_activity)
            )
            _add(observations, DIVERSITY_AXIS, "between_session_dispersion", session_child[session], bucket(tv, dispersion_edges))
    for row in restricted_features:
        motion = row["metrics"]["motion_mean_absolute_luma"]
        motion_label = bucket(float(motion), bins["motion_mean_absolute_luma"]) if motion is not None else "missing"
        blur_label = bucket(float(row["metrics"]["blur_edge_strength"]), bins["blur_edge_strength"])
        _add(observations, VISUAL_AXIS, "motion_by_blur", row["child_key"], f"{motion_label}|{blur_label}")
    targets, missing_axes, suppressed = _summarize(observations, cfg)
    critical_axes = set(
        extractor["generated_comparison_tolerances"]["critical_axes"]
    )
    blocking_statuses = {"INSUFFICIENT_SUPPORT", "HIGH_MISSINGNESS"}
    critical_failure = any(
        targets[axis]["status"] in blocking_statuses for axis in critical_axes
    )
    measured_axis_count = len(AXES) - missing_axes
    calibration_pass = (
        not critical_failure
        and measured_axis_count
        >= extractor["generated_comparison_tolerances"][
            "axes_required_within_tolerance"
        ]
    )
    measurement = {
        "schema_version": 2,
        "status": "PASS" if calibration_pass else "NO_GO",
        "source": "development_set_C_only",
        "extractor_contract": extractor,
        "extractor_repair_commitment_sha256": repair_commitment,
        "public_qualification_commitment_sha256": qualification[
            "public_qualification_commitment_sha256"
        ],
        "supersedes_original_calibration_commitment_sha256": expected_original_commitment,
        "measured_axes": list(AXES),
        "unmeasured": ["human_activity_labels", "speaker_diarization", "human_referent_ground_truth", "exact_object_counts", "full_distributional_fidelity"],
        "targets": targets,
        "joint_features": ["naming_by_referent_visibility", "naming_by_hand_action", "clutter_by_occlusion", "motion_by_blur"],
        "sampled_frame_count": sampled_frames,
        "scheduled_frame_count": scheduled_frames,
        "decode_failure_count": decode_failures,
        "grounding_event_count": grounding_events,
        "detector_invalid_box_count": sum(
            row["detector_invalid_box_count"] for row in restricted_features
        ),
        "measured_axis_count": measured_axis_count,
        "critical_axis_failure": critical_failure,
        "ffmpeg_sha256": file_digest(Path(ffmpeg)),
        "row_level_features_retained_governed_only": True,
        "external_target_values_exported": False,
        "omnibus_score": None,
    }
    measurement["calibration_commitment_sha256"] = digest(measurement)
    plan = _build_episode_plans(
        cfg,
        targets,
        measurement["calibration_commitment_sha256"],
        executable=calibration_pass,
    )
    output = args.restricted_root / "synthetic_one_hour/calibration_repair"
    write_private(output / "restricted_calibration_features.json", {"schema_version": 1, "rows": restricted_features})
    write_private(output / "restricted_calibration_targets.json", measurement)
    write_private(output / "restricted_episode_plans.json", plan)
    compact = {
        "status": "PASS" if calibration_pass else "NO_GO",
        "axis_count": len(AXES),
        "joint_count": 4,
        "sampled_frame_count": sampled_frames,
        "scheduled_frame_count": scheduled_frames,
        "decode_failure_count": decode_failures,
        "grounding_event_count": grounding_events,
        "missing_axis_count": missing_axes,
        "suppressed_cell_count": suppressed,
        "episode_plan_count": len(plan["plans"]),
        "extractor_repair_commitment_sha256": repair_commitment,
        "calibration_commitment_sha256": measurement["calibration_commitment_sha256"],
        "episode_plan_commitment_sha256": plan["episode_plan_commitment_sha256"],
    }
    write_private(output / "compact_calibration_result.json", compact)
    return compact


def report(args: argparse.Namespace) -> dict[str, Any]:
    path = args.restricted_root / "synthetic_one_hour/calibration_repair/compact_calibration_result.json"
    value = json.loads(path.read_text())
    print(
        compact_aggregate_json(
            value,
            allowed_fields=TERMINAL_FIELDS,
            sha256_fields=TERMINAL_HASH_FIELDS,
        )
    )
    return value


def report_axis_status(args: argparse.Namespace) -> dict[str, Any]:
    path = args.restricted_root / "synthetic_one_hour/calibration_repair/restricted_calibration_targets.json"
    value = json.loads(path.read_text())
    record: dict[str, Any] = {"status": value["status"]}
    for index, axis in enumerate(AXES, start=1):
        target = value["targets"][axis]
        record[f"axis_{index}_status"] = target["status"]
        record[f"axis_{index}_missing_fraction"] = float(target["missing_fraction"])
    print(compact_aggregate_json(record, allowed_fields=AXIS_STATUS_FIELDS))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare-public")
    prepare_parser.add_argument("--public-root", type=Path, required=True)
    prepare_parser.add_argument("--config", type=Path, required=True)
    qualify_parser = subparsers.add_parser("qualify-public")
    qualify_parser.add_argument("--public-root", type=Path, required=True)
    qualify_parser.add_argument("--config", type=Path, required=True)
    qualify_parser.add_argument("--device", default="cuda")
    activity_prepare_parser = subparsers.add_parser("activity-prepare")
    activity_prepare_parser.add_argument("--public-root", type=Path, required=True)
    activity_prepare_parser.add_argument("--manifest", type=Path, required=True)
    activity_prepare_parser.add_argument("--config", type=Path, required=True)
    activity_candidate_parser = subparsers.add_parser("activity-candidate")
    activity_candidate_parser.add_argument("--public-root", type=Path, required=True)
    activity_candidate_parser.add_argument("--manifest", type=Path, required=True)
    activity_candidate_parser.add_argument("--config", type=Path, required=True)
    activity_candidate_parser.add_argument("--candidate-id", required=True)
    activity_candidate_parser.add_argument(
        "--partition", choices=("development", "holdout"), required=True
    )
    activity_candidate_parser.add_argument("--device", default="cuda")
    activity_select_parser = subparsers.add_parser("activity-select")
    activity_select_parser.add_argument("--public-root", type=Path, required=True)
    activity_select_parser.add_argument("--manifest", type=Path, required=True)
    activity_select_parser.add_argument("--config", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--restricted-root", type=Path, required=True)
    run_parser.add_argument("--public-root", type=Path, required=True)
    run_parser.add_argument("--scratch-root", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--restricted-root", type=Path, required=True)
    axis_parser = subparsers.add_parser("report-axis-status")
    axis_parser.add_argument("--restricted-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-public":
        value = prepare_public(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=PUBLIC_PREP_FIELDS,
                sha256_fields=PUBLIC_PREP_HASH_FIELDS,
            )
        )
    elif args.command == "qualify-public":
        value = qualify_public(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=PUBLIC_QUALIFICATION_FIELDS,
                sha256_fields=PUBLIC_QUALIFICATION_HASH_FIELDS,
            )
        )
    elif args.command == "activity-prepare":
        value = prepare_activity_public(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=ACTIVITY_PREP_FIELDS,
                sha256_fields=ACTIVITY_PREP_HASH_FIELDS,
            )
        )
    elif args.command == "activity-candidate":
        value = run_activity_candidate(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=ACTIVITY_CANDIDATE_FIELDS,
                sha256_fields=ACTIVITY_CANDIDATE_HASH_FIELDS,
            )
        )
    elif args.command == "activity-select":
        value = select_activity_public(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=ACTIVITY_SELECTION_FIELDS,
                sha256_fields=ACTIVITY_SELECTION_HASH_FIELDS,
            )
        )
    elif args.command == "run":
        value = run(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TERMINAL_FIELDS,
                sha256_fields=TERMINAL_HASH_FIELDS,
            )
        )
    elif args.command == "report":
        report(args)
    else:
        report_axis_status(args)


if __name__ == "__main__":
    main()
