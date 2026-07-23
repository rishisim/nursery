from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .protocol import canonical_digest, guard_seed_operation
from .synthetic import CalibrationData


@dataclass(frozen=True, slots=True)
class SensorEventDetector:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    activity_weights: tuple[float, ...]
    boundary_weights: tuple[float, ...]
    training_digest: str

    def serializable(self) -> dict[str, Any]:
        return {
            "schema_version": "synthetic-sensor-event-detector-v2",
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "activity_weights": list(self.activity_weights),
            "boundary_weights": list(self.boundary_weights),
            "training_digest": self.training_digest,
            "lexical_supervision_used": False,
            "referent_supervision_used": False,
            "randomized_mapping_used": False,
        }


def _moving_mean(values: np.ndarray, width: int = 5) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _moving_std(values: np.ndarray, width: int = 5) -> np.ndarray:
    mean = _moving_mean(values, width)
    mean_square = _moving_mean(values * values, width)
    return np.sqrt(np.clip(mean_square - mean * mean, 0.0, None))


def raw_features(raw: Mapping[str, Any]) -> np.ndarray:
    imu = np.asarray(raw["imu"], dtype=np.float64)
    proprio = np.asarray(raw["proprio"], dtype=np.float64)
    contact = np.asarray(raw["contact"], dtype=np.float64)[:, 0]
    availability = np.asarray(raw["availability"], dtype=np.float64)
    accel = np.linalg.norm(imu[:, :3], axis=1)
    gyro = np.linalg.norm(imu[:, 3:], axis=1)
    proprio_energy = np.linalg.norm(proprio, axis=1)
    total = accel + gyro + proprio_energy + contact
    delta_accel = np.abs(np.gradient(accel))
    delta_gyro = np.abs(np.gradient(gyro))
    delta_state = np.abs(np.gradient(proprio_energy))
    return np.stack(
        [
            accel,
            gyro,
            proprio_energy,
            contact,
            delta_accel,
            delta_gyro,
            delta_state,
            _moving_mean(total),
            _moving_std(total),
            availability.mean(axis=1),
        ],
        axis=1,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> np.ndarray:
    x = np.column_stack([np.ones(len(features)), features])
    y = np.asarray(labels, dtype=np.float64)
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    positive_weight = min(12.0, negatives / positives)
    sample_weight = np.where(y > 0.5, positive_weight, 1.0)
    sample_weight /= sample_weight.mean()
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(iterations):
        prediction = _sigmoid(x @ weights)
        gradient = x.T @ (sample_weight * (prediction - y)) / len(x)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights


def fit_detector(
    calibration: CalibrationData, config: Mapping[str, Any]
) -> tuple[SensorEventDetector, dict[str, Any]]:
    if calibration.split != "train":
        raise ValueError("detector fitting requires the generic calibration train split")
    for seed in calibration.seeds:
        guard_seed_operation(config, operation="calibrate", seeds=[seed])
    oracle_by_id = {
        str(row["calibration_id"]): row for row in calibration.oracle_records
    }
    features: list[np.ndarray] = []
    active: list[np.ndarray] = []
    boundary: list[np.ndarray] = []
    for row in calibration.visible_records:
        oracle = oracle_by_id[str(row["calibration_id"])]
        features.append(raw_features(row["raw_stream"]))
        active.append(np.asarray(oracle["wearer_active"], dtype=np.float64))
        boundary.append(np.asarray(oracle["wearer_boundary"], dtype=np.float64))
    x = np.concatenate(features, axis=0)
    y_active = np.concatenate(active)
    y_boundary = np.concatenate(boundary)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (x - mean) / scale
    detector_config = config["calibration"]
    activity_weights = _fit_logistic(
        standardized,
        y_active,
        iterations=int(detector_config["detector_iterations"]),
        learning_rate=float(detector_config["detector_learning_rate"]),
        l2=float(detector_config["detector_l2"]),
    )
    boundary_weights = _fit_logistic(
        standardized,
        y_boundary,
        iterations=int(detector_config["detector_iterations"]),
        learning_rate=float(detector_config["detector_learning_rate"]),
        l2=float(detector_config["detector_l2"]),
    )
    training_digest = canonical_digest(
        {
            "provenance": calibration.provenance,
            "features": standardized.astype(float).tolist(),
            "active": y_active.astype(int).tolist(),
            "boundary": y_boundary.astype(int).tolist(),
        }
    )
    model = SensorEventDetector(
        feature_mean=tuple(map(float, mean)),
        feature_scale=tuple(map(float, scale)),
        activity_weights=tuple(map(float, activity_weights)),
        boundary_weights=tuple(map(float, boundary_weights)),
        training_digest=training_digest,
    )
    trace = {
        "schema_version": "synthetic-sensor-event-detector-fit-trace-v2",
        "calibration_seeds": list(calibration.seeds),
        "calibration_rows": len(calibration.visible_records),
        "timepoint_rows": len(x),
        "activity_positive_rate": float(y_active.mean()),
        "boundary_positive_rate": float(y_boundary.mean()),
        "iterations_per_head": int(detector_config["detector_iterations"]),
        "model_digest": canonical_digest(model.serializable()),
        "training_digest": training_digest,
        "lexical_supervision_used": False,
        "referent_supervision_used": False,
        "randomized_mapping_used": False,
    }
    return model, trace


def timepoint_probabilities(
    detector: SensorEventDetector, raw: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    features = raw_features(raw)
    standardized = (
        features - np.asarray(detector.feature_mean, dtype=np.float64)
    ) / np.asarray(detector.feature_scale, dtype=np.float64)
    x = np.column_stack([np.ones(len(features)), standardized])
    activity = _sigmoid(x @ np.asarray(detector.activity_weights, dtype=np.float64))
    boundary = _sigmoid(x @ np.asarray(detector.boundary_weights, dtype=np.float64))
    return activity, boundary


def candidate_evidence(
    detector: SensorEventDetector,
    raw: Mapping[str, Any],
    intervals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    activity, boundary = timepoint_probabilities(detector, raw)
    values: list[float] = []
    owner_probabilities: list[float] = []
    for interval in intervals:
        start = int(interval["start"])
        end = int(interval["end"])
        inside = activity[start : end + 1]
        owner_probability = float(np.mean(inside))
        owner_probabilities.append(owner_probability)
        endpoint_boundary = float(
            0.5
            * (
                np.max(boundary[max(0, start - 1) : min(len(boundary), start + 2)])
                + np.max(boundary[max(0, end - 1) : min(len(boundary), end + 2)])
            )
        )
        local_start = max(0, start - 3)
        local_end = min(len(activity), end + 4)
        outside_values = np.concatenate(
            [activity[local_start:start], activity[end + 1 : local_end]]
        )
        outside = float(np.mean(outside_values)) if len(outside_values) else 0.0
        probability = np.clip(owner_probability, 1e-5, 1.0 - 1e-5)
        logit = float(np.log(probability / (1.0 - probability)))
        values.append(logit + 0.55 * endpoint_boundary - 0.25 * outside)
    maximum = max(owner_probabilities, default=0.0)
    null_probability = float(np.clip(1.0 - maximum, 1e-5, 1.0 - 1e-5))
    null_logit = float(np.log(null_probability / (1.0 - null_probability)))
    sorted_probabilities = sorted(owner_probabilities, reverse=True)
    margin = (
        sorted_probabilities[0] - sorted_probabilities[1]
        if len(sorted_probabilities) > 1
        else (sorted_probabilities[0] if sorted_probabilities else 0.0)
    )
    availability = float(np.asarray(raw["availability"], dtype=float).mean())
    quality = float(np.clip((maximum - 0.45) * 1.8 + margin, 0.0, 1.0))
    return {
        "event_logits": tuple(values),
        "null_logit": null_logit,
        "owner_probabilities": tuple(owner_probabilities),
        "quality": quality,
        "availability": availability,
        "top_event_index": int(np.argmax(values)) if values else None,
        "timepoint_activity_probability": tuple(map(float, activity)),
        "timepoint_boundary_probability": tuple(map(float, boundary)),
    }


def evidence_for_episodes(
    detector: SensorEventDetector,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in episodes:
        if "raw_stream" not in row:
            continue
        output[str(row["episode_id"])] = candidate_evidence(
            detector, row["raw_stream"], row["events"]
        )
    return output


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _boundary_match_metrics(
    truth_indices: Sequence[int], prediction_indices: Sequence[int], tolerance: int
) -> dict[str, float]:
    unmatched = list(map(int, truth_indices))
    matched = 0
    for prediction in sorted(map(int, prediction_indices)):
        candidates = [
            (abs(prediction - truth), index)
            for index, truth in enumerate(unmatched)
            if abs(prediction - truth) <= tolerance
        ]
        if candidates:
            _, index = min(candidates)
            unmatched.pop(index)
            matched += 1
    precision = matched / max(len(prediction_indices), 1)
    recall = matched / max(len(truth_indices), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return float(wins / (len(positives) * len(negatives)))


def evaluate_detector(
    detector: SensorEventDetector,
    calibration: CalibrationData,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if calibration.split != "validation":
        raise ValueError("detector evaluation requires held-out generic calibration validation")
    for seed in calibration.seeds:
        guard_seed_operation(config, operation="evaluate", seeds=[seed])
    oracle_by_id = {
        str(row["calibration_id"]): row for row in calibration.oracle_records
    }
    activity_truth: list[int] = []
    activity_prediction: list[int] = []
    boundary_truth_indices: list[int] = []
    boundary_prediction_indices: list[int] = []
    offset = 0
    informative_owner_labels: list[int] = []
    informative_owner_scores: list[float] = []
    zero_owner_labels: list[int] = []
    zero_owner_scores: list[float] = []
    per_level: dict[str, dict[str, list[float]]] = {}
    activity_threshold = float(config["calibration"]["ownership_threshold"])
    boundary_threshold = float(config["calibration"]["boundary_threshold"])
    for row in calibration.visible_records:
        oracle = oracle_by_id[str(row["calibration_id"])]
        evidence = candidate_evidence(
            detector,
            row["raw_stream"],
            row["candidate_intervals"],
        )
        information = float(oracle["sensor_informativeness"])
        level = f"{information:g}"
        per_level.setdefault(level, {"labels": [], "scores": []})
        labels = [int(owner == "wearer") for owner in oracle["event_owners"]]
        scores = list(map(float, evidence["owner_probabilities"]))
        per_level[level]["labels"].extend(labels)
        per_level[level]["scores"].extend(scores)
        if information > 0:
            informative_owner_labels.extend(labels)
            informative_owner_scores.extend(scores)
            truth = np.asarray(oracle["wearer_active"], dtype=int)
            prediction = np.asarray(evidence["timepoint_activity_probability"]) >= activity_threshold
            activity_truth.extend(truth.tolist())
            activity_prediction.extend(prediction.astype(int).tolist())
            boundary_truth_indices.extend(
                [offset + int(index) for index in np.flatnonzero(oracle["wearer_boundary"])]
            )
            boundary_probability = np.asarray(
                evidence["timepoint_boundary_probability"], dtype=float
            )
            local_maximum = np.r_[
                False,
                (boundary_probability[1:-1] >= boundary_probability[:-2])
                & (boundary_probability[1:-1] >= boundary_probability[2:]),
                False,
            ]
            boundary_prediction_indices.extend(
                [
                    offset + int(index)
                    for index in np.flatnonzero(
                        local_maximum & (boundary_probability >= boundary_threshold)
                    )
                ]
            )
        else:
            zero_owner_labels.extend(labels)
            zero_owner_scores.extend(scores)
        offset += len(oracle["wearer_active"]) + 10
    activity_metrics = _binary_metrics(
        np.asarray(activity_truth, dtype=int),
        np.asarray(activity_prediction, dtype=int),
    )
    boundary_metrics = _boundary_match_metrics(
        boundary_truth_indices,
        boundary_prediction_indices,
        int(config["calibration"]["boundary_tolerance_samples"]),
    )
    informative_auc = _auc(informative_owner_labels, informative_owner_scores)
    zero_auc = _auc(zero_owner_labels, zero_owner_scores)
    thresholds = config["gates"]["detector"]
    checks = {
        "informative_timepoint_precision": activity_metrics["precision"]
        >= float(thresholds["minimum_informative_timepoint_precision"]),
        "informative_timepoint_recall": activity_metrics["recall"]
        >= float(thresholds["minimum_informative_timepoint_recall"]),
        "informative_boundary_f1": boundary_metrics["f1"]
        >= float(thresholds["minimum_informative_boundary_f1"]),
        "informative_candidate_owner_auc": informative_auc
        >= float(thresholds["minimum_informative_candidate_owner_auc"]),
        "zero_information_candidate_owner_auc_near_chance": abs(zero_auc - 0.5)
        <= float(thresholds["maximum_zero_information_auc_distance_from_chance"]),
    }
    return {
        "schema_version": "synthetic-sensor-event-detector-validation-v2",
        "valid": all(checks.values()),
        "checks": checks,
        "heldout_calibration_seeds": list(calibration.seeds),
        "informative_timepoint": activity_metrics,
        "informative_boundary": boundary_metrics,
        "informative_candidate_owner_auc": informative_auc,
        "zero_information_candidate_owner_auc": zero_auc,
        "candidate_owner_auc_by_informativeness": {
            level: _auc(values["labels"], values["scores"])
            for level, values in sorted(per_level.items(), key=lambda item: float(item[0]))
        },
        "detector_digest": canonical_digest(detector.serializable()),
        "lexical_supervision_used": False,
        "referent_supervision_used": False,
        "randomized_mapping_used": False,
    }
