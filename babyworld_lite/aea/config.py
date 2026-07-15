from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


AEA_SCIENTIFIC_ROLE = "adult_partly_scripted_sensor_format_analogue_not_developmental_evidence"
AEA_ARMS = {"null", "synchronized", "shuffled", "time_shifted"}


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return result


def _lexicon(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or len(value) < 2:
        raise ValueError(f"{name} must contain at least two canonical labels")
    forms: set[str] = set()
    for canonical, variants in value.items():
        if not str(canonical).strip() or not isinstance(variants, list) or not variants:
            raise ValueError(f"{name}.{canonical} must be a non-empty list")
        normalized = {str(item).lower() for item in variants}
        overlap = forms & normalized
        if overlap:
            raise ValueError(f"{name} has ambiguous forms: {sorted(overlap)}")
        forms |= normalized


def validate_aea_config(config: Mapping[str, Any]) -> None:
    profile = _mapping(config, "profile")
    if profile.get("scientific_role") != AEA_SCIENTIFIC_ROLE:
        raise ValueError(
            "AEA must be labeled as an adult, partly scripted sensor-format analogue; "
            "developmental-evidence claims are not allowed"
        )
    source = _mapping(config, "source")
    if not str(source.get("dataset_root", "")).strip():
        raise ValueError("source.dataset_root is required")
    if not str(source.get("subset_plan", "")).strip():
        raise ValueError("source.subset_plan is required")
    if not str(source.get("release", "")).strip():
        raise ValueError("source.release is required")

    window = _mapping(config, "window")
    duration = float(window.get("duration_seconds", 0))
    if duration <= 0:
        raise ValueError("window.duration_seconds must be positive")
    if int(window.get("rgb_frames", 0)) < 2:
        raise ValueError("window.rgb_frames must be at least 2")
    if float(window.get("imu_rate_hz", 0)) <= 0:
        raise ValueError("window.imu_rate_hz must be positive")
    if int(window.get("max_windows_per_action_per_sequence", 0)) < 1:
        raise ValueError("window.max_windows_per_action_per_sequence must be positive")
    if int(window.get("max_windows_per_sequence", 0)) < 2:
        raise ValueError("window.max_windows_per_sequence must be at least 2")

    labels = _mapping(config, "labels")
    _lexicon(labels.get("actions"), "labels.actions")
    _lexicon(labels.get("objects"), "labels.objects")
    if not 0 <= float(labels.get("minimum_asr_confidence", -1)) <= 1:
        raise ValueError("labels.minimum_asr_confidence must be in [0, 1]")
    if float(labels.get("object_max_gap_seconds", 0)) <= 0:
        raise ValueError("labels.object_max_gap_seconds must be positive")

    quality = _mapping(config, "quality")
    if not 0 < float(quality.get("minimum_imu_coverage", 0)) <= 1:
        raise ValueError("quality.minimum_imu_coverage must be in (0, 1]")
    if float(quality.get("maximum_imu_gap_ms", 0)) <= 0:
        raise ValueError("quality.maximum_imu_gap_ms must be positive")
    if float(quality.get("maximum_frame_time_error_ms", 0)) <= 0:
        raise ValueError("quality.maximum_frame_time_error_ms must be positive")

    experiment = _mapping(config, "experiment")
    arms = experiment.get("arms")
    if not isinstance(arms, list) or set(arms) != AEA_ARMS:
        raise ValueError(f"experiment.arms must contain exactly {sorted(AEA_ARMS)}")
    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(map(int, seeds))) != len(seeds):
        raise ValueError("experiment.seeds must contain at least two unique paired seeds")
    holdouts = experiment.get("held_out_compositions")
    if not isinstance(holdouts, list) or not holdouts:
        raise ValueError("experiment.held_out_compositions must be non-empty")
    for item in holdouts:
        if not isinstance(item, Mapping) or set(item) != {"action", "object"}:
            raise ValueError("each held-out composition needs only action and object")
    locked_training = _mapping(experiment, "locked_training")
    required_training = {
        "frame_count", "image_size", "max_text_length", "hidden_dim",
        "embedding_dim", "batch_size", "epochs", "learning_rate",
        "motor_weight", "time_shift", "bootstrap_samples", "motor_sample_count",
    }
    if set(locked_training) != required_training:
        raise ValueError(
            "experiment.locked_training must contain exactly "
            f"{sorted(required_training)}"
        )


def load_aea_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("AEA config must be a YAML mapping")
    output = copy.deepcopy(dict(config))
    validate_aea_config(output)
    return output
