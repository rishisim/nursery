from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


PROVISIONAL_CALIBRATION_STATUS = "provisional_not_babyview_matched"
ALIGNMENT_CONDITIONS = {"strong", "weak", "shuffled"}
MOTOR_CONDITIONS = {"null", "synchronized", "shuffled", "time_shifted"}


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _validate_probability_mapping(values: Mapping[str, Any], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    numeric = {str(key): float(value) for key, value in values.items()}
    if any(value < 0 for value in numeric.values()):
        raise ValueError(f"{name} weights must be non-negative")
    total = sum(numeric.values())
    if total <= 0:
        raise ValueError(f"{name} must have positive total weight")


def _composition_set(value: Any, name: str) -> set[tuple[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    output: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"shape", "action"}:
            raise ValueError(f"each {name} entry must contain only shape and action")
        output.add((str(item["shape"]), str(item["action"])))
    return output


def validate_grounding_config(config: Mapping[str, Any]) -> None:
    profile = _require_mapping(config, "profile")
    if not str(profile.get("name", "")).strip():
        raise ValueError("profile.name must be non-empty")
    status = profile.get("calibration_status")
    if status != PROVISIONAL_CALIBRATION_STATUS:
        raise ValueError(
            "pre-access grounding configs must set profile.calibration_status to "
            f"{PROVISIONAL_CALIBRATION_STATUS!r}; BabyView-matched claims are not allowed"
        )

    generation = _require_mapping(config, "generation")
    if int(generation.get("base_episodes", 0)) < 3:
        raise ValueError("generation.base_episodes must be at least 3 for cue shuffling")
    if int(generation.get("frame_stride", 0)) < 1:
        raise ValueError("generation.frame_stride must be at least 1")

    distributions = _require_mapping(config, "distributions")
    for key in ("target_shapes", "actions", "materials", "distractor_count"):
        values = distributions.get(key)
        if not isinstance(values, Mapping):
            raise ValueError(f"distributions.{key} must be a mapping")
        _validate_probability_mapping(values, f"distributions.{key}")

    calibration = _require_mapping(config, "calibration_targets")
    if float(calibration.get("utterance_rate_per_minute", 0)) <= 0:
        raise ValueError("calibration_targets.utterance_rate_per_minute must be positive")
    for key in (
        "utterances_per_activity_window",
        "utterance_length_words",
        "inter_utterance_interval_frames",
        "activity_window_frames",
        "target_visibility",
        "target_occlusion_fraction",
        "camera_motion_amplitude_px",
    ):
        values = calibration.get(key)
        if not isinstance(values, Mapping):
            raise ValueError(f"calibration_targets.{key} must be a mapping")
        _validate_probability_mapping(values, f"calibration_targets.{key}")
    if any(int(value) < 1 for value in calibration["utterances_per_activity_window"]):
        raise ValueError("utterances_per_activity_window keys must be positive integers")
    if any(int(value) < 1 for value in calibration["utterance_length_words"]):
        raise ValueError("utterance_length_words keys must be positive integers")
    if any(int(value) < 1 for value in calibration["inter_utterance_interval_frames"]):
        raise ValueError("inter_utterance_interval_frames keys must be positive integers")
    if any(not 1 <= int(value) <= 28 for value in calibration["activity_window_frames"]):
        raise ValueError("activity_window_frames keys must be between 1 and 28")
    if set(str(value) for value in calibration["target_visibility"]) != {"visible", "invisible"}:
        raise ValueError("target_visibility must contain visible and invisible")
    if any(not 0.0 <= float(value) <= 1.0 for value in calibration["target_occlusion_fraction"]):
        raise ValueError("target_occlusion_fraction keys must be between 0 and 1")

    conditions = _require_mapping(config, "conditions")
    alignment = conditions.get("language_alignment")
    motor = conditions.get("motor_cues")
    if not isinstance(alignment, list) or set(alignment) != ALIGNMENT_CONDITIONS:
        raise ValueError(
            "conditions.language_alignment must contain exactly strong, weak, and shuffled"
        )
    if not isinstance(motor, list) or set(motor) != MOTOR_CONDITIONS:
        raise ValueError(
            "conditions.motor_cues must contain exactly null, synchronized, shuffled, and time_shifted"
        )
    if int(conditions.get("motor_time_shift_frames", 0)) == 0:
        raise ValueError("conditions.motor_time_shift_frames must be non-zero")

    weak = _require_mapping(config, "weak_alignment")
    mixture = weak.get("mixture")
    if not isinstance(mixture, Mapping):
        raise ValueError("weak_alignment.mixture must be a mapping")
    required_mixture = {"current", "delayed", "ambiguous", "irrelevant", "silent"}
    if set(mixture) != required_mixture:
        raise ValueError(
            "weak_alignment.mixture must contain current, delayed, ambiguous, irrelevant, and silent"
        )
    _validate_probability_mapping(mixture, "weak_alignment.mixture")
    if not isinstance(weak.get("irrelevant_utterances"), list) or not weak["irrelevant_utterances"]:
        raise ValueError("weak_alignment.irrelevant_utterances must be a non-empty list")

    splits = _require_mapping(config, "splits")
    validation = _composition_set(splits.get("validation_compositions"), "validation_compositions")
    test = _composition_set(splits.get("test_compositions"), "test_compositions")
    overlap = validation & test
    if overlap:
        raise ValueError(f"validation and test compositions overlap: {sorted(overlap)}")
    if int(splits.get("minimum_base_episodes_per_holdout", 0)) < 2:
        raise ValueError("splits.minimum_base_episodes_per_holdout must be at least 2 for within-split shuffling")
    reserve = (len(validation) + len(test)) * int(splits["minimum_base_episodes_per_holdout"])
    if reserve >= int(generation["base_episodes"]):
        raise ValueError("held-out reserves must leave at least one training base episode")


def load_grounding_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("grounding configuration must be a YAML mapping")
    output = copy.deepcopy(dict(config))
    validate_grounding_config(output)
    return output
