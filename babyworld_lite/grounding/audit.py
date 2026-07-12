from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from babyworld_lite.grounding.config import PROVISIONAL_CALIBRATION_STATUS
from babyworld_lite.grounding.pipeline import observable_leakage_paths
from babyworld_lite.sim import FPS


def _hash_utterances(record: Mapping[str, Any]) -> str:
    texts = sorted(str(item["text"]) for item in record["model_inputs"]["utterances"])
    return hashlib.sha256(json.dumps(texts).encode()).hexdigest()


def _normalized(weights: Mapping[Any, Any]) -> dict[str, float]:
    total = sum(float(value) for value in weights.values())
    return {str(key): float(value) / total for key, value in weights.items()}


def audit_records(
    examples: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    oracle_by_id = {row["example_id"]: row for row in oracle}
    if len(oracle_by_id) != len(examples):
        failures.append("oracle/example cardinality mismatch")
    leakage = {row["example_id"]: observable_leakage_paths(row) for row in examples}
    leakage = {key: value for key, value in leakage.items() if value}
    if leakage:
        failures.append("model input allowlist/leakage violations")

    alignment_episode_multisets: dict[str, Counter[int]] = defaultdict(Counter)
    alignment_utterance_multisets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in examples:
        alignment_episode_multisets[row["alignment_condition"]][int(row["base_episode_id"])] += 1
        alignment_utterance_multisets[row["alignment_condition"]][_hash_utterances(row)] += 1
    episode_multisets_equal = len({tuple(sorted(counter.items())) for counter in alignment_episode_multisets.values()}) == 1
    utterance_multisets_equal = len({tuple(sorted(counter.items())) for counter in alignment_utterance_multisets.values()}) == 1
    if not episode_multisets_equal:
        failures.append("episode multisets differ across alignment arms")
    if not utterance_multisets_equal:
        failures.append("utterance multisets differ across alignment arms")

    split_hashes: dict[str, set[str]] = defaultdict(set)
    split_compositions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    base_seen: set[int] = set()
    shape_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    material_counts: Counter[str] = Counter()
    distractor_counts: Counter[str] = Counter()
    activity_window_counts: Counter[str] = Counter()
    visibility_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    camera_motion_counts: Counter[str] = Counter()
    utterance_count_counts: Counter[str] = Counter()
    utterance_length_counts: Counter[str] = Counter()
    interval_counts: Counter[str] = Counter()
    target_word_counts: Counter[str] = Counter()
    total_activity_frames = 0
    total_utterances = 0
    silent_activity_frames = 0
    shuffled_self_matches: list[str] = []
    cross_split_donors: list[str] = []
    base_split = {int(row["base_episode_id"]): row["split"] for row in examples}
    for row in examples:
        split_hashes[row["split"]].add(row["episode_hash"])
        target = row["evaluation_targets"]
        split_compositions[row["split"]].add((target["object_noun"], target["action_verb"]))
        if int(row["base_episode_id"]) not in base_seen:
            base_seen.add(int(row["base_episode_id"]))
            shape_counts[target["object_noun"]] += 1
            action_counts[target["action_verb"]] += 1
            material_counts[oracle_by_id[row["example_id"]]["object"]["material"]] += 1
            realization = oracle_by_id[row["example_id"]].get("calibration_realization", {})
            required_realization = {
                "activity_window_frames", "target_visible", "target_occlusion_fraction",
                "camera_motion_amplitude_px", "distractor_count", "utterance_count",
            }
            if set(realization) != required_realization:
                failures.append(f"missing calibration realization for base {row['base_episode_id']}")
            else:
                duration = int(realization["activity_window_frames"])
                utterances = row["model_inputs"]["utterances"]
                total_activity_frames += duration
                total_utterances += len(utterances)
                onsets_in_window = {
                    int(item["onset_frame"]) for item in utterances
                    if 0 <= int(item["onset_frame"]) < duration
                }
                silent_activity_frames += duration - len(onsets_in_window)
                distractor_counts[str(realization["distractor_count"])] += 1
                activity_window_counts[str(duration)] += 1
                visibility_counts["visible" if realization["target_visible"] else "invisible"] += 1
                occlusion_counts[str(realization["target_occlusion_fraction"])] += 1
                camera_motion_counts[str(realization["camera_motion_amplitude_px"])] += 1
                utterance_count_counts[str(len(utterances))] += 1
                word_inventory = {
                    *map(str, config["distributions"]["target_shapes"]),
                    *map(str, config["distributions"]["actions"]),
                }
                for item in utterances:
                    words = re.findall(r"[a-z]+(?:'[a-z]+)?", str(item["text"]).lower())
                    utterance_length_counts[str(len(words))] += 1
                    target_word_counts.update(word for word in words if word in word_inventory)
                onsets = sorted(int(item["onset_frame"]) for item in utterances)
                interval_counts.update(str(right - left) for left, right in zip(onsets, onsets[1:]))
        metadata = oracle_by_id.get(row["example_id"], {})
        if row["alignment_condition"] == "shuffled" and metadata.get("language_source_episode_id") == row["base_episode_id"]:
            shuffled_self_matches.append(row["example_id"])
        if row["motor_condition"] == "shuffled" and metadata.get("motor_source_episode_id") == row["base_episode_id"]:
            shuffled_self_matches.append(row["example_id"])
        if row["alignment_condition"] == "shuffled" and base_split.get(metadata.get("language_source_episode_id")) != row["split"]:
            cross_split_donors.append(row["example_id"])
        if row["motor_condition"] == "shuffled" and base_split.get(metadata.get("motor_source_episode_id")) != row["split"]:
            cross_split_donors.append(row["example_id"])
    split_pairs = [(left, right) for i, left in enumerate(split_hashes) for right in list(split_hashes)[i + 1:]]
    hash_overlap = {f"{left}|{right}": sorted(split_hashes[left] & split_hashes[right]) for left, right in split_pairs if split_hashes[left] & split_hashes[right]}
    composition_overlap = {f"{left}|{right}": sorted(split_compositions[left] & split_compositions[right]) for left, right in split_pairs if split_compositions[left] & split_compositions[right]}
    if hash_overlap:
        failures.append("episode hashes overlap across splits")
    if composition_overlap:
        failures.append("compositions overlap across splits")
    if shuffled_self_matches:
        failures.append("shuffled cue/language self matches found")
    if cross_split_donors:
        failures.append("shuffled donors cross split boundaries")

    profile_statuses = sorted({row["profile"]["calibration_status"] for row in examples})
    if profile_statuses != [PROVISIONAL_CALIBRATION_STATUS]:
        failures.append("profile is not explicitly provisional")
    expected_conditions = len(config["conditions"]["language_alignment"]) * len(config["conditions"]["motor_cues"])
    expected_count = len(base_seen) * expected_conditions
    if len(examples) != expected_count:
        failures.append("factorial condition coverage is incomplete")

    realized_rate = (
        total_utterances / (total_activity_frames / FPS) * 60.0
        if total_activity_frames else 0.0
    )
    silent_fraction = silent_activity_frames / total_activity_frames if total_activity_frames else 0.0

    return {
        "valid": not failures,
        "failures": failures,
        "profile": {"name": config["profile"]["name"], "calibration_status": profile_statuses},
        "counts": {
            "base_episodes": len(base_seen), "examples": len(examples),
            "splits": dict(Counter(row["split"] for row in examples)),
            "alignment_conditions": dict(Counter(row["alignment_condition"] for row in examples)),
            "motor_conditions": dict(Counter(row["motor_condition"] for row in examples)),
            "target_shapes": dict(shape_counts), "actions": dict(action_counts), "materials": dict(material_counts),
        },
        "configured_distributions": {
            key: _normalized(config["distributions"][key])
            for key in ("target_shapes", "actions", "materials", "distractor_count")
        },
        "configured_calibration_targets": {
            "utterance_rate_per_minute": float(config["calibration_targets"]["utterance_rate_per_minute"]),
            **{
                key: _normalized(config["calibration_targets"][key])
                for key in (
                    "utterances_per_activity_window", "utterance_length_words",
                    "inter_utterance_interval_frames", "activity_window_frames",
                    "target_visibility", "target_occlusion_fraction",
                    "camera_motion_amplitude_px",
                )
            },
        },
        "realized_calibration": {
            "utterance_rate_per_minute": round(realized_rate, 4),
            "utterances_per_activity_window": dict(utterance_count_counts),
            "utterance_length_words": dict(utterance_length_counts),
            "inter_utterance_interval_frames": dict(interval_counts),
            "silent_activity_frame_fraction": round(silent_fraction, 6),
            "activity_window_frames": dict(activity_window_counts),
            "target_visibility": dict(visibility_counts),
            "target_occlusion_fraction": dict(occlusion_counts),
            "camera_motion_amplitude_px": dict(camera_motion_counts),
            "distractor_count": dict(distractor_counts),
            "target_word_frequency": dict(target_word_counts),
        },
        "fairness": {
            "episode_multisets_identical_across_alignment": episode_multisets_equal,
            "utterance_multisets_identical_across_alignment": utterance_multisets_equal,
            "shuffled_self_matches": shuffled_self_matches,
            "cross_split_donors": cross_split_donors,
            "split_hash_overlap": hash_overlap,
            "split_composition_overlap": composition_overlap,
        },
        "leakage": {"model_input_violations": leakage, "oracle_file_separate": True},
        "renderer": {"policy": "scene_pixels_only_no_text_or_contact_marker"},
    }
