#!/usr/bin/env python3
"""Governed bounded C calibration and aggregate-conditioned episode planning."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
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
import tarfile
import time
from typing import Any
import urllib.error
import urllib.request
import zipfile

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
ACTIVITY_SIZING_FIELDS = frozenset(
    {
        "status",
        "item_count",
        "output_width",
        "expected_output_width",
        "load_runtime_seconds",
        "item_runtime_seconds",
        "total_runtime_seconds",
        "peak_vram_gib",
        "expected_peak_vram_gib_max",
        "external_call_count",
        "activity_sizing_commitment_sha256",
    }
)
ACTIVITY_SIZING_HASH_FIELDS = frozenset(
    {"activity_sizing_commitment_sha256"}
)
ACTIVITY_SELECTION_FIELDS = frozenset(
    {
        "status",
        "candidate_count",
        "eligible_candidate_count",
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
TUPLE_PREP_FIELDS = frozenset(
    {
        "status",
        "component_count",
        "repository_count",
        "weight_file_count",
        "license_record_count",
        "artifact_bytes",
        "archive_license_file_count",
        "restricted_mount_present",
        "model_inference_executed",
        "tuple_dependency_commitment_sha256",
    }
)
TUPLE_PREP_HASH_FIELDS = frozenset({"tuple_dependency_commitment_sha256"})
TUPLE_RUNTIME_PREP_FIELDS = frozenset(
    {
        "status",
        "dependency_count",
        "dependency_artifact_count",
        "installed_distribution_count",
        "additional_repository_count",
        "additional_model_file_count",
        "restricted_mount_present",
        "model_inference_executed",
        "runtime_dependency_commitment_sha256",
    }
)
TUPLE_RUNTIME_PREP_HASH_FIELDS = frozenset(
    {"runtime_dependency_commitment_sha256"}
)
TUPLE_SIZING_FIELDS = frozenset(
    {
        "status",
        "module_count",
        "finite_output_count",
        "failure_count",
        "external_call_count",
        "scientific_metric_count",
        "retained_prediction_count",
        "peak_vram_gib",
        "total_runtime_seconds",
        "tuple_sizing_commitment_sha256",
    }
)
TUPLE_SIZING_HASH_FIELDS = frozenset({"tuple_sizing_commitment_sha256"})
TUPLE_FIXTURE_PREP_FIELDS = frozenset(
    {
        "status",
        "source_archive_count",
        "partition_count",
        "language_lexical_item_count",
        "referent_attribute_item_count",
        "recurrence_pair_count",
        "hand_contact_item_count",
        "sensor_item_count",
        "order_action_item_count",
        "source_subject_overlap_count",
        "source_video_overlap_count",
        "source_object_overlap_count",
        "restricted_mount_present",
        "model_inference_executed",
        "public_fixture_manifest_commitment_sha256",
    }
)
TUPLE_FIXTURE_PREP_HASH_FIELDS = frozenset(
    {"public_fixture_manifest_commitment_sha256"}
)
TUPLE_AUDIO_SEED_FIELDS = frozenset(
    {"status", "audio_file_count", "audio_seed_commitment_sha256"}
)
TUPLE_AUDIO_SEED_HASH_FIELDS = frozenset({"audio_seed_commitment_sha256"})
TUPLE_FIXTURE_FEASIBILITY_FIELDS = frozenset(
    {
        "status",
        "coco_source_count",
        "visor_item_count",
        "action_item_count",
        "partition_count",
        "failing_family_count",
        "blocking_family",
        "blocking_partition",
        "blocking_stratum",
        "required_count",
        "available_count",
        "source_subject_overlap_count",
        "source_video_overlap_count",
        "source_object_overlap_count",
        "model_inference_executed",
        "media_rendering_executed",
        "restricted_mount_present",
        "fixture_feasibility_commitment_sha256",
    }
)
TUPLE_FIXTURE_FEASIBILITY_HASH_FIELDS = frozenset(
    {"fixture_feasibility_commitment_sha256"}
)
NLTK_DATA_COMMIT = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_RESOURCE_ARCHIVES = {
    "wordnet.zip": {
        "relative_url": "packages/corpora/wordnet.zip",
        "sha256": "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59",
    },
    "averaged_perceptron_tagger_eng.zip": {
        "relative_url": "packages/taggers/averaged_perceptron_tagger_eng.zip",
        "sha256": "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b",
    },
}
GROUNDING_DINO_DEFORM_ATTN_SOURCE_SHA256 = (
    "42aa71c7c47e6f930f48100924393adac95eb94aae0eef779bd7cad2d5bcc95d"
)
GROUNDING_DINO_DEFORM_ATTN_PATCHED_SHA256 = (
    "778efabd5d875a4aa457ede6948979a4196844fbd14bf7a76bc4d4b1440122c6"
)
GROUNDING_DINO_MODEL_SOURCE_SHA256 = (
    "cdfb48d5b15d6b98f3d2002f59ae4730740a1ecfbaeba324f6840c5e4666a5b8"
)
GROUNDING_DINO_MODEL_NO_VISUALIZER_SHA256 = (
    "0da7cea7ddbaddced76432d7a8bc13844dc69d3bee3ce5ae674c46fd0339c671"
)
TUPLE_LANGUAGE_ADAPTER_SHA256 = (
    "005f368bef97dfc791f43e45da8bbfe01ea22e8790b2032e9580b14b1ea62ac8"
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


def _tuple_amendment(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        value = cfg["calibration_C"]["extractor"][
            "mechanistic_training_tuple_amendment"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_AMENDMENT_NOT_FROZEN") from error
    if value.get("status") != (
        "FROZEN_BEFORE_NEW_PUBLIC_C_GENERATOR_OR_SYNTHETIC_LEARNER_OUTCOMES"
    ):
        raise RuntimeError("E_TUPLE_AMENDMENT_NOT_FROZEN")
    copy = json.loads(json.dumps(value))
    expected = copy.pop("amendment_commitment_sha256", None)
    if not isinstance(expected, str) or digest(copy) != expected:
        raise RuntimeError("E_TUPLE_AMENDMENT_COMMITMENT")
    axes = value.get("axes")
    if not isinstance(axes, list) or len(axes) != 7:
        raise RuntimeError("E_TUPLE_AXIS_SET")
    ids = [axis.get("id") for axis in axes]
    if len(ids) != len(set(ids)) or ids != [
        "adapter_qualified_yield",
        "noun_adjective_exposure",
        "utterance_centered_referent_visibility_dominance_ambiguity",
        "cross_episode_recurrence",
        "adjective_attribute_contrast",
        "hand_action_coupling",
        "egocentric_sensor_regime",
    ]:
        raise RuntimeError("E_TUPLE_AXIS_SET")
    if sum(axis.get("priority") == "critical" for axis in axes) != 5:
        raise RuntimeError("E_TUPLE_CRITICAL_AXIS_SET")
    if value["broad_activity_context"].get("status") != "DESCRIPTIVE_NONBLOCKING":
        raise RuntimeError("E_TUPLE_ACTIVITY_ROLE")
    return value


def _tuple_runtime_amendment(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        value = cfg["calibration_C"]["extractor"][
            "mechanistic_training_tuple_runtime_amendment"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_RUNTIME_NOT_FROZEN") from error
    if value.get("status") != (
        "FROZEN_BEFORE_LOCAL_RELOAD_SIZING_OR_PUBLIC_MODEL_OUTCOMES"
    ):
        raise RuntimeError("E_TUPLE_RUNTIME_NOT_FROZEN")
    copy = json.loads(json.dumps(value))
    expected = copy.pop("runtime_amendment_commitment_sha256", None)
    if not isinstance(expected, str) or digest(copy) != expected:
        raise RuntimeError("E_TUPLE_RUNTIME_COMMITMENT")
    dependencies = value.get("dependency_versions")
    if not isinstance(dependencies, dict) or len(dependencies) != 55:
        raise RuntimeError("E_TUPLE_RUNTIME_DEPENDENCY_SET")
    if value.get("added_dependency_wheels") != {
        "cloudpickle-3.1.1-py3-none-any.whl": {
            "sha256": "c8c5a44295039331ee9dad40ba100a9c7297b6f988e50e87ccdf3765a668350e",
            "license": "BSD-3-Clause",
            "role": "submitit runtime dependency absent from the pinned base container",
        },
        "submitit-1.5.3-py3-none-any.whl": {
            "sha256": "ccc35100da12fe916541489deccccb6b9fa93dae8c01ade53e7f643552dc1795",
            "license": "MIT",
            "role": "import-only dependency of the pinned EgoBabyVLM alignment package initializer",
        },
    }:
        raise RuntimeError("E_TUPLE_RUNTIME_ADDED_DEPENDENCY_SET")
    if value["local_reload_gate"].get(
        "all_seven_axes_and_order_dependent_action_control_must_pass"
    ) is not True or value["local_reload_gate"].get("module_count") != 8:
        raise RuntimeError("E_TUPLE_RUNTIME_GATE")
    return value


def _tuple_fixture_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        value = cfg["calibration_C"]["extractor"][
            "mechanistic_training_tuple_public_fixture_protocol"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_FIXTURE_PROTOCOL_NOT_FROZEN") from error
    if value.get("status") != "FROZEN_BEFORE_PUBLIC_MODEL_OUTCOMES":
        raise RuntimeError("E_TUPLE_FIXTURE_PROTOCOL_NOT_FROZEN")
    copy = json.loads(json.dumps(value))
    expected = copy.pop("protocol_commitment_sha256", None)
    if not isinstance(expected, str) or digest(copy) != expected:
        raise RuntimeError("E_TUPLE_FIXTURE_PROTOCOL_COMMITMENT")
    action = value.get("order_dependent_action_control")
    labels = action.get("labels") if isinstance(action, dict) else None
    prompts = action.get("prompt_ensembles") if isinstance(action, dict) else None
    code_pairs = action.get("class_code_pairs") if isinstance(action, dict) else None
    if (
        labels
        != [
            "open",
            "close",
            "take",
            "put",
            "sit_down",
            "stand_up",
            "turn_on",
            "turn_off",
        ]
        or set(prompts or {}) != set(labels)
        or any(len(prompts[label]) != 3 for label in labels)
        or not isinstance(code_pairs, list)
        or len(code_pairs) != 4
    ):
        raise RuntimeError("E_TUPLE_ACTION_FIXTURE_SET")
    return value


def _tuple_fixture_preparation_amendment(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    try:
        value = cfg["calibration_C"]["extractor"][
            "mechanistic_training_tuple_fixture_preparation_amendment"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_FIXTURE_PREPARATION_NOT_FROZEN") from error
    if value.get("status") != (
        "FROZEN_BEFORE_MANIFEST_CONSTRUCTION_OR_PUBLIC_MODEL_OUTCOMES"
    ):
        raise RuntimeError("E_TUPLE_FIXTURE_PREPARATION_NOT_FROZEN")
    copy = json.loads(json.dumps(value))
    expected = copy.pop("preparation_amendment_commitment_sha256", None)
    if not isinstance(expected, str) or digest(copy) != expected:
        raise RuntimeError("E_TUPLE_FIXTURE_PREPARATION_COMMITMENT")
    counts = value.get("counts_per_partition")
    if counts != {
        "language_lexical": 48,
        "referent_attribute": 64,
        "recurrence": 64,
        "hand_contact": 40,
        "sensor": 48,
        "order_action": 48,
    }:
        raise RuntimeError("E_TUPLE_FIXTURE_PREPARATION_COUNTS")
    if value.get("partitions") != ["development", "holdout"]:
        raise RuntimeError("E_TUPLE_FIXTURE_PREPARATION_PARTITIONS")
    ontology = value.get("public_object_ontology")
    if ontology != [
        "sports ball",
        "cup",
        "bottle",
        "bowl",
        "book",
        "chair",
        "apple",
        "banana",
    ]:
        raise RuntimeError("E_TUPLE_FIXTURE_PREPARATION_ONTOLOGY")
    return value


def _tuple_fixture_feasibility_repair(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        value = cfg["calibration_C"]["extractor"][
            "mechanistic_training_tuple_fixture_feasibility_repair_amendment"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_FIXTURE_FEASIBILITY_REPAIR_NOT_FROZEN") from error
    if value.get("status") != (
        "FROZEN_AFTER_PREMODEL_FIXTURE_YIELD_STOP_BEFORE_ANY_PUBLIC_MODEL_OUTCOME"
    ):
        raise RuntimeError("E_TUPLE_FIXTURE_FEASIBILITY_REPAIR_NOT_FROZEN")
    copy = json.loads(json.dumps(value))
    expected = copy.pop("fixture_feasibility_repair_commitment_sha256", None)
    if not isinstance(expected, str) or digest(copy) != expected:
        raise RuntimeError("E_TUPLE_FIXTURE_FEASIBILITY_REPAIR_COMMITMENT")
    selection = value.get("source_selection_repair")
    if (
        selection.get("old_COCO_area_fraction_range") != [0.03, 0.5]
        or selection.get("active_COCO_area_fraction_range") != [0.0, 0.5]
        or selection.get("unchanged_target_bbox_minimum_pixels") != [48, 48]
        or value.get("scientific_thresholds_changed") is not False
    ):
        raise RuntimeError("E_TUPLE_FIXTURE_FEASIBILITY_REPAIR_SCOPE")
    return value


def _tuple_sizing_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        value = cfg["calibration_C"]["extractor"][
            "mechanistic_training_tuple_sizing_validation_amendment"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_SIZING_VALIDATION_NOT_FROZEN") from error
    if value.get("status") != "FROZEN_BEFORE_SIZING_RERUN_OR_PUBLIC_FIXTURE_OUTCOMES":
        raise RuntimeError("E_TUPLE_SIZING_VALIDATION_NOT_FROZEN")
    copy = json.loads(json.dumps(value))
    expected = copy.pop("validation_commitment_sha256", None)
    if not isinstance(expected, str) or digest(copy) != expected:
        raise RuntimeError("E_TUPLE_SIZING_VALIDATION_COMMITMENT")
    if value.get("grounding_dino_output_rule") != {
        "sizing_caption": "public object.",
        "raw_pred_logits_nan_count_max": 0,
        "raw_pred_logits_positive_infinity_count_max": 0,
        "raw_negative_infinity_role": "official_padding_sentinel_only",
        "raw_negative_infinity_mask_required": "exact_complement_of_pinned_tokenizer_attention_mask_padded_to_model_max_text_len",
        "raw_active_position_nonfinite_count_max": 0,
        "raw_padding_position_non_negative_infinity_count_max": 0,
        "post_sigmoid_score_finite_fraction_required": 1.0,
        "pred_box_finite_fraction_required": 1.0,
        "pred_box_coordinate_range_inclusive": [0.0, 1.0],
    }:
        raise RuntimeError("E_TUPLE_SIZING_VALIDATION_RULE")
    return value


def _validate_grounding_sizing_counts(
    metrics: dict[str, int | float], rule: dict[str, Any]
) -> None:
    lower, upper = rule["pred_box_coordinate_range_inclusive"]
    if (
        metrics["raw_nan_count"] > rule["raw_pred_logits_nan_count_max"]
        or metrics["raw_positive_infinity_count"]
        > rule["raw_pred_logits_positive_infinity_count_max"]
        or metrics["raw_active_position_nonfinite_count"]
        > rule["raw_active_position_nonfinite_count_max"]
        or metrics["raw_padding_position_non_negative_infinity_count"]
        > rule["raw_padding_position_non_negative_infinity_count_max"]
        or metrics["post_sigmoid_nonfinite_count"] != 0
        or metrics["pred_box_nonfinite_count"] != 0
        or metrics["pred_box_min"] < lower
        or metrics["pred_box_max"] > upper
    ):
        raise RuntimeError("E_TUPLE_GROUNDING_NONFINITE")


def _stage_tuple_nltk_resources(
    public: Path, scratch: Path, cfg: dict[str, Any]
) -> Path:
    manifest_path = _tuple_run_root(public) / "dependency_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    commitment = manifest.pop("tuple_dependency_commitment_sha256", None)
    expected = cfg["calibration_C"]["extractor"][
        "mechanistic_training_tuple_premodel_result"
    ]["dependency_manifest_commitment_sha256"]
    if not isinstance(commitment, str) or commitment != expected or digest(manifest) != expected:
        raise RuntimeError("E_TUPLE_NLTK_MANIFEST")
    records = manifest.get("nltk_resource_files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("E_TUPLE_NLTK_MANIFEST")
    source = public / "models/nltk_data"
    top_levels = set()
    for record in records:
        relative = Path(str(record.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RuntimeError("E_TUPLE_NLTK_RESOURCE_PATH")
        path = source / relative
        if not path.is_file() or file_digest(path) != record.get("sha256"):
            raise RuntimeError("E_TUPLE_NLTK_RESOURCE_HASH")
        if len(relative.parts) > 1:
            top_levels.add(relative.parts[0])
    if top_levels != {"averaged_perceptron_tagger_eng", "wordnet"}:
        raise RuntimeError("E_TUPLE_NLTK_RESOURCE_SET")
    target = scratch / "nltk_data"
    (target / "taggers").mkdir(parents=True, exist_ok=False, mode=0o700)
    (target / "corpora").mkdir(parents=True, exist_ok=False, mode=0o700)
    os.symlink(
        source / "averaged_perceptron_tagger_eng",
        target / "taggers/averaged_perceptron_tagger_eng",
        target_is_directory=True,
    )
    os.symlink(
        source / "wordnet",
        target / "corpora/wordnet",
        target_is_directory=True,
    )
    return target


def _tuple_segment_window(
    segment: dict[str, Any], media_duration: float, amendment: dict[str, Any]
) -> dict[str, Any]:
    """Construct the frozen learner tuple window without inventing alignment."""
    if segment.get("status") != "ACCEPT":
        return {"status": "ABSTAIN", "reason": "ADAPTER_ABSTAIN"}
    text = str(segment.get("en", "")).strip()
    try:
        start = float(segment["start"])
        end = float(segment["end"])
        duration = float(media_duration)
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"status": "ABSTAIN", "reason": "INVALID_SEGMENT_BOUNDS"}
    if (
        not text
        or not all(math.isfinite(value) for value in (start, end, duration))
        or start < 0.0
        or end <= start
        or end > duration
    ):
        return {"status": "ABSTAIN", "reason": "INVALID_SEGMENT_BOUNDS"}
    requested = {
        "before": [start - 2.5, start - 1.5, start - 0.5],
        "during": [
            start + (index + 0.5) * (end - start) / 3.0 for index in range(3)
        ],
        "after": [end + 0.5, end + 1.5, end + 2.5],
    }
    samples = {
        position: [round(value, 6) for value in values if 0.0 <= value <= duration]
        for position, values in requested.items()
    }
    in_bounds = sum(len(values) for values in samples.values())
    minimum = float(amendment["tuple_window"]["minimum_in_bounds_sample_fraction"])
    if in_bounds / 9.0 < minimum:
        return {"status": "ABSTAIN", "reason": "INSUFFICIENT_IN_BOUNDS_FRAMES"}
    return {
        "status": "ACCEPT",
        "reason": None,
        "text_en": text,
        "segment_start": round(start, 6),
        "segment_end": round(end, 6),
        "mention_anchor": round((start + end) / 2.0, 6),
        "samples": samples,
    }


def _lexical_mentions(
    text: str,
    tagger,
    lemmatize,
    zipf_frequency,
    frequency_bands: dict[str, list[float]],
) -> list[dict[str, Any]]:
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    tagged = tagger(tokens)
    if len(tagged) != len(tokens):
        raise RuntimeError("E_TUPLE_LEXICAL_SILENT_TRUNCATION")
    output = []
    for index, (token, part_of_speech) in enumerate(tagged):
        if token != tokens[index]:
            raise RuntimeError("E_TUPLE_LEXICAL_ROUNDTRIP")
        if part_of_speech.startswith("NN"):
            kind, wordnet_pos = "noun", "n"
        elif part_of_speech.startswith("JJ"):
            kind, wordnet_pos = "adjective", "a"
        else:
            continue
        lemma = str(lemmatize(token.casefold(), wordnet_pos)).casefold()
        if not lemma or not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", lemma):
            raise RuntimeError("E_TUPLE_LEMMA_INVALID")
        value = float(zipf_frequency(lemma, "en"))
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("E_TUPLE_FREQUENCY_INVALID")
        matching = [
            label
            for label, bounds in frequency_bands.items()
            if float(bounds[0]) <= value < float(bounds[1])
            or (value == float(bounds[1]) == 8.0)
        ]
        if len(matching) != 1:
            raise RuntimeError("E_TUPLE_FREQUENCY_BAND")
        output.append(
            {
                "token_index": index,
                "token": token,
                "lemma": lemma,
                "part_of_speech": kind,
                "zipf_frequency": round(value, 6),
                "frequency_band": matching[0],
            }
        )
    return output


def _map_public_ontology(
    lemma: str, mapping: dict[str, list[str]]
) -> dict[str, Any]:
    matches = sorted(
        category
        for category, words in mapping.items()
        if lemma.casefold() in {str(word).casefold() for word in words}
    )
    if not matches:
        return {"status": "ABSTAIN", "reason": "ONTOLOGY_UNMATCHED"}
    if len(matches) != 1:
        return {"status": "ABSTAIN", "reason": "ONTOLOGY_AMBIGUOUS"}
    return {"status": "ACCEPT", "reason": None, "category": matches[0]}


def _valid_normalized_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        left, top, right, bottom = (float(value) for value in box)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        all(math.isfinite(value) for value in (left, top, right, bottom))
        and 0.0 <= left < right <= 1.0
        and 0.0 <= top < bottom <= 1.0
    )


def _referent_frame_proxy(
    candidates: list[dict[str, Any]],
    target_category: str,
    definitions: dict[str, Any],
    *,
    inference_succeeded: bool,
    negative_specificity_passed: bool,
) -> dict[str, Any]:
    if not inference_succeeded:
        return {"status": "ABSTAIN", "reason": "INFERENCE_FAILURE"}
    clean = []
    for candidate in candidates:
        try:
            area = float(candidate["mask_fraction"])
            center = float(candidate["center_distance"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"status": "ABSTAIN", "reason": "INVALID_GEOMETRY"}
        if (
            not _valid_normalized_box(candidate.get("box"))
            or not all(math.isfinite(value) for value in (area, center))
            or not 0.0 <= area <= 1.0
            or not 0.0 <= center <= math.sqrt(0.5)
        ):
            return {"status": "ABSTAIN", "reason": "INVALID_GEOMETRY"}
        clean.append({**candidate, "mask_fraction": area, "center_distance": center})
    targets = [value for value in clean if value.get("category") == target_category]
    visible_floor = float(definitions["visible_mask_fraction_min"])
    visible_targets = [value for value in targets if value["mask_fraction"] >= visible_floor]
    if not visible_targets:
        if not negative_specificity_passed:
            return {"status": "ABSTAIN", "reason": "NEGATIVE_NOT_QUALIFIED"}
        return {
            "status": "MEASURED_NEGATIVE",
            "reason": None,
            "visible": False,
            "dominant": False,
            "candidate_count_bin": "0",
        }
    target = max(visible_targets, key=lambda value: value["mask_fraction"])
    others = sorted(
        [value["mask_fraction"] for value in clean if value is not target],
        reverse=True,
    )
    ratio = target["mask_fraction"] / max(others[0], 1e-12) if others else math.inf
    candidate_count = len([value for value in clean if value["mask_fraction"] >= visible_floor])
    return {
        "status": "MEASURED_POSITIVE",
        "reason": None,
        "visible": True,
        "dominant": (
            target["mask_fraction"]
            >= float(definitions["dominant_target_mask_fraction_min"])
            and ratio >= float(definitions["dominant_area_ratio_to_next_candidate_min"])
        ),
        "candidate_count_bin": "2plus" if candidate_count >= 2 else "1",
        "target_mask_fraction": round(target["mask_fraction"], 6),
        "target_center_distance": round(target["center_distance"], 6),
        "target_to_next_area_ratio": None if math.isinf(ratio) else round(ratio, 6),
    }


def _validate_monotonic_track(track: list[dict[str, Any]]) -> bool:
    previous = -math.inf
    for row in track:
        try:
            timestamp = float(row["time"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(timestamp) or timestamp <= previous:
            return False
        if not _valid_normalized_box(row.get("box")):
            return False
        previous = timestamp
    return bool(track)


def _recurrence_decision(
    cosine_similarity: float,
    threshold: float,
    *,
    exact_duplicate: bool,
    perceptual_duplicate: bool,
) -> dict[str, Any]:
    values = (float(cosine_similarity), float(threshold))
    if not all(math.isfinite(value) for value in values) or not all(
        -1.0 <= value <= 1.0 for value in values
    ):
        return {"status": "ABSTAIN", "reason": "INVALID_SIMILARITY"}
    return {
        "status": "MEASURED",
        "reason": None,
        "same_referent": values[0] >= values[1],
        "visual_variation": not (exact_duplicate or perceptual_duplicate),
    }


def _tuple_run_root(public: Path) -> Path:
    return public / "runs/synthetic-video-calibration/mechanistic-tuples"


def _tuple_model_root(public: Path) -> Path:
    return public / "models/mechanistic-tuples"


def _tuple_fixture_root(public: Path) -> Path:
    return public / "public/mechanistic-training-tuple-fixtures"


def _fixture_order(seed: int, namespace: str, *parts: Any) -> str:
    return hashlib.sha256(
        "|".join([str(seed), namespace, *(str(part) for part in parts)]).encode()
    ).hexdigest()


def _fixture_partition(seed: int, namespace: str, identity: str) -> str:
    value = int(_fixture_order(seed, namespace, identity), 16)
    return "development" if value % 2 == 0 else "holdout"


def _parse_charades_actions(value: str) -> list[dict[str, Any]]:
    output = []
    for raw in str(value or "").split(";"):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 3 or not re.fullmatch(r"c\d{3}", fields[0]):
            raise RuntimeError("E_TUPLE_ACTION_ANNOTATION")
        try:
            start, end = float(fields[1]), float(fields[2])
        except ValueError as error:
            raise RuntimeError("E_TUPLE_ACTION_ANNOTATION") from error
        if not all(math.isfinite(item) for item in (start, end)) or not 0 <= start < end:
            raise RuntimeError("E_TUPLE_ACTION_ANNOTATION")
        output.append({"code": fields[0], "start": start, "end": end})
    return output


def _charades_direction_map(action: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in action["class_code_pairs"]:
        first_label, second_label = pair["pair"]
        for first_code, second_code in pair["matched_codes"]:
            if first_code in result or second_code in result:
                raise RuntimeError("E_TUPLE_ACTION_CODE_DUPLICATE")
            result[first_code] = first_label
            result[second_code] = second_label
    return result


def _select_charades_action_fixtures(
    rows: list[dict[str, str]],
    action: dict[str, Any],
    seed: int,
    excluded_subjects: set[str],
    excluded_videos: set[str],
) -> dict[str, list[dict[str, Any]]]:
    code_to_label = _charades_direction_map(action)
    opposite = {
        first: second
        for pair in action["class_code_pairs"]
        for first, second in (pair["pair"], tuple(reversed(pair["pair"])))
    }
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        partition: {label: [] for label in action["labels"]}
        for partition in ("development", "holdout")
    }
    for row in rows:
        video = str(row.get("id", ""))
        subject = str(row.get("subject", ""))
        if (
            not video
            or not subject
            or video in excluded_videos
            or subject in excluded_subjects
            or str(row.get("verified", "")).strip().casefold() != "yes"
            or not str(row.get("egocentric", "")).strip()
        ):
            continue
        try:
            duration = float(row["length"])
        except (KeyError, TypeError, ValueError):
            continue
        annotations = _parse_charades_actions(row.get("actions", ""))
        partition = _fixture_partition(
            seed, "mechanistic_action_partition", subject
        )
        for item in annotations:
            label = code_to_label.get(item["code"])
            if label is None:
                continue
            start = max(0.0, item["start"])
            end = min(duration, item["end"])
            interval = end - start
            if not 1.0 <= interval <= 12.0:
                continue
            opposite_label = opposite[label]
            overlaps_opposite = any(
                code_to_label.get(other["code"]) == opposite_label
                and max(start, other["start"]) < min(end, other["end"])
                for other in annotations
            )
            if overlaps_opposite:
                continue
            candidates[partition][label].append(
                {
                    "video": video,
                    "subject": subject,
                    "label": label,
                    "code": item["code"],
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "source_duration": round(duration, 6),
                }
            )
    selected: dict[str, list[dict[str, Any]]] = {}
    for partition in ("development", "holdout"):
        used: set[str] = set()
        selected[partition] = []
        for label in action["labels"]:
            ordered = sorted(
                candidates[partition][label],
                key=lambda row: _fixture_order(
                    seed,
                    "mechanistic_action",
                    partition,
                    label,
                    row["video"],
                    row["start"],
                    row["end"],
                ),
            )
            for row in ordered:
                if row["video"] in used:
                    continue
                used.add(row["video"])
                selected[partition].append(row)
                if sum(
                    item["label"] == label for item in selected[partition]
                ) == 6:
                    break
            if sum(item["label"] == label for item in selected[partition]) != 6:
                raise RuntimeError(
                    f"E_TUPLE_ACTION_FIXTURE_YIELD_{partition}_{label}"
                )
        selected[partition].sort(
            key=lambda row: (
                action["labels"].index(row["label"]),
                _fixture_order(
                    seed,
                    "mechanistic_action_final",
                    partition,
                    row["video"],
                    row["start"],
                ),
            )
        )
    return selected


def _valid_polygon_segmentation(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for polygon in value:
        if (
            not isinstance(polygon, list)
            or len(polygon) < 6
            or len(polygon) % 2
        ):
            return False
        try:
            coordinates = [float(item) for item in polygon]
        except (TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(item) for item in coordinates):
            return False
    return True


def _valid_visor_segments(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for polygon in value:
        if not isinstance(polygon, list) or len(polygon) < 3:
            return False
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2:
                return False
            try:
                coordinates = [float(item) for item in point]
            except (TypeError, ValueError, OverflowError):
                return False
            if not all(math.isfinite(item) and item >= 0 for item in coordinates):
                return False
    return True


def _select_coco_object_sources(
    instances: dict[str, Any],
    preparation: dict[str, Any],
    feasibility_repair: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    seed = int(preparation["seed"])
    ontology = preparation["public_object_ontology"]
    categories = {
        str(row["name"]): int(row["id"])
        for row in instances.get("categories", [])
        if str(row.get("name", "")) in ontology
    }
    if set(categories) != set(ontology):
        raise RuntimeError("E_TUPLE_COCO_CATEGORY_SET")
    images = {int(row["id"]): row for row in instances.get("images", [])}
    licenses = {
        int(row["id"]): row for row in instances.get("licenses", [])
    }
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        partition: {category: [] for category in ontology}
        for partition in preparation["partitions"]
    }
    category_by_id = {value: key for key, value in categories.items()}
    area_minimum, area_maximum = feasibility_repair["source_selection_repair"][
        "active_COCO_area_fraction_range"
    ]
    for annotation in instances.get("annotations", []):
        category = category_by_id.get(int(annotation.get("category_id", -1)))
        image = images.get(int(annotation.get("image_id", -1)))
        if category is None or image is None:
            continue
        try:
            width, height = int(image["width"]), int(image["height"])
            left, top, box_width, box_height = (
                float(item) for item in annotation["bbox"]
            )
            area = float(annotation["area"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (
            int(annotation.get("iscrowd", 1)) != 0
            or width < 480
            or height < 360
            or box_width < 48
            or box_height < 48
            or not float(area_minimum) <= area / (width * height) <= float(area_maximum)
            or not _valid_polygon_segmentation(annotation.get("segmentation"))
            or left < 0
            or top < 0
            or left + box_width > width
            or top + box_height > height
        ):
            continue
        partition = _fixture_partition(
            seed, "coco_object_partition", str(image["id"])
        )
        license_row = licenses.get(int(image.get("license", -1)), {})
        candidates[partition][category].append(
            {
                "image_id": int(image["id"]),
                "annotation_id": int(annotation["id"]),
                "category": category,
                "file_name": str(image["file_name"]),
                "width": width,
                "height": height,
                "bbox": [left, top, box_width, box_height],
                "segmentation": annotation["segmentation"],
                "license_id": int(image.get("license", -1)),
                "license_name": str(license_row.get("name", "")),
                "license_url": str(license_row.get("url", "")),
            }
        )
    selected: dict[str, list[dict[str, Any]]] = {}
    for partition in preparation["partitions"]:
        selected[partition] = []
        used_images: set[int] = set()
        for category in ontology:
            ordered = sorted(
                candidates[partition][category],
                key=lambda row: _fixture_order(
                    seed,
                    "coco_object",
                    partition,
                    category,
                    row["image_id"],
                    row["annotation_id"],
                ),
            )
            for row in ordered:
                if row["image_id"] in used_images:
                    continue
                selected[partition].append(row)
                used_images.add(row["image_id"])
                if sum(
                    item["category"] == category
                    for item in selected[partition]
                ) == 4:
                    break
            if sum(
                item["category"] == category for item in selected[partition]
            ) != 4:
                raise RuntimeError(
                    f"E_TUPLE_COCO_FIXTURE_YIELD_{partition}_{category}"
                )
        selected[partition].sort(
            key=lambda row: (
                ontology.index(row["category"]),
                _fixture_order(
                    seed,
                    "coco_object_final",
                    partition,
                    row["image_id"],
                    row["annotation_id"],
                ),
            )
        )
    return selected


def _visor_frame_truth(row: dict[str, Any]) -> dict[str, Any] | None:
    image = row.get("image")
    annotations = row.get("annotations")
    if not isinstance(image, dict) or not isinstance(annotations, list):
        return None
    hands = [
        item
        for item in annotations
        if str(item.get("name", "")) in {"left hand", "right hand"}
    ]
    all_ids = {str(item.get("id", "")) for item in annotations}
    for item in annotations:
        if not _valid_visor_segments(item.get("segments")):
            return None
    contact_links = [
        str(item.get("in_contact_object", ""))
        for item in hands
        if str(item.get("in_contact_object", ""))
    ]
    if any(link not in all_ids for link in contact_links):
        return None
    if not hands:
        stratum = "true_no_hand"
    elif contact_links:
        stratum = "hand_contact"
    else:
        stratum = "hand_no_contact"
    name = str(image.get("name", ""))
    video = str(image.get("video", ""))
    path = str(image.get("image_path", ""))
    if (
        not name
        or not video
        or Path(name).name != name
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        return None
    return {
        "video": video,
        "frame_name": name,
        "image_path": path,
        "stratum": stratum,
        "hand_visible": bool(hands),
        "contact": bool(contact_links),
        "hand_annotation_count": len(hands),
        "contact_object_count": len(set(contact_links)),
        "annotations": annotations,
    }


def _visor_fixture_availability(
    annotation_documents: list[dict[str, Any]], preparation: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    seed = int(preparation["seed"])
    targets = preparation["visor_selection"]["strata_per_partition"]
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        partition: {stratum: [] for stratum in targets}
        for partition in preparation["partitions"]
    }
    for document in annotation_documents:
        for row in document.get("video_annotations", []):
            truth = _visor_frame_truth(row)
            if truth is None:
                continue
            participant = truth["video"].split("_", 1)[0]
            if not re.fullmatch(r"P\d{2}", participant):
                continue
            partition = _fixture_partition(seed, "visor_partition", participant)
            candidates[partition][truth["stratum"]].append(
                {**truth, "participant": participant}
            )
    output: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for partition in preparation["partitions"]:
        output[partition] = []
        counts[partition] = {}
        video_counts: Counter[str] = Counter()
        for stratum, required in targets.items():
            ordered = sorted(
                candidates[partition][stratum],
                key=lambda row: _fixture_order(
                    seed,
                    "visor",
                    partition,
                    stratum,
                    row["video"],
                    row["frame_name"],
                ),
            )
            selected = 0
            for row in ordered:
                if video_counts[row["video"]] >= 4:
                    continue
                output[partition].append(row)
                video_counts[row["video"]] += 1
                selected += 1
                if selected == int(required):
                    break
            counts[partition][stratum] = selected
        output[partition].sort(
            key=lambda row: (
                list(targets).index(row["stratum"]),
                _fixture_order(
                    seed,
                    "visor_final",
                    partition,
                    row["video"],
                    row["frame_name"],
                ),
            )
        )
    return output, counts


def _select_visor_fixtures(
    annotation_documents: list[dict[str, Any]], preparation: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    output, counts = _visor_fixture_availability(annotation_documents, preparation)
    targets = preparation["visor_selection"]["strata_per_partition"]
    for partition in preparation["partitions"]:
        for stratum, required in targets.items():
            if counts[partition][stratum] != int(required):
                raise RuntimeError(
                    f"E_TUPLE_VISOR_FIXTURE_YIELD_{partition}_{stratum}"
                )
    return output


def _refuse_git_output(path: Path) -> None:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            raise RuntimeError("E_TUPLE_FIXTURE_OUTPUT_IN_GIT")


def prepare_tuple_audio_seed(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    preparation = _tuple_fixture_preparation_amendment(cfg)
    _refuse_git_output(args.output_root)
    if sys.platform != "darwin":
        raise RuntimeError("E_TUPLE_AUDIO_SEED_PLATFORM")
    say = Path("/usr/bin/say")
    if not say.is_file():
        raise RuntimeError("E_TUPLE_AUDIO_SEED_SAY")
    voices = subprocess.run(
        [str(say), "-v", "?"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if not any(line.startswith("Anna ") and "de_DE" in line for line in voices.splitlines()):
        raise RuntimeError("E_TUPLE_AUDIO_SEED_VOICE")
    grammar = {
        "sports ball": ("der", "Ball"),
        "cup": ("der", "Becher"),
        "bottle": ("die", "Flasche"),
        "bowl": ("die", "Schüssel"),
        "book": ("das", "Buch"),
        "chair": ("der", "Stuhl"),
        "apple": ("der", "Apfel"),
        "banana": ("die", "Banane"),
    }
    attributes = ["rot", "blau", "grün", "gelb", "groß", "klein", "rot", "blau"]
    scenarios = preparation["referent_attribute_rendering"][
        "scenarios_once_per_category"
    ]
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    records = []
    for partition in preparation["partitions"]:
        for category in preparation["public_object_ontology"]:
            article, noun = grammar[category]
            for ordinal, (scenario, attribute) in enumerate(
                zip(scenarios, attributes, strict=True)
            ):
                if scenario == "no_speech_visible_object":
                    continue
                prefix = "" if partition == "development" else "Schau, "
                phrase = f"{prefix}{article.capitalize()} {noun} ist {attribute}."
                slug = category.replace(" ", "-")
                target = root / partition / f"{slug}-{ordinal:02d}.aiff"
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                completed = subprocess.run(
                    [str(say), "-v", "Anna", "-r", "175", "-o", str(target), phrase],
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if completed.returncode or not target.is_file() or target.stat().st_size == 0:
                    raise RuntimeError("E_TUPLE_AUDIO_SEED_RENDER")
                os.chmod(target, 0o600)
                records.append(
                    {
                        "partition": partition,
                        "category": category,
                        "scenario": scenario,
                        "ordinal": ordinal,
                        "phrase_de": phrase,
                        "relative_path": str(target.relative_to(root)),
                        "sha256": file_digest(target),
                        "bytes": target.stat().st_size,
                    }
                )
    os_version = subprocess.run(
        ["/usr/bin/sw_vers", "-productVersion"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "SEALED_SELF_AUTHORED_PUBLIC_AUDIO_SEED",
        "preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "license": "CC0-1.0 self-authored text and rendered fixture audio",
        "voice": "macOS Anna de_DE",
        "rate_words_per_minute": 175,
        "platform_version": os_version,
        "say_binary_sha256": file_digest(say),
        "audio_file_count": len(records),
        "records": records,
    }
    manifest["audio_seed_commitment_sha256"] = digest(manifest)
    write_private(root / "audio-seed-manifest.json", manifest)
    return {
        "status": "PASS_AUDIO_SEED_SEALED",
        "audio_file_count": len(records),
        "audio_seed_commitment_sha256": manifest[
            "audio_seed_commitment_sha256"
        ],
    }


def _language_lexical_fixture_rows(
    preparation: dict[str, Any], partition: str
) -> list[dict[str, Any]]:
    nouns = {
        "sports ball": ("Ball", "ball"),
        "cup": ("Becher", "cup"),
        "bottle": ("Flasche", "bottle"),
        "bowl": ("Schüssel", "bowl"),
        "book": ("Buch", "book"),
        "chair": ("Stuhl", "chair"),
        "apple": ("Apfel", "apple"),
        "banana": ("Banane", "banana"),
    }
    development_templates = [
        "Here is the {noun}.",
        "Look at the red {noun}.",
        "The {noun} is blue.",
        "The {noun} appears and the {noun} returns.",
    ]
    holdout_templates = [
        "I can see a {noun}.",
        "A red {noun} is here.",
        "This {noun} looks blue.",
        "The {noun} leaves and later the {noun} comes back.",
    ]
    templates = (
        development_templates if partition == "development" else holdout_templates
    )
    output = []
    for category, (german, english) in nouns.items():
        for variant, template in enumerate(templates):
            text_en = template.format(noun=english)
            expected = [{"token": english, "part_of_speech": "noun"}]
            if variant in (1, 2):
                expected.append(
                    {
                        "token": "red" if variant == 1 else "blue",
                        "part_of_speech": "adjective",
                    }
                )
            if variant == 3:
                expected.append({"token": english, "part_of_speech": "noun"})
            output.append(
                {
                    "case_id": f"{partition}-accept-{category.replace(' ', '-')}-{variant}",
                    "partition": partition,
                    "expected_pipeline_status": "ACCEPT",
                    "expected_reason": None,
                    "prediction": {
                        "text": f"Der {german}",
                        "language": "de",
                        "words": [
                            {
                                "word": "Der",
                                "start": 2.6,
                                "end": 2.9,
                                "probability": 0.9,
                            },
                            {
                                "word": german,
                                "start": 2.9,
                                "end": 3.4,
                                "probability": 0.9,
                            },
                        ],
                    },
                    "audio_duration": 7.0,
                    "translation_status": "ACCEPT",
                    "text_en": text_en,
                    "segment": {
                        "status": "ACCEPT",
                        "start": 2.5,
                        "end": 4.5,
                        "en": text_en,
                    },
                    "expected_lexical_mentions": expected,
                    "expected_public_category": category,
                    "episode_id": f"{partition}-episode-{variant % 2}",
                }
            )
    reasons = [
        "LANGUAGE_MISMATCH",
        "EMPTY_ASR",
        "INVALID_TIMESTAMP",
        "LOW_CONFIDENCE",
        "EMPTY_TRANSLATION",
        "SILENT_TRUNCATION",
        "INSUFFICIENT_IN_BOUNDS_FRAMES",
        "ONTOLOGY_UNMATCHED",
    ]
    for reason in reasons:
        for repeat in range(2):
            prediction = {
                "text": "Der Ball",
                "language": "de",
                "words": [
                    {
                        "word": "Der",
                        "start": 2.6,
                        "end": 2.9,
                        "probability": 0.9,
                    },
                    {
                        "word": "Ball",
                        "start": 2.9,
                        "end": 3.4,
                        "probability": 0.9,
                    },
                ],
            }
            text_en = "The ball."
            segment = {
                "status": "ACCEPT",
                "start": 2.5,
                "end": 4.5,
                "en": text_en,
            }
            if reason == "LANGUAGE_MISMATCH":
                prediction["language"] = "en"
            elif reason == "EMPTY_ASR":
                prediction["text"] = ""
                prediction["words"] = []
            elif reason == "INVALID_TIMESTAMP":
                prediction["words"][1]["start"] = 2.8
            elif reason == "LOW_CONFIDENCE":
                for word in prediction["words"]:
                    word["probability"] = 0.2
            elif reason == "EMPTY_TRANSLATION":
                text_en = ""
                segment["en"] = ""
            elif reason == "SILENT_TRUNCATION":
                segment["status"] = "ABSTAIN"
            elif reason == "INSUFFICIENT_IN_BOUNDS_FRAMES":
                segment["start"] = 0.1
                segment["end"] = 0.5
            elif reason == "ONTOLOGY_UNMATCHED":
                text_en = "The cloud."
                segment["en"] = text_en
            output.append(
                {
                    "case_id": f"{partition}-abstain-{reason.casefold()}-{repeat}",
                    "partition": partition,
                    "expected_pipeline_status": "ABSTAIN",
                    "expected_reason": reason,
                    "prediction": prediction,
                    "audio_duration": 7.0,
                    "translation_status": (
                        "ABSTAIN" if reason in {"EMPTY_TRANSLATION", "SILENT_TRUNCATION"} else "ACCEPT"
                    ),
                    "text_en": text_en,
                    "segment": segment,
                    "expected_lexical_mentions": [],
                    "expected_public_category": None,
                    "episode_id": f"{partition}-abstain-episode-{repeat}",
                }
            )
    if len(output) != 48:
        raise RuntimeError("E_TUPLE_LANGUAGE_FIXTURE_COUNT")
    return output


def _coco_masked_crop(image_path: Path, source: dict[str, Any]):
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    if image.size != (int(source["width"]), int(source["height"])):
        raise RuntimeError("E_TUPLE_COCO_IMAGE_GEOMETRY")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in source["segmentation"]:
        points = [
            (float(polygon[index]), float(polygon[index + 1]))
            for index in range(0, len(polygon), 2)
        ]
        draw.polygon(points, fill=255)
    box = mask.getbbox()
    if box is None:
        raise RuntimeError("E_TUPLE_COCO_EMPTY_MASK")
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    crop = rgba.crop(box)
    if crop.width < 2 or crop.height < 2:
        raise RuntimeError("E_TUPLE_COCO_EMPTY_MASK")
    return crop


def _fixture_background(width: int, height: int, seed: int, identity: str):
    import numpy as np
    from PIL import Image, ImageDraw

    value = int(_fixture_order(seed, "authored_background", identity)[:8], 16)
    base = np.empty((height, width, 3), dtype=np.uint8)
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    colors = np.asarray(
        [
            85 + value % 50,
            95 + (value // 7) % 50,
            105 + (value // 13) % 50,
        ],
        dtype=np.float32,
    )
    for channel in range(3):
        base[:, :, channel] = np.clip(
            colors[channel] + 22 * x + 16 * y, 0, 255
        ).astype(np.uint8)
    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)
    for ordinal in range(9):
        key = int(
            _fixture_order(seed, "background_shape", identity, ordinal)[:8], 16
        )
        left = key % max(1, width - 70)
        top = (key // 11) % max(1, height - 55)
        shape_width = 24 + (key // 17) % 45
        shape_height = 18 + (key // 23) % 36
        color = tuple(35 + (key // (31 + channel * 7)) % 150 for channel in range(3))
        draw.rounded_rectangle(
            (left, top, left + shape_width, top + shape_height),
            radius=5,
            fill=color,
        )
    return image


def _tint_object(crop, attribute: str):
    from PIL import Image
    import numpy as np

    colors = {
        "red": (220, 50, 45),
        "blue": (45, 90, 220),
        "green": (50, 170, 75),
        "yellow": (230, 205, 45),
    }
    if attribute not in colors:
        return crop.copy()
    array = np.asarray(crop.convert("RGBA"), dtype=np.uint8).copy()
    luminance = array[:, :, :3].astype(np.float32).mean(axis=2, keepdims=True) / 255.0
    color = np.asarray(colors[attribute], dtype=np.float32).reshape(1, 1, 3)
    array[:, :, :3] = np.clip(0.30 * array[:, :, :3] + 0.70 * luminance * color, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA")


def _paste_masked_object(canvas, mask, crop, center: tuple[int, int], longest: int):
    from PIL import Image

    scale = float(longest) / max(crop.width, crop.height)
    size = (
        max(2, int(round(crop.width * scale))),
        max(2, int(round(crop.height * scale))),
    )
    resized = crop.resize(size, Image.Resampling.LANCZOS)
    left = int(round(center[0] - resized.width / 2))
    top = int(round(center[1] - resized.height / 2))
    if left < 0 or top < 0 or left + resized.width > canvas.width or top + resized.height > canvas.height:
        raise RuntimeError("E_TUPLE_COMPOSITE_GEOMETRY")
    canvas.alpha_composite(resized, (left, top))
    alpha = resized.getchannel("A")
    mask.paste(alpha, (left, top), alpha)


def _render_referent_fixture(
    preparation: dict[str, Any],
    partition: str,
    category: str,
    scenario: str,
    ordinal: int,
    target_crop,
    distractor_crop,
) -> tuple[list[Any], Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image

    geometry = preparation["referent_attribute_rendering"]["geometry"]
    width, height = int(geometry["width"]), int(geometry["height"])
    frame_count = int(geometry["frames"])
    fps = int(geometry["fps"])
    attributes = ["red", "blue", "green", "yellow", "big", "small", "red", "blue"]
    attribute = attributes[ordinal]
    target = _tint_object(target_crop, attribute)
    distractor = _tint_object(distractor_crop, "green")
    target_size = 72 if attribute == "small" else 130 if attribute == "big" else 104
    target_size = 118 if scenario == "persistent_dominant_with_small_distractor" else target_size
    visibility = {
        "persistent_clear": {"before", "during", "after"},
        "during_only": {"during"},
        "before_only": {"before"},
        "after_only": {"after"},
        "persistent_ambiguous": {"before", "during", "after"},
        "persistent_dominant_with_small_distractor": {"before", "during", "after"},
        "speech_no_referent": set(),
        "no_speech_visible_object": {"before", "during", "after"},
    }[scenario]
    frames = []
    masks = np.zeros((frame_count, height, width), dtype=np.uint8)
    for frame_index in range(frame_count):
        timestamp = frame_index / fps
        phase = "before" if timestamp < 2.5 else "during" if timestamp <= 4.5 else "after"
        canvas = _fixture_background(
            width,
            height,
            int(preparation["seed"]),
            f"{partition}|{category}|{scenario}",
        ).convert("RGBA")
        target_mask = Image.new("L", (width, height), 0)
        if phase in visibility:
            sway = int(round(8 * math.sin(frame_index / 7.0)))
            _paste_masked_object(
                canvas, target_mask, target, (width // 2 + sway, height // 2), target_size
            )
        if scenario in {"persistent_ambiguous", "persistent_dominant_with_small_distractor"}:
            distractor_mask = Image.new("L", (width, height), 0)
            distractor_size = target_size if scenario == "persistent_ambiguous" else 45
            _paste_masked_object(
                canvas,
                distractor_mask,
                distractor,
                (width // 4, height // 2 + 35),
                distractor_size,
            )
        frames.append(np.asarray(canvas.convert("RGB"), dtype=np.uint8))
        masks[frame_index] = (np.asarray(target_mask, dtype=np.uint8) > 0).astype(np.uint8)
    truth = {
        "attribute": attribute,
        "speech_present": scenario != "no_speech_visible_object",
        "visibility": {
            phase: phase in visibility for phase in ("before", "during", "after")
        },
        "dominant": scenario in {
            "persistent_clear",
            "during_only",
            "before_only",
            "after_only",
            "persistent_dominant_with_small_distractor",
            "no_speech_visible_object",
        },
        "candidate_count_bin": (
            "0" if scenario == "speech_no_referent" else "2plus" if scenario == "persistent_ambiguous" else "1"
        ),
    }
    return frames, masks, truth


def _write_fixture_video(
    frames: list[Any],
    fps: int,
    duration: float,
    audio: Path | None,
    target: Path,
) -> None:
    import imageio.v2 as imageio
    import imageio_ffmpeg

    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_video = target.with_suffix(".video.partial.mp4")
    writer = imageio.get_writer(
        temporary_video,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    output = target.with_suffix(target.suffix + ".partial")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(temporary_video),
    ]
    if audio is not None:
        command += ["-i", str(audio)]
        audio_filter = (
            f"anullsrc=r=22050:cl=mono:d={duration}[base];"
            "[1:a]atrim=duration=2.0,adelay=2500:all=1[spoken];"
            "[base][spoken]amix=inputs=2:duration=first[a]"
        )
    else:
        audio_filter = f"anullsrc=r=22050:cl=mono:d={duration}[a]"
    command += [
        "-filter_complex",
        audio_filter,
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-t",
        f"{duration:.6f}",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
    )
    temporary_video.unlink(missing_ok=True)
    if completed.returncode or not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError("E_TUPLE_FIXTURE_VIDEO_ENCODE")
    os.chmod(output, 0o600)
    output.replace(target)


def _render_recurrence_pair(
    first_crop,
    second_crop,
    stratum: str,
    ordinal: int,
) -> tuple[Any, Any]:
    from PIL import Image, ImageEnhance

    def normalized(crop):
        canvas = Image.new("RGBA", (224, 224), (112, 118, 125, 255))
        resized = crop.copy()
        resized.thumbnail((168, 168), Image.Resampling.LANCZOS)
        canvas.alpha_composite(
            resized,
            ((224 - resized.width) // 2, (224 - resized.height) // 2),
        )
        return canvas.convert("RGB")

    first = normalized(first_crop)
    second = normalized(second_crop)
    if stratum == "same_instance_transformed":
        second = ImageEnhance.Brightness(first).enhance(0.75 if ordinal % 2 else 1.25)
        second = second.rotate(4 if ordinal % 2 else -4, resample=Image.Resampling.BILINEAR)
    elif stratum == "same_instance_near_duplicate":
        second = ImageEnhance.Brightness(first).enhance(0.98 if ordinal % 2 else 1.02)
    return first, second


def _sensor_condition_frames(base_frames: list[Any], condition: str) -> list[Any]:
    import numpy as np
    from PIL import Image, ImageFilter

    output = []
    for index, raw in enumerate(base_frames):
        image = Image.fromarray(raw)
        if condition == "static":
            image = Image.fromarray(base_frames[0])
        elif condition in {"low_translation", "high_translation"}:
            amount = 2 if condition == "low_translation" else 14
            array = np.asarray(image)
            array = np.roll(array, shift=(index * amount) % image.width, axis=1)
            image = Image.fromarray(array)
        elif condition in {"mild_blur", "strong_blur"}:
            radius = 1.0 if condition == "mild_blur" else 4.0
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif condition in {"dark", "bright"}:
            scale = 0.45 if condition == "dark" else 1.45
            array = np.asarray(image, dtype=np.float32) * scale
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        elif condition == "hard_cut":
            if index >= len(base_frames) // 2:
                image = Image.fromarray(255 - np.asarray(image, dtype=np.uint8))
        else:
            raise RuntimeError("E_TUPLE_SENSOR_CONDITION")
        output.append(np.asarray(image, dtype=np.uint8))
    return output


def _sensor_truth(frames: list[Any], bins: dict[str, list[float]]) -> dict[str, Any]:
    from PIL import Image

    rows = []
    previous = None
    for raw in frames:
        image = Image.fromarray(raw)
        metrics = _image_metrics(image, previous)
        rows.append(metrics)
        previous = image
    truth = {}
    for name in ("brightness", "blur_edge_strength", "motion_mean_absolute_luma"):
        values = [float(row[name]) for row in rows if row[name] is not None]
        median = statistics.median(values)
        truth[name] = {
            "median": round(median, 8),
            "bin": bucket(median, bins[name]),
        }
    return truth


def _clone_public_repository(
    url: str, commit: str, target: Path
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    repository_url = url.removesuffix(".git")
    archive = target.parent / f"{target.name}-{commit}.zip"
    _download_public_artifact(f"{repository_url}/archive/{commit}.zip", archive)
    archive_record = {
        "archive_sha256": file_digest(archive),
        "archive_bytes": archive.stat().st_size,
    }
    marker = target / ".source-commit"
    if marker.is_file() and marker.read_text().strip() == commit:
        return archive_record
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        roots = {Path(name).parts[0] for name in names if Path(name).parts}
        if len(roots) != 1:
            raise RuntimeError("E_TUPLE_REPOSITORY_ARCHIVE_ROOT")
        for name in names:
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError("E_TUPLE_ARCHIVE_PATH")
        temporary = target.parent / f".{target.name}-extracting"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(mode=0o700)
        source.extractall(temporary)
        extracted = temporary / next(iter(roots))
        if target.exists():
            shutil.rmtree(target)
        extracted.replace(target)
        shutil.rmtree(temporary)
    marker.write_text(commit + "\n")
    os.chmod(marker, 0o600)
    return archive_record


def _download_public_artifact(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.is_file() and target.stat().st_size > 1024 * 1024:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "synthetic-video-research/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        status = int(getattr(response, "status", response.getcode()))
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if partial.stat().st_size <= 1024 * 1024:
        raise RuntimeError("E_TUPLE_ARTIFACT_TOO_SMALL")
    os.chmod(partial, 0o600)
    partial.replace(target)


def _download_exact_public_artifact(
    url: str, target: Path, expected_sha256: str
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.is_file() and file_digest(target) == expected_sha256:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "synthetic-video-research/1"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        with partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    os.chmod(partial, 0o600)
    if file_digest(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("E_TUPLE_EXACT_ARTIFACT_HASH")
    partial.replace(target)


def _download_public_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(5):
        request = urllib.request.Request(
            url, headers={"User-Agent": "synthetic-video-research/1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            break
        except (OSError, urllib.error.URLError):
            partial.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError("E_TUPLE_PUBLIC_FILE_EMPTY")
    os.chmod(partial, 0o600)
    partial.replace(target)


def _apply_grounding_dino_fallback_patch(code_root: Path) -> dict[str, str]:
    deform_path = (
        code_root
        / "groundingdino/models/GroundingDINO/ms_deform_attn.py"
    )
    original = "if torch.cuda.is_available() and value.is_cuda:"
    replacement = (
        "if '_C' in globals() and torch.cuda.is_available() and value.is_cuda:"
    )
    text = deform_path.read_text()
    if original in text:
        if file_digest(deform_path) != GROUNDING_DINO_DEFORM_ATTN_SOURCE_SHA256:
            raise RuntimeError("E_TUPLE_GROUNDING_PATCH_SOURCE")
        if text.count(original) != 1:
            raise RuntimeError("E_TUPLE_GROUNDING_PATCH_COUNT")
        deform_path.write_text(text.replace(original, replacement))
        os.chmod(deform_path, 0o600)
    elif (
        replacement not in text
        or text.count(replacement) != 1
        or file_digest(deform_path)
        != GROUNDING_DINO_DEFORM_ATTN_PATCHED_SHA256
    ):
        raise RuntimeError("E_TUPLE_GROUNDING_PATCH_STATE")

    model_path = (
        code_root
        / "groundingdino/models/GroundingDINO/groundingdino.py"
    )
    visualizer_import = (
        "from groundingdino.util.visualizer import COCOVisualizer\n"
    )
    model_text = model_path.read_text()
    if visualizer_import in model_text:
        if (
            file_digest(model_path) != GROUNDING_DINO_MODEL_SOURCE_SHA256
            or model_text.count(visualizer_import) != 1
            or model_text.count("COCOVisualizer") != 1
        ):
            raise RuntimeError("E_TUPLE_GROUNDING_VISUALIZER_PATCH_SOURCE")
        model_path.write_text(model_text.replace(visualizer_import, ""))
        os.chmod(model_path, 0o600)
    elif (
        "COCOVisualizer" in model_text
        or file_digest(model_path) != GROUNDING_DINO_MODEL_NO_VISUALIZER_SHA256
    ):
        raise RuntimeError("E_TUPLE_GROUNDING_VISUALIZER_PATCH_STATE")
    return {
        "deform_attention_original_sha256": GROUNDING_DINO_DEFORM_ATTN_SOURCE_SHA256,
        "deform_attention_patched_sha256": file_digest(deform_path),
        "model_original_sha256": GROUNDING_DINO_MODEL_SOURCE_SHA256,
        "model_patched_sha256": file_digest(model_path),
        "semantic_scope": "use the official PyTorch deformable-attention fallback only when the official compiled extension is absent and remove one unused visualization-only import; model computation is unchanged",
    }


def _safe_extract_zip(source: Path, target: Path) -> list[str]:
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        for name in names:
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError("E_TUPLE_ARCHIVE_PATH")
        license_names = [
            name
            for name in names
            if Path(name).name.casefold().startswith(("license", "copying", "notice"))
        ]
        marker = target / ".extracted-from-sha256"
        expected = file_digest(source)
        if not marker.is_file() or marker.read_text().strip() != expected:
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            archive.extractall(target)
            marker.write_text(expected + "\n")
            os.chmod(marker, 0o600)
    return sorted(license_names)


def _extract_selected_tar_members(
    source: Path, file_names: set[str], target: Path
) -> dict[str, dict[str, Any]]:
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    matched: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(source, "r") as archive:
        for member in archive:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("E_TUPLE_ARCHIVE_PATH")
            if not member.isfile() or path.name not in file_names:
                continue
            if path.name in matched:
                raise RuntimeError("E_TUPLE_TAR_MEMBER_DUPLICATE")
            matched[path.name] = member
        if set(matched) != file_names:
            raise RuntimeError("E_TUPLE_TAR_MEMBER_MISSING")
        records = {}
        for name in sorted(file_names):
            member = matched[name]
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("E_TUPLE_TAR_MEMBER_MISSING")
            destination = target / name
            partial = destination.with_suffix(destination.suffix + ".partial")
            with partial.open("wb") as output:
                shutil.copyfileobj(handle, output, length=1024 * 1024)
            os.chmod(partial, 0o600)
            partial.replace(destination)
            records[name] = {
                "sha256": file_digest(destination),
                "bytes": destination.stat().st_size,
            }
    return records


def _license_digest(repository: Path, expected: str) -> str:
    candidates = [
        path
        for path in repository.iterdir()
        if path.is_file() and path.name.casefold().startswith("license")
    ]
    matches = [path for path in candidates if file_digest(path) == expected]
    if len(matches) != 1:
        raise RuntimeError("E_TUPLE_CODE_LICENSE")
    return expected


def prepare_tuple_public(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    amendment = _tuple_amendment(cfg)
    prior_stack = cfg["calibration_C"]["extractor"][
        "domain_appropriate_redesign"
    ]["single_stack"]
    public = args.public_root
    model_root = _tuple_model_root(public)
    code_root = model_root / "code"
    weight_root = model_root / "weights"
    repositories = [
        (
            "EgoHOS",
            prior_stack["hand_object_action"]["repository"],
            prior_stack["hand_object_action"]["commit"],
            prior_stack["hand_object_action"]["code_license_sha256"],
        ),
        (
            "GroundingDINO",
            prior_stack["scene_and_referent_detection"]["repository"],
            prior_stack["scene_and_referent_detection"]["commit"],
            prior_stack["scene_and_referent_detection"]["code_license_sha256"],
        ),
        (
            "sam2",
            prior_stack["mask_tracking"]["repository"],
            prior_stack["mask_tracking"]["commit"],
            prior_stack["mask_tracking"]["code_and_weight_license_sha256"],
        ),
        (
            "dinov2",
            prior_stack["diversity_embeddings"]["repository"],
            prior_stack["diversity_embeddings"]["commit"],
            prior_stack["diversity_embeddings"]["code_and_weight_license_sha256"],
        ),
    ]
    repository_records = []
    for name, url, commit, license_sha256 in repositories:
        target = code_root / name
        archive_record = _clone_public_repository(url, commit, target)
        repository_records.append(
            {
                "name": name,
                "url": url,
                "commit": commit,
                **archive_record,
                "license_sha256": _license_digest(target, license_sha256),
                "immutable_commit_archive": True,
            }
        )
    downloads = [
        (
            "groundingdino_swint_ogc.pth",
            prior_stack["scene_and_referent_detection"]["checkpoint_url"],
        ),
        (
            "sam2.1_hiera_base_plus.pt",
            prior_stack["mask_tracking"]["checkpoint_url"],
        ),
        (
            "dinov2_vitb14_pretrain.pth",
            prior_stack["diversity_embeddings"]["checkpoint_url"],
        ),
        (
            "egohos_work_dirs.zip",
            "https://drive.usercontent.google.com/download?id=1DEJBeQ3cR1q7cjjzwDUIQVSoptT-y9U7&export=download&confirm=t",
        ),
    ]
    weight_records = []
    for name, url in downloads:
        path = weight_root / name
        _download_public_artifact(url, path)
        weight_records.append(
            {
                "name": name,
                "source": url,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
        )
    egohos_archive = weight_root / "egohos_work_dirs.zip"
    if not zipfile.is_zipfile(egohos_archive):
        raise RuntimeError("E_TUPLE_EGOHOS_ARCHIVE")
    archive_license_names = _safe_extract_zip(
        egohos_archive, model_root / "egohos-checkpoints"
    )
    pe_cfg = cfg["calibration_C"]["extractor"]["vision_model"]
    pe_path = weight_root / "PE-Core-L14-336.pt"
    _download_public_artifact(
        "https://huggingface.co/"
        f"{pe_cfg['repository']}/resolve/{pe_cfg['revision']}/"
        "PE-Core-L14-336.pt?download=true",
        pe_path,
    )
    if file_digest(pe_path) != pe_cfg["weights_sha256"]:
        raise RuntimeError("E_TUPLE_PE_CORE_WEIGHT")
    egohod_cfg = amendment["fixed_stack"]["temporal_action_control"]
    egohod_hash = re.search(r"SHA-256 ([0-9a-f]{64})", egohod_cfg)
    egohod_file = public / "models/activity-checkpoints/egovideo_large_best.pt"
    if not egohod_hash or not egohod_file.is_file() or file_digest(egohod_file) != egohod_hash.group(1):
        raise RuntimeError("E_TUPLE_EGOHOD_WEIGHT")
    nltk_root = public / "models/nltk_data"
    nltk_archives = model_root / "nltk-archives"
    nltk_records = []
    for name, resource in NLTK_RESOURCE_ARCHIVES.items():
        path = nltk_archives / name
        url = (
            "https://raw.githubusercontent.com/nltk/nltk_data/"
            f"{NLTK_DATA_COMMIT}/{resource['relative_url']}"
        )
        _download_public_artifact(url, path)
        if file_digest(path) != resource["sha256"]:
            raise RuntimeError("E_TUPLE_NLTK_RESOURCE_HASH")
        _safe_extract_zip(path, nltk_root)
        nltk_records.append(
            {
                "name": name,
                "source": url,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
        )
    nltk_files = sorted(path for path in nltk_root.rglob("*") if path.is_file())
    if not nltk_files:
        raise RuntimeError("E_TUPLE_NLTK_RESOURCES")
    wheel_root = model_root / "wheels"
    wheel_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    package_wheels = {}
    for package, version in {"nltk": "3.9.1", "wordfreq": "3.0.2"}.items():
        matches = list(wheel_root.glob(f"{package}-{version}-*.whl"))
        if not matches:
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
                    str(wheel_root),
                    f"{package}=={version}",
                ],
                check=True,
            )
            matches = list(wheel_root.glob(f"{package}-{version}-*.whl"))
        if len(matches) != 1:
            raise RuntimeError("E_TUPLE_PACKAGE_WHEEL")
        package_wheels[package] = matches[0]
    artifact_records = weight_records + nltk_records + [
        {
            "name": "PE-Core-L14-336.pt",
            "source": "facebook/PE-Core-L14-336",
            "sha256": file_digest(pe_path),
            "bytes": pe_path.stat().st_size,
        },
        {
            "name": "egohod_large_best.pt",
            "source": "official cached activity checkpoint",
            "sha256": file_digest(egohod_file),
            "bytes": egohod_file.stat().st_size,
        },
        *[
            {
                "name": wheel.name,
                "source": f"PyPI {package} "
                + ("3.9.1" if package == "nltk" else "3.0.2"),
                "sha256": file_digest(wheel),
                "bytes": wheel.stat().st_size,
            }
            for package, wheel in sorted(package_wheels.items())
        ],
    ]
    manifest = {
        "schema_version": 1,
        "status": "PASS_ARTIFACTS_READY_LOCAL_RELOAD_PENDING_BLIND_SIZING",
        "amendment_commitment_sha256": amendment["amendment_commitment_sha256"],
        "repositories": repository_records,
        "artifacts": artifact_records,
        "egohos_archive_license_files": archive_license_names,
        "weight_terms": {
            "EgoHOS": "official MIT-licensed repository explicitly distributes this checkpoint bundle for local inference; archive-specific license files recorded separately",
            "GroundingDINO": "official Apache-2.0 repository release artifact distributed for local inference",
            "sam2": "official Apache-2.0 code and checkpoints",
            "dinov2": "official Apache-2.0 code and checkpoints",
            "PE-Core": "official Apache-2.0 model card and pinned checkpoint",
            "EgoHOD": "official Apache-2.0 model card and pinned checkpoint",
            "wordfreq": "Apache-2.0 code and CC-BY-SA-4.0 redistributable data",
            "nltk": "Apache-2.0 code; pinned NLTK data packages are redistributed under their package records",
        },
        "nltk_resource_files": [
            {
                "relative_path": str(path.relative_to(nltk_root)),
                "sha256": file_digest(path),
            }
            for path in nltk_files
        ],
        "local_files_only_required": True,
        "telemetry_tracking_disabled": True,
        "restricted_mount_present": False,
        "model_inference_executed": False,
    }
    manifest["tuple_dependency_commitment_sha256"] = digest(manifest)
    output = _tuple_run_root(public)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_private(output / "dependency_manifest.json", manifest)
    return {
        "status": "PASS_ARTIFACTS_READY",
        "component_count": 7,
        "repository_count": len(repository_records),
        "weight_file_count": len(artifact_records),
        "license_record_count": len(manifest["weight_terms"]),
        "artifact_bytes": sum(record["bytes"] for record in artifact_records),
        "archive_license_file_count": len(archive_license_names),
        "restricted_mount_present": False,
        "model_inference_executed": False,
        "tuple_dependency_commitment_sha256": manifest[
            "tuple_dependency_commitment_sha256"
        ],
    }


def _fixture_archive(
    public: Path,
    record: dict[str, Any],
    file_name: str,
) -> Path:
    target = public / "public/source-archives" / file_name
    _download_exact_public_artifact(record["url"], target, record["sha256"])
    if target.stat().st_size != int(record.get("bytes", target.stat().st_size)):
        raise RuntimeError("E_TUPLE_FIXTURE_ARCHIVE_SIZE")
    return target


def _read_audio_seed_manifest(
    root: Path, preparation: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str, str], Path]]:
    path = root / "audio-seed-manifest.json"
    manifest = json.loads(path.read_text())
    commitment = manifest.pop("audio_seed_commitment_sha256", None)
    if (
        not isinstance(commitment, str)
        or digest(manifest) != commitment
        or manifest.get("preparation_amendment_commitment_sha256")
        != preparation["preparation_amendment_commitment_sha256"]
        or manifest.get("status") != "SEALED_SELF_AUTHORED_PUBLIC_AUDIO_SEED"
    ):
        raise RuntimeError("E_TUPLE_AUDIO_SEED_MANIFEST")
    manifest["audio_seed_commitment_sha256"] = commitment
    records = {}
    for row in manifest.get("records", []):
        relative = Path(str(row.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("E_TUPLE_AUDIO_SEED_PATH")
        source = root / relative
        if (
            not source.is_file()
            or file_digest(source) != row.get("sha256")
            or source.stat().st_size != int(row.get("bytes", -1))
        ):
            raise RuntimeError("E_TUPLE_AUDIO_SEED_HASH")
        key = (row["partition"], row["category"], row["scenario"])
        if key in records:
            raise RuntimeError("E_TUPLE_AUDIO_SEED_DUPLICATE")
        records[key] = source
    if len(records) != 112:
        raise RuntimeError("E_TUPLE_AUDIO_SEED_COUNT")
    return manifest, records


def _load_charades_rows(annotation_root: Path) -> list[dict[str, str]]:
    output = []
    for name in ("CharadesEgo_v1_train.csv", "CharadesEgo_v1_test.csv"):
        candidates = list(annotation_root.rglob(name))
        if len(candidates) != 1:
            raise RuntimeError("E_TUPLE_ACTION_ANNOTATION_FILE")
        with candidates[0].open(newline="", encoding="utf-8") as handle:
            output.extend(dict(row) for row in csv.DictReader(handle))
    return output


def _load_visor_annotation_documents(
    fixture_root: Path,
    preparation: dict[str, Any],
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], list[dict[str, Any]]]:
    base = preparation["source_archives"]["EPIC_KITCHENS_VISOR_validation"]
    source_root = fixture_root / "sources/VISOR"
    annotation_root = source_root / "annotations"
    index_url = f"{base['repository_root'].rstrip('/')}/{base['annotation_index']}"
    index_path = source_root / "validation-index.html"
    if not index_path.is_file():
        _download_public_file(index_url, index_path)
    names = sorted(
        set(
            re.findall(
                r"href=[\"']([^\"']+\.json)[\"']",
                index_path.read_text(errors="strict"),
            )
        )
    )
    if len(names) != 43 or any(Path(name).name != name for name in names):
        raise RuntimeError("E_TUPLE_VISOR_ANNOTATION_INDEX")
    documents = []
    provenance = [
        {
            "relative_path": str(index_path.relative_to(fixture_root)),
            "sha256": file_digest(index_path),
            "bytes": index_path.stat().st_size,
            "license": base["license"],
        }
    ]
    for name in names:
        target = annotation_root / name
        if not target.is_file():
            _download_public_file(index_url + name, target)
        document = json.loads(target.read_text())
        if not isinstance(document.get("video_annotations"), list):
            raise RuntimeError("E_TUPLE_VISOR_ANNOTATION_SCHEMA")
        documents.append(document)
        provenance.append(
            {
                "relative_path": str(target.relative_to(fixture_root)),
                "sha256": file_digest(target),
                "bytes": target.stat().st_size,
                "license": base["license"],
            }
        )
    return base, source_root, documents, provenance


def _prepare_visor_fixtures(
    fixture_root: Path,
    preparation: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    base, source_root, documents, provenance = _load_visor_annotation_documents(
        fixture_root, preparation
    )
    selected = _select_visor_fixtures(documents, preparation)
    output: dict[str, list[dict[str, Any]]] = {
        partition: [] for partition in preparation["partitions"]
    }
    zip_records: dict[str, dict[str, Any]] = {}
    for partition in preparation["partitions"]:
        for ordinal, row in enumerate(selected[partition]):
            video = row["video"]
            participant = row["participant"]
            archive_url = (
                f"{base['repository_root'].rstrip('/')}/"
                + base["frame_archive_template"].format(
                    participant=participant, video=video
                )
            )
            archive_path = source_root / "frame-archives" / participant / f"{video}.zip"
            if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
                _download_public_file(archive_url, archive_path)
            archive_key = str(archive_path.relative_to(fixture_root))
            if archive_key not in zip_records:
                zip_records[archive_key] = {
                    "relative_path": archive_key,
                    "sha256": file_digest(archive_path),
                    "bytes": archive_path.stat().st_size,
                    "license": base["license"],
                }
            with zipfile.ZipFile(archive_path) as archive:
                matches = [
                    name
                    for name in archive.namelist()
                    if Path(name).name == row["frame_name"]
                ]
                if len(matches) != 1:
                    raise RuntimeError("E_TUPLE_VISOR_FRAME_MEMBER")
                member = Path(matches[0])
                if member.is_absolute() or ".." in member.parts:
                    raise RuntimeError("E_TUPLE_ARCHIVE_PATH")
                target = (
                    fixture_root
                    / "media/hand-contact"
                    / partition
                    / f"{ordinal:03d}.jpg"
                )
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                partial = target.with_suffix(".jpg.partial")
                with archive.open(matches[0]) as source, partial.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
                os.chmod(partial, 0o600)
                partial.replace(target)
            from PIL import Image

            with Image.open(target) as image:
                width, height = image.size
                image.verify()
            for annotation in row["annotations"]:
                for polygon in annotation["segments"]:
                    if any(
                        not (0.0 <= float(x) <= width and 0.0 <= float(y) <= height)
                        for x, y in polygon
                    ):
                        raise RuntimeError("E_TUPLE_VISOR_GEOMETRY")
            output[partition].append(
                {
                    "fixture_ordinal": ordinal,
                    "stratum": row["stratum"],
                    "hand_visible": row["hand_visible"],
                    "contact": row["contact"],
                    "source_participant": participant,
                    "source_video": video,
                    "source_frame_name": row["frame_name"],
                    "media_relative_path": str(target.relative_to(fixture_root)),
                    "media_sha256": file_digest(target),
                    "media_bytes": target.stat().st_size,
                    "geometry_valid": True,
                }
            )
    provenance.extend(zip_records[key] for key in sorted(zip_records))
    return output, provenance


def prepare_tuple_fixture_feasibility(
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    protocol = _tuple_fixture_protocol(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    repair = _tuple_fixture_feasibility_repair(cfg)
    fixture_root = _tuple_fixture_root(args.public_root)
    fixture_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    sources = preparation["source_archives"]
    coco_annotations = _fixture_archive(
        args.public_root,
        sources["COCO_2017_instances"],
        "annotations_trainval2017.zip",
    )
    charades_annotations = _fixture_archive(
        args.public_root,
        sources["Charades_Ego_annotations"],
        "CharadesEgo.zip",
    )
    extracted = fixture_root / "sources/extracted"
    _safe_extract_zip(coco_annotations, extracted / "coco-annotations")
    _safe_extract_zip(charades_annotations, extracted / "charades-annotations")
    instances = json.loads(
        (
            extracted / "coco-annotations/annotations/instances_val2017.json"
        ).read_text()
    )
    coco = _select_coco_object_sources(instances, preparation, repair)
    _, _, visor_documents, visor_provenance = _load_visor_annotation_documents(
        fixture_root, preparation
    )
    visor, visor_counts = _visor_fixture_availability(
        visor_documents, preparation
    )
    visor_targets = preparation["visor_selection"]["strata_per_partition"]
    visor_deficits = [
        (partition, stratum, int(required), visor_counts[partition][stratum])
        for partition in preparation["partitions"]
        for stratum, required in visor_targets.items()
        if visor_counts[partition][stratum] != int(required)
    ]
    if visor_deficits:
        partition, stratum, required, available = visor_deficits[0]
        visor_subject_sets = {
            name: {row["participant"] for row in visor[name]}
            for name in preparation["partitions"]
        }
        visor_video_sets = {
            name: {row["video"] for row in visor[name]}
            for name in preparation["partitions"]
        }
        object_sets = {
            name: {row["image_id"] for row in coco[name]}
            for name in preparation["partitions"]
        }
        audits = {
            "source_subject_overlap_count": len(
                visor_subject_sets["development"] & visor_subject_sets["holdout"]
            ),
            "source_video_overlap_count": len(
                visor_video_sets["development"] & visor_video_sets["holdout"]
            ),
            "source_object_overlap_count": len(
                object_sets["development"] & object_sets["holdout"]
            ),
        }
        record = {
            "schema_version": 1,
            "status": "NO_GO_ANNOTATION_ONLY_FIXTURE_SOURCE_YIELD",
            "fixture_preparation_amendment_commitment_sha256": preparation[
                "preparation_amendment_commitment_sha256"
            ],
            "fixture_feasibility_repair_commitment_sha256": repair[
                "fixture_feasibility_repair_commitment_sha256"
            ],
            "public_fixture_protocol_commitment_sha256": protocol[
                "protocol_commitment_sha256"
            ],
            "blocking_family": "VISOR_hand_contact",
            "blocking_partition": partition,
            "blocking_stratum": stratum,
            "required_count": required,
            "available_count": available,
            "deficit_count": required - available,
            "coco_source_counts": {
                name: len(coco[name]) for name in preparation["partitions"]
            },
            "visor_stratum_counts": visor_counts,
            "audits": audits,
            "selections": {"coco": coco, "visor": visor},
            "visor_annotation_provenance": visor_provenance,
            "action_selection_status": "NOT_RUN_AFTER_BLOCKING_VISOR_SOURCE_NO_GO",
            "model_inference_executed": False,
            "media_rendering_executed": False,
            "large_Charades_video_archive_downloaded": False,
            "restricted_mount_present": False,
        }
        record["fixture_feasibility_commitment_sha256"] = digest(record)
        write_private(fixture_root / "fixture-feasibility.json", record)
        return {
            "status": "NO_GO_ANNOTATION_ONLY_FIXTURE_SOURCE_YIELD",
            "coco_source_count": sum(
                len(coco[name]) for name in preparation["partitions"]
            ),
            "visor_item_count": sum(
                sum(values.values()) for values in visor_counts.values()
            ),
            "action_item_count": 0,
            "partition_count": len(preparation["partitions"]),
            "failing_family_count": 1,
            "blocking_family": "VISOR_hand_contact",
            "blocking_partition": partition,
            "blocking_stratum": stratum,
            "required_count": required,
            "available_count": available,
            **audits,
            "model_inference_executed": False,
            "media_rendering_executed": False,
            "restricted_mount_present": False,
            "fixture_feasibility_commitment_sha256": record[
                "fixture_feasibility_commitment_sha256"
            ],
        }
    old_manifest = json.loads(
        (
            args.public_root
            / "public/manifests/charades-selection-manifest.json"
        ).read_text()
    )
    excluded_subjects = {
        row["subject"]
        for rows in old_manifest["partitions"].values()
        for row in rows
    }
    excluded_videos = {
        row["id"]
        for rows in old_manifest["partitions"].values()
        for row in rows
    }
    action = _select_charades_action_fixtures(
        _load_charades_rows(extracted / "charades-annotations"),
        protocol["order_dependent_action_control"],
        int(preparation["seed"]),
        excluded_subjects,
        excluded_videos,
    )
    subject_sets = {
        partition: {
            *(row["participant"] for row in visor[partition]),
            *(row["subject"] for row in action[partition]),
        }
        for partition in preparation["partitions"]
    }
    video_sets = {
        partition: {
            *(row["video"] for row in visor[partition]),
            *(row["video"] for row in action[partition]),
        }
        for partition in preparation["partitions"]
    }
    object_sets = {
        partition: {row["image_id"] for row in coco[partition]}
        for partition in preparation["partitions"]
    }
    audits = {
        "source_subject_overlap_count": len(
            subject_sets["development"] & subject_sets["holdout"]
        ),
        "source_video_overlap_count": len(
            video_sets["development"] & video_sets["holdout"]
        ),
        "source_object_overlap_count": len(
            object_sets["development"] & object_sets["holdout"]
        ),
    }
    counts = {
        partition: {
            "coco_source": len(coco[partition]),
            "visor": len(visor[partition]),
            "action": len(action[partition]),
        }
        for partition in preparation["partitions"]
    }
    expected = {"coco_source": 32, "visor": 40, "action": 48}
    failing = sum(
        counts[partition][family] != count
        for partition in preparation["partitions"]
        for family, count in expected.items()
    ) + sum(value != 0 for value in audits.values())
    if failing:
        raise RuntimeError("E_TUPLE_FIXTURE_FEASIBILITY_GATE")
    record = {
        "schema_version": 1,
        "status": "PASS_ANNOTATION_ONLY_FIXTURE_FEASIBILITY",
        "fixture_preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "fixture_feasibility_repair_commitment_sha256": repair[
            "fixture_feasibility_repair_commitment_sha256"
        ],
        "public_fixture_protocol_commitment_sha256": protocol[
            "protocol_commitment_sha256"
        ],
        "counts": counts,
        "audits": audits,
        "selections": {"coco": coco, "visor": visor, "action": action},
        "visor_annotation_provenance": visor_provenance,
        "model_inference_executed": False,
        "media_rendering_executed": False,
        "large_Charades_video_archive_downloaded": False,
        "restricted_mount_present": False,
    }
    record["fixture_feasibility_commitment_sha256"] = digest(record)
    write_private(fixture_root / "fixture-feasibility.json", record)
    return {
        "status": "PASS_ANNOTATION_ONLY_FIXTURE_FEASIBILITY",
        "coco_source_count": sum(row["coco_source"] for row in counts.values()),
        "visor_item_count": sum(row["visor"] for row in counts.values()),
        "action_item_count": sum(row["action"] for row in counts.values()),
        "partition_count": len(counts),
        "failing_family_count": 0,
        **audits,
        "model_inference_executed": False,
        "media_rendering_executed": False,
        "restricted_mount_present": False,
        "fixture_feasibility_commitment_sha256": record[
            "fixture_feasibility_commitment_sha256"
        ],
    }


def _prepare_coco_fixtures(
    fixture_root: Path,
    extracted: Path,
    preparation: dict[str, Any],
    feasibility_repair: dict[str, Any],
    cfg: dict[str, Any],
    audio_files: dict[tuple[str, str, str], Path],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, set[int]]]:
    import numpy as np
    from PIL import Image

    instances_path = extracted / "coco-annotations/annotations/instances_val2017.json"
    instances = json.loads(instances_path.read_text())
    coco_selected = _select_coco_object_sources(
        instances, preparation, feasibility_repair
    )
    crop_records: dict[str, dict[str, list[dict[str, Any]]]] = {
        partition: {category: [] for category in preparation["public_object_ontology"]}
        for partition in preparation["partitions"]
    }
    for partition in preparation["partitions"]:
        for source in coco_selected[partition]:
            image_path = extracted / "coco-images/val2017" / source["file_name"]
            if not image_path.is_file():
                raise RuntimeError("E_TUPLE_COCO_IMAGE_MISSING")
            crop = _coco_masked_crop(image_path, source)
            ordinal = sum(len(values) for values in crop_records[partition].values())
            target = (
                fixture_root
                / "sources/coco-crops"
                / partition
                / f"{ordinal:03d}.png"
            )
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            crop.save(target, format="PNG")
            os.chmod(target, 0o600)
            crop_records[partition][source["category"]].append(
                {
                    **source,
                    "source_image_sha256": file_digest(image_path),
                    "source_image_bytes": image_path.stat().st_size,
                    "crop_relative_path": str(target.relative_to(fixture_root)),
                    "crop_sha256": file_digest(target),
                    "crop_bytes": target.stat().st_size,
                }
            )
    geometry = preparation["referent_attribute_rendering"]["geometry"]
    scenarios = preparation["referent_attribute_rendering"][
        "scenarios_once_per_category"
    ]
    outputs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for partition in preparation["partitions"]:
        outputs[partition] = {
            "language_lexical": _language_lexical_fixture_rows(
                preparation, partition
            ),
            "referent_attribute": [],
            "recurrence": [],
            "sensor": [],
        }
        ontology = preparation["public_object_ontology"]
        base_frames: list[list[Any]] = []
        for category_index, category in enumerate(ontology):
            targets = crop_records[partition][category]
            distractor_category = ontology[(category_index + 1) % len(ontology)]
            distractors = crop_records[partition][distractor_category]
            for scenario_ordinal, scenario in enumerate(scenarios):
                target_record = targets[scenario_ordinal % len(targets)]
                distractor_record = distractors[scenario_ordinal % len(distractors)]
                target_crop = Image.open(
                    fixture_root / target_record["crop_relative_path"]
                ).convert("RGBA")
                distractor_crop = Image.open(
                    fixture_root / distractor_record["crop_relative_path"]
                ).convert("RGBA")
                frames, masks, truth = _render_referent_fixture(
                    preparation,
                    partition,
                    category,
                    scenario,
                    scenario_ordinal,
                    target_crop,
                    distractor_crop,
                )
                fixture_ordinal = len(outputs[partition]["referent_attribute"])
                media = (
                    fixture_root
                    / "media/referent-attribute"
                    / partition
                    / f"{fixture_ordinal:03d}.mp4"
                )
                mask_path = (
                    fixture_root
                    / "truth/referent-attribute"
                    / partition
                    / f"{fixture_ordinal:03d}.npz"
                )
                mask_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                np.savez_compressed(mask_path, target_mask=masks)
                os.chmod(mask_path, 0o600)
                audio = audio_files.get((partition, category, scenario))
                if truth["speech_present"] and audio is None:
                    raise RuntimeError("E_TUPLE_AUDIO_SEED_MISSING")
                _write_fixture_video(
                    frames,
                    int(geometry["fps"]),
                    float(geometry["duration_seconds"]),
                    audio,
                    media,
                )
                outputs[partition]["referent_attribute"].append(
                    {
                        "fixture_ordinal": fixture_ordinal,
                        "category": category,
                        "scenario": scenario,
                        "source_image_id": target_record["image_id"],
                        "source_annotation_id": target_record["annotation_id"],
                        "source_image_sha256": target_record["source_image_sha256"],
                        "source_license_id": target_record["license_id"],
                        "source_license_name": target_record["license_name"],
                        "source_license_url": target_record["license_url"],
                        "media_relative_path": str(media.relative_to(fixture_root)),
                        "media_sha256": file_digest(media),
                        "media_bytes": media.stat().st_size,
                        "mask_relative_path": str(mask_path.relative_to(fixture_root)),
                        "mask_sha256": file_digest(mask_path),
                        "truth": truth,
                        "utterance_start": 2.5,
                        "utterance_end": 4.5,
                    }
                )
                if scenario == "persistent_clear" and len(base_frames) < 6:
                    base_frames.append(frames)
        recurrence_strata = preparation["recurrence_recipe"]["strata_per_partition"]
        for stratum, count in recurrence_strata.items():
            for ordinal in range(int(count)):
                category_index = ordinal % len(ontology)
                category = ontology[category_index]
                first_record = crop_records[partition][category][
                    (ordinal // len(ontology)) % 4
                ]
                if stratum in {
                    "same_instance_transformed",
                    "same_instance_near_duplicate",
                }:
                    second_record = first_record
                elif stratum == "same_category_different_instance":
                    second_record = crop_records[partition][category][
                        ((ordinal // len(ontology)) + 1) % 4
                    ]
                else:
                    second_record = crop_records[partition][
                        ontology[(category_index + 1) % len(ontology)]
                    ][(ordinal // len(ontology)) % 4]
                first_crop = Image.open(
                    fixture_root / first_record["crop_relative_path"]
                ).convert("RGBA")
                second_crop = Image.open(
                    fixture_root / second_record["crop_relative_path"]
                ).convert("RGBA")
                first, second = _render_recurrence_pair(
                    first_crop, second_crop, stratum, ordinal
                )
                pair_ordinal = len(outputs[partition]["recurrence"])
                pair_root = fixture_root / "media/recurrence" / partition
                pair_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                first_path = pair_root / f"{pair_ordinal:03d}-a.png"
                second_path = pair_root / f"{pair_ordinal:03d}-b.png"
                first.save(first_path, format="PNG")
                second.save(second_path, format="PNG")
                os.chmod(first_path, 0o600)
                os.chmod(second_path, 0o600)
                outputs[partition]["recurrence"].append(
                    {
                        "fixture_ordinal": pair_ordinal,
                        "stratum": stratum,
                        "same_referent": preparation["recurrence_recipe"][
                            "same_referent_truth"
                        ][stratum],
                        "near_duplicate": preparation["recurrence_recipe"][
                            "near_duplicate_truth"
                        ][stratum],
                        "source_image_ids": [
                            first_record["image_id"],
                            second_record["image_id"],
                        ],
                        "first_relative_path": str(first_path.relative_to(fixture_root)),
                        "first_sha256": file_digest(first_path),
                        "second_relative_path": str(second_path.relative_to(fixture_root)),
                        "second_sha256": file_digest(second_path),
                    }
                )
        bins = cfg["calibration_C"]["extractor"]["fixed_numeric_bins"]
        conditions = preparation["sensor_recipe"]["conditions"]
        if len(base_frames) != 6:
            raise RuntimeError("E_TUPLE_SENSOR_BASE_COUNT")
        for base_ordinal, frames in enumerate(base_frames):
            for condition in conditions:
                rendered = _sensor_condition_frames(frames, condition)
                fixture_ordinal = len(outputs[partition]["sensor"])
                media = (
                    fixture_root
                    / "media/sensor"
                    / partition
                    / f"{fixture_ordinal:03d}.mp4"
                )
                _write_fixture_video(
                    rendered,
                    int(geometry["fps"]),
                    float(geometry["duration_seconds"]),
                    None,
                    media,
                )
                outputs[partition]["sensor"].append(
                    {
                        "fixture_ordinal": fixture_ordinal,
                        "base_ordinal": base_ordinal,
                        "condition": condition,
                        "media_relative_path": str(media.relative_to(fixture_root)),
                        "media_sha256": file_digest(media),
                        "media_bytes": media.stat().st_size,
                        "truth": _sensor_truth(rendered, bins),
                    }
                )
    object_sets = {
        partition: {
            row["image_id"]
            for category in crop_records[partition].values()
            for row in category
        }
        for partition in preparation["partitions"]
    }
    return outputs, object_sets


def _prepare_action_fixtures(
    args: argparse.Namespace,
    fixture_root: Path,
    extracted: Path,
    preparation: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    old_manifest_path = (
        args.public_root / "public/manifests/charades-selection-manifest.json"
    )
    old_manifest = json.loads(old_manifest_path.read_text())
    excluded_subjects = {
        row["subject"]
        for rows in old_manifest["partitions"].values()
        for row in rows
    }
    excluded_videos = {
        row["id"]
        for rows in old_manifest["partitions"].values()
        for row in rows
    }
    selected = _select_charades_action_fixtures(
        _load_charades_rows(extracted / "charades-annotations"),
        protocol["order_dependent_action_control"],
        int(preparation["seed"]),
        excluded_subjects,
        excluded_videos,
    )
    record = preparation["source_archives"]["Charades_Ego_480p_video"]
    archive = args.public_root / "public/source-archives/CharadesEgo_v1_480.tar"
    _download_public_artifact(record["url"], archive)
    if file_digest(archive) != record["sha256"]:
        raise RuntimeError("E_TUPLE_ACTION_ARCHIVE_HASH")
    selected_names = {
        f"{row['video']}.mp4" for rows in selected.values() for row in rows
    }
    media_root = fixture_root / "media/order-action/source"
    media_records = _extract_selected_tar_members(
        archive, selected_names, media_root
    )
    output = {partition: [] for partition in preparation["partitions"]}
    for partition in preparation["partitions"]:
        for ordinal, row in enumerate(selected[partition]):
            path = media_root / f"{row['video']}.mp4"
            output[partition].append(
                {
                    **row,
                    "fixture_ordinal": ordinal,
                    "media_relative_path": str(path.relative_to(fixture_root)),
                    "media_sha256": media_records[path.name]["sha256"],
                    "media_bytes": media_records[path.name]["bytes"],
                }
            )
    sets = {
        partition: {row["video"] for row in output[partition]}
        for partition in preparation["partitions"]
    }
    archive.unlink()
    return output, sets


def prepare_tuple_fixtures(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    amendment = _tuple_amendment(cfg)
    protocol = _tuple_fixture_protocol(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    feasibility_repair = _tuple_fixture_feasibility_repair(cfg)
    _verify_tuple_runtime_manifest(args.public_root, cfg)
    fixture_root = _tuple_fixture_root(args.public_root)
    fixture_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    audio_manifest, audio_files = _read_audio_seed_manifest(
        args.audio_seed_root, preparation
    )
    sources = preparation["source_archives"]
    coco_annotations = _fixture_archive(
        args.public_root,
        sources["COCO_2017_instances"],
        "annotations_trainval2017.zip",
    )
    coco_images = _fixture_archive(
        args.public_root,
        sources["COCO_2017_validation_images"],
        "val2017.zip",
    )
    charades_annotations = _fixture_archive(
        args.public_root,
        sources["Charades_Ego_annotations"],
        "CharadesEgo.zip",
    )
    extracted = fixture_root / "sources/extracted"
    _safe_extract_zip(coco_annotations, extracted / "coco-annotations")
    _safe_extract_zip(coco_images, extracted / "coco-images")
    _safe_extract_zip(charades_annotations, extracted / "charades-annotations")
    partitions, object_sets = _prepare_coco_fixtures(
        fixture_root,
        extracted,
        preparation,
        feasibility_repair,
        cfg,
        audio_files,
    )
    visor, visor_provenance = _prepare_visor_fixtures(
        fixture_root, preparation
    )
    action, action_video_sets = _prepare_action_fixtures(
        args, fixture_root, extracted, preparation, protocol
    )
    for partition in preparation["partitions"]:
        partitions[partition]["hand_contact"] = visor[partition]
        partitions[partition]["order_action"] = action[partition]
    subject_sets = {
        partition: {
            *(row["source_participant"] for row in visor[partition]),
            *(row["subject"] for row in action[partition]),
        }
        for partition in preparation["partitions"]
    }
    visor_video_sets = {
        partition: {row["source_video"] for row in visor[partition]}
        for partition in preparation["partitions"]
    }
    video_sets = {
        partition: visor_video_sets[partition] | action_video_sets[partition]
        for partition in preparation["partitions"]
    }
    audits = {
        "source_subject_overlap_count": len(
            subject_sets["development"] & subject_sets["holdout"]
        ),
        "source_video_overlap_count": len(
            video_sets["development"] & video_sets["holdout"]
        ),
        "source_object_overlap_count": len(
            object_sets["development"] & object_sets["holdout"]
        ),
        "fixture_counts": {
            partition: {
                family: len(rows) for family, rows in partitions[partition].items()
            }
            for partition in preparation["partitions"]
        },
        "all_source_media_hash_roundtrips": True,
        "all_labels_roundtrip": True,
        "all_geometry_finite_and_in_bounds": True,
    }
    if any(
        audits[key] != 0
        for key in (
            "source_subject_overlap_count",
            "source_video_overlap_count",
            "source_object_overlap_count",
        )
    ):
        raise RuntimeError("E_TUPLE_FIXTURE_PARTITION_OVERLAP")
    expected_counts = preparation["counts_per_partition"]
    if any(
        audits["fixture_counts"][partition] != expected_counts
        for partition in preparation["partitions"]
    ):
        raise RuntimeError("E_TUPLE_FIXTURE_COUNT")
    source_provenance = [
        {
            "source": key,
            "sha256": value["sha256"],
            "bytes": value.get("bytes"),
            "license": value.get("license", value.get("annotation_license")),
        }
        for key, value in sources.items()
        if "sha256" in value
    ] + visor_provenance
    fixture_manifest = {
        "schema_version": 1,
        "status": "SEALED_BEFORE_PUBLIC_DEVELOPMENT_INFERENCE",
        "mechanistic_tuple_amendment_commitment_sha256": amendment[
            "amendment_commitment_sha256"
        ],
        "public_fixture_protocol_commitment_sha256": protocol[
            "protocol_commitment_sha256"
        ],
        "fixture_preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "audio_seed_commitment_sha256": audio_manifest[
            "audio_seed_commitment_sha256"
        ],
        "source_provenance": source_provenance,
        "partitions": partitions,
        "audits": audits,
        "model_inference_executed": False,
        "restricted_mount_present": False,
        "development_outcome_opened": False,
        "holdout_outcome_opened": False,
    }
    fixture_manifest["public_fixture_manifest_commitment_sha256"] = digest(
        fixture_manifest
    )
    write_private(fixture_root / "fixture-manifest.json", fixture_manifest)
    total = {
        family: sum(
            len(partitions[partition][family])
            for partition in preparation["partitions"]
        )
        for family in preparation["counts_per_partition"]
    }
    return {
        "status": "PASS_PUBLIC_FIXTURES_SEALED_NO_MODEL_INFERENCE",
        "source_archive_count": len(source_provenance),
        "partition_count": len(preparation["partitions"]),
        "language_lexical_item_count": total["language_lexical"],
        "referent_attribute_item_count": total["referent_attribute"],
        "recurrence_pair_count": total["recurrence"],
        "hand_contact_item_count": total["hand_contact"],
        "sensor_item_count": total["sensor"],
        "order_action_item_count": total["order_action"],
        "source_subject_overlap_count": audits["source_subject_overlap_count"],
        "source_video_overlap_count": audits["source_video_overlap_count"],
        "source_object_overlap_count": audits["source_object_overlap_count"],
        "restricted_mount_present": False,
        "model_inference_executed": False,
        "public_fixture_manifest_commitment_sha256": fixture_manifest[
            "public_fixture_manifest_commitment_sha256"
        ],
    }


def prepare_tuple_runtime(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    amendment = _tuple_amendment(cfg)
    runtime = _tuple_runtime_amendment(cfg)
    model_root = _tuple_model_root(args.public_root)
    base_manifest_path = _tuple_run_root(args.public_root) / "dependency_manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    expected_base = cfg["calibration_C"]["extractor"][
        "mechanistic_training_tuple_premodel_result"
    ]["dependency_manifest_commitment_sha256"]
    if (
        base_manifest.get("tuple_dependency_commitment_sha256") != expected_base
        or digest(
            {
                key: value
                for key, value in base_manifest.items()
                if key != "tuple_dependency_commitment_sha256"
            }
        )
        != expected_base
        or base_manifest.get("amendment_commitment_sha256")
        != amendment["amendment_commitment_sha256"]
        or base_manifest.get("model_inference_executed") is not False
        or base_manifest.get("restricted_mount_present") is not False
    ):
        raise RuntimeError("E_TUPLE_BASE_DEPENDENCY_MANIFEST")

    egobaby = runtime["additional_public_artifacts"]["egobaby_loader"]
    egobaby_root = model_root / "code/egobabyvlm"
    egobaby_archive = _clone_public_repository(
        egobaby["repository"], egobaby["commit"], egobaby_root
    )
    egobaby_license = _license_digest(
        egobaby_root,
        "e668cfe2504c4ffa4bbc7dbd63d3302d7561f4df107cfe0ac693c0fe3fa6f01d",
    )
    grounding_patch = _apply_grounding_dino_fallback_patch(
        model_root / "code/GroundingDINO"
    )

    bert = runtime["additional_public_artifacts"]["bert_base_uncased"]
    bert_root = model_root / "bert-base-uncased"
    bert_records = []
    for name, expected in sorted(bert["files_sha256"].items()):
        path = bert_root / name
        _download_exact_public_artifact(
            "https://huggingface.co/"
            f"{bert['repository']}/resolve/{bert['revision']}/{name}?download=true",
            path,
            expected,
        )
        bert_records.append(
            {"name": name, "sha256": file_digest(path), "bytes": path.stat().st_size}
        )

    wheel_root = model_root / "runtime-distributions"
    wheel_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_only = {"antlr4-python3-runtime", "fvcore", "iopath", "mmcv"}
    requirements = [
        f"{package}=={version}"
        for package, version in sorted(runtime["dependency_versions"].items())
    ]
    for package, version in sorted(runtime["dependency_versions"].items()):
        command = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--no-deps",
            "--dest",
            str(wheel_root),
        ]
        if package in source_only:
            command.append("--no-binary=:all:")
        else:
            command.append("--only-binary=:all:")
        command.append(f"{package}=={version}")
        subprocess.run(command, check=True)
    dependency_artifacts = sorted(path for path in wheel_root.iterdir() if path.is_file())
    if len(dependency_artifacts) != len(requirements):
        raise RuntimeError("E_TUPLE_RUNTIME_ARTIFACT_COUNT")
    for filename, record in runtime["added_dependency_wheels"].items():
        path = wheel_root / filename
        if not path.is_file() or file_digest(path) != record["sha256"]:
            raise RuntimeError("E_TUPLE_RUNTIME_ADDED_DEPENDENCY_HASH")

    dependency_root = model_root / "runtime-pydeps"
    if dependency_root.exists():
        shutil.rmtree(dependency_root)
    dependency_root.mkdir(parents=True, mode=0o700)
    install_environment = {
        **os.environ,
        "MMCV_WITH_OPS": "0",
        "SAM2_BUILD_CUDA": "0",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheel_root),
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(dependency_root),
            *requirements,
        ],
        env=install_environment,
        check=True,
    )
    distributions = _installed_distributions(dependency_root)
    installed = {row["name"].casefold(): row["version"] for row in distributions}
    for package, version in runtime["dependency_versions"].items():
        normalized = package.replace("_", "-").casefold()
        matches = [
            actual
            for name, actual in installed.items()
            if name.replace("_", "-") == normalized
        ]
        if matches != [version]:
            raise RuntimeError("E_TUPLE_RUNTIME_DISTRIBUTION")

    manifest = {
        "schema_version": 1,
        "status": "PASS_RUNTIME_READY_LOCAL_RELOAD_PENDING_BLIND_SIZING",
        "amendment_commitment_sha256": amendment["amendment_commitment_sha256"],
        "runtime_amendment_commitment_sha256": runtime[
            "runtime_amendment_commitment_sha256"
        ],
        "base_dependency_manifest_commitment_sha256": expected_base,
        "base_container": runtime["base_container"],
        "egobaby_loader": {
            "repository": egobaby["repository"],
            "commit": egobaby["commit"],
            **egobaby_archive,
            "license_sha256": egobaby_license,
        },
        "bert_base_uncased": {
            "revision": bert["revision"],
            "license": bert["license"],
            "files": bert_records,
        },
        "dependency_artifacts": [
            {
                "name": path.name,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
            for path in dependency_artifacts
        ],
        "added_dependency_wheels": runtime["added_dependency_wheels"],
        "installed_distributions": distributions,
        "compatibility_adapters": runtime["compatibility_adapters"],
        "grounding_dino_compatibility_patch": grounding_patch,
        "local_files_only_required": True,
        "network_disabled_for_inference": True,
        "telemetry_tracking_disabled": True,
        "restricted_mount_present": False,
        "model_inference_executed": False,
    }
    manifest["runtime_dependency_commitment_sha256"] = digest(manifest)
    write_private(_tuple_run_root(args.public_root) / "runtime_manifest.json", manifest)
    return {
        "status": "PASS_RUNTIME_READY",
        "dependency_count": len(runtime["dependency_versions"]),
        "dependency_artifact_count": len(dependency_artifacts),
        "installed_distribution_count": len(distributions),
        "additional_repository_count": 1,
        "additional_model_file_count": len(bert_records),
        "restricted_mount_present": False,
        "model_inference_executed": False,
        "runtime_dependency_commitment_sha256": manifest[
            "runtime_dependency_commitment_sha256"
        ],
    }


def _verify_tuple_runtime_manifest(
    public: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    runtime = _tuple_runtime_amendment(cfg)
    path = _tuple_run_root(public) / "runtime_manifest.json"
    value = json.loads(path.read_text())
    commitment = value.pop("runtime_dependency_commitment_sha256", None)
    expected_base = cfg["calibration_C"]["extractor"][
        "mechanistic_training_tuple_premodel_result"
    ]["dependency_manifest_commitment_sha256"]
    if (
        not isinstance(commitment, str)
        or digest(value) != commitment
        or value.get("status")
        != "PASS_RUNTIME_READY_LOCAL_RELOAD_PENDING_BLIND_SIZING"
        or value.get("runtime_amendment_commitment_sha256")
        != runtime["runtime_amendment_commitment_sha256"]
        or value.get("base_dependency_manifest_commitment_sha256") != expected_base
        or value.get("restricted_mount_present") is not False
        or value.get("model_inference_executed") is not False
    ):
        raise RuntimeError("E_TUPLE_RUNTIME_MANIFEST")
    value["runtime_dependency_commitment_sha256"] = commitment
    return value


def _release_cuda(*values: Any) -> None:
    import gc
    import torch

    for value in values:
        del value
    gc.collect()
    torch.cuda.empty_cache()


def _install_egohos_segmentation_compatibility() -> None:
    import types
    import numpy as np
    import torch

    for name, value in {"int": int, "float": float, "bool": bool}.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    def prohibited(*_args, **_kwargs):
        raise RuntimeError("E_TUPLE_EGOHOS_UNUSED_MMCV_OP_INVOKED")

    class ProhibitedOperation(torch.nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            raise RuntimeError("E_TUPLE_EGOHOS_UNUSED_MMCV_OP_INVOKED")

    ops = types.ModuleType("mmcv.ops")
    ops.sigmoid_focal_loss = prohibited
    ops.point_sample = prohibited
    ops.PSAMask = ProhibitedOperation
    ops.CrissCrossAttention = ProhibitedOperation
    sys.modules["mmcv.ops"] = ops


def _grounding_fallback_consistency(device: str) -> None:
    import torch
    from groundingdino.models.GroundingDINO.ms_deform_attn import (
        multi_scale_deformable_attn_pytorch,
    )

    generator = torch.Generator().manual_seed(20260802)
    value = torch.rand((1, 4, 1, 2), generator=generator)
    shapes = torch.tensor([[2, 2]], dtype=torch.long)
    locations = torch.rand((1, 2, 1, 1, 2, 2), generator=generator)
    weights = torch.rand((1, 2, 1, 1, 2), generator=generator)
    weights = weights / weights.sum(dim=(-1, -2), keepdim=True)
    expected = multi_scale_deformable_attn_pytorch(
        value, shapes, locations, weights
    )
    observed = multi_scale_deformable_attn_pytorch(
        value.to(device), shapes.to(device), locations.to(device), weights.to(device)
    ).cpu()
    if not torch.allclose(observed, expected, rtol=1e-4, atol=1e-5):
        raise RuntimeError("E_TUPLE_GROUNDING_FALLBACK_NUMERICS")


def _tuple_dummy_image(size: int = 384):
    import numpy as np

    grid_y, grid_x = np.indices((size, size))
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = (grid_x % 256).astype(np.uint8)
    image[..., 1] = (grid_y % 256).astype(np.uint8)
    image[..., 2] = (((grid_x // 32 + grid_y // 32) % 2) * 255).astype(np.uint8)
    return image


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


def _verify_container_torch_not_shadowed(
    target: Path, container: dict[str, Any]
) -> None:
    distributions = {
        value["name"] for value in _installed_distributions(target)
    }
    if distributions.intersection({"torch", "torchvision"}):
        raise RuntimeError("E_ACTIVITY_CONTAINER_TORCH_SHADOWED")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch,torchvision; print(torch.__version__); print(torchvision.__version__)",
        ],
        env={**os.environ, "PYTHONPATH": str(target)},
        text=True,
        capture_output=True,
        check=True,
    )
    versions = completed.stdout.splitlines()
    if versions != [container["torch"], container["torchvision"]]:
        raise RuntimeError("E_ACTIVITY_CONTAINER_TORCH_VERSION")


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
        raise RuntimeError("E_ACTIVITY_CLIP_CODE_NOT_STAGED")
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
                    "--target",
                    str(target),
                    "--report",
                    str(report),
                    *(
                        ["--no-deps"]
                        if candidate_id == "egohod_egovideo_l_zero_shot"
                        else []
                    ),
                    *requirements,
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        os.chmod(report, 0o600)
        _verify_container_torch_not_shadowed(target, runtime["container"])
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
        "host_clean_tree_preflight": os.environ.get(
            "PHASE4_ACTIVITY_CODE_CLEAN"
        )
        == "1",
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
        or value.get("host_clean_tree_preflight") is not True
    ):
        raise RuntimeError("E_ACTIVITY_DEPENDENCY_MANIFEST")
    value["dependency_manifest_commitment_sha256"] = commitment
    return value


def _verify_repository_commit(path: Path, expected: str) -> None:
    git = shutil.which("git")
    if git is not None:
        completed = subprocess.run(
            [git, "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
        if completed.stdout.strip() != expected:
            raise RuntimeError("E_ACTIVITY_CODE_COMMIT")
        dirty = subprocess.run(
            [git, "-C", str(path), "diff", "--quiet"], check=False
        )
        if dirty.returncode != 0:
            raise RuntimeError("E_ACTIVITY_CODE_DIRTY")
        return
    git_root = path / ".git"
    head_path = git_root / "HEAD"
    if not head_path.is_file():
        raise RuntimeError("E_ACTIVITY_CODE_COMMIT")
    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        reference = head.removeprefix("ref: ")
        loose = git_root / reference
        if loose.is_file():
            head = loose.read_text().strip()
        else:
            packed = git_root / "packed-refs"
            matches = [
                line.split()[0]
                for line in packed.read_text().splitlines()
                if line and not line.startswith(("#", "^")) and line.split()[1] == reference
            ] if packed.is_file() else []
            if len(matches) != 1:
                raise RuntimeError("E_ACTIVITY_CODE_COMMIT")
            head = matches[0]
    if head != expected:
        raise RuntimeError("E_ACTIVITY_CODE_COMMIT")


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


def _install_egohod_optional_import_compatibility() -> None:
    import types
    import torch

    ipdb = types.ModuleType("ipdb")
    ipdb.set_trace = lambda: None
    cv2 = types.ModuleType("cv2")

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, value):
            if self.drop_prob != 0.0 and self.training:
                raise RuntimeError("E_EGOHOD_UNEXPECTED_STOCHASTIC_DEPTH")
            return value

    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = DropPath
    layers.to_2tuple = lambda value: value if isinstance(value, tuple) else (value, value)
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    timm_models = types.ModuleType("timm.models")
    timm_models.layers = layers
    timm = types.ModuleType("timm")
    timm.models = timm_models

    def constant_init(module, value, bias=0):
        if getattr(module, "weight", None) is not None:
            torch.nn.init.constant_(module.weight, value)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.constant_(module.bias, bias)

    def trunc_normal_init(module, mean=0.0, std=1.0, a=-2.0, b=2.0, bias=0):
        if getattr(module, "weight", None) is not None:
            torch.nn.init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.constant_(module.bias, bias)

    weight_init = types.ModuleType("mmengine.model.weight_init")
    weight_init.constant_init = constant_init
    weight_init.trunc_normal_init = trunc_normal_init
    mmengine_model = types.ModuleType("mmengine.model")
    mmengine_model.weight_init = weight_init
    mmengine = types.ModuleType("mmengine")
    mmengine.model = mmengine_model
    sys.modules.update(
        {
            "ipdb": ipdb,
            "cv2": cv2,
            "timm": timm,
            "timm.models": timm_models,
            "timm.models.layers": layers,
            "mmengine": mmengine,
            "mmengine.model": mmengine_model,
            "mmengine.model.weight_init": weight_init,
        }
    )


def _egohod_checkpoint_safe_globals() -> list[Any]:
    import argparse
    import builtins
    import collections
    import typing

    import numpy
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode

    return [
        argparse.Namespace,
        builtins.dict,
        builtins.int,
        builtins.list,
        collections.defaultdict,
        (numpy.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        numpy.dtype,
        numpy.dtypes.Float64DType,
        ContainerMetadata,
        Metadata,
        DictConfig,
        ListConfig,
        AnyNode,
        typing.Any,
    ]


def _load_egohod_activity_adapter(
    public: Path,
    candidate: dict[str, Any],
    cfg: dict[str, Any],
    labels: list[str],
    device: str,
    runtime_override: dict[str, Any] | None = None,
    prompt_groups_override: dict[str, list[str]] | None = None,
):
    import torch
    import torch.nn.functional as functional

    code_root = _activity_code_root(public, candidate["candidate_id"])
    clip_root = public / "models/activity-code/CLIP"
    runtime = runtime_override or _activity_config(cfg)["runtime_environment"][
        "egohod"
    ]
    _verify_repository_commit(code_root, candidate["code_commit"])
    _verify_repository_commit(clip_root, runtime["openai_CLIP_commit"])
    _install_egohod_optional_import_compatibility()
    sys.path.insert(0, str(clip_root))
    sys.path.insert(0, str(code_root))
    import clip
    from model.clip import CLIP
    from model.transformer import TextTransformer, VisionTransformer

    checkpoint = _activity_checkpoint_root(public, candidate["candidate_id"]) / candidate["weight_file"]
    if not checkpoint.is_file() or file_digest(checkpoint) != candidate["weight_sha256"]:
        raise RuntimeError("E_ACTIVITY_WEIGHT_COMMITMENT")
    checkpoint_loading = runtime["checkpoint_safe_load"]
    unsafe_globals = set(
        torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    )
    if unsafe_globals != set(checkpoint_loading["exact_allowed_globals"]):
        raise RuntimeError("E_EGOHOD_UNEXPECTED_CHECKPOINT_GLOBAL")
    with torch.serialization.safe_globals(_egohod_checkpoint_safe_globals()):
        loaded = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
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
    prompt_groups = prompt_groups_override or cfg["calibration_C"]["extractor"][
        "coverage_repair"
    ]["prompt_ensembles"]["activity"]
    if set(prompt_groups) != set(labels) or any(
        len(prompt_groups[label]) != 3 for label in labels
    ):
        raise RuntimeError("E_EGOHOD_PROMPT_ENSEMBLE")
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


def _size_tuple_grounding_and_sam(
    public: Path,
    device: str,
    image_array,
    sizing_validation: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from PIL import Image

    model_root = _tuple_model_root(public)
    grounding_root = model_root / "code/GroundingDINO"
    sys.path.insert(0, str(grounding_root))
    from groundingdino.datasets import transforms as grounding_transforms
    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig

    _grounding_fallback_consistency(device)
    arguments = SLConfig.fromfile(
        str(grounding_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    )
    arguments.device = device
    arguments.text_encoder_type = str(model_root / "bert-base-uncased")
    grounding = build_model(arguments)
    checkpoint_path = model_root / "weights/groundingdino_swint_ogc.pth"
    if file_digest(checkpoint_path) != (
        "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
    ):
        raise RuntimeError("E_TUPLE_GROUNDING_WEIGHT")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    grounding.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    grounding = grounding.to(device).eval()
    transform = grounding_transforms.Compose(
        [
            grounding_transforms.RandomResize([800], max_size=1333),
            grounding_transforms.ToTensor(),
            grounding_transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    tensor, _ = transform(Image.fromarray(image_array), None)
    rule = sizing_validation["grounding_dino_output_rule"]
    caption = rule["sizing_caption"]
    with torch.inference_mode():
        outputs = grounding(tensor[None].to(device), captions=[caption])
    raw_logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    scores = raw_logits.sigmoid()
    tokenizer_attention_mask = grounding.tokenizer(
        [caption], padding="longest", return_tensors="pt"
    ).attention_mask.bool()
    if (
        tokenizer_attention_mask.ndim != 2
        or tokenizer_attention_mask.shape[0] != 1
        or tokenizer_attention_mask.shape[1] > raw_logits.shape[-1]
    ):
        raise RuntimeError("E_TUPLE_GROUNDING_PADDING_MASK")
    expected_active = torch.zeros(
        raw_logits.shape[-1], dtype=torch.bool, device=raw_logits.device
    )
    expected_active[: tokenizer_attention_mask.shape[1]] = (
        tokenizer_attention_mask[0].to(raw_logits.device)
    )
    expected_active = expected_active.view(1, 1, -1).expand_as(raw_logits)
    metrics = {
        "raw_nan_count": int(torch.isnan(raw_logits).sum()),
        "raw_positive_infinity_count": int(torch.isposinf(raw_logits).sum()),
        "raw_active_position_nonfinite_count": int(
            (~torch.isfinite(raw_logits[expected_active])).sum()
        ),
        "raw_padding_position_non_negative_infinity_count": int(
            (~torch.isneginf(raw_logits[~expected_active])).sum()
        ),
        "post_sigmoid_nonfinite_count": int((~torch.isfinite(scores)).sum()),
        "pred_box_nonfinite_count": int((~torch.isfinite(boxes)).sum()),
        "pred_box_min": float(boxes.min()),
        "pred_box_max": float(boxes.max()),
    }
    _validate_grounding_sizing_counts(metrics, rule)
    grounding_width = int(scores.numel() + boxes.numel())
    del outputs, raw_logits, scores, boxes, tensor, checkpoint, grounding
    del tokenizer_attention_mask, expected_active, metrics
    _release_cuda()

    sam_root = model_root / "code/sam2"
    sys.path.insert(0, str(sam_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_path = model_root / "weights/sam2.1_hiera_base_plus.pt"
    if file_digest(sam_path) != (
        "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
    ):
        raise RuntimeError("E_TUPLE_SAM_WEIGHT")
    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        str(sam_path),
        device=device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(sam)
    predictor.set_image(image_array)
    import numpy as np

    masks, scores, logits = predictor.predict(
        box=np.asarray([64, 64, image_array.shape[1] - 64, image_array.shape[0] - 64]),
        multimask_output=False,
    )
    if (
        masks.shape[0] != 1
        or not np.isfinite(masks).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(logits).all()
    ):
        raise RuntimeError("E_TUPLE_SAM_NONFINITE")
    sam_width = int(masks.size + scores.size + logits.size)
    del predictor, sam, masks, scores, logits
    _release_cuda()
    return {"finite": True, "output_width": grounding_width + sam_width}


def _size_tuple_dinov2(public: Path, device: str) -> dict[str, Any]:
    import torch

    model_root = _tuple_model_root(public)
    sys.path.insert(0, str(model_root / "code/dinov2"))
    from dinov2.hub.backbones import dinov2_vitb14

    checkpoint_path = model_root / "weights/dinov2_vitb14_pretrain.pth"
    if file_digest(checkpoint_path) != (
        "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
    ):
        raise RuntimeError("E_TUPLE_DINOV2_WEIGHT")
    model = dinov2_vitb14(pretrained=False).to(device).eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    image = torch.zeros((1, 3, 224, 224), device=device)
    with torch.inference_mode():
        output = model(image)
    if output.shape != (1, 768) or not torch.isfinite(output).all():
        raise RuntimeError("E_TUPLE_DINOV2_OUTPUT")
    width = int(output.numel())
    del output, image, state, model
    _release_cuda()
    return {"finite": True, "output_width": width}


def _size_tuple_pe_core(
    public: Path, cfg: dict[str, Any], device: str, image_array
) -> dict[str, Any]:
    import torch
    from PIL import Image

    model, transform, text, slices = _load_vision(public, cfg, device)
    features, labels, margins = _vision_batch(
        model,
        transform,
        text,
        slices,
        [Image.fromarray(image_array)],
        None,
        device,
    )
    values = [float(value) for group in margins for value in group.values()]
    if not torch.isfinite(features).all() or not values or not all(
        math.isfinite(value) for value in values
    ):
        raise RuntimeError("E_TUPLE_PE_OUTPUT")
    width = int(features.numel() + len(values) + sum(len(row) for row in labels))
    del features, labels, margins, text, transform, model
    _release_cuda()
    return {"finite": True, "output_width": width}


def _load_egohos_segmentor(config_path: Path, checkpoint_path: Path, device: str):
    import numpy
    import torch
    import mmcv
    from mmseg.models import build_segmentor

    config = mmcv.Config.fromfile(str(config_path))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get("test_cfg"))
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path))
    expected_unsafe = {"numpy.core.multiarray.scalar", "numpy.dtype"}
    if unsafe != expected_unsafe:
        raise RuntimeError("E_TUPLE_EGOHOS_CHECKPOINT_GLOBAL")
    safe = [
        (numpy.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        numpy.dtype,
        numpy.dtypes.Float64DType,
    ]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["state_dict"].items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"E_TUPLE_EGOHOS_STATE missing={len(missing)} unexpected={len(unexpected)}"
        )
    return model.to(device).eval(), config


def _size_tuple_egohos(
    public: Path, device: str, image_array, scratch: Path
) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    _install_egohos_segmentation_compatibility()
    model_root = _tuple_model_root(public)
    sys.path.insert(0, str(model_root / "code/EgoHOS/mmsegmentation"))
    sys.path.insert(0, str(model_root / "code/EgoHOS"))
    work = model_root / "egohos-checkpoints/work_dirs"
    media_root = scratch / "egohos"
    image_root = media_root / "images"
    image_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    image_path = image_root / "public-dummy.png"
    Image.fromarray(image_array[:360, :480]).save(image_path)
    os.chmod(image_path, 0o600)
    tensor = torch.from_numpy(image_array[:360, :480].astype(np.float32))
    mean = torch.tensor([123.675, 116.28, 103.53])
    standard = torch.tensor([58.395, 57.12, 57.375])
    tensor = ((tensor - mean) / standard).permute(2, 0, 1).unsqueeze(0).to(device)
    stages = [
        ("seg_twohands_ccda", "best_mIoU_iter_56000.pth", "pred_twohands"),
        ("twohands_to_cb_ccda", "best_mIoU_iter_76000.pth", "pred_cb"),
        ("twohands_cb_to_obj1_ccda", "best_mIoU_iter_34000.pth", None),
    ]
    width = 0
    for folder, checkpoint_name, output_folder in stages:
        root = work / folder
        model, config = _load_egohos_segmentor(
            root / f"{folder}.py", root / checkpoint_name, device
        )
        metadata = {
            "filename": str(image_path),
            "ori_shape": (360, 480, 3),
            "img_shape": (360, 480, 3),
            "pad_shape": (360, 480, 3),
            "scale_factor": 1.0,
            "flip": False,
            "additional_channel": str(config.get("additional_channel", "")),
        }
        with torch.inference_mode():
            logits = model.encode_decode(tensor, [metadata])
        if logits.ndim != 4 or not torch.isfinite(logits).all():
            raise RuntimeError("E_TUPLE_EGOHOS_OUTPUT")
        prediction = logits.argmax(dim=1)[0].byte().cpu().numpy()
        width += int(logits.numel())
        if output_folder is not None:
            target_root = media_root / output_folder
            target_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = target_root / "public-dummy.png"
            Image.fromarray(prediction).save(target)
            os.chmod(target, 0o600)
        del prediction, logits, model
        _release_cuda()
    del tensor
    _release_cuda()
    return {"finite": True, "output_width": width}


def size_tuple_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    cfg = json.loads(args.config.read_text())
    amendment = _tuple_amendment(cfg)
    runtime = _tuple_runtime_amendment(cfg)
    fixture_protocol = _tuple_fixture_protocol(cfg)
    sizing_validation = _tuple_sizing_validation(cfg)
    dependency = _verify_tuple_runtime_manifest(args.public_root, cfg)
    if torch.cuda.device_count() != 1 or args.device != "cuda":
        raise RuntimeError("E_TUPLE_SIZING_TOPOLOGY")
    torch.cuda.reset_peak_memory_stats()
    image_array = _tuple_dummy_image(512)
    modules = []

    adapter_path = Path(__file__).resolve().with_name(
        "synthetic_video_language_adapter.py"
    )
    if not adapter_path.is_file() or file_digest(adapter_path) != (
        TUPLE_LANGUAGE_ADAPTER_SHA256
    ):
        raise RuntimeError("E_TUPLE_LANGUAGE_ADAPTER_SOURCE")
    from synthetic_video_language_adapter import validate_asr_prediction

    language = validate_asr_prediction(
        {
            "text": "Ein Ball",
            "language": "de",
            "words": [
                {"word": "Ein", "start": 0.0, "end": 0.4, "probability": 0.9},
                {"word": "Ball", "start": 0.4, "end": 0.8, "probability": 0.9},
            ],
        },
        1.0,
    )
    if language.get("status") != "ACCEPT":
        raise RuntimeError("E_TUPLE_LANGUAGE_SIZING")
    modules.append({"module": "adapter", "finite": True, "output_width": 1})

    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    from wordfreq import zipf_frequency

    nltk.data.path[:] = [
        str(
            _stage_tuple_nltk_resources(
                args.public_root, args.scratch_root, cfg
            )
        )
    ]
    mentions = _lexical_mentions(
        "The red ball",
        nltk.pos_tag,
        WordNetLemmatizer().lemmatize,
        zipf_frequency,
        amendment["axes"][1]["frequency_bands"],
    )
    if not mentions:
        raise RuntimeError("E_TUPLE_LEXICAL_SIZING")
    modules.append(
        {"module": "lexical", "finite": True, "output_width": len(mentions)}
    )

    from PIL import Image

    first = Image.fromarray(image_array)
    second = Image.fromarray(image_array[:, ::-1].copy())
    sensor = _image_metrics(second, first)
    if not all(value is None or math.isfinite(value) for value in sensor.values()):
        raise RuntimeError("E_TUPLE_SENSOR_SIZING")
    modules.append(
        {"module": "sensor", "finite": True, "output_width": len(sensor)}
    )

    grounding = _size_tuple_grounding_and_sam(
        args.public_root,
        args.device,
        image_array,
        sizing_validation,
    )
    modules.append({"module": "grounding_and_tracking", **grounding})
    modules.append(
        {"module": "recurrence", **_size_tuple_dinov2(args.public_root, args.device)}
    )
    modules.append(
        {"module": "attribute", **_size_tuple_pe_core(args.public_root, cfg, args.device, image_array)}
    )
    modules.append(
        {"module": "hand_contact", **_size_tuple_egohos(args.public_root, args.device, image_array, args.scratch_root)}
    )

    activity = cfg["calibration_C"]["extractor"][
        "activity_checkpoint_selection_amendment"
    ]
    activity_dependency = _verify_activity_dependency_manifest(
        args.public_root, activity
    )
    candidate = next(
        value
        for value in activity["bounded_candidates"]
        if value["candidate_id"] == "egohod_egovideo_l_zero_shot"
    )
    action_protocol = fixture_protocol["order_dependent_action_control"]
    labels = action_protocol["labels"]
    score, frame_count, _ = _load_egohod_activity_adapter(
        args.public_root,
        candidate,
        cfg,
        labels,
        args.device,
        runtime_override=activity["runtime_environment"]["egohod"],
        prompt_groups_override=action_protocol["prompt_ensembles"],
    )
    action_output = score(
        image_array[None].repeat(frame_count, axis=0)
    )
    if len(action_output) != len(labels) or not all(
        math.isfinite(value) for value in action_output
    ):
        raise RuntimeError("E_TUPLE_ACTION_SIZING")
    modules.append(
        {"module": "order_action", "finite": True, "output_width": len(action_output)}
    )
    del score, action_output
    _release_cuda()

    if len(modules) != runtime["local_reload_gate"]["module_count"]:
        raise RuntimeError("E_TUPLE_SIZING_MODULE_COUNT")
    if any(not row["finite"] or row["output_width"] <= 0 for row in modules):
        raise RuntimeError("E_TUPLE_SIZING_OUTPUT")
    peak = _gpu_peak_gib(args.device)
    if peak > float(runtime["local_reload_gate"]["peak_VRAM_GiB_max"]):
        raise RuntimeError("E_TUPLE_SIZING_VRAM")
    record = {
        "schema_version": 1,
        "status": "PASS_LABEL_BLIND_LOCAL_RELOAD_SIZING",
        "amendment_commitment_sha256": amendment["amendment_commitment_sha256"],
        "runtime_amendment_commitment_sha256": runtime[
            "runtime_amendment_commitment_sha256"
        ],
        "public_fixture_protocol_commitment_sha256": fixture_protocol[
            "protocol_commitment_sha256"
        ],
        "sizing_validation_commitment_sha256": sizing_validation[
            "validation_commitment_sha256"
        ],
        "runtime_dependency_commitment_sha256": dependency[
            "runtime_dependency_commitment_sha256"
        ],
        "activity_dependency_manifest_commitment_sha256": activity_dependency[
            "dependency_manifest_commitment_sha256"
        ],
        "modules": modules,
        "fixture_labels_used": False,
        "scientific_metric_computed": False,
        "prediction_or_score_retained": False,
        "external_call_count": 0,
        "restricted_mount_present": False,
        "peak_vram_gib": peak,
        "total_runtime_seconds": time.monotonic() - started,
    }
    record["tuple_sizing_commitment_sha256"] = digest(record)
    write_private(_tuple_run_root(args.public_root) / "sizing_result.json", record)
    return {
        "status": "PASS_LABEL_BLIND_SIZING",
        "module_count": len(modules),
        "finite_output_count": len(modules),
        "failure_count": 0,
        "external_call_count": 0,
        "scientific_metric_count": 0,
        "retained_prediction_count": 0,
        "peak_vram_gib": peak,
        "total_runtime_seconds": record["total_runtime_seconds"],
        "tuple_sizing_commitment_sha256": record[
            "tuple_sizing_commitment_sha256"
        ],
    }


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


def _install_videoprism_tokenizer_import_compatibility(code_root: Path) -> None:
    import types
    from typing import Protocol

    source = (code_root / "videoprism/models.py").read_text()
    references = set(re.findall(r"tokenizers\.([A-Za-z_][A-Za-z0-9_]*)", source))
    if references != {"SentencePieceTokenizer", "Tokenizer"}:
        raise RuntimeError("E_VIDEOPRISM_TOKENIZER_IMPORT_SURFACE")
    utility_source = (code_root / "videoprism/utils.py").read_text()
    gfile_references = set(
        re.findall(r"gfile\.([A-Za-z_][A-Za-z0-9_]*)", utility_source)
    )
    if gfile_references != {"GFile"}:
        raise RuntimeError("E_VIDEOPRISM_GFILE_IMPORT_SURFACE")

    class Tokenizer(Protocol):
        pass

    class SentencePieceTokenizer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("E_VIDEOPRISM_HOSTED_TOKENIZER_PATH_PROHIBITED")

    tokenizers = types.ModuleType("videoprism.tokenizers")
    tokenizers.Tokenizer = Tokenizer
    tokenizers.SentencePieceTokenizer = SentencePieceTokenizer
    gfile = types.ModuleType("tensorflow.io.gfile")

    def prohibited_gfile(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("E_VIDEOPRISM_HOSTED_GFILE_PATH_PROHIBITED")

    gfile.GFile = prohibited_gfile
    tensorflow_io = types.ModuleType("tensorflow.io")
    tensorflow_io.gfile = gfile
    tensorflow = types.ModuleType("tensorflow")
    tensorflow.io = tensorflow_io
    tensorflow._phase4_import_stub = True
    sys.modules["videoprism.tokenizers"] = tokenizers
    sys.modules["tensorflow"] = tensorflow
    sys.modules["tensorflow.io"] = tensorflow_io
    sys.modules["tensorflow.io.gfile"] = gfile


def _remove_videoprism_tensorflow_import_compatibility() -> None:
    tensorflow = sys.modules.get("tensorflow")
    if getattr(tensorflow, "_phase4_import_stub", False) is not True:
        raise RuntimeError("E_VIDEOPRISM_TENSORFLOW_STUB_IDENTITY")
    for name in ("tensorflow.io.gfile", "tensorflow.io", "tensorflow"):
        sys.modules.pop(name, None)


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
    _install_videoprism_tokenizer_import_compatibility(code_root)
    sys.path.insert(0, str(code_root))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from videoprism import models as videoprism
    _remove_videoprism_tensorflow_import_compatibility()

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


def size_activity_candidate(args: argparse.Namespace) -> dict[str, Any]:
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
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
    ):
        raise RuntimeError("E_ACTIVITY_OFFLINE_ENVIRONMENT")
    sizing = amendment["resource_ceiling"]["blind_sizing"]
    if sizing != {
        "fixture": "development manifest ordinal zero under the already frozen manifest order",
        "item_count_per_candidate": 1,
        "ordered_clip_only": True,
        "fixture_labels_used": False,
        "score_or_prediction_retention": "PROHIBITED",
        "scientific_metric_computation": "PROHIBITED",
        "recorded_fields": [
            "load_success",
            "finite_output_width",
            "wall_time",
            "peak_VRAM",
            "local_files_only_reload",
            "external_call_count",
        ],
        "per_candidate_wall_time_hours_max": 0.5,
        "aggregate_GPU_hours_max": 1.5,
        "pass_rule": "all three candidates load and emit the prospectively expected finite output width within their frozen VRAM and sizing wall-time ceilings with zero external calls",
    }:
        raise RuntimeError("E_ACTIVITY_SIZING_CONTRACT")
    fixture = manifest["partitions"]["development"][0]
    started = time.monotonic()
    infer, frame_count, _ = _load_activity_adapter(
        args.public_root, candidate, cfg, labels, args.device
    )
    load_runtime = time.monotonic() - started
    peak_vram = _gpu_peak_gib(args.device)
    media = (
        args.public_root
        / "public/charades/CharadesEgo_v1_480"
        / f"{fixture['id']}.mp4"
    )
    item_started = time.monotonic()
    frames, _ = _decode_uniform_activity_frames(media, frame_count)
    output = infer(frames)
    output_width = len(output)
    if not output or not all(math.isfinite(float(value)) for value in output):
        raise RuntimeError("E_ACTIVITY_SIZING_INVALID_OUTPUT")
    expected_width = int(candidate["expected_sizing_output_width"])
    if output_width != expected_width:
        raise RuntimeError("E_ACTIVITY_SIZING_OUTPUT_WIDTH")
    item_runtime = time.monotonic() - item_started
    total_runtime = time.monotonic() - started
    peak_vram = max(peak_vram, _gpu_peak_gib(args.device))
    expected_peak = float(candidate["expected_peak_vram_gib_max"])
    if peak_vram > expected_peak:
        raise RuntimeError("E_ACTIVITY_SIZING_VRAM_CEILING")
    if total_runtime > float(sizing["per_candidate_wall_time_hours_max"]) * 3600:
        raise RuntimeError("E_ACTIVITY_SIZING_WALL_CEILING")
    result = {
        "schema_version": 1,
        "status": "PASS_BLIND_RESOURCE_SIZING",
        "candidate_id": candidate["candidate_id"],
        "candidate_weight_sha256": candidate["weight_sha256"],
        "dependency_manifest_commitment_sha256": amendment[
            "runtime_environment"
        ]["shared"]["dependency_manifest_commitment_sha256"],
        "fixture_partition": "development",
        "fixture_manifest_ordinal": 0,
        "fixture_labels_used": False,
        "score_or_prediction_retained": False,
        "scientific_metric_computed": False,
        "ordered_clip_only": True,
        "item_count": 1,
        "output_width": output_width,
        "expected_output_width": expected_width,
        "load_runtime_seconds": load_runtime,
        "item_runtime_seconds": item_runtime,
        "total_runtime_seconds": total_runtime,
        "peak_vram_gib": peak_vram,
        "expected_peak_vram_gib_max": expected_peak,
        "external_call_count": 0,
        "local_files_only_reload": True,
        "telemetry_tracking_disabled": True,
        "restricted_mount_present": False,
    }
    result["activity_sizing_commitment_sha256"] = digest(result)
    write_private(
        _activity_run_root(args.public_root)
        / "sizing"
        / f"{candidate['candidate_id']}.json",
        result,
    )
    return {key: result[key] for key in ACTIVITY_SIZING_FIELDS}


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
    candidates = [
        public / "models/mechanistic-tuples/weights/PE-Core-L14-336.pt"
    ] + list(
        (public / "models/pe-hf-home/hub").glob(
            f"models--facebook--PE-Core-L14-336/snapshots/{frozen['revision']}/PE-Core-L14-336.pt"
        )
    )
    candidates = [
        path
        for path in candidates
        if path.is_file() and file_digest(path) == frozen["weights_sha256"]
    ]
    if len(candidates) != 1:
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
    activity_size_parser = subparsers.add_parser("activity-size")
    activity_size_parser.add_argument("--public-root", type=Path, required=True)
    activity_size_parser.add_argument("--manifest", type=Path, required=True)
    activity_size_parser.add_argument("--config", type=Path, required=True)
    activity_size_parser.add_argument("--candidate-id", required=True)
    activity_size_parser.add_argument("--device", default="cuda")
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
    tuple_prepare_parser = subparsers.add_parser("tuple-prepare")
    tuple_prepare_parser.add_argument("--public-root", type=Path, required=True)
    tuple_prepare_parser.add_argument("--config", type=Path, required=True)
    tuple_runtime_parser = subparsers.add_parser("tuple-runtime-prepare")
    tuple_runtime_parser.add_argument("--public-root", type=Path, required=True)
    tuple_runtime_parser.add_argument("--config", type=Path, required=True)
    tuple_size_parser = subparsers.add_parser("tuple-size")
    tuple_size_parser.add_argument("--public-root", type=Path, required=True)
    tuple_size_parser.add_argument("--scratch-root", type=Path, required=True)
    tuple_size_parser.add_argument("--config", type=Path, required=True)
    tuple_size_parser.add_argument("--device", default="cuda")
    tuple_audio_parser = subparsers.add_parser("tuple-audio-seed")
    tuple_audio_parser.add_argument("--output-root", type=Path, required=True)
    tuple_audio_parser.add_argument("--config", type=Path, required=True)
    tuple_fixtures_parser = subparsers.add_parser("tuple-fixtures-prepare")
    tuple_fixtures_parser.add_argument("--public-root", type=Path, required=True)
    tuple_fixtures_parser.add_argument("--audio-seed-root", type=Path, required=True)
    tuple_fixtures_parser.add_argument("--config", type=Path, required=True)
    tuple_feasibility_parser = subparsers.add_parser("tuple-fixtures-feasibility")
    tuple_feasibility_parser.add_argument("--public-root", type=Path, required=True)
    tuple_feasibility_parser.add_argument("--config", type=Path, required=True)
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
    elif args.command == "activity-size":
        value = size_activity_candidate(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=ACTIVITY_SIZING_FIELDS,
                sha256_fields=ACTIVITY_SIZING_HASH_FIELDS,
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
    elif args.command == "tuple-prepare":
        value = prepare_tuple_public(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TUPLE_PREP_FIELDS,
                sha256_fields=TUPLE_PREP_HASH_FIELDS,
            )
        )
    elif args.command == "tuple-runtime-prepare":
        value = prepare_tuple_runtime(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TUPLE_RUNTIME_PREP_FIELDS,
                sha256_fields=TUPLE_RUNTIME_PREP_HASH_FIELDS,
            )
        )
    elif args.command == "tuple-size":
        value = size_tuple_runtime(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TUPLE_SIZING_FIELDS,
                sha256_fields=TUPLE_SIZING_HASH_FIELDS,
            )
        )
    elif args.command == "tuple-audio-seed":
        value = prepare_tuple_audio_seed(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TUPLE_AUDIO_SEED_FIELDS,
                sha256_fields=TUPLE_AUDIO_SEED_HASH_FIELDS,
            )
        )
    elif args.command == "tuple-fixtures-prepare":
        value = prepare_tuple_fixtures(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TUPLE_FIXTURE_PREP_FIELDS,
                sha256_fields=TUPLE_FIXTURE_PREP_HASH_FIELDS,
            )
        )
    elif args.command == "tuple-fixtures-feasibility":
        value = prepare_tuple_fixture_feasibility(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TUPLE_FIXTURE_FEASIBILITY_FIELDS,
                sha256_fields=TUPLE_FIXTURE_FEASIBILITY_HASH_FIELDS,
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
