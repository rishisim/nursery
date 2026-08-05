#!/usr/bin/env python3
"""Governed bounded C calibration and aggregate-conditioned episode planning."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
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
import stat
import statistics
import subprocess
import sys
import tempfile
import tarfile
import time
import traceback
from typing import Any
import urllib.error
import urllib.parse
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
        "source_frame_overlap_count",
        "source_object_overlap_count",
        "target_hand_boundary_vertex_count",
        "target_hand_outside_canvas_vertex_count",
        "target_hand_boundary_item_count",
        "target_hand_outside_canvas_item_count",
        "restricted_mount_present",
        "model_inference_executed",
        "public_fixture_manifest_commitment_sha256",
    }
)
TUPLE_FIXTURE_PREP_HASH_FIELDS = frozenset(
    {"public_fixture_manifest_commitment_sha256"}
)
TUPLE_NO_HAND_REVIEW_PREP_FIELDS = frozenset(
    {
        "status",
        "partition_count",
        "nominee_count",
        "review_queue_count",
        "source_frame_count",
        "source_archive_count",
        "decode_failure_count",
        "contact_sheet_count",
        "restricted_mount_present",
        "model_inference_executed",
        "visor_hos_source_feasibility_commitment_sha256",
        "source_frame_materialization_commitment_sha256",
        "review_queue_commitment_sha256",
    }
)
TUPLE_NO_HAND_REVIEW_PREP_HASH_FIELDS = frozenset(
    {
        "visor_hos_source_feasibility_commitment_sha256",
        "source_frame_materialization_commitment_sha256",
        "review_queue_commitment_sha256",
    }
)
TUPLE_NO_HAND_REVIEW_SEAL_FIELDS = frozenset(
    {
        "status",
        "partition_count",
        "coded_count",
        "verified_no_hand_count",
        "visible_hand_count",
        "abstain_count",
        "unreviewed_count",
        "deficit_partition_count",
        "restricted_mount_present",
        "model_inference_executed",
        "visor_hos_source_feasibility_commitment_sha256",
        "review_queue_commitment_sha256",
        "review_labels_commitment_sha256",
        "verified_no_hand_seal_commitment_sha256",
    }
)
TUPLE_NO_HAND_REVIEW_SEAL_HASH_FIELDS = frozenset(
    {
        "visor_hos_source_feasibility_commitment_sha256",
        "review_queue_commitment_sha256",
        "review_labels_commitment_sha256",
        "verified_no_hand_seal_commitment_sha256",
    }
)
TUPLE_NO_HAND_REVIEW_INCOMPLETE_SEAL_FIELDS = frozenset(
    TUPLE_NO_HAND_REVIEW_SEAL_FIELDS
    - {"verified_no_hand_seal_commitment_sha256"}
)
TUPLE_NO_HAND_REVIEW_INCOMPLETE_SEAL_HASH_FIELDS = frozenset(
    TUPLE_NO_HAND_REVIEW_SEAL_HASH_FIELDS
    - {"verified_no_hand_seal_commitment_sha256"}
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
TUPLE_QUALIFICATION_FIELDS = frozenset(
    {
        "status",
        "partition",
        "module_count",
        "completed_module_count",
        "failed_module_count",
        "critical_axis_pass_count",
        "validated_axis_count",
        "action_control_status",
        "external_call_count",
        "invalid_retained_record_count",
        "silent_truncation_count",
        "public_qualification_commitment_sha256",
        "development_threshold_commitment_sha256",
    }
)
TUPLE_QUALIFICATION_HASH_FIELDS = frozenset(
    {
        "public_qualification_commitment_sha256",
        "development_threshold_commitment_sha256",
    }
)
TUPLE_HEALTH_FIELDS = frozenset(
    {
        "status",
        "attempt",
        "module_count",
        "completed_module_count",
        "failed_module_count",
        "case_count",
        "holdout_input_count",
        "scientific_metric_count",
        "failure_count",
        "invalid_retained_record_count",
        "silent_truncation_count",
        "external_call_count",
        "unaccounted_failure_count",
        "public_fixture_manifest_commitment_sha256",
        "runner_commitment_sha256",
        "config_commitment_sha256",
        "dependency_config_commitment_sha256",
        "microfixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
        "GPU_type",
        "GPU_count",
        "CPU_count",
        "memory_GiB",
        "wall_minutes",
        "GPU_hours",
        "new_storage_GiB",
        "direct_monetary_cost_USD",
        "cumulative_submission_count",
        "cumulative_wall_minutes",
        "cumulative_GPU_hours",
        "cumulative_new_storage_GiB",
        "cumulative_direct_monetary_cost_USD",
    }
)
TUPLE_HEALTH_HASH_FIELDS = frozenset(
    {
        "public_fixture_manifest_commitment_sha256",
        "runner_commitment_sha256",
        "config_commitment_sha256",
        "dependency_config_commitment_sha256",
        "microfixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
    }
)
TUPLE_HEALTH_FULL_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "route_id",
        "attempt",
        "public_fixture_manifest_commitment_sha256",
        "runner_commitment_sha256",
        "config_commitment_sha256",
        "dependency_config_commitment_sha256",
        "microfixture_manifest_commitment_sha256",
        "module_results",
        "module_count",
        "completed_module_count",
        "failed_module_count",
        "case_count",
        "holdout_input_count",
        "scientific_metric_count",
        "failure_count",
        "invalid_retained_record_count",
        "silent_truncation_count",
        "external_call_count",
        "unaccounted_failure_count",
        "network_disabled",
        "telemetry_disabled",
        "restricted_mount_present",
        "resource",
        "cumulative_resource",
        "engineering_health_commitment_sha256",
    }
)
TUPLE_PARTITION_INTEGRITY_FIELDS = frozenset(
    {
        "status",
        "partition",
        "module_count",
        "completed_module_count",
        "failed_module_count",
        "scientific_metric_count",
        "failure_count",
        "invalid_retained_record_count",
        "silent_truncation_count",
        "external_call_count",
        "unaccounted_failure_count",
        "public_fixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
        "partition_engineering_integrity_commitment_sha256",
    }
)
TUPLE_PARTITION_INTEGRITY_HASH_FIELDS = frozenset(
    {
        "public_fixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
        "partition_engineering_integrity_commitment_sha256",
    }
)
TUPLE_VISOR_HOS_SOURCE_FEASIBILITY_FIELDS = frozenset(
    {
        "status",
        "official_annotation_file_count",
        "official_annotation_bytes",
        "coco_source_count",
        "visor_contact_candidate_count",
        "visor_no_contact_candidate_count",
        "visor_no_hand_nominee_count",
        "action_item_count",
        "language_lexical_item_count",
        "referent_attribute_item_count",
        "recurrence_pair_count",
        "sensor_item_count",
        "partition_count",
        "failing_family_count",
        "pending_dependent_family_count",
        "source_subject_overlap_count",
        "source_video_overlap_count",
        "source_object_overlap_count",
        "model_inference_executed",
        "media_rendering_executed",
        "restricted_mount_present",
        "visor_hos_source_feasibility_commitment_sha256",
    }
)
TUPLE_VISOR_HOS_SOURCE_FEASIBILITY_HASH_FIELDS = frozenset(
    {"visor_hos_source_feasibility_commitment_sha256"}
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
TUPLE_WHISPER_SMALL_SHA256 = (
    "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
)
TUPLE_OPUS_MT_DE_EN_REVISION = "1a922f3b32a8e809e17a47d4b32142d8105924e5"
TUPLE_GROUNDING_DINO_SHA256 = (
    "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
)
TUPLE_SAM21_BASE_PLUS_SHA256 = (
    "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
)
TUPLE_PE_CORE_IDENTITY_SHA256 = (
    "1a7fd6c54dfa10efacc65bb966fd20b02f8db65b8875ada05bd3fecb223f010e"
)
TUPLE_ORDER_ACTION_EGOHOD_CANDIDATE_SHA256 = (
    "992a5ab7a812b511c6df92cf4892e998e1872982677093a61931c02faa51be93"
)
TUPLE_ORDER_ACTION_EGOHOD_RUNTIME_SHA256 = (
    "033c48fa4172701af5b234de2b9786b079e37c9c8a99af5bb3d056094e867553"
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


CONSTRUCT_ALIGNED_SOURCE_NO_GO_SHA256 = (
    "5f4aeff25da36cde4c35699de7031b63ae427d1aee072370bb3844e3c4413b37"
)
CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256 = (
    "842d5a16141d8b0a6bdc82d86fb405bcbb14bbbd4e6cfe645ffae328ad881a39"
)
ENGINEERING_HEALTH_AMENDMENT_SHA256 = (
    "d447a7e165136032a1fba43605d3f81881b41ec030c82e9028e1a8f5cb2c6205"
)
ENGINEERING_HEALTH_RESOURCE_REDIRECT_SHA256 = (
    "f7fc16f5c399c2a2d213b13a0d255a14b5b2f3ece41d62adaed17f61f186db6d"
)
ENGINEERING_HEALTH_DEPENDENCY_RESTORE_SHA256 = (
    "3c54503b4087fae1e993b0aa952823f988a088a5c6543760df1022e2dc046db4"
)
ENGINEERING_HEALTH_TOPOLOGY_GUARD_REPAIR_SHA256 = (
    "8db2d8ae04ee702ab3c68ff7c243afed0d8e4710c01f3f7865faa4975fb9a5b8"
)
ENGINEERING_HEALTH_BLOCKER_SHA256 = (
    "644028babc768e881276fa078b95349ba77f8418cb76d722e8baf2588f9d0f81"
)
ENGINEERING_HEALTH_REAUTHORIZATION_SHA256 = (
    "3271499c19a77ffab8c53e2cd2052ea14514682a3d4b73fd7c5179ceec4a7ff4"
)
ENGINEERING_HEALTH_REAUTHORIZATION_BLOCKER_SHA256 = (
    "59b1778b35cedd1cb020177e41fe6887371a5480f7ee6bf6e57f55d4c90edde3"
)
ENGINEERING_HEALTH_PARSER_REPAIR_REAUTHORIZATION_SHA256 = (
    "d9cf3feaa0f5c4d65978ca796b722f31e75ce1078b918d114ca35b298a148c8b"
)
ENGINEERING_HEALTH_PARSER_REPAIR_BLOCKER_SHA256 = (
    "b05dc8da3155561b182b3bcfa50c851f83828b34e063918306bdfb57fdedeb9c"
)
ENGINEERING_HEALTH_ITERATIVE_REAUTHORIZATION_SHA256 = (
    "3114e1763f65dbeb8b2f89bb2a0480c86f4266f888c1ac2ff740bee85d357ab9"
)
ENGINEERING_HEALTH_ITERATIVE_ATTEMPT_6_BLOCKER_SHA256 = (
    "e559cd535d2a6dd833d2588c75b180754260dd3ff68ea9f6731a0e4478a6d114"
)
ENGINEERING_HEALTH_PROGRESS_REPAIR_SHA256 = (
    "a2d1347bef14848a5238f9a10c6e94da8eaa68593aa3d97e8a460dfbf8694d07"
)
ENGINEERING_HEALTH_ATTEMPT_7_BLOCKER_SHA256 = (
    "03c09a61cedb29e04cf465287db693cd1248c53d45d9a7a47a777e6cdf1d594d"
)
ENGINEERING_HEALTH_EXTENDED_WALL_REPAIR_SHA256 = (
    "d2db51229719a0e64f84da9541a284d88c75a2fd32e2e186ba34e17ab5eed6e7"
)
ENGINEERING_HEALTH_SCHEDULER_POLICY_SHA256 = (
    "8ef8e53c2754fe13b91518c02f419a1d3c4f3162aa18648f7044986854d327d6"
)
ENGINEERING_HEALTH_ATTEMPT_8_BLOCKER_SHA256 = (
    "409c36d2c3ba4aefdd2f510c661ba363c000fc232dce2fcfba0151ba25f9aad7"
)
ENGINEERING_HEALTH_GIT_FALLBACK_REPAIR_SHA256 = (
    "b6a93e3a0b0b716d8bdd8fdd47656e69f8ff5b66c0a3ec8f96973e565a9066f9"
)
ENGINEERING_HEALTH_ATTEMPT_9_BLOCKER_SHA256 = (
    "2fed8750a5f60929e144aecb89ef6f3bb1d1d048d9cd1d1080e8f21393ae7a43"
)
ENGINEERING_HEALTH_HISTORICAL_LINEAGE_REPAIR_SHA256 = (
    "be09b2e0c3a2026957dbd20b18129e65cd1b8ca03a8a62fdb49d568a76f526b0"
)
ENGINEERING_HEALTH_ATTEMPT_10_BLOCKER_SHA256 = (
    "1575ef02678ef0a41e32d62af7b3c8807ce449df65243c97b60baddca8ad0257"
)
ENGINEERING_HEALTH_PORTABLE_AST_REPAIR_SHA256 = (
    "7896a53385551d206736cff11d803ed6f7600317abb8cd96c94ad8496f4b67c1"
)
ENGINEERING_HEALTH_ATTEMPT_11_BLOCKER_SHA256 = (
    "e86739340a2d969b04b823932f9583d3defe7ed92a6d849042a681d03b7fb2f5"
)
ENGINEERING_HEALTH_FIXTURE_BIND_REPAIR_SHA256 = (
    "cca49a50a895ab4ac94997fbfe7bc256efc4b9ae790b1925a5671dc3b1a1e3f7"
)
ENGINEERING_HEALTH_ATTEMPT_12_BLOCKER_SHA256 = (
    "5471156337683b5b3695ee26ea9acfa1c658acfb8400a6c681538eecf6e948bc"
)
ENGINEERING_HEALTH_SUBMISSION_EXPORT_REPAIR_SHA256 = (
    "e700576735251403138e34fa9918fa2ab0e3d723e2252871302fcd4da77516ba"
)
CONSTRUCT_ALIGNED_ACTION_COUNTS = {"development": 44, "holdout": 44}
CONSTRUCT_ALIGNED_ACTION_CLASS_COUNTS = {
    "development": {
        "open": 6,
        "close": 6,
        "take": 6,
        "put": 6,
        "sit_down": 6,
        "stand_up": 6,
        "turn_on": 5,
        "turn_off": 3,
    },
    "holdout": {
        "open": 6,
        "close": 6,
        "take": 6,
        "put": 6,
        "sit_down": 6,
        "stand_up": 6,
        "turn_on": 6,
        "turn_off": 2,
    },
}
CONSTRUCT_ALIGNED_ACTION_DEFICITS = [
    {
        "partition": "development",
        "label": "turn_on",
        "required_count": 6,
        "available_count": 5,
    },
    {
        "partition": "development",
        "label": "turn_off",
        "required_count": 6,
        "available_count": 3,
    },
    {
        "partition": "holdout",
        "label": "turn_off",
        "required_count": 6,
        "available_count": 2,
    },
]


def _construct_aligned_ltx_resume_amendment(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the sole prospective bridge over the sealed action-only no-go."""

    try:
        value = cfg["construct_aligned_ltx_resume_amendment"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_CONSTRUCT_ALIGNED_RESUME_NOT_FROZEN") from error
    if (
        cfg.get("schema_version") not in {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_BEFORE_RUNNER_CHANGE_NO_HAND_REVIEW_PUBLIC_DEVELOPMENT_HOLDOUT_C_GENERATOR_OR_SYNTHETIC_LEARNER_OUTCOMES"
    ):
        raise RuntimeError("E_CONSTRUCT_ALIGNED_RESUME_NOT_FROZEN")
    payload = json.loads(json.dumps(value))
    expected = payload.pop("amendment_commitment_sha256", None)
    if (
        expected != CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256
        or digest(payload) != expected
        or value.get("amendment_commitment_scope")
        != "canonical JSON of this amendment excluding amendment_commitment_sha256"
    ):
        raise RuntimeError("E_CONSTRUCT_ALIGNED_RESUME_COMMITMENT")
    preserved = value.get("prior_results_and_amendments_preserved", {})
    combined = value.get("combined_gate", {})
    source = value.get("source_reuse", {})
    diagnostic = value.get("action_diagnostic_execution", {})
    if (
        preserved.get("complete_source_no_go")
        != CONSTRUCT_ALIGNED_SOURCE_NO_GO_SHA256
        or preserved.get("prospective_H3_amendment")
        != "d907d2479ba88c7e51d25e935e429cb5860e550a82620313e502482383e2855d"
        or combined.get("critical_axes")
        != [
            "adapter_qualified_yield",
            "noun_adjective_exposure",
            "utterance_centered_referent_visibility_dominance_ambiguity",
            "cross_episode_recurrence",
            "adjective_attribute_contrast",
        ]
        or combined.get("supporting_axes")
        != ["hand_action_coupling", "egocentric_sensor_regime"]
        or combined.get("order_action")
        != "SUPPORTING_DIAGNOSTIC_NONBLOCKING_ON_THE_EXISTING_SUBJECT_VIDEO_DISJOINT_44_DEVELOPMENT_AND_44_HOLDOUT_FIXTURES"
        or source.get("prior_complete_source_no_go_remains_final") is not True
        or source.get("no_new_source_or_candidate") is not True
        or source.get("no_threshold_relaxation") is not True
        or source.get("action_fixture_counts")
        != CONSTRUCT_ALIGNED_ACTION_COUNTS
        or diagnostic.get("development_threshold_grid")
        != [0.0, 0.005, 0.01, 0.02, 0.05]
        or "if none passes" not in str(diagnostic.get("selection_rule", ""))
        or "always run" not in str(diagnostic.get("holdout_rule", ""))
        or "remain blocking" not in str(diagnostic.get("integrity_rule", ""))
    ):
        raise RuntimeError("E_CONSTRUCT_ALIGNED_RESUME_SCHEMA")
    return value


def _engineering_health_amendment(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the sole prospective engineering-health route."""

    try:
        value = cfg["learner_effective_engineering_health_amendment"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_AMENDMENT_NOT_FROZEN") from error
    if (
        cfg.get("schema_version") not in {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_BEFORE_ENGINEERING_HEALTH_OR_NEW_SCIENTIFIC_OUTCOME"
        or value.get("route_id") != "construct-aligned-engineering-health"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_AMENDMENT_NOT_FROZEN")
    payload = json.loads(json.dumps(value))
    expected = payload.pop("amendment_commitment_sha256", None)
    if (
        expected != ENGINEERING_HEALTH_AMENDMENT_SHA256
        or digest(payload) != expected
        or value.get("amendment_commitment_scope")
        != "canonical JSON of this amendment excluding amendment_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_AMENDMENT_COMMITMENT")
    prior = value.get("prior_public_development_result", {})
    micro = value.get("engineering_microfixture_suite", {})
    resource = value.get("bounded_resource_policy", {})
    withholding = value.get("metric_withholding", {})
    threshold_state = value.get("scientific_threshold_state", {})
    if (
        prior.get("public_qualification_commitment_sha256")
        != "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
        or prior.get("canonical_subtree_sha256")
        != "c43c7a678e3a2eac10ed5a5ac75c8964520931ec180ab9306585c76d198fb8c8"
        or micro.get("cases_per_module") != 4
        or micro.get("module_count") != 7
        or micro.get("total_case_count") != 28
        or set(micro.get("required_case_classes", {}))
        != set(TUPLE_QUALIFICATION_MODULE_IDS)
        or any(
            len(classes) != 4
            for classes in micro.get("required_case_classes", {}).values()
        )
        or resource.get("GPU_type") != "NVIDIA_A30_24GB"
        or resource.get("GPU_count") != 1
        or resource.get("CPU_count") != 8
        or resource.get("memory_GiB") != 32
        or resource.get("per_submission_wall_minutes_max") != 15
        or resource.get("initial_plus_repair_resmoke_submission_count_max") != 3
        or resource.get("aggregate_GPU_hours_max") != 0.75
        or resource.get("new_storage_GiB_max") != 10
        or resource.get("direct_monetary_cost_USD") != 0
        or withholding.get("microhealth_scientific_metric_count") != 0
        or withholding.get(
            "all_module_engineering_PASS_required_before_aggregate_scientific_metrics_release"
        )
        is not True
        or threshold_state.get("DINOv2_recurrence_cosine") != 0.85
        or threshold_state.get("outcome_driven_tuning") is not False
        or threshold_state.get("threshold_change_or_relaxation") is not False
        or value.get("new_engineering_or_scientific_outcome_opened") is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_AMENDMENT_SCHEMA")
    return value


def _engineering_health_resource_redirect(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the outcome-free H100 MIG scheduler-only redirect."""

    try:
        value = cfg["learner_effective_engineering_health_resource_redirect"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_RESOURCE_REDIRECT_NOT_FROZEN") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("amendment_commitment_sha256", None)
    if (
        cfg.get("schema_version") not in {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_BEFORE_H100_ENGINEERING_HEALTH_OR_NEW_SCIENTIFIC_OUTCOME"
        or value.get("scope") != "SCHEDULER_AND_RESOURCE_LATENCY_ONLY"
        or expected != ENGINEERING_HEALTH_RESOURCE_REDIRECT_SHA256
        or digest(payload) != expected
        or value.get("amendment_commitment_scope")
        != "canonical JSON of this amendment excluding amendment_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_RESOURCE_REDIRECT_COMMITMENT")
    preserved = value.get("preserved_without_change", {})
    canceled = value.get("canceled_A30_submission", {})
    eligibility = value.get("scheduler_only_eligibility_check", {})
    topology = value.get("active_health_topology", {})
    resource = value.get("bounded_resource_policy", {})
    if (
        preserved.get("engineering_health_amendment_sha256")
        != ENGINEERING_HEALTH_AMENDMENT_SHA256
        or preserved.get("prior_public_development_no_go_sha256")
        != "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
        or preserved.get("public_fixture_manifest_sha256")
        != "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
        or preserved.get("verified_no_hand_seal_sha256")
        != "a58ca3f10fd72ba2a7bfc2faf9c8c65b22a22913fcf2c92786859401b8d21c97"
        or preserved.get(
            "production_models_fixtures_thresholds_metrics_seeds_runner_behavior_repair_allowances_and_downstream_gates"
        )
        is not True
        or canceled.get("job_id") != 316158
        or canceled.get("partition") != "a30"
        or canceled.get("state") != "CANCELLED_BEFORE_ALLOCATION"
        or canceled.get("elapsed_seconds") != 0
        or canceled.get("GPU_hours") != 0
        or any(
            canceled.get(key) != 0
            for key in (
                "qualification_attempt_directory_count",
                "wrapper_record_count",
                "full_result_count",
                "compact_result_count",
                "scientific_metric_count",
            )
        )
        or canceled.get("engineering_outcome_opened") is not False
        or eligibility.get("partition") != "h100"
        or eligibility.get("eligible_node_count") != 1
        or eligibility.get("GRES") != "gpu:nvidia_h100_nvl_3g.47gb:1"
        or eligibility.get("real_job_submitted_by_check") is not False
        or eligibility.get("model_or_fixture_inference_executed") is not False
        or eligibility.get("scientific_or_engineering_outcome_opened") is not False
        or topology
        != {
            "partition": "h100",
            "nodes": 1,
            "tasks": 1,
            "GRES": "gpu:nvidia_h100_nvl_3g.47gb:1",
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "process_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "DDP": False,
            "per_submission_wall_minutes_max": 15,
        }
        or resource
        != {
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "DDP": False,
            "per_submission_wall_minutes_max": 15,
            "initial_plus_repair_resmoke_submission_count_max": 3,
            "aggregate_GPU_hours_max": 0.75,
            "new_storage_GiB_max": 10,
            "direct_monetary_cost_USD": 0,
        }
        or value.get("new_engineering_or_scientific_outcome_opened") is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_RESOURCE_REDIRECT_SCHEMA")
    return value


def _engineering_health_resource_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the effective health policy after validating its full lineage."""

    _engineering_health_amendment(cfg)
    redirect = _engineering_health_resource_redirect(cfg)
    _engineering_health_dependency_restore(cfg)
    _engineering_health_topology_guard_repair(cfg)
    if "learner_effective_engineering_health_submission_export_repair" in cfg:
        active = _engineering_health_submission_export_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "partition": active["partition"], "GRES": active["GRES"],
            "GPU_type": active["GPU_type"], "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"], "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active["attempt"],
            "aggregate_GPU_hours_max": active["active_aggregate_GPU_hours_max"],
            "new_storage_GiB_max": active["active_aggregate_new_storage_GiB_max"],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_fixture_bind_repair" in cfg:
        active = _engineering_health_fixture_bind_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "partition": active["partition"], "GRES": active["GRES"],
            "GPU_type": active["GPU_type"], "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"], "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active["attempt"],
            "aggregate_GPU_hours_max": active["active_aggregate_GPU_hours_max"],
            "new_storage_GiB_max": active["active_aggregate_new_storage_GiB_max"],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_portable_AST_repair" in cfg:
        active = _engineering_health_portable_ast_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "partition": active["partition"], "GRES": active["GRES"],
            "GPU_type": active["GPU_type"], "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"], "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active["attempt"],
            "aggregate_GPU_hours_max": active["active_aggregate_GPU_hours_max"],
            "new_storage_GiB_max": active["active_aggregate_new_storage_GiB_max"],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_historical_lineage_repair" in cfg:
        active = _engineering_health_historical_lineage_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "partition": active["partition"],
            "GRES": active["GRES"],
            "GPU_type": active["GPU_type"],
            "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"],
            "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active[
                "attempt"
            ],
            "aggregate_GPU_hours_max": active[
                "active_aggregate_GPU_hours_max"
            ],
            "new_storage_GiB_max": active[
                "active_aggregate_new_storage_GiB_max"
            ],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_git_fallback_repair" in cfg:
        active = _engineering_health_git_fallback_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "partition": active["partition"],
            "GRES": active["GRES"],
            "GPU_type": active["GPU_type"],
            "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"],
            "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active[
                "attempt"
            ],
            "aggregate_GPU_hours_max": active[
                "active_aggregate_GPU_hours_max"
            ],
            "new_storage_GiB_max": active[
                "active_aggregate_new_storage_GiB_max"
            ],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_scheduler_policy" in cfg:
        active = _engineering_health_scheduler_policy(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "partition": active["partition"],
            "GRES": active["GRES"],
            "GPU_type": active["GPU_type"],
            "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"],
            "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active[
                "attempt"
            ],
            "aggregate_GPU_hours_max": active[
                "active_aggregate_GPU_hours_max"
            ],
            "new_storage_GiB_max": active[
                "active_aggregate_new_storage_GiB_max"
            ],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_extended_wall_repair" in cfg:
        active = _engineering_health_extended_wall_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "GPU_type": active["GPU_type"],
            "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"],
            "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active["attempt"],
            "aggregate_GPU_hours_max": active[
                "active_aggregate_GPU_hours_max"
            ],
            "new_storage_GiB_max": active[
                "active_aggregate_new_storage_GiB_max"
            ],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_progress_repair" in cfg:
        active = _engineering_health_progress_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "GPU_type": active["GPU_type"],
            "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"],
            "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active["attempt"],
            "aggregate_GPU_hours_max": active[
                "active_aggregate_GPU_hours_max"
            ],
            "new_storage_GiB_max": active[
                "active_aggregate_new_storage_GiB_max"
            ],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_iterative_reauthorization" in cfg:
        active = _engineering_health_iterative_reauthorization(cfg)[
            "active_attempt_resource_policy"
        ]
        return {
            "GPU_type": active["GPU_type"],
            "GPU_count": active["GPU_count"],
            "CPU_count": active["CPU_count"],
            "memory_GiB": active["memory_GiB"],
            "DDP": active["DDP"],
            "per_submission_wall_minutes_max": active["wall_minutes_max"],
            "initial_plus_repair_resmoke_submission_count_max": active["attempt"],
            "aggregate_GPU_hours_max": active[
                "active_aggregate_GPU_hours_max"
            ],
            "new_storage_GiB_max": active[
                "active_aggregate_new_storage_GiB_max"
            ],
            "direct_monetary_cost_USD": active["direct_monetary_cost_USD"],
        }
    if "learner_effective_engineering_health_parser_repair_reauthorization" in cfg:
        return _engineering_health_parser_repair_reauthorization(cfg)[
            "effective_resource_policy"
        ]
    if "learner_effective_engineering_health_reauthorization" in cfg:
        return _engineering_health_reauthorization(cfg)[
            "effective_resource_policy"
        ]
    return redirect["bounded_resource_policy"]


def _engineering_health_attempt_gpu_type(
    cfg: dict[str, Any], attempt: int
) -> str:
    """Return the prospectively bound GPU type for one historical/current attempt."""

    if type(attempt) is not int or attempt < 1:
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    if "learner_effective_engineering_health_submission_export_repair" in cfg:
        policy = _engineering_health_submission_export_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        if attempt == int(policy["attempt"]):
            return str(policy["GPU_type"])
        historical = policy["historical_attempt_GPU_types"]
        if str(attempt) in historical:
            return str(historical[str(attempt)])
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    if "learner_effective_engineering_health_fixture_bind_repair" in cfg:
        policy = _engineering_health_fixture_bind_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        if attempt == int(policy["attempt"]):
            return str(policy["GPU_type"])
        historical = policy["historical_attempt_GPU_types"]
        if str(attempt) in historical:
            return str(historical[str(attempt)])
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    if "learner_effective_engineering_health_portable_AST_repair" in cfg:
        policy = _engineering_health_portable_ast_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        if attempt == int(policy["attempt"]):
            return str(policy["GPU_type"])
        historical = policy["historical_attempt_GPU_types"]
        if str(attempt) in historical:
            return str(historical[str(attempt)])
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    if "learner_effective_engineering_health_historical_lineage_repair" in cfg:
        policy = _engineering_health_historical_lineage_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        if attempt == int(policy["attempt"]):
            return str(policy["GPU_type"])
        historical = policy["historical_attempt_GPU_types"]
        if str(attempt) in historical:
            return str(historical[str(attempt)])
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    if "learner_effective_engineering_health_git_fallback_repair" in cfg:
        policy = _engineering_health_git_fallback_repair(cfg)[
            "active_attempt_resource_policy"
        ]
        if attempt == int(policy["attempt"]):
            return str(policy["GPU_type"])
        historical = policy["historical_attempt_GPU_types"]
        if str(attempt) in historical:
            return str(historical[str(attempt)])
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    if "learner_effective_engineering_health_scheduler_policy" in cfg:
        policy = _engineering_health_scheduler_policy(cfg)[
            "active_attempt_resource_policy"
        ]
        if attempt == int(policy["attempt"]):
            return str(policy["GPU_type"])
        historical = policy["historical_attempt_GPU_types"]
        if str(attempt) in historical:
            return str(historical[str(attempt)])
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    return str(_engineering_health_resource_policy(cfg)["GPU_type"])


def _engineering_health_dependency_restore(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the outcome-independent public cache restoration after attempt 1."""

    try:
        value = cfg["learner_effective_engineering_health_dependency_restore"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_DEPENDENCY_RESTORE_NOT_FROZEN") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("repair_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    trigger = value.get("triggering_attempt", {})
    execution = value.get("public_restore_execution", {})
    archive = value.get("active_language_dependency_archive", {})
    provenance = value.get("restored_public_dependency_provenance", {})
    budget = value.get("remaining_health_budget", {})
    expected_distributions = [
        ["accelerate", "1.14.0"],
        ["diffusers", "0.39.0"],
        ["huggingface_hub", "0.36.2"],
        ["imageio_ffmpeg", "0.6.0"],
        ["llvmlite", "0.44.0"],
        ["more_itertools", "10.7.0"],
        ["nltk", "3.9.1"],
        ["numba", "0.61.2"],
        ["numpy", "2.2.6"],
        ["openai_whisper", "20250625"],
        ["protobuf", "7.35.1"],
        ["safetensors", "0.8.0"],
        ["sentencepiece", "0.2.2"],
        ["tiktoken", "0.11.0"],
        ["tokenizers", "0.22.2"],
        ["transformers", "4.57.6"],
    ]
    if (
        cfg.get("schema_version") not in {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_H100_ATTEMPT_1_PREINFERENCE_DEPENDENCY_CACHE_MISS_BEFORE_ATTEMPT_2_OR_NEW_OUTCOME"
        or value.get("scope")
        != "OUTCOME_INDEPENDENT_PUBLIC_DEPENDENCY_CACHE_RESTORATION_ONLY"
        or expected != ENGINEERING_HEALTH_DEPENDENCY_RESTORE_SHA256
        or digest(payload) != expected
        or value.get("repair_commitment_scope")
        != "canonical JSON of this repair excluding repair_commitment_sha256"
        or preserved.get("engineering_health_amendment_sha256")
        != ENGINEERING_HEALTH_AMENDMENT_SHA256
        or preserved.get("H100_resource_redirect_sha256")
        != ENGINEERING_HEALTH_RESOURCE_REDIRECT_SHA256
        or preserved.get("historical_language_archive_sha256")
        != "27df87f4ec1900b4a11f307d42a18483903d38ddb9ed77f418b88fa299497e37"
        or preserved.get("historical_language_archive_bytes") != 542423040
        or preserved.get("prior_public_development_no_go_sha256")
        != "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
        or preserved.get("public_fixture_manifest_sha256")
        != "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
        or preserved.get(
            "production_models_fixtures_thresholds_metrics_seeds_module_behavior_scientific_gates_and_downstream_contract"
        )
        is not True
        or trigger.get("attempt") != 1
        or trigger.get("job_id") != 316325
        or trigger.get("state") != "FAILED"
        or trigger.get("exit_code") != "65:0"
        or trigger.get("elapsed_seconds") != 1
        or trigger.get("conservative_protocol_GPU_hours_charge") != 0.25
        or trigger.get("wrapper_record_count") != 1
        or any(
            trigger.get(key) != 0
            for key in (
                "full_result_count",
                "compact_result_count",
                "model_module_inference_count",
                "scientific_metric_count",
            )
        )
        or trigger.get("classification")
        != "ENGINEERING_DEPENDENCY_CACHE_MISS_NOT_SCIENTIFIC_NO_GO"
        or trigger.get("partial_scientific_or_engineering_metric_opened") is not False
        or execution.get("base_dependency_job", {}).get("job_id") != 316333
        or execution.get("base_dependency_job", {}).get("state") != "COMPLETED"
        or execution.get("base_dependency_job", {}).get("GPU_count") != 0
        or execution.get("language_dependency_job", {}).get("job_id") != 316335
        or execution.get("language_dependency_job", {}).get("state") != "COMPLETED"
        or execution.get("language_dependency_job", {}).get("GPU_count") != 0
        or execution.get("public_dependencies_only") is not True
        or execution.get("restricted_mount_present") is not False
        or execution.get("offline_local_files_only_reload_after_preparation")
        != "PASS"
        or execution.get("direct_monetary_cost_USD") != 0
        or archive.get("sha256")
        != "97ef52ecaa8c99db017e598d8a63d0d2170affef14ef46e7df7a656abd3a1a07"
        or archive.get("bytes") != 542423040
        or archive.get("normalized_regular_file_count") != 11512
        or archive.get("normalized_regular_file_bytes") != 532618028
        or archive.get("normalized_tree_commitment_sha256")
        != "34014b16541c10bc3eccfbdfa18255e346be2c36cf70bb3defd0c9b04f2d07af"
        or provenance.get("base_dependency_archive_sha256")
        != "5da2b13ef7ad018e2c85ccdcddce9f2f1860ee631b8e425c9d87331599781506"
        or provenance.get("base_dependency_archive_bytes") != 201246720
        or provenance.get("Whisper_small_sha256")
        != "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794"
        or provenance.get("OPUS_MT_de_en_revision")
        != "1a922f3b32a8e809e17a47d4b32142d8105924e5"
        or provenance.get("OPUS_MT_file_count") != 34
        or provenance.get("NLTK_package_families")
        != ["averaged_perceptron_tagger_eng", "omw-1.4", "wordnet"]
        or provenance.get("installed_distributions") != expected_distributions
        or provenance.get("restored_storage_file_count") != 65
        or provenance.get("restored_storage_bytes") != 2449082735
        or provenance.get("restored_storage_within_10_GiB_ceiling") is not True
        or budget.get("submission_count_consumed") != 1
        or budget.get("submission_count_remaining") != 2
        or budget.get("conservative_protocol_GPU_hours_charged") != 0.25
        or budget.get("conservative_protocol_GPU_hours_remaining") != 0.5
        or budget.get("new_storage_GiB_ceiling") != 10
        or budget.get("direct_monetary_cost_USD") != 0
        or value.get("new_model_fixture_scientific_or_learner_outcome_opened")
        is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_DEPENDENCY_RESTORE_COMMITMENT")
    return value


def _engineering_health_topology_guard_repair(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the final outcome-independent topology-guard wiring repair."""

    try:
        value = cfg["learner_effective_engineering_health_topology_guard_repair"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_TOPOLOGY_GUARD_REPAIR_NOT_FROZEN") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("repair_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    trigger = value.get("triggering_attempt", {})
    diagnosis = value.get("aggregate_read_only_diagnosis", {})
    repair = value.get("repair", {})
    budget = value.get("remaining_health_budget", {})
    if (
        cfg.get("schema_version") not in {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_H100_ATTEMPT_2_REDUNDANT_TOPOLOGY_GUARD_FAILURE_BEFORE_ATTEMPT_3_OR_NEW_OUTCOME"
        or value.get("scope")
        != "OUTCOME_INDEPENDENT_EXACT_SLURM_TOPOLOGY_VALIDATION_WIRING_REPAIR_ONLY"
        or expected != ENGINEERING_HEALTH_TOPOLOGY_GUARD_REPAIR_SHA256
        or digest(payload) != expected
        or value.get("repair_commitment_scope")
        != "canonical JSON of this repair excluding repair_commitment_sha256"
        or preserved.get("engineering_health_amendment_sha256")
        != ENGINEERING_HEALTH_AMENDMENT_SHA256
        or preserved.get("H100_resource_redirect_sha256")
        != ENGINEERING_HEALTH_RESOURCE_REDIRECT_SHA256
        or preserved.get("dependency_restore_sha256")
        != ENGINEERING_HEALTH_DEPENDENCY_RESTORE_SHA256
        or preserved.get("prior_public_development_no_go_sha256")
        != "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
        or preserved.get("public_fixture_manifest_sha256")
        != "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
        or preserved.get(
            "production_models_fixtures_thresholds_metrics_seeds_module_behavior_scientific_gates_and_downstream_contract"
        )
        is not True
        or trigger.get("attempt") != 2
        or trigger.get("job_id") != 316353
        or trigger.get("state") != "FAILED"
        or trigger.get("exit_code") != "66:0"
        or trigger.get("elapsed_seconds") != 5
        or trigger.get("conservative_protocol_GPU_hours_charge") != 0.25
        or trigger.get("wrapper_record_count") != 1
        or any(
            trigger.get(key) != 0
            for key in (
                "full_result_count",
                "compact_result_count",
                "private_trace_count",
                "stdout_bytes",
                "stderr_bytes",
                "model_module_inference_count",
                "scientific_metric_count",
            )
        )
        or trigger.get("classification")
        != "ENGINEERING_WRAPPER_TOPOLOGY_GUARD_FAILURE_NOT_SCIENTIFIC_NO_GO"
        or trigger.get("partial_scientific_or_engineering_metric_opened") is not False
        or diagnosis.get("authoritative_scontrol_record_present") is not True
        or diagnosis.get("authoritative_scontrol_predicate_count") != 7
        or diagnosis.get("authoritative_scontrol_predicate_pass_count") != 7
        or diagnosis.get("scheduler_requested_and_allocated_TRES_match") is not True
        or diagnosis.get("scientific_or_model_information_used") is not False
        or repair.get("scientific_A30_topology_path_changed") is not False
        or repair.get("health_resource_topology_changed") is not False
        or repair.get("model_fixture_threshold_metric_seed_or_gate_changed") is not False
        or repair.get("full_28_case_suite_restart_required") is not True
        or budget.get("submission_count_consumed") != 2
        or budget.get("submission_count_remaining") != 1
        or budget.get("conservative_protocol_GPU_hours_charged") != 0.5
        or budget.get("conservative_protocol_GPU_hours_remaining") != 0.25
        or budget.get("new_storage_GiB_ceiling") != 10
        or budget.get("direct_monetary_cost_USD") != 0
        or value.get("new_model_fixture_scientific_or_learner_outcome_opened")
        is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_TOPOLOGY_GUARD_REPAIR_COMMITMENT")
    return value


def _engineering_health_terminal_result(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the sealed final blocker and prevent a fourth health route."""

    try:
        value = cfg["learner_effective_engineering_health_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_TERMINAL_RESULT_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    attempt = value.get("final_attempt", {})
    compact = value.get("compact_aggregate", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    gate = value.get("terminal_gate", {})
    if (
        cfg.get("schema_version") not in {22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "ENGINEERING_BLOCKER_ROUTE_EXHAUSTED_BEFORE_MODEL_INFERENCE_NO_SCIENTIFIC_METRICS_OPENED"
        or expected != ENGINEERING_HEALTH_BLOCKER_SHA256
        or digest(payload) != expected
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this terminal result excluding blocker_commitment_sha256"
        or preserved.get("engineering_health_amendment_sha256")
        != ENGINEERING_HEALTH_AMENDMENT_SHA256
        or preserved.get("H100_resource_redirect_sha256")
        != ENGINEERING_HEALTH_RESOURCE_REDIRECT_SHA256
        or preserved.get("dependency_restore_sha256")
        != ENGINEERING_HEALTH_DEPENDENCY_RESTORE_SHA256
        or preserved.get("topology_guard_repair_sha256")
        != ENGINEERING_HEALTH_TOPOLOGY_GUARD_REPAIR_SHA256
        or attempt.get("attempt") != 3
        or attempt.get("job_id") != 316370
        or attempt.get("scheduler_state") != "COMPLETED"
        or attempt.get("scheduler_exit_code") != "0:0"
        or attempt.get("compact_stdout_record_count") != 1
        or attempt.get("compact_stdout_shape_valid") is not True
        or attempt.get("compact_stdout_hash_fields_valid") is not True
        or attempt.get("stderr_bytes") != 0
        or attempt.get("private_trace_count") != 7
        or attempt.get("model_module_inference_count") != 0
        or attempt.get("scientific_metric_count") != 0
        or compact.get("status") != "ENGINEERING_BLOCKER"
        or compact.get("attempt") != 3
        or compact.get("module_count") != 7
        or compact.get("completed_module_count") != 0
        or compact.get("failed_module_count") != 7
        or compact.get("case_count") != 28
        or compact.get("scientific_metric_count") != 0
        or compact.get("unaccounted_failure_count") != 1
        or compact.get("engineering_health_commitment_sha256")
        != "107f5d0dd58adda31e9932a18a41319d5d6317fafd9b769e36ad0d13650c3696"
        or diagnosis.get("declared_preflight_blocked_trace_count") != 6
        or diagnosis.get("unaccounted_exception_trace_count") != 1
        or diagnosis.get("unaccounted_exception_type") != "FileNotFoundError"
        or diagnosis.get("partial_scientific_metric_opened") is not False
        or resource.get("submission_count_used") != 3
        or resource.get("submission_count_remaining") != 0
        or resource.get("direct_monetary_cost_USD") != 0
        or gate.get("engineering_health_pass") is not False
        or gate.get("attempt_4_authorized") is not False
        or any(
            gate.get(key) is not False
            for key in (
                "public_development_authorized",
                "public_holdout_authorized",
                "governed_C_authorized",
                "LTX_preflight_or_generation_authorized",
                "synthetic_learner_authorized",
            )
        )
    ):
        raise RuntimeError("E_TUPLE_HEALTH_TERMINAL_RESULT_COMMITMENT")
    return value


def _engineering_health_reauthorization(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the one-submission post-blocker topology-attestation route."""

    _engineering_health_terminal_result(cfg)
    try:
        value = cfg["learner_effective_engineering_health_reauthorization"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_REAUTHORIZATION_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("reauthorization_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    repair = value.get("failure_specific_repair", {})
    attestation = repair.get("topology_attestation", {})
    resource = value.get("effective_resource_policy", {})
    execution = value.get("execution_and_stop_rule", {})
    if (
        cfg.get("schema_version") not in {23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_SEALED_BLOCKER_BEFORE_REAUTHORIZED_ATTEMPT_4_OR_NEW_SCIENTIFIC_OUTCOME"
        or value.get("scope")
        != "ONE_SUBMISSION_WRAPPER_TO_CONTAINER_TOPOLOGY_ATTESTATION_REPAIR_ONLY"
        or expected != ENGINEERING_HEALTH_REAUTHORIZATION_SHA256
        or digest(payload) != expected
        or value.get("reauthorization_commitment_scope")
        != "canonical JSON of this amendment excluding reauthorization_commitment_sha256"
        or preserved.get("sealed_engineering_blocker_sha256")
        != ENGINEERING_HEALTH_BLOCKER_SHA256
        or preserved.get("attempt_3_engineering_health_commitment_sha256")
        != "107f5d0dd58adda31e9932a18a41319d5d6317fafd9b769e36ad0d13650c3696"
        or preserved.get("prior_public_development_no_go_sha256")
        != "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
        or preserved.get("public_fixture_manifest_sha256")
        != "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
        or preserved.get(
            "models_weights_fixtures_preprocessing_thresholds_metrics_seeds_module_behavior_scientific_gates_and_downstream_contract"
        )
        is not True
        or repair.get("runner_scontrol_subprocess_inside_container") is not False
        or repair.get("same_complete_28_case_two_replicate_production_paths")
        is not True
        or repair.get("mocks_or_shallow_load_path") is not False
        or repair.get("scientific_metric_count") != 0
        or repair.get("holdout_input_count") != 0
        or repair.get("model_fixture_threshold_metric_seed_or_gate_changed")
        is not False
        or attestation.get("schema_version") != 1
        or attestation.get("raw_scontrol_record_retained_or_exposed") is not False
        or attestation.get("exact_fields")
        != [
            "schema_version",
            "attempt",
            "job_id",
            "partition",
            "node_count",
            "CPU_count",
            "task_count",
            "time_limit_minutes",
            "memory_per_CPU_GiB",
            "GRES",
            "predicate_count",
            "predicate_pass_count",
            "world_size",
            "local_world_size",
            "source",
        ]
        or attestation.get("expected_values")
        != {
            "attempt": 4,
            "partition": "h100",
            "node_count": 1,
            "CPU_count": 8,
            "task_count": 1,
            "time_limit_minutes": 15,
            "memory_per_CPU_GiB": 4,
            "GRES": "gpu:nvidia_h100_nvl_3g.47gb:1",
            "predicate_count": 7,
            "predicate_pass_count": 7,
            "world_size": 1,
            "local_world_size": 1,
            "source": "WRAPPER_SCONTROL_BEFORE_CONTAINER",
        }
        or resource
        != {
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "DDP": False,
            "per_submission_wall_minutes_max": 15,
            "initial_plus_repair_resmoke_submission_count_max": 4,
            "reauthorized_attempt": 4,
            "reauthorized_submission_count": 1,
            "prior_protocol_accounted_GPU_hours": 0.504019171529346,
            "reauthorized_GPU_hours_max": 0.25,
            "aggregate_GPU_hours_max": 0.754019171529346,
            "prior_health_run_storage_GiB": 1.4025717973709106e-06,
            "reauthorized_new_storage_GiB_max": 1.0,
            "new_storage_GiB_max": 1.0000014025717974,
            "direct_monetary_cost_USD": 0,
        }
        or execution.get("prospective_commit_and_push_required_before_submission")
        is not True
        or execution.get("attempt_ordinal") != 4
        or execution.get("submission_count") != 1
        or execution.get("repair_or_resmoke_cycles_after_attempt_4") != 0
        or execution.get("complete_suite_restart_required") is not True
        or execution.get("metric_withholding_unchanged") is not True
        or value.get("new_health_or_scientific_outcome_opened") is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_REAUTHORIZATION_COMMITMENT")
    return value


def _engineering_health_reauthorization_result(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the terminal result of the sole post-blocker submission."""

    reauthorization = _engineering_health_reauthorization(cfg)
    try:
        value = cfg["learner_effective_engineering_health_reauthorization_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_REAUTHORIZATION_RESULT_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    submission = value.get("submission_provenance", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    gate = value.get("terminal_gate", {})
    if (
        cfg.get("schema_version") not in {24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "ENGINEERING_BLOCKER_REAUTHORIZED_ATTEMPT_4_EXHAUSTED_BEFORE_RUNNER_ENTRY_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("route_id") != "construct-aligned-engineering-health"
        or expected != ENGINEERING_HEALTH_REAUTHORIZATION_BLOCKER_SHA256
        or digest(payload) != expected
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this terminal result excluding blocker_commitment_sha256"
        or preserved.get("reauthorization_commitment_sha256")
        != reauthorization["reauthorization_commitment_sha256"]
        or preserved.get("sealed_attempt_3_blocker_sha256")
        != ENGINEERING_HEALTH_BLOCKER_SHA256
        or preserved.get("prior_public_development_no_go_sha256")
        != "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
        or preserved.get("public_fixture_manifest_sha256")
        != "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
        or preserved.get(
            "models_weights_fixtures_preprocessing_thresholds_metrics_seeds_scientific_gates_and_downstream_contract"
        )
        is not True
        or submission.get("rejected_pre_submission_command_count") != 1
        or submission.get("rejected_pre_submission_job_count") != 0
        or submission.get("rejected_pre_submission_GPU_hours") != 0
        or submission.get("accepted_submission_count") != 1
        or submission.get("attempt") != 4
        or submission.get("job_id") != 316478
        or submission.get("scheduler_state") != "FAILED"
        or submission.get("scheduler_exit_code") != "2:0"
        or submission.get("scheduler_elapsed_seconds") != 15
        or submission.get("partition") != "h100"
        or submission.get("GRES") != "gpu:nvidia_h100_nvl_3g.47gb:1"
        or submission.get("GPU_type") != "NVIDIA_H100_NVL_3G_47GB_MIG"
        or submission.get("GPU_count") != 1
        or submission.get("CPU_count") != 8
        or submission.get("memory_GiB") != 32
        or submission.get("authoritative_topology_predicate_count") != 7
        or submission.get("authoritative_topology_predicate_pass_count") != 7
        or submission.get("wrapper_marker_count") != 1
        or submission.get("topology_attestation_count") != 1
        or submission.get("full_result_count") != 0
        or submission.get("private_trace_count") != 0
        or submission.get("stderr_line_count") != 8
        or submission.get("stderr_bytes") != 612
        or submission.get("stderr_sha256")
        != "c87246b4200baf4510c455b143910bbcf20089600e8bfa44d2f97ed7a86e884e"
        or submission.get("runner_source_sha256")
        != "24f2c5bba4015febe036b9ec1d5e614d413643d62483047dbce3f2af6b7f8146"
        or submission.get("prospective_config_file_sha256")
        != "9d5a1437cae7079225cd78d1073d0a5cc3f68cfa5e5bdb62034ce0713d354361"
        or submission.get("prospective_config_commitment_sha256")
        != "ff2275e3f68ad2d5ad69f790e141a5d6a2f50fc077fe7fc4ca42374bd132ce7f"
        or submission.get("wrapper_source_sha256")
        != "cc3794352ab95b22d647df1401e5116815b283a500ed5254afa548dc713ad54a"
        or submission.get("attempt_root_file_count") != 2
        or submission.get("model_module_inference_count") != 0
        or submission.get("scientific_metric_count") != 0
        or diagnosis.get("traceback_present") is not False
        or diagnosis.get("argparse_error_present") is not True
        or diagnosis.get("argparse_invalid_choice") is not True
        or diagnosis.get("rejected_argument") != "attempt_4"
        or diagnosis.get("committed_parser_choices") != [1, 2, 3]
        or diagnosis.get("failure_stage")
        != "CLI_ARGUMENT_VALIDATION_BEFORE_RUNNER_ENTRY_DEPENDENCY_PREFLIGHT_FIXTURE_PROJECTION_MODEL_LOADING_OR_MODULE_INFERENCE"
        or diagnosis.get("classification")
        != "ENGINEERING_CLI_CONFIGURATION_MISMATCH_ROUTE_EXHAUSTED_NOT_SCIENTIFIC_NO_GO"
        or diagnosis.get("partial_health_or_scientific_metric_opened") is not False
        or resource.get("accepted_submission_count") != 1
        or resource.get("submission_count_remaining") != 0
        or resource.get("attempt_4_conservative_protocol_GPU_hours_charged")
        != 0.25
        or resource.get("protocol_accounted_cumulative_GPU_hours")
        != 0.754019171529346
        or resource.get("aggregate_GPU_hours_ceiling") != 0.754019171529346
        or resource.get("attempt_4_new_storage_bytes") != 1092
        or resource.get("direct_monetary_cost_USD") != 0
        or gate.get("engineering_health_pass") is not False
        or gate.get("attempt_5_authorized") is not False
        or gate.get("decision")
        != "STOP_AT_EXACT_ENGINEERING_BLOCKER_WITH_ALL_SCIENTIFIC_METRICS_WITHHELD"
        or any(
            gate.get(key) is not False
            for key in (
                "public_development_authorized",
                "public_holdout_authorized",
                "governed_C_authorized",
                "LTX_preflight_or_generation_authorized",
                "synthetic_learner_authorized",
            )
        )
    ):
        raise RuntimeError("E_TUPLE_HEALTH_REAUTHORIZATION_RESULT_COMMITMENT")
    return value


def _engineering_health_parser_repair_reauthorization(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the sole post-attempt-4 parser-bound repair route."""

    terminal = _engineering_health_reauthorization_result(cfg)
    try:
        value = cfg[
            "learner_effective_engineering_health_parser_repair_reauthorization"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "E_TUPLE_HEALTH_PARSER_REPAIR_REAUTHORIZATION_MISSING"
        ) from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("reauthorization_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    repair = value.get("failure_specific_repair", {})
    resource = value.get("effective_resource_policy", {})
    execution = value.get("execution_and_stop_rule", {})
    if (
        cfg.get("schema_version") not in {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_ATTEMPT_4_BLOCKER_BEFORE_SINGLE_POST_BLOCKER_ATTEMPT_5_OR_NEW_SCIENTIFIC_OUTCOME"
        or value.get("scope") != "ONE_SUBMISSION_CLI_ATTEMPT_BOUND_REPAIR_ONLY"
        or expected != ENGINEERING_HEALTH_PARSER_REPAIR_REAUTHORIZATION_SHA256
        or digest(payload) != expected
        or value.get("reauthorization_commitment_scope")
        != "canonical JSON of this amendment excluding reauthorization_commitment_sha256"
        or preserved.get("attempt_3_blocker_sha256")
        != ENGINEERING_HEALTH_BLOCKER_SHA256
        or preserved.get("attempt_4_reauthorization_sha256")
        != ENGINEERING_HEALTH_REAUTHORIZATION_SHA256
        or preserved.get("attempt_4_blocker_sha256")
        != terminal["blocker_commitment_sha256"]
        or preserved.get("prior_public_development_no_go_sha256")
        != "4b7cd58345757ed0a51dfcdddf6641954e5e55269bf9ed64ca2385ccd2ec66bf"
        or preserved.get("public_fixture_manifest_sha256")
        != "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
        or preserved.get(
            "models_weights_fixtures_preprocessing_thresholds_metrics_seeds_module_behavior_scientific_gates_and_downstream_contract"
        )
        is not True
        or repair.get("old_parser_choices") != [1, 2, 3]
        or repair.get("new_parser_choices") != [1, 2, 3, 5]
        or repair.get("attempt_4_remains_rejected_and_sealed") is not True
        or repair.get("wrapper_health_attempt") != 5
        or repair.get("runner_authorized_attempt") != 5
        or repair.get("same_complete_28_case_two_replicate_production_paths")
        is not True
        or repair.get("same_compact_wrapper_topology_attestation") is not True
        or repair.get("runner_scontrol_subprocess_inside_container") is not False
        or repair.get("mocks_or_shallow_load_path") is not False
        or repair.get("scientific_metric_count") != 0
        or repair.get("holdout_input_count") != 0
        or repair.get("model_fixture_threshold_metric_seed_or_gate_changed")
        is not False
        or resource
        != {
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "DDP": False,
            "per_submission_wall_minutes_max": 15,
            "initial_plus_repair_resmoke_submission_count_max": 5,
            "reauthorized_attempt": 5,
            "reauthorized_submission_count": 1,
            "prior_protocol_accounted_GPU_hours": 0.754019171529346,
            "reauthorized_GPU_hours_max": 0.25,
            "aggregate_GPU_hours_max": 1.004019171529346,
            "prior_health_run_storage_GiB": 2.419576048851013e-06,
            "reauthorized_new_storage_GiB_max": 1.0,
            "new_storage_GiB_max": 1.0000024195760489,
            "direct_monetary_cost_USD": 0,
        }
        or execution.get("prospective_commit_and_push_required_before_submission")
        is not True
        or execution.get("attempt_ordinal") != 5
        or execution.get("submission_count") != 1
        or execution.get("repair_or_resmoke_cycles_after_attempt_5") != 0
        or execution.get("complete_suite_restart_required") is not True
        or execution.get("metric_withholding_unchanged") is not True
        or value.get("new_health_or_scientific_outcome_opened") is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PARSER_REPAIR_REAUTHORIZATION_COMMITMENT")
    return value


def _engineering_health_parser_repair_result(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the sealed terminal result from the sole post-blocker attempt."""

    reauthorization = _engineering_health_parser_repair_reauthorization(cfg)
    try:
        value = cfg["learner_effective_engineering_health_parser_repair_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "E_TUPLE_HEALTH_PARSER_REPAIR_RESULT_MISSING"
        ) from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    submission = value.get("submission_provenance", {})
    compact = value.get("compact_aggregate", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    gate = value.get("terminal_gate", {})
    if (
        cfg.get("schema_version") not in {26, 27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "ENGINEERING_BLOCKER_POST_BLOCKER_ATTEMPT_5_EXHAUSTED_BEFORE_MODEL_LOADING_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("route_id") != "construct-aligned-engineering-health"
        or expected != ENGINEERING_HEALTH_PARSER_REPAIR_BLOCKER_SHA256
        or digest(payload) != expected
        or value.get("parser_repair_reauthorization_sha256")
        != reauthorization["reauthorization_commitment_sha256"]
        or value.get("preserved_prior_blockers_sha256")
        != [
            ENGINEERING_HEALTH_BLOCKER_SHA256,
            ENGINEERING_HEALTH_REAUTHORIZATION_BLOCKER_SHA256,
        ]
        or submission.get("job_id") != 316537
        or submission.get("scheduler_state") != "COMPLETED"
        or submission.get("scheduler_exit_code") != "0:0"
        or submission.get("scheduler_elapsed_seconds") != 33
        or submission.get("attempt") != 5
        or submission.get("GPU_type") != "NVIDIA_H100_NVL_3G_47GB_MIG"
        or submission.get("GPU_count") != 1
        or submission.get("CPU_count") != 8
        or submission.get("memory_GiB") != 32
        or submission.get("wall_minutes_max") != 15
        or submission.get("DDP") is not False
        or submission.get("topology_predicate_count") != 7
        or submission.get("topology_predicate_pass_count") != 7
        or any(
            submission.get(key) != count
            for key, count in (
                ("wrapper_marker_count", 1),
                ("topology_attestation_count", 1),
                ("full_result_count", 1),
                ("private_trace_count", 7),
                ("compact_stdout_bytes", 1368),
                ("stderr_bytes", 0),
            )
        )
        or submission.get("runner_commitment_sha256")
        != "7aa4df21cf48e27cfdd2ab53cb91ad942c9cf3472581281426da0cb13bc04286"
        or submission.get("config_commitment_sha256")
        != "e97947d7d37784251298900306fa0f6d2983bcd748f27f9815685c82169bcaac"
        or submission.get("dependency_config_commitment_sha256")
        != "558fe40905fd30baa06acbd0b01c9d1a82d859a70ef49884c0366d7702a1747d"
        or submission.get("microfixture_manifest_commitment_sha256")
        != "981e1cf51ba9f538976aaf6cb2a806113f54656fe3a2dc4cc0095841bb6580ee"
        or submission.get("engineering_health_commitment_sha256")
        != "edd0d53f1fadc00a962312f285ca3f514ab2e4e76494eccbbdab8577e9c59c04"
        or submission.get("full_result_commitment_valid") is not True
        or submission.get("runner_and_config_commitments_match_deployed")
        is not True
        or compact
        != {
            "status": "ENGINEERING_BLOCKER",
            "attempt": 5,
            "module_count": 7,
            "completed_module_count": 0,
            "failed_module_count": 7,
            "case_count": 28,
            "holdout_input_count": 0,
            "scientific_metric_count": 0,
            "failure_count": 7,
            "invalid_retained_record_count": 0,
            "silent_truncation_count": 0,
            "external_call_count": 0,
            "unaccounted_failure_count": 0,
            "public_fixture_manifest_commitment_sha256": "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6",
            "network_disabled": True,
            "telemetry_disabled": True,
            "restricted_mount_present": False,
        }
        or diagnosis.get("first_stable_error_code")
        != "E_TUPLE_HEALTH_ARTIFACT_COMMITMENT"
        or diagnosis.get("first_exception_type") != "RuntimeError"
        or diagnosis.get("preflight_blocked_module_count") != 6
        or diagnosis.get("failure_stage")
        != "PRODUCTION_DEPENDENCY_PREFLIGHT_BASE_CONTAINER_IMMUTABLE_FILE_VERIFICATION_BEFORE_MODEL_LOADING"
        or diagnosis.get("artifact_family") != "BASE_CONTAINER"
        or diagnosis.get("execution_time_artifact_commitment_check_pass")
        is not False
        or diagnosis.get("post_run_host_artifact_present") is not True
        or diagnosis.get("post_run_host_artifact_sha256_matches_frozen")
        is not True
        or diagnosis.get("discrepancy_mechanism")
        != "UNRESOLVED_WITHIN_FROZEN_ROUTE"
        or diagnosis.get("classification")
        != "ENGINEERING_BASE_CONTAINER_COMMITMENT_PREFLIGHT_FAILURE_ROUTE_EXHAUSTED_NOT_SCIENTIFIC_NO_GO"
        or diagnosis.get("model_loading_started") is not False
        or diagnosis.get("model_module_inference_count") != 0
        or diagnosis.get("partial_health_or_scientific_metric_opened") is not False
        or diagnosis.get("raw_trace_text_released") is not False
        or resource.get("accepted_submission_count") != 1
        or resource.get("submission_count_remaining") != 0
        or resource.get("attempt_5_scheduler_elapsed_seconds") != 33
        or resource.get("protocol_accounted_cumulative_GPU_hours")
        != 0.7600439148479038
        or resource.get("aggregate_GPU_hours_ceiling") != 1.004019171529346
        or resource.get("scheduler_actual_elapsed_seconds_all_H100_attempts")
        != 74
        or resource.get("direct_monetary_cost_USD") != 0
        or gate
        != {
            "engineering_health_pass": False,
            "public_development_authorized": False,
            "public_holdout_authorized": False,
            "governed_C_authorized": False,
            "LTX_preflight_or_generation_authorized": False,
            "synthetic_learner_authorized": False,
            "attempt_6_authorized": False,
            "decision": "STOP_AT_EXACT_ENGINEERING_BLOCKER_WITH_ALL_SCIENTIFIC_METRICS_WITHHELD",
        }
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this terminal result excluding blocker_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PARSER_REPAIR_RESULT_COMMITMENT")
    return value


def _engineering_health_iterative_reauthorization(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the user-authorized rolling engineering repair route."""

    terminal = _engineering_health_parser_repair_result(cfg)
    try:
        value = cfg[
            "learner_effective_engineering_health_iterative_reauthorization"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "E_TUPLE_HEALTH_ITERATIVE_REAUTHORIZATION_MISSING"
        ) from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("reauthorization_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    diagnosis = value.get("read_only_root_cause_diagnosis", {})
    repair = value.get("failure_specific_repair", {})
    resource = value.get("active_attempt_resource_policy", {})
    rolling = value.get("rolling_execution_policy", {})
    if (
        cfg.get("schema_version") not in {27, 28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_ATTEMPT_5_BLOCKER_BEFORE_HOST_CONTAINER_ATTESTATION_REPAIR_ATTEMPT_6_OR_NEW_OUTCOME"
        or value.get("scope")
        != "HOST_CONTAINER_ATTESTATION_REPAIR_WITH_ROLLING_ORDINARY_ENGINEERING_REPAIR_AUTHORIZATION"
        or expected != ENGINEERING_HEALTH_ITERATIVE_REAUTHORIZATION_SHA256
        or digest(payload) != expected
        or value.get("reauthorization_commitment_scope")
        != "canonical JSON of this amendment excluding reauthorization_commitment_sha256"
        or preserved.get("attempt_3_blocker_sha256")
        != ENGINEERING_HEALTH_BLOCKER_SHA256
        or preserved.get("attempt_4_blocker_sha256")
        != ENGINEERING_HEALTH_REAUTHORIZATION_BLOCKER_SHA256
        or preserved.get("attempt_5_blocker_sha256")
        != terminal["blocker_commitment_sha256"]
        or preserved.get("public_fixture_manifest_sha256")
        != "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
        or preserved.get(
            "models_weights_fixtures_preprocessing_thresholds_metrics_seeds_module_behavior_scientific_gates_and_downstream_contract"
        )
        is not True
        or diagnosis
        != {
            "artifact_family": "BASE_CONTAINER",
            "host_entry_is_symlink": True,
            "host_symlink_target_present": True,
            "host_target_sha256_matches_frozen": True,
            "host_target_bytes": 3731320832,
            "running_container_namespace_symlink_target_present": False,
            "failure_mechanism": "the in-container preflight dereferenced a host-only SIF symlink target that is intentionally not mounted inside the running SIF namespace",
            "scientific_or_model_information_used": False,
            "new_model_or_scientific_outcome_opened": False,
        }
        or repair.get("container_attestation_schema_version") != 1
        or repair.get("container_attestation_mode_octal") != "0600"
        or repair.get("container_attestation_source")
        != "WRAPPER_HOST_BEFORE_CONTAINER"
        or repair.get(
            "runner_validates_canonical_attestation_job_run_mode_attempt_hash_bytes_and_predicates"
        )
        is not True
        or repair.get(
            "runner_no_longer_dereferences_host_only_SIF_target_inside_container"
        )
        is not True
        or repair.get("stable_dependency_commitment_uses_frozen_container_hash_and_bytes")
        is not True
        or repair.get("health_and_scientific_dependency_commitments_identical")
        is not True
        or repair.get("same_complete_28_case_two_replicate_production_paths")
        is not True
        or repair.get("mocks_or_shallow_load_path") is not False
        or repair.get("scientific_metric_count") != 0
        or repair.get("holdout_input_count") != 0
        or repair.get("model_fixture_threshold_metric_seed_or_gate_changed")
        is not False
        or resource
        != {
            "attempt": 6,
            "submission_count": 1,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "DDP": False,
            "wall_minutes_max": 15,
            "GPU_hours_max": 0.25,
            "new_storage_GiB_max": 1.0,
            "prior_protocol_accounted_GPU_hours": 0.7600439148479038,
            "active_aggregate_GPU_hours_max": 1.0100439148479038,
            "prior_health_run_storage_GiB": 2.8368085622787476e-6,
            "active_aggregate_new_storage_GiB_max": 1.0000028368085623,
            "direct_monetary_cost_USD": 0,
        }
        or rolling.get(
            "blanket_user_authorization_for_additional_ordinary_engineering_attempts"
        )
        is not True
        or rolling.get("each_attempt_uses_same_single_process_H100_slice_topology")
        is not True
        or rolling.get(
            "each_attempt_requires_exact_failure_specific_repair_committed_and_pushed_before_outcome"
        )
        is not True
        or rolling.get("full_microfixture_suite_restart_after_each_repair")
        is not True
        or rolling.get("metric_withholding_on_any_engineering_error") is not True
        or rolling.get(
            "no_extractor_model_fixture_source_threshold_partition_seed_metric_or_gate_change"
        )
        is not True
        or rolling.get("valid_complete_below_threshold_public_result_is_scientific_and_terminal")
        is not True
        or rolling.get("downstream_only_after_complete_public_pass") is not True
        or value.get("new_health_or_scientific_outcome_opened") is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ITERATIVE_REAUTHORIZATION_COMMITMENT")
    return value


def _engineering_health_iterative_attempt_6_result(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the timeout record from the first rolling-authorized attempt."""

    iterative = _engineering_health_iterative_reauthorization(cfg)
    try:
        value = cfg[
            "learner_effective_engineering_health_iterative_attempt_6_result"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "E_TUPLE_HEALTH_ITERATIVE_ATTEMPT_6_RESULT_MISSING"
        ) from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    submission = value.get("submission_provenance", {})
    compact = value.get("compact_aggregate", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    gate = value.get("terminal_gate", {})
    if (
        cfg.get("schema_version") not in {28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "ENGINEERING_TIMEOUT_ATTEMPT_6_BEFORE_MICROFIXTURE_PROJECTION_OR_FULL_RESULT_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("route_id") != "construct-aligned-engineering-health"
        or value.get("classification")
        != "ENGINEERING_RUNTIME_TIMEOUT_NOT_SCIENTIFIC_NO_GO"
        or value.get("preserved_attempt_5_blocker_sha256")
        != iterative["preserved_without_change"]["attempt_5_blocker_sha256"]
        or expected != ENGINEERING_HEALTH_ITERATIVE_ATTEMPT_6_BLOCKER_SHA256
        or digest(payload) != expected
        or submission
        != {
            "job_id": 316604,
            "attempt": 6,
            "scheduler_state": "TIMEOUT",
            "scheduler_exit_code": "0:0",
            "scheduler_elapsed_seconds": 924,
            "batch_CPU_seconds": 12.456,
            "batch_max_RSS_KiB": 6618920,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes_requested": 15,
            "DDP": False,
            "direct_monetary_cost_USD": 0,
        }
        or compact
        != {
            "wrapper_marker_count": 1,
            "container_attestation_count": 1,
            "topology_attestation_count": 1,
            "microfixture_manifest_count": 0,
            "full_result_count": 0,
            "private_trace_count": 0,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "stable_error_code_count": 0,
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "retained_file_count": 3,
            "retained_bytes": 830,
        }
        or diagnosis
        != {
            "wrapper_to_container_attestation_seconds": 6.0,
            "wrapper_to_topology_attestation_seconds": 8.0,
            "last_persisted_stage": "TOPOLOGY_ATTESTED_BEFORE_CONTAINER_RUNNER_PROGRESS_RECORD",
            "exact_in_container_stage_identified": False,
            "progress_instrumentation_required": True,
            "model_or_scientific_outcome_used": False,
        }
        or resource
        != {
            "attempt_GPU_hours_actual": 0.25666666666666665,
            "protocol_accounted_cumulative_GPU_hours_actual": 1.0167105815145705,
            "attempt_retained_storage_GiB": 7.729977369308472e-7,
            "protocol_accounted_cumulative_retained_storage_GiB": 3.6098062992095947e-6,
            "direct_monetary_cost_USD": 0,
        }
        or gate
        != {
            "scientific_decision_opened": False,
            "public_development_authorized": False,
            "governed_C_authorized": False,
            "LTX_or_synthetic_learner_run": False,
            "attempt_7_authorized_only_after_prospective_progress_repair_commit_and_push": True,
        }
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this result excluding blocker_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ITERATIVE_ATTEMPT_6_RESULT_COMMITMENT")
    return value


def _engineering_health_progress_repair(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the outcome-independent attempt-7 progress instrumentation."""

    attempt_6 = _engineering_health_iterative_attempt_6_result(cfg)
    try:
        value = cfg["learner_effective_engineering_health_progress_repair"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_PROGRESS_REPAIR_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("reauthorization_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    repair = value.get("failure_specific_repair", {})
    resource = value.get("active_attempt_resource_policy", {})
    execution = value.get("execution_and_stop_rule", {})
    if (
        cfg.get("schema_version") not in {28, 29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_ATTEMPT_6_TIMEOUT_BEFORE_PROGRESS_INSTRUMENTED_ATTEMPT_7_OR_NEW_OUTCOME"
        or value.get("scope") != "ENGINEERING_PROGRESS_INSTRUMENTATION_ONLY"
        or expected != ENGINEERING_HEALTH_PROGRESS_REPAIR_SHA256
        or digest(payload) != expected
        or preserved
        != {
            "attempt_6_blocker_sha256": attempt_6["blocker_commitment_sha256"],
            "attempt_5_blocker_sha256": ENGINEERING_HEALTH_PARSER_REPAIR_BLOCKER_SHA256,
            "public_fixture_manifest_sha256": "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6",
            "models_weights_fixtures_preprocessing_thresholds_metrics_seeds_module_behavior_scientific_gates_and_downstream_contract": True,
        }
        or repair
        != {
            "canonical_progress_record": "engineering-progress.json",
            "progress_record_schema_version": 1,
            "progress_record_mode_octal": "0600",
            "atomic_replace": True,
            "stable_stage_codes_only": True,
            "paths_filenames_hashes_prompts_rows_predictions_labels_or_metrics_recorded": False,
            "dependency_preflight_stage_boundaries_recorded": True,
            "fixture_projection_stage_boundaries_recorded": True,
            "module_ordinal_and_replicate_stage_boundaries_recorded": True,
            "full_28_case_two_replicate_suite_unchanged": True,
            "progress_record_used_for_scientific_metrics_or_selection": False,
            "scientific_metric_count": 0,
            "model_fixture_threshold_metric_seed_or_gate_changed": False,
        }
        or resource
        != {
            "attempt": 7,
            "submission_count": 1,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "DDP": False,
            "wall_minutes_max": 15,
            "GPU_hours_max": 0.25,
            "new_storage_GiB_max": 1.0,
            "prior_protocol_accounted_GPU_hours_actual": 1.0167105815145705,
            "active_aggregate_GPU_hours_max": 1.2667105815145705,
            "prior_health_run_storage_GiB_actual": 3.6098062992095947e-6,
            "active_aggregate_new_storage_GiB_max": 1.0000036098062992,
            "direct_monetary_cost_USD": 0,
        }
        or execution
        != {
            "attempt_7_is_complete_health_suite_not_shallow_diagnostic": True,
            "same_single_process_H100_slice_topology": True,
            "metric_withholding_on_timeout_or_error": True,
            "on_timeout_use_only_last_stable_progress_stage_for_next_outcome_independent_repair": True,
            "valid_complete_scientific_result_remains_terminal_under_frozen_gate": True,
        }
        or value.get("new_health_or_scientific_outcome_opened") is not False
        or value.get("reauthorization_commitment_scope")
        != "canonical JSON of this amendment excluding reauthorization_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PROGRESS_REPAIR_COMMITMENT")
    return value


def _engineering_health_attempt_7_result(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the metric-withheld dependency-preflight timeout."""

    progress = _engineering_health_progress_repair(cfg)
    try:
        value = cfg["learner_effective_engineering_health_attempt_7_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_7_RESULT_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    submission = value.get("submission_provenance", {})
    compact = value.get("compact_aggregate", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    gate = value.get("terminal_gate", {})
    if (
        cfg.get("schema_version") not in {29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "ENGINEERING_TIMEOUT_ATTEMPT_7_DURING_DEPENDENCY_PREFLIGHT_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("route_id") != "construct-aligned-engineering-health"
        or value.get("classification")
        != "ENGINEERING_WALL_CAP_TOO_SHORT_AFTER_VALIDATED_ACTION_WEIGHT_NOT_SCIENTIFIC_NO_GO"
        or value.get("preserved_progress_repair_sha256")
        != progress["reauthorization_commitment_sha256"]
        or expected != ENGINEERING_HEALTH_ATTEMPT_7_BLOCKER_SHA256
        or digest(payload) != expected
        or submission
        != {
            "job_id": 316641,
            "attempt": 7,
            "scheduler_state": "TIMEOUT",
            "scheduler_exit_code": "0:0",
            "scheduler_elapsed_seconds": 918,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes_requested": 15,
            "DDP": False,
            "direct_monetary_cost_USD": 0,
        }
        or compact
        != {
            "wrapper_marker_count": 1,
            "container_attestation_count": 1,
            "topology_attestation_count": 1,
            "progress_record_count": 1,
            "microfixture_manifest_count": 0,
            "full_result_count": 0,
            "private_trace_count": 0,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
            "stable_error_code_count": 0,
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "sensitive_detail_field_count": 0,
            "retained_file_count": 4,
            "retained_bytes": 1069,
        }
        or diagnosis
        != {
            "last_persisted_stage": "DEPENDENCY_REPOSITORY_ARCHIVE",
            "last_persisted_stage_ordinal": 9,
            "progress_update_count": 11,
            "progress_elapsed_seconds_rounded": 874.543,
            "last_completed_dependency_stage": "DEPENDENCY_ACTION_WEIGHT",
            "repository_archive_validation_completed": False,
            "pre_model_dependency_validation_executed": True,
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "model_or_scientific_outcome_used": False,
        }
        or resource
        != {
            "attempt_GPU_hours_actual": 0.255,
            "protocol_accounted_cumulative_GPU_hours_actual": 1.2717105815145704,
            "attempt_retained_storage_GiB": 9.955838322639465e-7,
            "protocol_accounted_cumulative_retained_storage_GiB": 4.605390131473541e-6,
            "direct_monetary_cost_USD": 0,
        }
        or gate
        != {
            "scientific_decision_opened": False,
            "public_development_authorized": False,
            "governed_C_authorized": False,
            "LTX_or_synthetic_learner_run": False,
            "attempt_8_authorized_only_after_prospective_extended_wall_repair_commit_and_push": True,
        }
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this result excluding blocker_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_7_RESULT_COMMITMENT")
    return value


def _engineering_health_extended_wall_repair(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the outcome-independent attempt-8 wall-cap correction."""

    attempt_7 = _engineering_health_attempt_7_result(cfg)
    try:
        value = cfg["learner_effective_engineering_health_extended_wall_repair"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_EXTENDED_WALL_REPAIR_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("reauthorization_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    repair = value.get("failure_specific_repair", {})
    historical = value.get("historical_incomplete_attempt_wall_minutes", {})
    resource = value.get("active_attempt_resource_policy", {})
    execution = value.get("execution_and_stop_rule", {})
    if (
        cfg.get("schema_version") not in {29, 30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_ATTEMPT_7_TIMEOUT_BEFORE_EXTENDED_WALL_ATTEMPT_8_OR_NEW_OUTCOME"
        or value.get("scope")
        != "OUTCOME_INDEPENDENT_HEALTH_WALL_CAP_EXTENSION_ONLY"
        or expected != ENGINEERING_HEALTH_EXTENDED_WALL_REPAIR_SHA256
        or digest(payload) != expected
        or preserved
        != {
            "attempt_7_blocker_sha256": attempt_7["blocker_commitment_sha256"],
            "attempt_6_blocker_sha256": ENGINEERING_HEALTH_ITERATIVE_ATTEMPT_6_BLOCKER_SHA256,
            "public_fixture_manifest_sha256": "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6",
            "models_weights_fixtures_preprocessing_thresholds_metrics_seeds_module_behavior_scientific_gates_and_downstream_contract": True,
        }
        or repair
        != {
            "diagnosis": "the 15-minute cap expired after exact dependency validation advanced through the 5130482632-byte action weight and entered repository-archive verification",
            "wall_cap_minutes_before": 15,
            "wall_cap_minutes_after": 60,
            "same_private_progress_schema": True,
            "same_complete_28_case_two_replicate_suite": True,
            "same_exact_artifact_rehash_and_tree_validation": True,
            "dependency_or_model_cache_substitution": False,
            "model_fixture_source_threshold_partition_seed_metric_or_gate_changed": False,
        }
        or historical != {"1": 15, "2": 15, "4": 15, "6": 15, "7": 15}
        or resource
        != {
            "attempt": 8,
            "submission_count": 1,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "DDP": False,
            "wall_minutes_max": 60,
            "GPU_hours_max": 1.0,
            "new_storage_GiB_max": 1.0,
            "prior_protocol_accounted_GPU_hours_actual": 1.2717105815145704,
            "active_aggregate_GPU_hours_max": 2.2600439148479035,
            "prior_health_run_storage_GiB_actual": 4.605390131473541e-6,
            "active_aggregate_new_storage_GiB_max": 1.0000046053901315,
            "direct_monetary_cost_USD": 0,
        }
        or execution
        != {
            "attempt_8_is_complete_health_suite_not_shallow_diagnostic": True,
            "same_single_process_H100_slice_topology": True,
            "metric_withholding_on_timeout_or_error": True,
            "commit_and_push_before_submission": True,
            "valid_complete_scientific_result_remains_terminal_under_frozen_gate": True,
        }
        or value.get("new_health_or_scientific_outcome_opened") is not False
        or value.get("reauthorization_commitment_scope")
        != "canonical JSON of this amendment excluding reauthorization_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_EXTENDED_WALL_REPAIR_COMMITMENT")
    return value


def _engineering_health_scheduler_policy(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the outcome-free earliest-eligible GPU selection for attempt 8."""

    extended = _engineering_health_extended_wall_repair(cfg)
    try:
        value = cfg["learner_effective_engineering_health_scheduler_policy"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_SCHEDULER_POLICY_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("amendment_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    standing = value.get("standing_selection_policy", {})
    canceled = value.get("canceled_submission", {})
    comparison = value.get("scheduler_only_comparison", {})
    active = value.get("active_attempt_resource_policy", {})
    attestation = value.get("topology_attestation_contract", {})
    execution = value.get("execution_and_stop_rule", {})
    candidates = comparison.get("candidates")
    if (
        cfg.get("schema_version") not in {30, 31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_ZERO_RUNTIME_H100_MIG_CANCELLATION_AND_SUBMISSION_RECHECK_BEFORE_FULL_H100_ATTEMPT_8_OR_NEW_MODEL_OUTCOME"
        or value.get("scope")
        != "STANDING_ZERO_COST_EARLIEST_SCHEDULER_ELIGIBLE_GPU_POLICY_AND_ATTEMPT_8_SELECTION_ONLY"
        or expected != ENGINEERING_HEALTH_SCHEDULER_POLICY_SHA256
        or digest(payload) != expected
        or value.get("amendment_commitment_scope")
        != "canonical JSON of this amendment excluding amendment_commitment_sha256"
        or preserved
        != {
            "extended_wall_repair_sha256": extended[
                "reauthorization_commitment_sha256"
            ],
            "attempt_7_blocker_sha256": ENGINEERING_HEALTH_ATTEMPT_7_BLOCKER_SHA256,
            "public_fixture_manifest_sha256": "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6",
            "models_weights_fixtures_inputs_seeds_thresholds_metrics_repair_allowances_scientific_gates_privacy_rules_and_downstream_contract": True,
        }
        or standing
        != {
            "eligible_GPU_types": [
                "NVIDIA_A30_24GB",
                "NVIDIA_H100_NVL",
                "NVIDIA_H100_NVL_3G_47GB_MIG",
                "NVIDIA_H200_NVL",
            ],
            "scheduler_test_only_immediately_before_engineering_submission": True,
            "earliest_estimated_start_wins": True,
            "one_live_job_max": 1,
            "cancel_only_pending_zero_runtime_no_output": True,
            "one_process_no_DDP": True,
            "zero_direct_monetary_cost_required": True,
            "freeze_exact_topology_before_scientific_metrics_can_open": True,
            "no_hardware_switch_in_response_to_engineering_or_scientific_outcomes": True,
        }
        or canceled
        != {
            "job_id": 316697,
            "partition": "h100",
            "GRES": "gpu:nvidia_h100_nvl_3g.47gb:1",
            "state": "CANCELLED_BEFORE_ALLOCATION",
            "elapsed_seconds": 0,
            "GPU_hours": 0,
            "attempt_file_count": 0,
            "attempt_bytes": 0,
            "engineering_outcome_opened": False,
            "scientific_metric_count": 0,
        }
        or comparison.get("checked_on") != "2026-08-04"
        or comparison.get("real_job_submitted_by_checks") is not False
        or comparison.get("request")
        != {
            "nodes": 1,
            "tasks": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes": 60,
            "DDP": False,
            "direct_monetary_cost_USD": 0,
        }
        or candidates
        != [
            {
                "GPU_type": "NVIDIA_A30_24GB",
                "partition": "a30",
                "GRES": "gpu:nvidia_a30:1",
                "eligible": True,
                "estimated_start": "2026-08-10T11:35:00-05:00",
            },
            {
                "GPU_type": "NVIDIA_H100_NVL",
                "partition": "h100",
                "GRES": "gpu:nvidia_h100_nvl:1",
                "eligible": True,
                "estimated_start": "2026-08-04T23:36:18-05:00",
            },
            {
                "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
                "partition": "h100",
                "GRES": "gpu:nvidia_h100_nvl_3g.47gb:1",
                "eligible": True,
                "estimated_start": "2026-08-05T02:52:48-05:00",
            },
            {
                "GPU_type": "NVIDIA_H200_NVL",
                "partition": "h200",
                "GRES": "gpu:nvidia_h200_nvl:1",
                "eligible": True,
                "estimated_start": "2026-08-05T03:17:18-05:00",
            },
        ]
        or comparison.get("selection_rule")
        != "minimum scheduler-estimated start among compatible eligible zero-cost one-process requests; fixed GPU-type order only breaks exact timestamp ties"
        or comparison.get("selected_GPU_type") != "NVIDIA_H100_NVL"
        or active
        != {
            "attempt": 8,
            "submission_count": 1,
            "partition": "h100",
            "GRES": "gpu:nvidia_h100_nvl:1",
            "GPU_type": "NVIDIA_H100_NVL",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "DDP": False,
            "wall_minutes_max": 60,
            "GPU_hours_max": 1.0,
            "new_storage_GiB_max": 1.0,
            "prior_protocol_accounted_GPU_hours_actual": 1.2717105815145704,
            "active_aggregate_GPU_hours_max": 2.2600439148479035,
            "prior_health_run_storage_GiB_actual": 4.605390131473541e-6,
            "active_aggregate_new_storage_GiB_max": 1.0000046053901315,
            "direct_monetary_cost_USD": 0,
            "historical_attempt_GPU_types": {
                "1": "NVIDIA_H100_NVL_3G_47GB_MIG",
                "2": "NVIDIA_H100_NVL_3G_47GB_MIG",
                "3": "NVIDIA_H100_NVL_3G_47GB_MIG",
                "4": "NVIDIA_H100_NVL_3G_47GB_MIG",
                "5": "NVIDIA_H100_NVL_3G_47GB_MIG",
                "6": "NVIDIA_H100_NVL_3G_47GB_MIG",
                "7": "NVIDIA_H100_NVL_3G_47GB_MIG",
            },
        }
        or attestation
        != {
            "partition": "h100",
            "node_count": 1,
            "task_count": 1,
            "CPU_count": 8,
            "time_limit_minutes": 60,
            "memory_per_CPU_GiB": 4,
            "GRES": "gpu:nvidia_h100_nvl:1",
            "expected_device_name": "NVIDIA H100 NVL",
            "visible_memory_GiB_min": 85,
            "visible_memory_GiB_max": 100,
            "world_size": 1,
            "local_world_size": 1,
        }
        or execution
        != {
            "attempt_8_is_complete_health_suite_not_shallow_diagnostic": True,
            "same_complete_28_case_two_replicate_suite": True,
            "metric_withholding_on_timeout_or_error": True,
            "commit_and_push_before_submission": True,
            "valid_complete_scientific_result_remains_terminal_under_frozen_gate": True,
        }
        or value.get("new_health_or_scientific_outcome_opened") is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_SCHEDULER_POLICY_COMMITMENT")
    return value


def _engineering_health_attempt_8_result(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the metric-withheld container-Git engineering failure."""

    scheduler = _engineering_health_scheduler_policy(cfg)
    try:
        value = cfg["learner_effective_engineering_health_attempt_8_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_8_RESULT_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    submission = value.get("submission_provenance", {})
    compact = value.get("compact_aggregate", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    gate = value.get("terminal_gate", {})
    if (
        cfg.get("schema_version") not in {31, 32, 33, 34, 35}
        or value.get("status")
        != "ENGINEERING_BLOCKER_ATTEMPT_8_DURING_DEPENDENCY_ACTIVITY_CODE_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("route_id") != "construct-aligned-engineering-health"
        or value.get("classification")
        != "ENGINEERING_CONTAINER_GIT_EXECUTABLE_MISSING_NOT_SCIENTIFIC_NO_GO"
        or value.get("preserved_scheduler_policy_sha256")
        != scheduler["amendment_commitment_sha256"]
        or expected != ENGINEERING_HEALTH_ATTEMPT_8_BLOCKER_SHA256
        or digest(payload) != expected
        or submission
        != {
            "job_id": 316777,
            "attempt": 8,
            "scheduler_state": "COMPLETED",
            "scheduler_exit_code": "0:0",
            "scheduler_elapsed_seconds": 199,
            "GPU_type": "NVIDIA_H100_NVL",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes_requested": 60,
            "DDP": False,
            "direct_monetary_cost_USD": 0,
        }
        or compact
        != {
            "module_count": 7,
            "completed_module_count": 0,
            "failed_module_count": 7,
            "case_count": 28,
            "holdout_input_count": 0,
            "scientific_metric_count": 0,
            "failure_count": 7,
            "invalid_retained_record_count": 0,
            "silent_truncation_count": 0,
            "external_call_count": 0,
            "unaccounted_failure_count": 1,
            "private_trace_count": 7,
            "preflight_blocked_trace_count": 6,
            "retained_file_count": 12,
            "retained_bytes": 5055,
            "stdout_bytes": 1354,
            "stderr_bytes": 0,
            "public_fixture_manifest_commitment_sha256": "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6",
            "runner_commitment_sha256": "75734064731b62400702299297f8c0a6ec9b8cbd54dd4896e53d9c265dbfbdf9",
            "config_commitment_sha256": "490f63e294d4751a962d41c1956b02e1ac51bafc8a42aed2f76aa8c8704b0c89",
            "dependency_config_commitment_sha256": "6c7816d4fdedf885a684c0e4ddc9c829aaca7ceaef068fccf01da1a405903e8c",
            "microfixture_manifest_commitment_sha256": "981e1cf51ba9f538976aaf6cb2a806113f54656fe3a2dc4cc0095841bb6580ee",
            "engineering_health_commitment_sha256": "5df1d3e169f17ecbd269e7dd8c29fc0b033a301fb6cd15e69eebecccdceb92c8",
        }
        or diagnosis
        != {
            "last_persisted_stage": "DEPENDENCY_ACTIVITY_CODE",
            "last_persisted_stage_ordinal": 14,
            "progress_update_count": 16,
            "progress_elapsed_seconds_rounded": 196.865,
            "file_not_found_trace_count": 1,
            "missing_git_executable_classification_count": 1,
            "container_git_executable_available": False,
            "host_clean_tree_attestation_required": True,
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "sensitive_detail_field_count": 0,
            "model_or_scientific_outcome_used": False,
        }
        or resource
        != {
            "attempt_GPU_hours_actual": 0.054775266514884104,
            "protocol_accounted_cumulative_GPU_hours_actual": 1.3148191813627879,
            "attempt_retained_storage_GiB": 1.6046687960624695e-6,
            "protocol_accounted_cumulative_retained_storage_GiB": 6.210058927536011e-6,
            "attempt_wall_minutes_actual": 3.286515990893046,
            "protocol_accounted_cumulative_wall_minutes_actual": 78.88915088176728,
            "direct_monetary_cost_USD": 0,
        }
        or gate
        != {
            "scientific_decision_opened": False,
            "public_development_authorized": False,
            "governed_C_authorized": False,
            "LTX_or_synthetic_learner_run": False,
            "attempt_9_authorized_only_after_prospective_git_fallback_repair_commit_and_push": True,
        }
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this result excluding blocker_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_8_RESULT_COMMITMENT")
    return value


def _engineering_health_git_fallback_repair(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the clean-tree-attested container-Git fallback for attempt 9."""

    attempt_8 = _engineering_health_attempt_8_result(cfg)
    try:
        value = cfg["learner_effective_engineering_health_git_fallback_repair"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_GIT_FALLBACK_REPAIR_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("repair_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    diagnosis = value.get("failure_specific_diagnosis", {})
    attestations = value.get("clean_tree_attestations", {})
    repair = value.get("failure_specific_repair", {})
    comparison = value.get("scheduler_only_comparison", {})
    active = value.get("active_attempt_resource_policy", {})
    topology = value.get("topology_attestation_contract", {})
    execution = value.get("execution_and_stop_rule", {})
    repository_attestations = attestations.get("repositories")
    expected_repositories = [
        {
            "family": "egohod_activity_code_tree",
            "expected_commit": "3682c1906eb33d55431751111bb81dabe854f449",
            "git_index_sha256": "a6caae2499485ff6195e4fbd565bd9bfe1a7c4f828cd69cac430b701a961416d",
            "git_index_bytes": 6399,
            "file_count": 49,
            "bytes": 2019900,
            "tree_commitment_sha256": "1d0ebcc69290dae8e7b33c0e759ee25b050e1d6b110aa6a5f681b7ef867174eb",
            "host_unexpected_status_count": 0,
        },
        {
            "family": "openai_clip_code_tree",
            "expected_commit": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
            "git_index_sha256": "7754654607dedf1df4fdb6ec4e1f04c595fa7084dffd7381730b131bb6933c59",
            "git_index_bytes": 2102,
            "file_count": 22,
            "bytes": 5394303,
            "tree_commitment_sha256": "3de34b6347e7e8303b2bb137c2afa3d6a4806a9362adee8e8c57451e8d1a2cbd",
            "host_unexpected_status_count": 0,
        },
    ]
    catalog = {
        "NVIDIA_A30_24GB": {
            "partition": "a30",
            "GRES": "gpu:nvidia_a30:1",
            "expected_device_name": "NVIDIA A30",
            "visible_memory_GiB_min": 23,
            "visible_memory_GiB_max": 25,
        },
        "NVIDIA_H100_NVL": {
            "partition": "h100",
            "GRES": "gpu:nvidia_h100_nvl:1",
            "expected_device_name": "NVIDIA H100 NVL",
            "visible_memory_GiB_min": 85,
            "visible_memory_GiB_max": 100,
        },
        "NVIDIA_H100_NVL_3G_47GB_MIG": {
            "partition": "h100",
            "GRES": "gpu:nvidia_h100_nvl_3g.47gb:1",
            "expected_device_name": "NVIDIA H100 NVL MIG 3g.47gb",
            "visible_memory_GiB_min": 45,
            "visible_memory_GiB_max": 50,
        },
        "NVIDIA_H200_NVL": {
            "partition": "h200",
            "GRES": "gpu:nvidia_h200_nvl:1",
            "expected_device_name": "NVIDIA H200 NVL",
            "visible_memory_GiB_min": 135,
            "visible_memory_GiB_max": 145,
        },
    }
    candidates = comparison.get("candidates")
    candidate_types = (
        {item.get("GPU_type") for item in candidates}
        if isinstance(candidates, list) and all(isinstance(item, dict) for item in candidates)
        else set()
    )
    candidate_shape_ok = (
        isinstance(candidates, list)
        and len(candidates) == len(catalog)
        and candidate_types == set(catalog)
        and all(
            set(item)
            == {"GPU_type", "partition", "GRES", "eligible", "estimated_start"}
            and item["partition"] == catalog[item["GPU_type"]]["partition"]
            and item["GRES"] == catalog[item["GPU_type"]]["GRES"]
            and item["eligible"] is True
            and isinstance(item["estimated_start"], str)
            and re.fullmatch(r"2026-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}-0[5-7]:00", item["estimated_start"])
            is not None
            for item in candidates
        )
    )
    winner = (
        min(
            candidates,
            key=lambda item: (
                item["estimated_start"],
                list(catalog).index(item["GPU_type"]),
            ),
        )
        if candidate_shape_ok
        else {}
    )
    selected_type = comparison.get("selected_GPU_type")
    selected = catalog.get(selected_type, {})
    expected_active = {
        "attempt": 9,
        "submission_count": 1,
        "partition": selected.get("partition"),
        "GRES": selected.get("GRES"),
        "GPU_type": selected_type,
        "GPU_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "DDP": False,
        "wall_minutes_max": 60,
        "GPU_hours_max": 1.0,
        "new_storage_GiB_max": 1.0,
        "prior_protocol_accounted_GPU_hours_actual": 1.3148191813627879,
        "active_aggregate_GPU_hours_max": 2.314819181362788,
        "prior_health_run_storage_GiB_actual": 6.210058927536011e-6,
        "active_aggregate_new_storage_GiB_max": 1.0000062100589275,
        "direct_monetary_cost_USD": 0,
        "historical_attempt_GPU_types": {
            "1": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "2": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "3": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "4": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "5": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "6": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "7": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "8": "NVIDIA_H100_NVL",
        },
    }
    expected_topology = {
        "partition": selected.get("partition"),
        "node_count": 1,
        "task_count": 1,
        "CPU_count": 8,
        "time_limit_minutes": 60,
        "memory_per_CPU_GiB": 4,
        "GRES": selected.get("GRES"),
        "expected_device_name": selected.get("expected_device_name"),
        "visible_memory_GiB_min": selected.get("visible_memory_GiB_min"),
        "visible_memory_GiB_max": selected.get("visible_memory_GiB_max"),
        "world_size": 1,
        "local_world_size": 1,
    }
    if (
        cfg.get("schema_version") not in {31, 32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_ATTEMPT_8_BLOCKER_BEFORE_GIT_FREE_TREE_ATTESTED_ATTEMPT_9_OR_NEW_OUTCOME"
        or value.get("scope")
        != "OUTCOME_INDEPENDENT_CONTAINER_GIT_ABSENCE_REPAIR_AND_ATTEMPT_9_TOPOLOGY_SELECTION"
        or expected != ENGINEERING_HEALTH_GIT_FALLBACK_REPAIR_SHA256
        or digest(payload) != expected
        or preserved
        != {
            "attempt_8_blocker_sha256": attempt_8["blocker_commitment_sha256"],
            "attempt_8_engineering_health_commitment_sha256": "5df1d3e169f17ecbd269e7dd8c29fc0b033a301fb6cd15e69eebecccdceb92c8",
            "scheduler_policy_sha256": ENGINEERING_HEALTH_SCHEDULER_POLICY_SHA256,
            "public_fixture_manifest_sha256": "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6",
            "models_weights_fixtures_inputs_seeds_thresholds_metrics_scientific_gates_privacy_rules_and_downstream_contract": True,
        }
        or diagnosis
        != {
            "failure_stage": "DEPENDENCY_ACTIVITY_CODE",
            "exception_type": "FileNotFoundError",
            "missing_executable": "git",
            "container_git_available": False,
            "host_git_available": True,
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "outcome_information_used_for_repair": False,
        }
        or attestations.get("host_git_clean_verification_before_freeze") is not True
        or attestations.get("official_commit_match_count") != 2
        or attestations.get("unexpected_status_count") != 0
        or repository_attestations != expected_repositories
        or repair
        != {
            "runner_uses_git_status_when_git_is_available": True,
            "runner_uses_pinned_HEAD_index_and_complete_worktree_tree_attestation_when_git_is_unavailable": True,
            "complete_tree_includes_relative_paths_file_sha256_and_bytes": True,
            "complete_tree_excludes_only_git_metadata_pycache_and_pyc": True,
            "untracked_or_modified_importable_source_detected": True,
            "container_package_or_dependency_change": False,
            "model_fixture_source_threshold_partition_seed_metric_or_gate_changed": False,
        }
        or comparison.get("checked_on") != "2026-08-04"
        or comparison.get("real_job_submitted_by_checks") is not False
        or comparison.get("request")
        != {
            "nodes": 1,
            "tasks": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes": 60,
            "DDP": False,
            "direct_monetary_cost_USD": 0,
        }
        or not candidate_shape_ok
        or comparison.get("selection_rule")
        != "minimum scheduler-estimated start among compatible eligible zero-cost one-process requests; fixed GPU-type order only breaks exact timestamp ties"
        or selected_type != winner.get("GPU_type")
        or active != expected_active
        or topology != expected_topology
        or execution
        != {
            "attempt_9_is_complete_health_suite_not_shallow_diagnostic": True,
            "same_complete_28_case_two_replicate_suite": True,
            "metric_withholding_on_timeout_or_error": True,
            "commit_and_push_before_submission": True,
            "one_live_job_max": 1,
            "no_repeated_scheduler_polling_or_unchanged_status_updates": True,
            "valid_complete_scientific_result_remains_terminal_under_frozen_gate": True,
        }
        or value.get("new_health_or_scientific_outcome_opened") is not False
        or value.get("repair_commitment_scope")
        != "canonical JSON of this amendment excluding repair_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_GIT_FALLBACK_REPAIR_COMMITMENT")
    return value


def _engineering_health_attempt_9_result(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the metric-withheld historical-lineage engineering failure."""

    git_fallback = _engineering_health_git_fallback_repair(cfg)
    try:
        value = cfg["learner_effective_engineering_health_attempt_9_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_9_RESULT_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    submission = value.get("submission_provenance", {})
    compact = value.get("compact_aggregate", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    gate = value.get("terminal_gate", {})
    if (
        cfg.get("schema_version") not in {32, 33, 34, 35}
        or value.get("status")
        != "ENGINEERING_BLOCKER_ATTEMPT_9_BEFORE_RUNNER_PROGRESS_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("route_id") != "construct-aligned-engineering-health"
        or value.get("classification")
        != "ENGINEERING_HISTORICAL_CONFIG_COMMITMENT_LINEAGE_MISSING_NOT_SCIENTIFIC_NO_GO"
        or value.get("preserved_git_fallback_repair_sha256")
        != git_fallback["repair_commitment_sha256"]
        or expected != ENGINEERING_HEALTH_ATTEMPT_9_BLOCKER_SHA256
        or digest(payload) != expected
        or submission
        != {
            "job_id": 316813,
            "attempt": 9,
            "scheduler_state": "FAILED",
            "scheduler_exit_code": "1:0",
            "scheduler_elapsed_seconds": 10,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes_requested": 60,
            "DDP": False,
            "direct_monetary_cost_USD": 0,
        }
        or compact
        != {
            "wrapper_marker_count": 1,
            "container_attestation_count": 1,
            "topology_attestation_count": 1,
            "progress_record_count": 0,
            "full_result_count": 0,
            "private_trace_count": 0,
            "stdout_bytes": 0,
            "stderr_bytes": 764,
            "stable_error_code_count": 1,
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "sensitive_detail_field_count": 0,
            "retained_file_count": 3,
            "retained_bytes": 804,
        }
        or diagnosis
        != {
            "failure_stage": "PRIOR_ATTEMPT_8_FULL_RESULT_VALIDATION_BEFORE_RUNNER_PROGRESS",
            "stable_error_code": "E_TUPLE_HEALTH_FULL_SCHEMA",
            "prior_attempt_8_full_result_present": True,
            "prior_attempt_8_engineering_health_commitment_sha256": "5df1d3e169f17ecbd269e7dd8c29fc0b033a301fb6cd15e69eebecccdceb92c8",
            "prior_attempt_8_config_commitment_sha256": "490f63e294d4751a962d41c1956b02e1ac51bafc8a42aed2f76aa8c8704b0c89",
            "runner_progress_started": False,
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "model_or_scientific_outcome_used": False,
        }
        or resource
        != {
            "attempt_GPU_hours_actual": 0.002777777777777778,
            "protocol_accounted_cumulative_GPU_hours_actual": 1.3175969591405656,
            "attempt_retained_storage_GiB": 7.487833499908447e-7,
            "protocol_accounted_cumulative_retained_storage_GiB": 6.9588422775268555e-6,
            "attempt_wall_minutes_actual": 0.16666666666666666,
            "protocol_accounted_cumulative_wall_minutes_actual": 79.05581754843395,
            "direct_monetary_cost_USD": 0,
        }
        or gate
        != {
            "scientific_decision_opened": False,
            "public_development_authorized": False,
            "governed_C_authorized": False,
            "LTX_or_synthetic_learner_run": False,
            "attempt_10_authorized_only_after_prospective_historical_lineage_repair_commit_and_push": True,
        }
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this result excluding blocker_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_9_RESULT_COMMITMENT")
    return value


def _engineering_health_historical_lineage_repair(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the attempt-10 historical full-record lineage repair."""

    attempt_9 = _engineering_health_attempt_9_result(cfg)
    attempt_8 = _engineering_health_attempt_8_result(cfg)
    try:
        value = cfg[
            "learner_effective_engineering_health_historical_lineage_repair"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_HISTORICAL_LINEAGE_REPAIR_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("repair_commitment_sha256", None)
    preserved = value.get("preserved_without_change", {})
    diagnosis = value.get("failure_specific_diagnosis", {})
    repair = value.get("failure_specific_repair", {})
    comparison = value.get("scheduler_only_comparison", {})
    active = value.get("active_attempt_resource_policy", {})
    topology = value.get("topology_attestation_contract", {})
    execution = value.get("execution_and_stop_rule", {})
    catalog = {
        "NVIDIA_A30_24GB": {
            "partition": "a30",
            "GRES": "gpu:nvidia_a30:1",
            "expected_device_name": "NVIDIA A30",
            "visible_memory_GiB_min": 23,
            "visible_memory_GiB_max": 25,
        },
        "NVIDIA_H100_NVL": {
            "partition": "h100",
            "GRES": "gpu:nvidia_h100_nvl:1",
            "expected_device_name": "NVIDIA H100 NVL",
            "visible_memory_GiB_min": 85,
            "visible_memory_GiB_max": 100,
        },
        "NVIDIA_H100_NVL_3G_47GB_MIG": {
            "partition": "h100",
            "GRES": "gpu:nvidia_h100_nvl_3g.47gb:1",
            "expected_device_name": "NVIDIA H100 NVL MIG 3g.47gb",
            "visible_memory_GiB_min": 45,
            "visible_memory_GiB_max": 50,
        },
        "NVIDIA_H200_NVL": {
            "partition": "h200",
            "GRES": "gpu:nvidia_h200_nvl:1",
            "expected_device_name": "NVIDIA H200 NVL",
            "visible_memory_GiB_min": 135,
            "visible_memory_GiB_max": 145,
        },
    }
    candidates = comparison.get("candidates")
    candidate_shape_ok = (
        isinstance(candidates, list)
        and len(candidates) == len(catalog)
        and {item.get("GPU_type") for item in candidates if isinstance(item, dict)}
        == set(catalog)
        and all(
            isinstance(item, dict)
            and set(item)
            == {"GPU_type", "partition", "GRES", "eligible", "estimated_start"}
            and item["partition"] == catalog[item["GPU_type"]]["partition"]
            and item["GRES"] == catalog[item["GPU_type"]]["GRES"]
            and item["eligible"] is True
            and isinstance(item["estimated_start"], str)
            and re.fullmatch(
                r"2026-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}-0[5-7]:00",
                item["estimated_start"],
            )
            is not None
            for item in candidates
        )
    )
    winner = (
        min(
            candidates,
            key=lambda item: (
                item["estimated_start"],
                list(catalog).index(item["GPU_type"]),
            ),
        )
        if candidate_shape_ok
        else {}
    )
    selected_type = comparison.get("selected_GPU_type")
    selected = catalog.get(selected_type, {})
    expected_active = {
        "attempt": 10,
        "submission_count": 1,
        "partition": selected.get("partition"),
        "GRES": selected.get("GRES"),
        "GPU_type": selected_type,
        "GPU_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "DDP": False,
        "wall_minutes_max": 60,
        "GPU_hours_max": 1.0,
        "new_storage_GiB_max": 1.0,
        "prior_protocol_accounted_GPU_hours_actual": 1.3175969591405656,
        "active_aggregate_GPU_hours_max": 2.3175969591405656,
        "prior_health_run_storage_GiB_actual": 6.9588422775268555e-6,
        "active_aggregate_new_storage_GiB_max": 1.0000069588422775,
        "direct_monetary_cost_USD": 0,
        "historical_incomplete_attempt_wall_minutes": {
            "1": 15,
            "2": 15,
            "4": 15,
            "6": 15,
            "7": 15,
            "9": 0.16666666666666666,
        },
        "historical_attempt_GPU_types": {
            "1": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "2": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "3": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "4": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "5": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "6": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "7": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "8": "NVIDIA_H100_NVL",
            "9": "NVIDIA_A30_24GB",
        },
    }
    expected_topology = {
        "partition": selected.get("partition"),
        "node_count": 1,
        "task_count": 1,
        "CPU_count": 8,
        "time_limit_minutes": 60,
        "memory_per_CPU_GiB": 4,
        "GRES": selected.get("GRES"),
        "expected_device_name": selected.get("expected_device_name"),
        "visible_memory_GiB_min": selected.get("visible_memory_GiB_min"),
        "visible_memory_GiB_max": selected.get("visible_memory_GiB_max"),
        "world_size": 1,
        "local_world_size": 1,
    }
    if (
        cfg.get("schema_version") not in {32, 33, 34, 35}
        or value.get("status")
        != "FROZEN_AFTER_ATTEMPT_9_BLOCKER_BEFORE_HISTORICAL_LINEAGE_REPAIRED_ATTEMPT_10_OR_NEW_OUTCOME"
        or value.get("scope")
        != "OUTCOME_INDEPENDENT_HISTORICAL_CONFIG_COMMITMENT_VALIDATION_REPAIR_AND_ATTEMPT_10_TOPOLOGY_SELECTION"
        or expected != ENGINEERING_HEALTH_HISTORICAL_LINEAGE_REPAIR_SHA256
        or digest(payload) != expected
        or preserved
        != {
            "attempt_9_blocker_sha256": attempt_9["blocker_commitment_sha256"],
            "attempt_8_blocker_sha256": attempt_8["blocker_commitment_sha256"],
            "attempt_8_engineering_health_commitment_sha256": "5df1d3e169f17ecbd269e7dd8c29fc0b033a301fb6cd15e69eebecccdceb92c8",
            "git_fallback_repair_sha256": ENGINEERING_HEALTH_GIT_FALLBACK_REPAIR_SHA256,
            "public_fixture_manifest_sha256": "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6",
            "models_weights_fixtures_inputs_seeds_thresholds_metrics_scientific_gates_privacy_rules_and_downstream_contract": True,
        }
        or diagnosis
        != {
            "failure_stage": "PRIOR_ATTEMPT_8_FULL_RESULT_VALIDATION_BEFORE_RUNNER_PROGRESS",
            "stable_error_code": "E_TUPLE_HEALTH_FULL_SCHEMA",
            "historical_attempt": 8,
            "historical_config_commitment_sha256": "490f63e294d4751a962d41c1956b02e1ac51bafc8a42aed2f76aa8c8704b0c89",
            "historical_engineering_health_commitment_sha256": "5df1d3e169f17ecbd269e7dd8c29fc0b033a301fb6cd15e69eebecccdceb92c8",
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "outcome_information_used_for_repair": False,
        }
        or repair
        != {
            "current_config_commitment_remains_required_for_current_attempt": True,
            "historical_config_commitment_allowed_only_for_attempt_8": True,
            "historical_attempt_8_config_commitment_sha256": "490f63e294d4751a962d41c1956b02e1ac51bafc8a42aed2f76aa8c8704b0c89",
            "historical_attempt_8_engineering_health_commitment_sha256": "5df1d3e169f17ecbd269e7dd8c29fc0b033a301fb6cd15e69eebecccdceb92c8",
            "attempt_config_and_health_commitments_must_match_sealed_attempt_8_result": True,
            "other_historical_or_unbound_commitments_rejected": True,
            "model_fixture_source_threshold_partition_seed_metric_or_gate_changed": False,
        }
        or comparison.get("checked_on") != "2026-08-05"
        or comparison.get("real_job_submitted_by_checks") is not False
        or comparison.get("request")
        != {
            "nodes": 1,
            "tasks": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes": 60,
            "DDP": False,
            "direct_monetary_cost_USD": 0,
        }
        or not candidate_shape_ok
        or comparison.get("selection_rule")
        != "minimum scheduler-estimated start among compatible eligible zero-cost one-process requests; fixed GPU-type order only breaks exact timestamp ties"
        or selected_type != winner.get("GPU_type")
        or active != expected_active
        or topology != expected_topology
        or execution
        != {
            "attempt_10_is_complete_health_suite_not_shallow_diagnostic": True,
            "same_complete_28_case_two_replicate_suite": True,
            "metric_withholding_on_timeout_or_error": True,
            "commit_and_push_before_submission": True,
            "one_live_job_max": 1,
            "no_repeated_scheduler_polling_or_unchanged_status_updates": True,
            "valid_complete_scientific_result_remains_terminal_under_frozen_gate": True,
        }
        or value.get("new_health_or_scientific_outcome_opened") is not False
        or value.get("repair_commitment_scope")
        != "canonical JSON of this amendment excluding repair_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_HISTORICAL_LINEAGE_REPAIR_COMMITMENT")
    return value


def _engineering_health_attempt_10_result(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the metric-withheld cross-Python AST engineering failure."""

    lineage = _engineering_health_historical_lineage_repair(cfg)
    try:
        value = cfg["learner_effective_engineering_health_attempt_10_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_10_RESULT_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    if (
        cfg.get("schema_version") not in {33, 34, 35}
        or value.get("status")
        != "ENGINEERING_BLOCKER_ATTEMPT_10_DURING_FIXTURE_VERIFICATION_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("route_id") != "construct-aligned-engineering-health"
        or value.get("classification")
        != "ENGINEERING_CROSS_PYTHON_AST_SERIALIZATION_VARIANCE_NOT_SCIENTIFIC_NO_GO"
        or value.get("preserved_historical_lineage_repair_sha256")
        != lineage["repair_commitment_sha256"]
        or expected != ENGINEERING_HEALTH_ATTEMPT_10_BLOCKER_SHA256
        or digest(payload) != expected
        or value.get("submission_provenance")
        != {
            "job_id": 316832,
            "attempt": 10,
            "scheduler_state": "COMPLETED",
            "scheduler_exit_code": "0:0",
            "scheduler_elapsed_seconds": 244,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes_requested": 60,
            "DDP": False,
            "direct_monetary_cost_USD": 0,
        }
        or value.get("compact_aggregate")
        != {
            "module_count": 7,
            "completed_module_count": 0,
            "failed_module_count": 7,
            "case_count": 28,
            "holdout_input_count": 0,
            "scientific_metric_count": 0,
            "failure_count": 7,
            "invalid_retained_record_count": 0,
            "silent_truncation_count": 0,
            "external_call_count": 0,
            "unaccounted_failure_count": 0,
            "private_trace_count": 7,
            "preflight_blocked_trace_count": 6,
            "retained_file_count": 12,
            "retained_bytes": 4731,
            "stdout_bytes": 1355,
            "stderr_bytes": 0,
            "public_fixture_manifest_commitment_sha256": "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6",
            "runner_commitment_sha256": "9193a4c9081e3fc94559a473356cd7544f50c5e22461adb8d1d3253b1ffe5a0d",
            "config_commitment_sha256": "ca100af8a99c93941d15c08f6b93ea0530869217d4d1f2cf4fdf682a7d708a5d",
            "dependency_config_commitment_sha256": "f62c8273ab8899c65965b2b1728b0bc88ba19821751d34f200a57ef196ef3076",
            "microfixture_manifest_commitment_sha256": "981e1cf51ba9f538976aaf6cb2a806113f54656fe3a2dc4cc0095841bb6580ee",
            "engineering_health_commitment_sha256": "21f1044fd4a8a7b05f1b66d9082cfbe38a271bacfe06691cfde416628b0237d8",
        }
        or value.get("stable_aggregate_diagnosis")
        != {
            "last_persisted_stage": "FIXTURE_VERIFICATION",
            "last_persisted_stage_ordinal": 0,
            "progress_update_count": 21,
            "first_stable_error_code": "E_TUPLE_VISOR_RASTERIZATION_REPAIR_RUNNER",
            "preflight_blocked_module_count": 6,
            "host_python_version": "3.14.6",
            "container_python_version": "3.11.13",
            "exact_geometry_source_bundle_match": True,
            "legacy_python_specific_AST_bundle_match": False,
            "model_module_inference_count": 0,
            "scientific_metric_count": 0,
            "sensitive_detail_field_count": 0,
            "model_or_scientific_outcome_used": False,
        }
        or value.get("resource_accounting")
        != {
            "attempt_GPU_hours_actual": 0.06667100356684791,
            "protocol_accounted_cumulative_GPU_hours_actual": 1.3842679627074135,
            "attempt_retained_storage_GiB": 1.2861564755439758e-6,
            "protocol_accounted_cumulative_retained_storage_GiB": 8.244998753070831e-6,
            "attempt_wall_minutes_actual": 4.000260214010875,
            "protocol_accounted_cumulative_wall_minutes_actual": 83.05607776244483,
            "direct_monetary_cost_USD": 0,
        }
        or value.get("terminal_gate")
        != {
            "scientific_decision_opened": False,
            "public_development_authorized": False,
            "governed_C_authorized": False,
            "LTX_or_synthetic_learner_run": False,
            "attempt_11_authorized_only_after_prospective_portable_AST_repair_commit_and_push": True,
        }
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this result excluding blocker_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_10_RESULT_COMMITMENT")
    return value


def _engineering_health_portable_ast_repair(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the attempt-11 portable geometry-AST compatibility repair."""

    attempt_10 = _engineering_health_attempt_10_result(cfg)
    try:
        value = cfg["learner_effective_engineering_health_portable_AST_repair"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_PORTABLE_AST_REPAIR_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("repair_commitment_sha256", None)
    comparison = value.get("scheduler_only_comparison", {})
    active = value.get("active_attempt_resource_policy", {})
    topology = value.get("topology_attestation_contract", {})
    catalog = {
        "NVIDIA_A30_24GB": ("a30", "gpu:nvidia_a30:1", "NVIDIA A30", 23, 25),
        "NVIDIA_H100_NVL": ("h100", "gpu:nvidia_h100_nvl:1", "NVIDIA H100 NVL", 85, 100),
        "NVIDIA_H100_NVL_3G_47GB_MIG": ("h100", "gpu:nvidia_h100_nvl_3g.47gb:1", "NVIDIA H100 NVL MIG 3g.47gb", 45, 50),
        "NVIDIA_H200_NVL": ("h200", "gpu:nvidia_h200_nvl:1", "NVIDIA H200 NVL", 135, 145),
    }
    candidates = comparison.get("candidates")
    candidate_shape_ok = (
        isinstance(candidates, list)
        and len(candidates) == len(catalog)
        and {item.get("GPU_type") for item in candidates if isinstance(item, dict)} == set(catalog)
        and all(
            isinstance(item, dict)
            and set(item) == {"GPU_type", "partition", "GRES", "eligible", "estimated_start"}
            and item["partition"] == catalog[item["GPU_type"]][0]
            and item["GRES"] == catalog[item["GPU_type"]][1]
            and item["eligible"] is True
            and isinstance(item["estimated_start"], str)
            and re.fullmatch(r"2026-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}-0[5-7]:00", item["estimated_start"])
            is not None
            for item in candidates
        )
    )
    winner = min(
        candidates,
        key=lambda item: (item["estimated_start"], list(catalog).index(item["GPU_type"])),
    ) if candidate_shape_ok else {}
    selected_type = comparison.get("selected_GPU_type")
    selected = catalog.get(selected_type, (None, None, None, None, None))
    expected_active = {
        "attempt": 11,
        "submission_count": 1,
        "partition": selected[0],
        "GRES": selected[1],
        "GPU_type": selected_type,
        "GPU_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "DDP": False,
        "wall_minutes_max": 60,
        "GPU_hours_max": 1.0,
        "new_storage_GiB_max": 1.0,
        "prior_protocol_accounted_GPU_hours_actual": 1.3842679627074135,
        "active_aggregate_GPU_hours_max": 2.3842679627074135,
        "prior_health_run_storage_GiB_actual": 8.244998753070831e-6,
        "active_aggregate_new_storage_GiB_max": 1.000008244998753,
        "direct_monetary_cost_USD": 0,
        "historical_incomplete_attempt_wall_minutes": {"1": 15, "2": 15, "4": 15, "6": 15, "7": 15, "9": 0.16666666666666666},
        "historical_attempt_GPU_types": {
            "1": "NVIDIA_H100_NVL_3G_47GB_MIG", "2": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "3": "NVIDIA_H100_NVL_3G_47GB_MIG", "4": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "5": "NVIDIA_H100_NVL_3G_47GB_MIG", "6": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "7": "NVIDIA_H100_NVL_3G_47GB_MIG", "8": "NVIDIA_H100_NVL",
            "9": "NVIDIA_A30_24GB", "10": "NVIDIA_A30_24GB",
        },
    }
    expected_topology = {
        "partition": selected[0], "node_count": 1, "task_count": 1,
        "CPU_count": 8, "time_limit_minutes": 60, "memory_per_CPU_GiB": 4,
        "GRES": selected[1], "expected_device_name": selected[2],
        "visible_memory_GiB_min": selected[3], "visible_memory_GiB_max": selected[4],
        "world_size": 1, "local_world_size": 1,
    }
    if (
        cfg.get("schema_version") not in {33, 34, 35}
        or value.get("status") != "FROZEN_AFTER_ATTEMPT_10_BLOCKER_BEFORE_PORTABLE_AST_REPAIRED_ATTEMPT_11_OR_NEW_OUTCOME"
        or value.get("scope") != "OUTCOME_INDEPENDENT_CROSS_PYTHON_GEOMETRY_AST_COMPATIBILITY_REPAIR_AND_ATTEMPT_11_TOPOLOGY_SELECTION"
        or expected != ENGINEERING_HEALTH_PORTABLE_AST_REPAIR_SHA256
        or digest(payload) != expected
        or value.get("preserved_without_change")
        != {
            "attempt_10_blocker_sha256": attempt_10["blocker_commitment_sha256"],
            "attempt_10_engineering_health_commitment_sha256": "21f1044fd4a8a7b05f1b66d9082cfbe38a271bacfe06691cfde416628b0237d8",
            "historical_lineage_repair_sha256": ENGINEERING_HEALTH_HISTORICAL_LINEAGE_REPAIR_SHA256,
            "historical_geometry_repair_commitment_sha256": "6084fd937c208feda00aa3dc1cf14d0ec56e8f13bd24b56e23e4a6a6553e61ef",
            "exact_geometry_source_bundle_sha256": "a7249fc9302cd1c04335b6d8dcde9c1410f3536cd55ddd6cf4532f36616189b4",
            "models_weights_fixtures_inputs_seeds_thresholds_metrics_scientific_gates_privacy_rules_and_downstream_contract": True,
        }
        or value.get("failure_specific_diagnosis")
        != {
            "host_python_version": "3.14.6", "container_python_version": "3.11.13",
            "exact_source_bundle_match": True, "legacy_AST_dump_match": False,
            "legacy_AST_dump_is_Python_version_dependent": True,
            "model_module_inference_count": 0, "scientific_metric_count": 0,
            "outcome_information_used_for_repair": False,
        }
        or value.get("failure_specific_repair")
        != {
            "portable_AST_projection_includes_node_types_and_fields": True,
            "portable_AST_projection_excludes_only_version_specific_type_params": True,
            "portable_AST_projection_excludes_locations": True,
            "portable_AST_bundle_sha256": "74d7707cd8d485aa00514f7f216759f848893923e990f94e3a1d19b935958b8d",
            "exact_source_bundle_remains_blocking": True,
            "historical_attempts_8_and_10_require_sealed_config_and_health_commitment_pairs": True,
            "other_historical_or_unbound_commitments_rejected": True,
            "model_fixture_source_threshold_partition_seed_metric_or_gate_changed": False,
        }
        or comparison.get("checked_on") != "2026-08-05"
        or comparison.get("real_job_submitted_by_checks") is not False
        or comparison.get("request") != {"nodes": 1, "tasks": 1, "CPU_count": 8, "memory_GiB": 32, "wall_minutes": 60, "DDP": False, "direct_monetary_cost_USD": 0}
        or not candidate_shape_ok
        or comparison.get("selection_rule") != "minimum scheduler-estimated start among compatible eligible zero-cost one-process requests; fixed GPU-type order only breaks exact timestamp ties"
        or selected_type != winner.get("GPU_type")
        or active != expected_active
        or topology != expected_topology
        or value.get("execution_and_stop_rule")
        != {
            "attempt_11_is_complete_health_suite_not_shallow_diagnostic": True,
            "same_complete_28_case_two_replicate_suite": True,
            "metric_withholding_on_timeout_or_error": True,
            "commit_and_push_before_submission": True,
            "one_live_job_max": 1,
            "no_repeated_scheduler_polling_or_unchanged_status_updates": True,
            "valid_complete_scientific_result_remains_terminal_under_frozen_gate": True,
        }
        or value.get("new_health_or_scientific_outcome_opened") is not False
        or value.get("repair_commitment_scope") != "canonical JSON of this amendment excluding repair_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PORTABLE_AST_REPAIR_COMMITMENT")
    return value


def _engineering_health_attempt_11_result(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the sealed incomplete-active-fixture-root engineering result."""

    portable = _engineering_health_portable_ast_repair(cfg)
    try:
        value = cfg["learner_effective_engineering_health_attempt_11_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_11_RESULT_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    compact = value.get("compact_aggregate", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    if (
        cfg.get("schema_version") not in {34, 35}
        or value.get("status")
        != "ENGINEERING_BLOCKER_ATTEMPT_11_DURING_FIXTURE_VERIFICATION_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("classification")
        != "ENGINEERING_INCOMPLETE_ACTIVE_PUBLIC_FIXTURE_ROOT_NOT_SCIENTIFIC_NO_GO"
        or value.get("preserved_portable_AST_repair_sha256")
        != portable["repair_commitment_sha256"]
        or expected != ENGINEERING_HEALTH_ATTEMPT_11_BLOCKER_SHA256
        or digest(payload) != expected
        or value.get("submission_provenance", {}).get("job_id") != 316845
        or value.get("submission_provenance", {}).get("attempt") != 11
        or value.get("submission_provenance", {}).get("scheduler_state")
        != "COMPLETED"
        or compact.get("module_count") != 7
        or compact.get("completed_module_count") != 0
        or compact.get("failed_module_count") != 7
        or compact.get("case_count") != 28
        or compact.get("scientific_metric_count") != 0
        or compact.get("unaccounted_failure_count") != 1
        or compact.get("engineering_health_commitment_sha256")
        != "2e409529a3de9a1536915e776f14d8ffa8e9f35aec88929d54c0a46d8b5debc4"
        or diagnosis.get("first_stable_root_cause_code")
        != "E_VISOR_HOS_SOURCE_FEASIBILITY_MISSING"
        or diagnosis.get("historical_full_fixture_verification_passed_with_current_runner_and_pinned_offline_container")
        is not True
        or diagnosis.get("model_module_inference_count") != 0
        or diagnosis.get("scientific_metric_count") != 0
        or resource.get("protocol_accounted_cumulative_GPU_hours_actual")
        != 1.4459219645791583
        or value.get("terminal_gate", {}).get("scientific_decision_opened")
        is not False
        or value.get("terminal_gate", {}).get("attempt_12_authorized_only_after_prospective_read_only_fixture_bind_repair_commit_and_push")
        is not True
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this result excluding blocker_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_11_RESULT_COMMITMENT")
    return value


def _engineering_health_fixture_bind_repair(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the read-only sealed-fixture bind and attempt-12 topology."""

    attempt_11 = _engineering_health_attempt_11_result(cfg)
    try:
        value = cfg["learner_effective_engineering_health_fixture_bind_repair"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_BIND_REPAIR_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("repair_commitment_sha256", None)
    comparison = value.get("scheduler_only_comparison", {})
    candidates = comparison.get("candidates")
    catalog = {
        "NVIDIA_A30_24GB": ("a30", "gpu:nvidia_a30:1", "NVIDIA A30", 23, 25),
        "NVIDIA_H100_NVL": ("h100", "gpu:nvidia_h100_nvl:1", "NVIDIA H100 NVL", 85, 100),
        "NVIDIA_H100_NVL_3G_47GB_MIG": ("h100", "gpu:nvidia_h100_nvl_3g.47gb:1", "NVIDIA H100 NVL MIG 3g.47gb", 45, 50),
        "NVIDIA_H200_NVL": ("h200", "gpu:nvidia_h200_nvl:1", "NVIDIA H200 NVL", 135, 145),
    }
    candidate_shape_ok = (
        isinstance(candidates, list)
        and len(candidates) == 4
        and {item.get("GPU_type") for item in candidates if isinstance(item, dict)}
        == set(catalog)
        and all(
            isinstance(item, dict)
            and set(item)
            == {"GPU_type", "partition", "GRES", "eligible", "estimated_start"}
            and item["partition"] == catalog[item["GPU_type"]][0]
            and item["GRES"] == catalog[item["GPU_type"]][1]
            and item["eligible"] is True
            and re.fullmatch(
                r"2026-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}-0[5-7]:00",
                str(item["estimated_start"]),
            )
            is not None
            for item in candidates
        )
    )
    winner = (
        min(
            candidates,
            key=lambda item: (
                item["estimated_start"],
                list(catalog).index(item["GPU_type"]),
            ),
        )
        if candidate_shape_ok
        else {}
    )
    selected_type = comparison.get("selected_GPU_type")
    selected = catalog.get(selected_type, (None, None, None, None, None))
    active = value.get("active_attempt_resource_policy", {})
    topology = value.get("topology_attestation_contract", {})
    repair = value.get("failure_specific_repair", {})
    expected_active = {
        "attempt": 12,
        "submission_count": 1,
        "partition": selected[0],
        "GRES": selected[1],
        "GPU_type": selected_type,
        "GPU_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "DDP": False,
        "wall_minutes_max": 60,
        "GPU_hours_max": 1.0,
        "new_storage_GiB_max": 1.0,
        "prior_protocol_accounted_GPU_hours_actual": 1.4459219645791583,
        "active_aggregate_GPU_hours_max": 2.4459219645791583,
        "prior_health_run_storage_GiB_actual": 9.519048035144806e-6,
        "active_aggregate_new_storage_GiB_max": 1.0000095190480351,
        "direct_monetary_cost_USD": 0,
        "historical_incomplete_attempt_wall_minutes": {
            "1": 15,
            "2": 15,
            "4": 15,
            "6": 15,
            "7": 15,
            "9": 0.16666666666666666,
        },
        "historical_attempt_GPU_types": {
            "1": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "2": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "3": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "4": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "5": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "6": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "7": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "8": "NVIDIA_H100_NVL",
            "9": "NVIDIA_A30_24GB",
            "10": "NVIDIA_A30_24GB",
            "11": "NVIDIA_A30_24GB",
        },
    }
    expected_topology = {
        "partition": selected[0],
        "node_count": 1,
        "task_count": 1,
        "CPU_count": 8,
        "time_limit_minutes": 60,
        "memory_per_CPU_GiB": 4,
        "GRES": selected[1],
        "expected_device_name": selected[2],
        "visible_memory_GiB_min": selected[3],
        "visible_memory_GiB_max": selected[4],
        "world_size": 1,
        "local_world_size": 1,
    }
    if (
        cfg.get("schema_version") not in {34, 35}
        or value.get("status")
        != "FROZEN_AFTER_ATTEMPT_11_BLOCKER_BEFORE_READ_ONLY_SEALED_FIXTURE_BIND_ATTEMPT_12_OR_NEW_OUTCOME"
        or value.get("scope")
        != "OUTCOME_INDEPENDENT_SEALED_PUBLIC_FIXTURE_ROOT_READ_ONLY_BIND_REPAIR_AND_ATTEMPT_12_TOPOLOGY_SELECTION"
        or expected != ENGINEERING_HEALTH_FIXTURE_BIND_REPAIR_SHA256
        or digest(payload) != expected
        or value.get("preserved_without_change", {}).get("attempt_11_blocker_sha256")
        != attempt_11["blocker_commitment_sha256"]
        or value.get("preserved_without_change", {}).get("public_fixture_manifest_sha256")
        != "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
        or repair.get("bind_mode") != "READ_ONLY"
        or repair.get("active_partial_fixture_tree_action")
        != "SHADOW_ONLY_NO_DELETE_NO_OVERWRITE_NO_COPY"
        or repair.get("fixture_source_root_path_recorded_in_Git_or_compact_output")
        is not False
        or repair.get("full_manifest_and_all_referenced_fixture_hashes_validated_with_current_runner_and_pinned_offline_container")
        is not True
        or repair.get("model_fixture_source_threshold_partition_seed_metric_or_gate_changed")
        is not False
        or comparison.get("checked_on") != "2026-08-05"
        or comparison.get("real_job_submitted_by_checks") is not False
        or not candidate_shape_ok
        or selected_type != winner.get("GPU_type")
        or active != expected_active
        or topology != expected_topology
        or value.get("new_health_or_scientific_outcome_opened") is not False
        or value.get("repair_commitment_scope")
        != "canonical JSON of this amendment excluding repair_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_BIND_REPAIR_COMMITMENT")
    return value


def _engineering_health_attempt_12_result(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the sealed pre-runner submission-export failure."""

    fixture_bind = _engineering_health_fixture_bind_repair(cfg)
    try:
        value = cfg["learner_effective_engineering_health_attempt_12_result"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_12_RESULT_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("blocker_commitment_sha256", None)
    submission = value.get("submission_provenance", {})
    diagnosis = value.get("stable_aggregate_diagnosis", {})
    resource = value.get("resource_accounting", {})
    if (
        cfg.get("schema_version") != 35
        or value.get("status")
        != "ENGINEERING_BLOCKER_ATTEMPT_12_BEFORE_RUNNER_ENTRY_NO_SCIENTIFIC_METRICS_OPENED"
        or value.get("classification")
        != "ENGINEERING_SUBMISSION_EXPORT_CONTRACT_FAILURE_NOT_SCIENTIFIC_NO_GO"
        or value.get("preserved_fixture_bind_repair_sha256")
        != fixture_bind["repair_commitment_sha256"]
        or expected != ENGINEERING_HEALTH_ATTEMPT_12_BLOCKER_SHA256
        or digest(payload) != expected
        or submission.get("job_id") != 316878
        or submission.get("attempt") != 12
        or submission.get("scheduler_state") != "FAILED"
        or submission.get("scheduler_exit_code") != "64:0"
        or submission.get("scheduler_elapsed_seconds") != 3
        or submission.get("submission_started_epoch") != 1785912624.0
        or diagnosis.get("wrapper_exit_code") != 64
        or diagnosis.get("required_export_field_count") != 9
        or diagnosis.get("present_required_export_field_count") != 8
        or diagnosis.get("missing_required_export_field_count") != 1
        or diagnosis.get("missing_export_field")
        != "PHASE4_HEALTH_WALL_MINUTES"
        or diagnosis.get("attempt_root_created") is not False
        or diagnosis.get("runner_entry_count") != 0
        or diagnosis.get("module_execution_count") != 0
        or diagnosis.get("scientific_metric_count") != 0
        or diagnosis.get("log_file_count") != 2
        or diagnosis.get("log_bytes") != 0
        or resource.get("protocol_accounted_cumulative_GPU_hours_actual")
        != 1.4467552979124916
        or value.get("terminal_gate", {}).get("scientific_decision_opened")
        is not False
        or value.get("terminal_gate", {}).get(
            "attempt_13_authorized_only_after_prospective_export_contract_repair_commit_and_push"
        )
        is not True
        or value.get("blocker_commitment_scope")
        != "canonical JSON of this result excluding blocker_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_12_RESULT_COMMITMENT")
    return value


def _engineering_health_submission_export_repair(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact missing-export repair and attempt-13 topology."""

    attempt_12 = _engineering_health_attempt_12_result(cfg)
    try:
        value = cfg[
            "learner_effective_engineering_health_submission_export_repair"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_SUBMISSION_EXPORT_REPAIR_MISSING") from error
    payload = json.loads(json.dumps(value))
    expected = payload.pop("repair_commitment_sha256", None)
    comparison = value.get("scheduler_only_comparison", {})
    candidates = comparison.get("candidates")
    catalog = {
        "NVIDIA_A30_24GB": ("a30", "gpu:nvidia_a30:1", "NVIDIA A30", 23, 25),
        "NVIDIA_H100_NVL": ("h100", "gpu:nvidia_h100_nvl:1", "NVIDIA H100 NVL", 85, 100),
        "NVIDIA_H100_NVL_3G_47GB_MIG": ("h100", "gpu:nvidia_h100_nvl_3g.47gb:1", "NVIDIA H100 NVL MIG 3g.47gb", 45, 50),
        "NVIDIA_H200_NVL": ("h200", "gpu:nvidia_h200_nvl:1", "NVIDIA H200 NVL", 135, 145),
    }
    candidate_shape_ok = (
        isinstance(candidates, list)
        and len(candidates) == 4
        and {item.get("GPU_type") for item in candidates if isinstance(item, dict)}
        == set(catalog)
        and all(
            isinstance(item, dict)
            and set(item)
            == {"GPU_type", "partition", "GRES", "eligible", "estimated_start"}
            and item["partition"] == catalog[item["GPU_type"]][0]
            and item["GRES"] == catalog[item["GPU_type"]][1]
            and item["eligible"] is True
            and re.fullmatch(
                r"2026-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}-0[5-7]:00",
                str(item["estimated_start"]),
            )
            is not None
            for item in candidates
        )
    )
    winner = (
        min(
            candidates,
            key=lambda item: (
                item["estimated_start"],
                list(catalog).index(item["GPU_type"]),
            ),
        )
        if candidate_shape_ok
        else {}
    )
    selected_type = comparison.get("selected_GPU_type")
    selected = catalog.get(selected_type, (None, None, None, None, None))
    active = value.get("active_attempt_resource_policy", {})
    topology = value.get("topology_attestation_contract", {})
    repair = value.get("failure_specific_repair", {})
    expected_active = {
        "attempt": 13,
        "submission_count": 1,
        "partition": selected[0],
        "GRES": selected[1],
        "GPU_type": selected_type,
        "GPU_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "DDP": False,
        "wall_minutes_max": 60,
        "GPU_hours_max": 1.0,
        "new_storage_GiB_max": 1.0,
        "prior_protocol_accounted_GPU_hours_actual": 1.4467552979124916,
        "active_aggregate_GPU_hours_max": 2.4467552979124916,
        "prior_health_run_storage_GiB_actual": 0.000009519048035144806,
        "active_aggregate_new_storage_GiB_max": 1.0000095190480351,
        "direct_monetary_cost_USD": 0,
        "historical_incomplete_attempt_wall_minutes": {
            "1": 15,
            "2": 15,
            "4": 15,
            "6": 15,
            "7": 15,
            "9": 0.16666666666666666,
            "12": 0.05,
        },
        "historical_attempt_GPU_types": {
            "1": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "2": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "3": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "4": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "5": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "6": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "7": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "8": "NVIDIA_H100_NVL",
            "9": "NVIDIA_A30_24GB",
            "10": "NVIDIA_A30_24GB",
            "11": "NVIDIA_A30_24GB",
            "12": "NVIDIA_A30_24GB",
        },
    }
    expected_topology = {
        "partition": selected[0],
        "node_count": 1,
        "task_count": 1,
        "CPU_count": 8,
        "time_limit_minutes": 60,
        "memory_per_CPU_GiB": 4,
        "GRES": selected[1],
        "expected_device_name": selected[2],
        "visible_memory_GiB_min": selected[3],
        "visible_memory_GiB_max": selected[4],
        "world_size": 1,
        "local_world_size": 1,
    }
    if (
        cfg.get("schema_version") != 35
        or value.get("status")
        != "FROZEN_AFTER_ATTEMPT_12_EXPORT_CONTRACT_FAILURE_BEFORE_ATTEMPT_13_OR_NEW_OUTCOME"
        or value.get("scope")
        != "OUTCOME_INDEPENDENT_SUBMISSION_EXPORT_CONTRACT_REPAIR_AND_ATTEMPT_13_TOPOLOGY_SELECTION"
        or expected != ENGINEERING_HEALTH_SUBMISSION_EXPORT_REPAIR_SHA256
        or digest(payload) != expected
        or value.get("preserved_without_change", {}).get("attempt_12_blocker_sha256")
        != attempt_12["blocker_commitment_sha256"]
        or repair.get("required_export_field") != "PHASE4_HEALTH_WALL_MINUTES"
        or repair.get("required_export_value") != "60"
        or repair.get("same_wrapper_preexisting_validation_retained") is not True
        or repair.get("submission_command_regression_test_required") is not True
        or repair.get("model_fixture_source_threshold_partition_seed_metric_or_gate_changed")
        is not False
        or comparison.get("checked_on") != "2026-08-05"
        or comparison.get("real_job_submitted_by_checks") is not False
        or not candidate_shape_ok
        or selected_type != winner.get("GPU_type")
        or active != expected_active
        or topology != expected_topology
        or value.get("new_health_or_scientific_outcome_opened") is not False
        or value.get("repair_commitment_scope")
        != "canonical JSON of this amendment excluding repair_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_SUBMISSION_EXPORT_REPAIR_COMMITMENT")
    return value


def _geometry_function_bundle_digests(
    path: Path, names: list[str]
) -> tuple[str, str]:
    """Hash the exact historical geometry implementation independently."""

    text = path.read_text()
    tree = ast.parse(text)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in set(names)
    ]
    if {node.name for node in selected} != set(names):
        raise RuntimeError("E_TUPLE_VISOR_GEOMETRY_BUNDLE_FUNCTION_SET")
    lines = text.splitlines(keepends=True)
    source = "\n".join(
        "".join(lines[node.lineno - 1 : node.end_lineno]) for node in selected
    ).encode()
    canonical_ast = "\n".join(
        ast.dump(node, include_attributes=False) for node in selected
    ).encode()
    return hashlib.sha256(source).hexdigest(), hashlib.sha256(canonical_ast).hexdigest()


def _portable_geometry_function_bundle_digests(
    path: Path, names: list[str]
) -> tuple[str, str]:
    """Hash exact source and a Python-version-stable AST projection."""

    text = path.read_text()
    tree = ast.parse(text)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in set(names)
    ]
    if {node.name for node in selected} != set(names):
        raise RuntimeError("E_TUPLE_VISOR_GEOMETRY_BUNDLE_FUNCTION_SET")
    lines = text.splitlines(keepends=True)
    source = "\n".join(
        "".join(lines[node.lineno - 1 : node.end_lineno]) for node in selected
    ).encode()

    def project(value: Any) -> Any:
        if isinstance(value, ast.AST):
            return {
                "node": type(value).__name__,
                "fields": {
                    field: project(getattr(value, field))
                    for field in value._fields
                    if field != "type_params"
                },
            }
        if isinstance(value, list):
            return [project(item) for item in value]
        if isinstance(value, tuple):
            return [project(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise RuntimeError("E_TUPLE_VISOR_GEOMETRY_PORTABLE_AST_VALUE")

    portable = [project(node) for node in selected]
    return hashlib.sha256(source).hexdigest(), digest(portable)


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


def _tuple_visor_hos_correction_amendment(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    try:
        value = cfg["calibration_C"]["extractor"][
            "mechanistic_training_tuple_visor_hos_correction_amendment"
        ]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_VISOR_HOS_CORRECTION_NOT_FROZEN") from error
    if value.get("status") != (
        "FROZEN_BEFORE_NEW_PUBLIC_SOURCE_INVENTORY_MODEL_C_GENERATOR_OR_LEARNER_OUTCOMES"
    ):
        raise RuntimeError("E_VISOR_HOS_CORRECTION_NOT_FROZEN")
    copy = json.loads(json.dumps(value))
    expected = copy.pop("amendment_commitment_sha256", None)
    if not isinstance(expected, str) or digest(copy) != expected:
        raise RuntimeError("E_VISOR_HOS_CORRECTION_COMMITMENT")
    artifact = value.get("official_annotation_artifact", {})
    semantic_reference = value.get("official_semantic_reference", {})
    sampler = value.get("partition_and_joint_sampler", {})
    counts = value.get("public_fixture_counts_per_partition", {})
    gate = value.get("public_execution_and_combined_gate", {})
    execution = value.get("qualification_execution_clarification", {})
    truth = value.get("truth_contract", {})
    if (
        artifact.get("combined_JSON_file_count") != 158
        or artifact.get("combined_bytes") != 868821446
        or semantic_reference.get("gen_coco_format_py_sha256")
        != "686a052c8676c8378438efcf90e97e71cd6abca576381b0ca560e6cb07759cd7"
        or semantic_reference.get("gen_coco_format_handside_contact_py_sha256")
        != "44feea718164ed171ee6cb24eb90cde402e429eb920c4ce728c00492b79084f6"
        or sampler.get("seed") != 20260802
        or sampler.get("partitions") != ["development", "holdout"]
        or sampler.get("quota_per_partition_per_stratum") != 48
        or sampler.get("final_item_count_per_partition") != 144
        or sampler.get("per_video_per_stratum_cap") != 4
        or sampler.get("per_source_frame_cap_across_strata") != 1
        or "visor_hos_joint" not in str(sampler.get("candidate_order", ""))
        or counts.get("hand_contact_items") != 144
        or counts.get("total") != 416
        or truth.get("boolean_values_are_invalid") is not True
        or "visually verifies" not in str(truth.get("no_hand_verification", ""))
        or execution.get("status") != "FROZEN_BEFORE_PUBLIC_DEVELOPMENT_INFERENCE"
        or execution.get("scientific_thresholds_changed") is not False
        or execution.get("phase_aggregation", {}).get("requested_samples") != 9
        or execution.get("phase_aggregation", {}).get("minimum_valid_samples") != 8
        or execution.get("grid_selection", {}).get("action", {}).get(
            "abstention_margin_grid"
        )
        != [0.0, 0.005, 0.01, 0.02, 0.05]
        or execution.get("grounding_geometry", {}).get("same_category_NMS_IoU")
        != 0.5
        or execution.get("pHash", {}).get("near_duplicate_hamming_distance_max")
        != 4
        or execution.get("recurrence_pixels", {}).get("full_canvas_unmasked_proxy")
        != "PROHIBITED"
    ):
        raise RuntimeError("E_VISOR_HOS_CORRECTION_SCHEMA")
    if gate.get("critical_axes") != list(TUPLE_CRITICAL_AXIS_IDS) or gate.get(
        "supporting_axes"
    ) != list(TUPLE_SUPPORTING_AXIS_IDS):
        raise RuntimeError("E_VISOR_HOS_CORRECTION_COMBINED_GATE")
    if (
        gate.get("one_supporting_axis_may_be_unmeasured") is not True
        or gate.get("broad_activity_context") != "DESCRIPTIVE_NONBLOCKING"
        or "all five critical axes" not in str(gate.get("combined_pass_rule", ""))
        or "at least six of seven" not in str(gate.get("combined_pass_rule", ""))
    ):
        raise RuntimeError("E_VISOR_HOS_CORRECTION_COMBINED_GATE")
    return value


def _public_fixture_geometry_rasterization_repair(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate the prospective VISOR polygon-to-mask engineering repair."""

    try:
        value = cfg["public_fixture_geometry_rasterization_repair"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_VISOR_RASTERIZATION_REPAIR_NOT_FROZEN") from error
    if value.get("status") != (
        "FROZEN_AFTER_PRESEAL_ENGINEERING_STOP_BEFORE_FIXTURE_OR_MODEL_OUTCOME"
    ):
        raise RuntimeError("E_TUPLE_VISOR_RASTERIZATION_REPAIR_NOT_FROZEN")
    payload = json.loads(json.dumps(value))
    expected = payload.pop("repair_commitment_sha256", None)
    if not isinstance(expected, str) or digest(payload) != expected:
        raise RuntimeError("E_TUPLE_VISOR_RASTERIZATION_REPAIR_COMMITMENT")
    current_runner = file_digest(Path(__file__))
    if current_runner != value.get("current_runner_source_sha256"):
        health = _engineering_health_amendment(cfg)
        compatibility = health.get("historical_geometry_compatibility", {})
        if (
            compatibility.get("historical_runner_sha256")
            != value.get("current_runner_source_sha256")
            or compatibility.get("historical_geometry_repair_commitment_sha256")
            != expected
        ):
            raise RuntimeError("E_TUPLE_VISOR_RASTERIZATION_REPAIR_RUNNER")
        if "learner_effective_engineering_health_portable_AST_repair" in cfg:
            portable = _engineering_health_portable_ast_repair(cfg)[
                "failure_specific_repair"
            ]
            source_sha256, ast_sha256 = (
                _portable_geometry_function_bundle_digests(
                    Path(__file__), compatibility.get("function_names", [])
                )
            )
            expected_ast = portable["portable_AST_bundle_sha256"]
        else:
            source_sha256, ast_sha256 = _geometry_function_bundle_digests(
                Path(__file__), compatibility.get("function_names", [])
            )
            expected_ast = compatibility.get("canonical_AST_bundle_sha256")
        if (
            source_sha256
            != compatibility.get("exact_function_source_bundle_sha256")
            or ast_sha256 != expected_ast
        ):
            raise RuntimeError("E_TUPLE_VISOR_RASTERIZATION_REPAIR_RUNNER")
    semantics = value.get("rasterization_semantics", {})
    if (
        semantics.get("numpy_distribution") != "numpy==1.26.4"
        or semantics.get("opencv_distribution")
        != "opencv-python-headless==4.10.0.84"
        or semantics.get("opencv_module_version") != "4.10.0"
        or semantics.get("manual_clip") is not False
        or semantics.get("pixel_tolerance") is not None
        or semantics.get("PIL_rasterization_fallback") is not False
        or value.get("scientific_thresholds_changed") is not False
        or value.get("source_selection_changed") is not False
    ):
        raise RuntimeError("E_TUPLE_VISOR_RASTERIZATION_REPAIR_SCOPE")
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
    """Stage the exact pinned archives; do not depend on a moved run manifest."""

    manifest_path = _tuple_run_root(public) / "dependency_manifest.json"
    expected = cfg["calibration_C"]["extractor"][
        "mechanistic_training_tuple_premodel_result"
    ]["dependency_manifest_commitment_sha256"]
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("E_TUPLE_NLTK_MANIFEST")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        commitment = manifest.pop("tuple_dependency_commitment_sha256", None)
        if commitment != expected or digest(manifest) != expected:
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
    _tuple_amendment(cfg)
    target = scratch / "nltk_data"
    if target.exists():
        raise RuntimeError("E_TUPLE_NLTK_STAGE_ALREADY_EXISTS")
    archives = _tuple_model_root(public) / "nltk-archives"
    destinations = {
        "wordnet.zip": target / "corpora",
        "averaged_perceptron_tagger_eng.zip": target / "taggers",
    }
    for name, destination in destinations.items():
        archive = archives / name
        record = NLTK_RESOURCE_ARCHIVES[name]
        if not archive.is_file() or file_digest(archive) != record["sha256"]:
            raise RuntimeError("E_TUPLE_NLTK_RESOURCE_HASH")
        _safe_extract_zip(archive, destination)
    required = {
        target / "corpora/wordnet",
        target / "taggers/averaged_perceptron_tagger_eng",
    }
    if any(
        not path.is_dir() or not any(item.is_file() for item in path.rglob("*"))
        for path in required
    ):
        raise RuntimeError("E_TUPLE_NLTK_RESOURCE_SET")
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


# The functions above intentionally retain the fixture semantics used by the
# sealed annotation-only source no-go.  The prospective VISOR-HOS correction
# uses the isolated helpers below so that the earlier result remains
# reproducible and cannot silently acquire the repaired label semantics.
VISOR_HOS_STRATA = (
    "contact",
    "explicit_no_contact",
    "verified_no_hand",
)
VISOR_HOS_GLOVE_NAMES = frozenset(
    {
        "oven glove",
        "gloves",
        "rubber glove",
        "left glove",
        "right glove",
        "glove",
    }
)
TUPLE_CRITICAL_AXIS_IDS = (
    "adapter_qualified_yield",
    "noun_adjective_exposure",
    "utterance_centered_referent_visibility_dominance_ambiguity",
    "cross_episode_recurrence",
    "adjective_attribute_contrast",
)
TUPLE_SUPPORTING_AXIS_IDS = (
    "hand_action_coupling",
    "egocentric_sensor_regime",
)


def _visor_hos_normalized_identifier(value: Any) -> str | None:
    """Normalize an official annotation ID without accepting JSON booleans."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return str(int(value))
    if isinstance(value, str) and value and value == value.strip():
        return value
    return None


def _visor_hos_contact_truth(
    hand: dict[str, Any], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Decode the explicit VISOR-HOS contact state for one visible hand.

    A resolved object annotation ID is contact and the one exact official
    negative sentinel is no-contact.  Missing, null, boolean, unresolved, and
    official ambiguous relations abstain rather than manufacturing a label.
    """

    if str(hand.get("name", "")) not in {"left hand", "right hand"}:
        return {"status": "ABSTAIN", "reason": "NOT_VISIBLE_HAND"}
    if not _valid_visor_segments(hand.get("segments")):
        return {"status": "ABSTAIN", "reason": "INVALID_HAND_MASK"}
    if "in_contact_object" not in hand:
        return {"status": "ABSTAIN", "reason": "MISSING_CONTACT_LABEL"}
    relation = hand["in_contact_object"]
    if isinstance(relation, bool):
        return {"status": "ABSTAIN", "reason": "BOOLEAN_CONTACT_LABEL"}
    if relation == "hand-not-in-contact":
        return {
            "status": "MEASURED",
            "reason": None,
            "contact_state": "no_contact",
            "contact": False,
        }
    if relation is None or relation in {
        "",
        "none",
        "null",
        "none-of-the-above",
        "inconclusive",
    }:
        return {"status": "ABSTAIN", "reason": "AMBIGUOUS_CONTACT_LABEL"}
    relation_id = _visor_hos_normalized_identifier(relation)
    if relation_id is None:
        return {"status": "ABSTAIN", "reason": "INVALID_CONTACT_OBJECT_ID"}
    matches = [
        item
        for item in annotations
        if _visor_hos_normalized_identifier(item.get("id")) == relation_id
    ]
    if len(matches) != 1:
        return {"status": "ABSTAIN", "reason": "UNRESOLVED_CONTACT_OBJECT_ID"}
    if not _valid_visor_segments(matches[0].get("segments")):
        return {"status": "ABSTAIN", "reason": "INVALID_CONTACT_OBJECT_MASK"}
    return {
        "status": "MEASURED",
        "reason": None,
        "contact_state": "contact",
        "contact": True,
    }


def _visor_hos_frame_candidates(row: dict[str, Any]) -> dict[str, Any]:
    """Return independent presence/contact candidates for one VISOR-HOS frame."""

    image = row.get("image")
    annotations = row.get("annotations")
    if not isinstance(image, dict) or not isinstance(annotations, list):
        return {"status": "INVALID", "reason": "INVALID_FRAME_RECORD"}
    name = str(image.get("name", ""))
    video = str(image.get("video", ""))
    path = str(image.get("image_path", ""))
    participant = video.split("_", 1)[0]
    if (
        not name
        or not video
        or not re.fullmatch(r"P\d{2}", participant)
        or Path(name).name != name
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        return {"status": "INVALID", "reason": "INVALID_FRAME_IDENTITY"}
    if any(
        not isinstance(item, dict)
        or not _valid_visor_segments(item.get("segments"))
        for item in annotations
    ):
        return {"status": "INVALID", "reason": "INVALID_ANNOTATION_MASK"}
    hands = [
        (ordinal, item)
        for ordinal, item in enumerate(annotations)
        if str(item.get("name", "")) in {"left hand", "right hand"}
    ]
    on_hand_glove_present = any(
        str(item.get("name", "")) in VISOR_HOS_GLOVE_NAMES
        and item.get("on_which_hand") not in (None, [], "")
        for item in annotations
    )
    base = {
        "participant": participant,
        "video": video,
        "frame_name": name,
        "image_path": path,
    }
    source_split = image.get("_source_split")
    if source_split is not None:
        if source_split not in {"train", "val"}:
            return {"status": "INVALID", "reason": "INVALID_SOURCE_SPLIT"}
        base["source_split"] = source_split
    if not hands:
        if on_hand_glove_present:
            return {
                "status": "ABSTAIN",
                "reason": "ON_HAND_GLOVE_WITHOUT_VISIBLE_HAND",
                "candidates": [],
                "abstained_hand_count": 1,
            }
        return {
            "status": "NOMINEE",
            "reason": None,
            "candidates": [
                {
                    **base,
                    "stratum": "no_hand_nominee",
                    "hand_visible": False,
                    "contact": None,
                    "target_hand_side": None,
                    "target_hand_ordinal": None,
                    "target_hand_segments": None,
                }
            ],
            "abstained_hand_count": 0,
        }
    candidates = []
    abstained = 0
    for ordinal, hand in hands:
        truth = _visor_hos_contact_truth(hand, annotations)
        if truth["status"] != "MEASURED":
            abstained += 1
            continue
        candidates.append(
            {
                **base,
                "stratum": (
                    "contact" if truth["contact"] else "explicit_no_contact"
                ),
                "hand_visible": True,
                "contact": truth["contact"],
                "target_hand_side": str(hand["name"]),
                "target_hand_ordinal": ordinal,
                "target_hand_segments": hand["segments"],
            }
        )
    return {
        "status": "ELIGIBLE" if candidates else "ABSTAIN",
        "reason": None if candidates else "NO_CONCLUSIVE_HAND_CONTACT_STATE",
        "candidates": candidates,
        "abstained_hand_count": abstained,
    }


def _visor_hos_participant_partitions(
    participants: set[str] | list[str] | tuple[str, ...], seed: int
) -> dict[str, str]:
    """Partition participants first by a frozen SHA-256 order and alternating deal."""

    unique = sorted(set(participants))
    if any(not re.fullmatch(r"P\d{2}", item) for item in unique):
        raise RuntimeError("E_VISOR_HOS_PARTICIPANT_SET")
    ordered = sorted(
        unique,
        key=lambda item: (_fixture_order(seed, "visor_hos_participant", item), item),
    )
    return {
        participant: ("development" if ordinal % 2 == 0 else "holdout")
        for ordinal, participant in enumerate(ordered)
    }


def _visor_hos_add_flow_edge(
    graph: dict[Any, list[list[Any]]], source: Any, target: Any, capacity: int
) -> None:
    forward: list[Any] = [target, len(graph[target]), int(capacity)]
    reverse: list[Any] = [source, len(graph[source]), 0]
    graph[source].append(forward)
    graph[target].append(reverse)


def _visor_hos_max_flow(
    graph: dict[Any, list[list[Any]]], source: Any, sink: Any
) -> int:
    """Small deterministic Dinic solver for the joint fixture constraints."""

    total = 0
    while True:
        levels = {source: 0}
        queue = [source]
        for node in queue:
            for target, _reverse, capacity in graph[node]:
                if capacity > 0 and target not in levels:
                    levels[target] = levels[node] + 1
                    queue.append(target)
        if sink not in levels:
            return total
        cursors = {node: 0 for node in graph}

        def send(node: Any, available: int) -> int:
            if node == sink:
                return available
            while cursors[node] < len(graph[node]):
                edge_index = cursors[node]
                target, reverse_index, capacity = graph[node][edge_index]
                if capacity > 0 and levels.get(target) == levels[node] + 1:
                    amount = send(target, min(available, capacity))
                    if amount:
                        graph[node][edge_index][2] -= amount
                        graph[target][reverse_index][2] += amount
                        return amount
                cursors[node] += 1
            return 0

        while True:
            amount = send(source, 10**9)
            if not amount:
                break
            total += amount


def _visor_hos_frame_key_set(
    values: set[tuple[str, str]] | frozenset[tuple[str, str]] | None,
    error_code: str,
) -> set[tuple[str, str]]:
    output: set[tuple[str, str]] = set()
    for item in values or ():
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            or Path(item[1]).name != item[1]
        ):
            raise RuntimeError(error_code)
        output.add(item)
    return output


def _visor_hos_candidate_order(
    seed: int, partition: str, row: dict[str, Any]
) -> tuple[str, str, str, str, str, int]:
    side = str(row.get("target_hand_side") or "none")
    return (
        _fixture_order(
            seed,
            "visor_hos_joint",
            partition,
            row["stratum"],
            row["video"],
            row["frame_name"],
            side,
        ),
        row["stratum"],
        row["video"],
        row["frame_name"],
        side,
        int(row.get("target_hand_ordinal") or 0),
    )


def _visor_hos_joint_sampler(
    annotation_documents: list[dict[str, Any]],
    *,
    seed: int,
    target_per_stratum: int = 48,
    per_video_stratum_cap: int = 4,
    verified_no_hand_frames: set[tuple[str, str]]
    | frozenset[tuple[str, str]]
    | None = None,
    correction_excluded_frame_keys: set[tuple[str, str]]
    | frozenset[tuple[str, str]]
    | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Jointly allocate explicit HOS strata without source-order dependence.

    Raw sparse frames with no hand annotation are nominees only.  They enter
    the specificity stratum solely when their exact public frame key appears
    in the independent, pre-inference verification set.
    """

    if target_per_stratum <= 0 or per_video_stratum_cap <= 0:
        raise RuntimeError("E_VISOR_HOS_SAMPLER_LIMIT")
    verified_no_hand = _visor_hos_frame_key_set(
        verified_no_hand_frames, "E_VISOR_HOS_VERIFIED_NO_HAND_KEY"
    )
    correction_excluded = _visor_hos_frame_key_set(
        correction_excluded_frame_keys, "E_VISOR_HOS_CORRECTION_EXCLUSION_KEY"
    )
    participants: set[str] = set()
    parsed_all: list[dict[str, Any]] = []
    invalid_frame_count = 0
    abstained_frame_count = 0
    abstained_hand_count = 0
    for document in annotation_documents:
        rows = document.get("video_annotations")
        if not isinstance(rows, list):
            invalid_frame_count += 1
            continue
        for row in rows:
            image = row.get("image") if isinstance(row, dict) else None
            video = str(image.get("video", "")) if isinstance(image, dict) else ""
            participant = video.split("_", 1)[0]
            if re.fullmatch(r"P\d{2}", participant):
                participants.add(participant)
            result = _visor_hos_frame_candidates(row) if isinstance(row, dict) else {
                "status": "INVALID"
            }
            if result["status"] == "INVALID":
                invalid_frame_count += 1
                continue
            abstained_hand_count += int(result["abstained_hand_count"])
            if result["status"] == "ABSTAIN":
                abstained_frame_count += 1
            parsed_all.extend(result["candidates"])
    partitions = _visor_hos_participant_partitions(participants, seed)
    no_hand_nominees = [
        row for row in parsed_all if row["stratum"] == "no_hand_nominee"
    ]
    excluded_source_frames = {
        (row["video"], row["frame_name"])
        for row in parsed_all
        if (row["video"], row["frame_name"]) in correction_excluded
    }
    parsed: list[dict[str, Any]] = []
    for row in parsed_all:
        frame_key = (row["video"], row["frame_name"])
        if frame_key in correction_excluded:
            continue
        if row["stratum"] == "no_hand_nominee":
            if frame_key not in verified_no_hand:
                continue
            parsed.append({**row, "stratum": "verified_no_hand"})
        else:
            parsed.append(row)
    raw_eligible = {
        stratum: sum(row["stratum"] == stratum for row in parsed)
        for stratum in VISOR_HOS_STRATA
    }
    selected: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    post_partition: dict[str, dict[str, int]] = {}
    post_cap: dict[str, dict[str, int]] = {}
    final: dict[str, dict[str, int]] = {}
    deficits: list[dict[str, Any]] = []
    for partition in ("development", "holdout"):
        candidates = [
            row for row in parsed if partitions[row["participant"]] == partition
        ]
        post_partition[partition] = {
            stratum: sum(row["stratum"] == stratum for row in candidates)
            for stratum in VISOR_HOS_STRATA
        }
        by_stratum_video: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        representative: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
        for row in candidates:
            frame = (row["video"], row["frame_name"])
            stratum = row["stratum"]
            by_stratum_video[(stratum, row["video"])].add(frame)
            key = (stratum, frame)
            prior = representative.get(key)
            if prior is None or _visor_hos_candidate_order(
                seed, partition, row
            ) < _visor_hos_candidate_order(seed, partition, prior):
                representative[key] = row
        post_cap[partition] = {
            stratum: sum(
                min(len(frames), per_video_stratum_cap)
                for (item_stratum, _video), frames in by_stratum_video.items()
                if item_stratum == stratum
            )
            for stratum in VISOR_HOS_STRATA
        }

        source = ("source", partition)
        sink = ("sink", partition)
        graph: dict[Any, list[list[Any]]] = defaultdict(list)
        stratum_nodes = {stratum: ("stratum", stratum) for stratum in VISOR_HOS_STRATA}
        for stratum in sorted(VISOR_HOS_STRATA):
            _visor_hos_add_flow_edge(
                graph, source, stratum_nodes[stratum], target_per_stratum
            )
        all_frames = sorted(
            {(row["video"], row["frame_name"]) for row in candidates},
            key=lambda frame: min(
                _visor_hos_candidate_order(seed, partition, row)
                for row in candidates
                if (row["video"], row["frame_name"]) == frame
            ),
        )
        for frame in all_frames:
            _visor_hos_add_flow_edge(graph, ("frame", *frame), sink, 1)
        video_nodes: dict[tuple[str, str], Any] = {}
        for key in sorted(by_stratum_video):
            stratum, video = key
            node = ("video_stratum", stratum, video)
            video_nodes[key] = node
            _visor_hos_add_flow_edge(
                graph, stratum_nodes[stratum], node, per_video_stratum_cap
            )
            for frame in sorted(
                by_stratum_video[key],
                key=lambda item: _visor_hos_candidate_order(
                    seed, partition, representative[(stratum, item)]
                ),
            ):
                _visor_hos_add_flow_edge(graph, node, ("frame", *frame), 1)
        _visor_hos_max_flow(graph, source, sink)
        for (stratum, video), node in video_nodes.items():
            for target, _reverse, capacity in graph[node]:
                if (
                    isinstance(target, tuple)
                    and target[:1] == ("frame",)
                    and capacity == 0
                ):
                    frame = (target[1], target[2])
                    selected[partition].append(representative[(stratum, frame)])
        selected[partition].sort(
            key=lambda row: _visor_hos_candidate_order(seed, partition, row)
        )
        final[partition] = {
            stratum: sum(row["stratum"] == stratum for row in selected[partition])
            for stratum in VISOR_HOS_STRATA
        }
        for stratum in VISOR_HOS_STRATA:
            if final[partition][stratum] != target_per_stratum:
                deficits.append(
                    {
                        "partition": partition,
                        "stratum": stratum,
                        "required_count": target_per_stratum,
                        "available_count": final[partition][stratum],
                    }
                )
    development = selected["development"]
    holdout = selected["holdout"]
    development_participants = {row["participant"] for row in development}
    holdout_participants = {row["participant"] for row in holdout}
    development_videos = {row["video"] for row in development}
    holdout_videos = {row["video"] for row in holdout}
    development_frames = {(row["video"], row["frame_name"]) for row in development}
    holdout_frames = {(row["video"], row["frame_name"]) for row in holdout}
    report = {
        "status": "PASS" if not deficits else "NO_GO",
        "raw_eligible_counts": raw_eligible,
        "post_partition_counts": post_partition,
        "post_cap_counts": post_cap,
        "final_counts": final,
        "deficits": deficits,
        "invalid_frame_count": invalid_frame_count,
        "abstained_frame_count": abstained_frame_count,
        "abstained_hand_count": abstained_hand_count,
        "no_hand_nominee_count": len(no_hand_nominees),
        "verified_no_hand_input_count": len(verified_no_hand),
        "matched_verified_no_hand_count": sum(
            (row["video"], row["frame_name"]) in verified_no_hand
            and (row["video"], row["frame_name"]) not in correction_excluded
            for row in no_hand_nominees
        ),
        "unverified_no_hand_nominee_count": sum(
            (row["video"], row["frame_name"]) not in verified_no_hand
            for row in no_hand_nominees
        ),
        "correction_excluded_frame_count": len(excluded_source_frames),
        "participant_overlap_count": len(
            development_participants & holdout_participants
        ),
        "video_overlap_count": len(development_videos & holdout_videos),
        "frame_overlap_count": len(development_frames & holdout_frames),
    }
    return selected, report


VISOR_HOS_NO_HAND_REVIEW_MAX_PER_PARTITION = 192
VISOR_HOS_NO_HAND_REVIEW_ITEMS_PER_SHEET = 8
VISOR_HOS_NO_HAND_REVIEW_LABELS = frozenset(
    {"yes", "no", "abstain", ""}
)


def _visor_hos_no_hand_review_order(
    seed: int, partition: str, row: dict[str, Any]
) -> tuple[str, str, str]:
    return (
        _fixture_order(
            seed,
            "visor_hos_no_hand_review",
            partition,
            row["video"],
            row["frame_name"],
        ),
        row["video"],
        row["frame_name"],
    )


def _visor_hos_no_hand_review_token(
    seed: int, partition: str, row: dict[str, Any]
) -> str:
    return digest(
        [
            "visor_hos_no_hand_review_token",
            seed,
            partition,
            row["video"],
            row["frame_name"],
        ]
    )[:24]


def _visor_hos_no_hand_review_nominees(
    annotation_documents: list[dict[str, Any]],
    *,
    seed: int,
    per_video_cap: int,
    max_per_partition: int = VISOR_HOS_NO_HAND_REVIEW_MAX_PER_PARTITION,
    correction_excluded_frame_keys: set[tuple[str, str]]
    | frozenset[tuple[str, str]]
    | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Freeze a blinded no-hand nominee queue without accepting absence as truth."""

    if per_video_cap <= 0 or max_per_partition <= 0:
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_LIMIT")
    correction_excluded = _visor_hos_frame_key_set(
        correction_excluded_frame_keys, "E_VISOR_HOS_CORRECTION_EXCLUSION_KEY"
    )
    participants: set[str] = set()
    nominees: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_frame_count = 0
    abstained_frame_count = 0
    duplicate_nominee_count = 0
    excluded_nominee_count = 0
    for document in annotation_documents:
        rows = document.get("video_annotations")
        if not isinstance(rows, list):
            invalid_frame_count += 1
            continue
        for row in rows:
            image = row.get("image") if isinstance(row, dict) else None
            video = str(image.get("video", "")) if isinstance(image, dict) else ""
            participant = video.split("_", 1)[0]
            if re.fullmatch(r"P\d{2}", participant):
                participants.add(participant)
            parsed = _visor_hos_frame_candidates(row) if isinstance(row, dict) else {
                "status": "INVALID",
                "candidates": [],
            }
            if parsed["status"] == "INVALID":
                invalid_frame_count += 1
                continue
            if parsed["status"] == "ABSTAIN":
                abstained_frame_count += 1
                continue
            if parsed["status"] != "NOMINEE":
                continue
            candidate = parsed["candidates"][0]
            frame_key = (candidate["video"], candidate["frame_name"])
            if frame_key in correction_excluded:
                excluded_nominee_count += 1
                continue
            if frame_key in nominees:
                if canonical(nominees[frame_key]) != canonical(candidate):
                    raise RuntimeError(
                        "E_VISOR_HOS_NO_HAND_REVIEW_CONFLICTING_DUPLICATE"
                    )
                duplicate_nominee_count += 1
                continue
            nominees[frame_key] = candidate
    partitions = _visor_hos_participant_partitions(participants, seed)
    queues: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    post_partition_counts: dict[str, int] = {}
    post_cap_counts: dict[str, int] = {}
    for partition in ("development", "holdout"):
        partition_rows = sorted(
            (
                row
                for row in nominees.values()
                if partitions[row["participant"]] == partition
            ),
            key=lambda row: _visor_hos_no_hand_review_order(
                seed, partition, row
            ),
        )
        post_partition_counts[partition] = len(partition_rows)
        video_counts: Counter[str] = Counter()
        capped = []
        for row in partition_rows:
            if video_counts[row["video"]] >= per_video_cap:
                continue
            video_counts[row["video"]] += 1
            capped.append(row)
        post_cap_counts[partition] = len(capped)
        for ordinal, row in enumerate(capped[:max_per_partition], start=1):
            queues[partition].append(
                {
                    **row,
                    "review_ordinal": ordinal,
                    "review_token": _visor_hos_no_hand_review_token(
                        seed, partition, row
                    ),
                }
            )
        if len({row["review_token"] for row in queues[partition]}) != len(
            queues[partition]
        ):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_TOKEN_COLLISION")
    development_frames = {
        (row["video"], row["frame_name"]) for row in queues["development"]
    }
    holdout_frames = {
        (row["video"], row["frame_name"]) for row in queues["holdout"]
    }
    development_participants = {
        row["participant"] for row in queues["development"]
    }
    holdout_participants = {row["participant"] for row in queues["holdout"]}
    development_videos = {row["video"] for row in queues["development"]}
    holdout_videos = {row["video"] for row in queues["holdout"]}
    report = {
        "raw_nominee_count": len(nominees),
        "post_partition_counts": post_partition_counts,
        "post_cap_counts": post_cap_counts,
        "queue_counts": {
            partition: len(rows) for partition, rows in queues.items()
        },
        "invalid_frame_count": invalid_frame_count,
        "abstained_frame_count": abstained_frame_count,
        "duplicate_nominee_count": duplicate_nominee_count,
        "correction_excluded_nominee_count": excluded_nominee_count,
        "participant_overlap_count": len(
            development_participants & holdout_participants
        ),
        "video_overlap_count": len(development_videos & holdout_videos),
        "frame_overlap_count": len(development_frames & holdout_frames),
    }
    return queues, report


def _visor_hos_review_source_frame(frame_root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeError("E_VISOR_HOS_REVIEW_FRAME_PATH")
    source = (frame_root / relative_path).resolve()
    try:
        source.relative_to(frame_root.resolve())
    except ValueError as error:
        raise RuntimeError("E_VISOR_HOS_REVIEW_FRAME_PATH") from error
    return source


def _write_private_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(value)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _render_visor_hos_no_hand_review_sheet(
    items: list[dict[str, Any]], sheet_ordinal: int
) -> bytes:
    from PIL import Image, ImageDraw, ImageOps

    columns = 4
    rows = 2
    tile_width = 480
    image_height = 360
    label_height = 40
    header_height = 40
    canvas = Image.new(
        "RGB",
        (
            columns * tile_width,
            header_height + rows * (image_height + label_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(items):
        column = index % columns
        row = index // columns
        left = column * tile_width
        top = header_height + row * (image_height + label_height)
        source = Path(item["resolved_source_path"])
        try:
            with Image.open(source) as opened:
                image = ImageOps.contain(opened.convert("RGB"), (tile_width, image_height))
            x = left + (tile_width - image.width) // 2
            y = top + (image_height - image.height) // 2
            canvas.paste(image, (x, y))
        except Exception:
            draw.rectangle(
                (left, top, left + tile_width - 1, top + image_height - 1),
                fill="#eeeeee",
                outline="#aa0000",
            )
            draw.text(
                (left + 8, top + image_height // 2),
                "DECODE FAILURE: code abstain",
                fill="#880000",
            )
        draw.rectangle(
            (
                left,
                top + image_height,
                left + tile_width - 1,
                top + image_height + label_height - 1,
            ),
            outline="black",
        )
        draw.text(
            (left + 8, top + image_height + 6),
            f"{item['review_ordinal']:03d}  {item['review_token']}",
            fill="black",
        )
    draw.text(
        (8, 10),
        (
            f"blind review sheet {sheet_ordinal:02d} — visible hand? "
            "code yes=no hand, no=hand visible, abstain=ambiguous"
        ),
        fill="black",
    )
    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def prepare_visor_hos_no_hand_review(
    annotation_documents: list[dict[str, Any]] | None,
    *,
    cfg: dict[str, Any],
    frame_root: Path,
    review_root: Path,
    correction_excluded_frame_keys: set[tuple[str, str]]
    | frozenset[tuple[str, str]]
    | None = None,
    preselected_queues: dict[str, list[dict[str, Any]]] | None = None,
    inventory_override: dict[str, Any] | None = None,
    source_feasibility_commitment_sha256: str | None = None,
    source_frame_materialization_commitment_sha256: str | None = None,
    construct_aligned_amendment_commitment_sha256: str | None = None,
) -> dict[str, Any]:
    """Render a fixed, model-blind public no-hand review queue outside Git."""

    amendment = _tuple_visor_hos_correction_amendment(cfg)
    sampler = amendment["partition_and_joint_sampler"]
    target = int(sampler["quota_per_partition_per_stratum"])
    if target != 48 or VISOR_HOS_NO_HAND_REVIEW_MAX_PER_PARTITION != 192:
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_PROTOCOL")
    _refuse_git_output(frame_root)
    _refuse_git_output(review_root)
    if preselected_queues is None:
        if annotation_documents is None:
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE")
        queues, inventory = _visor_hos_no_hand_review_nominees(
            annotation_documents,
            seed=int(sampler["seed"]),
            per_video_cap=int(sampler["per_video_per_stratum_cap"]),
            correction_excluded_frame_keys=correction_excluded_frame_keys,
        )
    else:
        if annotation_documents not in (None, []):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE")
        if (
            set(preselected_queues) != {"development", "holdout"}
            or not isinstance(inventory_override, dict)
            or not isinstance(source_feasibility_commitment_sha256, str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", source_feasibility_commitment_sha256
            )
            or not isinstance(
                source_frame_materialization_commitment_sha256, str
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                source_frame_materialization_commitment_sha256,
            )
            or not isinstance(
                construct_aligned_amendment_commitment_sha256, str
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                construct_aligned_amendment_commitment_sha256,
            )
        ):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_LINEAGE")
        queues = json.loads(json.dumps(preselected_queues))
        inventory = json.loads(json.dumps(inventory_override))
    items: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    contact_sheets: list[dict[str, Any]] = []
    rendered_sheets: dict[str, bytes] = {}
    decode_failure_count = 0
    for partition in ("development", "holdout"):
        for row in queues[partition]:
            source = _visor_hos_review_source_frame(
                frame_root, str(row["image_path"])
            )
            source_sha256 = file_digest(source) if source.is_file() else None
            decode_status = "PASS"
            try:
                from PIL import Image

                with Image.open(source) as image:
                    image.verify()
            except Exception:
                decode_status = "DECODE_FAILURE"
                decode_failure_count += 1
            items[partition].append(
                {
                    **row,
                    "resolved_source_path": str(source),
                    "source_frame_sha256": source_sha256,
                    "decode_status": decode_status,
                }
            )
        for start in range(
            0,
            len(items[partition]),
            VISOR_HOS_NO_HAND_REVIEW_ITEMS_PER_SHEET,
        ):
            chunk = items[partition][
                start : start + VISOR_HOS_NO_HAND_REVIEW_ITEMS_PER_SHEET
            ]
            sheet_ordinal = start // VISOR_HOS_NO_HAND_REVIEW_ITEMS_PER_SHEET + 1
            sheet_name = f"contact-sheet-{partition}-{sheet_ordinal:02d}.png"
            sheet = _render_visor_hos_no_hand_review_sheet(chunk, sheet_ordinal)
            rendered_sheets[sheet_name] = sheet
            contact_sheets.append(
                {
                    "partition": partition,
                    "sheet_ordinal": sheet_ordinal,
                    "relative_path": sheet_name,
                    "item_count": len(chunk),
                    "sha256": hashlib.sha256(sheet).hexdigest(),
                }
            )
    queue = {
        "schema_version": 1,
        "status": "READY_FOR_AUTHORIZED_APPLICANT_BLIND_REVIEW",
        "visor_hos_correction_amendment_commitment_sha256": amendment[
            "amendment_commitment_sha256"
        ],
        "seed": int(sampler["seed"]),
        "target_verified_no_hand_per_partition": target,
        "maximum_nominees_per_partition": (
            VISOR_HOS_NO_HAND_REVIEW_MAX_PER_PARTITION
        ),
        "per_video_nominee_cap": int(sampler["per_video_per_stratum_cap"]),
        "model_inference_executed_before_review": False,
        "model_output_fields_present": False,
        "label_semantics": {
            "yes": "visually verified no visible hand",
            "no": "a visible hand is present",
            "abstain": "ambiguous visibility decode failure or incomplete annotation",
        },
        "inventory": inventory,
        "partitions": items,
        "contact_sheets": contact_sheets,
    }
    if source_feasibility_commitment_sha256 is not None:
        queue["visor_hos_source_feasibility_commitment_sha256"] = (
            source_feasibility_commitment_sha256
        )
    if source_frame_materialization_commitment_sha256 is not None:
        queue["source_frame_materialization_commitment_sha256"] = (
            source_frame_materialization_commitment_sha256
        )
    if construct_aligned_amendment_commitment_sha256 is not None:
        queue["construct_aligned_ltx_resume_amendment_commitment_sha256"] = (
            construct_aligned_amendment_commitment_sha256
        )
    queue["review_queue_commitment_sha256"] = digest(queue)
    manifest_path = review_root / "review-queue.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        prior_commitment = prior.pop("review_queue_commitment_sha256", None)
        if prior_commitment != digest(prior) or prior != {
            key: value
            for key, value in queue.items()
            if key != "review_queue_commitment_sha256"
        }:
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_ALREADY_FROZEN")
        prior["review_queue_commitment_sha256"] = prior_commitment
        for sheet_name, sheet in rendered_sheets.items():
            path = review_root / sheet_name
            if not path.is_file() or file_digest(path) != hashlib.sha256(
                sheet
            ).hexdigest():
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SHEET_COMMITMENT")
    else:
        for sheet_name, sheet in rendered_sheets.items():
            _write_private_bytes(review_root / sheet_name, sheet)
        write_private(manifest_path, queue)
        labels = {
            "schema_version": 1,
            "review_queue_commitment_sha256": queue[
                "review_queue_commitment_sha256"
            ],
            "reviewer_role": "authorized_applicant",
            "labels": [
                {
                    "partition": partition,
                    "review_token": row["review_token"],
                    "label": "",
                }
                for partition in ("development", "holdout")
                for row in items[partition]
            ],
        }
        if construct_aligned_amendment_commitment_sha256 is not None:
            labels[
                "construct_aligned_ltx_resume_amendment_commitment_sha256"
            ] = construct_aligned_amendment_commitment_sha256
        write_private(review_root / "review-labels.json", labels)
    return {
        "status": queue["status"],
        "partition_count": 2,
        "nominee_count": int(inventory["raw_nominee_count"]),
        "review_queue_count": sum(len(rows) for rows in items.values()),
        "decode_failure_count": decode_failure_count,
        "contact_sheet_count": len(contact_sheets),
        "review_queue_commitment_sha256": queue[
            "review_queue_commitment_sha256"
        ],
    }


def _load_visor_hos_no_hand_review_queue(review_root: Path) -> dict[str, Any]:
    path = review_root / "review-queue.json"
    if not path.is_file():
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_QUEUE_MISSING")
    queue = json.loads(path.read_text())
    expected = queue.pop("review_queue_commitment_sha256", None)
    if not isinstance(expected, str) or digest(queue) != expected:
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_QUEUE_COMMITMENT")
    queue["review_queue_commitment_sha256"] = expected
    if (
        queue.get("status") != "READY_FOR_AUTHORIZED_APPLICANT_BLIND_REVIEW"
        or queue.get("model_inference_executed_before_review") is not False
        or queue.get("model_output_fields_present") is not False
        or queue.get("target_verified_no_hand_per_partition") != 48
        or queue.get("maximum_nominees_per_partition") != 192
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_QUEUE_SCHEMA")
    return queue


def _validate_visor_hos_no_hand_review_artifacts(
    queue: dict[str, Any], review_root: Path
) -> None:
    for sheet in queue.get("contact_sheets", []):
        if not isinstance(sheet, dict):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SHEET_SCHEMA")
        relative_path = sheet.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or Path(relative_path).name != relative_path
        ):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SHEET_SCHEMA")
        path = review_root / relative_path
        if not path.is_file() or file_digest(path) != sheet.get("sha256"):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SHEET_COMMITMENT")
    for partition in ("development", "holdout"):
        rows = queue.get("partitions", {}).get(partition)
        if not isinstance(rows, list):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_QUEUE_SCHEMA")
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_QUEUE_SCHEMA")
            path = Path(str(row.get("resolved_source_path", "")))
            expected = row.get("source_frame_sha256")
            if row.get("decode_status") == "PASS":
                if (
                    not isinstance(expected, str)
                    or not path.is_file()
                    or file_digest(path) != expected
                ):
                    raise RuntimeError(
                        "E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_COMMITMENT"
                    )


def seal_visor_hos_no_hand_review(
    *,
    review_root: Path,
    authorized_applicant_attested: bool,
    blind_to_egohos_output_attested: bool,
    egohos_inference_not_started_attested: bool,
) -> dict[str, Any]:
    """Seal applicant labels; only a 48-per-partition PASS can feed inference."""

    _refuse_git_output(review_root)
    if not all(
        (
            authorized_applicant_attested,
            blind_to_egohos_output_attested,
            egohos_inference_not_started_attested,
        )
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_ATTESTATION")
    queue = _load_visor_hos_no_hand_review_queue(review_root)
    _validate_visor_hos_no_hand_review_artifacts(queue, review_root)
    label_path = review_root / "review-labels.json"
    if not label_path.is_file():
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_LABELS_MISSING")
    labels = json.loads(label_path.read_text())
    expected_label_keys = {
            "schema_version",
            "review_queue_commitment_sha256",
            "reviewer_role",
            "labels",
    }
    active_commitment = queue.get(
        "construct_aligned_ltx_resume_amendment_commitment_sha256"
    )
    if active_commitment is not None:
        expected_label_keys.add(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        )
    if (
        set(labels) != expected_label_keys
        or labels.get(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        )
        != active_commitment
        or any(
            not isinstance(row, dict)
            or set(row) != {"partition", "review_token", "label"}
            for row in labels.get("labels", [])
        )
        or labels.get("schema_version") != 1
        or labels.get("reviewer_role") != "authorized_applicant"
        or labels.get("review_queue_commitment_sha256")
        != queue["review_queue_commitment_sha256"]
        or not isinstance(labels.get("labels"), list)
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_LABEL_SCHEMA")
    queued = [
        (partition, row)
        for partition in ("development", "holdout")
        for row in queue["partitions"][partition]
    ]
    expected_keys = [
        (partition, row["review_token"]) for partition, row in queued
    ]
    observed_keys = [
        (row.get("partition"), row.get("review_token"))
        for row in labels["labels"]
        if isinstance(row, dict)
    ]
    if observed_keys != expected_keys or len(observed_keys) != len(labels["labels"]):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_LABEL_ORDER")
    normalized_labels: dict[tuple[str, str], str] = {}
    for row in labels["labels"]:
        label = row.get("label")
        if label not in VISOR_HOS_NO_HAND_REVIEW_LABELS:
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_LABEL_VALUE")
        normalized_labels[(row["partition"], row["review_token"])] = label
    partition_records: dict[str, dict[str, Any]] = {}
    total_coded = 0
    total_no = 0
    total_abstain = 0
    total_unreviewed = 0
    total_selected = 0
    deficit_partition_count = 0
    partition_statuses: list[str] = []
    target = int(queue["target_verified_no_hand_per_partition"])
    for partition in ("development", "holdout"):
        partition_queue = queue["partitions"][partition]
        values = [
            normalized_labels[(partition, row["review_token"])]
            for row in partition_queue
        ]
        coded_indices = [index for index, value in enumerate(values) if value]
        if coded_indices != list(range(len(coded_indices))):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_NONCONTIGUOUS")
        for row, label in zip(partition_queue, values, strict=True):
            if row["decode_status"] != "PASS" and label not in {"", "abstain"}:
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_DECODE_LABEL")
        verified = [
            row
            for row, label in zip(partition_queue, values, strict=True)
            if label == "yes"
        ]
        selected = verified[:target]
        all_reviewed = len(coded_indices) == len(values)
        if len(selected) != target:
            deficit_partition_count += 1
            partition_statuses.append(
                "NO_GO" if all_reviewed else "INCOMPLETE_REVIEW"
            )
        else:
            partition_statuses.append("PASS")
        counts = Counter(values)
        total_coded += len(coded_indices)
        total_no += int(counts["no"])
        total_abstain += int(counts["abstain"])
        total_unreviewed += int(counts[""])
        total_selected += len(selected)
        partition_records[partition] = {
            "queue_count": len(partition_queue),
            "coded_count": len(coded_indices),
            "yes_count": int(counts["yes"]),
            "no_count": int(counts["no"]),
            "abstain_count": int(counts["abstain"]),
            "unreviewed_count": int(counts[""]),
            "selected_verified_no_hand_count": len(selected),
            "selected": [
                {
                    "review_token": row["review_token"],
                    "video": row["video"],
                    "frame_name": row["frame_name"],
                    "source_frame_sha256": row["source_frame_sha256"],
                }
                for row in selected
            ],
        }
    if "INCOMPLETE_REVIEW" in partition_statuses:
        status = "INCOMPLETE_REVIEW"
    elif "NO_GO" in partition_statuses:
        status = "NO_GO"
    else:
        status = "PASS"
    labels_payload = {
        "schema_version": 1,
        "review_queue_commitment_sha256": queue[
            "review_queue_commitment_sha256"
        ],
        "reviewer_role": "authorized_applicant",
        "blind_to_egohos_output_attested": True,
        "egohos_inference_not_started_attested": True,
        "labels": labels["labels"],
    }
    if active_commitment is not None:
        labels_payload[
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        ] = active_commitment
    labels_commitment = digest(labels_payload)
    if status != "INCOMPLETE_REVIEW":
        seal = {
            "schema_version": 1,
            "status": status,
            "review_queue_commitment_sha256": queue[
                "review_queue_commitment_sha256"
            ],
            "review_labels_commitment_sha256": labels_commitment,
            "reviewer_role": "authorized_applicant",
            "blind_to_egohos_output_attested": True,
            "egohos_inference_not_started_attested": True,
            "partitions": partition_records,
        }
        if active_commitment is not None:
            seal[
                "construct_aligned_ltx_resume_amendment_commitment_sha256"
            ] = active_commitment
        seal["verified_no_hand_seal_commitment_sha256"] = digest(seal)
        seal_path = review_root / "verified-no-hand-seal.json"
        if seal_path.exists():
            prior = json.loads(seal_path.read_text())
            expected = prior.pop("verified_no_hand_seal_commitment_sha256", None)
            if expected != digest(prior) or prior != {
                key: value
                for key, value in seal.items()
                if key != "verified_no_hand_seal_commitment_sha256"
            }:
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_ALREADY_SEALED")
        else:
            write_private(seal_path, seal)
        seal_commitment = seal["verified_no_hand_seal_commitment_sha256"]
    else:
        seal_commitment = None
    return {
        "status": status,
        "partition_count": 2,
        "coded_count": total_coded,
        "verified_no_hand_count": total_selected,
        "visible_hand_count": total_no,
        "abstain_count": total_abstain,
        "unreviewed_count": total_unreviewed,
        "deficit_partition_count": deficit_partition_count,
        "review_labels_commitment_sha256": labels_commitment,
        "verified_no_hand_seal_commitment_sha256": seal_commitment,
    }


def load_visor_hos_verified_no_hand_frames(
    review_root: Path,
) -> set[tuple[str, str]]:
    """Load the sole model-inference authorization artifact for no-hand truth."""

    queue = _load_visor_hos_no_hand_review_queue(review_root)
    _validate_visor_hos_no_hand_review_artifacts(queue, review_root)
    path = review_root / "verified-no-hand-seal.json"
    if not path.is_file():
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_NOT_SEALED")
    seal = json.loads(path.read_text())
    expected = seal.pop("verified_no_hand_seal_commitment_sha256", None)
    if not isinstance(expected, str) or digest(seal) != expected:
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SEAL_COMMITMENT")
    if (
        seal.get("status") != "PASS"
        or seal.get("review_queue_commitment_sha256")
        != queue["review_queue_commitment_sha256"]
        or seal.get("reviewer_role") != "authorized_applicant"
        or seal.get("blind_to_egohos_output_attested") is not True
        or seal.get("egohos_inference_not_started_attested") is not True
        or seal.get(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        )
        != queue.get(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        )
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_NOT_AUTHORIZED")
    output: set[tuple[str, str]] = set()
    for partition in ("development", "holdout"):
        record = seal.get("partitions", {}).get(partition, {})
        selected = record.get("selected")
        queue_rows = queue.get("partitions", {}).get(partition, [])
        queue_by_token = {
            row.get("review_token"): row
            for row in queue_rows
            if isinstance(row, dict)
        }
        if (
            record.get("selected_verified_no_hand_count") != 48
            or not isinstance(selected, list)
            or len(selected) != 48
            or len(queue_by_token) != len(queue_rows)
        ):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SELECTION_COUNT")
        selected_ordinals: list[int] = []
        for row in selected:
            queued = queue_by_token.get(row.get("review_token"))
            if (
                queued is None
                or row.get("video") != queued.get("video")
                or row.get("frame_name") != queued.get("frame_name")
                or row.get("source_frame_sha256")
                != queued.get("source_frame_sha256")
            ):
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SELECTION_KEY")
            key = (row.get("video"), row.get("frame_name"))
            if not all(isinstance(value, str) and value for value in key):
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SELECTION_KEY")
            selected_ordinals.append(int(queued["review_ordinal"]))
            output.add(key)
        if selected_ordinals != sorted(selected_ordinals) or len(
            set(selected_ordinals)
        ) != len(selected_ordinals):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SELECTION_ORDER")
    if len(output) != 96:
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SELECTION_OVERLAP")
    return output


def _tuple_combined_public_gate(
    axis_results: dict[str, dict[str, Any]],
    action_control_result: dict[str, Any],
    broad_activity_result: dict[str, Any] | None = None,
    *,
    action_control_blocks: bool = True,
) -> dict[str, Any]:
    """Apply the frozen five-critical-plus-six-of-seven decision once."""

    axis_ids = (*TUPLE_CRITICAL_AXIS_IDS, *TUPLE_SUPPORTING_AXIS_IDS)
    unexpected = sorted(set(axis_results) - set(axis_ids))
    if unexpected:
        raise RuntimeError("E_TUPLE_COMBINED_GATE_AXIS_SET")
    statuses = {
        axis_id: str(axis_results.get(axis_id, {}).get("status", "UNMEASURED"))
        for axis_id in axis_ids
    }
    critical_failures = [
        axis_id
        for axis_id in TUPLE_CRITICAL_AXIS_IDS
        if statuses[axis_id] != "PASS"
    ]
    validated_axis_count = sum(status == "PASS" for status in statuses.values())
    action_status = str(action_control_result.get("status", "UNMEASURED"))
    failures = [f"critical_axis:{axis_id}" for axis_id in critical_failures]
    if validated_axis_count < 6:
        failures.append("validated_axes_minimum:6_of_7")
    if action_control_blocks and action_status != "PASS":
        failures.append("order_dependent_action_control")
    return {
        "status": "PASS" if not failures else "NO_GO",
        "axis_statuses": statuses,
        "critical_axis_pass_count": 5 - len(critical_failures),
        "critical_axis_required_count": 5,
        "validated_axis_count": validated_axis_count,
        "validated_axis_required_count": 6,
        "critical_axis_failures": critical_failures,
        "unvalidated_axis_ids": [
            axis_id for axis_id in axis_ids if statuses[axis_id] != "PASS"
        ],
        "action_control_status": action_status,
        "action_control_used_in_gate": action_control_blocks,
        "broad_activity_status": str(
            (broad_activity_result or {}).get("status", "UNMEASURED")
        ),
        "broad_activity_used_in_gate": False,
        "combined_gate_failures": failures,
    }


TUPLE_QUALIFICATION_MODULE_IDS = (
    "adapter_and_lexical",
    "referent",
    "recurrence",
    "attribute",
    "hand_contact",
    "sensor",
    "order_action",
)

TUPLE_HEALTH_ERROR_FAMILIES = (
    "E_ACTIVITY_",
    "E_CONSTRUCT_",
    "E_EGOHOS_",
    "E_FROZEN_",
    "E_PRIVATE_",
    "E_SENSOR_",
    "E_TUPLE_",
    "E_VISOR_",
)


def _declared_health_error_codes() -> frozenset[str]:
    """Return only error identifiers literally committed in this runner.

    Dependency messages and input-derived strings must never be able to mint a
    new compact terminal error identifier.  The runner hash therefore also
    commits the complete set of codes that the health record may release.
    """

    source = Path(__file__).resolve().read_text(encoding="utf-8")
    return frozenset(re.findall(r"\bE_[A-Z0-9_]+\b", source))


def _tuple_health_fixture_rows(
    rows: list[dict[str, Any]], family: str
) -> dict[int, dict[str, Any]]:
    """Give sealed rows a deterministic public-only ordinal."""

    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, dict) for row in rows
    ):
        raise RuntimeError("E_TUPLE_HEALTH_DEVELOPMENT_FIXTURES")
    fixture_ordinals = [row.get("fixture_ordinal") for row in rows]
    if all(type(value) is int and int(value) >= 0 for value in fixture_ordinals):
        if len(fixture_ordinals) != len(set(fixture_ordinals)):
            raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_ORDINAL")
        return {
            int(row["fixture_ordinal"]): row
            for row in rows
        }
    if family != "language_lexical" or any(
        not isinstance(row.get("case_id"), str) or not row["case_id"]
        for row in rows
    ):
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_ORDINAL")
    ordered = sorted(rows, key=lambda row: str(row["case_id"]))
    if len({row["case_id"] for row in ordered}) != len(ordered):
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_ORDINAL")
    return dict(enumerate(ordered))


def _tuple_health_case_candidates(
    module_id: str,
    case_class: str,
    indexed: dict[int, dict[str, Any]],
) -> list[int]:
    """Map the frozen health classes to public fixture semantics only."""

    output: list[int] = []
    for ordinal, row in indexed.items():
        reason = row.get("expected_adapter_reason")
        scenario = row.get("scenario")
        truth = row.get("truth") if isinstance(row.get("truth"), dict) else {}
        stratum = row.get("stratum")
        condition = row.get("condition")
        control_role = row.get("control_role")
        matches = False
        if module_id == "adapter_and_lexical":
            parts = {
                mention.get("part_of_speech")
                for mention in row.get("expected_lexical_mentions", [])
                if isinstance(mention, dict)
            }
            matches = {
                "accepted_noun_and_adjective": (
                    row.get("expected_adapter_status") == "ACCEPT"
                    and {"noun", "adjective"} <= parts
                ),
                "language_abstention": reason in {
                    "LANGUAGE_MISMATCH",
                    "EMPTY_ASR",
                },
                "confidence_or_timestamp_abstention": reason in {
                    "LOW_CONFIDENCE",
                    "INVALID_TIMESTAMP",
                },
                "translation_or_truncation_abstention": reason in {
                    "EMPTY_TRANSLATION",
                    "SILENT_TRUNCATION",
                },
            }.get(case_class, False)
        elif module_id == "referent":
            matches = {
                "visible": scenario == "during_only",
                "absent": scenario == "speech_no_referent",
                "dominant": scenario
                == "persistent_dominant_with_small_distractor",
                "ambiguous": scenario == "persistent_ambiguous",
            }.get(case_class, False)
        elif module_id == "recurrence":
            matches = {
                "positive_same_referent_1": stratum
                == "same_instance_transformed",
                "positive_same_referent_2": stratum
                == "same_instance_near_duplicate",
                "negative_different_referent_1": stratum
                == "same_category_different_instance",
                "negative_different_referent_2": stratum
                == "different_category",
            }.get(case_class, False)
        elif module_id == "attribute":
            matches = {
                "positive_visible_contrast_1": scenario
                == "persistent_ambiguous",
                "positive_visible_contrast_2": scenario
                == "persistent_dominant_with_small_distractor",
                "negative_or_absent_contrast_1": scenario
                == "speech_no_referent",
                "negative_or_absent_contrast_2": scenario
                == "no_speech_visible_object",
            }.get(case_class, False)
            if matches and case_class.startswith("positive_"):
                matches = truth.get("attribute_contrast_expected", True) is True
        elif module_id == "hand_contact":
            matches = {
                "visible_hand": stratum != "verified_no_hand",
                "verified_no_hand": stratum == "verified_no_hand",
                "contact": stratum == "contact",
                "explicit_no_contact": stratum == "explicit_no_contact",
            }.get(case_class, False)
        elif module_id == "sensor":
            matches = {
                "static_or_baseline": condition == "static",
                "motion": condition in {"low_translation", "high_translation"},
                "blur_or_lighting": condition
                in {"mild_blur", "strong_blur", "dark", "bright"},
                "hard_cut_or_transition": condition == "hard_cut",
            }.get(case_class, False)
        elif module_id == "order_action":
            matches = control_role in {None, case_class}
        if matches:
            output.append(ordinal)
    return output


def _tuple_health_projection(
    manifest: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Freeze one metric-free, development-only microfixture projection."""

    health = _engineering_health_amendment(cfg)
    micro = health["engineering_microfixture_suite"]
    partitions = manifest.get("partitions")
    if (
        manifest.get("status") != "SEALED_BEFORE_PUBLIC_DEVELOPMENT_INFERENCE"
        or not isinstance(partitions, dict)
        or not isinstance(partitions.get("development"), dict)
    ):
        raise RuntimeError("E_TUPLE_HEALTH_DEVELOPMENT_FIXTURES")
    development = partitions["development"]
    family_by_module = {
        "adapter_and_lexical": "language_lexical",
        "referent": "referent_attribute",
        "recurrence": "recurrence",
        "attribute": "referent_attribute",
        "hand_contact": "hand_contact",
        "sensor": "sensor",
        "order_action": "order_action",
    }
    cases = []
    seed = int(micro["selection_seed"])
    for module_id in TUPLE_QUALIFICATION_MODULE_IDS:
        family = family_by_module[module_id]
        indexed = _tuple_health_fixture_rows(development.get(family, []), family)
        used: set[int] = set()
        selected_by_class: dict[str, int] = {}
        for case_class in micro["required_case_classes"][module_id]:
            candidates = [
                ordinal
                for ordinal in _tuple_health_case_candidates(
                    module_id, case_class, indexed
                )
                if ordinal not in used
            ]
            if (
                module_id == "attribute"
                and case_class == "positive_visible_contrast_2"
                and "positive_visible_contrast_1" in selected_by_class
            ):
                first = indexed[
                    selected_by_class["positive_visible_contrast_1"]
                ].get("attribute_pair_id")
                if isinstance(first, str):
                    candidates = [
                        ordinal
                        for ordinal in candidates
                        if indexed[ordinal].get("attribute_pair_id") == first
                    ]
            candidates.sort(
                key=lambda ordinal: _fixture_order(
                    seed, module_id, case_class, ordinal
                )
            )
            if not candidates:
                raise RuntimeError("E_TUPLE_HEALTH_CASE_CLASS_DEFICIT")
            selected = candidates[0]
            used.add(selected)
            selected_by_class[case_class] = selected
            cases.append(
                {
                    "module_id": module_id,
                    "case_class": case_class,
                    "source_family": family,
                    "source_fixture_ordinal": selected,
                }
            )
    if len(cases) != int(micro["total_case_count"]):
        raise RuntimeError("E_TUPLE_HEALTH_CASE_CLASS_DEFICIT")
    record = {
        "schema_version": 1,
        "status": "SEALED_ENGINEERING_HEALTH_MICROFIXTURES",
        "role": "EXECUTION_HEALTH_ONLY",
        "route_id": health["route_id"],
        "source_partition": "development",
        "selection_seed": seed,
        "public_fixture_manifest_commitment_sha256": manifest.get(
            "public_fixture_manifest_commitment_sha256"
        ),
        "engineering_health_amendment_commitment_sha256": health[
            "amendment_commitment_sha256"
        ],
        "module_count": len(TUPLE_QUALIFICATION_MODULE_IDS),
        "case_count": len(cases),
        "holdout_input_count": 0,
        "scientific_metric_count": 0,
        "cases": cases,
    }
    record["microfixture_manifest_commitment_sha256"] = digest(record)
    return record


def _tuple_health_error(
    module_id: str, error: BaseException, trace_root: Path
) -> dict[str, Any]:
    """Write private diagnostics and return only a stable compact code."""

    if module_id not in TUPLE_QUALIFICATION_MODULE_IDS:
        raise RuntimeError("E_TUPLE_HEALTH_MODULE_ID")
    trace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(trace_root, 0o700)
    ordinal = 1
    while (trace_root / f"{module_id}-{ordinal:02d}.trace").exists():
        ordinal += 1
    trace_path = trace_root / f"{module_id}-{ordinal:02d}.trace"
    trace_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    trace_path.write_text(trace_text, encoding="utf-8")
    os.chmod(trace_path, 0o600)
    message = str(error)
    match = re.match(r"^(E_[A-Z0-9_]+)(?:\s|$)", message)
    raw_code = match.group(1) if match else None
    if (
        raw_code is not None
        and raw_code in _declared_health_error_codes()
        and raw_code.startswith(TUPLE_HEALTH_ERROR_FAMILIES)
    ):
        suffix = raw_code[2:]
    else:
        suffix = "UNACCOUNTED_FAILURE"
    return {
        "module_id": module_id,
        "status": "ERROR",
        "error_code": f"E_TUPLE_HEALTH_{module_id.upper()}_{suffix}",
        "trace_written": True,
    }


def _tuple_health_budget(
    attempt: int, prior_attempts: list[dict[str, Any]], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Validate the fixed resource-redirect repair budget before submission."""

    resource = _engineering_health_resource_policy(cfg)
    maximum_attempts = int(
        resource["initial_plus_repair_resmoke_submission_count_max"]
    )
    if type(attempt) is not int or not 1 <= attempt <= maximum_attempts:
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    if not isinstance(prior_attempts, list):
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    observed_attempts = []
    total_gpu_hours = 0.0
    total_storage = 0.0
    for record in prior_attempts:
        if not isinstance(record, dict):
            raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
        values = record.get("resource", record)
        record_attempt = record.get("attempt")
        observed_attempts.append(record_attempt)
        if (
            values.get("GPU_type")
            != _engineering_health_attempt_gpu_type(cfg, record_attempt)
            or values.get("GPU_count") != resource["GPU_count"]
        ):
            raise RuntimeError("E_TUPLE_HEALTH_GPU_TOPOLOGY")
        wall = float(values.get("wall_minutes", math.inf))
        gpu_hours = float(values.get("GPU_hours", math.inf))
        storage = float(values.get("new_storage_GiB", math.inf))
        cost = float(values.get("direct_monetary_cost_USD", math.inf))
        if not all(math.isfinite(value) and value >= 0 for value in (
            wall, gpu_hours, storage, cost
        )):
            raise RuntimeError("E_TUPLE_HEALTH_RESOURCE_NONFINITE")
        if wall > float(resource["per_submission_wall_minutes_max"]):
            raise RuntimeError("E_TUPLE_HEALTH_WALL_BUDGET")
        if gpu_hours > float(resource["per_submission_wall_minutes_max"]) / 60.0:
            raise RuntimeError("E_TUPLE_HEALTH_GPU_HOUR_BUDGET")
        if cost != float(resource["direct_monetary_cost_USD"]):
            raise RuntimeError("E_TUPLE_HEALTH_COST_BUDGET")
        total_gpu_hours += gpu_hours
        total_storage += storage
    if observed_attempts != list(range(1, attempt)):
        raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
    if total_gpu_hours > float(resource["aggregate_GPU_hours_max"]):
        raise RuntimeError("E_TUPLE_HEALTH_GPU_HOUR_BUDGET")
    if total_storage > float(resource["new_storage_GiB_max"]):
        raise RuntimeError("E_TUPLE_HEALTH_STORAGE_BUDGET")
    return {
        "attempt": attempt,
        "GPU_type": resource["GPU_type"],
        "GPU_count": int(resource["GPU_count"]),
        "CPU_count": int(resource["CPU_count"]),
        "memory_GiB": int(resource["memory_GiB"]),
        "per_submission_wall_minutes_max": int(
            resource["per_submission_wall_minutes_max"]
        ),
        "remaining_GPU_hours": float(resource["aggregate_GPU_hours_max"])
        - total_gpu_hours,
        "remaining_storage_GiB": float(resource["new_storage_GiB_max"])
        - total_storage,
        "direct_monetary_cost_USD": float(
            resource["direct_monetary_cost_USD"]
        ),
    }


def _tuple_health_cumulative_resource(
    prior_attempts: list[dict[str, Any]], current: dict[str, Any]
) -> dict[str, Any]:
    resources = [record.get("resource", record) for record in prior_attempts]
    resources.append(current)
    return {
        "cumulative_submission_count": len(resources),
        "cumulative_wall_minutes": sum(
            float(value["wall_minutes"]) for value in resources
        ),
        "cumulative_GPU_hours": sum(
            float(value["GPU_hours"]) for value in resources
        ),
        "cumulative_new_storage_GiB": sum(
            float(value["new_storage_GiB"]) for value in resources
        ),
        "cumulative_direct_monetary_cost_USD": sum(
            float(value["direct_monetary_cost_USD"]) for value in resources
        ),
    }


def _tuple_health_wrapper_marker(
    attempt_root: Path, attempt: int, cfg: dict[str, Any]
) -> dict[str, Any]:
    path = attempt_root / "wrapper-started.json"
    if not path.is_file():
        raise RuntimeError("E_TUPLE_HEALTH_WRAPPER_MARKER")
    value = json.loads(path.read_text())
    policy = _engineering_health_resource_policy(cfg)
    expected_gpu_type = _engineering_health_attempt_gpu_type(cfg, attempt)
    if (
        set(value)
        != {
            "schema_version",
            "attempt",
            "submission_started_epoch",
            "GPU_type",
            "GPU_count",
            "CPU_count",
            "memory_GiB",
        }
        or value.get("schema_version") != 1
        or value.get("attempt") != attempt
        or value.get("GPU_type") != expected_gpu_type
        or value.get("GPU_count") != policy["GPU_count"]
        or value.get("CPU_count") != policy["CPU_count"]
        or value.get("memory_GiB") != policy["memory_GiB"]
        or not isinstance(value.get("submission_started_epoch"), (int, float))
        or not math.isfinite(float(value["submission_started_epoch"]))
        or float(value["submission_started_epoch"]) <= 0
        or float(value["submission_started_epoch"]) > time.time() + 5.0
    ):
        raise RuntimeError("E_TUPLE_HEALTH_WRAPPER_MARKER")
    return value


def _tuple_health_incomplete_attempt_resource(
    root: Path, attempt: int, cfg: dict[str, Any]
) -> dict[str, Any]:
    if (
        "learner_effective_engineering_health_submission_export_repair" in cfg
        and attempt == 12
    ):
        result = _engineering_health_attempt_12_result(cfg)
        submission = result["submission_provenance"]
        resource = result["resource_accounting"]
        return {
            "GPU_type": submission["GPU_type"],
            "GPU_count": submission["GPU_count"],
            "CPU_count": submission["CPU_count"],
            "memory_GiB": submission["memory_GiB"],
            "wall_minutes": resource["attempt_wall_minutes_actual"],
            "GPU_hours": resource["attempt_GPU_hours_actual"],
            "new_storage_GiB": resource["attempt_retained_storage_GiB"],
            "direct_monetary_cost_USD": resource[
                "direct_monetary_cost_USD"
            ],
            "submission_started_epoch": submission[
                "submission_started_epoch"
            ],
        }
    attempt_root = root / "health" / f"attempt-{attempt:02d}"
    marker = _tuple_health_wrapper_marker(attempt_root, attempt, cfg)
    policy = _engineering_health_resource_policy(cfg)
    historical = {}
    if "learner_effective_engineering_health_submission_export_repair" in cfg:
        historical = _engineering_health_submission_export_repair(cfg)[
            "active_attempt_resource_policy"
        ]["historical_incomplete_attempt_wall_minutes"]
    elif "learner_effective_engineering_health_fixture_bind_repair" in cfg:
        historical = _engineering_health_fixture_bind_repair(cfg)[
            "active_attempt_resource_policy"
        ]["historical_incomplete_attempt_wall_minutes"]
    elif "learner_effective_engineering_health_portable_AST_repair" in cfg:
        historical = _engineering_health_portable_ast_repair(cfg)[
            "active_attempt_resource_policy"
        ]["historical_incomplete_attempt_wall_minutes"]
    elif "learner_effective_engineering_health_historical_lineage_repair" in cfg:
        historical = _engineering_health_historical_lineage_repair(cfg)[
            "active_attempt_resource_policy"
        ]["historical_incomplete_attempt_wall_minutes"]
    elif "learner_effective_engineering_health_extended_wall_repair" in cfg:
        historical = _engineering_health_extended_wall_repair(cfg)[
            "historical_incomplete_attempt_wall_minutes"
        ]
    wall_minutes = float(
        historical.get(str(attempt), policy["per_submission_wall_minutes_max"])
    )
    return {
        "GPU_type": _engineering_health_attempt_gpu_type(cfg, attempt),
        "GPU_count": int(policy["GPU_count"]),
        "CPU_count": int(policy["CPU_count"]),
        "memory_GiB": int(policy["memory_GiB"]),
        "wall_minutes": wall_minutes,
        "GPU_hours": wall_minutes / 60.0,
        "new_storage_GiB": _tuple_health_tree_bytes(attempt_root) / (1024**3),
        "direct_monetary_cost_USD": float(
            policy["direct_monetary_cost_USD"]
        ),
        "submission_started_epoch": float(marker["submission_started_epoch"]),
    }


def _tuple_health_metric_release(
    module_results: list[dict[str, Any]], scientific_metrics: Any
) -> Any:
    """Release scientific aggregates only after seven unique health passes."""

    if (
        not isinstance(module_results, list)
        or len(module_results) != len(TUPLE_QUALIFICATION_MODULE_IDS)
        or {row.get("module_id") for row in module_results}
        != set(TUPLE_QUALIFICATION_MODULE_IDS)
        or any(row.get("status") != "PASS_ENGINEERING" for row in module_results)
    ):
        raise RuntimeError("E_TUPLE_HEALTH_METRICS_WITHHELD")
    return scientific_metrics


def _tuple_health_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("E_TUPLE_HEALTH_NONFINITE")
    if isinstance(value, dict):
        for item in value.values():
            _tuple_health_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _tuple_health_finite(item)


def _validate_tuple_health_module_result(
    row: Any, *, expected_case_count: int | None = None
) -> None:
    if not isinstance(row, dict):
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    if row.get("status") == "PASS_ENGINEERING":
        required = {
            "module_id",
            "status",
            "case_count",
            "failure_count",
            "invalid_retained_record_count",
            "silent_truncation_count",
            "external_call_count",
            "unaccounted_failure_count",
            "finite_in_bounds",
            "schema_valid",
            "deterministic_outputs",
            "round_trip_valid",
            "valid_negative_state_distinct",
            "abstention_state_distinct",
            "missing_error_state_distinct",
            "production_output_commitment_sha256",
        }
        if (
            set(row) != required
            or type(row.get("case_count")) is not int
            or int(row["case_count"]) <= 0
            or (
                expected_case_count is not None
                and row["case_count"] != expected_case_count
            )
            or not isinstance(
                row.get("production_output_commitment_sha256"), str
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}", row["production_output_commitment_sha256"]
            )
        ):
            raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
        if not all(
            row[key] is True
            for key in (
                "finite_in_bounds",
                "schema_valid",
                "deterministic_outputs",
                "round_trip_valid",
            )
        ):
            raise RuntimeError("E_TUPLE_HEALTH_OUTPUT_INTEGRITY")
        if not all(
            row[key] is True
            for key in (
                "valid_negative_state_distinct",
                "abstention_state_distinct",
                "missing_error_state_distinct",
            )
        ):
            raise RuntimeError("E_TUPLE_HEALTH_STATE_DISTINCTION")
        if any(
            row[key] != 0
            for key in (
                "failure_count",
                "invalid_retained_record_count",
                "silent_truncation_count",
                "external_call_count",
                "unaccounted_failure_count",
            )
        ):
            raise RuntimeError("E_TUPLE_HEALTH_OUTPUT_INTEGRITY")
    elif (
        set(row)
        != {
            "module_id",
            "status",
            "error_code",
            "trace_written",
        }
        or row.get("status") != "ERROR"
        or row.get("trace_written") is not True
        or not isinstance(row.get("error_code"), str)
        or not re.fullmatch(r"E_TUPLE_HEALTH_[A-Z0-9_]+", row["error_code"])
    ):
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    _tuple_health_finite(row)
    if json.loads(canonical(row)) != row:
        raise RuntimeError("E_TUPLE_HEALTH_MODULE_ROUND_TRIP")


def _validate_tuple_health_full(value: Any, cfg: dict[str, Any]) -> None:
    """Validate an external full health record without scientific fields."""

    _engineering_health_amendment(cfg)
    if not isinstance(value, dict):
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    if any(
        "metrics" in str(key).casefold()
        or key in {"scientific_metrics", "scores", "predictions", "labels"}
        for key in value
    ):
        raise RuntimeError("E_TUPLE_HEALTH_SCIENTIFIC_METRIC")
    _tuple_health_finite(value)
    if value.get("holdout_input_count") != 0:
        raise RuntimeError("E_TUPLE_HEALTH_HOLDOUT_PROHIBITED")
    modules = value.get("module_results")
    expected_config_commitments = {digest(cfg)}
    if (
        "learner_effective_engineering_health_reauthorization" in cfg
        and type(value.get("attempt")) is int
        and int(value["attempt"]) <= 3
    ):
        _engineering_health_reauthorization(cfg)
        expected_config_commitments.add(
            cfg["learner_effective_engineering_health_result"][
                "compact_aggregate"
            ]["config_commitment_sha256"]
        )
    if (
        "learner_effective_engineering_health_parser_repair_result" in cfg
        and value.get("attempt") == 5
    ):
        _engineering_health_parser_repair_result(cfg)
        expected_config_commitments.add(
            cfg["learner_effective_engineering_health_parser_repair_result"]
            ["submission_provenance"]["config_commitment_sha256"]
        )
    if (
        "learner_effective_engineering_health_historical_lineage_repair" in cfg
        and value.get("attempt") == 8
    ):
        attempt_8 = _engineering_health_attempt_8_result(cfg)["compact_aggregate"]
        expected_config_commitments.add(
            attempt_8["config_commitment_sha256"]
        )
        if (
            value.get("engineering_health_commitment_sha256")
            != attempt_8["engineering_health_commitment_sha256"]
        ):
            raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    if (
        "learner_effective_engineering_health_portable_AST_repair" in cfg
        and value.get("attempt") == 10
    ):
        attempt_10 = _engineering_health_attempt_10_result(cfg)[
            "compact_aggregate"
        ]
        expected_config_commitments.add(
            attempt_10["config_commitment_sha256"]
        )
        if (
            value.get("engineering_health_commitment_sha256")
            != attempt_10["engineering_health_commitment_sha256"]
        ):
            raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    if (
        "learner_effective_engineering_health_fixture_bind_repair" in cfg
        and value.get("attempt") == 11
    ):
        attempt_11 = _engineering_health_attempt_11_result(cfg)[
            "compact_aggregate"
        ]
        expected_config_commitments.add(
            attempt_11["config_commitment_sha256"]
        )
        if (
            value.get("engineering_health_commitment_sha256")
            != attempt_11["engineering_health_commitment_sha256"]
        ):
            raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    if (
        set(value) != set(TUPLE_HEALTH_FULL_FIELDS)
        or value.get("schema_version") != 1
        or value.get("route_id") != "construct-aligned-engineering-health"
        or value.get("module_count") != len(TUPLE_QUALIFICATION_MODULE_IDS)
        or value.get("case_count") != 28
        or value.get("scientific_metric_count") != 0
        or value.get("config_commitment_sha256")
        not in expected_config_commitments
        or not isinstance(modules, list)
        or len(modules) != len(TUPLE_QUALIFICATION_MODULE_IDS)
        or {row.get("module_id") for row in modules}
        != set(TUPLE_QUALIFICATION_MODULE_IDS)
        or value.get("network_disabled") is not True
        or value.get("telemetry_disabled") is not True
        or value.get("restricted_mount_present") is not False
    ):
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    pass_count = sum(row.get("status") == "PASS_ENGINEERING" for row in modules)
    error_count = sum(row.get("status") == "ERROR" for row in modules)
    if pass_count + error_count != len(modules):
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    if value.get("completed_module_count") != pass_count or value.get(
        "failed_module_count"
    ) != error_count:
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    if (
        value.get("failure_count") != error_count
        or value.get("invalid_retained_record_count") != 0
        or value.get("silent_truncation_count") != 0
        or value.get("external_call_count") != 0
        or value.get("unaccounted_failure_count")
        != sum(
            str(row.get("error_code", "")).endswith("UNACCOUNTED_FAILURE")
            for row in modules
        )
    ):
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    if value.get("status") == "PASS_ENGINEERING_HEALTH":
        if error_count or pass_count != len(modules):
            raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    elif value.get("status") != "ENGINEERING_BLOCKER":
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    for row in modules:
        _validate_tuple_health_module_result(row, expected_case_count=4)
    for key in (
        "public_fixture_manifest_commitment_sha256",
        "runner_commitment_sha256",
        "config_commitment_sha256",
        "dependency_config_commitment_sha256",
        "microfixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
    ):
        if not isinstance(value.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", value[key]
        ):
            raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    resource = value.get("resource")
    cumulative = value.get("cumulative_resource")
    if (
        not isinstance(resource, dict)
        or set(resource)
        != {
            "GPU_type",
            "GPU_count",
            "CPU_count",
            "memory_GiB",
            "wall_minutes",
            "GPU_hours",
            "new_storage_GiB",
            "direct_monetary_cost_USD",
        }
        or not isinstance(cumulative, dict)
        or set(cumulative)
        != {
            "cumulative_submission_count",
            "cumulative_wall_minutes",
            "cumulative_GPU_hours",
            "cumulative_new_storage_GiB",
            "cumulative_direct_monetary_cost_USD",
        }
    ):
        raise RuntimeError("E_TUPLE_HEALTH_FULL_SCHEMA")
    policy = _engineering_health_resource_policy(cfg)
    attempt = int(value.get("attempt", 0))
    expected_gpu_type = _engineering_health_attempt_gpu_type(cfg, attempt)
    _tuple_health_budget(
        attempt,
        [
            {
                "attempt": ordinal,
                "GPU_type": _engineering_health_attempt_gpu_type(cfg, ordinal),
                "GPU_count": policy["GPU_count"],
                "wall_minutes": 0.0,
                "GPU_hours": 0.0,
                "new_storage_GiB": 0.0,
                "direct_monetary_cost_USD": 0,
            }
            for ordinal in range(1, attempt)
        ],
        cfg,
    )
    if (
        resource.get("GPU_type") != expected_gpu_type
        or resource.get("GPU_count") != policy["GPU_count"]
        or resource.get("CPU_count") != policy["CPU_count"]
        or resource.get("memory_GiB") != policy["memory_GiB"]
    ):
        raise RuntimeError("E_TUPLE_HEALTH_GPU_TOPOLOGY")
    wall = float(resource.get("wall_minutes", math.inf))
    gpu_hours = float(resource.get("GPU_hours", math.inf))
    storage = float(resource.get("new_storage_GiB", math.inf))
    cost = float(resource.get("direct_monetary_cost_USD", math.inf))
    if any(value < 0 for value in (wall, gpu_hours, storage, cost)):
        raise RuntimeError("E_TUPLE_HEALTH_RESOURCE_NONFINITE")
    if wall > float(policy["per_submission_wall_minutes_max"]):
        raise RuntimeError("E_TUPLE_HEALTH_WALL_BUDGET")
    if gpu_hours > float(policy["aggregate_GPU_hours_max"]):
        raise RuntimeError("E_TUPLE_HEALTH_GPU_HOUR_BUDGET")
    if storage > float(policy["new_storage_GiB_max"]):
        raise RuntimeError("E_TUPLE_HEALTH_STORAGE_BUDGET")
    if cost != float(policy["direct_monetary_cost_USD"]):
        raise RuntimeError("E_TUPLE_HEALTH_COST_BUDGET")
    if (
        cumulative.get("cumulative_submission_count") != value.get("attempt")
        or float(cumulative.get("cumulative_wall_minutes", math.inf)) < wall
        or float(cumulative.get("cumulative_GPU_hours", math.inf)) < gpu_hours
        or float(cumulative.get("cumulative_new_storage_GiB", math.inf)) < storage
        or float(
            cumulative.get("cumulative_direct_monetary_cost_USD", math.inf)
        )
        != cost
        or float(cumulative["cumulative_GPU_hours"])
        > float(policy["aggregate_GPU_hours_max"])
        or float(cumulative["cumulative_new_storage_GiB"])
        > float(policy["new_storage_GiB_max"])
    ):
        raise RuntimeError("E_TUPLE_HEALTH_CUMULATIVE_RESOURCE")


def _tuple_health_compact(full: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "attempt",
        "module_count",
        "completed_module_count",
        "failed_module_count",
        "case_count",
        "holdout_input_count",
        "scientific_metric_count",
        "failure_count",
        "invalid_retained_record_count",
        "silent_truncation_count",
        "external_call_count",
        "unaccounted_failure_count",
        "public_fixture_manifest_commitment_sha256",
        "runner_commitment_sha256",
        "config_commitment_sha256",
        "dependency_config_commitment_sha256",
        "microfixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
    )
    compact = {key: full[key] for key in keys if key in full}
    compact.update(full["resource"])
    compact.update(full["cumulative_resource"])
    _validate_tuple_health_compact(compact)
    return compact


def _validate_tuple_health_compact(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != set(TUPLE_HEALTH_FIELDS)
        or value.get("status")
        not in {"PASS_ENGINEERING_HEALTH", "ENGINEERING_BLOCKER"}
        or value.get("scientific_metric_count") != 0
        or value.get("holdout_input_count") != 0
    ):
        raise RuntimeError("E_TUPLE_HEALTH_COMPACT_SCHEMA")
    for key, item in value.items():
        if isinstance(item, (dict, list, tuple)):
            raise RuntimeError("E_TUPLE_HEALTH_COMPACT_SCHEMA")
        if key.endswith("_sha256") and (
            not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
        ):
            raise RuntimeError("E_TUPLE_HEALTH_COMPACT_SCHEMA")
    _tuple_health_finite(value)


def _tuple_qualification_execution(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the pre-outcome execution clarification for tuple qualification."""

    correction = _tuple_visor_hos_correction_amendment(cfg)
    value = correction.get("qualification_execution_clarification")
    if not isinstance(value, dict):
        raise RuntimeError("E_TUPLE_QUALIFICATION_EXECUTION_NOT_FROZEN")
    if value.get("status") != "FROZEN_BEFORE_PUBLIC_DEVELOPMENT_INFERENCE":
        raise RuntimeError("E_TUPLE_QUALIFICATION_EXECUTION_NOT_FROZEN")
    phase = value.get("phase_aggregation", {})
    if (
        phase.get("requested_samples") != 9
        or phase.get("minimum_valid_samples") != 8
        or phase.get("phase_sample_count") != 3
        or phase.get("phase_visible_if_valid_positive_samples_min") != 2
    ):
        raise RuntimeError("E_TUPLE_QUALIFICATION_PHASE_RULE")
    phash = value.get("pHash", {})
    if (
        phash.get("algorithm")
        != "32x32_grayscale_DCT_low_8x8_excluding_DC_median_bits"
        or phash.get("near_duplicate_hamming_distance_max") != 4
    ):
        raise RuntimeError("E_TUPLE_QUALIFICATION_PHASH_RULE")
    sensor = value.get("sensor_direction_checks")
    if not isinstance(sensor, list) or len(sensor) != 6:
        raise RuntimeError("E_TUPLE_QUALIFICATION_SENSOR_RULE")
    grids = value.get("grid_selection", {})
    if set(grids) != {"grounding", "recurrence", "attribute", "hand", "action"}:
        raise RuntimeError("E_TUPLE_QUALIFICATION_GRID_RULE")
    return value


def _tuple_referent_audio_fixture(cfg: dict[str, Any]) -> dict[str, Any]:
    correction = _tuple_visor_hos_correction_amendment(cfg)
    value = correction["qualification_execution_clarification"].get(
        "referent_audio_fixture"
    )
    expected_articles = {
        "sports ball": "der",
        "cup": "der",
        "bottle": "die",
        "bowl": "die",
        "book": "das",
        "chair": "der",
        "apple": "der",
        "banana": "die",
    }
    if (
        not isinstance(value, dict)
        or value.get("active_external_location")
        != "public/mechanistic-training-tuple-fixtures/audio-seed"
        or "public/mechanistic-tuple-audio-seed" not in str(
            value.get("prior_predicative_seed_rule", "")
        )
        or value.get("language") != "de"
        or value.get("voice") != "macOS Anna de_DE at 175 words per minute"
        or value.get("development_template")
        != "{definite_article_capitalized} {inflected_attribute} {noun}."
        or value.get("holdout_template")
        != "Schau, {definite_article_lowercase} {inflected_attribute} {noun}."
        or value.get("definite_articles_by_public_category") != expected_articles
        or value.get("ordered_inflected_attributes_by_scenario_ordinal")
        != [
            "rote",
            "blaue",
            "grüne",
            "gelbe",
            "große",
            "kleine",
            "rote",
            "blaue",
        ]
        or value.get("muxed_speech_delay_seconds") != 2.5
        or value.get("maximum_spoken_audio_seconds") != 2.0
        or "actual pinned shared adapter output" not in str(
            value.get("prediction_rule", "")
        )
    ):
        raise RuntimeError("E_TUPLE_AUDIO_FIXTURE_RECIPE")
    return value


def _tuple_audio_phrase(
    recipe: dict[str, Any], partition: str, category: str, ordinal: int, noun: str
) -> str:
    article = recipe["definite_articles_by_public_category"].get(category)
    attributes = recipe["ordered_inflected_attributes_by_scenario_ordinal"]
    if (
        partition not in {"development", "holdout"}
        or not isinstance(article, str)
        or not 0 <= ordinal < len(attributes)
        or not isinstance(noun, str)
        or not noun
    ):
        raise RuntimeError("E_TUPLE_AUDIO_FIXTURE_PHRASE")
    if partition == "development":
        return recipe["development_template"].format(
            definite_article_capitalized=article.capitalize(),
            inflected_attribute=attributes[ordinal],
            noun=noun,
        )
    return recipe["holdout_template"].format(
        definite_article_lowercase=article,
        inflected_attribute=attributes[ordinal],
        noun=noun,
    )


def _tuple_fixture_file(
    fixture_root: Path,
    relative_value: Any,
    expected_sha256: Any,
    expected_bytes: Any | None = None,
) -> Path:
    relative = Path(str(relative_value))
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or any(not part or part in {".", ".."} for part in relative.parts)
    ):
        raise RuntimeError("E_TUPLE_QUALIFICATION_MEDIA_PATH")
    target = fixture_root / relative
    try:
        target.resolve().relative_to(fixture_root.resolve())
    except ValueError as error:
        raise RuntimeError("E_TUPLE_QUALIFICATION_MEDIA_PATH") from error
    if not target.is_file() or not isinstance(expected_sha256, str):
        raise RuntimeError("E_TUPLE_QUALIFICATION_MEDIA_MISSING")
    if file_digest(target) != expected_sha256:
        raise RuntimeError("E_TUPLE_QUALIFICATION_MEDIA_HASH")
    if expected_bytes is not None and target.stat().st_size != int(expected_bytes):
        raise RuntimeError("E_TUPLE_QUALIFICATION_MEDIA_BYTES")
    return target


TUPLE_RECURRENCE_FIXTURE_FIELDS = frozenset(
    {
        "fixture_ordinal",
        "stratum",
        "same_referent",
        "near_duplicate",
        "source_image_ids",
        "first_relative_path",
        "first_sha256",
        "first_bytes",
        "second_relative_path",
        "second_sha256",
        "second_bytes",
        "first_mask_relative_path",
        "first_mask_sha256",
        "first_mask_bytes",
        "second_mask_relative_path",
        "second_mask_sha256",
        "second_mask_bytes",
    }
)
TUPLE_RECURRENCE_STRATA = frozenset(
    {
        "same_instance_transformed",
        "same_instance_near_duplicate",
        "same_category_different_instance",
        "different_category",
    }
)


def _validate_tuple_recurrence_fixture_rows(
    rows: Any, fixture_root: Path | None = None
) -> None:
    """Fail closed on recurrence lineage and full-canvas binary alpha masks."""

    from PIL import Image

    if not isinstance(rows, list):
        raise RuntimeError("E_TUPLE_RECURRENCE_FIXTURE_SCHEMA")
    ordinals: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != TUPLE_RECURRENCE_FIXTURE_FIELDS:
            raise RuntimeError("E_TUPLE_RECURRENCE_FIXTURE_SCHEMA")
        ordinal = row["fixture_ordinal"]
        source_ids = row["source_image_ids"]
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or ordinal in ordinals
            or row["stratum"] not in TUPLE_RECURRENCE_STRATA
            or not isinstance(row["same_referent"], bool)
            or not isinstance(row["near_duplicate"], bool)
            or not isinstance(source_ids, list)
            or len(source_ids) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in source_ids)
        ):
            raise RuntimeError("E_TUPLE_RECURRENCE_FIXTURE_SCHEMA")
        ordinals.add(ordinal)
        resolved: dict[str, Path] = {}
        for prefix in ("first", "second", "first_mask", "second_mask"):
            relative = row[f"{prefix}_relative_path"]
            sha256 = row[f"{prefix}_sha256"]
            byte_count = row[f"{prefix}_bytes"]
            if (
                not isinstance(relative, str)
                or not relative
                or not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
                or isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count <= 0
            ):
                raise RuntimeError("E_TUPLE_RECURRENCE_FIXTURE_SCHEMA")
            if fixture_root is not None:
                resolved[prefix] = _tuple_fixture_file(
                    fixture_root, relative, sha256, byte_count
                )
        if fixture_root is None:
            continue
        image_sizes = {}
        for prefix in ("first", "second"):
            with Image.open(resolved[prefix]) as source:
                if source.format != "PNG":
                    raise RuntimeError("E_TUPLE_RECURRENCE_IMAGE")
                image_sizes[prefix] = source.size
        for prefix, image_prefix in (
            ("first_mask", "first"),
            ("second_mask", "second"),
        ):
            with Image.open(resolved[prefix]) as source:
                if source.format != "PNG" or source.mode != "L":
                    raise RuntimeError("E_TUPLE_RECURRENCE_MASK")
                mask = source.copy()
            active_values = {
                value for value, count in enumerate(mask.histogram()) if count
            }
            if (
                mask.size != image_sizes[image_prefix]
                or mask.size != (224, 224)
                or not active_values <= {0, 255}
                or 255 not in active_values
            ):
                raise RuntimeError("E_TUPLE_RECURRENCE_MASK")


def _verify_tuple_fixture_files_recursive(value: Any, fixture_root: Path) -> int:
    """Hash every manifest-referenced fixture file before model inference."""

    if isinstance(value, list):
        return sum(
            _verify_tuple_fixture_files_recursive(item, fixture_root)
            for item in value
        )
    if not isinstance(value, dict):
        return 0
    count = 0
    for key, relative in value.items():
        if not key.endswith("_relative_path"):
            continue
        prefix = key[: -len("_relative_path")]
        hash_key = f"{prefix}_sha256"
        if hash_key not in value:
            raise RuntimeError("E_TUPLE_QUALIFICATION_MEDIA_HASH_FIELD")
        if relative is None and value[hash_key] is None:
            continue
        _tuple_fixture_file(
            fixture_root,
            relative,
            value[hash_key],
            value.get(f"{prefix}_bytes"),
        )
        count += 1
    for item in value.values():
        if isinstance(item, (dict, list)):
            count += _verify_tuple_fixture_files_recursive(item, fixture_root)
    return count


def _verify_tuple_fixture_commitments(
    manifest: dict[str, Any], commitments: dict[str, str]
) -> None:
    for key, value in commitments.items():
        if manifest.get(key) != value:
            raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_PROVENANCE")


def _verify_tuple_fixture_manifest(
    public: Path, cfg: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    """Verify the complete external fixture seal before any model is loaded."""

    active = _construct_aligned_ltx_resume_amendment(cfg)
    amendment = _tuple_amendment(cfg)
    runtime = _tuple_runtime_amendment(cfg)
    protocol = _tuple_fixture_protocol(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    correction = _tuple_visor_hos_correction_amendment(cfg)
    rasterization_repair = _public_fixture_geometry_rasterization_repair(cfg)
    _tuple_qualification_execution(cfg)
    fixture_root = _tuple_fixture_root(public)
    source_feasibility = _load_construct_aligned_visor_hos_source_reuse(
        fixture_root, cfg
    )
    verified_no_hand = _load_visor_hos_verified_no_hand_lineage(
        fixture_root / "no-hand-review",
        expected_source_feasibility_commitment_sha256=source_feasibility[
            "visor_hos_source_feasibility_commitment_sha256"
        ],
        expected_construct_aligned_amendment_commitment_sha256=active[
            "amendment_commitment_sha256"
        ],
    )
    path = fixture_root / "fixture-manifest.json"
    if not path.is_file():
        raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_MANIFEST_MISSING")
    manifest = json.loads(path.read_text())
    payload = json.loads(json.dumps(manifest))
    expected = payload.pop("public_fixture_manifest_commitment_sha256", None)
    if not isinstance(expected, str) or digest(payload) != expected:
        raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_COMMITMENT")
    if (
        manifest.get("schema_version") != 3
        or manifest.get("status") != "SEALED_BEFORE_PUBLIC_DEVELOPMENT_INFERENCE"
    ):
        raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_STATUS")
    commitments = {
        "mechanistic_tuple_amendment_commitment_sha256": amendment[
            "amendment_commitment_sha256"
        ],
        "public_fixture_protocol_commitment_sha256": protocol[
            "protocol_commitment_sha256"
        ],
        "fixture_preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "visor_hos_correction_amendment_commitment_sha256": correction[
            "amendment_commitment_sha256"
        ],
        "visor_hos_source_feasibility_commitment_sha256": source_feasibility[
            "visor_hos_source_feasibility_commitment_sha256"
        ],
        "verified_no_hand_seal_commitment_sha256": verified_no_hand[
            "commitment_sha256"
        ],
        "construct_aligned_ltx_resume_amendment_commitment_sha256": active[
            "amendment_commitment_sha256"
        ],
        "public_fixture_geometry_rasterization_repair_commitment_sha256": (
            rasterization_repair["repair_commitment_sha256"]
        ),
    }
    _verify_tuple_fixture_commitments(manifest, commitments)
    no_hand_commitment = verified_no_hand["commitment_sha256"]
    if manifest.get("restricted_mount_present") is not False:
        raise RuntimeError("E_TUPLE_QUALIFICATION_RESTRICTED_MOUNT")
    if manifest.get("model_inference_executed") is not False:
        raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_OUTCOME_ORDER")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        "development",
        "holdout",
    }:
        raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_PARTITIONS")
    expected_counts = correction["public_fixture_counts_per_partition"]
    family_count_keys = {
        "language_lexical": "language_and_lexical_items",
        "referent_attribute": "referent_attribute_microclips",
        "recurrence": "recurrence_pairs",
        "hand_contact": "hand_contact_items",
        "sensor": "sensor_perturbation_clips",
        "order_action": "order_dependent_action_clips",
    }
    for partition in ("development", "holdout"):
        rows = partitions[partition]
        if not isinstance(rows, dict) or set(rows) != set(family_count_keys):
            raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_FAMILIES")
        for family, count_key in family_count_keys.items():
            expected_count = (
                CONSTRUCT_ALIGNED_ACTION_COUNTS[partition]
                if family == "order_action"
                else int(expected_counts[count_key])
            )
            if not isinstance(rows[family], list) or len(rows[family]) != int(
                expected_count
            ):
                raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_COUNT")
        _validate_construct_aligned_action_fixture_projection(
            rows["order_action"],
            source_feasibility["selections"]["charades_order_action"][partition],
            partition,
        )
        for row in rows["hand_contact"]:
            if not isinstance(row, dict):
                raise RuntimeError("E_TUPLE_QUALIFICATION_HAND_FIXTURE")
            if row.get("stratum") == "verified_no_hand" and row.get(
                "verified_no_hand_seal_commitment_sha256"
            ) != no_hand_commitment:
                raise RuntimeError("E_TUPLE_QUALIFICATION_NO_HAND_ROW_SEAL")
        hand_strata = Counter(row.get("stratum") for row in rows["hand_contact"])
        if hand_strata != {
            "contact": 48,
            "explicit_no_contact": 48,
            "verified_no_hand": 48,
        }:
            raise RuntimeError("E_TUPLE_QUALIFICATION_HAND_STRATA")
        _validate_tuple_egohos_fixture_rows(
            {
                "verified_no_hand_seal": {
                    "status": "PASS",
                    "verified_no_hand_seal_commitment_sha256": no_hand_commitment,
                }
            },
            rows["hand_contact"],
        )
        for row in rows["hand_contact"]:
            if row["stratum"] != "verified_no_hand":
                _read_tuple_egohos_target_mask(row, fixture_root)
        _validate_tuple_recurrence_fixture_rows(
            rows["recurrence"], fixture_root
        )
    audits = manifest.get("audits", {})
    for key in (
        "source_subject_overlap_count",
        "source_video_overlap_count",
        "source_frame_overlap_count",
        "source_object_overlap_count",
    ):
        if int(audits.get(key, -1)) != 0:
            raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_OVERLAP")
    visible_hand_rows = [
        row
        for partition in ("development", "holdout")
        for row in partitions[partition]["hand_contact"]
        if row["stratum"] != "verified_no_hand"
    ]
    expected_geometry_audits = {
        "target_hand_boundary_vertex_count": sum(
            int(row["target_hand_boundary_vertex_count"])
            for row in visible_hand_rows
        ),
        "target_hand_outside_canvas_vertex_count": sum(
            int(row["target_hand_outside_canvas_vertex_count"])
            for row in visible_hand_rows
        ),
        "target_hand_boundary_item_count": sum(
            int(int(row["target_hand_boundary_vertex_count"]) > 0)
            for row in visible_hand_rows
        ),
        "target_hand_outside_canvas_item_count": sum(
            int(int(row["target_hand_outside_canvas_vertex_count"]) > 0)
            for row in visible_hand_rows
        ),
    }
    if (
        audits.get("all_source_polygons_finite_nonnegative") is not True
        or audits.get(
            "all_rasterized_target_masks_binary_nonempty_exact_frame_in_bounds"
        )
        is not True
        or any(
            audits.get(key) != value
            for key, value in expected_geometry_audits.items()
        )
    ):
        raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_GEOMETRY_AUDIT")
    if _verify_tuple_fixture_files_recursive(partitions, fixture_root) <= 0:
        raise RuntimeError("E_TUPLE_QUALIFICATION_FIXTURE_FILES")
    if runtime["local_reload_gate"].get("zero_external_calls_crashes_silent_truncations_or_invalid_records") is not True:
        raise RuntimeError("E_TUPLE_QUALIFICATION_RUNTIME_GATE")
    return manifest, fixture_root


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _binary_classification_metrics(
    expected: list[bool], predicted: list[bool | None]
) -> dict[str, float]:
    if len(expected) != len(predicted) or not expected:
        raise RuntimeError("E_TUPLE_QUALIFICATION_BINARY_METRIC_INPUT")
    tp = sum(want and got is True for want, got in zip(expected, predicted, strict=True))
    tn = sum(not want and got is False for want, got in zip(expected, predicted, strict=True))
    fp = sum(not want and got is True for want, got in zip(expected, predicted, strict=True))
    fn = sum(want and got is not True for want, got in zip(expected, predicted, strict=True))
    positive = sum(expected)
    negative = len(expected) - positive
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, positive)
    specificity = _safe_divide(tn, negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "coverage": _safe_divide(sum(value is not None for value in predicted), len(predicted)),
    }


def _multiclass_macro_f1(
    expected: list[str], predicted: list[str | None], labels: list[str]
) -> float:
    if len(expected) != len(predicted) or not expected or len(labels) != len(set(labels)):
        raise RuntimeError("E_TUPLE_QUALIFICATION_MULTICLASS_METRIC_INPUT")
    values = []
    for label in labels:
        truth = [value == label for value in expected]
        guesses = [None if value is None else value == label for value in predicted]
        values.append(_binary_classification_metrics(truth, guesses)["f1"])
    return sum(values) / len(values)


def _weighted_kappa(
    expected: list[str], predicted: list[str | None], labels: list[str]
) -> float:
    """Linearly weighted Cohen kappa; abstentions are excluded and lower coverage is separate."""

    paired = [
        (want, got)
        for want, got in zip(expected, predicted, strict=True)
        if got is not None
    ]
    if not paired or len(labels) < 2:
        return 0.0
    index = {label: ordinal for ordinal, label in enumerate(labels)}
    if any(want not in index or got not in index for want, got in paired):
        raise RuntimeError("E_TUPLE_QUALIFICATION_KAPPA_LABEL")
    size = len(labels)
    observed = [[0.0] * size for _ in range(size)]
    truth_counts = [0.0] * size
    guess_counts = [0.0] * size
    for want, got in paired:
        first, second = index[want], index[str(got)]
        observed[first][second] += 1.0
        truth_counts[first] += 1.0
        guess_counts[second] += 1.0
    count = float(len(paired))
    disagreement = 0.0
    expected_disagreement = 0.0
    for first in range(size):
        for second in range(size):
            weight = abs(first - second) / (size - 1)
            disagreement += weight * observed[first][second] / count
            expected_disagreement += (
                weight * truth_counts[first] * guess_counts[second] / (count * count)
            )
    if expected_disagreement == 0.0:
        return 1.0 if disagreement == 0.0 else 0.0
    return 1.0 - disagreement / expected_disagreement


def _select_frozen_grid_result(
    rows: list[dict[str, Any]],
    *,
    primary_metric: str,
    threshold_fields: tuple[str, ...],
    require_eligible: bool = True,
) -> dict[str, Any] | None:
    """Choose one development row, conservatively resolving exact ties."""

    candidates = (
        [row for row in rows if row.get("eligible") is True]
        if require_eligible
        else list(rows)
    )
    if not candidates:
        return None
    for row in candidates:
        values = [row.get(primary_metric), *(row.get(key) for key in threshold_fields)]
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise RuntimeError("E_TUPLE_QUALIFICATION_GRID_RESULT")
    return max(
        candidates,
        key=lambda row: (
            float(row[primary_metric]),
            *(float(row[key]) for key in threshold_fields),
        ),
    )


def _tuple_module_failure(error: BaseException) -> dict[str, Any]:
    message = str(error)
    match = re.fullmatch(r"(E_[A-Z0-9_]+)(?:\s.*)?", message)
    return {
        "status": "ERROR",
        "error_code": match.group(1) if match else "E_UNACCOUNTED_MODULE_FAILURE",
        "metrics": {},
        "row_count": 0,
        "failure_count": 1,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
    }


def _collect_tuple_module_results(
    runners: dict[str, Any], context: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Run every independent public module before making one combined decision."""

    if set(runners) != set(TUPLE_QUALIFICATION_MODULE_IDS):
        raise RuntimeError("E_TUPLE_QUALIFICATION_MODULE_SET")
    output = {}
    for module_id in TUPLE_QUALIFICATION_MODULE_IDS:
        try:
            result = runners[module_id](context)
            allowed_statuses = {"PASS", "NO_GO", "UNMEASURED"}
            if module_id == "order_action":
                allowed_statuses.add("NO_GO_DIAGNOSTIC")
            if (
                not isinstance(result, dict)
                or result.get("status") not in allowed_statuses
            ):
                raise RuntimeError("E_TUPLE_QUALIFICATION_MODULE_RESULT")
            output[module_id] = result
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            output[module_id] = _tuple_module_failure(error)
    return output


def _tuple_axis(cfg: dict[str, Any], axis_id: str) -> dict[str, Any]:
    matches = [axis for axis in _tuple_amendment(cfg)["axes"] if axis["id"] == axis_id]
    if len(matches) != 1:
        raise RuntimeError("E_TUPLE_QUALIFICATION_AXIS_CONFIG")
    return matches[0]


def _tuple_public_ontology_mapping(preparation: dict[str, Any]) -> dict[str, list[str]]:
    mapping = {
        "sports ball": ["ball", "sports ball"],
        "cup": ["cup"],
        "bottle": ["bottle"],
        "bowl": ["bowl"],
        "book": ["book"],
        "chair": ["chair"],
        "apple": ["apple"],
        "banana": ["banana"],
    }
    if list(mapping) != preparation["public_object_ontology"]:
        raise RuntimeError("E_TUPLE_QUALIFICATION_ONTOLOGY")
    return mapping


def _adjacent_adjective_noun_spans(
    mentions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return only actual adjacent adjective-to-noun lexical events."""

    by_index: dict[int, dict[str, Any]] = {}
    for mention in mentions:
        index = mention.get("token_index")
        if not isinstance(index, int) or index < 0 or index in by_index:
            raise RuntimeError("E_TUPLE_ATTRIBUTE_SPAN_INDEX")
        if mention.get("part_of_speech") not in {"noun", "adjective"}:
            raise RuntimeError("E_TUPLE_ATTRIBUTE_SPAN_PART")
        lemma = mention.get("lemma")
        if not isinstance(lemma, str) or not re.fullmatch(
            r"[a-z]+(?:'[a-z]+)?", lemma
        ):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_SPAN_LEMMA")
        by_index[index] = mention
    output = []
    for index in sorted(by_index):
        adjective = by_index[index]
        noun = by_index.get(index + 1)
        if (
            adjective["part_of_speech"] == "adjective"
            and noun is not None
            and noun["part_of_speech"] == "noun"
        ):
            output.append(
                {"adjective": adjective["lemma"], "noun": noun["lemma"]}
            )
    return output


def _span_f1(
    expected: list[tuple[str, str, str]],
    predicted: list[tuple[str, str, str]],
) -> float:
    truth, guesses = Counter(expected), Counter(predicted)
    matched = sum((truth & guesses).values())
    precision = _safe_divide(matched, sum(guesses.values()))
    recall = _safe_divide(matched, sum(truth.values()))
    return _safe_divide(2.0 * precision * recall, precision + recall)


def _tuple_lexical_truth_checks(
    expected_mentions: list[dict[str, Any]],
    predicted_mentions: list[dict[str, Any]],
) -> list[bool]:
    output = []
    for ordinal, expected in enumerate(expected_mentions):
        if not isinstance(expected, dict) or not {
            "token",
            "part_of_speech",
            "expected_lemma",
            "expected_frequency_band",
        } <= set(expected):
            raise RuntimeError("E_TUPLE_LEXICAL_TRUTH_SCHEMA")
        if ordinal >= len(predicted_mentions):
            output.append(False)
            continue
        predicted = predicted_mentions[ordinal]
        output.append(
            str(predicted.get("token", "")).casefold()
            == str(expected["token"]).casefold()
            and predicted.get("part_of_speech") == expected["part_of_speech"]
            and predicted.get("lemma") == str(expected["expected_lemma"]).casefold()
            and predicted.get("frequency_band")
            == expected["expected_frequency_band"]
        )
    if len(predicted_mentions) > len(expected_mentions):
        output.extend([False] * (len(predicted_mentions) - len(expected_mentions)))
    return output


def _tuple_health_normalize_output(value: Any) -> Any:
    """Normalize an in-memory public output for private round-trip hashing."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("E_TUPLE_HEALTH_NONFINITE")
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise RuntimeError("E_TUPLE_HEALTH_MODULE_ROUND_TRIP")
        return {
            key: _tuple_health_normalize_output(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_tuple_health_normalize_output(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _tuple_health_normalize_output(item())
    raise RuntimeError("E_TUPLE_HEALTH_MODULE_ROUND_TRIP")


def _tuple_health_pass_result(
    module_id: str, case_count: int, production_output: Any
) -> dict[str, Any]:
    if module_id not in TUPLE_QUALIFICATION_MODULE_IDS or case_count <= 0:
        raise RuntimeError("E_TUPLE_HEALTH_MODULE_RESULT")
    normalized = _tuple_health_normalize_output(production_output)
    serialized = canonical(normalized)
    if json.loads(serialized) != normalized:
        raise RuntimeError("E_TUPLE_HEALTH_MODULE_ROUND_TRIP")
    value = {
        "module_id": module_id,
        "status": "PASS_ENGINEERING",
        "case_count": int(case_count),
        "failure_count": 0,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
        "unaccounted_failure_count": 0,
        "finite_in_bounds": True,
        "schema_valid": True,
        "deterministic_outputs": True,
        "round_trip_valid": True,
        "valid_negative_state_distinct": True,
        "abstention_state_distinct": True,
        "missing_error_state_distinct": True,
        "production_output_commitment_sha256": digest(normalized),
    }
    if json.loads(canonical(value)) != value:
        raise RuntimeError("E_TUPLE_HEALTH_MODULE_ROUND_TRIP")
    return value


def _tuple_language_lexical_health(
    context: dict[str, Any],
    validate_asr_prediction: Any,
    tagger: Any,
    lemmatize: Any,
    zipf_frequency: Any,
    mapping: dict[str, list[str]],
    frequency_bands: dict[str, list[float]],
) -> dict[str, Any]:
    """Exercise the exact adapter/lexical path without reading truth metrics."""

    amendment = _tuple_amendment(context["cfg"])
    states: set[str] = set()
    grounding_states: set[str] = set()
    production_output = []
    for row in context["rows"]["language_lexical"]:
        adjudication = validate_asr_prediction(
            row["prediction"], float(row["audio_duration"])
        )
        status = str(adjudication.get("status"))
        reason = adjudication.get("reason")
        if status not in {"ACCEPT", "ABSTAIN"} or (
            status == "ABSTAIN" and not isinstance(reason, str)
        ):
            raise RuntimeError("E_TUPLE_HEALTH_ADAPTER_STATE")
        states.add(status)
        output = {"adapter_status": status, "adapter_reason": reason}
        if status != "ACCEPT":
            grounding_states.add("ABSTAIN")
            output["grounding_status"] = "ABSTAIN"
            production_output.append(output)
            continue
        if row.get("translation_status") != "ACCEPT":
            grounding_states.add("ABSTAIN")
            output["grounding_status"] = "ABSTAIN"
            production_output.append(output)
            continue
        window = _tuple_segment_window(
            row["segment"], float(row["audio_duration"]), amendment
        )
        if window.get("status") != "ACCEPT":
            grounding_states.add("ABSTAIN")
            output["grounding_status"] = "ABSTAIN"
            production_output.append(output)
            continue
        mentions = _lexical_mentions(
            window["text_en"],
            tagger,
            lemmatize,
            zipf_frequency,
            frequency_bands,
        )
        for mention in mentions:
            if (
                mention.get("part_of_speech") not in {"noun", "adjective"}
                or not isinstance(mention.get("lemma"), str)
                or not isinstance(mention.get("frequency_band"), str)
            ):
                raise RuntimeError("E_TUPLE_HEALTH_LEXICAL_SCHEMA")
        noun_mappings = [
            _map_public_ontology(mention["lemma"], mapping)
            for mention in mentions
            if mention["part_of_speech"] == "noun"
        ]
        grounding_status = (
            "ACCEPT"
            if any(value.get("status") == "ACCEPT" for value in noun_mappings)
            else "ABSTAIN"
        )
        grounding_states.add(grounding_status)
        output.update(
            {
                "grounding_status": grounding_status,
                "mentions": [
                    {
                        key: mention.get(key)
                        for key in ("lemma", "part_of_speech", "frequency_band")
                    }
                    for mention in mentions
                ],
                "noun_mapping_statuses": [
                    value.get("status") for value in noun_mappings
                ],
                "adjective_noun_span_count": len(
                    _adjacent_adjective_noun_spans(mentions)
                ),
            }
        )
        production_output.append(output)
    if states != {"ACCEPT", "ABSTAIN"} or not grounding_states:
        raise RuntimeError("E_TUPLE_HEALTH_ADAPTER_STATE_DISTINCTION")
    return _tuple_health_pass_result(
        "adapter_and_lexical",
        len(context["rows"]["language_lexical"]),
        production_output,
    )


def _tuple_language_lexical_module(context: dict[str, Any]) -> dict[str, Any]:
    cfg = context["cfg"]
    rows = context["rows"]["language_lexical"]
    scratch = context["scratch_root"]
    amendment = _tuple_amendment(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    adapter_gate = _tuple_axis(cfg, "adapter_qualified_yield")["public_gate"]
    lexical_gate = _tuple_axis(cfg, "noun_adjective_exposure")["public_gate"]
    if file_digest(Path(__file__).resolve().with_name("synthetic_video_language_adapter.py")) != TUPLE_LANGUAGE_ADAPTER_SHA256:
        raise RuntimeError("E_TUPLE_LANGUAGE_ADAPTER_SOURCE")
    from synthetic_video_language_adapter import validate_asr_prediction

    import nltk
    from nltk.stem import WordNetLemmatizer
    from wordfreq import zipf_frequency

    nltk.data.path[:] = [str(_stage_tuple_nltk_resources(context["public_root"], scratch, cfg))]
    lemmatizer = WordNetLemmatizer()
    mapping = _tuple_public_ontology_mapping(preparation)
    frequency_bands = _tuple_axis(cfg, "noun_adjective_exposure")[
        "frequency_bands"
    ]
    if context.get("engineering_health") is True:
        return _tuple_language_lexical_health(
            context,
            validate_asr_prediction,
            nltk.pos_tag,
            lemmatizer.lemmatize,
            zipf_frequency,
            mapping,
            frequency_bands,
        )
    case_correct = 0
    silently_accepted_truncations = 0
    silently_accepted_invalid_timestamps = 0
    expected_by_pos: dict[str, list[tuple[str, str]]] = {
        "noun": [],
        "adjective": [],
    }
    predicted_by_pos: dict[str, list[tuple[str, str]]] = {
        "noun": [],
        "adjective": [],
    }
    lemma_band_checks: list[bool] = []
    expected_episode_counts: Counter[tuple[str, str, str]] = Counter()
    predicted_episode_counts: Counter[tuple[str, str, str]] = Counter()
    expected_adjective_noun_spans: list[tuple[str, str, str]] = []
    predicted_adjective_noun_spans: list[tuple[str, str, str]] = []
    output_rows = []
    for row in rows:
        adjudication = validate_asr_prediction(
            row["prediction"], float(row["audio_duration"])
        )
        adapter_status = str(adjudication["status"])
        adapter_reason = adjudication.get("reason")
        tuple_status = adapter_status
        tuple_reason = adapter_reason
        grounding_status = "ABSTAIN" if adapter_status != "ACCEPT" else "ACCEPT"
        grounding_reason = adapter_reason
        window: dict[str, Any] | None = None
        mentions: list[dict[str, Any]] = []
        if adapter_status == "ACCEPT":
            if row.get("translation_status") != "ACCEPT":
                adapter_status = "ABSTAIN"
                adapter_reason = (
                    "EMPTY_TRANSLATION"
                    if not str(row.get("text_en", "")).strip()
                    else "SILENT_TRUNCATION"
                )
                tuple_status, tuple_reason = adapter_status, adapter_reason
                grounding_status, grounding_reason = adapter_status, adapter_reason
            else:
                window = _tuple_segment_window(
                    row["segment"], float(row["audio_duration"]), amendment
                )
                if window["status"] != "ACCEPT":
                    tuple_status = "ABSTAIN"
                    tuple_reason = window["reason"]
                    grounding_status, grounding_reason = tuple_status, tuple_reason
        if tuple_status == "ACCEPT":
            mentions = _lexical_mentions(
                window["text_en"],
                nltk.pos_tag,
                lemmatizer.lemmatize,
                zipf_frequency,
                frequency_bands,
            )
            noun_categories = [
                _map_public_ontology(mention["lemma"], mapping)
                for mention in mentions
                if mention["part_of_speech"] == "noun"
            ]
            accepted_categories = [
                item["category"] for item in noun_categories if item["status"] == "ACCEPT"
            ]
            if not accepted_categories:
                grounding_status = "ABSTAIN"
                grounding_reason = "ONTOLOGY_UNMATCHED"
        old_reason = row.get("expected_reason")
        expected_adapter_status = str(
            row.get(
                "expected_adapter_status",
                "ABSTAIN"
                if old_reason
                in {
                    "LANGUAGE_MISMATCH",
                    "EMPTY_ASR",
                    "INVALID_TIMESTAMP",
                    "LOW_CONFIDENCE",
                    "EMPTY_TRANSLATION",
                    "SILENT_TRUNCATION",
                }
                else "ACCEPT",
            )
        )
        expected_adapter_reason = row.get(
            "expected_adapter_reason",
            old_reason if expected_adapter_status == "ABSTAIN" else None,
        )
        expected_tuple_status = str(
            row.get(
                "expected_tuple_status",
                "ABSTAIN"
                if old_reason == "INSUFFICIENT_IN_BOUNDS_FRAMES"
                or expected_adapter_status == "ABSTAIN"
                else "ACCEPT",
            )
        )
        expected_tuple_reason = row.get(
            "expected_tuple_reason",
            old_reason if expected_tuple_status == "ABSTAIN" else None,
        )
        expected_grounding_status = str(
            row.get(
                "expected_grounding_status",
                "ABSTAIN"
                if old_reason == "ONTOLOGY_UNMATCHED"
                or expected_tuple_status == "ABSTAIN"
                else "ACCEPT",
            )
        )
        expected_grounding_reason = row.get(
            "expected_grounding_reason",
            old_reason if expected_grounding_status == "ABSTAIN" else None,
        )
        case_correct += (
            adapter_status == expected_adapter_status
            and adapter_reason == expected_adapter_reason
            and tuple_status == expected_tuple_status
            and tuple_reason == expected_tuple_reason
            and grounding_status == expected_grounding_status
            and grounding_reason == expected_grounding_reason
        )
        if expected_adapter_reason == "SILENT_TRUNCATION" and adapter_status == "ACCEPT":
            silently_accepted_truncations += 1
        if expected_adapter_reason == "INVALID_TIMESTAMP" and adapter_status == "ACCEPT":
            silently_accepted_invalid_timestamps += 1
        expected_mentions = row.get("expected_lexical_mentions", [])
        episode = str(row["episode_id"])
        for ordinal, expected in enumerate(expected_mentions):
            part = str(expected["part_of_speech"])
            token = str(expected["token"]).casefold()
            expected_by_pos[part].append((str(row["case_id"]), f"{ordinal}:{token}"))
            expected_episode_counts[(episode, part, token)] += 1
        for ordinal, mention in enumerate(mentions):
            part = str(mention["part_of_speech"])
            token = str(mention["token"]).casefold()
            predicted_by_pos[part].append((str(row["case_id"]), f"{ordinal}:{token}"))
            predicted_episode_counts[(episode, part, mention["lemma"])] += 1
        lemma_band_checks.extend(
            _tuple_lexical_truth_checks(expected_mentions, mentions)
        )
        expected_constructions = row.get("expected_adjective_noun_spans")
        if not isinstance(expected_constructions, list):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_SPAN_TRUTH_MISSING")
        case_id = str(row["case_id"])
        for span in expected_constructions:
            if (
                not isinstance(span, dict)
                or set(span) != {"adjective", "noun"}
                or not all(
                    isinstance(span[key], str)
                    and re.fullmatch(r"[a-z]+(?:'[a-z]+)?", span[key])
                    for key in ("adjective", "noun")
                )
            ):
                raise RuntimeError("E_TUPLE_ATTRIBUTE_SPAN_TRUTH")
            expected_adjective_noun_spans.append(
                (case_id, span["adjective"], span["noun"])
            )
        observed_constructions = _adjacent_adjective_noun_spans(mentions)
        predicted_adjective_noun_spans.extend(
            (case_id, span["adjective"], span["noun"])
            for span in observed_constructions
        )
        output_rows.append(
            {
                "case_id": row["case_id"],
                "adapter_status": adapter_status,
                "adapter_reason": adapter_reason,
                "tuple_status": tuple_status,
                "tuple_reason": tuple_reason,
                "grounding_status": grounding_status,
                "grounding_reason": grounding_reason,
                "mention_count": len(mentions),
                "adjective_noun_span_count": len(observed_constructions),
            }
        )

    def span_metrics(part: str) -> dict[str, float]:
        truth, predicted = Counter(expected_by_pos[part]), Counter(predicted_by_pos[part])
        matched = sum((truth & predicted).values())
        precision = _safe_divide(matched, sum(predicted.values()))
        recall = _safe_divide(matched, sum(truth.values()))
        return {"precision": precision, "recall": recall}

    noun = span_metrics("noun")
    adjective = span_metrics("adjective")
    repetition_keys = set(expected_episode_counts) | set(predicted_episode_counts)
    repetition_exact = _safe_divide(
        sum(expected_episode_counts[key] == predicted_episode_counts[key] for key in repetition_keys),
        len(repetition_keys),
    )
    case_accuracy = _safe_divide(case_correct, len(rows))
    lemma_band_exact = _safe_divide(sum(lemma_band_checks), len(lemma_band_checks))
    adjective_noun_span_f1 = _span_f1(
        expected_adjective_noun_spans, predicted_adjective_noun_spans
    )
    context["_tuple_attribute_language_metrics"] = {
        "adjective_noun_span_f1": adjective_noun_span_f1,
        "dependency_status": "MEASURED",
    }
    adapter_pass = (
        case_accuracy >= float(adapter_gate["tuple_acceptance_and_abstention_case_accuracy_required"])
        and silently_accepted_truncations <= int(adapter_gate["silent_truncation_count_max"])
        and silently_accepted_invalid_timestamps <= int(adapter_gate["invalid_timestamp_count_max"])
    )
    lexical_pass = (
        noun["precision"] >= float(lexical_gate["noun_span_precision_min"])
        and noun["recall"] >= float(lexical_gate["noun_span_recall_min"])
        and adjective["precision"] >= float(lexical_gate["adjective_span_precision_min"])
        and adjective["recall"] >= float(lexical_gate["adjective_span_recall_min"])
        and lemma_band_exact >= float(lexical_gate["lemma_and_frequency_band_exact_fraction_min"])
        and repetition_exact >= float(lexical_gate["episode_repetition_count_exact_fraction_required"])
    )
    return {
        "status": "PASS" if adapter_pass and lexical_pass else "NO_GO",
        "axis_results": {
            "adapter_qualified_yield": {
                "status": "PASS" if adapter_pass else "NO_GO",
                "metrics": {
                    "case_accuracy": case_accuracy,
                    "silent_truncation_count": silently_accepted_truncations,
                    "invalid_timestamp_count": silently_accepted_invalid_timestamps,
                },
            },
            "noun_adjective_exposure": {
                "status": "PASS" if lexical_pass else "NO_GO",
                "metrics": {
                    "noun_span_precision": noun["precision"],
                    "noun_span_recall": noun["recall"],
                    "adjective_span_precision": adjective["precision"],
                    "adjective_span_recall": adjective["recall"],
                    "adjective_noun_span_f1": adjective_noun_span_f1,
                    "lemma_and_frequency_band_exact_fraction": lemma_band_exact,
                    "episode_repetition_count_exact_fraction": repetition_exact,
                },
            },
        },
        "metrics": {"case_accuracy": case_accuracy},
        "rows": output_rows,
        "row_count": len(rows),
        "failure_count": 0,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": silently_accepted_truncations,
        "external_call_count": 0,
    }


def _tuple_language_artifact_record(public: Path) -> dict[str, Any]:
    """Verify the selected adapter bytes and the immutable local HF revision."""

    required_environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    if any(os.environ.get(key) != value for key, value in required_environment.items()):
        raise RuntimeError("E_TUPLE_REFERENT_ADAPTER_OFFLINE_ENVIRONMENT")
    if os.environ.get("WANDB_DISABLED", "").casefold() != "true":
        raise RuntimeError("E_TUPLE_REFERENT_ADAPTER_TELEMETRY")
    adapter_path = Path(__file__).resolve().with_name(
        "synthetic_video_language_adapter.py"
    )
    if (
        not adapter_path.is_file()
        or file_digest(adapter_path) != TUPLE_LANGUAGE_ADAPTER_SHA256
    ):
        raise RuntimeError("E_TUPLE_LANGUAGE_ADAPTER_SOURCE")
    whisper_path = public / "models/whisper/small.pt"
    if (
        not whisper_path.is_file()
        or file_digest(whisper_path) != TUPLE_WHISPER_SMALL_SHA256
    ):
        raise RuntimeError("E_TUPLE_REFERENT_WHISPER_WEIGHT")
    translation_root = public / "models/opus-mt-de-en"
    required_names = {
        "config.json",
        "source.spm",
        "target.spm",
        "tokenizer_config.json",
    }
    files = sorted(
        path
        for path in translation_root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(translation_root).parts
    )
    if not required_names <= {path.name for path in files} or not any(
        path.name in {"model.safetensors", "pytorch_model.bin"} for path in files
    ):
        raise RuntimeError("E_TUPLE_REFERENT_TRANSLATION_ARTIFACT")
    metadata_files = sorted(
        (translation_root / ".cache/huggingface/download").rglob("*.metadata")
    )
    revisions = set()
    for path in metadata_files:
        lines = path.read_text(errors="strict").splitlines()
        if not lines or not re.fullmatch(r"[0-9a-f]{40}", lines[0]):
            raise RuntimeError("E_TUPLE_REFERENT_TRANSLATION_REVISION")
        revisions.add(lines[0])
    if revisions != {TUPLE_OPUS_MT_DE_EN_REVISION}:
        raise RuntimeError("E_TUPLE_REFERENT_TRANSLATION_REVISION")
    tree = [
        {
            "relative_path": str(path.relative_to(translation_root)),
            "sha256": file_digest(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    return {
        "adapter_sha256": TUPLE_LANGUAGE_ADAPTER_SHA256,
        "whisper_small_sha256": TUPLE_WHISPER_SMALL_SHA256,
        "opus_mt_de_en_revision": TUPLE_OPUS_MT_DE_EN_REVISION,
        "opus_mt_de_en_file_count": len(tree),
        "opus_mt_de_en_tree_commitment_sha256": digest(tree),
    }


def _tuple_build_referent_adapter_observation(
    adjudication: dict[str, Any],
    media_duration: float,
    amendment: dict[str, Any],
    ontology_mapping: dict[str, list[str]],
    *,
    tagger,
    lemmatize,
    zipf_frequency,
    frequency_bands: dict[str, list[float]],
) -> dict[str, Any]:
    """Create one text-free grounding event from the actual shared adapter output."""

    if adjudication.get("status") != "ACCEPT":
        return {
            "status": "ABSTAIN",
            "reason": str(adjudication.get("reason") or "ADAPTER_ABSTAIN"),
            "abstention_reason": str(
                adjudication.get("reason") or "ADAPTER_ABSTAIN"
            ),
            "mentions": [],
        }
    words = adjudication.get("words")
    if not isinstance(words, list) or not words:
        return {
            "status": "ABSTAIN",
            "reason": "EMPTY_ASR",
            "abstention_reason": "EMPTY_ASR",
            "mentions": [],
        }
    segment = {
        "status": "ACCEPT",
        "start": words[0].get("start"),
        "end": words[-1].get("end"),
        "en": adjudication.get("text_en", ""),
    }
    window = _tuple_segment_window(segment, media_duration, amendment)
    if window.get("status") != "ACCEPT":
        return {
            "status": "ABSTAIN",
            "reason": window.get("reason"),
            "abstention_reason": window.get("reason"),
            "mentions": [],
        }
    mentions = _lexical_mentions(
        window["text_en"],
        tagger,
        lemmatize,
        zipf_frequency,
        frequency_bands,
    )
    mappings = [
        _map_public_ontology(mention["lemma"], ontology_mapping)
        for mention in mentions
        if mention["part_of_speech"] == "noun"
    ]
    categories = sorted(
        {value["category"] for value in mappings if value["status"] == "ACCEPT"}
    )
    if not categories:
        return {
            "status": "ABSTAIN",
            "reason": "ONTOLOGY_UNMATCHED",
            "abstention_reason": "ONTOLOGY_UNMATCHED",
            "mentions": mentions,
        }
    if len(categories) != 1:
        return {
            "status": "ABSTAIN",
            "reason": "ONTOLOGY_AMBIGUOUS",
            "abstention_reason": "ONTOLOGY_AMBIGUOUS",
            "mentions": mentions,
        }
    return {
        "status": "ACCEPT",
        "reason": None,
        "abstention_reason": None,
        "category": categories[0],
        "segment_start": window["segment_start"],
        "segment_end": window["segment_end"],
        "mention_anchor": window["mention_anchor"],
        "samples": window["samples"],
        "noun_mention_count": sum(
            mention["part_of_speech"] == "noun" for mention in mentions
        ),
        "mentions": mentions,
    }


def _tuple_referent_adapter_observations(
    context: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Run the actual pinned adapter; authored truth never controls eligibility."""

    module_cache = context.setdefault("module_cache", {})
    cached = module_cache.get("referent_grounding_events")
    if isinstance(cached, dict):
        return cached
    from synthetic_video_language_adapter import (
        translate_accepted,
        validate_asr_prediction,
        whisper_prediction,
    )

    import nltk
    from nltk.stem import WordNetLemmatizer
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    import whisper
    from wordfreq import zipf_frequency

    public = context["public_root"]
    fixture_root = context["fixture_root"]
    cfg = context["cfg"]
    amendment = _tuple_amendment(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    artifact = _tuple_language_artifact_record(public)
    nltk_root = Path(context["scratch_root"]) / "referent-adapter-resources"
    nltk.data.path[:] = [str(_stage_tuple_nltk_resources(public, nltk_root, cfg))]
    lemmatizer = WordNetLemmatizer()
    ontology_mapping = _tuple_public_ontology_mapping(preparation)
    frequency_bands = _tuple_axis(cfg, "noun_adjective_exposure")[
        "frequency_bands"
    ]
    device = str(context["device"])
    asr = whisper.load_model(
        str(public / "models/whisper/small.pt"), device=device
    )
    translation_root = public / "models/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(
        translation_root, local_files_only=True
    )
    translator = MarianMTModel.from_pretrained(
        translation_root, local_files_only=True
    ).to(device).eval()
    decoding = {
        "temperature": 0.0,
        "beam_size": 5,
        "fp16": device.startswith("cuda"),
    }
    observations: dict[int, dict[str, Any]] = {}
    try:
        for row in context["rows"]["referent_attribute"]:
            ordinal = int(row["fixture_ordinal"])
            if ordinal in observations:
                raise RuntimeError("E_TUPLE_REFERENT_DUPLICATE_ORDINAL")
            media = _tuple_fixture_file(
                fixture_root,
                row["media_relative_path"],
                row["media_sha256"],
                row.get("media_bytes"),
            )
            samples = whisper.load_audio(str(media))
            media_duration = float(len(samples)) / float(whisper.audio.SAMPLE_RATE)
            if not math.isfinite(media_duration) or media_duration <= 0.0:
                raise RuntimeError("E_TUPLE_REFERENT_AUDIO_DURATION")
            prediction = whisper_prediction(asr, media, decoding)
            adjudication = validate_asr_prediction(
                prediction, media_duration, confidence_min=0.35
            )
            with torch.inference_mode():
                adjudication = translate_accepted(
                    adjudication, tokenizer, translator, max_new_tokens=128
                )
            observation = _tuple_build_referent_adapter_observation(
                adjudication,
                media_duration,
                amendment,
                ontology_mapping,
                tagger=nltk.pos_tag,
                lemmatize=lemmatizer.lemmatize,
                zipf_frequency=zipf_frequency,
                frequency_bands=frequency_bands,
            )
            observations[ordinal] = {"fixture_ordinal": ordinal, **observation}
    finally:
        del asr, translator
        _release_cuda()
    result_root = Path(context["scratch_root"]) / "referent-adapter-observations"
    _require_external_or_ignored_output(result_root)
    result_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    serial = {
        "schema_version": 1,
        "partition": context["partition"],
        "artifact_provenance": artifact,
        "observations": [observations[key] for key in sorted(observations)],
        "raw_or_translated_text_retained": False,
        "external_call_count": 0,
    }
    serial["observation_commitment_sha256"] = digest(serial)
    write_private(result_root / "observations.json", serial)
    module_cache["referent_adapter_events"] = {
        ordinal: {
            "status": row["status"],
            "abstention_reason": row["abstention_reason"],
            "mentions": row["mentions"] if row["status"] == "ACCEPT" else [],
        }
        for ordinal, row in observations.items()
    }
    module_cache["referent_grounding_events"] = observations
    return observations


def _tuple_referent_truth_record(
    row: dict[str, Any], fixture_root: Path
) -> dict[str, Any]:
    """Validate the corrected phase truth and every registered sampled mask."""

    phases = ("before", "during", "after")
    truth = row.get("truth")
    if not isinstance(truth, dict):
        raise RuntimeError("E_TUPLE_REFERENT_TRUTH_SCHEMA")
    visibility = truth.get("visibility_by_phase")
    dominance = truth.get("dominance_by_phase")
    candidate_counts = truth.get("candidate_count_by_phase")
    if (
        not isinstance(visibility, dict)
        or set(visibility) != set(phases)
        or any(type(visibility[phase]) is not bool for phase in phases)
        or not isinstance(dominance, dict)
        or set(dominance) != set(phases)
        or not isinstance(candidate_counts, dict)
        or set(candidate_counts) != set(phases)
        or any(candidate_counts[phase] not in {"0", "1", "2plus"} for phase in phases)
    ):
        raise RuntimeError("E_TUPLE_REFERENT_TRUTH_SCHEMA")
    for phase in phases:
        if visibility[phase]:
            if type(dominance[phase]) is not bool or candidate_counts[phase] == "0":
                raise RuntimeError("E_TUPLE_REFERENT_TRUTH_SCHEMA")
        elif dominance[phase] is not None:
            raise RuntimeError("E_TUPLE_REFERENT_TRUTH_SCHEMA")
    sampled = truth.get("sampled_mask_truth")
    if not isinstance(sampled, list) or not 8 <= len(sampled) <= 9:
        raise RuntimeError("E_TUPLE_REFERENT_SAMPLED_TRUTH_SCHEMA")
    phase_counts: Counter[str] = Counter()
    prior_time = -math.inf
    for sample in sampled:
        if not isinstance(sample, dict) or sample.get("phase") not in phases:
            raise RuntimeError("E_TUPLE_REFERENT_SAMPLED_TRUTH_SCHEMA")
        phase = str(sample["phase"])
        try:
            sample_time = float(sample["sample_time"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("E_TUPLE_REFERENT_SAMPLED_TRUTH_SCHEMA") from error
        if not math.isfinite(sample_time) or sample_time <= prior_time:
            raise RuntimeError("E_TUPLE_REFERENT_SAMPLED_TRUTH_SCHEMA")
        prior_time = sample_time
        phase_counts[phase] += 1
        target_path = _tuple_fixture_file(
            fixture_root,
            sample.get("target_mask_relative_path"),
            sample.get("target_mask_sha256"),
            sample.get("target_mask_bytes"),
        )
        from PIL import Image

        with Image.open(target_path) as source:
            target = source.convert("L")
            if target.width < 1 or target.height < 1:
                raise RuntimeError("E_TUPLE_REFERENT_TRUTH_MASK_GEOMETRY")
            target_present = target.getbbox() is not None
        distractor_relative = sample.get("distractor_mask_relative_path")
        distractor_sha = sample.get("distractor_mask_sha256")
        if distractor_relative is None or distractor_sha is None:
            if distractor_relative is not None or distractor_sha is not None:
                raise RuntimeError("E_TUPLE_REFERENT_TRUTH_MASK_SCHEMA")
            distractor_present = False
        else:
            distractor_path = _tuple_fixture_file(
                fixture_root,
                distractor_relative,
                distractor_sha,
                sample.get("distractor_mask_bytes"),
            )
            with Image.open(distractor_path) as source:
                distractor = source.convert("L")
                if distractor.size != target.size:
                    raise RuntimeError("E_TUPLE_REFERENT_TRUTH_MASK_GEOMETRY")
                distractor_present = distractor.getbbox() is not None
        expected_count = (
            "2plus"
            if target_present and distractor_present
            else "1"
            if target_present or distractor_present
            else "0"
        )
        if (
            target_present != visibility[phase]
            or expected_count != candidate_counts[phase]
        ):
            raise RuntimeError("E_TUPLE_REFERENT_TRUTH_MASK_ROUNDTRIP")
    if sum(phase_counts.values()) < 8 or any(
        not 2 <= phase_counts[phase] <= 3 for phase in phases
    ):
        raise RuntimeError("E_TUPLE_REFERENT_SAMPLED_TRUTH_SCHEMA")
    if "2plus" in candidate_counts.values():
        target_image = str(row.get("source_image_id", ""))
        target_annotation = str(row.get("source_annotation_id", ""))
        target_sha = str(row.get("source_image_sha256", ""))
        distractor_image = str(row.get("distractor_source_image_id", ""))
        distractor_annotation = str(
            row.get("distractor_source_annotation_id", "")
        )
        distractor_sha = str(row.get("distractor_source_image_sha256", ""))
        if (
            row.get("distractor_source_category") != row.get("category")
            or row.get("distractor_source_distinct_from_target") is not True
            or not all(
                re.fullmatch(r"[0-9a-f]{64}", value)
                for value in (target_sha, distractor_sha)
            )
            or not all(
                value
                for value in (
                    target_image,
                    target_annotation,
                    distractor_image,
                    distractor_annotation,
                )
            )
            or target_image == distractor_image
            or target_annotation == distractor_annotation
            or target_sha == distractor_sha
        ):
            raise RuntimeError("E_TUPLE_REFERENT_DISTRACTOR_PROVENANCE")
    return {
        "visibility_by_phase": dict(visibility),
        "dominance_by_phase": dict(dominance),
        "candidate_count_by_phase": dict(candidate_counts),
    }


def _tuple_decode_frames_at_times(media: Path, sample_times: list[float]) -> list[Any]:
    import decord

    reader = decord.VideoReader(str(media), ctx=decord.cpu(0), num_threads=1)
    if len(reader) < 1:
        raise RuntimeError("E_TUPLE_REFERENT_DECODE_EMPTY")
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError("E_TUPLE_REFERENT_DECODE_FPS")
    indices = [
        min(len(reader) - 1, max(0, int(round(float(value) * fps))))
        for value in sample_times
    ]
    if len(indices) != len(set(indices)):
        raise RuntimeError("E_TUPLE_REFERENT_DECODE_DUPLICATE_FRAME")
    frames = reader.get_batch(indices).asnumpy()
    if (
        frames.ndim != 4
        or frames.shape[0] != len(sample_times)
        or frames.shape[-1] != 3
    ):
        raise RuntimeError("E_TUPLE_REFERENT_DECODE_TRUNCATION")
    return [frames[index] for index in range(len(frames))]


def _tuple_grounding_phrase_positions(tokenizer, caption: str, category: str) -> list[int]:
    if caption != f"{category}." or category != category.casefold().strip():
        raise RuntimeError("E_TUPLE_GROUNDING_CAPTION")
    caption_tokens = tokenizer(caption, add_special_tokens=True)["input_ids"]
    category_tokens = tokenizer(category, add_special_tokens=False)["input_ids"]
    if (
        not isinstance(caption_tokens, list)
        or not isinstance(category_tokens, list)
        or not category_tokens
    ):
        raise RuntimeError("E_TUPLE_GROUNDING_PHRASE_TOKEN")
    starts = [
        index
        for index in range(len(caption_tokens) - len(category_tokens) + 1)
        if caption_tokens[index : index + len(category_tokens)] == category_tokens
    ]
    if len(starts) != 1:
        raise RuntimeError("E_TUPLE_GROUNDING_PHRASE_TOKEN")
    return list(range(starts[0], starts[0] + len(category_tokens)))


def _tuple_cxcywh_to_normalized_xyxy(raw_box: Any) -> list[float] | None:
    try:
        center_x, center_y, width, height = (float(value) for value in raw_box)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(
        math.isfinite(value) for value in (center_x, center_y, width, height)
    ):
        return None
    left = min(1.0, max(0.0, center_x - width / 2.0))
    top = min(1.0, max(0.0, center_y - height / 2.0))
    right = min(1.0, max(0.0, center_x + width / 2.0))
    bottom = min(1.0, max(0.0, center_y + height / 2.0))
    box = [left, top, right, bottom]
    return box if _valid_normalized_box(box) else None


def _tuple_box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return _safe_divide(intersection, first_area + second_area - intersection)


def _tuple_same_category_nms(
    candidates: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise RuntimeError("E_TUPLE_GROUNDING_NMS_THRESHOLD")
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row["box_score"]),
            -float(row["text_score"]),
            tuple(float(value) for value in row["box"]),
        ),
    )
    retained = []
    for candidate in ordered:
        if not _valid_normalized_box(candidate.get("box")):
            raise RuntimeError("E_TUPLE_GROUNDING_NMS_BOX")
        if all(
            _tuple_box_iou(candidate["box"], prior["box"]) <= iou_threshold
            for prior in retained
        ):
            retained.append(candidate)
    return retained


def _load_tuple_grounding_stack(public: Path, device: str):
    import torch

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
    if file_digest(checkpoint_path) != TUPLE_GROUNDING_DINO_SHA256:
        raise RuntimeError("E_TUPLE_GROUNDING_WEIGHT")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    incompatible = grounding.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("E_TUPLE_GROUNDING_STATE")
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
    sam_root = model_root / "code/sam2"
    sys.path.insert(0, str(sam_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_path = model_root / "weights/sam2.1_hiera_base_plus.pt"
    if file_digest(sam_path) != TUPLE_SAM21_BASE_PLUS_SHA256:
        raise RuntimeError("E_TUPLE_SAM_WEIGHT")
    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        str(sam_path),
        device=device,
        apply_postprocessing=False,
    )
    return grounding, transform, SAM2ImagePredictor(sam), sam


def _tuple_grounding_frame_candidates(
    grounding,
    transform,
    predictor,
    frame: Any,
    category: str,
    *,
    device: str,
    minimum_box_score: float,
    minimum_text_score: float,
    nms_iou: float,
    strict_engineering: bool = False,
) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image
    import torch

    image = Image.fromarray(frame).convert("RGB")
    caption = f"{category}."
    phrase_positions = _tuple_grounding_phrase_positions(
        grounding.tokenizer, caption, category
    )
    tensor, _ = transform(image, None)
    with torch.inference_mode():
        output = grounding(tensor[None].to(device), captions=[caption])
    logits = output.get("pred_logits")
    raw_boxes = output.get("pred_boxes")
    if (
        logits is None
        or raw_boxes is None
        or logits.ndim != 3
        or raw_boxes.ndim != 3
        or logits.shape[0] != 1
        or raw_boxes.shape[0] != 1
        or logits.shape[1] != raw_boxes.shape[1]
        or raw_boxes.shape[2] != 4
        or max(phrase_positions) >= logits.shape[2]
    ):
        raise RuntimeError("E_TUPLE_GROUNDING_OUTPUT_SHAPE")
    phrase_scores = logits[0, :, phrase_positions].sigmoid()
    if not torch.isfinite(phrase_scores).all() or not torch.isfinite(raw_boxes).all():
        raise RuntimeError("E_TUPLE_GROUNDING_NONFINITE")
    preliminary = []
    for ordinal in range(raw_boxes.shape[1]):
        values = phrase_scores[ordinal]
        box_score = float(values.max())
        text_score = float(values.min())
        if box_score < minimum_box_score or text_score < minimum_text_score:
            continue
        box = _tuple_cxcywh_to_normalized_xyxy(
            raw_boxes[0, ordinal].detach().cpu().tolist()
        )
        if box is None:
            raise RuntimeError("E_TUPLE_GROUNDING_BOX_GEOMETRY")
        preliminary.append(
            {
                "category": category,
                "box": box,
                "box_score": box_score,
                "text_score": text_score,
            }
        )
    retained = _tuple_same_category_nms(preliminary, nms_iou)
    predictor.set_image(np.asarray(image, dtype=np.uint8))
    height, width = frame.shape[:2]
    output_candidates = []
    for candidate in retained:
        left, top, right, bottom = candidate["box"]
        pixel_box = np.asarray(
            [left * width, top * height, right * width, bottom * height],
            dtype=np.float32,
        )
        try:
            masks, scores, mask_logits = predictor.predict(
                box=pixel_box, multimask_output=False
            )
            mask_values = np.asarray(masks)
            while mask_values.ndim > 2 and mask_values.shape[0] == 1:
                mask_values = mask_values[0]
            valid = (
                mask_values.shape == (height, width)
                and np.isfinite(mask_values).all()
                and np.isfinite(np.asarray(scores)).all()
                and np.isfinite(np.asarray(mask_logits)).all()
            )
            mask = mask_values.astype(bool) if valid else None
            valid = bool(valid and mask is not None and mask.any())
        except Exception as error:
            if strict_engineering:
                raise RuntimeError("E_TUPLE_GROUNDING_SAM_INFERENCE") from error
            mask = None
            valid = False
        if not valid:
            output_candidates.append(
                {**candidate, "valid": False, "reason": "INVALID_OR_EMPTY_MASK"}
            )
            continue
        locations = np.argwhere(mask)
        center_y, center_x = locations.mean(axis=0)
        normalized_x = center_x / max(1, width - 1)
        normalized_y = center_y / max(1, height - 1)
        output_candidates.append(
            {
                **candidate,
                "valid": True,
                "reason": None,
                "mask": mask,
                "mask_fraction": float(mask.mean()),
                "center_distance": math.hypot(
                    normalized_x - 0.5, normalized_y - 0.5
                ),
            }
        )
    return output_candidates


def _tuple_grounding_sampled_tracks(
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Infer nine sampled frames once and retain masks only in the in-memory cache."""

    module_cache = context.setdefault("module_cache", {})
    cached = module_cache.get("grounding_sampled_tracks")
    if isinstance(cached, list):
        return cached
    cfg = context["cfg"]
    execution = _tuple_qualification_execution(cfg)
    minimum_valid = int(execution["phase_aggregation"]["minimum_valid_samples"])
    if context.get("engineering_health") is True:
        execution_thresholds = _tuple_health_inference_thresholds(cfg)
        minimum_box = execution_thresholds["Grounding_DINO_box_score"]
        minimum_text = execution_thresholds["Grounding_DINO_text_score"]
    else:
        grids = _tuple_frozen_threshold_grids(cfg)
        minimum_box = min(grids["Grounding_DINO_box_score"])
        minimum_text = min(grids["Grounding_DINO_text_score"])
    nms_iou = float(execution["grounding_geometry"]["same_category_NMS_IoU"])
    observations = _tuple_referent_adapter_observations(context)
    rows = context["rows"]["referent_attribute"]
    fixture_root = context["fixture_root"]
    accepted_count = sum(
        observations.get(int(row["fixture_ordinal"]), {}).get("status") == "ACCEPT"
        and observations[int(row["fixture_ordinal"])].get("category")
        == row.get("category")
        for row in rows
    )
    stack = None
    if accepted_count:
        stack = _load_tuple_grounding_stack(
            context["public_root"], str(context["device"])
        )
    tracks = []
    try:
        for row in rows:
            ordinal = int(row["fixture_ordinal"])
            truth = _tuple_referent_truth_record(row, fixture_root)
            observation = observations.get(ordinal)
            if not isinstance(observation, dict):
                raise RuntimeError("E_TUPLE_REFERENT_ADAPTER_OBSERVATION_MISSING")
            track = {
                "fixture_ordinal": ordinal,
                "category": row["category"],
                "scenario": row["scenario"],
                "attribute_pair_id": row.get("attribute_pair_id"),
                "attribute": truth.get("attribute", row.get("truth", {}).get("attribute")),
                "truth": truth,
                "adapter_observation": observation,
                "samples": [],
            }
            if observation.get("status") != "ACCEPT":
                tracks.append(track)
                continue
            if observation.get("category") != row.get("category"):
                track["category_mismatch"] = True
                tracks.append(track)
                continue
            sample_rows = [
                (phase, float(value))
                for phase in ("before", "during", "after")
                for value in observation["samples"].get(phase, [])
            ]
            if len(sample_rows) < minimum_valid:
                tracks.append(track)
                continue
            media = _tuple_fixture_file(
                fixture_root,
                row["media_relative_path"],
                row["media_sha256"],
                row.get("media_bytes"),
            )
            try:
                frames = _tuple_decode_frames_at_times(
                    media, [value for _phase, value in sample_rows]
                )
            except RuntimeError as error:
                if context.get("engineering_health") is True:
                    raise RuntimeError("E_TUPLE_REFERENT_DECODE") from error
                track["decode_error_code"] = str(error).split()[0]
                tracks.append(track)
                continue
            assert stack is not None
            grounding, transform, predictor, _sam = stack
            for (phase, sample_time), frame in zip(
                sample_rows, frames, strict=True
            ):
                try:
                    candidates = _tuple_grounding_frame_candidates(
                        grounding,
                        transform,
                        predictor,
                        frame,
                        str(observation["category"]),
                        device=str(context["device"]),
                        minimum_box_score=minimum_box,
                        minimum_text_score=minimum_text,
                        nms_iou=nms_iou,
                        strict_engineering=context.get("engineering_health")
                        is True,
                    )
                    track["samples"].append(
                        {
                            "sample_time": sample_time,
                            "phase": phase,
                            "image": frame,
                            "inference_succeeded": True,
                            "candidates": candidates,
                        }
                    )
                except RuntimeError as error:
                    if context.get("engineering_health") is True:
                        raise RuntimeError(
                            "E_TUPLE_REFERENT_MODEL_INFERENCE"
                        ) from error
                    track["samples"].append(
                        {
                            "sample_time": sample_time,
                            "phase": phase,
                            "image": frame,
                            "inference_succeeded": False,
                            "error_code": str(error).split()[0],
                            "candidates": [],
                        }
                    )
            tracks.append(track)
    finally:
        if stack is not None:
            grounding, _transform, predictor, sam = stack
            del grounding, predictor, sam
            _release_cuda()
    module_cache["grounding_sampled_tracks"] = tracks
    return tracks


def _tuple_phase_majority(values: list[Any]) -> Any | None:
    if not values:
        return None
    counts = Counter(values)
    highest = max(counts.values())
    winners = [value for value, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 else None


def _tuple_referent_track_prediction(
    track: dict[str, Any],
    box_threshold: float,
    text_threshold: float,
    definitions: dict[str, Any],
    minimum_valid_samples: int,
) -> dict[str, Any]:
    observation = track["adapter_observation"]
    if observation.get("status") != "ACCEPT":
        return {
            "status": "NO_ADAPTER_EVENT",
            "reason": observation.get("abstention_reason"),
            "invalid_retained_record_count": 0,
            "inference_failure_count": 0,
            "scientific_abstention_count": 0,
            "category_error_count": 0,
        }
    if track.get("category_mismatch") is True or observation.get(
        "category"
    ) != track.get("category"):
        return {
            "status": "ABSTAIN",
            "reason": "PUBLIC_CATEGORY_MISMATCH",
            "invalid_retained_record_count": 0,
            "inference_failure_count": 0,
            "scientific_abstention_count": 1,
            "category_error_count": 1,
        }
    inference_failures = sum(
        sample.get("inference_succeeded") is not True for sample in track["samples"]
    )
    if len(track["samples"]) < minimum_valid_samples:
        decode_failed = "decode_error_code" in track
        return {
            "status": "ABSTAIN",
            "reason": track.get("decode_error_code", "INSUFFICIENT_VALID_SAMPLES"),
            "invalid_retained_record_count": 0,
            "inference_failure_count": 1 if decode_failed else inference_failures,
            "scientific_abstention_count": 0 if decode_failed else 1,
            "category_error_count": 0,
        }
    measured_samples = []
    positive_track = []
    for sample in track["samples"]:
        if sample.get("inference_succeeded") is not True:
            continue
        retained = [
            candidate
            for candidate in sample["candidates"]
            if float(candidate["box_score"]) >= box_threshold
            and float(candidate["text_score"]) >= text_threshold
        ]
        if any(candidate.get("valid") is not True for candidate in retained):
            return {
                "status": "ABSTAIN",
                "reason": "INVALID_OR_EMPTY_RETAINED_MASK",
                "invalid_retained_record_count": 1,
                "inference_failure_count": inference_failures,
                "scientific_abstention_count": 0,
                "category_error_count": 0,
            }
        proxy_candidates = [
            {
                "category": candidate["category"],
                "box": candidate["box"],
                "mask_fraction": candidate["mask_fraction"],
                "center_distance": candidate["center_distance"],
            }
            for candidate in retained
        ]
        proxy = _referent_frame_proxy(
            proxy_candidates,
            str(observation["category"]),
            definitions,
            inference_succeeded=True,
            # Public specificity is qualified by the same frozen metric below;
            # a downstream valid negative remains unavailable if that gate fails.
            negative_specificity_passed=True,
        )
        if not str(proxy.get("status", "")).startswith("MEASURED_"):
            continue
        measured_samples.append(
            {
                "sample_time": float(sample["sample_time"]),
                "phase": sample["phase"],
                "proxy": proxy,
            }
        )
        if proxy["visible"]:
            target = max(
                retained, key=lambda candidate: float(candidate["mask_fraction"])
            )
            positive_track.append(
                {"time": float(sample["sample_time"]), "box": target["box"]}
            )
    if len(measured_samples) < minimum_valid_samples:
        true_failure_count = inference_failures
        return {
            "status": "ABSTAIN",
            "reason": "INSUFFICIENT_VALID_SAMPLES",
            "invalid_retained_record_count": 0,
            "inference_failure_count": true_failure_count,
            "scientific_abstention_count": 0 if true_failure_count else 1,
            "category_error_count": 0,
        }
    geometry_valid = not positive_track or _validate_monotonic_track(positive_track)
    if not geometry_valid:
        return {
            "status": "ABSTAIN",
            "reason": "INVALID_MONOTONIC_TRACK",
            "invalid_retained_record_count": 1,
            "inference_failure_count": inference_failures,
            "scientific_abstention_count": 0,
            "category_error_count": 0,
        }
    phase_predictions = {}
    for phase in ("before", "during", "after"):
        phase_rows = [row["proxy"] for row in measured_samples if row["phase"] == phase]
        if len(phase_rows) < 2:
            return {
                "status": "ABSTAIN",
                "reason": "INSUFFICIENT_PHASE_SAMPLES",
                "invalid_retained_record_count": 0,
                "inference_failure_count": inference_failures,
                "scientific_abstention_count": 0 if inference_failures else 1,
                "category_error_count": 0,
            }
        visible = _tuple_phase_majority([bool(row["visible"]) for row in phase_rows])
        count_bin = _tuple_phase_majority(
            [str(row["candidate_count_bin"]) for row in phase_rows]
        )
        if visible is None or count_bin is None:
            return {
                "status": "ABSTAIN",
                "reason": "AMBIGUOUS_PHASE_VOTE",
                "invalid_retained_record_count": 0,
                "inference_failure_count": inference_failures,
                "scientific_abstention_count": 0 if inference_failures else 1,
                "category_error_count": 0,
            }
        dominant = None
        if visible:
            dominant = _tuple_phase_majority(
                [bool(row["dominant"]) for row in phase_rows if row["visible"]]
            )
            if dominant is None:
                return {
                    "status": "ABSTAIN",
                    "reason": "AMBIGUOUS_DOMINANCE_VOTE",
                    "invalid_retained_record_count": 0,
                    "inference_failure_count": inference_failures,
                    "scientific_abstention_count": 0 if inference_failures else 1,
                    "category_error_count": 0,
                }
        phase_predictions[phase] = {
            "visible": visible,
            "dominant": dominant,
            "candidate_count_bin": count_bin,
        }
    if any(track["truth"]["visibility_by_phase"].values()) and not positive_track:
        return {
            "status": "ABSTAIN",
            "reason": "LOW_CONFIDENCE_OR_NO_DETECTION",
            "invalid_retained_record_count": 0,
            "inference_failure_count": inference_failures,
            "scientific_abstention_count": 0 if inference_failures else 1,
            "category_error_count": 0,
        }
    return {
        "status": "MEASURED",
        "reason": None,
        "category": observation["category"],
        "phase_predictions": phase_predictions,
        "valid_geometry_and_monotonic_track": True,
        "invalid_retained_record_count": 0,
        "inference_failure_count": inference_failures,
        "scientific_abstention_count": 0,
        "category_error_count": 0,
    }


def _tuple_referent_metrics(
    tracks: list[dict[str, Any]],
    box_threshold: float,
    text_threshold: float,
    definitions: dict[str, Any],
    minimum_valid_samples: int,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, int]]:
    predictions = [
        _tuple_referent_track_prediction(
            track,
            box_threshold,
            text_threshold,
            definitions,
            minimum_valid_samples,
        )
        for track in tracks
    ]
    eligible = [
        index
        for index, track in enumerate(tracks)
        if track["adapter_observation"].get("status") == "ACCEPT"
    ]
    visibility_truth: list[str] = []
    visibility_predicted: list[str | None] = []
    ambiguity_truth: list[str] = []
    ambiguity_predicted: list[str | None] = []
    dominance_truth: list[str] = []
    dominance_predicted: list[str | None] = []
    event_truth_present: list[bool] = []
    event_predicted_present: list[bool | None] = []
    compact_rows = []
    for index in eligible:
        truth = tracks[index]["truth"]
        prediction = predictions[index]
        measured = prediction["status"] == "MEASURED"
        phase_predictions = prediction.get("phase_predictions", {})
        for phase in ("before", "during", "after"):
            visible_truth = bool(truth["visibility_by_phase"][phase])
            visible_prediction = (
                bool(phase_predictions[phase]["visible"]) if measured else None
            )
            visibility_truth.append("visible" if visible_truth else "not_visible")
            visibility_predicted.append(
                None
                if visible_prediction is None
                else "visible"
                if visible_prediction
                else "not_visible"
            )
            ambiguity_truth.append(truth["candidate_count_by_phase"][phase])
            ambiguity_predicted.append(
                phase_predictions[phase]["candidate_count_bin"]
                if measured
                else None
            )
            if visible_truth:
                dominance_truth.append(
                    "dominant"
                    if truth["dominance_by_phase"][phase]
                    else "not_dominant"
                )
                dominance_predicted.append(
                    None
                    if not measured or not phase_predictions[phase]["visible"]
                    else "dominant"
                    if phase_predictions[phase]["dominant"]
                    else "not_dominant"
                )
        expected_present = any(truth["visibility_by_phase"].values())
        event_truth_present.append(expected_present)
        event_predicted_present.append(
            any(
                phase_predictions[phase]["visible"]
                for phase in ("before", "during", "after")
            )
            if measured
            else None
        )
        compact_rows.append(
            {
                "fixture_ordinal": tracks[index]["fixture_ordinal"],
                "status": prediction["status"],
                "reason": prediction.get("reason"),
                "phase_predictions": phase_predictions,
            }
        )
    visibility_macro_f1 = _multiclass_macro_f1(
        visibility_truth,
        visibility_predicted,
        ["not_visible", "visible"],
    ) if visibility_truth else 0.0
    ambiguity_macro_f1 = _multiclass_macro_f1(
        ambiguity_truth,
        ambiguity_predicted,
        ["0", "1", "2plus"],
    ) if ambiguity_truth else 0.0
    dominance_kappa = _weighted_kappa(
        dominance_truth,
        dominance_predicted,
        ["not_dominant", "dominant"],
    ) if dominance_truth else 0.0
    no_referent = (
        _binary_classification_metrics(event_truth_present, event_predicted_present)[
            "specificity"
        ]
        if event_truth_present
        else 0.0
    )
    measured_count = sum(predictions[index]["status"] == "MEASURED" for index in eligible)
    geometry_count = sum(
        not int(predictions[index].get("invalid_retained_record_count", 0))
        and not int(predictions[index].get("inference_failure_count", 0))
        for index in eligible
    )
    metrics = {
        "event_coverage": _safe_divide(measured_count, len(eligible)),
        "visibility_timing_macro_f1": visibility_macro_f1,
        "no_referent_specificity": no_referent,
        "dominance_weighted_kappa": dominance_kappa,
        "ambiguity_macro_f1": ambiguity_macro_f1,
        "valid_geometry_and_monotonic_track_fraction": _safe_divide(
            geometry_count, len(eligible)
        ),
    }
    integrity = {
        "invalid_retained_record_count": sum(
            int(predictions[index].get("invalid_retained_record_count", 0))
            for index in eligible
        ),
        "inference_failure_count": sum(
            int(predictions[index].get("inference_failure_count", 0))
            for index in eligible
        ),
        "scientific_abstention_count": sum(
            int(predictions[index].get("scientific_abstention_count", 0))
            for index in eligible
        ),
        "category_error_count": sum(
            int(predictions[index].get("category_error_count", 0))
            for index in eligible
        ),
        "eligible_event_count": len(eligible),
        "measured_event_count": measured_count,
    }
    return metrics, compact_rows, integrity


def _tuple_referent_gate_pass(
    metrics: dict[str, float], gate: dict[str, Any]
) -> bool:
    return (
        metrics["event_coverage"] >= float(gate["event_coverage_min"])
        and metrics["visibility_timing_macro_f1"]
        >= float(gate["visibility_timing_macro_f1_min"])
        and metrics["no_referent_specificity"]
        >= float(gate["no_referent_specificity_min"])
        and metrics["dominance_weighted_kappa"]
        >= float(gate["dominance_weighted_kappa_min"])
        and metrics["ambiguity_macro_f1"]
        >= float(gate["ambiguity_macro_f1_min"])
        and metrics["valid_geometry_and_monotonic_track_fraction"]
        >= float(gate["valid_geometry_and_monotonic_track_fraction_required"])
    )


def _tuple_referent_module(context: dict[str, Any]) -> dict[str, Any]:
    cfg = context["cfg"]
    axis_id = "utterance_centered_referent_visibility_dominance_ambiguity"
    axis = _tuple_axis(cfg, axis_id)
    gate = axis["public_gate"]
    definitions = axis["definitions"]
    execution = _tuple_qualification_execution(cfg)
    minimum_valid = int(execution["phase_aggregation"]["minimum_valid_samples"])
    tracks = _tuple_grounding_sampled_tracks(context)
    if context.get("engineering_health") is True:
        accepted = 0
        for track in tracks:
            observation = track.get("adapter_observation", {})
            if observation.get("status") == "ABSTAIN":
                if track.get("samples"):
                    raise RuntimeError("E_TUPLE_HEALTH_REFERENT_ABSTAIN_STATE")
                continue
            if observation.get("status") != "ACCEPT":
                raise RuntimeError("E_TUPLE_HEALTH_REFERENT_ADAPTER_STATE")
            accepted += 1
            if track.get("category_mismatch") or "decode_error_code" in track:
                raise RuntimeError("E_TUPLE_HEALTH_REFERENT_CATEGORY_OR_DECODE")
            samples = track.get("samples")
            if not isinstance(samples, list) or len(samples) < minimum_valid:
                raise RuntimeError("E_TUPLE_HEALTH_REFERENT_SAMPLE_COUNT")
            prior_time = -math.inf
            for sample in samples:
                sample_time = float(sample.get("sample_time", math.nan))
                if (
                    not math.isfinite(sample_time)
                    or sample_time <= prior_time
                    or sample.get("phase") not in {"before", "during", "after"}
                    or sample.get("inference_succeeded") is not True
                    or not isinstance(sample.get("candidates"), list)
                ):
                    raise RuntimeError("E_TUPLE_HEALTH_REFERENT_SAMPLE_SCHEMA")
                prior_time = sample_time
                image = sample.get("image")
                shape = getattr(image, "shape", ())
                if len(shape) != 3 or shape[-1] != 3:
                    raise RuntimeError("E_TUPLE_HEALTH_REFERENT_IMAGE_SCHEMA")
                for candidate in sample["candidates"]:
                    if (
                        candidate.get("valid") is not True
                        or not _valid_normalized_box(candidate.get("box"))
                        or not all(
                            isinstance(candidate.get(key), (int, float))
                            and math.isfinite(float(candidate[key]))
                            for key in (
                                "box_score",
                                "text_score",
                                "mask_fraction",
                                "center_distance",
                            )
                        )
                        or not 0.0 <= float(candidate["box_score"]) <= 1.0
                        or not 0.0 <= float(candidate["text_score"]) <= 1.0
                        or not 0.0 < float(candidate["mask_fraction"]) <= 1.0
                    ):
                        raise RuntimeError(
                            "E_TUPLE_HEALTH_REFERENT_CANDIDATE_SCHEMA"
                        )
                    mask = candidate.get("mask")
                    if getattr(mask, "shape", None) != tuple(shape[:2]):
                        raise RuntimeError(
                            "E_TUPLE_HEALTH_REFERENT_MASK_GEOMETRY"
                        )
        if accepted < 1:
            raise RuntimeError("E_TUPLE_HEALTH_REFERENT_MODEL_NOT_EXERCISED")
        production_output = [
            {
                "adapter_status": track.get("adapter_observation", {}).get(
                    "status"
                ),
                "category_mismatch": bool(track.get("category_mismatch")),
                "decode_error": "decode_error_code" in track,
                "samples": [
                    {
                        "phase": sample["phase"],
                        "sample_time": float(sample["sample_time"]),
                        "image_shape": list(sample["image"].shape),
                        "candidates": [
                            {
                                "valid": candidate["valid"],
                                "box": candidate["box"],
                                "box_score": candidate["box_score"],
                                "text_score": candidate["text_score"],
                                "mask_fraction": candidate["mask_fraction"],
                                "center_distance": candidate["center_distance"],
                                "mask_shape": list(candidate["mask"].shape),
                            }
                            for candidate in sample["candidates"]
                        ],
                    }
                    for sample in track.get("samples", [])
                ],
            }
            for track in tracks
        ]
        return _tuple_health_pass_result(
            "referent", len(tracks), production_output
        )
    grids = _tuple_frozen_threshold_grids(cfg)
    if context["partition"] == "development":
        candidates = []
        for box_threshold in grids["Grounding_DINO_box_score"]:
            for text_threshold in grids["Grounding_DINO_text_score"]:
                metrics, rows, integrity = _tuple_referent_metrics(
                    tracks,
                    box_threshold,
                    text_threshold,
                    definitions,
                    minimum_valid,
                )
                candidates.append(
                    {
                        "Grounding_DINO_box_score": box_threshold,
                        "Grounding_DINO_text_score": text_threshold,
                        **metrics,
                        "eligible": _tuple_referent_gate_pass(metrics, gate)
                        and not integrity["invalid_retained_record_count"]
                        and not integrity["inference_failure_count"],
                        "_rows": rows,
                        "_integrity": integrity,
                    }
                )
        selected = _select_frozen_grid_result(
            candidates,
            primary_metric="visibility_timing_macro_f1",
            threshold_fields=(
                "Grounding_DINO_box_score",
                "Grounding_DINO_text_score",
            ),
        )
        reported = selected or max(
            candidates,
            key=lambda row: (
                float(row["visibility_timing_macro_f1"]),
                float(row["Grounding_DINO_box_score"]),
                float(row["Grounding_DINO_text_score"]),
            ),
        )
        selected_thresholds = (
            {
                "Grounding_DINO_box_score": float(
                    selected["Grounding_DINO_box_score"]
                ),
                "Grounding_DINO_text_score": float(
                    selected["Grounding_DINO_text_score"]
                ),
            }
            if selected is not None
            else {}
        )
    else:
        thresholds = context["thresholds"]
        box_threshold = float(thresholds["Grounding_DINO_box_score"])
        text_threshold = float(thresholds["Grounding_DINO_text_score"])
        metrics, rows, integrity = _tuple_referent_metrics(
            tracks,
            box_threshold,
            text_threshold,
            definitions,
            minimum_valid,
        )
        reported = {
            "Grounding_DINO_box_score": box_threshold,
            "Grounding_DINO_text_score": text_threshold,
            **metrics,
            "eligible": _tuple_referent_gate_pass(metrics, gate)
            and not integrity["invalid_retained_record_count"]
            and not integrity["inference_failure_count"],
            "_rows": rows,
            "_integrity": integrity,
        }
        selected_thresholds = {
            "Grounding_DINO_box_score": box_threshold,
            "Grounding_DINO_text_score": text_threshold,
        }
    passed = bool(reported["eligible"] and selected_thresholds)
    if selected_thresholds:
        context.setdefault("module_cache", {})[
            "selected_grounding_thresholds"
        ] = dict(selected_thresholds)
        context.setdefault("_selected_thresholds", {}).update(
            selected_thresholds
        )
    metrics = {
        key: value
        for key, value in reported.items()
        if key
        not in {
            "eligible",
            "_rows",
            "_integrity",
            "Grounding_DINO_box_score",
            "Grounding_DINO_text_score",
        }
    }
    integrity = reported["_integrity"]
    return {
        "status": "PASS" if passed else "NO_GO",
        "axis_results": {
            axis_id: {
                "status": "PASS" if passed else "NO_GO",
                "metrics": metrics,
            }
        },
        "metrics": metrics,
        "selected_thresholds": selected_thresholds,
        "rows": reported["_rows"],
        "row_count": len(context["rows"]["referent_attribute"]),
        "failure_count": int(integrity["inference_failure_count"]),
        "abstention_count": int(integrity["scientific_abstention_count"]),
        "category_error_count": int(integrity["category_error_count"]),
        "invalid_retained_record_count": int(
            integrity["invalid_retained_record_count"]
        ),
        "silent_truncation_count": 0,
        "external_call_count": 0,
    }


def _decode_uniform_tuple_frames(media: Path, frame_count: int) -> Any:
    frames, source_count = _decode_uniform_activity_frames(media, frame_count)
    if source_count < frame_count:
        raise RuntimeError("E_TUPLE_QUALIFICATION_SOURCE_FRAME_COUNT")
    return frames


EGOHOS_STAGE_CLASSES = {
    "stage1": frozenset({0, 1, 2}),
    "stage2": frozenset({0, 1}),
    "stage3": frozenset({0, 1, 2, 3}),
}
EGOHOS_TARGET_SIDE_CLASSES = {
    "left hand": (1, 1),
    "right hand": (2, 2),
}


def _tuple_egohos_stage_class_map(
    stage: str, value: Any, expected_shape: tuple[int, int] | None = None
) -> Any:
    import numpy as np

    if stage not in EGOHOS_STAGE_CLASSES:
        raise RuntimeError("E_TUPLE_EGOHOS_STAGE_ID")
    mask = np.asarray(value)
    if (
        mask.ndim != 2
        or mask.size == 0
        or (expected_shape is not None and mask.shape != expected_shape)
        or not np.issubdtype(mask.dtype, np.number)
        or not np.isfinite(mask).all()
        or not np.equal(mask, np.floor(mask)).all()
        or not set(int(item) for item in np.unique(mask)).issubset(
            EGOHOS_STAGE_CLASSES[stage]
        )
    ):
        raise RuntimeError(f"E_TUPLE_EGOHOS_{stage.upper()}_MASK")
    return mask.astype(np.uint8, copy=False)


def _tuple_egohos_binary_dilation(mask: Any) -> Any:
    """Return the frozen one-pixel, eight-connected binary dilation."""

    import numpy as np

    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype != np.bool_:
        raise RuntimeError("E_TUPLE_EGOHOS_DILATION_MASK")
    padded = np.pad(value, 1, mode="constant", constant_values=False)
    return np.logical_or.reduce(
        [
            padded[y : y + value.shape[0], x : x + value.shape[1]]
            for y in range(3)
            for x in range(3)
        ]
    )


def _tuple_egohos_stage_masks(prediction: dict[str, Any]) -> dict[str, Any]:
    """Validate finite, in-bounds official EgoHOS stage class maps."""

    if not isinstance(prediction, dict) or set(prediction) != set(
        EGOHOS_STAGE_CLASSES
    ):
        raise RuntimeError("E_TUPLE_EGOHOS_STAGE_SET")
    output: dict[str, Any] = {}
    shape = None
    for stage in EGOHOS_STAGE_CLASSES:
        value = _tuple_egohos_stage_class_map(stage, prediction[stage])
        if shape is None:
            shape = value.shape
        elif value.shape != shape:
            raise RuntimeError("E_TUPLE_EGOHOS_STAGE_SHAPE")
        output[stage] = value
    return output


def _tuple_egohos_target_mask(value: Any, shape: tuple[int, int]) -> Any:
    import numpy as np

    mask = np.asarray(value)
    if (
        mask.ndim != 2
        or mask.shape != shape
        or mask.size == 0
        or not np.issubdtype(mask.dtype, np.number)
        or not np.isfinite(mask).all()
        or not set(int(item) for item in np.unique(mask)).issubset({0, 1, 255})
    ):
        raise RuntimeError("E_TUPLE_EGOHOS_TARGET_MASK")
    output = mask > 0
    if not output.any():
        raise RuntimeError("E_TUPLE_EGOHOS_TARGET_MASK_EMPTY")
    return output


def _tuple_egohos_mask_observation(
    prediction: dict[str, Any],
    *,
    minimum_mask_fraction: float,
    target_hand_side: str | None,
    target_hand_mask: Any | None,
) -> dict[str, Any]:
    """Interpret the official three-stage outputs for one registered target.

    Hand presence is scored independently.  Contact requires a target-side
    first-order object and a one-pixel-dilated dense-contact boundary adjacent
    to the target-side hand.  Residual/discordant evidence abstains; it is not
    converted to a no-contact label.
    """

    import numpy as np

    threshold = float(minimum_mask_fraction)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise RuntimeError("E_TUPLE_EGOHOS_MASK_THRESHOLD")
    masks = _tuple_egohos_stage_masks(prediction)
    shape = masks["stage1"].shape
    pixels = int(masks["stage1"].size)
    left_fraction = float((masks["stage1"] == 1).sum()) / pixels
    right_fraction = float((masks["stage1"] == 2).sum()) / pixels
    fractions = {
        "left_hand_mask_fraction": left_fraction,
        "right_hand_mask_fraction": right_fraction,
    }
    if target_hand_side is None:
        if target_hand_mask is not None:
            raise RuntimeError("E_TUPLE_EGOHOS_NO_HAND_TARGET_MASK")
        return {
            "status": "MEASURED",
            "reason": None,
            "hand_visible": max(left_fraction, right_fraction) >= threshold,
            "contact": None,
            **fractions,
        }
    if target_hand_side not in EGOHOS_TARGET_SIDE_CLASSES:
        raise RuntimeError("E_TUPLE_EGOHOS_TARGET_SIDE")
    if target_hand_mask is None:
        raise RuntimeError("E_TUPLE_EGOHOS_TARGET_MASK_MISSING")
    truth = _tuple_egohos_target_mask(target_hand_mask, shape)
    hand_class, object_class = EGOHOS_TARGET_SIDE_CLASSES[target_hand_side]
    hand = masks["stage1"] == hand_class
    hand_fraction = float(hand.sum()) / pixels
    if hand_fraction < threshold:
        return {
            "status": "MEASURED",
            "reason": "TARGET_HAND_NOT_DETECTED",
            "hand_visible": False,
            "contact": None,
            "target_hand_mask_fraction": hand_fraction,
            **fractions,
        }
    if not np.logical_and(hand, truth).any():
        return {
            "status": "ABSTAIN",
            "reason": "TARGET_SIDE_MASK_DISCORDANT_WITH_GROUND_TRUTH",
            "hand_visible": None,
            "contact": None,
            "target_hand_mask_fraction": hand_fraction,
            **fractions,
        }
    boundary = masks["stage2"] == 1
    interacting_object = np.logical_or(
        masks["stage3"] == object_class,
        masks["stage3"] == 3,
    )
    object_fraction = float(interacting_object.sum()) / pixels
    adjacent_boundary = np.logical_and(
        boundary, _tuple_egohos_binary_dilation(hand)
    )
    object_boundary_overlap = np.logical_and(
        interacting_object, _tuple_egohos_binary_dilation(boundary)
    )
    object_present = object_fraction >= threshold
    if object_present and adjacent_boundary.any() and object_boundary_overlap.any():
        contact: bool | None = True
        reason = None
        status = "MEASURED"
    elif not interacting_object.any() and not adjacent_boundary.any():
        contact = False
        reason = None
        status = "MEASURED"
    else:
        contact = None
        reason = "CONTACT_EVIDENCE_DISCORDANT"
        status = "ABSTAIN"
    return {
        "status": status,
        "reason": reason,
        "hand_visible": True,
        "contact": contact,
        "target_hand_mask_fraction": hand_fraction,
        "target_object_mask_fraction": object_fraction,
        "adjacent_contact_boundary_present": bool(adjacent_boundary.any()),
        "object_boundary_overlap_present": bool(object_boundary_overlap.any()),
        **fractions,
    }


def _tuple_egohos_verified_no_hand_commitment(
    context: dict[str, Any], rows: list[dict[str, Any]]
) -> str:
    """Require a compact PASS seal and an exact row-to-seal provenance link."""

    seal = context.get("verified_no_hand_seal")
    if not isinstance(seal, dict) or seal.get("status") != "PASS":
        raise RuntimeError("E_TUPLE_EGOHOS_NO_HAND_PASS_SEAL")
    commitment = seal.get("verified_no_hand_seal_commitment_sha256")
    if not isinstance(commitment, str) or not re.fullmatch(r"[0-9a-f]{64}", commitment):
        raise RuntimeError("E_TUPLE_EGOHOS_NO_HAND_PASS_SEAL")
    no_hand_rows = [row for row in rows if row.get("stratum") == "verified_no_hand"]
    if not no_hand_rows or any(
        row.get("verified_no_hand_seal_commitment_sha256") != commitment
        for row in no_hand_rows
    ):
        raise RuntimeError("E_TUPLE_EGOHOS_NO_HAND_SEAL_LINK")
    return commitment


def _validate_tuple_egohos_fixture_rows(
    context: dict[str, Any], rows: list[dict[str, Any]]
) -> str:
    """Fail before model loading when fixture truth or retained masks are absent."""

    commitment = _tuple_egohos_verified_no_hand_commitment(context, rows)
    allowed = set(VISOR_HOS_STRATA)
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("E_TUPLE_EGOHOS_FIXTURE_ROWS")
    for row in rows:
        stratum = row.get("stratum")
        if stratum not in allowed:
            raise RuntimeError("E_TUPLE_EGOHOS_FIXTURE_STRATUM")
        if not all(
            isinstance(row.get(key), str) and row[key]
            for key in ("media_relative_path", "media_sha256")
        ):
            raise RuntimeError("E_TUPLE_EGOHOS_FIXTURE_MEDIA")
        if stratum == "verified_no_hand":
            if (
                row.get("contact") is not None
                or row.get("target_hand_side") is not None
                or row.get("target_hand_mask_relative_path") is not None
                or row.get("target_hand_mask_sha256") is not None
                or row.get("target_hand_mask_bytes") is not None
                or row.get("target_hand_mask_width") is not None
                or row.get("target_hand_mask_height") is not None
                or row.get("target_hand_boundary_vertex_count") != 0
                or row.get("target_hand_outside_canvas_vertex_count") != 0
                or row.get("target_hand_outside_canvas_component_count") != 0
                or row.get("target_hand_mask_exact_frame_binary_nonempty")
                is not False
                or row.get("verified_no_hand_seal_commitment_sha256")
                != commitment
            ):
                raise RuntimeError("E_TUPLE_EGOHOS_NO_HAND_FIXTURE_TRUTH")
            continue
        expected_contact = stratum == "contact"
        if (
            row.get("contact") is not expected_contact
            or row.get("target_hand_side") not in EGOHOS_TARGET_SIDE_CLASSES
            or type(row.get("target_hand_mask_width")) is not int
            or int(row["target_hand_mask_width"]) <= 0
            or type(row.get("target_hand_mask_height")) is not int
            or int(row["target_hand_mask_height"]) <= 0
            or row.get("source_polygon_finite_nonnegative") is not True
            or row.get("target_hand_mask_exact_frame_binary_nonempty") is not True
            or row.get("geometry_valid") is not True
            or not all(
                isinstance(row.get(key), str) and row[key]
                for key in (
                    "target_hand_mask_relative_path",
                    "target_hand_mask_sha256",
                )
            )
            or any(
                type(row.get(key)) is not int or int(row[key]) < 0
                for key in (
                    "target_hand_boundary_vertex_count",
                    "target_hand_outside_canvas_vertex_count",
                    "target_hand_outside_canvas_component_count",
                )
            )
        ):
            raise RuntimeError("E_TUPLE_EGOHOS_VISIBLE_HAND_FIXTURE_TRUTH")
    return commitment


def _read_tuple_egohos_target_mask(
    row: dict[str, Any],
    fixture_root: Path,
    *,
    require_source_alignment: bool = True,
):
    """Decode one sealed target mask without silently normalizing corruption."""

    import numpy as np
    from PIL import Image

    mask_path = _tuple_fixture_file(
        fixture_root,
        row["target_hand_mask_relative_path"],
        row["target_hand_mask_sha256"],
        row.get("target_hand_mask_bytes"),
    )
    source_size = None
    if require_source_alignment:
        source_path = _tuple_fixture_file(
            fixture_root,
            row["media_relative_path"],
            row["media_sha256"],
            row.get("media_bytes"),
        )
        with Image.open(source_path) as source:
            source_size = source.size
            if source_size[0] <= 0 or source_size[1] <= 0:
                raise RuntimeError("E_TUPLE_EGOHOS_SOURCE_IMAGE")
            source.verify()
    if any(
        type(row.get(key)) is not int or int(row[key]) <= 0
        for key in ("target_hand_mask_width", "target_hand_mask_height")
    ):
        raise RuntimeError("E_TUPLE_EGOHOS_TARGET_MASK_SERIALIZATION")
    declared_size = (
        int(row["target_hand_mask_width"]),
        int(row["target_hand_mask_height"]),
    )
    with Image.open(mask_path) as opened:
        if (
            opened.format != "PNG"
            or opened.mode != "L"
            or (source_size is not None and opened.size != source_size)
            or opened.size != declared_size
        ):
            raise RuntimeError("E_TUPLE_EGOHOS_TARGET_MASK_SERIALIZATION")
        target = np.asarray(opened).copy()
    if (
        target.shape != (declared_size[1], declared_size[0])
        or target.dtype != np.uint8
        or not np.isin(target, (0, 255)).all()
        or not (target == 255).any()
    ):
        raise RuntimeError("E_TUPLE_EGOHOS_TARGET_MASK_SERIALIZATION")
    return target


def _tuple_egohos_predict_rows(
    public: Path,
    device: str,
    rows: list[dict[str, Any]],
    fixture_root: Path,
    scratch: Path,
) -> list[dict[str, Any]]:
    """Stage exact source frames, then run the pinned official test pipeline."""

    from PIL import Image

    _require_external_or_ignored_output(scratch)
    run_root = scratch / "egohos-public-qualification"
    image_root = run_root / "images"
    image_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    image_paths = []
    original_shapes = []
    for ordinal, row in enumerate(rows):
        source = _tuple_fixture_file(
            fixture_root,
            row["media_relative_path"],
            row["media_sha256"],
            row.get("media_bytes"),
        )
        with Image.open(source) as opened:
            width, height = opened.size
            if opened.mode != "RGB" or width <= 0 or height <= 0:
                raise RuntimeError("E_TUPLE_EGOHOS_SOURCE_IMAGE")
            opened.verify()
        suffix = source.suffix.casefold()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            raise RuntimeError("E_TUPLE_EGOHOS_SOURCE_SUFFIX")
        target = image_root / f"item-{ordinal:03d}{suffix}"
        shutil.copy2(source, target)
        os.chmod(target, 0o600)
        if file_digest(target) != row["media_sha256"]:
            raise RuntimeError("E_TUPLE_EGOHOS_STAGED_IMAGE_HASH")
        image_paths.append(target)
        original_shapes.append((height, width))
    return _tuple_egohos_run_official_pipeline(
        public,
        device,
        image_paths,
        original_shapes,
        run_root,
    )


def _tuple_egohos_observations(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    fixture_root: Path,
    threshold: float,
    *,
    strict_engineering: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    if len(rows) != len(predictions) or not rows:
        raise RuntimeError("E_TUPLE_EGOHOS_PREDICTION_COUNT")
    observations = []
    invalid = 0
    for ordinal, (row, prediction) in enumerate(
        zip(rows, predictions, strict=True)
    ):
        target_mask = None
        if row["stratum"] != "verified_no_hand":
            target_mask = _read_tuple_egohos_target_mask(
                row, fixture_root, require_source_alignment=False
            )
        try:
            if target_mask is not None:
                stage_shape = _tuple_egohos_stage_masks(prediction)["stage1"].shape
                if target_mask.shape != stage_shape:
                    raise RuntimeError("E_TUPLE_EGOHOS_TARGET_MASK_ALIGNMENT")
            observation = _tuple_egohos_mask_observation(
                prediction,
                minimum_mask_fraction=threshold,
                target_hand_side=row.get("target_hand_side"),
                target_hand_mask=target_mask,
            )
        except RuntimeError as error:
            if strict_engineering:
                raise RuntimeError("E_TUPLE_EGOHOS_OBSERVATION_SCHEMA") from error
            code = str(error).split(maxsplit=1)[0]
            observation = {
                "status": "ABSTAIN",
                "reason": code if code.startswith("E_") else "E_TUPLE_EGOHOS_INVALID",
                "hand_visible": None,
                "contact": None,
            }
            invalid += 1
        observations.append(
            {
                "fixture_ordinal": int(row.get("fixture_ordinal", ordinal)),
                "stratum": row["stratum"],
                "target_hand_side": row.get("target_hand_side"),
                **observation,
            }
        )
    return observations, invalid


def _tuple_egohos_threshold_metrics(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    fixture_root: Path,
    threshold: float,
) -> dict[str, Any]:
    observations, invalid = _tuple_egohos_observations(
        rows, predictions, fixture_root, threshold
    )
    expected_presence = [row["stratum"] != "verified_no_hand" for row in rows]
    predicted_presence = [row["hand_visible"] for row in observations]
    presence = _binary_classification_metrics(
        expected_presence, predicted_presence
    )
    contact_indices = [
        ordinal
        for ordinal, row in enumerate(rows)
        if row["stratum"] != "verified_no_hand"
    ]
    contact_truth = [bool(rows[ordinal]["contact"]) for ordinal in contact_indices]
    contact_predicted = [observations[ordinal]["contact"] for ordinal in contact_indices]
    contact_expected_labels = [
        "contact" if value else "no_contact" for value in contact_truth
    ]
    contact_predicted_labels = [
        None if value is None else "contact" if value else "no_contact"
        for value in contact_predicted
    ]
    contact_binary = _binary_classification_metrics(
        contact_truth, contact_predicted
    )
    contact_macro_f1 = _multiclass_macro_f1(
        contact_expected_labels,
        contact_predicted_labels,
        ["contact", "no_contact"],
    )
    contact_coverage = _safe_divide(
        sum(value is not None for value in contact_predicted),
        len(contact_predicted),
    )
    coverage = min(presence["coverage"], contact_coverage)
    return {
        "hand_sensitivity": presence["recall"],
        "hand_specificity": presence["specificity"],
        "contact_no_contact_macro_f1": contact_macro_f1,
        # The fixed German mention is centered on the registered frame.  This
        # is therefore the declared non-independent contact-positive F1 only.
        "mention_contact_alignment_f1": contact_binary["f1"],
        "coverage": coverage,
        "hand_presence_coverage": presence["coverage"],
        "contact_state_coverage": contact_coverage,
        "visible_hand_item_count": len(contact_indices),
        "verified_no_hand_item_count": len(rows) - len(contact_indices),
        "contact_metric_item_count": len(contact_indices),
        "invalid_retained_record_count": invalid,
        "rows": observations,
    }


def _tuple_egohos_metrics_pass(
    metrics: dict[str, Any], gate: dict[str, Any]
) -> bool:
    return (
        metrics["hand_sensitivity"] >= float(gate["hand_sensitivity_min"])
        and metrics["hand_specificity"] >= float(gate["hand_specificity_min"])
        and metrics["contact_no_contact_macro_f1"]
        >= float(gate["contact_no_contact_macro_f1_min"])
        and metrics["mention_contact_alignment_f1"]
        >= float(gate["mention_contact_alignment_f1_min"])
        and metrics["coverage"] >= float(gate["coverage_min"])
        and metrics["invalid_retained_record_count"] == 0
    )


def _tuple_hand_contact_module(context: dict[str, Any]) -> dict[str, Any]:
    """Qualify EgoHOS hand presence and visible-hand contact as separate tasks."""

    cfg = context["cfg"]
    rows = context["rows"]["hand_contact"]
    fixture_root = context["fixture_root"]
    _validate_tuple_egohos_fixture_rows(context, rows)
    for row in rows:
        if row["stratum"] != "verified_no_hand":
            _read_tuple_egohos_target_mask(row, fixture_root)
    predictions = _tuple_egohos_predict_rows(
        context["public_root"],
        context["device"],
        rows,
        fixture_root,
        context["scratch_root"],
    )
    gate = _tuple_axis(cfg, "hand_action_coupling")["public_gate"]
    grid = _tuple_amendment(cfg)["public_qualification"][
        "threshold_development"
    ]["EgoHOS_min_mask_fraction_grid"]
    if grid != [0.0025, 0.005, 0.01, 0.02]:
        raise RuntimeError("E_TUPLE_EGOHOS_THRESHOLD_GRID")
    if context.get("engineering_health") is True:
        health_threshold = _tuple_health_inference_thresholds(cfg)[
            "EgoHOS_min_mask_fraction"
        ]
        observations, invalid = _tuple_egohos_observations(
            rows,
            predictions,
            fixture_root,
            health_threshold,
            strict_engineering=True,
        )
        if invalid or len(observations) != len(rows):
            raise RuntimeError("E_TUPLE_HEALTH_HAND_CONTACT_TRUNCATION")
        for observation in observations:
            if (
                observation.get("status") not in {"MEASURED", "ABSTAIN"}
                or not isinstance(observation.get("hand_visible"), (bool, type(None)))
                or not isinstance(observation.get("contact"), (bool, type(None)))
                or any(
                    not isinstance(observation.get(key), (int, float))
                    or not math.isfinite(float(observation[key]))
                    or not 0.0 <= float(observation[key]) <= 1.0
                    for key in (
                        "left_hand_mask_fraction",
                        "right_hand_mask_fraction",
                    )
                )
            ):
                raise RuntimeError("E_TUPLE_HEALTH_HAND_CONTACT_OUTPUT_SCHEMA")
        return _tuple_health_pass_result(
            "hand_contact", len(observations), observations
        )
    if context["partition"] == "development":
        candidates = []
        for threshold in grid:
            metrics = _tuple_egohos_threshold_metrics(
                rows, predictions, fixture_root, float(threshold)
            )
            candidates.append(
                {
                    "EgoHOS_min_mask_fraction": float(threshold),
                    **metrics,
                    "eligible": _tuple_egohos_metrics_pass(metrics, gate),
                }
            )
        selected = _select_frozen_grid_result(
            candidates,
            primary_metric="contact_no_contact_macro_f1",
            threshold_fields=("EgoHOS_min_mask_fraction",),
        )
        chosen = selected or max(
            candidates,
            key=lambda row: (
                row["contact_no_contact_macro_f1"],
                row["EgoHOS_min_mask_fraction"],
            ),
        )
        threshold = (
            float(selected["EgoHOS_min_mask_fraction"])
            if selected is not None
            else None
        )
    else:
        threshold = float(context["thresholds"]["EgoHOS_min_mask_fraction"])
        if threshold not in [float(value) for value in grid]:
            raise RuntimeError("E_TUPLE_EGOHOS_HOLDOUT_THRESHOLD")
        chosen = {
            "EgoHOS_min_mask_fraction": threshold,
            **_tuple_egohos_threshold_metrics(
                rows, predictions, fixture_root, threshold
            ),
        }
        chosen["eligible"] = _tuple_egohos_metrics_pass(chosen, gate)
        selected = chosen if chosen["eligible"] else None
    passed = threshold is not None and selected is not None
    rows_output = chosen.pop("rows")
    metrics = {
        key: value
        for key, value in chosen.items()
        if key not in {"eligible", "EgoHOS_min_mask_fraction"}
    }
    invalid = int(metrics["invalid_retained_record_count"])
    return {
        "status": "PASS" if passed else "NO_GO",
        "axis_results": {
            "hand_action_coupling": {
                "status": "PASS" if passed else "NO_GO",
                "metrics": metrics,
            }
        },
        "metrics": metrics,
        "selected_thresholds": (
            {"EgoHOS_min_mask_fraction": threshold}
            if threshold is not None
            else {}
        ),
        "rows": rows_output,
        "row_count": len(rows),
        "failure_count": 0,
        "invalid_retained_record_count": invalid,
        "silent_truncation_count": 0,
        "external_call_count": 0,
    }


def _tuple_sensor_observation(frames: Any) -> dict[str, Any]:
    from PIL import Image

    rows = []
    previous = None
    for raw in frames:
        image = Image.fromarray(raw)
        metrics = _image_metrics(image, previous)
        rows.append(metrics)
        previous = image
    output = {}
    for name in ("brightness", "blur_edge_strength", "motion_mean_absolute_luma"):
        values = [float(row[name]) for row in rows if row[name] is not None]
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError("E_TUPLE_QUALIFICATION_SENSOR_NONFINITE")
        output[name] = statistics.median(values)
        output[f"{name}_max"] = max(values)
    return output


def _tuple_sensor_module(context: dict[str, Any]) -> dict[str, Any]:
    cfg = context["cfg"]
    rows = context["rows"]["sensor"]
    fixture_root = context["fixture_root"]
    gate = _tuple_axis(cfg, "egocentric_sensor_regime")["public_gate"]
    bins = cfg["calibration_C"]["extractor"]["fixed_numeric_bins"]
    observations = []
    valid = 0
    exact_bins = 0
    bin_total = 0
    for row in rows:
        media = _tuple_fixture_file(
            fixture_root,
            row["media_relative_path"],
            row["media_sha256"],
            row.get("media_bytes"),
        )
        values = _tuple_sensor_observation(_decode_uniform_tuple_frames(media, 16))
        predicted_bins = {}
        for name in ("brightness", "blur_edge_strength", "motion_mean_absolute_luma"):
            predicted_bins[name] = bucket(float(values[name]), bins[name])
            if context.get("engineering_health") is not True:
                exact_bins += predicted_bins[name] == row["truth"][name]["bin"]
                bin_total += 1
        valid += 1
        observations.append(
            {
                "base_ordinal": int(row["base_ordinal"]),
                "condition": row["condition"],
                "values": values,
                "predicted_bins": predicted_bins,
            }
        )
    if context.get("engineering_health") is True:
        required_conditions = {
            "static",
            "low_translation",
            "high_translation",
            "mild_blur",
            "strong_blur",
            "dark",
            "bright",
            "hard_cut",
        }
        if len(observations) != len(rows) or any(
            row.get("condition") not in required_conditions
            or set(row.get("predicted_bins", {}))
            != {
                "brightness",
                "blur_edge_strength",
                "motion_mean_absolute_luma",
            }
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in row.get("values", {}).values()
            )
            for row in observations
        ):
            raise RuntimeError("E_TUPLE_HEALTH_SENSOR_OUTPUT_SCHEMA")
        return _tuple_health_pass_result("sensor", len(observations), observations)
    by_base = defaultdict(dict)
    for row in observations:
        by_base[row["base_ordinal"]][row["condition"]] = row["values"]
    direction_checks: list[bool] = []
    for values in by_base.values():
        if set(values) != {
            "static",
            "low_translation",
            "high_translation",
            "mild_blur",
            "strong_blur",
            "dark",
            "bright",
            "hard_cut",
        }:
            raise RuntimeError("E_TUPLE_QUALIFICATION_SENSOR_CONDITIONS")
        direction_checks.extend(
            [
                values["high_translation"]["motion_mean_absolute_luma"]
                > values["low_translation"]["motion_mean_absolute_luma"],
                values["low_translation"]["motion_mean_absolute_luma"]
                > values["static"]["motion_mean_absolute_luma"],
                values["strong_blur"]["blur_edge_strength"]
                < values["mild_blur"]["blur_edge_strength"],
                values["dark"]["brightness"] < values["static"]["brightness"],
                values["bright"]["brightness"] > values["static"]["brightness"],
                values["hard_cut"]["motion_mean_absolute_luma_max"]
                > values["static"]["motion_mean_absolute_luma_max"],
            ]
        )
    finite_fraction = _safe_divide(valid, len(rows))
    bin_accuracy = _safe_divide(exact_bins, bin_total)
    direction_fraction = _safe_divide(sum(direction_checks), len(direction_checks))
    passed = (
        finite_fraction >= float(gate["finite_in_bounds_fraction_required"])
        and direction_fraction
        >= float(gate["controlled_perturbation_direction_fraction_required"])
        and bin_accuracy >= float(gate["frozen_bin_accuracy_min"])
    )
    return {
        "status": "PASS" if passed else "NO_GO",
        "axis_results": {
            "egocentric_sensor_regime": {
                "status": "PASS" if passed else "NO_GO",
                "metrics": {
                    "finite_in_bounds_fraction": finite_fraction,
                    "controlled_perturbation_direction_fraction": direction_fraction,
                    "frozen_bin_accuracy": bin_accuracy,
                },
            }
        },
        "metrics": {
            "finite_in_bounds_fraction": finite_fraction,
            "controlled_perturbation_direction_fraction": direction_fraction,
            "frozen_bin_accuracy": bin_accuracy,
        },
        "rows": observations,
        "row_count": len(rows),
        "failure_count": 0,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
    }


def _decode_tuple_action_interval(
    media: Path, start: float, end: float, source_duration: float, frame_count: int
) -> Any:
    import decord
    import numpy as np

    if not all(math.isfinite(value) for value in (start, end, source_duration)):
        raise RuntimeError("E_TUPLE_ACTION_INTERVAL")
    clip_start = max(0.0, start - 0.25)
    clip_end = min(source_duration, end + 0.25)
    if clip_end <= clip_start:
        raise RuntimeError("E_TUPLE_ACTION_INTERVAL")
    reader = decord.VideoReader(str(media), ctx=decord.cpu(0), num_threads=1)
    if len(reader) < 2:
        raise RuntimeError("E_TUPLE_ACTION_DECODE_EMPTY")
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError("E_TUPLE_ACTION_FPS")
    first = max(0, min(len(reader) - 1, int(math.ceil(clip_start * fps))))
    last = max(first, min(len(reader) - 1, int(math.floor(clip_end * fps))))
    indices = np.linspace(first, last, frame_count, dtype=np.int64)
    if len(set(int(value) for value in indices)) != frame_count:
        raise RuntimeError("E_TUPLE_ACTION_DISTINCT_FRAME_COUNT")
    frames = reader.get_batch(indices.tolist()).asnumpy()
    if frames.ndim != 4 or frames.shape[0] != frame_count or frames.shape[-1] != 3:
        raise RuntimeError("E_TUPLE_ACTION_SILENT_TRUNCATION")
    return frames


def _tuple_action_predictions(
    raw_rows: list[dict[str, Any]],
    margin: float,
    labels: list[str],
    opposite: dict[str, str],
) -> tuple[list[str | None], dict[str, float]]:
    predicted: list[str | None] = []
    covered_rows = []
    for row in raw_rows:
        pair = (row["label"], opposite[row["label"]])
        first, second = (float(row["ordered_scores"][labels.index(label)]) for label in pair)
        if abs(first - second) < margin:
            predicted.append(None)
            continue
        predicted.append(pair[0] if first > second else pair[1])
        covered_rows.append(row)
    expected = [str(row["label"]) for row in raw_rows]
    macro_f1 = _multiclass_macro_f1(expected, predicted, labels)
    coverage = _safe_divide(len(covered_rows), len(raw_rows))
    correct_margin = _safe_divide(
        sum(
            float(row["ordered_scores"][labels.index(row["label"])])
            > float(row["ordered_scores"][labels.index(opposite[row["label"]])])
            for row in covered_rows
        ),
        len(covered_rows),
    )
    ordered_over_reversed = _safe_divide(
        sum(
            float(row["ordered_scores"][labels.index(row["label"])])
            > float(row["reversed_scores"][labels.index(row["label"])])
            for row in covered_rows
        ),
        len(covered_rows),
    )
    ordered_over_repeated = _safe_divide(
        sum(
            (
                float(row["ordered_scores"][labels.index(row["label"])])
                - float(row["ordered_scores"][labels.index(opposite[row["label"]])])
            )
            > (
                float(row["repeated_center_scores"][labels.index(row["label"])])
                - float(
                    row["repeated_center_scores"][
                        labels.index(opposite[row["label"]])
                    ]
                )
            )
            for row in covered_rows
        ),
        len(covered_rows),
    )
    return predicted, {
        "ordered_action_direction_macro_f1": macro_f1,
        "opposite_pair_correct_margin_fraction": correct_margin,
        "ordered_over_time_reversed_target_score_fraction": ordered_over_reversed,
        "ordered_over_repeated_center_confidence_fraction": ordered_over_repeated,
        "coverage": coverage,
    }


def _tuple_action_metrics_pass(
    metrics: dict[str, Any], gate: dict[str, Any]
) -> bool:
    """Apply every frozen action and integrity floor to one grid point."""

    required = {
        "ordered_action_direction_macro_f1",
        "opposite_pair_correct_margin_fraction",
        "ordered_over_time_reversed_target_score_fraction",
        "ordered_over_repeated_center_confidence_fraction",
        "coverage",
    }
    if not required <= set(metrics) or not all(
        isinstance(metrics[key], (int, float))
        and math.isfinite(float(metrics[key]))
        for key in required
    ):
        raise RuntimeError("E_TUPLE_ACTION_METRIC_RECORD")
    return (
        metrics["ordered_action_direction_macro_f1"]
        >= float(gate["ordered_action_direction_macro_f1_min"])
        and metrics["opposite_pair_correct_margin_fraction"]
        >= float(gate["opposite_pair_correct_margin_fraction_min"])
        and metrics["ordered_over_time_reversed_target_score_fraction"]
        >= float(gate["ordered_over_time_reversed_target_score_fraction_min"])
        and metrics["ordered_over_repeated_center_confidence_fraction"]
        >= float(gate["ordered_over_repeated_center_confidence_fraction_min"])
        and metrics["coverage"] >= float(gate["coverage_min"])
        and int(metrics.get("failure_count", 0)) == 0
        and int(metrics.get("invalid_retained_record_count", 0)) == 0
        and int(metrics.get("silent_truncation_count", 0)) == 0
        and int(metrics.get("external_call_count", 0)) == 0
    )


def _select_tuple_action_development(
    candidates: list[dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any] | None:
    qualified = [
        {**candidate, "eligible": _tuple_action_metrics_pass(candidate, gate)}
        for candidate in candidates
    ]
    return _select_frozen_grid_result(
        qualified,
        primary_metric="ordered_action_direction_macro_f1",
        threshold_fields=("abstention_margin",),
    )


def _select_tuple_action_diagnostic(
    candidates: list[dict[str, Any]], gate: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Select a passing action point or the frozen diagnostic fallback once."""

    qualified = [
        {**candidate, "eligible": _tuple_action_metrics_pass(candidate, gate)}
        for candidate in candidates
    ]
    selected = _select_frozen_grid_result(
        qualified,
        primary_metric="ordered_action_direction_macro_f1",
        threshold_fields=("abstention_margin",),
    )
    if selected is not None:
        return selected, True
    fallback = _select_frozen_grid_result(
        qualified,
        primary_metric="ordered_action_direction_macro_f1",
        threshold_fields=("abstention_margin",),
        require_eligible=False,
    )
    if fallback is None:
        raise RuntimeError("E_TUPLE_ACTION_DIAGNOSTIC_SELECTION")
    return fallback, False


def _tuple_order_action_egohod_wiring(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the exact action diagnostic without reopening activity selection."""

    _engineering_health_amendment(cfg)
    active = _construct_aligned_ltx_resume_amendment(cfg)
    tuple_amendment = _tuple_amendment(cfg)
    if (
        active["combined_gate"].get("order_action")
        != "SUPPORTING_DIAGNOSTIC_NONBLOCKING_ON_THE_EXISTING_SUBJECT_VIDEO_DISJOINT_44_DEVELOPMENT_AND_44_HOLDOUT_FIXTURES"
        or "already pinned EgoHOD EgoVideo-L checkpoint"
        not in tuple_amendment["genuinely_order_dependent_action_control"].get(
            "model", ""
        )
    ):
        raise RuntimeError("E_TUPLE_ACTION_EGOHOD_IDENTITY")
    try:
        historical = cfg["calibration_C"]["extractor"][
            "activity_checkpoint_selection_amendment"
        ]
        candidates = historical["bounded_candidates"]
        runtime = historical["runtime_environment"]["egohod"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_ACTION_EGOHOD_IDENTITY") from error
    matches = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_id") == "egohod_egovideo_l_zero_shot"
    ]
    if (
        len(matches) != 1
        or digest(matches[0]) != TUPLE_ORDER_ACTION_EGOHOD_CANDIDATE_SHA256
        or digest(runtime) != TUPLE_ORDER_ACTION_EGOHOD_RUNTIME_SHA256
    ):
        raise RuntimeError("E_TUPLE_ACTION_EGOHOD_IDENTITY")
    return matches[0], runtime


def _tuple_order_action_module(context: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    cfg = context["cfg"]
    rows = context["rows"]["order_action"]
    fixture_root = context["fixture_root"]
    device = context["device"]
    protocol = _tuple_fixture_protocol(cfg)["order_dependent_action_control"]
    amendment = _tuple_amendment(cfg)["genuinely_order_dependent_action_control"]
    labels = list(protocol["labels"])
    opposite = {
        first: second
        for pair in protocol["class_code_pairs"]
        for first, second in (pair["pair"], tuple(reversed(pair["pair"])))
    }
    candidate, runtime = _tuple_order_action_egohod_wiring(cfg)
    score, frame_count, _ = _load_egohod_activity_adapter(
        context["public_root"],
        candidate,
        cfg,
        labels,
        device,
        runtime_override=runtime,
        prompt_groups_override=protocol["prompt_ensembles"],
    )
    raw = []
    try:
        for row in rows:
            media = _tuple_fixture_file(
                fixture_root,
                row["media_relative_path"],
                row["media_sha256"],
                row.get("media_bytes"),
            )
            frames = _decode_tuple_action_interval(
                media,
                float(row["start"]),
                float(row["end"]),
                float(row["source_duration"]),
                frame_count,
            )
            controls = {
                "ordered_scores": frames,
                "reversed_scores": frames[::-1].copy(),
                "repeated_center_scores": np.repeat(
                    frames[len(frames) // 2 : len(frames) // 2 + 1],
                    len(frames),
                    axis=0,
                ),
            }
            raw.append(
                {
                    "fixture_ordinal": int(row["fixture_ordinal"]),
                    "label": row["label"],
                    **{
                        key: _finite_vector(
                            score(frames_value),
                            len(labels),
                            "E_TUPLE_ACTION_NONFINITE_SCORE",
                        )
                        for key, frames_value in controls.items()
                    },
                }
            )
    finally:
        del score
        _release_cuda()
    if context.get("engineering_health") is True:
        if len(raw) != len(rows):
            raise RuntimeError("E_TUPLE_HEALTH_ORDER_ACTION_TRUNCATION")
        for value in raw:
            for key in (
                "ordered_scores",
                "reversed_scores",
                "repeated_center_scores",
            ):
                scores = value.get(key)
                if (
                    not isinstance(scores, list)
                    or len(scores) != len(labels)
                    or not all(
                        isinstance(score_value, (int, float))
                        and math.isfinite(float(score_value))
                        for score_value in scores
                    )
                ):
                    raise RuntimeError(
                        "E_TUPLE_HEALTH_ORDER_ACTION_OUTPUT_SCHEMA"
                    )
        return _tuple_health_pass_result("order_action", len(raw), raw)
    gate = amendment["gate"]
    if context["partition"] == "development":
        candidates = []
        for margin in protocol["development_abstention_margin_grid"]:
            _predicted, metrics = _tuple_action_predictions(
                raw, float(margin), labels, opposite
            )
            candidates.append(
                {
                    "abstention_margin": float(margin),
                    **metrics,
                }
            )
        selected, passed = _select_tuple_action_diagnostic(candidates, gate)
        threshold = float(selected["abstention_margin"])
        metrics = selected
    else:
        threshold = float(context["thresholds"]["action_abstention_margin"])
        if threshold not in [
            float(value) for value in protocol["development_abstention_margin_grid"]
        ]:
            raise RuntimeError("E_TUPLE_ACTION_HOLDOUT_THRESHOLD")
        _predicted, metrics = _tuple_action_predictions(raw, threshold, labels, opposite)
        passed = _tuple_action_metrics_pass(metrics, gate)
    return {
        "status": "PASS" if passed else "NO_GO_DIAGNOSTIC",
        "metrics": metrics,
        "selected_thresholds": (
            {"action_abstention_margin": threshold} if threshold is not None else {}
        ),
        "rows": raw,
        "row_count": len(rows),
        "failure_count": 0,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
    }


def _tuple_perceptual_hash(image: Any) -> int:
    import numpy as np

    pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    indices = np.arange(32)
    frequencies = np.arange(32)[:, None]
    basis = np.cos(np.pi * (2 * indices + 1) * frequencies / 64)
    basis[0] *= 1 / np.sqrt(2)
    basis *= np.sqrt(2 / 32)
    low = (basis @ pixels @ basis.T)[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    return sum(
        int(value > median) << index for index, value in enumerate(low)
    )


def _load_tuple_dinov2(public: Path, device: str):
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
    return model


def _tuple_dinov2_features(model: Any, images: list[Any], device: str) -> Any:
    import torch
    import torch.nn.functional as functional
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform

    tensors = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    standard = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    for image in images:
        tensor = transform.pil_to_tensor(image.convert("RGB")).float() / 255.0
        tensor = transform.resize(
            tensor, 256, interpolation=InterpolationMode.BICUBIC, antialias=True
        )
        tensor = transform.center_crop(tensor, [224, 224])
        tensors.append((tensor - mean) / standard)
    with torch.inference_mode():
        output = functional.normalize(model(torch.stack(tensors).to(device)), dim=-1)
    if output.ndim != 2 or output.shape[0] != len(images) or not torch.isfinite(output).all():
        raise RuntimeError("E_TUPLE_DINOV2_OUTPUT")
    return output.cpu()


def _tuple_recurrence_metrics(
    raw: list[dict[str, Any]], threshold: float, phash_max: int
) -> dict[str, float]:
    expected = [bool(row["same_referent"]) for row in raw]
    predicted = [float(row["cosine_similarity"]) >= threshold for row in raw]
    classification = _binary_classification_metrics(expected, predicted)
    near_truth = [bool(row["near_duplicate"]) for row in raw]
    near_predicted = [
        int(row["phash_hamming_distance"]) <= phash_max for row in raw
    ]
    near = _binary_classification_metrics(near_truth, near_predicted)
    negative_indices = [index for index, value in enumerate(expected) if not value]
    negative_fpr = _safe_divide(
        sum(predicted[index] for index in negative_indices), len(negative_indices)
    )
    return {
        "same_object_cross_episode_balanced_accuracy": classification[
            "balanced_accuracy"
        ],
        "recurrence_event_f1": classification["f1"],
        "near_duplicate_pair_balanced_accuracy": near["balanced_accuracy"],
        "negative_pair_false_positive_rate": negative_fpr,
        "coverage": 1.0,
    }


def _tuple_recurrence_module(context: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    cfg = context["cfg"]
    rows = context["rows"]["recurrence"]
    fixture_root = context["fixture_root"]
    gate = _tuple_axis(cfg, "cross_episode_recurrence")["public_gate"]
    execution = _tuple_qualification_execution(cfg)
    phash_max = int(execution["pHash"]["near_duplicate_hamming_distance_max"])
    _validate_tuple_recurrence_fixture_rows(rows, fixture_root)
    model = _load_tuple_dinov2(context["public_root"], context["device"])
    raw = []
    try:
        for row in rows:
            first_path = _tuple_fixture_file(
                fixture_root, row["first_relative_path"], row["first_sha256"]
            )
            second_path = _tuple_fixture_file(
                fixture_root, row["second_relative_path"], row["second_sha256"]
            )
            first_mask_path = _tuple_fixture_file(
                fixture_root,
                row["first_mask_relative_path"],
                row["first_mask_sha256"],
            )
            second_mask_path = _tuple_fixture_file(
                fixture_root,
                row["second_mask_relative_path"],
                row["second_mask_sha256"],
            )
            with Image.open(first_path) as source:
                first = source.convert("RGB").copy()
            with Image.open(second_path) as source:
                second = source.convert("RGB").copy()
            with Image.open(first_mask_path) as source:
                first_mask = source.convert("L").copy()
            with Image.open(second_mask_path) as source:
                second_mask = source.convert("L").copy()
            if (
                first_mask.size != first.size
                or second_mask.size != second.size
                or first_mask.getbbox() is None
                or second_mask.getbbox() is None
            ):
                raise RuntimeError("E_TUPLE_RECURRENCE_MASK")
            neutral = Image.new("RGB", first.size, (112, 118, 125))
            neutral.paste(first, mask=first_mask)
            first = neutral
            neutral = Image.new("RGB", second.size, (112, 118, 125))
            neutral.paste(second, mask=second_mask)
            second = neutral
            features = _tuple_dinov2_features(
                model, [first, second], context["device"]
            )
            similarity = float(features[0] @ features[1])
            if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
                raise RuntimeError("E_TUPLE_RECURRENCE_SIMILARITY")
            raw.append(
                {
                    "fixture_ordinal": int(row["fixture_ordinal"]),
                    "same_referent": bool(row["same_referent"]),
                    "near_duplicate": bool(row["near_duplicate"]),
                    "cosine_similarity": similarity,
                    "exact_duplicate": row["first_sha256"] == row["second_sha256"],
                    "phash_hamming_distance": (
                        _tuple_perceptual_hash(first)
                        ^ _tuple_perceptual_hash(second)
                    ).bit_count(),
                }
            )
    finally:
        del model
        _release_cuda()
    if context.get("engineering_health") is True:
        if len(raw) != len(rows):
            raise RuntimeError("E_TUPLE_HEALTH_RECURRENCE_TRUNCATION")
        for value in raw:
            if (
                not isinstance(value.get("same_referent"), bool)
                or not isinstance(value.get("near_duplicate"), bool)
                or not isinstance(value.get("cosine_similarity"), (int, float))
                or not math.isfinite(float(value["cosine_similarity"]))
                or not -1.0 <= float(value["cosine_similarity"]) <= 1.0
                or type(value.get("phash_hamming_distance")) is not int
                or not 0 <= int(value["phash_hamming_distance"]) <= 63
            ):
                raise RuntimeError("E_TUPLE_HEALTH_RECURRENCE_OUTPUT_SCHEMA")
        return _tuple_health_pass_result("recurrence", len(raw), raw)
    grids = _tuple_amendment(cfg)["public_qualification"][
        "threshold_development"
    ]["DINOv2_recurrence_cosine_grid"]
    if context["partition"] == "development":
        threshold = float(
            _engineering_health_amendment(cfg)["scientific_threshold_state"]
            ["DINOv2_recurrence_cosine"]
        )
        if threshold != 0.85 or threshold not in [float(value) for value in grids]:
            raise RuntimeError("E_TUPLE_RECURRENCE_FIXED_THRESHOLD")
        metrics = _tuple_recurrence_metrics(raw, threshold, phash_max)
        selected = {
            "eligible": (
                metrics["same_object_cross_episode_balanced_accuracy"]
                >= float(gate["same_object_cross_episode_balanced_accuracy_min"])
                and metrics["recurrence_event_f1"]
                >= float(gate["recurrence_event_f1_min"])
                and metrics["near_duplicate_pair_balanced_accuracy"]
                >= float(gate["near_duplicate_pair_balanced_accuracy_min"])
                and metrics["negative_pair_false_positive_rate"]
                <= float(gate["negative_pair_false_positive_rate_max"])
                and metrics["coverage"] >= float(gate["coverage_min"])
            )
        }
    else:
        threshold = float(context["thresholds"]["DINOv2_recurrence_cosine"])
        metrics = _tuple_recurrence_metrics(raw, threshold, phash_max)
        selected = {"eligible": (
            metrics["same_object_cross_episode_balanced_accuracy"]
            >= float(gate["same_object_cross_episode_balanced_accuracy_min"])
            and metrics["recurrence_event_f1"] >= float(gate["recurrence_event_f1_min"])
            and metrics["near_duplicate_pair_balanced_accuracy"]
            >= float(gate["near_duplicate_pair_balanced_accuracy_min"])
            and metrics["negative_pair_false_positive_rate"]
            <= float(gate["negative_pair_false_positive_rate_max"])
            and metrics["coverage"] >= float(gate["coverage_min"])
        )}
    passed = threshold is not None and selected is not None and selected["eligible"] is True
    return {
        "status": "PASS" if passed else "NO_GO",
        "axis_results": {
            "cross_episode_recurrence": {
                "status": "PASS" if passed else "NO_GO",
                "metrics": {key: value for key, value in metrics.items() if key != "eligible"},
            }
        },
        "metrics": {key: value for key, value in metrics.items() if key != "eligible"},
        "selected_thresholds": (
            {"DINOv2_recurrence_cosine": threshold}
            if threshold is not None
            else {}
        ),
        "rows": raw,
        "row_count": len(rows),
        "failure_count": 0,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
    }


TUPLE_ATTRIBUTE_VALUES = {
    "color": ("red", "blue", "green", "yellow"),
    "relative_size": ("big", "small"),
}
TUPLE_ATTRIBUTE_PALETTE = {
    "red": (220.0, 50.0, 45.0),
    "blue": (45.0, 90.0, 220.0),
    "green": (50.0, 170.0, 75.0),
    "yellow": (230.0, 205.0, 45.0),
}


def _tuple_attribute_prompt_groups(cfg: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    frozen = _tuple_qualification_execution(cfg)["attribute_prompts"]
    templates = frozen.get("template_order")
    values = frozen.get("public_qualified_values")
    if templates != [
        "a {value} object",
        "an object that is {value}",
        "the visible object is {value}",
    ] or values != {
        "color": ["red", "blue", "green", "yellow"],
        "relative_size": ["big", "small"],
    }:
        raise RuntimeError("E_TUPLE_ATTRIBUTE_PROMPT_CONTRACT")
    return {
        family: {
            value: [template.format(value=value) for template in templates]
            for value in family_values
        }
        for family, family_values in values.items()
    }


def _tuple_attribute_palette_label(median_rgb: Any) -> str:
    import numpy as np

    value = np.asarray(median_rgb, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise RuntimeError("E_TUPLE_ATTRIBUTE_PALETTE_INPUT")
    return min(
        TUPLE_ATTRIBUTE_PALETTE,
        key=lambda label: (
            float(
                np.square(
                    value - np.asarray(TUPLE_ATTRIBUTE_PALETTE[label])
                ).sum()
            ),
            label,
        ),
    )


def _tuple_attribute_masked_image(image_array: Any, predicted_mask: Any):
    import numpy as np
    from PIL import Image

    measurements = _predicted_mask_attribute_measurements(
        image_array, predicted_mask, mask_role="predicted_SAM_mask"
    )
    image = np.asarray(image_array, dtype=np.uint8)
    selected = np.asarray(predicted_mask) > 0
    neutral = np.full(image.shape, (112, 118, 125), dtype=np.uint8)
    neutral[selected] = image[selected]
    return Image.fromarray(neutral, mode="RGB"), measurements


def _tuple_attribute_noun_lemma(category: str) -> str:
    mapping = {
        "sports ball": "ball",
        "cup": "cup",
        "bottle": "bottle",
        "bowl": "bowl",
        "book": "book",
        "chair": "chair",
        "apple": "apple",
        "banana": "banana",
    }
    if category not in mapping:
        raise RuntimeError("E_TUPLE_ATTRIBUTE_CATEGORY")
    return mapping[category]


def _tuple_attribute_truth(row: dict[str, Any]) -> dict[str, Any]:
    required_row = {
        "fixture_ordinal",
        "category",
        "scenario",
        "source_image_id",
        "source_annotation_id",
        "attribute_pair_source_index",
        "attribute_pair_id",
        "episode_id",
        "truth",
    }
    if not required_row <= set(row) or not isinstance(row["truth"], dict):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_FIXTURE_SCHEMA")
    truth = row["truth"]
    required_truth = {
        "attribute",
        "attribute_family",
        "target_longest_side_pixels",
        "background_identity_role",
        "visibility_by_phase",
        "dominance_by_phase",
        "candidate_count_by_phase",
        "sample_count_by_phase",
        "sampled_mask_truth",
        "attribute_contrast_expected",
        "attribute_null_reason",
        "reference_mask_role",
        "deterministic_measurement_mask_role",
    }
    if not required_truth <= set(truth):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_FIXTURE_SCHEMA")
    family = truth["attribute_family"]
    attribute = truth["attribute"]
    if (
        family not in TUPLE_ATTRIBUTE_VALUES
        or attribute not in TUPLE_ATTRIBUTE_VALUES[family]
        or truth["reference_mask_role"] != "TRUTH_ONLY_NOT_A_MEASUREMENT_INPUT"
        or truth["deterministic_measurement_mask_role"] != "predicted_SAM_mask"
        or not isinstance(truth["attribute_contrast_expected"], bool)
        or set(truth["visibility_by_phase"]) != {"before", "during", "after"}
        or not all(
            isinstance(value, bool)
            for value in truth["visibility_by_phase"].values()
        )
    ):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_FIXTURE_SCHEMA")
    expected_null_reason = None if truth["attribute_contrast_expected"] else str(
        truth["attribute_null_reason"]
    )
    if (truth["attribute_null_reason"] is None) != (
        truth["attribute_contrast_expected"] is True
    ) or expected_null_reason not in {
        None,
        "NO_ACCEPTED_ADJECTIVE_NOUN_SPAN",
        "NO_PREDICTED_REFERENT_MASK",
    }:
        raise RuntimeError("E_TUPLE_ATTRIBUTE_FIXTURE_SCHEMA")
    if family == "relative_size":
        if (
            truth["target_longest_side_pixels"]
            != REFERENT_ATTRIBUTE_SIZE_LONGEST_PIXELS[attribute]
            or truth["background_identity_role"] != "shared_relative_size_pair"
            or not isinstance(row["attribute_pair_id"], str)
            or not row["attribute_pair_id"]
        ):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_SIZE_PAIR_SCHEMA")
    elif row["attribute_pair_id"] is not None:
        raise RuntimeError("E_TUPLE_ATTRIBUTE_COLOR_PAIR_SCHEMA")
    return truth


def _tuple_attribute_language_accepts(
    row: dict[str, Any], adapter_event: Any
) -> bool:
    if not isinstance(adapter_event, dict) or set(adapter_event) != {
        "status",
        "abstention_reason",
        "mentions",
    }:
        raise RuntimeError("E_TUPLE_ATTRIBUTE_ADAPTER_EVENT_SCHEMA")
    status = adapter_event["status"]
    reason = adapter_event["abstention_reason"]
    mentions = adapter_event["mentions"]
    if status not in {"ACCEPT", "ABSTAIN"} or not isinstance(mentions, list):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_ADAPTER_EVENT_SCHEMA")
    if status == "ABSTAIN":
        if not isinstance(reason, str) or not reason or mentions:
            raise RuntimeError("E_TUPLE_ATTRIBUTE_ADAPTER_EVENT_SCHEMA")
        return False
    if reason is not None:
        raise RuntimeError("E_TUPLE_ATTRIBUTE_ADAPTER_EVENT_SCHEMA")
    expected = {
        "adjective": str(row["truth"]["attribute"]),
        "noun": _tuple_attribute_noun_lemma(str(row["category"])),
    }
    return expected in _adjacent_adjective_noun_spans(mentions)


def _tuple_attribute_target_candidate(
    sample: dict[str, Any], category: str, thresholds: dict[str, float]
) -> dict[str, Any] | None:
    candidates = sample.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_SCHEMA")
    retained = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_SCHEMA")
        if candidate.get("category") != category:
            continue
        box_score = candidate.get("box_score")
        text_score = candidate.get("text_score")
        if not isinstance(box_score, (int, float)) or not isinstance(
            text_score, (int, float)
        ):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_SCHEMA")
        if (
            float(box_score) < thresholds["Grounding_DINO_box_score"]
            or float(text_score) < thresholds["Grounding_DINO_text_score"]
        ):
            continue
        if not _valid_normalized_box(candidate.get("box")):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_PREDICTED_BOX_INVALID")
        center_distance = candidate.get("center_distance", 0.0)
        if not isinstance(center_distance, (int, float)) or not math.isfinite(
            float(center_distance)
        ):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_SCHEMA")
        retained.append(candidate)
    if not retained:
        return None
    return max(
        retained,
        key=lambda candidate: (
            float(candidate["box_score"]),
            float(candidate["text_score"]),
            -float(candidate.get("center_distance", 0.0)),
            tuple(float(value) for value in candidate["box"]),
        ),
    )


def _tuple_attribute_apply_size_pairs(output: list[dict[str, Any]]) -> None:
    size_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        if row["attribute_family"] == "relative_size":
            size_pairs[str(row["attribute_pair_id"])].append(row)
    for rows in size_pairs.values():
        if (
            len(rows) != 2
            or {row["attribute"] for row in rows} != {"big", "small"}
            or len({row["episode_id"] for row in rows}) != 1
            or len({row["source_image_id"] for row in rows}) != 1
            or len({row["source_annotation_id"] for row in rows}) != 1
            or {row["target_longest_side_pixels"] for row in rows} != {130, 72}
            or {row["background_identity_role"] for row in rows}
            != {"shared_relative_size_pair"}
        ):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_SIZE_PAIR_SCHEMA")
        by_value = {row["attribute"]: row for row in rows}
        big = by_value["big"]["predicted_mask_bbox_longest_side_median"]
        small = by_value["small"]["predicted_mask_bbox_longest_side_median"]
        ratio_pass = (
            isinstance(big, (int, float))
            and isinstance(small, (int, float))
            and float(small) > 0.0
            and float(big) / float(small) >= 1.5
        )
        for value, row in by_value.items():
            row["deterministic_label"] = value if ratio_pass else None
            row["predicted_size_ratio_pass"] = ratio_pass


def _tuple_attribute_raw_rows(
    context: dict[str, Any],
    model: Any,
    transform: Any,
    text: Any,
    slices: dict[str, tuple[int, int, list[str]]],
) -> list[dict[str, Any]]:
    import numpy as np

    tracks = _tuple_grounding_sampled_tracks(context)
    source_by_ordinal = {
        int(row["fixture_ordinal"]): row
        for row in context["rows"]["referent_attribute"]
    }
    if len(source_by_ordinal) != len(context["rows"]["referent_attribute"]):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_FIXTURE_ORDINAL")
    thresholds = (
        context.get("_selected_thresholds", {})
        if context["partition"] == "development"
        else context["thresholds"]
    )
    if not {
        "Grounding_DINO_box_score",
        "Grounding_DINO_text_score",
    } <= set(thresholds):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_THRESHOLD")
    cache = context.get("module_cache")
    if not isinstance(cache, dict) or not isinstance(
        cache.get("referent_adapter_events"), dict
    ):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_ADAPTER_EVENTS_MISSING")
    adapter_events = cache["referent_adapter_events"]
    output = []
    seen: set[int] = set()
    for track in tracks:
        if not isinstance(track, dict) or not isinstance(
            track.get("fixture_ordinal"), int
        ):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_SCHEMA")
        ordinal = int(track["fixture_ordinal"])
        if ordinal in seen or ordinal not in source_by_ordinal:
            raise RuntimeError("E_TUPLE_ATTRIBUTE_FIXTURE_ORDINAL")
        seen.add(ordinal)
        row = source_by_ordinal[ordinal]
        truth = _tuple_attribute_truth(row)
        expected_grounding_truth = {
            key: truth[key]
            for key in (
                "visibility_by_phase",
                "dominance_by_phase",
                "candidate_count_by_phase",
            )
        }
        if (
            track.get("category") != row["category"]
            or track.get("scenario") != row["scenario"]
            or track.get("truth") != expected_grounding_truth
        ):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_ROUNDTRIP")
        adapter_event = adapter_events.get(ordinal)
        language_span_accepted = _tuple_attribute_language_accepts(
            row, adapter_event
        )
        samples = track.get("samples")
        if not isinstance(samples, list) or (
            adapter_event["status"] == "ACCEPT" and len(samples) < 8
        ) or (adapter_event["status"] == "ABSTAIN" and samples):
            raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_SCHEMA")
        masked_images = []
        measurements = []
        invalid_mask = False
        for sample in samples:
            if (
                not isinstance(sample, dict)
                or sample.get("phase") not in {"before", "during", "after"}
                or "image" not in sample
            ):
                raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_SCHEMA")
            candidate = _tuple_attribute_target_candidate(
                sample, str(row["category"]), thresholds
            )
            if candidate is None:
                continue
            try:
                masked, measured = _tuple_attribute_masked_image(
                    sample["image"], candidate.get("mask")
                )
            except RuntimeError as error:
                if str(error).startswith("E_TUPLE_ATTRIBUTE_PREDICTED_MASK_"):
                    invalid_mask = True
                    continue
                raise
            masked_images.append(masked)
            measurements.append(measured)
        mask_valid = bool(masked_images) and not invalid_mask
        pe_label = None
        pe_margin = None
        if mask_valid:
            features, _labels, _margins = _vision_batch(
                model,
                transform,
                text,
                slices,
                masked_images,
                None,
                context["device"],
            )
            family = str(truth["attribute_family"])
            start, stop, names = slices[family]
            scores = np.median(
                (features @ text[start:stop].T).numpy(), axis=0
            )
            if scores.shape != (len(names),) or not np.isfinite(scores).all():
                raise RuntimeError("E_TUPLE_ATTRIBUTE_PE_OUTPUT")
            order = np.argsort(scores)[::-1]
            pe_label = names[int(order[0])]
            pe_margin = float(scores[int(order[0])] - scores[int(order[1])])
        deterministic_label = None
        if mask_valid and truth["attribute_family"] == "color":
            median_rgb = np.median(
                np.asarray(
                    [measurement["median_rgb"] for measurement in measurements],
                    dtype=np.float64,
                ),
                axis=0,
            )
            deterministic_label = _tuple_attribute_palette_label(median_rgb)
        output.append(
            {
                "fixture_ordinal": ordinal,
                "category": row["category"],
                "scenario": row["scenario"],
                "attribute": truth["attribute"],
                "attribute_family": truth["attribute_family"],
                "contrast_expected": truth["attribute_contrast_expected"],
                "mask_measurement_expected": any(
                    truth["visibility_by_phase"].values()
                ),
                "language_span_accepted": language_span_accepted,
                "mask_measurement_valid": mask_valid,
                "invalid_predicted_mask": invalid_mask,
                "selected_sample_count": len(masked_images),
                "pe_label": pe_label,
                "pe_margin": pe_margin,
                "deterministic_label": deterministic_label,
                "predicted_mask_bbox_longest_side_median": (
                    float(
                        statistics.median(
                            measurement["bbox_longest_side_fraction"]
                            for measurement in measurements
                        )
                    )
                    if mask_valid
                    else None
                ),
                "attribute_pair_id": row["attribute_pair_id"],
                "episode_id": row["episode_id"],
                "source_image_id": row["source_image_id"],
                "source_annotation_id": row["source_annotation_id"],
                "target_longest_side_pixels": truth[
                    "target_longest_side_pixels"
                ],
                "background_identity_role": truth["background_identity_role"],
            }
        )
    if seen != set(source_by_ordinal):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_GROUNDING_TRUNCATION")
    _tuple_attribute_apply_size_pairs(output)
    return output


def _tuple_attribute_metrics(
    raw: list[dict[str, Any]], margin: float, adjective_noun_span_f1: float
) -> dict[str, float]:
    labels = [
        *TUPLE_ATTRIBUTE_VALUES["color"],
        *TUPLE_ATTRIBUTE_VALUES["relative_size"],
    ]
    expected_labels = []
    predicted_labels: list[str | None] = []
    null_expected = []
    null_predicted = []
    measurement_expected = []
    measurement_valid = []
    for row in raw:
        prediction = None
        if (
            row["language_span_accepted"] is True
            and row["mask_measurement_valid"] is True
            and isinstance(row["pe_margin"], (int, float))
            and float(row["pe_margin"]) >= margin
            and row["pe_label"] == row["deterministic_label"]
        ):
            prediction = str(row["pe_label"])
        if row["mask_measurement_expected"]:
            measurement_expected.append(True)
            measurement_valid.append(row["mask_measurement_valid"] is True)
        if row["contrast_expected"]:
            expected_labels.append(str(row["attribute"]))
            predicted_labels.append(prediction)
        else:
            null_expected.append(False)
            null_predicted.append(prediction is not None)
    if not expected_labels or not null_expected:
        raise RuntimeError("E_TUPLE_ATTRIBUTE_METRIC_STRATA")
    visible_f1 = _multiclass_macro_f1(expected_labels, predicted_labels, labels)
    coverage = _safe_divide(
        sum(value is not None for value in predicted_labels), len(predicted_labels)
    )
    null_specificity = _binary_classification_metrics(
        null_expected, null_predicted
    )["specificity"]
    mask_fraction = _safe_divide(sum(measurement_valid), len(measurement_expected))
    return {
        "adjective_noun_span_f1": float(adjective_noun_span_f1),
        "eligible_visual_attribute_coverage": coverage,
        "visible_contrast_macro_f1": visible_f1,
        "null_contrast_specificity": null_specificity,
        "valid_mask_measurement_fraction": mask_fraction,
    }


def _tuple_attribute_module(context: dict[str, Any]) -> dict[str, Any]:
    cfg = context["cfg"]
    gate = _tuple_axis(cfg, "adjective_attribute_contrast")["public_gate"]
    if context.get("engineering_health") is True:
        thresholds = _tuple_health_inference_thresholds(cfg)
        context.setdefault("_selected_thresholds", {}).update(
            {
                "Grounding_DINO_box_score": thresholds[
                    "Grounding_DINO_box_score"
                ],
                "Grounding_DINO_text_score": thresholds[
                    "Grounding_DINO_text_score"
                ],
            }
        )
    prompts = _tuple_attribute_prompt_groups(cfg)
    model, transform, text, slices = _load_vision(
        context["public_root"],
        cfg,
        context["device"],
        prompt_groups_override=prompts,
    )
    try:
        raw = _tuple_attribute_raw_rows(context, model, transform, text, slices)
    finally:
        del model
        _release_cuda()
    if context.get("engineering_health") is True:
        if len(raw) != len(context["rows"]["referent_attribute"]):
            raise RuntimeError("E_TUPLE_HEALTH_ATTRIBUTE_TRUNCATION")
        pe_exercised = 0
        for value in raw:
            if (
                not isinstance(value.get("language_span_accepted"), bool)
                or not isinstance(value.get("mask_measurement_valid"), bool)
                or not isinstance(value.get("invalid_predicted_mask"), bool)
                or value.get("invalid_predicted_mask") is True
                or type(value.get("selected_sample_count")) is not int
                or int(value["selected_sample_count"]) < 0
            ):
                raise RuntimeError("E_TUPLE_HEALTH_ATTRIBUTE_OUTPUT_SCHEMA")
            if value["mask_measurement_valid"]:
                pe_exercised += 1
                if (
                    not isinstance(value.get("pe_label"), str)
                    or not isinstance(value.get("pe_margin"), (int, float))
                    or not math.isfinite(float(value["pe_margin"]))
                ):
                    raise RuntimeError("E_TUPLE_HEALTH_ATTRIBUTE_PE_OUTPUT")
        if pe_exercised < 1:
            raise RuntimeError("E_TUPLE_HEALTH_ATTRIBUTE_PE_NOT_EXERCISED")
        production_output = [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "episode_id",
                    "source_image_id",
                    "source_annotation_id",
                }
            }
            for row in raw
        ]
        return _tuple_health_pass_result(
            "attribute", len(raw), production_output
        )
    language_metrics = context.get("_tuple_attribute_language_metrics")
    span_f1 = (
        float(language_metrics["adjective_noun_span_f1"])
        if isinstance(language_metrics, dict)
        and isinstance(language_metrics.get("adjective_noun_span_f1"), (int, float))
        else 0.0
    )

    def evaluate(margin: float) -> dict[str, Any]:
        metrics = _tuple_attribute_metrics(raw, margin, span_f1)
        eligible = (
            metrics["adjective_noun_span_f1"]
            >= float(gate["adjective_noun_span_f1_min"])
            and metrics["eligible_visual_attribute_coverage"]
            >= float(gate["eligible_visual_attribute_coverage_min"])
            and metrics["visible_contrast_macro_f1"]
            >= float(gate["visible_contrast_macro_f1_min"])
            and metrics["null_contrast_specificity"]
            >= float(gate["null_contrast_specificity_min"])
            and metrics["valid_mask_measurement_fraction"]
            >= float(gate["valid_mask_measurement_fraction_required"])
        )
        return {
            "PE_Core_attribute_margin": float(margin),
            **metrics,
            "eligible": eligible,
        }

    if context["partition"] == "development":
        grid = _tuple_amendment(cfg)["public_qualification"][
            "threshold_development"
        ]["PE_Core_attribute_margin_grid"]
        candidates = [evaluate(float(margin)) for margin in grid]
        selected = _select_frozen_grid_result(
            candidates,
            primary_metric="visible_contrast_macro_f1",
            threshold_fields=("PE_Core_attribute_margin",),
        )
        display = selected or max(
            candidates,
            key=lambda row: (
                row["visible_contrast_macro_f1"],
                row["PE_Core_attribute_margin"],
            ),
        )
        threshold = (
            float(selected["PE_Core_attribute_margin"])
            if selected is not None
            else None
        )
    else:
        threshold = float(context["thresholds"]["PE_Core_attribute_margin"])
        display = evaluate(threshold)
        selected = display if display["eligible"] else None
    passed = threshold is not None and selected is not None
    metrics = {key: value for key, value in display.items() if key not in {
        "eligible",
        "PE_Core_attribute_margin",
    }}
    compact_rows = [
        {
            key: row[key]
            for key in (
                "fixture_ordinal",
                "attribute",
                "attribute_family",
                "contrast_expected",
                "mask_measurement_expected",
                "language_span_accepted",
                "mask_measurement_valid",
                "invalid_predicted_mask",
                "selected_sample_count",
                "pe_label",
                "pe_margin",
                "deterministic_label",
            )
        }
        for row in raw
    ]
    return {
        "status": "PASS" if passed else "NO_GO",
        "axis_results": {
            "adjective_attribute_contrast": {
                "status": "PASS" if passed else "NO_GO",
                "metrics": metrics,
            }
        },
        "metrics": metrics,
        "selected_thresholds": (
            {"PE_Core_attribute_margin": threshold}
            if threshold is not None
            else {}
        ),
        "rows": compact_rows,
        "row_count": len(raw),
        "failure_count": 0,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
    }


def _refuse_git_output(path: Path) -> None:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            raise RuntimeError("E_TUPLE_FIXTURE_OUTPUT_IN_GIT")


def _require_external_or_ignored_output(path: Path) -> None:
    """Permit public outputs only outside Git or under an ignored root.

    Public qualification normally runs in the external Juno public root.  The
    ignored-root exception keeps local tests and explicitly ignored disposable
    runs possible without allowing a source artifact to land in a tracked
    worktree by accident.  Git worktrees use a `.git` file, so walking only for
    `.git` directories is not sufficient here.
    """

    resolved = path.resolve()
    git_executable = shutil.which("git")
    if git_executable is None:
        # Minimal inference containers need not carry Git.  Walking for either
        # a `.git` directory or worktree pointer still lets us reject a path
        # inside a repository; only a genuinely external path is accepted
        # when `check-ignore` is unavailable.
        _refuse_git_output(resolved)
        return
    probe = resolved
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    completed = subprocess.run(
        [git_executable, "-C", str(probe), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        return
    repository = Path(completed.stdout.strip()).resolve()
    try:
        resolved.relative_to(repository)
    except ValueError:
        return
    ignored = subprocess.run(
        [
            git_executable,
            "-C",
            str(repository),
            "check-ignore",
            "--quiet",
            "--",
            str(resolved),
        ],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ignored.returncode != 0:
        raise RuntimeError("E_TUPLE_FIXTURE_OUTPUT_NOT_EXTERNAL_OR_IGNORED")


TUPLE_MODULE_AXIS_IDS = {
    "adapter_and_lexical": (
        "adapter_qualified_yield",
        "noun_adjective_exposure",
    ),
    "referent": (
        "utterance_centered_referent_visibility_dominance_ambiguity",
    ),
    "recurrence": ("cross_episode_recurrence",),
    "attribute": ("adjective_attribute_contrast",),
    "hand_contact": ("hand_action_coupling",),
    "sensor": ("egocentric_sensor_regime",),
    "order_action": (),
}

TUPLE_MODULE_RUNNER_NAMES = {
    "adapter_and_lexical": "_tuple_language_lexical_module",
    "referent": "_tuple_referent_module",
    "recurrence": "_tuple_recurrence_module",
    "attribute": "_tuple_attribute_module",
    "hand_contact": "_tuple_hand_contact_module",
    "sensor": "_tuple_sensor_module",
    "order_action": "_tuple_order_action_module",
}


def _missing_tuple_module_runner(module_id: str):
    def run(_context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"E_TUPLE_QUALIFICATION_MODULE_UNIMPLEMENTED {module_id}")

    return run


def _development_unqualified_tuple_module_runner(module_id: str):
    def run(_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "UNMEASURED",
            "axis_results": {
                axis_id: {
                    "status": "UNMEASURED",
                    "metrics": {},
                    "reason": "DEVELOPMENT_MODULE_DID_NOT_QUALIFY",
                }
                for axis_id in TUPLE_MODULE_AXIS_IDS[module_id]
            },
            "metrics": {},
            "selected_thresholds": {},
            "rows": [],
            "row_count": 0,
            "failure_count": 0,
            "invalid_retained_record_count": 0,
            "silent_truncation_count": 0,
            "external_call_count": 0,
        }

    return run


def _tuple_module_runners() -> dict[str, Any]:
    """Resolve every preregistered module without manufacturing a result."""

    if set(TUPLE_MODULE_RUNNER_NAMES) != set(TUPLE_QUALIFICATION_MODULE_IDS):
        raise RuntimeError("E_TUPLE_QUALIFICATION_MODULE_REGISTRY")
    output = {}
    for module_id in TUPLE_QUALIFICATION_MODULE_IDS:
        value = globals().get(TUPLE_MODULE_RUNNER_NAMES[module_id])
        output[module_id] = (
            value if callable(value) else _missing_tuple_module_runner(module_id)
        )
    return output


def _tuple_health_inference_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    """Validate fixed execution minima without opening scientific selection."""

    observed = _tuple_amendment(cfg)["public_qualification"][
        "threshold_development"
    ]
    expected = {
        "Grounding_DINO_box_score_grid": [0.2, 0.25, 0.3, 0.35, 0.4],
        "Grounding_DINO_text_score_grid": [0.15, 0.2, 0.25, 0.3],
        "EgoHOS_min_mask_fraction_grid": [0.0025, 0.005, 0.01, 0.02],
        "PE_Core_attribute_margin_grid": [0.0, 0.01, 0.02, 0.05, 0.1],
        "DINOv2_recurrence_cosine_grid": [0.8, 0.85, 0.9, 0.95],
    }
    if any(observed.get(key) != values for key, values in expected.items()):
        raise RuntimeError("E_TUPLE_HEALTH_EXECUTION_THRESHOLD_COMMITMENT")
    return {
        "Grounding_DINO_box_score": 0.2,
        "Grounding_DINO_text_score": 0.15,
        "EgoHOS_min_mask_fraction": 0.0025,
    }


def _tuple_frozen_threshold_grids(cfg: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    development = _tuple_amendment(cfg)["public_qualification"][
        "threshold_development"
    ]
    action = _tuple_fixture_protocol(cfg)["order_dependent_action_control"]
    grids = {
        "Grounding_DINO_box_score": development[
            "Grounding_DINO_box_score_grid"
        ],
        "Grounding_DINO_text_score": development[
            "Grounding_DINO_text_score_grid"
        ],
        "EgoHOS_min_mask_fraction": development[
            "EgoHOS_min_mask_fraction_grid"
        ],
        "PE_Core_attribute_margin": development[
            "PE_Core_attribute_margin_grid"
        ],
        "DINOv2_recurrence_cosine": development[
            "DINOv2_recurrence_cosine_grid"
        ],
        "action_abstention_margin": action[
            "development_abstention_margin_grid"
        ],
    }
    output = {}
    for key, values in grids.items():
        if not isinstance(values, list) or not values:
            raise RuntimeError("E_TUPLE_QUALIFICATION_THRESHOLD_GRID")
        numeric = tuple(float(value) for value in values)
        if len(set(numeric)) != len(numeric) or not all(
            math.isfinite(value) for value in numeric
        ):
            raise RuntimeError("E_TUPLE_QUALIFICATION_THRESHOLD_GRID")
        output[key] = numeric
    return output


def _tuple_selected_thresholds(
    cfg: dict[str, Any],
    module_results: dict[str, dict[str, Any]],
    *,
    expected: dict[str, float] | None = None,
) -> dict[str, float]:
    """Merge module thresholds and prove that no value escaped its frozen grid."""

    grids = _tuple_frozen_threshold_grids(cfg)
    output: dict[str, float] = {}
    for module_id in TUPLE_QUALIFICATION_MODULE_IDS:
        selected = module_results[module_id].get("selected_thresholds", {})
        if not isinstance(selected, dict):
            raise RuntimeError("E_TUPLE_QUALIFICATION_THRESHOLD_RECORD")
        for key, raw_value in selected.items():
            if key not in grids or not isinstance(raw_value, (int, float)):
                raise RuntimeError("E_TUPLE_QUALIFICATION_THRESHOLD_NOT_FROZEN")
            value = float(raw_value)
            if not math.isfinite(value) or value not in grids[key]:
                raise RuntimeError("E_TUPLE_QUALIFICATION_THRESHOLD_NOT_FROZEN")
            if key in output and output[key] != value:
                raise RuntimeError("E_TUPLE_QUALIFICATION_THRESHOLD_CONFLICT")
            output[key] = value
    if expected is not None:
        normalized = {key: float(value) for key, value in expected.items()}
        if any(key not in grids or value not in grids[key] for key, value in normalized.items()):
            raise RuntimeError("E_TUPLE_QUALIFICATION_THRESHOLD_SEAL_GRID")
        if any(key not in normalized or normalized[key] != value for key, value in output.items()):
            raise RuntimeError("E_TUPLE_HOLDOUT_THRESHOLD_CHANGED")
    return dict(sorted(output.items()))


def _tuple_axis_results_from_modules(
    module_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Register every axis even when its module errors or abstains."""

    output: dict[str, dict[str, Any]] = {}
    for module_id in TUPLE_QUALIFICATION_MODULE_IDS:
        result = module_results[module_id]
        expected_axes = TUPLE_MODULE_AXIS_IDS[module_id]
        supplied = result.get("axis_results", {})
        if supplied is None:
            supplied = {}
        if not isinstance(supplied, dict) or set(supplied) - set(expected_axes):
            raise RuntimeError("E_TUPLE_QUALIFICATION_AXIS_RESULT_SET")
        for axis_id in expected_axes:
            axis = supplied.get(axis_id)
            if axis is None:
                status = (
                    "ERROR" if result.get("status") == "ERROR" else "UNMEASURED"
                )
                axis = {"status": status, "metrics": {}}
            if not isinstance(axis, dict) or axis.get("status") not in {
                "PASS",
                "NO_GO",
                "UNMEASURED",
                "ERROR",
            }:
                raise RuntimeError("E_TUPLE_QUALIFICATION_AXIS_RESULT")
            output[axis_id] = axis
    if set(output) != set((*TUPLE_CRITICAL_AXIS_IDS, *TUPLE_SUPPORTING_AXIS_IDS)):
        raise RuntimeError("E_TUPLE_QUALIFICATION_AXIS_RESULT_SET")
    return output


def _tuple_qualification_integrity(
    module_results: dict[str, dict[str, Any]],
) -> dict[str, int]:
    fields = (
        "failure_count",
        "invalid_retained_record_count",
        "silent_truncation_count",
        "external_call_count",
    )
    output = {
        key: sum(int(result.get(key, 0)) for result in module_results.values())
        for key in fields
    }
    output["error_module_count"] = sum(
        result.get("status") == "ERROR" for result in module_results.values()
    )
    output["unaccounted_failure_count"] = sum(
        result.get("error_code") == "E_UNACCOUNTED_MODULE_FAILURE"
        for result in module_results.values()
    )
    if any(value < 0 for value in output.values()):
        raise RuntimeError("E_TUPLE_QUALIFICATION_NEGATIVE_INTEGRITY_COUNT")
    return output


def _validate_tuple_qualification_record(value: Any) -> None:
    """Reject non-finite or non-JSON module outputs before any result is sealed."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("E_TUPLE_QUALIFICATION_NONFINITE_RESULT")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_tuple_qualification_record(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise RuntimeError("E_TUPLE_QUALIFICATION_NONJSON_RESULT")
        for item in value.values():
            _validate_tuple_qualification_record(item)
        return
    raise RuntimeError("E_TUPLE_QUALIFICATION_NONJSON_RESULT")


def _apply_tuple_integrity_gate(
    combined: dict[str, Any], integrity: dict[str, int]
) -> dict[str, Any]:
    output = json.loads(json.dumps(combined))
    failures = list(output["combined_gate_failures"])
    for key in (
        "failure_count",
        "invalid_retained_record_count",
        "silent_truncation_count",
        "external_call_count",
        "error_module_count",
        "unaccounted_failure_count",
    ):
        if integrity[key]:
            failures.append(f"integrity:{key}")
    output["integrity"] = integrity
    output["combined_gate_failures"] = failures
    output["status"] = "PASS" if not failures else "NO_GO"
    return output


def _tuple_qualification_root(public: Path) -> Path:
    return (
        _tuple_run_root(public)
        / "construct-aligned-engineering-health"
        / "qualification"
    )


def _tuple_legacy_qualification_paths(public: Path) -> dict[str, Path]:
    """Historical job-315542 paths are read-only under the new route."""

    root = _tuple_run_root(public) / "qualification"
    return {
        "development_result": root / "development-result.json",
        "development_threshold_seal": root / "development-threshold-seal.json",
        "holdout_result": root / "holdout-result.json",
    }


def _tuple_qualification_paths(public: Path) -> dict[str, Path]:
    root = _tuple_qualification_root(public)
    return {
        "development_result": root / "development-result.json",
        "development_threshold_seal": root / "development-threshold-seal.json",
        "holdout_result": root / "holdout-result.json",
    }


def _tuple_qualification_transaction_path(public: Path, partition: str) -> Path:
    if partition not in {"development", "holdout"}:
        raise RuntimeError("E_TUPLE_QUALIFICATION_PARTITION")
    return _tuple_qualification_root(public) / "transactions" / f"{partition}.json"


def _tuple_qualification_transaction(
    partition: str, full: dict[str, Any], seal: dict[str, Any] | None
) -> dict[str, Any]:
    if partition == "development" and not isinstance(seal, dict):
        raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
    if partition == "holdout" and seal is not None:
        raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
    value = {
        "schema_version": 1,
        "status": "SEALED_SCIENTIFIC_OUTCOME_TRANSACTION",
        "partition": partition,
        "scientific_result": full,
        "development_threshold_seal": seal,
    }
    value["transaction_commitment_sha256"] = digest(value)
    return value


def _validate_tuple_qualification_transaction(
    value: Any, partition: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "status",
            "partition",
            "scientific_result",
            "development_threshold_seal",
            "transaction_commitment_sha256",
        }
        or value.get("schema_version") != 1
        or value.get("status") != "SEALED_SCIENTIFIC_OUTCOME_TRANSACTION"
        or value.get("partition") != partition
    ):
        raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
    expected = value["transaction_commitment_sha256"]
    if not isinstance(expected, str) or digest(
        {
            key: item
            for key, item in value.items()
            if key != "transaction_commitment_sha256"
        }
    ) != expected:
        raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
    full = value.get("scientific_result")
    seal = value.get("development_threshold_seal")
    if not isinstance(full, dict) or full.get("partition") != partition:
        raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
    result_expected = full.get("public_qualification_commitment_sha256")
    if not isinstance(result_expected, str) or digest(
        {
            key: item
            for key, item in full.items()
            if key != "public_qualification_commitment_sha256"
        }
    ) != result_expected:
        raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
    if partition == "development":
        if not isinstance(seal, dict):
            raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
        seal_expected = seal.get("development_threshold_commitment_sha256")
        if not isinstance(seal_expected, str) or digest(
            {
                key: item
                for key, item in seal.items()
                if key != "development_threshold_commitment_sha256"
            }
        ) != seal_expected:
            raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
    elif seal is not None:
        raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION")
    return full, seal


def _recover_tuple_qualification_transaction(
    public: Path,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    partition: str,
) -> dict[str, Any] | None:
    transaction_path = _tuple_qualification_transaction_path(public, partition)
    if not transaction_path.is_file():
        return None
    full, seal = _validate_tuple_qualification_transaction(
        json.loads(transaction_path.read_text()), partition
    )
    paths = _tuple_qualification_paths(public)
    targets = (
        [paths["development_result"], paths["development_threshold_seal"]]
        if partition == "development"
        else [paths["holdout_result"]]
    )
    if all(path.is_file() for path in targets):
        return None
    result_path = (
        paths["development_result"]
        if partition == "development"
        else paths["holdout_result"]
    )
    if result_path.is_file():
        if json.loads(result_path.read_text()) != full:
            raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION_CONFLICT")
    else:
        write_private_new(result_path, full)
    if partition == "development":
        assert seal is not None
        seal_path = paths["development_threshold_seal"]
        if seal_path.is_file():
            if json.loads(seal_path.read_text()) != seal:
                raise RuntimeError("E_TUPLE_QUALIFICATION_TRANSACTION_CONFLICT")
        else:
            write_private_new(seal_path, seal)
        loaded_seal, development = _load_tuple_development_pair(
            public,
            cfg,
            manifest,
            missing_code="E_TUPLE_DEVELOPMENT_PAIR_MISSING",
        )
        return _tuple_qualification_compact(
            development,
            loaded_seal["development_threshold_commitment_sha256"],
        )
    return _tuple_qualification_compact(
        full, full["development_threshold_commitment_sha256"]
    )


def _refuse_tuple_qualification_overwrite(public: Path, partition: str) -> None:
    paths = _tuple_qualification_paths(public)
    if partition == "development":
        targets = (
            paths["development_result"],
            paths["development_threshold_seal"],
            paths["holdout_result"],
        )
        code = "E_TUPLE_DEVELOPMENT_RESULT_ALREADY_EXISTS"
    elif partition == "holdout":
        targets = (paths["holdout_result"],)
        code = "E_TUPLE_HOLDOUT_RESULT_ALREADY_EXISTS"
    else:
        raise RuntimeError("E_TUPLE_QUALIFICATION_PARTITION")
    if any(path.exists() for path in targets):
        raise RuntimeError(code)


def _tuple_development_module_commitment(
    fixture_commitment: str,
    module_results: dict[str, dict[str, Any]],
    axis_results: dict[str, dict[str, Any]],
    combined_gate: dict[str, Any],
) -> str:
    return digest(
        {
            "fixture_manifest_commitment_sha256": fixture_commitment,
            "module_results": module_results,
            "axis_results": axis_results,
            "combined_gate": combined_gate,
        }
    )


def _tuple_development_threshold_seal(
    cfg: dict[str, Any],
    fixture_manifest: dict[str, Any],
    module_results: dict[str, dict[str, Any]],
    axis_results: dict[str, dict[str, Any]],
    combined_gate: dict[str, Any],
) -> dict[str, Any]:
    active = _construct_aligned_ltx_resume_amendment(cfg)
    thresholds = _tuple_selected_thresholds(cfg, module_results)
    required_by_module = {
        "referent": {
            "Grounding_DINO_box_score",
            "Grounding_DINO_text_score",
        },
        "recurrence": {"DINOv2_recurrence_cosine"},
        "attribute": {"PE_Core_attribute_margin"},
        "hand_contact": {"EgoHOS_min_mask_fraction"},
        "order_action": {"action_abstention_margin"},
    }
    for module_id, required in required_by_module.items():
        status = module_results[module_id].get("status")
        selected = module_results[module_id].get("selected_thresholds", {})
        if status == "PASS" or (
            module_id == "order_action" and status == "NO_GO_DIAGNOSTIC"
        ) or (
            module_id == "recurrence" and status == "NO_GO"
        ):
            if not required <= set(selected):
                raise RuntimeError("E_TUPLE_QUALIFICATION_PASS_WITHOUT_THRESHOLD")
        elif selected:
            raise RuntimeError("E_TUPLE_QUALIFICATION_FAILED_MODULE_THRESHOLD")
    module_commitment = _tuple_development_module_commitment(
        fixture_manifest["public_fixture_manifest_commitment_sha256"],
        module_results,
        axis_results,
        combined_gate,
    )
    seal = {
        "schema_version": 1,
        "status": (
            "PASS_DEVELOPMENT_THRESHOLDS_SEALED"
            if combined_gate["status"] == "PASS"
            else "NO_GO_DEVELOPMENT_COMBINED_GATE"
        ),
        "partition": "development",
        "holdout_authorized": combined_gate["status"] == "PASS",
        "public_fixture_manifest_commitment_sha256": fixture_manifest[
            "public_fixture_manifest_commitment_sha256"
        ],
        "visor_hos_correction_amendment_commitment_sha256": fixture_manifest[
            "visor_hos_correction_amendment_commitment_sha256"
        ],
        "verified_no_hand_seal_commitment_sha256": fixture_manifest[
            "verified_no_hand_seal_commitment_sha256"
        ],
        "construct_aligned_ltx_resume_amendment_commitment_sha256": active[
            "amendment_commitment_sha256"
        ],
        "development_module_result_commitment_sha256": module_commitment,
        "development_module_statuses": {
            module_id: module_results[module_id]["status"]
            for module_id in TUPLE_QUALIFICATION_MODULE_IDS
        },
        "selected_thresholds": thresholds,
        "combined_gate_status": combined_gate["status"],
        "holdout_reprompt_refit_or_threshold_change": "PROHIBITED",
    }
    seal["development_threshold_commitment_sha256"] = digest(seal)
    return seal


def _load_tuple_development_pair(
    public: Path,
    cfg: dict[str, Any],
    fixture_manifest: dict[str, Any],
    *,
    missing_code: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _tuple_qualification_paths(public)
    result_present = paths["development_result"].is_file()
    seal_present = paths["development_threshold_seal"].is_file()
    if result_present != seal_present:
        raise RuntimeError("E_TUPLE_DEVELOPMENT_PAIR_PARTIAL")
    if not result_present:
        raise RuntimeError(missing_code)
    seal = json.loads(paths["development_threshold_seal"].read_text())
    payload = json.loads(json.dumps(seal))
    expected = payload.pop("development_threshold_commitment_sha256", None)
    if not isinstance(expected, str) or digest(payload) != expected:
        raise RuntimeError("E_TUPLE_DEVELOPMENT_THRESHOLD_COMMITMENT")
    combined_status = seal.get("combined_gate_status")
    expected_status = {
        "PASS": "PASS_DEVELOPMENT_THRESHOLDS_SEALED",
        "NO_GO": "NO_GO_DEVELOPMENT_COMBINED_GATE",
    }.get(combined_status)
    if (
        expected_status is None
        or seal.get("status") != expected_status
        or seal.get("holdout_authorized") is not (combined_status == "PASS")
        or seal.get("partition") != "development"
        or seal.get("holdout_reprompt_refit_or_threshold_change") != "PROHIBITED"
    ):
        raise RuntimeError("E_TUPLE_DEVELOPMENT_PAIR_STATUS")
    for key in (
        "public_fixture_manifest_commitment_sha256",
        "visor_hos_correction_amendment_commitment_sha256",
        "verified_no_hand_seal_commitment_sha256",
        "construct_aligned_ltx_resume_amendment_commitment_sha256",
    ):
        if seal.get(key) != fixture_manifest.get(key):
            raise RuntimeError("E_TUPLE_DEVELOPMENT_THRESHOLD_PROVENANCE")
    thresholds = seal.get("selected_thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("E_TUPLE_DEVELOPMENT_THRESHOLD_RECORD")
    statuses = seal.get("development_module_statuses")
    if (
        not isinstance(statuses, dict)
        or set(statuses) != set(TUPLE_QUALIFICATION_MODULE_IDS)
        or any(
            status
            not in {
                "PASS",
                "NO_GO",
                "NO_GO_DIAGNOSTIC",
                "UNMEASURED",
                "ERROR",
            }
            for status in statuses.values()
        )
    ):
        raise RuntimeError("E_TUPLE_DEVELOPMENT_MODULE_STATUS_SEAL")
    development = json.loads(paths["development_result"].read_text())
    result_payload = json.loads(json.dumps(development))
    result_expected = result_payload.pop("public_qualification_commitment_sha256", None)
    if not isinstance(result_expected, str) or digest(result_payload) != result_expected:
        raise RuntimeError("E_TUPLE_DEVELOPMENT_RESULT_COMMITMENT")
    development_module_results = development.get("module_results", {})
    development_statuses = {
        module_id: development_module_results.get(module_id, {}).get("status")
        for module_id in TUPLE_QUALIFICATION_MODULE_IDS
    }
    if (
        development.get("schema_version") != 1
        or development.get("status") != expected_status
        or development.get("partition") != "development"
        or development.get("public_fixture_manifest_commitment_sha256")
        != fixture_manifest.get("public_fixture_manifest_commitment_sha256")
        or development.get("visor_hos_correction_amendment_commitment_sha256")
        != fixture_manifest.get("visor_hos_correction_amendment_commitment_sha256")
        or development.get("verified_no_hand_seal_commitment_sha256")
        != fixture_manifest.get("verified_no_hand_seal_commitment_sha256")
        or development.get(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        )
        != fixture_manifest.get(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        )
        or development.get("development_threshold_commitment_sha256") != expected
        or development.get("development_module_result_commitment_sha256")
        != seal.get("development_module_result_commitment_sha256")
        or development.get("selected_thresholds") != thresholds
        or development.get("combined_gate", {}).get("status") != combined_status
        or development_statuses != statuses
    ):
        raise RuntimeError("E_TUPLE_DEVELOPMENT_RESULT_PROVENANCE")
    if set(development_module_results) != set(TUPLE_QUALIFICATION_MODULE_IDS):
        raise RuntimeError("E_TUPLE_DEVELOPMENT_MODULE_RESULT_SET")
    if _tuple_selected_thresholds(cfg, development_module_results) != thresholds:
        raise RuntimeError("E_TUPLE_DEVELOPMENT_THRESHOLD_PROVENANCE")
    recomputed_module_commitment = _tuple_development_module_commitment(
        fixture_manifest["public_fixture_manifest_commitment_sha256"],
        development.get("module_results", {}),
        development.get("axis_results", {}),
        development.get("combined_gate", {}),
    )
    if recomputed_module_commitment != seal.get(
        "development_module_result_commitment_sha256"
    ):
        raise RuntimeError("E_TUPLE_DEVELOPMENT_MODULE_COMMITMENT")
    return seal, development


def _load_tuple_development_threshold_seal(
    public: Path,
    cfg: dict[str, Any],
    fixture_manifest: dict[str, Any],
) -> dict[str, Any]:
    seal, _ = _load_tuple_development_pair(
        public,
        cfg,
        fixture_manifest,
        missing_code="E_TUPLE_HOLDOUT_BEFORE_DEVELOPMENT_SEAL",
    )
    if seal["holdout_authorized"] is not True:
        raise RuntimeError("E_TUPLE_HOLDOUT_NOT_AUTHORIZED")
    return seal


def _reuse_tuple_development_pair(
    public: Path,
    cfg: dict[str, Any],
    fixture_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    paths = _tuple_qualification_paths(public)
    present = (
        paths["development_result"].is_file(),
        paths["development_threshold_seal"].is_file(),
    )
    if present == (False, False):
        if paths["holdout_result"].exists():
            raise RuntimeError("E_TUPLE_DEVELOPMENT_PAIR_MISSING_AFTER_HOLDOUT")
        return None
    seal, development = _load_tuple_development_pair(
        public,
        cfg,
        fixture_manifest,
        missing_code="E_TUPLE_DEVELOPMENT_PAIR_MISSING",
    )
    return _tuple_qualification_compact(
        development, seal["development_threshold_commitment_sha256"]
    )


def _tuple_qualification_compact(
    full: dict[str, Any], threshold_commitment: str
) -> dict[str, Any]:
    modules = full["module_results"]
    combined = full["combined_gate"]
    integrity = combined["integrity"]
    return {
        "status": full["status"],
        "partition": str(full["partition"]).upper(),
        "module_count": len(modules),
        "completed_module_count": sum(
            result.get("status")
            in {"PASS", "NO_GO", "NO_GO_DIAGNOSTIC", "UNMEASURED"}
            for result in modules.values()
        ),
        "failed_module_count": int(integrity["error_module_count"]),
        "critical_axis_pass_count": int(combined["critical_axis_pass_count"]),
        "validated_axis_count": int(combined["validated_axis_count"]),
        "action_control_status": str(combined["action_control_status"]),
        "external_call_count": int(integrity["external_call_count"]),
        "invalid_retained_record_count": int(
            integrity["invalid_retained_record_count"]
        ),
        "silent_truncation_count": int(integrity["silent_truncation_count"]),
        "public_qualification_commitment_sha256": full[
            "public_qualification_commitment_sha256"
        ],
        "development_threshold_commitment_sha256": threshold_commitment,
    }


def _tuple_health_root(public: Path) -> Path:
    return _tuple_run_root(public) / "construct-aligned-engineering-health"


def _tuple_health_tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _tuple_health_verify_file(
    path: Path, expected_sha256: str, expected_bytes: int | None = None
) -> dict[str, Any]:
    if (
        not path.is_file()
        or file_digest(path) != expected_sha256
        or (
            expected_bytes is not None
            and path.stat().st_size != int(expected_bytes)
        )
    ):
        raise RuntimeError("E_TUPLE_HEALTH_ARTIFACT_COMMITMENT")
    return {
        "sha256": expected_sha256,
        "bytes": path.stat().st_size,
    }


def _tuple_health_tree_commitment(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE")
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if (
            ".git" in relative.parts
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
        ):
            continue
        records.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
        )
    if not records:
        raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE")
    return {
        "file_count": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "tree_commitment_sha256": digest(records),
    }


def _tuple_health_verify_archive_tree(
    archive: Path,
    root: Path,
    *,
    replacements: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Bind every importable extracted source byte to its frozen archive."""

    replacements = replacements or {}
    expected: dict[str, tuple[str, int]] = {}
    with zipfile.ZipFile(archive) as source:
        files = [member for member in source.infolist() if not member.is_dir()]
        roots = {
            Path(member.filename).parts[0]
            for member in files
            if Path(member.filename).parts
        }
        if len(roots) != 1:
            raise RuntimeError("E_TUPLE_HEALTH_CODE_ARCHIVE_ROOT")
        archive_root = next(iter(roots))
        for member in files:
            parts = Path(member.filename).parts
            if not parts or parts[0] != archive_root or len(parts) < 2:
                raise RuntimeError("E_TUPLE_HEALTH_CODE_ARCHIVE_ROOT")
            relative = Path(*parts[1:]).as_posix()
            hasher = hashlib.sha256()
            with source.open(member) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            archive_sha = hasher.hexdigest()
            replacement = replacements.get(relative)
            if replacement is not None and archive_sha != replacement[0]:
                raise RuntimeError("E_TUPLE_HEALTH_CODE_ARCHIVE_SOURCE")
            expected[relative] = (
                replacement[1] if replacement is not None else archive_sha,
                int(member.file_size),
            )
    observed_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name != ".source-commit"
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pyc"
    }
    if set(observed_paths) != set(expected):
        raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE_SET")
    records = []
    for relative in sorted(expected):
        path = observed_paths[relative]
        expected_sha, archive_bytes = expected[relative]
        observed_sha = file_digest(path)
        if observed_sha != expected_sha:
            raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE_COMMITMENT")
        records.append(
            {
                "relative_path": relative,
                "sha256": observed_sha,
                "bytes": path.stat().st_size,
                "archive_bytes": archive_bytes,
            }
        )
    return {
        "file_count": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "tree_commitment_sha256": digest(records),
    }


def _tuple_health_verify_git_tree(
    root: Path,
    expected_commit: str,
    attestation: dict[str, Any] | None = None,
) -> None:
    """Verify one exact clean repository with or without a container Git binary."""

    expected_fields = {
        "family",
        "expected_commit",
        "git_index_sha256",
        "git_index_bytes",
        "file_count",
        "bytes",
        "tree_commitment_sha256",
        "host_unexpected_status_count",
    }
    _verify_repository_commit(root, expected_commit)
    git = shutil.which("git")
    if attestation is None:
        if git is None:
            raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE_ATTESTATION")
        completed = subprocess.run(
            [
                git,
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        unexpected = [
            line
            for line in completed.stdout.splitlines()
            if "__pycache__/" not in line[3:] and not line[3:].endswith(".pyc")
        ]
        if unexpected:
            raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE_DIRTY")
        return
    if (
        not isinstance(attestation, dict)
        or set(attestation) != expected_fields
        or attestation.get("expected_commit") != expected_commit
        or attestation.get("host_unexpected_status_count") != 0
    ):
        raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE_ATTESTATION")
    git_index = root / ".git/index"
    tree = _tuple_health_tree_commitment(root)
    unexpected_symlink = any(
        path.is_symlink()
        and ".git" not in path.relative_to(root).parts
        and "__pycache__" not in path.relative_to(root).parts
        for path in root.rglob("*")
    )
    if (
        not git_index.is_file()
        or git_index.is_symlink()
        or unexpected_symlink
        or file_digest(git_index) != attestation["git_index_sha256"]
        or git_index.stat().st_size != int(attestation["git_index_bytes"])
        or tree
        != {
            "file_count": attestation["file_count"],
            "bytes": attestation["bytes"],
            "tree_commitment_sha256": attestation[
                "tree_commitment_sha256"
            ],
        }
    ):
        raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE_ATTESTATION")
    if git is not None:
        completed = subprocess.run(
            [
                git,
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        unexpected = []
        for line in completed.stdout.splitlines():
            path = line[3:]
            if "__pycache__/" in path or path.endswith(".pyc"):
                continue
            unexpected.append(line)
        if unexpected:
            raise RuntimeError("E_TUPLE_HEALTH_CODE_TREE_DIRTY")


def _tuple_health_configuration_preflight(cfg: dict[str, Any]) -> str:
    """Validate every module-facing frozen field before any model is loaded."""

    active = _construct_aligned_ltx_resume_amendment(cfg)
    health = _engineering_health_amendment(cfg)
    dependency_restore = _engineering_health_dependency_restore(cfg)
    topology_guard_repair = _engineering_health_topology_guard_repair(cfg)
    git_fallback_repair = (
        _engineering_health_git_fallback_repair(cfg)
        if "learner_effective_engineering_health_git_fallback_repair" in cfg
        else None
    )
    fixture_bind_repair = (
        _engineering_health_fixture_bind_repair(cfg)
        if "learner_effective_engineering_health_fixture_bind_repair" in cfg
        else None
    )
    submission_export_repair = (
        _engineering_health_submission_export_repair(cfg)
        if "learner_effective_engineering_health_submission_export_repair" in cfg
        else None
    )
    amendment = _tuple_amendment(cfg)
    runtime = _tuple_runtime_amendment(cfg)
    fixture_protocol = _tuple_fixture_protocol(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    correction = _tuple_visor_hos_correction_amendment(cfg)
    execution = _tuple_qualification_execution(cfg)
    audio = _tuple_referent_audio_fixture(cfg)
    thresholds = _tuple_health_inference_thresholds(cfg)
    ontology = _tuple_public_ontology_mapping(preparation)
    prompts = _tuple_attribute_prompt_groups(cfg)
    action_candidate, action_runtime = _tuple_order_action_egohod_wiring(cfg)
    axes = {
        axis_id: _tuple_axis(cfg, axis_id)
        for axis_id in (*TUPLE_CRITICAL_AXIS_IDS, *TUPLE_SUPPORTING_AXIS_IDS)
    }
    bins = cfg["calibration_C"]["extractor"].get("fixed_numeric_bins")
    required_bins = {
        "brightness",
        "blur_edge_strength",
        "motion_mean_absolute_luma",
    }
    if not isinstance(bins, dict) or not required_bins <= set(bins):
        raise RuntimeError("E_TUPLE_HEALTH_SENSOR_BIN_CONFIG")
    for key in required_bins:
        values = bins[key]
        if (
            not isinstance(values, list)
            or len(values) < 2
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in values
            )
            or any(
                float(values[index]) >= float(values[index + 1])
                for index in range(len(values) - 1)
            )
        ):
            raise RuntimeError("E_TUPLE_HEALTH_SENSOR_BIN_CONFIG")
    if set(_tuple_module_runners()) != set(TUPLE_QUALIFICATION_MODULE_IDS):
        raise RuntimeError("E_TUPLE_QUALIFICATION_MODULE_REGISTRY")
    adapter = Path(__file__).resolve().with_name(
        "synthetic_video_language_adapter.py"
    )
    if file_digest(adapter) != TUPLE_LANGUAGE_ADAPTER_SHA256:
        raise RuntimeError("E_TUPLE_LANGUAGE_ADAPTER_SOURCE")
    return digest(
        {
            "active": active,
            "health": health,
            "dependency_restore": dependency_restore,
            "topology_guard_repair": topology_guard_repair,
            "git_fallback_repair": git_fallback_repair,
            "fixture_bind_repair": fixture_bind_repair,
            "submission_export_repair": submission_export_repair,
            "tuple_amendment": amendment,
            "runtime": runtime,
            "fixture_protocol": fixture_protocol,
            "preparation": preparation,
            "correction": correction,
            "execution": execution,
            "audio": audio,
            "thresholds": thresholds,
            "ontology": ontology,
            "attribute_prompts": prompts,
            "action_candidate": action_candidate,
            "action_runtime": action_runtime,
            "axes": axes,
            "sensor_bins": {key: bins[key] for key in sorted(required_bins)},
            "module_ids": list(TUPLE_QUALIFICATION_MODULE_IDS),
            "language_adapter_sha256": TUPLE_LANGUAGE_ADAPTER_SHA256,
        }
    )


def _tuple_health_dependency_preflight(
    public: Path,
    cfg: dict[str, Any],
    container_record: dict[str, Any],
    progress: Any | None = None,
) -> dict[str, Any]:
    """Rehash every production dependency family before GPU model loading."""

    def mark(stage: str, ordinal: int) -> None:
        if progress is not None:
            progress(stage, ordinal, 0, 0)

    mark("DEPENDENCY_CONFIGURATION", 1)
    health = _engineering_health_amendment(cfg)
    dependency_restore = _engineering_health_dependency_restore(cfg)
    topology_guard_repair = _engineering_health_topology_guard_repair(cfg)
    configuration_commitment = _tuple_health_configuration_preflight(cfg)
    mark("DEPENDENCY_RUNTIME_MANIFEST", 2)
    runtime_cfg = _tuple_runtime_amendment(cfg)
    runtime = _verify_tuple_runtime_manifest(public, cfg)
    model_root = _tuple_model_root(public)
    records: list[dict[str, Any]] = []
    container = runtime_cfg["base_container"]
    if container_record != {
        "sha256": container["sha256"],
        "bytes": 3731320832,
    }:
        raise RuntimeError("E_TUPLE_HEALTH_CONTAINER_ATTESTATION")
    records.append(
        {
            "family": "container",
            **container_record,
        }
    )
    if container["sha256"] != health["pre_GPU_prerequisite_validation"][
        "container_sha256"
    ]:
        raise RuntimeError("E_TUPLE_HEALTH_CONTAINER_COMMITMENT")

    mark("DEPENDENCY_RUNTIME_DISTRIBUTIONS", 3)
    for artifact in runtime.get("dependency_artifacts", []):
        records.append(
            {
                "family": "runtime_distribution",
                **_tuple_health_verify_file(
                    model_root / "runtime-distributions" / artifact["name"],
                    artifact["sha256"],
                    artifact["bytes"],
                ),
            }
        )
    mark("DEPENDENCY_TEXT_ENCODER", 4)
    for artifact in runtime.get("bert_base_uncased", {}).get("files", []):
        records.append(
            {
                "family": "bert_base_uncased",
                **_tuple_health_verify_file(
                    model_root / "bert-base-uncased" / artifact["name"],
                    artifact["sha256"],
                    artifact["bytes"],
                ),
            }
        )
    if _installed_distributions(model_root / "runtime-pydeps") != runtime.get(
        "installed_distributions"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_RUNTIME_DISTRIBUTIONS")
    for ordinal, (family, root) in enumerate((
        ("runtime_pydeps_tree", model_root / "runtime-pydeps"),
        (
            "activity_pydeps_tree",
            public / "models/activity-pydeps/egohod_egovideo_l_zero_shot",
        ),
    ), start=5):
        mark("DEPENDENCY_CODE_TREE", ordinal)
        records.append({"family": family, **_tuple_health_tree_commitment(root)})

    premodel = cfg["calibration_C"]["extractor"][
        "mechanistic_training_tuple_premodel_result"
    ]
    if premodel.get("status") != (
        "PASS_ARTIFACTS_READY_LOCAL_RELOAD_PENDING_BLIND_SIZING"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PREMODEL_STATUS")
    artifact_paths = {
        "groundingdino_swint_ogc.pth": model_root
        / "weights/groundingdino_swint_ogc.pth",
        "sam2.1_hiera_base_plus.pt": model_root
        / "weights/sam2.1_hiera_base_plus.pt",
        "dinov2_vitb14_pretrain.pth": model_root
        / "weights/dinov2_vitb14_pretrain.pth",
        "egohos_work_dirs.zip": model_root / "weights/egohos_work_dirs.zip",
        "PE-Core-L14-336.pt": _resolve_tuple_pe_core_checkpoint(public, cfg),
        "wordnet.zip": model_root / "nltk-archives/wordnet.zip",
        "averaged_perceptron_tagger_eng.zip": model_root
        / "nltk-archives/averaged_perceptron_tagger_eng.zip",
        "nltk-3.9.1-py3-none-any.whl": model_root
        / "wheels/nltk-3.9.1-py3-none-any.whl",
        "wordfreq-3.0.2-py3-none-any.whl": model_root
        / "wheels/wordfreq-3.0.2-py3-none-any.whl",
    }
    mark("DEPENDENCY_PUBLIC_ARTIFACTS", 7)
    for name, path in artifact_paths.items():
        records.append(
            {
                "family": "tuple_public_artifact",
                **_tuple_health_verify_file(
                    path, premodel["artifact_sha256"][name]
                ),
            }
        )
    mark("DEPENDENCY_ACTION_WEIGHT", 8)
    action_candidate, _runtime = _tuple_order_action_egohod_wiring(cfg)
    action_path = _resolve_tuple_egohod_checkpoint(public, action_candidate)
    records.append(
        {
            "family": "egohod",
            **_tuple_health_verify_file(
                action_path,
                premodel["artifact_sha256"]["egohod_large_best.pt"],
                action_candidate["weight_bytes"],
            ),
        }
    )

    repository_names = {
        "EgoHOS": "EgoHOS",
        "GroundingDINO": "GroundingDINO",
        "SAM2": "sam2",
        "DINOv2": "dinov2",
    }
    for ordinal, (family, directory) in enumerate(
        repository_names.items(), start=9
    ):
        mark("DEPENDENCY_REPOSITORY_ARCHIVE", ordinal)
        source = premodel["repository_archives"][family]
        archive = model_root / "code" / f"{directory}-{source['commit']}.zip"
        records.append(
            {
                "family": "code_archive",
                **_tuple_health_verify_file(archive, source["archive_sha256"]),
            }
        )
        marker = model_root / "code" / directory / ".source-commit"
        if not marker.is_file() or marker.read_text().strip() != source["commit"]:
            raise RuntimeError("E_TUPLE_HEALTH_CODE_REVISION")
        replacements = (
            {
                "groundingdino/models/GroundingDINO/ms_deform_attn.py": (
                    GROUNDING_DINO_DEFORM_ATTN_SOURCE_SHA256,
                    GROUNDING_DINO_DEFORM_ATTN_PATCHED_SHA256,
                ),
                "groundingdino/models/GroundingDINO/groundingdino.py": (
                    GROUNDING_DINO_MODEL_SOURCE_SHA256,
                    GROUNDING_DINO_MODEL_NO_VISUALIZER_SHA256,
                ),
            }
            if family == "GroundingDINO"
            else {}
        )
        records.append(
            {
                "family": f"{family}_extracted_tree",
                **_tuple_health_verify_archive_tree(
                    archive,
                    model_root / "code" / directory,
                    replacements=replacements,
                ),
            }
        )
    grounding_root = model_root / "code/GroundingDINO"
    if (
        file_digest(
            grounding_root
            / "groundingdino/models/GroundingDINO/ms_deform_attn.py"
        )
        != GROUNDING_DINO_DEFORM_ATTN_PATCHED_SHA256
        or file_digest(
            grounding_root
            / "groundingdino/models/GroundingDINO/groundingdino.py"
        )
        != GROUNDING_DINO_MODEL_NO_VISUALIZER_SHA256
    ):
        raise RuntimeError("E_TUPLE_HEALTH_GROUNDING_PATCH")

    mark("DEPENDENCY_LEARNER_CODE", 13)
    archive_record = runtime.get("egobaby_loader", {})
    egobaby_archive = (
        model_root
        / "code"
        / f"egobabyvlm-{archive_record.get('commit')}.zip"
    )
    records.append(
        {
            "family": "egobaby_code_archive",
            **_tuple_health_verify_file(
                egobaby_archive,
                archive_record.get("archive_sha256"),
                archive_record.get("archive_bytes"),
            ),
        }
    )
    egobaby_marker = model_root / "code/egobabyvlm/.source-commit"
    if (
        not egobaby_marker.is_file()
        or egobaby_marker.read_text().strip() != archive_record.get("commit")
    ):
        raise RuntimeError("E_TUPLE_HEALTH_CODE_REVISION")
    records.append(
        {
            "family": "egobaby_extracted_tree",
            **_tuple_health_verify_archive_tree(
                egobaby_archive, model_root / "code/egobabyvlm"
            ),
        }
    )

    mark("DEPENDENCY_ACTIVITY_CODE", 14)
    action_candidate, action_runtime = _tuple_order_action_egohod_wiring(cfg)
    activity_code = _activity_code_root(public, action_candidate["candidate_id"])
    clip_code = public / "models/activity-code/CLIP"
    tree_attestations: dict[str, dict[str, Any]] = {}
    if "learner_effective_engineering_health_git_fallback_repair" in cfg:
        tree_attestations = {
            item["family"]: item
            for item in _engineering_health_git_fallback_repair(cfg)[
                "clean_tree_attestations"
            ]["repositories"]
        }
    _tuple_health_verify_git_tree(
        activity_code,
        action_candidate["code_commit"],
        tree_attestations.get("egohod_activity_code_tree"),
    )
    _tuple_health_verify_git_tree(
        clip_code,
        action_runtime["openai_CLIP_commit"],
        tree_attestations.get("openai_clip_code_tree"),
    )
    records.extend(
        [
            {
                "family": "egohod_activity_code_tree",
                **_tuple_health_tree_commitment(activity_code),
            },
            {
                "family": "openai_clip_code_tree",
                **_tuple_health_tree_commitment(clip_code),
            },
        ]
    )
    for family, path in (
        ("health_wrapper", Path(__file__).resolve().with_name("qualify_synthetic_video_calibration.sbatch")),
        ("language_adapter", Path(__file__).resolve().with_name("synthetic_video_language_adapter.py")),
    ):
        records.append(
            {
                "family": family,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            }
        )

    mark("DEPENDENCY_HAND_ARCHIVE_MEMBERS", 15)
    egohos_archive = model_root / "weights/egohos_work_dirs.zip"
    expected_members = (
        ("seg_twohands_ccda", "seg_twohands_ccda.py"),
        ("seg_twohands_ccda", "best_mIoU_iter_56000.pth"),
        ("twohands_to_cb_ccda", "twohands_to_cb_ccda.py"),
        ("twohands_to_cb_ccda", "best_mIoU_iter_76000.pth"),
        ("twohands_cb_to_obj1_ccda", "twohands_cb_to_obj1_ccda.py"),
        ("twohands_cb_to_obj1_ccda", "best_mIoU_iter_34000.pth"),
    )
    with zipfile.ZipFile(egohos_archive) as source:
        names = source.namelist()
        for folder, filename in expected_members:
            suffix = f"work_dirs/{folder}/{filename}"
            matches = [name for name in names if name.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError("E_TUPLE_HEALTH_EGOHOS_ARCHIVE_MEMBER")
            member_hash = hashlib.sha256()
            with source.open(matches[0]) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    member_hash.update(chunk)
            extracted = model_root / "egohos-checkpoints" / suffix
            if (
                not extracted.is_file()
                or file_digest(extracted) != member_hash.hexdigest()
            ):
                raise RuntimeError("E_TUPLE_HEALTH_EGOHOS_EXTRACTED_MEMBER")
            records.append(
                {
                    "family": "egohos_extracted_member",
                    "sha256": member_hash.hexdigest(),
                    "bytes": extracted.stat().st_size,
                }
            )

    mark("DEPENDENCY_LANGUAGE_ARCHIVE", 16)
    language_archive = dependency_restore["active_language_dependency_archive"]
    records.append(
        {
            "family": "language_dependency_archive",
            **_tuple_health_verify_file(
                public / "models/phase4-language-pydeps.tar",
                language_archive["sha256"],
                language_archive["bytes"],
            ),
        }
    )
    mark("DEPENDENCY_LANGUAGE_MODELS", 17)
    language = _tuple_language_artifact_record(public)
    record = {
        "schema_version": 1,
        "artifact_count": len(records),
        "artifact_bytes": sum(int(value["bytes"]) for value in records),
        "runtime_dependency_commitment_sha256": runtime[
            "runtime_dependency_commitment_sha256"
        ],
        "language_tree_commitment_sha256": language[
            "opus_mt_de_en_tree_commitment_sha256"
        ],
        "records_commitment_sha256": digest(records),
        "configuration_validation_commitment_sha256": configuration_commitment,
        "network_disabled": True,
        "telemetry_disabled": True,
    }
    record["dependency_config_commitment_sha256"] = digest(
        {
            "preflight": record,
            "configuration_validation_commitment_sha256": configuration_commitment,
            "engineering_health_amendment_commitment_sha256": health[
                "amendment_commitment_sha256"
            ],
            "engineering_health_dependency_restore_commitment_sha256": (
                dependency_restore["repair_commitment_sha256"]
            ),
            "engineering_health_topology_guard_repair_commitment_sha256": (
                topology_guard_repair["repair_commitment_sha256"]
            ),
            **(
                {
                    "engineering_health_submission_export_repair_commitment_sha256": _engineering_health_submission_export_repair(
                        cfg
                    )["repair_commitment_sha256"]
                }
                if "learner_effective_engineering_health_submission_export_repair"
                in cfg
                else {}
            ),
            **(
                {
                    "engineering_health_fixture_bind_repair_commitment_sha256": _engineering_health_fixture_bind_repair(
                        cfg
                    )["repair_commitment_sha256"]
                }
                if "learner_effective_engineering_health_fixture_bind_repair"
                in cfg
                else {}
            ),
            **(
                {
                    "engineering_health_portable_AST_repair_commitment_sha256": _engineering_health_portable_ast_repair(
                        cfg
                    )["repair_commitment_sha256"]
                }
                if "learner_effective_engineering_health_portable_AST_repair"
                in cfg
                else {}
            ),
            **(
                {
                    "engineering_health_historical_lineage_repair_commitment_sha256": _engineering_health_historical_lineage_repair(
                        cfg
                    )["repair_commitment_sha256"]
                }
                if "learner_effective_engineering_health_historical_lineage_repair"
                in cfg
                else {}
            ),
            **(
                {
                    "engineering_health_git_fallback_repair_commitment_sha256": _engineering_health_git_fallback_repair(
                        cfg
                    )["repair_commitment_sha256"]
                }
                if "learner_effective_engineering_health_git_fallback_repair"
                in cfg
                else {}
            ),
        }
    )
    mark("DEPENDENCY_COMPLETE", 18)
    return record


def _tuple_health_selected_rows(
    manifest: dict[str, Any], projection: dict[str, Any], module_id: str
) -> dict[str, list[dict[str, Any]]]:
    families = {
        "language_lexical",
        "referent_attribute",
        "recurrence",
        "hand_contact",
        "sensor",
        "order_action",
    }
    output = {family: [] for family in families}
    cases = [
        value
        for value in projection["cases"]
        if value.get("module_id") == module_id
    ]
    if len(cases) != 4:
        raise RuntimeError("E_TUPLE_HEALTH_CASE_CLASS_DEFICIT")
    family = str(cases[0]["source_family"])
    if family not in families or any(value["source_family"] != family for value in cases):
        raise RuntimeError("E_TUPLE_HEALTH_SOURCE_FAMILY")
    indexed = _tuple_health_fixture_rows(
        manifest["partitions"]["development"][family], family
    )
    for case in cases:
        ordinal = int(case["source_fixture_ordinal"])
        if ordinal not in indexed:
            raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_ORDINAL")
        output[family].append(indexed[ordinal])
    return output


def _tuple_health_topology_attestation(
    path: Path, attempt: int, cfg: dict[str, Any]
) -> str:
    """Validate the wrapper's compact scheduler attestation without Slurm tools."""

    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal() or int(job_id) <= 0:
        raise RuntimeError("E_TUPLE_HEALTH_TOPOLOGY_ATTESTATION")
    if (
        "learner_effective_engineering_health_submission_export_repair" in cfg
        and attempt == 13
    ):
        topology = _engineering_health_submission_export_repair(cfg)[
            "topology_attestation_contract"
        ]
        expected_partition = topology["partition"]
        expected_wall_minutes = topology["time_limit_minutes"]
        expected_gres = topology["GRES"]
    elif (
        "learner_effective_engineering_health_fixture_bind_repair" in cfg
        and attempt == 12
    ):
        topology = _engineering_health_fixture_bind_repair(cfg)[
            "topology_attestation_contract"
        ]
        expected_partition = topology["partition"]
        expected_wall_minutes = topology["time_limit_minutes"]
        expected_gres = topology["GRES"]
    elif (
        "learner_effective_engineering_health_portable_AST_repair" in cfg
        and attempt == 11
    ):
        topology = _engineering_health_portable_ast_repair(cfg)[
            "topology_attestation_contract"
        ]
        expected_partition = topology["partition"]
        expected_wall_minutes = topology["time_limit_minutes"]
        expected_gres = topology["GRES"]
    elif (
        "learner_effective_engineering_health_historical_lineage_repair" in cfg
        and attempt == 10
    ):
        topology = _engineering_health_historical_lineage_repair(cfg)[
            "topology_attestation_contract"
        ]
        expected_partition = topology["partition"]
        expected_wall_minutes = topology["time_limit_minutes"]
        expected_gres = topology["GRES"]
    elif (
        "learner_effective_engineering_health_git_fallback_repair" in cfg
        and attempt == 9
    ):
        topology = _engineering_health_git_fallback_repair(cfg)[
            "topology_attestation_contract"
        ]
        expected_partition = topology["partition"]
        expected_wall_minutes = topology["time_limit_minutes"]
        expected_gres = topology["GRES"]
    elif (
        "learner_effective_engineering_health_scheduler_policy" in cfg
        and attempt == 8
    ):
        topology = _engineering_health_scheduler_policy(cfg)[
            "topology_attestation_contract"
        ]
        expected_partition = topology["partition"]
        expected_wall_minutes = topology["time_limit_minutes"]
        expected_gres = topology["GRES"]
    else:
        expected_partition = "h100"
        expected_wall_minutes = 60 if attempt == 8 else 15
        expected_gres = "gpu:nvidia_h100_nvl_3g.47gb:1"
    expected = {
        "schema_version": 1,
        "attempt": attempt,
        "job_id": int(job_id),
        "partition": expected_partition,
        "node_count": 1,
        "CPU_count": 8,
        "task_count": 1,
        "time_limit_minutes": expected_wall_minutes,
        "memory_per_CPU_GiB": 4,
        "GRES": expected_gres,
        "predicate_count": 7,
        "predicate_pass_count": 7,
        "world_size": 1,
        "local_world_size": 1,
        "source": "WRAPPER_SCONTROL_BEFORE_CONTAINER",
    }
    try:
        mode = stat.S_IMODE(path.lstat().st_mode)
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_TOPOLOGY_ATTESTATION") from error
    if (
        not path.is_file()
        or path.is_symlink()
        or mode != 0o600
        or value != expected
        or raw != canonical(value) + b"\n"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_TOPOLOGY_ATTESTATION")
    return file_digest(path)


def _tuple_container_attestation(
    path: Path,
    cfg: dict[str, Any],
    *,
    run_mode: str,
    attempt: int | None,
) -> dict[str, Any]:
    """Validate the host-side SIF attestation without dereferencing it in-container."""

    if run_mode not in {"health", "development", "holdout"}:
        raise RuntimeError("E_TUPLE_HEALTH_CONTAINER_ATTESTATION")
    if (run_mode == "health") != (type(attempt) is int):
        raise RuntimeError("E_TUPLE_HEALTH_CONTAINER_ATTESTATION")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal() or int(job_id) <= 0:
        raise RuntimeError("E_TUPLE_HEALTH_CONTAINER_ATTESTATION")
    container = _tuple_runtime_amendment(cfg)["base_container"]
    expected = {
        "artifact_family": "BASE_CONTAINER",
        "attempt": attempt,
        "bytes": 3731320832,
        "host_entry_is_symlink": True,
        "job_id": int(job_id),
        "predicate_count": 4,
        "predicate_pass_count": 4,
        "resolved_target_regular_file": True,
        "run_mode": run_mode,
        "schema_version": 1,
        "sha256": container["sha256"],
        "source": "WRAPPER_HOST_BEFORE_CONTAINER",
    }
    try:
        mode = stat.S_IMODE(path.lstat().st_mode)
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_CONTAINER_ATTESTATION") from error
    if (
        not path.is_file()
        or path.is_symlink()
        or mode != 0o600
        or value != expected
        or raw != canonical(value) + b"\n"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_CONTAINER_ATTESTATION")
    return {"sha256": value["sha256"], "bytes": value["bytes"]}


def _tuple_fixture_bind_attestation(
    path: Path,
    cfg: dict[str, Any],
    *,
    run_mode: str,
    attempt: int | None,
) -> str:
    """Validate the wrapper's path-free read-only sealed-fixture bind record."""

    if run_mode not in {"health", "development", "holdout"}:
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_BIND_ATTESTATION")
    if (run_mode == "health") != (type(attempt) is int):
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_BIND_ATTESTATION")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdecimal() or int(job_id) <= 0:
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_BIND_ATTESTATION")
    repair = _engineering_health_fixture_bind_repair(cfg)[
        "failure_specific_repair"
    ]
    source = repair["source_record_file"]
    no_hand = repair["verified_no_hand_seal_file"]
    manifest = repair["fixture_manifest_file"]
    expected = {
        "artifact_family": "SEALED_PUBLIC_FIXTURE_BIND",
        "attempt": attempt,
        "fixture_manifest_bytes": manifest["bytes"],
        "fixture_manifest_file_sha256": manifest["sha256"],
        "job_id": int(job_id),
        "path_field_count": 0,
        "predicate_count": 9,
        "predicate_pass_count": 9,
        "read_only": True,
        "run_mode": run_mode,
        "schema_version": 1,
        "source": "WRAPPER_HOST_BEFORE_CONTAINER",
        "source_bound_at_original_absolute_alias": True,
        "source_bound_over_active_fixture_target": True,
        "source_record_bytes": source["bytes"],
        "source_record_file_sha256": source["sha256"],
        "verified_no_hand_seal_bytes": no_hand["bytes"],
        "verified_no_hand_seal_file_sha256": no_hand["sha256"],
    }
    try:
        mode = stat.S_IMODE(path.lstat().st_mode)
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_BIND_ATTESTATION") from error
    if (
        not path.is_file()
        or path.is_symlink()
        or mode != 0o600
        or value != expected
        or raw != canonical(value) + b"\n"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_BIND_ATTESTATION")
    return file_digest(path)


TUPLE_HEALTH_PROGRESS_STAGES = frozenset(
    {
        "RUNNER_ENTRY",
        "ATTESTATIONS_VALIDATED",
        "DEPENDENCY_CONFIGURATION",
        "DEPENDENCY_RUNTIME_MANIFEST",
        "DEPENDENCY_RUNTIME_DISTRIBUTIONS",
        "DEPENDENCY_TEXT_ENCODER",
        "DEPENDENCY_CODE_TREE",
        "DEPENDENCY_PUBLIC_ARTIFACTS",
        "DEPENDENCY_ACTION_WEIGHT",
        "DEPENDENCY_REPOSITORY_ARCHIVE",
        "DEPENDENCY_LEARNER_CODE",
        "DEPENDENCY_ACTIVITY_CODE",
        "DEPENDENCY_HAND_ARCHIVE_MEMBERS",
        "DEPENDENCY_LANGUAGE_ARCHIVE",
        "DEPENDENCY_LANGUAGE_MODELS",
        "DEPENDENCY_COMPLETE",
        "FIXTURE_VERIFICATION",
        "MICROFIXTURE_PROJECTION",
        "MODULE_EXECUTION",
        "HEALTH_COMPLETE",
    }
)


def _write_tuple_health_progress(
    path: Path,
    *,
    attempt: int,
    stage: str,
    stage_ordinal: int,
    module_ordinal: int,
    replicate: int,
    update_count: int,
    submission_started_epoch: float,
) -> None:
    """Persist only a stable aggregate engineering stage for timeout diagnosis."""

    elapsed = max(0.0, time.time() - submission_started_epoch)
    if (
        type(attempt) is not int
        or attempt < 1
        or stage not in TUPLE_HEALTH_PROGRESS_STAGES
        or type(stage_ordinal) is not int
        or not 0 <= stage_ordinal <= 18
        or type(module_ordinal) is not int
        or not 0 <= module_ordinal <= len(TUPLE_QUALIFICATION_MODULE_IDS)
        or type(replicate) is not int
        or replicate not in {0, 1, 2}
        or type(update_count) is not int
        or update_count < 1
        or not math.isfinite(elapsed)
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PROGRESS_RECORD")
    value = {
        "schema_version": 1,
        "attempt": attempt,
        "stage": stage,
        "stage_ordinal": stage_ordinal,
        "module_ordinal": module_ordinal,
        "replicate": replicate,
        "update_count": update_count,
        "elapsed_seconds": elapsed,
        "scientific_metric_count": 0,
        "sensitive_detail_field_count": 0,
    }
    write_private(path, value)
    if (
        not path.is_file()
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
        or json.loads(path.read_bytes()) != value
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PROGRESS_RECORD")


def _tuple_health_topology(
    device: str,
    engineering_health: bool = True,
    *,
    topology_attestation: Path | None = None,
    attempt: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> str | None:
    attestation_commitment = None
    if engineering_health:
        if (
            topology_attestation is None
            or type(attempt) is not int
            or not isinstance(cfg, dict)
        ):
            raise RuntimeError("E_TUPLE_HEALTH_TOPOLOGY_ATTESTATION")
        attestation_commitment = _tuple_health_topology_attestation(
            topology_attestation, int(attempt), cfg
        )

    import torch

    device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else ""
    if engineering_health:
        expected_gpu_type = _engineering_health_attempt_gpu_type(cfg, int(attempt))
        total_memory = (
            int(torch.cuda.get_device_properties(0).total_memory)
            if torch.cuda.device_count() == 1
            else 0
        )
        if (
            "learner_effective_engineering_health_submission_export_repair"
            in cfg
            and int(attempt) == 13
        ):
            topology = _engineering_health_submission_export_repair(cfg)[
                "topology_attestation_contract"
            ]
            configured_name = str(topology["expected_device_name"])
            expected_device = (
                device_name.startswith("NVIDIA H100 NVL")
                if expected_gpu_type == "NVIDIA_H100_NVL_3G_47GB_MIG"
                else device_name == configured_name
            )
            expected_memory = (
                int(topology["visible_memory_GiB_min"]) * 1024**3
                <= total_memory
                <= int(topology["visible_memory_GiB_max"]) * 1024**3
            )
        elif (
            "learner_effective_engineering_health_fixture_bind_repair" in cfg
            and int(attempt) == 12
        ):
            topology = _engineering_health_fixture_bind_repair(cfg)[
                "topology_attestation_contract"
            ]
            configured_name = str(topology["expected_device_name"])
            expected_device = (
                device_name.startswith("NVIDIA H100 NVL")
                if expected_gpu_type == "NVIDIA_H100_NVL_3G_47GB_MIG"
                else device_name == configured_name
            )
            expected_memory = (
                int(topology["visible_memory_GiB_min"]) * 1024**3
                <= total_memory
                <= int(topology["visible_memory_GiB_max"]) * 1024**3
            )
        elif (
            "learner_effective_engineering_health_portable_AST_repair" in cfg
            and int(attempt) == 11
        ):
            topology = _engineering_health_portable_ast_repair(cfg)[
                "topology_attestation_contract"
            ]
            configured_name = str(topology["expected_device_name"])
            expected_device = (
                device_name.startswith("NVIDIA H100 NVL")
                if expected_gpu_type == "NVIDIA_H100_NVL_3G_47GB_MIG"
                else device_name == configured_name
            )
            expected_memory = (
                int(topology["visible_memory_GiB_min"]) * 1024**3
                <= total_memory
                <= int(topology["visible_memory_GiB_max"]) * 1024**3
            )
        elif (
            "learner_effective_engineering_health_historical_lineage_repair"
            in cfg
            and int(attempt) == 10
        ):
            topology = _engineering_health_historical_lineage_repair(cfg)[
                "topology_attestation_contract"
            ]
            configured_name = str(topology["expected_device_name"])
            expected_device = (
                device_name.startswith("NVIDIA H100 NVL")
                if expected_gpu_type == "NVIDIA_H100_NVL_3G_47GB_MIG"
                else device_name == configured_name
            )
            expected_memory = (
                int(topology["visible_memory_GiB_min"]) * 1024**3
                <= total_memory
                <= int(topology["visible_memory_GiB_max"]) * 1024**3
            )
        elif (
            "learner_effective_engineering_health_git_fallback_repair" in cfg
            and int(attempt) == 9
        ):
            topology = _engineering_health_git_fallback_repair(cfg)[
                "topology_attestation_contract"
            ]
            configured_name = str(topology["expected_device_name"])
            expected_device = (
                device_name.startswith("NVIDIA H100 NVL")
                if expected_gpu_type == "NVIDIA_H100_NVL_3G_47GB_MIG"
                else device_name == configured_name
            )
            expected_memory = (
                int(topology["visible_memory_GiB_min"]) * 1024**3
                <= total_memory
                <= int(topology["visible_memory_GiB_max"]) * 1024**3
            )
        elif expected_gpu_type == "NVIDIA_A30_24GB":
            expected_device = device_name == "NVIDIA A30"
            expected_memory = 23 * 1024**3 <= total_memory <= 25 * 1024**3
        elif expected_gpu_type == "NVIDIA_H100_NVL":
            expected_device = device_name == "NVIDIA H100 NVL"
            expected_memory = 85 * 1024**3 <= total_memory <= 100 * 1024**3
        elif expected_gpu_type == "NVIDIA_H100_NVL_3G_47GB_MIG":
            expected_device = device_name.startswith("NVIDIA H100 NVL")
            expected_memory = 45 * 1024**3 <= total_memory <= 50 * 1024**3
        else:
            raise RuntimeError("E_TUPLE_HEALTH_GPU_TOPOLOGY")
        scheduler_topology = attestation_commitment is not None
    else:
        expected_device = device_name == "NVIDIA A30"
        expected_memory = True
        scheduler_topology = all(
            (
                os.environ.get("SLURM_JOB_PARTITION") == "a30",
                os.environ.get("SLURM_JOB_NUM_NODES") == "1",
                os.environ.get("SLURM_NTASKS") == "1",
                os.environ.get("SLURM_CPUS_PER_TASK") == "8",
                os.environ.get("SLURM_GPUS_ON_NODE") == "1",
            )
        )
    if (
        device != "cuda"
        or torch.cuda.device_count() != 1
        or not expected_device
        or not expected_memory
        or not scheduler_topology
        or os.environ.get("WORLD_SIZE", "1") != "1"
        or os.environ.get("LOCAL_WORLD_SIZE", "1") != "1"
    ):
        raise RuntimeError("E_TUPLE_HEALTH_GPU_TOPOLOGY")
    return attestation_commitment


def run_tuple_health(args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded production-path microqualification with zero metrics."""

    cfg = json.loads(args.config.read_text())
    if "learner_effective_engineering_health_submission_export_repair" in cfg:
        _engineering_health_submission_export_repair(cfg)
        if int(args.attempt) != 13:
            raise RuntimeError("E_TUPLE_HEALTH_SUBMISSION_EXPORT_REPAIR_ATTEMPT")
    elif "learner_effective_engineering_health_fixture_bind_repair" in cfg:
        _engineering_health_fixture_bind_repair(cfg)
        if int(args.attempt) != 12:
            raise RuntimeError("E_TUPLE_HEALTH_FIXTURE_BIND_REPAIR_ATTEMPT")
    elif "learner_effective_engineering_health_portable_AST_repair" in cfg:
        _engineering_health_portable_ast_repair(cfg)
        if int(args.attempt) != 11:
            raise RuntimeError("E_TUPLE_HEALTH_PORTABLE_AST_REPAIR_ATTEMPT")
    elif "learner_effective_engineering_health_historical_lineage_repair" in cfg:
        _engineering_health_historical_lineage_repair(cfg)
        if int(args.attempt) != 10:
            raise RuntimeError("E_TUPLE_HEALTH_HISTORICAL_LINEAGE_REPAIR_ATTEMPT")
    elif "learner_effective_engineering_health_git_fallback_repair" in cfg:
        _engineering_health_git_fallback_repair(cfg)
        if int(args.attempt) != 9:
            raise RuntimeError("E_TUPLE_HEALTH_GIT_FALLBACK_REPAIR_ATTEMPT")
    elif "learner_effective_engineering_health_extended_wall_repair" in cfg:
        _engineering_health_extended_wall_repair(cfg)
        if int(args.attempt) != 8:
            raise RuntimeError("E_TUPLE_HEALTH_EXTENDED_WALL_REPAIR_ATTEMPT")
    elif "learner_effective_engineering_health_progress_repair" in cfg:
        _engineering_health_progress_repair(cfg)
        if int(args.attempt) != 7:
            raise RuntimeError("E_TUPLE_HEALTH_PROGRESS_REPAIR_ATTEMPT")
    elif "learner_effective_engineering_health_iterative_reauthorization" in cfg:
        _engineering_health_iterative_reauthorization(cfg)
        if int(args.attempt) != 6:
            raise RuntimeError("E_TUPLE_HEALTH_ITERATIVE_REAUTHORIZED_ATTEMPT")
    elif "learner_effective_engineering_health_parser_repair_result" in cfg:
        _engineering_health_parser_repair_result(cfg)
        raise RuntimeError("E_TUPLE_HEALTH_ROUTE_EXHAUSTED")
    elif "learner_effective_engineering_health_parser_repair_reauthorization" in cfg:
        _engineering_health_parser_repair_reauthorization(cfg)
        if int(args.attempt) != 5:
            raise RuntimeError("E_TUPLE_HEALTH_PARSER_REPAIR_REAUTHORIZED_ATTEMPT")
    elif "learner_effective_engineering_health_reauthorization_result" in cfg:
        _engineering_health_reauthorization_result(cfg)
        raise RuntimeError("E_TUPLE_HEALTH_ROUTE_EXHAUSTED")
    elif "learner_effective_engineering_health_result" in cfg:
        if "learner_effective_engineering_health_reauthorization" not in cfg:
            _engineering_health_terminal_result(cfg)
            raise RuntimeError("E_TUPLE_HEALTH_ROUTE_EXHAUSTED")
        _engineering_health_reauthorization(cfg)
        if int(args.attempt) != 4:
            raise RuntimeError("E_TUPLE_HEALTH_REAUTHORIZED_ATTEMPT")
    health = _engineering_health_amendment(cfg)
    root = _tuple_health_root(args.public_root)
    _require_external_or_ignored_output(root)
    _require_external_or_ignored_output(args.scratch_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    attempt_root = root / "health" / f"attempt-{int(args.attempt):02d}"
    result_path = attempt_root / "full-result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        expected = existing.pop("engineering_health_commitment_sha256", None)
        existing["engineering_health_commitment_sha256"] = expected
        if not isinstance(expected, str) or digest(
            {key: value for key, value in existing.items() if key != "engineering_health_commitment_sha256"}
        ) != expected:
            raise RuntimeError("E_TUPLE_HEALTH_RESULT_COMMITMENT")
        _validate_tuple_health_full(existing, cfg)
        return _tuple_health_compact(existing)
    attempt_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker = _tuple_health_wrapper_marker(attempt_root, int(args.attempt), cfg)
    if args.container_attestation != attempt_root / "container-attestation.json":
        raise RuntimeError("E_TUPLE_HEALTH_CONTAINER_ATTESTATION")
    expected_attempt_files = {
        "container-attestation.json",
        "topology-attestation.json",
        "wrapper-started.json",
    }
    if "learner_effective_engineering_health_submission_export_repair" in cfg:
        _engineering_health_submission_export_repair(cfg)
        expected_attempt_files.add("fixture-bind-attestation.json")
    elif "learner_effective_engineering_health_fixture_bind_repair" in cfg:
        expected_attempt_files.add("fixture-bind-attestation.json")
    if set(path.name for path in attempt_root.iterdir()) != expected_attempt_files:
        raise RuntimeError("E_TUPLE_HEALTH_PARTIAL_ATTEMPT")
    submission_started_epoch = float(marker["submission_started_epoch"])
    trace_root = attempt_root / "traces"
    prior = []
    for ordinal in range(1, int(args.attempt)):
        prior_path = root / "health" / f"attempt-{ordinal:02d}" / "full-result.json"
        if prior_path.is_file():
            value = json.loads(prior_path.read_text())
            _validate_tuple_health_full(value, cfg)
            payload = {
                key: item
                for key, item in value.items()
                if key != "engineering_health_commitment_sha256"
            }
            if digest(payload) != value.get(
                "engineering_health_commitment_sha256"
            ):
                raise RuntimeError("E_TUPLE_HEALTH_RESULT_COMMITMENT")
            resource = value["resource"]
        else:
            resource = _tuple_health_incomplete_attempt_resource(
                root, ordinal, cfg
            )
        prior.append({"attempt": ordinal, "resource": resource})
    available = _tuple_health_budget(int(args.attempt), prior, cfg)
    before_bytes = _tuple_health_tree_bytes(root)
    progress_path = attempt_root / "engineering-progress.json"
    progress_update_count = 0

    def record_progress(
        stage: str,
        stage_ordinal: int,
        module_ordinal: int = 0,
        replicate: int = 0,
    ) -> None:
        nonlocal progress_update_count
        progress_update_count += 1
        _write_tuple_health_progress(
            progress_path,
            attempt=int(args.attempt),
            stage=stage,
            stage_ordinal=stage_ordinal,
            module_ordinal=module_ordinal,
            replicate=replicate,
            update_count=progress_update_count,
            submission_started_epoch=submission_started_epoch,
        )

    record_progress("RUNNER_ENTRY", 0)
    module_results: list[dict[str, Any]] = []
    projection: dict[str, Any] | None = None
    dependency: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    topology_attestation_commitment: str | None = None
    try:
        topology_attestation_commitment = _tuple_health_topology(
            str(args.device),
            topology_attestation=attempt_root / "topology-attestation.json",
            attempt=int(args.attempt),
            cfg=cfg,
        )
        container_record = _tuple_container_attestation(
            args.container_attestation,
            cfg,
            run_mode="health",
            attempt=int(args.attempt),
        )
        if "learner_effective_engineering_health_fixture_bind_repair" in cfg:
            _tuple_fixture_bind_attestation(
                attempt_root / "fixture-bind-attestation.json",
                cfg,
                run_mode="health",
                attempt=int(args.attempt),
            )
        record_progress("ATTESTATIONS_VALIDATED", 0)
        if (
            os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
            or os.environ.get("HF_HUB_DISABLE_TELEMETRY") != "1"
            or os.environ.get("WANDB_DISABLED", "").casefold() != "true"
        ):
            raise RuntimeError("E_TUPLE_HEALTH_OFFLINE_ENVIRONMENT")
        dependency = _tuple_health_dependency_preflight(
            args.public_root,
            cfg,
            container_record,
            record_progress,
        )
        record_progress("FIXTURE_VERIFICATION", 0)
        manifest, fixture_root = _verify_tuple_fixture_manifest(
            args.public_root, cfg
        )
        record_progress("MICROFIXTURE_PROJECTION", 0)
        projection = _tuple_health_projection(manifest, cfg)
        projection_path = root / "microfixture-manifest.json"
        if projection_path.is_file():
            if json.loads(projection_path.read_text()) != projection:
                raise RuntimeError("E_TUPLE_HEALTH_PROJECTION_CHANGED")
        else:
            write_private_new(projection_path, projection)
        no_hand = manifest["verified_no_hand_seal_commitment_sha256"]
        runners = _tuple_module_runners()
        for module_ordinal, module_id in enumerate(
            TUPLE_QUALIFICATION_MODULE_IDS, start=1
        ):
            module_scratch = args.scratch_root / module_id
            module_scratch.mkdir(parents=True, exist_ok=False, mode=0o700)
            try:
                replicates = []
                for replicate in (1, 2):
                    record_progress(
                        "MODULE_EXECUTION",
                        0,
                        module_ordinal,
                        replicate,
                    )
                    replicate_scratch = module_scratch / f"replicate-{replicate}"
                    replicate_scratch.mkdir(mode=0o700)
                    context = {
                        "cfg": cfg,
                        "public_root": args.public_root,
                        "scratch_root": replicate_scratch,
                        "fixture_root": fixture_root,
                        "fixture_manifest": manifest,
                        "fixture_manifest_commitment_sha256": manifest[
                            "public_fixture_manifest_commitment_sha256"
                        ],
                        "partition": "development",
                        "rows": _tuple_health_selected_rows(
                            manifest, projection, module_id
                        ),
                        "device": args.device,
                        "thresholds": {},
                        "verified_no_hand_seal": {
                            "status": "PASS",
                            "verified_no_hand_seal_commitment_sha256": no_hand,
                        },
                        "module_cache": {},
                        "engineering_health": True,
                    }
                    result = runners[module_id](context)
                    _validate_tuple_health_module_result(
                        result, expected_case_count=4
                    )
                    if result.get("module_id") != module_id:
                        raise RuntimeError("E_TUPLE_HEALTH_MODULE_RESULT")
                    replicates.append(result)
                if (
                    replicates[0]["production_output_commitment_sha256"]
                    != replicates[1]["production_output_commitment_sha256"]
                ):
                    raise RuntimeError("E_TUPLE_HEALTH_NONDETERMINISTIC_OUTPUT")
                module_results.append(replicates[0])
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                module_results.append(
                    _tuple_health_error(module_id, error, trace_root)
                )
        record_progress("HEALTH_COMPLETE", 0)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if not module_results:
            for ordinal, module_id in enumerate(TUPLE_QUALIFICATION_MODULE_IDS):
                module_results.append(
                    _tuple_health_error(
                        module_id,
                        error
                        if ordinal == 0
                        else RuntimeError("E_TUPLE_HEALTH_PREFLIGHT_BLOCKED"),
                        trace_root,
                    )
                )
    elapsed_seconds = max(0.0, time.time() - submission_started_epoch)
    after_bytes = _tuple_health_tree_bytes(root)
    new_storage_gib = max(0, after_bytes - before_bytes) / (1024**3)
    resource = {
        "GPU_type": available["GPU_type"],
        "GPU_count": available["GPU_count"],
        "CPU_count": available["CPU_count"],
        "memory_GiB": available["memory_GiB"],
        "wall_minutes": elapsed_seconds / 60.0,
        "GPU_hours": elapsed_seconds / 3600.0,
        "new_storage_GiB": new_storage_gib,
        "direct_monetary_cost_USD": 0,
    }
    cumulative_resource = _tuple_health_cumulative_resource(prior, resource)
    if resource["wall_minutes"] > available["per_submission_wall_minutes_max"]:
        raise RuntimeError("E_TUPLE_HEALTH_WALL_BUDGET")
    if resource["GPU_hours"] > available["remaining_GPU_hours"]:
        raise RuntimeError("E_TUPLE_HEALTH_GPU_HOUR_BUDGET")
    if resource["new_storage_GiB"] > available["remaining_storage_GiB"]:
        raise RuntimeError("E_TUPLE_HEALTH_STORAGE_BUDGET")
    passed = sum(
        value.get("status") == "PASS_ENGINEERING" for value in module_results
    )
    failures = len(module_results) - passed
    full = {
        "schema_version": 1,
        "status": (
            "PASS_ENGINEERING_HEALTH" if failures == 0 else "ENGINEERING_BLOCKER"
        ),
        "route_id": health["route_id"],
        "attempt": int(args.attempt),
        "public_fixture_manifest_commitment_sha256": (
            manifest.get("public_fixture_manifest_commitment_sha256")
            if manifest is not None
            else health["preserved_commitments"]["public_fixture_manifest_sha256"]
        ),
        "runner_commitment_sha256": file_digest(Path(__file__).resolve()),
        "config_commitment_sha256": digest(cfg),
        "dependency_config_commitment_sha256": (
            dependency["dependency_config_commitment_sha256"]
            if dependency is not None
            else digest(
                {
                    "config_sha256": file_digest(args.config),
                    "health_amendment": health["amendment_commitment_sha256"],
                    "topology_attestation_commitment_sha256": (
                        topology_attestation_commitment
                        or "PREFLIGHT_BLOCKED_BEFORE_ATTESTATION"
                    ),
                }
            )
        ),
        "microfixture_manifest_commitment_sha256": (
            projection["microfixture_manifest_commitment_sha256"]
            if projection is not None
            else digest(
                {
                    "status": "PREFLIGHT_BLOCKED_BEFORE_PROJECTION",
                    "fixture": health["preserved_commitments"][
                        "public_fixture_manifest_sha256"
                    ],
                }
            )
        ),
        "module_results": module_results,
        "module_count": len(TUPLE_QUALIFICATION_MODULE_IDS),
        "completed_module_count": passed,
        "failed_module_count": failures,
        "case_count": 28,
        "holdout_input_count": 0,
        "scientific_metric_count": 0,
        "failure_count": failures,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
        "unaccounted_failure_count": sum(
            str(value.get("error_code", "")).endswith("UNACCOUNTED_FAILURE")
            for value in module_results
        ),
        "network_disabled": True,
        "telemetry_disabled": True,
        "restricted_mount_present": False,
        "resource": resource,
        "cumulative_resource": cumulative_resource,
    }
    full["engineering_health_commitment_sha256"] = digest(full)
    _validate_tuple_health_full(full, cfg)
    write_private_new(result_path, full)
    return _tuple_health_compact(full)


def _load_tuple_health_pass(
    public: Path, cfg: dict[str, Any], fixture_commitment: str
) -> dict[str, Any]:
    if "learner_effective_engineering_health_submission_export_repair" in cfg:
        _engineering_health_submission_export_repair(cfg)
    elif "learner_effective_engineering_health_fixture_bind_repair" in cfg:
        _engineering_health_fixture_bind_repair(cfg)
    elif "learner_effective_engineering_health_portable_AST_repair" in cfg:
        _engineering_health_portable_ast_repair(cfg)
    elif "learner_effective_engineering_health_git_fallback_repair" in cfg:
        _engineering_health_git_fallback_repair(cfg)
    elif "learner_effective_engineering_health_extended_wall_repair" in cfg:
        _engineering_health_extended_wall_repair(cfg)
    elif "learner_effective_engineering_health_progress_repair" in cfg:
        _engineering_health_progress_repair(cfg)
    elif "learner_effective_engineering_health_iterative_reauthorization" in cfg:
        _engineering_health_iterative_reauthorization(cfg)
    elif "learner_effective_engineering_health_parser_repair_result" in cfg:
        _engineering_health_parser_repair_result(cfg)
        raise RuntimeError("E_TUPLE_HEALTH_ROUTE_EXHAUSTED")
    elif "learner_effective_engineering_health_parser_repair_reauthorization" in cfg:
        _engineering_health_parser_repair_reauthorization(cfg)
    elif "learner_effective_engineering_health_reauthorization_result" in cfg:
        _engineering_health_reauthorization_result(cfg)
        raise RuntimeError("E_TUPLE_HEALTH_ROUTE_EXHAUSTED")
    elif "learner_effective_engineering_health_result" in cfg:
        if "learner_effective_engineering_health_reauthorization" not in cfg:
            _engineering_health_terminal_result(cfg)
            raise RuntimeError("E_TUPLE_HEALTH_ROUTE_EXHAUSTED")
        _engineering_health_reauthorization(cfg)
    root = _tuple_health_root(public) / "health"
    passes = []
    observed = []
    maximum_attempts = int(
        _engineering_health_resource_policy(cfg)[
            "initial_plus_repair_resmoke_submission_count_max"
        ]
    )
    for attempt in range(1, maximum_attempts + 1):
        path = root / f"attempt-{attempt:02d}" / "full-result.json"
        if not path.is_file():
            if (
                "learner_effective_engineering_health_submission_export_repair"
                in cfg
                and attempt == 12
            ):
                if attempt != len(observed) + 1:
                    raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
                observed.append(
                    {
                        "resource": _tuple_health_incomplete_attempt_resource(
                            _tuple_health_root(public), attempt, cfg
                        )
                    }
                )
                continue
            marker = root / f"attempt-{attempt:02d}" / "wrapper-started.json"
            if marker.is_file():
                if attempt != len(observed) + 1:
                    raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
                observed.append(
                    {
                        "resource": _tuple_health_incomplete_attempt_resource(
                            _tuple_health_root(public), attempt, cfg
                        )
                    }
                )
            continue
        if attempt != len(observed) + 1:
            raise RuntimeError("E_TUPLE_HEALTH_ATTEMPT_BUDGET")
        value = json.loads(path.read_text())
        _validate_tuple_health_full(value, cfg)
        payload = {
            key: item
            for key, item in value.items()
            if key != "engineering_health_commitment_sha256"
        }
        if digest(payload) != value.get("engineering_health_commitment_sha256"):
            raise RuntimeError("E_TUPLE_HEALTH_RESULT_COMMITMENT")
        expected_cumulative = _tuple_health_cumulative_resource(
            [{"resource": prior["resource"]} for prior in observed],
            value["resource"],
        )
        if value["cumulative_resource"] != expected_cumulative:
            raise RuntimeError("E_TUPLE_HEALTH_CUMULATIVE_RESOURCE")
        observed.append(value)
        if value["status"] == "PASS_ENGINEERING_HEALTH":
            passes.append(value)
    if len(passes) != 1:
        raise RuntimeError("E_TUPLE_SCIENCE_BEFORE_ENGINEERING_HEALTH_PASS")
    value = passes[0]
    if (
        value["public_fixture_manifest_commitment_sha256"]
        != fixture_commitment
        or value["runner_commitment_sha256"]
        != file_digest(Path(__file__).resolve())
        or value["config_commitment_sha256"] != digest(cfg)
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PASS_PROVENANCE")
    return value


def _tuple_partition_integrity_compact(full: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: full[key]
        for key in TUPLE_PARTITION_INTEGRITY_FIELDS
        if key in full
    }
    if (
        set(compact) != set(TUPLE_PARTITION_INTEGRITY_FIELDS)
        or compact["status"]
        not in {"PASS_ENGINEERING_INTEGRITY", "ENGINEERING_BLOCKER"}
        or compact["scientific_metric_count"] != 0
    ):
        raise RuntimeError("E_TUPLE_PARTITION_INTEGRITY_COMPACT_SCHEMA")
    return compact


def _validate_tuple_partition_integrity_full(value: Any) -> None:
    required = {
        "schema_version",
        "status",
        "partition",
        "attempt",
        "module_results",
        "module_count",
        "completed_module_count",
        "failed_module_count",
        "scientific_metric_count",
        "failure_count",
        "invalid_retained_record_count",
        "silent_truncation_count",
        "external_call_count",
        "unaccounted_failure_count",
        "public_fixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
        "runner_commitment_sha256",
        "network_disabled",
        "telemetry_disabled",
        "restricted_mount_present",
        "partition_engineering_integrity_commitment_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != 1
        or value.get("partition") not in {"DEVELOPMENT", "HOLDOUT"}
        or type(value.get("attempt")) is not int
        or not 1 <= value["attempt"] <= 3
        or value.get("module_count") != len(TUPLE_QUALIFICATION_MODULE_IDS)
        or value.get("scientific_metric_count") != 0
        or any(
            value.get(key) is not expected
            for key, expected in (
                ("network_disabled", True),
                ("telemetry_disabled", True),
                ("restricted_mount_present", False),
            )
        )
    ):
        raise RuntimeError("E_TUPLE_PARTITION_INTEGRITY_FULL_SCHEMA")
    modules = value.get("module_results")
    if (
        not isinstance(modules, list)
        or len(modules) != len(TUPLE_QUALIFICATION_MODULE_IDS)
        or {row.get("module_id") for row in modules}
        != set(TUPLE_QUALIFICATION_MODULE_IDS)
    ):
        raise RuntimeError("E_TUPLE_PARTITION_INTEGRITY_FULL_SCHEMA")
    for row in modules:
        _validate_tuple_health_module_result(row)
    passed = sum(row["status"] == "PASS_ENGINEERING" for row in modules)
    failed = len(modules) - passed
    expected_status = (
        "PASS_ENGINEERING_INTEGRITY" if failed == 0 else "ENGINEERING_BLOCKER"
    )
    if (
        value.get("status") != expected_status
        or value.get("completed_module_count") != passed
        or value.get("failed_module_count") != failed
        or value.get("failure_count") != failed
        or value.get("invalid_retained_record_count") != 0
        or value.get("silent_truncation_count") != 0
        or value.get("external_call_count") != 0
        or value.get("unaccounted_failure_count")
        != sum(
            str(row.get("error_code", "")).endswith("UNACCOUNTED_FAILURE")
            for row in modules
        )
    ):
        raise RuntimeError("E_TUPLE_PARTITION_INTEGRITY_FULL_SCHEMA")
    for key in (
        "public_fixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
        "runner_commitment_sha256",
        "partition_engineering_integrity_commitment_sha256",
    ):
        if not isinstance(value.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", value[key]
        ):
            raise RuntimeError("E_TUPLE_PARTITION_INTEGRITY_FULL_SCHEMA")
    commitment = value["partition_engineering_integrity_commitment_sha256"]
    if digest(
        {
            key: item
            for key, item in value.items()
            if key != "partition_engineering_integrity_commitment_sha256"
        }
    ) != commitment:
        raise RuntimeError("E_TUPLE_PARTITION_INTEGRITY_COMMITMENT")


def _tuple_partition_engineering_integrity(
    context: dict[str, Any],
    partition: str,
    trace_root: Path,
) -> list[dict[str, Any]]:
    """Execute every complete-partition raw path before any metric helper."""

    results = []
    try:
        runners = _tuple_module_runners()
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return [
            _tuple_health_error(
                module_id,
                error
                if ordinal == 0
                else RuntimeError("E_TUPLE_HEALTH_PREFLIGHT_BLOCKED"),
                trace_root,
            )
            for ordinal, module_id in enumerate(TUPLE_QUALIFICATION_MODULE_IDS)
        ]
    base_scratch = Path(context["scratch_root"])
    family_by_module = {
        "adapter_and_lexical": "language_lexical",
        "referent": "referent_attribute",
        "recurrence": "recurrence",
        "attribute": "referent_attribute",
        "hand_contact": "hand_contact",
        "sensor": "sensor",
        "order_action": "order_action",
    }
    for module_id in TUPLE_QUALIFICATION_MODULE_IDS:
        module_context = dict(context)
        module_context["engineering_health"] = True
        module_context["module_cache"] = {}
        module_context["scratch_root"] = base_scratch / module_id
        module_context["scratch_root"].mkdir(
            parents=True, exist_ok=False, mode=0o700
        )
        try:
            result = runners[module_id](module_context)
            expected_case_count = len(
                module_context["rows"][family_by_module[module_id]]
            )
            _validate_tuple_health_module_result(
                result, expected_case_count=expected_case_count
            )
            if result.get("module_id") != module_id:
                raise RuntimeError("E_TUPLE_PARTITION_INTEGRITY_MODULE_RESULT")
            results.append(result)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            results.append(_tuple_health_error(module_id, error, trace_root))
    return results


def _tuple_scientific_module_results(
    runners: dict[str, Any],
    context: dict[str, Any],
    trace_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Discard all scientific outputs if any module raises."""

    output: dict[str, dict[str, Any]] = {}
    errors = []
    if set(runners) != set(TUPLE_QUALIFICATION_MODULE_IDS):
        raise RuntimeError("E_TUPLE_QUALIFICATION_MODULE_SET")
    for module_id in TUPLE_QUALIFICATION_MODULE_IDS:
        try:
            result = runners[module_id](context)
            allowed = {"PASS", "NO_GO", "UNMEASURED"}
            if module_id == "order_action":
                allowed.add("NO_GO_DIAGNOSTIC")
            if not isinstance(result, dict) or result.get("status") not in allowed:
                raise RuntimeError("E_TUPLE_QUALIFICATION_MODULE_RESULT")
            output[module_id] = result
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            errors.append(_tuple_health_error(module_id, error, trace_root))
    return ({}, errors) if errors else (output, [])


def _tuple_finalize_scientific_partition(
    *,
    cfg: dict[str, Any],
    active: dict[str, Any],
    correction: dict[str, Any],
    manifest: dict[str, Any],
    partition: str,
    no_hand_commitment: str,
    threshold_seal: dict[str, Any] | None,
    integrity_results: list[dict[str, Any]],
    module_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Build one all-or-nothing scientific decision after engineering PASS."""

    module_results = _tuple_health_metric_release(
        integrity_results, module_results
    )
    _validate_tuple_qualification_record(module_results)
    observed_thresholds = _tuple_selected_thresholds(
        cfg,
        module_results,
        expected=(
            threshold_seal["selected_thresholds"]
            if threshold_seal is not None
            else None
        ),
    )
    selected_thresholds = (
        dict(threshold_seal["selected_thresholds"])
        if threshold_seal is not None
        else observed_thresholds
    )
    axis_results = _tuple_axis_results_from_modules(module_results)
    action_result = module_results["order_action"]
    combined = _tuple_combined_public_gate(
        axis_results,
        action_result,
        {"status": "DESCRIPTIVE_NOT_RERUN"},
        action_control_blocks=False,
    )
    combined = _apply_tuple_integrity_gate(
        combined, _tuple_qualification_integrity(module_results)
    )
    partition_module_commitment = _tuple_development_module_commitment(
        manifest["public_fixture_manifest_commitment_sha256"],
        module_results,
        axis_results,
        combined,
    )
    if partition == "development":
        seal = _tuple_development_threshold_seal(
            cfg, manifest, module_results, axis_results, combined
        )
        if selected_thresholds != seal["selected_thresholds"]:
            raise RuntimeError("E_TUPLE_QUALIFICATION_THRESHOLD_SEAL_MISMATCH")
        threshold_commitment = seal[
            "development_threshold_commitment_sha256"
        ]
        status = seal["status"]
        development_module_commitment = partition_module_commitment
    else:
        if threshold_seal is None:
            raise RuntimeError("E_TUPLE_HOLDOUT_BEFORE_DEVELOPMENT_SEAL")
        seal = None
        threshold_commitment = threshold_seal[
            "development_threshold_commitment_sha256"
        ]
        development_module_commitment = threshold_seal[
            "development_module_result_commitment_sha256"
        ]
        status = (
            "PASS_PUBLIC_COMBINED_GATE"
            if combined["status"] == "PASS"
            else "NO_GO_PUBLIC_COMBINED_GATE"
        )
    full = {
        "schema_version": 1,
        "status": status,
        "partition": partition,
        "public_fixture_manifest_commitment_sha256": manifest[
            "public_fixture_manifest_commitment_sha256"
        ],
        "visor_hos_correction_amendment_commitment_sha256": correction[
            "amendment_commitment_sha256"
        ],
        "verified_no_hand_seal_commitment_sha256": no_hand_commitment,
        "construct_aligned_ltx_resume_amendment_commitment_sha256": active[
            "amendment_commitment_sha256"
        ],
        "development_threshold_commitment_sha256": threshold_commitment,
        "development_module_result_commitment_sha256": development_module_commitment,
        "partition_module_result_commitment_sha256": partition_module_commitment,
        "selected_thresholds": selected_thresholds,
        "module_results": module_results,
        "axis_results": axis_results,
        "combined_gate": combined,
        "broad_activity_context": {
            "status": "DESCRIPTIVE_NOT_RERUN",
            "used_in_gate": False,
        },
        "restricted_mount_present": False,
        "network_disabled": True,
    }
    full["public_qualification_commitment_sha256"] = digest(full)
    compact = _tuple_qualification_compact(full, threshold_commitment)
    return full, seal, compact


def qualify_tuple_public(args: argparse.Namespace) -> dict[str, Any]:
    """Run one public partition and make exactly one complete stage decision."""

    partition = str(args.partition)
    if partition not in {"development", "holdout"}:
        raise RuntimeError("E_TUPLE_QUALIFICATION_PARTITION")
    cfg = json.loads(args.config.read_text())
    if "learner_effective_engineering_health_submission_export_repair" in cfg:
        _engineering_health_submission_export_repair(cfg)
    elif "learner_effective_engineering_health_fixture_bind_repair" in cfg:
        _engineering_health_fixture_bind_repair(cfg)
    active = _construct_aligned_ltx_resume_amendment(cfg)
    _tuple_qualification_execution(cfg)
    output_root = _tuple_qualification_root(args.public_root)
    _require_external_or_ignored_output(output_root)
    _require_external_or_ignored_output(args.scratch_root)
    _tuple_health_topology(str(args.device), False)
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("HF_HUB_DISABLE_TELEMETRY") != "1"
        or os.environ.get("WANDB_DISABLED", "").casefold() != "true"
    ):
        raise RuntimeError("E_TUPLE_QUALIFICATION_OFFLINE_ENVIRONMENT")
    if "learner_effective_engineering_health_fixture_bind_repair" in cfg:
        _tuple_fixture_bind_attestation(
            args.fixture_bind_attestation,
            cfg,
            run_mode=partition,
            attempt=None,
        )
    _verify_tuple_runtime_manifest(args.public_root, cfg)
    manifest, fixture_root = _verify_tuple_fixture_manifest(args.public_root, cfg)
    correction = _tuple_visor_hos_correction_amendment(cfg)
    if manifest.get("visor_hos_correction_amendment_commitment_sha256") != correction[
        "amendment_commitment_sha256"
    ]:
        raise RuntimeError("E_TUPLE_QUALIFICATION_CORRECTION_COMMITMENT")
    no_hand_commitment = manifest.get("verified_no_hand_seal_commitment_sha256")
    if not isinstance(no_hand_commitment, str) or not re.fullmatch(
        r"[0-9a-f]{64}", no_hand_commitment
    ):
        raise RuntimeError("E_TUPLE_QUALIFICATION_NO_HAND_SEAL")
    health_pass = _load_tuple_health_pass(
        args.public_root,
        cfg,
        manifest["public_fixture_manifest_commitment_sha256"],
    )
    container_record = _tuple_container_attestation(
        args.container_attestation,
        cfg,
        run_mode=partition,
        attempt=None,
    )
    dependency = _tuple_health_dependency_preflight(
        args.public_root,
        cfg,
        container_record,
    )
    if (
        dependency["dependency_config_commitment_sha256"]
        != health_pass["dependency_config_commitment_sha256"]
    ):
        raise RuntimeError("E_TUPLE_HEALTH_PASS_DEPENDENCY_PROVENANCE")
    recovered = _recover_tuple_qualification_transaction(
        args.public_root, cfg, manifest, partition
    )
    if recovered is not None:
        return recovered
    if partition == "holdout":
        _refuse_tuple_qualification_overwrite(args.public_root, partition)
    if partition == "development":
        reused = _reuse_tuple_development_pair(args.public_root, cfg, manifest)
        if reused is not None:
            return reused
    threshold_seal = (
        _load_tuple_development_threshold_seal(args.public_root, cfg, manifest)
        if partition == "holdout"
        else None
    )
    args.scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    base_context = {
        "cfg": cfg,
        "public_root": args.public_root,
        "fixture_root": fixture_root,
        "fixture_manifest": manifest,
        "fixture_manifest_commitment_sha256": manifest[
            "public_fixture_manifest_commitment_sha256"
        ],
        "partition": partition,
        "rows": manifest["partitions"][partition],
        "device": args.device,
        "thresholds": (
            dict(threshold_seal["selected_thresholds"])
            if threshold_seal is not None
            else {}
        ),
        "verified_no_hand_seal": {
            "status": "PASS",
            "verified_no_hand_seal_commitment_sha256": no_hand_commitment,
        },
    }
    integrity_root = output_root / "engineering-integrity"
    integrity_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    prior_integrity = sorted(
        path
        for path in integrity_root.iterdir()
        if re.fullmatch(
            rf"{re.escape(partition)}-attempt-[0-9]{{2}}\.json", path.name
        )
    )
    observed_attempts = [
        int(path.stem.rsplit("-", maxsplit=1)[1]) for path in prior_integrity
    ]
    if observed_attempts != list(range(1, len(prior_integrity) + 1)):
        raise RuntimeError("E_TUPLE_PARTITION_ENGINEERING_ATTEMPT_BUDGET")
    if len(prior_integrity) >= 3:
        raise RuntimeError("E_TUPLE_PARTITION_ENGINEERING_ATTEMPT_BUDGET")
    integrity_attempt = len(prior_integrity) + 1
    integrity_context = {
        **base_context,
        "scratch_root": args.scratch_root / f"{partition}-engineering-integrity",
        "module_cache": {},
    }
    integrity_results = _tuple_partition_engineering_integrity(
        integrity_context,
        partition,
        integrity_root / f"{partition}-attempt-{integrity_attempt:02d}-traces",
    )
    integrity_passed = sum(
        value.get("status") == "PASS_ENGINEERING"
        for value in integrity_results
    )
    integrity_failed = len(TUPLE_QUALIFICATION_MODULE_IDS) - integrity_passed
    integrity_full = {
        "schema_version": 1,
        "status": (
            "PASS_ENGINEERING_INTEGRITY"
            if integrity_failed == 0
            else "ENGINEERING_BLOCKER"
        ),
        "partition": partition.upper(),
        "attempt": integrity_attempt,
        "module_results": integrity_results,
        "module_count": len(TUPLE_QUALIFICATION_MODULE_IDS),
        "completed_module_count": integrity_passed,
        "failed_module_count": integrity_failed,
        "scientific_metric_count": 0,
        "failure_count": integrity_failed,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
        "unaccounted_failure_count": sum(
            str(value.get("error_code", "")).endswith("UNACCOUNTED_FAILURE")
            for value in integrity_results
        ),
        "public_fixture_manifest_commitment_sha256": manifest[
            "public_fixture_manifest_commitment_sha256"
        ],
        "engineering_health_commitment_sha256": health_pass[
            "engineering_health_commitment_sha256"
        ],
        "runner_commitment_sha256": file_digest(Path(__file__).resolve()),
        "network_disabled": True,
        "telemetry_disabled": True,
        "restricted_mount_present": False,
    }
    integrity_full["partition_engineering_integrity_commitment_sha256"] = digest(
        integrity_full
    )
    _validate_tuple_partition_integrity_full(integrity_full)
    write_private_new(
        integrity_root / f"{partition}-attempt-{integrity_attempt:02d}.json",
        integrity_full,
    )
    if integrity_failed:
        return _tuple_partition_integrity_compact(integrity_full)

    scientific_root = args.scratch_root / f"{partition}-scientific"
    scientific_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    context = {
        **base_context,
        "scratch_root": scientific_root,
        "module_cache": {},
    }
    runners = _tuple_module_runners()
    if threshold_seal is not None:
        threshold_modules = {
            "referent",
            "recurrence",
            "attribute",
            "hand_contact",
            "order_action",
        }
        for module_id in threshold_modules:
            development_status = threshold_seal["development_module_statuses"][
                module_id
            ]
            if development_status != "PASS" and not (
                module_id == "order_action"
                and development_status == "NO_GO_DIAGNOSTIC"
            ):
                runners[module_id] = _development_unqualified_tuple_module_runner(
                    module_id
                )
    module_results, scientific_errors = _tuple_scientific_module_results(
        runners,
        context,
        integrity_root
        / f"{partition}-attempt-{integrity_attempt:02d}-scientific-traces",
    )
    if scientific_errors:
        errors_by_module = {
            row["module_id"]: row for row in scientific_errors
        }
        blocker_modules = [
            errors_by_module.get(row["module_id"], row)
            for row in integrity_results
        ]
        scientific_blocker = {
            **integrity_full,
            "status": "ENGINEERING_BLOCKER",
            "module_results": blocker_modules,
            "completed_module_count": len(TUPLE_QUALIFICATION_MODULE_IDS)
            - len(scientific_errors),
            "failed_module_count": len(scientific_errors),
            "failure_count": len(scientific_errors),
            "unaccounted_failure_count": sum(
                str(value.get("error_code", "")).endswith(
                    "UNACCOUNTED_FAILURE"
                )
                for value in scientific_errors
            ),
        }
        scientific_blocker.pop(
            "partition_engineering_integrity_commitment_sha256", None
        )
        scientific_blocker[
            "partition_engineering_integrity_commitment_sha256"
        ] = digest(scientific_blocker)
        _validate_tuple_partition_integrity_full(scientific_blocker)
        write_private_new(
            integrity_root
            / f"{partition}-attempt-{integrity_attempt:02d}-scientific-blocker.json",
            scientific_blocker,
        )
        return _tuple_partition_integrity_compact(scientific_blocker)
    try:
        full, seal, compact = _tuple_finalize_scientific_partition(
            cfg=cfg,
            active=active,
            correction=correction,
            manifest=manifest,
            partition=partition,
            no_hand_commitment=no_hand_commitment,
            threshold_seal=threshold_seal,
            integrity_results=integrity_results,
            module_results=module_results,
        )
        paths = _tuple_qualification_paths(args.public_root)
        transaction = _tuple_qualification_transaction(partition, full, seal)
        write_private_new(
            _tuple_qualification_transaction_path(args.public_root, partition),
            transaction,
        )
        write_private_new(paths[f"{partition}_result"], full)
        if seal is not None:
            write_private_new(paths["development_threshold_seal"], seal)
        return compact
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        finalize_trace_root = (
            integrity_root
            / f"{partition}-attempt-{integrity_attempt:02d}-finalize-traces"
        )
        blocker_modules = [
            _tuple_health_error(
                module_id,
                error
                if ordinal == 0
                else RuntimeError("E_TUPLE_HEALTH_PREFLIGHT_BLOCKED"),
                finalize_trace_root,
            )
            for ordinal, module_id in enumerate(TUPLE_QUALIFICATION_MODULE_IDS)
        ]
        finalize_blocker = {
            **integrity_full,
            "status": "ENGINEERING_BLOCKER",
            "module_results": blocker_modules,
            "completed_module_count": 0,
            "failed_module_count": len(blocker_modules),
            "failure_count": len(blocker_modules),
            "unaccounted_failure_count": sum(
                str(value["error_code"]).endswith("UNACCOUNTED_FAILURE")
                for value in blocker_modules
            ),
        }
        finalize_blocker.pop(
            "partition_engineering_integrity_commitment_sha256", None
        )
        finalize_blocker[
            "partition_engineering_integrity_commitment_sha256"
        ] = digest(finalize_blocker)
        _validate_tuple_partition_integrity_full(finalize_blocker)
        write_private_new(
            integrity_root
            / f"{partition}-attempt-{integrity_attempt:02d}-finalize-blocker.json",
            finalize_blocker,
        )
        return _tuple_partition_integrity_compact(finalize_blocker)


def prepare_tuple_audio_seed(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    preparation = _tuple_fixture_preparation_amendment(cfg)
    correction = _tuple_visor_hos_correction_amendment(cfg)
    audio_recipe = _tuple_referent_audio_fixture(cfg)
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
    nouns = {
        "sports ball": "Ball",
        "cup": "Becher",
        "bottle": "Flasche",
        "bowl": "Schüssel",
        "book": "Buch",
        "chair": "Stuhl",
        "apple": "Apfel",
        "banana": "Banane",
    }
    scenarios = preparation["referent_attribute_rendering"][
        "scenarios_once_per_category"
    ]
    root = args.output_root.resolve()
    active_parts = Path(audio_recipe["active_external_location"]).parts
    if tuple(root.parts[-len(active_parts) :]) != active_parts:
        raise RuntimeError("E_TUPLE_AUDIO_SEED_CANONICAL_ROOT")
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    records = []
    for partition in preparation["partitions"]:
        for category in preparation["public_object_ontology"]:
            noun = nouns[category]
            for ordinal, scenario in enumerate(scenarios):
                if scenario == "no_speech_visible_object":
                    continue
                phrase = _tuple_audio_phrase(
                    audio_recipe, partition, category, ordinal, noun
                )
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
        "schema_version": 2,
        "status": "SEALED_SELF_AUTHORED_PUBLIC_AUDIO_SEED",
        "preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "visor_hos_correction_amendment_commitment_sha256": correction[
            "amendment_commitment_sha256"
        ],
        "referent_audio_fixture_recipe": audio_recipe,
        "referent_audio_fixture_recipe_commitment_sha256": digest(audio_recipe),
        "active_external_location": audio_recipe["active_external_location"],
        "license": "CC0-1.0 self-authored text and rendered fixture audio",
        "language": "de",
        "voice": "macOS Anna de_DE",
        "rate_words_per_minute": 175,
        "muxed_speech_delay_seconds": audio_recipe[
            "muxed_speech_delay_seconds"
        ],
        "maximum_spoken_audio_seconds": audio_recipe[
            "maximum_spoken_audio_seconds"
        ],
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
    templates = development_templates if partition == "development" else holdout_templates
    output = []
    for category, (german, english) in nouns.items():
        for variant, template in enumerate(templates):
            text_en = template.format(noun=english)
            if variant == 0:
                expected = [{"token": english, "part_of_speech": "noun"}]
            elif variant == 1:
                expected = [
                    {"token": "red", "part_of_speech": "adjective"},
                    {"token": english, "part_of_speech": "noun"},
                ]
            elif variant == 2:
                expected = [
                    {"token": english, "part_of_speech": "noun"},
                    {"token": "blue", "part_of_speech": "adjective"},
                ]
            else:
                expected = [
                    {"token": english, "part_of_speech": "noun"},
                    {"token": english, "part_of_speech": "noun"},
                ]
            expected_adjective_noun_spans = (
                [{"adjective": "red", "noun": english}] if variant == 1 else []
            )
            expected = [
                {
                    **item,
                    "expected_lemma": item["token"],
                    "expected_frequency_band": "high",
                }
                for item in expected
            ]
            output.append(
                {
                    "case_id": f"{partition}-accept-{category.replace(' ', '-')}-{variant}",
                    "partition": partition,
                    "expected_adapter_status": "ACCEPT",
                    "expected_adapter_reason": None,
                    "expected_tuple_status": "ACCEPT",
                    "expected_tuple_reason": None,
                    "expected_grounding_status": "ACCEPT",
                    "expected_grounding_reason": None,
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
                    "expected_adjective_noun_spans": expected_adjective_noun_spans,
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
            adapter_status = (
                "ABSTAIN"
                if reason
                in {
                    "LANGUAGE_MISMATCH",
                    "EMPTY_ASR",
                    "INVALID_TIMESTAMP",
                    "LOW_CONFIDENCE",
                    "EMPTY_TRANSLATION",
                    "SILENT_TRUNCATION",
                }
                else "ACCEPT"
            )
            tuple_status = (
                "ABSTAIN"
                if adapter_status == "ABSTAIN"
                or reason == "INSUFFICIENT_IN_BOUNDS_FRAMES"
                else "ACCEPT"
            )
            expected_mentions = (
                [
                    {
                        "token": "cloud",
                        "part_of_speech": "noun",
                        "expected_lemma": "cloud",
                        "expected_frequency_band": "high",
                    }
                ]
                if reason == "ONTOLOGY_UNMATCHED"
                else []
            )
            output.append(
                {
                    "case_id": f"{partition}-abstain-{reason.casefold()}-{repeat}",
                    "partition": partition,
                    "expected_adapter_status": adapter_status,
                    "expected_adapter_reason": (
                        reason if adapter_status == "ABSTAIN" else None
                    ),
                    "expected_tuple_status": tuple_status,
                    "expected_tuple_reason": (
                        reason if tuple_status == "ABSTAIN" else None
                    ),
                    "expected_grounding_status": "ABSTAIN",
                    "expected_grounding_reason": reason,
                    "prediction": prediction,
                    "audio_duration": 7.0,
                    "translation_status": (
                        "ABSTAIN" if reason in {"EMPTY_TRANSLATION", "SILENT_TRUNCATION"} else "ACCEPT"
                    ),
                    "text_en": text_en,
                    "segment": segment,
                    "expected_lexical_mentions": expected_mentions,
                    "expected_adjective_noun_spans": [],
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


REFERENT_ATTRIBUTE_VALUES = (
    "red",
    "blue",
    "green",
    "yellow",
    "big",
    "small",
    "red",
    "blue",
)
REFERENT_ATTRIBUTE_SIZE_LONGEST_PIXELS = {"big": 130, "small": 72}
REFERENT_DISTRACTOR_SCENARIOS = frozenset(
    {
        "persistent_ambiguous",
        "persistent_dominant_with_small_distractor",
    }
)


def _referent_attribute_source_index(ordinal: int, source_count: int) -> int:
    """Keep the public big/small contrast paired on one exact source crop."""

    if source_count <= 0 or not 0 <= ordinal < len(REFERENT_ATTRIBUTE_VALUES):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_SOURCE_INDEX")
    if REFERENT_ATTRIBUTE_VALUES[ordinal] in {"big", "small"}:
        return 0
    return ordinal % source_count


def _referent_attribute_source_records(
    records: list[dict[str, Any]], ordinal: int, scenario: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Select a same-category, independently sourced distractor only when used."""

    target_index = _referent_attribute_source_index(ordinal, len(records))
    target = records[target_index]
    if scenario not in REFERENT_DISTRACTOR_SCENARIOS:
        return target, None
    if len(records) < 2:
        raise RuntimeError("E_TUPLE_DISTRACTOR_SOURCE_YIELD")
    distractor = records[(target_index + 1) % len(records)]
    required = ("category", "image_id", "annotation_id", "source_image_sha256")
    if any(key not in target or key not in distractor for key in required):
        raise RuntimeError("E_TUPLE_DISTRACTOR_SOURCE_IDENTITY")
    if target["category"] != distractor["category"]:
        raise RuntimeError("E_TUPLE_DISTRACTOR_CATEGORY_MISMATCH")
    if (
        target["image_id"] == distractor["image_id"]
        or target["annotation_id"] == distractor["annotation_id"]
        or target["source_image_sha256"] == distractor["source_image_sha256"]
    ):
        raise RuntimeError("E_TUPLE_DISTRACTOR_SOURCE_IDENTITY")
    return target, distractor


def _referent_attribute_episode_id(
    partition: str,
    category: str,
    scenario: str,
    ordinal: int,
    source_index: int,
) -> str:
    if not partition or not category or not scenario or not 0 <= ordinal < len(
        REFERENT_ATTRIBUTE_VALUES
    ):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_EPISODE_ID")
    if REFERENT_ATTRIBUTE_VALUES[ordinal] in {"big", "small"}:
        return f"{partition}|{category}|relative-size|{source_index}"
    return f"{partition}|{category}|{scenario}"


def _mask_fraction(mask: Any) -> float:
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("E_TUPLE_COMPOSITE_MASK_GEOMETRY")
    return float((array > 0).mean())


def _referent_attribute_phase_truth(
    target_masks: Any,
    distractor_masks: Any,
    *,
    fps: int,
    duration_seconds: float,
    utterance_start: float,
    utterance_end: float,
    definitions: dict[str, Any],
) -> dict[str, Any]:
    """Derive sampled phase truth from the retained authored masks.

    The exact masks are reference labels only.  They are never a permissible
    deterministic measurement input; model measurements must use a predicted
    SAM mask through ``_predicted_mask_attribute_measurements`` below.
    """

    import numpy as np

    targets = np.asarray(target_masks)
    distractors = np.asarray(distractor_masks)
    if (
        targets.ndim != 3
        or targets.shape != distractors.shape
        or targets.shape[0] < 1
        or fps <= 0
    ):
        raise RuntimeError("E_TUPLE_COMPOSITE_MASK_GEOMETRY")
    values = (
        float(duration_seconds),
        float(utterance_start),
        float(utterance_end),
    )
    if (
        not all(math.isfinite(value) for value in values)
        or values[0] <= 0.0
        or not 0.0 <= values[1] < values[2] <= values[0]
    ):
        raise RuntimeError("E_TUPLE_COMPOSITE_PHASE_BOUNDS")
    visible_floor = float(definitions["visible_mask_fraction_min"])
    dominant_floor = float(definitions["dominant_target_mask_fraction_min"])
    ratio_floor = float(definitions["dominant_area_ratio_to_next_candidate_min"])
    if not (
        0.0 <= visible_floor <= 1.0
        and 0.0 <= dominant_floor <= 1.0
        and math.isfinite(ratio_floor)
        and ratio_floor >= 1.0
    ):
        raise RuntimeError("E_TUPLE_COMPOSITE_PHASE_DEFINITION")
    requested = {
        "before": [utterance_start - 2.5, utterance_start - 1.5, utterance_start - 0.5],
        "during": [
            utterance_start
            + (index + 0.5) * (utterance_end - utterance_start) / 3.0
            for index in range(3)
        ],
        "after": [utterance_end + 0.5, utterance_end + 1.5, utterance_end + 2.5],
    }
    sample_rows: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in requested
    }
    maximum_frame_time = (targets.shape[0] - 1) / fps
    for phase, timestamps in requested.items():
        for timestamp in timestamps:
            if not 0.0 <= timestamp <= duration_seconds or timestamp > maximum_frame_time:
                continue
            frame_index = int(round(timestamp * fps))
            if not 0 <= frame_index < targets.shape[0]:
                continue
            target_fraction = _mask_fraction(targets[frame_index])
            distractor_fraction = _mask_fraction(distractors[frame_index])
            target_visible = target_fraction >= visible_floor
            candidate_count = int(target_visible) + int(
                distractor_fraction >= visible_floor
            )
            ratio = target_fraction / max(distractor_fraction, 1e-12)
            sample_rows[phase].append(
                {
                    "phase": phase,
                    "sample_time": round(float(timestamp), 6),
                    "frame_index": frame_index,
                    "target_visible": target_visible,
                    "candidate_count_bin": (
                        "2plus" if candidate_count >= 2 else str(candidate_count)
                    ),
                    "dominant": (
                        target_visible
                        and target_fraction >= dominant_floor
                        and ratio >= ratio_floor
                    ),
                    "target_fraction": target_fraction,
                    "distractor_fraction": distractor_fraction,
                }
            )
    sample_counts = {phase: len(rows) for phase, rows in sample_rows.items()}
    if sum(sample_counts.values()) < 8 or any(count < 2 for count in sample_counts.values()):
        raise RuntimeError("E_TUPLE_COMPOSITE_PHASE_COVERAGE")

    visibility: dict[str, bool] = {}
    candidates: dict[str, str] = {}
    dominance: dict[str, bool | None] = {}
    target_fraction_medians: dict[str, float] = {}
    distractor_fraction_medians: dict[str, float] = {}
    candidate_order = {"0": 0, "1": 1, "2plus": 2}
    for phase, rows in sample_rows.items():
        visibility[phase] = sum(row["target_visible"] for row in rows) >= 2
        counts = Counter(row["candidate_count_bin"] for row in rows)
        candidates[phase] = max(
            counts,
            key=lambda label: (counts[label], candidate_order[label]),
        )
        visible_rows = [row for row in rows if row["target_visible"]]
        dominance[phase] = (
            sum(row["dominant"] for row in visible_rows) >= 2
            if visibility[phase]
            else None
        )
        target_fraction_medians[phase] = round(
            float(statistics.median(row["target_fraction"] for row in rows)), 8
        )
        distractor_fraction_medians[phase] = round(
            float(statistics.median(row["distractor_fraction"] for row in rows)),
            8,
        )
    return {
        "visibility_by_phase": visibility,
        "candidate_count_by_phase": candidates,
        "dominance_by_phase": dominance,
        "sample_count_by_phase": sample_counts,
        "sampled_mask_truth": [
            row
            for phase in ("before", "during", "after")
            for row in sample_rows[phase]
        ],
        "target_mask_fraction_median_by_phase": target_fraction_medians,
        "distractor_mask_fraction_median_by_phase": distractor_fraction_medians,
    }


def _predicted_mask_attribute_measurements(
    image_array: Any,
    predicted_mask: Any,
    *,
    mask_role: str,
) -> dict[str, Any]:
    """Measure pixels only through an explicitly identified predicted SAM mask."""

    import numpy as np

    if mask_role != "predicted_SAM_mask":
        raise RuntimeError("E_TUPLE_ATTRIBUTE_REFERENCE_MASK_PROHIBITED")
    image = np.asarray(image_array)
    mask = np.asarray(predicted_mask)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or mask.ndim != 2
        or image.shape[:2] != mask.shape
        or image.size == 0
        or not np.isfinite(image).all()
        or not np.isfinite(mask).all()
    ):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_PREDICTED_MASK_GEOMETRY")
    selected = mask > 0
    if not selected.any():
        raise RuntimeError("E_TUPLE_ATTRIBUTE_PREDICTED_MASK_EMPTY")
    y_indices, x_indices = np.nonzero(selected)
    height, width = mask.shape
    median_rgb = np.median(image[selected].astype(np.float64), axis=0)
    if not np.isfinite(median_rgb).all():
        raise RuntimeError("E_TUPLE_ATTRIBUTE_PREDICTED_MASK_NONFINITE")
    box_width = int(x_indices.max() - x_indices.min() + 1)
    box_height = int(y_indices.max() - y_indices.min() + 1)
    return {
        "mask_fraction": round(float(selected.mean()), 8),
        "median_rgb": [round(float(value), 6) for value in median_rgb],
        "bbox_width_fraction": round(box_width / width, 8),
        "bbox_height_fraction": round(box_height / height, 8),
        "bbox_longest_side_fraction": round(
            max(box_width / width, box_height / height), 8
        ),
    }


def _render_referent_fixture(
    preparation: dict[str, Any],
    partition: str,
    category: str,
    scenario: str,
    ordinal: int,
    target_crop,
    distractor_crop,
    definitions: dict[str, Any],
) -> tuple[list[Any], Any, Any, dict[str, Any]]:
    import numpy as np
    from PIL import Image

    geometry = preparation["referent_attribute_rendering"]["geometry"]
    width, height = int(geometry["width"]), int(geometry["height"])
    frame_count = int(geometry["frames"])
    fps = int(geometry["fps"])
    if not 0 <= ordinal < len(REFERENT_ATTRIBUTE_VALUES):
        raise RuntimeError("E_TUPLE_ATTRIBUTE_ORDINAL")
    attribute = REFERENT_ATTRIBUTE_VALUES[ordinal]
    target = _tint_object(target_crop, attribute)
    if scenario in REFERENT_DISTRACTOR_SCENARIOS:
        if distractor_crop is None:
            raise RuntimeError("E_TUPLE_DISTRACTOR_SOURCE_MISSING")
        distractor = _tint_object(distractor_crop, "green")
    else:
        distractor = None
    target_size = REFERENT_ATTRIBUTE_SIZE_LONGEST_PIXELS.get(attribute, 104)
    background_identity = (
        f"{partition}|{category}|relative-size-pair"
        if attribute in {"big", "small"}
        else f"{partition}|{category}|{scenario}"
    )
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
    target_masks = np.zeros((frame_count, height, width), dtype=np.uint8)
    distractor_masks = np.zeros((frame_count, height, width), dtype=np.uint8)
    for frame_index in range(frame_count):
        timestamp = frame_index / fps
        phase = "before" if timestamp < 2.5 else "during" if timestamp <= 4.5 else "after"
        canvas = _fixture_background(
            width,
            height,
            int(preparation["seed"]),
            background_identity,
        ).convert("RGBA")
        target_mask = Image.new("L", (width, height), 0)
        distractor_mask = Image.new("L", (width, height), 0)
        if phase in visibility:
            sway = int(round(8 * math.sin(frame_index / 7.0)))
            _paste_masked_object(
                canvas, target_mask, target, (width // 2 + sway, height // 2), target_size
            )
        if scenario in REFERENT_DISTRACTOR_SCENARIOS:
            distractor_size = target_size if scenario == "persistent_ambiguous" else 45
            _paste_masked_object(
                canvas,
                distractor_mask,
                distractor,
                (width // 4, height // 2 + 35),
                distractor_size,
            )
        frames.append(np.asarray(canvas.convert("RGB"), dtype=np.uint8))
        target_masks[frame_index] = (
            np.asarray(target_mask, dtype=np.uint8) > 0
        ).astype(np.uint8)
        distractor_masks[frame_index] = (
            np.asarray(distractor_mask, dtype=np.uint8) > 0
        ).astype(np.uint8)
    phase_truth = _referent_attribute_phase_truth(
        target_masks,
        distractor_masks,
        fps=fps,
        duration_seconds=float(geometry["duration_seconds"]),
        utterance_start=float(
            preparation["referent_attribute_rendering"][
                "utterance_interval_seconds"
            ][0]
        ),
        utterance_end=float(
            preparation["referent_attribute_rendering"][
                "utterance_interval_seconds"
            ][1]
        ),
        definitions=definitions,
    )
    truth = {
        "attribute": attribute,
        "attribute_family": (
            "relative_size" if attribute in {"big", "small"} else "color"
        ),
        "target_longest_side_pixels": target_size,
        "background_identity_role": (
            "shared_relative_size_pair"
            if attribute in {"big", "small"}
            else "scenario_specific"
        ),
        "speech_present": scenario != "no_speech_visible_object",
        **phase_truth,
        "attribute_contrast_expected": (
            scenario != "no_speech_visible_object"
            and any(phase_truth["visibility_by_phase"].values())
        ),
        "attribute_null_reason": (
            "NO_ACCEPTED_ADJECTIVE_NOUN_SPAN"
            if scenario == "no_speech_visible_object"
            else "NO_PREDICTED_REFERENT_MASK"
            if not any(phase_truth["visibility_by_phase"].values())
            else None
        ),
        "reference_mask_role": "TRUTH_ONLY_NOT_A_MEASUREMENT_INPUT",
        "deterministic_measurement_mask_role": "predicted_SAM_mask",
    }
    return frames, target_masks, distractor_masks, truth


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
    output = target.with_suffix(".partial.mp4")
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
            f"anullsrc=r=22050:cl=mono,atrim=duration={duration}[base];"
            "[1:a]atrim=duration=2.0,adelay=2500[spoken];"
            "[base][spoken]amix=inputs=2:duration=first[a]"
        )
    else:
        audio_filter = (
            f"anullsrc=r=22050:cl=mono,atrim=duration={duration}[a]"
        )
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
) -> tuple[Any, Any, Any, Any]:
    from PIL import Image, ImageEnhance

    def normalized(crop):
        canvas = Image.new("RGBA", (224, 224), (112, 118, 125, 255))
        resized = crop.convert("RGBA").copy()
        resized.thumbnail((168, 168), Image.Resampling.LANCZOS)
        location = ((224 - resized.width) // 2, (224 - resized.height) // 2)
        canvas.alpha_composite(resized, location)
        mask = Image.new("L", canvas.size, 0)
        mask.paste(resized.getchannel("A"), location)
        mask = mask.point(lambda value: 255 if value >= 128 else 0, mode="L")
        if mask.getbbox() is None:
            raise RuntimeError("E_TUPLE_RECURRENCE_SOURCE_MASK_EMPTY")
        return canvas.convert("RGB"), mask

    first, first_mask = normalized(first_crop)
    second, second_mask = normalized(second_crop)
    if stratum == "same_instance_transformed":
        second = ImageEnhance.Brightness(first).enhance(0.75 if ordinal % 2 else 1.25)
        angle = 4 if ordinal % 2 else -4
        second = second.rotate(angle, resample=Image.Resampling.BILINEAR)
        second_mask = first_mask.rotate(
            angle, resample=Image.Resampling.NEAREST, fillcolor=0
        )
    elif stratum == "same_instance_near_duplicate":
        second = ImageEnhance.Brightness(first).enhance(0.98 if ordinal % 2 else 1.02)
        second_mask = first_mask.copy()
    for mask in (first_mask, second_mask):
        if mask.size != (224, 224) or mask.getbbox() is None or set(
            value for value, count in enumerate(mask.histogram()) if count
        ) - {0, 255}:
            raise RuntimeError("E_TUPLE_RECURRENCE_MASK")
    return first, second, first_mask, second_mask


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
                headers = getattr(response, "headers", {})
                expected_text = headers.get("Content-Length") if headers else None
                expected_bytes = None
                if expected_text is not None:
                    try:
                        expected_bytes = int(expected_text)
                    except (TypeError, ValueError) as error:
                        raise RuntimeError(
                            "E_TUPLE_PUBLIC_FILE_CONTENT_LENGTH"
                        ) from error
                    if expected_bytes < 0:
                        raise RuntimeError("E_TUPLE_PUBLIC_FILE_CONTENT_LENGTH")
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if (
                expected_bytes is not None
                and partial.stat().st_size != expected_bytes
            ):
                partial.unlink(missing_ok=True)
                if attempt == 4:
                    raise RuntimeError("E_TUPLE_PUBLIC_FILE_TRUNCATED")
                time.sleep(2**attempt)
                continue
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
    root: Path, cfg: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str, str], Path]]:
    preparation = _tuple_fixture_preparation_amendment(cfg)
    correction = _tuple_visor_hos_correction_amendment(cfg)
    recipe = _tuple_referent_audio_fixture(cfg)
    active_parts = Path(recipe["active_external_location"]).parts
    if tuple(root.resolve().parts[-len(active_parts) :]) != active_parts:
        raise RuntimeError("E_TUPLE_AUDIO_SEED_CANONICAL_ROOT")
    path = root / "audio-seed-manifest.json"
    manifest = json.loads(path.read_text())
    commitment = manifest.pop("audio_seed_commitment_sha256", None)
    if (
        not isinstance(commitment, str)
        or digest(manifest) != commitment
        or manifest.get("schema_version") != 2
        or manifest.get("preparation_amendment_commitment_sha256")
        != preparation["preparation_amendment_commitment_sha256"]
        or manifest.get("visor_hos_correction_amendment_commitment_sha256")
        != correction["amendment_commitment_sha256"]
        or manifest.get("referent_audio_fixture_recipe") != recipe
        or manifest.get("referent_audio_fixture_recipe_commitment_sha256")
        != digest(recipe)
        or manifest.get("active_external_location")
        != recipe["active_external_location"]
        or manifest.get("language") != recipe["language"]
        or manifest.get("voice") != "macOS Anna de_DE"
        or manifest.get("rate_words_per_minute") != 175
        or manifest.get("muxed_speech_delay_seconds")
        != recipe["muxed_speech_delay_seconds"]
        or manifest.get("maximum_spoken_audio_seconds")
        != recipe["maximum_spoken_audio_seconds"]
        or manifest.get("audio_file_count") != 112
        or manifest.get("status") != "SEALED_SELF_AUTHORED_PUBLIC_AUDIO_SEED"
    ):
        raise RuntimeError("E_TUPLE_AUDIO_SEED_MANIFEST")
    manifest["audio_seed_commitment_sha256"] = commitment
    nouns = {
        "sports ball": "Ball",
        "cup": "Becher",
        "bottle": "Flasche",
        "bowl": "Schüssel",
        "book": "Buch",
        "chair": "Stuhl",
        "apple": "Apfel",
        "banana": "Banane",
    }
    scenarios = preparation["referent_attribute_rendering"][
        "scenarios_once_per_category"
    ]
    expected: dict[tuple[str, str, str], tuple[int, str, str]] = {}
    for partition in preparation["partitions"]:
        for category in preparation["public_object_ontology"]:
            for ordinal, scenario in enumerate(scenarios):
                if scenario == "no_speech_visible_object":
                    continue
                slug = category.replace(" ", "-")
                expected[(partition, category, scenario)] = (
                    ordinal,
                    _tuple_audio_phrase(
                        recipe, partition, category, ordinal, nouns[category]
                    ),
                    f"{partition}/{slug}-{ordinal:02d}.aiff",
                )
    records = {}
    rows = manifest.get("records")
    if not isinstance(rows, list):
        raise RuntimeError("E_TUPLE_AUDIO_SEED_MANIFEST")
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("E_TUPLE_AUDIO_SEED_MANIFEST")
        key = (row.get("partition"), row.get("category"), row.get("scenario"))
        expected_row = expected.get(key)
        if expected_row is None:
            raise RuntimeError("E_TUPLE_AUDIO_SEED_RECIPE")
        ordinal, phrase, expected_relative = expected_row
        if (
            row.get("ordinal") != ordinal
            or row.get("phrase_de") != phrase
            or row.get("relative_path") != expected_relative
        ):
            raise RuntimeError("E_TUPLE_AUDIO_SEED_RECIPE")
        relative = Path(str(row.get("relative_path", "")))
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or any(not part or part in {".", ".."} for part in relative.parts)
        ):
            raise RuntimeError("E_TUPLE_AUDIO_SEED_PATH")
        source = root / relative
        if (
            not source.is_file()
            or file_digest(source) != row.get("sha256")
            or source.stat().st_size != int(row.get("bytes", -1))
        ):
            raise RuntimeError("E_TUPLE_AUDIO_SEED_HASH")
        if key in records:
            raise RuntimeError("E_TUPLE_AUDIO_SEED_DUPLICATE")
        records[key] = source
    if set(records) != set(expected):
        raise RuntimeError("E_TUPLE_AUDIO_SEED_COUNT")
    return manifest, records


def _load_charades_rows(annotation_root: Path) -> list[dict[str, str]]:
    output = []
    # The paired CSVs contain both the third-person source and its first-person
    # partner.  The action controls are explicitly first-person, so load the
    # official first-person-only tables rather than accepting every row merely
    # because its `egocentric` link field is populated.
    for name in (
        "CharadesEgo_v1_train_only1st.csv",
        "CharadesEgo_v1_test_only1st.csv",
    ):
        candidates = list(annotation_root.rglob(name))
        if len(candidates) != 1:
            raise RuntimeError("E_TUPLE_ACTION_ANNOTATION_FILE")
        with candidates[0].open(newline="", encoding="utf-8") as handle:
            output.extend(dict(row) for row in csv.DictReader(handle))
    return output


def _reconstruct_prior_activity_selection(
    rows: list[dict[str, str]], fixture: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Recreate the frozen broad-context fixture identities from public CSVs.

    The original row/file manifest remains external.  Its exact public seed,
    code map, partition rule and greedy selection rule are retained in the
    canonical config, which is sufficient to reconstruct the exclusion set
    without reopening any model outcome or depending on the missing copy.
    """

    seed = str(int(fixture["seed"]))
    label_code_map = fixture["label_code_map"]
    if not isinstance(label_code_map, dict) or len(label_code_map) != 8:
        raise RuntimeError("E_TUPLE_PRIOR_ACTIVITY_LABEL_MAP")
    code_sets = {
        str(label): {str(code) for code in codes}
        for label, codes in label_code_map.items()
    }
    candidates = []
    for row in rows:
        try:
            duration = float(row["length"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (
            row.get("verified") != "Yes"
            or not math.isfinite(duration)
            or not 5.0 <= duration <= 60.0
        ):
            continue
        codes = {
            item.split()[0]
            for item in str(row.get("actions", "")).split(";")
            if item.strip()
        }
        labels = sorted(
            label for label, values in code_sets.items() if codes & values
        )
        video = str(row.get("id", ""))
        subject = str(row.get("subject", ""))
        if not labels or not video or not subject:
            continue
        partition_hash = hashlib.sha256(
            f"{seed}|partition|{subject}".encode()
        ).hexdigest()
        partition = (
            "development" if int(partition_hash, 16) % 2 == 0 else "holdout"
        )
        candidates.append(
            {
                "id": video,
                "subject": subject,
                "labels": labels,
                "partition": partition,
            }
        )
    selected: dict[str, list[dict[str, Any]]] = {}
    item_counts = {
        "development": int(fixture["development_items"]),
        "holdout": int(fixture["holdout_items"]),
    }
    for partition, item_count in item_counts.items():
        pool = [row for row in candidates if row["partition"] == partition]
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[str] = set()
        label_counts: Counter[str] = Counter()
        while len(chosen) < item_count:
            deficits = {
                label: max(0, 6 - label_counts[label]) for label in code_sets
            }

            def selection_key(row: dict[str, Any]) -> tuple[int, int, str]:
                return (
                    -sum(deficits[label] > 0 for label in row["labels"]),
                    -sum(deficits[label] for label in row["labels"]),
                    hashlib.sha256(
                        f"{seed}|{partition}|{row['id']}".encode()
                    ).hexdigest(),
                )

            remaining = [row for row in pool if row["id"] not in chosen_ids]
            if not remaining:
                raise RuntimeError("E_TUPLE_PRIOR_ACTIVITY_SELECTION_YIELD")
            item = min(remaining, key=selection_key)
            chosen.append(item)
            chosen_ids.add(item["id"])
            label_counts.update(item["labels"])
        selected[partition] = chosen
    return selected


def _prior_activity_exclusions(
    annotation_root: Path, cfg: dict[str, Any]
) -> tuple[set[str], set[str], dict[str, Any]]:
    try:
        fixture = cfg["calibration_C"]["extractor"][
            "activity_checkpoint_selection_amendment"
        ]["public_activity_fixture"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_PRIOR_ACTIVITY_FIXTURE") from error
    selected = _reconstruct_prior_activity_selection(
        _load_charades_rows(annotation_root), fixture
    )
    expected_subjects = {
        "development": int(fixture["development_subjects"]),
        "holdout": int(fixture["holdout_subjects"]),
    }
    expected_minimum = {
        "development": int(
            fixture["development_minimum_positive_count_across_labels"]
        ),
        "holdout": int(
            fixture["holdout_minimum_positive_count_across_labels"]
        ),
    }
    labels = set(fixture["label_code_map"])
    counts: dict[str, dict[str, int]] = {}
    for partition, rows in selected.items():
        label_counts = Counter(
            label for row in rows for label in row["labels"]
        )
        counts[partition] = {
            "item_count": len(rows),
            "subject_count": len({row["subject"] for row in rows}),
            "minimum_positive_count": min(
                (label_counts[label] for label in labels), default=0
            ),
        }
        if (
            counts[partition]["item_count"]
            != int(fixture[f"{partition}_items"])
            or counts[partition]["subject_count"] != expected_subjects[partition]
            or counts[partition]["minimum_positive_count"]
            != expected_minimum[partition]
        ):
            raise RuntimeError("E_TUPLE_PRIOR_ACTIVITY_RECONSTRUCTION")
    subjects = {
        partition: {row["subject"] for row in rows}
        for partition, rows in selected.items()
    }
    videos = {
        partition: {row["id"] for row in rows}
        for partition, rows in selected.items()
    }
    if (
        subjects["development"] & subjects["holdout"]
        or videos["development"] & videos["holdout"]
    ):
        raise RuntimeError("E_TUPLE_PRIOR_ACTIVITY_OVERLAP")
    manifest_commitment = fixture.get("manifest_commitment_sha256")
    if not isinstance(manifest_commitment, str) or not re.fullmatch(
        r"[0-9a-f]{64}", manifest_commitment
    ):
        raise RuntimeError("E_TUPLE_PRIOR_ACTIVITY_COMMITMENT")
    identity_commitment = digest(
        {
            partition: [
                {"id": row["id"], "subject": row["subject"]}
                for row in selected[partition]
            ]
            for partition in ("development", "holdout")
        }
    )
    record = {
        "status": "PASS_FROZEN_PRIOR_ACTIVITY_EXCLUSIONS_RECONSTRUCTED",
        "frozen_manifest_commitment_sha256": manifest_commitment,
        "selection_identity_commitment_sha256": identity_commitment,
        "counts": counts,
        "subject_overlap_count": 0,
        "video_overlap_count": 0,
    }
    return (
        subjects["development"] | subjects["holdout"],
        videos["development"] | videos["holdout"],
        record,
    )


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


def _download_sized_public_file(url: str, target: Path, expected_bytes: int) -> None:
    """Download one official public file, retaining valid resumable state."""

    if expected_bytes <= 0:
        raise RuntimeError("E_VISOR_HOS_RESOURCE_SIZE")
    if target.is_file() and target.stat().st_size == expected_bytes:
        return
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.is_file() and partial.stat().st_size == expected_bytes:
        os.chmod(partial, 0o600)
        partial.replace(target)
        return
    if partial.is_file() and partial.stat().st_size > expected_bytes:
        partial.unlink()
    for attempt in range(5):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "synthetic-video-research/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = int(getattr(response, "status", response.getcode()))
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size == expected_bytes:
                break
            if partial.stat().st_size > expected_bytes:
                partial.unlink()
        except (OSError, urllib.error.URLError):
            if attempt == 4:
                raise
        if attempt == 4:
            raise RuntimeError("E_VISOR_HOS_RESOURCE_SIZE")
        time.sleep(2**attempt)
    if not partial.is_file() or partial.stat().st_size != expected_bytes:
        raise RuntimeError("E_VISOR_HOS_RESOURCE_SIZE")
    os.chmod(partial, 0o600)
    partial.replace(target)


def _visor_hos_resource_rows(
    package: dict[str, Any], split: str, frozen: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate one pinned official CKAN package without inferring names."""

    result = package.get("result") if package.get("success") is True else None
    resources = result.get("resources") if isinstance(result, dict) else None
    if (
        not isinstance(resources, list)
        or result.get("name") != frozen.get("dataset_id")
        or result.get("revision_id") != frozen.get("revision_id")
        or len(resources) != int(frozen.get("JSON_file_count", -1))
    ):
        raise RuntimeError("E_VISOR_HOS_CATALOG_REVISION")
    output = []
    seen: set[str] = set()
    for resource in resources:
        if not isinstance(resource, dict):
            raise RuntimeError("E_VISOR_HOS_CATALOG_RESOURCE")
        name = str(resource.get("name", ""))
        url = str(resource.get("url", ""))
        try:
            size = int(resource["size"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise RuntimeError("E_VISOR_HOS_CATALOG_RESOURCE") from error
        parsed = urllib.parse.urlsplit(url)
        expected_path = (
            "/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/"
            "GroundTruth-SparseAnnotations/annotations/"
            f"{split}/{urllib.parse.quote(name)}"
        )
        if (
            name in seen
            or not re.fullmatch(r"P\d{2}_\d{2,3}\.json", name)
            or Path(name).name != name
            or parsed.scheme != "https"
            or parsed.hostname != "data.bris.ac.uk"
            or parsed.path != expected_path
            or parsed.query
            or parsed.fragment
            or size <= 0
            or resource.get("hash") not in (None, "")
        ):
            raise RuntimeError("E_VISOR_HOS_CATALOG_RESOURCE")
        seen.add(name)
        output.append(
            {
                "split": split,
                "name": name,
                "url": url,
                "bytes": size,
                "resource_id": str(resource.get("id", "")),
            }
        )
    if sum(row["bytes"] for row in output) != int(frozen.get("bytes", -1)):
        raise RuntimeError("E_VISOR_HOS_CATALOG_BYTES")
    return sorted(output, key=lambda row: row["name"].encode("utf-8"))


def _visor_hos_annotation_commitments(
    artifact_root: Path, paths: list[Path]
) -> tuple[str, str, bytes, list[dict[str, Any]]]:
    """Compute the two frozen locally resolved 158-file commitments exactly."""

    ordered = sorted(paths, key=lambda path: path.relative_to(artifact_root).as_posix().encode("utf-8"))
    shasum_lines: list[bytes] = []
    framed = hashlib.sha256()
    records = []
    for path in ordered:
        relative = path.relative_to(artifact_root).as_posix()
        relative_bytes = relative.encode("utf-8")
        size = path.stat().st_size
        sha256 = file_digest(path)
        shasum_lines.append(f"{sha256}  {relative}\n".encode("ascii"))
        framed.update(relative_bytes)
        framed.update(b"\0")
        framed.update(str(size).encode("ascii"))
        framed.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                framed.update(chunk)
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256,
                "bytes": size,
            }
        )
    manifest_bytes = b"".join(shasum_lines)
    return (
        hashlib.sha256(manifest_bytes).hexdigest(),
        framed.hexdigest(),
        manifest_bytes,
        records,
    )


def _load_official_visor_hos_annotation_paths(
    fixture_root: Path, amendment: dict[str, Any]
) -> tuple[list[Path], dict[str, Any]]:
    """Resolve and verify exactly the pinned official train+validation set."""

    frozen = amendment["official_annotation_artifact"]
    source_root = fixture_root / "sources/VISOR-HOS"
    artifact_root = source_root / "annotations"
    catalog_root = source_root / "catalog"
    all_rows: list[dict[str, Any]] = []
    catalog_records = []
    for split, key in (("train", "train_resource"), ("val", "validation_resource")):
        split_frozen = frozen[key]
        package_url = (
            "https://data.bris.ac.uk/data/api/3/action/package_show?id="
            + split_frozen["dataset_id"]
        )
        catalog_path = catalog_root / f"{split}-package.json"
        _download_public_file(package_url, catalog_path)
        package = json.loads(catalog_path.read_text())
        rows = _visor_hos_resource_rows(package, split, split_frozen)
        all_rows.extend(rows)
        catalog_records.append(
            {
                "split": split,
                "dataset_id": split_frozen["dataset_id"],
                "revision_id": split_frozen["revision_id"],
                "catalog_sha256": file_digest(catalog_path),
                "resource_count": len(rows),
                "resource_bytes": sum(row["bytes"] for row in rows),
            }
        )
    expected_paths = {
        artifact_root / row["split"] / row["name"] for row in all_rows
    }
    if len(expected_paths) != int(frozen["combined_JSON_file_count"]):
        raise RuntimeError("E_VISOR_HOS_RESOURCE_SET")
    for row in all_rows:
        target = artifact_root / row["split"] / row["name"]
        _download_sized_public_file(row["url"], target, int(row["bytes"]))
    discovered = set()
    for split in ("train", "val"):
        root = artifact_root / split
        if root.is_dir():
            discovered.update(root.glob("*.json"))
    if discovered != expected_paths:
        raise RuntimeError("E_VISOR_HOS_RESOURCE_SET")
    paths = sorted(
        expected_paths,
        key=lambda path: path.relative_to(artifact_root).as_posix().encode("utf-8"),
    )
    if (
        len(paths) != int(frozen["combined_JSON_file_count"])
        or sum(path.stat().st_size for path in paths) != int(frozen["combined_bytes"])
    ):
        raise RuntimeError("E_VISOR_HOS_RESOURCE_SET")
    manifest_commitment, framed_commitment, manifest_bytes, records = (
        _visor_hos_annotation_commitments(artifact_root, paths)
    )
    if manifest_commitment != frozen[
        "external_sorted_relative_path_and_SHA256_manifest_commitment_sha256"
    ]:
        raise RuntimeError("E_VISOR_HOS_SHASUM_MANIFEST_COMMITMENT")
    if framed_commitment != frozen[
        "external_path_size_and_bytes_framed_content_commitment_sha256"
    ]:
        raise RuntimeError("E_VISOR_HOS_BYTES_FRAMED_COMMITMENT")
    manifest_path = source_root / "sha256_manifest.txt"
    manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path.write_bytes(manifest_bytes)
    os.chmod(manifest_path, 0o600)
    provenance = {
        "schema_version": 1,
        "status": "PASS_EXACT_OFFICIAL_ARTIFACT",
        "doi": frozen["doi"],
        "catalogs": catalog_records,
        "files": records,
        "file_count": len(paths),
        "bytes": sum(path.stat().st_size for path in paths),
        "shasum_manifest_commitment_sha256": manifest_commitment,
        "bytes_framed_content_commitment_sha256": framed_commitment,
        "locally_resolved_not_author_published_hashes": True,
    }
    provenance["artifact_provenance_commitment_sha256"] = digest(provenance)
    write_private(source_root / "artifact-provenance.json", provenance)
    return paths, provenance


def _load_visor_hos_correction_exclusions(
    fixture_root: Path, amendment: dict[str, Any]
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Pin the unlicensed reference bytes and use only its exclusion keys."""

    reference = amendment["official_semantic_reference"]
    required_file_hashes = {
        "data_preparation/gen_coco_format.py": reference.get(
            "gen_coco_format_py_sha256"
        ),
        "data_preparation/gen_coco_format_handside_contact.py": reference.get(
            "gen_coco_format_handside_contact_py_sha256"
        ),
        "data_preparation/correct.json": reference.get("correct_json_sha256"),
        "README.md": reference.get("README_sha256"),
    }
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in required_file_hashes.values()):
        raise RuntimeError("E_VISOR_HOS_SEMANTIC_HASH_SET")
    source_root = fixture_root / "sources/VISOR-HOS/semantic-reference"
    archive = source_root / f"VISOR-HOS-{reference['commit']}.tar.gz"
    archive_url = (
        "https://codeload.github.com/epic-kitchens/VISOR-HOS/tar.gz/"
        + reference["commit"]
    )
    _download_exact_public_artifact(archive_url, archive, reference["archive_sha256"])
    extracted_bytes: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as handle:
        for relative, expected in required_file_hashes.items():
            matches = [
                member
                for member in handle.getmembers()
                if member.isfile() and member.name.endswith("/" + relative)
            ]
            if len(matches) != 1:
                raise RuntimeError("E_VISOR_HOS_SEMANTIC_MEMBER_SET")
            stream = handle.extractfile(matches[0])
            if stream is None:
                raise RuntimeError("E_VISOR_HOS_SEMANTIC_MEMBER_SET")
            payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError("E_VISOR_HOS_SEMANTIC_MEMBER_HASH")
            extracted_bytes[relative] = payload
    correction = json.loads(
        extracted_bytes["data_preparation/correct.json"].decode("utf-8")
    )
    if not isinstance(correction, dict) or not correction:
        raise RuntimeError("E_VISOR_HOS_CORRECTION_TABLE")
    exclusions: set[tuple[str, str]] = set()
    for frame_name, values in correction.items():
        match = re.fullmatch(r"(P\d{2}_\d{2,3})_frame_\d+\.jpg", frame_name)
        if match is None or not isinstance(values, dict) or not values:
            raise RuntimeError("E_VISOR_HOS_CORRECTION_TABLE")
        exclusions.add((match.group(1), frame_name))
    record = {
        "schema_version": 1,
        "status": "PASS_REFERENCE_ONLY_EXCLUSION_KEYS",
        "repository": reference["repository"],
        "commit": reference["commit"],
        "archive_sha256": reference["archive_sha256"],
        "required_file_hashes": required_file_hashes,
        "code_terms": reference["code_terms"],
        "repository_code_imported_or_executed": False,
        "correction_values_applied": False,
        "correction_frame_count": len(exclusions),
    }
    record["semantic_reference_commitment_sha256"] = digest(record)
    write_private(source_root / "semantic-reference.json", record)
    return exclusions, record


def _iter_visor_hos_documents(paths: list[Path]):
    for path in paths:
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or not isinstance(
            document.get("video_annotations"), list
        ):
            raise RuntimeError("E_VISOR_HOS_ANNOTATION_SCHEMA")
        source_split = path.parent.name
        if source_split not in {"train", "val"}:
            raise RuntimeError("E_VISOR_HOS_ANNOTATION_SPLIT")
        for row in document["video_annotations"]:
            image = row.get("image") if isinstance(row, dict) else None
            if not isinstance(image, dict) or "_source_split" in image:
                raise RuntimeError("E_VISOR_HOS_ANNOTATION_SCHEMA")
            image["_source_split"] = source_split
        yield document


def _visor_hos_source_inventory(
    annotation_documents,
    *,
    seed: int,
    target_per_stratum: int,
    no_hand_review_queue_ceiling: int,
    per_video_stratum_cap: int,
    correction_excluded_frame_keys: set[tuple[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Allocate source-qualified contact states and unverified negatives once.

    The no-hand branch deliberately remains `no_hand_nominee`.  This source
    check can establish that a bounded review queue exists, but only the later
    blinded visual-review seal can convert any nominee into ground truth.
    """

    strata = ("contact", "explicit_no_contact", "no_hand_nominee")
    quotas = {
        "contact": target_per_stratum,
        "explicit_no_contact": target_per_stratum,
        "no_hand_nominee": no_hand_review_queue_ceiling,
    }
    minimums = {stratum: target_per_stratum for stratum in strata}
    if (
        target_per_stratum <= 0
        or no_hand_review_queue_ceiling < target_per_stratum
        or per_video_stratum_cap <= 0
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_LIMIT")
    correction_excluded = _visor_hos_frame_key_set(
        correction_excluded_frame_keys, "E_VISOR_HOS_CORRECTION_EXCLUSION_KEY"
    )
    participants: set[str] = set()
    source_frame_keys: set[tuple[str, str]] = set()
    parsed_all: list[dict[str, Any]] = []
    invalid_frame_count = 0
    abstained_frame_count = 0
    abstained_hand_count = 0
    for document in annotation_documents:
        rows = document.get("video_annotations")
        if not isinstance(rows, list):
            invalid_frame_count += 1
            continue
        for row in rows:
            image = row.get("image") if isinstance(row, dict) else None
            video = str(image.get("video", "")) if isinstance(image, dict) else ""
            frame_name = str(image.get("name", "")) if isinstance(image, dict) else ""
            participant = video.split("_", 1)[0]
            if re.fullmatch(r"P\d{2}", participant):
                participants.add(participant)
            if video and frame_name and Path(frame_name).name == frame_name:
                source_frame_keys.add((video, frame_name))
            result = _visor_hos_frame_candidates(row) if isinstance(row, dict) else {
                "status": "INVALID",
                "candidates": [],
                "abstained_hand_count": 0,
            }
            if result["status"] == "INVALID":
                invalid_frame_count += 1
                continue
            abstained_hand_count += int(result.get("abstained_hand_count", 0))
            if result["status"] == "ABSTAIN":
                abstained_frame_count += 1
            for candidate in result.get("candidates", []):
                frame_key = (candidate["video"], candidate["frame_name"])
                if frame_key in correction_excluded:
                    continue
                parsed_all.append(candidate)
    partitions = _visor_hos_participant_partitions(participants, seed)
    raw_eligible = {
        stratum: sum(row["stratum"] == stratum for row in parsed_all)
        for stratum in strata
    }
    selected: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    post_partition: dict[str, dict[str, int]] = {}
    post_cap: dict[str, dict[str, int]] = {}
    final: dict[str, dict[str, int]] = {}
    deficits: list[dict[str, Any]] = []
    for partition in ("development", "holdout"):
        candidates = [
            row for row in parsed_all if partitions[row["participant"]] == partition
        ]
        post_partition[partition] = {
            stratum: sum(row["stratum"] == stratum for row in candidates)
            for stratum in strata
        }
        by_stratum_video: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        representative: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
        for row in candidates:
            frame = (row["video"], row["frame_name"])
            stratum = row["stratum"]
            by_stratum_video[(stratum, row["video"])].add(frame)
            key = (stratum, frame)
            prior = representative.get(key)
            if prior is None or _visor_hos_candidate_order(
                seed, partition, row
            ) < _visor_hos_candidate_order(seed, partition, prior):
                representative[key] = row
        post_cap[partition] = {
            stratum: sum(
                min(len(frames), per_video_stratum_cap)
                for (candidate_stratum, _video), frames in by_stratum_video.items()
                if candidate_stratum == stratum
            )
            for stratum in strata
        }
        source = ("source", partition)
        sink = ("sink", partition)
        graph: dict[Any, list[list[Any]]] = defaultdict(list)
        stratum_nodes = {stratum: ("stratum", stratum) for stratum in strata}
        for stratum in sorted(strata):
            _visor_hos_add_flow_edge(
                graph, source, stratum_nodes[stratum], quotas[stratum]
            )
        all_frames = sorted(
            {(row["video"], row["frame_name"]) for row in candidates},
            key=lambda frame: min(
                _visor_hos_candidate_order(seed, partition, row)
                for row in candidates
                if (row["video"], row["frame_name"]) == frame
            ),
        )
        for frame in all_frames:
            _visor_hos_add_flow_edge(graph, ("frame", *frame), sink, 1)
        video_nodes: dict[tuple[str, str], Any] = {}
        for key in sorted(by_stratum_video):
            stratum, video = key
            node = ("video_stratum", stratum, video)
            video_nodes[key] = node
            _visor_hos_add_flow_edge(
                graph, stratum_nodes[stratum], node, per_video_stratum_cap
            )
            for frame in sorted(
                by_stratum_video[key],
                key=lambda item: _visor_hos_candidate_order(
                    seed, partition, representative[(stratum, item)]
                ),
            ):
                _visor_hos_add_flow_edge(graph, node, ("frame", *frame), 1)
        _visor_hos_max_flow(graph, source, sink)
        for (stratum, _video), node in video_nodes.items():
            for target, _reverse, capacity in graph[node]:
                if (
                    isinstance(target, tuple)
                    and target[:1] == ("frame",)
                    and capacity == 0
                ):
                    frame = (target[1], target[2])
                    selected[partition].append(representative[(stratum, frame)])
        selected[partition].sort(
            key=lambda row: _visor_hos_candidate_order(seed, partition, row)
        )
        final[partition] = {
            stratum: sum(row["stratum"] == stratum for row in selected[partition])
            for stratum in strata
        }
        for stratum in strata:
            if final[partition][stratum] < minimums[stratum]:
                deficits.append(
                    {
                        "partition": partition,
                        "stratum": stratum,
                        "required_count": minimums[stratum],
                        "available_count": final[partition][stratum],
                    }
                )
    development_participants = {
        row["participant"] for row in selected["development"]
    }
    holdout_participants = {row["participant"] for row in selected["holdout"]}
    development_videos = {row["video"] for row in selected["development"]}
    holdout_videos = {row["video"] for row in selected["holdout"]}
    development_frames = {
        (row["video"], row["frame_name"]) for row in selected["development"]
    }
    holdout_frames = {
        (row["video"], row["frame_name"]) for row in selected["holdout"]
    }
    correction_matches = source_frame_keys & correction_excluded
    overlap = {
        "participant_overlap_count": len(
            development_participants & holdout_participants
        ),
        "video_overlap_count": len(development_videos & holdout_videos),
        "frame_overlap_count": len(development_frames & holdout_frames),
    }
    integrity_failures = []
    if len(correction_matches) != len(correction_excluded):
        integrity_failures.append("CORRECTION_FRAME_NOT_FOUND")
    if any(overlap.values()):
        integrity_failures.append("CROSS_PARTITION_OVERLAP")
    report = {
        "status": "PASS_SOURCE_NOMINEES" if not deficits and not integrity_failures else "NO_GO",
        "raw_eligible_counts": raw_eligible,
        "post_partition_counts": post_partition,
        "post_cap_counts": post_cap,
        "final_counts": final,
        "deficits": deficits,
        "integrity_failures": integrity_failures,
        "invalid_frame_count": invalid_frame_count,
        "abstained_frame_count": abstained_frame_count,
        "abstained_hand_count": abstained_hand_count,
        "no_hand_items_are_unverified_nominees": True,
        "no_hand_review_queue_ceiling": no_hand_review_queue_ceiling,
        "correction_table_frame_count": len(correction_excluded),
        "correction_excluded_source_frame_count": len(correction_matches),
        "correction_values_applied": False,
        **overlap,
    }
    return selected, report


def _charades_action_source_inventory(
    rows: list[dict[str, str]],
    action: dict[str, Any],
    seed: int,
    excluded_subjects: set[str],
    excluded_videos: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Collect every frozen direction-label yield before one family decision."""

    code_to_label = _charades_direction_map(action)
    opposite = {
        first: second
        for pair in action["class_code_pairs"]
        for first, second in (pair["pair"], tuple(reversed(pair["pair"])))
    }
    candidates = {
        partition: {label: [] for label in action["labels"]}
        for partition in ("development", "holdout")
    }
    invalid_row_count = 0
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
            annotations = _parse_charades_actions(row.get("actions", ""))
        except (KeyError, TypeError, ValueError, RuntimeError):
            invalid_row_count += 1
            continue
        partition = _fixture_partition(
            seed, "mechanistic_action_partition", subject
        )
        for item in annotations:
            label = code_to_label.get(item["code"])
            if label is None:
                continue
            start = max(0.0, item["start"])
            end = min(duration, item["end"])
            if not 1.0 <= end - start <= 12.0:
                continue
            if any(
                code_to_label.get(other["code"]) == opposite[label]
                and max(start, other["start"]) < min(end, other["end"])
                for other in annotations
            ):
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
    selected = {"development": [], "holdout": []}
    raw_counts: dict[str, dict[str, int]] = {}
    final_counts: dict[str, dict[str, int]] = {}
    deficits = []
    for partition in ("development", "holdout"):
        raw_counts[partition] = {
            label: len(candidates[partition][label]) for label in action["labels"]
        }
        used: set[str] = set()
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
        final_counts[partition] = {
            label: sum(row["label"] == label for row in selected[partition])
            for label in action["labels"]
        }
        for label in action["labels"]:
            if final_counts[partition][label] != 6:
                deficits.append(
                    {
                        "partition": partition,
                        "label": label,
                        "required_count": 6,
                        "available_count": final_counts[partition][label],
                    }
                )
    development_subjects = {row["subject"] for row in selected["development"]}
    holdout_subjects = {row["subject"] for row in selected["holdout"]}
    development_videos = {row["video"] for row in selected["development"]}
    holdout_videos = {row["video"] for row in selected["holdout"]}
    report = {
        "status": "PASS" if not deficits else "NO_GO",
        "raw_counts": raw_counts,
        "final_counts": final_counts,
        "deficits": deficits,
        "invalid_row_count": invalid_row_count,
        "subject_overlap_count": len(development_subjects & holdout_subjects),
        "video_overlap_count": len(development_videos & holdout_videos),
    }
    if report["subject_overlap_count"] or report["video_overlap_count"]:
        report["status"] = "NO_GO"
    return selected, report


def _load_active_visor_hos_source_feasibility(
    fixture_root: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Load the historical all-family PASS source seal without reinterpretation."""

    amendment = _tuple_visor_hos_correction_amendment(cfg)
    protocol = _tuple_fixture_protocol(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    repair = _tuple_fixture_feasibility_repair(cfg)
    path = fixture_root / "visor-hos-source-feasibility.json"
    if not path.is_file():
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_MISSING")
    record = json.loads(path.read_text())
    payload = json.loads(json.dumps(record))
    expected = payload.pop("visor_hos_source_feasibility_commitment_sha256", None)
    if not isinstance(expected, str) or digest(payload) != expected:
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_COMMITMENT")
    commitments = {
        "visor_hos_correction_amendment_commitment_sha256": amendment[
            "amendment_commitment_sha256"
        ],
        "fixture_preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "fixture_feasibility_repair_commitment_sha256": repair[
            "fixture_feasibility_repair_commitment_sha256"
        ],
        "public_fixture_protocol_commitment_sha256": protocol[
            "protocol_commitment_sha256"
        ],
    }
    if any(record.get(key) != value for key, value in commitments.items()):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_PROVENANCE")
    if (
        record.get("status")
        != "PASS_SOURCE_FEASIBILITY_PENDING_NO_HAND_REVIEW"
        or record.get("no_hand_truth_opened") is not False
        or record.get("model_inference_executed") is not False
        or record.get("media_rendering_executed") is not False
        or record.get("restricted_mount_present") is not False
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_STATUS")
    families = record.get("families", {})
    if (
        not isinstance(families, dict)
        or record.get("failing_family_names") != []
        or any(
            not str(value.get("status", "")).startswith("PASS")
            for value in families.values()
            if isinstance(value, dict)
        )
        or any(not isinstance(value, dict) for value in families.values())
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_FAMILY")
    required_statuses = {
        "visor_hos_contact": "PASS",
        "visor_hos_explicit_no_contact": "PASS",
        "visor_hos_no_hand_nominees": "PASS_NOMINEE_QUEUE_READY",
        "visor_hos_integrity": "PASS",
        "cross_partition_source_independence": "PASS",
    }
    if any(
        not isinstance(families.get(family), dict)
        or families[family].get("status") != status
        for family, status in required_statuses.items()
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_FAMILY")
    selections = record.get("selections", {}).get("visor_hos_source_nominees")
    if not isinstance(selections, dict) or set(selections) != {
        "development",
        "holdout",
    }:
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION")
    partition_participants: dict[str, set[str]] = {}
    partition_videos: dict[str, set[str]] = {}
    partition_frames: dict[str, set[tuple[str, str]]] = {}
    for partition in ("development", "holdout"):
        rows = selections[partition]
        if not isinstance(rows, list):
            raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION")
        counts = Counter(row.get("stratum") for row in rows if isinstance(row, dict))
        if (
            counts["contact"] != 48
            or counts["explicit_no_contact"] != 48
            or not 48 <= counts["no_hand_nominee"] <= 192
            or sum(counts.values()) != len(rows)
        ):
            raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION_COUNT")
        video_stratum_counts: Counter[tuple[str, str]] = Counter()
        frames: set[tuple[str, str]] = set()
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("stratum")
                not in {"contact", "explicit_no_contact", "no_hand_nominee"}
                or row.get("source_split") not in {"train", "val"}
                or not re.fullmatch(r"P\d{2}", str(row.get("participant", "")))
                or not isinstance(row.get("video"), str)
                or not isinstance(row.get("frame_name"), str)
            ):
                raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION")
            key = (row["video"], row["frame_name"])
            if key in frames:
                raise RuntimeError("E_VISOR_HOS_SOURCE_FRAME_DUPLICATE")
            frames.add(key)
            video_stratum_counts[(row["video"], row["stratum"])] += 1
        if any(value > 4 for value in video_stratum_counts.values()):
            raise RuntimeError("E_VISOR_HOS_SOURCE_VIDEO_CAP")
        partition_participants[partition] = {row["participant"] for row in rows}
        partition_videos[partition] = {row["video"] for row in rows}
        partition_frames[partition] = frames
    if (
        partition_participants["development"] & partition_participants["holdout"]
        or partition_videos["development"] & partition_videos["holdout"]
        or partition_frames["development"] & partition_frames["holdout"]
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_OVERLAP")
    audits = record.get("audits", {})
    if any(
        int(audits.get(key, -1)) != 0
        for key in (
            "source_subject_overlap_count",
            "source_video_overlap_count",
            "source_object_overlap_count",
        )
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_OVERLAP")
    return record


def _load_construct_aligned_visor_hos_source_reuse(
    fixture_root: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    active = _construct_aligned_ltx_resume_amendment(cfg)
    amendment = _tuple_visor_hos_correction_amendment(cfg)
    protocol = _tuple_fixture_protocol(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    repair = _tuple_fixture_feasibility_repair(cfg)
    path = fixture_root / "visor-hos-source-feasibility.json"
    if not path.is_file():
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_MISSING")
    record = json.loads(path.read_text())
    payload = json.loads(json.dumps(record))
    expected = payload.pop("visor_hos_source_feasibility_commitment_sha256", None)
    if (
        not isinstance(expected, str)
        or expected != CONSTRUCT_ALIGNED_SOURCE_NO_GO_SHA256
        or expected
        != active["prior_results_and_amendments_preserved"][
            "complete_source_no_go"
        ]
        or digest(payload) != expected
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_COMMITMENT")
    commitments = {
        "visor_hos_correction_amendment_commitment_sha256": amendment[
            "amendment_commitment_sha256"
        ],
        "fixture_preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "fixture_feasibility_repair_commitment_sha256": repair[
            "fixture_feasibility_repair_commitment_sha256"
        ],
        "public_fixture_protocol_commitment_sha256": protocol[
            "protocol_commitment_sha256"
        ],
    }
    if any(record.get(key) != value for key, value in commitments.items()):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_PROVENANCE")
    if (
        record.get("schema_version") != 2
        or record.get("status")
        != "NO_GO_COMPLETE_SOURCE_FEASIBILITY"
        or record.get("no_hand_truth_opened") is not False
        or record.get("no_hand_review_required_before_public_model_inference")
        is not True
        or record.get("model_inference_executed") is not False
        or record.get("media_rendering_executed") is not False
        or record.get("large_Charades_video_archive_downloaded") is not False
        or record.get("restricted_mount_present") is not False
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_STATUS")
    families = record.get("families", {})
    exact_family_statuses = {
        "official_visor_hos_artifact": "PASS",
        "visor_hos_semantic_reference": "PASS",
        "visor_hos_contact": "PASS",
        "visor_hos_explicit_no_contact": "PASS",
        "visor_hos_no_hand_nominees": "PASS_NOMINEE_QUEUE_READY",
        "visor_hos_integrity": "PASS",
        "coco_composite_sources": "PASS",
        "language_and_lexical": "PASS",
        "referent_attribute_composite": "PASS",
        "recurrence": "PASS",
        "sensor": "PASS",
        "charades_order_action": "NO_GO",
        "cross_partition_source_independence": "PASS",
    }
    if (
        not isinstance(families, dict)
        or set(families) != set(exact_family_statuses)
        or record.get("failing_family_names") != ["charades_order_action"]
        or any(
            not isinstance(families.get(family), dict)
            or families[family].get("status") != status
            for family, status in exact_family_statuses.items()
        )
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_FAMILY")
    required_statuses = {
        "visor_hos_contact": "PASS",
        "visor_hos_explicit_no_contact": "PASS",
        "visor_hos_no_hand_nominees": "PASS_NOMINEE_QUEUE_READY",
        "visor_hos_integrity": "PASS",
        "cross_partition_source_independence": "PASS",
    }
    if any(
        not isinstance(families.get(family), dict)
        or families[family].get("status") != status
        for family, status in required_statuses.items()
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_FAMILY")
    selections = record.get("selections", {}).get("visor_hos_source_nominees")
    if not isinstance(selections, dict) or set(selections) != {
        "development",
        "holdout",
    }:
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION")
    partition_participants: dict[str, set[str]] = {}
    partition_videos: dict[str, set[str]] = {}
    partition_frames: dict[str, set[tuple[str, str]]] = {}
    for partition in ("development", "holdout"):
        rows = selections[partition]
        if not isinstance(rows, list):
            raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION")
        counts = Counter(
            row.get("stratum") for row in rows if isinstance(row, dict)
        )
        if (
            counts["contact"] != 48
            or counts["explicit_no_contact"] != 48
            or not 48 <= counts["no_hand_nominee"] <= 192
            or sum(counts.values()) != len(rows)
        ):
            raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION_COUNT")
        video_stratum_counts: Counter[tuple[str, str]] = Counter()
        frames: set[tuple[str, str]] = set()
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("stratum")
                not in {"contact", "explicit_no_contact", "no_hand_nominee"}
                or row.get("source_split") not in {"train", "val"}
                or not re.fullmatch(r"P\d{2}", str(row.get("participant", "")))
                or not isinstance(row.get("video"), str)
                or not isinstance(row.get("frame_name"), str)
            ):
                raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION")
            key = (row["video"], row["frame_name"])
            if key in frames:
                raise RuntimeError("E_VISOR_HOS_SOURCE_FRAME_DUPLICATE")
            frames.add(key)
            video_stratum_counts[(row["video"], row["stratum"])] += 1
        if any(value > 4 for value in video_stratum_counts.values()):
            raise RuntimeError("E_VISOR_HOS_SOURCE_VIDEO_CAP")
        partition_participants[partition] = {
            row["participant"] for row in rows
        }
        partition_videos[partition] = {row["video"] for row in rows}
        partition_frames[partition] = frames
    if (
        partition_participants["development"]
        & partition_participants["holdout"]
        or partition_videos["development"] & partition_videos["holdout"]
        or partition_frames["development"] & partition_frames["holdout"]
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_OVERLAP")
    audits = record.get("audits", {})
    if any(
        int(audits.get(key, -1)) != 0
        for key in (
            "source_subject_overlap_count",
            "source_video_overlap_count",
            "source_object_overlap_count",
        )
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_OVERLAP")
    action = record.get("selections", {}).get("charades_order_action")
    inventory = record.get("action_inventory")
    if (
        not isinstance(action, dict)
        or set(action) != {"development", "holdout"}
        or not isinstance(inventory, dict)
        or inventory.get("status") != "NO_GO"
        or inventory.get("deficits") != CONSTRUCT_ALIGNED_ACTION_DEFICITS
        or inventory.get("final_counts")
        != CONSTRUCT_ALIGNED_ACTION_CLASS_COUNTS
        or int(inventory.get("subject_overlap_count", -1)) != 0
        or int(inventory.get("video_overlap_count", -1)) != 0
    ):
        raise RuntimeError("E_CONSTRUCT_ALIGNED_ACTION_SOURCE_LINEAGE")
    action_protocol = protocol["order_dependent_action_control"]
    labels = list(action_protocol["labels"])
    code_to_label = {
        code: label
        for pair in action_protocol["class_code_pairs"]
        for label, codes in zip(
            pair["pair"], zip(*pair["matched_codes"], strict=True), strict=True
        )
        for code in codes
    }
    action_subjects: dict[str, set[str]] = {}
    action_videos: dict[str, set[str]] = {}
    for partition in ("development", "holdout"):
        rows = action[partition]
        if not isinstance(rows, list) or len(rows) != CONSTRUCT_ALIGNED_ACTION_COUNTS[
            partition
        ]:
            raise RuntimeError("E_CONSTRUCT_ALIGNED_ACTION_SOURCE_COUNT")
        counts = Counter()
        subjects: set[str] = set()
        videos: set[str] = set()
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row)
                != {
                    "video",
                    "subject",
                    "label",
                    "code",
                    "start",
                    "end",
                    "source_duration",
                }
                or row.get("label") not in CONSTRUCT_ALIGNED_ACTION_CLASS_COUNTS[
                    partition
                ]
                or not isinstance(row.get("subject"), str)
                or not row["subject"]
                or not isinstance(row.get("video"), str)
                or not row["video"]
                or row["video"] in videos
                or code_to_label.get(row.get("code")) != row.get("label")
                or not all(
                    isinstance(row.get(key), (int, float))
                    and not isinstance(row.get(key), bool)
                    and math.isfinite(float(row[key]))
                    for key in ("start", "end", "source_duration")
                )
                or not 0.0 <= float(row["start"]) < float(row["end"])
                <= float(row["source_duration"])
                or not 1.0 <= float(row["end"]) - float(row["start"]) <= 12.0
            ):
                raise RuntimeError("E_CONSTRUCT_ALIGNED_ACTION_SOURCE_ROW")
            counts[row["label"]] += 1
            subjects.add(row["subject"])
            videos.add(row["video"])
        if dict(counts) != CONSTRUCT_ALIGNED_ACTION_CLASS_COUNTS[partition]:
            raise RuntimeError("E_CONSTRUCT_ALIGNED_ACTION_SOURCE_COUNT")
        expected_order = sorted(
            rows,
            key=lambda row: (
                labels.index(row["label"]),
                _fixture_order(
                    int(preparation["seed"]),
                    "mechanistic_action_final",
                    partition,
                    row["video"],
                    row["start"],
                ),
            ),
        )
        if rows != expected_order:
            raise RuntimeError("E_CONSTRUCT_ALIGNED_ACTION_SOURCE_ORDER")
        action_subjects[partition] = subjects
        action_videos[partition] = videos
    if (
        action_subjects["development"] & action_subjects["holdout"]
        or action_videos["development"] & action_videos["holdout"]
    ):
        raise RuntimeError("E_CONSTRUCT_ALIGNED_ACTION_SOURCE_OVERLAP")
    return record


def _validate_construct_aligned_action_fixture_projection(
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    partition: str,
) -> None:
    """Prove the active action fixtures are the sealed rows plus media fields."""

    source_keys = {
        "video",
        "subject",
        "label",
        "code",
        "start",
        "end",
        "source_duration",
    }
    added_keys = {
        "fixture_ordinal",
        "media_relative_path",
        "media_sha256",
        "media_bytes",
    }
    if (
        partition not in CONSTRUCT_ALIGNED_ACTION_COUNTS
        or len(rows) != CONSTRUCT_ALIGNED_ACTION_COUNTS[partition]
        or len(source_rows) != CONSTRUCT_ALIGNED_ACTION_COUNTS[partition]
    ):
        raise RuntimeError("E_CONSTRUCT_ALIGNED_ACTION_FIXTURE_COUNT")
    for ordinal, (row, source) in enumerate(zip(rows, source_rows, strict=True)):
        if (
            not isinstance(row, dict)
            or not isinstance(source, dict)
            or set(row) != source_keys | added_keys
            or {key: row[key] for key in source_keys} != source
            or row.get("fixture_ordinal") != ordinal
            or not isinstance(row.get("media_relative_path"), str)
            or not row["media_relative_path"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("media_sha256", "")))
            or not isinstance(row.get("media_bytes"), int)
            or isinstance(row.get("media_bytes"), bool)
            or row["media_bytes"] <= 0
        ):
            raise RuntimeError("E_CONSTRUCT_ALIGNED_ACTION_FIXTURE_PROJECTION")


def _active_visor_hos_no_hand_review_queues(
    source_record: dict[str, Any], cfg: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Recover the exact frozen nominee queues without rerunning selection."""

    amendment = _tuple_visor_hos_correction_amendment(cfg)
    seed = int(amendment["partition_and_joint_sampler"]["seed"])
    per_video_cap = int(
        amendment["partition_and_joint_sampler"]["per_video_per_stratum_cap"]
    )
    source = source_record.get("selections", {}).get(
        "visor_hos_source_nominees"
    )
    inventory = source_record.get("visor_hos_inventory", {}).get(
        "no_hand_review_queue_inventory"
    )
    if (
        not isinstance(source, dict)
        or set(source) != {"development", "holdout"}
        or not isinstance(inventory, dict)
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_RECORD")
    output: dict[str, list[dict[str, Any]]] = {}
    participant_sets: dict[str, set[str]] = {}
    video_sets: dict[str, set[str]] = {}
    frame_sets: dict[str, set[tuple[str, str]]] = {}
    for partition in ("development", "holdout"):
        rows = [
            json.loads(json.dumps(row))
            for row in source[partition]
            if isinstance(row, dict) and row.get("stratum") == "no_hand_nominee"
        ]
        rows.sort(key=lambda row: int(row.get("review_ordinal", -1)))
        if (
            not 48 <= len(rows) <= VISOR_HOS_NO_HAND_REVIEW_MAX_PER_PARTITION
            or inventory.get("queue_counts", {}).get(partition) != len(rows)
            or [row.get("review_ordinal") for row in rows]
            != list(range(1, len(rows) + 1))
        ):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_COUNT")
        video_counts: Counter[str] = Counter()
        frames: set[tuple[str, str]] = set()
        for row in rows:
            image_path = Path(str(row.get("image_path", "")))
            frame_key = (str(row.get("video", "")), str(row.get("frame_name", "")))
            if (
                row.get("hand_visible") is not False
                or row.get("contact") is not None
                or row.get("target_hand_side") is not None
                or row.get("target_hand_segments") is not None
                or row.get("source_split") not in {"train", "val"}
                or not re.fullmatch(r"P\d{2}", str(row.get("participant", "")))
                or not frame_key[0]
                or frame_key[0].split("_", 1)[0] != row.get("participant")
                or Path(frame_key[1]).name != frame_key[1]
                or image_path.is_absolute()
                or not image_path.parts
                or ".." in image_path.parts
                or image_path.name != frame_key[1]
                or row.get("review_token")
                != _visor_hos_no_hand_review_token(seed, partition, row)
                or frame_key in frames
            ):
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_ROW")
            frames.add(frame_key)
            video_counts[frame_key[0]] += 1
        if any(value > per_video_cap for value in video_counts.values()):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_VIDEO_CAP")
        output[partition] = rows
        participant_sets[partition] = {row["participant"] for row in rows}
        video_sets[partition] = {row["video"] for row in rows}
        frame_sets[partition] = frames
    overlap = {
        "participant_overlap_count": len(
            participant_sets["development"] & participant_sets["holdout"]
        ),
        "video_overlap_count": len(
            video_sets["development"] & video_sets["holdout"]
        ),
        "frame_overlap_count": len(
            frame_sets["development"] & frame_sets["holdout"]
        ),
    }
    if any(overlap.values()) or any(
        int(inventory.get(key, -1)) != 0 for key in overlap
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_OVERLAP")
    return output, json.loads(json.dumps(inventory))


def _materialize_active_visor_hos_no_hand_review_frames(
    fixture_root: Path,
    review_root: Path,
    preparation: dict[str, Any],
    source_record: dict[str, Any],
    queues: dict[str, list[dict[str, Any]]],
    construct_aligned_amendment_commitment_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Download only frozen nominee archives and extract only nominee frames."""

    if (
        construct_aligned_amendment_commitment_sha256
        != CONSTRUCT_ALIGNED_RESUME_AMENDMENT_SHA256
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_LINEAGE")
    _require_external_or_ignored_output(review_root)
    frame_root = review_root / "source-frames"
    base = preparation["source_archives"]["EPIC_KITCHENS_VISOR_validation"]
    archive_root = fixture_root / "sources/VISOR-HOS/frame-archives"
    archive_specs: dict[str, tuple[str, Path]] = {}
    for partition in ("development", "holdout"):
        for row in queues[partition]:
            split = row["source_split"]
            participant = row["participant"]
            video = row["video"]
            archive_url = (
                f"{base['repository_root'].rstrip('/')}/rgb_frames/"
                f"{split}/{participant}/{video}.zip"
            )
            archive_path = archive_root / split / participant / f"{video}.zip"
            archive_key = str(archive_path.relative_to(fixture_root))
            spec = (archive_url, archive_path)
            if archive_key in archive_specs and archive_specs[archive_key] != spec:
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_ARCHIVE_COLLISION")
            archive_specs[archive_key] = spec

    def ensure_archive(spec: tuple[str, Path]) -> None:
        archive_url, archive_path = spec
        if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
            _download_public_file(archive_url, archive_path)
        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_ARCHIVE_INVALID")

    ordered_specs = [archive_specs[key] for key in sorted(archive_specs)]
    worker_count = min(4, len(ordered_specs))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(ensure_archive, spec) for spec in ordered_specs]
        for future in futures:
            future.result()

    archive_records: dict[str, dict[str, Any]] = {}
    frame_records: list[dict[str, Any]] = []
    target_paths: set[Path] = set()
    for partition in ("development", "holdout"):
        for row in queues[partition]:
            split = row["source_split"]
            participant = row["participant"]
            video = row["video"]
            archive_url = (
                f"{base['repository_root'].rstrip('/')}/rgb_frames/"
                f"{split}/{participant}/{video}.zip"
            )
            archive_path = archive_root / split / participant / f"{video}.zip"
            archive_key = str(archive_path.relative_to(fixture_root))
            archive_records.setdefault(
                archive_key,
                {
                    "relative_path": archive_key,
                    "url": archive_url,
                    "sha256": file_digest(archive_path),
                    "bytes": archive_path.stat().st_size,
                    "license": base["license"],
                },
            )
            with zipfile.ZipFile(archive_path) as archive:
                matches = [
                    name
                    for name in archive.namelist()
                    if Path(name).name == row["frame_name"]
                ]
                if len(matches) != 1:
                    raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_FRAME_MEMBER")
                member = Path(matches[0])
                if member.is_absolute() or ".." in member.parts:
                    raise RuntimeError("E_TUPLE_ARCHIVE_PATH")
                target = _visor_hos_review_source_frame(
                    frame_root, row["image_path"]
                )
                if target in target_paths:
                    raise RuntimeError(
                        "E_VISOR_HOS_NO_HAND_REVIEW_FRAME_PATH_COLLISION"
                    )
                target_paths.add(target)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                partial = target.with_suffix(target.suffix + ".partial")
                with archive.open(matches[0]) as source, partial.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
                os.chmod(partial, 0o600)
                partial.replace(target)
            try:
                from PIL import Image

                with Image.open(target) as image:
                    width, height = image.size
                    image.verify()
            except Exception as error:
                raise RuntimeError(
                    "E_VISOR_HOS_NO_HAND_REVIEW_FRAME_DECODE"
                ) from error
            if width <= 0 or height <= 0:
                raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_FRAME_DECODE")
            frame_records.append(
                {
                    "partition": partition,
                    "review_token": row["review_token"],
                    "source_relative_path": str(target.relative_to(review_root)),
                    "sha256": file_digest(target),
                    "bytes": target.stat().st_size,
                    "width": width,
                    "height": height,
                }
            )
    record = {
        "schema_version": 1,
        "status": "SEALED_PUBLIC_NOMINEE_FRAMES_BEFORE_APPLICANT_REVIEW",
        "visor_hos_source_feasibility_commitment_sha256": source_record[
            "visor_hos_source_feasibility_commitment_sha256"
        ],
        "construct_aligned_ltx_resume_amendment_commitment_sha256": (
            construct_aligned_amendment_commitment_sha256
        ),
        "archive_count": len(archive_records),
        "archive_download_worker_count": worker_count,
        "source_frame_count": len(frame_records),
        "archives": [archive_records[key] for key in sorted(archive_records)],
        "frames": frame_records,
        "model_inference_executed": False,
        "restricted_mount_present": False,
    }
    record["source_frame_materialization_commitment_sha256"] = digest(record)
    path = review_root / "source-frame-materialization.json"
    if path.exists():
        prior = json.loads(path.read_text())
        expected = prior.pop(
            "source_frame_materialization_commitment_sha256", None
        )
        if expected != digest(prior) or prior != {
            key: value
            for key, value in record.items()
            if key != "source_frame_materialization_commitment_sha256"
        }:
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_FRAMES_ALREADY_FROZEN")
    else:
        write_private(path, record)
    return frame_root, record


def _load_visor_hos_no_hand_frame_materialization(
    review_root: Path,
    expected_source_feasibility_commitment_sha256: str,
    expected_construct_aligned_amendment_commitment_sha256: str | None = None,
) -> dict[str, Any]:
    path = review_root / "source-frame-materialization.json"
    if not path.is_file():
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_FRAMES_MISSING")
    record = json.loads(path.read_text())
    payload = json.loads(json.dumps(record))
    expected = payload.pop("source_frame_materialization_commitment_sha256", None)
    if not isinstance(expected, str) or digest(payload) != expected:
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_FRAMES_COMMITMENT")
    if (
        record.get("status")
        != "SEALED_PUBLIC_NOMINEE_FRAMES_BEFORE_APPLICANT_REVIEW"
        or record.get("visor_hos_source_feasibility_commitment_sha256")
        != expected_source_feasibility_commitment_sha256
        or (
            expected_construct_aligned_amendment_commitment_sha256 is not None
            and record.get(
                "construct_aligned_ltx_resume_amendment_commitment_sha256"
            )
            != expected_construct_aligned_amendment_commitment_sha256
        )
        or record.get("model_inference_executed") is not False
        or record.get("restricted_mount_present") is not False
        or record.get("archive_count") != len(record.get("archives", []))
        or record.get("source_frame_count") != len(record.get("frames", []))
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_FRAMES_LINEAGE")
    return record


def prepare_active_visor_hos_no_hand_review(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    active = _construct_aligned_ltx_resume_amendment(cfg)
    fixture_root = _tuple_fixture_root(args.public_root)
    review_root = fixture_root / "no-hand-review"
    _require_external_or_ignored_output(args.public_root)
    source_record = _load_construct_aligned_visor_hos_source_reuse(
        fixture_root, cfg
    )
    queues, inventory = _active_visor_hos_no_hand_review_queues(
        source_record, cfg
    )
    preparation = _tuple_fixture_preparation_amendment(cfg)
    frame_root, materialization = (
        _materialize_active_visor_hos_no_hand_review_frames(
            fixture_root,
            review_root,
            preparation,
            source_record,
            queues,
            active["amendment_commitment_sha256"],
        )
    )
    compact = prepare_visor_hos_no_hand_review(
        None,
        cfg=cfg,
        frame_root=frame_root,
        review_root=review_root,
        preselected_queues=queues,
        inventory_override=inventory,
        source_feasibility_commitment_sha256=source_record[
            "visor_hos_source_feasibility_commitment_sha256"
        ],
        source_frame_materialization_commitment_sha256=materialization[
            "source_frame_materialization_commitment_sha256"
        ],
        construct_aligned_amendment_commitment_sha256=active[
            "amendment_commitment_sha256"
        ],
    )
    compact.update(
        {
            "source_frame_count": materialization["source_frame_count"],
            "source_archive_count": materialization["archive_count"],
            "restricted_mount_present": False,
            "model_inference_executed": False,
            "visor_hos_source_feasibility_commitment_sha256": source_record[
                "visor_hos_source_feasibility_commitment_sha256"
            ],
            "source_frame_materialization_commitment_sha256": materialization[
                "source_frame_materialization_commitment_sha256"
            ],
        }
    )
    return compact


def seal_active_visor_hos_no_hand_review(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    active = _construct_aligned_ltx_resume_amendment(cfg)
    fixture_root = _tuple_fixture_root(args.public_root)
    review_root = fixture_root / "no-hand-review"
    _require_external_or_ignored_output(args.public_root)
    source_record = _load_construct_aligned_visor_hos_source_reuse(
        fixture_root, cfg
    )
    source_commitment = source_record[
        "visor_hos_source_feasibility_commitment_sha256"
    ]
    queue = _load_visor_hos_no_hand_review_queue(review_root)
    materialization = _load_visor_hos_no_hand_frame_materialization(
        review_root,
        source_commitment,
        active["amendment_commitment_sha256"],
    )
    amendment = _tuple_visor_hos_correction_amendment(cfg)
    if (
        queue.get("visor_hos_source_feasibility_commitment_sha256")
        != source_commitment
        or queue.get("source_frame_materialization_commitment_sha256")
        != materialization["source_frame_materialization_commitment_sha256"]
        or queue.get("visor_hos_correction_amendment_commitment_sha256")
        != amendment["amendment_commitment_sha256"]
        or queue.get(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        )
        != active["amendment_commitment_sha256"]
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_LINEAGE")
    compact = seal_visor_hos_no_hand_review(
        review_root=review_root,
        authorized_applicant_attested=args.authorized_applicant_attested,
        blind_to_egohos_output_attested=args.blind_to_egohos_output_attested,
        egohos_inference_not_started_attested=(
            args.egohos_inference_not_started_attested
        ),
    )
    seal_commitment = compact.get("verified_no_hand_seal_commitment_sha256")
    if compact.get("status") == "INCOMPLETE_REVIEW" and seal_commitment is None:
        compact.pop("verified_no_hand_seal_commitment_sha256", None)
    elif compact.get("status") not in {"PASS", "NO_GO"} or not isinstance(
        seal_commitment, str
    ) or not re.fullmatch(r"[0-9a-f]{64}", seal_commitment):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SEAL_STATUS")
    compact.update(
        {
            "restricted_mount_present": False,
            "model_inference_executed": False,
            "visor_hos_source_feasibility_commitment_sha256": source_commitment,
            "review_queue_commitment_sha256": queue[
                "review_queue_commitment_sha256"
            ],
        }
    )
    return compact


def _load_visor_hos_verified_no_hand_lineage(
    review_root: Path,
    *,
    expected_source_feasibility_commitment_sha256: str | None = None,
    expected_construct_aligned_amendment_commitment_sha256: str | None = None,
) -> dict[str, Any]:
    verified = load_visor_hos_verified_no_hand_frames(review_root)
    queue = _load_visor_hos_no_hand_review_queue(review_root)
    if expected_source_feasibility_commitment_sha256 is not None:
        if (
            not re.fullmatch(
                r"[0-9a-f]{64}", expected_source_feasibility_commitment_sha256
            )
            or queue.get("visor_hos_source_feasibility_commitment_sha256")
            != expected_source_feasibility_commitment_sha256
        ):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_LINEAGE")
        materialization = _load_visor_hos_no_hand_frame_materialization(
            review_root,
            expected_source_feasibility_commitment_sha256,
            expected_construct_aligned_amendment_commitment_sha256,
        )
        if queue.get("source_frame_materialization_commitment_sha256") != (
            materialization["source_frame_materialization_commitment_sha256"]
        ):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_LINEAGE")
    if expected_construct_aligned_amendment_commitment_sha256 is not None:
        if (
            queue.get(
                "construct_aligned_ltx_resume_amendment_commitment_sha256"
            )
            != expected_construct_aligned_amendment_commitment_sha256
        ):
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_LINEAGE")
    seal_path = review_root / "verified-no-hand-seal.json"
    seal = json.loads(seal_path.read_text())
    commitment = seal.get("verified_no_hand_seal_commitment_sha256")
    payload = json.loads(json.dumps(seal))
    payload.pop("verified_no_hand_seal_commitment_sha256", None)
    if not isinstance(commitment, str) or digest(payload) != commitment:
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SEAL_COMMITMENT")
    if (
        expected_construct_aligned_amendment_commitment_sha256 is not None
        and seal.get(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        )
        != expected_construct_aligned_amendment_commitment_sha256
    ):
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SOURCE_LINEAGE")
    by_partition: dict[str, list[dict[str, Any]]] = {}
    for partition in ("development", "holdout"):
        rows = seal.get("partitions", {}).get(partition, {}).get("selected")
        if not isinstance(rows, list) or len(rows) != 48:
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SELECTION_COUNT")
        by_partition[partition] = rows
    if verified != {
        (row["video"], row["frame_name"])
        for rows in by_partition.values()
        for row in rows
    }:
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SELECTION_KEY")
    return {
        "commitment_sha256": commitment,
        "queue_commitment_sha256": queue["review_queue_commitment_sha256"],
        "source_feasibility_commitment_sha256": queue.get(
            "visor_hos_source_feasibility_commitment_sha256"
        ),
        "source_frame_materialization_commitment_sha256": queue.get(
            "source_frame_materialization_commitment_sha256"
        ),
        "construct_aligned_ltx_resume_amendment_commitment_sha256": queue.get(
            "construct_aligned_ltx_resume_amendment_commitment_sha256"
        ),
        "partitions": by_partition,
    }


def _merge_active_visor_hos_selections(
    source_record: dict[str, Any], verified: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    source = source_record["selections"]["visor_hos_source_nominees"]
    output: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    all_partition_frames: dict[str, set[tuple[str, str]]] = {}
    for partition in ("development", "holdout"):
        rows = source.get(partition)
        if not isinstance(rows, list):
            raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION")
        contact = [row for row in rows if row.get("stratum") == "contact"]
        no_contact = [
            row for row in rows if row.get("stratum") == "explicit_no_contact"
        ]
        nominees = {
            (row.get("video"), row.get("frame_name")): row
            for row in rows
            if row.get("stratum") == "no_hand_nominee"
        }
        if (
            len(contact) != 48
            or len(no_contact) != 48
            or len(nominees)
            != sum(row.get("stratum") == "no_hand_nominee" for row in rows)
        ):
            raise RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_SELECTION_COUNT")
        verified_rows = []
        for seal_row in verified["partitions"][partition]:
            key = (seal_row.get("video"), seal_row.get("frame_name"))
            nominee = nominees.get(key)
            if (
                nominee is None
                or not isinstance(seal_row.get("source_frame_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", seal_row["source_frame_sha256"])
            ):
                raise RuntimeError("E_VISOR_HOS_VERIFIED_NO_HAND_NOT_NOMINATED")
            verified_rows.append(
                {
                    **nominee,
                    "stratum": "verified_no_hand",
                    "verified_source_frame_sha256": seal_row[
                        "source_frame_sha256"
                    ],
                    "review_token": seal_row["review_token"],
                }
            )
        if len(verified_rows) != 48:
            raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_SELECTION_COUNT")
        output[partition] = contact + no_contact + verified_rows
        frames = {(row["video"], row["frame_name"]) for row in output[partition]}
        if len(frames) != 144:
            raise RuntimeError("E_VISOR_HOS_SOURCE_FRAME_DUPLICATE")
        for row in output[partition]:
            if row.get("source_split") not in {"train", "val"}:
                raise RuntimeError("E_VISOR_HOS_SOURCE_SPLIT")
        all_partition_frames[partition] = frames
    audits = {
        "source_frame_overlap_count": len(
            all_partition_frames["development"]
            & all_partition_frames["holdout"]
        ),
        "source_frame_duplicate_count": 0,
    }
    if audits["source_frame_overlap_count"]:
        raise RuntimeError("E_VISOR_HOS_SOURCE_FRAME_OVERLAP")
    return output, audits


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


def _load_visor_hos_opencv():
    """Load only the pinned runtime rasterizer; never fall back to Pillow."""

    import importlib.metadata

    try:
        distribution_version = importlib.metadata.version("opencv-python-headless")
        import cv2
    except Exception as error:
        raise RuntimeError("E_TUPLE_VISOR_RASTERIZER_IMPORT") from error
    if distribution_version != "4.10.0.84" or cv2.__version__ != "4.10.0":
        raise RuntimeError("E_TUPLE_VISOR_RASTERIZER_VERSION")
    return cv2


def _rasterize_visor_hos_target_hand_mask(
    segments: Any,
    width: int,
    height: int,
    *,
    cv2_module: Any | None = None,
):
    """Apply the pinned VISOR-HOS int32/fixed-canvas mask semantics."""

    import numpy as np

    if width <= 0 or height <= 0 or not _valid_visor_segments(segments):
        raise RuntimeError("E_TUPLE_VISOR_GEOMETRY")
    cv2_module = cv2_module or _load_visor_hos_opencv()
    maximum = int(np.iinfo(np.int32).max)
    contours = []
    boundary_vertex_count = 0
    outside_canvas_vertex_count = 0
    outside_canvas_component_count = 0
    for polygon in segments:
        points = np.asarray(polygon, dtype=np.float64)
        truncated = np.trunc(points)
        if (
            points.shape != (len(polygon), 2)
            or not np.isfinite(points).all()
            or (points < 0.0).any()
            or (truncated > maximum).any()
        ):
            raise RuntimeError("E_TUPLE_VISOR_GEOMETRY")
        boundary = (points[:, 0] == width) | (points[:, 1] == height)
        outside = (points[:, 0] > width) | (points[:, 1] > height)
        boundary_vertex_count += int(boundary.sum())
        outside_canvas_vertex_count += int(outside.sum())
        outside_canvas_component_count += int(outside.any())
        contours.append(truncated.astype(np.int32))
    union = np.zeros((height, width), dtype=np.uint8)
    try:
        cv2_module.fillPoly(union, contours, color=(1, 1, 1))
    except Exception as error:
        raise RuntimeError("E_TUPLE_VISOR_RASTERIZER_FAILURE") from error
    if (
        union.shape != (height, width)
        or union.dtype != np.uint8
        or not np.isin(union, (0, 1)).all()
    ):
        raise RuntimeError("E_TUPLE_VISOR_RASTERIZER_OUTPUT")
    if not union.any():
        raise RuntimeError("E_TUPLE_VISOR_EMPTY_HAND_MASK")
    return {
        "mask": union,
        "boundary_vertex_count": boundary_vertex_count,
        "outside_canvas_vertex_count": outside_canvas_vertex_count,
        "outside_canvas_component_count": outside_canvas_component_count,
    }


def _write_visor_hos_target_hand_mask(
    segments: Any,
    width: int,
    height: int,
    target: Path,
    *,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    rasterized = _rasterize_visor_hos_target_hand_mask(
        segments, width, height, cv2_module=cv2_module
    )
    expected = rasterized["mask"] * np.uint8(255)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        Image.fromarray(expected).save(partial, format="PNG")
        os.chmod(partial, 0o600)
        with Image.open(partial) as check:
            if (
                check.format != "PNG"
                or check.mode != "L"
                or check.size != (width, height)
            ):
                raise RuntimeError("E_TUPLE_VISOR_HAND_MASK_ROUNDTRIP")
            observed = np.asarray(check).copy()
        if (
            observed.shape != (height, width)
            or observed.dtype != np.uint8
            or not np.isin(observed, (0, 255)).all()
            or not (observed == 255).any()
            or not np.array_equal(observed, expected)
        ):
            raise RuntimeError("E_TUPLE_VISOR_HAND_MASK_ROUNDTRIP")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "path": target,
        "sha256": file_digest(target),
        "bytes": target.stat().st_size,
        "width": width,
        "height": height,
        "boundary_vertex_count": rasterized["boundary_vertex_count"],
        "outside_canvas_vertex_count": rasterized[
            "outside_canvas_vertex_count"
        ],
        "outside_canvas_component_count": rasterized[
            "outside_canvas_component_count"
        ],
    }


def _prepare_active_visor_hos_fixtures(
    fixture_root: Path,
    preparation: dict[str, Any],
    cfg: dict[str, Any],
    review_root: Path,
    source_record: dict[str, Any] | None = None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Materialize only corrected, source-sealed VISOR-HOS fixtures."""

    _require_external_or_ignored_output(review_root)
    active = _construct_aligned_ltx_resume_amendment(cfg)
    source_record = source_record or _load_construct_aligned_visor_hos_source_reuse(
        fixture_root, cfg
    )
    verified = _load_visor_hos_verified_no_hand_lineage(
        review_root,
        expected_source_feasibility_commitment_sha256=source_record[
            "visor_hos_source_feasibility_commitment_sha256"
        ],
        expected_construct_aligned_amendment_commitment_sha256=active[
            "amendment_commitment_sha256"
        ],
    )
    selected, frame_audits = _merge_active_visor_hos_selections(
        source_record, verified
    )
    amendment = _tuple_visor_hos_correction_amendment(cfg)
    base = preparation["source_archives"]["EPIC_KITCHENS_VISOR_validation"]
    source_root = fixture_root / "sources/VISOR-HOS"
    output: dict[str, list[dict[str, Any]]] = {
        partition: [] for partition in preparation["partitions"]
    }
    zip_records: dict[str, dict[str, Any]] = {}
    for partition in preparation["partitions"]:
        for ordinal, row in enumerate(selected[partition]):
            video = row["video"]
            participant = row["participant"]
            split = row["source_split"]
            archive_url = (
                f"{base['repository_root'].rstrip('/')}/rgb_frames/"
                f"{split}/{participant}/{video}.zip"
            )
            archive_path = (
                source_root
                / "frame-archives"
                / split
                / participant
                / f"{video}.zip"
            )
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
            source_frame_sha256 = file_digest(target)
            expected_verified_sha256 = row.get("verified_source_frame_sha256")
            if (
                expected_verified_sha256 is not None
                and source_frame_sha256 != expected_verified_sha256
            ):
                raise RuntimeError("E_VISOR_HOS_VERIFIED_FRAME_HASH")
            target_segments = row.get("target_hand_segments")
            target_hand_mask: dict[str, Any] | None = None
            if row["stratum"] in {"contact", "explicit_no_contact"}:
                target_hand_mask = _write_visor_hos_target_hand_mask(
                    target_segments,
                    width,
                    height,
                    fixture_root
                    / "truth/hand-contact"
                    / partition
                    / f"{ordinal:03d}-target-hand.png",
                )
            elif row["stratum"] != "verified_no_hand" or target_segments is not None:
                raise RuntimeError("E_VISOR_HOS_ACTIVE_STRATUM")
            output[partition].append(
                {
                    "fixture_ordinal": ordinal,
                    "stratum": row["stratum"],
                    "hand_visible": row["hand_visible"],
                    "contact": row["contact"],
                    "target_hand_side": row.get("target_hand_side"),
                    "target_hand_mask_relative_path": (
                        str(target_hand_mask["path"].relative_to(fixture_root))
                        if target_hand_mask is not None
                        else None
                    ),
                    "target_hand_mask_sha256": (
                        target_hand_mask["sha256"]
                        if target_hand_mask is not None
                        else None
                    ),
                    "target_hand_mask_bytes": (
                        target_hand_mask["bytes"]
                        if target_hand_mask is not None
                        else None
                    ),
                    "target_hand_mask_width": (
                        target_hand_mask["width"]
                        if target_hand_mask is not None
                        else None
                    ),
                    "target_hand_mask_height": (
                        target_hand_mask["height"]
                        if target_hand_mask is not None
                        else None
                    ),
                    "target_hand_boundary_vertex_count": (
                        target_hand_mask["boundary_vertex_count"]
                        if target_hand_mask is not None
                        else 0
                    ),
                    "target_hand_outside_canvas_vertex_count": (
                        target_hand_mask["outside_canvas_vertex_count"]
                        if target_hand_mask is not None
                        else 0
                    ),
                    "target_hand_outside_canvas_component_count": (
                        target_hand_mask["outside_canvas_component_count"]
                        if target_hand_mask is not None
                        else 0
                    ),
                    "source_split": split,
                    "source_participant": participant,
                    "source_video": video,
                    "source_frame_name": row["frame_name"],
                    "source_frame_sha256": source_frame_sha256,
                    "media_relative_path": str(target.relative_to(fixture_root)),
                    "media_sha256": source_frame_sha256,
                    "media_bytes": target.stat().st_size,
                    "source_polygon_finite_nonnegative": True,
                    "target_hand_mask_exact_frame_binary_nonempty": (
                        target_hand_mask is not None
                    ),
                    "geometry_valid": True,
                    "verified_no_hand_review_token": row.get("review_token"),
                    "verified_no_hand_seal_commitment_sha256": (
                        verified["commitment_sha256"]
                        if row["stratum"] == "verified_no_hand"
                        else None
                    ),
                }
            )
    source_commitment = source_record[
        "visor_hos_source_feasibility_commitment_sha256"
    ]
    provenance = [
        {
            "source": "VISOR_HOS_OFFICIAL_TRAIN_VALIDATION_ANNOTATIONS",
            "sha256": amendment["official_annotation_artifact"][
                "external_sorted_relative_path_and_SHA256_manifest_commitment_sha256"
            ],
            "bytes": amendment["official_annotation_artifact"][
                "combined_bytes"
            ],
            "license": "CC-BY-NC-4.0",
        },
        {
            "source": "VISOR_HOS_SOURCE_FEASIBILITY_RECORD",
            "sha256": source_commitment,
            "bytes": (
                fixture_root / "visor-hos-source-feasibility.json"
            ).stat().st_size,
            "license": "public aggregate and selection metadata",
        },
        {
            "source": "VISOR_HOS_VERIFIED_NO_HAND_REVIEW_SEAL",
            "sha256": verified["commitment_sha256"],
            "bytes": (review_root / "verified-no-hand-seal.json").stat().st_size,
            "license": "self-authored public review metadata",
        },
        *(zip_records[key] for key in sorted(zip_records)),
    ]
    lineage = {
        "visor_hos_correction_amendment_commitment_sha256": amendment[
            "amendment_commitment_sha256"
        ],
        "visor_hos_source_feasibility_commitment_sha256": source_commitment,
        "verified_no_hand_seal_commitment_sha256": verified[
            "commitment_sha256"
        ],
        "construct_aligned_ltx_resume_amendment_commitment_sha256": active[
            "amendment_commitment_sha256"
        ],
        **frame_audits,
    }
    return output, provenance, lineage


def _source_feasibility_failure(error: Exception) -> dict[str, Any]:
    message = str(error)
    code = message if re.fullmatch(r"E_[A-Z0-9_]+", message) else type(error).__name__
    return {
        "status": (
            "BLOCKED_ENGINEERING"
            if isinstance(error, (OSError, urllib.error.URLError))
            else "NO_GO"
        ),
        "error_code": code,
    }


def prepare_tuple_visor_hos_fixture_feasibility(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run every independent source check for the active VISOR-HOS recipe."""

    cfg = json.loads(args.config.read_text())
    protocol = _tuple_fixture_protocol(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    repair = _tuple_fixture_feasibility_repair(cfg)
    amendment = _tuple_visor_hos_correction_amendment(cfg)
    fixture_root = _tuple_fixture_root(args.public_root)
    _require_external_or_ignored_output(args.public_root)
    fixture_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    families: dict[str, dict[str, Any]] = {}
    selections: dict[str, Any] = {}
    artifact_paths: list[Path] = []
    correction_exclusions: set[tuple[str, str]] = set()
    try:
        artifact_paths, artifact_provenance = (
            _load_official_visor_hos_annotation_paths(fixture_root, amendment)
        )
        families["official_visor_hos_artifact"] = {
            "status": "PASS",
            "file_count": artifact_provenance["file_count"],
            "bytes": artifact_provenance["bytes"],
            "artifact_provenance_commitment_sha256": artifact_provenance[
                "artifact_provenance_commitment_sha256"
            ],
        }
    except Exception as error:
        families["official_visor_hos_artifact"] = _source_feasibility_failure(
            error
        )
    try:
        correction_exclusions, semantic_record = (
            _load_visor_hos_correction_exclusions(fixture_root, amendment)
        )
        families["visor_hos_semantic_reference"] = {
            "status": "PASS",
            "correction_frame_count": len(correction_exclusions),
            "semantic_reference_commitment_sha256": semantic_record[
                "semantic_reference_commitment_sha256"
            ],
        }
    except Exception as error:
        families["visor_hos_semantic_reference"] = _source_feasibility_failure(
            error
        )

    sources = preparation["source_archives"]
    extracted = fixture_root / "sources/extracted"
    coco: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    try:
        coco_annotations = _fixture_archive(
            args.public_root,
            sources["COCO_2017_instances"],
            "annotations_trainval2017.zip",
        )
        _safe_extract_zip(coco_annotations, extracted / "coco-annotations")
        instances = json.loads(
            (
                extracted
                / "coco-annotations/annotations/instances_val2017.json"
            ).read_text()
        )
        coco = _select_coco_object_sources(instances, preparation, repair)
        families["coco_composite_sources"] = {
            "status": "PASS",
            "counts": {
                partition: len(coco[partition])
                for partition in preparation["partitions"]
            },
        }
        selections["coco"] = coco
    except Exception as error:
        families["coco_composite_sources"] = _source_feasibility_failure(error)

    language_rows: dict[str, list[dict[str, Any]]] = {}
    try:
        language_rows = {
            partition: _language_lexical_fixture_rows(preparation, partition)
            for partition in preparation["partitions"]
        }
        expected_language = int(
            amendment["public_fixture_counts_per_partition"][
                "language_and_lexical_items"
            ]
        )
        language_counts = {
            partition: len(rows) for partition, rows in language_rows.items()
        }
        families["language_and_lexical"] = {
            "status": (
                "PASS"
                if all(value == expected_language for value in language_counts.values())
                else "NO_GO"
            ),
            "counts": language_counts,
            "required_per_partition": expected_language,
        }
        selections["language_and_lexical"] = language_rows
    except Exception as error:
        families["language_and_lexical"] = _source_feasibility_failure(error)

    coco_ready = families["coco_composite_sources"]["status"] == "PASS"
    ontology_count = len(preparation["public_object_ontology"])
    referent_count = ontology_count * len(
        preparation["referent_attribute_rendering"]["scenarios_once_per_category"]
    )
    recurrence_count = sum(
        int(value)
        for value in preparation["recurrence_recipe"][
            "strata_per_partition"
        ].values()
    )
    sensor_count = int(
        preparation["sensor_recipe"]["base_scenes_per_partition"]
    ) * len(preparation["sensor_recipe"]["conditions"])
    derived_source_checks = (
        (
            "referent_attribute_composite",
            referent_count,
            int(
                amendment["public_fixture_counts_per_partition"][
                    "referent_attribute_microclips"
                ]
            ),
        ),
        (
            "recurrence",
            recurrence_count,
            int(
                amendment["public_fixture_counts_per_partition"][
                    "recurrence_pairs"
                ]
            ),
        ),
        (
            "sensor",
            sensor_count,
            int(
                amendment["public_fixture_counts_per_partition"][
                    "sensor_perturbation_clips"
                ]
            ),
        ),
    )
    for family, count, required in derived_source_checks:
        if not coco_ready:
            families[family] = {
                "status": "DEPENDENCY_UNAVAILABLE",
                "dependency": "coco_composite_sources",
                "counts": {"development": 0, "holdout": 0},
                "required_per_partition": required,
            }
        else:
            families[family] = {
                "status": "PASS" if count == required else "NO_GO",
                "counts": {"development": count, "holdout": count},
                "required_per_partition": required,
            }

    visor: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    visor_report: dict[str, Any] = {}
    if (
        families["official_visor_hos_artifact"]["status"] == "PASS"
        and families["visor_hos_semantic_reference"]["status"] == "PASS"
    ):
        try:
            sampler = amendment["partition_and_joint_sampler"]
            visor, visor_report = _visor_hos_source_inventory(
                _iter_visor_hos_documents(artifact_paths),
                seed=int(sampler["seed"]),
                target_per_stratum=int(sampler["quota_per_partition_per_stratum"]),
                no_hand_review_queue_ceiling=192,
                per_video_stratum_cap=int(sampler["per_video_per_stratum_cap"]),
                correction_excluded_frame_keys=correction_exclusions,
            )
            no_hand_queues, no_hand_queue_report = (
                _visor_hos_no_hand_review_nominees(
                    _iter_visor_hos_documents(artifact_paths),
                    seed=int(sampler["seed"]),
                    per_video_cap=int(sampler["per_video_per_stratum_cap"]),
                    correction_excluded_frame_keys=correction_exclusions,
                )
            )
            for partition in preparation["partitions"]:
                visor[partition] = [
                    row
                    for row in visor[partition]
                    if row["stratum"] != "no_hand_nominee"
                ] + no_hand_queues[partition]
                visor[partition].sort(
                    key=lambda row: (
                        VISOR_HOS_STRATA.index(row["stratum"])
                        if row["stratum"] in VISOR_HOS_STRATA
                        else 2,
                        _visor_hos_candidate_order(
                            int(sampler["seed"]), partition, row
                        ),
                    )
                )
                visor_report["final_counts"][partition][
                    "no_hand_nominee"
                ] = len(no_hand_queues[partition])
            visor_report["no_hand_review_queue_inventory"] = no_hand_queue_report
            selections["visor_hos_source_nominees"] = visor
            for stratum, family in (
                ("contact", "visor_hos_contact"),
                ("explicit_no_contact", "visor_hos_explicit_no_contact"),
                ("no_hand_nominee", "visor_hos_no_hand_nominees"),
            ):
                counts = {
                    partition: visor_report["final_counts"][partition][stratum]
                    for partition in preparation["partitions"]
                }
                families[family] = {
                    "status": (
                        "PASS_NOMINEE_QUEUE_READY"
                        if stratum == "no_hand_nominee"
                        and all(value >= 48 for value in counts.values())
                        else (
                            "PASS"
                            if stratum != "no_hand_nominee"
                            and all(value == 48 for value in counts.values())
                            else "NO_GO"
                        )
                    ),
                    "counts": counts,
                    "required_per_partition": 48,
                }
            families["visor_hos_integrity"] = {
                "status": (
                    "PASS"
                    if visor_report["status"] == "PASS_SOURCE_NOMINEES"
                    else "NO_GO"
                ),
                "participant_overlap_count": visor_report[
                    "participant_overlap_count"
                ],
                "video_overlap_count": visor_report["video_overlap_count"],
                "frame_overlap_count": visor_report["frame_overlap_count"],
                "correction_values_applied": False,
            }
        except Exception as error:
            failure = _source_feasibility_failure(error)
            for family in (
                "visor_hos_contact",
                "visor_hos_explicit_no_contact",
                "visor_hos_no_hand_nominees",
                "visor_hos_integrity",
            ):
                families[family] = dict(failure)
    else:
        for family in (
            "visor_hos_contact",
            "visor_hos_explicit_no_contact",
            "visor_hos_no_hand_nominees",
            "visor_hos_integrity",
        ):
            families[family] = {
                "status": "DEPENDENCY_UNAVAILABLE",
                "dependency": "official_artifact_or_semantic_reference",
            }

    action: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    action_report: dict[str, Any] = {}
    try:
        charades_annotations = _fixture_archive(
            args.public_root,
            sources["Charades_Ego_annotations"],
            "CharadesEgo.zip",
        )
        _safe_extract_zip(charades_annotations, extracted / "charades-annotations")
        (
            excluded_subjects,
            excluded_videos,
            prior_activity_exclusion,
        ) = _prior_activity_exclusions(extracted / "charades-annotations", cfg)
        action, action_report = _charades_action_source_inventory(
            _load_charades_rows(extracted / "charades-annotations"),
            protocol["order_dependent_action_control"],
            int(preparation["seed"]),
            excluded_subjects,
            excluded_videos,
        )
        families["charades_order_action"] = {
            "status": action_report["status"],
            "counts": {
                partition: len(action[partition])
                for partition in preparation["partitions"]
            },
            "deficits": action_report["deficits"],
            "prior_activity_exclusion": prior_activity_exclusion,
        }
        selections["charades_order_action"] = action
    except Exception as error:
        families["charades_order_action"] = _source_feasibility_failure(error)

    subject_sets = {
        partition: {
            *(f"visor:{row['participant']}" for row in visor[partition]),
            *(f"charades:{row['subject']}" for row in action[partition]),
        }
        for partition in preparation["partitions"]
    }
    video_sets = {
        partition: {
            *(f"visor:{row['video']}" for row in visor[partition]),
            *(f"charades:{row['video']}" for row in action[partition]),
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
    families["cross_partition_source_independence"] = {
        "status": "PASS" if not any(audits.values()) else "NO_GO",
        **audits,
    }
    nonpass = {
        name: record
        for name, record in families.items()
        if not str(record.get("status", "")).startswith("PASS")
    }
    blocked = any(
        record.get("status") == "BLOCKED_ENGINEERING"
        for record in nonpass.values()
    )
    status = (
        "BLOCKED_COMPLETE_SOURCE_FEASIBILITY_ENGINEERING"
        if blocked
        else (
            "NO_GO_COMPLETE_SOURCE_FEASIBILITY"
            if nonpass
            else "PASS_SOURCE_FEASIBILITY_PENDING_NO_HAND_REVIEW"
        )
    )
    record = {
        "schema_version": 2,
        "status": status,
        "visor_hos_correction_amendment_commitment_sha256": amendment[
            "amendment_commitment_sha256"
        ],
        "fixture_preparation_amendment_commitment_sha256": preparation[
            "preparation_amendment_commitment_sha256"
        ],
        "fixture_feasibility_repair_commitment_sha256": repair[
            "fixture_feasibility_repair_commitment_sha256"
        ],
        "public_fixture_protocol_commitment_sha256": protocol[
            "protocol_commitment_sha256"
        ],
        "families": families,
        "failing_family_names": sorted(nonpass),
        "selections": selections,
        "visor_hos_inventory": visor_report,
        "action_inventory": action_report,
        "audits": audits,
        "no_hand_truth_opened": False,
        "no_hand_review_required_before_public_model_inference": True,
        "model_inference_executed": False,
        "media_rendering_executed": False,
        "large_Charades_video_archive_downloaded": False,
        "restricted_mount_present": False,
    }
    record["visor_hos_source_feasibility_commitment_sha256"] = digest(record)
    write_private(fixture_root / "visor-hos-source-feasibility.json", record)
    raw_visor = visor_report.get("raw_eligible_counts", {})
    artifact_family = families["official_visor_hos_artifact"]
    return {
        "status": status,
        "official_annotation_file_count": int(
            artifact_family.get("file_count", 0)
        ),
        "official_annotation_bytes": int(artifact_family.get("bytes", 0)),
        "coco_source_count": sum(len(rows) for rows in coco.values()),
        "visor_contact_candidate_count": int(raw_visor.get("contact", 0)),
        "visor_no_contact_candidate_count": int(
            raw_visor.get("explicit_no_contact", 0)
        ),
        "visor_no_hand_nominee_count": int(
            raw_visor.get("no_hand_nominee", 0)
        ),
        "action_item_count": sum(len(rows) for rows in action.values()),
        "language_lexical_item_count": sum(
            len(rows) for rows in language_rows.values()
        ),
        "referent_attribute_item_count": (
            2 * referent_count if families["referent_attribute_composite"]["status"] == "PASS" else 0
        ),
        "recurrence_pair_count": (
            2 * recurrence_count if families["recurrence"]["status"] == "PASS" else 0
        ),
        "sensor_item_count": (
            2 * sensor_count if families["sensor"]["status"] == "PASS" else 0
        ),
        "partition_count": 2,
        "failing_family_count": len(nonpass),
        "pending_dependent_family_count": int(
            families["visor_hos_no_hand_nominees"]["status"]
            == "PASS_NOMINEE_QUEUE_READY"
        ),
        **audits,
        "model_inference_executed": False,
        "media_rendering_executed": False,
        "restricted_mount_present": False,
        "visor_hos_source_feasibility_commitment_sha256": record[
            "visor_hos_source_feasibility_commitment_sha256"
        ],
    }


def _prepare_tuple_fixture_feasibility_legacy(
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
            "blocking_family": "VISOR_HAND_CONTACT",
            "blocking_partition": partition.upper(),
            "blocking_stratum": stratum.upper(),
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
            "blocking_family": "VISOR_HAND_CONTACT",
            "blocking_partition": partition.upper(),
            "blocking_stratum": stratum.upper(),
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
    excluded_subjects, excluded_videos, _ = _prior_activity_exclusions(
        extracted / "charades-annotations", cfg
    )
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
        "blocking_family": "NONE",
        "blocking_partition": "NONE",
        "blocking_stratum": "NONE",
        "required_count": 0,
        "available_count": 0,
        **audits,
        "model_inference_executed": False,
        "media_rendering_executed": False,
        "restricted_mount_present": False,
        "fixture_feasibility_commitment_sha256": record[
            "fixture_feasibility_commitment_sha256"
        ],
    }


def prepare_tuple_fixture_feasibility(args: argparse.Namespace) -> dict[str, Any]:
    recipe = str(getattr(args, "recipe", "active-visor-hos"))
    if recipe == "active-visor-hos":
        return prepare_tuple_visor_hos_fixture_feasibility(args)
    if recipe == "legacy-sealed":
        return _prepare_tuple_fixture_feasibility_legacy(args)
    raise RuntimeError("E_TUPLE_FIXTURE_FEASIBILITY_RECIPE")


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
    grounding_definitions = next(
        axis["definitions"]
        for axis in _tuple_amendment(cfg)["axes"]
        if axis["id"]
        == "utterance_centered_referent_visibility_dominance_ambiguity"
    )
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
        for category in ontology:
            targets = crop_records[partition][category]
            for scenario_ordinal, scenario in enumerate(scenarios):
                source_index = _referent_attribute_source_index(
                    scenario_ordinal, len(targets)
                )
                target_record, distractor_record = (
                    _referent_attribute_source_records(
                        targets, scenario_ordinal, scenario
                    )
                )
                target_crop = Image.open(
                    fixture_root / target_record["crop_relative_path"]
                ).convert("RGBA")
                distractor_crop = (
                    Image.open(
                        fixture_root / distractor_record["crop_relative_path"]
                    ).convert("RGBA")
                    if distractor_record is not None
                    else None
                )
                frames, target_masks, distractor_masks, truth = (
                    _render_referent_fixture(
                        preparation,
                        partition,
                        category,
                        scenario,
                        scenario_ordinal,
                        target_crop,
                        distractor_crop,
                        grounding_definitions,
                    )
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
                np.savez_compressed(
                    mask_path,
                    target_mask=target_masks,
                    distractor_mask=distractor_masks,
                )
                os.chmod(mask_path, 0o600)
                sampled_mask_rows = []
                for sample_ordinal, sample in enumerate(
                    truth.pop("sampled_mask_truth")
                ):
                    frame_index = int(sample["frame_index"])
                    mask_row = dict(sample)
                    for role, masks in (
                        ("target", target_masks),
                        ("distractor", distractor_masks),
                    ):
                        sample_mask_path = (
                            mask_path.parent
                            / f"{fixture_ordinal:03d}-{sample_ordinal:02d}-{role}.png"
                        )
                        Image.fromarray(
                            (masks[frame_index] > 0).astype(np.uint8) * 255,
                            mode="L",
                        ).save(sample_mask_path, format="PNG")
                        os.chmod(sample_mask_path, 0o600)
                        mask_row[f"{role}_mask_relative_path"] = str(
                            sample_mask_path.relative_to(fixture_root)
                        )
                        mask_row[f"{role}_mask_sha256"] = file_digest(
                            sample_mask_path
                        )
                        mask_row[f"{role}_mask_bytes"] = (
                            sample_mask_path.stat().st_size
                        )
                    sampled_mask_rows.append(mask_row)
                truth["sampled_mask_truth"] = sampled_mask_rows
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
                attribute_pair_id = (
                    _referent_attribute_episode_id(
                        partition,
                        category,
                        scenario,
                        scenario_ordinal,
                        source_index,
                    )
                    if truth["attribute_family"] == "relative_size"
                    else None
                )
                outputs[partition]["referent_attribute"].append(
                    {
                        "fixture_ordinal": fixture_ordinal,
                        "category": category,
                        "scenario": scenario,
                        "source_image_id": target_record["image_id"],
                        "source_annotation_id": target_record["annotation_id"],
                        "source_image_sha256": target_record["source_image_sha256"],
                        "distractor_source_category": (
                            distractor_record["category"]
                            if distractor_record is not None
                            else None
                        ),
                        "distractor_source_image_id": (
                            distractor_record["image_id"]
                            if distractor_record is not None
                            else None
                        ),
                        "distractor_source_annotation_id": (
                            distractor_record["annotation_id"]
                            if distractor_record is not None
                            else None
                        ),
                        "distractor_source_image_sha256": (
                            distractor_record["source_image_sha256"]
                            if distractor_record is not None
                            else None
                        ),
                        "distractor_source_distinct_from_target": (
                            True if distractor_record is not None else None
                        ),
                        "attribute_pair_source_index": source_index,
                        "attribute_pair_id": attribute_pair_id,
                        "episode_id": _referent_attribute_episode_id(
                            partition,
                            category,
                            scenario,
                            scenario_ordinal,
                            source_index,
                        ),
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
                first, second, first_mask, second_mask = _render_recurrence_pair(
                    first_crop, second_crop, stratum, ordinal
                )
                pair_ordinal = len(outputs[partition]["recurrence"])
                pair_root = fixture_root / "media/recurrence" / partition
                pair_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                mask_root = fixture_root / "truth/recurrence" / partition
                mask_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                first_path = pair_root / f"{pair_ordinal:03d}-a.png"
                second_path = pair_root / f"{pair_ordinal:03d}-b.png"
                first_mask_path = mask_root / f"{pair_ordinal:03d}-a-mask.png"
                second_mask_path = mask_root / f"{pair_ordinal:03d}-b-mask.png"
                first.save(first_path, format="PNG")
                second.save(second_path, format="PNG")
                first_mask.save(first_mask_path, format="PNG")
                second_mask.save(second_mask_path, format="PNG")
                os.chmod(first_path, 0o600)
                os.chmod(second_path, 0o600)
                os.chmod(first_mask_path, 0o600)
                os.chmod(second_mask_path, 0o600)
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
                        "first_bytes": first_path.stat().st_size,
                        "second_relative_path": str(second_path.relative_to(fixture_root)),
                        "second_sha256": file_digest(second_path),
                        "second_bytes": second_path.stat().st_size,
                        "first_mask_relative_path": str(
                            first_mask_path.relative_to(fixture_root)
                        ),
                        "first_mask_sha256": file_digest(first_mask_path),
                        "first_mask_bytes": first_mask_path.stat().st_size,
                        "second_mask_relative_path": str(
                            second_mask_path.relative_to(fixture_root)
                        ),
                        "second_mask_sha256": file_digest(second_mask_path),
                        "second_mask_bytes": second_mask_path.stat().st_size,
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
    for partition in preparation["partitions"]:
        _validate_tuple_recurrence_fixture_rows(
            outputs[partition]["recurrence"], fixture_root
        )
    return outputs, object_sets


def _prepare_action_fixtures(
    args: argparse.Namespace,
    fixture_root: Path,
    extracted: Path,
    preparation: dict[str, Any],
    protocol: dict[str, Any],
    cfg: dict[str, Any],
    source_record: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    del extracted, protocol, cfg
    selected = json.loads(
        json.dumps(source_record["selections"]["charades_order_action"])
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
        _validate_construct_aligned_action_fixture_projection(
            output[partition], selected[partition], partition
        )
    sets = {
        partition: {row["video"] for row in output[partition]}
        for partition in preparation["partitions"]
    }
    archive.unlink()
    return output, sets


def _tuple_fixture_preparation_compact(
    fixture_manifest: dict[str, Any],
) -> dict[str, Any]:
    partitions = fixture_manifest["partitions"]
    total = {
        family: sum(len(partitions[partition][family]) for partition in partitions)
        for family in (
            "language_lexical",
            "referent_attribute",
            "recurrence",
            "hand_contact",
            "sensor",
            "order_action",
        )
    }
    audits = fixture_manifest["audits"]
    return {
        "status": "PASS_PUBLIC_FIXTURES_SEALED_NO_MODEL_INFERENCE",
        "source_archive_count": len(fixture_manifest["source_provenance"]),
        "partition_count": len(partitions),
        "language_lexical_item_count": total["language_lexical"],
        "referent_attribute_item_count": total["referent_attribute"],
        "recurrence_pair_count": total["recurrence"],
        "hand_contact_item_count": total["hand_contact"],
        "sensor_item_count": total["sensor"],
        "order_action_item_count": total["order_action"],
        "source_subject_overlap_count": audits["source_subject_overlap_count"],
        "source_video_overlap_count": audits["source_video_overlap_count"],
        "source_frame_overlap_count": audits["source_frame_overlap_count"],
        "source_object_overlap_count": audits["source_object_overlap_count"],
        "target_hand_boundary_vertex_count": audits[
            "target_hand_boundary_vertex_count"
        ],
        "target_hand_outside_canvas_vertex_count": audits[
            "target_hand_outside_canvas_vertex_count"
        ],
        "target_hand_boundary_item_count": audits[
            "target_hand_boundary_item_count"
        ],
        "target_hand_outside_canvas_item_count": audits[
            "target_hand_outside_canvas_item_count"
        ],
        "restricted_mount_present": False,
        "model_inference_executed": False,
        "public_fixture_manifest_commitment_sha256": fixture_manifest[
            "public_fixture_manifest_commitment_sha256"
        ],
    }


def prepare_tuple_fixtures(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(args.config.read_text())
    fixture_root = _tuple_fixture_root(args.public_root)
    _require_external_or_ignored_output(args.public_root)
    qualification_paths = {
        **{
            f"new_{key}": value
            for key, value in _tuple_qualification_paths(
                args.public_root
            ).items()
        },
        **{
            f"legacy_{key}": value
            for key, value in _tuple_legacy_qualification_paths(
                args.public_root
            ).items()
        },
    }
    if any(path.exists() for path in qualification_paths.values()):
        raise RuntimeError("E_TUPLE_FIXTURE_REPLACEMENT_AFTER_QUALIFICATION")
    active = _construct_aligned_ltx_resume_amendment(cfg)
    amendment = _tuple_amendment(cfg)
    correction = _tuple_visor_hos_correction_amendment(cfg)
    rasterization_repair = _public_fixture_geometry_rasterization_repair(cfg)
    protocol = _tuple_fixture_protocol(cfg)
    preparation = _tuple_fixture_preparation_amendment(cfg)
    feasibility_repair = _tuple_fixture_feasibility_repair(cfg)
    _verify_tuple_runtime_manifest(args.public_root, cfg)
    fixture_manifest_path = fixture_root / "fixture-manifest.json"
    if fixture_manifest_path.exists():
        before = file_digest(fixture_manifest_path)
        existing, _ = _verify_tuple_fixture_manifest(args.public_root, cfg)
        if file_digest(fixture_manifest_path) != before:
            raise RuntimeError("E_TUPLE_FIXTURE_MANIFEST_CHANGED_DURING_REUSE")
        return _tuple_fixture_preparation_compact(existing)
    fixture_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    review_root = Path(
        getattr(args, "no_hand_review_root", None)
        or fixture_root / "no-hand-review"
    )
    if review_root.resolve() != (fixture_root / "no-hand-review").resolve():
        raise RuntimeError("E_VISOR_HOS_NO_HAND_REVIEW_CANONICAL_ROOT")
    source_record = _load_construct_aligned_visor_hos_source_reuse(
        fixture_root, cfg
    )
    audio_manifest, audio_files = _read_audio_seed_manifest(
        args.audio_seed_root, cfg
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
    visor, visor_provenance, visor_lineage = _prepare_active_visor_hos_fixtures(
        fixture_root, preparation, cfg, review_root, source_record
    )
    action, action_video_sets = _prepare_action_fixtures(
        args,
        fixture_root,
        extracted,
        preparation,
        protocol,
        cfg,
        source_record,
    )
    for partition in preparation["partitions"]:
        partitions[partition]["hand_contact"] = visor[partition]
        partitions[partition]["order_action"] = action[partition]
    subject_sets = {
        partition: {
            *(f"visor:{row['source_participant']}" for row in visor[partition]),
            *(f"charades:{row['subject']}" for row in action[partition]),
        }
        for partition in preparation["partitions"]
    }
    visor_video_sets = {
        partition: {row["source_video"] for row in visor[partition]}
        for partition in preparation["partitions"]
    }
    video_sets = {
        partition: {
            *(f"visor:{value}" for value in visor_video_sets[partition]),
            *(f"charades:{value}" for value in action_video_sets[partition]),
        }
        for partition in preparation["partitions"]
    }
    frame_sets = {
        partition: {
            (row["source_video"], row["source_frame_name"])
            for row in visor[partition]
        }
        for partition in preparation["partitions"]
    }
    if any(
        len(frame_sets[partition]) != len(visor[partition])
        for partition in preparation["partitions"]
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FRAME_DUPLICATE")
    audits = {
        "source_subject_overlap_count": len(
            subject_sets["development"] & subject_sets["holdout"]
        ),
        "source_video_overlap_count": len(
            video_sets["development"] & video_sets["holdout"]
        ),
        "source_frame_overlap_count": len(
            frame_sets["development"] & frame_sets["holdout"]
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
        "all_source_polygons_finite_nonnegative": True,
        "all_rasterized_target_masks_binary_nonempty_exact_frame_in_bounds": True,
        "target_hand_boundary_vertex_count": sum(
            int(row["target_hand_boundary_vertex_count"])
            for partition in preparation["partitions"]
            for row in visor[partition]
        ),
        "target_hand_outside_canvas_vertex_count": sum(
            int(row["target_hand_outside_canvas_vertex_count"])
            for partition in preparation["partitions"]
            for row in visor[partition]
        ),
        "target_hand_boundary_item_count": sum(
            int(int(row["target_hand_boundary_vertex_count"]) > 0)
            for partition in preparation["partitions"]
            for row in visor[partition]
        ),
        "target_hand_outside_canvas_item_count": sum(
            int(int(row["target_hand_outside_canvas_vertex_count"]) > 0)
            for partition in preparation["partitions"]
            for row in visor[partition]
        ),
    }
    if any(
        audits[key] != 0
        for key in (
            "source_subject_overlap_count",
            "source_video_overlap_count",
            "source_frame_overlap_count",
            "source_object_overlap_count",
        )
    ):
        raise RuntimeError("E_TUPLE_FIXTURE_PARTITION_OVERLAP")
    if (
        audits["source_frame_overlap_count"]
        != visor_lineage["source_frame_overlap_count"]
    ):
        raise RuntimeError("E_VISOR_HOS_SOURCE_FRAME_AUDIT_MISMATCH")
    active_counts = correction["public_fixture_counts_per_partition"]
    expected_counts = {
        "language_lexical": active_counts["language_and_lexical_items"],
        "referent_attribute": active_counts["referent_attribute_microclips"],
        "recurrence": active_counts["recurrence_pairs"],
        "hand_contact": active_counts["hand_contact_items"],
        "sensor": active_counts["sensor_perturbation_clips"],
        "order_action": CONSTRUCT_ALIGNED_ACTION_COUNTS["development"],
    }
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
        "schema_version": 3,
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
        "visor_hos_correction_amendment_commitment_sha256": visor_lineage[
            "visor_hos_correction_amendment_commitment_sha256"
        ],
        "visor_hos_source_feasibility_commitment_sha256": visor_lineage[
            "visor_hos_source_feasibility_commitment_sha256"
        ],
        "verified_no_hand_seal_commitment_sha256": visor_lineage[
            "verified_no_hand_seal_commitment_sha256"
        ],
        "construct_aligned_ltx_resume_amendment_commitment_sha256": active[
            "amendment_commitment_sha256"
        ],
        "public_fixture_geometry_rasterization_repair_commitment_sha256": (
            rasterization_repair["repair_commitment_sha256"]
        ),
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
    write_private_new(fixture_manifest_path, fixture_manifest)
    return _tuple_fixture_preparation_compact(fixture_manifest)


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


def _resolve_tuple_egohod_checkpoint(
    public: Path, candidate: dict[str, Any]
) -> Path:
    """Resolve only the frozen official EgoHOD bytes from known cache layouts."""

    expected = candidate.get("weight_sha256")
    if (
        candidate.get("candidate_id") != "egohod_egovideo_l_zero_shot"
        or expected
        != "71faa0b6e5ebfb912238a099b16b1ff8b6b0a74cbb5b9eb43d5ad8bc92f880da"
    ):
        raise RuntimeError("E_ACTIVITY_WEIGHT_COMMITMENT")
    candidates = (
        _activity_checkpoint_root(public, candidate["candidate_id"])
        / candidate["weight_file"],
        public / "models/activity-checkpoints/egovideo_large_best.pt",
        _tuple_model_root(public) / "weights/egohod_large_best.pt",
    )
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise RuntimeError("E_ACTIVITY_WEIGHT_COMMITMENT")
    observed = {path: file_digest(path) for path in existing}
    if any(value != expected for value in observed.values()):
        raise RuntimeError("E_ACTIVITY_WEIGHT_COMMITMENT")
    return next(path for path in candidates if path in observed)


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

    checkpoint = _resolve_tuple_egohod_checkpoint(public, candidate)
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


def _tuple_egohos_config_item(value: Any, key: str) -> Any:
    try:
        return value[key]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_TUPLE_EGOHOS_TEST_PIPELINE") from error


def _validate_tuple_egohos_test_pipeline(config: Any) -> None:
    """Require the pinned official mmseg test preprocessing recipe exactly."""

    data = _tuple_egohos_config_item(config, "data")
    test = _tuple_egohos_config_item(data, "test")
    pipeline = _tuple_egohos_config_item(test, "pipeline")
    if not isinstance(pipeline, (list, tuple)) or len(pipeline) != 2:
        raise RuntimeError("E_TUPLE_EGOHOS_TEST_PIPELINE")
    load = pipeline[0]
    augment = pipeline[1]
    transforms = _tuple_egohos_config_item(augment, "transforms")
    if (
        _tuple_egohos_config_item(load, "type") != "LoadImageFromFile"
        or _tuple_egohos_config_item(augment, "type") != "MultiScaleFlipAug"
        or tuple(_tuple_egohos_config_item(augment, "img_scale")) != (360, 480)
        or _tuple_egohos_config_item(augment, "flip") is not False
        or not isinstance(transforms, (list, tuple))
        or [
            _tuple_egohos_config_item(item, "type") for item in transforms
        ]
        != ["Resize", "RandomFlip", "Normalize", "ImageToTensor", "Collect"]
        or _tuple_egohos_config_item(transforms[0], "keep_ratio") is not True
        or list(_tuple_egohos_config_item(transforms[2], "mean"))
        != [123.675, 116.28, 103.53]
        or list(_tuple_egohos_config_item(transforms[2], "std"))
        != [58.395, 57.12, 57.375]
        or _tuple_egohos_config_item(transforms[2], "to_rgb") is not True
        or list(_tuple_egohos_config_item(transforms[3], "keys")) != ["img"]
        or list(_tuple_egohos_config_item(transforms[4], "keys")) != ["img"]
    ):
        raise RuntimeError("E_TUPLE_EGOHOS_TEST_PIPELINE")


def _tuple_egohos_official_stage(
    model: Any,
    config: Any,
    image_paths: list[Path],
    original_shapes: list[tuple[int, int]],
    *,
    stage: str,
    run_root: Path,
    output_folder: str | None,
    inference_segmentor: Any,
) -> list[Any]:
    """Run one official mmseg stage and retain its original-size class maps."""

    from PIL import Image

    if len(image_paths) != len(original_shapes) or not image_paths:
        raise RuntimeError("E_TUPLE_EGOHOS_IMAGE_COUNT")
    _validate_tuple_egohos_test_pipeline(config)
    expected_channel = {
        "stage1": "none",
        "stage2": "twohands",
        "stage3": "twohands_cb",
    }.get(stage)
    if expected_channel is None:
        raise RuntimeError("E_TUPLE_EGOHOS_STAGE_ID")
    existing_channel = config.get("additional_channel")
    if existing_channel is not None and not isinstance(existing_channel, str):
        raise RuntimeError("E_TUPLE_EGOHOS_ADDITIONAL_CHANNEL")
    normalized_channel = (
        existing_channel.casefold() if isinstance(existing_channel, str) else None
    )
    allowed_channels = (
        {None, "", "none"}
        if stage == "stage1"
        else {None, "", expected_channel}
    )
    if normalized_channel not in allowed_channels:
        raise RuntimeError("E_TUPLE_EGOHOS_ADDITIONAL_CHANNEL")
    config["additional_channel"] = (
        existing_channel or ""
        if stage == "stage1"
        else expected_channel
    )
    model.cfg = config
    output = []
    for image_path, original_shape in zip(
        image_paths, original_shapes, strict=True
    ):
        result = inference_segmentor(model, str(image_path))
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError("E_TUPLE_EGOHOS_OFFICIAL_RESULT")
        prediction = _tuple_egohos_stage_class_map(
            stage, result[0], expected_shape=original_shape
        )
        output.append(prediction)
        if output_folder is not None:
            target_root = run_root / output_folder
            target_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            target = target_root / f"{image_path.stem}.png"
            Image.fromarray(prediction).save(target, format="PNG")
            os.chmod(target, 0o600)
    return output


def _tuple_egohos_run_official_pipeline(
    public: Path,
    device: str,
    image_paths: list[Path],
    original_shapes: list[tuple[int, int]],
    run_root: Path,
) -> list[dict[str, Any]]:
    """Run the official sequential mmseg inference pipeline for all three stages."""

    _install_egohos_segmentation_compatibility()
    model_root = _tuple_model_root(public)
    sys.path.insert(0, str(model_root / "code/EgoHOS/mmsegmentation"))
    sys.path.insert(0, str(model_root / "code/EgoHOS"))
    from mmseg.apis import inference_segmentor

    work = model_root / "egohos-checkpoints/work_dirs"
    predictions = [dict() for _path in image_paths]
    stages = (
        ("stage1", "seg_twohands_ccda", "best_mIoU_iter_56000.pth", "pred_twohands"),
        ("stage2", "twohands_to_cb_ccda", "best_mIoU_iter_76000.pth", "pred_cb"),
        ("stage3", "twohands_cb_to_obj1_ccda", "best_mIoU_iter_34000.pth", None),
    )
    for stage, folder, checkpoint_name, output_folder in stages:
        root = work / folder
        model, config = _load_egohos_segmentor(
            root / f"{folder}.py", root / checkpoint_name, device
        )
        try:
            stage_predictions = _tuple_egohos_official_stage(
                model,
                config,
                image_paths,
                original_shapes,
                stage=stage,
                run_root=run_root,
                output_folder=output_folder,
                inference_segmentor=inference_segmentor,
            )
            for record, prediction in zip(
                predictions, stage_predictions, strict=True
            ):
                record[stage] = prediction
        finally:
            del model
            _release_cuda()
    for prediction in predictions:
        _tuple_egohos_stage_masks(prediction)
    return predictions


def _size_tuple_egohos(
    public: Path, device: str, image_array, scratch: Path
) -> dict[str, Any]:
    from PIL import Image

    media_root = scratch / "egohos"
    image_root = media_root / "images"
    image_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    image_path = image_root / "public-dummy.png"
    Image.fromarray(image_array).save(image_path)
    os.chmod(image_path, 0o600)
    height, width = image_array.shape[:2]
    predictions = _tuple_egohos_run_official_pipeline(
        public,
        device,
        [image_path],
        [(height, width)],
        media_root,
    )
    output_width = sum(
        int(mask.size) for record in predictions for mask in record.values()
    )
    return {"finite": True, "output_width": output_width}


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


def write_private_new(path: Path, value: Any) -> None:
    """Atomically create a private record without ever replacing a seal."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(canonical(value) + b"\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o600)
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise RuntimeError("E_PRIVATE_OUTPUT_ALREADY_EXISTS") from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


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


def _resolve_tuple_pe_core_checkpoint(
    public: Path, cfg: dict[str, Any]
) -> Path:
    """Resolve byte-identical frozen PE-Core copies with a canonical preference."""

    try:
        frozen = cfg["calibration_C"]["extractor"]["vision_model"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("E_FROZEN_VISION_MODEL") from error
    if digest(frozen) != TUPLE_PE_CORE_IDENTITY_SHA256:
        raise RuntimeError("E_FROZEN_VISION_MODEL")
    canonical_path = (
        public / "models/mechanistic-tuples/weights/PE-Core-L14-336.pt"
    )
    cached_path = (
        public
        / "models/pe-hf-home/hub"
        / "models--facebook--PE-Core-L14-336"
        / "snapshots"
        / frozen["revision"]
        / "PE-Core-L14-336.pt"
    )
    existing = [
        path
        for path in (canonical_path, cached_path)
        if path.is_file()
    ]
    if not existing:
        raise RuntimeError("E_FROZEN_VISION_MODEL")
    try:
        observed = {path: file_digest(path) for path in existing}
    except OSError as error:
        raise RuntimeError("E_FROZEN_VISION_MODEL") from error
    if any(value != frozen["weights_sha256"] for value in observed.values()):
        raise RuntimeError("E_FROZEN_VISION_MODEL")
    return canonical_path if canonical_path in observed else sorted(observed)[0]


def _load_vision(
    public: Path,
    cfg: dict[str, Any],
    device: str,
    prompt_groups_override: dict[str, dict[str, list[str]]] | None = None,
):
    import torch
    from apps.alignment_scoring.third_party.perception_models.core.vision_encoder import (
        pe,
        transforms,
    )

    checkpoint = _resolve_tuple_pe_core_checkpoint(public, cfg)
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
    model.load_ckpt(str(checkpoint))
    model = model.to(device).eval()
    transform = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    extractor = cfg["calibration_C"]["extractor"]
    repair = extractor.get("coverage_repair")
    prompt_groups = prompt_groups_override or (
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
    cfg = json.loads(args.config.read_text())
    extractor = cfg["calibration_C"]["extractor"]
    active_tuple = extractor.get(
        "mechanistic_training_tuple_visor_hos_correction_amendment"
    )
    if isinstance(active_tuple, dict) and active_tuple.get("status") == (
        "FROZEN_BEFORE_NEW_PUBLIC_SOURCE_INVENTORY_MODEL_C_GENERATOR_OR_LEARNER_OUTCOMES"
    ):
        raise RuntimeError(
            "E_LEGACY_BROAD_CALIBRATION_SUPERSEDED_BY_ACTIVE_TUPLE_PROTOCOL"
        )
    import imageio_ffmpeg
    import numpy as np

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
    tuple_health_parser = subparsers.add_parser("tuple-health")
    tuple_health_parser.add_argument("--public-root", type=Path, required=True)
    tuple_health_parser.add_argument("--scratch-root", type=Path, required=True)
    tuple_health_parser.add_argument("--config", type=Path, required=True)
    tuple_health_parser.add_argument(
        "--container-attestation", type=Path, required=True
    )
    tuple_health_parser.add_argument(
        "--attempt", type=int, choices=(1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13), required=True
    )
    tuple_health_parser.add_argument("--device", default="cuda")
    tuple_qualify_parser = subparsers.add_parser("tuple-qualify")
    tuple_qualify_parser.add_argument("--public-root", type=Path, required=True)
    tuple_qualify_parser.add_argument("--scratch-root", type=Path, required=True)
    tuple_qualify_parser.add_argument("--config", type=Path, required=True)
    tuple_qualify_parser.add_argument(
        "--container-attestation", type=Path, required=True
    )
    tuple_qualify_parser.add_argument(
        "--fixture-bind-attestation", type=Path, required=True
    )
    tuple_qualify_parser.add_argument(
        "--partition", choices=("development", "holdout"), required=True
    )
    tuple_qualify_parser.add_argument("--device", default="cuda")
    tuple_audio_parser = subparsers.add_parser("tuple-audio-seed")
    tuple_audio_parser.add_argument("--output-root", type=Path, required=True)
    tuple_audio_parser.add_argument("--config", type=Path, required=True)
    tuple_no_hand_prepare_parser = subparsers.add_parser(
        "tuple-no-hand-review-prepare"
    )
    tuple_no_hand_prepare_parser.add_argument(
        "--public-root", type=Path, required=True
    )
    tuple_no_hand_prepare_parser.add_argument("--config", type=Path, required=True)
    tuple_no_hand_seal_parser = subparsers.add_parser("tuple-no-hand-review-seal")
    tuple_no_hand_seal_parser.add_argument("--public-root", type=Path, required=True)
    tuple_no_hand_seal_parser.add_argument("--config", type=Path, required=True)
    tuple_no_hand_seal_parser.add_argument(
        "--authorized-applicant-attested", action="store_true"
    )
    tuple_no_hand_seal_parser.add_argument(
        "--blind-to-egohos-output-attested", action="store_true"
    )
    tuple_no_hand_seal_parser.add_argument(
        "--egohos-inference-not-started-attested", action="store_true"
    )
    tuple_fixtures_parser = subparsers.add_parser("tuple-fixtures-prepare")
    tuple_fixtures_parser.add_argument("--public-root", type=Path, required=True)
    tuple_fixtures_parser.add_argument("--audio-seed-root", type=Path, required=True)
    tuple_fixtures_parser.add_argument("--no-hand-review-root", type=Path)
    tuple_fixtures_parser.add_argument("--config", type=Path, required=True)
    tuple_feasibility_parser = subparsers.add_parser("tuple-fixtures-feasibility")
    tuple_feasibility_parser.add_argument("--public-root", type=Path, required=True)
    tuple_feasibility_parser.add_argument("--config", type=Path, required=True)
    tuple_feasibility_parser.add_argument(
        "--recipe",
        choices=("active-visor-hos", "legacy-sealed"),
        default="active-visor-hos",
    )
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
    elif args.command == "tuple-health":
        value = run_tuple_health(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TUPLE_HEALTH_FIELDS,
                sha256_fields=TUPLE_HEALTH_HASH_FIELDS,
            )
        )
    elif args.command == "tuple-qualify":
        value = qualify_tuple_public(args)
        engineering_only = "scientific_metric_count" in value
        print(
            compact_aggregate_json(
                value,
                allowed_fields=(
                    TUPLE_PARTITION_INTEGRITY_FIELDS
                    if engineering_only
                    else TUPLE_QUALIFICATION_FIELDS
                ),
                sha256_fields=(
                    TUPLE_PARTITION_INTEGRITY_HASH_FIELDS
                    if engineering_only
                    else TUPLE_QUALIFICATION_HASH_FIELDS
                ),
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
    elif args.command == "tuple-no-hand-review-prepare":
        value = prepare_active_visor_hos_no_hand_review(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TUPLE_NO_HAND_REVIEW_PREP_FIELDS,
                sha256_fields=TUPLE_NO_HAND_REVIEW_PREP_HASH_FIELDS,
            )
        )
    elif args.command == "tuple-no-hand-review-seal":
        value = seal_active_visor_hos_no_hand_review(args)
        sealed = "verified_no_hand_seal_commitment_sha256" in value
        print(
            compact_aggregate_json(
                value,
                allowed_fields=(
                    TUPLE_NO_HAND_REVIEW_SEAL_FIELDS
                    if sealed
                    else TUPLE_NO_HAND_REVIEW_INCOMPLETE_SEAL_FIELDS
                ),
                sha256_fields=(
                    TUPLE_NO_HAND_REVIEW_SEAL_HASH_FIELDS
                    if sealed
                    else TUPLE_NO_HAND_REVIEW_INCOMPLETE_SEAL_HASH_FIELDS
                ),
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
        allowed_fields = (
            TUPLE_VISOR_HOS_SOURCE_FEASIBILITY_FIELDS
            if args.recipe == "active-visor-hos"
            else TUPLE_FIXTURE_FEASIBILITY_FIELDS
        )
        hash_fields = (
            TUPLE_VISOR_HOS_SOURCE_FEASIBILITY_HASH_FIELDS
            if args.recipe == "active-visor-hos"
            else TUPLE_FIXTURE_FEASIBILITY_HASH_FIELDS
        )
        print(
            compact_aggregate_json(
                value,
                allowed_fields=allowed_fields,
                sha256_fields=hash_fields,
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
