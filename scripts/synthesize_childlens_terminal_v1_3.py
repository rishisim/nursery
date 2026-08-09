#!/usr/bin/env python3
"""Synthesize the repository-safe ChildLens v1.3 terminal decision.

This program deliberately has no quarantine path, media reader, browser client,
or network client.  It consumes only explicitly named aggregate receipts in the
v1.3 repository namespace.  Missing or malformed evidence is a bounded REVISE,
never an inferred success.  STOP requires an explicit, locked, high-grade
failure receipt; GO requires a completed and irreversibly locked author audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


VERSION = "childlens-model-assisted-terminal-synthesizer-v1.3.0"

AUTHOR_READY = "CHILDLENS_AUTHOR_AUDIT_READY"
GO = "CHILDLENS_MODEL_ASSISTED_FEASIBILITY_GO"
REVISE = "CHILDLENS_MODEL_ASSISTED_FEASIBILITY_REVISE"
STOP = "CHILDLENS_MODEL_ASSISTED_FEASIBILITY_STOP"
TERMINAL_STATES = frozenset({AUTHOR_READY, GO, REVISE, STOP})
TERMINAL_RE = re.compile(
    r"CHILDLENS_(?:AUTHOR_AUDIT_READY|MODEL_ASSISTED_FEASIBILITY_(?:GO|REVISE|STOP))"
)

SELECTED_ITEM_COUNT = 15
PRIMARY_AUDIT_SECONDS = 15 * 60
MAX_ESCALATION_SECONDS = 15 * 60
FULL_PILOT_WINDOW_COUNT = 912
FULL_PILOT_SPEECH_MINUTES = 135.25
MAX_CPU_WORKERS = 4
MAX_MPS_HEAVY_PROCESSES = 1

EXPECTED_HISTORY = {
    "v1": (23, "35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf"),
    "v1_1": (40, "3acd969804d71979ba8071e2446e8ee1ceb3194fc90dcfda802a99e296b7bf48"),
    "v1_2": (53, "f3bbfd9051e33fc33e44b9cab3c7546b77b1acba4904cc73fb116a3cbe7da01f"),
}
V1_PATHS = (
    "docs/childlens_feasibility_v1",
    "output/childlens_feasibility_v1",
    "scripts/validate_childlens_feasibility_v1.py",
    "tests/test_childlens_feasibility_v1.py",
)
V1_1_DIRECTORIES = (
    "docs/childlens_feasibility_v1_1",
    "output/childlens_feasibility_v1_1",
)
V1_2_DIRECTORIES = (
    "docs/childlens_feasibility_v1_2",
    "output/childlens_feasibility_v1_2",
)

REQUIRED_THRESHOLD_NAMES = frozenset(
    {
        "speech_utterance_boundary",
        "source_role_non_child_priority",
        "non_child_transcript_error_and_coverage",
        "coarse_referential_status",
        "noun_object_candidate_precision_coverage",
        "verb_action_candidate_precision_coverage",
    }
)

INPUT_RELATIVE_PATHS = {
    "protocol": "output/childlens_feasibility_v1_3/protocol_freeze_receipt.json",
    "sampling": "output/childlens_feasibility_v1_3/author_audit_sampling_receipt.json",
    "pseudo": "output/childlens_feasibility_v1_3/pseudo_annotation_receipt.json",
    "workflow": "output/childlens_feasibility_v1_3/author_workflow_receipt.json",
}
OPTIONAL_AUDIT_RELATIVE_PATH = (
    "output/childlens_feasibility_v1_3/author_audit_result_receipt.json"
)
EXPECTED_SCHEMAS = {
    "protocol": "childlens-v1.3-protocol-freeze-receipt-v1",
    "sampling": "childlens-author-audit-sampling-receipt-v1.3.0",
    "pseudo": "childlens-v1.3-pseudo-annotation-receipt-v1",
    "workflow": "childlens-v1.3-author-workflow-receipt-v1",
    "audit": "childlens-v1.3-author-audit-result-receipt-v1",
}
OUTPUT_RELATIVE_PATHS = {
    "decision": "output/childlens_feasibility_v1_3/decision_record.json",
    "executive": "docs/childlens_feasibility_v1_3/executive_decision_report.md",
}

FORBIDDEN_KEYS = frozenset(
    {
        "participant_id",
        "child_id",
        "session_id",
        "recording_id",
        "media_id",
        "speaker_id",
        "filename",
        "file_name",
        "source_filename",
        "relative_path",
        "absolute_path",
        "media_path",
        "audio_path",
        "frame_path",
        "transcript_text",
        "utterance_text",
        "lexical_string",
        "raw_hypothesis",
        "model_label_payload",
        "exact_timestamp",
        "timecode",
        "start_time",
        "end_time",
        "cookie",
        "credential",
        "authorization",
        "api_key",
        "access_token",
        "refresh_token",
    }
)
ABSOLUTE_PATH_RE = re.compile(r"(?i)(?:/Users/|file://|\\Users\\)")
MEDIA_NAME_RE = re.compile(
    r"(?i)(?<![a-z0-9_.-])[a-z0-9][a-z0-9_.-]{2,}\."
    r"(?:mp4|mov|mkv|avi|webm|wav|mp3|m4a|aac|flac|jpg|jpeg|png|webp|srt|vtt)\b"
)
UUID_RE = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)


class SynthesisError(RuntimeError):
    """A payload-suppressing failure with a stable public code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Decision:
    terminal_state: str
    basis_code: str
    exact_next_task: str
    author_action_only_remaining: bool
    audit_complete: bool
    escalation_available: bool
    fundamental_failure_established: bool


def _fail(code: str) -> None:
    raise SynthesisError(code)


def _normalise_key(value: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _walk(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield _normalise_key(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _privacy_preflight(value: Mapping[str, Any]) -> None:
    for key, child in _walk(value):
        if key in FORBIDDEN_KEYS:
            _fail("E_INPUT_RESTRICTED_FIELD")
        if isinstance(child, str) and (
            ABSOLUTE_PATH_RE.search(child)
            or MEDIA_NAME_RE.search(child)
            or UUID_RE.search(child)
            or TERMINAL_RE.search(child)
        ):
            _fail("E_INPUT_RESTRICTED_OR_TERMINAL_TEXT")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_size <= 1
            or info.st_size > 2 * 1024 * 1024
        ):
            _fail("E_INPUT_FILE")
        value = json.loads(path.read_text(encoding="utf-8"))
    except SynthesisError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_INPUT_FILE")
    if not isinstance(value, dict):
        _fail("E_INPUT_SCHEMA")
    _privacy_preflight(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_history(root: Path, entries: tuple[str, ...]) -> set[Path]:
    files: set[Path] = set()
    for entry in entries:
        path = root / entry
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(
                child for child in path.rglob("*")
                if child.is_file() and "__pycache__" not in child.parts
            )
    return files


def _history_files(root: Path, version: str) -> list[Path]:
    if version == "v1":
        files = _expand_history(root, V1_PATHS)
    elif version == "v1_1":
        files = _expand_history(root, V1_1_DIRECTORIES)
        for directory in ("scripts", "tests"):
            files.update((root / directory).glob("*childlens*v1_1*"))
    elif version == "v1_2":
        files = _expand_history(root, V1_2_DIRECTORIES)
        for directory in ("scripts", "tests"):
            files.update((root / directory).glob("*childlens*v1_2*"))
    else:  # pragma: no cover
        raise ValueError(version)
    return sorted(
        (path for path in files if path.is_file() and "__pycache__" not in path.parts),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _history_status(root: Path) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for version, (expected_count, expected_digest) in EXPECTED_HISTORY.items():
        files = _history_files(root, version)
        lines = sorted(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
        )
        digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
        status[version] = {
            "artifact_count": len(files),
            "artifact_set_digest": digest,
            "expected_artifact_count": expected_count,
            "expected_artifact_set_digest": expected_digest,
            "preserved": len(files) == expected_count and digest == expected_digest,
        }
    return status


def _is_sha256(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value))


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _all_true(value: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(value.get(key) is True for key in keys)


def _all_false(value: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(value.get(key) is False for key in keys)


def _protocol_valid(value: Mapping[str, Any]) -> bool:
    thresholds = value.get("thresholds")
    threshold_names_ok = bool(
        isinstance(thresholds, dict)
        and set(thresholds) == REQUIRED_THRESHOLD_NAMES
        and all(
            isinstance(spec, dict)
            and spec.get("frozen_before_author_labels") is True
            and isinstance(spec.get("pass_rule"), str)
            and 4 <= len(spec["pass_rule"]) <= 500
            for spec in thresholds.values()
        )
    )
    return bool(
        value.get("schema_version") == EXPECTED_SCHEMAS["protocol"]
        and value.get("status") == "FROZEN"
        and value.get("selected_item_count") == SELECTED_ITEM_COUNT
        and _number(value.get("primary_audit_speech_seconds")) == PRIMARY_AUDIT_SECONDS
        and _number(value.get("maximum_additional_speech_seconds"))
        == MAX_ESCALATION_SECONDS
        and _all_true(
            value,
            (
                "frozen_before_author_labels",
                "selection_hash_deterministic",
                "selection_independent_of_model_predictions",
                "selection_independent_of_model_confidence",
                "selection_independent_of_lexical_content",
                "selection_independent_of_visual_salience",
                "selection_independent_of_apparent_success",
                "author_uses_raw_audio_video",
                "author_blinded_to_model_predictions",
                "author_record_lock_required_before_prediction_reveal",
                "author_record_lock_irreversible",
                "bounded_escalation_only_if_borderline",
                "thresholds_frozen_before_author_labels",
                "cluster_aware_uncertainty_required",
                "small_sample_limitation_required",
                "pseudo_labels_for_aggregate_calibration_only",
                "simulator_oracle_labels_primary_evaluation_truth",
            ),
        )
        and _all_false(
            value,
            (
                "second_human_required",
                "adjudicator_required",
                "inter_human_reliability_available",
                "unaudited_pseudo_labels_primary_evaluation_truth",
            ),
        )
        and value.get("agreement_target") == "MODEL_HUMAN_NOT_INTER_ANNOTATOR"
        and threshold_names_ok
    )


def _sampling_valid(value: Mapping[str, Any]) -> bool:
    primary_seconds = _number(value.get("primary_total_seconds"))
    reserve_seconds = _number(value.get("reserve_total_seconds"))
    per_item_or_fallback = bool(
        value.get("primary_one_minute_per_item") is True
        or value.get("deficit_redistribution_used") is True
    )
    return bool(
        value.get("schema_version") == EXPECTED_SCHEMAS["sampling"]
        and value.get("status") == "AUTHOR_AUDIT_SAMPLES_READY"
        and value.get("selected_item_count") == SELECTED_ITEM_COUNT
        and primary_seconds == PRIMARY_AUDIT_SECONDS
        and reserve_seconds is not None
        and 0 <= reserve_seconds <= MAX_ESCALATION_SECONDS
        and per_item_or_fallback
        and _all_true(
            value,
            (
                "primary_reserve_disjoint",
                "all_selected_items_represented_in_primary",
                "all_selected_items_represented_in_reserve",
                "model_prediction_independent",
                "model_confidence_independent",
                "lexical_content_independent",
                "visual_content_independent",
                "selection_uses_only_official_windows_and_frozen_binding",
                "audit_sample_frozen_before_predictions",
                "author_prediction_blinding_required",
                "reserve_packet_sealed_separately",
            ),
        )
        and _all_false(
            value,
            (
                "prediction_payload_accepted_by_sampler",
                "exact_intervals_exported",
                "item_identifiers_exported",
            ),
        )
        and value.get("maximum_reserve_activation_count") == 1
        and value.get("reserve_activation")
        == "BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK"
        and all(
            _is_sha256(value.get(name))
            for name in (
                "frozen_v1_2_selection_digest",
                "official_windows_manifest_sha256",
                "sampler_policy_sha256",
                "primary_packet_sha256",
                "reserve_packet_sha256",
            )
        )
    )


def _pseudo_valid(value: Mapping[str, Any]) -> bool:
    cpu_workers = value.get("maximum_cpu_workers")
    mps_processes = value.get("maximum_mps_heavy_processes")
    return bool(
        value.get("schema_version") == EXPECTED_SCHEMAS["pseudo"]
        and value.get("status") == "COMPLETE"
        and value.get("candidate_window_count") == FULL_PILOT_WINDOW_COUNT
        and _number(value.get("candidate_speech_minutes")) == FULL_PILOT_SPEECH_MINUTES
        and isinstance(cpu_workers, int)
        and not isinstance(cpu_workers, bool)
        and 1 <= cpu_workers <= MAX_CPU_WORKERS
        and mps_processes in (0, 1)
        and _all_true(
            value,
            (
                "fixed_versioned_instruments",
                "all_instrument_licenses_audited",
                "model_hashes_verified",
                "model_downloads_completed_before_restricted_processing",
                "local_offline_only",
                "network_disabled_during_restricted_inference",
                "inference_subprocess_network_blocked",
                "no_hosted_or_cloud_content_path",
                "quarantine_only",
                "checkpoint_resume_enabled",
                "pseudo_labels_marked_not_ground_truth",
                "author_audit_only_human_labeled_childlens_evidence",
                "simulator_oracle_labels_primary_evaluation_truth",
                "aggregate_safe_receipt_only",
            ),
        )
        and _all_false(
            value,
            (
                "external_api_used",
                "external_upload",
                "telemetry_enabled",
                "restricted_data_egress",
                "instrument_embeddings_entered_learner",
                "instrument_features_entered_learner",
                "instrument_weights_entered_learner",
                "instrument_tokenizers_entered_learner",
                "instrument_vocabularies_entered_learner",
                "instrument_scores_entered_learner",
                "unaudited_pseudo_labels_primary_evaluation_truth",
                "learner_training_executed",
                "corpus_tokenizer_trained",
                "causal_arm_executed",
                "scientific_acquisition_outcome_executed",
            ),
        )
    )


def _workflow_valid(value: Mapping[str, Any]) -> bool:
    estimate = _number(value.get("estimated_author_minutes"))
    prediction_state_valid = bool(
        (value.get("predictions_hidden") is True and value.get("prediction_join_enabled") is False)
        or (
            value.get("human_audit_complete") is True
            and value.get("prediction_join_enabled") is True
        )
    )
    return bool(
        value.get("schema_version") == EXPECTED_SCHEMAS["workflow"]
        and value.get("status") == "READY"
        and value.get("route") == "AUTHOR_AUDIT_A"
        and value.get("audit_item_count") == SELECTED_ITEM_COUNT
        and _number(value.get("audit_speech_minutes")) == PRIMARY_AUDIT_SECONDS / 60
        and _number(value.get("audit_speech_seconds")) == PRIMARY_AUDIT_SECONDS
        and estimate is not None
        and 1 <= estimate <= 8 * 60
        and _all_true(
            value,
            (
                "workflow_ready",
                "loopback_only",
                "owner_private",
                "autosave",
                "autosave_to_quarantine_only",
                "resumable",
                "predictions_hidden_before_author_lock",
                "author_record_lock_immutable",
                "comparison_requires_lock",
                "selected_independent_of_model_outputs",
                "sample_independent_of_model_outputs",
                "uncertain_unusable_routes",
                "uncertain_route_available",
                "unusable_route_available",
                "qualification_instruction_present",
            ),
        )
        and _all_false(
            value,
            (
                "external_hosting",
                "network_exposure",
                "model_predictions_revealed_before_lock",
                "human_evidence_fabricated",
            ),
        )
        and prediction_state_valid
        and value.get("inter_human_reliability_available") is False
        and value.get("primary_evaluation_truth") == "SIMULATOR_ORACLE_ONLY"
        and value.get("unaudited_pseudo_label_primary_evaluation_truth") is False
    )


def _audit_valid_shape(value: Mapping[str, Any]) -> bool:
    if value.get("schema_version") != EXPECTED_SCHEMAS["audit"]:
        return False
    if value.get("audit_complete") is not True:
        return False
    if not _all_true(
        value,
        (
            "qualified_author_confirmed",
            "author_used_raw_audio_video",
            "author_blinded_until_irreversible_lock",
            "author_record_irreversibly_locked",
            "model_predictions_revealed_only_after_lock",
            "sample_selection_digest_match",
            "thresholds_digest_match",
            "cluster_aware_uncertainty_reported",
            "small_sample_limitation_reported",
        ),
    ):
        return False
    if not _all_false(
        value,
        (
            "author_record_changed_after_lock",
            "thresholds_changed_after_author_labels",
            "sample_changed_after_author_labels",
            "human_evidence_fabricated",
            "inter_human_reliability_claimed",
        ),
    ):
        return False
    additional_seconds = _number(value.get("additional_audit_speech_seconds_used"))
    if additional_seconds is None or not 0 <= additional_seconds <= MAX_ESCALATION_SECONDS:
        return False
    reserve = value.get("reserve_activation")
    if reserve is not None:
        if not isinstance(reserve, dict) or not all(
            isinstance(reserve.get(key), bool)
            for key in ("eligible", "activated_once", "pending_author_lock")
        ):
            return False
        if reserve.get("pending_author_lock") is True and reserve.get("activated_once") is not True:
            return False
    results = value.get("threshold_results")
    return bool(
        isinstance(results, dict)
        and set(results) == REQUIRED_THRESHOLD_NAMES
        and all(
            isinstance(result, dict)
            and result.get("status") in {"PASS", "BORDERLINE", "HARD_FAIL", "NOT_ESTIMABLE"}
            and result.get("threshold_frozen") is True
            for result in results.values()
        )
    )


def decide(
    protocol: Mapping[str, Any],
    sampling: Mapping[str, Any],
    pseudo: Mapping[str, Any],
    workflow: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
) -> Decision:
    """Apply the frozen fail-closed decision order to aggregate evidence."""

    prepared = (
        _protocol_valid(protocol)
        and _sampling_valid(sampling)
        and _pseudo_valid(pseudo)
        and _workflow_valid(workflow)
    )
    if not prepared:
        fundamental = bool(
            pseudo.get("fundamental_local_instrument_failure_established") is True
            and pseudo.get("fundamental_failure_evidence_grade")
            == "REPRODUCIBLE_LOCAL_INSTRUMENT_INCOMPATIBILITY"
            and pseudo.get("bounded_correction_exhausted") is True
        )
        if fundamental:
            return Decision(
                STOP,
                "REPRODUCIBLE_LOCAL_INSTRUMENT_FAILURE_AFTER_BOUNDED_CORRECTION",
                "RETAIN_V1_3_STOP_RECORD_AND_DO_NOT_USE_UNAUDITED_CHILDLENS_PSEUDO_LABELS_AS_EVALUATION_TRUTH",
                False,
                False,
                False,
                True,
            )
        return Decision(
            REVISE,
            "PREPARATION_OR_SAFETY_RECEIPT_INCOMPLETE",
            "CORRECT_THE_BOUNDED_LOCAL_PIPELINE_OR_RECEIPT_FAILURE_WITHOUT_CHANGING_THE_FROZEN_SAMPLE_OR_THRESHOLDS",
            False,
            False,
            False,
            False,
        )

    if audit is None or audit.get("audit_complete") is not True:
        return Decision(
            AUTHOR_READY,
            "FULL_LOCAL_PSEUDO_ANNOTATION_AND_BLINDED_AUTHOR_PACKET_READY",
            "QUALIFIED_AUTHOR_COMPLETES_AND_IRREVERSIBLY_LOCKS_AUTHOR_AUDIT_A_THEN_RUNS_MODEL_HUMAN_COMPARISON",
            True,
            False,
            True,
            False,
        )

    if not _audit_valid_shape(audit):
        return Decision(
            REVISE,
            "AUTHOR_AUDIT_RECEIPT_FAILED_LOCK_OR_BLINDING_VALIDATION",
            "REPAIR_ONLY_THE_BOUNDED_AUDIT_RECEIPT_OR_WORKFLOW_FAILURE_WITHOUT_RELABELING_LOCKED_AUTHOR_RECORDS",
            False,
            True,
            False,
            False,
        )

    results = audit["threshold_results"]
    statuses = {name: result["status"] for name, result in results.items()}
    if all(status == "PASS" for status in statuses.values()):
        return Decision(
            GO,
            "ALL_FROZEN_MODEL_HUMAN_AGREEMENT_THRESHOLDS_PASS",
            "DESIGN_THE_NEXT_CHILD_ONLY_PROTOCOL_FREEZE_USING_ONLY_AUDITED_AGGREGATES_AND_SIMULATOR_ORACLE_EVALUATION_LABELS",
            False,
            True,
            False,
            False,
        )

    borderline = any(status == "BORDERLINE" for status in statuses.values())
    escalation_used = _number(audit.get("additional_audit_speech_seconds_used")) or 0.0
    reserve = audit.get("reserve_activation")
    reserve_activated = bool(
        isinstance(reserve, dict) and reserve.get("activated_once") is True
    )
    reserve_pending = bool(
        reserve_activated and reserve.get("pending_author_lock") is True
    )
    escalation_available = bool(
        borderline and escalation_used == 0.0 and not reserve_activated
    )
    if escalation_available:
        return Decision(
            REVISE,
            "PRIMARY_AUTHOR_AUDIT_BORDERLINE_WITH_SINGLE_ESCALATION_AVAILABLE",
            "RUN_THE_PREDECLARED_HASH_DETERMINISTIC_BLINDED_ESCALATION_OF_AT_MOST_15_ADDITIONAL_SPEECH_MINUTES_ONCE",
            True,
            True,
            True,
            False,
        )
    if borderline and reserve_pending:
        return Decision(
            REVISE,
            "ONE_TIME_BORDERLINE_RESERVE_ALREADY_ACTIVATED_AND_PENDING_AUTHOR_LOCK",
            "COMPLETE_THE_ALREADY_ACTIVATED_BLINDED_RESERVE_AUTHOR_AUDIT_LOCK_AND_COMPARISON_WITHOUT_A_SECOND_ACTIVATION",
            True,
            True,
            False,
            False,
        )

    fundamental = bool(
        audit.get("fundamental_calibration_failure_established") is True
        and audit.get("fundamental_failure_evidence_grade")
        == "LOCKED_QUALIFIED_AUTHOR_AUDIT"
        and any(status == "HARD_FAIL" for status in statuses.values())
    )
    if fundamental:
        return Decision(
            STOP,
            "LOCKED_AUTHOR_AUDIT_ESTABLISHES_FUNDAMENTAL_CALIBRATION_FAILURE",
            "RETAIN_V1_3_STOP_RECORD_AND_DO_NOT_USE_UNAUDITED_CHILDLENS_PSEUDO_LABELS_AS_EVALUATION_TRUTH",
            False,
            True,
            False,
            True,
        )

    return Decision(
        REVISE,
        "FROZEN_AGREEMENT_THRESHOLDS_NOT_ALL_PASS",
        "APPLY_ONLY_A_PREDECLARED_BOUNDED_TECHNICAL_CORRECTION_OR_REPORT_THE_MODEL_QUALITY_LIMITATION",
        False,
        True,
        False,
        False,
    )


def _decision_payload(
    decision: Decision,
    inputs: Mapping[str, Path],
    audit_path: Path | None,
    history: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    hashes = {name: _sha256(path) for name, path in inputs.items()}
    if audit_path is not None:
        hashes["audit"] = _sha256(audit_path)
    return {
        "schema_version": "childlens-v1.3-decision-record-v1",
        "synthesizer_version": VERSION,
        "terminal_state": decision.terminal_state,
        "decision_basis_code": decision.basis_code,
        "exact_next_task": decision.exact_next_task,
        "author_action_only_remaining": decision.author_action_only_remaining,
        "author_audit_complete": decision.audit_complete,
        "bounded_escalation_available": decision.escalation_available,
        "fundamental_failure_established": decision.fundamental_failure_established,
        "model_outputs_are_pseudo_labels_not_ground_truth": True,
        "author_audited_subset_only_human_labeled_childlens_evidence": True,
        "inter_human_reliability_available": False,
        "simulator_oracle_labels_primary_later_evaluation_truth": True,
        "hosted_or_cloud_model_inspected_restricted_content": False,
        "restricted_network_egress": False,
        "learner_training_executed": False,
        "corpus_tokenizer_trained": False,
        "causal_arm_executed": False,
        "scientific_acquisition_outcome_executed": False,
        "historical_v1_preserved": True,
        "historical_v1_1_preserved": True,
        "historical_v1_2_preserved": True,
        "historical_artifact_sets": history,
        "input_receipt_sha256": hashes,
    }


def _executive_text(decision: Decision) -> str:
    return (
        "# ChildLens v1.3 model-assisted feasibility decision\n\n"
        f"{decision.terminal_state}\n\n"
        f"Decision basis: `{decision.basis_code}`.\n\n"
        "All machine annotations are local measurement-instrument pseudo-labels, "
        "not ground truth. Only the locked author-audited subset can become "
        "human-labeled ChildLens evidence. Inter-human reliability is unavailable. "
        "Any later causal evaluation must use simulator oracle labels as its primary truth.\n\n"
        f"Exact next task: `{decision.exact_next_task}`.\n"
    )


def _atomic_write(path: Path, payload: bytes, root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError:
        _fail("E_OUTPUT_CONFINEMENT")
    if path.exists() or path.is_symlink():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            _fail("E_OUTPUT_CONFINEMENT")
    descriptor, temporary = tempfile.mkstemp(prefix=".childlens-v1-3-", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def synthesize(root: Path, *, check_historical: bool = True) -> Decision:
    root = root.resolve(strict=True)
    history = _history_status(root)
    if check_historical and not all(entry["preserved"] is True for entry in history.values()):
        _fail("E_HISTORICAL_MUTATION")
    inputs = {name: root / relative for name, relative in INPUT_RELATIVE_PATHS.items()}
    values = {name: _load_object(path) for name, path in inputs.items()}
    for name, value in values.items():
        if value.get("schema_version") != EXPECTED_SCHEMAS[name]:
            _fail("E_INPUT_SCHEMA")
    audit_path = root / OPTIONAL_AUDIT_RELATIVE_PATH
    audit = None
    if audit_path.exists() or audit_path.is_symlink():
        audit = _load_object(audit_path)
        if audit.get("schema_version") != EXPECTED_SCHEMAS["audit"]:
            _fail("E_INPUT_SCHEMA")
    decision = decide(
        values["protocol"], values["sampling"], values["pseudo"], values["workflow"], audit
    )
    payload = _decision_payload(
        decision, inputs, audit_path if audit is not None else None, history
    )
    decision_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    executive_bytes = _executive_text(decision).encode("utf-8")
    _atomic_write(root / OUTPUT_RELATIVE_PATHS["executive"], executive_bytes, root)
    _atomic_write(root / OUTPUT_RELATIVE_PATHS["decision"], decision_bytes, root)
    return decision


def main() -> int:
    if len(sys.argv) != 1:
        print("E_NO_ARGUMENTS", file=sys.stderr)
        return 2
    try:
        decision = synthesize(Path(__file__).resolve().parents[1])
    except SynthesisError as error:
        print(error.code, file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "terminal_state": decision.terminal_state}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
