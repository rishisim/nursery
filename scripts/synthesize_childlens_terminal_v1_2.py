#!/usr/bin/env python3
"""Deterministically synthesize the repository-safe ChildLens v1.2 terminal bundle.

The command-line entry point accepts no arguments.  Scientific evidence is read
only from four repository-safe aggregate receipts and two immutable-history
receipts.  It never discovers or reads a quarantine, media, annotations,
transcripts, identifiers, browser state, or credentials.

All output text is derived from a small allowlist of aggregate fields.  Values
from receipts are never interpolated wholesale.  The five outputs are staged,
fsynced, and replaced as an all-or-rollback transaction, with the decision
record replaced last.
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


VERSION = "childlens-terminal-synthesizer-v1.2.0"
SELECTED_COUNT = 15
RAW_CAP_BYTES = 20 * 1024**3
MIN_CANDIDATE_SPEECH_MINUTES = 30.0
MIN_CANDIDATE_UTTERANCES = 300
MIN_DOUBLE_CODE_FRACTION = 0.20
ADMISSION_CEILING_BYTES = 18 * 1024**3
NAMESPACE_CAP_BYTES = 73 * 1024**3
POST_PEAK_FREE_FLOOR_BYTES = 50 * 1024**3
NONRAW_RESERVE_BYTES = 4 * 1024**3
FROZEN_DISPLAY_TOTAL_BYTES = 14_071_800_000
FROZEN_ROUNDING_UPPER_BYTES = 14_572_800_000
FROZEN_TRANSFER_OVERHEAD_BYTES = 397_386_240
FROZEN_CONSERVATIVE_ADMISSION_BYTES = 14_970_186_240
FROZEN_SELECTION_DIGEST = (
    "61526ea6cebc256314b0b9e574fb1a5986fcd83a534979e93134c4163c55253f"
)
PUBLIC_SNAPSHOT_COMMIT = "4856662653b2fa183e53268d088cffba02a33443"
RETENTION_DEADLINE = "2027-07-31"
REQUIRED_STRUCTURAL_METRICS = (
    "restricted_manifest_file_integrity_match",
    "probe_success",
    "video_stream_present",
    "audio_stream_present",
    "duration_consistency_evidenced",
    "full_decode_success",
    "corruption_absence_evidenced",
    "speech_presence_window_expectation_satisfied",
    "annotation_linkage_verified",
)

HUMAN_READY = "CHILDLENS_HUMAN_VALIDATION_READY"
GO = "CHILDLENS_FEASIBILITY_GO"
REVISE = "CHILDLENS_FEASIBILITY_REVISE"
STOP = "CHILDLENS_FEASIBILITY_STOP"
TERMINAL_STATES = frozenset({HUMAN_READY, GO, REVISE, STOP})
TERMINAL_RE = re.compile(
    r"CHILDLENS_(?:HUMAN_VALIDATION_READY|FEASIBILITY_(?:GO|REVISE|STOP))"
)

INCIDENT_SUMMARY = (
    "RESTRICTED_QUARANTINE_PATH_APPEARED_IN_LOCAL_TOOL_LOG_"
    "REDACTED_FROM_REPOSITORY_AND_USER_FACING_ARTIFACTS_NO_CORPUS_PAYLOAD"
)
EXPECTED_V1_COUNT = 23
EXPECTED_V1_DIGEST = "35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf"
EXPECTED_V1_1_COUNT = 40
EXPECTED_V1_1_DIGEST = "3acd969804d71979ba8071e2446e8ee1ceb3194fc90dcfda802a99e296b7bf48"

INPUT_RELATIVE_PATHS = {
    "acquisition": "output/childlens_feasibility_v1_2/acquisition_receipt.json",
    "diagnostics": "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json",
    "workflow": "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json",
    "immutability": "output/childlens_feasibility_v1_2/immutability_receipt.json",
    "baseline": "output/childlens_feasibility_v1_2/immutability_baseline_receipt.json",
}
EXPECTED_SCHEMAS = {
    "acquisition": "childlens-v1.2-acquisition-receipt-v1",
    "diagnostics": "childlens-v1.2-automated-diagnostics-receipt-v1",
    "workflow": "childlens-v1.2-human-workflow-receipt-v1",
    "immutability": "childlens-v1.2-immutability-receipt-v1",
    "baseline": "childlens-immutable-baseline-v1.2.0",
}
OUTPUT_RELATIVE_PATHS = {
    "executive": "docs/childlens_feasibility_v1_2/executive_decision_report.md",
    "acquisition": "docs/childlens_feasibility_v1_2/native_selective_acquisition_report.md",
    "measurement": "docs/childlens_feasibility_v1_2/local_measurement_pilot_report.md",
    "commands": "docs/childlens_feasibility_v1_2/commands_tests_and_limitations_v1_2.md",
    "decision": "output/childlens_feasibility_v1_2/decision_record.json",
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
        "transcript_text",
        "utterance_text",
        "raw_transcript",
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
    """A fixed-code, payload-suppressing synthesis failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Decision:
    terminal_state: str
    semantic_label: str
    basis_code: str
    bounded_problem_remaining: bool
    fundamental_corpus_failure_established: bool
    only_remaining_human_evidence: bool
    all_essential_frozen_gates_pass: bool
    exact_next_task: str


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
            or info.st_size > 4 * 1024 * 1024
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _check_input_confinement(root: Path, paths: Mapping[str, Path]) -> None:
    for path in paths.values():
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            _fail("E_INPUT_FILE")
        if not _is_relative_to(resolved, root):
            _fail("E_INPUT_CONFINEMENT")


def _check_output_confinement(root: Path, paths: list[Path]) -> None:
    for path in paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            parent = path.parent.resolve(strict=True)
            metadata = path.lstat() if path.exists() or path.is_symlink() else None
        except OSError:
            _fail("E_OUTPUT_CONFINEMENT")
        if not _is_relative_to(parent, root) or (
            metadata is not None
            and (stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode))
        ):
            _fail("E_OUTPUT_CONFINEMENT")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        converted = float(value)
        if math.isfinite(converted):
            return converted
    return None


def _is_sha256(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value))


def _all_false(value: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(value.get(key) is False for key in keys)


def _all_true(value: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(value.get(key) is True for key in keys)


def _validate_immutability(
    immutability: Mapping[str, Any], baseline: Mapping[str, Any]
) -> None:
    expected = {
        "v1_artifact_count": EXPECTED_V1_COUNT,
        "v1_set_digest": EXPECTED_V1_DIGEST,
        "v1_1_artifact_count": EXPECTED_V1_1_COUNT,
        "v1_1_set_digest": EXPECTED_V1_1_DIGEST,
        "v1_preserved": True,
        "v1_1_preserved": True,
        "historical_decisions_overwritten": False,
    }
    if any(immutability.get(key) != expected_value for key, expected_value in expected.items()):
        _fail("E_HISTORY_IMMUTABILITY")
    if (
        baseline.get("schema_version") != "childlens-immutable-baseline-v1.2.0"
        or baseline.get("validation_status") != "PASS"
        or not isinstance(baseline.get("v1"), dict)
        or not isinstance(baseline.get("v1_1"), dict)
        or baseline["v1"].get("artifact_count") != EXPECTED_V1_COUNT
        or baseline["v1"].get("artifact_set_digest") != EXPECTED_V1_DIGEST
        or baseline["v1_1"].get("artifact_count") != EXPECTED_V1_1_COUNT
        or baseline["v1_1"].get("artifact_set_digest") != EXPECTED_V1_1_DIGEST
        or baseline["v1"].get("historical_decision_preserved") is not True
        or baseline["v1_1"].get("historical_decision_preserved") is not True
    ):
        _fail("E_HISTORY_BASELINE")


def _acquisition_pass(value: Mapping[str, Any]) -> bool:
    raw_bytes = value.get("raw_media_bytes_acquired")
    admission_bound = value.get("pre_transfer_upper_bound_bytes")
    margin = value.get("upper_bound_margin_to_raw_cap_bytes")
    method = value.get("storage_admission_method")
    method_pass = method == "CONSERVATIVE_DISPLAY_BOUND_WITH_HARD_COUNTER"
    if method == "CONSERVATIVE_DISPLAY_BOUND_WITH_HARD_COUNTER":
        method_pass = bool(
            admission_bound == FROZEN_CONSERVATIVE_ADMISSION_BYTES
            and margin == RAW_CAP_BYTES - FROZEN_CONSERVATIVE_ADMISSION_BYTES
            and _is_int(value.get("hard_abort_before_bytes"))
            and value["hard_abort_before_bytes"] == RAW_CAP_BYTES - 1
        )
    pre_namespace = value.get("pre_transfer_namespace_bytes")
    pre_free = value.get("pre_transfer_free_bytes")
    projected_namespace = value.get("projected_namespace_peak_bytes")
    projected_free = value.get("projected_post_peak_free_bytes")
    post_free = value.get("post_transfer_free_bytes")
    after_namespace = value.get("namespace_bytes_after_acquisition")
    storage_pass = bool(
        _is_int(pre_namespace)
        and pre_namespace >= 0
        and _is_int(pre_free)
        and pre_free >= 0
        and projected_namespace
        == pre_namespace + FROZEN_CONSERVATIVE_ADMISSION_BYTES + NONRAW_RESERVE_BYTES
        and projected_namespace <= NAMESPACE_CAP_BYTES
        and projected_free
        == pre_free - FROZEN_CONSERVATIVE_ADMISSION_BYTES - NONRAW_RESERVE_BYTES
        and projected_free >= POST_PEAK_FREE_FLOOR_BYTES
        and _is_int(post_free)
        and post_free >= POST_PEAK_FREE_FLOOR_BYTES
        and _is_int(after_namespace)
        and 0 <= after_namespace <= NAMESPACE_CAP_BYTES
    )
    return bool(
        value.get("selected_media_count") == SELECTED_COUNT
        and value.get("acquired_media_count") == SELECTED_COUNT
        and value.get("verified_against_frozen_selection_count") == SELECTED_COUNT
        and value.get("acquisition_complete") is True
        and _is_int(raw_bytes)
        and 0 < raw_bytes < RAW_CAP_BYTES
        and value.get("raw_cap_bytes") == RAW_CAP_BYTES
        and method_pass
        and value.get("displayed_parsed_total_bytes") == FROZEN_DISPLAY_TOTAL_BYTES
        and value.get("rounding_upper_bound_bytes") == FROZEN_ROUNDING_UPPER_BYTES
        and value.get("transfer_overhead_bytes") == FROZEN_TRANSFER_OVERHEAD_BYTES
        and value.get("admission_ceiling_bytes") == ADMISSION_CEILING_BYTES
        and value.get("predeclared_nonraw_reserve_bytes") == NONRAW_RESERVE_BYTES
        and value.get("namespace_cap_bytes") == NAMESPACE_CAP_BYTES
        and value.get("projected_post_peak_free_floor_bytes")
        == POST_PEAK_FREE_FLOOR_BYTES
        and storage_pass
        and value.get("frozen_selection_digest") == FROZEN_SELECTION_DIGEST
        and value.get("immutable_public_snapshot_commit") == PUBLIC_SNAPSHOT_COMMIT
        and value.get("pilot_source")
        == "ACCESSIBLE_LIVE_VIEW_BOUND_TO_RESTRICTED_MANIFEST_DIGEST"
        and value.get("live_to_public_snapshot_byte_equivalence") == "NOT_PROVEN"
        and value.get("live_snapshot_limitation_disclosed") is True
        and value.get("release_binding_verified") is True
        and value.get("restricted_manifest_digest_verified") is True
        and _is_sha256(value.get("restricted_manifest_sha256"))
        and _is_sha256(value.get("restricted_native_transfer_receipt_sha256"))
        and value.get("retention_deadline") == RETENTION_DEADLINE
        and value.get("retention_control_recorded") is True
        and value.get("sequential_or_tightly_bounded_transfer") is True
        and value.get("cumulative_stream_counter_enforced") is True
        and _all_false(
            value,
            (
                "full_archive_downloaded",
                "selection_changed",
                "selection_content_inspected_before_transfer",
                "duplicate_full_resolution_copies_created",
                "source_filenames_exported",
                "restricted_identifiers_exported",
                "exact_timestamps_exported",
                "restricted_manifest_exported",
                "external_or_cloud_transfer",
            ),
        )
    )


def _metric_all_pass(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("status") == "ALL_PASS"
        and value.get("pass_count") == SELECTED_COUNT
        and value.get("fail_count") == 0
        and value.get("cell_suppressed") is False
    )


def _diagnostics_pass(
    value: Mapping[str, Any], acquisition: Mapping[str, Any]
) -> bool:
    metrics = value.get("structural_metrics")
    return bool(
        value.get("source_acquired_media_count") == SELECTED_COUNT
        and value.get("media_preparation_attempted_count") == SELECTED_COUNT
        and value.get("container_probe_attempted_count") == SELECTED_COUNT
        and value.get("container_probe_success_count") == SELECTED_COUNT
        and value.get("decode_attempted_count") == SELECTED_COUNT
        and value.get("decode_success_count") == SELECTED_COUNT
        and value.get("audio_stream_media_count") == SELECTED_COUNT
        and value.get("speech_presence_window_media_count") == SELECTED_COUNT
        and _is_int(value.get("cell_suppression_k"))
        and value.get("cell_suppression_k") >= 5
        and isinstance(metrics, dict)
        and all(_metric_all_pass(metrics.get(name)) for name in REQUIRED_STRUCTURAL_METRICS)
        and value.get("small_nonzero_failure_cells_exported") is False
        and value.get("measurement_source") == "LOCAL_RESTRICTED_MEDIA_AUDIT_V1_2"
        and value.get("measurement_executed_on_acquired_bytes") is True
        and value.get("speech_window_actual_source")
        == "OFFICIAL_ANNOTATION_WINDOWS_LINKED_TO_ACQUIRED_MEDIA"
        and value.get("annotation_linkage_actual_source")
        == "RESTRICTED_CANONICAL_MANIFEST"
        and value.get("pilot_selection_sha256") == FROZEN_SELECTION_DIGEST
        and _is_sha256(value.get("source_native_transfer_receipt_sha256"))
        and value.get("source_native_transfer_receipt_sha256")
        == acquisition.get("restricted_native_transfer_receipt_sha256")
        and _is_sha256(value.get("restricted_measurement_receipt_sha256"))
        and _number(value.get("official_speech_window_union_minutes")) is not None
        and value.get("official_speech_window_union_minutes") >= 0
        and value.get("automated_preparation_complete") is True
        and value.get("diagnostics_passed_for_human_workflow") is True
        and value.get("aggregate_only") is True
        and value.get("local_offline_only") is True
        and value.get("automated_outputs_not_gold") is True
        and _all_false(
            value,
            (
                "lexical_content_exported",
                "transcript_text_exported",
                "speaker_labels_exported",
                "frames_exported",
                "external_api_used",
                "learner_data_created",
                "instrument_features_passed_to_learner",
                "instrument_embeddings_passed_to_learner",
                "instrument_tokenizer_or_vocabulary_passed_to_learner",
            ),
        )
    )


def _candidate_bound_honest(
    diagnostics: Mapping[str, Any], workflow: Mapping[str, Any]
) -> bool:
    diagnostic_minutes = _number(diagnostics.get("official_speech_window_union_minutes"))
    workflow_minutes = _number(workflow.get("populated_candidate_speech_window_minutes"))
    minute_route = bool(
        diagnostic_minutes is not None
        and workflow_minutes is not None
        and diagnostic_minutes >= MIN_CANDIDATE_SPEECH_MINUTES
        and workflow_minutes >= MIN_CANDIDATE_SPEECH_MINUTES
        and abs(diagnostic_minutes - workflow_minutes) <= 0.01
        and workflow.get("candidate_unit_type")
        == "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE"
        and workflow.get("populated_candidate_utterance_count") == 0
    )
    utterance_route = bool(
        _is_int(workflow.get("populated_candidate_utterance_count"))
        and workflow["populated_candidate_utterance_count"] >= MIN_CANDIDATE_UTTERANCES
        and workflow.get("candidate_unit_type") == "HUMAN_CORRECTED_UTTERANCE"
        and workflow.get("genuine_human_validation_complete") is True
    )
    return minute_route or utterance_route


def _workflow_structural_pass(
    value: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> bool:
    first = _number(value.get("estimated_first_authorized_human_minutes"))
    second = _number(value.get("estimated_second_independent_human_minutes"))
    adjudication = _number(value.get("estimated_adjudication_minutes"))
    total = _number(value.get("estimated_total_human_minutes"))
    fraction = _number(value.get("double_code_fraction_target"))
    unit = value.get("candidate_unit_type")
    candidate_utterances = value.get("populated_candidate_utterance_count")
    candidate_windows = value.get("populated_candidate_speech_window_count")
    candidate_minutes = _number(value.get("populated_candidate_speech_window_minutes"))
    referential_ready = bool(
        (
            unit == "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE"
            and value.get("referential_assignment_eligible_count") == 0
            and value.get("referential_double_code_assigned_count") == 0
            and value.get("referential_double_code_assigned_fraction") is None
            and value.get("referential_double_code_sample_frozen") is False
            and value.get("referential_assignment_rule_frozen") is True
            and value.get("referential_inventory_pending_human_segmentation") is True
            and value.get("double_code_fraction_target") == 0.20
        )
        or _actual_double_code_ready(value)
    )
    return bool(
        value.get("workflow_populated") is True
        and value.get("ready_for_authorized_human") is True
        and value.get("selected_media_source_count") == SELECTED_COUNT
        and _is_int(candidate_utterances)
        and candidate_utterances >= 0
        and _is_int(candidate_windows)
        and candidate_windows >= 0
        and candidate_minutes is not None
        and candidate_minutes >= 0
        and all(item is not None and item > 0 for item in (first, second, adjudication))
        and total is not None
        and abs(total - first - second - adjudication) <= 1e-6
        and fraction is not None
        and fraction >= MIN_DOUBLE_CODE_FRACTION
        and referential_ready
        and value.get("language_establishment_required_before_transcription") is True
        and value.get("language_matched_humans_required") is True
        and value.get("human_qualification_screening_enabled") is True
        and value.get("language_routes_blinded_and_resumable") is True
        and value.get("workflow_source")
        == "LOCAL_RESTRICTED_HUMAN_VALIDATION_PACKET_V1_2"
        and value.get("packet_populated_from_actual_acquired_media") is True
        and value.get("pilot_selection_sha256") == FROZEN_SELECTION_DIGEST
        and _is_sha256(value.get("source_native_transfer_receipt_sha256"))
        and value.get("source_native_transfer_receipt_sha256")
        == acquisition.get("restricted_native_transfer_receipt_sha256")
        and _is_sha256(value.get("source_measurement_receipt_sha256"))
        and value.get("source_measurement_receipt_sha256")
        == diagnostics.get("restricted_measurement_receipt_sha256")
        and _is_sha256(value.get("restricted_workflow_receipt_sha256"))
        and _all_true(
            value,
            (
                "local_only",
                "autosave_to_restricted_quarantine_only",
                "resumable_batches",
                "input_validation_enabled",
                "instrument_condition_blinded",
            ),
        )
        and _all_false(
            value,
            (
                "identifiers_displayed",
                "source_filenames_displayed",
                "exact_timestamps_exported",
                "transcript_text_exported",
                "external_service_used",
                "human_evidence_fabricated",
            ),
        )
    )


def _actual_double_code_ready(value: Mapping[str, Any]) -> bool:
    eligible = value.get("referential_assignment_eligible_count")
    doubled = value.get("referential_double_code_assigned_count")
    fraction = _number(value.get("referential_double_code_assigned_fraction"))
    return bool(
        _is_int(eligible)
        and eligible >= 5
        and _is_int(doubled)
        and 0 <= doubled <= eligible
        and fraction is not None
        and abs(fraction - doubled / eligible) <= 1e-6
        and fraction >= MIN_DOUBLE_CODE_FRACTION
        and value.get("referential_double_code_sample_frozen") is True
    )


def _human_go_evidence(value: Mapping[str, Any]) -> bool:
    return _all_true(
        value,
        (
            "genuine_human_validation_complete",
            "independent_reliability_requirement_complete",
            "frozen_reliability_thresholds_pass",
            "actual_language_established_by_qualified_humans",
            "language_matched_coders_verified",
            "all_frozen_feasibility_gates_pass",
        ),
    ) and value.get("human_evidence_fabricated") is False and _actual_double_code_ready(value)


def _fundamental_failure(value: Mapping[str, Any]) -> bool:
    allowed_basis = {
        "COMPLETE_CORPUS_LEVEL_NO_USABLE_AUDIO",
        "GENUINE_HUMAN_VALIDATED_NO_NON_CHILD_INPUT",
        "GENUINE_HUMAN_VALIDATED_NO_ANNOTATABLE_VISIBLE_REFERENTS",
        "GENUINE_HUMAN_VALIDATED_NO_LEAKAGE_RESISTANT_HELDOUT_EVALUATION",
    }
    basis = value.get("fundamental_failure_basis_code")
    level = value.get("fundamental_failure_evidence_level")
    human_basis = isinstance(basis, str) and basis.startswith("GENUINE_HUMAN_VALIDATED_")
    qualified_human_failure = bool(
        _all_true(
            value,
            (
                "genuine_human_validation_complete",
                "independent_reliability_requirement_complete",
                "actual_language_established_by_qualified_humans",
                "language_matched_coders_verified",
                "frozen_reliability_thresholds_pass",
            ),
        )
        and value.get("human_evidence_fabricated") is False
        and value.get("all_frozen_feasibility_gates_pass") is False
        and _actual_double_code_ready(value)
    )
    return bool(
        value.get("fundamental_corpus_failure_established") is True
        and basis in allowed_basis
        and (
            level == "COMPLETE_CORPUS_LEVEL_STRUCTURAL"
            or (
                level == "GENUINE_INDEPENDENT_HUMAN_VALIDATION"
                and human_basis
                and qualified_human_failure
            )
        )
    )


def _choose_decision(
    acquisition: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> Decision:
    acquisition_ok = _acquisition_pass(acquisition)
    diagnostics_ok = _diagnostics_pass(diagnostics, acquisition)
    workflow_ok = _workflow_structural_pass(workflow, acquisition, diagnostics)
    candidate_ok = _candidate_bound_honest(diagnostics, workflow)
    human_go = _human_go_evidence(workflow)
    fundamental = _fundamental_failure(workflow) or _fundamental_failure(diagnostics)

    if fundamental:
        if human_go:
            _fail("E_CONTRADICTORY_TERMINAL_EVIDENCE")
        return Decision(
            STOP,
            "fundamental corpus failure established",
            str(
                workflow.get("fundamental_failure_basis_code")
                or diagnostics.get("fundamental_failure_basis_code")
            ),
            False,
            True,
            False,
            False,
            "WAIT_FOR_A_SEPARATELY_INITIALIZED_STUDY_WITHOUT_MIXING_CHILD_CORPORA",
        )

    if human_go:
        if not (acquisition_ok and diagnostics_ok and workflow_ok and candidate_ok):
            _fail("E_GO_EVIDENCE_CONTRADICTION")
        return Decision(
            GO,
            "all frozen feasibility gates passed",
            "GENUINE_INDEPENDENT_HUMAN_EVIDENCE_AND_ALL_FROZEN_GATES_PASS",
            False,
            False,
            False,
            True,
            "DESIGN_THE_NEXT_SCIENTIFIC_PROTOCOL_FREEZE_WITHOUT_RUNNING_A_LEARNER",
        )

    if acquisition_ok and diagnostics_ok and workflow_ok and candidate_ok:
        if (
            workflow.get("genuine_human_validation_complete") is not False
            or workflow.get("independent_reliability_requirement_complete") is not False
        ):
            _fail("E_HUMAN_EVIDENCE_STATE")
        return Decision(
            HUMAN_READY,
            "blinded local human validation is ready",
            "COMPLETE_AUTOMATED_PREPARATION_AND_HONEST_FROZEN_CANDIDATE_BOUND",
            False,
            False,
            True,
            False,
            "AUTHORIZED_LANGUAGE_MATCHED_HUMANS_COMPLETE_TWO_INDEPENDENT_PASSES_AND_ADJUDICATION_THEN_RERUN_SYNTHESIS",
        )

    if not acquisition_ok:
        basis = "BOUNDED_ACQUISITION_OR_VERIFICATION_CORRECTION_REQUIRED"
        next_task = "CORRECT_THE_BOUNDED_NATIVE_ACQUISITION_OR_VERIFICATION_FAILURE_AND_RERUN_LOCAL_PREPARATION"
    elif not diagnostics_ok:
        basis = "BOUNDED_LOCAL_DIAGNOSTIC_CORRECTION_REQUIRED"
        next_task = "CORRECT_THE_BOUNDED_LOCAL_DECODE_OR_LINKAGE_FAILURE_WITHOUT_RESELECTION_AND_RERUN_DIAGNOSTICS"
    elif not workflow_ok:
        basis = "BOUNDED_HUMAN_WORKFLOW_POPULATION_CORRECTION_REQUIRED"
        next_task = "CORRECT_THE_BOUNDED_LOCAL_WORKFLOW_POPULATION_OR_PRIVACY_CONTROL_AND_RERUN_SYNTHESIS"
    else:
        basis = "FROZEN_CANDIDATE_MINIMUM_NOT_YET_HONESTLY_SUPPORTED"
        next_task = "RESOLVE_THE_BOUNDED_CANDIDATE_COVERAGE_SHORTFALL_WITHIN_THE_FROZEN_PROTOCOL_WITHOUT_CALLING_WINDOWS_UTTERANCES"
    return Decision(
        REVISE,
        "a bounded correction remains",
        basis,
        True,
        False,
        False,
        False,
        next_task,
    )


def _count_text(value: Any) -> str:
    if value == 0 or (_is_int(value) and value >= 5):
        return str(value)
    return "cell-suppressed"


def _minutes_text(value: Any) -> str:
    number = _number(value)
    return "unavailable" if number is None else f"{number:.2f}"


def _reports(
    decision: Decision,
    acquisition: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    workflow: Mapping[str, Any],
) -> dict[str, str]:
    executive = f"""# ChildLens v1.2 executive decision

Decision: **{decision.semantic_label}.** The machine-readable terminal value is recorded exactly once in the decision record, not repeated here.

The decision is based only on repository-safe aggregate receipts. The frozen selection, scientific thresholds, and controlling v1.1 permissions were not reopened. Historical v1 and v1.1 decisions remain immutable. No learner, tokenizer, checkpoint, causal arm, or scientific acquisition outcome was run.

Evidence basis: `{decision.basis_code}`. Acquisition, local diagnostics, workflow readiness, and human evidence were evaluated independently; automated windows, transcripts, roles, and referential proposals were not treated as gold.

Privacy disposition: raw media, decoded audio, manifests, identifiers, exact times, transcript text, frames, and row-level hashes remain outside the repository. The required prior local-tool-log path incident is disclosed categorically in the decision record; no corpus payload was exposed and no restricted data egress occurred.

Exact next task: `{decision.exact_next_task}`.
"""

    acquisition_report = f"""# ChildLens v1.2 native selective acquisition report

The frozen selected count was {SELECTED_COUNT}. The repository-safe receipt reports {_count_text(acquisition.get('acquired_media_count'))} acquired objects and {_count_text(acquisition.get('verified_against_frozen_selection_count'))} objects verified against the frozen selection. Acquisition completion is `{str(acquisition.get('acquisition_complete') is True).lower()}`.

Raw acquisition stayed below the frozen 20-GiB cap. Admission method: `{acquisition.get('storage_admission_method', 'UNAVAILABLE') if acquisition.get('storage_admission_method') in {'NATIVE_EXACT_BYTES', 'CONSERVATIVE_DISPLAY_BOUND_WITH_HARD_COUNTER'} else 'UNAVAILABLE'}`. Transfer was sequential or tightly bounded: `{str(acquisition.get('sequential_or_tightly_bounded_transfer') is True).lower()}`.

No full archive, reselection, pre-transfer content inspection, duplicate full-resolution copy, external transfer, source name, restricted identifier, exact time, or restricted manifest was exported. Exact object hashes and linkage remain restricted; this report contains aggregate counts only.
"""

    measurement_report = f"""# ChildLens v1.2 local measurement pilot report

Local preparation attempted {_count_text(diagnostics.get('media_preparation_attempted_count'))} frozen media objects. Container probes succeeded for {_count_text(diagnostics.get('container_probe_success_count'))}; full decode succeeded for {_count_text(diagnostics.get('decode_success_count'))}; usable audio streams were present for {_count_text(diagnostics.get('audio_stream_media_count'))}. Duration consistency, corruption, and annotation linkage were evaluated locally.

The prepared candidate-duration bound is {_minutes_text(diagnostics.get('official_speech_window_union_minutes'))} unioned minutes from official speech-presence windows. Those windows are event intervals, not utterances. The automated inferred-utterance count is zero. Language, transcript text, timing correction, speaker role, referential status, reliability, and adjudication remain genuine qualified-human judgments unless the workflow receipt explicitly records their completed independent disposition.

All processing represented here was local and aggregate-only. No external API was used. Instrument outputs, features, embeddings, weights, tokenizers, vocabularies, and scores were not passed to learner data. No learner data was created.
"""

    commands_report = f"""# ChildLens v1.2 commands, tests, and unresolved limitations

## Deterministic commands

```bash
python3 scripts/synthesize_childlens_terminal_v1_2.py
python3 scripts/validate_childlens_feasibility_v1_2.py
pytest -q tests/test_synthesize_childlens_terminal_v1_2.py tests/test_validate_childlens_feasibility_v1_2.py
```

The synthesizer validated fixed repository-safe receipt schemas, immutable v1/v1.1 baselines, scientific boundary flags, the acquisition cap, 15-item cross-checks, the instrument firewall, honest candidate-unit semantics, and terminal prerequisites before writing. Its tests are synthetic and never read restricted data. Output replacement is staged and all-or-rollback, with the decision record installed last.

## Provenance and privacy boundary

The evidence ancestry is the accessible ChildLens release plus simulator-defined counterfactual modalities only. No AEA or BabyView empirical data, aggregates, priors, vocabulary, tokenizer, checkpoint, weights, or results entered the artifacts. Public methodological references are not empirical ancestry. Raw ChildLens payloads and restricted derivatives remain in the owner-only local quarantine.

The category-only incident receipt is `{INCIDENT_SUMMARY}`. The literal quarantine path is intentionally absent. No corpus payload was exposed in tool output, and no restricted payload is present in the repository.

## Limitations and exact next task

Automated structural success cannot establish language, transcription accuracy, speaker role, lexical recurrence, visible reference, ambiguity, lag, held-out exposure, or reliability. Official speech-presence windows are not utterances. Genuine independent human work is never inferred from model or Codex agreement.

Exact next task: `{decision.exact_next_task}`.
"""
    return {
        "executive": executive,
        "acquisition": acquisition_report,
        "measurement": measurement_report,
        "commands": commands_report,
    }


def _decision_record(
    decision: Decision,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "childlens-v1.2-decision-record-v1",
        "synthesizer_version": VERSION,
        "terminal_state": decision.terminal_state,
        "decision_basis_code": decision.basis_code,
        "exact_next_task": decision.exact_next_task,
        "input_receipt_sha256": dict(sorted(input_hashes.items())),
        "historical_v1_preserved": True,
        "historical_v1_1_preserved": True,
        "permission_reopened": False,
        "selection_reopened": False,
        "scientific_thresholds_changed": False,
        "learner_training_executed": False,
        "scientific_outcome_executed": False,
        "causal_arm_executed": False,
        "tokenizer_trained": False,
        "checkpoint_created": False,
        "aea_empirical_ancestry": False,
        "babyview_empirical_ancestry": False,
        "restricted_data_egress": False,
        "corpus_payload_exposed_in_tool_output": False,
        "restricted_payload_in_repository": False,
        "external_instrument_artifacts_entered_learner": False,
        "local_tool_log_quarantine_path_redaction_incident_disclosed": True,
        "local_tool_log_quarantine_path_redaction_incident_summary": INCIDENT_SUMMARY,
        "only_remaining_evidence_is_authorized_human_validation": decision.only_remaining_human_evidence,
        "all_essential_frozen_gates_pass": decision.all_essential_frozen_gates_pass,
        "bounded_problem_remaining": decision.bounded_problem_remaining,
        "fundamental_corpus_failure_established": decision.fundamental_corpus_failure_established,
    }


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _stage(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".v1_2-terminal-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _restore(path: Path, payload: bytes | None) -> None:
    if payload is None:
        path.unlink(missing_ok=True)
        return
    temporary = _stage(path, payload)
    _replace(temporary, path)


def _atomic_bundle(payloads: Mapping[Path, bytes], decision_path: Path) -> None:
    ordered = sorted((path for path in payloads if path != decision_path), key=str)
    ordered.append(decision_path)
    staged: dict[Path, Path] = {}
    originals: dict[Path, bytes | None] = {}
    replaced: list[Path] = []
    try:
        for path in ordered:
            originals[path] = path.read_bytes() if path.is_file() else None
            staged[path] = _stage(path, payloads[path])
        for path in ordered:
            _replace(staged[path], path)
            staged.pop(path, None)
            replaced.append(path)
        for directory in sorted({path.parent for path in ordered}, key=str):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        rollback_failed = False
        for path in reversed(replaced):
            try:
                _restore(path, originals[path])
            except Exception:
                rollback_failed = True
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        if rollback_failed:
            _fail("E_ATOMIC_ROLLBACK")
        _fail("E_ATOMIC_WRITE")


def synthesize(root: Path) -> Decision:
    """Synthesize a terminal bundle under ``root``; exposed for synthetic tests."""

    try:
        resolved = root.resolve(strict=True)
    except OSError:
        _fail("E_REPOSITORY_ROOT")
    paths = {name: resolved / relative for name, relative in INPUT_RELATIVE_PATHS.items()}
    _check_input_confinement(resolved, paths)
    receipts = {name: _load_object(path) for name, path in paths.items()}
    if any(
        receipts[name].get("schema_version") != expected
        for name, expected in EXPECTED_SCHEMAS.items()
    ):
        _fail("E_INPUT_SCHEMA_VERSION")
    _validate_immutability(receipts["immutability"], receipts["baseline"])

    decision = _choose_decision(
        receipts["acquisition"], receipts["diagnostics"], receipts["workflow"]
    )
    reports = _reports(
        decision, receipts["acquisition"], receipts["diagnostics"], receipts["workflow"]
    )
    input_hashes = {name: _sha256(path) for name, path in paths.items()}
    record = _decision_record(decision, input_hashes)

    payloads: dict[Path, bytes] = {
        resolved / OUTPUT_RELATIVE_PATHS[name]: text.encode("utf-8")
        for name, text in reports.items()
    }
    decision_path = resolved / OUTPUT_RELATIVE_PATHS["decision"]
    payloads[decision_path] = _json_bytes(record)

    _check_output_confinement(resolved, list(payloads))

    combined = b"\n".join(payloads.values()).decode("utf-8")
    literals = TERMINAL_RE.findall(combined)
    if literals != [decision.terminal_state]:
        _fail("E_TERMINAL_LITERAL_CARDINALITY")
    _atomic_bundle(payloads, decision_path)
    return decision


def main() -> int:
    if len(sys.argv) != 1:
        print("FAIL: E_NO_ARGUMENTS")
        return 2
    root = Path(__file__).resolve().parents[1]
    try:
        synthesize(root)
    except SynthesisError as exc:
        print(f"FAIL: {exc.code}")
        return 1
    print("PASS: CHILDLENS_V1_2_TERMINAL_BUNDLE_SYNTHESIZED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
