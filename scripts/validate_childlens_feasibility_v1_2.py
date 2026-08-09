#!/usr/bin/env python3
"""Fail-closed repository validator for ChildLens feasibility v1.2.

This validator reads only versioned repository reports and receipts. It must
never inspect the restricted quarantine, Keeper, browser state, email, or media.
Diagnostics identify the artifact and problem class but suppress matched text.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DOCS_REQUIRED = (
    "commands_tests_and_limitations_v1_2.md",
    "executive_decision_report.md",
    "human_validation_workflow_handoff.md",
    "local_measurement_pilot_report.md",
    "native_selective_acquisition_report.md",
    "privacy_validation_handoff.md",
)
OUTPUT_REQUIRED = (
    "acquisition_receipt.json",
    "automated_diagnostics_receipt.json",
    "decision_record.json",
    "human_validation_workflow_receipt.json",
    "immutability_receipt.json",
)

TERMINAL_STATES = frozenset(
    {
        "CHILDLENS_HUMAN_VALIDATION_READY",
        "CHILDLENS_FEASIBILITY_GO",
        "CHILDLENS_FEASIBILITY_REVISE",
        "CHILDLENS_FEASIBILITY_STOP",
    }
)
TERMINAL_RE = re.compile(
    r"CHILDLENS_(?:HUMAN_VALIDATION_READY|FEASIBILITY_(?:GO|REVISE|STOP))"
)

V1_NAMESPACES = (
    "docs/childlens_feasibility_v1",
    "output/childlens_feasibility_v1",
    "scripts/validate_childlens_feasibility_v1.py",
    "tests/test_childlens_feasibility_v1.py",
)
V1_1_DIRECTORIES = (
    "docs/childlens_feasibility_v1_1",
    "output/childlens_feasibility_v1_1",
)
V1_1_CODE_GLOBS = (
    ("scripts", "*childlens*v1_1*"),
    ("tests", "*childlens*v1_1*"),
)
EXPECTED_V1_COUNT = 23
EXPECTED_V1_DIGEST = "35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf"
EXPECTED_V1_1_COUNT = 40
EXPECTED_V1_1_DIGEST = "3acd969804d71979ba8071e2446e8ee1ceb3194fc90dcfda802a99e296b7bf48"

SELECTED_MEDIA_COUNT = 15
RAW_CAP_BYTES = 20 * 1024**3
MIN_RAW_CAP_MARGIN_BYTES = 2 * 1024**3
MAX_CONSERVATIVE_ADMISSION_BYTES = RAW_CAP_BYTES - MIN_RAW_CAP_MARGIN_BYTES
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
CELL_SUPPRESSION_K_MIN = 5

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

TRACKED_ARTIFACT_SUFFIXES = frozenset(
    {".json", ".md", ".toml", ".txt", ".yaml", ".yml"}
)

PROHIBITED_PAYLOAD_EXTENSIONS = frozenset(
    {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".wav", ".mp3",
        ".m4a", ".aac", ".flac", ".jpg", ".jpeg", ".png", ".webp",
        ".gif", ".bmp", ".tiff", ".srt", ".vtt", ".zip", ".tar",
        ".gz", ".csv", ".tsv", ".sqlite", ".db",
    }
)

RESTRICTED_MACHINE_KEYS = frozenset(
    {
        "participant_id", "participant_identifier", "participant_key",
        "child_id", "child_identifier", "session_id", "session_identifier",
        "session_key", "recording_id", "recording_identifier", "episode_id",
        "video_id", "audio_id", "media_id", "speaker_id", "speaker_cluster_id",
        "row_id", "filename", "file_name", "source_filename", "relative_path",
        "absolute_path", "media_path", "audio_path", "frame_path", "raw_video",
        "raw_audio", "frame", "frames", "thumbnail", "raw_transcript",
        "transcript_text", "utterance_text", "lexical_string", "word_string",
        "speaker_embedding", "voiceprint", "instrument_embedding",
        "cookie", "cookies", "credential", "credentials", "authorization",
        "api_key", "access_token", "refresh_token",
        "exact_timestamp", "media_timestamp", "timecode", "start_time",
        "end_time", "onset_time", "offset_time", "utterance_start",
        "utterance_end", "frame_time",
    }
)

ABSOLUTE_USER_PATH_RE = re.compile(r"(?i)(?:/Users/|file://|\\Users\\)")
MEDIA_FILENAME_RE = re.compile(
    r"(?i)(?<![a-z0-9_.-])[a-z0-9][a-z0-9_.-]{2,}\."
    r"(?:mp4|mov|mkv|avi|webm|wav|mp3|m4a|aac|flac|jpg|jpeg|png|webp|srt|vtt)\b"
)
MEDIA_CLOCK_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?!\d)")
ISO_DATETIME_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})?\b"
)
SUBTITLE_CUE_RE = re.compile(
    r"(?m)^\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?\s*-->\s*\d{1,2}:\d{2}:\d{2}"
)
DIALOGUE_RE = re.compile(
    r"(?im)^\s*(?:child|non_child|adult|caregiver|speaker[_ -]?\d+)\s*:\s+\S+"
)
UUID_RE = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}\b")


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _issue(issues: list[Issue], code: str, path: Path, root: Path, message: str) -> None:
    issues.append(Issue(code, _relative(path, root), message))


def _normalise_key(key: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _walk(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            normalised = _normalise_key(key)
            yield normalised, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _load_json(path: Path, root: Path, issues: list[Issue]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _issue(
            issues,
            "INVALID_JSON",
            path,
            root,
            f"parse failed ({exc.__class__.__name__}); content suppressed",
        )
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _historical_files(root: Path, version: str) -> list[Path]:
    files: set[Path] = set()
    if version == "v1":
        for name in V1_NAMESPACES:
            path = root / name
            if path.is_file():
                files.add(path)
            elif path.is_dir():
                files.update(item for item in path.rglob("*") if item.is_file())
    elif version == "v1_1":
        for name in V1_1_DIRECTORIES:
            path = root / name
            if path.is_dir():
                files.update(item for item in path.rglob("*") if item.is_file())
        for directory, pattern in V1_1_CODE_GLOBS:
            path = root / directory
            if path.is_dir():
                files.update(item for item in path.glob(pattern) if item.is_file())
    else:
        raise ValueError(f"unsupported historical version {version}")
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _set_digest(root: Path, files: list[Path]) -> tuple[int, str]:
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
    ]
    return len(files), hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value))


def _is_v1_2_related_artifact(relative: Path) -> bool:
    """Identify report-like v1.2 files without following or reading untracked data."""

    text = relative.as_posix().lower()
    return (
        relative.suffix.lower() in TRACKED_ARTIFACT_SUFFIXES | PROHIBITED_PAYLOAD_EXTENSIONS
        and "childlens" in text
        and ("v1_2" in text or "v1.2" in text)
    )


def _git_tracked_v1_2_artifacts(root: Path) -> list[Path]:
    """Return only Git-indexed v1.2 report artifacts.

    `git ls-files` is intentionally used instead of filesystem discovery so
    this additional scan cannot ingest a restricted untracked quarantine.
    The explicit docs/output namespaces are scanned separately, including
    while they are still untracked during construction.
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    files: list[Path] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            relative = Path(raw.decode("utf-8"))
        except UnicodeDecodeError:
            continue
        if relative.is_absolute() or ".." in relative.parts or not _is_v1_2_related_artifact(relative):
            continue
        candidate = root / relative
        try:
            if candidate.is_file() or candidate.is_symlink():
                files.append(candidate)
        except OSError:
            continue
    return sorted(set(files), key=lambda item: item.relative_to(root).as_posix())


def _metric_is_all_pass(metric: Any, denominator: int = SELECTED_MEDIA_COUNT) -> bool:
    return bool(
        isinstance(metric, dict)
        and metric.get("status") == "ALL_PASS"
        and metric.get("pass_count") == denominator
        and metric.get("fail_count") == 0
        and metric.get("cell_suppressed") is False
    )


def _metric_public_pass_count(metric: Any) -> int | None | object:
    """Return the only non-leaking public pass count, or a private sentinel."""

    invalid = _metric_public_pass_count
    if not isinstance(metric, dict):
        return invalid
    status = metric.get("status")
    if status in {"ALL_PASS", "NONE_PASS", "MIXED_COUNTS_EXPORTED"}:
        return metric.get("pass_count")
    if status == "MIXED_SMALL_CELL_SUPPRESSED":
        return None
    return invalid


def _check_suppressed_metric(
    metric: Any,
    name: str,
    denominator: int,
    cell_suppression_k: int,
    path: Path,
    root: Path,
    issues: list[Issue],
) -> None:
    """Validate an aggregate binary metric without allowing small-cell leakage."""

    if not isinstance(metric, dict):
        _issue(issues, "STRUCTURAL_METRIC_INVALID", path, root, f"required metric {name} is absent or invalid")
        return
    status = metric.get("status")
    passed = metric.get("pass_count")
    failed = metric.get("fail_count")
    suppressed = metric.get("cell_suppressed")
    valid = False
    if status == "ALL_PASS":
        valid = passed == denominator and failed == 0 and suppressed is False
    elif status == "NONE_PASS":
        valid = passed == 0 and failed == denominator and suppressed is False
    elif status == "MIXED_SMALL_CELL_SUPPRESSED":
        valid = passed is None and failed is None and suppressed is True
    elif status == "MIXED_COUNTS_EXPORTED":
        valid = (
            _is_int(passed)
            and _is_int(failed)
            and passed >= cell_suppression_k
            and failed >= cell_suppression_k
            and passed + failed == denominator
            and suppressed is False
        )
    if not valid:
        _issue(
            issues,
            "STRUCTURAL_METRIC_INVALID",
            path,
            root,
            f"metric {name} is inconsistent or exposes a small nonzero cell",
        )


def _complete_acquisition_evidence(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("selected_media_count") == SELECTED_MEDIA_COUNT
        and value.get("acquired_media_count") == SELECTED_MEDIA_COUNT
        and value.get("verified_against_frozen_selection_count") == SELECTED_MEDIA_COUNT
        and value.get("acquisition_complete") is True
        and value.get("release_binding_verified") is True
        and value.get("restricted_manifest_digest_verified") is True
        and _is_int(value.get("raw_media_bytes_acquired"))
        and 0 < value.get("raw_media_bytes_acquired") < RAW_CAP_BYTES
        and _is_sha256(value.get("restricted_manifest_sha256"))
        and _is_sha256(value.get("restricted_native_transfer_receipt_sha256"))
    )


def _complete_diagnostic_evidence(value: Any) -> bool:
    metrics = value.get("structural_metrics") if isinstance(value, dict) else None
    return bool(
        isinstance(value, dict)
        and value.get("automated_preparation_complete") is True
        and value.get("media_preparation_attempted_count") == SELECTED_MEDIA_COUNT
        and value.get("diagnostics_passed_for_human_workflow") is True
        and isinstance(metrics, dict)
        and all(_metric_is_all_pass(metrics.get(name)) for name in REQUIRED_STRUCTURAL_METRICS)
    )


def _actual_candidate_minimum_reached(workflow: Any) -> bool:
    if not isinstance(workflow, dict):
        return False
    utterances = workflow.get("populated_candidate_utterance_count")
    speech_minutes = workflow.get("populated_candidate_speech_window_minutes")
    unit = workflow.get("candidate_unit_type")
    utterance_route = (
        unit == "HUMAN_CORRECTED_UTTERANCE"
        and _is_int(utterances)
        and utterances >= 300
    )
    window_route = (
        unit == "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE"
        and utterances == 0
        and _is_int(workflow.get("populated_candidate_speech_window_count"))
        and workflow.get("populated_candidate_speech_window_count") >= CELL_SUPPRESSION_K_MIN
        and _is_number(speech_minutes)
        and speech_minutes >= 30
    )
    return utterance_route or window_route


def _actual_double_code_ready(workflow: Any) -> bool:
    if not isinstance(workflow, dict):
        return False
    eligible = workflow.get("referential_assignment_eligible_count")
    doubled = workflow.get("referential_double_code_assigned_count")
    fraction = workflow.get("referential_double_code_assigned_fraction")
    return bool(
        _is_int(eligible)
        and eligible >= CELL_SUPPRESSION_K_MIN
        and _is_int(doubled)
        and doubled >= 0
        and doubled <= eligible
        and _is_number(fraction)
        and abs(float(fraction) - doubled / eligible) <= 1e-6
        and fraction >= 0.20
        and workflow.get("referential_double_code_sample_frozen") is True
    )


def _honest_pre_referential_double_code_state(workflow: Any) -> bool:
    """Accept a frozen assignment rule before genuine utterance units exist.

    A Human-Ready packet may be populated from official speech windows, but
    those windows must not be relabelled as utterances or referential items.
    The actual achieved counts therefore remain zero/null until authorized
    humans create the adjudicated utterance inventory.
    """

    return bool(
        isinstance(workflow, dict)
        and workflow.get("candidate_unit_type") == "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE"
        and workflow.get("referential_assignment_eligible_count") == 0
        and workflow.get("referential_double_code_assigned_count") == 0
        and workflow.get("referential_double_code_assigned_fraction") is None
        and workflow.get("referential_double_code_sample_frozen") is False
        and workflow.get("referential_assignment_rule_frozen") is True
        and workflow.get("referential_inventory_pending_human_segmentation") is True
        and workflow.get("double_code_fraction_target") == 0.20
    )


def _workflow_packet_ready(workflow: Any) -> bool:
    return bool(
        isinstance(workflow, dict)
        and all(
            workflow.get(key) is True
            for key in (
                "workflow_populated",
                "ready_for_authorized_human",
                "local_only",
                "resumable_batches",
                "input_validation_enabled",
                "instrument_condition_blinded",
                "language_establishment_required_before_transcription",
                "language_matched_humans_required",
                "human_qualification_screening_enabled",
                "language_routes_blinded_and_resumable",
            )
        )
        and workflow.get("selected_media_source_count") == SELECTED_MEDIA_COUNT
        and _actual_candidate_minimum_reached(workflow)
        and (
            _actual_double_code_ready(workflow)
            or _honest_pre_referential_double_code_state(workflow)
        )
    )


def _check_privacy(path: Path, text: str, root: Path, issues: list[Issue]) -> None:
    checks = (
        (ABSOLUTE_USER_PATH_RE, "ABSOLUTE_LOCAL_PATH_EXPORTED", "absolute local path found; value suppressed"),
        (MEDIA_FILENAME_RE, "MEDIA_FILENAME_EXPORTED", "media filename-like text found; value suppressed"),
        (MEDIA_CLOCK_RE, "EXACT_MEDIA_TIME_EXPORTED", "media clock-like text found; value suppressed"),
        (ISO_DATETIME_RE, "EXACT_AUDIT_TIME_EXPORTED", "exact datetime found; value suppressed"),
        (SUBTITLE_CUE_RE, "SUBTITLE_CUE_EXPORTED", "subtitle timing cue found; value suppressed"),
        (DIALOGUE_RE, "TRANSCRIPT_LIKE_TEXT_EXPORTED", "dialogue/transcript-like text found; value suppressed"),
        (UUID_RE, "IDENTIFIER_LIKE_TEXT_EXPORTED", "UUID-like identifier found; value suppressed"),
        (EMAIL_RE, "EMAIL_EXPORTED", "email address found; value suppressed"),
    )
    for pattern, code, message in checks:
        if pattern.search(text):
            _issue(issues, code, path, root, message)


def _check_decision(
    decision: Any,
    acquisition: Any,
    diagnostics: Any,
    workflow: Any,
    path: Path,
    root: Path,
    issues: list[Issue],
) -> None:
    if not isinstance(decision, dict):
        _issue(issues, "DECISION_RECORD_INVALID", path, root, "decision record missing or not an object")
        return
    terminal = decision.get("terminal_state")
    if terminal not in TERMINAL_STATES:
        _issue(issues, "TERMINAL_STATE_INVALID", path, root, "terminal state is absent or invalid")

    false_flags = (
        "permission_reopened",
        "selection_reopened",
        "scientific_thresholds_changed",
        "learner_training_executed",
        "scientific_outcome_executed",
        "causal_arm_executed",
        "tokenizer_trained",
        "checkpoint_created",
        "aea_empirical_ancestry",
        "babyview_empirical_ancestry",
        "restricted_data_egress",
        "corpus_payload_exposed_in_tool_output",
        "restricted_payload_in_repository",
        "external_instrument_artifacts_entered_learner",
    )
    if any(decision.get(key) is not False for key in false_flags):
        _issue(issues, "BOUNDARY_FLAG_VIOLATION", path, root, "one or more mandatory false boundary flags are absent or not false")
    true_flags = (
        "historical_v1_preserved",
        "historical_v1_1_preserved",
        "local_tool_log_quarantine_path_redaction_incident_disclosed",
    )
    if any(decision.get(key) is not True for key in true_flags):
        _issue(issues, "REQUIRED_DISCLOSURE_OR_HISTORY_MISSING", path, root, "historical preservation or required incident disclosure is absent")
    expected_incident = (
        "RESTRICTED_QUARANTINE_PATH_APPEARED_IN_LOCAL_TOOL_LOG_"
        "REDACTED_FROM_REPOSITORY_AND_USER_FACING_ARTIFACTS_NO_CORPUS_PAYLOAD"
    )
    if decision.get("local_tool_log_quarantine_path_redaction_incident_summary") != expected_incident:
        _issue(issues, "INCIDENT_SUMMARY_MISSING", path, root, "required category-level local-tool-log path incident summary is absent")

    if terminal == "CHILDLENS_HUMAN_VALIDATION_READY":
        if not _complete_acquisition_evidence(acquisition):
            _issue(issues, "READY_WITHOUT_COMPLETE_ACQUISITION", path, root, "human-ready state lacks complete 15-item acquisition verification")
        if not _complete_diagnostic_evidence(diagnostics):
            _issue(issues, "READY_WITHOUT_AUTOMATED_PREPARATION", path, root, "human-ready state lacks complete successful automated preparation")
        if not _workflow_packet_ready(workflow):
            _issue(issues, "READY_WITHOUT_USABLE_WORKFLOW", path, root, "human-ready state lacks populated frozen-minimum blinded workflow")
        if not isinstance(workflow, dict) or workflow.get("genuine_human_validation_complete") is not False or workflow.get("human_evidence_fabricated") is not False:
            _issue(issues, "READY_HUMAN_EVIDENCE_STATE_INVALID", path, root, "human-ready state must leave genuine human validation pending and forbid fabrication")
        if decision.get("only_remaining_evidence_is_authorized_human_validation") is not True:
            _issue(issues, "READY_HAS_OTHER_REMAINING_BLOCKER", path, root, "human-ready state does not attest that human evidence is the only remaining item")

    if terminal == "CHILDLENS_FEASIBILITY_GO":
        if not _complete_acquisition_evidence(acquisition):
            _issue(issues, "GO_WITHOUT_COMPLETE_ACQUISITION", path, root, "GO lacks complete 15-item release-bound acquisition evidence")
        if not _complete_diagnostic_evidence(diagnostics):
            _issue(issues, "GO_WITHOUT_COMPLETE_DIAGNOSTICS", path, root, "GO lacks all-pass local structural diagnostic evidence")
        if not _workflow_packet_ready(workflow) or not _actual_double_code_ready(workflow):
            _issue(issues, "GO_WITHOUT_COMPLETE_WORKFLOW", path, root, "GO lacks a populated workflow with actual frozen double-code assignments")
        if not isinstance(workflow, dict) or workflow.get("genuine_human_validation_complete") is not True or workflow.get("independent_reliability_requirement_complete") is not True or workflow.get("human_evidence_fabricated") is not False:
            _issue(issues, "GO_WITHOUT_GENUINE_HUMAN_EVIDENCE", path, root, "GO lacks genuine completed independent human evidence")
        if not isinstance(workflow, dict) or any(
            workflow.get(key) is not True
            for key in (
                "actual_language_established_by_qualified_humans",
                "language_matched_coders_verified",
                "frozen_reliability_thresholds_pass",
            )
        ):
            _issue(issues, "GO_WITHOUT_QUALIFIED_LANGUAGE_EVIDENCE", path, root, "GO lacks qualified language establishment or frozen reliability evidence")
        if decision.get("all_essential_frozen_gates_pass") is not True:
            _issue(issues, "GO_WITHOUT_ALL_GATES", path, root, "GO lacks all-essential-gates attestation")

    if terminal == "CHILDLENS_FEASIBILITY_REVISE" and decision.get("bounded_problem_remaining") is not True:
        _issue(issues, "REVISE_WITHOUT_BOUNDED_PROBLEM", path, root, "REVISE lacks a bounded remaining problem")
    if terminal == "CHILDLENS_FEASIBILITY_STOP" and decision.get("fundamental_corpus_failure_established") is not True:
        _issue(issues, "STOP_WITHOUT_FUNDAMENTAL_FAILURE", path, root, "STOP lacks a fundamental corpus-failure basis")


def _check_acquisition(value: Any, path: Path, root: Path, issues: list[Issue]) -> None:
    if not isinstance(value, dict):
        _issue(issues, "ACQUISITION_RECEIPT_INVALID", path, root, "acquisition receipt missing or not an object")
        return
    mandatory_false = (
        "full_archive_downloaded", "selection_changed", "selection_content_inspected_before_transfer",
        "duplicate_full_resolution_copies_created", "source_filenames_exported",
        "restricted_identifiers_exported", "exact_timestamps_exported",
        "restricted_manifest_exported", "external_or_cloud_transfer",
    )
    if any(value.get(key) is not False for key in mandatory_false):
        _issue(issues, "ACQUISITION_BOUNDARY_VIOLATION", path, root, "one or more acquisition boundary flags are absent or not false")
    if value.get("selected_media_count") != SELECTED_MEDIA_COUNT:
        _issue(issues, "FROZEN_SELECTION_COUNT_MISMATCH", path, root, "selected count differs from frozen 15-item selection")
    acquired = value.get("acquired_media_count")
    verified = value.get("verified_against_frozen_selection_count")
    if not _is_int(acquired) or not 0 <= acquired <= SELECTED_MEDIA_COUNT or not _is_int(verified) or not 0 <= verified <= acquired:
        _issue(issues, "ACQUISITION_COUNT_INVALID", path, root, "acquired/verified aggregate counts are inconsistent")
    raw_bytes = value.get("raw_media_bytes_acquired")
    if not _is_int(raw_bytes) or not 0 <= raw_bytes < RAW_CAP_BYTES:
        _issue(issues, "RAW_CAP_VIOLATION", path, root, "raw acquired bytes are absent, invalid, or not below 20 GiB")
    if value.get("raw_cap_bytes") != RAW_CAP_BYTES:
        _issue(issues, "RAW_CAP_NOT_FROZEN", path, root, "raw cap is absent or changed")
    method = value.get("storage_admission_method")
    if method != "CONSERVATIVE_DISPLAY_BOUND_WITH_HARD_COUNTER":
        _issue(issues, "ADMISSION_METHOD_INVALID", path, root, "frozen conservative storage amendment is missing or changed")

    display_total = value.get("displayed_parsed_total_bytes")
    rounding_upper = value.get("rounding_upper_bound_bytes")
    overhead = value.get("transfer_overhead_bytes")
    bound = value.get("pre_transfer_upper_bound_bytes")
    margin = value.get("upper_bound_margin_to_raw_cap_bytes")
    hard_abort = value.get("hard_abort_before_bytes")
    arithmetic_expected = {
        "displayed_parsed_total_bytes": FROZEN_DISPLAY_TOTAL_BYTES,
        "rounding_upper_bound_bytes": FROZEN_ROUNDING_UPPER_BYTES,
        "transfer_overhead_bytes": FROZEN_TRANSFER_OVERHEAD_BYTES,
        "pre_transfer_upper_bound_bytes": FROZEN_CONSERVATIVE_ADMISSION_BYTES,
        "admission_ceiling_bytes": ADMISSION_CEILING_BYTES,
        "upper_bound_margin_to_raw_cap_bytes": RAW_CAP_BYTES - FROZEN_CONSERVATIVE_ADMISSION_BYTES,
        "hard_abort_before_bytes": RAW_CAP_BYTES - 1,
        "predeclared_nonraw_reserve_bytes": NONRAW_RESERVE_BYTES,
    }
    if any(value.get(key) != expected for key, expected in arithmetic_expected.items()) or not (
        _is_int(display_total)
        and _is_int(rounding_upper)
        and _is_int(overhead)
        and _is_int(bound)
        and rounding_upper >= display_total
        and bound == rounding_upper + overhead
        and bound <= MAX_CONSERVATIVE_ADMISSION_BYTES
        and _is_int(margin)
        and margin == RAW_CAP_BYTES - bound
        and margin >= MIN_RAW_CAP_MARGIN_BYTES
        and _is_int(hard_abort)
        and hard_abort == RAW_CAP_BYTES - 1
    ):
        _issue(issues, "CONSERVATIVE_ADMISSION_ARITHMETIC_INVALID", path, root, "frozen conservative admission components, caps, or margins are inconsistent")

    pre_namespace = value.get("pre_transfer_namespace_bytes")
    pre_free = value.get("pre_transfer_free_bytes")
    projected_namespace = value.get("projected_namespace_peak_bytes")
    projected_free = value.get("projected_post_peak_free_bytes")
    post_free = value.get("post_transfer_free_bytes")
    after_namespace = value.get("namespace_bytes_after_acquisition")
    storage_valid = (
        value.get("namespace_cap_bytes") == NAMESPACE_CAP_BYTES
        and value.get("projected_post_peak_free_floor_bytes") == POST_PEAK_FREE_FLOOR_BYTES
        and _is_int(pre_namespace)
        and pre_namespace >= 0
        and _is_int(pre_free)
        and pre_free >= 0
        and _is_int(projected_namespace)
        and projected_namespace == pre_namespace + FROZEN_CONSERVATIVE_ADMISSION_BYTES + NONRAW_RESERVE_BYTES
        and projected_namespace <= NAMESPACE_CAP_BYTES
        and _is_int(projected_free)
        and projected_free == pre_free - FROZEN_CONSERVATIVE_ADMISSION_BYTES - NONRAW_RESERVE_BYTES
        and projected_free >= POST_PEAK_FREE_FLOOR_BYTES
        and _is_int(post_free)
        and post_free >= POST_PEAK_FREE_FLOOR_BYTES
        and _is_int(after_namespace)
        and after_namespace >= 0
        and after_namespace <= NAMESPACE_CAP_BYTES
    )
    if not storage_valid:
        _issue(issues, "STORAGE_CAP_OR_FREE_FLOOR_INVALID", path, root, "namespace cap, projected arithmetic, post-transfer free floor, or actual namespace evidence is inconsistent")

    release_booleans = (
        "release_binding_verified",
        "restricted_manifest_digest_verified",
        "live_snapshot_limitation_disclosed",
    )
    if any(not isinstance(value.get(key), bool) for key in release_booleans) or any(
        value.get(key) != expected
        for key, expected in {
            "frozen_selection_digest": FROZEN_SELECTION_DIGEST,
            "immutable_public_snapshot_commit": PUBLIC_SNAPSHOT_COMMIT,
            "pilot_source": "ACCESSIBLE_LIVE_VIEW_BOUND_TO_RESTRICTED_MANIFEST_DIGEST",
            "live_to_public_snapshot_byte_equivalence": "NOT_PROVEN",
            "live_snapshot_limitation_disclosed": True,
            "retention_deadline": RETENTION_DEADLINE,
            "retention_control_recorded": True,
        }.items()
    ):
        _issue(issues, "RELEASE_RETENTION_BINDING_INVALID", path, root, "selection/release binding, live-snapshot limitation, or retention evidence is absent or changed")
    restricted_digest = value.get("restricted_manifest_sha256")
    transfer_digest = value.get("restricted_native_transfer_receipt_sha256")
    if acquired:
        if not _is_sha256(restricted_digest) or not _is_sha256(transfer_digest):
            _issue(issues, "ACTUAL_ACQUISITION_SOURCE_EVIDENCE_INVALID", path, root, "actual restricted manifest/native-transfer evidence digest is absent")
    elif restricted_digest is not None or transfer_digest is not None:
        if not all(item is None or _is_sha256(item) for item in (restricted_digest, transfer_digest)):
            _issue(issues, "ACTUAL_ACQUISITION_SOURCE_EVIDENCE_INVALID", path, root, "acquisition evidence digest is malformed")
    if value.get("sequential_or_tightly_bounded_transfer") is not True:
        _issue(issues, "TRANSFER_CONCURRENCY_NOT_BOUNDED", path, root, "bounded transfer attestation is absent")
    if value.get("cumulative_stream_counter_enforced") is not True:
        _issue(issues, "HARD_COUNTER_NOT_ATTESTED", path, root, "hard cumulative byte counter evidence is absent")


def _check_diagnostics(value: Any, acquisition: Any, path: Path, root: Path, issues: list[Issue]) -> None:
    if not isinstance(value, dict):
        _issue(issues, "DIAGNOSTICS_RECEIPT_INVALID", path, root, "diagnostics receipt missing or not an object")
        return
    mandatory_true = ("aggregate_only", "local_offline_only", "automated_outputs_not_gold")
    mandatory_false = (
        "lexical_content_exported", "transcript_text_exported", "speaker_labels_exported",
        "frames_exported", "external_api_used", "learner_data_created",
        "instrument_features_passed_to_learner", "instrument_embeddings_passed_to_learner",
        "instrument_tokenizer_or_vocabulary_passed_to_learner",
    )
    if any(value.get(key) is not True for key in mandatory_true) or any(value.get(key) is not False for key in mandatory_false):
        _issue(issues, "DIAGNOSTIC_BOUNDARY_VIOLATION", path, root, "diagnostic aggregate/local/not-gold boundary flags are inconsistent")
    acquired = acquisition.get("acquired_media_count") if isinstance(acquisition, dict) else None
    examined = value.get("media_preparation_attempted_count")
    if value.get("source_acquired_media_count") != acquired:
        _issue(issues, "DIAGNOSTIC_ACQUISITION_CROSSCHECK_MISMATCH", path, root, "diagnostic receipt does not agree with acquired media count")
    if not _is_int(examined) or not _is_int(acquired) or not 0 <= examined <= acquired:
        _issue(issues, "DIAGNOSTIC_COUNT_MISMATCH", path, root, "diagnostic count exceeds or is inconsistent with acquisition count")
    for key in ("container_probe_attempted_count", "decode_attempted_count"):
        count = value.get(key)
        if not _is_int(count) or not _is_int(examined) or not 0 <= count <= examined:
            _issue(issues, "DIAGNOSTIC_SUBCOUNT_INVALID", path, root, "diagnostic subcount is invalid")
    for success_key, attempted_key, metric_key in (
        ("container_probe_success_count", "container_probe_attempted_count", "probe_success"),
        ("decode_success_count", "decode_attempted_count", "full_decode_success"),
    ):
        success = value.get(success_key)
        attempted = value.get(attempted_key)
        metric = value.get("structural_metrics", {}).get(metric_key) if isinstance(value.get("structural_metrics"), dict) else None
        suppressed_failure = isinstance(metric, dict) and metric.get("status") == "MIXED_SMALL_CELL_SUPPRESSED"
        if not _is_int(attempted) or not (
            (_is_int(success) and 0 <= success <= attempted)
            or (success is None and suppressed_failure)
        ):
            _issue(issues, "DIAGNOSTIC_SUCCESS_COUNT_INVALID", path, root, "diagnostic success count exceeds attempts or is invalid")

    cell_k = value.get("cell_suppression_k")
    metrics = value.get("structural_metrics")
    if not _is_int(cell_k) or cell_k < CELL_SUPPRESSION_K_MIN or not isinstance(metrics, dict):
        _issue(issues, "STRUCTURAL_METRIC_SET_INVALID", path, root, "cell-suppressed actual-source structural metric set is absent")
    else:
        for name in REQUIRED_STRUCTURAL_METRICS:
            _check_suppressed_metric(
                metrics.get(name), name, SELECTED_MEDIA_COUNT, cell_k, path, root, issues
            )
        for count_key, metric_name in (
            ("container_probe_success_count", "probe_success"),
            ("decode_success_count", "full_decode_success"),
            ("audio_stream_media_count", "audio_stream_present"),
            (
                "speech_presence_window_media_count",
                "speech_presence_window_expectation_satisfied",
            ),
        ):
            expected = _metric_public_pass_count(metrics.get(metric_name))
            if expected is _metric_public_pass_count or value.get(count_key) != expected:
                _issue(issues, "DIAGNOSTIC_FLAT_COUNT_MISMATCH", path, root, "flat diagnostic count disagrees with its cell-suppressed source metric")
        all_structural_pass = all(
            _metric_is_all_pass(metrics.get(name))
            for name in REQUIRED_STRUCTURAL_METRICS
        )
        if value.get("diagnostics_passed_for_human_workflow") is not all_structural_pass:
            _issue(issues, "DIAGNOSTIC_PASS_FLAG_MISMATCH", path, root, "diagnostic pass flag disagrees with required structural metrics")
        if value.get("automated_preparation_complete") is not all_structural_pass:
            _issue(issues, "AUTOMATED_PREPARATION_FLAG_MISMATCH", path, root, "automated preparation completion disagrees with required structural metrics")
    if value.get("small_nonzero_failure_cells_exported") is not False:
        _issue(issues, "SMALL_CELL_EXPORT_CONTROL_INVALID", path, root, "small nonzero diagnostic failures are not attested suppressed")

    actual_source_expected = {
        "measurement_source": "LOCAL_RESTRICTED_MEDIA_AUDIT_V1_2",
        "measurement_executed_on_acquired_bytes": True,
        "speech_window_actual_source": "OFFICIAL_ANNOTATION_WINDOWS_LINKED_TO_ACQUIRED_MEDIA",
        "annotation_linkage_actual_source": "RESTRICTED_CANONICAL_MANIFEST",
    }
    if any(value.get(key) != expected for key, expected in actual_source_expected.items()):
        _issue(issues, "DIAGNOSTIC_ACTUAL_SOURCE_INVALID", path, root, "local actual-source measurement/linkage evidence is absent")
    selection_digest = value.get("pilot_selection_sha256")
    transfer_digest = value.get("source_native_transfer_receipt_sha256")
    measurement_digest = value.get("restricted_measurement_receipt_sha256")
    acquisition_transfer_digest = (
        acquisition.get("restricted_native_transfer_receipt_sha256")
        if isinstance(acquisition, dict)
        else None
    )
    if (
        selection_digest != FROZEN_SELECTION_DIGEST
        or not _is_sha256(measurement_digest)
        or not _is_sha256(transfer_digest)
        or transfer_digest != acquisition_transfer_digest
    ):
        _issue(issues, "DIAGNOSTIC_SOURCE_DIGEST_MISMATCH", path, root, "diagnostic actual-source digests do not bind to the frozen acquisition")

    union_minutes = value.get("official_speech_window_union_minutes")
    if not _is_number(union_minutes) or union_minutes < 0:
        _issue(issues, "SPEECH_WINDOW_DURATION_INVALID", path, root, "official speech-window union duration is absent or invalid")


def _check_workflow(
    value: Any,
    acquisition: Any,
    diagnostics: Any,
    path: Path,
    root: Path,
    issues: list[Issue],
) -> None:
    if not isinstance(value, dict):
        _issue(issues, "WORKFLOW_RECEIPT_INVALID", path, root, "workflow receipt missing or not an object")
        return
    mandatory_true = (
        "local_only", "autosave_to_restricted_quarantine_only", "resumable_batches",
        "input_validation_enabled", "instrument_condition_blinded",
        "language_establishment_required_before_transcription",
        "language_matched_humans_required", "human_qualification_screening_enabled",
        "language_routes_blinded_and_resumable",
    )
    mandatory_false = (
        "identifiers_displayed", "source_filenames_displayed", "exact_timestamps_exported",
        "transcript_text_exported", "external_service_used", "human_evidence_fabricated",
    )
    if any(value.get(key) is not True for key in mandatory_true) or any(value.get(key) is not False for key in mandatory_false):
        _issue(issues, "WORKFLOW_BOUNDARY_VIOLATION", path, root, "workflow locality/blinding/privacy flags are inconsistent")
    fraction = value.get("double_code_fraction_target")
    if not _is_number(fraction) or not 0.20 <= fraction <= 1.0:
        _issue(issues, "DOUBLE_CODE_TARGET_INVALID", path, root, "double-code target is below 20 percent or invalid")

    eligible = value.get("referential_assignment_eligible_count")
    if eligible == 0:
        if not _honest_pre_referential_double_code_state(value):
            _issue(issues, "PRE_REFERENTIAL_DOUBLE_CODE_STATE_INVALID", path, root, "zero-item referential state does not honestly preserve the pending frozen assignment rule")
    elif not _actual_double_code_ready(value):
        _issue(issues, "ACTUAL_DOUBLE_CODE_EVIDENCE_INVALID", path, root, "actual double-code numerator, denominator, fraction, or freeze state is inconsistent")

    estimates = (
        value.get("estimated_first_authorized_human_minutes"),
        value.get("estimated_second_independent_human_minutes"),
        value.get("estimated_adjudication_minutes"),
    )
    estimated_total = value.get("estimated_total_human_minutes")
    if (
        not all(_is_number(item) and item > 0 for item in estimates)
        or not _is_number(estimated_total)
        or abs(float(estimated_total) - sum(float(item) for item in estimates)) > 1e-6
    ):
        _issue(issues, "HUMAN_TIME_ESTIMATE_INVALID", path, root, "separate first-human, second-human, adjudication, and total estimates are absent or inconsistent")
    source_count = value.get("selected_media_source_count")
    if not _is_int(source_count) or not 0 <= source_count <= SELECTED_MEDIA_COUNT:
        _issue(issues, "WORKFLOW_SOURCE_COUNT_INVALID", path, root, "workflow source-media aggregate is invalid")

    if value.get("candidate_unit_type") not in {
        "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE",
        "HUMAN_CORRECTED_UTTERANCE",
    }:
        _issue(issues, "CANDIDATE_UNIT_TYPE_INVALID", path, root, "candidate unit type is absent or conflates windows with utterances")
    utterances = value.get("populated_candidate_utterance_count")
    window_count = value.get("populated_candidate_speech_window_count")
    window_minutes = value.get("populated_candidate_speech_window_minutes")
    if (
        not _is_int(utterances)
        or utterances < 0
        or not _is_int(window_count)
        or window_count < 0
        or not _is_number(window_minutes)
        or window_minutes < 0
    ):
        _issue(issues, "CANDIDATE_AGGREGATE_INVALID", path, root, "candidate utterance/window count or duration aggregate is invalid")

    actual_source_expected = {
        "workflow_source": "LOCAL_RESTRICTED_HUMAN_VALIDATION_PACKET_V1_2",
        "packet_populated_from_actual_acquired_media": True,
    }
    if any(value.get(key) != expected for key, expected in actual_source_expected.items()):
        _issue(issues, "WORKFLOW_ACTUAL_SOURCE_INVALID", path, root, "workflow is not attested populated from local acquired media")
    workflow_digest = value.get("restricted_workflow_receipt_sha256")
    transfer_digest = value.get("source_native_transfer_receipt_sha256")
    measurement_digest = value.get("source_measurement_receipt_sha256")
    if (
        not _is_sha256(workflow_digest)
        or not _is_sha256(transfer_digest)
        or not _is_sha256(measurement_digest)
        or not isinstance(acquisition, dict)
        or transfer_digest != acquisition.get("restricted_native_transfer_receipt_sha256")
        or not isinstance(diagnostics, dict)
        or measurement_digest != diagnostics.get("restricted_measurement_receipt_sha256")
        or value.get("pilot_selection_sha256") != FROZEN_SELECTION_DIGEST
    ):
        _issue(issues, "WORKFLOW_SOURCE_DIGEST_MISMATCH", path, root, "workflow actual-source digests do not bind to acquisition and diagnostics")
    if isinstance(diagnostics, dict) and _is_number(window_minutes) and _is_number(
        diagnostics.get("official_speech_window_union_minutes")
    ) and abs(float(window_minutes) - float(diagnostics["official_speech_window_union_minutes"])) > 1e-6:
        _issue(issues, "WORKFLOW_SPEECH_DURATION_CROSSCHECK_MISMATCH", path, root, "workflow candidate speech-window duration disagrees with diagnostics")


def validate(root: Path) -> list[Issue]:
    root = root.resolve()
    docs = root / "docs" / "childlens_feasibility_v1_2"
    output = root / "output" / "childlens_feasibility_v1_2"
    issues: list[Issue] = []

    for directory, names in ((docs, DOCS_REQUIRED), (output, OUTPUT_REQUIRED)):
        for name in names:
            path = directory / name
            if not path.is_file():
                _issue(issues, "MISSING_REQUIRED_FILE", path, root, "required v1.2 artifact absent")

    artifacts: list[Path] = []
    for directory in (docs, output):
        if directory.is_dir():
            artifacts.extend(path for path in directory.rglob("*") if path.is_file())

    # The explicit report namespaces are inspected even while under
    # construction and untracked. Outside them, inspect only Git-indexed,
    # report-like v1.2 artifacts; never enumerate untracked files.
    privacy_artifacts = sorted(
        set(artifacts) | set(_git_tracked_v1_2_artifacts(root)),
        key=lambda item: item.relative_to(root).as_posix(),
    )

    terminals: set[str] = set()
    parsed: dict[str, Any] = {}
    report_artifact_set = set(artifacts)
    for path in privacy_artifacts:
        if path.is_symlink():
            _issue(issues, "ARTIFACT_SYMLINK_REJECTED", path, root, "v1.2 artifact symlink rejected without following target")
            continue
        if path.suffix.lower() in PROHIBITED_PAYLOAD_EXTENSIONS:
            _issue(issues, "PROHIBITED_PAYLOAD_EXTENSION", path, root, "payload-like file present in report namespace")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            _issue(issues, "UNREADABLE_ARTIFACT", path, root, f"read failed ({exc.__class__.__name__})")
            continue
        if path in report_artifact_set:
            terminals.update(TERMINAL_RE.findall(text))
        _check_privacy(path, text, root, issues)
        if path.parent == output and path.suffix.lower() == ".json":
            value = _load_json(path, root, issues)
            if value is not None:
                parsed[path.name] = value
                for key, _ in _walk(value):
                    if key in RESTRICTED_MACHINE_KEYS:
                        _issue(issues, "RESTRICTED_MACHINE_FIELD", path, root, "restricted row-level field found; value suppressed")

    if len(terminals) != 1:
        _issue(issues, "TERMINAL_STATE_SET_INVALID", output / "decision_record.json", root, "v1.2 report namespace must contain exactly one terminal literal")

    v1_count, v1_digest = _set_digest(root, _historical_files(root, "v1"))
    v1_1_count, v1_1_digest = _set_digest(root, _historical_files(root, "v1_1"))
    if (v1_count, v1_digest) != (EXPECTED_V1_COUNT, EXPECTED_V1_DIGEST):
        _issue(issues, "V1_IMMUTABILITY_MISMATCH", output / "immutability_receipt.json", root, "historical v1 artifact set changed")
    if (v1_1_count, v1_1_digest) != (EXPECTED_V1_1_COUNT, EXPECTED_V1_1_DIGEST):
        _issue(issues, "V1_1_IMMUTABILITY_MISMATCH", output / "immutability_receipt.json", root, "historical v1.1 artifact set changed")

    immutability = parsed.get("immutability_receipt.json")
    if not isinstance(immutability, dict) or any(
        immutability.get(key) != expected
        for key, expected in {
            "v1_artifact_count": EXPECTED_V1_COUNT,
            "v1_set_digest": EXPECTED_V1_DIGEST,
            "v1_1_artifact_count": EXPECTED_V1_1_COUNT,
            "v1_1_set_digest": EXPECTED_V1_1_DIGEST,
            "v1_preserved": True,
            "v1_1_preserved": True,
            "historical_decisions_overwritten": False,
        }.items()
    ):
        _issue(issues, "IMMUTABILITY_RECEIPT_MISMATCH", output / "immutability_receipt.json", root, "immutability receipt is absent or inconsistent")

    acquisition = parsed.get("acquisition_receipt.json")
    diagnostics = parsed.get("automated_diagnostics_receipt.json")
    workflow = parsed.get("human_validation_workflow_receipt.json")
    decision = parsed.get("decision_record.json")
    _check_acquisition(acquisition, output / "acquisition_receipt.json", root, issues)
    _check_diagnostics(diagnostics, acquisition, output / "automated_diagnostics_receipt.json", root, issues)
    _check_workflow(
        workflow,
        acquisition,
        diagnostics,
        output / "human_validation_workflow_receipt.json",
        root,
        issues,
    )
    _check_decision(decision, acquisition, diagnostics, workflow, output / "decision_record.json", root, issues)

    if isinstance(decision, dict) and len(terminals) == 1 and decision.get("terminal_state") not in terminals:
        _issue(issues, "TERMINAL_STATE_CROSSCHECK_MISMATCH", output / "decision_record.json", root, "decision record disagrees with terminal literal set")

    return sorted(issues, key=lambda item: (item.path, item.code, item.message))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    issues = validate(root)
    if issues:
        print(f"FAIL: {len(issues)} ChildLens feasibility v1.2 validation issue(s)")
        for issue in issues:
            print(f"- {issue.code} [{issue.path}]: {issue.message}")
        return 1
    print("PASS: ChildLens feasibility v1.2 artifacts are immutable, internally consistent, and privacy-clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
