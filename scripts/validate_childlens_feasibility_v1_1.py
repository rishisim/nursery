#!/usr/bin/env python3
"""Fail-closed consistency/privacy validator for ChildLens feasibility v1.1.

The validator reads only repository artifacts. It never reads Keeper, email, or
the restricted quarantine, and it never echoes matched content in diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DOCS_REQUIRED = (
    "annotation_instrument_contract_amendment_v1_1.md",
    "annotation_schema_modality_audit_v1_1.md",
    "commands_tests_and_limitations_v1_1.md",
    "continuation_scope.md",
    "executive_decision_report.md",
    "frozen_feasibility_rubric_v1_1.md",
    "future_parallel_execution_plan_v1_1.md",
    "human_validation_packet_handoff.md",
    "instrument_preflight.md",
    "language_transcription_diarization_feasibility_v1_1.md",
    "lexical_grounding_referential_feasibility_v1_1.md",
    "manifest_and_selection_engineering.md",
    "nonidentifying_manifest_and_count_reconciliation.md",
    "permission_resolution.md",
    "provenance_privacy_audit_v1_1.md",
    "redacted_extractor_validation.md",
    "resource_admission_preflight.md",
    "v1_immutability_receipt.json",
)

OUTPUT_REQUIRED = (
    "annotation_structural_receipt.json",
    "decision_record.json",
    "instrument_preflight.json",
    "permission_receipt.json",
    "pilot_preselection_receipt.json",
    "quarantine_admission_receipt.json",
    "release_binding_receipt.json",
    "resource_admission_preflight.json",
)

V1_NAMESPACES = (
    "docs/childlens_feasibility_v1",
    "output/childlens_feasibility_v1",
    "scripts/validate_childlens_feasibility_v1.py",
    "tests/test_childlens_feasibility_v1.py",
)

EXPECTED_V1_DIGEST = "35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf"
EXPECTED_RELEASE_RECEIPT_DIGEST = "2aa1dcf53fc5592965019afe5df97b51ccc2fe05e70acb2d51aa396c73843805"
EXPECTED_TERMINAL = "CHILDLENS_FEASIBILITY_REVISE"
TERMINAL_RE = re.compile(r"CHILDLENS_FEASIBILITY_(?:GO|REVISE|STOP)")
EXPECTED_GATE_IDS = (
    "G1_TERMS",
    "G2_RELEASE_GROUPING",
    "G3_AUDIO_TIMING",
    "G4_TRANSCRIPT_ROLE",
    "G5_INPUT_LEXICON",
    "G6_REFERENTIAL_ANNOTATION",
    "G7_HELDOUT_EVALUATION",
    "G8_BASELINE_CEILING",
    "G9_SIMULATOR_CALIBRATION",
    "G10_RESOURCES",
)

PROHIBITED_EXTENSIONS = frozenset(
    {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".wav", ".mp3", ".m4a",
        ".aac", ".flac", ".jpg", ".jpeg", ".png", ".webp", ".srt", ".vtt",
        ".csv", ".tsv", ".zip",
    }
)
RESTRICTED_MACHINE_KEYS = frozenset(
    {
        "participant_id", "participant_key", "session_id", "session_key",
        "video_id", "media_id", "speaker_id", "filename", "file_name",
        "relative_path", "absolute_path", "media_path", "raw_transcript",
        "transcript_text", "utterance_text", "exact_timestamp", "media_timestamp",
        "utterance_start", "utterance_end",
    }
)
ABSOLUTE_RESTRICTED_ROOT_RE = re.compile(
    r"/Users/[^/\s]+/(?:[^\s]*/)*\.childlens_restricted(?:/|\b)", re.IGNORECASE
)
MEDIA_FILENAME_RE = re.compile(
    r"(?i)(?<![a-z0-9_.-])[a-z0-9][a-z0-9_.-]{2,}\.(?:mp4|mov|mkv|avi|webm|wav|mp3|m4a|aac|flac|jpg|jpeg|png|srt|vtt)\b"
)
SUBTITLE_CUE_RE = re.compile(
    r"(?m)^\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?\s*-->\s*\d{1,2}:\d{2}:\d{2}"
)
STALE_STATUS_PHRASES = (
    "PENDING_CONTENT_BLIND_PRESELECTION",
    "NOT_PERFORMED_FROM_AGGREGATES",
    "No pilot episode was selected",
    "G8_BASELINE_CEILING = UNRESOLVED",
)


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str


def _issue(issues: list[Issue], code: str, path: Path, root: Path, message: str) -> None:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.name
    issues.append(Issue(code, rel, message))


def _load_json(path: Path, root: Path, issues: list[Issue]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _issue(issues, "INVALID_JSON", path, root, f"parse failed ({exc.__class__.__name__}); content suppressed")
        return None


def _walk(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _v1_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in V1_NAMESPACES:
        path = root / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _v1_set_digest(root: Path) -> tuple[int, str]:
    files = _v1_files(root)
    lines = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
    ]
    serialized = "".join(sorted(lines)).encode("utf-8")
    return len(files), hashlib.sha256(serialized).hexdigest()


def validate(root: Path) -> list[Issue]:
    root = root.resolve()
    docs = root / "docs" / "childlens_feasibility_v1_1"
    output = root / "output" / "childlens_feasibility_v1_1"
    issues: list[Issue] = []

    for directory, required in ((docs, DOCS_REQUIRED), (output, OUTPUT_REQUIRED)):
        for name in required:
            path = directory / name
            if not path.is_file():
                _issue(issues, "MISSING_REQUIRED_FILE", path, root, "required artifact absent")

    artifact_files = []
    for directory in (docs, output):
        if directory.is_dir():
            artifact_files.extend(path for path in directory.rglob("*") if path.is_file())

    terminal_literals: set[str] = set()
    parsed: dict[str, Any] = {}
    for path in artifact_files:
        if path.suffix.lower() in PROHIBITED_EXTENSIONS:
            _issue(issues, "PROHIBITED_PAYLOAD_EXTENSION", path, root, "payload-like file present")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            _issue(issues, "UNREADABLE_ARTIFACT", path, root, f"read failed ({exc.__class__.__name__})")
            continue
        terminal_literals.update(TERMINAL_RE.findall(text))
        if ABSOLUTE_RESTRICTED_ROOT_RE.search(text):
            _issue(issues, "RESTRICTED_ROOT_EXPORTED", path, root, "absolute restricted-root string found; value suppressed")
        if MEDIA_FILENAME_RE.search(text):
            _issue(issues, "MEDIA_FILENAME_EXPORTED", path, root, "media filename-like string found; value suppressed")
        if SUBTITLE_CUE_RE.search(text):
            _issue(issues, "SUBTITLE_CUE_EXPORTED", path, root, "subtitle timing cue found; value suppressed")
        if any(phrase in text for phrase in STALE_STATUS_PHRASES):
            _issue(issues, "STALE_STATUS_TEXT", path, root, "superseded preselection or gate status remains")
        if path.suffix == ".json":
            value = _load_json(path, root, issues)
            if value is not None:
                parsed[path.name] = value
                for key, _ in _walk(value):
                    if key.lower() in RESTRICTED_MACHINE_KEYS:
                        _issue(issues, "RESTRICTED_MACHINE_FIELD", path, root, "restricted row-level field found; value suppressed")

    if terminal_literals != {EXPECTED_TERMINAL}:
        _issue(issues, "TERMINAL_DECISION_SET", output / "decision_record.json", root, "v1.1 terminal literal set is not exactly the expected decision")

    v1_count, v1_digest = _v1_set_digest(root)
    if v1_count != 23 or v1_digest != EXPECTED_V1_DIGEST:
        _issue(issues, "V1_IMMUTABILITY_MISMATCH", docs / "v1_immutability_receipt.json", root, "historical v1 count/digest mismatch")

    release_path = output / "release_binding_receipt.json"
    if release_path.is_file() and _sha256(release_path) != EXPECTED_RELEASE_RECEIPT_DIGEST:
        _issue(issues, "RELEASE_RECEIPT_DIGEST_MISMATCH", release_path, root, "bound release receipt changed")

    decision = parsed.get("decision_record.json")
    permission = parsed.get("permission_receipt.json")
    release = parsed.get("release_binding_receipt.json")
    structural = parsed.get("annotation_structural_receipt.json")
    pilot = parsed.get("pilot_preselection_receipt.json")
    quarantine = parsed.get("quarantine_admission_receipt.json")

    if isinstance(decision, dict):
        if decision.get("terminal_decision") != EXPECTED_TERMINAL:
            _issue(issues, "DECISION_RECORD_MISMATCH", output / "decision_record.json", root, "decision record does not contain expected terminal decision")
        gates = decision.get("gate_assessments")
        if not isinstance(gates, list) or tuple(row.get("gate_id") for row in gates if isinstance(row, dict)) != EXPECTED_GATE_IDS:
            _issue(issues, "GATE_SET_MISMATCH", output / "decision_record.json", root, "gate order or membership mismatch")
        elif all(row.get("status") in {"PASS", "PASS_WITH_LIMITATIONS", "PASS_WITH_CONTROLLING_CONSTRAINTS"} for row in gates):
            _issue(issues, "REVISE_LOGIC_VIOLATION", output / "decision_record.json", root, "REVISE has no conditional or unresolved gate")
        if any(decision.get(key) is not False for key in ("scientific_outcome_executed", "learner_training_executed", "causal_arm_executed", "media_pilot_executed")):
            _issue(issues, "UNAUTHORIZED_OUTCOME_STATE", output / "decision_record.json", root, "an outcome/training/media-pilot flag is not false")

    if not isinstance(permission, dict) or permission.get("status") != "PASS_WITH_CONTROLLING_CONSTRAINTS":
        _issue(issues, "PERMISSION_STATUS_MISMATCH", output / "permission_receipt.json", root, "G1 permission receipt is missing or inconsistent")

    if isinstance(release, dict):
        live = release.get("accessible_live_view", {})
        binding = release.get("binding", {})
        if live.get("video_object_count") != 192 or live.get("final_annotation_object_count") != 192:
            _issue(issues, "RELEASE_COUNT_MISMATCH", output / "release_binding_receipt.json", root, "release inventory counts changed")
        if live.get("video_exact_byte_attribute_count") != 0 or binding.get("live_to_public_snapshot_byte_equivalence") != "NOT_PROVEN":
            _issue(issues, "RELEASE_LIMITATION_MISMATCH", output / "release_binding_receipt.json", root, "exact-byte/equivalence limitation changed")
    else:
        _issue(issues, "RELEASE_RECEIPT_ABSENT", output / "release_binding_receipt.json", root, "release receipt missing or invalid")

    if isinstance(structural, dict):
        local = structural.get("local_annotation_summary", {})
        if local.get("media_annotation_one_to_one_linkage") != "STRUCTURAL_CANONICAL_STEM_LINK_PROVEN_192_OF_192_BYTE_CHECKSUM_LINK_NOT_PROVEN" or local.get("participant_table_structural_link_complete_count") != 192:
            _issue(issues, "STRUCTURAL_LINKAGE_MISMATCH", output / "annotation_structural_receipt.json", root, "structural linkage state changed")

    if isinstance(pilot, dict):
        required_pilot = {
            "status": "FROZEN_PRESELECTION_NOT_ACQUISITION_ADMISSION",
            "selected_video_count": 15,
            "selection_participant_group_count": 15,
            "maximum_selected_videos_per_participant": 1,
            "selection_content_blind": True,
            "placeholder_video_sizes_are_scientific_evidence": False,
            "canonical_final_manifest_complete": False,
            "video_acquisition_started": False,
            "human_validation_started": False,
            "learner_training_started": False,
        }
        if any(pilot.get(key) != expected for key, expected in required_pilot.items()):
            _issue(issues, "PILOT_STATE_MISMATCH", output / "pilot_preselection_receipt.json", root, "frozen preselection state changed")
    else:
        _issue(issues, "PILOT_RECEIPT_ABSENT", output / "pilot_preselection_receipt.json", root, "pilot receipt missing or invalid")

    if isinstance(quarantine, dict):
        zero_fields = ("video_file_count", "decoded_audio_file_count", "frame_file_count", "transcript_file_count")
        if quarantine.get("status") != "ADMITTED_FOR_METADATA_AND_ANNOTATION_SHARDS_ONLY" or any(quarantine.get(key) != 0 for key in zero_fields) or quarantine.get("video_transfer_admitted") is not False:
            _issue(issues, "QUARANTINE_SCOPE_MISMATCH", output / "quarantine_admission_receipt.json", root, "quarantine admission scope changed")

    return sorted(issues, key=lambda item: (item.path, item.code, item.message))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    issues = validate(root)
    if issues:
        print(f"FAIL: {len(issues)} ChildLens feasibility v1.1 validation issue(s)")
        for issue in issues:
            print(f"- {issue.code} [{issue.path}]: {issue.message}")
        return 1
    print("PASS: ChildLens feasibility v1.1 artifacts are internally consistent and privacy-clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
