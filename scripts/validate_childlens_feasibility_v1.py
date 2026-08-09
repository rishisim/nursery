#!/usr/bin/env python3
"""Fail-closed validation for the public ChildLens feasibility-v1 artifacts.

This validator never reads a corpus or a restricted manifest.  It scans only the
two report namespaces declared below and deliberately omits matched values from
diagnostics so that a validation log cannot amplify a privacy mistake.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DOCS_REQUIRED = (
    "executive_decision_report.md",
    "release_terms_permissions_matrix.md",
    "release_manifest_summary.json",
    "annotation_schema_modality_audit.md",
    "language_transcription_diarization_feasibility.md",
    "lexical_grounding_referential_feasibility.md",
    "proposed_annotation_instrument_contract_amendment.md",
    "storage_compute_annotation_projection.md",
    "future_parallel_execution_plan.md",
    "frozen_feasibility_rubric_v1.json",
    "frozen_pilot_protocol_v1.json",
    "decision_record_v1.json",
    "provenance_privacy_audit.md",
    "commands_tests_and_limitations.md",
)

OUTPUT_REQUIRED = (
    "release_manifest_nonidentifying.json",
    "audit_evidence_index.json",
)

TERMINAL_STATES = (
    "CHILDLENS_FEASIBILITY_GO",
    "CHILDLENS_FEASIBILITY_REVISE",
    "CHILDLENS_FEASIBILITY_STOP",
)

MACHINE_EXTENSIONS = frozenset({".json", ".jsonl", ".yaml", ".yml", ".csv", ".tsv"})
PROHIBITED_PAYLOAD_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".webm",
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".flac",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".bmp",
        ".tiff",
        ".srt",
        ".vtt",
    }
)

# Exact normalized keys that have row-level values in the restricted workspace.
# Availability booleans such as participant_grouping_available do not match.
RESTRICTED_JSON_KEYS = frozenset(
    {
        "participant_id",
        "participant_identifier",
        "participant_key",
        "child_id",
        "child_identifier",
        "session_id",
        "session_identifier",
        "session_key",
        "episode_id",
        "episode_identifier",
        "episode_key",
        "video_id",
        "audio_id",
        "media_id",
        "speaker_id",
        "speaker_cluster_id",
        "row_id",
        "filename",
        "file_name",
        "relative_path",
        "absolute_path",
        "media_path",
        "frame_path",
        "raw_video",
        "raw_audio",
        "frame",
        "frames",
        "thumbnail",
        "raw_transcript",
        "transcript_text",
        "utterance_text",
        "lexical_string",
        "word_string",
        "speaker_embedding",
        "voiceprint",
        "instrument_embedding",
    }
)

RESTRICTED_TIME_KEYS = frozenset(
    {
        "timestamp",
        "exact_timestamp",
        "media_timestamp",
        "timecode",
        "start_time",
        "end_time",
        "onset_time",
        "offset_time",
        "utterance_start",
        "utterance_end",
        "frame_time",
    }
)

FORBIDDEN_SOURCE_PATTERNS = {
    "historical_adult_corpus": re.compile(r"(?i)(?<![a-z0-9])aea(?![a-z0-9])|aria\s+everyday\s+activities"),
    "other_child_corpus": re.compile(r"(?i)(?<![a-z0-9])babyview(?![a-z0-9])"),
}

ANCESTRY_CONTEXT = frozenset(
    {
        "ancestry",
        "ancestor",
        "source",
        "source_family",
        "data_source",
        "empirical",
        "calibration",
        "training",
        "learner",
        "tokenizer",
        "checkpoint",
        "weight",
        "model_artifact",
        "vocabulary",
        "result",
        "prior",
        "distribution",
    }
)

# These contexts document the boundary rather than supply ancestry.
EXEMPT_REFERENCE_CONTEXT = frozenset(
    {
        "forbidden",
        "prohibited",
        "deny",
        "denied",
        "excluded",
        "exclusion",
        "boundary",
        "method_reference",
        "methodological_reference",
        "citation",
        "public_paper",
        "source_url",
    }
)

MEDIA_FILENAME_RE = re.compile(
    r"(?i)(?<![a-z0-9_.-])[a-z0-9][a-z0-9_.-]{2,}\.(?:mp4|mov|mkv|avi|webm|wav|mp3|m4a|aac|flac|jpg|jpeg|png|webp|srt|vtt)\b"
)
MEDIA_CLOCK_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?!\d)")
AUDIT_DATETIME_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})?\b"
)
SUBTITLE_CUE_RE = re.compile(
    r"(?m)^\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?\s*-->\s*\d{1,2}:\d{2}:\d{2}"
)
DIALOGUE_RE = re.compile(r"(?im)^\s*(?:child|non_child|adult|caregiver|speaker[_ -]?\d+)\s*:\s+\S+")
DIRECT_ID_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(participant|child|session|episode|video|audio|media|speaker)[ _-]?(?:id|identifier|key)\s*[:=]\s*(.+?)\s*$"
)

SAFE_DIRECT_ID_VALUES = frozenset(
    {
        "none",
        "null",
        "absent",
        "unavailable",
        "unknown",
        "suppressed",
        "redacted",
        "forbidden",
        "prohibited",
        "not available",
        "not applicable",
        "not collected",
    }
)


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


def _normalise_key(key: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _machine_scalars(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _machine_scalars(child, path + (_normalise_key(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _machine_scalars(child, path + (f"[{index}]",))
    else:
        yield path, value


def _load_json(path: Path, root: Path, issues: list[Issue]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(
            Issue(
                "INVALID_JSON",
                _relative(path, root),
                f"JSON parse failed ({exc.__class__.__name__}); content was not echoed",
            )
        )
        return None


def _check_json_privacy(path: Path, root: Path, value: Any, issues: list[Issue]) -> None:
    for field_path, scalar in _machine_scalars(value):
        named = [item for item in field_path if not item.startswith("[")]
        leaf = named[-1] if named else ""
        display_path = ".".join(field_path) or "<root>"
        if leaf in RESTRICTED_JSON_KEYS:
            issues.append(
                Issue(
                    "RESTRICTED_MACHINE_FIELD",
                    _relative(path, root),
                    f"restricted row-level field at {display_path}; value suppressed",
                )
            )
        if leaf in RESTRICTED_TIME_KEYS:
            issues.append(
                Issue(
                    "EXACT_MEDIA_TIME_FIELD",
                    _relative(path, root),
                    f"exact media-timing field at {display_path}; value suppressed",
                )
            )

        context = set(named)
        joined_context = ".".join(named)
        scalar_text = scalar if isinstance(scalar, str) else ""
        matched_sources = [
            label
            for label, pattern in FORBIDDEN_SOURCE_PATTERNS.items()
            if pattern.search(joined_context) or (scalar_text and pattern.search(scalar_text))
        ]
        if not matched_sources:
            continue

        explicitly_false = scalar is False or (
            isinstance(scalar, str)
            and scalar.strip().lower()
            in {"false", "none", "absent", "excluded", "prohibited", "denied", "not_used", "not used"}
        )
        reference_only = any(
            any(marker in component for marker in EXEMPT_REFERENCE_CONTEXT) for component in named
        )
        ancestry_shaped = any(
            any(marker in component for marker in ANCESTRY_CONTEXT) for component in named
        )
        if explicitly_false or reference_only:
            continue
        if ancestry_shaped or scalar_text.strip().lower() in {"aea", "babyview", "aria everyday activities"}:
            issues.append(
                Issue(
                    "FORBIDDEN_EMPIRICAL_ANCESTRY",
                    _relative(path, root),
                    f"forbidden source marker in machine-readable ancestry field {display_path}; value suppressed",
                )
            )


def _check_report_text(path: Path, root: Path, text: str, issues: list[Issue]) -> None:
    rel = _relative(path, root)
    # Provenance/audit datetimes are reportable.  Remove only complete ISO-8601
    # datetimes before searching for media clock offsets.
    timing_scan_text = AUDIT_DATETIME_RE.sub("", text)
    if MEDIA_FILENAME_RE.search(text):
        issues.append(Issue("MEDIA_FILENAME_PATTERN", rel, "media filename-like payload detected; value suppressed"))
    if SUBTITLE_CUE_RE.search(timing_scan_text):
        issues.append(Issue("SUBTITLE_CUE_PATTERN", rel, "subtitle timing cue detected; value suppressed"))
    elif MEDIA_CLOCK_RE.search(timing_scan_text):
        issues.append(Issue("EXACT_MEDIA_CLOCK_PATTERN", rel, "clock-level media timestamp detected; value suppressed"))
    if DIALOGUE_RE.search(text):
        issues.append(Issue("DIALOGUE_PAYLOAD_PATTERN", rel, "speaker-labelled dialogue-like text detected; value suppressed"))
    for match in DIRECT_ID_RE.finditer(text):
        candidate = match.group(2).strip().strip("`\"'").lower()
        if candidate not in SAFE_DIRECT_ID_VALUES and not candidate.startswith(("none ", "not ", "suppressed ", "redacted ")):
            issues.append(
                Issue(
                    "DIRECT_IDENTIFIER_ASSIGNMENT",
                    rel,
                    f"direct {match.group(1).lower()} identifier assignment detected; value suppressed",
                )
            )


def _required_file_issues(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    docs = root / "docs" / "childlens_feasibility_v1"
    output = root / "output" / "childlens_feasibility_v1"
    for filename in DOCS_REQUIRED:
        path = docs / filename
        if not path.is_file():
            issues.append(Issue("MISSING_REQUIRED_FILE", _relative(path, root), "required document is missing"))
    for filename in OUTPUT_REQUIRED:
        path = output / filename
        if not path.is_file():
            issues.append(Issue("MISSING_REQUIRED_FILE", _relative(path, root), "required evidence artifact is missing"))
    return issues


def _report_files(root: Path) -> Iterable[Path]:
    for base in (root / "docs" / "childlens_feasibility_v1", root / "output" / "childlens_feasibility_v1"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() or path.is_symlink():
                yield path


def _scan_report_namespaces(root: Path, issues: list[Issue]) -> None:
    for path in _report_files(root):
        rel = _relative(path, root)
        if path.is_symlink():
            issues.append(Issue("SYMLINK_NOT_ALLOWED", rel, "report namespaces may not contain symlinks"))
            continue
        if path.suffix.lower() in PROHIBITED_PAYLOAD_EXTENSIONS:
            issues.append(Issue("PROHIBITED_PAYLOAD_EXTENSION", rel, "media/subtitle payload is not reportable"))
            continue
        try:
            if path.stat().st_size > 10 * 1024 * 1024:
                issues.append(Issue("OVERSIZE_REPORT_ARTIFACT", rel, "report artifact exceeds the 10 MiB safety limit"))
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            issues.append(Issue("UNREADABLE_REPORT_ARTIFACT", rel, f"text read failed ({exc.__class__.__name__})"))
            continue
        _check_report_text(path, root, text, issues)
        if path.suffix.lower() == ".json":
            value = _load_json(path, root, issues)
            if value is not None:
                _check_json_privacy(path, root, value, issues)


def _gate_rows(record: dict[str, Any]) -> Sequence[Any] | None:
    rows = record.get("gate_assessments")
    if rows is None:
        rows = record.get("gates")
    return rows if isinstance(rows, list) else None


def _bounded_correction(row: dict[str, Any]) -> str:
    for key in ("bounded_correction", "bounded_revision", "correction", "revision"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _check_decision(root: Path, issues: list[Issue]) -> None:
    docs = root / "docs" / "childlens_feasibility_v1"
    record_path = docs / "decision_record_v1.json"
    rubric_path = docs / "frozen_feasibility_rubric_v1.json"
    executive_path = docs / "executive_decision_report.md"
    if not record_path.is_file():
        issues.append(
            Issue(
                "DECISION_RECORD_ABSENT",
                _relative(record_path, root),
                "terminal validation requires the final decision record",
            )
        )
        return
    record = _load_json(record_path, root, issues)
    rubric = _load_json(rubric_path, root, issues) if rubric_path.is_file() else None
    if not isinstance(record, dict) or not isinstance(rubric, dict):
        return

    serialised = json.dumps(record, sort_keys=True)
    state_occurrences = [state for state in TERMINAL_STATES for _ in re.finditer(re.escape(state), serialised)]
    selected = record.get("terminal_decision")
    if len(state_occurrences) != 1 or selected not in TERMINAL_STATES:
        issues.append(
            Issue(
                "TERMINAL_STATE_CARDINALITY",
                _relative(record_path, root),
                "decision record must contain exactly one terminal-state literal in terminal_decision",
            )
        )
        selected = selected if selected in TERMINAL_STATES else None

    if record.get("scientific_outcome_authorized") is not False:
        issues.append(
            Issue(
                "OUTCOME_AUTHORIZATION_INVALID",
                _relative(record_path, root),
                "scientific_outcome_authorized must be exactly false",
            )
        )
    corpus = record.get("empirical_corpus")
    if not isinstance(corpus, str) or "CHILDLENS" not in corpus.upper():
        issues.append(
            Issue(
                "CORPUS_BOUNDARY_INVALID",
                _relative(record_path, root),
                "empirical_corpus must identify the ChildLens-only release boundary",
            )
        )

    expected_rubric_id = rubric.get("rubric_id")
    if record.get("rubric_id") != expected_rubric_id:
        issues.append(
            Issue(
                "RUBRIC_ID_MISMATCH",
                _relative(record_path, root),
                "decision record rubric_id does not match the frozen rubric",
            )
        )

    rubric_gates = rubric.get("gates")
    expected = {
        row.get("gate_id"): bool(row.get("essential"))
        for row in rubric_gates
        if isinstance(row, dict) and isinstance(row.get("gate_id"), str)
    } if isinstance(rubric_gates, list) else {}
    rows = _gate_rows(record)
    observed: dict[str, dict[str, Any]] = {}
    allowed_statuses = set(rubric.get("statuses", ()))
    if not expected or rows is None:
        issues.append(
            Issue(
                "GATE_INVENTORY_INVALID",
                _relative(record_path, root),
                "frozen rubric gates and decision gate_assessments must both be nonempty lists",
            )
        )
    else:
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("gate_id"), str):
                issues.append(Issue("GATE_ROW_INVALID", _relative(record_path, root), "gate row lacks a string gate_id"))
                continue
            gate_id = row["gate_id"]
            if gate_id in observed:
                issues.append(Issue("DUPLICATE_GATE", _relative(record_path, root), f"duplicate assessment for {gate_id}"))
                continue
            observed[gate_id] = row
            if row.get("status") not in allowed_statuses:
                issues.append(Issue("GATE_STATUS_INVALID", _relative(record_path, root), f"invalid status for {gate_id}"))
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        if missing:
            issues.append(Issue("MISSING_GATE_ASSESSMENT", _relative(record_path, root), "missing gate assessments: " + ", ".join(missing)))
        if extra:
            issues.append(Issue("EXTRA_GATE_ASSESSMENT", _relative(record_path, root), "unknown gate assessments: " + ", ".join(extra)))

    essential_rows = [observed[gate_id] for gate_id, essential in expected.items() if essential and gate_id in observed]
    essential_statuses = [str(row.get("status")) for row in essential_rows]
    if selected == "CHILDLENS_FEASIBILITY_GO" and (
        len(essential_rows) != sum(expected.values()) or any(status != "PASS" for status in essential_statuses)
    ):
        issues.append(Issue("GO_LOGIC_VIOLATION", _relative(record_path, root), "GO requires every essential gate to be PASS"))
    elif selected == "CHILDLENS_FEASIBILITY_REVISE":
        nonpass = [row for row in essential_rows if row.get("status") != "PASS"]
        if any(row.get("status") == "FAIL" for row in essential_rows) or not nonpass:
            issues.append(
                Issue(
                    "REVISE_LOGIC_VIOLATION",
                    _relative(record_path, root),
                    "REVISE requires at least one conditional/unresolved essential gate and no essential FAIL",
                )
            )
        for row in nonpass:
            if not _bounded_correction(row):
                issues.append(
                    Issue(
                        "REVISE_CORRECTION_MISSING",
                        _relative(record_path, root),
                        f"non-PASS gate {row.get('gate_id')} lacks a bounded correction",
                    )
                )
    elif selected == "CHILDLENS_FEASIBILITY_STOP" and not any(
        row.get("status") == "FAIL" for row in essential_rows
    ):
        issues.append(Issue("STOP_LOGIC_VIOLATION", _relative(record_path, root), "STOP requires at least one essential gate to be FAIL"))

    if executive_path.is_file() and selected:
        try:
            executive = executive_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return
        pattern = re.compile(
            r"(?im)^\s*(?:terminal\s+)?decision\s*:\s*`?(CHILDLENS_FEASIBILITY_(?:GO|REVISE|STOP))`?\s*$"
        )
        matches = pattern.findall(executive)
        if matches != [selected]:
            issues.append(
                Issue(
                    "EXECUTIVE_DECISION_MISMATCH",
                    _relative(executive_path, root),
                    "executive report must contain exactly one Terminal decision line matching the record",
                )
            )


def validate(root: Path) -> list[Issue]:
    root = root.resolve()
    issues = _required_file_issues(root)
    _scan_report_namespaces(root, issues)
    _check_decision(root, issues)
    return sorted(issues, key=lambda item: (item.path, item.code, item.message))


def _default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_default_root(), help="repository root")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable validation summary")
    args = parser.parse_args(argv)
    issues = validate(args.root)
    if args.json:
        print(
            json.dumps(
                {
                    "validator": "childlens-feasibility-v1",
                    "status": "PASS" if not issues else "FAIL",
                    "issue_count": len(issues),
                    "issues": [asdict(issue) for issue in issues],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif issues:
        print(f"FAIL: {len(issues)} issue(s)")
        for issue in issues:
            print(f"- {issue.code} {issue.path}: {issue.message}")
    else:
        print("PASS: ChildLens feasibility-v1 report artifacts satisfy structural, privacy-pattern, ancestry, and decision checks")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
