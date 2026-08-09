#!/usr/bin/env python3
"""Aggregate-only structural audit of quarantined ChildLens final annotations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


VERSION = "childlens-final-annotation-audit-v1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_BYTES = 4096


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _safe_file_token(value: str) -> str:
    return Path(value.replace("\\", "/")).name.casefold()


def _canonical_stem(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(value).stem.casefold())


def _read_participant_links(
    path: Path,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        reader = csv.reader(handle, dialect)
        header = next(reader)
        headers = [_norm(value) for value in header]
        file_candidates = [i for i, h in enumerate(headers) if re.search(r"file|video|media", h)]
        participant_candidates = [
            i for i, h in enumerate(headers) if re.search(r"participant|child|subject|anonym", h)
        ]
        date_candidates = [i for i, h in enumerate(headers) if re.search(r"recording.*date|date|session", h)]
        if len(file_candidates) != 1 or len(participant_candidates) != 1 or len(date_candidates) != 1:
            raise ValueError("E_PARTICIPANT_SCHEMA")
        links: dict[str, tuple[str, str]] = {}
        canonical_links: dict[str, tuple[str, str]] = {}
        for row in reader:
            if len(row) != len(headers):
                raise ValueError("E_PARTICIPANT_ROW")
            token = _safe_file_token(row[file_candidates[0]].strip())
            participant = row[participant_candidates[0]].strip()
            session = row[date_candidates[0]].strip()
            if not token or not participant or not session or token in links:
                raise ValueError("E_PARTICIPANT_LINK")
            links[token] = (participant, session)
            canonical = _canonical_stem(token)
            if not canonical or canonical in canonical_links:
                raise ValueError("E_PARTICIPANT_CANONICAL_LINK")
            canonical_links[canonical] = (participant, session)
        return links, canonical_links, len(headers)


def audit(annotation_root: Path, participant_table: Path) -> dict[str, object]:
    root = annotation_root.resolve(strict=True)
    participant = participant_table.resolve(strict=True)
    for path in (root, participant):
        if path == REPO_ROOT or REPO_ROOT in path.parents:
            raise ValueError("E_PATH_INSIDE_REPOSITORY")
    files = sorted(root.rglob("*.json"))
    if not files:
        raise ValueError("E_NO_ANNOTATIONS")
    participant_links, canonical_participant_links, participant_column_count = _read_participant_links(
        participant
    )

    parsed = 0
    video_link_complete = 0
    duration_complete = 0
    annotation_list_complete = 0
    participant_link_complete = 0
    participant_exact_link_complete = 0
    participant_canonical_link_complete = 0
    total_annotation_rows = 0
    timing_key_files = 0
    activity_key_files = 0
    location_key_files = 0
    speech_like_value_files = 0
    speaker_role_key_files = 0
    lexical_text_key_files = 0
    duration_string_count = 0
    duration_numeric_count = 0
    distinct_media_tokens: set[str] = set()
    distinct_participants: set[str] = set()
    distinct_sessions: set[tuple[str, str]] = set()

    for path in files:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError("E_JSON_PARSE") from exc
        if not isinstance(document, dict):
            raise ValueError("E_JSON_ROOT")
        parsed += 1
        media_value = document.get("video_name", document.get("videoName"))
        media_token = _safe_file_token(media_value) if isinstance(media_value, str) else ""
        if media_token:
            video_link_complete += 1
            if media_token in distinct_media_tokens:
                raise ValueError("E_DUPLICATE_MEDIA_LINK")
            distinct_media_tokens.add(media_token)
            grouping = participant_links.get(media_token)
            if grouping is not None:
                participant_exact_link_complete += 1
            else:
                grouping = canonical_participant_links.get(_canonical_stem(media_token))
                if grouping is not None:
                    participant_canonical_link_complete += 1
            if grouping is not None:
                participant_link_complete += 1
                distinct_participants.add(grouping[0])
                distinct_sessions.add(grouping)
        duration = document.get("duration")
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
            duration_complete += 1
            duration_numeric_count += 1
        elif isinstance(duration, str) and duration.strip():
            duration_string_count += 1
        annotations = document.get("annotations")
        if isinstance(annotations, list):
            annotation_list_complete += 1
            total_annotation_rows += len(annotations)
        else:
            annotations = []

        keys: set[str] = set()
        strings: list[str] = []
        stack: list[object] = [annotations]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    keys.add(_norm(str(key)))
                    stack.append(item)
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str):
                strings.append(value.casefold())
        if any(re.search(r"(^|_)(start|end|time|frame|onset|offset|duration)($|_)", key) for key in keys):
            timing_key_files += 1
        if any("activity" in key or "label" in key or "category" in key for key in keys):
            activity_key_files += 1
        if any("location" in key or "place" in key or "setting" in key for key in keys):
            location_key_files += 1
        if any(re.search(r"speech|talk|speak|vocal", value) for value in strings):
            speech_like_value_files += 1
        if any(re.search(r"speaker|voice.*type|speaker.*role", key) for key in keys):
            speaker_role_key_files += 1
        if any(re.search(r"transcript|utterance|word|lexical|sentence", key) for key in keys):
            lexical_text_key_files += 1

    receipt: dict[str, object] = {
        "schema_version": VERSION,
        "status": "ok",
        "annotation_file_count": len(files),
        "json_parse_complete_count": parsed,
        "unique_media_link_count": len(distinct_media_tokens),
        "media_link_complete_count": video_link_complete,
        "duration_complete_count": duration_complete,
        "annotation_list_complete_count": annotation_list_complete,
        "participant_table_row_count": len(participant_links),
        "participant_table_column_count": participant_column_count,
        "participant_media_link_complete_count": participant_link_complete,
        "participant_exact_link_complete_count": participant_exact_link_complete,
        "participant_canonical_link_complete_count": participant_canonical_link_complete,
        "linked_unique_participant_count": len(distinct_participants),
        "linked_unique_participant_session_count": len(distinct_sessions),
        "total_annotation_row_count": total_annotation_rows,
        "duration_numeric_count": duration_numeric_count,
        "duration_string_count": duration_string_count,
        "timing_key_file_count": timing_key_files,
        "activity_or_label_key_file_count": activity_key_files,
        "location_key_file_count": location_key_files,
        "speech_like_value_file_count": speech_like_value_files,
        "speaker_role_key_file_count": speaker_role_key_files,
        "lexical_text_key_file_count": lexical_text_key_files,
        "raw_values_returned": False,
        "filenames_returned": False,
        "identifiers_returned": False,
        "exact_timestamps_returned": False,
        "annotation_rows_returned": False,
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("E_OUTPUT_SIZE")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--participant-table", required=True)
    args = parser.parse_args()
    try:
        receipt = audit(Path(args.annotations), Path(args.participant_table))
    except Exception as exc:
        code = str(exc) if str(exc).startswith("E_") else "E_INTERNAL"
        print(json.dumps({"schema_version": VERSION, "status": "error", "error_code": code}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
