#!/usr/bin/env python3
"""Emit an allowlisted, aggregate-only receipt for the restricted participant CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path


VERSION = "childlens-participant-table-audit-v1.1.0"
MAX_OUTPUT_BYTES = 4096
REPO_ROOT = Path(__file__).resolve().parents[1]


def _category(headers: list[str], pattern: str) -> int | None:
    regex = re.compile(pattern)
    matches = [index for index, value in enumerate(headers) if regex.search(value)]
    return matches[0] if len(matches) == 1 else None


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def audit(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("E_PATH_INSIDE_REPOSITORY")
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError("E_PATH_INVALID")

    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error as exc:
            raise ValueError("E_DELIMITER") from exc
        reader = csv.reader(handle, dialect)
        header_raw = next(reader, None)
        if header_raw is None or not header_raw:
            raise ValueError("E_HEADER_MISSING")
        headers = [_normalise(value) for value in header_raw]
        if any(not value for value in headers) or len(set(headers)) != len(headers):
            raise ValueError("E_HEADER_INVALID")

        participant_index = _category(headers, r"(^|_)(participant|child|subject)(_|$)|anonym")
        date_index = _category(headers, r"(^|_)(recording_?date|date|session|visit)(_|$)")
        language_index = _category(headers, r"(^|_)(language|lang|native_?language)(_|$)")
        age_index = _category(headers, r"(^|_)(age|age_?years)(_|$)")

        rows = 0
        complete_participant = 0
        complete_date = 0
        complete_grouping = 0
        nonempty_language = 0
        participant_values: set[str] = set()
        session_values: set[tuple[str, str]] = set()
        language_values: set[str] = set()
        for row in reader:
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != len(headers):
                raise ValueError("E_ROW_WIDTH")
            rows += 1
            participant = row[participant_index].strip() if participant_index is not None else ""
            date_value = row[date_index].strip() if date_index is not None else ""
            if participant:
                complete_participant += 1
                participant_values.add(participant)
            if date_value:
                complete_date += 1
            if participant and date_value:
                complete_grouping += 1
                session_values.add((participant, date_value))
            if language_index is not None:
                language = row[language_index].strip()
                if language:
                    nonempty_language += 1
                    language_values.add(language.casefold())

    receipt: dict[str, object] = {
        "schema_version": VERSION,
        "status": "ok",
        "row_count": rows,
        "column_count": len(headers),
        "participant_grouping_field_present": participant_index is not None,
        "recording_or_session_field_present": date_index is not None,
        "language_field_present": language_index is not None,
        "age_field_present": age_index is not None,
        "participant_complete_row_count": complete_participant,
        "recording_or_session_complete_row_count": complete_date,
        "participant_session_complete_row_count": complete_grouping,
        "unique_participant_count": len(participant_values),
        "unique_participant_session_count": len(session_values),
        "language_nonempty_row_count": nonempty_language,
        "language_distinct_value_count": len(language_values),
        "cell_suppression_threshold": 5,
        "raw_values_returned": False,
        "headers_returned": False,
        "identifiers_returned": False,
        "exact_dates_returned": False,
        "language_labels_returned": False,
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("E_OUTPUT_SIZE")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    try:
        receipt = audit(Path(args.input))
    except Exception as exc:
        code = str(exc)
        if not code.startswith("E_"):
            code = "E_INTERNAL"
        print(json.dumps({"schema_version": VERSION, "status": "error", "error_code": code}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
