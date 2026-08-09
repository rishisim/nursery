#!/usr/bin/env python3
"""Build quarantined ChildLens manifest input from official metadata annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from collections import defaultdict
from pathlib import Path


VERSION = "childlens-restricted-input-preparer-v1.1.0"
INPUT_SCHEMA = "childlens-restricted-manifest-input-v1.1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _basename(value: str) -> str:
    return Path(value.replace("\\", "/")).name


def _stem(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(_basename(value)).stem.casefold())


def _opaque(secret: bytes, kind: str, value: str) -> str:
    return hmac.new(secret, f"{kind}\0{value}".encode(), hashlib.sha256).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path == REPO_ROOT or REPO_ROOT in path.resolve().parents:
        raise ValueError("E_OUTPUT_INSIDE_REPOSITORY")
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _load_or_create_secret(path: Path) -> bytes:
    if path.exists():
        secret = path.read_bytes()
        if len(secret) != 32:
            raise ValueError("E_SECRET_SIZE")
        return secret
    if path == REPO_ROOT or REPO_ROOT in path.resolve().parents:
        raise ValueError("E_SECRET_INSIDE_REPOSITORY")
    secret = secrets.token_bytes(32)
    _write_atomic(path, secret)
    return secret


def _participant_links(path: Path) -> dict[str, tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        reader = csv.reader(handle, dialect)
        header = next(reader)
        headers = [_norm(value) for value in header]
        candidates = {
            "file": [i for i, h in enumerate(headers) if re.search(r"file|video|media", h)],
            "participant": [
                i for i, h in enumerate(headers) if re.search(r"participant|child|subject|anonym", h)
            ],
            "session": [i for i, h in enumerate(headers) if re.search(r"recording.*date|date|session", h)],
        }
        if any(len(value) != 1 for value in candidates.values()):
            raise ValueError("E_PARTICIPANT_SCHEMA")
        links: dict[str, tuple[str, str]] = {}
        for row in reader:
            if len(row) != len(headers):
                raise ValueError("E_PARTICIPANT_ROW")
            key = _stem(row[candidates["file"][0]])
            grouping = (
                row[candidates["participant"][0]].strip(),
                row[candidates["session"][0]].strip(),
            )
            if not key or not all(grouping) or key in links:
                raise ValueError("E_PARTICIPANT_LINK")
            links[key] = grouping
        return links


def _number(mapping: dict[str, object], patterns: tuple[str, ...]) -> float | None:
    for key, value in mapping.items():
        normalized = _norm(key)
        if normalized in patterns and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _metadata(document: dict[str, object]) -> tuple[str, int, str, str, str | None]:
    media_value = document.get("video_name", document.get("videoName"))
    annotations = document.get("annotations")
    if not isinstance(media_value, str) or not isinstance(annotations, list):
        raise ValueError("E_ANNOTATION_SCHEMA")
    activity_duration: dict[str, float] = defaultdict(float)
    location_duration: dict[str, float] = defaultdict(float)
    speech_present = False
    maximum_end = 0.0
    for raw in annotations:
        if not isinstance(raw, dict):
            raise ValueError("E_ANNOTATION_ROW")
        event = raw.get("eventId")
        if not isinstance(event, str) or not event.strip():
            raise ValueError("E_EVENT_ID")
        duration = _number(raw, ("duration",)) or 0.0
        start = _number(raw, ("start", "start_time", "starttime", "onset"))
        end = _number(raw, ("end", "end_time", "endtime", "offset"))
        if end is not None:
            maximum_end = max(maximum_end, end)
        elif start is not None and duration > 0:
            maximum_end = max(maximum_end, start + duration)
        normalized_event = _norm(event)
        if re.search(r"speech|talk|speak|vocal", normalized_event):
            speech_present = True
        if normalized_event == "location":
            fields = raw.get("fields")
            if isinstance(fields, dict):
                location = fields.get("Type of Location")
                if isinstance(location, str) and location.strip():
                    location_duration[location.strip()] += max(duration, 0.0)
        elif normalized_event != "exclude":
            activity_duration[event.strip()] += max(duration, 0.0)
    if maximum_end <= 0:
        raise ValueError("E_DURATION_PROXY")
    dominant_activity = (
        sorted(activity_duration.items(), key=lambda item: (-item[1], item[0].casefold()))[0][0]
        if activity_duration
        else "UNKNOWN"
    )
    dominant_location = (
        sorted(location_duration.items(), key=lambda item: (-item[1], item[0].casefold()))[0][0]
        if location_duration
        else None
    )
    return (
        _basename(media_value),
        max(1, round(maximum_end * 1000)),
        dominant_activity,
        "PRESENT" if speech_present else "ABSENT",
        dominant_location,
    )


def prepare(
    annotations_root: Path,
    participant_table: Path,
    secret_path: Path,
    output_path: Path,
    observation_receipt_sha256: str,
) -> dict[str, object]:
    root = annotations_root.resolve(strict=True)
    participant_path = participant_table.resolve(strict=True)
    for path in (root, participant_path):
        if path == REPO_ROOT or REPO_ROOT in path.parents:
            raise ValueError("E_INPUT_INSIDE_REPOSITORY")
    if not re.fullmatch(r"[0-9a-f]{64}", observation_receipt_sha256):
        raise ValueError("E_OBSERVATION_DIGEST")
    secret = _load_or_create_secret(secret_path)
    links = _participant_links(participant_path)
    annotation_files = sorted(root.rglob("*.json"))
    if not annotation_files:
        raise ValueError("E_NO_ANNOTATIONS")

    objects: list[dict[str, object]] = []
    annotation_rows: list[dict[str, object]] = []
    media_rows: list[dict[str, object]] = []
    participant_table_object_key = _opaque(secret, "object", str(participant_path))
    objects.append({
        "object_key": participant_table_object_key,
        "source_locator": str(participant_path),
        "top_level_class": "PARTICIPANT_TABLE",
        "size_bytes": participant_path.stat().st_size,
        "source_checksum_sha256": None,
        "local_sha256": _sha_file(participant_path),
        "remote_version": None,
        "remote_etag": None,
        "available_for_selective_copy": True,
        "container_parseable": True,
    })
    seen_media: set[str] = set()
    for annotation_path in annotation_files:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("E_JSON_ROOT")
        media_name, duration_ms, activity, speech, location = _metadata(document)
        canonical = _stem(media_name)
        grouping = links.get(canonical)
        if grouping is None or canonical in seen_media:
            raise ValueError("E_MEDIA_GROUPING")
        seen_media.add(canonical)
        media_key = _opaque(secret, "media", canonical)
        video_object_key = _opaque(secret, "object-video", canonical)
        annotation_object_key = _opaque(secret, "object-annotation", str(annotation_path))
        annotation_key = _opaque(secret, "annotation", str(annotation_path))
        participant_key = _opaque(secret, "participant", grouping[0])
        session_key = _opaque(secret, "session", f"{grouping[0]}\0{grouping[1]}")
        objects.extend([
            {
                "object_key": video_object_key,
                "source_locator": f"/ChildLens/videos/{media_name}",
                "top_level_class": "VIDEO",
                "size_bytes": 1,
                "source_checksum_sha256": None,
                "local_sha256": None,
                "remote_version": None,
                "remote_etag": None,
                "available_for_selective_copy": True,
                "container_parseable": None,
            },
            {
                "object_key": annotation_object_key,
                "source_locator": str(annotation_path),
                "top_level_class": "ANNOTATION",
                "size_bytes": annotation_path.stat().st_size,
                "source_checksum_sha256": None,
                "local_sha256": _sha_file(annotation_path),
                "remote_version": None,
                "remote_etag": None,
                "available_for_selective_copy": True,
                "container_parseable": True,
            },
        ])
        annotation_rows.append({
            "annotation_key": annotation_key,
            "object_key": annotation_object_key,
            "linked_media_key": media_key,
            "representation_kind": "FINAL",
        })
        media_rows.append({
            "media_key": media_key,
            "object_key": video_object_key,
            "participant_key": participant_key,
            "session_key": session_key,
            "duration_milliseconds": duration_ms,
            "coarse_activity_label": activity,
            "speech_presence_bin": speech,
            "location_label": location,
        })
    if len(media_rows) != len(links):
        raise ValueError("E_INCOMPLETE_MEDIA_LINKAGE")

    key_receipt = _sha_bytes(_canonical({
        "algorithm": "HMAC-SHA-256",
        "domain_separation": True,
        "secret_bytes": 32,
        "secret_exported": False,
    }))
    document = {
        "schema_version": INPUT_SCHEMA,
        "release": {
            "doi": "10.17617/4.fe",
            "keeper_library_id": "c8ed0104-b793-4c35-817e-302afd4e036b",
            "source_role": "LIVE_REVISION",
            "public_doi_snapshot_commit": "4856662653b2fa183e53268d088cffba02a33443",
            "observed_revision_commit": None,
            "observation_date": "2026-07-21",
            "observation_receipt_sha256": observation_receipt_sha256,
            "doi_snapshot_object_inventory_sha256": None,
            "doi_snapshot_inventory_receipt_sha256": None,
            "session_definition": "PARTICIPANT_PLUS_RECORDING_DATE",
        },
        "objects": objects,
        "annotations": annotation_rows,
        "media": media_rows,
        "attestations": {
            "internal_keys_are_hmac_sha256": True,
            "key_derivation_receipt_sha256": key_receipt,
            "metadata_frozen_before_content_inspection": True,
            "no_content_fields_used": True,
        },
    }
    _write_atomic(output_path, _canonical(document) + b"\n")
    return {
        "schema_version": VERSION,
        "status": "ok",
        "media_count": len(media_rows),
        "annotation_count": len(annotation_rows),
        "participant_group_count": len({row["participant_key"] for row in media_rows}),
        "session_group_count": len({row["session_key"] for row in media_rows}),
        "input_sha256": _sha_bytes(_canonical(document)),
        "video_size_values_are_preflight_placeholders": True,
        "raw_values_returned": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--participant-table", required=True)
    parser.add_argument("--secret", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation-receipt-sha256", required=True)
    args = parser.parse_args()
    try:
        receipt = prepare(
            Path(args.annotations),
            Path(args.participant_table),
            Path(args.secret),
            Path(args.output),
            args.observation_receipt_sha256,
        )
    except Exception as exc:
        code = str(exc) if str(exc).startswith("E_") else "E_INTERNAL"
        print(json.dumps({"schema_version": VERSION, "status": "error", "error_code": code}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
