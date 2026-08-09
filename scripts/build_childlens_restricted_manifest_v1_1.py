#!/usr/bin/env python3
"""Build a quarantined ChildLens manifest and an aggregate-only pilot receipt.

The input and restricted output are operational artifacts and may contain
filenames and opaque release-local keys.  They must remain outside the
repository.  The reportable output is constructed from a fixed allowlist and
never contains row-level keys, source locators, per-object checksums, or
per-object metadata.

This program performs no network access, media acquisition, decoding, or
scientific learner work.  Diagnostics are constant error codes so an invalid
restricted value is never echoed to a terminal or tool response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL_PATH = REPO_ROOT / "docs" / "childlens_feasibility_v1" / "frozen_pilot_protocol_v1.json"

INPUT_SCHEMA = "childlens-restricted-manifest-input-v1.1.0"
RESTRICTED_SCHEMA = "childlens-restricted-canonical-manifest-v1.1.0"
REPORTABLE_SCHEMA = "childlens-manifest-selection-aggregate-receipt-v1.1.0"
FROZEN_PROTOCOL_ID = "childlens-lexical-feasibility-pilot-v1.0.0"
FROZEN_TARGET = 15
FROZEN_MINIMUM = 12
FROZEN_MAXIMUM = 18
CELL_SUPPRESSION_THRESHOLD = 5

GIB = 1024**3
HARD_RAW_CAP_BYTES = 20 * GIB
HARD_NAMESPACE_CAP_BYTES = 73 * GIB
HARD_FREE_SPACE_FLOOR_BYTES = 50 * GIB

TOP_LEVEL_CLASSES = (
    "VIDEO",
    "ANNOTATION",
    "IMAGE",
    "METADATA",
    "PARTICIPANT_TABLE",
    "README_TERMS",
    "CERTIFICATE",
    "OTHER",
)
SOURCE_ROLES = frozenset({"DOI_SNAPSHOT", "LIVE_REVISION"})
SPEECH_BINS = frozenset({"PRESENT", "ABSENT", "MIXED", "UNKNOWN"})
ANNOTATION_REPRESENTATIONS = frozenset({"FINAL", "PLATFORM", "OTHER"})

HEX_64 = re.compile(r"^[0-9a-f]{64}$")
HEX_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TOP_KEYS = frozenset({"schema_version", "release", "objects", "annotations", "media", "attestations"})
RELEASE_KEYS = frozenset(
    {
        "doi",
        "keeper_library_id",
        "source_role",
        "public_doi_snapshot_commit",
        "observed_revision_commit",
        "observation_date",
        "observation_receipt_sha256",
        "doi_snapshot_object_inventory_sha256",
        "doi_snapshot_inventory_receipt_sha256",
        "session_definition",
    }
)
ATTESTATION_KEYS = frozenset(
    {
        "internal_keys_are_hmac_sha256",
        "key_derivation_receipt_sha256",
        "metadata_frozen_before_content_inspection",
        "no_content_fields_used",
    }
)
OBJECT_KEYS = frozenset(
    {
        "object_key",
        "source_locator",
        "top_level_class",
        "size_bytes",
        "source_checksum_sha256",
        "local_sha256",
        "remote_version",
        "remote_etag",
        "available_for_selective_copy",
        "container_parseable",
    }
)
ANNOTATION_KEYS = frozenset(
    {"annotation_key", "object_key", "linked_media_key", "representation_kind"}
)
MEDIA_KEYS = frozenset(
    {
        "media_key",
        "object_key",
        "participant_key",
        "session_key",
        "duration_milliseconds",
        "coarse_activity_label",
        "speech_presence_bin",
        "location_label",
    }
)


class ManifestError(RuntimeError):
    """A fail-closed validation error whose message is a constant code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ResourcePreflight:
    raw_cap_bytes: int
    projected_namespace_peak_bytes: int
    volume_free_before_bytes: int


def _fail(code: str) -> None:
    raise ManifestError(code)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _selection_hash(release_digest: str, internal_episode_key: str) -> str:
    material = f"{release_digest}{FROZEN_PROTOCOL_ID}{internal_episode_key}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], allowed: frozenset[str], code: str) -> None:
    if set(value) != allowed:
        _fail(code)


def _require_dict(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _require_list(value: Any, code: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(code)
    return value


def _require_bool(value: Any, code: str) -> bool:
    if not isinstance(value, bool):
        _fail(code)
    return value


def _require_nonempty_string(value: Any, code: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        _fail(code)
    return value


def _require_optional_string(value: Any, code: str, *, maximum: int = 512) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, code, maximum=maximum)


def _require_opaque_key(value: Any, code: str) -> str:
    if not isinstance(value, str) or not HEX_64.fullmatch(value):
        _fail(code)
    return value


def _require_optional_digest(value: Any, code: str) -> str | None:
    if value is None:
        return None
    return _require_opaque_key(value, code)


def _require_nonnegative_integer(value: Any, code: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(code)
    return value


def _normalise_input(document: Any) -> dict[str, Any]:
    root = _require_dict(document, "INPUT_ROOT_NOT_OBJECT")
    _require_exact_keys(root, TOP_KEYS, "INPUT_TOP_LEVEL_FIELDS_INVALID")
    if root["schema_version"] != INPUT_SCHEMA:
        _fail("INPUT_SCHEMA_VERSION_INVALID")

    release = _require_dict(root["release"], "RELEASE_NOT_OBJECT")
    _require_exact_keys(release, RELEASE_KEYS, "RELEASE_FIELDS_INVALID")
    doi = _require_nonempty_string(release["doi"], "RELEASE_DOI_INVALID", maximum=128)
    if doi != "10.17617/4.fe":
        _fail("RELEASE_DOI_UNEXPECTED")
    library_id = _require_nonempty_string(
        release["keeper_library_id"], "KEEPER_LIBRARY_ID_INVALID", maximum=128
    )
    source_role = release["source_role"]
    if source_role not in SOURCE_ROLES:
        _fail("SOURCE_ROLE_INVALID")
    public_commit = _require_nonempty_string(
        release["public_doi_snapshot_commit"], "PUBLIC_SNAPSHOT_COMMIT_INVALID", maximum=64
    )
    if not HEX_REVISION.fullmatch(public_commit):
        _fail("PUBLIC_SNAPSHOT_COMMIT_INVALID")
    observed_commit = _require_optional_string(
        release["observed_revision_commit"], "OBSERVED_REVISION_COMMIT_INVALID", maximum=64
    )
    if observed_commit is not None and not HEX_REVISION.fullmatch(observed_commit):
        _fail("OBSERVED_REVISION_COMMIT_INVALID")
    observation_date = _require_nonempty_string(
        release["observation_date"], "OBSERVATION_DATE_INVALID", maximum=10
    )
    if not DATE.fullmatch(observation_date):
        _fail("OBSERVATION_DATE_INVALID")
    observation_receipt = _require_opaque_key(
        release["observation_receipt_sha256"], "OBSERVATION_RECEIPT_DIGEST_INVALID"
    )
    comparison_digest = _require_optional_digest(
        release["doi_snapshot_object_inventory_sha256"],
        "DOI_SNAPSHOT_OBJECT_INVENTORY_DIGEST_INVALID",
    )
    comparison_receipt = _require_optional_digest(
        release["doi_snapshot_inventory_receipt_sha256"],
        "DOI_SNAPSHOT_INVENTORY_RECEIPT_DIGEST_INVALID",
    )
    if (comparison_digest is None) != (comparison_receipt is None):
        _fail("DOI_SNAPSHOT_COMPARISON_RECEIPT_INCOMPLETE")
    if release["session_definition"] != "PARTICIPANT_PLUS_RECORDING_DATE":
        _fail("SESSION_DEFINITION_INVALID")
    if source_role == "DOI_SNAPSHOT" and observed_commit != public_commit:
        _fail("DOI_SNAPSHOT_COMMIT_BINDING_INVALID")

    attestations = _require_dict(root["attestations"], "ATTESTATIONS_NOT_OBJECT")
    _require_exact_keys(attestations, ATTESTATION_KEYS, "ATTESTATION_FIELDS_INVALID")
    for key in (
        "internal_keys_are_hmac_sha256",
        "metadata_frozen_before_content_inspection",
        "no_content_fields_used",
    ):
        if _require_bool(attestations[key], "ATTESTATION_VALUE_INVALID") is not True:
            _fail("REQUIRED_ATTESTATION_FALSE")
    key_receipt = _require_opaque_key(
        attestations["key_derivation_receipt_sha256"], "KEY_DERIVATION_RECEIPT_INVALID"
    )

    objects: list[dict[str, Any]] = []
    object_keys: set[str] = set()
    source_locators: set[str] = set()
    for raw in _require_list(root["objects"], "OBJECTS_NOT_LIST"):
        row = _require_dict(raw, "OBJECT_ROW_NOT_OBJECT")
        _require_exact_keys(row, OBJECT_KEYS, "OBJECT_FIELDS_INVALID")
        object_key = _require_opaque_key(row["object_key"], "OBJECT_KEY_INVALID")
        if object_key in object_keys:
            _fail("DUPLICATE_OBJECT_KEY")
        object_keys.add(object_key)
        source_locator = _require_nonempty_string(
            row["source_locator"], "SOURCE_LOCATOR_INVALID", maximum=4096
        )
        if source_locator in source_locators:
            _fail("DUPLICATE_SOURCE_LOCATOR")
        source_locators.add(source_locator)
        top_class = row["top_level_class"]
        if top_class not in TOP_LEVEL_CLASSES:
            _fail("TOP_LEVEL_CLASS_INVALID")
        size_bytes = _require_nonnegative_integer(
            row["size_bytes"], "OBJECT_SIZE_INVALID", allow_none=True
        )
        source_checksum = _require_optional_digest(
            row["source_checksum_sha256"], "SOURCE_CHECKSUM_INVALID"
        )
        local_sha256 = _require_optional_digest(row["local_sha256"], "LOCAL_CHECKSUM_INVALID")
        remote_version = _require_optional_string(
            row["remote_version"], "REMOTE_VERSION_INVALID", maximum=512
        )
        remote_etag = _require_optional_string(row["remote_etag"], "REMOTE_ETAG_INVALID", maximum=512)
        available = _require_bool(row["available_for_selective_copy"], "OBJECT_AVAILABILITY_INVALID")
        parseable = row["container_parseable"]
        if parseable is not None:
            parseable = _require_bool(parseable, "CONTAINER_PARSEABLE_INVALID")
        objects.append(
            {
                "object_key": object_key,
                "source_locator": source_locator,
                "top_level_class": top_class,
                "size_bytes": size_bytes,
                "source_checksum_sha256": source_checksum,
                "local_sha256": local_sha256,
                "remote_version": remote_version,
                "remote_etag": remote_etag,
                "available_for_selective_copy": available,
                "container_parseable": parseable,
            }
        )
    if not objects:
        _fail("OBJECT_INVENTORY_EMPTY")

    annotations: list[dict[str, Any]] = []
    annotation_keys: set[str] = set()
    annotation_object_keys: set[str] = set()
    for raw in _require_list(root["annotations"], "ANNOTATIONS_NOT_LIST"):
        row = _require_dict(raw, "ANNOTATION_ROW_NOT_OBJECT")
        _require_exact_keys(row, ANNOTATION_KEYS, "ANNOTATION_FIELDS_INVALID")
        annotation_key = _require_opaque_key(row["annotation_key"], "ANNOTATION_KEY_INVALID")
        if annotation_key in annotation_keys:
            _fail("DUPLICATE_ANNOTATION_KEY")
        annotation_keys.add(annotation_key)
        object_key = _require_opaque_key(row["object_key"], "ANNOTATION_OBJECT_KEY_INVALID")
        if object_key in annotation_object_keys:
            _fail("DUPLICATE_ANNOTATION_OBJECT_LINK")
        annotation_object_keys.add(object_key)
        linked_media_key = row["linked_media_key"]
        if linked_media_key is not None:
            linked_media_key = _require_opaque_key(linked_media_key, "LINKED_MEDIA_KEY_INVALID")
        representation = row["representation_kind"]
        if representation not in ANNOTATION_REPRESENTATIONS:
            _fail("ANNOTATION_REPRESENTATION_INVALID")
        annotations.append(
            {
                "annotation_key": annotation_key,
                "object_key": object_key,
                "linked_media_key": linked_media_key,
                "representation_kind": representation,
            }
        )

    media: list[dict[str, Any]] = []
    media_keys: set[str] = set()
    media_object_keys: set[str] = set()
    for raw in _require_list(root["media"], "MEDIA_NOT_LIST"):
        row = _require_dict(raw, "MEDIA_ROW_NOT_OBJECT")
        _require_exact_keys(row, MEDIA_KEYS, "MEDIA_FIELDS_INVALID")
        media_key = _require_opaque_key(row["media_key"], "MEDIA_KEY_INVALID")
        if media_key in media_keys:
            _fail("DUPLICATE_MEDIA_KEY")
        media_keys.add(media_key)
        object_key = _require_opaque_key(row["object_key"], "MEDIA_OBJECT_KEY_INVALID")
        if object_key in media_object_keys:
            _fail("DUPLICATE_MEDIA_OBJECT_LINK")
        media_object_keys.add(object_key)
        participant_key = row["participant_key"]
        if participant_key is not None:
            participant_key = _require_opaque_key(participant_key, "PARTICIPANT_KEY_INVALID")
        session_key = row["session_key"]
        if session_key is not None:
            session_key = _require_opaque_key(session_key, "SESSION_KEY_INVALID")
        duration_ms = _require_nonnegative_integer(
            row["duration_milliseconds"], "MEDIA_DURATION_INVALID", allow_none=True
        )
        if duration_ms == 0:
            _fail("MEDIA_DURATION_INVALID")
        activity = _require_nonempty_string(
            row["coarse_activity_label"], "ACTIVITY_LABEL_INVALID", maximum=256
        )
        speech = row["speech_presence_bin"]
        if speech not in SPEECH_BINS:
            _fail("SPEECH_BIN_INVALID")
        location = _require_optional_string(row["location_label"], "LOCATION_LABEL_INVALID", maximum=256)
        media.append(
            {
                "media_key": media_key,
                "object_key": object_key,
                "participant_key": participant_key,
                "session_key": session_key,
                "duration_milliseconds": duration_ms,
                "coarse_activity_label": activity,
                "speech_presence_bin": speech,
                "location_label": location,
            }
        )

    object_by_key = {row["object_key"]: row for row in objects}
    media_by_key = {row["media_key"]: row for row in media}
    for row in media:
        obj = object_by_key.get(row["object_key"])
        if obj is None or obj["top_level_class"] != "VIDEO":
            _fail("MEDIA_OBJECT_LINK_INVALID")
    for row in annotations:
        obj = object_by_key.get(row["object_key"])
        if obj is None or obj["top_level_class"] != "ANNOTATION":
            _fail("ANNOTATION_OBJECT_LINK_INVALID")
        linked = row["linked_media_key"]
        if linked is not None and linked not in media_by_key:
            _fail("ANNOTATION_MEDIA_LINK_INVALID")

    session_to_participant: dict[str, str] = {}
    for row in media:
        participant = row["participant_key"]
        session = row["session_key"]
        if (participant is None) != (session is None):
            _fail("PARTIAL_GROUPING_LINK")
        if session is not None:
            prior = session_to_participant.setdefault(session, participant)
            if prior != participant:
                _fail("SESSION_CROSSES_PARTICIPANTS")

    return {
        "schema_version": INPUT_SCHEMA,
        "release": {
            "doi": doi,
            "keeper_library_id": library_id,
            "source_role": source_role,
            "public_doi_snapshot_commit": public_commit,
            "observed_revision_commit": observed_commit,
            "observation_date": observation_date,
            "observation_receipt_sha256": observation_receipt,
            "doi_snapshot_object_inventory_sha256": comparison_digest,
            "doi_snapshot_inventory_receipt_sha256": comparison_receipt,
            "session_definition": release["session_definition"],
        },
        "attestations": {
            "internal_keys_are_hmac_sha256": True,
            "key_derivation_receipt_sha256": key_receipt,
            "metadata_frozen_before_content_inspection": True,
            "no_content_fields_used": True,
        },
        "objects": sorted(objects, key=lambda row: row["object_key"]),
        "annotations": sorted(annotations, key=lambda row: row["annotation_key"]),
        "media": sorted(media, key=lambda row: row["media_key"]),
    }


def _release_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical release-only payload used for the selection digest."""

    release = dict(document["release"])
    # A snapshot comparison digest is an assertion about this manifest and
    # therefore cannot participate in the digest being compared.
    release.pop("doi_snapshot_object_inventory_sha256")
    release.pop("doi_snapshot_inventory_receipt_sha256")
    return {
        "manifest_schema_version": RESTRICTED_SCHEMA,
        "release": release,
        "attestations": document["attestations"],
        "records": {
            "objects": document["objects"],
            "linkages": document["annotations"],
            "groupings": [
                {
                    "media_key": row["media_key"],
                    "participant_key": row["participant_key"],
                    "session_key": row["session_key"],
                }
                for row in document["media"]
            ],
            "selection_metadata": [
                {
                    "media_key": row["media_key"],
                    "object_key": row["object_key"],
                    "duration_milliseconds": row["duration_milliseconds"],
                    "coarse_activity_label": row["coarse_activity_label"],
                    "speech_presence_bin": row["speech_presence_bin"],
                    "location_label": row["location_label"],
                }
                for row in document["media"]
            ],
        },
    }


def _load_and_validate_frozen_protocol(path: Path = FROZEN_PROTOCOL_PATH) -> str:
    try:
        raw = path.read_bytes()
        protocol = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeError):
        _fail("FROZEN_PROTOCOL_UNREADABLE")
    expected_selection = (
        "Deterministic balanced allocation over metadata strata, then hash-order sampling using "
        "SHA-256(release_manifest_digest || protocol_id || internal_episode_key)."
    )
    try:
        valid = (
            protocol["protocol_id"] == FROZEN_PROTOCOL_ID
            and protocol["target_sample"]["videos"] == FROZEN_TARGET
            and protocol["target_sample"]["allowable_range"] == [FROZEN_MINIMUM, FROZEN_MAXIMUM]
            and protocol["target_sample"]["selection"] == expected_selection
            and protocol["acquisition"]["full_archive_download"] is False
            and protocol["stopping"]["no_training"].startswith("Do not train")
        )
    except (KeyError, TypeError, AttributeError):
        valid = False
    if not valid:
        _fail("FROZEN_PROTOCOL_MISMATCH")
    return hashlib.sha256(raw).hexdigest()


def _validate_resources(preflight: ResourcePreflight) -> None:
    values = (
        preflight.raw_cap_bytes,
        preflight.projected_namespace_peak_bytes,
        preflight.volume_free_before_bytes,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        _fail("RESOURCE_PREFLIGHT_VALUE_INVALID")
    if preflight.raw_cap_bytes > HARD_RAW_CAP_BYTES:
        _fail("RAW_CAP_EXCEEDS_FROZEN_LIMIT")
    if preflight.projected_namespace_peak_bytes > HARD_NAMESPACE_CAP_BYTES:
        _fail("NAMESPACE_PEAK_EXCEEDS_FROZEN_LIMIT")
    if preflight.volume_free_before_bytes - preflight.projected_namespace_peak_bytes < HARD_FREE_SPACE_FLOOR_BYTES:
        _fail("POST_PEAK_FREE_SPACE_BELOW_FLOOR")


def _duration_tertiles(media: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ordered = sorted(media, key=lambda row: (row["duration_milliseconds"], row["media_key"]))
    labels = ("LOW", "MIDDLE", "HIGH")
    total = len(ordered)
    return {
        row["media_key"]: labels[min(2, (index * 3) // total)]
        for index, row in enumerate(ordered)
    }


def _eligible_media(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    object_by_key = {row["object_key"]: row for row in document["objects"]}
    linked_media = {
        row["linked_media_key"]
        for row in document["annotations"]
        if row["linked_media_key"] is not None
    }
    annotated_rows = [row for row in document["media"] if row["media_key"] in linked_media]
    if any(row["participant_key"] is None or row["session_key"] is None for row in annotated_rows):
        _fail("GROUPING_INCOMPLETE_FOR_ANNOTATED_MEDIA")
    eligible: list[dict[str, Any]] = []
    for row in annotated_rows:
        obj = object_by_key[row["object_key"]]
        if (
            obj["available_for_selective_copy"]
            and obj["size_bytes"] is not None
            and obj["size_bytes"] > 0
            and row["duration_milliseconds"] is not None
        ):
            eligible.append({**row, "size_bytes": obj["size_bytes"]})
    return eligible


def _select_pilot(
    document: Mapping[str, Any], release_digest: str, preflight: ResourcePreflight, target: int
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(target, bool) or not isinstance(target, int) or not FROZEN_MINIMUM <= target <= FROZEN_MAXIMUM:
        _fail("PILOT_TARGET_OUTSIDE_FROZEN_RANGE")
    _validate_resources(preflight)
    eligible = _eligible_media(document)
    if not eligible:
        _fail("PILOT_ELIGIBLE_POOL_EMPTY")
    tertiles = _duration_tertiles(eligible)

    by_stratum: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        stratum = (
            row["coarse_activity_label"],
            row["speech_presence_bin"],
            tertiles[row["media_key"]],
            row["location_label"] if row["location_label"] is not None else "__UNAVAILABLE__",
        )
        candidate = {
            **row,
            "stratum": stratum,
            "selection_hash": _selection_hash(release_digest, row["media_key"]),
        }
        by_stratum[stratum].append(candidate)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: row["selection_hash"])

    stratum_tiebreak = {
        stratum: hashlib.sha256(
            f"{release_digest}{FROZEN_PROTOCOL_ID}".encode("utf-8") + _canonical_bytes(list(stratum))
        ).hexdigest()
        for stratum in by_stratum
    }
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    participant_counts: Counter[str] = Counter()
    stratum_counts: Counter[tuple[str, str, str, str]] = Counter()
    raw_bytes = 0
    skipped_for_bytes = False

    for per_participant_limit in (1, 2):
        while len(selected) < target:
            options: list[tuple[int, str, str, dict[str, Any]]] = []
            for stratum, candidates in by_stratum.items():
                for candidate in candidates:
                    if candidate["media_key"] in selected_keys:
                        continue
                    if participant_counts[candidate["participant_key"]] >= per_participant_limit:
                        continue
                    if raw_bytes + candidate["size_bytes"] > preflight.raw_cap_bytes:
                        skipped_for_bytes = True
                        continue
                    options.append(
                        (
                            stratum_counts[stratum],
                            stratum_tiebreak[stratum],
                            candidate["selection_hash"],
                            candidate,
                        )
                    )
                    break
            if not options:
                break
            candidate = min(options, key=lambda option: option[:3])[3]
            selected.append(candidate)
            selected_keys.add(candidate["media_key"])
            participant_counts[candidate["participant_key"]] += 1
            stratum_counts[candidate["stratum"]] += 1
            raw_bytes += candidate["size_bytes"]
        if len(selected) >= target:
            break

    if len(selected) < FROZEN_MINIMUM:
        _fail("PILOT_MINIMUM_NOT_REACHABLE")
    if any(count > 2 for count in participant_counts.values()):
        _fail("PILOT_PARTICIPANT_CAP_INTERNAL_ERROR")
    if raw_bytes > preflight.raw_cap_bytes or raw_bytes > HARD_RAW_CAP_BYTES:
        _fail("PILOT_RAW_CAP_INTERNAL_ERROR")

    adjustment_reasons: list[str] = []
    if len(selected) < target:
        if skipped_for_bytes:
            adjustment_reasons.append("RAW_BYTE_CAP")
        if len(eligible) < target or not skipped_for_bytes:
            adjustment_reasons.append("ELIGIBLE_POOL_OR_EMPTY_STRATUM")
    return selected, adjustment_reasons


def _partition_suppression(counts: Mapping[str, int]) -> dict[str, int | str]:
    """Suppress positive small cells plus one complementary non-small cell."""

    positive = {key for key, value in counts.items() if value > 0}
    suppressed = {key for key, value in counts.items() if 0 < value < CELL_SUPPRESSION_THRESHOLD}
    if suppressed and positive - suppressed:
        complement = max(positive - suppressed, key=lambda key: (counts[key], key))
        suppressed.add(complement)
    return {
        key: ("SUPPRESSED" if key in suppressed else value)
        for key, value in sorted(counts.items())
    }


def _single_cell(value: int) -> int | str:
    if 0 < value < CELL_SUPPRESSION_THRESHOLD:
        return "SUPPRESSED"
    return value


def _object_inventory_digest(objects: Sequence[Mapping[str, Any]]) -> str | None:
    """Return an identity-neutral exact-byte multiset digest when possible."""

    comparable: list[dict[str, Any]] = []
    for row in objects:
        checksum = row["local_sha256"] or row["source_checksum_sha256"]
        if checksum is None or row["size_bytes"] is None:
            return None
        comparable.append({"sha256": checksum, "size_bytes": row["size_bytes"]})
    comparable.sort(key=lambda row: (row["sha256"], row["size_bytes"]))
    return _sha256(comparable)


def _release_binding(
    document: Mapping[str, Any], object_inventory_digest: str | None
) -> dict[str, Any]:
    release = document["release"]
    comparison = release["doi_snapshot_object_inventory_sha256"]
    limitations: list[str] = []
    if comparison is not None and object_inventory_digest is not None and comparison != object_inventory_digest:
        _fail("DOI_SNAPSHOT_OBJECT_INVENTORY_MISMATCH")

    if release["source_role"] == "DOI_SNAPSHOT":
        binding_mode = "DOI_SNAPSHOT_COMMIT_PLUS_RESTRICTED_MANIFEST"
        remotely_immutable = True
        equivalence = "SOURCE_BOUND_TO_PUBLIC_DOI_SNAPSHOT_COMMIT"
    elif release["observed_revision_commit"] is not None:
        binding_mode = "NAMED_LIVE_COMMIT_PLUS_RESTRICTED_MANIFEST"
        remotely_immutable = True
        equivalence = "NOT_PROVEN"
    else:
        binding_mode = "RESTRICTED_MANIFEST_DIGEST_PLUS_OBSERVATION_RECEIPT"
        remotely_immutable = False
        equivalence = "NOT_PROVEN"
        limitations.append(
            "The live remote revision is unpinned; the pilot is locally bound to the restricted manifest digest and observation receipt."
        )

    if comparison is not None:
        if object_inventory_digest is not None:
            equivalence = "PROVEN_BY_COMPLETE_SIZE_AND_SHA256_MULTISET_DIGEST"
        else:
            equivalence = "NOT_PROVEN_CURRENT_OBJECT_CHECKSUMS_INCOMPLETE"
            limitations.append(
                "A DOI snapshot inventory receipt exists, but current object checksums are incomplete, so byte equivalence is not claimed."
            )
    elif release["source_role"] != "DOI_SNAPSHOT":
        limitations.append("Exact byte equivalence to the public DOI snapshot has not been proven.")
    elif object_inventory_digest is None:
        limitations.append(
            "The DOI commit identity is pinned, but checksums were not independently available for every object."
        )

    return {
        "doi": release["doi"],
        "keeper_library_id": release["keeper_library_id"],
        "public_doi_snapshot_commit": release["public_doi_snapshot_commit"],
        "source_role": release["source_role"],
        "binding_mode": binding_mode,
        "remote_revision_immutable": remotely_immutable,
        "observation_receipt_bound": True,
        "doi_snapshot_equivalence": equivalence,
        "limitations": limitations,
    }


def build_artifacts(
    raw_document: Any,
    *,
    preflight: ResourcePreflight,
    target: int = FROZEN_TARGET,
    protocol_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate restricted metadata and build restricted/reportable artifacts."""

    document = _normalise_input(raw_document)
    protocol_digest = protocol_sha256 or _load_and_validate_frozen_protocol()
    release_payload = _release_payload(document)
    release_digest = _sha256(release_payload)
    selected, adjustment_reasons = _select_pilot(document, release_digest, preflight, target)

    restricted_selection = [
        {
            "selection_rank": rank,
            "media_key": row["media_key"],
            "object_key": row["object_key"],
            "participant_key": row["participant_key"],
            "session_key": row["session_key"],
            "selection_hash": row["selection_hash"],
            "stratum": {
                "coarse_activity_label": row["stratum"][0],
                "speech_presence_bin": row["stratum"][1],
                "duration_tertile": row["stratum"][2],
                "location_label": row["stratum"][3],
            },
            "size_bytes": row["size_bytes"],
        }
        for rank, row in enumerate(selected, start=1)
    ]
    selection_digest = _sha256(restricted_selection)
    restricted = {
        **release_payload,
        "canonical_restricted_manifest_sha256": release_digest,
        "frozen_protocol_id": FROZEN_PROTOCOL_ID,
        "frozen_protocol_sha256": protocol_digest,
        "pilot_selection": restricted_selection,
        "pilot_selection_sha256": selection_digest,
        "adjustment_reasons": adjustment_reasons,
    }

    objects = document["objects"]
    class_counts = Counter(row["top_level_class"] for row in objects)
    class_suppression = _partition_suppression(
        {top_class: class_counts[top_class] for top_class in TOP_LEVEL_CLASSES}
    )
    inventory: list[dict[str, Any]] = []
    for top_class in TOP_LEVEL_CLASSES:
        rows = [row for row in objects if row["top_level_class"] == top_class]
        count_report = class_suppression[top_class]
        known_checksums = {
            row["local_sha256"] or row["source_checksum_sha256"]
            for row in rows
            if row["local_sha256"] is not None or row["source_checksum_sha256"] is not None
        }
        counts_suppressed = count_report == "SUPPRESSED"
        inventory.append(
            {
                "top_level_class": top_class,
                "availability": bool(rows),
                "file_count": count_report,
                "byte_total": (
                    "SUPPRESSED"
                    if counts_suppressed or any(row["size_bytes"] is None for row in rows)
                    else sum(row["size_bytes"] for row in rows)
                ),
                "distinct_known_content_count": (
                    "SUPPRESSED" if counts_suppressed else _single_cell(len(known_checksums))
                ),
                "checksum_complete": bool(rows)
                and all(
                    row["local_sha256"] is not None or row["source_checksum_sha256"] is not None
                    for row in rows
                ),
            }
        )

    media_keys = {row["media_key"] for row in document["media"]}
    linked_media = {
        row["linked_media_key"]
        for row in document["annotations"]
        if row["linked_media_key"] is not None
    }
    annotation_partition = _partition_suppression(
        {
            "linked": sum(row["linked_media_key"] is not None for row in document["annotations"]),
            "unlinked": sum(row["linked_media_key"] is None for row in document["annotations"]),
        }
    )
    media_partition = _partition_suppression(
        {
            "with_annotation": len(media_keys & linked_media),
            "without_annotation": len(media_keys - linked_media),
        }
    )
    grouped = [
        row
        for row in document["media"]
        if row["participant_key"] is not None and row["session_key"] is not None
    ]
    participant_count = len({row["participant_key"] for row in grouped})
    session_count = len({row["session_key"] for row in grouped})
    selected_participants = Counter(row["participant_key"] for row in selected)
    selected_raw_bytes = sum(row["size_bytes"] for row in selected)
    selected_strata = {
        "coarse_activity_label": len({row["stratum"][0] for row in selected}),
        "speech_presence_bin": len({row["stratum"][1] for row in selected}),
        "duration_tertile": len({row["stratum"][2] for row in selected}),
        "location_label": len({row["stratum"][3] for row in selected}),
        "combined_metadata_stratum": len({row["stratum"] for row in selected}),
    }
    object_inventory_digest = _object_inventory_digest(objects)
    reportable = {
        "receipt_schema_version": REPORTABLE_SCHEMA,
        "artifact_status": "NONIDENTIFYING_CELL_SUPPRESSED_AGGREGATE_ONLY",
        "scientific_outcome_run": False,
        "release_binding": {
            **_release_binding(document, object_inventory_digest),
            "canonical_restricted_manifest_sha256": release_digest,
            "canonical_object_inventory_sha256": object_inventory_digest or "UNAVAILABLE",
        },
        "inventory_summary": inventory,
        "annotation_linkage_summary": {
            "annotation_records": annotation_partition,
            "media_records": media_partition,
            "complete_foreign_key_integrity": True,
            "cell_suppression_threshold": CELL_SUPPRESSION_THRESHOLD,
            "complementary_suppression_applied": True,
        },
        "grouping_summary": {
            "participant_field_present": True,
            "session_field_present": True,
            "session_definition": "PARTICIPANT_PLUS_RECORDING_DATE",
            "complete_for_annotated_media": True,
            "reportable_participant_group_count": _single_cell(participant_count),
            "reportable_session_group_count": _single_cell(session_count),
        },
        "frozen_pilot_selection": {
            "protocol_id": FROZEN_PROTOCOL_ID,
            "protocol_sha256": protocol_digest,
            "requested_video_count": target,
            "allowable_video_range": [FROZEN_MINIMUM, FROZEN_MAXIMUM],
            "selected_video_count": len(selected),
            "selected_raw_bytes": selected_raw_bytes,
            "selected_participant_group_count": len(selected_participants),
            "maximum_videos_per_participant": max(selected_participants.values()),
            "all_selected_participants_distinct": max(selected_participants.values()) == 1,
            "selection_sha256": selection_digest,
            "selection_formula": "SHA-256(release_manifest_digest || protocol_id || internal_episode_key)",
            "content_blind_metadata_fields": [
                "coarse_activity_label",
                "speech_presence_bin",
                "duration_tertile",
                "location_label",
            ],
            "reportable_distinct_selected_strata": {
                key: _single_cell(value) for key, value in sorted(selected_strata.items())
            },
            "target_adjusted": len(selected) != target,
            "adjustment_reasons": adjustment_reasons,
        },
        "resource_admission": {
            "raw_cap_bytes": preflight.raw_cap_bytes,
            "hard_raw_cap_bytes": HARD_RAW_CAP_BYTES,
            "projected_namespace_peak_bytes": preflight.projected_namespace_peak_bytes,
            "hard_namespace_cap_bytes": HARD_NAMESPACE_CAP_BYTES,
            "projected_post_peak_free_bytes": (
                preflight.volume_free_before_bytes - preflight.projected_namespace_peak_bytes
            ),
            "hard_post_peak_free_floor_bytes": HARD_FREE_SPACE_FLOOR_BYTES,
            "admitted": True,
        },
        "privacy": {
            "contains_filenames_or_paths": False,
            "contains_row_level_keys": False,
            "contains_per_object_checksums": False,
            "contains_exact_media_timestamps": False,
            "contains_transcript_or_lexical_content": False,
            "cell_suppression_applied": True,
        },
    }
    return restricted, reportable


def _is_inside(path: Path, root: Path) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=True)
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _atomic_json_write(path: Path, value: Any, *, mode: int) -> None:
    if not path.parent.exists() or not path.parent.is_dir():
        _fail("OUTPUT_PARENT_UNAVAILABLE")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".childlens-v1-1-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_artifacts(
    restricted_output: Path,
    reportable_output: Path,
    restricted: Mapping[str, Any],
    reportable: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Write artifacts atomically after enforcing the quarantine boundary."""

    if _is_inside(restricted_output, repo_root):
        _fail("RESTRICTED_OUTPUT_INSIDE_REPOSITORY")
    if restricted_output.resolve(strict=False) == reportable_output.resolve(strict=False):
        _fail("RESTRICTED_AND_REPORTABLE_OUTPUT_COLLIDE")
    _atomic_json_write(restricted_output, restricted, mode=0o600)
    try:
        _atomic_json_write(reportable_output, reportable, mode=0o644)
    except Exception:
        # Do not silently leave a restricted artifact without the public receipt.
        try:
            restricted_output.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a restricted ChildLens v1.1 manifest and aggregate pilot receipt."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--restricted-output", type=Path, required=True)
    parser.add_argument("--reportable-output", type=Path, required=True)
    parser.add_argument("--raw-cap-bytes", type=int, required=True)
    parser.add_argument("--projected-namespace-peak-bytes", type=int, required=True)
    parser.add_argument("--volume-free-before-bytes", type=int, required=True)
    parser.add_argument("--target", type=int, default=FROZEN_TARGET)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if _is_inside(args.restricted_output, REPO_ROOT):
            _fail("RESTRICTED_OUTPUT_INSIDE_REPOSITORY")
        try:
            raw_document = json.loads(args.input.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            _fail("INPUT_UNREADABLE")
        protocol_digest = _load_and_validate_frozen_protocol()
        restricted, reportable = build_artifacts(
            raw_document,
            preflight=ResourcePreflight(
                raw_cap_bytes=args.raw_cap_bytes,
                projected_namespace_peak_bytes=args.projected_namespace_peak_bytes,
                volume_free_before_bytes=args.volume_free_before_bytes,
            ),
            target=args.target,
            protocol_sha256=protocol_digest,
        )
        write_artifacts(args.restricted_output, args.reportable_output, restricted, reportable)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "canonical_restricted_manifest_sha256": reportable["release_binding"][
                        "canonical_restricted_manifest_sha256"
                    ],
                    "selected_video_count": reportable["frozen_pilot_selection"][
                        "selected_video_count"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except ManifestError as exc:
        print(json.dumps({"status": "FAIL", "error_code": exc.code}, sort_keys=True), file=sys.stderr)
        return 2
    except Exception:
        print(json.dumps({"status": "FAIL", "error_code": "UNEXPECTED_INTERNAL_ERROR"}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
