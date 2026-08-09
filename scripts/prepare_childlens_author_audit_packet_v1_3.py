#!/usr/bin/env python3
"""Prepare prediction-blind ChildLens v1.3 author-audit samples.

This module is deliberately not a command-line program.  A trusted, no-argument
quarantine orchestrator must call :func:`prepare_author_audit_samples` with
already-resolved paths.  That keeps restricted paths out of argv and prevents
this component from discovering a quarantine or corpus content on its own.

Only the frozen v1.2 selection digest, opaque item keys, content-addressed media
bindings, media durations, and official speech-presence windows can influence
sampling.  Model predictions, confidence, lexical content, visual salience,
frames, audio, transcripts, and speaker labels are neither accepted nor read.
Exact item keys and intervals are written only to owner-private restricted
packets.  The optional public receipt contains aggregate counts and digests.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import stat
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


VERSION = "childlens-author-audit-sampler-v1.3.0"
PRIMARY_SCHEMA = "childlens-author-audit-primary-sample-v1.3.0"
RESERVE_SCHEMA = "childlens-author-audit-reserve-sample-v1.3.0"
RECEIPT_SCHEMA = "childlens-author-audit-sampling-receipt-v1.3.0"
INPUT_SCHEMA = "childlens-restricted-measurement-manifest-v1.2.1"

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_V1_2_SELECTION_DIGEST = (
    "61526ea6cebc256314b0b9e574fb1a5986fcd83a534979e93134c4163c55253f"
)
EXPECTED_ITEM_COUNT = 15
TARGET_SAMPLE_MS = 15 * 60 * 1000
PREFERRED_ITEM_MS = 60 * 1000
MINIMUM_CLUSTER_MS_PER_SAMPLE = 1000
MAX_MANIFEST_BYTES = 128 * 1024**2
INDEX_SENTINEL = ".metadata_never_index"

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
OPAQUE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{12,80}$")
CONTENT_ADDRESS_RE = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/)*([0-9a-f]{64})\.bin$"
)
TOP_LEVEL_FIELDS = frozenset({"schema_version", "pilot_selection_sha256", "items"})
ITEM_FIELDS = frozenset(
    {
        "blinded_item_key",
        "media_relpath",
        "expected_size_bytes",
        "expected_media_sha256",
        "reference_duration_seconds",
        "reference_duration_basis",
        "speech_presence_expectation",
        "speech_windows",
    }
)
WINDOW_FIELDS = frozenset({"start_seconds", "end_seconds"})

POLICY: Mapping[str, Any] = {
    "schema_version": VERSION,
    "frozen_selection_digest": FROZEN_V1_2_SELECTION_DIGEST,
    "expected_item_count": EXPECTED_ITEM_COUNT,
    "primary_total_duration_ms": TARGET_SAMPLE_MS,
    "reserve_total_duration_ms": TARGET_SAMPLE_MS,
    "preferred_duration_per_item_per_sample_ms": PREFERRED_ITEM_MS,
    "minimum_cluster_duration_per_sample_ms": MINIMUM_CLUSTER_MS_PER_SAMPLE,
    "time_grid": "INTEGER_MILLISECONDS_START_CEILING_END_FLOOR_CLIPPED_TO_MEDIA_DURATION",
    "window_normalization": "SORT_COALESCE_OVERLAP_OR_ADJACENCY",
    "allocation": (
        "PREFER_60000_MS_PER_ITEM_PER_SAMPLE_THEN_HASH_ORDERED_DEFICIT_REDISTRIBUTION"
    ),
    "placement": "HASH_DERIVED_DISJOINT_BLOCKS_IN_UNIONED_SPEECH_TIME",
    "reserve_activation": "BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK",
    "selection_basis_fields": [
        "pilot_selection_sha256",
        "blinded_item_key",
        "reference_duration_seconds",
        "official_speech_window_start_seconds",
        "official_speech_window_end_seconds",
    ],
    "model_predictions_used": False,
    "confidence_used": False,
    "lexical_content_used": False,
    "visual_content_used": False,
}


class SamplingError(RuntimeError):
    """A fixed fail-closed diagnostic safe for a caller to report."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise SamplingError(code)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("E_CANONICAL_JSON")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _private_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _private_regular(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _validate_restricted_root(root_arg: Path) -> Path:
    if not root_arg.is_absolute():
        _fail("E_RESTRICTED_ROOT_NOT_ABSOLUTE")
    try:
        root = root_arg.resolve(strict=True)
    except OSError:
        _fail("E_RESTRICTED_ROOT_MISSING")
    if root == REPO_ROOT or _is_relative_to(root, REPO_ROOT):
        _fail("E_RESTRICTED_ROOT_INSIDE_REPOSITORY")
    if not _private_directory(root) or not _private_regular(root / INDEX_SENTINEL):
        _fail("E_RESTRICTED_ROOT_CONTROLS")
    return root


def _restricted_member(root: Path, path_arg: Path, *, must_exist: bool) -> Path:
    if not path_arg.is_absolute():
        _fail("E_RESTRICTED_PATH_NOT_ABSOLUTE")
    try:
        parent = path_arg.parent.resolve(strict=True)
    except OSError:
        _fail("E_RESTRICTED_PATH_PARENT")
    if not _is_relative_to(parent, root) or not _private_directory(parent):
        _fail("E_RESTRICTED_PATH_BOUNDARY")
    relative = parent.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            if stat.S_ISLNK(cursor.lstat().st_mode):
                _fail("E_RESTRICTED_PATH_SYMLINK")
        except OSError:
            _fail("E_RESTRICTED_PATH_PARENT")
    candidate = parent / path_arg.name
    if must_exist:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            _fail("E_RESTRICTED_INPUT_MISSING")
        if not _is_relative_to(resolved, root) or not _private_regular(candidate):
            _fail("E_RESTRICTED_INPUT_CONTROLS")
        return resolved
    if candidate.exists() and not _private_regular(candidate):
        _fail("E_RESTRICTED_OUTPUT_CONTROLS")
    return candidate


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        if path.stat().st_size <= 0 or path.stat().st_size > MAX_MANIFEST_BYTES:
            _fail("E_MANIFEST_SIZE")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_MANIFEST_READ")
    if not isinstance(value, dict):
        _fail("E_MANIFEST_SCHEMA")
    return value


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("E_TIME_VALUE")
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        _fail("E_TIME_VALUE")
    if not number.is_finite():
        _fail("E_TIME_VALUE")
    return number


def _start_ms(value: Any) -> int:
    return int((_decimal(value) * 1000).to_integral_value(rounding=ROUND_CEILING))


def _end_ms(value: Any) -> int:
    return int((_decimal(value) * 1000).to_integral_value(rounding=ROUND_FLOOR))


def _coalesce(windows: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(windows):
        if start < 0 or end <= start:
            _fail("E_SPEECH_WINDOW")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _hash(selection: str, purpose: str, item_key: str) -> str:
    return hashlib.sha256(
        f"{selection}\0{VERSION}\0{purpose}\0{item_key}".encode("utf-8")
    ).hexdigest()


def _hash_int(selection: str, purpose: str, item_key: str) -> int:
    return int(_hash(selection, purpose, item_key), 16)


def _allocate(
    capacities: Mapping[str, int], selection: str
) -> tuple[dict[str, int], dict[str, int]]:
    primary = {
        key: min(PREFERRED_ITEM_MS, capacity // 2)
        for key, capacity in capacities.items()
    }
    reserve = dict(primary)
    if any(value < MINIMUM_CLUSTER_MS_PER_SAMPLE for value in primary.values()):
        _fail("E_CLUSTER_COVERAGE_INSUFFICIENT")
    primary_deficit = TARGET_SAMPLE_MS - sum(primary.values())
    reserve_deficit = TARGET_SAMPLE_MS - sum(reserve.values())
    if primary_deficit < 0 or reserve_deficit < 0:
        _fail("E_ALLOCATION_INTERNAL")
    remaining = {
        key: capacities[key] - primary[key] - reserve[key]
        for key in capacities
    }

    def distribute(target: dict[str, int], deficit: int, purpose: str) -> None:
        for key in sorted(
            remaining,
            key=lambda item: (_hash(selection, purpose, item), item),
        ):
            if deficit == 0:
                break
            grant = min(deficit, remaining[key])
            target[key] += grant
            remaining[key] -= grant
            deficit -= grant
        if deficit:
            _fail("E_TOTAL_COVERAGE_INSUFFICIENT")

    distribute(primary, primary_deficit, "primary-deficit-donor-rank")
    distribute(reserve, reserve_deficit, "reserve-deficit-donor-rank")
    if sum(primary.values()) != TARGET_SAMPLE_MS or sum(reserve.values()) != TARGET_SAMPLE_MS:
        _fail("E_ALLOCATION_INTERNAL")
    return primary, reserve


def _map_union_block(
    windows: Sequence[tuple[int, int]], offset_ms: int, duration_ms: int
) -> list[dict[str, int]]:
    cursor = 0
    remaining = duration_ms
    segments: list[dict[str, int]] = []
    for start, end in windows:
        length = end - start
        if offset_ms >= cursor + length:
            cursor += length
            continue
        local_offset = max(0, offset_ms - cursor)
        segment_start = start + local_offset
        take = min(remaining, end - segment_start)
        if take > 0:
            segments.append({"start_ms": segment_start, "end_ms": segment_start + take})
            remaining -= take
            offset_ms += take
        cursor += length
        if remaining == 0:
            break
    if remaining:
        _fail("E_UNION_MAPPING_INTERNAL")
    return segments


def _physical_overlap(
    first: Sequence[Mapping[str, int]], second: Sequence[Mapping[str, int]]
) -> bool:
    return any(
        max(left["start_ms"], right["start_ms"])
        < min(left["end_ms"], right["end_ms"])
        for left in first
        for right in second
    )


def _place_samples(
    *,
    item_key: str,
    selection: str,
    windows: Sequence[tuple[int, int]],
    primary_duration: int,
    reserve_duration: int,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    capacity = sum(end - start for start, end in windows)
    slack = capacity - primary_duration - reserve_duration
    if slack < 0:
        _fail("E_PLACEMENT_INTERNAL")
    lead = _hash_int(selection, "placement-leading-slack", item_key) % (slack + 1)
    gap_capacity = slack - lead
    gap = _hash_int(selection, "placement-middle-gap", item_key) % (gap_capacity + 1)
    reserve_first = bool(_hash_int(selection, "placement-orientation", item_key) & 1)
    if reserve_first:
        reserve_offset = lead
        primary_offset = lead + reserve_duration + gap
    else:
        primary_offset = lead
        reserve_offset = lead + primary_duration + gap
    primary = _map_union_block(windows, primary_offset, primary_duration)
    reserve = _map_union_block(windows, reserve_offset, reserve_duration)
    if _physical_overlap(primary, reserve):
        _fail("E_SAMPLE_OVERLAP_INTERNAL")
    return primary, reserve


def _validate_manifest(
    manifest: Mapping[str, Any], expected_selection_digest: str
) -> tuple[str, list[dict[str, Any]]]:
    if set(manifest) != TOP_LEVEL_FIELDS or manifest.get("schema_version") != INPUT_SCHEMA:
        _fail("E_MANIFEST_SCHEMA")
    selection = manifest.get("pilot_selection_sha256")
    if (
        not isinstance(selection, str)
        or not HEX64_RE.fullmatch(selection)
        or selection != expected_selection_digest
    ):
        _fail("E_SELECTION_BINDING")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != EXPECTED_ITEM_COUNT:
        _fail("E_ITEM_COUNT")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict) or set(raw) != ITEM_FIELDS:
            _fail("E_ITEM_SCHEMA")
        item_key = raw.get("blinded_item_key")
        if (
            not isinstance(item_key, str)
            or not OPAQUE_KEY_RE.fullmatch(item_key)
            or item_key in seen
        ):
            _fail("E_ITEM_KEY")
        seen.add(item_key)
        media_relpath = raw.get("media_relpath")
        media_sha = raw.get("expected_media_sha256")
        if (
            not isinstance(media_relpath, str)
            or not isinstance(media_sha, str)
            or not HEX64_RE.fullmatch(media_sha)
        ):
            _fail("E_MEDIA_BINDING")
        content_match = CONTENT_ADDRESS_RE.fullmatch(media_relpath)
        relative_parts = PurePosixPath(media_relpath).parts
        if (
            content_match is None
            or content_match.group(1) != media_sha
            or PurePosixPath(media_relpath).is_absolute()
            or any(part in {"", ".", ".."} for part in relative_parts)
        ):
            _fail("E_MEDIA_BINDING")
        size = raw.get("expected_size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            _fail("E_MEDIA_BINDING")
        duration_ms = _end_ms(raw.get("reference_duration_seconds"))
        if duration_ms <= 0 or not isinstance(raw.get("reference_duration_basis"), str):
            _fail("E_DURATION")
        if raw.get("speech_presence_expectation") != "PRESENT":
            _fail("E_CLUSTER_COVERAGE_INSUFFICIENT")
        windows_raw = raw.get("speech_windows")
        if not isinstance(windows_raw, list) or not windows_raw:
            _fail("E_CLUSTER_COVERAGE_INSUFFICIENT")
        windows: list[tuple[int, int]] = []
        for window in windows_raw:
            if not isinstance(window, dict) or set(window) != WINDOW_FIELDS:
                _fail("E_WINDOW_SCHEMA")
            start = _start_ms(window.get("start_seconds"))
            end = min(duration_ms, _end_ms(window.get("end_seconds")))
            if start < 0 or end <= start:
                _fail("E_SPEECH_WINDOW")
            windows.append((start, end))
        validated.append(
            {
                "item_key": item_key,
                "media_relpath": media_relpath,
                "media_sha256": media_sha,
                "windows": _coalesce(windows),
            }
        )
    return selection, validated


def _write_immutable(path: Path, value: Any, *, private: bool) -> None:
    encoded = _canonical(value) + b"\n"
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError:
            _fail("E_OUTPUT_READ")
        if existing != encoded:
            _fail("E_IMMUTABLE_OUTPUT_CONFLICT")
        return
    mode = 0o600 if private else 0o644
    temporary = path.parent / f".tmp-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except OSError:
        _fail("E_OUTPUT_WRITE")
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _public_output(path: Path) -> Path:
    if not path.is_absolute():
        _fail("E_PUBLIC_OUTPUT_NOT_ABSOLUTE")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError:
        _fail("E_PUBLIC_OUTPUT_PARENT")
    if path.exists() and (not path.is_file() or path.is_symlink()):
        _fail("E_PUBLIC_OUTPUT_CONTROLS")
    return parent / path.name


def prepare_author_audit_samples(
    *,
    restricted_root: Path,
    official_windows_manifest_path: Path,
    primary_packet_path: Path,
    reserve_packet_path: Path,
    public_receipt_path: Path | None = None,
    expected_selection_digest: str = FROZEN_V1_2_SELECTION_DIGEST,
) -> Mapping[str, Any]:
    """Create immutable primary/reserve packets and an aggregate-safe receipt.

    The caller must supply every path.  The function performs no quarantine or
    corpus discovery and never opens media.  It returns the same aggregate-only
    mapping written to ``public_receipt_path``; row-level packets remain private.
    """

    if not isinstance(expected_selection_digest, str) or not HEX64_RE.fullmatch(
        expected_selection_digest
    ):
        _fail("E_SELECTION_BINDING")
    root = _validate_restricted_root(restricted_root)
    manifest_path = _restricted_member(
        root, official_windows_manifest_path, must_exist=True
    )
    primary_path = _restricted_member(root, primary_packet_path, must_exist=False)
    reserve_path = _restricted_member(root, reserve_packet_path, must_exist=False)
    if len({manifest_path, primary_path, reserve_path}) != 3:
        _fail("E_PATH_COLLISION")
    manifest = _read_manifest(manifest_path)
    selection, items = _validate_manifest(manifest, expected_selection_digest)
    capacities = {
        row["item_key"]: sum(end - start for start, end in row["windows"])
        for row in items
    }
    primary_allocations, reserve_allocations = _allocate(capacities, selection)
    policy = dict(POLICY)
    policy["frozen_selection_digest"] = expected_selection_digest
    policy_sha = _digest(policy)
    manifest_sha = _digest(manifest)

    primary_items: list[dict[str, Any]] = []
    reserve_items: list[dict[str, Any]] = []
    for row in items:
        key = row["item_key"]
        primary_segments, reserve_segments = _place_samples(
            item_key=key,
            selection=selection,
            windows=row["windows"],
            primary_duration=primary_allocations[key],
            reserve_duration=reserve_allocations[key],
        )
        audit_key = "AA-" + _hash(selection, "author-audit-key", key)[:32]
        join_key = "PJ-" + _hash(selection, "prediction-join-key", key)[:32]
        common = {
            "audit_item_key": audit_key,
            "prediction_join_key": join_key,
            "media_relpath": row["media_relpath"],
            "expected_media_sha256": row["media_sha256"],
        }
        primary_items.append(
            {
                **common,
                "duration_ms": primary_allocations[key],
                "segments": primary_segments,
            }
        )
        reserve_items.append(
            {
                **common,
                "duration_ms": reserve_allocations[key],
                "segments": reserve_segments,
            }
        )
    primary_items.sort(key=lambda row: row["audit_item_key"])
    reserve_items.sort(key=lambda row: row["audit_item_key"])

    base = {
        "frozen_selection_digest": selection,
        "sampler_policy_sha256": policy_sha,
        "official_windows_manifest_sha256": manifest_sha,
        "item_count": EXPECTED_ITEM_COUNT,
        "opaque_key_attestation": True,
        "selection_independent_of_model_outputs": True,
        "sample_frozen_before_predictions": True,
        "model_predictions_present": False,
        "model_predictions_used_for_selection": False,
        "exact_intervals_restricted": True,
    }
    primary_packet = {
        "schema_version": PRIMARY_SCHEMA,
        "sample_kind": "PRIMARY_AUTHOR_AUDIT",
        "route": "AUTHOR_AUDIT_A",
        "activation": "READY_BEFORE_MODEL_REVEAL",
        **base,
        "total_duration_ms": TARGET_SAMPLE_MS,
        "items": primary_items,
    }
    reserve_packet = {
        "schema_version": RESERVE_SCHEMA,
        "sample_kind": "BORDERLINE_ESCALATION_RESERVE",
        "route": "AUTHOR_AUDIT_A_BORDERLINE_RESERVE",
        "activation": "BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK",
        "maximum_activation_count": 1,
        **base,
        "total_duration_ms": TARGET_SAMPLE_MS,
        "items": reserve_items,
    }
    primary_sha = _digest(primary_packet)
    reserve_sha = _digest(reserve_packet)
    _write_immutable(primary_path, primary_packet, private=True)
    _write_immutable(reserve_path, reserve_packet, private=True)

    one_minute_primary = all(value == PREFERRED_ITEM_MS for value in primary_allocations.values())
    one_minute_reserve = all(value == PREFERRED_ITEM_MS for value in reserve_allocations.values())
    receipt: Mapping[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "AUTHOR_AUDIT_SAMPLES_READY",
        "frozen_v1_2_selection_digest": selection,
        "sampler_policy_sha256": policy_sha,
        "official_windows_manifest_sha256": manifest_sha,
        "primary_packet_sha256": primary_sha,
        "reserve_packet_sha256": reserve_sha,
        "selected_item_count": EXPECTED_ITEM_COUNT,
        "all_selected_items_represented_in_primary": True,
        "all_selected_items_represented_in_reserve": True,
        "primary_total_seconds": TARGET_SAMPLE_MS // 1000,
        "reserve_total_seconds": TARGET_SAMPLE_MS // 1000,
        "primary_one_minute_per_item": one_minute_primary,
        "reserve_one_minute_per_item": one_minute_reserve,
        "primary_reserve_disjoint": True,
        "audit_sample_frozen_before_predictions": True,
        "author_prediction_blinding_required": True,
        "reserve_packet_sealed_separately": True,
        "selection_uses_only_official_windows_and_frozen_binding": True,
        "model_prediction_independent": True,
        "model_confidence_independent": True,
        "lexical_content_independent": True,
        "visual_content_independent": True,
        "prediction_payload_accepted_by_sampler": False,
        "exact_intervals_exported": False,
        "item_identifiers_exported": False,
        "reserve_activation": "BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK",
        "maximum_reserve_activation_count": 1,
        "deficit_redistribution_used": not (
            one_minute_primary and one_minute_reserve
        ),
    }
    if public_receipt_path is not None:
        public_path = _public_output(public_receipt_path)
        if _is_relative_to(public_path.resolve(strict=False), root):
            _fail("E_PUBLIC_OUTPUT_INSIDE_RESTRICTED_ROOT")
        _write_immutable(public_path, receipt, private=False)
    return receipt


__all__ = [
    "FROZEN_V1_2_SELECTION_DIGEST",
    "SamplingError",
    "prepare_author_audit_samples",
]
