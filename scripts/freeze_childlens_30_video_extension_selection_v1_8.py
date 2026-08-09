#!/usr/bin/env python3
"""Freeze the content-blind ChildLens 15-item extension inside quarantine.

The zero-argument program reads only the already bound restricted manifest,
the immutable original 15-item measurement manifest, and locally verified
official annotation files. It never opens media or model output. Exact rows,
participant/session keys, locators, intervals, and sizes remain in quarantine;
stdout and the repository receive only a K-safe aggregate receipt.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "docs/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "frozen_extension_protocol_v1_8.json"
)
PREDECESSOR = (
    ROOT
    / "output/nursery_program_convergence_v1"
    / "childlens_pseudo_calibration_gemma_substitution_receipt.json"
)
PUBLIC_RECEIPT = (
    ROOT
    / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "selection_and_admission_receipt.json"
)
AUTHOR = ROOT / "scripts/childlens_author_audit_v1_3.py"
POST = ROOT / "scripts/run_childlens_post_acquisition_v1_2.py"

VERSION = "nursery-childlens-30-video-extension-selection-v1.8.0"
PROTOCOL_SHA256 = "e8d7ca48ed888f990730861e8f5a97274743dfc0e08befcdf40a508356c2aa56"
PREDECESSOR_SHA256 = "2918e5094071e003931f79286945ad515ad41a2c5987eee3e98b14ea398f5b56"
PROTOCOL_ID = "childlens-provisional-pseudo-calibration-one-time-30-video-extension"
FULL_SCHEMA = "childlens-restricted-manifest-input-v1.1.0"
MEASUREMENT_SCHEMA = "childlens-restricted-measurement-manifest-v1.2.1"
PLAN_SCHEMA = "nursery-childlens-30-video-extension-restricted-plan-v1.8.0"
RECEIPT_SCHEMA = "nursery-childlens-30-video-extension-selection-receipt-v1.8.0"
ORIGINAL_COUNT = 15
ADDITIONAL_COUNT = 15
SAMPLE_MS = 60_000
WINDOWS_PER_ITEM = 9
WINDOW_MS = 10_000
FRAME_MARGIN_MS = 5_000
DERIVED_BOUND_BPS = 3_000_000
DERIVED_CAP_BYTES = 4 * 1024**3
RAW_CAP_BYTES = 20 * 1024**3
NAMESPACE_CAP_BYTES = 73 * 1024**3
FREE_FLOOR_BYTES = 50 * 1024**3
RETENTION_DEADLINE = "2027-07-31"
MAX_JSON_BYTES = 128 * 1024**2
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PATH_TOKEN = re.compile(r"(?i)(?:/users/|file://|\\\\users\\\\)")
MEDIA_TOKEN = re.compile(r"(?i)\b\S+\.(?:mp4|mov|mkv|avi|webm|wav|m4a|json)\b")


class ExtensionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise ExtensionError(code)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail("E_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        _fail("E_MODULE")
    return module


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
        _fail("E_CANONICAL")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        _fail("E_FILE")
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("E_FILE")
        payload = path.read_bytes()
        if not payload or len(payload) > MAX_JSON_BYTES:
            _fail("E_FILE")
        return json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_FILE")


def _private_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _private_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _write_once(path: Path, value: Any, *, private: bool) -> None:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        os.chmod(path.parent, 0o700)
    if path.exists():
        if path.read_bytes() != payload:
            _fail("E_IMMUTABLE_CONFLICT")
        return
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    mode = 0o600 if private else 0o644
    try:
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, mode)
    except OSError:
        _fail("E_WRITE")
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _release_binding_digest(document: Mapping[str, Any]) -> str:
    release = document.get("release")
    if not isinstance(release, Mapping):
        _fail("E_MANIFEST")
    required = {
        "doi",
        "doi_snapshot_inventory_receipt_sha256",
        "doi_snapshot_object_inventory_sha256",
        "keeper_library_id",
        "observation_date",
        "observation_receipt_sha256",
        "observed_revision_commit",
        "public_doi_snapshot_commit",
        "session_definition",
        "source_role",
    }
    if set(release) != required:
        _fail("E_MANIFEST")
    return _digest(dict(release))


def _coalesce_ms(windows: Sequence[tuple[float, float]], duration_ms: int) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    for start, end in windows:
        left = max(0, math.ceil(float(start) * 1000.0))
        right = min(duration_ms, math.floor(float(end) * 1000.0))
        if right > left:
            cleaned.append((left, right))
    merged: list[list[int]] = []
    for start, end in sorted(cleaned):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _map_union_range(
    windows: Sequence[tuple[int, int]], offset: int, duration: int
) -> list[tuple[int, int]]:
    cursor = 0
    remaining = duration
    result: list[tuple[int, int]] = []
    for start, end in windows:
        length = end - start
        if offset >= cursor + length:
            cursor += length
            continue
        local = max(0, offset - cursor)
        source_start = start + local
        take = min(remaining, end - source_start)
        if take > 0:
            result.append((source_start, source_start + take))
            offset += take
            remaining -= take
        cursor += length
        if remaining == 0:
            break
    if remaining:
        _fail("E_SAMPLE_MAPPING")
    return result


def _map_union_point(windows: Sequence[tuple[int, int]], offset: int) -> int:
    cursor = 0
    for start, end in windows:
        length = end - start
        if offset < cursor + length:
            return start + (offset - cursor)
        cursor += length
    if offset == cursor and windows:
        return windows[-1][1]
    _fail("E_SAMPLE_MAPPING")


def _duration_tertiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ordered = sorted(rows, key=lambda row: (int(row["duration_milliseconds"]), str(row["media_key"])))
    labels = ("LOW", "MIDDLE", "HIGH")
    total = len(ordered)
    return {
        str(row["media_key"]): labels[min(2, (index * 3) // total)]
        for index, row in enumerate(ordered)
    }


def _select(
    candidates: Sequence[Mapping[str, Any]], release_digest: str
) -> list[dict[str, Any]]:
    tertiles = _duration_tertiles(candidates)
    by_stratum: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        row = dict(raw)
        stratum = (
            str(row["coarse_activity_label"]),
            str(row["speech_presence_bin"]),
            tertiles[str(row["media_key"])],
            str(row["location_label"]) if row.get("location_label") is not None else "__UNAVAILABLE__",
        )
        row["stratum"] = stratum
        row["selection_hash"] = hashlib.sha256(
            f"{release_digest}{PROTOCOL_ID}{row['media_key']}".encode("utf-8")
        ).hexdigest()
        by_stratum[stratum].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: (row["selection_hash"], row["media_key"]))
    stratum_hash = {
        stratum: hashlib.sha256(
            release_digest.encode("utf-8")
            + PROTOCOL_ID.encode("utf-8")
            + _canonical(list(stratum))
        ).hexdigest()
        for stratum in by_stratum
    }
    selected: list[dict[str, Any]] = []
    participants: set[str] = set()
    keys: set[str] = set()
    counts: Counter[tuple[str, str, str, str]] = Counter()
    while len(selected) < ADDITIONAL_COUNT:
        options: list[tuple[int, str, str, dict[str, Any]]] = []
        for stratum, rows in by_stratum.items():
            candidate = next(
                (
                    row
                    for row in rows
                    if row["media_key"] not in keys
                    and row["participant_key"] not in participants
                ),
                None,
            )
            if candidate is not None:
                options.append(
                    (counts[stratum], stratum_hash[stratum], candidate["selection_hash"], candidate)
                )
        if not options:
            _fail("E_SELECTION_INSUFFICIENT")
        chosen = min(options, key=lambda row: row[:3])[3]
        selected.append(chosen)
        keys.add(str(chosen["media_key"]))
        participants.add(str(chosen["participant_key"]))
        counts[chosen["stratum"]] += 1
    return selected


def build_selection(
    full: Mapping[str, Any],
    original: Mapping[str, Any],
    *,
    annotation_loader: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build exact private plan and aggregate receipt without opening media."""

    if full.get("schema_version") != FULL_SCHEMA or original.get("schema_version") != MEASUREMENT_SCHEMA:
        _fail("E_MANIFEST")
    if not all(isinstance(full.get(key), list) for key in ("objects", "media", "annotations")):
        _fail("E_MANIFEST")
    original_items = original.get("items")
    if not isinstance(original_items, list) or len(original_items) != ORIGINAL_COUNT:
        _fail("E_ORIGINAL_SAMPLE")
    original_keys = {row.get("blinded_item_key") for row in original_items if isinstance(row, Mapping)}
    if len(original_keys) != ORIGINAL_COUNT or not all(isinstance(value, str) for value in original_keys):
        _fail("E_ORIGINAL_SAMPLE")
    media_by_key = {
        row.get("media_key"): row
        for row in full["media"]
        if isinstance(row, Mapping) and isinstance(row.get("media_key"), str)
    }
    if not original_keys.issubset(media_by_key):
        _fail("E_ORIGINAL_LINKAGE")
    original_participants = {media_by_key[key].get("participant_key") for key in original_keys}
    if None in original_participants or len(original_participants) != ORIGINAL_COUNT:
        _fail("E_ORIGINAL_GROUPING")
    objects = {
        row.get("object_key"): row
        for row in full["objects"]
        if isinstance(row, Mapping) and isinstance(row.get("object_key"), str)
    }
    links: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in full["annotations"]:
        if isinstance(row, Mapping) and isinstance(row.get("linked_media_key"), str):
            links[str(row["linked_media_key"])].append(row)
    candidates: list[dict[str, Any]] = []
    for raw in full["media"]:
        if not isinstance(raw, Mapping):
            continue
        media_key = raw.get("media_key")
        participant = raw.get("participant_key")
        session = raw.get("session_key")
        obj = objects.get(raw.get("object_key"))
        duration_ms = raw.get("duration_milliseconds")
        if (
            not isinstance(media_key, str)
            or media_key in original_keys
            or not isinstance(participant, str)
            or participant in original_participants
            or not isinstance(session, str)
            or raw.get("speech_presence_bin") != "PRESENT"
            or isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or duration_ms <= 0
            or not isinstance(obj, Mapping)
            or obj.get("top_level_class") != "VIDEO"
            or obj.get("available_for_selective_copy") is not True
            or isinstance(obj.get("size_bytes"), bool)
            or not isinstance(obj.get("size_bytes"), int)
            or obj["size_bytes"] <= 0
            or not isinstance(obj.get("source_locator"), str)
            or not links.get(media_key)
        ):
            continue
        all_windows: list[tuple[float, float]] = []
        linkage: list[str] = []
        annotation_count = 0
        invalid_count = 0
        for link in links[media_key]:
            annotation_obj = objects.get(link.get("object_key"))
            if not isinstance(annotation_obj, Mapping):
                _fail("E_ANNOTATION_LINK")
            loaded = annotation_loader(annotation_obj)
            if not isinstance(loaded, tuple) or len(loaded) != 3:
                _fail("E_ANNOTATION_LINK")
            windows, invalid, digest = loaded
            all_windows.extend(windows)
            invalid_count += int(invalid)
            linkage.append(str(digest))
            annotation_count += 1
        windows_ms = _coalesce_ms(all_windows, duration_ms)
        if sum(end - start for start, end in windows_ms) < SAMPLE_MS:
            continue
        candidates.append(
            {
                **dict(raw),
                "size_bytes": int(obj["size_bytes"]),
                "source_locator": str(obj["source_locator"]),
                "official_windows_ms": windows_ms,
                "annotation_sha256s": sorted(linkage),
                "annotation_count": annotation_count,
                "invalid_timing_count": invalid_count,
            }
        )
    release_digest = _release_binding_digest(full)
    selected = _select(candidates, release_digest)
    selected_identity = [
        {
            "rank": index,
            "media_key": row["media_key"],
            "participant_key": row["participant_key"],
            "session_key": row["session_key"],
            "object_key": row["object_key"],
            "selection_hash": row["selection_hash"],
            "stratum": list(row["stratum"]),
        }
        for index, row in enumerate(selected, 1)
    ]
    additional_digest = _digest(selected_identity)
    original_digest = original.get("pilot_selection_sha256")
    if not isinstance(original_digest, str) or not HEX64.fullmatch(original_digest):
        _fail("E_ORIGINAL_SAMPLE")
    combined_digest = _digest(
        {
            "original_selection_sha256": original_digest,
            "additional_selection_sha256": additional_digest,
            "protocol_sha256": PROTOCOL_SHA256,
        }
    )
    plan_rows: list[dict[str, Any]] = []
    projected_clip_bytes = 0
    for index, row in enumerate(selected, 1):
        capacity = sum(end - start for start, end in row["official_windows_ms"])
        material = f"{combined_digest}\0{PROTOCOL_ID}\0speech-block\0{row['media_key']}"
        offset = int(hashlib.sha256(material.encode("utf-8")).hexdigest(), 16) % (
            capacity - SAMPLE_MS + 1
        )
        segments = _map_union_range(row["official_windows_ms"], offset, SAMPLE_MS)
        centers = [
            _map_union_point(row["official_windows_ms"], offset + ((2 * q + 1) * SAMPLE_MS) // 18)
            for q in range(WINDOWS_PER_ITEM)
        ]
        clip_start = max(0, min(start for start, _ in segments) - FRAME_MARGIN_MS)
        clip_end = min(
            int(row["duration_milliseconds"]),
            max(end for _, end in segments) + FRAME_MARGIN_MS,
        )
        clip_duration = clip_end - clip_start
        if clip_duration < WINDOW_MS:
            _fail("E_CLIP_SPAN")
        candidate_windows: list[dict[str, int]] = []
        for center in centers:
            start = center - FRAME_MARGIN_MS - clip_start
            start = max(0, min(start, clip_duration - WINDOW_MS))
            candidate_windows.append({"start_ms": start, "end_ms": start + WINDOW_MS})
        if len(candidate_windows) != WINDOWS_PER_ITEM:
            _fail("E_WINDOW_COUNT")
        sample_segments = [
            {"start_ms": start - clip_start, "end_ms": end - clip_start}
            for start, end in segments
        ]
        projected = math.ceil((clip_duration / 1000.0) * DERIVED_BOUND_BPS / 8.0)
        projected_clip_bytes += projected
        plan_rows.append(
            {
                "selection_rank": index,
                "media_key": row["media_key"],
                "participant_key": row["participant_key"],
                "session_key": row["session_key"],
                "object_key": row["object_key"],
                "selection_hash": row["selection_hash"],
                "stratum": list(row["stratum"]),
                "source_locator": row["source_locator"],
                "manifest_display_size_bytes": row["size_bytes"],
                "duration_ms": row["duration_milliseconds"],
                "annotation_sha256s": row["annotation_sha256s"],
                "annotation_count": row["annotation_count"],
                "invalid_timing_count": row["invalid_timing_count"],
                "clip_source_start_ms": clip_start,
                "clip_source_end_ms": clip_end,
                "sample_segments_clip_ms": sample_segments,
                "candidate_windows_clip_ms": candidate_windows,
                "projected_clip_bound_bytes": projected,
            }
        )
    if projected_clip_bytes > DERIVED_CAP_BYTES:
        _fail("E_DERIVED_ADMISSION")
    plan = {
        "schema_version": PLAN_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "predecessor_receipt_sha256": PREDECESSOR_SHA256,
        "restricted_release_binding_sha256": release_digest,
        "original_selection_sha256": original_digest,
        "additional_selection_sha256": additional_digest,
        "combined_selection_sha256": combined_digest,
        "original_item_count": ORIGINAL_COUNT,
        "additional_item_count": ADDITIONAL_COUNT,
        "combined_item_count": ORIGINAL_COUNT + ADDITIONAL_COUNT,
        "additional_total_speech_seconds": ADDITIONAL_COUNT * SAMPLE_MS // 1000,
        "additional_candidate_window_count": ADDITIONAL_COUNT * WINDOWS_PER_ITEM,
        "combined_candidate_window_count": 137 + ADDITIONAL_COUNT * WINDOWS_PER_ITEM,
        "selection_used_model_output": False,
        "selection_opened_media": False,
        "selection_used_lexical_or_visual_content": False,
        "retention_deadline": RETENTION_DEADLINE,
        "items": plan_rows,
    }
    original_participant_set = set(original_participants)
    additional_participant_set = {row["participant_key"] for row in plan_rows}
    if (
        len(additional_participant_set) != ADDITIONAL_COUNT
        or original_participant_set & additional_participant_set
    ):
        _fail("E_SELECTION_GROUPING")
    total_source = sum(int(row["manifest_display_size_bytes"]) for row in plan_rows)
    max_source = max(int(row["manifest_display_size_bytes"]) for row in plan_rows)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "EXTENSION_SELECTION_FROZEN",
        "protocol_sha256": PROTOCOL_SHA256,
        "predecessor_receipt_sha256": PREDECESSOR_SHA256,
        "restricted_plan_sha256": _digest(plan),
        "additional_selection_sha256": additional_digest,
        "combined_selection_sha256": combined_digest,
        "eligible_unused_media_count": len(candidates),
        "eligible_unused_participant_count": len({row["participant_key"] for row in candidates}),
        "original_item_count": ORIGINAL_COUNT,
        "additional_item_count": ADDITIONAL_COUNT,
        "combined_item_count": ORIGINAL_COUNT + ADDITIONAL_COUNT,
        "all_additional_participants_distinct": True,
        "all_combined_participants_distinct": True,
        "additional_speech_seconds": ADDITIONAL_COUNT * SAMPLE_MS // 1000,
        "combined_speech_seconds": 1800,
        "additional_candidate_window_count": ADDITIONAL_COUNT * WINDOWS_PER_ITEM,
        "combined_candidate_window_count": 137 + ADDITIONAL_COUNT * WINDOWS_PER_ITEM,
        "selected_source_display_bytes_rounded_up_gib": math.ceil(total_source / 1024**3),
        "largest_selected_source_display_bytes_rounded_up_gib": math.ceil(max_source / 1024**3),
        "derived_clip_bound_bytes_rounded_up_gib": math.ceil(projected_clip_bytes / 1024**3),
        "derived_clip_cap_bytes": DERIVED_CAP_BYTES,
        "simultaneous_raw_cap_bytes": RAW_CAP_BYTES,
        "namespace_cap_bytes": NAMESPACE_CAP_BYTES,
        "free_space_floor_bytes": FREE_FLOOR_BYTES,
        "selection_used_model_output": False,
        "selection_opened_media": False,
        "selection_used_lexical_content": False,
        "selection_used_visual_content": False,
        "media_size_used_for_ranking": False,
        "post_failure_extension_disclosed": True,
        "maximum_additional_selection_attempts": 1,
        "further_expansion_allowed": False,
        "restricted_identifiers_exported": False,
        "exact_intervals_exported": False,
        "source_paths_or_filenames_exported": False,
        "small_cells_exported": False,
        "scientific_endpoint_opened": False,
    }
    return plan, receipt


def _public_guard(value: Any) -> None:
    encoded = _canonical(value).decode("utf-8")
    if PATH_TOKEN.search(encoded) or MEDIA_TOKEN.search(encoded):
        _fail("E_PUBLIC_PRIVACY")
    forbidden = {
        "media_key",
        "participant_key",
        "session_key",
        "object_key",
        "source_locator",
        "clip_source_start_ms",
        "clip_source_end_ms",
        "sample_segments_clip_ms",
        "candidate_windows_clip_ms",
        "annotation_sha256s",
    }
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if forbidden.intersection(current):
                _fail("E_PUBLIC_PRIVACY")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def execute() -> Mapping[str, Any]:
    if _sha256_file(PROTOCOL) != PROTOCOL_SHA256 or _sha256_file(PREDECESSOR) != PREDECESSOR_SHA256:
        _fail("E_IMMUTABLE_BINDING")
    protocol = _read_json(PROTOCOL)
    if (
        protocol.get("status") != "FROZEN_BEFORE_ADDITIONAL_ITEM_SELECTION_OR_MEDIA_OPEN"
        or protocol.get("sample", {}).get("additional_items") != ADDITIONAL_COUNT
        or protocol.get("sample", {}).get("no_further_expansion") is not True
    ):
        _fail("E_PROTOCOL")
    author = _load_module("childlens_extension_author_v18", AUTHOR)
    post = _load_module("childlens_extension_post_v18", POST)
    runtime = Path(author.discover_runtime_root()).resolve(strict=True)
    control_root = Path(post._quarantine_control_root(runtime)).resolve(strict=True)
    if (
        not _private_directory(runtime)
        or not _private_directory(control_root)
        or not _inside(runtime, control_root)
        or not _private_file(control_root / ".metadata_never_index")
    ):
        _fail("E_QUARANTINE")
    full_path = runtime / "post_acquisition_v1_2/frozen_v1_1_restricted_input.json"
    original_path = runtime / "post_acquisition_v1_2/restricted_measurement_manifest.json"
    if not _private_file(full_path) or not _private_file(original_path):
        _fail("E_MANIFEST")
    full = _read_json(full_path)
    original = _read_json(original_path)

    def annotation_loader(annotation_object: Mapping[str, Any]) -> tuple[list[tuple[float, float]], int, str]:
        locator = annotation_object.get("source_locator")
        digest = annotation_object.get("local_sha256")
        if (
            annotation_object.get("top_level_class") != "ANNOTATION"
            or not isinstance(locator, str)
            or not isinstance(digest, str)
            or not HEX64.fullmatch(digest)
        ):
            _fail("E_ANNOTATION_LINK")
        try:
            _, document = post._verify_file_inside(control_root, locator, digest)
            windows, invalid = post.extract_official_speech_windows(document)
        except Exception:
            _fail("E_ANNOTATION_LINK")
        return windows, int(invalid), digest

    plan, receipt = build_selection(full, original, annotation_loader=annotation_loader)
    extension_root = runtime / "provisional_calibration_v1/extension_v1_8"
    if extension_root.exists() and not _private_directory(extension_root):
        _fail("E_QUARANTINE")
    extension_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(extension_root, 0o700)
    plan_path = extension_root / "restricted_extension_plan.json"
    _write_once(plan_path, plan, private=True)
    _public_guard(receipt)
    _write_once(PUBLIC_RECEIPT, receipt, private=False)
    return receipt


def main() -> int:
    old_umask = os.umask(0o077)
    try:
        receipt = execute()
        print(
            json.dumps(
                {
                    "status": "ok",
                    "state": receipt["status"],
                    "additional_item_count": receipt["additional_item_count"],
                    "combined_item_count": receipt["combined_item_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    except ExtensionError as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
