#!/usr/bin/env python3
"""Run the versioned ten-participant ChildLens measurement expansion v3.

Restricted identifiers, paths, locators, intervals, media, transcripts, and
row-level results remain in the owner-private ChildLens quarantine.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_expansion_v3.json"
PUBLIC_ROOT = ROOT / "output/childlens_alignment_bridge_expansion_v3"
SELECTION_RECEIPT = PUBLIC_ROOT / "selection_freeze_receipt.json"
PRIVATE_RELATIVE = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_expansion_v3"
)
PRIVATE_PLAN = PRIVATE_RELATIVE / "restricted_expansion_plan.json"

PROTOCOL_SHA256 = "787f64eba92a6a2f206e09a447b2f595691230349ba8f17c800faa0e50108f02"
RELEASE_MANIFEST_SHA256 = (
    "a603239d2c96946662390c4dc45c543f1652055e6edd21e156a6f351f78db22a"
)
V1_SPLIT_SHA256 = "926391535972830289315b95e6b4e4893a3e82aeac03b371ab3212ca324b906a"
V2_RESULT_SHA256 = "ab61da2ecdedaf33809591df427afd913bea45763ce65656938d2d463814c44f"
EXPANSION_COUNT = 10
SAMPLE_MILLISECONDS = 60_000
CLIP_MARGIN_MILLISECONDS = 5_000
DERIVED_BOUND_BITS_PER_SECOND = 3_000_000
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SPEECH_RE = re.compile(r"speech|talk|speak|vocal", re.IGNORECASE)
PATH_TOKEN = re.compile(r"(?i)(?:/users/|file://|\\users\\)")
MEDIA_TOKEN = re.compile(r"(?i)\b\S+\.(?:mp4|mov|mkv|avi|webm|wav|m4a)\b")

sys.path.insert(0, str(ROOT))
from babyworld_lite.childlens_alignment_bridge_v3.selection import (  # noqa: E402
    canonical_bytes,
    coalesce_milliseconds,
    deterministic_participant_selection,
    digest,
    map_union_range,
)


class ExpansionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise ExpansionError(code)


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
    except OSError:
        _fail("E_FILE")
    return value.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("E_FILE")
        return json.loads(path.read_text(encoding="utf-8"))
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


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_once(path: Path, value: Any, *, private: bool) -> None:
    payload = canonical_bytes(value) + b"\n"
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
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
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


def _discover_runtime() -> Path:
    search = ROOT.parent
    candidates: list[Path] = []
    for hidden in search.iterdir():
        if (
            hidden.name.startswith(".")
            and "childlens" in hidden.name.casefold()
            and _private_directory(hidden)
        ):
            for manifest in hidden.rglob("restricted_manifest/preselection_manifest.json"):
                if _private_file(manifest):
                    candidates.append(manifest.parent)
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) != 1:
        _fail("E_RUNTIME_DISCOVERY")
    runtime = unique[0]
    if not _private_directory(runtime) or ROOT == runtime or _inside(runtime, ROOT):
        _fail("E_RUNTIME_CONTROL")
    if not _private_file(runtime / ".metadata_never_index"):
        _fail("E_RUNTIME_CONTROL")
    return runtime


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _timing_number(
    row: Mapping[str, Any],
    names: Sequence[str],
) -> float | None:
    accepted = set(names)
    for key, value in row.items():
        if _normalise_key(str(key)) in accepted:
            number = _finite_number(value)
            if number is not None:
                return number
    return None


def _speech_windows(
    document: Mapping[str, Any],
) -> tuple[list[tuple[float, float]], int]:
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        _fail("E_ANNOTATION_SCHEMA")
    windows: list[tuple[float, float]] = []
    invalid = 0
    for raw in annotations:
        if not isinstance(raw, Mapping):
            _fail("E_ANNOTATION_SCHEMA")
        event_values = [
            value
            for key, value in raw.items()
            if _normalise_key(str(key))
            in {"eventid", "event_id", "category", "label", "type"}
            and isinstance(value, str)
        ]
        if not any(SPEECH_RE.search(value) for value in event_values):
            continue
        start = _timing_number(
            raw,
            ("start", "start_time", "starttime", "onset", "time"),
        )
        end = _timing_number(raw, ("end", "end_time", "endtime", "offset"))
        duration = _timing_number(raw, ("duration",))
        if end is None and start is not None and duration is not None:
            end = start + duration
        if start is None or end is None or start < 0 or end <= start:
            invalid += 1
            continue
        windows.append((start, end))
    return windows, invalid


def _safe_annotation(
    runtime: Path,
    locator: str,
    expected_sha256: str,
) -> Mapping[str, Any]:
    path = Path(locator).resolve()
    control_root = runtime.parent.resolve()
    if (
        not _inside(path, control_root)
        or not _private_file(path)
        or not HEX64.fullmatch(expected_sha256)
        or _sha256_file(path) != expected_sha256
    ):
        _fail("E_ANNOTATION_BINDING")
    value = _read_json(path)
    if not isinstance(value, Mapping):
        _fail("E_ANNOTATION_SCHEMA")
    return value


def _public_guard(value: Any) -> None:
    encoded = canonical_bytes(value).decode("utf-8")
    if PATH_TOKEN.search(encoded) or MEDIA_TOKEN.search(encoded):
        _fail("E_PUBLIC_PRIVACY")
    forbidden = {
        "participant_key",
        "media_key",
        "session_key",
        "object_key",
        "source_locator",
        "annotation_locator",
        "sample_segments_source_ms",
        "clip_source_start_ms",
        "clip_source_end_ms",
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


def _config() -> Mapping[str, Any]:
    if _sha256_file(CONFIG) != PROTOCOL_SHA256:
        _fail("E_PROTOCOL_HASH")
    value = _read_json(CONFIG)
    if (
        value.get("schema_version")
        != "childlens-alignment-bridge-measurement-expansion-v3.0.0"
        or value.get("status") != "FROZEN_BEFORE_EXPANSION_SELECTION_OR_MEDIA_OPEN"
        or value.get("scope", {}).get("additional_participant_distinct_recordings")
        != EXPANSION_COUNT
        or value.get("scope", {}).get("locked_alignment_allowed") is not False
        or value.get("scope", {}).get("further_expansion_under_this_protocol")
        is not False
    ):
        _fail("E_PROTOCOL")
    return value


def freeze_selection() -> Mapping[str, Any]:
    _config()
    runtime = _discover_runtime()
    release_path = runtime / "preselection_manifest.json"
    split_path = (
        runtime
        / "provisional_calibration_v1/childlens_alignment_bridge_v1/"
        "restricted_split_manifest.json"
    )
    v2_result_path = (
        runtime
        / "provisional_calibration_v1/childlens_alignment_bridge_remediation_v2/"
        "restricted_remediation_result.json"
    )
    if (
        _sha256_file(release_path) != RELEASE_MANIFEST_SHA256
        or _sha256_file(split_path) != V1_SPLIT_SHA256
        or _sha256_file(v2_result_path) != V2_RESULT_SHA256
    ):
        _fail("E_PRIOR_BINDING")
    release = _read_json(release_path)
    split = _read_json(split_path)
    if (
        release.get("manifest_schema_version")
        != "childlens-restricted-canonical-manifest-v1.1.0"
        or split.get("schema_version")
        != "childlens-alignment-bridge-restricted-split-v1.0.0"
        or not isinstance(split.get("items"), list)
        or len(split["items"]) != 30
    ):
        _fail("E_MANIFEST")
    records = release.get("records")
    if not isinstance(records, Mapping):
        _fail("E_MANIFEST")
    for key in ("objects", "groupings", "linkages", "selection_metadata"):
        if not isinstance(records.get(key), list):
            _fail("E_MANIFEST")
    objects = {
        row["object_key"]: row
        for row in records["objects"]
        if isinstance(row, Mapping) and isinstance(row.get("object_key"), str)
    }
    groupings = {
        row["media_key"]: row
        for row in records["groupings"]
        if isinstance(row, Mapping) and isinstance(row.get("media_key"), str)
    }
    metadata = {
        row["media_key"]: row
        for row in records["selection_metadata"]
        if isinstance(row, Mapping) and isinstance(row.get("media_key"), str)
    }
    annotation_links: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records["linkages"]:
        if isinstance(row, Mapping) and isinstance(row.get("linked_media_key"), str):
            annotation_links[str(row["linked_media_key"])].append(row)
    prior_participants = {
        row.get("participant_key")
        for row in split["items"]
        if isinstance(row, Mapping)
    }
    if (
        None in prior_participants
        or len(prior_participants) != 30
        or sum(row.get("split") == "locked" for row in split["items"]) != 22
    ):
        _fail("E_PRIOR_SPLIT")
    candidates: list[dict[str, Any]] = []
    for media_key, raw in metadata.items():
        grouping = groupings.get(media_key)
        media_object = objects.get(raw.get("object_key"))
        links = annotation_links.get(media_key, [])
        if (
            not isinstance(grouping, Mapping)
            or grouping.get("participant_key") in prior_participants
            or raw.get("speech_presence_bin") != "PRESENT"
            or not isinstance(media_object, Mapping)
            or media_object.get("top_level_class") != "VIDEO"
            or media_object.get("available_for_selective_copy") is not True
            or not isinstance(media_object.get("source_locator"), str)
            or type(media_object.get("size_bytes")) is not int
            or media_object["size_bytes"] <= 0
            or type(raw.get("duration_milliseconds")) is not int
            or raw["duration_milliseconds"] <= SAMPLE_MILLISECONDS
            or not links
        ):
            continue
        all_windows: list[tuple[float, float]] = []
        annotation_rows: list[dict[str, str]] = []
        invalid_timing = 0
        for link in links:
            annotation_object = objects.get(link.get("object_key"))
            if (
                not isinstance(annotation_object, Mapping)
                or annotation_object.get("top_level_class") != "ANNOTATION"
                or annotation_object.get("available_for_selective_copy") is not True
                or not isinstance(annotation_object.get("source_locator"), str)
                or not isinstance(annotation_object.get("local_sha256"), str)
            ):
                _fail("E_ANNOTATION_LINK")
            document = _safe_annotation(
                runtime,
                str(annotation_object["source_locator"]),
                str(annotation_object["local_sha256"]),
            )
            windows, invalid = _speech_windows(document)
            all_windows.extend(windows)
            invalid_timing += invalid
            annotation_rows.append(
                {
                    "locator": str(annotation_object["source_locator"]),
                    "sha256": str(annotation_object["local_sha256"]),
                }
            )
        coalesced = coalesce_milliseconds(
            all_windows,
            int(raw["duration_milliseconds"]),
        )
        speech_milliseconds = sum(end - start for start, end in coalesced)
        if speech_milliseconds < SAMPLE_MILLISECONDS:
            continue
        candidates.append(
            {
                **dict(raw),
                **dict(grouping),
                "source_locator": str(media_object["source_locator"]),
                "size_bytes": int(media_object["size_bytes"]),
                "official_windows_ms": coalesced,
                "annotation_rows": annotation_rows,
                "invalid_timing_count": invalid_timing,
            }
        )
    release_binding = digest(
        {
            "release": release.get("release"),
            "canonical_restricted_manifest_sha256": release.get(
                "canonical_restricted_manifest_sha256"
            ),
        }
    )
    selected = deterministic_participant_selection(
        candidates,
        count=EXPANSION_COUNT,
        release_binding=release_binding,
        protocol_sha256=PROTOCOL_SHA256,
    )
    plan_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, 1):
        available = sum(end - start for start, end in row["official_windows_ms"])
        offset = int(
            hashlib.sha256(
                (
                    PROTOCOL_SHA256
                    + "|speech-block|"
                    + row["media_key"]
                ).encode("utf-8")
            ).hexdigest(),
            16,
        ) % (available - SAMPLE_MILLISECONDS + 1)
        segments = map_union_range(
            row["official_windows_ms"],
            offset,
            SAMPLE_MILLISECONDS,
        )
        clip_start = max(0, min(start for start, _ in segments) - CLIP_MARGIN_MILLISECONDS)
        clip_end = min(
            int(row["duration_milliseconds"]),
            max(end for _, end in segments) + CLIP_MARGIN_MILLISECONDS,
        )
        clip_duration = clip_end - clip_start
        projected_clip_bytes = math.ceil(
            clip_duration
            / 1000.0
            * DERIVED_BOUND_BITS_PER_SECOND
            / 8.0
        )
        plan_rows.append(
            {
                "selection_rank": rank,
                "selection_hash": row["selection_hash"],
                "stratum": list(row["stratum"]),
                "participant_key": row["participant_key"],
                "media_key": row["media_key"],
                "session_key": row["session_key"],
                "object_key": row["object_key"],
                "source_locator": row["source_locator"],
                "manifest_display_size_bytes": row["size_bytes"],
                "duration_milliseconds": row["duration_milliseconds"],
                "annotation_rows": row["annotation_rows"],
                "invalid_timing_count": row["invalid_timing_count"],
                "sample_segments_source_ms": [
                    {"start_ms": start, "end_ms": end}
                    for start, end in segments
                ],
                "clip_source_start_ms": clip_start,
                "clip_source_end_ms": clip_end,
                "projected_clip_bytes": projected_clip_bytes,
            }
        )
    selected_participants = {row["participant_key"] for row in plan_rows}
    if (
        len(plan_rows) != EXPANSION_COUNT
        or len(selected_participants) != EXPANSION_COUNT
        or selected_participants & prior_participants
    ):
        _fail("E_SELECTION_GROUPING")
    selection_identity = [
        {
            "rank": row["selection_rank"],
            "selection_hash": row["selection_hash"],
            "participant_key": row["participant_key"],
            "media_key": row["media_key"],
        }
        for row in plan_rows
    ]
    plan = {
        "schema_version": "childlens-alignment-bridge-expansion-plan-v3.0.0",
        "protocol_sha256": PROTOCOL_SHA256,
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "v1_split_sha256": V1_SPLIT_SHA256,
        "v2_result_sha256": V2_RESULT_SHA256,
        "release_binding_sha256": release_binding,
        "selection_sha256": digest(selection_identity),
        "selection_used_media_or_model_outcomes": False,
        "locked_participant_count_excluded": 22,
        "prior_participant_count_excluded": 30,
        "items": plan_rows,
    }
    private_path = runtime / PRIVATE_PLAN
    _write_once(private_path, plan, private=True)
    total_source = sum(row["manifest_display_size_bytes"] for row in plan_rows)
    total_projected = sum(row["projected_clip_bytes"] for row in plan_rows)
    receipt = {
        "schema_version": "childlens-alignment-bridge-expansion-selection-v3.0.0",
        "status": "FROZEN_BEFORE_MEDIA_ACQUISITION_OR_MEASUREMENT",
        "protocol_sha256": PROTOCOL_SHA256,
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "v1_split_sha256": V1_SPLIT_SHA256,
        "v2_result_sha256": V2_RESULT_SHA256,
        "restricted_plan_sha256": _sha256_file(private_path),
        "selection_sha256": plan["selection_sha256"],
        "eligible_unused_recording_count": len(candidates),
        "eligible_unused_participant_count": len(
            {row["participant_key"] for row in candidates}
        ),
        "selected_recording_count": EXPANSION_COUNT,
        "selected_participant_count": EXPANSION_COUNT,
        "all_selected_participants_distinct": True,
        "all_prior_30_participants_excluded": True,
        "all_prior_22_locked_participants_excluded": True,
        "locked_outcomes_loaded_or_evaluated": 0,
        "selection_used_media_content": False,
        "selection_used_model_output": False,
        "selection_used_lexical_text": False,
        "released_timing_used_for_signal_eligibility_and_sampling_only": True,
        "sample_speech_seconds": EXPANSION_COUNT * 60,
        "source_display_bytes_rounded_up_gib": math.ceil(total_source / 1024**3),
        "largest_source_display_bytes_rounded_up_gib": math.ceil(
            max(row["manifest_display_size_bytes"] for row in plan_rows)
            / 1024**3
        ),
        "projected_retained_clip_bytes_rounded_up_gib": math.ceil(
            total_projected / 1024**3
        ),
        "external_volume_used": False,
        "further_expansion_allowed": False,
        "restricted_values_exported": False,
    }
    _public_guard(receipt)
    _write_once(SELECTION_RECEIPT, receipt, private=False)
    return receipt


def locked_status() -> Mapping[str, Any]:
    return {
        "locked_evaluation_authorized": False,
        "locked_rows_loaded_or_evaluated": 0,
        "reason": "EXPANSION_DEVELOPMENT_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze-selection", "locked-status"))
    args = parser.parse_args()
    old_umask = os.umask(0o077)
    try:
        result = (
            freeze_selection()
            if args.command == "freeze-selection"
            else locked_status()
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except ExpansionError as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
