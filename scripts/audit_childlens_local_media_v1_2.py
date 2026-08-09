#!/usr/bin/env python3
"""Fail-closed aggregate-only local media audit for ChildLens v1.2.

The restricted manifest, media paths, exact durations, speech windows, and
per-item findings remain in the quarantine process.  The returned receipt has
a fixed aggregate schema and never carries a path, filename, identifier,
timestamp, transcript, codec inventory, or item-level result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


VERSION = "childlens-local-media-audit-v1.2.1"
MANIFEST_VERSION = "childlens-restricted-measurement-manifest-v1.2.1"
REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_ITEMS = 18
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PROBE_BYTES = 1024 * 1024
MAX_PUBLIC_RECEIPT_BYTES = 8192
DEFAULT_CELL_SUPPRESSION_K = 5
MANIFEST_KEYS = frozenset({"schema_version", "pilot_selection_sha256", "items"})
ITEM_KEYS = frozenset(
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
WINDOW_KEYS = frozenset({"start_seconds", "end_seconds"})
DURATION_BASES = frozenset({"AUTHORITATIVE_MEDIA_DURATION", "ANNOTATION_MAX_END"})
SPEECH_PRESENCE_EXPECTATIONS = frozenset({"PRESENT", "ABSENT"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _reject_symlink_chain(root: Path, relative: Path) -> None:
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("E_SYMLINK")


def _safe_positive_number(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(code)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(code)
    return number


def _binary_metric(pass_count: int, total: int, k: int) -> dict[str, Any]:
    """Return a binary aggregate without disclosing a nonzero cell below k."""
    if total < 0 or pass_count < 0 or pass_count > total:
        raise ValueError("E_METRIC")
    fail_count = total - pass_count
    if total == 0:
        return {
            "status": "NO_ELIGIBLE_ITEMS",
            "pass_count": 0,
            "fail_count": 0,
            "cell_suppressed": False,
        }
    if fail_count == 0:
        return {
            "status": "ALL_PASS",
            "pass_count": total,
            "fail_count": 0,
            "cell_suppressed": False,
        }
    if pass_count == 0:
        return {
            "status": "NONE_PASS",
            "pass_count": 0,
            "fail_count": total,
            "cell_suppressed": False,
        }
    if min(pass_count, fail_count) < k:
        return {
            "status": "MIXED_SMALL_CELL_SUPPRESSED",
            "pass_count": None,
            "fail_count": None,
            "cell_suppressed": True,
        }
    return {
        "status": "MIXED_COUNTS_EXPORTED",
        "pass_count": pass_count,
        "fail_count": fail_count,
        "cell_suppressed": False,
    }


def _count_metric(count: int, k: int) -> dict[str, Any]:
    if count < 0:
        raise ValueError("E_METRIC")
    if 0 < count < k:
        return {"count": None, "status": "SMALL_CELL_SUPPRESSED", "cell_suppressed": True}
    return {"count": count, "status": "EXPORTED", "cell_suppressed": False}


def _parse_manifest(manifest_path: Path, quarantine_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not _is_inside(manifest_path, quarantine_root):
        raise ValueError("E_MANIFEST_OUTSIDE_QUARANTINE")
    relative_manifest = manifest_path.relative_to(quarantine_root)
    _reject_symlink_chain(quarantine_root, relative_manifest)
    if not manifest_path.is_file() or not stat.S_ISREG(manifest_path.stat().st_mode):
        raise ValueError("E_MANIFEST_FILE")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("E_MANIFEST_SIZE")
    if stat.S_IMODE(manifest_path.stat().st_mode) & 0o077:
        raise ValueError("E_MANIFEST_PERMISSIONS")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("E_MANIFEST_JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("E_MANIFEST_SCHEMA")
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError("E_MANIFEST_VERSION")
    selection_digest = manifest.get("pilot_selection_sha256")
    if not isinstance(selection_digest, str) or len(selection_digest) != 64:
        raise ValueError("E_SELECTION_DIGEST")
    try:
        int(selection_digest, 16)
    except ValueError as exc:
        raise ValueError("E_SELECTION_DIGEST") from exc
    items = manifest.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise ValueError("E_ITEM_COUNT")
    seen_keys: set[str] = set()
    parsed_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != ITEM_KEYS:
            raise ValueError("E_ITEM_SCHEMA")
        blinded_key = item.get("blinded_item_key")
        if not isinstance(blinded_key, str) or not blinded_key or blinded_key in seen_keys:
            raise ValueError("E_BLINDED_KEY")
        seen_keys.add(blinded_key)
        rel_value = item.get("media_relpath")
        if not isinstance(rel_value, str) or not rel_value or "\x00" in rel_value:
            raise ValueError("E_MEDIA_RELPATH")
        relative = Path(rel_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("E_MEDIA_RELPATH")
        _reject_symlink_chain(quarantine_root, relative)
        media_path = (quarantine_root / relative).resolve(strict=True)
        if not _is_inside(media_path, quarantine_root):
            raise ValueError("E_MEDIA_OUTSIDE_QUARANTINE")
        if not media_path.is_file() or not stat.S_ISREG(media_path.stat().st_mode):
            raise ValueError("E_MEDIA_FILE")
        if stat.S_IMODE(media_path.stat().st_mode) & 0o077:
            raise ValueError("E_MEDIA_PERMISSIONS")
        expected_size = item.get("expected_size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError("E_MEDIA_SIZE")
        expected_sha256 = item.get("expected_media_sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError("E_MEDIA_DIGEST")
        try:
            int(expected_sha256, 16)
        except ValueError as exc:
            raise ValueError("E_MEDIA_DIGEST") from exc
        reference_duration = _safe_positive_number(item.get("reference_duration_seconds"), "E_DURATION")
        reference_basis = item.get("reference_duration_basis")
        if reference_basis not in DURATION_BASES:
            raise ValueError("E_DURATION_BASIS")
        speech_presence_expectation = item.get("speech_presence_expectation")
        if speech_presence_expectation not in SPEECH_PRESENCE_EXPECTATIONS:
            raise ValueError("E_SPEECH_PRESENCE_EXPECTATION")
        windows = item.get("speech_windows")
        if not isinstance(windows, list):
            raise ValueError("E_WINDOWS")
        parsed_windows: list[tuple[float, float]] = []
        for window in windows:
            if not isinstance(window, dict) or set(window) != WINDOW_KEYS:
                raise ValueError("E_WINDOW_SCHEMA")
            start = window.get("start_seconds")
            end = window.get("end_seconds")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
            ):
                raise ValueError("E_WINDOW_TIME")
            start_number, end_number = float(start), float(end)
            if not math.isfinite(start_number) or not math.isfinite(end_number):
                raise ValueError("E_WINDOW_TIME")
            parsed_windows.append((start_number, end_number))
        parsed_items.append(
            {
                "media_path": media_path,
                "expected_size": expected_size,
                "expected_sha256": expected_sha256,
                "reference_duration": reference_duration,
                "reference_basis": reference_basis,
                "speech_presence_expectation": speech_presence_expectation,
                "windows": parsed_windows,
            }
        )
    return manifest, parsed_items


def _run_probe(ffprobe: Path, media_path: Path, timeout_seconds: int) -> dict[str, Any] | None:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-protocol_whitelist",
        "file,crypto,data,pipe",
        "-show_entries",
        "format=duration:stream=codec_type,duration",
        "-of",
        "json",
        str(media_path),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_seconds,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or len(completed.stdout) > MAX_PROBE_BYTES:
        return None
    try:
        value = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _probe_facts(probe: dict[str, Any] | None) -> tuple[bool, bool, bool, float | None, float | None]:
    if probe is None:
        return False, False, False, None, None
    streams = probe.get("streams")
    stream_rows = streams if isinstance(streams, list) else []
    video_present = any(
        isinstance(row, dict) and row.get("codec_type") == "video" for row in stream_rows
    )
    audio_rows = [
        row for row in stream_rows if isinstance(row, dict) and row.get("codec_type") == "audio"
    ]
    audio_present = bool(audio_rows)
    format_row = probe.get("format")
    duration: float | None = None
    if isinstance(format_row, dict):
        try:
            candidate = float(format_row.get("duration"))
            if math.isfinite(candidate) and candidate > 0:
                duration = candidate
        except (TypeError, ValueError):
            pass
    audio_duration: float | None = None
    for row in audio_rows:
        try:
            candidate = float(row.get("duration"))
            if math.isfinite(candidate) and candidate > 0:
                audio_duration = max(audio_duration or candidate, candidate)
        except (TypeError, ValueError):
            continue
    return True, video_present, audio_present, duration, audio_duration


def _run_full_decode(ffmpeg: Path, media_path: Path, timeout_seconds: int) -> bool:
    command = [
        str(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-protocol_whitelist",
        "file,crypto,data,pipe",
        "-i",
        str(media_path),
        "-map",
        "0:v?",
        "-map",
        "0:a?",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_seconds,
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def audit(
    quarantine_root_input: Path,
    manifest_input: Path,
    ffprobe: Path,
    ffmpeg: Path,
    *,
    timeout_seconds: int = 7200,
    cell_suppression_k: int = DEFAULT_CELL_SUPPRESSION_K,
) -> dict[str, Any]:
    if timeout_seconds < 1 or not 2 <= cell_suppression_k <= 10:
        raise ValueError("E_CONFIGURATION")
    if quarantine_root_input.is_symlink():
        raise ValueError("E_QUARANTINE_SYMLINK")
    quarantine_root = quarantine_root_input.resolve(strict=True)
    if quarantine_root == REPO_ROOT or REPO_ROOT in quarantine_root.parents:
        raise ValueError("E_QUARANTINE_INSIDE_REPOSITORY")
    root_stat = quarantine_root.stat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("E_QUARANTINE_DIRECTORY")
    if root_stat.st_uid != os.getuid() or stat.S_IMODE(root_stat.st_mode) & 0o077:
        raise ValueError("E_QUARANTINE_PERMISSIONS")
    manifest_path = manifest_input.resolve(strict=True)
    for executable, code in ((ffprobe, "E_FFPROBE"), (ffmpeg, "E_FFMPEG")):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(code)
    manifest, items = _parse_manifest(manifest_path, quarantine_root)

    integrity_match = probe_success = video_present = audio_present = 0
    duration_reference_valid = authoritative_duration_reference = audio_duration_consistent = 0
    duration_consistency_evidenced = corruption_absence_evidenced = 0
    decode_success = speech_presence_window_expectation_satisfied = 0
    media_with_windows = 0
    valid_windows = invalid_windows = total_windows = 0

    for item in items:
        media_path = item["media_path"]
        integrity_ok = (
            media_path.stat().st_size == item["expected_size"]
            and _sha256_file(media_path) == item["expected_sha256"]
        )
        integrity_match += int(integrity_ok)
        if not integrity_ok:
            total_windows += len(item["windows"])
            invalid_windows += len(item["windows"])
            continue
        probe = _run_probe(ffprobe, media_path, timeout_seconds)
        probed, video, audio, duration, audio_duration = _probe_facts(probe)
        probe_success += int(probed)
        video_present += int(video)
        audio_present += int(audio)
        reference_duration = item["reference_duration"]
        reference_basis = item["reference_basis"]
        authoritative_duration_reference += int(reference_basis == "AUTHORITATIVE_MEDIA_DURATION")
        if reference_basis == "AUTHORITATIVE_MEDIA_DURATION":
            tolerance = max(2.0, reference_duration * 0.02)
            duration_ok = duration is not None and abs(duration - reference_duration) <= tolerance
        else:
            duration_ok = duration is not None and reference_duration <= duration + 0.5
        duration_reference_valid += int(duration_ok)
        audio_duration_ok = (
            duration is not None
            and audio_duration is not None
            and abs(audio_duration - duration) <= max(2.0, duration * 0.02)
        )
        audio_duration_consistent += int(audio_duration_ok)
        decode_ok = _run_full_decode(ffmpeg, media_path, timeout_seconds)
        decode_success += int(decode_ok)
        duration_consistency_evidenced += int(duration_ok and audio_duration_ok)
        corruption_absence_evidenced += int(probed and decode_ok)

        windows = item["windows"]
        media_with_windows += int(bool(windows))
        total_windows += len(windows)
        valid_item_windows = 0
        for start, end in windows:
            is_valid = (
                duration is not None
                and start >= 0
                and end > start
                and end <= duration + max(0.5, duration * 0.005)
            )
            valid_item_windows += int(is_valid)
            valid_windows += int(is_valid)
            invalid_windows += int(not is_valid)
        expectation = item["speech_presence_expectation"]
        expectation_ok = (
            valid_item_windows >= 1
            if expectation == "PRESENT"
            else len(windows) == 0
        )
        speech_presence_window_expectation_satisfied += int(expectation_ok)

    item_count = len(items)
    metrics = {
        "restricted_manifest_file_integrity_match": _binary_metric(
            integrity_match, item_count, cell_suppression_k
        ),
        "probe_success": _binary_metric(probe_success, item_count, cell_suppression_k),
        "video_stream_present": _binary_metric(video_present, item_count, cell_suppression_k),
        "audio_stream_present": _binary_metric(audio_present, item_count, cell_suppression_k),
        "duration_reference_structurally_valid": _binary_metric(
            duration_reference_valid, item_count, cell_suppression_k
        ),
        "authoritative_duration_reference_coverage": _binary_metric(
            authoritative_duration_reference, item_count, cell_suppression_k
        ),
        "audio_duration_consistent": _binary_metric(
            audio_duration_consistent, item_count, cell_suppression_k
        ),
        "duration_consistency_evidenced": _binary_metric(
            duration_consistency_evidenced, item_count, cell_suppression_k
        ),
        "full_decode_success": _binary_metric(decode_success, item_count, cell_suppression_k),
        "corruption_absence_evidenced": _binary_metric(
            corruption_absence_evidenced, item_count, cell_suppression_k
        ),
        "speech_presence_window_expectation_satisfied": _binary_metric(
            speech_presence_window_expectation_satisfied, item_count, cell_suppression_k
        ),
        "media_with_speech_windows": _binary_metric(
            media_with_windows, item_count, cell_suppression_k
        ),
    }
    if invalid_windows == 0:
        window_validity = _binary_metric(total_windows, total_windows, cell_suppression_k)
    else:
        window_validity = _binary_metric(valid_windows, total_windows, cell_suppression_k)
    all_required_pass = all(
        value["status"] == "ALL_PASS"
        for key, value in metrics.items()
        if key not in {"media_with_speech_windows", "authoritative_duration_reference_coverage"}
    ) and invalid_windows == 0
    receipt: dict[str, Any] = {
        "schema_version": VERSION,
        "status": "ALL_STRUCTURAL_CHECKS_PASS" if all_required_pass else "REVIEW_REQUIRED",
        "scope": "LOCAL_RESTRICTED_MEDIA_AGGREGATE_ONLY",
        "pilot_selection_sha256": manifest["pilot_selection_sha256"],
        "restricted_manifest_sha256": _canonical_sha256(manifest),
        "media_count": item_count,
        "cell_suppression_k": cell_suppression_k,
        "container_audio_decode_metrics": metrics,
        "speech_window_structure": {
            "window_total": _count_metric(total_windows, cell_suppression_k),
            "window_validity": window_validity,
        },
        "quarantine_controls": {
            "outside_repository": True,
            "owner_only_root_permissions": True,
            "manifest_owner_only_permissions": True,
            "symlink_components_rejected": True,
        },
        "execution_controls": {
            "network_protocols_allowed": False,
            "subprocess_output_discarded_or_parsed_in_process": True,
            "per_item_results_exported": False,
            "source_media_modified": False,
        },
        "privacy_export": {
            "paths_or_filenames_exported": False,
            "identifiers_exported": False,
            "exact_timestamps_exported": False,
            "transcript_or_lexical_content_exported": False,
            "frames_or_audio_exported": False,
            "small_nonzero_cells_exported": False,
        },
        "scientific_boundary": {
            "learner_or_tokenizer_trained": False,
            "causal_arm_run": False,
            "automated_measurements_are_gold": False,
            "human_validation_required": True,
            "instrument_output_may_enter_learner": False,
        },
        "toolchain": {
            "ffprobe_sha256": _sha256_file(ffprobe),
            "ffmpeg_sha256": _sha256_file(ffmpeg),
        },
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_PUBLIC_RECEIPT_BYTES:
        raise ValueError("E_OUTPUT_SIZE")
    return receipt


def _fixed_error(code: str) -> dict[str, str]:
    return {"schema_version": VERSION, "status": "error", "error_code": code}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarantine-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--cell-suppression-k", type=int, default=DEFAULT_CELL_SUPPRESSION_K)
    args = parser.parse_args()
    try:
        receipt = audit(
            Path(args.quarantine_root),
            Path(args.manifest),
            Path(args.ffprobe),
            Path(args.ffmpeg),
            timeout_seconds=args.timeout_seconds,
            cell_suppression_k=args.cell_suppression_k,
        )
    except Exception as exc:
        code = str(exc) if str(exc).startswith("E_") else "E_INTERNAL"
        print(json.dumps(_fixed_error(code), sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
