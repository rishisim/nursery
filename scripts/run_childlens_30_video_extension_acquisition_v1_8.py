#!/usr/bin/env python3
"""Acquire and derive the frozen ChildLens v1.8 extension sequentially.

This zero-argument controller retrieves the read-only Keeper token from the
fixed macOS Keychain slot, authenticates through the existing native Seafile
client, and handles one selected source object at a time. Exact locators,
identifiers, timestamps, hashes, and filenames remain in owner-private
quarantine. Only an aggregate receipt is written to the repository.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "scripts/childlens_author_audit_v1_3.py"
LAUNCHER = ROOT / "scripts/launch_childlens_native_transfer_v1_2.py"
TRANSFER = ROOT / "scripts/childlens_native_transfer_v1_2.py"
SELECTION_RECEIPT = (
    ROOT
    / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "selection_and_admission_receipt.json"
)
PUBLIC_RECEIPT = (
    ROOT
    / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
    / "acquisition_and_clipping_receipt.json"
)
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
FFPROBE = Path("/opt/homebrew/bin/ffprobe")

VERSION = "nursery-childlens-30-video-extension-acquisition-v1.8.0"
PLAN_SCHEMA = "nursery-childlens-30-video-extension-restricted-plan-v1.8.0"
CHECKPOINT_SCHEMA = "nursery-childlens-30-video-extension-transfer-checkpoint-v1.8.0"
PUBLIC_SCHEMA = "nursery-childlens-30-video-extension-acquisition-receipt-v1.8.0"
SELECTION_RECEIPT_SHA256 = "85399959c18775c8f9588879075f9d91e166cf73c92697aa3eeeb830b774d9b1"
RESTRICTED_PLAN_SHA256 = "06eef6223908237e4e898a22caf46eea00f00acccbe3b8e304b57daa16339bed"
PROTOCOL_SHA256 = "e8d7ca48ed888f990730861e8f5a97274743dfc0e08befcdf40a508356c2aa56"
ITEM_COUNT = 15
RAW_CAP_BYTES = 20 * 1024**3
DERIVED_CAP_BYTES = 4 * 1024**3
NAMESPACE_CAP_BYTES = 73 * 1024**3
FREE_FLOOR_BYTES = 50 * 1024**3
CHUNK_BYTES = 4 * 1024**2
MAX_SAFE_JSON_BYTES = 128 * 1024**2
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PATH_TOKEN = re.compile(r"(?i)(?:/users/|file://|\\\\users\\\\)")
MEDIA_TOKEN = re.compile(r"(?i)\b\S+\.(?:mp4|mov|mkv|avi|webm|wav|m4a)\b")


class AcquisitionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise AcquisitionError(code)


def _load(name: str, path: Path) -> Any:
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
        _fail("E_FILE_HASH")
    return digest.hexdigest()


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


def _control_root(runtime: Path) -> Path:
    candidate = runtime.resolve(strict=True)
    for _ in range(6):
        sentinel = candidate / ".metadata_never_index"
        if (
            _private_directory(candidate)
            and _private_file(sentinel)
            and candidate != ROOT
            and not _inside(candidate, ROOT)
        ):
            return candidate
        parent = candidate.parent
        if (
            parent == candidate
            or parent == ROOT
            or _inside(parent, ROOT)
            or not _private_directory(parent)
        ):
            break
        candidate = parent
    _fail("E_QUARANTINE")


def _read_json(path: Path) -> Any:
    try:
        if not _private_file(path) and _inside(path.resolve(), ROOT):
            if not path.is_file() or path.is_symlink():
                _fail("E_JSON")
        payload = path.read_bytes()
        if not payload or len(payload) > MAX_SAFE_JSON_BYTES:
            _fail("E_JSON")
        return json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_JSON")


def _write_atomic(path: Path, value: Any, *, private: bool, immutable: bool = False) -> None:
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    if private:
        os.chmod(path.parent, 0o700)
    if immutable and path.exists():
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


def _run_quiet(command: list[str], *, timeout: int) -> None:
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            close_fds=True,
            timeout=timeout,
            env={
                "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
                "LANG": "C",
                "LC_ALL": "C",
            },
        )
    except (OSError, subprocess.SubprocessError):
        _fail("E_LOCAL_TOOL")
    if completed.returncode != 0:
        _fail("E_LOCAL_TOOL")


def _probe(path: Path) -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            [
                str(FFPROBE),
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            close_fds=True,
            timeout=300,
            env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        _fail("E_PROBE")
    if completed.returncode != 0 or len(completed.stdout) > 1024**2:
        _fail("E_PROBE")
    try:
        value = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError):
        _fail("E_PROBE")
    if not isinstance(value, Mapping):
        _fail("E_PROBE")
    return value


def _duration_and_streams(probe: Mapping[str, Any]) -> tuple[float, set[str]]:
    format_row = probe.get("format")
    streams = probe.get("streams")
    if not isinstance(format_row, Mapping) or not isinstance(streams, list):
        _fail("E_PROBE")
    try:
        duration = float(format_row["duration"])
    except (KeyError, TypeError, ValueError):
        _fail("E_PROBE")
    kinds = {
        str(row.get("codec_type"))
        for row in streams
        if isinstance(row, Mapping) and isinstance(row.get("codec_type"), str)
    }
    if not math.isfinite(duration) or duration <= 0 or not {"video", "audio"}.issubset(kinds):
        _fail("E_PROBE")
    return duration, kinds


def _namespace_bytes(root: Path) -> int:
    total = 0
    try:
        for directory, directories, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            directories[:] = [
                name for name in directories if not (parent / name).is_symlink()
            ]
            for name in files:
                member = parent / name
                info = member.lstat()
                if stat.S_ISLNK(info.st_mode):
                    continue
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
    except OSError:
        _fail("E_QUARANTINE")
    return total


def _capacity(root: Path, required: int) -> None:
    usage = shutil.disk_usage(root)
    if (
        required < 0
        or required > RAW_CAP_BYTES
        or usage.free - required < FREE_FLOOR_BYTES
        or _namespace_bytes(root) + required > NAMESPACE_CAP_BYTES
    ):
        _fail("E_CAPACITY")


def _validate_plan(plan: Any, selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (
        not isinstance(plan, Mapping)
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("protocol_sha256") != PROTOCOL_SHA256
        or plan.get("additional_selection_sha256") != selection.get("additional_selection_sha256")
        or plan.get("combined_selection_sha256") != selection.get("combined_selection_sha256")
        or plan.get("additional_item_count") != ITEM_COUNT
        or plan.get("additional_total_speech_seconds") != 900
        or plan.get("additional_candidate_window_count") != 135
        or not isinstance(plan.get("items"), list)
        or len(plan["items"]) != ITEM_COUNT
    ):
        _fail("E_PLAN")
    rows: list[dict[str, Any]] = []
    ranks: list[int] = []
    participants: set[str] = set()
    for raw in plan["items"]:
        if not isinstance(raw, Mapping):
            _fail("E_PLAN")
        row = dict(raw)
        if (
            type(row.get("selection_rank")) is not int
            or not isinstance(row.get("participant_key"), str)
            or row["participant_key"] in participants
            or not isinstance(row.get("source_locator"), str)
            or type(row.get("manifest_display_size_bytes")) is not int
            or row["manifest_display_size_bytes"] <= 0
            or type(row.get("clip_source_start_ms")) is not int
            or type(row.get("clip_source_end_ms")) is not int
            or row["clip_source_end_ms"] <= row["clip_source_start_ms"]
            or not isinstance(row.get("sample_segments_clip_ms"), list)
            or not isinstance(row.get("candidate_windows_clip_ms"), list)
            or len(row["candidate_windows_clip_ms"]) != 9
        ):
            _fail("E_PLAN")
        sample_ms = sum(
            int(segment["end_ms"]) - int(segment["start_ms"])
            for segment in row["sample_segments_clip_ms"]
        )
        if sample_ms != 60_000:
            _fail("E_PLAN")
        ranks.append(row["selection_rank"])
        participants.add(row["participant_key"])
        rows.append(row)
    if ranks != list(range(1, ITEM_COUNT + 1)):
        _fail("E_PLAN")
    return rows


def _checkpoint_template(plan: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol_sha256": PROTOCOL_SHA256,
        "restricted_plan_sha256": RESTRICTED_PLAN_SHA256,
        "additional_selection_sha256": plan["additional_selection_sha256"],
        "combined_selection_sha256": plan["combined_selection_sha256"],
        "status": "IN_PROGRESS",
        "items": [
            {
                "selection_rank": row["selection_rank"],
                "object_key": row["object_key"],
                "source_exact_bytes": None,
                "source_sha256": None,
                "clip_relative_path": None,
                "clip_bytes": None,
                "clip_sha256": None,
                "clip_duration_seconds": None,
                "status": "PENDING",
            }
            for row in rows
        ],
    }


def _validate_checkpoint(
    checkpoint: Any, template: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        _fail("E_CHECKPOINT")
    expected = {
        "schema_version",
        "protocol_sha256",
        "restricted_plan_sha256",
        "additional_selection_sha256",
        "combined_selection_sha256",
        "status",
        "items",
    }
    if set(checkpoint) != expected or any(
        checkpoint.get(key) != template.get(key)
        for key in (
            "schema_version",
            "protocol_sha256",
            "restricted_plan_sha256",
            "additional_selection_sha256",
            "combined_selection_sha256",
        )
    ):
        _fail("E_CHECKPOINT")
    items = checkpoint.get("items")
    if not isinstance(items, list) or len(items) != ITEM_COUNT:
        _fail("E_CHECKPOINT")
    for expected_item, item in zip(template["items"], items):
        if (
            not isinstance(item, Mapping)
            or item.get("selection_rank") != expected_item["selection_rank"]
            or item.get("object_key") != expected_item["object_key"]
            or item.get("status") not in {"PENDING", "COMPLETE"}
        ):
            _fail("E_CHECKPOINT")
    return dict(checkpoint)


def _verify_completed(root: Path, item: Mapping[str, Any]) -> None:
    relative = item.get("clip_relative_path")
    digest = item.get("clip_sha256")
    size = item.get("clip_bytes")
    if (
        not isinstance(relative, str)
        or not relative.startswith("provisional_calibration_v1/extension_v1_8/clips/")
        or ".." in Path(relative).parts
        or not isinstance(digest, str)
        or not HEX64.fullmatch(digest)
        or type(size) is not int
        or size <= 0
    ):
        _fail("E_CHECKPOINT")
    clip = root / relative
    if (
        not _inside(clip.resolve(strict=True), root)
        or not _private_file(clip)
        or clip.stat().st_size != size
        or _sha256_file(clip) != digest
    ):
        _fail("E_CHECKPOINT")
    _duration_and_streams(_probe(clip))


def _public_guard(value: Any) -> None:
    text = _canonical(value).decode("utf-8")
    if PATH_TOKEN.search(text) or MEDIA_TOKEN.search(text):
        _fail("E_PUBLIC_PRIVACY")
    forbidden = {
        "object_key",
        "participant_key",
        "source_locator",
        "source_sha256",
        "clip_sha256",
        "clip_relative_path",
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
    if (
        _sha256_file(SELECTION_RECEIPT) != SELECTION_RECEIPT_SHA256
        or not FFMPEG.is_file()
        or not FFPROBE.is_file()
    ):
        _fail("E_PUBLIC_BINDING")
    selection = _read_json(SELECTION_RECEIPT)
    if selection.get("status") != "EXTENSION_SELECTION_FROZEN":
        _fail("E_PUBLIC_BINDING")
    author = _load("childlens_extension_acquisition_author_v18", AUTHOR)
    launcher = _load("childlens_extension_acquisition_launcher_v18", LAUNCHER)
    transfer = _load("childlens_extension_acquisition_transfer_v18", TRANSFER)
    root = Path(author.discover_runtime_root()).resolve(strict=True)
    if not _private_directory(root) or root == ROOT or _inside(root, ROOT):
        _fail("E_QUARANTINE")
    control_root = _control_root(root)
    if not _inside(root, control_root):
        _fail("E_QUARANTINE")
    namespace = root / "provisional_calibration_v1/extension_v1_8"
    if not _private_directory(namespace):
        _fail("E_QUARANTINE")
    plan_path = namespace / "restricted_extension_plan.json"
    if not _private_file(plan_path) or _digest(_read_json(plan_path)) != RESTRICTED_PLAN_SHA256:
        _fail("E_PLAN")
    plan = _read_json(plan_path)
    rows = _validate_plan(plan, selection)
    clips = namespace / "clips"
    transient = namespace / "transient"
    for directory in (clips, transient):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
        if not _private_directory(directory):
            _fail("E_QUARANTINE")
    transient_entries = list(transient.iterdir())
    if any(
        entry.name != "source.bin"
        or entry.is_symlink()
        or not _private_file(entry)
        for entry in transient_entries
    ) or len(transient_entries) > 1:
        _fail("E_TRANSIENT_NOT_RESUMABLE")
    checkpoint_path = namespace / "restricted_transfer_checkpoint.json"
    template = _checkpoint_template(plan, rows)
    checkpoint = (
        _validate_checkpoint(_read_json(checkpoint_path), template)
        if checkpoint_path.exists()
        else template
    )
    for item in checkpoint["items"]:
        if item["status"] == "COMPLETE":
            _verify_completed(root, item)
    incomplete = [
        (row, item)
        for row, item in zip(rows, checkpoint["items"])
        if item["status"] != "COMPLETE"
    ]
    secrets_store = launcher.MacOSKeychain()
    token = secrets_store.read()
    if not launcher.TOKEN_PATTERN.fullmatch(token):
        _fail("E_KEYCHAIN_TOKEN")
    bundle = launcher.discover_bundle()
    previous = os.environ.get(launcher.TOKEN_ENVIRONMENT_VARIABLE)
    completed_all = False
    try:
        os.environ[launcher.TOKEN_ENVIRONMENT_VARIABLE] = token
        client = transfer.SeafileNativeClient(bundle.config)
        for row, item in incomplete:
            metadata = client.exact_metadata(row["source_locator"])
            exact = int(metadata.size_bytes)
            if exact <= 0 or exact > RAW_CAP_BYTES:
                _fail("E_REMOTE_SIZE")
            remaining_clip_bound = sum(
                int(candidate["projected_clip_bound_bytes"])
                for candidate, state in zip(rows, checkpoint["items"])
                if state["status"] != "COMPLETE"
            )
            _capacity(
                control_root, exact + min(remaining_clip_bound, DERIVED_CAP_BYTES)
            )
            raw_path = transient / "source.bin"
            pending_clip = transient / "derived.mp4"
            transferred = 0
            digest = hashlib.sha256()
            if raw_path.exists():
                if (
                    not _private_file(raw_path)
                    or raw_path.stat().st_size != exact
                ):
                    with contextlib.suppress(OSError):
                        raw_path.unlink()
                else:
                    transferred = exact
                    with raw_path.open("rb") as handle:
                        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
                            digest.update(block)
            if transferred == 0:
                response = client.open_download(row["source_locator"])
                try:
                    descriptor = os.open(
                        raw_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                    )
                    with os.fdopen(descriptor, "wb") as handle:
                        while True:
                            block = response.read(CHUNK_BYTES)
                            if not block:
                                break
                            transferred += len(block)
                            if transferred > exact or transferred > RAW_CAP_BYTES:
                                _fail("E_STREAM_CAP")
                            handle.write(block)
                            digest.update(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                finally:
                    response.close()
            if transferred != exact or not _private_file(raw_path):
                _fail("E_TRANSFER_SIZE")
            source_duration, _ = _duration_and_streams(_probe(raw_path))
            expected_duration = float(row["duration_ms"]) / 1000.0
            if abs(source_duration - expected_duration) > max(2.0, expected_duration * 0.02):
                _fail("E_SOURCE_DURATION")
            clip_start = float(row["clip_source_start_ms"]) / 1000.0
            clip_duration = (
                float(row["clip_source_end_ms"] - row["clip_source_start_ms"]) / 1000.0
            )
            _run_quiet(
                [
                    str(FFMPEG),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{clip_start:.3f}",
                    "-i",
                    str(raw_path),
                    "-t",
                    f"{clip_duration:.3f}",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-vf",
                    "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2,fps=15,format=yuv420p",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "24",
                    "-c:a",
                    "aac",
                    "-ac",
                    "1",
                    "-b:a",
                    "96k",
                    "-movflags",
                    "+faststart",
                    "-y",
                    str(pending_clip),
                ],
                timeout=7200,
            )
            actual_clip_duration, _ = _duration_and_streams(_probe(pending_clip))
            if abs(actual_clip_duration - clip_duration) > max(1.0, clip_duration * 0.02):
                _fail("E_CLIP_DURATION")
            clip_size = pending_clip.stat().st_size
            if clip_size <= 0:
                _fail("E_CLIP")
            clip_sha = _sha256_file(pending_clip)
            destination = clips / f"{clip_sha}.mp4"
            if destination.exists():
                if (
                    not _private_file(destination)
                    or destination.stat().st_size != clip_size
                    or _sha256_file(destination) != clip_sha
                ):
                    _fail("E_CLIP_CONFLICT")
                pending_clip.unlink()
            else:
                os.replace(pending_clip, destination)
                os.chmod(destination, 0o600)
            raw_path.unlink()
            item.update(
                {
                    "source_exact_bytes": transferred,
                    "source_sha256": digest.hexdigest(),
                    "clip_relative_path": destination.relative_to(root).as_posix(),
                    "clip_bytes": clip_size,
                    "clip_sha256": clip_sha,
                    "clip_duration_seconds": round(actual_clip_duration, 3),
                    "status": "COMPLETE",
                }
            )
            checkpoint["status"] = (
                "COMPLETE"
                if all(state["status"] == "COMPLETE" for state in checkpoint["items"])
                else "IN_PROGRESS"
            )
            _write_atomic(checkpoint_path, checkpoint, private=True)
            if any(transient.iterdir()):
                _fail("E_TRANSIENT_NOT_EMPTY")
            _capacity(control_root, 0)
        completed_all = all(item["status"] == "COMPLETE" for item in checkpoint["items"])
    except transfer.TransferError:
        _fail("E_NATIVE_TRANSFER")
    finally:
        for path in (transient / "source.bin", transient / "derived.mp4"):
            with contextlib.suppress(OSError):
                path.unlink()
        if previous is None:
            os.environ.pop(launcher.TOKEN_ENVIRONMENT_VARIABLE, None)
        else:
            os.environ[launcher.TOKEN_ENVIRONMENT_VARIABLE] = previous
        token = ""
    if not completed_all:
        _fail("E_INCOMPLETE")
    secrets_store.delete()
    total_source = sum(int(item["source_exact_bytes"]) for item in checkpoint["items"])
    total_clips = sum(int(item["clip_bytes"]) for item in checkpoint["items"])
    if total_clips > DERIVED_CAP_BYTES:
        _fail("E_DERIVED_CAP")
    usage = shutil.disk_usage(control_root)
    receipt = {
        "schema_version": PUBLIC_SCHEMA,
        "status": "EXTENSION_ACQUISITION_AND_CLIPPING_COMPLETE",
        "protocol_sha256": PROTOCOL_SHA256,
        "selection_receipt_sha256": SELECTION_RECEIPT_SHA256,
        "restricted_plan_sha256": RESTRICTED_PLAN_SHA256,
        "restricted_checkpoint_sha256": _digest(checkpoint),
        "additional_item_count": ITEM_COUNT,
        "completed_item_count": ITEM_COUNT,
        "source_transfer_bytes_rounded_up_gib": math.ceil(total_source / 1024**3),
        "retained_clip_bytes_rounded_up_gib": math.ceil(total_clips / 1024**3),
        "retained_clip_cap_bytes": DERIVED_CAP_BYTES,
        "maximum_concurrent_full_source_objects": 1,
        "simultaneous_raw_cap_bytes": RAW_CAP_BYTES,
        "transient_full_sources_removed": True,
        "all_clips_have_video_and_audio": True,
        "all_clip_durations_consistent": True,
        "exact_remote_metadata_used": True,
        "sequential_transfer": True,
        "duplicate_full_resolution_copies_created": False,
        "full_archive_downloaded": False,
        "current_free_space_rounded_down_gib": math.floor(usage.free / 1024**3),
        "free_space_floor_bytes": FREE_FLOOR_BYTES,
        "free_space_floor_satisfied": usage.free >= FREE_FLOOR_BYTES,
        "namespace_cap_bytes": NAMESPACE_CAP_BYTES,
        "retention_deadline": "2027-07-31",
        "restricted_payload_exported": False,
        "identifiers_or_paths_exported": False,
        "exact_timestamps_exported": False,
        "frames_audio_or_transcripts_exported": False,
        "external_or_cloud_transfer": False,
        "scientific_endpoint_opened": False,
    }
    _public_guard(receipt)
    _write_atomic(PUBLIC_RECEIPT, receipt, private=False, immutable=True)
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
                    "completed_item_count": receipt["completed_item_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    except AcquisitionError as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
