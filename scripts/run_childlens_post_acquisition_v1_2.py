#!/usr/bin/env python3
"""Fail-closed ChildLens v1.2 post-acquisition orchestrator.

The command deliberately accepts no arguments.  It discovers the one sealed
native-transfer bundle, verifies every restricted binding in process, prepares
the owner-only measurement and human-workflow manifests, runs the local media
audit, initializes the human workflow, and publishes three aggregate receipts.

Restricted paths, object keys, media hashes, annotation hashes, exact windows,
and row-level results never cross the quarantine boundary or stdout.  Official
speech-presence bouts are retained as coarse windows; this program never calls
them utterances and never invents candidate utterances from them.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
SEARCH_ROOT = REPO_ROOT.parent
OUTPUT_ROOT = REPO_ROOT / "output" / "childlens_feasibility_v1_2"

VERSION = "childlens-post-acquisition-orchestrator-v1.2.0"
EXPECTED_COUNT = 15
RETENTION_DEADLINE = "2027-07-31"
FROZEN_SKELETON_SHA256 = "ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5"
FROZEN_RESTRICTED_INPUT_SHA256 = "81282e6a4d8178b1559a7473d22c70a36e44450467b6b5e2ff70ad9b7e9f049a"
FROZEN_SELECTION_DIGEST = "61526ea6cebc256314b0b9e574fb1a5986fcd83a534979e93134c4163c55253f"
RAW_CAP_BYTES = 20 * 1024**3
NAMESPACE_CAP_BYTES = 73 * 1024**3
FREE_SPACE_FLOOR_BYTES = 50 * 1024**3
NONRAW_RESERVE_BYTES = 4 * 1024**3
FROZEN_DISPLAY_TOTAL_BYTES = 14_071_800_000
FROZEN_ROUNDING_UPPER_BYTES = 14_572_800_000
FROZEN_TRANSFER_OVERHEAD_BYTES = 397_386_240
FROZEN_CONSERVATIVE_ADMISSION_BYTES = 14_970_186_240
ADMISSION_CEILING_BYTES = 18 * 1024**3
PRE_TRANSFER_NAMESPACE_BYTES = 20_451_328
PRE_TRANSFER_FREE_BYTES = 131_236_425_728
PUBLIC_SNAPSHOT_COMMIT = "4856662653b2fa183e53268d088cffba02a33443"
MAX_RESTRICTED_JSON_BYTES = 128 * 1024**2
MAX_ANNOTATION_JSON_BYTES = 32 * 1024**2
CPU_WORKER_CAP = 2
CELL_SUPPRESSION_K = 5
SPEECH_RE = re.compile(r"speech|talk|speak|vocal", re.IGNORECASE)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
ABSOLUTE_PATH_RE = re.compile(r"(?i)(?:/Users/|file://|\\Users\\)")
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
MEDIA_NAME_RE = re.compile(
    r"(?i)(?<![a-z0-9_.-])[a-z0-9][a-z0-9_.-]{2,}\."
    r"(?:mp4|mov|mkv|avi|webm|wav|mp3|m4a|aac|flac|jpg|jpeg|png|srt|vtt)\b"
)


class OrchestratorError(RuntimeError):
    """A fixed diagnostic code safe for terminal output."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise OrchestratorError(code)


def _load_module(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail("E_MODULE_IMPORT")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        _fail("E_MODULE_IMPORT")
    return module


def _modules() -> tuple[Any, Any, Any, Any]:
    transfer = _load_module(
        "childlens_native_transfer_v1_2",
        SCRIPT_PATH.with_name("childlens_native_transfer_v1_2.py"),
    )
    launcher = _load_module(
        "launch_childlens_native_transfer_v1_2",
        SCRIPT_PATH.with_name("launch_childlens_native_transfer_v1_2.py"),
    )
    auditor = _load_module(
        "audit_childlens_local_media_v1_2",
        SCRIPT_PATH.with_name("audit_childlens_local_media_v1_2.py"),
    )
    workflow = _load_module(
        "childlens_human_validation_v1_2",
        SCRIPT_PATH.with_name("childlens_human_validation_v1_2.py"),
    )
    return transfer, launcher, auditor, workflow


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        _fail("E_FILE_HASH")
    return digest.hexdigest()


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


def _reject_symlink_chain(root: Path, candidate: Path) -> None:
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except OSError:
        _fail("E_SYMLINK_OR_ESCAPE")
    if not _is_relative_to(resolved_candidate, resolved_root):
        _fail("E_SYMLINK_OR_ESCAPE")
    try:
        # Prefer the lexical path so a symlink *inside* quarantine remains
        # visible to lstat.  If macOS presents the same absolute path through
        # its /var -> /private/var alias, fall back to the resolved pair.
        relative = candidate.relative_to(root)
        cursor = root
    except ValueError:
        try:
            relative = resolved_candidate.relative_to(resolved_root)
            cursor = resolved_root
        except ValueError:
            _fail("E_SYMLINK_OR_ESCAPE")
    if ".." in relative.parts:
        _fail("E_SYMLINK_OR_ESCAPE")
    for part in relative.parts:
        cursor /= part
        try:
            if stat.S_ISLNK(cursor.lstat().st_mode):
                _fail("E_SYMLINK_OR_ESCAPE")
        except OSError:
            _fail("E_SYMLINK_OR_ESCAPE")


def _read_private_json(path: Path, root: Path, *, maximum: int) -> Mapping[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        size = resolved.stat().st_size
    except OSError:
        _fail("E_RESTRICTED_JSON")
    if (
        not _is_relative_to(resolved, root)
        or not _private_regular(path)
        or size <= 0
        or size > maximum
    ):
        _fail("E_RESTRICTED_JSON")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_RESTRICTED_JSON")
    if not isinstance(value, dict):
        _fail("E_RESTRICTED_JSON")
    return value


def _find_canonical_document(
    root: Path,
    expected_digest: str,
    *,
    excluded_root: Path | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    matches: list[tuple[Path, Mapping[str, Any]]] = []
    excluded = excluded_root.resolve() if excluded_root is not None else None
    try:
        walker = os.walk(root, followlinks=False)
        for directory, directories, files in walker:
            parent = Path(directory)
            if excluded is not None and (
                parent.resolve() == excluded or _is_relative_to(parent.resolve(), excluded)
            ):
                directories[:] = []
                continue
            directories[:] = [
                name
                for name in directories
                if _private_directory(parent / name)
                and not (
                    excluded is not None
                    and (parent / name).resolve() == excluded
                )
            ]
            for name in files:
                path = parent / name
                if not name.lower().endswith(".json") or not _private_regular(path):
                    continue
                try:
                    if path.stat().st_size > MAX_RESTRICTED_JSON_BYTES:
                        continue
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict) and _digest(value) == expected_digest:
                    matches.append((path, value))
    except OSError:
        _fail("E_FROZEN_DISCOVERY")
    if len(matches) != 1:
        _fail("E_FROZEN_DOCUMENT_MISSING" if not matches else "E_FROZEN_DOCUMENT_AMBIGUOUS")
    return matches[0]


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _timing_number(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    accepted = set(names)
    for key, value in row.items():
        if _normalise_key(str(key)) in accepted:
            number = _finite_number(value)
            if number is not None:
                return number
    return None


def extract_official_speech_windows(document: Mapping[str, Any]) -> tuple[list[tuple[float, float]], int]:
    """Return official timed speech-like bouts and invalid timing row count.

    ``start + duration`` is a representation conversion of explicit timing,
    not utterance inference.  Non-speech rows are ignored and no gap-filling or
    speech segmentation is performed.
    """

    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        _fail("E_ANNOTATION_SCHEMA")
    windows: list[tuple[float, float]] = []
    invalid = 0
    for raw in annotations:
        if not isinstance(raw, dict):
            _fail("E_ANNOTATION_SCHEMA")
        event_values = [
            value
            for key, value in raw.items()
            if _normalise_key(str(key)) in {"eventid", "event_id", "category", "label", "type"}
            and isinstance(value, str)
        ]
        if not any(SPEECH_RE.search(value) for value in event_values):
            continue
        start = _timing_number(raw, ("start", "start_time", "starttime", "onset", "time"))
        end = _timing_number(raw, ("end", "end_time", "endtime", "offset"))
        duration = _timing_number(raw, ("duration",))
        if end is None and start is not None and duration is not None:
            end = start + duration
        if start is None or end is None or start < 0 or end <= start:
            invalid += 1
            continue
        windows.append((start, end))
    return coalesce_windows(windows), invalid


def coalesce_windows(windows: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in windows)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            _fail("E_WINDOW_INVALID")
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


@dataclass(frozen=True, repr=False)
class RestrictedPreparation:
    measurement_path: Path
    packet_path: Path
    skeleton_path: Path
    restricted_input_path: Path
    native_receipt_path: Path
    pilot_selection_sha256: str
    canonical_manifest_sha256: str
    native_receipt_sha256: str
    download_plan_sha256: str
    raw_bytes: int
    annotation_verified_count: int
    media_verified_count: int
    speech_window_source_count: int
    no_speech_window_source_count: int
    speech_window_count: int
    unioned_speech_seconds: float
    invalid_speech_timing_count: int
    total_media_seconds: float
    release_immutable_revision_used: bool


def _namespace_size(root: Path) -> int:
    total = 0
    try:
        for directory, directories, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            directories[:] = [name for name in directories if not (parent / name).is_symlink()]
            for name in files:
                path = parent / name
                metadata = path.lstat()
                if stat.S_ISREG(metadata.st_mode):
                    total += metadata.st_size
    except OSError:
        _fail("E_NAMESPACE_SCAN")
    return total


def _quarantine_control_root(root: Path) -> Path:
    """Return the outermost bounded private ancestor carrying controls."""

    candidate = root.resolve(strict=True)
    controlled: Path | None = None
    for _depth in range(5):
        if _private_directory(candidate) and _private_regular(
            candidate / ".metadata_never_index"
        ):
            controlled = candidate
        parent = candidate.parent
        if parent == candidate or parent == REPO_ROOT or _is_relative_to(parent, REPO_ROOT):
            break
        if not _private_directory(parent):
            break
        candidate = parent
    if controlled is None:
        _fail("E_INDEXING_EXCLUSION")
    return controlled


def _ensure_nested_index_sentinel(root: Path) -> None:
    """Mirror the recursive no-index control at the workflow data root."""

    sentinel = root / ".metadata_never_index"
    if sentinel.exists():
        if not _private_regular(sentinel):
            _fail("E_INDEXING_EXCLUSION")
        return
    try:
        descriptor = os.open(sentinel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        _fail("E_INDEXING_EXCLUSION")


def _recheck_quarantine(root: Path, config: Mapping[str, Any]) -> tuple[int, int]:
    try:
        resolved = root.resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_ROOT")
    if resolved == REPO_ROOT or _is_relative_to(resolved, REPO_ROOT) or not _private_directory(resolved):
        _fail("E_QUARANTINE_ROOT")
    # The transfer bundle may use a private nested data root, while Spotlight
    # and namespace controls are attached to its authorized quarantine
    # ancestor.  A no-index sentinel on an ancestor applies recursively.  Use
    # the nearest owner-private ancestor as the control/cap root without
    # changing any receipt-relative media path.
    control_root = _quarantine_control_root(resolved)
    attestations = config.get("quarantine_attestations")
    if not isinstance(attestations, dict) or any(
        attestations.get(key) is not True
        for key in (
            "owner_only_access_verified",
            "outside_git_repository_verified",
            "spotlight_excluded_verified",
            "backup_excluded_or_encrypted_local_only_verified",
            "signed_agreement_controls_verified",
        )
    ):
        _fail("E_QUARANTINE_ATTESTATIONS")
    if attestations.get("retention_deadline") != RETENTION_DEADLINE:
        _fail("E_RETENTION_MISMATCH")
    if date.today() > date.fromisoformat(RETENTION_DEADLINE):
        _fail("E_RETENTION_EXPIRED")
    namespace = _namespace_size(control_root)
    try:
        volume = os.statvfs(control_root)
    except OSError:
        _fail("E_CAPACITY")
    free = volume.f_bavail * volume.f_frsize
    if namespace > NAMESPACE_CAP_BYTES or free < FREE_SPACE_FLOOR_BYTES:
        _fail("E_CAPACITY")
    return namespace, free


def _write_restricted_immutable(path: Path, value: Any) -> None:
    encoded = _canonical(value) + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if path.exists():
        if not _private_regular(path):
            _fail("E_RESTRICTED_OUTPUT")
        try:
            existing = path.read_bytes()
        except OSError:
            _fail("E_RESTRICTED_OUTPUT")
        if existing != encoded:
            _fail("E_RESTRICTED_OUTPUT_IMMUTABLE")
        return
    temporary = path.parent / f".tmp-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except OSError:
        _fail("E_RESTRICTED_OUTPUT")
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _verify_file_inside(root: Path, locator: str, expected_sha: str) -> tuple[Path, Mapping[str, Any]]:
    candidate = Path(locator).expanduser()
    if not candidate.is_absolute():
        _fail("E_ANNOTATION_LOCATOR")
    try:
        _reject_symlink_chain(root, candidate)
    except OrchestratorError as exc:
        if exc.code == "E_SYMLINK_OR_ESCAPE":
            _fail("E_ANNOTATION_PATH_BOUNDARY")
        raise
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        _fail("E_ANNOTATION_FILE")
    if (
        not _is_relative_to(resolved, root)
        or not _private_regular(candidate)
        or resolved.stat().st_size > MAX_ANNOTATION_JSON_BYTES
        or _sha256_file(resolved) != expected_sha
    ):
        _fail("E_ANNOTATION_INTEGRITY")
    document = _read_private_json(resolved, root, maximum=MAX_ANNOTATION_JSON_BYTES)
    return resolved, document


def prepare_restricted_artifacts(
    *,
    root: Path,
    annotation_root: Path | None = None,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    skeleton_path: Path,
    skeleton: Mapping[str, Any],
    restricted_input_path: Path,
    restricted_input: Mapping[str, Any],
    workflow: Any,
) -> RestrictedPreparation:
    """Verify all bindings and create the two owner-only restricted manifests."""

    annotation_boundary = annotation_root or root
    frozen_keys, frozen_selection = workflow._validate_frozen_skeleton(skeleton_path, root)
    frozen_bindings = workflow._validate_restricted_input(
        restricted_input_path, root, frozen_keys
    )
    native_bindings = workflow._validate_native_receipt(
        Path(config["_native_receipt_path"]), root, frozen_selection
    )
    if receipt.get("status") != "COMPLETE" or len(native_bindings) != EXPECTED_COUNT:
        _fail("E_NATIVE_RECEIPT_INCOMPLETE")

    media_rows = restricted_input.get("media")
    object_rows = restricted_input.get("objects")
    annotation_rows = restricted_input.get("annotations")
    if not all(isinstance(value, list) for value in (media_rows, object_rows, annotation_rows)):
        _fail("E_RESTRICTED_INPUT_SCHEMA")
    media_by_key = {
        row.get("media_key"): row
        for row in media_rows
        if isinstance(row, dict) and isinstance(row.get("media_key"), str)
    }
    objects_by_key: dict[str, Mapping[str, Any]] = {}
    for row in object_rows:
        if not isinstance(row, dict) or not isinstance(row.get("object_key"), str):
            continue
        key = row["object_key"]
        if key in objects_by_key:
            _fail("E_OBJECT_DUPLICATE")
        objects_by_key[key] = row
    links: dict[str, list[Mapping[str, Any]]] = {}
    for row in annotation_rows:
        if isinstance(row, dict) and row.get("linked_media_key") in frozen_keys:
            links.setdefault(row["linked_media_key"], []).append(row)

    measurement_items: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    raw_bytes = 0
    unioned_seconds = 0.0
    speech_sources = 0
    speech_window_count = 0
    invalid_timing_count = 0
    total_media_seconds = 0.0
    verified_annotations = 0
    verified_media = 0

    ordering = sorted(
        frozen_keys,
        key=lambda key: hashlib.sha256(
            f"{frozen_selection}\0human-workflow-v1.2\0{key}".encode("utf-8")
        ).hexdigest(),
    )
    for index, media_key in enumerate(ordering, 1):
        media_row = media_by_key.get(media_key)
        frozen = frozen_bindings.get(media_key)
        if not isinstance(media_row, dict) or not isinstance(frozen, dict):
            _fail("E_FROZEN_BINDING")
        object_key = frozen["source_object_key"]
        native = native_bindings.get(object_key)
        if not isinstance(native, dict):
            _fail("E_NATIVE_BINDING")
        media_path = root / native["media_relpath"]
        try:
            _reject_symlink_chain(root, media_path)
        except OrchestratorError as exc:
            if exc.code == "E_SYMLINK_OR_ESCAPE":
                _fail("E_MEDIA_PATH_BOUNDARY")
            raise
        try:
            media_resolved = media_path.resolve(strict=True)
        except OSError:
            _fail("E_MEDIA_FILE")
        if (
            not _is_relative_to(media_resolved, root)
            or not _private_regular(media_path)
            or media_resolved.stat().st_size != native["transferred_bytes"]
            or _sha256_file(media_resolved) != native["media_sha256"]
        ):
            _fail("E_MEDIA_INTEGRITY")
        verified_media += 1
        raw_bytes += native["transferred_bytes"]

        linked_shas: list[str] = []
        item_windows: list[tuple[float, float]] = []
        item_links = links.get(media_key, [])
        if not item_links:
            _fail("E_ANNOTATION_LINK_MISSING")
        for link in item_links:
            annotation_object = objects_by_key.get(link.get("object_key"))
            if not isinstance(annotation_object, dict):
                _fail("E_ANNOTATION_OBJECT")
            local_sha = annotation_object.get("local_sha256")
            locator = annotation_object.get("source_locator")
            if (
                annotation_object.get("top_level_class") != "ANNOTATION"
                or not isinstance(local_sha, str)
                or not HEX64_RE.fullmatch(local_sha)
                or not isinstance(locator, str)
            ):
                _fail("E_ANNOTATION_OBJECT")
            _annotation_path, document = _verify_file_inside(
                annotation_boundary, locator, local_sha
            )
            windows, invalid = extract_official_speech_windows(document)
            item_windows.extend(windows)
            invalid_timing_count += invalid
            linked_shas.append(local_sha)
            verified_annotations += 1
        expected_linkage = hashlib.sha256(
            media_key.encode("utf-8")
            + b"\0"
            + b"\0".join(value.encode("utf-8") for value in sorted(linked_shas))
        ).hexdigest()
        if expected_linkage != frozen["annotation_linkage_sha256"]:
            _fail("E_ANNOTATION_LINKAGE_DIGEST")

        item_windows = coalesce_windows(item_windows)
        speech_window_count += len(item_windows)
        speech_sources += int(bool(item_windows))
        unioned_seconds += sum(end - start for start, end in item_windows)
        duration_seconds = frozen["duration_ms"] / 1000.0
        total_media_seconds += duration_seconds
        expectation = media_row.get("speech_presence_bin")
        if expectation not in {"PRESENT", "ABSENT"}:
            _fail("E_SPEECH_EXPECTATION")
        measurement_items.append(
            {
                "blinded_item_key": media_key,
                "media_relpath": native["media_relpath"],
                "expected_size_bytes": native["transferred_bytes"],
                "expected_media_sha256": native["media_sha256"],
                "reference_duration_seconds": duration_seconds,
                "reference_duration_basis": "ANNOTATION_MAX_END",
                "speech_presence_expectation": expectation,
                "speech_windows": [
                    {"start_seconds": start, "end_seconds": end}
                    for start, end in item_windows
                ],
            }
        )
        packet_rows.append(
            {
                "internal_key": media_key,
                "display_key": f"HV-{index:03d}",
                "media_relpath": native["media_relpath"],
                "stratum_key": frozen["stratum_key"],
                "duration_ms": frozen["duration_ms"],
                "batch_number": (index - 1) // 5 + 1,
                "source_object_key": object_key,
                "media_sha256": native["media_sha256"],
                "annotation_linkage_sha256": frozen["annotation_linkage_sha256"],
            }
        )

    if raw_bytes >= RAW_CAP_BYTES or verified_media != EXPECTED_COUNT:
        _fail("E_RAW_CAP_OR_COUNT")
    measurement = {
        "schema_version": "childlens-restricted-measurement-manifest-v1.2.1",
        "pilot_selection_sha256": frozen_selection,
        "items": measurement_items,
    }
    packet = {
        "schema_version": workflow.VERSION,
        "opaque_key_attestation": True,
        "frozen_selection_digest": frozen_selection,
        "items": packet_rows,
    }
    restricted_dir = root / "post_acquisition_v1_2"
    measurement_path = restricted_dir / "restricted_measurement_manifest.json"
    packet_path = restricted_dir / "restricted_human_packet.json"
    _write_restricted_immutable(measurement_path, measurement)
    _write_restricted_immutable(packet_path, packet)
    return RestrictedPreparation(
        measurement_path=measurement_path,
        packet_path=packet_path,
        skeleton_path=skeleton_path,
        restricted_input_path=restricted_input_path,
        native_receipt_path=Path(config["_native_receipt_path"]),
        pilot_selection_sha256=frozen_selection,
        canonical_manifest_sha256=receipt["canonical_restricted_manifest_sha256"],
        native_receipt_sha256=receipt["restricted_receipt_sha256"],
        download_plan_sha256=receipt["download_plan_sha256"],
        raw_bytes=raw_bytes,
        annotation_verified_count=verified_annotations,
        media_verified_count=verified_media,
        speech_window_source_count=speech_sources,
        no_speech_window_source_count=EXPECTED_COUNT - speech_sources,
        speech_window_count=speech_window_count,
        unioned_speech_seconds=unioned_seconds,
        invalid_speech_timing_count=invalid_timing_count,
        total_media_seconds=total_media_seconds,
        release_immutable_revision_used=receipt.get("immutable_commit_id") is not None,
    )


def _metric_pass_count(metric: Mapping[str, Any], total: int) -> int | None:
    status_value = metric.get("status")
    if status_value == "ALL_PASS":
        return total
    if status_value == "NONE_PASS":
        return 0
    value = metric.get("pass_count")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _suppressed_count(value: int, total: int = EXPECTED_COUNT) -> tuple[int | None, bool]:
    complement = total - value
    if value in (0, total) or min(value, complement) >= CELL_SUPPRESSION_K:
        return value, False
    return None, True


def estimate_human_labor(prepared: RestrictedPreparation) -> dict[str, int | float | str]:
    """Conservative packet-specific estimate using the frozen base labor rates."""

    sample_minutes = min(30.0, prepared.unioned_speech_seconds / 60.0)
    language_review_minutes = min(prepared.total_media_seconds / 60.0, EXPECTED_COUNT * 2.0)
    first = language_review_minutes + sample_minutes * (5.0 + 8.0)
    second = language_review_minutes + sample_minutes * (5.0 + 8.0 * 0.20)
    adjudication = language_review_minutes * 0.50 + sample_minutes * (5.0 + 5.0 + 8.0 + 1.6) * 0.40
    first_i = math.ceil(first)
    second_i = math.ceil(second)
    adjudication_i = math.ceil(adjudication)
    return {
        "basis": "ACTUAL_PACKET_DURATIONS_WITH_FROZEN_BASE_5X_TIMING_8X_REFERENTIAL_40_PERCENT_ADJUDICATION",
        "actual_packet_item_count": EXPECTED_COUNT,
        "planned_speech_review_minutes": round(sample_minutes, 2),
        "first_pass_minutes": first_i,
        "second_independent_pass_minutes": second_i,
        "adjudication_minutes": adjudication_i,
        "total_minutes": first_i + second_i + adjudication_i,
    }


def _public_guard(value: Any) -> None:
    encoded = _canonical(value).decode("utf-8")
    if ABSOLUTE_PATH_RE.search(encoded) or EMAIL_RE.search(encoded) or MEDIA_NAME_RE.search(encoded):
        _fail("E_PUBLIC_PRIVACY")
    forbidden_keys = {
        "participant_id",
        "session_id",
        "source_filename",
        "source_path",
        "media_path",
        "transcript",
        "transcript_text",
        "utterance_text",
        "exact_timestamp",
        "start_time",
        "end_time",
        "authorization",
        "access_token",
    }
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if forbidden_keys.intersection(_normalise_key(str(key)) for key in current):
                _fail("E_PUBLIC_PRIVACY")
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def build_public_receipts(
    *,
    prepared: RestrictedPreparation,
    config: Mapping[str, Any],
    audit_receipt: Mapping[str, Any],
    workflow_readiness: Mapping[str, Any],
    namespace_bytes: int,
    free_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = audit_receipt.get("container_audio_decode_metrics")
    if not isinstance(metrics, dict):
        _fail("E_AUDIT_RECEIPT")
    probe = _metric_pass_count(metrics.get("probe_success", {}), EXPECTED_COUNT)
    audio = _metric_pass_count(metrics.get("audio_stream_present", {}), EXPECTED_COUNT)
    decode = _metric_pass_count(metrics.get("full_decode_success", {}), EXPECTED_COUNT)
    duration = _metric_pass_count(metrics.get("duration_consistency_evidenced", {}), EXPECTED_COUNT)
    corruption = _metric_pass_count(metrics.get("corruption_absence_evidenced", {}), EXPECTED_COUNT)
    expectation = _metric_pass_count(
        metrics.get("speech_presence_window_expectation_satisfied", {}), EXPECTED_COUNT
    )
    windows, windows_suppressed = _suppressed_count(prepared.speech_window_source_count)
    no_windows, no_windows_suppressed = _suppressed_count(prepared.no_speech_window_source_count)
    invalid_timing, invalid_timing_suppressed = _suppressed_count(
        min(prepared.invalid_speech_timing_count, EXPECTED_COUNT)
    )
    annotation_source_count, annotation_source_suppressed = _suppressed_count(
        min(prepared.annotation_verified_count, EXPECTED_COUNT)
    )
    all_structural = all(
        value == EXPECTED_COUNT for value in (probe, audio, decode, duration, corruption, expectation)
    ) and audit_receipt.get("status") == "ALL_STRUCTURAL_CHECKS_PASS"
    manifest_digest = prepared.canonical_manifest_sha256
    measurement_receipt_digest = _digest(audit_receipt)
    workflow_source_digest = _digest(workflow_readiness)
    admission_total = sum(
        int(row["upper_bound_bytes"])
        for row in config["admission"]["conservative_bounds"]
    )
    if (
        admission_total != FROZEN_CONSERVATIVE_ADMISSION_BYTES
        or config["admission"].get("predeclared_nonraw_reserve_bytes") != NONRAW_RESERVE_BYTES
    ):
        _fail("E_FROZEN_ADMISSION")
    acquisition = {
        "schema_version": "childlens-v1.2-acquisition-receipt-v1",
        "status": "COMPLETE",
        "selected_media_count": EXPECTED_COUNT,
        "acquired_media_count": EXPECTED_COUNT,
        "verified_against_frozen_selection_count": prepared.media_verified_count,
        "acquisition_complete": True,
        "raw_media_bytes_acquired": prepared.raw_bytes,
        "raw_cap_bytes": RAW_CAP_BYTES,
        "storage_admission_method": "CONSERVATIVE_DISPLAY_BOUND_WITH_HARD_COUNTER",
        "displayed_parsed_total_bytes": FROZEN_DISPLAY_TOTAL_BYTES,
        "rounding_upper_bound_bytes": FROZEN_ROUNDING_UPPER_BYTES,
        "transfer_overhead_bytes": FROZEN_TRANSFER_OVERHEAD_BYTES,
        "pre_transfer_upper_bound_bytes": admission_total,
        "admission_ceiling_bytes": ADMISSION_CEILING_BYTES,
        "upper_bound_margin_to_raw_cap_bytes": RAW_CAP_BYTES - admission_total,
        "hard_abort_before_bytes": RAW_CAP_BYTES - 1,
        "predeclared_nonraw_reserve_bytes": NONRAW_RESERVE_BYTES,
        "sequential_or_tightly_bounded_transfer": True,
        "cumulative_stream_counter_enforced": True,
        "release_binding_verified": True,
        "restricted_manifest_digest_verified": True,
        "restricted_manifest_sha256": manifest_digest,
        "restricted_native_transfer_receipt_sha256": prepared.native_receipt_sha256,
        "frozen_selection_digest": prepared.pilot_selection_sha256,
        "immutable_public_snapshot_commit": PUBLIC_SNAPSHOT_COMMIT,
        "pilot_source": "ACCESSIBLE_LIVE_VIEW_BOUND_TO_RESTRICTED_MANIFEST_DIGEST",
        "live_to_public_snapshot_byte_equivalence": "NOT_PROVEN",
        "live_snapshot_limitation_disclosed": True,
        "media_content_hash_linkage_verified": prepared.media_verified_count == EXPECTED_COUNT,
        "annotation_hash_linkage_verified": prepared.annotation_verified_count >= EXPECTED_COUNT,
        "namespace_cap_bytes": NAMESPACE_CAP_BYTES,
        "projected_post_peak_free_floor_bytes": FREE_SPACE_FLOOR_BYTES,
        "pre_transfer_namespace_bytes": PRE_TRANSFER_NAMESPACE_BYTES,
        "pre_transfer_free_bytes": PRE_TRANSFER_FREE_BYTES,
        "projected_namespace_peak_bytes": PRE_TRANSFER_NAMESPACE_BYTES
        + admission_total
        + NONRAW_RESERVE_BYTES,
        "projected_post_peak_free_bytes": PRE_TRANSFER_FREE_BYTES
        - admission_total
        - NONRAW_RESERVE_BYTES,
        "post_transfer_free_bytes": free_bytes,
        "namespace_bytes_after_acquisition": namespace_bytes,
        "retention_deadline": RETENTION_DEADLINE,
        "retention_control_recorded": True,
        "full_archive_downloaded": False,
        "selection_changed": False,
        "selection_content_inspected_before_transfer": False,
        "duplicate_full_resolution_copies_created": False,
        "source_filenames_exported": False,
        "restricted_identifiers_exported": False,
        "exact_timestamps_exported": False,
        "restricted_manifest_exported": False,
        "external_or_cloud_transfer": False,
    }
    structural_metrics = {
        name: dict(metrics[name])
        for name in (
            "restricted_manifest_file_integrity_match",
            "probe_success",
            "video_stream_present",
            "audio_stream_present",
            "duration_consistency_evidenced",
            "full_decode_success",
            "corruption_absence_evidenced",
            "speech_presence_window_expectation_satisfied",
        )
    }
    structural_metrics["annotation_linkage_verified"] = {
        "status": "ALL_PASS",
        "pass_count": EXPECTED_COUNT,
        "fail_count": 0,
        "cell_suppressed": False,
    }
    diagnostics = {
        "schema_version": "childlens-v1.2-automated-diagnostics-receipt-v1",
        "status": "PASS" if all_structural else "REVIEW_REQUIRED",
        "source_acquired_media_count": EXPECTED_COUNT,
        "media_preparation_attempted_count": EXPECTED_COUNT,
        "container_probe_attempted_count": EXPECTED_COUNT,
        "container_probe_success_count": probe,
        "decode_attempted_count": EXPECTED_COUNT,
        "decode_success_count": decode,
        "audio_stream_media_count": audio,
        "duration_consistent_media_count": duration,
        "corruption_free_media_count": corruption,
        "annotation_linkage_verified_count": EXPECTED_COUNT,
        "annotation_source_file_verified_count": annotation_source_count,
        "annotation_source_file_verified_count_suppressed": annotation_source_suppressed,
        "speech_presence_window_media_count": expectation,
        "speech_presence_window_actual_source_count": windows,
        "speech_presence_window_actual_source_count_suppressed": windows_suppressed,
        "no_speech_window_media_count": no_windows,
        "no_speech_window_media_count_suppressed": no_windows_suppressed,
        "speech_expectation_concordant_media_count": expectation,
        "invalid_official_speech_timing_row_count": invalid_timing,
        "invalid_official_speech_timing_row_count_suppressed": invalid_timing_suppressed,
        "unioned_official_speech_window_count": (
            prepared.speech_window_count
            if prepared.speech_window_count == 0 or prepared.speech_window_count >= CELL_SUPPRESSION_K
            else None
        ),
        "unioned_official_speech_window_count_suppressed": 0 < prepared.speech_window_count < CELL_SUPPRESSION_K,
        "cell_suppression_k": CELL_SUPPRESSION_K,
        "structural_metrics": structural_metrics,
        "small_nonzero_failure_cells_exported": False,
        "measurement_source": "LOCAL_RESTRICTED_MEDIA_AUDIT_V1_2",
        "measurement_executed_on_acquired_bytes": True,
        "speech_window_actual_source": "OFFICIAL_ANNOTATION_WINDOWS_LINKED_TO_ACQUIRED_MEDIA",
        "annotation_linkage_actual_source": "RESTRICTED_CANONICAL_MANIFEST",
        "pilot_selection_sha256": prepared.pilot_selection_sha256,
        "source_native_transfer_receipt_sha256": prepared.native_receipt_sha256,
        "restricted_measurement_receipt_sha256": measurement_receipt_digest,
        "official_speech_window_union_minutes": round(prepared.unioned_speech_seconds / 60.0, 2),
        "candidate_duration_basis": "UNIONED_OFFICIAL_SPEECH_PRESENCE_WINDOWS",
        "official_speech_windows_are_utterances": False,
        "candidate_utterances_inferred": 0,
        "cpu_worker_cap": CPU_WORKER_CAP,
        "effective_heavy_decode_concurrency": 1,
        "network_protocols_allowed": False,
        "automated_preparation_complete": all_structural,
        "diagnostics_passed_for_human_workflow": all_structural,
        "aggregate_only": True,
        "local_offline_only": True,
        "automated_outputs_not_gold": True,
        "lexical_content_exported": False,
        "transcript_text_exported": False,
        "speaker_labels_exported": False,
        "frames_exported": False,
        "external_api_used": False,
        "learner_data_created": False,
        "instrument_features_passed_to_learner": False,
        "instrument_embeddings_passed_to_learner": False,
        "instrument_tokenizer_or_vocabulary_passed_to_learner": False,
    }
    labor = estimate_human_labor(prepared)
    runtime_ready = workflow_readiness.get("runtime_handoff_ready") is True
    candidate_minutes = round(prepared.unioned_speech_seconds / 60.0, 2)
    ready = all_structural and runtime_ready and candidate_minutes >= 30.0
    public_window_count = (
        prepared.speech_window_count
        if prepared.speech_window_count == 0 or prepared.speech_window_count >= CELL_SUPPRESSION_K
        else None
    )
    workflow_receipt = {
        "schema_version": "childlens-v1.2-human-workflow-receipt-v1",
        "status": "READY_FOR_AUTHORIZED_HUMANS" if ready else "BOUNDED_REVIEW_REQUIRED",
        "selected_media_source_count": EXPECTED_COUNT,
        "workflow_populated": True,
        "ready_for_authorized_human": ready,
        "runtime_handoff_ready": runtime_ready,
        "populated_candidate_utterance_count": 0,
        "populated_candidate_speech_window_count": public_window_count,
        "populated_candidate_speech_window_count_suppressed": 0
        < prepared.speech_window_count
        < CELL_SUPPRESSION_K,
        "populated_candidate_speech_window_minutes": candidate_minutes,
        "candidate_unit_type": "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE",
        "candidate_duration_basis": "UNIONED_OFFICIAL_SPEECH_PRESENCE_WINDOWS",
        "speech_window_source_count": windows,
        "speech_window_source_count_suppressed": windows_suppressed,
        "no_speech_window_source_count": no_windows,
        "no_speech_window_source_count_suppressed": no_windows_suppressed,
        "double_code_fraction_target": 0.20,
        "referential_assignment_eligible_count": 0,
        "referential_double_code_assigned_count": 0,
        "referential_double_code_assigned_fraction": None,
        "referential_double_code_sample_frozen": False,
        "referential_assignment_rule_frozen": True,
        "referential_inventory_pending_human_segmentation": True,
        "language_establishment_required_before_transcription": True,
        "language_matched_humans_required": True,
        "human_qualification_screening_enabled": True,
        "language_routes_blinded_and_resumable": True,
        "actual_language_established_by_qualified_humans": False,
        "language_matched_coders_verified": False,
        "frozen_reliability_thresholds_pass": False,
        "first_pass_and_second_pass_must_be_independent": True,
        "estimated_first_authorized_human_minutes": labor["first_pass_minutes"],
        "estimated_second_independent_human_minutes": labor["second_independent_pass_minutes"],
        "estimated_adjudication_minutes": labor["adjudication_minutes"],
        "estimated_total_human_minutes": labor["total_minutes"],
        "labor_estimate_basis": labor["basis"],
        "labor_estimate_actual_packet_item_count": labor["actual_packet_item_count"],
        "labor_estimate_planned_speech_review_minutes": labor["planned_speech_review_minutes"],
        "workflow_source": "LOCAL_RESTRICTED_HUMAN_VALIDATION_PACKET_V1_2",
        "packet_populated_from_actual_acquired_media": True,
        "pilot_selection_sha256": prepared.pilot_selection_sha256,
        "source_native_transfer_receipt_sha256": prepared.native_receipt_sha256,
        "source_measurement_receipt_sha256": measurement_receipt_digest,
        "restricted_workflow_receipt_sha256": workflow_source_digest,
        "genuine_human_validation_complete": False,
        "independent_reliability_requirement_complete": False,
        "human_evidence_fabricated": False,
        "local_only": True,
        "autosave_to_restricted_quarantine_only": True,
        "resumable_batches": True,
        "input_validation_enabled": True,
        "instrument_condition_blinded": True,
        "identifiers_displayed": False,
        "source_filenames_displayed": False,
        "exact_timestamps_exported": False,
        "transcript_text_exported": False,
        "external_service_used": False,
    }
    for value in (acquisition, diagnostics, workflow_receipt):
        _public_guard(value)
    return acquisition, diagnostics, workflow_receipt


def _atomic_public_batch(output_root: Path, documents: Mapping[str, Mapping[str, Any]]) -> None:
    output_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for name, value in documents.items():
            destination = output_root / name
            temporary = output_root / f".tmp-{secrets.token_hex(12)}"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, destination))
        for temporary, destination in staged:
            os.replace(temporary, destination)
            os.chmod(destination, 0o644)
    except OSError:
        _fail("E_PUBLIC_OUTPUT")
    finally:
        for temporary, _destination in staged:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def _find_executable(name: str) -> Path:
    candidates = [
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
    ]
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    _fail("E_LOCAL_MEDIA_TOOL")


def _load_sealed_bundle(
    *,
    search_root: Path,
    repository_root: Path,
    transfer: Any,
    launcher: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        bundle = launcher.discover_bundle(
            search_root=search_root, repository_root=repository_root
        )
        # ``discover_bundle`` returns the controller-normalized plan/config.
        # Revalidating the normalized conservative bounds would reject the
        # derived ``upper_bound_bytes`` field that validation itself adds.
        # Receipt validation below is the second independent binding check.
        plan = bundle.plan
        config = bundle.config
        receipt_value = transfer._read_json(bundle.restricted_receipt_path, "E_RECEIPT_READ")
        receipt = transfer._validate_receipt(receipt_value, plan, config)
    except Exception:
        _fail("E_NATIVE_BUNDLE")
    if receipt.get("status") != "COMPLETE" or len(receipt.get("items", [])) != EXPECTED_COUNT:
        _fail("E_NATIVE_BUNDLE_INCOMPLETE")
    config = dict(config)
    config["_native_receipt_path"] = str(bundle.restricted_receipt_path)
    return bundle, plan, config, receipt


def execute(
    *,
    search_root: Path = SEARCH_ROOT,
    repository_root: Path = REPO_ROOT,
    output_root: Path = OUTPUT_ROOT,
    frozen_skeleton_sha256: str = FROZEN_SKELETON_SHA256,
    frozen_restricted_input_sha256: str = FROZEN_RESTRICTED_INPUT_SHA256,
    bundle_loader: Callable[..., tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]] | None = None,
    ffprobe: Path | None = None,
    ffmpeg: Path | None = None,
) -> dict[str, str]:
    transfer, launcher, auditor, workflow = _modules()
    loader = bundle_loader or _load_sealed_bundle
    bundle, plan, config, receipt = loader(
        search_root=search_root,
        repository_root=repository_root,
        transfer=transfer,
        launcher=launcher,
    )
    root = Path(bundle.quarantine_root).resolve(strict=True)
    namespace_before, free_before = _recheck_quarantine(root, config)
    control_root = _quarantine_control_root(root)
    _ensure_nested_index_sentinel(root)
    generated_root = root / "post_acquisition_v1_2"
    lock_dir = generated_root
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(lock_dir, 0o700)
    # The workflow is intentionally rooted at the transfer bundle so its
    # content-addressed media paths remain receipt-relative.  Copy only the two
    # small canonical binding documents into that root; their frozen digests
    # remain unchanged.  Source annotations are verified in place against the
    # enclosing authorized control root and are not duplicated.
    skeleton_path = lock_dir / "frozen_v1_1_human_packet.json"
    restricted_input_path = lock_dir / "frozen_v1_1_restricted_input.json"
    copies_exist = skeleton_path.exists() or restricted_input_path.exists()
    if copies_exist:
        if not skeleton_path.exists() or not restricted_input_path.exists():
            _fail("E_FROZEN_COPY_INCOMPLETE")
        skeleton = _read_private_json(
            skeleton_path, root, maximum=MAX_RESTRICTED_JSON_BYTES
        )
        restricted_input = _read_private_json(
            restricted_input_path, root, maximum=MAX_RESTRICTED_JSON_BYTES
        )
        if (
            _digest(skeleton) != frozen_skeleton_sha256
            or _digest(restricted_input) != frozen_restricted_input_sha256
        ):
            _fail("E_FROZEN_COPY_DIGEST")
    else:
        _source_skeleton_path, skeleton = _find_canonical_document(
            control_root,
            frozen_skeleton_sha256,
            excluded_root=generated_root,
        )
        _source_restricted_input_path, restricted_input = _find_canonical_document(
            control_root,
            frozen_restricted_input_sha256,
            excluded_root=generated_root,
        )
        _write_restricted_immutable(skeleton_path, skeleton)
        _write_restricted_immutable(restricted_input_path, restricted_input)
    lock_path = lock_dir / ".orchestrator.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            _fail("E_ALREADY_RUNNING")
        prepared = prepare_restricted_artifacts(
            root=root,
            annotation_root=control_root,
            plan=plan,
            config=config,
            receipt=receipt,
            skeleton_path=skeleton_path,
            skeleton=skeleton,
            restricted_input_path=restricted_input_path,
            restricted_input=restricted_input,
            workflow=workflow,
        )
        if prepared.pilot_selection_sha256 != plan.get("pilot_selection_sha256"):
            _fail("E_FROZEN_SELECTION")
        if (
            frozen_skeleton_sha256 == FROZEN_SKELETON_SHA256
            and prepared.pilot_selection_sha256 != FROZEN_SELECTION_DIGEST
        ):
            _fail("E_FROZEN_SELECTION")
        namespace_after_prepare, free_after_prepare = _recheck_quarantine(root, config)
        if namespace_after_prepare < namespace_before or free_after_prepare > free_before + 1024**3:
            _fail("E_CAPACITY_OBSERVATION")
        probe_path = ffprobe or _find_executable("ffprobe")
        decode_path = ffmpeg or _find_executable("ffmpeg")
        try:
            audit_receipt = auditor.audit(
                root,
                prepared.measurement_path,
                probe_path,
                decode_path,
                timeout_seconds=7200,
                cell_suppression_k=CELL_SUPPRESSION_K,
            )
        except Exception:
            _fail("E_LOCAL_MEDIA_AUDIT")

        previous_root = os.environ.get(workflow.ROOT_ENV)
        os.environ[workflow.ROOT_ENV] = str(root)
        try:
            policy_path = root / workflow.POLICY_FILE
            if not policy_path.exists():
                workflow.bootstrap_policy(root, attest=True)
            workflow.initialize(
                root,
                prepared.packet_path,
                prepared.skeleton_path,
                prepared.restricted_input_path,
                prepared.native_receipt_path,
            )
            workflow_readiness = workflow.validate_readiness(root)
        except Exception:
            _fail("E_HUMAN_WORKFLOW")
        finally:
            if previous_root is None:
                os.environ.pop(workflow.ROOT_ENV, None)
            else:
                os.environ[workflow.ROOT_ENV] = previous_root
        namespace_final, free_final = _recheck_quarantine(root, config)
        acquisition, diagnostics, workflow_receipt = build_public_receipts(
            prepared=prepared,
            config=config,
            audit_receipt=audit_receipt,
            workflow_readiness=workflow_readiness,
            namespace_bytes=namespace_final,
            free_bytes=free_final,
        )
        _atomic_public_batch(
            output_root,
            {
                "acquisition_receipt.json": acquisition,
                "automated_diagnostics_receipt.json": diagnostics,
                "human_validation_workflow_receipt.json": workflow_receipt,
            },
        )
    finally:
        os.close(lock_fd)
    return {"status": "ok", "state": "POST_ACQUISITION_RECEIPTS_READY"}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    old_umask = os.umask(0o077)
    try:
        if arguments:
            _fail("E_ARGUMENTS")
        execute()
    except OrchestratorError as exc:
        print(json.dumps({"error_code": exc.code, "status": "error"}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"error_code": "E_INTERNAL", "status": "error"}, sort_keys=True))
        return 3
    finally:
        os.umask(old_umask)
    print(json.dumps({"state": "POST_ACQUISITION_RECEIPTS_READY", "status": "ok"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
