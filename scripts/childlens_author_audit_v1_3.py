#!/usr/bin/env python3
"""Fail-closed controller for the ChildLens v1.3 single-author audit.

The controller consumes a separately prepared, model-independent fifteen-minute
sample inside the approved quarantine.  Human labels and model pseudo-labels
remain in separate stores.  The prediction store is integrity-bound at
initialization but cannot be parsed or returned through this module until the
entire AUTHOR_AUDIT_A record has been irreversibly locked.

This module never discovers corpus media or generates model annotations.  Its
only repository-safe outputs are aggregate readiness/progress receipts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


VERSION = "childlens-author-audit-workflow-v1.3.0"
PACKET_VERSION = "childlens-author-audit-primary-sample-v1.3.0"
PREDICTION_BINDING_VERSION = "childlens-author-audit-prediction-binding-v1.3.0"
RECEIPT_VERSION = "childlens-v1.3-author-workflow-receipt-v1"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILE = REPO_ROOT / "docs/childlens_feasibility_v1_3/frozen_model_assisted_author_audit_protocol_v1_3.json"
ROOT_ENV = "CHILDLENS_V13_QUARANTINE_ROOT"
DISCOVERY_SEARCH_ROOT = REPO_ROOT.parent
POLICY_FILE = ".childlens_v1_3_quarantine_policy.json"
INDEX_SENTINEL = ".metadata_never_index"
WORKFLOW_DIR = "author_audit_v1_3"
DATABASE_FILE = "author_audit.sqlite3"
SECRET_FILE = ".author_audit_hmac_key"
AUTHOR_ASSIGNMENT_FILE = "author_assignment_secret.json"
SAMPLE_SNAPSHOT_FILE = "primary_sample_snapshot.json"
PREDICTION_BINDING_SNAPSHOT_FILE = "prediction_binding_snapshot.json"
RETENTION_DEADLINE = "2027-07-31"
ROUTE = "AUTHOR_AUDIT_A"
FROZEN_ITEM_COUNT = 15
FROZEN_TOTAL_DURATION_MS = 900_000
MINIMUM_ITEM_DURATION_MS = 1_000
SYNTHETIC_INTERACTION_ESTIMATE_MINUTES = 75

ROLE_VALUES = frozenset({"NON_CHILD", "CHILD", "OVERLAP", "UNCERTAIN", "NONSPEECH"})
ITEM_DISPOSITIONS = frozenset({"ANNOTATED", "NO_LINGUISTIC_SPEECH", "UNCERTAIN", "UNUSABLE"})
AUDIO_USABILITY_VALUES = frozenset({"USABLE", "PARTIAL", "UNUSABLE", "UNCERTAIN"})
REFERENTIAL_VALUES = frozenset({"VISIBLE_CANDIDATE", "NULL_NOT_VISIBLE", "IRRELEVANT", "UNDECIDABLE", "UNUSABLE"})
CANDIDATE_VALUES = frozenset({"PRESENT", "ABSENT", "UNCERTAIN", "NOT_APPLICABLE"})
CANDIDATE_COUNT_BANDS = frozenset({"ZERO", "ONE", "TWO", "THREE_PLUS", "UNKNOWN"})
MENTION_FAMILIES = frozenset({"NOUN_OBJECT", "VERB_ACTION"})
LANGUAGE_SPECIAL = frozenset({"MIXED_OR_CODE_SWITCHED", "UNDECIDABLE"})
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
OPAQUE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{12,96}$")
DISPLAY_KEY_RE = re.compile(r"^AUDIT-[0-9]{3}$")
CONTENT_ADDRESS_RE = re.compile(r"^(?:[A-Za-z0-9_-]+/)*[0-9a-f]{64}\.(?:bin|mp4|mov|mkv|webm|wav|m4a)$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PACKET_KEYS = frozenset(
    {
        "participant",
        "participant_id",
        "session",
        "session_id",
        "filename",
        "source_filename",
        "source_path",
        "absolute_path",
        "transcript",
        "text",
        "model_prediction",
        "model_score",
        "confidence",
    }
)

_DISCOVERED_RUNTIME_ROOT: Path | None = None


class WorkflowError(RuntimeError):
    """Fixed-code exception safe for logs and user-facing UI."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise WorkflowError(code)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.exists() and stat.S_ISLNK(current.lstat().st_mode):
            _fail("E_SYMLINK_COMPONENT")


def _private_file(path: Path) -> bool:
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


def _load_json_file(path: Path, error: str) -> Mapping[str, Any]:
    if not _private_file(path):
        _fail(error)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail(error)
    if not isinstance(value, dict):
        _fail(error)
    return value


def _load_policy(root: Path) -> Mapping[str, Any]:
    policy = _load_json_file(root / POLICY_FILE, "E_QUARANTINE_POLICY_INVALID")
    required = (
        "owner_only_verified",
        "git_exclusion_verified",
        "indexing_exclusion_verified",
        "backup_exclusion_verified_or_encrypted_local_only",
        "local_only",
        "signed_agreement_controls_inherited",
    )
    if policy.get("schema_version") != VERSION or any(policy.get(key) is not True for key in required):
        _fail("E_QUARANTINE_POLICY_INVALID")
    if policy.get("retention_deadline") != RETENTION_DEADLINE:
        _fail("E_RETENTION_POLICY_MISMATCH")
    if date.today() > date.fromisoformat(RETENTION_DEADLINE):
        _fail("E_RETENTION_DEADLINE_PASSED")
    return policy


def _validate_root_controls(root_arg: str | os.PathLike[str]) -> Path:
    requested = Path(root_arg).expanduser()
    if not requested.is_absolute():
        _fail("E_QUARANTINE_ROOT_NOT_ABSOLUTE")
    try:
        root = requested.resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_ROOT_MISSING")
    _reject_symlink_components(root)
    if root == REPO_ROOT or _is_relative_to(root, REPO_ROOT):
        _fail("E_QUARANTINE_INSIDE_REPOSITORY")
    metadata = root.stat()
    if not root.is_dir() or root.is_symlink() or metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        _fail("E_QUARANTINE_NOT_OWNER_ONLY")
    if not _private_file(root / INDEX_SENTINEL):
        _fail("E_INDEXING_SENTINEL_MISSING")
    _load_policy(root)
    return root


def validate_quarantine_root(root_arg: str | os.PathLike[str]) -> Path:
    requested = Path(root_arg).expanduser()
    if not requested.is_absolute():
        _fail("E_QUARANTINE_ROOT_NOT_ABSOLUTE")
    try:
        resolved = requested.resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_ROOT_MISSING")
    environment = os.environ.get(ROOT_ENV)
    if environment:
        authorized_path = Path(environment).expanduser()
        if not authorized_path.is_absolute():
            _fail("E_QUARANTINE_ROOT_NOT_ABSOLUTE")
        try:
            authorized = authorized_path.resolve(strict=True)
        except OSError:
            _fail("E_QUARANTINE_ROOT_MISSING")
    elif _DISCOVERED_RUNTIME_ROOT is not None:
        authorized = _DISCOVERED_RUNTIME_ROOT
    else:
        _fail("E_QUARANTINE_ENV_MISSING")
    if resolved != authorized:
        _fail("E_QUARANTINE_ROOT_MISMATCH")
    return _validate_root_controls(resolved)


def _private_discovery_directory(path: Path, hidden_root: Path) -> bool:
    current = path
    while True:
        try:
            metadata = current.lstat()
        except OSError:
            return False
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            return False
        if current == hidden_root:
            return True
        if hidden_root not in current.parents:
            return False
        current = current.parent


def discover_runtime_root(*, search_root: Path = DISCOVERY_SEARCH_ROOT, repository_root: Path = REPO_ROOT) -> Path:
    """Authorize exactly one initialized private v1.3 runtime without path input."""

    global _DISCOVERED_RUNTIME_ROOT
    _DISCOVERED_RUNTIME_ROOT = None
    try:
        search = search_root.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
    except OSError:
        _fail("E_RUNTIME_DISCOVERY_ROOT")
    if repository.parent != search:
        _fail("E_RUNTIME_DISCOVERY_ROOT")
    try:
        hidden_roots = [
            candidate.resolve(strict=True)
            for candidate in search.iterdir()
            if candidate != repository
            and candidate.name.startswith(".")
            and "childlens" in candidate.name.lower()
            and candidate.is_dir()
            and not candidate.is_symlink()
            and candidate.stat().st_uid == os.getuid()
            and stat.S_IMODE(candidate.stat().st_mode) & 0o077 == 0
        ]
    except OSError:
        _fail("E_RUNTIME_DISCOVERY_ROOT")
    candidates: list[Path] = []
    for hidden_root in sorted(hidden_roots, key=lambda value: value.name):
        try:
            for directory, directories, _files in os.walk(hidden_root, followlinks=False):
                current = Path(directory)
                if not _private_discovery_directory(current, hidden_root):
                    directories[:] = []
                    continue
                directories[:] = [
                    name
                    for name in directories
                    if name not in {"raw_v1_2", "pseudo_annotations_v1_3", "browser_profile", "browser_cache", "browser_downloads"}
                    and not (current / name).is_symlink()
                    and _private_discovery_directory(current / name, hidden_root)
                ]
                workflow_dir = current / WORKFLOW_DIR
                required = (
                    current / POLICY_FILE,
                    current / INDEX_SENTINEL,
                    workflow_dir / DATABASE_FILE,
                    workflow_dir / SECRET_FILE,
                    workflow_dir / AUTHOR_ASSIGNMENT_FILE,
                    workflow_dir / SAMPLE_SNAPSHOT_FILE,
                )
                if not all(_private_file(path) for path in required):
                    continue
                try:
                    candidates.append(_validate_root_controls(current))
                except WorkflowError:
                    continue
        except OSError:
            _fail("E_RUNTIME_DISCOVERY_SCAN")
    unique = sorted(set(candidates), key=str)
    if len(unique) != 1:
        _fail("E_RUNTIME_ROOT_AMBIGUOUS" if unique else "E_RUNTIME_ROOT_NOT_FOUND")
    _DISCOVERED_RUNTIME_ROOT = unique[0]
    return unique[0]


def bootstrap_policy(root_arg: str | os.PathLike[str], *, attest: bool) -> Path:
    """Create the v1.3 policy receipt only after explicit operator attestation."""

    if not attest:
        _fail("E_ATTESTATION_REQUIRED")
    requested = Path(root_arg).expanduser()
    environment = os.environ.get(ROOT_ENV)
    if not environment or not requested.is_absolute() or not Path(environment).expanduser().is_absolute():
        _fail("E_QUARANTINE_ENV_MISSING")
    try:
        root = requested.resolve(strict=True)
        authorized = Path(environment).expanduser().resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_ROOT_MISSING")
    if root != authorized or root == REPO_ROOT or _is_relative_to(root, REPO_ROOT):
        _fail("E_QUARANTINE_ROOT_MISMATCH")
    _reject_symlink_components(root)
    metadata = root.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        _fail("E_QUARANTINE_NOT_OWNER_ONLY")
    if not _private_file(root / INDEX_SENTINEL):
        _fail("E_INDEXING_SENTINEL_MISSING")
    payload = {
        "schema_version": VERSION,
        "created_at_utc": _utc_now(),
        "retention_deadline": RETENTION_DEADLINE,
        "owner_only_verified": True,
        "git_exclusion_verified": True,
        "indexing_exclusion_verified": True,
        "backup_exclusion_verified_or_encrypted_local_only": True,
        "local_only": True,
        "signed_agreement_controls_inherited": True,
        "operator_attestation": "explicit",
    }
    path = root / POLICY_FILE
    _atomic_write(path, _canonical(payload) + b"\n")
    return path


def _workflow_paths(root_arg: str | os.PathLike[str]) -> tuple[Path, Path, Path]:
    root = validate_quarantine_root(root_arg)
    directory = root / WORKFLOW_DIR
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    return root, directory / DATABASE_FILE, directory / SECRET_FILE


def _load_or_create_secret(path: Path) -> bytes:
    if path.exists():
        if not _private_file(path):
            _fail("E_SECRET_PERMISSIONS")
        value = path.read_bytes()
        if len(value) != 32:
            _fail("E_SECRET_INVALID")
        return value
    value = secrets.token_bytes(32)
    _atomic_write(path, value)
    return value


@contextlib.contextmanager
def _connect(database: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        for candidate in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm")):
            with contextlib.suppress(FileNotFoundError):
                os.chmod(candidate, 0o600)


@contextlib.contextmanager
def _db(root_arg: str | os.PathLike[str]) -> Iterator[tuple[sqlite3.Connection, bytes, Path]]:
    _, database, secret_path = _workflow_paths(root_arg)
    secret = _load_or_create_secret(secret_path)
    with _connect(database) as connection:
        yield connection, secret, database


SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_meta (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    schema_version INTEGER NOT NULL,
    workflow_version TEXT NOT NULL,
    route TEXT NOT NULL CHECK(route='AUTHOR_AUDIT_A'),
    sample_digest TEXT NOT NULL,
    packet_digest TEXT NOT NULL,
    protocol_digest TEXT NOT NULL,
    workflow_state TEXT NOT NULL CHECK(workflow_state IN ('AUTHOR_BLIND_OPEN','AUTHOR_LOCKED','PREDICTION_JOIN_ENABLED')),
    prediction_binding_digest TEXT,
    prediction_store_relpath TEXT,
    prediction_store_sha256 TEXT,
    estimated_author_minutes INTEGER NOT NULL CHECK(estimated_author_minutes > 0),
    created_at_utc TEXT NOT NULL,
    author_pass_locked_at_utc TEXT,
    author_record_hmac TEXT,
    inter_human_reliability_available INTEGER NOT NULL DEFAULT 0 CHECK(inter_human_reliability_available=0),
    pseudo_labels_are_ground_truth INTEGER NOT NULL DEFAULT 0 CHECK(pseudo_labels_are_ground_truth=0),
    primary_evaluation_truth TEXT NOT NULL CHECK(primary_evaluation_truth='SIMULATOR_ORACLE_ONLY')
);
CREATE TABLE IF NOT EXISTS audit_items (
    item_id INTEGER PRIMARY KEY,
    internal_key TEXT NOT NULL UNIQUE,
    prediction_join_key TEXT NOT NULL UNIQUE,
    display_key TEXT NOT NULL UNIQUE,
    media_relpath TEXT NOT NULL,
    media_sha256 TEXT NOT NULL,
    duration_ms INTEGER NOT NULL CHECK(duration_ms>=1000)
);
CREATE TABLE IF NOT EXISTS audit_segments (
    item_id INTEGER NOT NULL REFERENCES audit_items(item_id),
    segment_index INTEGER NOT NULL CHECK(segment_index >= 0),
    start_ms INTEGER NOT NULL CHECK(start_ms >= 0),
    end_ms INTEGER NOT NULL CHECK(end_ms > start_ms),
    PRIMARY KEY(item_id, segment_index)
);
CREATE TABLE IF NOT EXISTS author_slot (
    route TEXT PRIMARY KEY CHECK(route='AUTHOR_AUDIT_A'),
    author_hmac TEXT NOT NULL,
    first_opened_at_utc TEXT
);
CREATE TABLE IF NOT EXISTS item_labels (
    item_id INTEGER PRIMARY KEY REFERENCES audit_items(item_id),
    disposition TEXT NOT NULL CHECK(disposition IN ('ANNOTATED','NO_LINGUISTIC_SPEECH','UNCERTAIN','UNUSABLE')),
    language_code TEXT NOT NULL,
    language_competence TEXT NOT NULL CHECK(language_competence IN ('NATIVE','FLUENT','PROFICIENT','INSUFFICIENT','UNDECIDABLE')),
    wer_applicability TEXT NOT NULL CHECK(wer_applicability IN ('APPLICABLE','NOT_APPLICABLE','UNDECIDABLE')),
    audio_usability TEXT NOT NULL CHECK(audio_usability IN ('USABLE','PARTIAL','UNUSABLE','UNCERTAIN')),
    locked INTEGER NOT NULL DEFAULT 0 CHECK(locked IN (0,1)),
    updated_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS author_utterances (
    utterance_id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES audit_items(item_id),
    display_utterance_key TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    onset_ms INTEGER NOT NULL CHECK(onset_ms >= 0),
    offset_ms INTEGER NOT NULL CHECK(offset_ms > onset_ms),
    source_text TEXT NOT NULL,
    speaker_role TEXT NOT NULL CHECK(speaker_role IN ('NON_CHILD','CHILD','OVERLAP','UNCERTAIN','NONSPEECH')),
    referential_status TEXT NOT NULL CHECK(referential_status IN ('VISIBLE_CANDIDATE','NULL_NOT_VISIBLE','IRRELEVANT','UNDECIDABLE','UNUSABLE')),
    noun_object_decision TEXT NOT NULL CHECK(noun_object_decision IN ('PRESENT','ABSENT','UNCERTAIN','NOT_APPLICABLE')),
    verb_action_decision TEXT NOT NULL CHECK(verb_action_decision IN ('PRESENT','ABSENT','UNCERTAIN','NOT_APPLICABLE')),
    uncertain_unusable_reason TEXT NOT NULL DEFAULT '',
    locked INTEGER NOT NULL DEFAULT 0 CHECK(locked IN (0,1)),
    updated_at_utc TEXT NOT NULL,
    UNIQUE(item_id, display_utterance_key),
    FOREIGN KEY(item_id, segment_index) REFERENCES audit_segments(item_id, segment_index)
);
CREATE TABLE IF NOT EXISTS author_mentions (
    mention_id INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES author_utterances(utterance_id),
    display_mention_key TEXT NOT NULL,
    mention_family TEXT NOT NULL CHECK(mention_family IN ('NOUN_OBJECT','VERB_ACTION')),
    mention_start_char INTEGER NOT NULL CHECK(mention_start_char >= 0),
    mention_end_char INTEGER NOT NULL CHECK(mention_end_char > mention_start_char),
    referential_status TEXT NOT NULL CHECK(referential_status IN ('VISIBLE_CANDIDATE','NULL_NOT_VISIBLE','UNDECIDABLE','UNUSABLE')),
    candidate_count_band TEXT NOT NULL CHECK(candidate_count_band IN ('ZERO','ONE','TWO','THREE_PLUS','UNKNOWN')),
    visible_segment_index INTEGER,
    visible_onset_ms INTEGER,
    visible_offset_ms INTEGER,
    uncertain_unusable_reason TEXT NOT NULL DEFAULT '',
    locked INTEGER NOT NULL DEFAULT 0 CHECK(locked IN (0,1)),
    updated_at_utc TEXT NOT NULL,
    UNIQUE(utterance_id, display_mention_key)
);
CREATE TABLE IF NOT EXISTS audit_log (
    event_id INTEGER PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    event TEXT NOT NULL
);
"""

EXPECTED_SCHEMA_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "workflow_meta": (
        "singleton", "schema_version", "workflow_version", "route",
        "sample_digest", "packet_digest", "protocol_digest", "workflow_state",
        "prediction_binding_digest", "prediction_store_relpath",
        "prediction_store_sha256", "estimated_author_minutes", "created_at_utc",
        "author_pass_locked_at_utc", "author_record_hmac",
        "inter_human_reliability_available", "pseudo_labels_are_ground_truth",
        "primary_evaluation_truth",
    ),
    "audit_items": (
        "item_id", "internal_key", "prediction_join_key", "display_key",
        "media_relpath", "media_sha256", "duration_ms",
    ),
    "audit_segments": ("item_id", "segment_index", "start_ms", "end_ms"),
    "author_slot": ("route", "author_hmac", "first_opened_at_utc"),
    "item_labels": (
        "item_id", "disposition", "language_code", "language_competence",
        "wer_applicability", "audio_usability", "locked", "updated_at_utc",
    ),
    "author_utterances": (
        "utterance_id", "item_id", "display_utterance_key", "segment_index",
        "onset_ms", "offset_ms", "source_text", "speaker_role",
        "referential_status", "noun_object_decision", "verb_action_decision",
        "uncertain_unusable_reason", "locked", "updated_at_utc",
    ),
    "author_mentions": (
        "mention_id", "utterance_id", "display_mention_key", "mention_family",
        "mention_start_char", "mention_end_char", "referential_status",
        "candidate_count_band", "visible_segment_index", "visible_onset_ms",
        "visible_offset_ms", "uncertain_unusable_reason", "locked",
        "updated_at_utc",
    ),
    "audit_log": ("event_id", "created_at_utc", "event"),
}


def _assert_schema_compatible(connection: sqlite3.Connection) -> None:
    for table, expected in EXPECTED_SCHEMA_COLUMNS.items():
        actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
        if actual != expected:
            _fail("E_SCHEMA_INCOMPATIBLE")


def _walk_keys(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _require_hex(value: object, code: str) -> str:
    if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
        _fail(code)
    return value


def _resolve_private_confined(root: Path, relative: object, code: str) -> Path:
    if not isinstance(relative, str):
        _fail(code)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        _fail(code)
    try:
        resolved = (root / candidate).resolve(strict=True)
    except OSError:
        _fail(code)
    if not _is_relative_to(resolved, root) or not _private_file(resolved):
        _fail(code)
    return resolved


def _confined_input_relative(root: Path, path_arg: str | os.PathLike[str], code: str) -> str:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        return os.fspath(path)
    try:
        resolved = path.resolve(strict=True)
        return os.fspath(resolved.relative_to(root.resolve(strict=True)))
    except (OSError, ValueError):
        _fail(code)


def _validate_packet(packet: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], str, int]:
    if packet.get("schema_version") != PACKET_VERSION:
        _fail("E_PACKET_VERSION")
    if packet.get("opaque_key_attestation") is not True:
        _fail("E_OPAQUE_KEY_ATTESTATION")
    if packet.get("selection_independent_of_model_outputs") is not True or packet.get("sample_frozen_before_predictions") is not True:
        _fail("E_SAMPLE_INDEPENDENCE_ATTESTATION")
    if packet.get("route") != ROUTE or packet.get("sample_kind") != "PRIMARY_AUTHOR_AUDIT" or packet.get("activation") != "READY_BEFORE_MODEL_REVEAL":
        _fail("E_SAMPLE_POLICY")
    if packet.get("model_predictions_present") is not False or packet.get("model_predictions_used_for_selection") is not False:
        _fail("E_SAMPLE_INDEPENDENCE_ATTESTATION")
    if packet.get("exact_intervals_restricted") is not True:
        _fail("E_SAMPLE_POLICY")
    if FORBIDDEN_PACKET_KEYS.intersection(_walk_keys(packet)):
        _fail("E_PACKET_RESTRICTED_OR_MODEL_FIELD")
    for key in ("frozen_selection_digest", "sampler_policy_sha256", "official_windows_manifest_sha256"):
        _require_hex(packet.get(key), "E_PACKET_BINDING")
    if packet.get("item_count") != FROZEN_ITEM_COUNT or packet.get("total_duration_ms") != FROZEN_TOTAL_DURATION_MS:
        _fail("E_PRIMARY_SAMPLE_SIZE")
    estimated = SYNTHETIC_INTERACTION_ESTIMATE_MINUTES
    items = packet.get("items")
    if not isinstance(items, list) or len(items) != FROZEN_ITEM_COUNT:
        _fail("E_PRIMARY_SAMPLE_SIZE")
    validated: list[dict[str, Any]] = []
    internal_keys: set[str] = set()
    join_keys: set[str] = set()
    total_packet_duration = 0
    for index, row in enumerate(items, start=1):
        if not isinstance(row, Mapping):
            _fail("E_SAMPLE_ITEM")
        internal = row.get("audit_item_key")
        join_key = row.get("prediction_join_key")
        if not isinstance(internal, str) or not OPAQUE_KEY_RE.fullmatch(internal):
            _fail("E_SAMPLE_ITEM_KEY")
        if not isinstance(join_key, str) or not OPAQUE_KEY_RE.fullmatch(join_key):
            _fail("E_PREDICTION_JOIN_KEY")
        if internal in internal_keys or join_key in join_keys:
            _fail("E_SAMPLE_ITEM_DUPLICATE")
        internal_keys.add(internal)
        join_keys.add(join_key)
        relative = row.get("media_relpath")
        if not isinstance(relative, str) or not CONTENT_ADDRESS_RE.fullmatch(relative):
            _fail("E_MEDIA_PATH_NOT_CONTENT_ADDRESSED")
        media_sha = _require_hex(row.get("expected_media_sha256"), "E_MEDIA_DIGEST")
        if Path(relative).stem != media_sha:
            _fail("E_MEDIA_DIGEST")
        media = _resolve_private_confined(root, relative, "E_MEDIA_INVALID")
        if _sha256_file(media) != media_sha:
            _fail("E_MEDIA_DIGEST")
        segments = row.get("segments")
        if not isinstance(segments, list) or not segments:
            _fail("E_SEGMENTS_INVALID")
        clean_segments: list[dict[str, int]] = []
        total = 0
        last_end = -1
        for segment in segments:
            if not isinstance(segment, Mapping):
                _fail("E_SEGMENTS_INVALID")
            start = segment.get("start_ms")
            end = segment.get("end_ms")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or start < last_end
            ):
                _fail("E_SEGMENTS_INVALID")
            clean_segments.append({"start_ms": start, "end_ms": end})
            total += end - start
            last_end = end
        declared_duration = row.get("duration_ms")
        if (
            not isinstance(declared_duration, int)
            or isinstance(declared_duration, bool)
            or declared_duration < MINIMUM_ITEM_DURATION_MS
            or total != declared_duration
        ):
            _fail("E_ITEM_DURATION")
        total_packet_duration += declared_duration
        validated.append(
            {
                "audit_item_key": internal,
                "prediction_join_key": join_key,
                "display_key": f"AUDIT-{index:03d}",
                "media_relpath": relative,
                "expected_media_sha256": media_sha,
                "duration_ms": declared_duration,
                "segments": clean_segments,
            }
        )
    if total_packet_duration != FROZEN_TOTAL_DURATION_MS:
        _fail("E_PRIMARY_SAMPLE_SIZE")
    calculated_sample_digest = _sha256(_canonical(packet))
    return validated, calculated_sample_digest, estimated


def _validate_prediction_binding(binding: Mapping[str, Any], root: Path, sample_digest: str) -> tuple[str, str]:
    if binding.get("schema_version") != PREDICTION_BINDING_VERSION:
        _fail("E_PREDICTION_BINDING_VERSION")
    if binding.get("sample_digest") != sample_digest:
        _fail("E_PREDICTION_SAMPLE_MISMATCH")
    if binding.get("local_offline_inference_attested") is not True or binding.get("network_disabled_during_inference") is not True:
        _fail("E_PREDICTION_LOCAL_ONLY_ATTESTATION")
    if binding.get("pseudo_labels_are_ground_truth") is not False:
        _fail("E_PSEUDO_LABEL_BOUNDARY")
    if binding.get("primary_evaluation_truth") != "SIMULATOR_ORACLE_ONLY":
        _fail("E_PSEUDO_LABEL_BOUNDARY")
    relative = binding.get("prediction_store_relpath")
    store = _resolve_private_confined(root, relative, "E_PREDICTION_STORE_INVALID")
    digest = _require_hex(binding.get("prediction_store_sha256"), "E_PREDICTION_STORE_DIGEST")
    if _sha256_file(store) != digest:
        _fail("E_PREDICTION_STORE_DIGEST")
    return str(relative), digest


def initialize(root_arg: str | os.PathLike[str], packet_path: str | os.PathLike[str]) -> dict[str, Any]:
    root, database, secret_path = _workflow_paths(root_arg)
    packet_file = _resolve_private_confined(
        root,
        _confined_input_relative(root, packet_path, "E_PACKET_FILE"),
        "E_PACKET_FILE",
    )
    packet = _load_json_file(packet_file, "E_PACKET_FILE")
    items, sample_digest, estimate = _validate_packet(packet, root)
    packet_digest = _sha256(_canonical(packet))
    if not PROTOCOL_FILE.is_file():
        _fail("E_PROTOCOL_MISSING")
    protocol_digest = _sha256_file(PROTOCOL_FILE)
    secret = _load_or_create_secret(secret_path)
    with _connect(database) as connection:
        connection.executescript(SCHEMA)
        _assert_schema_compatible(connection)
        existing = connection.execute("SELECT packet_digest FROM workflow_meta WHERE singleton=1").fetchone()
        if existing:
            if existing["packet_digest"] != packet_digest:
                _fail("E_INITIALIZATION_IMMUTABLE")
            return {"status": "already_initialized", "workflow_ready": True, "audit_item_count": FROZEN_ITEM_COUNT}
        author_token = secrets.token_urlsafe(48)
        author_hmac = hmac.new(secret, b"author\0" + author_token.encode("utf-8"), hashlib.sha256).hexdigest()
        connection.execute(
            """INSERT INTO workflow_meta(
               singleton,schema_version,workflow_version,route,sample_digest,
               packet_digest,protocol_digest,workflow_state,
               prediction_binding_digest,prediction_store_relpath,
               prediction_store_sha256,estimated_author_minutes,created_at_utc,
               author_pass_locked_at_utc,author_record_hmac,
               inter_human_reliability_available,pseudo_labels_are_ground_truth,
               primary_evaluation_truth)
               VALUES (1,?,?,?,?,?,?,'AUTHOR_BLIND_OPEN',NULL,NULL,NULL,?,?,NULL,NULL,0,0,'SIMULATOR_ORACLE_ONLY')""",
            (
                SCHEMA_VERSION,
                VERSION,
                ROUTE,
                sample_digest,
                packet_digest,
                protocol_digest,
                estimate,
                _utc_now(),
            ),
        )
        connection.execute("INSERT INTO author_slot(route,author_hmac) VALUES (?,?)", (ROUTE, author_hmac))
        for item_id, row in enumerate(items, start=1):
            connection.execute(
                "INSERT INTO audit_items VALUES (?,?,?,?,?,?,?)",
                (item_id, row["audit_item_key"], row["prediction_join_key"], row["display_key"], row["media_relpath"], row["expected_media_sha256"], row["duration_ms"]),
            )
            for segment_index, segment in enumerate(row["segments"]):
                connection.execute(
                    "INSERT INTO audit_segments VALUES (?,?,?,?)",
                    (item_id, segment_index, segment["start_ms"], segment["end_ms"]),
                )
        _atomic_write(root / WORKFLOW_DIR / AUTHOR_ASSIGNMENT_FILE, _canonical({"schema_version": VERSION, "route": ROUTE, "token": author_token}) + b"\n")
        _atomic_write(root / WORKFLOW_DIR / SAMPLE_SNAPSHOT_FILE, _canonical(packet) + b"\n")
    return {"status": "initialized", "workflow_ready": True, "audit_item_count": FROZEN_ITEM_COUNT}


def _author_hash(secret: bytes, token: str) -> str:
    value = token.strip()
    if not 40 <= len(value) <= 200:
        _fail("E_AUTHOR_TOKEN")
    return hmac.new(secret, b"author\0" + value.encode("utf-8"), hashlib.sha256).hexdigest()


def _authorize(connection: sqlite3.Connection, secret: bytes, token: str) -> str:
    supplied = _author_hash(secret, token)
    row = connection.execute("SELECT author_hmac FROM author_slot WHERE route=?", (ROUTE,)).fetchone()
    if not row or not hmac.compare_digest(supplied, row["author_hmac"]):
        _fail("E_AUTHOR_NOT_AUTHORIZED")
    connection.execute("UPDATE author_slot SET first_opened_at_utc=COALESCE(first_opened_at_utc,?) WHERE route=?", (_utc_now(), ROUTE))
    return supplied


def assign_author(root_arg: str | os.PathLike[str], author_token: str) -> dict[str, Any]:
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
    return {"route": ROUTE, "authorized": True}


def launcher_author_token(root_arg: str | os.PathLike[str]) -> str:
    """Load the sole author token only for a locally discovered app process.

    The token is returned to the in-process UI and is never rendered, printed,
    placed in a URL, or copied into a child-process environment.
    """

    root = validate_quarantine_root(root_arg)
    if _DISCOVERED_RUNTIME_ROOT is None or root != _DISCOVERED_RUNTIME_ROOT:
        _fail("E_LAUNCHER_CAPABILITY_REQUIRED")
    assignment = _load_json_file(
        root / WORKFLOW_DIR / AUTHOR_ASSIGNMENT_FILE,
        "E_AUTHOR_ASSIGNMENT",
    )
    if assignment.get("schema_version") != VERSION or assignment.get("route") != ROUTE:
        _fail("E_AUTHOR_ASSIGNMENT")
    token = assignment.get("token")
    if not isinstance(token, str):
        _fail("E_AUTHOR_ASSIGNMENT")
    with _db(root) as (connection, secret, _):
        _authorize(connection, secret, token)
    return token


def _ensure_unlocked(connection: sqlite3.Connection) -> None:
    row = connection.execute("SELECT author_pass_locked_at_utc FROM workflow_meta WHERE singleton=1").fetchone()
    if not row:
        _fail("E_NOT_INITIALIZED")
    if row["author_pass_locked_at_utc"] is not None:
        _fail("E_AUTHOR_PASS_LOCKED")


def _item_row(connection: sqlite3.Connection, display_key: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM audit_items WHERE display_key=?", (display_key,)).fetchone()
    if not row:
        _fail("E_ITEM_NOT_FOUND")
    return row


def _validate_language(value: str) -> str:
    normalized = value.strip()
    if normalized in LANGUAGE_SPECIAL or LANGUAGE_RE.fullmatch(normalized):
        return normalized
    _fail("E_LANGUAGE_INVALID")


def get_task_batch(root_arg: str | os.PathLike[str], *, author_token: str, batch_size: int = 5) -> list[dict[str, Any]]:
    if not 1 <= batch_size <= 15:
        _fail("E_BATCH_SIZE")
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        rows = connection.execute(
            """SELECT i.display_key, COALESCE(l.locked,0) AS locked,
                      COALESCE((SELECT COUNT(*) FROM author_utterances u WHERE u.item_id=i.item_id),0) AS utterance_count
               FROM audit_items i LEFT JOIN item_labels l ON l.item_id=i.item_id
               WHERE COALESCE(l.locked,0)=0 ORDER BY i.item_id LIMIT ?""",
            (batch_size,),
        ).fetchall()
        return [{"display_key": row["display_key"], "locked": bool(row["locked"]), "utterance_count": row["utterance_count"]} for row in rows]


def media_segments_for_display(root_arg: str | os.PathLike[str], *, author_token: str, display_key: str) -> tuple[Path, list[tuple[int, int]]]:
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        item = _item_row(connection, display_key)
        root = validate_quarantine_root(root_arg)
        media = _resolve_private_confined(root, item["media_relpath"], "E_MEDIA_INVALID")
        if _sha256_file(media) != item["media_sha256"]:
            _fail("E_MEDIA_DIGEST")
        segments = connection.execute("SELECT start_ms,end_ms FROM audit_segments WHERE item_id=? ORDER BY segment_index", (item["item_id"],)).fetchall()
        return media, [(row["start_ms"], row["end_ms"]) for row in segments]


def own_item_record(root_arg: str | os.PathLike[str], *, author_token: str, display_key: str) -> dict[str, Any]:
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        item = _item_row(connection, display_key)
        label = connection.execute("SELECT * FROM item_labels WHERE item_id=?", (item["item_id"],)).fetchone()
        utterances = connection.execute(
            "SELECT utterance_id,display_utterance_key,segment_index,onset_ms,offset_ms,source_text,speaker_role,referential_status,noun_object_decision,verb_action_decision,uncertain_unusable_reason,locked FROM author_utterances WHERE item_id=? ORDER BY segment_index,onset_ms,utterance_id",
            (item["item_id"],),
        ).fetchall()
        mentions = connection.execute(
            """SELECT m.mention_id,u.display_utterance_key,m.display_mention_key,
                      m.mention_family,m.mention_start_char,m.mention_end_char,
                      m.referential_status,m.candidate_count_band,
                      m.visible_segment_index,m.visible_onset_ms,m.visible_offset_ms,
                      m.uncertain_unusable_reason,m.locked
               FROM author_mentions m JOIN author_utterances u ON u.utterance_id=m.utterance_id
               WHERE u.item_id=? ORDER BY u.utterance_id,m.mention_id""",
            (item["item_id"],),
        ).fetchall()
        return {
            "display_key": display_key,
            "item_label": dict(label) if label else None,
            "utterances": [dict(row) for row in utterances],
            "mentions": [dict(row) for row in mentions],
        }


def save_item_label(root_arg: str | os.PathLike[str], *, author_token: str, display_key: str, disposition: str, language_code: str, language_competence: str, audio_usability: str, wer_applicability: str = "UNDECIDABLE") -> dict[str, Any]:
    if disposition not in ITEM_DISPOSITIONS or audio_usability not in AUDIO_USABILITY_VALUES:
        _fail("E_ITEM_LABEL")
    if language_competence not in {"NATIVE", "FLUENT", "PROFICIENT", "INSUFFICIENT", "UNDECIDABLE"}:
        _fail("E_LANGUAGE_COMPETENCE")
    if wer_applicability not in {"APPLICABLE", "NOT_APPLICABLE", "UNDECIDABLE"}:
        _fail("E_WER_APPLICABILITY")
    language = _validate_language(language_code)
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        _ensure_unlocked(connection)
        item = _item_row(connection, display_key)
        existing = connection.execute("SELECT locked FROM item_labels WHERE item_id=?", (item["item_id"],)).fetchone()
        if existing and existing["locked"]:
            _fail("E_ITEM_LOCKED")
        connection.execute(
            """INSERT INTO item_labels(item_id,disposition,language_code,language_competence,wer_applicability,audio_usability,locked,updated_at_utc)
               VALUES (?,?,?,?,?,?,0,?) ON CONFLICT(item_id) DO UPDATE SET
               disposition=excluded.disposition,language_code=excluded.language_code,
               language_competence=excluded.language_competence,wer_applicability=excluded.wer_applicability,
               audio_usability=excluded.audio_usability,
               updated_at_utc=excluded.updated_at_utc""",
            (item["item_id"], disposition, language, language_competence, wer_applicability, audio_usability, _utc_now()),
        )
        connection.execute("INSERT INTO audit_log(created_at_utc,event) VALUES (?,?)", (_utc_now(), "ITEM_LABEL_AUTOSAVED"))
    return {"status": "saved"}


def _validate_candidate_combination(status: str, noun: str, verb: str) -> None:
    if status not in REFERENTIAL_VALUES or noun not in CANDIDATE_VALUES or verb not in CANDIDATE_VALUES:
        _fail("E_REFERENTIAL_LABEL")
    if status in {"IRRELEVANT", "UNDECIDABLE", "UNUSABLE"} and (noun != "NOT_APPLICABLE" or verb != "NOT_APPLICABLE"):
        _fail("E_REFERENTIAL_COMBINATION")
    if status in {"VISIBLE_CANDIDATE", "NULL_NOT_VISIBLE"} and noun == "NOT_APPLICABLE" and verb == "NOT_APPLICABLE":
        _fail("E_REFERENTIAL_COMBINATION")


def save_utterance(root_arg: str | os.PathLike[str], *, author_token: str, display_key: str, display_utterance_key: str, segment_index: int, onset_ms: int, offset_ms: int, source_text: str, speaker_role: str, referential_status: str, noun_object_decision: str, verb_action_decision: str, uncertain_unusable_reason: str = "") -> dict[str, Any]:
    utterance_key = display_utterance_key.strip()
    if not re.fullmatch(r"U-[0-9]{4}", utterance_key):
        _fail("E_UTTERANCE_KEY")
    if speaker_role not in ROLE_VALUES:
        _fail("E_ROLE")
    _validate_candidate_combination(referential_status, noun_object_decision, verb_action_decision)
    text_value = source_text.strip()
    reason = uncertain_unusable_reason.strip()
    if not text_value or len(text_value) > 4000:
        _fail("E_SOURCE_TEXT")
    if len(reason) > 500:
        _fail("E_REASON")
    if not isinstance(segment_index, int) or isinstance(segment_index, bool) or not isinstance(onset_ms, int) or not isinstance(offset_ms, int):
        _fail("E_UTTERANCE_TIMING")
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        _ensure_unlocked(connection)
        item = _item_row(connection, display_key)
        label = connection.execute("SELECT locked FROM item_labels WHERE item_id=?", (item["item_id"],)).fetchone()
        if label and label["locked"]:
            _fail("E_ITEM_LOCKED")
        segment = connection.execute("SELECT start_ms,end_ms FROM audit_segments WHERE item_id=? AND segment_index=?", (item["item_id"], segment_index)).fetchone()
        if not segment or onset_ms < 0 or offset_ms <= onset_ms or offset_ms > segment["end_ms"] - segment["start_ms"]:
            _fail("E_UTTERANCE_TIMING")
        existing = connection.execute("SELECT locked FROM author_utterances WHERE item_id=? AND display_utterance_key=?", (item["item_id"], utterance_key)).fetchone()
        if existing and existing["locked"]:
            _fail("E_UTTERANCE_LOCKED")
        connection.execute(
            """INSERT INTO author_utterances(item_id,display_utterance_key,segment_index,onset_ms,offset_ms,source_text,speaker_role,referential_status,noun_object_decision,verb_action_decision,uncertain_unusable_reason,locked,updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?) ON CONFLICT(item_id,display_utterance_key) DO UPDATE SET
               segment_index=excluded.segment_index,onset_ms=excluded.onset_ms,offset_ms=excluded.offset_ms,
               source_text=excluded.source_text,speaker_role=excluded.speaker_role,
               referential_status=excluded.referential_status,noun_object_decision=excluded.noun_object_decision,
               verb_action_decision=excluded.verb_action_decision,
               uncertain_unusable_reason=excluded.uncertain_unusable_reason,
               updated_at_utc=excluded.updated_at_utc""",
            (item["item_id"], utterance_key, segment_index, onset_ms, offset_ms, text_value, speaker_role, referential_status, noun_object_decision, verb_action_decision, reason, _utc_now()),
        )
        connection.execute("INSERT INTO audit_log(created_at_utc,event) VALUES (?,?)", (_utc_now(), "UTTERANCE_AUTOSAVED"))
    return {"status": "saved"}


def delete_draft_utterance(root_arg: str | os.PathLike[str], *, author_token: str, display_key: str, display_utterance_key: str) -> dict[str, Any]:
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        _ensure_unlocked(connection)
        item = _item_row(connection, display_key)
        row = connection.execute("SELECT locked FROM author_utterances WHERE item_id=? AND display_utterance_key=?", (item["item_id"], display_utterance_key)).fetchone()
        if not row:
            _fail("E_UTTERANCE_NOT_FOUND")
        if row["locked"]:
            _fail("E_UTTERANCE_LOCKED")
        utterance = connection.execute(
            "SELECT utterance_id FROM author_utterances WHERE item_id=? AND display_utterance_key=?",
            (item["item_id"], display_utterance_key),
        ).fetchone()
        connection.execute(
            "DELETE FROM author_mentions WHERE utterance_id=?", (utterance["utterance_id"],)
        )
        connection.execute("DELETE FROM author_utterances WHERE item_id=? AND display_utterance_key=?", (item["item_id"], display_utterance_key))
    return {"status": "deleted"}


def save_mention(
    root_arg: str | os.PathLike[str],
    *,
    author_token: str,
    display_key: str,
    display_utterance_key: str,
    display_mention_key: str,
    mention_family: str,
    mention_start_char: int,
    mention_end_char: int,
    referential_status: str,
    candidate_count_band: str,
    visible_segment_index: int | None = None,
    visible_onset_ms: int | None = None,
    visible_offset_ms: int | None = None,
    uncertain_unusable_reason: str = "",
) -> dict[str, Any]:
    mention_key = display_mention_key.strip()
    if not re.fullmatch(r"M-[0-9]{4}", mention_key):
        _fail("E_MENTION_KEY")
    if mention_family not in MENTION_FAMILIES:
        _fail("E_MENTION_FAMILY")
    if referential_status not in {
        "VISIBLE_CANDIDATE",
        "NULL_NOT_VISIBLE",
        "UNDECIDABLE",
        "UNUSABLE",
    } or candidate_count_band not in CANDIDATE_COUNT_BANDS:
        _fail("E_MENTION_LABEL")
    reason = uncertain_unusable_reason.strip()
    if len(reason) > 500:
        _fail("E_REASON")
    if referential_status == "VISIBLE_CANDIDATE":
        if candidate_count_band not in {"ONE", "TWO", "THREE_PLUS"}:
            _fail("E_MENTION_COMBINATION")
        if None in (visible_segment_index, visible_onset_ms, visible_offset_ms):
            _fail("E_VISIBLE_BAND_REQUIRED")
    elif referential_status == "NULL_NOT_VISIBLE":
        if candidate_count_band != "ZERO" or any(
            value is not None
            for value in (visible_segment_index, visible_onset_ms, visible_offset_ms)
        ):
            _fail("E_MENTION_COMBINATION")
    else:
        if candidate_count_band != "UNKNOWN" or any(
            value is not None
            for value in (visible_segment_index, visible_onset_ms, visible_offset_ms)
        ) or not reason:
            _fail("E_MENTION_COMBINATION")
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        _ensure_unlocked(connection)
        item = _item_row(connection, display_key)
        label = connection.execute(
            "SELECT locked FROM item_labels WHERE item_id=?", (item["item_id"],)
        ).fetchone()
        if label and label["locked"]:
            _fail("E_ITEM_LOCKED")
        utterance = connection.execute(
            """SELECT utterance_id,source_text,locked FROM author_utterances
               WHERE item_id=? AND display_utterance_key=?""",
            (item["item_id"], display_utterance_key),
        ).fetchone()
        if not utterance:
            _fail("E_UTTERANCE_NOT_FOUND")
        if utterance["locked"]:
            _fail("E_UTTERANCE_LOCKED")
        if (
            not isinstance(mention_start_char, int)
            or isinstance(mention_start_char, bool)
            or not isinstance(mention_end_char, int)
            or isinstance(mention_end_char, bool)
            or mention_start_char < 0
            or mention_end_char <= mention_start_char
            or mention_end_char > len(utterance["source_text"])
        ):
            _fail("E_MENTION_SPAN")
        if referential_status == "VISIBLE_CANDIDATE":
            segment = connection.execute(
                "SELECT start_ms,end_ms FROM audit_segments WHERE item_id=? AND segment_index=?",
                (item["item_id"], visible_segment_index),
            ).fetchone()
            if (
                not segment
                or not isinstance(visible_onset_ms, int)
                or not isinstance(visible_offset_ms, int)
                or visible_onset_ms < 0
                or visible_offset_ms <= visible_onset_ms
                or visible_offset_ms > segment["end_ms"] - segment["start_ms"]
            ):
                _fail("E_VISIBLE_BAND")
        existing = connection.execute(
            "SELECT locked FROM author_mentions WHERE utterance_id=? AND display_mention_key=?",
            (utterance["utterance_id"], mention_key),
        ).fetchone()
        if existing and existing["locked"]:
            _fail("E_MENTION_LOCKED")
        connection.execute(
            """INSERT INTO author_mentions(
               utterance_id,display_mention_key,mention_family,
               mention_start_char,mention_end_char,referential_status,
               candidate_count_band,visible_segment_index,visible_onset_ms,
               visible_offset_ms,uncertain_unusable_reason,locked,updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)
               ON CONFLICT(utterance_id,display_mention_key) DO UPDATE SET
               mention_family=excluded.mention_family,
               mention_start_char=excluded.mention_start_char,
               mention_end_char=excluded.mention_end_char,
               referential_status=excluded.referential_status,
               candidate_count_band=excluded.candidate_count_band,
               visible_segment_index=excluded.visible_segment_index,
               visible_onset_ms=excluded.visible_onset_ms,
               visible_offset_ms=excluded.visible_offset_ms,
               uncertain_unusable_reason=excluded.uncertain_unusable_reason,
               updated_at_utc=excluded.updated_at_utc""",
            (
                utterance["utterance_id"], mention_key, mention_family,
                mention_start_char, mention_end_char, referential_status,
                candidate_count_band, visible_segment_index, visible_onset_ms,
                visible_offset_ms, reason, _utc_now(),
            ),
        )
        connection.execute(
            "INSERT INTO audit_log(created_at_utc,event) VALUES (?,?)",
            (_utc_now(), "MENTION_AUTOSAVED"),
        )
    return {"status": "saved"}


def delete_draft_mention(
    root_arg: str | os.PathLike[str],
    *,
    author_token: str,
    display_key: str,
    display_utterance_key: str,
    display_mention_key: str,
) -> dict[str, Any]:
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        _ensure_unlocked(connection)
        item = _item_row(connection, display_key)
        row = connection.execute(
            """SELECT m.mention_id,m.locked FROM author_mentions m
               JOIN author_utterances u ON u.utterance_id=m.utterance_id
               WHERE u.item_id=? AND u.display_utterance_key=?
               AND m.display_mention_key=?""",
            (item["item_id"], display_utterance_key, display_mention_key),
        ).fetchone()
        if not row:
            _fail("E_MENTION_NOT_FOUND")
        if row["locked"]:
            _fail("E_MENTION_LOCKED")
        connection.execute("DELETE FROM author_mentions WHERE mention_id=?", (row["mention_id"],))
    return {"status": "deleted"}


def lock_item(root_arg: str | os.PathLike[str], *, author_token: str, display_key: str) -> dict[str, Any]:
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        _ensure_unlocked(connection)
        item = _item_row(connection, display_key)
        label = connection.execute("SELECT * FROM item_labels WHERE item_id=?", (item["item_id"],)).fetchone()
        if not label:
            _fail("E_ITEM_LABEL_REQUIRED")
        if label["locked"]:
            return {"status": "already_locked"}
        utterance_count = connection.execute("SELECT COUNT(*) FROM author_utterances WHERE item_id=?", (item["item_id"],)).fetchone()[0]
        if label["disposition"] == "ANNOTATED" and utterance_count == 0:
            _fail("E_UTTERANCE_REQUIRED")
        if label["disposition"] != "ANNOTATED" and utterance_count != 0:
            _fail("E_DISPOSITION_UTTERANCE_CONFLICT")
        utterances = connection.execute(
            """SELECT utterance_id,referential_status,noun_object_decision,
                      verb_action_decision,uncertain_unusable_reason
               FROM author_utterances WHERE item_id=?""",
            (item["item_id"],),
        ).fetchall()
        for utterance in utterances:
            families = {
                row[0]
                for row in connection.execute(
                    "SELECT mention_family FROM author_mentions WHERE utterance_id=?",
                    (utterance["utterance_id"],),
                )
            }
            if utterance["referential_status"] in {
                "VISIBLE_CANDIDATE",
                "NULL_NOT_VISIBLE",
            } and not families:
                _fail("E_MENTION_REQUIRED")
            if utterance["referential_status"] == "IRRELEVANT" and families:
                _fail("E_IRRELEVANT_MENTION_CONFLICT")
            if utterance["referential_status"] in {"UNDECIDABLE", "UNUSABLE"} and not utterance["uncertain_unusable_reason"]:
                _fail("E_REASON_REQUIRED")
            for family, field in (
                ("NOUN_OBJECT", "noun_object_decision"),
                ("VERB_ACTION", "verb_action_decision"),
            ):
                decision = utterance[field]
                if decision == "PRESENT" and family not in families:
                    _fail("E_MENTION_REQUIRED")
                if decision in {"ABSENT", "NOT_APPLICABLE"} and family in families:
                    _fail("E_CANDIDATE_DECISION_CONFLICT")
        connection.execute("UPDATE author_utterances SET locked=1 WHERE item_id=?", (item["item_id"],))
        connection.execute(
            """UPDATE author_mentions SET locked=1 WHERE utterance_id IN
               (SELECT utterance_id FROM author_utterances WHERE item_id=?)""",
            (item["item_id"],),
        )
        connection.execute("UPDATE item_labels SET locked=1,updated_at_utc=? WHERE item_id=?", (_utc_now(), item["item_id"]))
        connection.execute("INSERT INTO audit_log(created_at_utc,event) VALUES (?,?)", (_utc_now(), "ITEM_IRREVERSIBLY_LOCKED"))
    return {"status": "locked"}


def _author_record_payload(connection: sqlite3.Connection) -> dict[str, Any]:
    meta = connection.execute(
        "SELECT protocol_digest,sample_digest,packet_digest FROM workflow_meta WHERE singleton=1"
    ).fetchone()
    labels = [dict(row) for row in connection.execute("SELECT * FROM item_labels ORDER BY item_id")]
    utterances = [dict(row) for row in connection.execute("SELECT * FROM author_utterances ORDER BY item_id,segment_index,onset_ms,utterance_id")]
    mentions = [dict(row) for row in connection.execute("SELECT * FROM author_mentions ORDER BY utterance_id,mention_id")]
    return {
        "route": ROUTE,
        "protocol_digest": meta["protocol_digest"],
        "sample_digest": meta["sample_digest"],
        "packet_digest": meta["packet_digest"],
        "completeness": {
            "locked_item_count": sum(1 for row in labels if row["locked"] == 1),
            "expected_item_count": FROZEN_ITEM_COUNT,
            "utterance_count": len(utterances),
            "mention_count": len(mentions),
        },
        "labels": labels,
        "utterances": utterances,
        "mentions": mentions,
        "inter_human_reliability_available": False,
    }


def lock_author_pass(root_arg: str | os.PathLike[str], *, author_token: str, confirm_blinded: bool) -> dict[str, Any]:
    if confirm_blinded is not True:
        _fail("E_BLINDING_CONFIRMATION_REQUIRED")
    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        meta = connection.execute("SELECT author_pass_locked_at_utc,author_record_hmac FROM workflow_meta WHERE singleton=1").fetchone()
        if meta["author_pass_locked_at_utc"]:
            return {"status": "already_locked", "human_audit_complete": True}
        locked = connection.execute("SELECT COUNT(*) FROM item_labels WHERE locked=1").fetchone()[0]
        if locked != FROZEN_ITEM_COUNT:
            _fail("E_ALL_ITEMS_NOT_LOCKED")
        payload = _author_record_payload(connection)
        record_hmac = hmac.new(secret, b"author-record\0" + _canonical(payload), hashlib.sha256).hexdigest()
        connection.execute(
            "UPDATE workflow_meta SET author_pass_locked_at_utc=?,author_record_hmac=?,workflow_state='AUTHOR_LOCKED' WHERE singleton=1",
            (_utc_now(), record_hmac),
        )
        connection.execute("INSERT INTO audit_log(created_at_utc,event) VALUES (?,?)", (_utc_now(), "AUTHOR_PASS_IRREVERSIBLY_LOCKED"))
    return {"status": "locked", "human_audit_complete": True, "prediction_binding_may_now_be_attached": True}


def attach_prediction_store_after_lock(
    root_arg: str | os.PathLike[str],
    prediction_binding_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Bind the prediction store only after the author record is immutable."""

    root = validate_quarantine_root(root_arg)
    relative = _confined_input_relative(
        root, prediction_binding_path, "E_PREDICTION_BINDING_FILE"
    )
    binding_file = _resolve_private_confined(
        root, relative, "E_PREDICTION_BINDING_FILE"
    )
    binding = _load_json_file(binding_file, "E_PREDICTION_BINDING_FILE")
    with _db(root) as (connection, _secret, _):
        meta = connection.execute(
            "SELECT sample_digest,workflow_state,prediction_binding_digest FROM workflow_meta WHERE singleton=1"
        ).fetchone()
        if not meta or meta["workflow_state"] not in {
            "AUTHOR_LOCKED",
            "PREDICTION_JOIN_ENABLED",
        }:
            _fail("E_PREDICTION_BIND_REQUIRES_AUTHOR_LOCK")
        store_relpath, store_sha = _validate_prediction_binding(
            binding, root, meta["sample_digest"]
        )
        binding_digest = _sha256(_canonical(binding))
        if meta["workflow_state"] == "PREDICTION_JOIN_ENABLED":
            if meta["prediction_binding_digest"] != binding_digest:
                _fail("E_PREDICTION_BINDING_IMMUTABLE")
            return {
                "status": "already_attached",
                "workflow_state": "PREDICTION_JOIN_ENABLED",
            }
        connection.execute(
            """UPDATE workflow_meta SET workflow_state='PREDICTION_JOIN_ENABLED',
               prediction_binding_digest=?,prediction_store_relpath=?,prediction_store_sha256=?
               WHERE singleton=1""",
            (binding_digest, store_relpath, store_sha),
        )
        connection.execute(
            "INSERT INTO audit_log(created_at_utc,event) VALUES (?,?)",
            (_utc_now(), "PREDICTION_JOIN_ENABLED_AFTER_AUTHOR_LOCK"),
        )
        _atomic_write(
            root / WORKFLOW_DIR / PREDICTION_BINDING_SNAPSHOT_FILE,
            _canonical(binding) + b"\n",
        )
    return {"status": "attached", "workflow_state": "PREDICTION_JOIN_ENABLED"}


def open_predictions_after_lock(root_arg: str | os.PathLike[str], *, author_token: str) -> Mapping[str, Any]:
    """Return restricted pseudo-labels only after the human record is immutable."""

    with _db(root_arg) as (connection, secret, _):
        _authorize(connection, secret, author_token)
        meta = connection.execute("SELECT workflow_state,prediction_store_relpath,prediction_store_sha256 FROM workflow_meta WHERE singleton=1").fetchone()
        if not meta or meta["workflow_state"] != "PREDICTION_JOIN_ENABLED":
            _fail("E_PREDICTIONS_BLINDED_UNTIL_AUTHOR_LOCK")
        root = validate_quarantine_root(root_arg)
        store = _resolve_private_confined(root, meta["prediction_store_relpath"], "E_PREDICTION_STORE_INVALID")
        if _sha256_file(store) != meta["prediction_store_sha256"]:
            _fail("E_PREDICTION_STORE_DIGEST")
        try:
            value = json.loads(store.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            _fail("E_PREDICTION_STORE_INVALID")
        if not isinstance(value, Mapping):
            _fail("E_PREDICTION_STORE_INVALID")
        return value


def progress(root_arg: str | os.PathLike[str]) -> dict[str, Any]:
    with _db(root_arg) as (connection, _secret, _):
        meta = connection.execute("SELECT estimated_author_minutes,author_pass_locked_at_utc,workflow_state FROM workflow_meta WHERE singleton=1").fetchone()
        if not meta:
            _fail("E_NOT_INITIALIZED")
        locked = connection.execute("SELECT COUNT(*) FROM item_labels WHERE locked=1").fetchone()[0]
        drafts = connection.execute("SELECT COUNT(*) FROM item_labels WHERE locked=0").fetchone()[0]
        utterances = connection.execute("SELECT COUNT(*) FROM author_utterances").fetchone()[0]
        estimate = meta["estimated_author_minutes"]
        remaining = math.ceil(estimate * (FROZEN_ITEM_COUNT - locked) / FROZEN_ITEM_COUNT)
        return {
            "route": ROUTE,
            "audit_item_count": FROZEN_ITEM_COUNT,
            "audit_speech_minutes": FROZEN_TOTAL_DURATION_MS / 60_000,
            "locked_item_count": locked,
            "draft_item_count": drafts,
            "remaining_item_count": FROZEN_ITEM_COUNT - locked,
            "utterance_count": utterances,
            "estimated_author_minutes": estimate,
            "estimated_remaining_minutes": remaining,
            "human_audit_complete": meta["author_pass_locked_at_utc"] is not None,
            "workflow_state": meta["workflow_state"],
        }


def aggregate_receipt(root_arg: str | os.PathLike[str]) -> dict[str, Any]:
    status = progress(root_arg)
    return {
        "schema_version": RECEIPT_VERSION,
        "status": "READY",
        "workflow_ready": True,
        "audit_item_count": status["audit_item_count"],
        "audit_speech_minutes": status["audit_speech_minutes"],
        "audit_speech_seconds": FROZEN_TOTAL_DURATION_MS // 1000,
        "route": ROUTE,
        "loopback_only": True,
        "owner_private": True,
        "autosave": True,
        "autosave_to_quarantine_only": True,
        "resumable": True,
        "predictions_hidden": status["workflow_state"] != "PREDICTION_JOIN_ENABLED",
        "predictions_hidden_before_author_lock": True,
        "author_record_lock_immutable": True,
        "comparison_requires_lock": True,
        "selected_independent_of_model_outputs": True,
        "sample_independent_of_model_outputs": True,
        "uncertain_unusable_routes": True,
        "uncertain_route_available": True,
        "unusable_route_available": True,
        "qualification_instruction_present": True,
        "external_hosting": False,
        "network_exposure": False,
        "model_predictions_revealed_before_lock": False,
        "human_evidence_fabricated": False,
        "estimated_author_minutes": status["estimated_author_minutes"],
        "human_audit_complete": status["human_audit_complete"],
        "prediction_join_enabled": status["workflow_state"] == "PREDICTION_JOIN_ENABLED",
        "inter_human_reliability_available": False,
        "primary_evaluation_truth": "SIMULATOR_ORACLE_ONLY",
        "unaudited_pseudo_label_permitted_uses": ["AGGREGATE_CALIBRATION", "CANDIDATE_GENERATION"],
        "unaudited_pseudo_label_primary_evaluation_truth": False,
    }


def validate_workflow(root_arg: str | os.PathLike[str]) -> dict[str, Any]:
    root = validate_quarantine_root(root_arg)
    with _db(root) as (connection, secret, database):
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            _fail("E_DATABASE_INTEGRITY")
        meta = connection.execute("SELECT * FROM workflow_meta WHERE singleton=1").fetchone()
        if not meta or meta["workflow_version"] != VERSION or meta["route"] != ROUTE:
            _fail("E_WORKFLOW_META")
        _assert_schema_compatible(connection)
        if _sha256_file(PROTOCOL_FILE) != meta["protocol_digest"]:
            _fail("E_PROTOCOL_DIGEST")
        if connection.execute("SELECT COUNT(*) FROM audit_items").fetchone()[0] != FROZEN_ITEM_COUNT:
            _fail("E_ITEM_COUNT")
        if connection.execute("SELECT SUM(duration_ms) FROM audit_items").fetchone()[0] != FROZEN_TOTAL_DURATION_MS:
            _fail("E_TOTAL_DURATION")
        if not _private_file(database) or not _private_file(root / WORKFLOW_DIR / SECRET_FILE):
            _fail("E_RUNTIME_PERMISSIONS")
        sample = _load_json_file(root / WORKFLOW_DIR / SAMPLE_SNAPSHOT_FILE, "E_SAMPLE_SNAPSHOT")
        if _sha256(_canonical(sample)) != meta["packet_digest"]:
            _fail("E_SAMPLE_SNAPSHOT_DIGEST")
        if meta["workflow_state"] == "PREDICTION_JOIN_ENABLED":
            binding = _load_json_file(
                root / WORKFLOW_DIR / PREDICTION_BINDING_SNAPSHOT_FILE,
                "E_PREDICTION_BINDING_SNAPSHOT",
            )
            if _sha256(_canonical(binding)) != meta["prediction_binding_digest"]:
                _fail("E_PREDICTION_BINDING_DIGEST")
            store = _resolve_private_confined(
                root,
                meta["prediction_store_relpath"],
                "E_PREDICTION_STORE_INVALID",
            )
            if _sha256_file(store) != meta["prediction_store_sha256"]:
                _fail("E_PREDICTION_STORE_DIGEST")
        elif any(
            meta[key] is not None
            for key in (
                "prediction_binding_digest",
                "prediction_store_relpath",
                "prediction_store_sha256",
            )
        ):
            _fail("E_PRELOCK_PREDICTION_MOUNT")
        if meta["author_pass_locked_at_utc"]:
            expected = hmac.new(secret, b"author-record\0" + _canonical(_author_record_payload(connection)), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, meta["author_record_hmac"] or ""):
                _fail("E_AUTHOR_RECORD_INTEGRITY")
    return {"status": "STRUCTURAL_PASS", "workflow_ready": True, "human_audit_complete": aggregate_receipt(root)["human_audit_complete"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    sub = parser.add_subparsers(dest="command", required=True)
    bootstrap = sub.add_parser("bootstrap-policy")
    bootstrap.add_argument("--root", required=True)
    bootstrap.add_argument("--attest-all-controls", action="store_true")
    init = sub.add_parser("init")
    init.add_argument("--root", required=True)
    init.add_argument("--packet", required=True)
    attach = sub.add_parser("attach-predictions-after-lock")
    attach.add_argument("--root", required=True)
    attach.add_argument("--prediction-binding", required=True)
    for name in ("validate", "readiness"):
        route = sub.add_parser(name)
        route.add_argument("--root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "bootstrap-policy":
            bootstrap_policy(args.root, attest=args.attest_all_controls)
            result: Mapping[str, Any] = {"status": "policy_created"}
        elif args.command == "init":
            result = initialize(args.root, args.packet)
        elif args.command == "attach-predictions-after-lock":
            result = attach_prediction_store_after_lock(
                args.root, args.prediction_binding
            )
        elif args.command == "validate":
            result = validate_workflow(args.root)
        else:
            result = aggregate_receipt(args.root)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except WorkflowError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("E_INTERNAL", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
