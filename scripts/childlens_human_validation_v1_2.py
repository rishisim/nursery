#!/usr/bin/env python3
"""Fail-closed local workflow for the ChildLens v1.2 human pilot.

This module never discovers media or corpus metadata.  It consumes a restricted,
opaque packet manifest prepared inside the authorized quarantine and keeps every
row-level record there.  Repository-safe output is limited to aggregate status
and validation receipts that contain no item keys or lexical content.
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
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


VERSION = "childlens-human-validation-workflow-v1.2.0"
SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_ENV = "CHILDLENS_V12_QUARANTINE_ROOT"
DISCOVERY_SEARCH_ROOT = REPO_ROOT.parent
POLICY_FILE = ".childlens_v1_2_quarantine_policy.json"
INDEX_SENTINEL = ".metadata_never_index"
WORKFLOW_DIR = "human_validation_v1_2"
DATABASE_FILE = "validation.sqlite3"
SECRET_FILE = ".workflow_hmac_key"
CODER_ASSIGNMENTS_FILE = "coder_assignment_secrets.json"
RETENTION_DEADLINE = "2027-07-31"
FROZEN_SELECTED_COUNT = 15
FROZEN_V1_1_HUMAN_PACKET_SHA256 = "ae8503fc5c21fc6df0b08c763202eab074b2526f83ef847d2faca3ae4eb217a5"
FROZEN_V1_1_RESTRICTED_INPUT_SHA256 = "81282e6a4d8178b1559a7473d22c70a36e44450467b6b5e2ff70ad9b7e9f049a"

LANGUAGE_SPECIAL = frozenset({"MIXED_OR_CODE_SWITCHED", "UNDECIDABLE"})
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
ROLE_VALUES = frozenset({"NON_CHILD", "CHILD", "OVERLAP", "UNCERTAIN", "NONSPEECH"})
REFERENTIAL_VALUES = frozenset(
    {
        "VISIBLE_SINGLE",
        "VISIBLE_MULTIPLE",
        "NULL_NOT_VISIBLE",
        "IRRELEVANT",
        "UNDECIDABLE",
        "UNUSABLE",
    }
)
MENTION_FAMILIES = frozenset({"NOUN_OBJECT", "VERB_ACTION", "BOTH", "NEITHER"})
CANDIDATE_BANDS = frozenset({"ZERO", "ONE", "TWO", "THREE_PLUS", "UNKNOWN"})
AUDIO_INTEGRITY_VALUES = frozenset({"USABLE", "PARTIAL", "UNUSABLE", "UNDECIDABLE"})
LANGUAGE_SLOTS = ("LANGUAGE_A", "LANGUAGE_B")
TIMING_SLOTS = ("TIMING_A", "TIMING_B")
REFERENTIAL_SLOTS = ("REFERENTIAL_A", "REFERENTIAL_B")
ADJUDICATOR_SLOT = "ADJUDICATOR"
SLOTS = frozenset((*LANGUAGE_SLOTS, *TIMING_SLOTS, *REFERENTIAL_SLOTS, ADJUDICATOR_SLOT))
INDEPENDENCE_PAIRS = (
    frozenset(LANGUAGE_SLOTS),
    frozenset(TIMING_SLOTS),
    frozenset(REFERENTIAL_SLOTS),
)
OPAQUE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{12,80}$")
DISPLAY_KEY_RE = re.compile(r"^HV-[0-9]{3,5}$")
OPAQUE_MEDIA_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{11,79}\.(?:mp4|mov|mkv|webm|wav|m4a)$")
CONTENT_ADDRESS_RE = re.compile(r"^[0-9a-f]{64}\.bin$")
FORBIDDEN_MANIFEST_KEYS = frozenset(
    {
        "participant_id",
        "participant",
        "session_id",
        "session",
        "source_filename",
        "filename",
        "source_path",
        "absolute_path",
        "transcript",
        "text",
        "exact_timestamp",
        "recording_date",
    }
)

# Set only by ``discover_runtime_root`` after a unique initialized workflow has
# passed the same filesystem and policy checks as the explicit CLI route.  This
# process-local capability lets the no-argument UI avoid putting the restricted
# root in argv or the environment.  It is never serialized or exported.
_DISCOVERED_RUNTIME_ROOT: Path | None = None


class WorkflowError(RuntimeError):
    """An intentionally terse error safe for logs and UI."""

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


def _load_policy(root: Path) -> Mapping[str, Any]:
    policy_path = root / POLICY_FILE
    if not policy_path.is_file() or policy_path.is_symlink():
        _fail("E_QUARANTINE_POLICY_MISSING")
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _fail("E_QUARANTINE_POLICY_INVALID")
    required_true = (
        "owner_only_verified",
        "git_exclusion_verified",
        "indexing_exclusion_verified",
        "backup_exclusion_verified_or_encrypted_local_only",
        "local_only",
    )
    if policy.get("schema_version") != VERSION or any(policy.get(k) is not True for k in required_true):
        _fail("E_QUARANTINE_POLICY_INVALID")
    if policy.get("retention_deadline") != RETENTION_DEADLINE:
        _fail("E_RETENTION_POLICY_MISMATCH")
    if date.today() > date.fromisoformat(RETENTION_DEADLINE):
        _fail("E_RETENTION_DEADLINE_PASSED")
    return policy


def _validate_root_controls(root_arg: str | os.PathLike[str]) -> Path:
    """Validate the filesystem and policy controls without authorizing a root."""

    requested = Path(root_arg).expanduser()
    if not requested.is_absolute():
        _fail("E_QUARANTINE_ROOT_NOT_ABSOLUTE")
    try:
        root = requested.resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_ROOT_MISSING")
    _reject_symlink_components(root)
    if _is_relative_to(root, REPO_ROOT) or root == REPO_ROOT:
        _fail("E_QUARANTINE_INSIDE_REPOSITORY")
    if not root.is_dir() or root.is_symlink():
        _fail("E_QUARANTINE_ROOT_INVALID")
    metadata = root.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        _fail("E_QUARANTINE_NOT_OWNER_ONLY")
    if not (root / INDEX_SENTINEL).is_file():
        _fail("E_INDEXING_SENTINEL_MISSING")
    _load_policy(root)
    return root


def validate_quarantine_root(root_arg: str | os.PathLike[str]) -> Path:
    """Resolve and validate an explicitly or locally discovered restricted root.

    Administrative CLI calls retain the path-plus-environment double opt-in.
    The no-argument human UI instead receives a process-local authorization only
    after :func:`discover_runtime_root` finds exactly one initialized workflow.
    """

    requested = Path(root_arg).expanduser()
    if not requested.is_absolute():
        _fail("E_QUARANTINE_ROOT_NOT_ABSOLUTE")
    try:
        requested_resolved = requested.resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_ROOT_MISSING")
    env_value = os.environ.get(ROOT_ENV)
    if env_value:
        env_path = Path(env_value).expanduser()
        if not env_path.is_absolute():
            _fail("E_QUARANTINE_ROOT_NOT_ABSOLUTE")
        try:
            authorized = env_path.resolve(strict=True)
        except OSError:
            _fail("E_QUARANTINE_ROOT_MISSING")
    elif _DISCOVERED_RUNTIME_ROOT is not None:
        authorized = _DISCOVERED_RUNTIME_ROOT
    else:
        _fail("E_QUARANTINE_ENV_MISSING")
    if requested_resolved != authorized:
        _fail("E_QUARANTINE_ROOT_MISMATCH")
    return _validate_root_controls(requested_resolved)


def _private_runtime_file(path: Path) -> bool:
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


def discover_runtime_root(
    *,
    search_root: Path = DISCOVERY_SEARCH_ROOT,
    repository_root: Path = REPO_ROOT,
) -> Path:
    """Authorize exactly one initialized private workflow without path input.

    Discovery is deliberately narrow: only owner-only, ChildLens-labelled hidden
    siblings of the repository are searched, symlinks and non-private directory
    chains are pruned, and a candidate must already contain the policy, no-index
    sentinel, database, HMAC secret, and coder-assignment receipt.  Errors never
    include candidate names or paths.
    """

    global _DISCOVERED_RUNTIME_ROOT
    # A failed or ambiguous rediscovery must revoke any authorization retained
    # from an earlier call in this process.
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
            and not candidate.is_symlink()
            and candidate.is_dir()
            and candidate.stat().st_uid == os.getuid()
            and stat.S_IMODE(candidate.stat().st_mode) & 0o077 == 0
        ]
    except OSError:
        _fail("E_RUNTIME_DISCOVERY_ROOT")

    candidates: list[Path] = []
    for hidden_root in sorted(hidden_roots, key=lambda value: value.name):
        try:
            walker = os.walk(hidden_root, followlinks=False)
            for directory, directories, _files in walker:
                current = Path(directory)
                if not _private_discovery_directory(current, hidden_root):
                    directories[:] = []
                    continue
                directories[:] = [
                    name
                    for name in directories
                    if not (current / name).is_symlink()
                    and _private_discovery_directory(current / name, hidden_root)
                    and name not in {"raw_v1_2", "browser_profile", "browser_cache", "browser_downloads"}
                ]
                workflow_dir = current / WORKFLOW_DIR
                required_files = (
                    current / POLICY_FILE,
                    current / INDEX_SENTINEL,
                    workflow_dir / DATABASE_FILE,
                    workflow_dir / SECRET_FILE,
                    workflow_dir / CODER_ASSIGNMENTS_FILE,
                )
                if not all(_private_runtime_file(path) for path in required_files):
                    continue
                if not _private_discovery_directory(workflow_dir, hidden_root):
                    continue
                try:
                    controlled = _validate_root_controls(current)
                except WorkflowError:
                    continue
                candidates.append(controlled)
        except OSError:
            _fail("E_RUNTIME_DISCOVERY_SCAN")
    unique = sorted(set(candidates), key=lambda value: str(value))
    if len(unique) != 1:
        _fail("E_RUNTIME_ROOT_AMBIGUOUS" if unique else "E_RUNTIME_ROOT_NOT_FOUND")
    _DISCOVERED_RUNTIME_ROOT = unique[0]
    return unique[0]


def bootstrap_policy(root_arg: str | os.PathLike[str], *, attest: bool) -> Path:
    """Create the policy receipt only after the operator explicitly attests checks."""

    if not attest:
        _fail("E_ATTESTATION_REQUIRED")
    env_value = os.environ.get(ROOT_ENV)
    requested = Path(root_arg).expanduser()
    if not env_value or not requested.is_absolute() or not Path(env_value).expanduser().is_absolute():
        _fail("E_QUARANTINE_ENV_MISSING")
    try:
        root = requested.resolve(strict=True)
        authorized = Path(env_value).expanduser().resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_ROOT_MISSING")
    if root != authorized or _is_relative_to(root, REPO_ROOT) or root == REPO_ROOT:
        _fail("E_QUARANTINE_ROOT_MISMATCH")
    _reject_symlink_components(root)
    metadata = root.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        _fail("E_QUARANTINE_NOT_OWNER_ONLY")
    if not (root / INDEX_SENTINEL).is_file():
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
        "operator_attestation": "explicit",
    }
    policy_path = root / POLICY_FILE
    _atomic_write(policy_path, _canonical(payload) + b"\n", mode=0o600)
    return policy_path


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if path.exists() and path.is_symlink():
        _fail("E_SYMLINK_TARGET")
    temporary = path.parent / f".tmp-{secrets.token_hex(12)}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _workflow_paths(root_arg: str | os.PathLike[str]) -> tuple[Path, Path, Path]:
    root = validate_quarantine_root(root_arg)
    workflow = root / WORKFLOW_DIR
    if workflow.exists() and (workflow.is_symlink() or not workflow.is_dir()):
        _fail("E_WORKFLOW_PATH_INVALID")
    workflow.mkdir(mode=0o700, exist_ok=True)
    os.chmod(workflow, 0o700)
    database = workflow / DATABASE_FILE
    secret = workflow / SECRET_FILE
    return workflow, database, secret


def _load_or_create_secret(secret_path: Path) -> bytes:
    if secret_path.exists():
        if secret_path.is_symlink() or not secret_path.is_file():
            _fail("E_SECRET_INVALID")
        if stat.S_IMODE(secret_path.stat().st_mode) & 0o077:
            _fail("E_SECRET_PERMISSIONS")
        secret = secret_path.read_bytes()
        if len(secret) != 32:
            _fail("E_SECRET_INVALID")
        return secret
    secret = secrets.token_bytes(32)
    _atomic_write(secret_path, secret, mode=0o600)
    return secret


@contextlib.contextmanager
def _connect(database: Path) -> Iterator[sqlite3.Connection]:
    if database.exists() and database.is_symlink():
        _fail("E_DATABASE_SYMLINK")
    old_umask = os.umask(0o077)
    try:
        connection = sqlite3.connect(database, timeout=30)
    finally:
        os.umask(old_umask)
    try:
        connection.row_factory = sqlite3.Row
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
        with contextlib.suppress(FileNotFoundError):
            os.chmod(database, 0o600)
        for suffix in ("-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                os.chmod(Path(str(database) + suffix), 0o600)


@contextlib.contextmanager
def _db(root_arg: str | os.PathLike[str]) -> Iterator[tuple[sqlite3.Connection, bytes, Path]]:
    _, database, secret_path = _workflow_paths(root_arg)
    secret = _load_or_create_secret(secret_path)
    with _connect(database) as connection:
        yield connection, secret, database


def _coder_hash(secret: bytes, coder_token: str) -> str:
    token = coder_token.strip()
    if len(token) < 12 or len(token) > 200:
        _fail("E_CODER_TOKEN_LENGTH")
    return hmac.new(secret, b"coder\0" + token.encode("utf-8"), hashlib.sha256).hexdigest()


def _audit(connection: sqlite3.Connection, actor_hash: str, event: str, stage: str) -> None:
    connection.execute(
        "INSERT INTO audit_log(created_at_utc, actor_hash, event, stage) VALUES (?, ?, ?, ?)",
        (_utc_now(), actor_hash, event, stage),
    )


SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL,
    workflow_version TEXT NOT NULL,
    packet_digest TEXT NOT NULL,
    selection_digest TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    retention_deadline TEXT NOT NULL,
    referential_sample_frozen INTEGER NOT NULL DEFAULT 0 CHECK (referential_sample_frozen IN (0,1))
);
CREATE TABLE IF NOT EXISTS items (
    item_id INTEGER PRIMARY KEY,
    internal_key TEXT NOT NULL UNIQUE,
    display_key TEXT NOT NULL UNIQUE,
    media_relpath TEXT NOT NULL,
    stratum_key TEXT NOT NULL,
    duration_ms INTEGER NOT NULL CHECK (duration_ms > 0),
    batch_number INTEGER NOT NULL CHECK (batch_number > 0)
);
CREATE TABLE IF NOT EXISTS coder_slots (
    slot TEXT PRIMARY KEY,
    coder_hash TEXT NOT NULL,
    assigned_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS language_labels (
    item_id INTEGER NOT NULL REFERENCES items(item_id),
    slot TEXT NOT NULL CHECK (slot IN ('LANGUAGE_A','LANGUAGE_B')),
    language_code TEXT NOT NULL,
    competence TEXT NOT NULL CHECK (competence IN ('NATIVE','FLUENT','PROFICIENT','INSUFFICIENT')),
    audio_integrity TEXT NOT NULL CHECK (audio_integrity IN ('USABLE','PARTIAL','UNUSABLE','UNDECIDABLE')),
    speech_present INTEGER NOT NULL CHECK (speech_present IN (0,1)),
    overlap_present INTEGER NOT NULL CHECK (overlap_present IN (0,1)),
    locked INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0,1)),
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY(item_id, slot)
);
CREATE TABLE IF NOT EXISTS adjudicated_languages (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
    language_code TEXT NOT NULL,
    audio_integrity TEXT NOT NULL CHECK (audio_integrity IN ('USABLE','PARTIAL','UNUSABLE','UNDECIDABLE')),
    speech_present INTEGER NOT NULL CHECK (speech_present IN (0,1)),
    overlap_present INTEGER NOT NULL CHECK (overlap_present IN (0,1)),
    reason_code TEXT NOT NULL,
    adjudicator_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS timing_segments (
    segment_id INTEGER PRIMARY KEY,
    internal_key TEXT NOT NULL UNIQUE,
    display_key TEXT NOT NULL,
    item_id INTEGER NOT NULL REFERENCES items(item_id),
    slot TEXT NOT NULL CHECK (slot IN ('TIMING_A','TIMING_B')),
    onset_ms INTEGER NOT NULL CHECK (onset_ms >= 0),
    offset_ms INTEGER NOT NULL CHECK (offset_ms > onset_ms),
    source_text TEXT NOT NULL,
    language_code TEXT NOT NULL,
    speaker_role TEXT NOT NULL CHECK (speaker_role IN ('NON_CHILD','CHILD','OVERLAP','UNCERTAIN','NONSPEECH')),
    locked INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0,1)),
    updated_at_utc TEXT NOT NULL,
    UNIQUE(item_id, slot, display_key)
);
CREATE TABLE IF NOT EXISTS timing_item_locks (
    item_id INTEGER NOT NULL REFERENCES items(item_id),
    slot TEXT NOT NULL CHECK (slot IN ('TIMING_A','TIMING_B')),
    locked_at_utc TEXT NOT NULL,
    PRIMARY KEY(item_id, slot)
);
CREATE TABLE IF NOT EXISTS timing_adjudication_item_locks (
    item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
    accepted_utterance_count INTEGER NOT NULL CHECK (accepted_utterance_count >= 0),
    completion_reason TEXT NOT NULL CHECK (completion_reason IN ('ALL_SOURCES_DISPOSITIONED','FROZEN_THRESHOLD_REACHED')),
    completed_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS adjudicated_utterances (
    utterance_id INTEGER PRIMARY KEY,
    internal_key TEXT NOT NULL UNIQUE,
    display_key TEXT NOT NULL UNIQUE,
    item_id INTEGER NOT NULL REFERENCES items(item_id),
    source_segment_a INTEGER REFERENCES timing_segments(segment_id),
    source_segment_b INTEGER REFERENCES timing_segments(segment_id),
    onset_ms INTEGER NOT NULL CHECK (onset_ms >= 0),
    offset_ms INTEGER NOT NULL CHECK (offset_ms > onset_ms),
    source_text TEXT NOT NULL,
    language_code TEXT NOT NULL,
    speaker_role TEXT NOT NULL CHECK (speaker_role IN ('NON_CHILD','CHILD','OVERLAP','UNCERTAIN','NONSPEECH')),
    reason_code TEXT NOT NULL,
    locked INTEGER NOT NULL DEFAULT 1 CHECK (locked = 1),
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS referential_assignments (
    utterance_id INTEGER PRIMARY KEY REFERENCES adjudicated_utterances(utterance_id),
    double_code INTEGER NOT NULL CHECK (double_code IN (0,1)),
    assignment_hash TEXT NOT NULL,
    proposal_condition TEXT NOT NULL DEFAULT 'HUMAN_ONLY_REFERENCE' CHECK (proposal_condition = 'HUMAN_ONLY_REFERENCE'),
    frozen_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS referential_labels (
    utterance_id INTEGER NOT NULL REFERENCES adjudicated_utterances(utterance_id),
    slot TEXT NOT NULL CHECK (slot IN ('REFERENTIAL_A','REFERENTIAL_B')),
    status TEXT NOT NULL CHECK (status IN ('VISIBLE_SINGLE','VISIBLE_MULTIPLE','NULL_NOT_VISIBLE','IRRELEVANT','UNDECIDABLE','UNUSABLE')),
    mention_family TEXT NOT NULL CHECK (mention_family IN ('NOUN_OBJECT','VERB_ACTION','BOTH','NEITHER')),
    candidate_band TEXT NOT NULL CHECK (candidate_band IN ('ZERO','ONE','TWO','THREE_PLUS','UNKNOWN')),
    boundary_onset_ms INTEGER,
    boundary_offset_ms INTEGER,
    boundary_censored INTEGER NOT NULL DEFAULT 0 CHECK (boundary_censored IN (0,1)),
    locked INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0,1)),
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY(utterance_id, slot),
    CHECK (boundary_onset_ms IS NULL OR boundary_onset_ms >= 0),
    CHECK (boundary_offset_ms IS NULL OR boundary_offset_ms >= boundary_onset_ms)
);
CREATE TABLE IF NOT EXISTS adjudications (
    target_kind TEXT NOT NULL CHECK (target_kind IN ('LANGUAGE','TIMING_ROLE_TEXT','REFERENTIAL')),
    target_internal_id INTEGER NOT NULL,
    final_value_json TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    adjudicator_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY(target_kind, target_internal_id)
);
CREATE TABLE IF NOT EXISTS adjudicated_referential (
    utterance_id INTEGER PRIMARY KEY REFERENCES adjudicated_utterances(utterance_id),
    status TEXT NOT NULL CHECK (status IN ('VISIBLE_SINGLE','VISIBLE_MULTIPLE','NULL_NOT_VISIBLE','IRRELEVANT','UNDECIDABLE','UNUSABLE')),
    mention_family TEXT NOT NULL CHECK (mention_family IN ('NOUN_OBJECT','VERB_ACTION','BOTH','NEITHER')),
    candidate_band TEXT NOT NULL CHECK (candidate_band IN ('ZERO','ONE','TWO','THREE_PLUS','UNKNOWN')),
    boundary_onset_ms INTEGER,
    boundary_offset_ms INTEGER,
    boundary_censored INTEGER NOT NULL DEFAULT 0 CHECK (boundary_censored IN (0,1)),
    reason_code TEXT NOT NULL,
    adjudicator_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    CHECK (boundary_onset_ms IS NULL OR boundary_onset_ms >= 0),
    CHECK (boundary_offset_ms IS NULL OR boundary_offset_ms >= boundary_onset_ms)
);
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    actor_hash TEXT NOT NULL,
    event TEXT NOT NULL,
    stage TEXT NOT NULL
);
"""


def _validate_language(value: str) -> str:
    if value in LANGUAGE_SPECIAL:
        return value
    normalized = value.strip()
    if not LANGUAGE_RE.fullmatch(normalized):
        _fail("E_LANGUAGE_CODE_INVALID")
    return normalized


def _validate_referential_combination(
    status: str,
    mention_family: str,
    candidate_band: str,
    boundary_onset_ms: int | None,
    boundary_offset_ms: int | None,
    boundary_censored: bool,
) -> None:
    if boundary_censored and boundary_onset_ms is None:
        _fail("E_CENSORING_WITHOUT_BOUNDARY")
    if status == "VISIBLE_SINGLE":
        valid = mention_family != "NEITHER" and candidate_band == "ONE"
    elif status == "VISIBLE_MULTIPLE":
        valid = mention_family != "NEITHER" and candidate_band in {"TWO", "THREE_PLUS"}
    elif status == "NULL_NOT_VISIBLE":
        valid = mention_family != "NEITHER" and candidate_band == "ZERO"
    elif status == "IRRELEVANT":
        valid = mention_family == "NEITHER" and candidate_band == "ZERO"
    else:
        valid = mention_family == "NEITHER" and candidate_band == "UNKNOWN"
    if not valid:
        _fail("E_REFERENTIAL_COMBINATION_INVALID")


def _validate_manifest(manifest: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != VERSION:
        _fail("E_PACKET_VERSION")
    if manifest.get("opaque_key_attestation") is not True:
        _fail("E_OPAQUE_KEY_ATTESTATION")
    selection_digest = manifest.get("frozen_selection_digest")
    if not isinstance(selection_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", selection_digest):
        _fail("E_SELECTION_DIGEST")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != FROZEN_SELECTED_COUNT:
        _fail("E_PACKET_ITEM_COUNT")
    seen_internal: set[str] = set()
    seen_display: set[str] = set()
    seen_sources: set[str] = set()
    media_bindings: dict[str, str] = {}
    validated: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict) or FORBIDDEN_MANIFEST_KEYS.intersection(row):
            _fail("E_PACKET_RESTRICTED_FIELD")
        allowed = {
            "internal_key",
            "display_key",
            "media_relpath",
            "stratum_key",
            "duration_ms",
            "batch_number",
            "source_object_key",
            "media_sha256",
            "annotation_linkage_sha256",
        }
        if set(row) != allowed:
            _fail("E_PACKET_FIELDS")
        internal = row.get("internal_key")
        display = row.get("display_key")
        relative = row.get("media_relpath")
        stratum = row.get("stratum_key")
        duration = row.get("duration_ms")
        batch = row.get("batch_number")
        source_object_key = row.get("source_object_key")
        media_sha256 = row.get("media_sha256")
        annotation_linkage = row.get("annotation_linkage_sha256")
        if not isinstance(internal, str) or not OPAQUE_KEY_RE.fullmatch(internal):
            _fail("E_INTERNAL_KEY_NOT_OPAQUE")
        if not isinstance(display, str) or not DISPLAY_KEY_RE.fullmatch(display):
            _fail("E_DISPLAY_KEY_INVALID")
        if not isinstance(stratum, str) or not OPAQUE_KEY_RE.fullmatch(stratum):
            _fail("E_STRATUM_KEY_NOT_OPAQUE")
        if not isinstance(relative, str) or Path(relative).is_absolute() or len(Path(relative).parts) != 2:
            _fail("E_MEDIA_PATH_INVALID")
        media_parts = Path(relative).parts
        valid_media_name = (
            (media_parts[0] == "media" and OPAQUE_MEDIA_RE.fullmatch(media_parts[1]))
            or (media_parts[0] == "raw_v1_2" and CONTENT_ADDRESS_RE.fullmatch(media_parts[1]))
        )
        if not valid_media_name:
            _fail("E_MEDIA_NAME_NOT_OPAQUE")
        if any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (source_object_key, media_sha256, annotation_linkage)
        ):
            _fail("E_MEDIA_BINDING_FIELD")
        if media_parts[0] == "raw_v1_2" and media_parts[1] != f"{media_sha256}.bin":
            _fail("E_MEDIA_CONTENT_ADDRESS_MISMATCH")
        media_candidate = root / relative
        try:
            media_metadata = media_candidate.lstat()
            resolved_media = media_candidate.resolve(strict=True)
        except OSError:
            _fail("E_MEDIA_MISSING")
        if (
            not _is_relative_to(resolved_media, root)
            or stat.S_ISLNK(media_metadata.st_mode)
            or not stat.S_ISREG(media_metadata.st_mode)
            or media_metadata.st_size <= 0
            or media_metadata.st_uid != os.getuid()
            or stat.S_IMODE(media_metadata.st_mode) & 0o077
        ):
            _fail("E_MEDIA_OUTSIDE_QUARANTINE")
        if not isinstance(duration, int) or duration <= 0:
            _fail("E_DURATION_INVALID")
        if not isinstance(batch, int) or not 1 <= batch <= 99:
            _fail("E_BATCH_INVALID")
        prior_media_sha = media_bindings.get(relative)
        if prior_media_sha is not None and prior_media_sha != media_sha256:
            _fail("E_MEDIA_DEDUP_BINDING_MISMATCH")
        # A content-addressed object may authenticate multiple distinct source
        # objects.  Hash it once; every later row must carry the same SHA/path.
        actual_media_sha256 = prior_media_sha or _sha256_file(media_candidate)
        if actual_media_sha256 != media_sha256:
            _fail("E_MEDIA_DIGEST_MISMATCH")
        if internal in seen_internal or display in seen_display:
            _fail("E_PACKET_DUPLICATE")
        if source_object_key in seen_sources:
            _fail("E_SOURCE_OBJECT_DUPLICATE")
        seen_internal.add(internal)
        seen_display.add(display)
        seen_sources.add(source_object_key)
        media_bindings[relative] = media_sha256
        validated.append(dict(row))
    return validated


def _validate_frozen_skeleton(path_arg: str | os.PathLike[str], root: Path) -> tuple[set[str], str]:
    path = Path(path_arg).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _fail("E_FROZEN_SKELETON_MISSING")
    if not _is_relative_to(resolved, root) or path.is_symlink():
        _fail("E_FROZEN_SKELETON_OUTSIDE_QUARANTINE")
    metadata = resolved.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        _fail("E_FROZEN_SKELETON_PERMISSIONS")
    try:
        skeleton = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _fail("E_FROZEN_SKELETON_INVALID")
    digest = _sha256(_canonical(skeleton))
    if digest != FROZEN_V1_1_HUMAN_PACKET_SHA256:
        _fail("E_FROZEN_SKELETON_DIGEST_MISMATCH")
    if (
        skeleton.get("schema_version") != "childlens-human-validation-packet-v1.1.0"
        or skeleton.get("selected_count") != FROZEN_SELECTED_COUNT
        or not isinstance(skeleton.get("pilot_selection_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", skeleton["pilot_selection_sha256"])
        or not isinstance(skeleton.get("items"), list)
        or len(skeleton["items"]) != FROZEN_SELECTED_COUNT
    ):
        _fail("E_FROZEN_SKELETON_INVALID")
    keys: set[str] = set()
    for row in skeleton["items"]:
        if not isinstance(row, dict) or set(row) != {
            "blinded_item_key",
            "acquisition_status",
            "language_judgment",
            "audio_integrity",
            "utterance_timing_text_role_review",
            "referential_status_review",
            "adjudication_status",
        }:
            _fail("E_FROZEN_SKELETON_INVALID")
        key = row.get("blinded_item_key")
        if not isinstance(key, str) or not OPAQUE_KEY_RE.fullmatch(key) or key in keys:
            _fail("E_FROZEN_SKELETON_INVALID")
        keys.add(key)
    return keys, skeleton["pilot_selection_sha256"]


def _restricted_json(path_arg: str | os.PathLike[str], root: Path, missing_code: str) -> Mapping[str, Any]:
    root = root.resolve(strict=True)
    path = Path(path_arg).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        _fail(missing_code)
    if not _is_relative_to(resolved, root) or path.is_symlink():
        _fail("E_RESTRICTED_BINDING_OUTSIDE_QUARANTINE")
    metadata = resolved.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        _fail("E_RESTRICTED_BINDING_PERMISSIONS")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _fail("E_RESTRICTED_BINDING_INVALID")
    if not isinstance(value, dict):
        _fail("E_RESTRICTED_BINDING_INVALID")
    return value


def _validate_restricted_input(
    path_arg: str | os.PathLike[str], root: Path, frozen_keys: set[str]
) -> dict[str, dict[str, Any]]:
    document = _restricted_json(path_arg, root, "E_FROZEN_RESTRICTED_INPUT_MISSING")
    if _sha256(_canonical(document)) != FROZEN_V1_1_RESTRICTED_INPUT_SHA256:
        _fail("E_FROZEN_RESTRICTED_INPUT_DIGEST_MISMATCH")
    media = document.get("media")
    objects = document.get("objects")
    annotations = document.get("annotations")
    if not isinstance(media, list) or not isinstance(objects, list) or not isinstance(annotations, list):
        _fail("E_FROZEN_RESTRICTED_INPUT_INVALID")
    media_by_key = {
        row.get("media_key"): row
        for row in media
        if isinstance(row, dict) and isinstance(row.get("media_key"), str)
    }
    if not frozen_keys.issubset(media_by_key):
        _fail("E_FROZEN_ITEM_SET_MISMATCH")
    duration_rows = [
        row
        for row in media_by_key.values()
        if isinstance(row.get("duration_milliseconds"), int) and row["duration_milliseconds"] > 0
    ]
    if len(duration_rows) != len(media_by_key):
        _fail("E_FROZEN_RESTRICTED_INPUT_INVALID")
    ordered = sorted(duration_rows, key=lambda row: (row["duration_milliseconds"], row["media_key"]))
    total = len(ordered)
    tertile_labels = ("LOW", "MIDDLE", "HIGH")
    tertiles = {
        row["media_key"]: tertile_labels[min(2, (index * 3) // total)]
        for index, row in enumerate(ordered)
    }
    object_by_key = {
        row.get("object_key"): row
        for row in objects
        if isinstance(row, dict) and isinstance(row.get("object_key"), str)
    }
    annotation_shas: dict[str, list[str]] = {}
    for link in annotations:
        if not isinstance(link, dict):
            continue
        media_key = link.get("linked_media_key")
        annotation_object = object_by_key.get(link.get("object_key"))
        if media_key in frozen_keys and isinstance(annotation_object, dict):
            local_sha = annotation_object.get("local_sha256")
            if isinstance(local_sha, str) and re.fullmatch(r"[0-9a-f]{64}", local_sha):
                annotation_shas.setdefault(media_key, []).append(local_sha)
    result: dict[str, dict[str, Any]] = {}
    for key in frozen_keys:
        row = media_by_key[key]
        object_key = row.get("object_key")
        if not isinstance(object_key, str) or not re.fullmatch(r"[0-9a-f]{64}", object_key):
            _fail("E_FROZEN_MEDIA_OBJECT_BINDING")
        linked_shas = sorted(annotation_shas.get(key, []))
        if not linked_shas:
            _fail("E_FROZEN_ANNOTATION_BINDING")
        stratum = [
            row.get("coarse_activity_label"),
            row.get("speech_presence_bin"),
            tertiles[key],
            row.get("location_label") if row.get("location_label") is not None else "__UNAVAILABLE__",
        ]
        result[key] = {
            "source_object_key": object_key,
            "duration_ms": row["duration_milliseconds"],
            "stratum_key": _sha256(_canonical(stratum)),
            "annotation_linkage_sha256": _sha256(
                key.encode() + b"\0" + b"\0".join(value.encode() for value in linked_shas)
            ),
        }
    return result


def _validate_native_receipt(
    path_arg: str | os.PathLike[str], root: Path, frozen_selection_digest: str
) -> dict[str, dict[str, Any]]:
    receipt = _restricted_json(path_arg, root, "E_NATIVE_RECEIPT_MISSING")
    claimed = receipt.get("restricted_receipt_sha256")
    resealed = dict(receipt)
    resealed["restricted_receipt_sha256"] = None
    if (
        receipt.get("schema_version") != "childlens-native-transfer-restricted-receipt-v1.2.0"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("pilot_selection_sha256") != frozen_selection_digest
        or not isinstance(claimed, str)
        or not hmac.compare_digest(claimed, _sha256(_canonical(resealed)))
        or not isinstance(receipt.get("items"), list)
        or len(receipt["items"]) != FROZEN_SELECTED_COUNT
    ):
        _fail("E_NATIVE_RECEIPT_INVALID")
    result: dict[str, dict[str, Any]] = {}
    for row in receipt["items"]:
        if not isinstance(row, dict) or row.get("status") != "COMPLETE":
            _fail("E_NATIVE_RECEIPT_INVALID")
        object_key = row.get("object_key")
        local_sha = row.get("local_sha256")
        relative = row.get("stored_relative_path")
        transferred = row.get("transferred_bytes")
        if (
            not isinstance(object_key, str)
            or not re.fullmatch(r"[0-9a-f]{64}", object_key)
            or not isinstance(local_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", local_sha)
            or relative != f"raw_v1_2/{local_sha}.bin"
            or not isinstance(transferred, int)
            or transferred <= 0
            or object_key in result
        ):
            _fail("E_NATIVE_RECEIPT_INVALID")
        result[object_key] = {
            "media_sha256": local_sha,
            "media_relpath": relative,
            "transferred_bytes": transferred,
        }
    return result


def _preassign_coders(connection: sqlite3.Connection, secret: bytes, workflow_dir: Path) -> str:
    assignment_path = workflow_dir / CODER_ASSIGNMENTS_FILE
    if assignment_path.exists():
        _fail("E_CODER_ASSIGNMENTS_ALREADY_EXIST")
    assignments: dict[str, str] = {}
    created = _utc_now()
    for slot in sorted(SLOTS):
        token = secrets.token_urlsafe(32)
        assignments[slot] = token
        connection.execute(
            "INSERT INTO coder_slots(slot, coder_hash, assigned_at_utc) VALUES (?, ?, ?)",
            (slot, _coder_hash(secret, token), created),
        )
    payload = {
        "schema_version": VERSION,
        "created_at_utc": created,
        "coordinator_distribution": "OFFLINE_AUTHORIZED_ONLY",
        "tokens": assignments,
    }
    encoded = _canonical(payload) + b"\n"
    _atomic_write(assignment_path, encoded, mode=0o600)
    return _sha256(_canonical(payload))


def initialize(
    root_arg: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    frozen_skeleton_path: str | os.PathLike[str],
    frozen_restricted_input_path: str | os.PathLike[str],
    native_transfer_receipt_path: str | os.PathLike[str],
) -> dict[str, Any]:
    root = validate_quarantine_root(root_arg)
    frozen_keys, frozen_selection_digest = _validate_frozen_skeleton(frozen_skeleton_path, root)
    frozen_bindings = _validate_restricted_input(frozen_restricted_input_path, root, frozen_keys)
    native_bindings = _validate_native_receipt(native_transfer_receipt_path, root, frozen_selection_digest)
    manifest_file = Path(manifest_path).expanduser()
    try:
        manifest_resolved = manifest_file.resolve(strict=True)
    except OSError:
        _fail("E_PACKET_MISSING")
    if not _is_relative_to(manifest_resolved, root) or manifest_file.is_symlink():
        _fail("E_PACKET_OUTSIDE_QUARANTINE")
    manifest_metadata = manifest_resolved.stat()
    if manifest_metadata.st_uid != os.getuid() or stat.S_IMODE(manifest_metadata.st_mode) & 0o077:
        _fail("E_PACKET_PERMISSIONS")
    try:
        manifest: Mapping[str, Any] = json.loads(manifest_resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _fail("E_PACKET_INVALID")
    items = _validate_manifest(manifest, root)
    if manifest["frozen_selection_digest"] != frozen_selection_digest:
        _fail("E_SELECTION_DIGEST_MISMATCH")
    if {row["internal_key"] for row in items} != frozen_keys:
        _fail("E_FROZEN_ITEM_SET_MISMATCH")
    for row in items:
        frozen = frozen_bindings[row["internal_key"]]
        native = native_bindings.get(frozen["source_object_key"])
        if native is None or any(
            (
                row["source_object_key"] != frozen["source_object_key"],
                row["duration_ms"] != frozen["duration_ms"],
                row["stratum_key"] != frozen["stratum_key"],
                row["annotation_linkage_sha256"] != frozen["annotation_linkage_sha256"],
                row["media_sha256"] != native["media_sha256"],
                row["media_relpath"] != native["media_relpath"],
            )
        ):
            _fail("E_KEY_MEDIA_LINKAGE_MISMATCH")
    packet_digest = _sha256(_canonical(manifest))
    workflow_dir, _database_path, _secret_path = _workflow_paths(root)
    with _db(root) as (connection, secret, database):
        connection.executescript(SCHEMA)
        existing = connection.execute("SELECT packet_digest FROM workflow_meta WHERE singleton = 1").fetchone()
        if existing:
            if existing["packet_digest"] != packet_digest:
                _fail("E_PACKET_IMMUTABLE")
            return {"status": "already_initialized", "item_count": len(items), "packet_digest": packet_digest}
        connection.execute(
            """INSERT INTO workflow_meta
               (singleton, schema_version, workflow_version, packet_digest, selection_digest,
                created_at_utc, retention_deadline, referential_sample_frozen)
               VALUES (1, ?, ?, ?, ?, ?, ?, 0)""",
            (SCHEMA_VERSION, VERSION, packet_digest, manifest["frozen_selection_digest"], _utc_now(), RETENTION_DEADLINE),
        )
        for row in items:
            connection.execute(
                """INSERT INTO items
                   (internal_key, display_key, media_relpath, stratum_key, duration_ms, batch_number)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    row["internal_key"],
                    row["display_key"],
                    row["media_relpath"],
                    row["stratum_key"],
                    row["duration_ms"],
                    row["batch_number"],
                ),
            )
        assignment_digest = _preassign_coders(connection, secret, workflow_dir)
    os.chmod(database, 0o600)
    return {
        "status": "initialized",
        "item_count": len(items),
        "packet_digest": packet_digest,
        "frozen_skeleton_sha256": FROZEN_V1_1_HUMAN_PACKET_SHA256,
        "coder_assignment_receipt_sha256": assignment_digest,
    }


def assign_coder(root_arg: str | os.PathLike[str], slot: str, coder_token: str) -> str:
    if slot not in SLOTS:
        _fail("E_SLOT_INVALID")
    with _db(root_arg) as (connection, secret, _database):
        coder_hash = _coder_hash(secret, coder_token)
        existing = connection.execute("SELECT coder_hash FROM coder_slots WHERE slot = ?", (slot,)).fetchone()
        if not existing or not hmac.compare_digest(existing["coder_hash"], coder_hash):
            _fail("E_CODER_NOT_AUTHORIZED")
        return coder_hash


def _require_actor(connection: sqlite3.Connection, secret: bytes, slot: str, coder_token: str) -> str:
    if slot not in SLOTS:
        _fail("E_SLOT_INVALID")
    actor = _coder_hash(secret, coder_token)
    row = connection.execute("SELECT coder_hash FROM coder_slots WHERE slot = ?", (slot,)).fetchone()
    if not row or not hmac.compare_digest(row["coder_hash"], actor):
        _fail("E_CODER_NOT_AUTHORIZED")
    return actor


def _item(connection: sqlite3.Connection, display_key: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM items WHERE display_key = ?", (display_key,)).fetchone()
    if not row:
        _fail("E_ITEM_NOT_FOUND")
    return row


def save_language_label(
    root_arg: str | os.PathLike[str],
    *,
    slot: str,
    coder_token: str,
    display_key: str,
    language_code: str,
    competence: str,
    audio_integrity: str,
    speech_present: bool,
    overlap_present: bool,
    lock: bool = False,
) -> None:
    if slot not in LANGUAGE_SLOTS:
        _fail("E_LANGUAGE_SLOT")
    language_code = _validate_language(language_code)
    if competence not in {"NATIVE", "FLUENT", "PROFICIENT", "INSUFFICIENT"}:
        _fail("E_COMPETENCE_INVALID")
    if audio_integrity not in AUDIO_INTEGRITY_VALUES:
        _fail("E_AUDIO_INTEGRITY_INVALID")
    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, slot, coder_token)
        item = _item(connection, display_key)
        prior = connection.execute(
            "SELECT locked FROM language_labels WHERE item_id = ? AND slot = ?", (item["item_id"], slot)
        ).fetchone()
        if prior and prior["locked"]:
            _fail("E_RECORD_LOCKED")
        connection.execute(
            """INSERT INTO language_labels
               (item_id, slot, language_code, competence, audio_integrity, speech_present,
                overlap_present, locked, updated_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(item_id, slot) DO UPDATE SET
                 language_code=excluded.language_code, competence=excluded.competence,
                 audio_integrity=excluded.audio_integrity, speech_present=excluded.speech_present,
                 overlap_present=excluded.overlap_present, locked=excluded.locked,
                 updated_at_utc=excluded.updated_at_utc""",
            (
                item["item_id"], slot, language_code, competence, audio_integrity,
                int(speech_present), int(overlap_present), int(lock), _utc_now(),
            ),
        )
        _audit(connection, actor, "LANGUAGE_LOCKED" if lock else "LANGUAGE_SAVED", slot)


def language_adjudication_tasks(
    root_arg: str | os.PathLike[str], *, coder_token: str, batch_size: int = 20
) -> list[dict[str, Any]]:
    """Return only locked A/B language records to the assigned adjudicator."""

    if not 1 <= batch_size <= 50:
        _fail("E_BATCH_REQUEST_INVALID")
    with _db(root_arg) as (connection, secret, _database):
        _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        rows = connection.execute(
            """SELECT i.display_key, a.language_code language_a, a.audio_integrity audio_a,
                      a.speech_present speech_a, a.overlap_present overlap_a,
                      b.language_code language_b, b.audio_integrity audio_b,
                      b.speech_present speech_b, b.overlap_present overlap_b
               FROM items i
               JOIN language_labels a ON a.item_id=i.item_id AND a.slot='LANGUAGE_A' AND a.locked=1
               JOIN language_labels b ON b.item_id=i.item_id AND b.slot='LANGUAGE_B' AND b.locked=1
               LEFT JOIN adjudicated_languages j ON j.item_id=i.item_id
               WHERE j.item_id IS NULL
               ORDER BY i.batch_number, i.display_key LIMIT ?""",
            (batch_size,),
        ).fetchall()
        return [dict(row) for row in rows]


def adjudicate_language(
    root_arg: str | os.PathLike[str],
    *,
    coder_token: str,
    display_item_key: str,
    language_code: str,
    audio_integrity: str,
    speech_present: bool,
    overlap_present: bool,
    reason_code: str,
) -> None:
    language_code = _validate_language(language_code)
    if audio_integrity not in AUDIO_INTEGRITY_VALUES:
        _fail("E_AUDIO_INTEGRITY_INVALID")
    if not reason_code.strip() or len(reason_code) > 200:
        _fail("E_REASON_REQUIRED")
    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        item = _item(connection, display_item_key)
        locked = connection.execute(
            "SELECT COUNT(*) n FROM language_labels WHERE item_id=? AND locked=1", (item["item_id"],)
        ).fetchone()["n"]
        if locked != 2:
            _fail("E_LANGUAGE_SOURCES_NOT_LOCKED")
        if connection.execute("SELECT 1 FROM adjudicated_languages WHERE item_id=?", (item["item_id"],)).fetchone():
            _fail("E_ADJUDICATION_IMMUTABLE")
        connection.execute(
            "INSERT INTO adjudicated_languages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item["item_id"], language_code, audio_integrity, int(speech_present),
                int(overlap_present), reason_code.strip(), actor, _utc_now(),
            ),
        )
        final_value = {
            "language_code": language_code,
            "audio_integrity": audio_integrity,
            "speech_present": bool(speech_present),
            "overlap_present": bool(overlap_present),
        }
        connection.execute(
            "INSERT INTO adjudications VALUES ('LANGUAGE', ?, ?, ?, ?, ?)",
            (item["item_id"], json.dumps(final_value, sort_keys=True), reason_code.strip(), actor, _utc_now()),
        )
        _audit(connection, actor, "LANGUAGE_ADJUDICATED", ADJUDICATOR_SLOT)


def add_timing_segment(
    root_arg: str | os.PathLike[str],
    *,
    slot: str,
    coder_token: str,
    display_item_key: str,
    display_segment_key: str,
    onset_ms: int,
    offset_ms: int,
    source_text: str,
    language_code: str,
    speaker_role: str,
    lock: bool = False,
) -> None:
    if slot not in TIMING_SLOTS:
        _fail("E_TIMING_SLOT")
    if speaker_role not in ROLE_VALUES:
        _fail("E_ROLE_INVALID")
    language_code = _validate_language(language_code)
    if not re.fullmatch(r"U-[0-9]{4,6}", display_segment_key):
        _fail("E_SEGMENT_DISPLAY_KEY")
    if not isinstance(onset_ms, int) or not isinstance(offset_ms, int) or onset_ms < 0 or offset_ms <= onset_ms:
        _fail("E_TIMING_INVALID")
    text_value = source_text.strip()
    if speaker_role == "NONSPEECH":
        if text_value not in {"", "[NONSPEECH]"}:
            _fail("E_NONSPEECH_TEXT")
        text_value = "[NONSPEECH]"
    elif not text_value:
        _fail("E_SOURCE_TEXT_REQUIRED")
    if len(text_value) > 5000:
        _fail("E_SOURCE_TEXT_TOO_LONG")
    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, slot, coder_token)
        language_routes = connection.execute("SELECT COUNT(*) n FROM adjudicated_languages").fetchone()["n"]
        item_total = connection.execute("SELECT COUNT(*) n FROM items").fetchone()["n"]
        if language_routes != item_total:
            _fail("E_LANGUAGE_PHASE_INCOMPLETE")
        item = _item(connection, display_item_key)
        if offset_ms > item["duration_ms"]:
            _fail("E_TIMING_OUTSIDE_MEDIA")
        item_lock = connection.execute(
            "SELECT 1 FROM timing_item_locks WHERE item_id = ? AND slot = ?", (item["item_id"], slot)
        ).fetchone()
        if item_lock:
            _fail("E_ITEM_LOCKED")
        prior = connection.execute(
            "SELECT segment_id, locked FROM timing_segments WHERE item_id = ? AND slot = ? AND display_key = ?",
            (item["item_id"], slot, display_segment_key),
        ).fetchone()
        if prior and prior["locked"]:
            _fail("E_RECORD_LOCKED")
        internal_key = (
            connection.execute(
                "SELECT internal_key FROM timing_segments WHERE item_id = ? AND slot = ? AND display_key = ?",
                (item["item_id"], slot, display_segment_key),
            ).fetchone()
            or {"internal_key": secrets.token_urlsafe(18)}
        )["internal_key"]
        connection.execute(
            """INSERT INTO timing_segments
               (internal_key, display_key, item_id, slot, onset_ms, offset_ms, source_text,
                language_code, speaker_role, locked, updated_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(item_id, slot, display_key) DO UPDATE SET
                 onset_ms=excluded.onset_ms, offset_ms=excluded.offset_ms,
                 source_text=excluded.source_text, language_code=excluded.language_code,
                 speaker_role=excluded.speaker_role, locked=excluded.locked,
                 updated_at_utc=excluded.updated_at_utc""",
            (
                internal_key, display_segment_key, item["item_id"], slot, onset_ms, offset_ms,
                text_value, language_code, speaker_role, int(lock), _utc_now(),
            ),
        )
        _audit(connection, actor, "TIMING_SEGMENT_LOCKED" if lock else "TIMING_SEGMENT_SAVED", slot)


def lock_timing_item(
    root_arg: str | os.PathLike[str], *, slot: str, coder_token: str, display_item_key: str
) -> None:
    if slot not in TIMING_SLOTS:
        _fail("E_TIMING_SLOT")
    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, slot, coder_token)
        item = _item(connection, display_item_key)
        if not connection.execute(
            "SELECT 1 FROM timing_segments WHERE item_id = ? AND slot = ?", (item["item_id"], slot)
        ).fetchone():
            _fail("E_NO_SEGMENTS")
        connection.execute(
            "UPDATE timing_segments SET locked = 1 WHERE item_id = ? AND slot = ?", (item["item_id"], slot)
        )
        connection.execute(
            "INSERT OR IGNORE INTO timing_item_locks(item_id, slot, locked_at_utc) VALUES (?, ?, ?)",
            (item["item_id"], slot, _utc_now()),
        )
        _audit(connection, actor, "TIMING_ITEM_LOCKED", slot)


def own_timing_segments(
    root_arg: str | os.PathLike[str], *, slot: str, coder_token: str, display_item_key: str
) -> list[dict[str, Any]]:
    """Return one coder's own drafts only; no peer records are exposed."""

    if slot not in TIMING_SLOTS:
        _fail("E_TIMING_SLOT")
    with _db(root_arg) as (connection, secret, _database):
        _require_actor(connection, secret, slot, coder_token)
        item = _item(connection, display_item_key)
        rows = connection.execute(
            """SELECT display_key, onset_ms, offset_ms, source_text, language_code,
                      speaker_role, locked
               FROM timing_segments WHERE item_id=? AND slot=? ORDER BY onset_ms, display_key""",
            (item["item_id"], slot),
        ).fetchall()
        return [dict(row) for row in rows]


def timing_adjudication_tasks(
    root_arg: str | os.PathLike[str], *, coder_token: str, batch_size: int = 10
) -> list[dict[str, Any]]:
    """Return preserved locked source records only after both passes are closed."""

    if not 1 <= batch_size <= 20:
        _fail("E_BATCH_REQUEST_INVALID")
    with _db(root_arg) as (connection, secret, _database):
        _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        items = connection.execute(
            """SELECT i.item_id, i.display_key FROM items i
               JOIN timing_item_locks a ON a.item_id=i.item_id AND a.slot='TIMING_A'
               JOIN timing_item_locks b ON b.item_id=i.item_id AND b.slot='TIMING_B'
               LEFT JOIN timing_adjudication_item_locks done ON done.item_id=i.item_id
               WHERE done.item_id IS NULL
               ORDER BY i.batch_number, i.display_key LIMIT ?""",
            (batch_size,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for item in items:
            source_rows = connection.execute(
                """SELECT segment_id, display_key, slot, onset_ms, offset_ms, source_text,
                          language_code, speaker_role
                   FROM timing_segments WHERE item_id=? AND locked=1 ORDER BY slot, onset_ms""",
                (item["item_id"],),
            ).fetchall()
            accepted = connection.execute(
                "SELECT COUNT(*) n FROM adjudicated_utterances WHERE item_id=?", (item["item_id"],)
            ).fetchone()["n"]
            result.append(
                {
                    "display_key": item["display_key"],
                    "locked_sources": [dict(row) for row in source_rows],
                    "accepted_count": accepted,
                }
            )
        return result


def complete_timing_adjudication_item(
    root_arg: str | os.PathLike[str], *, coder_token: str, display_item_key: str
) -> None:
    """Irreversibly close one adjudicated item so later items cannot starve."""

    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        item = _item(connection, display_item_key)
        if connection.execute(
            "SELECT referential_sample_frozen FROM workflow_meta WHERE singleton=1"
        ).fetchone()["referential_sample_frozen"]:
            _fail("E_REFERENTIAL_SAMPLE_ALREADY_FROZEN")
        closed = connection.execute(
            "SELECT COUNT(*) n FROM timing_item_locks WHERE item_id=?", (item["item_id"],)
        ).fetchone()["n"]
        if closed != 2:
            _fail("E_TIMING_SOURCES_NOT_CLOSED")
        accepted = connection.execute(
            "SELECT COUNT(*) n FROM adjudicated_utterances WHERE item_id=?", (item["item_id"],)
        ).fetchone()["n"]
        if accepted < 1:
            _fail("E_NO_ACCEPTED_UTTERANCES")
        total_sources = connection.execute(
            "SELECT COUNT(*) n FROM timing_segments WHERE item_id=? AND locked=1", (item["item_id"],)
        ).fetchone()["n"]
        dispositioned_sources = connection.execute(
            """SELECT COUNT(DISTINCT segment_id) n FROM (
                 SELECT source_segment_a segment_id FROM adjudicated_utterances
                   WHERE item_id=? AND source_segment_a IS NOT NULL
                 UNION
                 SELECT source_segment_b segment_id FROM adjudicated_utterances
                   WHERE item_id=? AND source_segment_b IS NOT NULL
               )""",
            (item["item_id"], item["item_id"]),
        ).fetchone()["n"]
        stopping = connection.execute(
            """SELECT COUNT(*) n, COALESCE(SUM(offset_ms-onset_ms),0) speech_ms
               FROM adjudicated_utterances WHERE speaker_role!='NONSPEECH'"""
        ).fetchone()
        threshold_reached = stopping["n"] >= 300 or stopping["speech_ms"] >= 1_800_000
        if dispositioned_sources != total_sources and not threshold_reached:
            _fail("E_TIMING_SOURCES_UNDISPOSITIONED")
        completion_reason = "FROZEN_THRESHOLD_REACHED" if threshold_reached else "ALL_SOURCES_DISPOSITIONED"
        connection.execute(
            """INSERT INTO timing_adjudication_item_locks
               (item_id, accepted_utterance_count, completion_reason, completed_at_utc)
               VALUES (?, ?, ?, ?)""",
            (item["item_id"], accepted, completion_reason, _utc_now()),
        )
        _audit(connection, actor, "TIMING_ADJUDICATION_ITEM_COMPLETED", ADJUDICATOR_SLOT)


def next_utterance_display_key(root_arg: str | os.PathLike[str], *, coder_token: str) -> str:
    """Allocate the next opaque UI key without revealing an internal identifier."""

    with _db(root_arg) as (connection, secret, _database):
        _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        rows = connection.execute("SELECT display_key FROM adjudicated_utterances").fetchall()
        highest = max((int(row["display_key"].split("-", 1)[1]) for row in rows), default=0)
        if highest >= 999999:
            _fail("E_UTTERANCE_KEY_EXHAUSTED")
        return f"R-{highest + 1:04d}"


def adjudicate_utterance(
    root_arg: str | os.PathLike[str],
    *,
    coder_token: str,
    display_item_key: str,
    display_utterance_key: str,
    source_segment_a: int | None,
    source_segment_b: int | None,
    onset_ms: int,
    offset_ms: int,
    source_text: str,
    language_code: str,
    speaker_role: str,
    reason_code: str,
) -> None:
    if speaker_role not in ROLE_VALUES:
        _fail("E_ROLE_INVALID")
    language_code = _validate_language(language_code)
    if not re.fullmatch(r"R-[0-9]{4,6}", display_utterance_key):
        _fail("E_UTTERANCE_DISPLAY_KEY")
    if not reason_code.strip() or len(reason_code) > 200:
        _fail("E_REASON_REQUIRED")
    if not isinstance(onset_ms, int) or not isinstance(offset_ms, int) or onset_ms < 0 or offset_ms <= onset_ms:
        _fail("E_TIMING_INVALID")
    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        item = _item(connection, display_item_key)
        if connection.execute(
            "SELECT referential_sample_frozen FROM workflow_meta WHERE singleton=1"
        ).fetchone()["referential_sample_frozen"]:
            _fail("E_REFERENTIAL_SAMPLE_ALREADY_FROZEN")
        if connection.execute(
            "SELECT 1 FROM timing_adjudication_item_locks WHERE item_id=?", (item["item_id"],)
        ).fetchone():
            _fail("E_TIMING_ADJUDICATION_ITEM_COMPLETED")
        closed_passes = connection.execute(
            "SELECT COUNT(*) n FROM timing_item_locks WHERE item_id=?", (item["item_id"],)
        ).fetchone()["n"]
        if closed_passes != 2:
            _fail("E_TIMING_SOURCES_NOT_CLOSED")
        stopping_before = connection.execute(
            """SELECT COUNT(*) n, COALESCE(SUM(offset_ms-onset_ms),0) speech_ms
               FROM adjudicated_utterances WHERE speaker_role!='NONSPEECH'"""
        ).fetchone()
        if stopping_before["n"] >= 300 or stopping_before["speech_ms"] >= 1_800_000:
            _fail("E_FROZEN_MINIMUM_REACHED_FREEZE_REQUIRED")
        if offset_ms > item["duration_ms"]:
            _fail("E_TIMING_OUTSIDE_MEDIA")
        for segment_id, required_slot in ((source_segment_a, "TIMING_A"), (source_segment_b, "TIMING_B")):
            if segment_id is None:
                continue
            segment = connection.execute(
                "SELECT item_id, slot, locked FROM timing_segments WHERE segment_id = ?", (segment_id,)
            ).fetchone()
            if not segment or segment["item_id"] != item["item_id"] or segment["slot"] != required_slot or not segment["locked"]:
                _fail("E_ADJUDICATION_SOURCE_INVALID")
            if connection.execute(
                """SELECT 1 FROM adjudicated_utterances
                   WHERE source_segment_a=? OR source_segment_b=?""",
                (segment_id, segment_id),
            ).fetchone():
                _fail("E_ADJUDICATION_SOURCE_ALREADY_DISPOSITIONED")
        if source_segment_a is None and source_segment_b is None:
            _fail("E_ADJUDICATION_SOURCE_REQUIRED")
        if connection.execute(
            "SELECT 1 FROM adjudicated_utterances WHERE display_key = ?", (display_utterance_key,)
        ).fetchone():
            _fail("E_ADJUDICATION_IMMUTABLE")
        text_value = source_text.strip()
        if speaker_role == "NONSPEECH":
            text_value = "[NONSPEECH]"
        elif not text_value:
            _fail("E_SOURCE_TEXT_REQUIRED")
        if len(text_value) > 5000:
            _fail("E_SOURCE_TEXT_TOO_LONG")
        cursor = connection.execute(
            """INSERT INTO adjudicated_utterances
               (internal_key, display_key, item_id, source_segment_a, source_segment_b,
                onset_ms, offset_ms, source_text, language_code, speaker_role, reason_code,
                locked, created_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                secrets.token_urlsafe(18), display_utterance_key, item["item_id"], source_segment_a,
                source_segment_b, onset_ms, offset_ms, text_value, language_code, speaker_role,
                reason_code.strip(), _utc_now(),
            ),
        )
        final_value = {
            "utterance_internal_id": cursor.lastrowid,
            "preserved_source_a": source_segment_a is not None,
            "preserved_source_b": source_segment_b is not None,
        }
        connection.execute(
            "INSERT INTO adjudications VALUES ('TIMING_ROLE_TEXT', ?, ?, ?, ?, ?)",
            (cursor.lastrowid, json.dumps(final_value, sort_keys=True), reason_code.strip(), actor, _utc_now()),
        )
        _audit(connection, actor, "TIMING_ADJUDICATED", ADJUDICATOR_SLOT)


def freeze_referential_sample(root_arg: str | os.PathLike[str], *, coder_token: str) -> dict[str, Any]:
    """Freeze all eligible A assignments and >=20% B assignments per opaque stratum."""

    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        meta = connection.execute("SELECT * FROM workflow_meta WHERE singleton = 1").fetchone()
        if meta["referential_sample_frozen"]:
            counts = connection.execute(
                "SELECT COUNT(*) n, SUM(double_code) d FROM referential_assignments"
            ).fetchone()
            return {"eligible_count": counts["n"], "double_coded_count": counts["d"], "status": "already_frozen"}
        stopping = connection.execute(
            """SELECT COUNT(*) n, COALESCE(SUM(offset_ms-onset_ms),0) speech_ms
               FROM adjudicated_utterances WHERE speaker_role!='NONSPEECH'"""
        ).fetchone()
        if stopping["n"] < 300 and stopping["speech_ms"] < 1_800_000:
            _fail("E_FROZEN_MINIMUM_NOT_REACHED")
        contributing = connection.execute(
            "SELECT COUNT(DISTINCT item_id) n FROM adjudicated_utterances"
        ).fetchone()["n"]
        completed_contributing = connection.execute(
            """SELECT COUNT(DISTINCT u.item_id) n FROM adjudicated_utterances u
               JOIN timing_adjudication_item_locks done ON done.item_id=u.item_id"""
        ).fetchone()["n"]
        if not contributing or completed_contributing != contributing:
            _fail("E_TIMING_ADJUDICATION_NOT_COMPLETED")
        rows = connection.execute(
            """SELECT u.utterance_id, i.internal_key item_internal_key, i.stratum_key,
                      u.onset_ms, u.offset_ms,
                      COALESCE(a.display_key,'NO_SOURCE_A') source_display_a,
                      COALESCE(b.display_key,'NO_SOURCE_B') source_display_b
               FROM adjudicated_utterances u JOIN items i ON i.item_id = u.item_id
               LEFT JOIN timing_segments a ON a.segment_id=u.source_segment_a
               LEFT JOIN timing_segments b ON b.segment_id=u.source_segment_b
               WHERE u.speaker_role = 'NON_CHILD'
               ORDER BY i.stratum_key, u.internal_key"""
        ).fetchall()
        if not rows:
            _fail("E_NO_REFERENTIAL_ITEMS")
        by_stratum: dict[str, list[tuple[str, sqlite3.Row]]] = {}
        for row in rows:
            assignment_hash = hashlib.sha256(
                b"childlens-v1.2-referential-double-code\0"
                + meta["selection_digest"].encode()
                + b"\0"
                + row["item_internal_key"].encode()
                + b"\0"
                + str(row["onset_ms"]).encode()
                + b"\0"
                + str(row["offset_ms"]).encode()
                + b"\0"
                + row["source_display_a"].encode()
                + b"\0"
                + row["source_display_b"].encode()
            ).hexdigest()
            by_stratum.setdefault(row["stratum_key"], []).append((assignment_hash, row))
        double_ids: set[int] = set()
        for candidates in by_stratum.values():
            candidates.sort(key=lambda pair: pair[0])
            take = math.ceil(0.2 * len(candidates))
            double_ids.update(pair[1]["utterance_id"] for pair in candidates[:take])
        frozen_at = _utc_now()
        for candidates in by_stratum.values():
            for assignment_hash, row in candidates:
                connection.execute(
                    """INSERT INTO referential_assignments
                       (utterance_id, double_code, assignment_hash, proposal_condition, frozen_at_utc)
                       VALUES (?, ?, ?, 'HUMAN_ONLY_REFERENCE', ?)""",
                    (row["utterance_id"], int(row["utterance_id"] in double_ids), assignment_hash, frozen_at),
                )
        connection.execute("UPDATE workflow_meta SET referential_sample_frozen = 1 WHERE singleton = 1")
        _audit(connection, actor, "REFERENTIAL_SAMPLE_FROZEN", ADJUDICATOR_SLOT)
        return {
            "status": "frozen",
            "eligible_count": len(rows),
            "double_coded_count": len(double_ids),
            "double_code_fraction": len(double_ids) / len(rows),
        }


def save_referential_label(
    root_arg: str | os.PathLike[str],
    *,
    slot: str,
    coder_token: str,
    display_utterance_key: str,
    status: str,
    mention_family: str,
    candidate_band: str,
    boundary_onset_ms: int | None = None,
    boundary_offset_ms: int | None = None,
    boundary_censored: bool = False,
    lock: bool = False,
) -> None:
    if slot not in REFERENTIAL_SLOTS:
        _fail("E_REFERENTIAL_SLOT")
    if status not in REFERENTIAL_VALUES:
        _fail("E_REFERENTIAL_STATUS_INVALID")
    if mention_family not in MENTION_FAMILIES:
        _fail("E_MENTION_FAMILY_INVALID")
    if candidate_band not in CANDIDATE_BANDS:
        _fail("E_CANDIDATE_BAND_INVALID")
    if (boundary_onset_ms is None) != (boundary_offset_ms is None):
        _fail("E_BOUNDARY_PAIR_REQUIRED")
    if boundary_onset_ms is not None and (
        not isinstance(boundary_onset_ms, int)
        or not isinstance(boundary_offset_ms, int)
        or boundary_onset_ms < 0
        or boundary_offset_ms < boundary_onset_ms
    ):
        _fail("E_BOUNDARY_INVALID")
    visible = status in {"VISIBLE_SINGLE", "VISIBLE_MULTIPLE"}
    if not visible and boundary_onset_ms is not None:
        _fail("E_BOUNDARY_FOR_NONVISIBLE")
    _validate_referential_combination(
        status,
        mention_family,
        candidate_band,
        boundary_onset_ms,
        boundary_offset_ms,
        boundary_censored,
    )
    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, slot, coder_token)
        row = connection.execute(
            """SELECT u.utterance_id, u.onset_ms, a.double_code
               FROM adjudicated_utterances u
               JOIN referential_assignments a ON a.utterance_id = u.utterance_id
               WHERE u.display_key = ?""",
            (display_utterance_key,),
        ).fetchone()
        if not row:
            _fail("E_REFERENTIAL_ASSIGNMENT_NOT_FOUND")
        if slot == "REFERENTIAL_B" and not row["double_code"]:
            _fail("E_NOT_DOUBLE_CODED")
        if boundary_onset_ms is not None:
            window_start = max(0, row["onset_ms"] - 5000)
            window_end = row["onset_ms"] + 5000
            if boundary_onset_ms < window_start or boundary_offset_ms > window_end:
                _fail("E_BOUNDARY_OUTSIDE_FROZEN_WINDOW")
        prior = connection.execute(
            "SELECT locked FROM referential_labels WHERE utterance_id = ? AND slot = ?",
            (row["utterance_id"], slot),
        ).fetchone()
        if prior and prior["locked"]:
            _fail("E_RECORD_LOCKED")
        connection.execute(
            """INSERT INTO referential_labels
               (utterance_id, slot, status, mention_family, candidate_band, boundary_onset_ms,
                boundary_offset_ms, boundary_censored, locked, updated_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(utterance_id, slot) DO UPDATE SET
                 status=excluded.status, mention_family=excluded.mention_family,
                 candidate_band=excluded.candidate_band, boundary_onset_ms=excluded.boundary_onset_ms,
                 boundary_offset_ms=excluded.boundary_offset_ms,
                 boundary_censored=excluded.boundary_censored, locked=excluded.locked,
                 updated_at_utc=excluded.updated_at_utc""",
            (
                row["utterance_id"], slot, status, mention_family, candidate_band,
                boundary_onset_ms, boundary_offset_ms, int(boundary_censored), int(lock), _utc_now(),
            ),
        )
        _audit(connection, actor, "REFERENTIAL_LOCKED" if lock else "REFERENTIAL_SAVED", slot)


def referential_task_context(
    root_arg: str | os.PathLike[str], *, slot: str, coder_token: str, display_utterance_key: str
) -> dict[str, Any]:
    """Return the accepted text/window to its assigned referential coder only."""

    if slot not in REFERENTIAL_SLOTS:
        _fail("E_REFERENTIAL_SLOT")
    with _db(root_arg) as (connection, secret, _database):
        _require_actor(connection, secret, slot, coder_token)
        row = connection.execute(
            """SELECT u.utterance_id, u.display_key, u.onset_ms, u.offset_ms, u.source_text,
                      u.language_code, a.double_code, i.display_key item_display_key
               FROM adjudicated_utterances u
               JOIN referential_assignments a ON a.utterance_id=u.utterance_id
               JOIN items i ON i.item_id=u.item_id
               WHERE u.display_key=?""",
            (display_utterance_key,),
        ).fetchone()
        if not row or (slot == "REFERENTIAL_B" and not row["double_code"]):
            _fail("E_REFERENTIAL_ASSIGNMENT_NOT_FOUND")
        return dict(row)


def referential_adjudication_tasks(
    root_arg: str | os.PathLike[str], *, coder_token: str, batch_size: int = 20
) -> list[dict[str, Any]]:
    """Reveal both locked labels only to the adjudicator and only after independence."""

    if not 1 <= batch_size <= 50:
        _fail("E_BATCH_REQUEST_INVALID")
    with _db(root_arg) as (connection, secret, _database):
        _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        rows = connection.execute(
            """SELECT u.display_key, u.onset_ms utterance_onset_ms, u.source_text,
                      la.status status_a, la.mention_family family_a,
                      la.candidate_band band_a, la.boundary_onset_ms onset_a,
                      la.boundary_offset_ms offset_a, la.boundary_censored censored_a,
                      lb.status status_b, lb.mention_family family_b,
                      lb.candidate_band band_b, lb.boundary_onset_ms onset_b,
                      lb.boundary_offset_ms offset_b, lb.boundary_censored censored_b
               FROM referential_assignments r
               JOIN adjudicated_utterances u ON u.utterance_id=r.utterance_id
               JOIN referential_labels la ON la.utterance_id=r.utterance_id
                    AND la.slot='REFERENTIAL_A' AND la.locked=1
               JOIN referential_labels lb ON lb.utterance_id=r.utterance_id
                    AND lb.slot='REFERENTIAL_B' AND lb.locked=1
               LEFT JOIN adjudicated_referential j ON j.utterance_id=r.utterance_id
               WHERE r.double_code=1 AND j.utterance_id IS NULL
               ORDER BY r.assignment_hash LIMIT ?""",
            (batch_size,),
        ).fetchall()
        return [dict(row) for row in rows]


def adjudicate_referential(
    root_arg: str | os.PathLike[str],
    *,
    coder_token: str,
    display_utterance_key: str,
    status: str,
    mention_family: str,
    candidate_band: str,
    reason_code: str,
    boundary_onset_ms: int | None = None,
    boundary_offset_ms: int | None = None,
    boundary_censored: bool = False,
) -> None:
    if status not in REFERENTIAL_VALUES:
        _fail("E_REFERENTIAL_STATUS_INVALID")
    if mention_family not in MENTION_FAMILIES:
        _fail("E_MENTION_FAMILY_INVALID")
    if candidate_band not in CANDIDATE_BANDS:
        _fail("E_CANDIDATE_BAND_INVALID")
    if not reason_code.strip() or len(reason_code) > 200:
        _fail("E_REASON_REQUIRED")
    if (boundary_onset_ms is None) != (boundary_offset_ms is None):
        _fail("E_BOUNDARY_PAIR_REQUIRED")
    if boundary_onset_ms is not None and (
        not isinstance(boundary_onset_ms, int)
        or not isinstance(boundary_offset_ms, int)
        or boundary_onset_ms < 0
        or boundary_offset_ms < boundary_onset_ms
    ):
        _fail("E_BOUNDARY_INVALID")
    if status not in {"VISIBLE_SINGLE", "VISIBLE_MULTIPLE"} and boundary_onset_ms is not None:
        _fail("E_BOUNDARY_FOR_NONVISIBLE")
    _validate_referential_combination(
        status,
        mention_family,
        candidate_band,
        boundary_onset_ms,
        boundary_offset_ms,
        boundary_censored,
    )
    with _db(root_arg) as (connection, secret, _database):
        actor = _require_actor(connection, secret, ADJUDICATOR_SLOT, coder_token)
        row = connection.execute(
            """SELECT u.utterance_id, u.onset_ms
               FROM adjudicated_utterances u
               JOIN referential_assignments r ON r.utterance_id=u.utterance_id AND r.double_code=1
               JOIN referential_labels a ON a.utterance_id=u.utterance_id
                    AND a.slot='REFERENTIAL_A' AND a.locked=1
               JOIN referential_labels b ON b.utterance_id=u.utterance_id
                    AND b.slot='REFERENTIAL_B' AND b.locked=1
               WHERE u.display_key=?""",
            (display_utterance_key,),
        ).fetchone()
        if not row:
            _fail("E_ADJUDICATION_SOURCE_INVALID")
        if boundary_onset_ms is not None:
            if boundary_onset_ms < max(0, row["onset_ms"] - 5000) or boundary_offset_ms > row["onset_ms"] + 5000:
                _fail("E_BOUNDARY_OUTSIDE_FROZEN_WINDOW")
        if connection.execute(
            "SELECT 1 FROM adjudicated_referential WHERE utterance_id=?", (row["utterance_id"],)
        ).fetchone():
            _fail("E_ADJUDICATION_IMMUTABLE")
        connection.execute(
            """INSERT INTO adjudicated_referential
               (utterance_id, status, mention_family, candidate_band, boundary_onset_ms,
                boundary_offset_ms, boundary_censored, reason_code, adjudicator_hash, created_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["utterance_id"], status, mention_family, candidate_band, boundary_onset_ms,
                boundary_offset_ms, int(boundary_censored), reason_code.strip(), actor, _utc_now(),
            ),
        )
        final_value = {
            "status": status,
            "mention_family": mention_family,
            "candidate_band": candidate_band,
            "boundary_present": boundary_onset_ms is not None,
            "boundary_censored": bool(boundary_censored),
        }
        connection.execute(
            "INSERT INTO adjudications VALUES ('REFERENTIAL', ?, ?, ?, ?, ?)",
            (row["utterance_id"], json.dumps(final_value, sort_keys=True), reason_code.strip(), actor, _utc_now()),
        )
        _audit(connection, actor, "REFERENTIAL_ADJUDICATED", ADJUDICATOR_SLOT)


def get_task_batch(
    root_arg: str | os.PathLike[str], *, slot: str, coder_token: str, batch_size: int = 20
) -> list[dict[str, Any]]:
    """Return only opaque display keys and minimum task state; never peer labels."""

    if slot not in SLOTS or not 1 <= batch_size <= 50:
        _fail("E_BATCH_REQUEST_INVALID")
    with _db(root_arg) as (connection, secret, _database):
        _require_actor(connection, secret, slot, coder_token)
        if slot in LANGUAGE_SLOTS:
            rows = connection.execute(
                """SELECT i.display_key, i.batch_number,
                          CASE WHEN l.item_id IS NULL THEN 'PENDING'
                               WHEN l.locked = 1 THEN 'LOCKED' ELSE 'DRAFT' END state
                   FROM items i LEFT JOIN language_labels l ON l.item_id=i.item_id AND l.slot=?
                   WHERE l.item_id IS NULL OR l.locked=0
                   ORDER BY i.batch_number, i.display_key LIMIT ?""",
                (slot, batch_size),
            ).fetchall()
        elif slot in TIMING_SLOTS:
            rows = connection.execute(
                """SELECT i.display_key, i.batch_number,
                          CASE WHEN x.item_id IS NULL THEN 'PENDING' ELSE 'DRAFT' END state
                   FROM items i LEFT JOIN timing_item_locks x ON x.item_id=i.item_id AND x.slot=?
                   WHERE x.item_id IS NULL
                     AND (SELECT COUNT(*) FROM adjudicated_languages)=(SELECT COUNT(*) FROM items)
                   ORDER BY i.batch_number, i.display_key LIMIT ?""",
                (slot, batch_size),
            ).fetchall()
        elif slot in REFERENTIAL_SLOTS:
            rows = connection.execute(
                """SELECT u.display_key, i.batch_number,
                          CASE WHEN l.utterance_id IS NULL THEN 'PENDING'
                               WHEN l.locked=1 THEN 'LOCKED' ELSE 'DRAFT' END state
                   FROM referential_assignments a
                   JOIN adjudicated_utterances u ON u.utterance_id=a.utterance_id
                   JOIN items i ON i.item_id=u.item_id
                   LEFT JOIN referential_labels l ON l.utterance_id=u.utterance_id AND l.slot=?
                   WHERE (?='REFERENTIAL_A' OR a.double_code=1)
                     AND (l.utterance_id IS NULL OR l.locked=0)
                   ORDER BY i.batch_number, a.assignment_hash LIMIT ?""",
                (slot, slot, batch_size),
            ).fetchall()
        else:
            return []
        return [dict(row) for row in rows]


def media_for_display(
    root_arg: str | os.PathLike[str], *, slot: str, coder_token: str, display_key: str
) -> Path:
    """Resolve media internally. Callers must never render or log the returned path."""

    with _db(root_arg) as (connection, secret, _database):
        _require_actor(connection, secret, slot, coder_token)
        if display_key.startswith("HV-"):
            if slot in REFERENTIAL_SLOTS:
                _fail("E_TASK_NOT_ASSIGNED")
            item = _item(connection, display_key)
        elif display_key.startswith("R-"):
            if slot not in (*REFERENTIAL_SLOTS, ADJUDICATOR_SLOT):
                _fail("E_TASK_NOT_ASSIGNED")
            item = connection.execute(
                """SELECT i.* FROM adjudicated_utterances u
                   JOIN referential_assignments r ON r.utterance_id=u.utterance_id
                   JOIN items i ON i.item_id=u.item_id
                   WHERE u.display_key=?
                     AND (?='REFERENTIAL_A'
                          OR (?='REFERENTIAL_B' AND r.double_code=1)
                          OR (?='ADJUDICATOR' AND r.double_code=1
                              AND EXISTS (SELECT 1 FROM referential_labels a
                                          WHERE a.utterance_id=u.utterance_id
                                            AND a.slot='REFERENTIAL_A' AND a.locked=1)
                              AND EXISTS (SELECT 1 FROM referential_labels b
                                          WHERE b.utterance_id=u.utterance_id
                                            AND b.slot='REFERENTIAL_B' AND b.locked=1)))""",
                (display_key, slot, slot, slot),
            ).fetchone()
            if not item:
                _fail("E_ITEM_NOT_FOUND")
        else:
            _fail("E_DISPLAY_KEY_INVALID")
        root = validate_quarantine_root(root_arg)
        candidate = root / item["media_relpath"]
        try:
            metadata = candidate.lstat()
            path = candidate.resolve(strict=True)
        except OSError:
            _fail("E_MEDIA_MISSING")
        if (
            not _is_relative_to(path, root)
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("E_MEDIA_OUTSIDE_QUARANTINE")
        return path


def progress(root_arg: str | os.PathLike[str]) -> dict[str, Any]:
    """Return nonlexical progress aggregates only."""

    with _db(root_arg) as (connection, _secret, _database):
        total_items = connection.execute("SELECT COUNT(*) n FROM items").fetchone()["n"]
        language = {
            slot: connection.execute(
                "SELECT COUNT(*) n FROM language_labels WHERE slot=? AND locked=1", (slot,)
            ).fetchone()["n"]
            for slot in LANGUAGE_SLOTS
        }
        timing_locks = {
            slot: connection.execute(
                "SELECT COUNT(*) n FROM timing_item_locks WHERE slot=?", (slot,)
            ).fetchone()["n"]
            for slot in TIMING_SLOTS
        }
        adjudicated = connection.execute(
            """SELECT COUNT(*) n, COALESCE(SUM(offset_ms-onset_ms),0) speech_ms
               FROM adjudicated_utterances WHERE speaker_role!='NONSPEECH'"""
        ).fetchone()
        ref = connection.execute(
            """SELECT COUNT(*) n, COALESCE(SUM(double_code),0) d,
                      COALESCE(SUM(CASE WHEN la.locked=1 THEN 1 ELSE 0 END),0) a_done,
                      COALESCE(SUM(CASE WHEN lb.locked=1 THEN 1 ELSE 0 END),0) b_done
               FROM referential_assignments r
               LEFT JOIN referential_labels la ON la.utterance_id=r.utterance_id AND la.slot='REFERENTIAL_A'
               LEFT JOIN referential_labels lb ON lb.utterance_id=r.utterance_id AND lb.slot='REFERENTIAL_B'"""
        ).fetchone()
        count = adjudicated["n"]
        speech_seconds = adjudicated["speech_ms"] / 1000
        return {
            "workflow_version": VERSION,
            "items_total": total_items,
            "language_locked": language,
            "timing_items_locked": timing_locks,
            "adjudicated_utterances": count,
            "adjudicated_speech_minutes": round(speech_seconds / 60, 2),
            "frozen_minimum_reached": bool(count >= 300 or speech_seconds >= 1800),
            "referential_items": ref["n"],
            "referential_double_coded_required": ref["d"],
            "referential_a_locked": ref["a_done"],
            "referential_b_locked": ref["b_done"],
        }


def _nominal_alpha(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(a != b for a, b in pairs) / len(pairs)
    counts: dict[str, int] = {}
    for a, b in pairs:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    total = 2 * len(pairs)
    if total < 2:
        return None
    expected = (total * total - sum(count * count for count in counts.values())) / (total * (total - 1))
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, 1):
        current = [i]
        for j, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _macro_f1(pairs: Sequence[tuple[str, str]], classes: Sequence[str]) -> float | None:
    if not pairs:
        return None
    scores: list[float] = []
    for value in classes:
        true_positive = sum(a == value and b == value for a, b in pairs)
        false_positive = sum(a == value and b != value for a, b in pairs)
        false_negative = sum(a != value and b == value for a, b in pairs)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return sum(scores) / len(scores)


def reliability_aggregates(root_arg: str | os.PathLike[str]) -> dict[str, Any]:
    """Compute pre-adjudication reliability without exporting text or row keys."""

    with _db(root_arg) as (connection, _secret, _database):
        language_rows = connection.execute(
            """SELECT a.language_code a, b.language_code b
               FROM language_labels a JOIN language_labels b ON b.item_id=a.item_id
               WHERE a.slot='LANGUAGE_A' AND b.slot='LANGUAGE_B' AND a.locked=1 AND b.locked=1"""
        ).fetchall()
        timing_rows = connection.execute(
            """SELECT a.speaker_role role_a, b.speaker_role role_b,
                      a.source_text text_a, b.source_text text_b,
                      ABS(a.onset_ms-b.onset_ms) onset_error,
                      ABS(a.offset_ms-b.offset_ms) offset_error
               FROM adjudicated_utterances u
               LEFT JOIN timing_segments a ON a.segment_id=u.source_segment_a
               LEFT JOIN timing_segments b ON b.segment_id=u.source_segment_b"""
        ).fetchall()
        ref_rows = connection.execute(
            """SELECT a.status status_a, b.status status_b,
                      a.boundary_onset_ms onset_a, a.boundary_offset_ms offset_a,
                      b.boundary_onset_ms onset_b, b.boundary_offset_ms offset_b,
                      a.boundary_censored censored_a, b.boundary_censored censored_b
               FROM referential_assignments r
               JOIN referential_labels a ON a.utterance_id=r.utterance_id
                    AND a.slot='REFERENTIAL_A' AND a.locked=1
               JOIN referential_labels b ON b.utterance_id=r.utterance_id
                    AND b.slot='REFERENTIAL_B' AND b.locked=1
               WHERE r.double_code=1"""
        ).fetchall()
        language_pairs = [(row["a"], row["b"]) for row in language_rows]
        role_pairs = [
            (row["role_a"] if row["role_a"] is not None else "MISSING", row["role_b"] if row["role_b"] is not None else "MISSING")
            for row in timing_rows
        ]
        matched_timing = [row for row in timing_rows if row["role_a"] is not None and row["role_b"] is not None]
        unmatched_a = sum(row["role_a"] is None for row in timing_rows)
        unmatched_b = sum(row["role_b"] is None for row in timing_rows)
        grapheme_edits = 0
        grapheme_reference = 0
        for row in timing_rows:
            left = unicodedata.normalize("NFC", row["text_a"] or "")
            right = unicodedata.normalize("NFC", row["text_b"] or "")
            grapheme_edits += _edit_distance(left, right)
            grapheme_reference += max(1, len(right))
        status_pairs = [(row["status_a"], row["status_b"]) for row in ref_rows]
        boundary_rows = [
            row
            for row in ref_rows
            if row["onset_a"] is not None
            and row["onset_b"] is not None
            and not row["censored_a"]
            and not row["censored_b"]
        ]
        return {
            "language_double_coded_count": len(language_pairs),
            "language_exact_agreement": None
            if not language_pairs
            else round(sum(a == b for a, b in language_pairs) / len(language_pairs), 6),
            "timing_validation_record_count": len(timing_rows),
            "timing_matched_double_coded_count": len(matched_timing),
            "timing_unmatched_a_missing_count": unmatched_a,
            "timing_unmatched_b_missing_count": unmatched_b,
            "timing_matched_coverage_fraction": None
            if not timing_rows
            else round(len(matched_timing) / len(timing_rows), 6),
            "timing_both_edges_within_500ms_fraction": None
            if not timing_rows
            else round(
                sum(
                    row["onset_error"] is not None
                    and row["offset_error"] is not None
                    and row["onset_error"] <= 500
                    and row["offset_error"] <= 500
                    for row in timing_rows
                )
                / len(timing_rows),
                6,
            ),
            "speaker_role_exact_agreement": None
            if not role_pairs
            else round(sum(a == b for a, b in role_pairs) / len(role_pairs), 6),
            "speaker_role_nominal_alpha": None
            if not role_pairs
            else round(_nominal_alpha(role_pairs), 6) if _nominal_alpha(role_pairs) is not None else None,
            "speaker_role_four_class_macro_f1": None
            if not role_pairs
            else round(_macro_f1(role_pairs, ("NON_CHILD", "CHILD", "OVERLAP", "UNCERTAIN")), 6),
            "paired_source_text_codepoint_error_rate": None
            if not timing_rows
            else round(grapheme_edits / grapheme_reference, 6),
            "referential_double_coded_locked_count": len(status_pairs),
            "referential_status_exact_agreement": None
            if not status_pairs
            else round(sum(a == b for a, b in status_pairs) / len(status_pairs), 6),
            "referential_status_nominal_alpha": None
            if not status_pairs
            else round(_nominal_alpha(status_pairs), 6) if _nominal_alpha(status_pairs) is not None else None,
            "matched_uncensored_boundary_count": len(boundary_rows),
            "matched_boundaries_within_1s_fraction": None
            if not boundary_rows
            else round(
                sum(
                    abs(row["onset_a"] - row["onset_b"]) <= 1000
                    and abs(row["offset_a"] - row["offset_b"]) <= 1000
                    for row in boundary_rows
                )
                / len(boundary_rows),
                6,
            ),
        }


def validate_readiness(root_arg: str | os.PathLike[str]) -> dict[str, Any]:
    """Distinguish runtime handoff readiness from completed human evidence."""

    structural = validate_workflow(root_arg)
    with _db(root_arg) as (connection, _secret, _database):
        counts = {
            "items": connection.execute("SELECT COUNT(*) n FROM items").fetchone()["n"],
            "slots": connection.execute("SELECT COUNT(*) n FROM coder_slots").fetchone()["n"],
            "language_locked": connection.execute(
                "SELECT COUNT(*) n FROM language_labels WHERE locked=1"
            ).fetchone()["n"],
            "language_adjudicated": connection.execute(
                "SELECT COUNT(*) n FROM adjudicated_languages"
            ).fetchone()["n"],
            "timing_items_completed": connection.execute(
                "SELECT COUNT(*) n FROM timing_adjudication_item_locks"
            ).fetchone()["n"],
            "timing_contributing_items": connection.execute(
                "SELECT COUNT(DISTINCT item_id) n FROM adjudicated_utterances"
            ).fetchone()["n"],
            "timing_completed_contributing_items": connection.execute(
                """SELECT COUNT(DISTINCT u.item_id) n FROM adjudicated_utterances u
                   JOIN timing_adjudication_item_locks done ON done.item_id=u.item_id"""
            ).fetchone()["n"],
            "referential": connection.execute(
                "SELECT COUNT(*) n FROM referential_assignments"
            ).fetchone()["n"],
            "referential_a": connection.execute(
                "SELECT COUNT(*) n FROM referential_labels WHERE slot='REFERENTIAL_A' AND locked=1"
            ).fetchone()["n"],
            "referential_b": connection.execute(
                "SELECT COUNT(*) n FROM referential_labels WHERE slot='REFERENTIAL_B' AND locked=1"
            ).fetchone()["n"],
            "referential_b_required": connection.execute(
                "SELECT COALESCE(SUM(double_code),0) n FROM referential_assignments"
            ).fetchone()["n"],
            "referential_adjudicated": connection.execute(
                "SELECT COUNT(*) n FROM adjudicated_referential"
            ).fetchone()["n"],
        }
        frozen = bool(
            connection.execute(
                "SELECT referential_sample_frozen FROM workflow_meta WHERE singleton=1"
            ).fetchone()["referential_sample_frozen"]
        )
    prog = progress(root_arg)
    reliability = reliability_aggregates(root_arg)
    runtime_ready = structural["status"] == "STRUCTURAL_PASS" and counts["items"] == 15 and counts["slots"] == len(SLOTS)
    human_complete = all(
        (
            counts["language_locked"] == 30,
            counts["language_adjudicated"] == 15,
            counts["timing_items_completed"] >= 1,
            counts["timing_completed_contributing_items"] == counts["timing_contributing_items"],
            prog["frozen_minimum_reached"],
            frozen,
            counts["referential"] > 0,
            counts["referential_a"] == counts["referential"],
            counts["referential_b"] == counts["referential_b_required"],
            counts["referential_adjudicated"] == counts["referential_b_required"],
        )
    )
    reliability_pass = bool(
        human_complete
        and reliability["timing_matched_coverage_fraction"] is not None
        and reliability["timing_matched_coverage_fraction"] >= 0.70
        and reliability["timing_both_edges_within_500ms_fraction"] is not None
        and reliability["timing_both_edges_within_500ms_fraction"] >= 0.70
        and reliability["speaker_role_four_class_macro_f1"] is not None
        and reliability["speaker_role_four_class_macro_f1"] >= 0.80
        and reliability["referential_status_nominal_alpha"] is not None
        and reliability["referential_status_nominal_alpha"] >= 0.67
        and reliability["matched_boundaries_within_1s_fraction"] is not None
        and reliability["matched_boundaries_within_1s_fraction"] >= 0.80
    )
    return {
        "workflow_version": VERSION,
        "status": "HUMAN_EVIDENCE_COMPLETE" if human_complete else "LANGUAGE_PASS_READY" if runtime_ready else "NOT_READY",
        "runtime_handoff_ready": runtime_ready,
        "genuine_human_evidence_complete": human_complete,
        "frozen_reliability_thresholds_pass": reliability_pass,
        "frozen_minimum_reached": prog["frozen_minimum_reached"],
        "referential_inventory_frozen": frozen,
        "counts": counts,
        "pre_adjudication_reliability": reliability,
    }


def validate_workflow(root_arg: str | os.PathLike[str]) -> dict[str, Any]:
    """Run structural validation without exporting row-level content."""

    with _db(root_arg) as (connection, _secret, database):
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign = connection.execute("PRAGMA foreign_key_check").fetchall()
        meta = connection.execute("SELECT * FROM workflow_meta WHERE singleton=1").fetchone()
        if not meta or integrity != "ok" or foreign:
            _fail("E_DATABASE_INTEGRITY")
        if stat.S_IMODE(database.stat().st_mode) & 0o077:
            _fail("E_DATABASE_PERMISSIONS")
        bad_roles = connection.execute(
            "SELECT COUNT(*) n FROM timing_segments WHERE speaker_role NOT IN ('NON_CHILD','CHILD','OVERLAP','UNCERTAIN','NONSPEECH')"
        ).fetchone()["n"]
        bad_status = connection.execute(
            """SELECT COUNT(*) n FROM referential_labels
               WHERE status NOT IN ('VISIBLE_SINGLE','VISIBLE_MULTIPLE','NULL_NOT_VISIBLE','IRRELEVANT','UNDECIDABLE','UNUSABLE')"""
        ).fetchone()["n"]
        distinct_violation = 0
        for pair in INDEPENDENCE_PAIRS:
            placeholders = ",".join("?" for _ in pair)
            row = connection.execute(
                f"SELECT COUNT(*) n, COUNT(DISTINCT coder_hash) d FROM coder_slots WHERE slot IN ({placeholders})",
                tuple(pair),
            ).fetchone()
            if row["n"] == 2 and row["d"] != 2:
                distinct_violation += 1
        ref = connection.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(double_code),0) d FROM referential_assignments"
        ).fetchone()
        ref_fraction = (ref["d"] / ref["n"]) if ref["n"] else None
        failures: list[str] = []
        if bad_roles:
            failures.append("ROLE_ONTOLOGY")
        if bad_status:
            failures.append("REFERENTIAL_ONTOLOGY")
        if distinct_violation:
            failures.append("CODER_INDEPENDENCE")
        if meta["referential_sample_frozen"] and (not ref["n"] or ref_fraction is None or ref_fraction < 0.2):
            failures.append("DOUBLE_CODE_FRACTION")
        return {
            "workflow_version": VERSION,
            "status": "STRUCTURAL_PASS" if not failures else "STRUCTURAL_FAIL",
            "database_integrity": "PASS",
            "database_owner_only": True,
            "coder_independence": "PASS" if not distinct_violation else "FAIL",
            "ontology_validation": "PASS" if not (bad_roles or bad_status) else "FAIL",
            "referential_sample_frozen": bool(meta["referential_sample_frozen"]),
            "referential_double_code_fraction": None if ref_fraction is None else round(ref_fraction, 6),
            "failures": failures,
        }


def _safe_cli(action: callable) -> int:
    try:
        result = action()
    except WorkflowError as exc:
        print(json.dumps({"status": "error", "error_code": exc.code}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 3
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChildLens v1.2 restricted human-validation workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    bootstrap = sub.add_parser("bootstrap-policy")
    bootstrap.add_argument("--root", required=True)
    bootstrap.add_argument("--attest-all-controls", action="store_true")
    init = sub.add_parser("init")
    init.add_argument("--root", required=True)
    init.add_argument("--packet", required=True)
    init.add_argument("--frozen-v1-1-skeleton", required=True)
    init.add_argument("--frozen-v1-1-restricted-input", required=True)
    init.add_argument("--native-transfer-receipt", required=True)
    status = sub.add_parser("status")
    status.add_argument("--root", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--root", required=True)
    readiness = sub.add_parser("readiness")
    readiness.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    if args.command == "bootstrap-policy":
        return _safe_cli(lambda: {"status": "ok", "policy_created": bool(bootstrap_policy(args.root, attest=args.attest_all_controls))})
    if args.command == "init":
        return _safe_cli(
            lambda: initialize(
                args.root,
                args.packet,
                args.frozen_v1_1_skeleton,
                args.frozen_v1_1_restricted_input,
                args.native_transfer_receipt,
            )
        )
    if args.command == "status":
        return _safe_cli(lambda: progress(args.root))
    if args.command == "readiness":
        return _safe_cli(lambda: validate_readiness(args.root))
    return _safe_cli(lambda: validate_workflow(args.root))


if __name__ == "__main__":
    raise SystemExit(main())
