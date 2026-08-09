#!/usr/bin/env python3
"""Fail-closed local-inference boundary for ChildLens v1.3.

This module is deliberately content agnostic.  It does not discover a runtime,
media, manifests, or model outputs.  A quarantine-side coordinator may import
it and supply already-authorized paths.  Restricted inputs and pseudo-labels
are passed to fixed offline adapters only through inherited file descriptors;
adapter stdout and stderr are discarded and row-level output is never parsed
for a repository-facing receipt.

The public interface separates two irreversible phases:

* ``seal_instruments`` validates fixed, separately licensed instrument
  receipts after all downloads are complete and before restricted data is
  mounted; and
* ``RestrictedInferenceRunner`` refuses to invoke an adapter unless an OS
  network-isolation backend passes an active, local socket-denial sentinel.

The code is generic infrastructure.  It neither downloads models nor performs
ChildLens inference on its own.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


VERSION = "childlens-local-inference-firewall-v1.3.0"
INSTRUMENT_RECEIPT_SCHEMA = "childlens-fixed-instrument-receipt-v1.3.0"
INSTRUMENT_SEAL_SCHEMA = "childlens-fixed-instrument-seal-v1.3.0"
AGGREGATE_RECEIPT_SCHEMA = "childlens-v1.3-pseudo-annotation-receipt-v1"
ADAPTER_CONTRACT = "childlens-offline-fd-adapter-v1.3.0"
ADAPTER_CONTRACT_SHA256 = hashlib.sha256(ADAPTER_CONTRACT.encode("utf-8")).hexdigest()
CHECKPOINT_SCHEMA = "childlens-offline-checkpoint-v1.3.0"

MIN_CPU_WORKERS = 2
MAX_CPU_WORKERS = 4
MAX_MPS_HEAVY_PROCESSES = 1
MAX_PSEUDO_LABEL_BYTES = 32 * 1024 * 1024
CELL_SUPPRESSION_K = 5
FULL_PILOT_WINDOW_COUNT = 912
FULL_PILOT_SPEECH_MINUTES = 135.25
FROZEN_SELECTED_ITEM_COUNT = 15

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,80}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+:-]{0,79}$")
LICENSE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+(): /-]{1,119}$")
ABSOLUTE_PATH_RE = re.compile(r"(?i)(?:/Users/|/home/|file://|\\Users\\)")
RESTRICTED_NAME_RE = re.compile(
    r"(?i)(?:participant|subject|child_id|session_id|filename|source_path|"
    r"timestamp|transcript|utterance_text|frame_path|media_path)"
)

TASKS = frozenset(
    {
        "VAD_SEGMENTATION",
        "MULTILINGUAL_ASR",
        "SPEAKER_ROLE_AID",
        "REFERENTIAL_CANDIDATE",
    }
)
RESOURCE_CLASSES = frozenset({"CPU", "MPS_HEAVY"})
ROLE_AID_LABELS = frozenset({"NON_CHILD", "CHILD", "OVERLAP", "UNCERTAIN"})

# These profiles freeze only the local adapter contract and resource envelope.
# Artifact/version/license hashes remain mandatory in the instrument receipt.
# No download URL or executable path is embedded here.
FIXED_ADAPTER_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "silero_vad": {
        "task": "VAD_SEGMENTATION",
        "resource_class": "CPU",
        "model_family": "Silero VAD",
        "required_model_revision": "6.2.1",
        "required_adapter_flags": ("--offline",),
    },
    "whisper_cpp_large_v3_turbo": {
        "task": "MULTILINGUAL_ASR",
        "resource_class": "MPS_HEAVY",
        "model_family": "whisper.cpp ggml-large-v3-turbo.bin",
        "required_model_revision": "5359861c739e955e79d9a303bcbc70fb988958b1",
        "required_adapter_flags": ("--offline", "--word-timestamps"),
    },
    "conservative_role_aid": {
        "task": "SPEAKER_ROLE_AID",
        "resource_class": "CPU",
        "model_family": "deterministic conservative acoustic role rules",
        "required_model_revision": "childlens-role-rules-v1.3.0",
        "required_adapter_flags": ("--offline", "--uncertain-fallback"),
        "allowed_labels": tuple(sorted(ROLE_AID_LABELS)),
        "required_fallback": "UNCERTAIN",
    },
    "qwen2_vl_2b_instruct": {
        "task": "REFERENTIAL_CANDIDATE",
        "resource_class": "MPS_HEAVY",
        "model_family": "Qwen/Qwen2-VL-2B-Instruct",
        "required_model_revision": "895c3a49bc3fa70a340399125c650a463535e71c",
        "required_adapter_flags": ("--offline", "--bounded-frame-batch"),
    },
}

_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "instrument_key",
        "adapter_profile",
        "task",
        "resource_class",
        "software",
        "model",
        "acquisition",
        "boundary",
    }
)
_SOFTWARE_KEYS = frozenset(
    {
        "name",
        "version",
        "code_sha256",
        "executable_sha256",
        "adapter_contract_sha256",
        "license_identifier",
        "license_evidence_sha256",
    }
)
_MODEL_KEYS = frozenset(
    {
        "name",
        "version",
        "artifact_set_sha256",
        "license_identifier",
        "license_evidence_sha256",
        "authoritative_source_verified",
    }
)
_ACQUISITION_KEYS = frozenset(
    {
        "completed_before_restricted_processing",
        "hashes_verified",
        "license_review_status",
        "clickthrough_accepted_by_automation",
        "telemetry_disabled",
    }
)
_BOUNDARY_KEYS = frozenset(
    {
        "offline_only",
        "external_api_allowed",
        "external_upload_allowed",
        "network_isolation_required",
        "restricted_outputs_quarantine_only",
        "learner_weight_ancestry_allowed",
        "learner_tokenizer_or_vocabulary_ancestry_allowed",
        "learner_embedding_or_feature_ancestry_allowed",
        "learner_score_or_confidence_ancestry_allowed",
        "primary_evaluation_truth_allowed",
        "simulator_oracle_replacement_allowed",
    }
)


class FirewallError(RuntimeError):
    """A fixed diagnostic code safe to expose."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise FirewallError(code)


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


def _sha256_artifact(path: Path) -> str:
    """Hash a file or a symlink-free private artifact tree deterministically."""

    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir() or path.is_symlink():
        _fail("E_MODEL_ARTIFACT")
    digest = hashlib.sha256()
    try:
        entries = sorted(path.rglob("*"), key=lambda child: child.relative_to(path).as_posix())
    except OSError:
        _fail("E_MODEL_ARTIFACT")
    for child in entries:
        if child.is_symlink():
            _fail("E_MODEL_ARTIFACT")
        relative = child.relative_to(path).as_posix().encode("utf-8")
        if child.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif child.is_file():
            digest.update(b"F\0" + relative + b"\0")
            try:
                with child.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError:
                _fail("E_MODEL_ARTIFACT")
        else:
            _fail("E_MODEL_ARTIFACT")
    return digest.hexdigest()


def _exact_keys(value: Any, expected: frozenset[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != expected:
        _fail(code)
    return value


def _fixed_string(value: Any, pattern: re.Pattern[str], code: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        _fail(code)
    return value


def _fixed_bool(value: Any, expected: bool, code: str) -> None:
    if value is not expected:
        _fail(code)


def validate_instrument_receipt(receipt: Mapping[str, Any]) -> str:
    """Validate a fixed local instrument receipt and return its digest.

    The validator intentionally rejects extra fields.  Source paths, URLs,
    access tokens, free-form notes, and model outputs therefore cannot be
    smuggled into the public instrument seal.
    """

    root = _exact_keys(receipt, _RECEIPT_KEYS, "E_INSTRUMENT_RECEIPT_SCHEMA")
    if root["schema_version"] != INSTRUMENT_RECEIPT_SCHEMA:
        _fail("E_INSTRUMENT_RECEIPT_VERSION")
    _fixed_string(root["instrument_key"], KEY_RE, "E_INSTRUMENT_KEY")
    profile_name = _fixed_string(root["adapter_profile"], KEY_RE, "E_ADAPTER_PROFILE")
    profile = FIXED_ADAPTER_PROFILES.get(profile_name)
    if profile is None:
        _fail("E_ADAPTER_PROFILE")
    if root["task"] not in TASKS or root["task"] != profile["task"]:
        _fail("E_INSTRUMENT_TASK")
    if root["resource_class"] not in RESOURCE_CLASSES or root["resource_class"] != profile[
        "resource_class"
    ]:
        _fail("E_RESOURCE_CLASS")

    software = _exact_keys(root["software"], _SOFTWARE_KEYS, "E_SOFTWARE_RECEIPT")
    _fixed_string(software["name"], KEY_RE, "E_SOFTWARE_RECEIPT")
    _fixed_string(software["version"], VERSION_RE, "E_SOFTWARE_RECEIPT")
    for key in ("code_sha256", "executable_sha256", "adapter_contract_sha256"):
        _fixed_string(software[key], HEX64_RE, "E_SOFTWARE_DIGEST")
    if software["adapter_contract_sha256"] != ADAPTER_CONTRACT_SHA256:
        _fail("E_ADAPTER_CONTRACT_DIGEST")
    _fixed_string(software["license_identifier"], LICENSE_RE, "E_SOFTWARE_LICENSE")
    _fixed_string(software["license_evidence_sha256"], HEX64_RE, "E_SOFTWARE_LICENSE")

    model = _exact_keys(root["model"], _MODEL_KEYS, "E_MODEL_RECEIPT")
    if not isinstance(model["name"], str) or not (1 <= len(model["name"]) <= 120):
        _fail("E_MODEL_RECEIPT")
    _fixed_string(model["version"], VERSION_RE, "E_MODEL_RECEIPT")
    _fixed_string(model["artifact_set_sha256"], HEX64_RE, "E_MODEL_DIGEST")
    _fixed_string(model["license_identifier"], LICENSE_RE, "E_MODEL_LICENSE")
    _fixed_string(model["license_evidence_sha256"], HEX64_RE, "E_MODEL_LICENSE")
    _fixed_bool(model["authoritative_source_verified"], True, "E_MODEL_SOURCE")
    required_revision = profile.get("required_model_revision")
    if required_revision is not None and model["version"] != required_revision:
        _fail("E_MODEL_REVISION")

    acquisition = _exact_keys(root["acquisition"], _ACQUISITION_KEYS, "E_ACQUISITION_RECEIPT")
    for key in ("completed_before_restricted_processing", "hashes_verified", "telemetry_disabled"):
        _fixed_bool(acquisition[key], True, "E_ACQUISITION_NOT_SEALED")
    if acquisition["license_review_status"] != "PASS":
        _fail("E_LICENSE_NOT_APPROVED")
    _fixed_bool(
        acquisition["clickthrough_accepted_by_automation"],
        False,
        "E_AUTOMATED_CLICKTHROUGH",
    )

    boundary = _exact_keys(root["boundary"], _BOUNDARY_KEYS, "E_INSTRUMENT_BOUNDARY")
    for key in (
        "offline_only",
        "network_isolation_required",
        "restricted_outputs_quarantine_only",
    ):
        _fixed_bool(boundary[key], True, "E_INSTRUMENT_BOUNDARY")
    for key in (
        "external_api_allowed",
        "external_upload_allowed",
        "learner_weight_ancestry_allowed",
        "learner_tokenizer_or_vocabulary_ancestry_allowed",
        "learner_embedding_or_feature_ancestry_allowed",
        "learner_score_or_confidence_ancestry_allowed",
        "primary_evaluation_truth_allowed",
        "simulator_oracle_replacement_allowed",
    ):
        _fixed_bool(boundary[key], False, "E_INSTRUMENT_BOUNDARY")

    serialized = _canonical(root).decode("utf-8")
    if ABSOLUTE_PATH_RE.search(serialized) or RESTRICTED_NAME_RE.search(serialized):
        _fail("E_INSTRUMENT_RECEIPT_PRIVACY")
    return _digest(root)


def seal_instruments(receipts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Freeze a complete instrument set before any restricted processing."""

    if not receipts:
        _fail("E_EMPTY_INSTRUMENT_SET")
    keys: set[str] = set()
    digests: list[str] = []
    profiles: list[str] = []
    for receipt in receipts:
        digest = validate_instrument_receipt(receipt)
        key = str(receipt["instrument_key"])
        if key in keys or digest in digests:
            _fail("E_DUPLICATE_INSTRUMENT")
        keys.add(key)
        digests.append(digest)
        profiles.append(str(receipt["adapter_profile"]))
    body: dict[str, Any] = {
        "schema_version": INSTRUMENT_SEAL_SCHEMA,
        "phase": "RESTRICTED_EXECUTION_NETWORK_DISABLED",
        "all_downloads_completed_before_restricted_processing": True,
        "instrument_count": len(receipts),
        "instrument_receipt_sha256": sorted(digests),
        "adapter_profiles": sorted(profiles),
    }
    body["seal_sha256"] = _digest(body)
    return body


def validate_instrument_seal(
    seal: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> None:
    expected = seal_instruments(receipts)
    if not isinstance(seal, dict) or seal != expected:
        _fail("E_INSTRUMENT_SEAL_MISMATCH")


@dataclass(frozen=True)
class ResourceBudget:
    cpu_workers: int
    mps_heavy_processes: int = 1

    def validate(self) -> None:
        if isinstance(self.cpu_workers, bool) or not (
            MIN_CPU_WORKERS <= self.cpu_workers <= MAX_CPU_WORKERS
        ):
            _fail("E_CPU_WORKER_LIMIT")
        if self.mps_heavy_processes != MAX_MPS_HEAVY_PROCESSES:
            _fail("E_MPS_PROCESS_LIMIT")


@dataclass(frozen=True)
class NetworkIsolationBackend:
    """An immutable, internally generated OS network-isolation prefix."""

    name: str
    prefix: tuple[str, ...]

    @classmethod
    def detect(cls) -> "NetworkIsolationBackend":
        system = platform.system()
        if system == "Darwin" and Path("/usr/bin/sandbox-exec").is_file():
            # ``deny network*`` covers socket creation, bind, listen, connect,
            # inbound, and outbound access for the adapter and descendants.
            profile = "(version 1)(allow default)(deny network*)"
            return cls("MACOS_SANDBOX_DENY_NETWORK", ("/usr/bin/sandbox-exec", "-p", profile))
        if system == "Linux":
            bwrap = shutil.which("bwrap")
            if bwrap:
                return cls(
                    "LINUX_BWRAP_UNSHARE_NET",
                    (
                        str(Path(bwrap).resolve()),
                        "--die-with-parent",
                        "--unshare-net",
                        "--ro-bind",
                        "/",
                        "/",
                    ),
                )
        _fail("E_NETWORK_ISOLATION_UNAVAILABLE")

    def command(
        self, argv: Sequence[str], *, quarantine_root: Path | None = None
    ) -> list[str]:
        if not argv or any(not isinstance(part, str) or not part for part in argv):
            _fail("E_ADAPTER_COMMAND")
        if self.name not in {"MACOS_SANDBOX_DENY_NETWORK", "LINUX_BWRAP_UNSHARE_NET"}:
            _fail("E_NETWORK_ISOLATION_BACKEND")
        if self.name == "MACOS_SANDBOX_DENY_NETWORK":
            base_profile = "(version 1)(allow default)(deny network*)"
            expected = (
                "/usr/bin/sandbox-exec",
                "-p",
                base_profile,
            )
            if self.prefix != expected:
                _fail("E_NETWORK_ISOLATION_BACKEND")
            prefix = self.prefix
            if quarantine_root is not None:
                try:
                    root = str(quarantine_root.resolve(strict=True))
                except OSError:
                    _fail("E_NETWORK_ISOLATION_BACKEND")
                escaped = root.replace("\\", "\\\\").replace('"', '\\"')
                profile = (
                    base_profile
                    + '(deny file-write* (require-not (subpath "'
                    + escaped
                    + '")))'
                )
                prefix = ("/usr/bin/sandbox-exec", "-p", profile)
        elif (
            len(self.prefix) != 6
            or Path(self.prefix[0]).name != "bwrap"
            or self.prefix[1:] != (
                "--die-with-parent",
                "--unshare-net",
                "--ro-bind",
                "/",
                "/",
            )
        ):
            _fail("E_NETWORK_ISOLATION_BACKEND")
        else:
            prefix = self.prefix
            if quarantine_root is not None:
                try:
                    root = str(quarantine_root.resolve(strict=True))
                except OSError:
                    _fail("E_NETWORK_ISOLATION_BACKEND")
                prefix = (*self.prefix, "--bind", root, root)
        return [*prefix, *argv]


_NETWORK_SENTINEL_CODE = r"""
import socket, sys
families = [socket.AF_INET]
if getattr(socket, "has_ipv6", False):
    families.append(socket.AF_INET6)
for family in families:
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        address = ("127.0.0.1", 0) if family == socket.AF_INET else ("::1", 0)
        sock.bind(address)
        sock.listen(1)
    except (OSError, PermissionError):
        continue
    else:
        sock.close()
        raise SystemExit(73)
raise SystemExit(0)
"""


def scrubbed_subprocess_environment(extra: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Return a minimal environment with no inherited credentials or proxies."""

    allowed = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_DISABLED": "true",
        "DO_NOT_TRACK": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    if extra:
        allowed_extra = {
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "CHILDLENS_ADAPTER_CONTRACT",
        }
        for key, value in extra.items():
            if key not in allowed_extra or not isinstance(value, str) or len(value) > 80:
                _fail("E_SUBPROCESS_ENV")
            allowed[key] = value
    return allowed


def verify_network_isolation(
    backend: NetworkIsolationBackend,
    *,
    timeout_seconds: int = 20,
    process_runner: Any = subprocess.run,
) -> None:
    """Actively prove that local IPv4/IPv6 socket use is denied."""

    command = backend.command((sys.executable, "-I", "-c", _NETWORK_SENTINEL_CODE))
    try:
        completed = process_runner(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=scrubbed_subprocess_environment(),
            check=False,
            timeout=timeout_seconds,
            close_fds=True,
        )
    except Exception:
        _fail("E_NETWORK_SENTINEL")
    if completed.returncode != 0:
        _fail("E_NETWORK_SENTINEL")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _owner_private_directory(path: Path) -> bool:
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


def _owner_private_file(path: Path) -> bool:
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


def _resolve_confined(path: Path, root: Path, *, must_exist: bool) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=must_exist)
    except OSError:
        _fail("E_QUARANTINE_CONFINEMENT")
    if not _is_relative_to(resolved, resolved_root) or resolved == resolved_root:
        _fail("E_QUARANTINE_CONFINEMENT")
    cursor = resolved_root
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError:
        _fail("E_QUARANTINE_CONFINEMENT")
    for part in relative.parts:
        cursor /= part
        if cursor.exists():
            try:
                if stat.S_ISLNK(cursor.lstat().st_mode):
                    _fail("E_QUARANTINE_CONFINEMENT")
            except OSError:
                _fail("E_QUARANTINE_CONFINEMENT")
    return resolved


def validate_quarantine_root(root: Path, repository_root: Path) -> Path:
    """Validate an explicitly supplied root; never discover one."""

    if not root.is_absolute() or not repository_root.is_absolute():
        _fail("E_QUARANTINE_CONFINEMENT")
    try:
        resolved = root.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
    except OSError:
        _fail("E_QUARANTINE_CONFINEMENT")
    if resolved == repository or _is_relative_to(resolved, repository):
        _fail("E_QUARANTINE_IN_REPOSITORY")
    if not _owner_private_directory(resolved):
        _fail("E_QUARANTINE_NOT_PRIVATE")
    sentinel = resolved / ".metadata_never_index"
    if not sentinel.is_file() or sentinel.is_symlink():
        _fail("E_QUARANTINE_INDEXING_CONTROL")
    return resolved


@dataclass(frozen=True)
class WorkItem:
    """Opaque work reference retained only in the quarantine checkpoint."""

    item_sha256: str
    media_path: Path


@dataclass(frozen=True)
class AdapterInvocation:
    instrument_key: str
    adapter_profile: str
    executable: Path
    model_artifact: Path
    task: str
    resource_class: str
    extra_flags: tuple[str, ...] = ()


def validate_adapter_invocation(
    invocation: AdapterInvocation,
    receipt: Mapping[str, Any],
    quarantine_root: Path,
) -> None:
    digest = validate_instrument_receipt(receipt)
    del digest
    if invocation.instrument_key != receipt["instrument_key"]:
        _fail("E_ADAPTER_RECEIPT_MISMATCH")
    if invocation.adapter_profile != receipt["adapter_profile"]:
        _fail("E_ADAPTER_RECEIPT_MISMATCH")
    if invocation.task != receipt["task"] or invocation.resource_class != receipt["resource_class"]:
        _fail("E_ADAPTER_RECEIPT_MISMATCH")
    profile = FIXED_ADAPTER_PROFILES[invocation.adapter_profile]
    if tuple(invocation.extra_flags) != tuple(profile["required_adapter_flags"]):
        _fail("E_ADAPTER_FLAGS")
    executable = _resolve_confined(invocation.executable, quarantine_root, must_exist=True)
    model = _resolve_confined(invocation.model_artifact, quarantine_root, must_exist=True)
    if not _owner_private_file(executable) or not os.access(executable, os.X_OK):
        _fail("E_ADAPTER_EXECUTABLE")
    if not (_owner_private_file(model) or _owner_private_directory(model)):
        _fail("E_MODEL_ARTIFACT")
    if _sha256_file(executable) != receipt["software"]["executable_sha256"]:
        _fail("E_ADAPTER_EXECUTABLE_DIGEST")
    if _sha256_artifact(model) != receipt["model"]["artifact_set_sha256"]:
        _fail("E_MODEL_ARTIFACT_DIGEST")


class CheckpointStore:
    """Quarantine-only crash-safe state; it never stores public artifacts."""

    def __init__(self, path: Path, quarantine_root: Path):
        self.root = quarantine_root
        self.path = _resolve_confined(path, quarantine_root, must_exist=False)
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self.connection = sqlite3.connect(self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            self.close()
            _fail("E_CHECKPOINT_PERMISSION")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS work (
                item_sha256 TEXT NOT NULL,
                instrument_sha256 TEXT NOT NULL,
                task TEXT NOT NULL,
                restricted_media_path TEXT NOT NULL,
                output_sha256 TEXT,
                output_bytes INTEGER,
                state TEXT NOT NULL CHECK(state IN ('PENDING','RUNNING','COMPLETE','FAILED')),
                attempts INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (item_sha256, instrument_sha256, task)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.connection.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register(
        self,
        items: Sequence[WorkItem],
        *,
        instrument_sha256: str,
        task: str,
    ) -> None:
        if task not in TASKS or HEX64_RE.fullmatch(instrument_sha256) is None:
            _fail("E_WORK_REGISTRATION")
        seen_digests: set[str] = set()
        seen_inodes: set[tuple[int, int]] = set()
        rows: list[tuple[str, str, str, str]] = []
        for item in items:
            if HEX64_RE.fullmatch(item.item_sha256) is None or item.item_sha256 in seen_digests:
                _fail("E_DUPLICATE_MEDIA")
            media = _resolve_confined(item.media_path, self.root, must_exist=True)
            if not _owner_private_file(media):
                _fail("E_RESTRICTED_MEDIA_CONTROL")
            metadata = media.stat()
            inode = (metadata.st_dev, metadata.st_ino)
            if inode in seen_inodes:
                _fail("E_DUPLICATE_MEDIA")
            if _sha256_file(media) != item.item_sha256:
                _fail("E_RESTRICTED_MEDIA_DIGEST")
            seen_digests.add(item.item_sha256)
            seen_inodes.add(inode)
            rows.append((item.item_sha256, instrument_sha256, task, str(media)))
        try:
            with self.connection:
                for row in rows:
                    self.connection.execute(
                        """
                        INSERT INTO work (
                            item_sha256, instrument_sha256, task,
                            restricted_media_path, state
                        ) VALUES (?, ?, ?, ?, 'PENDING')
                        ON CONFLICT(item_sha256, instrument_sha256, task) DO NOTHING
                        """,
                        row,
                    )
        except sqlite3.Error:
            _fail("E_CHECKPOINT_WRITE")

    def resume_after_interruption(self) -> int:
        try:
            with self.connection:
                cursor = self.connection.execute(
                    "UPDATE work SET state='PENDING' WHERE state='RUNNING'"
                )
            return int(cursor.rowcount)
        except sqlite3.Error:
            _fail("E_CHECKPOINT_WRITE")

    def claim_next(self, instrument_sha256: str, task: str) -> tuple[str, Path] | None:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                """
                SELECT item_sha256, restricted_media_path
                FROM work
                WHERE instrument_sha256=? AND task=? AND state IN ('PENDING','FAILED')
                ORDER BY item_sha256
                LIMIT 1
                """,
                (instrument_sha256, task),
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            self.connection.execute(
                """
                UPDATE work SET state='RUNNING', attempts=attempts+1
                WHERE item_sha256=? AND instrument_sha256=? AND task=?
                """,
                (row[0], instrument_sha256, task),
            )
            self.connection.commit()
        except sqlite3.Error:
            self.connection.rollback()
            _fail("E_CHECKPOINT_WRITE")
        return str(row[0]), Path(str(row[1]))

    def complete(
        self,
        item_sha256: str,
        instrument_sha256: str,
        task: str,
        output_sha256: str,
        output_bytes: int,
    ) -> None:
        if HEX64_RE.fullmatch(output_sha256) is None or not (0 < output_bytes <= MAX_PSEUDO_LABEL_BYTES):
            _fail("E_PSEUDO_LABEL_OUTPUT")
        try:
            with self.connection:
                cursor = self.connection.execute(
                    """
                    UPDATE work
                    SET state='COMPLETE', output_sha256=?, output_bytes=?
                    WHERE item_sha256=? AND instrument_sha256=? AND task=? AND state='RUNNING'
                    """,
                    (output_sha256, output_bytes, item_sha256, instrument_sha256, task),
                )
            if cursor.rowcount != 1:
                _fail("E_CHECKPOINT_STATE")
        except sqlite3.Error:
            _fail("E_CHECKPOINT_WRITE")

    def fail(self, item_sha256: str, instrument_sha256: str, task: str) -> None:
        try:
            with self.connection:
                self.connection.execute(
                    """
                    UPDATE work SET state='FAILED'
                    WHERE item_sha256=? AND instrument_sha256=? AND task=? AND state='RUNNING'
                    """,
                    (item_sha256, instrument_sha256, task),
                )
        except sqlite3.Error:
            _fail("E_CHECKPOINT_WRITE")

    def aggregate_counts(self) -> Mapping[str, int]:
        try:
            rows = self.connection.execute(
                "SELECT state, COUNT(*) FROM work GROUP BY state"
            ).fetchall()
        except sqlite3.Error:
            _fail("E_CHECKPOINT_READ")
        counts = {state: 0 for state in ("PENDING", "RUNNING", "COMPLETE", "FAILED")}
        for state, count in rows:
            counts[str(state)] = int(count)
        return counts

    def aggregate_counts_by_task(self) -> Mapping[str, Mapping[str, int]]:
        try:
            rows = self.connection.execute(
                "SELECT task, state, COUNT(*) FROM work GROUP BY task, state"
            ).fetchall()
        except sqlite3.Error:
            _fail("E_CHECKPOINT_READ")
        result = {
            task: {state: 0 for state in ("PENDING", "RUNNING", "COMPLETE", "FAILED")}
            for task in TASKS
        }
        for task, state, count in rows:
            if task not in TASKS or state not in result[str(task)]:
                _fail("E_CHECKPOINT_READ")
            result[str(task)][str(state)] = int(count)
        return result


@contextlib.contextmanager
def _exclusive_mps(lock_path: Path, quarantine_root: Path) -> Iterator[None]:
    confined = _resolve_confined(lock_path, quarantine_root, must_exist=False)
    confined.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(confined, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _fail("E_MPS_PROCESS_LIMIT")
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class RestrictedInferenceRunner:
    """Run one sealed local adapter with checkpoint/resume and no network."""

    def __init__(
        self,
        *,
        quarantine_root: Path,
        repository_root: Path,
        checkpoint: CheckpointStore,
        seal: Mapping[str, Any],
        receipts: Sequence[Mapping[str, Any]],
        backend: NetworkIsolationBackend,
        resource_budget: ResourceBudget,
        process_runner: Any = subprocess.run,
    ):
        self.root = validate_quarantine_root(quarantine_root, repository_root)
        validate_instrument_seal(seal, receipts)
        resource_budget.validate()
        self.checkpoint = checkpoint
        self.seal = seal
        self.receipts = {str(row["instrument_key"]): row for row in receipts}
        self.backend = backend
        self.budget = resource_budget
        self.process_runner = process_runner

    def _output_path(self, invocation: AdapterInvocation, item_sha256: str) -> Path:
        instrument_digest = validate_instrument_receipt(self.receipts[invocation.instrument_key])
        return (
            self.root
            / "pseudo_annotations_v1_3"
            / invocation.task.casefold()
            / instrument_digest
            / f"{item_sha256}.json"
        )

    def _invoke_one(self, invocation: AdapterInvocation, media_path: Path, output_path: Path) -> None:
        verify_network_isolation(self.backend, process_runner=self.process_runner)
        output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(output_path.parent, 0o700)
        executable_fd = os.open(invocation.executable, os.O_RDONLY)
        media_fd = os.open(media_path, os.O_RDONLY)
        model_flags = os.O_RDONLY
        if invocation.model_artifact.is_dir():
            model_flags |= getattr(os, "O_DIRECTORY", 0)
        model_fd = os.open(invocation.model_artifact, model_flags)
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=".pending-", suffix=".json", dir=output_path.parent
        )
        temporary: Path | None = Path(temporary_name)
        os.chmod(temporary, 0o600)
        try:
            # The adapter executable itself is a generic, receipt-hashed
            # quarantine tool. Restricted media, model artifacts, outputs,
            # object keys, and source filenames remain descriptor-only.
            # Executing scripts through /dev/fd is not portable on macOS, so
            # the generic adapter path is the sole filesystem path in argv.
            adapter_argv = [
                str(invocation.executable),
                "--contract",
                ADAPTER_CONTRACT,
                "--task",
                invocation.task,
                "--media-fd",
                str(media_fd),
                "--model-fd",
                str(model_fd),
                "--output-fd",
                str(temporary_fd),
                *invocation.extra_flags,
            ]
            command = self.backend.command(adapter_argv, quarantine_root=self.root)
            try:
                completed = self.process_runner(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=scrubbed_subprocess_environment(
                        {
                            "OMP_NUM_THREADS": str(self.budget.cpu_workers),
                            "MKL_NUM_THREADS": str(self.budget.cpu_workers),
                            "VECLIB_MAXIMUM_THREADS": str(self.budget.cpu_workers),
                            "CHILDLENS_ADAPTER_CONTRACT": ADAPTER_CONTRACT,
                        }
                    ),
                    pass_fds=(executable_fd, media_fd, model_fd, temporary_fd),
                    close_fds=True,
                    check=False,
                    timeout=3600,
                )
            except Exception:
                _fail("E_OFFLINE_ADAPTER")
            if completed.returncode != 0:
                _fail("E_OFFLINE_ADAPTER")
            os.fsync(temporary_fd)
            size = temporary.stat().st_size
            if size <= 0 or size > MAX_PSEUDO_LABEL_BYTES:
                _fail("E_PSEUDO_LABEL_OUTPUT")
            os.replace(temporary, output_path)
            temporary = None
        finally:
            for descriptor in (temporary_fd, model_fd, media_fd, executable_fd):
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink()

    def run(self, invocation: AdapterInvocation) -> Mapping[str, Any]:
        receipt = self.receipts.get(invocation.instrument_key)
        if receipt is None:
            _fail("E_ADAPTER_RECEIPT_MISMATCH")
        validate_adapter_invocation(invocation, receipt, self.root)
        instrument_digest = validate_instrument_receipt(receipt)
        self.checkpoint.resume_after_interruption()
        parallelism = self.budget.cpu_workers if invocation.resource_class == "CPU" else 1
        while True:
            batch: list[tuple[str, Path]] = []
            for _ in range(parallelism):
                claimed = self.checkpoint.claim_next(instrument_digest, invocation.task)
                if claimed is None:
                    break
                batch.append(claimed)
            if not batch:
                break

            def process(claimed: tuple[str, Path]) -> tuple[str, str, int]:
                try:
                    item_sha256, media_path = claimed
                    output_path = self._output_path(invocation, item_sha256)
                    if invocation.resource_class == "MPS_HEAVY":
                        with _exclusive_mps(
                            self.root / ".locks" / "v1_3_mps_heavy.lock", self.root
                        ):
                            self._invoke_one(invocation, media_path, output_path)
                    else:
                        self._invoke_one(invocation, media_path, output_path)
                    return item_sha256, _sha256_file(output_path), output_path.stat().st_size
                except FirewallError:
                    raise
                except Exception:
                    _fail("E_OFFLINE_ADAPTER")

            first_error: FirewallError | None = None
            results: list[tuple[str, str, int]] = []
            if parallelism == 1:
                try:
                    results.append(process(batch[0]))
                except FirewallError as exc:
                    first_error = exc
            else:
                with ThreadPoolExecutor(
                    max_workers=parallelism,
                    thread_name_prefix="childlens-v13-offline-cpu",
                ) as pool:
                    futures = [(claimed, pool.submit(process, claimed)) for claimed in batch]
                    for claimed, future in futures:
                        try:
                            results.append(future.result())
                        except FirewallError as exc:
                            self.checkpoint.fail(
                                claimed[0], instrument_digest, invocation.task
                            )
                            if first_error is None:
                                first_error = exc
            completed_items = {row[0] for row in results}
            for item_sha256, output_digest, output_size in results:
                self.checkpoint.complete(
                    item_sha256,
                    instrument_digest,
                    invocation.task,
                    output_digest,
                    output_size,
                )
            if first_error is not None:
                for item_sha256, _ in batch:
                    if item_sha256 not in completed_items:
                        self.checkpoint.fail(item_sha256, instrument_digest, invocation.task)
                raise first_error
        return build_aggregate_receipt(
            self.checkpoint.aggregate_counts(),
            task_counts=self.checkpoint.aggregate_counts_by_task(),
            adapter_profiles=tuple(str(row["adapter_profile"]) for row in self.receipts.values()),
            isolation_backend=self.backend.name,
            cpu_workers=self.budget.cpu_workers,
        )


def build_aggregate_receipt(
    counts: Mapping[str, int],
    *,
    task_counts: Mapping[str, Mapping[str, int]],
    adapter_profiles: Sequence[str],
    isolation_backend: str,
    cpu_workers: int,
) -> Mapping[str, Any]:
    """Build the only allowlisted public result of this firewall.

    The function receives state counts, never pseudo-label payloads.  It emits
    neither task-level cells nor item-level records.
    """

    expected_states = frozenset({"PENDING", "RUNNING", "COMPLETE", "FAILED"})
    if frozenset(counts) != expected_states:
        _fail("E_AGGREGATE_SCHEMA")
    clean: dict[str, int] = {}
    for key in expected_states:
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail("E_AGGREGATE_SCHEMA")
        clean[key.casefold()] = value
    total = sum(clean.values())
    if not (MIN_CPU_WORKERS <= cpu_workers <= MAX_CPU_WORKERS):
        _fail("E_AGGREGATE_SCHEMA")
    if frozenset(task_counts) != TASKS:
        _fail("E_AGGREGATE_SCHEMA")
    normalized_task_counts: dict[str, dict[str, int]] = {}
    for task in TASKS:
        states = task_counts[task]
        if frozenset(states) != expected_states:
            _fail("E_AGGREGATE_SCHEMA")
        normalized_task_counts[task] = {}
        for state in expected_states:
            value = states[state]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail("E_AGGREGATE_SCHEMA")
            normalized_task_counts[task][state] = value
    profile_set = frozenset(adapter_profiles)
    if len(adapter_profiles) != len(profile_set) or not profile_set.issubset(FIXED_ADAPTER_PROFILES):
        _fail("E_AGGREGATE_SCHEMA")
    full_profile_set = profile_set == frozenset(FIXED_ADAPTER_PROFILES)
    full_task_coverage = all(
        normalized_task_counts[task]["COMPLETE"] == FROZEN_SELECTED_ITEM_COUNT
        and sum(normalized_task_counts[task].values()) == FROZEN_SELECTED_ITEM_COUNT
        for task in TASKS
    )
    checkpoint_totals_match = all(
        clean[state.casefold()]
        == sum(normalized_task_counts[task][state] for task in TASKS)
        for state in expected_states
    )
    if not checkpoint_totals_match:
        _fail("E_AGGREGATE_SCHEMA")
    if isolation_backend not in {
        "MACOS_SANDBOX_DENY_NETWORK",
        "LINUX_BWRAP_UNSHARE_NET",
    }:
        _fail("E_AGGREGATE_SCHEMA")
    receipt: dict[str, Any] = {
        "schema_version": AGGREGATE_RECEIPT_SCHEMA,
        "status": (
            "COMPLETE"
            if total > 0
            and clean["complete"] == total
            and full_profile_set
            and full_task_coverage
            else "INCOMPLETE"
        ),
        "scope": "AGGREGATE_ONLY_PSEUDO_LABELS_NEVER_GROUND_TRUTH",
        "candidate_window_count": FULL_PILOT_WINDOW_COUNT,
        "candidate_speech_minutes": FULL_PILOT_SPEECH_MINUTES,
        "maximum_cpu_workers": cpu_workers,
        "maximum_mps_heavy_processes": MAX_MPS_HEAVY_PROCESSES,
        # These exact top-level fields are intentionally duplicated from the
        # structured blocks below so the terminal synthesizer can fail closed
        # without interpreting nested or free-form content.
        "local_offline_only": True,
        "network_disabled_during_restricted_inference": True,
        "inference_subprocess_network_blocked": True,
        "no_hosted_or_cloud_content_path": True,
        "quarantine_only": True,
        "external_api_used": False,
        "external_upload": False,
        "telemetry_enabled": False,
        "restricted_data_egress": False,
        "fixed_versioned_instruments": True,
        "all_instrument_licenses_audited": True,
        "model_hashes_verified": True,
        "model_downloads_completed_before_restricted_processing": True,
        "checkpoint_resume_enabled": True,
        "pseudo_labels_marked_not_ground_truth": True,
        "author_audit_only_human_labeled_childlens_evidence": True,
        "simulator_oracle_labels_primary_evaluation_truth": True,
        "aggregate_safe_receipt_only": True,
        "instrument_weights_to_learner": False,
        "instrument_tokenizers_or_vocabularies_to_learner": False,
        "instrument_features_or_embeddings_to_learner": False,
        "instrument_scores_or_confidences_to_learner": False,
        "pseudo_labels_primary_evaluation_truth": False,
        "scientific_learner_trained": False,
        "causal_outcome_run": False,
        "instrument_embeddings_entered_learner": False,
        "instrument_features_entered_learner": False,
        "instrument_weights_entered_learner": False,
        "instrument_tokenizers_entered_learner": False,
        "instrument_vocabularies_entered_learner": False,
        "instrument_scores_entered_learner": False,
        "unaudited_pseudo_labels_primary_evaluation_truth": False,
        "learner_training_executed": False,
        "corpus_tokenizer_trained": False,
        "causal_arm_executed": False,
        "scientific_acquisition_outcome_executed": False,
        "progress": {
            "work_units_total": total,
            "work_units_complete": clean["complete"],
            "work_units_failed": clean["failed"],
            "work_units_pending": clean["pending"] + clean["running"],
            "item_or_task_cells_exported": False,
            "cell_suppression_k": CELL_SUPPRESSION_K,
        },
        "instrument_set": {
            "fixed_instrument_count": len(profile_set),
            "all_downloads_completed_before_restricted_processing": True,
            "licenses_and_hashes_verified_before_processing": True,
        },
        "security": {
            "network_disabled_during_restricted_inference": True,
            "no_hosted_or_cloud_content_path": True,
            "external_api_used": False,
            "external_upload": False,
            "telemetry_enabled": False,
            "inference_subprocess_network_blocked": True,
            "quarantine_only": True,
            "isolation_backend": isolation_backend,
            "subprocess_environment_scrubbed": True,
            "subprocess_filesystem_write_confined": True,
        },
        "resource_policy": {
            "cpu_workers": cpu_workers,
            "cpu_worker_range_enforced": [MIN_CPU_WORKERS, MAX_CPU_WORKERS],
            "mps_heavy_process_limit": MAX_MPS_HEAVY_PROCESSES,
            "checkpoint_resume_enabled": True,
            "duplicate_media_rejected": True,
        },
        "scientific_boundary": {
            "pseudo_labels_are_ground_truth": False,
            "author_audit_is_only_human_labeled_evidence": True,
            "primary_evaluation_truth_allowed": False,
            "simulator_oracle_required_for_later_primary_evaluation": True,
            "learner_weight_ancestry": False,
            "learner_tokenizer_or_vocabulary_ancestry": False,
            "learner_embedding_or_feature_ancestry": False,
            "learner_score_or_confidence_ancestry": False,
            "scientific_learner_or_causal_outcome_run": False,
        },
        "privacy_export": {
            "paths": False,
            "identifiers": False,
            "filenames": False,
            "exact_timestamps": False,
            "transcript_or_lexical_content": False,
            "frames_or_model_payloads": False,
        },
    }
    encoded = _canonical(receipt).decode("utf-8")
    if ABSOLUTE_PATH_RE.search(encoded) or RESTRICTED_NAME_RE.search(encoded):
        # Field names in ``privacy_export`` are fixed public assertions; no
        # values may contain restricted-name-shaped free text.
        values_only = json.dumps(list(_walk_string_values(receipt)), sort_keys=True)
        if ABSOLUTE_PATH_RE.search(values_only) or RESTRICTED_NAME_RE.search(values_only):
            _fail("E_AGGREGATE_PRIVACY")
    return receipt


def _walk_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_string_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_string_values(child)


def main(argv: Sequence[str] | None = None) -> int:
    """No operational CLI: a quarantine coordinator must import this module."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        print(json.dumps({"schema_version": VERSION, "status": "error", "error_code": "E_ARGUMENTS"}))
        return 2
    print(
        json.dumps(
            {
                "schema_version": VERSION,
                "status": "LIBRARY_READY_NO_CONTENT_ACCESSED",
                "fixed_adapter_profiles": sorted(FIXED_ADAPTER_PROFILES),
                "operational_cli_enabled": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
