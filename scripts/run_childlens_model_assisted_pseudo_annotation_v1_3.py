#!/usr/bin/env python3
"""Zero-argument ChildLens v1.3 local pseudo-annotation runner.

The public phase validates and seals the fixed, permissively licensed local
instrument cache without opening ChildLens data.  It then re-executes the
restricted phase under an OS-level network-denial backend.  The restricted
phase discovers the already-approved owner-private runtime internally, checks
the frozen v1.2 manifest, and writes only quarantine-local pseudo-labels and a
checkpoint database.  A strict aggregate receipt crosses the process boundary;
no item row, path, timestamp, transcript, frame, identifier, or model payload
does.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
ADAPTER_PATH = REPO_ROOT / "scripts/childlens_local_pseudo_adapter_v1_3.py"
FIREWALL_PATH = REPO_ROOT / "scripts/childlens_local_inference_firewall_v1_3.py"
V1_2_WORKFLOW_PATH = REPO_ROOT / "scripts/childlens_human_validation_v1_2.py"
PUBLIC_RECEIPT_PATH = REPO_ROOT / "output/childlens_feasibility_v1_3/pseudo_annotation_receipt.json"
LICENSE_EVIDENCE_PATH = REPO_ROOT / "output/childlens_feasibility_v1_3/local_instrument_provenance.json"
INSTRUMENT_CACHE = Path("/Users/rishisim/Library/Application Support/ChildLens Instruments/v1.3")
VENV_PYTHON = INSTRUMENT_CACHE / "venv/bin/python3.10"
SITE_PACKAGES = INSTRUMENT_CACHE / "venv/lib/python3.10/site-packages"
SILERO_MODEL = INSTRUMENT_CACHE / "models/silero-vad-6.2.1/silero_vad.onnx"
SILERO_RUNTIME_MODEL = SITE_PACKAGES / "silero_vad/data/silero_vad.onnx"
WHISPER_SOURCE = INSTRUMENT_CACHE / ".worktrees/whisper.cpp-v1.9.1"
WHISPER_CLI = WHISPER_SOURCE / "build/bin/whisper-cli"
WHISPER_MODEL = INSTRUMENT_CACHE / "models/ggml-large-v3-turbo.bin"
QWEN_MODEL = INSTRUMENT_CACHE / "models/Qwen2-VL-2B-Instruct"
MANIFEST_ROOT = INSTRUMENT_CACHE / "manifests"
QWEN_MANIFEST = MANIFEST_ROOT / "qwen2_vl_2b_instruct_public_cache_manifest_v1_3.json"
SILERO_MANIFEST = MANIFEST_ROOT / "silero_vad_6_2_1_public_cache_manifest_v1_3.json"
WHISPER_BUNDLE_MANIFEST = MANIFEST_ROOT / "whisper_cpp_v1_9_1_build_bundle_manifest_v1_3.json"
FFMPEG = Path("/opt/homebrew/bin/ffmpeg").resolve()

VERSION = "childlens-model-assisted-pseudo-runner-v1.3.0"
MEASUREMENT_SCHEMA = "childlens-restricted-measurement-manifest-v1.2.1"
SEAL_ENVELOPE_SCHEMA = "childlens-v1.3-restricted-run-seal-v1"
DIAGNOSTIC_SCHEMA = "childlens-v1.3-restricted-diagnostic-v1"
FROZEN_SELECTION_DIGEST = "61526ea6cebc256314b0b9e574fb1a5986fcd83a534979e93134c4163c55253f"
EXPECTED_ITEMS = 15
EXPECTED_WINDOWS = 912
EXPECTED_SPEECH_MINUTES = 135.25
CPU_WORKERS = 2
GIB = 1024**3
NAMESPACE_CAP_BYTES = 73 * GIB
FREE_SPACE_FLOOR_BYTES = 50 * GIB
FIXED_SCRATCH_RESERVE_BYTES = 2 * GIB
MAX_RESULT_BYTES = 128 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_ITEM_OUTPUT_BYTES = 64 * 1024 * 1024
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
CONTENT_PATH_RE = re.compile(r"^(?:[A-Za-z0-9_.-]+/)+[0-9a-f]{64}\.[A-Za-z0-9]{2,8}$")
FORBIDDEN_SELECTION_KEY_RE = re.compile(
    r"(?i)(?:model_prediction|model_score|confidence|transcript|lexical|visual_salience|apparent_success)"
)

EXPECTED_HASHES = {
    "silero": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    "whisper": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    "whisper_cli": "9613b31e5380c184ae29ccb1d4046953d7037e8eb55308c9f1a34f145143b892",
    "qwen_shard_1": "994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9",
    "qwen_shard_2": "92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833",
}
EXPECTED_PUBLIC_MANIFESTS = {
    "qwen": {
        "file": "5254d4565d12097c9049b1dceb12b306c2d716e18894b0d0143825e29c7f10c7",
        "aggregate": "56f8306e4799e9cd9e5222831d5e856f9616a508f7b9761cd2de2ddd05eceb05",
        "entries": 40,
    },
    "silero": {
        "file": "92fa22b784239596692405eb6e061c506fbc5677739c6574836c290456e0010e",
        "aggregate": "68cb8972c6439d6c77ca79053b4d2ec222fe1fbcac4a214b4d7a1c96ea4990c0",
        "entries": 2,
    },
    "whisper_bundle": {
        "file": "f69466209b4d1c6f9e252fd115b3f4f40684b6e83b3703e47582a1104056f82d",
        "aggregate": "9b13133224ff0e86ca48b3ab1138864080d9034718650d564eb91e665c2ee252",
        "entries": 19,
    },
}
WHISPER_SOURCE_REVISION = "f049fff95a089aa9969deb009cdd4892b3e74916"
WHISPER_MODEL_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
QWEN_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"

QWEN_REQUIRED_FILES = (
    "LICENSE",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
REQUIRED_PACKAGE_VERSIONS = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.53.3",
    "silero-vad": "6.2.1",
    "onnxruntime": "1.23.2",
}

ALLOWED_DIAGNOSTIC_CODES = frozenset(
    {
        "E_ARGUMENTS",
        "E_CANONICAL_JSON",
        "E_FAIL_CLOSED",
        "E_INSTRUMENT_CACHE",
        "E_INSTRUMENT_RUNTIME",
        "E_INSTRUMENT_RUNTIME_VERSION",
        "E_INSTRUMENT_SEAL_MISMATCH",
        "E_PUBLIC_CACHE_MANIFEST",
        "E_SILERO_HASH",
        "E_WHISPER_HASH",
        "E_WHISPER_EXECUTABLE_HASH",
        "E_WHISPER_SOURCE_REVISION",
        "E_QWEN_HASH",
        "E_NETWORK_ISOLATION_UNAVAILABLE",
        "E_NETWORK_ISOLATION_BACKEND",
        "E_NETWORK_SENTINEL",
        "E_RUNTIME_ROOT",
        "E_QUARANTINE_CONFINEMENT",
        "E_QUARANTINE_IN_REPOSITORY",
        "E_QUARANTINE_NOT_PRIVATE",
        "E_QUARANTINE_INDEXING_CONTROL",
        "E_STORAGE_AUDIT",
        "E_NAMESPACE_CAP",
        "E_FREE_SPACE_FLOOR",
        "E_RESTRICTED_CONTROL",
        "E_RESTRICTED_INPUT",
        "E_RESTRICTED_MEDIA",
        "E_FROZEN_MANIFEST",
        "E_FROZEN_WINDOWS",
        "E_MODEL_DEPENDENT_SELECTION",
        "E_CHECKPOINT_PERMISSION",
        "E_CHECKPOINT_READ",
        "E_CHECKPOINT_WRITE",
        "E_CHECKPOINT_STATE",
        "E_LOCAL_INSTRUMENT",
        "E_ADAPTER_MODE",
        "E_PSEUDO_OUTPUT",
        "E_AUDIO_OUTPUT_SCHEMA",
        "E_VLM_OUTPUT_SCHEMA",
        "E_AGGREGATE_SCHEMA",
        "E_AGGREGATE_PRIVACY",
        "E_PUBLIC_RECEIPT",
        "E_PUBLIC_RECEIPT_PRIVACY",
    }
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("E_LOCAL_MODULE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


firewall = _load_module(FIREWALL_PATH, "childlens_local_inference_firewall_v1_3_for_runner")


class RunnerError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise RunnerError(code)


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
        _fail("E_INSTRUMENT_CACHE")
    return digest.hexdigest()


def _private_regular(path: Path, *, executable: bool = False) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISREG(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and (not executable or os.access(path, os.X_OK))
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _tree_digest(root: Path, relative_files: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_files):
        path = root / relative
        if not _private_regular(path):
            _fail("E_INSTRUMENT_CACHE")
        digest.update(relative.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _verify_public_manifest(
    manifest_path: Path,
    artifact_root: Path,
    *,
    expected_file_sha256: str,
    expected_aggregate_sha256: str,
    expected_entry_count: int,
) -> str:
    if not _private_regular(manifest_path) or _sha256_file(manifest_path) != expected_file_sha256:
        _fail("E_PUBLIC_CACHE_MANIFEST")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        _fail("E_PUBLIC_CACHE_MANIFEST")
    entries = document.get("entries") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("schema") != "childlens-public-model-cache-manifest-v1.3.0"
        or document.get("aggregate_scheme")
        != "sha256(kind_nul_relpath_nul_size_nul_file_sha256_lf)"
        or document.get("aggregate_sha256") != expected_aggregate_sha256
        or document.get("entry_count") != expected_entry_count
        or not isinstance(entries, list)
        or len(entries) != expected_entry_count
    ):
        _fail("E_PUBLIC_CACHE_MANIFEST")
    aggregate = hashlib.sha256()
    declared: set[str] = set()
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"kind", "relative_path", "sha256", "size_bytes"}:
            _fail("E_PUBLIC_CACHE_MANIFEST")
        kind = row["kind"]
        relative = row["relative_path"]
        size = row["size_bytes"]
        digest = row["sha256"]
        if (
            kind not in {"file", "symlink"}
            or not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in declared
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or HEX64_RE.fullmatch(digest) is None
        ):
            _fail("E_PUBLIC_CACHE_MANIFEST")
        path = artifact_root / relative
        try:
            metadata = path.lstat()
        except OSError:
            _fail("E_PUBLIC_CACHE_MANIFEST")
        if kind == "file":
            if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                _fail("E_PUBLIC_CACHE_MANIFEST")
            observed_size = metadata.st_size
            observed_digest = _sha256_file(path)
        else:
            if not stat.S_ISLNK(metadata.st_mode):
                _fail("E_PUBLIC_CACHE_MANIFEST")
            target = os.readlink(path).encode("utf-8")
            if Path(target.decode("utf-8")).is_absolute() or ".." in Path(target.decode("utf-8")).parts:
                _fail("E_PUBLIC_CACHE_MANIFEST")
            observed_size = len(target)
            observed_digest = hashlib.sha256(target).hexdigest()
        if observed_size != size or observed_digest != digest:
            _fail("E_PUBLIC_CACHE_MANIFEST")
        aggregate.update(
            (b"F" if kind == "file" else b"L")
            + b"\0"
            + relative.encode("utf-8")
            + b"\0"
            + str(size).encode("ascii")
            + b"\0"
            + digest.encode("ascii")
            + b"\n"
        )
        declared.add(relative)
    observed = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed != declared or aggregate.hexdigest() != expected_aggregate_sha256:
        _fail("E_PUBLIC_CACHE_MANIFEST")
    return expected_aggregate_sha256


def _installed_package_versions(site_packages: Path = SITE_PACKAGES) -> Mapping[str, str]:
    try:
        distributions = importlib.metadata.distributions(path=[str(site_packages)])
        result = {
            str(distribution.metadata["Name"]).casefold(): distribution.version
            for distribution in distributions
            if distribution.metadata.get("Name")
        }
    except Exception:
        _fail("E_INSTRUMENT_RUNTIME")
    return result


def _git_head(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", str(path), "rev-parse", "HEAD"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            close_fds=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except Exception:
        _fail("E_INSTRUMENT_CACHE")
    value = completed.stdout.decode("ascii", errors="ignore").strip()
    if completed.returncode != 0 or value != WHISPER_SOURCE_REVISION:
        _fail("E_WHISPER_SOURCE_REVISION")
    return value


def _instrument_receipt(
    *,
    profile: str,
    instrument_key: str,
    software_hash: str,
    executable_hash: str,
    model_name: str,
    model_version: str,
    model_hash: str,
    software_license: str,
    model_license: str,
    license_hash: str,
) -> Mapping[str, Any]:
    fixed = firewall.FIXED_ADAPTER_PROFILES[profile]
    return {
        "schema_version": firewall.INSTRUMENT_RECEIPT_SCHEMA,
        "instrument_key": instrument_key,
        "adapter_profile": profile,
        "task": fixed["task"],
        "resource_class": fixed["resource_class"],
        "software": {
            "name": "childlens-local-pseudo-adapter",
            "version": "1.3.0",
            "code_sha256": software_hash,
            "executable_sha256": executable_hash,
            "adapter_contract_sha256": firewall.ADAPTER_CONTRACT_SHA256,
            "license_identifier": software_license,
            "license_evidence_sha256": license_hash,
        },
        "model": {
            "name": model_name,
            "version": model_version,
            "artifact_set_sha256": model_hash,
            "license_identifier": model_license,
            "license_evidence_sha256": license_hash,
            "authoritative_source_verified": True,
        },
        "acquisition": {
            "completed_before_restricted_processing": True,
            "hashes_verified": True,
            "license_review_status": "PASS",
            "clickthrough_accepted_by_automation": False,
            "telemetry_disabled": True,
        },
        "boundary": {
            "offline_only": True,
            "external_api_allowed": False,
            "external_upload_allowed": False,
            "network_isolation_required": True,
            "restricted_outputs_quarantine_only": True,
            "learner_weight_ancestry_allowed": False,
            "learner_tokenizer_or_vocabulary_ancestry_allowed": False,
            "learner_embedding_or_feature_ancestry_allowed": False,
            "learner_score_or_confidence_ancestry_allowed": False,
            "primary_evaluation_truth_allowed": False,
            "simulator_oracle_replacement_allowed": False,
        },
    }


def prepare_instrument_seal() -> Mapping[str, Any]:
    """Validate exact public artifacts before any restricted runtime discovery."""

    required_regular = (
        ADAPTER_PATH,
        LICENSE_EVIDENCE_PATH,
        SILERO_MODEL,
        SILERO_RUNTIME_MODEL,
        WHISPER_CLI,
        WHISPER_MODEL,
        FFMPEG,
    )
    if not all(_private_regular(path, executable=path in (WHISPER_CLI, FFMPEG)) for path in required_regular):
        _fail("E_INSTRUMENT_CACHE")
    try:
        python_resolved = VENV_PYTHON.resolve(strict=True)
    except OSError:
        _fail("E_INSTRUMENT_RUNTIME")
    if not python_resolved.is_file() or not os.access(python_resolved, os.X_OK):
        _fail("E_INSTRUMENT_RUNTIME")
    versions = _installed_package_versions()
    if any(versions.get(name) != version for name, version in REQUIRED_PACKAGE_VERSIONS.items()):
        _fail("E_INSTRUMENT_RUNTIME_VERSION")
    _git_head(WHISPER_SOURCE)
    if _sha256_file(SILERO_MODEL) != EXPECTED_HASHES["silero"]:
        _fail("E_SILERO_HASH")
    if _sha256_file(SILERO_RUNTIME_MODEL) != EXPECTED_HASHES["silero"]:
        _fail("E_SILERO_HASH")
    if _sha256_file(WHISPER_CLI) != EXPECTED_HASHES["whisper_cli"]:
        _fail("E_WHISPER_EXECUTABLE_HASH")
    if _sha256_file(WHISPER_MODEL) != EXPECTED_HASHES["whisper"]:
        _fail("E_WHISPER_HASH")
    if _sha256_file(QWEN_MODEL / QWEN_REQUIRED_FILES[5]) != EXPECTED_HASHES["qwen_shard_1"]:
        _fail("E_QWEN_HASH")
    if _sha256_file(QWEN_MODEL / QWEN_REQUIRED_FILES[6]) != EXPECTED_HASHES["qwen_shard_2"]:
        _fail("E_QWEN_HASH")
    qwen_hash = _verify_public_manifest(
        QWEN_MANIFEST,
        QWEN_MODEL,
        expected_file_sha256=EXPECTED_PUBLIC_MANIFESTS["qwen"]["file"],
        expected_aggregate_sha256=EXPECTED_PUBLIC_MANIFESTS["qwen"]["aggregate"],
        expected_entry_count=EXPECTED_PUBLIC_MANIFESTS["qwen"]["entries"],
    )
    silero_manifest_hash = _verify_public_manifest(
        SILERO_MANIFEST,
        SILERO_MODEL.parent,
        expected_file_sha256=EXPECTED_PUBLIC_MANIFESTS["silero"]["file"],
        expected_aggregate_sha256=EXPECTED_PUBLIC_MANIFESTS["silero"]["aggregate"],
        expected_entry_count=EXPECTED_PUBLIC_MANIFESTS["silero"]["entries"],
    )
    whisper_bundle_hash = _verify_public_manifest(
        WHISPER_BUNDLE_MANIFEST,
        WHISPER_CLI.parent,
        expected_file_sha256=EXPECTED_PUBLIC_MANIFESTS["whisper_bundle"]["file"],
        expected_aggregate_sha256=EXPECTED_PUBLIC_MANIFESTS["whisper_bundle"]["aggregate"],
        expected_entry_count=EXPECTED_PUBLIC_MANIFESTS["whisper_bundle"]["entries"],
    )
    adapter_hash = _sha256_file(ADAPTER_PATH)
    python_hash = _sha256_file(python_resolved)
    license_hash = _sha256_file(LICENSE_EVIDENCE_PATH)
    role_hash = hashlib.sha256(
        b"childlens-role-rules-v1.3.0:UNCERTAIN-without-semantic-speaker-anchor"
    ).hexdigest()
    receipts = [
        _instrument_receipt(
            profile="silero_vad",
            instrument_key="silero-vad-fixed",
            software_hash=adapter_hash,
            executable_hash=python_hash,
            model_name="Silero VAD",
            model_version="6.2.1",
            model_hash=EXPECTED_HASHES["silero"],
            software_license="MIT",
            model_license="MIT",
            license_hash=license_hash,
        ),
        _instrument_receipt(
            profile="whisper_cpp_large_v3_turbo",
            instrument_key="whisper-cpp-fixed",
            software_hash=adapter_hash,
            executable_hash=python_hash,
            model_name="whisper.cpp ggml-large-v3-turbo.bin",
            model_version=WHISPER_MODEL_REVISION,
            model_hash=EXPECTED_HASHES["whisper"],
            software_license="MIT",
            model_license="MIT",
            license_hash=license_hash,
        ),
        _instrument_receipt(
            profile="conservative_role_aid",
            instrument_key="conservative-role-aid",
            software_hash=adapter_hash,
            executable_hash=python_hash,
            model_name="deterministic conservative acoustic role rules",
            model_version="childlens-role-rules-v1.3.0",
            model_hash=role_hash,
            software_license="Internal-policy",
            model_license="Internal-policy",
            license_hash=license_hash,
        ),
        _instrument_receipt(
            profile="qwen2_vl_2b_instruct",
            instrument_key="qwen2-vl-fixed",
            software_hash=adapter_hash,
            executable_hash=python_hash,
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            model_version=QWEN_REVISION,
            model_hash=qwen_hash,
            software_license="Apache-2.0",
            model_license="Apache-2.0",
            license_hash=license_hash,
        ),
    ]
    seal = firewall.seal_instruments(receipts)
    return {
        "schema_version": SEAL_ENVELOPE_SCHEMA,
        "runner_version": VERSION,
        "adapter_sha256": adapter_hash,
        "venv_python_sha256": python_hash,
        "ffmpeg_sha256": _sha256_file(FFMPEG),
        "qwen_required_tree_sha256": qwen_hash,
        "silero_public_cache_aggregate_sha256": silero_manifest_hash,
        "whisper_build_bundle_aggregate_sha256": whisper_bundle_hash,
        "receipts": receipts,
        "seal": seal,
    }


def validate_instrument_seal(envelope: Mapping[str, Any]) -> None:
    expected = prepare_instrument_seal()
    if envelope != expected:
        _fail("E_INSTRUMENT_SEAL_MISMATCH")
    firewall.validate_instrument_seal(envelope["seal"], envelope["receipts"])


def _read_fd_json(descriptor: int, maximum: int) -> Any:
    try:
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            payload = handle.read(maximum + 1)
        if not payload or len(payload) > maximum:
            _fail("E_RESTRICTED_CONTROL")
        return json.loads(payload)
    except RunnerError:
        raise
    except Exception:
        _fail("E_RESTRICTED_CONTROL")


def _write_fd_json(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    if len(payload) > MAX_RESULT_BYTES:
        _fail("E_PUBLIC_RECEIPT")
    try:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            handle.write(payload)
            handle.flush()
            # ``result_fd`` is normally a pipe to the network-isolated parent.
            # fsync(2) is invalid for pipes on macOS; flushing is sufficient
            # because the parent reads only after the child closes its copy.
            try:
                metadata = os.fstat(handle.fileno())
            except OSError:
                _fail("E_PUBLIC_RECEIPT")
            if stat.S_ISREG(metadata.st_mode):
                os.fsync(handle.fileno())
    except OSError:
        _fail("E_PUBLIC_RECEIPT")


def _diagnostic_stage(code: str) -> str:
    if code.startswith(("E_INSTRUMENT", "E_PUBLIC_CACHE", "E_SILERO", "E_WHISPER", "E_QWEN")):
        return "INSTRUMENT_PREFLIGHT"
    if code.startswith("E_NETWORK"):
        return "NETWORK_ISOLATION"
    if code.startswith(("E_RUNTIME", "E_QUARANTINE")):
        return "RUNTIME_DISCOVERY"
    if code.startswith(("E_STORAGE", "E_NAMESPACE", "E_FREE_SPACE")):
        return "STORAGE_GUARD"
    if code.startswith(("E_RESTRICTED", "E_FROZEN", "E_MODEL_DEPENDENT")):
        return "FROZEN_MANIFEST"
    if code.startswith("E_CHECKPOINT"):
        return "CHECKPOINT"
    if code.startswith(("E_LOCAL_INSTRUMENT", "E_ADAPTER", "E_PSEUDO", "E_AUDIO", "E_VLM")):
        return "LOCAL_INFERENCE"
    if code.startswith("E_AGGREGATE"):
        return "AGGREGATE_EXPORT"
    return "FAIL_CLOSED"


def _diagnostic_envelope(error: BaseException) -> Mapping[str, Any]:
    observed = getattr(error, "code", "E_FAIL_CLOSED")
    code = observed if isinstance(observed, str) and observed in ALLOWED_DIAGNOSTIC_CODES else "E_FAIL_CLOSED"
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "status": "FAILED",
        "code": code,
        "stage": _diagnostic_stage(code),
        "restricted_payload_exported": False,
        "paths_or_item_rows_exported": False,
    }


def _validate_diagnostic(value: Any) -> Mapping[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "status",
            "code",
            "stage",
            "restricted_payload_exported",
            "paths_or_item_rows_exported",
        }
        or value.get("schema_version") != DIAGNOSTIC_SCHEMA
        or value.get("status") != "FAILED"
        or value.get("code") not in ALLOWED_DIAGNOSTIC_CODES
        or value.get("stage") != _diagnostic_stage(str(value.get("code")))
        or value.get("restricted_payload_exported") is not False
        or value.get("paths_or_item_rows_exported") is not False
    ):
        _fail("E_PUBLIC_RECEIPT")
    return value


def _incomplete_receipt_from_diagnostic(
    diagnostic: Mapping[str, Any], *, isolation_backend: str
) -> Mapping[str, Any]:
    checked = _validate_diagnostic(diagnostic)
    counts = {"PENDING": EXPECTED_ITEMS * len(firewall.TASKS), "RUNNING": 0, "COMPLETE": 0, "FAILED": 0}
    task_counts = {
        task: {"PENDING": EXPECTED_ITEMS, "RUNNING": 0, "COMPLETE": 0, "FAILED": 0}
        for task in firewall.TASKS
    }
    receipt = dict(
        firewall.build_aggregate_receipt(
            counts,
            task_counts=task_counts,
            adapter_profiles=tuple(firewall.FIXED_ADAPTER_PROFILES),
            isolation_backend=isolation_backend,
            cpu_workers=CPU_WORKERS,
        )
    )
    receipt["storage_policy"] = {
        "namespace_cap_gib": 73,
        "free_space_floor_gib": 50,
        "preflight_projection_enforced": True,
        "during_process_monitoring_enforced": True,
        "post_item_cleanup_recheck_enforced": True,
    }
    receipt["diagnostic"] = {
        "code": checked["code"],
        "stage": checked["stage"],
        "progress_basis": "CONSERVATIVE_ZERO_COMPLETE_DIAGNOSTIC_FALLBACK",
        "restricted_payload_exported": False,
        "paths_or_item_rows_exported": False,
    }
    return receipt


def _read_private_json(path: Path, root: Path, maximum: int) -> Any:
    try:
        resolved = path.resolve(strict=True)
        safe_root = root.resolve(strict=True)
    except OSError:
        _fail("E_RESTRICTED_INPUT")
    if not _inside(resolved, safe_root) or not _private_regular(resolved):
        _fail("E_RESTRICTED_INPUT")
    try:
        payload = resolved.read_bytes()
        if not payload or len(payload) > maximum:
            _fail("E_RESTRICTED_INPUT")
        return json.loads(payload)
    except RunnerError:
        raise
    except Exception:
        _fail("E_RESTRICTED_INPUT")


def _secure_directory(path: Path, root: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail("E_QUARANTINE_CONFINEMENT")
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            _fail("E_QUARANTINE_CONFINEMENT")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    resolved = path.resolve(strict=True)
    if not _inside(resolved, root) or path.is_symlink():
        _fail("E_QUARANTINE_CONFINEMENT")
    return resolved


@dataclass(frozen=True)
class PilotItem:
    digest: str
    media: Path
    windows: tuple[tuple[float, float], ...]
    duration_seconds: float


def validate_measurement_manifest(document: Any, root: Path) -> tuple[PilotItem, ...]:
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != MEASUREMENT_SCHEMA
        or document.get("pilot_selection_sha256") != FROZEN_SELECTION_DIGEST
        or not isinstance(document.get("items"), list)
        or len(document["items"]) != EXPECTED_ITEMS
    ):
        _fail("E_FROZEN_MANIFEST")
    items: list[PilotItem] = []
    digests: set[str] = set()
    inodes: set[tuple[int, int]] = set()
    window_count = 0
    seconds = 0.0
    for row in document["items"]:
        if not isinstance(row, dict):
            _fail("E_FROZEN_MANIFEST")
        if any(FORBIDDEN_SELECTION_KEY_RE.search(str(key)) for key in row):
            _fail("E_MODEL_DEPENDENT_SELECTION")
        digest = row.get("expected_media_sha256")
        relative = row.get("media_relpath")
        windows_value = row.get("speech_windows")
        duration = row.get("reference_duration_seconds")
        if (
            not isinstance(digest, str)
            or HEX64_RE.fullmatch(digest) is None
            or digest in digests
            or not isinstance(relative, str)
            or CONTENT_PATH_RE.fullmatch(relative) is None
            or not isinstance(windows_value, list)
            or not windows_value
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or float(duration) <= 0
        ):
            _fail("E_FROZEN_MANIFEST")
        media = (root / relative).resolve(strict=True)
        if not _inside(media, root) or not _private_regular(media):
            _fail("E_RESTRICTED_MEDIA")
        metadata = media.stat()
        inode = (metadata.st_dev, metadata.st_ino)
        if inode in inodes or _sha256_file(media) != digest:
            _fail("E_RESTRICTED_MEDIA")
        checked_windows: list[tuple[float, float]] = []
        previous = -1.0
        for window in windows_value:
            if not isinstance(window, dict) or set(window) != {"start_seconds", "end_seconds"}:
                _fail("E_FROZEN_WINDOWS")
            start = window["start_seconds"]
            end = window["end_seconds"]
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or float(start) < previous
                or float(start) < 0
                or float(end) <= float(start)
            ):
                _fail("E_FROZEN_WINDOWS")
            checked_windows.append((float(start), float(end)))
            previous = float(end)
            seconds += float(end) - float(start)
        window_count += len(checked_windows)
        digests.add(digest)
        inodes.add(inode)
        items.append(PilotItem(digest, media, tuple(checked_windows), float(duration)))
    if window_count != EXPECTED_WINDOWS or round(seconds / 60.0, 2) != EXPECTED_SPEECH_MINUTES:
        _fail("E_FROZEN_WINDOWS")
    return tuple(sorted(items, key=lambda item: item.digest))


class Checkpoint:
    TASKS = tuple(sorted(firewall.TASKS))

    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path)
        os.chmod(path, 0o600)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS work (
                item_sha256 TEXT NOT NULL,
                task TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('PENDING','RUNNING','COMPLETE','FAILED')),
                output_sha256 TEXT,
                output_bytes INTEGER,
                PRIMARY KEY(item_sha256, task)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def register(self, items: Sequence[PilotItem]) -> None:
        with self.connection:
            for item in items:
                for task in self.TASKS:
                    self.connection.execute(
                        "INSERT INTO work(item_sha256,task,state) VALUES(?,?,'PENDING') "
                        "ON CONFLICT(item_sha256,task) DO NOTHING",
                        (item.digest, task),
                    )
            self.connection.execute("UPDATE work SET state='PENDING' WHERE state IN ('RUNNING','FAILED')")

    def complete_valid(self, item: str, tasks: Sequence[str], output: Path) -> bool:
        rows = self.connection.execute(
            f"SELECT task,state,output_sha256,output_bytes FROM work WHERE item_sha256=? AND task IN ({','.join('?' for _ in tasks)})",
            (item, *tasks),
        ).fetchall()
        if len(rows) != len(tasks) or any(row[1] != "COMPLETE" for row in rows) or not output.is_file():
            return False
        digest = _sha256_file(output)
        size = output.stat().st_size
        return all(row[2] == digest and row[3] == size for row in rows)

    def running(self, item: str, tasks: Sequence[str]) -> None:
        with self.connection:
            for task in tasks:
                self.connection.execute(
                    "UPDATE work SET state='RUNNING',output_sha256=NULL,output_bytes=NULL WHERE item_sha256=? AND task=?",
                    (item, task),
                )

    def finish(self, item: str, tasks: Sequence[str], output: Path) -> None:
        digest = _sha256_file(output)
        size = output.stat().st_size
        if not (0 < size <= MAX_ITEM_OUTPUT_BYTES):
            _fail("E_PSEUDO_OUTPUT")
        with self.connection:
            for task in tasks:
                self.connection.execute(
                    "UPDATE work SET state='COMPLETE',output_sha256=?,output_bytes=? WHERE item_sha256=? AND task=? AND state='RUNNING'",
                    (digest, size, item, task),
                )

    def failed(self, item: str, tasks: Sequence[str]) -> None:
        with self.connection:
            for task in tasks:
                self.connection.execute(
                    "UPDATE work SET state='FAILED' WHERE item_sha256=? AND task=?",
                    (item, task),
                )

    def aggregate(self) -> tuple[Mapping[str, int], Mapping[str, Mapping[str, int]]]:
        states = {state: 0 for state in ("PENDING", "RUNNING", "COMPLETE", "FAILED")}
        task_counts = {
            task: {state: 0 for state in states}
            for task in self.TASKS
        }
        for task, state, count in self.connection.execute(
            "SELECT task,state,COUNT(*) FROM work GROUP BY task,state"
        ):
            task_counts[str(task)][str(state)] = int(count)
            states[str(state)] += int(count)
        return states, task_counts


def _atomic_private(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".pending-{os.getpid()}-{path.name}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except Exception:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _namespace_bytes(root: Path) -> int:
    total = 0
    try:
        for directory, directories, files in os.walk(root, followlinks=False):
            base = Path(directory)
            directories[:] = [name for name in directories if not (base / name).is_symlink()]
            for name in files:
                path = base / name
                metadata = path.lstat()
                if stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                    total += metadata.st_size
    except OSError:
        _fail("E_STORAGE_AUDIT")
    return total


@dataclass(frozen=True)
class StorageGuard:
    root: Path

    def assert_limits(self, *, projected_growth: int = 0) -> None:
        if projected_growth < 0:
            _fail("E_STORAGE_AUDIT")
        namespace = _namespace_bytes(self.root)
        try:
            free = shutil.disk_usage(self.root).free
        except OSError:
            _fail("E_STORAGE_AUDIT")
        if namespace > NAMESPACE_CAP_BYTES or namespace + projected_growth > NAMESPACE_CAP_BYTES:
            _fail("E_NAMESPACE_CAP")
        if free < FREE_SPACE_FLOOR_BYTES or free - projected_growth < FREE_SPACE_FLOOR_BYTES:
            _fail("E_FREE_SPACE_FLOOR")


def _adapter_environment(work_dir: Path) -> Mapping[str, str]:
    value = dict(firewall.scrubbed_subprocess_environment({
        "OMP_NUM_THREADS": str(CPU_WORKERS),
        "MKL_NUM_THREADS": str(CPU_WORKERS),
        "VECLIB_MAXIMUM_THREADS": str(CPU_WORKERS),
    }))
    value.update(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "TMPDIR": str(work_dir),
        }
    )
    return value


def _adapter_command(backend: Any, argv: Sequence[str], work_dir: Path) -> list[str]:
    """Return the fixed network sandbox after validating the private cwd.

    Write confinement is achieved by descriptor-only restricted input/output,
    a private cwd/TMPDIR, relative-only adapter temporaries, and post-run
    confinement checks. A broader macOS ``deny file-write*`` profile is not
    used because Metal requires private system-managed cache writes.
    """

    metadata = work_dir.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("E_QUARANTINE_CONFINEMENT")
    return backend.command(argv)


def _invoke_adapter(
    *,
    backend: Any,
    mode: str,
    item: PilotItem,
    job: Mapping[str, Any],
    output: Path,
    scratch_root: Path,
    audio_output: Path | None = None,
    storage_guard: StorageGuard | None = None,
    projected_growth: int = 0,
) -> None:
    scratch = Path(tempfile.mkdtemp(prefix="job-", dir=scratch_root))
    os.chmod(scratch, 0o700)
    media_fd = job_fd = output_fd = audio_fd = -1
    pending = scratch / "adapter-output.json"
    job_path = scratch / "job.json"
    try:
        _atomic_private(job_path, _canonical(job))
        media_fd = os.open(item.media, os.O_RDONLY)
        job_fd = os.open(job_path, os.O_RDONLY)
        output_fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        argv = [
            str(VENV_PYTHON),
            "-I",
            str(ADAPTER_PATH),
            "--mode",
            mode,
            "--job-fd",
            str(job_fd),
            "--media-fd",
            str(media_fd),
            "--output-fd",
            str(output_fd),
        ]
        pass_fds = [media_fd, job_fd, output_fd]
        if mode == "audio":
            argv.extend(
                [
                    "--ffmpeg",
                    str(FFMPEG),
                    "--whisper-cli",
                    str(WHISPER_CLI),
                    "--whisper-model",
                    str(WHISPER_MODEL),
                    "--silero-model",
                    str(SILERO_MODEL),
                ]
            )
        elif mode == "vlm" and audio_output is not None:
            audio_fd = os.open(audio_output, os.O_RDONLY)
            pass_fds.append(audio_fd)
            argv.extend(
                [
                    "--audio-fd",
                    str(audio_fd),
                    "--qwen-model",
                    str(QWEN_MODEL),
                ]
            )
        else:
            _fail("E_ADAPTER_MODE")
        if storage_guard is not None:
            storage_guard.assert_limits(projected_growth=projected_growth)
        process = subprocess.Popen(
            _adapter_command(backend, argv, scratch),
            cwd=scratch,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_adapter_environment(scratch),
            pass_fds=tuple(pass_fds),
            close_fds=True,
        )
        deadline = time.monotonic() + 8 * 60 * 60
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.terminate()
                with contextlib.suppress(Exception):
                    process.wait(timeout=10)
                if process.poll() is None:
                    process.kill()
                _fail("E_LOCAL_INSTRUMENT")
            if storage_guard is not None:
                try:
                    storage_guard.assert_limits()
                except RunnerError:
                    process.terminate()
                    with contextlib.suppress(Exception):
                        process.wait(timeout=10)
                    if process.poll() is None:
                        process.kill()
                    raise
            time.sleep(2.0)
        os.close(output_fd)
        output_fd = -1
        if process.returncode != 0 or not pending.is_file():
            _fail("E_LOCAL_INSTRUMENT")
        size = pending.stat().st_size
        if not (0 < size <= MAX_ITEM_OUTPUT_BYTES):
            _fail("E_PSEUDO_OUTPUT")
        os.replace(pending, output)
        os.chmod(output, 0o600)
    except RunnerError:
        raise
    except Exception:
        _fail("E_LOCAL_INSTRUMENT")
    finally:
        for descriptor in (audio_fd, output_fd, job_fd, media_fd):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        shutil.rmtree(scratch, ignore_errors=True)
        if storage_guard is not None:
            storage_guard.assert_limits()


def _validate_audio_output(path: Path, window_count: int) -> None:
    document = _read_private_json(path, path.parent.parent, MAX_ITEM_OUTPUT_BYTES)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != "childlens-restricted-audio-pseudo-labels-v1.3.0"
        or document.get("pseudo_labels_are_ground_truth") is not False
        or document.get("primary_evaluation_truth_allowed") is not False
        or document.get("official_window_count") != window_count
        or not isinstance(document.get("asr_hypotheses"), list)
        or not isinstance(document.get("vad_boundary_hypotheses"), list)
        or not isinstance(document.get("speaker_role_hypotheses"), list)
    ):
        _fail("E_AUDIO_OUTPUT_SCHEMA")


def _validate_vlm_output(path: Path, window_count: int) -> None:
    document = _read_private_json(path, path.parent.parent, MAX_ITEM_OUTPUT_BYTES)
    candidates = document.get("candidates") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != "childlens-restricted-referential-pseudo-labels-v1.3.0"
        or document.get("pseudo_labels_are_ground_truth") is not False
        or document.get("primary_evaluation_truth_allowed") is not False
        or document.get("frame_selection_prediction_independent") is not True
        or document.get("confidence_adaptive_resampling") is not False
        or document.get("window_count") != window_count
        or not isinstance(candidates, list)
        or len(candidates) != window_count
    ):
        _fail("E_VLM_OUTPUT_SCHEMA")


def _discover_runtime_root() -> Path:
    workflow = _load_module(V1_2_WORKFLOW_PATH, "childlens_human_validation_v1_2_for_v13_runner")
    try:
        root = Path(workflow.discover_runtime_root()).resolve(strict=True)
    except Exception:
        _fail("E_RUNTIME_ROOT")
    return firewall.validate_quarantine_root(root, REPO_ROOT)


def restricted_execute(envelope: Mapping[str, Any], *, backend: Any | None = None) -> Mapping[str, Any]:
    validate_instrument_seal(envelope)
    backend = backend or firewall.NetworkIsolationBackend.detect()
    firewall.verify_network_isolation(backend)
    os.umask(0o077)
    root = _discover_runtime_root()
    storage_guard = StorageGuard(root)
    storage_guard.assert_limits()
    manifest = _read_private_json(
        root / "post_acquisition_v1_2/restricted_measurement_manifest.json",
        root,
        MAX_MANIFEST_BYTES,
    )
    items = validate_measurement_manifest(manifest, root)
    namespace = _secure_directory(root / "model_assisted_v1_3", root)
    outputs = _secure_directory(namespace / "pseudo_annotations", root)
    audio_dir = _secure_directory(outputs / "audio", root)
    vlm_dir = _secure_directory(outputs / "referential", root)
    scratch = _secure_directory(namespace / "scratch", root)
    locks = _secure_directory(namespace / ".locks", root)
    checkpoint = Checkpoint(namespace / "checkpoint.sqlite3")
    checkpoint.register(items)
    lock_fd = os.open(locks / "mps-heavy.lock", os.O_RDWR | os.O_CREAT, 0o600)
    technical_failure = False
    try:
        for item in items:
            windows = [
                {"start_seconds": start, "end_seconds": end}
                for start, end in item.windows
            ]
            audio_tasks = ("VAD_SEGMENTATION", "MULTILINGUAL_ASR", "SPEAKER_ROLE_AID")
            audio_output = audio_dir / f"{item.digest}.json"
            if not checkpoint.complete_valid(item.digest, audio_tasks, audio_output):
                checkpoint.running(item.digest, audio_tasks)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    _invoke_adapter(
                        backend=backend,
                        mode="audio",
                        item=item,
                        job={
                            "schema_version": "childlens-restricted-audio-job-v1.3.0",
                            "windows": windows,
                        },
                        output=audio_output,
                        scratch_root=scratch,
                        storage_guard=storage_guard,
                        projected_growth=max(
                            FIXED_SCRATCH_RESERVE_BYTES,
                            round(item.duration_seconds * 16000 * 2 * 2),
                        ),
                    )
                    _validate_audio_output(audio_output, len(windows))
                    checkpoint.finish(item.digest, audio_tasks, audio_output)
                except (RunnerError, firewall.FirewallError) as exc:
                    checkpoint.failed(item.digest, audio_tasks)
                    if getattr(exc, "code", "") not in {
                        "E_LOCAL_INSTRUMENT",
                        "E_AUDIO_OUTPUT_SCHEMA",
                        "E_PSEUDO_OUTPUT",
                    }:
                        raise
                    technical_failure = True
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            if technical_failure:
                break
            vlm_tasks = ("REFERENTIAL_CANDIDATE",)
            vlm_output = vlm_dir / f"{item.digest}.json"
            if not checkpoint.complete_valid(item.digest, vlm_tasks, vlm_output):
                checkpoint.running(item.digest, vlm_tasks)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    _invoke_adapter(
                        backend=backend,
                        mode="vlm",
                        item=item,
                        job={
                            "schema_version": "childlens-restricted-vlm-job-v1.3.0",
                            "windows": windows,
                        },
                        output=vlm_output,
                        scratch_root=scratch,
                        audio_output=audio_output,
                        storage_guard=storage_guard,
                        projected_growth=FIXED_SCRATCH_RESERVE_BYTES,
                    )
                    _validate_vlm_output(vlm_output, len(windows))
                    checkpoint.finish(item.digest, vlm_tasks, vlm_output)
                except (RunnerError, firewall.FirewallError) as exc:
                    checkpoint.failed(item.digest, vlm_tasks)
                    if getattr(exc, "code", "") not in {
                        "E_LOCAL_INSTRUMENT",
                        "E_VLM_OUTPUT_SCHEMA",
                        "E_PSEUDO_OUTPUT",
                    }:
                        raise
                    technical_failure = True
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            if technical_failure:
                break
        counts, task_counts = checkpoint.aggregate()
        receipt = firewall.build_aggregate_receipt(
            counts,
            task_counts=task_counts,
            adapter_profiles=tuple(firewall.FIXED_ADAPTER_PROFILES),
            isolation_backend=backend.name,
            cpu_workers=CPU_WORKERS,
        )
        receipt = dict(receipt)
        receipt["storage_policy"] = {
            "namespace_cap_gib": 73,
            "free_space_floor_gib": 50,
            "preflight_projection_enforced": True,
            "during_process_monitoring_enforced": True,
            "post_item_cleanup_recheck_enforced": True,
        }
        return receipt
    finally:
        os.close(lock_fd)
        checkpoint.close()


def _validate_public_receipt(receipt: Any) -> Mapping[str, Any]:
    if not isinstance(receipt, dict):
        _fail("E_PUBLIC_RECEIPT")
    progress = receipt.get("progress")
    security = receipt.get("security")
    privacy = receipt.get("privacy_export")
    storage = receipt.get("storage_policy")
    status = receipt.get("status")
    complete = progress.get("work_units_complete") if isinstance(progress, dict) else None
    if (
        receipt.get("schema_version") != firewall.AGGREGATE_RECEIPT_SCHEMA
        or status not in {"COMPLETE", "INCOMPLETE"}
        or receipt.get("candidate_window_count") != EXPECTED_WINDOWS
        or receipt.get("candidate_speech_minutes") != EXPECTED_SPEECH_MINUTES
        or not isinstance(progress, dict)
        or progress.get("work_units_total") != EXPECTED_ITEMS * len(firewall.TASKS)
        or not isinstance(complete, int)
        or isinstance(complete, bool)
        or not (0 <= complete <= EXPECTED_ITEMS * len(firewall.TASKS))
        or (status == "COMPLETE" and complete != EXPECTED_ITEMS * len(firewall.TASKS))
        or (status == "INCOMPLETE" and complete == EXPECTED_ITEMS * len(firewall.TASKS))
        or progress.get("item_or_task_cells_exported") is not False
        or not isinstance(security, dict)
        or security.get("network_disabled_during_restricted_inference") is not True
        or security.get("no_hosted_or_cloud_content_path") is not True
        or security.get("external_api_used") is not False
        or security.get("external_upload") is not False
        or not isinstance(privacy, dict)
        or any(value is not False for value in privacy.values())
        or not isinstance(storage, dict)
        or storage.get("namespace_cap_gib") != 73
        or storage.get("free_space_floor_gib") != 50
        or storage.get("preflight_projection_enforced") is not True
        or storage.get("during_process_monitoring_enforced") is not True
        or storage.get("post_item_cleanup_recheck_enforced") is not True
    ):
        _fail("E_PUBLIC_RECEIPT")
    encoded = _canonical(receipt).decode("utf-8")
    if re.search(r"(?i)(?:/Users/|/home/|file://|\.mp4|\.mov|transcript_text|participant_id)", encoded):
        _fail("E_PUBLIC_RECEIPT_PRIVACY")
    return receipt


def _atomic_public_receipt(receipt: Mapping[str, Any]) -> None:
    PUBLIC_RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    temporary = PUBLIC_RECEIPT_PATH.with_suffix(".json.pending")
    temporary.write_bytes(payload)
    os.replace(temporary, PUBLIC_RECEIPT_PATH)


def public_execute(*, process_runner: Any = subprocess.run) -> Mapping[str, Any]:
    envelope = prepare_instrument_seal()
    backend = firewall.NetworkIsolationBackend.detect()
    seal_read, seal_write = os.pipe()
    result_read, result_write = os.pipe()
    try:
        os.write(seal_write, _canonical(envelope))
        os.close(seal_write)
        seal_write = -1
        command = backend.command(
            [
                str(VENV_PYTHON),
                "-I",
                str(SCRIPT_PATH),
                "--restricted-phase",
                "--seal-fd",
                str(seal_read),
                "--result-fd",
                str(result_write),
            ]
        )
        environment = dict(firewall.scrubbed_subprocess_environment())
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        completed = process_runner(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            pass_fds=(seal_read, result_write),
            close_fds=True,
            check=False,
            timeout=24 * 60 * 60,
        )
        os.close(result_write)
        result_write = -1
        with os.fdopen(result_read, "rb") as handle:
            payload = handle.read(MAX_RESULT_BYTES + 1)
        result_read = -1
        if not payload or len(payload) > MAX_RESULT_BYTES:
            _fail("E_RESTRICTED_INFERENCE")
        try:
            result = json.loads(payload)
        except Exception:
            _fail("E_RESTRICTED_INFERENCE")
        if completed.returncode == 0:
            receipt = _validate_public_receipt(result)
        else:
            receipt = _validate_public_receipt(
                _incomplete_receipt_from_diagnostic(
                    result, isolation_backend=backend.name
                )
            )
        _atomic_public_receipt(receipt)
        return receipt
    except RunnerError:
        raise
    except Exception:
        _fail("E_RESTRICTED_INFERENCE")
    finally:
        for descriptor in (seal_read, seal_write, result_read, result_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restricted-phase", action="store_true")
    parser.add_argument("--seal-fd", type=int)
    parser.add_argument("--result-fd", type=int)
    args = parser.parse_args(list(argv))
    if args.restricted_phase != (args.seal_fd is not None and args.result_fd is not None):
        _fail("E_ARGUMENTS")
    return args


def main() -> int:
    args: argparse.Namespace | None = None
    try:
        args = parse_args(sys.argv[1:])
        if args.restricted_phase:
            envelope = _read_fd_json(args.seal_fd, MAX_RESULT_BYTES)
            receipt = restricted_execute(envelope)
            _write_fd_json(args.result_fd, receipt)
        else:
            receipt = public_execute()
            print(
                "CHILDLENS_V13_PSEUDO_ANNOTATION_COMPLETE"
                if receipt.get("status") == "COMPLETE"
                else "CHILDLENS_V13_PSEUDO_ANNOTATION_INCOMPLETE"
            )
        return 0
    except (RunnerError, firewall.FirewallError) as exc:
        if args is not None and args.restricted_phase and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd_json(args.result_fd, _diagnostic_envelope(exc))
        else:
            print(getattr(exc, "code", "E_FAIL_CLOSED"))
        return 1
    except Exception as exc:
        if args is not None and args.restricted_phase and args.result_fd is not None:
            with contextlib.suppress(Exception):
                _write_fd_json(args.result_fd, _diagnostic_envelope(exc))
        else:
            print("E_FAIL_CLOSED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
