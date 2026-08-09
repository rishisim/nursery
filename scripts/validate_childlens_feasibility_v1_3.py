#!/usr/bin/env python3
"""Fail-closed repository/privacy/immutability validator for ChildLens v1.3.

Only Git-indexed files and explicit v1.3 docs/output namespaces are inspected.
The validator never searches for, resolves, or reads a quarantine.  Findings
contain stable problem classes and repository-relative paths, never matched
payloads.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_PATH = REPO_ROOT / "scripts/synthesize_childlens_terminal_v1_3.py"
SPEC = importlib.util.spec_from_file_location("childlens_v1_3_synth_for_validator", SYNTH_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("E_SYNTH_IMPORT")
synth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = synth
SPEC.loader.exec_module(synth)


EXPECTED_HISTORY = {
    "v1": (23, "35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf"),
    "v1_1": (40, "3acd969804d71979ba8071e2446e8ee1ceb3194fc90dcfda802a99e296b7bf48"),
    "v1_2": (53, "f3bbfd9051e33fc33e44b9cab3c7546b77b1acba4904cc73fb116a3cbe7da01f"),
}

V1_PATHS = (
    "docs/childlens_feasibility_v1",
    "output/childlens_feasibility_v1",
    "scripts/validate_childlens_feasibility_v1.py",
    "tests/test_childlens_feasibility_v1.py",
)
V1_1_DIRECTORIES = (
    "docs/childlens_feasibility_v1_1",
    "output/childlens_feasibility_v1_1",
)
V1_2_DIRECTORIES = (
    "docs/childlens_feasibility_v1_2",
    "output/childlens_feasibility_v1_2",
)

REQUIRED_PATHS = (
    "docs/childlens_feasibility_v1_3/frozen_model_assisted_author_audit_protocol_v1_3.json",
    "docs/childlens_feasibility_v1_3/local_inference_firewall_v1_3.md",
    "docs/childlens_feasibility_v1_3/local_instrument_license_audit.md",
    "docs/childlens_feasibility_v1_3/executive_decision_report.md",
    "output/childlens_feasibility_v1_3/local_instrument_provenance.json",
    "output/childlens_feasibility_v1_3/local_instrument_cache_reconciliation.json",
    "output/childlens_feasibility_v1_3/protocol_freeze_receipt.json",
    "output/childlens_feasibility_v1_3/author_audit_sampling_receipt.json",
    "output/childlens_feasibility_v1_3/pseudo_annotation_receipt.json",
    "output/childlens_feasibility_v1_3/author_workflow_receipt.json",
    "output/childlens_feasibility_v1_3/decision_record.json",
)

PROHIBITED_PAYLOAD_EXTENSIONS = frozenset(
    {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".wav", ".mp3", ".m4a",
        ".aac", ".flac", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
        ".tiff", ".srt", ".vtt", ".zip", ".tar", ".gz", ".csv", ".tsv",
        ".sqlite", ".sqlite3", ".db", ".parquet", ".arrow",
    }
)
TEXT_SUFFIXES = frozenset({".json", ".md", ".py", ".mjs", ".js", ".txt", ".yaml", ".yml"})

RESTRICTED_MACHINE_KEYS = synth.FORBIDDEN_KEYS | frozenset(
    {
        "participant_identifier",
        "participant_key",
        "child_identifier",
        "session_identifier",
        "session_key",
        "recording_identifier",
        "episode_id",
        "video_id",
        "audio_id",
        "speaker_cluster_id",
        "row_id",
        "raw_video",
        "raw_audio",
        "frame",
        "frames",
        "thumbnail",
        "raw_transcript",
        "word_string",
        "speaker_embedding",
        "voiceprint",
        "instrument_embedding",
        "cookies",
        "credentials",
        "media_timestamp",
        "onset_time",
        "offset_time",
        "utterance_start",
        "utterance_end",
        "frame_time",
    }
)
ABSOLUTE_USER_PATH_RE = re.compile(r"(?i)(?:/Users/|file://|\\Users\\)")
MEDIA_FILENAME_RE = synth.MEDIA_NAME_RE
MEDIA_CLOCK_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?!\d)")
SUBTITLE_CUE_RE = re.compile(
    r"(?m)^\s*\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,6})?\s*-->\s*\d{1,2}:\d{2}:\d{2}"
)
DIALOGUE_RE = re.compile(
    r"(?im)^\s*(?:child|non_child|adult|caregiver|speaker[_ -]?\d+)\s*:\s+\S+"
)
UUID_RE = synth.UUID_RE
EMAIL_RE = re.compile(r"(?i)\b[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9.-]+\.[a-z]{2,}\b")

HOSTED_CONTENT_CODE_PATTERNS = (
    re.compile(r"(?m)^\s*(?:from|import)\s+openai\b"),
    re.compile(r"(?m)^\s*(?:from|import)\s+anthropic\b"),
    re.compile(r"(?m)^\s*(?:from|import)\s+google\.generativeai\b"),
    re.compile(r"(?i)api\.openai\.com"),
    re.compile(r"(?i)api\.anthropic\.com"),
    re.compile(r"(?i)generativelanguage\.googleapis\.com"),
)


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _issue(issues: list[Issue], code: str, path: Path, root: Path, message: str) -> None:
    issues.append(Issue(code, _relative(path, root), message))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand(root: Path, entries: tuple[str, ...]) -> set[Path]:
    files: set[Path] = set()
    for entry in entries:
        path = root / entry
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(
                child for child in path.rglob("*")
                if child.is_file() and "__pycache__" not in child.parts
            )
    return files


def historical_files(root: Path, version: str) -> list[Path]:
    if version == "v1":
        files = _expand(root, V1_PATHS)
    elif version == "v1_1":
        files = _expand(root, V1_1_DIRECTORIES)
        for directory in ("scripts", "tests"):
            files.update((root / directory).glob("*childlens*v1_1*"))
    elif version == "v1_2":
        files = _expand(root, V1_2_DIRECTORIES)
        for directory in ("scripts", "tests"):
            files.update((root / directory).glob("*childlens*v1_2*"))
    else:  # pragma: no cover
        raise ValueError(version)
    return sorted(
        (path for path in files if path.is_file() and "__pycache__" not in path.parts),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def artifact_set_digest(root: Path, files: list[Path]) -> tuple[int, str]:
    lines = sorted(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files
    )
    return len(files), hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def historical_status(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for version, (expected_count, expected_digest) in EXPECTED_HISTORY.items():
        count, digest = artifact_set_digest(root, historical_files(root, version))
        result[version] = {
            "artifact_count": count,
            "artifact_set_digest": digest,
            "expected_artifact_count": expected_count,
            "expected_artifact_set_digest": expected_digest,
            "preserved": count == expected_count and digest == expected_digest,
        }
    return result


def _normalise_key(key: object) -> str:
    return synth._normalise_key(key)


def _walk(value: Any) -> Iterator[tuple[str, Any]]:
    yield from synth._walk(value)


def _load_json(path: Path, root: Path, issues: list[Issue]) -> Any | None:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise OSError("unsafe file kind")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _issue(issues, "INVALID_JSON", path, root, "parse/file-kind failure; content suppressed")
        return None


def _explicit_v1_3_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in ("docs/childlens_feasibility_v1_3", "output/childlens_feasibility_v1_3"):
        path = root / relative
        if path.is_dir():
            files.update(child for child in path.rglob("*") if child.is_file() or child.is_symlink())
    for directory in ("scripts", "tests"):
        path = root / directory
        if path.is_dir():
            files.update(child for child in path.glob("*childlens*v1_3*") if child.is_file() or child.is_symlink())
    return sorted(files, key=lambda path: _relative(path, root))


def _git_tracked_v1_3_files(root: Path) -> list[Path]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    files: list[Path] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            relative = Path(raw.decode("utf-8"))
        except UnicodeDecodeError:
            continue
        text = relative.as_posix().lower()
        if relative.is_absolute() or ".." in relative.parts:
            continue
        if "childlens" not in text or ("v1_3" not in text and "v1.3" not in text):
            continue
        candidate = root / relative
        if candidate.is_file() or candidate.is_symlink():
            files.append(candidate)
    return files


def _scan_file(path: Path, root: Path, issues: list[Issue]) -> None:
    try:
        info = path.lstat()
    except OSError:
        _issue(issues, "UNREADABLE_ARTIFACT", path, root, "metadata failure")
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _issue(issues, "UNSAFE_ARTIFACT_KIND", path, root, "symlink/non-regular artifact")
        return
    if path.suffix.lower() in PROHIBITED_PAYLOAD_EXTENSIONS:
        _issue(issues, "RESTRICTED_PAYLOAD_EXTENSION", path, root, "payload-like extension")
        return
    if path.suffix.lower() not in TEXT_SUFFIXES or info.st_size > 4 * 1024 * 1024:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _issue(issues, "UNREADABLE_TEXT_ARTIFACT", path, root, "read failure")
        return
    is_code = path.suffix.lower() in {".py", ".mjs", ".js"}
    if not is_code:
        for pattern, code in (
            (ABSOLUTE_USER_PATH_RE, "ABSOLUTE_USER_PATH"),
            (MEDIA_FILENAME_RE, "MEDIA_FILENAME"),
            (MEDIA_CLOCK_RE, "EXACT_MEDIA_CLOCK"),
            (SUBTITLE_CUE_RE, "SUBTITLE_CUE"),
            (DIALOGUE_RE, "TRANSCRIPT_LIKE_DIALOGUE"),
            (UUID_RE, "UUID_IDENTIFIER"),
            (EMAIL_RE, "EMAIL_ADDRESS"),
        ):
            if pattern.search(text):
                _issue(issues, code, path, root, "possible restricted value; matched text suppressed")
    if path.suffix.lower() == ".json":
        value = _load_json(path, root, issues)
        if value is not None:
            public_instrument_inventory = path.name in {
                "local_instrument_provenance.json",
                "local_instrument_cache_reconciliation.json",
            }
            public_frozen_protocol = path.name == "frozen_model_assisted_author_audit_protocol_v1_3.json"
            for key, child in _walk(value):
                # Public model artifact names are ordinary provenance, not
                # ChildLens source names.  All other filename fields remain
                # prohibited in the v1.3 report namespace.
                if key in RESTRICTED_MACHINE_KEYS and not (
                    public_instrument_inventory and key in {"filename", "file_name"}
                ) and not (
                    public_frozen_protocol and key == "transcript_text" and isinstance(child, bool)
                ):
                    _issue(issues, "RESTRICTED_MACHINE_KEY", path, root, "restricted key; value suppressed")
                    break
    if path.suffix.lower() == ".py" and (path.parent == root / "scripts" or path.parent == root / "tests"):
        hosted = False
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.Import):
                    module = " ".join(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                if any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in ("openai", "anthropic", "google.generativeai")
                ):
                    hosted = True
                    break
        # The validator/test files intentionally contain endpoint signatures as
        # sentinels. Other v1.3 code may not contain those endpoints.
        is_validator_sentinel = path.name in {
            "validate_childlens_feasibility_v1_3.py",
            "test_validate_childlens_feasibility_v1_3.py",
        }
        if not is_validator_sentinel and any(pattern.search(text) for pattern in HOSTED_CONTENT_CODE_PATTERNS[3:]):
            hosted = True
        if hosted:
            _issue(
                issues,
                "HOSTED_CONTENT_CODE_PATH",
                path,
                root,
                "hosted-model client or endpoint in v1.3 code",
            )


def _check_receipt_semantics(root: Path, issues: list[Issue]) -> None:
    loaded: dict[str, Any] = {}
    for name, relative in synth.INPUT_RELATIVE_PATHS.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            continue
        value = _load_json(path, root, issues)
        if isinstance(value, dict):
            loaded[name] = value
    if len(loaded) != len(synth.INPUT_RELATIVE_PATHS):
        return
    validators = {
        "protocol": synth._protocol_valid,
        "sampling": synth._sampling_valid,
        "pseudo": synth._pseudo_valid,
        "workflow": synth._workflow_valid,
    }
    for name, validator in validators.items():
        if not validator(loaded[name]):
            _issue(
                issues,
                f"{name.upper()}_RECEIPT_FAIL_CLOSED",
                root / synth.INPUT_RELATIVE_PATHS[name],
                root,
                "required aggregate safety/readiness condition missing or false",
            )

    protocol_doc = (
        root
        / "docs/childlens_feasibility_v1_3/frozen_model_assisted_author_audit_protocol_v1_3.json"
    )
    protocol_document = _load_json(protocol_doc, root, issues)
    if isinstance(protocol_document, dict):
        receipt = loaded["protocol"]
        if receipt.get("protocol_document_sha256") != _sha256(protocol_doc):
            _issue(
                issues,
                "PROTOCOL_DOCUMENT_BINDING_MISMATCH",
                root / synth.INPUT_RELATIVE_PATHS["protocol"],
                root,
                "protocol receipt does not bind the frozen public document",
            )
        frozen = protocol_document.get("freeze_binding")
        frozen_digest = frozen.get("frozen_payload_sha256") if isinstance(frozen, dict) else None
        if receipt.get("frozen_payload_sha256") != frozen_digest:
            _issue(
                issues,
                "PROTOCOL_PAYLOAD_BINDING_MISMATCH",
                root / synth.INPUT_RELATIVE_PATHS["protocol"],
                root,
                "receipt/internal frozen-payload digest mismatch",
            )

    provenance_path = root / "output/childlens_feasibility_v1_3/local_instrument_provenance.json"
    provenance = _load_json(provenance_path, root, issues) if provenance_path.is_file() else None
    if not isinstance(provenance, dict) or not (
        provenance.get("schema") == "childlens-local-instrument-provenance-v1.3.1"
        and provenance.get("scope") == "public_sources_only_no_restricted_input_or_execution"
        and provenance.get("disposition")
        == "APPROVED_FOR_PUBLIC_ACQUISITION_SYNTHETIC_SMOKE_TEST_AND_OFFLINE_QUARANTINED_INFERENCE"
    ):
        _issue(
            issues,
            "INSTRUMENT_PROVENANCE_INVALID",
            provenance_path,
            root,
            "public license/hash provenance is absent or not approved",
        )
    cache_path = root / "output/childlens_feasibility_v1_3/local_instrument_cache_reconciliation.json"
    cache = _load_json(cache_path, root, issues) if cache_path.is_file() else None
    if not isinstance(cache, dict) or not (
        cache.get("schema") == "childlens-public-instrument-cache-reconciliation-v1.3.0"
        and cache.get("scope") == "public_instruments_and_synthetic_inputs_only"
        and cache.get("disposition")
        == "REQUIRED_LOCAL_INSTRUMENTS_HASH_VERIFIED_AND_SYNTHETIC_SMOKE_PASSED"
        and cache.get("restricted_childlens_input_used") is False
        and cache.get("restricted_inference_run") is False
        and cache.get("hosted_or_cloud_model_used") is False
        and cache.get("cache_paths_exported") is False
    ):
        _issue(
            issues,
            "INSTRUMENT_CACHE_RECONCILIATION_INVALID",
            cache_path,
            root,
            "public cache/hash/synthetic-smoke receipt is absent or invalid",
        )
    audit_path = root / synth.OPTIONAL_AUDIT_RELATIVE_PATH
    audit: Mapping[str, Any] | None = None
    if audit_path.exists() or audit_path.is_symlink():
        value = _load_json(audit_path, root, issues)
        if isinstance(value, dict):
            audit = value
            if value.get("audit_complete") is True and not synth._audit_valid_shape(value):
                _issue(
                    issues,
                    "AUTHOR_LOCK_OR_BLINDING_INVALID",
                    audit_path,
                    root,
                    "completed audit fails immutable-lock/blinding schema",
                )

    expected = synth.decide(
        loaded["protocol"], loaded["sampling"], loaded["pseudo"], loaded["workflow"], audit
    )
    decision_path = root / synth.OUTPUT_RELATIVE_PATHS["decision"]
    decision = _load_json(decision_path, root, issues)
    if not isinstance(decision, dict):
        return
    if decision.get("terminal_state") not in synth.TERMINAL_STATES:
        _issue(issues, "INVALID_TERMINAL_STATE", decision_path, root, "unknown terminal literal")
    elif decision.get("terminal_state") != expected.terminal_state:
        _issue(issues, "TERMINAL_STATE_EVIDENCE_MISMATCH", decision_path, root, "decision does not follow receipts")
    if decision.get("schema_version") != "childlens-v1.3-decision-record-v1":
        _issue(issues, "DECISION_SCHEMA", decision_path, root, "decision schema mismatch")
    if decision.get("decision_basis_code") != expected.basis_code:
        _issue(issues, "DECISION_BASIS_MISMATCH", decision_path, root, "basis does not follow receipts")
    expected_hashes = {name: _sha256(root / relative) for name, relative in synth.INPUT_RELATIVE_PATHS.items()}
    if audit is not None:
        expected_hashes["audit"] = _sha256(audit_path)
    if decision.get("input_receipt_sha256") != expected_hashes:
        _issue(issues, "STALE_DECISION_INPUT_BINDING", decision_path, root, "decision input hashes are stale")
    if decision.get("historical_artifact_sets") != historical_status(root):
        _issue(issues, "STALE_HISTORY_BINDING", decision_path, root, "decision history binding is stale")
    required_false = (
        "hosted_or_cloud_model_inspected_restricted_content",
        "restricted_network_egress",
        "learner_training_executed",
        "corpus_tokenizer_trained",
        "causal_arm_executed",
        "scientific_acquisition_outcome_executed",
    )
    if not all(decision.get(key) is False for key in required_false):
        _issue(issues, "PROHIBITED_BOUNDARY_CROSSED", decision_path, root, "prohibited action not false")
    if not all(
        decision.get(key) is True
        for key in ("historical_v1_preserved", "historical_v1_1_preserved", "historical_v1_2_preserved")
    ):
        _issue(issues, "HISTORY_NOT_DECLARED_PRESERVED", decision_path, root, "history declaration false")

    executive_path = root / synth.OUTPUT_RELATIVE_PATHS["executive"]
    try:
        executive = executive_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return
    literals = synth.TERMINAL_RE.findall(executive)
    if literals != [expected.terminal_state]:
        _issue(issues, "EXECUTIVE_TERMINAL_LITERAL", executive_path, root, "must contain exactly one evidence-backed state")


def validate(root: Path) -> list[Issue]:
    root = root.resolve(strict=True)
    issues: list[Issue] = []
    for relative in REQUIRED_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            _issue(issues, "MISSING_REQUIRED_ARTIFACT", path, root, "required public artifact missing")

    for version, status in historical_status(root).items():
        if not status["preserved"]:
            _issue(
                issues,
                "HISTORICAL_MUTATION",
                root / f"docs/childlens_feasibility_{version}",
                root,
                f"{version} artifact count/digest differs from frozen baseline",
            )

    files = set(_explicit_v1_3_files(root)) | set(_git_tracked_v1_3_files(root))
    for path in sorted(files, key=lambda item: _relative(item, root)):
        _scan_file(path, root, issues)
    _check_receipt_semantics(root, issues)
    return issues


def main() -> int:
    if len(sys.argv) != 1:
        print("E_NO_ARGUMENTS", file=sys.stderr)
        return 2
    issues = validate(REPO_ROOT)
    payload = {
        "schema_version": "childlens-v1.3-validation-result-v1",
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": [issue.__dict__ for issue in issues],
        "historical_status": historical_status(REPO_ROOT),
        "quarantine_inspected": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
