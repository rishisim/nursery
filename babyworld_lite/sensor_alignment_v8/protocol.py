from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import PROTOCOL_ID

OPERATIONS = (
    "generate",
    "condition",
    "detector_infer",
    "fit",
    "predict",
    "score",
    "control",
    "factor",
    "recompute",
)
ALLOWED_PURPOSES = frozenset({"fixture_only", "excluded_smoke"})
FORBIDDEN_VISIBLE_FIELDS = frozenset(
    {
        "answer_index",
        "correct_candidate_id",
        "event_concepts",
        "event_owners",
        "grounded",
        "intended_concept",
        "lexicon_oracle",
        "oracle",
        "target_event_index",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_bytes(value))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
        for row in rows
    )
    destination.write_bytes(payload)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _integer_leaves(value: Any) -> set[int]:
    output: set[int] = set()
    if isinstance(value, bool):
        return output
    if isinstance(value, int):
        output.add(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            output.update(_integer_leaves(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            output.update(_integer_leaves(child))
    return output


def load_config(path: str | Path, *, repository_root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text())
    if value["protocol"]["id"] != PROTOCOL_ID:
        raise ValueError("wrong protocol id")
    if value["protocol"]["status"] not in {"pre_freeze", "frozen"}:
        raise ValueError("v8 configuration status must be pre_freeze or frozen")
    root = Path(repository_root).resolve() if repository_root else config_path.parents[1]
    prior: set[int] = set()
    for relative in value["registries"]["prior_registry_files"]:
        prior_path = root / str(relative)
        if not prior_path.is_file():
            raise FileNotFoundError(prior_path)
        prior_value = yaml.safe_load(prior_path.read_text())
        prior.update(
            _integer_leaves(
                prior_value.get("seeds", prior_value.get("registries", {}))
            )
        )
    registries: dict[str, dict[str, list[int]]] = {}
    sets: dict[str, set[int]] = {}
    for name in ("fixture_only", "excluded_smoke", "future_development", "confirmation_reserve"):
        registry = value["registries"][name]
        registries[name] = {
            role: list(map(int, registry[role])) for role in ("corpus", "model", "inference")
        }
        role_sets = [set(registries[name][role]) for role in ("corpus", "model", "inference")]
        if any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i)):
            raise ValueError(f"overlap between roles in {name}")
        sets[name] = set().union(*role_sets)
    sets["prior_v1_v7"] = prior
    names = list(sets)
    overlaps = {
        f"{left}|{right}": sorted(sets[left] & sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if sets[left] & sets[right]
    }
    if overlaps:
        raise ValueError(f"identifier registries overlap: {overlaps}")
    value["resolved_registries"] = {**registries, "prior_v1_v7": sorted(prior)}
    return value


def require_frozen_config(config: Mapping[str, Any]) -> None:
    if config["protocol"]["status"] != "frozen":
        raise RuntimeError("excluded smoke and package freeze require protocol.status=frozen")


@dataclass(frozen=True, slots=True)
class IdentifierReference:
    role: str
    value: int


class IdentifierFirewall:
    def __init__(self, config: Mapping[str, Any], *, allowed_purposes: Sequence[str]):
        unknown = set(allowed_purposes) - ALLOWED_PURPOSES
        if unknown:
            raise PermissionError(f"forbidden purposes: {sorted(unknown)}")
        self.allowed_purposes = frozenset(allowed_purposes)
        self.registries = config["resolved_registries"]
        self.log: list[dict[str, Any]] = []

    def authorize(
        self,
        operation: str,
        references: Sequence[IdentifierReference],
        *,
        purpose: str,
    ) -> str:
        if operation not in OPERATIONS:
            raise ValueError(operation)
        allowed = purpose in self.allowed_purposes and purpose in ALLOWED_PURPOSES
        reason = "allowed"
        for reference in references:
            if reference.role not in {"corpus", "model", "inference"}:
                allowed = False
                reason = "unknown-role"
                break
            expected = set(self.registries[purpose][reference.role]) if purpose in ALLOWED_PURPOSES else set()
            if int(reference.value) not in expected:
                allowed = False
                reason = "identifier-not-in-purpose-role"
                break
            for forbidden in ("future_development", "confirmation_reserve", "prior_v1_v7"):
                registry = self.registries[forbidden]
                values = set(registry) if forbidden == "prior_v1_v7" else {
                    item for role_values in registry.values() for item in role_values
                }
                if int(reference.value) in values:
                    allowed = False
                    reason = f"blocked-{forbidden}"
                    break
        record = {
            "operation": operation,
            "purpose": purpose,
            "references": [
                {"role": reference.role, "value": int(reference.value)}
                for reference in references
            ],
            "status": "ALLOW" if allowed else "BLOCK",
            "reason": reason,
        }
        self.log.append(record)
        if not allowed:
            raise PermissionError(record)
        return canonical_digest(record)


def registry_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = config["resolved_registries"]
    return {
        "protocol_id": PROTOCOL_ID,
        "registries": resolved,
        "pairwise_disjoint": True,
        "prior_identifier_count": len(resolved["prior_v1_v7"]),
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }


def reject_oracle_fields(value: Any, *, path: str = "visible") -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_VISIBLE_FIELDS & set(value)
        if forbidden:
            raise ValueError(f"oracle/key fields in {path}: {sorted(forbidden)}")
        for key, child in value.items():
            reject_oracle_fields(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            reject_oracle_fields(child, path=f"{path}[{index}]")


def manifest_for_files(root: str | Path, files: Iterable[str | Path]) -> dict[str, Any]:
    base = Path(root).resolve()
    rows = []
    for item in sorted({Path(value).as_posix() for value in files}):
        path = base / item
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": item, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": rows, "file_count": len(rows), "digest": canonical_digest(rows)}


def verify_file_manifest(root: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    base = Path(root).resolve()
    problems = []
    for row in manifest.get("files", []):
        path = base / str(row["path"])
        if not path.is_file():
            problems.append({"path": row["path"], "problem": "missing"})
            continue
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed["bytes"] != int(row["bytes"]) or observed["sha256"] != row["sha256"]:
            problems.append({"path": row["path"], "problem": "digest_or_size", "observed": observed})
    digest_ok = canonical_digest(manifest.get("files", [])) == manifest.get("digest")
    return {
        "status": "PASS" if not problems and digest_ok else "FAIL",
        "checked_files": len(manifest.get("files", [])),
        "manifest_digest_valid": digest_ok,
        "problems": problems,
    }


def verify_prior_preservation(
    repository_root: str | Path,
    before_path: str | Path,
    after_path: str | Path,
    proof_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    before_lines = Path(before_path).read_text().splitlines()
    before: dict[str, tuple[str, int]] = {}
    for line in before_lines:
        digest, size, relative = line.split("\t", 2)
        before[relative] = (digest, int(size))
    after_rows = []
    missing = []
    changed = []
    for relative, (expected_digest, expected_size) in sorted(before.items()):
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        after_rows.append(f"{digest}\t{size}\t{relative}")
        if digest != expected_digest or size != expected_size:
            changed.append(relative)
    Path(after_path).write_text("\n".join(after_rows) + "\n")
    proof = {
        "status": "PASS" if not missing and not changed and len(after_rows) == len(before) else "FAIL",
        "baseline_file_count": len(before),
        "after_file_count": len(after_rows),
        "missing_paths": missing,
        "changed_paths": changed,
        "byte_for_byte_preserved": not missing and not changed and len(after_rows) == len(before),
        "before_manifest_sha256": sha256_file(before_path),
        "after_manifest_sha256": sha256_file(after_path),
    }
    write_json(proof_path, proof)
    return proof


def environment_record(repository_root: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    executable = root / str(config["environment"]["python"])
    if Path(sys.executable).absolute() != executable:
        raise RuntimeError(f"freeze must run under {executable}, got {sys.executable}")
    return {
        "python_executable": str(executable),
        "python_sha256": sha256_file(executable),
        "python_version": sys.version,
        "test_roots": [
            str(config["environment"]["version_test_root"]),
            str(config["environment"]["repository_test_root"]),
        ],
        "official_commands": [
            f"PYTHONDONTWRITEBYTECODE=1 {executable} -m pytest -q -p no:cacheprovider {config['environment']['version_test_root']}",
            f"PYTHONDONTWRITEBYTECODE=1 {executable} -m pytest -q -p no:cacheprovider {config['environment']['repository_test_root']}",
        ],
        "bytecode_writes_disabled": True,
        "repository_test_collection_scoped_to_tests": True,
    }
