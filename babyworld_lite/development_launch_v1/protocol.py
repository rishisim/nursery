from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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
    "inference",
    "recompute",
)
REHEARSAL_PURPOSES = frozenset({"fixture_rehearsal", "excluded_rehearsal"})
EXECUTABLE_PURPOSES = frozenset({*REHEARSAL_PURPOSES, "development"})
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
    destination.write_bytes(
        b"".join(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n"
            for row in rows
        )
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line
    ]


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


def _registry(value: Mapping[str, Any]) -> dict[str, list[int]]:
    return {
        role: list(map(int, value[role]))
        for role in ("corpus", "model", "inference")
    }


def _flatten(registry: Mapping[str, Sequence[int]]) -> set[int]:
    return {
        int(value)
        for values in registry.values()
        for value in values
    }


def load_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text())
    if value["protocol"]["id"] != PROTOCOL_ID:
        raise ValueError("wrong development launch protocol id")
    if value["protocol"]["status"] not in {"pre_freeze", "frozen"}:
        raise ValueError("development launch status must be pre_freeze or frozen")
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else config_path.parents[1]
    )
    prior: set[int] = set()
    for relative in value["registries"]["prior_registry_files"]:
        prior_value = yaml.safe_load((root / str(relative)).read_text())
        prior.update(
            _integer_leaves(
                prior_value.get("seeds", prior_value.get("registries", {}))
            )
        )
    qualification_path = root / str(
        value["registries"]["qualification_registry_file"]
    )
    qualification = yaml.safe_load(qualification_path.read_text())
    qualification_registries = qualification["registries"]
    registries = {
        name: _registry(value["registries"][name])
        for name in (
            "fixture_rehearsal",
            "excluded_rehearsal",
            "development",
            "confirmation_reserve",
        )
    }
    for name, registry in registries.items():
        role_sets = [set(registry[role]) for role in ("corpus", "model", "inference")]
        if any(
            role_sets[left] & role_sets[right]
            for left in range(3)
            for right in range(left)
        ):
            raise ValueError(f"role overlap in {name}")
    if registries["development"]["corpus"] != list(
        map(int, qualification_registries["future_development"]["corpus"])
    ):
        raise ValueError("development corpus registry does not carry the V8 reserve")
    if registries["development"]["model"] != list(
        map(int, qualification_registries["future_development"]["model"])
    ):
        raise ValueError("development model registry does not carry the V8 reserve")
    if registries["confirmation_reserve"]["corpus"] != list(
        map(int, qualification_registries["confirmation_reserve"]["corpus"])
    ):
        raise ValueError("confirmation corpus registry does not carry the V8 reserve")
    if registries["confirmation_reserve"]["model"] != list(
        map(int, qualification_registries["confirmation_reserve"]["model"])
    ):
        raise ValueError("confirmation model registry does not carry the V8 reserve")
    prior.update(_flatten(_registry(qualification_registries["fixture_only"])))
    prior.update(_flatten(_registry(qualification_registries["excluded_smoke"])))
    named_sets = {name: _flatten(registry) for name, registry in registries.items()}
    named_sets["prior_v1_v8_used"] = prior
    names = list(named_sets)
    overlaps = {
        f"{left}|{right}": sorted(named_sets[left] & named_sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if named_sets[left] & named_sets[right]
    }
    if overlaps:
        raise ValueError(f"development launch identifier overlap: {overlaps}")
    if len(registries["development"]["corpus"]) != int(
        value["analysis"]["development_corpus_count"]
    ):
        raise ValueError("development corpus count differs from frozen analysis")
    if len(registries["development"]["model"]) != int(
        value["analysis"]["stochastic_model_replicates"]
    ):
        raise ValueError("model replicate count differs from frozen analysis")
    value["resolved_registries"] = {
        **registries,
        "prior_v1_v8_used": sorted(prior),
    }
    value["gates"] = dict(value["qualification_gates"])
    return value


def require_frozen(config: Mapping[str, Any]) -> None:
    if config["protocol"]["status"] != "frozen":
        raise RuntimeError("one-shot development launch package is not frozen")


@dataclass(frozen=True, slots=True)
class IdentifierReference:
    role: str
    value: int


class IdentifierFirewall:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        allowed_purpose: str,
        development_authorized: bool = False,
    ):
        if allowed_purpose not in EXECUTABLE_PURPOSES:
            raise PermissionError(f"unsupported purpose: {allowed_purpose}")
        if allowed_purpose == "development" and not development_authorized:
            raise PermissionError("development requires sealed authorization")
        self.allowed_purpose = allowed_purpose
        self.development_authorized = bool(development_authorized)
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
        allowed = purpose == self.allowed_purpose
        reason = "allowed"
        if purpose == "development" and not self.development_authorized:
            allowed = False
            reason = "development-not-authorized"
        if purpose == "confirmation_reserve":
            allowed = False
            reason = "confirmation-never-supported-by-this-package"
        expected_registry = (
            self.registries[purpose]
            if purpose in EXECUTABLE_PURPOSES
            else {"corpus": [], "model": [], "inference": []}
        )
        forbidden = (
            _flatten(self.registries["confirmation_reserve"])
            | set(self.registries["prior_v1_v8_used"])
        )
        for reference in references:
            if reference.role not in {"corpus", "model", "inference"}:
                allowed = False
                reason = "unknown-role"
                break
            if int(reference.value) not in set(expected_registry[reference.role]):
                allowed = False
                reason = "identifier-not-in-purpose-role"
                break
            if int(reference.value) in forbidden:
                allowed = False
                reason = "blocked-prior-or-confirmation"
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


def manifest_for_files(
    root: str | Path, files: Iterable[str | Path]
) -> dict[str, Any]:
    base = Path(root).resolve()
    rows = []
    for item in sorted({Path(value).as_posix() for value in files}):
        path = base / item
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": item,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "files": rows,
        "file_count": len(rows),
        "digest": canonical_digest(rows),
    }


def verify_file_manifest(
    root: str | Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    base = Path(root).resolve()
    problems = []
    for row in manifest.get("files", []):
        path = base / str(row["path"])
        if not path.is_file():
            problems.append({"path": row["path"], "problem": "missing"})
            continue
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        expected = {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])}
        if observed != expected:
            problems.append(
                {
                    "path": row["path"],
                    "problem": "digest_or_size",
                    "expected": expected,
                    "observed": observed,
                }
            )
    digest_valid = manifest.get("digest") == canonical_digest(
        manifest.get("files", [])
    )
    return {
        "status": (
            "PASS"
            if not problems and manifest.get("files") and digest_valid
            else "FAIL"
        ),
        "checked_files": len(manifest.get("files", [])),
        "manifest_digest_valid": digest_valid,
        "problems": problems,
    }


def registry_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "registries": config["resolved_registries"],
        "development_corpus_count": len(
            config["resolved_registries"]["development"]["corpus"]
        ),
        "development_model_count": len(
            config["resolved_registries"]["development"]["model"]
        ),
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "confirmation_authorized": False,
    }
