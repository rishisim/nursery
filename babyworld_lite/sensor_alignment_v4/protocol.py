from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

PROTOCOL_ID = "synthetic-benchmark-protocol-fidelity-v4"
REGISTRIES = (
    "fixture_only",
    "excluded_smoke",
    "future_development",
    "confirmation_reserve",
)
OPERATIONS = (
    "generate",
    "detector_infer",
    "fit",
    "predict",
    "score",
    "read",
    "summarize",
    "control",
    "infer",
)
FORBIDDEN_VISIBLE_KEYS = frozenset(
    {
        "answer",
        "answers",
        "event_owners",
        "grounded",
        "intended_concept",
        "intended_index",
        "lexicon",
        "lexicon_oracle",
        "oracle",
        "semantic_label",
        "target_event_index",
        "target_index",
        "truth",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(
    path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False
) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )


def _registry_values(config: Mapping[str, Any], name: str) -> dict[str, set[int]]:
    return {
        role: set(map(int, values))
        for role, values in config["seeds"][name].items()
    }


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if config["protocol"]["id"] != PROTOCOL_ID:
        raise ValueError("wrong v4 protocol id")
    if config["protocol"]["status"] not in {"ready_to_freeze", "frozen"}:
        raise ValueError("v4 protocol is neither ready nor frozen")
    sets: list[tuple[str, set[int]]] = []
    for name in REGISTRIES:
        role_sets = _registry_values(config, name)
        flattened = set().union(*role_sets.values())
        if sum(len(values) for values in role_sets.values()) != len(flattened):
            raise ValueError(f"roles overlap within {name}")
        sets.append((name, flattened))
    prior = set(map(int, config["seeds"]["prior_v1_v3"]))
    sets.append(("prior_v1_v3", prior))
    for index, (name, values) in enumerate(sets):
        for other_name, other_values in sets[:index]:
            overlap = values & other_values
            if overlap:
                raise ValueError(f"seed registries overlap: {name}/{other_name}: {sorted(overlap)}")
    detector_path = Path(path).resolve().parent.parent / config["detector"]["frozen_v2_path"]
    observed = sha256_file(detector_path)
    expected = str(config["detector"]["frozen_v2_sha256"])
    if observed != expected:
        raise ValueError(f"frozen detector hash mismatch: {observed} != {expected}")
    return config


@dataclass(frozen=True)
class SeedReference:
    role: str
    seed: int


class SeedFirewall:
    def __init__(self, config: Mapping[str, Any], *, allowed_purposes: Sequence[str]):
        unknown = set(allowed_purposes) - {"fixture_only", "excluded_smoke"}
        if unknown:
            raise ValueError(f"unsafe allowed purpose: {sorted(unknown)}")
        self.config = config
        self.allowed_purposes = frozenset(allowed_purposes)
        self.log: list[dict[str, Any]] = []

    def authorize(
        self,
        operation: str,
        references: Sequence[SeedReference],
        *,
        purpose: str,
    ) -> str:
        refs = [SeedReference(str(ref.role), int(ref.seed)) for ref in references]
        reason = "allowed"
        allowed = True
        if operation not in OPERATIONS:
            allowed = False
            reason = "unknown_operation"
        elif purpose not in self.allowed_purposes:
            allowed = False
            reason = "purpose_not_authorized_in_this_process"
        elif purpose not in {"fixture_only", "excluded_smoke"}:
            allowed = False
            reason = "outcome_registry_forbidden_during_qualification"
        else:
            registry = _registry_values(self.config, purpose)
            for ref in refs:
                if ref.role not in registry or ref.seed not in registry[ref.role]:
                    allowed = False
                    reason = f"seed_not_in_{purpose}_{ref.role}_registry"
                    break
        record = {
            "sequence": len(self.log),
            "operation": operation,
            "purpose": purpose,
            "references": [
                {"role": ref.role, "seed": ref.seed} for ref in refs
            ],
            "status": "ALLOW" if allowed else "BLOCK",
            "reason": reason,
        }
        self.log.append(record)
        if not allowed:
            raise PermissionError(
                f"v4 seed firewall blocked {operation}/{purpose}: {reason}: "
                f"{[(ref.role, ref.seed) for ref in refs]}"
            )
        return canonical_digest(record)


def reject_oracle_fields(value: Any, *, path: str = "visible") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_VISIBLE_KEYS or "oracle" in normalized:
                raise ValueError(f"forbidden oracle field at {path}.{key}")
            reject_oracle_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_oracle_fields(child, path=f"{path}[{index}]")


def registry_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "registries": {
            name: {
                role: list(map(int, values))
                for role, values in config["seeds"][name].items()
            }
            for name in REGISTRIES
        },
        "prior_v1_v3": list(map(int, config["seeds"]["prior_v1_v3"])),
        "allowed_during_qualification": ["fixture_only", "excluded_smoke"],
        "blocked_during_this_task": [
            "future_development",
            "confirmation_reserve",
            "prior_v1_v3",
        ],
        "operations": list(OPERATIONS),
    }
