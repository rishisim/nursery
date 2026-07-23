"""Fail-closed child-only provenance and corpus-isolation policy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any


CONTRACT_VERSION = "child-only-scientific-contract-v1.0.0"
SCHEMA_VERSION = "child-only-episode-v1.0.0"
CONSTRUCTION_PROFILE = "NONSCIENTIFIC_CONSTRUCTION_FIXTURE"
CHILD_CORPORA = frozenset({"BABYVIEW", "CHILDLENS"})
ALLOWED_SOURCE_FAMILIES = frozenset(
    {"SYNTHETIC_FIXTURE", "BABYVIEW", "CHILDLENS", "OFFICIAL_CODE_REFERENCE"}
)

# These tokens are checked as words/path components to avoid false positives
# such as the letters "aria" inside an unrelated longer word.
_FORBIDDEN_SOURCE_PATTERNS = (
    re.compile(r"(^|[^a-z0-9])aea([^a-z0-9]|$)", re.IGNORECASE),
    re.compile(r"aria[ _-]*everyday[ _-]*activities", re.IGNORECASE),
    re.compile(r"adult[ _-]*sensor[ _-]*analogue", re.IGNORECASE),
)


class PolicyViolation(ValueError):
    """Raised before I/O or modeling when provenance is unsafe or incomplete."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _walk_strings(child)


def reject_forbidden_source_markers(value: Any) -> None:
    """Reject any adult-data identifier without opening referenced paths."""

    for text in _walk_strings(value):
        if any(pattern.search(text) for pattern in _FORBIDDEN_SOURCE_PATTERNS):
            raise PolicyViolation("forbidden adult-data source marker in child-only provenance")


def _require_exact_keys(record: Mapping[str, Any], expected: set[str], context: str) -> None:
    observed = set(record)
    if observed != expected:
        raise PolicyViolation(
            f"{context} keys must be exactly {sorted(expected)}; got {sorted(observed)}"
        )


def _validate_artifact_nodes(
    nodes: list[Mapping[str, Any]],
    *,
    selected_corpus: str | None,
    claim_tier: str,
    study_instance_id: str,
) -> None:
    expected = {
        "artifact_id",
        "digest",
        "role",
        "source_family",
        "study_instance_id",
        "corpus_instance_id",
        "parents",
        "claim_tier",
    }
    by_id: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        _require_exact_keys(node, expected, "artifact node")
        artifact_id = str(node["artifact_id"])
        if not artifact_id or artifact_id in by_id:
            raise PolicyViolation("artifact IDs must be nonempty and unique")
        if str(node["source_family"]) not in ALLOWED_SOURCE_FAMILIES:
            raise PolicyViolation("unknown source family")
        if str(node["claim_tier"]) != claim_tier:
            raise PolicyViolation("artifact claim tier differs from study claim tier")
        if str(node["study_instance_id"]) != study_instance_id:
            raise PolicyViolation("artifact belongs to a different study instance")
        if not re.fullmatch(r"[0-9a-f]{64}", str(node["digest"])):
            raise PolicyViolation("artifact digest must be lowercase SHA-256")
        by_id[artifact_id] = node

    for node in nodes:
        for parent in node["parents"]:
            if parent not in by_id:
                raise PolicyViolation("provenance parent is outside the declared DAG")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in visiting:
            raise PolicyViolation("provenance ancestry contains a cycle")
        if artifact_id in visited:
            return
        visiting.add(artifact_id)
        for parent in by_id[artifact_id]["parents"]:
            visit(str(parent))
        visiting.remove(artifact_id)
        visited.add(artifact_id)

    for artifact_id in by_id:
        visit(artifact_id)

    child_sources = {
        str(node["source_family"])
        for node in nodes
        if str(node["source_family"]) in CHILD_CORPORA
    }
    if selected_corpus is None:
        if child_sources:
            raise PolicyViolation("construction provenance cannot contain child empirical artifacts")
        if any(str(node["source_family"]) != "SYNTHETIC_FIXTURE" for node in nodes):
            raise PolicyViolation("construction artifacts may only derive from synthetic fixtures")
    else:
        if child_sources != {selected_corpus}:
            raise PolicyViolation("scientific provenance must resolve to exactly the selected child corpus")
        for node in nodes:
            source = str(node["source_family"])
            corpus_instance_id = node["corpus_instance_id"]
            if source in CHILD_CORPORA and not corpus_instance_id:
                raise PolicyViolation("child empirical artifacts require a corpus-instance ID")


def validate_provenance(document: Mapping[str, Any]) -> None:
    """Validate a complete provenance document against the v1 contract.

    Construction is the only valid state before restricted access and explicit
    corpus selection.  A scientific document must bind exactly one of BabyView
    or ChildLens and cannot contain cross-corpus ancestry.
    """

    reject_forbidden_source_markers(document)
    expected = {
        "contract_version",
        "policy_version",
        "study_instance_id",
        "selected_corpus",
        "restricted_access_available",
        "claim_tier",
        "profile_label",
        "scientific_outcome_authorized",
        "audio_tts_status",
        "primary_learner",
        "artifact_nodes",
    }
    _require_exact_keys(document, expected, "provenance document")
    if document["contract_version"] != CONTRACT_VERSION:
        raise PolicyViolation("wrong child-only contract version")
    if document["policy_version"] != "child-only-provenance-policy-v1.0.0":
        raise PolicyViolation("wrong provenance policy version")
    if document["audio_tts_status"] != "DEFERRED":
        raise PolicyViolation("audio/TTS is deferred in v1")
    if document["primary_learner"] != "TEMPORAL_CLIP_PLUS_SCRATCH":
        raise PolicyViolation("the v1 primary learner must be temporal CLIP+ from scratch")
    if document["scientific_outcome_authorized"] is not False:
        raise PolicyViolation("this construction stage never authorizes an acquisition outcome")

    selected = document["selected_corpus"]
    if selected is not None and selected not in CHILD_CORPORA:
        raise PolicyViolation("selected_corpus must be null, BABYVIEW, or CHILDLENS")
    access = document["restricted_access_available"]
    if not isinstance(access, bool):
        raise PolicyViolation("restricted_access_available must be boolean")

    claim_tier = str(document["claim_tier"])
    profile = document["profile_label"]
    if selected is None:
        if access or claim_tier != "CONSTRUCTION_ONLY" or profile != CONSTRUCTION_PROFILE:
            raise PolicyViolation("unselected pre-access work must remain a labeled construction fixture")
    else:
        if not access:
            raise PolicyViolation("a scientific corpus cannot be selected before access is available")
        if claim_tier != "SCIENTIFIC_CHILD_ONLY" or profile is not None:
            raise PolicyViolation("selected-corpus work must use the child-only scientific claim tier")

    nodes = document["artifact_nodes"]
    if not isinstance(nodes, list) or not nodes:
        raise PolicyViolation("provenance requires at least one artifact node")
    _validate_artifact_nodes(
        nodes,
        selected_corpus=selected,
        claim_tier=claim_tier,
        study_instance_id=str(document["study_instance_id"]),
    )


def validate_fresh_corpus_initializations(receipts: list[Mapping[str, Any]]) -> None:
    """Reject shared tokenizer/checkpoint lineage across corpus instances.

    Tensor values may coincide under paired seeds; independent construction and
    nonshared lineage, rather than forced numerical inequality, is the gate.
    """

    expected = {
        "selected_corpus",
        "corpus_instance_id",
        "tokenizer_artifact_id",
        "tokenizer_training_corpus_instance_id",
        "model_artifact_id",
        "model_initialization",
        "parent_checkpoint",
        "construction_receipt_id",
    }
    seen_artifacts: set[str] = set()
    seen_receipts: set[str] = set()
    for receipt in receipts:
        reject_forbidden_source_markers(receipt)
        _require_exact_keys(receipt, expected, "initialization receipt")
        corpus = str(receipt["selected_corpus"])
        instance = str(receipt["corpus_instance_id"])
        if corpus not in CHILD_CORPORA or not instance:
            raise PolicyViolation("initialization receipt needs one child corpus instance")
        if receipt["tokenizer_training_corpus_instance_id"] != instance:
            raise PolicyViolation("tokenizer must be trained only on its selected corpus instance")
        if receipt["model_initialization"] != "SCRATCH" or receipt["parent_checkpoint"] is not None:
            raise PolicyViolation("scientific learner must initialize from scratch with no parent checkpoint")
        artifact_ids = {str(receipt["tokenizer_artifact_id"]), str(receipt["model_artifact_id"])}
        if seen_artifacts.intersection(artifact_ids):
            raise PolicyViolation("tokenizer/model artifacts cannot be shared across corpus instances")
        receipt_id = str(receipt["construction_receipt_id"])
        if receipt_id in seen_receipts:
            raise PolicyViolation("initialization receipts must be independently created")
        seen_artifacts.update(artifact_ids)
        seen_receipts.add(receipt_id)


def load_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("policy_version") != "child-only-provenance-policy-v1.0.0":
        raise PolicyViolation("wrong provenance policy version")
    forbidden = document.get("forbidden_adult_data_transfer", {})
    if forbidden.get("fail_closed") is not True or set(forbidden.get("forbidden_source_identifiers", [])) != {
        "AEA",
        "ARIA_EVERYDAY_ACTIVITIES",
    }:
        raise PolicyViolation("machine-readable policy does not explicitly fail closed on AEA")
    one_corpus = document.get("one_corpus_at_a_time", {})
    if one_corpus.get("exactly_one_for_scientific_instance") is not True:
        raise PolicyViolation("machine-readable policy does not enforce one child corpus at a time")
    return document
