#!/usr/bin/env python3
"""Validate the repository-safe ChildLens v1.3 protocol freeze.

This module handles generic protocol metadata only.  The authoritative
restricted sampler is ``prepare_childlens_author_audit_packet_v1_3.py``.  This
validator neither discovers a quarantine nor accepts item rows or intervals.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "childlens-model-assisted-author-audit-protocol-v1.3.0"
PUBLIC_RECEIPT_SCHEMA_VERSION = "childlens-v1.3-protocol-freeze-receipt-v1"
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "childlens_feasibility_v1_3"
    / "frozen_model_assisted_author_audit_protocol_v1_3.json"
)


class ProtocolError(ValueError):
    """Fail-closed protocol error containing no restricted value."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _frozen_payload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    names = protocol.get("freeze_binding", {}).get("frozen_top_level_fields")
    if not isinstance(names, list) or not names:
        raise ProtocolError("E_PROTOCOL_FREEZE_FIELDS")
    try:
        return {name: protocol[name] for name in names}
    except KeyError as exc:
        raise ProtocolError("E_PROTOCOL_FROZEN_FIELD_MISSING") from exc


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("E_PROTOCOL_SCHEMA")
    freeze = protocol.get("freeze_binding")
    if not isinstance(freeze, Mapping):
        raise ProtocolError("E_PROTOCOL_FREEZE")
    expected = freeze.get("frozen_payload_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ProtocolError("E_PROTOCOL_FREEZE_DIGEST")
    if sha256_json(_frozen_payload(protocol)) != expected:
        raise ProtocolError("E_PROTOCOL_MUTATED")

    sampling = protocol.get("audit_sampling", {})
    required_sampling = {
        "selected_item_count": 15,
        "primary_target_total_ms": 900_000,
        "reserve_target_total_ms": 900_000,
        "reserve_max_total_ms": 900_000,
        "preferred_per_item_ms": 60_000,
        "minimum_per_item_per_sample_ms": 1_000,
    }
    if any(sampling.get(key) != value for key, value in required_sampling.items()):
        raise ProtocolError("E_PROTOCOL_SAMPLING_CONTRACT")
    if set(sampling.get("permitted_item_input_fields", [])) != {
        "blinded_item_key",
        "media_relpath",
        "expected_size_bytes",
        "expected_media_sha256",
        "reference_duration_seconds",
        "reference_duration_basis",
        "speech_presence_expectation",
        "speech_windows",
    }:
        raise ProtocolError("E_PROTOCOL_SAMPLING_ALLOWLIST")
    independence = sampling.get("selection_independence", {})
    for key in (
        "model_predictions",
        "confidence",
        "lexical_content",
        "transcript_text",
        "visual_content",
        "visual_salience",
        "referential_status",
        "apparent_success",
        "learner_outcome",
    ):
        if independence.get(key) is not True:
            raise ProtocolError("E_PROTOCOL_SELECTION_INDEPENDENCE")

    escalation = protocol.get("bounded_escalation", {})
    if escalation.get("maximum_activations") != 1:
        raise ProtocolError("E_PROTOCOL_ESCALATION_COUNT")
    if escalation.get("maximum_additional_speech_ms") != 900_000:
        raise ProtocolError("E_PROTOCOL_ESCALATION_DURATION")
    if escalation.get("activation_condition") != "PRIMARY_BORDERLINE_ONLY":
        raise ProtocolError("E_PROTOCOL_ESCALATION_CONDITION")
    if escalation.get("sample_source") != "PREFROZEN_DISJOINT_BORDERLINE_RESERVE":
        raise ProtocolError("E_PROTOCOL_ESCALATION_SAMPLE")

    author = protocol.get("author_audit", {})
    author_expectations = {
        "author_blinded_to_predictions": True,
        "prediction_store_mounted_in_author_app_before_lock": False,
        "author_lock_required_before_comparison": True,
        "author_lock_mutable": False,
        "comparison_before_author_lock": False,
        "second_human_required": False,
        "adjudicator_required": False,
        "inter_human_reliability_available": False,
        "model_human_agreement_only": True,
    }
    if any(author.get(key) is not value for key, value in author_expectations.items()):
        raise ProtocolError("E_PROTOCOL_AUTHOR_BOUNDARY")

    boundaries = protocol.get("scientific_boundaries", {})
    required_false = (
        "hosted_model_content_access",
        "cloud_or_external_inference",
        "pseudo_labels_are_ground_truth",
        "unaudited_pseudo_labels_are_primary_evaluation_truth",
        "instrument_ancestry_enters_learner",
        "scientific_learner_outcome_authorized",
    )
    if any(boundaries.get(name) is not False for name in required_false):
        raise ProtocolError("E_PROTOCOL_SCIENTIFIC_BOUNDARY")
    if boundaries.get("simulator_oracle_labels_are_primary_causal_evaluation_truth") is not True:
        raise ProtocolError("E_PROTOCOL_ORACLE_BOUNDARY")

    expected_gates = {
        "boundary_matching",
        "source_role",
        "non_child_transcription",
        "coarse_referential",
        "noun_object_candidates",
        "verb_action_candidates",
    }
    gates = protocol.get("agreement_gates")
    if not isinstance(gates, Mapping) or set(gates) != expected_gates:
        raise ProtocolError("E_PROTOCOL_GATE_REGISTRY")
    for gate in gates.values():
        if not isinstance(gate, Mapping) or not {
            "minimum_support",
            "pass_thresholds",
            "cluster_bound_floors",
            "hard_fail_thresholds",
        }.issubset(gate):
            raise ProtocolError("E_PROTOCOL_GATE_SHAPE")


def load_protocol(path: Path | str = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("E_PROTOCOL_READ") from exc
    if not isinstance(value, dict):
        raise ProtocolError("E_PROTOCOL_TYPE")
    validate_protocol(value)
    return value


def build_public_protocol_receipt(
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return policy targets only; observed sample facts need the runtime receipt."""

    if protocol is None:
        protocol = load_protocol()
    else:
        validate_protocol(protocol)
    sampling = protocol["audit_sampling"]
    author = protocol["author_audit"]
    escalation = protocol["bounded_escalation"]
    independence = sampling["selection_independence"]
    return {
        "schema_version": PUBLIC_RECEIPT_SCHEMA_VERSION,
        "protocol_frozen_payload_sha256": protocol["freeze_binding"][
            "frozen_payload_sha256"
        ],
        "frozen_before_human_labels": True,
        "target_selected_item_count": sampling["selected_item_count"],
        "target_selected_speech_minutes": sampling["primary_target_total_ms"] / 60_000,
        "target_reserve_speech_minutes": sampling["reserve_target_total_ms"] / 60_000,
        "one_minute_per_item_preferred": True,
        "minimum_seconds_per_item_per_sample": sampling[
            "minimum_per_item_per_sample_ms"
        ]
        / 1_000,
        "selection_hash_deterministic": sampling["selection_hash_deterministic"],
        "selection_independent_of_model_predictions": independence["model_predictions"],
        "selection_independent_of_confidence": independence["confidence"],
        "selection_independent_of_lexical_content": independence["lexical_content"],
        "selection_independent_of_visual_salience": independence["visual_salience"],
        "selection_independent_of_apparent_success": independence["apparent_success"],
        "author_blinded_to_predictions": author["author_blinded_to_predictions"],
        "author_lock_required_before_comparison": author[
            "author_lock_required_before_comparison"
        ],
        "second_human_required": author["second_human_required"],
        "inter_human_reliability_available": author[
            "inter_human_reliability_available"
        ],
        "model_human_agreement_only": author["model_human_agreement_only"],
        "bounded_escalation_max_additional_speech_minutes": escalation[
            "maximum_additional_speech_ms"
        ]
        / 60_000,
        "escalation_only_if_borderline": True,
        "thresholds_frozen": True,
        "threshold_registry_sha256": sha256_json(protocol["agreement_gates"]),
        "observed_sample_receipt_required_separately": True,
    }


if __name__ == "__main__":
    validated = load_protocol()
    print(
        json.dumps(
            {
                "status": "PROTOCOL_VALID",
                "schema_version": validated["schema_version"],
                "frozen_payload_sha256": validated["freeze_binding"][
                    "frozen_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
