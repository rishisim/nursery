#!/usr/bin/env python3
"""Derive the aggregate v1.3 protocol-freeze receipt from the public protocol.

No restricted inputs are accepted. The entry point is zero-argument and uses
only the fixed repository protocol document.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "docs/childlens_feasibility_v1_3/frozen_model_assisted_author_audit_protocol_v1_3.json"
)
OUTPUT = ROOT / "output/childlens_feasibility_v1_3/protocol_freeze_receipt.json"
GATE_MAP = {
    "boundary_matching": "speech_utterance_boundary",
    "source_role": "source_role_non_child_priority",
    "non_child_transcription": "non_child_transcript_error_and_coverage",
    "coarse_referential": "coarse_referential_status",
    "noun_object_candidates": "noun_object_candidate_precision_coverage",
    "verb_action_candidates": "verb_action_candidate_precision_coverage",
}

PROTOCOL_MODULE_PATH = ROOT / "scripts/childlens_author_audit_protocol_v1_3.py"
_SPEC = importlib.util.spec_from_file_location(
    "childlens_author_audit_protocol_v1_3_for_receipt", PROTOCOL_MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("E_PROTOCOL_VALIDATOR_IMPORT")
protocol_validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol_validator)


class ReceiptError(RuntimeError):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise ReceiptError("E_PUBLIC_PROTOCOL_NOT_FROZEN")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_receipt(protocol: dict[str, Any], document_sha256: str) -> dict[str, Any]:
    try:
        protocol_validator.validate_protocol(protocol)
    except protocol_validator.ProtocolError as exc:
        raise ReceiptError("E_PUBLIC_PROTOCOL_NOT_FROZEN") from exc
    _require(protocol.get("schema_version") == "childlens-model-assisted-author-audit-protocol-v1.3.0")
    _require(protocol.get("status") == "FROZEN_BEFORE_AUTHOR_LABELS")
    sampling = protocol.get("audit_sampling")
    author = protocol.get("author_audit")
    boundaries = protocol.get("scientific_boundaries")
    escalation = protocol.get("bounded_escalation")
    uncertainty = protocol.get("cluster_aware_uncertainty")
    gates = protocol.get("agreement_gates")
    freeze = protocol.get("freeze_binding")
    for value in (sampling, author, boundaries, escalation, uncertainty, gates, freeze):
        _require(isinstance(value, dict))
    _require(set(gates) == set(GATE_MAP))
    _require(all(
        isinstance(gate, dict)
        and all(name in gate for name in ("minimum_support", "pass_thresholds", "cluster_bound_floors", "hard_fail_thresholds"))
        for gate in gates.values()
    ))
    independence = sampling.get("selection_independence")
    _require(isinstance(independence, dict))
    _require(sampling.get("selected_item_count") == 15)
    _require(sampling.get("primary_target_total_ms") == 900_000)
    _require(sampling.get("reserve_target_total_ms") == 900_000)
    _require(sampling.get("reserve_frozen_with_primary_before_author_labels") is True)
    _require(sampling.get("selection_hash_deterministic") is True)
    _require(author.get("author_uses_raw_audio_video") is True)
    _require(author.get("author_blinded_to_predictions") is True)
    _require(author.get("author_lock_required_before_comparison") is True)
    _require(author.get("author_lock_mutable") is False)
    _require(author.get("second_human_required") is False)
    _require(author.get("adjudicator_required") is False)
    _require(author.get("inter_human_reliability_available") is False)
    _require(author.get("model_human_agreement_only") is True)
    _require(escalation.get("maximum_activations") == 1)
    _require(escalation.get("maximum_additional_speech_ms") == 900_000)
    _require(escalation.get("activation_condition") == "PRIMARY_BORDERLINE_ONLY")
    _require(uncertainty.get("cluster_unit") is not None)
    _require(uncertainty.get("small_sample_inference") is not None)
    _require(boundaries.get("hosted_model_content_access") is False)
    _require(boundaries.get("cloud_or_external_inference") is False)
    _require(boundaries.get("unaudited_pseudo_labels_are_primary_evaluation_truth") is False)
    _require(boundaries.get("simulator_oracle_labels_are_primary_causal_evaluation_truth") is True)
    frozen_digest = freeze.get("frozen_payload_sha256")
    _require(isinstance(frozen_digest, str) and len(frozen_digest) == 64)
    return {
        "schema_version": "childlens-v1.3-protocol-freeze-receipt-v1",
        "status": "FROZEN",
        "protocol_document_sha256": document_sha256,
        "frozen_payload_sha256": frozen_digest,
        "selected_item_count": 15,
        "primary_audit_speech_seconds": 900,
        "maximum_additional_speech_seconds": 900,
        "frozen_before_author_labels": True,
        "selection_hash_deterministic": True,
        "selection_independent_of_model_predictions": independence.get("model_predictions") is True,
        "selection_independent_of_model_confidence": independence.get("confidence") is True,
        "selection_independent_of_lexical_content": independence.get("lexical_content") is True,
        "selection_independent_of_visual_salience": independence.get("visual_salience") is True,
        "selection_independent_of_apparent_success": independence.get("apparent_success") is True,
        "author_uses_raw_audio_video": True,
        "author_blinded_to_model_predictions": True,
        "author_record_lock_required_before_prediction_reveal": True,
        "author_record_lock_irreversible": True,
        "bounded_escalation_only_if_borderline": True,
        "thresholds_frozen_before_author_labels": True,
        "cluster_aware_uncertainty_required": True,
        "small_sample_limitation_required": True,
        "pseudo_labels_for_aggregate_calibration_only": True,
        "simulator_oracle_labels_primary_evaluation_truth": True,
        "second_human_required": False,
        "adjudicator_required": False,
        "inter_human_reliability_available": False,
        "unaudited_pseudo_labels_primary_evaluation_truth": False,
        "agreement_target": "MODEL_HUMAN_NOT_INTER_ANNOTATOR",
        "thresholds": {
            output_name: {
                "frozen_before_author_labels": True,
                "pass_rule": "MEET_MINIMUM_SUPPORT_AND_ALL_POINT_AND_CLUSTER_BOUND_PASS_THRESHOLDS",
                "frozen_gate_digest": hashlib.sha256(
                    json.dumps(gates[input_name], sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
            for input_name, output_name in GATE_MAP.items()
        },
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".protocol-freeze-v1-3-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    if len(sys.argv) != 1:
        print("E_NO_ARGUMENTS", file=sys.stderr)
        return 2
    try:
        info = PROTOCOL.lstat()
        _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode))
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        _require(isinstance(protocol, dict))
        receipt = build_receipt(protocol, _sha256(PROTOCOL))
    except (OSError, UnicodeError, json.JSONDecodeError, ReceiptError):
        print("E_PUBLIC_PROTOCOL_NOT_FROZEN", file=sys.stderr)
        return 2
    _atomic_write(OUTPUT, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": "ok", "receipt_schema": receipt["schema_version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
