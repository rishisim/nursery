from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import PROTOCOL_ID, canonical_digest, write_json

REQUIRED_GATES = (
    "v8_version_ready",
    "v8_adversarial_audit_pass",
    "fixture_rehearsal_pass",
    "excluded_rehearsal_pass",
    "exact_runner_independent_reproduction",
    "negative_source_and_input_mutations",
    "official_tests",
    "freeze_integrity",
    "prior_v1_v8_preservation",
    "identifier_registry_reconciliation",
    "development_registry_has_40_corpora",
    "three_stochastic_model_replicates",
    "inference_contract_frozen",
    "cue_free_endpoints_frozen",
    "side_modality_firewall",
    "confirmation_command_absent",
    "development_outcome_count_zero",
    "confirmation_outcome_count_zero",
    "one_shot_output_absent",
    "one_shot_command_frozen",
    "traceability",
    "complete_manifest",
    "terminal_regeneration",
)


def package_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    gates = dict(value.get("gates", {}))
    missing = sorted(set(REQUIRED_GATES) - set(gates))
    unexpected = sorted(set(gates) - set(REQUIRED_GATES))
    contradictions = list(value.get("contradictions", []))
    stale = list(value.get("stale_inputs", []))
    well_formed = not missing and not unexpected and not contradictions and not stale
    passed = well_formed and all(gates.get(name) is True for name in REQUIRED_GATES)
    return {
        "decision": "DEVELOPMENT_PACKAGE_SEALED" if passed else "REVISE",
        "protocol_id": PROTOCOL_ID,
        "claim_scope": "narrow_raw_sensor_assisted_weak_lexical_action_grounding",
        "infant_learning_claim_authorized": False,
        "ecological_validity_claim_authorized": False,
        "development_authorized": passed,
        "confirmation_authorized": False,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "gates": gates,
        "input_contract": {
            "well_formed": well_formed,
            "missing_gates": missing,
            "unexpected_gates": unexpected,
            "contradictions": contradictions,
            "stale_inputs": stale,
        },
        "reason": (
            "Every frozen launch-package gate passed; a later separate one-shot development run is authorized."
            if passed
            else "At least one launch-package gate failed or the adjudication input was incomplete."
        ),
    }


def adjudicate_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(input_path).read_text())
    decision = package_decision(value)
    write_json(output_path, decision, overwrite=Path(output_path).exists())
    return decision


def authorization_payload(
    *,
    terminal_sha256: str,
    freeze_receipt_sha256: str,
    snapshot_manifest_sha256: str,
    excluded_rehearsal_sha256: str,
    adversarial_audit_sha256: str,
    official_tests_sha256: str,
    exact_command: str,
    development_registry: Mapping[str, Any],
    confirmation_registry: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "nursery-development-launch-ready-v2",
        "protocol_id": PROTOCOL_ID,
        "status": "DEVELOPMENT_LAUNCH_READY",
        "development_authorized": True,
        "confirmation_authorized": False,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "claim_scope": "narrow_raw_sensor_assisted_weak_lexical_action_grounding",
        "infant_learning_claim_authorized": False,
        "ecological_validity_claim_authorized": False,
        "package_terminal_sha256": terminal_sha256,
        "freeze_receipt_sha256": freeze_receipt_sha256,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "excluded_rehearsal_sha256": excluded_rehearsal_sha256,
        "adversarial_audit_sha256": adversarial_audit_sha256,
        "official_tests_sha256": official_tests_sha256,
        "development_registry": dict(development_registry),
        "confirmation_reserve": dict(confirmation_registry),
        "exact_one_shot_command": exact_command,
        "authorization_scope": "one later development cohort only",
        "confirmation_requires_new_separate_authorization": True,
    }
    return {
        **payload,
        "authorization_digest": canonical_digest(payload),
    }


def verify_authorization(value: Mapping[str, Any]) -> bool:
    payload = {
        key: child
        for key, child in value.items()
        if key != "authorization_digest"
    }
    return (
        value.get("status") == "DEVELOPMENT_LAUNCH_READY"
        and value.get("development_authorized") is True
        and value.get("confirmation_authorized") is False
        and int(value.get("development_outcome_count", -1)) == 0
        and int(value.get("confirmation_outcome_count", -1)) == 0
        and value.get("authorization_digest") == canonical_digest(payload)
    )

