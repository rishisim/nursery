from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import canonical_bytes

REQUIRED_GATES = (
    "actual_frozen_v2_detector_runtime",
    "all_conditions_executed",
    "complete_manifest",
    "condition_design",
    "confirmation_outcome_count_zero",
    "corpus_design",
    "detector_capacity_predeclared_bounds",
    "direct_capacity_control_executed",
    "factor_refits_executed",
    "freeze_integrity",
    "future_development_outcome_count_zero",
    "identifiability_sensor_free_genuine_fit",
    "independent_recompute",
    "joint_iterations_and_competition_numeric",
    "negative_recompute_mutations",
    "no_projection_or_answer_key_fit",
    "noun_sensor_weight_zero",
    "official_tests",
    "oracle_control_executed",
    "order_invariance",
    "prior_v1_v4_preservation",
    "scientific_outcome_count_zero",
    "seed_firewall",
    "sensor_corruption_control_executed",
    "sensor_helpful_not_required",
    "side_modalities_absent_at_language_evaluation",
    "terminal_regeneration",
    "traceability",
)


def terminal_decision(adjudication_inputs: Mapping[str, Any]) -> dict[str, Any]:
    supplied = adjudication_inputs.get("gates", {})
    missing = sorted(set(REQUIRED_GATES) - set(supplied))
    unexpected = sorted(set(supplied) - set(REQUIRED_GATES))
    contradictions = list(adjudication_inputs.get("contradictions", []))
    stale_inputs = list(adjudication_inputs.get("stale_inputs", []))
    well_formed = not missing and not unexpected and not contradictions and not stale_inputs
    gates = {name: bool(supplied.get(name, False)) for name in REQUIRED_GATES}
    all_pass = well_formed and all(gates.values())
    structural_failure = bool(adjudication_inputs.get("identifiability_structural_failure", False))
    if all_pass:
        decision = "V5_READY"
        reason = "Every frozen package-qualification gate passed. This is package readiness only."
    elif structural_failure or bool(adjudication_inputs.get("oracle_or_manufactured_scoring_required", False)):
        decision = "STOP"
        reason = "The benchmark is structurally unidentifiable without oracle leakage or manufactured scoring."
    else:
        decision = "REVISE"
        reason = "At least one repairable frozen package-qualification gate failed or was missing, stale, or contradictory."
    return {
        "decision": decision,
        "protocol_id": "synthetic-identifiability-qualification-v5",
        "package_readiness_only": True,
        "scientific_go": False,
        "future_development_authorized": False,
        "confirmation_authorized": False,
        "scientific_outcome_authorized": False,
        "gates": gates,
        "input_contract": {
            "well_formed": well_formed,
            "missing_gates": missing,
            "unexpected_gates": unexpected,
            "contradictions": contradictions,
            "stale_inputs": stale_inputs,
        },
        "identifiability_structural_failure": structural_failure,
        "reason": reason,
    }


def adjudicate_file(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(input_path).read_text())
    decision = terminal_decision(value)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_bytes(decision))
    return decision
