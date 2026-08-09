#!/usr/bin/env python3
"""Fail-closed validation for public Nursery convergence artifacts.

This module intentionally accepts only repository-safe protocol, decision, and
aggregate receipt documents.  It has no quarantine discovery or media-reading
code and must never be given a restricted manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_DECISION = (
    "REUSE_EXISTING_SYNTHETIC_MECHANISM_PROOF_AND_FREEZE_MINIMAL_REALISM_EXTENSION"
)
SUPERSEDED_DECISION = (
    "EXISTING_SYNTHETIC_EVIDENCE_INADEQUATE_NEW_MINIMAL_PROTOCOL_REQUIRED"
)
ACTIVE_PROTOCOL = (
    "docs/nursery_program_convergence_v1/"
    "frozen_calibration_conditioned_sensitivity_protocol.json"
)
PSEUDO_PROTOCOL = (
    "docs/nursery_program_convergence_v1/"
    "frozen_childlens_pseudo_calibration_protocol.json"
)
SUPERSEDED_PROTOCOL = (
    "docs/nursery_program_convergence_v1/frozen_minimal_realism_protocol.json"
)
LEXICAL_STOP = "CHILDLENS_LEXICAL_FEASIBILITY_STOP_CURRENT_RESOURCES"
ANCESTRY_MODE = "CHILDLENS_AGGREGATE_CALIBRATED_SEPARATE_PROTOCOL"
SUPPRESSED_K5 = "SUPPRESSED_K5"

HISTORICAL_REFERENCES = {
    "v1": "docs/childlens_feasibility_v1/executive_decision_report.md",
    "v1_1": "output/childlens_feasibility_v1_1/decision_record.json",
    "v1_2": "output/childlens_feasibility_v1_2/decision_record.json",
    "v1_3": "output/childlens_feasibility_v1_3/decision_record.json",
    "v1_3_1": "output/childlens_feasibility_v1_3_1/decision_record.json",
}
IMMUTABILITY_RECEIPT = (
    "output/childlens_feasibility_v1_3_1/immutability_receipt.json"
)


@dataclass(frozen=True)
class Issue:
    code: str
    location: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "location": self.location,
            "message": self.message,
        }


def _issue(issues: list[Issue], code: str, location: str, message: str) -> None:
    issues.append(Issue(code=code, location=location, message=message))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _false(document: Mapping[str, Any], key: str) -> bool:
    return document.get(key) is False


def round_interval_outward(
    lower: float,
    upper: float,
    step: float,
    *,
    clamp: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Round an interval conservatively to a fixed grid.

    Decimal arithmetic avoids binary-float cases that could accidentally round
    an upper bound inward.
    """

    values = (lower, upper, step, clamp[0], clamp[1])
    if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in values):
        raise ValueError("interval, step, and clamp values must be finite")
    if step <= 0 or lower > upper or clamp[0] > clamp[1]:
        raise ValueError("invalid interval, step, or clamp")

    lo = Decimal(str(lower))
    hi = Decimal(str(upper))
    quantum = Decimal(str(step))
    floor_value = (lo / quantum).to_integral_value(rounding=ROUND_FLOOR) * quantum
    ceiling_value = (hi / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum
    floor_value = max(floor_value, Decimal(str(clamp[0])))
    ceiling_value = min(ceiling_value, Decimal(str(clamp[1])))
    return float(floor_value), float(ceiling_value)


def categorical_k5_disposition(
    counts: Mapping[str, int], *, minimum_k: int = 5
) -> str:
    """Return a field-level publication disposition with complementary suppression."""

    if minimum_k != 5 or not counts:
        return SUPPRESSED_K5
    if any(type(value) is not int or value < 0 for value in counts.values()):
        return SUPPRESSED_K5
    total = sum(counts.values())
    if total < minimum_k:
        return SUPPRESSED_K5
    for value in counts.values():
        complement = total - value
        if 0 < value < minimum_k or 0 < complement < minimum_k:
            return SUPPRESSED_K5
    return "PUBLISH_OUTWARD_ROUNDED_INTERVALS_ONLY"


def _validate_two_paths(pseudo: Mapping[str, Any], issues: list[Issue]) -> None:
    paths = _mapping(pseudo.get("instrument_paths"))
    for family in ("speech_timing_and_text", "referential_and_visual"):
        rows = paths.get(family)
        location = f"pseudo.instrument_paths.{family}"
        if not isinstance(rows, list):
            _issue(issues, "PATH_FAMILY_MISSING", location, "required path family is absent")
            continue
        required = [row for row in rows if isinstance(row, Mapping) and row.get("required") is True]
        ids = [row.get("id") for row in required]
        families = [row.get("family_id") for row in required]
        if len(required) < 2:
            _issue(issues, "PATH_COUNT_BELOW_TWO", location, "two active required paths are mandatory")
        if any(not isinstance(value, str) or not value for value in ids + families):
            _issue(issues, "PATH_ID_OR_FAMILY_MISSING", location, "every required path needs id and family_id")
        if len(set(ids)) != len(ids) or len(set(families)) != len(families):
            _issue(issues, "PATHS_NOT_DISTINCT", location, "required path and family identifiers must be distinct")
        if any(row.get("output_status") != "pseudo_label_not_ground_truth" for row in required):
            _issue(issues, "PATH_OUTPUT_SEMANTICS", location, "every path must be marked pseudo-label, not ground truth")

    if paths.get("hosted_models") != "forbidden" or paths.get("restricted_inference_network") != "denied":
        _issue(issues, "HOSTED_OR_NETWORK_PATH", "pseudo.instrument_paths", "hosted inference must be forbidden and restricted inference network-denied")


def validate_public_artifacts(
    decision: Mapping[str, Any],
    active: Mapping[str, Any],
    superseded: Mapping[str, Any],
    pseudo: Mapping[str, Any],
    immutability: Mapping[str, Any],
) -> list[Issue]:
    issues: list[Issue] = []

    if decision.get("schema_version") != "nursery-program-convergence-decision-v2":
        _issue(issues, "DECISION_SCHEMA", "decision.schema_version", "unexpected convergence decision schema")
    if decision.get("decision") != EXPECTED_DECISION:
        _issue(issues, "DECISION_LITERAL", "decision.decision", "exact revised decision literal is required")
    if decision.get("supersedes_convergence_decision") != SUPERSEDED_DECISION:
        _issue(issues, "SUPERSEDED_DECISION_LITERAL", "decision.supersedes_convergence_decision", "exact predecessor decision must be retained")
    if decision.get("active_protocol") != ACTIVE_PROTOCOL:
        _issue(issues, "ACTIVE_PROTOCOL", "decision.active_protocol", "active protocol path is not the calibration-conditioned protocol")
    if decision.get("retained_stronger_protocol") != SUPERSEDED_PROTOCOL:
        _issue(issues, "RETAINED_PROTOCOL", "decision.retained_stronger_protocol", "superseded stronger protocol reference is missing")
    if active.get("decision") != EXPECTED_DECISION:
        _issue(issues, "ACTIVE_DECISION_MISMATCH", "active.decision", "active protocol and program decision must match exactly")
    if superseded.get("superseded_by") != ACTIVE_PROTOCOL or not str(superseded.get("status", "")).startswith("SUPERSEDED_"):
        _issue(issues, "SUPERSESSION_NOT_FAIL_CLOSED", "superseded", "old protocol must identify its active successor and remain explicitly superseded")

    childlens = _mapping(decision.get("childlens_disposition"))
    if childlens.get("prior_terminal_state") != LEXICAL_STOP or childlens.get("lexical_reopening_allowed") is not False:
        _issue(issues, "LEXICAL_STOP_NOT_PRESERVED", "decision.childlens_disposition", "lexical STOP must remain terminal and reopening forbidden")
    if childlens.get("active_calibration_protocol") != PSEUDO_PROTOCOL or childlens.get("active_sensitivity_protocol") != ACTIVE_PROTOCOL:
        _issue(issues, "CHILDLENS_PROTOCOL_BINDING", "decision.childlens_disposition", "calibration and sensitivity protocol bindings are stale")
    if childlens.get("model_agreement_is_ground_truth") is not False or childlens.get("evaluation_truth") != "SIMULATOR_ORACLE_ONLY":
        _issue(issues, "GROUND_TRUTH_BOUNDARY", "decision.childlens_disposition", "model agreement cannot be truth and simulator oracle must be the only evaluation truth")

    ancestry = _mapping(active.get("ancestry"))
    if ancestry.get("mode") != ANCESTRY_MODE:
        _issue(issues, "ANCESTRY_MODE", "active.ancestry.mode", "active protocol must be the separate aggregate-calibrated protocol")
    if ancestry.get("AEA_empirical_ancestry") is not False or ancestry.get("BabyView_empirical_ancestry") is not False or ancestry.get("pooling") is not False:
        _issue(issues, "FORBIDDEN_EMPIRICAL_ANCESTRY", "active.ancestry", "AEA/BabyView ancestry and pooling must be false")
    allowed = str(ancestry.get("allowed_childlens_input", "")).lower()
    if not all(token in allowed for token in ("cell-suppressed", "outward-rounded", "aggregate")):
        _issue(issues, "CHILDLENS_INPUT_TOO_BROAD", "active.ancestry.allowed_childlens_input", "only cell-suppressed outward-rounded aggregates may enter")
    forbidden = str(ancestry.get("forbidden_childlens_input", "")).lower()
    for token in ("raw media", "text", "item rows", "timestamps", "identifiers", "pseudo-label payloads", "confidence", "weights", "checkpoints", "effect sizes"):
        if token not in forbidden:
            _issue(issues, "FORBIDDEN_INPUT_OMITTED", "active.ancestry.forbidden_childlens_input", f"missing forbidden input class: {token}")

    if decision.get("AEA_empirical_ancestry") is not False or decision.get("BabyView_empirical_ancestry") is not False or decision.get("cross_corpus_pooling") is not False:
        _issue(issues, "DECISION_ANCESTRY", "decision", "decision must reject AEA/BabyView ancestry and cross-corpus pooling")

    corpus = _mapping(pseudo.get("corpus_boundary"))
    if corpus.get("AEA_empirical_ancestry") is not False or corpus.get("BabyView_empirical_ancestry") is not False or corpus.get("pooling") is not False or corpus.get("raw_or_item_level_export") is not False:
        _issue(issues, "PSEUDO_CORPUS_BOUNDARY", "pseudo.corpus_boundary", "pseudo calibration violates ancestry, pooling, or item-export boundary")
    binding = _mapping(pseudo.get("simulator_binding"))
    if binding.get("ancestry_mode") != ANCESTRY_MODE or binding.get("authorized_consumer") != ACTIVE_PROTOCOL or binding.get("empirical_free_minimal_realism_consumer") is not False:
        _issue(issues, "PSEUDO_CONSUMER_BINDING", "pseudo.simulator_binding", "aggregate calibration must bind only to the separate active protocol")
    _validate_two_paths(pseudo, issues)

    privacy = _mapping(pseudo.get("privacy_and_execution"))
    for key in ("no_external_api", "no_telemetry", "no_repository_restricted_payload", "aggregate_only_tool_output"):
        if privacy.get(key) is not True:
            _issue(issues, "PRIVACY_FLAG", f"pseudo.privacy_and_execution.{key}", "required privacy flag must be true")

    gates = _mapping(pseudo.get("predeclared_gates"))
    if gates.get("minimum_export_cell_items") != 5:
        _issue(issues, "K_SUPPRESSION", "pseudo.predeclared_gates.minimum_export_cell_items", "public aggregate K must remain exactly 5")
    if gates.get("agreement_is_not_a_pass_gate") is not True:
        _issue(issues, "AGREEMENT_GATE", "pseudo.predeclared_gates.agreement_is_not_a_pass_gate", "agreement cannot control usability")
    uncertainty = _mapping(pseudo.get("uncertainty"))
    rounding = _mapping(uncertainty.get("grid_rounding"))
    if "0.10" not in str(rounding.get("rate_dimensions", "")) or "0.5" not in str(rounding.get("absolute_lag_event_units", "")) or "[0,1]" not in str(rounding.get("clamp", "")):
        _issue(issues, "OUTWARD_ROUNDING_UNFROZEN", "pseudo.uncertainty.grid_rounding", "rate, lag, and clamp grids must be frozen")
    if "minimum lower" not in str(uncertainty.get("envelope", "")).lower() or "maximum upper" not in str(uncertainty.get("envelope", "")).lower() or "round outward" not in str(uncertainty.get("envelope", "")).lower():
        _issue(issues, "ENVELOPE_RULE", "pseudo.uncertainty.envelope", "envelope must take extrema and round outward")

    execution = _mapping(active.get("execution"))
    if not (_false(decision, "new_scientific_outcome_run") and _false(decision, "scientific_outcome_authorized") and _false(decision, "scientific_endpoint_opened") and execution.get("outcome_authorized") is False):
        _issue(issues, "OUTCOME_AUTHORIZED", "decision/active.execution", "no scientific outcome or endpoint may be authorized/opened")
    if "NOT_YET_IMPLEMENTED_OR_AUTHORIZED" not in str(decision.get("planned_launch_command_status", "")) or "PLANNED_NOT_YET_IMPLEMENTED_OR_AUTHORIZED" not in str(execution.get("launch_command_status", "")):
        _issue(issues, "LAUNCH_STATUS", "decision/active.execution", "launch statuses must remain explicitly unauthorized")

    artifact_sets = _mapping(immutability.get("artifact_sets"))
    for version in ("v1", "v1_1", "v1_2", "v1_3"):
        row = _mapping(artifact_sets.get(version))
        if row.get("preserved") is not True or not isinstance(row.get("artifact_set_sha256"), str):
            _issue(issues, "HISTORY_IMMUTABILITY", f"immutability.artifact_sets.{version}", "historical set must be preserved and digest-bound")
    if immutability.get("prior_decisions_overwritten") is not False or immutability.get("followup_namespace") != "childlens_feasibility_v1_3_1":
        _issue(issues, "HISTORY_OVERWRITE", "immutability", "v1-v1.3 history and v1.3.1 follow-up reference must remain immutable")

    return issues


_SENSITIVE_KEYS = {
    "transcript_text",
    "translated_text",
    "source_filename",
    "source_path",
    "participant_id",
    "item_id",
    "exact_timestamp",
    "exact_interval",
    "frame_content",
    "audio_content",
    "item_level_prediction",
    "confidence_score",
    "raw_model_payload",
    "item_rows",
    "human_validation",
    "inter_human_reliability",
    "ground_truth",
    "transcript_correctness",
    "referential_truth",
}


def validate_public_calibration_receipt(receipt: Mapping[str, Any]) -> list[Issue]:
    """Validate a future repository-safe aggregate receipt.

    The public document is treated as hostile input.  Any nonempty restricted
    field fails even when another policy flag claims the export is safe.
    """

    issues: list[Issue] = []

    def walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_location = f"{location}.{key}"
                if str(key).lower() in _SENSITIVE_KEYS and child not in (False, None, "", [], {}):
                    _issue(issues, "RESTRICTED_PUBLIC_FIELD", child_location, "restricted or item-level payload is nonempty")
                walk(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            _issue(issues, "NONFINITE_PUBLIC_VALUE", location, "NaN and infinity are forbidden")

    walk(receipt, "receipt")
    policy = _mapping(receipt.get("public_export"))
    required_false = (
        "raw_counts",
        "paths",
        "identifiers",
        "filenames",
        "exact_timestamps_or_intervals",
        "transcript_or_lexical_content",
        "frames_or_audio",
        "item_level_predictions",
        "confidence_or_raw_model_payload",
        "free_form_errors",
    )
    if policy.get("minimum_cluster_k") != 5 or policy.get("complementary_suppression") is not True:
        _issue(issues, "PUBLIC_K_POLICY", "receipt.public_export", "K=5 and complementary suppression are mandatory")
    for key in required_false:
        if policy.get(key) is not False:
            _issue(issues, "PUBLIC_EXPORT_POLICY", f"receipt.public_export.{key}", "field must be false")
    if receipt.get("pseudo_labels_are_ground_truth") is not False or receipt.get("human_evidence_available") is not False or receipt.get("simulator_oracle_only_evaluation_truth") is not True:
        _issue(issues, "PUBLIC_SEMANTICS", "receipt", "receipt must deny human/ground-truth status and preserve simulator-oracle truth")
    return issues


def load_public_bundle(root: Path) -> tuple[dict[str, Any], ...]:
    paths = (
        "output/nursery_program_convergence_v1/decision_record.json",
        ACTIVE_PROTOCOL,
        SUPERSEDED_PROTOCOL,
        PSEUDO_PROTOCOL,
        IMMUTABILITY_RECEIPT,
    )
    values: list[dict[str, Any]] = []
    for relative in paths:
        path = root / relative
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"public artifact must be a JSON object: {relative}")
        values.append(value)
    return tuple(values)


def validate_repository(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    try:
        bundle = load_public_bundle(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [Issue("PUBLIC_ARTIFACT_LOAD", "repository", str(exc))]
    issues.extend(validate_public_artifacts(*bundle))
    for version, relative in HISTORICAL_REFERENCES.items():
        if not (root / relative).is_file():
            _issue(issues, "HISTORICAL_REFERENCE_MISSING", relative, f"missing immutable {version} reference")
    return issues


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--calibration-receipt", type=Path)
    args = parser.parse_args(argv)

    issues = validate_repository(args.repository.resolve())
    if args.calibration_receipt is not None:
        try:
            receipt = json.loads(args.calibration_receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(Issue("CALIBRATION_RECEIPT_LOAD", str(args.calibration_receipt), str(exc)))
        else:
            if not isinstance(receipt, Mapping):
                issues.append(Issue("CALIBRATION_RECEIPT_SCHEMA", str(args.calibration_receipt), "receipt must be a JSON object"))
            else:
                issues.extend(validate_public_calibration_receipt(receipt))

    result = {
        "schema_version": "nursery-program-convergence-validation-v1",
        "status": "PASS" if not issues else "FAIL",
        "issue_count": len(issues),
        "issues": [issue.as_dict() for issue in issues],
        "restricted_or_quarantine_access": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
