from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_gemma_substitution_receipt.json"
DECISION = ROOT / "output/nursery_program_convergence_v1/gemma_two_model_calibration_gate_decision_v1_7.json"
MEMO = ROOT / "docs/nursery_program_convergence_v1/gemma_two_model_calibration_gate_disposition_v1_7.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_decision_is_bound_to_the_completed_aggregate_receipt() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["terminal_state"] == "GENUINE_USER_ONLY_BLOCKER"
    assert decision["calibration"]["receipt_sha256"] == sha256(RECEIPT)
    assert receipt["status"] == decision["calibration"]["status"] == "CALIBRATION_REVISE"
    assert receipt["gate_failures"] == decision["calibration"]["failed_gates"] == [
        "ENVELOPE_NULL_OR_IRRELEVANT"
    ]


def test_null_envelope_failure_is_k5_suppression_not_an_effect_result() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    observed = receipt["calibration_ranges"]["null_or_irrelevant"]
    assert set(observed) == set(decision["calibration"]["null_or_irrelevant_statuses"])
    assert all(value == {"status": "SUPPRESSED_K5"} for value in observed.values())
    assert set(decision["calibration"]["null_or_irrelevant_statuses"].values()) == {
        "SUPPRESSED_K5"
    }
    assert receipt["scientific_outcome_run"] is False
    assert receipt["scientific_endpoint_opened"] is False


def test_no_fallback_preoutcome_authorization_or_causal_output_exists() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["automatic_fallback_allowed"] is False
    assert decision["causal_outcome"] == {
        "authorization_seal_exists": False,
        "endpoint_opened": False,
        "output_exists": False,
        "preoutcome_receipt_exists": False,
        "run": False,
    }
    assert not (ROOT / "output/nursery_program_convergence_v1/symbolic_preoutcome_receipt_v2.json").exists()
    assert not (ROOT / "output/nursery_program_convergence_v1/symbolic_one_shot_authorization_v2.json").exists()
    assert not (ROOT / "output/nursery_calibration_conditioned_symbolic_grounding_v2").exists()


def test_public_disposition_contains_no_restricted_payload_markers() -> None:
    combined = (DECISION.read_text(encoding="utf-8") + MEMO.read_text(encoding="utf-8")).casefold()
    forbidden = (
        "/users/",
        "participant_id",
        "media_relpath",
        "transcript_text",
        "raw_response",
        "prediction_join_key",
        "expected_media_sha256",
    )
    assert not any(marker in combined for marker in forbidden)
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["privacy"]["restricted_payload_exported"] is False
    assert decision["privacy"]["exact_sub_k_counts_exported"] is False


def test_options_require_an_explicit_user_choice() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["decision_required"] is True
    assert len(decision["options"]) == 4
    assert sum(option["recommended"] for option in decision["options"]) == 1
    assert decision["options"][0] == {
        "id": "A_CONTENT_BLIND_K5_EXTENSION",
        "recommended": True,
    }
