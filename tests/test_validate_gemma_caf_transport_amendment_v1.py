from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_gemma_caf_transport_amendment_v1.py"
AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json"
BASE = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_common_schema_canary_contract_v1.json"
TERMINAL = ROOT / "output/nursery_program_convergence_v1/prototype_gemma_activation_terminal_decision_v1.json"
VALIDATION = ROOT / "output/nursery_program_convergence_v1/prototype_gemma_activation_validation_receipt_v1.json"
FAILURE = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_prototype_common_schema_public_canary_failure_v1.json"
DISPOSITION = ROOT / "docs/nursery_program_convergence_v1/prototype_gemma_activation_technical_revision_disposition_v1.md"
OVERRIDE = ROOT / "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json"
SPEC = importlib.util.spec_from_file_location("validate_caf", SCRIPT)
assert SPEC and SPEC.loader
v = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v
SPEC.loader.exec_module(v)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return v._sha256(path.read_bytes())


def _receipt() -> dict:
    return {"caf_transport_amendment_sha256": v.AMENDMENT_SHA256, "caf_transport_amendment_path": "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json", "no_automatic_fallback_override_sha256": v.NO_FALLBACK_SHA256, "output_namespace": v.NEW_NAMESPACE, "historical_output_overwritten": False, "canonical_output_overwritten": False, "intermediate_container": "CAF", "final_wav_contract_unchanged": True, "parser_corrections_spent_before_revised_canary": 0, "parser_allowance_preserved": True, "restricted_content_accessed_before_activation": False, "restricted_inference_run": False, "scientific_outcome_run": False}


def _codes(issues: list) -> set[str]:
    return {issue.code for issue in issues}


def test_live_amendment_and_prior_receipts_pass() -> None:
    assert v.validate_prior_receipts(_json(TERMINAL), _sha(TERMINAL), _json(VALIDATION), _sha(VALIDATION), _json(FAILURE), _sha(FAILURE), _sha(DISPOSITION)) == []
    assert v.validate_no_fallback(_json(OVERRIDE), _sha(OVERRIDE)) == []
    assert v.validate_amendment(_json(AMENDMENT), _sha(AMENDMENT), _json(BASE), _sha(BASE)) == []
    assert v.validate_revised_chain_receipt(_receipt(), v.AMENDMENT_SHA256) == []


@pytest.mark.parametrize("field", ["historical_artifacts_modified", "public_canary_run", "restricted_content_accessed", "restricted_inference_run", "scientific_outcome_run"])
def test_amendment_is_additive_and_pre_execution(field: str) -> None:
    amendment = _json(AMENDMENT)
    amendment[field] = True
    expected = "CAF_ADDITIVE_BOUNDARY" if field == "historical_artifacts_modified" else "CAF_FREEZE_ORDER"
    assert expected in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256, _json(BASE), _sha(BASE)))


@pytest.mark.parametrize(("row", "field", "value"), [(0, "amended_value", v.NEW_INVOCATION.replace(".caf", ".wav")), (1, "JSON_pointer", "/other"), (0, "frozen_value", "bad")])
def test_only_exact_two_aiff_to_caf_changes(row: int, field: str, value: str) -> None:
    amendment = _json(AMENDMENT)
    amendment["exact_transport_delta"]["changes"][row][field] = value
    assert "CAF_EXACT_DELTA" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256, _json(BASE), _sha(BASE)))


def test_pointer_precondition_binds_original_contract() -> None:
    base = _json(BASE)
    base["public_fixture"]["audio"]["local_TTS"]["invocation"] = "changed"
    assert "CAF_POINTER_PRECONDITION" in _codes(v.validate_amendment(_json(AMENDMENT), v.AMENDMENT_SHA256, base, v.BASE_CONTRACT_SHA256))


@pytest.mark.parametrize(("field", "value"), [("channels", 2), ("sample_format", "float"), ("sample_rate_hz", 22050), ("sample_count", 159999), ("duration_seconds", 9.9), ("normalization_or_loudness_filter", True)])
def test_final_wav_contract_cannot_change(field: str, value: object) -> None:
    amendment = _json(AMENDMENT)
    amendment["preserved_public_fixture_contract"]["final_WAV"][field] = value
    assert "CAF_FINAL_WAV" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256, _json(BASE), _sha(BASE)))


def test_parser_allowance_is_unspent_and_preserved() -> None:
    amendment = _json(AMENDMENT)
    amendment["exact_transport_delta"]["does_not_spend_outer_fence_parser_correction"] = False
    assert "CAF_PARSER_BUDGET" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256, _json(BASE), _sha(BASE)))
    amendment = _json(AMENDMENT)
    amendment["preserved_public_fixture_contract"]["outer_fence_parser_correction_used_before_amendment"] = 1
    assert "CAF_PRESERVED_FIXTURE" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256, _json(BASE), _sha(BASE)))


@pytest.mark.parametrize("field", ["primary_item_count", "primary_total_seconds", "candidate_window_count", "minimum_export_cell_items", "K5_complement_protection", "simulator_oracle_remains_only_evaluation_truth"])
def test_scientific_sample_and_export_contract_unchanged(field: str) -> None:
    amendment = _json(AMENDMENT)
    amendment["preserved_sample_export_and_gate_contract"][field] = "changed"
    assert "CAF_SCIENTIFIC_CONTRACT" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256, _json(BASE), _sha(BASE)))


def test_prior_receipt_mutation_rejected() -> None:
    terminal = _json(TERMINAL)
    terminal["decision_basis"]["parser_corrections_used"] = 1
    assert "PRIOR_TERMINAL_STATE" in _codes(v.validate_prior_receipts(terminal, v.PRIOR_TERMINAL_SHA256, _json(VALIDATION), v.PRIOR_VALIDATION_SHA256, _json(FAILURE), v.PRIOR_FAILURE_SHA256, v.PRIOR_DISPOSITION_SHA256))


@pytest.mark.parametrize("field", ["historical_output_overwritten", "canonical_output_overwritten", "restricted_content_accessed_before_activation", "restricted_inference_run", "scientific_outcome_run"])
def test_revised_receipt_cannot_overwrite_or_run_early(field: str) -> None:
    receipt = _receipt()
    receipt[field] = True
    expected = "CAF_RECEIPT_OUTPUT" if "overwritten" in field else "CAF_RECEIPT_ORDER"
    assert expected in _codes(v.validate_revised_chain_receipt(receipt, v.AMENDMENT_SHA256))


def test_revised_receipt_binds_no_fallback_override() -> None:
    receipt = _receipt()
    receipt["no_automatic_fallback_override_sha256"] = "f" * 64
    assert "CAF_RECEIPT_FALLBACK_BINDING" in _codes(v.validate_revised_chain_receipt(receipt, v.AMENDMENT_SHA256))


def test_no_fallback_override_rejects_qwen_only_removal() -> None:
    override = _json(OVERRIDE)
    override["automatic_fallbacks_forbidden"].remove("QWEN3_ONLY_CALIBRATION")
    assert "NO_FALLBACK_SET" in _codes(v.validate_no_fallback(override, v.NO_FALLBACK_SHA256))


def test_validator_has_no_model_or_restricted_execution_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in ("subprocess", "torch.load", "mlx.core", "rglob(", "os.walk", "quarantine_root", "restricted_manifest"):
        assert token not in source
