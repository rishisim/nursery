from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_childlens_language_blocker_v1_3_1.py"
SPEC = importlib.util.spec_from_file_location("childlens_language_blocker_validator_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_repository_disposition_passes_fail_closed_validator():
    result = MODULE.validate()
    assert result["status"] == "PASS", result["issues"]
    assert result["issue_count"] == 0
    assert all(value["preserved"] for value in result["history"].values())


def test_model_runtime_receipts_are_public_synthetic_and_not_human_evidence():
    gemma = json.loads((ROOT / "output/childlens_feasibility_v1_3_1/gemma4_public_synthetic_runtime_bakeoff.json").read_text())
    qwen = json.loads((ROOT / "output/childlens_feasibility_v1_3_1/qwen3_asr_public_synthetic_runtime_receipt.json").read_text())
    assert gemma["childlens_or_quarantine_accessed"] is False
    assert gemma["network_denied_during_inference"] is True
    assert gemma["interpretation"]["single_fixture_accuracy_generalizes_to_childlens"] is False
    assert gemma["interpretation"]["restricted_childlens_rerun_authorized"] is False
    assert qwen["childlens_or_quarantine_accessed"] is False
    assert qwen["network_denial_sentinel_passed"] is True
    assert qwen["model_output_is_human_evidence"] is False
    assert qwen["restricted_inference_authorized"] is False
