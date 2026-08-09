from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_gemma_template_order_amendment_v1_3.py"
AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json"
CAF_AMENDMENT = ROOT / "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json"
OVERRIDE = ROOT / "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json"
CAF_RUNNER = ROOT / "scripts/run_gemma4_prototype_common_schema_canary_v1_2.py"
CAF_FAILURE = ROOT / "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/public_common_schema_canary_failure_v1.json"
LEGACY_WORKER = ROOT / "scripts/nursery_gemma4_referential_worker.py"
SPEC = importlib.util.spec_from_file_location("validate_template_order", SCRIPT)
assert SPEC and SPEC.loader
v = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v
SPEC.loader.exec_module(v)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return v._sha256(path.read_bytes())


def _codes(issues: list) -> set[str]:
    return {issue.code for issue in issues}


def _receipt() -> dict:
    return {"template_order_amendment_sha256": v.AMENDMENT_SHA256, "template_order_amendment_path": "docs/nursery_program_convergence_v1/gemma4_template_placeholder_order_v1_3/frozen_template_placeholder_order_amendment_v1_3.json", "caf_transport_amendment_sha256": v.CAF_AMENDMENT_SHA256, "no_automatic_fallback_override_sha256": v.NO_FALLBACK_SHA256, "prior_caf_runner_sha256": v.CAF_RUNNER_SHA256, "prior_caf_failure_receipt_sha256": v.CAF_FAILURE_SHA256, "declarative_user_content_order_preserved": True, "rendered_audio_placeholder_relocation_count": 1, "rendered_placeholder_order": "AUDIO_THEN_FIVE_CONTIGUOUS_IMAGES", "system_prompt_occurrences": 1, "user_prompt_occurrences": 1, "audio_placeholder_occurrences": 1, "image_placeholder_occurrences": 5, "five_image_placeholders_contiguous": True, "non_audio_rendered_bytes_unchanged": True, "intermediate_container": "CAF", "final_wav_contract_unchanged": True, "five_frame_contract_unchanged": True, "parser_corrections_spent_before_revised_canary": 0, "parser_allowance_preserved": True, "output_namespace": v.NEW_NAMESPACE, "historical_output_overwritten": False, "canonical_output_overwritten": False, "restricted_content_accessed_before_activation": False, "restricted_inference_run": False, "scientific_outcome_run": False}


def test_live_amendment_and_bound_inputs_pass() -> None:
    assert v.validate_bound_inputs(_sha(CAF_AMENDMENT), _sha(OVERRIDE), _sha(CAF_RUNNER), _json(CAF_FAILURE), _sha(CAF_FAILURE), _sha(LEGACY_WORKER)) == []
    assert v.validate_amendment(_json(AMENDMENT), _sha(AMENDMENT)) == []
    assert v.validate_revised_receipt(_receipt(), v.AMENDMENT_SHA256) == []


@pytest.mark.parametrize("field", ["historical_artifacts_modified", "public_canary_run", "restricted_content_accessed", "restricted_inference_run", "scientific_outcome_run"])
def test_additive_pre_execution_freeze(field: str) -> None:
    amendment = _json(AMENDMENT)
    amendment[field] = True
    expected = "TEMPLATE_ADDITIVE_BOUNDARY" if field == "historical_artifacts_modified" else "TEMPLATE_FREEZE_ORDER"
    assert expected in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256))


def test_caf_runner_and_failure_bytes_are_exact() -> None:
    assert "TEMPLATE_BOUND_INPUT" in _codes(v.validate_bound_inputs(v.CAF_AMENDMENT_SHA256, v.NO_FALLBACK_SHA256, "f" * 64, _json(CAF_FAILURE), v.CAF_FAILURE_SHA256, v.LEGACY_WORKER_SHA256))
    failure = _json(CAF_FAILURE)
    failure["automatic_calibration_fallback_allowed"] = True
    assert "CAF_FAILURE_IMMUTABILITY" in _codes(v.validate_bound_inputs(v.CAF_AMENDMENT_SHA256, v.NO_FALLBACK_SHA256, v.CAF_RUNNER_SHA256, failure, v.CAF_FAILURE_SHA256, v.LEGACY_WORKER_SHA256))


@pytest.mark.parametrize(("field", "value"), [("maximum_relocated_substrings", 2), ("relocated_substring_utf8", v.IMAGE), ("frozen_observed_anchor_sequence", v.NEW_SEQUENCE), ("amended_anchor_sequence", v.OLD_SEQUENCE), ("declarative_message_content_order_remains", ["image_1", "audio"])])
def test_exact_single_audio_relocation(field: str, value: object) -> None:
    amendment = _json(AMENDMENT)
    amendment["exact_post_render_delta"][field] = value
    assert "TEMPLATE_EXACT_RELOCATION" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256))


@pytest.mark.parametrize("field", ["declarative_messages_changed", "system_prompt_text_changed", "user_prompt_text_changed", "template_arguments_changed", "placeholder_counts_changed", "non_placeholder_rendered_bytes_changed", "media_argument_order_changed", "semantic_fields_changed", "model_runtime_or_artifact_fields_changed", "scientific_fields_changed", "parser_allowance_changed"])
def test_no_other_field_may_change(field: str) -> None:
    amendment = _json(AMENDMENT)
    amendment["exact_post_render_delta"][field] = True
    assert "TEMPLATE_SCOPE_EXPANSION" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256))


def test_parser_allowance_unspent_and_preserved() -> None:
    amendment = _json(AMENDMENT)
    amendment["exact_post_render_delta"]["does_not_spend_outer_fence_parser_correction"] = False
    assert "TEMPLATE_PARSER_BUDGET" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256))
    amendment = _json(AMENDMENT)
    amendment["preserved_public_canary_contract"]["outer_fence_parser_correction_used_before_amendment"] = 1
    assert "TEMPLATE_PARSER_BUDGET" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256))


@pytest.mark.parametrize("field", ["exact_system_and_user_prompt_text_unchanged", "exact_common_schema_sha256_unchanged", "final_WAV_contract_unchanged", "five_frame_generator_and_offsets_unchanged"])
def test_prompt_schema_wav_and_frames_unchanged(field: str) -> None:
    amendment = _json(AMENDMENT)
    amendment["preserved_public_canary_contract"][field] = False
    assert "TEMPLATE_CANARY_CONTRACT" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256))


@pytest.mark.parametrize("field", ["sample_15_items_900_seconds_137_candidate_windows_unchanged", "K5_complement_protection_and_outward_rounding_unchanged", "coverage_abstention_path_and_schema_gates_unchanged", "Qwen3_read_only_digest_contract_unchanged"])
def test_sample_and_scientific_gates_unchanged(field: str) -> None:
    amendment = _json(AMENDMENT)
    amendment["preserved_artifact_sample_and_scientific_contract"][field] = False
    assert "TEMPLATE_SCIENTIFIC_CONTRACT" in _codes(v.validate_amendment(amendment, v.AMENDMENT_SHA256))


@pytest.mark.parametrize("field", ["historical_output_overwritten", "canonical_output_overwritten", "restricted_content_accessed_before_activation", "restricted_inference_run", "scientific_outcome_run"])
def test_revised_receipt_no_overwrite_or_early_execution(field: str) -> None:
    receipt = _receipt()
    receipt[field] = True
    expected = "TEMPLATE_RECEIPT_OUTPUT" if "overwritten" in field else "TEMPLATE_RECEIPT_ORDER"
    assert expected in _codes(v.validate_revised_receipt(receipt, v.AMENDMENT_SHA256))


@pytest.mark.parametrize(("field", "value"), [("rendered_audio_placeholder_relocation_count", 2), ("rendered_placeholder_order", "OTHER"), ("declarative_user_content_order_preserved", False), ("image_placeholder_occurrences", 4), ("non_audio_rendered_bytes_unchanged", False)])
def test_revised_receipt_exact_relocation(field: str, value: object) -> None:
    receipt = _receipt()
    receipt[field] = value
    codes = _codes(v.validate_revised_receipt(receipt, v.AMENDMENT_SHA256))
    assert codes & {"TEMPLATE_RECEIPT_RELOCATION", "TEMPLATE_RECEIPT_COUNTS"}


def test_validator_has_no_model_or_restricted_execution_path() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in ("subprocess", "torch.load", "mlx.core", "rglob(", "os.walk", "quarantine_root", "restricted_manifest"):
        assert token not in source
