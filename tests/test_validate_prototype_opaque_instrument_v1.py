from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/validate_prototype_opaque_instrument_v1.py"
STOP_PATH = ROOT / "output/nursery_program_convergence_v1/gemma4_replacement_terminal_decision.json"
AMENDMENT_PATH = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_e4b_opaque_instrument_provenance_amendment_v1.json"
CANARY_CONTRACT_PATH = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_common_schema_canary_contract_v1.json"
ACTIVATION_SCHEMA_PATH = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_activation_receipt_schema_v1.json"
RUNTIME_ERRATUM_PATH = ROOT / "docs/nursery_program_convergence_v1/frozen_gemma4_prototype_runtime_pin_erratum_v1_1.json"
NO_FALLBACK_PATH = ROOT / "docs/nursery_program_convergence_v1/frozen_no_automatic_fallback_override_v1.json"
PROVENANCE_ACTIVATION_PATH = ROOT / "output/nursery_program_convergence_v1/gemma4_e4b_opaque_instrument_provenance_activation_receipt_v1.json"
SPEC = importlib.util.spec_from_file_location("validate_opaque", MODULE_PATH)
assert SPEC and SPEC.loader
v = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v
SPEC.loader.exec_module(v)


@pytest.fixture()
def stop() -> dict:
    return json.loads(STOP_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def amendment() -> dict:
    return json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def provenance_activation() -> dict:
    return json.loads(PROVENANCE_ACTIVATION_PATH.read_text(encoding="utf-8"))


def _sha(char: str) -> str:
    return char * 64


def _contract(amendment_sha: str) -> dict:
    contract = json.loads(CANARY_CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["immutable_bindings"]["prototype_provenance_amendment"]["sha256"] = amendment_sha
    contract["immutable_bindings"]["prototype_provenance_amendment"]["path"] = str(AMENDMENT_PATH.relative_to(ROOT))
    contract["terminal_state"]["contract_frozen"] = True
    return contract


def _canary(amendment_sha: str, contract_sha: str) -> dict:
    return {
        "schema_version": v.CANARY_SCHEMA,
        "status": "PUBLIC_COMMON_SCHEMA_CANARY_PASS",
        "amendment_sha256": amendment_sha,
        "canary_contract_sha256": contract_sha,
        "runtime_pin_erratum_path": v.RUNTIME_ERRATUM_PATH,
        "runtime_pin_erratum_sha256": v.RUNTIME_ERRATUM_SHA256,
        "tts_executable_sha256": _sha("7"),
        "os_build_and_voice_inventory_sha256": _sha("8"),
        "fixture_manifest_sha256": _sha("9"),
        "instrument_id": v.OPAQUE_INSTRUMENT_ID,
        "conversion_revision": v.CACHED_CONVERSION_REVISION,
        "artifact_manifest_sha256": v.CACHED_ARTIFACT_MANIFEST_SHA256,
        "restricted_common_schema_sha256": v.BASELINE_COMMON_SCHEMA_SHA256,
        "exact_output_schema_sha256": v.EXACT_OUTPUT_SCHEMA_SHA256,
        "self_generated_public_fixture_only": True,
        "joint_audio_plus_five_frames": True,
        "five_frames_consumed_in_order": True,
        "exact_schema_valid": True,
        "network_denial_sentinel_passed": True,
        "no_external_request": True,
        "restricted_payload_accessed": False,
        "download_performed": False,
        "clickthrough_accepted": False,
        "hosted_or_cloud_inference_used": False,
        "restricted_inference_authorized_by_canary": False,
        "corrections_used": 0,
        "correction_scope": "NONE",
        "semantic_tuning_performed": False,
        "semantic_quality_gate_used": False,
    }


def _schema_instance(schema: dict, root: dict) -> object:
    if "$ref" in schema:
        return _schema_instance(dict(v._resolve_local_ref(root, schema["$ref"])), root)
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if "enum" in schema:
        return copy.deepcopy(schema["enum"][0])
    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties", {})
        return {key: _schema_instance(properties[key], root) for key in schema.get("required", [])}
    if kind == "string":
        if schema.get("pattern") == "^[0-9a-f]{64}$":
            return _sha("a")
        if schema.get("pattern") == "^2026-[0-9]{2}-[0-9]{2}$":
            return "2026-07-22"
        return "synthetic"
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    raise AssertionError(f"cannot synthesize schema node: {schema}")


def _activation_schema() -> dict:
    return json.loads(ACTIVATION_SCHEMA_PATH.read_text(encoding="utf-8"))


def _runtime_erratum() -> dict:
    return json.loads(RUNTIME_ERRATUM_PATH.read_text(encoding="utf-8"))


def _activation(canary_sha: str) -> dict:
    original = _activation_schema()
    effective = v.effective_activation_schema(original, _runtime_erratum())
    receipt = _schema_instance(effective, effective)
    assert isinstance(receipt, dict)
    receipt["public_canary_binding"]["receipt_sha256"] = canary_sha
    return receipt


def _calibration(activation_sha: str) -> dict:
    return {
        "schema_version": v.CALIBRATION_SCHEMA,
        "decision": "CALIBRATION_PASS_INTERNAL_PROTOTYPE",
        "activation_receipt_sha256": activation_sha,
        "no_automatic_fallback_override_sha256": v.NO_FALLBACK_OVERRIDE_SHA256,
        "referential_instrument_ids": [v.OPAQUE_INSTRUMENT_ID, v.QWEN3_ID],
        "qwen3_reused_read_only": True,
        "primary_item_count": 15,
        "primary_total_seconds": 900,
        "candidate_window_count": 137,
        "sample_reselected": False,
        "public_export": {
            "minimum_cluster_k": 5,
            "complementary_suppression": True,
            "rate_rounding_step": 0.1,
            "lag_rounding_step_event_units": 0.5,
            "raw_counts": False,
            "item_level_results": False,
            "transcript_or_lexical_content": False,
            "frames_or_audio": False,
            "identifiers_or_paths": False,
            "exact_timestamps": False,
            "confidence_or_raw_payload": False,
        },
        "all_frozen_calibration_gates_passed": True,
        "aggregate_only": True,
        "simulator_oracle_only_evaluation_truth": True,
        "pseudo_labels_are_ground_truth": False,
        "human_validation_claimed": False,
        "scientific_outcome_run": False,
        "scientific_endpoint_opened": False,
        "semantic_tuning_performed": False,
    }


def _construction() -> dict:
    return {
        "schema_version": v.CONSTRUCTION_GATE_SCHEMA,
        "decision": "CONSTRUCTION_AND_FALSIFICATION_PASS",
        "construction_gate_passed": True,
        "falsification_gate_passed": True,
    }


def _seal(calibration_sha: str, construction_sha: str) -> dict:
    return {
        "schema_version": v.OUTCOME_SEAL_SCHEMA,
        "status": "OUTCOME_AUTHORIZED_GATES_PASS",
        "calibration_receipt_sha256": calibration_sha,
        "construction_falsification_receipt_sha256": construction_sha,
        "no_automatic_fallback_override_sha256": v.NO_FALLBACK_OVERRIDE_SHA256,
        "calibration_gate_passed": True,
        "construction_gate_passed": True,
        "falsification_gate_passed": True,
        "authorization_used_restricted_item_payload": False,
        "scientific_outcome_run": False,
    }


def _codes(issues: list) -> set[str]:
    return {row.code for row in issues}


def test_live_amendment_and_synthetic_future_chain_pass(stop: dict, amendment: dict, provenance_activation: dict) -> None:
    amendment_sha = v._sha256(AMENDMENT_PATH.read_bytes())
    contract = _contract(amendment_sha)
    contract_sha = v.canonical_digest(contract)
    canary = _canary(amendment_sha, contract_sha)
    canary_sha = v.canonical_digest(canary)
    activation_schema = _activation_schema()
    runtime_erratum = _runtime_erratum()
    activation_schema_sha = v._sha256(ACTIVATION_SCHEMA_PATH.read_bytes())
    runtime_erratum_sha = v._sha256(RUNTIME_ERRATUM_PATH.read_bytes())
    activation = _activation(canary_sha)
    activation_sha = v.canonical_digest(activation)
    calibration = _calibration(activation_sha)
    calibration_sha = v.canonical_digest(calibration)
    construction = _construction()
    construction_sha = v.canonical_digest(construction)
    assert v.validate_amendment(amendment, stop, v.PRIOR_STOP_SHA256) == []
    assert v.validate_provenance_activation(provenance_activation, amendment_sha) == []
    assert v.validate_canary_contract(contract, amendment_sha) == []
    assert v.validate_canary(canary, amendment_sha, contract_sha) == []
    assert v.validate_activation(
        activation,
        activation_schema,
        activation_schema_sha,
        runtime_erratum,
        runtime_erratum_sha,
        canary_sha,
    ) == []
    assert v.validate_calibration(calibration, activation_sha) == []
    assert v.validate_outcome_seal(_seal(calibration_sha, construction_sha), calibration, calibration_sha, construction, construction_sha) == []


@pytest.mark.parametrize("field", ["restricted_gemma_inference_run", "calibration_receipt_replaced", "scientific_outcome_run", "third_visual_model_tried"])
def test_prior_stop_is_immutable(stop: dict, field: str) -> None:
    stop[field] = True
    assert "PRIOR_STOP_MUTATED" in _codes(v.validate_prior_stop(stop, v.PRIOR_STOP_SHA256))


def test_prior_stop_byte_hash_is_exact(stop: dict) -> None:
    assert "PRIOR_STOP_IDENTITY" in _codes(v.validate_prior_stop(stop, _sha("f")))


def test_live_no_automatic_fallback_override_passes() -> None:
    override = json.loads(NO_FALLBACK_PATH.read_text(encoding="utf-8"))
    override_sha = v._sha256(NO_FALLBACK_PATH.read_bytes())
    assert v.validate_no_fallback_override(override, override_sha) == []


@pytest.mark.parametrize("fallback", ["QWEN3_ONLY_CALIBRATION", "CONSERVATIVE_UNBOUNDED_OR_FULL_DOMAIN_SYNTHETIC_SENSITIVITY", "EMPIRICAL_FREE_AUTOMATIC_OUTCOME", "THIRD_VISUAL_MODEL"])
def test_no_automatic_fallback_may_be_removed(fallback: str) -> None:
    override = json.loads(NO_FALLBACK_PATH.read_text(encoding="utf-8"))
    override["automatic_fallbacks_forbidden"].remove(fallback)
    assert "NO_FALLBACK_SET" in _codes(v.validate_no_fallback_override(override, v.NO_FALLBACK_OVERRIDE_SHA256))


def test_provenance_activation_cannot_fully_activate(amendment: dict, provenance_activation: dict) -> None:
    amendment_sha = v._sha256(AMENDMENT_PATH.read_bytes())
    provenance_activation["gate_result"]["restricted_inference_authorized_by_this_receipt"] = True
    assert "PROVENANCE_GATE_SCOPE" in _codes(v.validate_provenance_activation(provenance_activation, amendment_sha))


@pytest.mark.parametrize("field", ["redistribution_upload_publication_or_sharing", "new_download_or_clickthrough", "learner_ancestry_allowed"])
def test_provenance_activation_internal_only(amendment: dict, provenance_activation: dict, field: str) -> None:
    amendment_sha = v._sha256(AMENDMENT_PATH.read_bytes())
    provenance_activation["covenants"][field] = True
    assert "PROVENANCE_COVENANT" in _codes(v.validate_provenance_activation(provenance_activation, amendment_sha))


@pytest.mark.parametrize("field", ["conversion_revision", "artifact_manifest_sha256", "artifact_bytes", "file_count"])
def test_exact_cached_artifact_is_required(stop: dict, amendment: dict, field: str) -> None:
    amendment["exact_cached_artifact"][field] = "bad"
    assert "OPAQUE_ARTIFACT_BINDING" in _codes(v.validate_amendment(amendment, stop, v.PRIOR_STOP_SHA256))


@pytest.mark.parametrize("field", ["restricted_content_accessed_to_draft", "new_model_downloaded", "license_or_clickthrough_accepted"])
def test_no_restricted_access_download_or_clickthrough(stop: dict, amendment: dict, field: str) -> None:
    amendment[field] = True
    assert "AMENDMENT_ORDER" in _codes(v.validate_amendment(amendment, stop, v.PRIOR_STOP_SHA256))


def test_prior_stop_reference_cannot_be_rewritten(stop: dict, amendment: dict) -> None:
    amendment["historical_decisions_preserved"]["gemma_replacement_stop"]["sha256"] = _sha("f")
    assert "STOP_IMMUTABILITY" in _codes(v.validate_amendment(amendment, stop, v.PRIOR_STOP_SHA256))


@pytest.mark.parametrize("field", ["same_15_items_900_seconds_137_windows", "minimum_export_cell_items", "agreement_is_not_a_pass_gate", "qwen3_read_only_digest_match"])
def test_activation_gates_remain_frozen(stop: dict, amendment: dict, field: str) -> None:
    amendment["unchanged_non_provenance_activation_gates"][field] = False
    assert "CONTRACT_CHANGED" in _codes(v.validate_amendment(amendment, stop, v.PRIOR_STOP_SHA256))


def test_known_opaque_limitations_cannot_be_hidden(stop: dict, amendment: dict) -> None:
    amendment["known_provenance_limitations"]["reconstructible_from_upstream"] = True
    assert "OPAQUE_LIMITATIONS" in _codes(v.validate_amendment(amendment, stop, v.PRIOR_STOP_SHA256))


def test_canary_contract_exact_schema_and_qwen_read_only(amendment: dict) -> None:
    amendment_sha = v._sha256(AMENDMENT_PATH.read_bytes())
    contract = _contract(amendment_sha)
    contract["exact_common_schema"]["additionalProperties"] = True
    assert "CANARY_CONTRACT_SCHEMA" in _codes(v.validate_canary_contract(contract, amendment_sha))
    contract = _contract(amendment_sha)
    contract["qwen3_read_only_binding"]["restricted_hypotheses_recomputed"] = True
    assert "QWEN3_READ_ONLY" in _codes(v.validate_canary_contract(contract, amendment_sha))


def test_canary_contract_freezes_local_german_tts_fixture(amendment: dict) -> None:
    amendment_sha = v._sha256(AMENDMENT_PATH.read_bytes())
    contract = _contract(amendment_sha)
    contract["public_fixture"]["audio"]["local_TTS"]["voice"] = "Other"
    assert "CANARY_CONTRACT_FIXTURE" in _codes(v.validate_canary_contract(contract, amendment_sha))


def test_canary_contract_is_bound_to_amendment(amendment: dict) -> None:
    amendment_sha = v._sha256(AMENDMENT_PATH.read_bytes())
    contract = _contract(amendment_sha)
    contract["immutable_bindings"]["prototype_provenance_amendment"]["sha256"] = _sha("f")
    assert "CANARY_CONTRACT_BINDING" in _codes(v.validate_canary_contract(contract, amendment_sha))


@pytest.mark.parametrize("field", ["joint_audio_plus_five_frames", "five_frames_consumed_in_order", "exact_schema_valid", "network_denial_sentinel_passed", "no_external_request"])
def test_exact_common_schema_canary_is_conjunctive(field: str) -> None:
    canary = _canary(_sha("a"), _sha("b"))
    canary[field] = False
    assert "CANARY_REQUIREMENT" in _codes(v.validate_canary(canary, _sha("a"), _sha("b")))


def test_canary_cannot_authorize_restricted_access() -> None:
    canary = _canary(_sha("a"), _sha("b"))
    canary["restricted_inference_authorized_by_canary"] = True
    assert "CANARY_BOUNDARY" in _codes(v.validate_canary(canary, _sha("a"), _sha("b")))


def test_one_correction_only_and_no_semantic_tuning() -> None:
    canary = _canary(_sha("a"), _sha("b"))
    canary["corrections_used"] = 2
    assert "CANARY_CORRECTION" in _codes(v.validate_canary(canary, _sha("a"), _sha("b")))
    canary = _canary(_sha("a"), _sha("b"))
    canary["semantic_tuning_performed"] = True
    assert "CANARY_SEMANTIC_TUNING" in _codes(v.validate_canary(canary, _sha("a"), _sha("b")))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("artifact_binding", "cached_rehash_exact_match"), False),
        (("public_canary_binding", "exact_schema_valid"), False),
        (("qwen3_read_only_binding", "restricted_hypotheses_recomputed"), True),
        (("unchanged_contract", "candidate_window_count"), 136),
        (("authorization_boundary", "restricted_inference_run"), True),
        (("privacy_and_scientific_attestations", "scientific_outcome_authorized"), True),
        (("authorization_boundary", "restricted_inference_run"), 0),
    ],
)
def test_frozen_nested_activation_schema_rejects_gate_mutations(path: tuple[str, str], value: object) -> None:
    schema = _activation_schema()
    erratum = _runtime_erratum()
    activation = _activation(_sha("b"))
    activation[path[0]][path[1]] = value
    assert "ACTIVATION_SCHEMA_VALIDATION" in _codes(
        v.validate_activation(
            activation,
            schema,
            v.ACTIVATION_RECEIPT_SCHEMA_SHA256,
            erratum,
            v.RUNTIME_ERRATUM_SHA256,
            _sha("b"),
        )
    )


def test_activation_schema_rejects_extra_properties_and_wrong_schema_bytes() -> None:
    schema = _activation_schema()
    erratum = _runtime_erratum()
    activation = _activation(_sha("b"))
    activation["unexpected"] = True
    assert "ACTIVATION_SCHEMA_VALIDATION" in _codes(
        v.validate_activation(activation, schema, v.ACTIVATION_RECEIPT_SCHEMA_SHA256, erratum, v.RUNTIME_ERRATUM_SHA256, _sha("b"))
    )
    activation = _activation(_sha("b"))
    assert "ACTIVATION_SCHEMA_IDENTITY" in _codes(
        v.validate_activation(activation, schema, _sha("f"), erratum, v.RUNTIME_ERRATUM_SHA256, _sha("b"))
    )


def test_activation_must_bind_exact_canary_bytes() -> None:
    schema = _activation_schema()
    erratum = _runtime_erratum()
    activation = _activation(_sha("a"))
    assert "ACTIVATION_CANARY_BINDING" in _codes(
        v.validate_activation(activation, schema, v.ACTIVATION_RECEIPT_SCHEMA_SHA256, erratum, v.RUNTIME_ERRATUM_SHA256, _sha("b"))
    )


def test_activation_requires_exact_runtime_erratum() -> None:
    schema = _activation_schema()
    erratum = _runtime_erratum()
    erratum["exact_corrections"][0]["corrected_value"] = "3.11.0"
    activation = _activation(_sha("b"))
    assert "RUNTIME_ERRATUM_CORRECTIONS" in _codes(
        v.validate_activation(activation, schema, v.ACTIVATION_RECEIPT_SCHEMA_SHA256, erratum, v.RUNTIME_ERRATUM_SHA256, _sha("b"))
    )


def test_calibration_rejects_third_model_and_sample_change() -> None:
    receipt = _calibration(_sha("a"))
    receipt["referential_instrument_ids"].append("third_model")
    assert "CALIBRATION_INSTRUMENTS" in _codes(v.validate_calibration(receipt, _sha("a")))
    receipt = _calibration(_sha("a"))
    receipt["candidate_window_count"] = 136
    assert "CALIBRATION_SAMPLE" in _codes(v.validate_calibration(receipt, _sha("a")))


@pytest.mark.parametrize(("field", "value"), [("minimum_cluster_k", 4), ("complementary_suppression", False), ("rate_rounding_step", 0.05), ("lag_rounding_step_event_units", 0.25), ("raw_counts", True)])
def test_calibration_k5_rounding_and_aggregate_only(field: str, value: object) -> None:
    receipt = _calibration(_sha("a"))
    receipt["public_export"][field] = value
    assert "CALIBRATION_EXPORT" in _codes(v.validate_calibration(receipt, _sha("a")))


def test_outcome_requires_passed_calibration_and_construction() -> None:
    calibration = _calibration(_sha("a"))
    construction = _construction()
    calibration_sha = v.canonical_digest(calibration)
    construction_sha = v.canonical_digest(construction)
    seal = _seal(calibration_sha, construction_sha)
    bad_calibration = copy.deepcopy(calibration)
    bad_calibration["decision"] = "CALIBRATION_REVISE"
    assert "OUTCOME_WITHOUT_CALIBRATION" in _codes(v.validate_outcome_seal(seal, bad_calibration, calibration_sha, construction, construction_sha))
    bad_construction = copy.deepcopy(construction)
    bad_construction["falsification_gate_passed"] = False
    assert "OUTCOME_WITHOUT_CONSTRUCTION" in _codes(v.validate_outcome_seal(seal, calibration, calibration_sha, bad_construction, construction_sha))


def test_validator_has_no_restricted_discovery_or_outcome_runner() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for token in ("rglob(", "os.walk", "quarantine_root", "restricted_manifest", "subprocess", "torch.load", "mlx.core", "run_scientific_outcome"):
        assert token not in source
