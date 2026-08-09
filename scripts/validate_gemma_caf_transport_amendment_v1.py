#!/usr/bin/env python3
"""Validate the public-only AIFF-to-CAF transport amendment.

The validator reads only explicit public contract/receipt paths. It has no model,
media, quarantine, or restricted-data execution path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


AMENDMENT_SCHEMA = "nursery-gemma4-public-canary-caf-transport-amendment-v1"
AMENDMENT_SHA256 = "3ecf4bf6c62920d73177d029c18918f38366d34f7d7997642037696def37bbd0"
BASE_CONTRACT_SHA256 = "9b894fcfd47df93824d25a467961d2c9f3398666dbe824367f2d8e1afe0635e9"
RUNTIME_ERRATUM_SHA256 = "b84da17121dd3235b1b6799d91c077d542e30e384f5cff5e131ec6c9755cecc1"
ACTIVATION_SCHEMA_SHA256 = "0668d8d304a1be30e43e9df2c05469a338d647c262418a04c6d067d2e77f8b05"
PRIOR_TERMINAL_SHA256 = "046b06209bd644ae5f3ebcb0f83bb37edca42f0fdb3a1dfff2bb492a0ffecb90"
PRIOR_VALIDATION_SHA256 = "c5a94a2098a7e07e0d43c77da9d018444773519a0e5adaca276fcaa0ff2340c3"
PRIOR_FAILURE_SHA256 = "ef5031b3b90a3611eb488ff9fa76a48e0b2f2a0af0ab5e2230316d3234a0eda6"
PRIOR_DISPOSITION_SHA256 = "ee04891eb010a5244ec5de2a813154c58d9e48a7ddfbcc6612106ed759b74a7d"
NO_FALLBACK_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
NEW_NAMESPACE = "output/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1"
NEW_OUTPUTS = {
    "future_public_canary_receipt": "public_common_schema_canary_receipt_v1.json",
    "future_public_canary_failure": "public_common_schema_canary_failure_v1.json",
    "future_runtime_lock_receipt": "runtime_lock_receipt_v1.json",
    "future_activation_receipt": "full_activation_receipt_v1.json",
}
OLD_INVOCATION = "/usr/bin/say -v Anna -r 170 -o <owner-private-public-scratch>/german.aiff --data-format=LEI16@22050 'Nimm die rote Tasse.'"
NEW_INVOCATION = "/usr/bin/say -v Anna -r 170 -o <owner-private-public-scratch>/german.caf --data-format=LEI16@22050 'Nimm die rote Tasse.'"
OLD_NORMALIZATION = "/opt/homebrew/bin/ffmpeg -nostdin -hide_banner -loglevel error -i german.aiff -af apad=pad_dur=10 -t 10 -vn -ac 1 -ar 16000 -c:a pcm_s16le -bitexact german.wav"
NEW_NORMALIZATION = "/opt/homebrew/bin/ffmpeg -nostdin -hide_banner -loglevel error -i german.caf -af apad=pad_dur=10 -t 10 -vn -ac 1 -ar 16000 -c:a pcm_s16le -bitexact german.wav"


@dataclass(frozen=True)
class Issue:
    code: str
    location: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "location": self.location, "message": self.message}


def _issue(issues: list[Issue], code: str, location: str, message: str) -> None:
    issues.append(Issue(code, location, message))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _pointer_get(document: Mapping[str, Any], pointer: str) -> Any:
    value: Any = document
    if not pointer.startswith("/"):
        raise KeyError(pointer)
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            raise KeyError(pointer)
        value = value[token]
    return value


def _pointer_set(document: dict[str, Any], pointer: str, replacement: Any) -> None:
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]
    value: Any = document
    for token in tokens[:-1]:
        value = value[token]
    value[tokens[-1]] = replacement


def apply_caf_overlay(base_contract: Mapping[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(dict(base_contract))
    _pointer_set(effective, "/public_fixture/audio/local_TTS/invocation", NEW_INVOCATION)
    _pointer_set(effective, "/public_fixture/audio/ffmpeg_normalization", NEW_NORMALIZATION)
    return effective


def validate_prior_receipts(
    terminal: Mapping[str, Any],
    terminal_sha256: str,
    validation: Mapping[str, Any],
    validation_sha256: str,
    failure: Mapping[str, Any],
    failure_sha256: str,
    disposition_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if terminal_sha256 != PRIOR_TERMINAL_SHA256 or terminal.get("terminal_state") != "PROTOTYPE_TECHNICAL_REVISE_PRE_OUTCOME":
        _issue(issues, "PRIOR_TERMINAL_IMMUTABILITY", "prior_terminal", "exact prior technical-REVISE decision is required")
    basis = _mapping(terminal.get("decision_basis"))
    if basis.get("fresh_public_common_schema_canary") != "TECHNICAL_FAIL_BEFORE_MODEL_LOAD" or basis.get("parser_corrections_used") != 0 or basis.get("restricted_calibration_run") is not False or basis.get("causal_outcome_run") is not False:
        _issue(issues, "PRIOR_TERMINAL_STATE", "prior_terminal.decision_basis", "prior pre-model technical failure was reinterpreted")
    if validation_sha256 != PRIOR_VALIDATION_SHA256 or validation.get("status") != "VALID_TECHNICAL_REVISE_PRE_OUTCOME":
        _issue(issues, "PRIOR_VALIDATION_IMMUTABILITY", "prior_validation", "exact prior validation receipt is required")
    disposition = _mapping(validation.get("gate_disposition"))
    for field in ("parser_correction_withheld_as_inapplicable", "canonical_canary_activation_outputs_absent", "restricted_calibration_output_absent", "preoutcome_and_authorization_outputs_absent", "causal_output_root_absent"):
        if disposition.get(field) is not True:
            _issue(issues, "PRIOR_VALIDATION_STATE", f"prior_validation.gate_disposition.{field}", "prior fail-closed state changed")
    if failure_sha256 != PRIOR_FAILURE_SHA256 or failure.get("status") != "REVISE" or failure.get("restricted_payload_accessed") is not False or failure.get("raw_model_output_exported") is not False:
        _issue(issues, "PRIOR_FAILURE_IMMUTABILITY", "prior_failure", "exact safe pre-model failure receipt is required")
    if disposition_sha256 != PRIOR_DISPOSITION_SHA256:
        _issue(issues, "PRIOR_DISPOSITION_IMMUTABILITY", "prior_disposition", "prior narrative disposition bytes changed")
    return issues


def validate_no_fallback(override: Mapping[str, Any], override_sha256: str) -> list[Issue]:
    issues: list[Issue] = []
    if override_sha256 != NO_FALLBACK_SHA256 or override.get("schema_version") != "nursery-prototype-no-automatic-fallback-override-v1":
        _issue(issues, "NO_FALLBACK_IDENTITY", "no_fallback_override", "exact no-automatic-fallback override is required")
    expected = {"QWEN3_ONLY_CALIBRATION", "CONSERVATIVE_UNBOUNDED_OR_FULL_DOMAIN_SYNTHETIC_SENSITIVITY", "EMPIRICAL_FREE_AUTOMATIC_OUTCOME", "THIRD_VISUAL_MODEL", "ALTERED_CLAIM_WORDING_TO_ENABLE_AN_OUTCOME"}
    observed = override.get("automatic_fallbacks_forbidden")
    if not isinstance(observed, list) or len(observed) != 5 or set(observed) != expected:
        _issue(issues, "NO_FALLBACK_SET", "no_fallback_override.automatic_fallbacks_forbidden", "automatic fallback set changed")
    failure = _mapping(override.get("failure_behavior"))
    if failure.get("restricted_calibration_fallback_allowed") is not False or failure.get("scientific_endpoint_allowed") is not False:
        _issue(issues, "NO_FALLBACK_BEHAVIOR", "no_fallback_override.failure_behavior", "failed Gemma must stop without automatic scientific branch")
    return issues


def validate_amendment(
    amendment: Mapping[str, Any],
    amendment_sha256: str,
    base_contract: Mapping[str, Any],
    base_contract_sha256: str,
) -> list[Issue]:
    issues: list[Issue] = []
    if amendment_sha256 != AMENDMENT_SHA256 or amendment.get("schema_version") != AMENDMENT_SCHEMA or amendment.get("status") != "FROZEN_PRE_CANARY_CAF_TRANSPORT_ONLY":
        _issue(issues, "CAF_AMENDMENT_IDENTITY", "amendment", "CAF amendment bytes/schema/status changed")
    if amendment.get("amendment_mode") != "ADDITIVE_CONTENT_INDEPENDENT_TRANSPORT_OVERLAY" or amendment.get("historical_artifacts_modified") is not False:
        _issue(issues, "CAF_ADDITIVE_BOUNDARY", "amendment", "CAF amendment must be additive and preserve history")
    for field in ("public_canary_run", "restricted_content_accessed", "restricted_inference_run", "scientific_outcome_run"):
        if amendment.get(field) is not False:
            _issue(issues, "CAF_FREEZE_ORDER", f"amendment.{field}", "amendment was not frozen before canary/restricted/outcome work")
    trigger = _mapping(amendment.get("trigger"))
    if trigger.get("classification") != "TECHNICAL_FAIL_BEFORE_MODEL_LOAD" or trigger.get("model_loaded") is not False or trigger.get("model_generation_called") is not False or trigger.get("raw_model_output_existed") is not False or trigger.get("semantic_or_scientific_evidence_involved") is not False or trigger.get("outer_fence_parser_corrections_spent") != 0:
        _issue(issues, "CAF_TRIGGER_SCOPE", "amendment.trigger", "CAF trigger was not a content-independent pre-model transport failure")

    bindings = _mapping(amendment.get("immutable_bindings"))
    expected_bindings = {
        "prior_common_schema_canary_contract": BASE_CONTRACT_SHA256,
        "runtime_pin_erratum": RUNTIME_ERRATUM_SHA256,
        "activation_receipt_schema": ACTIVATION_SCHEMA_SHA256,
        "calibration_protocol": "db5336726f4841e592cb9c73df5ab22a0887d1bd34c7cf673146f38de9e86961",
        "calibration_conditioned_sensitivity_protocol": "c6b7afab6c8914a968b0f06706a222d2771863c2a3fd2017184c17d431bfa7d6",
    }
    for field, digest in expected_bindings.items():
        if _mapping(bindings.get(field)).get("sha256") != digest:
            _issue(issues, "CAF_IMMUTABLE_BINDING", f"amendment.immutable_bindings.{field}", "bound predecessor digest changed")
    if base_contract_sha256 != BASE_CONTRACT_SHA256:
        _issue(issues, "CAF_BASE_CONTRACT_BYTES", "base_contract", "base canary contract bytes changed")

    delta = _mapping(amendment.get("exact_transport_delta"))
    changes = delta.get("changes")
    expected_changes = [
        ("/public_fixture/audio/local_TTS/invocation", OLD_INVOCATION, NEW_INVOCATION),
        ("/public_fixture/audio/ffmpeg_normalization", OLD_NORMALIZATION, NEW_NORMALIZATION),
    ]
    observed: list[tuple[Any, Any, Any]] = []
    if isinstance(changes, list):
        for row in changes:
            mapped = _mapping(row)
            observed.append((mapped.get("JSON_pointer"), mapped.get("frozen_value"), mapped.get("amended_value")))
    if delta.get("maximum_changed_contract_fields") != 2 or observed != expected_changes:
        _issue(issues, "CAF_EXACT_DELTA", "amendment.exact_transport_delta", "exact two-field AIFF-to-CAF delta changed")
    for pointer, frozen, _ in expected_changes:
        try:
            actual = _pointer_get(base_contract, pointer)
        except KeyError:
            actual = None
        if actual != frozen:
            _issue(issues, "CAF_POINTER_PRECONDITION", f"base_contract{pointer}", "frozen pointer/value precondition failed")
    for field in ("semantic_fields_changed", "model_runtime_or_artifact_fields_changed", "scientific_fields_changed"):
        if delta.get(field) != 0:
            _issue(issues, "CAF_SCOPE_EXPANSION", f"amendment.exact_transport_delta.{field}", "non-transport field changed")
    if delta.get("parser_allowance_changed") is not False or delta.get("does_not_spend_outer_fence_parser_correction") is not True:
        _issue(issues, "CAF_PARSER_BUDGET", "amendment.exact_transport_delta", "parser allowance was changed or spent")

    effective = apply_caf_overlay(base_contract)
    audio = _mapping(_mapping(effective.get("public_fixture")).get("audio"))
    if _mapping(audio.get("local_TTS")).get("invocation") != NEW_INVOCATION or audio.get("ffmpeg_normalization") != NEW_NORMALIZATION:
        _issue(issues, "CAF_OVERLAY_APPLICATION", "effective_contract.public_fixture.audio", "CAF overlay failed")
    preserved = _mapping(amendment.get("preserved_public_fixture_contract"))
    final_wav = _mapping(preserved.get("final_WAV"))
    expected_wav = {"filename": "german.wav", "container": "WAV", "channels": 1, "sample_format": "signed 16-bit little-endian PCM", "sample_rate_hz": 16000, "sample_count": 160000, "duration_seconds": 10.0, "normalization_or_loudness_filter": False, "final_hash_bound_in_future_canary_receipt": True}
    if dict(final_wav) != expected_wav:
        _issue(issues, "CAF_FINAL_WAV", "amendment.preserved_public_fixture_contract.final_WAV", "final WAV contract changed")
    expected_preserved = {"phrase_utf8": "Nimm die rote Tasse.", "language": "de-DE", "voice": "Anna", "voice_locale": "de_DE", "rate_words_per_minute": 170, "TTS_data_format": "LEI16@22050", "frame_count": 5, "frame_offsets_seconds": [-5.0, -2.5, 0.0, 2.5, 5.0], "exact_prompt_sha256": "93e51fb982bd3ea47dd4defbd24e818cf67aee89c6df9d98fd1ec40fc2188ff6", "exact_common_schema_sha256": "19b7ae8948aed46410e1307c0a302303813422995c8f4cdc19949118693a8089", "exact_parser_contract_sha256": "af8e5988febb42e378bb3b6c8cbbb2799cca167cb4b3f4be5d32732a6035974a", "outer_fence_parser_correction_maximum": 1, "outer_fence_parser_correction_used_before_amendment": 0}
    for field, value in expected_preserved.items():
        if preserved.get(field) != value:
            _issue(issues, "CAF_PRESERVED_FIXTURE", f"amendment.preserved_public_fixture_contract.{field}", "non-container fixture/parser field changed")

    sample = _mapping(amendment.get("preserved_sample_export_and_gate_contract"))
    for field, value in {"primary_item_count": 15, "primary_total_seconds": 900, "candidate_window_count": 137, "reserve_activation": False, "reselection": False, "minimum_export_cell_items": 5, "K5_complement_protection": True, "outward_rounding_unchanged": True, "agreement_is_not_a_pass_gate": True, "simulator_oracle_remains_only_evaluation_truth": True, "causal_thresholds_changed": False, "calibration_range_or_scientific_protocol_changed": False}.items():
        if sample.get(field) != value:
            _issue(issues, "CAF_SCIENTIFIC_CONTRACT", f"amendment.preserved_sample_export_and_gate_contract.{field}", "sample/export/scientific gate changed")

    namespace = _mapping(amendment.get("new_immutable_output_namespace"))
    if namespace.get("path") != NEW_NAMESPACE or namespace.get("must_not_exist_before_execution") is not True or namespace.get("historical_output_paths_overwritten") is not False:
        _issue(issues, "CAF_OUTPUT_NAMESPACE", "amendment.new_immutable_output_namespace", "new namespace is not fail-closed against overwrite")
    for field, filename in NEW_OUTPUTS.items():
        if namespace.get(field) != filename:
            _issue(issues, "CAF_OUTPUT_NAMESPACE", f"amendment.new_immutable_output_namespace.{field}", "CAF output filename changed")
    rule = _mapping(amendment.get("overlay_application_rule"))
    if rule.get("reject_if_any_bound_digest_differs") is not True or rule.get("reject_if_any_unlisted_field_changes") is not True or rule.get("reuse_prior_AIFF_failure_or_success_receipt_as_CAF_receipt") is not False:
        _issue(issues, "CAF_APPLICATION_RULE", "amendment.overlay_application_rule", "CAF overlay no-reuse/no-unlisted-change rule weakened")
    authorization = _mapping(amendment.get("authorization_boundary"))
    for field in ("amendment_is_canary_execution", "amendment_is_activation_receipt", "public_canary_authorized_by_amendment_alone", "restricted_inference_authorized", "scientific_outcome_authorized", "model_agreement_or_pseudo_labels_are_ground_truth"):
        if authorization.get(field) is not False:
            _issue(issues, "CAF_AUTHORIZATION_BOUNDARY", f"amendment.authorization_boundary.{field}", "transport amendment improperly authorized work/truth")
    return issues


def validate_revised_chain_receipt(receipt: Mapping[str, Any], amendment_sha256: str) -> list[Issue]:
    """Validate the common additive bindings shared by future CAF receipts."""

    issues: list[Issue] = []
    if receipt.get("caf_transport_amendment_sha256") != amendment_sha256 or receipt.get("caf_transport_amendment_path") != "docs/nursery_program_convergence_v1/gemma4_caf_canary_transport_v1/frozen_caf_transport_amendment_v1.json":
        _issue(issues, "CAF_RECEIPT_BINDING", "receipt", "revised receipt is not byte-bound to CAF amendment")
    if receipt.get("no_automatic_fallback_override_sha256") != NO_FALLBACK_SHA256:
        _issue(issues, "CAF_RECEIPT_FALLBACK_BINDING", "receipt", "revised receipt is not bound to no-fallback override")
    if receipt.get("output_namespace") != NEW_NAMESPACE or receipt.get("historical_output_overwritten") is not False or receipt.get("canonical_output_overwritten") is not False:
        _issue(issues, "CAF_RECEIPT_OUTPUT", "receipt", "revised receipt reused or overwrote historical/canonical output")
    if receipt.get("intermediate_container") != "CAF" or receipt.get("final_wav_contract_unchanged") is not True:
        _issue(issues, "CAF_RECEIPT_TRANSPORT", "receipt", "revised receipt changed more than intermediate transport")
    if receipt.get("parser_corrections_spent_before_revised_canary") != 0 or receipt.get("parser_allowance_preserved") is not True:
        _issue(issues, "CAF_RECEIPT_PARSER", "receipt", "parser budget was spent or altered by transport repair")
    for field in ("restricted_content_accessed_before_activation", "restricted_inference_run", "scientific_outcome_run"):
        if receipt.get(field) is not False:
            _issue(issues, "CAF_RECEIPT_ORDER", f"receipt.{field}", "revised receipt crossed activation/outcome ordering")
    return issues


def _load(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, _sha256(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--prior-terminal", type=Path, required=True)
    parser.add_argument("--prior-validation", type=Path, required=True)
    parser.add_argument("--prior-failure", type=Path, required=True)
    parser.add_argument("--prior-disposition", type=Path, required=True)
    parser.add_argument("--no-fallback-override", type=Path, required=True)
    parser.add_argument("--revised-receipt", type=Path)
    args = parser.parse_args(argv)

    amendment, amendment_sha = _load(args.amendment)
    base, base_sha = _load(args.base_contract)
    terminal, terminal_sha = _load(args.prior_terminal)
    validation, validation_sha = _load(args.prior_validation)
    failure, failure_sha = _load(args.prior_failure)
    disposition_sha = _sha256(args.prior_disposition.read_bytes())
    override, override_sha = _load(args.no_fallback_override)
    issues = validate_prior_receipts(terminal, terminal_sha, validation, validation_sha, failure, failure_sha, disposition_sha)
    issues.extend(validate_no_fallback(override, override_sha))
    issues.extend(validate_amendment(amendment, amendment_sha, base, base_sha))
    if args.revised_receipt:
        revised, _ = _load(args.revised_receipt)
        issues.extend(validate_revised_chain_receipt(revised, amendment_sha))
    result = {"schema_version": "nursery-gemma-caf-transport-validation-v1", "status": "PASS" if not issues else "FAIL", "issue_count": len(issues), "issues": [issue.as_dict() for issue in issues], "restricted_access": False, "model_or_outcome_run": False}
    print(json.dumps(result, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
