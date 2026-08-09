from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/synthesize_childlens_terminal_v1_3.py"
SPEC = importlib.util.spec_from_file_location("synthesize_childlens_terminal_v1_3", SCRIPT)
assert SPEC and SPEC.loader
synth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = synth
SPEC.loader.exec_module(synth)


def valid_inputs() -> tuple[dict, dict, dict, dict]:
    thresholds = {
        name: {
            "frozen_before_author_labels": True,
            "pass_rule": f"Predeclared aggregate pass rule for {name}",
        }
        for name in synth.REQUIRED_THRESHOLD_NAMES
    }
    protocol = {
        "schema_version": synth.EXPECTED_SCHEMAS["protocol"],
        "status": "FROZEN",
        "selected_item_count": 15,
        "primary_audit_speech_seconds": 900,
        "maximum_additional_speech_seconds": 900,
        "frozen_before_author_labels": True,
        "selection_hash_deterministic": True,
        "selection_independent_of_model_predictions": True,
        "selection_independent_of_model_confidence": True,
        "selection_independent_of_lexical_content": True,
        "selection_independent_of_visual_salience": True,
        "selection_independent_of_apparent_success": True,
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
        "thresholds": thresholds,
    }
    sampling = {
        "schema_version": synth.EXPECTED_SCHEMAS["sampling"],
        "status": "AUTHOR_AUDIT_SAMPLES_READY",
        "selected_item_count": 15,
        "primary_total_seconds": 900,
        "reserve_total_seconds": 900,
        "primary_one_minute_per_item": True,
        "deficit_redistribution_used": False,
        "primary_reserve_disjoint": True,
        "all_selected_items_represented_in_primary": True,
        "all_selected_items_represented_in_reserve": True,
        "model_prediction_independent": True,
        "model_confidence_independent": True,
        "lexical_content_independent": True,
        "visual_content_independent": True,
        "selection_uses_only_official_windows_and_frozen_binding": True,
        "audit_sample_frozen_before_predictions": True,
        "author_prediction_blinding_required": True,
        "reserve_packet_sealed_separately": True,
        "prediction_payload_accepted_by_sampler": False,
        "exact_intervals_exported": False,
        "item_identifiers_exported": False,
        "maximum_reserve_activation_count": 1,
        "reserve_activation": "BORDERLINE_ONLY_AFTER_PRIMARY_AUTHOR_RECORD_LOCK",
        "frozen_v1_2_selection_digest": "a" * 64,
        "official_windows_manifest_sha256": "b" * 64,
        "sampler_policy_sha256": "c" * 64,
        "primary_packet_sha256": "d" * 64,
        "reserve_packet_sha256": "e" * 64,
    }
    pseudo = {
        "schema_version": synth.EXPECTED_SCHEMAS["pseudo"],
        "status": "COMPLETE",
        "candidate_window_count": 912,
        "candidate_speech_minutes": 135.25,
        "maximum_cpu_workers": 4,
        "maximum_mps_heavy_processes": 1,
        "fixed_versioned_instruments": True,
        "all_instrument_licenses_audited": True,
        "model_hashes_verified": True,
        "model_downloads_completed_before_restricted_processing": True,
        "local_offline_only": True,
        "network_disabled_during_restricted_inference": True,
        "inference_subprocess_network_blocked": True,
        "no_hosted_or_cloud_content_path": True,
        "quarantine_only": True,
        "checkpoint_resume_enabled": True,
        "pseudo_labels_marked_not_ground_truth": True,
        "author_audit_only_human_labeled_childlens_evidence": True,
        "simulator_oracle_labels_primary_evaluation_truth": True,
        "aggregate_safe_receipt_only": True,
        "external_api_used": False,
        "external_upload": False,
        "telemetry_enabled": False,
        "restricted_data_egress": False,
        "instrument_embeddings_entered_learner": False,
        "instrument_features_entered_learner": False,
        "instrument_weights_entered_learner": False,
        "instrument_tokenizers_entered_learner": False,
        "instrument_vocabularies_entered_learner": False,
        "instrument_scores_entered_learner": False,
        "unaudited_pseudo_labels_primary_evaluation_truth": False,
        "learner_training_executed": False,
        "corpus_tokenizer_trained": False,
        "causal_arm_executed": False,
        "scientific_acquisition_outcome_executed": False,
    }
    workflow = {
        "schema_version": synth.EXPECTED_SCHEMAS["workflow"],
        "status": "READY",
        "route": "AUTHOR_AUDIT_A",
        "audit_item_count": 15,
        "audit_speech_minutes": 15.0,
        "audit_speech_seconds": 900,
        "estimated_author_minutes": 60,
        "workflow_ready": True,
        "loopback_only": True,
        "owner_private": True,
        "autosave": True,
        "autosave_to_quarantine_only": True,
        "resumable": True,
        "predictions_hidden": True,
        "predictions_hidden_before_author_lock": True,
        "author_record_lock_immutable": True,
        "comparison_requires_lock": True,
        "selected_independent_of_model_outputs": True,
        "sample_independent_of_model_outputs": True,
        "uncertain_unusable_routes": True,
        "uncertain_route_available": True,
        "unusable_route_available": True,
        "qualification_instruction_present": True,
        "external_hosting": False,
        "network_exposure": False,
        "model_predictions_revealed_before_lock": False,
        "human_evidence_fabricated": False,
        "human_audit_complete": False,
        "prediction_join_enabled": False,
        "inter_human_reliability_available": False,
        "primary_evaluation_truth": "SIMULATOR_ORACLE_ONLY",
        "unaudited_pseudo_label_primary_evaluation_truth": False,
    }
    return protocol, sampling, pseudo, workflow


def completed_audit(status: str = "PASS") -> dict:
    return {
        "schema_version": synth.EXPECTED_SCHEMAS["audit"],
        "audit_complete": True,
        "qualified_author_confirmed": True,
        "author_used_raw_audio_video": True,
        "author_blinded_until_irreversible_lock": True,
        "author_record_irreversibly_locked": True,
        "model_predictions_revealed_only_after_lock": True,
        "sample_selection_digest_match": True,
        "thresholds_digest_match": True,
        "cluster_aware_uncertainty_reported": True,
        "small_sample_limitation_reported": True,
        "author_record_changed_after_lock": False,
        "thresholds_changed_after_author_labels": False,
        "sample_changed_after_author_labels": False,
        "human_evidence_fabricated": False,
        "inter_human_reliability_claimed": False,
        "additional_audit_speech_seconds_used": 0,
        "threshold_results": {
            name: {"status": status, "threshold_frozen": True}
            for name in synth.REQUIRED_THRESHOLD_NAMES
        },
    }


def write_inputs(root: Path, values: tuple[dict, dict, dict, dict]) -> None:
    for (name, relative), value in zip(synth.INPUT_RELATIVE_PATHS.items(), values):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")


def test_ready_requires_complete_safe_local_pipeline_and_blinded_workflow() -> None:
    assert synth.decide(*valid_inputs(), None).terminal_state == synth.AUTHOR_READY


@pytest.mark.parametrize(
    ("receipt_index", "field", "unsafe_value"),
    [
        (2, "no_hosted_or_cloud_content_path", False),
        (2, "network_disabled_during_restricted_inference", False),
        (2, "external_api_used", True),
        (2, "unaudited_pseudo_labels_primary_evaluation_truth", True),
        (1, "model_prediction_independent", False),
        (1, "lexical_content_independent", False),
        (3, "predictions_hidden", False),
        (3, "comparison_requires_lock", False),
    ],
)
def test_safety_or_independence_failure_is_fail_closed_revise(
    receipt_index: int, field: str, unsafe_value: object
) -> None:
    values = list(valid_inputs())
    values[receipt_index][field] = unsafe_value
    assert synth.decide(*values, None).terminal_state == synth.REVISE


def test_go_requires_locked_author_audit_and_every_frozen_threshold_pass() -> None:
    assert synth.decide(*valid_inputs(), completed_audit()).terminal_state == synth.GO


def test_prediction_reveal_without_lock_cannot_go() -> None:
    audit = completed_audit()
    audit["author_blinded_until_irreversible_lock"] = False
    assert synth.decide(*valid_inputs(), audit).terminal_state == synth.REVISE


def test_borderline_allows_exactly_one_bounded_escalation() -> None:
    audit = completed_audit()
    first = next(iter(synth.REQUIRED_THRESHOLD_NAMES))
    audit["threshold_results"][first]["status"] = "BORDERLINE"
    decision = synth.decide(*valid_inputs(), audit)
    assert decision.terminal_state == synth.REVISE
    assert decision.escalation_available is True
    audit["additional_audit_speech_seconds_used"] = 900
    decision = synth.decide(*valid_inputs(), audit)
    assert decision.terminal_state == synth.REVISE
    assert decision.escalation_available is False


def test_activated_reserve_pending_author_lock_cannot_be_activated_again() -> None:
    audit = completed_audit()
    first = next(iter(synth.REQUIRED_THRESHOLD_NAMES))
    audit["threshold_results"][first]["status"] = "BORDERLINE"
    audit["reserve_activation"] = {
        "eligible": True,
        "activated_once": True,
        "pending_author_lock": True,
    }
    audit["additional_audit_speech_seconds_used"] = 0
    decision = synth.decide(*valid_inputs(), audit)
    assert decision.terminal_state == synth.REVISE
    assert decision.escalation_available is False
    assert decision.author_action_only_remaining is True
    assert decision.basis_code == (
        "ONE_TIME_BORDERLINE_RESERVE_ALREADY_ACTIVATED_AND_PENDING_AUTHOR_LOCK"
    )
    assert "WITHOUT_A_SECOND_ACTIVATION" in decision.exact_next_task


def test_pending_reserve_without_prior_activation_fails_audit_shape() -> None:
    audit = completed_audit()
    audit["reserve_activation"] = {
        "eligible": True,
        "activated_once": False,
        "pending_author_lock": True,
    }
    assert synth._audit_valid_shape(audit) is False
    assert synth.decide(*valid_inputs(), audit).terminal_state == synth.REVISE


def test_stop_requires_explicit_locked_high_grade_fundamental_failure() -> None:
    audit = completed_audit("HARD_FAIL")
    assert synth.decide(*valid_inputs(), audit).terminal_state == synth.REVISE
    audit["fundamental_calibration_failure_established"] = True
    audit["fundamental_failure_evidence_grade"] = "LOCKED_QUALIFIED_AUTHOR_AUDIT"
    assert synth.decide(*valid_inputs(), audit).terminal_state == synth.STOP


def test_synthesize_writes_only_one_terminal_literal_per_decision_output(tmp_path: Path) -> None:
    write_inputs(tmp_path, valid_inputs())
    decision = synth.synthesize(tmp_path, check_historical=False)
    assert decision.terminal_state == synth.AUTHOR_READY
    record = json.loads((tmp_path / synth.OUTPUT_RELATIVE_PATHS["decision"]).read_text())
    report = (tmp_path / synth.OUTPUT_RELATIVE_PATHS["executive"]).read_text()
    assert record["terminal_state"] == synth.AUTHOR_READY
    assert synth.TERMINAL_RE.findall(report) == [synth.AUTHOR_READY]


def test_restricted_receipt_field_is_rejected_before_synthesis(tmp_path: Path) -> None:
    values = valid_inputs()
    values[2]["transcript_text"] = "restricted"
    write_inputs(tmp_path, values)
    with pytest.raises(synth.SynthesisError, match="E_INPUT_RESTRICTED_FIELD"):
        synth.synthesize(tmp_path, check_historical=False)


def test_actual_historical_namespaces_match_synthesizer_baselines() -> None:
    status = synth._history_status(REPO_ROOT)
    assert all(entry["preserved"] is True for entry in status.values())
