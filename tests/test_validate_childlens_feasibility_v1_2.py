from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_childlens_feasibility_v1_2.py"
SPEC = importlib.util.spec_from_file_location("validate_childlens_feasibility_v1_2", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _copy_history(root: Path) -> None:
    for relative in (
        "docs/childlens_feasibility_v1",
        "output/childlens_feasibility_v1",
        "docs/childlens_feasibility_v1_1",
        "output/childlens_feasibility_v1_1",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPO_ROOT / relative, target)
    for relative in (
        "scripts/validate_childlens_feasibility_v1.py",
        "tests/test_childlens_feasibility_v1.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    for directory, pattern in validator.V1_1_CODE_GLOBS:
        for source in (REPO_ROOT / directory).glob(pattern):
            target = root / directory / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _safe_receipts(state: str = "CHILDLENS_HUMAN_VALIDATION_READY") -> dict[str, dict]:
    transfer_digest = "a" * 64
    manifest_digest = "b" * 64
    measurement_digest = "c" * 64
    workflow_digest = "d" * 64
    pre_namespace = 20 * 1024**2
    pre_free = 131_236_425_728
    acquisition = {
        "schema_version": "childlens-v1.2-acquisition-receipt-v1",
        "selected_media_count": 15,
        "acquired_media_count": 15,
        "verified_against_frozen_selection_count": 15,
        "acquisition_complete": True,
        "raw_media_bytes_acquired": 8 * 1024**3,
        "raw_cap_bytes": 20 * 1024**3,
        "storage_admission_method": "CONSERVATIVE_DISPLAY_BOUND_WITH_HARD_COUNTER",
        "displayed_parsed_total_bytes": validator.FROZEN_DISPLAY_TOTAL_BYTES,
        "rounding_upper_bound_bytes": validator.FROZEN_ROUNDING_UPPER_BYTES,
        "transfer_overhead_bytes": validator.FROZEN_TRANSFER_OVERHEAD_BYTES,
        "pre_transfer_upper_bound_bytes": validator.FROZEN_CONSERVATIVE_ADMISSION_BYTES,
        "admission_ceiling_bytes": validator.ADMISSION_CEILING_BYTES,
        "upper_bound_margin_to_raw_cap_bytes": validator.RAW_CAP_BYTES - validator.FROZEN_CONSERVATIVE_ADMISSION_BYTES,
        "hard_abort_before_bytes": 20 * 1024**3 - 1,
        "predeclared_nonraw_reserve_bytes": validator.NONRAW_RESERVE_BYTES,
        "namespace_cap_bytes": validator.NAMESPACE_CAP_BYTES,
        "projected_post_peak_free_floor_bytes": validator.POST_PEAK_FREE_FLOOR_BYTES,
        "pre_transfer_namespace_bytes": pre_namespace,
        "pre_transfer_free_bytes": pre_free,
        "projected_namespace_peak_bytes": pre_namespace + validator.FROZEN_CONSERVATIVE_ADMISSION_BYTES + validator.NONRAW_RESERVE_BYTES,
        "projected_post_peak_free_bytes": pre_free - validator.FROZEN_CONSERVATIVE_ADMISSION_BYTES - validator.NONRAW_RESERVE_BYTES,
        "post_transfer_free_bytes": 110 * 1024**3,
        "namespace_bytes_after_acquisition": 9 * 1024**3,
        "frozen_selection_digest": validator.FROZEN_SELECTION_DIGEST,
        "immutable_public_snapshot_commit": validator.PUBLIC_SNAPSHOT_COMMIT,
        "pilot_source": "ACCESSIBLE_LIVE_VIEW_BOUND_TO_RESTRICTED_MANIFEST_DIGEST",
        "live_to_public_snapshot_byte_equivalence": "NOT_PROVEN",
        "live_snapshot_limitation_disclosed": True,
        "release_binding_verified": True,
        "restricted_manifest_digest_verified": True,
        "restricted_manifest_sha256": manifest_digest,
        "restricted_native_transfer_receipt_sha256": transfer_digest,
        "retention_deadline": validator.RETENTION_DEADLINE,
        "retention_control_recorded": True,
        "full_archive_downloaded": False,
        "selection_changed": False,
        "selection_content_inspected_before_transfer": False,
        "duplicate_full_resolution_copies_created": False,
        "source_filenames_exported": False,
        "restricted_identifiers_exported": False,
        "exact_timestamps_exported": False,
        "restricted_manifest_exported": False,
        "external_or_cloud_transfer": False,
        "sequential_or_tightly_bounded_transfer": True,
        "cumulative_stream_counter_enforced": True,
    }
    all_pass_metric = {
        "status": "ALL_PASS",
        "pass_count": 15,
        "fail_count": 0,
        "cell_suppressed": False,
    }
    diagnostics = {
        "schema_version": "childlens-v1.2-automated-diagnostics-receipt-v1",
        "aggregate_only": True,
        "local_offline_only": True,
        "automated_outputs_not_gold": True,
        "media_preparation_attempted_count": 15,
        "source_acquired_media_count": 15,
        "container_probe_attempted_count": 15,
        "container_probe_success_count": 15,
        "decode_attempted_count": 15,
        "decode_success_count": 15,
        "audio_stream_media_count": 15,
        "speech_presence_window_media_count": 15,
        "cell_suppression_k": 5,
        "structural_metrics": {
            name: dict(all_pass_metric) for name in validator.REQUIRED_STRUCTURAL_METRICS
        },
        "small_nonzero_failure_cells_exported": False,
        "measurement_source": "LOCAL_RESTRICTED_MEDIA_AUDIT_V1_2",
        "measurement_executed_on_acquired_bytes": True,
        "speech_window_actual_source": "OFFICIAL_ANNOTATION_WINDOWS_LINKED_TO_ACQUIRED_MEDIA",
        "annotation_linkage_actual_source": "RESTRICTED_CANONICAL_MANIFEST",
        "pilot_selection_sha256": validator.FROZEN_SELECTION_DIGEST,
        "source_native_transfer_receipt_sha256": transfer_digest,
        "restricted_measurement_receipt_sha256": measurement_digest,
        "official_speech_window_union_minutes": 32.5,
        "automated_preparation_complete": True,
        "diagnostics_passed_for_human_workflow": True,
        "lexical_content_exported": False,
        "transcript_text_exported": False,
        "speaker_labels_exported": False,
        "frames_exported": False,
        "external_api_used": False,
        "learner_data_created": False,
        "instrument_features_passed_to_learner": False,
        "instrument_embeddings_passed_to_learner": False,
        "instrument_tokenizer_or_vocabulary_passed_to_learner": False,
    }
    workflow = {
        "schema_version": "childlens-v1.2-human-workflow-receipt-v1",
        "workflow_populated": True,
        "ready_for_authorized_human": True,
        "local_only": True,
        "autosave_to_restricted_quarantine_only": True,
        "resumable_batches": True,
        "input_validation_enabled": True,
        "instrument_condition_blinded": True,
        "identifiers_displayed": False,
        "source_filenames_displayed": False,
        "exact_timestamps_exported": False,
        "transcript_text_exported": False,
        "external_service_used": False,
        "human_evidence_fabricated": False,
        "genuine_human_validation_complete": False,
        "independent_reliability_requirement_complete": False,
        "selected_media_source_count": 15,
        "candidate_unit_type": "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE",
        "populated_candidate_utterance_count": 0,
        "populated_candidate_speech_window_count": 64,
        "populated_candidate_speech_window_minutes": 32.5,
        "double_code_fraction_target": 0.20,
        "referential_assignment_eligible_count": 0,
        "referential_double_code_assigned_count": 0,
        "referential_double_code_assigned_fraction": None,
        "referential_double_code_sample_frozen": False,
        "referential_assignment_rule_frozen": True,
        "referential_inventory_pending_human_segmentation": True,
        "language_establishment_required_before_transcription": True,
        "language_matched_humans_required": True,
        "human_qualification_screening_enabled": True,
        "language_routes_blinded_and_resumable": True,
        "actual_language_established_by_qualified_humans": False,
        "language_matched_coders_verified": False,
        "frozen_reliability_thresholds_pass": False,
        "estimated_first_authorized_human_minutes": 420,
        "estimated_second_independent_human_minutes": 240,
        "estimated_adjudication_minutes": 120,
        "estimated_total_human_minutes": 780,
        "workflow_source": "LOCAL_RESTRICTED_HUMAN_VALIDATION_PACKET_V1_2",
        "packet_populated_from_actual_acquired_media": True,
        "pilot_selection_sha256": validator.FROZEN_SELECTION_DIGEST,
        "source_native_transfer_receipt_sha256": transfer_digest,
        "source_measurement_receipt_sha256": measurement_digest,
        "restricted_workflow_receipt_sha256": workflow_digest,
    }
    decision = {
        "schema_version": "childlens-v1.2-decision-record-v1",
        "terminal_state": state,
        "historical_v1_preserved": True,
        "historical_v1_1_preserved": True,
        "permission_reopened": False,
        "selection_reopened": False,
        "scientific_thresholds_changed": False,
        "learner_training_executed": False,
        "scientific_outcome_executed": False,
        "causal_arm_executed": False,
        "tokenizer_trained": False,
        "checkpoint_created": False,
        "aea_empirical_ancestry": False,
        "babyview_empirical_ancestry": False,
        "restricted_data_egress": False,
        "corpus_payload_exposed_in_tool_output": False,
        "restricted_payload_in_repository": False,
        "external_instrument_artifacts_entered_learner": False,
        "local_tool_log_quarantine_path_redaction_incident_disclosed": True,
        "local_tool_log_quarantine_path_redaction_incident_summary": "RESTRICTED_QUARANTINE_PATH_APPEARED_IN_LOCAL_TOOL_LOG_REDACTED_FROM_REPOSITORY_AND_USER_FACING_ARTIFACTS_NO_CORPUS_PAYLOAD",
        "only_remaining_evidence_is_authorized_human_validation": True,
        "all_essential_frozen_gates_pass": False,
        "bounded_problem_remaining": state == "CHILDLENS_FEASIBILITY_REVISE",
        "fundamental_corpus_failure_established": state == "CHILDLENS_FEASIBILITY_STOP",
    }
    return {
        "acquisition_receipt.json": acquisition,
        "automated_diagnostics_receipt.json": diagnostics,
        "human_validation_workflow_receipt.json": workflow,
        "decision_record.json": decision,
    }


def _fixture(tmp_path: Path, state: str = "CHILDLENS_HUMAN_VALIDATION_READY") -> Path:
    root = tmp_path / "repo"
    _copy_history(root)
    docs = root / "docs/childlens_feasibility_v1_2"
    output = root / "output/childlens_feasibility_v1_2"
    docs.mkdir(parents=True)
    output.mkdir(parents=True)
    for name in validator.DOCS_REQUIRED:
        (docs / name).write_text(
            f"# Synthetic aggregate report\n\nTerminal: {state}. No corpus payload is included.\n",
            encoding="utf-8",
        )
    for name, value in _safe_receipts(state).items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    immutability = {
        "schema_version": "childlens-v1.2-immutability-receipt-v1",
        "v1_artifact_count": validator.EXPECTED_V1_COUNT,
        "v1_set_digest": validator.EXPECTED_V1_DIGEST,
        "v1_1_artifact_count": validator.EXPECTED_V1_1_COUNT,
        "v1_1_set_digest": validator.EXPECTED_V1_1_DIGEST,
        "v1_preserved": True,
        "v1_1_preserved": True,
        "historical_decisions_overwritten": False,
    }
    (output / "immutability_receipt.json").write_text(json.dumps(immutability, indent=2) + "\n", encoding="utf-8")
    return root


def _codes(root: Path) -> set[str]:
    return {issue.code for issue in validator.validate(root)}


def test_synthetic_human_ready_fixture_passes(tmp_path: Path) -> None:
    assert validator.validate(_fixture(tmp_path)) == []


def test_human_ready_requires_all_fifteen_verified(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/acquisition_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["verified_against_frozen_selection_count"] = 14
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "READY_WITHOUT_COMPLETE_ACQUISITION" in _codes(root)


def test_human_ready_requires_populated_frozen_minimum(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["populated_candidate_speech_window_minutes"] = 29.9
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "READY_WITHOUT_USABLE_WORKFLOW" in _codes(root)


def test_receipts_crosscheck_acquisition_decode_and_workflow_counts(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    diagnostics_path = root / "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["source_acquired_media_count"] = 14
    diagnostics["decode_success_count"] = 16
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    workflow_path = root / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["selected_media_source_count"] = 14
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    codes = _codes(root)
    assert "DIAGNOSTIC_ACQUISITION_CROSSCHECK_MISMATCH" in codes
    assert "DIAGNOSTIC_SUCCESS_COUNT_INVALID" in codes
    assert "READY_WITHOUT_USABLE_WORKFLOW" in codes


def test_human_ready_rejects_fabricated_human_evidence(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["human_evidence_fabricated"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    codes = _codes(root)
    assert "WORKFLOW_BOUNDARY_VIOLATION" in codes
    assert "READY_HUMAN_EVIDENCE_STATE_INVALID" in codes


def test_go_requires_genuine_independent_human_evidence(tmp_path: Path) -> None:
    root = _fixture(tmp_path, "CHILDLENS_FEASIBILITY_GO")
    assert "GO_WITHOUT_GENUINE_HUMAN_EVIDENCE" in _codes(root)


def test_exactly_one_terminal_literal_is_required(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "docs/childlens_feasibility_v1_2/extra.md"
    path.write_text("CHILDLENS_FEASIBILITY_REVISE\n", encoding="utf-8")
    assert "TERMINAL_STATE_SET_INVALID" in _codes(root)


def test_restricted_filename_path_id_time_and_transcript_are_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    docs = root / "docs/childlens_feasibility_v1_2"
    cases = {
        "filename.md": "restricted_sample.mp4",
        "path.md": "/Users/example/hidden/place",
        "id.md": "123e4567-e89b-12d3-a456-426614174000",
        "time.md": "00:01:02.300",
        "transcript.md": "CHILD: synthetic words",
    }
    for name, content in cases.items():
        (docs / name).write_text(content, encoding="utf-8")
    codes = _codes(root)
    assert "MEDIA_FILENAME_EXPORTED" in codes
    assert "ABSOLUTE_LOCAL_PATH_EXPORTED" in codes
    assert "IDENTIFIER_LIKE_TEXT_EXPORTED" in codes
    assert "EXACT_MEDIA_TIME_EXPORTED" in codes
    assert "TRANSCRIPT_LIKE_TEXT_EXPORTED" in codes


def test_restricted_machine_field_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/extra.json"
    path.write_text(json.dumps({"participant_id": "synthetic"}), encoding="utf-8")
    assert "RESTRICTED_MACHINE_FIELD" in _codes(root)


def test_media_payload_file_is_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/aggregate.wav"
    path.write_bytes(b"synthetic")
    assert "PROHIBITED_PAYLOAD_EXTENSION" in _codes(root)


def test_conservative_storage_amendment_requires_two_gib_margin(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/acquisition_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["pre_transfer_upper_bound_bytes"] = 19 * 1024**3
    value["upper_bound_margin_to_raw_cap_bytes"] = 1024**3
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "CONSERVATIVE_ADMISSION_ARITHMETIC_INVALID" in _codes(root)


def test_cross_corpus_and_outcome_flags_must_be_false(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/decision_record.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["aea_empirical_ancestry"] = True
    value["learner_training_executed"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "BOUNDARY_FLAG_VIOLATION" in _codes(root)


def test_required_incident_disclosure_cannot_be_omitted(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/decision_record.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["local_tool_log_quarantine_path_redaction_incident_disclosed"] = False
    value["local_tool_log_quarantine_path_redaction_incident_summary"] = "OMITTED"
    path.write_text(json.dumps(value), encoding="utf-8")
    codes = _codes(root)
    assert "REQUIRED_DISCLOSURE_OR_HISTORY_MISSING" in codes
    assert "INCIDENT_SUMMARY_MISSING" in codes


def test_v1_and_v1_1_mutations_are_rejected(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    v1 = root / "docs/childlens_feasibility_v1/executive_decision_report.md"
    v1.write_text(v1.read_text(encoding="utf-8") + "\nmutation\n", encoding="utf-8")
    v1_1 = root / "docs/childlens_feasibility_v1_1/executive_decision_report.md"
    v1_1.write_text(v1_1.read_text(encoding="utf-8") + "\nmutation\n", encoding="utf-8")
    codes = _codes(root)
    assert "V1_IMMUTABILITY_MISMATCH" in codes
    assert "V1_1_IMMUTABILITY_MISMATCH" in codes


@pytest.mark.parametrize(
    "metric_name",
    (
        "audio_stream_present",
        "full_decode_success",
        "duration_consistency_evidenced",
        "corruption_absence_evidenced",
        "speech_presence_window_expectation_satisfied",
        "annotation_linkage_verified",
    ),
)
def test_human_ready_requires_each_all_pass_structural_metric(
    tmp_path: Path, metric_name: str
) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["structural_metrics"][metric_name] = {
        "status": "MIXED_SMALL_CELL_SUPPRESSED",
        "pass_count": None,
        "fail_count": None,
        "cell_suppressed": True,
    }
    count_key = {
        "audio_stream_present": "audio_stream_media_count",
        "full_decode_success": "decode_success_count",
        "speech_presence_window_expectation_satisfied": "speech_presence_window_media_count",
    }.get(metric_name)
    if count_key:
        value[count_key] = None
    value["automated_preparation_complete"] = False
    value["diagnostics_passed_for_human_workflow"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "READY_WITHOUT_AUTOMATED_PREPARATION" in _codes(root)


def _complete_go_human_evidence(root: Path) -> None:
    workflow_path = root / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow.update(
        {
            "candidate_unit_type": "HUMAN_CORRECTED_UTTERANCE",
            "populated_candidate_utterance_count": 320,
            "referential_assignment_eligible_count": 320,
            "referential_double_code_assigned_count": 64,
            "referential_double_code_assigned_fraction": 0.20,
            "referential_double_code_sample_frozen": True,
            "referential_inventory_pending_human_segmentation": False,
            "genuine_human_validation_complete": True,
            "independent_reliability_requirement_complete": True,
            "actual_language_established_by_qualified_humans": True,
            "language_matched_coders_verified": True,
            "frozen_reliability_thresholds_pass": True,
        }
    )
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    decision_path = root / "output/childlens_feasibility_v1_2/decision_record.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["all_essential_frozen_gates_pass"] = True
    decision["only_remaining_evidence_is_authorized_human_validation"] = False
    decision_path.write_text(json.dumps(decision), encoding="utf-8")


def test_complete_go_crosschecked_fixture_passes(tmp_path: Path) -> None:
    root = _fixture(tmp_path, "CHILDLENS_FEASIBILITY_GO")
    _complete_go_human_evidence(root)
    assert validator.validate(root) == []


def test_go_crosschecks_acquisition_and_diagnostics(tmp_path: Path) -> None:
    root = _fixture(tmp_path, "CHILDLENS_FEASIBILITY_GO")
    _complete_go_human_evidence(root)
    acquisition_path = root / "output/childlens_feasibility_v1_2/acquisition_receipt.json"
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    acquisition["acquisition_complete"] = False
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")
    diagnostics_path = root / "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["structural_metrics"]["audio_stream_present"] = {
        "status": "MIXED_SMALL_CELL_SUPPRESSED",
        "pass_count": None,
        "fail_count": None,
        "cell_suppressed": True,
    }
    diagnostics["audio_stream_media_count"] = None
    diagnostics["automated_preparation_complete"] = False
    diagnostics["diagnostics_passed_for_human_workflow"] = False
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    codes = _codes(root)
    assert "GO_WITHOUT_COMPLETE_ACQUISITION" in codes
    assert "GO_WITHOUT_COMPLETE_DIAGNOSTICS" in codes


def test_release_retention_and_live_snapshot_limitation_are_required(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/acquisition_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["retention_deadline"] = "2027-08-01"
    value["live_snapshot_limitation_disclosed"] = False
    value["immutable_public_snapshot_commit"] = "0" * 40
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "RELEASE_RETENTION_BINDING_INVALID" in _codes(root)


def test_namespace_and_free_floor_arithmetic_are_required(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/acquisition_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["projected_post_peak_free_bytes"] = validator.POST_PEAK_FREE_FLOOR_BYTES - 1
    value["namespace_bytes_after_acquisition"] = validator.NAMESPACE_CAP_BYTES + 1
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "STORAGE_CAP_OR_FREE_FLOOR_INVALID" in _codes(root)


def test_revise_permits_suppressed_small_failure_cell(tmp_path: Path) -> None:
    root = _fixture(tmp_path, "CHILDLENS_FEASIBILITY_REVISE")
    path = root / "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["structural_metrics"]["audio_stream_present"] = {
        "status": "MIXED_SMALL_CELL_SUPPRESSED",
        "pass_count": None,
        "fail_count": None,
        "cell_suppressed": True,
    }
    value["audio_stream_media_count"] = None
    value["automated_preparation_complete"] = False
    value["diagnostics_passed_for_human_workflow"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    assert validator.validate(root) == []


def test_small_nonzero_failure_cell_cannot_be_exported_exactly(tmp_path: Path) -> None:
    root = _fixture(tmp_path, "CHILDLENS_FEASIBILITY_REVISE")
    path = root / "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["structural_metrics"]["audio_stream_present"] = {
        "status": "MIXED_COUNTS_EXPORTED",
        "pass_count": 14,
        "fail_count": 1,
        "cell_suppressed": False,
    }
    value["audio_stream_media_count"] = 14
    value["automated_preparation_complete"] = False
    value["diagnostics_passed_for_human_workflow"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    assert "STRUCTURAL_METRIC_INVALID" in _codes(root)


def test_human_ready_accepts_honest_zero_actual_referential_units(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    assert validator.validate(root) == []
    path = root / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["referential_double_code_assigned_fraction"] = 0.20
    path.write_text(json.dumps(value), encoding="utf-8")
    codes = _codes(root)
    assert "PRE_REFERENTIAL_DOUBLE_CODE_STATE_INVALID" in codes
    assert "READY_WITHOUT_USABLE_WORKFLOW" in codes


def test_workflow_requires_qualified_language_controls_and_separate_labor(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["language_matched_humans_required"] = False
    value["estimated_second_independent_human_minutes"] = 0
    path.write_text(json.dumps(value), encoding="utf-8")
    codes = _codes(root)
    assert "WORKFLOW_BOUNDARY_VIOLATION" in codes
    assert "HUMAN_TIME_ESTIMATE_INVALID" in codes
    assert "READY_WITHOUT_USABLE_WORKFLOW" in codes


def test_actual_source_digests_crosscheck_acquisition_diagnostics_workflow(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    diagnostics_path = root / "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["source_native_transfer_receipt_sha256"] = "e" * 64
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    workflow_path = root / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["source_measurement_receipt_sha256"] = "f" * 64
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")
    codes = _codes(root)
    assert "DIAGNOSTIC_SOURCE_DIGEST_MISMATCH" in codes
    assert "WORKFLOW_SOURCE_DIGEST_MISMATCH" in codes


def test_privacy_scan_includes_git_tracked_v1_2_artifact_outside_namespaces(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    tracked = root / "notes/childlens_feasibility_v1_2_private.md"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("/Users/example/restricted/place", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", tracked.relative_to(root)], check=True)
    assert "ABSOLUTE_LOCAL_PATH_EXPORTED" in _codes(root)
