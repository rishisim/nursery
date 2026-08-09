from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/synthesize_childlens_terminal_v1_2.py"
SPEC = importlib.util.spec_from_file_location("synthesize_childlens_terminal_v1_2", SCRIPT)
assert SPEC and SPEC.loader
synth = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = synth
SPEC.loader.exec_module(synth)

VALIDATOR_SCRIPT = REPO_ROOT / "scripts/validate_childlens_feasibility_v1_2.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_childlens_feasibility_v1_2_for_synthesis_test", VALIDATOR_SCRIPT
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = validator
VALIDATOR_SPEC.loader.exec_module(validator)


def _inputs(root: Path) -> tuple[dict, dict, dict]:
    output = root / "output/childlens_feasibility_v1_2"
    output.mkdir(parents=True)
    acquisition = {
        "schema_version": "childlens-v1.2-acquisition-receipt-v1",
        "selected_media_count": 15,
        "acquired_media_count": 15,
        "verified_against_frozen_selection_count": 15,
        "acquisition_complete": True,
        "raw_media_bytes_acquired": 8 * 1024**3,
        "raw_cap_bytes": 20 * 1024**3,
        "storage_admission_method": "CONSERVATIVE_DISPLAY_BOUND_WITH_HARD_COUNTER",
        "displayed_parsed_total_bytes": synth.FROZEN_DISPLAY_TOTAL_BYTES,
        "rounding_upper_bound_bytes": synth.FROZEN_ROUNDING_UPPER_BYTES,
        "transfer_overhead_bytes": synth.FROZEN_TRANSFER_OVERHEAD_BYTES,
        "pre_transfer_upper_bound_bytes": synth.FROZEN_CONSERVATIVE_ADMISSION_BYTES,
        "admission_ceiling_bytes": synth.ADMISSION_CEILING_BYTES,
        "upper_bound_margin_to_raw_cap_bytes": synth.RAW_CAP_BYTES
        - synth.FROZEN_CONSERVATIVE_ADMISSION_BYTES,
        "hard_abort_before_bytes": 20 * 1024**3 - 1,
        "predeclared_nonraw_reserve_bytes": synth.NONRAW_RESERVE_BYTES,
        "namespace_cap_bytes": synth.NAMESPACE_CAP_BYTES,
        "projected_post_peak_free_floor_bytes": synth.POST_PEAK_FREE_FLOOR_BYTES,
        "pre_transfer_namespace_bytes": 20 * 1024**2,
        "pre_transfer_free_bytes": 131_236_425_728,
        "projected_namespace_peak_bytes": 20 * 1024**2
        + synth.FROZEN_CONSERVATIVE_ADMISSION_BYTES
        + synth.NONRAW_RESERVE_BYTES,
        "projected_post_peak_free_bytes": 131_236_425_728
        - synth.FROZEN_CONSERVATIVE_ADMISSION_BYTES
        - synth.NONRAW_RESERVE_BYTES,
        "post_transfer_free_bytes": 110 * 1024**3,
        "namespace_bytes_after_acquisition": 9 * 1024**3,
        "frozen_selection_digest": synth.FROZEN_SELECTION_DIGEST,
        "immutable_public_snapshot_commit": synth.PUBLIC_SNAPSHOT_COMMIT,
        "pilot_source": "ACCESSIBLE_LIVE_VIEW_BOUND_TO_RESTRICTED_MANIFEST_DIGEST",
        "live_to_public_snapshot_byte_equivalence": "NOT_PROVEN",
        "live_snapshot_limitation_disclosed": True,
        "release_binding_verified": True,
        "restricted_manifest_digest_verified": True,
        "restricted_manifest_sha256": "b" * 64,
        "restricted_native_transfer_receipt_sha256": "a" * 64,
        "retention_deadline": synth.RETENTION_DEADLINE,
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
    all_pass = {
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
        "source_acquired_media_count": 15,
        "media_preparation_attempted_count": 15,
        "container_probe_attempted_count": 15,
        "container_probe_success_count": 15,
        "decode_attempted_count": 15,
        "decode_success_count": 15,
        "audio_stream_media_count": 15,
        "duration_consistent_count": 15,
        "corruption_free_count": 15,
        "annotation_linkage_verified_count": 15,
        "speech_expectation_concordant_count": 15,
        "speech_presence_window_media_count": 15,
        "cell_suppression_k": 5,
        "structural_metrics": {
            name: dict(all_pass) for name in synth.REQUIRED_STRUCTURAL_METRICS
        },
        "small_nonzero_failure_cells_exported": False,
        "measurement_source": "LOCAL_RESTRICTED_MEDIA_AUDIT_V1_2",
        "measurement_executed_on_acquired_bytes": True,
        "speech_window_actual_source": "OFFICIAL_ANNOTATION_WINDOWS_LINKED_TO_ACQUIRED_MEDIA",
        "annotation_linkage_actual_source": "RESTRICTED_CANONICAL_MANIFEST",
        "pilot_selection_sha256": synth.FROZEN_SELECTION_DIGEST,
        "source_native_transfer_receipt_sha256": "a" * 64,
        "restricted_measurement_receipt_sha256": "c" * 64,
        "official_speech_window_union_minutes": 31.25,
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
        "populated_candidate_utterance_count": 0,
        "populated_candidate_speech_window_count": 45,
        "populated_candidate_speech_window_minutes": 31.25,
        "candidate_unit_type": "OFFICIAL_SPEECH_WINDOW_NOT_UTTERANCE",
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
        "pilot_selection_sha256": synth.FROZEN_SELECTION_DIGEST,
        "source_native_transfer_receipt_sha256": "a" * 64,
        "source_measurement_receipt_sha256": "c" * 64,
        "restricted_workflow_receipt_sha256": "d" * 64,
    }
    immutability = {
        "schema_version": "childlens-v1.2-immutability-receipt-v1",
        "v1_artifact_count": synth.EXPECTED_V1_COUNT,
        "v1_set_digest": synth.EXPECTED_V1_DIGEST,
        "v1_1_artifact_count": synth.EXPECTED_V1_1_COUNT,
        "v1_1_set_digest": synth.EXPECTED_V1_1_DIGEST,
        "v1_preserved": True,
        "v1_1_preserved": True,
        "historical_decisions_overwritten": False,
    }
    baseline = {
        "schema_version": "childlens-immutable-baseline-v1.2.0",
        "validation_status": "PASS",
        "v1": {
            "artifact_count": synth.EXPECTED_V1_COUNT,
            "artifact_set_digest": synth.EXPECTED_V1_DIGEST,
            "historical_decision_preserved": True,
        },
        "v1_1": {
            "artifact_count": synth.EXPECTED_V1_1_COUNT,
            "artifact_set_digest": synth.EXPECTED_V1_1_DIGEST,
            "historical_decision_preserved": True,
        },
    }
    values = {
        "acquisition_receipt.json": acquisition,
        "automated_diagnostics_receipt.json": diagnostics,
        "human_validation_workflow_receipt.json": workflow,
        "immutability_receipt.json": immutability,
        "immutability_baseline_receipt.json": baseline,
    }
    for name, value in values.items():
        (output / name).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return acquisition, diagnostics, workflow


def _terminal_literals(root: Path) -> list[str]:
    found: list[str] = []
    for directory in (
        root / "docs/childlens_feasibility_v1_2",
        root / "output/childlens_feasibility_v1_2",
    ):
        for path in directory.glob("*"):
            if path.is_file():
                found.extend(synth.TERMINAL_RE.findall(path.read_text(encoding="utf-8")))
    return found


def _copy_historical_namespaces(root: Path) -> None:
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


def test_union_official_window_minutes_make_human_workflow_ready(tmp_path: Path) -> None:
    _inputs(tmp_path)
    decision = synth.synthesize(tmp_path)
    assert decision.terminal_state == synth.HUMAN_READY
    assert _terminal_literals(tmp_path) == [synth.HUMAN_READY]
    measurement = (tmp_path / synth.OUTPUT_RELATIVE_PATHS["measurement"]).read_text()
    assert "event intervals, not utterances" in measurement


def test_synthesized_human_ready_bundle_passes_repository_validator(tmp_path: Path) -> None:
    _copy_historical_namespaces(tmp_path)
    _inputs(tmp_path)
    docs = tmp_path / "docs/childlens_feasibility_v1_2"
    docs.mkdir(parents=True)
    for name in ("human_validation_workflow_handoff.md", "privacy_validation_handoff.md"):
        (docs / name).write_text("# Aggregate-only handoff\n\nNo restricted payload.\n")
    synth.synthesize(tmp_path)
    assert validator.validate(tmp_path) == []


def test_window_count_cannot_masquerade_as_utterance_count(tmp_path: Path) -> None:
    _, diagnostics, workflow = _inputs(tmp_path)
    diagnostics["official_speech_window_union_minutes"] = 29.99
    workflow["populated_candidate_speech_window_minutes"] = 29.99
    workflow["populated_candidate_utterance_count"] = 300
    output = tmp_path / "output/childlens_feasibility_v1_2"
    (output / "automated_diagnostics_receipt.json").write_text(json.dumps(diagnostics))
    (output / "human_validation_workflow_receipt.json").write_text(json.dumps(workflow))
    assert synth.synthesize(tmp_path).terminal_state == synth.REVISE


def test_genuine_independent_human_evidence_and_all_gates_are_required_for_go(
    tmp_path: Path,
) -> None:
    _, _, workflow = _inputs(tmp_path)
    workflow.update(
        {
            "genuine_human_validation_complete": True,
            "independent_reliability_requirement_complete": True,
            "frozen_reliability_thresholds_pass": True,
            "actual_language_established_by_qualified_humans": True,
            "language_matched_coders_verified": True,
            "all_frozen_feasibility_gates_pass": True,
            "candidate_unit_type": "HUMAN_CORRECTED_UTTERANCE",
            "populated_candidate_utterance_count": 320,
            "referential_assignment_eligible_count": 320,
            "referential_double_code_assigned_count": 64,
            "referential_double_code_assigned_fraction": 0.20,
            "referential_double_code_sample_frozen": True,
            "referential_inventory_pending_human_segmentation": False,
        }
    )
    path = tmp_path / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    path.write_text(json.dumps(workflow))
    assert synth.synthesize(tmp_path).terminal_state == synth.GO
    assert _terminal_literals(tmp_path) == [synth.GO]


def test_incomplete_acquisition_is_bounded_revise_not_stop(tmp_path: Path) -> None:
    acquisition, _, _ = _inputs(tmp_path)
    acquisition["acquired_media_count"] = 0
    acquisition["verified_against_frozen_selection_count"] = 0
    acquisition["acquisition_complete"] = False
    acquisition["raw_media_bytes_acquired"] = 0
    path = tmp_path / "output/childlens_feasibility_v1_2/acquisition_receipt.json"
    path.write_text(json.dumps(acquisition))
    decision = synth.synthesize(tmp_path)
    assert decision.terminal_state == synth.REVISE
    assert decision.fundamental_corpus_failure_established is False


def test_missing_diagnostic_metric_is_fail_closed_revise_not_exception(tmp_path: Path) -> None:
    _, diagnostics, _ = _inputs(tmp_path)
    diagnostics.pop("cell_suppression_k")
    path = tmp_path / "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json"
    path.write_text(json.dumps(diagnostics))
    assert synth.synthesize(tmp_path).terminal_state == synth.REVISE


def test_stop_requires_explicit_high_grade_fundamental_evidence(tmp_path: Path) -> None:
    _, _, workflow = _inputs(tmp_path)
    workflow.update(
        {
            "fundamental_corpus_failure_established": True,
            "fundamental_failure_basis_code": "GENUINE_HUMAN_VALIDATED_NO_NON_CHILD_INPUT",
            "fundamental_failure_evidence_level": "GENUINE_INDEPENDENT_HUMAN_VALIDATION",
            "genuine_human_validation_complete": True,
            "independent_reliability_requirement_complete": True,
            "actual_language_established_by_qualified_humans": True,
            "language_matched_coders_verified": True,
            "frozen_reliability_thresholds_pass": True,
            "all_frozen_feasibility_gates_pass": False,
            "candidate_unit_type": "HUMAN_CORRECTED_UTTERANCE",
            "populated_candidate_utterance_count": 320,
            "referential_assignment_eligible_count": 320,
            "referential_double_code_assigned_count": 64,
            "referential_double_code_assigned_fraction": 0.20,
            "referential_double_code_sample_frozen": True,
            "referential_inventory_pending_human_segmentation": False,
        }
    )
    path = tmp_path / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    path.write_text(json.dumps(workflow))
    assert synth.synthesize(tmp_path).terminal_state == synth.STOP


def test_unsubstantiated_fundamental_flag_does_not_stop(tmp_path: Path) -> None:
    _, _, workflow = _inputs(tmp_path)
    workflow["fundamental_corpus_failure_established"] = True
    path = tmp_path / "output/childlens_feasibility_v1_2/human_validation_workflow_receipt.json"
    path.write_text(json.dumps(workflow))
    assert synth.synthesize(tmp_path).terminal_state == synth.HUMAN_READY


def test_input_with_restricted_field_fails_without_outputs(tmp_path: Path) -> None:
    _, diagnostics, _ = _inputs(tmp_path)
    diagnostics["participant_id"] = "synthetic"
    path = tmp_path / "output/childlens_feasibility_v1_2/automated_diagnostics_receipt.json"
    path.write_text(json.dumps(diagnostics))
    with pytest.raises(synth.SynthesisError, match="E_INPUT_RESTRICTED_FIELD"):
        synth.synthesize(tmp_path)
    assert not (tmp_path / synth.OUTPUT_RELATIVE_PATHS["decision"]).exists()


def test_output_symlink_is_rejected_without_following_it(tmp_path: Path) -> None:
    _inputs(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("unchanged\n", encoding="utf-8")
    target = tmp_path / synth.OUTPUT_RELATIVE_PATHS["executive"]
    target.parent.mkdir(parents=True)
    target.symlink_to(outside)
    with pytest.raises(synth.SynthesisError, match="E_OUTPUT_CONFINEMENT"):
        synth.synthesize(tmp_path)
    assert outside.read_text(encoding="utf-8") == "unchanged\n"


def test_wrong_receipt_schema_fails_before_output(tmp_path: Path) -> None:
    acquisition, _, _ = _inputs(tmp_path)
    acquisition["schema_version"] = "unexpected"
    path = tmp_path / "output/childlens_feasibility_v1_2/acquisition_receipt.json"
    path.write_text(json.dumps(acquisition))
    with pytest.raises(synth.SynthesisError, match="E_INPUT_SCHEMA_VERSION"):
        synth.synthesize(tmp_path)


def test_deterministic_bytes_for_same_receipts(tmp_path: Path) -> None:
    _inputs(tmp_path)
    synth.synthesize(tmp_path)
    first = {
        name: (tmp_path / relative).read_bytes()
        for name, relative in synth.OUTPUT_RELATIVE_PATHS.items()
    }
    synth.synthesize(tmp_path)
    second = {
        name: (tmp_path / relative).read_bytes()
        for name, relative in synth.OUTPUT_RELATIVE_PATHS.items()
    }
    assert first == second


def test_atomic_failure_rolls_back_every_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _inputs(tmp_path)
    targets = [tmp_path / relative for relative in synth.OUTPUT_RELATIVE_PATHS.values()]
    for index, target in enumerate(targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"old-{index}\n", encoding="utf-8")
    originals = {target: target.read_bytes() for target in targets}
    real_replace = os.replace
    calls = 0

    def failing_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(synth, "_replace", failing_replace)
    with pytest.raises(synth.SynthesisError, match="E_ATOMIC_WRITE"):
        synth.synthesize(tmp_path)
    assert {target: target.read_bytes() for target in targets} == originals


def test_decision_is_installed_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _inputs(tmp_path)
    destinations: list[Path] = []
    real_replace = os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        destinations.append(destination)
        real_replace(source, destination)

    monkeypatch.setattr(synth, "_replace", recording_replace)
    synth.synthesize(tmp_path)
    assert destinations[-1] == tmp_path / synth.OUTPUT_RELATIVE_PATHS["decision"]


def test_no_argument_entrypoint_rejects_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "unexpected"])
    assert synth.main() == 2


def test_template_contains_no_terminal_literal() -> None:
    template = REPO_ROOT / "templates/childlens_terminal_synthesis_v1_2.json"
    value = json.loads(template.read_text(encoding="utf-8"))
    assert value["quarantine_read_permitted"] is False
    assert synth.TERMINAL_RE.search(template.read_text(encoding="utf-8")) is None
