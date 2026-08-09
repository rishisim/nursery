from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


selection = _load(
    "test_childlens_extension_selection_v18",
    "scripts/freeze_childlens_30_video_extension_selection_v1_8.py",
)
calibration = _load(
    "test_childlens_extension_calibration_v18",
    "scripts/run_childlens_30_video_calibration_extension_v1_8.py",
)
acquisition = _load(
    "test_childlens_extension_acquisition_v18",
    "scripts/run_childlens_30_video_extension_acquisition_v1_8.py",
)
wrapper = _load(
    "test_childlens_extension_gemma_wrapper_v18",
    "scripts/nursery_gemma4_referential_worker_extension_v1_8.py",
)
qwen_worker = _load(
    "test_childlens_extension_qwen_worker_v18",
    "scripts/nursery_local_challenger_worker_extension_v1_8.py",
)


def _synthetic_inputs():
    release = {
        "doi": "synthetic",
        "doi_snapshot_inventory_receipt_sha256": None,
        "doi_snapshot_object_inventory_sha256": None,
        "keeper_library_id": "synthetic",
        "observation_date": "2026-07-22",
        "observation_receipt_sha256": "a" * 64,
        "observed_revision_commit": "b" * 40,
        "public_doi_snapshot_commit": "c" * 40,
        "session_definition": "synthetic",
        "source_role": "DOI_SNAPSHOT",
    }
    objects = []
    media = []
    annotations = []
    for index in range(45):
        media_key = f"media_{index:03d}"
        media_object = f"video_{index:03d}"
        annotation_object = f"annotation_{index:03d}"
        objects.extend(
            [
                {
                    "object_key": media_object,
                    "top_level_class": "VIDEO",
                    "available_for_selective_copy": True,
                    "size_bytes": (index + 1) * 1_000_000,
                    "source_locator": f"/ChildLens/videos/synthetic_{index:03d}.mp4",
                },
                {
                    "object_key": annotation_object,
                    "top_level_class": "ANNOTATION",
                    "local_sha256": f"{index + 1:064x}",
                    "source_locator": f"/private/synthetic_{index:03d}.json",
                },
            ]
        )
        media.append(
            {
                "media_key": media_key,
                "object_key": media_object,
                "participant_key": f"participant_{index:03d}",
                "session_key": f"session_{index:03d}",
                "coarse_activity_label": f"activity_{index % 5}",
                "speech_presence_bin": "PRESENT",
                "duration_milliseconds": 180_000 + index * 1_000,
                "location_label": f"location_{index % 3}",
            }
        )
        annotations.append(
            {
                "annotation_key": f"ann_{index:03d}",
                "linked_media_key": media_key,
                "object_key": annotation_object,
                "representation_kind": "JSON",
            }
        )
    full = {
        "schema_version": selection.FULL_SCHEMA,
        "release": release,
        "objects": objects,
        "media": media,
        "annotations": annotations,
        "attestations": {},
    }
    original = {
        "schema_version": selection.MEASUREMENT_SCHEMA,
        "pilot_selection_sha256": "d" * 64,
        "items": [
            {
                "blinded_item_key": f"media_{index:03d}",
                "media_relpath": f"raw/{index:064x}.bin",
                "expected_size_bytes": 1,
                "expected_media_sha256": f"{index:064x}",
                "reference_duration_seconds": 180,
                "reference_duration_basis": "synthetic",
                "speech_presence_expectation": "PRESENT",
                "speech_windows": [{"start_seconds": 0, "end_seconds": 180}],
            }
            for index in range(15)
        ],
    }
    return full, original


def _annotation_loader(row):
    return [(0.0, 180.0)], 0, row["local_sha256"]


def test_extension_selection_is_deterministic_distinct_and_content_blind():
    full, original = _synthetic_inputs()
    plan_a, receipt_a = selection.build_selection(
        full, original, annotation_loader=_annotation_loader
    )
    full_reweighted = json.loads(json.dumps(full))
    for row in full_reweighted["objects"]:
        if row["top_level_class"] == "VIDEO":
            row["size_bytes"] = 999_000_000 - row["size_bytes"]
    plan_b, receipt_b = selection.build_selection(
        full_reweighted, original, annotation_loader=_annotation_loader
    )
    assert plan_a["additional_selection_sha256"] == plan_b["additional_selection_sha256"]
    assert receipt_a["additional_item_count"] == 15
    assert receipt_a["combined_item_count"] == 30
    assert receipt_a["all_combined_participants_distinct"] is True
    assert receipt_a["selection_used_model_output"] is False
    assert receipt_a["selection_opened_media"] is False
    assert receipt_a["media_size_used_for_ranking"] is False
    assert receipt_a["further_expansion_allowed"] is False
    assert plan_a["additional_total_speech_seconds"] == 900
    assert plan_a["additional_candidate_window_count"] == 135
    assert all(
        sum(
            span["end_ms"] - span["start_ms"]
            for span in item["sample_segments_clip_ms"]
        )
        == 60_000
        and len(item["candidate_windows_clip_ms"]) == 9
        for item in plan_a["items"]
    )
    selection._public_guard(receipt_a)


def test_protocol_freezes_one_extension_and_unchanged_gates():
    protocol = json.loads(
        (
            ROOT
            / "docs/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
            / "frozen_extension_protocol_v1_8.json"
        ).read_text()
    )
    assert protocol["sample"]["original_items_retained_unchanged"] == 15
    assert protocol["sample"]["additional_items"] == 15
    assert protocol["sample"]["combined_items"] == 30
    assert protocol["sample"]["no_further_expansion"] is True
    assert protocol["instruments"]["third_visual_model_allowed"] is False
    assert protocol["instruments"]["semantic_prompt_tuning_allowed"] is False
    assert protocol["unchanged_calibration_gates"]["minimum_export_cell_items"] == 5
    assert (
        protocol["unchanged_calibration_gates"]["schema_validity_minimum"] == 0.95
    )
    assert protocol["terminal_rules"]["causal_endpoint"].startswith("Closed until")


def _passing_ranges():
    published = {
        "qwen3asr_plus_qwen3vl_path": {
            "status": "PUBLISHED",
            "interval": [0.1, 0.3],
        },
        "qwen3asr_plus_gemma4_path": {
            "status": "PUBLISHED",
            "interval": [0.1, 0.4],
        },
        "model_intersection": {"status": "PUBLISHED", "interval": [0.0, 0.2]},
        "model_union": {"status": "PUBLISHED", "interval": [0.2, 0.5]},
        "conservative_envelope": {
            "status": "PUBLISHED",
            "interval": [0.0, 0.5],
        },
    }
    ranges = {field: dict(published) for field in calibration.base.FIELDS}
    for category, bins in calibration.base.CATEGORY_FIELDS.items():
        ranges[category] = {
            value: (
                dict(published)
                if index == 0
                else {"conservative_envelope": {"status": "SUPPRESSED_K5"}}
            )
            for index, value in enumerate(bins)
        }
    return ranges


def test_combined_gate_fails_closed_on_suppressed_required_envelope():
    ranges = _passing_ranges()
    visual = {"visual_item_coverage_interval": [1.0, 1.0]}
    speech = {"challenger_nonempty_item_coverage_interval": [0.9, 1.0]}
    abstention = {
        dimension: {
            "qwen3asr_plus_qwen3vl_path": 0.1,
            "qwen3asr_plus_gemma4_path": 0.2,
        }
        for dimension in [*calibration.base.FIELDS, *calibration.base.CATEGORY_FIELDS]
    }
    schema = {
        "qwen3asr_plus_qwen3vl_path": 0.99,
        "qwen3asr_plus_gemma4_path": 0.98,
    }
    assert calibration._gate(ranges, visual, speech, abstention, schema) == []
    ranges["null_or_irrelevant"]["conservative_envelope"] = {
        "status": "SUPPRESSED_K5"
    }
    assert "ENVELOPE_NULL_OR_IRRELEVANT" in calibration._gate(
        ranges, visual, speech, abstention, schema
    )


def test_gemma_extension_changes_only_predeclared_count():
    assert wrapper.worker.WINDOW_COUNT == 135
    assert wrapper.worker.legacy.WINDOW_COUNT == 135
    assert wrapper.worker.FRAME_OFFSETS == (-5.0, -2.5, 0.0, 2.5, 5.0)
    assert (
        wrapper.worker.prompt_and_schema_digest()
        == calibration.gemma_worker.prompt_and_schema_digest()
    )
    assert wrapper.worker.EXACT_SCHEMA == calibration.gemma_worker.EXACT_SCHEMA


def test_public_receipts_reject_restricted_payload_shapes():
    for guard in (
        selection._public_guard,
        acquisition._public_guard,
        calibration._public_guard,
    ):
        try:
            guard({"status": "bad", "participant_key": "restricted"})
        except Exception:
            pass
        else:
            raise AssertionError("restricted key was not rejected")


def test_v184_error_envelope_is_fail_closed_and_payload_free():
    assert (
        calibration._restricted_error_code(
            {
                "schema_version": "nursery-childlens-restricted-error-envelope-v1.8.4",
                "status": "ERROR",
                "failure_code": "E_QWEN_ASR",
            }
        )
        == "E_QWEN_ASR"
    )
    assert (
        calibration._restricted_error_code(
            {
                "schema_version": "nursery-childlens-restricted-error-envelope-v1.8.4",
                "status": "ERROR",
                "failure_code": "E_QWEN_ASR",
                "transcript": "forbidden",
            }
        )
        is None
    )
    assert (
        calibration._restricted_error_code(
            {
                "schema_version": "nursery-childlens-restricted-error-envelope-v1.8.4",
                "status": "ERROR",
                "failure_code": "../../escape",
            }
        )
        is None
    )


def test_v184_amendment_changes_diagnostics_only():
    amendment = json.loads(
        (
            ROOT
            / "docs/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
            / "restricted_error_envelope_amendment_v1_8_4.json"
        ).read_text()
    )
    assert amendment["status"] == "FROZEN_BEFORE_RETRY"
    assert amendment["scope"] == "CONTENT_INDEPENDENT_DIAGNOSTIC_TRANSPORT_ONLY"
    assert amendment["restricted_gemma_inference_opened_before_amendment"] is False
    assert amendment["scientific_endpoint_opened_before_amendment"] is False
    assert (
        ROOT
        / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
        / "combined_calibration_failure_v1_8_4.json"
    ).is_file()


def test_v185_candidate_windows_allow_overlap_but_not_reordering():
    assert qwen_worker._validate_candidate_windows(
        [
            {"start_seconds": 0.0, "end_seconds": 10.0},
            {"start_seconds": 5.0, "end_seconds": 15.0},
        ]
    ) == [(0.0, 10.0), (5.0, 15.0)]
    for invalid in (
        [
            {"start_seconds": 5.0, "end_seconds": 15.0},
            {"start_seconds": 4.0, "end_seconds": 14.0},
        ],
        [{"start_seconds": 5.0, "end_seconds": 5.0}],
        [{"start_seconds": -1.0, "end_seconds": 4.0}],
    ):
        try:
            qwen_worker._validate_candidate_windows(invalid)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid candidate windows were accepted")


def test_v185_speech_intervals_remain_nonoverlapping():
    try:
        qwen_worker.worker._validate_intervals(
            [
                {"start_seconds": 0.0, "end_seconds": 10.0},
                {"start_seconds": 5.0, "end_seconds": 15.0},
            ]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("overlapping speech extraction intervals were accepted")
    assert (
        ROOT
        / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
        / "combined_calibration_failure_v1_8_5.json"
    ).is_file()


def test_v186_gemma_count_binding_is_exact_and_additive():
    amendment = json.loads(
        (
            ROOT
            / "docs/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
            / "gemma_extension_count_binding_amendment_v1_8_6.json"
        ).read_text()
    )
    assert amendment["status"] == "FROZEN_BEFORE_RETRY"
    assert amendment["restricted_gemma_checkpoint_count_before_amendment"] == 0
    assert amendment["scientific_endpoint_opened_before_amendment"] is False
    assert (
        ROOT
        / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
        / "combined_calibration_failure_v1_8_6.json"
    ).is_file()


def test_v187_gemma_output_schema_binding_is_exact():
    amendment = json.loads(
        (
            ROOT
            / "docs/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
            / "gemma_output_schema_binding_amendment_v1_8_7.json"
        ).read_text()
    )
    assert amendment["status"] == "FROZEN_BEFORE_ASSEMBLY_RETRY"
    assert amendment["completed_gemma_checkpoint_count_before_amendment"] == 135
    assert amendment["gemma_inference_rerun_allowed"] is False
    assert (
        calibration.gemma_worker.legacy.OUTPUT_SCHEMA
        == calibration.gemma_base.GEMMA_OUTPUT_SCHEMA
    )
    assert (
        ROOT
        / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
        / "combined_calibration_failure_v1_8_7.json"
    ).is_file()


def test_v188_isolated_author_alias_is_exact_and_fail_closed():
    name = "childlens_author_audit_v1_3"
    prior = sys.modules.pop(name, None)
    first = object()
    second = object()
    try:
        calibration._bind_author_import_alias(first)
        assert sys.modules[name] is first
        try:
            calibration._bind_author_import_alias(second)
        except calibration.CombinedCalibrationError as exc:
            assert exc.code == "E_AUTHOR_ALIAS"
        else:
            raise AssertionError("conflicting author alias was accepted")
    finally:
        sys.modules.pop(name, None)
        if prior is not None:
            sys.modules[name] = prior
    assert (
        ROOT
        / "output/nursery_program_convergence_v1/childlens_30_video_extension_v1_8"
        / "combined_calibration_failure_v1_8_8.json"
    ).is_file()


def test_v189_initializer_dependency_alias_is_hash_pinned():
    assert calibration._sha256_file(calibration.AUTHOR) == calibration.AUTHOR_SHA256
    assert (
        calibration._sha256_file(calibration.HUMAN_V12)
        == calibration.HUMAN_V12_SHA256
    )
    name = "childlens_human_validation_v1_2"
    prior = sys.modules.pop(name, None)
    module = object()
    try:
        calibration._bind_import_alias(name, module)
        assert sys.modules[name] is module
    finally:
        sys.modules.pop(name, None)
        if prior is not None:
            sys.modules[name] = prior
    assert calibration.PUBLIC_FAILURE.name == "combined_calibration_failure_v1_8_9.json"


def test_v1810_historical_qwen_worker_is_restored():
    historical = ROOT / "scripts/nursery_local_challenger_worker.py"
    assert (
        calibration._sha256_file(historical)
        == "e785ea475aee62fd72ab9bc064f3c9c3dcbc189b4361e996963edf9f2ab14f88"
    )
    assert calibration.QWEN_WORKER.name == "nursery_local_challenger_worker_extension_v1_8.py"


def test_acquisition_is_sequential_keychain_only_and_fail_closed():
    source = (
        ROOT / "scripts/run_childlens_30_video_extension_acquisition_v1_8.py"
    ).read_text()
    assert "MacOSKeychain()" in source
    assert "maximum_concurrent_full_source_objects" in source
    assert acquisition.RAW_CAP_BYTES == 20 * 1024**3
    assert acquisition.FREE_FLOOR_BYTES == 50 * 1024**3
    assert acquisition.ITEM_COUNT == 15
    assert "print(token" not in source
    assert "Authorization" not in source
    assert "response.read(CHUNK_BYTES)" in source
    assert 'raw_path.unlink()' in source
