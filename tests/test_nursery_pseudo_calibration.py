from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nursery_pseudo_calibration.py"
SPEC = importlib.util.spec_from_file_location("nursery_pseudo_calibration_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_german_normalization_is_frozen_and_translation_free() -> None:
    assert module._normalize("  ÜBER—Bälle! ") == "über bälle"


def test_boundary_match_uses_collar_and_maximum_cardinality() -> None:
    left = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.5, "end": 2.2, "text": "b"},
    ]
    right = [
        {"start": 0.2, "end": 1.2, "text": "a"},
        {"start": 1.7, "end": 2.4, "text": "b"},
        {"start": 9.0, "end": 10.0, "text": "x"},
    ]
    assert module._match_boundaries(left, right) == [(0, 0), (1, 1)]


def test_boundary_match_excludes_outside_500ms_collar() -> None:
    left = [{"start": 0.0, "end": 1.0, "text": "a"}]
    right = [{"start": 0.501, "end": 1.0, "text": "a"}]
    assert module._match_boundaries(left, right) == []


def test_edit_distance_is_symmetric_at_distance_level() -> None:
    assert module._edit_distance(list("haus"), list("maus")) == 1
    assert module._edit_distance(list("maus"), list("haus")) == 1


def test_rate_rounding_is_outward() -> None:
    assert module._round_range(0.31) == [0.3, 0.4]
    assert module._round_range(1.0) == [1.0, 1.0]


def test_cluster_interval_fails_closed_below_k() -> None:
    values = {str(index): [index % 2 == 0] for index in range(4)}
    assert module._cluster_interval(values) == {"status": "SUPPRESSED_K5"}


def test_cluster_interval_uses_complementary_suppression() -> None:
    values = {str(index): [index == 0] for index in range(10)}
    assert module._cluster_interval(values) == {"status": "SUPPRESSED_K5"}


def test_cluster_bootstrap_is_deterministic_and_outward_rounded() -> None:
    values = {str(index): [index % 2 == 0, index % 3 == 0] for index in range(15)}
    first = module._cluster_interval(values)
    second = module._cluster_interval(values)
    assert first == second
    assert first["status"] == "PUBLISHED"
    assert all(round(value * 10) == value * 10 for value in first["interval"])


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("visible_candidate", True),
        ("null_or_irrelevant", False),
        ("partial_or_clear_visibility", True),
        ("noun_object_support", True),
        ("verb_action_support", False),
    ],
)
def test_binary_projection(field: str, expected: bool) -> None:
    row = {
        "referential_status": "visible_candidate",
        "visibility_bin": "partial",
        "lexical_support": "noun_object",
    }
    assert module._binary(row, field) is expected


def test_baseline_crosswalk_never_exports_lexical_strings() -> None:
    result = module._baseline_common(
        {
            "referential_status": "VISIBLE_CANDIDATE",
            "noun_object_candidates": ["restricted nonce"],
            "verb_action_candidates": [],
            "machine_parse_succeeded": True,
        }
    )
    assert "restricted nonce" not in json.dumps(result)
    assert result["candidate_count_bin"] == "one"


def test_public_receipt_rejects_absolute_path_payload() -> None:
    receipt = json.loads((ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json").read_text())
    receipt["leak"] = "/Users/example/restricted"
    with pytest.raises(module.CalibrationError, match="E_PUBLIC_PRIVACY"):
        module._validate_public_receipt(receipt)


def test_live_receipt_is_revise_and_never_an_outcome() -> None:
    receipt = json.loads((ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json").read_text())
    module._validate_public_receipt(receipt)
    assert receipt["status"] == "CALIBRATION_REVISE"
    assert receipt["engineering_retry"]["attempts"] == 1
    assert receipt["scientific_outcome_run"] is False
    assert receipt["pseudo_labels_are_ground_truth"] is False


def test_retry_amendment_forbids_semantic_adaptation() -> None:
    amendment = json.loads((ROOT / "output/nursery_program_convergence_v1/qwen2_schema_retry_amendment.json").read_text())
    assert amendment["maximum_attempts"] == 1
    assert amendment["semantic_outputs_inspected_to_activate"] is False
    assert amendment["agreement_direction_inspected_to_activate"] is False
    assert amendment["thresholds_changed"] is False


def test_public_source_has_no_cloud_or_hosted_inference_call() -> None:
    source = SCRIPT.read_text(encoding="utf-8").casefold()
    for token in ("openai.com", "api.openai", "anthropic", "requests.post", "httpx.post"):
        assert token not in source


def test_public_export_contains_no_restricted_field_values() -> None:
    receipt = json.loads((ROOT / "output/nursery_program_convergence_v1/childlens_pseudo_calibration_receipt.json").read_text())
    encoded = json.dumps(receipt, sort_keys=True).casefold()
    for token in ("/users/", "media_relpath", "expected_media_sha256", "prediction_join_key", "transcript_text"):
        assert token not in encoded
