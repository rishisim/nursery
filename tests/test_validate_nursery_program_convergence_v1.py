from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/validate_nursery_program_convergence_v1.py"
SPEC = importlib.util.spec_from_file_location("validate_convergence", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.fixture()
def bundle() -> list[dict]:
    return [
        _load("output/nursery_program_convergence_v1/decision_record.json"),
        _load(validator.ACTIVE_PROTOCOL),
        _load(validator.SUPERSEDED_PROTOCOL),
        _load(validator.PSEUDO_PROTOCOL),
        _load(validator.IMMUTABILITY_RECEIPT),
    ]


def _codes(documents: list[dict]) -> set[str]:
    return {issue.code for issue in validator.validate_public_artifacts(*documents)}


def _safe_public_receipt() -> dict:
    return {
        "schema_version": "nursery-childlens-pseudo-calibration-receipt-v1",
        "public_export": {
            "raw_counts": False,
            "minimum_cluster_k": 5,
            "complementary_suppression": True,
            "paths": False,
            "identifiers": False,
            "filenames": False,
            "exact_timestamps_or_intervals": False,
            "transcript_or_lexical_content": False,
            "frames_or_audio": False,
            "item_level_predictions": False,
            "confidence_or_raw_model_payload": False,
            "free_form_errors": False,
        },
        "calibration_dimensions": {
            "visibility": {
                "instrument_a_interval": [0.2, 0.7],
                "instrument_b_interval": [0.3, 0.8],
                "intersection_interval": [0.2, 0.6],
                "union_interval": [0.3, 0.9],
                "envelope": [0.2, 0.9],
            }
        },
        "pseudo_labels_are_ground_truth": False,
        "human_evidence_available": False,
        "simulator_oracle_only_evaluation_truth": True,
    }


def test_live_public_bundle_passes_and_never_discovers_quarantine(bundle: list[dict]) -> None:
    assert _codes(bundle) == set()
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "quarantine" in source  # module explicitly documents the prohibition
    assert "rglob(" not in source
    assert "os.walk" not in source


def test_exact_decision_and_active_protocol_are_fail_closed(bundle: list[dict]) -> None:
    bad = copy.deepcopy(bundle)
    bad[0]["decision"] = "CLOSE_ENOUGH"
    assert "DECISION_LITERAL" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[0]["active_protocol"] = validator.SUPERSEDED_PROTOCOL
    assert "ACTIVE_PROTOCOL" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[1]["decision"] = "DIFFERENT_DECISION"
    assert "ACTIVE_DECISION_MISMATCH" in _codes(bad)


def test_superseded_protocol_cannot_be_reactivated(bundle: list[dict]) -> None:
    bad = copy.deepcopy(bundle)
    bad[2]["status"] = "FROZEN_FOR_IMPLEMENTATION"
    bad[2].pop("superseded_by", None)
    assert "SUPERSESSION_NOT_FAIL_CLOSED" in _codes(bad)


@pytest.mark.parametrize("field", ["AEA_empirical_ancestry", "BabyView_empirical_ancestry", "cross_corpus_pooling"])
def test_decision_rejects_other_empirical_ancestry(bundle: list[dict], field: str) -> None:
    bad = copy.deepcopy(bundle)
    bad[0][field] = True
    assert "DECISION_ANCESTRY" in _codes(bad)


def test_active_protocol_rejects_aea_babyview_and_pooling(bundle: list[dict]) -> None:
    for field in ("AEA_empirical_ancestry", "BabyView_empirical_ancestry", "pooling"):
        bad = copy.deepcopy(bundle)
        bad[1]["ancestry"][field] = True
        assert "FORBIDDEN_EMPIRICAL_ANCESTRY" in _codes(bad)


def test_only_aggregate_childlens_calibration_is_accepted(bundle: list[dict]) -> None:
    bad = copy.deepcopy(bundle)
    bad[1]["ancestry"]["allowed_childlens_input"] = "ChildLens pseudo labels"
    assert "CHILDLENS_INPUT_TOO_BROAD" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[3]["simulator_binding"]["empirical_free_minimal_realism_consumer"] = True
    assert "PSEUDO_CONSUMER_BINDING" in _codes(bad)


def test_each_task_family_requires_two_distinct_model_families(bundle: list[dict]) -> None:
    bad = copy.deepcopy(bundle)
    rows = bad[3]["instrument_paths"]["referential_and_visual"]
    rows[1]["family_id"] = rows[0]["family_id"]
    assert "PATHS_NOT_DISTINCT" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[3]["instrument_paths"]["speech_timing_and_text"][1]["required"] = False
    assert "PATH_COUNT_BELOW_TWO" in _codes(bad)


def test_hosted_or_network_path_is_rejected(bundle: list[dict]) -> None:
    bad = copy.deepcopy(bundle)
    bad[3]["instrument_paths"]["hosted_models"] = "allowed"
    assert "HOSTED_OR_NETWORK_PATH" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[3]["privacy_and_execution"]["no_external_api"] = False
    assert "PRIVACY_FLAG" in _codes(bad)


def test_k5_complementary_suppression() -> None:
    assert validator.categorical_k5_disposition({"a": 4, "b": 11}) == validator.SUPPRESSED_K5
    assert validator.categorical_k5_disposition({"a": 11, "b": 4}) == validator.SUPPRESSED_K5
    assert validator.categorical_k5_disposition({"a": 5, "b": 10}) == "PUBLISH_OUTWARD_ROUNDED_INTERVALS_ONLY"
    assert validator.categorical_k5_disposition({"a": -1, "b": 16}) == validator.SUPPRESSED_K5
    assert validator.categorical_k5_disposition({}, minimum_k=5) == validator.SUPPRESSED_K5


def test_k_threshold_and_agreement_are_frozen(bundle: list[dict]) -> None:
    bad = copy.deepcopy(bundle)
    bad[3]["predeclared_gates"]["minimum_export_cell_items"] = 4
    assert "K_SUPPRESSION" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[3]["predeclared_gates"]["agreement_is_not_a_pass_gate"] = False
    assert "AGREEMENT_GATE" in _codes(bad)


@pytest.mark.parametrize(
    ("interval", "step", "expected"),
    [
        ((0.21, 0.79), 0.1, (0.2, 0.8)),
        ((0.2, 0.8), 0.1, (0.2, 0.8)),
        ((-0.01, 1.01), 0.1, (0.0, 1.0)),
        ((0.51, 1.49), 0.5, (0.5, 1.0)),
    ],
)
def test_outward_rounding_never_rounds_inward(interval: tuple[float, float], step: float, expected: tuple[float, float]) -> None:
    actual = validator.round_interval_outward(*interval, step)
    assert actual == expected
    assert actual[0] <= max(interval[0], 0.0)
    assert actual[1] >= min(interval[1], 1.0)


@pytest.mark.parametrize(
    "args",
    [
        (0.8, 0.2, 0.1),
        (0.2, 0.8, 0.0),
        (float("nan"), 0.8, 0.1),
        (0.2, float("inf"), 0.1),
    ],
)
def test_outward_rounding_rejects_invalid_input(args: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        validator.round_interval_outward(*args)


def test_rounding_and_envelope_rules_cannot_be_removed(bundle: list[dict]) -> None:
    bad = copy.deepcopy(bundle)
    bad[3]["uncertainty"]["grid_rounding"].pop("rate_dimensions")
    assert "OUTWARD_ROUNDING_UNFROZEN" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[3]["uncertainty"]["envelope"] = "average all estimates"
    assert "ENVELOPE_RULE" in _codes(bad)


def test_public_aggregate_receipt_accepts_only_safe_schema() -> None:
    receipt = _safe_public_receipt()
    assert validator.validate_public_calibration_receipt(receipt) == []

    for key, value in (
        ("transcript_text", "restricted words"),
        ("source_filename", "source.mp4"),
        ("participant_id", "person"),
        ("item_rows", [{"item_id": "x"}]),
        ("exact_interval", [1.0, 2.0]),
        ("confidence_score", 0.9),
        ("frame_content", "base64"),
        ("human_validation", "complete"),
        ("ground_truth", "model consensus"),
    ):
        bad = copy.deepcopy(receipt)
        bad[key] = value
        assert any(issue.code == "RESTRICTED_PUBLIC_FIELD" for issue in validator.validate_public_calibration_receipt(bad))


def test_policy_booleans_cannot_hide_payload() -> None:
    bad = _safe_public_receipt()
    bad["public_export"]["transcript_or_lexical_content"] = False
    bad["nested"] = {"transcript_text": "still restricted"}
    assert any(issue.code == "RESTRICTED_PUBLIC_FIELD" for issue in validator.validate_public_calibration_receipt(bad))


def test_public_receipt_requires_k5_and_pseudo_only_semantics() -> None:
    bad = _safe_public_receipt()
    bad["public_export"]["minimum_cluster_k"] = 3
    assert any(issue.code == "PUBLIC_K_POLICY" for issue in validator.validate_public_calibration_receipt(bad))

    bad = _safe_public_receipt()
    bad["pseudo_labels_are_ground_truth"] = True
    assert any(issue.code == "PUBLIC_SEMANTICS" for issue in validator.validate_public_calibration_receipt(bad))


def test_no_causal_outcome_is_authorized_or_opened(bundle: list[dict]) -> None:
    for field in ("new_scientific_outcome_run", "scientific_outcome_authorized", "scientific_endpoint_opened"):
        bad = copy.deepcopy(bundle)
        bad[0][field] = True
        assert "OUTCOME_AUTHORIZED" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[1]["execution"]["outcome_authorized"] = True
    assert "OUTCOME_AUTHORIZED" in _codes(bad)


def test_lexical_stop_cannot_be_reopened(bundle: list[dict]) -> None:
    bad = copy.deepcopy(bundle)
    bad[0]["childlens_disposition"]["lexical_reopening_allowed"] = True
    assert "LEXICAL_STOP_NOT_PRESERVED" in _codes(bad)


def test_v1_through_v1_3_1_immutability_references_exist(bundle: list[dict]) -> None:
    assert validator.validate_repository(ROOT) == []
    assert set(validator.HISTORICAL_REFERENCES) == {"v1", "v1_1", "v1_2", "v1_3", "v1_3_1"}
    for path in validator.HISTORICAL_REFERENCES.values():
        assert (ROOT / path).is_file()

    bad = copy.deepcopy(bundle)
    bad[4]["artifact_sets"]["v1_2"]["preserved"] = False
    assert "HISTORY_IMMUTABILITY" in _codes(bad)

    bad = copy.deepcopy(bundle)
    bad[4]["prior_decisions_overwritten"] = True
    assert "HISTORY_OVERWRITE" in _codes(bad)


def test_validator_has_no_aea_babyview_or_quarantine_input_paths() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden_literals = (
        "output/aea_",
        "data/aea",
        ".external/",
        "quarantine_root",
        "restricted_manifest",
    )
    for literal in forbidden_literals:
        assert literal not in source
