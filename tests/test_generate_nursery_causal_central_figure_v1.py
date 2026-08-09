from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_nursery_causal_central_figure_v1.py"
SPEC = importlib.util.spec_from_file_location("nursery_causal_central_figure_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def fixture_projection(*, all_pass: bool = True) -> dict:
    positive = 28 / 40
    negative = 4 / 40
    cell = {
        "synthetic_corpus_count": 40,
        "mean_effect": 0.05,
        "one_sided_95_lower": 0.01,
        "positive_corpus_fraction": positive,
        "exact_one_sided_sign_p": module.sign_p_value(28),
        "large_negative_fraction": negative,
        "passed": True,
    }
    contrasts = {
        family: {
            comparator: {endpoint: dict(cell) for endpoint in module.ENDPOINTS}
            for comparator in module.COMPARATORS
        }
        for family in module.FAMILIES
    }
    if not all_pass:
        target = contrasts[module.FAMILIES[-1]][module.COMPARATORS[-1]][module.ENDPOINTS[-1]]
        target["mean_effect"] = 0.02
        target["passed"] = False
    return {
        "schema_version": module.INPUT_SCHEMA,
        "status": "SCIENTIFIC_AGGREGATE_VALIDATED",
        "bindings": {
            "no_automatic_fallback_override_sha256": module.OVERRIDE_SHA256,
            "protocol_sha256": "1" * 64,
            "execution_contract_sha256": "2" * 64,
            "public_calibration_receipt_sha256": "3" * 64,
            "preoutcome_receipt_sha256": "4" * 64,
            "one_shot_authorization_sha256": "5" * 64,
            "aggregate_results_sha256": "6" * 64,
            "execution_receipt_sha256": "7" * 64,
        },
        "validity": {
            "gemma_path_passed": True,
            "automatic_fallback_used": False,
            "scientific_outcome_run": True,
            "fixture_only": False,
            "preoutcome_all_passed": True,
            "one_shot_complete": True,
            "complete_paired_bundles": True,
            "cue_free_evaluation": True,
            "simulator_oracle_only": True,
            "privacy_passed": True,
            "ancestry_passed": True,
            "restricted_payload_absent": True,
            "item_level_rows_absent": True,
        },
        "thresholds": dict(module.THRESHOLDS),
        "contrasts": contrasts,
        "falsifications": {
            "confidence_only_rule_passed": True,
            "side_only_leakage_passed": True,
            "event_selection_disconnection_passed": True,
        },
    }


def write_projection(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_positive_and_valid_null_are_determined_only_by_frozen_cells() -> None:
    positive, count, cells = module.validate_projection(fixture_projection(all_pass=True))
    assert positive == "POSITIVE_PROTOTYPE_EFFECT"
    assert count == 40 and len(cells) == 40
    null, count, cells = module.validate_projection(fixture_projection(all_pass=False))
    assert null == "VALID_NULL_OR_NONPROMOTION"
    assert count == 39 and len(cells) == 40


def test_gemma_failure_or_fallback_cannot_be_called_valid_null() -> None:
    value = fixture_projection(all_pass=False)
    value["validity"]["gemma_path_passed"] = False
    with pytest.raises(module.ReportingError, match="E_VALIDITY_GATE"):
        module.validate_projection(value)
    value = fixture_projection(all_pass=False)
    value["validity"]["automatic_fallback_used"] = True
    with pytest.raises(module.ReportingError, match="E_VALIDITY_GATE"):
        module.validate_projection(value)


def test_override_digest_and_frozen_thresholds_are_exact() -> None:
    value = fixture_projection()
    value["bindings"]["no_automatic_fallback_override_sha256"] = "0" * 64
    with pytest.raises(module.ReportingError, match="E_FALLBACK_OVERRIDE_BINDING"):
        module.validate_projection(value)
    value = fixture_projection()
    value["thresholds"]["minimum_mean_effect"] = 0.029
    with pytest.raises(module.ReportingError, match="E_THRESHOLD_BINDING"):
        module.validate_projection(value)


@pytest.mark.parametrize(
    "key",
    [
        "transcript_excerpt",
        "participant_uuid",
        "sourceFile",
        "frame_b64",
        "start_ms",
        "window_predictions",
        "restricted_manifest_sha256",
        "item_rows",
        "corpus_seed_rows",
    ],
)
def test_closed_schema_rejects_restricted_or_row_level_near_miss_keys(key: str) -> None:
    value = fixture_projection()
    value[key] = "sentinel"
    with pytest.raises(module.ReportingError, match="E_TOP_SCHEMA"):
        module.validate_projection(value)


def test_unknown_nested_key_and_inconsistent_self_report_are_rejected() -> None:
    value = fixture_projection()
    cell = value["contrasts"][module.FAMILIES[0]][module.COMPARATORS[0]][module.ENDPOINTS[0]]
    cell["raw_denominator"] = 40
    with pytest.raises(module.ReportingError, match="E_CELL_SCHEMA"):
        module.validate_projection(value)
    value = fixture_projection()
    value["contrasts"][module.FAMILIES[0]][module.COMPARATORS[0]][module.ENDPOINTS[0]]["passed"] = False
    with pytest.raises(module.ReportingError, match="E_CELL_PASS_MISMATCH"):
        module.validate_projection(value)


def test_nonfinite_off_grid_and_bad_sign_p_are_rejected() -> None:
    value = fixture_projection()
    value["contrasts"][module.FAMILIES[0]][module.COMPARATORS[0]][module.ENDPOINTS[0]]["mean_effect"] = float("nan")
    with pytest.raises(module.ReportingError, match="E_EFFECT_VALUE"):
        module.validate_projection(value)
    value = fixture_projection()
    value["contrasts"][module.FAMILIES[0]][module.COMPARATORS[0]][module.ENDPOINTS[0]]["positive_corpus_fraction"] = 0.61
    with pytest.raises(module.ReportingError, match="E_FRACTION_GRID"):
        module.validate_projection(value)
    value = fixture_projection()
    value["contrasts"][module.FAMILIES[0]][module.COMPARATORS[0]][module.ENDPOINTS[0]]["exact_one_sided_sign_p"] = 0.5
    with pytest.raises(module.ReportingError, match="E_SIGN_P"):
        module.validate_projection(value)


def test_generator_writes_fixed_svg_and_summary_without_echoing_arbitrary_input(tmp_path: Path) -> None:
    source = tmp_path / "aggregate.json"
    svg = tmp_path / "figure.svg"
    summary = tmp_path / "summary.json"
    write_projection(source, fixture_projection(all_pass=False))
    receipt = module.generate(source, svg, summary)
    assert receipt["decision"] == "VALID_NULL_OR_NONPROMOTION"
    assert receipt["passing_directional_cells"] == 39
    assert svg.read_text(encoding="utf-8").startswith("<svg")
    assert "VALID_NULL_OR_NONPROMOTION" in svg.read_text(encoding="utf-8")
    assert json.loads(summary.read_text(encoding="utf-8"))["total_directional_cells"] == 40


def test_invalid_input_creates_no_output_and_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "aggregate.json"
    svg = tmp_path / "figure.svg"
    summary = tmp_path / "summary.json"
    invalid = fixture_projection()
    invalid["item_rows"] = []
    write_projection(source, invalid)
    with pytest.raises(module.ReportingError):
        module.generate(source, svg, summary)
    assert not svg.exists() and not summary.exists()
    write_projection(source, fixture_projection())
    svg.write_text("sentinel", encoding="utf-8")
    with pytest.raises(module.ReportingError, match="E_OUTPUT_EXISTS"):
        module.generate(source, svg, summary)
    assert svg.read_text(encoding="utf-8") == "sentinel"
    assert not summary.exists()


def test_symlink_input_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    link = tmp_path / "link.json"
    write_projection(target, fixture_projection())
    link.symlink_to(target)
    with pytest.raises(module.ReportingError, match="E_INPUT_FILE"):
        module.generate(link, tmp_path / "figure.svg", tmp_path / "summary.json")


def test_source_has_no_network_discovery_or_restricted_path_primitives() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "requests",
        "urllib",
        "socket",
        "subprocess",
        ".glob(",
        ".rglob(",
        "os.walk",
        "quarantine",
        ".external",
    ):
        assert forbidden not in source
