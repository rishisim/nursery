from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGER_PATH = ROOT / "scripts/package_nursery_causal_outcome_reporting_v1.py"
MEMO_PATH = ROOT / "scripts/synthesize_nursery_causal_outcome_memo_v1.py"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


packager = load_module("test_nursery_reporting_packager_v1", PACKAGER_PATH)
memo = load_module("test_nursery_reporting_memo_v1", MEMO_PATH)
figure = packager.figure


def synthetic_calibration() -> dict:
    source = json.loads((ROOT / "tests/fixtures/calibration_conditioned_symbolic_ranges_v1.json").read_text())

    def expand(value):
        if isinstance(value, dict):
            if set(value) == {"synthetic_fixture_nominal"}:
                return {
                    family: copy.deepcopy(value["synthetic_fixture_nominal"])
                    for family in packager.runner.REPLACEMENT_RANGE_FAMILIES
                }
            return {key: expand(child) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child) for child in value]
        return value

    return {
        "schema_version": packager.runner.PROTOTYPE_CALIBRATION_SCHEMA,
        "status": "CALIBRATION_PASS",
        "decision": "CALIBRATION_PASS_INTERNAL_PROTOTYPE",
        "caf_transport_amendment_sha256": packager.runner.CAF_TRANSPORT_AMENDMENT_SHA256,
        "no_automatic_fallback_override_sha256": packager.OVERRIDE_SHA256,
        "automatic_fallback_allowed": False,
        "scientific_endpoint_if_gemma_fails": False,
        "instrument_paths": list(packager.runner.REPLACEMENT_RANGE_FAMILIES[:2]),
        "calibration_ranges": expand(source["calibration_ranges"]),
        "public_export": {
            "minimum_cluster_k": 5,
            "complementary_suppression": True,
            **{field: False for field in packager.runner.PUBLIC_FALSE_FIELDS},
        },
        "security": {
            "restricted_inference_network_disabled": True,
            "external_api_used": False,
            "hosted_or_cloud_content_path": False,
            "quarantine_only_pseudo_payloads": True,
        },
        "ancestry": {
            "AEA_empirical_ancestry": False,
            "BabyView_empirical_ancestry": False,
            "cross_corpus_pooling": False,
            "learner_ancestry_from_instruments": False,
        },
        "pseudo_labels_are_ground_truth": False,
        "human_evidence_available": False,
        "simulator_oracle_only_evaluation_truth": True,
        "scientific_outcome_run": False,
    }


def synthetic_preoutcome(calibration: dict) -> dict:
    return {
        "schema_version": packager.runner.PREOUTCOME_SCHEMA,
        "status": "PASS",
        "fixture_only": False,
        "scientific_endpoints_opened": False,
        "calibration_binding_sha256": packager.runner._digest(calibration),
        "execution_contract_sha256": packager.CONTRACT_CANONICAL_SHA256,
        "protocol_erratum_sha256": packager.ERRATUM_CANONICAL_SHA256,
        "runner_sha256": packager.RUNNER_FILE_SHA256,
        "gates": {key: True for key in packager.runner.PREOUTCOME_GATES},
        "all_gates_passed": True,
    }


def synthetic_aggregate(*, all_pass: bool = True) -> dict:
    cell = {
        "corpus_count": 40,
        "mean_effect": 0.05,
        "one_sided_95_lower": 0.01,
        "positive_corpus_fraction": 28 / 40,
        "exact_one_sided_sign_p": figure.sign_p_value(28),
        "large_negative_fraction": 4 / 40,
        "passed": True,
    }
    contrasts = {
        family: {
            comparator: {endpoint: dict(cell) for endpoint in figure.ENDPOINTS}
            for comparator in figure.COMPARATORS
        }
        for family in figure.FAMILIES
    }
    if not all_pass:
        target = contrasts[figure.FAMILIES[-1]][figure.COMPARATORS[-1]][figure.ENDPOINTS[-1]]
        target["mean_effect"] = 0.02
        target["passed"] = False
    return {
        "schema_version": packager.runner.MERGE_SCHEMA,
        "fixture_only": False,
        "merge_order": list(range(740001, 740080, 2)),
        "bundle_sha256s": [hashlib.sha256(f"bundle-{index}".encode()).hexdigest() for index in range(40)],
        "contrasts": contrasts,
        "confidence_only_falsification": {
            "confidence_endpoint_present": False,
            "rule": "BOTH_NOUN_AND_VERB_WRONG_OR_TIED_TOP_MAPPINGS_MUST_BECOME_UNIQUELY_CORRECT",
            "top_mapping_endpoints_only": list(figure.ENDPOINTS),
        },
        "side_only_leakage_falsification": {
            "maximum_above_chance": 0.02,
            "required_pre_outcome": True,
        },
        "all_range_family_endpoint_comparator_gates_passed": all_pass,
        "terminal_interpretation": "PROTOTYPE_PASS" if all_pass else "PROTOTYPE_STOP",
    }


def synthetic_execution(aggregate: dict) -> dict:
    return {
        "schema_version": packager.runner.VERSION,
        "status": "SCIENTIFIC_ONE_SHOT_COMPLETE",
        "authorization_id": "synthetic-test-authorization-0001",
        "bundle_count": 40,
        "deterministic_merge_sha256": packager.digest(aggregate),
        "scientific_outcome_run": True,
        "cue_free_evaluation": True,
        "paired_complete_bundles": True,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def source_files(tmp_path: Path, *, all_pass: bool = True):
    calibration = synthetic_calibration()
    preoutcome = synthetic_preoutcome(calibration)
    aggregate = synthetic_aggregate(all_pass=all_pass)
    execution = synthetic_execution(aggregate)
    paths = [tmp_path / name for name in ("calibration.json", "preoutcome.json", "aggregate_results.json", "execution_receipt.json")]
    for path, value in zip(paths, (calibration, preoutcome, aggregate, execution), strict=True):
        write_json(path, value)
    return paths, (calibration, preoutcome, aggregate, execution)


def run_package(tmp_path: Path, *, all_pass: bool = True):
    paths, values = source_files(tmp_path, all_pass=all_pass)
    projection_path = tmp_path / "projection.json"
    projection = packager.package(
        calibration_path=paths[0],
        preoutcome_path=paths[1],
        authorization_seal_sha256="a" * 64,
        aggregate_path=paths[2],
        execution_path=paths[3],
        output_path=projection_path,
        enforce_repo_paths=False,
    )
    return projection_path, projection, paths, values


def test_packager_builds_exact_positive_projection_and_transforms_count(tmp_path: Path) -> None:
    projection_path, projection, _, _ = run_package(tmp_path)
    decision, passing, cells = figure.validate_projection(projection)
    assert decision == "POSITIVE_PROTOTYPE_EFFECT"
    assert passing == 40 and len(cells) == 40
    assert json.loads(projection_path.read_text()) == projection
    encoded = json.dumps(projection)
    assert '"corpus_count"' not in encoded
    assert '"synthetic_corpus_count"' in encoded
    assert "merge_order" not in encoded and "bundle_sha256s" not in encoded


def test_packager_preserves_valid_null_without_relabeling_as_failure(tmp_path: Path) -> None:
    _, projection, _, _ = run_package(tmp_path, all_pass=False)
    decision, passing, _ = figure.validate_projection(projection)
    assert decision == "VALID_NULL_OR_NONPROMOTION"
    assert passing == 39
    assert projection["validity"]["gemma_path_passed"] is True
    assert projection["validity"]["automatic_fallback_used"] is False


def test_no_fallback_calibration_or_failed_preoutcome_is_rejected_before_write(tmp_path: Path) -> None:
    paths, values = source_files(tmp_path)
    calibration, preoutcome, _, _ = values
    calibration["automatic_fallback_allowed"] = True
    write_json(paths[0], calibration)
    output = tmp_path / "projection.json"
    with pytest.raises(packager.PackagingError, match="E_CALIBRATION_GATE"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="a" * 64,
            aggregate_path=paths[2], execution_path=paths[3], output_path=output, enforce_repo_paths=False
        )
    assert not output.exists()
    calibration["automatic_fallback_allowed"] = False
    write_json(paths[0], calibration)
    preoutcome["gates"][packager.runner.PREOUTCOME_GATES[0]] = False
    preoutcome["all_gates_passed"] = False
    write_json(paths[1], preoutcome)
    with pytest.raises(packager.PackagingError, match="E_PREOUTCOME_GATE"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="a" * 64,
            aggregate_path=paths[2], execution_path=paths[3], output_path=output, enforce_repo_paths=False
        )
    assert not output.exists()


def test_restricted_calibration_field_and_aggregate_rows_are_rejected(tmp_path: Path) -> None:
    paths, values = source_files(tmp_path)
    values[0]["transcript_text"] = "synthetic sentinel"
    write_json(paths[0], values[0])
    output = tmp_path / "projection.json"
    with pytest.raises(packager.PackagingError, match="E_CALIBRATION_GATE"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="a" * 64,
            aggregate_path=paths[2], execution_path=paths[3], output_path=output, enforce_repo_paths=False
        )
    assert not output.exists()


def test_aggregate_extra_rows_execution_mismatch_and_bad_auth_digest_reject(tmp_path: Path) -> None:
    paths, values = source_files(tmp_path)
    aggregate = values[2]
    aggregate["result_rows"] = []
    write_json(paths[2], aggregate)
    output = tmp_path / "projection.json"
    with pytest.raises(packager.PackagingError, match="E_AGGREGATE_SCHEMA"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="a" * 64,
            aggregate_path=paths[2], execution_path=paths[3], output_path=output, enforce_repo_paths=False
        )
    assert not output.exists()
    aggregate.pop("result_rows")
    write_json(paths[2], aggregate)
    execution = values[3]
    execution["deterministic_merge_sha256"] = "0" * 64
    write_json(paths[3], execution)
    with pytest.raises(packager.PackagingError, match="E_EXECUTION_GATE"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="a" * 64,
            aggregate_path=paths[2], execution_path=paths[3], output_path=output, enforce_repo_paths=False
        )
    with pytest.raises(packager.PackagingError, match="E_AUTHORIZATION_DIGEST"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="not-a-hash",
            aggregate_path=paths[2], execution_path=paths[3], output_path=output, enforce_repo_paths=False
        )


def test_positive_and_valid_null_memos_are_synthesized_from_hash_bound_figure(tmp_path: Path) -> None:
    for suffix, all_pass, expected in (
        ("positive", True, "POSITIVE_PROTOTYPE_EFFECT"),
        ("null", False, "VALID_NULL_OR_NONPROMOTION"),
    ):
        case = tmp_path / suffix
        case.mkdir()
        projection_path, _, _, _ = run_package(case, all_pass=all_pass)
        svg = case / "figure.svg"
        summary = case / "figure_summary.json"
        figure.generate(projection_path, svg, summary)
        memo_path = case / "memo.md"
        receipt = memo.synthesize(
            projection_path=projection_path,
            figure_summary_path=summary,
            figure_svg_path=svg,
            output_memo_path=memo_path,
            enforce_repo_paths=False,
        )
        text = memo_path.read_text()
        assert receipt["decision"] == expected
        assert expected in text
        assert "Automatic fallback: none" in text
        assert "simulator-oracle" in text
        assert "pseudo-labels and model agreement are not human truth" in text


def test_tampered_figure_summary_creates_no_memo(tmp_path: Path) -> None:
    projection_path, _, _, _ = run_package(tmp_path, all_pass=False)
    svg = tmp_path / "figure.svg"
    summary = tmp_path / "figure_summary.json"
    figure.generate(projection_path, svg, summary)
    value = json.loads(summary.read_text())
    value["passing_directional_cells"] = 40
    write_json(summary, value)
    output = tmp_path / "memo.md"
    with pytest.raises(memo.MemoError, match="E_FIGURE_SUMMARY_BINDING"):
        memo.synthesize(
            projection_path=projection_path,
            figure_summary_path=summary,
            figure_svg_path=svg,
            output_memo_path=output,
            enforce_repo_paths=False,
        )
    assert not output.exists()


def test_symlink_source_and_existing_projection_are_fail_closed(tmp_path: Path) -> None:
    paths, _ = source_files(tmp_path)
    link = tmp_path / "aggregate-link.json"
    link.symlink_to(paths[2])
    output = tmp_path / "projection.json"
    with pytest.raises(packager.PackagingError, match="E_INPUT_FILE"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="a" * 64,
            aggregate_path=link, execution_path=paths[3], output_path=output, enforce_repo_paths=False
        )
    output.write_text("sentinel")
    with pytest.raises(packager.PackagingError, match="E_OUTPUT_EXISTS_OR_UNSAFE"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="a" * 64,
            aggregate_path=paths[2], execution_path=paths[3], output_path=output, enforce_repo_paths=False
        )
    assert output.read_text() == "sentinel"


def test_production_mode_rejects_inputs_outside_repository_public_output(tmp_path: Path) -> None:
    paths, _ = source_files(tmp_path)
    with pytest.raises(packager.PackagingError, match="E_INPUT_LOCATION"):
        packager.package(
            calibration_path=paths[0], preoutcome_path=paths[1], authorization_seal_sha256="a" * 64,
            aggregate_path=paths[2], execution_path=paths[3], output_path=tmp_path / "projection.json",
            enforce_repo_paths=True,
        )


def test_sources_have_no_network_subprocess_or_recursive_discovery() -> None:
    combined = PACKAGER_PATH.read_text() + MEMO_PATH.read_text()
    for forbidden in ("requests", "urllib", "socket", "subprocess", ".glob(", ".rglob(", "os.walk"):
        assert forbidden not in combined
