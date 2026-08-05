from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from nursery_egobaby_preflight.contract import compact_aggregate_json


SPEC = importlib.util.spec_from_file_location(
    "public_readiness", Path("scripts/run_synthetic_video_calibration.py")
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )


def _config_with_topology() -> dict:
    config = _config()
    amendment = config["public_only_calibration_readiness_amendment"]
    topology = {
        "status": "FROZEN_SCHEDULER_ONLY_BEFORE_MODEL_OUTCOMES",
        "route_id": amendment["route_id"],
        "amendment_commitment_sha256": amendment[
            "amendment_commitment_sha256"
        ],
        "selected_at": "2026-08-05T00:00:00Z",
        "selection_evidence": "SCHEDULER_ONLY_NO_MODEL_OUTCOME",
        "GPU_type": "NVIDIA_A30_24GB",
        "partition": "a30",
        "GRES": "gpu:nvidia_a30:1",
        "expected_device_name_prefix": "NVIDIA A30",
        "visible_memory_GiB_min": 23,
        "visible_memory_GiB_max": 25,
        "node_count": 1,
        "task_count": 1,
        "GPU_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "single_process": True,
        "DDP": False,
        "micro_wall_minutes": 5,
        "scientific_partition_wall_minutes": 15,
        "micro_attempt_count_max": 2,
        "aggregate_GPU_hours_max": 2.0 / 3.0,
        "new_storage_GiB_max": 10,
        "direct_monetary_cost_USD": 0,
    }
    topology["topology_commitment_sha256"] = MODULE.digest(topology)
    config["public_only_calibration_readiness_topology"] = topology
    return config


def _event(
    case_id: str,
    episode_id: str,
    part: str,
    lemma: str,
    band: str = "high",
) -> dict:
    return {
        "case_id": case_id,
        "episode_id": episode_id,
        "part_of_speech": part,
        "lemma": lemma,
        "frequency_band": band,
    }


def test_public_readiness_amendment_preserves_terminal_prior_result() -> None:
    config = _config()
    amendment = MODULE._public_only_readiness_amendment(config)
    assert amendment["amendment_commitment_sha256"] == (
        "03bbf64749b0302a16e97f9b999674e287b7f4fa801df905d09b72ea9c39eeae"
    )
    assert amendment["prior_terminal_result_preserved"][
        "result_commitment_sha256"
    ] == "42338302949e27e0ed7c3f6e8a5f70e10bb380a5e8158378e89f5ff87c350e9d"
    assert amendment["absolute_scope"]["public_only"] is True
    assert amendment["absolute_scope"]["governed_C"] is False


def test_public_readiness_topology_is_separately_committed() -> None:
    config = _config_with_topology()
    topology = MODULE._public_readiness_topology(config)
    assert topology["GPU_count"] == 1
    assert topology["micro_wall_minutes"] == 5
    assert topology["scientific_partition_wall_minutes"] == 15

    mutated = json.loads(json.dumps(config))
    mutated["public_only_calibration_readiness_topology"]["GPU_count"] = 2
    with pytest.raises(RuntimeError, match="E_PUBLIC_READINESS_TOPOLOGY_COMMITMENT"):
        MODULE._public_readiness_topology(mutated)


def test_public_readiness_fixture_result_is_sealed_before_inference() -> None:
    result = MODULE._public_readiness_fixture_result(_config())
    assert result["readiness_fixture_commitment_sha256"] == (
        "2b30f892ee327e2a1e39e35e41fec7a1e60adeef9b3df4c4c594e8eab45e1b1b"
    )
    assert result["attribute_item_count"] == 64
    assert result["manual_annotation_count"] == 0
    assert result["model_inference_executed"] is False


def test_zero_job_preallocation_repair_changes_no_scientific_rule() -> None:
    result = MODULE._public_readiness_preallocation_repair(_config())
    assert result["job_created"] is False
    assert result["GPU_allocated"] is False
    assert result["GPU_hours"] == 0
    assert result["micro_attempt_consumed"] is False
    assert result["model_fixture_threshold_partition_metric_or_gate_changed"] is False


def test_readiness_lexical_metrics_measure_aggregate_estimands() -> None:
    expected = [
        _event("c1", "e1", "adjective", "red"),
        _event("c1", "e1", "noun", "ball"),
        _event("c2", "e1", "noun", "ball"),
        _event("c3", "e2", "adjective", "small", "mid"),
        _event("c3", "e2", "noun", "cup", "low"),
    ]
    metrics = MODULE._readiness_lexical_aggregate_metrics(
        expected, list(expected), {"c1", "c2", "c3"}, {"c1", "c2", "c3"}
    )
    assert metrics["eligible_case_coverage"] == 1.0
    assert metrics["noun_total_count_relative_error"] == 0.0
    assert metrics["adjective_type_count_relative_error"] == 0.0
    assert metrics["frequency_band_total_variation"] == 0.0
    assert metrics["episode_repetition_distribution_total_variation"] == 0.0
    assert metrics["long_tail_share_absolute_error"] == 0.0
    gate = _config()["public_only_calibration_readiness_amendment"][
        "axis_methods"
    ]["noun_adjective_exposure"]["gate"]
    assert MODULE._readiness_lexical_gate_pass(metrics, gate) is True

    missing = MODULE._readiness_lexical_aggregate_metrics(
        expected,
        expected[:-2],
        {"c1", "c2", "c3"},
        {"c1", "c2"},
    )
    assert MODULE._readiness_lexical_gate_pass(missing, gate) is False


def test_readiness_attribute_metrics_cover_all_families_and_semantics() -> None:
    rows = []
    for family, pair in MODULE.READINESS_ATTRIBUTE_PAIRS.items():
        first, second = pair
        rows.extend(
            [
                {
                    "family": family,
                    "expected_relation": "positive",
                    "mentioned_label": first,
                    "pe_label": first,
                    "pe_margin": 1.0,
                    "deterministic_label": first,
                },
                {
                    "family": family,
                    "expected_relation": "opposite",
                    "mentioned_label": first,
                    "pe_label": second,
                    "pe_margin": 1.0,
                    "deterministic_label": second,
                },
                {
                    "family": family,
                    "expected_relation": "null",
                    "mentioned_label": None,
                    "pe_label": first,
                    "pe_margin": 1.0,
                    "deterministic_label": first,
                },
                {
                    "family": family,
                    "expected_relation": "ambiguous",
                    "mentioned_label": first,
                    "pe_label": first,
                    "pe_margin": 0.0,
                    "deterministic_label": first,
                },
            ]
        )
    mask = {
        "predicted_mask_coverage": 1.0,
        "median_predicted_mask_IoU": 1.0,
        "predicted_mask_negative_specificity": 1.0,
        "invalid_mask_count": 0,
        "positive_sample_count": 16,
        "negative_sample_count": 16,
    }
    metrics = MODULE._readiness_attribute_metrics(rows, 0.1, 1.0, mask)
    gate = _config()["public_only_calibration_readiness_amendment"][
        "axis_methods"
    ]["adjective_attribute_contrast"]["gate"]
    assert set(metrics["family_metrics"]) == {
        "color",
        "relative_size",
        "shape",
        "state",
    }
    assert all(
        value == {"macro_f1": 1.0, "coverage": 1.0}
        for value in metrics["family_metrics"].values()
    )
    assert MODULE._readiness_attribute_gate_pass(metrics, gate) is True

    rows[0]["pe_label"] = MODULE.READINESS_ATTRIBUTE_PAIRS["color"][1]
    degraded = MODULE._readiness_attribute_metrics(rows, 0.1, 1.0, mask)
    assert MODULE._readiness_attribute_gate_pass(degraded, gate) is False


def test_attribute_executes_after_independent_referent_no_go(tmp_path: Path) -> None:
    calls = []

    def runner(module_id: str):
        def execute(_context: dict) -> dict:
            calls.append(module_id)
            status = "NO_GO" if module_id == "referent" else "PASS"
            if module_id == "order_action":
                return {"status": "NO_GO_DIAGNOSTIC"}
            return {
                "status": status,
                "axis_results": {
                    axis_id: {"status": status, "metrics": {}}
                    for axis_id in MODULE.TUPLE_MODULE_AXIS_IDS[module_id]
                },
                "metrics": {},
                "selected_thresholds": {},
                "rows": [],
                "row_count": 0,
                "failure_count": 0,
                "invalid_retained_record_count": 0,
                "silent_truncation_count": 0,
                "external_call_count": 0,
            }

        return execute

    runners = {
        module_id: runner(module_id)
        for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
    }
    results, errors = MODULE._public_readiness_scientific_module_results(
        runners, {}, tmp_path
    )
    assert errors == []
    assert results["referent"]["status"] == "NO_GO"
    assert results["attribute"]["status"] == "PASS"
    assert calls == list(MODULE.TUPLE_QUALIFICATION_MODULE_IDS)


def test_programmatic_readiness_overlay_is_balanced_and_deterministic(
    tmp_path: Path,
) -> None:
    public_root = tmp_path / "external-public"
    args = argparse.Namespace(public_root=public_root, config=Path(
        "configs/synthetic_video_real_only_proof.json"
    ))
    first = MODULE.prepare_public_readiness_fixtures(args)
    second = MODULE.prepare_public_readiness_fixtures(args)
    assert first == second
    assert first["attribute_item_count"] == 64
    assert first["manual_annotation_count"] == 0
    manifest, _ = MODULE._load_public_readiness_fixture_manifest(
        public_root, _config()
    )
    for partition in ("development", "holdout"):
        rows = manifest["partitions"][partition]
        assert len(rows) == 32
        counts = {
            (family, semantic): sum(
                row["family"] == family and row["semantic_case"] == semantic
                for row in rows
            )
            for family in MODULE.READINESS_ATTRIBUTE_PAIRS
            for semantic in MODULE.READINESS_ATTRIBUTE_SEMANTICS
        }
        assert set(counts.values()) == {2}


def test_readiness_thresholds_cannot_escape_frozen_grid() -> None:
    config = _config()
    module_results = {
        module_id: {"selected_thresholds": {}}
        for module_id in MODULE.TUPLE_QUALIFICATION_MODULE_IDS
    }
    module_results["attribute"]["selected_thresholds"] = {
        "PE_Core_attribute_margin": 0.03
    }
    with pytest.raises(RuntimeError, match="E_PUBLIC_READINESS_THRESHOLD_NOT_FROZEN"):
        MODULE._public_readiness_selected_thresholds(config, module_results)


def test_readiness_combined_gate_keeps_action_diagnostic_nonblocking() -> None:
    axes = {
        axis_id: {"status": "PASS", "metrics": {}}
        for axis_id in (
            *MODULE.TUPLE_CRITICAL_AXIS_IDS,
            *MODULE.TUPLE_SUPPORTING_AXIS_IDS,
        )
    }
    decision = MODULE._tuple_combined_public_gate(
        axes,
        {"status": "NO_GO_DIAGNOSTIC"},
        {"status": "DESCRIPTIVE_ONLY"},
        action_control_blocks=False,
    )
    assert decision["status"] == "PASS"
    assert decision["action_control_used_in_gate"] is False

    axes[MODULE.TUPLE_CRITICAL_AXIS_IDS[0]]["status"] = "NO_GO"
    assert MODULE._tuple_combined_public_gate(
        axes, {"status": "PASS"}, action_control_blocks=False
    )["status"] == "NO_GO"


def test_readiness_cli_output_cannot_print_rows_or_paths() -> None:
    record = {
        field: 0
        for field in MODULE.PUBLIC_READINESS_HEALTH_FIELDS
    }
    record["status"] = "PASS_ENGINEERING_HEALTH"
    for field in MODULE.PUBLIC_READINESS_HEALTH_HASH_FIELDS:
        record[field] = "a" * 64
    record["rows"] = []
    with pytest.raises(ValueError, match="whitelist"):
        compact_aggregate_json(
            record,
            allowed_fields=MODULE.PUBLIC_READINESS_HEALTH_FIELDS,
            sha256_fields=MODULE.PUBLIC_READINESS_HEALTH_HASH_FIELDS,
        )


def test_wrapper_has_readiness_blocking_job_contract_without_poll_loop() -> None:
    wrapper = Path("scripts/qualify_synthetic_video_calibration.sbatch").read_text()
    assert "PHASE4_READINESS_RUN_MODE" in wrapper
    assert "readiness-prepare" in wrapper
    assert "readiness-health" in wrapper
    assert "readiness-qualify" in wrapper
    assert "--network none" in wrapper
    assert "require_readiness_topology" in wrapper
    assert "ChildLens" not in wrapper
    assert "BabyView" not in wrapper
    assert "squeue" not in wrapper
    assert "sleep " not in wrapper
