from __future__ import annotations

import importlib.util
import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "calibration_health", Path("scripts/run_synthetic_video_calibration.py")
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MODULE_IDS = (
    "adapter_and_lexical",
    "referent",
    "recurrence",
    "attribute",
    "hand_contact",
    "sensor",
    "order_action",
)
FIXTURE_COMMITMENT = (
    "2758557fe4844225220192eb285526d90b8420f730b946374d03163c7903dae6"
)


def _config() -> dict:
    return json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )


def _historical_health_config() -> dict:
    config = _config()
    config["schema_version"] = 21
    config.pop("learner_effective_engineering_health_result")
    config.pop("learner_effective_engineering_health_reauthorization")
    config.pop("learner_effective_engineering_health_reauthorization_result")
    config.pop("learner_effective_engineering_health_parser_repair_reauthorization")
    config.pop("learner_effective_engineering_health_parser_repair_result")
    config.pop("learner_effective_engineering_health_iterative_reauthorization")
    config.pop("learner_effective_engineering_health_iterative_attempt_6_result")
    config.pop("learner_effective_engineering_health_progress_repair")
    config.pop("learner_effective_engineering_health_attempt_7_result")
    config.pop("learner_effective_engineering_health_extended_wall_repair")
    return config


def _write_topology_attestation(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    attempt: int = 4,
    job_id: int = 316371,
) -> Path:
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    value = {
        "schema_version": 1,
        "attempt": attempt,
        "job_id": job_id,
        "partition": "h100",
        "node_count": 1,
        "CPU_count": 8,
        "task_count": 1,
        "time_limit_minutes": 60 if attempt == 8 else 15,
        "memory_per_CPU_GiB": 4,
        "GRES": "gpu:nvidia_h100_nvl_3g.47gb:1",
        "predicate_count": 7,
        "predicate_pass_count": 7,
        "world_size": 1,
        "local_world_size": 1,
        "source": "WRAPPER_SCONTROL_BEFORE_CONTAINER",
    }
    path = root / "topology-attestation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MODULE.canonical(value) + b"\n")
    path.chmod(0o600)
    return path


def _write_container_attestation(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_mode: str = "health",
    attempt: int | None = 6,
    job_id: int = 316538,
) -> Path:
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    value = {
        "artifact_family": "BASE_CONTAINER",
        "attempt": attempt,
        "bytes": 3731320832,
        "host_entry_is_symlink": True,
        "job_id": job_id,
        "predicate_count": 4,
        "predicate_pass_count": 4,
        "resolved_target_regular_file": True,
        "run_mode": run_mode,
        "schema_version": 1,
        "sha256": "f274f1ac3726376b762b557ff9a07203b2d42aac3157a7a354b998e589c35792",
        "source": "WRAPPER_HOST_BEFORE_CONTAINER",
    }
    path = root / "container-attestation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MODULE.canonical(value) + b"\n")
    path.chmod(0o600)
    return path


def _development_rows() -> dict[str, list[dict]]:
    language = [
        {
            "case_id": "development-accept-ball-1",
            "partition": "development",
            "expected_adapter_status": "ACCEPT",
            "expected_adapter_reason": None,
            "expected_lexical_mentions": [
                {"token": "red", "part_of_speech": "adjective"},
                {"token": "ball", "part_of_speech": "noun"},
            ],
        },
        {
            "case_id": "development-abstain-language-mismatch-0",
            "partition": "development",
            "expected_adapter_status": "ABSTAIN",
            "expected_adapter_reason": "LANGUAGE_MISMATCH",
            "expected_lexical_mentions": [],
        },
        {
            "case_id": "development-abstain-invalid-timestamp-0",
            "partition": "development",
            "expected_adapter_status": "ABSTAIN",
            "expected_adapter_reason": "INVALID_TIMESTAMP",
            "expected_lexical_mentions": [],
        },
        {
            "case_id": "development-abstain-silent-truncation-0",
            "partition": "development",
            "expected_adapter_status": "ABSTAIN",
            "expected_adapter_reason": "SILENT_TRUNCATION",
            "expected_lexical_mentions": [],
        },
    ]
    referent_attribute = [
        {
            "fixture_ordinal": 0,
            "scenario": "during_only",
            "truth": {
                "visibility_by_phase": {
                    "before": False,
                    "during": True,
                    "after": False,
                },
                "attribute_contrast_expected": True,
                "attribute_null_reason": None,
            },
        },
        {
            "fixture_ordinal": 1,
            "scenario": "speech_no_referent",
            "truth": {
                "visibility_by_phase": {
                    "before": False,
                    "during": False,
                    "after": False,
                },
                "attribute_contrast_expected": False,
                "attribute_null_reason": "NO_PREDICTED_REFERENT_MASK",
            },
        },
        {
            "fixture_ordinal": 2,
            "scenario": "persistent_dominant_with_small_distractor",
            "truth": {
                "visibility_by_phase": {
                    "before": True,
                    "during": True,
                    "after": True,
                },
                "attribute_contrast_expected": True,
                "attribute_null_reason": None,
            },
        },
        {
            "fixture_ordinal": 3,
            "scenario": "persistent_ambiguous",
            "truth": {
                "visibility_by_phase": {
                    "before": True,
                    "during": True,
                    "after": True,
                },
                "attribute_contrast_expected": True,
                "attribute_null_reason": None,
            },
        },
        {
            "fixture_ordinal": 4,
            "scenario": "persistent_clear",
            "truth": {
                "visibility_by_phase": {
                    "before": True,
                    "during": True,
                    "after": True,
                },
                "attribute_contrast_expected": True,
                "attribute_null_reason": None,
            },
        },
        {
            "fixture_ordinal": 5,
            "scenario": "no_speech_visible_object",
            "truth": {
                "visibility_by_phase": {
                    "before": True,
                    "during": True,
                    "after": True,
                },
                "attribute_contrast_expected": False,
                "attribute_null_reason": "NO_ACCEPTED_ADJECTIVE_NOUN_SPAN",
            },
        },
    ]
    recurrence = [
        {
            "fixture_ordinal": 0,
            "stratum": "same_instance_transformed",
            "same_referent": True,
            "near_duplicate": False,
        },
        {
            "fixture_ordinal": 1,
            "stratum": "same_instance_near_duplicate",
            "same_referent": True,
            "near_duplicate": True,
        },
        {
            "fixture_ordinal": 2,
            "stratum": "same_category_different_instance",
            "same_referent": False,
            "near_duplicate": False,
        },
        {
            "fixture_ordinal": 3,
            "stratum": "different_category",
            "same_referent": False,
            "near_duplicate": False,
        },
    ]
    hand_contact = [
        {"fixture_ordinal": 0, "stratum": "contact", "visible_hand": True},
        {
            "fixture_ordinal": 1,
            "stratum": "verified_no_hand",
            "visible_hand": False,
        },
        {"fixture_ordinal": 2, "stratum": "contact", "visible_hand": True},
        {
            "fixture_ordinal": 3,
            "stratum": "explicit_no_contact",
            "visible_hand": True,
        },
    ]
    sensor = [
        {"fixture_ordinal": 0, "condition": "static"},
        {"fixture_ordinal": 1, "condition": "high_translation"},
        {"fixture_ordinal": 2, "condition": "strong_blur"},
        {"fixture_ordinal": 3, "condition": "hard_cut"},
    ]
    order_action = [
        {
            "fixture_ordinal": 0,
            "control_role": "ordered_positive_1",
            "video": "development-action-0",
        },
        {
            "fixture_ordinal": 1,
            "control_role": "ordered_positive_2",
            "video": "development-action-1",
        },
        {
            "fixture_ordinal": 2,
            "control_role": "reversed_control",
            "video": "development-action-2",
        },
        {
            "fixture_ordinal": 3,
            "control_role": "repeated_center_control",
            "video": "development-action-3",
        },
    ]
    return {
        "language_lexical": language,
        "referent_attribute": referent_attribute,
        "recurrence": recurrence,
        "hand_contact": hand_contact,
        "sensor": sensor,
        "order_action": order_action,
    }


def _fixture_manifest() -> dict:
    development = _development_rows()
    holdout = {
        family: [
            {
                "fixture_ordinal": 9000 + ordinal,
                "partition": "holdout",
                "poison": "holdout-must-never-be-projected",
            }
            for ordinal, _row in enumerate(rows)
        ]
        for family, rows in development.items()
    }
    return {
        "schema_version": 3,
        "status": "SEALED_BEFORE_PUBLIC_DEVELOPMENT_INFERENCE",
        "public_fixture_manifest_commitment_sha256": FIXTURE_COMMITMENT,
        "verified_no_hand_seal_commitment_sha256": "a" * 64,
        "partitions": {"development": development, "holdout": holdout},
    }


def _healthy_modules() -> list[dict]:
    return [
        {
            "module_id": module_id,
            "status": "PASS_ENGINEERING",
            "case_count": 4,
            "failure_count": 0,
            "invalid_retained_record_count": 0,
            "silent_truncation_count": 0,
            "external_call_count": 0,
            "unaccounted_failure_count": 0,
            "finite_in_bounds": True,
            "schema_valid": True,
            "deterministic_outputs": True,
            "round_trip_valid": True,
            "valid_negative_state_distinct": True,
            "abstention_state_distinct": True,
            "missing_error_state_distinct": True,
            "production_output_commitment_sha256": MODULE.digest(
                {"module_id": module_id, "case_count": 4}
            ),
        }
        for module_id in MODULE_IDS
    ]


def _healthy_full() -> dict:
    full = {
        "schema_version": 1,
        "status": "PASS_ENGINEERING_HEALTH",
        "route_id": "construct-aligned-engineering-health",
        "attempt": 1,
        "public_fixture_manifest_commitment_sha256": FIXTURE_COMMITMENT,
        "runner_commitment_sha256": "b" * 64,
        "config_commitment_sha256": MODULE.digest(_config()),
        "dependency_config_commitment_sha256": "c" * 64,
        "microfixture_manifest_commitment_sha256": "d" * 64,
        "module_results": _healthy_modules(),
        "module_count": 7,
        "completed_module_count": 7,
        "failed_module_count": 0,
        "case_count": 28,
        "holdout_input_count": 0,
        "scientific_metric_count": 0,
        "failure_count": 0,
        "invalid_retained_record_count": 0,
        "silent_truncation_count": 0,
        "external_call_count": 0,
        "unaccounted_failure_count": 0,
        "network_disabled": True,
        "telemetry_disabled": True,
        "restricted_mount_present": False,
        "resource": {
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
            "wall_minutes": 10.0,
            "GPU_hours": 1 / 6,
            "new_storage_GiB": 1.0,
            "direct_monetary_cost_USD": 0,
        },
        "cumulative_resource": {
            "cumulative_submission_count": 1,
            "cumulative_wall_minutes": 10.0,
            "cumulative_GPU_hours": 1 / 6,
            "cumulative_new_storage_GiB": 1.0,
            "cumulative_direct_monetary_cost_USD": 0,
        },
    }
    full["engineering_health_commitment_sha256"] = MODULE.digest(full)
    return full


def test_tuple_health_projection_is_exact_complete_and_order_stable() -> None:
    config = _config()
    manifest = _fixture_manifest()
    first = MODULE._tuple_health_projection(manifest, config)

    reversed_manifest = json.loads(json.dumps(manifest))
    for rows in reversed_manifest["partitions"]["development"].values():
        rows.reverse()
    second = MODULE._tuple_health_projection(reversed_manifest, config)

    assert first == second
    assert first["role"] == "EXECUTION_HEALTH_ONLY"
    assert first["source_partition"] == "development"
    assert first["case_count"] == 28
    assert first["module_count"] == 7
    assert first["holdout_input_count"] == 0
    assert first["scientific_metric_count"] == 0
    assert len(first["cases"]) == 28
    assert [case["module_id"] for case in first["cases"]] == [
        module_id for module_id in MODULE_IDS for _ordinal in range(4)
    ]
    expected = config["learner_effective_engineering_health_amendment"][
        "engineering_microfixture_suite"
    ]["required_case_classes"]
    assert {
        module_id: [
            case["case_class"]
            for case in first["cases"]
            if case["module_id"] == module_id
        ]
        for module_id in MODULE_IDS
    } == expected
    assert "holdout-must-never-be-projected" not in json.dumps(first)


@pytest.mark.parametrize(
    "mutation,error_code",
    [
        ("missing_class", "E_TUPLE_HEALTH_CASE_CLASS_DEFICIT"),
        ("holdout_only", "E_TUPLE_HEALTH_DEVELOPMENT_FIXTURES"),
    ],
)
def test_tuple_health_projection_fails_closed_on_case_deficit_or_holdout(
    mutation: str, error_code: str
) -> None:
    manifest = _fixture_manifest()
    if mutation == "missing_class":
        manifest["partitions"]["development"]["recurrence"] = manifest[
            "partitions"
        ]["development"]["recurrence"][:3]
    else:
        manifest["partitions"]["development"] = {
            family: [] for family in manifest["partitions"]["development"]
        }
    with pytest.raises(RuntimeError, match=error_code):
        MODULE._tuple_health_projection(manifest, _config())


def test_tuple_health_error_is_specific_compact_and_writes_private_trace(
    tmp_path: Path,
) -> None:
    raw = "participant-name /restricted/source/file.mp4"
    record = MODULE._tuple_health_error(
        "referent", RuntimeError(raw), tmp_path
    )

    assert record == {
        "module_id": "referent",
        "status": "ERROR",
        "error_code": "E_TUPLE_HEALTH_REFERENT_UNACCOUNTED_FAILURE",
        "trace_written": True,
    }
    assert raw not in json.dumps(record)
    traces = list(tmp_path.iterdir())
    assert len(traces) == 1
    assert traces[0].stat().st_mode & 0o777 == 0o600
    assert raw in traces[0].read_text()


def test_tuple_health_error_preserves_stable_code_without_raw_message(
    tmp_path: Path,
) -> None:
    first = MODULE._tuple_health_error(
        "hand_contact",
        RuntimeError("E_TUPLE_EGOHOS_OBSERVATION_SCHEMA private-detail-one"),
        tmp_path,
    )
    second = MODULE._tuple_health_error(
        "hand_contact",
        RuntimeError("E_TUPLE_EGOHOS_OBSERVATION_SCHEMA private-detail-two"),
        tmp_path,
    )
    assert first["error_code"] == second["error_code"] == (
        "E_TUPLE_HEALTH_HAND_CONTACT_TUPLE_EGOHOS_OBSERVATION_SCHEMA"
    )
    assert "private-detail" not in json.dumps([first, second])
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_MODULE_ID"):
        MODULE._tuple_health_error("unknown", RuntimeError("boom"), tmp_path)


@pytest.mark.parametrize(
    "sentinel",
    ("E_SENTINEL_PRIVATE_ROW", "E_TUPLE_SENTINEL_PRIVATE_ROW"),
)
def test_tuple_health_error_cannot_self_authorize_arbitrary_error_codes(
    tmp_path: Path, sentinel: str
) -> None:
    record = MODULE._tuple_health_error(
        "referent", RuntimeError(f"{sentinel} private-detail"), tmp_path
    )
    assert record["error_code"] == (
        "E_TUPLE_HEALTH_REFERENT_UNACCOUNTED_FAILURE"
    )
    assert sentinel not in json.dumps(record)


def test_tuple_health_trace_names_are_fixed_and_input_independent(
    tmp_path: Path,
) -> None:
    MODULE._tuple_health_error(
        "referent", RuntimeError("first /restricted/name.mp4"), tmp_path
    )
    MODULE._tuple_health_error(
        "referent", RuntimeError("second participant-name"), tmp_path
    )
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["referent-01.trace", "referent-02.trace"]
    assert all(re.fullmatch(r"referent-[0-9]{2}\.trace", name) for name in names)


def test_tuple_health_full_and_compact_validators_withhold_case_details() -> None:
    full = _healthy_full()
    MODULE._validate_tuple_health_full(full, _config())
    compact = MODULE._tuple_health_compact(full)
    MODULE._validate_tuple_health_compact(compact)

    assert compact["status"] == "PASS_ENGINEERING_HEALTH"
    assert compact["module_count"] == 7
    assert compact["completed_module_count"] == 7
    assert compact["case_count"] == 28
    assert compact["holdout_input_count"] == 0
    assert compact["scientific_metric_count"] == 0
    serialized = json.dumps(compact).casefold()
    for prohibited in (
        "case_class",
        "module_results",
        "path",
        "filename",
        "label",
        "prediction",
        "score",
        "metrics",
    ):
        assert prohibited not in serialized


@pytest.mark.parametrize(
    "forbidden_key,forbidden_value",
    [
        ("case_details", [{"case_id": "private-row"}]),
        ("trace_path", "/restricted/trace.log"),
        ("row_hashes", ["e" * 64]),
        ("model_sha256", "e" * 64),
        ("labels", ["private-label"]),
        ("predictions", [0.4]),
        ("scores", [0.4]),
        ("metrics", {"macro_f1": 0.9}),
    ],
)
def test_tuple_health_compact_validator_rejects_row_level_or_scientific_fields(
    forbidden_key: str, forbidden_value: object
) -> None:
    compact = MODULE._tuple_health_compact(_healthy_full())
    compact[forbidden_key] = forbidden_value
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_COMPACT_SCHEMA"):
        MODULE._validate_tuple_health_compact(compact)


def test_tuple_health_full_validator_rejects_metrics_holdout_and_nonfinite() -> None:
    full = _healthy_full()
    full["scientific_metrics"] = {"macro_f1": 0.9}
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_SCIENTIFIC_METRIC"):
        MODULE._validate_tuple_health_full(full, _config())

    full = _healthy_full()
    full["holdout_input_count"] = 1
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_HOLDOUT_PROHIBITED"):
        MODULE._validate_tuple_health_full(full, _config())

    full = _healthy_full()
    full["resource"]["GPU_hours"] = float("nan")
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_NONFINITE"):
        MODULE._validate_tuple_health_full(full, _config())

    full = _healthy_full()
    full["module_results"][0]["valid_negative_state_distinct"] = False
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_STATE_DISTINCTION"):
        MODULE._validate_tuple_health_full(full, _config())


def test_tuple_health_metric_release_requires_seven_unique_passed_modules() -> None:
    scientific = {"realistic_macro": 0.51}
    assert MODULE._tuple_health_metric_release(
        _healthy_modules(), scientific
    ) == scientific

    errored = _healthy_modules()
    errored[2]["status"] = "ERROR"
    errored[2]["error_code"] = "E_TUPLE_HEALTH_RECURRENCE_RUNTIME"
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_METRICS_WITHHELD"):
        MODULE._tuple_health_metric_release(errored, scientific)

    missing = _healthy_modules()[:-1]
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_METRICS_WITHHELD"):
        MODULE._tuple_health_metric_release(missing, scientific)

    duplicate = _healthy_modules()
    duplicate[-1]["module_id"] = duplicate[0]["module_id"]
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_METRICS_WITHHELD"):
        MODULE._tuple_health_metric_release(duplicate, scientific)

    wrong_status = _healthy_modules()
    wrong_status[0]["status"] = "PASS"
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_METRICS_WITHHELD"):
        MODULE._tuple_health_metric_release(wrong_status, scientific)


def test_tuple_health_budget_enforces_active_extended_wall_attempt_and_limits() -> None:
    config = _config()
    prior = [
        {
            "attempt": 1,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "wall_minutes": 15.0,
            "GPU_hours": 0.25,
            "new_storage_GiB": 0.0,
            "direct_monetary_cost_USD": 0,
        },
        {
            "attempt": 2,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "wall_minutes": 15.0,
            "GPU_hours": 0.25,
            "new_storage_GiB": 2.905726432800293e-07,
            "direct_monetary_cost_USD": 0,
        },
        {
            "attempt": 3,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "wall_minutes": 0.24115029176076253,
            "GPU_hours": 0.004019171529346042,
            "new_storage_GiB": 1.1119991540908813e-06,
            "direct_monetary_cost_USD": 0,
        },
        {
            "attempt": 4,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "wall_minutes": 15.0,
            "GPU_hours": 0.25,
            "new_storage_GiB": 1.0170042514801025e-06,
            "direct_monetary_cost_USD": 0,
        },
        {
            "attempt": 5,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "wall_minutes": 0.36148459911346437,
            "GPU_hours": 0.006024743318557739,
            "new_storage_GiB": 9.872019290924072e-7,
            "direct_monetary_cost_USD": 0,
        },
        {
            "attempt": 6,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "wall_minutes": 15.0,
            "GPU_hours": 0.25,
            "new_storage_GiB": 7.729977369308472e-7,
            "direct_monetary_cost_USD": 0,
        },
        {
            "attempt": 7,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "wall_minutes": 15.0,
            "GPU_hours": 0.25,
            "new_storage_GiB": 9.955838322639465e-7,
            "direct_monetary_cost_USD": 0,
        },
    ]
    budget = MODULE._tuple_health_budget(8, prior, config)
    assert budget["attempt"] == 8
    assert budget["GPU_type"] == "NVIDIA_H100_NVL_3G_47GB_MIG"
    assert budget["GPU_count"] == 1
    assert budget["per_submission_wall_minutes_max"] == 60
    assert budget["remaining_GPU_hours"] == pytest.approx(1.0)
    assert budget["remaining_storage_GiB"] == pytest.approx(1.0)
    assert budget["direct_monetary_cost_USD"] == 0

    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_ATTEMPT_BUDGET"):
        MODULE._tuple_health_budget(9, prior, config)


@pytest.mark.parametrize(
    "field,value,error_code",
    [
        ("GPU_type", "NVIDIA_A30_24GB", "E_TUPLE_HEALTH_GPU_TOPOLOGY"),
        ("GPU_count", 2, "E_TUPLE_HEALTH_GPU_TOPOLOGY"),
        ("wall_minutes", 60.1, "E_TUPLE_HEALTH_WALL_BUDGET"),
        ("GPU_hours", 1.1, "E_TUPLE_HEALTH_GPU_HOUR_BUDGET"),
        ("new_storage_GiB", 1.1, "E_TUPLE_HEALTH_STORAGE_BUDGET"),
        ("direct_monetary_cost_USD", 0.01, "E_TUPLE_HEALTH_COST_BUDGET"),
    ],
)
def test_tuple_health_budget_rejects_topology_or_resource_overrun(
    field: str, value: object, error_code: str
) -> None:
    prior = [
        {
            "attempt": 1,
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "wall_minutes": 10.0,
            "GPU_hours": 0.0,
            "new_storage_GiB": 0.0,
            "direct_monetary_cost_USD": 0,
            field: value,
        }
    ]
    with pytest.raises(RuntimeError, match=error_code):
        MODULE._tuple_health_budget(2, prior, _config())


def test_h100_resource_redirect_is_hash_bound_and_scientifically_narrow() -> None:
    config = _config()
    original = MODULE._engineering_health_amendment(config)
    redirect = MODULE._engineering_health_resource_redirect(config)
    assert original["amendment_commitment_sha256"] == (
        "d447a7e165136032a1fba43605d3f81881b41ec030c82e9028e1a8f5cb2c6205"
    )
    assert redirect["scope"] == "SCHEDULER_AND_RESOURCE_LATENCY_ONLY"
    assert redirect["preserved_without_change"][
        "production_models_fixtures_thresholds_metrics_seeds_runner_behavior_repair_allowances_and_downstream_gates"
    ] is True
    assert redirect["canceled_A30_submission"]["engineering_outcome_opened"] is False
    assert redirect["canceled_A30_submission"]["scientific_metric_count"] == 0

    mutated = json.loads(json.dumps(config))
    amendment = mutated["learner_effective_engineering_health_resource_redirect"]
    amendment["active_health_topology"]["GPU_count"] = 2
    payload = json.loads(json.dumps(amendment))
    payload.pop("amendment_commitment_sha256")
    amendment["amendment_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_RESOURCE_REDIRECT_COMMITMENT"
    ):
        MODULE._engineering_health_resource_redirect(mutated)


def test_dependency_restore_is_hash_bound_preinference_and_scientifically_narrow() -> None:
    config = _config()
    restore = MODULE._engineering_health_dependency_restore(config)
    assert restore["repair_commitment_sha256"] == (
        "3c54503b4087fae1e993b0aa952823f988a088a5c6543760df1022e2dc046db4"
    )
    assert restore["triggering_attempt"]["classification"] == (
        "ENGINEERING_DEPENDENCY_CACHE_MISS_NOT_SCIENTIFIC_NO_GO"
    )
    assert restore["triggering_attempt"]["model_module_inference_count"] == 0
    assert restore["triggering_attempt"]["scientific_metric_count"] == 0
    assert restore["public_restore_execution"][
        "offline_local_files_only_reload_after_preparation"
    ] == "PASS"
    assert restore["active_language_dependency_archive"] == {
        "relative_location": "models/phase4-language-pydeps.tar under the public root",
        "sha256": "97ef52ecaa8c99db017e598d8a63d0d2170affef14ef46e7df7a656abd3a1a07",
        "bytes": 542423040,
        "normalized_regular_file_count": 11512,
        "normalized_regular_file_bytes": 532618028,
        "normalized_tree_commitment_sha256": "34014b16541c10bc3eccfbdfa18255e346be2c36cf70bb3defd0c9b04f2d07af",
        "normalized_tree_commitment_method": "SHA256 of canonical JSON sorted list of relative_path SHA256 and bytes for every regular tar member after removing leading dot-slash; tar metadata excluded",
        "identity_rule": "this rebuilt tar is the sole active byte identity for attempt 2 and later health/scientific runs; the historical tar identity remains preserved for prior provenance",
        "verification": "require exact byte count and SHA-256 before scratch extraction",
    }
    historical = config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_visor_hos_correction_amendment"
    ]["resource_and_governance"]["language_dependency_archive"]
    assert historical["sha256"] == (
        "27df87f4ec1900b4a11f307d42a18483903d38ddb9ed77f418b88fa299497e37"
    )
    assert restore["remaining_health_budget"]["submission_count_remaining"] == 2
    assert restore["remaining_health_budget"][
        "conservative_protocol_GPU_hours_remaining"
    ] == 0.5

    mutated = json.loads(json.dumps(config))
    repair = mutated["learner_effective_engineering_health_dependency_restore"]
    repair["active_language_dependency_archive"]["sha256"] = "0" * 64
    payload = json.loads(json.dumps(repair))
    payload.pop("repair_commitment_sha256")
    repair["repair_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_DEPENDENCY_RESTORE_COMMITMENT"
    ):
        MODULE._engineering_health_dependency_restore(mutated)


def test_topology_guard_repair_is_hash_bound_preinference_and_scientifically_narrow() -> None:
    config = _config()
    repair = MODULE._engineering_health_topology_guard_repair(config)
    assert repair["repair_commitment_sha256"] == (
        "8db2d8ae04ee702ab3c68ff7c243afed0d8e4710c01f3f7865faa4975fb9a5b8"
    )
    assert repair["triggering_attempt"]["classification"] == (
        "ENGINEERING_WRAPPER_TOPOLOGY_GUARD_FAILURE_NOT_SCIENTIFIC_NO_GO"
    )
    assert repair["triggering_attempt"]["model_module_inference_count"] == 0
    assert repair["triggering_attempt"]["scientific_metric_count"] == 0
    assert repair["aggregate_read_only_diagnosis"][
        "authoritative_scontrol_predicate_pass_count"
    ] == 7
    assert repair["repair"]["health_resource_topology_changed"] is False
    assert repair["repair"]["model_fixture_threshold_metric_seed_or_gate_changed"] is False
    assert repair["remaining_health_budget"]["submission_count_remaining"] == 1
    assert repair["remaining_health_budget"][
        "conservative_protocol_GPU_hours_remaining"
    ] == 0.25

    mutated = json.loads(json.dumps(config))
    value = mutated["learner_effective_engineering_health_topology_guard_repair"]
    value["repair"]["health_resource_topology_changed"] = True
    payload = json.loads(json.dumps(value))
    payload.pop("repair_commitment_sha256")
    value["repair_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_TOPOLOGY_GUARD_REPAIR_COMMITMENT"
    ):
        MODULE._engineering_health_topology_guard_repair(mutated)


def test_terminal_blocker_is_preserved_and_reauthorization_is_hash_bound(
    tmp_path: Path,
) -> None:
    config = _config()
    result = MODULE._engineering_health_terminal_result(config)
    assert result["blocker_commitment_sha256"] == (
        "644028babc768e881276fa078b95349ba77f8418cb76d722e8baf2588f9d0f81"
    )
    assert result["final_attempt"]["job_id"] == 316370
    assert result["compact_aggregate"]["status"] == "ENGINEERING_BLOCKER"
    assert result["compact_aggregate"]["scientific_metric_count"] == 0
    assert result["stable_aggregate_diagnosis"]["unaccounted_exception_type"] == (
        "FileNotFoundError"
    )
    assert result["resource_accounting"]["submission_count_remaining"] == 0
    assert result["terminal_gate"]["attempt_4_authorized"] is False

    reauthorization = MODULE._engineering_health_reauthorization(config)
    assert reauthorization["reauthorization_commitment_sha256"] == (
        "3271499c19a77ffab8c53e2cd2052ea14514682a3d4b73fd7c5179ceec4a7ff4"
    )
    assert reauthorization["effective_resource_policy"]["reauthorized_attempt"] == 4
    assert reauthorization["effective_resource_policy"][
        "reauthorized_submission_count"
    ] == 1
    assert reauthorization["failure_specific_repair"][
        "runner_scontrol_subprocess_inside_container"
    ] is False

    terminal_reauthorization = (
        MODULE._engineering_health_reauthorization_result(config)
    )
    assert terminal_reauthorization["blocker_commitment_sha256"] == (
        "59b1778b35cedd1cb020177e41fe6887371a5480f7ee6bf6e57f55d4c90edde3"
    )
    assert terminal_reauthorization["submission_provenance"]["job_id"] == 316478
    assert terminal_reauthorization["submission_provenance"][
        "model_module_inference_count"
    ] == 0
    assert terminal_reauthorization["stable_aggregate_diagnosis"][
        "argparse_invalid_choice"
    ] is True
    assert terminal_reauthorization["terminal_gate"]["attempt_5_authorized"] is False

    parser_repair = MODULE._engineering_health_parser_repair_reauthorization(
        config
    )
    assert parser_repair["reauthorization_commitment_sha256"] == (
        "d9cf3feaa0f5c4d65978ca796b722f31e75ce1078b918d114ca35b298a148c8b"
    )
    assert parser_repair["failure_specific_repair"]["new_parser_choices"] == [
        1,
        2,
        3,
        5,
    ]
    assert parser_repair["effective_resource_policy"]["reauthorized_attempt"] == 5
    assert parser_repair["execution_and_stop_rule"][
        "repair_or_resmoke_cycles_after_attempt_5"
    ] == 0

    terminal_parser_repair = MODULE._engineering_health_parser_repair_result(
        config
    )
    assert terminal_parser_repair["blocker_commitment_sha256"] == (
        "b05dc8da3155561b182b3bcfa50c851f83828b34e063918306bdfb57fdedeb9c"
    )
    assert terminal_parser_repair["submission_provenance"]["job_id"] == 316537
    assert terminal_parser_repair["compact_aggregate"][
        "completed_module_count"
    ] == 0
    assert terminal_parser_repair["compact_aggregate"][
        "scientific_metric_count"
    ] == 0
    assert terminal_parser_repair["stable_aggregate_diagnosis"][
        "first_stable_error_code"
    ] == "E_TUPLE_HEALTH_ARTIFACT_COMMITMENT"
    assert terminal_parser_repair["terminal_gate"]["attempt_6_authorized"] is False

    iterative = MODULE._engineering_health_iterative_reauthorization(config)
    assert iterative["reauthorization_commitment_sha256"] == (
        "3114e1763f65dbeb8b2f89bb2a0480c86f4266f888c1ac2ff740bee85d357ab9"
    )
    assert iterative["read_only_root_cause_diagnosis"][
        "host_entry_is_symlink"
    ] is True
    assert iterative["read_only_root_cause_diagnosis"][
        "running_container_namespace_symlink_target_present"
    ] is False
    assert iterative["active_attempt_resource_policy"]["attempt"] == 6
    assert iterative["rolling_execution_policy"][
        "blanket_user_authorization_for_additional_ordinary_engineering_attempts"
    ] is True

    attempt_6 = MODULE._engineering_health_iterative_attempt_6_result(config)
    assert attempt_6["blocker_commitment_sha256"] == (
        "e559cd535d2a6dd833d2588c75b180754260dd3ff68ea9f6731a0e4478a6d114"
    )
    assert attempt_6["submission_provenance"]["job_id"] == 316604
    assert attempt_6["compact_aggregate"]["scientific_metric_count"] == 0
    assert attempt_6["stable_aggregate_diagnosis"][
        "exact_in_container_stage_identified"
    ] is False

    progress = MODULE._engineering_health_progress_repair(config)
    assert progress["reauthorization_commitment_sha256"] == (
        "a2d1347bef14848a5238f9a10c6e94da8eaa68593aa3d97e8a460dfbf8694d07"
    )
    assert progress["active_attempt_resource_policy"]["attempt"] == 7
    assert progress["failure_specific_repair"][
        "progress_record_used_for_scientific_metrics_or_selection"
    ] is False

    attempt_7 = MODULE._engineering_health_attempt_7_result(config)
    assert attempt_7["blocker_commitment_sha256"] == (
        "03c09a61cedb29e04cf465287db693cd1248c53d45d9a7a47a777e6cdf1d594d"
    )
    assert attempt_7["submission_provenance"]["job_id"] == 316641
    assert attempt_7["stable_aggregate_diagnosis"][
        "last_completed_dependency_stage"
    ] == "DEPENDENCY_ACTION_WEIGHT"
    assert attempt_7["compact_aggregate"]["scientific_metric_count"] == 0

    extended = MODULE._engineering_health_extended_wall_repair(config)
    assert extended["reauthorization_commitment_sha256"] == (
        "d2db51229719a0e64f84da9541a284d88c75a2fd32e2e186ba34e17ab5eed6e7"
    )
    assert extended["active_attempt_resource_policy"]["attempt"] == 8
    assert extended["active_attempt_resource_policy"]["wall_minutes_max"] == 60
    assert extended["failure_specific_repair"][
        "model_fixture_source_threshold_partition_seed_metric_or_gate_changed"
    ] is False
    effective = MODULE._engineering_health_resource_policy(config)
    assert effective["per_submission_wall_minutes_max"] == 60
    assert effective["initial_plus_repair_resmoke_submission_count_max"] == 8

    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_EXTENDED_WALL_REPAIR_ATTEMPT",
    ):
        MODULE.run_tuple_health(
            argparse.Namespace(
                public_root=tmp_path / "public",
                scratch_root=tmp_path / "scratch",
                config=Path("configs/synthetic_video_real_only_proof.json"),
                container_attestation=tmp_path / "container-attestation.json",
                attempt=3,
                device="cuda",
            )
        )

    mutated = json.loads(json.dumps(config))
    mutated["learner_effective_engineering_health_result"]["terminal_gate"][
        "attempt_4_authorized"
    ] = True
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_TERMINAL_RESULT_COMMITMENT"
    ):
        MODULE._engineering_health_terminal_result(mutated)

    mutated = json.loads(json.dumps(config))
    amended = mutated["learner_effective_engineering_health_reauthorization"]
    amended["effective_resource_policy"]["reauthorized_submission_count"] = 2
    payload = json.loads(json.dumps(amended))
    payload.pop("reauthorization_commitment_sha256")
    amended["reauthorization_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_REAUTHORIZATION_COMMITMENT"
    ):
        MODULE._engineering_health_reauthorization(mutated)

    mutated = json.loads(json.dumps(config))
    result = mutated[
        "learner_effective_engineering_health_reauthorization_result"
    ]
    result["terminal_gate"]["attempt_5_authorized"] = True
    payload = json.loads(json.dumps(result))
    payload.pop("blocker_commitment_sha256")
    result["blocker_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_REAUTHORIZATION_RESULT_COMMITMENT",
    ):
        MODULE._engineering_health_reauthorization_result(mutated)

    mutated = json.loads(json.dumps(config))
    parser_repair = mutated[
        "learner_effective_engineering_health_parser_repair_reauthorization"
    ]
    parser_repair["failure_specific_repair"]["attempt_4_remains_rejected_and_sealed"] = False
    payload = json.loads(json.dumps(parser_repair))
    payload.pop("reauthorization_commitment_sha256")
    parser_repair["reauthorization_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_PARSER_REPAIR_REAUTHORIZATION_COMMITMENT",
    ):
        MODULE._engineering_health_parser_repair_reauthorization(mutated)

    mutated = json.loads(json.dumps(config))
    result = mutated["learner_effective_engineering_health_parser_repair_result"]
    result["terminal_gate"]["attempt_6_authorized"] = True
    payload = json.loads(json.dumps(result))
    payload.pop("blocker_commitment_sha256")
    result["blocker_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_PARSER_REPAIR_RESULT_COMMITMENT",
    ):
        MODULE._engineering_health_parser_repair_result(mutated)

    mutated = json.loads(json.dumps(config))
    iterative = mutated[
        "learner_effective_engineering_health_iterative_reauthorization"
    ]
    iterative["failure_specific_repair"][
        "runner_no_longer_dereferences_host_only_SIF_target_inside_container"
    ] = False
    payload = json.loads(json.dumps(iterative))
    payload.pop("reauthorization_commitment_sha256")
    iterative["reauthorization_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_ITERATIVE_REAUTHORIZATION_COMMITMENT",
    ):
        MODULE._engineering_health_iterative_reauthorization(mutated)

    mutated = json.loads(json.dumps(config))
    result = mutated["learner_effective_engineering_health_iterative_attempt_6_result"]
    result["compact_aggregate"]["scientific_metric_count"] = 1
    payload = json.loads(json.dumps(result))
    payload.pop("blocker_commitment_sha256")
    result["blocker_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_ITERATIVE_ATTEMPT_6_RESULT_COMMITMENT",
    ):
        MODULE._engineering_health_iterative_attempt_6_result(mutated)

    mutated = json.loads(json.dumps(config))
    progress = mutated["learner_effective_engineering_health_progress_repair"]
    progress["failure_specific_repair"]["stable_stage_codes_only"] = False
    payload = json.loads(json.dumps(progress))
    payload.pop("reauthorization_commitment_sha256")
    progress["reauthorization_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_PROGRESS_REPAIR_COMMITMENT",
    ):
        MODULE._engineering_health_progress_repair(mutated)

    mutated = json.loads(json.dumps(config))
    result = mutated["learner_effective_engineering_health_attempt_7_result"]
    result["compact_aggregate"]["scientific_metric_count"] = 1
    payload = json.loads(json.dumps(result))
    payload.pop("blocker_commitment_sha256")
    result["blocker_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_ATTEMPT_7_RESULT_COMMITMENT",
    ):
        MODULE._engineering_health_attempt_7_result(mutated)

    mutated = json.loads(json.dumps(config))
    repair = mutated["learner_effective_engineering_health_extended_wall_repair"]
    repair["failure_specific_repair"]["same_exact_artifact_rehash_and_tree_validation"] = False
    payload = json.loads(json.dumps(repair))
    payload.pop("reauthorization_commitment_sha256")
    repair["reauthorization_commitment_sha256"] = MODULE.digest(payload)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_EXTENDED_WALL_REPAIR_COMMITMENT",
    ):
        MODULE._engineering_health_extended_wall_repair(mutated)


def test_h100_health_topology_uses_wrapper_attestation_and_effective_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    h100_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: "NVIDIA H100 NVL MIG 3g.47gb",
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=47 * 1024**3
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=h100_cuda))
    attestation = _write_topology_attestation(tmp_path, monkeypatch)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scontrol subprocess must not run inside the container")
        ),
    )
    commitment = MODULE._tuple_health_topology(
        "cuda", topology_attestation=attestation, attempt=4
    )
    assert commitment == MODULE.file_digest(attestation)
    attempt_8_attestation = _write_topology_attestation(
        tmp_path / "attempt-08", monkeypatch, attempt=8, job_id=316700
    )
    commitment = MODULE._tuple_health_topology(
        "cuda", topology_attestation=attempt_8_attestation, attempt=8
    )
    assert commitment == MODULE.file_digest(attempt_8_attestation)

    bad = json.loads(attestation.read_text())
    bad["task_count"] = 2
    attestation.write_bytes(MODULE.canonical(bad) + b"\n")
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_TOPOLOGY_ATTESTATION"):
        MODULE._tuple_health_topology(
            "cuda", topology_attestation=attestation, attempt=4
        )


def test_container_attestation_replaces_unavailable_in_container_sif_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = _write_container_attestation(
        tmp_path / "health", monkeypatch, attempt=6, job_id=316538
    )
    assert MODULE._tuple_container_attestation(
        health,
        _config(),
        run_mode="health",
        attempt=6,
    ) == {
        "sha256": "f274f1ac3726376b762b557ff9a07203b2d42aac3157a7a354b998e589c35792",
        "bytes": 3731320832,
    }

    development = _write_container_attestation(
        tmp_path / "development",
        monkeypatch,
        run_mode="development",
        attempt=None,
        job_id=316539,
    )
    assert MODULE._tuple_container_attestation(
        development,
        _config(),
        run_mode="development",
        attempt=None,
    )["bytes"] == 3731320832

    original = json.loads(health.read_text())
    monkeypatch.setenv("SLURM_JOB_ID", "316538")
    for key, changed_value in (
        ("job_id", 316537),
        ("run_mode", "development"),
        ("attempt", 5),
        ("predicate_pass_count", 3),
        ("sha256", "0" * 64),
        ("bytes", 1),
    ):
        changed = dict(original)
        changed[key] = changed_value
        health.write_bytes(MODULE.canonical(changed) + b"\n")
        with pytest.raises(
            RuntimeError,
            match="E_TUPLE_HEALTH_CONTAINER_ATTESTATION",
        ):
            MODULE._tuple_container_attestation(
                health,
                _config(),
                run_mode="health",
                attempt=6,
            )


def test_health_and_science_configuration_commitments_ignore_outcome_state() -> None:
    health_config = _config()
    health_commitment = MODULE._tuple_health_configuration_preflight(health_config)
    science_config = json.loads(json.dumps(health_config))
    science_config["status"] = (
        "COMMITTED_ENGINEERING_HEALTH_PASS_PENDING_SCIENTIFIC_DEVELOPMENT"
    )
    science_config["learner_effective_engineering_health_pass_result"] = {
        "outcome_only_test_record": True
    }
    assert (
        MODULE._tuple_health_configuration_preflight(science_config)
        == health_commitment
    )


def test_progress_record_is_private_aggregate_only_and_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "engineering-progress.json"
    MODULE._write_tuple_health_progress(
        path,
        attempt=7,
        stage="DEPENDENCY_CODE_TREE",
        stage_ordinal=5,
        module_ordinal=0,
        replicate=0,
        update_count=4,
        submission_started_epoch=__import__("time").time() - 1.0,
    )
    value = json.loads(path.read_text())
    assert path.stat().st_mode & 0o777 == 0o600
    assert value["attempt"] == 7
    assert value["stage"] == "DEPENDENCY_CODE_TREE"
    assert value["scientific_metric_count"] == 0
    assert value["sensitive_detail_field_count"] == 0
    assert not any(
        token in key.casefold()
        for key in value
        for token in ("path", "filename", "hash", "prompt", "prediction", "label")
    )
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_PROGRESS_RECORD"):
        MODULE._write_tuple_health_progress(
            path,
            attempt=7,
            stage="UNDECLARED_STAGE",
            stage_ordinal=5,
            module_ordinal=0,
            replicate=0,
            update_count=5,
            submission_started_epoch=__import__("time").time(),
        )


def test_tuple_topology_separates_h100_health_from_a30_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_GPUS_ON_NODE": "1",
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
    }.items():
        monkeypatch.setenv(key, value)

    h100_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: "NVIDIA H100 NVL",
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=47 * 1024**3
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=h100_cuda))
    monkeypatch.setenv("SLURM_JOB_PARTITION", "h100")
    attestation = _write_topology_attestation(tmp_path, monkeypatch)
    MODULE._tuple_health_topology(
        "cuda", topology_attestation=attestation, attempt=4
    )
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_GPU_TOPOLOGY"):
        MODULE._tuple_health_topology("cuda", False)

    a30_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: "NVIDIA A30",
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=24 * 1024**3
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=a30_cuda))
    monkeypatch.setenv("SLURM_JOB_PARTITION", "a30")
    MODULE._tuple_health_topology("cuda", False)
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_TOPOLOGY_ATTESTATION"):
        MODULE._tuple_health_topology("cuda")


def test_tuple_health_trace_root_is_private_even_with_permissive_umask(
    tmp_path: Path,
) -> None:
    previous = os.umask(0)
    try:
        MODULE._tuple_health_error(
            "sensor", RuntimeError("E_SENSOR_DECODE private"), tmp_path
        )
    finally:
        os.umask(previous)
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in tmp_path.iterdir())


def test_tuple_health_cli_emits_one_exact_allowlisted_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    compact = MODULE._tuple_health_compact(_healthy_full())
    observed_attempts: list[int] = []

    def run_health(args: argparse.Namespace) -> dict:
        observed_attempts.append(args.attempt)
        return compact

    monkeypatch.setattr(MODULE, "run_tuple_health", run_health)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_synthetic_video_calibration.py",
            "tuple-health",
            "--public-root",
            str(tmp_path / "public"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--config",
            str(tmp_path / "config.json"),
            "--container-attestation",
            str(tmp_path / "container-attestation.json"),
            "--attempt",
            "7",
            "--device",
            "cuda",
        ],
    )
    MODULE.main()
    lines = capsys.readouterr().out.splitlines()
    assert observed_attempts == [7]
    assert len(lines) == 1
    assert set(json.loads(lines[0])) == set(MODULE.TUPLE_HEALTH_FIELDS)


def test_attempt4_terminal_diagnosis_matches_frozen_cli_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_synthetic_video_calibration.py",
            "tuple-health",
            "--public-root",
            str(tmp_path / "public"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--config",
            "configs/synthetic_video_real_only_proof.json",
            "--container-attestation",
            str(tmp_path / "container-attestation.json"),
            "--attempt",
            "4",
            "--device",
            "cuda",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        MODULE.main()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid choice: '4'" in captured.err


def test_output_guard_rejects_unignored_path_in_another_git_repository(
    tmp_path: Path,
) -> None:
    import subprocess

    repository = tmp_path / "other-repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_FIXTURE_OUTPUT_NOT_EXTERNAL_OR_IGNORED",
    ):
        MODULE._require_external_or_ignored_output(repository / "runs")


def test_partition_registry_failure_is_compact_and_withholds_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_tuple_module_runners",
        lambda: (_ for _ in ()).throw(
            RuntimeError("E_TUPLE_QUALIFICATION_MODULE_REGISTRY")
        ),
    )
    results = MODULE._tuple_partition_engineering_integrity(
        {"scratch_root": tmp_path / "scratch"},
        "development",
        tmp_path / "traces",
    )
    assert len(results) == 7
    assert all(result["status"] == "ERROR" for result in results)
    assert not any("metric" in key.casefold() for result in results for key in result)


def test_health_orchestration_never_calls_scientific_release_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = tmp_path / "public"
    scratch = tmp_path / "scratch"
    historical_config = json.loads(
        Path("configs/synthetic_video_real_only_proof.json").read_text()
    )
    historical_config["schema_version"] = 21
    historical_config.pop("learner_effective_engineering_health_result")
    historical_config.pop("learner_effective_engineering_health_reauthorization")
    historical_config.pop("learner_effective_engineering_health_reauthorization_result")
    historical_config.pop("learner_effective_engineering_health_parser_repair_reauthorization")
    historical_config.pop("learner_effective_engineering_health_parser_repair_result")
    historical_config.pop("learner_effective_engineering_health_iterative_reauthorization")
    historical_config.pop("learner_effective_engineering_health_iterative_attempt_6_result")
    historical_config.pop("learner_effective_engineering_health_progress_repair")
    historical_config.pop("learner_effective_engineering_health_attempt_7_result")
    historical_config.pop("learner_effective_engineering_health_extended_wall_repair")
    config_path = tmp_path / "preterminal-proof.json"
    MODULE.write_private_new(config_path, historical_config)
    manifest = _fixture_manifest()
    attempt_root = MODULE._tuple_health_root(public) / "health/attempt-01"
    attempt_root.mkdir(parents=True, mode=0o700)
    MODULE.write_private_new(
        attempt_root / "wrapper-started.json",
        {
            "schema_version": 1,
            "attempt": 1,
            "submission_started_epoch": __import__("time").time(),
            "GPU_type": "NVIDIA_H100_NVL_3G_47GB_MIG",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
        },
    )
    _write_topology_attestation(
        attempt_root, monkeypatch, attempt=1, job_id=316325
    )
    container_attestation = _write_container_attestation(
        attempt_root, monkeypatch, attempt=1, job_id=316325
    )
    calls: list[str] = []

    def runner(module_id: str):
        def execute(_context):
            calls.append(module_id)
            return MODULE._tuple_health_pass_result(
                module_id, 4, {"module_id": module_id, "stable": True}
            )

        return execute

    def forbidden(*_args, **_kwargs):
        raise AssertionError("scientific helper opened during engineering health")

    for name in (
        "_tuple_frozen_threshold_grids",
        "_tuple_selected_thresholds",
        "_tuple_axis_results_from_modules",
        "_tuple_combined_public_gate",
        "_tuple_qualification_integrity",
    ):
        monkeypatch.setattr(MODULE, name, forbidden)
    monkeypatch.setattr(
        MODULE, "_tuple_health_topology", lambda *_args, **_kwargs: "a" * 64
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_health_dependency_preflight",
        lambda *_: {"dependency_config_commitment_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        MODULE,
        "_verify_tuple_fixture_manifest",
        lambda *_: (manifest, public / "fixtures"),
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_module_runners",
        lambda: {module_id: runner(module_id) for module_id in MODULE_IDS},
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("WANDB_DISABLED", "true")

    compact = MODULE.run_tuple_health(
        argparse.Namespace(
            public_root=public,
            scratch_root=scratch,
            config=config_path,
            container_attestation=container_attestation,
            attempt=1,
            device="cuda",
        )
    )
    assert compact["status"] == "PASS_ENGINEERING_HEALTH"
    assert compact["scientific_metric_count"] == 0
    assert calls == [module_id for module_id in MODULE_IDS for _ in range(2)]


def test_partition_crash_withholds_metrics_and_preserves_legacy_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = tmp_path / "public"
    scratch = tmp_path / "scratch"
    config_path = Path("configs/synthetic_video_real_only_proof.json").resolve()
    config = json.loads(config_path.read_text())
    correction = MODULE._tuple_visor_hos_correction_amendment(config)
    manifest = _fixture_manifest()
    manifest.update(
        {
            "visor_hos_correction_amendment_commitment_sha256": correction[
                "amendment_commitment_sha256"
            ],
            "verified_no_hand_seal_commitment_sha256": "a" * 64,
        }
    )
    legacy = MODULE._tuple_legacy_qualification_paths(public)[
        "development_result"
    ]
    MODULE.write_private_new(legacy, {"sealed_prior_job": 315542})
    legacy_bytes = legacy.read_bytes()
    legacy_mtime = legacy.stat().st_mtime_ns
    family_by_module = {
        "adapter_and_lexical": "language_lexical",
        "referent": "referent_attribute",
        "recurrence": "recurrence",
        "attribute": "referent_attribute",
        "hand_contact": "hand_contact",
        "sensor": "sensor",
        "order_action": "order_action",
    }

    def runner(module_id: str):
        def execute(context):
            if module_id == "referent":
                raise RuntimeError("E_TUPLE_REFERENT_DECODE")
            count = len(context["rows"][family_by_module[module_id]])
            return MODULE._tuple_health_pass_result(
                module_id, count, {"module_id": module_id, "valid": True}
            )

        return execute

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setattr(MODULE, "_tuple_health_topology", lambda *_: None)
    monkeypatch.setattr(MODULE, "_verify_tuple_runtime_manifest", lambda *_: {})
    monkeypatch.setattr(
        MODULE,
        "_verify_tuple_fixture_manifest",
        lambda *_: (manifest, public / "fixtures"),
    )
    monkeypatch.setattr(
        MODULE,
        "_load_tuple_health_pass",
        lambda *_: {
            "engineering_health_commitment_sha256": "c" * 64,
            "dependency_config_commitment_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_health_dependency_preflight",
        lambda *_: {"dependency_config_commitment_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        MODULE,
        "_tuple_module_runners",
        lambda: {module_id: runner(module_id) for module_id in MODULE_IDS},
    )
    container_attestation = _write_container_attestation(
        tmp_path / "development-container-attestation",
        monkeypatch,
        run_mode="development",
        attempt=None,
        job_id=316539,
    )

    compact = MODULE.qualify_tuple_public(
        argparse.Namespace(
            public_root=public,
            scratch_root=scratch,
            config=config_path,
            container_attestation=container_attestation,
            partition="development",
            device="cuda",
        )
    )
    assert compact["status"] == "ENGINEERING_BLOCKER"
    assert compact["scientific_metric_count"] == 0
    assert compact["failed_module_count"] == 1
    paths = MODULE._tuple_qualification_paths(public)
    assert not paths["development_result"].exists()
    assert not paths["development_threshold_seal"].exists()
    assert legacy.read_bytes() == legacy_bytes
    assert legacy.stat().st_mtime_ns == legacy_mtime
    trace = (
        MODULE._tuple_qualification_root(public)
        / "engineering-integrity/development-attempt-01-traces/referent-01.trace"
    )
    assert trace.is_file()
    assert trace.stat().st_mode & 0o777 == 0o600


def test_partition_integrity_rejects_metric_bearing_pass_record() -> None:
    row = _healthy_modules()[0]
    row["metrics"] = {"macro_f1": 1.0}
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_module_result(row, expected_case_count=4)
