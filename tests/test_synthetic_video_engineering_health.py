from __future__ import annotations

import importlib.util
import argparse
import json
import os
import re
import subprocess
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


def test_git_free_clean_tree_attestation_detects_modified_or_untracked_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    tracked = repository / "model.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repository), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Codex Test",
            "-c",
            "user.email=codex@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    tree = MODULE._tuple_health_tree_commitment(repository)
    index = repository / ".git/index"
    attestation = {
        "family": "test_code_tree",
        "expected_commit": commit,
        "git_index_sha256": MODULE.file_digest(index),
        "git_index_bytes": index.stat().st_size,
        **tree,
        "host_unexpected_status_count": 0,
    }
    monkeypatch.setattr(
        MODULE.shutil,
        "which",
        lambda name: None if name == "git" else __import__("shutil").which(name),
    )
    MODULE._tuple_health_verify_git_tree(repository, commit, attestation)

    index_digest = MODULE.file_digest(index)
    monkeypatch.setattr(MODULE.shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("attested verification must not invoke Git")
        ),
    )
    MODULE._tuple_health_verify_git_tree(repository, commit, attestation)
    assert MODULE.file_digest(index) == index_digest

    tracked.write_text("VALUE = 2\n")
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_CODE_TREE_ATTESTATION"
    ):
        MODULE._tuple_health_verify_git_tree(repository, commit, attestation)

    tracked.write_text("VALUE = 1\n")
    (repository / "untracked.py").write_text("VALUE = 3\n")
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_CODE_TREE_ATTESTATION"
    ):
        MODULE._tuple_health_verify_git_tree(repository, commit, attestation)

    (repository / "untracked.py").unlink()
    (repository / "untracked-link.py").symlink_to(tracked)
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_CODE_TREE_ATTESTATION"
    ):
        MODULE._tuple_health_verify_git_tree(repository, commit, attestation)


def test_portable_geometry_ast_is_hash_bound_and_ignores_unrelated_code(
    tmp_path: Path,
) -> None:
    config = _config()
    compatibility = config["learner_effective_engineering_health_amendment"][
        "historical_geometry_compatibility"
    ]
    source_path = Path("scripts/run_synthetic_video_calibration.py")
    source_sha256, portable_ast_sha256 = (
        MODULE._portable_geometry_function_bundle_digests(
            source_path, compatibility["function_names"]
        )
    )
    assert source_sha256 == compatibility["exact_function_source_bundle_sha256"]
    assert portable_ast_sha256 == (
        "74d7707cd8d485aa00514f7f216759f848893923e990f94e3a1d19b935958b8d"
    )

    copy = tmp_path / "runner.py"
    copy.write_text(source_path.read_text() + "\ndef unrelated_fixture():\n    return 1\n")
    assert MODULE._portable_geometry_function_bundle_digests(
        copy, compatibility["function_names"]
    ) == (source_sha256, portable_ast_sha256)


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
    config.pop("learner_effective_engineering_health_scheduler_policy")
    config.pop("learner_effective_engineering_health_attempt_8_result")
    config.pop("learner_effective_engineering_health_git_fallback_repair")
    config.pop("learner_effective_engineering_health_attempt_9_result")
    config.pop("learner_effective_engineering_health_historical_lineage_repair")
    config.pop("learner_effective_engineering_health_attempt_10_result")
    config.pop("learner_effective_engineering_health_portable_AST_repair")
    config.pop("learner_effective_engineering_health_attempt_11_result")
    config.pop("learner_effective_engineering_health_fixture_bind_repair")
    config.pop("learner_effective_engineering_health_attempt_12_result")
    config.pop("learner_effective_engineering_health_submission_export_repair")
    config.pop("learner_effective_engineering_health_attempt_13_result")
    config.pop("learner_effective_engineering_health_NLTK_matplotlib_repair")
    config.pop("learner_effective_engineering_health_attempt_14_result")
    config.pop(
        "learner_effective_engineering_health_generic_GRES_serialization_repair"
    )
    config.pop("learner_effective_engineering_health_attempt_15_result")
    config.pop("learner_effective_engineering_health_active_dispatch_repair")
    config.pop("learner_effective_engineering_health_attempt_16_result")
    config.pop(
        "learner_effective_engineering_health_historical_resource_dispatch_repair"
    )
    config.pop("learner_effective_engineering_health_attempt_17_result")
    config.pop(
        "learner_effective_engineering_health_ffmpeg_prettytable_repair"
    )
    config.pop("learner_effective_engineering_health_attempt_18_result")
    config.pop(
        "learner_effective_engineering_health_historical_full_result_lineage_repair"
    )
    config.pop("learner_effective_engineering_health_attempt_19_result")
    config.pop(
        "learner_effective_engineering_health_grounding_state_compatibility_repair"
    )
    config.pop("learner_effective_engineering_health_pass_result")
    return config


def _write_topology_attestation(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    attempt: int = 4,
    job_id: int = 316371,
    gpu_type: str = "NVIDIA_H100_NVL_3G_47GB_MIG",
) -> Path:
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    if gpu_type == "NVIDIA_A30_24GB":
        partition = "a30"
        gres = "gpu:nvidia_a30:1"
    elif gpu_type == "NVIDIA_H100_NVL":
        partition = "h100"
        gres = "gpu:nvidia_h100_nvl:1"
    elif gpu_type == "NVIDIA_H200_NVL":
        partition = "h200"
        gres = "gpu:nvidia_h200_nvl:1"
    else:
        partition = "h100"
        gres = "gpu:nvidia_h100_nvl_3g.47gb:1"
    value = {
        "schema_version": 1,
        "attempt": attempt,
        "job_id": job_id,
        "partition": partition,
        "node_count": 1,
        "CPU_count": 8,
        "task_count": 1,
        "time_limit_minutes": 60 if attempt in {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} else 15,
        "memory_per_CPU_GiB": 4,
        "GRES": gres,
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


def _write_fixture_bind_attestation(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_mode: str,
    attempt: int | None,
    job_id: int,
) -> Path:
    repair = MODULE._engineering_health_fixture_bind_repair(_config())[
        "failure_specific_repair"
    ]
    source = repair["source_record_file"]
    no_hand = repair["verified_no_hand_seal_file"]
    manifest = repair["fixture_manifest_file"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "fixture-bind-attestation.json"
    MODULE.write_private_new(
        path,
        {
            "artifact_family": "SEALED_PUBLIC_FIXTURE_BIND",
            "attempt": attempt,
            "fixture_manifest_bytes": manifest["bytes"],
            "fixture_manifest_file_sha256": manifest["sha256"],
            "job_id": job_id,
            "path_field_count": 0,
            "predicate_count": 9,
            "predicate_pass_count": 9,
            "read_only": True,
            "run_mode": run_mode,
            "schema_version": 1,
            "source": "WRAPPER_HOST_BEFORE_CONTAINER",
            "source_bound_at_original_absolute_alias": True,
            "source_bound_over_active_fixture_target": True,
            "source_record_bytes": source["bytes"],
            "source_record_file_sha256": source["sha256"],
            "verified_no_hand_seal_bytes": no_hand["bytes"],
            "verified_no_hand_seal_file_sha256": no_hand["sha256"],
        },
    )
    monkeypatch.setenv("SLURM_JOB_ID", str(job_id))
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
    visor = MODULE._tuple_health_error(
        "adapter_and_lexical",
        RuntimeError("E_VISOR_HOS_SOURCE_FEASIBILITY_MISSING private-detail"),
        tmp_path,
    )
    assert visor["error_code"] == (
        "E_TUPLE_HEALTH_ADAPTER_AND_LEXICAL_VISOR_HOS_SOURCE_FEASIBILITY_MISSING"
    )
    assert "private-detail" not in json.dumps(visor)
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


def test_historical_attempt_8_full_record_requires_both_sealed_commitments() -> None:
    full = _healthy_full()
    full["attempt"] = 8
    full["config_commitment_sha256"] = (
        "490f63e294d4751a962d41c1956b02e1ac51bafc8a42aed2f76aa8c8704b0c89"
    )
    full["engineering_health_commitment_sha256"] = (
        "5df1d3e169f17ecbd269e7dd8c29fc0b033a301fb6cd15e69eebecccdceb92c8"
    )
    full["resource"]["GPU_type"] = "NVIDIA_H100_NVL"
    full["cumulative_resource"]["cumulative_submission_count"] = 8
    MODULE._validate_tuple_health_full(full, _config())

    wrong_config = json.loads(json.dumps(full))
    wrong_config["config_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_config, _config())

    wrong_health = json.loads(json.dumps(full))
    wrong_health["engineering_health_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_health, _config())


def test_historical_attempt_10_full_record_requires_both_sealed_commitments() -> None:
    full = _healthy_full()
    full["attempt"] = 10
    full["config_commitment_sha256"] = (
        "ca100af8a99c93941d15c08f6b93ea0530869217d4d1f2cf4fdf682a7d708a5d"
    )
    full["engineering_health_commitment_sha256"] = (
        "21f1044fd4a8a7b05f1b66d9082cfbe38a271bacfe06691cfde416628b0237d8"
    )
    full["resource"]["GPU_type"] = "NVIDIA_A30_24GB"
    full["cumulative_resource"]["cumulative_submission_count"] = 10
    MODULE._validate_tuple_health_full(full, _config())

    wrong_config = json.loads(json.dumps(full))
    wrong_config["config_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_config, _config())

    wrong_health = json.loads(json.dumps(full))
    wrong_health["engineering_health_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_health, _config())


def test_historical_attempt_11_full_record_requires_both_sealed_commitments() -> None:
    full = _healthy_full()
    full["attempt"] = 11
    full["config_commitment_sha256"] = (
        "ecb1647874791878a584e143e657b66f250d4a1df842538e0223b3d7b833aa55"
    )
    full["engineering_health_commitment_sha256"] = (
        "2e409529a3de9a1536915e776f14d8ffa8e9f35aec88929d54c0a46d8b5debc4"
    )
    full["resource"]["GPU_type"] = "NVIDIA_A30_24GB"
    full["cumulative_resource"]["cumulative_submission_count"] = 11
    MODULE._validate_tuple_health_full(full, _config())

    wrong_config = json.loads(json.dumps(full))
    wrong_config["config_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_config, _config())

    wrong_health = json.loads(json.dumps(full))
    wrong_health["engineering_health_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_health, _config())


def test_historical_attempt_13_full_record_requires_both_sealed_commitments() -> None:
    full = _healthy_full()
    full["status"] = "ENGINEERING_BLOCKER"
    full["attempt"] = 13
    full["config_commitment_sha256"] = (
        "faaedf74ec8ca6d3f4887174e6def45030be25e96c0eca8336e4c24b70055c6c"
    )
    full["engineering_health_commitment_sha256"] = (
        "abc903036dceb88376f6bd9ae50d8bfa263b3fc9791d5dd3d5eb6cad5f9cbe70"
    )
    for index in range(3, 7):
        module_id = full["module_results"][index]["module_id"]
        error_code = (
            f"E_TUPLE_HEALTH_{module_id.upper()}_UNACCOUNTED_FAILURE"
            if index == 3
            else f"E_TUPLE_HEALTH_{module_id.upper()}_RUNTIME"
        )
        full["module_results"][index] = {
            "module_id": module_id,
            "status": "ERROR",
            "error_code": error_code,
            "trace_written": True,
        }
    full["completed_module_count"] = 3
    full["failed_module_count"] = 4
    full["failure_count"] = 4
    full["unaccounted_failure_count"] = 1
    full["resource"]["GPU_type"] = "NVIDIA_A30_24GB"
    full["cumulative_resource"]["cumulative_submission_count"] = 13
    MODULE._validate_tuple_health_full(full, _config())

    wrong_config = json.loads(json.dumps(full))
    wrong_config["config_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_config, _config())

    wrong_health = json.loads(json.dumps(full))
    wrong_health["engineering_health_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_health, _config())


def test_historical_attempt_17_full_record_requires_both_sealed_commitments() -> None:
    full = _healthy_full()
    full["status"] = "ENGINEERING_BLOCKER"
    full["attempt"] = 17
    full["config_commitment_sha256"] = (
        "86ddc49f67fe741dad5d4e3a86d93994412a64b7d48d8fdc37e67f509769a2e7"
    )
    full["engineering_health_commitment_sha256"] = (
        "55aed5b0d368fcf3ad96a8b2f91e535eee1e9030180f20d097dc0e64fc4b7f1f"
    )
    for module_id in ("referent", "attribute", "hand_contact"):
        index = MODULE_IDS.index(module_id)
        full["module_results"][index] = {
            "module_id": module_id,
            "status": "ERROR",
            "error_code": f"E_TUPLE_HEALTH_{module_id.upper()}_UNACCOUNTED_FAILURE",
            "trace_written": True,
        }
    full["completed_module_count"] = 4
    full["failed_module_count"] = 3
    full["failure_count"] = 3
    full["unaccounted_failure_count"] = 3
    full["resource"]["GPU_type"] = "NVIDIA_A30_24GB"
    full["cumulative_resource"]["cumulative_submission_count"] = 17
    MODULE._validate_tuple_health_full(full, _config())

    wrong_config = json.loads(json.dumps(full))
    wrong_config["config_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_config, _config())

    wrong_health = json.loads(json.dumps(full))
    wrong_health["engineering_health_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_health, _config())


def test_historical_attempt_19_full_record_requires_both_sealed_commitments() -> None:
    full = _healthy_full()
    full["status"] = "ENGINEERING_BLOCKER"
    full["attempt"] = 19
    full["config_commitment_sha256"] = (
        "c9cd75f0d813bc0bbf506aa97691e883af88647f09af1ecb758bacb3ef171916"
    )
    full["engineering_health_commitment_sha256"] = (
        "b0a0a56b79a927355c3123f352057d038cea41e3e7dab202802f6ebf3bb64b17"
    )
    for module_id in ("referent", "attribute"):
        index = MODULE_IDS.index(module_id)
        full["module_results"][index] = {
            "module_id": module_id,
            "status": "ERROR",
            "error_code": f"E_TUPLE_HEALTH_{module_id.upper()}_TUPLE_GROUNDING_STATE",
            "trace_written": True,
        }
    full["completed_module_count"] = 5
    full["failed_module_count"] = 2
    full["failure_count"] = 2
    full["resource"]["GPU_type"] = "NVIDIA_A30_24GB"
    full["cumulative_resource"]["cumulative_submission_count"] = 19
    MODULE._validate_tuple_health_full(full, _config())

    wrong_config = json.loads(json.dumps(full))
    wrong_config["config_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_config, _config())

    wrong_health = json.loads(json.dumps(full))
    wrong_health["engineering_health_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_health, _config())


def test_attempt_20_full_pass_requires_the_sealed_health_provenance() -> None:
    config = _config()
    sealed = config["learner_effective_engineering_health_pass_result"]
    compact = sealed["compact_aggregate"]
    full = _healthy_full()
    full["attempt"] = 20
    for key in (
        "public_fixture_manifest_commitment_sha256",
        "runner_commitment_sha256",
        "config_commitment_sha256",
        "dependency_config_commitment_sha256",
        "microfixture_manifest_commitment_sha256",
        "engineering_health_commitment_sha256",
    ):
        full[key] = compact[key]
    full["resource"] = {
        "GPU_type": "NVIDIA_A30_24GB",
        "GPU_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "wall_minutes": 7.504362936814626,
        "GPU_hours": 0.1250727156135771,
        "new_storage_GiB": 1.703854650259018e-5,
        "direct_monetary_cost_USD": 0,
    }
    full["cumulative_resource"] = {
        "cumulative_submission_count": 20,
        "cumulative_wall_minutes": 113.93733958403274,
        "cumulative_GPU_hours": 1.8989556597338781,
        "cumulative_new_storage_GiB": 5.7155266404151917e-5,
        "cumulative_direct_monetary_cost_USD": 0.0,
    }
    MODULE._validate_tuple_health_full(full, config)

    wrong_config = json.loads(json.dumps(full))
    wrong_config["config_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_config, config)

    wrong_health = json.loads(json.dumps(full))
    wrong_health["engineering_health_commitment_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_FULL_SCHEMA"):
        MODULE._validate_tuple_health_full(wrong_health, config)


def test_nltk_manifest_allows_only_the_exact_hashed_provenance_marker(
    tmp_path: Path,
) -> None:
    config = _config()
    public = tmp_path / "public"
    source = public / "models/nltk_data"
    source.mkdir(parents=True)
    records = []
    marker = source / ".extracted-from-sha256"
    marker.write_bytes(
        b"6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b\n"
    )
    assert marker.stat().st_size == 65
    assert MODULE.file_digest(marker) == (
        "f5fac8169578d94a1f568e047af30ecf001d964f99248d822a841436a35d4407"
    )
    records.append(
        {"relative_path": marker.name, "sha256": MODULE.file_digest(marker)}
    )
    for top_level, count in (
        ("averaged_perceptron_tagger_eng", 10),
        ("wordnet", 11),
    ):
        for index in range(count):
            path = source / top_level / f"resource-{index:02d}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{top_level}-{index}".encode())
            records.append(
                {
                    "relative_path": path.relative_to(source).as_posix(),
                    "sha256": MODULE.file_digest(path),
                }
            )
    manifest = {"nltk_resource_files": records}
    commitment = MODULE.digest(manifest)
    manifest["tuple_dependency_commitment_sha256"] = commitment
    manifest_path = MODULE._tuple_run_root(public) / "dependency_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(MODULE.canonical(manifest) + b"\n")
    config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_premodel_result"
    ]["dependency_manifest_commitment_sha256"] = commitment

    staged = MODULE._stage_tuple_nltk_resources(public, tmp_path / "scratch", config)
    assert (staged / "taggers/averaged_perceptron_tagger_eng").is_symlink()
    assert (staged / "corpora/wordnet").is_symlink()

    unexpected = source / ".unexpected-marker"
    marker.rename(unexpected)
    records[0] = {
        "relative_path": unexpected.name,
        "sha256": MODULE.file_digest(unexpected),
    }
    manifest = {"nltk_resource_files": records}
    commitment = MODULE.digest(manifest)
    manifest["tuple_dependency_commitment_sha256"] = commitment
    manifest_path.write_bytes(MODULE.canonical(manifest) + b"\n")
    config["calibration_C"]["extractor"][
        "mechanistic_training_tuple_premodel_result"
    ]["dependency_manifest_commitment_sha256"] = commitment
    with pytest.raises(RuntimeError, match="E_TUPLE_NLTK_RESOURCE_SET"):
        MODULE._stage_tuple_nltk_resources(
            public, tmp_path / "unexpected-scratch", config
        )


def test_nltk_matplotlib_repair_rejects_changed_public_wheel_identity() -> None:
    config = _config()
    MODULE._engineering_health_nltk_matplotlib_repair(config)
    mutated = json.loads(json.dumps(config))
    mutated["learner_effective_engineering_health_NLTK_matplotlib_repair"][
        "public_runtime_dependency_repair"
    ]["artifacts"][0]["sha256"] = "e" * 64
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_NLTK_MATPLOTLIB_REPAIR_COMMITMENT",
    ):
        MODULE._engineering_health_nltk_matplotlib_repair(mutated)


def test_generic_gres_repair_preserves_attempt_14_and_rejects_mutation() -> None:
    config = _config()
    attempt = MODULE._engineering_health_attempt_14_result(config)
    assert attempt["blocker_commitment_sha256"] == (
        "584392429481f2a6d5dfc16d1fed8bfe2b49902c08f5767ad143e0feaabe0824"
    )
    assert attempt["compact_aggregate"]["scientific_metric_count"] == 0
    repair = MODULE._engineering_health_generic_gres_repair(config)
    assert repair["repair_commitment_sha256"] == (
        "ce9040c8bd1ab65d9dc0cf943bc18774f9486a0df6030b60696e81a27b81ce0a"
    )
    assert repair["active_attempt_resource_policy"]["attempt"] == 15
    assert repair["failure_specific_repair"][
        "in_container_exact_device_name_and_23_to_25_GiB_memory_attestation_remains_blocking"
    ] is True
    mutated = json.loads(json.dumps(config))
    mutated[
        "learner_effective_engineering_health_generic_GRES_serialization_repair"
    ]["failure_specific_repair"]["additional_exact_scheduler_TresPerNode_form"] = (
        "gres/gpu:2"
    )
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_GENERIC_GRES_REPAIR_COMMITMENT"
    ):
        MODULE._engineering_health_generic_gres_repair(mutated)


def test_active_dispatch_repair_preserves_attempt_15_and_rejects_mutation() -> None:
    config = _config()
    attempt = MODULE._engineering_health_attempt_15_result(config)
    assert attempt["blocker_commitment_sha256"] == (
        "7a5710234a2ffb072fa9259cf22dc5c9e62338f35466ef699f3e26cb9db427bb"
    )
    assert attempt["stable_aggregate_diagnosis"]["stable_error_code"] == (
        "E_TUPLE_HEALTH_NLTK_MATPLOTLIB_REPAIR_ATTEMPT"
    )
    assert attempt["compact_aggregate"]["scientific_metric_count"] == 0
    repair = MODULE._engineering_health_active_dispatch_repair(config)
    assert repair["repair_commitment_sha256"] == (
        "0c617e6314a914b5ad30e167eceb4a3981c72bfa7661558c9887cfa19037398f"
    )
    assert repair["active_attempt_resource_policy"]["attempt"] == 16
    mutated = json.loads(json.dumps(config))
    mutated["learner_effective_engineering_health_active_dispatch_repair"][
        "failure_specific_repair"
    ]["attempt_16_requires_active_dispatch_repair_validation_first"] = False
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_ACTIVE_DISPATCH_REPAIR_COMMITMENT"
    ):
        MODULE._engineering_health_active_dispatch_repair(mutated)


def test_historical_resource_dispatch_repair_preserves_attempt_16() -> None:
    config = _config()
    attempt = MODULE._engineering_health_attempt_16_result(config)
    assert attempt["blocker_commitment_sha256"] == (
        "6120cc49b0fe802ba292899bb9773ddf7549a4b886b184b67f282f8af9949816"
    )
    assert attempt["stable_aggregate_diagnosis"]["stable_error_code"] == (
        "E_TUPLE_HEALTH_GPU_HOUR_BUDGET"
    )
    assert attempt["compact_aggregate"]["scientific_metric_count"] == 0
    repair = MODULE._engineering_health_historical_resource_dispatch_repair(config)
    assert repair["repair_commitment_sha256"] == (
        "857b5353858d936dcaef3ea2c8ab2f6a1f569ace3f464f2de80dd831c273f87d"
    )
    assert repair["active_attempt_resource_policy"]["attempt"] == 17


def test_ffmpeg_prettytable_repair_preserves_attempt_17_and_rejects_mutation() -> None:
    config = _config()
    attempt = MODULE._engineering_health_attempt_17_result(config)
    assert attempt["blocker_commitment_sha256"] == (
        "bcb7e203dd20013ecd50e7a50d1f09a75aeb59f766f193843eb2ac4ba5fa762d"
    )
    assert attempt["compact_aggregate"]["scientific_metric_count"] == 0
    assert attempt["stable_aggregate_diagnosis"][
        "referent_and_attribute_missing_exact_bundled_ffmpeg_command"
    ] is True
    assert attempt["stable_aggregate_diagnosis"][
        "hand_contact_missing_exact_prettytable_import"
    ] is True
    repair = MODULE._engineering_health_ffmpeg_prettytable_repair(config)
    assert repair["repair_commitment_sha256"] == (
        "f1d4153ae0835b90b0e0ba3749b7beef7c21e846a4d5f344d6dede4c1a092124"
    )
    assert repair["active_attempt_resource_policy"]["attempt"] == 18

    mutated = json.loads(json.dumps(config))
    mutated["learner_effective_engineering_health_ffmpeg_prettytable_repair"][
        "failure_specific_repair"
    ]["prettytable_runtime"]["wheel_sha256"] = "e" * 64
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_FFMPEG_PRETTYTABLE_REPAIR_COMMITMENT",
    ):
        MODULE._engineering_health_ffmpeg_prettytable_repair(mutated)


def test_historical_full_result_lineage_repair_preserves_attempt_18() -> None:
    config = _config()
    attempt = MODULE._engineering_health_attempt_18_result(config)
    assert attempt["blocker_commitment_sha256"] == (
        "68ad48be2c03db7d06f3939a8e9a7ac92a2dd1d6e37d9b80370a5207a8227c9e"
    )
    assert attempt["stable_aggregate_diagnosis"]["stable_error_code"] == (
        "E_TUPLE_HEALTH_FULL_SCHEMA"
    )
    assert attempt["compact_aggregate"]["scientific_metric_count"] == 0
    repair = MODULE._engineering_health_historical_full_result_lineage_repair(
        config
    )
    assert repair["repair_commitment_sha256"] == (
        "343c98dcbd4f78f838a8f854a7b6d3393349058d64e22efac663d738fe485ca9"
    )
    assert repair["active_attempt_resource_policy"]["attempt"] == 19

    mutated = json.loads(json.dumps(config))
    mutated[
        "learner_effective_engineering_health_historical_full_result_lineage_repair"
    ]["preserved_without_change"]["attempt_17_config_commitment_sha256"] = (
        "e" * 64
    )
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_HISTORICAL_FULL_RESULT_LINEAGE_REPAIR_COMMITMENT",
    ):
        MODULE._engineering_health_historical_full_result_lineage_repair(mutated)


def test_grounding_state_repair_preserves_attempt_19_and_rejects_mutation() -> None:
    config = _config()
    attempt = MODULE._engineering_health_attempt_19_result(config)
    assert attempt["blocker_commitment_sha256"] == (
        "af86c7a955acac78a905adcbd2d4f85c3d4cb9da362fc1db718b29f3ed053bad"
    )
    assert attempt["compact_aggregate"]["scientific_metric_count"] == 0
    assert attempt["stable_aggregate_diagnosis"]["missing_key_count"] == 0
    assert attempt["stable_aggregate_diagnosis"]["unexpected_keys"] == [
        "bert.embeddings.position_ids",
        "label_enc.weight",
    ]
    repair = MODULE._engineering_health_grounding_state_compatibility_repair(
        config
    )
    assert repair["repair_commitment_sha256"] == (
        "3ecdaa380536c23c0a6b4d695c22a3d9be5209dab1090b001c4e667cc6e5cbeb"
    )
    assert repair["active_attempt_resource_policy"]["attempt"] == 20

    mutated = json.loads(json.dumps(config))
    mutated[
        "learner_effective_engineering_health_grounding_state_compatibility_repair"
    ]["failure_specific_repair"]["required_unexpected_keys_exact"].append(
        "unapproved.weight"
    )
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_GROUNDING_STATE_COMPATIBILITY_REPAIR_COMMITMENT",
    ):
        MODULE._engineering_health_grounding_state_compatibility_repair(mutated)


def test_grounding_state_compatibility_is_exact_and_order_independent() -> None:
    compatible = SimpleNamespace(
        missing_keys=[],
        unexpected_keys=["label_enc.weight", "bert.embeddings.position_ids"],
    )
    record = MODULE._tuple_grounding_state_compatibility(compatible, _config())
    assert record == {
        "missing_key_count": 0,
        "unexpected_key_count": 2,
        "compatibility_repair_commitment_sha256": (
            "3ecdaa380536c23c0a6b4d695c22a3d9be5209dab1090b001c4e667cc6e5cbeb"
        ),
    }

    for incompatible in (
        SimpleNamespace(
            missing_keys=["missing.weight"],
            unexpected_keys=[
                "bert.embeddings.position_ids",
                "label_enc.weight",
            ],
        ),
        SimpleNamespace(
            missing_keys=[],
            unexpected_keys=[
                "bert.embeddings.position_ids",
                "label_enc.weight",
                "unapproved.weight",
            ],
        ),
    ):
        with pytest.raises(RuntimeError, match="E_TUPLE_GROUNDING_STATE"):
            MODULE._tuple_grounding_state_compatibility(incompatible, _config())


def test_grounding_state_compatibility_is_shared_by_sizing_and_production() -> None:
    source = Path("scripts/run_synthetic_video_calibration.py").read_text()
    assert source.count(
        "_tuple_grounding_state_compatibility(incompatible, cfg)"
    ) == 2


def test_attempt_20_engineering_pass_is_exact_and_unlocks_development_only() -> None:
    config = _config()
    result = MODULE._engineering_health_attempt_20_pass_result(config)
    assert result["pass_commitment_sha256"] == (
        "25486c1a4217ecd4f1a4eecfbdf90f5802d79c9300ea038155030448b2089839"
    )
    assert result["compact_aggregate"]["status"] == "PASS_ENGINEERING_HEALTH"
    assert result["compact_aggregate"]["scientific_metric_count"] == 0
    assert set(result["module_health"].values()) == {"PASS_ENGINEERING"}
    assert result["terminal_gate"] == {
        "engineering_health_pass": True,
        "public_development_authorized": True,
        "public_holdout_authorized": False,
        "governed_C_authorized": False,
        "LTX_or_synthetic_learner_run": False,
    }

    mutated = json.loads(json.dumps(config))
    mutated["learner_effective_engineering_health_pass_result"][
        "terminal_gate"
    ]["public_holdout_authorized"] = True
    with pytest.raises(
        RuntimeError, match="E_TUPLE_HEALTH_ATTEMPT_20_PASS_COMMITMENT"
    ):
        MODULE._engineering_health_attempt_20_pass_result(mutated)


def test_public_development_attempt_1_and_mask_roundtrip_repair_are_exact() -> None:
    config = _config()
    result = MODULE._public_development_engineering_attempt_1_result(config)
    repair = MODULE._public_development_truth_mask_roundtrip_repair(config)
    assert result["result_commitment_sha256"] == (
        "f8c315531e80049a1c0c8860dfdb99587908c97793ed6f705a911c2492756221"
    )
    assert result["scientific_metric_count"] == 0
    assert result["failed_module_count"] == 2
    assert repair["repair_commitment_sha256"] == (
        "638d5c3c43ccb81421e5667b00308aee68fbeed7cd2f80502c3395ba385f82e2"
    )
    assert repair[
        "fixture_source_partition_model_seed_threshold_metric_or_gate_changed"
    ] is False
    assert repair["public_development_integrity_attempt"] == 2

    mutated = json.loads(json.dumps(config))
    mutated["learner_effective_public_development_truth_mask_roundtrip_repair"][
        "per_sample_visible_mask_fraction_min"
    ] = 0.0
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_DEVELOPMENT_TRUTH_MASK_ROUNDTRIP_REPAIR_COMMITMENT",
    ):
        MODULE._public_development_truth_mask_roundtrip_repair(mutated)


def test_public_development_attempt_2_and_attribute_dependency_repair_are_exact() -> None:
    config = _config()
    result = MODULE._public_development_engineering_attempt_2_result(config)
    repair = MODULE._public_development_attribute_dependency_repair(config)
    assert result["result_commitment_sha256"] == (
        "b35f6a083a432cd6f2ab00dc01e860b7086d9c6b160c4efe1c1f8e22a9e78e27"
    )
    assert result["integrity_completed_module_count"] == 7
    assert result["scientific_metric_count"] == 0
    assert repair["repair_commitment_sha256"] == (
        "602c273dbfc60ef91af083c2baad77d2831981765c4a79f7a8ccf4b8b4b5073b"
    )
    assert repair["referent_failure_remains_gate_failure"] is True
    assert repair["attribute_unmeasured_remains_critical_gate_failure"] is True
    assert repair["public_development_integrity_attempt"] == 3

    mutated = json.loads(json.dumps(config))
    mutated[
        "learner_effective_public_development_attribute_dependency_repair"
    ]["attribute_unmeasured_remains_critical_gate_failure"] = False
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_DEVELOPMENT_ATTRIBUTE_DEPENDENCY_REPAIR_COMMITMENT",
    ):
        MODULE._public_development_attribute_dependency_repair(mutated)


def test_complete_public_development_no_go_is_hash_bound_and_terminal() -> None:
    config = _config()
    result = MODULE._public_development_terminal_result(config)
    assert result["result_commitment_sha256"] == (
        "42338302949e27e0ed7c3f6e8a5f70e10bb380a5e8158378e89f5ff87c350e9d"
    )
    assert result["critical_axis_pass_count"] == 2
    assert result["critical_axis_required_count"] == 5
    assert result["validated_axis_count"] == 3
    assert result["validated_axis_required_count"] == 6
    assert result["terminal_gate"] == {
        "public_development_pass": False,
        "public_holdout_authorized": False,
        "governed_C_authorized": False,
        "LTX_or_synthetic_learner_run": False,
    }

    mutated = json.loads(json.dumps(config))
    mutated["learner_effective_public_development_terminal_result"][
        "critical_axis_pass_count"
    ] = 3
    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_PUBLIC_DEVELOPMENT_TERMINAL_RESULT_COMMITMENT",
    ):
        MODULE._public_development_terminal_result(mutated)

    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_PUBLIC_DEVELOPMENT_SCIENTIFIC_NO_GO_TERMINAL",
    ):
        MODULE.qualify_tuple_public(
            argparse.Namespace(
                partition="holdout",
                config=Path("configs/synthetic_video_real_only_proof.json"),
            )
        )


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


def test_tuple_health_budget_enforces_active_lineage_repair_attempt_and_limits() -> None:
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
    prior.append(
        {
            "attempt": 8,
            "GPU_type": "NVIDIA_H100_NVL",
            "GPU_count": 1,
            "wall_minutes": 3.286515990893046,
            "GPU_hours": 0.054775266514884104,
            "new_storage_GiB": 1.6046687960624695e-6,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 9,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 0.16666666666666666,
            "GPU_hours": 0.002777777777777778,
            "new_storage_GiB": 7.487833499908447e-7,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 10,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 4.000260214010875,
            "GPU_hours": 0.06667100356684791,
            "new_storage_GiB": 1.2861564755439758e-6,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 11,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 3.6992401123046874,
            "GPU_hours": 0.061654001871744794,
            "new_storage_GiB": 1.2740492820739746e-6,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 12,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 0.05,
            "GPU_hours": 0.0008333333333333334,
            "new_storage_GiB": 0.0,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 13,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 5.937931791941325,
            "GPU_hours": 0.09896552986568875,
            "new_storage_GiB": 0.000009451061487197876,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 14,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 0.15,
            "GPU_hours": 0.0025,
            "new_storage_GiB": 1.1408701539039612e-6,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 15,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 0.16666666666666666,
            "GPU_hours": 0.002777777777777778,
            "new_storage_GiB": 1.430511474609375e-6,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 16,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 0.18333333333333332,
            "GPU_hours": 0.0030555555555555557,
            "new_storage_GiB": 1.430511474609375e-6,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 17,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 6.407094264030457,
            "GPU_hours": 0.10678490440050761,
            "new_storage_GiB": 6.639398634433746e-06,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 18,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 0.16666666666666666,
            "GPU_hours": 0.002777777777777778,
            "new_storage_GiB": 1.430511474609375e-06,
            "direct_monetary_cost_USD": 0,
        }
    )
    prior.append(
        {
            "attempt": 19,
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "wall_minutes": 6.615966049830119,
            "GPU_hours": 0.11026610083050198,
            "new_storage_GiB": 9.074807167053223e-06,
            "direct_monetary_cost_USD": 0,
        }
    )
    budget = MODULE._tuple_health_budget(20, prior, config)
    assert budget["attempt"] == 20
    assert budget["GPU_type"] == MODULE._engineering_health_resource_policy(config)[
        "GPU_type"
    ]
    assert budget["GPU_count"] == 1
    assert budget["per_submission_wall_minutes_max"] == 60
    assert budget["remaining_GPU_hours"] == pytest.approx(1.0)
    assert budget["remaining_storage_GiB"] == pytest.approx(1.0)
    assert budget["direct_monetary_cost_USD"] == 0

    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_ATTEMPT_BUDGET"):
        MODULE._tuple_health_budget(21, prior, config)


def test_attempt_12_pre_runner_failure_has_sealed_marker_free_resource() -> None:
    resource = MODULE._tuple_health_incomplete_attempt_resource(
        Path("/does-not-exist"), 12, _config()
    )
    assert resource == {
        "GPU_type": "NVIDIA_A30_24GB",
        "GPU_count": 1,
        "CPU_count": 8,
        "memory_GiB": 32,
        "wall_minutes": 0.05,
        "GPU_hours": 0.0008333333333333334,
        "new_storage_GiB": 0.0,
        "direct_monetary_cost_USD": 0,
        "submission_started_epoch": 1785912624.0,
    }


@pytest.mark.parametrize(
    "attempt,wall_minutes",
    [
        (14, 0.15),
        (15, 0.16666666666666666),
        (16, 0.18333333333333332),
        (18, 0.16666666666666666),
    ],
)
def test_active_resource_dispatch_uses_exact_incomplete_attempt_wall(
    tmp_path: Path, attempt: int, wall_minutes: float
) -> None:
    root = tmp_path / "health-root"
    attempt_root = root / "health" / f"attempt-{attempt:02d}"
    MODULE.write_private_new(
        attempt_root / "wrapper-started.json",
        {
            "schema_version": 1,
            "attempt": attempt,
            "submission_started_epoch": __import__("time").time(),
            "GPU_type": "NVIDIA_A30_24GB",
            "GPU_count": 1,
            "CPU_count": 8,
            "memory_GiB": 32,
        },
    )
    resource = MODULE._tuple_health_incomplete_attempt_resource(
        root, attempt, _config()
    )
    assert resource["wall_minutes"] == pytest.approx(wall_minutes)
    assert resource["GPU_hours"] == pytest.approx(wall_minutes / 60.0)


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
    attempt_8 = MODULE._engineering_health_attempt_8_result(config)
    assert attempt_8["blocker_commitment_sha256"] == (
        "409c36d2c3ba4aefdd2f510c661ba363c000fc232dce2fcfba0151ba25f9aad7"
    )
    assert attempt_8["compact_aggregate"]["scientific_metric_count"] == 0
    repair = MODULE._engineering_health_git_fallback_repair(config)
    assert repair["repair_commitment_sha256"] == (
        "b6a93e3a0b0b716d8bdd8fdd47656e69f8ff5b66c0a3ec8f96973e565a9066f9"
    )
    assert repair["active_attempt_resource_policy"]["attempt"] == 9
    assert repair["failure_specific_repair"][
        "model_fixture_source_threshold_partition_seed_metric_or_gate_changed"
    ] is False
    attempt_9 = MODULE._engineering_health_attempt_9_result(config)
    assert attempt_9["blocker_commitment_sha256"] == (
        "2fed8750a5f60929e144aecb89ef6f3bb1d1d048d9cd1d1080e8f21393ae7a43"
    )
    assert attempt_9["submission_provenance"]["job_id"] == 316813
    assert attempt_9["compact_aggregate"]["scientific_metric_count"] == 0
    lineage = MODULE._engineering_health_historical_lineage_repair(config)
    assert lineage["repair_commitment_sha256"] == (
        "be09b2e0c3a2026957dbd20b18129e65cd1b8ca03a8a62fdb49d568a76f526b0"
    )
    assert lineage["active_attempt_resource_policy"]["attempt"] == 10
    assert lineage["failure_specific_repair"][
        "other_historical_or_unbound_commitments_rejected"
    ] is True
    attempt_10 = MODULE._engineering_health_attempt_10_result(config)
    assert attempt_10["blocker_commitment_sha256"] == (
        "1575ef02678ef0a41e32d62af7b3c8807ce449df65243c97b60baddca8ad0257"
    )
    assert attempt_10["submission_provenance"]["job_id"] == 316832
    assert attempt_10["compact_aggregate"]["scientific_metric_count"] == 0
    portable = MODULE._engineering_health_portable_ast_repair(config)
    assert portable["repair_commitment_sha256"] == (
        "7896a53385551d206736cff11d803ed6f7600317abb8cd96c94ad8496f4b67c1"
    )
    assert portable["active_attempt_resource_policy"]["attempt"] == 11
    assert portable["failure_specific_repair"][
        "portable_AST_bundle_sha256"
    ] == "74d7707cd8d485aa00514f7f216759f848893923e990f94e3a1d19b935958b8d"
    attempt_11 = MODULE._engineering_health_attempt_11_result(config)
    assert attempt_11["blocker_commitment_sha256"] == (
        "e86739340a2d969b04b823932f9583d3defe7ed92a6d849042a681d03b7fb2f5"
    )
    assert attempt_11["submission_provenance"]["job_id"] == 316845
    assert attempt_11["compact_aggregate"]["scientific_metric_count"] == 0
    fixture_bind = MODULE._engineering_health_fixture_bind_repair(config)
    assert fixture_bind["repair_commitment_sha256"] == (
        "cca49a50a895ab4ac94997fbfe7bc256efc4b9ae790b1925a5671dc3b1a1e3f7"
    )
    assert fixture_bind["active_attempt_resource_policy"]["attempt"] == 12
    assert fixture_bind["failure_specific_repair"]["bind_mode"] == "READ_ONLY"
    attempt_12 = MODULE._engineering_health_attempt_12_result(config)
    assert attempt_12["blocker_commitment_sha256"] == (
        "5471156337683b5b3695ee26ea9acfa1c658acfb8400a6c681538eecf6e948bc"
    )
    assert attempt_12["submission_provenance"]["job_id"] == 316878
    assert attempt_12["stable_aggregate_diagnosis"]["runner_entry_count"] == 0
    export_repair = MODULE._engineering_health_submission_export_repair(config)
    assert export_repair["repair_commitment_sha256"] == (
        "e700576735251403138e34fa9918fa2ab0e3d723e2252871302fcd4da77516ba"
    )
    assert export_repair["active_attempt_resource_policy"]["attempt"] == 13
    assert export_repair["failure_specific_repair"]["required_export_value"] == "60"
    attempt_13 = MODULE._engineering_health_attempt_13_result(config)
    assert attempt_13["blocker_commitment_sha256"] == (
        "a50e361b9f6ca9f6367b2b26190a726e4d005d08b6c9563dbd1dc976e1903bb1"
    )
    assert attempt_13["submission_provenance"]["job_id"] == 316887
    assert attempt_13["compact_aggregate"]["scientific_metric_count"] == 0
    dependency_repair = MODULE._engineering_health_nltk_matplotlib_repair(config)
    assert dependency_repair["repair_commitment_sha256"] == (
        "37e9c960ea2e60b664933fd82734722a458b13adfdd32163cb2b4a71f0181036"
    )
    assert dependency_repair["active_attempt_resource_policy"]["attempt"] == 14
    assert dependency_repair["public_runtime_dependency_repair"]["wheel_count"] == 8
    attempt_14 = MODULE._engineering_health_attempt_14_result(config)
    assert attempt_14["blocker_commitment_sha256"] == (
        "584392429481f2a6d5dfc16d1fed8bfe2b49902c08f5767ad143e0feaabe0824"
    )
    assert attempt_14["submission_provenance"]["job_id"] == 316905
    assert attempt_14["compact_aggregate"]["scientific_metric_count"] == 0
    gres_repair = MODULE._engineering_health_generic_gres_repair(config)
    assert gres_repair["repair_commitment_sha256"] == (
        "ce9040c8bd1ab65d9dc0cf943bc18774f9486a0df6030b60696e81a27b81ce0a"
    )
    assert gres_repair["active_attempt_resource_policy"]["attempt"] == 15
    attempt_15 = MODULE._engineering_health_attempt_15_result(config)
    assert attempt_15["blocker_commitment_sha256"] == (
        "7a5710234a2ffb072fa9259cf22dc5c9e62338f35466ef699f3e26cb9db427bb"
    )
    assert attempt_15["submission_provenance"]["job_id"] == 316918
    assert attempt_15["compact_aggregate"]["scientific_metric_count"] == 0
    dispatch_repair = MODULE._engineering_health_active_dispatch_repair(config)
    assert dispatch_repair["repair_commitment_sha256"] == (
        "0c617e6314a914b5ad30e167eceb4a3981c72bfa7661558c9887cfa19037398f"
    )
    assert dispatch_repair["active_attempt_resource_policy"]["attempt"] == 16
    attempt_16 = MODULE._engineering_health_attempt_16_result(config)
    assert attempt_16["blocker_commitment_sha256"] == (
        "6120cc49b0fe802ba292899bb9773ddf7549a4b886b184b67f282f8af9949816"
    )
    assert attempt_16["submission_provenance"]["job_id"] == 316924
    assert attempt_16["compact_aggregate"]["scientific_metric_count"] == 0
    resource_dispatch = (
        MODULE._engineering_health_historical_resource_dispatch_repair(config)
    )
    assert resource_dispatch["repair_commitment_sha256"] == (
        "857b5353858d936dcaef3ea2c8ab2f6a1f569ace3f464f2de80dd831c273f87d"
    )
    assert resource_dispatch["active_attempt_resource_policy"]["attempt"] == 17
    attempt_17 = MODULE._engineering_health_attempt_17_result(config)
    assert attempt_17["blocker_commitment_sha256"] == (
        "bcb7e203dd20013ecd50e7a50d1f09a75aeb59f766f193843eb2ac4ba5fa762d"
    )
    assert attempt_17["submission_provenance"]["job_id"] == 316933
    assert attempt_17["compact_aggregate"]["scientific_metric_count"] == 0
    assert attempt_17["module_health"] == {
        "adapter_and_lexical": "PASS_ENGINEERING",
        "referent": "ERROR",
        "recurrence": "PASS_ENGINEERING",
        "attribute": "ERROR",
        "hand_contact": "ERROR",
        "sensor": "PASS_ENGINEERING",
        "order_action": "PASS_ENGINEERING",
    }
    runtime_repair = MODULE._engineering_health_ffmpeg_prettytable_repair(config)
    assert runtime_repair["repair_commitment_sha256"] == (
        "f1d4153ae0835b90b0e0ba3749b7beef7c21e846a4d5f344d6dede4c1a092124"
    )
    assert runtime_repair["active_attempt_resource_policy"]["attempt"] == 18
    assert runtime_repair["failure_specific_repair"]["bundled_ffmpeg"][
        "binary_sha256"
    ] == "e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99"
    assert runtime_repair["failure_specific_repair"]["prettytable_runtime"][
        "wheel_sha256"
    ] == "b3346e0e6f79180833aebaac088ae926340586cf6d7d991b9eb125b65f72313a"
    attempt_18 = MODULE._engineering_health_attempt_18_result(config)
    assert attempt_18["blocker_commitment_sha256"] == (
        "68ad48be2c03db7d06f3939a8e9a7ac92a2dd1d6e37d9b80370a5207a8227c9e"
    )
    assert attempt_18["submission_provenance"]["job_id"] == 316944
    assert attempt_18["compact_aggregate"]["scientific_metric_count"] == 0
    lineage_repair = (
        MODULE._engineering_health_historical_full_result_lineage_repair(config)
    )
    assert lineage_repair["repair_commitment_sha256"] == (
        "343c98dcbd4f78f838a8f854a7b6d3393349058d64e22efac663d738fe485ca9"
    )
    assert lineage_repair["active_attempt_resource_policy"]["attempt"] == 19
    attempt_19 = MODULE._engineering_health_attempt_19_result(config)
    assert attempt_19["blocker_commitment_sha256"] == (
        "af86c7a955acac78a905adcbd2d4f85c3d4cb9da362fc1db718b29f3ed053bad"
    )
    assert attempt_19["submission_provenance"]["job_id"] == 316954
    assert attempt_19["compact_aggregate"]["scientific_metric_count"] == 0
    grounding_repair = (
        MODULE._engineering_health_grounding_state_compatibility_repair(config)
    )
    assert grounding_repair["repair_commitment_sha256"] == (
        "3ecdaa380536c23c0a6b4d695c22a3d9be5209dab1090b001c4e667cc6e5cbeb"
    )
    assert grounding_repair["active_attempt_resource_policy"]["attempt"] == 20
    effective = MODULE._engineering_health_resource_policy(config)
    assert effective["per_submission_wall_minutes_max"] == 60
    assert effective["initial_plus_repair_resmoke_submission_count_max"] == 20

    with pytest.raises(
        RuntimeError,
        match="E_TUPLE_HEALTH_ROUTE_EXHAUSTED_PASS",
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
        "cuda", topology_attestation=attestation, attempt=4, cfg=_config()
    )
    assert commitment == MODULE.file_digest(attestation)
    full_h100_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: "NVIDIA H100 NVL",
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=90 * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=full_h100_cuda)
    )
    attempt_8_attestation = _write_topology_attestation(
        tmp_path / "attempt-08",
        monkeypatch,
        attempt=8,
        job_id=316700,
        gpu_type="NVIDIA_H100_NVL",
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_8_attestation,
        attempt=8,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_8_attestation)

    a30_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: "NVIDIA A30",
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=24 * 1024**3
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=a30_cuda))
    attempt_9_attestation = _write_topology_attestation(
        tmp_path / "attempt-09",
        monkeypatch,
        attempt=9,
        job_id=316778,
        gpu_type="NVIDIA_A30_24GB",
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_9_attestation,
        attempt=9,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_9_attestation)

    attempt_10_gpu_type = MODULE._engineering_health_historical_lineage_repair(
        _config()
    )["active_attempt_resource_policy"]["GPU_type"]
    device_by_type = {
        "NVIDIA_A30_24GB": ("NVIDIA A30", 24),
        "NVIDIA_H100_NVL": ("NVIDIA H100 NVL", 90),
        "NVIDIA_H100_NVL_3G_47GB_MIG": ("NVIDIA H100 NVL", 47),
        "NVIDIA_H200_NVL": ("NVIDIA H200 NVL", 140),
    }
    attempt_10_name, attempt_10_memory = device_by_type[attempt_10_gpu_type]
    attempt_10_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_10_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_10_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_10_cuda)
    )
    attempt_10_attestation = _write_topology_attestation(
        tmp_path / "attempt-10",
        monkeypatch,
        attempt=10,
        job_id=316814,
        gpu_type=attempt_10_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_10_attestation,
        attempt=10,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_10_attestation)

    attempt_11_gpu_type = MODULE._engineering_health_portable_ast_repair(
        _config()
    )["active_attempt_resource_policy"]["GPU_type"]
    attempt_11_name, attempt_11_memory = device_by_type[attempt_11_gpu_type]
    attempt_11_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_11_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_11_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_11_cuda)
    )
    attempt_11_attestation = _write_topology_attestation(
        tmp_path / "attempt-11",
        monkeypatch,
        attempt=11,
        job_id=316833,
        gpu_type=attempt_11_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_11_attestation,
        attempt=11,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_11_attestation)

    attempt_12_gpu_type = MODULE._engineering_health_fixture_bind_repair(
        _config()
    )["active_attempt_resource_policy"]["GPU_type"]
    attempt_12_name, attempt_12_memory = device_by_type[attempt_12_gpu_type]
    attempt_12_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_12_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_12_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_12_cuda)
    )
    attempt_12_attestation = _write_topology_attestation(
        tmp_path / "attempt-12",
        monkeypatch,
        attempt=12,
        job_id=316846,
        gpu_type=attempt_12_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_12_attestation,
        attempt=12,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_12_attestation)

    attempt_13_gpu_type = MODULE._engineering_health_submission_export_repair(
        _config()
    )["active_attempt_resource_policy"]["GPU_type"]
    attempt_13_name, attempt_13_memory = device_by_type[attempt_13_gpu_type]
    attempt_13_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_13_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_13_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_13_cuda)
    )
    attempt_13_attestation = _write_topology_attestation(
        tmp_path / "attempt-13",
        monkeypatch,
        attempt=13,
        job_id=316879,
        gpu_type=attempt_13_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_13_attestation,
        attempt=13,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_13_attestation)

    attempt_14_gpu_type = MODULE._engineering_health_nltk_matplotlib_repair(
        _config()
    )["active_attempt_resource_policy"]["GPU_type"]
    attempt_14_name, attempt_14_memory = device_by_type[attempt_14_gpu_type]
    attempt_14_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_14_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_14_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_14_cuda)
    )
    attempt_14_attestation = _write_topology_attestation(
        tmp_path / "attempt-14",
        monkeypatch,
        attempt=14,
        job_id=316888,
        gpu_type=attempt_14_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_14_attestation,
        attempt=14,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_14_attestation)

    attempt_15_gpu_type = MODULE._engineering_health_generic_gres_repair(
        _config()
    )["active_attempt_resource_policy"]["GPU_type"]
    attempt_15_name, attempt_15_memory = device_by_type[attempt_15_gpu_type]
    attempt_15_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_15_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_15_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_15_cuda)
    )
    attempt_15_attestation = _write_topology_attestation(
        tmp_path / "attempt-15",
        monkeypatch,
        attempt=15,
        job_id=316906,
        gpu_type=attempt_15_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_15_attestation,
        attempt=15,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_15_attestation)

    attempt_16_gpu_type = MODULE._engineering_health_active_dispatch_repair(
        _config()
    )["active_attempt_resource_policy"]["GPU_type"]
    attempt_16_name, attempt_16_memory = device_by_type[attempt_16_gpu_type]
    attempt_16_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_16_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_16_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_16_cuda)
    )
    attempt_16_attestation = _write_topology_attestation(
        tmp_path / "attempt-16",
        monkeypatch,
        attempt=16,
        job_id=316919,
        gpu_type=attempt_16_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_16_attestation,
        attempt=16,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_16_attestation)

    attempt_17_gpu_type = (
        MODULE._engineering_health_historical_resource_dispatch_repair(
            _config()
        )["active_attempt_resource_policy"]["GPU_type"]
    )
    attempt_17_name, attempt_17_memory = device_by_type[attempt_17_gpu_type]
    attempt_17_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_17_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_17_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_17_cuda)
    )
    attempt_17_attestation = _write_topology_attestation(
        tmp_path / "attempt-17",
        monkeypatch,
        attempt=17,
        job_id=316925,
        gpu_type=attempt_17_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_17_attestation,
        attempt=17,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_17_attestation)

    attempt_18_gpu_type = MODULE._engineering_health_ffmpeg_prettytable_repair(
        _config()
    )["active_attempt_resource_policy"]["GPU_type"]
    attempt_18_name, attempt_18_memory = device_by_type[attempt_18_gpu_type]
    attempt_18_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_18_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_18_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_18_cuda)
    )
    attempt_18_attestation = _write_topology_attestation(
        tmp_path / "attempt-18",
        monkeypatch,
        attempt=18,
        job_id=316934,
        gpu_type=attempt_18_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_18_attestation,
        attempt=18,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_18_attestation)

    attempt_19_gpu_type = (
        MODULE._engineering_health_historical_full_result_lineage_repair(
            _config()
        )["active_attempt_resource_policy"]["GPU_type"]
    )
    attempt_19_name, attempt_19_memory = device_by_type[attempt_19_gpu_type]
    attempt_19_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_19_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_19_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_19_cuda)
    )
    attempt_19_attestation = _write_topology_attestation(
        tmp_path / "attempt-19",
        monkeypatch,
        attempt=19,
        job_id=316945,
        gpu_type=attempt_19_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_19_attestation,
        attempt=19,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_19_attestation)

    attempt_20_gpu_type = (
        MODULE._engineering_health_grounding_state_compatibility_repair(
            _config()
        )["active_attempt_resource_policy"]["GPU_type"]
    )
    attempt_20_name, attempt_20_memory = device_by_type[attempt_20_gpu_type]
    attempt_20_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda _index: attempt_20_name,
        get_device_properties=lambda _index: SimpleNamespace(
            total_memory=attempt_20_memory * 1024**3
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=attempt_20_cuda)
    )
    attempt_20_attestation = _write_topology_attestation(
        tmp_path / "attempt-20",
        monkeypatch,
        attempt=20,
        job_id=316955,
        gpu_type=attempt_20_gpu_type,
    )
    commitment = MODULE._tuple_health_topology(
        "cuda",
        topology_attestation=attempt_20_attestation,
        attempt=20,
        cfg=_config(),
    )
    assert commitment == MODULE.file_digest(attempt_20_attestation)

    bad = json.loads(attestation.read_text())
    bad["task_count"] = 2
    attestation.write_bytes(MODULE.canonical(bad) + b"\n")
    with pytest.raises(RuntimeError, match="E_TUPLE_HEALTH_TOPOLOGY_ATTESTATION"):
        MODULE._tuple_health_topology(
            "cuda", topology_attestation=attestation, attempt=4, cfg=_config()
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


def test_fixture_bind_attestation_is_path_free_read_only_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_fixture_bind_attestation(
        tmp_path,
        monkeypatch,
        run_mode="health",
        attempt=12,
        job_id=316846,
    )
    commitment = MODULE._tuple_fixture_bind_attestation(
        path,
        _config(),
        run_mode="health",
        attempt=12,
    )
    assert commitment == MODULE.file_digest(path)
    value = json.loads(path.read_text())
    assert value["path_field_count"] == 0
    assert value["read_only"] is True
    assert not any(
        "root" in key or "path" in key
        for key in value
        if key != "path_field_count"
    )

    for key, changed in (("read_only", False), ("path_field_count", 1)):
        tampered = dict(value)
        tampered[key] = changed
        path.write_bytes(MODULE.canonical(tampered) + b"\n")
        with pytest.raises(
            RuntimeError,
            match="E_TUPLE_HEALTH_FIXTURE_BIND_ATTESTATION",
        ):
            MODULE._tuple_fixture_bind_attestation(
                path,
                _config(),
                run_mode="health",
                attempt=12,
            )
    path.write_bytes(MODULE.canonical(value) + b"\n")


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
        "cuda", topology_attestation=attestation, attempt=4, cfg=_config()
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
    historical_config.pop("learner_effective_engineering_health_scheduler_policy")
    historical_config.pop("learner_effective_engineering_health_attempt_8_result")
    historical_config.pop("learner_effective_engineering_health_git_fallback_repair")
    historical_config.pop("learner_effective_engineering_health_attempt_9_result")
    historical_config.pop(
        "learner_effective_engineering_health_historical_lineage_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_10_result")
    historical_config.pop(
        "learner_effective_engineering_health_portable_AST_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_11_result")
    historical_config.pop(
        "learner_effective_engineering_health_fixture_bind_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_12_result")
    historical_config.pop(
        "learner_effective_engineering_health_submission_export_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_13_result")
    historical_config.pop(
        "learner_effective_engineering_health_NLTK_matplotlib_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_14_result")
    historical_config.pop(
        "learner_effective_engineering_health_generic_GRES_serialization_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_15_result")
    historical_config.pop(
        "learner_effective_engineering_health_active_dispatch_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_16_result")
    historical_config.pop(
        "learner_effective_engineering_health_historical_resource_dispatch_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_17_result")
    historical_config.pop(
        "learner_effective_engineering_health_ffmpeg_prettytable_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_18_result")
    historical_config.pop(
        "learner_effective_engineering_health_historical_full_result_lineage_repair"
    )
    historical_config.pop("learner_effective_engineering_health_attempt_19_result")
    historical_config.pop(
        "learner_effective_engineering_health_grounding_state_compatibility_repair"
    )
    historical_config.pop("learner_effective_engineering_health_pass_result")
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
    monkeypatch.setattr(
        MODULE,
        "_public_development_truth_mask_roundtrip_repair",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        MODULE,
        "_public_development_attribute_dependency_repair",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        MODULE,
        "_public_development_terminal_result",
        lambda _cfg: None,
    )
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
    fixture_bind_attestation = _write_fixture_bind_attestation(
        tmp_path / "development-fixture-bind-attestation",
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
            fixture_bind_attestation=fixture_bind_attestation,
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
