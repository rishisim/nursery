from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap

import pytest
import yaml

import babyworld_lite.corrective_alignment_v3.parallel as parallel_module
import babyworld_lite.corrective_alignment_v3.integrity as integrity_module
from babyworld_lite.corrective_alignment_v3.integrity import (
    CLOSED_PROTOCOL_DESELECTION,
    _verify_anticipated_launch_manifest,
    freeze_prefreeze_evidence,
    operation_identifier_audit,
    verify_prefreeze_evidence,
)
from babyworld_lite.corrective_alignment_v3.adjudicator import (
    AUTHORIZATION_FIELDS,
    PACKAGE_GATES,
    authorization_payload,
    verify_authorization,
)
from babyworld_lite.corrective_alignment_v3.statistics import dependence_diagnostics
from babyworld_lite.corrective_alignment_v3.generator import generate_corpus
from babyworld_lite.corrective_alignment_v3.learner import (
    disagreement_mechanism_qualification,
    fit_corrective_mil,
    predict_without_side,
)
from babyworld_lite.corrective_alignment_v3.parallel import (
    _candidate_scientific_decision,
    _development_validity_audit,
    _execute_development_cohort,
    _execute_worker_batches,
    _prepare_subprocess_requests,
    _validate_subprocess_failure,
    _validate_shard,
    _validate_shard_inventory,
    _verify_worker_persisted_input_commitment,
    build_work_plan,
    create_worker_boundary_commitment,
    execute_cohort,
    prepare_persisted_inputs,
    run_parallel_from_persisted,
    validate_subprocess_scheduler_artifacts,
    verify_worker_boundary_commitment,
)
from babyworld_lite.corrective_alignment_v3.protocol import (
    AuthorizationCapability,
    IdentifierFirewall,
    IdentifierReference,
    atomic_rename_directory_noreplace,
    _issue_worker_development_capability,
    canonical_digest,
    issue_development_capability,
    load_config,
    manifest_for_paths,
    manifest_for_tree,
    read_json,
    registry_snapshot,
    require_lstat_absent,
    sha256_file,
    tree_file_paths,
    verify_exact_manifest,
    verify_development_capability,
    verify_ledger,
    write_json,
)
import scripts.run_synthetic_corrective_alignment_v3 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/synthetic_corrective_alignment_v3.yaml"
EVIDENCE_ROOT = ROOT.parents[2] if ROOT.name == "frozen_source_snapshot" else ROOT
SCIENTIFIC_CORE_RELATIVES = (
    "babyworld_lite/corrective_alignment_v3/generator.py",
    "babyworld_lite/corrective_alignment_v3/learner.py",
    "babyworld_lite/corrective_alignment_v3/statistics.py",
)


def _copy_scientific_core(destination_root: Path) -> None:
    for relative in SCIENTIFIC_CORE_RELATIVES:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG, repository_root=ROOT)


def _firewall(config, purpose="construction_micro"):
    return IdentifierFirewall(config, purpose=purpose)


def _child_environment_base(config):
    return parallel_module._child_environment_base(
        inherited_allowlist=config["parallel"][
            "child_inherited_environment_allowlist"
        ],
        thread_environment=config["parallel"]["thread_environment"],
    )


def _make_prequalification_lock_fixture(tmp_path, monkeypatch, config):
    root = tmp_path / "repository"
    package = root / "output/synthetic_corrective_development_launch_package_v3"
    package.mkdir(parents=True)
    config_path = root / "configs/synthetic_corrective_alignment_v3.yaml"
    config_path.parent.mkdir(parents=True)
    config_bytes = CONFIG.read_bytes()
    frozen_status = b"  status: frozen\n"
    prefreeze_status = b"  status: pre_freeze\n"
    if config_bytes.count(frozen_status) == 1:
        config_bytes = config_bytes.replace(frozen_status, prefreeze_status, 1)
    elif config_bytes.count(prefreeze_status) != 1:
        raise AssertionError("fixture source config has no unique protocol status")
    config_path.write_bytes(config_bytes)
    runner_path = root / "scripts/run_synthetic_corrective_alignment_v3.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# locked runner\n", encoding="utf-8")
    _copy_scientific_core(root)
    rationale = root / "docs/locked-rationale.md"
    rationale.parent.mkdir(parents=True)
    rationale.write_text("locked rationale\n", encoding="utf-8")
    tracked = (
        "scripts/run_synthetic_corrective_alignment_v3.py",
        "configs/synthetic_corrective_alignment_v3.yaml",
        "docs/locked-rationale.md",
    )
    monkeypatch.setattr(runner, "TRACKED", tracked)
    fixture_config = copy.deepcopy(config)
    fixture_config["protocol"]["status"] = "pre_freeze"
    fixture_config["repository_root"] = str(root)
    prior_relative = str(
        fixture_config["registries"]["canonical_prior_registry"]
    )
    prior_source = ROOT / prior_relative
    prior_destination = root / prior_relative
    prior_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prior_source, prior_destination)
    write_json(
        package / "outcome_registry.json",
        {
            "schema_version": "nursery-corrective-outcome-registry-v3",
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
            "development_output_exists": False,
            "confirmation_output_exists": False,
            "authorization_consumed": False,
        },
    )
    write_json(
        package / "preserved_attempt_history.json",
        {"status": "FAILED_NO_OUTCOME", "scientific_outcome": False},
    )
    qualification_environment = {
        "schema_version": "test-environment-lock-v1",
        "python": "fixture",
    }
    monkeypatch.setattr(
        runner,
        "environment_record",
        lambda _root, _config: copy.deepcopy(qualification_environment),
    )
    for name, value in fixture_config["parallel"]["thread_environment"].items():
        monkeypatch.setenv(str(name), str(value))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return {
        "root": root,
        "package": package,
        "config_path": config_path,
        "runner_path": runner_path,
        "config": fixture_config,
        "environment": qualification_environment,
    }


def _make_snapshot_receipt_fixture(base: Path):
    package = base / "package"
    package.mkdir(parents=True)
    source = base / "source"
    runner_relative = "scripts/run_synthetic_corrective_alignment_v3.py"
    prior_relative = "output/synthetic_development_launch_package_v3/frozen_seed_registries.json"
    source_runner = source / runner_relative
    source_prior = source / prior_relative
    source_runner.parent.mkdir(parents=True)
    source_prior.parent.mkdir(parents=True)
    source_runner.write_bytes((ROOT / runner_relative).read_bytes())
    source_prior.write_bytes((ROOT / prior_relative).read_bytes())
    _copy_scientific_core(source)

    config_bytes = CONFIG.read_bytes()
    if config_bytes.count(b"  status: pre_freeze\n") == 1:
        frozen_config_bytes = config_bytes.replace(
            b"  status: pre_freeze\n", b"  status: frozen\n", 1
        )
    else:
        assert config_bytes.count(b"  status: frozen\n") == 1
        frozen_config_bytes = config_bytes
    environment = {
        "schema_version": "test-frozen-environment-v1",
        "python_executable": sys.executable,
    }
    design_lock = {
        "schema_version": "nursery-corrective-prequalification-design-lock-v3",
        "status": "LOCKED_BEFORE_OFFICIAL_FIXTURES",
        "tracked_nonconfig_manifest": manifest_for_paths(
            source,
            [runner_relative, prior_relative, *SCIENTIFIC_CORE_RELATIVES],
        ),
        "anticipated_frozen_config_sha256": canonical_digest(
            frozen_config_bytes.hex()
        ),
        "qualification_environment": environment,
    }
    # The verifier expects a real SHA-256 for this particular byte anchor.
    import hashlib

    design_lock["anticipated_frozen_config_sha256"] = hashlib.sha256(
        frozen_config_bytes
    ).hexdigest()
    write_json(package / "prequalification_design_lock.json", design_lock)
    write_json(
        package / "micro_attempt_consumed.json",
        {"status": "ONE_MICRO_ATTEMPT_CONSUMED_BEFORE_LAUNCHER_PREFLIGHT"},
    )
    write_json(package / "worker_launcher_preflight.json", {"status": "PASS"})
    write_json(package / "locked_evidence.json", {"status": "PASS"})
    freeze_prefreeze_evidence(package)

    snapshot = package / "frozen_source_snapshot"
    snapshot_runner = snapshot / runner_relative
    snapshot_config = snapshot / "configs/synthetic_corrective_alignment_v3.yaml"
    snapshot_prior = snapshot / prior_relative
    snapshot_runner.parent.mkdir(parents=True)
    snapshot_config.parent.mkdir(parents=True)
    snapshot_prior.parent.mkdir(parents=True)
    snapshot_runner.write_bytes(source_runner.read_bytes())
    snapshot_config.write_bytes(frozen_config_bytes)
    snapshot_prior.write_bytes(source_prior.read_bytes())
    _copy_scientific_core(snapshot)
    manifest = manifest_for_tree(snapshot)
    write_json(snapshot / "snapshot_manifest.json", manifest)
    frozen_config = load_config(snapshot_config, repository_root=snapshot)
    write_json(package / "frozen_environment.json", environment)
    write_json(package / "frozen_seed_registries.json", registry_snapshot(frozen_config))
    prefreeze_contract = read_json(package / "prefreeze_evidence_contract.json")
    receipt = {
        "schema_version": "nursery-corrective-freeze-receipt-v3",
        "status": "FROZEN",
        "protocol_id": integrity_module.PROTOCOL_ID,
        "config_status": "frozen",
        "snapshot_manifest_sha256": sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        "snapshot_file_count": manifest["file_count"],
        "snapshot_digest": manifest["digest"],
        "frozen_config_sha256": sha256_file(snapshot_config),
        "frozen_runner_sha256": sha256_file(snapshot_runner),
        "frozen_environment_sha256": sha256_file(
            package / "frozen_environment.json"
        ),
        "frozen_registries_sha256": sha256_file(
            package / "frozen_seed_registries.json"
        ),
        "prequalification_design_lock_sha256": sha256_file(
            package / "prequalification_design_lock.json"
        ),
        "micro_attempt_receipt_sha256": sha256_file(
            package / "micro_attempt_consumed.json"
        ),
        "worker_launcher_preflight_sha256": sha256_file(
            package / "worker_launcher_preflight.json"
        ),
        "prefreeze_evidence_manifest_sha256": sha256_file(
            package / "prefreeze_evidence_manifest.json"
        ),
        "prefreeze_evidence_contract_sha256": sha256_file(
            package / "prefreeze_evidence_contract.json"
        ),
        "prefreeze_evidence_digest": prefreeze_contract["manifest_digest"],
        "prefreeze_evidence_file_count": prefreeze_contract["file_count"],
        "tracked_files": sorted(row["path"] for row in manifest["files"]),
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "confirmation_authorized": False,
    }
    write_json(package / "freeze_receipt.json", receipt)
    return package, snapshot, receipt


def test_project_module_origins_are_confined_to_the_selected_source_root():
    expected = {
        "babyworld_lite.corrective_alignment_v3.protocol",
        "babyworld_lite.corrective_alignment_v3.generator",
        "babyworld_lite.corrective_alignment_v3.learner",
        "babyworld_lite.corrective_alignment_v3.statistics",
        "babyworld_lite.corrective_alignment_v3.parallel",
        "babyworld_lite.corrective_alignment_v3.adjudicator",
        "babyworld_lite.corrective_alignment_v3.integrity",
    }
    for name in expected:
        module = sys.modules[name]
        assert Path(module.__file__).resolve().is_relative_to(ROOT.resolve())


def test_prequalification_lock_rejects_source_config_and_registry_mutations(
    tmp_path, monkeypatch, config
):
    fixture = _make_prequalification_lock_fixture(tmp_path, monkeypatch, config)
    lock = runner._create_prequalification_design_lock(
        fixture["root"], fixture["package"], fixture["config"]
    )
    assert lock["status"] == "LOCKED_BEFORE_OFFICIAL_FIXTURES"
    assert lock["scientific_core_verification"]["status"] == "PASS"
    prefreeze = runner._verify_prequalification_design_lock(
        fixture["root"],
        fixture["package"],
        fixture["config"],
        require_fixture_contracts=False,
    )
    assert prefreeze["status"] == "PASS"

    original_runner = fixture["runner_path"].read_bytes()
    fixture["runner_path"].write_bytes(original_runner + b"# mutation\n")
    assert runner._verify_prequalification_design_lock(
        fixture["root"],
        fixture["package"],
        fixture["config"],
        require_fixture_contracts=False,
    )["status"] == "FAIL"
    fixture["runner_path"].write_bytes(original_runner)

    anticipated_frozen = runner._anticipated_frozen_config_bytes(
        fixture["config_path"]
    )
    frozen_config = copy.deepcopy(fixture["config"])
    frozen_config["protocol"]["status"] = "frozen"
    fixture["config_path"].write_bytes(anticipated_frozen)
    assert runner._verify_prequalification_design_lock(
        fixture["root"],
        fixture["package"],
        frozen_config,
        require_fixture_contracts=False,
    )["status"] == "PASS"

    changed_config = copy.deepcopy(frozen_config)
    changed_config["learner"]["epochs"] += 1
    assert runner._verify_prequalification_design_lock(
        fixture["root"],
        fixture["package"],
        changed_config,
        require_fixture_contracts=False,
    )["status"] == "FAIL"

    registry_path = fixture["package"] / "outcome_registry.json"
    registry = read_json(registry_path)
    registry["development_outcome_count"] = 1
    write_json(registry_path, registry, overwrite=True)
    assert runner._verify_prequalification_design_lock(
        fixture["root"],
        fixture["package"],
        frozen_config,
        require_fixture_contracts=False,
    )["status"] == "FAIL"


def test_prequalification_lock_refuses_any_prior_official_attempt_path(
    tmp_path, monkeypatch, config
):
    fixture = _make_prequalification_lock_fixture(tmp_path, monkeypatch, config)
    forbidden = fixture["package"] / "parallel_micro_attempt_1_jobs_1"
    forbidden.mkdir()
    with pytest.raises(RuntimeError, match="predate design lock"):
        runner._create_prequalification_design_lock(
            fixture["root"], fixture["package"], fixture["config"]
        )


@pytest.mark.parametrize("mutation", ["extra_file", "empty_directory", "symlink"])
def test_prequalification_lock_binds_prior_attempt_directory_inventory(
    tmp_path, monkeypatch, config, mutation
):
    fixture = _make_prequalification_lock_fixture(tmp_path, monkeypatch, config)
    prior_attempt = fixture["package"] / "prior_attempt"
    prior_attempt.mkdir()
    write_json(prior_attempt / "evidence.json", {"status": "PRESERVED"})
    runner._create_prequalification_design_lock(
        fixture["root"], fixture["package"], fixture["config"]
    )
    if mutation == "extra_file":
        write_json(prior_attempt / "extra.json", {"unexpected": True})
    elif mutation == "empty_directory":
        (prior_attempt / "empty").mkdir()
    else:
        (prior_attempt / "link").symlink_to("evidence.json")
    assert runner._verify_prequalification_design_lock(
        fixture["root"],
        fixture["package"],
        fixture["config"],
        require_fixture_contracts=False,
    )["status"] == "FAIL"


def test_registry_allocation_is_fresh_disjoint_and_confirmation_sealed(
    tmp_path, config
):
    registries = config["resolved_registries"]
    sets = {
        name: {value for values in registry.values() for value in values}
        for name, registry in registries.items()
    }
    assert all(
        not sets[left] & sets[right]
        for index, left in enumerate(sets)
        for right in list(sets)[index + 1 :]
    )
    assert len(registries["development"]["corpus"]) == 100
    assert len(registries["development"]["model"]) == 3
    assert len(registries["confirmation_reserve"]["corpus"]) == 100
    assert all(340000 <= value <= 349999 for values in sets.values() for value in values)
    assert not any(
        281000 <= value <= 281999 or 289000 <= value <= 289999
        for values in sets.values()
        for value in values
    )
    assert not config["firewalls"]["confirmation_command_exists"]
    assert not config["firewalls"]["confirmation_execution_supported"]
    for field, value in (
        ("minimum_benchmark_speedup", 1.0),
        ("resource_contingency_multiplier", 1.0),
        ("maximum_host_ram_fraction", 0.9),
        ("maximum_current_available_ram_fraction", 0.9),
        ("maximum_current_free_disk_fraction", 0.9),
        ("minimum_logical_cpu_count", 1),
    ):
        repository = tmp_path / field
        config_path = repository / "configs/mutated.yaml"
        config_path.parent.mkdir(parents=True)
        prior_relative = str(config["registries"]["canonical_prior_registry"])
        prior_destination = repository / prior_relative
        prior_destination.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / prior_relative, prior_destination)
        mutated = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        mutated["parallel"][field] = value
        config_path.write_text(
            yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="direct-subprocess backend contract"):
            load_config(config_path, repository_root=repository)


def test_development_has_no_boolean_authorization_bypass(config):
    with pytest.raises(PermissionError):
        IdentifierFirewall(config, purpose="development", capability=True)  # type: ignore[arg-type]
    with pytest.raises((PermissionError, TypeError)):
        AuthorizationCapability(
            purpose="development",
            authorization_digest="a" * 64,
            parsed_contract_digest="b" * 64,
            claim_path="/tmp/forged",
            claim_sha256="c" * 64,
            resolved_output_root="/tmp/forged-output",
            required_jobs=4,
        )
    with pytest.raises(PermissionError, match="generic cohort API"):
        execute_cohort(
            repository_root=ROOT,
            config_path=CONFIG,
            output_root=ROOT / "output/forbidden-direct-development-test",
            config=config,
            purpose="development",
            jobs=4,
            capability=None,
        )


def test_work_plan_is_exact_unique_cross_product(config):
    plan = build_work_plan(config, purpose="construction_micro")
    expected = len(config["resolved_registries"]["construction_micro"]["corpus"]) * len(
        config["resolved_registries"]["construction_micro"]["model"]
    )
    assert plan["unit_count"] == expected
    assert len({row["unit_id"] for row in plan["units"]}) == expected
    assert all(
        row["condition_order"] == config["design"]["conditions"]
        for row in plan["units"]
    )


@pytest.mark.parametrize(
    "purpose,validity,primary,informativeness,dependence,causal,expected",
    [
        ("excluded_rehearsal", "NOT_APPLICABLE", "PASS", "PASS", "PASS", "PASS", "SCIENTIFIC_INFERENCE_SUPPRESSED"),
        ("development", "FAIL", "PASS", "PASS", "PASS", "PASS", "REVISE_INVALID_DEVELOPMENT_COHORT"),
        ("development", "PASS", "FAIL", "PASS", "PASS", "FAIL", "CORRECTIVE_STUDY_STOP_NO_SUPPORT"),
        ("development", "PASS", "PASS", "FAIL", "PASS", "PASS", "REVISE_UNINFORMATIVE"),
        ("development", "PASS", "PASS", "PASS", "FAIL", "PASS", "REVISE_UNINFORMATIVE"),
        ("development", "PASS", "PASS", "PASS", "PASS", "FAIL", "REVISE_CAUSAL_ATTRIBUTION_FAILED"),
        ("development", "PASS", "PASS", "PASS", "PASS", "PASS", "GO"),
    ],
)
def test_scientific_decision_maps_controls_dependence_and_primary_status(
    purpose, validity, primary, informativeness, dependence, causal, expected
):
    assert _candidate_scientific_decision(
        purpose=purpose,
        development_validity_status=validity,
        primary_status=primary,
        informativeness_status=informativeness,
        dependence_status=dependence,
        causal_attribution_status=causal,
    ) == expected


def test_development_validity_rechecks_heterogeneity_and_model_stochasticity(
    config,
):
    corpus_seeds = list(config["resolved_registries"]["development"]["corpus"])
    model_seeds = list(config["resolved_registries"]["development"]["model"])
    conditions = list(config["design"]["conditions"])
    corpus_audits = []
    for index, corpus_seed in enumerate(corpus_seeds):
        stratum = ("correction", "ambiguous", "recoverable")[index % 3]
        stratum_rank = index // 3
        foil_rate_draw = {
            "correction": 0.88 + 0.002 * stratum_rank,
            "ambiguous": 0.70 + 0.003 * stratum_rank,
            "recoverable": 0.35 + 0.005 * stratum_rank,
        }[stratum]
        corpus_audits.append(
            {
                "corpus_seed": corpus_seed,
                "action_geometry_digest": canonical_digest(["action", index]),
                "episode_geometry_digest": canonical_digest(["episode", index]),
                "heldout_composition_digest": canonical_digest(["split", index]),
                "repetition_count_digest": canonical_digest(["repetition", index]),
                "candidate_count_histogram_digest": canonical_digest(
                    ["candidates", index]
                ),
                "ambiguity_stratum": stratum,
                "grounded_episode_count": 60 + index,
                "episode_count": 200,
                "realized_foil_present_rate": 0.40 + 0.001 * index,
                "observed_lag_mean": float(index),
                "realized_mean_event_observation_peak": 0.20 + 0.001 * index,
                "grounded_rate_draw": 0.58 + 0.0024 * index,
                "foil_rate_draw": foil_rate_draw,
                "visibility_draw": 0.52 + 0.0038 * index,
                "lag_mean_draw": -12.0 + 0.24 * index,
                "lag_sd_draw": 3.0 + 0.05 * index,
                "side_informativity_draw": 0.65 + 0.0027 * index,
                "side_noise_draw": 0.04 + 0.001 * index,
                "action_geometry": {
                    "all_rows_normalized": True,
                    "all_entries_nonnegative": True,
                    "all_margins_pass": True,
                    "slots": {
                        slot: {
                            "mean_pairwise_distance": 0.10 + 0.001 * index,
                            "minimum_self_cross_dot_margin": 0.30,
                        }
                        for slot in ("primitive", "manner")
                    },
                },
                "train_evaluation_use_same_action_prototypes": True,
                "condition_audit": {
                    "shuffle_preserves_learner_visible_evidence_marginal_by_block": True,
                    "detector_manipulation": {
                        "synchronized": {
                            "strict_event_or_null_accuracy": 0.50
                            + 0.001 * index
                        }
                    },
                },
                "visible_identifiers_are_opaque_and_truth_independent_in_format": True,
                "train_evaluation_generator_namespaces_disjoint": True,
                "provenance": {"train_evaluation_instance_id_overlap": 0},
                "every_constituent_exposed_in_training": True,
                "all_training_candidate_events_exclude_heldout_compositions": True,
                "heldout_candidate_event_count": 0,
                "evaluation_side_fields_absent": True,
                "evaluation_top_level_schema_exact_allowlist": True,
            }
        )
    merged_results = []
    for corpus_seed in corpus_seeds:
        for model_seed in model_seeds:
            condition_rows = []
            for condition in conditions:
                model_offset = 0.01 * (model_seeds.index(model_seed) + 1)
                semantics = {}
                for slot, dimension in (
                    ("primitive", int(config["design"]["primitive_concepts"])),
                    ("manner", int(config["design"]["manner_concepts"])),
                ):
                    for token_index in range(dimension):
                        values = [1.0 / dimension] * dimension
                        values[0] += model_offset
                        values[1] -= model_offset
                        semantics[f"{slot}|opaque-{slot}-{token_index}"] = values
                model = {
                    "schema_version": "nursery-corrective-lexicon-model-v2",
                    "semantics": semantics,
                    "null_rates": {
                        key: 0.20 + model_offset for key in sorted(semantics)
                    },
                    "model_seed": model_seed,
                    "epochs": int(config["learner"]["epochs"]),
                    "learner": config["learner"]["name"],
                    "training_side_state_serialized": False,
                    "detector_serialized": False,
                    "oracle_serialized": False,
                    "evaluation_key_serialized": False,
                }
                model_digest = canonical_digest(model)
                epochs = int(config["learner"]["epochs"])
                epoch_parameters = [
                    canonical_digest(
                        ["parameter", corpus_seed, model_seed, condition, epoch]
                    )
                    for epoch in range(epochs)
                ]
                epoch_parameters[-1] = canonical_digest(
                    {
                        "semantics": model["semantics"],
                        "null_rates": model["null_rates"],
                    }
                )
                epoch_orders = [
                    canonical_digest(
                        ["order", corpus_seed, model_seed, condition, epoch]
                    )
                    for epoch in range(epochs)
                ]
                condition_rows.append(
                    {
                        "condition": condition,
                        "model": model,
                        "model_digest": model_digest,
                        "prediction_digest": canonical_digest(
                            ["prediction", corpus_seed, model_seed, condition]
                        ),
                        "trace": {
                            "schema_version": "nursery-corrective-training-trace-v2",
                            "condition": condition,
                            "mutation": "active",
                            "model_seed": model_seed,
                            "epochs": epochs,
                            "initial_parameter_digest": canonical_digest(
                                ["initial", corpus_seed, model_seed, condition]
                            ),
                            "epoch_parameter_digests": epoch_parameters,
                            "distinct_epoch_parameter_digests": epochs,
                            "epoch_order_digests": epoch_orders,
                            "distinct_epoch_order_digests": epochs,
                            "semantic_side_weight_effective": float(
                                config["learner"]["semantic_side_weight"]
                            ),
                            "null_side_weight_effective": float(
                                config["learner"]["null_side_weight"]
                            ),
                            "null_update_frozen_at_initial_prior": False,
                            "disagreement_gate": False,
                            "argmax_prefilter": False,
                            "all_candidate_events_enter_posterior": True,
                            "semantic_disagreement_count": 1,
                            "side_changed_event_top_count": 1,
                            "mean_null_posterior": 0.2,
                            "mean_event_posterior_mass": 0.8,
                            "minimum_event_posterior_mass": 0.5,
                            "semantic_update_scaled_by_event_mass": True,
                            "mean_semantic_update_l1": 0.1,
                            "mean_null_update_absolute": 0.1,
                            "model_digest": model_digest,
                            "oracle_fields_consumed": False,
                            "evaluation_keys_consumed": False,
                        },
                    }
                )
            merged_results.append(
                {
                    "corpus_seed": corpus_seed,
                    "model_seed": model_seed,
                    "condition_results": condition_rows,
                }
            )
    averaged_metric_names = {
        "lexical_acquisition_top1",
        "heldout_composition_top1",
        "presence_balanced_accuracy",
    }
    for kind in ("lexical", "composition", "presence", "zero_exposure"):
        for field in (
            "strict_top1",
            "mean_rank",
            "mrr",
            "multiclass_log_loss",
            "multiclass_brier",
            "tie_rate",
            "exact_chance",
        ):
            averaged_metric_names.add(f"{kind}.{field}")
    averaged_metric_names.update(
        {"presence.present.strict_top1", "presence.null.strict_top1"}
    )

    def averaged_metrics():
        values = {name: 0.25 for name in averaged_metric_names}
        for name in (
            "lexical_acquisition_top1",
            "heldout_composition_top1",
            "presence_balanced_accuracy",
        ):
            values[name] = 0.11
        values["zero_exposure.strict_top1"] = 0.0
        values["zero_exposure.tie_rate"] = 1.0
        return values

    variation_values = [0.10, 0.11, 0.12]
    averaged = {
        "schema_version": "nursery-corrective-model-average-v2",
        "independent_unit": "corpus_seed",
        "model_seed_order": model_seeds,
        "exact_model_seed_set_required": True,
        "floating_reduction": "math.fsum_in_frozen_model_seed_order",
        "rows": [
            {
                "corpus_seed": corpus_seed,
                "condition": condition,
                "model_seed_order": model_seeds,
                "model_replicates_averaged": len(model_seeds),
                "metrics": averaged_metrics(),
            }
            for corpus_seed in corpus_seeds
            for condition in conditions
        ],
        "within_corpus_model_variation": [
            {
                "corpus_seed": corpus_seed,
                "condition": condition,
                "metric": metric,
                "sample_sd": 0.01,
                "range": 0.02,
                "values": variation_values,
            }
            for corpus_seed in corpus_seeds
            for condition in conditions
            for metric in (
                "lexical_acquisition_top1",
                "heldout_composition_top1",
                "presence_balanced_accuracy",
            )
        ],
    }
    result = _development_validity_audit(
        purpose="development",
        averaged=averaged,
        merged_results=merged_results,
        corpus_audits=corpus_audits,
        config=config,
    )
    assert result["status"] == "PASS", {
        name: value for name, value in result["checks"].items() if not value
    }
    assert result["checks"][
        "initialization_order_and_fitted_states_all_vary"
    ]
    assert result["scaled_distinct_minima"] == {
        "minimum_distinct_action_geometry_digests": 75,
        "minimum_distinct_episode_geometry_digests": 75,
        "minimum_distinct_heldout_split_digests": 50,
        "minimum_distinct_repetition_digests": 75,
        "minimum_distinct_candidate_count_histograms": 25,
        "minimum_distinct_grounded_episode_counts": 25,
    }
    assert config["development_validity_gates"] == {
        "minimum_distinct_factor_draws": 75,
        "factor_draw_reference_fraction": 0.75,
        "require_factor_draws_within_frozen_ranges": True,
        "require_trace_state_digest_binding": True,
        "require_variation_values_bound_to_averages": True,
    }
    assert result["factor_draw_distinct_minimum"] == 75
    reordered_averaged = copy.deepcopy(averaged)
    reordered_averaged["rows"].reverse()
    reordered_averaged["within_corpus_model_variation"].reverse()
    reordered = _development_validity_audit(
        purpose="development",
        averaged=reordered_averaged,
        merged_results=list(reversed(merged_results)),
        corpus_audits=list(reversed(corpus_audits)),
        config=config,
    )
    assert canonical_digest(reordered) == canonical_digest(result)
    changed_audits = copy.deepcopy(corpus_audits)
    for row in changed_audits:
        row["visibility_draw"] = 0.6
    assert _development_validity_audit(
        purpose="development",
        averaged=averaged,
        merged_results=merged_results,
        corpus_audits=changed_audits,
        config=config,
    )["status"] == "FAIL"
    string_boolean_audits = copy.deepcopy(corpus_audits)
    string_boolean_audits[0]["evaluation_side_fields_absent"] = "False"
    assert _development_validity_audit(
        purpose="development",
        averaged=averaged,
        merged_results=merged_results,
        corpus_audits=string_boolean_audits,
        config=config,
    )["status"] == "FAIL"
    out_of_range_audits = copy.deepcopy(corpus_audits)
    out_of_range_audits[0]["grounded_rate_draw"] = -1.0
    with pytest.raises(RuntimeError, match="outside its frozen range"):
        _development_validity_audit(
            purpose="development",
            averaged=averaged,
            merged_results=merged_results,
            corpus_audits=out_of_range_audits,
            config=config,
        )
    wrong_stratum_audits = copy.deepcopy(corpus_audits)
    wrong_stratum_audits[0]["ambiguity_stratum"] = "forged"
    with pytest.raises(RuntimeError, match="ambiguity stratum"):
        _development_validity_audit(
            purpose="development",
            averaged=averaged,
            merged_results=merged_results,
            corpus_audits=wrong_stratum_audits,
            config=config,
        )
    two_level_audits = copy.deepcopy(corpus_audits)
    for index, row in enumerate(two_level_audits):
        high = bool(index % 2)
        row["grounded_rate_draw"] = 0.70 if high else 0.60
        row["visibility_draw"] = 0.80 if high else 0.60
        row["lag_mean_draw"] = 5.0 if high else -5.0
        row["lag_sd_draw"] = 6.0 if high else 4.0
        row["side_informativity_draw"] = 0.85 if high else 0.70
        row["side_noise_draw"] = 0.10 if high else 0.06
        row["foil_rate_draw"] = {
            "correction": 0.94 if high else 0.90,
            "ambiguous": 0.80 if high else 0.72,
            "recoverable": 0.50 if high else 0.40,
        }[row["ambiguity_stratum"]]
    two_level_result = _development_validity_audit(
        purpose="development",
        averaged=averaged,
        merged_results=merged_results,
        corpus_audits=two_level_audits,
        config=config,
    )
    assert two_level_result["status"] == "FAIL"
    assert two_level_result["factor_draw_distinct_minimum"] == 75
    assert not two_level_result["checks"]["all_frozen_factor_draws_vary"]
    duplicate_audits = [*copy.deepcopy(corpus_audits[:-1]), copy.deepcopy(corpus_audits[0])]
    with pytest.raises(RuntimeError, match="corpus-audit inventory"):
        _development_validity_audit(
            purpose="development",
            averaged=averaged,
            merged_results=merged_results,
            corpus_audits=duplicate_audits,
            config=config,
        )
    changed_results = copy.deepcopy(merged_results)
    target_corpus = corpus_seeds[0]
    for row in changed_results:
        if row["corpus_seed"] == target_corpus:
            synchronized = next(
                child
                for child in row["condition_results"]
                if child["condition"] == "synchronized"
            )
            synchronized["trace"]["initial_parameter_digest"] = "0" * 64
    assert _development_validity_audit(
        purpose="development",
        averaged=averaged,
        merged_results=changed_results,
        corpus_audits=corpus_audits,
        config=config,
    )["status"] == "FAIL"
    metadata_only_results = copy.deepcopy(merged_results)
    reference_state = None
    for row in metadata_only_results:
        if row["corpus_seed"] != target_corpus:
            continue
        synchronized = next(
            child
            for child in row["condition_results"]
            if child["condition"] == "synchronized"
        )
        if reference_state is None:
            reference_state = {
                "semantics": copy.deepcopy(synchronized["model"]["semantics"]),
                "null_rates": copy.deepcopy(synchronized["model"]["null_rates"]),
            }
        synchronized["model"]["semantics"] = copy.deepcopy(
            reference_state["semantics"]
        )
        synchronized["model"]["null_rates"] = copy.deepcopy(
            reference_state["null_rates"]
        )
        synchronized["model_digest"] = canonical_digest(synchronized["model"])
        synchronized["trace"]["model_digest"] = synchronized["model_digest"]
        synchronized["trace"]["epoch_parameter_digests"][-1] = canonical_digest(
            reference_state
        )
        synchronized["trace"]["distinct_epoch_parameter_digests"] = len(
            set(synchronized["trace"]["epoch_parameter_digests"])
        )
    assert _development_validity_audit(
        purpose="development",
        averaged=averaged,
        merged_results=metadata_only_results,
        corpus_audits=corpus_audits,
        config=config,
    )["status"] == "FAIL"
    malformed_results = copy.deepcopy(merged_results)
    synchronized = next(
        child
        for child in malformed_results[0]["condition_results"]
        if child["condition"] == "synchronized"
    )
    synchronized["trace"].pop("evaluation_keys_consumed")
    with pytest.raises(RuntimeError, match="training trace contract"):
        _development_validity_audit(
            purpose="development",
            averaged=averaged,
            merged_results=malformed_results,
            corpus_audits=corpus_audits,
            config=config,
        )
    malformed_dimension_results = copy.deepcopy(merged_results)
    synchronized = next(
        child
        for child in malformed_dimension_results[0]["condition_results"]
        if child["condition"] == "synchronized"
    )
    primitive_key = next(
        key for key in synchronized["model"]["semantics"] if key.startswith("primitive|")
    )
    synchronized["model"]["semantics"][primitive_key] = [0.5, 0.5]
    synchronized["model_digest"] = canonical_digest(synchronized["model"])
    synchronized["trace"]["model_digest"] = synchronized["model_digest"]
    synchronized["trace"]["epoch_parameter_digests"][-1] = canonical_digest(
        {
            "semantics": synchronized["model"]["semantics"],
            "null_rates": synchronized["model"]["null_rates"],
        }
    )
    with pytest.raises(RuntimeError, match="training trace contract"):
        _development_validity_audit(
            purpose="development",
            averaged=averaged,
            merged_results=malformed_dimension_results,
            corpus_audits=corpus_audits,
            config=config,
        )
    impossible_trace_results = copy.deepcopy(merged_results)
    synchronized = next(
        child
        for child in impossible_trace_results[0]["condition_results"]
        if child["condition"] == "synchronized"
    )
    synchronized["trace"].update(
        {
            "semantic_side_weight_effective": 0.0,
            "null_side_weight_effective": 0.0,
            "null_update_frozen_at_initial_prior": True,
            "disagreement_gate": True,
            "semantic_disagreement_count": 0,
            "side_changed_event_top_count": 0,
            "mean_null_posterior": math.nan,
            "mean_event_posterior_mass": math.nan,
            "minimum_event_posterior_mass": math.nan,
            "mean_semantic_update_l1": 0.0,
            "mean_null_update_absolute": 0.0,
        }
    )
    with pytest.raises(RuntimeError, match="training trace contract"):
        _development_validity_audit(
            purpose="development",
            averaged=averaged,
            merged_results=impossible_trace_results,
            corpus_audits=corpus_audits,
            config=config,
        )
    unbound_state_results = copy.deepcopy(merged_results)
    synchronized = next(
        child
        for child in unbound_state_results[0]["condition_results"]
        if child["condition"] == "synchronized"
    )
    synchronized["trace"]["epoch_parameter_digests"][-1] = "f" * 64
    with pytest.raises(RuntimeError, match="training trace contract"):
        _development_validity_audit(
            purpose="development",
            averaged=averaged,
            merged_results=unbound_state_results,
            corpus_audits=corpus_audits,
            config=config,
        )
    nonfinite_audits = copy.deepcopy(corpus_audits)
    for index, row in enumerate(nonfinite_audits):
        row["lag_sd_draw"] = float("nan") if index % 2 == 0 else float("inf")
        row["side_noise_draw"] = float("nan")
    with pytest.raises(RuntimeError, match="factor draws"):
        _development_validity_audit(
            purpose="development",
            averaged=averaged,
            merged_results=merged_results,
            corpus_audits=nonfinite_audits,
            config=config,
        )
    forged_variation = copy.deepcopy(averaged)
    forged_variation["within_corpus_model_variation"][0]["metric"] = "forged_metric"
    with pytest.raises(RuntimeError, match="model-variation inventory"):
        _development_validity_audit(
            purpose="development",
            averaged=forged_variation,
            merged_results=merged_results,
            corpus_audits=corpus_audits,
            config=config,
        )
    missing_variation = copy.deepcopy(averaged)
    missing_variation["within_corpus_model_variation"].pop()
    with pytest.raises(RuntimeError, match="model-variation inventory"):
        _development_validity_audit(
            purpose="development",
            averaged=missing_variation,
            merged_results=merged_results,
            corpus_audits=corpus_audits,
            config=config,
        )
    duplicate_variation = copy.deepcopy(averaged)
    duplicate_variation["within_corpus_model_variation"][-1] = copy.deepcopy(
        duplicate_variation["within_corpus_model_variation"][0]
    )
    with pytest.raises(RuntimeError, match="model-variation inventory"):
        _development_validity_audit(
            purpose="development",
            averaged=duplicate_variation,
            merged_results=merged_results,
            corpus_audits=corpus_audits,
            config=config,
        )
    self_consistent_unbound_variation = copy.deepcopy(averaged)
    changed_variation = self_consistent_unbound_variation[
        "within_corpus_model_variation"
    ][0]
    changed_variation["values"] = [0.20, 0.21, 0.22]
    changed_variation["sample_sd"] = 0.01
    changed_variation["range"] = 0.02
    with pytest.raises(RuntimeError, match="model-variation mean binding"):
        _development_validity_audit(
            purpose="development",
            averaged=self_consistent_unbound_variation,
            merged_results=merged_results,
            corpus_audits=corpus_audits,
            config=config,
        )
    missing_average_rows = copy.deepcopy(averaged)
    missing_average_rows["rows"] = []
    with pytest.raises(RuntimeError, match="averaged row inventory"):
        _development_validity_audit(
            purpose="development",
            averaged=missing_average_rows,
            merged_results=merged_results,
            corpus_audits=corpus_audits,
            config=config,
        )


def test_present_null_dependence_rejects_equal_vectors_even_if_lexical_differs(config):
    rows = []
    for seed, lexical_effect in ((1, 0.1), (2, 0.2), (3, 0.3)):
        rows.extend(
            [
                {
                    "corpus_seed": seed,
                    "condition": "absent",
                    "metrics": {
                        "lexical_acquisition_top1": 0.2,
                        "presence.present.strict_top1": 0.25,
                        "presence.null.strict_top1": 0.35,
                    },
                },
                {
                    "corpus_seed": seed,
                    "condition": "synchronized",
                    "metrics": {
                        "lexical_acquisition_top1": 0.2 + lexical_effect,
                        "presence.present.strict_top1": 0.35 + 0.05 * seed,
                        "presence.null.strict_top1": 0.45 + 0.05 * seed,
                    },
                },
                {
                    "corpus_seed": seed,
                    "condition": "shuffled",
                    "metrics": {
                        "lexical_acquisition_top1": 0.1,
                        "presence.present.strict_top1": 0.15,
                        "presence.null.strict_top1": 0.25,
                    },
                },
            ]
        )
    diagnostics = dependence_diagnostics({"rows": rows}, config)
    assert diagnostics["status"] == "FAIL"
    assert diagnostics["present_null_exact_vector_equality"] is True
    assert diagnostics[
        "synchronized_minus_absent_lexical_effects_secondary_diagnostic"
    ] != diagnostics["null_effects"]


def test_shard_inventory_rejects_duplicate_missing_extra_and_symlink_units(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    for unit_id in ("u1", "u2"):
        (shards / unit_id).mkdir()
    assert _validate_shard_inventory(shards, ["u1", "u2"]) == ["u1", "u2"]
    with pytest.raises(RuntimeError, match="duplicate expected"):
        _validate_shard_inventory(shards, ["u1", "u1"])
    (shards / "u2").rmdir()
    with pytest.raises(RuntimeError, match="completeness"):
        _validate_shard_inventory(shards, ["u1", "u2"])
    (shards / "extra").mkdir()
    with pytest.raises(RuntimeError, match="completeness"):
        _validate_shard_inventory(shards, ["u1"])
    (shards / "extra").rmdir()
    (shards / "linked").symlink_to("u1")
    with pytest.raises(RuntimeError, match="symlink shard entries"):
        _validate_shard_inventory(shards, ["u1"])
    root_link = tmp_path / "shards-link"
    root_link.symlink_to("shards")
    with pytest.raises(RuntimeError, match="real directory"):
        _validate_shard_inventory(root_link, ["u1"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("scientific_outcome", True),
        ("confirmation_outcome", True),
        ("development_outcome_count", 1),
    ],
)
def test_self_consistently_resealed_shard_outcome_mutations_fail_closed(
    tmp_path, config, field, value
):
    unit = build_work_plan(config, purpose="construction_micro")["units"][0]
    shard = tmp_path / "shard"
    shard.mkdir()
    result = {
        "schema_version": "nursery-corrective-worker-result-v3",
        "purpose": "construction_micro",
        "unit_id": unit["unit_id"],
        "corpus_seed": unit["corpus_seed"],
        "model_seed": unit["model_seed"],
        "input_manifest_sha256": "a" * 64,
        "condition_results": [],
        "mutation_results": [],
        "scientific_contract_digest": "b" * 64,
        "publication_state": "UNPUBLISHED_CANDIDATE",
        "scientific_outcome": False,
        "development_outcome_count": 0,
        "confirmation_outcome": False,
    }
    result[field] = value
    write_json(shard / "result.json", result)
    write_json(shard / "operation_ledger.json", {})
    manifest = manifest_for_paths(
        shard, ["result.json", "operation_ledger.json"]
    )
    write_json(shard / "manifest.json", manifest)
    write_json(
        shard / "COMPLETE.json",
        {
            "schema_version": "nursery-corrective-shard-complete-v3",
            "unit_id": unit["unit_id"],
            "result_sha256": sha256_file(shard / "result.json"),
            "ledger_sha256": sha256_file(shard / "operation_ledger.json"),
            "manifest_sha256": sha256_file(shard / "manifest.json"),
            "status": "COMPLETE",
        },
    )
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        _validate_shard(
            shard,
            unit,
            config,
            purpose="construction_micro",
            input_manifest_sha256="a" * 64,
            scientific_contract_digest="b" * 64,
            authorization_digest=None,
            parsed_contract_digest=None,
        )


@pytest.mark.parametrize(
    "missing", ["result.json", "operation_ledger.json", "manifest.json", "COMPLETE.json"]
)
def test_partial_shards_fail_before_adjudication(tmp_path, config, missing):
    unit = build_work_plan(config, purpose="construction_micro")["units"][0]
    shard = tmp_path / "shard"
    shard.mkdir()
    for name in {"result.json", "operation_ledger.json", "manifest.json", "COMPLETE.json"} - {
        missing
    }:
        write_json(shard / name, {})
    with pytest.raises(RuntimeError, match="exact file set"):
        _validate_shard(
            shard,
            unit,
            config,
            purpose="construction_micro",
            input_manifest_sha256="a" * 64,
            scientific_contract_digest="b" * 64,
            authorization_digest=None,
            parsed_contract_digest=None,
        )


def test_generated_corpus_separates_worker_inputs_keys_and_side(config):
    seed = config["resolved_registries"]["construction_micro"]["corpus"][0]
    corpus = generate_corpus(seed, config, _firewall(config))
    learner_input = corpus["learner_input"]
    protected = corpus["protected_adjudication"]
    assert "evaluation_keys" not in learner_input
    assert "training_oracle" not in learner_input
    assert protected["evaluation_keys"]
    assert protected["training_oracle"]
    assert all("side_stream" not in row for row in learner_input["training_episodes"])
    forbidden = set(config["firewalls"]["forbidden_visible_fields"])
    for prompt in learner_input["evaluation_prompts"]:
        assert not (set(prompt) & forbidden)
        assert prompt["generator_provenance"].endswith("eval-v2")
    assert corpus["audit"]["every_constituent_exposed_in_training"]
    assert corpus["audit"][
        "all_training_candidate_events_exclude_heldout_compositions"
    ]
    assert corpus["audit"]["heldout_candidate_event_count"] == 0
    heldout = {tuple(value) for value in corpus["audit"]["heldout_compositions"]}
    assert all(
        tuple(candidate) not in heldout
        for row in protected["training_oracle"]
        for candidate in row["candidate_compositions"]
    )
    assert corpus["audit"]["train_evaluation_generator_namespaces_disjoint"]
    condition_audit = corpus["audit"]["condition_audit"]
    assert condition_audit["language_visual_records_identical"]
    assert condition_audit["shuffle_exact_bijection"]
    assert condition_audit["shuffle_no_self_donor"]
    assert condition_audit["shuffle_no_same_composition"]
    assert condition_audit[
        "shuffle_preserves_learner_visible_evidence_marginal_by_block"
    ]
    assert condition_audit["corrupted_detector_strict_accuracy_zero"]
    assert condition_audit["shift_non_circular"]
    assert "corpus_seed" not in learner_input
    assert all(
        len(row["episode_id"].split("-", 1)[1]) == 20
        for row in learner_input["training_episodes"]
    )
    assert all(
        len(prompt["prompt_id"].rsplit("-", 1)[1]) == 20
        and all(len(candidate["candidate_id"].split("-", 1)[1]) == 20 for candidate in prompt["candidates"])
        for prompt in learner_input["evaluation_prompts"]
    )
    assert corpus["audit"]["visible_identifiers_are_opaque_and_truth_independent_in_format"]
    assert corpus["audit"]["evaluation_top_level_schema_exact_allowlist"]


def test_action_geometry_varies_beyond_coordinate_permutation_and_is_shared(config):
    seeds = config["resolved_registries"]["construction_micro"]["corpus"][:2]
    corpora = [
        generate_corpus(seed, config, _firewall(config)) for seed in seeds
    ]
    audits = [corpus["audit"] for corpus in corpora]
    assert audits[0]["action_geometry_digest"] != audits[1]["action_geometry_digest"]
    for slot in ("primitive", "manner"):
        left = audits[0]["action_geometry"]["slots"][slot][
            "sorted_pairwise_distances"
        ]
        right = audits[1]["action_geometry"]["slots"][slot][
            "sorted_pairwise_distances"
        ]
        assert left != right
        assert all(
            audit["action_geometry"]["slots"][slot][
                "minimum_self_cross_dot_margin"
            ]
            >= config["qualification_gates"][
                "minimum_action_geometry_self_cross_dot_margin"
            ]
            for audit in audits
        )
    assert all(
        audit["action_geometry"]["all_rows_normalized"]
        and audit["action_geometry"]["all_entries_nonnegative"]
        and audit["action_geometry"]["all_margins_pass"]
        and audit["train_evaluation_use_same_action_prototypes"]
        for audit in audits
    )
    assert all(
        "prototype" not in json.dumps(corpus["learner_input"]).lower()
        for corpus in corpora
    )


def test_train_evaluation_provenance_is_computed_and_detects_namespace_collision(config):
    seed = config["resolved_registries"]["construction_micro"]["corpus"][0]
    corpus = generate_corpus(seed, config, _firewall(config))
    assert corpus["audit"]["provenance"]["train_evaluation_instance_id_overlap"] == 0
    assert corpus["audit"]["train_evaluation_generator_namespaces_disjoint"]
    collided = copy.deepcopy(config)
    collided["design"]["evaluation"]["lexical_eval_rng_namespace"] = collided[
        "design"
    ]["evaluation"]["train_rng_namespace"]
    collided_corpus = generate_corpus(seed, collided, _firewall(collided))
    assert not collided_corpus["audit"][
        "train_evaluation_generator_namespaces_disjoint"
    ]


def test_corrective_disagreement_microcases_and_mutations(config):
    artifact = (
        EVIDENCE_ROOT
        / "output/synthetic_corrective_development_launch_package_v3/mechanism_qualification.json"
    )
    if not artifact.is_file():
        pytest.skip("formal registered mechanism qualification has not been executed yet")
    result = read_json(artifact)
    assert result["status"] == "PASS"
    assert result["case_count"] == 32
    assert result["flip_fractions"]["active"] >= 0.75
    assert all(
        value <= 0.25
        for key, value in result["flip_fractions"].items()
        if key != "active"
    )
    assert result["null_suppression"]["status"] == "PASS"


def test_model_seed_changes_initialization_order_and_fitted_state(config):
    seed = config["resolved_registries"]["construction_micro"]["corpus"][0]
    corpus = generate_corpus(seed, config, _firewall(config))
    episodes = corpus["learner_input"]["training_episodes"]
    evidence = corpus["learner_input"]["condition_evidence"]["synchronized"]
    models = []
    model_objects = []
    traces = []
    for model_seed in config["resolved_registries"]["construction_micro"]["model"]:
        model, trace = fit_corrective_mil(
            episodes,
            evidence,
            config,
            model_seed=model_seed,
            condition="synchronized",
        )
        model_objects.append(model)
        models.append(canonical_digest(model.serializable()))
        traces.append(trace)
    assert len(set(models)) == len(models)
    assert len({row["initial_parameter_digest"] for row in traces}) == len(traces)
    assert len({tuple(row["epoch_order_digests"]) for row in traces}) == len(traces)
    assert all(row["distinct_epoch_parameter_digests"] > 1 for row in traces)
    assert all(row["mutation"] == "active" for row in traces)
    assert all(
        row["epoch_parameter_digests"][-1]
        == canonical_digest(
            {
                "semantics": model.serializable()["semantics"],
                "null_rates": model.serializable()["null_rates"],
            }
        )
        for model, row in zip(model_objects, traces)
    )
    assert all(row["all_candidate_events_enter_posterior"] for row in traces)
    assert all(not row["argmax_prefilter"] for row in traces)
    disconnected, disconnected_trace = fit_corrective_mil(
        episodes,
        evidence,
        config,
        model_seed=config["resolved_registries"]["construction_micro"]["model"][0],
        condition="synchronized",
        mutation="null_update_disconnected",
    )
    assert all(
        value == pytest.approx(float(config["learner"]["null_prior"]))
        for value in disconnected.null_rates.values()
    )
    assert disconnected_trace["mean_null_update_absolute"] == 0.0
    assert disconnected_trace["null_update_frozen_at_initial_prior"] is True
    side_zero, side_zero_trace = fit_corrective_mil(
        episodes,
        evidence,
        config,
        model_seed=config["resolved_registries"]["construction_micro"]["model"][0],
        condition="synchronized",
        mutation="null_side_zero",
    )
    assert side_zero_trace["null_side_weight_effective"] == 0.0
    assert side_zero_trace["mean_null_update_absolute"] > 0.0
    assert side_zero.null_rates != disconnected.null_rates


def test_prediction_is_side_free_and_rejects_sentinel(config):
    seed = config["resolved_registries"]["construction_micro"]["corpus"][0]
    corpus = generate_corpus(seed, config, _firewall(config))
    learner_input = corpus["learner_input"]
    model, _ = fit_corrective_mil(
        learner_input["training_episodes"],
        learner_input["condition_evidence"]["synchronized"],
        config,
        model_seed=config["resolved_registries"]["construction_micro"]["model"][0],
        condition="synchronized",
    )
    first = predict_without_side(model, learner_input["evaluation_prompts"], config)
    second = predict_without_side(model, copy.deepcopy(learner_input["evaluation_prompts"]), config)
    assert first == second
    assert model.serializable()["training_side_state_serialized"] is False
    for alias, value in (
        ("side_stream", [1.0]),
        ("event_logits", [9.0]),
        ("null_logit", 9.0),
        ("detector_input_present", True),
        ("positive_control", True),
        ("exact_window", True),
        ("corruption", "sentinel"),
        ("answer_index", 0),
    ):
        poisoned = copy.deepcopy(learner_input["evaluation_prompts"])
        poisoned[0][alias] = value
        with pytest.raises(ValueError, match="allowlist|forbidden fields"):
            predict_without_side(model, poisoned, config)
    nested = copy.deepcopy(learner_input["evaluation_prompts"])
    nested[0]["candidates"][0]["detector_feature"] = 1.0
    with pytest.raises(ValueError, match="allowlist|forbidden fields"):
        predict_without_side(model, nested, config)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "file_count", "unsorted", "traversal", "extra", "missing"],
)
def test_exact_manifest_rejects_structural_mutations(tmp_path, mutation):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    manifest = manifest_for_paths(tmp_path, ["a.txt", "b.txt"])
    write_json(tmp_path / "manifest.json", manifest)
    if mutation == "duplicate":
        changed = copy.deepcopy(manifest)
        changed["files"].append(copy.deepcopy(changed["files"][0]))
        changed["file_count"] += 1
        changed["digest"] = canonical_digest(changed["files"])
    elif mutation == "file_count":
        changed = {**manifest, "file_count": 999}
    elif mutation == "unsorted":
        changed = copy.deepcopy(manifest)
        changed["files"].reverse()
        changed["digest"] = canonical_digest(changed["files"])
    elif mutation == "traversal":
        changed = copy.deepcopy(manifest)
        changed["files"][0]["path"] = "../a.txt"
        changed["digest"] = canonical_digest(changed["files"])
    elif mutation == "extra":
        (tmp_path / "extra.txt").write_text("x")
        changed = manifest
    else:
        (tmp_path / "a.txt").unlink()
        changed = manifest
    result = verify_exact_manifest(
        tmp_path,
        changed,
        manifest_filename="manifest.json",
    )
    assert result["status"] == "FAIL"


def test_exact_manifest_rejects_symlink_and_broken_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("x")
    link = tmp_path / "link.txt"
    link.symlink_to(target.name)
    with pytest.raises(RuntimeError, match="symlink"):
        manifest_for_paths(tmp_path, ["link.txt"])
    target.unlink()
    with pytest.raises(RuntimeError, match="symlink"):
        manifest_for_tree(tmp_path)


def test_exact_manifest_rejects_empty_hidden_directory_and_root_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("a")
    manifest = manifest_for_paths(root, ["a.txt"])
    write_json(root / "manifest.json", manifest)
    (root / ".partial").mkdir()
    result = verify_exact_manifest(root, manifest, manifest_filename="manifest.json")
    assert result["status"] == "FAIL"
    assert "unexpected_directories" in result["problems"]
    link = tmp_path / "root-link"
    link.symlink_to(root.name)
    with pytest.raises(RuntimeError, match="real directory root"):
        verify_exact_manifest(link, manifest, manifest_filename="manifest.json")


def test_anticipated_launch_manifest_requires_exact_atomic_file_set(tmp_path):
    package = tmp_path / "package"
    staging = tmp_path / "seal-staging"
    package.mkdir()
    staging.mkdir()
    write_json(package / "core.json", {"status": "PASS"})
    write_json(staging / "package_terminal.json", {"status": "PASS"})
    write_json(
        staging / "CORRECTIVE_DEVELOPMENT_LAUNCH_READY.json",
        {"status": "PASS"},
    )
    package_rows = manifest_for_tree(package)["files"]
    seal_rows = [
        {**row, "path": f"launch_seal/{row['path']}"}
        for row in manifest_for_tree(staging)["files"]
    ]
    rows = sorted([*package_rows, *seal_rows], key=lambda row: row["path"])
    manifest = {
        "schema_version": "nursery-exact-file-manifest-v1",
        "file_count": len(rows),
        "files": rows,
        "digest_scheme": "canonical-json-of-sorted-file-rows",
        "digest": canonical_digest(rows),
    }
    assert _verify_anticipated_launch_manifest(package, staging, manifest)[
        "status"
    ] == "PASS"
    hidden = staging / ".unexpected-empty"
    hidden.mkdir()
    assert _verify_anticipated_launch_manifest(package, staging, manifest)[
        "status"
    ] == "FAIL"
    hidden.rmdir()
    for mutation in ("missing", "duplicate", "substitution"):
        changed = copy.deepcopy(manifest)
        if mutation == "missing":
            changed["files"].pop()
        elif mutation == "duplicate":
            changed["files"].append(copy.deepcopy(changed["files"][0]))
        else:
            changed["files"][0]["path"] = "launch_seal/substituted.json"
        changed["file_count"] = len(changed["files"])
        changed["digest"] = canonical_digest(changed["files"])
        assert _verify_anticipated_launch_manifest(package, staging, changed)[
            "status"
        ] == "FAIL"


@pytest.mark.parametrize(
    "mutation", ["change", "delete", "extra", "empty_directory", "symlink"]
)
def test_prefreeze_evidence_rejects_every_bound_tree_mutation(tmp_path, mutation):
    package = tmp_path / "package"
    evidence = package / "evidence"
    evidence.mkdir(parents=True)
    original = evidence / "a.json"
    write_json(original, {"value": 1})
    freeze_prefreeze_evidence(package)
    assert verify_prefreeze_evidence(package)["status"] == "PASS"
    if mutation == "change":
        original.write_text('{"value":2}\n', encoding="utf-8")
    elif mutation == "delete":
        original.unlink()
    elif mutation == "extra":
        write_json(evidence / "b.json", {"value": 2})
    elif mutation == "empty_directory":
        (evidence / "empty").mkdir()
    else:
        original.unlink()
        original.symlink_to("missing.json")
    assert verify_prefreeze_evidence(package)["status"] == "FAIL"
    with pytest.raises(FileExistsError, match="pristine"):
        freeze_prefreeze_evidence(package)


def test_snapshot_receipt_rejects_schema_protocol_and_tracked_file_mutations(
    tmp_path,
):
    package, snapshot, receipt = _make_snapshot_receipt_fixture(tmp_path / "base")
    assert integrity_module.verify_snapshot(snapshot)["status"] == "PASS"
    receipt_path = package / "freeze_receipt.json"
    mutations = []
    changed = copy.deepcopy(receipt)
    changed["unexpected"] = True
    mutations.append(changed)
    changed = copy.deepcopy(receipt)
    changed["protocol_id"] = "forged-protocol"
    mutations.append(changed)
    changed = copy.deepcopy(receipt)
    changed["tracked_files"] = changed["tracked_files"][:-1]
    mutations.append(changed)
    changed = copy.deepcopy(receipt)
    changed["development_outcome_count"] = 1
    mutations.append(changed)
    for changed in mutations:
        write_json(receipt_path, changed, overwrite=True)
        assert integrity_module.verify_snapshot(snapshot)["status"] == "FAIL"
    write_json(receipt_path, receipt, overwrite=True)

    package2, snapshot2, receipt2 = _make_snapshot_receipt_fixture(
        tmp_path / "source-reseal"
    )
    runner_path = snapshot2 / "scripts/run_synthetic_corrective_alignment_v3.py"
    runner_path.write_bytes(runner_path.read_bytes() + b"# forged reseal\n")
    manifest = manifest_for_tree(snapshot2, exclude=["snapshot_manifest.json"])
    write_json(snapshot2 / "snapshot_manifest.json", manifest, overwrite=True)
    receipt2["snapshot_manifest_sha256"] = sha256_file(
        snapshot2 / "snapshot_manifest.json"
    )
    receipt2["snapshot_file_count"] = manifest["file_count"]
    receipt2["snapshot_digest"] = manifest["digest"]
    receipt2["frozen_runner_sha256"] = sha256_file(runner_path)
    receipt2["tracked_files"] = sorted(row["path"] for row in manifest["files"])
    write_json(package2 / "freeze_receipt.json", receipt2, overwrite=True)
    assert integrity_module.verify_snapshot(snapshot2)["status"] == "FAIL"

    package3, snapshot3, receipt3 = _make_snapshot_receipt_fixture(
        tmp_path / "environment-reseal"
    )
    write_json(
        package3 / "frozen_environment.json",
        {"schema_version": "test-frozen-environment-v1", "forged": True},
        overwrite=True,
    )
    receipt3["frozen_environment_sha256"] = sha256_file(
        package3 / "frozen_environment.json"
    )
    write_json(package3 / "freeze_receipt.json", receipt3, overwrite=True)
    assert integrity_module.verify_snapshot(snapshot3)["status"] == "FAIL"


def test_parallel_micro_completion_rejects_mutated_or_partial_state(
    tmp_path, monkeypatch, config
):
    package = tmp_path / "package"
    inputs = package / "parallel_micro_attempt_1_persisted_inputs"
    jobs_one = package / "parallel_micro_attempt_1_jobs_1"
    jobs_n = package / "parallel_micro_attempt_1_jobs_n"
    for path in (inputs, jobs_one, jobs_n):
        path.mkdir(parents=True)
    write_json(inputs / "input_manifest.json", {"status": "PASS"})
    write_json(jobs_one / "complete_manifest.json", {"status": "PASS"})
    write_json(jobs_n / "complete_manifest.json", {"status": "PASS"})
    write_json(package / "prequalification_design_lock.json", {"status": "PASS"})
    equivalence = {
        "schema_version": "test-equivalence-v1",
        "status": "PASS",
        "byte_identical": True,
    }
    write_json(package / "parallel_equivalence_attempt_1.json", equivalence)
    write_json(package / "parallel_benchmark_attempt_1.json", {"status": "PASS"})
    write_json(package / "micro_attempt_consumed.json", {"status": "CONSUMED"})
    write_json(package / "worker_launcher_preflight.json", {"status": "PASS"})
    micro_attempt_validation = {
        "status": "PASS",
        "receipt_sha256": sha256_file(package / "micro_attempt_consumed.json"),
        "worker_launcher_preflight_sha256": sha256_file(
            package / "worker_launcher_preflight.json"
        ),
    }
    monkeypatch.setattr(runner, "compare_adjudication_trees", lambda *_: equivalence)
    monkeypatch.setattr(
        runner, "validate_benchmark_report", lambda *args, **kwargs: {"status": "PASS"}
    )
    monkeypatch.setattr(
        runner, "verify_persisted_inputs", lambda *_: {"status": "PASS"}
    )
    monkeypatch.setattr(
        runner,
        "_validate_micro_attempt_receipt",
        lambda *args, **kwargs: copy.deepcopy(micro_attempt_validation),
    )
    completion = {
        "schema_version": "nursery-corrective-parallel-micro-completion-v3",
        "status": "PASS",
        "prequalification_design_lock_sha256": sha256_file(
            package / "prequalification_design_lock.json"
        ),
        "micro_attempt_receipt_sha256": sha256_file(
            package / "micro_attempt_consumed.json"
        ),
        "worker_launcher_preflight_sha256": sha256_file(
            package / "worker_launcher_preflight.json"
        ),
        "micro_attempt_validation_digest": canonical_digest(
            micro_attempt_validation
        ),
        "input_manifest_sha256": sha256_file(inputs / "input_manifest.json"),
        "jobs_1_complete_manifest_sha256": sha256_file(
            jobs_one / "complete_manifest.json"
        ),
        "jobs_n_complete_manifest_sha256": sha256_file(
            jobs_n / "complete_manifest.json"
        ),
        "parallel_equivalence_sha256": sha256_file(
            package / "parallel_equivalence_attempt_1.json"
        ),
        "parallel_benchmark_sha256": sha256_file(
            package / "parallel_benchmark_attempt_1.json"
        ),
        "frozen_jobs": int(config["parallel"]["frozen_jobs"]),
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "non_replayable": True,
    }
    completion_path = package / "parallel_micro_attempt_1_completion.json"
    write_json(completion_path, completion)
    assert runner._validate_parallel_micro_attempt_1(
        tmp_path, package, config
    )["status"] == "PASS"
    for field, value in (
        ("status", "FAIL"),
        ("jobs_n_complete_manifest_sha256", "0" * 64),
        ("frozen_jobs", int(config["parallel"]["frozen_jobs"]) + 1),
        ("development_outcome_count", 1),
        ("non_replayable", False),
    ):
        changed = copy.deepcopy(completion)
        changed[field] = value
        write_json(completion_path, changed, overwrite=True)
        assert runner._validate_parallel_micro_attempt_1(
            tmp_path, package, config
        )["status"] == "FAIL"
    write_json(completion_path, completion, overwrite=True)
    (jobs_n / "complete_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        runner._validate_parallel_micro_attempt_1(tmp_path, package, config)


def test_benchmark_validator_recomputes_gates_and_telemetry(
    tmp_path, config, monkeypatch
):
    monkeypatch.setattr(
        integrity_module,
        "_available_memory_bytes",
        lambda: 1 << 50,
    )
    monkeypatch.setattr(
        integrity_module.resource,
        "getrusage",
        lambda _scope: type(
            "DeterministicUsage", (), {"ru_maxrss": 1_000_000}
        )(),
    )
    plan = build_work_plan(config, purpose="construction_micro")
    expected_environment = dict(config["parallel"]["thread_environment"])

    def make_runtime(root: Path, jobs: int):
        (root / "runtime").mkdir(parents=True)
        (root / "adjudication").mkdir()
        write_json(root / "adjudication/work_plan.json", plan)
        rows = []
        for index, unit in enumerate(plan["units"]):
            unit_id = str(unit["unit_id"])
            shard = root / "shards" / unit_id
            shard.mkdir(parents=True)
            write_json(shard / "manifest.json", {"unit_id": unit_id})
            rows.append(
                {
                    "unit_id": unit_id,
                    "shard_manifest_sha256": sha256_file(
                        shard / "manifest.json"
                    ),
                    "wall_seconds": 0.05 + index * 0.001,
                    "worker_peak_rss_native_units": 100 + index,
                    "bootstrap_environment_digest": "b" * 64,
                    "worker_boundary_digest": "a" * 64,
                    "worker_boundary_operational_file_count": 11,
                    "worker_input_regular_files_verified": 1,
                    "worker_full_tree_verifications": 0,
                    "thread_environment_observed": expected_environment,
                    "native_threadpools": [
                        {
                            "internal_api": "openblas",
                            "prefix": "libopenblas",
                            "num_threads": 1,
                        }
                    ],
                    "python_startup_flags_observed": dict(
                        config["environment"]["worker_python_startup_flags"]
                    ),
                }
            )
        rows.sort(key=lambda row: row["unit_id"])
        telemetry = {
            "schema_version": "nursery-corrective-runtime-v3",
            "jobs": jobs,
            "units": rows,
            "wall_seconds_sum": float(sum(row["wall_seconds"] for row in rows)),
            "maximum_worker_peak_rss_native_units": max(
                row["worker_peak_rss_native_units"] for row in rows
            ),
            "excluded_from_adjudication": True,
        }
        write_json(root / "runtime/runtime_telemetry.json", telemetry)
        write_json(
            root / "runtime/scheduler_validation.json",
            {
                "status": "PASS",
                "jobs": int(jobs),
                "maximum_observed_concurrency": int(jobs),
            },
        )
        complete = manifest_for_tree(root)
        write_json(root / "complete_manifest.json", complete)
        return telemetry

    jobs_one = tmp_path / "jobs-one"
    jobs_n = tmp_path / "jobs-n"
    inputs = tmp_path / "inputs"
    telemetry_one = make_runtime(jobs_one, 1)
    make_runtime(jobs_n, int(config["parallel"]["frozen_jobs"]))
    inputs.mkdir()
    write_json(inputs / "input_manifest.json", {"status": "PASS"})
    micro_units = len(plan["units"])
    development_units = (
        len(config["resolved_registries"]["development"]["corpus"])
        * len(config["resolved_registries"]["development"]["model"])
    )
    report = integrity_module.benchmark_report(
        jobs_one_root=jobs_one,
        jobs_n_root=jobs_n,
        input_root=inputs,
        wall_jobs_one=2.0,
        wall_jobs_n=1.0,
        input_generation_wall=0.1,
        final_assembly_probe_wall=0.1,
        jobs_n=int(config["parallel"]["frozen_jobs"]),
        development_corpus_count=len(
            config["resolved_registries"]["development"]["corpus"]
        ),
        micro_corpus_count=len(
            config["resolved_registries"]["construction_micro"]["corpus"]
        ),
        development_unit_count=development_units,
        micro_unit_count=micro_units,
        minimum_speedup=float(config["parallel"]["minimum_benchmark_speedup"]),
        wall_time_contingency_multiplier=float(
            config["parallel"]["wall_time_contingency_multiplier"]
        ),
        resource_contingency_multiplier=float(
            config["parallel"]["resource_contingency_multiplier"]
        ),
        parent_memory_projection_method=str(
            config["parallel"]["parent_memory_projection_method"]
        ),
        maximum_host_ram_fraction=float(
            config["parallel"]["maximum_host_ram_fraction"]
        ),
        maximum_current_available_ram_fraction=float(
            config["parallel"]["maximum_current_available_ram_fraction"]
        ),
        maximum_current_free_disk_fraction=float(
            config["parallel"]["maximum_current_free_disk_fraction"]
        ),
        minimum_logical_cpu_count=int(
            config["parallel"]["minimum_logical_cpu_count"]
        ),
    )
    assert report["status"] == "PASS"
    assert report[
        "projected_development_full_cohort_by_unit_scaling_seconds"
    ] > 0.0
    assert report[
        "full_cohort_projection_includes_parent_merge_scoring_adjudication"
    ] is True
    assert report["parent_memory_unit_scaling_factor"] == (
        development_units / micro_units
    )
    assert report["projected_development_parent_peak_rss_bytes"] == math.ceil(
        report["parent_peak_rss_bytes"] * development_units / micro_units
    )
    assert report["conservative_concurrent_peak_rss_bytes"] == math.ceil(
        (
            report["maximum_worker_peak_rss_bytes"]
            * int(config["parallel"]["frozen_jobs"])
            + report["projected_development_parent_peak_rss_bytes"]
        )
        * float(config["parallel"]["resource_contingency_multiplier"])
    )
    kwargs = {
        "jobs_one_root": jobs_one,
        "jobs_n_root": jobs_n,
        "input_root": inputs,
        "config": config,
    }
    assert integrity_module.validate_benchmark_report(report, **kwargs)[
        "status"
    ] == "PASS"
    for field, replacement in (
        ("parent_memory_projection_method", "unscaled_parent_peak"),
        ("parent_memory_unit_scaling_factor", 1.0),
        (
            "projected_development_parent_peak_rss_bytes",
            report["projected_development_parent_peak_rss_bytes"] - 1,
        ),
        (
            "conservative_concurrent_peak_rss_bytes",
            report["conservative_concurrent_peak_rss_bytes"] - 1,
        ),
    ):
        changed_memory = copy.deepcopy(report)
        changed_memory[field] = replacement
        assert integrity_module.validate_benchmark_report(
            changed_memory, **kwargs
        )["status"] == "FAIL"
    physical_ceiling_failure = copy.deepcopy(report)
    physical_ceiling_failure["host_physical_memory_bytes"] = max(
        1,
        math.floor(
            report["conservative_concurrent_peak_rss_bytes"]
            / float(config["parallel"]["maximum_host_ram_fraction"])
        )
        - 1,
    )
    physical_ceiling_failure["host_available_memory_bytes_at_benchmark"] = 1 << 50
    assert integrity_module.validate_benchmark_report(
        physical_ceiling_failure, **kwargs
    )["status"] == "FAIL"
    available_ceiling_failure = copy.deepcopy(report)
    available_ceiling_failure["host_physical_memory_bytes"] = 1 << 50
    available_ceiling_failure["host_available_memory_bytes_at_benchmark"] = max(
        1,
        math.floor(
            report["conservative_concurrent_peak_rss_bytes"]
            / float(
                config["parallel"]["maximum_current_available_ram_fraction"]
            )
        )
        - 1,
    )
    assert integrity_module.validate_benchmark_report(
        available_ceiling_failure, **kwargs
    )["status"] == "FAIL"
    package = tmp_path / "walltime-package"
    (package / "preservation").mkdir(parents=True)
    write_json(package / "parallel_benchmark_attempt_1.json", report)
    write_json(package / "preservation/preservation_proof.json", {"status": "PASS"})
    write_json(
        package / "preservation_attempt_consumed.json",
        {"status": "ONE_PRESERVATION_ATTEMPT_CONSUMED"},
    )
    write_json(
        package / "preservation_timing.json",
        {
            "schema_version": "nursery-corrective-preservation-timing-v3",
            "status": "PASS",
            "jobs": int(config["parallel"]["frozen_jobs"]),
            "wall_seconds": 3.0,
            "preservation_proof_sha256": sha256_file(
                package / "preservation/preservation_proof.json"
            ),
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
        },
    )
    write_json(
        package / "preservation_completion.json",
        {
            "schema_version": "nursery-corrective-preservation-completion-v3",
            "status": "PASS",
            "jobs": int(config["parallel"]["frozen_jobs"]),
            "preservation_attempt_receipt_sha256": sha256_file(
                package / "preservation_attempt_consumed.json"
            ),
            "preservation_proof_sha256": sha256_file(
                package / "preservation/preservation_proof.json"
            ),
            "preservation_timing_sha256": sha256_file(
                package / "preservation_timing.json"
            ),
            "non_replayable": True,
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
        },
    )
    assert runner._validate_preservation_completion(package, config)[
        "status"
    ] == "PASS"
    walltime = runner._development_command_walltime_estimate(
        package,
        config,
        measured_fixed_validation_probe_seconds=2.0,
    )
    write_json(package / "development_command_walltime_estimate.json", walltime)
    assert runner._validate_development_command_walltime_estimate(
        package,
        config,
    )["status"] == "PASS"
    changed_walltime = copy.deepcopy(walltime)
    changed_walltime["estimated_full_development_command_wall_seconds"] += 1.0
    write_json(
        package / "development_command_walltime_estimate.json",
        changed_walltime,
        overwrite=True,
    )
    assert runner._validate_development_command_walltime_estimate(
        package,
        config,
    )["status"] == "FAIL"
    write_json(
        package / "development_command_walltime_estimate.json",
        walltime,
        overwrite=True,
    )
    timing = read_json(package / "preservation_timing.json")
    timing["wall_seconds"] = 4.0
    write_json(package / "preservation_timing.json", timing, overwrite=True)
    assert runner._validate_preservation_completion(package, config)[
        "status"
    ] == "FAIL"
    assert runner._validate_development_command_walltime_estimate(
        package,
        config,
    )["status"] == "FAIL"
    changed = copy.deepcopy(report)
    changed["gates"]["speedup"] = False
    assert integrity_module.validate_benchmark_report(changed, **kwargs)[
        "status"
    ] == "FAIL"
    telemetry_one["maximum_worker_peak_rss_native_units"] = 1
    write_json(
        jobs_one / "runtime/runtime_telemetry.json", telemetry_one, overwrite=True
    )
    complete = manifest_for_tree(jobs_one, exclude=["complete_manifest.json"])
    write_json(jobs_one / "complete_manifest.json", complete, overwrite=True)
    assert integrity_module.validate_benchmark_report(report, **kwargs)[
        "status"
    ] == "FAIL"


def test_construction_attempt_receipt_rejects_mutated_hash_chain(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    for name in (
        "prequalification_design_lock.json",
        "micro_attempt_consumed.json",
        "worker_launcher_preflight.json",
        "parallel_equivalence_attempt_1.json",
        "parallel_benchmark_attempt_1.json",
        "parallel_micro_attempt_1_completion.json",
    ):
        write_json(package / name, {"name": name})
    lock_verification = {"status": "PASS", "checks": {"locked": True}}
    micro_validation = {"status": "PASS", "checks": {"complete": True}}
    receipt = {
        "schema_version": "nursery-corrective-construction-attempt-v3",
        "status": "ONE_CONSTRUCTION_ATTEMPT_CONSUMED",
        "prequalification_design_lock_sha256": sha256_file(
            package / "prequalification_design_lock.json"
        ),
        "micro_attempt_receipt_sha256": sha256_file(
            package / "micro_attempt_consumed.json"
        ),
        "worker_launcher_preflight_sha256": sha256_file(
            package / "worker_launcher_preflight.json"
        ),
        "micro_lock_verification_digest": canonical_digest(lock_verification),
        "parallel_equivalence_sha256": sha256_file(
            package / "parallel_equivalence_attempt_1.json"
        ),
        "parallel_benchmark_sha256": sha256_file(
            package / "parallel_benchmark_attempt_1.json"
        ),
        "parallel_micro_completion_sha256": sha256_file(
            package / "parallel_micro_attempt_1_completion.json"
        ),
        "parallel_micro_validation_digest": canonical_digest(micro_validation),
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "non_replayable": True,
    }
    receipt_path = package / "construction_qualification_attempt_consumed.json"
    write_json(receipt_path, receipt)
    assert runner._validate_construction_attempt_receipt(
        package,
        micro_lock_verification=lock_verification,
        micro_validation=micro_validation,
    )["status"] == "PASS"
    for field, value in (
        ("schema_version", "forged"),
        ("micro_lock_verification_digest", "0" * 64),
        ("parallel_micro_completion_sha256", "0" * 64),
        ("development_outcome_count", 1),
        ("non_replayable", False),
    ):
        changed = copy.deepcopy(receipt)
        changed[field] = value
        write_json(receipt_path, changed, overwrite=True)
        assert runner._validate_construction_attempt_receipt(
            package,
            micro_lock_verification=lock_verification,
            micro_validation=micro_validation,
        )["status"] == "FAIL"


def test_recompute_comparison_rejects_changed_adjudication_bytes(tmp_path):
    rehearsal = tmp_path / "rehearsal/adjudication"
    recompute = tmp_path / "recompute/adjudication"
    rehearsal.mkdir(parents=True)
    recompute.mkdir(parents=True)
    expected_files = {
        "work_plan.json",
        "execution_contract.json",
        "merged_worker_results.json",
        "merged_operation_ledgers.json",
        "parent_operation_ledger.json",
        "parent_ledger_verification.json",
        "scored_condition_units.json",
        "scored_mutation_units.json",
        "model_averages.json",
        "primary_inference.json",
        "present_null_dependence.json",
        "mechanism_mutations.json",
        "informativeness.json",
        "development_validity.json",
        "scientific_summary.json",
        "completeness.json",
        "manifest.json",
    }
    for name in expected_files:
        write_json(rehearsal / name, {"name": name, "value": 1})
        write_json(recompute / name, {"name": name, "value": 1})
    assert integrity_module.recompute_comparison_value(
        rehearsal.parent, recompute.parent
    )["status"] == "PASS"
    write_json(recompute / "model_averages.json", {"value": 2}, overwrite=True)
    assert integrity_module.recompute_comparison_value(
        rehearsal.parent, recompute.parent
    )["status"] == "FAIL"
    jobs_one = tmp_path / "jobs-one/adjudication"
    jobs_n = tmp_path / "jobs-n/adjudication"
    jobs_one.mkdir(parents=True)
    jobs_n.mkdir(parents=True)
    for name in expected_files:
        write_json(jobs_one / name, {"name": name, "value": 1})
        write_json(jobs_n / name, {"name": name, "value": 1})
    assert integrity_module.compare_adjudication_trees(
        jobs_one.parent, jobs_n.parent
    )["status"] == "PASS"
    incomplete_one = tmp_path / "incomplete-one/adjudication"
    incomplete_n = tmp_path / "incomplete-n/adjudication"
    incomplete_one.mkdir(parents=True)
    incomplete_n.mkdir(parents=True)
    for name in expected_files - {"model_averages.json"}:
        write_json(incomplete_one / name, {"name": name, "value": 1})
        write_json(incomplete_n / name, {"name": name, "value": 1})
    incomplete = integrity_module.compare_adjudication_trees(
        incomplete_one.parent, incomplete_n.parent
    )
    assert incomplete["status"] == "FAIL"
    assert not incomplete["expected_file_set_complete"]
    extra_one = tmp_path / "extra-one/adjudication"
    extra_n = tmp_path / "extra-n/adjudication"
    extra_one.mkdir(parents=True)
    extra_n.mkdir(parents=True)
    for name in expected_files | {"extra.json"}:
        write_json(extra_one / name, {"name": name, "value": 1})
        write_json(extra_n / name, {"name": name, "value": 1})
    extra = integrity_module.compare_adjudication_trees(
        extra_one.parent, extra_n.parent
    )
    assert extra["status"] == "FAIL"
    assert not extra["expected_file_set_complete"]
    write_json(
        recompute / "model_averages.json",
        {"name": "model_averages.json", "value": 1},
        overwrite=True,
    )
    write_json(recompute / "extra.json", {"value": 1})
    assert integrity_module.recompute_comparison_value(
        rehearsal.parent, recompute.parent
    )["status"] == "FAIL"


def test_ledger_requires_exact_count_chain_unit_and_references(config):
    purpose = "construction_micro"
    corpus_seed = config["resolved_registries"][purpose]["corpus"][0]
    model_seed = config["resolved_registries"][purpose]["model"][0]
    firewall = _firewall(config)
    expected = ["fit", "predict", "score"]
    for operation in expected:
        firewall.authorize(
            operation,
            [
                IdentifierReference("corpus", corpus_seed),
                IdentifierReference("model", model_seed),
            ],
            unit_id="unit",
            local_sequence=len(firewall.operations),
        )
    ledger = firewall.ledger()
    assert verify_ledger(
        ledger,
        purpose=purpose,
        unit_id="unit",
        expected_operations=expected,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
    )["status"] == "PASS"
    for mutate in ("missing", "duplicate", "wrong_unit", "wrong_ref", "broken_chain"):
        changed = copy.deepcopy(ledger)
        if mutate == "missing":
            changed["entries"].pop()
            changed["entry_count"] -= 1
        elif mutate == "duplicate":
            changed["entries"].append(copy.deepcopy(changed["entries"][-1]))
            changed["entry_count"] += 1
        elif mutate == "wrong_unit":
            changed["entries"][0]["unit_id"] = "other"
        elif mutate == "wrong_ref":
            changed["entries"][0]["references"][0]["value"] += 2
        else:
            changed["entries"][1]["previous_digest"] = "0" * 64
        assert verify_ledger(
            changed,
            purpose=purpose,
            unit_id="unit",
            expected_operations=expected,
            corpus_seed=corpus_seed,
            model_seed=model_seed,
        )["status"] == "FAIL"


def test_firewall_and_operation_audit_reject_old_reserve_and_wrong_purpose_ids(
    tmp_path, config
):
    values = {
        "old_development": 281001,
        "old_confirmation": 289001,
        "failed_corrective_v1": 320001,
        "current_confirmation": config["resolved_registries"][
            "confirmation_reserve"
        ]["corpus"][0],
        "wrong_purpose": config["resolved_registries"]["development"]["corpus"][0],
    }
    corpus = config["resolved_registries"]["construction_micro"]["corpus"][0]
    for label, value in values.items():
        firewall = _firewall(config)
        with pytest.raises(PermissionError):
            firewall.authorize(
                "generate",
                [IdentifierReference("corpus", value)],
                unit_id=f"forbidden-{label}",
                local_sequence=0,
            )
        package = tmp_path / label
        package.mkdir()
        write_json(
            package / "operation_ledger.json",
            {
                "purpose": "construction_micro",
                "entries": [{"references": [{"role": "corpus", "value": value}]}],
            },
        )
        assert operation_identifier_audit(package, config)["status"] == "FAIL"
    valid = _firewall(config)
    valid.authorize(
        "generate",
        [IdentifierReference("corpus", corpus)],
        unit_id="valid",
        local_sequence=0,
    )


def test_authorization_exact_schema_and_digest_mutations():
    base = {
        key: (
            False
            if key in {"confirmation_authorized"}
            else True
            if key in {"development_authorized", "authorization_non_replayable"}
            else 0
            if key in {"development_outcome_count", "confirmation_outcome_count"}
            else "x"
        )
        for key in AUTHORIZATION_FIELDS
        if key
        not in {
            "schema_version",
            "protocol_id",
            "status",
            "authorization_scope",
            "authorization_digest",
        }
    }
    base.update(
        {
            "authorization_scope": "exactly_one_later_development_attempt",
            "required_jobs": 4,
            "required_thread_environment": {},
            "development_registry": {"corpus": [], "model": [], "inference": []},
            "exact_argv": ["run-development"],
        }
    )
    for field in (
        "resolved_repository_root",
        "resolved_snapshot_root",
        "resolved_authorization_path",
        "resolved_output_root",
        "resolved_staging_root",
        "resolved_inner_staging_root",
        "resolved_publication_seal_path",
        "resolved_claim_path",
        "resolved_capability_receipt_path",
        "required_cwd",
        "required_python_executable",
        "required_runner_path",
    ):
        base[field] = f"/fixture/{field}"
    value = authorization_payload(**base)
    assert verify_authorization(value)["status"] == "PASS"
    changed = copy.deepcopy(value)
    changed["required_jobs"] = 3
    assert verify_authorization(changed)["status"] == "FAIL"
    changed = copy.deepcopy(value)
    changed["unexpected"] = True
    changed["authorization_digest"] = canonical_digest(
        {key: child for key, child in changed.items() if key != "authorization_digest"}
    )
    assert verify_authorization(changed)["status"] == "FAIL"


def _make_preflight_package(tmp_path: Path, monkeypatch, config) -> dict:
    repository = tmp_path / "repository"
    package = repository / "output/synthetic_corrective_development_launch_package_v3"
    snapshot = package / "frozen_source_snapshot"
    snapshot.mkdir(parents=True)
    snapshot_config = snapshot / "configs/synthetic_corrective_alignment_v3.yaml"
    snapshot_config.parent.mkdir(parents=True)
    value = yaml.safe_load(CONFIG.read_text())
    value["protocol"]["status"] = "frozen"
    snapshot_config.write_text(yaml.safe_dump(value, sort_keys=False))
    prior_source = ROOT / config["registries"]["canonical_prior_registry"]
    prior = snapshot / config["registries"]["canonical_prior_registry"]
    prior.parent.mkdir(parents=True)
    shutil.copyfile(prior_source, prior)
    live_runner = Path(runner.__file__).resolve()
    snapshot_runner = snapshot / "scripts/run_synthetic_corrective_alignment_v3.py"
    snapshot_runner.parent.mkdir(parents=True)
    shutil.copyfile(live_runner, snapshot_runner)
    _copy_scientific_core(snapshot)
    snapshot_manifest = manifest_for_tree(snapshot, exclude=["snapshot_manifest.json"])
    write_json(snapshot / "snapshot_manifest.json", snapshot_manifest)
    frozen_config = load_config(snapshot_config, repository_root=snapshot)
    environment_source = (
        ROOT.parents[2] if ROOT.name == "frozen_source_snapshot" else ROOT
    )
    frozen_environment = runner.environment_record(environment_source, config)
    write_json(
        package / "prequalification_design_lock.json",
        {
            "schema_version": "nursery-corrective-prequalification-design-lock-v3",
            "status": "LOCKED_BEFORE_OFFICIAL_FIXTURES",
            "tracked_nonconfig_manifest": manifest_for_paths(
                snapshot,
                [
                    "scripts/run_synthetic_corrective_alignment_v3.py",
                    str(config["registries"]["canonical_prior_registry"]),
                    *SCIENTIFIC_CORE_RELATIVES,
                ],
            ),
            "anticipated_frozen_config_sha256": sha256_file(snapshot_config),
            "qualification_environment": frozen_environment,
        },
    )
    write_json(
        package / "micro_attempt_consumed.json",
        {"status": "ONE_MICRO_ATTEMPT_CONSUMED_BEFORE_LAUNCHER_PREFLIGHT"},
    )
    write_json(package / "worker_launcher_preflight.json", {"status": "PASS"})
    prefreeze_contract = freeze_prefreeze_evidence(package)
    write_json(package / "frozen_environment.json", frozen_environment)
    write_json(package / "frozen_seed_registries.json", registry_snapshot(frozen_config))
    write_json(
        package / "freeze_receipt.json",
        {
            "schema_version": "nursery-corrective-freeze-receipt-v3",
            "status": "FROZEN",
            "protocol_id": frozen_config["protocol"]["id"],
            "config_status": "frozen",
            "snapshot_manifest_sha256": sha256_file(
                snapshot / "snapshot_manifest.json"
            ),
            "snapshot_file_count": snapshot_manifest["file_count"],
            "snapshot_digest": snapshot_manifest["digest"],
            "frozen_config_sha256": sha256_file(snapshot_config),
            "frozen_runner_sha256": sha256_file(snapshot_runner),
            "frozen_environment_sha256": sha256_file(
                package / "frozen_environment.json"
            ),
            "frozen_registries_sha256": sha256_file(
                package / "frozen_seed_registries.json"
            ),
            "prequalification_design_lock_sha256": sha256_file(
                package / "prequalification_design_lock.json"
            ),
            "micro_attempt_receipt_sha256": sha256_file(
                package / "micro_attempt_consumed.json"
            ),
            "worker_launcher_preflight_sha256": sha256_file(
                package / "worker_launcher_preflight.json"
            ),
            "prefreeze_evidence_manifest_sha256": sha256_file(
                package / "prefreeze_evidence_manifest.json"
            ),
            "prefreeze_evidence_contract_sha256": sha256_file(
                package / "prefreeze_evidence_contract.json"
            ),
            "prefreeze_evidence_digest": prefreeze_contract["manifest_digest"],
            "prefreeze_evidence_file_count": prefreeze_contract["file_count"],
            "tracked_files": sorted(
                [
                    "configs/synthetic_corrective_alignment_v3.yaml",
                    "scripts/run_synthetic_corrective_alignment_v3.py",
                    str(config["registries"]["canonical_prior_registry"]),
                    *SCIENTIFIC_CORE_RELATIVES,
                ]
            ),
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
            "confirmation_authorized": False,
        },
    )
    artifacts = [
        package / "excluded_rehearsal/cohort_summary.json",
        package / "independent_recompute_comparison.json",
        package / "official_test_execution_report.json",
        package / "parallel_benchmark_attempt_1.json",
        package / "preservation/preservation_proof.json",
    ]
    for path in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"status": "PASS"})
    write_json(
        package / "outcome_registry.json",
        {
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
            "authorization_consumed": False,
        },
    )
    core = manifest_for_tree(
        package,
        exclude=[
            "package_core_manifest.json",
        ],
    )
    write_json(package / "package_core_manifest.json", core)
    seal = package / "launch_seal"
    seal.mkdir()
    write_json(
        seal / "package_terminal.json",
        {"terminal_state": "CORRECTIVE_DEVELOPMENT_LAUNCH_READY"},
    )
    output = repository / "output/synthetic_corrective_development_v3_one_shot"
    staging = repository / "output/.synthetic_corrective_development_v3.staging"
    inner_staging = (
        repository / "output/..synthetic_corrective_development_v3.staging.cohort-staging"
    )
    publication_seal = output / "SCIENTIFIC_OUTCOME.json"
    claim = package / "development_authorization_consumed.json"
    capability_receipt = package / "development_capability_issued.json"
    authorization_path = seal / "CORRECTIVE_DEVELOPMENT_LAUNCH_READY.json"
    argv = list(frozen_config["commands"]["development_argv"])
    thread_environment = dict(frozen_config["parallel"]["thread_environment"])
    for name, child in thread_environment.items():
        monkeypatch.setenv(name, child)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    authorization = authorization_payload(
        resolved_repository_root=str(repository.resolve()),
        resolved_snapshot_root=str(snapshot.resolve()),
        resolved_authorization_path=str(authorization_path.resolve()),
        resolved_output_root=str(output.resolve()),
        resolved_staging_root=str(staging.resolve()),
        resolved_inner_staging_root=str(inner_staging.resolve()),
        resolved_publication_seal_path=str(publication_seal.resolve()),
        resolved_claim_path=str(claim.resolve()),
        resolved_capability_receipt_path=str(capability_receipt.resolve()),
        required_cwd=str(repository.resolve()),
        required_jobs=4,
        required_thread_environment=thread_environment,
        required_python_executable=frozen_environment["python_executable"],
        required_python_sha256=frozen_environment["python_sha256"],
        required_runner_path=str(snapshot_runner.resolve()),
        required_runner_sha256=sha256_file(snapshot_runner),
        required_config_sha256=sha256_file(snapshot_config),
        required_snapshot_manifest_sha256=sha256_file(snapshot / "snapshot_manifest.json"),
        required_freeze_receipt_sha256=sha256_file(package / "freeze_receipt.json"),
        required_prequalification_design_lock_sha256=sha256_file(
            package / "prequalification_design_lock.json"
        ),
        required_micro_attempt_sha256=sha256_file(
            package / "micro_attempt_consumed.json"
        ),
        required_worker_launcher_preflight_sha256=sha256_file(
            package / "worker_launcher_preflight.json"
        ),
        required_environment_sha256=sha256_file(package / "frozen_environment.json"),
        required_registries_sha256=sha256_file(
            package / "frozen_seed_registries.json"
        ),
        required_package_core_manifest_sha256=sha256_file(
            package / "package_core_manifest.json"
        ),
        required_package_terminal_sha256=sha256_file(seal / "package_terminal.json"),
        required_excluded_rehearsal_sha256=sha256_file(
            package / "excluded_rehearsal/cohort_summary.json"
        ),
        required_recompute_comparison_sha256=sha256_file(
            package / "independent_recompute_comparison.json"
        ),
        required_official_tests_sha256=sha256_file(
            package / "official_test_execution_report.json"
        ),
        required_benchmark_sha256=sha256_file(
            package / "parallel_benchmark_attempt_1.json"
        ),
        required_preservation_proof_sha256=sha256_file(
            package / "preservation/preservation_proof.json"
        ),
        development_registry=frozen_config["resolved_registries"]["development"],
        confirmation_reserve_digest=canonical_digest(
            frozen_config["resolved_registries"]["confirmation_reserve"]
        ),
        exact_argv=argv,
        exact_shell_command=frozen_config["commands"]["development_one_shot"],
    )
    write_json(authorization_path, authorization)
    complete = manifest_for_tree(
        package, exclude=["launch_seal/complete_file_manifest.json"]
    )
    write_json(seal / "complete_file_manifest.json", complete)
    monkeypatch.chdir(repository)
    monkeypatch.setattr(runner, "__file__", str(snapshot_runner))
    monkeypatch.setattr(
        runner,
        "environment_record",
        lambda _repository, _config: copy.deepcopy(frozen_environment),
    )
    monkeypatch.setattr(runner, "_require_frozen_module_origins", lambda _snapshot: None)
    monkeypatch.setattr(
        runner,
        "_require_frozen_design_transition",
        lambda *args: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "_verify_or_create_predecessor_v1_evidence",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "_verify_or_create_predecessor_v2_evidence",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "reverify_prior_preservation",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(sys, "argv", [str(snapshot_runner), *argv])
    return {
        "repository": repository,
        "package": package,
        "snapshot": snapshot,
        "authorization": authorization_path,
        "output": output,
        "staging": staging,
        "inner_staging": inner_staging,
        "publication_seal": publication_seal,
        "claim": claim,
        "capability_receipt": capability_receipt,
        "argv": argv,
    }


def _issue_fixture_development_capability(fixture: dict) -> tuple[object, dict]:
    repository, frozen_config, authorization, parsed_contract, preflight_proof = (
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=4,
        )
    )
    assert repository == fixture["repository"]
    parsed_digest = canonical_digest(parsed_contract)
    claim = {
        "schema_version": "nursery-corrective-authorization-consumption-v3",
        "status": "ATTEMPT_CONSUMED",
        "authorization_digest": authorization["authorization_digest"],
        "resolved_authorization_path": str(fixture["authorization"]),
        "authorization_sha256": sha256_file(fixture["authorization"]),
        "resolved_snapshot_root": str(fixture["snapshot"]),
        "parsed_contract_digest": parsed_digest,
        "resolved_output_root": str(fixture["output"]),
        "resolved_staging_root": str(fixture["staging"]),
        "resolved_inner_staging_root": str(fixture["inner_staging"]),
        "resolved_capability_receipt_path": str(fixture["capability_receipt"]),
        "required_jobs": 4,
        "non_replayable": True,
        "created_before_any_guarded_development_operation": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }
    write_json(fixture["claim"], claim)
    arguments = {
        "authorization_digest": authorization["authorization_digest"],
        "parsed_contract_digest": parsed_digest,
        "claim_path": fixture["claim"],
        "claim_sha256": sha256_file(fixture["claim"]),
        "resolved_output_root": fixture["output"],
        "resolved_staging_root": fixture["staging"],
        "resolved_inner_staging_root": fixture["inner_staging"],
        "issuance_receipt_path": fixture["capability_receipt"],
        "required_jobs": 4,
        "preflight_proof": preflight_proof,
    }
    capability = issue_development_capability(**arguments)
    return capability, {"arguments": arguments, "config": frozen_config}


def test_capability_issuance_is_nonreplayable_and_path_job_bound(
    tmp_path, monkeypatch, config
):
    fixture = _make_preflight_package(tmp_path, monkeypatch, config)
    capability, context = _issue_fixture_development_capability(fixture)
    verify_development_capability(
        capability,
        required_jobs=4,
        required_scope="parent",
        resolved_output_root=fixture["output"],
        resolved_staging_root=fixture["staging"],
        resolved_inner_staging_root=fixture["inner_staging"],
    )
    with pytest.raises(AttributeError, match="immutable"):
        capability.required_jobs = 3
    object.__setattr__(capability, "required_jobs", 3)
    with pytest.raises(PermissionError, match="identity|claim|receipt"):
        verify_development_capability(capability)
    object.__setattr__(capability, "required_jobs", 4)
    verify_development_capability(capability, required_scope="parent")
    object.__setattr__(capability, "scope", "worker")
    object.__setattr__(capability, "unit_id", "forged-unit")
    with pytest.raises(PermissionError, match="identity"):
        verify_development_capability(capability)
    object.__setattr__(capability, "scope", "parent")
    object.__setattr__(capability, "unit_id", None)
    with pytest.raises(FileExistsError, match="pristine"):
        issue_development_capability(**context["arguments"])
    without_preflight_proof = {
        **context["arguments"],
        "preflight_proof": None,
    }
    with pytest.raises(PermissionError, match="preflight proof"):
        issue_development_capability(**without_preflight_proof)
    with pytest.raises(PermissionError, match="job count"):
        verify_development_capability(capability, required_jobs=3)
    with pytest.raises(PermissionError, match="origin mismatch|staging root mismatch"):
        _execute_development_cohort(
            repository_root=fixture["snapshot"],
            config_path=fixture["snapshot"]
            / "configs/synthetic_corrective_alignment_v3.yaml",
            output_root=fixture["staging"].with_name("alternate-staging"),
            config=context["config"],
            jobs=4,
            capability=capability,
        )
    alternate_claim = fixture["package"] / "alternate_consumed_claim.json"
    write_json(alternate_claim, read_json(fixture["claim"]))
    alternate_arguments = {
        **context["arguments"],
        "claim_path": alternate_claim,
        "claim_sha256": sha256_file(alternate_claim),
    }
    with pytest.raises(PermissionError, match="preflight proof|bind capability request"):
        issue_development_capability(**alternate_arguments)
    fixture["claim"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="claim changed"):
        verify_development_capability(capability)


def test_capability_rechecks_authorization_and_worker_cannot_forge_chain(
    tmp_path, monkeypatch, config
):
    fixture = _make_preflight_package(tmp_path, monkeypatch, config)
    capability, _context = _issue_fixture_development_capability(fixture)
    authorization = read_json(fixture["authorization"])
    authorization["required_jobs"] = 3
    write_json(fixture["authorization"], authorization, overwrite=True)
    with pytest.raises(PermissionError, match="authorization|authority|chain"):
        verify_development_capability(capability)

    forged = tmp_path / "forged"
    forged.mkdir()
    missing_authorization = forged / "missing-authorization.json"
    claim = forged / "claim.json"
    receipt = forged / "receipt.json"
    claim_value = {
        "schema_version": "nursery-corrective-authorization-consumption-v3",
        "status": "ATTEMPT_CONSUMED",
        "authorization_digest": "a" * 64,
        "resolved_authorization_path": str(missing_authorization),
        "authorization_sha256": "b" * 64,
        "resolved_snapshot_root": str(forged / "missing-snapshot"),
        "parsed_contract_digest": "c" * 64,
        "resolved_output_root": str(forged / "output"),
        "resolved_staging_root": str(forged / "staging"),
        "resolved_inner_staging_root": str(forged / "inner"),
        "resolved_capability_receipt_path": str(receipt),
        "required_jobs": 4,
        "non_replayable": True,
        "created_before_any_guarded_development_operation": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
    }
    write_json(claim, claim_value)
    receipt_payload = {
        "schema_version": "nursery-corrective-capability-issuance-v3",
        "status": "PARENT_CAPABILITY_ISSUED",
        "authorization_digest": "a" * 64,
        "parsed_contract_digest": "c" * 64,
        "claim_path": str(claim),
        "claim_sha256": sha256_file(claim),
        "resolved_output_root": str(forged / "output"),
        "resolved_staging_root": str(forged / "staging"),
        "resolved_inner_staging_root": str(forged / "inner"),
        "required_jobs": 4,
        "scope": "parent",
        "unit_id": None,
        "preflight_proof_identity_digest": "d" * 64,
        "parent_issuance_non_replayable": True,
    }
    write_json(
        receipt,
        {
            **receipt_payload,
            "receipt_digest": canonical_digest(receipt_payload),
        },
    )
    with pytest.raises((PermissionError, FileNotFoundError)):
        _issue_worker_development_capability(
            authorization_digest="a" * 64,
            parsed_contract_digest="c" * 64,
            claim_path=claim,
            claim_sha256=sha256_file(claim),
            issuance_receipt_path=receipt,
            issuance_receipt_sha256=sha256_file(receipt),
            resolved_output_root=forged / "output",
            resolved_staging_root=forged / "staging",
            resolved_inner_staging_root=forged / "inner",
            required_jobs=4,
            unit_id="forged-unit",
        )


def test_preflight_binds_actual_paths_jobs_argv_and_replay(tmp_path, monkeypatch, config):
    fixture = _make_preflight_package(tmp_path, monkeypatch, config)
    result = runner._strict_development_preflight(
        snapshot=fixture["snapshot"],
        authorization_path=fixture["authorization"],
        output_root=fixture["output"],
        jobs=4,
    )
    assert result[0] == fixture["repository"]
    alternate_authorization = fixture["repository"] / "alternate-authorization.json"
    shutil.copyfile(fixture["authorization"], alternate_authorization)
    with pytest.raises(PermissionError, match="authorization path"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=alternate_authorization,
            output_root=fixture["output"],
            jobs=4,
        )
    alternate_snapshot = (
        tmp_path
        / "alternate-repository/output/synthetic_corrective_development_launch_package_v3/frozen_source_snapshot"
    )
    shutil.copytree(fixture["snapshot"], alternate_snapshot)
    with pytest.raises(
        PermissionError, match="snapshot verification|exact frozen runner"
    ):
        runner._strict_development_preflight(
            snapshot=alternate_snapshot,
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=4,
        )
    with pytest.raises(PermissionError, match="worker count"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=3,
        )
    with pytest.raises(PermissionError, match="actual parsed path"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["repository"] / "output/alternate",
            jobs=4,
        )
    changed_argv = list(fixture["argv"])
    changed_argv[-1] = "1"
    monkeypatch.setattr(sys, "argv", [str(fixture["snapshot"] / "scripts/run_synthetic_corrective_alignment_v3.py"), *changed_argv])
    with pytest.raises(PermissionError, match="argv"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=4,
        )
    monkeypatch.setattr(sys, "argv", [str(fixture["snapshot"] / "scripts/run_synthetic_corrective_alignment_v3.py"), *fixture["argv"]])
    monkeypatch.setattr(
        runner,
        "reverify_prior_preservation",
        lambda *args, **kwargs: {"status": "FAIL"},
    )
    with pytest.raises(PermissionError, match="live predecessor|preservation"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=4,
        )
    monkeypatch.setattr(
        runner,
        "reverify_prior_preservation",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    write_json(fixture["claim"], {"consumed": True})
    with pytest.raises(FileExistsError, match="pristine"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=4,
        )


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "broken_symlink"])
def test_preflight_rejects_every_preexisting_output_type(
    tmp_path, monkeypatch, config, kind
):
    fixture = _make_preflight_package(tmp_path, monkeypatch, config)
    output = fixture["output"]
    if kind == "file":
        output.write_text("x")
    elif kind == "directory":
        output.mkdir()
    elif kind == "symlink":
        target = output.with_name("target")
        target.write_text("x")
        output.symlink_to(target.name)
    else:
        output.symlink_to("missing-target")
    with pytest.raises(FileExistsError, match="pristine"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=output,
            jobs=4,
        )


@pytest.mark.parametrize(
    "target_name", ["staging", "inner_staging", "claim", "capability_receipt"]
)
def test_preflight_requires_every_transaction_path_pristine(
    tmp_path, monkeypatch, config, target_name
):
    fixture = _make_preflight_package(tmp_path, monkeypatch, config)
    target = fixture[target_name]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="pristine"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=4,
        )


def test_preflight_rejects_symlinked_transaction_ancestry(
    tmp_path, monkeypatch, config
):
    fixture = _make_preflight_package(tmp_path, monkeypatch, config)
    real = fixture["repository"] / "real-output"
    real.mkdir()
    link = fixture["repository"] / "linked-output"
    link.symlink_to(real.name)
    authorization = read_json(fixture["authorization"])
    authorization["resolved_staging_root"] = str(link / "staging")
    authorization["authorization_digest"] = canonical_digest(
        {
            key: value
            for key, value in authorization.items()
            if key != "authorization_digest"
        }
    )
    fixture["authorization"].write_text(
        json.dumps(authorization, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=4,
        )


def test_failure_before_publish_leaves_final_output_absent(
    tmp_path, monkeypatch, config
):
    output = tmp_path / "scientific-output"

    def fake_prepare(root, *_args, **_kwargs):
        Path(root).mkdir(parents=True)
        return {"status": "PASS", "input_manifest_sha256": "a" * 64}

    def fail_parallel(**_kwargs):
        raise RuntimeError("injected worker failure")

    monkeypatch.setattr(parallel_module, "prepare_persisted_inputs", fake_prepare)
    monkeypatch.setattr(parallel_module, "run_parallel_from_persisted", fail_parallel)
    with pytest.raises(RuntimeError, match="injected worker failure"):
        execute_cohort(
            repository_root=ROOT,
            config_path=CONFIG,
            output_root=output,
            config=config,
            purpose="construction_micro",
            jobs=1,
        )
    assert not output.exists()
    assert not (output / "ONE_SHOT_OUTCOME.json").exists()


def test_successful_cohort_assembly_moves_summary_and_publishes_exact_tree(
    tmp_path, monkeypatch, config
):
    output = tmp_path / "cohort"
    summary = {
        "schema_version": "nursery-corrective-cohort-summary-v3",
        "status": "PASS",
        "purpose": "construction_micro",
        "scientific_decision": "SCIENTIFIC_INFERENCE_SUPPRESSED",
    }
    observed = {}

    def fake_prepare(root, *_args, **_kwargs):
        Path(root).mkdir(parents=True)
        write_json(Path(root) / "input.json", {"status": "PASS"})
        write_json(Path(root) / "input_manifest.json", {"status": "PASS"})
        return {"status": "PASS", "input_manifest_sha256": "a" * 64}

    def fake_parallel(*, output_root, expected_input_manifest_sha256, **_kwargs):
        observed["expected_input_manifest_sha256"] = expected_input_manifest_sha256
        compute = Path(output_root)
        compute.mkdir()
        for name in ("adjudication", "runtime", "shards"):
            child = compute / name
            child.mkdir()
            write_json(child / "evidence.json", {"status": "PASS"})
        write_json(
            compute / "adjudication/execution_contract.json",
            {"scientific_contract_digest": "c" * 64},
        )
        write_json(
            compute / "runtime/scheduler_validation.json",
            {"status": "PASS"},
        )
        write_json(compute / "cohort_summary.json", summary)
        return copy.deepcopy(summary)

    monkeypatch.setattr(parallel_module, "prepare_persisted_inputs", fake_prepare)
    monkeypatch.setattr(parallel_module, "run_parallel_from_persisted", fake_parallel)
    monkeypatch.setattr(
        parallel_module,
        "validate_subprocess_scheduler_artifacts",
        lambda **_kwargs: {"status": "PASS"},
    )
    result = execute_cohort(
        repository_root=ROOT,
        config_path=CONFIG,
        output_root=output,
        config=config,
        purpose="construction_micro",
        jobs=1,
    )
    assert result == summary
    assert observed["expected_input_manifest_sha256"] == "a" * 64
    assert read_json(output / "cohort_summary.json") == summary
    assert not (output / "parallel_compute").exists()
    complete = read_json(output / "complete_manifest.json")
    assert verify_exact_manifest(
        output,
        complete,
        manifest_filename="complete_manifest.json",
    )["status"] == "PASS"
    assert tree_file_paths(output) == [
        "adjudication/evidence.json",
        "adjudication/execution_contract.json",
        "cohort_summary.json",
        "complete_manifest.json",
        "persisted_inputs/input.json",
        "persisted_inputs/input_manifest.json",
        "runtime/evidence.json",
        "runtime/scheduler_validation.json",
        "shards/evidence.json",
    ]


def test_cohort_summary_mismatch_fails_before_atomic_publication(
    tmp_path, monkeypatch, config
):
    output = tmp_path / "cohort"

    def fake_prepare(root, *_args, **_kwargs):
        Path(root).mkdir(parents=True)
        return {"status": "PASS", "input_manifest_sha256": "a" * 64}

    def fake_parallel(*, output_root, **_kwargs):
        compute = Path(output_root)
        compute.mkdir()
        for name in ("adjudication", "runtime", "shards"):
            (compute / name).mkdir()
        write_json(compute / "cohort_summary.json", {"status": "PERSISTED"})
        return {"status": "RETURNED"}

    monkeypatch.setattr(parallel_module, "prepare_persisted_inputs", fake_prepare)
    monkeypatch.setattr(parallel_module, "run_parallel_from_persisted", fake_parallel)
    with pytest.raises(RuntimeError, match="persisted cohort summary"):
        execute_cohort(
            repository_root=ROOT,
            config_path=CONFIG,
            output_root=output,
            config=config,
            purpose="construction_micro",
            jobs=1,
        )
    with pytest.raises(FileNotFoundError):
        output.lstat()


def _make_authorized_development_staging(
    tmp_path: Path,
    monkeypatch,
    config,
) -> tuple[Path, Path, dict, dict, object, object]:
    fixture = _make_preflight_package(tmp_path, monkeypatch, config)
    capability, _context = _issue_fixture_development_capability(fixture)
    staging = fixture["staging"]
    output = fixture["output"]
    staging.mkdir()
    write_json(staging / "payload.json", {"candidate": "complete"})
    summary = {
        "candidate_decision": "CORRECTIVE_STUDY_STOP_NO_SUPPORT",
        "scientific_contract_digest": "c" * 64,
    }
    write_json(staging / "cohort_summary.json", summary)
    manifest = manifest_for_tree(staging, exclude=["complete_manifest.json"])
    write_json(staging / "complete_manifest.json", manifest)
    authorization = read_json(fixture["authorization"])
    proof = runner._issue_development_publication_proof(
        staging=staging,
        output_root=output,
        authorization=authorization,
        summary=summary,
        capability=capability,
    )
    return staging, output, authorization, summary, capability, proof


@pytest.mark.parametrize(
    "failpoint", ["seal_write", "published_verify", "rename", "destination_race"]
)
def test_atomic_publication_failpoints_leave_authorized_final_root_absent(
    tmp_path, monkeypatch, config, failpoint
):
    staging, output, authorization, summary, capability, proof = (
        _make_authorized_development_staging(tmp_path, monkeypatch, config)
    )
    if failpoint == "seal_write":
        original_write = runner.write_json

        def injected_write(path, value):
            if Path(path).name == "SCIENTIFIC_OUTCOME.json":
                raise RuntimeError("injected seal failure")
            return original_write(path, value)

        monkeypatch.setattr(runner, "write_json", injected_write)
    elif failpoint == "published_verify":
        original_verify = runner.verify_exact_manifest
        calls = {"count": 0}

        def injected_verify(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                return {"status": "FAIL", "problems": ["injected"]}
            return original_verify(*args, **kwargs)

        monkeypatch.setattr(runner, "verify_exact_manifest", injected_verify)
    elif failpoint == "rename":
        monkeypatch.setattr(
            runner,
            "atomic_rename_directory_noreplace",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("injected rename failure")),
        )
    else:
        def inject_destination_race(source, destination):
            Path(destination).mkdir()
            return atomic_rename_directory_noreplace(source, destination)

        monkeypatch.setattr(
            runner,
            "atomic_rename_directory_noreplace",
            inject_destination_race,
        )
    with pytest.raises((RuntimeError, FileExistsError), match="injected|published output|already exists"):
        runner._publish_development_staging(
            staging=staging,
            output_root=output,
            authorization=authorization,
            summary=summary,
            capability=capability,
            publication_proof=proof,
        )
    if failpoint == "destination_race":
        assert output.is_dir()
        assert not (output / "SCIENTIFIC_OUTCOME.json").exists()
        assert staging.is_dir()
    else:
        assert not output.exists()
    assert not output.is_symlink()


def test_atomic_publication_success_is_one_exact_directory_tree(
    tmp_path, monkeypatch, config
):
    staging, output, authorization, summary, capability, proof = (
        _make_authorized_development_staging(tmp_path, monkeypatch, config)
    )
    result = runner._publish_development_staging(
        staging=staging,
        output_root=output,
        authorization=authorization,
        summary=summary,
        capability=capability,
        publication_proof=proof,
    )
    assert result["scientific_outcome"] is True
    assert not staging.exists()
    published = read_json(output / "published_complete_manifest.json")
    assert verify_exact_manifest(
        output,
        published,
        manifest_filename="published_complete_manifest.json",
    )["status"] == "PASS"
    assert read_json(output / "SCIENTIFIC_OUTCOME.json")["publication_state"] == (
        "EFFECTIVE_ONLY_AT_AUTHORIZED_FINAL_ROOT"
    )


def test_publication_proof_rejects_summary_or_scope_substitution(
    tmp_path, monkeypatch, config
):
    staging, output, authorization, summary, capability, _proof = (
        _make_authorized_development_staging(tmp_path, monkeypatch, config)
    )
    changed_summary = {**summary, "candidate_decision": "GO"}
    with pytest.raises(PermissionError, match="summary"):
        runner._issue_development_publication_proof(
            staging=staging,
            output_root=output,
            authorization=authorization,
            summary=changed_summary,
            capability=capability,
        )
    object.__setattr__(capability, "scope", "worker")
    object.__setattr__(capability, "unit_id", "forged")
    with pytest.raises(PermissionError, match="identity|scope"):
        runner._issue_development_publication_proof(
            staging=staging,
            output_root=output,
            authorization=authorization,
            summary=summary,
            capability=capability,
        )


def test_publication_rechecks_persisted_authorization_after_proof(
    tmp_path, monkeypatch, config
):
    staging, output, authorization, summary, capability, proof = (
        _make_authorized_development_staging(tmp_path, monkeypatch, config)
    )
    changed = copy.deepcopy(authorization)
    changed["authorization_digest"] = "f" * 64
    with pytest.raises(PermissionError, match="authorization changed"):
        runner._publish_development_staging(
            staging=staging,
            output_root=output,
            authorization=changed,
            summary=summary,
            capability=capability,
            publication_proof=proof,
        )
    assert not output.exists()


def test_direct_scientific_publication_without_capability_is_rejected(tmp_path):
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    summary = {
        "candidate_decision": "GO",
        "scientific_contract_digest": "c" * 64,
    }
    write_json(staging / "cohort_summary.json", summary)
    write_json(staging / "complete_manifest.json", manifest_for_tree(staging))
    with pytest.raises(PermissionError, match="publication|capability|issued"):
        runner._publish_development_staging(
            staging=staging,
            output_root=output,
            authorization={
                "resolved_publication_seal_path": str(
                    output / "SCIENTIFIC_OUTCOME.json"
                ),
                "authorization_digest": "a" * 64,
            },
            summary=summary,
            capability=None,
            publication_proof=None,
        )
    assert not output.exists()


def test_atomic_directory_publication_never_replaces_existing_destination(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    write_json(source / "payload.json", {"source": True})
    with pytest.raises(FileExistsError):
        atomic_rename_directory_noreplace(source, destination)
    assert source.is_dir()
    assert destination.is_dir()
    assert not (destination / "payload.json").exists()


def test_finalize_package_requires_sealed_evidence_proof(tmp_path, config):
    package = tmp_path / "output/synthetic_corrective_development_launch_package_v3"
    package.mkdir(parents=True)
    with pytest.raises(PermissionError, match="finalization proof|launch seal"):
        integrity_module.finalize_package(
            tmp_path,
            package,
            config,
            gates={name: True for name in PACKAGE_GATES},
            finalization_proof=None,
        )
    snapshot = package / "frozen_source_snapshot"
    (snapshot / "configs").mkdir(parents=True)
    (snapshot / "scripts").mkdir()
    (snapshot / "configs/synthetic_corrective_alignment_v3.yaml").write_bytes(
        CONFIG.read_bytes()
    )
    (snapshot / "scripts/run_synthetic_corrective_alignment_v3.py").write_text(
        "# frozen fixture runner\n", encoding="utf-8"
    )
    for path, value in (
            (package / "prequalification_design_lock.json", {"status": "LOCKED"}),
            (package / "micro_attempt_consumed.json", {"status": "CONSUMED"}),
            (package / "worker_launcher_preflight.json", {"status": "PASS"}),
        (snapshot / "snapshot_manifest.json", {"status": "PASS"}),
        (package / "freeze_receipt.json", {"status": "FROZEN"}),
        (package / "preservation_attempt_consumed.json", {"status": "CONSUMED"}),
        (package / "preservation/preservation_proof.json", {"status": "PASS"}),
        (package / "preservation_timing.json", {"status": "PASS"}),
        (package / "preservation_completion.json", {"status": "FAIL"}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    with pytest.raises(RuntimeError, match="preservation completion chain"):
        runner._consume_finalization_attempt_and_validate_preservation(
            package, snapshot, config
        )
    assert (package / "finalization_attempt_consumed.json").is_file()
    with pytest.raises(FileExistsError, match="pristine"):
        runner._consume_finalization_attempt_and_validate_preservation(
            package, snapshot, config
        )


def test_finalize_package_revalidates_forged_factory_proof_and_config(
    tmp_path, monkeypatch, config
):
    root = tmp_path / "repository"
    package = root / "package"
    snapshot_config = (
        package
        / "frozen_source_snapshot/configs/synthetic_corrective_alignment_v3.yaml"
    )
    snapshot_config.parent.mkdir(parents=True)
    snapshot_config.write_text("protocol: frozen\n", encoding="utf-8")
    gates = {name: True for name in PACKAGE_GATES}
    write_json(
        package / "adjudication_inputs.json",
        {
            "schema_version": "nursery-corrective-package-adjudication-input-v3",
            "gates": gates,
            "contradictions": [],
            "development_outcome_count": 0,
            "confirmation_outcome_count": 0,
        },
    )
    frozen_config = copy.deepcopy(config)
    frozen_config["protocol"]["status"] = "frozen"
    forged = integrity_module.FinalizationProof(
        repository_root=str(root.resolve()),
        package_root=str(package.resolve()),
        config_digest=canonical_digest(frozen_config),
        snapshot_config_sha256=sha256_file(snapshot_config),
        gates_digest=canonical_digest(gates),
        adjudication_inputs_sha256=sha256_file(
            package / "adjudication_inputs.json"
        ),
        evidence_hashes={},
        validation_digest="0" * 64,
        _seal=integrity_module._FINALIZATION_PROOF_FACTORY_SEAL,
    )
    monkeypatch.setattr(
        integrity_module,
        "_issue_finalization_proof",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("fresh verification required")
        ),
    )
    with pytest.raises(PermissionError, match="fresh verification required"):
        integrity_module.finalize_package(
            root,
            package,
            frozen_config,
            gates=gates,
            finalization_proof=forged,
        )
    changed_config = copy.deepcopy(frozen_config)
    changed_config["learner"]["epochs"] += 1
    with pytest.raises(PermissionError, match="evidence changed"):
        integrity_module.finalize_package(
            root,
            package,
            changed_config,
            gates=gates,
            finalization_proof=forged,
        )
    assert not (package / "launch_seal").exists()


def test_runner_final_gate_keys_exactly_match_adjudicator_contract():
    tree = ast.parse(textwrap.dedent(inspect.getsource(runner._finalize)))
    gate_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gates"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    ]
    assert len(gate_assignments) == 1
    keys = {
        key.value
        for key in gate_assignments[0].value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert keys == set(PACKAGE_GATES)


def _scheduler_arguments(
    tmp_path,
    config,
    unit,
    purpose,
    jobs,
    boundary,
    input_manifest_sha256,
    *,
    input_root=None,
    shard_base=None,
):
    repository = Path(str(boundary["repository_root"]))
    input_root = Path(input_root) if input_root is not None else tmp_path / "inputs"
    shard_base = Path(shard_base) if shard_base is not None else tmp_path / "shards"
    return {
        "unit": unit,
        "purpose": purpose,
        "repository_root": str(repository),
        "config_path": str(
            repository / "configs/synthetic_corrective_alignment_v3.yaml"
        ),
        "input_root": str(input_root),
        "input_manifest_sha256": input_manifest_sha256,
        "shard_base": str(shard_base),
        "thread_environment": config["parallel"]["thread_environment"],
        "authorization_digest": "",
        "parsed_contract_digest": "",
        "scientific_contract_digest": "2" * 64,
        "claim_path": "",
        "claim_sha256": "",
        "issuance_receipt_path": "",
        "issuance_receipt_sha256": "",
        "development_output_root": "",
        "development_staging_root": "",
        "development_inner_staging_root": "",
        "rehearsal_snapshot_root": "",
        "rehearsal_output_root": "",
        "rehearsal_parallel_output_root": "",
        "rehearsal_receipt_path": "",
        "rehearsal_receipt_sha256": "",
        "required_jobs": int(jobs),
        "worker_boundary_commitment": boundary,
        "expected_launcher_evidence": {
            "chain": boundary["python_launcher_chain"],
            "chain_digest": boundary["python_launcher_chain_digest"],
            "pyvenv_cfg_path": boundary["pyvenv_cfg_path"],
            "pyvenv_cfg_sha256": boundary["pyvenv_cfg_sha256"],
        },
        "worker_python_startup_flags": dict(
            config["environment"]["worker_python_startup_flags"]
        ),
    }


def _boundary_repository(tmp_path, config, lock_value=None):
    repository = tmp_path / "boundary-repository"
    relatives = sorted(
        {
            *parallel_module._WORKER_OPERATIONAL_RELATIVES,
            str(config["registries"]["canonical_prior_registry"]),
        }
    )
    for relative in relatives:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    bound_config = load_config(
        repository / "configs/synthetic_corrective_alignment_v3.yaml",
        repository_root=repository,
    )
    lock = (
        repository
        / str(bound_config["paths"]["package_root"])
        / "prequalification_design_lock.json"
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        lock,
        lock_value
        if lock_value is not None
        else {"status": "LOCKED_BEFORE_OFFICIAL_FIXTURES"},
    )
    return repository, bound_config, lock


def _frozen_boundary_repository(tmp_path, config):
    live_root = tmp_path / "frozen-live-root"
    package = (
        live_root
        / "output/synthetic_corrective_development_launch_package_v3"
    )
    snapshot = package / "frozen_source_snapshot"
    relatives = sorted(
        {
            *parallel_module._WORKER_OPERATIONAL_RELATIVES,
            str(config["registries"]["canonical_prior_registry"]),
        }
    )
    for relative in relatives:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    frozen_config = load_config(
        snapshot / "configs/synthetic_corrective_alignment_v3.yaml",
        repository_root=snapshot,
    )
    frozen_config["protocol"]["status"] = "frozen"
    for path, value in (
        (snapshot / "snapshot_manifest.json", {"status": "PASS"}),
        (package / "freeze_receipt.json", {"status": "FROZEN"}),
        (package / "frozen_environment.json", {"status": "FROZEN"}),
        (
            package / "prequalification_design_lock.json",
            {"status": "LOCKED_BEFORE_OFFICIAL_FIXTURES"},
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    return snapshot, frozen_config, package / "prequalification_design_lock.json"


def _complete_scheduler_validation_fixture(tmp_path, config, *, cohort_layout=False):
    staging = tmp_path
    compute = tmp_path
    if cohort_layout:
        staging = tmp_path / ".published.cohort-staging"
        compute = staging / "parallel_compute"
    runtime = compute / "runtime"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.mkdir()
    inputs = staging / ("persisted_inputs" if cohort_layout else "inputs")
    inputs.mkdir()
    write_json(inputs / "input_manifest.json", {"fixture": "scheduler-anchor"})
    input_manifest_sha256 = sha256_file(inputs / "input_manifest.json")
    adjudication = compute / "adjudication"
    adjudication.mkdir()
    write_json(
        adjudication / "execution_contract.json",
        {
            "purpose": "construction_micro",
            "input_manifest_sha256": input_manifest_sha256,
            "authorization_digest": None,
            "parsed_contract_digest": None,
            "scientific_contract_digest": "2" * 64,
        },
    )
    repository, bound_config, lock = _boundary_repository(tmp_path, config)
    boundary = create_worker_boundary_commitment(
        repository_root=repository,
        config=bound_config,
        prequalification_lock=lock,
    )
    launcher_evidence = {
        "chain": boundary["python_launcher_chain"],
        "chain_digest": boundary["python_launcher_chain_digest"],
        "pyvenv_cfg_path": boundary["pyvenv_cfg_path"],
        "pyvenv_cfg_sha256": boundary["pyvenv_cfg_sha256"],
    }
    execution_contract = read_json(adjudication / "execution_contract.json")
    execution_contract["python_launcher_evidence"] = launcher_evidence
    write_json(
        adjudication / "execution_contract.json",
        execution_contract,
        overwrite=True,
    )
    unit = build_work_plan(bound_config, purpose="construction_micro")["units"][0]
    plan = {
        "purpose": "construction_micro",
        "model_seed_order": [int(unit["model_seed"])],
        "units": [unit],
    }
    child_environment = _child_environment_base(bound_config)
    scheduler = _prepare_subprocess_requests(
        plan=plan,
        arguments_by_id={
            unit["unit_id"]: _scheduler_arguments(
                tmp_path,
                bound_config,
                unit,
                "construction_micro",
                1,
                boundary,
                input_manifest_sha256,
                input_root=inputs,
                shard_base=compute / "shards",
            )
        },
        runtime_root=runtime,
        child_environment_base=child_environment,
    )
    shards = compute / "shards"
    shards.mkdir()
    shard = shards / unit["unit_id"]
    shard.mkdir()
    write_json(shard / "result.json", {"unit_id": unit["unit_id"]})
    write_json(shard / "operation_ledger.json", {"unit_id": unit["unit_id"]})
    shard_manifest = manifest_for_paths(
        shard,
        ["result.json", "operation_ledger.json"],
    )
    write_json(shard / "manifest.json", shard_manifest)
    write_json(
        shard / "COMPLETE.json",
        {
            "schema_version": "nursery-corrective-shard-complete-v3",
            "unit_id": unit["unit_id"],
            "result_sha256": sha256_file(shard / "result.json"),
            "ledger_sha256": sha256_file(shard / "operation_ledger.json"),
            "manifest_sha256": sha256_file(shard / "manifest.json"),
            "status": "COMPLETE",
        },
    )
    request_path = scheduler["request_paths"][unit["unit_id"]]
    request = read_json(request_path)
    telemetry = {
        "unit_id": unit["unit_id"],
        "shard_manifest_sha256": sha256_file(shard / "manifest.json"),
        "wall_seconds": 0.01,
        "worker_peak_rss_native_units": 1,
        "bootstrap_environment_digest": canonical_digest(
            request["arguments"]["expected_bootstrap_environment"]
        ),
        "worker_boundary_digest": boundary["boundary_digest"],
        "worker_boundary_operational_file_count": len(
            boundary["operational_manifest"]["files"]
        ),
        "worker_input_regular_files_verified": 1,
        "worker_full_tree_verifications": 0,
        "thread_environment_observed": {
            str(name): str(value)
            for name, value in config["parallel"]["thread_environment"].items()
        },
        "native_threadpools": [
            {
                "internal_api": row["internal_api"],
                "prefix": row["prefix"],
                "num_threads": row["num_threads"],
            }
            for row in boundary["native_threadpools"]
        ],
        "python_startup_flags_observed": dict(
            config["environment"]["worker_python_startup_flags"]
        ),
    }
    write_json(
        scheduler["receipts"] / f"{unit['unit_id']}.json",
        {
            "schema_version": "nursery-corrective-subprocess-success-v3",
            "status": "PASS",
            "unit_id": unit["unit_id"],
            "request_sha256": sha256_file(request_path),
            "request_manifest_sha256": scheduler["request_manifest_sha256"],
            "telemetry": telemetry,
            "scientific_outcome": False,
        },
    )
    for suffix in ("stdout", "stderr"):
        (scheduler["logs"] / f"{unit['unit_id']}.{suffix}").write_bytes(b"")
    receipt_manifest = manifest_for_paths(
        scheduler["receipts"], [f"{unit['unit_id']}.json"]
    )
    log_manifest = manifest_for_paths(
        scheduler["logs"],
        [f"{unit['unit_id']}.stdout", f"{unit['unit_id']}.stderr"],
    )
    write_json(scheduler["receipts"] / "manifest.json", receipt_manifest)
    write_json(scheduler["logs"] / "manifest.json", log_manifest)
    (scheduler["bootstrap_tmp"] / unit["unit_id"]).rmdir()
    scheduler["bootstrap_tmp"].rmdir()
    scheduler["failures"].rmdir()
    write_json(
        scheduler["root"] / "scheduler_summary.json",
        {
            "schema_version": "nursery-corrective-subprocess-scheduler-v3",
            "status": "PASS",
            "backend": "direct_subprocess_v1",
            "jobs": 1,
            "launch_order": [unit["unit_id"]],
            "maximum_observed_concurrency": 1,
            "wave_key": "model_seed",
            "waves": [
                {
                    "model_seed": int(unit["model_seed"]),
                    "unit_ids": [unit["unit_id"]],
                    "launch_order_slice": [unit["unit_id"]],
                    "complete_before_next_wave": True,
                }
            ],
            "request_manifest_sha256": scheduler["request_manifest_sha256"],
            "receipt_manifest_sha256": sha256_file(
                scheduler["receipts"] / "manifest.json"
            ),
            "log_manifest_sha256": sha256_file(
                scheduler["logs"] / "manifest.json"
            ),
            "validated_empty_failure_and_bootstrap_roots_removed": True,
            "scheduler_parent_pid": os.getpid(),
            "scientific_outcome": False,
        },
    )
    return (
        plan,
        scheduler,
        shards,
        child_environment,
        repository,
        bound_config,
        runtime,
        inputs,
    )


def test_scheduler_evidence_is_independently_recomputed_and_mutations_fail(
    tmp_path,
    config,
    monkeypatch,
):
    (
        plan,
        scheduler,
        shards,
        child_environment,
        repository,
        bound_config,
        runtime,
        inputs,
    ) = (
        _complete_scheduler_validation_fixture(tmp_path, config)
    )
    projected_config = copy.deepcopy(bound_config)
    projected_config.pop("repository_root")
    monkeypatch.setattr(
        parallel_module,
        "_require_project_module_origins",
        lambda _repository: None,
    )
    report = validate_subprocess_scheduler_artifacts(
        runtime_root=runtime,
        shard_root=shards,
        plan=plan,
        jobs=1,
        config=projected_config,
        expected_repository_root=repository,
        expected_config_path=(
            repository / "configs/synthetic_corrective_alignment_v3.yaml"
        ),
        child_environment_base=child_environment,
        input_root=inputs,
        expected_input_manifest_sha256=sha256_file(
            inputs / "input_manifest.json"
        ),
        expected_scientific_contract_digest="2" * 64,
        persist=True,
    )
    assert report["status"] == "PASS"
    assert report == read_json(tmp_path / "runtime/scheduler_validation.json")

    relocation_base = tmp_path / "relocation"
    (
        relocated_plan,
        _relocated_scheduler,
        relocated_shards,
        relocated_child_environment,
        relocated_repository,
        relocated_config,
        relocated_runtime,
        relocated_inputs,
    ) = _complete_scheduler_validation_fixture(
        relocation_base, config, cohort_layout=True
    )
    relocated_projected = copy.deepcopy(relocated_config)
    relocated_projected.pop("repository_root")
    immediate_relocation_report = validate_subprocess_scheduler_artifacts(
        runtime_root=relocated_runtime,
        shard_root=relocated_shards,
        plan=relocated_plan,
        jobs=1,
        config=relocated_projected,
        expected_repository_root=relocated_repository,
        expected_config_path=(
            relocated_repository / "configs/synthetic_corrective_alignment_v3.yaml"
        ),
        child_environment_base=relocated_child_environment,
        input_root=relocated_inputs,
        expected_input_manifest_sha256=sha256_file(
            relocated_inputs / "input_manifest.json"
        ),
        expected_scientific_contract_digest="2" * 64,
        persist=True,
    )
    published = relocation_base / "published"
    relocation_staging = relocation_base / ".published.cohort-staging"
    relocation_compute = relocation_staging / "parallel_compute"
    for name in ("adjudication", "runtime", "shards"):
        shutil.move(str(relocation_compute / name), str(relocation_staging / name))
    relocation_compute.rmdir()
    relocation_staging.rename(published)
    published_report = validate_subprocess_scheduler_artifacts(
        runtime_root=published / "runtime",
        shard_root=published / "shards",
        plan=relocated_plan,
        jobs=1,
        config=relocated_projected,
        expected_repository_root=relocated_repository,
        expected_config_path=(
            relocated_repository / "configs/synthetic_corrective_alignment_v3.yaml"
        ),
        child_environment_base=relocated_child_environment,
        input_root=published / "persisted_inputs",
        expected_input_manifest_sha256=sha256_file(
            published / "persisted_inputs/input_manifest.json"
        ),
        expected_scientific_contract_digest="2" * 64,
        persist=False,
        relocated_from_cohort_staging=relocation_staging,
    )
    assert published_report == immediate_relocation_report
    with pytest.raises(RuntimeError, match="relocation layout"):
        validate_subprocess_scheduler_artifacts(
            runtime_root=published / "runtime",
            shard_root=published / "shards",
            plan=relocated_plan,
            jobs=1,
            config=relocated_projected,
            expected_repository_root=relocated_repository,
            expected_config_path=(
                relocated_repository
                / "configs/synthetic_corrective_alignment_v3.yaml"
            ),
            child_environment_base=relocated_child_environment,
            input_root=published / "persisted_inputs",
            expected_input_manifest_sha256=sha256_file(
                published / "persisted_inputs/input_manifest.json"
            ),
            expected_scientific_contract_digest="2" * 64,
            persist=False,
            relocated_from_cohort_staging=relocation_base / "wrong-staging",
        )
    relocation_staging.mkdir()
    with pytest.raises(RuntimeError, match="survived publication"):
        validate_subprocess_scheduler_artifacts(
            runtime_root=published / "runtime",
            shard_root=published / "shards",
            plan=relocated_plan,
            jobs=1,
            config=relocated_projected,
            expected_repository_root=relocated_repository,
            expected_config_path=(
                relocated_repository
                / "configs/synthetic_corrective_alignment_v3.yaml"
            ),
            child_environment_base=relocated_child_environment,
            input_root=published / "persisted_inputs",
            expected_input_manifest_sha256=sha256_file(
                published / "persisted_inputs/input_manifest.json"
            ),
            expected_scientific_contract_digest="2" * 64,
            persist=False,
            relocated_from_cohort_staging=relocation_staging,
        )
    relocation_staging.rmdir()
    receipt_path = next(
        path
        for path in scheduler["receipts"].glob("*.json")
        if path.name != "manifest.json"
    )
    receipt_bytes = receipt_path.read_bytes()
    manifest_bytes = (scheduler["receipts"] / "manifest.json").read_bytes()
    receipt = read_json(receipt_path)
    receipt["telemetry"]["worker_full_tree_verifications"] = 1
    receipt_path.write_bytes(parallel_module.canonical_bytes(receipt))
    (scheduler["receipts"] / "manifest.json").write_bytes(
        parallel_module.canonical_bytes(
            manifest_for_paths(scheduler["receipts"], [receipt_path.name])
        )
    )
    summary_path = scheduler["root"] / "scheduler_summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = read_json(summary_path)
    summary["receipt_manifest_sha256"] = sha256_file(
        scheduler["receipts"] / "manifest.json"
    )
    summary_path.write_bytes(parallel_module.canonical_bytes(summary))
    with pytest.raises(RuntimeError, match="receipt/telemetry"):
        validate_subprocess_scheduler_artifacts(
            runtime_root=tmp_path / "runtime",
            shard_root=shards,
            plan=plan,
            jobs=1,
            config=projected_config,
            expected_repository_root=repository,
            expected_config_path=(
                repository / "configs/synthetic_corrective_alignment_v3.yaml"
            ),
            child_environment_base=child_environment,
            input_root=tmp_path / "inputs",
            expected_input_manifest_sha256=sha256_file(
                tmp_path / "inputs/input_manifest.json"
            ),
            expected_scientific_contract_digest="2" * 64,
            persist=False,
        )
    receipt_path.write_bytes(receipt_bytes)
    (scheduler["receipts"] / "manifest.json").write_bytes(manifest_bytes)
    summary_path.write_bytes(summary_bytes)
    request_path = next(
        path
        for path in scheduler["requests"].glob("*.json")
        if path.name != "manifest.json"
    )
    request_bytes = request_path.read_bytes()
    request_manifest_path = scheduler["requests"] / "manifest.json"
    request_manifest_bytes = request_manifest_path.read_bytes()
    for field, replacement in (
        ("input_root", str(tmp_path / "alternate-inputs")),
        ("config_path", str(tmp_path / "alternate-config.yaml")),
        ("shard_base", str(tmp_path / "alternate-shards")),
        ("scientific_contract_digest", "f" * 64),
        ("claim_path", str(tmp_path / "forged-claim.json")),
    ):
        request = read_json(request_path)
        request["arguments"][field] = replacement
        request_path.write_bytes(parallel_module.canonical_bytes(request))
        request_manifest_path.write_bytes(
            parallel_module.canonical_bytes(
                manifest_for_paths(scheduler["requests"], [request_path.name])
            )
        )
        receipt = read_json(receipt_path)
        receipt["request_sha256"] = sha256_file(request_path)
        receipt["request_manifest_sha256"] = sha256_file(request_manifest_path)
        receipt_path.write_bytes(parallel_module.canonical_bytes(receipt))
        (scheduler["receipts"] / "manifest.json").write_bytes(
            parallel_module.canonical_bytes(
                manifest_for_paths(scheduler["receipts"], [receipt_path.name])
            )
        )
        summary = read_json(summary_path)
        summary["request_manifest_sha256"] = sha256_file(request_manifest_path)
        summary["receipt_manifest_sha256"] = sha256_file(
            scheduler["receipts"] / "manifest.json"
        )
        summary_path.write_bytes(parallel_module.canonical_bytes(summary))
        with pytest.raises(RuntimeError, match="request contract"):
            validate_subprocess_scheduler_artifacts(
                runtime_root=tmp_path / "runtime",
                shard_root=shards,
                plan=plan,
                jobs=1,
                config=projected_config,
                expected_repository_root=repository,
                expected_config_path=(
                    repository / "configs/synthetic_corrective_alignment_v3.yaml"
                ),
                child_environment_base=child_environment,
                input_root=tmp_path / "inputs",
                expected_input_manifest_sha256=sha256_file(
                    tmp_path / "inputs/input_manifest.json"
                ),
                expected_scientific_contract_digest="2" * 64,
                persist=False,
            )
        request_path.write_bytes(request_bytes)
        request_manifest_path.write_bytes(request_manifest_bytes)
        receipt_path.write_bytes(receipt_bytes)
        (scheduler["receipts"] / "manifest.json").write_bytes(manifest_bytes)
        summary_path.write_bytes(summary_bytes)
    summary = read_json(summary_path)
    summary["maximum_observed_concurrency"] = 0
    summary_path.write_bytes(parallel_module.canonical_bytes(summary))
    with pytest.raises(RuntimeError, match="summary contract"):
        validate_subprocess_scheduler_artifacts(
            runtime_root=tmp_path / "runtime",
            shard_root=shards,
            plan=plan,
            jobs=1,
            config=projected_config,
            expected_repository_root=repository,
            expected_config_path=(
                repository / "configs/synthetic_corrective_alignment_v3.yaml"
            ),
            child_environment_base=child_environment,
            input_root=tmp_path / "inputs",
            expected_input_manifest_sha256=sha256_file(
                tmp_path / "inputs/input_manifest.json"
            ),
            expected_scientific_contract_digest="2" * 64,
            persist=False,
        )


def test_worker_boundary_and_member_validation_are_bounded_and_mutation_safe(
    tmp_path,
    monkeypatch,
    config,
):
    repository, bound_config, lock = _boundary_repository(
        tmp_path,
        config,
        {"status": "LOCKED"},
    )
    boundary = create_worker_boundary_commitment(
        repository_root=repository,
        config=bound_config,
        prequalification_lock=lock,
    )
    verification = verify_worker_boundary_commitment(
        boundary,
        repository_root=repository,
        config=bound_config,
        require_loaded_module_origins=False,
    )
    assert verification["status"] == "PASS"
    assert verification["full_tree_verifications"] == 0
    resealed = copy.deepcopy(boundary)
    resealed["operational_manifest"]["files"][0]["sha256"] = "0" * 64
    resealed["operational_manifest"]["digest"] = canonical_digest(
        resealed["operational_manifest"]["files"]
    )
    resealed["boundary_digest"] = canonical_digest(
        {key: value for key, value in resealed.items() if key != "boundary_digest"}
    )
    with pytest.raises(PermissionError, match="operational-boundary"):
        verify_worker_boundary_commitment(
            resealed,
            repository_root=repository,
            config=bound_config,
            require_loaded_module_origins=False,
        )

    with pytest.raises(PermissionError, match="lock path mismatch"):
        create_worker_boundary_commitment(
            repository_root=repository,
            config=bound_config,
            prequalification_lock=(
                repository / "configs/synthetic_corrective_alignment_v3.yaml"
            ),
        )
    frozen_repository, frozen_config, frozen_lock = _frozen_boundary_repository(
        tmp_path, config
    )
    frozen_boundary = create_worker_boundary_commitment(
        repository_root=frozen_repository,
        config=frozen_config,
        prequalification_lock=frozen_lock,
    )
    assert len(frozen_boundary["special_evidence"]) == 4
    assert [row["path"] for row in frozen_boundary["special_evidence"]] == sorted(
        [
            str(frozen_repository / "snapshot_manifest.json"),
            str(frozen_repository.parent / "freeze_receipt.json"),
            str(frozen_repository.parent / "frozen_environment.json"),
            str(frozen_lock),
        ]
    )
    assert verify_worker_boundary_commitment(
        frozen_boundary,
        repository_root=frozen_repository,
        config=frozen_config,
        require_loaded_module_origins=False,
    )["status"] == "PASS"

    unrelated = tmp_path / "unrelated-special-evidence.json"
    unrelated.write_bytes(frozen_lock.read_bytes())
    unrelated_stat = unrelated.lstat()
    unrelated_row = {
        "path": str(unrelated),
        "bytes": unrelated_stat.st_size,
        "mode": format(unrelated_stat.st_mode & 0o777, "04o"),
        "sha256": sha256_file(unrelated),
    }
    special_mutations = []
    omitted = copy.deepcopy(frozen_boundary)
    omitted["special_evidence"] = omitted["special_evidence"][:-1]
    special_mutations.append(omitted)
    duplicate = copy.deepcopy(frozen_boundary)
    duplicate["special_evidence"].append(
        copy.deepcopy(duplicate["special_evidence"][0])
    )
    special_mutations.append(duplicate)
    added = copy.deepcopy(frozen_boundary)
    added["special_evidence"].append(unrelated_row)
    special_mutations.append(added)
    substituted = copy.deepcopy(frozen_boundary)
    substituted["special_evidence"][0] = unrelated_row
    special_mutations.append(substituted)
    reordered = copy.deepcopy(frozen_boundary)
    reordered["special_evidence"] = list(
        reversed(reordered["special_evidence"])
    )
    special_mutations.append(reordered)
    traversal = copy.deepcopy(frozen_boundary)
    first_path = Path(traversal["special_evidence"][0]["path"])
    traversal["special_evidence"][0]["path"] = str(
        first_path.parent / "ignored" / ".." / first_path.name
    )
    special_mutations.append(traversal)
    for changed_special in special_mutations:
        changed_special["boundary_digest"] = canonical_digest(
            {
                key: value
                for key, value in changed_special.items()
                if key != "boundary_digest"
            }
        )
        with pytest.raises(PermissionError, match="special-evidence inventory"):
            verify_worker_boundary_commitment(
                changed_special,
                repository_root=frozen_repository,
                config=frozen_config,
                require_loaded_module_origins=False,
            )

    traversal_config = copy.deepcopy(bound_config)
    traversal_config["paths"]["package_root"] = "output/../escape"
    with pytest.raises(PermissionError, match="package layout"):
        parallel_module._worker_boundary_special_inventory(
            repository, traversal_config
        )
    with pytest.raises(PermissionError, match="frozen snapshot layout"):
        parallel_module._worker_boundary_special_inventory(
            tmp_path / "unconfigured/frozen_source_snapshot", frozen_config
        )
    forged_target = tmp_path / "forged-lock.json"
    forged_target.write_bytes(lock.read_bytes())
    lock.unlink()
    lock.symlink_to(forged_target)
    with pytest.raises(PermissionError, match="stable regular file"):
        verify_worker_boundary_commitment(
            boundary,
            repository_root=repository,
            config=bound_config,
            require_loaded_module_origins=False,
        )

    observed_counts = []
    real_sha256 = parallel_module.sha256_file
    for unrelated_count in (4, 100):
        input_root = tmp_path / f"inputs-{unrelated_count}"
        corpus = input_root / "corpus-1"
        corpus.mkdir(parents=True)
        write_json(corpus / "learner_input.json", {"training_episodes": []})
        for index in range(unrelated_count):
            write_json(input_root / f"unrelated-{index}.json", {"index": index})
        manifest = manifest_for_tree(input_root)
        write_json(input_root / "input_manifest.json", manifest)
        expected_sha = real_sha256(input_root / "input_manifest.json")
        calls = []

        def counted(path):
            calls.append(Path(path))
            return real_sha256(path)

        monkeypatch.setattr(parallel_module, "sha256_file", counted)
        _verify_worker_persisted_input_commitment(
            input_root,
            expected_manifest_sha256=expected_sha,
            required_relative="corpus-1/learner_input.json",
        )
        observed_counts.append(len(calls))
        monkeypatch.setattr(parallel_module, "sha256_file", real_sha256)
    assert observed_counts == [1, 1]
    worker_source = inspect.getsource(parallel_module._worker)
    for forbidden_call in (
        "verify_persisted_inputs(",
        "_verify_frozen_execution_boundary(",
        "build_work_plan(",
        "environment_record(",
        "verify_snapshot(",
    ):
        assert forbidden_call not in worker_source


def test_historical_worker_boundary_allows_only_exact_status_transition(
    tmp_path, config
):
    repository = tmp_path / "repository"
    relatives = sorted(
        {
            *parallel_module._WORKER_OPERATIONAL_RELATIVES,
            str(config["registries"]["canonical_prior_registry"]),
        }
    )
    for relative in relatives:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    prefreeze = load_config(
        repository / "configs/synthetic_corrective_alignment_v3.yaml",
        repository_root=repository,
    )
    prefreeze_projection = copy.deepcopy(prefreeze)
    prefreeze_projection.pop("repository_root", None)
    anticipated_projection = copy.deepcopy(prefreeze_projection)
    anticipated_projection["protocol"]["status"] = "frozen"
    config_path = repository / "configs/synthetic_corrective_alignment_v3.yaml"
    anticipated_bytes = runner._anticipated_frozen_config_bytes(config_path)
    lock = (
        repository
        / str(prefreeze["paths"]["package_root"])
        / "prequalification_design_lock.json"
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        lock,
        {
            "prefreeze_config": prefreeze_projection,
            "prefreeze_config_digest": canonical_digest(prefreeze_projection),
            "prefreeze_config_sha256": sha256_file(config_path),
            "anticipated_frozen_config_digest": canonical_digest(
                anticipated_projection
            ),
            "anticipated_frozen_config_sha256": hashlib.sha256(
                anticipated_bytes
            ).hexdigest(),
            "allowed_postqualification_delta": {
                "path": "protocol.status",
                "before": "pre_freeze",
                "after": "frozen",
            },
        },
    )
    boundary = create_worker_boundary_commitment(
        repository_root=repository,
        config=prefreeze,
        prequalification_lock=lock,
    )
    config_path.write_bytes(anticipated_bytes)
    frozen = load_config(config_path, repository_root=repository)
    assert verify_worker_boundary_commitment(
        boundary,
        repository_root=repository,
        config=frozen,
        allow_historical_prefreeze_transition=True,
        require_loaded_module_origins=False,
    )["status"] == "PASS"
    config_path.write_bytes(anticipated_bytes + b"\n")
    changed = load_config(config_path, repository_root=repository)
    with pytest.raises(PermissionError, match="operational-boundary"):
        verify_worker_boundary_commitment(
            boundary,
            repository_root=repository,
            config=changed,
            allow_historical_prefreeze_transition=True,
            require_loaded_module_origins=False,
        )


def test_rehearsal_validator_binds_worker_boundary_and_suppressed_candidate():
    source = inspect.getsource(runner._rehearsal_contract_validation)
    tree = ast.parse(source)
    expected_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "expected_execution_fields"
            for target in node.targets
        )
    ]
    assert len(expected_assignments) == 1
    fields = {
        child.value
        for child in expected_assignments[0].value.elts
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }
    assert "worker_boundary_digest" in fields
    assert "python_launcher_evidence" in fields
    runtime_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "expected_runtime_fields"
            for target in node.targets
        )
    ]
    assert len(runtime_assignments) == 1
    runtime_fields = {
        child.value
        for child in runtime_assignments[0].value.elts
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }
    assert "python_launcher_evidence" in runtime_fields
    assert source.count('get("python_launcher_evidence")') == 2
    assert source.count('summary.get("candidate_decision")') == 1
    assert source.count('== "SCIENTIFIC_INFERENCE_SUPPRESSED"') >= 2


def test_child_environment_is_allowlisted_and_forbidden_overrides_fail(
    monkeypatch,
    config,
):
    for name, value in config["parallel"]["thread_environment"].items():
        monkeypatch.setenv(str(name), str(value))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("UNRELATED_PARENT_SECRET", "must-not-propagate")
    child = _child_environment_base(config)
    assert "UNRELATED_PARENT_SECRET" not in child
    assert set(child) <= (
        set(config["parallel"]["child_inherited_environment_allowlist"])
        | set(config["parallel"]["thread_environment"])
        | {"PYTHONDONTWRITEBYTECODE"}
    )


def test_parent_and_worker_python_startup_flags_are_exactly_bound(
    monkeypatch, config
):
    assert (
        parallel_module.python_startup_flags_projection()
        == config["environment"]["parent_python_startup_flags"]
    )
    changed = dict(config["environment"]["parent_python_startup_flags"])
    changed["hash_randomization"] = 1
    monkeypatch.setattr(
        integrity_module,
        "python_startup_flags_projection",
        lambda: changed,
    )
    with pytest.raises(RuntimeError, match="python_startup_flags"):
        integrity_module.environment_record(ROOT, config)
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    with pytest.raises(RuntimeError, match="PYTHONWARNINGS"):
        parallel_module._require_thread_environment(config)


def test_scheduler_signal_handler_is_one_shot_and_restores_handlers(
    monkeypatch,
    config,
):
    prior_term = signal.getsignal(signal.SIGTERM)
    prior_hup = signal.getsignal(signal.SIGHUP)

    def interrupt_inside(**_kwargs):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        try:
            handler(signal.SIGTERM, None)
        except InterruptedError:
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
            raise

    monkeypatch.setattr(
        parallel_module,
        "_execute_worker_batches_impl",
        interrupt_inside,
    )
    with pytest.raises(InterruptedError, match="received signal"):
        _execute_worker_batches(
            plan={},
            jobs=1,
            scheduler={},
            repository_root=ROOT,
            worker_entrypoint=ROOT
            / "scripts/run_synthetic_corrective_alignment_v3_worker.py",
            python_executable=sys.executable,
            child_environment_base=_child_environment_base(config),
        )
    assert signal.getsignal(signal.SIGTERM) == prior_term
    assert signal.getsignal(signal.SIGHUP) == prior_hup


def test_worker_nonzero_exit_terminates_siblings_and_returns_closed_failure(
    tmp_path,
    monkeypatch,
    config,
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    units = [
        {"unit_id": "corpus-1__model-1", "model_seed": 1},
        {"unit_id": "corpus-2__model-1", "model_seed": 1},
    ]
    plan = {"model_seed_order": [1], "units": units}
    scheduler = _prepare_subprocess_requests(
        plan=plan,
        arguments_by_id={
            unit["unit_id"]: {"unit": {"unit_id": unit["unit_id"]}}
            for unit in units
        },
        runtime_root=runtime,
        child_environment_base=_child_environment_base(config),
    )
    processes = {}

    class FakeProcess:
        next_pid = 9100

        def __init__(self, argv, **_kwargs):
            request = Path(argv[argv.index("--request") + 1])
            self.unit_id = request.stem
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            self.returncode = 1 if self.unit_id == "corpus-1__model-1" else None
            processes[self.pid] = self
            if self.returncode == 1:
                request_value = read_json(request)
                write_json(
                    request_value["failure_receipt_path"],
                    {
                        "schema_version": "nursery-corrective-subprocess-failure-v3",
                        "status": "FAILED_NO_OUTCOME",
                        "unit_id": self.unit_id,
                        "request_sha256": sha256_file(request),
                        "request_manifest_sha256": argv[
                            argv.index("--request-manifest-sha256") + 1
                        ],
                        "exception": "RuntimeError",
                        "message": "injected worker failure",
                        "scientific_outcome": False,
                    },
                )

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            assert timeout is not None
            return self.returncode

    def fake_killpg(pid, signal_value):
        processes[pid].returncode = -int(signal_value)

    monkeypatch.setattr(parallel_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(parallel_module.os, "killpg", fake_killpg)
    telemetry, failures = _execute_worker_batches(
        plan=plan,
        jobs=2,
        scheduler=scheduler,
        repository_root=ROOT,
        worker_entrypoint=ROOT
        / "scripts/run_synthetic_corrective_alignment_v3_worker.py",
        python_executable=sys.executable,
        child_environment_base=_child_environment_base(config),
    )
    assert telemetry == []
    assert failures[0]["unit_id"] == "corpus-1__model-1"
    assert failures[0]["exception"] == "SubprocessExit"
    assert processes[9101].returncode == -15
    assert (scheduler["root"] / "FAILED_NO_OUTCOME.json").is_file()


def test_direct_subprocess_backend_has_no_multiprocessing_primitives(config):
    source = Path(parallel_module.__file__).read_text()
    assert "concurrent.futures" not in source
    assert "ProcessPoolExecutor" not in source
    assert "multiprocessing" not in source
    assert "SC_SEM_NSEMS_MAX" not in source
    assert config["parallel"]["backend"] == "direct_subprocess_v1"
    assert config["parallel"]["wave_key"] == "model_seed"
    assert config["parallel"]["subprocess_new_session"] is True
    worker_source = (
        ROOT / "scripts/run_synthetic_corrective_alignment_v3_worker.py"
    ).read_text()
    assert "sys.path.remove(str(SOURCE_ROOT))" in worker_source
    assert "sys.path.insert(0, str(SOURCE_ROOT))" in worker_source
    assert "imported parallel code outside SOURCE_ROOT" in worker_source


def test_real_construction_work_plan_unit_ids_are_accepted_by_request_preparation(
    tmp_path,
    config,
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    plan = build_work_plan(config, purpose="construction_micro")
    scheduler = _prepare_subprocess_requests(
        plan=plan,
        arguments_by_id={
            unit["unit_id"]: {"unit": unit}
            for unit in plan["units"]
        },
        runtime_root=runtime,
        child_environment_base=_child_environment_base(config),
    )
    assert len(scheduler["request_paths"]) == plan["unit_count"] == 8
    assert verify_exact_manifest(
        scheduler["requests"],
        read_json(scheduler["requests"] / "manifest.json"),
        manifest_filename="manifest.json",
    )["status"] == "PASS"


def test_predecessor_v1_tree_is_fully_rehashed_and_failure_is_detected(monkeypatch):
    files, directories, audit = runner._predecessor_v1_evidence(EVIDENCE_ROOT)
    assert files["file_count"] == 174
    assert files["digest"] == (
        "cb4c9596f8ed5cd9bc64311162c3abb09d3f6765a55a3a26915c0f1e630414e4"
    )
    assert directories["directory_count"] == 51
    assert directories["symlink_count"] == 0
    assert audit["status"] == "PASS_PRESERVED_FAILED_NO_OUTCOME"
    assert audit["development_outcome_count"] == 0
    assert audit["confirmation_outcome_count"] == 0
    assert audit["failure"]["independent_traceback_transcript_persisted"] is False
    real_sha256 = runner.sha256_file

    def changed_failure_receipt(path):
        if Path(path).name == "OFFICIAL_MICRO_ATTEMPT_4_FAILED.json":
            return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(runner, "sha256_file", changed_failure_receipt)
    with pytest.raises(RuntimeError, match="failure receipt"):
        runner._predecessor_v1_evidence(EVIDENCE_ROOT)


def test_real_subprocess_bootstrap_failure_is_semaphore_free_and_closed(
    tmp_path,
    config,
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    unit_id = "corpus-1__model-1"
    plan = {"model_seed_order": [1], "units": [{"unit_id": unit_id, "model_seed": 1}]}
    scheduler = _prepare_subprocess_requests(
        plan=plan,
        arguments_by_id={
            unit_id: {
                "unit": {"unit_id": unit_id},
                "purpose": "invalid_non_scientific_smoke",
                "repository_root": str(ROOT),
                "config_path": str(CONFIG),
                "input_root": str(tmp_path / "missing-input"),
                "shard_base": str(tmp_path / "never-created-shards"),
                "thread_environment": config["parallel"]["thread_environment"],
                "worker_python_startup_flags": dict(
                    config["environment"]["worker_python_startup_flags"]
                ),
                "expected_launcher_evidence": (
                    parallel_module._python_launcher_evidence(sys.executable)
                ),
            }
        },
        runtime_root=runtime,
        child_environment_base=_child_environment_base(config),
    )
    request_path = scheduler["request_paths"][unit_id]
    environment = _child_environment_base(config)
    environment["TMPDIR"] = str(scheduler["bootstrap_tmp"] / unit_id)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            "-P",
            str(ROOT / "scripts/run_synthetic_corrective_alignment_v3_worker.py"),
            "--request",
            str(request_path),
            "--request-sha256",
            sha256_file(request_path),
            "--request-manifest-sha256",
            scheduler["request_manifest_sha256"],
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    assert result.returncode == 1
    failure = read_json(scheduler["failures"] / f"{unit_id}.json")
    assert failure["status"] == "FAILED_NO_OUTCOME"
    assert failure["scientific_outcome"] is False
    assert "SC_SEM_NSEMS_MAX" not in failure["message"]
    assert "bootstrap environment" not in failure["message"]
    assert "Python startup flags" not in failure["message"]
    assert "worker_boundary_commitment" in failure["message"]
    assert not (tmp_path / "never-created-shards").exists()


def test_subprocess_scheduler_bounds_concurrency_and_enforces_wave_barriers(
    tmp_path,
    monkeypatch,
    config,
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    units = [
        {"unit_id": "corpus-1__model-1", "model_seed": 1},
        {"unit_id": "corpus-1__model-2", "model_seed": 2},
        {"unit_id": "corpus-2__model-1", "model_seed": 1},
        {"unit_id": "corpus-2__model-2", "model_seed": 2},
    ]
    plan = {"model_seed_order": [1, 2], "units": units}
    scheduler = _prepare_subprocess_requests(
        plan=plan,
        arguments_by_id={
            unit["unit_id"]: {"unit": {"unit_id": unit["unit_id"]}}
            for unit in units
        },
        runtime_root=runtime,
        child_environment_base=_child_environment_base(config),
    )
    launches = []

    class SuccessfulProcess:
        next_pid = 9200

        def __init__(self, argv, **_kwargs):
            request_path = Path(argv[argv.index("--request") + 1])
            request = read_json(request_path)
            self.unit_id = request["unit_id"]
            self.pid = SuccessfulProcess.next_pid
            SuccessfulProcess.next_pid += 1
            self.polls_remaining = 2 if self.unit_id.startswith("corpus-1") else 1
            launches.append(self.unit_id)
            telemetry = {
                "unit_id": self.unit_id,
                "shard_manifest_sha256": "a" * 64,
                "wall_seconds": 0.01,
                "worker_peak_rss_native_units": 1,
                "bootstrap_environment_digest": canonical_digest(
                    request["arguments"]["expected_bootstrap_environment"]
                ),
                "thread_environment_observed": {
                    str(name): str(value)
                    for name, value in config["parallel"]["thread_environment"].items()
                },
                "native_threadpools": [{"internal_api": "test", "prefix": "test", "num_threads": 1}],
                "python_startup_flags_observed": dict(
                    config["environment"]["worker_python_startup_flags"]
                ),
            }
            write_json(
                request["success_receipt_path"],
                {
                    "schema_version": "nursery-corrective-subprocess-success-v3",
                    "status": "PASS",
                    "unit_id": self.unit_id,
                    "request_sha256": sha256_file(request_path),
                    "request_manifest_sha256": scheduler[
                        "request_manifest_sha256"
                    ],
                    "telemetry": telemetry,
                    "scientific_outcome": False,
                },
            )

        def poll(self):
            self.polls_remaining -= 1
            return 0 if self.polls_remaining <= 0 else None

    monkeypatch.setattr(parallel_module.subprocess, "Popen", SuccessfulProcess)
    telemetry, failures = _execute_worker_batches(
        plan=plan,
        jobs=2,
        scheduler=scheduler,
        repository_root=ROOT,
        worker_entrypoint=ROOT
        / "scripts/run_synthetic_corrective_alignment_v3_worker.py",
        python_executable=sys.executable,
        child_environment_base=_child_environment_base(config),
    )
    assert failures == []
    assert [row["unit_id"] for row in telemetry] == [unit["unit_id"] for unit in units]
    assert launches == [
        "corpus-1__model-1",
        "corpus-2__model-1",
        "corpus-1__model-2",
        "corpus-2__model-2",
    ]
    summary = read_json(scheduler["root"] / "scheduler_summary.json")
    assert summary["maximum_observed_concurrency"] == 2
    assert [row["model_seed"] for row in summary["waves"]] == [1, 2]
    assert all(row["complete_before_next_wave"] for row in summary["waves"])


def test_subprocess_worker_rejects_resealed_alternate_receipt_path(
    tmp_path,
    monkeypatch,
    config,
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    unit_id = "corpus-1__model-1"
    scheduler = _prepare_subprocess_requests(
        plan={"model_seed_order": [1], "units": [{"unit_id": unit_id, "model_seed": 1}]},
        arguments_by_id={unit_id: {"unit": {"unit_id": unit_id}}},
        runtime_root=runtime,
        child_environment_base=_child_environment_base(config),
    )
    request_path = scheduler["request_paths"][unit_id]
    request = read_json(request_path)
    request["success_receipt_path"] = str(
        scheduler["receipts"] / "alternate.json"
    )
    request_path.write_bytes(parallel_module.canonical_bytes(request))
    manifest = manifest_for_paths(scheduler["requests"], [request_path.name])
    (scheduler["requests"] / "manifest.json").write_bytes(
        parallel_module.canonical_bytes(manifest)
    )
    monkeypatch.setenv("TMPDIR", str(scheduler["bootstrap_tmp"] / unit_id))
    assert parallel_module._subprocess_worker_main(
        request_path,
        sha256_file(request_path),
        sha256_file(scheduler["requests"] / "manifest.json"),
    ) == 1
    assert not (scheduler["receipts"] / "alternate.json").exists()
    assert read_json(scheduler["failures"] / f"{unit_id}.json")["status"] == (
        "FAILED_NO_OUTCOME"
    )


def test_runner_has_no_generic_scientific_recompute_or_confirmation_bypass():
    source = Path(runner.__file__).read_text()
    assert '"run-confirmation"' not in source
    assert '"recompute"' not in source.split("choices=(", 1)[1].split(")", 1)[0]
    assert "recompute-rehearsal" in source


def test_parallel_requires_preparation_manifest_commitment_and_leaves_no_output(
    tmp_path, monkeypatch, config
):
    parameter = inspect.signature(run_parallel_from_persisted).parameters[
        "expected_input_manifest_sha256"
    ]
    assert parameter.default is inspect.Parameter.empty
    for name, value in config["parallel"]["thread_environment"].items():
        monkeypatch.setenv(str(name), str(value))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    inputs = tmp_path / "inputs"
    prepared = prepare_persisted_inputs(
        inputs,
        config,
        purpose="construction_micro",
    )
    output = tmp_path / "compute"
    with pytest.raises(RuntimeError, match="changed after preparation"):
        run_parallel_from_persisted(
            repository_root=ROOT,
            config_path=CONFIG,
            input_root=inputs,
            output_root=output,
            config=config,
            purpose="construction_micro",
            jobs=1,
            expected_input_manifest_sha256=(
                "0" * 64
                if prepared["input_manifest_sha256"] != "0" * 64
                else "1" * 64
            ),
        )
    with pytest.raises(FileNotFoundError):
        output.lstat()


def test_excluded_rehearsal_deep_apis_require_receipt_capability(config):
    with pytest.raises(PermissionError, match="requires verified capability"):
        IdentifierFirewall(config, purpose="excluded_rehearsal")
    with pytest.raises(PermissionError, match="receipt envelope is missing"):
        parallel_module._worker(
            {
                "purpose": "excluded_rehearsal",
                "thread_environment": {},
                "scheduler_parent_pid": os.getppid(),
                "expected_bootstrap_environment": dict(os.environ),
                "worker_python_startup_flags": (
                    parallel_module.python_startup_flags_projection()
                ),
            }
        )


def test_official_test_contract_has_exact_one_historical_deselection():
    expected = (
        "tests/test_synthetic_development_launch_v3.py::"
        "test_exact_one_shot_command_and_output_are_frozen"
    )
    assert CLOSED_PROTOCOL_DESELECTION == expected
    source = inspect.getsource(integrity_module.run_tests)
    assert source.count('"--deselect"') == 1
    assert source.count("CLOSED_PROTOCOL_DESELECTION") >= 2


def test_official_test_report_validator_rejects_mutable_groups_and_bindings(
    tmp_path,
):
    snapshot = tmp_path / "snapshot"
    full_root = tmp_path / "full"
    package = tmp_path / "package"
    (snapshot / "configs").mkdir(parents=True)
    (snapshot / "scripts").mkdir(parents=True)
    (full_root / "tests").mkdir(parents=True)
    (full_root / "output/synthetic_development_v3_one_shot").mkdir(
        parents=True
    )
    package.mkdir()
    write_json(snapshot / "snapshot_manifest.json", {"status": "FROZEN"})
    (snapshot / "configs/synthetic_corrective_alignment_v3.yaml").write_text(
        "protocol: frozen\n", encoding="utf-8"
    )
    (snapshot / "scripts/run_synthetic_corrective_alignment_v3.py").write_text(
        "# frozen runner\n", encoding="utf-8"
    )
    deselected_source = full_root / "tests/test_synthetic_development_launch_v3.py"
    deselected_source.write_text("# historical test\n", encoding="utf-8")
    write_json(package / "freeze_receipt.json", {"status": "FROZEN"})
    write_json(package / "frozen_environment.json", {"status": "FROZEN"})
    python = Path(sys.executable).resolve()
    required = integrity_module.REQUIRED_CORRECTIVE_TEST_GROUPS
    names = sorted({name for values in required.values() for name in values})
    nodeids = [
        f"tests/test_synthetic_corrective_alignment_v3.py::{name}"
        for name in names
    ]
    group_rows = {
        group: {
            "required_tests": tests,
            "passed_tests": sorted(tests),
            "missing_or_not_passed": [],
            "status": "PASS",
        }
        for group, tests in required.items()
    }
    commands = [
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "-rA",
            "-p",
            "no:cacheprovider",
            "tests/test_synthetic_corrective_alignment_v3.py",
        ],
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "-rA",
            "-p",
            "no:cacheprovider",
            "--deselect",
            CLOSED_PROTOCOL_DESELECTION,
            "tests",
        ],
    ]
    bindings = {
        "snapshot_root": str(snapshot.resolve()),
        "full_repository_root": str(full_root.resolve()),
        "snapshot_manifest_sha256": sha256_file(
            snapshot / "snapshot_manifest.json"
        ),
        "config_sha256": sha256_file(
            snapshot / "configs/synthetic_corrective_alignment_v3.yaml"
        ),
        "runner_sha256": sha256_file(
            snapshot / "scripts/run_synthetic_corrective_alignment_v3.py"
        ),
        "freeze_receipt_sha256": sha256_file(package / "freeze_receipt.json"),
        "frozen_environment_sha256": sha256_file(
            package / "frozen_environment.json"
        ),
        "python_executable": str(python),
        "python_sha256": sha256_file(python),
    }
    report = {
        "schema_version": "nursery-corrective-official-tests-v3",
        "status": "PASS",
        "executions": [
            {
                "command": commands[0],
                "cwd": str(snapshot.resolve()),
                "exit_code": 0,
                "output": "".join(f"PASSED {nodeid}\n" for nodeid in nodeids),
            },
            {
                "command": commands[1],
                "cwd": str(full_root.resolve()),
                "exit_code": 0,
                "output": "1 deselected\n",
            },
        ],
        "passed_nodeids": sorted(nodeids),
        "passed_test_count": len(nodeids),
        "required_test_groups": group_rows,
        "required_inventory_status": "PASS",
        "closed_protocol_deselections": [
            {
                "nodeid": CLOSED_PROTOCOL_DESELECTION,
                "reason": integrity_module.CLOSED_PROTOCOL_DESELECTION_RATIONALE,
                "source_sha256": sha256_file(deselected_source),
                "authoritative_old_output_exists": True,
            }
        ],
        "observed_deselected_count": 1,
        "deselection_contract_status": "PASS",
        "unclassified_passed_tests": [],
        "evidence_bindings": bindings,
    }
    kwargs = {
        "snapshot_root": snapshot,
        "full_repository_root": full_root,
        "executable": python,
        "package_root": package,
    }
    assert integrity_module.validate_official_test_report(report, **kwargs)[
        "status"
    ] == "PASS"
    changed = copy.deepcopy(report)
    changed["required_test_groups"].pop(next(iter(required)))
    assert integrity_module.validate_official_test_report(changed, **kwargs)[
        "status"
    ] == "FAIL"
    changed = copy.deepcopy(report)
    changed["evidence_bindings"]["config_sha256"] = "0" * 64
    assert integrity_module.validate_official_test_report(changed, **kwargs)[
        "status"
    ] == "FAIL"
    changed = copy.deepcopy(report)
    changed["executions"][0]["exit_code"] = 1
    assert integrity_module.validate_official_test_report(changed, **kwargs)[
        "status"
    ] == "FAIL"


def test_prior_preservation_reverification_detects_postproof_mutation(
    tmp_path, config
):
    root = tmp_path / "repository"
    root.mkdir()
    protected = root / "protected.bin"
    protected.write_bytes(b"abc")
    audit = root / "output/synthetic_confirmation_adversarial_audit_v1"
    preservation_source = audit / "preservation"
    preservation_source.mkdir(parents=True)
    baseline_hash = preservation_source / "baseline.sha256"
    baseline_stat = preservation_source / "baseline.stat"
    baseline_symlinks = preservation_source / "baseline.symlinks"
    baseline_hash.write_text(
        f"{sha256_file(protected)}  ./protected.bin\n", encoding="utf-8"
    )
    baseline_stat.write_text(
        f"{protected.stat().st_size} {format(protected.stat().st_mode & 0o777, 'o')} ./protected.bin\n",
        encoding="utf-8",
    )
    baseline_symlinks.write_bytes(b"\n")
    write_json(audit / "authoritative.json", {"status": "CLOSED"})
    audit_manifest = manifest_for_tree(audit)
    write_json(audit / "complete_file_manifest.json", audit_manifest)
    fixture_config = copy.deepcopy(config)
    fixture_config["paths"].update(
        {
            "preservation_baseline": baseline_hash.relative_to(root).as_posix(),
            "preservation_baseline_sha256": sha256_file(baseline_hash),
            "preservation_stat": baseline_stat.relative_to(root).as_posix(),
            "preservation_stat_sha256": sha256_file(baseline_stat),
            "preservation_symlinks": baseline_symlinks.relative_to(root).as_posix(),
            "preservation_symlinks_sha256": sha256_file(baseline_symlinks),
            "preservation_audit_manifest_sha256": sha256_file(
                audit / "complete_file_manifest.json"
            ),
        }
    )
    output = root / "proof"
    proof = integrity_module.verify_prior_preservation(
        root, output, fixture_config, jobs=1
    )
    assert proof["status"] == "PASS"
    assert integrity_module.reverify_prior_preservation(
        root, output, fixture_config, jobs=1
    )["status"] == "PASS"
    with pytest.raises(FileExistsError, match="pristine"):
        integrity_module.verify_prior_preservation(
            root, output, fixture_config, jobs=1
        )
    protected.write_bytes(b"xyz")
    assert integrity_module.reverify_prior_preservation(
        root, output, fixture_config, jobs=1
    )["status"] == "FAIL"


def test_frozen_context_rejects_environment_drift(
    tmp_path, monkeypatch, config
):
    repository = tmp_path / "repository"
    snapshot = repository / "output/package/frozen_source_snapshot"
    runner_path = snapshot / "scripts/run_synthetic_corrective_alignment_v3.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# frozen runner\n", encoding="utf-8")
    frozen_config = copy.deepcopy(config)
    frozen_config["protocol"]["status"] = "frozen"
    frozen_record = {"schema_version": "test-environment-v1", "digest": "locked"}
    write_json(snapshot.parent / "frozen_environment.json", frozen_record)
    for name, value in frozen_config["parallel"]["thread_environment"].items():
        monkeypatch.setenv(str(name), str(value))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(runner, "__file__", str(runner_path))
    monkeypatch.setattr(runner, "_require_frozen_module_origins", lambda *_: None)
    monkeypatch.setattr(runner, "verify_snapshot", lambda *_: {"status": "PASS"})
    monkeypatch.setattr(runner, "_repository_from_snapshot", lambda *_: repository)
    monkeypatch.setattr(runner, "load_config", lambda *args, **kwargs: frozen_config)
    monkeypatch.setattr(runner, "require_frozen", lambda *_: None)
    monkeypatch.setattr(
        runner, "_require_frozen_design_transition", lambda *args: {"status": "PASS"}
    )
    monkeypatch.setattr(runner, "environment_record", lambda *args: frozen_record)
    monkeypatch.setattr(
        runner,
        "_verify_or_create_predecessor_v1_evidence",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "_verify_or_create_predecessor_v2_evidence",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    assert runner._frozen_context(str(snapshot))[0] == repository
    monkeypatch.setattr(
        runner,
        "environment_record",
        lambda *args: {"schema_version": "test-environment-v1", "digest": "drift"},
    )
    with pytest.raises(RuntimeError, match="runtime differs"):
        runner._frozen_context(str(snapshot))
    monkeypatch.setattr(runner, "environment_record", lambda *args: frozen_record)
    first_name = next(iter(frozen_config["parallel"]["thread_environment"]))
    monkeypatch.setenv(first_name, "forged")
    with pytest.raises(RuntimeError, match="thread environment mismatch"):
        runner._frozen_context(str(snapshot))


def test_generic_rehearsal_apis_cannot_bypass_one_shot_capability(tmp_path, config):
    seal = parallel_module._REHEARSAL_INTERNAL_SEAL
    with pytest.raises(PermissionError, match="consumed attempt capability"):
        prepare_persisted_inputs(
            tmp_path / "inputs",
            config,
            purpose="excluded_rehearsal",
            _rehearsal_seal=seal,
        )
    with pytest.raises(PermissionError, match="consumed attempt capability"):
        run_parallel_from_persisted(
            repository_root=ROOT,
            config_path=CONFIG,
            input_root=tmp_path / "missing-inputs",
            output_root=tmp_path / "parallel",
            config=config,
            purpose="excluded_rehearsal",
            jobs=int(config["parallel"]["frozen_jobs"]),
            expected_input_manifest_sha256="0" * 64,
            _rehearsal_seal=seal,
        )
    with pytest.raises(PermissionError, match="consumed attempt capability"):
        execute_cohort(
            repository_root=ROOT,
            config_path=CONFIG,
            output_root=tmp_path / "cohort",
            config=config,
            purpose="excluded_rehearsal",
            jobs=int(config["parallel"]["frozen_jobs"]),
            _rehearsal_seal=seal,
        )


def test_direct_live_rehearsal_execution_cannot_consume_frozen_attempt(
    tmp_path, monkeypatch, config
):
    fixture = _make_preflight_package(tmp_path, monkeypatch, config)
    frozen = load_config(
        fixture["snapshot"] / "configs/synthetic_corrective_alignment_v3.yaml",
        repository_root=fixture["snapshot"],
    )
    kwargs = {
        "repository_root": fixture["snapshot"],
        "config_path": fixture["snapshot"]
        / "configs/synthetic_corrective_alignment_v3.yaml",
        "output_root": fixture["package"] / "excluded_rehearsal",
        "config": frozen,
        "jobs": int(frozen["parallel"]["frozen_jobs"]),
    }
    receipt = fixture["package"] / "excluded_rehearsal_attempt_consumed.json"
    with pytest.raises(PermissionError, match="origin mismatch"):
        parallel_module._execute_excluded_rehearsal_cohort(**kwargs)
    assert not receipt.exists()


def test_postfreeze_lifecycle_receipts_are_nonreplayable_and_exact(tmp_path):
    root = tmp_path / "repository"
    package = root / "output/synthetic_corrective_development_launch_package_v3"
    snapshot = package / "frozen_source_snapshot"
    for source in (root, snapshot):
        config_path = source / "configs/synthetic_corrective_alignment_v3.yaml"
        runner_path = source / "scripts/run_synthetic_corrective_alignment_v3.py"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        runner_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("protocol: frozen\n", encoding="utf-8")
        runner_path.write_text("# frozen runner\n", encoding="utf-8")
    write_json(package / "prequalification_design_lock.json", {"status": "LOCKED"})
    write_json(package / "micro_attempt_consumed.json", {"status": "CONSUMED"})
    write_json(package / "worker_launcher_preflight.json", {"status": "PASS"})
    write_json(
        package / "construction_qualification_attempt_consumed.json",
        {"status": "CONSUMED"},
    )
    fixture_config = {"parallel": {"frozen_jobs": 4}}
    runner._consume_lifecycle_attempt(package, root, fixture_config, "freeze")
    with pytest.raises(FileExistsError, match="pristine"):
        runner._consume_lifecycle_attempt(package, root, fixture_config, "freeze")
    write_json(snapshot / "snapshot_manifest.json", {"status": "FROZEN"})
    write_json(package / "freeze_receipt.json", {"status": "FROZEN"})
    prerequisites = (
        (
            "excluded_rehearsal_completion.json",
            "independent_recompute",
        ),
        (
            "independent_recompute_comparison.json",
            "official_tests",
        ),
        (
            "official_test_execution_report.json",
            "preservation",
        ),
            (
                "preservation_completion.json",
                "finalization",
            ),
    )
    for relative, operation in prerequisites:
        write_json(package / relative, {"status": "PASS"})
        runner._consume_lifecycle_attempt(
            package,
            snapshot,
            fixture_config,
            operation,
        )
        with pytest.raises(FileExistsError, match="pristine"):
            runner._consume_lifecycle_attempt(
                package,
                snapshot,
                fixture_config,
                operation,
            )
    assert runner._validate_lifecycle_attempt_receipts(
        root,
        package,
        snapshot,
        fixture_config,
    )["status"] == "PASS"
    receipt = package / "official_tests_attempt_consumed.json"
    changed = read_json(receipt)
    changed["jobs"] = 3
    write_json(receipt, changed, overwrite=True)
    assert runner._validate_lifecycle_attempt_receipts(
        root,
        package,
        snapshot,
        fixture_config,
    )["status"] == "FAIL"


def test_parallel_artifacts_are_byte_identical_when_available():
    package = EVIDENCE_ROOT / "output/synthetic_corrective_development_launch_package_v3"
    report = package / "parallel_equivalence_attempt_1.json"
    if not report.is_file():
        pytest.skip("parallel equivalence artifact is created later in the package sequence")
    value = read_json(report)
    assert value["status"] == "PASS"
    assert value["byte_identical"]


def test_v3_scientific_core_is_exactly_inherited_from_v2():
    expected_sources = {
        "generator.py": (
            "32e9f2e024c7e65414fb5bbd39f525d02fedde3a9a063de015db55ac77696fc7"
        ),
        "learner.py": (
            "cbabfcaf570edfe5ef96b68c4d07b69b17fdcbdb7c79a6ada711f4703514615f"
        ),
        "statistics.py": (
            "55a24f5ee80c814298c744c06780b9c901cb68b71329ff2efd32462e6789610f"
        ),
    }
    for name, expected in expected_sources.items():
        predecessor = (
            EVIDENCE_ROOT / "babyworld_lite/corrective_alignment_v2" / name
        )
        successor = ROOT / "babyworld_lite/corrective_alignment_v3" / name
        assert predecessor.read_bytes() == successor.read_bytes()
        assert sha256_file(successor) == expected

    v2 = yaml.safe_load(
        (EVIDENCE_ROOT / "configs/synthetic_corrective_alignment_v2.yaml").read_text()
    )
    v3 = yaml.safe_load(CONFIG.read_text())
    sections = ["design", "learner", "analysis", "qualification_gates"]
    projection_v2 = {name: v2[name] for name in sections}
    projection_v3 = {name: v3[name] for name in sections}
    assert projection_v3 == projection_v2
    assert canonical_digest(projection_v3) == (
        "4b91a07bc758a40c8aad5e5f72804bb80034e6208f2dfc3f204ee0c66ba7cf18"
    )
    claim_and_leakage_v2 = {
        "protocol": {
            name: v2["protocol"][name]
            for name in (
                "scientific_claim",
                "infant_learning_claim_authorized",
                "ecological_validity_claim_authorized",
                "prior_outcomes_authorized",
            )
        },
        "firewalls": v2["firewalls"],
    }
    claim_and_leakage_v3 = {
        "protocol": {
            name: v3["protocol"][name]
            for name in claim_and_leakage_v2["protocol"]
        },
        "firewalls": v3["firewalls"],
    }
    assert claim_and_leakage_v3 == claim_and_leakage_v2
    assert canonical_digest(claim_and_leakage_v3) == (
        "56c814c398fe98eac81849f117c72765997424f5618e971605218c3d9761c39c"
    )


def test_predecessor_v2_tree_is_fully_rehashed_and_failure_is_detected(
    monkeypatch,
):
    manifest, audit = runner._predecessor_v2_evidence(EVIDENCE_ROOT)
    assert manifest["digest"] == (
        "f55226f79a4fdcedca32116e1590cdd640ed2332df890f60a5eb4e3777f34d22"
    )
    assert manifest["file_count"] == 41
    assert manifest["directory_count_including_root"] == 24
    assert manifest["regular_file_bytes"] == 3_011_243
    assert manifest["symlink_count"] == 0
    assert audit["status"] == "PASS_PRESERVED_FAILED_NO_OUTCOME"
    assert audit["published_worker_shards"] == 0
    assert audit["development_outcome_count"] == 0
    assert audit["confirmation_outcome_count"] == 0

    real_read = runner._read_confined_stable_regular_bytes

    def mutate_one_file(path, boundary):
        data, metadata = real_read(path, boundary)
        if Path(path).name == "outcome_registry.json":
            data += b"\n"
        return data, metadata

    monkeypatch.setattr(
        runner,
        "_read_confined_stable_regular_bytes",
        mutate_one_file,
    )
    with pytest.raises(RuntimeError, match="retired v2 full tree changed"):
        runner._predecessor_v2_evidence(EVIDENCE_ROOT)


def test_real_launcher_preflight_uses_lexical_venv_and_resolved_base_fails(
    config,
):
    evidence = runner._worker_launcher_preflight(EVIDENCE_ROOT, config)
    assert evidence["status"] == "PASS"
    assert evidence["scientific_outcome"] is False
    assert evidence["seed_identifiers_used"] is False
    probe = evidence["probe"]
    lexical = Path(probe["python_executable"])
    resolved = Path(probe["python_executable_resolved"])
    assert lexical == Path(sys.executable).absolute()
    assert lexical != resolved
    assert Path(probe["sys_prefix"]) != Path(probe["sys_base_prefix"])
    assert "threadpoolctl" in probe["required_module_origins"]

    environment = _child_environment_base(config)
    completed = subprocess.run(
        [
            str(resolved),
            "-B",
            "-s",
            "-P",
            str(
                EVIDENCE_ROOT
                / "scripts/run_synthetic_corrective_alignment_v3_worker.py"
            ),
            "--bootstrap-probe",
        ],
        cwd=EVIDENCE_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "threadpoolctl" in completed.stderr


def test_micro_attempt_receipt_is_semantically_validated_and_nonreplayable(
    tmp_path,
    config,
):
    package = tmp_path / "package"
    package.mkdir()
    for name in (
        "PREDECESSOR_V1_FAILURE_AUDIT.json",
        "PREDECESSOR_V2_FAILURE_AUDIT.json",
        "PREDECESSOR_V2_TREE_MANIFEST.json",
    ):
        write_json(package / name, {"status": "PASS", "name": name})
    receipt = runner._consume_micro_attempt(package, EVIDENCE_ROOT, config)
    with pytest.raises(FileExistsError, match="pristine"):
        runner._consume_micro_attempt(package, EVIDENCE_ROOT, config)
    preflight = runner._worker_launcher_preflight(EVIDENCE_ROOT, config)
    write_json(package / "worker_launcher_preflight.json", preflight)
    environment = runner.environment_record(EVIDENCE_ROOT, config)
    lock = {
        "tracked_nonconfig_manifest": receipt["tracked_nonconfig_manifest"],
        "prefreeze_config_sha256": receipt["config_sha256"],
        "qualification_environment": environment,
        "preexisting_package_manifest": manifest_for_tree(package),
    }
    write_json(package / "prequalification_design_lock.json", lock)
    validation = runner._validate_micro_attempt_receipt(
        EVIDENCE_ROOT,
        package,
        config,
    )
    assert validation["status"] == "PASS"
    changed = read_json(package / "micro_attempt_consumed.json")
    changed["non_replayable"] = False
    write_json(
        package / "micro_attempt_consumed.json",
        changed,
        overwrite=True,
    )
    assert runner._validate_micro_attempt_receipt(
        EVIDENCE_ROOT,
        package,
        config,
    )["status"] == "FAIL"
    write_json(
        package / "micro_attempt_consumed.json",
        receipt,
        overwrite=True,
    )
    for field in ("argv", "cwd", "parallel_sha256", "probe"):
        forged_preflight = copy.deepcopy(preflight)
        if field == "argv":
            forged_preflight[field] = ["totally", "forged"]
        elif field == "cwd":
            forged_preflight[field] = "/forged"
        elif field == "parallel_sha256":
            forged_preflight[field] = "0" * 64
        else:
            forged_preflight[field] = {
                "status": "FORGED",
                "seed_identifiers_used": True,
            }
        write_json(
            package / "worker_launcher_preflight.json",
            forged_preflight,
            overwrite=True,
        )
        forged_lock = copy.deepcopy(lock)
        forged_lock["preexisting_package_manifest"] = manifest_for_tree(
            package,
            exclude=["prequalification_design_lock.json"],
        )
        write_json(
            package / "prequalification_design_lock.json",
            forged_lock,
            overwrite=True,
        )
        assert runner._validate_micro_attempt_receipt(
            EVIDENCE_ROOT,
            package,
            config,
        )["status"] == "FAIL"


def test_standard_library_bootstrap_writes_bound_import_failure_receipt(
    tmp_path,
    config,
):
    source = tmp_path / "isolated_source"
    worker = source / "scripts/run_synthetic_corrective_alignment_v3_worker.py"
    parallel = source / "babyworld_lite/corrective_alignment_v3/parallel.py"
    worker.parent.mkdir(parents=True)
    parallel.parent.mkdir(parents=True)
    shutil.copyfile(
        EVIDENCE_ROOT
        / "scripts/run_synthetic_corrective_alignment_v3_worker.py",
        worker,
    )
    (source / "babyworld_lite/__init__.py").write_text("", encoding="utf-8")
    (parallel.parent / "__init__.py").write_text("", encoding="utf-8")
    parallel.write_text(
        'raise ModuleNotFoundError("injected bootstrap dependency")\n',
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    unit_id = "corpus-1__model-1"
    scheduler = _prepare_subprocess_requests(
        plan={
            "model_seed_order": [1],
            "units": [{"unit_id": unit_id, "model_seed": 1}],
        },
        arguments_by_id={unit_id: {"unit": {"unit_id": unit_id}}},
        runtime_root=runtime,
        child_environment_base=_child_environment_base(config),
    )
    request_path = scheduler["request_paths"][unit_id]
    environment = _child_environment_base(config)
    environment["TMPDIR"] = str(scheduler["bootstrap_tmp"] / unit_id)
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            "-P",
            str(worker),
            "--request",
            str(request_path),
            "--request-sha256",
            sha256_file(request_path),
            "--request-manifest-sha256",
            scheduler["request_manifest_sha256"],
        ],
        cwd=source,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    failure_path = scheduler["failures"] / f"{unit_id}.json"
    failure = read_json(failure_path)
    assert failure == {
        "schema_version": "nursery-corrective-subprocess-failure-v3",
        "status": "FAILED_NO_OUTCOME",
        "unit_id": unit_id,
        "request_sha256": sha256_file(request_path),
        "request_manifest_sha256": scheduler["request_manifest_sha256"],
        "exception": "ModuleNotFoundError",
        "message": "injected bootstrap dependency",
        "scientific_outcome": False,
    }
    assert not (scheduler["receipts"] / f"{unit_id}.json").exists()


def test_subprocess_failure_receipt_is_exact_and_mutations_fail(tmp_path):
    request = tmp_path / "request.json"
    failure = tmp_path / "failure.json"
    success = tmp_path / "success.json"
    write_json(request, {"request": "bound"})
    manifest_sha256 = "a" * 64
    unit_id = "corpus-1__model-1"
    receipt = {
        "schema_version": "nursery-corrective-subprocess-failure-v3",
        "status": "FAILED_NO_OUTCOME",
        "unit_id": unit_id,
        "request_sha256": sha256_file(request),
        "request_manifest_sha256": manifest_sha256,
        "exception": "RuntimeError",
        "message": "injected",
        "scientific_outcome": False,
    }
    write_json(failure, receipt)
    assert _validate_subprocess_failure(
        unit_id=unit_id,
        request_path=request,
        failure_path=failure,
        success_path=success,
        request_manifest_sha256=manifest_sha256,
    ) == sha256_file(failure)
    mutations = [
        {"unexpected": True},
        {"exception": None},
        {"exception": 123},
        {"exception": {}},
        {"exception": ""},
        {"message": None},
        {"message": 123},
        {"message": {}},
        {"message": ""},
    ]
    for mutation in mutations:
        changed = {**receipt, **mutation}
        write_json(failure, changed, overwrite=True)
        with pytest.raises(RuntimeError, match="commitment mismatch"):
            _validate_subprocess_failure(
                unit_id=unit_id,
                request_path=request,
                failure_path=failure,
                success_path=success,
                request_manifest_sha256=manifest_sha256,
            )

    write_json(failure, receipt, overwrite=True)
    write_json(success, {"forged": "success"})
    with pytest.raises(FileExistsError, match="pristine"):
        _validate_subprocess_failure(
            unit_id=unit_id,
            request_path=request,
            failure_path=failure,
            success_path=success,
            request_manifest_sha256=manifest_sha256,
        )

    symlink_failure = tmp_path / "symlink-failure.json"
    symlink_failure.symlink_to(failure.name)
    with pytest.raises(RuntimeError, match="regular file"):
        _validate_subprocess_failure(
            unit_id=unit_id,
            request_path=request,
            failure_path=symlink_failure,
            success_path=tmp_path / "other-success.json",
            request_manifest_sha256=manifest_sha256,
        )


def test_publication_rechecks_preservation_immediately_before_atomic_rename(
    tmp_path,
    monkeypatch,
    config,
):
    staging, output, authorization, summary, capability, proof = (
        _make_authorized_development_staging(tmp_path, monkeypatch, config)
    )
    baseline = {
        "status": "PASS",
        "predecessor_v1_verification_digest": (
            proof.predecessor_v1_verification_digest
        ),
        "predecessor_v2_verification_digest": (
            proof.predecessor_v2_verification_digest
        ),
        "prior_preservation_verification_digest": (
            proof.prior_preservation_verification_digest
        ),
    }
    calls = {"count": 0}

    def preservation_changes_after_proof(_authorization):
        calls["count"] += 1
        if calls["count"] == 1:
            return copy.deepcopy(baseline)
        return {
            **baseline,
            "prior_preservation_verification_digest": "f" * 64,
        }

    monkeypatch.setattr(
        runner,
        "_live_development_publication_preservation",
        preservation_changes_after_proof,
    )
    with pytest.raises(PermissionError, match="preservation changed"):
        runner._publish_development_staging(
            staging=staging,
            output_root=output,
            authorization=authorization,
            summary=summary,
            capability=capability,
            publication_proof=proof,
        )
    assert calls["count"] == 2
    assert staging.is_dir()
    assert not output.exists()
