from __future__ import annotations

import ast
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap

import pytest
import yaml

import babyworld_lite.corrective_alignment_v1.parallel as parallel_module
import babyworld_lite.corrective_alignment_v1.integrity as integrity_module
from babyworld_lite.corrective_alignment_v1.integrity import (
    CLOSED_PROTOCOL_DESELECTION,
    _verify_anticipated_launch_manifest,
    freeze_prefreeze_evidence,
    operation_identifier_audit,
    verify_prefreeze_evidence,
)
from babyworld_lite.corrective_alignment_v1.adjudicator import (
    AUTHORIZATION_FIELDS,
    PACKAGE_GATES,
    authorization_payload,
    verify_authorization,
)
from babyworld_lite.corrective_alignment_v1.statistics import dependence_diagnostics
from babyworld_lite.corrective_alignment_v1.generator import generate_corpus
from babyworld_lite.corrective_alignment_v1.learner import (
    disagreement_mechanism_qualification,
    fit_corrective_mil,
    predict_without_side,
)
from babyworld_lite.corrective_alignment_v1.parallel import (
    _candidate_scientific_decision,
    _execute_development_cohort,
    _execute_worker_batches,
    _validate_shard,
    _validate_shard_inventory,
    build_work_plan,
    execute_cohort,
    prepare_persisted_inputs,
    run_parallel_from_persisted,
)
from babyworld_lite.corrective_alignment_v1.protocol import (
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
import scripts.run_synthetic_corrective_alignment_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/synthetic_corrective_alignment_v1.yaml"
EVIDENCE_ROOT = ROOT.parents[2] if ROOT.name == "frozen_source_snapshot" else ROOT


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG, repository_root=ROOT)


def _firewall(config, purpose="construction_micro"):
    return IdentifierFirewall(config, purpose=purpose)


def _make_prequalification_lock_fixture(tmp_path, monkeypatch, config):
    root = tmp_path / "repository"
    package = root / "output/synthetic_corrective_development_launch_package_v1"
    package.mkdir(parents=True)
    config_path = root / "configs/synthetic_corrective_alignment_v1.yaml"
    config_path.parent.mkdir(parents=True)
    config_bytes = CONFIG.read_bytes()
    frozen_status = b"  status: frozen\n"
    prefreeze_status = b"  status: pre_freeze\n"
    if config_bytes.count(frozen_status) == 1:
        config_bytes = config_bytes.replace(frozen_status, prefreeze_status, 1)
    elif config_bytes.count(prefreeze_status) != 1:
        raise AssertionError("fixture source config has no unique protocol status")
    config_path.write_bytes(config_bytes)
    runner_path = root / "scripts/run_synthetic_corrective_alignment_v1.py"
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("# locked runner\n", encoding="utf-8")
    rationale = root / "docs/locked-rationale.md"
    rationale.parent.mkdir(parents=True)
    rationale.write_text("locked rationale\n", encoding="utf-8")
    tracked = (
        "scripts/run_synthetic_corrective_alignment_v1.py",
        "configs/synthetic_corrective_alignment_v1.yaml",
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
            "schema_version": "nursery-corrective-outcome-registry-v1",
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
    runner_relative = "scripts/run_synthetic_corrective_alignment_v1.py"
    prior_relative = "output/synthetic_development_launch_package_v3/frozen_seed_registries.json"
    source_runner = source / runner_relative
    source_prior = source / prior_relative
    source_runner.parent.mkdir(parents=True)
    source_prior.parent.mkdir(parents=True)
    source_runner.write_bytes((ROOT / runner_relative).read_bytes())
    source_prior.write_bytes((ROOT / prior_relative).read_bytes())

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
        "schema_version": "nursery-corrective-prequalification-design-lock-v1",
        "status": "LOCKED_BEFORE_OFFICIAL_FIXTURES",
        "tracked_nonconfig_manifest": manifest_for_paths(
            source, [runner_relative, prior_relative]
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
    write_json(package / "locked_evidence.json", {"status": "PASS"})
    freeze_prefreeze_evidence(package)

    snapshot = package / "frozen_source_snapshot"
    snapshot_runner = snapshot / runner_relative
    snapshot_config = snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
    snapshot_prior = snapshot / prior_relative
    snapshot_runner.parent.mkdir(parents=True)
    snapshot_config.parent.mkdir(parents=True)
    snapshot_prior.parent.mkdir(parents=True)
    snapshot_runner.write_bytes(source_runner.read_bytes())
    snapshot_config.write_bytes(frozen_config_bytes)
    snapshot_prior.write_bytes(source_prior.read_bytes())
    manifest = manifest_for_tree(snapshot)
    write_json(snapshot / "snapshot_manifest.json", manifest)
    frozen_config = load_config(snapshot_config, repository_root=snapshot)
    write_json(package / "frozen_environment.json", environment)
    write_json(package / "frozen_seed_registries.json", registry_snapshot(frozen_config))
    prefreeze_contract = read_json(package / "prefreeze_evidence_contract.json")
    receipt = {
        "schema_version": "nursery-corrective-freeze-receipt-v1",
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
        "babyworld_lite.corrective_alignment_v1.protocol",
        "babyworld_lite.corrective_alignment_v1.generator",
        "babyworld_lite.corrective_alignment_v1.learner",
        "babyworld_lite.corrective_alignment_v1.statistics",
        "babyworld_lite.corrective_alignment_v1.parallel",
        "babyworld_lite.corrective_alignment_v1.adjudicator",
        "babyworld_lite.corrective_alignment_v1.integrity",
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
    forbidden = fixture["package"] / "parallel_micro_attempt_4_jobs_1"
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


def test_registry_allocation_is_fresh_disjoint_and_confirmation_sealed(config):
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
    assert all(320000 <= value <= 329999 for values in sets.values() for value in values)
    assert not any(
        281000 <= value <= 281999 or 289000 <= value <= 289999
        for values in sets.values()
        for value in values
    )
    assert not config["firewalls"]["confirmation_command_exists"]
    assert not config["firewalls"]["confirmation_execution_supported"]


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
    "purpose,primary,informativeness,dependence,causal,expected",
    [
        ("excluded_rehearsal", "PASS", "PASS", "PASS", "PASS", "SCIENTIFIC_INFERENCE_SUPPRESSED"),
        ("development", "FAIL", "PASS", "PASS", "FAIL", "CORRECTIVE_STUDY_STOP_NO_SUPPORT"),
        ("development", "PASS", "FAIL", "PASS", "PASS", "REVISE_UNINFORMATIVE"),
        ("development", "PASS", "PASS", "FAIL", "PASS", "REVISE_UNINFORMATIVE"),
        ("development", "PASS", "PASS", "PASS", "FAIL", "REVISE_CAUSAL_ATTRIBUTION_FAILED"),
        ("development", "PASS", "PASS", "PASS", "PASS", "GO"),
    ],
)
def test_scientific_decision_maps_controls_dependence_and_primary_status(
    purpose, primary, informativeness, dependence, causal, expected
):
    assert _candidate_scientific_decision(
        purpose=purpose,
        primary_status=primary,
        informativeness_status=informativeness,
        dependence_status=dependence,
        causal_attribution_status=causal,
    ) == expected


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
        "schema_version": "nursery-corrective-worker-result-v1",
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
            "schema_version": "nursery-corrective-shard-complete-v1",
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
        assert prompt["generator_provenance"].endswith("eval-v1")
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
        / "output/synthetic_corrective_development_launch_package_v1/mechanism_qualification.json"
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
    traces = []
    for model_seed in config["resolved_registries"]["construction_micro"]["model"]:
        model, trace = fit_corrective_mil(
            episodes,
            evidence,
            config,
            model_seed=model_seed,
            condition="synchronized",
        )
        models.append(canonical_digest(model.serializable()))
        traces.append(trace)
    assert len(set(models)) == len(models)
    assert len({row["initial_parameter_digest"] for row in traces}) == len(traces)
    assert len({tuple(row["epoch_order_digests"]) for row in traces}) == len(traces)
    assert all(row["distinct_epoch_parameter_digests"] > 1 for row in traces)
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
    runner_path = snapshot2 / "scripts/run_synthetic_corrective_alignment_v1.py"
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
    inputs = package / "parallel_micro_attempt_4_persisted_inputs"
    jobs_one = package / "parallel_micro_attempt_4_jobs_1"
    jobs_n = package / "parallel_micro_attempt_4_jobs_n"
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
    write_json(package / "parallel_equivalence_attempt_4.json", equivalence)
    write_json(package / "parallel_benchmark_attempt_4.json", {"status": "PASS"})
    monkeypatch.setattr(runner, "compare_adjudication_trees", lambda *_: equivalence)
    monkeypatch.setattr(
        runner, "validate_benchmark_report", lambda *args, **kwargs: {"status": "PASS"}
    )
    monkeypatch.setattr(
        runner, "verify_persisted_inputs", lambda *_: {"status": "PASS"}
    )
    completion = {
        "schema_version": "nursery-corrective-parallel-micro-completion-v1",
        "status": "PASS",
        "prequalification_design_lock_sha256": sha256_file(
            package / "prequalification_design_lock.json"
        ),
        "input_manifest_sha256": sha256_file(inputs / "input_manifest.json"),
        "jobs_1_complete_manifest_sha256": sha256_file(
            jobs_one / "complete_manifest.json"
        ),
        "jobs_n_complete_manifest_sha256": sha256_file(
            jobs_n / "complete_manifest.json"
        ),
        "parallel_equivalence_sha256": sha256_file(
            package / "parallel_equivalence_attempt_4.json"
        ),
        "parallel_benchmark_sha256": sha256_file(
            package / "parallel_benchmark_attempt_4.json"
        ),
        "frozen_jobs": int(config["parallel"]["frozen_jobs"]),
        "scientific_inference_suppressed": True,
        "development_outcome_count": 0,
        "confirmation_outcome_count": 0,
        "non_replayable": True,
    }
    completion_path = package / "parallel_micro_attempt_4_completion.json"
    write_json(completion_path, completion)
    assert runner._validate_parallel_micro_attempt_4(
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
        assert runner._validate_parallel_micro_attempt_4(
            tmp_path, package, config
        )["status"] == "FAIL"
    write_json(completion_path, completion, overwrite=True)
    (jobs_n / "complete_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        runner._validate_parallel_micro_attempt_4(tmp_path, package, config)


def test_benchmark_validator_recomputes_gates_and_telemetry(tmp_path, config):
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
                    "thread_environment_observed": expected_environment,
                    "native_threadpools": [
                        {
                            "internal_api": "openblas",
                            "prefix": "libopenblas",
                            "num_threads": 1,
                        }
                    ],
                }
            )
        rows.sort(key=lambda row: row["unit_id"])
        telemetry = {
            "schema_version": "nursery-corrective-runtime-v1",
            "jobs": jobs,
            "units": rows,
            "wall_seconds_sum": float(sum(row["wall_seconds"] for row in rows)),
            "maximum_worker_peak_rss_native_units": max(
                row["worker_peak_rss_native_units"] for row in rows
            ),
            "excluded_from_adjudication": True,
        }
        write_json(root / "runtime/runtime_telemetry.json", telemetry)
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
        resource_contingency_multiplier=float(
            config["parallel"]["resource_contingency_multiplier"]
        ),
        maximum_host_ram_fraction=float(
            config["parallel"]["maximum_host_ram_fraction"]
        ),
        maximum_current_free_disk_fraction=float(
            config["parallel"]["maximum_current_free_disk_fraction"]
        ),
        minimum_logical_cpu_count=int(
            config["parallel"]["minimum_logical_cpu_count"]
        ),
    )
    assert report["status"] == "PASS"
    kwargs = {
        "jobs_one_root": jobs_one,
        "jobs_n_root": jobs_n,
        "input_root": inputs,
        "config": config,
    }
    assert integrity_module.validate_benchmark_report(report, **kwargs)[
        "status"
    ] == "PASS"
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
        "parallel_equivalence_attempt_4.json",
        "parallel_benchmark_attempt_4.json",
        "parallel_micro_attempt_4_completion.json",
    ):
        write_json(package / name, {"name": name})
    lock_verification = {"status": "PASS", "checks": {"locked": True}}
    micro_validation = {"status": "PASS", "checks": {"complete": True}}
    receipt = {
        "schema_version": "nursery-corrective-construction-attempt-v1",
        "status": "ONE_CONSTRUCTION_ATTEMPT_CONSUMED",
        "prequalification_design_lock_sha256": sha256_file(
            package / "prequalification_design_lock.json"
        ),
        "micro_lock_verification_digest": canonical_digest(lock_verification),
        "parallel_equivalence_sha256": sha256_file(
            package / "parallel_equivalence_attempt_4.json"
        ),
        "parallel_benchmark_sha256": sha256_file(
            package / "parallel_benchmark_attempt_4.json"
        ),
        "parallel_micro_completion_sha256": sha256_file(
            package / "parallel_micro_attempt_4_completion.json"
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
    package = repository / "output/synthetic_corrective_development_launch_package_v1"
    snapshot = package / "frozen_source_snapshot"
    snapshot.mkdir(parents=True)
    snapshot_config = snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
    snapshot_config.parent.mkdir(parents=True)
    value = yaml.safe_load(CONFIG.read_text())
    value["protocol"]["status"] = "frozen"
    snapshot_config.write_text(yaml.safe_dump(value, sort_keys=False))
    prior_source = ROOT / config["registries"]["canonical_prior_registry"]
    prior = snapshot / config["registries"]["canonical_prior_registry"]
    prior.parent.mkdir(parents=True)
    shutil.copyfile(prior_source, prior)
    live_runner = Path(runner.__file__).resolve()
    snapshot_runner = snapshot / "scripts/run_synthetic_corrective_alignment_v1.py"
    snapshot_runner.parent.mkdir(parents=True)
    shutil.copyfile(live_runner, snapshot_runner)
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
            "schema_version": "nursery-corrective-prequalification-design-lock-v1",
            "status": "LOCKED_BEFORE_OFFICIAL_FIXTURES",
            "tracked_nonconfig_manifest": manifest_for_paths(
                snapshot,
                [
                    "scripts/run_synthetic_corrective_alignment_v1.py",
                    str(config["registries"]["canonical_prior_registry"]),
                ],
            ),
            "anticipated_frozen_config_sha256": sha256_file(snapshot_config),
            "qualification_environment": frozen_environment,
        },
    )
    prefreeze_contract = freeze_prefreeze_evidence(package)
    write_json(package / "frozen_environment.json", frozen_environment)
    write_json(package / "frozen_seed_registries.json", registry_snapshot(frozen_config))
    write_json(
        package / "freeze_receipt.json",
        {
            "schema_version": "nursery-corrective-freeze-receipt-v1",
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
                    "configs/synthetic_corrective_alignment_v1.yaml",
                    "scripts/run_synthetic_corrective_alignment_v1.py",
                    str(config["registries"]["canonical_prior_registry"]),
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
        package / "parallel_benchmark_attempt_4.json",
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
    output = repository / "output/synthetic_corrective_development_v1_one_shot"
    staging = repository / "output/.synthetic_corrective_development_v1.staging"
    inner_staging = (
        repository / "output/..synthetic_corrective_development_v1.staging.cohort-staging"
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
            package / "parallel_benchmark_attempt_4.json"
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
        "schema_version": "nursery-corrective-authorization-consumption-v1",
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
            / "configs/synthetic_corrective_alignment_v1.yaml",
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
        "schema_version": "nursery-corrective-authorization-consumption-v1",
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
        "schema_version": "nursery-corrective-capability-issuance-v1",
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
    monkeypatch.setattr(sys, "argv", [str(fixture["snapshot"] / "scripts/run_synthetic_corrective_alignment_v1.py"), *changed_argv])
    with pytest.raises(PermissionError, match="argv"):
        runner._strict_development_preflight(
            snapshot=fixture["snapshot"],
            authorization_path=fixture["authorization"],
            output_root=fixture["output"],
            jobs=4,
        )
    monkeypatch.setattr(sys, "argv", [str(fixture["snapshot"] / "scripts/run_synthetic_corrective_alignment_v1.py"), *fixture["argv"]])
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
        "schema_version": "nursery-corrective-cohort-summary-v1",
        "status": "PASS",
        "purpose": "construction_micro",
        "scientific_decision": "SCIENTIFIC_INFERENCE_SUPPRESSED",
    }
    observed = {}

    def fake_prepare(root, *_args, **_kwargs):
        Path(root).mkdir(parents=True)
        write_json(Path(root) / "input.json", {"status": "PASS"})
        return {"status": "PASS", "input_manifest_sha256": "a" * 64}

    def fake_parallel(*, output_root, expected_input_manifest_sha256, **_kwargs):
        observed["expected_input_manifest_sha256"] = expected_input_manifest_sha256
        compute = Path(output_root)
        compute.mkdir()
        for name in ("adjudication", "runtime", "shards"):
            child = compute / name
            child.mkdir()
            write_json(child / "evidence.json", {"status": "PASS"})
        write_json(compute / "cohort_summary.json", summary)
        return copy.deepcopy(summary)

    monkeypatch.setattr(parallel_module, "prepare_persisted_inputs", fake_prepare)
    monkeypatch.setattr(parallel_module, "run_parallel_from_persisted", fake_parallel)
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
        "cohort_summary.json",
        "complete_manifest.json",
        "persisted_inputs/input.json",
        "runtime/evidence.json",
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
    package = tmp_path / "output/synthetic_corrective_development_launch_package_v1"
    package.mkdir(parents=True)
    with pytest.raises(PermissionError, match="finalization proof|launch seal"):
        integrity_module.finalize_package(
            tmp_path,
            package,
            config,
            gates={name: True for name in PACKAGE_GATES},
            finalization_proof=None,
        )


def test_finalize_package_revalidates_forged_factory_proof_and_config(
    tmp_path, monkeypatch, config
):
    root = tmp_path / "repository"
    package = root / "package"
    snapshot_config = (
        package
        / "frozen_source_snapshot/configs/synthetic_corrective_alignment_v1.yaml"
    )
    snapshot_config.parent.mkdir(parents=True)
    snapshot_config.write_text("protocol: frozen\n", encoding="utf-8")
    gates = {name: True for name in PACKAGE_GATES}
    write_json(
        package / "adjudication_inputs.json",
        {
            "schema_version": "nursery-corrective-package-adjudication-input-v1",
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


def test_worker_baseexception_terminates_pool_and_returns_closed_failure(
    monkeypatch,
):
    events = []

    class FakeFuture:
        def result(self):
            raise KeyboardInterrupt("injected interruption")

        def cancel(self):
            events.append("cancel")

    class FakeExecutor:
        def __init__(self, **_kwargs):
            events.append("create")

        def submit(self, _function, _arguments):
            return FakeFuture()

        def shutdown(self, **_kwargs):
            events.append("shutdown")

    monkeypatch.setattr(parallel_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(parallel_module, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        parallel_module,
        "_terminate_process_pool",
        lambda _executor: events.append("terminate"),
    )
    plan = {
        "model_seed_order": [1],
        "units": [{"unit_id": "u", "model_seed": 1}],
    }
    telemetry, failures = _execute_worker_batches(
        plan=plan,
        arguments_by_id={"u": {}},
        jobs=1,
        context=object(),
    )
    assert telemetry == []
    assert failures == [
        {
            "unit_id": "u",
            "exception": "KeyboardInterrupt",
            "message": "injected interruption",
        }
    ]
    assert "cancel" in events
    assert events.count("terminate") == 1


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
    (snapshot / "configs/synthetic_corrective_alignment_v1.yaml").write_text(
        "protocol: frozen\n", encoding="utf-8"
    )
    (snapshot / "scripts/run_synthetic_corrective_alignment_v1.py").write_text(
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
        f"tests/test_synthetic_corrective_alignment_v1.py::{name}"
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
            "tests/test_synthetic_corrective_alignment_v1.py",
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
            snapshot / "configs/synthetic_corrective_alignment_v1.yaml"
        ),
        "runner_sha256": sha256_file(
            snapshot / "scripts/run_synthetic_corrective_alignment_v1.py"
        ),
        "freeze_receipt_sha256": sha256_file(package / "freeze_receipt.json"),
        "frozen_environment_sha256": sha256_file(
            package / "frozen_environment.json"
        ),
        "python_executable": str(python),
        "python_sha256": sha256_file(python),
    }
    report = {
        "schema_version": "nursery-corrective-official-tests-v1",
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
    runner_path = snapshot / "scripts/run_synthetic_corrective_alignment_v1.py"
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
        fixture["snapshot"] / "configs/synthetic_corrective_alignment_v1.yaml",
        repository_root=fixture["snapshot"],
    )
    kwargs = {
        "repository_root": fixture["snapshot"],
        "config_path": fixture["snapshot"]
        / "configs/synthetic_corrective_alignment_v1.yaml",
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
    package = root / "output/synthetic_corrective_development_launch_package_v1"
    snapshot = package / "frozen_source_snapshot"
    for source in (root, snapshot):
        config_path = source / "configs/synthetic_corrective_alignment_v1.yaml"
        runner_path = source / "scripts/run_synthetic_corrective_alignment_v1.py"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        runner_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("protocol: frozen\n", encoding="utf-8")
        runner_path.write_text("# frozen runner\n", encoding="utf-8")
    write_json(package / "prequalification_design_lock.json", {"status": "LOCKED"})
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
            "preservation/preservation_proof.json",
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
    package = EVIDENCE_ROOT / "output/synthetic_corrective_development_launch_package_v1"
    report = package / "parallel_equivalence_attempt_4.json"
    if not report.is_file():
        pytest.skip("parallel equivalence artifact is created later in the package sequence")
    value = read_json(report)
    assert value["status"] == "PASS"
    assert value["byte_identical"]
