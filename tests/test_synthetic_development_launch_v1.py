from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from babyworld_lite.development_launch_v1.adjudicator import (
    REQUIRED_GATES,
    authorization_payload,
    package_decision,
    verify_authorization,
)
from babyworld_lite.development_launch_v1.protocol import (
    IdentifierFirewall,
    IdentifierReference,
    load_config,
    manifest_for_files,
    verify_file_manifest,
)
from babyworld_lite.development_launch_v1.statistics import (
    average_model_replicates,
    bounded_paired_inference,
)
from babyworld_lite.development_launch_v1.study import execute_cohort
from babyworld_lite.sensor_alignment_v8.benchmark import generate_corpus
from babyworld_lite.sensor_alignment_v8.protocol import reject_oracle_fields
from scripts.run_synthetic_development_launch_v1 import (
    _verify_development_authorization,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/synthetic_development_launch_v1.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG, repository_root=ROOT)


def fixture_firewall(config):
    return IdentifierFirewall(
        config,
        allowed_purpose="fixture_rehearsal",
    )


def test_development_registries_carry_v8_reserves_and_are_disjoint(config):
    resolved = config["resolved_registries"]
    assert len(resolved["development"]["corpus"]) == 40
    assert len(resolved["development"]["model"]) == 3
    assert len(resolved["development"]["inference"]) == 1
    assert len(resolved["confirmation_reserve"]["corpus"]) == 40
    assert len(resolved["confirmation_reserve"]["model"]) == 3
    assert resolved["development"]["corpus"][0] == 281001
    assert resolved["confirmation_reserve"]["corpus"][0] == 289001
    names = [
        "fixture_rehearsal",
        "excluded_rehearsal",
        "development",
        "confirmation_reserve",
    ]
    sets = {
        name: {
            value
            for values in resolved[name].values()
            for value in values
        }
        for name in names
    }
    sets["prior_v1_v8_used"] = set(resolved["prior_v1_v8_used"])
    assert all(
        not sets[left] & sets[right]
        for index, left in enumerate(sets)
        for right in list(sets)[index + 1 :]
    )


def test_firewall_blocks_development_without_authorization_and_confirmation(config):
    with pytest.raises(PermissionError):
        IdentifierFirewall(config, allowed_purpose="development")
    firewall = fixture_firewall(config)
    with pytest.raises(PermissionError):
        firewall.authorize(
            "generate",
            [IdentifierReference("corpus", 281001)],
            purpose="development",
        )
    with pytest.raises(PermissionError):
        firewall.authorize(
            "generate",
            [IdentifierReference("corpus", 289001)],
            purpose="confirmation_reserve",
        )
    assert firewall.authorize(
        "generate",
        [IdentifierReference("corpus", 290001)],
        purpose="fixture_rehearsal",
    )


def test_fixture_corpus_retains_exact_v8_design(config):
    corpus = generate_corpus(
        290001,
        config,
        fixture_firewall(config),
        purpose="fixture_rehearsal",
    )
    assert corpus.audit["status"] == "PASS"
    assert corpus.audit["noun_target_owner_independent"]
    assert corpus.audit["action_relation_and_target_position_exact"]
    assert corpus.audit["maximum_conditional_noun_target_owner_gap"] == 0.0
    assert (
        corpus.audit["factor_assignment_construction"][
            "maximum_exact_persisted_audit_cramers_v"
        ]
        <= 0.16
    )
    for prompt in corpus.evaluation_prompts:
        reject_oracle_fields(prompt)
        assert set(prompt).isdisjoint(
            {"raw_stream", "event_owners", "target_event_index", "answer_index"}
        )


def _fake_metrics(value: float) -> dict:
    return {
        "fractional_accuracy": value,
        "mean_correct_probability": value,
        "mean_exact_candidate_set_chance": 0.2,
        "fractional_margin_over_chance": value - 0.2,
        "probability_margin_over_chance": value - 0.2,
        "brier": 0.01,
        "exact_chance_brier": 0.1,
        "brier_improvement_over_exact_chance": 0.09,
        "log_loss": 0.1,
        "exact_chance_log_loss": 1.0,
        "log_loss_improvement_over_exact_chance": 0.9,
        "tie_frequency": 0.0,
        "mean_tied_maxima": 1.0,
        "calibration_ece_5_bin": 0.1,
        "count": 2,
    }


def test_model_replicates_are_averaged_within_corpus(config):
    rows = []
    for model_seed, value in zip((290101, 290103, 290107), (0.6, 0.7, 0.8)):
        by_kind_presence = {
            kind: {
                presence: _fake_metrics(value)
                for presence in ("present", "null")
            }
            for kind in ("primitive", "manner", "action", "noun")
        }
        rows.append(
            {
                "corpus_seed": 290001,
                "model_seed": model_seed,
                "condition": "synchronized",
                "model_digest": f"digest-{model_seed}",
                "metrics": {"by_kind_presence": by_kind_presence},
            }
        )
    result = average_model_replicates(
        rows,
        config,
        purpose="fixture_rehearsal",
    )
    assert result["status"] == "PASS"
    assert result["distinct_model_digests_per_corpus_condition"]
    assert all(row["model_replicates_averaged"] == 3 for row in result["rows"])
    assert all(
        row["metrics"]["mean_correct_probability"] == pytest.approx(0.7)
        for row in result["rows"]
    )


def test_bounded_inference_fails_closed_on_fixture_degeneracy(config):
    firewall = fixture_firewall(config)
    positive = bounded_paired_inference(
        [0.1, 0.1],
        config,
        firewall,
        purpose="fixture_rehearsal",
        inference_seed=290401,
        contrast="fixture-positive-point-mass",
        lower_bound_threshold=0.005,
    )
    assert positive["degenerate_mode"] == "PASS_POINT_MASS_WITH_EXACT_SIGN_TEST"
    assert positive["status"] == "FAIL"
    json.dumps(positive, allow_nan=False)
    zero = bounded_paired_inference(
        [0.0, 0.0],
        config,
        firewall,
        purpose="fixture_rehearsal",
        inference_seed=290401,
        contrast="fixture-zero-point-mass",
        lower_bound_threshold=0.005,
    )
    assert zero["degenerate_mode"] == "FAIL_DEGENERATE_ZERO"
    assert zero["status"] == "FAIL"
    json.dumps(zero, allow_nan=False)
    noninferior = bounded_paired_inference(
        [-0.01, -0.01],
        config,
        firewall,
        purpose="fixture_rehearsal",
        inference_seed=290401,
        contrast="fixture-null-noninferiority",
        lower_bound_threshold=-0.02,
    )
    assert noninferior["positive_count"] == 2
    assert noninferior["positive_relative_to_required_bound"]


def test_authorization_is_digest_bound_and_confirmation_false(config):
    value = authorization_payload(
        terminal_sha256="a" * 64,
        freeze_receipt_sha256="b" * 64,
        snapshot_manifest_sha256="c" * 64,
        excluded_rehearsal_sha256="d" * 64,
        adversarial_audit_sha256="e" * 64,
        official_tests_sha256="f" * 64,
        exact_command=config["commands"]["development_one_shot"],
        development_registry=config["resolved_registries"]["development"],
        confirmation_registry=config["resolved_registries"][
            "confirmation_reserve"
        ],
    )
    assert verify_authorization(value)
    assert value["development_authorized"]
    assert not value["confirmation_authorized"]
    changed = copy.deepcopy(value)
    changed["development_registry"]["corpus"][0] += 1
    assert not verify_authorization(changed)


def test_final_runner_authorization_guard_verifies_manifest_and_zero_outcomes(
    config, tmp_path
):
    repository = tmp_path / "repository"
    package = repository / "output/package"
    snapshot = package / "frozen_source_snapshot"
    adversarial = (
        repository
        / "output/synthetic_identifiability_adversarial_audit_v8/run_2.json"
    )
    for path, value in (
        (package / "package_terminal.json", {"decision": "DEVELOPMENT_PACKAGE_SEALED"}),
        (package / "freeze_receipt.json", {"status": "FROZEN"}),
        (package / "excluded_rehearsal/cohort_summary.json", {"status": "PASS"}),
        (package / "official_test_execution_report.json", {"status": "PASS"}),
        (
            package / "outcome_registry.json",
            {
                "development_outcome_count": 0,
                "confirmation_outcome_count": 0,
            },
        ),
        (snapshot / "snapshot_manifest.json", {"files": []}),
        (adversarial, {"status": "PASS"}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    authorization = authorization_payload(
        terminal_sha256=__import__("hashlib").sha256(
            (package / "package_terminal.json").read_bytes()
        ).hexdigest(),
        freeze_receipt_sha256=__import__("hashlib").sha256(
            (package / "freeze_receipt.json").read_bytes()
        ).hexdigest(),
        snapshot_manifest_sha256=__import__("hashlib").sha256(
            (snapshot / "snapshot_manifest.json").read_bytes()
        ).hexdigest(),
        excluded_rehearsal_sha256=__import__("hashlib").sha256(
            (package / "excluded_rehearsal/cohort_summary.json").read_bytes()
        ).hexdigest(),
        adversarial_audit_sha256=__import__("hashlib").sha256(
            adversarial.read_bytes()
        ).hexdigest(),
        official_tests_sha256=__import__("hashlib").sha256(
            (package / "official_test_execution_report.json").read_bytes()
        ).hexdigest(),
        exact_command=config["commands"]["development_one_shot"],
        development_registry=config["resolved_registries"]["development"],
        confirmation_registry=config["resolved_registries"][
            "confirmation_reserve"
        ],
    )
    authorization_path = package / "DEVELOPMENT_LAUNCH_READY.json"
    authorization_path.write_text(json.dumps(authorization))
    files = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "complete_file_manifest.json"
    ]
    manifest = manifest_for_files(package, files)
    manifest.update(
        {
            "independent_second_pass_verification": {"status": "PASS"},
            "self_excluded_by_definition": True,
        }
    )
    (package / "complete_file_manifest.json").write_text(json.dumps(manifest))
    assert _verify_development_authorization(
        authorization_path, snapshot, config
    )["status"] == "DEVELOPMENT_LAUNCH_READY"
    (package / "outcome_registry.json").write_text(
        json.dumps(
            {
                "development_outcome_count": 1,
                "confirmation_outcome_count": 0,
            }
        )
    )
    with pytest.raises(PermissionError):
        _verify_development_authorization(authorization_path, snapshot, config)


def test_package_adjudicator_fails_closed():
    all_pass = {name: True for name in REQUIRED_GATES}
    sealed = package_decision({"gates": all_pass})
    assert sealed["decision"] == "DEVELOPMENT_PACKAGE_SEALED"
    assert sealed["development_authorized"]
    assert not sealed["confirmation_authorized"]
    missing = package_decision({"gates": {}})
    assert missing["decision"] == "REVISE"
    failed = dict(all_pass)
    failed["one_shot_output_absent"] = False
    assert package_decision({"gates": failed})["decision"] == "REVISE"


def test_confirmation_command_is_absent_and_claim_is_narrow(config):
    runner = (
        ROOT / "scripts/run_synthetic_development_launch_v1.py"
    ).read_text()
    assert "run-confirmation" not in runner
    assert not config["firewalls"]["confirmation_command_exists"]
    assert (
        config["protocol"]["scientific_claim"]
        == "narrow_raw_sensor_assisted_weak_lexical_action_grounding"
    )
    assert not config["protocol"]["infant_learning_claim_authorized"]
    assert not config["protocol"]["ecological_validity_claim_authorized"]


def test_exact_one_shot_command_and_output_are_frozen(config):
    command = config["commands"]["development_one_shot"]
    assert "run-development" in command
    assert "DEVELOPMENT_LAUNCH_READY.json" in command
    assert "synthetic_development_v1_one_shot" in command
    assert not (ROOT / "output/synthetic_development_v1_one_shot").exists()


def test_development_and_confirmation_cannot_run_through_rehearsal_api(
    config, tmp_path
):
    with pytest.raises(PermissionError):
        execute_cohort(
            ROOT,
            config,
            tmp_path / "development",
            purpose="development",
            development_authorized=False,
        )
    with pytest.raises(PermissionError):
        execute_cohort(
            ROOT,
            config,
            tmp_path / "confirmation",
            purpose="confirmation_reserve",
        )


def test_manifest_rejects_one_byte_mutation(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    manifest = manifest_for_files(tmp_path, ["a.txt"])
    assert verify_file_manifest(tmp_path, manifest)["status"] == "PASS"
    (tmp_path / "a.txt").write_text("aa")
    assert verify_file_manifest(tmp_path, manifest)["status"] == "FAIL"


def test_inference_contract_uses_corpus_as_unit_and_iut(config):
    analysis = config["analysis"]
    assert analysis["independent_unit"] == "corpus_seed"
    assert (
        analysis["model_replicate_handling"]
        == "average_within_corpus_before_inference"
    )
    assert analysis["intersection_union_rule"] == (
        "every_co_primary_contrast_must_pass"
    )
    assert set(analysis["co_primary_contrasts"]) == {
        "synchronized_minus_absent_channel",
        "synchronized_minus_randomized_shuffle",
    }
    assert analysis["bootstrap_replicates"] == 20000


def test_held_out_endpoint_contract_is_cue_free(config):
    endpoints = config["design"]["held_out_endpoints"]
    assert set(endpoints) == {"instance", "composition", "component", "noun"}
    assert all(endpoints.values())
    forbidden = set(
        config["firewalls"]["side_modality_evaluation_fields_forbidden"]
    )
    assert {"raw_stream", "target_event_index", "answer_index"} <= forbidden


def test_v8_qualification_and_adversarial_evidence_are_ready():
    terminal = json.loads(
        (
            ROOT
            / "output/synthetic_identifiability_qualification_v8/"
            "terminal_decision.json"
        ).read_text()
    )
    adversarial = json.loads(
        (
            ROOT
            / "output/synthetic_identifiability_adversarial_audit_v8/run_2.json"
        ).read_text()
    )
    assert terminal["decision"] == "VERSION_READY"
    assert adversarial["decision"] == "SURVIVES_ADVERSARIAL_AUDIT"
    assert adversarial["development_outcome_count"] == 0
    assert adversarial["confirmation_outcome_count"] == 0
