from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from babyworld_lite.sensor_alignment_v2 import detector as v2_detector
from babyworld_lite.sensor_alignment_v8.adjudicator import REQUIRED_GATES, terminal_decision
from babyworld_lite.sensor_alignment_v8.benchmark import condition_audit, condition_view, generate_corpus
from babyworld_lite.sensor_alignment_v8.controls import (
    fit_direct_capacity_control,
    fit_oracle_control,
)
from babyworld_lite.sensor_alignment_v8.detector_runtime import (
    DetectorRuntime,
    detector_capacity_audit,
)
from babyworld_lite.sensor_alignment_v8.integrity import (
    validate_traceability,
    write_complete_manifest,
)
from babyworld_lite.sensor_alignment_v8.learner import (
    fit_joint_cross_situational,
    predict_prompts,
    reorder_prompts_and_keys,
    score_predictions,
)
from babyworld_lite.sensor_alignment_v8.protocol import (
    OPERATIONS,
    IdentifierFirewall,
    IdentifierReference,
    canonical_digest,
    load_config,
    manifest_for_files,
    reject_oracle_fields,
    verify_file_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/synthetic_identifiability_qualification_v8.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG, repository_root=ROOT)


def guard(config):
    return IdentifierFirewall(config, allowed_purposes=["fixture_only"])


@pytest.fixture(scope="module")
def corpus(config):
    return generate_corpus(280001, config, guard(config), purpose="fixture_only")


@pytest.fixture(scope="module")
def conditioned(config, corpus):
    rows = {}
    donors = {}
    for condition in config["design"]["conditions"]:
        rows[condition], donors[condition] = condition_view(
            corpus.visible_episodes, condition, config
        )
    return rows, donors


@pytest.fixture(scope="module")
def synchronized_evidence(config, conditioned):
    runtime = DetectorRuntime.load(
        ROOT / config["detector"]["frozen_v2_path"],
        config["detector"]["frozen_v2_sha256"],
    )
    return runtime.infer_episodes(
        conditioned[0]["synchronized"],
        guard(config),
        corpus_seed=280001,
        purpose="fixture_only",
    )


def test_v8_registries_are_disjoint_and_block_every_v1_v7_identifier(config):
    resolved = config["resolved_registries"]
    names = [
        "fixture_only",
        "excluded_smoke",
        "future_development",
        "confirmation_reserve",
    ]
    sets = {
        name: {value for values in resolved[name].values() for value in values}
        for name in names
    }
    sets["prior_v1_v7"] = set(resolved["prior_v1_v7"])
    assert len(sets["prior_v1_v7"]) >= 150
    assert all(
        not sets[left] & sets[right]
        for index, left in enumerate(sets)
        for right in list(sets)[index + 1 :]
    )


def test_v8_firewall_blocks_prior_development_confirmation_for_all_operations(config):
    firewall = guard(config)
    for operation in OPERATIONS:
        for purpose, value in (
            ("future_development", 281001),
            ("confirmation_reserve", 289001),
            ("fixture_only", config["resolved_registries"]["prior_v1_v7"][0]),
        ):
            with pytest.raises(PermissionError):
                firewall.authorize(
                    operation,
                    [IdentifierReference("corpus", value)],
                    purpose=purpose,
                )
    assert firewall.authorize(
        "generate", [IdentifierReference("corpus", 280001)], purpose="fixture_only"
    )


def test_v8_corpus_crosses_required_factors_absence_null_and_noun_independence(
    config, corpus
):
    audit = corpus.audit
    assert audit["status"] == "PASS"
    assert audit["target_absent_episodes"] > 0
    assert audit["learnable_null_option"]
    assert audit["noun_target_owner_independent"]
    assert set(audit["candidate_count_distribution"]) == {2, 3, 4, 5}
    assert audit["factor_order_correlation_guard"]
    assert audit["factor_control_correlation_guard"]
    construction = audit["factor_assignment_construction"]
    assert construction["status"] == "PASS"
    assert construction["corpus_seed_invariant"]
    assert construction["action_candidate_lattice_seed_invariant"]
    assert construction["search_trials_per_assignment"] == config["design"][
        "factor_assignment_search_trials"
    ]
    assert all(
        1 <= cell["trials_executed"] <= cell["maximum_trials"]
        for cell in construction["cells"]
    )
    assert (
        construction["maximum_exact_persisted_audit_cramers_v"]
        <= construction["bound"]
        < config["gates"]["maximum_factor_control_cramers_v"]
    )
    assert construction["exact_persisted_factor_control_cramers_v"] == audit[
        "maximum_factor_control_cramers_v"
    ]
    assert construction["exact_persisted_pairwise_cramers_v"] == audit[
        "maximum_pairwise_cramers_v"
    ]
    assert audit["all_action_relation_cells_present"]
    assert audit["action_relation_and_target_position_exact"]
    assert audit["maximum_conditional_noun_target_owner_gap"] == 0.0
    assert all(
        value["all_cells_equal_and_positive"]
        for value in audit["noun_target_by_owner_joint_audits"].values()
    )
    assert all(
        value["all_owner_positions_equal_and_positive"]
        for value in audit["noun_absent_owner_position_audits"].values()
    )


def test_v8_visible_oracle_and_key_ledgers_are_separate(corpus):
    for row in corpus.visible_episodes:
        reject_oracle_fields(row)
    for prompt in corpus.evaluation_prompts:
        reject_oracle_fields(prompt)
        assert set(prompt).isdisjoint({"raw_stream", "target_event_index", "answer_index"})
    assert all("answer_index" in row for row in corpus.evaluation_keys)
    assert all("target_event_index" in row for row in corpus.oracle_episodes)


def test_v8_matched_conditions_and_time_shifts_are_real(config, corpus, conditioned):
    rows, donors = conditioned
    audit = condition_audit(corpus.visible_episodes, rows, donors, config)
    assert audit["status"] == "PASS"
    first = corpus.visible_episodes[0]["raw_stream"]
    assert rows["shift_minus_8"][0]["raw_stream"] != first
    assert rows["shift_plus_8"][0]["raw_stream"] != first
    assert rows["sensor_corrupted"][0]["raw_stream"] != first
    assert "raw_stream" not in rows["absent_channel"][0]


def test_v8_actual_frozen_v2_detector_class_and_function_are_invoked(
    config, conditioned, monkeypatch
):
    original = v2_detector.candidate_evidence
    calls = []

    def wrapped(model, raw, intervals):
        calls.append((type(model), len(intervals)))
        return original(model, raw, intervals)

    monkeypatch.setattr(v2_detector, "candidate_evidence", wrapped)
    runtime = DetectorRuntime.load(
        ROOT / config["detector"]["frozen_v2_path"],
        config["detector"]["frozen_v2_sha256"],
    )
    runtime.infer_episode(conditioned[0]["synchronized"][0])
    assert calls[0][0] is v2_detector.SensorEventDetector
    assert runtime.inference_calls == 1
    with pytest.raises(ValueError):
        DetectorRuntime.load(ROOT / config["detector"]["frozen_v2_path"], "0" * 64)


def test_v8_detector_capacity_is_helpful_imperfect_and_stratified(
    config, corpus, synchronized_evidence
):
    audit = detector_capacity_audit(synchronized_evidence, corpus.oracle_episodes, config)
    assert audit["status"] == "PASS"
    assert audit["not_perfect_everywhere"]
    assert audit["not_chance_everywhere"]
    assert audit["strata"]["informative"]["fractional_accuracy"] > audit["strata"]["zero_information"]["fractional_accuracy"]


def test_v8_order_invariant_metrics_and_fractional_ties(
    config, corpus, conditioned
):
    model, trace = fit_joint_cross_situational(
        conditioned[0]["absent_channel"],
        {},
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    prompts = list(corpus.evaluation_prompts)
    keys = list(corpus.evaluation_keys)
    predictions = predict_prompts(
        model, prompts, config, guard(config), corpus_seed=280001, purpose="fixture_only"
    )
    original = score_predictions(
        predictions,
        keys,
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    reordered_prompts, reordered_keys = reorder_prompts_and_keys(prompts, keys)
    reordered_predictions = predict_prompts(
        model,
        reordered_prompts,
        config,
        guard(config),
        corpus_seed=280001,
        purpose="fixture_only",
    )
    reordered = score_predictions(
        reordered_predictions,
        reordered_keys,
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    assert canonical_digest(original) == canonical_digest(reordered)
    tie_prediction = [
        {
            "prompt_id": "tie",
            "kind": "action",
            "candidate_ids": ["a", "b"],
            "scores": [0.0, 0.0],
            "probabilities": [0.5, 0.5],
        }
    ]
    first = score_predictions(
        tie_prediction,
        [{"prompt_id": "tie", "answer_index": 0}],
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    second_prediction = [{**tie_prediction[0], "candidate_ids": ["b", "a"]}]
    second = score_predictions(
        second_prediction,
        [{"prompt_id": "tie", "answer_index": 1}],
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    assert first["overall"]["fractional_accuracy"] == second["overall"]["fractional_accuracy"] == 0.5
    assert trace["post_fit_projection_applied"] is False


def test_v8_sensor_free_gate_uses_genuine_fitted_outputs(config, corpus, conditioned):
    model, trace = fit_joint_cross_situational(
        conditioned[0]["absent_channel"],
        {},
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    metrics = score_predictions(
        predict_prompts(
            model,
            corpus.evaluation_prompts,
            config,
            guard(config),
            corpus_seed=280001,
            purpose="fixture_only",
        ),
        corpus.evaluation_keys,
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    for kind in ("primitive", "manner", "action", "noun"):
        for presence in ("present", "null"):
            subgroup = metrics["by_kind_presence"][kind][presence]
            assert subgroup["fractional_margin_over_chance"] >= config["gates"][
                f"minimum_{presence}_fractional_margin_over_exact_chance"
            ]
            assert subgroup["probability_margin_over_chance"] >= config["gates"][
                f"minimum_{presence}_probability_margin_over_exact_chance"
            ]
            assert subgroup["log_loss_improvement_over_exact_chance"] >= config[
                "gates"
            ]["minimum_log_loss_improvement_over_exact_chance"]
            assert subgroup["brier_improvement_over_exact_chance"] >= config[
                "gates"
            ]["minimum_brier_improvement_over_exact_chance"]
    assert not trace["condition_specific_parameter_replacement"]
    assert not trace["evaluation_keys_consumed"]
    assert not trace["oracle_fields_consumed"]


def test_v8_target_absent_training_is_required_for_every_null_subgroup(
    config, corpus, conditioned
):
    full_model, _ = fit_joint_cross_situational(
        conditioned[0]["absent_channel"],
        {},
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    grounded_ids = {
        row["episode_id"] for row in corpus.oracle_episodes if row["grounded"]
    }
    ablated_model, _ = fit_joint_cross_situational(
        [
            row
            for row in conditioned[0]["absent_channel"]
            if row["episode_id"] in grounded_ids
        ],
        {},
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )

    def score(model):
        return score_predictions(
            predict_prompts(
                model,
                corpus.evaluation_prompts,
                config,
                guard(config),
                corpus_seed=280001,
                purpose="fixture_only",
            ),
            corpus.evaluation_keys,
            config,
            guard(config),
            corpus_seed=280001,
            model_seed=280101,
            purpose="fixture_only",
        )

    full = score(full_model)
    ablated = score(ablated_model)
    for kind in ("primitive", "manner", "action", "noun"):
        drop = (
            full["by_kind_presence"][kind]["null"]["fractional_accuracy"]
            - ablated["by_kind_presence"][kind]["null"]["fractional_accuracy"]
        )
        assert drop >= config["gates"]["minimum_null_ablation_fractional_drop"]


def test_v8_iterations_competition_and_sensor_change_numeric_fit(
    config, corpus, conditioned, synchronized_evidence
):
    arguments = (
        conditioned[0]["synchronized"],
        synchronized_evidence,
        config,
        guard(config),
    )
    configured, trace = fit_joint_cross_situational(
        *arguments,
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    one, _ = fit_joint_cross_situational(
        *arguments,
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
        iterations_override=1,
    )
    no_competition, _ = fit_joint_cross_situational(
        *arguments,
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
        competition_override=0.0,
    )
    sensor_free, _ = fit_joint_cross_situational(
        conditioned[0]["absent_channel"],
        {},
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    assert canonical_digest(configured.serializable()) != canonical_digest(one.serializable())
    assert canonical_digest(configured.serializable()) != canonical_digest(no_competition.serializable())
    assert canonical_digest(configured.serializable()) != canonical_digest(
        sensor_free.serializable()
    )
    assert trace["distinct_iteration_model_digests"] > 1
    assert trace["sensor_semantic_sharpening_weight"] > 0.0
    assert trace["sensor_semantic_corroboration_by_iteration"]
    configured_metrics = score_predictions(
        predict_prompts(
            configured,
            corpus.evaluation_prompts,
            config,
            guard(config),
            corpus_seed=280001,
            purpose="fixture_only",
        ),
        corpus.evaluation_keys,
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    sensor_free_metrics = score_predictions(
        predict_prompts(
            sensor_free,
            corpus.evaluation_prompts,
            config,
            guard(config),
            corpus_seed=280001,
            purpose="fixture_only",
        ),
        corpus.evaluation_keys,
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    assert (
        configured_metrics["by_kind_presence"]["action"]["present"][
            "mean_correct_probability"
        ]
        - sensor_free_metrics["by_kind_presence"]["action"]["present"][
            "mean_correct_probability"
        ]
        >= config["gates"]["minimum_present_sensor_probability_delta"]
    )
    assert (
        configured_metrics["by_kind_presence"]["action"]["null"][
            "mean_correct_probability"
        ]
        - sensor_free_metrics["by_kind_presence"]["action"]["null"][
            "mean_correct_probability"
        ]
        >= -config["gates"]["maximum_null_sensor_probability_harm"]
    )


def test_v8_noun_sensor_weight_is_exactly_zero(config, conditioned, synchronized_evidence):
    _, trace = fit_joint_cross_situational(
        conditioned[0]["synchronized"],
        synchronized_evidence,
        config,
        guard(config),
        corpus_seed=280001,
        model_seed=280101,
        purpose="fixture_only",
    )
    assert trace["noun_sensor_weight"] == 0.0
    assert trace["sensor_applications_by_slot"]["noun"] == 0


def test_v8_oracle_and_direct_controls_are_executed_not_constant_and_corruptible(
    config, corpus
):
    fitters = (
        (fit_oracle_control, (corpus.visible_episodes, corpus.oracle_episodes)),
        (fit_direct_capacity_control, (corpus.evaluation_prompts, corpus.evaluation_keys)),
    )
    for fitter, arguments in fitters:
        model, trace = fitter(
            *arguments,
            guard(config),
            corpus_seed=280001,
            model_seed=280101,
            purpose="fixture_only",
        )
        predictions = predict_prompts(
            model,
            corpus.evaluation_prompts,
            config,
            guard(config),
            corpus_seed=280001,
            purpose="fixture_only",
        )
        baseline = score_predictions(
            predictions,
            corpus.evaluation_keys,
            config,
            guard(config),
            corpus_seed=280001,
            model_seed=280101,
            purpose="fixture_only",
        )["overall"]["fractional_accuracy"]
        corrupted = [
            {**row, "answer_index": (int(row["answer_index"]) + 1) % len(prompt["candidates"])}
            for row, prompt in zip(corpus.evaluation_keys, corpus.evaluation_prompts)
        ]
        corrupted_score = score_predictions(
            predictions,
            corrupted,
            config,
            guard(config),
            corpus_seed=280001,
            model_seed=280101,
            purpose="fixture_only",
        )["overall"]["fractional_accuracy"]
        assert baseline > corrupted_score
        assert trace["assigned_score_constant"] is False


def test_v8_input_and_source_manifests_fail_on_one_byte_mutation(tmp_path):
    (tmp_path / "value.txt").write_text("original")
    manifest = manifest_for_files(tmp_path, ["value.txt"])
    assert verify_file_manifest(tmp_path, manifest)["status"] == "PASS"
    (tmp_path / "value.txt").write_text("changed")
    assert verify_file_manifest(tmp_path, manifest)["status"] == "FAIL"


def test_v8_traceability_breaks_each_reference_class(tmp_path):
    repository = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    repository.mkdir()
    artifacts.mkdir()
    (repository / "module.py").write_text("def existing():\n    return True\n")
    (artifacts / "artifact.json").write_text("{}\n")
    base = {
        "requirements": [
            {
                "symbols": ["module.py:existing"],
                "tests": ["tests/test_x.py::test_real"],
                "artifacts": ["artifact.json"],
                "gates": ["real_gate"],
            }
        ]
    }
    nodeids = ["tests/test_x.py::test_real"]
    gates = {"real_gate": True}
    assert validate_traceability(repository, artifacts, base, nodeids, gates)["status"] == "PASS"
    mutations = {
        "symbols": ("symbols", "module.py:missing"),
        "tests": ("tests", "tests/test_x.py::test_missing"),
        "artifacts": ("artifacts", "missing.json"),
        "gates": ("gates", "missing_gate"),
    }
    for expected_class, (field, value) in mutations.items():
        broken = copy.deepcopy(base)
        broken["requirements"][0][field] = [value]
        result = validate_traceability(repository, artifacts, broken, nodeids, gates)
        assert result["status"] == "FAIL"
        assert result["reference_class_failures"][expected_class]


def test_v8_adjudicator_fails_closed_and_uses_stop_only_for_structural_failure():
    all_pass = {name: True for name in REQUIRED_GATES}
    ready = terminal_decision({"gates": all_pass})
    assert ready["decision"] == "VERSION_READY"
    missing = terminal_decision({"gates": {}})
    assert missing["decision"] == "REVISE"
    structurally_failed = dict(all_pass)
    structurally_failed["identifiability_sensor_free_genuine_fit"] = False
    stopped = terminal_decision(
        {"gates": structurally_failed, "identifiability_structural_failure": True}
    )
    assert stopped["decision"] == "STOP"


def test_v8_complete_manifest_is_per_file_and_second_pass_verified(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("bb")
    manifest = write_complete_manifest(tmp_path)
    assert manifest["file_count"] == 2
    assert manifest["independent_second_pass_verification"]["status"] == "PASS"
    assert all(set(row) == {"path", "bytes", "sha256"} for row in manifest["files"])


def test_v8_preserved_v5_v7_audit_records_revise_and_zero_outcomes():
    audit = json.loads(
        (
            ROOT
            / "docs/synthetic_identifiability_qualification_v8_v5_v7_audit.json"
        ).read_text()
    )
    assert audit["v5"]["terminal_decision"] == "REVISE"
    assert audit["v6"]["status"] == "INVALIDATED_PRE_FREEZE"
    assert audit["v6"]["excluded_smoke_executed"] is False
    assert audit["v7"]["terminal_decision"] == "REVISE"
    assert audit["development_outcome_count"] == 0
    assert audit["confirmation_outcome_count"] == 0
