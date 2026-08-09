from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from babyworld_lite.sensor_alignment_v2 import detector as v2_detector
from babyworld_lite.sensor_alignment_v4.analysis import (
    factor_stratification_audit,
    infer_mean,
)
from babyworld_lite.sensor_alignment_v4.controls import (
    fit_direct_capacity,
    fit_oracle_alignment,
)
from babyworld_lite.sensor_alignment_v4.corpus import (
    EvalPrompt,
    audit_corpus,
    condition_view,
    generate_corpus,
)
from babyworld_lite.sensor_alignment_v4.detector_runtime import DetectorRuntime
from babyworld_lite.sensor_alignment_v4.learner import (
    fit_joint_competitive,
    predict_prompts,
    score_predictions,
)
from babyworld_lite.sensor_alignment_v4.protocol import (
    OPERATIONS,
    SeedFirewall,
    SeedReference,
    canonical_digest,
    load_config,
    reject_oracle_fields,
)
from babyworld_lite.sensor_alignment_v4.study import (
    compare_core_artifacts,
    factor_analysis,
    verify_preservation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/synthetic_protocol_fidelity_v4.yaml"
OUTPUT = ROOT / "output/synthetic_protocol_fidelity_v4"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def firewall(config):
    return SeedFirewall(config, allowed_purposes=["fixture_only"])


@pytest.fixture(scope="module")
def corpus(config):
    return generate_corpus(140001, config, firewall(config), purpose="fixture_only")


@pytest.fixture(scope="module")
def synchronized(corpus):
    return condition_view(corpus.visible_episodes, "synchronized")[0]


@pytest.fixture(scope="module")
def detector_evidence(config, synchronized):
    runtime = DetectorRuntime.load(
        ROOT / config["detector"]["frozen_v2_path"],
        config["detector"]["frozen_v2_sha256"],
    )
    return runtime.infer_episodes(
        synchronized,
        firewall(config),
        corpus_seed=140001,
        purpose="fixture_only",
    )


def _all_prior_seed_values() -> set[int]:
    output: set[int] = set()
    for name in (
        "synthetic_weak_alignment_recovery_v1.yaml",
        "synthetic_sensor_event_robustness_v2.yaml",
        "synthetic_component_robustness_v3.yaml",
    ):
        seeds = yaml.safe_load((ROOT / "configs" / name).read_text())["seeds"]

        def walk(value):
            if isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
            elif isinstance(value, int):
                output.add(value)

        walk(seeds)
    return output


def test_registries_cover_every_prior_seed_and_are_disjoint(config):
    assert _all_prior_seed_values() <= set(config["seeds"]["prior_v1_v3"])
    registries = []
    for name in (
        "fixture_only",
        "excluded_smoke",
        "future_development",
        "confirmation_reserve",
    ):
        values = {seed for role in config["seeds"][name].values() for seed in role}
        registries.append(values)
    registries.append(set(config["seeds"]["prior_v1_v3"]))
    assert all(
        not registries[index] & registries[other]
        for index in range(len(registries))
        for other in range(index)
    )


def test_seed_firewall_blocks_every_operation_and_bypass(config):
    guard = firewall(config)
    for operation in OPERATIONS:
        with pytest.raises(PermissionError):
            guard.authorize(
                operation,
                [SeedReference("corpus", 149001)],
                purpose="confirmation_reserve",
            )
        with pytest.raises(PermissionError):
            guard.authorize(
                operation,
                [SeedReference("corpus", 141001)],
                purpose="future_development",
            )
        with pytest.raises(PermissionError):
            guard.authorize(
                operation,
                [SeedReference("corpus", 31013)],
                purpose="fixture_only",
            )
    with pytest.raises(PermissionError):
        guard.authorize(
            "generate", [SeedReference("model", 140001)], purpose="fixture_only"
        )
    token = guard.authorize(
        "generate", [SeedReference("corpus", 140001)], purpose="fixture_only"
    )
    assert token and guard.log[-1]["status"] == "ALLOW"
    assert any(row["status"] == "BLOCK" for row in guard.log)


def test_exact_relation_balance_and_noun_sensor_independence(corpus):
    audit = corpus.audit
    assert audit["status"] == "PASS"
    assert audit["exact_action_relation_cell_balance"]
    assert audit["exact_noun_target_owner_factorial"]
    assert audit["noun_target_owner_rate"] == 0.25
    assert audit["noun_distractor_owner_rate_per_opportunity"] == 0.25
    for counts in audit["action_relation_counts_by_concept"].values():
        assert set(counts.values()) == {16}
    for counts in audit["noun_target_owner_position_joint_counts"].values():
        assert len(counts) == 16 and set(counts.values()) == {1}


def test_v3_like_noun_target_coupling_fails_structural_audit(corpus):
    corrupted = copy.deepcopy(list(corpus.oracle_episodes))
    for row in corrupted:
        if row["family"] == "noun":
            row["event_owners"] = [
                index == row["target_event_index"] for index in range(4)
            ]
    audit = audit_corpus(corpus.visible_episodes, corrupted)
    assert audit["status"] == "FAIL"
    assert not audit["noun_owner_independent_of_identity_and_target"]


def test_visibility_changes_observations_and_lag_changes_speech_time(corpus):
    rows = list(corpus.visible_episodes)
    low = next(row for row in rows if row["factor_values"]["visibility"] == 0.55)
    high = next(row for row in rows if row["factor_values"]["visibility"] == 1.0)
    assert low["events"][0]["action_observation"] != high["events"][0]["action_observation"]
    lag_zero = next(row for row in rows if row["factor_values"]["lag"] == 0)
    lag_eight = next(row for row in rows if row["factor_values"]["lag"] == 8)
    assert lag_zero["utterance"]["speech_time"] != lag_eight["utterance"]["speech_time"]


def test_actual_frozen_detector_is_loaded_and_invoked(config, synchronized, monkeypatch):
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
    runtime.infer_episode(synchronized[0])
    audit = runtime.audit(expected_minimum_calls=1)
    assert calls == [(v2_detector.SensorEventDetector, 4)]
    assert audit["status"] == "PASS"
    assert audit["inference_calls"] == 1
    assert not audit["hash_only_placeholder"]


def test_hash_only_detector_placeholder_and_wrong_digest_fail(config):
    runtime = DetectorRuntime.load(
        ROOT / config["detector"]["frozen_v2_path"],
        config["detector"]["frozen_v2_sha256"],
    )
    assert runtime.audit(expected_minimum_calls=1)["status"] == "FAIL"
    assert runtime.audit(expected_minimum_calls=1)["hash_only_placeholder"]
    with pytest.raises(ValueError):
        DetectorRuntime.load(
            ROOT / config["detector"]["frozen_v2_path"], "0" * 64
        )


def test_joint_competition_iterations_and_weight_materially_participate(
    config, synchronized, detector_evidence
):
    guard = firewall(config)
    configured, trace = fit_joint_competitive(
        synchronized,
        detector_evidence,
        config,
        guard,
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
        use_sensor=True,
    )
    one_iteration, _ = fit_joint_competitive(
        synchronized,
        detector_evidence,
        config,
        guard,
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
        use_sensor=True,
        iterations_override=1,
    )
    no_competition, _ = fit_joint_competitive(
        synchronized,
        detector_evidence,
        config,
        guard,
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
        use_sensor=True,
        competition_override=0.0,
    )
    assert canonical_digest(configured.serializable()) != canonical_digest(
        one_iteration.serializable()
    )
    assert canonical_digest(configured.serializable()) != canonical_digest(
        no_competition.serializable()
    )
    assert trace["iterations_executed"] == config["learner"]["iterations"]
    assert any(value > 0 for value in trace["iteration_competition_penalties"])
    assert trace["sensor_applications_by_slot"]["noun"] == 0


def test_declared_visibility_and_lag_are_not_inert(config, corpus, detector_evidence):
    rows = list(corpus.visible_episodes)
    digests = {}
    for factor, levels in (("visibility", [0.55, 1.0]), ("lag", [0, 8])):
        for level in levels:
            selected = [
                row
                for row in rows
                if float(row["factor_values"][factor]) == float(level)
            ]
            selected_evidence = {
                row["episode_id"]: detector_evidence[row["episode_id"]]
                for row in selected
            }
            model, _ = fit_joint_competitive(
                selected,
                selected_evidence,
                config,
                firewall(config),
                corpus_seed=140001,
                model_seed=140101,
                purpose="fixture_only",
                use_sensor=True,
            )
            digests[(factor, level)] = canonical_digest(model.serializable())
        assert digests[(factor, levels[0])] != digests[(factor, levels[1])]


def test_factor_rows_are_real_strata_and_pooled_copies_fail(
    config, corpus, synchronized, detector_evidence
):
    rows = factor_analysis(
        corpus,
        synchronized,
        detector_evidence,
        config,
        firewall(config),
        model_seed=140101,
        purpose="fixture_only",
    )
    assert factor_stratification_audit(rows)["status"] == "PASS"
    pooled = copy.deepcopy(rows)
    for row in pooled:
        row["level"] = "pooled_training_distribution"
        row["training_episode_ids_digest"] = "same"
        row["model_digest"] = "same"
    assert factor_stratification_audit(pooled)["status"] == "FAIL"


def test_oracle_fields_rejected_and_keys_cannot_change_predictions(
    config, corpus, synchronized
):
    corrupted = copy.deepcopy(synchronized)
    corrupted[0]["target_event_index"] = 0
    with pytest.raises(ValueError):
        reject_oracle_fields(corrupted[0])
    with pytest.raises(ValueError):
        fit_joint_competitive(
            corrupted,
            {},
            config,
            firewall(config),
            corpus_seed=140001,
            model_seed=140101,
            purpose="fixture_only",
            use_sensor=False,
        )
    model, _ = fit_joint_competitive(
        synchronized,
        {},
        config,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
        use_sensor=False,
    )
    predictions = predict_prompts(
        model,
        corpus.evaluation_prompts,
        firewall(config),
        corpus_seed=140001,
        purpose="fixture_only",
    )
    permuted_keys = tuple(
        replace(key, answer_index=(key.answer_index + 1) % 6)
        for key in corpus.evaluation_keys
    )
    assert predictions == predict_prompts(
        model,
        corpus.evaluation_prompts,
        firewall(config),
        corpus_seed=140001,
        purpose="fixture_only",
    )
    original_score = score_predictions(
        predictions,
        corpus.evaluation_keys,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
    )
    permuted_score = score_predictions(
        predictions,
        permuted_keys,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
    )
    assert original_score != permuted_score


def test_sensor_free_baseline_is_non_degenerate_without_manner_label_guessing(
    config, corpus, synchronized
):
    model, trace = fit_joint_competitive(
        synchronized,
        {},
        config,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
        use_sensor=False,
    )
    score = _score_model(config, corpus, model)
    assert score["action_accuracy"] == 0.5
    assert trace["sensor_free_symmetric_projection_applied"]
    assert all(
        value == (0.5, 0.5)
        for key, value in model.prototypes.items()
        if key.startswith("manner|")
    )


def _score_model(config, corpus, model):
    predictions = predict_prompts(
        model,
        corpus.evaluation_prompts,
        firewall(config),
        corpus_seed=140001,
        purpose="fixture_only",
    )
    return score_predictions(
        predictions,
        corpus.evaluation_keys,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
    )


def test_positive_controls_are_executed_and_fail_on_corruption(
    config, corpus, synchronized
):
    clean_oracle, trace = fit_oracle_alignment(
        synchronized,
        corpus.oracle_episodes,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
    )
    clean_oracle_score = _score_model(config, corpus, clean_oracle)["action_accuracy"]
    assert clean_oracle_score >= config["gates"]["minimum_oracle_action_accuracy"]
    assert trace["target_index_reads"] > 0 and not trace["assigned_score_constant"]
    corrupted_oracle = copy.deepcopy(list(corpus.oracle_episodes))
    for row in corrupted_oracle:
        target_concept = row["intended_concept"]
        candidates = row["event_concepts"]
        if row["family"] == "action":
            target_primitive, target_manner = divmod(target_concept, 2)
            wrong = next(
                index
                for index, value in enumerate(candidates)
                if value // 2 != target_primitive and value % 2 != target_manner
            )
        else:
            wrong = next(
                index for index, value in enumerate(candidates) if value != target_concept
            )
        row["target_event_index"] = wrong
    corrupt_oracle_model, _ = fit_oracle_alignment(
        synchronized,
        corrupted_oracle,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
    )
    assert _score_model(config, corpus, corrupt_oracle_model)["action_accuracy"] < config[
        "gates"
    ]["minimum_oracle_action_accuracy"]

    direct, direct_trace = fit_direct_capacity(
        corpus.evaluation_prompts,
        corpus.evaluation_keys,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
    )
    assert _score_model(config, corpus, direct)["action_accuracy"] == 1.0
    assert not direct_trace["assigned_score_constant"]
    corrupted_prompts = tuple(
        EvalPrompt(
            prompt.prompt_id,
            prompt.kind,
            tuple([1.0 / len(prompt.observation)] * len(prompt.observation)),
            prompt.candidate_words,
        )
        for prompt in corpus.evaluation_prompts
    )
    corrupt_direct, _ = fit_direct_capacity(
        corrupted_prompts,
        corpus.evaluation_keys,
        firewall(config),
        corpus_seed=140001,
        model_seed=140101,
        purpose="fixture_only",
    )
    assert _score_model(config, corpus, corrupt_direct)["action_accuracy"] < config[
        "gates"
    ]["minimum_direct_capacity_action_accuracy"]


def test_zero_variance_and_nondegenerate_inference_are_explicit(config):
    arguments = {
        "seed": 140151,
        "resamples": 300,
        "alpha": 0.05,
        "zero_variance_tolerance": 1e-12,
    }
    degenerate = infer_mean([0.4, 0.4, 0.4], **arguments)
    assert degenerate["method"] == "degenerate_point_mass"
    assert degenerate["ci_low"] == degenerate["ci_high"] == 0.4
    assert not degenerate["bootstrap_interval_computed"]
    assert not degenerate["population_uncertainty_estimated"]
    regular = infer_mean([0.1, 0.2, 0.35, 0.5], **arguments)
    assert regular["method"] == "studentized_bootstrap_t"
    assert regular["valid_studentized_resamples"] > 0
    assert regular["ci_low"] < regular["ci_high"]


def test_reproduction_requires_real_second_directory_and_byte_identity(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "core.json").write_text('{"x":1}\n')
    (second / "core.json").write_text('{"x":1}\n')
    passing = compare_core_artifacts(first, second, tmp_path / "pass.json")
    assert passing["status"] == "PASS"
    assert passing["reproduction_executed"]
    (second / "core.json").write_text('{"x":2}\n')
    failing = compare_core_artifacts(first, second, tmp_path / "fail.json")
    assert failing["status"] == "FAIL"
    missing = compare_core_artifacts(first, tmp_path / "missing", tmp_path / "missing.json")
    assert missing["status"] == "FAIL"
    assert not missing["reproduction_executed"]


def test_complete_v1_v3_preservation_baseline_is_current(tmp_path):
    proof = verify_preservation(
        ROOT,
        OUTPUT / "preserved_v1_v3_hashes_before.tsv",
        tmp_path / "after.tsv",
        tmp_path / "proof.json",
    )
    assert proof["status"] == "PASS"
    assert proof["baseline_file_count"] == 725
    assert proof["before_manifest_sha256"]


def test_v3_forensic_disposition_is_unambiguous():
    text = (ROOT / "docs/synthetic_component_robustness_v3_forensic_audit_v4.md").read_text()
    assert "scientific execution INVALID / NEEDS REVISION" in text
    assert "v3 STOP retained" in text
    assert "No v3 file was changed and v3 was not rerun" in text
