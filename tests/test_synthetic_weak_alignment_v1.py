from __future__ import annotations

import copy
from dataclasses import asdict, fields
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from babyworld_lite.weak_alignment.analysis import paired_hierarchical_bootstrap
from babyworld_lite.weak_alignment.learners import (
    LexiconModel,
    evaluate_without_auxiliary_modality,
    fit_learner,
    modality_withholding_contract,
)
from babyworld_lite.weak_alignment.protocol import (
    guard_records_for_read_or_summary,
    guard_seed_operation,
    load_protocol_config,
    make_confirmation_manifest,
    validate_protocol_config,
    verify_confirmation_manifest,
)
from babyworld_lite.weak_alignment.synthetic import (
    EvaluationItem,
    MODEL_VISIBLE_KEYS,
    SIDE_CONDITIONS,
    generate_corpus,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "synthetic_weak_alignment_recovery_v1.yaml"
FIXTURE_CORPUS_SEED = 70123
FIXTURE_MODEL_SEED = 7
FIXTURE_RESERVED_CORPUS_SEED = 99001
FIXTURE_RESERVED_MODEL_SEED = 991


@pytest.fixture(scope="module")
def fixture_config() -> dict:
    """Non-study seeds: deterministic implementation fixture, never a development outcome."""
    config = copy.deepcopy(load_protocol_config(CONFIG))
    config["seeds"]["development"] = {
        "corpus": [FIXTURE_CORPUS_SEED],
        "model": [FIXTURE_MODEL_SEED],
    }
    config["seeds"]["confirmation_reserve"] = {
        "corpus": [FIXTURE_RESERVED_CORPUS_SEED],
        "model": [FIXTURE_RESERVED_MODEL_SEED],
    }
    config["inference"]["bootstrap_samples"] = 1000
    validate_protocol_config(config)
    return config


@pytest.fixture(scope="module")
def fixture_corpus(fixture_config):
    return generate_corpus(fixture_config, FIXTURE_CORPUS_SEED)


def test_frozen_protocol_declares_required_factors_conditions_and_reserve() -> None:
    config = load_protocol_config(CONFIG)
    assert config["protocol"]["status"] == "frozen"
    assert set(config["factors"]) == {
        "speech_action_lag",
        "grounded_utterance_rate",
        "candidate_event_count",
        "action_visibility_rate",
        "word_occurrence_count",
        "side_informativeness",
    }
    assert set(config["conditions"]["training_side_modality"]) == set(SIDE_CONDITIONS)
    assert not (
        set(config["seeds"]["development"]["corpus"])
        & set(config["seeds"]["confirmation_reserve"]["corpus"])
    )


def test_confirmation_manifest_contains_identifiers_but_no_outcomes(fixture_config) -> None:
    manifest = make_confirmation_manifest(fixture_config)
    checks = verify_confirmation_manifest(manifest, fixture_config)
    assert all(checks.values())
    assert manifest["outcome_fields"] == []
    assert manifest["authorization_present"] is False
    assert not any("accuracy" in key or "loss" in key for key in manifest)


@pytest.mark.parametrize("operation", ["generate", "train", "evaluate", "read", "summarize"])
def test_confirmation_reserve_guard_blocks_every_public_operation(
    fixture_config, operation: str
) -> None:
    with pytest.raises(PermissionError, match="confirmation reserve guard blocked"):
        guard_seed_operation(
            fixture_config,
            operation=operation,
            corpus_seed=FIXTURE_RESERVED_CORPUS_SEED,
            model_seed=FIXTURE_RESERVED_MODEL_SEED,
        )


def test_confirmation_records_cannot_be_read_or_summarized(fixture_config) -> None:
    records = [{
        "corpus_seed": FIXTURE_RESERVED_CORPUS_SEED,
        "model_seed": FIXTURE_RESERVED_MODEL_SEED,
    }]
    for operation in ("read", "summarize"):
        with pytest.raises(PermissionError):
            guard_records_for_read_or_summary(records, fixture_config, operation=operation)


def test_fixture_generator_is_deterministic_balanced_and_oracle_separate(
    fixture_config, fixture_corpus
) -> None:
    repeated = generate_corpus(fixture_config, FIXTURE_CORPUS_SEED)
    assert fixture_corpus.visible_episodes == repeated.visible_episodes
    assert fixture_corpus.oracle_episodes == repeated.oracle_episodes
    assert fixture_corpus.synchronized_side == repeated.synchronized_side
    assert len(fixture_corpus.visible_episodes) == 216
    assert fixture_corpus.audits["valid"] is True
    assert fixture_corpus.audits["checks"]["balanced_factor_grid_within_repetition"]
    assert all(set(row) == MODEL_VISIBLE_KEYS for row in fixture_corpus.visible_episodes)
    visible_text = repr(fixture_corpus.visible_episodes).lower()
    assert "true_action" not in visible_text
    assert "target_event_index" not in visible_text


def test_all_side_conditions_share_inventory_and_absence_is_structural(fixture_corpus) -> None:
    hashes = fixture_corpus.audits["condition_inventory_hashes"]
    assert len(set(hashes.values())) == 1
    assert all("side_scores" not in row for row in fixture_corpus.condition_views["absent"])
    for condition in set(SIDE_CONDITIONS) - {"absent"}:
        assert all("side_scores" in row for row in fixture_corpus.condition_views[condition])
    assert len({
        value for value in fixture_corpus.audits["side_distribution_hashes"].values()
        if value is not None
    }) == 1


def test_group_safe_shuffled_donors_are_bijective_and_matched(fixture_corpus) -> None:
    checks = fixture_corpus.audits["donor_checks"]
    assert all(checks.values())
    assert len(set(fixture_corpus.donor_map.values())) == len(fixture_corpus.donor_map)


def test_informative_fixture_cues_identify_referent_or_null_but_zero_information_does_not(
    fixture_corpus,
) -> None:
    oracle = {row["episode_id"]: row for row in fixture_corpus.oracle_episodes}
    by_information: dict[float, list[float]] = {}
    by_chance: dict[float, list[float]] = {}
    for row in fixture_corpus.condition_views["synchronized"]:
        truth = oracle[row["episode_id"]]
        information = float(truth["factors"]["side_informativeness"])
        n = len(row["events"])
        answer = n if truth["target_event_index"] is None else int(truth["target_event_index"])
        by_information.setdefault(information, []).append(
            float(int(np.asarray(row["side_scores"]).argmax()) == answer)
        )
        by_chance.setdefault(information, []).append(1.0 / (n + 1))
    zero_gap = np.mean(by_information[0.0]) - np.mean(by_chance[0.0])
    informative_gap = np.mean(by_information[0.75] + by_information[0.95]) - np.mean(
        by_chance[0.75] + by_chance[0.95]
    )
    assert abs(zero_gap) < 0.15
    assert informative_gap > 0.30


def test_learners_share_initialization_order_steps_and_fit_fixture(fixture_config, fixture_corpus) -> None:
    traces = []
    models = []
    for condition in ("synchronized", "absent"):
        model, trace = fit_learner(
            episodes=fixture_corpus.condition_views[condition],
            learner="latent_mil_cross_occurrence",
            corpus_seed=FIXTURE_CORPUS_SEED,
            model_seed=FIXTURE_MODEL_SEED,
            config=fixture_config,
        )
        traces.append(trace)
        models.append(model)
    assert traces[0]["initialization_digest"] == traces[1]["initialization_digest"]
    assert traces[0]["data_order_digest"] == traces[1]["data_order_digest"]
    assert traces[0]["inventory_digest"] == traces[1]["inventory_digest"]
    assert traces[0]["optimizer_steps"] == traces[1]["optimizer_steps"]
    assert traces[0]["training_auxiliary_input_consumed"] is True
    assert traces[1]["training_auxiliary_input_consumed"] is False
    assert all(isinstance(model, LexiconModel) for model in models)


def test_oracle_control_requires_physically_separate_oracle_rows(fixture_config, fixture_corpus) -> None:
    with pytest.raises(ValueError, match="separate oracle rows"):
        fit_learner(
            episodes=fixture_corpus.condition_views["absent"],
            learner="oracle_alignment",
            corpus_seed=FIXTURE_CORPUS_SEED,
            model_seed=FIXTURE_MODEL_SEED,
            config=fixture_config,
        )
    model, _trace = fit_learner(
        episodes=fixture_corpus.condition_views["absent"],
        learner="oracle_alignment",
        corpus_seed=FIXTURE_CORPUS_SEED,
        model_seed=FIXTURE_MODEL_SEED,
        config=fixture_config,
        oracle_rows=fixture_corpus.oracle_episodes,
    )
    assert model.action_prototypes


def test_final_evaluation_is_fail_closed_and_has_no_auxiliary_state(fixture_config, fixture_corpus) -> None:
    model, _trace = fit_learner(
        episodes=fixture_corpus.condition_views["synchronized"],
        learner="latent_mil_cross_occurrence",
        corpus_seed=FIXTURE_CORPUS_SEED,
        model_seed=FIXTURE_MODEL_SEED,
        config=fixture_config,
    )
    result = evaluate_without_auxiliary_modality(
        model,
        fixture_corpus.evaluation_items,
        corpus_seed=FIXTURE_CORPUS_SEED,
        config=fixture_config,
    )
    assert 0 <= result["heldout_composition_action_6way_macro_accuracy"] <= 1
    assert result["training_auxiliary_modality_structurally_unavailable"] is True
    contract = modality_withholding_contract()
    assert contract["valid"] is True
    assert not any(
        token in field.name.lower()
        for field in fields(EvaluationItem)
        for token in ("side", "cue", "motor", "imu", "touch")
    )
    assert not any(
        token in field.name.lower()
        for field in fields(LexiconModel)
        for token in ("side", "cue", "motor", "imu", "touch")
    )
    with pytest.raises(TypeError, match="fail-closed"):
        evaluate_without_auxiliary_modality(
            model,
            [asdict(fixture_corpus.evaluation_items[0])],
            corpus_seed=FIXTURE_CORPUS_SEED,
            config=fixture_config,
        )


def test_bootstrap_uses_paired_corpus_model_units_not_episode_rows(fixture_config) -> None:
    records = []
    for condition, value in (("synchronized", 0.75), ("shuffled", 0.50)):
        records.append({
            "analysis_kind": "main",
            "learner": "latent_mil_cross_occurrence",
            "condition": condition,
            "corpus_seed": FIXTURE_CORPUS_SEED,
            "model_seed": FIXTURE_MODEL_SEED,
            "metrics": {"metric": value},
        })
    result = paired_hierarchical_bootstrap(
        records,
        fixture_config,
        learner="latent_mil_cross_occurrence",
        metric="metric",
        left_condition="synchronized",
        right_condition="shuffled",
    )
    assert result["point_estimate"] == pytest.approx(0.25)
    assert result["n_paired_units"] == 1
    assert "no episode/window resampling" in result["bootstrap_method"]


def test_protocol_source_record_precedes_outcomes_and_flags_stale_documents() -> None:
    sources = json.loads(
        (ROOT / "docs" / "synthetic_weak_alignment_recovery_v1_primary_sources.json").read_text()
    )
    protocol = (ROOT / "docs" / "synthetic_weak_alignment_recovery_v1_protocol.md").read_text()
    assert sources["search_completed_before_outcome_runs"] is True
    assert all(source["accessed"] == "2026-07-15" for source in sources["sources"])
    assert "24 tests" in protocol
    assert "13/1,414" in protocol
    assert "terminally stopped" in protocol
