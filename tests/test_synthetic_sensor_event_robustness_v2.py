from __future__ import annotations

import copy
from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np
import pytest

from babyworld_lite.sensor_alignment_v2.analysis import corpus_condition_contrast
from babyworld_lite.sensor_alignment_v2.detector import (
    evaluate_detector,
    evidence_for_episodes,
    fit_detector,
)
from babyworld_lite.sensor_alignment_v2.learners import (
    EvaluationPolicy,
    LexicalModel,
    evaluate_final,
    evaluation_firewall_contract,
    fit_learner,
    make_evaluation_policy,
)
from babyworld_lite.sensor_alignment_v2.protocol import (
    guard_records,
    guard_seed_operation,
    load_protocol_config,
    make_confirmation_manifest,
    validate_protocol_config,
    verify_confirmation_manifest,
)
from babyworld_lite.sensor_alignment_v2.synthetic import (
    FINAL_FORBIDDEN_FRAGMENTS,
    FORBIDDEN_RAW_KEYS,
    MODEL_VISIBLE_KEYS,
    RAW_STREAM_KEYS,
    EvaluationItem,
    condition_view,
    generate_calibration_data,
    generate_corpus,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "synthetic_sensor_event_robustness_v2.yaml"
PROTOCOL = ROOT / "docs" / "synthetic_sensor_event_robustness_v2_protocol.md"
SOURCES = ROOT / "docs" / "synthetic_sensor_event_robustness_v2_primary_sources.json"


@pytest.fixture(scope="module")
def fixture_config() -> dict:
    """Registered implementation fixtures only; these are never study outcomes."""
    config = copy.deepcopy(load_protocol_config(CONFIG))
    fixture = config["seeds"]["fixture_only"]
    config["seeds"]["development"] = {
        "corpus": [fixture["corpus"], fixture["corpus_secondary"]],
        "model": [fixture["model"], fixture["model_secondary"]],
    }
    config["seeds"]["generic_calibration"] = {
        "train": [fixture["calibration_train"]],
        "validation": [fixture["calibration_validation"]],
    }
    config["seeds"]["confirmation_reserve"] = {
        "corpus": [fixture["reserve_corpus"]],
        "model": [fixture["reserve_model"]],
        "calibration": [fixture["reserve_calibration"]],
    }
    # Match the frozen total calibration sizes while retaining fixture identifiers.
    config["calibration"]["episodes_per_train_seed"] = 336
    config["calibration"]["episodes_per_validation_seed"] = 120
    validate_protocol_config(config, require_development_size=False)
    return config


@pytest.fixture(scope="module")
def fixture_corpus(fixture_config):
    seed = fixture_config["seeds"]["development"]["corpus"][0]
    return generate_corpus(fixture_config, seed)


@pytest.fixture(scope="module")
def fixture_detector_bundle(fixture_config):
    train = generate_calibration_data(fixture_config, split="train")
    validation = generate_calibration_data(fixture_config, split="validation")
    detector, trace = fit_detector(train, fixture_config)
    metrics = evaluate_detector(detector, validation, fixture_config)
    return train, validation, detector, trace, metrics


def test_protocol_has_twenty_independent_corpora_and_disjoint_seed_registries() -> None:
    config = load_protocol_config(CONFIG)
    assert len(config["seeds"]["development"]["corpus"]) == 20
    assert len(config["seeds"]["development"]["model"]) == 2
    active = {
        *config["seeds"]["development"]["corpus"],
        *config["seeds"]["development"]["model"],
        *config["seeds"]["generic_calibration"]["train"],
        *config["seeds"]["generic_calibration"]["validation"],
    }
    reserve = {
        *config["seeds"]["confirmation_reserve"]["corpus"],
        *config["seeds"]["confirmation_reserve"]["model"],
        *config["seeds"]["confirmation_reserve"]["calibration"],
    }
    v1 = {
        *config["seeds"]["v1_forbidden"]["corpus"],
        *config["seeds"]["v1_forbidden"]["model"],
    }
    assert not active & reserve
    assert not active & v1
    assert not reserve & v1


def test_protocol_predeclares_iut_corpus_inference_and_no_bootstrap() -> None:
    config = load_protocol_config(CONFIG)
    assert config["inference"]["independent_unit"] == "corpus_seed"
    assert config["inference"]["model_seed_role"] == (
        "algorithmic_replicate_averaged_within_corpus"
    )
    assert config["inference"]["multiplicity"].startswith("intersection_union")
    assert config["inference"]["bootstrap_primary_inference"] == "forbidden"
    text = PROTOCOL.read_text()
    assert "FROZEN BEFORE ANY V2 OUTCOME-PRODUCING RUN" in text
    assert "20 corpus effects are the only independent empirical units" in text


def test_source_record_covers_every_required_research_category() -> None:
    payload = json.loads(SOURCES.read_text())
    categories = {source["category"] for source in payload["sources"]}
    assert {
        "raw_multimodal_wearable_activity_recognition",
        "wearable_event_detection_and_segmentation",
        "weakly_supervised_temporal_localization",
        "reliability_aware_multimodal_fusion",
        "learning_using_privileged_information",
        "cross_situational_word_learning",
        "compositional_generalization",
        "few_cluster_inference",
        "few_cluster_randomization_inference",
        "co_primary_multiplicity",
    } <= categories
    assert payload["search_completed_before_outcome_runs"] is True


def test_confirmation_manifest_has_identifiers_but_no_outcomes(fixture_config) -> None:
    manifest = make_confirmation_manifest(fixture_config)
    assert all(verify_confirmation_manifest(manifest, fixture_config).values())
    assert manifest["outcome_fields"] == []
    assert manifest["authorization_present"] is False
    assert manifest["authorized_operations"] == []


@pytest.mark.parametrize(
    "operation", ["generate", "calibrate", "train", "evaluate", "read", "summarize"]
)
def test_confirmation_guard_blocks_every_operation(fixture_config, operation: str) -> None:
    reserve = fixture_config["seeds"]["confirmation_reserve"]
    with pytest.raises(PermissionError, match="confirmation reserve guard blocked"):
        guard_seed_operation(
            fixture_config,
            operation=operation,
            seeds=[reserve["corpus"][0], reserve["model"][0], reserve["calibration"][0]],
        )


def test_confirmation_records_cannot_be_read_or_summarized(fixture_config) -> None:
    reserve = fixture_config["seeds"]["confirmation_reserve"]
    records = [
        {
            "corpus_seed": reserve["corpus"][0],
            "model_seed": reserve["model"][0],
        }
    ]
    for operation in ("read", "summarize"):
        with pytest.raises(PermissionError):
            guard_records(records, fixture_config, operation=operation)


def test_generator_is_deterministic_balanced_and_oracle_separate(
    fixture_config, fixture_corpus
) -> None:
    repeated = generate_corpus(fixture_config, fixture_corpus.corpus_seed)
    assert fixture_corpus.visible_episodes == repeated.visible_episodes
    assert fixture_corpus.oracle_episodes == repeated.oracle_episodes
    assert fixture_corpus.audits["valid"] is True
    assert len(fixture_corpus.visible_episodes) == 240
    assert fixture_corpus.audits["checks"][
        "factor_marginals_balanced_within_both_families"
    ]
    assert all(set(row) == MODEL_VISIBLE_KEYS for row in fixture_corpus.visible_episodes)
    assert all(
        key not in row
        for row in fixture_corpus.visible_episodes
        for key in ("target_event_index", "event_owners", "grounded")
    )
    assert all(
        key in row
        for row in fixture_corpus.oracle_episodes
        for key in ("target_event_index", "event_owners", "true_event_boundaries")
    )


def test_raw_stream_is_six_axis_plus_state_contact_without_semantic_keys(
    fixture_corpus,
) -> None:
    for episode in fixture_corpus.visible_episodes:
        raw = episode["raw_stream"]
        assert set(raw) == RAW_STREAM_KEYS
        assert all(len(value) == 6 for value in raw["imu"])
        assert all(len(value) == 2 for value in raw["proprio"])
        assert all(len(value) == 1 for value in raw["contact"])
        assert not any(
            token in str(key).lower()
            for key in raw
            for token in FORBIDDEN_RAW_KEYS
        )
        assert all(isinstance(value, (int, float)) for row in raw["imu"] for value in row)


def test_action_and_matched_noun_ownership_constructions_are_distinct_only_in_oracle(
    fixture_corpus,
) -> None:
    oracle = list(fixture_corpus.oracle_episodes)
    action_targets = [
        row
        for row in oracle
        if row["family"] == "action" and row["target_event_index"] is not None
    ]
    assert action_targets
    assert all(
        row["event_owners"][int(row["target_event_index"])] == "wearer"
        for row in action_targets
    )
    assert fixture_corpus.audits["checks"][
        "noun_target_ownership_matches_bag_prevalence_empirically"
    ]
    assert {
        row["selectivity_sensor_temporal_coupling"] for row in oracle
    } == {
        "target_relevant_for_action",
        "event_ownership_independent_of_noun_target",
    }


def test_all_conditions_are_paired_and_absence_is_structural(
    fixture_config, fixture_corpus
) -> None:
    checks = fixture_corpus.audits["condition_checks"]
    assert all(checks.values())
    assert fixture_corpus.audits["donor_checks"]["bijection"]
    assert fixture_corpus.audits["donor_checks"]["different_lexical_group"]
    absent = condition_view(fixture_corpus, "absent", fixture_config)
    assert all("raw_stream" not in row for row in absent)
    for condition in set(fixture_config["conditions"]["names"]) - {"absent"}:
        assert all(
            "raw_stream" in row
            for row in condition_view(fixture_corpus, condition, fixture_config)
        )
    hashes = fixture_corpus.audits["condition_raw_row_multiset_hashes"]
    assert len({value for value in hashes.values() if value is not None}) == 1


def test_calibration_is_generic_separate_and_lexically_blind(
    fixture_detector_bundle,
) -> None:
    train, validation, detector, trace, _metrics = fixture_detector_bundle
    assert set(train.seeds).isdisjoint(validation.seeds)
    assert train.provenance["valid"] and validation.provenance["valid"]
    assert train.provenance["lexical_targets_present"] is False
    assert train.provenance["referent_targets_present"] is False
    assert train.provenance["randomized_word_mappings_present"] is False
    assert trace["lexical_supervision_used"] is False
    assert trace["referent_supervision_used"] is False
    serialized = detector.serializable()
    assert serialized["lexical_supervision_used"] is False
    assert "token_prototypes" not in serialized


def test_detector_passes_heldout_activity_boundary_owner_and_zero_info_gates(
    fixture_detector_bundle,
) -> None:
    *_rest, metrics = fixture_detector_bundle
    assert metrics["valid"] is True
    assert all(metrics["checks"].values())
    assert metrics["informative_timepoint"]["precision"] >= 0.70
    assert metrics["informative_timepoint"]["recall"] >= 0.70
    assert metrics["informative_boundary"]["f1"] >= 0.60
    assert abs(metrics["zero_information_candidate_owner_auc"] - 0.5) <= 0.12


def test_detector_evidence_is_derived_from_raw_and_has_no_oracle_pointer(
    fixture_config, fixture_corpus, fixture_detector_bundle
) -> None:
    detector = fixture_detector_bundle[2]
    synchronized = condition_view(fixture_corpus, "synchronized", fixture_config)
    evidence = evidence_for_episodes(detector, synchronized)
    assert set(evidence) == {row["episode_id"] for row in synchronized}
    assert all(
        set(value)
        == {
            "event_logits",
            "null_logit",
            "owner_probabilities",
            "quality",
            "availability",
            "top_event_index",
            "timepoint_activity_probability",
            "timepoint_boundary_probability",
        }
        for value in evidence.values()
    )
    assert not any(
        forbidden in value
        for value in evidence.values()
        for forbidden in ("target_event_index", "grounded", "word", "mapping")
    )


def _fit_primary(
    fixture_config, fixture_corpus, detector, condition: str, model_seed: int | None = None
):
    rows = condition_view(fixture_corpus, condition, fixture_config)
    evidence = evidence_for_episodes(detector, rows)
    seed = (
        fixture_config["seeds"]["development"]["model"][0]
        if model_seed is None
        else model_seed
    )
    return fit_learner(
        episodes=rows,
        learner="sensor_latent_cross_occurrence",
        condition=condition,
        corpus_seed=fixture_corpus.corpus_seed,
        model_seed=seed,
        config=fixture_config,
        derived_evidence=evidence,
    )


def test_reliability_accepts_synchronized_action_and_rejects_noun_and_corruption(
    fixture_config, fixture_corpus, fixture_detector_bundle
) -> None:
    detector = fixture_detector_bundle[2]
    synchronized_model, synchronized = _fit_primary(
        fixture_config, fixture_corpus, detector, "synchronized"
    )
    shuffled_model, shuffled = _fit_primary(
        fixture_config, fixture_corpus, detector, "shuffled"
    )
    absent_model, absent = _fit_primary(
        fixture_config, fixture_corpus, detector, "absent"
    )
    sync_action = [
        value["trust"]
        for key, value in synchronized["reliability_by_token"].items()
        if key.startswith(("primitive|", "manner|"))
    ]
    sync_noun = [
        value["trust"]
        for key, value in synchronized["reliability_by_token"].items()
        if key.startswith("noun|")
    ]
    assert sync_action and min(sync_action) == 0.80
    assert sync_noun and max(sync_noun) == 0.0
    assert max(value["trust"] for value in shuffled["reliability_by_token"].values()) == 0.0
    assert max(value["trust"] for value in absent["reliability_by_token"].values()) == 0.0
    assert shuffled_model.token_prototypes == absent_model.token_prototypes
    assert synchronized_model.token_prototypes != absent_model.token_prototypes


def test_model_seeds_create_real_algorithmic_variation(
    fixture_config, fixture_corpus, fixture_detector_bundle
) -> None:
    detector = fixture_detector_bundle[2]
    first_seed, second_seed = fixture_config["seeds"]["development"]["model"]
    first_model, first_trace = _fit_primary(
        fixture_config, fixture_corpus, detector, "synchronized", first_seed
    )
    second_model, second_trace = _fit_primary(
        fixture_config, fixture_corpus, detector, "synchronized", second_seed
    )
    assert first_trace["initialization_digest"] != second_trace["initialization_digest"]
    assert first_trace["active_evidence_updates"] != second_trace["active_evidence_updates"]
    assert first_model.serializable() != second_model.serializable()


def test_oracle_and_pointer_controls_require_separate_oracle_rows(
    fixture_config, fixture_corpus
) -> None:
    rows = condition_view(fixture_corpus, "absent", fixture_config)
    seed = fixture_config["seeds"]["development"]["model"][0]
    with pytest.raises(ValueError, match="separate oracle rows"):
        fit_learner(
            episodes=rows,
            learner="oracle_event_alignment_upper",
            condition="absent",
            corpus_seed=fixture_corpus.corpus_seed,
            model_seed=seed,
            config=fixture_config,
        )
    model, _trace = fit_learner(
        episodes=rows,
        learner="oracle_event_alignment_upper",
        condition="absent",
        corpus_seed=fixture_corpus.corpus_seed,
        model_seed=seed,
        config=fixture_config,
        oracle_rows=fixture_corpus.oracle_episodes,
    )
    assert model.token_prototypes


def test_final_evaluation_is_typed_cue_free_and_fail_closed(
    fixture_config, fixture_corpus, fixture_detector_bundle
) -> None:
    model, _trace = _fit_primary(
        fixture_config, fixture_corpus, fixture_detector_bundle[2], "synchronized"
    )
    policy = make_evaluation_policy(fixture_config)
    metrics = evaluate_final(
        model,
        fixture_corpus.evaluation_items,
        corpus_seed=fixture_corpus.corpus_seed,
        policy=policy,
    )
    assert metrics["training_only_channels_structurally_unavailable"] is True
    assert metrics["arbitrary_zero_word_exposure"] == 0
    assert 0.0 <= metrics["structured_heldout_concept_action_6way_accuracy"] <= 1.0
    assert evaluation_firewall_contract()["valid"] is True
    assert not any(
        fragment in field.name.lower()
        for field in fields(EvaluationItem)
        for fragment in FINAL_FORBIDDEN_FRAGMENTS
    )
    assert not any(
        fragment in field.name.lower()
        for field in fields(LexicalModel)
        for fragment in FINAL_FORBIDDEN_FRAGMENTS
    )
    with pytest.raises(TypeError, match="EvaluationItem schema"):
        evaluate_final(
            model,
            [asdict(fixture_corpus.evaluation_items[0])],
            corpus_seed=fixture_corpus.corpus_seed,
            policy=policy,
        )


def test_evaluation_policy_blocks_reserved_seed_even_with_valid_model(
    fixture_config, fixture_corpus, fixture_detector_bundle
) -> None:
    model, _trace = _fit_primary(
        fixture_config, fixture_corpus, fixture_detector_bundle[2], "synchronized"
    )
    reserve = fixture_config["seeds"]["confirmation_reserve"]
    policy = EvaluationPolicy(
        allowed_corpora=(reserve["corpus"][0],),
        allowed_models=(model.model_seed,),
        blocked_values=(reserve["corpus"][0],),
    )
    with pytest.raises(PermissionError, match="confirmation reserve"):
        evaluate_final(
            model,
            fixture_corpus.evaluation_items,
            corpus_seed=reserve["corpus"][0],
            policy=policy,
        )


def test_corpus_contrast_averages_models_before_inference(fixture_config) -> None:
    records = []
    corpora = fixture_config["seeds"]["development"]["corpus"]
    models = fixture_config["seeds"]["development"]["model"]
    for corpus_index, corpus in enumerate(corpora):
        for model_index, model in enumerate(models):
            base = 0.4 + 0.02 * corpus_index + 0.01 * model_index
            for condition, value in (
                ("synchronized", base + 0.2),
                ("absent", base),
            ):
                records.append(
                    {
                        "analysis_kind": "main",
                        "learner": "sensor_latent_cross_occurrence",
                        "condition": condition,
                        "corpus_seed": corpus,
                        "model_seed": model,
                        "metrics": {"metric": value},
                    }
                )
    result = corpus_condition_contrast(
        records,
        fixture_config,
        learner="sensor_latent_cross_occurrence",
        metric="metric",
        left="synchronized",
        right="absent",
    )
    assert result["n_independent_corpus_seeds"] == 2
    assert result["degrees_of_freedom"] == 1
    assert result["point_estimate"] == pytest.approx(0.2)
    assert len(result["unit_effects"]) == 2


def test_zero_exposure_panel_is_balanced_chance_by_construction(
    fixture_config, fixture_corpus
) -> None:
    model = LexicalModel(token_prototypes={}, learner="empty", model_seed=fixture_config["seeds"]["development"]["model"][0])
    result = evaluate_final(
        model,
        fixture_corpus.evaluation_items,
        corpus_seed=fixture_corpus.corpus_seed,
        policy=make_evaluation_policy(fixture_config),
    )
    assert result["zero_exposure_word_6way_accuracy"] == pytest.approx(1.0 / 6.0)
    assert result["n_structured_items"] > 0
    assert result["n_seen_combination_items"] > 0
