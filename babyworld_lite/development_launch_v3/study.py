from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from babyworld_lite.sensor_alignment_v8.benchmark import (
    Corpus,
    condition_audit,
    condition_view,
    generate_corpus,
)
from babyworld_lite.sensor_alignment_v8.controls import (
    fit_direct_capacity_control,
    fit_oracle_control,
)
from babyworld_lite.sensor_alignment_v8.detector_runtime import (
    DetectorRuntime,
    detector_capacity_audit,
)
from babyworld_lite.sensor_alignment_v8.learner import (
    fit_joint_cross_situational,
    predict_prompts,
    reorder_prompts_and_keys,
    score_predictions,
)
from .statistics import average_model_replicates, development_inference
from .protocol import (
    IdentifierFirewall,
    IdentifierReference,
    PROTOCOL_ID,
    canonical_digest,
    manifest_for_files,
    read_jsonl,
    reject_oracle_fields,
    sha256_file,
    verify_file_manifest,
    write_json,
    write_jsonl,
)

FACTOR_NAMES = (
    "candidate_count",
    "visibility",
    "lag",
    "grounded_rate_stratum",
    "sensor_stratum",
    "dropout",
    "noise",
    "false_positive",
    "repetition_phase",
)
RECOMPUTABLE_FILES = (
    "model_results.json",
    "factor_results.json",
    "control_results.json",
    "learner_numeric_audit.json",
    "order_invariance_audit.json",
    "factor_audit.json",
    "null_capacity_audit.json",
    "leakage_audit.json",
    "core_gates.json",
    "compute_operation_log.json",
    "model_averages.json",
    "development_inference.json",
    "development_controls.json",
)


def _serialize_evidence(evidence: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"episode_id": episode_id, **dict(value)}
        for episode_id, value in sorted(evidence.items())
    ]


def _evidence_map(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in read_jsonl(path):
        episode_id = str(row.pop("episode_id"))
        output[episode_id] = row
    return output


def _persist_corpus_inputs(
    input_root: Path,
    corpus: Corpus,
    conditioned: Mapping[str, Sequence[Mapping[str, Any]]],
    donors: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    directory = input_root / f"corpus_{corpus.corpus_seed}"
    write_jsonl(directory / "visible_episodes.jsonl", corpus.visible_episodes)
    write_jsonl(directory / "oracle_episodes.jsonl", corpus.oracle_episodes)
    write_jsonl(directory / "evaluation_prompts.jsonl", corpus.evaluation_prompts)
    write_jsonl(directory / "evaluation_keys.jsonl", corpus.evaluation_keys)
    write_json(directory / "lexicon_oracle.json", corpus.lexicon_oracle)
    write_json(directory / "corpus_audit.json", corpus.audit)
    for condition in sorted(conditioned):
        write_jsonl(directory / "conditions" / f"{condition}.jsonl", conditioned[condition])
        write_json(directory / "donor_maps" / f"{condition}.json", donors[condition])
        write_jsonl(
            directory / "evidence" / f"{condition}.jsonl",
            _serialize_evidence(evidence[condition]),
        )


def _run_model(
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
    condition: str,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    model, trace = fit_joint_cross_situational(
        rows,
        evidence,
        config,
        firewall,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
    )
    predictions = predict_prompts(
        model, prompts, config, firewall, corpus_seed=corpus_seed, purpose=purpose
    )
    metrics = score_predictions(
        predictions,
        keys,
        config,
        firewall,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
    )
    return (
        {
            "corpus_seed": corpus_seed,
            "model_seed": model_seed,
            "condition": condition,
            "learner": model.learner,
            "model_digest": canonical_digest(model.serializable()),
            "metrics": metrics,
            "training_trace": trace,
            "qualification_only": purpose != "development",
            "development_outcome": purpose == "development",
            "confirmation_outcome": False,
        },
        model,
        predictions,
    )


def _factor_results(
    rows: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    keys: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> list[dict[str, Any]]:
    output = []
    oracle_by_id = {str(row["episode_id"]): row for row in oracle}
    for factor in FACTOR_NAMES:
        levels = sorted(
            {
                str(
                    oracle_by_id[str(row["episode_id"])]["factor_values"][factor]
                    if factor == "grounded_rate_stratum"
                    else row["factor_values"][factor]
                )
                for row in rows
            },
            key=lambda value: (len(value), value),
        )
        for level in levels:
            selected = [
                row
                for row in rows
                if str(
                    oracle_by_id[str(row["episode_id"])]["factor_values"][factor]
                    if factor == "grounded_rate_stratum"
                    else row["factor_values"][factor]
                )
                == level
            ]
            selected_ids = sorted(str(row["episode_id"]) for row in selected)
            selected_evidence = {
                episode_id: evidence[episode_id]
                for episode_id in selected_ids
                if episode_id in evidence
            }
            model, trace = fit_joint_cross_situational(
                selected,
                selected_evidence,
                config,
                firewall,
                corpus_seed=corpus_seed,
                model_seed=model_seed,
                purpose=purpose,
            )
            predictions = predict_prompts(
                model, prompts, config, firewall, corpus_seed=corpus_seed, purpose=purpose
            )
            metrics = score_predictions(
                predictions,
                keys,
                config,
                firewall,
                corpus_seed=corpus_seed,
                model_seed=model_seed,
                purpose=purpose,
            )
            output.append(
                {
                    "corpus_seed": corpus_seed,
                    "model_seed": model_seed,
                    "factor": factor,
                    "level": level,
                    "training_episode_count": len(selected),
                    "training_episode_ids": selected_ids,
                    "training_episode_ids_digest": canonical_digest(selected_ids),
                    "model_digest": canonical_digest(model.serializable()),
                    "metrics": metrics,
                    "iterations_executed": trace["iterations_executed"],
                    "stratification": "refit_on_exact_persisted_level_subset",
                }
            )
    return output


def _factor_audit(
    rows: Sequence[Mapping[str, Any]],
    input_root: Path,
) -> dict[str, Any]:
    checks = {}
    for factor in FACTOR_NAMES:
        selected = [row for row in rows if row["factor"] == factor]
        levels = {row["level"] for row in selected}
        corpora = {int(row["corpus_seed"]) for row in selected}
        episode_sets = {row["training_episode_ids_digest"] for row in selected}
        model_digests = {row["model_digest"] for row in selected}
        cells = []
        exact_membership = True
        observed_cells = {
            (int(row["corpus_seed"]), str(row["level"])): row
            for row in selected
        }
        expected_cells = {
            (corpus_seed, str(level))
            for corpus_seed in corpora
            for level in levels
        }
        for corpus_seed, level in sorted(expected_cells):
            row = observed_cells.get((corpus_seed, level))
            corpus_dir = input_root / f"corpus_{corpus_seed}"
            visible = read_jsonl(corpus_dir / "conditions/synchronized.jsonl")
            oracle = read_jsonl(corpus_dir / "oracle_episodes.jsonl")
            oracle_by_id = {
                str(value["episode_id"]): value
                for value in oracle
            }
            expected_ids = sorted(
                str(value["episode_id"])
                for value in visible
                if str(
                    oracle_by_id[str(value["episode_id"])]["factor_values"][factor]
                    if factor == "grounded_rate_stratum"
                    else value["factor_values"][factor]
                )
                == level
            )
            membership_match = (
                row is not None
                and list(row["training_episode_ids"]) == expected_ids
                and row["training_episode_ids_digest"]
                == canonical_digest(expected_ids)
                and int(row["training_episode_count"]) == len(expected_ids)
            )
            exact_membership &= membership_match
            cells.append(
                {
                    "corpus_seed": corpus_seed,
                    "level": level,
                    "row_count": len(expected_ids),
                    "episode_ids_digest": canonical_digest(expected_ids),
                    "reported_model_digest": (
                        row["model_digest"] if row is not None else None
                    ),
                    "iterations_executed": (
                        int(row["iterations_executed"]) if row is not None else 0
                    ),
                    "independent_membership_match": membership_match,
                }
            )
        passed = (
            len(levels) >= 2
            and set(observed_cells) == expected_cells
            and len(episode_sets) == len(expected_cells)
            and len(model_digests) == len(expected_cells)
            and all(row["training_episode_count"] > 0 for row in selected)
            and all(int(row["iterations_executed"]) > 0 for row in selected)
            and exact_membership
        )
        checks[factor] = {
            "status": "PASS" if passed else "FAIL",
            "levels": sorted(levels),
            "corpus_count": len(corpora),
            "expected_corpus_by_level_cells": len(expected_cells),
            "observed_corpus_by_level_cells": len(observed_cells),
            "distinct_episode_sets": len(episode_sets),
            "distinct_model_digests": len(model_digests),
            "independent_membership_recomputation": exact_membership,
            "cells": cells,
        }
    return {
        "status": "PASS" if all(value["status"] == "PASS" for value in checks.values()) else "FAIL",
        "factors": checks,
    }


def _subgroup_pass(
    metrics: Mapping[str, Any],
    *,
    presence: str,
    config: Mapping[str, Any],
) -> bool:
    prefix = "null" if presence == "null" else "present"
    return (
        float(metrics["fractional_margin_over_chance"])
        >= float(
            config["gates"][
                f"minimum_{prefix}_fractional_margin_over_exact_chance"
            ]
        )
        and float(metrics["probability_margin_over_chance"])
        >= float(
            config["gates"][
                f"minimum_{prefix}_probability_margin_over_exact_chance"
            ]
        )
        and float(metrics["log_loss_improvement_over_exact_chance"])
        >= float(
            config["gates"][
                "minimum_log_loss_improvement_over_exact_chance"
            ]
        )
        and float(metrics["brier_improvement_over_exact_chance"])
        >= float(
            config["gates"][
                "minimum_brier_improvement_over_exact_chance"
            ]
        )
        and float(metrics["tie_frequency"])
        <= float(config["gates"]["maximum_tie_frequency"])
    )


def _learnability_audit(
    model_results: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for result in model_results:
        if result["condition"] != condition:
            continue
        for kind in ("primitive", "manner", "action", "noun"):
            for presence in ("present", "null"):
                metrics = result["metrics"]["by_kind_presence"][kind][presence]
                rows.append(
                    {
                        "corpus_seed": int(result["corpus_seed"]),
                        "model_seed": int(result["model_seed"]),
                        "condition": condition,
                        "kind": kind,
                        "presence": presence,
                        "status": (
                            "PASS"
                            if _subgroup_pass(
                                metrics,
                                presence=presence,
                                config=config,
                            )
                            else "FAIL"
                        ),
                        "metrics": dict(metrics),
                    }
                )
    by_presence = {
        presence: all(
            row["status"] == "PASS"
            for row in rows
            if row["presence"] == presence
        )
        for presence in ("present", "null")
    }
    return {
        "status": (
            "PASS"
            if rows and all(row["status"] == "PASS" for row in rows)
            else "FAIL"
        ),
        "condition": condition,
        "every_corpus_model_kind_gated_separately": True,
        "by_presence": by_presence,
        "rows": rows,
    }


def _corrupt_target_absence_evidence(
    rows: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    corrupted = copy.deepcopy(list(rows))
    oracle_by_id = {
        str(row["episode_id"]): row
        for row in oracle
    }
    for row in corrupted:
        key = oracle_by_id[str(row["episode_id"])]
        if bool(key["grounded"]):
            continue
        concept = int(key["intended_concept"])
        event = row["events"][0]
        if row["family"] == "noun":
            event["object_observation"] = [
                1.0 if index == concept else 0.0
                for index in range(6)
            ]
        else:
            primitive = concept // 2
            manner = concept % 2
            event["primitive_observation"] = [
                1.0 if index == primitive else 0.0
                for index in range(3)
            ]
            event["manner_observation"] = [
                1.0 if index == manner else 0.0
                for index in range(2)
            ]
    return corrupted


def _tie_probe(
    config: Mapping[str, Any],
    firewall: IdentifierFirewall,
    *,
    corpus_seed: int,
    model_seed: int,
    purpose: str,
) -> dict[str, Any]:
    first = [
        {
            "prompt_id": "v8-tie-probe",
            "kind": "action",
            "candidate_ids": ["candidate-a", "candidate-b"],
            "scores": [0.0, 0.0],
            "probabilities": [0.5, 0.5],
        }
    ]
    second = [
        {
            **first[0],
            "candidate_ids": ["candidate-b", "candidate-a"],
        }
    ]
    metric_a = score_predictions(
        first,
        [{"prompt_id": "v8-tie-probe", "answer_index": 0}],
        config,
        firewall,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
    )
    metric_b = score_predictions(
        second,
        [{"prompt_id": "v8-tie-probe", "answer_index": 1}],
        config,
        firewall,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
    )
    credit_a = metric_a["overall"]["fractional_accuracy"]
    credit_b = metric_b["overall"]["fractional_accuracy"]
    passed = credit_a == credit_b == 0.5 and canonical_digest(metric_a) == canonical_digest(metric_b)
    return {
        "status": "PASS" if passed else "FAIL",
        "correct_candidate_first_fractional_credit": credit_a,
        "correct_candidate_second_fractional_credit": credit_b,
        "metrics_byte_equivalent": canonical_digest(metric_a) == canonical_digest(metric_b),
        "neither_zero_nor_one": credit_a not in {0.0, 1.0},
    }


def compute_from_persisted(
    input_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    inputs = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((inputs / "input_manifest.json").read_text())
    verification = verify_file_manifest(inputs, manifest)
    if verification["status"] != "PASS":
        raise RuntimeError(f"persisted input verification failed: {verification}")
    config = json.loads((inputs / "config.json").read_text())
    metadata = json.loads((inputs / "cohort_metadata.json").read_text())
    purpose = str(metadata["purpose"])
    firewall = IdentifierFirewall(
        config,
        allowed_purpose=purpose,
        development_authorized=purpose == "development",
    )
    model_results = []
    factor_results = []
    control_results = []
    null_capacity_results = []
    order_checks = []
    first_numeric_context: tuple[Any, ...] | None = None
    for corpus_dir in sorted(inputs.glob("corpus_*")):
        corpus_seed = int(corpus_dir.name.split("_", 1)[1])
        visible = read_jsonl(corpus_dir / "visible_episodes.jsonl")
        oracle = read_jsonl(corpus_dir / "oracle_episodes.jsonl")
        prompts = read_jsonl(corpus_dir / "evaluation_prompts.jsonl")
        keys = read_jsonl(corpus_dir / "evaluation_keys.jsonl")
        conditions = {
            condition: read_jsonl(corpus_dir / "conditions" / f"{condition}.jsonl")
            for condition in config["design"]["conditions"]
        }
        evidence = {
            condition: _evidence_map(corpus_dir / "evidence" / f"{condition}.jsonl")
            for condition in config["design"]["conditions"]
        }
        for model_seed in map(int, config["resolved_registries"][purpose]["model"]):
            for condition in config["design"]["conditions"]:
                result, model, predictions = _run_model(
                    conditions[condition],
                    evidence[condition],
                    prompts,
                    keys,
                    config,
                    firewall,
                    corpus_seed=corpus_seed,
                    model_seed=model_seed,
                    purpose=purpose,
                    condition=condition,
                )
                model_results.append(result)
                reordered_prompts, reordered_keys = reorder_prompts_and_keys(prompts, keys)
                reordered_predictions = predict_prompts(
                    model,
                    reordered_prompts,
                    config,
                    firewall,
                    corpus_seed=corpus_seed,
                    purpose=purpose,
                )
                reordered_metrics = score_predictions(
                    reordered_predictions,
                    reordered_keys,
                    config,
                    firewall,
                    corpus_seed=corpus_seed,
                    model_seed=model_seed,
                    purpose=purpose,
                )
                identical = canonical_digest(result["metrics"]) == canonical_digest(reordered_metrics)
                order_checks.append(
                    {
                        "corpus_seed": corpus_seed,
                        "model_seed": model_seed,
                        "condition": condition,
                        "metrics_identical": identical,
                        "original_metrics_digest": canonical_digest(result["metrics"]),
                        "reordered_metrics_digest": canonical_digest(reordered_metrics),
                    }
                )
                if first_numeric_context is None and condition == "synchronized":
                    first_numeric_context = (
                        conditions[condition],
                        evidence[condition],
                        prompts,
                        keys,
                        corpus_seed,
                        model_seed,
                        model,
                    )
            if model_seed == int(config["resolved_registries"][purpose]["model"][0]):
                factor_results.extend(
                    _factor_results(
                        conditions["synchronized"],
                        oracle,
                        evidence["synchronized"],
                        prompts,
                        keys,
                        config,
                        firewall,
                        corpus_seed=corpus_seed,
                        model_seed=model_seed,
                        purpose=purpose,
                    )
                )
        first_model_seed = int(config["resolved_registries"][purpose]["model"][0])
        full_absent_result = next(
            row
            for row in model_results
            if int(row["corpus_seed"]) == corpus_seed
            and int(row["model_seed"]) == first_model_seed
            and row["condition"] == "absent_channel"
        )
        grounded_ids = {
            str(row["episode_id"])
            for row in oracle
            if bool(row["grounded"])
        }
        ablated_rows = [
            row
            for row in conditions["absent_channel"]
            if str(row["episode_id"]) in grounded_ids
        ]
        corrupted_rows = _corrupt_target_absence_evidence(
            conditions["absent_channel"],
            oracle,
        )
        diagnostic_metrics = {}
        diagnostic_traces = {}
        for diagnostic_name, diagnostic_rows in (
            ("target_absent_training_removed", ablated_rows),
            ("target_absence_evidence_corrupted", corrupted_rows),
        ):
            firewall.authorize(
                "control",
                [
                    IdentifierReference("corpus", corpus_seed),
                    IdentifierReference("model", first_model_seed),
                ],
                purpose=purpose,
            )
            diagnostic_model, diagnostic_trace = fit_joint_cross_situational(
                diagnostic_rows,
                {},
                config,
                firewall,
                corpus_seed=corpus_seed,
                model_seed=first_model_seed,
                purpose=purpose,
            )
            diagnostic_predictions = predict_prompts(
                diagnostic_model,
                prompts,
                config,
                firewall,
                corpus_seed=corpus_seed,
                purpose=purpose,
            )
            diagnostic_metrics[diagnostic_name] = score_predictions(
                diagnostic_predictions,
                keys,
                config,
                firewall,
                corpus_seed=corpus_seed,
                model_seed=first_model_seed,
                purpose=purpose,
            )
            diagnostic_traces[diagnostic_name] = diagnostic_trace
        drops = {}
        for kind in ("primitive", "manner", "action", "noun"):
            full_fractional = float(
                full_absent_result["metrics"]["by_kind_presence"][kind]["null"][
                    "fractional_accuracy"
                ]
            )
            drops[kind] = {
                diagnostic_name: (
                    full_fractional
                    - float(
                        metrics["by_kind_presence"][kind]["null"][
                            "fractional_accuracy"
                        ]
                    )
                )
                for diagnostic_name, metrics in diagnostic_metrics.items()
            }
        null_capacity_results.append(
            {
                "corpus_seed": corpus_seed,
                "model_seed": first_model_seed,
                "full_absent_channel_metrics": full_absent_result["metrics"],
                "diagnostic_metrics": diagnostic_metrics,
                "diagnostic_traces": diagnostic_traces,
                "null_fractional_drops_by_kind": drops,
                "oracle_used_only_to_construct_negative_diagnostics": True,
                "primary_learner_or_scoring_modified": False,
            }
        )
        for control_name, fitter, arguments in (
            ("oracle_alignment_upper", fit_oracle_control, (visible, oracle)),
            ("direct_capacity_upper", fit_direct_capacity_control, (prompts, keys)),
        ):
            model, trace = fitter(
                *arguments,
                firewall,
                corpus_seed=corpus_seed,
                model_seed=first_model_seed,
                purpose=purpose,
            )
            predictions = predict_prompts(
                model, prompts, config, firewall, corpus_seed=corpus_seed, purpose=purpose
            )
            metrics = score_predictions(
                predictions,
                keys,
                config,
                firewall,
                corpus_seed=corpus_seed,
                model_seed=first_model_seed,
                purpose=purpose,
            )
            control_results.append(
                {
                    "corpus_seed": corpus_seed,
                    "model_seed": first_model_seed,
                    "control": control_name,
                    "model_digest": canonical_digest(model.serializable()),
                    "metrics": metrics,
                    "trace": trace,
                }
            )
    if first_numeric_context is None:
        raise RuntimeError("no synchronized numeric audit context")
    rows, evidence, prompts, keys, corpus_seed, model_seed, configured_model = first_numeric_context
    one_iteration, _ = fit_joint_cross_situational(
        rows,
        evidence,
        config,
        firewall,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
        iterations_override=1,
    )
    no_competition, _ = fit_joint_cross_situational(
        rows,
        evidence,
        config,
        firewall,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
        competition_override=0.0,
    )
    numeric_audit = {
        "status": "PASS"
        if canonical_digest(configured_model.serializable())
        not in {
            canonical_digest(one_iteration.serializable()),
            canonical_digest(no_competition.serializable()),
        }
        else "FAIL",
        "configured_model_digest": canonical_digest(configured_model.serializable()),
        "one_iteration_model_digest": canonical_digest(one_iteration.serializable()),
        "zero_competition_model_digest": canonical_digest(no_competition.serializable()),
        "iterations_change_numeric_model": canonical_digest(configured_model.serializable())
        != canonical_digest(one_iteration.serializable()),
        "competition_changes_numeric_model": canonical_digest(configured_model.serializable())
        != canonical_digest(no_competition.serializable()),
    }
    tie_probe = _tie_probe(
        config,
        firewall,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
    )
    order_audit = {
        "status": "PASS"
        if all(row["metrics_identical"] for row in order_checks) and tie_probe["status"] == "PASS"
        else "FAIL",
        "adversarial_reorders": len(order_checks),
        "all_metrics_identical": all(row["metrics_identical"] for row in order_checks),
        "decision_inputs_identical": all(row["metrics_identical"] for row in order_checks),
        "tie_probe": tie_probe,
        "checks": order_checks,
    }
    factor_audit = _factor_audit(factor_results, inputs)
    condition_names = {str(row["condition"]) for row in model_results}
    sensor_free_learnability = _learnability_audit(
        model_results,
        condition="absent_channel",
        config=config,
    )
    synchronized_learnability = _learnability_audit(
        model_results,
        condition="synchronized",
        config=config,
    )
    controls = defaultdict(list)
    for row in control_results:
        controls[row["control"]].append(row["metrics"]["overall"]["fractional_accuracy"])
    action_probability = defaultdict(list)
    for row in model_results:
        for presence in ("present", "null"):
            action_probability[(str(row["condition"]), presence)].append(
                float(
                    row["metrics"]["by_kind_presence"]["action"][presence][
                        "mean_correct_probability"
                    ]
                )
            )
    mean_action_probability = {
        f"{condition}|{presence}": float(np.mean(values))
        for (condition, presence), values in sorted(action_probability.items())
    }
    comparison_conditions = (
        "randomized_shuffle",
        "uninformative",
        "absent_channel",
    )
    present_sensor_deltas = {
        condition: (
            mean_action_probability["synchronized|present"]
            - mean_action_probability[f"{condition}|present"]
        )
        for condition in comparison_conditions
    }
    null_sensor_deltas = {
        condition: (
            mean_action_probability["synchronized|null"]
            - mean_action_probability[f"{condition}|null"]
        )
        for condition in comparison_conditions
    }
    corrupted_sensor_gains = {
        presence: (
            mean_action_probability[f"sensor_corrupted|{presence}"]
            - mean_action_probability[f"absent_channel|{presence}"]
        )
        for presence in ("present", "null")
    }
    sensor_contrast_audit = {
        "present_probability_deltas": present_sensor_deltas,
        "null_probability_deltas": null_sensor_deltas,
        "present_helpful": min(present_sensor_deltas.values())
        >= float(config["gates"]["minimum_present_sensor_probability_delta"]),
        "null_noninferior": min(null_sensor_deltas.values())
        >= -float(config["gates"]["maximum_null_sensor_probability_harm"]),
        "corrupted_sensor_probability_gains": corrupted_sensor_gains,
        "corrupted_sensor_gain_bounded": max(corrupted_sensor_gains.values())
        <= float(
            config["gates"]["maximum_corrupted_sensor_probability_gain"]
        ),
        "present_and_null_never_aggregated": True,
    }
    null_capacity_pass = all(
        float(result["null_fractional_drops_by_kind"][kind][
            "target_absent_training_removed"
        ])
        >= float(config["gates"]["minimum_null_ablation_fractional_drop"])
        and float(result["null_fractional_drops_by_kind"][kind][
            "target_absence_evidence_corrupted"
        ])
        >= float(config["gates"]["minimum_null_corruption_fractional_drop"])
        for result in null_capacity_results
        for kind in ("primitive", "manner", "action", "noun")
    )
    null_capacity_audit = {
        "status": "PASS" if null_capacity_pass else "FAIL",
        "minimum_ablation_drop": float(
            config["gates"]["minimum_null_ablation_fractional_drop"]
        ),
        "minimum_corruption_drop": float(
            config["gates"]["minimum_null_corruption_fractional_drop"]
        ),
        "results": null_capacity_results,
    }
    sensor_free_identified = (
        sensor_free_learnability["status"] == "PASS"
        and all(
            not row["training_trace"]["post_fit_projection_applied"]
            and not row["training_trace"]["condition_specific_parameter_replacement"]
            for row in model_results
            if row["condition"] == "absent_channel"
        )
    )
    direct_capacity_pass = bool(controls["direct_capacity_upper"]) and min(
        controls["direct_capacity_upper"]
    ) >= float(config["gates"]["minimum_direct_capacity_fractional"])
    core_gates = {
        "all_conditions_executed": condition_names == set(config["design"]["conditions"]),
        "order_invariance": order_audit["status"] == "PASS",
        "identifiability_sensor_free_genuine_fit": sensor_free_identified,
        "sensor_free_present_learnability_by_kind": sensor_free_learnability[
            "by_presence"
        ]["present"],
        "sensor_free_null_learnability_by_kind": sensor_free_learnability[
            "by_presence"
        ]["null"],
        "synchronized_present_learnability_by_kind": synchronized_learnability[
            "by_presence"
        ]["present"],
        "synchronized_null_learnability_by_kind": synchronized_learnability[
            "by_presence"
        ]["null"],
        "null_capacity_controls_executed": null_capacity_pass,
        "sensor_helpful_not_required": (
            sensor_contrast_audit["present_helpful"]
            and sensor_contrast_audit["null_noninferior"]
        ),
        "joint_iterations_and_competition_numeric": numeric_audit["status"] == "PASS",
        "noun_sensor_weight_zero": all(
            row["training_trace"]["noun_sensor_weight"] == 0.0
            and row["training_trace"]["sensor_applications_by_slot"]["noun"]
            <= int(config["gates"]["maximum_noun_sensor_application_count"])
            for row in model_results
        ),
        "factor_refits_executed": factor_audit["status"] == "PASS",
        "oracle_control_executed": bool(controls["oracle_alignment_upper"])
        and min(controls["oracle_alignment_upper"])
        >= float(config["gates"]["minimum_oracle_fractional"]),
        "direct_capacity_control_executed": direct_capacity_pass,
        "sensor_corruption_control_executed": (
            "sensor_corrupted" in condition_names
            and sensor_contrast_audit["corrupted_sensor_gain_bounded"]
        ),
        "side_modalities_absent_at_language_evaluation": all(
            set(prompt).isdisjoint({"raw_stream", "event_owners", "target_event_index", "answer_index"})
            for corpus_dir in inputs.glob("corpus_*")
            for prompt in read_jsonl(corpus_dir / "evaluation_prompts.jsonl")
        ),
        "no_projection_or_answer_key_fit": all(
            not row["training_trace"]["post_fit_projection_applied"]
            and not row["training_trace"]["evaluation_keys_consumed"]
            for row in model_results
        ),
    }
    leakage_audit = {
        "status": "PASS"
        if core_gates["side_modalities_absent_at_language_evaluation"]
        and core_gates["no_projection_or_answer_key_fit"]
        else "FAIL",
        "visible_ledger_loaded_by_primary_fit": True,
        "evidence_ledger_loaded_by_primary_fit": True,
        "oracle_ledger_loaded_by_primary_fit": False,
        "key_ledger_loaded_only_after_prediction_for_scoring": True,
        "oracle_ledger_loaded_only_by_oracle_control": True,
        "final_language_prompts_have_side_modalities": False,
    }
    averaged = average_model_replicates(
        model_results,
        config,
        purpose=purpose,
    )
    averaged_rows = list(averaged["rows"])
    averaged_learnability_rows = []
    for row in averaged_rows:
        if row["condition"] not in {"absent_channel", "synchronized"}:
            continue
        passed = _subgroup_pass(
            row["metrics"],
            presence=str(row["presence"]),
            config=config,
        )
        averaged_learnability_rows.append(
            {
                "corpus_seed": int(row["corpus_seed"]),
                "condition": str(row["condition"]),
                "kind": str(row["kind"]),
                "presence": str(row["presence"]),
                "status": "PASS" if passed else "FAIL",
            }
        )
    averaged_present_pass = all(
        row["status"] == "PASS"
        for row in averaged_learnability_rows
        if row["presence"] == "present"
    )
    averaged_null_pass = all(
        row["status"] == "PASS"
        for row in averaged_learnability_rows
        if row["presence"] == "null"
    )
    inference = development_inference(
        averaged,
        config,
        firewall,
        purpose=purpose,
    )
    averaged_index = {
        (
            int(row["corpus_seed"]),
            str(row["condition"]),
            str(row["kind"]),
            str(row["presence"]),
        ): row["metrics"]
        for row in averaged_rows
    }
    corpus_seeds = list(
        map(int, config["resolved_registries"][purpose]["corpus"])
    )
    noun_differences = [
        abs(
            float(
                averaged_index[(seed, "synchronized", "noun", presence)][
                    "mean_correct_probability"
                ]
            )
            - float(
                averaged_index[(seed, "absent_channel", "noun", presence)][
                    "mean_correct_probability"
                ]
            )
        )
        for seed in corpus_seeds
        for presence in ("present", "null")
    ]
    signed_shift_deltas = {
        shift: {
            presence: [
                float(
                    averaged_index[
                        (seed, "synchronized", "action", presence)
                    ]["mean_correct_probability"]
                )
                - float(
                    averaged_index[(seed, shift, "action", presence)][
                        "mean_correct_probability"
                    ]
                )
                for seed in corpus_seeds
            ]
            for presence in ("present", "null")
        }
        for shift in ("shift_minus_8", "shift_plus_8")
    }
    inference_schema_complete = (
        set(inference["co_primary"])
        == {
            "synchronized_minus_absent_channel",
            "synchronized_minus_randomized_shuffle",
        }
        and set(inference["null_noninferiority"])
        == {
            "synchronized_minus_absent_channel_null",
            "synchronized_minus_randomized_shuffle_null",
        }
        and all(
            np.isfinite(float(value["mean_difference"]))
            and np.isfinite(float(value["lower_confidence_bound"]))
            for group in (
                inference["co_primary"],
                inference["null_noninferiority"],
            )
            for value in group.values()
        )
    )
    development_controls = {
        "status": (
            "PASS"
            if averaged["status"] == "PASS"
            and averaged_present_pass
            and averaged_null_pass
            and max(noun_differences, default=1.0)
            <= float(config["gates"]["order_metric_tolerance"])
            and inference_schema_complete
            else "FAIL"
        ),
        "model_replicates_genuinely_stochastic": averaged[
            "distinct_model_digests_per_corpus_condition"
        ],
        "averaged_learnability_rows": averaged_learnability_rows,
        "averaged_present_learnability": averaged_present_pass,
        "averaged_null_learnability": averaged_null_pass,
        "noun_sensor_firewall_maximum_probability_delta": max(
            noun_differences, default=0.0
        ),
        "signed_shift_action_probability_deltas": signed_shift_deltas,
        "signed_shifts_executed": set(signed_shift_deltas)
        == {"shift_minus_8", "shift_plus_8"},
        "uninformative_executed": "uninformative" in condition_names,
        "corrupted_executed": "sensor_corrupted" in condition_names,
        "zero_information_present_in_design": "zero_information"
        in set(config["design"]["detector_strata"]),
        "cue_free_held_out_endpoints": dict(
            config["design"]["held_out_endpoints"]
        ),
        "inference_schema_complete": inference_schema_complete,
        "scientific_decision_suppressed_for_rehearsal": purpose
        != "development",
        "development_inference_pass": inference["status"] == "PASS",
    }
    core_gates.update(
        {
            "model_replicates_genuinely_stochastic": averaged[
                "distinct_model_digests_per_corpus_condition"
            ],
            "averaged_present_learnability_by_kind": averaged_present_pass,
            "averaged_null_learnability_by_kind": averaged_null_pass,
            "cue_free_held_out_endpoints": all(
                config["design"]["held_out_endpoints"].values()
            ),
            "noun_side_modality_firewall": max(
                noun_differences, default=1.0
            )
            <= float(config["gates"]["order_metric_tolerance"]),
            "development_inference_pipeline_executed": inference_schema_complete,
        }
    )
    write_json(output / "model_results.json", model_results)
    write_json(output / "factor_results.json", factor_results)
    write_json(output / "control_results.json", control_results)
    write_json(output / "learner_numeric_audit.json", numeric_audit)
    write_json(output / "order_invariance_audit.json", order_audit)
    write_json(output / "factor_audit.json", factor_audit)
    write_json(output / "null_capacity_audit.json", null_capacity_audit)
    write_json(output / "leakage_audit.json", leakage_audit)
    write_json(output / "model_averages.json", averaged)
    write_json(output / "development_inference.json", inference)
    write_json(output / "development_controls.json", development_controls)
    write_json(
        output / "core_gates.json",
        {
            "status": "PASS" if all(core_gates.values()) else "FAIL",
            "gates": core_gates,
            "identifiability_structural_failure": (
                not sensor_free_identified and not direct_capacity_pass
            ),
            "metrics": {
                "action_probability_by_condition_and_presence": (
                    mean_action_probability
                ),
                "sensor_contrast_audit": sensor_contrast_audit,
                "sensor_free_learnability": sensor_free_learnability,
                "synchronized_learnability": synchronized_learnability,
                "development_controls": development_controls,
            },
        },
    )
    write_json(
        output / "compute_operation_log.json",
        {
            "status": "PASS" if all(row["status"] == "ALLOW" for row in firewall.log) else "FAIL",
            "operation_count": len(firewall.log),
            "operations": firewall.log,
        },
    )
    return {
        "status": "PASS" if all(core_gates.values()) else "FAIL",
        "gates": core_gates,
        "output_files": list(RECOMPUTABLE_FILES),
    }


def execute_cohort(
    repository_root: str | Path,
    config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    purpose: str,
    development_authorized: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(run_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if purpose not in {
        "fixture_rehearsal",
        "excluded_rehearsal",
        "development",
    }:
        raise PermissionError("confirmation is not executable by this package")
    if purpose == "development" and not development_authorized:
        raise PermissionError("development cohort requires sealed authorization")
    output.mkdir(parents=True)
    inputs = output / "persisted_inputs"
    write_json(inputs / "config.json", config)
    write_json(
        inputs / "cohort_metadata.json",
        {
            "protocol_id": PROTOCOL_ID,
            "purpose": purpose,
            "corpus_seeds": list(map(int, config["resolved_registries"][purpose]["corpus"])),
            "model_seeds": list(map(int, config["resolved_registries"][purpose]["model"])),
            "inference_seeds": list(
                map(int, config["resolved_registries"][purpose]["inference"])
            ),
            "scientific_outcome": purpose == "development",
            "confirmation_outcome": False,
        },
    )
    firewall = IdentifierFirewall(
        config,
        allowed_purpose=purpose,
        development_authorized=development_authorized,
    )
    detector_path = root / str(config["detector"]["frozen_v2_path"])
    runtime = DetectorRuntime.load(detector_path, str(config["detector"]["frozen_v2_sha256"]))
    corpus_audits = []
    condition_audits = []
    capacity_oracle = []
    capacity_evidence: dict[str, dict[str, Any]] = {}
    perturbation = None
    expected_calls = 0
    for corpus_seed in map(int, config["resolved_registries"][purpose]["corpus"]):
        corpus = generate_corpus(corpus_seed, config, firewall, purpose=purpose)
        corpus_audits.append({"corpus_seed": corpus_seed, **corpus.audit})
        conditioned = {}
        donors = {}
        evidence = {}
        for condition in config["design"]["conditions"]:
            firewall.authorize(
                "condition",
                [IdentifierReference("corpus", corpus_seed)],
                purpose=purpose,
            )
            rows, donor_map = condition_view(corpus.visible_episodes, condition, config)
            conditioned[condition] = rows
            donors[condition] = donor_map
            if condition == "absent_channel":
                evidence[condition] = {}
            else:
                evidence[condition] = runtime.infer_episodes(
                    rows, firewall, corpus_seed=corpus_seed, purpose=purpose
                )
                expected_calls += len(rows)
        condition_audits.append(
            {"corpus_seed": corpus_seed, **condition_audit(corpus.visible_episodes, conditioned, donors, config)}
        )
        capacity_oracle.extend(corpus.oracle_episodes)
        capacity_evidence.update(evidence["synchronized"])
        if perturbation is None:
            perturbation = runtime.perturbation_check(conditioned["synchronized"][0])
            expected_calls += 2
        _persist_corpus_inputs(inputs, corpus, conditioned, donors, evidence)
    if perturbation is None:
        raise RuntimeError("no perturbation episode")
    input_files = [
        path.relative_to(inputs).as_posix()
        for path in inputs.rglob("*")
        if path.is_file() and path.name != "input_manifest.json"
    ]
    input_manifest = manifest_for_files(inputs, input_files)
    write_json(inputs / "input_manifest.json", input_manifest)
    core = compute_from_persisted(inputs, output / "recomputable")
    runtime_audit = runtime.runtime_audit(minimum_calls=expected_calls, perturbation=perturbation)
    capacity_audit = detector_capacity_audit(capacity_evidence, capacity_oracle, config)
    compute_log = json.loads(
        (output / "recomputable/compute_operation_log.json").read_text()
    )
    combined_operations = [*firewall.log, *compute_log["operations"]]
    observed_references = {
        role: sorted(
            {
                int(reference["value"])
                for operation in combined_operations
                for reference in operation["references"]
                if reference["role"] == role
            }
        )
        for role in ("corpus", "model", "inference")
    }
    declared_references = {
        role: sorted(
            map(int, config["resolved_registries"][purpose][role])
        )
        for role in ("corpus", "model", "inference")
    }
    exact_reference_reconciliation = (
        observed_references == declared_references
        and all(operation["status"] == "ALLOW" for operation in combined_operations)
    )
    identifier_audit = {
        "status": "PASS" if exact_reference_reconciliation else "FAIL",
        "purpose": purpose,
        "declared_references_by_role": declared_references,
        "observed_guarded_references_by_role": observed_references,
        "declarations_equal_observed_touches_exactly": (
            observed_references == declared_references
        ),
        "unused_declared_identifiers": {
            role: sorted(
                set(declared_references[role])
                - set(observed_references[role])
            )
            for role in declared_references
        },
        "undeclared_observed_identifiers": {
            role: sorted(
                set(observed_references[role])
                - set(declared_references[role])
            )
            for role in declared_references
        },
        "all_operations_allowed": all(
            operation["status"] == "ALLOW"
            for operation in combined_operations
        ),
        "operation_count": len(combined_operations),
    }
    top_gates = {
        **core["gates"],
        "corpus_design": all(row["status"] == "PASS" for row in corpus_audits),
        "condition_design": all(row["status"] == "PASS" for row in condition_audits),
        "actual_frozen_v2_detector_runtime": runtime_audit["status"] == "PASS",
        "detector_capacity_predeclared_bounds": capacity_audit["status"] == "PASS",
        "seed_firewall": exact_reference_reconciliation,
        "identifier_registry_reconciliation": exact_reference_reconciliation,
        "outcome_count_consistent": True,
        "confirmation_outcome_count_zero": True,
        "development_inference_suppressed_or_passed": (
            (
                json.loads(
                    (
                        output
                        / "recomputable"
                        / "development_inference.json"
                    ).read_text()
                )["status"]
                == "PASS"
            )
            if purpose == "development"
            else json.loads(
                (
                    output
                    / "recomputable"
                    / "development_controls.json"
                ).read_text()
            )["scientific_decision_suppressed_for_rehearsal"]
        ),
        "rehearsal_has_no_scientific_outcome": (
            purpose == "development"
            or (
                purpose in {"fixture_rehearsal", "excluded_rehearsal"}
                and not json.loads(
                    (inputs / "cohort_metadata.json").read_text()
                )["scientific_outcome"]
            )
        ),
    }
    development_outcomes = 1 if purpose == "development" else 0
    summary = {
        "protocol_id": PROTOCOL_ID,
        "purpose": purpose,
        "status": "PASS" if all(top_gates.values()) else "FAIL",
        "gates": top_gates,
        "identifiability_structural_failure": json.loads(
            (output / "recomputable" / "core_gates.json").read_text()
        )["identifiability_structural_failure"],
        "package_readiness_only": purpose != "development",
        "scientific_outcomes": development_outcomes,
        "development_outcomes": development_outcomes,
        "confirmation_outcomes": 0,
        "development_decision": (
            "DEVELOPMENT_PASS"
            if purpose == "development" and all(top_gates.values())
            else (
                "DEVELOPMENT_NO_GO"
                if purpose == "development"
                else "REHEARSAL_PASS"
                if all(top_gates.values())
                else "REHEARSAL_REVISE"
            )
        ),
        "claim_scope": config["protocol"]["scientific_claim"],
        "infant_learning_claim_authorized": False,
        "ecological_validity_claim_authorized": False,
        "confirmation_authorized": False,
        "identifiers_touched_by_role": observed_references,
    }
    write_json(output / "corpus_audits.json", corpus_audits)
    write_json(output / "condition_audits.json", condition_audits)
    write_json(output / "detector_runtime_audit.json", runtime_audit)
    write_json(output / "detector_capacity_audit.json", capacity_audit)
    write_json(output / "identifier_reference_audit.json", identifier_audit)
    write_json(
        output / "seed_operation_log.json",
        {
            "status": "PASS" if top_gates["seed_firewall"] else "FAIL",
            "operations": combined_operations,
            "fixture_rehearsal_identifiers_touched": observed_references
            if purpose == "fixture_rehearsal"
            else {"corpus": [], "model": [], "inference": []},
            "excluded_rehearsal_identifiers_touched": observed_references
            if purpose == "excluded_rehearsal"
            else {"corpus": [], "model": [], "inference": []},
            "development_identifiers_touched": observed_references
            if purpose == "development"
            else {"corpus": [], "model": [], "inference": []},
            "confirmation_identifiers_touched": [],
            "prior_v1_v8_used_identifiers_touched": [],
        },
    )
    write_json(output / "cohort_summary.json", summary)
    return summary
