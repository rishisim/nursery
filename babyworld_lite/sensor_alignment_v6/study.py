from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .benchmark import Corpus, condition_audit, condition_view, generate_corpus
from .controls import (
    fit_corrupted_null_supervision_control,
    fit_direct_capacity_control,
    fit_oracle_control,
)
from .detector_runtime import DetectorRuntime, detector_capacity_audit
from .learner import (
    fit_joint_cross_situational,
    predict_prompts,
    reorder_prompts_and_keys,
    score_predictions,
)
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
    "separated_learnability_audit.json",
    "null_capacity_audit.json",
    "leakage_audit.json",
    "core_gates.json",
    "compute_operation_log.json",
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
            "qualification_only": True,
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


def _factor_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = {}
    for factor in FACTOR_NAMES:
        selected = [row for row in rows if row["factor"] == factor]
        levels = {row["level"] for row in selected}
        corpora = {int(row["corpus_seed"]) for row in selected}
        corpus_level_keys = [
            (int(row["corpus_seed"]), str(row["level"])) for row in selected
        ]
        expected_keys = {
            (corpus_seed, str(level))
            for corpus_seed in corpora
            for level in levels
        }
        episode_sets = {row["training_episode_ids_digest"] for row in selected}
        model_digests = {row["model_digest"] for row in selected}
        rows_by_key = {
            f"corpus={corpus_seed}|level={level}": {
                "corpus_seed": corpus_seed,
                "level": level,
                "training_episode_count": int(row["training_episode_count"]),
                "training_episode_ids": list(row["training_episode_ids"]),
                "training_episode_ids_digest": str(
                    row["training_episode_ids_digest"]
                ),
                "model_digest": str(row["model_digest"]),
                "iterations_executed": int(row["iterations_executed"]),
            }
            for row in selected
            for corpus_seed, level in [
                (int(row["corpus_seed"]), str(row["level"]))
            ]
        }
        passed = (
            len(levels) >= 2
            and bool(corpora)
            and set(corpus_level_keys) == expected_keys
            and len(corpus_level_keys) == len(expected_keys)
            and len(episode_sets) == len(expected_keys)
            and len(model_digests) == len(expected_keys)
            and all(row["training_episode_count"] > 0 for row in selected)
            and all(row["iterations_executed"] > 0 for row in selected)
            and all(
                canonical_digest(sorted(map(str, row["training_episode_ids"])))
                == row["training_episode_ids_digest"]
                for row in selected
            )
        )
        checks[factor] = {
            "status": "PASS" if passed else "FAIL",
            "levels": sorted(levels),
            "corpora": sorted(corpora),
            "expected_corpus_by_level_refits": len(expected_keys),
            "observed_corpus_by_level_refits": len(corpus_level_keys),
            "distinct_episode_sets": len(episode_sets),
            "distinct_model_digests": len(model_digests),
            "corpus_by_level": dict(sorted(rows_by_key.items())),
        }
    return {
        "status": "PASS" if all(value["status"] == "PASS" for value in checks.values()) else "FAIL",
        "factors": checks,
    }


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
            "prompt_id": "v6-tie-probe",
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
        [{"prompt_id": "v6-tie-probe", "answer_index": 0}],
        config,
        firewall,
        corpus_seed=corpus_seed,
        model_seed=model_seed,
        purpose=purpose,
    )
    metric_b = score_predictions(
        second,
        [{"prompt_id": "v6-tie-probe", "answer_index": 1}],
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
    metadata = json.loads((inputs / "qualification_metadata.json").read_text())
    purpose = str(metadata["purpose"])
    firewall = IdentifierFirewall(config, allowed_purposes=[purpose])
    model_results = []
    factor_results = []
    control_results = []
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
        for control_name, fitter, arguments in (
            ("oracle_alignment_upper", fit_oracle_control, (visible, oracle)),
            ("direct_capacity_upper", fit_direct_capacity_control, (prompts, keys)),
            (
                "corrupted_null_supervision",
                fit_corrupted_null_supervision_control,
                (prompts, keys),
            ),
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
        grounded_ids = {
            str(row["episode_id"]) for row in oracle if bool(row["grounded"])
        }
        ablated_rows = [
            row
            for row in conditions["absent_channel"]
            if str(row["episode_id"]) in grounded_ids
        ]
        ablated_model, ablated_trace = fit_joint_cross_situational(
            ablated_rows,
            {},
            config,
            firewall,
            corpus_seed=corpus_seed,
            model_seed=first_model_seed,
            purpose=purpose,
        )
        ablated_predictions = predict_prompts(
            ablated_model,
            prompts,
            config,
            firewall,
            corpus_seed=corpus_seed,
            purpose=purpose,
        )
        ablated_metrics = score_predictions(
            ablated_predictions,
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
                "control": "target_absent_training_removed",
                "model_digest": canonical_digest(ablated_model.serializable()),
                "metrics": ablated_metrics,
                "trace": {
                    **ablated_trace,
                    "oracle_used_only_to_remove_target_absent_rows_for_negative_diagnostic": True,
                    "retained_training_episode_count": len(ablated_rows),
                    "removed_training_episode_count": len(visible)
                    - len(ablated_rows),
                    "primary_learner": False,
                },
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
    factor_audit = _factor_audit(factor_results)
    condition_names = {str(row["condition"]) for row in model_results}
    separated_rows = []
    for result in model_results:
        for kind in ("primitive", "manner", "action", "noun"):
            for presence in ("present", "null"):
                metric = result["metrics"]["by_kind_presence"][kind][presence]
                fractional_threshold = float(
                    config["gates"][
                        f"minimum_{presence}_fractional_margin_over_exact_chance"
                    ]
                )
                probability_threshold = float(
                    config["gates"][
                        f"minimum_{presence}_probability_margin_over_exact_chance"
                    ]
                )
                checks = {
                    "fractional_margin": metric["fractional_margin_over_chance"]
                    >= fractional_threshold,
                    "probability_margin": metric["probability_margin_over_chance"]
                    >= probability_threshold,
                    "log_loss": metric[
                        "log_loss_improvement_over_exact_chance"
                    ]
                    >= float(
                        config["gates"][
                            "minimum_log_loss_improvement_over_exact_chance"
                        ]
                    ),
                    "brier": metric["brier_improvement_over_exact_chance"]
                    >= float(
                        config["gates"][
                            "minimum_brier_improvement_over_exact_chance"
                        ]
                    ),
                    "tie_rate": metric["tie_frequency"]
                    <= float(config["gates"]["maximum_tie_frequency"]),
                }
                separated_rows.append(
                    {
                        "corpus_seed": int(result["corpus_seed"]),
                        "model_seed": int(result["model_seed"]),
                        "condition": str(result["condition"]),
                        "kind": kind,
                        "presence": presence,
                        "status": "PASS" if all(checks.values()) else "FAIL",
                        "checks": checks,
                        "metrics": metric,
                    }
                )
    present_pass = all(
        row["status"] == "PASS"
        for row in separated_rows
        if row["presence"] == "present"
    )
    null_pass = all(
        row["status"] == "PASS"
        for row in separated_rows
        if row["presence"] == "null"
    )
    sensor_free_present_pass = all(
        row["status"] == "PASS"
        for row in separated_rows
        if row["condition"] == "absent_channel"
        and row["presence"] == "present"
    )
    sensor_free_null_pass = all(
        row["status"] == "PASS"
        for row in separated_rows
        if row["condition"] == "absent_channel" and row["presence"] == "null"
    )
    separated_audit = {
        "status": "PASS" if present_pass and null_pass else "FAIL",
        "aggregation_policy": "every_corpus_model_condition_kind_presence_cell_must_pass",
        "present_status": "PASS" if present_pass else "FAIL",
        "null_status": "PASS" if null_pass else "FAIL",
        "sensor_free_present_status": (
            "PASS" if sensor_free_present_pass else "FAIL"
        ),
        "sensor_free_null_status": "PASS" if sensor_free_null_pass else "FAIL",
        "rows": separated_rows,
    }

    def action_probability(condition: str, presence: str) -> float:
        values = [
            row["metrics"]["by_kind_presence"]["action"][presence][
                "mean_correct_probability"
            ]
            for row in model_results
            if row["condition"] == condition
        ]
        return float(np.mean(values))

    synchronized_present_probability = action_probability("synchronized", "present")
    synchronized_null_probability = action_probability("synchronized", "null")
    comparison_present_probability = float(
        np.mean(
            [
                action_probability(condition, "present")
                for condition in (
                    "randomized_shuffle",
                    "uninformative",
                    "absent_channel",
                )
            ]
        )
    )
    comparison_null_probability = float(
        np.mean(
            [
                action_probability(condition, "null")
                for condition in (
                    "randomized_shuffle",
                    "uninformative",
                    "absent_channel",
                )
            ]
        )
    )

    control_by_key = {
        (int(row["corpus_seed"]), str(row["control"])): row
        for row in control_results
    }
    primary_by_key = {
        (int(row["corpus_seed"]), int(row["model_seed"]), str(row["condition"])): row
        for row in model_results
    }
    null_capacity_rows = []
    for corpus_seed_value in sorted(
        {int(row["corpus_seed"]) for row in control_results}
    ):
        first_model_seed = int(config["resolved_registries"][purpose]["model"][0])
        baseline = primary_by_key[
            (corpus_seed_value, first_model_seed, "absent_channel")
        ]
        ablated = control_by_key[
            (corpus_seed_value, "target_absent_training_removed")
        ]
        direct = control_by_key[(corpus_seed_value, "direct_capacity_upper")]
        corrupted = control_by_key[
            (corpus_seed_value, "corrupted_null_supervision")
        ]
        for kind in ("primitive", "manner", "action", "noun"):
            baseline_metric = baseline["metrics"]["by_kind_presence"][kind]["null"]
            ablated_metric = ablated["metrics"]["by_kind_presence"][kind]["null"]
            direct_metric = direct["metrics"]["by_kind_presence"][kind]["null"]
            corrupted_metric = corrupted["metrics"]["by_kind_presence"][kind]["null"]
            ablation_drop = (
                baseline_metric["fractional_accuracy"]
                - ablated_metric["fractional_accuracy"]
            )
            corruption_drop = (
                direct_metric["fractional_accuracy"]
                - corrupted_metric["fractional_accuracy"]
            )
            checks = {
                "target_absent_training_removal_reduces_null": ablation_drop
                >= float(
                    config["gates"]["minimum_null_ablation_fractional_drop"]
                ),
                "corrupting_null_supervision_reduces_null": corruption_drop
                >= float(
                    config["gates"]["minimum_null_corruption_fractional_drop"]
                ),
            }
            null_capacity_rows.append(
                {
                    "corpus_seed": corpus_seed_value,
                    "model_seed": first_model_seed,
                    "kind": kind,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                    "checks": checks,
                    "baseline_sensor_free_null": baseline_metric,
                    "target_absent_training_removed_null": ablated_metric,
                    "direct_capacity_null": direct_metric,
                    "corrupted_null_supervision_null": corrupted_metric,
                    "ablation_fractional_drop": ablation_drop,
                    "corruption_fractional_drop": corruption_drop,
                }
            )
    null_capacity_audit = {
        "status": (
            "PASS"
            if null_capacity_rows
            and all(row["status"] == "PASS" for row in null_capacity_rows)
            else "FAIL"
        ),
        "rows": null_capacity_rows,
        "primary_fit_uses_oracle_for_null_supervision": False,
        "oracle_use_confined_to_declared_negative_diagnostics": True,
    }
    control_fractionals: defaultdict[str, list[float]] = defaultdict(list)
    for row in control_results:
        if row["control"] not in {
            "oracle_alignment_upper",
            "direct_capacity_upper",
        }:
            continue
        for kind in ("primitive", "manner", "action", "noun"):
            for presence in ("present", "null"):
                control_fractionals[row["control"]].append(
                    row["metrics"]["by_kind_presence"][kind][presence][
                        "fractional_accuracy"
                    ]
                )
    sensor_free_identified = (
        sensor_free_present_pass
        and sensor_free_null_pass
        and all(
            not row["training_trace"]["post_fit_projection_applied"]
            and not row["training_trace"]["condition_specific_parameter_replacement"]
            for row in model_results
            if row["condition"] == "absent_channel"
        )
    )
    core_gates = {
        "all_conditions_executed": condition_names == set(config["design"]["conditions"]),
        "order_invariance": order_audit["status"] == "PASS",
        "present_learnability_separated": present_pass,
        "null_learnability_separated": null_pass,
        "identifiability_sensor_free_genuine_fit": sensor_free_identified,
        "null_capacity_controls": null_capacity_audit["status"] == "PASS",
        "sensor_helpful_not_required": synchronized_present_probability
        - comparison_present_probability
        >= float(config["gates"]["minimum_present_sensor_probability_delta"])
        and comparison_null_probability
        - synchronized_null_probability
        <= float(config["gates"]["maximum_null_sensor_probability_harm"]),
        "joint_iterations_and_competition_numeric": numeric_audit["status"] == "PASS",
        "noun_sensor_weight_zero": all(
            row["training_trace"]["noun_sensor_weight"] == 0.0
            and row["training_trace"]["sensor_applications_by_slot"]["noun"]
            <= int(config["gates"]["maximum_noun_sensor_application_count"])
            for row in model_results
        ),
        "factor_refits_executed": factor_audit["status"] == "PASS",
        "oracle_control_executed": bool(control_fractionals["oracle_alignment_upper"])
        and min(control_fractionals["oracle_alignment_upper"])
        >= float(config["gates"]["minimum_oracle_fractional"]),
        "direct_capacity_control_executed": bool(control_fractionals["direct_capacity_upper"])
        and min(control_fractionals["direct_capacity_upper"])
        >= float(config["gates"]["minimum_direct_capacity_fractional"]),
        "sensor_corruption_control_executed": "sensor_corrupted" in condition_names,
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
        "oracle_ledger_loaded_only_by_declared_controls_and_ablation_diagnostics": True,
        "target_absent_ablation_is_not_primary_or_decision_substitution": True,
        "final_language_prompts_have_side_modalities": False,
    }
    write_json(output / "model_results.json", model_results)
    write_json(output / "factor_results.json", factor_results)
    write_json(output / "control_results.json", control_results)
    write_json(output / "learner_numeric_audit.json", numeric_audit)
    write_json(output / "order_invariance_audit.json", order_audit)
    write_json(output / "factor_audit.json", factor_audit)
    write_json(output / "separated_learnability_audit.json", separated_audit)
    write_json(output / "null_capacity_audit.json", null_capacity_audit)
    write_json(output / "leakage_audit.json", leakage_audit)
    write_json(
        output / "core_gates.json",
        {
            "status": "PASS" if all(core_gates.values()) else "FAIL",
            "gates": core_gates,
            "identifiability_structural_failure": False,
            "metrics": {
                "synchronized_action_present_probability": synchronized_present_probability,
                "comparison_action_present_probability": comparison_present_probability,
                "present_sensor_probability_delta": synchronized_present_probability
                - comparison_present_probability,
                "synchronized_action_null_probability": synchronized_null_probability,
                "comparison_action_null_probability": comparison_null_probability,
                "null_sensor_probability_difference": synchronized_null_probability
                - comparison_null_probability,
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


def execute_qualification(
    repository_root: str | Path,
    config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    purpose: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(run_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if purpose not in {"fixture_only", "excluded_smoke"}:
        raise PermissionError("v6 may only execute fixture or the one excluded smoke")
    output.mkdir(parents=True)
    inputs = output / "persisted_inputs"
    write_json(inputs / "config.json", config)
    write_json(
        inputs / "qualification_metadata.json",
        {
            "protocol_id": PROTOCOL_ID,
            "purpose": purpose,
            "corpus_seeds": list(map(int, config["resolved_registries"][purpose]["corpus"])),
            "model_seeds": list(map(int, config["resolved_registries"][purpose]["model"])),
            "scientific_outcome": False,
        },
    )
    firewall = IdentifierFirewall(config, allowed_purposes=[purpose])
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
    compute_operation_log = json.loads(
        (output / "recomputable" / "compute_operation_log.json").read_text()
    )
    all_operations = [*firewall.log, *compute_operation_log["operations"]]
    observed_references = {
        role: sorted(
            {
                int(reference["value"])
                for operation in all_operations
                for reference in operation["references"]
                if reference["role"] == role
            }
        )
        for role in ("corpus", "model", "inference")
    }
    declared_references = {
        role: sorted(map(int, config["resolved_registries"][purpose][role]))
        for role in ("corpus", "model", "inference")
    }
    identifier_audit = {
        "status": (
            "PASS" if observed_references == declared_references else "FAIL"
        ),
        "purpose": purpose,
        "declared_references": declared_references,
        "observed_guarded_references": observed_references,
        "declared_equals_observed_exactly": observed_references
        == declared_references,
        "unused_declared_references": {
            role: sorted(
                set(declared_references[role]) - set(observed_references[role])
            )
            for role in ("corpus", "model", "inference")
        },
        "undeclared_observed_references": {
            role: sorted(
                set(observed_references[role]) - set(declared_references[role])
            )
            for role in ("corpus", "model", "inference")
        },
        "operation_count": len(all_operations),
    }
    runtime_audit = runtime.runtime_audit(minimum_calls=expected_calls, perturbation=perturbation)
    capacity_audit = detector_capacity_audit(capacity_evidence, capacity_oracle, config)
    top_gates = {
        **core["gates"],
        "corpus_design": all(row["status"] == "PASS" for row in corpus_audits),
        "condition_design": all(row["status"] == "PASS" for row in condition_audits),
        "actual_frozen_v2_detector_runtime": runtime_audit["status"] == "PASS",
        "detector_capacity_predeclared_bounds": capacity_audit["status"] == "PASS",
        "seed_firewall": all(
            row["status"] == "ALLOW" for row in all_operations
        ),
        "registry_touch_reconciliation": identifier_audit["status"] == "PASS",
        "future_development_outcome_count_zero": True,
        "confirmation_outcome_count_zero": True,
        "scientific_outcome_count_zero": True,
    }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "purpose": purpose,
        "status": "PASS" if all(top_gates.values()) else "FAIL",
        "gates": top_gates,
        "identifiability_structural_failure": json.loads(
            (output / "recomputable" / "core_gates.json").read_text()
        )["identifiability_structural_failure"],
        "package_readiness_only": True,
        "scientific_outcomes": 0,
        "future_development_outcomes": 0,
        "confirmation_outcomes": 0,
        "corpus_seeds_touched": observed_references["corpus"],
        "model_seeds_touched": observed_references["model"],
        "inference_identifiers_touched": observed_references["inference"],
    }
    write_json(output / "corpus_audits.json", corpus_audits)
    write_json(output / "condition_audits.json", condition_audits)
    write_json(output / "detector_runtime_audit.json", runtime_audit)
    write_json(output / "detector_capacity_audit.json", capacity_audit)
    write_json(output / "identifier_reference_audit.json", identifier_audit)
    write_json(
        output / "seed_operation_log.json",
        {
            "status": (
                "PASS"
                if top_gates["seed_firewall"]
                and top_gates["registry_touch_reconciliation"]
                else "FAIL"
            ),
            "operations": all_operations,
            "fixture_identifiers_touched": observed_references
            if purpose == "fixture_only"
            else {"corpus": [], "model": [], "inference": []},
            "excluded_smoke_identifiers_touched": observed_references
            if purpose == "excluded_smoke"
            else {"corpus": [], "model": [], "inference": []},
            "retired_pre_freeze_identifiers_touched": {
                "corpus": [],
                "model": [],
                "inference": [],
            },
            "future_development_identifiers_touched": {
                "corpus": [],
                "model": [],
                "inference": [],
            },
            "confirmation_identifiers_touched": {
                "corpus": [],
                "model": [],
                "inference": [],
            },
            "prior_v1_v5_identifiers_touched": {
                "corpus": [],
                "model": [],
                "inference": [],
            },
        },
    )
    write_json(output / "qualification_summary.json", summary)
    return summary
