from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from babyworld_lite.aea.report import write_aea_report
from babyworld_lite.aea.splits import (
    AEASplit,
    held_out_composition_splits,
    held_out_location_splits,
    held_out_wearer_session_splits,
)
from babyworld_lite.grounding.pilot_data import (
    AEAWindowAdapter,
    GroundingCorpus,
    WordTokenizer,
    load_jsonl,
)
from babyworld_lite.grounding.pilot_experiment import (
    PilotConfig,
    evaluate_without_motor,
    metadata_only_shortcut_check,
    motor_only_manipulation_check,
    resolve_device,
    train_one_arm,
)


def paired_hierarchical_ci(
    rows: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    """Bootstrap split units, then paired seeds within sampled units."""
    units = sorted({str(row["split"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    values = {(str(row["split"]), int(row["seed"])): float(row["value"]) for row in rows}
    complete_units = [unit for unit in units if all((unit, item) in values for item in seeds)]
    if not complete_units or not seeds:
        raise ValueError("paired hierarchical CI needs complete split x seed cells")
    matrix = np.asarray([[values[(unit, item)] for item in seeds] for unit in complete_units])
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_units = rng.integers(0, len(complete_units), size=len(complete_units))
        unit_means = []
        for unit in sampled_units:
            sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
            unit_means.append(matrix[unit, sampled_seeds].mean())
        boot[index] = np.mean(unit_means)
    return {
        "mean": float(matrix.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "n_split_units": len(complete_units),
        "n_paired_seeds": len(seeds),
        "method": "hierarchical bootstrap over split units then paired seeds",
    }


def _label_recovery_control(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> dict[str, Any]:
    recovered = []
    for index in indices:
        row = records[index]
        surface = str(row["annotation_provenance"]["action_surface"])
        transcript = str(row["model_inputs"]["transcript"]).lower()
        recovered.append(surface in transcript)
    return {
        "action_anchor_present_in_own_transcript_rate": float(np.mean(recovered)) if recovered else None,
        "expected": 1.0,
        "role": "annotation/timestamp integrity control, not evidence of visual grounding",
    }


def locked_training_config_matches(
    pilot_config: PilotConfig, protocol: Mapping[str, Any]
) -> bool:
    locked = protocol.get("locked_training", {})
    observed = asdict(pilot_config)
    return bool(locked) and all(observed.get(key) == value for key, value in locked.items())


def _run_split(
    records: Sequence[Mapping[str, Any]],
    corpus: GroundingCorpus,
    split: AEASplit,
    seeds: Sequence[int],
    arms: Sequence[str],
    pilot_config: PilotConfig,
    device,
) -> dict[str, Any]:
    tokenizer = WordTokenizer.fit([corpus.text(index) for index in split.train_indices])
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            model, protocol, dataset = train_one_arm(
                corpus, split.train_indices, tokenizer, arm, seed, pilot_config, device
            )
            metrics = evaluate_without_motor(
                model, corpus, split.test_indices, tokenizer,
                split.supported_actions, pilot_config, device,
            )
            manipulation = motor_only_manipulation_check(
                corpus, split.train_indices, split.test_indices, tokenizer,
                arm, split.supported_actions, pilot_config, seed,
            )
            donor_group_matches = (
                sum(
                    corpus.metadata(index).episode_group == corpus.metadata(dataset.donors[index]).episode_group
                    for index in dataset.indices
                ) / len(dataset.indices)
            )
            history = protocol["training_history"]
            runs.append({
                "seed": int(seed),
                "arm": arm,
                "metrics": metrics,
                "training_protocol": protocol,
                "motor_only_manipulation": manipulation,
                "donor_episode_group_match_rate": donor_group_matches if arm == "shuffled" else None,
                "optimization_positive_control": {
                    "loss_first_epoch": history[0]["loss"],
                    "loss_last_epoch": history[-1]["loss"],
                    "loss_nonincreasing": history[-1]["loss"] <= history[0]["loss"],
                },
            })
    sync = {
        run["seed"]: run["metrics"]["action_2afc_macro_accuracy"]
        for run in runs if run["arm"] == "synchronized"
    }
    shuffled = {
        run["seed"]: run["metrics"]["action_2afc_macro_accuracy"]
        for run in runs if run["arm"] == "shuffled"
    }
    null = {
        run["seed"]: run["metrics"]["action_2afc_macro_accuracy"]
        for run in runs if run["arm"] == "null"
    }
    pairing = {}
    for seed in seeds:
        seed_runs = [run for run in runs if run["seed"] == seed]
        pairing[str(seed)] = {
            "same_initialization_across_arms": len({
                run["training_protocol"]["initialization_digest"] for run in seed_runs
            }) == 1,
            "same_optimizer_steps_across_arms": len({
                run["training_protocol"]["optimizer_steps"] for run in seed_runs
            }) == 1,
            "same_training_order_across_arms": len({
                run["training_protocol"]["training_order_digest"] for run in seed_runs
            }) == 1,
        }
    return {
        "name": split.name,
        "family": split.family,
        "split_audit": split.audit,
        "actions": list(split.supported_actions),
        "metadata_only_shortcut": metadata_only_shortcut_check(
            corpus, split.train_indices, split.test_indices, split.supported_actions
        ),
        "label_recovery_positive_control": _label_recovery_control(records, split.test_indices),
        "paired_protocol_checks": pairing,
        "runs": runs,
        "paired_seed_differences": {
            "synchronized_minus_shuffled": {str(seed): sync[seed] - shuffled[seed] for seed in seeds},
            "synchronized_minus_null": {str(seed): sync[seed] - null[seed] for seed in seeds},
        },
    }


def run_aea_experiment(
    examples_path: str | Path,
    experiment_config: Mapping[str, Any],
    out_dir: str | Path,
    pilot_config: PilotConfig,
    seeds: Sequence[int] | None = None,
    families: Sequence[str] = (
        "held_out_location", "held_out_wearer_session", "held_out_composition"
    ),
    device_name: str = "auto",
    infrastructure_smoke: bool = False,
    maximum_splits: int = 0,
) -> dict[str, Any]:
    records = load_jsonl(examples_path)
    if not records or records[0].get("schema_version") != "aea-grounding-v1":
        raise ValueError("AEA experiment requires aea-grounding-v1 examples")
    adapter = AEAWindowAdapter(Path(examples_path).parent)
    corpus = GroundingCorpus(
        records, adapter, pilot_config.frame_count, pilot_config.image_size,
        pilot_config.motor_sample_count,
    )
    protocol = experiment_config["experiment"]
    resolved_seeds = list(seeds or map(int, protocol["seeds"]))
    if len(resolved_seeds) < 2:
        raise ValueError("paired confidence intervals require multiple seeds")
    arms = list(protocol["arms"])
    minimum = int(protocol["minimum_test_examples_per_action"])
    splits: list[AEASplit] = []
    if "held_out_location" in families:
        splits.extend(held_out_location_splits(
            records, protocol["held_out_location_folds"], minimum
        ))
    if "held_out_wearer_session" in families:
        splits.extend(held_out_wearer_session_splits(records, minimum))
    if "held_out_composition" in families:
        splits.extend(held_out_composition_splits(
            records,
            [(str(row["action"]), str(row["object"])) for row in protocol["held_out_compositions"]],
            minimum,
        ))
    if maximum_splits:
        splits = splits[:maximum_splits]
    if not splits:
        raise ValueError("no leakage-safe AEA evaluation split has enough support")
    device = resolve_device(device_name)
    results = [
        _run_split(records, corpus, split, resolved_seeds, arms, pilot_config, device)
        for split in splits
    ]
    family_summaries: dict[str, Any] = {}
    for family in sorted({row["family"] for row in results}):
        family_rows = [row for row in results if row["family"] == family]
        differences = [
            {"split": row["name"], "seed": int(seed), "value": value}
            for row in family_rows
            for seed, value in row["paired_seed_differences"]["synchronized_minus_shuffled"].items()
        ]
        null_differences = [
            {"split": row["name"], "seed": int(seed), "value": value}
            for row in family_rows
            for seed, value in row["paired_seed_differences"]["synchronized_minus_null"].items()
        ]
        family_summaries[family] = {
            "split_count": len(family_rows),
            "paired_lift": paired_hierarchical_ci(
                differences, int(protocol["bootstrap_samples"]), 9817
            ),
            "synchronized_minus_null": paired_hierarchical_ci(
                null_differences, int(protocol["bootstrap_samples"]), 5197
            ),
        }
    primary_family = family_summaries.get("held_out_location")
    primary_interval = primary_family["paired_lift"] if primary_family else None
    all_audits_valid = all(row["split_audit"]["valid"] for row in results)
    all_pairing_valid = all(
        all(checks.values())
        for row in results for checks in row["paired_protocol_checks"].values()
    )
    label_control_valid = all(
        row["label_recovery_positive_control"]["action_anchor_present_in_own_transcript_rate"] == 1.0
        for row in results
    )
    shuffled_group_safe = all(
        run["donor_episode_group_match_rate"] == 0.0
        for row in results for run in row["runs"] if run["arm"] == "shuffled"
    )
    threshold = float(protocol["effect_threshold_absolute_accuracy"])
    sync_minus_null = (
        primary_family["synchronized_minus_null"] if primary_family else None
    )
    locked_training_match = locked_training_config_matches(pilot_config, protocol)
    primary_complete = bool(
        primary_family
        and primary_family["split_count"] == len(protocol["held_out_location_folds"])
        and resolved_seeds == list(map(int, protocol["seeds"]))
        and locked_training_match
    )
    claim_gate = bool(
        not infrastructure_smoke
        and primary_complete
        and primary_interval
        and primary_interval["mean"] >= threshold
        and primary_interval["ci95_low"] > 0
        and sync_minus_null
        and sync_minus_null["mean"] > 0
        and all_audits_valid
        and all_pairing_valid
        and label_control_valid
        and shuffled_group_safe
    )
    result = {
        "schema_version": "aea-grounding-results-v1",
        "scientific_status": (
            "infrastructure_smoke_test_not_a_real_data_finding"
            if infrastructure_smoke else
            "real_aea_effect_estimate" if primary_complete else
            "incomplete_real_data_run_not_a_primary_finding"
        ),
        "scientific_role": experiment_config["profile"]["scientific_role"],
        "examples_path": str(examples_path),
        "data_summary": {
            "windows": len(records),
            "sequences": len({str(row["sequence_id"]) for row in records}),
            "locations": dict(Counter(str(row["location"]) for row in records)),
            "actions": dict(Counter(
                str(row["evaluation_targets"]["action_verb"]) for row in records
            )),
        },
        "device": str(device),
        "pilot_config": asdict(pilot_config),
        "paired_seeds": resolved_seeds,
        "arms": arms,
        "primary_test": {
            "family": "held_out_location",
            "metric": "action-balanced 2AFC macro accuracy",
            "motor": "withheld by omission; motor encoder is never called",
        },
        "evaluation_results": results,
        "family_summaries": family_summaries,
        "primary_estimand": {
            "name": "synchronized_minus_split_local_episode_shuffled",
            "interval": primary_interval,
            "synchronized_minus_null_interval": sync_minus_null,
            "minimum_effect_threshold": threshold,
            "complete_locked_location_folds_and_seed_schedule": primary_complete,
            "locked_training_configuration_match": locked_training_match,
            "claim_gate_passed": claim_gate,
            "audit_gates_passed": (
                all_audits_valid and all_pairing_valid
                and label_control_valid and shuffled_group_safe
            ),
        },
        "positive_controls": {
            "action_anchor_recovery_passed": label_control_valid,
            "paired_protocol_passed": all_pairing_valid,
            "shuffled_same_episode_group_rate_zero": shuffled_group_safe,
            "optimization_history_retained_per_run": True,
            "motor_only_manipulation_retained_per_run": True,
        },
        "limitations": {
            "adult_partly_scripted": True,
            "developmental_evidence": False,
            "action_labels": "ASR lexical anchors, not human action annotations",
            "wearer_identity": "release-visible session proxy, not persistent identity",
            "location_5": "contains scripts 4 and 5 only",
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aea_results.json").write_text(json.dumps(result, indent=2))
    write_aea_report(result, out / "aea_report.md")
    return result
