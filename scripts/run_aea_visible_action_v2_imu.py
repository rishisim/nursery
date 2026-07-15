#!/usr/bin/env python3
"""Conditional group-held-out IMU viability diagnostic for AEA v2."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.visible_action_v2 import (  # noqa: E402
    apply_protocol_amendment,
    load_json,
    sha256_file,
    validate_frozen_development_inputs,
)
from scripts.run_aea_dev_imu_diagnostic import extract_imu_features  # noqa: E402


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v2 IMU artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def balanced_accuracy(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None
) -> float:
    resolved_labels = list(labels) if labels is not None else sorted(set(map(str, y_true)))
    per_label = [
        np.mean([
            str(prediction) == label
            for truth, prediction in zip(y_true, y_pred) if str(truth) == label
        ]) for label in resolved_labels if any(str(truth) == label for truth in y_true)
    ]
    return float(np.mean(per_label))


def _per_class(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    return {
        label: float(np.mean([
            str(prediction) == label
            for truth, prediction in zip(y_true, y_pred) if str(truth) == label
        ])) for label in sorted(set(map(str, y_true)))
    }


def _cluster_bootstrap(
    records: Sequence[Mapping[str, Any]],
    samples: int,
    seed: int,
    difference: bool = False,
) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[str(row["event_group"])].append(row)
    groups = sorted(by_group)
    fixed_labels = sorted({str(row["true_label"]) for row in records})
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        selected = [row for group in sampled for row in by_group[str(group)]]
        synchronized = balanced_accuracy(
            [str(row["true_label"]) for row in selected],
            [str(row["synchronized_prediction"]) for row in selected],
            fixed_labels,
        )
        if difference:
            donor_values = []
            donor_keys = sorted(selected[0]["donor_predictions"])
            for key in donor_keys:
                donor_values.append(balanced_accuracy(
                    [str(row["true_label"]) for row in selected],
                    [str(row["donor_predictions"][key]) for row in selected],
                    fixed_labels,
                ))
            values.append(synchronized - float(np.mean(donor_values)))
        else:
            values.append(synchronized)
    return {
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "samples": int(samples),
        "seed": int(seed),
        "unit": "event_group",
    }


def _chance_permutation(
    records: Sequence[Mapping[str, Any]], samples: int, seed: int, observed: float
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_fold: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_fold[int(row["fold"])].append(index)
    true = np.asarray([str(row["true_label"]) for row in records], dtype=object)
    predicted = [str(row["synchronized_prediction"]) for row in records]
    null = []
    for _ in range(samples):
        permuted = true.copy()
        for indices in by_fold.values():
            permuted[indices] = rng.permutation(permuted[indices])
        null.append(balanced_accuracy(permuted.tolist(), predicted))
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null)),
        "p_value_plus_one": float((1 + sum(value >= observed for value in null)) / (samples + 1)),
        "samples": int(samples),
        "seed": int(seed),
        "method": "within-fold label permutation of saved held-out predictions",
    }


def _paired_randomization(
    records: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    donor_keys = sorted(records[0]["donor_predictions"])
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[str(row["event_group"])].append(row)
    differences = []
    for group in sorted(by_group):
        rows = by_group[group]
        sync = np.mean([
            str(row["synchronized_prediction"]) == str(row["true_label"])
            for row in rows
        ])
        donor = np.mean([
            np.mean([
                str(row["donor_predictions"][key]) == str(row["true_label"])
                for row in rows
            ])
            for key in donor_keys
        ])
        differences.append(sync - float(donor))
    rng = np.random.default_rng(seed)
    null = []
    differences_array = np.asarray(differences)
    for _ in range(samples):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(differences_array))
        null.append(float(np.mean(signs * differences_array)))
    observed = float(np.mean(differences_array))
    return {
        "observed_group_mean_accuracy_difference": observed,
        "p_value_plus_one": float((1 + sum(value >= observed for value in null)) / (samples + 1)),
        "samples": int(samples),
        "seed": int(seed),
        "event_groups": len(differences),
        "method": "paired event-group sign-flip randomization",
    }


def run(
    protocol_path: Path,
    amendment_path: Path,
    agreement_path: Path,
    split_path: Path,
    capacity_path: Path,
    development_path: Path,
    partition_path: Path,
    prior_prelabel_path: Path,
    data_root: Path,
    output: Path,
) -> dict[str, Any]:
    base = load_json(protocol_path)
    amendment = load_json(amendment_path)
    if amendment.get("parent_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("protocol amendment parent hash mismatch")
    protocol = apply_protocol_amendment(base, amendment)
    agreement = load_json(agreement_path)
    split = load_json(split_path)
    capacity = load_json(capacity_path)
    gates = {
        "annotation_gate_passed": agreement.get("annotation_gate_passed") is True,
        "split_and_donor_gate_passed": split.get("split_gate_passed") is True,
        "action_head_capacity_passed": capacity.get("capacity_gate_passed") is True,
    }
    preflight = {
        "schema_version": "aea-visible-action-imu-preflight-v2",
        "protocol_id": protocol["protocol_id"],
        "gate_inputs": gates,
        "authorized_by_stage_gates": all(gates.values()),
        "written_before_imu_access": True,
        "development_imu_arrays_opened_at_write": 0,
        "reserve_imu_arrays_opened_at_write": 0,
        "reserve_rgb_files_opened_at_write": 0,
    }
    _write_json_new(output / "imu_preflight_receipt.json", preflight)
    if not all(gates.values()):
        result = {
            "schema_version": "aea-visible-action-imu-v2",
            "protocol_id": protocol["protocol_id"],
            "scientific_role": protocol["scientific_role"],
            "status": "not_run_stage_gate_failed",
            "gate_inputs": gates,
            "imu_gate_passed": False,
            "development_imu_arrays_opened": 0,
            "reserve_imu_arrays_opened": 0,
            "expensive_modeling_stopped": True,
        }
        _write_json_new(output / "imu_results.json", result)
        return result

    development_rows, _prior, source_checks = validate_frozen_development_inputs(
        development_path, partition_path, prior_prelabel_path, protocol
    )
    by_example = {str(row["example_id"]): row for row in development_rows}
    endpoint_rows = list(split["endpoint_row_manifest"])
    if any(str(row["example_id"]) not in by_example for row in endpoint_rows):
        raise ValueError("IMU endpoint contains a non-development example")
    reserve_groups = set(map(str, protocol["sources"]["reserve_event_groups"]))
    if any(str(row["event_group"]) in reserve_groups for row in endpoint_rows):
        raise ValueError("reserve group entered conditional IMU endpoint")

    features = []
    opened = 0
    data_root_resolved = data_root.resolve()
    for endpoint in endpoint_rows:
        source = by_example[str(endpoint["example_id"])]
        path = (data_root / source["model_inputs"]["imu_path"]).resolve()
        if data_root_resolved not in path.parents:
            raise ValueError("development IMU path escaped data root")
        values = np.load(path, allow_pickle=False)
        opened += 1
        features.append(extract_imu_features(
            values, float(source["model_inputs"]["imu_rate_hz"])
        ))
    matrix = np.stack(features)
    if matrix.shape[1] != 129 or not np.isfinite(matrix).all():
        raise ValueError(f"frozen IMU feature contract failed: {matrix.shape}")
    labels = np.asarray([str(row["observable_action"]) for row in endpoint_rows], dtype=object)
    predictions: list[dict[str, Any] | None] = [None] * len(endpoint_rows)
    fold_diagnostics = []
    train_accuracies = []
    for fold in split["folds"]:
        train = np.asarray(fold["train_indices"], dtype=int)
        test = np.asarray(fold["test_indices"], dtype=int)
        scaler = StandardScaler().fit(matrix[train])
        train_x = scaler.transform(matrix[train])
        test_x = scaler.transform(matrix[test])
        classifier = LogisticRegression(
            C=float(protocol["imu"]["C"]),
            penalty="l2",
            class_weight="balanced",
            max_iter=int(protocol["imu"]["max_iter"]),
            solver="lbfgs",
        ).fit(train_x, labels[train])
        train_prediction = classifier.predict(train_x)
        sync_prediction = classifier.predict(test_x)
        train_accuracy = balanced_accuracy(labels[train].tolist(), train_prediction.tolist())
        train_accuracies.append(train_accuracy)
        donor_predictions: dict[str, list[str]] = {}
        for seed, donor in fold["imu_diagnostic_test_donors"].items():
            if not donor["feasible"]:
                raise ValueError("frozen test donor map is infeasible after stage gate")
            id_to_index = {str(row["example_id"]): index for index, row in enumerate(endpoint_rows)}
            donor_indices = np.asarray([
                id_to_index[donor["donor_map"][str(endpoint_rows[index]["example_id"])]]
                for index in test
            ], dtype=int)
            donor_predictions[str(seed)] = classifier.predict(
                scaler.transform(matrix[donor_indices])
            ).tolist()
        for position, index in enumerate(test):
            predictions[int(index)] = {
                "example_id": str(endpoint_rows[int(index)]["example_id"]),
                "event_group": str(endpoint_rows[int(index)]["event_group"]),
                "fold": int(fold["fold"]),
                "true_label": str(labels[int(index)]),
                "synchronized_prediction": str(sync_prediction[position]),
                "donor_predictions": {
                    seed: str(values[position]) for seed, values in donor_predictions.items()
                },
            }
        fold_diagnostics.append({
            "fold": int(fold["fold"]),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_balanced_accuracy": train_accuracy,
            "synchronized_test_balanced_accuracy": balanced_accuracy(
                labels[test].tolist(), sync_prediction.tolist()
            ),
            "train_event_groups": fold["train_event_groups"],
            "test_event_groups": fold["test_event_groups"],
            "standardizer_fit_rows": train.tolist(),
            "classifier_iterations": classifier.n_iter_.tolist(),
        })
    if any(row is None for row in predictions):
        raise AssertionError("every endpoint row must receive one held-out prediction")
    prediction_rows = [dict(row) for row in predictions if row is not None]
    truth = [str(row["true_label"]) for row in prediction_rows]
    sync = [str(row["synchronized_prediction"]) for row in prediction_rows]
    synchronized_accuracy = balanced_accuracy(truth, sync)
    donor_seeds = sorted(prediction_rows[0]["donor_predictions"])
    donor_accuracies = {
        seed: balanced_accuracy(
            truth, [str(row["donor_predictions"][seed]) for row in prediction_rows]
        ) for seed in donor_seeds
    }
    donor_mean = float(np.mean(list(donor_accuracies.values())))
    lift = synchronized_accuracy - donor_mean
    train_mean = float(np.mean(train_accuracies))
    gap = train_mean - synchronized_accuracy
    chance = 1.0 / len(split["retained_labels"])
    imu_config = protocol["imu"]
    bootstrap = _cluster_bootstrap(
        prediction_rows, int(imu_config["bootstrap_samples"]),
        int(imu_config["bootstrap_seed"]), difference=False,
    )
    paired_bootstrap = _cluster_bootstrap(
        prediction_rows, int(imu_config["bootstrap_samples"]),
        int(imu_config["bootstrap_seed"]), difference=True,
    )
    permutation = _chance_permutation(
        prediction_rows, int(imu_config["chance_permutations"]),
        int(imu_config["chance_permutation_seed"]), synchronized_accuracy,
    )
    paired_randomization = _paired_randomization(
        prediction_rows, int(imu_config["paired_randomizations"]),
        int(imu_config["paired_randomization_seed"]),
    )
    checks = {
        "synchronized_accuracy_above_chance_margin": synchronized_accuracy
        >= chance + float(imu_config["minimum_above_macro_chance"]),
        "bootstrap_lower_above_chance": bootstrap["ci95_low"] > chance,
        "chance_permutation_p": permutation["p_value_plus_one"]
        <= float(imu_config["chance_permutation_p_maximum"]),
        "synchronized_minus_donor_minimum": lift
        >= float(imu_config["minimum_synchronized_minus_donor"]),
        "paired_bootstrap_lower_above_zero": paired_bootstrap["ci95_low"] > 0,
        "paired_randomization_p": paired_randomization["p_value_plus_one"]
        <= float(imu_config["paired_randomization_p_maximum"]),
        "train_minus_heldout_gap": gap
        <= float(imu_config["maximum_train_minus_heldout_gap"]),
    }
    result = {
        "schema_version": "aea-visible-action-imu-v2",
        "protocol_id": protocol["protocol_id"],
        "scientific_role": protocol["scientific_role"],
        "status": "complete",
        "label_role": "model_assisted_development_labels_only",
        "configuration": dict(imu_config),
        "labels": list(split["retained_labels"]),
        "rows": len(endpoint_rows),
        "event_groups": len({str(row["event_group"]) for row in endpoint_rows}),
        "macro_chance": chance,
        "mean_train_balanced_accuracy": train_mean,
        "heldout_synchronized_balanced_accuracy": synchronized_accuracy,
        "heldout_synchronized_by_action": _per_class(truth, sync),
        "train_minus_heldout_gap": gap,
        "donor_balanced_accuracy_by_seed": donor_accuracies,
        "mean_donor_balanced_accuracy": donor_mean,
        "synchronized_minus_donor": lift,
        "synchronized_bootstrap": bootstrap,
        "synchronized_chance_permutation": permutation,
        "paired_lift_bootstrap": paired_bootstrap,
        "paired_randomization": paired_randomization,
        "fold_diagnostics": fold_diagnostics,
        "heldout_predictions": prediction_rows,
        "gate_checks": checks,
        "imu_gate_passed": all(checks.values()),
        "source_integrity": source_checks,
        "feature_dimension": int(matrix.shape[1]),
        "all_features_finite": bool(np.isfinite(matrix).all()),
        "development_imu_arrays_opened": opened,
        "reserve_imu_arrays_opened": 0,
        "reserve_rgb_files_opened": 0,
    }
    _write_json_new(output / "imu_results.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_visible_action_v2")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--amendment", type=Path, default=root / "preregistered_protocol_amendment_1.json")
    parser.add_argument("--agreement", type=Path, default=root / "agreement_report.json")
    parser.add_argument("--split", type=Path, default=root / "split_donor_feasibility.json")
    parser.add_argument("--capacity", type=Path, default=root / "capacity_results.json")
    parser.add_argument("--development", type=Path, default=Path("output/aea_dev_learnability_v1/development_examples.jsonl"))
    parser.add_argument("--partition", type=Path, default=Path("output/aea_dev_learnability_v1/partition_manifest.json"))
    parser.add_argument("--prior-prelabel", type=Path, default=Path("output/aea_dev_learnability_v1/audit_manifest_prelabel.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/aea_processed"))
    parser.add_argument("--out", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.protocol, args.amendment, args.agreement, args.split, args.capacity,
        args.development, args.partition, args.prior_prelabel, args.data_root, args.out,
    )
    print(json.dumps({
        "status": result["status"],
        "imu_gate_passed": result.get("imu_gate_passed", False),
        "development_imu_arrays_opened": result["development_imu_arrays_opened"],
        "reserve_imu_arrays_opened": result["reserve_imu_arrays_opened"],
    }, indent=2))


if __name__ == "__main__":
    main()
