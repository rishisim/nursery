#!/usr/bin/env python3
"""Frozen conditional held-out coarse-action IMU diagnostic for AEA v3."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.coarse_action_v3 import (  # noqa: E402
    MODELED_ACTIONS,
    PROTOCOL_ID,
    load_json,
    load_jsonl,
    sha256_file,
    verify_protocol_freeze,
)
from scripts.run_aea_dev_imu_diagnostic import extract_imu_features  # noqa: E402
from scripts.run_aea_visible_action_v2_imu import (  # noqa: E402
    _chance_permutation,
    _cluster_bootstrap,
    _paired_randomization,
    _per_class,
    balanced_accuracy,
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v3 IMU artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, freeze = verify_protocol_freeze(
        args.protocol, args.freeze_receipt, args.preregistration, args.codebook
    )
    agreement = load_json(args.agreement)
    split = load_json(args.split)
    capacity = load_json(args.capacity)
    gates = {
        "coarse_annotation_gate_passed": agreement.get(
            "coarse_annotation_gate_passed"
        )
        is True,
        "split_and_donor_gate_passed": split.get("split_gate_passed") is True,
        "action_head_capacity_passed": capacity.get("capacity_gate_passed") is True,
    }
    preflight = {
        "schema_version": "aea-coarse-action-imu-preflight-v3",
        "protocol_id": PROTOCOL_ID,
        "gate_inputs": gates,
        "authorized_by_stage_gates": all(gates.values()),
        "written_before_imu_access": True,
        "development_imu_arrays_opened_at_write": 0,
        "reserve_imu_arrays_opened_at_write": 0,
        "reserve_rgb_files_opened_at_write": 0,
        "protocol_freeze_checks": freeze,
    }
    _write_json_new(args.out / "imu_preflight_receipt.json", preflight)
    if not all(gates.values()):
        result = {
            "schema_version": "aea-coarse-action-imu-v3",
            "protocol_id": PROTOCOL_ID,
            "scientific_role": protocol["scientific_role"],
            "status": "not_run_stage_gate_failed",
            "gate_inputs": gates,
            "imu_gate_passed": False,
            "development_imu_arrays_opened": 0,
            "reserve_imu_arrays_opened": 0,
            "reserve_rgb_files_opened": 0,
            "expensive_modeling_stopped": True,
        }
        _write_json_new(args.out / "imu_results.json", result)
        return result

    source = protocol["sources"]
    source_hash_checks = {
        "development_examples": sha256_file(args.development)
        == source["development_examples_sha256"],
        "partition_manifest": sha256_file(args.partition)
        == source["partition_manifest_sha256"],
    }
    if not all(source_hash_checks.values()):
        raise ValueError(f"v3 IMU frozen source hash mismatch: {source_hash_checks}")
    development_rows = load_jsonl(args.development)
    manifest = load_json(args.manifest)
    fixed_ids = {str(row["example_id"]) for row in manifest["rows"]}
    by_example = {str(row["example_id"]): row for row in development_rows}
    endpoint_rows = list(split["endpoint_row_manifest"])
    endpoint_ids = [str(row["example_id"]) for row in endpoint_rows]
    reserve_groups = set(map(str, source["reserve_event_groups"]))
    endpoint_checks = {
        "unique_endpoint_ids": len(endpoint_ids) == len(set(endpoint_ids)),
        "endpoint_is_fixed_sample_subset": set(endpoint_ids) <= fixed_ids,
        "endpoint_is_development_subset": set(endpoint_ids) <= set(by_example),
        "zero_reserve_groups": not any(
            str(row["event_group"]) in reserve_groups for row in endpoint_rows
        ),
        "exact_two_labels": set(split["retained_labels"]) == set(MODELED_ACTIONS),
    }
    if not all(endpoint_checks.values()):
        raise ValueError(f"v3 IMU endpoint validation failed: {endpoint_checks}")

    data_root = args.data_root.resolve()
    features = []
    opened_paths = []
    for endpoint in endpoint_rows:
        row = by_example[str(endpoint["example_id"])]
        path = (args.data_root / str(row["model_inputs"]["imu_path"])).resolve()
        if data_root not in path.parents:
            raise ValueError("development IMU path escaped processed-data root")
        values = np.load(path, allow_pickle=False)
        opened_paths.append(str(path.relative_to(data_root)))
        features.append(extract_imu_features(
            values, float(row["model_inputs"]["imu_rate_hz"])
        ))
    matrix = np.stack(features)
    if matrix.shape[1] != 129 or not np.isfinite(matrix).all():
        raise ValueError(f"frozen v3 IMU feature contract failed: {matrix.shape}")
    labels = np.asarray(
        [str(row["observable_action"]) for row in endpoint_rows], dtype=object
    )
    prediction_slots: list[dict[str, Any] | None] = [None] * len(endpoint_rows)
    fold_diagnostics = []
    train_accuracies = []
    config = protocol["imu"]
    id_to_index = {
        str(row["example_id"]): index for index, row in enumerate(endpoint_rows)
    }
    for fold in split["folds"]:
        train = np.asarray(fold["train_indices"], dtype=int)
        test = np.asarray(fold["test_indices"], dtype=int)
        scaler = StandardScaler().fit(matrix[train])
        train_x = scaler.transform(matrix[train])
        test_x = scaler.transform(matrix[test])
        classifier = LogisticRegression(
            C=float(config["C"]),
            penalty="l2",
            class_weight="balanced",
            max_iter=int(config["max_iter"]),
            solver="lbfgs",
        ).fit(train_x, labels[train])
        train_prediction = classifier.predict(train_x)
        sync_prediction = classifier.predict(test_x)
        train_accuracy = balanced_accuracy(
            labels[train].tolist(), train_prediction.tolist(), MODELED_ACTIONS
        )
        train_accuracies.append(train_accuracy)
        donor_predictions: dict[str, list[str]] = {}
        for seed, donor in fold["imu_diagnostic_test_donors"].items():
            if not donor["feasible"]:
                raise ValueError("frozen v3 test donor map became infeasible")
            donor_indices = np.asarray([
                id_to_index[
                    donor["donor_map"][str(endpoint_rows[index]["example_id"])]
                ]
                for index in test
            ], dtype=int)
            donor_predictions[str(seed)] = classifier.predict(
                scaler.transform(matrix[donor_indices])
            ).tolist()
        for position, index in enumerate(test):
            prediction_slots[int(index)] = {
                "example_id": str(endpoint_rows[int(index)]["example_id"]),
                "event_group": str(endpoint_rows[int(index)]["event_group"]),
                "fold": int(fold["fold"]),
                "true_label": str(labels[int(index)]),
                "synchronized_prediction": str(sync_prediction[position]),
                "donor_predictions": {
                    seed: str(values[position])
                    for seed, values in donor_predictions.items()
                },
            }
        fold_diagnostics.append({
            "fold": int(fold["fold"]),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_balanced_accuracy": train_accuracy,
            "synchronized_test_balanced_accuracy": balanced_accuracy(
                labels[test].tolist(), sync_prediction.tolist(), MODELED_ACTIONS
            ),
            "train_event_groups": fold["train_event_groups"],
            "test_event_groups": fold["test_event_groups"],
            "standardizer_fit_rows": train.tolist(),
            "classifier_iterations": classifier.n_iter_.tolist(),
        })
    if any(row is None for row in prediction_slots):
        raise AssertionError("every v3 endpoint row must receive one held-out prediction")
    records = [dict(row) for row in prediction_slots if row is not None]
    truth = [str(row["true_label"]) for row in records]
    synchronized = [str(row["synchronized_prediction"]) for row in records]
    synchronized_accuracy = balanced_accuracy(truth, synchronized, MODELED_ACTIONS)
    donor_keys = sorted(records[0]["donor_predictions"])
    donor_accuracies = {
        seed: balanced_accuracy(
            truth,
            [str(row["donor_predictions"][seed]) for row in records],
            MODELED_ACTIONS,
        )
        for seed in donor_keys
    }
    donor_mean = float(np.mean(list(donor_accuracies.values())))
    lift = synchronized_accuracy - donor_mean
    train_mean = float(np.mean(train_accuracies))
    gap = train_mean - synchronized_accuracy
    chance = 0.5
    bootstrap = _cluster_bootstrap(
        records,
        int(config["bootstrap_samples"]),
        int(config["bootstrap_seed"]),
        difference=False,
    )
    paired_bootstrap = _cluster_bootstrap(
        records,
        int(config["bootstrap_samples"]),
        int(config["bootstrap_seed"]),
        difference=True,
    )
    permutation = _chance_permutation(
        records,
        int(config["chance_permutations"]),
        int(config["chance_permutation_seed"]),
        synchronized_accuracy,
    )
    randomization = _paired_randomization(
        records,
        int(config["paired_randomizations"]),
        int(config["paired_randomization_seed"]),
    )
    checks = {
        "synchronized_accuracy_above_chance_margin": synchronized_accuracy
        >= chance + float(config["minimum_above_macro_chance"]),
        "bootstrap_lower_above_chance": bootstrap["ci95_low"] > chance,
        "chance_permutation_p": permutation["p_value_plus_one"]
        <= float(config["chance_permutation_p_maximum"]),
        "synchronized_minus_donor_minimum": lift
        >= float(config["minimum_synchronized_minus_donor"]),
        "paired_bootstrap_lower_above_zero": paired_bootstrap["ci95_low"] > 0,
        "paired_randomization_p": randomization["p_value_plus_one"]
        <= float(config["paired_randomization_p_maximum"]),
        "train_minus_heldout_gap": gap
        <= float(config["maximum_train_minus_heldout_gap"]),
    }
    result = {
        "schema_version": "aea-coarse-action-imu-v3",
        "protocol_id": PROTOCOL_ID,
        "scientific_role": protocol["scientific_role"],
        "status": "complete",
        "label_role": "model_assisted_development_labels_only",
        "configuration": dict(config),
        "labels": list(split["retained_labels"]),
        "support": dict(Counter(labels.tolist())),
        "rows": len(endpoint_rows),
        "event_groups": len({str(row["event_group"]) for row in endpoint_rows}),
        "macro_chance": chance,
        "mean_train_balanced_accuracy": train_mean,
        "heldout_synchronized_balanced_accuracy": synchronized_accuracy,
        "heldout_synchronized_by_action": _per_class(truth, synchronized),
        "train_minus_heldout_gap": gap,
        "donor_balanced_accuracy_by_seed": donor_accuracies,
        "mean_donor_balanced_accuracy": donor_mean,
        "synchronized_minus_donor": lift,
        "synchronized_bootstrap": bootstrap,
        "synchronized_chance_permutation": permutation,
        "paired_lift_bootstrap": paired_bootstrap,
        "paired_randomization": randomization,
        "fold_diagnostics": fold_diagnostics,
        "heldout_predictions": records,
        "gate_checks": checks,
        "imu_gate_passed": all(checks.values()),
        "source_hash_checks": source_hash_checks,
        "endpoint_checks": endpoint_checks,
        "feature_dimension": int(matrix.shape[1]),
        "all_features_finite": bool(np.isfinite(matrix).all()),
        "development_imu_arrays_opened": len(opened_paths),
        "development_imu_relative_paths_sha256": __import__("hashlib").sha256(
            ("\n".join(opened_paths) + "\n").encode()
        ).hexdigest(),
        "reserve_imu_arrays_opened": 0,
        "reserve_rgb_files_opened": 0,
    }
    _write_json_new(args.out / "imu_results.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_coarse_action_v3")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--freeze-receipt", type=Path, default=root / "protocol_freeze_receipt.json")
    parser.add_argument("--preregistration", type=Path, default=Path("docs/aea_coarse_action_v3_preregistration.md"))
    parser.add_argument("--codebook", type=Path, default=root / "annotation_codebook.md")
    parser.add_argument("--manifest", type=Path, default=root / "fixed_dense_manifest.json")
    parser.add_argument("--agreement", type=Path, default=root / "agreement_report.json")
    parser.add_argument("--split", type=Path, default=root / "split_donor_feasibility.json")
    parser.add_argument("--capacity", type=Path, default=root / "capacity_results.json")
    parser.add_argument("--development", type=Path, default=Path("output/aea_dev_learnability_v1/development_examples.jsonl"))
    parser.add_argument("--partition", type=Path, default=Path("output/aea_dev_learnability_v1/partition_manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/aea_processed"))
    parser.add_argument("--out", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "imu_gate_passed": result.get("imu_gate_passed", False),
        "development_imu_arrays_opened": result["development_imu_arrays_opened"],
        "reserve_imu_arrays_opened": result["reserve_imu_arrays_opened"],
    }, indent=2))


if __name__ == "__main__":
    main()
