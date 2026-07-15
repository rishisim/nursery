#!/usr/bin/env python3
"""Preregistered, development-only AEA head-IMU diagnostic.

This runner deliberately performs and persists all metadata-only support, fold,
reserve-isolation, and donor-bijection checks before it opens any IMU array.
An infeasible split-local donor bijection invalidates the paired endpoint; it is
never repaired by changing folds, reusing donors, or relaxing event groups.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES_PER_AXIS = (
    "mean", "std", "minimum", "maximum", "median", "p10", "p25", "p75",
    "p90", "rms", "mean_absolute", "mean_square", "difference_mean",
    "difference_std", "difference_rms",
)
FFT_BANDS_HZ = ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 25.0))
EXPECTED_FEATURE_DIMENSION = 6 * len(FEATURE_NAMES_PER_AXIS) + 15 + 6 * len(FFT_BANDS_HZ)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_development_corpus(
    rows: Sequence[Mapping[str, Any]],
    development_path: Path,
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact manifest membership without opening a sensor file."""
    receipt_path = development_path.with_suffix(".receipt.json")
    if not receipt_path.is_file():
        raise ValueError(f"missing development materialization receipt: {receipt_path}")
    receipt = load_json(receipt_path)
    observed_development_hash = sha256_file(development_path)
    receipt_hash_match = observed_development_hash == receipt.get("development_examples_sha256")
    if not receipt_hash_match:
        raise ValueError("development JSONL hash does not match its frozen receipt")

    entries = {str(item["example_id"]): item for item in partition["entries"]}
    expected_development = {
        example_id for example_id, item in entries.items()
        if item["partition"] == "development"
    }
    confirmation_ids = {
        example_id for example_id, item in entries.items()
        if item["partition"] == "confirmation"
    }
    confirmation_groups = set(map(str, partition["confirmation_event_groups"]))
    observed_ids = [str(row["example_id"]) for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("development JSONL contains duplicate example IDs")
    observed_set = set(observed_ids)
    exact_manifest_membership = observed_set == expected_development
    reserve_id_overlap = sorted(observed_set & confirmation_ids)
    reserve_group_overlap = sorted({
        str(row["event_group"]) for row in rows
        if str(row["event_group"]) in confirmation_groups
    })
    metadata_matches_manifest = all(
        str(row["event_group"]) == str(entries[str(row["example_id"])]["event_group"])
        and str(row["sequence_id"]) == str(entries[str(row["example_id"])]["sequence_id"])
        and str(row["evaluation_targets"]["action_verb"])
        == str(entries[str(row["example_id"])]["action_verb"])
        for row in rows
        if str(row["example_id"]) in entries
    )
    partition_receipt_hash_match = (
        receipt.get("partition_manifest_sha256") == sha256_file(Path(partition["_path"]))
        if "_path" in partition else None
    )
    checks = {
        "development_jsonl_receipt_hash_match": receipt_hash_match,
        "partition_manifest_receipt_hash_match": partition_receipt_hash_match,
        "exact_development_manifest_membership": exact_manifest_membership,
        "metadata_matches_partition_manifest": metadata_matches_manifest,
        "zero_confirmation_id_overlap": not reserve_id_overlap,
        "zero_confirmation_event_group_overlap": not reserve_group_overlap,
        "observed_development_windows": len(rows),
        "expected_development_windows": len(expected_development),
        "confirmation_ids_found": reserve_id_overlap,
        "confirmation_event_groups_found": reserve_group_overlap,
    }
    required = [
        receipt_hash_match,
        partition_receipt_hash_match is not False,
        exact_manifest_membership,
        metadata_matches_manifest,
        not reserve_id_overlap,
        not reserve_group_overlap,
    ]
    if not all(required):
        raise ValueError(f"development/reserve isolation validation failed: {checks}")
    return checks


def globally_supported_fine_actions(
    rows: Sequence[Mapping[str, Any]], minimum_windows: int, minimum_groups: int
) -> list[str]:
    counts: Counter[str] = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        action = str(row["evaluation_targets"]["action_verb"])
        counts[action] += 1
        groups[action].add(str(row["event_group"]))
    return sorted(
        action for action, count in counts.items()
        if count >= minimum_windows and len(groups[action]) >= minimum_groups
    )


def build_endpoints(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    minimum_windows = int(protocol["support"]["minimum_windows"])
    minimum_groups = int(protocol["support"]["minimum_event_groups"])
    eligible_fine = globally_supported_fine_actions(rows, minimum_windows, minimum_groups)

    reverse_coarse: dict[str, str] = {}
    for category, actions in protocol["coarse_mapping"].items():
        for action in actions:
            if action in reverse_coarse:
                raise ValueError(f"fine action appears in multiple coarse classes: {action}")
            reverse_coarse[str(action)] = str(category)
    missing_actions = sorted({
        str(row["evaluation_targets"]["action_verb"]) for row in rows
    } - set(reverse_coarse))
    if missing_actions:
        raise ValueError(f"actions absent from frozen coarse mapping: {missing_actions}")

    coarse_counts: Counter[str] = Counter()
    coarse_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        category = reverse_coarse[str(row["evaluation_targets"]["action_verb"])]
        coarse_counts[category] += 1
        coarse_groups[category].add(str(row["event_group"]))
    eligible_coarse = sorted(
        category for category, count in coarse_counts.items()
        if count >= minimum_windows and len(coarse_groups[category]) >= minimum_groups
    )
    eligible_high_motion = sorted(
        set(eligible_fine) & set(map(str, protocol["semantic_high_motion_actions"]))
    )

    definitions = {
        "fine": (set(eligible_fine), lambda row: str(row["evaluation_targets"]["action_verb"])),
        "coarse": (set(eligible_coarse), lambda row: reverse_coarse[str(row["evaluation_targets"]["action_verb"])]),
        "semantic_high_motion_fine": (
            set(eligible_high_motion), lambda row: str(row["evaluation_targets"]["action_verb"])
        ),
    }
    endpoints: dict[str, dict[str, Any]] = {}
    for name, (eligible_labels, labeler) in definitions.items():
        endpoint_rows = []
        for source_index, row in enumerate(rows):
            label = labeler(row)
            if label in eligible_labels:
                endpoint_rows.append({
                    "source_index": source_index,
                    "example_id": str(row["example_id"]),
                    "event_group": str(row["event_group"]),
                    "action_verb": str(row["evaluation_targets"]["action_verb"]),
                    "label": label,
                })
        endpoints[name] = {
            "name": name,
            "eligible_labels": sorted(eligible_labels),
            "rows": endpoint_rows,
            "support": support_summary(endpoint_rows),
        }
    return endpoints


def support_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(str(row["label"]) for row in rows)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row["label"])].add(str(row["event_group"]))
    return {
        label: {"windows": counts[label], "event_groups": len(groups[label])}
        for label in sorted(counts)
    }


def make_endpoint_folds(
    endpoint: Mapping[str, Any], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = endpoint["rows"]
    labels = np.asarray([str(row["label"]) for row in rows], dtype=object)
    groups = np.asarray([str(row["event_group"]) for row in rows], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=int(protocol["folds"]["count"]),
        shuffle=bool(protocol["folds"]["shuffle"]),
        random_state=int(protocol["folds"]["random_state"]),
    )
    folds = []
    held_out_counts = np.zeros(len(rows), dtype=np.int64)
    for fold_index, (train, test) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        held_out_counts[test] += 1
        train_groups = set(groups[train].tolist())
        test_groups = set(groups[test].tolist())
        folds.append({
            "fold": fold_index,
            "train_indices": train.astype(int).tolist(),
            "test_indices": test.astype(int).tolist(),
            "train_event_groups": sorted(train_groups),
            "test_event_groups": sorted(test_groups),
            "group_overlap": sorted(train_groups & test_groups),
            "train_support": support_summary([rows[int(index)] for index in train]),
            "test_support": support_summary([rows[int(index)] for index in test]),
        })
    if not np.all(held_out_counts == 1):
        raise AssertionError("each endpoint row must appear in exactly one held-out fold")
    if any(fold["group_overlap"] for fold in folds):
        raise AssertionError("event group crossed an endpoint fold")
    return folds


def group_safe_bijection(
    indices: Sequence[int], endpoint_rows: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, Any]:
    """Find a seeded deterministic whole-window bijection via bipartite matching."""
    indices = list(map(int, indices))
    groups = {index: str(endpoint_rows[index]["event_group"]) for index in indices}
    group_counts = Counter(groups.values())
    recipients = sorted(
        indices,
        key=lambda index: hashlib.sha256(
            f"aea-imu-recipient-v1|{seed}|{endpoint_rows[index]['example_id']}".encode()
        ).hexdigest(),
    )
    donor_order = sorted(
        indices,
        key=lambda index: hashlib.sha256(
            f"aea-imu-donor-v1|{seed}|{endpoint_rows[index]['example_id']}".encode()
        ).hexdigest(),
    )
    donor_to_recipient: dict[int, int] = {}

    def augment(recipient: int, seen: set[int]) -> bool:
        for donor in donor_order:
            if donor in seen or groups[donor] == groups[recipient]:
                continue
            seen.add(donor)
            if donor not in donor_to_recipient or augment(donor_to_recipient[donor], seen):
                donor_to_recipient[donor] = recipient
                return True
        return False

    matched = 0
    for recipient in recipients:
        if augment(recipient, set()):
            matched += 1
    recipient_to_donor = {recipient: donor for donor, recipient in donor_to_recipient.items()}
    feasible = matched == len(indices)
    donor_map = {
        str(endpoint_rows[recipient]["example_id"]): str(endpoint_rows[recipient_to_donor[recipient]]["example_id"])
        for recipient in sorted(indices)
    } if feasible else None
    donor_values = list(recipient_to_donor.values()) if feasible else []
    self_matches = sum(recipient == recipient_to_donor[recipient] for recipient in indices) if feasible else None
    same_group_matches = sum(
        groups[recipient] == groups[recipient_to_donor[recipient]] for recipient in indices
    ) if feasible else None
    digest = (
        hashlib.sha256(json.dumps(donor_map, sort_keys=True).encode()).hexdigest()
        if donor_map is not None else None
    )
    return {
        "feasible": feasible,
        "windows": len(indices),
        "event_group_counts": dict(sorted(group_counts.items())),
        "largest_event_group": max(group_counts.values(), default=0),
        "necessary_half_size_bound_passed": max(group_counts.values(), default=0) <= len(indices) / 2,
        "matching_size": matched,
        "whole_window_bijection": feasible and len(set(donor_values)) == len(indices),
        "self_match_count": self_matches,
        "same_event_group_match_count": same_group_matches,
        "pairing_digest": digest,
        "donor_map": donor_map,
    }


def donor_preflight(
    endpoint: Mapping[str, Any], folds: Sequence[Mapping[str, Any]], donor_seeds: Sequence[int]
) -> dict[str, Any]:
    fold_results = []
    all_sides_feasible = True
    for fold in folds:
        sides: dict[str, Any] = {}
        for side in ("train", "test"):
            per_seed = {
                str(seed): group_safe_bijection(fold[f"{side}_indices"], endpoint["rows"], int(seed))
                for seed in donor_seeds
            }
            feasible = all(item["feasible"] for item in per_seed.values())
            all_sides_feasible &= feasible
            sides[side] = {"feasible_for_all_donor_seeds": feasible, "by_seed": per_seed}
        fold_results.append({"fold": fold["fold"], "sides": sides})
    return {
        "checked_before_opening_imu": True,
        "all_fold_sides_feasible": all_sides_feasible,
        "paired_synchronized_vs_shuffled_endpoint_valid": all_sides_feasible,
        "invalid_rule": None if all_sides_feasible else (
            "At least one fixed fold side has no different-event-group whole-window donor "
            "bijection; frozen protocol forbids relaxing donors or changing folds."
        ),
        "folds": fold_results,
    }


def extract_imu_features(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] < 2:
        raise ValueError(f"expected a complete time x 6 trajectory, got {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("IMU values and sampling rate must be finite")
    features: list[float] = []
    for axis in range(6):
        column = values[:, axis]
        differences = np.diff(column)
        features.extend([
            float(np.mean(column)), float(np.std(column)), float(np.min(column)),
            float(np.max(column)), float(np.median(column)), float(np.percentile(column, 10)),
            float(np.percentile(column, 25)), float(np.percentile(column, 75)),
            float(np.percentile(column, 90)), float(np.sqrt(np.mean(column ** 2))),
            float(np.mean(np.abs(column))), float(np.mean(column ** 2)),
            float(np.mean(differences)), float(np.std(differences)),
            float(np.sqrt(np.mean(differences ** 2))),
        ])
    for left in range(6):
        for right in range(left + 1, 6):
            x = values[:, left] - np.mean(values[:, left])
            y = values[:, right] - np.mean(values[:, right])
            denominator = float(np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)))
            features.append(float(np.sum(x * y) / denominator) if denominator > 0 else 0.0)
    frequencies = np.fft.rfftfreq(values.shape[0], d=1.0 / sample_rate_hz)
    for axis in range(6):
        energy = np.abs(np.fft.rfft(values[:, axis])) ** 2
        total = float(np.sum(energy))
        for band_index, (low, high) in enumerate(FFT_BANDS_HZ):
            mask = (frequencies >= low) & (
                frequencies <= high if band_index == len(FFT_BANDS_HZ) - 1 else frequencies < high
            )
            features.append(float(np.sum(energy[mask]) / total) if total > 0 else 0.0)
    result = np.asarray(features, dtype=np.float64)
    if result.shape != (EXPECTED_FEATURE_DIMENSION,) or not np.isfinite(result).all():
        raise AssertionError(f"invalid fixed feature vector: {result.shape}")
    return result


def balanced_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    labels = sorted(set(map(str, y_true.tolist())))
    if not labels:
        raise ValueError("balanced accuracy requires observations")
    return float(np.mean([np.mean(y_pred[y_true == label] == label) for label in labels]))


def action_performance(
    y_true: Sequence[str], y_pred: Sequence[str], groups: Sequence[str]
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    groups = np.asarray(groups, dtype=object)
    return {
        label: {
            "windows": int(np.sum(y_true == label)),
            "event_groups": len(set(map(str, groups[y_true == label].tolist()))),
            "recall": float(np.mean(y_pred[y_true == label] == label)),
        }
        for label in sorted(set(map(str, y_true.tolist())))
    }


def cluster_bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups = np.asarray(sorted(set(map(str, groups.tolist()))), dtype=object)
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        boot[sample] = balanced_accuracy(y_true[indices], y_pred[indices])
    return {
        "estimate": balanced_accuracy(y_true, y_pred),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "replicates": samples,
        "seed": seed,
        "cluster": "event_group",
    }


def within_fold_label_permutation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_ids: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    observed = balanced_accuracy(y_true, y_pred)
    rng = np.random.default_rng(seed)
    null = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        permuted = y_true.copy()
        for fold in sorted(set(map(int, fold_ids.tolist()))):
            indices = np.flatnonzero(fold_ids == fold)
            permuted[indices] = rng.permutation(permuted[indices])
        null[sample] = balanced_accuracy(permuted, y_pred)
    return {
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_ci95_low": float(np.quantile(null, 0.025)),
        "null_ci95_high": float(np.quantile(null, 0.975)),
        "one_sided_p_value": float((1 + np.sum(null >= observed)) / (samples + 1)),
        "replicates": samples,
        "seed": seed,
        "method": "within-fixed-fold label permutation preserving fold class counts",
    }


def run_synchronized_cv(
    endpoint: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    all_features: np.ndarray,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rows = endpoint["rows"]
    source_indices = np.asarray([int(row["source_index"]) for row in rows], dtype=int)
    features = all_features[source_indices]
    labels = np.asarray([str(row["label"]) for row in rows], dtype=object)
    groups = np.asarray([str(row["event_group"]) for row in rows], dtype=object)
    oof_pred = np.empty(len(rows), dtype=object)
    fold_ids = np.full(len(rows), -1, dtype=int)
    fold_results = []
    pooled_train_true: list[str] = []
    pooled_train_pred: list[str] = []
    pooled_train_groups: list[str] = []
    for fold in folds:
        train = np.asarray(fold["train_indices"], dtype=int)
        test = np.asarray(fold["test_indices"], dtype=int)
        scaler = StandardScaler().fit(features[train])
        train_features = scaler.transform(features[train])
        test_features = scaler.transform(features[test])
        classifier = LogisticRegression(
            C=float(protocol["imu"]["C"]),
            penalty="l2",
            class_weight=str(protocol["imu"]["class_weight"]),
            max_iter=int(protocol["imu"]["max_iter"]),
            solver="lbfgs",
            random_state=0,
        )
        classifier.fit(train_features, labels[train])
        train_pred = classifier.predict(train_features)
        test_pred = classifier.predict(test_features)
        oof_pred[test] = test_pred
        fold_ids[test] = int(fold["fold"])
        pooled_train_true.extend(labels[train].tolist())
        pooled_train_pred.extend(train_pred.tolist())
        pooled_train_groups.extend(groups[train].tolist())
        fold_results.append({
            "fold": int(fold["fold"]),
            "train_windows": len(train),
            "held_out_windows": len(test),
            "train_balanced_accuracy": balanced_accuracy(labels[train], train_pred),
            "held_out_balanced_accuracy": balanced_accuracy(labels[test], test_pred),
            "train_per_action": action_performance(labels[train], train_pred, groups[train]),
            "held_out_per_action": action_performance(labels[test], test_pred, groups[test]),
            "classes_fit": list(map(str, classifier.classes_.tolist())),
            "iterations": list(map(int, classifier.n_iter_.tolist())),
            "scaler_fit_on_train_only": True,
            "train_event_groups": fold["train_event_groups"],
            "held_out_event_groups": fold["test_event_groups"],
            "event_group_overlap": fold["group_overlap"],
        })
    if np.any(fold_ids < 0) or any(value is None for value in oof_pred.tolist()):
        raise AssertionError("incomplete synchronized out-of-fold prediction coverage")
    bootstrap = cluster_bootstrap_interval(
        labels, oof_pred, groups,
        int(protocol["imu"]["bootstrap_samples"]), int(protocol["imu"]["bootstrap_seed"]),
    )
    chance = within_fold_label_permutation(
        labels, oof_pred, fold_ids,
        int(protocol["imu"]["chance_permutation_samples"]),
        int(protocol["imu"]["chance_permutation_seed"]),
    )
    return {
        "scientific_role": "descriptive synchronized_IMU_vs_chance_only",
        "held_out_balanced_accuracy": bootstrap["estimate"],
        "held_out_event_group_cluster_bootstrap": bootstrap,
        "held_out_chance_permutation": chance,
        "train_balanced_accuracy_pooled_fold_occurrences": balanced_accuracy(
            pooled_train_true, pooled_train_pred
        ),
        "train_per_action_pooled_fold_occurrences": action_performance(
            pooled_train_true, pooled_train_pred, pooled_train_groups
        ),
        "held_out_per_action": action_performance(labels, oof_pred, groups),
        "folds": fold_results,
        "oof_predictions": [
            {
                "example_id": str(row["example_id"]),
                "event_group": str(row["event_group"]),
                "label": str(labels[index]),
                "prediction": str(oof_pred[index]),
                "fold": int(fold_ids[index]),
            }
            for index, row in enumerate(rows)
        ],
    }


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development", type=Path,
        default=Path("output/aea_dev_learnability_v1/development_examples.jsonl"),
    )
    parser.add_argument(
        "--partition", type=Path,
        default=Path("output/aea_dev_learnability_v1/partition_manifest.json"),
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("output/aea_dev_learnability_v1/preregistered_protocol.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/aea_processed"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("output/aea_dev_learnability_v1/imu_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite prior evidence: {args.out}")
    protocol = load_json(args.protocol)
    partition = load_json(args.partition)
    partition["_path"] = str(args.partition)
    rows = load_jsonl(args.development)
    isolation = validate_development_corpus(rows, args.development, partition)
    endpoints = build_endpoints(rows, protocol)
    folds = {name: make_endpoint_folds(endpoint, protocol) for name, endpoint in endpoints.items()}
    preflight = {
        name: donor_preflight(endpoint, folds[name], list(map(int, protocol["imu"]["donor_seeds"])))
        for name, endpoint in endpoints.items()
    }
    fold_checks = {
        "each_endpoint_row_held_out_exactly_once": True,
        "zero_train_held_out_event_group_overlap": all(
            not fold["group_overlap"] for endpoint_folds in folds.values() for fold in endpoint_folds
        ),
    }
    result: dict[str, Any] = {
        "schema_version": "aea-dev-imu-diagnostic-v1",
        "protocol_id": protocol["protocol_id"],
        "scientific_role": "adult_partly_scripted_sensor_format_analogue_not_developmental_evidence",
        "status": "donor_preflight_complete_imu_not_opened",
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "development_jsonl": str(args.development),
            "development_jsonl_sha256": sha256_file(args.development),
            "partition_manifest": str(args.partition),
            "partition_manifest_sha256": sha256_file(args.partition),
            "protocol": str(args.protocol),
            "protocol_sha256": sha256_file(args.protocol),
            "data_root": str(args.data_root),
        },
        "development_isolation_checks": isolation,
        "fold_checks": fold_checks,
        "endpoints": {
            name: {
                "eligible_labels": endpoint["eligible_labels"],
                "support": endpoint["support"],
                "windows": len(endpoint["rows"]),
                "folds": folds[name],
                "donor_preflight": preflight[name],
            }
            for name, endpoint in endpoints.items()
        },
        "sensor_access": {
            "imu_arrays_opened": False,
            "confirmation_arrays_opened": False,
            "preflight_persisted_before_imu_access": True,
        },
    }
    write_result(args.out, result)

    # Only development rows have survived the exact-manifest check above. Sensor
    # access begins here, after the preflight artifact has been persisted.
    all_features = []
    imu_shapes: Counter[str] = Counter()
    for row in rows:
        relative = Path(str(row["model_inputs"]["imu_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe IMU path in development row: {relative}")
        imu_path = args.data_root / relative
        values = np.load(imu_path, allow_pickle=False)
        imu_shapes[str(tuple(values.shape))] += 1
        all_features.append(
            extract_imu_features(values, float(row["model_inputs"]["imu_rate_hz"]))
        )
    feature_matrix = np.stack(all_features)
    feature_checks = {
        "feature_dimension": int(feature_matrix.shape[1]),
        "expected_feature_dimension": EXPECTED_FEATURE_DIMENSION,
        "dimension_matches_frozen_specification": feature_matrix.shape[1] == EXPECTED_FEATURE_DIMENSION,
        "all_features_finite": bool(np.isfinite(feature_matrix).all()),
        "development_trajectories_loaded": len(feature_matrix),
        "trajectory_shapes": dict(sorted(imu_shapes.items())),
        "confirmation_arrays_opened": False,
    }
    synchronized = {
        name: run_synchronized_cv(endpoint, folds[name], feature_matrix, protocol)
        for name, endpoint in endpoints.items()
    }
    any_invalid_donor_side = any(
        not item["all_fold_sides_feasible"] for item in preflight.values()
    )
    hard_checks = {
        **{key: value for key, value in isolation.items() if isinstance(value, bool)},
        **fold_checks,
        "fixed_feature_dimension": feature_checks["dimension_matches_frozen_specification"],
        "finite_features": feature_checks["all_features_finite"],
        "zero_shuffled_self_matches": False if any_invalid_donor_side else True,
        "zero_shuffled_same_event_group_matches": False if any_invalid_donor_side else True,
        "whole_window_donor_bijections_all_fold_sides": not any_invalid_donor_side,
        "paired_sync_shuffled_folds_labels_features_identical": False if any_invalid_donor_side else True,
        "confirmation_arrays_not_opened": True,
    }
    result.update({
        "status": (
            "paired_imu_endpoints_invalid_descriptive_synchronized_vs_chance_complete"
            if any_invalid_donor_side else "paired_preflight_valid_but_paired_model_not_run_by_this_diagnostic"
        ),
        "sensor_access": {
            "imu_arrays_opened": True,
            "development_arrays_opened": len(rows),
            "confirmation_arrays_opened": False,
            "preflight_persisted_before_imu_access": True,
        },
        "feature_specification": {
            "per_axis_time_statistics": list(FEATURE_NAMES_PER_AXIS),
            "axis_pair_pearson_correlations": 15,
            "fft_energy_proportion_bands_hz": [list(item) for item in FFT_BANDS_HZ],
            "feature_standardization": "StandardScaler fit on each training fold only",
            "checks": feature_checks,
        },
        "classifier": {
            "implementation": "sklearn.linear_model.LogisticRegression",
            "C": float(protocol["imu"]["C"]),
            "penalty": "l2",
            "class_weight": protocol["imu"]["class_weight"],
            "max_iter": int(protocol["imu"]["max_iter"]),
            "solver": "lbfgs multinomial for multiclass",
        },
        "synchronized_vs_chance_descriptive": synchronized,
        "paired_synchronized_vs_shuffled": {
            "computed": False,
            "valid": False if any_invalid_donor_side else None,
            "estimate": None,
            "confidence_interval": None,
            "randomization_p_value": None,
            "gate_passed": False,
            "reason": (
                "Frozen whole-window different-event-group donor bijection is infeasible on at "
                "least one fixed fold side. No donors, folds, or endpoints were relaxed."
                if any_invalid_donor_side else
                "Paired modeling was outside the bounded descriptive fallback requested here."
            ),
        },
        "hard_integrity_checks": hard_checks,
        "all_hard_integrity_checks_passed": all(hard_checks.values()),
        "imu_gate": {
            "fine": False,
            "coarse": False,
            "semantic_high_motion_fine": False,
            "reason": "paired synchronized-minus-shuffled gate is invalid and cannot pass",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation": (
            "Synchronized IMU-vs-chance estimates are descriptive only. They cannot substitute "
            "for the preregistered paired synchronized-minus-shuffled estimand, and authorize no "
            "confirmation access, locked experiment, or external outreach."
        ),
    })
    write_result(args.out, result)


if __name__ == "__main__":
    main()
