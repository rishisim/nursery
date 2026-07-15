#!/usr/bin/env python3
"""Frozen, development-only AEA learnability analysis.

The runner deliberately accepts the materialized development JSONL, never the
monolithic source.  It writes one raw JSON result from which the preregistered
decision can be reproduced without re-running a model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.grounding.pilot_data import (  # noqa: E402
    AEAWindowAdapter,
    GroundingCorpus,
    WordTokenizer,
    action_prompt,
    collate_motor_free_examples,
)
from babyworld_lite.grounding.pilot_experiment import PilotConfig, resolve_device  # noqa: E402
from babyworld_lite.grounding.pilot_model import (  # noqa: E402
    GroundingModel,
    symmetric_contrastive_loss,
)


SCRIPT_SCHEMA = "aea-dev-learnability-raw-v1"
CAPACITY_SALT = "aea-capacity-v1"
AUDIT_LABELS = {
    "clear_match", "plausible_or_ambiguous", "mismatch", "not_visually_judgeable"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def action_of(row: Mapping[str, Any]) -> str:
    return str(row["evaluation_targets"]["action_verb"])


def validate_development_source(
    rows: Sequence[Mapping[str, Any]],
    partition: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail before pixels/IMU are opened if the dev-only source is not exact."""
    expected = {
        str(item["example_id"])
        for item in partition["entries"] if item["partition"] == "development"
    }
    confirmation_ids = {
        str(item["example_id"])
        for item in partition["entries"] if item["partition"] == "confirmation"
    }
    observed = [str(row["example_id"]) for row in rows]
    observed_set = set(observed)
    reserve_groups = set(map(str, protocol["confirmation_event_groups"]))
    checks = {
        "partition_protocol_matches": (
            partition["protocol_id"] == protocol["protocol_id"]
            and partition["source_examples_sha256"] == protocol["source_examples_sha256"]
        ),
        "unique_development_ids": len(observed) == len(observed_set),
        "development_id_set_exact": observed_set == expected,
        "confirmation_id_overlap_zero": not bool(observed_set & confirmation_ids),
        "confirmation_group_overlap_zero": not any(
            str(row["event_group"]) in reserve_groups for row in rows
        ),
        "development_count_exact": len(rows) == int(protocol["counts"]["development_windows"]),
        "schema_exact": all(row.get("schema_version") == "aea-grounding-v1" for row in rows),
    }
    if not all(checks.values()):
        raise ValueError(f"development source integrity failure: {checks}")
    return checks


def supported_fine_actions(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> tuple[str, ...]:
    minimum_n = int(protocol["support"]["minimum_windows"])
    minimum_groups = int(protocol["support"]["minimum_event_groups"])
    by_action: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[action_of(row)].append(row)
    return tuple(sorted(
        action for action, items in by_action.items()
        if len(items) >= minimum_n
        and len({str(item["event_group"]) for item in items}) >= minimum_groups
    ))


def fine_to_coarse(protocol: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for coarse, actions in protocol["coarse_mapping"].items():
        for action in actions:
            if action in result:
                raise ValueError(f"fine action appears in two coarse mappings: {action}")
            result[str(action)] = str(coarse)
    return result


def build_endpoints(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    fine = supported_fine_actions(rows, protocol)
    coarse_map = fine_to_coarse(protocol)
    high_set = set(map(str, protocol["semantic_high_motion_actions"]))
    high = tuple(action for action in fine if action in high_set)
    coarse_members: dict[str, list[str]] = defaultdict(list)
    # Coarse eligibility is assessed at the category grain. It must not
    # inherit the fine-label support filter, which would silently discard
    # supported categories composed of individually sparse member verbs.
    for action in sorted({action_of(row) for row in rows}):
        if action in coarse_map:
            coarse_members[coarse_map[action]].append(action)
    minimum_n = int(protocol["support"]["minimum_windows"])
    minimum_groups = int(protocol["support"]["minimum_event_groups"])
    eligible_coarse: dict[str, tuple[str, ...]] = {}
    for label, members in coarse_members.items():
        items = [row for row in rows if action_of(row) in members]
        if len(items) >= minimum_n and len({str(r["event_group"]) for r in items}) >= minimum_groups:
            eligible_coarse[label] = tuple(sorted(members))
    endpoints = {
        "fine": {
            "labels": fine,
            "row_label": {str(r["example_id"]): action_of(r) for r in rows if action_of(r) in fine},
            "prompt_members": {action: (action,) for action in fine},
        },
        "coarse": {
            "labels": tuple(sorted(eligible_coarse)),
            "row_label": {
                str(r["example_id"]): coarse_map[action_of(r)] for r in rows
                if action_of(r) in coarse_map and coarse_map[action_of(r)] in eligible_coarse
            },
            "prompt_members": eligible_coarse,
        },
        "semantic_high_motion": {
            "labels": high,
            "row_label": {str(r["example_id"]): action_of(r) for r in rows if action_of(r) in high},
            "prompt_members": {action: (action,) for action in high},
        },
    }
    for name, endpoint in endpoints.items():
        if len(endpoint["labels"]) < 2:
            raise ValueError(f"endpoint {name} has fewer than two eligible labels")
    return endpoints


def make_folds(
    rows: Sequence[Mapping[str, Any]],
    row_label: Mapping[str, str],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:  # pragma: no cover - depends on analysis runtime
        raise RuntimeError("scikit-learn from requirements.txt is required") from exc
    eligible = [i for i, row in enumerate(rows) if str(row["example_id"]) in row_label]
    labels = np.asarray([row_label[str(rows[i]["example_id"])] for i in eligible])
    groups = np.asarray([str(rows[i]["event_group"]) for i in eligible])
    cfg = protocol["folds"]
    splitter = StratifiedGroupKFold(
        n_splits=int(cfg["count"]), shuffle=bool(cfg["shuffle"]),
        random_state=int(cfg["random_state"]),
    )
    folds = []
    for fold, (train_pos, test_pos) in enumerate(
        splitter.split(np.zeros(len(eligible)), labels, groups), start=1
    ):
        train = [eligible[int(i)] for i in train_pos]
        test = [eligible[int(i)] for i in test_pos]
        folds.append({"fold": fold, "train_indices": train, "test_indices": test})
    validate_folds(rows, eligible, folds)
    return folds


def validate_folds(
    rows: Sequence[Mapping[str, Any]], eligible: Sequence[int], folds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    held_out = Counter(i for fold in folds for i in fold["test_indices"])
    checks = {
        "each_eligible_row_held_out_once": held_out == Counter({i: 1 for i in eligible}),
        "train_test_row_overlap_zero": all(
            not (set(fold["train_indices"]) & set(fold["test_indices"])) for fold in folds
        ),
        "train_test_event_group_overlap_zero": all(
            not (
                {str(rows[i]["event_group"]) for i in fold["train_indices"]}
                & {str(rows[i]["event_group"]) for i in fold["test_indices"]}
            ) for fold in folds
        ),
        "train_test_sequence_overlap_zero": all(
            not (
                {str(rows[i]["sequence_id"]) for i in fold["train_indices"]}
                & {str(rows[i]["sequence_id"]) for i in fold["test_indices"]}
            ) for fold in folds
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"fold leakage/inclusion failure: {checks}")
    return checks


def group_derangement(
    indices: Sequence[int], groups: Mapping[int, str], seed: int
) -> dict[int, int]:
    """Whole-window bijection with no self/same-group donors, or hard failure."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_group[str(groups[index])].append(int(index))
    if len(by_group) < 2:
        raise ValueError("group derangement requires at least two groups")
    largest = max(map(len, by_group.values()))
    if largest > len(indices) - largest:
        raise ValueError("group derangement impossible: largest group exceeds half the side")
    rng = np.random.default_rng(seed)
    ordered: list[int] = []
    for group in sorted(by_group):
        values = np.asarray(by_group[group], dtype=np.int64)
        ordered.extend(map(int, values[rng.permutation(len(values))]))
    donors = ordered[largest:] + ordered[:largest]
    result = dict(zip(ordered, donors))
    if set(result) != set(indices) or set(result.values()) != set(indices):
        raise AssertionError("donor mapping is not a bijection")
    if any(i == d or str(groups[i]) == str(groups[d]) for i, d in result.items()):
        raise AssertionError("donor mapping violates self/group exclusion")
    return result


def donor_preflight(
    rows: Sequence[Mapping[str, Any]], folds: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    groups = {i: str(row["event_group"]) for i, row in enumerate(rows)}
    cells = []
    all_valid = True
    for fold in folds:
        for seed in seeds:
            for side in ("train", "test"):
                indices = fold[f"{side}_indices"]
                try:
                    mapping = group_derangement(indices, groups, int(seed))
                    cells.append({
                        "fold": fold["fold"], "seed": int(seed), "side": side,
                        "valid": True,
                        "mapping_digest": canonical_digest(sorted(mapping.items())),
                        "self_match_rate": 0.0, "same_event_group_rate": 0.0,
                    })
                except ValueError as exc:
                    all_valid = False
                    cells.append({
                        "fold": fold["fold"], "seed": int(seed), "side": side,
                        "valid": False, "reason": str(exc),
                    })
    return {"all_cells_valid": all_valid, "cells": cells}


def select_capacity_subset(
    rows: Sequence[Mapping[str, Any]], eligible_actions: Sequence[str], size: int
) -> list[int]:
    by_action: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if action_of(row) in eligible_actions:
            by_action[action_of(row)].append(index)
    total_support = {action: len(values) for action, values in by_action.items()}
    action_n: Counter[str] = Counter()
    group_n: Counter[str] = Counter()
    selected: list[int] = []
    while len(selected) < min(size, sum(map(len, by_action.values()))):
        available = [action for action, values in by_action.items() if values]
        action = min(available, key=lambda a: (action_n[a], total_support[a], a))
        chosen = min(by_action[action], key=lambda i: (
            group_n[str(rows[i]["event_group"])],
            hashlib.sha256(f"{CAPACITY_SALT}|{rows[i]['example_id']}".encode()).hexdigest(),
            str(rows[i]["example_id"]),
        ))
        by_action[action].remove(chosen)
        action_n[action] += 1
        group_n[str(rows[chosen]["event_group"])] += 1
        selected.append(chosen)
    return selected


class VideoTextDataset(Dataset[dict[str, Any]]):
    """Training dataset whose code path has no motor method or field."""
    def __init__(
        self, corpus: GroundingCorpus, indices: Sequence[int], tokenizer: WordTokenizer,
        max_text_length: int = 16,
    ):
        self.corpus = corpus
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_text_length = int(max_text_length)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = self.indices[position]
        tokens, mask = self.tokenizer.encode(self.corpus.text(index), self.max_text_length)
        return {
            "video": self.corpus.video(index), "tokens": tokens, "text_mask": mask,
            "episode_index": index, "metadata": self.corpus.metadata(index),
            "text": self.corpus.text(index),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_video_text(
    corpus: GroundingCorpus, indices: Sequence[int], tokenizer: WordTokenizer,
    config: PilotConfig, seed: int, device: torch.device,
) -> tuple[GroundingModel, dict[str, Any]]:
    seed_everything(seed)
    dataset = VideoTextDataset(corpus, indices, tokenizer, config.max_text_length)
    generator = torch.Generator().manual_seed(seed + 1709)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, generator=generator,
        num_workers=0, collate_fn=collate_motor_free_examples,
    )
    model = GroundingModel(
        len(tokenizer), config.hidden_dim, config.embedding_dim, motor_dim=6
    ).to(device)
    initial = model_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    order = hashlib.sha256()
    history = []
    steps = 0
    for _epoch in range(config.epochs):
        total = 0.0
        batches = 0
        model.train()
        for batch in loader:
            order.update(json.dumps(batch["episode_index"]).encode())
            video = batch["video"].to(device)
            tokens = batch["tokens"].to(device)
            mask = batch["text_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = symmetric_contrastive_loss(
                model.encode_video(video), model.encode_text(tokens, mask), model.logit_scale
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
            steps += 1
        history.append({"video_text_loss": total / max(batches, 1)})
    return model, {
        "initialization_digest": initial,
        "training_order_digest": order.hexdigest(),
        "optimizer_steps": steps,
        "batches_per_epoch": math.ceil(len(dataset) / config.batch_size),
        "training_history": history,
        "motor_weight": 0.0,
        "motor_file_accesses": 0,
        "loss": "symmetric_video_text_contrastive_only",
    }


@torch.inference_mode()
def grounding_scores(
    model: GroundingModel, corpus: GroundingCorpus, indices: Sequence[int],
    tokenizer: WordTokenizer, labels: Sequence[str],
    prompt_members: Mapping[str, Sequence[str]], config: PilotConfig, device: torch.device,
) -> list[dict[str, Any]]:
    """Raw category scores; coarse scores average fixed member-prompt cosines."""
    model.eval()
    result = []
    for index in indices:
        video = model.encode_video(corpus.video(index)[None].to(device))[0]
        metadata = corpus.metadata(index)
        label_scores = []
        member_scores: dict[str, dict[str, float]] = {}
        for label in labels:
            prompts = []
            for member in prompt_members[label]:
                tokens, mask = tokenizer.encode(action_prompt(metadata, member), config.max_text_length)
                score = float(video @ model.encode_text(tokens[None].to(device), mask[None].to(device))[0])
                prompts.append(score)
            member_scores[label] = dict(zip(prompt_members[label], prompts))
            label_scores.append(float(np.mean(prompts)))
        result.append({
            "row_index": int(index), "scores": label_scores,
            "member_prompt_scores": member_scores,
        })
    return result


def pairwise_credit(scores: Sequence[float], true_index: int) -> float:
    correct = float(scores[true_index])
    comparisons = [
        1.0 if correct > float(score) else 0.0 if correct < float(score) else 0.5
        for i, score in enumerate(scores) if i != true_index
    ]
    return float(np.mean(comparisons))


def macro_accuracy(rows: Sequence[Mapping[str, Any]], value: str = "credit") -> float:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_label[str(row["true_label"])].append(float(row[value]))
    return float(np.mean([np.mean(values) for values in by_label.values()]))


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], samples: int, seed: int,
    statistic: Callable[[Sequence[Mapping[str, Any]]], float] = macro_accuracy,
) -> dict[str, Any]:
    """Cluster bootstrap; absent-label replicates average over labels present."""
    groups = sorted({str(row["event_group"]) for row in rows})
    if not groups:
        raise ValueError("cluster bootstrap needs rows")
    by_group = {group: [row for row in rows if str(row["event_group"]) == group] for group in groups}
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = rng.choice(groups, len(groups), replace=True)
        replicate = [row for group in selected for row in by_group[str(group)]]
        values[sample] = statistic(replicate)
    return {
        "estimate": statistic(rows), "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)), "samples": samples,
        "seed": seed, "cluster": "event_group",
        "missing_class_handling": "macro average over labels present in each replicate",
    }


def grounding_permutation(
    rows: Sequence[Mapping[str, Any]], labels: Sequence[str], samples: int, seed: int
) -> dict[str, Any]:
    observed = macro_accuracy(rows)
    by_fold: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["fold"])].append(row)
    rng = np.random.default_rng(seed)
    null = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        permuted = []
        for fold in sorted(by_fold):
            cell = by_fold[fold]
            truths = rng.permutation([str(row["true_label"]) for row in cell])
            for row, truth in zip(cell, truths):
                item = dict(row)
                item["true_label"] = str(truth)
                item["credit"] = pairwise_credit(item["scores"], labels.index(str(truth)))
                permuted.append(item)
        null[sample] = macro_accuracy(permuted)
    return {
        "observed": observed,
        "p_value_plus_one": float((1 + np.sum(null >= observed)) / (samples + 1)),
        "samples": samples, "seed": seed,
        "method": "within-fold true-label permutation preserving counts; saved seed-mean scores fixed",
    }


def seed_average_predictions(
    predictions: Sequence[Mapping[str, Any]], labels: Sequence[str], rows: Sequence[Mapping[str, Any]],
    row_label: Mapping[str, str], split: str,
) -> list[dict[str, Any]]:
    cells: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for pred in predictions:
        if pred["split"] == split:
            cells[(int(pred["fold"]), int(pred["row_index"]))].append(pred)
    averaged = []
    for (fold, index), values in sorted(cells.items()):
        scores = np.mean(np.asarray([item["scores"] for item in values]), axis=0)
        true = row_label[str(rows[index]["example_id"])]
        averaged.append({
            "fold": fold, "row_index": index, "example_id": str(rows[index]["example_id"]),
            "event_group": str(rows[index]["event_group"]), "true_label": true,
            "scores": scores.tolist(), "credit": pairwise_credit(scores, labels.index(true)),
            "seed_count": len(values),
        })
    return averaged


def endpoint_performance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_label[str(row["true_label"])].append(float(row["credit"]))
    return {
        "macro_accuracy": macro_accuracy(rows),
        "per_action": {
            label: {"n": len(values), "accuracy": float(np.mean(values))}
            for label, values in sorted(by_label.items())
        },
        "n": len(rows),
    }


def run_grounding_endpoint(
    name: str, endpoint: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], data_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    cfg = protocol["grounding"]
    config = PilotConfig(
        frame_count=int(cfg["frame_count"]), image_size=int(cfg["image_size"]),
        hidden_dim=int(cfg["hidden_dim"]), embedding_dim=int(cfg["embedding_dim"]),
        batch_size=int(cfg["batch_size"]), epochs=int(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]), motor_weight=0.0,
    )
    corpus = GroundingCorpus(rows, AEAWindowAdapter(data_root), config.frame_count, config.image_size)
    raw_predictions = []
    protocols = []
    labels = list(endpoint["labels"])
    for fold in folds:
        tokenizer = WordTokenizer.fit([corpus.text(i) for i in fold["train_indices"]])
        for seed in cfg["seeds"]:
            model, training = train_video_text(
                corpus, fold["train_indices"], tokenizer, config, int(seed), device
            )
            protocols.append({"fold": fold["fold"], "seed": int(seed), **training})
            for split in ("train", "test"):
                for pred in grounding_scores(
                    model, corpus, fold[f"{split}_indices"], tokenizer, labels,
                    endpoint["prompt_members"], config, device,
                ):
                    raw_predictions.append({
                        "fold": fold["fold"], "seed": int(seed), "split": split, **pred
                    })
    heldout = seed_average_predictions(raw_predictions, labels, rows, endpoint["row_label"], "test")
    training = seed_average_predictions(raw_predictions, labels, rows, endpoint["row_label"], "train")
    interval = cluster_bootstrap(
        heldout, int(cfg["bootstrap_samples"]), int(cfg["bootstrap_seed"])
    )
    permutation = grounding_permutation(
        heldout, labels, int(cfg["permutation_samples"]), int(cfg["permutation_seed"])
    )
    gate = bool(
        interval["estimate"] >= float(cfg["pass_accuracy"])
        and interval["ci95_low"] > float(cfg["chance"])
        and permutation["p_value_plus_one"] <= 0.05
    )
    return {
        "endpoint": name, "labels": labels,
        "prompt_members": {k: list(v) for k, v in endpoint["prompt_members"].items()},
        "folds": [{
            "fold": f["fold"],
            "train_ids": [str(rows[i]["example_id"]) for i in f["train_indices"]],
            "test_ids": [str(rows[i]["example_id"]) for i in f["test_indices"]],
        } for f in folds],
        "training_protocols": protocols, "raw_predictions": raw_predictions,
        "seed_averaged_oof_predictions": heldout,
        "inference_order": "average candidate scores across the three seeds per row, then pool OOF rows",
        "heldout": endpoint_performance(heldout), "training": endpoint_performance(training),
        "event_group_bootstrap": interval, "chance_permutation": permutation,
        "gate_passed": gate, "motor_file_accesses": 0,
    }


def run_capacity_control(
    rows: Sequence[Mapping[str, Any]], fine: Mapping[str, Any], protocol: Mapping[str, Any],
    data_root: Path, device: torch.device,
) -> dict[str, Any]:
    cfg = protocol["grounding"]
    indices = select_capacity_subset(rows, fine["labels"], int(cfg["positive_control_sample_size"]))
    config = PilotConfig(
        frame_count=int(cfg["frame_count"]), image_size=int(cfg["image_size"]),
        hidden_dim=int(cfg["hidden_dim"]), embedding_dim=int(cfg["embedding_dim"]),
        batch_size=int(cfg["batch_size"]), epochs=int(cfg["positive_control_epochs"]),
        learning_rate=float(cfg["learning_rate"]), motor_weight=0.0,
    )
    corpus = GroundingCorpus(rows, AEAWindowAdapter(data_root), config.frame_count, config.image_size)
    tokenizer = WordTokenizer.fit([corpus.text(i) for i in indices])
    model, training = train_video_text(
        corpus, indices, tokenizer, config, int(cfg["positive_control_seed"]), device
    )
    predictions = grounding_scores(
        model, corpus, indices, tokenizer, fine["labels"], fine["prompt_members"], config, device
    )
    scored = []
    labels = list(fine["labels"])
    for pred in predictions:
        index = int(pred["row_index"])
        true = fine["row_label"][str(rows[index]["example_id"])]
        scored.append({
            "true_label": true, "credit": pairwise_credit(pred["scores"], labels.index(true))
        })
    first = training["training_history"][0]["video_text_loss"]
    final = training["training_history"][-1]["video_text_loss"]
    reduction = (first - final) / first if first else -math.inf
    accuracy = macro_accuracy(scored)
    passed = bool(
        accuracy >= float(cfg["positive_control_accuracy"])
        and reduction >= float(cfg["positive_control_loss_reduction"])
    )
    return {
        "subset_ids": [str(rows[i]["example_id"]) for i in indices],
        "subset_digest": canonical_digest([str(rows[i]["example_id"]) for i in indices]),
        "selection_salt": CAPACITY_SALT, "accuracy": accuracy,
        "loss_first_epoch": first, "loss_final_epoch": final,
        "proportional_loss_reduction": reduction, "training_protocol": training,
        "gate_passed": passed, "motor_file_accesses": 0,
    }


def imu_features(values: np.ndarray, rate_hz: float) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < 2:
        raise ValueError(f"expected complete N x 6 IMU, got {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("IMU values/rate must be finite and positive")
    features: list[float] = []
    for axis in range(6):
        x = values[:, axis].astype(np.float64)
        dx = np.diff(x)
        features.extend([
            np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
            *np.quantile(x, [0.10, 0.25, 0.75, 0.90]),
            np.sqrt(np.mean(x * x)), np.mean(np.abs(x)), np.mean(x * x),
            np.mean(dx), np.std(dx), np.sqrt(np.mean(dx * dx)),
        ])
    correlation = np.corrcoef(values, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    features.extend(correlation[np.triu_indices(6, 1)])
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / rate_hz)
    bands = ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 25.0))
    for axis in range(6):
        energy = np.abs(np.fft.rfft(values[:, axis].astype(np.float64))) ** 2
        total = float(np.sum(energy))
        for band_index, (low, high) in enumerate(bands):
            mask = (frequencies >= low) & (
                frequencies <= high if band_index == len(bands) - 1 else frequencies < high
            )
            features.append(float(np.sum(energy[mask]) / total) if total > 0 else 0.0)
    result = np.asarray(features, dtype=np.float64)
    if result.shape != (129,) or not np.isfinite(result).all():
        raise ValueError(f"invalid fixed IMU features: {result.shape}")
    return result


def load_imu_matrix(rows: Sequence[Mapping[str, Any]], data_root: Path) -> np.ndarray:
    result = []
    for row in rows:
        inputs = row["model_inputs"]
        if len(inputs.get("imu_channels", ())) != 6:
            raise ValueError("IMU channel count is not six")
        values = np.load(data_root / str(inputs["imu_path"]), allow_pickle=False)
        result.append(imu_features(values, float(inputs["imu_rate_hz"])))
    return np.stack(result)


def class_balanced_accuracy(rows: Sequence[Mapping[str, Any]], value: str = "correct") -> float:
    return macro_accuracy(rows, value=value)


def classification_permutation(
    rows: Sequence[Mapping[str, Any]], samples: int, seed: int
) -> dict[str, Any]:
    observed = class_balanced_accuracy(rows)
    by_fold: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["fold"])].append(row)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(samples):
        permuted = []
        for fold in sorted(by_fold):
            cell = by_fold[fold]
            truths = rng.permutation([str(row["true_label"]) for row in cell])
            permuted.extend({**row, "true_label": str(truth), "correct": float(row["predicted_label"] == truth)}
                            for row, truth in zip(cell, truths))
        exceed += class_balanced_accuracy(permuted) >= observed
    return {
        "observed": observed, "p_value_plus_one": (exceed + 1) / (samples + 1),
        "samples": samples, "seed": seed,
    }


def run_sync_imu_endpoint(
    name: str, endpoint: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]], features: np.ndarray, protocol: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn from requirements.txt is required") from exc
    cfg = protocol["imu"]
    raw = []
    row_label = endpoint["row_label"]
    for fold in folds:
        train, test = fold["train_indices"], fold["test_indices"]
        y_train = [row_label[str(rows[i]["example_id"])] for i in train]
        scaler = StandardScaler().fit(features[train])
        classifier = LogisticRegression(
            C=float(cfg["C"]), penalty="l2", class_weight=str(cfg["class_weight"]),
            max_iter=int(cfg["max_iter"]), solver="lbfgs",
            random_state=0,
        ).fit(scaler.transform(features[train]), y_train)
        predicted = classifier.predict(scaler.transform(features[test]))
        for index, pred in zip(test, predicted):
            true = row_label[str(rows[index]["example_id"])]
            raw.append({
                "fold": fold["fold"], "row_index": int(index),
                "example_id": str(rows[index]["example_id"]),
                "event_group": str(rows[index]["event_group"]),
                "true_label": true, "predicted_label": str(pred),
                "correct": float(str(pred) == true),
            })
    permutation = classification_permutation(
        raw, int(cfg["chance_permutation_samples"]), int(cfg["chance_permutation_seed"])
    )
    sync_interval = cluster_bootstrap(
        raw, int(cfg["bootstrap_samples"]), int(cfg["bootstrap_seed"]),
        statistic=class_balanced_accuracy,
    )
    return {
        "endpoint": name, "feature_dimension": int(features.shape[1]),
        "classifier": {
            "kind": "multinomial_logistic_regression", "C": float(cfg["C"]),
            "penalty": "l2", "class_weight": cfg["class_weight"],
            "max_iter": int(cfg["max_iter"]), "standardization": "training-fold-only",
        },
        "synchronized_oof_predictions": raw,
        "synchronized": {
            "performance": endpoint_performance([
                {**row, "credit": row["correct"]} for row in raw
            ]),
            "event_group_bootstrap": sync_interval,
            "chance_permutation": permutation,
        },
        "donor_preflight": preflight,
        "shuffled_condition_executed": False,
        "shuffled_condition_reason": (
            "preflight found frozen split-side derangement infeasible; rule forbids relaxation"
            if not preflight["all_cells_valid"] else
            "not reached (all-cell feasibility unexpectedly true)"
        ),
        "paired_lift": None,
        "paired_gate_passed": False,
        "hard_pairing_check_passed": bool(preflight["all_cells_valid"]),
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total < 1:
        return {"successes": successes, "n": total, "rate": None, "ci95_low": None, "ci95_high": None}
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes, "n": total, "rate": p,
        "ci95_low": center - radius, "ci95_high": center + radius,
    }


def audit_summary(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(labels)
    counts = Counter(str(row["audit_label"]) for row in labels)
    unknown = set(counts) - AUDIT_LABELS
    if unknown:
        raise ValueError(f"unknown audit labels: {sorted(unknown)}")
    judgeable = n - counts["not_visually_judgeable"]
    plausible = counts["clear_match"] + counts["plausible_or_ambiguous"]
    metrics = {
        "judgeable": wilson(judgeable, n),
        "clear_match": wilson(counts["clear_match"], n),
        "clear_or_plausible": wilson(plausible, n),
        "mismatch": wilson(counts["mismatch"], n),
    }
    return {"n": n, "counts": dict(counts), "metrics": metrics}


def load_and_validate_audit(
    path: Path, prelabel: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload = load_json(path)
    labels = payload["rows"] if isinstance(payload, Mapping) else payload
    expected = {str(row["example_id"]) for row in prelabel["rows"]}
    by_id = {str(row["example_id"]): row for row in labels}
    development_ids = {str(row["example_id"]) for row in rows}
    if set(by_id) != expected or not set(by_id) <= development_ids:
        raise ValueError("audit labels do not exactly match frozen development audit manifest")
    if len(labels) != int(protocol["audit"]["sample_size"]):
        raise ValueError("audit label count differs from frozen sample size")
    ordered = [by_id[str(row["example_id"])] for row in prelabel["rows"]]
    overall = audit_summary(ordered)
    action_by_id = {str(row["example_id"]): action_of(row) for row in rows}
    summaries = {"overall": overall}
    for name, endpoint in endpoints.items():
        eligible_ids = set(endpoint["row_label"])
        summaries[name] = audit_summary([row for row in ordered if str(row["example_id"]) in eligible_ids])
    thresholds = protocol["audit"]
    for summary in summaries.values():
        m = summary["metrics"]
        summary["gate_passed"] = bool(
            m["judgeable"]["rate"] is not None
            and m["judgeable"]["rate"] >= float(thresholds["minimum_judgeable"])
            and m["clear_or_plausible"]["ci95_low"] >= float(thresholds["minimum_clear_or_plausible_ci_low"])
            and m["mismatch"]["ci95_high"] <= float(thresholds["maximum_mismatch_ci_high"])
        )
    return {
        "labels_path": str(path), "labels_sha256": sha256_file(path),
        "manifest_exact_match": True, "summaries": summaries,
        "rows": ordered,
        "coverage": {
            "actions": dict(Counter(action_by_id[str(r["example_id"])] for r in ordered)),
            "event_groups": len({str(r["event_group"]) for r in prelabel["rows"]}),
            "locations": len({str(r["location"]) for r in prelabel["rows"]}),
        },
    }


def mechanical_decision(
    audit: Mapping[str, Any], capacity_pass: bool,
    grounding: Mapping[str, Mapping[str, Any]], imu: Mapping[str, Mapping[str, Any]],
    hard_checks_pass: bool,
) -> dict[str, Any]:
    fine_joint = grounding["fine"]["gate_passed"] and imu["fine"]["paired_gate_passed"]
    coarse_joint = grounding["coarse"]["gate_passed"] and imu["coarse"]["paired_gate_passed"]
    high_joint = (
        grounding["semantic_high_motion"]["gate_passed"]
        and imu["semantic_high_motion"]["paired_gate_passed"]
    )
    overall_audit = audit["summaries"]["overall"]["gate_passed"]
    if not hard_checks_pass:
        recommendation, reason = "REVISE", "hard integrity/pairing check failed; endpoints invalid"
    elif not capacity_pass:
        recommendation, reason = "REVISE", "capacity control failed"
    elif overall_audit and fine_joint:
        recommendation, reason = "GO", "all fine endpoint gates passed"
    elif (
        audit["summaries"]["coarse"]["gate_passed"] and coarse_joint
    ) or (
        audit["summaries"]["semantic_high_motion"]["gate_passed"] and high_joint
    ):
        recommendation, reason = "REVISE", "a predeclared alternative jointly passed"
    else:
        recommendation, reason = "STOP", "capacity/integrity passed but audit or all joint endpoints failed"
    return {
        "recommendation": recommendation, "reason": reason,
        "fine_joint": bool(fine_joint), "coarse_joint": bool(coarse_joint),
        "semantic_high_motion_joint": bool(high_joint),
        "authorizes": (
            "separately preregistered prospective reserve run only" if recommendation == "GO"
            else "implementation/protocol repair without reserve access" if recommendation == "REVISE"
            else "do not run locked experiment or contact Professor Frank from this evidence"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-examples", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/aea_processed"),
        help="Root for development RGB/IMU relative paths (never a confirmation-specific root)",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--audit-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    protocol = load_json(args.protocol)
    partition = load_json(args.partition_manifest)
    prelabel = load_json(args.audit_manifest)
    rows = load_jsonl(args.development_examples)
    source_checks = validate_development_source(rows, partition, protocol)
    endpoints = build_endpoints(rows, protocol)
    folds = {name: make_folds(rows, endpoint["row_label"], protocol)
             for name, endpoint in endpoints.items()}
    fold_checks = {
        name: validate_folds(
            rows,
            [i for i, row in enumerate(rows) if str(row["example_id"]) in endpoints[name]["row_label"]],
            value,
        ) for name, value in folds.items()
    }
    audit = load_and_validate_audit(args.audit_labels, prelabel, rows, endpoints, protocol)
    device = resolve_device(args.device)
    grounding = {
        name: run_grounding_endpoint(
            name, endpoint, rows, folds[name], protocol, args.data_root, device
        ) for name, endpoint in endpoints.items()
    }
    capacity = run_capacity_control(
        rows, endpoints["fine"], protocol, args.data_root, device
    )
    features = load_imu_matrix(rows, args.data_root)
    imu = {}
    for name, endpoint in endpoints.items():
        preflight = donor_preflight(rows, folds[name], protocol["imu"]["donor_seeds"])
        imu[name] = run_sync_imu_endpoint(
            name, endpoint, rows, folds[name], features, protocol, preflight
        )
    hard_checks = {
        **source_checks,
        "fold_checks_all_pass": all(all(check.values()) for check in fold_checks.values()),
        "grounding_motor_file_accesses_zero": all(
            result["motor_file_accesses"] == 0 for result in grounding.values()
        ) and capacity["motor_file_accesses"] == 0,
        "finite_imu_features": bool(np.isfinite(features).all()),
        "fixed_imu_feature_dimension_129": features.shape[1] == 129,
        "all_paired_donor_cells_feasible": all(
            result["hard_pairing_check_passed"] for result in imu.values()
        ),
    }
    hard_pass = all(hard_checks.values())
    decision = mechanical_decision(
        audit, capacity["gate_passed"], grounding, imu, hard_pass
    )
    result = {
        "schema_version": SCRIPT_SCHEMA, "protocol_id": protocol["protocol_id"],
        "scientific_role": protocol["scientific_role"],
        "development_examples": str(args.development_examples),
        "development_examples_sha256": sha256_file(args.development_examples),
        "frozen_source_examples_sha256": protocol["source_examples_sha256"],
        "confirmation_files_opened": 0,
        "command": " ".join(map(str, sys.argv)), "device": str(device),
        "protocol": protocol, "hard_checks": hard_checks,
        "fold_checks": fold_checks,
        "endpoint_definitions": {
            name: {"labels": list(value["labels"]),
                   "eligible_ids": sorted(value["row_label"]),
                   "prompt_members": {k: list(v) for k, v in value["prompt_members"].items()}}
            for name, value in endpoints.items()
        },
        "audit": audit, "capacity_control": capacity,
        "grounding": grounding, "imu": imu, "decision": decision,
        "limitations": [
            "adult partly scripted sensor-format analogue; not developmental evidence",
            "ASR anchors are noisy lexical labels, not human action annotations",
            "reserve was used in an earlier infrastructure smoke and is only prospective from v1 onward",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
