from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from babyworld_lite.grounding.pilot_data import (
    ARM_NAMES,
    ArmDataset,
    ExampleMetadata,
    GroundingCorpus,
    MotorFreeEvaluationDataset,
    WordTokenizer,
    action_prompt,
    collate_examples,
    collate_motor_free_examples,
    held_out_composition_indices,
    load_pilot_source,
    preassigned_split_indices,
    validate_action_inventory,
)
from babyworld_lite.grounding.pilot_model import GroundingModel, grounding_loss


DEFAULT_HOLDOUTS = (
    ("cup", "push"),
    ("ball", "grasp"),
    ("block", "poke"),
    ("plush", "tap"),
)


@dataclass(frozen=True)
class PilotConfig:
    frame_count: int = 6
    image_size: int = 48
    max_text_length: int = 16
    hidden_dim: int = 48
    embedding_dim: int = 32
    batch_size: int = 32
    epochs: int = 4
    learning_rate: float = 1e-3
    motor_weight: float = 0.25
    time_shift: int = 2
    bootstrap_samples: int = 2000


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device)
        for key in ("video", "tokens", "text_mask", "motor")
    }


def _make_dataset(
    corpus: GroundingCorpus,
    indices: Sequence[int],
    tokenizer: WordTokenizer,
    arm: str,
    config: PilotConfig,
    manipulation_seed: int,
) -> ArmDataset:
    return ArmDataset(
        corpus=corpus,
        indices=indices,
        tokenizer=tokenizer,
        arm=arm,
        max_text_length=config.max_text_length,
        manipulation_seed=manipulation_seed,
        time_shift=config.time_shift,
    )


def train_one_arm(
    corpus: GroundingCorpus,
    train_indices: Sequence[int],
    tokenizer: WordTokenizer,
    arm: str,
    seed: int,
    config: PilotConfig,
    device: torch.device,
) -> tuple[GroundingModel, dict[str, Any], ArmDataset]:
    _seed_everything(seed)
    dataset = _make_dataset(corpus, train_indices, tokenizer, arm, config, seed + 7103)
    generator = torch.Generator().manual_seed(seed + 1709)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_examples,
    )
    model = GroundingModel(
        vocabulary_size=len(tokenizer),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    ).to(device)
    initial_digest = _state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    steps = 0
    observed_order = hashlib.sha256()
    model.train()
    for epoch in range(config.epochs):
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            observed_order.update(json.dumps(batch["episode_index"]).encode())
            values = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = grounding_loss(
                model,
                values["video"],
                values["tokens"],
                values["text_mask"],
                values["motor"],
                config.motor_weight,
            )
            loss.backward()
            optimizer.step()
            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
            steps += 1
        history.append({key: value / max(count, 1) for key, value in totals.items()})
    return model, {
        "initialization_digest": initial_digest,
        "training_order_digest": observed_order.hexdigest(),
        "optimizer_steps": steps,
        "batches_per_epoch": math.ceil(len(dataset) / config.batch_size),
        "training_history": history,
    }, dataset


def _encode_candidate_texts(
    model: GroundingModel,
    tokenizer: WordTokenizer,
    metadata: Sequence[ExampleMetadata],
    actions: Sequence[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    encoded = [
        tokenizer.encode(action_prompt(item, action), max_length)
        for item in metadata
        for action in actions
    ]
    tokens = torch.stack([item[0] for item in encoded]).to(device)
    mask = torch.stack([item[1] for item in encoded]).to(device)
    return model.encode_text(tokens, mask).reshape(len(metadata), len(actions), -1)


@torch.inference_mode()
def evaluate_without_motor(
    model: GroundingModel,
    corpus: GroundingCorpus,
    test_indices: Sequence[int],
    tokenizer: WordTokenizer,
    actions: Sequence[str],
    config: PilotConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Primary test path: motor tensors are neither loaded nor passed to the model."""
    dataset = MotorFreeEvaluationDataset(
        corpus, test_indices, tokenizer, config.max_text_length
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motor_free_examples,
    )
    model.eval()
    two_afc_by_action: dict[str, list[float]] = {action: [] for action in actions}
    four_way_by_action: dict[str, list[float]] = {action: [] for action in actions}
    all_video: list[torch.Tensor] = []
    all_text: list[torch.Tensor] = []
    for batch in loader:
        # No motor field exists anywhere in the primary-test batch.
        video = model.encode_video(batch["video"].to(device))
        text = model.encode_text(
            batch["tokens"].to(device), batch["text_mask"].to(device)
        )
        candidates = _encode_candidate_texts(
            model, tokenizer, batch["metadata"], actions, config.max_text_length, device
        )
        scores = torch.einsum("bd,bad->ba", video, candidates).cpu()
        for row, metadata in enumerate(batch["metadata"]):
            correct = actions.index(metadata.action)
            four_way_by_action[metadata.action].append(float(scores[row].argmax() == correct))
            comparisons = [
                float(scores[row, correct] > scores[row, wrong])
                for wrong in range(len(actions))
                if wrong != correct
            ]
            two_afc_by_action[metadata.action].extend(comparisons)
        all_video.append(video.cpu())
        all_text.append(text.cpu())
    video_matrix = torch.cat(all_video)
    text_matrix = torch.cat(all_text)
    similarities = video_matrix @ text_matrix.T
    target = torch.arange(len(similarities))
    ranks_vt = similarities.argsort(dim=1, descending=True)
    ranks_tv = similarities.T.argsort(dim=1, descending=True)
    metrics = {
        "action_2afc_macro_accuracy": float(np.mean([
            np.mean(values) for values in two_afc_by_action.values() if values
        ])),
        "action_4way_macro_accuracy": float(np.mean([
            np.mean(values) for values in four_way_by_action.values() if values
        ])),
        "video_to_text_recall_at_1": float((ranks_vt[:, :1] == target[:, None]).any(1).float().mean()),
        "video_to_text_recall_at_5": float((ranks_vt[:, : min(5, len(target))] == target[:, None]).any(1).float().mean()),
        "text_to_video_recall_at_1": float((ranks_tv[:, :1] == target[:, None]).any(1).float().mean()),
        "text_to_video_recall_at_5": float((ranks_tv[:, : min(5, len(target))] == target[:, None]).any(1).float().mean()),
        "action_2afc_by_action": {
            action: float(np.mean(values)) for action, values in two_afc_by_action.items() if values
        },
        "n_test": len(test_indices),
        "primary_test_modalities": ["rendered_rgb_frames", "utterance_text"],
        "motor_hard_masked": True,
    }
    return metrics


def metadata_only_shortcut_check(
    corpus: GroundingCorpus,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    actions: Sequence[str],
) -> dict[str, float]:
    counts: dict[tuple[str, str], np.ndarray] = {}
    global_counts = np.ones(len(actions), dtype=np.float64)
    for index in train_indices:
        item = corpus.metadata(index)
        key = (item.shape, item.color)
        counts.setdefault(key, np.ones(len(actions), dtype=np.float64))
        counts[key][actions.index(item.action)] += 1
        global_counts[actions.index(item.action)] += 1
    by_action: dict[str, list[float]] = {action: [] for action in actions}
    for index in test_indices:
        item = corpus.metadata(index)
        scores = counts.get((item.shape, item.color), global_counts)
        correct = actions.index(item.action)
        by_action[item.action].extend(
            float(scores[correct] > scores[wrong])
            for wrong in range(len(actions)) if wrong != correct
        )
    return {
        "action_2afc_macro_accuracy": float(np.mean([
            np.mean(values) for values in by_action.values() if values
        ])),
        "chance": 0.5,
    }


def motor_only_manipulation_check(
    corpus: GroundingCorpus,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    tokenizer: WordTokenizer,
    arm: str,
    actions: Sequence[str],
    config: PilotConfig,
    seed: int,
) -> dict[str, float | None]:
    train = _make_dataset(corpus, train_indices, tokenizer, arm, config, seed + 7103)
    test = _make_dataset(corpus, test_indices, tokenizer, arm, config, seed + 7103)
    centroids: list[torch.Tensor] = []
    for action in actions:
        values = [
            train._motor(index).flatten()
            for index in train.indices
            if corpus.metadata(index).action == action
        ]
        centroids.append(torch.stack(values).mean(0))
    centroid_matrix = torch.stack(centroids)
    by_action: dict[str, list[float]] = {action: [] for action in actions}
    for index in test.indices:
        item = corpus.metadata(index)
        distances = ((centroid_matrix - test._motor(index).flatten()) ** 2).mean(1)
        by_action[item.action].append(float(actions[int(distances.argmin())] == item.action))
    source_match: float | None = None
    if arm == "shuffled":
        source_match = float(np.mean([
            corpus.metadata(index).action == corpus.metadata(test.donors[index]).action
            for index in test.indices
        ]))
    return {
        "nearest_centroid_action_macro_accuracy": float(np.mean([
            np.mean(values) for values in by_action.values() if values
        ])),
        "shuffled_source_action_match_rate": source_match,
        "shuffled_self_match_rate": float(np.mean([
            index == test.donors[index] for index in test.indices
        ])) if arm == "shuffled" else None,
    }


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(array, size=len(array), replace=True).mean() for _ in range(samples)
    ])
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "n_seeds": int(len(array)),
    }


def run_pilot(
    episodes_path: str | Path,
    out_dir: str | Path,
    seeds: Sequence[int],
    arms: Sequence[str] = ARM_NAMES,
    heldouts: Sequence[tuple[str, str]] | None = None,
    max_episodes: int = 0,
    alignment_condition: str = "weak",
    device_name: str = "auto",
    config: PilotConfig = PilotConfig(),
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("multiple seeds are required for confidence intervals")
    if any(arm not in ARM_NAMES for arm in arms):
        raise KeyError(f"arms must be drawn from {ARM_NAMES}")
    records, adapter, source_schema = load_pilot_source(
        episodes_path, alignment_condition=alignment_condition
    )
    if max_episodes:
        records = records[:max_episodes]
    resolved_holdouts = tuple(heldouts or DEFAULT_HOLDOUTS)
    if source_schema == "grounding-v0" and heldouts is None:
        train_indices, test_indices = preassigned_split_indices(records)
        resolved_holdouts = tuple(sorted({
            (adapter.metadata(records[index]).shape, adapter.metadata(records[index]).action)
            for index in test_indices
        }))
    else:
        train_indices, test_indices = held_out_composition_indices(
            records, adapter, resolved_holdouts
        )
    actions = validate_action_inventory(records, adapter)
    corpus = GroundingCorpus(records, adapter, config.frame_count, config.image_size)
    tokenizer = WordTokenizer.fit([corpus.text(index) for index in train_indices])
    device = resolve_device(device_name)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            model, protocol, _dataset = train_one_arm(
                corpus, train_indices, tokenizer, arm, seed, config, device
            )
            metrics = evaluate_without_motor(
                model, corpus, test_indices, tokenizer, actions, config, device
            )
            manipulation = motor_only_manipulation_check(
                corpus, train_indices, test_indices, tokenizer, arm, actions, config, seed
            )
            runs.append({
                "seed": seed,
                "arm": arm,
                "metrics": metrics,
                "training_protocol": protocol,
                "motor_only_manipulation": manipulation,
            })
    metric_names = (
        "action_2afc_macro_accuracy", "action_4way_macro_accuracy",
        "video_to_text_recall_at_1", "video_to_text_recall_at_5",
        "text_to_video_recall_at_1", "text_to_video_recall_at_5",
    )
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for arm in arms:
        aggregate[arm] = {
            metric: bootstrap_ci(
                [run["metrics"][metric] for run in runs if run["arm"] == arm],
                config.bootstrap_samples,
                6203,
            )
            for metric in metric_names
        }
    lifts: dict[str, dict[str, float]] = {}
    if "synchronized" in arms:
        sync = {
            run["seed"]: run["metrics"]["action_2afc_macro_accuracy"]
            for run in runs if run["arm"] == "synchronized"
        }
        for control in ("shuffled", "null", "time_shifted"):
            if control in arms:
                other = {
                    run["seed"]: run["metrics"]["action_2afc_macro_accuracy"]
                    for run in runs if run["arm"] == control
                }
                differences = [sync[seed] - other[seed] for seed in seeds]
                lifts[f"synchronized_minus_{control}"] = bootstrap_ci(
                    differences, config.bootstrap_samples, 8831
                )
    initialization_pairing = {
        str(seed): len({
            run["training_protocol"]["initialization_digest"]
            for run in runs if run["seed"] == seed
        }) == 1
        for seed in seeds
    }
    step_pairing = {
        str(seed): len({
            run["training_protocol"]["optimizer_steps"]
            for run in runs if run["seed"] == seed
        }) == 1
        for seed in seeds
    }
    order_pairing = {
        str(seed): len({
            run["training_protocol"]["training_order_digest"]
            for run in runs if run["seed"] == seed
        }) == 1
        for seed in seeds
    }
    result = {
        "schema_version": "grounding-pilot-results-v1",
        "config": asdict(config),
        "episodes_path": str(episodes_path),
        "source_schema": source_schema,
        "alignment_condition": alignment_condition if source_schema == "grounding-v0" else "perfect_legacy",
        "device": str(device),
        "seeds": list(seeds),
        "arms": list(arms),
        "held_out_compositions": [list(item) for item in resolved_holdouts],
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "model_input_allowlist": [
            "rendered RGB scene pixels", "utterance text", "low-level hand x/y/vx/vy"
        ],
        "primary_test": {
            "metric": "action-balanced held-out-composition 2AFC",
            "motor_input": "hard-masked by omission; motor encoder is never called",
        },
        "paired_protocol_checks": {
            "same_initialization_across_arms": initialization_pairing,
            "same_optimizer_steps_across_arms": step_pairing,
            "same_training_order_across_arms": order_pairing,
            "same_seeded_schedule": True,
        },
        "metadata_only_shortcut": metadata_only_shortcut_check(
            corpus, train_indices, test_indices, actions
        ),
        "runs": runs,
        "aggregate": aggregate,
        "paired_lifts": lifts,
        "claim_gate": (
            "Do not claim a synchronized-motor effect unless synchronized exceeds "
            "null and shuffled across paired seeds, the metadata-only shortcut does "
            "not exceed chance, and shuffled "
            "trajectories have zero self-matches."
        ),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_results.json").write_text(json.dumps(result, indent=2))
    return result
