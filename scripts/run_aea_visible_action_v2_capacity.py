#!/usr/bin/env python3
"""Frozen staged AEA v2 action-head capacity and transcript controls."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.visible_action_v2 import (  # noqa: E402
    apply_protocol_amendment,
    canonical_digest,
    load_json,
    sha256_file,
)
from babyworld_lite.grounding.pilot_data import WordTokenizer  # noqa: E402
from babyworld_lite.grounding.pilot_model import (  # noqa: E402
    TextEncoder,
    VideoEncoder,
    symmetric_contrastive_loss,
)


PROMPTS = {
    "locomotion_posture": "The visible wearer is walking or changing posture.",
    "reach_grasp": "The visible wearer is reaching for or grasping an object.",
    "transport_place": "The visible wearer is carrying or placing an object.",
    "state_change_operate": "The visible wearer is changing or operating an object or device.",
    "food_material_handling": "The visible wearer is handling food or other material.",
    "clean_groom": "The visible wearer is cleaning or grooming.",
}


class ActionHead(nn.Module):
    def __init__(self, class_count: int, hidden_dim: int):
        super().__init__()
        self.video = VideoEncoder(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, class_count)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.video(video))


class VideoTranscriptModel(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_dim: int):
        super().__init__()
        self.video = VideoEncoder(hidden_dim, hidden_dim)
        self.text = TextEncoder(vocabulary_size, hidden_dim, hidden_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v2 capacity artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int], class_count: int) -> float:
    return float(np.mean([
        np.mean([pred == label for truth, pred in zip(y_true, y_pred) if truth == label])
        for label in range(class_count)
    ]))


def _load_dense_videos(
    endpoint_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    root: Path,
    frame_count: int,
    image_size: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    by_example = {str(row["example_id"]): row for row in manifest["rows"]}
    frame_indices = np.rint(np.linspace(0, 30, frame_count)).astype(int).tolist()
    if len(set(frame_indices)) != frame_count:
        raise AssertionError("frozen dense frame subsample contains duplicates")
    videos = []
    resolved_rows = []
    evidence_root = (root / "dense_evidence").resolve()
    for endpoint in endpoint_rows:
        source = by_example[str(endpoint["example_id"])]
        frames = []
        for index in frame_indices:
            path = (root / source["dense_frame_paths"][index]).resolve()
            if evidence_root not in path.parents:
                raise ValueError("capacity frame path escaped v2 dense evidence")
            with Image.open(path) as image:
                pixels = np.asarray(
                    image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                ).copy()
            frames.append(torch.from_numpy(pixels).permute(2, 0, 1))
        videos.append(torch.stack(frames))
        resolved_rows.append({**dict(endpoint), "transcript": str(source["transcript"])})
    return torch.stack(videos), resolved_rows


def _action_head_run(
    videos: torch.Tensor,
    label_indices: torch.Tensor,
    labels: Sequence[str],
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    _seed(seed)
    model = ActionHead(len(labels), int(config["hidden_dim"])).to(device)
    initialization = _state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    counts = torch.bincount(label_indices, minlength=len(labels)).float()
    class_weights = (len(label_indices) / (len(labels) * counts)).to(device)
    generator = torch.Generator().manual_seed(seed + 9001)
    history = []
    order_records = []
    batch_size = int(config["batch_size"])
    for _epoch in range(int(config["epochs"])):
        order = torch.randperm(len(videos), generator=generator)
        order_records.append(order.tolist())
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            logits = model(videos[indices].to(device))
            loss = F.cross_entropy(logits, label_indices[indices].to(device), weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    model.eval()
    with torch.no_grad():
        logits = model(videos.to(device)).cpu()
    predictions = logits.argmax(dim=1).tolist()
    truth = label_indices.tolist()
    per_class = {
        label: float(np.mean([
            prediction == index for target, prediction in zip(truth, predictions)
            if target == index
        ]))
        for index, label in enumerate(labels)
    }
    return {
        "seed": int(seed),
        "initialization_sha256": initialization,
        "training_order_sha256": canonical_digest(order_records),
        "optimizer_steps": int(config["epochs"]) * int(np.ceil(len(videos) / batch_size)),
        "loss_first_epoch": history[0],
        "loss_final_epoch": history[-1],
        "proportional_loss_reduction": float((history[0] - history[-1]) / history[0]),
        "training_balanced_accuracy": _balanced_accuracy(truth, predictions, len(labels)),
        "training_accuracy_by_action": per_class,
        "training_history": history,
        "predictions": [
            {"row": index, "true_label": labels[target], "predicted_label": labels[prediction],
             "logits": logits[index].tolist()}
            for index, (target, prediction) in enumerate(zip(truth, predictions))
        ],
    }


def _encode_texts(
    tokenizer: WordTokenizer, texts: Sequence[str], max_length: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, masks = zip(*(tokenizer.encode(text, max_length) for text in texts))
    return torch.stack(tokens), torch.stack(masks)


def _pairwise_credit(scores: Sequence[float], true_index: int) -> float:
    own = float(scores[true_index])
    credits = [1.0 if own > score else 0.5 if own == score else 0.0
               for index, score in enumerate(scores) if index != true_index]
    return float(np.mean(credits))


def _transcript_run(
    videos: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed = int(config["seed"])
    _seed(seed)
    transcripts = [str(row["transcript"]) for row in rows]
    prompts = [PROMPTS[label] for label in labels]
    tokenizer = WordTokenizer.fit([*transcripts, *prompts])
    transcript_tokens, transcript_masks = _encode_texts(tokenizer, transcripts)
    prompt_tokens, prompt_masks = _encode_texts(tokenizer, prompts)
    model = VideoTranscriptModel(len(tokenizer), int(config["hidden_dim"])).to(device)
    initialization = _state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    generator = torch.Generator().manual_seed(seed + 9001)
    order_records = []
    history = []
    batch_size = int(config["batch_size"])
    for _epoch in range(int(config["epochs"])):
        order = torch.randperm(len(videos), generator=generator)
        order_records.append(order.tolist())
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            video_embedding = model.video(videos[indices].to(device))
            text_embedding = model.text(
                transcript_tokens[indices].to(device), transcript_masks[indices].to(device)
            )
            loss = symmetric_contrastive_loss(video_embedding, text_embedding, model.logit_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    model.eval()
    with torch.no_grad():
        video_embedding = model.video(videos.to(device))
        prompt_embedding = model.text(prompt_tokens.to(device), prompt_masks.to(device))
        scores = (video_embedding @ prompt_embedding.T).cpu().numpy()
    label_index = {label: index for index, label in enumerate(labels)}
    credits = [
        _pairwise_credit(scores[index], label_index[str(row["observable_action"])])
        for index, row in enumerate(rows)
    ]
    by_action = {
        label: float(np.mean([
            credit for credit, row in zip(credits, rows)
            if str(row["observable_action"]) == label
        ])) for label in labels
    }
    return {
        "seed": seed,
        "initialization_sha256": initialization,
        "training_order_sha256": canonical_digest(order_records),
        "optimizer_steps": int(config["epochs"]) * int(np.ceil(len(videos) / batch_size)),
        "loss_first_epoch": history[0],
        "loss_final_epoch": history[-1],
        "proportional_loss_reduction": float((history[0] - history[-1]) / history[0]),
        "training_action_balanced_2afc": float(np.mean(list(by_action.values()))),
        "training_2afc_by_action": by_action,
        "training_history": history,
        "prompts": {label: PROMPTS[label] for label in labels},
        "motor_encoder_instantiated": False,
        "imu_files_opened": 0,
    }


def run(
    protocol_path: Path,
    amendment_path: Path,
    manifest_path: Path,
    agreement_path: Path,
    split_path: Path,
    output: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = load_json(protocol_path)
    amendment = load_json(amendment_path)
    if amendment.get("parent_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("protocol amendment parent hash mismatch")
    protocol = apply_protocol_amendment(base, amendment)
    agreement = load_json(agreement_path)
    split = load_json(split_path)
    gate_inputs = {
        "annotation_gate_passed": agreement.get("annotation_gate_passed") is True,
        "split_and_donor_gate_passed": split.get("split_gate_passed") is True,
    }
    common = {
        "protocol_id": protocol["protocol_id"],
        "scientific_role": protocol["scientific_role"],
        "model_assisted_labels_only": True,
        "gate_inputs": gate_inputs,
        "reserve_rgb_or_imu_files_opened": 0,
    }
    if not all(gate_inputs.values()):
        capacity = {
            "schema_version": "aea-visible-action-capacity-v2",
            **common,
            "status": "not_run_stage_gate_failed",
            "capacity_gate_passed": False,
            "expensive_modeling_stopped": True,
        }
        transcript = {
            "schema_version": "aea-visible-action-transcript-control-v2",
            **common,
            "status": "not_run_action_head_capacity_not_passed",
            "decision_gating": False,
        }
        _write_json_new(output / "capacity_results.json", capacity)
        _write_json_new(output / "transcript_control_results.json", transcript)
        return capacity, transcript

    manifest = load_json(manifest_path)
    endpoint_rows = list(split["endpoint_row_manifest"])
    labels = tuple(map(str, split["retained_labels"]))
    config = protocol["capacity"]
    videos, resolved_rows = _load_dense_videos(
        endpoint_rows, manifest, output,
        int(config["frames"]), int(config["image_size"]),
    )
    label_index = {label: index for index, label in enumerate(labels)}
    label_indices = torch.tensor([
        label_index[str(row["observable_action"])] for row in resolved_rows
    ], dtype=torch.long)
    device = _device(device_name)
    runs = [
        _action_head_run(videos, label_indices, labels, config, int(seed), device)
        for seed in config["seeds"]
    ]
    mean_accuracy = float(np.mean([row["training_balanced_accuracy"] for row in runs]))
    minimum_accuracy = float(min(row["training_balanced_accuracy"] for row in runs))
    mean_reduction = float(np.mean([row["proportional_loss_reduction"] for row in runs]))
    checks = {
        "mean_training_balanced_accuracy": mean_accuracy
        >= float(config["mean_training_balanced_accuracy_minimum"]),
        "every_seed_training_balanced_accuracy": minimum_accuracy
        >= float(config["minimum_seed_training_balanced_accuracy"]),
        "mean_loss_reduction": mean_reduction
        >= float(config["mean_loss_reduction_minimum"]),
    }
    capacity = {
        "schema_version": "aea-visible-action-capacity-v2",
        **common,
        "status": "complete",
        "device": str(device),
        "configuration": dict(config),
        "rows": len(resolved_rows),
        "labels": list(labels),
        "support": dict(Counter(str(row["observable_action"]) for row in resolved_rows)),
        "mean_training_balanced_accuracy": mean_accuracy,
        "minimum_seed_training_balanced_accuracy": minimum_accuracy,
        "mean_proportional_loss_reduction": mean_reduction,
        "gate_checks": checks,
        "capacity_gate_passed": all(checks.values()),
        "runs": runs,
        "media_access": {
            "dense_development_rgb_items_opened": len(resolved_rows),
            "raw_vrs_opened": 0,
            "imu_arrays_opened": 0,
            "reserve_files_opened": 0,
        },
    }
    _write_json_new(output / "capacity_results.json", capacity)
    if capacity["capacity_gate_passed"]:
        transcript_config = {
            **{key: config[key] for key in ("frames", "image_size", "hidden_dim", "batch_size", "learning_rate")},
            "epochs": int(protocol["transcript_control"]["epochs"]),
            "seed": int(protocol["transcript_control"]["seed"]),
        }
        transcript_run = _transcript_run(videos, resolved_rows, labels, transcript_config, device)
        transcript = {
            "schema_version": "aea-visible-action-transcript-control-v2",
            **common,
            "status": "complete",
            "device": str(device),
            "configuration": transcript_config,
            "chance": float(protocol["transcript_control"]["chance"]),
            "descriptive_capacity_reference": float(protocol["transcript_control"]["descriptive_capacity_reference"]),
            "decision_gating": False,
            "result": transcript_run,
        }
    else:
        transcript = {
            "schema_version": "aea-visible-action-transcript-control-v2",
            **common,
            "status": "not_run_action_head_capacity_failed",
            "decision_gating": False,
        }
    _write_json_new(output / "transcript_control_results.json", transcript)
    return capacity, transcript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("output/aea_visible_action_v2")
    parser.add_argument("--protocol", type=Path, default=root / "preregistered_protocol.json")
    parser.add_argument("--amendment", type=Path, default=root / "preregistered_protocol_amendment_1.json")
    parser.add_argument("--manifest", type=Path, default=root / "dense_clip_manifest.json")
    parser.add_argument("--agreement", type=Path, default=root / "agreement_report.json")
    parser.add_argument("--split", type=Path, default=root / "split_donor_feasibility.json")
    parser.add_argument("--out", type=Path, default=root)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capacity, transcript = run(
        args.protocol, args.amendment, args.manifest, args.agreement,
        args.split, args.out, args.device,
    )
    print(json.dumps({
        "capacity_status": capacity["status"],
        "capacity_gate_passed": capacity.get("capacity_gate_passed", False),
        "transcript_status": transcript["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
