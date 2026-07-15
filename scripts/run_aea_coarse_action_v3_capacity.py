#!/usr/bin/env python3
"""Frozen same-row video-only coarse-action capacity control for AEA v3."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.coarse_action_v3 import (  # noqa: E402
    MODELED_ACTIONS,
    PROTOCOL_ID,
    load_json,
    sha256_file,
    verify_protocol_freeze,
)
from scripts.run_aea_visible_action_v2_capacity import (  # noqa: E402
    _action_head_run,
    _device,
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v3 capacity artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _load_dense_videos(
    endpoint_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    repository_root: Path,
    frame_count: int,
    image_size: int,
) -> torch.Tensor:
    by_example = {str(row["example_id"]): row for row in manifest["rows"]}
    frame_indices = np.rint(np.linspace(0, 30, frame_count)).astype(int).tolist()
    if len(set(frame_indices)) != frame_count:
        raise AssertionError("v3 frozen frame subsample contains duplicates")
    evidence_root = (
        repository_root / "output/aea_visible_action_v2/dense_evidence"
    ).resolve()
    videos = []
    for endpoint in endpoint_rows:
        source = by_example[str(endpoint["example_id"])]
        frames = []
        for index in frame_indices:
            path = (repository_root / str(source["dense_frame_paths"][index])).resolve()
            if evidence_root not in path.parents:
                raise ValueError("capacity frame escaped immutable v2 dense-evidence root")
            with Image.open(path) as image:
                pixels = np.asarray(
                    image.convert("RGB").resize(
                        (image_size, image_size), Image.Resampling.BILINEAR
                    ),
                    dtype=np.uint8,
                ).copy()
            frames.append(torch.from_numpy(pixels).permute(2, 0, 1))
        videos.append(torch.stack(frames))
    return torch.stack(videos)


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol, freeze = verify_protocol_freeze(
        args.protocol, args.freeze_receipt, args.preregistration, args.codebook
    )
    agreement = load_json(args.agreement)
    split = load_json(args.split)
    gate_inputs = {
        "coarse_annotation_gate_passed": agreement.get(
            "coarse_annotation_gate_passed"
        )
        is True,
        "split_and_donor_gate_passed": split.get("split_gate_passed") is True,
    }
    preflight = {
        "schema_version": "aea-coarse-action-capacity-preflight-v3",
        "protocol_id": PROTOCOL_ID,
        "written_before_dense_image_access": True,
        "gate_inputs": gate_inputs,
        "authorized_by_stage_gates": all(gate_inputs.values()),
        "dense_development_rgb_items_opened_at_write": 0,
        "raw_vrs_opened_at_write": 0,
        "imu_arrays_opened_at_write": 0,
        "reserve_files_opened_at_write": 0,
        "natural_transcripts_used": False,
        "protocol_freeze_checks": freeze,
    }
    _write_json_new(args.out / "capacity_preflight_receipt.json", preflight)
    common = {
        "protocol_id": PROTOCOL_ID,
        "scientific_role": protocol["scientific_role"],
        "model_assisted_labels_only": True,
        "gate_inputs": gate_inputs,
        "natural_transcripts_used": False,
        "imu_used": False,
        "raw_vrs_opened": 0,
        "reserve_rgb_or_imu_files_opened": 0,
    }
    if not all(gate_inputs.values()):
        result = {
            "schema_version": "aea-coarse-action-capacity-v3",
            **common,
            "status": "not_run_stage_gate_failed",
            "capacity_gate_passed": False,
            "expensive_modeling_stopped": True,
            "dense_development_rgb_items_opened": 0,
        }
        _write_json_new(args.out / "capacity_results.json", result)
        _write_json_new(args.out / "natural_transcript_model_receipt.json", {
            "schema_version": "aea-coarse-action-natural-transcript-model-receipt-v3",
            "protocol_id": PROTOCOL_ID,
            "status": "not_run_not_authorized",
            "reason": "v3 language viability is decided by the frozen pass-consensus alignment gate; natural transcripts cannot substitute for the action-head capacity control",
            "model_runs": 0,
        })
        return result

    manifest = load_json(args.manifest)
    endpoint_rows = list(split["endpoint_row_manifest"])
    labels = tuple(split["retained_labels"])
    if set(labels) != set(MODELED_ACTIONS):
        raise ValueError("capacity endpoint is not the frozen two-class ontology")
    config = protocol["capacity"]
    videos = _load_dense_videos(
        endpoint_rows,
        manifest,
        Path.cwd().resolve(),
        int(config["frames"]),
        int(config["image_size"]),
    )
    label_index = {label: index for index, label in enumerate(labels)}
    label_indices = torch.tensor([
        label_index[str(row["observable_action"])] for row in endpoint_rows
    ], dtype=torch.long)
    device = _device(args.device)
    runs = [
        _action_head_run(
            videos,
            label_indices,
            labels,
            config,
            int(seed),
            device,
        )
        for seed in config["seeds"]
    ]
    mean_accuracy = float(np.mean([
        row["training_balanced_accuracy"] for row in runs
    ]))
    minimum_accuracy = float(min(
        row["training_balanced_accuracy"] for row in runs
    ))
    mean_reduction = float(np.mean([
        row["proportional_loss_reduction"] for row in runs
    ]))
    checks = {
        "mean_training_balanced_accuracy": mean_accuracy
        >= float(config["mean_training_balanced_accuracy_minimum"]),
        "every_seed_training_balanced_accuracy": minimum_accuracy
        >= float(config["minimum_seed_training_balanced_accuracy"]),
        "mean_loss_reduction": mean_reduction
        >= float(config["mean_loss_reduction_minimum"]),
    }
    result = {
        "schema_version": "aea-coarse-action-capacity-v3",
        **common,
        "status": "complete",
        "device": str(device),
        "configuration": dict(config),
        "rows": len(endpoint_rows),
        "labels": list(labels),
        "support": dict(Counter(
            str(row["observable_action"]) for row in endpoint_rows
        )),
        "mean_training_balanced_accuracy": mean_accuracy,
        "minimum_seed_training_balanced_accuracy": minimum_accuracy,
        "mean_proportional_loss_reduction": mean_reduction,
        "gate_checks": checks,
        "capacity_gate_passed": all(checks.values()),
        "runs": runs,
        "media_access": {
            "dense_development_rgb_items_opened": len(endpoint_rows),
            "dense_frames_per_item": int(config["frames"]),
            "raw_vrs_opened": 0,
            "imu_arrays_opened": 0,
            "reserve_files_opened": 0,
        },
        "input_sha256": {
            "manifest": sha256_file(args.manifest),
            "agreement": sha256_file(args.agreement),
            "split": sha256_file(args.split),
        },
    }
    _write_json_new(args.out / "capacity_results.json", result)
    _write_json_new(args.out / "natural_transcript_model_receipt.json", {
        "schema_version": "aea-coarse-action-natural-transcript-model-receipt-v3",
        "protocol_id": PROTOCOL_ID,
        "status": "not_run_not_authorized",
        "reason": "v3 language viability is decided by the frozen pass-consensus alignment gate; natural transcripts cannot substitute for the action-head capacity control",
        "model_runs": 0,
        "capacity_gate_passed": result["capacity_gate_passed"],
    })
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
    parser.add_argument("--out", type=Path, default=root)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "capacity_gate_passed": result.get("capacity_gate_passed", False),
        "mean_training_balanced_accuracy": result.get(
            "mean_training_balanced_accuracy"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
