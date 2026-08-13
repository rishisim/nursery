#!/usr/bin/env python3
"""One-shot, offline CLIP-L provenance diagnostic using the frozen P4 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from pilot_p4 import P4Error, atomic_json, inside, sha256


def task_aggregate(raw_by_style, tasks):
    records = defaultdict(list)
    for style_records in raw_by_style.values():
        for record in style_records:
            records[record["task_name"]].append(record)
    if set(records) != set(tasks):
        raise P4Error("diagnostic task inventory mismatch")
    scores = {}
    for task, values in records.items():
        if task.startswith("lex_"):
            bins = defaultdict(list)
            for value in values:
                label = str(value["metadata"].get("frequency_bin", "unknown"))
                if label in {"[1,2)", "[2,4)"}:
                    label = "[1,4)"
                bins[label].append(value)
            scores[task] = 100 * sum(
                sum(x["prediction"] == x["target"] for x in group) / len(group)
                for group in bins.values()
            ) / len(bins)
        else:
            scores[task] = 100 * sum(x["prediction"] == x["target"] for x in values) / len(values)
    return scores


def run(config, config_path, scratch, durable, source, archive, data_root, output, run_id):
    namespace = scratch / "evaluation_data" / config["storage"]["scratch_namespace"]
    for path in (output,):
        if not inside(path, namespace):
            raise P4Error("diagnostic output escapes its isolated namespace")
    if not inside(archive, scratch / "evaluation_data" / "pilot_p4_calibration_only"):
        raise P4Error("official archive escapes P4 calibration storage")
    if not inside(data_root, scratch / "evaluation_data" / "pilot_p4_calibration_only"):
        raise P4Error("benchmark data escapes P4 calibration storage")
    forbidden = "\n".join(map(str, (namespace, output, archive, data_root))).lower()
    if any(token in forbidden for token in ("raw_data", "derived_data", "pilot_p3", "checkpoints", "babyview")):
        raise P4Error("forbidden data or training namespace referenced")
    if archive.stat().st_size != config["archive"]["size_bytes"] or sha256(archive) != config["archive"]["sha256"]:
        raise P4Error("official archive gate failed")
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                          text=True, stdout=subprocess.PIPE).stdout.strip()
    if head != config["source"]["commit"]:
        raise P4Error("source commit mismatch")
    model_file = namespace / "models" / config["model"]["artifact_name"]
    if sha256(model_file) != config["model"]["artifact_sha256"]:
        raise P4Error("CLIP-L artifact digest mismatch")
    detail_dir = durable / "run_records" / config["storage"]["durable_record_namespace"]
    marker = detail_dir / config["execution"]["completion_marker"]
    if marker.exists():
        raise P4Error("CLIP-L diagnostic already produced a score")

    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "WANDB_DISABLED": "true", "WANDB_MODE": "disabled"})
    sys.path.insert(0, str(source))
    import torch
    original_load = torch.load
    def trusted_load(path, *args, **kwargs):
        if Path(path).resolve() == model_file.resolve():
            kwargs["weights_only"] = False
        return original_load(path, *args, **kwargs)
    torch.load = trusted_load
    from omegaconf import OmegaConf
    from evaluation.multimodal.machine_devbench.base import MachineDevBenchEvalModule

    tasks = config["inventory"]["lexical_tasks"] + config["inventory"]["grammatical_tasks"]
    per_style, raw = {}, {}
    for style in config["inventory"]["styles"]:
        cfg = OmegaConf.create({
            "_target_": "evaluation.multimodal.machine_devbench.base.MachineDevBenchEvalModule",
            "name": "machine_devbench", "output_dir": str(output), "data_root": str(data_root),
            "style": style, "tasks": tasks, "batch_size": 1, "num_workers": 8, "seed": 42,
            "model": {"_target_": "apps.baselines.clip.openclip_extractor.CLIPFeatureExtractor",
                      "name": "clip-vit-large", "kwargs": {"model_name": config["model"]["model_name"],
                      "pretrained": str(model_file)}}})
        payload = MachineDevBenchEvalModule(cfg).run()
        per_style[style], raw[style] = payload["results"], payload["raw_records"]

    scores = task_aggregate(raw, tasks)
    scores["lexical"] = (scores["lex_nouns"] + scores["lex_adjectives"]) / 2
    scores["grammatical"] = sum(scores[x] for x in config["inventory"]["grammatical_tasks"]) / 8
    scores["overall"] = (scores["lexical"] + scores["grammatical"]) / 2
    target = config["published_targets_percent"]
    errors = {key: scores[key] - target[key] for key in target}
    task_keys = tasks
    mae = sum(abs(errors[key]) for key in task_keys) / len(task_keys)
    aggregate = {
        "schema_version": 1, "record_type": "pilot_p4_clip_l_provenance_diagnostic_aggregate",
        "pilot_id": "juno_sample", "run_id": run_id, "status": "diagnostic_complete",
        "classification": config["classification"], "model_role": config["model"]["role"],
        "results_percent": {key: round(value, 6) for key, value in scores.items()},
        "signed_errors_percentage_points": {key: round(value, 6) for key, value in errors.items()},
        "comparison": {"ten_task_mean_absolute_error": round(mae, 6),
                       "maximum_absolute_task_error": round(max(abs(errors[x]) for x in task_keys), 6),
                       "closer_than_recorded_clip_b": mae < config["comparison"]["recorded_clip_b_ten_task_mean_absolute_error"]},
        "verification": {"single_completed_execution": True, "offline": True,
                         "archive_hash_passed": True, "model_hash_passed": True,
                         "frozen_inventory_passed": True, "original_clip_b_record_untouched": True,
                         "training_reference_guard_passed": True},
        "not_the_babyview_or_pilot_model": True, "cannot_initialize_training": True,
        "scientific_status_effect": "none", "p5_started": False
    }
    detail = {"run_id": run_id, "config_sha256": sha256(config_path), "per_style": per_style,
              "raw_predictions": raw, "aggregate": aggregate}
    detail_path = detail_dir / f"{run_id}.json"
    atomic_json(detail_path, detail)
    aggregate["governed_detail"] = {"sha256": sha256(detail_path), "owner_only": True}
    atomic_json(detail_dir / f"{run_id}-aggregate.json", aggregate)
    atomic_json(marker, {"run_id": run_id, "score_produced": True,
                         "config_sha256": sha256(config_path), "detail_sha256": sha256(detail_path)})
    print(json.dumps(aggregate, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "scratch", "durable", "source", "archive", "data-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.config, args.scratch, args.durable, args.source,
        args.archive, args.data_root, args.output, args.run_id)


if __name__ == "__main__":
    main()
