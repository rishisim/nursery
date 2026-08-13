#!/usr/bin/env python3
"""Pilot P4: safely prepare and run the isolated Machine-DevBench calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


class P4Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_member(member: tarfile.TarInfo, wrapper: str) -> None:
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != wrapper:
        raise P4Error("archive member violates the frozen root/path contract")
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        raise P4Error("archive links/devices/FIFOs are prohibited")
    if not (member.isfile() or member.isdir()):
        raise P4Error("unexpected archive member type")


def safe_extract(archive: Path, output: Path, config: dict) -> list[dict[str, object]]:
    expected = config["archive"]
    if archive.stat().st_size != expected["size_bytes"] or sha256(archive) != expected["sha256"]:
        raise P4Error("archive size or SHA-256 mismatch")
    if output.exists() and any(output.iterdir()):
        raise P4Error("extraction destination is not empty")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    inventory = []
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        for member in members:
            validate_member(member, expected["expected_wrapper"])
        bundle.extractall(output, members=members, filter="data")
    root = output / expected["expected_wrapper"]
    observed_roots = sorted(item.name for item in root.iterdir() if item.is_dir())
    if observed_roots != sorted(expected["expected_roots"]):
        raise P4Error("unexpected extracted root layout")
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        inventory.append({"path": str(path.relative_to(output)), "size_bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    return inventory


def merge_bin(label: str) -> str:
    return "[1,4)" if label in {"[1,2)", "[2,4)"} else label


def prediction_lexical(positive: float, negative: float) -> int:
    return 0 if positive > negative else 1


def prediction_grammatical(matrix: list[list[float]]) -> int:
    matched = matrix[0][0] + matrix[1][1]
    swapped = matrix[0][1] + matrix[1][0]
    return 0 if matched > swapped else 1


def verify_namespace(config: dict, scratch: Path, durable: Path, source: Path,
                     archive: Path, data_root: Path, output: Path) -> None:
    namespace = config["storage"]["scratch_namespace"]
    allowed_scratch = scratch / "evaluation_data" / namespace
    for path in (archive, data_root, output):
        if not inside(path, allowed_scratch):
            raise P4Error("P4 path escapes the isolated calibration namespace")
    if not inside(source, scratch / "caches" / "p2_source"):
        raise P4Error("pinned source is outside the approved code cache")
    if not inside(durable / "run_records" / config["storage"]["durable_record_namespace"], durable):
        raise P4Error("durable record path is invalid")
    forbidden = ("raw_data", "derived_data", "pilot_p3", "checkpoints", "babyview")
    referenced = "\n".join(map(str, (archive, data_root, output, source))).lower()
    if any(token in referenced for token in forbidden):
        raise P4Error("forbidden data/training namespace referenced")


def verify_source(config: dict, source: Path) -> None:
    import subprocess
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                          text=True, stdout=subprocess.PIPE).stdout.strip()
    if head != config["source"]["commit"]:
        raise P4Error("pinned source commit mismatch")
    for relative, expected in config["source"]["code_sha256"].items():
        if sha256(source / relative) != expected:
            raise P4Error("pinned evaluator code hash mismatch")


def run(config: dict, config_path: Path, scratch: Path, durable: Path, source: Path,
        archive: Path, data_root: Path, output: Path, run_id: str) -> dict:
    verify_namespace(config, scratch, durable, source, archive, data_root, output)
    verify_source(config, source)
    if sha256(archive) != config["archive"]["sha256"]:
        raise P4Error("archive changed after preparation")
    model_file = scratch / "evaluation_data" / config["storage"]["scratch_namespace"] / "models" / "ViT-B-16.pt"
    if sha256(model_file) != config["model"]["artifact_sha256"]:
        raise P4Error("calibration weight digest mismatch")
    marker = durable / "run_records" / config["storage"]["durable_record_namespace"] / "score_complete.json"
    if marker.exists():
        raise P4Error("one completed calibration already exists")

    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "WANDB_DISABLED": "true", "WANDB_MODE": "disabled"})
    import sys
    sys.path.insert(0, str(source))
    import torch
    original_torch_load = torch.load
    def trusted_calibration_load(path, *args, **kwargs):
        if Path(path).resolve() == model_file.resolve():
            kwargs["weights_only"] = False
        return original_torch_load(path, *args, **kwargs)
    torch.load = trusted_calibration_load
    from omegaconf import OmegaConf
    from evaluation.multimodal.machine_devbench.base import MachineDevBenchEvalModule
    from evaluation.multimodal.machine_devbench.metrics import ResultAggregator

    tasks = config["inventory"]["lexical_tasks"] + config["inventory"]["grammatical_tasks"]
    pooled = ResultAggregator()
    per_style, raw = {}, {}
    for style in config["inventory"]["styles"]:
        cfg = OmegaConf.create({
            "_target_": "evaluation.multimodal.machine_devbench.base.MachineDevBenchEvalModule",
            "name": "machine_devbench", "output_dir": str(output), "data_root": str(data_root),
            "style": style, "tasks": tasks, "batch_size": config["model"]["batch_size"],
            "num_workers": 8, "seed": 42,
            "model": {"_target_": "apps.baselines.clip.openclip_extractor.CLIPFeatureExtractor",
                      "name": "clip-vit-base", "kwargs": {"model_name": config["model"]["model_name"],
                      "pretrained": str(model_file)}}
        })
        payload = MachineDevBenchEvalModule(cfg).run()
        per_style[style] = payload["results"]
        raw[style] = payload["raw_records"]
        for record in payload["raw_records"]:
            pooled.add(record["task_name"], record["prediction"], record["target"], record["metadata"])
    total = pooled.compute()
    observed = round(total["overall"]["accuracy"] * 100, 6)
    aggregate = {
        "schema_version": 1, "record_type": "pilot_p4_privacy_safe_aggregate",
        "pilot_id": "juno_sample", "engineering_run_id": run_id, "stage": "P4",
        "status": "p4_complete" if config["acceptance"]["minimum_percent"] <= observed <= config["acceptance"]["maximum_percent"] else "p4_incomplete",
        "classification": config["classification"], "scientific_status_effect": "none",
        "inventory_status": "incomplete_inventory", "model_role": config["model"]["role"],
        "not_the_babyview_or_pilot_model": True, "cannot_initialize_training": True,
        "full_reproduction_recalibration_required": True,
        "results_percent": {"lexical": round(total["by_task_type"]["lexical"]["accuracy"] * 100, 6),
                            "grammatical": round(total["by_task_type"]["grammatical"]["accuracy"] * 100, 6),
                            "overall": observed,
                            "by_task": {key: round(value["accuracy"] * 100, 6) for key, value in total["by_task"].items()}},
        "acceptance": {**config["acceptance"], "passed": config["acceptance"]["minimum_percent"] <= observed <= config["acceptance"]["maximum_percent"]},
        "verification": {"archive_hash_passed": True, "model_hash_passed": True, "offline": True,
                         "frozen_inventory_passed": set(total["by_task"]) == set(tasks),
                         "single_completed_execution": True, "training_reference_guard_passed": True},
        "p0_status_preserved": True, "p1_status_preserved": True, "p2_status_preserved": True,
        "p3_status_preserved": True, "p5_started": False
    }
    detail_dir = durable / "run_records" / config["storage"]["durable_record_namespace"]
    detailed = {"run_id": run_id, "config_sha256": sha256(config_path), "results": total,
                "per_style": per_style, "raw_predictions": raw, "aggregate": aggregate}
    detail_path = detail_dir / f"{run_id}.json"
    atomic_json(detail_path, detailed)
    aggregate["governed_detailed_record"] = {"opaque_run_id": run_id, "sha256": sha256(detail_path), "owner_only": True}
    atomic_json(detail_dir / f"{run_id}-aggregate.json", aggregate)
    atomic_json(marker, {"run_id": run_id, "score_produced": True, "detail_sha256": sha256(detail_path),
                         "config_sha256": sha256(config_path)})
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--durable", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--run-id", default="p4-calibration")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "extract":
        inventory = safe_extract(args.archive, args.data_root.parent, config)
        if args.inventory is None:
            raise P4Error("--inventory is required")
        atomic_json(args.inventory, inventory)
    else:
        aggregate = run(config, args.config, args.scratch, args.durable, args.source,
                        args.archive, args.data_root, args.output, args.run_id)
        print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
