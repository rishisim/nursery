from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from babyworld_lite.grounding.machine_devbench_adapter import (
    NurseryMachineDevBenchExtractor,
    resolve_eval_device,
)


OFFICIAL_COMPONENTS = (
    "core/protocols/feature_extractor.py",
    "evaluation/data/machine_devbench.py",
    "evaluation/multimodal/machine_devbench/base.py",
    "evaluation/multimodal/machine_devbench/metrics.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_official(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    if not (repo / "evaluation/data/machine_devbench.py").is_file():
        raise FileNotFoundError(f"not an EgoBabyVLM checkout: {repo}")
    sys.path.insert(0, str(repo))
    from core.protocols import MultiModalFeatureExtractor
    from evaluation.data.machine_devbench import BenchmarkData
    from evaluation.multimodal.machine_devbench.metrics import (
        ResultAggregator,
        build_summary,
        merge_style_results,
    )

    return {
        "MultiModalFeatureExtractor": MultiModalFeatureExtractor,
        "BenchmarkData": BenchmarkData,
        "ResultAggregator": ResultAggregator,
        "build_summary": build_summary,
        "merge_style_results": merge_style_results,
    }


def _make_model(args: argparse.Namespace, official_repo: Path) -> tuple[Any, dict[str, Any]]:
    device = resolve_eval_device(args.device)
    if args.model == "nursery":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --model nursery")
        model = NurseryMachineDevBenchExtractor(args.checkpoint, device=str(device))
        info = {
            "kind": "nursery",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "checkpoint_metadata": model.checkpoint_metadata,
            "static_image_rule": (
                "repeat each benchmark image across the checkpoint frame_count"
            ),
        }
    else:
        extractor_path = official_repo / "apps/baselines/clip/openclip_extractor.py"
        spec = importlib.util.spec_from_file_location(
            "egobabyvlm_official_openclip_extractor", extractor_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load official extractor: {extractor_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CLIPFeatureExtractor = module.CLIPFeatureExtractor

        model = CLIPFeatureExtractor(
            model_name=args.openclip_model,
            pretrained=args.openclip_pretrained,
            device=device,
        )
        info = {
            "kind": "official_openclip_extractor",
            "model_name": args.openclip_model,
            "pretrained": args.openclip_pretrained,
        }
    model.eval()
    return model, {**info, "device": str(device), "feature_dim": model.feature_dim}


def _predict(model: Any, task_name: str, batch: dict[str, Any]) -> list[int]:
    features = model.extract_features({"image": batch["image"], "text": batch["text"]})
    similarity = model.compute_similarity(
        features["image_features"], features["text_features"], normalize=True
    ).detach().cpu()
    if task_name.startswith("lex_"):
        return [
            0 if similarity[2 * index, index] > similarity[2 * index + 1, index] else 1
            for index in range(similarity.shape[1])
        ]
    return [
        0
        if (
            similarity[2 * index, 2 * index]
            + similarity[2 * index + 1, 2 * index + 1]
        )
        > (
            similarity[2 * index, 2 * index + 1]
            + similarity[2 * index + 1, 2 * index]
        )
        else 1
        for index in range(similarity.shape[1] // 2)
    ]


def _evaluate_style(
    model: Any,
    data_root: Path,
    style: str,
    batch_size: int,
    max_trials: int,
    components: dict[str, Any],
) -> dict[str, Any]:
    benchmark = components["BenchmarkData"](data_root, style=style)
    aggregator = components["ResultAggregator"]()
    raw_records: list[dict[str, Any]] = []
    tasks = benchmark.get_tasks()
    if not tasks:
        raise ValueError(f"no official tasks found for style={style!r} at {data_root}")
    for task_index, task_name in enumerate(tasks, start=1):
        dataset = benchmark.build_dataset(task_name)
        limit = len(dataset) if max_trials <= 0 else min(len(dataset), max_trials)
        started = time.perf_counter()
        print(
            f"[{style}] task {task_index}/{len(tasks)} {task_name}: {limit} trials",
            flush=True,
        )
        for start in range(0, limit, batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, limit))]
            batch = dataset.collate_fn(samples)
            metadata = batch.pop("metadata")
            with torch.inference_mode():
                predictions = _predict(model, task_name, batch)
            for meta, prediction in zip(metadata, predictions, strict=True):
                aggregator.add(task_name, int(prediction), 0, meta)
                raw_records.append(
                    {
                        "task_name": task_name,
                        "prediction": int(prediction),
                        "target": 0,
                        "metadata": meta,
                    }
                )
        print(
            f"[{style}] completed {task_name} in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return {"results": aggregator.compute(), "raw_records": raw_records}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official EgoBabyVLM Machine-DevBench datasets, trial equations, "
            "and aggregation without the Slurm-only launcher."
        )
    )
    parser.add_argument("--official-repo", type=Path, default=Path(".external/egobabyvlm"))
    parser.add_argument("--data-root", type=Path, default=Path(".external/cache/machine_devbench"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("openclip", "nursery"), default="openclip")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--openclip-model", default="ViT-L-14-quickgelu")
    parser.add_argument("--openclip-pretrained", default="openai")
    parser.add_argument("--styles", nargs="+", default=["realistic", "cartoon"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-trials-per-task", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    official_repo = args.official_repo.resolve()
    data_root = args.data_root.resolve()
    components = _load_official(official_repo)
    model, model_info = _make_model(args, official_repo)
    if not isinstance(model, components["MultiModalFeatureExtractor"]):
        raise TypeError("model does not implement the official MultiModalFeatureExtractor protocol")

    per_style: dict[str, dict[str, Any]] = {}
    total_aggregator = components["ResultAggregator"]()
    for style in args.styles:
        payload = _evaluate_style(
            model,
            data_root,
            style,
            args.batch_size,
            args.max_trials_per_task,
            components,
        )
        per_style[style] = payload
        for record in payload["raw_records"]:
            total_aggregator.add(
                record["task_name"],
                record["prediction"],
                record["target"],
                record["metadata"],
            )

    style_results = {
        style: payload["results"] for style, payload in per_style.items()
    }
    total_results = total_aggregator.compute()
    merged = components["merge_style_results"](style_results, total_results)
    manifest_paths = sorted(data_root.rglob("manifest_*.json"))
    output = {
        "schema_version": "nursery-machine-devbench-run-v1",
        "official_source": {
            "repository": "https://github.com/facebookresearch/egobabyvlm",
            "commit": _git_commit(official_repo),
            "component_sha256": {
                relative: _sha256(official_repo / relative)
                for relative in OFFICIAL_COMPONENTS
            },
        },
        "benchmark": {
            "data_root": str(data_root),
            "styles": list(args.styles),
            "manifest_count": len(manifest_paths),
            "manifest_set_sha256": hashlib.sha256(
                "".join(_sha256(path) for path in manifest_paths).encode()
            ).hexdigest(),
        },
        "execution": {
            "batch_size": args.batch_size,
            "max_trials_per_task": args.max_trials_per_task,
            "batching_note": (
                "The official collate order and per-trial scoring equations are unchanged; "
                "batching only amortizes encoder calls."
            ),
        },
        "model": model_info,
        "summary": components["build_summary"](merged),
        "total": merged["total"],
        "by_style": merged["by_style"],
        "raw_records": {
            style: payload["raw_records"] for style, payload in per_style.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "out": str(args.out),
        "model": model_info,
        "summary": output["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
