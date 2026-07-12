from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from babyworld_lite.grounding.pilot_checkpoint import CHECKPOINT_SCHEMA_VERSION
from babyworld_lite.grounding.pilot_data import WordTokenizer


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _task_name(path: Path) -> str:
    name = path.name
    if "manifest_grammatical_" in name:
        return "gram_" + name.split("manifest_grammatical_", 1)[1].rsplit("_", 1)[0]
    return "lex_" + name.split("manifest_", 1)[1].rsplit("_", 1)[0]


def _evaluated_captions(item: dict[str, Any], task: str) -> Iterable[str]:
    if task.startswith("lex_"):
        yield str(item["caption_positive"])
    else:
        yield str(item["caption_a"])
        yield str(item["caption_b"])


def build_coverage_report(data_root: Path, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')!r}")
    vocabulary = set(payload["tokenizer_vocabulary"]) - {
        WordTokenizer.PAD,
        WordTokenizer.UNK,
    }
    manifests = sorted(data_root.rglob("manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"no Machine-DevBench manifests found under {data_root}")

    target_trials: list[tuple[str, str, str]] = []
    caption_tokens: list[str] = []
    per_task_words: dict[str, set[str]] = defaultdict(set)
    per_task_trial_words: dict[str, list[str]] = defaultdict(list)
    for manifest in manifests:
        task = _task_name(manifest)
        data = json.loads(manifest.read_text())
        style = str(data.get("style", manifest.stem.rsplit("_", 1)[-1]))
        for item in data.get("items", []):
            target = str(item.get("word", "")).lower()
            if target:
                target_trials.append((style, task, target))
                per_task_words[task].add(target)
                per_task_trial_words[task].append(target)
            for caption in _evaluated_captions(item, task):
                caption_tokens.extend(WordTokenizer.words(caption))

    unique_targets = sorted({word for _style, _task, word in target_trials})
    unique_caption_tokens = sorted(set(caption_tokens))
    trial_covered = sum(word in vocabulary for _style, _task, word in target_trials)
    token_covered = sum(word in vocabulary for word in caption_tokens)
    by_task: dict[str, Any] = {}
    for task in sorted(per_task_words):
        words = per_task_words[task]
        trials = per_task_trial_words[task]
        covered_words = sorted(words & vocabulary)
        by_task[task] = {
            "unique_targets": len(words),
            "unique_targets_covered": len(covered_words),
            "unique_target_coverage": _ratio(len(covered_words), len(words)),
            "trial_weighted_target_coverage": _ratio(
                sum(word in vocabulary for word in trials), len(trials)
            ),
            "covered_targets": covered_words,
            "oov_targets": sorted(words - vocabulary),
        }

    target_counts = Counter(word for _style, _task, word in target_trials)
    return {
        "schema_version": "nursery-machine-devbench-coverage-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": payload["metadata"],
        "benchmark": {
            "data_root": str(data_root),
            "manifest_count": len(manifests),
            "target_trials": len(target_trials),
        },
        "nursery_vocabulary": {
            "size_excluding_special_tokens": len(vocabulary),
            "tokens": sorted(vocabulary),
        },
        "target_word_coverage": {
            "unique_total": len(unique_targets),
            "unique_covered": sum(word in vocabulary for word in unique_targets),
            "unique_fraction": _ratio(
                sum(word in vocabulary for word in unique_targets), len(unique_targets)
            ),
            "trial_weighted_fraction": _ratio(trial_covered, len(target_trials)),
            "covered_targets": sorted(set(unique_targets) & vocabulary),
            "most_common_oov_targets": [
                {"word": word, "trials": count}
                for word, count in target_counts.most_common()
                if word not in vocabulary
            ][:50],
        },
        "evaluated_caption_token_coverage": {
            "unique_total": len(unique_caption_tokens),
            "unique_covered": sum(word in vocabulary for word in unique_caption_tokens),
            "unique_fraction": _ratio(
                sum(word in vocabulary for word in unique_caption_tokens),
                len(unique_caption_tokens),
            ),
            "occurrence_weighted_fraction": _ratio(token_covered, len(caption_tokens)),
        },
        "by_task": by_task,
        "interpretation": (
            "Exact-token coverage is a compatibility diagnostic, not an evaluation score. "
            "Low target coverage predicts UNK collisions and makes the provisional Nursery "
            "checkpoint unsuitable for substantive Machine-DevBench conclusions until its "
            "training vocabulary and experience distribution are expanded."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    target = report["target_word_coverage"]
    caption = report["evaluated_caption_token_coverage"]
    lines = [
        "# Nursery × Machine-DevBench vocabulary coverage",
        "",
        f"- Benchmark: {report['benchmark']['manifest_count']} official manifests / "
        f"{report['benchmark']['target_trials']} target-word trials.",
        f"- Nursery vocabulary: {report['nursery_vocabulary']['size_excluding_special_tokens']} "
        "tokens (special tokens excluded).",
        f"- Unique benchmark target coverage: {target['unique_covered']}/{target['unique_total']} "
        f"({100 * target['unique_fraction']:.1f}%).",
        f"- Trial-weighted target coverage: {100 * target['trial_weighted_fraction']:.1f}%.",
        f"- Unique evaluated-caption token coverage: {caption['unique_covered']}/"
        f"{caption['unique_total']} ({100 * caption['unique_fraction']:.1f}%).",
        f"- Occurrence-weighted evaluated-caption coverage: "
        f"{100 * caption['occurrence_weighted_fraction']:.1f}%.",
        "",
        "| Task | Unique targets | Covered | Unique coverage | Trial-weighted |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, values in report["by_task"].items():
        lines.append(
            f"| {task} | {values['unique_targets']} | {values['unique_targets_covered']} | "
            f"{100 * values['unique_target_coverage']:.1f}% | "
            f"{100 * values['trial_weighted_target_coverage']:.1f}% |"
        )
    lines.extend(["", "Interpretation: " + report["interpretation"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact Nursery vocabulary coverage on Machine-DevBench.")
    parser.add_argument("--data-root", type=Path, default=Path(".external/cache/machine_devbench"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("output/machine_devbench"))
    args = parser.parse_args()
    report = build_coverage_report(args.data_root.resolve(), args.checkpoint.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "nursery_vocabulary_coverage.json"
    markdown_path = args.out_dir / "nursery_vocabulary_coverage.md"
    json_path.write_text(json.dumps(report, indent=2))
    markdown_path.write_text(_markdown(report))
    print(markdown_path.read_text())


if __name__ == "__main__":
    main()
