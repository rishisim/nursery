"""Phase 4 public-only language gate and governed-asset contract validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


ARMS = ("Real-full", "Synthetic-full", "Real-small", "Mixed")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\wäöüß]+", " ", text, flags=re.UNICODE)
    return [word for word in text.split() if word]


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        nxt = [i]
        for j, hyp in enumerate(hypothesis, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (ref != hyp)))
        row = nxt
    return row[-1]


def corpus_wer(items: list[dict[str, str]]) -> float:
    errors = words = 0
    for item in items:
        ref = normalize_words(item["reference"])
        hyp = normalize_words(item["hypothesis"])
        errors += edit_distance(ref, hyp)
        words += len(ref)
    if not words:
        raise ValueError("WER reference contains no words")
    return errors / words


def chrf(reference: str, hypothesis: str, max_order: int = 6) -> float:
    reference = " ".join(normalize_words(reference))
    hypothesis = " ".join(normalize_words(hypothesis))
    scores: list[float] = []
    for order in range(1, max_order + 1):
        ref = [reference[i : i + order] for i in range(max(0, len(reference) - order + 1))]
        hyp = [hypothesis[i : i + order] for i in range(max(0, len(hypothesis) - order + 1))]
        if not ref or not hyp:
            continue
        ref_counts = {gram: ref.count(gram) for gram in set(ref)}
        hyp_counts = {gram: hyp.count(gram) for gram in set(hyp)}
        overlap = sum(min(count, hyp_counts.get(gram, 0)) for gram, count in ref_counts.items())
        precision = overlap / len(hyp)
        recall = overlap / len(ref)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def validate_timestamps(words: list[dict[str, Any]], duration: float) -> bool:
    previous = 0.0
    if not words:
        return False
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if not (0.0 <= previous <= start <= end <= duration + 1e-6):
            return False
        previous = end
    return True


def validate_result(config: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    thresholds = config["thresholds"]
    required = {
        "candidate_id", "public_corpus_wer", "translation_chrf",
        "mean_word_confidence", "abstention_rate", "timestamp_valid_fraction",
        "round_trip_fraction", "manifest_completeness_fraction", "crashes",
        "silent_truncations", "offline_reload_pass", "telemetry_disabled",
        "artifact_manifest_sha256", "runtime",
    }
    missing = sorted(required - result.keys())
    if missing:
        return [f"missing result fields: {', '.join(missing)}"]
    comparisons = (
        ("public_corpus_wer", result["public_corpus_wer"] <= thresholds["public_set_corpus_wer_max"]),
        ("translation_chrf", result["translation_chrf"] >= thresholds["translation_chrf_min"]),
        ("mean_word_confidence", result["mean_word_confidence"] >= thresholds["mean_word_confidence_min"]),
        ("abstention_rate", result["abstention_rate"] <= thresholds["max_abstention_rate"]),
        ("timestamp_valid_fraction", result["timestamp_valid_fraction"] == thresholds["timestamp_monotonic_in_bounds_fraction"]),
        ("round_trip_fraction", result["round_trip_fraction"] == thresholds["round_trip_fraction"]),
        ("manifest_completeness_fraction", result["manifest_completeness_fraction"] == thresholds["manifest_completeness_fraction"]),
    )
    errors.extend(name for name, passed in comparisons if not passed)
    if result["crashes"] != 0:
        errors.append("crashes")
    if result["silent_truncations"] != 0:
        errors.append("silent_truncations")
    if not result["offline_reload_pass"]:
        errors.append("offline_reload_pass")
    if not result["telemetry_disabled"]:
        errors.append("telemetry_disabled")
    if not re.fullmatch(r"[0-9a-f]{64}", result["artifact_manifest_sha256"]):
        errors.append("artifact_manifest_sha256")
    return errors


def select_candidate(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    if {item["candidate_id"] for item in results} != set(config["candidate_order"]):
        raise ValueError("results must contain exactly the frozen candidate family")
    evaluated = []
    for result in results:
        failures = validate_result(config, result)
        evaluated.append({**result, "gate": "PASS" if not failures else "FAIL", "failures": failures})
    passing = [item for item in evaluated if item["gate"] == "PASS"]
    passing.sort(key=lambda item: (
        item["public_corpus_wer"],
        -item["translation_chrf"],
        config["candidates"][item["candidate_id"]]["resource_rank"],
    ))
    decision = {
        "schema_version": 1,
        "gate": "phase4_public_language_pipeline",
        "status": "PASS" if passing else "NO-GO",
        "selected_candidate": passing[0]["candidate_id"] if passing else None,
        "candidates": evaluated,
        "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
    }
    decision["decision_sha256"] = hashlib.sha256(canonical_json(decision)).hexdigest()
    return decision


def validate_common_assets(record: dict[str, Any]) -> None:
    required_assets = {"machine_devbench_lexical", "held_out_real_temporal_retrieval"}
    if set(record.get("assets", {})) != required_assets:
        raise ValueError("exactly the two frozen common asset families are required")
    for name, asset in record["assets"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", asset.get("commitment_sha256", "")):
            raise ValueError(f"{name} lacks a SHA-256 commitment")
        if asset.get("sealed") is not True:
            raise ValueError(f"{name} is not sealed")
    expected = {name: asset["commitment_sha256"] for name, asset in record["assets"].items()}
    if set(record.get("arm_consumers", {})) != set(ARMS):
        raise ValueError("all four frozen arms must be present")
    for arm, assets in record["arm_consumers"].items():
        if assets != expected:
            raise ValueError(f"{arm} does not consume the identical common assets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-common-assets", type=Path)
    args = parser.parse_args()
    if args.validate_common_assets:
        validate_common_assets(json.loads(args.validate_common_assets.read_text()))
        print("common asset contract: PASS")
        return
    if not args.results or not args.output:
        parser.error("--results and --output are required for selection")
    config = json.loads(args.config.read_text())
    results = [json.loads(path.read_text()) for path in args.results]
    decision = select_candidate(config, results)
    args.output.write_bytes(canonical_json(decision))
    print(f"language gate: {decision['status']}")


if __name__ == "__main__":
    main()
