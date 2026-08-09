#!/usr/bin/env python3
"""Validate a private review and emit only K=5-safe aggregate diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


K = 5


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - margin), 3), round(min(1.0, center + margin), 3)]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_binary(name: str, positive: int, total: int) -> dict:
    negative = total - positive
    if min(positive, negative) < K:
        return {"metric": name, "status": "SUPPRESSED_K5"}
    return {
        "metric": name,
        "status": "PUBLISHED",
        "positive_count": positive,
        "other_count": negative,
        "total": total,
        "positive_rate": round(positive / total, 3),
        "wilson_95_interval": wilson(positive, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packet = read_json(args.packet)
    labels = read_json(args.labels)
    protocol = read_json(args.protocol)
    rows = packet.get("rows")
    reviews = labels.get("rows")
    if (
        packet.get("restricted") is not True
        or labels.get("restricted") is not True
        or not isinstance(rows, list)
        or not isinstance(reviews, list)
        or len(rows) != 27
        or len(reviews) != 27
        or [row.get("audit_index") for row in rows] != list(range(1, 28))
        or [row.get("audit_index") for row in reviews] != list(range(1, 28))
    ):
        raise RuntimeError("private review alignment failed")

    qwen_referential = Counter(
        row["qwen_existing_prediction"]["referential_status"] for row in rows
    )
    qwen_lag = Counter(
        row["qwen_existing_prediction"]["lag_event_unit_bin"] for row in rows
    )
    speech_yes = sum(
        row["speech_referent_identifiable"] == "yes" for row in reviews
    )
    visible_yes = sum(
        row["visible_object_or_action_candidate_present"] == "yes"
        for row in reviews
    )
    lag_yes = sum(row["qwen_lag_bin_supported"] == "yes" for row in reviews)
    clear_yes = sum(row["ambiguity_or_null"] == "clear_candidate" for row in reviews)
    defensible_yes = sum(row["qwen_answer_defensible"] == "yes" for row in reviews)
    total = len(reviews)
    threshold = int(protocol["decision_rule"][
        "larger_human_grounded_calibration_reopen"
    ].split("at least ")[1].split(" of ")[0])
    lower_bound = wilson(defensible_yes, total)[0]
    reopen = defensible_yes >= threshold and lower_bound >= 0.5

    # Export only cells or binary complements at least K=5.
    qwen_visible = qwen_referential.get("visible_candidate", 0)
    qwen_ambiguous_or_other = total - qwen_visible
    qwen_overlap = qwen_lag.get("overlap", 0)
    qwen_nonoverlap = total - qwen_overlap
    if min(qwen_visible, qwen_ambiguous_or_other, qwen_overlap, qwen_nonoverlap) < K:
        raise RuntimeError("planned public composition cells are not K=5 safe")

    output = {
        "schema_version": "nursery-qwen-accuracy-audit-aggregate-v1.0.0",
        "status": "PROVISIONAL_VISUAL_TEMPORAL_PLAUSIBILITY_ONLY",
        "as_of": "2026-07-22",
        "scope": {
            "audited_existing_windows": total,
            "distinct_retained_extension_items": 15,
            "parent_frozen_calibration_items": 30,
            "source_population_existing_extension_windows": 135,
            "qwen_rerun": False,
            "new_acquisition": False,
            "causal_endpoint_opened": False,
            "scientific_outcome_run": False,
        },
        "sampling": {
            "content_blind": True,
            "prediction_blind": True,
            "one_window_from_every_retained_item_plus_second_window_from_twelve_items": True,
        },
        "qwen_sample_composition_k5_safe": {
            "referential_visible_candidate": qwen_visible,
            "referential_ambiguous_or_other": qwen_ambiguous_or_other,
            "lag_overlap": qwen_overlap,
            "lag_nonoverlap": qwen_nonoverlap,
        },
        "review_metrics": {
            "speech_referent_identifiable": safe_binary(
                "speech_referent_identifiable", speech_yes, total
            ),
            "visible_candidate_present": safe_binary(
                "visible_candidate_present", visible_yes, total
            ),
            "qwen_exact_lag_bin_visually_supported": safe_binary(
                "qwen_exact_lag_bin_visually_supported", lag_yes, total
            ),
            "clear_candidate_rather_than_ambiguity_null_or_undecidable": safe_binary(
                "clear_candidate_rather_than_ambiguity_null_or_undecidable",
                clear_yes,
                total,
            ),
            "qwen_complete_answer_defensible": safe_binary(
                "qwen_complete_answer_defensible", defensible_yes, total
            ),
        },
        "decision": {
            "reopen_larger_human_grounded_calibration_study": reopen,
            "screening_threshold_count": threshold,
            "screening_threshold_wilson_lower_bound": 0.5,
            "observed_defensible_count": defensible_yes,
            "observed_defensible_wilson_lower_bound": lower_bound,
        },
        "limitations": [
            "Reviewer was not a qualified German interpreter.",
            "This is a single-reviewer visual/temporal plausibility audit, not human gold or inter-rater reliability.",
            "Only the 15 retained v1.8 extension clips were reviewable; the original 15 videos were not reacquired.",
            "The existing five-frame representation can miss events between frames.",
            "ASR errors and utterance-window alignment errors are inseparable from Qwen3-VL errors in this diagnostic.",
            "The sample is too small for publication-grade accuracy estimation.",
        ],
        "privacy": {
            "minimum_cell_k": K,
            "complementary_suppression_applied": True,
            "raw_or_per_window_payload_exported": False,
            "restricted_review_stayed_local": True,
        },
        "provenance": {
            "protocol_sha256": digest(args.protocol),
            "private_packet_sha256_recorded_publicly": False,
            "private_labels_sha256_recorded_publicly": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
