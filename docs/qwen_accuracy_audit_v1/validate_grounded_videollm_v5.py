#!/usr/bin/env python3
"""Validate v5 public Grounded-VideoLLM results without exporting model text."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-protocol", type=Path, required=True)
    parser.add_argument("--canary-protocol", type=Path, required=True)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path, required=True)
    parser.add_argument("--smoke-runner", type=Path, required=True)
    parser.add_argument("--canary-runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime_protocol = read_json(args.runtime_protocol)
    canary_protocol = read_json(args.canary_protocol)
    smoke = read_json(args.smoke_receipt)
    canary = read_json(args.canary_receipt)
    rows = canary["results"]

    expected_cases = [case["case"] for case in canary_protocol["cases"]]
    observed_cases = [row["case"] for row in rows]
    if expected_cases != observed_cases or len(rows) != 8:
        raise RuntimeError("public canary case alignment failed")

    if not (
        runtime_protocol["status"] == "FROZEN_BEFORE_V5_MODEL_LOAD_OR_OUTPUT"
        and smoke["status"] == "PASS"
        and smoke["schema_parseable"] is True
        and smoke["restricted_data_accessed"] is False
        and smoke["network_denial_sentinel_passed"] is True
        and canary["network_denial_sentinel_passed"] is True
    ):
        raise RuntimeError("v5 freeze, smoke, or isolation gate failed")

    runner_text = args.smoke_runner.read_text() + args.canary_runner.read_text()
    forbidden_references = [
        "review_packet",
        "review_labels",
        "existing_asr_hypothesis",
        "local_sheet",
        "/sheets/",
    ]
    present_forbidden = [
        reference for reference in forbidden_references if reference in runner_text
    ]
    if present_forbidden:
        raise RuntimeError(
            f"public runner contains restricted-packet reference: {present_forbidden}"
        )

    schema_valid = sum(bool(row["schema_valid"]) for row in rows)
    correct = sum(bool(row["referential_plus_lag_correct"]) for row in rows)
    status_counts = Counter(row["predicted_status"] for row in rows)
    lag_counts = Counter(row["predicted_lag"] for row in rows)
    oracle_status_correct = sum(
        row["predicted_status"] == row["oracle_status"] for row in rows
    )
    oracle_lag_correct = sum(
        row["predicted_lag"] == row["oracle_lag"] for row in rows
    )
    if (
        schema_valid != canary["schema_valid_count"]
        or correct != canary["oracle_referential_plus_lag_correct_count"]
        or schema_valid != 8
        or correct != 4
        or status_counts != Counter({"present": 8})
        or lag_counts != Counter({"lead": 6, "overlap": 1, "lag": 1})
        or oracle_status_correct != 6
        or oracle_lag_correct != 4
        or canary["passed"] is not False
    ):
        raise RuntimeError("independent public canary recomputation failed")

    threshold = canary_protocol["pass"]
    independently_passed = (
        schema_valid >= threshold["exact_schema_minimum"]
        and correct >= threshold["oracle_referential_plus_lag_correct_minimum"]
    )
    if independently_passed:
        raise RuntimeError("canary unexpectedly passed independent recomputation")

    output = {
        "schema_version": "nursery-grounded-videollm-v5-validation-v1.0.0",
        "validated_at": "2026-07-23",
        "status": "READY_TO_SHARE_WITH_CAVEATS",
        "question": "Did the independent Grounded-VideoLLM instrument pass the frozen public admission screen required before comparison with Qwen on 27 restricted ChildLens windows?",
        "answer": "No. It passed runtime and schema checks but failed the public semantic gate 4/8 versus a 6/8 minimum.",
        "checks": {
            "runtime_protocol_frozen_before_output": True,
            "smoke_passed": True,
            "network_denied": True,
            "public_runner_restricted_packet_reference_scan_passed": True,
            "canary_case_order_and_count_match_frozen_protocol": True,
            "schema_valid_count_independently_recomputed": schema_valid,
            "referential_plus_lag_correct_count_independently_recomputed": correct,
            "oracle_status_correct_count": oracle_status_correct,
            "oracle_lag_correct_count": oracle_lag_correct,
            "predicted_status_counts": dict(sorted(status_counts.items())),
            "predicted_lag_counts": dict(sorted(lag_counts.items())),
            "public_canary_passed": independently_passed,
            "restricted_stage_authorized": False,
        },
        "decision": {
            "grounded_videollm_substantially_better_than_qwen": "NOT_DEMONSTRATED",
            "run_27_restricted_windows": False,
            "reopen_larger_human_grounded_calibration_study": False,
        },
        "caveats": [
            "The public canary is a small synthetic admission screen, not a paired ChildLens accuracy estimate.",
            "Grounded-VideoLLM was not run on the 27 restricted windows because the prospectively frozen gate failed.",
            "No qualified German human lexical review was performed or claimed.",
            "The exact released checkpoint is not documented as cleanly separable into general versus benchmark-subset-tuned variants.",
        ],
        "privacy": {
            "public_synthetic_aggregate_only": True,
            "raw_model_text_exported": False,
            "restricted_packet_opened_by_public_runners": False,
            "restricted_inference_run": False,
        },
        "provenance": {
            "runtime_protocol_sha256": digest(args.runtime_protocol),
            "canary_protocol_sha256": digest(args.canary_protocol),
            "smoke_private_receipt_sha256": digest(args.smoke_receipt),
            "canary_private_receipt_sha256": digest(args.canary_receipt),
            "smoke_runner_sha256": digest(args.smoke_runner),
            "canary_runner_sha256": digest(args.canary_runner),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
