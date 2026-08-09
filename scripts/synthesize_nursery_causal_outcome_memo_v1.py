#!/usr/bin/env python3
"""Synthesize the concise Michael-facing memo from validated report artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
FIGURE_MODULE_PATH = ROOT / "scripts/generate_nursery_causal_central_figure_v1.py"
CAVEAT_PATH = ROOT / "docs/nursery_program_convergence_v1/mandatory_causal_outcome_caveat_box_v1.md"
CAVEAT_SHA256 = "f6c5d1d2c091935d68aefb1e2b1d5a35a579105a17850acde55297e59ee47831"
OUTPUT_ROOT = (ROOT / "output").resolve()
SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "decision",
        "source_aggregate_projection_sha256",
        "no_automatic_fallback_override_sha256",
        "passing_directional_cells",
        "total_directional_cells",
        "first_nonpassing_cell",
        "scientific_outcome_values_hardcoded",
        "restricted_content_accessed",
        "automatic_fallback_used",
        "svg_sha256",
    }
)


def load_module() -> Any:
    specification = importlib.util.spec_from_file_location("nursery_memo_figure_v1", FIGURE_MODULE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("E_MODULE_LOAD")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


figure = load_module()


class MemoError(RuntimeError):
    pass


def fail(code: str) -> None:
    raise MemoError(code)


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                result.update(block)
    except OSError:
        fail("E_INPUT_READ")
    return result.hexdigest()


def read_regular(path: Path, maximum: int) -> bytes:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            fail("E_INPUT_FILE")
        return path.read_bytes()
    except MemoError:
        raise
    except OSError:
        fail("E_INPUT_READ")


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(read_regular(path, 2 * 1024 * 1024))
    except (UnicodeError, json.JSONDecodeError):
        fail("E_INPUT_JSON")
    if not isinstance(value, Mapping):
        fail("E_INPUT_JSON")
    return value


def safe_repo_output_input(path: Path) -> None:
    try:
        path.resolve(strict=True).relative_to(OUTPUT_ROOT)
    except (OSError, ValueError):
        fail("E_INPUT_LOCATION")


def safe_repo_output_target(path: Path) -> None:
    try:
        path.parent.resolve(strict=True).relative_to(OUTPUT_ROOT)
    except (OSError, ValueError):
        fail("E_OUTPUT_LOCATION")


def validate_summary(
    summary: Mapping[str, Any],
    *,
    projection_sha256: str,
    svg_sha256: str,
    decision: str,
    passing: int,
    first_failure: Mapping[str, str] | None,
) -> None:
    if frozenset(str(key) for key in summary) != SUMMARY_KEYS:
        fail("E_FIGURE_SUMMARY_SCHEMA")
    if (
        summary.get("schema_version") != figure.OUTPUT_SCHEMA
        or summary.get("status") != "REPORTING_SHELL_RENDERED_FROM_VALIDATED_AGGREGATES"
        or summary.get("decision") != decision
        or summary.get("source_aggregate_projection_sha256") != projection_sha256
        or summary.get("no_automatic_fallback_override_sha256") != figure.OVERRIDE_SHA256
        or summary.get("passing_directional_cells") != passing
        or summary.get("total_directional_cells") != 40
        or summary.get("first_nonpassing_cell") != first_failure
        or summary.get("scientific_outcome_values_hardcoded") is not False
        or summary.get("restricted_content_accessed") is not False
        or summary.get("automatic_fallback_used") is not False
        or summary.get("svg_sha256") != svg_sha256
    ):
        fail("E_FIGURE_SUMMARY_BINDING")


def display_endpoint(value: str) -> str:
    return {
        figure.ENDPOINTS[0]: "noun/object",
        figure.ENDPOINTS[1]: "verb/action",
    }[value]


def display_comparator(value: str) -> str:
    return {
        figure.COMPARATORS[0]: "absent-side",
        figure.COMPARATORS[1]: "group-shuffled",
        figure.COMPARATORS[2]: "balanced time-shifted",
        figure.COMPARATORS[3]: "corrupted/uninformative",
    }[value]


def display_family(value: str) -> str:
    return {
        figure.FAMILIES[0]: "Qwen-ASR + Qwen-VL path",
        figure.FAMILIES[1]: "Qwen-ASR + Gemma-4 path",
        figure.FAMILIES[2]: "model intersection",
        figure.FAMILIES[3]: "model union",
        figure.FAMILIES[4]: "conservative envelope",
    }[value]


def render_memo(
    *,
    decision: str,
    passing: int,
    first_failure: Mapping[str, str] | None,
    projection_sha256: str,
    svg_sha256: str,
    caveat: str,
) -> bytes:
    if decision == "POSITIVE_PROTOTYPE_EFFECT":
        finding = (
            "All 40 preregistered range-family × comparator × endpoint cells met the frozen directional gates. "
            "This supports a provisional synchronized-cue mechanism in the tested simulator and learner family; it does not establish naturalistic acquisition."
        )
        failure_line = "None; all directional cells passed."
        next_decision = "Treat this as a bounded prototype mechanism result and ask explicitly before any ecological validation, publication, or new study."
    elif decision == "VALID_NULL_OR_NONPROMOTION" and first_failure is not None:
        finding = (
            f"The authorized one-shot outcome was valid, but only {passing}/40 directional cells passed; the mechanism is not promoted. "
            "This is a valid non-positive result for the tested simulator, learner, controls, and envelope—not proof of zero effect or general impossibility."
        )
        failure_line = (
            f"{display_family(first_failure['range_family'])}; "
            f"{display_comparator(first_failure['comparator'])}; "
            f"{display_endpoint(first_failure['endpoint'])}."
        )
        next_decision = "Archive the one-shot result as nonpromoting. Any redesign, new model, threshold change, or new study requires a separate user decision."
    else:
        fail("E_DECISION")
    body = f"""# Michael-facing causal outcome memo

## Bottom line

**{decision}**

{finding}

## Readout

- Directional cells passing: **{passing}/40**
- First nonpassing preregistered cell: {failure_line}
- Evaluation: cue-free; simulator-oracle truth only
- Automatic fallback: none
- Aggregate projection SHA-256: `{projection_sha256}`
- Central figure SHA-256: `{svg_sha256}`

## Decision rule applied

A positive result requires all five range families, four comparators, and both co-primary endpoints to pass the frozen mean-lift, one-sided-lower-bound, positive-corpus, and large-negative gates. A valid null/nonpromotion requires a valid authorized one-shot outcome and all falsifications, but at least one failed directional cell. A technical, privacy, ancestry, pairing, Gemma, or falsification failure is not a valid null and would instead require a user decision with no automatic fallback.

## Mandatory caveat

{caveat.strip()}

## Recommended next decision

{next_decision}

This memo was synthesized only from the validated aggregate reporting projection and its hash-bound figure summary. It does not authorize another outcome or scientific branch.
"""
    return body.encode("utf-8")


def exclusive_write(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except OSError:
        fail("E_OUTPUT_EXISTS_OR_UNSAFE")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        fail("E_OUTPUT_WRITE")


def synthesize(
    *,
    projection_path: Path,
    figure_summary_path: Path,
    figure_svg_path: Path,
    output_memo_path: Path,
    enforce_repo_paths: bool = True,
) -> Mapping[str, Any]:
    if enforce_repo_paths:
        for path in (projection_path, figure_summary_path, figure_svg_path):
            safe_repo_output_input(path)
        safe_repo_output_target(output_memo_path)
    if output_memo_path.exists() or not output_memo_path.parent.is_dir():
        fail("E_OUTPUT_EXISTS_OR_UNSAFE")
    projection = read_json(projection_path)
    summary = read_json(figure_summary_path)
    read_regular(figure_svg_path, 8 * 1024 * 1024)
    try:
        decision, passing, cells = figure.validate_projection(projection)
    except figure.ReportingError:
        fail("E_PROJECTION")
    first_failure = next(
        (
            {"range_family": family, "comparator": comparator, "endpoint": endpoint}
            for family, comparator, endpoint, cell in cells
            if not cell["passed"]
        ),
        None,
    )
    projection_sha256 = sha256_file(projection_path)
    svg_sha256 = sha256_file(figure_svg_path)
    validate_summary(
        summary,
        projection_sha256=projection_sha256,
        svg_sha256=svg_sha256,
        decision=decision,
        passing=passing,
        first_failure=first_failure,
    )
    if sha256_file(CAVEAT_PATH) != CAVEAT_SHA256:
        fail("E_CAVEAT_BINDING")
    try:
        caveat = read_regular(CAVEAT_PATH, 16 * 1024).decode("utf-8")
    except UnicodeError:
        fail("E_CAVEAT_BINDING")
    memo = render_memo(
        decision=decision,
        passing=passing,
        first_failure=first_failure,
        projection_sha256=projection_sha256,
        svg_sha256=svg_sha256,
        caveat=caveat,
    )
    exclusive_write(output_memo_path, memo)
    return {
        "status": "MICHAEL_MEMO_WRITTEN",
        "decision": decision,
        "passing_cells": passing,
        "memo_sha256": hashlib.sha256(memo).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--figure-summary", required=True, type=Path)
    parser.add_argument("--figure-svg", required=True, type=Path)
    parser.add_argument("--output-memo", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = synthesize(
            projection_path=args.projection,
            figure_summary_path=args.figure_summary,
            figure_svg_path=args.figure_svg,
            output_memo_path=args.output_memo,
        )
    except MemoError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
