#!/usr/bin/env python3
"""Render the frozen Michael-facing central figure from aggregate causal output.

The program accepts one closed-schema aggregate projection.  It never reads
bundle rows, calibration payloads, restricted content, environment paths, or
network resources.  Invalid input is reported with a fixed error code and no
offending value is echoed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping


VERSION = "nursery-causal-central-figure-generator-v1"
INPUT_SCHEMA = "nursery-causal-reporting-aggregate-projection-v1"
OUTPUT_SCHEMA = "nursery-causal-reporting-summary-v1"
OVERRIDE_SHA256 = "cb9e7061a5725050e6a71a7b6e41f6f5e7bf30d104bc46bae233816ca99a90b7"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

FAMILIES = (
    "qwen3asr_plus_qwen3vl_path",
    "qwen3asr_plus_gemma4_path",
    "model_intersection",
    "model_union",
    "conservative_envelope",
)
COMPARATORS = (
    "weak_vl_absent_side",
    "weak_vl_group_shuffled_side",
    "weak_vl_balanced_time_shifted_side",
    "weak_vl_corrupted_uninformative_side",
)
ENDPOINTS = (
    "cue_free_heldout_noun_object_macro_top1",
    "cue_free_heldout_verb_action_macro_top1",
)
BINDING_KEYS = frozenset(
    {
        "no_automatic_fallback_override_sha256",
        "protocol_sha256",
        "execution_contract_sha256",
        "public_calibration_receipt_sha256",
        "preoutcome_receipt_sha256",
        "one_shot_authorization_sha256",
        "aggregate_results_sha256",
        "execution_receipt_sha256",
    }
)
VALIDITY_KEYS = frozenset(
    {
        "gemma_path_passed",
        "automatic_fallback_used",
        "scientific_outcome_run",
        "fixture_only",
        "preoutcome_all_passed",
        "one_shot_complete",
        "complete_paired_bundles",
        "cue_free_evaluation",
        "simulator_oracle_only",
        "privacy_passed",
        "ancestry_passed",
        "restricted_payload_absent",
        "item_level_rows_absent",
    }
)
THRESHOLDS = {
    "minimum_mean_effect": 0.03,
    "lower_bound_must_exceed": 0.0,
    "minimum_positive_corpus_fraction": 0.6,
    "maximum_large_negative_fraction": 0.2,
    "large_negative_threshold": -0.05,
    "one_sided_t_critical_95_df39": 1.684875,
}
CELL_KEYS = frozenset(
    {
        "synthetic_corpus_count",
        "mean_effect",
        "one_sided_95_lower",
        "positive_corpus_fraction",
        "exact_one_sided_sign_p",
        "large_negative_fraction",
        "passed",
    }
)
FALSIFICATION_KEYS = frozenset(
    {
        "confidence_only_rule_passed",
        "side_only_leakage_passed",
        "event_selection_disconnection_passed",
    }
)
TOP_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "bindings",
        "validity",
        "thresholds",
        "contrasts",
        "falsifications",
    }
)


class ReportingError(RuntimeError):
    pass


def fail(code: str) -> None:
    raise ReportingError(code)


def exact_keys(value: Any, expected: frozenset[str] | set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(str(key) for key in value) != frozenset(expected):
        fail(code)
    return value


def finite_number(value: Any, *, minimum: float, maximum: float, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(code)
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        fail(code)
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                result.update(block)
    except OSError:
        fail("E_INPUT_READ")
    return result.hexdigest()


def read_input(path: Path) -> Mapping[str, Any]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > 2 * 1024 * 1024
        ):
            fail("E_INPUT_FILE")
        payload = path.read_bytes()
        document = json.loads(payload)
    except ReportingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        fail("E_INPUT_READ")
    if not isinstance(document, Mapping):
        fail("E_INPUT_SCHEMA")
    return document


def sign_p_value(successes: int, total: int = 40) -> float:
    return sum(math.comb(total, k) for k in range(successes, total + 1)) / (2**total)


def validate_projection(document: Mapping[str, Any]) -> tuple[str, int, list[tuple[str, str, str, Mapping[str, Any]]]]:
    exact_keys(document, TOP_KEYS, "E_TOP_SCHEMA")
    if document.get("schema_version") != INPUT_SCHEMA or document.get("status") != "SCIENTIFIC_AGGREGATE_VALIDATED":
        fail("E_INPUT_STATUS")

    bindings = exact_keys(document.get("bindings"), BINDING_KEYS, "E_BINDING_SCHEMA")
    for value in bindings.values():
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            fail("E_BINDING_VALUE")
    if bindings["no_automatic_fallback_override_sha256"] != OVERRIDE_SHA256:
        fail("E_FALLBACK_OVERRIDE_BINDING")

    validity = exact_keys(document.get("validity"), VALIDITY_KEYS, "E_VALIDITY_SCHEMA")
    required_true = VALIDITY_KEYS - {"automatic_fallback_used", "fixture_only"}
    if any(validity[key] is not True for key in required_true):
        fail("E_VALIDITY_GATE")
    if validity["automatic_fallback_used"] is not False or validity["fixture_only"] is not False:
        fail("E_VALIDITY_GATE")

    thresholds = exact_keys(document.get("thresholds"), set(THRESHOLDS), "E_THRESHOLD_SCHEMA")
    for key, expected in THRESHOLDS.items():
        observed = finite_number(thresholds[key], minimum=-10.0, maximum=10.0, code="E_THRESHOLD_VALUE")
        if observed != expected:
            fail("E_THRESHOLD_BINDING")

    falsifications = exact_keys(document.get("falsifications"), FALSIFICATION_KEYS, "E_FALSIFICATION_SCHEMA")
    if any(falsifications[key] is not True for key in FALSIFICATION_KEYS):
        fail("E_FALSIFICATION_GATE")

    contrasts = exact_keys(document.get("contrasts"), set(FAMILIES), "E_FAMILY_SCHEMA")
    cells: list[tuple[str, str, str, Mapping[str, Any]]] = []
    pass_count = 0
    for family in FAMILIES:
        comparators = exact_keys(contrasts[family], set(COMPARATORS), "E_COMPARATOR_SCHEMA")
        for comparator in COMPARATORS:
            endpoints = exact_keys(comparators[comparator], set(ENDPOINTS), "E_ENDPOINT_SCHEMA")
            for endpoint in ENDPOINTS:
                cell = exact_keys(endpoints[endpoint], CELL_KEYS, "E_CELL_SCHEMA")
                if cell["synthetic_corpus_count"] != 40 or isinstance(cell["synthetic_corpus_count"], bool):
                    fail("E_SYNTHETIC_CORPUS_COUNT")
                mean = finite_number(cell["mean_effect"], minimum=-1.0, maximum=1.0, code="E_EFFECT_VALUE")
                lower = finite_number(cell["one_sided_95_lower"], minimum=-1.0, maximum=1.0, code="E_EFFECT_VALUE")
                positive = finite_number(cell["positive_corpus_fraction"], minimum=0.0, maximum=1.0, code="E_FRACTION_VALUE")
                sign_p = finite_number(cell["exact_one_sided_sign_p"], minimum=0.0, maximum=1.0, code="E_FRACTION_VALUE")
                large_negative = finite_number(cell["large_negative_fraction"], minimum=0.0, maximum=1.0, code="E_FRACTION_VALUE")
                if lower > mean + 1e-12:
                    fail("E_LOWER_BOUND")
                positive_count = round(positive * 40)
                negative_count = round(large_negative * 40)
                if abs(positive - positive_count / 40) > 1e-12 or abs(large_negative - negative_count / 40) > 1e-12:
                    fail("E_FRACTION_GRID")
                if abs(sign_p - sign_p_value(positive_count)) > 1e-12:
                    fail("E_SIGN_P")
                computed = (
                    mean >= THRESHOLDS["minimum_mean_effect"]
                    and lower > THRESHOLDS["lower_bound_must_exceed"]
                    and positive >= THRESHOLDS["minimum_positive_corpus_fraction"]
                    and large_negative <= THRESHOLDS["maximum_large_negative_fraction"]
                )
                if not isinstance(cell["passed"], bool) or cell["passed"] is not computed:
                    fail("E_CELL_PASS_MISMATCH")
                pass_count += int(computed)
                cells.append((family, comparator, endpoint, cell))
    if len(cells) != 40:
        fail("E_CELL_INVENTORY")
    decision = "POSITIVE_PROTOTYPE_EFFECT" if pass_count == 40 else "VALID_NULL_OR_NONPROMOTION"
    return decision, pass_count, cells


def label_family(value: str) -> str:
    return {
        "qwen3asr_plus_qwen3vl_path": "Qwen ASR + Qwen VL",
        "qwen3asr_plus_gemma4_path": "Qwen ASR + Gemma 4",
        "model_intersection": "Model intersection",
        "model_union": "Model union",
        "conservative_envelope": "Conservative envelope",
    }[value]


def label_comparator(value: str) -> str:
    return {
        "weak_vl_absent_side": "Absent",
        "weak_vl_group_shuffled_side": "Shuffled",
        "weak_vl_balanced_time_shifted_side": "Time-shifted",
        "weak_vl_corrupted_uninformative_side": "Corrupted",
    }[value]


def render_svg(decision: str, pass_count: int, cells: list[tuple[str, str, str, Mapping[str, Any]]]) -> bytes:
    width, height = 1500, 1100
    panel_x = {ENDPOINTS[0]: 360, ENDPOINTS[1]: 940}
    panel_w = 450
    x_min, x_max = -1.0, 1.0
    observed = [float(cell[key]) for _, _, _, cell in cells for key in ("mean_effect", "one_sided_95_lower")]
    x_min = min(-0.05, min(observed) - 0.03)
    x_max = max(0.08, max(observed) + 0.03)
    span = x_max - x_min

    def xpos(endpoint: str, value: float) -> float:
        return panel_x[endpoint] + (value - x_min) / span * panel_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#18212b}.title{font-size:27px;font-weight:700}.sub{font-size:16px;fill:#4d5a68}.panel{font-size:19px;font-weight:700}.row{font-size:11px}.family{font-size:12px;font-weight:700}.axis{stroke:#8d98a5;stroke-width:1}.zero{stroke:#1e2933;stroke-width:1.5}.gate{stroke:#be6a15;stroke-width:1.5;stroke-dasharray:5 4}.pass{fill:#246b5f;stroke:#16483f;stroke-width:1}.fail{fill:#fbfaf7;stroke:#a33d31;stroke-width:2}.ci{stroke:#566573;stroke-width:2}.badge{font-size:9px;fill:#4d5a68}.footer{font-size:12px;fill:#4d5a68}.banner{font-size:16px;font-weight:700}</style>',
        '<text x="50" y="45" class="title">Do synchronized training-only event cues survive the bounded weak-alignment envelope?</text>',
        '<text x="50" y="73" class="sub">Cue-free simulator-oracle grounding • synchronized minus matched control</text>',
        f'<rect x="50" y="90" width="1400" height="42" rx="7" fill="#{"dcefe9" if decision == "POSITIVE_PROTOTYPE_EFFECT" else "f4e4df"}"/>',
        f'<text x="70" y="117" class="banner">{html.escape(decision)} • {pass_count}/40 directional cells pass</text>',
        '<text x="360" y="162" class="panel">Noun → object</text>',
        '<text x="940" y="162" class="panel">Verb → action</text>',
    ]
    for endpoint in ENDPOINTS:
        for value, css in ((0.0, "zero"), (0.03, "gate")):
            x = xpos(endpoint, value)
            parts.append(f'<line x1="{x:.2f}" y1="180" x2="{x:.2f}" y2="990" class="{css}"/>')
        parts.append(f'<text x="{xpos(endpoint,0.0)-8:.2f}" y="1010" class="row">0</text>')
        parts.append(f'<text x="{xpos(endpoint,0.03)-18:.2f}" y="1028" class="row">+3 pp gate</text>')

    lookup = {(family, comparator, endpoint): cell for family, comparator, endpoint, cell in cells}
    y = 195
    for family in FAMILIES:
        parts.append(f'<text x="50" y="{y+11}" class="family">{html.escape(label_family(family))}</text>')
        for comparator in COMPARATORS:
            y += 36
            parts.append(f'<text x="105" y="{y+4}" class="row">{html.escape(label_comparator(comparator))}</text>')
            for endpoint in ENDPOINTS:
                cell = lookup[(family, comparator, endpoint)]
                mean = float(cell["mean_effect"])
                lower = float(cell["one_sided_95_lower"])
                x0, x1 = xpos(endpoint, lower), xpos(endpoint, mean)
                parts.append(f'<line x1="{x0:.2f}" y1="{y}" x2="{x1:.2f}" y2="{y}" class="ci"/>')
                css = "pass" if cell["passed"] else "fail"
                parts.append(f'<circle cx="{x1:.2f}" cy="{y}" r="5" class="{css}"/>')
                pos = 100 * float(cell["positive_corpus_fraction"])
                neg = 100 * float(cell["large_negative_fraction"])
                parts.append(f'<text x="{panel_x[endpoint]+panel_w+8}" y="{y+3}" class="badge">+{pos:.0f}% / −{neg:.0f}%</text>')
        y += 18
    parts.extend(
        [
            '<text x="50" y="1048" class="footer">Point = mean paired lift; line = one-sided 95% lower bound. Badge = positive-corpus % / large-negative %.</text>',
            '<text x="50" y="1073" class="footer">Synthetic prototype only • cue absent at evaluation • simulator-oracle truth • ChildLens aggregate calibration is model-derived, not human gold • no infant or naturalistic acquisition claim</text>',
            '</svg>',
        ]
    )
    return ("\n".join(parts) + "\n").encode("utf-8")


def exclusive_write(path: Path, payload: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        fail("E_OUTPUT_EXISTS")
    except OSError:
        fail("E_OUTPUT_WRITE")


def generate(input_path: Path, svg_path: Path, summary_path: Path) -> Mapping[str, Any]:
    if svg_path == summary_path or svg_path.exists() or summary_path.exists():
        fail("E_OUTPUT_EXISTS")
    if not svg_path.parent.is_dir() or not summary_path.parent.is_dir():
        fail("E_OUTPUT_PARENT")
    document = read_input(input_path)
    decision, pass_count, cells = validate_projection(document)
    svg = render_svg(decision, pass_count, cells)
    first_failure = next(
        (
            {"range_family": family, "comparator": comparator, "endpoint": endpoint}
            for family, comparator, endpoint, cell in cells
            if not cell["passed"]
        ),
        None,
    )
    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "REPORTING_SHELL_RENDERED_FROM_VALIDATED_AGGREGATES",
        "decision": decision,
        "source_aggregate_projection_sha256": sha256_file(input_path),
        "no_automatic_fallback_override_sha256": OVERRIDE_SHA256,
        "passing_directional_cells": pass_count,
        "total_directional_cells": 40,
        "first_nonpassing_cell": first_failure,
        "scientific_outcome_values_hardcoded": False,
        "restricted_content_accessed": False,
        "automatic_fallback_used": False,
        "svg_sha256": sha256_bytes(svg),
    }
    summary_payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    exclusive_write(svg_path, svg)
    try:
        exclusive_write(summary_path, summary_payload)
    except ReportingError:
        try:
            svg_path.unlink()
        except OSError:
            pass
        raise
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-input", required=True, type=Path)
    parser.add_argument("--output-svg", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = generate(args.aggregate_input, args.output_svg, args.output_summary)
    except ReportingError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "decision": summary["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
