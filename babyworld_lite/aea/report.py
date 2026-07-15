from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _fmt(interval: Mapping[str, Any] | None) -> str:
    if not interval:
        return "not estimated"
    return (
        f"{100 * float(interval['mean']):+.2f} pp "
        f"(95% CI {100 * float(interval['ci95_low']):+.2f}, "
        f"{100 * float(interval['ci95_high']):+.2f})"
    )


def aea_report_markdown(result: Mapping[str, Any]) -> str:
    primary = result.get("primary_estimand", {})
    data = result.get("data_summary", {})
    lines = [
        "# Nursery AEA real-data phase report",
        "",
        f"**Status:** {result['scientific_status']}",
        "",
        f"Data represented in this run: {data.get('windows', 'unknown')} windows from "
        f"{data.get('sequences', 'unknown')} recordings across "
        f"{len(data.get('locations', {}))} locations.",
        "",
        "AEA is an adult, partly scripted sensor-format analogue. It is not developmental evidence and is not represented as BabyView-matched.",
        "",
        "## Primary question",
        "",
        "Does correctly synchronized six-axis head IMU during training improve motor-withheld video-language action grounding relative to split-local, whole-episode-shuffled IMU?",
        "",
        f"Primary synchronized − episode-shuffled estimate: **{_fmt(primary.get('interval'))}**.",
        "",
        f"Claim gate: **{'passed' if primary.get('claim_gate_passed') else 'not passed'}**.",
        "",
        "## Protocol safeguards",
        "",
        "- Accelerometer and gyroscope are resampled together on a fixed grid in SI units.",
        "- Whole windows stay with their source sequence; concurrent views are grouped or purged.",
        "- Shuffled IMU donors are split-local, come from a different performed-event group (including concurrent partner recordings), and form a permutation.",
        "- Seeds, initialization, batch order, optimizer schedule, and test inputs are paired across arms.",
        "- Locked training configuration match: "
        f"{bool(primary.get('locked_training_configuration_match'))}"
        + (" (expected for this smoke configuration)." if result.get("scientific_status") == "infrastructure_smoke_test_not_a_real_data_finding" else "."),
        "- The primary evaluator never loads IMU or calls the motor encoder.",
        "",
        "## Evaluation families",
        "",
        "| Family | Splits run | Synchronized − shuffled |",
        "|---|---:|---:|",
    ]
    for family, value in result.get("family_summaries", {}).items():
        lines.append(f"| {family} | {value['split_count']} | {_fmt(value.get('paired_lift'))} |")
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "Action labels are noisy ASR lexical anchors, object labels are nearby ASR lexical items, and the wearer split uses a release-visible location+script+recording proxy rather than persistent identity. Smoke-test numbers validate plumbing only. Real-data estimates are findings only when the report status explicitly says so and all audit gates pass.",
        "",
    ])
    return "\n".join(lines)


def write_aea_report(result: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(aea_report_markdown(result))
