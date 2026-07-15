from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Mapping


def _pp(value: float) -> float:
    return round(100.0 * float(value), 4)


def _interval_text(interval: Mapping[str, Any] | None) -> str:
    if not interval:
        return "not estimated"
    return (
        f"{_pp(interval['mean']):+.2f} pp "
        f"(95% CI {_pp(interval['ci95_low']):+.2f}, "
        f"{_pp(interval['ci95_high']):+.2f})"
    )


def build_report_artifact(
    result: Mapping[str, Any], preprocess: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the canonical bounded report payload used by the portable reader."""
    generated_at = datetime.now(timezone.utc).isoformat()
    primary = result["primary_estimand"]
    interval = primary.get("interval")
    data = result["data_summary"]
    raw_location_rows: list[tuple[Any, ...]] = []
    for split in result.get("evaluation_results", []):
        if split.get("family") != "held_out_location":
            continue
        differences = [
            float(value)
            for value in split["paired_seed_differences"][
                "synchronized_minus_shuffled"
            ].values()
        ]
        for seed, difference in zip(
            split["paired_seed_differences"]["synchronized_minus_shuffled"],
            differences,
        ):
            raw_location_rows.append((
                str(split["name"]),
                int(split["split_audit"]["held_out_location"]),
                int(seed),
                float(difference),
                int(split["split_audit"]["test_count"]),
                len(split["actions"]),
                int(bool(split["split_audit"]["valid"])),
            ))
    location_sql = """SELECT
  split,
  held_out_location,
  ROUND(100.0 * AVG(lift), 4) AS mean_lift_pp,
  ROUND(100.0 * MIN(lift), 4) AS minimum_seed_lift_pp,
  ROUND(100.0 * MAX(lift), 4) AS maximum_seed_lift_pp,
  COUNT(*) AS paired_seed_count,
  MAX(test_windows) AS test_windows,
  MAX(supported_action_count) AS supported_action_count,
  MIN(audit_valid) AS audit_valid
FROM paired_location_lifts
GROUP BY split, held_out_location
ORDER BY held_out_location"""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE paired_location_lifts ("
        "split TEXT, held_out_location INTEGER, seed INTEGER, lift REAL, "
        "test_windows INTEGER, supported_action_count INTEGER, audit_valid INTEGER)"
    )
    connection.executemany(
        "INSERT INTO paired_location_lifts VALUES (?, ?, ?, ?, ?, ?, ?)",
        raw_location_rows,
    )
    location_rows = [dict(row) for row in connection.execute(location_sql)]
    connection.close()

    family_rows = []
    for family, summary in sorted(result.get("family_summaries", {}).items()):
        lift = summary["paired_lift"]
        family_rows.append({
            "family": str(family),
            "split_count": int(summary["split_count"]),
            "mean_lift_pp": _pp(lift["mean"]),
            "ci95_low_pp": _pp(lift["ci95_low"]),
            "ci95_high_pp": _pp(lift["ci95_high"]),
            "paired_seed_count": int(lift["n_paired_seeds"]),
            "split_unit_count": int(lift["n_split_units"]),
        })

    headline_rows = [{
        "windows": int(data["windows"]),
        "recordings": int(data["sequences"]),
        "locations": len(data["locations"]),
        "primary_lift_pp": _pp(interval["mean"]) if interval else None,
        "primary_ci_low_pp": _pp(interval["ci95_low"]) if interval else None,
        "primary_ci_high_pp": _pp(interval["ci95_high"]) if interval else None,
        "claim_gate_passed": bool(primary["claim_gate_passed"]),
    }]
    result_audit_rows = [
        {"check": "Leakage and pairing gates", "passed": bool(primary["audit_gates_passed"])},
        {"check": "Locked training configuration", "passed": bool(primary["locked_training_configuration_match"])},
        {"check": "Complete locked folds and seeds", "passed": bool(primary["complete_locked_location_folds_and_seed_schedule"])},
        {"check": "IMU omitted from primary test", "passed": result["primary_test"]["motor"].startswith("withheld")},
        {"check": "Claim gate", "passed": bool(primary["claim_gate_passed"])},
    ]
    preprocess_rows: list[dict[str, Any]] = []
    if preprocess:
        audit = preprocess.get("audit", {})
        quality = audit.get("quality", {})
        preprocess_rows = [{
            "acquisition_complete": bool(preprocess.get("acquisition_complete")),
            "plan_recordings": int(preprocess.get("sequence_plan_count", 0)),
            "available_vrs_recordings": int(preprocess.get("source_vrs_present_count", 0)),
            "processed_recordings": int(preprocess.get("sequence_processed_count", 0)),
            "accepted_windows": int(preprocess.get("window_count", 0)),
            "audit_valid": bool(audit.get("valid")),
            "minimum_imu_coverage": quality.get("minimum_imu_coverage"),
            "maximum_frame_error_ms": quality.get("maximum_frame_time_error_ms"),
            "imu_error_count": quality.get("imu_error_count"),
            "frame_error_count": quality.get("frame_error_count"),
        }]

    family_table = [
        "| Family | Splits | Mean lift, pp | 95% CI, pp | Paired seeds |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['family']} | {row['split_count']} | {row['mean_lift_pp']:+.2f} | "
            f"[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}] | "
            f"{row['paired_seed_count']} |"
            for row in family_rows
        ],
    ]
    result_audit_table = [
        "| Eligibility check | Passed |",
        "|---|---:|",
        *[
            f"| {row['check']} | {'yes' if row['passed'] else 'no'} |"
            for row in result_audit_rows
        ],
    ]
    preprocess_table: list[str] = []
    if preprocess_rows:
        row = preprocess_rows[0]
        preprocess_table = [
            "| Planned | Available VRS | Processed | Accepted windows | Minimum IMU coverage | Max RGB error | Sensor/file errors |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {row['plan_recordings']} | {row['available_vrs_recordings']} | "
            f"{row['processed_recordings']} | {row['accepted_windows']} | "
            f"{100 * float(row['minimum_imu_coverage']):.2f}% | "
            f"{float(row['maximum_frame_error_ms']):.2f} ms | "
            f"{int(row['imu_error_count']) + int(row['frame_error_count'])} |",
        ]

    sources = [
        {
            "id": "experiment-results",
            "label": "AEA four-arm experiment results",
            "path": "aea_results.json",
        },
        {
            "id": "location-lift-query",
            "label": "Held-out-location paired lift aggregation",
            "path": "aea_results.json",
            "query": {
                "engine": "sqlite",
                "sql": location_sql,
                "description": "Aggregates paired synchronized-minus-shuffled seed differences within held-out location folds.",
                "tables_used": ["paired_location_lifts"],
                "filters": [
                    "family = held_out_location",
                    "paired synchronized and shuffled seed cells only",
                ],
                "metric_definitions": [
                    "mean_lift_pp = 100 × AVG(synchronized_accuracy − shuffled_accuracy)"
                ],
            },
        },
        {
            "id": "locked-protocol",
            "label": "Locked AEA protocol",
            "path": "configs/aea_real.yaml",
        },
    ]
    if preprocess:
        sources.append({
            "id": "preprocess-audit",
            "label": "AEA preprocessing and sensor audit",
            "path": "preprocess_summary.json",
        })

    status = str(result["scientific_status"])
    status_explanation = (
        "This is an infrastructure smoke test on real AEA windows, not a real-data effect finding."
        if status == "infrastructure_smoke_test_not_a_real_data_finding"
        else "This is a real AEA effect estimate under the locked protocol."
        if status == "real_aea_effect_estimate"
        else "This run is incomplete and is not a primary finding."
    )
    title = "Nursery AEA real-data phase"
    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
        {
            "id": "technical-summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": "experiment-results",
            "body": (
                "## Technical summary\n\n"
                f"**Status:** `{status}`. {status_explanation}\n\n"
                f"**Data represented:** {data['windows']} accepted windows from "
                f"{data['sequences']} recordings across {len(data['locations'])} locations.\n\n"
                f"**Primary synchronized − event-shuffled estimate:** "
                f"{_interval_text(interval)}. The claim gate "
                f"{'passed' if primary['claim_gate_passed'] else 'did not pass'}."
            ),
        },
        {
            "id": "location-finding",
            "type": "markdown",
            "layout": "full",
            "sourceId": "experiment-results",
            "body": (
                "## The location-fold pattern is diagnostic, not developmental evidence\n\n"
                "Each bar is the mean paired synchronized-minus-shuffled difference within one held-out location. "
                "Positive values favor synchronized IMU during training; the zero line marks no lift. "
                "The chart is a smoke diagnostic whenever the report status says so, regardless of sign."
            ),
        },
        {"id": "location-chart-block", "type": "chart", "chartId": "location-lift-chart", "layout": "full"},
        {
            "id": "scope-and-definitions",
            "type": "markdown",
            "layout": "full",
            "sourceId": "locked-protocol",
            "body": (
                "## Scope, population, and metric definitions\n\n"
                "AEA contains adult, partly scripted activities and is used only as a sensor-format analogue. "
                "It is not developmental evidence and is not BabyView-matched. The primary endpoint is action-balanced "
                "2AFC macro accuracy on RGB plus transcript with IMU absent from the evaluation batch. The primary "
                "estimand subtracts a split-local whole-event-shuffled IMU arm from the correctly synchronized arm."
            ),
        },
        {
            "id": "family-evidence",
            "type": "markdown",
            "layout": "full",
            "sourceId": "experiment-results",
            "body": (
                "## Evaluation families stay separate\n\n"
                "Held-out location is primary. Wearer-session-proxy and action-object-composition results are secondary "
                "and retain separate uncertainty intervals so they cannot silently replace the locked endpoint.\n\n"
                + "\n".join(family_table)
            ),
        },
        {
            "id": "experimental-design",
            "type": "markdown",
            "layout": "full",
            "sourceId": "locked-protocol",
            "body": (
                "## Four arms isolate synchronization during training\n\n"
                "Null, synchronized, split-local event-shuffled, and time-shifted arms share architecture, initialization, "
                "batch order, optimizer schedule, and update count within paired seeds. Concurrent recordings of the same "
                "performed event share a donor group. The primary test path constructs no motor tensor and never calls the "
                "motor encoder. Confidence intervals resample held-out split units and paired seeds hierarchically."
            ),
        },
        {
            "id": "result-audit",
            "type": "markdown",
            "layout": "full",
            "sourceId": "experiment-results",
            "body": (
                "## Claim controls remain closed unless every lock passes\n\n"
                "The audit below checks the leakage, pairing, training-schedule, completion, and motor-withholding conditions "
                "that determine whether an estimate is eligible for interpretation.\n\n"
                + "\n".join(result_audit_table)
            ),
        },
    ]
    if preprocess:
        blocks.extend([
            {
                "id": "sensor-audit",
                "type": "markdown",
                "layout": "full",
                "sourceId": "preprocess-audit",
                "body": (
                    "## Sensor coverage is audited before modeling\n\n"
                    "The preprocessing audit checks acquisition completeness, finite six-axis arrays, fixed resampled shape, "
                    "RGB timestamp error, allowlisted model inputs, and absence of unnecessary MPS features.\n\n"
                    + "\n".join(preprocess_table)
                ),
            },
        ])
    blocks.extend([
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Limitations prevent a developmental claim\n\n"
                "Action targets are ASR lexical anchors rather than human action annotations; object targets are nearby lexical "
                "items and may not name the manipulated object. The wearer split uses a release-visible location+script+recording "
                "proxy, not persistent identity. Location 5 covers only scripts 4 and 5. AEA cannot establish a benefit in infants "
                "or in natural BabyView data."
            ),
        },
        {
            "id": "next-steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Recommended next steps\n\n"
                "1. Treat smoke numbers only as pipeline validation.\n"
                "2. Run the complete locked five-fold, seven-seed, eight-epoch primary schedule before using `real_aea_effect_estimate`.\n"
                "3. Inspect positive controls and per-action support before interpreting any aggregate.\n"
                "4. Use the AEA result to refine the apples-to-apples BabyView protocol, not as developmental evidence."
            ),
        },
        {
            "id": "further-questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "Would human-verified action/object annotations change the estimate? Does lift vary with head-motion intensity, "
                "speech timing, or action class? Which sensor-format differences must be calibrated before an eventual BabyView comparison?"
            ),
        },
    ])

    cards: list[dict[str, Any]] = []
    charts = [{
        "id": "location-lift-chart",
        "title": "Synchronized minus event-shuffled lift by held-out location",
        "subtitle": "Mean paired-seed difference in action-balanced 2AFC accuracy, percentage points",
        "intent": "comparison",
        "question": "Is the synchronized-training lift consistent across held-out locations?",
        "rationale": "Five discrete location folds are best compared as bars around an explicit zero reference.",
        "comparisonContext": {
            "baseline": "split-local whole-performed-event-shuffled IMU",
            "grain": "held-out location fold",
            "unit": "percentage points",
        },
        "type": "bar",
        "dataset": "location_lifts",
        "sourceId": "location-lift-query",
        "encodings": {
            "x": {"field": "held_out_location", "type": "ordinal", "label": "Held-out location"},
            "y": {"field": "mean_lift_pp", "type": "quantitative", "label": "Lift", "unit": "pp", "format": "number"},
            "tooltip": [
                {"field": "minimum_seed_lift_pp", "type": "quantitative", "label": "Minimum paired-seed lift", "unit": "pp"},
                {"field": "maximum_seed_lift_pp", "type": "quantitative", "label": "Maximum paired-seed lift", "unit": "pp"},
                {"field": "paired_seed_count", "type": "quantitative", "label": "Paired seeds"},
                {"field": "test_windows", "type": "quantitative", "label": "Test windows"},
            ],
        },
        "valueFormat": "number",
        "unit": "pp",
        "layout": "full",
        "labels": {"values": "all"},
        "palette": {"kind": "diverging", "midpoint": 0},
        "referenceLines": [{"axis": "y", "value": 0, "label": "No lift", "color": "neutral", "lineStyle": "solid"}],
        "settings": {"sort": "custom", "showValues": True},
        "surface": {"surface": "card", "showControls": False, "viewMode": "visualization"},
    }]
    tables: list[dict[str, Any]] = []

    snapshot_datasets = {
        "headline": headline_rows,
        "location_lifts": location_rows,
        "family_summaries": family_rows,
        "result_audits": result_audit_rows,
    }
    if preprocess:
        snapshot_datasets["preprocess_audit"] = preprocess_rows
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical report for the Nursery/BabyWorld AEA four-arm real-data phase.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "fixture" if status == "infrastructure_smoke_test_not_a_real_data_finding" else "ready",
            "datasets": snapshot_datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }
    notes = {
        "audience": "technical",
        "delivery_mode": "portable_html",
        "required_structure_mapping": {
            "title": "title",
            "technical_summary": "technical-summary",
            "key_findings_with_visual_evidence": ["location-finding", "location-chart-block", "family-evidence"],
            "scope_data_metric_definitions": "scope-and-definitions",
            "methodology_and_experimental_design": "experimental-design",
            "limitations_uncertainty_robustness": ["result-audit", "limitations"],
            "recommended_next_steps": "next-steps",
            "further_questions": "further-questions",
        },
        "chart_map": [{
            "section": "location-finding",
            "question": "Is synchronized-minus-shuffled lift consistent across held-out locations?",
            "family": "Comparison & Ranking",
            "type": "bar",
            "fields": ["held_out_location", "mean_lift_pp"],
            "takeaway": "Shows fold heterogeneity around a zero reference without treating a smoke sign as a finding.",
            "palette_policy": "single diverging root around zero with labels and a neutral reference line",
            "delivery": "native canonical artifact chart",
        }],
        "visual_omissions": [
            "Family intervals are an exact audit table because there are too few family aggregates for an additional honest chart."
        ],
        "source_inventory": [source["id"] for source in sources],
        "metric_definitions": [
            "lift_pp = 100 × (synchronized action-balanced 2AFC accuracy − shuffled accuracy)",
            "95% CI uses hierarchical resampling over split units and paired seeds",
        ],
        "source_filters": [
            "primary family is leave-one-location-out",
            "primary test uses RGB and transcript only",
            "synchronized and shuffled arms are paired within seed and split",
        ],
    }
    return artifact, notes


def write_report_artifact(
    result: Mapping[str, Any], preprocess: Mapping[str, Any] | None, output: str | Path
) -> tuple[Path, Path]:
    import json

    artifact, notes = build_report_artifact(result, preprocess)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2))
    notes_path = path.with_name("report_source_notes.json")
    notes_path.write_text(json.dumps(notes, indent=2))
    return path, notes_path
