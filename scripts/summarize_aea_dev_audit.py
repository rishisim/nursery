#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


VALID_LABELS = {
    "clear_match",
    "plausible_or_ambiguous",
    "mismatch",
    "not_visually_judgeable",
}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {"successes": successes, "total": total, "rate": None, "ci95_low": None, "ci95_high": None}
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "ci95_low": max(0.0, center - margin),
        "ci95_high": min(1.0, center + margin),
        "method": "two-sided 95% Wilson interval",
    }


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    labels = Counter(str(row["audit_label"]) for row in values)
    total = len(values)
    judgeable = total - labels["not_visually_judgeable"]
    clear_or_plausible = labels["clear_match"] + labels["plausible_or_ambiguous"]
    return {
        "n": total,
        "label_counts": {key: labels[key] for key in sorted(VALID_LABELS)},
        "judgeable": wilson(judgeable, total),
        "clear_match": wilson(labels["clear_match"], total),
        "clear_or_plausible": wilson(clear_or_plausible, total),
        "mismatch": wilson(labels["mismatch"], total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prelabel", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels_doc = json.loads(args.labels.read_text())
    prelabel_doc = json.loads(args.prelabel.read_text())
    protocol = json.loads(args.protocol.read_text())
    rows = list(labels_doc["rows"])
    expected = {str(row["example_id"]) for row in prelabel_doc["rows"]}
    observed = {str(row["example_id"]) for row in rows}
    if len(rows) != int(protocol["audit"]["sample_size"]) or observed != expected:
        raise ValueError("audit labels do not match the frozen prelabel manifest")
    if any(str(row["audit_label"]) not in VALID_LABELS or not str(row["rationale"]).strip() for row in rows):
        raise ValueError("audit labels must be complete and valid")
    reserve = set(map(str, protocol["confirmation_event_groups"]))
    if any(str(row["event_group"]) in reserve for row in rows):
        raise AssertionError("confirmation group entered audit labels")
    development = [json.loads(line) for line in args.development_examples.read_text().splitlines() if line.strip()]
    action_groups: dict[str, set[str]] = defaultdict(set)
    action_counts: Counter[str] = Counter()
    for row in development:
        action = str(row["evaluation_targets"]["action_verb"])
        action_counts[action] += 1
        action_groups[action].add(str(row["event_group"]))
    modeled_high_motion = {
        action for action in protocol["semantic_high_motion_actions"]
        if action_counts[action] >= int(protocol["support"]["minimum_windows"])
        and len(action_groups[action]) >= int(protocol["support"]["minimum_event_groups"])
    }
    inverse_coarse = {
        action: category
        for category, actions in protocol["coarse_mapping"].items()
        for action in actions
    }
    by_action = {
        action: summarize_rows([row for row in rows if row["action_verb"] == action])
        for action in sorted({str(row["action_verb"]) for row in rows})
    }
    by_coarse = {
        category: summarize_rows([row for row in rows if inverse_coarse.get(str(row["action_verb"])) == category])
        for category in protocol["coarse_mapping"]
    }
    high_motion_rows = [row for row in rows if str(row["action_verb"]) in set(protocol["semantic_high_motion_actions"])]
    modeled_high_motion_rows = [row for row in rows if str(row["action_verb"]) in modeled_high_motion]
    overall = summarize_rows(rows)
    thresholds = protocol["audit"]
    gates = {
        "judgeable_pass": overall["judgeable"]["rate"] >= float(thresholds["minimum_judgeable"]),
        "clear_or_plausible_pass": overall["clear_or_plausible"]["ci95_low"] >= float(thresholds["minimum_clear_or_plausible_ci_low"]),
        "mismatch_pass": overall["mismatch"]["ci95_high"] <= float(thresholds["maximum_mismatch_ci_high"]),
    }
    result = {
        "schema_version": "aea-dev-audit-summary-v1",
        "protocol_id": protocol["protocol_id"],
        "scope": "development_only",
        "confirmation_groups_inspected": [],
        "coverage": {
            "actions": len(by_action),
            "event_groups": len({str(row["event_group"]) for row in rows}),
            "locations": sorted({int(row["location"]) for row in rows}),
            "frames_per_window": 8,
        },
        "overall": overall,
        "by_action": by_action,
        "by_coarse_category": by_coarse,
        "semantic_high_motion": summarize_rows(high_motion_rows),
        "modeled_semantic_high_motion_actions": sorted(modeled_high_motion),
        "modeled_semantic_high_motion": summarize_rows(modeled_high_motion_rows),
        "gates": {**gates, "passed": all(gates.values())},
        "limitations": [
            "single auditor",
            "eight sparse frames rather than continuous video",
            "plausible category retains timing and agency ambiguity",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
