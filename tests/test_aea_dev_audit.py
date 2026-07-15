from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_aea_dev_partition import audit_sample
from scripts.summarize_aea_dev_audit import summarize_rows, wilson


ROOT = Path(__file__).resolve().parents[1]


def test_audit_sample_is_deterministic_and_group_aware() -> None:
    rows = [
        {
            "example_id": f"row-{index}",
            "event_group": f"group-{index % 3}",
            "evaluation_targets": {"action_verb": "get" if index < 5 else "put"},
        }
        for index in range(10)
    ]
    first = audit_sample(rows, 6, "test-salt")
    second = audit_sample(rows, 6, "test-salt")
    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
    assert {row["evaluation_targets"]["action_verb"] for row in first} == {"get", "put"}
    assert len({row["event_group"] for row in first}) == 3


def test_wilson_and_audit_summary_denominators() -> None:
    interval = wilson(14, 48)
    assert interval["rate"] == 14 / 48
    assert 0.18 < interval["ci95_low"] < 0.19
    assert 0.43 < interval["ci95_high"] < 0.44
    rows = [
        {"audit_label": "clear_match"},
        {"audit_label": "plausible_or_ambiguous"},
        {"audit_label": "mismatch"},
        {"audit_label": "not_visually_judgeable"},
    ]
    summary = summarize_rows(rows)
    assert summary["judgeable"]["successes"] == 3
    assert summary["clear_or_plausible"]["successes"] == 2
    assert summary["mismatch"]["total"] == 4


def test_frozen_manifests_exclude_confirmation_from_development_and_audit() -> None:
    protocol = json.loads((ROOT / "output/aea_dev_learnability_v1/preregistered_protocol.json").read_text())
    partition = json.loads((ROOT / "output/aea_dev_learnability_v1/partition_manifest.json").read_text())
    audit = json.loads((ROOT / "output/aea_dev_learnability_v1/audit_labels.json").read_text())
    reserve = set(protocol["confirmation_event_groups"])
    development = {row["event_group"] for row in partition["entries"] if row["partition"] == "development"}
    confirmation = {row["event_group"] for row in partition["entries"] if row["partition"] == "confirmation"}
    assert confirmation == reserve
    assert not development & confirmation
    assert len(audit["rows"]) == protocol["audit"]["sample_size"]
    assert not {row["event_group"] for row in audit["rows"]} & reserve
    assert all(len(row["frame_paths"]) == 8 for row in audit["rows"])

