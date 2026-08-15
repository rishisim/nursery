#!/usr/bin/env python3
"""Verify frozen Phase-0 invariants and the privacy-safe Phase-1 record."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "spec.json"
PILOT_DECISION_PATH = ROOT / "pilots" / "juno_sample" / "decision_record.json"
EXPECTED_COMMIT = "224621caf0628270b6115845ac75a65b984234a3"
EXPECTED_ARCHIVE_SHA256 = (
    "5f4c662919e3b0abdb85f5d6520350f10c374f087886f4257bbe1bd336f3b13c"
)
EXPECTED_IGNORED = {"logs", "caches"}
EXPECTED_VISIBLE = {
    "raw_data",
    "derived_data",
    "local_storage",
    "data",
    "downloads",
    "output",
    "outputs",
    "runs",
    "artifacts",
    "checkpoints",
    "weights",
    "generated",
    "tmp",
    "rendered",
    "rendered_reports",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    require(spec["schema_version"] >= 2, "Phase-1 schema is missing")

    phase = spec["phase"]
    require(phase["number"] == 1, "unexpected current scientific phase")
    require(
        phase["status"] == "blocked_by_exact_usable_subset_evidence",
        "Phase 1 status was changed without evidence",
    )

    scope = spec["scope"]
    require(scope["model_row"] == "BabyView CLIP+", "wrong model row")
    require(scope["benchmark"] == "Machine-DevBench", "wrong benchmark")
    require(scope["tasks"] == ["lexical", "grammatical"], "scope broadened")
    require(scope["evaluation_family"] == "cross-modal grounding", "wrong family")
    require(scope["exact_replay_status"] == "not_claimed", "exact replay overclaimed")

    require(spec["paper"]["arxiv_id"] == "2605.19130", "wrong arXiv paper")
    require(spec["paper"]["version"] == "v1", "paper version not frozen")
    require(spec["official_code"]["commit"] == EXPECTED_COMMIT, "wrong code pin")
    require(spec["training_data"]["release"] == "2025.1", "wrong data release")

    asset = spec["evaluation_data"]["asset"]
    require(spec["evaluation_data"]["tag"] == "Eval-Data", "wrong release tag")
    require(asset["name"] == "MachineDevBench.tar", "wrong evaluation asset")
    require(asset["size_bytes"] == 1_635_409_920, "wrong archive size")
    require(asset["sha256"] == EXPECTED_ARCHIVE_SHA256, "wrong archive digest")
    require(is_sha256(spec["official_code"]["lockfile_sha256"]), "bad lock digest")

    targets = spec["targets_percent"]
    require(targets["chance"] == 50.0 and targets["runs"] == 3, "wrong target runs")
    require(targets["lexical"] == {"mean": 53.4, "stddev": 0.7}, "wrong lexical target")
    require(
        targets["grammatical"] == {"mean": 53.8, "stddev": 2.4},
        "wrong grammatical target",
    )
    require(targets["overall"] == {"mean": 53.6, "stddev": 1.5}, "wrong overall target")
    require(
        abs((targets["lexical"]["mean"] + targets["grammatical"]["mean"]) / 2
            - targets["overall"]["mean"]) < 1e-12,
        "mean aggregation fails",
    )

    policy = spec["data_and_initialization_policy"]
    require(policy["scientific_run_initialization"] == "from_scratch_only", "not from scratch")
    require(policy["external_pretrained_initialization_allowed"] is False, "external init allowed")

    storage = spec["storage_policy"]
    require(storage["mode"] == "juno_two_tier_storage", "wrong storage mode")
    require(storage["fallback_to_local_or_worktree_storage_allowed"] is False, "local fallback allowed")
    require(storage["promotion_requires_checksum_verification"] is True, "promotion is unverified")
    require(storage["force_worktree_removal_allowed"] is False, "forced removal allowed")

    phase1 = spec["phase1_evidence"]
    require(phase1["status"] == phase["status"], "Phase-1 statuses disagree")
    require(phase1["populated_ledger_trackable"] is False, "sensitive ledger made trackable")
    require((ROOT / phase1["aggregate_report"]).is_file(), "aggregate report is missing")
    aggregates = phase1["governed_match_aggregates"]
    require(aggregates["mapped_bv_main_recordings"] == 4610, "BV-main count changed")
    require(aggregates["mapped_bv_main_hours"] == 868.4602, "BV-main hours changed")
    require(phase1["download_workflow_probe"]["bulk_download_started"] is False, "bulk download overclaimed")

    decisions = spec["assumptions_and_deviations"]
    require(decisions["source_of_truth"] == "this object", "decision ledger moved")
    decision_ids = [entry["id"] for entry in decisions["entries"]]
    require(len(decision_ids) == len(set(decision_ids)), "duplicate decision IDs")

    unresolved = spec["unresolved_reproduction_variables"]
    unresolved_ids = [entry["id"] for entry in unresolved]
    require(len(unresolved_ids) == len(set(unresolved_ids)), "duplicate unresolved IDs")
    require("U018" in unresolved_ids, "exact-subset blocker disappeared")
    require(all(entry["status"] == "unresolved" for entry in unresolved), "variable silently resolved")

    pilot = json.loads(PILOT_DECISION_PATH.read_text(encoding="utf-8"))
    fixed = pilot["fixed_decisions"]
    require(pilot["status"] in {"p0_complete", "p1_complete", "p2_complete", "p3_complete", "p4_incomplete", "p4_complete", "p5_complete", "p6_complete", "p7_complete"}, "pilot P0 contract is not preserved")
    require(fixed["input_count"] == 2, "pilot input count changed")
    require(fixed["additional_video_downloads_authorized"] is False, "pilot download scope broadened")
    require(fixed["tokenizer"]["target_vocabulary_size"] == 380, "authorized pilot vocabulary changed")
    require(fixed["tokenizer"]["historical_initial_target"].startswith("2048_stopped"), "initial stopped vocabulary gate lost")
    require(fixed["vision_architecture"] == "DINOv2_ViT-B/14", "pilot vision shape changed")
    require(fixed["text_architecture"] == "BERT-base_12x768x12", "pilot text shape changed")
    require(fixed["initialization"] == "random_for_all_pilot_learned_components", "pilot init changed")
    require(pilot["scientific_status_effect"] == "none", "pilot changes scientific status")
    require(pilot["full_reproduction_recalibration_required"] is True, "full recalibration dropped")

    ignored = set(spec["required_ignored_roots"])
    require(ignored == EXPECTED_IGNORED, "ignored-root contract changed")
    for root in sorted(ignored):
        probe = ROOT / root / ".verification-probe"
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(probe)], cwd=ROOT, check=False
        )
        require(result.returncode == 0, f"generated root is not ignored: {root}/")

    visible = set(spec["required_visible_if_accidentally_local"])
    require(visible == EXPECTED_VISIBLE, "visible-root contract changed")
    for root in sorted(visible):
        probe = ROOT / root / ".verification-probe"
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(probe)], cwd=ROOT, check=False
        )
        require(result.returncode != 0, f"important root is silently ignored: {root}/")

    digest = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    print(f"Frozen reproduction verification passed (spec sha256={digest})")


if __name__ == "__main__":
    main()
