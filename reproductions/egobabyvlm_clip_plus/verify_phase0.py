#!/usr/bin/env python3
"""Dependency-free validation for the frozen EgoBabyVLM Phase-0 spec."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "spec.json"
EXPECTED_COMMIT = "224621caf0628270b6115845ac75a65b984234a3"
EXPECTED_ARCHIVE_SHA256 = "5f4c662919e3b0abdb85f5d6520350f10c374f087886f4257bbe1bd336f3b13c"
EXPECTED_IGNORED = {"logs", "caches"}
EXPECTED_VISIBLE = {
    "raw_data", "derived_data", "local_storage", "data", "downloads",
    "output", "outputs", "runs", "artifacts", "checkpoints", "weights",
    "generated", "tmp", "rendered", "rendered_reports",
}
PHASE1_SCHEMA = ROOT / "inclusion_ledger.schema.json"
PHASE1_REPORT = ROOT / "PHASE1_REPORT.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    scope = spec["scope"]
    require(scope["model_row"] == "BabyView CLIP+", "wrong model row")
    require(scope["benchmark"] == "Machine-DevBench", "wrong benchmark")
    require(scope["tasks"] == ["lexical", "grammatical"], "scope broadened")
    require(scope["evaluation_family"] == "cross-modal grounding", "wrong family")
    require(scope["exact_replay_status"] == "not_claimed", "exact replay overclaimed")

    require(spec["paper"]["arxiv_id"] == "2605.19130", "wrong arXiv paper")
    require(spec["paper"]["version"] == "v1", "paper version not frozen")
    require(spec["official_code"]["commit"] == EXPECTED_COMMIT, "wrong code pin")
    require(spec["training_data"]["release"] == "2025.1", "wrong BabyView release")

    asset = spec["evaluation_data"]["asset"]
    require(spec["evaluation_data"]["tag"] == "Eval-Data", "wrong release tag")
    require(asset["name"] == "MachineDevBench.tar", "wrong evaluation asset")
    require(asset["size_bytes"] == 1_635_409_920, "wrong archive size")
    require(is_sha256(asset["sha256"]), "archive SHA-256 is malformed")
    require(asset["sha256"] == EXPECTED_ARCHIVE_SHA256, "wrong archive SHA-256")
    require(is_sha256(spec["official_code"]["lockfile_sha256"]), "lock SHA-256 malformed")

    targets = spec["targets_percent"]
    require(targets["chance"] == 50.0 and targets["runs"] == 3, "wrong chance/runs")
    require(targets["lexical"] == {"mean": 53.4, "stddev": 0.7}, "wrong lexical target")
    require(targets["grammatical"] == {"mean": 53.8, "stddev": 2.4}, "wrong grammatical target")
    require(targets["overall"] == {"mean": 53.6, "stddev": 1.5}, "wrong overall target")
    expected_overall = (targets["lexical"]["mean"] + targets["grammatical"]["mean"]) / 2
    require(abs(expected_overall - targets["overall"]["mean"]) < 1e-12, "mean aggregation fails")
    require("unweighted arithmetic mean" in targets["aggregation"], "aggregation undefined")

    policy = spec["data_and_initialization_policy"]
    require(policy["scientific_run_initialization"] == "from_scratch_only", "run not from scratch")
    require(policy["external_pretrained_initialization_allowed"] is False, "external init allowed")
    require(any("public or web pretrained weights" in item for item in policy["forbidden"]),
            "external pretrained initialization is not explicitly forbidden")

    storage = spec["storage_policy"]
    require(storage["mode"] == "juno_two_tier_storage", "wrong storage mode")
    require(storage["remote"] == {
        "host": "juno.hpcre.utdallas.edu", "user": "dal503972"
    }, "wrong Juno endpoint")
    require(storage["durable"]["root"] == "/work/dal503972/egobabyvlm_clip_plus",
            "wrong durable Juno root")
    require(storage["scratch"]["root"] ==
            "/scratch/juno/dal503972/egobabyvlm_clip_plus", "wrong scratch Juno root")
    require(storage["linked_worktrees_are_disposable"] is True, "worktrees not disposable")
    require(storage["force_worktree_removal_allowed"] is False, "forced removal allowed")
    require(storage["fallback_to_local_or_worktree_storage_allowed"] is False,
            "local fallback allowed")
    require(storage["promotion_requires_checksum_verification"] is True,
            "promotion checksum not required")
    require(storage["backup_required"] is True, "persistent storage lacks backup policy")

    ledger = spec["assumptions_and_deviations"]
    require(ledger["source_of_truth"] == "this object" and ledger["entries"], "ledger missing")
    unresolved = spec["unresolved_reproduction_variables"]
    require(len(unresolved) >= 17, "unresolved-variable inventory unexpectedly incomplete")
    require(all(item["status"] == "unresolved" for item in unresolved), "variable silently resolved")
    require(any(item["id"] == "U018" for item in unresolved), "subset discrepancy blocker missing")

    phase = spec["phase"]
    require(phase["number"] == 1, "Phase 1 status not recorded")
    require(phase["status"] == "blocked_by_external_evidence", "Phase 1 status overclaimed")
    audit = spec["phase1_audit"]
    require(audit["populated_ledger_trackable"] is False, "private ledger made trackable")
    require(audit["complete_release_inventory_assessed"] is False, "inventory falsely assessed")
    require(spec["training_data"]["published_duration_hours"]["reconciliation_status"] == "not_reconciled", "duration discrepancy falsely reconciled")

    schema = json.loads(PHASE1_SCHEMA.read_text(encoding="utf-8"))
    required = set(schema["required"])
    expected_fields = {"record_key", "databrary_reference", "source_filename", "sha256",
                       "duration_seconds", "audio_present", "audio_usable", "decision",
                       "exclusion_reason", "group_key", "release_membership",
                       "probe_version", "probe_error"}
    require(required == expected_fields, "private ledger schema fields changed")
    reasons = schema["properties"]["exclusion_reason"]["enum"]
    require(reasons == audit["controlled_exclusion_reasons"], "exclusion vocabularies differ")
    report = PHASE1_REPORT.read_text(encoding="utf-8")
    require("blocked by external evidence" in report.lower(), "report status missing")
    require("894 → ~863 reconciled: no" in report, "report overclaims reconciliation")

    audit_source = (ROOT / "audit_phase1.py").read_text(encoding="utf-8")
    require("stream_sha256" in audit_source and "os.replace" in audit_source, "streaming/atomic audit invariants missing")
    require("ledger output must not be inside the Git repository" in audit_source, "repository privacy boundary missing")
    require("validate_governed_output" in audit_source, "governed-storage boundary missing")

    ignored = set(spec["required_ignored_roots"])
    require(ignored == EXPECTED_IGNORED, "required ignored-root set changed")
    for root in sorted(ignored):
        probe = ROOT / root / ".phase0-ignore-probe"
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(probe)], cwd=ROOT, check=False
        )
        require(result.returncode == 0, f"generated root is not ignored: {root}/")

    visible = set(spec["required_visible_if_accidentally_local"])
    require(visible == EXPECTED_VISIBLE, "required visible-root set changed")
    for root in sorted(visible):
        probe = ROOT / root / ".phase0-visible-probe"
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(probe)], cwd=ROOT, check=False
        )
        require(result.returncode != 0, f"important local root is silently ignored: {root}/")

    # A stable digest is useful in logs without creating an output artifact.
    digest = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    print(f"Phase 0 verification passed (spec sha256={digest})")


if __name__ == "__main__":
    main()
