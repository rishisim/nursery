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
EXPECTED_IGNORED = {
    "raw_data", "derived_data", "downloads", "runs", "outputs", "artifacts",
    "checkpoints", "logs", "caches", "rendered_reports",
}


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

    ledger = spec["assumptions_and_deviations"]
    require(ledger["source_of_truth"] == "this object" and ledger["entries"], "ledger missing")
    unresolved = spec["unresolved_reproduction_variables"]
    require(len(unresolved) >= 17, "unresolved-variable inventory unexpectedly incomplete")
    require(all(item["status"] == "unresolved" for item in unresolved), "variable silently resolved")

    ignored = set(spec["required_ignored_roots"])
    require(ignored == EXPECTED_IGNORED, "required ignored-root set changed")
    for root in sorted(ignored):
        probe = ROOT / root / ".phase0-ignore-probe"
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(probe)], cwd=ROOT, check=False
        )
        require(result.returncode == 0, f"generated root is not ignored: {root}/")

    # A stable digest is useful in logs without creating an output artifact.
    digest = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    print(f"Phase 0 verification passed (spec sha256={digest})")


if __name__ == "__main__":
    main()
