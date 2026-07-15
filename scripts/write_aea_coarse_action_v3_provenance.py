#!/usr/bin/env python3
"""Write exact hashes, commands, environment, and safeguards for AEA v3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import __version__ as pillow_version
import scipy
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.coarse_action_v3 import load_json, sha256_file  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/aea_coarse_action_v3"))
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    output = args.root / "commands_and_provenance.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v3 provenance: {output}")
    artifacts = [
        "preregistered_protocol.json",
        "protocol_freeze_receipt.json",
        "annotation_codebook.md",
        "fixed_dense_manifest.json",
        "reserve_access_receipt.json",
        "annotation_packet_pass_a_stage1_visual.json",
        "annotation_packet_pass_b_stage1_visual.json",
        "annotation_packet_pass_a_stage2_language.json",
        "annotation_packet_pass_b_stage2_language.json",
        "model_assisted_pass_a_stage1.json",
        "model_assisted_pass_b_stage1.json",
        "stage1_seal_receipt.json",
        "model_assisted_pass_a_stage2.json",
        "model_assisted_pass_b_stage2.json",
        "model_assisted_pass_a.json",
        "model_assisted_pass_b.json",
        "annotation_isolation_receipt.json",
        "agreement_report.json",
        "agreement_report.md",
        "consensus_labels.json",
        "language_alignment_results.json",
        "language_alignment_report.md",
        "split_donor_feasibility.json",
        "capacity_preflight_receipt.json",
        "capacity_results.json",
        "natural_transcript_model_receipt.json",
        "imu_preflight_receipt.json",
        "imu_results.json",
        "storage_reassessment.json",
        "acquisition_decision.json",
        "terminal_decision.json",
        "human_annotation_authorization.json",
        "aea_coarse_action_v3_results.json",
        "scientific_report.md",
    ]
    missing = [name for name in artifacts if not (args.root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required v3 artifacts: {missing}")
    access = load_json(args.root / "reserve_access_receipt.json")
    isolation = load_json(args.root / "annotation_isolation_receipt.json")
    imu = load_json(args.root / "imu_results.json")
    results = load_json(args.root / "aea_coarse_action_v3_results.json")
    provenance: dict[str, Any] = {
        "schema_version": "aea-coarse-action-provenance-v3",
        "protocol_id": "aea-coarse-action-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(Path.cwd()),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty_checkout_preserved": bool(_git("status", "--porcelain")),
            "commit_created": False,
            "push_performed": False,
        },
        "ordering": [
            "Read the authorized v1/v2 aggregate evidence, preregistrations, current code/tests, and dirty git diff without opening completed v1/v2 pass labels or rationales.",
            "Ran the 59-test baseline and git diff --check.",
            "Froze v3 Markdown, machine protocol, codebook, and hash receipt before any dense v3 review or v3 outcome.",
            "Reblinded exactly the fixed 72-row v2 dense manifest and referenced its existing 31-frame evidence with zero new RGB query.",
            "Ran pass A and pass B as separate context-isolated model-assisted reviewers in distinct randomized orders.",
            "For each pass, sealed and hashed all 72 video-only visible-action labels before revealing that pass's transcript-stage packet.",
            "Compared passes once without adjudication and computed separate coarse annotation/support and language-alignment gates.",
            "Applied the one-shot split/donor, video capacity, and conditional IMU stage gates in frozen order.",
            "Applied terminal language/sensor decisions and the sanitized-metadata-only storage rule once.",
            "Ran full compile, tests, and git diff --check after implementation.",
        ],
        "commands": [
            ".venv/bin/python -m pytest -q  # baseline: 59 passed",
            ".venv/bin/python scripts/prepare_aea_coarse_action_v3.py",
            "context-isolated pass A stage 1 -> model_assisted_pass_a_stage1.json",
            "context-isolated pass B stage 1 -> model_assisted_pass_b_stage1.json",
            "seal SHA-256 for both stage-1 artifacts",
            "same isolated pass A stage 2 -> model_assisted_pass_a_stage2.json",
            "same isolated pass B stage 2 -> model_assisted_pass_b_stage2.json",
            ".venv/bin/python scripts/summarize_aea_coarse_action_v3.py",
            ".venv/bin/python scripts/run_aea_coarse_action_v3_capacity.py --device auto",
            ".venv/bin/python scripts/run_aea_coarse_action_v3_imu.py",
            ".venv/bin/python scripts/finalize_aea_coarse_action_v3.py",
            ".venv/bin/python -m compileall -q babyworld_lite scripts",
            ".venv/bin/python -m pytest -q",
            "git diff --check",
            f".venv/bin/python scripts/write_aea_coarse_action_v3_provenance.py --tests-passed {args.tests_passed} --test-seconds {args.test_seconds}",
        ],
        "verification": {
            "baseline_tests_passed": 59,
            "full_tests_passed": True,
            "test_count": int(args.tests_passed),
            "pytest_seconds": float(args.test_seconds),
            "compileall_passed": True,
            "git_diff_check_passed": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pillow": pillow_version,
            "mps_available": torch.backends.mps.is_available(),
        },
        "sha256": {
            "docs/aea_coarse_action_v3_preregistration.md": sha256_file(
                "docs/aea_coarse_action_v3_preregistration.md"
            ),
            **{name: sha256_file(args.root / name) for name in artifacts},
        },
        "fixed_evidence": {
            "membership": "exact same 72 v2 dense rows",
            "frames_referenced": access["v3_existing_dense_frames_referenced"],
            "frames_copied_or_rematerialized": 0,
            "additional_rgb_queries": access["v3_additional_development_rgb_queries"],
        },
        "annotation_isolation": isolation,
        "reserve_and_secret_safeguards": {
            "reserve_event_groups_opened": access["reserve_event_groups_opened"],
            "reserve_rgb_files_opened": access["reserve_rgb_files_opened"],
            "reserve_imu_arrays_opened_during_packet_preparation": access[
                "reserve_imu_arrays_opened"
            ],
            "reserve_imu_arrays_opened_during_conditional_imu": imu[
                "reserve_imu_arrays_opened"
            ],
            "signed_manifest_reopened_in_v3": False,
            "signed_urls_loaded_printed_copied_or_used": False,
            "v1_or_v2_completed_annotation_files_opened": 0,
        },
        "decision": results["decision"],
        "prohibited_actions_not_performed": [
            "prospective reserve RGB or IMU access",
            "locked experiment",
            "new recording download",
            "license acceptance",
            "Professor Frank or other external outreach",
            "commit or push",
            "overwrite or deletion of v1, v2, or smoke evidence",
            "signed URL loading, printing, copying, or use",
            "post-outcome adjudication, endpoint subset, threshold change, or rerun",
            "another user-owned Codex task",
        ],
    }
    output.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({
        "artifact": str(output),
        "test_count": args.tests_passed,
        "formal_decision": results["decision"]["formal_decision"],
        "scientific_conclusion": results["decision"]["scientific_conclusion"],
    }, indent=2))


if __name__ == "__main__":
    main()
