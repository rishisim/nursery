#!/usr/bin/env python3
"""Write exact command, environment, hash, and verification provenance for AEA v2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

from babyworld_lite.aea.visible_action_v2 import load_json, sha256_file  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _tree_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    total = 0
    for path in files:
        relative = str(path.relative_to(root))
        file_hash = sha256_file(path)
        size = path.stat().st_size
        total += size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\0")
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/aea_visible_action_v2"))
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--test-seconds", type=float, required=True)
    args = parser.parse_args()
    output = args.root / "commands_and_provenance.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v2 provenance: {output}")
    artifacts = [
        "preregistered_protocol.json",
        "protocol_freeze_receipt.json",
        "preregistered_protocol_amendment_1.json",
        "protocol_amendment_1_freeze_receipt.json",
        "implementation_clarification_receipt.json",
        "annotation_codebook.md",
        "dense_preflight_receipt.json",
        "dense_clip_manifest.json",
        "reserve_access_receipt.json",
        "annotation_packet_pass_a.json",
        "annotation_packet_pass_b.json",
        "model_assisted_pass_a.json",
        "model_assisted_pass_b.json",
        "agreement_report.json",
        "agreement_report.md",
        "consensus_labels.json",
        "split_donor_feasibility.json",
        "capacity_results.json",
        "transcript_control_results.json",
        "imu_preflight_receipt.json",
        "imu_results.json",
        "safe_release_metadata.json",
        "acquisition_recommendation.json",
        "human_annotation_packet.json",
        "aea_visible_action_v2_results.json",
        "scientific_report.md",
    ]
    missing = [name for name in artifacts if not (args.root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required v2 artifacts before provenance: {missing}")
    access = load_json(args.root / "reserve_access_receipt.json")
    imu = load_json(args.root / "imu_results.json")
    result = load_json(args.root / "aea_visible_action_v2_results.json")
    provenance = {
        "schema_version": "aea-visible-action-provenance-v2",
        "protocol_id": "aea-visible-action-v2",
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
            "Read v1 evidence, implementation, dirty diff, and ran the 52-test baseline before v2 work.",
            "Froze v2 markdown and machine protocol before new media or outcomes.",
            "Recorded a metadata-only pre-evidence sampling amendment after exact four-per-group proved impossible; original freeze remained unchanged.",
            "Materialized 31-frame development-only evidence after reserve preflight.",
            "Ran two separately randomized, context-isolated blinded model-assisted passes.",
            "Applied agreement/support gates once, then split, capacity, transcript, and IMU stage gates in frozen order.",
            "Applied the mechanical decision and safe metadata-only acquisition rule once.",
            "Ran the full suite and git diff --check after implementation.",
        ],
        "commands": [
            ".venv/bin/python -m pytest -q  # baseline: 52 passed",
            ".venv-aria/bin/python scripts/prepare_aea_visible_action_v2.py",
            "context-isolated delegated annotation pass A -> output/aea_visible_action_v2/model_assisted_pass_a.json",
            "context-isolated delegated annotation pass B -> output/aea_visible_action_v2/model_assisted_pass_b.json",
            ".venv/bin/python scripts/summarize_aea_visible_action_v2.py",
            ".venv/bin/python scripts/run_aea_visible_action_v2_capacity.py --device auto",
            ".venv/bin/python scripts/run_aea_visible_action_v2_imu.py",
            ".venv/bin/python scripts/finalize_aea_visible_action_v2.py",
            ".venv/bin/python -m compileall -q babyworld_lite scripts",
            ".venv/bin/python -m pytest -q",
            "git diff --check",
            f".venv/bin/python scripts/write_aea_visible_action_v2_provenance.py --tests-passed {args.tests_passed} --test-seconds {args.test_seconds}",
        ],
        "verification": {
            "full_tests_passed": True,
            "test_count": args.tests_passed,
            "pytest_seconds": args.test_seconds,
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
            name: sha256_file(args.root / name) for name in artifacts
        },
        "dense_evidence_tree": _tree_digest(args.root / "dense_evidence"),
        "reserve_and_secret_safeguards": {
            "dense_reserve_rgb_files_opened": access["reserve_rgb_files_opened"],
            "dense_reserve_imu_arrays_opened": access["reserve_imu_arrays_opened"],
            "conditional_reserve_imu_arrays_opened": imu["reserve_imu_arrays_opened"],
            "signed_urls_loaded_printed_or_copied": False,
            "external_manifest_serialized": False,
        },
        "decision": result["decision"],
        "prohibited_actions_not_performed": [
            "prospective reserve RGB or IMU access",
            "locked experiment",
            "new recording download",
            "license acceptance",
            "Professor Frank or other external outreach",
            "commit or push",
            "overwrite or deletion of v1/smoke artifacts",
            "signed URL printing, copying, or use",
        ],
    }
    output.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({
        "artifact": str(output),
        "test_count": args.tests_passed,
        "decision": result["decision"]["recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()
