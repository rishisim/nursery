#!/usr/bin/env python3
"""Fail-closed doctor for the frozen ChildLens media-grounding environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EGO_COMMIT = "224621caf0628270b6115845ac75a65b984234a3"


def _run(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return result.returncode, (result.stdout + result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks: dict[str, dict[str, object]] = {}
    for executable in ("ffmpeg", "ffprobe", "say"):
        path = shutil.which(executable)
        checks[executable] = {"passed": path is not None, "path": path}

    voice_code, voices = _run(["say", "-v", "?"])
    checks["german_tts_anna"] = {
        "passed": voice_code == 0 and any(
            line.startswith("Anna") and "de_DE" in line for line in voices.splitlines()
        ),
        "route": "macOS say / Anna / de_DE / no voice cloning",
    }

    ego = ROOT / ".external" / "egobabyvlm"
    ego_code, ego_commit = _run(["git", "-C", str(ego), "rev-parse", "HEAD"])
    checks["egobabyvlm_pin"] = {
        "passed": ego_code == 0 and ego_commit == EXPECTED_EGO_COMMIT,
        "actual": ego_commit,
        "expected": EXPECTED_EGO_COMMIT,
    }
    learner = ROOT / ".external" / "envs" / "media-grounding-learner" / "bin" / "python"
    learner_code, learner_output = _run(
        [
            str(learner),
            "-c",
            (
                "import sys,torch,torchvision;"
                "sys.path.insert(0,'.external/egobabyvlm');"
                "from apps.baselines.clip.data.transforms import build_eval_transform;"
                "from apps.baselines.clip.modeling.multimodal_model import MultiModalModel;"
                "print(torch.__version__,torchvision.__version__,torch.backends.mps.is_available())"
            ),
        ]
    )
    checks["official_learner_interfaces"] = {
        "passed": learner_code == 0,
        "detail": learner_output,
    }

    blender = Path.home() / "blender" / "blender-4.2.1-macos-arm64" / "Blender.app"
    checks["blender_arm64"] = {
        "passed": blender.exists(),
        "path": str(blender),
        "version": "4.2.1",
    }
    receipt_path = ROOT / "output" / "childlens_media_grounding_environment" / "engine_gate_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    checks["engine_acceptance"] = {
        "passed": receipt.get("selected_engine_gate") == "PASS",
        "receipt_sha256": (
            hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            if receipt_path.exists()
            else None
        ),
        "decision": receipt.get("selected_engine_gate", "MISSING"),
    }

    passed = all(bool(row["passed"]) for row in checks.values())
    result = {
        "schema_version": "childlens-media-grounding-doctor-v1.0.0",
        "passed": passed,
        "checks": checks,
    }
    print(json.dumps(result, indent=2) if args.json else "\n".join(
        f"{'PASS' if row['passed'] else 'FAIL'} {name}: {row}"
        for name, row in checks.items()
    ))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
