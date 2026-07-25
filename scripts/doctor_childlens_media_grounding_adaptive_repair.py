#!/usr/bin/env python3
"""Fail-closed doctor for the ChildLens adaptive media-repair amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_media_grounding_adaptive_repair.json"
ENGINEERING = ROOT / "output/childlens_media_grounding_adaptive_repair/engineering_receipt.json"
HISTORICAL_HASHES = {
    "configs/childlens_media_grounding_environment.json": "ac6542baa58d029780f56e6f36b0e82346ec32f80365f18310e45167c4ba8ecd",
    "docs/childlens_media_grounding_environment_protocol.md": "3582b688f6789a553c250a8232374a9b0d2787c6649686160c9e0f85ebc33dcd",
    "output/childlens_media_grounding_environment/engine_gate_receipt.json": "447fb3e353f14b11b58155f15f84f3d92837c65d35dbd02e6d7cde3128c95fdd",
    "output/childlens_media_grounding_environment/terminal_report.md": "13110b599111fffa697e32d33c1304fe2eeb9e21bb1968f3b8ef382ca47ace3c",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv: list[str]) -> tuple[int, str]:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    return result.returncode, (result.stdout + result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    engineering = json.loads(ENGINEERING.read_text())
    historical = {
        name: digest(ROOT / name) == expected
        for name, expected in HISTORICAL_HASHES.items()
    }
    checks: dict[str, dict[str, object]] = {
        "historical_no_go_immutable": {
            "passed": all(historical.values()),
            "files": historical,
        },
        "scope_sealed": {
            "passed": (
                config["scope"]["sole_empirical_source"] == "ChildLens"
                and config["scope"]["development_participants"] == 18
                and config["scope"]["locked_participants"] == 22
                and config["scope"]["locked_participants_must_remain_unread"] is True
                and config["scope"]["final_cue_lift_experiment_authorized"] is False
            ),
        },
        "blender_physical_contract": {
            "passed": (
                engineering["blender"]["gate"] == "PASS"
                and engineering["blender"]["positive_contact"] is True
                and engineering["blender"]["negative_no_contact"] is True
                and engineering["blender"]["joint_change_rad"] > 0
                and engineering["blender"]["dynamic_gyro_peak_rad_s"] > 0
                and engineering["blender"]["static_gyro_peak_rad_s"] == 0
                and engineering["blender"]["deterministic_state_replay"] is True
            ),
        },
        "german_tts_receipt": {
            "passed": (
                engineering["tts"]["gate"] == "PASS_TECHNICAL"
                and engineering["tts"]["repeat_byte_identical"] is True
                and engineering["tts"]["clipped_fraction"] == 0
                and shutil.which("say") is not None
                and shutil.which("ffmpeg") is not None
                and shutil.which("ffprobe") is not None
            ),
        },
    }
    learner = ROOT / ".external/envs/media-grounding-learner/bin/python"
    learner_code, learner_detail = command([
        str(learner),
        "-c",
        (
            "import sys,submitit,torch,torchvision;"
            "sys.path.insert(0,'.external/egobabyvlm');"
            "from apps.baselines.clip.data.transforms import build_eval_transform;"
            "from apps.baselines.clip.modeling.multimodal_model import MultiModalModel;"
            "print(torch.__version__,torchvision.__version__,submitit.__version__)"
        ),
    ])
    checks["official_learner_interfaces"] = {
        "passed": learner_code == 0,
        "detail": learner_detail,
    }
    checks["participant_disjoint_calibration"] = {
        "passed": False,
        "code": config["stop_condition"]["code"],
        "detail": config["stop_condition"]["reason"],
    }
    checks["cue_lift_sealed"] = {
        "passed": (
            config["stop_condition"]["cue_lift_arms_run"] is False
            and engineering["cue_lift_arms_run"] is False
        ),
    }
    ready = all(row["passed"] for row in checks.values())
    result = {
        "schema_version": "childlens-media-grounding-adaptive-doctor-1.0.0",
        "ready": ready,
        "decision": "MEDIA_GROUNDING_ENV_READY" if ready else "NO_GO_ADAPTIVE_MEDIA_REPAIR",
        "checks": checks,
    }
    print(json.dumps(result, indent=2) if args.json else "\n".join(
        f"{'PASS' if row['passed'] else 'FAIL'} {name}: {row}"
        for name, row in checks.items()
    ))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
