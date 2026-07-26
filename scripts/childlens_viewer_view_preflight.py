#!/usr/bin/env python3
"""Fail-closed, media-blind host preflight for ReViV/ViPE calibration."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/childlens_viewer_view_preflight_v1.json"


def _capture(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def evaluate_host(
    config_path: Path = DEFAULT_CONFIG,
    *,
    which: Callable[[str], str | None] = shutil.which,
    system: str | None = None,
    machine: str | None = None,
) -> dict[str, object]:
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    required = protocol["hardware_gate"]
    detected_system = system or platform.system()
    detected_machine = machine or platform.machine()
    nvidia_smi = which("nvidia-smi")
    nvcc = which("nvcc")
    cuda_available = bool(nvidia_smi and nvcc)
    reasons: list[str] = []
    if not nvidia_smi:
        reasons.append("nvidia_smi_absent")
    if not nvcc:
        reasons.append("nvcc_absent")
    if detected_system == "Darwin" and detected_machine == "arm64":
        reasons.append("apple_silicon_metal_only_not_official_cuda_path")
    decision = (
        "PASS_LOCAL_CUDA_EXECUTION_PATH"
        if cuda_available
        else required["stop_if_unavailable"]
    )
    return {
        "schema": "ChildLensViewerViewHostPreflightResult.v1",
        "media_opened": False,
        "network_private_data_transfer": False,
        "system": detected_system,
        "machine": detected_machine,
        "nvidia_smi_present": bool(nvidia_smi),
        "nvcc_present": bool(nvcc),
        "cuda_version_text": _capture([nvcc, "--version"]) if nvcc else "",
        "official_local_path_available": cuda_available,
        "decision": decision,
        "reasons": reasons,
        "next_stage_authorized": cuda_available,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_host(args.config)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["next_stage_authorized"] else 2)


if __name__ == "__main__":
    main()
