#!/usr/bin/env python3
"""Fail-closed doctor for the asset-rich fidelity terminal record."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_asset_rich_grounding_amendment.json"
RECEIPT = ROOT / "output/childlens_asset_rich_grounding/engineering_receipt.json"
PILOTS = ROOT / "tmp/childlens_asset_rich/pilots"
HISTORICAL = {
    "output/childlens_fold_evidence_recovery/terminal_report.md": "d079c15e6c077b65cd9bed6031a6d405d845c9db",
    "output/childlens_media_grounding_adaptive_repair/terminal_report.md": "2fd85233f9fa327ae76e76a17993f2410544308e",
    "babyworld_lite/childlens_media_repair/blender_adapter.py": "35b0b6e8ec9e607155c52dde4d7d3935cbf52730",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["decision"] == "NO_GO_ASSET_RICH_FIDELITY"
    assert config["scope"]["cue_lift_experiment_authorized"] is False
    assert receipt["downstream"]["cue_lift_arms_instantiated"] is False
    assert receipt["downstream"]["independent_distribution_generator_built"] is False
    assert receipt["frozen_pilot_gate"]["distribution_generation_permitted"] is False
    assert config["alignment_amendment"]["historical_values_are_seconds"] is False
    for relative, expected in HISTORICAL.items():
        actual = subprocess.check_output(
            ["git", "hash-object", relative], cwd=ROOT, text=True
        ).strip()
        assert actual == expected, f"historical record changed: {relative}"
    for pilot in receipt["pilots"]:
        directory = PILOTS / pilot["episode_id"]
        assert sha256(directory / "pilot.mp4") == pilot["mp4_sha256"]
        assert sha256(directory / "speech.wav") == pilot["wav_sha256"]
        assert sha256(directory / "representative.png") == pilot["still_sha256"]
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,width,height,r_frame_rate,sample_rate",
            "-of", "json", directory / "pilot.mp4",
        ], text=True))
        assert float(probe["format"]["duration"]) == 20.0
        video = next(item for item in probe["streams"] if item["codec_type"] == "video")
        audio = next(item for item in probe["streams"] if item["codec_type"] == "audio")
        assert (video["width"], video["height"], video["r_frame_rate"]) == (640, 360, "30/1")
        assert audio["sample_rate"] == "24000"
    print(json.dumps({
        "doctor": "PASS",
        "decision": receipt["decision"],
        "pilots_verified": len(receipt["pilots"]),
        "historical_records_unchanged": True,
        "cue_lift_arms_run": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
