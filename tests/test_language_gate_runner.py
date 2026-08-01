import json
from pathlib import Path


def test_runner_uses_frozen_candidate_family_and_offline_reload():
    source = Path("scripts/run_synthetic_video_language_gate.py").read_text()
    config = json.loads(Path("configs/synthetic_video_language_gate.json").read_text())
    assert "for item in config[\"candidate_order\"]" in source
    assert "local_files_only=True" in source
    assert "HF_HUB_OFFLINE" in source
    assert "asr[\"artifact_sha256\"]" in source
    assert len(config["candidates"]) == 2
    assert config["decoding"]["temperature"] == 0
    assert config["decoding"]["beam_size"] == 5
    assert "temperature=decoding[\"temperature\"]" in source
    assert "synthetic_video_language_adapter" in source
