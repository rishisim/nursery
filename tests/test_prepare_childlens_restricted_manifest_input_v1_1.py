from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_childlens_restricted_manifest_input_v1_1.py"
SPEC = importlib.util.spec_from_file_location("prepare_manifest_input_v1_1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preparer_writes_restricted_hmac_manifest(tmp_path: Path) -> None:
    participant = tmp_path / "participants.csv"
    with participant.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["video_file", "anonymized_child", "recording_date"])
        writer.writerow(["SYNTHETIC.mp4", "SYNTHETIC_CHILD", "2099-01-01"])
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "synthetic.json").write_text(json.dumps({
        "video_name": "SYNTHETIC.mp4",
        "annotations": [
            {"eventId": "play", "start": 0.0, "duration": 4.0},
            {"eventId": "location", "start": 0.0, "duration": 4.0,
             "fields": {"Type of Location": "room"}},
        ],
    }), encoding="utf-8")
    output = tmp_path / "manifest.json"
    receipt = MODULE.prepare(
        annotations,
        participant,
        tmp_path / "secret.bin",
        output,
        "a" * 64,
    )
    document = json.loads(output.read_text())
    assert receipt["media_count"] == 1
    assert document["media"][0]["duration_milliseconds"] == 4000
    assert len(document["media"][0]["participant_key"]) == 64
    assert "SYNTHETIC_CHILD" not in str(document)
