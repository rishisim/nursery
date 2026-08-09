from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_childlens_annotations_v1_1.py"
SPEC = importlib.util.spec_from_file_location("annotation_audit_v1_1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit = MODULE.audit


def test_structural_receipt_excludes_sentinels(tmp_path: Path) -> None:
    participant = tmp_path / "participants.csv"
    with participant.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["video_file", "anonymized_child", "recording_date"])
        writer.writerow(["SECRET.mp4", "SECRET_CHILD", "2099-01-01"])
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "SECRET.json").write_text(
        json.dumps({
            "videoName": "SECRET.mp4",
            "duration": 4.0,
            "annotations": [{"label": "SECRET speech", "start": 0, "end": 1}],
        }),
        encoding="utf-8",
    )
    receipt = audit(annotations, participant)
    assert receipt["annotation_file_count"] == 1
    assert receipt["participant_media_link_complete_count"] == 1
    assert receipt["speech_like_value_file_count"] == 1
    assert "SECRET" not in str(receipt)
