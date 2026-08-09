from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_childlens_local_instruments_v1_2.py"
SPEC = importlib.util.spec_from_file_location("childlens_local_instruments_v1_2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_inventory_is_aggregate_only_and_preserves_firewall() -> None:
    receipt = MODULE.audit()
    serialized = json.dumps(receipt, sort_keys=True)
    assert receipt["scope"] == "LOCAL_HOST_INVENTORY_NO_RESTRICTED_INPUT"
    assert receipt["boundary_receipt"]["restricted_data_accessed"] is False
    assert receipt["boundary_receipt"]["model_or_weight_acquired"] is False
    assert receipt["instrument_to_learner_firewall"]["instrument_features_or_embeddings_to_learner"] is False
    assert receipt["route_disposition"]["automated_speaker_role"] == "EXCLUDED_HUMAN_ONLY"
    assert receipt["license_and_access_controls"]["gated_conditions_accepted"] is False
    assert "/Users/" not in serialized
    assert "participant_id" not in serialized
    assert "transcript_text" not in serialized


def test_missing_speech_stack_cannot_be_reported_ready() -> None:
    receipt = MODULE.audit()
    packages = {row["package"]: row for row in receipt["packages"]}
    if not packages["faster-whisper"]["installed"] or not packages["ctranslate2"]["installed"]:
        assert receipt["route_disposition"]["offline_asr_proposals"] == "NOT_LOCAL_DO_NOT_RUN"
    if not packages["whisperx"]["installed"]:
        assert receipt["route_disposition"]["offline_alignment_proposals"] == "NOT_LOCAL_DO_NOT_RUN"


def test_ffmpeg_binary_is_not_marked_for_redistribution() -> None:
    receipt = MODULE.audit()
    ffmpeg = receipt["executables"]["ffmpeg"]
    if ffmpeg["available"]:
        assert "LOCAL_EXECUTION_ONLY_NO_REDISTRIBUTION" in ffmpeg["license_state"]
