import copy

import pytest

from scripts.run_synthetic_video_ltx_api import PROMPT_COMMITMENT, ltx_payload, verify_plan
from scripts.run_synthetic_video_prototype import PrototypeError, digest


def frozen_plan():
    plan = {
        "schema_version": 1,
        "model": "minimax/hailuo-3",
        "duration": 10,
        "resolution": "2K",
        "aspect_ratio": "16:9",
        "generate_audio": False,
        "prompt": "fixed public prompt",
        "negative_constraints": [],
        "tts": {"language": "de", "text": "", "intervals": []},
    }
    plan["commitment_sha256"] = digest(plan)
    return plan


def test_ltx_payload_reuses_prompt_and_disables_native_audio(monkeypatch):
    plan = frozen_plan()
    monkeypatch.setattr("scripts.run_synthetic_video_ltx_api.PROMPT_COMMITMENT", plan["commitment_sha256"])
    payload = ltx_payload(plan)
    assert payload["prompt"] == plan["prompt"]
    assert payload == {
        "prompt": "fixed public prompt",
        "duration": 10,
        "resolution": "1080p",
        "aspect_ratio": "16:9",
        "fps": 24,
        "generate_audio": False,
    }


def test_plan_verification_rejects_post_freeze_prompt_edit(monkeypatch):
    plan = frozen_plan()
    monkeypatch.setattr("scripts.run_synthetic_video_ltx_api.PROMPT_COMMITMENT", plan["commitment_sha256"])
    edited = copy.deepcopy(plan)
    edited["prompt"] += " edited"
    with pytest.raises(PrototypeError, match="E_FROZEN_PROMPT_COMMITMENT"):
        verify_plan(edited)


def test_repository_pin_matches_frozen_public_plan():
    assert PROMPT_COMMITMENT == "63e20f20c29cf6d88116172ec245a10b788753581fc4643e773104f7eb6a71cb"
