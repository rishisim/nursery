from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import wave
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/smoke_gemma4_mlx_vlm_v1_3_1.py"
)
SPEC = importlib.util.spec_from_file_location("gemma4_public_smoke_v1_3_1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generated_fixtures_are_synthetic_image_and_audio() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        image_path, audio_path = MODULE.create_synthetic_fixtures(Path(temporary))
        with Image.open(image_path) as image:
            assert image.size == (256, 256)
            assert image.mode == "RGB"
            assert image.getpixel((128, 128))[0] > 200
        with wave.open(str(audio_path), "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getframerate() == 16_000
            assert audio.getnframes() == 16_000


def test_strict_schema_accepts_good_and_rejects_extra_or_bad_enum() -> None:
    valid = json.dumps(MODULE.SYNTHETIC_VALID_OUTPUT)
    assert MODULE.validate_response_text(valid) == MODULE.SYNTHETIC_VALID_OUTPUT

    extra = dict(MODULE.SYNTHETIC_VALID_OUTPUT)
    extra["unexpected"] = True
    with pytest.raises(MODULE.SmokeError, match="E_RESPONSE_SCHEMA"):
        MODULE.validate_response_text(json.dumps(extra))

    bad_enum = json.loads(valid)
    bad_enum["events"][0]["source_role"] = "MODEL_GUESS"
    with pytest.raises(MODULE.SmokeError, match="E_RESPONSE_SCHEMA"):
        MODULE.validate_response_text(json.dumps(bad_enum))


def test_allowlisted_requests_have_joint_inputs_and_strict_schema() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        image_path, audio_path = MODULE.create_synthetic_fixtures(Path(temporary))
        for model_id in MODULE.MODEL_IDS.values():
            request = MODULE.build_loopback_request(model_id, image_path, audio_path)
            content = request["messages"][0]["content"]
            assert [part["type"] for part in content] == [
                "input_image",
                "text",
                "input_audio",
            ]
            assert request["response_format"]["type"] == "json_schema"
            assert request["response_format"]["json_schema"]["strict"] is True
            assert request["response_format"]["json_schema"]["schema"] == MODULE.STRICT_SCHEMA

        with pytest.raises(MODULE.SmokeError, match="E_MODEL_NOT_ALLOWLISTED"):
            MODULE.build_loopback_request("unapproved/model", image_path, audio_path)


def test_preflight_is_honest_not_executed_receipt() -> None:
    report = MODULE.preflight_report()
    assert report["status"] == "NOT_EXECUTED"
    assert report["scope"] == "PUBLIC_SYNTHETIC_ONLY"
    assert report["hosted_or_cloud_inference_used"] is False
    assert report["childlens_or_quarantine_accessed"] is False
    assert report["model_weights_downloaded"] is False
    assert report["license_clickthrough_accepted"] is False
    assert report["model_inference_executed"] is False
    assert report["wall_time_or_memory_compared"] is False
    assert report["synthetic_schema_harness"]["server_call_executed"] is False
    assert report["synthetic_schema_harness"]["valid_fixture_accepted"] is True
    assert report["synthetic_schema_harness"]["extra_property_rejected"] is True
    assert report["synthetic_schema_harness"]["joint_audio_image_request_shape_built"] is True
    assert report["unmeasured_fields"]
    assert report["host"]["physical_memory_gib"] in (None, 32.0)
    assert report["documentation_preflight"]["conversion_license_metadata_consistent"] is False
    assert "not_runtime_memory" in next(
        key for key in report["documentation_preflight"] if key.startswith("published_")
    )


def test_script_has_no_network_client_import() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"requests", "urllib", "httpx", "openai", "huggingface_hub"})


def test_model_dispositions_do_not_claim_end_to_end_execution() -> None:
    report = MODULE.preflight_report()
    for disposition in report["model_dispositions"].values():
        assert disposition["mlx_vlm_joint_audio_image_strict_schema"] == "NOT_EXECUTED"
        assert disposition["shortlist_status"] == "CONDITIONAL_SMOKE_REQUIRED"
