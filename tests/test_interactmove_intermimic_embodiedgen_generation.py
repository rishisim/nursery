from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import yaml

from babyworld_lite.interactmove_intermimic.activity import load_activity_spec


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_interactmove_intermimic_embodiedgen_generation.py"
ACTIVITY = ROOT / "configs" / "interactmove_intermimic_activity.json"
PROTOCOL = ROOT / "configs" / "interactmove_intermimic.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("fresh_generation_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_activity_is_valid_and_seeded() -> None:
    activity = load_activity_spec(ACTIVITY)

    assert activity["activity_id"] == "red-mug-mouth-return"
    assert activity["target_request"]["resolution_rule"] == {
        "method": "sole_manipulated_instance",
        "expected_source_node_key": None,
    }
    assert activity["seeds"] == {
        "master": 2026081500,
        "embodiedgen_image": 2026081501,
        "embodiedgen_asset": 2026081502,
        "embodiedgen_layout": 2026081503,
        "interactmove": 2026081504,
        "intermimic": 2026081505,
        "render": 2026081506,
    }


def test_generation_protocol_is_real_robot_free_path() -> None:
    runner = _load_runner()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    generation = runner._generation_config(protocol)

    assert generation["gpt_model"] == "gpt-4.1"
    assert generation["text_to_image_backend"] == "sd35"
    assert generation["image_to_3d_backend"] == "SAM3D"
    assert generation["invoke_upstream_sim_cli"] is False
    assert generation["robot_actor_loaded"] is False
    assert generation["background_dataset"][
        "retrieval_is_fresh_background_generation"
    ] is False


def test_gpt_config_returns_only_redacted_metadata(tmp_path: Path) -> None:
    runner = _load_runner()
    config = tmp_path / "gpt_config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "agent_type": "openai-platform",
                "openai-platform": {
                    "endpoint": "https://api.openai.com/v1",
                    "api_key": "test-secret-value",
                    "api_version": None,
                    "model_name": "gpt-5.4",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)

    try:
        metadata = runner._configure_gpt(config)
        assert metadata == {
            "agent_type": "openai-platform",
            "endpoint": "https://api.openai.com/v1/",
            "api_version": None,
            "model_name": "gpt-5.4",
            "api_key_present": True,
        }
        assert "api_key" not in metadata
    finally:
        os.environ.pop("API_KEY", None)
        os.environ.pop("ENDPOINT", None)
        os.environ.pop("MODEL_NAME", None)


def test_public_platform_client_explicitly_disables_azure_fallback() -> None:
    runner = _load_runner()

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModule:
        GPTclient = FakeClient
        GPT_CLIENT = object()

    os.environ["ENDPOINT"] = "https://api.openai.com/v1/"
    os.environ["API_KEY"] = "test-secret-value"
    os.environ["MODEL_NAME"] = "gpt-4.1"
    try:
        client = runner._install_openai_platform_client(FakeModule)
        assert FakeModule.GPT_CLIENT is client
        assert client.kwargs["api_version"] is None
        assert client.kwargs["endpoint"] == "https://api.openai.com/v1/"
        assert client.kwargs["model_name"] == "gpt-4.1"
        assert client.kwargs["check_connection"] is False
    finally:
        os.environ.pop("ENDPOINT", None)
        os.environ.pop("API_KEY", None)
        os.environ.pop("MODEL_NAME", None)


def test_generation_help_states_actual_and_deferred_boundaries() -> None:
    runner = _load_runner()
    help_text = " ".join(runner._parser().format_help().split())

    assert "GPT layout" in help_text
    assert "SD3.5" in help_text
    assert "SAM3D" in help_text
    assert "never calls sim_cli" in help_text
    assert "never loads a robot actor" in help_text
