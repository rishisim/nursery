from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

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

    assert generation["gpt_api"] == "responses"
    assert generation["gpt_model"] == "gpt-5.6-luna"
    assert generation["gpt_reasoning_effort"] == "none"
    assert generation["openai_sdk_version"] == "3.1.0"
    assert generation["text_to_image_backend"] == "sd35"
    assert generation["image_to_3d_backend"] == "SAM3D"
    assert generation["sam3d_model_repository"] == "facebook/sam-3d-objects"
    assert generation["sam3d_model_commit"] == (
        "2e73555018d2741ccd486e56c24fac41155a1dc6"
    )
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
                    "model_name": "gpt-5.6-luna",
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
            "api_mode": "responses",
            "endpoint": "https://api.openai.com/v1/",
            "api_version": None,
            "model_name": "gpt-5.6-luna",
            "reasoning_effort": "none",
            "sdk_version": "3.1.0",
            "api_key_present": True,
        }
        assert "api_key" not in metadata
    finally:
        os.environ.pop("API_KEY", None)
        os.environ.pop("ENDPOINT", None)
        os.environ.pop("MODEL_NAME", None)


def test_public_platform_client_uses_responses_without_chat_or_azure(
    tmp_path: Path,
) -> None:
    runner = _load_runner()

    class FakeResponses:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(output_text="YES")

    fake_responses = FakeResponses()
    fake_sdk_client = SimpleNamespace(responses=fake_responses)

    class FakeModule:
        GPT_CLIENT = object()

    os.environ["ENDPOINT"] = "https://api.openai.com/v1/"
    os.environ["API_KEY"] = "test-secret-value"
    os.environ["MODEL_NAME"] = "gpt-5.6-luna"
    try:
        adapter = runner._install_openai_responses_client(
            FakeModule, client=fake_sdk_client
        )
        assert FakeModule.GPT_CLIENT is adapter
        assert adapter.endpoint == "https://api.openai.com/v1/"
        assert adapter.model_name == "gpt-5.6-luna"

        png = tmp_path / "canary.png"
        jpeg = tmp_path / "canary.jpg"
        png.write_bytes(b"png-payload")
        jpeg.write_bytes(b"jpeg-payload")
        response = adapter.query(
            "Judge these images.",
            image_base64=[png, jpeg],
            system_role="Return YES or NO.",
            params={
                "temperature": 1.0,
                "top_p": 0.95,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.5,
                "max_tokens": 500,
            },
        )
        assert response == "YES"
        payload = fake_responses.calls[-1]
        assert "messages" not in payload
        assert payload["model"] == "gpt-5.6-luna"
        assert payload["reasoning"] == {"effort": "none"}
        assert payload["max_output_tokens"] == 500
        assert payload["store"] is False
        for chat_parameter in (
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "temperature",
            "top_p",
        ):
            assert chat_parameter not in payload
        assert payload["input"][0] == {
            "role": "system",
            "content": [{"type": "input_text", "text": "Return YES or NO."}],
        }
        user_content = payload["input"][1]["content"]
        assert user_content[0] == {
            "type": "input_text",
            "text": "Judge these images.",
        }
        assert [item["type"] for item in user_content] == [
            "input_text",
            "input_image",
            "input_image",
        ]
        assert user_content[1]["image_url"].startswith("data:image/png;base64,")
        assert user_content[2]["image_url"].startswith("data:image/jpeg;base64,")
        assert user_content[1]["detail"] == "auto"
        assert user_content[2]["detail"] == "auto"
    finally:
        os.environ.pop("ENDPOINT", None)
        os.environ.pop("API_KEY", None)
        os.environ.pop("MODEL_NAME", None)


def test_responses_adapter_rejects_unknown_or_conflicting_params() -> None:
    runner = _load_runner()
    adapter = runner._ResponsesGPTClient(
        endpoint="https://api.openai.com/v1/",
        api_key="test-secret-value",
        model_name="gpt-5.6-luna",
        reasoning_effort="none",
        client=SimpleNamespace(responses=SimpleNamespace()),
    )

    assert adapter.query("hello", params={"seed": 1}) is None
    assert adapter.query(
        "hello", params={"max_tokens": 500, "max_output_tokens": 501}
    ) is None


def test_source_state_rejects_tracked_embodiedgen_changes(tmp_path: Path) -> None:
    runner = _load_runner()
    source = tmp_path / "EmbodiedGen"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test User"],
        check=True,
    )
    tracked = source / "tracked.py"
    tracked.write_text("clean = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "tracked.py"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-qm", "fixture"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runner.EMBODIEDGEN_COMMIT = commit

    assert runner._source_state(source)["tracked_files_clean"] is True
    tracked.write_text("clean = False\n", encoding="utf-8")
    try:
        runner._source_state(source)
    except ValueError as exc:
        assert "tracked source checkout is dirty" in str(exc)
    else:
        raise AssertionError("dirty tracked source checkout was accepted")


def test_generation_help_states_actual_and_deferred_boundaries() -> None:
    runner = _load_runner()
    help_text = " ".join(runner._parser().format_help().split())

    assert "GPT layout" in help_text
    assert "SD3.5" in help_text
    assert "SAM3D" in help_text
    assert "never calls sim_cli" in help_text
    assert "never loads a robot actor" in help_text
