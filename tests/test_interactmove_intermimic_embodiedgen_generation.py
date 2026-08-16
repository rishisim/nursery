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
    assert generation["image_to_3d_backend"] == "TRELLIS"
    assert generation["trellis"]["source_commit"] == (
        "55a8e8164b195bbf927e0978f00e76c835e6011f"
    )
    assert generation["trellis"]["checkpoint_revision"] == (
        "25e0d31ffbebe4b5a97464dd851910efc3002d96"
    )
    assert generation["sam3d_comparison"]["access_status"] == (
        "pending_not_admitted"
    )
    assert generation["sam3d_comparison"]["blocks_trellis_run"] is False
    assert generation["resume_source"]["job_id"] == "328320"
    assert generation["asset_resume_source"]["job_id"] == "328381"
    assert generation["asset_resume_source"]["nodes"]["red mug"] == {
        "prompt": (
            "glossy crimson ceramic mug with curved handle, thick rim, and "
            "smooth reflective surface"
        ),
        "reuse_result": False,
        "rejected_results": [
            {
                "job_id": "328392",
                "generation_receipt_sha256": (
                    "6a5160b0f5e0a3bef4ae4a2a51831426adfbcb76b65d0b6b944c32bfef41ef3c"
                ),
                "reason": "malformed_vertical_side_protrusion",
            },
            {
                "job_id": "328413",
                "generation_receipt_sha256": (
                    "c3e4f6670400bb7a32017da8480627c389879df741819b75e9248ddbd1065f74"
                ),
                "reason": (
                    "duplicate_handles_found_by_target_specific_multiview_review"
                ),
            },
        ],
    }
    target_policy = generation["asset_resume_source"]["target_geometry_policy"]
    assert target_policy["source_node_key"] == "red mug"
    assert target_policy["exact_handle_count"] == 1
    assert target_policy["trellis_retry_seeds"] == [33936, 62468]
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
    trellis = source / "thirdparty" / "TRELLIS"
    flexicubes = trellis / "trellis" / "representations" / "mesh" / "flexicubes"
    for path, constant in (
        (trellis, "TRELLIS_SOURCE_COMMIT"),
        (flexicubes, "TRELLIS_FLEXICUBES_COMMIT"),
    ):
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "config", "user.name", "Test User"],
            check=True,
        )
        marker = path / "marker.txt"
        marker.write_text("pinned\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(path), "add", "marker.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(path), "commit", "-qm", "fixture"], check=True
        )
        setattr(
            runner,
            constant,
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        )

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
    assert "TRELLIS" in help_text
    assert "never calls sim_cli" in help_text
    assert "never loads a robot actor" in help_text


def test_resume_conditioning_image_is_hash_preserving(tmp_path: Path) -> None:
    runner = _load_runner()
    source_image = tmp_path / "source.png"
    source_raw = tmp_path / "source_raw.png"
    source_image.write_bytes(b"accepted-image")
    source_raw.write_bytes(b"accepted-raw-image")
    original_calls = []

    def original(*args, **kwargs):
        original_calls.append((args, kwargs))
        return False

    module = SimpleNamespace(text_to_image=original)
    reused = runner._install_resume_conditioning_images(
        module,
        {
            "table": {
                "prompt": "frozen table prompt",
                "image_path": source_image,
                "raw_image_path": source_raw,
                "image_sha256": runner._sha256(source_image),
                "raw_image_sha256": runner._sha256(source_raw),
            }
        },
        initial_image_seed=2026081501,
    )
    destination = tmp_path / "output" / "table.png"
    destination.parent.mkdir()

    assert module.text_to_image(
        "frozen table prompt",
        str(destination),
        4,
        25,
        7.0,
        1,
        seed=2026081501,
    ) is True
    assert destination.read_bytes() == source_image.read_bytes()
    assert destination.with_name("table_raw.png").read_bytes() == (
        source_raw.read_bytes()
    )
    assert reused == {"table"}
    assert original_calls == []

    assert module.text_to_image(
        "frozen table prompt",
        str(destination),
        4,
        25,
        7.0,
        1,
        seed=50494,
    ) is True
    assert destination.read_bytes() == source_image.read_bytes()

    assert module.text_to_image(
        "another prompt",
        str(tmp_path / "output" / "mug.png"),
        4,
        25,
        7.0,
        1,
        seed=2026081501,
    ) is False
    assert len(original_calls) == 1


def test_resume_conditioning_image_maps_native_spaces_to_filename_underscores(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    source_image = tmp_path / "source.png"
    source_raw = tmp_path / "source_raw.png"
    source_image.write_bytes(b"accepted-red-mug-image")
    source_raw.write_bytes(b"accepted-red-mug-raw-image")
    module = SimpleNamespace(
        text_to_image=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected SD3.5 generation")
        )
    )
    reused = runner._install_resume_conditioning_images(
        module,
        {
            "red mug": {
                "prompt": "frozen red mug prompt",
                "image_path": source_image,
                "raw_image_path": source_raw,
                "image_sha256": runner._sha256(source_image),
                "raw_image_sha256": runner._sha256(source_raw),
            }
        },
        initial_image_seed=2026081501,
    )
    destination = tmp_path / "output" / "red_mug.png"
    destination.parent.mkdir()

    assert module.text_to_image(
        "frozen red mug prompt",
        str(destination),
        4,
        25,
        7.0,
        1,
        seed=2026081501,
    ) is True
    assert destination.read_bytes() == source_image.read_bytes()
    assert reused == {"red mug"}


def test_resume_assets_verifies_complete_receipt_manifest(tmp_path: Path) -> None:
    runner = _load_runner()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    generation = runner._generation_config(protocol)
    generation["asset_resume_source"]["nodes"] = {
        "table": {
            "prompt": "table prompt",
            "reuse_result": True,
            "rejected_results": [],
        }
    }
    root = tmp_path / "partial"
    files = {
        "images/table.png": b"image",
        "images/table_raw.png": b"raw-image",
        "asset3d/table/result/table.urdf": b"<robot/>",
        **{
            f"asset3d/table/result/renders/image_color/{index:04d}.png": (
                f"view-{index}".encode()
            )
            for index in range(4)
        },
    }
    records = []
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": runner._sha256(path),
            }
        )
    receipt = {
        "status": "failed",
        "models": {
            "image_to_3d_backend": "TRELLIS",
            "trellis": generation["trellis"],
        },
        "error": {"message": "resume initial image seed mismatch for table"},
        "files": records,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generation["asset_resume_source"]["generation_receipt_sha256"] = (
        runner._sha256(receipt_path)
    )
    args = SimpleNamespace(
        resume_assets=root,
        resume_asset_receipt=receipt_path,
    )

    resumed = runner._resume_assets(args, generation)

    assert set(resumed["nodes"]) == {"table"}
    assert resumed["nodes"]["table"]["reuse_result"] is True
    assert len(resumed["nodes"]["table"]["render_paths"]) == 4
    assert resumed["nodes"]["table"]["result_manifest_sha256"]
    (root / "asset3d/table/result/table.urdf").write_bytes(b"tampered")
    try:
        runner._resume_assets(args, generation)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered resumed TRELLIS asset was accepted")


def test_rejected_result_reuses_only_hash_bound_conditioning_image(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    image = tmp_path / "red_mug.png"
    raw_image = tmp_path / "red_mug_raw.png"
    image.write_bytes(b"accepted-red-mug-image")
    raw_image.write_bytes(b"accepted-red-mug-raw-image")
    rejections = [
        {
            "job_id": "328392",
            "generation_receipt_sha256": "a" * 64,
            "reason": "malformed_vertical_side_protrusion",
        }
    ]

    records = runner._conditioning_records(
        {"images": {}},
        {
            "nodes": {
                "red mug": {
                    "prompt": "frozen red mug prompt",
                    "reuse_result": False,
                    "rejected_results": rejections,
                    "image_path": image,
                    "raw_image_path": raw_image,
                }
            }
        },
        asset_source_job_id="328381",
    )

    assert records["red mug"] == {
        "prompt": "frozen red mug prompt",
        "image_path": image,
        "raw_image_path": raw_image,
        "image_sha256": runner._sha256(image),
        "raw_image_sha256": runner._sha256(raw_image),
        "source_job_id": "328381",
        "rejected_results": rejections,
    }


def test_target_geometry_quality_requires_exact_plain_yes(tmp_path: Path) -> None:
    runner = _load_runner()
    calls = []

    class Client:
        def query(self, prompt, *, image_base64, system_role):
            calls.append((prompt, image_base64, system_role))
            return "  NO — the mug has two handles.  "

    views = [tmp_path / f"{index:04d}.png" for index in range(4)]
    policy = {
        "review_prompt": "Require exactly one connected mug handle.",
    }

    result = runner._query_target_geometry_quality(Client(), policy, views)

    assert result == "NO — the mug has two handles."
    assert calls == [
        (
            "Require exactly one connected mug handle.",
            views,
            (
                "You are a strict 3D asset geometry inspector. Follow the "
                "requested exact output format."
            ),
        )
    ]


def test_multiview_quality_prompt_disambiguates_camera_views() -> None:
    runner = _load_runner()
    prompt = " ".join(runner.MULTIVIEW_QUALITY_PREAMBLE.split())

    assert "same single generated 3D asset" in prompt
    assert "not multiple object instances" in prompt
