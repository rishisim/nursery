from __future__ import annotations

import io
import json
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest

from nursery_egobaby_preflight.contract import (
    canonical_json_sha256,
    file_sha256,
)
from nursery_egobaby_preflight.synthetic_video_pilot import (
    compile_prompt,
    load_config as load_public_pilot_config,
    media_summary,
    validate_final_media,
)
from nursery_egobaby_preflight.synthetic_video_quality_ceiling import (
    FalQueueClient,
    _audio_payload_sha256,
    compile_comparison_work_order,
    execute_comparison,
    load_quality_config,
    normalize_seedance_with_ltx_audio,
    planned_cost_usd,
    render_blinded_gallery,
    validate_quality_config,
    validate_work_order,
    write_comparison_work_order,
)

QUALITY_CONFIG_PATH = Path("configs/synthetic_video_quality_ceiling.json")
PUBLIC_CONFIG_PATH = Path("configs/synthetic_video_public_pilot.json")


@pytest.fixture
def quality_config() -> dict:
    return load_quality_config(QUALITY_CONFIG_PATH)


@pytest.fixture
def public_config() -> dict:
    return load_public_pilot_config(PUBLIC_CONFIG_PATH)


def test_quality_ceiling_contract_is_public_only_and_cost_bounded(
    quality_config: dict,
    public_config: dict,
) -> None:
    validate_quality_config(quality_config, public_config)
    assert planned_cost_usd(quality_config) == Decimal("6.0680")
    assert quality_config["candidate"]["endpoint"] == (
        "bytedance/seedance-2.0/text-to-video"
    )
    assert quality_config["candidate"]["request"]["generate_audio"] is False
    assert "seed" not in quality_config["candidate"]["request"]
    assert quality_config["candidate"]["seed_control"] == (
        "UNSUPPORTED_BY_CURRENT_LIVE_INPUT_SCHEMA"
    )
    assert quality_config["candidate"]["platform_headers"]["X-Fal-No-Retry"] == "1"
    assert quality_config["candidate"]["platform_headers"]["X-Fal-Store-IO"] == "0"
    assert quality_config["authorization"]["scientific_training_use_authorized"] is False
    assert quality_config["governance"]["seedance_output_as_training_data"].startswith(
        "BLOCKED"
    )


def test_comparison_compiles_exact_ltx_prompts_and_four_seedance_requests(
    quality_config: dict,
    public_config: dict,
) -> None:
    order = compile_comparison_work_order(
        quality_config,
        public_config,
        "seedance-preview-test",
    )
    assert len(order["attempts"]) == 4
    assert len(order["pairs"]) == 4
    assert order["planned_cost_usd"] == 6.068
    assert order["scientific_training_use_authorized"] is False
    assert set(order["blinding_key"]) == {
        card["blinded_display_id"]
        for pair in order["pairs"]
        for card in pair["presentation"]
    }
    public_order = {
        key: value for key, value in order.items() if key != "blinding_key"
    }
    expected_hash = public_order.pop("work_order_sha256")
    assert canonical_json_sha256(public_order) == expected_hash
    for attempt in order["attempts"]:
        expected_prompt = compile_prompt(
            public_config,
            "ltx",
            attempt["scene_id"],
        )
        assert attempt["prompt"] == expected_prompt
        assert attempt["request"]["prompt"] == expected_prompt
        assert "seed" not in attempt["request"]
        assert attempt["provider_seed_requested"] is None
        assert attempt["request"]["duration"] == "5"
        assert attempt["planned_charge_usd"] == 1.517


def test_work_order_rejects_rehashed_request_tampering(
    quality_config: dict,
    public_config: dict,
) -> None:
    order = compile_comparison_work_order(
        quality_config,
        public_config,
        "seedance-preview-test",
    )
    order["attempts"][0]["request"]["prompt"] = "tampered public prompt"
    unhashed = {
        key: value
        for key, value in order.items()
        if key not in {"work_order_sha256", "blinding_key"}
    }
    order["work_order_sha256"] = canonical_json_sha256(unhashed)
    with pytest.raises(ValueError, match="work-order attempts changed"):
        validate_work_order(quality_config, public_config, order)


def test_paid_runner_requires_exact_frozen_spend(
    quality_config: dict,
    public_config: dict,
    tmp_path: Path,
) -> None:
    order = compile_comparison_work_order(
        quality_config,
        public_config,
        "seedance-preview-test",
    )
    with pytest.raises(ValueError, match="exactly match"):
        execute_comparison(
            quality_config,
            public_config,
            order,
            repository_root=tmp_path,
            run_root=tmp_path / "run",
            approved_spend_usd=Decimal("7"),
            api_key="unused",
        )


def test_work_order_separates_blinding_key_and_family_free_qa(
    quality_config: dict,
    public_config: dict,
    tmp_path: Path,
) -> None:
    order = compile_comparison_work_order(
        quality_config,
        public_config,
        "seedance-preview-test",
    )
    path = write_comparison_work_order(tmp_path, order, public_config)
    public_order = json.loads(path.read_text())
    key = json.loads((tmp_path / "blinding_key.json").read_text())
    assert "blinding_key" not in public_order
    assert canonical_json_sha256(key) == public_order["blinding_key_sha256"]
    qa_paths = sorted((tmp_path / "qa").glob("*.json"))
    assert len(qa_paths) == 8
    for qa_path in qa_paths:
        qa = json.loads(qa_path.read_text())
        assert "family" not in qa
        assert "attempt_id" not in qa
        assert qa["blinded_display_id"] == qa_path.stem
        assert len(qa["items"]) == 8


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._payload.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_fal_client_applies_privacy_headers_without_serializing_key(
    quality_config: dict,
    tmp_path: Path,
) -> None:
    calls = []
    responses = iter(
        [
            b'{"request_id":"request-1","response_url":"https://queue.fal.run/response","status_url":"https://queue.fal.run/status"}',
            b'{"status":"COMPLETED","metrics":{"inference_time":12.5}}',
            b'{"video":{"url":"https://v3b.fal.media/files/b/a/video.mp4"},"seed":314159}',
            b'{"token":"cdn-token"}',
            b"video-bytes",
        ]
    )

    def opener(request: object, timeout: float) -> _FakeResponse:
        calls.append((request, timeout))
        return _FakeResponse(next(responses))

    client = FalQueueClient(
        "secret-value",
        quality_config["candidate"]["platform_headers"],
        opener=opener,
    )
    submission = client.submit(
        quality_config["candidate"]["endpoint_url"],
        {"prompt": "public prompt"},
    )
    status = client.status(submission["status_url"])
    result = client.result(submission["response_url"])
    destination = tmp_path / "raw.mp4"
    client.download_private_file(result["video"]["url"], destination)

    assert status["status"] == "COMPLETED"
    assert destination.read_bytes() == b"video-bytes"
    submit_headers = {
        key.lower(): value for key, value in calls[0][0].header_items()
    }
    assert submit_headers["authorization"] == "Key secret-value"
    assert submit_headers["x-fal-no-retry"] == "1"
    assert submit_headers["x-fal-store-io"] == "0"
    download_headers = {
        key.lower(): value for key, value in calls[-1][0].header_items()
    }
    assert download_headers["authorization"] == "Bearer cdn-token"
    serialized = json.dumps(
        {
            "submission": submission,
            "status": status,
            "result": result,
        }
    )
    assert "secret-value" not in serialized
    assert "cdn-token" not in serialized


def test_fal_client_rejects_untrusted_credential_destination(
    quality_config: dict,
    tmp_path: Path,
) -> None:
    client = FalQueueClient(
        "secret-value",
        quality_config["candidate"]["platform_headers"],
        opener=lambda *_args, **_kwargs: pytest.fail("network opener was called"),
    )
    with pytest.raises(ValueError, match="untrusted URL"):
        client.status("https://example.com/request/status")
    with pytest.raises(ValueError, match="untrusted URL"):
        client.download_private_file(
            "https://example.com/video.mp4",
            tmp_path / "unused.mp4",
        )


def test_gallery_hides_family_names_and_paths(
    quality_config: dict,
    public_config: dict,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    repository_root = tmp_path / "repository"
    order = compile_comparison_work_order(
        quality_config,
        public_config,
        "seedance-preview-test",
    )
    write_comparison_work_order(run_root, order, public_config)
    for pair in order["pairs"]:
        for card in pair["presentation"]:
            if card["family"] == "ltx":
                path = repository_root / card["paths"]["final_video"]
            else:
                path = run_root / card["paths"]["final_video"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")
    public_order = json.loads((run_root / "work_order.json").read_text())
    gallery = render_blinded_gallery(
        quality_config,
        public_config,
        public_order,
        repository_root=repository_root,
        run_root=run_root,
    )
    page = gallery.read_text()
    assert "Clip A" in page and "Clip B" in page
    assert "ltx" not in page.lower()
    assert "seedance" not in page.lower()
    assert "/ltx/" not in page
    assert "/seedance/" not in page
    aliases = list((gallery.parent / "media").glob("*.mp4"))
    assert len(aliases) == 8
    assert all(path.is_symlink() for path in aliases)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_seedance_normalization_copies_exact_ltx_audio(
    public_config: dict,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "seedance_raw.mp4"
    baseline = tmp_path / "ltx_final.mp4"
    video_only = tmp_path / "video_only.mp4"
    final = tmp_path / "final.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=1280x720:r=24",
            "-frames:v",
            "120",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(raw),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=1280x704:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=5.041666667",
            "-frames:v",
            "121",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(baseline),
        ],
        check=True,
    )
    hashes = normalize_seedance_with_ltx_audio(
        public_config,
        raw_video=raw,
        ltx_final_video=baseline,
        video_only=video_only,
        final_video=final,
    )
    summary = media_summary(final)
    assert validate_final_media(public_config, summary) == []
    assert summary["width"] == 1280
    assert summary["height"] == 704
    assert summary["frame_count"] == 121
    assert hashes["baseline_audio_payload_sha256"] == _audio_payload_sha256(
        baseline
    )
    assert hashes["candidate_audio_payload_sha256"] == _audio_payload_sha256(
        final
    )
    assert file_sha256(str(final)) == summary["sha256"]
