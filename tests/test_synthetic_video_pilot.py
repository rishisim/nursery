from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nursery_egobaby_preflight.contract import canonical_json_sha256
from nursery_egobaby_preflight.synthetic_video_pilot import (
    build_model_command,
    compile_prompt,
    compile_retry,
    compile_work_order,
    load_config,
    media_summary,
    mux_modular_audio,
    patch_wan_sdpa_fallback,
    patch_wan_ti2v_import_surface,
    qa_template,
    render_gallery,
    validate_final_media,
    write_work_order,
)

CONFIG_PATH = Path("configs/synthetic_video_public_pilot.json")


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


def test_frozen_public_only_contract(config: dict) -> None:
    assert config["status"] == "FROZEN_READY_FOR_CLOUD_PREFLIGHT"
    assert config["authorization"]["scientific_or_generator_selection_use"] is False
    assert config["authorization"]["governed_or_child_derived_generator_work_authorized"] is False
    assert config["privacy_boundary"]["cloud_outputs_private_by_default"] is True
    prohibited = " ".join(config["privacy_boundary"]["prohibited"])
    assert "ChildLens" in prohibited
    assert "BabyView" in prohibited
    assert config["cloud"]["paid_launch_requires_explicit_spend_confirmation"] is True


def test_compact_cloud_preflight_matches_frozen_protocol(config: dict) -> None:
    result = json.loads(
        Path("results/synthetic_video_public_pilot_preflight.json").read_text()
    )
    assert result["status"] == "PASS"
    assert result["protocol_sha256"] == canonical_json_sha256(config)
    assert result["inference_executed"] is False
    assert result["restricted_or_child_derived_input_used"] is False
    assert result["persistence"]["private_verified"] is True
    assert result["next_gate"]["maximum_preview_gpu_charge_usd"] == 20.0


def test_preview_compiles_exactly_eight_initial_attempts(config: dict) -> None:
    order = compile_work_order(config, "preview", "preview-test")
    assert len(order["attempts"]) == 8
    assert [attempt["family"] for attempt in order["attempts"]] == ["wan"] * 4 + ["ltx"] * 4
    assert {attempt["seed"] for attempt in order["attempts"]} == {314159}
    assert {attempt["attempt_number"] for attempt in order["attempts"]} == {1}
    assert all(attempt["corrective_category"] is None for attempt in order["attempts"])
    assert order["scientific_or_generator_selection_use"] is False
    assert order["protocol_sha256"] == canonical_json_sha256(config)


def test_cloud_preflight_compiles_no_inference_attempts(config: dict) -> None:
    order = compile_work_order(config, "cloud_preflight", "preflight-test", family="wan")
    assert order["attempts"] == []
    assert order["families"] == ["wan"]


def test_model_specific_prompts_preserve_one_semantic_plan(config: dict) -> None:
    wan = compile_prompt(config, "wan", "pick_up")
    ltx = compile_prompt(config, "ltx", "pick_up")
    shared = config["scenes"][0]["prompt_en"]
    assert shared in wan
    assert shared in ltx
    assert config["prompt_contract"]["wan_audio_instruction"] in wan
    assert config["prompt_contract"]["ltx_modular_audio_instruction"] in ltx
    assert "ChildLens" not in wan + ltx
    with pytest.raises(ValueError, match="allowed only"):
        compile_prompt(config, "wan", "pick_up", joint_audio_diagnostic=True)


def test_retry_uses_only_frozen_corrective_suffix(config: dict) -> None:
    initial = compile_work_order(config, "preview", "preview-test", family="wan")["attempts"][0]
    retry = compile_retry(config, initial, "identity")
    assert retry["attempt_number"] == 2
    assert retry["corrective_category"] == "identity"
    assert retry["prompt"].endswith(config["corrective_suffixes"]["identity"])
    assert retry["attempt_id"].endswith("__a2__modular")
    with pytest.raises(ValueError, match="unknown corrective"):
        compile_retry(config, initial, "make_it_better")


def test_commands_use_pinned_official_runners_and_common_delivery(config: dict, tmp_path: Path) -> None:
    order = compile_work_order(config, "preview", "preview-test")
    wan_attempt = order["attempts"][0]
    ltx_attempt = order["attempts"][4]
    wan = build_model_command(
        config,
        wan_attempt,
        source_root=tmp_path / "Wan2.2",
        weights_root=tmp_path / "wan-weights",
        raw_output=tmp_path / "wan.mp4",
        python_executable="/env/wan/python",
    )
    assert wan.argv[:4] == (
        "/env/wan/python",
        str((tmp_path / "Wan2.2" / "generate.py").resolve()),
        "--task",
        "ti2v-5B",
    )
    assert "--use_prompt_extend" not in wan.argv
    assert wan.argv[wan.argv.index("--frame_num") + 1] == "121"
    assert wan.argv[wan.argv.index("--size") + 1] == "1280*704"

    ltx = build_model_command(
        config,
        ltx_attempt,
        source_root=tmp_path / "LTX-2",
        weights_root=tmp_path / "ltx-weights",
        raw_output=tmp_path / "ltx.mp4",
    )
    assert ltx.argv[:4] == ("uv", "run", "python", "-m")
    assert ltx.argv[4] == "ltx_pipelines.distilled"
    assert "--enhance-prompt" not in ltx.argv
    assert ltx.argv[ltx.argv.index("--num-frames") + 1] == "121"
    assert ltx.argv[ltx.argv.index("--offload") + 1] == "cpu"


def test_wan_sdpa_runtime_adaptation_is_exact_and_auditable(tmp_path: Path) -> None:
    model = tmp_path / "wan" / "modules" / "model.py"
    model.parent.mkdir(parents=True)
    model.write_text(
        "import torch\n"
        "from .attention import flash_attention\n"
        "\n"
        "def forward(q, k, v):\n"
        "    return flash_attention(q, k, v)\n"
    )
    package = tmp_path / "wan" / "__init__.py"
    package.write_text(
        "from . import configs, distributed, modules\n"
        "from .image2video import WanI2V\n"
        "from .speech2video import WanS2V\n"
        "from .text2video import WanT2V\n"
        "from .textimage2video import WanTI2V\n"
        "from .animate import WanAnimate"
    )

    records = [
        patch_wan_ti2v_import_surface(tmp_path),
        patch_wan_sdpa_fallback(tmp_path),
    ]

    assert [record["path"] for record in records] == [
        "wan/__init__.py",
        "wan/modules/model.py",
    ]
    assert all(
        record["original_sha256"] != record["patched_sha256"] for record in records
    )
    assert package.read_text().endswith("from .textimage2video import WanTI2V")
    assert "WanS2V" not in package.read_text()
    assert "attention as flash_attention" in model.read_text()
    with pytest.raises(RuntimeError, match="exactly one eager optional import block"):
        patch_wan_ti2v_import_surface(tmp_path)
    with pytest.raises(RuntimeError, match="exactly one strict attention import"):
        patch_wan_sdpa_fallback(tmp_path)


def test_work_order_writes_qa_templates_without_media(config: dict, tmp_path: Path) -> None:
    order = compile_work_order(config, "preview", "preview-test", family="wan")
    path = write_work_order(tmp_path, config, order)
    assert path.exists()
    first = order["attempts"][0]
    qa = json.loads((tmp_path / first["paths"]["qa_template"]).read_text())
    assert qa == qa_template(config, first)
    assert qa["rater_id"] is None
    assert len(qa["items"]) == 8


def test_gallery_is_side_by_side_and_marks_missing_media(config: dict, tmp_path: Path) -> None:
    order = compile_work_order(config, "preview", "preview-test")
    write_work_order(tmp_path, config, order)
    output = render_gallery(config, order, tmp_path)
    page = output.read_text()
    assert "WAN" in page
    assert "LTX" in page
    assert "Das ist der Becher" in page
    assert "Media not present yet" in page
    assert "Engineering preview only" in page


def test_runner_merges_downloaded_family_roots(config: dict, tmp_path: Path) -> None:
    run_root = tmp_path / "downloaded-run"
    for family in ("wan", "ltx"):
        family_root = run_root / "families" / family
        order = compile_work_order(config, "preview", "preview-test", family=family)
        write_work_order(family_root, config, order)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_synthetic_video_public_pilot.py",
            "gallery",
            "--run-root",
            str(run_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    merged = json.loads((run_root / "work_order.json").read_text())
    assert payload["attempt_count"] == 8
    assert [attempt["family"] for attempt in merged["attempts"][:2]] == ["wan", "ltx"]
    assert all(
        attempt["paths"]["final_video"].startswith(f"families/{attempt['family']}/")
        for attempt in merged["attempts"]
    )
    assert (run_root / "gallery" / "index.html").exists()


def test_cloud_plan_is_bound_to_passed_preflight_and_twenty_dollar_cap() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_synthetic_video_public_pilot.py",
            "cloud-plan",
            "--run-id",
            "preview-test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "READY_REQUIRES_EXPLICIT_SPEND_CONFIRMATION"
    assert plan["maximum_preview_gpu_charge_usd"] == 20.0
    assert plan["launch_order"] == ["wan", "ltx"]
    assert [job["family"] for job in plan["jobs"]] == ["wan", "ltx"]
    assert all(job["flavor"] == "a100-large" for job in plan["jobs"])
    assert all(job["timeout"] == "4h" for job in plan["jobs"])
    assert all(job["maximum_charge_usd"] == 10.0 for job in plan["jobs"])
    assert all(job["secrets"] == ["HF_TOKEN"] for job in plan["jobs"])
    assert all(
        plan["validated_nursery_commit"] in job["script"] for job in plan["jobs"]
    )
    assert all("--profile" in job["script_args"] for job in plan["jobs"])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_modular_mux_produces_conforming_media(config: dict, tmp_path: Path) -> None:
    raw = tmp_path / "raw.mp4"
    wav = tmp_path / "utterance.wav"
    video_only = tmp_path / "video_only.mp4"
    final = tmp_path / "final.mp4"
    delivery = config["delivery"]
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
            f"color=c=red:s={delivery['width']}x{delivery['height']}:r={delivery['fps']}",
            "-frames:v",
            str(delivery["num_frames"]),
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
            "sine=frequency=440:duration=1",
            str(wav),
        ],
        check=True,
    )
    mux_modular_audio(
        config,
        "pick_up",
        raw_video=raw,
        tts_wav=wav,
        video_only=video_only,
        final_video=final,
    )
    summary = media_summary(final)
    assert validate_final_media(config, summary) == []
    assert summary["has_audio"] is True
    assert summary["width"] == 1280
    assert summary["height"] == 704
