"""Deterministic compiler and records for the public-only video pilot."""

from __future__ import annotations

import html
import json
import math
import os
import re
import shlex
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from .contract import canonical_json_bytes, canonical_json_sha256, file_sha256

FAMILY_IDS = ("wan", "ltx")
PROFILE_IDS = ("cloud_preflight", "preview", "formal")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
WAN_STRICT_ATTENTION_IMPORT = b"from .attention import flash_attention\n"
WAN_FALLBACK_ATTENTION_IMPORT = b"from .attention import attention as flash_attention\n"
WAN_EAGER_OPTIONAL_IMPORTS = (
    b"from .image2video import WanI2V\n"
    b"from .speech2video import WanS2V\n"
    b"from .text2video import WanT2V\n"
    b"from .textimage2video import WanTI2V\n"
    b"from .animate import WanAnimate"
)
WAN_TI2V_IMPORT = b"from .textimage2video import WanTI2V"


@dataclass(frozen=True)
class CommandSpec:
    """One external model command plus its required working directory."""

    argv: tuple[str, ...]
    cwd: str

    def shell_preview(self) -> str:
        return f"cd {shlex.quote(self.cwd)} && {shlex.join(self.argv)}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique(values: Iterable[Any], label: str) -> None:
    materialized = list(values)
    _require(len(materialized) == len(set(materialized)), f"{label} must be unique")


def _validate_revision(value: Any, label: str) -> None:
    _require(
        isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value)),
        f"{label} must be a full 40-character lowercase Git revision",
    )


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject config drift that would weaken the frozen public-only boundary."""

    _require(config.get("schema_version") == 1, "schema_version must be 1")
    _require(config.get("pilot_id") == "synthetic-video-public-only-qualitative-pilot", "unexpected pilot_id")

    authorization = config.get("authorization", {})
    _require(authorization.get("public_only_preview_authorized") is True, "public preview is not authorized")
    _require(authorization.get("scientific_or_generator_selection_use") is False, "preview cannot select a model")
    _require(
        authorization.get("governed_or_child_derived_generator_work_authorized") is False,
        "public preview must not authorize governed generator work",
    )

    privacy = config.get("privacy_boundary", {})
    prohibited_text = " ".join(privacy.get("prohibited", [])).lower()
    _require("childlens" in prohibited_text and "babyview" in prohibited_text, "restricted sources must be prohibited")
    _require(privacy.get("cloud_outputs_private_by_default") is True, "cloud output must default to private")

    delivery = config.get("delivery", {})
    _require(delivery.get("width") == 1280 and delivery.get("height") == 704, "delivery must stay 1280x704")
    _require(delivery.get("fps") == 24 and delivery.get("num_frames") == 121, "delivery must stay 121 frames at 24 fps")
    expected_duration = delivery["num_frames"] / delivery["fps"]
    _require(
        math.isclose(delivery.get("nominal_duration_s", 0.0), expected_duration, abs_tol=1e-12),
        "nominal duration must equal frame count divided by fps",
    )

    models = config.get("models", {})
    _require(tuple(models) == FAMILY_IDS, f"model order must be exactly {FAMILY_IDS!r}")
    for family in FAMILY_IDS:
        model = models[family]
        _validate_revision(model.get("source_commit"), f"{family} source_commit")
        _validate_revision(model.get("weights_revision"), f"{family} weights_revision")
    _validate_revision(models["ltx"].get("text_encoder_revision"), "ltx text_encoder_revision")
    _require(models["wan"].get("prompt_extension") is False, "Wan prompt extension must remain disabled")
    _require(models["ltx"].get("prompt_enhancement") is False, "LTX prompt enhancement must remain disabled")

    tts = config.get("tts", {})
    _validate_revision(tts.get("voice_revision"), "TTS voice_revision")
    _require(tts.get("voice_cloning") is False, "voice cloning is prohibited")
    _require(tts.get("voice_id") == "de_DE-thorsten-medium", "unexpected TTS voice")

    scenes = config.get("scenes", [])
    _require(len(scenes) == 4, "exactly four scenes are required")
    scene_ids = [scene.get("id") for scene in scenes]
    _require(
        scene_ids == ["pick_up", "transfer", "occlusion_persistence", "action_transition"],
        "scene order or identity changed",
    )
    _unique(scene_ids, "scene IDs")
    for scene in scenes:
        _require(bool(SAFE_ID.fullmatch(scene["id"])), f"unsafe scene id: {scene['id']!r}")
        _require(scene.get("utterance_de", "").strip(), f"{scene['id']} has no German utterance")
        _require(scene.get("prompt_en", "").strip(), f"{scene['id']} has no prompt")
        timeline = scene.get("timeline", [])
        _require(timeline, f"{scene['id']} timeline is empty")
        previous_end = 0.0
        for event in timeline:
            start = float(event["start_s"])
            end = float(event["end_s"])
            _require(0.0 <= start < end <= delivery["nominal_duration_s"] + 0.01, f"{scene['id']} has invalid timeline")
            _require(start >= previous_end - 1e-9, f"{scene['id']} timeline is not monotonic")
            previous_end = end
        target_start, target_end = scene["requested_target_word_interval_s"]
        visible_start, visible_end = scene["visible_referent_interval_s"]
        _require(
            visible_start <= target_start - 0.5 and visible_end >= target_end + 0.5,
            f"{scene['id']} lacks the frozen 500 ms referent padding",
        )
        _require(
            0.0 <= scene["utterance_onset_s"] < delivery["nominal_duration_s"],
            f"{scene['id']} utterance onset is outside the clip",
        )

    profiles = config.get("profiles", {})
    _require(tuple(profiles) == PROFILE_IDS, f"profile order must be exactly {PROFILE_IDS!r}")
    for profile_id, profile in profiles.items():
        _require(profile["families"] == list(FAMILY_IDS), f"{profile_id} family order changed")
        _require(set(profile["scene_ids"]).issubset(set(scene_ids)), f"{profile_id} has an unknown scene")
        _unique(profile["scene_ids"], f"{profile_id} scene IDs")
        _unique(profile["seeds"], f"{profile_id} seeds")
        _require(all(isinstance(seed, int) and seed >= 0 for seed in profile["seeds"]), f"{profile_id} has invalid seeds")
    _require(profiles["cloud_preflight"]["execute_generation"] is False, "cloud preflight must not infer")
    _require(profiles["cloud_preflight"]["max_attempts_per_seed"] == 0, "cloud preflight must have zero attempts")
    _require(profiles["preview"]["seeds"] == [314159], "preview must use one frozen seed")
    _require(profiles["preview"]["max_attempts_per_seed"] == 1, "preview cannot retry")
    _require(profiles["formal"]["max_attempts_per_seed"] == 2, "formal profile must preserve one retry")

    corrective = config.get("corrective_suffixes", {})
    _require(
        tuple(corrective) == ("hand", "identity", "camera", "transition", "referent_timing", "safety"),
        "corrective taxonomy changed",
    )
    qa = config.get("human_qa", {})
    _require(len(qa.get("items", [])) == 8, "exactly eight human QA items are required")
    _unique((item["id"] for item in qa["items"]), "human QA IDs")

    cloud = config.get("cloud", {})
    _require(cloud.get("output_repository_private") is True, "cloud repository must be private")
    _require(cloud.get("paid_launch_requires_explicit_spend_confirmation") is True, "paid launch needs confirmation")
    _require(float(cloud.get("maximum_preview_gpu_charge_usd", 0)) > 0, "cloud cost cap is missing")


def scene_by_id(config: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    for scene in config["scenes"]:
        if scene["id"] == scene_id:
            return scene
    raise KeyError(f"unknown scene {scene_id!r}")


def compile_prompt(
    config: Mapping[str, Any],
    family: str,
    scene_id: str,
    *,
    correction: str | None = None,
    joint_audio_diagnostic: bool = False,
) -> str:
    """Compile one immutable semantic scene into family-specific prompt syntax."""

    if family not in FAMILY_IDS:
        raise ValueError(f"unknown family {family!r}")
    if joint_audio_diagnostic and not (family == "ltx" and scene_id == "pick_up"):
        raise ValueError("joint-audio diagnostic is allowed only for LTX scene pick_up")
    scene = scene_by_id(config, scene_id)
    contract = config["prompt_contract"]
    chunks = [
        contract["shared_prefix"],
        scene["prompt_en"],
        contract["modular_audio_instruction"],
    ]
    if joint_audio_diagnostic:
        chunks.append(contract["joint_audio_diagnostic_instruction"])
    elif family == "ltx":
        chunks.append(contract["ltx_modular_audio_instruction"])
    else:
        chunks.append(contract["wan_audio_instruction"])
    chunks.append(contract["shared_suffix"])
    if correction is not None:
        if correction not in config["corrective_suffixes"]:
            raise ValueError(f"unknown corrective category {correction!r}")
        chunks.append(config["corrective_suffixes"][correction])
    return " ".join(chunk.strip() for chunk in chunks)


def _attempt_id(family: str, scene_id: str, seed: int, attempt_number: int, mode: str = "modular") -> str:
    return f"{family}__{scene_id}__s{seed}__a{attempt_number}__{mode}"


def compile_work_order(
    config: Mapping[str, Any],
    profile_id: str,
    run_id: str,
    *,
    family: str | None = None,
) -> dict[str, Any]:
    """Create a prospective work order; no files, models, or media are touched."""

    validate_config(config)
    if profile_id not in config["profiles"]:
        raise ValueError(f"unknown profile {profile_id!r}")
    if not SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id must contain only lowercase letters, digits, underscores, and hyphens")

    profile = config["profiles"][profile_id]
    families = [family] if family else list(profile["families"])
    if any(item not in profile["families"] for item in families):
        raise ValueError(f"family must be one of {profile['families']!r}")

    attempts: list[dict[str, Any]] = []
    if profile["execute_generation"]:
        for selected_family in families:
            for scene_id in profile["scene_ids"]:
                for seed in profile["seeds"]:
                    prompt = compile_prompt(config, selected_family, scene_id)
                    attempt_id = _attempt_id(selected_family, scene_id, seed, 1)
                    attempts.append(
                        {
                            "attempt_id": attempt_id,
                            "family": selected_family,
                            "scene_id": scene_id,
                            "seed": seed,
                            "attempt_number": 1,
                            "mode": "modular",
                            "corrective_category": None,
                            "prompt": prompt,
                            "prompt_sha256": canonical_json_sha256(prompt),
                            "status": "planned",
                            "paths": {
                                "raw_video": f"attempts/{attempt_id}/raw.mp4",
                                "video_only": f"attempts/{attempt_id}/video_only.mp4",
                                "tts_audio": f"attempts/{attempt_id}/utterance.wav",
                                "final_video": f"attempts/{attempt_id}/final.mp4",
                                "record": f"attempts/{attempt_id}/attempt.json",
                                "qa_template": f"attempts/{attempt_id}/qa.json"
                            }
                        }
                    )

    work_order = {
        "schema_version": 1,
        "pilot_id": config["pilot_id"],
        "run_id": run_id,
        "profile": profile_id,
        "purpose": profile["purpose"],
        "compiled_at": utc_now(),
        "protocol_sha256": canonical_json_sha256(config),
        "scientific_or_generator_selection_use": False,
        "privacy_boundary": config["privacy_boundary"],
        "delivery": config["delivery"],
        "families": families,
        "attempts": attempts,
    }
    work_order["work_order_sha256"] = canonical_json_sha256(work_order)
    return work_order


def compile_retry(
    config: Mapping[str, Any],
    initial_attempt: Mapping[str, Any],
    corrective_category: str,
) -> dict[str, Any]:
    """Compile the single formal retry without allowing free-form prompt edits."""

    if initial_attempt.get("attempt_number") != 1 or initial_attempt.get("mode") != "modular":
        raise ValueError("retry source must be one initial modular attempt")
    family = str(initial_attempt["family"])
    scene_id = str(initial_attempt["scene_id"])
    seed = int(initial_attempt["seed"])
    prompt = compile_prompt(config, family, scene_id, correction=corrective_category)
    retry = {
        **{key: value for key, value in initial_attempt.items() if key not in {"paths", "status", "prompt", "prompt_sha256"}},
        "attempt_id": _attempt_id(family, scene_id, seed, 2),
        "attempt_number": 2,
        "corrective_category": corrective_category,
        "prompt": prompt,
        "prompt_sha256": canonical_json_sha256(prompt),
        "status": "planned",
    }
    attempt_id = retry["attempt_id"]
    retry["paths"] = {
        "raw_video": f"attempts/{attempt_id}/raw.mp4",
        "video_only": f"attempts/{attempt_id}/video_only.mp4",
        "tts_audio": f"attempts/{attempt_id}/utterance.wav",
        "final_video": f"attempts/{attempt_id}/final.mp4",
        "record": f"attempts/{attempt_id}/attempt.json",
        "qa_template": f"attempts/{attempt_id}/qa.json",
    }
    return retry


def qa_template(config: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "attempt_id": attempt["attempt_id"],
        "blinded_display_id": canonical_json_sha256(attempt["attempt_id"])[:12],
        "rater_id": None,
        "rated_at": None,
        "items": [
            {
                "id": item["id"],
                "question": item["question"],
                "response": None,
                "confidence": None,
                "note": None,
            }
            for item in config["human_qa"]["items"]
        ],
        "overall_note": None,
    }


def write_work_order(root: str | Path, config: Mapping[str, Any], work_order: Mapping[str, Any]) -> Path:
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    work_order_path = root_path / "work_order.json"
    work_order_path.write_bytes(canonical_json_bytes(work_order))
    for attempt in work_order["attempts"]:
        attempt_root = root_path / "attempts" / attempt["attempt_id"]
        attempt_root.mkdir(parents=True, exist_ok=True)
        (attempt_root / "planned_attempt.json").write_bytes(canonical_json_bytes(attempt))
        (attempt_root / "qa.json").write_bytes(canonical_json_bytes(qa_template(config, attempt)))
    return work_order_path


def patch_wan_sdpa_fallback(source_root: str | Path) -> dict[str, str]:
    """Bind Wan TI2V to its official FlashAttention-or-SDPA dispatcher."""

    source = Path(source_root).resolve()
    target = source / "wan" / "modules" / "model.py"
    original = target.read_bytes()
    matches = original.count(WAN_STRICT_ATTENTION_IMPORT)
    if matches != 1:
        raise RuntimeError(
            "frozen Wan source no longer has exactly one strict attention import"
        )
    original_sha256 = file_sha256(str(target))
    target.write_bytes(
        original.replace(
            WAN_STRICT_ATTENTION_IMPORT,
            WAN_FALLBACK_ATTENTION_IMPORT,
            1,
        )
    )
    return {
        "path": str(target.relative_to(source)),
        "reason": (
            "the frozen TI2V model imports the strict FlashAttention entrypoint; "
            "bind the official attention dispatcher so its SDPA fallback is reachable"
        ),
        "original_sha256": original_sha256,
        "patched_sha256": file_sha256(str(target)),
    }


def patch_wan_ti2v_import_surface(source_root: str | Path) -> dict[str, str]:
    """Avoid importing unrelated Wan pipelines and their optional dependencies."""

    source = Path(source_root).resolve()
    target = source / "wan" / "__init__.py"
    original = target.read_bytes()
    matches = original.count(WAN_EAGER_OPTIONAL_IMPORTS)
    if matches != 1:
        raise RuntimeError(
            "frozen Wan source no longer has exactly one eager optional import block"
        )
    original_sha256 = file_sha256(str(target))
    target.write_bytes(
        original.replace(
            WAN_EAGER_OPTIONAL_IMPORTS,
            WAN_TI2V_IMPORT,
            1,
        )
    )
    return {
        "path": str(target.relative_to(source)),
        "reason": (
            "the frozen package eagerly imports unrelated S2V and Animate pipelines; "
            "expose only the prespecified TI2V runner so unused optional dependencies "
            "cannot block TI2V startup"
        ),
        "original_sha256": original_sha256,
        "patched_sha256": file_sha256(str(target)),
    }


def build_model_command(
    config: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    source_root: str | Path,
    weights_root: str | Path,
    raw_output: str | Path,
    python_executable: str = "python",
    uv_executable: str = "uv",
) -> CommandSpec:
    """Return the exact pinned official runner command for one attempt."""

    source = Path(source_root).resolve()
    weights = Path(weights_root).resolve()
    output = Path(raw_output).resolve()
    family = attempt["family"]
    delivery = config["delivery"]
    model = config["models"][family]

    if family == "wan":
        argv = (
            python_executable,
            str(source / "generate.py"),
            "--task",
            "ti2v-5B",
            "--size",
            f"{delivery['width']}*{delivery['height']}",
            "--frame_num",
            str(delivery["num_frames"]),
            "--ckpt_dir",
            str(weights),
            "--base_seed",
            str(attempt["seed"]),
            "--sample_solver",
            model["sample_solver"],
            "--sample_steps",
            str(model["sample_steps"]),
            "--sample_shift",
            str(model["sample_shift"]),
            "--sample_guide_scale",
            str(model["sample_guide_scale"]),
            "--prompt",
            attempt["prompt"],
            "--save_file",
            str(output),
        )
        return CommandSpec(argv=argv, cwd=str(source))

    if family == "ltx":
        argv = (
            uv_executable,
            "run",
            "python",
            "-m",
            "ltx_pipelines.distilled",
            "--distilled-checkpoint-path",
            str(weights / model["checkpoint_filename"]),
            "--spatial-upsampler-path",
            str(weights / model["spatial_upsampler_filename"]),
            "--gemma-root",
            str(weights / "gemma"),
            "--prompt",
            attempt["prompt"],
            "--output-path",
            str(output),
            "--seed",
            str(attempt["seed"]),
            "--height",
            str(delivery["height"]),
            "--width",
            str(delivery["width"]),
            "--num-frames",
            str(delivery["num_frames"]),
            "--frame-rate",
            str(delivery["fps"]),
            "--offload",
            model["offload"],
        )
        return CommandSpec(argv=argv, cwd=str(source))

    raise ValueError(f"unknown family {family!r}")


def synthesize_tts(
    config: Mapping[str, Any],
    scene_id: str,
    *,
    voice_model: str | Path,
    output_wav: str | Path,
    python_executable: str = "python",
) -> None:
    scene = scene_by_id(config, scene_id)
    tts = config["tts"]
    command = [
        python_executable,
        "-m",
        "piper",
        "-m",
        str(Path(voice_model).resolve()),
        "-f",
        str(Path(output_wav).resolve()),
        "--volume",
        str(tts["volume"]),
        "--length-scale",
        str(tts["length_scale"]),
        "--",
        scene["utterance_de"],
    ]
    subprocess.run(command, check=True)


def ffprobe(path: str | Path, ffprobe_executable: str = "ffprobe") -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe_executable,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(Path(path).resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _stream(probe: Mapping[str, Any], codec_type: str) -> Mapping[str, Any] | None:
    return next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == codec_type), None)


def media_summary(path: str | Path, ffprobe_executable: str = "ffprobe") -> dict[str, Any]:
    source = Path(path).resolve()
    probe = ffprobe(source, ffprobe_executable)
    video = _stream(probe, "video")
    audio = _stream(probe, "audio")
    if video is None:
        raise ValueError(f"{source} has no video stream")
    rate_text = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    frame_rate = float(Fraction(rate_text))
    duration_text = video.get("duration") or probe.get("format", {}).get("duration")
    duration = float(duration_text) if duration_text is not None else None
    return {
        "path": source.name,
        "bytes": source.stat().st_size,
        "sha256": file_sha256(str(source)),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": frame_rate,
        "duration_s": duration,
        "frame_count": int(video["nb_frames"]) if video.get("nb_frames", "").isdigit() else None,
        "video_codec": video.get("codec_name"),
        "has_audio": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "audio_sample_rate_hz": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
    }


def validate_final_media(config: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    delivery = config["delivery"]
    errors = []
    if summary.get("width") != delivery["width"] or summary.get("height") != delivery["height"]:
        errors.append("resolution mismatch")
    if not math.isclose(float(summary.get("fps") or 0), delivery["fps"], abs_tol=0.01):
        errors.append("frame-rate mismatch")
    duration = summary.get("duration_s")
    if duration is None or abs(float(duration) - delivery["nominal_duration_s"]) > 1.0 / delivery["fps"] + 0.02:
        errors.append("duration mismatch")
    if summary.get("has_audio") is not True:
        errors.append("final clip has no audio")
    return errors


def mux_modular_audio(
    config: Mapping[str, Any],
    scene_id: str,
    *,
    raw_video: str | Path,
    tts_wav: str | Path,
    video_only: str | Path,
    final_video: str | Path,
    ffmpeg_executable: str = "ffmpeg",
) -> None:
    """Strip any native audio, then place the frozen TTS waveform at its onset."""

    scene = scene_by_id(config, scene_id)
    delivery = config["delivery"]
    raw = str(Path(raw_video).resolve())
    video_path = str(Path(video_only).resolve())
    wav = str(Path(tts_wav).resolve())
    final = str(Path(final_video).resolve())
    common = [ffmpeg_executable, "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run(
        [
            *common,
            "-i",
            raw,
            "-an",
            "-vf",
            f"fps={delivery['fps']},scale={delivery['width']}:{delivery['height']}:flags=lanczos",
            "-frames:v",
            str(delivery["num_frames"]),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            video_path,
        ],
        check=True,
    )
    delay_ms = round(float(scene["utterance_onset_s"]) * 1000)
    subprocess.run(
        [
            *common,
            "-i",
            video_path,
            "-i",
            wav,
            "-filter_complex",
            f"[1:a]adelay={delay_ms}:all=1,apad=whole_dur={delivery['nominal_duration_s']}[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-ar",
            str(delivery["audio_sample_rate_hz"]),
            "-t",
            f"{delivery['nominal_duration_s']:.9f}",
            "-movflags",
            "+faststart",
            final,
        ],
        check=True,
    )


def complete_attempt_record(
    config: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    started_at: str,
    finished_at: str,
    raw_summary: Mapping[str, Any],
    final_summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    errors = validate_final_media(config, final_summary)
    return {
        "schema_version": 1,
        "attempt": dict(attempt),
        "status": "media_valid" if not errors else "invalid_media",
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime": dict(runtime),
        "raw_media": dict(raw_summary),
        "final_media": dict(final_summary),
        "conformance_errors": errors,
        "human_qa_status": "pending",
    }


def render_gallery(
    config: Mapping[str, Any],
    work_order: Mapping[str, Any],
    run_root: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    root = Path(run_root).resolve()
    output = Path(output_path).resolve() if output_path else root / "gallery" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    by_scene: dict[str, list[Mapping[str, Any]]] = {}
    for attempt in work_order["attempts"]:
        by_scene.setdefault(attempt["scene_id"], []).append(attempt)

    cards = []
    for scene_id in config["profiles"][work_order["profile"]]["scene_ids"]:
        scene = scene_by_id(config, scene_id)
        attempts = by_scene.get(scene_id, [])
        media_cards = []
        for attempt in attempts:
            final_path = root / attempt["paths"]["final_video"]
            rel = os.path.relpath(final_path, output.parent)
            qa_path = root / attempt["paths"]["qa_template"]
            qa_rel = os.path.relpath(qa_path, output.parent)
            exists_note = "" if final_path.exists() else '<p class="missing">Media not present yet.</p>'
            media_cards.append(
                f"""
                <article class="attempt">
                  <h3>{html.escape(attempt['family'].upper())}</h3>
                  <video controls preload="metadata" src="{html.escape(rel)}"></video>
                  {exists_note}
                  <p>Blinded ID: <code>{canonical_json_sha256(attempt['attempt_id'])[:12]}</code></p>
                  <p><a href="{html.escape(qa_rel)}">QA record</a></p>
                </article>
                """
            )
        cards.append(
            f"""
            <section>
              <h2>{html.escape(scene['label'])}</h2>
              <p class="utterance" lang="de">{html.escape(scene['utterance_de'])}</p>
              <div class="attempts">{''.join(media_cards)}</div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic-video public pilot</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0 auto; max-width: 1180px; padding: 2rem; background: #111827; color: #f3f4f6; }}
    header, section {{ background: #1f2937; border: 1px solid #374151; border-radius: 14px; margin: 1rem 0; padding: 1.25rem; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .warning {{ color: #fbbf24; }}
    .attempts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }}
    .attempt {{ background: #111827; border-radius: 10px; padding: 1rem; }}
    video {{ width: 100%; aspect-ratio: 1280 / 704; background: #000; border-radius: 8px; }}
    .utterance {{ font-size: 1.1rem; }}
    .missing {{ color: #fca5a5; }}
    a {{ color: #93c5fd; }}
    code {{ color: #d1fae5; }}
  </style>
</head>
<body>
  <header>
    <h1>Synthetic-video public-only qualitative pilot</h1>
    <p>Run <code>{html.escape(str(work_order['run_id']))}</code> · profile <code>{html.escape(str(work_order['profile']))}</code></p>
    <p class="warning">Engineering preview only. These clips cannot select a generator or support a ChildLens or scientific claim.</p>
  </header>
  {''.join(cards)}
</body>
</html>
"""
    output.write_text(page)
    return output
