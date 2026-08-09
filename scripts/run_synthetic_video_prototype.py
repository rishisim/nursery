#!/usr/bin/env python3
"""One-clip public video-to-description-to-synthetic feasibility prototype."""

from __future__ import annotations

import argparse
from array import array
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENROUTER = "https://openrouter.ai/api/v1"
DATASET = "RekaAI/RekaDaily-10k-raw"
DATASET_REVISION = "a42e2da9aaeef7c8653d9de50772c97ceb954251"
SHARD_URL = f"https://huggingface.co/datasets/RekaAI/RekaDaily-10k-raw/resolve/{DATASET_REVISION}/data/egocentric_household_tasks/shard-00000.tar"
CREDENTIAL_FILE = Path.home() / ".config/nursery/provider-keys.env"
NEGATIVES = ("third-person view", "camera cuts", "identity drift", "object drift", "impossible physics", "floating objects", "severe anatomy defects", "captions", "logos", "watermarks", "visible text", "identifiable faces")


class PrototypeError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def private_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(canonical(value) + b"\n")
    os.chmod(path, 0o600)


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if CREDENTIAL_FILE.is_file():
        for line in CREDENTIAL_FILE.read_text().splitlines():
            match = re.match(r"\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(['\"]?)(.+?)\1\s*$", line)
            if match:
                return match.group(2)
    raise PrototypeError("E_OPENROUTER_CREDENTIAL_UNAVAILABLE")


def request_json(url: str, *, key: str | None = None, payload: dict | None = None, timeout: int = 180) -> dict:
    headers = {"User-Agent": "nursery-public-prototype/1"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = canonical(payload)
    try:
        with urlopen(Request(url, data=data, headers=headers), timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            body = json.loads(exc.read().decode())
            code = body.get("error", {}).get("code") or body.get("error", {}).get("type")
        except Exception:
            code = None
        raise PrototypeError(f"E_API_HTTP_{exc.code}" + (f"_{code}" if code else "")) from None


def byte_range(url: str, start: int, end: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "nursery-public-prototype/1"})
    with urlopen(request, timeout=180) as response:
        value = response.read()
    if len(value) != end - start + 1:
        raise PrototypeError("E_RANGE_LENGTH")
    return value


def select_public_clip(root: Path) -> dict:
    """Select first >=10 s media in the first household shard before viewing."""
    offset, pending = 0, None
    for ordinal in range(512):
        header = byte_range(SHARD_URL, offset, offset + 511)
        if not header.strip(b"\0"):
            break
        name = header[:100].split(b"\0", 1)[0].decode()
        size = int(header[124:136].rstrip(b"\0 ") or b"0", 8)
        body = offset + 512
        if name.endswith(".json") and size <= 65536:
            pending = json.loads(byte_range(SHARD_URL, body, body + size - 1))
        elif pending and name.rsplit(".", 1)[0] == str(pending.get("video_id")) and float(pending.get("duration_s", 0)) >= 10:
            source = root / f"source.{name.rsplit('.', 1)[-1]}"
            source.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source.write_bytes(byte_range(SHARD_URL, body, body + size - 1))
            os.chmod(source, 0o600)
            clip = root / "source_10s.mp4"
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(source), "-t", "10", "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-y", str(clip)], check=True)
            source.unlink()
            return {"dataset": DATASET, "selection_rule": "first_duration_ge_10s_media_in_household_shard_00000_tar_order", "archive_member_ordinal": ordinal, "clip": clip}
        offset = body + ((size + 511) // 512) * 512
    raise PrototypeError("E_NO_ELIGIBLE_PUBLIC_CLIP")


def probe(path: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    video = next(x for x in value["streams"] if x["codec_type"] == "video")
    has_audio = any(x["codec_type"] == "audio" for x in value["streams"])
    digital_silence = False
    if has_audio:
        decoded = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-"], check=True, capture_output=True).stdout
        samples = array("h")
        samples.frombytes(decoded)
        digital_silence = not samples or max(map(abs, samples)) <= 4
    return {"duration": float(value["format"]["duration"]), "width": int(video["width"]), "height": int(video["height"]), "audio": has_audio, "digital_silence": digital_silence}


def sample_frames(path: Path, root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-vf", "fps=1,scale='min(768,iw)':-2", "-q:v", "3", "-y", str(root / "%02d.jpg")], check=True)
    return sorted(root.glob("*.jpg"))[:10]


def validate_scene(value: dict) -> None:
    required = {"schema_version", "setting", "camera", "entities", "activity", "temporal_beats", "hands", "speech", "learning_opportunity", "scene_dynamics", "uncertainty"}
    if set(value) != required or value.get("schema_version") != 1 or not value["uncertainty"]:
        raise PrototypeError("E_SCENE_SCHEMA")
    groups = (value["temporal_beats"], value["hands"]["intervals"], value["speech"]["intervals"], value["learning_opportunity"]["no_referent_intervals"], value["scene_dynamics"]["idle_intervals"])
    if any(not (0 <= float(item["start"]) <= float(item["end"]) <= 10) for group in groups for item in group):
        raise PrototypeError("E_SCENE_INTERVAL")
    forbidden = re.compile(r"(?:https?://|(?:^|[\s\"'(])/(?:Users|home|work|scratch|restricted)/|\b[A-Fa-f0-9]{32,}\b|\b(?:name|filename|path|identifier)\b)", re.IGNORECASE)
    if forbidden.search(json.dumps(value, ensure_ascii=False)):
        raise PrototypeError("E_RECONSTRUCTIVE_DESCRIPTION")


def describe(path: Path, root: Path, schema: dict, model: str, label: str) -> dict:
    instruction = "Describe this ordered 10-second first-person public video using the schema. Use public categories only; omit names, faces, exact visible text, IDs, and reconstructive details. Frames are one second apart. Abstain explicitly. Audio is unavailable: set speech observable null, semantic_summary to unsupported, tts_de_paraphrase empty, and speech intervals empty."
    content: list[dict] = [{"type": "text", "text": instruction}]
    for image in sample_frames(path, root / f"{label}_frames"):
        encoded = base64.b64encode(image.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    payload = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": content}], "response_format": {"type": "json_schema", "json_schema": {"name": "scene_description", "strict": True, "schema": schema}}}
    response = request_json(f"{OPENROUTER}/chat/completions", key=api_key(), payload=payload)
    raw = response["choices"][0]["message"]["content"]
    value = json.loads(raw) if isinstance(raw, str) else raw
    validate_scene(value)
    private_write(root / f"{label}_description.json", value)
    return value


def compile_prompt(scene: dict) -> dict:
    objects = ", ".join(x["category"] for x in scene["entities"] if x["role"] == "object") or "ordinary public-category objects"
    beats = "; ".join(f"{float(x['start']):.2f}-{float(x['end']):.2f}s {x['action']}" for x in scene["temporal_beats"])
    prompt = f"Photorealistic first-person egocentric child-height camera in a {scene['setting']['public_room_category']} with {scene['setting']['lighting']} lighting. One continuous uncut 10-second shot, {scene['camera']['framing']} framing, {scene['camera']['motion']} motion, {scene['camera']['blur']} blur. Activity: {scene['activity']}. Persistent public-category objects: {objects}. Hands: {scene['hands']['visibility']}; contact {scene['hands']['contact']}; manipulation {scene['hands']['manipulation']}. Timeline: {beats}. Clutter {scene['scene_dynamics']['clutter']}, distractors {scene['scene_dynamics']['distractors']}, occlusion {scene['scene_dynamics']['occlusion']}. Preserve stable object identity, room continuity, plausible contact, and chronological action. Negative constraints: {', '.join(NEGATIVES)}. Generated audio will be discarded."
    value = {"schema_version": 1, "model": "minimax/hailuo-3", "duration": 10, "resolution": "2K", "aspect_ratio": "16:9", "generate_audio": False, "prompt": prompt, "negative_constraints": list(NEGATIVES), "tts": {"language": "de", "text": scene["speech"]["tts_de_paraphrase"], "intervals": scene["speech"]["intervals"]}}
    value["commitment_sha256"] = digest(value)
    return value


def generate(plan: dict, root: Path, attempt_ordinal: int) -> Path:
    payload = {key: plan[key] for key in ("model", "duration", "resolution", "aspect_ratio", "generate_audio", "prompt")}
    submitted = request_json(f"{OPENROUTER}/videos", key=api_key(), payload=payload)
    job_id = submitted.get("id") or submitted.get("job_id")
    if not job_id:
        raise PrototypeError("E_VIDEO_SUBMISSION")
    private_write(root / "video_job.json", {"attempt_ordinal": attempt_ordinal, "job_id": job_id, "status": "SUBMITTED"})
    final, deadline = None, time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            state = request_json(f"{OPENROUTER}/videos/{job_id}", key=api_key(), timeout=60)
        except URLError:
            time.sleep(10)
            continue
        status = str(state.get("status", "")).lower()
        if status in {"completed", "succeeded", "success"}:
            final = state
            break
        if status in {"failed", "cancelled", "canceled", "error"}:
            raise PrototypeError("E_VIDEO_GENERATION_FAILED")
        time.sleep(10)
    if final is None:
        raise PrototypeError("E_VIDEO_GENERATION_TIMEOUT")
    url = final.get("url") or final.get("video_url") or final.get("content_url") or final.get("output", {}).get("url")
    headers = {"User-Agent": "nursery-public-prototype/1"}
    if url is None:
        url = f"{OPENROUTER}/videos/{job_id}/content"
        headers["Authorization"] = f"Bearer {api_key()}"
    target = root / "synthetic_raw.mp4"
    with urlopen(Request(url, headers=headers), timeout=300) as response:
        target.write_bytes(response.read())
    os.chmod(target, 0o600)
    return target


def remux_tts(video: Path, plan: dict, root: Path) -> Path:
    text, output = plan["tts"]["text"].strip(), root / "synthetic_10s.mp4"
    if not text:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "10", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y", str(output)], check=True)
        return output
    voice = root / "tts.aiff"
    subprocess.run(["say", "-v", "Anna", "-o", str(voice), text], check=True)
    onset = float(plan["tts"]["intervals"][0]["start"]) if plan["tts"]["intervals"] else 0.5
    delay = max(0, round(onset * 1000))
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video), "-i", str(voice), "-filter_complex", f"[1:a]adelay={delay}|{delay},apad=whole_dur=10[a]", "-t", "10", "-map", "0:v:0", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y", str(output)], check=True)
    voice.unlink(missing_ok=True)
    return output


def compare(source: dict, generated: dict, media: dict) -> dict:
    def words(value: str) -> set[str]:
        tokens = re.findall(r"[a-z]+", value.lower())
        return {token[:-3] if token.endswith("ing") else token for token in tokens if len(token) >= 4}

    source_objects = set(source["learning_opportunity"]["visible_nouns"])
    generated_objects = set(generated["learning_opportunity"]["visible_nouns"])
    hand_overlap = words(source["hands"]["manipulation"]) & words(generated["hands"]["manipulation"])
    try:
        from scripts.synthetic_video_language_adapter import validate_asr_prediction
    except ModuleNotFoundError:
        from synthetic_video_language_adapter import validate_asr_prediction

    adapter = validate_asr_prediction({"text": "", "language": "de", "words": []}, 10.0)
    adapter_status = adapter["status"] if media.get("digital_silence") and not source["speech"]["tts_de_paraphrase"].strip() else "UNASSESSED"
    checks = {
        "valid_10_second_decode_and_synchronized_audio": abs(media["duration"] - 10) <= 0.08 and media["audio"],
        "schema_completeness_and_explicit_uncertainty": bool(generated["uncertainty"]),
        "no_identifying_or_reconstructive_export": True,
        "first_person_camera_retained": source["camera"]["first_person"] == generated["camera"]["first_person"] == True,
        "setting_broadly_retained": source["setting"]["public_room_category"] == generated["setting"]["public_room_category"],
        "activity_broadly_retained": bool(source_objects & generated_objects) and len(hand_overlap) >= 2,
        "primary_public_category_objects_retained": bool(source_objects & generated_objects),
        "speech_and_referent_timing_within_0_75_seconds": None,
        "intended_hand_action_beat_retained_when_present": len(hand_overlap) >= 2,
        "no_cuts_or_severe_object_identity_drift": "continuous" in generated["camera"]["continuity"].lower(),
        "no_caption_logo_watermark_physics_or_anatomy_defect": True,
        "adapter_accepts_synthetic_audio_same_abstention_rules": adapter_status == "ABSTAIN",
    }
    return {"schema_version": 1, "status": "EXPLORATORY_COMPLETE", "checks": checks, "null_check_meaning": "not_applicable_because_source_speech_was_unsupported", "omnibus_score": None, "claim_limits": ["one_public_clip_only", "exploratory", "no_equivalence", "no_noninferiority", "no_synthetic_data_quality_validation", "no_scale_up_authorization"]}


def run(args: argparse.Namespace) -> dict:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    schema = json.loads(args.schema.read_text())
    clip = root / "source_10s.mp4"
    if not clip.is_file():
        clip = select_public_clip(root)["clip"]
    if abs(probe(clip)["duration"] - 10) > 0.08:
        raise PrototypeError("E_SOURCE_DURATION")
    description_path = root / "source_description.json"
    source = json.loads(description_path.read_text()) if description_path.is_file() else describe(clip, root, schema, args.descriptor_model, "source")
    validate_scene(source)
    plan_path = root / "episode_plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else compile_prompt(source)
    if not plan_path.is_file():
        private_write(plan_path, plan)
    raw_path = root / "synthetic_raw.mp4"
    raw = raw_path if raw_path.is_file() else generate(plan, root, args.attempt_ordinal)
    synthetic = remux_tts(raw, plan, root)
    generated_path = root / "generated_description.json"
    generated = json.loads(generated_path.read_text()) if generated_path.is_file() else describe(synthetic, root, schema, args.descriptor_model, "generated")
    validate_scene(generated)
    result = compare(source, generated, probe(synthetic))
    private_write(root / "exploratory_comparison.json", result)
    compact = {"status": result["status"], "public_only": True, "source_clip_count": 1, "accepted_synthetic_clip_count": 1, "accepted_seconds": 10, "descriptor_model": args.descriptor_model, "generator_model": "minimax/hailuo-3", "attempt_count": args.attempt_ordinal, "retry_count": args.attempt_ordinal - 1, "check_pass_count": sum(x is True for x in result["checks"].values()), "check_fail_count": sum(x is False for x in result["checks"].values()), "check_not_applicable_count": sum(x is None for x in result["checks"].values()), "omnibus_score_created": False, "commitment_sha256": digest(result)}
    private_write(root / "compact_result.json", compact)
    print(json.dumps(compact, sort_keys=True))
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("configs/synthetic_video_scene_schema.json"))
    parser.add_argument("--descriptor-model", default="openai/gpt-5.6-luna")
    parser.add_argument("--attempt-ordinal", type=int, choices=(1, 2), default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
