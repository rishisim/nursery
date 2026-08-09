#!/usr/bin/env python3
"""Run one pinned Gemma 4 MLX checkpoint on self-generated public fixtures.

This script has no ChildLens/quarantine discovery and accepts no arbitrary
paths.  It is intended to be launched under macOS ``sandbox-exec`` with all
network access denied.  Raw model output remains in the public bakeoff cache;
the repository receives only an aggregate, synthetic-only receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

import mlx.core as mx
from jsonschema import Draft202012Validator
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image, ImageDraw


VERSION = "childlens-gemma4-public-synthetic-runtime-v1.3.1"
CACHE_ROOT = Path.home() / "Library/Application Support/ChildLens Public Model Bakeoff/v1.3.1"
MODELS = {
    "e4b": {
        "name": "Gemma 4 E4B IT 4-bit MLX",
        "directory": "gemma-4-e4b-it-4bit",
        "conversion_revision": "475b9088d29754a3379866cf5aeb6b41acd313c2",
        "upstream_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    },
    "12b": {
        "name": "Gemma 4 12B Unified IT 4-bit MLX",
        "directory": "gemma-4-12B-it-4bit",
        "conversion_revision": "73bcf09092aa277861d5a191b989b666f7f32e8f",
        "upstream_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    },
}
REFERENCE = "Der rote Ball liegt auf dem Tisch."
SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["language", "source_transcript", "visual_object", "spoken_object_visible", "source_role"],
    "properties": {
        "language": {"type": "string", "enum": ["de", "other", "und"]},
        "source_transcript": {"type": "string", "minLength": 1, "maxLength": 200},
        "visual_object": {"type": "string", "enum": ["red_ball", "other", "uncertain"]},
        "spoken_object_visible": {"type": ["boolean", "null"]},
        "source_role": {"type": "string", "enum": ["NON_CHILD", "CHILD", "UNCERTAIN"]},
    },
}


class BakeoffError(RuntimeError):
    pass


def _network_denial_sentinel() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            return False
    except OSError:
        return True


def _model_manifest_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts and not path.is_symlink()
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
        total += size
    if not files or total <= 0:
        raise BakeoffError("E_MODEL_MANIFEST")
    return digest.hexdigest(), total


def _create_fixtures(root: Path) -> tuple[list[str], str]:
    frames: list[str] = []
    for index in range(5):
        image = Image.new("RGB", (384, 256), (240, 240, 230))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 170, 364, 230), fill=(130, 85, 45))
        center = 90 + index * 45
        draw.ellipse((center - 32, 128, center + 32, 192), fill=(220, 25, 25))
        path = root / f"frame-{index}.png"
        image.save(path)
        frames.append(str(path))
    aiff = root / "synthetic-german.aiff"
    wav = root / "synthetic-german.wav"
    spoken = subprocess.run(
        ["/usr/bin/say", "-v", "Anna", REFERENCE, "-o", str(aiff)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False, timeout=60,
    )
    converted = subprocess.run(
        ["/opt/homebrew/bin/ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False, timeout=60,
    )
    if spoken.returncode or converted.returncode or not wav.is_file():
        raise BakeoffError("E_SYNTHETIC_FIXTURE")
    return frames, str(wav)


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    first = stripped.find("{")
    last = stripped.rfind("}")
    if 0 <= first < last:
        candidates.append(stripped[first:last + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise BakeoffError("E_OUTPUT_NOT_JSON")


def _normalize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\wäöüß]+", " ", normalized, flags=re.UNICODE).strip()
    return normalized.split()


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, lvalue in enumerate(left, start=1):
        current = [i]
        for j, rvalue in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lvalue != rvalue)))
        previous = current
    return previous[-1]


def _wer(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    return _edit_distance(ref, hyp) / max(1, len(ref))


def _cer(reference: str, hypothesis: str) -> float:
    ref = list(" ".join(_normalize(reference)))
    hyp = list(" ".join(_normalize(hypothesis)))
    return _edit_distance(ref, hyp) / max(1, len(ref))


def execute(model_key: str) -> dict[str, Any]:
    if model_key not in MODELS:
        raise BakeoffError("E_MODEL_KEY")
    if not _network_denial_sentinel():
        raise BakeoffError("E_NETWORK_NOT_DENIED")
    record = MODELS[model_key]
    model_path = CACHE_ROOT / str(record["directory"])
    if not model_path.is_dir():
        raise BakeoffError("E_MODEL_MISSING")
    manifest, bytes_total = _model_manifest_digest(model_path)
    with tempfile.TemporaryDirectory(prefix="gemma4-public-synthetic-") as temporary:
        frames, audio = _create_fixtures(Path(temporary))
        prompt_text = (
            "Analyze all five synthetic frames together with the synthetic audio. "
            "Return exactly one JSON object and no markdown. The object must have exactly these keys: "
            "language (de|other|und), source_transcript (verbatim source language), "
            "visual_object (red_ball|other|uncertain), spoken_object_visible (true|false|null), "
            "source_role (NON_CHILD|CHILD|UNCERTAIN)."
        )
        mx.reset_peak_memory()
        started = time.perf_counter()
        model, processor = load(str(model_path), lazy=False)
        config = load_config(str(model_path))
        prompt = apply_chat_template(
            processor, config, prompt_text, num_images=len(frames), num_audios=1,
            chat_template_kwargs={"enable_thinking": False},
        )
        result = generate(
            model=model, processor=processor, prompt=prompt,
            image=frames, audio=[audio], max_tokens=220, temperature=0.0, verbose=False,
        )
        wall = time.perf_counter() - started
        peak = int(mx.get_peak_memory())
        raw = str(result.text)
        raw_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        parsed: dict[str, Any] | None = None
        schema_valid = False
        try:
            parsed = _extract_json(raw)
            schema_valid = not list(Draft202012Validator(SCHEMA).iter_errors(parsed))
        except BakeoffError:
            pass
        transcript = str(parsed.get("source_transcript", "")) if parsed else ""
        metrics = {
            "language_correct": bool(parsed and parsed.get("language") == "de"),
            "transcript_wer": round(_wer(REFERENCE, transcript), 4) if transcript else 1.0,
            "transcript_cer": round(_cer(REFERENCE, transcript), 4) if transcript else 1.0,
            "visual_object_correct": bool(parsed and parsed.get("visual_object") == "red_ball"),
            "joint_relation_correct": bool(parsed and parsed.get("spoken_object_visible") is True),
            "role_correct_for_synthetic_adult_voice": bool(parsed and parsed.get("source_role") == "NON_CHILD"),
        }
    return {
        "schema_version": VERSION,
        "model_key": model_key,
        "model_name": record["name"],
        "conversion_revision": record["conversion_revision"],
        "upstream_revision": record["upstream_revision"],
        "model_artifact_manifest_sha256": manifest,
        "model_artifact_bytes": bytes_total,
        "scope": "PUBLIC_SELF_GENERATED_SYNTHETIC_ONLY",
        "network_denial_sentinel_passed": True,
        "hosted_or_cloud_inference_used": False,
        "childlens_or_quarantine_accessed": False,
        "joint_audio_and_five_frames_executed": True,
        "strict_schema_requested_by_frozen_prompt": True,
        "strict_schema_output_valid": schema_valid,
        "raw_output_sha256": raw_digest,
        "synthetic_accuracy": metrics,
        "peak_mlx_unified_memory_gib": round(peak / (1024**3), 3),
        "wall_time_seconds": round(wall, 3),
        "semantic_result_is_human_evidence": False,
        "restricted_inference_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.parent.resolve() != (CACHE_ROOT / "results").resolve():
        raise BakeoffError("E_OUTPUT_DIRECTORY")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    value = execute(args.model)
    pending = output.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(pending, 0o600)
    os.replace(pending, output)
    print(json.dumps({"status": "COMPLETE", "model_key": args.model, "schema_valid": value["strict_schema_output_valid"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BakeoffError as exc:
        print(json.dumps({"status": "BLOCKED", "code": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
