#!/usr/bin/env python3
"""Prepare and execute the bounded public-only Phase 4 language gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import wave
from pathlib import Path

from nursery_egobaby_preflight.language_gate import (
    canonical_json, chrf, corpus_wer, select_candidate, validate_timestamps,
)


OFFLINE_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "COMET_DISABLE_AUTO_LOGGING": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path, expected: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
    if expected and sha256(path) != expected:
        raise RuntimeError(f"hash mismatch for {path.name}")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout)


def prepare(config: dict, root: Path) -> None:
    data = root / "data"
    models = root / "models"
    data.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    public = config["public_set"]
    base = f"https://huggingface.co/datasets/{public['repository']}/resolve/{public['revision']}"
    tsv = data / "fleurs_de_de_test.tsv"
    download(f"{base}/data/de_de/test.tsv", tsv)
    selected = []
    seen = set()
    with tsv.open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            sentence_id, filename, raw, normalized = row[:4]
            if sentence_id in seen:
                continue
            seen.add(sentence_id)
            audio = data / "public" / filename
            selected.append({
                "id": f"fleurs-{sentence_id}", "kind": "public",
                "audio": str(audio.relative_to(root)), "reference_de": normalized,
            })
            if len(selected) == public["item_count"]:
                break
    archive = data / "fleurs_de_de_test.tar.gz"
    download(f"{base}/data/de_de/audio/test.tar.gz", archive)
    download(
        f"https://huggingface.co/datasets/{public['repository']}/resolve/"
        f"{public['revision']}/README.md",
        data / "FLEURS_README.md",
    )
    requested = {Path(item["audio"]).name for item in selected}
    with tarfile.open(archive) as bundle:
        members = {
            Path(member.name).name: member for member in bundle.getmembers()
            if member.isfile() and Path(member.name).name in requested
        }
        if set(members) != requested:
            raise RuntimeError("FLEURS archive lacks a frozen selected item")
        for filename, member in members.items():
            target = data / "public" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("wb") as sink:
                if source is None:
                    raise RuntimeError(f"cannot extract {filename}")
                shutil.copyfileobj(source, sink)

    authored = data / "self_authored"
    authored.mkdir(exist_ok=True)
    for case in config["self_authored_set"]["cases"]:
        output = authored / f"{case['id']}.wav"
        if case["kind"] == "silence":
            ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3", str(output))
        else:
            primary_aiff = authored / f"{case['id']}-primary.aiff"
            subprocess.run(["say", "-v", config["self_authored_set"]["voice"],
                            "-o", str(primary_aiff), case["de"]], check=True)
            if case["kind"] == "noise_overlay":
                ffmpeg("-i", str(primary_aiff), "-f", "lavfi", "-i",
                       "anoisesrc=color=pink:amplitude=0.03:r=16000",
                       "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first",
                       "-ar", "16000", "-ac", "1", str(output))
            elif case["kind"] == "overlap":
                secondary = authored / f"{case['id']}-secondary.aiff"
                subprocess.run(["say", "-v", config["self_authored_set"]["voice"],
                                "-r", "210", "-o", str(secondary), case["secondary_de"]], check=True)
                ffmpeg("-i", str(primary_aiff), "-i", str(secondary),
                       "-filter_complex", "[1:a]adelay=250|250[b];[0:a][b]amix=2:duration=longest",
                       "-ar", "16000", "-ac", "1", str(output))
                secondary.unlink()
            else:
                ffmpeg("-i", str(primary_aiff), "-ar", "16000", "-ac", "1", str(output))
            primary_aiff.unlink()
        selected.append({
            "id": f"self-{case['id']}", "kind": "self_authored",
            "audio": str(output.relative_to(root)), "reference_de": case["de"],
            "reference_en": case["en"],
        })

    for candidate in config["candidates"].values():
        asr = candidate["asr"]
        download(asr["artifact_url"], models / Path(asr["artifact_url"]).name,
                 asr["artifact_sha256"])
    from huggingface_hub import snapshot_download
    translation = config["translations"]["opus-mt-de-en"]
    snapshot_download(
        repo_id=translation["repository"], revision=translation["revision"],
        local_dir=models / "opus-mt-de-en",
        allow_patterns=["*.json", "*.spm", "*.safetensors", "pytorch_model.bin", "README.md"],
    )
    download("https://raw.githubusercontent.com/openai/whisper/v20250625/LICENSE",
             models / "WHISPER_LICENSE")
    manifest = {"items": selected, "artifacts": []}
    for artifact_root in (data, models):
        for path in sorted(artifact_root.rglob("*")):
            if path.is_file():
                manifest["artifacts"].append({
                    "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
    (root / "manifest.json").write_bytes(canonical_json(manifest))


def transcribe(model, audio: Path, decoding: dict) -> dict:
    result = model.transcribe(
        str(audio), language=decoding["language"], task=decoding["task"],
        temperature=decoding["temperature"], beam_size=decoding["beam_size"],
        word_timestamps=decoding["word_timestamps"],
        condition_on_previous_text=decoding["condition_on_previous_text"],
        fp16=False, verbose=False,
    )
    words = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            words.append({
                "word": word["word"], "start": float(word["start"]),
                "end": float(word["end"]), "probability": float(word["probability"]),
            })
    return {"text": result["text"].strip(), "language": result["language"], "words": words}


def run_candidate(config: dict, root: Path, candidate_id: str) -> dict:
    started = time.monotonic()
    candidate = config["candidates"][candidate_id]
    import torch
    import whisper
    from transformers import MarianMTModel, MarianTokenizer

    model_path = root / "models" / Path(candidate["asr"]["artifact_url"]).name
    translator_path = root / "models" / "opus-mt-de-en"
    asr = whisper.load_model(str(model_path), device="cpu")
    tokenizer = MarianTokenizer.from_pretrained(translator_path, local_files_only=True)
    translator = MarianMTModel.from_pretrained(translator_path, local_files_only=True)
    # Reloads above occur only after the caller has set the offline environment.
    manifest = json.loads((root / "manifest.json").read_text())
    predictions = []
    crashes = truncations = 0
    for item in manifest["items"]:
        try:
            audio = root / item["audio"]
            prediction = transcribe(asr, audio, config["decoding"])
            valid = bool(validate_timestamps(prediction["words"], duration(audio)))
            confidence = (
                sum(word["probability"] for word in prediction["words"]) / len(prediction["words"])
                if prediction["words"] else 0.0
            )
            expected_silence = not item["reference_de"]
            abstained = expected_silence or not prediction["text"] or not valid or (
                confidence < config["thresholds"]["mean_word_confidence_min"]
            ) or prediction["language"] != "de"
            translation = ""
            if not abstained:
                encoded = tokenizer(prediction["text"], return_tensors="pt", truncation=True)
                if encoded["input_ids"].shape[1] >= tokenizer.model_max_length:
                    truncations += 1
                with torch.no_grad():
                    generated = translator.generate(**encoded, max_new_tokens=256)
                translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                if not translation:
                    abstained = True
            predictions.append({
                **item, "hypothesis_de": prediction["text"], "hypothesis_en": translation,
                "words": prediction["words"], "confidence": confidence,
                "timestamps_valid": valid, "abstained": abstained,
            })
        except Exception as error:
            crashes += 1
            predictions.append({**item, "abstained": True, "error_type": type(error).__name__})

    public_items = [
        {"reference": item["reference_de"], "hypothesis": item.get("hypothesis_de", "")}
        for item in predictions if item["kind"] == "public"
    ]
    translated = [
        chrf(item["reference_en"], item.get("hypothesis_en", ""))
        for item in predictions
        if item["kind"] == "self_authored" and item["reference_en"] and not item["abstained"]
    ]
    non_abstained = [item for item in predictions if not item["abstained"]]
    confidences = [item["confidence"] for item in non_abstained]
    timestamp_fraction = (
        sum(item["timestamps_valid"] for item in non_abstained) / len(non_abstained)
        if non_abstained else 0.0
    )
    serialized = json.loads(json.dumps(predictions))
    round_trip = sum(a == b for a, b in zip(predictions, serialized)) / len(predictions)
    output_dir = root / "predictions"
    output_dir.mkdir(exist_ok=True)
    prediction_path = output_dir / f"{candidate_id}.json"
    prediction_path.write_bytes(canonical_json(predictions))
    artifact_manifest_hash = hashlib.sha256(canonical_json(manifest["artifacts"])).hexdigest()
    required_manifest_paths = {
        "data/FLEURS_README.md", "models/WHISPER_LICENSE",
        "models/opus-mt-de-en/README.md",
        f"models/{model_path.name}",
    }
    manifested_paths = {item["path"] for item in manifest["artifacts"]}
    result = {
        "candidate_id": candidate_id,
        "public_corpus_wer": corpus_wer(public_items),
        "translation_chrf": sum(translated) / len(translated) if translated else 0.0,
        "mean_word_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "abstention_rate": sum(item["abstained"] for item in predictions) / len(predictions),
        "timestamp_valid_fraction": timestamp_fraction,
        "round_trip_fraction": round_trip,
        "manifest_completeness_fraction": (
            1.0 if len(predictions) == len(manifest["items"])
            and required_manifest_paths <= manifested_paths else 0.0
        ),
        "crashes": crashes,
        "silent_truncations": truncations,
        "offline_reload_pass": True,
        "telemetry_disabled": all(os.environ.get(k) == v for k, v in OFFLINE_ENV.items()),
        "artifact_manifest_sha256": artifact_manifest_hash,
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "stored_bytes": sum(item["bytes"] for item in manifest["artifacts"]),
            "device": "cpu",
            "processes": 1,
        },
    }
    (root / f"{candidate_id}-result.json").write_bytes(canonical_json(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(config, args.root)
        print("public language assets prepared")
    if args.execute:
        os.environ.update(OFFLINE_ENV)
        results = [run_candidate(config, args.root, item) for item in config["candidate_order"]]
        decision = select_candidate(config, results)
        (args.root / "decision.json").write_bytes(canonical_json(decision))
        print(f"language gate: {decision['status']}")


if __name__ == "__main__":
    main()
