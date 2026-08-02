#!/usr/bin/env python3
"""Governed preparation for the frozen lean equal-duration pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from build_synthetic_video_phase4_assets import canonical, digest, translate_segments, write_json


class PilotError(RuntimeError):
    pass


def load_rows(root: Path, role: str) -> list[dict]:
    plan = json.loads((root / "restricted_proof_plan.json").read_text())
    acquisition = json.loads((root / f"restricted_{role}_acquisition.json").read_text())["items"]
    selected = plan["selected_training"] if role == "training" else plan["selected_validation"]
    rows = []
    for source in selected:
        key = digest({"participant_key": source["participant_key"], "media_key": source["media_key"]})
        item = acquisition.get(key)
        if not item or item.get("status") != "COMPLETE":
            continue
        rows.append({
            "asset_key": key,
            "child_key": source["participant_key"],
            "session_key": source["session_key"],
            "file": f"{role}_media/{item['file']}",
        })
    return rows


def transcribe_until(root: Path, public: Path, rows: list[dict], checkpoint_path: Path, accepted_target: float | None) -> dict:
    import whisper
    from transformers import MarianMTModel, MarianTokenizer

    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {
        "adapter_contract": "frozen_language_adapter_v1", "items": {}
    }
    asr = whisper.load_model(str(public / "models/whisper/small.pt"), device="cuda")
    translation_root = public / "models/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(translation_root, local_files_only=True)
    translator = MarianMTModel.from_pretrained(translation_root, local_files_only=True).to("cuda")
    accepted_seconds = sum(
        max(0.0, segment["end"] - segment["start"])
        for item in checkpoint["items"].values() for segment in item["segments"]
        if segment["status"] == "ACCEPT"
    )
    for row in rows:
        if accepted_target is not None and accepted_seconds >= accepted_target:
            break
        if row["asset_key"] in checkpoint["items"]:
            continue
        media = root / row["file"]
        result = asr.transcribe(str(media), language="de", task="transcribe", temperature=0,
                                beam_size=5, word_timestamps=True, condition_on_previous_text=False,
                                fp16=True, verbose=False)
        duration = len(whisper.load_audio(str(media))) / whisper.audio.SAMPLE_RATE
        segments = translate_segments(tokenizer, translator, result["segments"], result.get("language"), duration, 0.35)
        checkpoint["items"][row["asset_key"]] = {
            "segments": segments, "language": result.get("language"), "child_key": row["child_key"],
            "session_key": row["session_key"], "file": row["file"],
        }
        accepted_seconds += sum(max(0.0, s["end"] - s["start"]) for s in segments if s["status"] == "ACCEPT")
        write_json(checkpoint_path, checkpoint)
    del asr, translator
    torch.cuda.empty_cache()
    return checkpoint


def extract_frame(media: Path, target: Path, timestamp: float) -> bool:
    temporary = target.with_suffix(".tmp.jpg")
    run = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{timestamp:.6f}",
                          "-i", str(media), "-frames:v", "1", "-q:v", "2", "-y", str(temporary)],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if run.returncode or not temporary.is_file() or not temporary.stat().st_size:
        temporary.unlink(missing_ok=True)
        return False
    os.chmod(temporary, 0o600)
    temporary.replace(target)
    return True


def build_manifest(root: Path, checkpoint: dict, role: str, credited_limit: float | None) -> tuple[list[dict], float]:
    frame_root = root / "frames" / role
    frame_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    records, credited = [], 0.0
    for asset_key, item in checkpoint["items"].items():
        for index, segment in enumerate(item["segments"]):
            if segment["status"] != "ACCEPT" or (credited_limit is not None and credited >= credited_limit):
                continue
            available = max(0.0, segment["end"] - segment["start"])
            use = available if credited_limit is None else min(available, credited_limit - credited)
            if use <= 0:
                continue
            names = []
            for offset_index, fraction in enumerate((0.125, 0.375, 0.625, 0.875)):
                timestamp = segment["start"] + use * fraction
                name = f"{digest([asset_key,index,offset_index])}.jpg"
                target = frame_root / name
                if extract_frame(root / item["file"], target, timestamp):
                    names.append(name)
            if not names:
                continue
            records.append({"utterance": segment["en"], "frame_filenames": names,
                            "utterance_id": digest([asset_key, index])})
            credited += use
    return records, credited


def prepare(root: Path, public: Path) -> dict:
    train_rows, validation_rows = load_rows(root, "training"), load_rows(root, "validation")
    train = transcribe_until(root, public, train_rows, root / "restricted_training_asr.json", 3960.0)
    accepted = sum(max(0.0, s["end"] - s["start"]) for x in train["items"].values()
                   for s in x["segments"] if s["status"] == "ACCEPT")
    if accepted < 3960.0:
        raise PilotError(f"E_REAL_1H_YIELD:{accepted:.3f}")
    validation = transcribe_until(root, public, validation_rows, root / "restricted_validation_asr.json", None)
    train_manifest, credited = build_manifest(root, train, "training", 3600.0)
    validation_manifest, _ = build_manifest(root, validation, "validation", None)
    if abs(credited - 3600.0) > 1e-6:
        raise PilotError("E_EXACT_CREDIT")
    write_json(root / "restricted_train_1h_manifest.json", train_manifest)
    write_json(root / "restricted_validation_manifest.json", validation_manifest)
    text_path = root / "restricted_train_1h_text.txt"
    text_path.write_text("\n".join(row["utterance"] for row in train_manifest) + "\n")
    os.chmod(text_path, 0o600)
    record = {"status": "PASS", "credited_seconds": credited, "reserve_accepted_seconds": accepted - credited,
              "training_record_count": len(train_manifest), "validation_record_count": len(validation_manifest),
              "training_source_recordings_processed": len(train["items"]),
              "validation_source_recordings_processed": len(validation["items"])}
    record["commitment"] = hashlib.sha256(canonical(record)).hexdigest()
    write_json(root / "compact_prepare_result.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepare", nargs="?")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve(), args.public.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
