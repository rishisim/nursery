#!/usr/bin/env python3
"""Network-denied Qwen3-ASR/aligner smoke on self-generated German TTS."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import unicodedata
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification, AutoProcessor


VERSION = "childlens-qwen3-asr-public-synthetic-smoke-v1.3.1"
CACHE_ROOT = Path.home() / "Library/Application Support/ChildLens Public Model Bakeoff/v1.3.1"
ASR = CACHE_ROOT / "qwen3-asr-1.7b-hf"
ALIGNER = CACHE_ROOT / "qwen3-forced-aligner-0.6b-hf"
OUTPUT = CACHE_ROOT / "results/qwen3-asr-aligner.json"
REFERENCE = "Der rote Ball liegt auf dem Tisch."
REVISIONS = {
    "asr": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    "aligner": "c07281df297b9905d24a508279258cccf987a064",
}


def _network_denied() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            return False
    except OSError:
        return True


def _manifest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts and not path.is_symlink())
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative}\0{size}\0{sha}\n".encode())
        total += size
    if not files:
        raise RuntimeError("E_MODEL_MANIFEST")
    return digest.hexdigest(), total


def _normalize(value: str) -> list[str]:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\wäöüß]+", " ", value, flags=re.UNICODE).strip().split()


def _distance(left: list[str], right: list[str]) -> int:
    prior = list(range(len(right) + 1))
    for i, lvalue in enumerate(left, 1):
        current = [i]
        for j, rvalue in enumerate(right, 1):
            current.append(min(current[-1] + 1, prior[j] + 1, prior[j - 1] + (lvalue != rvalue)))
        prior = current
    return prior[-1]


def _error_rates(hypothesis: str) -> tuple[float, float]:
    ref_words = _normalize(REFERENCE)
    hyp_words = _normalize(hypothesis)
    ref_chars = list(" ".join(ref_words))
    hyp_chars = list(" ".join(hyp_words))
    return (
        round(_distance(ref_words, hyp_words) / max(1, len(ref_words)), 4),
        round(_distance(ref_chars, hyp_chars) / max(1, len(ref_chars)), 4),
    )


class PeakRSS:
    def __init__(self) -> None:
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()


def _audio(directory: Path) -> Path:
    aiff = directory / "public-synthetic.aiff"
    wav = directory / "public-synthetic.wav"
    one = subprocess.run(["/usr/bin/say", "-v", "Anna", REFERENCE, "-o", str(aiff)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60)
    two = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(wav)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=60)
    if one.returncode or two.returncode or not wav.is_file():
        raise RuntimeError("E_SYNTHETIC_AUDIO")
    return wav


def execute() -> dict[str, object]:
    if not _network_denied():
        raise RuntimeError("E_NETWORK_NOT_DENIED")
    if not torch.backends.mps.is_available():
        raise RuntimeError("E_MPS_UNAVAILABLE")
    asr_manifest, asr_bytes = _manifest(ASR)
    aligner_manifest, aligner_bytes = _manifest(ALIGNER)
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-public-synthetic-") as temporary:
        audio = _audio(Path(temporary))
        with PeakRSS() as asr_rss:
            started = time.perf_counter()
            processor = AutoProcessor.from_pretrained(str(ASR), local_files_only=True)
            model = AutoModelForMultimodalLM.from_pretrained(str(ASR), torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to("mps").eval()
            inputs = processor.apply_transcription_request(audio=str(audio)).to("mps", torch.bfloat16)
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=128)
            torch.mps.synchronize()
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            parsed = processor.decode(generated, return_format="parsed")[0]
            asr_seconds = time.perf_counter() - started
        transcript = str(parsed.get("transcription", ""))
        language = str(parsed.get("language", ""))
        wer, cer = _error_rates(transcript)
        del output_ids, generated, inputs, model
        gc.collect()
        torch.mps.empty_cache()

        with PeakRSS() as aligner_rss:
            started = time.perf_counter()
            aligner_processor = AutoProcessor.from_pretrained(str(ALIGNER), local_files_only=True)
            aligner_model = AutoModelForTokenClassification.from_pretrained(str(ALIGNER), torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to("mps").eval()
            aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(audio=str(audio), transcript=transcript, language="German")
            aligner_inputs = aligner_inputs.to("mps", torch.bfloat16)
            with torch.inference_mode():
                outputs = aligner_model(**aligner_inputs)
            torch.mps.synchronize()
            aligned = aligner_processor.decode_forced_alignment(
                outputs.logits, aligner_inputs["input_ids"], word_lists,
                timestamp_token_id=aligner_model.config.timestamp_token_id,
            )[0]
            aligner_seconds = time.perf_counter() - started
        monotonic = all(
            isinstance(row.get("start_time"), (int, float))
            and isinstance(row.get("end_time"), (int, float))
            and row["end_time"] >= row["start_time"]
            and (index == 0 or row["start_time"] >= aligned[index - 1]["start_time"])
            for index, row in enumerate(aligned)
        )
    return {
        "schema_version": VERSION,
        "status": "COMPLETE_PUBLIC_SYNTHETIC_ONLY",
        "scope": "SELF_GENERATED_GERMAN_TTS_ONLY",
        "childlens_or_quarantine_accessed": False,
        "hosted_or_cloud_inference_used": False,
        "network_denial_sentinel_passed": True,
        "device": "MPS",
        "asr": {
            "revision": REVISIONS["asr"],
            "artifact_manifest_sha256": asr_manifest,
            "artifact_bytes": asr_bytes,
            "language_identification_correct": language.casefold() in {"german", "de", "deutsch"},
            "transcript_wer": wer,
            "transcript_cer": cer,
            "wall_time_seconds": round(asr_seconds, 3),
            "peak_process_rss_gib": round(asr_rss.peak / (1024**3), 3),
        },
        "forced_aligner": {
            "revision": REVISIONS["aligner"],
            "artifact_manifest_sha256": aligner_manifest,
            "artifact_bytes": aligner_bytes,
            "word_timestamp_count": len(aligned),
            "timestamp_count_matches_asr_words": len(aligned) == len(word_lists[0]),
            "timestamps_monotonic_and_nonnegative": monotonic,
            "wall_time_seconds": round(aligner_seconds, 3),
            "peak_process_rss_gib": round(aligner_rss.peak / (1024**3), 3),
        },
        "model_output_is_human_evidence": False,
        "restricted_inference_authorized": False,
    }


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        result = execute()
    except Exception as exc:
        result = {
            "schema_version": VERSION,
            "status": "TECHNICAL_FAILURE_PUBLIC_SYNTHETIC_ONLY",
            "failure_class": type(exc).__name__,
            "childlens_or_quarantine_accessed": False,
            "hosted_or_cloud_inference_used": False,
            "model_output_is_human_evidence": False,
            "restricted_inference_authorized": False,
        }
    pending = OUTPUT.with_suffix(".pending")
    pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(pending, 0o600)
    os.replace(pending, OUTPUT)
    print(json.dumps({"status": result["status"]}))
