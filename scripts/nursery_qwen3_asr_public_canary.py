#!/usr/bin/env python3
"""Public synthetic, network-denied Qwen3-ASR/aligner canary.

This script never discovers or opens the ChildLens quarantine. It emits only
booleans, versions, aggregate resource facts, and public-artifact digests.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import psutil
import torch
from qwen_asr import Qwen3ASRModel


ROOT = Path("/Users/rishisim/Library/Application Support/ChildLens Instruments/provisional-calibration-v1")
ASR = ROOT / "models/Qwen3-ASR-1.7B"
ALIGNER = ROOT / "models/Qwen3-ForcedAligner-0.6B"
OUTPUT = Path("/Users/rishisim/Documents/research/nursery/output/nursery_program_convergence_v1/qwen3_asr_public_canary.json")
REFERENCE = "Der rote Ball liegt auf dem Tisch."
REVISIONS = {
    "asr": "7278e1e70fe206f11671096ffdd38061171dd6e5",
    "aligner": "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
}


def network_is_denied() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            return False
    except OSError:
        return True


def tree_manifest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total = 0
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or ".cache" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(block)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_digest.hexdigest()}\n".encode())
        total += size
        count += 1
    if not count:
        raise RuntimeError("E_EMPTY_MODEL")
    return digest.hexdigest(), total, count


def make_audio(directory: Path) -> Path:
    aiff = directory / "synthetic.aiff"
    wav = directory / "synthetic.wav"
    subprocess.run(
        ["/usr/bin/say", "-v", "Anna", REFERENCE, "-o", str(aiff)],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
    )
    subprocess.run(
        [
            "/opt/homebrew/bin/ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(aiff),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
    )
    return wav


def execute() -> dict[str, object]:
    if not network_is_denied():
        raise RuntimeError("E_NETWORK_NOT_DENIED")
    if not torch.backends.mps.is_available():
        raise RuntimeError("E_MPS_UNAVAILABLE")
    asr_digest, asr_bytes, asr_files = tree_manifest(ASR)
    aligner_digest, aligner_bytes, aligner_files = tree_manifest(ALIGNER)
    peak_before = psutil.Process().memory_info().rss
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nursery-qwen3-public-") as temporary:
        audio = make_audio(Path(temporary))
        model = Qwen3ASRModel.from_pretrained(
            str(ASR),
            dtype=torch.float16,
            device_map="mps",
            max_inference_batch_size=1,
            max_new_tokens=128,
            forced_aligner=str(ALIGNER),
            forced_aligner_kwargs={"dtype": torch.float16, "device_map": "mps"},
        )
        results = model.transcribe(
            audio=str(audio),
            language="German",
            return_time_stamps=True,
        )
    elapsed = time.perf_counter() - started
    if len(results) != 1:
        raise RuntimeError("E_RESULT_COUNT")
    result = results[0]
    text = str(result.text)
    language = str(result.language)
    timestamps = list(result.time_stamps or [])
    monotonic = True
    prior = -1.0
    for row in timestamps:
        start = float(row.start_time)
        end = float(row.end_time)
        if start < 0 or end < start or start < prior:
            monotonic = False
        prior = start
    return {
        "schema_version": "nursery-qwen3-asr-public-canary-v1",
        "status": "PASS",
        "scope": "SELF_GENERATED_GERMAN_TTS_ONLY",
        "childlens_or_quarantine_accessed": False,
        "hosted_or_cloud_inference_used": False,
        "network_denial_sentinel_passed": True,
        "device": "mps",
        "asr_revision": REVISIONS["asr"],
        "aligner_revision": REVISIONS["aligner"],
        "asr_manifest_sha256": asr_digest,
        "aligner_manifest_sha256": aligner_digest,
        "asr_artifact_bytes": asr_bytes,
        "aligner_artifact_bytes": aligner_bytes,
        "asr_file_count": asr_files,
        "aligner_file_count": aligner_files,
        "language_identification_german": language.casefold() in {"german", "de", "deutsch"},
        "nonempty_transcript": bool(text.strip()),
        "word_timestamp_count": len(timestamps),
        "timestamps_monotonic_and_nonnegative": monotonic,
        "wall_time_seconds": round(elapsed, 3),
        "process_rss_delta_gib": round((psutil.Process().memory_info().rss - peak_before) / 1024**3, 3),
        "pseudo_output_is_human_evidence": False,
        "restricted_inference_authorized_by_canary": False,
    }


def main() -> int:
    try:
        result = execute()
    except Exception as exc:
        result = {
            "schema_version": "nursery-qwen3-asr-public-canary-v1",
            "status": "FAIL",
            "failure_class": type(exc).__name__,
            "childlens_or_quarantine_accessed": False,
            "hosted_or_cloud_inference_used": False,
            "pseudo_output_is_human_evidence": False,
            "restricted_inference_authorized_by_canary": False,
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pending = OUTPUT.with_suffix(".pending")
    pending.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(pending, 0o600)
    os.replace(pending, OUTPUT)
    print(result["status"])
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
