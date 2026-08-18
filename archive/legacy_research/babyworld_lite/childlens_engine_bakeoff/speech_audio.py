"""Create separately authoritative speech and timing records for an episode."""

from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import sha256_file, write_json


def _read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16 speech segment: {path}")
        rate = handle.getframerate()
        samples = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype="<i2"
        ).copy()
    return samples, rate


def _write_pcm16_mono(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _word_alignment(
    text: str, start_s: float, end_s: float
) -> list[dict[str, Any]]:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not words:
        return []
    weights = np.asarray([max(1, len(word)) for word in words], dtype=float)
    boundaries = np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
    duration = end_s - start_s
    return [
        {
            "word": word,
            "start_s": start_s + duration * float(boundaries[index]),
            "end_s": start_s + duration * float(boundaries[index + 1]),
        }
        for index, word in enumerate(words)
    ]


def generate_authoritative_speech(
    episode: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Synthesize local speech, mix it on the episode clock, and align text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = output_dir / "speech_segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    speech = episode["language"]["speech"]
    say_path = Path("/usr/bin/say")
    ffmpeg_path = Path("/opt/homebrew/bin/ffmpeg")
    if sha256_file(say_path) != speech["say_executable_sha256"]:
        raise RuntimeError("macOS say executable does not match the frozen pin")
    if sha256_file(ffmpeg_path) != speech["ffmpeg_executable_sha256"]:
        raise RuntimeError("FFmpeg executable does not match the frozen pin")

    sample_rate = int(speech["sample_rate_hz"])
    duration_s = float(episode["activity"]["duration_s"])
    full_samples = np.zeros(round(duration_s * sample_rate), dtype=np.int32)
    alignments = []
    transcript_lines = []
    for utterance in episode["language"]["utterances"]:
        identifier = utterance["id"]
        aiff_path = segment_dir / f"{identifier}.aiff"
        wav_path = segment_dir / f"{identifier}.wav"
        subprocess.run(
            [
                str(say_path),
                "-v",
                speech["voice"],
                "-r",
                str(speech["rate_words_per_minute"]),
                "-o",
                str(aiff_path),
                utterance["text"],
            ],
            check=True,
        )
        subprocess.run(
            [
                str(ffmpeg_path),
                "-v",
                "error",
                "-y",
                "-i",
                str(aiff_path),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ],
            check=True,
        )
        samples, actual_rate = _read_pcm16_mono(wav_path)
        if actual_rate != sample_rate:
            raise RuntimeError("converted speech segment has the wrong rate")
        start_sample = round(float(utterance["start_s"]) * sample_rate)
        end_sample = start_sample + len(samples)
        if end_sample > len(full_samples):
            raise ValueError(f"utterance {identifier} exceeds episode duration")
        end_s = end_sample / sample_rate
        if end_s > float(utterance["must_end_by_s"]):
            raise ValueError(
                f"utterance {identifier} ends at {end_s:.3f}s, after its "
                f"frozen event window {utterance['must_end_by_s']:.3f}s"
            )
        full_samples[start_sample:end_sample] += samples.astype(np.int32)
        row = {
            **utterance,
            "end_s": end_s,
            "duration_s": len(samples) / sample_rate,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "segment_sha256": sha256_file(wav_path),
            "words": _word_alignment(
                utterance["text"], float(utterance["start_s"]), end_s
            ),
        }
        alignments.append(row)
        transcript_lines.append(
            f"{utterance['start_s']:.6f}\t{end_s:.6f}\t{utterance['text']}"
        )

    clipped_samples = int(np.sum(np.abs(full_samples) > np.iinfo(np.int16).max))
    full_samples = np.clip(
        full_samples, np.iinfo(np.int16).min, np.iinfo(np.int16).max
    ).astype(np.int16)
    waveform_path = output_dir / "speech.wav"
    _write_pcm16_mono(waveform_path, full_samples, sample_rate)
    transcript_path = output_dir / "transcript.txt"
    transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
    alignment_path = output_dir / "speech_alignment.json"
    alignment = {
        "schema": "AuthoritativeSpeechAlignment.v1",
        "clock": "episode_seconds",
        "utterances": alignments,
        "word_alignment_method": (
            "deterministic character-proportional subdivision within each "
            "synthesized utterance; not acoustic forced alignment"
        ),
    }
    write_json(alignment_path, alignment)
    receipt = {
        "schema": "AuthoritativeSpeechQA.v1",
        "engine": speech["engine"],
        "voice": speech["voice"],
        "locale": speech["locale"],
        "sample_rate_hz": sample_rate,
        "channels": 1,
        "sample_format": "pcm_s16le",
        "duration_s": len(full_samples) / sample_rate,
        "sample_count": len(full_samples),
        "utterance_count": len(alignments),
        "clipped_samples": clipped_samples,
        "neural_render_audio_used": False,
        "waveform_sha256": sha256_file(waveform_path),
        "transcript_sha256": sha256_file(transcript_path),
        "alignment_sha256": sha256_file(alignment_path),
    }
    write_json(output_dir / "speech_qa.json", receipt)
    return receipt
