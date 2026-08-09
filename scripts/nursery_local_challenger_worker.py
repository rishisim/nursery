#!/usr/bin/env python3
"""Restricted, descriptor-only local challenger worker for pseudo-calibration.

This file contains generic inference code. It is launched only by the public
coordinator under an OS network-denial sandbox. Restricted media and jobs enter
through inherited descriptors; results leave through one inherited descriptor.
The worker never writes to stdout/stderr and never accepts a filesystem path to
ChildLens content.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import wave
from typing import Any, Iterable, Mapping


ASR_SCHEMA = "nursery-childlens-qwen3-asr-restricted-v1"
VLM_SCHEMA = "nursery-childlens-qwen3-vl-restricted-v1"
FRAME_OFFSETS = (-5.0, -2.5, 0.0, 2.5, 5.0)
ALLOWED = {
    "candidate_count_bin": {"zero", "one", "two", "three_or_more", "abstain"},
    "visibility_bin": {"none", "partial", "clear", "abstain"},
    "referential_status": {"visible_candidate", "null_or_irrelevant", "ambiguous", "undecidable"},
    "lexical_support": {"noun_object", "verb_action", "both", "neither", "undecidable"},
    "lag_event_unit_bin": {"lead_two_plus", "lead_one", "overlap", "lag_one", "lag_two_plus", "undecidable"},
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_fd(descriptor: int, maximum: int = 64 * 1024 * 1024) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise RuntimeError("E_INPUT")
    return json.loads(payload)


def _write_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _network_denied() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            return False
    except OSError:
        return True


def _validate_intervals(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("E_INTERVALS")
    result: list[tuple[float, float]] = []
    previous = -1.0
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"start_seconds", "end_seconds"}:
            raise RuntimeError("E_INTERVALS")
        start, end = row["start_seconds"], row["end_seconds"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < previous
            or float(start) < 0
            or float(end) <= float(start)
        ):
            raise RuntimeError("E_INTERVALS")
        result.append((float(start), float(end)))
        previous = float(end)
    return result


def _validate_items(job: Any, mode: str) -> list[dict[str, Any]]:
    if (
        not isinstance(job, Mapping)
        or job.get("schema_version") != f"nursery-childlens-{mode}-job-v1"
        or job.get("sample_item_count") != 15
        or job.get("sample_total_seconds") != 900
        or not isinstance(job.get("items"), list)
        or len(job["items"]) != 15
    ):
        raise RuntimeError("E_JOB")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    seconds = 0.0
    for row in job["items"]:
        if not isinstance(row, Mapping):
            raise RuntimeError("E_JOB")
        key = row.get("opaque_key")
        media_fd = row.get("media_fd")
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(ch not in "0123456789abcdef" for ch in key)
            or key in seen
            or not isinstance(media_fd, int)
            or isinstance(media_fd, bool)
            or media_fd < 3
        ):
            raise RuntimeError("E_JOB")
        intervals = _validate_intervals(row.get("intervals"))
        seconds += sum(end - start for start, end in intervals)
        clean: dict[str, Any] = {"opaque_key": key, "media_fd": media_fd, "intervals": intervals}
        if mode == "vlm":
            windows = _validate_intervals(row.get("candidate_windows")) if row.get("candidate_windows") else []
            transcript = row.get("transcript_hypotheses")
            if not isinstance(transcript, list):
                raise RuntimeError("E_JOB")
            clean["candidate_windows"] = windows
            clean["transcript_hypotheses"] = transcript
        result.append(clean)
        seen.add(key)
    if abs(seconds - 900.0) > 0.01:
        raise RuntimeError("E_JOB")
    return result


def _run_quiet(command: list[str], *, pass_fds: tuple[int, ...] = (), timeout: int = 7200) -> None:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=pass_fds,
        check=False,
        timeout=timeout,
        env={
            "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "HOME": os.environ.get("HOME", "/var/empty"),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError("E_LOCAL_TOOL")


def _decode_concat(ffmpeg: str, media_fd: int, intervals: list[tuple[float, float]], target: Path) -> list[dict[str, float]]:
    full = target.with_name(target.stem + "-full.wav")
    os.lseek(media_fd, 0, os.SEEK_SET)
    _run_quiet(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", f"/dev/fd/{media_fd}", "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", "-y", str(full),
        ],
        pass_fds=(media_fd,),
    )
    mapping: list[dict[str, float]] = []
    try:
        with wave.open(str(full), "rb") as source:
            if source.getnchannels() != 1 or source.getframerate() != 16000 or source.getsampwidth() != 2:
                raise RuntimeError("E_AUDIO_FORMAT")
            with wave.open(str(target), "wb") as sink:
                sink.setnchannels(1)
                sink.setsampwidth(2)
                sink.setframerate(16000)
                cursor = 0
                for start, end in intervals:
                    left = max(0, round(start * 16000))
                    right = min(source.getnframes(), round(end * 16000))
                    if right <= left:
                        raise RuntimeError("E_AUDIO_INTERVAL")
                    source.setpos(left)
                    sink.writeframes(source.readframes(right - left))
                    mapping.append(
                        {
                            "concat_start_seconds": cursor / 16000.0,
                            "concat_end_seconds": (cursor + right - left) / 16000.0,
                            "source_start_seconds": left / 16000.0,
                            "source_end_seconds": right / 16000.0,
                        }
                    )
                    cursor += right - left
    finally:
        full.unlink(missing_ok=True)
    return mapping


def _map_span(start: float, end: float, mapping: Iterable[Mapping[str, float]]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for row in mapping:
        left = max(start, float(row["concat_start_seconds"]))
        right = min(end, float(row["concat_end_seconds"]))
        if right <= left:
            continue
        source_start = float(row["source_start_seconds"]) + left - float(row["concat_start_seconds"])
        source_end = float(row["source_start_seconds"]) + right - float(row["concat_start_seconds"])
        result.append({"start_seconds": round(source_start, 3), "end_seconds": round(source_end, 3)})
    return result


def run_asr(job: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from qwen_asr import Qwen3ASRModel

    items = _validate_items(job, "asr")
    model = Qwen3ASRModel.from_pretrained(
        args.asr_model,
        dtype=torch.float16,
        device_map="mps",
        max_inference_batch_size=1,
        max_new_tokens=1024,
        forced_aligner=args.aligner_model,
        forced_aligner_kwargs={"dtype": torch.float16, "device_map": "mps"},
    )
    output_items: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-") as temporary:
        directory = Path(temporary)
        for index, item in enumerate(items):
            audio = directory / f"audio-{index:03d}.wav"
            mapping = _decode_concat(args.ffmpeg, item["media_fd"], item["intervals"], audio)
            result_list = model.transcribe(audio=str(audio), language=None, return_time_stamps=True)
            if len(result_list) != 1:
                raise RuntimeError("E_ASR_RESULT")
            result = result_list[0]
            words: list[dict[str, Any]] = []
            for word in list(result.time_stamps or []):
                start = float(word.start_time)
                end = float(word.end_time)
                intervals = _map_span(start, end, mapping)
                if not intervals:
                    continue
                words.append({"text": str(word.text), "source_intervals": intervals})
            utterances: list[dict[str, Any]] = []
            for word in words:
                for interval in word["source_intervals"]:
                    text = word["text"].strip()
                    if not text:
                        continue
                    if (
                        utterances
                        and interval["start_seconds"] - utterances[-1]["source_intervals"][-1]["end_seconds"] <= 0.8
                        and interval["start_seconds"] >= utterances[-1]["source_intervals"][-1]["start_seconds"]
                    ):
                        utterances[-1]["text"] = (utterances[-1]["text"] + " " + text).strip()
                        utterances[-1]["source_intervals"].append(interval)
                    else:
                        utterances.append({"text": text, "source_intervals": [interval]})
            output_items.append(
                {
                    "opaque_key": item["opaque_key"],
                    "language_hypothesis": str(result.language)[:64],
                    "transcript_hypothesis": str(result.text),
                    "word_hypotheses": words,
                    "utterance_hypotheses": utterances,
                    "abstain": not bool(str(result.text).strip()),
                }
            )
    return {
        "schema_version": ASR_SCHEMA,
        "pseudo_labels_are_ground_truth": False,
        "human_validation": False,
        "network_disabled_during_inference": True,
        "items": output_items,
    }


def _parse_exact(value: str) -> dict[str, str]:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != set(ALLOWED):
        raise RuntimeError("E_SCHEMA")
    for key, allowed in ALLOWED.items():
        if parsed[key] not in allowed:
            raise RuntimeError("E_SCHEMA")
    return parsed


def _overlaps(intervals: Any, start: float, end: float) -> bool:
    if not isinstance(intervals, list):
        return False
    for row in intervals:
        if isinstance(row, Mapping):
            left, right = row.get("start_seconds"), row.get("end_seconds")
            if isinstance(left, (int, float)) and isinstance(right, (int, float)) and float(left) < end and float(right) > start:
                return True
    return False


def _extract_frame(ffmpeg: str, media_fd: int, target_seconds: float, destination: Path) -> None:
    os.lseek(media_fd, 0, os.SEEK_SET)
    _run_quiet(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{target_seconds:.3f}",
            "-i", f"/dev/fd/{media_fd}", "-frames:v", "1", "-vf", "scale='min(448,iw)':-2",
            "-y", str(destination),
        ],
        pass_fds=(media_fd,),
        timeout=300,
    )
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("E_FRAME")


def run_vlm(job: Any, args: argparse.Namespace) -> dict[str, Any]:
    from mlx_vlm import apply_chat_template, generate, load

    items = _validate_items(job, "vlm")
    model, processor = load(args.vlm_model, lazy=False, strict=True)
    output_items: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen3-vl-") as temporary:
        directory = Path(temporary)
        for item_index, item in enumerate(items):
            candidates: list[dict[str, Any]] = []
            for window_index, (start, end) in enumerate(item["candidate_windows"]):
                center = (start + end) / 2.0
                frame_paths: list[str] = []
                for frame_index, offset in enumerate(FRAME_OFFSETS):
                    target = max(0.0, center + offset)
                    frame = directory / f"f-{item_index:03d}-{window_index:04d}-{frame_index}.png"
                    _extract_frame(args.ffmpeg, item["media_fd"], target, frame)
                    frame_paths.append(str(frame))
                transcript = " ".join(
                    str(row.get("text", "")).strip()
                    for row in item["transcript_hypotheses"]
                    if isinstance(row, Mapping) and _overlaps(row.get("source_intervals"), start, end)
                ).strip()
                instruction = (
                    "You are a fixed local German-language measurement instrument. Five ordered frames are "
                    "at offsets -5,-2.5,0,+2.5,+5 seconds around one speech window. The local ASR hypothesis is: "
                    + json.dumps(transcript, ensure_ascii=False)
                    + ". Return only one JSON object with exactly these keys and allowed values: "
                    "candidate_count_bin [zero,one,two,three_or_more,abstain]; visibility_bin "
                    "[none,partial,clear,abstain]; referential_status [visible_candidate,null_or_irrelevant,"
                    "ambiguous,undecidable]; lexical_support [noun_object,verb_action,both,neither,undecidable]; "
                    "lag_event_unit_bin [lead_two_plus,lead_one,overlap,lag_one,lag_two_plus,undecidable]. "
                    "Count de-duplicated visible noun/object or verb/action candidates related to the speech, not "
                    "all visible entities. Use undecidable/abstain when evidence is insufficient. No prose."
                )
                prompt = apply_chat_template(
                    processor, model.config, instruction, num_images=5, enable_thinking=False
                )
                response = generate(
                    model, processor, prompt, image=frame_paths, max_tokens=192,
                    temperature=0.0, verbose=False, enable_thinking=False,
                )
                try:
                    parsed = _parse_exact(response.text)
                    schema_valid = True
                except Exception:
                    parsed = {
                        "candidate_count_bin": "abstain",
                        "visibility_bin": "abstain",
                        "referential_status": "undecidable",
                        "lexical_support": "undecidable",
                        "lag_event_unit_bin": "undecidable",
                    }
                    schema_valid = False
                parsed.update(
                    {
                        "window_index": window_index,
                        "window_start_seconds": start,
                        "window_end_seconds": end,
                        "schema_valid": schema_valid,
                    }
                )
                candidates.append(parsed)
                for frame in frame_paths:
                    Path(frame).unlink(missing_ok=True)
            output_items.append({"opaque_key": item["opaque_key"], "candidates": candidates})
    return {
        "schema_version": VLM_SCHEMA,
        "pseudo_labels_are_ground_truth": False,
        "human_validation": False,
        "network_disabled_during_inference": True,
        "items": output_items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("asr", "vlm"), required=True)
    parser.add_argument("--job-fd", type=int, required=True)
    parser.add_argument("--output-fd", type=int, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--asr-model")
    parser.add_argument("--aligner-model")
    parser.add_argument("--vlm-model")
    args = parser.parse_args()
    if args.mode == "asr" and not (args.asr_model and args.aligner_model):
        parser.error("ASR paths required")
    if args.mode == "vlm" and not args.vlm_model:
        parser.error("VLM path required")
    return args


def main() -> int:
    try:
        args = parse_args()
        if not _network_denied():
            return 3
        job = _read_fd(args.job_fd)
        result = run_asr(job, args) if args.mode == "asr" else run_vlm(job, args)
        _write_fd(args.output_fd, result)
        return 0
    except Exception:
        return 2


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
