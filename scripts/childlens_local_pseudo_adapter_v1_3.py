#!/usr/bin/env python3
"""Quarantine-only local instruments for the ChildLens v1.3 feasibility audit.

This program is never a learner and never emits a repository-facing result.  It
is launched only by ``run_childlens_model_assisted_pseudo_annotation_v1_3.py``
inside an OS network-denied process.  Restricted inputs and outputs are passed
through inherited descriptors; stdout and stderr are not an output channel.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


VERSION = "childlens-local-pseudo-adapter-v1.3.0"
AUDIO_SCHEMA = "childlens-restricted-audio-pseudo-labels-v1.3.0"
VLM_SCHEMA = "childlens-restricted-referential-pseudo-labels-v1.3.0"
FRAME_OFFSETS_SECONDS = (-5.0, -2.5, 0.0, 2.5, 5.0)
FIXED_MAX_FRAME_EDGE_PIXELS = 448
REFERENTIAL_VALUES = {
    "VISIBLE_CANDIDATE",
    "NULL_NOT_VISIBLE",
    "IRRELEVANT",
    "UNDECIDABLE",
    "UNUSABLE",
}
ROLE_VALUES = {"NON_CHILD", "CHILD", "OVERLAP", "UNCERTAIN", "NONSPEECH"}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json_fd(descriptor: int, maximum: int) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise RuntimeError("E_ADAPTER_INPUT")
    return json.loads(payload)


def _write_json_fd(descriptor: int, value: Any) -> None:
    payload = _canonical(value)
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_windows(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("E_WINDOWS")
    result: list[tuple[float, float]] = []
    previous = -1.0
    for row in value:
        if not isinstance(row, dict) or set(row) != {"start_seconds", "end_seconds"}:
            raise RuntimeError("E_WINDOWS")
        start = row["start_seconds"]
        end = row["end_seconds"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0
            or float(end) <= float(start)
            or float(start) < previous
        ):
            raise RuntimeError("E_WINDOWS")
        result.append((float(start), float(end)))
        previous = float(end)
    return result


def _run_quiet(
    command: list[str], *, timeout: int, pass_fds: tuple[int, ...] = ()
) -> None:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        close_fds=True,
        pass_fds=pass_fds,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError("E_LOCAL_INSTRUMENT")


def _decode_and_concatenate(
    *, ffmpeg: str, media_fd: int, windows: list[tuple[float, float]]
) -> tuple[Path, list[dict[str, float]]]:
    full_audio = Path("decoded-full.wav")
    concatenated = Path("official-speech-windows.wav")
    _run_quiet(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            f"/dev/fd/{media_fd}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(full_audio),
        ],
        timeout=7200,
        pass_fds=(media_fd,),
    )
    mapping: list[dict[str, float]] = []
    with wave.open(str(full_audio), "rb") as source:
        if source.getnchannels() != 1 or source.getframerate() != 16000 or source.getsampwidth() != 2:
            raise RuntimeError("E_AUDIO_FORMAT")
        with wave.open(str(concatenated), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16000)
            cursor_frames = 0
            for source_start, source_end in windows:
                start_frame = max(0, round(source_start * 16000))
                end_frame = min(source.getnframes(), round(source_end * 16000))
                if end_frame <= start_frame:
                    raise RuntimeError("E_AUDIO_WINDOW")
                source.setpos(start_frame)
                target.writeframes(source.readframes(end_frame - start_frame))
                duration_frames = end_frame - start_frame
                mapping.append(
                    {
                        "concat_start_seconds": cursor_frames / 16000.0,
                        "concat_end_seconds": (cursor_frames + duration_frames) / 16000.0,
                        "source_start_seconds": start_frame / 16000.0,
                        "source_end_seconds": end_frame / 16000.0,
                    }
                )
                cursor_frames += duration_frames
    full_audio.unlink()
    return concatenated, mapping


def _source_intervals(
    start: float, end: float, mapping: Iterable[Mapping[str, float]]
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for row in mapping:
        left = max(start, row["concat_start_seconds"])
        right = min(end, row["concat_end_seconds"])
        if right <= left:
            continue
        source_left = row["source_start_seconds"] + left - row["concat_start_seconds"]
        source_right = row["source_start_seconds"] + right - row["concat_start_seconds"]
        result.append({"start_seconds": source_left, "end_seconds": source_right})
    return result


def _whisper_segments(document: Any, mapping: list[dict[str, float]]) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(document, dict):
        raise RuntimeError("E_WHISPER_OUTPUT")
    result = document.get("result")
    language = "UNDETERMINED"
    if isinstance(result, dict) and isinstance(result.get("language"), str):
        language = result["language"][:64]
    transcription = document.get("transcription")
    if not isinstance(transcription, list):
        raise RuntimeError("E_WHISPER_OUTPUT")
    segments: list[dict[str, Any]] = []
    for row in transcription:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            continue
        offsets = row.get("offsets")
        if not isinstance(offsets, dict):
            continue
        start_ms = offsets.get("from")
        end_ms = offsets.get("to")
        if not isinstance(start_ms, (int, float)) or not isinstance(end_ms, (int, float)):
            continue
        start = max(0.0, float(start_ms) / 1000.0)
        end = max(start, float(end_ms) / 1000.0)
        intervals = _source_intervals(start, end, mapping)
        if not intervals:
            continue
        segments.append(
            {
                "concat_start_seconds": start,
                "concat_end_seconds": end,
                "source_intervals": intervals,
                "text": row["text"],
            }
        )
    return language, segments


def _silero_speech_segments(waveform: Any, model_path: str) -> list[tuple[float, float]]:
    """Run the fixed Silero 6.2.1 ONNX graph without importing torchaudio.

    The state/context mechanics and default 0.5/0.35 hysteresis parameters are
    frozen from the project's permissively licensed v6.2.1 reference wrapper.
    """

    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )
    signal = np.asarray(waveform, dtype=np.float32)
    chunk_size = 512
    context_size = 64
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, context_size), dtype=np.float32)
    probabilities: list[float] = []
    for offset in range(0, len(signal), chunk_size):
        chunk = signal[offset : offset + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        value = np.concatenate((context, chunk.reshape(1, -1)), axis=1)
        output, state = session.run(
            None,
            {
                "input": value,
                "state": state,
                "sr": np.array(16000, dtype=np.int64),
            },
        )
        probabilities.append(float(output[0][0]))
        context = value[:, -context_size:]

    threshold = 0.5
    negative_threshold = 0.35
    minimum_speech = round(0.250 * 16000)
    minimum_silence = round(0.100 * 16000)
    speech_pad = round(0.030 * 16000)
    triggered = False
    current_start = 0
    temporary_end = 0
    sample_segments: list[list[int]] = []
    for index, probability in enumerate(probabilities):
        current = index * chunk_size
        if probability >= threshold and not triggered:
            triggered = True
            current_start = current
            temporary_end = 0
            continue
        if probability >= threshold and temporary_end:
            temporary_end = 0
        if probability < negative_threshold and triggered:
            if not temporary_end:
                temporary_end = current
            if current - temporary_end >= minimum_silence:
                if temporary_end - current_start > minimum_speech:
                    sample_segments.append([current_start, temporary_end])
                triggered = False
                temporary_end = 0
    if triggered and len(signal) - current_start > minimum_speech:
        sample_segments.append([current_start, len(signal)])
    for index, segment in enumerate(sample_segments):
        if index == 0:
            segment[0] = max(0, segment[0] - speech_pad)
        if index < len(sample_segments) - 1:
            gap = sample_segments[index + 1][0] - segment[1]
            if gap < 2 * speech_pad:
                segment[1] += gap // 2
                sample_segments[index + 1][0] -= gap // 2
            else:
                segment[1] = min(len(signal), segment[1] + speech_pad)
                sample_segments[index + 1][0] = max(
                    0, sample_segments[index + 1][0] - speech_pad
                )
        else:
            segment[1] = min(len(signal), segment[1] + speech_pad)
    return [(start / 16000.0, end / 16000.0) for start, end in sample_segments]


def _silero_vad(
    audio_path: Path, mapping: list[dict[str, float]], model_path: str
) -> list[dict[str, Any]]:
    import numpy as np

    with wave.open(str(audio_path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    waveform = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    hypotheses = _silero_speech_segments(waveform, model_path)
    result: list[dict[str, Any]] = []
    for start, end in hypotheses:
        intervals = _source_intervals(start, end, mapping)
        if intervals:
            result.append(
                {
                    "concat_start_seconds": start,
                    "concat_end_seconds": end,
                    "source_intervals": intervals,
                }
            )
    return result


def run_audio(args: argparse.Namespace) -> None:
    job = _read_json_fd(args.job_fd, 4 * 1024 * 1024)
    if not isinstance(job, dict) or set(job) != {"schema_version", "windows"}:
        raise RuntimeError("E_AUDIO_JOB")
    if job["schema_version"] != "childlens-restricted-audio-job-v1.3.0":
        raise RuntimeError("E_AUDIO_JOB")
    windows = _validate_windows(job["windows"])
    audio_path, mapping = _decode_and_concatenate(
        ffmpeg=args.ffmpeg, media_fd=args.media_fd, windows=windows
    )
    vad = _silero_vad(audio_path, mapping, args.silero_model)
    prefix = Path("whisper-output")
    _run_quiet(
        [
            args.whisper_cli,
            "-m",
            args.whisper_model,
            "-f",
            str(audio_path),
            "-l",
            "auto",
            "-dtw",
            "large.v3.turbo",
            "-sow",
            "-ojf",
            "-of",
            str(prefix),
            "-np",
        ],
        timeout=7200,
    )
    whisper_path = prefix.with_suffix(".json")
    document = json.loads(whisper_path.read_text(encoding="utf-8"))
    language, segments = _whisper_segments(document, mapping)
    role = [
        {
            "source_intervals": segment["source_intervals"],
            "role": "UNCERTAIN",
            "basis": "CONSERVATIVE_NO_SEMANTIC_SPEAKER_ANCHOR",
        }
        for segment in segments
    ]
    _write_json_fd(
        args.output_fd,
        {
            "schema_version": AUDIO_SCHEMA,
            "pseudo_labels_are_ground_truth": False,
            "primary_evaluation_truth_allowed": False,
            "language_hypothesis": language,
            "official_window_count": len(windows),
            "official_window_seconds": sum(end - start for start, end in windows),
            "window_mapping": mapping,
            "vad_boundary_hypotheses": vad,
            "asr_hypotheses": segments,
            "speaker_role_hypotheses": role,
        },
    )


def _overlaps(intervals: Any, start: float, end: float) -> bool:
    if not isinstance(intervals, list):
        return False
    return any(
        isinstance(row, dict)
        and isinstance(row.get("start_seconds"), (int, float))
        and isinstance(row.get("end_seconds"), (int, float))
        and float(row["start_seconds"]) < end
        and float(row["end_seconds"]) > start
        for row in intervals
    )


def _decode_fixed_frames(media_fd: int, targets: list[float]) -> list[Any]:
    import av

    frames: list[Any] = []
    with os.fdopen(os.dup(media_fd), "rb") as media:
        container = av.open(media, mode="r")
        stream = next((candidate for candidate in container.streams if candidate.type == "video"), None)
        if stream is None:
            raise RuntimeError("E_VIDEO_STREAM")
        for target in targets:
            container.seek(
                int(max(0.0, target - 1.0) * av.time_base),
                any_frame=False,
                backward=True,
            )
            selected = None
            for frame in container.decode(stream):
                timestamp = frame.time
                selected = frame
                if timestamp is None or float(timestamp) >= target:
                    break
            if selected is None:
                raise RuntimeError("E_FRAME_DECODE")
            # Bound visual tokens and MPS memory independently of scene
            # content.  The same deterministic resize applies to every frame;
            # no confidence-, salience-, or success-based resampling occurs.
            image = selected.to_image().convert("RGB")
            image.thumbnail(
                (FIXED_MAX_FRAME_EDGE_PIXELS, FIXED_MAX_FRAME_EDGE_PIXELS),
                resample=Image.Resampling.LANCZOS,
            )
            frames.append(image)
        container.close()
    return frames


def _candidate_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for row in value[:20]:
        if isinstance(row, str):
            cleaned = " ".join(row.split())[:160]
            if cleaned:
                result.append(cleaned)
    return result


def _parse_candidate(raw: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        document: Any = {}
    else:
        with contextlib.suppress(json.JSONDecodeError):
            document = json.loads(match.group(0))
        if "document" not in locals():
            document = {}
    status = document.get("referential_status") if isinstance(document, dict) else None
    if status not in REFERENTIAL_VALUES:
        status = "UNDECIDABLE"
    role = document.get("source_role") if isinstance(document, dict) else None
    if role not in ROLE_VALUES:
        role = "UNCERTAIN"
    return {
        "referential_status": status,
        "noun_object_candidates": _candidate_strings(
            document.get("noun_object_candidates") if isinstance(document, dict) else None
        ),
        "verb_action_candidates": _candidate_strings(
            document.get("verb_action_candidates") if isinstance(document, dict) else None
        ),
        "source_role": role,
        "machine_parse_succeeded": bool(match is not None and isinstance(document, dict)),
        "raw_local_hypothesis": raw[:4096],
    }


def _load_qwen(model_path: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if not torch.backends.mps.is_available():
        raise RuntimeError("E_MPS_UNAVAILABLE")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        attn_implementation="eager",
    ).to("mps")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    return torch, model, processor


def _infer_candidate(
    torch: Any, model: Any, processor: Any, images: list[Any], transcript: str
) -> dict[str, Any]:
    prompt = (
        "You are a fixed local measurement instrument. Using only these five fixed-time frames "
        "and the local ASR hypothesis below, return exactly one JSON object with keys "
        "referential_status, noun_object_candidates, verb_action_candidates, source_role. "
        "referential_status must be VISIBLE_CANDIDATE, NULL_NOT_VISIBLE, IRRELEVANT, or "
        "UNDECIDABLE. source_role must be NON_CHILD, CHILD, OVERLAP, or UNCERTAIN. "
        "Use UNCERTAIN or UNDECIDABLE rather than guessing. ASR hypothesis: "
        + (transcript if transcript.strip() else "[NO_USABLE_ASR_HYPOTHESIS]")
    )
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to("mps") if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    prompt_length = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return _parse_candidate(raw)


def run_vlm(args: argparse.Namespace) -> None:
    job = _read_json_fd(args.job_fd, 4 * 1024 * 1024)
    audio = _read_json_fd(args.audio_fd, 32 * 1024 * 1024)
    if (
        not isinstance(job, dict)
        or set(job) != {"schema_version", "windows"}
        or job["schema_version"] != "childlens-restricted-vlm-job-v1.3.0"
        or not isinstance(audio, dict)
        or audio.get("schema_version") != AUDIO_SCHEMA
    ):
        raise RuntimeError("E_VLM_JOB")
    windows = _validate_windows(job["windows"])
    segments = audio.get("asr_hypotheses")
    if not isinstance(segments, list):
        raise RuntimeError("E_VLM_JOB")
    torch, model, processor = _load_qwen(args.qwen_model)
    candidates: list[dict[str, Any]] = []
    for window_index, (start, end) in enumerate(windows):
        center = (start + end) / 2.0
        targets = [max(0.0, center + offset) for offset in FRAME_OFFSETS_SECONDS]
        transcript = " ".join(
            row["text"]
            for row in segments
            if isinstance(row, dict)
            and isinstance(row.get("text"), str)
            and _overlaps(row.get("source_intervals"), start, end)
        )
        try:
            frames = _decode_fixed_frames(args.media_fd, targets)
            result = _infer_candidate(torch, model, processor, frames, transcript)
        except Exception:
            result = {
                "referential_status": "UNUSABLE",
                "noun_object_candidates": [],
                "verb_action_candidates": [],
                "source_role": "UNCERTAIN",
                "machine_parse_succeeded": False,
                "raw_local_hypothesis": "",
            }
        result.update(
            {
                "window_index": window_index,
                "window_start_seconds": start,
                "window_end_seconds": end,
                "fixed_frame_times_seconds": targets,
                "frame_offsets_seconds": list(FRAME_OFFSETS_SECONDS),
            }
        )
        candidates.append(result)
    if candidates and all(row["referential_status"] == "UNUSABLE" for row in candidates):
        raise RuntimeError("E_VLM_STRUCTURALLY_UNAVAILABLE")
    _write_json_fd(
        args.output_fd,
        {
            "schema_version": VLM_SCHEMA,
            "pseudo_labels_are_ground_truth": False,
            "primary_evaluation_truth_allowed": False,
            "frame_selection_prediction_independent": True,
            "confidence_adaptive_resampling": False,
            "window_count": len(windows),
            "candidates": candidates,
        },
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("audio", "vlm"), required=True)
    parser.add_argument("--job-fd", type=int, required=True)
    parser.add_argument("--media-fd", type=int, required=True)
    parser.add_argument("--output-fd", type=int, required=True)
    parser.add_argument("--audio-fd", type=int)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--whisper-cli")
    parser.add_argument("--whisper-model")
    parser.add_argument("--silero-model")
    parser.add_argument("--qwen-model")
    args = parser.parse_args(argv)
    if args.mode == "audio" and not all(
        (args.ffmpeg, args.whisper_cli, args.whisper_model, args.silero_model)
    ):
        parser.error("audio instrument arguments required")
    if args.mode == "vlm" and (args.audio_fd is None or not args.qwen_model):
        parser.error("vlm instrument arguments required")
    return args


def main() -> int:
    try:
        args = parse_args(sys.argv[1:])
        if args.mode == "audio":
            run_audio(args)
        else:
            run_vlm(args)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
