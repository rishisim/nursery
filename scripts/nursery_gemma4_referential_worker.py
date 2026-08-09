#!/usr/bin/env python3
"""Descriptor-only Gemma referential worker for the frozen substitution.

Restricted media, the fixed job, and the checkpoint directory enter only as
inherited file descriptors. The coordinator launches this worker under OS
network denial. The worker emits no stdout/stderr content and never accepts a
ChildLens path, filename, transcript, identifier, or timestamp on its command
line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
import wave
from typing import Any, Mapping, Sequence


JOB_SCHEMA = "nursery-childlens-gemma4-referential-job-v1"
CHECKPOINT_SCHEMA = "nursery-childlens-gemma4-referential-checkpoint-row-v1"
OUTPUT_SCHEMA = "nursery-childlens-gemma4-referential-restricted-v1"
FRAME_OFFSETS = (-5.0, -2.5, 0.0, 2.5, 5.0)
WINDOW_COUNT = 137
ASCII_WHITESPACE = " \t\r\n\f\v"
SYSTEM_PROMPT = (
    "You are a fixed local measurement instrument. Use only the supplied audio clip and five ordered "
    "frames. Return only the exact JSON object requested. Do not add prose or markdown."
)
USER_PROMPT = (
    "The audio is the complete frozen candidate window. The five frames are ordered at offsets "
    "-5,-2.5,0,+2.5,+5 seconds from the window center. Classify only what is supported jointly by the "
    "spoken audio and visible frames. Count de-duplicated visible noun/object or verb/action candidates "
    "related to the speech, not all visible entities. Return exactly one JSON object with exactly these "
    "five keys and one allowed string value per key: candidate_count_bin=[zero,one,two,three_or_more,"
    "abstain]; visibility_bin=[none,partial,clear,abstain]; referential_status=[visible_candidate,"
    "null_or_irrelevant,ambiguous,undecidable]; lexical_support=[noun_object,verb_action,both,neither,"
    "undecidable]; lag_event_unit_bin=[lead_two_plus,lead_one,overlap,lag_one,lag_two_plus,undecidable]. "
    "Use abstain or undecidable whenever the audio, language, visibility, reference, or temporal relation "
    "is insufficient. Do not output confidence, candidate text, transcript text, translation, explanation, "
    "prose, or markdown."
)
ENUMS = {
    "candidate_count_bin": ("zero", "one", "two", "three_or_more", "abstain"),
    "visibility_bin": ("none", "partial", "clear", "abstain"),
    "referential_status": ("visible_candidate", "null_or_irrelevant", "ambiguous", "undecidable"),
    "lexical_support": ("noun_object", "verb_action", "both", "neither", "undecidable"),
    "lag_event_unit_bin": ("lead_two_plus", "lead_one", "overlap", "lag_one", "lag_two_plus", "undecidable"),
}
ALLOWED = {field: set(values) for field, values in ENUMS.items()}
EXACT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        field: {"type": "string", "enum": list(values)}
        for field, values in ENUMS.items()
    },
    "required": list(ALLOWED),
    "additionalProperties": False,
}
INVALID = {
    "candidate_count_bin": "abstain",
    "visibility_bin": "abstain",
    "referential_status": "undecidable",
    "lexical_support": "undecidable",
    "lag_event_unit_bin": "undecidable",
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
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("E_INPUT") from exc


def _network_denied() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            return False
    except OSError:
        return True


def _private_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    )


def _validate_spans(value: Any, *, nonoverlap: bool) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("E_SPANS")
    result: list[tuple[float, float]] = []
    previous_start = -1.0
    previous_end = -1.0
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"start_seconds", "end_seconds"}:
            raise RuntimeError("E_SPANS")
        start, end = row["start_seconds"], row["end_seconds"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0.0
            or float(end) <= float(start)
            or float(start) < previous_start
            or (nonoverlap and float(start) < previous_end)
        ):
            raise RuntimeError("E_SPANS")
        result.append((float(start), float(end)))
        previous_start = float(start)
        previous_end = float(end)
    return result


def _validate_job(job: Any) -> tuple[list[dict[str, Any]], int, str]:
    required = {
        "schema_version",
        "sample_item_count",
        "sample_total_seconds",
        "candidate_window_count",
        "resume_from",
        "parser_mode",
        "items",
    }
    if (
        not isinstance(job, Mapping)
        or set(job) != required
        or job.get("schema_version") != JOB_SCHEMA
        or job.get("sample_item_count") != 15
        or job.get("sample_total_seconds") != 900
        or job.get("candidate_window_count") != WINDOW_COUNT
        or type(job.get("resume_from")) is not int
        or not 0 <= job["resume_from"] <= WINDOW_COUNT
        or job.get("parser_mode") not in {"PRIMARY", "ONE_OUTER_FENCE"}
        or not isinstance(job.get("items"), list)
        or len(job["items"]) != 15
    ):
        raise RuntimeError("E_JOB")
    result: list[dict[str, Any]] = []
    keys: list[str] = []
    seconds = 0.0
    windows_total = 0
    for row in job["items"]:
        if not isinstance(row, Mapping) or set(row) != {"opaque_key", "media_fd", "intervals", "candidate_windows"}:
            raise RuntimeError("E_JOB")
        key, media_fd = row["opaque_key"], row["media_fd"]
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
            or key in keys
            or type(media_fd) is not int
            or media_fd < 3
        ):
            raise RuntimeError("E_JOB")
        intervals = _validate_spans(row["intervals"], nonoverlap=True)
        windows = _validate_spans(row["candidate_windows"], nonoverlap=False)
        seconds += sum(end - start for start, end in intervals)
        windows_total += len(windows)
        keys.append(key)
        result.append(
            {"opaque_key": key, "media_fd": media_fd, "intervals": intervals, "candidate_windows": windows}
        )
    if keys != sorted(keys) or abs(seconds - 900.0) > 0.01 or windows_total != WINDOW_COUNT:
        raise RuntimeError("E_JOB")
    return result, int(job["resume_from"]), str(job["parser_mode"])


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError("E_SCHEMA")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise RuntimeError("E_SCHEMA")


def _parse_exact(value: str, parser_mode: str = "PRIMARY") -> dict[str, str]:
    text = value.strip(ASCII_WHITESPACE)
    if parser_mode == "ONE_OUTER_FENCE":
        prefix = "```json\n" if text.startswith("```json\n") else "```\n" if text.startswith("```\n") else ""
        if prefix:
            if not text.endswith("\n```") or text.count("```") != 2:
                raise RuntimeError("E_SCHEMA")
            text = text[len(prefix) : -4].strip(ASCII_WHITESPACE)
    elif parser_mode != "PRIMARY":
        raise RuntimeError("E_SCHEMA")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError("E_SCHEMA") from exc
    if not isinstance(parsed, dict) or set(parsed) != set(ALLOWED):
        raise RuntimeError("E_SCHEMA")
    for key, allowed in ALLOWED.items():
        if type(parsed[key]) is not str or parsed[key] not in allowed:
            raise RuntimeError("E_SCHEMA")
    return {key: str(parsed[key]) for key in ALLOWED}


def _prompt_messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                *({"type": "image"} for _ in FRAME_OFFSETS),
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]


def prompt_and_schema_digest() -> str:
    return hashlib.sha256(_canonical({"messages": _prompt_messages(), "enums": ENUMS})).hexdigest()


def prompt_digest() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8") + b"\0" + USER_PROMPT.encode("utf-8")).hexdigest()


def exact_schema_digest() -> str:
    return hashlib.sha256(_canonical(EXACT_SCHEMA)).hexdigest()


def _run_quiet(command: list[str], *, pass_fds: tuple[int, ...] = (), timeout: int = 7200) -> None:
    environment = {
        "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=pass_fds,
        check=False,
        timeout=timeout,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError("E_LOCAL_TOOL")


def _decode_full_audio(ffmpeg: str, media_fd: int, target: Path) -> None:
    os.lseek(media_fd, 0, os.SEEK_SET)
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
            str(target),
        ],
        pass_fds=(media_fd,),
    )
    if not target.is_file() or target.stat().st_size <= 44:
        raise RuntimeError("E_AUDIO")


def _slice_exact_audio(source_path: Path, start: float, end: float, target: Path) -> None:
    with wave.open(str(source_path), "rb") as source:
        if source.getnchannels() != 1 or source.getframerate() != 16000 or source.getsampwidth() != 2:
            raise RuntimeError("E_AUDIO_FORMAT")
        left = max(0, round(start * 16000))
        right = min(source.getnframes(), round(end * 16000))
        if right <= left:
            raise RuntimeError("E_AUDIO_INTERVAL")
        source.setpos(left)
        frames = source.readframes(right - left)
    with wave.open(str(target), "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(16000)
        sink.writeframes(frames)


def _extract_frame(ffmpeg: str, media_fd: int, target_seconds: float, destination: Path) -> None:
    os.lseek(media_fd, 0, os.SEEK_SET)
    _run_quiet(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{target_seconds:.3f}",
            "-i",
            f"/dev/fd/{media_fd}",
            "-frames:v",
            "1",
            "-vf",
            "scale='min(448,iw)':-2",
            "-y",
            str(destination),
        ],
        pass_fds=(media_fd,),
        timeout=300,
    )
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("E_FRAME")


def _load_instrument(model_path: str) -> tuple[Any, Any, Any, Any]:
    from mlx_vlm import generate, load
    from mlx_vlm.utils import load_config

    model, processor = load(model_path, lazy=False)
    config = load_config(model_path)
    return model, processor, config, generate


def _render_frozen_prompt(processor: Any) -> str:
    rendered = processor.apply_chat_template(
        _prompt_messages(),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("E_CONTENT_ORDER")
    system_position = rendered.find(SYSTEM_PROMPT)
    audio_position = rendered.find("<|audio|>")
    image_positions: list[int] = []
    cursor = 0
    for _ in FRAME_OFFSETS:
        position = rendered.find("<|image|>", cursor)
        if position < 0:
            raise RuntimeError("E_CONTENT_ORDER")
        image_positions.append(position)
        cursor = position + len("<|image|>")
    user_position = rendered.find(USER_PROMPT)
    if (
        system_position < 0
        or audio_position < 0
        or user_position < 0
        or rendered.count("<|audio|>") != 1
        or rendered.count("<|image|>") != 5
        or not (system_position < audio_position < image_positions[0])
        or image_positions != sorted(image_positions)
        or not image_positions[-1] < user_position
    ):
        raise RuntimeError("E_CONTENT_ORDER")
    return rendered


def _infer_one(
    model: Any,
    processor: Any,
    config: Any,
    generate: Any,
    frames: Sequence[Path],
    audio: Path,
    parser_mode: str,
) -> tuple[dict[str, str], bool, str]:
    if len(frames) != 5 or not all(path.is_file() for path in frames) or not audio.is_file():
        raise RuntimeError("E_INPUT_FIXTURE")
    prompt = _render_frozen_prompt(processor)
    result = generate(
        model=model,
        processor=processor,
        prompt=prompt,
        image=[str(path) for path in frames],
        audio=[str(audio)],
        max_tokens=192,
        temperature=0.0,
        verbose=False,
    )
    raw = str(result.text)
    try:
        parsed = _parse_exact(raw, parser_mode)
        return parsed, True, raw
    except Exception:
        return dict(INVALID), False, raw


def _checkpoint_name(index: int) -> str:
    if not 0 <= index < WINDOW_COUNT:
        raise RuntimeError("E_CHECKPOINT_INDEX")
    return f"{index:06d}.json"


def _write_checkpoint(checkpoint_dir_fd: int, index: int, value: Mapping[str, Any]) -> None:
    final = _checkpoint_name(index)
    pending = f".pending-{os.getpid()}-{final}"
    descriptor = -1
    try:
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=checkpoint_dir_fd)
        payload = _canonical(value)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(pending, final, src_dir_fd=checkpoint_dir_fd, dst_dir_fd=checkpoint_dir_fd)
        os.fsync(checkpoint_dir_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(pending, dir_fd=checkpoint_dir_fd)
        except FileNotFoundError:
            pass


def run(job: Any, args: argparse.Namespace) -> None:
    items, resume_from, parser_mode = _validate_job(job)
    temporary_root = Path(os.environ.get("TMPDIR", ""))
    if not temporary_root.is_absolute() or not _private_directory(temporary_root):
        raise RuntimeError("E_TMPDIR")
    model, processor, config, generate = _load_instrument(args.model)
    global_index = 0
    with tempfile.TemporaryDirectory(prefix="gemma-referential-", dir=temporary_root) as temporary:
        directory = Path(temporary)
        os.chmod(directory, 0o700)
        for item_index, item in enumerate(items):
            item_windows = item["candidate_windows"]
            first = global_index
            last = first + len(item_windows)
            if last <= resume_from:
                global_index = last
                continue
            full_audio = directory / f"a-{item_index:03d}-full.wav"
            _decode_full_audio(args.ffmpeg, item["media_fd"], full_audio)
            try:
                for local_index, (start, end) in enumerate(item_windows):
                    current = first + local_index
                    if current < resume_from:
                        continue
                    clip = directory / f"a-{current:06d}.wav"
                    frames = [directory / f"f-{current:06d}-{frame_index}.png" for frame_index in range(5)]
                    try:
                        _slice_exact_audio(full_audio, start, end, clip)
                        center = (start + end) / 2.0
                        for frame_path, offset in zip(frames, FRAME_OFFSETS):
                            _extract_frame(args.ffmpeg, item["media_fd"], max(0.0, center + offset), frame_path)
                        parsed, schema_valid, raw = _infer_one(
                            model, processor, config, generate, frames, clip, parser_mode
                        )
                        checkpoint = {
                            "schema_version": CHECKPOINT_SCHEMA,
                            "global_window_index": current,
                            "opaque_key": item["opaque_key"],
                            "window_index": local_index,
                            "window_start_seconds": start,
                            "window_end_seconds": end,
                            **parsed,
                            "schema_valid": schema_valid,
                            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                            "raw_response": raw,
                            "pseudo_labels_are_ground_truth": False,
                            "human_validation": False,
                        }
                        _write_checkpoint(args.checkpoint_dir_fd, current, checkpoint)
                    finally:
                        clip.unlink(missing_ok=True)
                        for frame in frames:
                            frame.unlink(missing_ok=True)
            finally:
                full_audio.unlink(missing_ok=True)
            global_index = last
    if global_index != WINDOW_COUNT:
        raise RuntimeError("E_WINDOW_COUNT")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job-fd", type=int, required=True)
    parser.add_argument("--checkpoint-dir-fd", type=int, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not _network_denied():
            return 3
        run(_read_fd(args.job_fd), args)
        return 0
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
