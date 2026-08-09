#!/usr/bin/env python3
"""One predeclared schema-only Qwen2-VL engineering retry.

Restricted inputs and outputs use inherited descriptors. The worker is launched
inside the existing network-denial sandbox and emits no stdout/stderr content.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import re
import socket
from pathlib import Path
import sys
from typing import Any, Mapping


_BASE_PATH = Path(__file__).with_name("childlens_local_pseudo_adapter_v1_3.py")
_SPEC = importlib.util.spec_from_file_location("nursery_qwen2_retry_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("E_BASE_MODULE")
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)


SCHEMA = "nursery-childlens-qwen2-vl-schema-retry-restricted-v1"
OFFSETS = (-5.0, -2.5, 0.0, 2.5, 5.0)
ALLOWED = {
    "candidate_count_bin": {"zero", "one", "two", "three_or_more", "abstain"},
    "visibility_bin": {"none", "partial", "clear", "abstain"},
    "referential_status": {"visible_candidate", "null_or_irrelevant", "ambiguous", "undecidable"},
    "lexical_support": {"noun_object", "verb_action", "both", "neither", "undecidable"},
    "lag_event_unit_bin": {"lead_two_plus", "lead_one", "overlap", "lag_one", "lag_two_plus", "undecidable"},
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_fd(descriptor: int) -> Any:
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        payload = handle.read(64 * 1024 * 1024 + 1)
    if not payload or len(payload) > 64 * 1024 * 1024:
        raise RuntimeError("E_INPUT")
    return json.loads(payload)


def write_fd(descriptor: int, value: Any) -> None:
    with os.fdopen(os.dup(descriptor), "wb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())


def network_denied() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            return False
    except OSError:
        return True


def validate_intervals(value: Any, *, allow_empty: bool = False) -> list[tuple[float, float]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise RuntimeError("E_INTERVALS")
    result = []
    previous = -1.0
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"start_seconds", "end_seconds"}:
            raise RuntimeError("E_INTERVALS")
        start, end = row["start_seconds"], row["end_seconds"]
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(float(start)) or not math.isfinite(float(end))
            or float(start) < previous or float(start) < 0 or float(end) <= float(start)
        ):
            raise RuntimeError("E_INTERVALS")
        result.append((float(start), float(end)))
        previous = float(end)
    return result


def overlaps(intervals: Any, start: float, end: float) -> bool:
    return isinstance(intervals, list) and any(
        isinstance(row, Mapping)
        and isinstance(row.get("start_seconds"), (int, float))
        and isinstance(row.get("end_seconds"), (int, float))
        and float(row["start_seconds"]) < end
        and float(row["end_seconds"]) > start
        for row in intervals
    )


def parse_exact(raw: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        raise RuntimeError("E_SCHEMA")
    value = json.loads(match.group(0))
    if not isinstance(value, dict) or set(value) != set(ALLOWED):
        raise RuntimeError("E_SCHEMA")
    for key, allowed in ALLOWED.items():
        if value[key] not in allowed:
            raise RuntimeError("E_SCHEMA")
    return value


def infer(torch: Any, model: Any, processor: Any, images: list[Any], transcript: str) -> tuple[dict[str, str], bool]:
    instruction = (
        "You are a fixed local German-language measurement instrument. Five ordered frames are at "
        "offsets -5,-2.5,0,+2.5,+5 seconds around one speech window. The local ASR hypothesis is: "
        + json.dumps(transcript, ensure_ascii=False)
        + ". Return only one JSON object with exactly these keys and allowed values: "
        "candidate_count_bin [zero,one,two,three_or_more,abstain]; visibility_bin "
        "[none,partial,clear,abstain]; referential_status [visible_candidate,null_or_irrelevant,"
        "ambiguous,undecidable]; lexical_support [noun_object,verb_action,both,neither,undecidable]; "
        "lag_event_unit_bin [lead_two_plus,lead_one,overlap,lag_one,lag_two_plus,undecidable]. "
        "Count de-duplicated visible noun/object or verb/action candidates related to the speech, not all "
        "visible entities. Use undecidable/abstain when evidence is insufficient. No prose or markdown."
    )
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": instruction}]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to("mps") if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=192, do_sample=False, num_beams=1, use_cache=True)
    raw = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    try:
        return parse_exact(raw), True
    except Exception:
        return {
            "candidate_count_bin": "abstain",
            "visibility_bin": "abstain",
            "referential_status": "undecidable",
            "lexical_support": "undecidable",
            "lag_event_unit_bin": "undecidable",
        }, False


def execute(job: Any, model_path: str) -> dict[str, Any]:
    if (
        not isinstance(job, Mapping)
        or job.get("schema_version") != "nursery-childlens-qwen2-vlm-retry-job-v1"
        or job.get("sample_item_count") != 15
        or job.get("sample_total_seconds") != 900
        or not isinstance(job.get("items"), list)
        or len(job["items"]) != 15
    ):
        raise RuntimeError("E_JOB")
    try:
        torch, model, processor = base._load_qwen(model_path)
    except Exception as exc:
        raise RuntimeError("E_QWEN2_MODEL_LOAD") from exc
    output_items = []
    total_seconds = 0.0
    for item in job["items"]:
        key, media_fd = item.get("opaque_key"), item.get("media_fd")
        if not isinstance(key, str) or len(key) != 64 or not isinstance(media_fd, int):
            raise RuntimeError("E_JOB")
        intervals = validate_intervals(item.get("intervals"))
        total_seconds += sum(end - start for start, end in intervals)
        windows = validate_intervals(item.get("candidate_windows"), allow_empty=True)
        transcript_rows = item.get("transcript_hypotheses")
        if not isinstance(transcript_rows, list):
            raise RuntimeError("E_JOB")
        candidates = []
        for index, (start, end) in enumerate(windows):
            center = (start + end) / 2.0
            try:
                os.lseek(media_fd, 0, os.SEEK_SET)
                frames = base._decode_fixed_frames(media_fd, [max(0.0, center + offset) for offset in OFFSETS])
            except Exception as exc:
                raise RuntimeError("E_QWEN2_FRAME_DECODE") from exc
            transcript = " ".join(
                str(row.get("text", "")).strip()
                for row in transcript_rows
                if isinstance(row, Mapping) and overlaps(row.get("source_intervals"), start, end)
            ).strip()
            try:
                result, valid = infer(torch, model, processor, frames, transcript)
            except Exception as exc:
                raise RuntimeError("E_QWEN2_INFERENCE") from exc
            result.update({
                "window_index": index,
                "window_start_seconds": start,
                "window_end_seconds": end,
                "schema_valid": valid,
            })
            candidates.append(result)
        output_items.append({"opaque_key": key, "candidates": candidates})
    if abs(total_seconds - 900.0) > 0.01:
        raise RuntimeError("E_JOB")
    return {
        "schema_version": SCHEMA,
        "pseudo_labels_are_ground_truth": False,
        "human_validation": False,
        "network_disabled_during_inference": True,
        "schema_only_engineering_retry": True,
        "items": output_items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job-fd", type=int, required=True)
    parser.add_argument("--output-fd", type=int, required=True)
    parser.add_argument("--qwen-model", required=True)
    try:
        args = parser.parse_args()
        if not network_denied():
            return 3
        write_fd(args.output_fd, execute(read_fd(args.job_fd), args.qwen_model))
        return 0
    except Exception as exc:
        code = str(exc)
        if re.fullmatch(r"E_[A-Z0-9_]+", code) is None:
            code = f"E_{type(exc).__name__.upper()}"
        if "args" in locals():
            with contextlib.suppress(Exception):
                write_fd(args.output_fd, {"schema_version": "nursery-childlens-qwen2-retry-failure-v1", "failure_code": code})
        return 2


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
