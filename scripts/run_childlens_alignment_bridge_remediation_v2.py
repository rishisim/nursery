#!/usr/bin/env python3
"""Run the development-only ChildLens speech-measurement remediation v2."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import wave
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_remediation_v2.json"
V1_FREEZE = ROOT / "output/childlens_alignment_bridge_v1/freeze_receipt.json"
V1_DEV = ROOT / "output/childlens_alignment_bridge_v1/development_gate_report_v1_0_1.json"
PUBLIC_ROOT = ROOT / "output/childlens_alignment_bridge_remediation_v2"
FREEZE_RECEIPT = PUBLIC_ROOT / "freeze_receipt.json"
SEGMENT_RECEIPT = PUBLIC_ROOT / "segment_freeze_receipt.json"
REPORT_JSON = PUBLIC_ROOT / "remediation_report.json"

PRIVATE_RELATIVE = Path("provisional_calibration_v1/childlens_alignment_bridge_remediation_v2")
PRIVATE_DEVELOPMENT = PRIVATE_RELATIVE / "restricted_development_manifest.json"
PRIVATE_SEGMENTS = PRIVATE_RELATIVE / "restricted_frozen_segment_manifest.json"
PRIVATE_RESULT = PRIVATE_RELATIVE / "restricted_remediation_result.json"

INSTRUMENT_ROOT = Path.home() / "Library/Application Support/ChildLens Instruments"
WHISPER_CLI = (
    INSTRUMENT_ROOT
    / "v1.3/.worktrees/whisper.cpp-v1.9.1/build/bin/whisper-cli"
)
PRIMARY_MODEL = INSTRUMENT_ROOT / "v2/models/ggml-large-v3.bin"
SENSITIVITY_MODEL = INSTRUMENT_ROOT / "v1.3/models/ggml-large-v3-turbo.bin"
SILERO_MODEL = (
    INSTRUMENT_ROOT / "v1.3/models/silero-vad-6.2.1/silero_vad.onnx"
)
EMBEDDING_MODEL = (
    INSTRUMENT_ROOT / "v2/models/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_PYTHON = INSTRUMENT_ROOT / "v1.3/venv/bin/python3.10"
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"
EXPECTED_HASHES = {
    WHISPER_CLI: "9613b31e5380c184ae29ccb1d4046953d7037e8eb55308c9f1a34f145143b892",
    PRIMARY_MODEL: "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
    SENSITIVITY_MODEL: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    SILERO_MODEL: "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    EMBEDDING_MODEL / "model.safetensors": (
        "eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b"
    ),
}
V1_PROTOCOL_SHA256 = "0bb98508214c1c668f04bcc5a2722ab3b1117c72ec9c23ac1f5accf91d648639"
V1_SPLIT_SHA256 = "926391535972830289315b95e6b4e4893a3e82aeac03b371ab3212ca324b906a"
V1_DEV_SHA256 = "ae88ed4ef093c567da20eb3ee63e1a7beb5cf8fcd6381d12ed04de218cb32dda"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_706
SPEECH_EVENTS = {
    "child_talking",
    "other_person_talking",
    "overheard_speech",
    "singing/humming",
}

sys.path.insert(0, str(ROOT))
from babyworld_lite.childlens_alignment_bridge_v1.preflight import (  # noqa: E402
    BridgeError,
    character_similarity,
    participant_bootstrap_interval,
)
from babyworld_lite.childlens_alignment_bridge_v2.diagnostics import (  # noqa: E402
    interval_overlap_seconds,
    interval_set_precision_recall,
    matched_boundary_f1,
    normalize_text,
)
from babyworld_lite.childlens_alignment_bridge_v2.segmentation import (  # noqa: E402
    silero_speech_segments,
)


def _fail(code: str) -> None:
    raise BridgeError(code)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _fail("E_CANONICAL")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        if not path.is_file() or path.is_symlink():
            _fail("E_FILE")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("E_FILE")


def _private_file(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _private_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) & 0o077 == 0
    )


def _write_once(path: Path, value: Any, *, private: bool) -> None:
    payload = _canonical(value) + b"\n"
    file_mode = 0o600 if private else 0o644
    directory_mode = 0o700 if private else 0o755
    path.parent.mkdir(parents=True, exist_ok=True, mode=directory_mode)
    if private:
        os.chmod(path.parent, 0o700)
    if path.exists():
        if path.read_bytes() != payload:
            _fail("E_IMMUTABLE_CONFLICT")
        return
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, file_mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, file_mode)
    finally:
        if pending.exists():
            pending.unlink()


def _discover_runtime() -> Path:
    candidates = []
    for hidden in ROOT.parent.iterdir():
        if (
            hidden.name.startswith(".")
            and "childlens" in hidden.name.casefold()
            and _private_directory(hidden)
        ):
            candidates.extend(
                path.parents[1]
                for path in hidden.rglob(
                    "post_acquisition_v1_2/restricted_measurement_manifest.json"
                )
                if _private_file(path)
            )
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) != 1:
        _fail("E_RUNTIME_DISCOVERY")
    runtime = unique[0]
    if not _private_directory(runtime) or not _private_file(
        runtime / ".metadata_never_index"
    ):
        _fail("E_RUNTIME_CONTROL")
    return runtime


def _config() -> tuple[Mapping[str, Any], str]:
    config = _read_json(CONFIG)
    if (
        config.get("schema_version")
        != "childlens-alignment-bridge-remediation-preflight-v2.0.0"
        or config.get("status")
        != "FROZEN_BEFORE_REMEDIATION_SEGMENTATION_OR_ASR_COMPARISON"
        or config.get("scope", {}).get("locked_evaluation_allowed") is not False
        or config.get("scope", {}).get("simulator_generation_allowed") is not False
    ):
        _fail("E_PROTOCOL")
    return config, _sha256_file(CONFIG)


def _validate_instruments() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file() or _sha256_file(path) != expected:
            _fail("E_INSTRUMENT_HASH")
    for path in (WHISPER_CLI, EMBEDDING_PYTHON, FFMPEG, SANDBOX_EXEC):
        if not path.is_file() or not os.access(path, os.X_OK):
            _fail("E_INSTRUMENT_EXECUTABLE")


def freeze() -> Mapping[str, Any]:
    config, protocol_sha256 = _config()
    _validate_instruments()
    if (
        _sha256_file(V1_DEV) != V1_DEV_SHA256
        or _read_json(V1_FREEZE).get("protocol_sha256") != V1_PROTOCOL_SHA256
        or _read_json(V1_FREEZE).get("restricted_split_manifest_sha256")
        != V1_SPLIT_SHA256
    ):
        _fail("E_V1_BINDING")
    runtime = _discover_runtime()
    v1_private = (
        runtime
        / "provisional_calibration_v1/childlens_alignment_bridge_v1/"
        "restricted_split_manifest.json"
    )
    if not _private_file(v1_private) or _sha256_file(v1_private) != V1_SPLIT_SHA256:
        _fail("E_V1_BINDING")
    v1_split = _read_json(v1_private)
    development = [
        dict(row) for row in v1_split.get("items", []) if row.get("split") == "development"
    ]
    if len(development) != 8 or len({row["participant_key"] for row in development}) != 8:
        _fail("E_DEVELOPMENT_BINDING")
    private = {
        "schema_version": "childlens-alignment-bridge-remediation-development-v2.0.0",
        "protocol_sha256": protocol_sha256,
        "v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "v1_split_sha256": V1_SPLIT_SHA256,
        "development_count": 8,
        "locked_rows_copied": 0,
        "items": development,
    }
    private_path = runtime / PRIVATE_DEVELOPMENT
    _write_once(private_path, private, private=True)
    receipt = {
        "schema_version": "childlens-alignment-bridge-remediation-freeze-v2.0.0",
        "status": "FROZEN_BEFORE_SEGMENTATION",
        "protocol_sha256": protocol_sha256,
        "v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "v1_split_sha256": V1_SPLIT_SHA256,
        "v1_artifacts_modified": False,
        "restricted_development_manifest_sha256": _sha256_file(private_path),
        "development_participants": 8,
        "locked_rows_copied_or_evaluated": 0,
        "instrument_hashes_verified": True,
        "primary_asr_model_sha256": EXPECTED_HASHES[PRIMARY_MODEL],
        "sensitivity_asr_model_sha256": EXPECTED_HASHES[SENSITIVITY_MODEL],
        "vad_model_sha256": EXPECTED_HASHES[SILERO_MODEL],
        "embedding_model_sha256": EXPECTED_HASHES[
            EMBEDDING_MODEL / "model.safetensors"
        ],
        "restricted_values_exported": False,
    }
    _write_once(FREEZE_RECEIPT, receipt, private=False)
    return receipt


def _sandbox_reexec(command: str) -> Mapping[str, Any]:
    environment = dict(os.environ)
    environment["CHILDLENS_V2_NETWORK_DENIED"] = "1"
    process = subprocess.run(
        [
            str(SANDBOX_EXEC),
            "-p",
            SANDBOX_PROFILE,
            str(EMBEDDING_PYTHON),
            str(Path(__file__).resolve()),
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=14_400,
        env=environment,
    )
    if process.returncode != 0:
        _fail("E_RESTRICTED_WORKER")
    try:
        result = json.loads(process.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        _fail("E_RESTRICTED_WORKER")
    if result.get("status") != "ok":
        _fail("E_RESTRICTED_WORKER")
    return result


def _extract_audio(row: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output.parent, 0o700)
    duration = float(row["sample_span_end_seconds"]) - float(
        row["sample_span_start_seconds"]
    )
    process = subprocess.run(
        [
            str(FFMPEG),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{float(row['sample_span_start_seconds']):.6f}",
            "-i",
            str(row["media_path"]),
            "-t",
            f"{duration:.6f}",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=False,
        capture_output=True,
        timeout=600,
    )
    if process.returncode != 0 or not output.is_file():
        _fail("E_AUDIO_EXTRACTION")
    os.chmod(output, 0o600)


def _read_waveform(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getframerate() != 16_000
            or handle.getnchannels() != 1
            or handle.getsampwidth() != 2
        ):
            _fail("E_AUDIO_FORMAT")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _speech_annotations(row: Mapping[str, Any]) -> list[dict[str, float]]:
    document = _read_json(Path(row["annotation_path"]))
    offset = float(row.get("source_offset_seconds", 0.0))
    span_start = float(row["sample_span_start_seconds"])
    span_end = float(row["sample_span_end_seconds"])
    result = []
    for annotation in document.get("annotations", []):
        if annotation.get("eventId") not in SPEECH_EVENTS:
            continue
        try:
            start = float(annotation["startTime"]) - offset
            end = float(annotation["endTime"]) - offset
        except (KeyError, TypeError, ValueError):
            continue
        start = max(start, span_start)
        end = min(end, span_end)
        if end > start:
            result.append(
                {
                    "start_seconds": start - span_start,
                    "end_seconds": end - span_start,
                }
            )
    if not result:
        _fail("E_SPEECH_ANNOTATION")
    return result


def _accept_segments(
    segments: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    *,
    duration_seconds: float,
) -> list[dict[str, float]]:
    expanded = [
        {
            "start_seconds": max(0.0, float(row["start_seconds"]) - 0.25),
            "end_seconds": min(duration_seconds, float(row["end_seconds"]) + 0.25),
        }
        for row in references
    ]
    accepted = []
    for row in segments:
        duration = float(row["end_seconds"]) - float(row["start_seconds"])
        overlap = sum(interval_overlap_seconds(row, reference) for reference in expanded)
        if duration > 0 and min(overlap, duration) / duration >= 0.5:
            accepted.append(
                {
                    "start_seconds": float(row["start_seconds"]),
                    "end_seconds": float(row["end_seconds"]),
                }
            )
    return accepted


def _prepare_segments_restricted() -> Mapping[str, Any]:
    if os.environ.get("CHILDLENS_V2_NETWORK_DENIED") != "1":
        _fail("E_NETWORK_DENIAL")
    config, protocol_sha256 = _config()
    runtime = _discover_runtime()
    development_path = runtime / PRIVATE_DEVELOPMENT
    freeze_receipt = _read_json(FREEZE_RECEIPT)
    if (
        not _private_file(development_path)
        or _sha256_file(development_path)
        != freeze_receipt.get("restricted_development_manifest_sha256")
    ):
        _fail("E_FREEZE_BINDING")
    development = _read_json(development_path)
    private_root = runtime / PRIVATE_RELATIVE
    audio_root = private_root / "audio"
    items = []
    for row in development["items"]:
        item_key = str(row["item_key"])
        audio_path = audio_root / f"{item_key}.wav"
        if not audio_path.exists():
            _extract_audio(row, audio_path)
        elif not _private_file(audio_path):
            _fail("E_AUDIO_CONTROL")
        waveform = _read_waveform(audio_path)
        duration = len(waveform) / 16_000
        references = _speech_annotations(row)
        base_raw = silero_speech_segments(waveform, str(SILERO_MODEL))
        base = _accept_segments(base_raw, references, duration_seconds=duration)
        padded_waveform = np.pad(waveform, (4_000, 4_000))
        perturbed_raw = silero_speech_segments(padded_waveform, str(SILERO_MODEL))
        shifted = []
        for segment in perturbed_raw:
            start = max(0.0, float(segment["start_seconds"]) - 0.25)
            end = min(duration, float(segment["end_seconds"]) - 0.25)
            if end > start:
                shifted.append({"start_seconds": start, "end_seconds": end})
        perturbed = _accept_segments(shifted, references, duration_seconds=duration)
        frozen_segments = []
        for index, segment in enumerate(base):
            frozen_segments.append(
                {
                    "segment_key": hashlib.sha256(
                        f"{protocol_sha256}|{item_key}|{index}".encode()
                    ).hexdigest(),
                    "start_seconds": segment["start_seconds"],
                    "end_seconds": segment["end_seconds"],
                    "expanded_start_seconds": max(
                        0.0, segment["start_seconds"] - 0.25
                    ),
                    "expanded_end_seconds": min(
                        duration, segment["end_seconds"] + 0.25
                    ),
                }
            )
        items.append(
            {
                "participant_key": row["participant_key"],
                "item_key": item_key,
                "audio_path": str(audio_path),
                "audio_sha256": _sha256_file(audio_path),
                "audio_duration_seconds": duration,
                "reference_speech_intervals": references,
                "base_vad_segments": frozen_segments,
                "perturbed_vad_segments": perturbed,
            }
        )
    segment_manifest = {
        "schema_version": "childlens-alignment-bridge-remediation-segments-v2.0.0",
        "protocol_sha256": protocol_sha256,
        "restricted_development_manifest_sha256": _sha256_file(development_path),
        "locked_rows_loaded": 0,
        "asr_run_before_segment_freeze": False,
        "items": items,
    }
    segment_path = runtime / PRIVATE_SEGMENTS
    _write_once(segment_path, segment_manifest, private=True)
    count = sum(len(row["base_vad_segments"]) for row in items)
    seconds = sum(
        segment["end_seconds"] - segment["start_seconds"]
        for row in items
        for segment in row["base_vad_segments"]
    )
    receipt = {
        "schema_version": "childlens-alignment-bridge-remediation-segment-freeze-v2.0.0",
        "status": "SHARED_SEGMENTS_FROZEN_BEFORE_ASR",
        "protocol_sha256": protocol_sha256,
        "restricted_segment_manifest_sha256": _sha256_file(segment_path),
        "development_participants": len(items),
        "locked_rows_loaded": 0,
        "accepted_segment_count": count,
        "accepted_speech_seconds": round(seconds, 3),
        "same_segments_for_both_asr_systems": True,
        "network_denial_backend": "MACOS_SANDBOX_DENY_NETWORK",
        "restricted_values_exported": False,
    }
    _write_once(SEGMENT_RECEIPT, receipt, private=False)
    return {"status": "ok", "state": receipt["status"]}


def _write_wave(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    clipped = np.clip(waveform, -1.0, 0.9999695)
    pcm = (clipped * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())
    os.chmod(path, 0o600)


def _concatenate_segments(
    waveform: np.ndarray,
    segments: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    expanded: bool,
) -> list[dict[str, Any]]:
    silence = np.zeros(8_000, dtype=np.float32)
    chunks = []
    mapping = []
    cursor = 0
    for segment in segments:
        start_key = "expanded_start_seconds" if expanded else "start_seconds"
        end_key = "expanded_end_seconds" if expanded else "end_seconds"
        start_sample = round(float(segment[start_key]) * 16_000)
        end_sample = round(float(segment[end_key]) * 16_000)
        chunk = waveform[start_sample:end_sample]
        concat_start = cursor / 16_000
        chunks.append(chunk)
        cursor += len(chunk)
        concat_end = cursor / 16_000
        mapping.append(
            {
                "segment_key": segment["segment_key"],
                "concat_start_seconds": concat_start,
                "concat_end_seconds": concat_end,
            }
        )
        chunks.append(silence)
        cursor += len(silence)
    combined = np.concatenate(chunks) if chunks else np.zeros(16_000, dtype=np.float32)
    _write_wave(output, combined)
    return mapping


def _run_whisper(
    *,
    model: Path,
    dtw: str,
    audio: Path,
    prefix: Path,
) -> Mapping[str, Any]:
    process = subprocess.run(
        [
            str(WHISPER_CLI),
            "-m",
            str(model),
            "-f",
            str(audio),
            "-l",
            "auto",
            "-bo",
            "5",
            "-bs",
            "5",
            "-tp",
            "0",
            "-dtw",
            dtw,
            "-sow",
            "-ojf",
            "-of",
            str(prefix),
            "-np",
        ],
        check=False,
        capture_output=True,
        timeout=7_200,
    )
    output = prefix.with_suffix(".json")
    if process.returncode != 0 or not output.is_file():
        _fail("E_ASR")
    os.chmod(output, 0o600)
    return _read_json(output)


def _language(document: Mapping[str, Any]) -> str:
    value = normalize_text(str(document.get("result", {}).get("language", "")))
    return {"german": "de", "deu": "de", "english": "en", "eng": "en"}.get(
        value, value
    )


def _map_transcript(
    document: Mapping[str, Any],
    mapping: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    texts: dict[str, list[str]] = {str(row["segment_key"]): [] for row in mapping}
    for row in document.get("transcription", []):
        offsets = row.get("offsets")
        text = normalize_text(str(row.get("text", "")))
        if not isinstance(offsets, Mapping) or not text:
            continue
        try:
            start = float(offsets["from"]) / 1000.0
            end = float(offsets["to"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        candidates = []
        for target in mapping:
            overlap = max(
                0.0,
                min(end, float(target["concat_end_seconds"]))
                - max(start, float(target["concat_start_seconds"])),
            )
            if overlap > 0:
                candidates.append((overlap, str(target["segment_key"])))
        if candidates:
            _, key = max(candidates)
            texts[key].append(text)
    return {key: " ".join(values) for key, values in texts.items()}


def _embed_cosines(
    rows: Sequence[Mapping[str, str]],
    *,
    private_root: Path,
) -> dict[str, float]:
    input_path = private_root / "embedding_input.json"
    output_path = private_root / "embedding_cosines.json"
    _write_once(input_path, {"rows": list(rows)}, private=True)
    process = subprocess.run(
        [
            str(EMBEDDING_PYTHON),
            str(Path(__file__).resolve()),
            "_embed-worker",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        timeout=3_600,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CHILDLENS_V2_NETWORK_DENIED": "1",
        },
    )
    if process.returncode != 0 or not _private_file(output_path):
        _fail("E_EMBEDDING")
    result = _read_json(output_path)
    return {str(row["key"]): float(row["cosine"]) for row in result["rows"]}


def _embed_worker(input_path: Path, output_path: Path) -> None:
    if os.environ.get("CHILDLENS_V2_NETWORK_DENIED") != "1":
        _fail("E_NETWORK_DENIAL")
    from transformers import AutoModel, AutoTokenizer
    import torch

    document = _read_json(input_path)
    rows = document.get("rows", [])
    texts = []
    for row in rows:
        texts.extend([str(row["left"]), str(row["right"])])
    tokenizer = AutoTokenizer.from_pretrained(
        str(EMBEDDING_MODEL), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(EMBEDDING_MODEL), local_files_only=True
    ).eval()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    output_rows = []
    for index, row in enumerate(rows):
        cosine = float(embeddings[index * 2] @ embeddings[index * 2 + 1])
        output_rows.append({"key": row["key"], "cosine": cosine})
    _write_once(output_path, {"rows": output_rows}, private=True)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def _bootstrap(values: Sequence[float], seed_offset: int) -> tuple[float, float]:
    return participant_bootstrap_interval(
        values,
        confidence=0.9,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED + seed_offset,
    )


def _timing_item(row: Mapping[str, Any]) -> dict[str, Any]:
    base = row["base_vad_segments"]
    perturbed = row["perturbed_vad_segments"]
    references = row["reference_speech_intervals"]
    precision, recall = interval_set_precision_recall(base, references)
    boundary_f1 = matched_boundary_f1(base, perturbed, tolerance_seconds=0.5)
    base_seconds = sum(
        float(segment["end_seconds"]) - float(segment["start_seconds"])
        for segment in base
    )
    perturbed_seconds = sum(
        float(segment["end_seconds"]) - float(segment["start_seconds"])
        for segment in perturbed
    )
    duration_changes = []
    unmatched = set(range(len(perturbed)))
    for segment in base:
        candidates = []
        for index in unmatched:
            other = perturbed[index]
            error = abs(float(segment["start_seconds"]) - float(other["start_seconds"]))
            error += abs(float(segment["end_seconds"]) - float(other["end_seconds"]))
            if error <= 1.0:
                candidates.append((error, index))
        if candidates:
            _, index = min(candidates)
            unmatched.remove(index)
            other = perturbed[index]
            duration_changes.append(
                abs(
                    (float(segment["end_seconds"]) - float(segment["start_seconds"]))
                    - (
                        float(other["end_seconds"])
                        - float(other["start_seconds"])
                    )
                )
            )
    return {
        "segment_count": len(base),
        "speech_seconds": base_seconds,
        "speech_coverage": base_seconds / float(row["audio_duration_seconds"]),
        "annotation_precision": precision,
        "annotation_recall": recall,
        "boundary_f1": boundary_f1,
        "median_duration_change": _median(duration_changes),
        "coverage_ratio": perturbed_seconds / base_seconds if base_seconds else 0.0,
    }


def _evaluate_restricted() -> Mapping[str, Any]:
    if os.environ.get("CHILDLENS_V2_NETWORK_DENIED") != "1":
        _fail("E_NETWORK_DENIAL")
    config, protocol_sha256 = _config()
    runtime = _discover_runtime()
    segment_path = runtime / PRIVATE_SEGMENTS
    segment_receipt = _read_json(SEGMENT_RECEIPT)
    if (
        not _private_file(segment_path)
        or _sha256_file(segment_path)
        != segment_receipt.get("restricted_segment_manifest_sha256")
    ):
        _fail("E_SEGMENT_BINDING")
    manifest = _read_json(segment_path)
    if len(manifest["items"]) != 8 or manifest.get("locked_rows_loaded") != 0:
        _fail("E_DEVELOPMENT_ONLY")
    private_root = runtime / PRIVATE_RELATIVE
    asr_root = private_root / "asr"
    per_item = []
    embedding_pairs = []
    segment_records = []
    for item in manifest["items"]:
        waveform = _read_waveform(Path(item["audio_path"]))
        item_key = str(item["item_key"])
        base_audio = asr_root / f"{item_key}-base.wav"
        expanded_audio = asr_root / f"{item_key}-expanded.wav"
        base_mapping = _concatenate_segments(
            waveform, item["base_vad_segments"], base_audio, expanded=False
        )
        expanded_mapping = _concatenate_segments(
            waveform, item["base_vad_segments"], expanded_audio, expanded=True
        )
        documents = {}
        for model_name, model, dtw in (
            ("primary", PRIMARY_MODEL, "large.v3"),
            ("sensitivity", SENSITIVITY_MODEL, "large.v3.turbo"),
        ):
            for condition, audio, mapping in (
                ("base", base_audio, base_mapping),
                ("expanded", expanded_audio, expanded_mapping),
            ):
                prefix = asr_root / f"{item_key}-{model_name}-{condition}"
                output = prefix.with_suffix(".json")
                if output.exists():
                    document = _read_json(output)
                else:
                    document = _run_whisper(
                        model=model, dtw=dtw, audio=audio, prefix=prefix
                    )
                documents[(model_name, condition)] = {
                    "language": _language(document),
                    "text": _map_transcript(document, mapping),
                }
        for segment in item["base_vad_segments"]:
            key = str(segment["segment_key"])
            record = {
                "participant_key": item["participant_key"],
                "segment_key": key,
                "primary_base": documents[("primary", "base")]["text"].get(key, ""),
                "sensitivity_base": documents[("sensitivity", "base")]["text"].get(
                    key, ""
                ),
                "primary_expanded": documents[("primary", "expanded")]["text"].get(
                    key, ""
                ),
                "sensitivity_expanded": documents[
                    ("sensitivity", "expanded")
                ]["text"].get(key, ""),
            }
            segment_records.append(record)
            for label, left, right in (
                ("matched", record["primary_base"], record["sensitivity_base"]),
                ("primary_self", record["primary_base"], record["primary_expanded"]),
                (
                    "sensitivity_self",
                    record["sensitivity_base"],
                    record["sensitivity_expanded"],
                ),
            ):
                if left and right:
                    embedding_pairs.append(
                        {
                            "key": f"{key}:{label}",
                            "left": left,
                            "right": right,
                        }
                    )
        timing = _timing_item(item)
        per_item.append(
            {
                "participant_key": item["participant_key"],
                "primary_language": documents[("primary", "base")]["language"],
                "sensitivity_language": documents[("sensitivity", "base")]["language"],
                "timing": timing,
            }
        )
    cosines = _embed_cosines(embedding_pairs, private_root=private_root)
    by_participant: dict[str, list[Mapping[str, Any]]] = {}
    for record in segment_records:
        by_participant.setdefault(record["participant_key"], []).append(record)
    for item in per_item:
        records = by_participant.get(item["participant_key"], [])
        matched_char = [
            character_similarity(row["primary_base"], row["sensitivity_base"])
            for row in records
            if row["primary_base"] and row["sensitivity_base"]
        ]
        primary_self_char = [
            character_similarity(row["primary_base"], row["primary_expanded"])
            for row in records
            if row["primary_base"] and row["primary_expanded"]
        ]
        sensitivity_self_char = [
            character_similarity(row["sensitivity_base"], row["sensitivity_expanded"])
            for row in records
            if row["sensitivity_base"] and row["sensitivity_expanded"]
        ]
        item["transcript"] = {
            "frozen_segment_count": len(records),
            "primary_nonempty_count": sum(bool(row["primary_base"]) for row in records),
            "sensitivity_nonempty_count": sum(
                bool(row["sensitivity_base"]) for row in records
            ),
            "both_nonempty_count": len(matched_char),
            "matched_char_median": _median(matched_char),
            "matched_embedding_median": _median(
                [
                    cosines[f"{row['segment_key']}:matched"]
                    for row in records
                    if f"{row['segment_key']}:matched" in cosines
                ]
            ),
            "primary_self_char_median": _median(primary_self_char),
            "sensitivity_self_char_median": _median(sensitivity_self_char),
            "primary_self_embedding_median": _median(
                [
                    cosines[f"{row['segment_key']}:primary_self"]
                    for row in records
                    if f"{row['segment_key']}:primary_self" in cosines
                ]
            ),
            "sensitivity_self_embedding_median": _median(
                [
                    cosines[f"{row['segment_key']}:sensitivity_self"]
                    for row in records
                    if f"{row['segment_key']}:sensitivity_self" in cosines
                ]
            ),
        }

    timing_values = {
        key: [float(item["timing"][key]) for item in per_item]
        for key in (
            "annotation_precision",
            "annotation_recall",
            "boundary_f1",
            "median_duration_change",
            "coverage_ratio",
        )
    }
    transcript_values = {
        key: [float(item["transcript"][key]) for item in per_item]
        for key in (
            "matched_char_median",
            "matched_embedding_median",
            "primary_self_char_median",
            "sensitivity_self_char_median",
            "primary_self_embedding_median",
            "sensitivity_self_embedding_median",
        )
    }
    total_segments = sum(item["timing"]["segment_count"] for item in per_item)
    total_speech = sum(item["timing"]["speech_seconds"] for item in per_item)
    primary_nonempty_items = sum(
        item["transcript"]["primary_nonempty_count"] > 0 for item in per_item
    )
    sensitivity_nonempty_items = sum(
        item["transcript"]["sensitivity_nonempty_count"] > 0 for item in per_item
    )
    primary_usable = sum(
        item["transcript"]["primary_nonempty_count"] for item in per_item
    )
    frozen_segments = sum(
        item["transcript"]["frozen_segment_count"] for item in per_item
    )
    primary_german = sum(item["primary_language"] == "de" for item in per_item) / 8
    language_agreement = sum(
        item["primary_language"] == item["sensitivity_language"] for item in per_item
    ) / 8
    gates = config["gates"]
    g1 = gates["G1_signal_and_vad"]
    g2 = gates["G2_matched_asr"]
    g3 = gates["G3_timing_shift_robustness"]
    boundary_interval = _bootstrap(timing_values["boundary_f1"], 1)
    char_interval = _bootstrap(transcript_values["matched_char_median"], 2)
    embedding_interval = _bootstrap(transcript_values["matched_embedding_median"], 3)
    coverage_ratio_median = _median(timing_values["coverage_ratio"])
    checks = {
        "G1_nonempty_participants": (
            sum(item["timing"]["segment_count"] > 0 for item in per_item) / 8
            >= g1["development_participant_nonempty_segment_fraction_min"]
        ),
        "G1_segment_count": total_segments >= g1["accepted_segments_min"],
        "G1_speech_seconds": total_speech >= g1["accepted_speech_seconds_min"],
        "G1_annotation_precision": (
            _median(timing_values["annotation_precision"])
            >= g1["participant_median_annotation_overlap_precision_min"]
        ),
        "G1_annotation_recall": (
            _median(timing_values["annotation_recall"])
            >= g1["participant_median_annotation_overlap_recall_min"]
        ),
        "G1_boundary_median": (
            _median(timing_values["boundary_f1"])
            >= g1["participant_median_vad_self_consistency_boundary_f1_min"]
        ),
        "G1_boundary_lower": (
            boundary_interval[0]
            >= g1["participant_bootstrap_boundary_f1_90pct_lower_min"]
        ),
        "G1_duration_change": (
            _median(timing_values["median_duration_change"])
            <= g1["participant_median_absolute_duration_change_seconds_max"]
        ),
        "G1_coverage_ratio": (
            g1["participant_median_perturbed_to_base_speech_coverage_ratio_range"][0]
            <= coverage_ratio_median
            <= g1[
                "participant_median_perturbed_to_base_speech_coverage_ratio_range"
            ][1]
        ),
        "G2_primary_nonempty": primary_nonempty_items / 8
        >= g2["primary_nonempty_item_fraction_min"],
        "G2_sensitivity_nonempty": sensitivity_nonempty_items / 8
        >= g2["sensitivity_nonempty_item_fraction_min"],
        "G2_usable_count": primary_usable >= g2["primary_usable_utterances_min"],
        "G2_usable_fraction": (
            primary_usable / frozen_segments
            if frozen_segments
            else 0.0
        )
        >= g2["usable_utterance_fraction_of_frozen_segments_min"],
        "G2_primary_german": primary_german
        >= g2["primary_german_item_fraction_min"],
        "G2_language_agreement": language_agreement
        >= g2["primary_sensitivity_language_agreement_min"],
        "G2_character_median": (
            _median(transcript_values["matched_char_median"])
            >= g2["participant_median_normalized_character_similarity_min"]
        ),
        "G2_character_lower": char_interval[0]
        >= g2["participant_cluster_bootstrap_character_90pct_lower_min"],
        "G2_embedding_median": (
            _median(transcript_values["matched_embedding_median"])
            >= g2["participant_median_embedding_cosine_min"]
        ),
        "G2_embedding_lower": embedding_interval[0]
        >= g2["participant_cluster_bootstrap_embedding_90pct_lower_min"],
        "G3_primary_character": (
            _median(transcript_values["primary_self_char_median"])
            >= g3["participant_median_primary_character_self_similarity_min"]
        ),
        "G3_sensitivity_character": (
            _median(transcript_values["sensitivity_self_char_median"])
            >= g3["participant_median_sensitivity_character_self_similarity_min"]
        ),
        "G3_primary_embedding": (
            _median(transcript_values["primary_self_embedding_median"])
            >= g3["participant_median_primary_embedding_self_similarity_min"]
        ),
        "G3_sensitivity_embedding": (
            _median(transcript_values["sensitivity_self_embedding_median"])
            >= g3["participant_median_sensitivity_embedding_self_similarity_min"]
        ),
    }
    gate_g1 = all(value for key, value in checks.items() if key.startswith("G1_"))
    gate_g2 = all(value for key, value in checks.items() if key.startswith("G2_"))
    gate_g3 = all(value for key, value in checks.items() if key.startswith("G3_"))
    matched_char = _median(transcript_values["matched_char_median"])
    v1_char = float(_read_json(V1_DEV)["language_model_model_diagnostic"][
        "character_similarity_participant_median"
    ])
    improvement = matched_char - v1_char
    boundary_major = improvement >= 0.20 and checks["G2_character_median"]
    causes = []
    yield_checks = (
        checks["G1_nonempty_participants"],
        checks["G1_segment_count"],
        checks["G1_speech_seconds"],
        checks["G1_annotation_precision"],
        checks["G1_annotation_recall"],
    )
    if not all(yield_checks):
        causes.append("INSUFFICIENT_OR_UNSUITABLE_CHILDLENS_SPEECH_SIGNAL")
    if all(yield_checks) and not all(
        (
            checks["G1_boundary_median"],
            checks["G1_boundary_lower"],
            checks["G1_duration_change"],
            checks["G1_coverage_ratio"],
        )
    ):
        causes.append("INADEQUATE_VAD_BOUNDARY_ESTIMATION")
    if gate_g1 and not (gate_g2 and gate_g3):
        causes.append("INADEQUATE_GERMAN_ASR_STABILITY")
    if not causes and gate_g1 and gate_g2 and gate_g3:
        causes.append("STABILITY_PASS_ACCURACY_STILL_REQUIRES_GERMAN_HUMAN")
    recommend_locked = gate_g1 and gate_g2 and gate_g3
    private_result = {
        "schema_version": "childlens-alignment-bridge-remediation-result-v2.0.0",
        "protocol_sha256": protocol_sha256,
        "restricted_segment_manifest_sha256": _sha256_file(segment_path),
        "development_only": True,
        "locked_rows_loaded_or_evaluated": 0,
        "per_item": per_item,
        "segment_records": segment_records,
        "checks": checks,
        "causes": causes,
        "recommend_separately_authorized_locked_evaluation": recommend_locked,
    }
    private_path = runtime / PRIVATE_RESULT
    _write_once(private_path, private_result, private=True)
    public = {
        "schema_version": "childlens-alignment-bridge-remediation-report-v2.0.0",
        "status": (
            "DEVELOPMENT_REMEDIATION_PASS_RECOMMEND_SEPARATE_LOCKED_AUTHORIZATION"
            if recommend_locked
            else "DEVELOPMENT_REMEDIATION_STOP"
        ),
        "protocol_sha256": protocol_sha256,
        "restricted_segment_manifest_sha256": _sha256_file(segment_path),
        "restricted_result_sha256": _sha256_file(private_path),
        "development_participants": 8,
        "locked_rows_loaded_or_evaluated": 0,
        "timing": {
            "accepted_segment_count": total_segments,
            "accepted_speech_seconds": round(total_speech, 3),
            "participant_median_annotation_precision": round(
                _median(timing_values["annotation_precision"]), 3
            ),
            "participant_median_annotation_recall": round(
                _median(timing_values["annotation_recall"]), 3
            ),
            "participant_median_vad_self_consistency_boundary_f1": round(
                _median(timing_values["boundary_f1"]), 3
            ),
            "boundary_f1_bootstrap_90pct": [
                round(boundary_interval[0], 3),
                round(boundary_interval[1], 3),
            ],
            "participant_median_absolute_duration_change_seconds": round(
                _median(timing_values["median_duration_change"]), 3
            ),
            "participant_median_perturbed_to_base_coverage_ratio": round(
                coverage_ratio_median, 3
            ),
        },
        "matched_transcript": {
            "primary_nonempty_item_fraction": round(primary_nonempty_items / 8, 3),
            "sensitivity_nonempty_item_fraction": round(
                sensitivity_nonempty_items / 8, 3
            ),
            "primary_usable_segment_count": primary_usable,
            "primary_usable_segment_fraction": round(
                primary_usable / frozen_segments if frozen_segments else 0.0, 3
            ),
            "primary_german_item_fraction": round(primary_german, 3),
            "language_id_agreement_fraction": round(language_agreement, 3),
            "participant_median_character_similarity": round(matched_char, 3),
            "character_similarity_bootstrap_90pct": [
                round(char_interval[0], 3),
                round(char_interval[1], 3),
            ],
            "participant_median_embedding_cosine": round(
                _median(transcript_values["matched_embedding_median"]), 3
            ),
            "embedding_cosine_bootstrap_90pct": [
                round(embedding_interval[0], 3),
                round(embedding_interval[1], 3),
            ],
        },
        "timing_shift_robustness": {
            "primary_character_self_similarity_median": round(
                _median(transcript_values["primary_self_char_median"]), 3
            ),
            "sensitivity_character_self_similarity_median": round(
                _median(transcript_values["sensitivity_self_char_median"]), 3
            ),
            "primary_embedding_self_similarity_median": round(
                _median(transcript_values["primary_self_embedding_median"]), 3
            ),
            "sensitivity_embedding_self_similarity_median": round(
                _median(transcript_values["sensitivity_self_embedding_median"]), 3
            ),
        },
        "boundary_attribution": {
            "v1_independent_boundary_character_similarity_median": round(v1_char, 3),
            "v2_matched_segment_character_similarity_median": round(matched_char, 3),
            "absolute_improvement": round(improvement, 3),
            "boundary_mismatch_major_contributor": boundary_major,
        },
        "checks": checks,
        "gate_G1_signal_and_vad_pass": gate_g1,
        "gate_G2_matched_asr_pass": gate_g2,
        "gate_G3_timing_shift_robustness_pass": gate_g3,
        "diagnostic_causes": causes,
        "human_validation": False,
        "ground_truth": False,
        "recommend_separately_authorized_locked_evaluation": recommend_locked,
        "locked_evaluation_run": False,
        "alignment_scoring_run": False,
        "simulator_or_cue_training_run": False,
        "restricted_values_exported": False,
    }
    _write_once(REPORT_JSON, public, private=False)
    return {"status": "ok", "state": public["status"]}


def locked_status() -> Mapping[str, Any]:
    if not REPORT_JSON.is_file():
        return {
            "locked_evaluation_authorized": False,
            "reason": "REMEDIATION_NOT_COMPLETE",
        }
    report = _read_json(REPORT_JSON)
    return {
        "locked_evaluation_authorized": False,
        "recommendation_for_separate_authorization": bool(
            report.get("recommend_separately_authorized_locked_evaluation")
        ),
        "reason": "SEPARATE_USER_AUTHORIZATION_REQUIRED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "freeze",
            "prepare-segments",
            "_prepare-segments-restricted",
            "evaluate",
            "_evaluate-restricted",
            "_embed-worker",
            "locked-status",
        ),
    )
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    old_umask = os.umask(0o077)
    try:
        if args.command == "freeze":
            result = freeze()
        elif args.command == "prepare-segments":
            result = _sandbox_reexec("_prepare-segments-restricted")
        elif args.command == "_prepare-segments-restricted":
            result = _prepare_segments_restricted()
        elif args.command == "evaluate":
            result = _sandbox_reexec("_evaluate-restricted")
        elif args.command == "_evaluate-restricted":
            result = _evaluate_restricted()
        elif args.command == "_embed-worker":
            if not args.input or not args.output:
                _fail("E_EMBEDDING_ARGS")
            _embed_worker(Path(args.input), Path(args.output))
            result = {"status": "ok"}
        else:
            result = locked_status()
        print(json.dumps(result, sort_keys=True))
        return 0
    except BridgeError as exc:
        print(json.dumps({"status": "error", "error_code": str(exc)}, sort_keys=True))
        return 2
    except Exception:
        print(json.dumps({"status": "error", "error_code": "E_INTERNAL"}, sort_keys=True))
        return 2
    finally:
        os.umask(old_umask)


if __name__ == "__main__":
    raise SystemExit(main())
