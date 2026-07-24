#!/usr/bin/env python3
"""Generate, resume, and validate the bounded ChildLens v5 calibration."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/childlens_alignment_bridge_v5.json"
PACKAGE = ROOT / "babyworld_lite/childlens_alignment_bridge_v5/preflight.py"
PROTOCOL = ROOT / "docs/childlens_alignment_bridge_v5_protocol.md"
PUBLIC_ROOT = ROOT / "output/childlens_alignment_bridge_v5"
FREEZE_RECEIPT = PUBLIC_ROOT / "stage0_freeze_and_stop_receipt.json"
POSITIVE_RECEIPT = PUBLIC_ROOT / "synthetic_positive_control.json"
DECISION_REPORT = PUBLIC_ROOT / "development_decision.json"
DECISION_MARKDOWN = PUBLIC_ROOT / "development_decision.md"
VALIDATION_RECEIPT = PUBLIC_ROOT / "validation_receipt.json"
ADMIN_RECEIPT = PUBLIC_ROOT / "administrative_correction_receipt.json"
CLEAN_STAGE0_RECEIPT = PUBLIC_ROOT / "clean_stage0_freeze_receipt.json"
CLEAN_POSITIVE_RECEIPT = PUBLIC_ROOT / "clean_synthetic_positive_control.json"
CLEAN_ACQUISITION_RECEIPT = PUBLIC_ROOT / "clean_acquisition_receipt.json"
CLEAN_MODEL_RECEIPT = PUBLIC_ROOT / "clean_model_hash_receipt.json"
CLEAN_CALIBRATION_REPORT = PUBLIC_ROOT / "clean_calibration_summary.json"
CLEAN_LAG_REPORT = PUBLIC_ROOT / "clean_lag_response_report.json"
CLEAN_DECISION_REPORT = PUBLIC_ROOT / "clean_development_decision.json"
CLEAN_DECISION_MARKDOWN = PUBLIC_ROOT / "clean_development_decision.md"
CLEAN_VALIDATION_RECEIPT = PUBLIC_ROOT / "clean_validation_receipt.json"
CLEAN_RUN_ID = "childlens-v5-clean-20260723-a"
PRIVATE_ADMIN_RELATIVE = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_v5/administrative"
)
PRIVATE_INPUT = PRIVATE_ADMIN_RELATIVE / "development_only_scientific_input.json"
PRIVATE_ATTESTATION = PRIVATE_ADMIN_RELATIVE / "development_only_attestation.json"
PRIVATE_CLEAN_RELATIVE = Path(
    "provisional_calibration_v1/childlens_alignment_bridge_v5/clean_run"
)
PRIVATE_STAGE0_SELECTION = (
    PRIVATE_CLEAN_RELATIVE / "restricted_stage0_selection.json"
)
PRIVATE_ACQUISITION_CHECKPOINT = (
    PRIVATE_CLEAN_RELATIVE / "restricted_acquisition_checkpoint.json"
)
PRIVATE_ACQUISITION_FAILURE = (
    PRIVATE_CLEAN_RELATIVE / "restricted_last_acquisition_failure.json"
)
PRIVATE_CLIPS = PRIVATE_CLEAN_RELATIVE / "clips"
PRIVATE_TRANSIENT = PRIVATE_CLEAN_RELATIVE / "transient"
PRIVATE_FEATURE_SHARDS = PRIVATE_CLEAN_RELATIVE / "feature_shards"
PRIVATE_FEATURES = PRIVATE_CLEAN_RELATIVE / "restricted_frontend_features.npz"
PRIVATE_FEATURE_MANIFEST = (
    PRIVATE_CLEAN_RELATIVE / "restricted_feature_manifest.json"
)
PRIVATE_WEIGHTS = PRIVATE_CLEAN_RELATIVE / "restricted_projection_weights.npz"
PRIVATE_RESULT = PRIVATE_CLEAN_RELATIVE / "restricted_development_result.json"
MODEL_ROOT = ROOT / ".external/models/childlens_bridge_v5"
AUDIO_MODEL = MODEL_ROOT / "wav2vec2-xls-r-300m"
VISION_MODEL = MODEL_ROOT / "dinov2-small"
INSTRUMENT_PYTHON = (
    Path.home()
    / "Library/Application Support/ChildLens Instruments/"
    "provisional-calibration-v1/qwen-asr-venv/bin/python"
)
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)"
FROZEN_CONFIG_SHA256 = (
    "b048ca4f4950eaf37d8e751a88ee358d5eabeb83b5187302502d9e08d62b130d"
)
ORIGINAL_INCIDENT_RECEIPT_SHA256 = (
    "10d46eda4dc08e5cb98b0e9752fe9e4d94cac431549c6eb2bf6505b1a3cbc71d"
)
ORIGINAL_RUNNER_SHA256 = (
    "3cc07dddd8aa6e762899488955026c508e3c4f3d8b80a87c68f632f4deb6914a"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SPEECH_RE = re.compile(r"speech|talk|speak|vocal", re.IGNORECASE)
ENRICHED_ACTIVITY_RE = re.compile(
    r"book|read|object.?play|toy|draw|craft|pretend|music|instrument",
    re.IGNORECASE,
)
EXPECTED_MODEL_HASHES = {
    AUDIO_MODEL / "pytorch_model.bin": (
        "d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0"
    ),
    AUDIO_MODEL / "config.json": (
        "0bffa0d0e98153e883b828d86491f3c6062cb563dc9d7a9cfd1790da30c286ac"
    ),
    AUDIO_MODEL / "preprocessor_config.json": (
        "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335"
    ),
    VISION_MODEL / "model.safetensors": (
        "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
    ),
    VISION_MODEL / "config.json": (
        "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
    ),
    VISION_MODEL / "preprocessor_config.json": (
        "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
    ),
}

PRIOR_SCOPES = {
    "v1": [
        "configs/childlens_alignment_bridge_v1.json",
        "babyworld_lite/childlens_alignment_bridge_v1",
        "output/childlens_alignment_bridge_v1",
        "scripts/run_childlens_alignment_bridge_preflight_v1.py",
    ],
    "v2": [
        "configs/childlens_alignment_bridge_remediation_v2.json",
        "babyworld_lite/childlens_alignment_bridge_v2",
        "output/childlens_alignment_bridge_remediation_v2",
        "scripts/run_childlens_alignment_bridge_remediation_v2.py",
    ],
    "v3": [
        "configs/childlens_alignment_bridge_expansion_v3.json",
        "babyworld_lite/childlens_alignment_bridge_v3",
        "output/childlens_alignment_bridge_expansion_v3",
        "scripts/run_childlens_alignment_bridge_expansion_v3.py",
        "scripts/acquire_childlens_alignment_bridge_expansion_v3.py",
        "scripts/measure_childlens_alignment_bridge_expansion_v3.py",
    ],
    "v4": [
        "configs/childlens_alignment_bridge_v4.json",
        "babyworld_lite/childlens_alignment_bridge_v4",
        "output/childlens_alignment_bridge_v4",
        "scripts/run_childlens_alignment_bridge_v4.py",
    ],
}
EXPECTED_PRIOR_TREES = {
    "v1": "4f668c666636e47e72a3c2162cb30590f89077f0e4e6ef8e69b149f4966df679",
    "v2": "30df10121591465cd4c357ead6b4aab8c2dff30b2a292106faa277661c621e45",
    "v3": "29718e4741bf31f27e5001e6a6398423745b1d5b8688032b3e9140a3f5ef90fa",
    "v4": "d9b62ecd5c747a22779ea0624bca7c5eca707da86c1727d6abce812ecbc673d2",
}
EXPECTED_DEVELOPMENT_MANIFEST_HASHES = {
    "immutable_original_eight": (
        "9cb87b853eb43636d6baf09c281725eb98255b6cf73439025b47d627e84da5a8"
    ),
    "immutable_v3_expansion_ten": (
        "028efa424ed4dc2fe511a0b723b38f4feccba7f4de7db055934218dff1fe705d"
    ),
}

sys.path.insert(0, str(ROOT))
from babyworld_lite.childlens_alignment_bridge_v5.preflight import (  # noqa: E402
    BridgeV5Error,
    canonical_bytes,
    common_duration_target,
    deterministic_folds,
    digest,
    keyed_hash,
    lag_grid,
    public_guard,
    run_embedding_positive_control,
    score_temporal_projectors,
    terminal_decision,
    train_temporal_projectors,
    trainable_parameter_count,
)


def _sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


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


def _secure_generated_file(path: Path) -> bool:
    """Apply the quarantine file mode before enforcing the private-file guard."""

    if not path.is_file() or path.is_symlink():
        return False
    os.chmod(path, 0o600)
    return _private_file(path)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _write_private(path: Path, value: Any) -> None:
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    pending = path.parent / f".pending-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(
            pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise BridgeV5Error("E_PRIVATE_WRITE") from exc
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _tree_digest(entries: Sequence[str]) -> str:
    rows: list[dict[str, str]] = []
    for entry in entries:
        path = ROOT / entry
        files = (
            [path]
            if path.is_file()
            else sorted(
                value
                for value in path.rglob("*")
                if value.is_file()
                and "__pycache__" not in value.parts
                and value.suffix != ".pyc"
            )
        )
        if not files:
            raise BridgeV5Error("E_PRIOR_TREE")
        for value in files:
            rows.append(
                {
                    "path": value.relative_to(ROOT).as_posix(),
                    "sha256": _sha256_file(value),
                }
            )
    return digest(rows)


def _validate_prior_trees() -> dict[str, str]:
    observed = {
        version: _tree_digest(entries) for version, entries in PRIOR_SCOPES.items()
    }
    if observed != EXPECTED_PRIOR_TREES:
        raise BridgeV5Error("E_PRIOR_IMMUTABILITY")
    return observed


def _config() -> dict[str, Any]:
    value = _read_json(CONFIG)
    if (
        value.get("schema_version")
        != "childlens-calibration-recovery-scale-lag-v5.0.0"
        or value.get("scope", {}).get("development_participants") != 18
        or value.get("scope", {}).get(
            "locked_rows_may_be_loaded_inspected_scored_summarized_or_decoded"
        )
        is not False
        or value.get("learner", {}).get("route_count") != 1
        or value.get("learner", {}).get("alternative_search_after_childlens_outcomes")
        is not False
        or value.get("instruments", {})
        .get("audio", {})
        .get("tokenizer_decoder_ctc_asr_translation_or_language_id_loaded")
        is not False
        or value.get("terminal_states")
        != [
            "PASS_DETECTABLE_STRUCTURE",
            "PASS_PRECISE_WEAK_OR_FLAT",
            "NO_GO_UNINFORMATIVE",
        ]
    ):
        raise BridgeV5Error("E_PROTOCOL")
    expected_parameters = trainable_parameter_count(1024, 384, 256, 128)
    if value["learner"]["trainable_parameter_count"] != expected_parameters:
        raise BridgeV5Error("E_PARAMETER_COUNT")
    return value


def _discover_attested_runtime() -> Path:
    """Discover only the v5 attestation, never a legacy manifest."""

    candidates: list[Path] = []
    for hidden in ROOT.parent.iterdir():
        if (
            hidden.name.startswith(".")
            and "childlens" in hidden.name.casefold()
            and _private_directory(hidden)
        ):
            candidates.extend(
                path.parents[3]
                for path in hidden.rglob(
                    "restricted_manifest/provisional_calibration_v1/"
                    "childlens_alignment_bridge_v5/administrative/"
                    "development_only_attestation.json"
                )
                if _private_file(path)
            )
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) != 1:
        raise BridgeV5Error("E_ATTESTED_RUNTIME_DISCOVERY")
    runtime = unique[0]
    if (
        not _private_directory(runtime)
        or not _private_file(runtime / ".metadata_never_index")
        or _inside(runtime, ROOT)
    ):
        raise BridgeV5Error("E_ATTESTED_RUNTIME_CONTROL")
    return runtime


def _attested_input() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load only the exact-18 attestation and its bound scientific input."""

    runtime = _discover_attested_runtime()
    attestation_path = runtime / PRIVATE_ATTESTATION
    input_path = runtime / PRIVATE_INPUT
    if (
        not _private_file(attestation_path)
        or not _private_file(input_path)
        or not ADMIN_RECEIPT.is_file()
        or ADMIN_RECEIPT.is_symlink()
    ):
        raise BridgeV5Error("E_ATTESTED_INPUT")
    public = _read_json(ADMIN_RECEIPT)
    attestation = _read_json(attestation_path)
    scientific_input = _read_json(input_path)
    if (
        public.get("status") != "PASS"
        or public.get("clean_run_id") != CLEAN_RUN_ID
        or public.get("frozen_v5_config_sha256") != FROZEN_CONFIG_SHA256
        or public.get("incident_receipt_sha256")
        != ORIGINAL_INCIDENT_RECEIPT_SHA256
        or public.get("attestation_sha256") != _sha256_file(attestation_path)
        or public.get("scientific_input_sha256") != _sha256_file(input_path)
        or public.get("locked_participant_count") != 0
        or public.get("legacy_mixed_scope_inputs_available_to_scientific_process")
        is not False
        or attestation.get("status") != "PASS"
        or attestation.get("scientific_input_sha256") != _sha256_file(input_path)
        or attestation.get("development_participant_count") != 18
        or attestation.get("locked_participant_count") != 0
        or attestation.get(
            "legacy_mixed_scope_inputs_available_to_scientific_process"
        )
        is not False
        or scientific_input.get("schema_version")
        != "childlens-v5-development-only-scientific-input-v1.0.0"
        or scientific_input.get("clean_run_id") != CLEAN_RUN_ID
        or scientific_input.get("frozen_v5_config_sha256")
        != FROZEN_CONFIG_SHA256
        or scientific_input.get("development_participant_count") != 18
        or scientific_input.get("locked_participant_count") != 0
        or not isinstance(scientific_input.get("items"), list)
    ):
        raise BridgeV5Error("E_ATTESTED_INPUT")
    items = scientific_input["items"]
    participants = {
        str(row.get("participant_key", ""))
        for row in items
        if isinstance(row, Mapping)
    }
    if (
        len(participants) != 18
        or "" in participants
        or any(not isinstance(row, Mapping) for row in items)
        or Counter(str(row.get("cohort", "")) for row in items).keys()
        != {"original_eight", "v3_expansion_ten"}
    ):
        raise BridgeV5Error("E_ATTESTED_SCOPE")
    return runtime, scientific_input, attestation


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _timing_number(
    row: Mapping[str, Any], names: Sequence[str]
) -> float | None:
    accepted = set(names)
    for key, value in row.items():
        if _normalise_key(str(key)) in accepted:
            number = _finite_number(value)
            if number is not None:
                return number
    return None


def _speech_windows(document: Mapping[str, Any]) -> tuple[list[tuple[int, int]], int]:
    annotations = document.get("annotations")
    if not isinstance(annotations, list):
        raise BridgeV5Error("E_ANNOTATION_SCHEMA")
    windows: list[tuple[int, int]] = []
    invalid = 0
    for raw in annotations:
        if not isinstance(raw, Mapping):
            raise BridgeV5Error("E_ANNOTATION_SCHEMA")
        event_values = [
            value
            for key, value in raw.items()
            if _normalise_key(str(key))
            in {"eventid", "event_id", "category", "label", "type"}
            and isinstance(value, str)
        ]
        if not any(SPEECH_RE.search(value) for value in event_values):
            continue
        start = _timing_number(
            raw, ("start", "start_time", "starttime", "onset", "time")
        )
        end = _timing_number(raw, ("end", "end_time", "endtime", "offset"))
        duration = _timing_number(raw, ("duration",))
        if end is None and start is not None and duration is not None:
            end = start + duration
        if start is None or end is None or start < 0 or end <= start:
            invalid += 1
            continue
        windows.append((math.ceil(start * 1000), math.floor(end * 1000)))
    return windows, invalid


def _coalesce(
    windows: Sequence[tuple[int, int]], duration_ms: int
) -> list[tuple[int, int]]:
    cleaned = sorted(
        (max(0, start), min(duration_ms, end))
        for start, end in windows
        if min(duration_ms, end) > max(0, start)
    )
    merged: list[list[int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _safe_annotation(
    runtime: Path, locator: str, expected_sha256: str
) -> Mapping[str, Any]:
    path = Path(locator)
    if not path.is_absolute():
        path = runtime / path
    path = path.resolve()
    control_root = runtime.parent.resolve()
    if (
        not _inside(path, control_root)
        or not _private_file(path)
        or not HEX64.fullmatch(expected_sha256)
        or _sha256_file(path) != expected_sha256
    ):
        raise BridgeV5Error("E_ANNOTATION_BINDING")
    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise BridgeV5Error("E_ANNOTATION_SCHEMA")
    return value


def _mode(values: Sequence[Any]) -> str:
    normalized = [
        str(value) if value not in (None, "") else "__UNAVAILABLE__"
        for value in values
    ]
    counts = Counter(normalized)
    return min(counts, key=lambda value: (-counts[value], value))


def _overlap(
    left: int, right: int, speech: Sequence[tuple[int, int]]
) -> int:
    return sum(max(0, min(right, end) - max(left, start)) for start, end in speech)


def _map_union_range(
    rows: Sequence[Mapping[str, Any]], offset: int, duration: int
) -> list[dict[str, int | str]]:
    remaining = duration
    cursor = 0
    result: list[dict[str, int | str]] = []
    for row in rows:
        capacity = int(row["usable_end_ms"]) - int(row["usable_start_ms"])
        if offset >= cursor + capacity:
            cursor += capacity
            continue
        local = max(0, offset - cursor)
        start = int(row["usable_start_ms"]) + local
        take = min(remaining, int(row["usable_end_ms"]) - start)
        if take > 0:
            result.append(
                {
                    "source_object_key": str(row["source_object_key"]),
                    "start_ms": start,
                    "end_ms": start + take,
                }
            )
            offset += take
            remaining -= take
        cursor += capacity
        if remaining == 0:
            break
    if remaining:
        raise BridgeV5Error("E_DURATION_SUPPORT")
    return result


def _candidate_windows_in_cores(
    cores: Sequence[Mapping[str, Any]],
    speech_by_source: Mapping[str, Sequence[tuple[int, int]]],
    *,
    duration_ms: int,
    maximum: int,
    seed: str,
    participant: str,
) -> list[dict[str, Any]]:
    proposed: list[dict[str, Any]] = []
    for core in cores:
        source = str(core["source_object_key"])
        start = int(core.get("start_ms", core.get("usable_start_ms")))
        end = int(core.get("end_ms", core.get("usable_end_ms")))
        cursor = start
        while cursor + duration_ms <= end:
            right = cursor + duration_ms
            support = _overlap(cursor, right, speech_by_source[source])
            if support > 0:
                proposed.append(
                    {
                        "source_object_key": source,
                        "start_ms": cursor,
                        "end_ms": right,
                        "released_speech_support_fraction": support / duration_ms,
                        "selection_hash": keyed_hash(
                            seed, participant, duration_ms, source, cursor
                        ),
                    }
                )
            cursor = right
    selected = sorted(proposed, key=lambda row: str(row["selection_hash"]))[:maximum]
    selected.sort(
        key=lambda row: (
            str(row["source_object_key"]),
            int(row["start_ms"]),
        )
    )
    return selected


def clean_stage0() -> dict[str, Any]:
    """Freeze the clean exact-18 selection without decoding media."""

    config = _config()
    if _sha256_file(CONFIG) != FROZEN_CONFIG_SHA256:
        raise BridgeV5Error("E_FROZEN_METHOD_CHANGED")
    prior = _validate_prior_trees()
    runtime, scientific_input, attestation = _attested_input()
    del attestation
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in scientific_input["items"]:
        row = dict(raw)
        participant = str(row["participant_key"])
        duration_ms = int(row["source_duration_milliseconds"])
        if (
            duration_ms <= 0
            or not isinstance(row.get("source_object_key"), str)
            or not isinstance(row.get("source_locator"), str)
            or not isinstance(row.get("annotation_bindings"), list)
        ):
            raise BridgeV5Error("E_SCIENTIFIC_INPUT_ROW")
        speech: list[tuple[int, int]] = []
        invalid = 0
        for binding in row["annotation_bindings"]:
            if not isinstance(binding, Mapping):
                raise BridgeV5Error("E_ANNOTATION_BINDING")
            document = _safe_annotation(
                runtime,
                str(binding.get("source_locator", "")),
                str(binding.get("local_sha256", "")),
            )
            windows, bad = _speech_windows(document)
            speech.extend(windows)
            invalid += bad
        row["released_speech_windows_ms"] = _coalesce(speech, duration_ms)
        row["invalid_released_timing_rows"] = invalid
        grouped[participant].append(row)
    if len(grouped) != 18:
        raise BridgeV5Error("E_ATTESTED_SCOPE")

    availability = [
        {
            "participant_key": participant,
            "available_released_seconds": sum(
                int(row["source_duration_milliseconds"]) for row in rows
            )
            / 1000,
        }
        for participant, rows in grouped.items()
    ]
    duration_config = config["stage0"]["uniform_duration"]
    target_seconds = common_duration_target(
        availability,
        participant_count=18,
        minimum_seconds=int(duration_config["minimum_seconds_per_participant"]),
        preferred_seconds=int(duration_config["preferred_seconds_per_participant"]),
        maximum_seconds=int(duration_config["maximum_seconds_per_participant"]),
        quantum_seconds=int(duration_config["quantum_seconds"]),
        minimum_total_hours=float(duration_config["minimum_total_hours"]),
    )
    target_ms = target_seconds * 1000
    fold_rows: list[dict[str, str]] = []
    private_participants: list[dict[str, Any]] = []
    seed = str(config["cross_validation"]["seed"])
    lag_multiples = [1, 2, 4]
    grids = lag_grid(
        [int(value) for value in config["windows_and_lags"]["window_durations_seconds"]],
        lag_multiples,
    )
    selected_recordings: set[str] = set()
    all_window_counts: Counter[int] = Counter()
    minimum_counts: dict[int, int] = {}
    invalid_total = 0
    total_acquisition_ms = 0
    for participant, rows in sorted(grouped.items()):
        observation_inventory = [
            {
                **row,
                "usable_start_ms": 0,
                "usable_end_ms": int(row["source_duration_milliseconds"]),
            }
            for row in sorted(rows, key=lambda value: str(value["source_object_key"]))
        ]
        total_usable = sum(
            int(row["usable_end_ms"]) - int(row["usable_start_ms"])
            for row in observation_inventory
        )
        if total_usable < target_ms:
            raise BridgeV5Error("E_DURATION_SUPPORT")
        offset = int(
            keyed_hash(seed, "observation", participant), 16
        ) % (total_usable - target_ms + 1)
        cores = _map_union_range(observation_inventory, offset, target_ms)
        speech_by_source = {
            str(row["source_object_key"]): row["released_speech_windows_ms"]
            for row in observation_inventory
        }
        duration_by_source = {
            str(row["source_object_key"]): int(
                row["source_duration_milliseconds"]
            )
            for row in observation_inventory
        }
        windows_by_duration: dict[str, list[dict[str, Any]]] = {}
        acquisition_proposals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for core in cores:
            acquisition_proposals[str(core["source_object_key"])].append(
                (int(core["start_ms"]), int(core["end_ms"]))
            )
        for duration in sorted(grids):
            margin_ms = 4 * duration * 1000
            usable_for_scale = [
                {
                    **row,
                    "usable_start_ms": margin_ms,
                    "usable_end_ms": (
                        int(row["source_duration_milliseconds"]) - margin_ms
                    ),
                }
                for row in observation_inventory
                if int(row["source_duration_milliseconds"])
                > 2 * margin_ms + duration * 1000
            ]
            windows = _candidate_windows_in_cores(
                usable_for_scale,
                speech_by_source,
                duration_ms=duration * 1000,
                maximum=min(
                    int(
                        config["windows_and_lags"][
                            "maximum_real_windows_per_participant_per_duration"
                        ]
                    ),
                    target_ms // (duration * 1000),
                ),
                seed=seed,
                participant=participant,
            )
            for window in windows:
                window["signed_lag_controls_ms"] = {
                    str(lag): {
                        "start_ms": int(window["start_ms"]) + lag * 1000,
                        "end_ms": int(window["end_ms"]) + lag * 1000,
                    }
                    for lag in grids[duration]
                }
                source = str(window["source_object_key"])
                controls = list(window["signed_lag_controls_ms"].values())
                acquisition_proposals[source].append(
                    (
                        min(int(value["start_ms"]) for value in controls),
                        max(int(value["end_ms"]) for value in controls),
                    )
                )
            windows_by_duration[str(duration)] = windows
            all_window_counts[duration] += len(windows)
            minimum_counts[duration] = min(
                minimum_counts.get(duration, len(windows)), len(windows)
            )
        acquisition_intervals: dict[str, list[dict[str, int]]] = {}
        for source, proposals in sorted(acquisition_proposals.items()):
            merged = _coalesce(proposals, duration_by_source[source])
            acquisition_intervals[source] = [
                {"start_ms": start, "end_ms": end} for start, end in merged
            ]
            selected_recordings.add(source)
            total_acquisition_ms += sum(end - start for start, end in merged)
        invalid_total += sum(
            int(row["invalid_released_timing_rows"]) for row in rows
        )
        cohort = str(rows[0]["cohort"])
        fold_rows.append(
            {
                "participant_key": participant,
                "cohort": cohort,
                "activity_bin": _mode(
                    [row.get("released_activity_label") for row in rows]
                ),
                "location_bin": _mode(
                    [row.get("released_location_label") for row in rows]
                ),
                "speech_support_bin": _mode(
                    [row.get("released_speech_support_bin") for row in rows]
                ),
            }
        )
        private_participants.append(
            {
                "participant_key": participant,
                "cohort": cohort,
                "core_observation_intervals": cores,
                "windows_by_duration_seconds": windows_by_duration,
                "acquisition_intervals_by_source": acquisition_intervals,
            }
        )
    folds = deterministic_folds(fold_rows, seed=seed, fold_count=3, fold_size=6)
    for row in private_participants:
        row["evaluation_fold"] = folds[str(row["participant_key"])]
    required = int(
        config["windows_and_lags"]["minimum_evaluable_windows_per_participant_per_duration"]
    )
    support_pass = all(minimum_counts.get(duration, 0) >= required for duration in grids)
    selection = {
        "schema_version": "childlens-v5-clean-stage0-selection-v1.0.0",
        "status": (
            "FROZEN_BEFORE_SELECTIVE_ACQUISITION_OR_MEDIA_DECODING"
            if support_pass
            else "TERMINAL_SUPPORT_FAILURE_BEFORE_ACQUISITION"
        ),
        "clean_run_id": CLEAN_RUN_ID,
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "administrative_scientific_input_sha256": _sha256_file(
            runtime / PRIVATE_INPUT
        ),
        "administrative_attestation_sha256": _sha256_file(
            runtime / PRIVATE_ATTESTATION
        ),
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "target_seconds_per_participant": target_seconds,
        "total_core_observation_hours": 18 * target_seconds / 3600,
        "folds": folds,
        "participants": private_participants,
    }
    private_path = runtime / PRIVATE_STAGE0_SELECTION
    _write_private(private_path, selection)
    positive = run_embedding_positive_control(
        seed=config["positive_control"]["seed"],
        participants=config["positive_control"]["participants"],
        windows_per_participant=config["positive_control"]["windows_per_participant"],
        confidence=config["inference"]["confidence"],
    )
    _write(CLEAN_POSITIVE_RECEIPT, positive)
    fold_counts = sorted(Counter(folds.values()).values())
    receipt = {
        "schema_version": "childlens-v5-clean-stage0-freeze-v1.0.0",
        "status": selection["status"],
        "clean_run_id": CLEAN_RUN_ID,
        "original_incident_preserved": True,
        "original_incident_receipt_sha256": ORIGINAL_INCIDENT_RECEIPT_SHA256,
        "administrative_correction_receipt_sha256": _sha256_file(ADMIN_RECEIPT),
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "scientific_method_changed": False,
        "scientific_runner_sha256": _sha256_file(Path(__file__)),
        "prior_v1_v2_v3_v4_immutable": True,
        "prior_tree_sha256": prior,
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "legacy_mixed_scope_manifests_opened_by_scientific_process": False,
        "target_seconds_per_participant": target_seconds,
        "total_core_observation_hours": 18 * target_seconds / 3600,
        "source_recordings_selected": len(selected_recordings),
        "bounded_acquisition_hours": total_acquisition_ms / 3_600_000,
        "fold_counts": fold_counts,
        "minimum_windows_per_participant_by_duration_seconds": {
            str(key): value for key, value in sorted(minimum_counts.items())
        },
        "total_windows_by_duration_seconds": {
            str(key): value for key, value in sorted(all_window_counts.items())
        },
        "minimum_required_windows_per_participant_duration": required,
        "support_gate_pass": support_pass,
        "invalid_released_timing_rows": invalid_total,
        "private_selection_sha256": _sha256_file(private_path),
        "fold_identity_sha256": digest(
            sorted(
                (
                    keyed_hash(seed, participant),
                    fold,
                )
                for participant, fold in folds.items()
            )
        ),
        "synthetic_positive_control_pass": bool(positive["pass"]),
        "new_media_decoded": False,
        "new_media_acquired": False,
        "childlens_embeddings_training_or_scores_computed": False,
        "restricted_values_exported": False,
    }
    public_guard(receipt)
    _write(CLEAN_STAGE0_RECEIPT, receipt)
    return receipt


def _transfer_module():
    path = ROOT / "scripts/acquire_childlens_alignment_bridge_expansion_v3.py"
    spec = importlib.util.spec_from_file_location("childlens_v5_transfer_base", path)
    if spec is None or spec.loader is None:
        raise BridgeV5Error("E_TRANSFER_RUNTIME")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_private_clip(module, runtime: Path, clip: Mapping[str, Any]) -> None:
    relative = clip.get("relative_path")
    sha256 = clip.get("sha256")
    size = clip.get("bytes")
    if (
        not isinstance(relative, str)
        or not relative.startswith(PRIVATE_CLIPS.as_posix() + "/")
        or ".." in Path(relative).parts
        or not isinstance(sha256, str)
        or not HEX64.fullmatch(sha256)
        or type(size) is not int
        or size <= 0
    ):
        raise BridgeV5Error("E_ACQUISITION_CHECKPOINT")
    path = (runtime / relative).resolve()
    if (
        not _inside(path, runtime)
        or not _private_file(path)
        or path.stat().st_size != size
        or _sha256_file(path) != sha256
    ):
        raise BridgeV5Error("E_ACQUISITION_CHECKPOINT")
    _, video, audio = module._probe(path)
    if not video or not audio:
        raise BridgeV5Error("E_ACQUISITION_CHECKPOINT")


def clean_acquire() -> dict[str, Any]:
    """Acquire only frozen development bounds, one full source at a time."""

    if _sha256_file(CONFIG) != FROZEN_CONFIG_SHA256:
        raise BridgeV5Error("E_FROZEN_METHOD_CHANGED")
    if (
        not CLEAN_STAGE0_RECEIPT.is_file()
        or _read_json(CLEAN_STAGE0_RECEIPT).get("status")
        != "FROZEN_BEFORE_SELECTIVE_ACQUISITION_OR_MEDIA_DECODING"
    ):
        raise BridgeV5Error("E_STAGE0_NOT_FROZEN")
    runtime, scientific_input, _ = _attested_input()
    selection_path = runtime / PRIVATE_STAGE0_SELECTION
    if (
        not _private_file(selection_path)
        or _sha256_file(selection_path)
        != _read_json(CLEAN_STAGE0_RECEIPT).get("private_selection_sha256")
    ):
        raise BridgeV5Error("E_STAGE0_BINDING")
    selection = _read_json(selection_path)
    if (
        selection.get("clean_run_id") != CLEAN_RUN_ID
        or selection.get("locked_participant_count") != 0
        or selection.get("status")
        != "FROZEN_BEFORE_SELECTIVE_ACQUISITION_OR_MEDIA_DECODING"
    ):
        raise BridgeV5Error("E_STAGE0_BINDING")
    source_rows = {
        str(row["source_object_key"]): row for row in scientific_input["items"]
    }
    intervals_by_source: dict[str, list[dict[str, int]]] = defaultdict(list)
    participant_by_source: dict[str, str] = {}
    for participant in selection["participants"]:
        participant_key = str(participant["participant_key"])
        for source, intervals in participant["acquisition_intervals_by_source"].items():
            if source in participant_by_source and participant_by_source[source] != participant_key:
                raise BridgeV5Error("E_ACQUISITION_SCOPE")
            participant_by_source[source] = participant_key
            intervals_by_source[source].extend(dict(value) for value in intervals)
    if (
        set(intervals_by_source) - set(source_rows)
        or len(set(participant_by_source.values())) != 18
        or not intervals_by_source
    ):
        raise BridgeV5Error("E_ACQUISITION_SCOPE")
    plan = [
        {
            "source_object_key": source,
            "participant_key": participant_by_source[source],
            "source_locator": source_rows[source]["source_locator"],
            "expected_size_bytes": source_rows[source]["source_size_bytes"],
            "expected_duration_ms": source_rows[source][
                "source_duration_milliseconds"
            ],
            "intervals": sorted(
                intervals_by_source[source],
                key=lambda value: (int(value["start_ms"]), int(value["end_ms"])),
            ),
        }
        for source in sorted(intervals_by_source)
    ]
    module = _transfer_module()
    for path in (module.FFMPEG, module.FFPROBE, module.KEYCHAIN_TOOL):
        if not path.is_file():
            raise BridgeV5Error("E_TRANSFER_RUNTIME")
    namespace = runtime / PRIVATE_CLEAN_RELATIVE
    clips_directory = runtime / PRIVATE_CLIPS
    transient_directory = runtime / PRIVATE_TRANSIENT
    for directory in (namespace, clips_directory, transient_directory):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        if not _private_directory(directory):
            raise BridgeV5Error("E_PRIVATE_NAMESPACE")
    checkpoint_path = runtime / PRIVATE_ACQUISITION_CHECKPOINT
    template = {
        "schema_version": "childlens-v5-clean-acquisition-v1.0.0",
        "clean_run_id": CLEAN_RUN_ID,
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "private_selection_sha256": _sha256_file(selection_path),
        "status": "IN_PROGRESS",
        "items": [
            {
                "source_object_key": row["source_object_key"],
                "source_exact_bytes": None,
                "source_sha256": None,
                "clips": [],
                "status": "PENDING",
            }
            for row in plan
        ],
    }
    if checkpoint_path.exists():
        checkpoint = _read_json(checkpoint_path)
        if (
            not _private_file(checkpoint_path)
            or checkpoint.get("schema_version") != template["schema_version"]
            or checkpoint.get("clean_run_id") != CLEAN_RUN_ID
            or checkpoint.get("private_selection_sha256")
            != template["private_selection_sha256"]
            or not isinstance(checkpoint.get("items"), list)
            or len(checkpoint["items"]) != len(plan)
        ):
            raise BridgeV5Error("E_ACQUISITION_CHECKPOINT")
    else:
        checkpoint = template
    for expected, actual in zip(template["items"], checkpoint["items"]):
        if (
            expected["source_object_key"] != actual.get("source_object_key")
            or actual.get("status") not in {"PENDING", "IN_PROGRESS", "COMPLETE"}
            or not isinstance(actual.get("clips"), list)
        ):
            raise BridgeV5Error("E_ACQUISITION_CHECKPOINT")
        for clip in actual["clips"]:
            _verify_private_clip(module, runtime, clip)
    transient_source = transient_directory / "source.bin"
    transient_clip = transient_directory / "derived.mp4"
    for path in (transient_source, transient_clip):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    try:
        token = module._read_token()
    except Exception as exc:
        code = getattr(exc, "code", "E_CREDENTIAL")
        raise BridgeV5Error(str(code)) from exc
    credential_deleted = False
    try:
        transfer_config = _read_json(runtime / "transfer_v1_2/config.json")
        client = module.SeafileClient(transfer_config, token)
        for plan_row, item in zip(plan, checkpoint["items"]):
            if item["status"] == "COMPLETE":
                continue
            exact_size = client.exact_size(plan_row["source_locator"])
            if (
                exact_size <= 0
                or exact_size > 100 * 1024**3
                or shutil.disk_usage(runtime.parent).free - exact_size < 50 * 1024**3
            ):
                raise BridgeV5Error("E_ACQUISITION_CAPACITY_OR_SIZE")
            response = client.open_download(plan_row["source_locator"])
            source_digest = hashlib.sha256()
            transferred = 0
            try:
                descriptor = os.open(
                    transient_source,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    while True:
                        block = response.read(4 * 1024**2)
                        if not block:
                            break
                        transferred += len(block)
                        if transferred > exact_size:
                            raise BridgeV5Error("E_ACQUISITION_STREAM")
                        handle.write(block)
                        source_digest.update(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                response.close()
            if transferred != exact_size or not _private_file(transient_source):
                raise BridgeV5Error("E_ACQUISITION_STREAM")
            source_duration, source_video, source_audio = module._probe(
                transient_source
            )
            expected_duration = int(plan_row["expected_duration_ms"]) / 1000
            if (
                not source_video
                or not source_audio
                or abs(source_duration - expected_duration)
                > max(2.0, expected_duration * 0.02)
            ):
                raise BridgeV5Error("E_ACQUISITION_SOURCE_MEDIA")
            item["source_exact_bytes"] = transferred
            item["source_sha256"] = source_digest.hexdigest()
            item["status"] = "IN_PROGRESS"
            _write_private(checkpoint_path, checkpoint)
            existing = {
                (int(clip["source_start_ms"]), int(clip["source_end_ms"])): clip
                for clip in item["clips"]
            }
            for ordinal, interval in enumerate(plan_row["intervals"]):
                start_ms = int(interval["start_ms"])
                end_ms = int(interval["end_ms"])
                if (start_ms, end_ms) in existing:
                    continue
                duration_seconds = (end_ms - start_ms) / 1000
                command = [
                    str(module.FFMPEG),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_ms / 1000:.3f}",
                    "-i",
                    str(transient_source),
                    "-t",
                    f"{duration_seconds:.3f}",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-vf",
                    (
                        "scale=640:640:force_original_aspect_ratio=decrease:"
                        "force_divisible_by=2,fps=15,format=yuv420p"
                    ),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "24",
                    "-c:a",
                    "aac",
                    "-ac",
                    "1",
                    "-b:a",
                    "96k",
                    "-movflags",
                    "+faststart",
                    "-y",
                    str(transient_clip),
                ]
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=7200,
                )
                clip_secured = (
                    result.returncode == 0
                    and _secure_generated_file(transient_clip)
                )
                if result.returncode != 0 or not clip_secured:
                    _write_private(
                        runtime / PRIVATE_ACQUISITION_FAILURE,
                        {
                            "schema_version": (
                                "childlens-v5-private-acquisition-failure-v1.0.0"
                            ),
                            "clean_run_id": CLEAN_RUN_ID,
                            "error_code": "E_ACQUISITION_CLIP",
                            "source_object_key": plan_row["source_object_key"],
                            "interval_ordinal": ordinal,
                            "source_start_ms": start_ms,
                            "source_end_ms": end_ms,
                            "probed_source_duration_seconds": source_duration,
                            "ffmpeg_returncode": result.returncode,
                            "ffmpeg_stderr_tail": result.stderr.decode(
                                "utf-8", errors="replace"
                            )[-4000:],
                        },
                    )
                    raise BridgeV5Error("E_ACQUISITION_CLIP")
                actual_duration, clip_video, clip_audio = module._probe(
                    transient_clip
                )
                if (
                    not clip_video
                    or not clip_audio
                    or abs(actual_duration - duration_seconds)
                    > max(1.0, duration_seconds * 0.02)
                ):
                    raise BridgeV5Error("E_ACQUISITION_CLIP_MEDIA")
                clip_sha256 = _sha256_file(transient_clip)
                destination = clips_directory / f"{clip_sha256}.mp4"
                clip_bytes = transient_clip.stat().st_size
                if destination.exists():
                    if (
                        not _private_file(destination)
                        or destination.stat().st_size != clip_bytes
                        or _sha256_file(destination) != clip_sha256
                    ):
                        raise BridgeV5Error("E_ACQUISITION_CLIP_CONFLICT")
                    transient_clip.unlink()
                else:
                    os.replace(transient_clip, destination)
                    os.chmod(destination, 0o600)
                item["clips"].append(
                    {
                        "ordinal": ordinal,
                        "source_start_ms": start_ms,
                        "source_end_ms": end_ms,
                        "relative_path": destination.relative_to(runtime).as_posix(),
                        "bytes": clip_bytes,
                        "sha256": clip_sha256,
                        "duration_seconds": round(actual_duration, 3),
                    }
                )
                _write_private(checkpoint_path, checkpoint)
            transient_source.unlink()
            item["status"] = "COMPLETE"
            checkpoint["status"] = (
                "COMPLETE"
                if all(value["status"] == "COMPLETE" for value in checkpoint["items"])
                else "IN_PROGRESS"
            )
            _write_private(checkpoint_path, checkpoint)
            if any(transient_directory.iterdir()):
                raise BridgeV5Error("E_TRANSIENT_NOT_EMPTY")
    finally:
        token = ""
        for path in (transient_source, transient_clip):
            with contextlib.suppress(OSError):
                path.unlink()
        try:
            module._delete_token()
            credential_deleted = True
        except Exception as exc:
            code = getattr(exc, "code", "E_CREDENTIAL_DELETE")
            raise BridgeV5Error(str(code)) from exc
    if (
        checkpoint.get("status") != "COMPLETE"
        or not all(item["status"] == "COMPLETE" for item in checkpoint["items"])
        or any(transient_directory.iterdir())
    ):
        raise BridgeV5Error("E_ACQUISITION_INCOMPLETE")
    total_source = sum(int(item["source_exact_bytes"]) for item in checkpoint["items"])
    all_clips = [clip for item in checkpoint["items"] for clip in item["clips"]]
    total_retained = sum(int(clip["bytes"]) for clip in all_clips)
    receipt = {
        "schema_version": "childlens-v5-clean-acquisition-receipt-v1.0.0",
        "status": "COMPLETE",
        "clean_run_id": CLEAN_RUN_ID,
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "clean_stage0_receipt_sha256": _sha256_file(CLEAN_STAGE0_RECEIPT),
        "restricted_checkpoint_sha256": _sha256_file(checkpoint_path),
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "selected_source_recordings": len(plan),
        "completed_source_recordings": len(checkpoint["items"]),
        "retained_bounded_clip_count": len(all_clips),
        "source_transfer_gib_rounded_up": math.ceil(total_source / 1024**3),
        "retained_clip_gib_rounded_up": math.ceil(total_retained / 1024**3),
        "maximum_concurrent_full_sources": 1,
        "transient_full_sources_removed": True,
        "transient_directory_empty": True,
        "temporary_read_only_credential_deleted": credential_deleted,
        "all_retained_clips_video_and_audio_verified": True,
        "external_volume_aea_or_babyview_used": False,
        "locked_media_annotations_or_outcomes_accessed": False,
        "childlens_embeddings_training_or_scores_computed": False,
        "restricted_values_exported": False,
    }
    public_guard(receipt)
    _write(CLEAN_ACQUISITION_RECEIPT, receipt)
    return receipt


def _verify_frozen_models(*, write: bool) -> dict[str, Any]:
    if not INSTRUMENT_PYTHON.is_file() or not os.access(INSTRUMENT_PYTHON, os.X_OK):
        raise BridgeV5Error("E_INSTRUMENT_RUNTIME")
    verified: dict[str, str] = {}
    for path, expected in EXPECTED_MODEL_HASHES.items():
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
            raise BridgeV5Error("E_MODEL_HASH")
        verified[path.name + ":" + path.parent.name] = expected
    receipt = {
        "schema_version": "childlens-v5-frozen-model-hash-receipt-v1.0.0",
        "status": "PASS",
        "clean_run_id": CLEAN_RUN_ID,
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "audio_repository": "facebook/wav2vec2-xls-r-300m",
        "audio_revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
        "audio_weights_sha256": EXPECTED_MODEL_HASHES[
            AUDIO_MODEL / "pytorch_model.bin"
        ],
        "audio_config_sha256": EXPECTED_MODEL_HASHES[AUDIO_MODEL / "config.json"],
        "audio_preprocessor_sha256": EXPECTED_MODEL_HASHES[
            AUDIO_MODEL / "preprocessor_config.json"
        ],
        "vision_repository": "facebook/dinov2-small",
        "vision_revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "vision_weights_sha256": EXPECTED_MODEL_HASHES[
            VISION_MODEL / "model.safetensors"
        ],
        "vision_config_sha256": EXPECTED_MODEL_HASHES[VISION_MODEL / "config.json"],
        "vision_preprocessor_sha256": EXPECTED_MODEL_HASHES[
            VISION_MODEL / "preprocessor_config.json"
        ],
        "verified_file_count": len(verified),
        "tokenizer_decoder_ctc_asr_translation_or_language_id_loaded": False,
    }
    public_guard(receipt)
    if write:
        _write(CLEAN_MODEL_RECEIPT, receipt)
    return receipt


def _source_scientific_metadata(
    runtime: Path, scientific_input: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in scientific_input["items"]:
        source = str(raw["source_object_key"])
        speech: list[tuple[int, int]] = []
        for binding in raw["annotation_bindings"]:
            document = _safe_annotation(
                runtime,
                str(binding["source_locator"]),
                str(binding["local_sha256"]),
            )
            values, _ = _speech_windows(document)
            speech.extend(values)
        result[source] = {
            "participant_key": str(raw["participant_key"]),
            "activity": str(
                raw.get("released_activity_label") or "__UNAVAILABLE__"
            ),
            "location": str(
                raw.get("released_location_label") or "__UNAVAILABLE__"
            ),
            "speech_windows_ms": _coalesce(
                speech, int(raw["source_duration_milliseconds"])
            ),
        }
    return result


def _measurement_inventory(
    runtime: Path,
    selection: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    source_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    clips_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transfer = _transfer_module()
    for item in checkpoint["items"]:
        source = str(item["source_object_key"])
        for clip in item["clips"]:
            _verify_private_clip(transfer, runtime, clip)
            clips_by_source[source].append(dict(clip))
    requests_by_key: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for participant in selection["participants"]:
        participant_key = str(participant["participant_key"])
        fold = int(participant["evaluation_fold"])
        for duration_text, windows in participant[
            "windows_by_duration_seconds"
        ].items():
            duration = int(duration_text)
            for window in windows:
                source = str(window["source_object_key"])
                row_key = keyed_hash(
                    CLEAN_RUN_ID,
                    participant_key,
                    duration,
                    str(window["selection_hash"]),
                )
                lag_requests: dict[str, str] = {}
                for lag, bounds in window["signed_lag_controls_ms"].items():
                    start_ms = int(bounds["start_ms"])
                    end_ms = int(bounds["end_ms"])
                    request_key = keyed_hash(
                        CLEAN_RUN_ID, source, start_ms, end_ms
                    )
                    lag_requests[str(lag)] = request_key
                    if request_key not in requests_by_key:
                        support = _overlap(
                            start_ms,
                            end_ms,
                            source_metadata[source]["speech_windows_ms"],
                        ) / (end_ms - start_ms)
                        requests_by_key[request_key] = {
                            "request_key": request_key,
                            "source_object_key": source,
                            "participant_key": participant_key,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "duration_seconds": duration,
                            "released_speech_support_fraction": support,
                            "activity": source_metadata[source]["activity"],
                            "location": source_metadata[source]["location"],
                        }
                rows.append(
                    {
                        "row_key": row_key,
                        "participant_key": participant_key,
                        "fold": fold,
                        "duration_seconds": duration,
                        "source_object_key": source,
                        "activity": source_metadata[source]["activity"],
                        "location": source_metadata[source]["location"],
                        "lag_requests": lag_requests,
                    }
                )
    requests = [requests_by_key[key] for key in sorted(requests_by_key)]
    for index, request in enumerate(requests):
        request["index"] = index
        containing = [
            clip
            for clip in clips_by_source[request["source_object_key"]]
            if int(clip["source_start_ms"]) <= int(request["start_ms"])
            and int(clip["source_end_ms"]) >= int(request["end_ms"])
        ]
        if len(containing) != 1:
            raise BridgeV5Error("E_REQUEST_CLIP_BINDING")
        request["clip_relative_path"] = containing[0]["relative_path"]
        request["clip_start_ms"] = int(containing[0]["source_start_ms"])
    if len(rows) != 3049 or len(requests) != 13526:
        raise BridgeV5Error("E_FROZEN_WINDOW_INVENTORY")
    return requests, rows, clips_by_source


def _decode_clip_audio(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            str(FFMPEG),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=3600,
    )
    if result.returncode != 0:
        raise BridgeV5Error("E_AUDIO_DECODE")
    signal = np.frombuffer(result.stdout, dtype="<f4").copy()
    if signal.size < 16000:
        raise BridgeV5Error("E_AUDIO_DECODE")
    return signal


def _decode_target_frames(path: Path, targets_ms: Sequence[int]) -> dict[int, Any]:
    import av

    targets = sorted(set(int(value) for value in targets_ms))
    output: dict[int, Any] = {}
    container = av.open(str(path))
    try:
        stream = next(
            (value for value in container.streams if value.type == "video"), None
        )
        if stream is None:
            raise BridgeV5Error("E_VIDEO_DECODE")
        target_index = 0
        prior: tuple[float, Any] | None = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            timestamp_ms = float(frame.pts * frame.time_base) * 1000
            while (
                target_index < len(targets)
                and timestamp_ms >= targets[target_index]
            ):
                selected = frame
                if prior is not None and abs(prior[0] - targets[target_index]) <= abs(
                    timestamp_ms - targets[target_index]
                ):
                    selected = prior[1]
                output[targets[target_index]] = selected.to_image().convert("RGB")
                target_index += 1
            prior = (timestamp_ms, frame)
            if target_index == len(targets):
                break
        if len(output) != len(targets):
            raise BridgeV5Error("E_VIDEO_DECODE")
        return output
    finally:
        container.close()


def _write_private_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, pending_name = tempfile.mkstemp(
        prefix=".pending-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    pending = Path(pending_name)
    try:
        os.chmod(pending, 0o600)
        np.savez_compressed(pending, **arrays)
        os.replace(pending, path)
        os.chmod(path, 0o600)
    finally:
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()


def _extract_feature_shard(
    *,
    clip_path: Path,
    clip_start_ms: int,
    requests: Sequence[Mapping[str, Any]],
    audio_model: Any,
    vision_model: Any,
    vision_processor: Any,
    torch: Any,
    device: Any,
) -> dict[str, np.ndarray]:
    signal = _decode_clip_audio(clip_path)
    indices = np.asarray([int(row["index"]) for row in requests], dtype=np.int32)
    audio_stats = np.empty((len(requests), 2, 1024), dtype=np.float32)
    vision_stats = np.empty((len(requests), 2, 384), dtype=np.float32)
    rms = np.empty(len(requests), dtype=np.float32)
    non_silent = np.empty(len(requests), dtype=np.float32)
    clipped = np.empty(len(requests), dtype=np.float32)
    motion = np.empty(len(requests), dtype=np.float32)
    persistence = np.empty(len(requests), dtype=np.float32)
    scene_change = np.empty(len(requests), dtype=np.float32)

    for duration in (2, 6, 18):
        positions = [
            position
            for position, row in enumerate(requests)
            if int(row["duration_seconds"]) == duration
        ]
        batch_size = {2: 24, 6: 8, 18: 3}[duration]
        for left in range(0, len(positions), batch_size):
            batch_positions = positions[left : left + batch_size]
            waveforms: list[np.ndarray] = []
            for position in batch_positions:
                row = requests[position]
                start = round(
                    (int(row["start_ms"]) - clip_start_ms) * 16
                )
                expected = duration * 16000
                waveform = signal[start : start + expected]
                if len(waveform) != expected:
                    raise BridgeV5Error("E_AUDIO_DECODE")
                rms[position] = math.sqrt(float(np.mean(np.square(waveform))))
                non_silent[position] = float(np.mean(np.abs(waveform) >= 1e-3))
                clipped[position] = float(np.mean(np.abs(waveform) >= 0.999))
                normalized = (waveform - waveform.mean()) / math.sqrt(
                    float(waveform.var()) + 1e-7
                )
                waveforms.append(normalized.astype(np.float32))
            tensor = torch.from_numpy(np.stack(waveforms)).to(device)
            with torch.inference_mode():
                hidden = audio_model(tensor).last_hidden_state
                mean = hidden.mean(dim=1)
                standard = hidden.std(dim=1, unbiased=False)
            values = (
                torch.stack((mean, standard), dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            audio_stats[batch_positions] = values

    targets_by_request: list[list[int]] = []
    all_targets: list[int] = []
    for row in requests:
        local_start = int(row["start_ms"]) - clip_start_ms
        targets = [
            local_start + 500 + second * 1000
            for second in range(int(row["duration_seconds"]))
        ]
        targets_by_request.append(targets)
        all_targets.extend(targets)
    frames = _decode_target_frames(clip_path, all_targets)
    frame_embeddings: dict[int, np.ndarray] = {}
    unique_targets = sorted(frames)
    for left in range(0, len(unique_targets), 32):
        batch_targets = unique_targets[left : left + 32]
        pixels = vision_processor(
            images=[frames[target] for target in batch_targets],
            return_tensors="pt",
        ).pixel_values.to(device)
        with torch.inference_mode():
            embedded = vision_model(pixel_values=pixels).pooler_output
        values = embedded.detach().cpu().numpy().astype(np.float32)
        for target, value in zip(batch_targets, values):
            frame_embeddings[target] = value
    for position, targets in enumerate(targets_by_request):
        values = np.stack([frame_embeddings[target] for target in targets])
        vision_stats[position, 0] = values.mean(axis=0)
        vision_stats[position, 1] = values.std(axis=0)
        grayscale = [
            np.asarray(frames[target].convert("L").resize((64, 64)), dtype=np.float32)
            / 255.0
            for target in targets
        ]
        differences = np.asarray(
            [
                float(np.mean(np.abs(right - left)))
                for left, right in zip(grayscale, grayscale[1:])
            ],
            dtype=np.float32,
        )
        motion[position] = float(differences.mean()) if differences.size else 0.0
        if len(values) > 1:
            normalized = values / np.maximum(
                np.linalg.norm(values, axis=1, keepdims=True), 1e-12
            )
            adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
            persistence[position] = float(adjacent.mean())
        else:
            persistence[position] = 1.0
        scene_change[position] = (
            float(np.mean(differences >= 0.2)) if differences.size else 0.0
        )
    return {
        "indices": indices,
        "audio_stats": audio_stats,
        "vision_stats": vision_stats,
        "rms": rms,
        "non_silent": non_silent,
        "clipped": clipped,
        "motion": motion,
        "persistence": persistence,
        "scene_change": scene_change,
    }


def _extract_frontends_worker(
    runtime: Path,
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoImageProcessor, Dinov2Model, Wav2Vec2Model

    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    audio_model = Wav2Vec2Model.from_pretrained(
        AUDIO_MODEL, local_files_only=True
    ).eval().to(device)
    vision_processor = AutoImageProcessor.from_pretrained(
        VISION_MODEL, local_files_only=True, use_fast=False
    )
    vision_model = Dinov2Model.from_pretrained(
        VISION_MODEL, local_files_only=True
    ).eval().to(device)
    shard_directory = runtime / PRIVATE_FEATURE_SHARDS
    shard_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(shard_directory, 0o700)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for request in requests:
        grouped[str(request["clip_relative_path"])].append(request)
    for relative, clip_requests in sorted(grouped.items()):
        shard_key = keyed_hash(CLEAN_RUN_ID, relative)
        shard_path = shard_directory / f"{shard_key}.npz"
        if shard_path.is_file() and _private_file(shard_path):
            with np.load(shard_path, allow_pickle=False) as existing:
                if (
                    existing["indices"].tolist()
                    == [int(row["index"]) for row in clip_requests]
                    and existing["audio_stats"].shape
                    == (len(clip_requests), 2, 1024)
                    and existing["vision_stats"].shape
                    == (len(clip_requests), 2, 384)
                ):
                    continue
        clip_path = (runtime / relative).resolve()
        if not _inside(clip_path, runtime) or not _private_file(clip_path):
            raise BridgeV5Error("E_REQUEST_CLIP_BINDING")
        values = _extract_feature_shard(
            clip_path=clip_path,
            clip_start_ms=int(clip_requests[0]["clip_start_ms"]),
            requests=clip_requests,
            audio_model=audio_model,
            vision_model=vision_model,
            vision_processor=vision_processor,
            torch=torch,
            device=device,
        )
        _write_private_npz(shard_path, **values)
        if device.type == "mps":
            torch.mps.empty_cache()
    total = len(requests)
    result = {
        "audio_stats": np.empty((total, 2, 1024), dtype=np.float32),
        "vision_stats": np.empty((total, 2, 384), dtype=np.float32),
        "rms": np.empty(total, dtype=np.float32),
        "non_silent": np.empty(total, dtype=np.float32),
        "clipped": np.empty(total, dtype=np.float32),
        "motion": np.empty(total, dtype=np.float32),
        "persistence": np.empty(total, dtype=np.float32),
        "scene_change": np.empty(total, dtype=np.float32),
    }
    seen: set[int] = set()
    for shard_path in sorted(shard_directory.glob("*.npz")):
        if not _private_file(shard_path):
            raise BridgeV5Error("E_PRIVATE_FEATURES")
        with np.load(shard_path, allow_pickle=False) as shard:
            indices = shard["indices"].astype(int)
            if any(index in seen or index < 0 or index >= total for index in indices):
                raise BridgeV5Error("E_PRIVATE_FEATURES")
            seen.update(indices.tolist())
            for key in result:
                result[key][indices] = shard[key]
    if seen != set(range(total)):
        raise BridgeV5Error("E_PREPROCESSING_COMPLETION")
    return result


def _sequence_from_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise BridgeV5Error("E_FEATURE_SHAPE")
    return np.stack(
        (values[:, 0] - values[:, 1], values[:, 0] + values[:, 1]), axis=1
    )


def _train_scale_balanced_projectors(
    audio: np.ndarray,
    vision: np.ndarray,
    participants: Sequence[str],
    durations: Sequence[int],
    *,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    import torch

    audio = _sequence_from_stats(audio)
    vision = _sequence_from_stats(vision)
    grouped: dict[str, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (participant, duration) in enumerate(zip(participants, durations)):
        grouped[str(participant)][int(duration)].append(index)
    if any(
        set(by_duration) != {2, 6, 18} or any(not rows for rows in by_duration.values())
        for by_duration in grouped.values()
    ):
        raise BridgeV5Error("E_TRAIN_SUPPORT")

    class Projector(torch.nn.Module):
        def __init__(self, dimension: int):
            super().__init__()
            self.norm = torch.nn.LayerNorm(2 * dimension)
            self.first = torch.nn.Linear(2 * dimension, 256)
            self.dropout = torch.nn.Dropout(0.1)
            self.second = torch.nn.Linear(256, 128)

        def forward(self, values):
            pooled = torch.cat(
                (values.mean(dim=1), values.std(dim=1, unbiased=False)), dim=1
            )
            hidden = torch.nn.functional.gelu(self.first(self.norm(pooled)))
            return torch.nn.functional.normalize(
                self.second(self.dropout(hidden)), dim=1
            )

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    audio_projector = Projector(1024)
    vision_projector = Projector(384)
    optimizer = torch.optim.AdamW(
        [*audio_projector.parameters(), *vision_projector.parameters()],
        lr=0.0003,
        weight_decay=0.01,
    )
    audio_tensor = torch.from_numpy(audio)
    vision_tensor = torch.from_numpy(vision)
    participant_order = sorted(grouped)
    scales = (2, 6, 18)
    for epoch in range(80):
        for step in range(12):
            selected: list[int] = []
            for participant_index, participant in enumerate(participant_order):
                scale = scales[(epoch * 12 + step + participant_index) % 3]
                candidates = grouped[participant][scale]
                choice = int(
                    keyed_hash(str(seed), epoch, step, participant, scale), 16
                ) % len(candidates)
                selected.append(candidates[choice])
            projected_audio = audio_projector(audio_tensor[selected])
            projected_vision = vision_projector(vision_tensor[selected])
            logits = projected_audio @ projected_vision.T / 0.07
            labels = torch.arange(len(selected))
            loss = 0.5 * (
                torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*audio_projector.parameters(), *vision_projector.parameters()],
                1.0,
            )
            optimizer.step()

    def export(module) -> dict[str, np.ndarray]:
        return {
            name: value.detach().cpu().numpy().copy()
            for name, value in module.state_dict().items()
        }

    return {"audio": export(audio_projector), "vision": export(vision_projector)}


def _cluster_interval_values(
    values: Sequence[float], *, seed: int, confidence: float = 0.9
) -> list[float]:
    clean = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if clean.size < 5:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, clean.size, size=(10_000, clean.size))
    estimates = clean[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    return [
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1 - alpha)),
    ]


def _metric_summary(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    clean = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if clean.size < 5:
        return {"status": "SUPPRESSED_K_LT_5"}
    interval = _cluster_interval_values(clean.tolist(), seed=seed)
    return {
        "participant_count": int(clean.size),
        "participant_mean": round(float(clean.mean()), 4),
        "participant_cluster_90pct": [round(value, 4) for value in interval],
        "participant_quantiles_10_50_90": [
            round(float(value), 4)
            for value in np.quantile(clean, (0.1, 0.5, 0.9))
        ],
    }


def _finite_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(child) for child in value]
    return value


def _nuisance_standardization(
    requests: Sequence[Mapping[str, Any]], features: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    log_rms = np.log(np.maximum(features["rms"], 1e-8))
    motion = np.asarray(features["motion"], dtype=np.float64)
    z_energy = np.empty(len(requests), dtype=np.float64)
    z_motion = np.empty(len(requests), dtype=np.float64)
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for request in requests:
        grouped[
            (
                str(request["participant_key"]),
                int(request["duration_seconds"]),
            )
        ].append(int(request["index"]))
    for indices in grouped.values():
        for values, target in ((log_rms, z_energy), (motion, z_motion)):
            selected = values[indices]
            standard = max(float(selected.std()), 1e-8)
            target[indices] = (selected - float(selected.mean())) / standard
    return z_energy, z_motion


def _nuisance_match(
    left_index: int,
    right_index: int,
    requests: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
    z_energy: np.ndarray,
    z_motion: np.ndarray,
) -> bool:
    left = requests[left_index]
    right = requests[right_index]
    return bool(
        int(left["duration_seconds"]) == int(right["duration_seconds"])
        and str(left["activity"]) == str(right["activity"])
        and str(left["location"]) == str(right["location"])
        and abs(
            float(left["released_speech_support_fraction"])
            - float(right["released_speech_support_fraction"])
        )
        <= 0.1
        and abs(float(z_energy[left_index] - z_energy[right_index])) <= 0.25
        and abs(float(z_motion[left_index] - z_motion[right_index])) <= 0.25
        and abs(
            float(
                features["persistence"][left_index]
                - features["persistence"][right_index]
            )
        )
        <= 0.05
    )


def _fold_heterogeneity(
    fold_participants: Mapping[int, Sequence[float]]
) -> tuple[float, float, list[float]]:
    means: list[float] = []
    variances: list[float] = []
    for fold in range(3):
        values = np.asarray(fold_participants.get(fold, []), dtype=np.float64)
        values = values[np.isfinite(values)]
        means.append(float(values.mean()) if values.size else math.nan)
        variances.append(
            float(values.var(ddof=1) / values.size)
            if values.size > 1
            else math.nan
        )
    if any(not math.isfinite(value) for value in means + variances):
        return math.inf, math.inf, means
    weights = 1 / np.maximum(np.asarray(variances), 1e-12)
    pooled = float(np.sum(weights * means) / weights.sum())
    q = float(np.sum(weights * np.square(np.asarray(means) - pooled)))
    i2 = 0.0 if q <= 0 else max(0.0, (q - 2) / q)
    return i2, max(means) - min(means), means


def _public_calibration_summary(
    *,
    selection: Mapping[str, Any],
    source_metadata: Mapping[str, Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    participant_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "speech_bouts": [],
            "speech_gaps": [],
            "speech_ms": 0,
            "observation_ms": 0,
            "activity_ms": Counter(),
            "location_ms": Counter(),
            "recurrences": 0,
        }
    )
    for participant in selection["participants"]:
        key = str(participant["participant_key"])
        prior_activity: str | None = None
        for core in participant["core_observation_intervals"]:
            source = str(core["source_object_key"])
            start = int(core["start_ms"])
            end = int(core["end_ms"])
            meta = source_metadata[source]
            metric = participant_metrics[key]
            metric["observation_ms"] += end - start
            metric["activity_ms"][meta["activity"]] += end - start
            metric["location_ms"][meta["location"]] += end - start
            if prior_activity is not None and prior_activity == meta["activity"]:
                metric["recurrences"] += 1
            prior_activity = str(meta["activity"])
            clipped_bouts: list[tuple[int, int]] = []
            for left, right in meta["speech_windows_ms"]:
                overlap_left = max(start, int(left))
                overlap_right = min(end, int(right))
                if overlap_right > overlap_left:
                    clipped_bouts.append((overlap_left, overlap_right))
                    metric["speech_bouts"].append(
                        (overlap_right - overlap_left) / 1000
                    )
                    metric["speech_ms"] += overlap_right - overlap_left
            metric["speech_gaps"].extend(
                (right[0] - left[1]) / 1000
                for left, right in zip(clipped_bouts, clipped_bouts[1:])
                if right[0] > left[1]
            )
    zero_rows: dict[str, list[int]] = defaultdict(list)
    request_index = {
        str(request["request_key"]): int(request["index"])
        for request in requests
    }
    enriched_zero_rows: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        participant = str(row["participant_key"])
        index = request_index[str(row["lag_requests"]["0"])]
        zero_rows[participant].append(index)
        if ENRICHED_ACTIVITY_RE.search(str(row["activity"])):
            enriched_zero_rows[participant].append(index)
    metrics: dict[str, list[float]] = defaultdict(list)
    activity_participants: dict[str, list[float]] = defaultdict(list)
    location_participants: dict[str, list[float]] = defaultdict(list)
    for participant, value in sorted(participant_metrics.items()):
        minutes = value["observation_ms"] / 60_000
        metrics["speech_bout_duration_seconds"].append(
            float(np.mean(value["speech_bouts"])) if value["speech_bouts"] else 0.0
        )
        metrics["speech_gap_seconds"].append(
            float(np.mean(value["speech_gaps"])) if value["speech_gaps"] else 0.0
        )
        metrics["speech_bout_density_per_minute"].append(
            len(value["speech_bouts"]) / minutes
        )
        metrics["speech_seconds_per_observation_minute"].append(
            value["speech_ms"] / 1000 / minutes
        )
        metrics["activity_recurrence_per_hour"].append(
            value["recurrences"] / (minutes / 60)
        )
        metrics["candidate_event_density_per_minute"].append(
            len(zero_rows[participant]) / minutes
        )
        indices = zero_rows[participant]
        metrics["audio_log_rms"].append(
            float(np.mean(np.log(np.maximum(features["rms"][indices], 1e-8))))
        )
        metrics["audio_non_silent_fraction"].append(
            float(np.mean(features["non_silent"][indices]))
        )
        metrics["audio_clipped_fraction"].append(
            float(np.mean(features["clipped"][indices]))
        )
        metrics["motion"].append(float(np.mean(features["motion"][indices])))
        metrics["adjacent_frame_persistence"].append(
            float(np.mean(features["persistence"][indices]))
        )
        metrics["scene_change_rate"].append(
            float(np.mean(features["scene_change"][indices]))
        )
    all_activity_labels = sorted(
        {
            label
            for value in participant_metrics.values()
            for label in value["activity_ms"]
        }
    )
    all_location_labels = sorted(
        {
            label
            for value in participant_metrics.values()
            for label in value["location_ms"]
        }
    )
    for value in participant_metrics.values():
        for label in all_activity_labels:
            activity_participants[label].append(
                value["activity_ms"].get(label, 0) / value["observation_ms"]
            )
        for label in all_location_labels:
            location_participants[label].append(
                value["location_ms"].get(label, 0) / value["observation_ms"]
            )
    activity = {
        label: _metric_summary(values, seed=20260800 + index)
        for index, (label, values) in enumerate(sorted(activity_participants.items()))
        if len(values) >= 5
    }
    location = {
        label: _metric_summary(values, seed=20260900 + index)
        for index, (label, values) in enumerate(sorted(location_participants.items()))
        if len(values) >= 5
    }
    enriched_participants = {
        participant for participant, indices in enriched_zero_rows.items() if indices
    }
    enriched_metrics: dict[str, list[float]] = defaultdict(list)
    for participant, indices in enriched_zero_rows.items():
        if not indices:
            continue
        enriched_metrics["released_speech_support_fraction"].append(
            float(
                np.mean(
                    [
                        requests[index]["released_speech_support_fraction"]
                        for index in indices
                    ]
                )
            )
        )
        enriched_metrics["audio_log_rms"].append(
            float(np.mean(np.log(np.maximum(features["rms"][indices], 1e-8))))
        )
        enriched_metrics["motion"].append(
            float(np.mean(features["motion"][indices]))
        )
        enriched_metrics["adjacent_frame_persistence"].append(
            float(np.mean(features["persistence"][indices]))
        )
    return {
        "schema_version": "childlens-v5-model-independent-calibration-v1.0.0",
        "status": "COMPLETE",
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "observation_hours": 4.5,
        "representative_stratum_role": "PRIMARY",
        "representative_participant_clustered": {
            name: _metric_summary(values, seed=20260750 + index)
            for index, (name, values) in enumerate(sorted(metrics.items()))
        },
        "grounding_enriched_recording_level_fallback": {
            "participant_count": len(enriched_participants),
            "status": (
                "REPORTED_CONDITIONALLY"
                if len(enriched_participants) >= 5
                else "SUPPRESSED_K_LT_5"
            ),
            "participant_clustered": (
                {
                    name: _metric_summary(values, seed=20261100 + index)
                    for index, (name, values) in enumerate(
                        sorted(enriched_metrics.items())
                    )
                }
                if len(enriched_participants) >= 5
                else {}
            ),
            "not_used_to_replace_natural_mixture": True,
        },
        "natural_activity_mixture_weights_k_safe": activity,
        "coarse_location_shares_k_safe": location,
        "speech_activity_overlap_role": (
            "speech seconds and activity shares measured over identical "
            "representative observation time"
        ),
        "privacy": {
            "minimum_cell_participants": 5,
            "complementary_suppression_applied": True,
            "restricted_rows_or_identifiers_exported": False,
        },
    }


def _analyze_features_worker(
    *,
    selection: Mapping[str, Any],
    source_metadata: Mapping[str, Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    features: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    request_index = {
        str(request["request_key"]): int(request["index"])
        for request in requests
    }
    z_energy, z_motion = _nuisance_standardization(requests, features)
    row_matches: dict[str, dict[str, bool]] = {}
    support_counts: dict[tuple[str, int, int], int] = Counter()
    for row in rows:
        zero = request_index[str(row["lag_requests"]["0"])]
        matches: dict[str, bool] = {}
        for lag, request_key in row["lag_requests"].items():
            control = request_index[str(request_key)]
            matched = lag == "0" or _nuisance_match(
                zero, control, requests, features, z_energy, z_motion
            )
            matches[str(lag)] = matched
            if matched:
                support_counts[
                    (
                        str(row["participant_key"]),
                        int(row["duration_seconds"]),
                        int(lag),
                    )
                ] += 1
        row_matches[str(row["row_key"])] = matches

    states: dict[tuple[int, int], dict[str, dict[str, np.ndarray]]] = {}
    weight_arrays: dict[str, np.ndarray] = {}
    seeds = (20260723, 20260724, 20260725)
    for fold in range(3):
        training = [row for row in rows if int(row["fold"]) != fold]
        training_indices = [
            request_index[str(row["lag_requests"]["0"])] for row in training
        ]
        for seed in seeds:
            state = _train_scale_balanced_projectors(
                features["audio_stats"][training_indices],
                features["vision_stats"][training_indices],
                [str(row["participant_key"]) for row in training],
                [int(row["duration_seconds"]) for row in training],
                seed=seed,
            )
            states[(fold, seed)] = state
            for modality, parameters in state.items():
                for name, values in parameters.items():
                    safe_name = name.replace(".", "_")
                    weight_arrays[
                        f"fold{fold}_seed{seed}_{modality}_{safe_name}"
                    ] = values

    row_scores: dict[str, dict[str, float]] = {}
    shuffle_scores: dict[str, float] = {}
    fold_curves: dict[int, list[float]] = {}
    participant_curves: dict[str, dict[int, dict[int, float]]] = {}
    participant_primary: dict[str, float] = {}
    participant_amplitude: dict[str, float] = {}
    participant_shuffle: dict[str, float] = {}
    participant_fold: dict[str, int] = {}
    participant_asymmetry: dict[str, float] = {}
    fold_participant_primary: dict[int, list[float]] = defaultdict(list)
    all_lags = {
        2: (-8, -4, -2, 0, 2, 4, 8),
        6: (-24, -12, -6, 0, 6, 12, 24),
        18: (-72, -36, -18, 0, 18, 36, 72),
    }
    for fold in range(3):
        evaluation = [row for row in rows if int(row["fold"]) == fold]
        pair_rows: list[tuple[Mapping[str, Any], int]] = []
        audio_indices: list[int] = []
        visual_indices: list[int] = []
        for row in evaluation:
            zero = request_index[str(row["lag_requests"]["0"])]
            for lag, request_key in row["lag_requests"].items():
                pair_rows.append((row, int(lag)))
                audio_indices.append(zero)
                visual_indices.append(request_index[str(request_key)])
        seed_scores: list[np.ndarray] = []
        for seed in seeds:
            seed_scores.append(
                score_temporal_projectors(
                    _sequence_from_stats(features["audio_stats"][audio_indices]),
                    _sequence_from_stats(features["vision_stats"][visual_indices]),
                    states[(fold, seed)],
                )
            )
        mean_scores = np.mean(np.stack(seed_scores), axis=0)
        for (row, lag), score in zip(pair_rows, mean_scores):
            row_scores.setdefault(str(row["row_key"]), {})[str(lag)] = float(score)

        for duration in (2, 6, 18):
            duration_rows = [
                row for row in evaluation if int(row["duration_seconds"]) == duration
            ]
            used: set[str] = set()
            donor_for: dict[str, Mapping[str, Any]] = {}
            for row in sorted(
                duration_rows,
                key=lambda value: keyed_hash(
                    CLEAN_RUN_ID, "shuffle", str(value["row_key"])
                ),
            ):
                left = request_index[str(row["lag_requests"]["0"])]
                candidates: list[tuple[float, str, Mapping[str, Any]]] = []
                for donor in duration_rows:
                    if (
                        str(donor["participant_key"]) == str(row["participant_key"])
                        or str(donor["row_key"]) in used
                    ):
                        continue
                    right = request_index[str(donor["lag_requests"]["0"])]
                    if not _nuisance_match(
                        left, right, requests, features, z_energy, z_motion
                    ):
                        continue
                    distance = (
                        float(z_energy[left] - z_energy[right]) ** 2
                        + float(z_motion[left] - z_motion[right]) ** 2
                        + float(
                            features["persistence"][left]
                            - features["persistence"][right]
                        )
                        ** 2
                    )
                    candidates.append(
                        (
                            distance,
                            keyed_hash(
                                CLEAN_RUN_ID,
                                "shuffle-tie",
                                str(row["row_key"]),
                                str(donor["row_key"]),
                            ),
                            donor,
                        )
                    )
                if candidates:
                    donor = min(candidates, key=lambda value: (value[0], value[1]))[2]
                    donor_for[str(row["row_key"])] = donor
                    used.add(str(donor["row_key"]))
            if donor_for:
                recipients = [
                    row
                    for row in duration_rows
                    if str(row["row_key"]) in donor_for
                ]
                shuffle_audio = [
                    request_index[str(row["lag_requests"]["0"])]
                    for row in recipients
                ]
                shuffle_visual = [
                    request_index[
                        str(
                            donor_for[str(row["row_key"])]["lag_requests"]["0"]
                        )
                    ]
                    for row in recipients
                ]
                values = []
                for seed in seeds:
                    values.append(
                        score_temporal_projectors(
                            _sequence_from_stats(
                                features["audio_stats"][shuffle_audio]
                            ),
                            _sequence_from_stats(
                                features["vision_stats"][shuffle_visual]
                            ),
                            states[(fold, seed)],
                        )
                    )
                for row, score in zip(recipients, np.mean(np.stack(values), axis=0)):
                    shuffle_scores[str(row["row_key"])] = float(score)

    for participant in selection["participants"]:
        participant_key = str(participant["participant_key"])
        fold = int(participant["evaluation_fold"])
        participant_fold[participant_key] = fold
        participant_rows = [
            row for row in rows if str(row["participant_key"]) == participant_key
        ]
        curves: dict[int, dict[int, float]] = {}
        scale_primary: list[float] = []
        scale_amplitude: list[float] = []
        scale_asymmetry: list[float] = []
        shuffle_differences: list[float] = []
        for duration in (2, 6, 18):
            duration_rows = [
                row
                for row in participant_rows
                if int(row["duration_seconds"]) == duration
            ]
            curve: dict[int, float] = {}
            lag_differences: dict[int, float] = {}
            for lag in all_lags[duration]:
                selected = [
                    row
                    for row in duration_rows
                    if row_matches[str(row["row_key"])][str(lag)]
                ]
                values = [
                    row_scores[str(row["row_key"])][str(lag)] for row in selected
                ]
                curve[lag] = float(np.mean(values)) if values else math.nan
                if lag:
                    differences = [
                        row_scores[str(row["row_key"])]["0"]
                        - row_scores[str(row["row_key"])][str(lag)]
                        for row in selected
                    ]
                    lag_differences[lag] = (
                        float(np.mean(differences)) if differences else math.nan
                    )
            curves[duration] = curve
            long = [
                lag_differences[lag]
                for lag in (-4 * duration, -2 * duration, 2 * duration, 4 * duration)
                if math.isfinite(lag_differences.get(lag, math.nan))
            ]
            if len(long) == 4:
                scale_primary.append(float(np.mean(long)))
            finite_curve = [
                value for value in curve.values() if math.isfinite(value)
            ]
            if len(finite_curve) == 7:
                scale_amplitude.append(max(finite_curve) - min(finite_curve))
            negative = [
                curve[-duration],
                curve[-2 * duration],
                curve[-4 * duration],
            ]
            positive = [
                curve[duration],
                curve[2 * duration],
                curve[4 * duration],
            ]
            if all(math.isfinite(value) for value in negative + positive):
                scale_asymmetry.append(
                    float(np.mean(positive) - np.mean(negative))
                )
            shuffle_differences.extend(
                row_scores[str(row["row_key"])]["0"]
                - shuffle_scores[str(row["row_key"])]
                for row in duration_rows
                if str(row["row_key"]) in shuffle_scores
            )
        participant_curves[participant_key] = curves
        participant_primary[participant_key] = (
            float(np.mean(scale_primary)) if len(scale_primary) == 3 else math.nan
        )
        participant_amplitude[participant_key] = (
            float(np.mean(scale_amplitude))
            if len(scale_amplitude) == 3
            else math.nan
        )
        participant_asymmetry[participant_key] = (
            float(np.mean(scale_asymmetry))
            if len(scale_asymmetry) == 3
            else math.nan
        )
        participant_shuffle[participant_key] = (
            float(np.mean(shuffle_differences))
            if shuffle_differences
            else math.nan
        )
        fold_participant_primary[fold].append(participant_primary[participant_key])

    curve_coordinates = [
        (duration, lag)
        for duration in (2, 6, 18)
        for lag in all_lags[duration]
    ]
    for fold in range(3):
        fold_curves[fold] = [
            float(
                np.nanmean(
                    [
                        participant_curves[participant][duration][lag]
                        for participant in participant_curves
                        if participant_fold[participant] == fold
                    ]
                )
            )
            for duration, lag in curve_coordinates
        ]
    fold_curve_correlations: list[float] = []
    for fold in range(3):
        left = np.asarray(fold_curves[fold])
        right = np.nanmean(
            np.asarray([fold_curves[other] for other in range(3) if other != fold]),
            axis=0,
        )
        valid = np.isfinite(left) & np.isfinite(right)
        fold_curve_correlations.append(
            float(np.corrcoef(left[valid], right[valid])[0, 1])
            if valid.sum() >= 3
            else math.nan
        )

    primary_values = list(participant_primary.values())
    finite_primary_values = [
        value for value in primary_values if math.isfinite(value)
    ]
    primary_interval = _cluster_interval_values(primary_values, seed=20260726)
    primary_mean = float(np.nanmean(primary_values))
    amplitude_values = list(participant_amplitude.values())
    amplitude_interval = _cluster_interval_values(
        amplitude_values, seed=20260727
    )
    i2, fold_range, fold_means = _fold_heterogeneity(
        fold_participant_primary
    )
    expected_support_cells = {
        (
            str(participant["participant_key"]),
            duration,
            lag,
        )
        for participant in selection["participants"]
        for duration in (2, 6, 18)
        for lag in all_lags[duration]
        if lag != 0
    }
    minimum_support = min(
        (support_counts.get(cell, 0) for cell in expected_support_cells),
        default=0,
    )
    support_gate = bool(
        expected_support_cells.issubset(support_counts)
        and min(support_counts[cell] for cell in expected_support_cells) >= 40
    )
    shuffle_supported = sum(
        math.isfinite(value) for value in participant_shuffle.values()
    )
    shared = {
        "governance": True,
        "support": support_gate,
        "positive_control": bool(_read_json(CLEAN_POSITIVE_RECEIPT)["pass"]),
        "preprocessing": True,
        "shortcut_interpretable": bool(
            all(
                all(math.isfinite(value) for value in fold_curves[fold])
                for fold in range(3)
            )
            and shuffle_supported >= 5
        ),
        "precision": bool(primary_interval[1] - primary_interval[0] <= 0.04),
        "heterogeneity": bool(i2 <= 0.5 and fold_range <= 0.03),
    }
    detectable = bool(
        primary_mean >= 0.02
        and primary_interval[0] > 0
        and sum(value > 0 for value in primary_values if math.isfinite(value)) >= 12
        and all(value > 0 for value in fold_means)
        and all(value >= 0.5 for value in fold_curve_correlations)
    )
    weak_or_flat = bool(
        primary_interval[0] >= -0.02
        and primary_interval[1] <= 0.02
        and amplitude_interval[1] <= 0.02
    )
    gates = {
        **shared,
        "detectable_structure": detectable,
        "precise_weak_or_flat": weak_or_flat,
    }
    decision = terminal_decision(gates)
    public_fold_means = [
        (
            round(float(np.mean(finite)), 4)
            if len(
                finite := [
                    value
                    for value in fold_participant_primary.get(fold, [])
                    if math.isfinite(value)
                ]
            )
            >= 5
            else None
        )
        for fold in range(3)
    ]
    public_fold_correlations = []
    for fold in range(3):
        participant_keys = [
            participant
            for participant in participant_curves
            if participant_fold[participant] == fold
        ]
        minimum_curve_participants = min(
            sum(
                math.isfinite(participant_curves[key][duration][lag])
                for key in participant_keys
            )
            for duration, lag in curve_coordinates
        )
        public_fold_correlations.append(
            round(fold_curve_correlations[fold], 4)
            if minimum_curve_participants >= 5
            else None
        )

    public_curve: dict[str, list[dict[str, Any]]] = {}
    for duration in (2, 6, 18):
        public_curve[str(duration)] = []
        for lag in all_lags[duration]:
            values = [
                participant_curves[participant][duration][lag]
                for participant in participant_curves
            ]
            summary = _metric_summary(
                values, seed=20261000 + duration * 10 + lag
            )
            counts = [
                support_counts.get((participant, duration, lag), 0)
                for participant in participant_curves
            ]
            public_curve[str(duration)].append(
                {
                    "signed_lag_seconds": lag,
                    "cosine": summary,
                    "minimum_matched_windows_per_participant": min(counts),
                    "median_matched_windows_per_participant": round(
                        float(np.median(counts)), 1
                    ),
                    "maximum_matched_windows_per_participant": max(counts),
                    "participants_meeting_frozen_40_window_gate": sum(
                        value >= 40 for value in counts
                    ),
                }
            )
    lag_report = {
        "schema_version": "childlens-v5-cross-validated-lag-response-v1.0.0",
        "status": "COMPLETE",
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "fold_participant_counts": [6, 6, 6],
        "learner_seeds": list(seeds),
        "frontends_frozen": True,
        "learner_trainable_parameter_count": 792832,
        "alignment_versus_signed_lag": public_curve,
        "primary_zero_minus_signed_2x_4x": {
            **_metric_summary(primary_values, seed=20260726),
            "participant_cluster_90pct": [
                round(value, 4) for value in primary_interval
            ],
        },
        "signed_lag_asymmetry": _metric_summary(
            list(participant_asymmetry.values()), seed=20260728
        ),
        "curve_amplitude": {
            **_metric_summary(amplitude_values, seed=20260727),
            "participant_cluster_90pct": [
                round(value, 4) for value in amplitude_interval
            ],
        },
        "zero_minus_participant_excluding_shuffle": _metric_summary(
            list(participant_shuffle.values()), seed=20260729
        ),
        "fold_primary_contrasts": public_fold_means,
        "fold_curve_correlations_with_other_folds": public_fold_correlations,
        "cross_fold_i2": round(i2, 4) if math.isfinite(i2) else None,
        "fold_primary_max_minus_min": (
            round(fold_range, 4) if math.isfinite(fold_range) else None
        ),
        "minimum_matched_windows_across_observed_cells": minimum_support,
        "shuffle_evaluable_participants": shuffle_supported,
        "short_lag_persistent_scene_audit": _metric_summary(
            [
                float(
                    np.mean(
                        [
                            features["persistence"][
                                request_index[
                                    str(
                                        row["lag_requests"][
                                            str(int(row["duration_seconds"]))
                                        ]
                                    )
                                ]
                            ]
                            for row in rows
                            if str(row["participant_key"]) == participant
                        ]
                    )
                )
                for participant in participant_curves
            ],
            seed=20260730,
        ),
        "shuffle_is_secondary_not_standalone_evidence": True,
    }
    calibration = _public_calibration_summary(
        selection=selection,
        source_metadata=source_metadata,
        requests=requests,
        rows=rows,
        features=features,
    )
    decision_report = {
        "schema_version": "childlens-v5-clean-development-decision-v1.0.0",
        "status": "TERMINAL_DEVELOPMENT_DECISION",
        "decision": decision,
        "gates": gates,
        "primary_mean_lift": (
            round(primary_mean, 4) if len(finite_primary_values) >= 5 else None
        ),
        "primary_participant_cluster_90pct": [
            round(value, 4) for value in primary_interval
        ],
        "primary_evaluable_participants": (
            len(finite_primary_values)
            if len(finite_primary_values) >= 5
            else "SUPPRESSED_K_LT_5"
        ),
        "support_gate_minimum_required": 40,
        "support_gate_minimum_observed": minimum_support,
        "positive_participants": (
            sum(value > 0 for value in finite_primary_values)
            if len(finite_primary_values) >= 5
            else None
        ),
        "cross_fold_i2": round(i2, 4) if math.isfinite(i2) else None,
        "fold_primary_max_minus_min": (
            round(fold_range, 4) if math.isfinite(fold_range) else None
        ),
        "interpretation": (
            (
                "The frozen nuisance-matching support gate failed, leaving the "
                "participant-level primary contrast, precision, and heterogeneity "
                "not estimable at the preregistered support level. This is an "
                "uninformative development result, not evidence of absent true "
                "grounding."
            )
            if not support_gate
            else (
                "This is a development-only distributional calibration result. "
                "It does not establish absence of true grounding when weak or "
                "flat, and it does not authorize simulation or side-cue training."
            )
        ),
        "passing_result_would_eventually_permit": (
            "After separate locked confirmation and separate authorization, "
            "simulation calibrated to the measured ChildLens ages-3-to-5 "
            "distribution could use these identical instruments and learner "
            "to estimate lift from synchronized training-only physical side cues."
        ),
        "passing_result_would_not_establish": [
            "infant calibration",
            "naturalistic German lexical grounding",
            "causal grounding in ChildLens",
            "real-world physical side-cue lift",
            "authorization to generate simulator episodes or train side-cue arms",
        ],
        "locked_confirmation_run": False,
        "simulator_or_side_cue_training_run": False,
    }
    for value in (lag_report, calibration, decision_report):
        public_guard(value)
    restricted = {
        "schema_version": "childlens-v5-restricted-development-result-v1.0.0",
        "clean_run_id": CLEAN_RUN_ID,
        "participant_primary": participant_primary,
        "participant_amplitude": participant_amplitude,
        "participant_shuffle": participant_shuffle,
        "participant_curves": participant_curves,
        "support_counts": {
            "|".join(map(str, key)): value for key, value in support_counts.items()
        },
        "public": {
            "calibration": calibration,
            "lag_report": lag_report,
            "decision": decision_report,
        },
    }
    return _finite_json(restricted), weight_arrays


def clean_worker() -> dict[str, Any]:
    if os.environ.get("CHILDLENS_V5_NETWORK_DENIED") != "1":
        raise BridgeV5Error("E_WORKER_SANDBOX")
    _verify_frozen_models(write=False)
    runtime, scientific_input, _ = _attested_input()
    selection_path = runtime / PRIVATE_STAGE0_SELECTION
    checkpoint_path = runtime / PRIVATE_ACQUISITION_CHECKPOINT
    if (
        not _private_file(selection_path)
        or not _private_file(checkpoint_path)
        or _sha256_file(selection_path)
        != _read_json(CLEAN_STAGE0_RECEIPT)["private_selection_sha256"]
        or _sha256_file(checkpoint_path)
        != _read_json(CLEAN_ACQUISITION_RECEIPT)["restricted_checkpoint_sha256"]
    ):
        raise BridgeV5Error("E_CLEAN_RUN_BINDING")
    selection = _read_json(selection_path)
    checkpoint = _read_json(checkpoint_path)
    if (
        selection.get("development_participant_count") != 18
        or selection.get("locked_participant_count") != 0
        or checkpoint.get("status") != "COMPLETE"
    ):
        raise BridgeV5Error("E_CLEAN_RUN_BINDING")
    source_metadata = _source_scientific_metadata(runtime, scientific_input)
    requests, rows, _ = _measurement_inventory(
        runtime, selection, checkpoint, source_metadata
    )
    features = _extract_frontends_worker(runtime, requests)
    feature_path = runtime / PRIVATE_FEATURES
    _write_private_npz(feature_path, **features)
    feature_manifest = {
        "schema_version": "childlens-v5-restricted-feature-manifest-v1.0.0",
        "clean_run_id": CLEAN_RUN_ID,
        "requests": requests,
        "rows": rows,
        "embedding_completion_fraction": 1.0,
        "tokenizer_decoder_ctc_asr_translation_or_language_id_loaded": False,
    }
    feature_manifest_path = runtime / PRIVATE_FEATURE_MANIFEST
    _write_private(feature_manifest_path, feature_manifest)
    restricted, weights = _analyze_features_worker(
        selection=selection,
        source_metadata=source_metadata,
        requests=requests,
        rows=rows,
        features=features,
    )
    weights_path = runtime / PRIVATE_WEIGHTS
    _write_private_npz(weights_path, **weights)
    result_path = runtime / PRIVATE_RESULT
    _write_private(result_path, restricted)
    return {
        "status": "ok",
        "feature_count": len(requests),
        "row_count": len(rows),
        "restricted_features_sha256": _sha256_file(feature_path),
        "restricted_feature_manifest_sha256": _sha256_file(feature_manifest_path),
        "restricted_weights_sha256": _sha256_file(weights_path),
        "restricted_result_sha256": _sha256_file(result_path),
        "development_decision": restricted["public"]["decision"]["decision"],
    }


def _decision_markdown(value: Mapping[str, Any]) -> str:
    gates = value["gates"]
    primary = value["primary_mean_lift"]
    primary_text = (
        f"{float(primary):.4f}" if primary is not None else "privacy-suppressed"
    )
    interval = value["primary_participant_cluster_90pct"]
    interval_text = (
        str(interval)
        if all(bound is not None for bound in interval)
        else "not estimable at the frozen support gate"
    )
    return "\n".join(
        [
            "# ChildLens calibration recovery v5 — clean development decision",
            "",
            f"**Terminal state: `{value['decision']}`**",
            "",
            (
                "The unchanged frozen V5 route completed on the 18 development "
                "participants. All 22 locked participants remained sealed."
            ),
            "",
            (
                f"The participant-clustered primary mean lift was "
                f"{primary_text}, with a 90% interval of "
                f"{interval_text}. The minimum "
                f"matched support was {value['support_gate_minimum_observed']} "
                f"windows versus the frozen requirement of "
                f"{value['support_gate_minimum_required']}."
            ),
            "",
            "Frozen gate results:",
            "",
            *[
                f"- {name}: {'PASS' if passed else 'FAIL'}"
                for name, passed in gates.items()
            ],
            "",
            value["interpretation"],
            "",
            "A passing result would permit only a separately authorized locked "
            "confirmation, followed by separate authorization for the simulation "
            "bridge. No simulator or side-cue condition ran here.",
            "",
        ]
    )


def clean_run() -> dict[str, Any]:
    _verify_frozen_models(write=True)
    if (
        not CLEAN_ACQUISITION_RECEIPT.is_file()
        or _read_json(CLEAN_ACQUISITION_RECEIPT).get("status") != "COMPLETE"
    ):
        raise BridgeV5Error("E_ACQUISITION_INCOMPLETE")
    environment = dict(os.environ)
    environment["CHILDLENS_V5_NETWORK_DENIED"] = "1"
    process = subprocess.run(
        [
            str(SANDBOX_EXEC),
            "-p",
            SANDBOX_PROFILE,
            str(INSTRUMENT_PYTHON),
            str(Path(__file__).resolve()),
            "clean-worker",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=43_200,
        env=environment,
        text=True,
    )
    if process.returncode != 0:
        diagnostic = process.stderr[-4000:]
        runtime, _, _ = _attested_input()
        _write_private(
            runtime / PRIVATE_CLEAN_RELATIVE / "restricted_last_worker_failure.json",
            {
                "schema_version": "childlens-v5-private-worker-failure-v1.0.0",
                "returncode": process.returncode,
                "stderr_tail": diagnostic,
            },
        )
        raise BridgeV5Error("E_RESTRICTED_WORKER")
    try:
        worker = json.loads(process.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise BridgeV5Error("E_RESTRICTED_WORKER") from exc
    if worker.get("status") != "ok":
        raise BridgeV5Error("E_RESTRICTED_WORKER")
    runtime, _, _ = _attested_input()
    restricted_path = runtime / PRIVATE_RESULT
    if (
        not _private_file(restricted_path)
        or _sha256_file(restricted_path) != worker["restricted_result_sha256"]
    ):
        raise BridgeV5Error("E_PRIVATE_RESULT")
    restricted = _read_json(restricted_path)
    public = restricted["public"]
    for value in public.values():
        public_guard(value)
    _write(CLEAN_CALIBRATION_REPORT, public["calibration"])
    _write(CLEAN_LAG_REPORT, public["lag_report"])
    _write(CLEAN_DECISION_REPORT, public["decision"])
    CLEAN_DECISION_MARKDOWN.write_text(
        _decision_markdown(public["decision"]), encoding="utf-8"
    )
    validation = {
        "schema_version": "childlens-v5-clean-validation-v1.0.0",
        "status": "PASS",
        "clean_run_id": CLEAN_RUN_ID,
        "frozen_v5_config_sha256": FROZEN_CONFIG_SHA256,
        "scientific_runner_sha256": _sha256_file(Path(__file__)),
        "protocol_document_sha256": _sha256_file(PROTOCOL),
        "frozen_preflight_package_sha256": _sha256_file(PACKAGE),
        "original_incident_preserved": True,
        "original_incident_receipt_sha256": ORIGINAL_INCIDENT_RECEIPT_SHA256,
        "prior_v1_v2_v3_v4_immutable": True,
        "prior_tree_sha256": _validate_prior_trees(),
        "development_participant_count": 18,
        "locked_participant_count": 0,
        "legacy_mixed_scope_inputs_opened_by_scientific_process": False,
        "selective_acquisition_complete": True,
        "transient_full_sources_removed": True,
        "temporary_credential_deleted": True,
        "model_hash_receipt_sha256": _sha256_file(CLEAN_MODEL_RECEIPT),
        "restricted_features_sha256": worker["restricted_features_sha256"],
        "restricted_feature_manifest_sha256": worker[
            "restricted_feature_manifest_sha256"
        ],
        "restricted_weights_sha256": worker["restricted_weights_sha256"],
        "restricted_result_sha256": worker["restricted_result_sha256"],
        "public_calibration_sha256": _sha256_file(CLEAN_CALIBRATION_REPORT),
        "public_lag_report_sha256": _sha256_file(CLEAN_LAG_REPORT),
        "public_decision_sha256": _sha256_file(CLEAN_DECISION_REPORT),
        "development_decision": public["decision"]["decision"],
        "privacy_suppression_pass": True,
        "restricted_values_exported": False,
        "external_volume_aea_or_babyview_used": False,
        "locked_media_annotations_or_outcomes_accessed": False,
        "simulator_or_side_cue_training_run": False,
    }
    public_guard(validation)
    _write(CLEAN_VALIDATION_RECEIPT, validation)
    return validation


def freeze_and_stop() -> dict[str, Any]:
    if FREEZE_RECEIPT.exists():
        raise BridgeV5Error("E_IMMUTABLE_INCIDENT_RECORD")
    config = _config()
    prior = _validate_prior_trees()
    positive = run_embedding_positive_control(
        seed=config["positive_control"]["seed"],
        participants=config["positive_control"]["participants"],
        windows_per_participant=config["positive_control"]["windows_per_participant"],
        confidence=config["inference"]["confidence"],
    )
    _write(POSITIVE_RECEIPT, positive)
    gates = {
        "governance": False,
        "support": False,
        "positive_control": bool(positive["pass"]),
        "preprocessing": False,
        "shortcut_interpretable": False,
        "precision": False,
        "heterogeneity": False,
        "detectable_structure": False,
        "precise_weak_or_flat": False,
    }
    decision = terminal_decision(gates)
    freeze = {
        "schema_version": "childlens-calibration-recovery-stage0-v5.0.0",
        "status": "TERMINAL_FAIL_CLOSED_BEFORE_DATA_SELECTION_FREEZE",
        "protocol_created_before_new_media_decoding_or_childlens_outcomes": True,
        "protocol_sha256": _sha256_file(CONFIG),
        "protocol_document_sha256": _sha256_file(PROTOCOL),
        "package_sha256": _sha256_file(PACKAGE),
        "runner_sha256": _sha256_file(Path(__file__)),
        "prior_v1_v2_v3_v4_immutable": True,
        "prior_tree_sha256": prior,
        "development_only_manifest_bindings_from_immutable_v4_record": (
            EXPECTED_DEVELOPMENT_MANIFEST_HASHES
        ),
        "public_model_bindings": {
            modality: {
                "repository": binding["repository"],
                "revision": binding["revision"],
                "weights_sha256": binding["weights_sha256"],
                "config_sha256": binding["config_sha256"],
                "preprocessor_config_sha256": binding[
                    "preprocessor_config_sha256"
                ],
            }
            for modality, binding in config["instruments"].items()
            if modality in {"audio", "vision"}
        },
        "public_model_files_downloaded_for_v5": False,
        "local_model_file_hash_verification_required_before_any_future_execution": True,
        "data_selection_frozen": False,
        "source_object_bindings_or_exact_intervals_exported": False,
        "stage0_governance_gate_pass": False,
        "zero_locked_access_certifiable": False,
        "mixed_scope_legacy_manifests_parsed_during_inventory": 2,
        "locked_row_values_identifiers_intervals_media_or_outcomes_exported": False,
        "new_media_decoded": False,
        "new_media_acquired": False,
        "childlens_embeddings_computed": False,
        "childlens_training_or_scoring_run": False,
        "locked_media_decoded_or_scored": False,
        "external_volume_aea_or_babyview_used": False,
        "temporary_credentials_accessed_or_created": False,
        "transient_full_source_objects_created": 0,
        "simulator_or_side_cue_training_run": False,
        "positive_control_pass": bool(positive["pass"]),
        "development_decision": decision,
        "stop_reason": "ZERO_LOCKED_ROW_LOADING_CANNOT_BE_CERTIFIED_AFTER_MIXED_SCOPE_MANIFEST_PARSE",
    }
    public_guard(freeze)
    _write(FREEZE_RECEIPT, freeze)
    report = {
        "schema_version": "childlens-calibration-recovery-decision-v5.0.0",
        "decision": decision,
        "status": "TERMINAL_DEVELOPMENT_STOP_AT_STAGE0",
        "gates": gates,
        "childlens_audiovisual_estimates_exist": False,
        "multi_hour_calibration_summary_exists": False,
        "interpretation": "The run is uninformative because governance failed before development selection, acquisition, or audiovisual estimation; it is not evidence that ChildLens lacks alignment.",
        "passing_result_would_eventually_permit": "After separate locked confirmation and separate simulator authorization, a distribution-matched simulation could compare otherwise identical learning with and without synchronized training-only physical side cues.",
        "passing_result_would_not_establish": [
            "infant calibration",
            "naturalistic German lexical grounding",
            "causal grounding in ChildLens",
            "real-world physical side-cue lift",
            "authorization to generate a simulator or train side-cue conditions"
        ],
    }
    public_guard(report)
    _write(DECISION_REPORT, report)
    DECISION_MARKDOWN.write_text(
        "\n".join(
            [
                "# ChildLens calibration recovery v5 — development decision",
                "",
                "**Decision: `NO_GO_UNINFORMATIVE`**",
                "",
                "V5 stopped at Stage 0. The inventory process parsed legacy "
                "mixed-scope manifests, so zero locked-row loading cannot be "
                "certified under the prospective governance rule. No identifiers, "
                "intervals, media, or outcomes were exported, but this is still a "
                "fail-closed governance failure.",
                "",
                "No new ChildLens media was acquired or decoded. No ChildLens "
                "embedding, learner training, scoring, lag curve, calibration "
                "summary, locked evaluation, simulator generation, or side-cue "
                "condition ran. The synthetic sensitivity check passed but is not "
                "empirical evidence.",
                "",
                "This decision does not imply that ChildLens is intrinsically "
                "uninformative. A clean rerun would require a separately generated "
                "outer-scope receipt and input containing exactly the immutable 18 "
                "development participants and zero locked rows before the v5 "
                "process starts.",
                "",
                "A future passing development result could only recommend separate "
                "locked confirmation. After that confirmation and separate "
                "authorization, Michael Frank’s bridge would support a "
                "distribution-matched simulation comparing identical learning "
                "with and without synchronized training-only physical side cues. "
                "It would still not establish infant calibration, naturalistic "
                "German lexical grounding, causal ChildLens grounding, or "
                "real-world side-cue lift.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return freeze


def validate(*, write: bool = False) -> dict[str, Any]:
    _config()
    prior = _validate_prior_trees()
    for path in (FREEZE_RECEIPT, POSITIVE_RECEIPT, DECISION_REPORT):
        public_guard(_read_json(path))
    freeze = _read_json(FREEZE_RECEIPT)
    decision = _read_json(DECISION_REPORT)
    positive = _read_json(POSITIVE_RECEIPT)
    if (
        freeze.get("protocol_sha256") != _sha256_file(CONFIG)
        or freeze.get("protocol_document_sha256") != _sha256_file(PROTOCOL)
        or freeze.get("package_sha256") != _sha256_file(PACKAGE)
        or freeze.get("runner_sha256") != ORIGINAL_RUNNER_SHA256
        or freeze.get("prior_tree_sha256") != prior
        or freeze.get("public_model_bindings")
        != {
            modality: {
                "repository": binding["repository"],
                "revision": binding["revision"],
                "weights_sha256": binding["weights_sha256"],
                "config_sha256": binding["config_sha256"],
                "preprocessor_config_sha256": binding[
                    "preprocessor_config_sha256"
                ],
            }
            for modality, binding in _config()["instruments"].items()
            if modality in {"audio", "vision"}
        }
        or freeze.get("zero_locked_access_certifiable") is not False
        or freeze.get("new_media_decoded") is not False
        or freeze.get("childlens_training_or_scoring_run") is not False
        or decision.get("decision") != "NO_GO_UNINFORMATIVE"
        or positive.get("pass") is not True
    ):
        raise BridgeV5Error("E_VALIDATION_BINDING")
    receipt = {
        "schema_version": "childlens-calibration-recovery-validation-v5.0.0",
        "status": "PASS_FOR_TERMINAL_STAGE0_RECORD",
        "protocol_sha256": _sha256_file(CONFIG),
        "protocol_document_sha256": _sha256_file(PROTOCOL),
        "package_sha256": _sha256_file(PACKAGE),
        "runner_sha256": ORIGINAL_RUNNER_SHA256,
        "prior_v1_v2_v3_v4_immutable": True,
        "prior_tree_sha256": prior,
        "public_privacy_guard_pass": True,
        "public_model_revision_and_upstream_hash_bindings_verified": True,
        "local_model_file_hash_verification_deferred_due_stage0_stop": True,
        "minimum_public_cell_participants": 5,
        "complementary_suppression_required": True,
        "restricted_identifiers_intervals_media_embeddings_scores_or_weights_exported": False,
        "zero_locked_access_certifiable": False,
        "new_media_acquired_or_decoded": False,
        "childlens_outcome_analysis_run": False,
        "external_volume_aea_or_babyview_used": False,
        "selective_transfer_cleanup_required": False,
        "temporary_credentials_created_or_accessed": False,
        "synthetic_positive_control_pass": True,
        "development_decision": "NO_GO_UNINFORMATIVE",
    }
    public_guard(receipt)
    if write:
        if not VALIDATION_RECEIPT.is_file() or _read_json(VALIDATION_RECEIPT) != receipt:
            raise BridgeV5Error("E_IMMUTABLE_INCIDENT_RECORD")
    return receipt


def main() -> int:
    previous_umask = os.umask(0o077)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "command",
            choices=(
                "freeze-and-stop",
                "validate",
                "clean-stage0",
                "clean-acquire",
                "clean-run",
                "clean-worker",
            ),
            help="Validate immutable history or run the attested clean Stage 0.",
        )
        args = parser.parse_args()
        if args.command == "freeze-and-stop":
            result = freeze_and_stop()
        elif args.command == "clean-stage0":
            result = clean_stage0()
        elif args.command == "clean-acquire":
            result = clean_acquire()
        elif args.command == "clean-run":
            result = clean_run()
        elif args.command == "clean-worker":
            result = clean_worker()
        else:
            result = validate()
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    raise SystemExit(main())
