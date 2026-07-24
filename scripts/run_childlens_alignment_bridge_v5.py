#!/usr/bin/env python3
"""Generate, resume, and validate the bounded ChildLens v5 calibration."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any


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
    terminal_decision,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("freeze-and-stop", "validate", "clean-stage0"),
        help="Validate immutable history or run the attested clean Stage 0.",
    )
    args = parser.parse_args()
    if args.command == "freeze-and-stop":
        result = freeze_and_stop()
    elif args.command == "clean-stage0":
        result = clean_stage0()
    else:
        result = validate()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
