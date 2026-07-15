from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image
import yaml

from babyworld_lite.aea.config import load_aea_config, validate_aea_config
from babyworld_lite.aea.manifest import AEASequenceId


SCHEMA_VERSION = "aea-grounding-v1"
IMU_CHANNELS = (
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
)


@dataclass(frozen=True)
class SpeechWord:
    start_ns: int
    end_ns: int
    written: str
    normalized: str
    confidence: float

    @property
    def midpoint_ns(self) -> int:
        return (self.start_ns + self.end_ns) // 2


@dataclass(frozen=True)
class WindowLabel:
    start_ns: int
    end_ns: int
    anchor_ns: int
    action: str
    action_surface: str
    action_confidence: float
    object_noun: str
    object_surface: str
    object_confidence: float | None
    object_gap_seconds: float | None
    transcript: str


class SequenceProvider(Protocol):
    imu_stream_label: str

    def image_at(self, timestamp_ns: int) -> tuple[np.ndarray, int]: ...

    def imu_between(self, start_ns: int, end_ns: int) -> tuple[np.ndarray, np.ndarray]: ...


def normalize_word(value: str) -> str:
    return re.sub(r"[^a-z']", "", value.lower())


def read_speech(path: str | Path) -> list[SpeechWord]:
    rows: list[SpeechWord] = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            written = str(row["written"]).strip()
            normalized = normalize_word(written)
            if not normalized:
                continue
            rows.append(SpeechWord(
                start_ns=int(row["startTime_ns"]),
                end_ns=int(row["endTime_ns"]),
                written=written,
                normalized=normalized,
                confidence=float(row["confidence"]),
            ))
    rows.sort(key=lambda row: (row.start_ns, row.end_ns))
    return rows


def _reverse_lexicon(lexicon: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {
        normalize_word(surface): str(canonical)
        for canonical, surfaces in lexicon.items()
        for surface in surfaces
    }


def build_windows(words: Sequence[SpeechWord], config: Mapping[str, Any]) -> list[WindowLabel]:
    labels = config["labels"]
    window = config["window"]
    actions = _reverse_lexicon(labels["actions"])
    objects = _reverse_lexicon(labels["objects"])
    minimum_confidence = float(labels["minimum_asr_confidence"])
    duration_ns = round(float(window["duration_seconds"]) * 1e9)
    before_ns = round(float(window.get("seconds_before_anchor", window["duration_seconds"] / 2)) * 1e9)
    separation_ns = round(float(window["minimum_same_action_separation_seconds"]) * 1e9)
    object_gap_ns = round(float(labels["object_max_gap_seconds"]) * 1e9)
    max_per_action = int(window["max_windows_per_action_per_sequence"])
    max_total = int(window["max_windows_per_sequence"])
    object_words = [
        word for word in words
        if word.normalized in objects and word.confidence >= minimum_confidence
    ]
    counts: Counter[str] = Counter()
    last_anchor: dict[str, int] = {}
    output: list[WindowLabel] = []
    for word in words:
        action = actions.get(word.normalized)
        if action is None or word.confidence < minimum_confidence:
            continue
        if counts[action] >= max_per_action:
            continue
        if word.midpoint_ns - last_anchor.get(action, -10**30) < separation_ns:
            continue
        nearest: SpeechWord | None = None
        if object_words:
            candidate = min(object_words, key=lambda item: abs(item.midpoint_ns - word.midpoint_ns))
            if abs(candidate.midpoint_ns - word.midpoint_ns) <= object_gap_ns:
                nearest = candidate
        start_ns = word.midpoint_ns - before_ns
        end_ns = start_ns + duration_ns
        transcript_words = [
            item.written for item in words
            if item.end_ns >= start_ns and item.start_ns <= end_ns
        ]
        if not transcript_words:
            continue
        output.append(WindowLabel(
            start_ns=start_ns,
            end_ns=end_ns,
            anchor_ns=word.midpoint_ns,
            action=action,
            action_surface=word.normalized,
            action_confidence=word.confidence,
            object_noun=objects.get(nearest.normalized, "") if nearest else "",
            object_surface=nearest.normalized if nearest else "",
            object_confidence=nearest.confidence if nearest else None,
            object_gap_seconds=(
                abs(nearest.midpoint_ns - word.midpoint_ns) / 1e9 if nearest else None
            ),
            transcript=" ".join(transcript_words),
        ))
        counts[action] += 1
        last_anchor[action] = word.midpoint_ns
        if len(output) >= max_total:
            break
    return output


def resample_imu(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    start_ns: int,
    end_ns: int,
    rate_hz: float,
    maximum_gap_ms: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    samples = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or samples.ndim != 2 or samples.shape != (len(timestamps), 6):
        raise ValueError("raw IMU must have timestamps [N] and six-axis values [N, 6]")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0) or not np.isfinite(samples).all():
        raise ValueError("raw IMU timestamps/values are invalid")
    count = round((end_ns - start_ns) / 1e9 * rate_hz)
    if count < 2:
        raise ValueError("resampling grid is too short")
    grid = start_ns + np.arange(count, dtype=np.float64) / rate_hz * 1e9
    right = np.searchsorted(timestamps, grid, side="left")
    left = right - 1
    in_range = (left >= 0) & (right < len(timestamps))
    safe_left = np.clip(left, 0, len(timestamps) - 1)
    safe_right = np.clip(right, 0, len(timestamps) - 1)
    gaps = timestamps[safe_right] - timestamps[safe_left]
    valid = in_range & (gaps <= maximum_gap_ms * 1e6)
    result = np.stack([
        np.interp(grid, timestamps.astype(np.float64), samples[:, channel])
        for channel in range(6)
    ], axis=1).astype(np.float32)
    coverage = float(valid.mean())
    raw_rate = float((len(timestamps) - 1) / ((timestamps[-1] - timestamps[0]) / 1e9))
    return result, {
        "raw_sample_count": len(timestamps),
        "raw_rate_hz": raw_rate,
        "resampled_sample_count": len(result),
        "resampled_rate_hz": float(rate_hz),
        "coverage_fraction": coverage,
        "maximum_raw_gap_ms": float(np.diff(timestamps).max() / 1e6),
    }


class ProjectAriaSequenceProvider:
    """Thin optional-runtime adapter around the official VRS data provider."""

    def __init__(self, vrs_path: str | Path, imu_stream_label: str):
        try:
            from projectaria_tools.core import data_provider
            from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        except ImportError as exc:
            raise RuntimeError(
                "projectaria-tools is required for VRS preprocessing; install it in the "
                "dedicated Aria environment with: python3 -m pip install 'projectaria-tools[all]'"
            ) from exc
        self._TimeDomain = TimeDomain
        self._TimeQueryOptions = TimeQueryOptions
        self.provider = data_provider.create_vrs_data_provider(str(vrs_path))
        if self.provider is None:
            raise RuntimeError(f"could not open VRS: {vrs_path}")
        self.rgb_stream = self.provider.get_stream_id_from_label("camera-rgb")
        self.imu_stream = self.provider.get_stream_id_from_label(imu_stream_label)
        if self.rgb_stream is None or self.imu_stream is None:
            raise RuntimeError("required camera-rgb or configured IMU stream is absent")
        self.imu_stream_label = imu_stream_label
        self._imu_timestamps = np.asarray(
            self.provider.get_timestamps_ns(self.imu_stream, TimeDomain.DEVICE_TIME),
            dtype=np.int64,
        )

    def image_at(self, timestamp_ns: int) -> tuple[np.ndarray, int]:
        image, record = self.provider.get_image_data_by_time_ns(
            self.rgb_stream, int(timestamp_ns), self._TimeDomain.DEVICE_TIME,
            self._TimeQueryOptions.CLOSEST,
        )
        # Gen1 RGB pixels are stored rotated relative to upright viewing.
        array = np.rot90(image.to_numpy_array(), -1).copy()
        return array, int(record.capture_timestamp_ns)

    def imu_between(self, start_ns: int, end_ns: int) -> tuple[np.ndarray, np.ndarray]:
        left = max(0, int(np.searchsorted(self._imu_timestamps, start_ns, side="left")) - 1)
        right = min(
            len(self._imu_timestamps),
            int(np.searchsorted(self._imu_timestamps, end_ns, side="right")) + 1,
        )
        timestamps: list[int] = []
        values: list[list[float]] = []
        for index in range(left, right):
            sample = self.provider.get_imu_data_by_index(self.imu_stream, index)
            if not sample.accel_valid or not sample.gyro_valid:
                continue
            timestamps.append(int(sample.capture_timestamp_ns))
            values.append([
                *map(float, sample.accel_msec2),
                *map(float, sample.gyro_radsec),
            ])
        return np.asarray(timestamps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def _write_window(
    provider: SequenceProvider,
    label: WindowLabel,
    sequence_id: AEASequenceId,
    window_index: int,
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    window_id = f"{sequence_id.name}_w{window_index:04d}"
    relative = Path("windows") / sequence_id.name / window_id
    directory = output_root / relative
    raw_times, raw_values = provider.imu_between(label.start_ns, label.end_ns)
    imu, quality = resample_imu(
        raw_times, raw_values, label.start_ns, label.end_ns,
        float(config["window"]["imu_rate_hz"]),
        float(config["quality"]["maximum_imu_gap_ms"]),
    )
    if quality["coverage_fraction"] < float(config["quality"]["minimum_imu_coverage"]):
        raise ValueError(
            f"{window_id} IMU coverage {quality['coverage_fraction']:.3f} is below threshold"
        )

    frame_count = int(config["window"]["rgb_frames"])
    query_times = np.linspace(label.start_ns, label.end_ns, frame_count, endpoint=False)
    query_times += (label.end_ns - label.start_ns) / frame_count / 2
    frame_arrays: list[np.ndarray] = []
    capture_times: list[int] = []
    for timestamp in query_times:
        pixels, capture_ns = provider.image_at(int(timestamp))
        frame_arrays.append(np.asarray(pixels, dtype=np.uint8))
        capture_times.append(capture_ns)
    quality["maximum_frame_time_error_ms"] = float(
        np.max(np.abs(np.asarray(capture_times) - query_times)) / 1e6
    )
    if quality["maximum_frame_time_error_ms"] > float(
        config["quality"]["maximum_frame_time_error_ms"]
    ):
        raise ValueError(
            f"{window_id} RGB timestamp error "
            f"{quality['maximum_frame_time_error_ms']:.3f} ms is above threshold"
        )
    frames_dir = directory / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []
    for index, pixels in enumerate(frame_arrays):
        frame_path = frames_dir / f"{index:03d}.jpg"
        Image.fromarray(pixels).convert("RGB").save(
            frame_path, quality=int(config["storage"]["jpeg_quality"]), optimize=True
        )
        frame_paths.append(str(frame_path.relative_to(output_root)))

    imu_path = directory / "imu.npy"
    np.save(imu_path, imu, allow_pickle=False)
    event_group = sequence_id.event_group
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": window_id,
        "sequence_id": sequence_id.name,
        "event_group": event_group,
        "wearer_session_group": sequence_id.wearer_session_group,
        "location": sequence_id.location,
        "script": sequence_id.script,
        "sequence": sequence_id.sequence,
        "recording": sequence_id.recording,
        "window": {
            "start_device_time_ns": label.start_ns,
            "end_device_time_ns": label.end_ns,
            "anchor_device_time_ns": label.anchor_ns,
            "duration_seconds": (label.end_ns - label.start_ns) / 1e9,
        },
        "model_inputs": {
            "frame_paths": frame_paths,
            "frame_capture_device_time_ns": capture_times,
            "transcript": label.transcript,
            "imu_path": str(imu_path.relative_to(output_root)),
            "imu_channels": list(IMU_CHANNELS),
            "imu_stream": provider.imu_stream_label,
            "imu_rate_hz": float(config["window"]["imu_rate_hz"]),
        },
        "evaluation_targets": {
            "action_verb": label.action,
            "object_noun": label.object_noun,
        },
        "annotation_provenance": {
            "action_source": "speech.csv lexical anchor",
            "action_surface": label.action_surface,
            "action_asr_confidence": label.action_confidence,
            "object_source": "nearest speech.csv lexical item within locked gap" if label.object_noun else "absent",
            "object_surface": label.object_surface,
            "object_asr_confidence": label.object_confidence,
            "object_gap_seconds": label.object_gap_seconds,
            "human_action_annotation": False,
        },
        "quality": quality,
    }


def preprocess_aea(
    config: Mapping[str, Any],
    output_root: str | Path,
    allow_missing_vrs: bool = False,
    provider_factory=ProjectAriaSequenceProvider,
) -> dict[str, Any]:
    validate_aea_config(config)
    source_root = Path(config["source"]["dataset_root"])
    with Path(config["source"]["subset_plan"]).open() as handle:
        plan = yaml.safe_load(handle)
    if str(plan.get("dataset", {}).get("release")) != str(config["source"]["release"]):
        raise ValueError("AEA subset plan release does not match preprocessing config")
    sequence_names = list(plan["selection"]["sequences"])
    source_vrs_present_count = sum(
        (source_root / name / "recording.vrs").is_file()
        and (source_root / name / "speech.csv").is_file()
        for name in sequence_names
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for name in sequence_names:
        sequence_id = AEASequenceId.parse(name)
        sequence_dir = source_root / name
        vrs = sequence_dir / "recording.vrs"
        speech = sequence_dir / "speech.csv"
        if not vrs.is_file() or not speech.is_file():
            if allow_missing_vrs:
                skipped[name] = "missing recording.vrs or speech.csv"
                continue
            raise FileNotFoundError(f"missing required AEA files under {sequence_dir}")
        words = read_speech(speech)
        windows = build_windows(words, config)
        if not windows:
            skipped[name] = "no eligible ASR action-anchor windows"
            continue
        provider = provider_factory(vrs, str(config["window"]["imu_stream"]))
        for index, label in enumerate(windows):
            try:
                records.append(_write_window(
                    provider, label, sequence_id, index, output, config
                ))
            except ValueError as exc:
                skipped[f"{name}/window-{index}"] = str(exc)

    examples_path = output / "examples.jsonl"
    examples_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    (output / "config_snapshot.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False))
    from babyworld_lite.aea.audit import audit_aea_records
    audit = audit_aea_records(records, output, config)
    summary = {
        "schema_version": "aea-preprocess-summary-v1",
        "scientific_status": (
            "real_aea_windows_preprocessed_no_effect_estimate"
            if records else "infrastructure_only_no_real_windows"
        ),
        "sequence_plan_count": len(sequence_names),
        "source_vrs_present_count": source_vrs_present_count,
        "acquisition_complete": source_vrs_present_count == len(sequence_names),
        "sequence_processed_count": len({row["sequence_id"] for row in records}),
        "window_count": len(records),
        "experiment_ready_for_split_construction": bool(
            records and audit["valid"] and source_vrs_present_count == len(sequence_names)
        ),
        "skipped": skipped,
        "audit": audit,
    }
    (output / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
