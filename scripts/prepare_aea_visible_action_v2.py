#!/usr/bin/env python3
"""Build the frozen development-only AEA v2 dense annotation packets.

The script validates exact v1 development membership and writes a preflight
receipt before constructing or opening any VRS path. It queries RGB only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parents[1]))

from babyworld_lite.aea.visible_action_v2 import (  # noqa: E402
    apply_protocol_amendment,
    canonical_digest,
    load_json,
    packet_order,
    select_frozen_sample,
    sha256_file,
    validate_frozen_development_inputs,
)


class RGBOnlyProjectAriaProvider:
    """Project Aria adapter that never resolves or queries an IMU stream."""

    def __init__(self, vrs_path: Path):
        try:
            from projectaria_tools.core import data_provider
            from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
        except ImportError as exc:  # pragma: no cover - dedicated Aria runtime
            raise RuntimeError("projectaria-tools is required for dense RGB extraction") from exc
        self._time_domain = TimeDomain
        self._query_options = TimeQueryOptions
        self._provider = data_provider.create_vrs_data_provider(str(vrs_path))
        if self._provider is None:
            raise RuntimeError(f"could not open development VRS: {vrs_path.name}")
        self._rgb_stream = self._provider.get_stream_id_from_label("camera-rgb")
        if self._rgb_stream is None:
            raise RuntimeError("camera-rgb stream is absent")

    def image_at(self, timestamp_ns: int) -> tuple[np.ndarray, int]:
        image, record = self._provider.get_image_data_by_time_ns(
            self._rgb_stream,
            int(timestamp_ns),
            self._time_domain.DEVICE_TIME,
            self._query_options.CLOSEST,
        )
        upright = np.rot90(image.to_numpy_array(), -1).copy()
        return upright, int(record.capture_timestamp_ns)


def _write_json_new(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen v2 artifact: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def _contact_sheet(
    frames: Sequence[Image.Image],
    indices: Sequence[int],
    path: Path,
    columns: int,
    title: str,
) -> None:
    cell = 236
    image_size = 224
    header = 42
    rows = (len(indices) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell, header + rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill="black")
    for position, frame_index in enumerate(indices):
        row, column = divmod(position, columns)
        thumbnail = ImageOps.contain(frames[frame_index].convert("RGB"), (image_size, image_size))
        x = column * cell + (cell - thumbnail.width) // 2
        y = header + row * cell + (cell - thumbnail.height) // 2
        canvas.paste(thumbnail, (x, y))
        border = "#d62728" if frame_index in (15, 16) else "black"
        width = 4 if frame_index in (15, 16) else 1
        draw.rectangle((x, y, x + thumbnail.width - 1, y + thumbnail.height - 1), outline=border, width=width)
        draw.rectangle((x + 3, y + 3, x + 42, y + 24), fill="white")
        draw.text((x + 8, y + 7), f"{frame_index:02d}", fill=border)
    path.parent.mkdir(parents=True, exist_ok=False)
    canvas.save(path, quality=90, optimize=True)


def _dense_times(window: Mapping[str, Any], count: int) -> np.ndarray:
    start = int(window["start_device_time_ns"])
    end = int(window["end_device_time_ns"])
    width = (end - start) / count
    return start + (np.arange(count, dtype=np.float64) + 0.5) * width


def _packet_rows(
    ordered: Sequence[Mapping[str, Any]], output: Path, pass_name: str
) -> list[dict[str, Any]]:
    rows = []
    for rank, row in enumerate(ordered, start=1):
        media = output / "dense_evidence" / str(row["blind_id"])
        rows.append({
            "packet_rank": rank,
            "blind_id": str(row["blind_id"]),
            "anchored_asr_verb": str(row["asr_action_verb"]),
            "transcript": str(row["transcript"]),
            "frame_count": 31,
            "anchor_guidance": "ASR anchor is centered between frames 15 and 16",
            "contact_sheet": str((media / "contact_sheet.jpg").resolve()),
            "detail_sheet_00_15": str((media / "detail_00_15.jpg").resolve()),
            "detail_sheet_16_30": str((media / "detail_16_30.jpg").resolve()),
            "frame_directory": str((media / "frames").resolve()),
            "annotation": {
                "blind_id": str(row["blind_id"]),
                "asr_refers_to_visible_wearer_action": None,
                "agency": None,
                "temporal_alignment": None,
                "observable_action": None,
                "confidence": None,
                "evidence_frame_start": None,
                "evidence_frame_end": None,
                "rationale": None,
            },
        })
    return rows


def prepare(
    development_path: Path,
    partition_path: Path,
    prior_prelabel_path: Path,
    protocol_path: Path,
    amendment_path: Path,
    raw_root: Path,
    output: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    amendment = load_json(amendment_path)
    if amendment.get("parent_protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("frozen protocol amendment parent hash mismatch")
    protocol = apply_protocol_amendment(protocol, amendment)
    rows, prior_rows, preflight = validate_frozen_development_inputs(
        development_path, partition_path, prior_prelabel_path, protocol
    )
    sample = select_frozen_sample(rows, prior_rows, protocol)
    reserve_groups = set(map(str, protocol["sources"]["reserve_event_groups"]))
    if any(str(row["event_group"]) in reserve_groups for row in sample):
        raise AssertionError("reserve group entered frozen v2 sample")
    if len({str(row["event_group"]) for row in sample}) != 18:
        raise AssertionError("frozen sample does not cover all development groups")
    development_group_support = Counter(str(row["event_group"]) for row in rows)
    sample_group_support = Counter(str(row["event_group"]) for row in sample)
    if any(
        sample_group_support[group] < min(4, count)
        for group, count in development_group_support.items()
    ):
        raise AssertionError("amended frozen per-group sampling minimum was not met")

    output.mkdir(parents=True, exist_ok=True)
    dense_root = output / "dense_evidence"
    if dense_root.exists():
        raise FileExistsError(f"refusing to overwrite dense evidence: {dense_root}")

    frame_count = int(protocol["dense_evidence"]["frames"])
    manifest_rows = []
    for row in sample:
        relative = Path("dense_evidence") / str(row["blind_id"])
        manifest_rows.append({
            **dict(row),
            "dense_frame_paths": [str(relative / "frames" / f"{index:03d}.jpg") for index in range(frame_count)],
            "contact_sheet": str(relative / "contact_sheet.jpg"),
            "detail_sheets": [
                str(relative / "detail_00_15.jpg"),
                str(relative / "detail_16_30.jpg"),
            ],
        })
    manifest = {
        "schema_version": "aea-visible-action-dense-manifest-v2",
        "protocol_id": protocol["protocol_id"],
        "sample_size": len(sample),
        "sample_digest": canonical_digest([
            {key: row[key] for key in ("blind_id", "example_id", "event_group")}
            for row in sample
        ]),
        "selection": protocol["sample"],
        "dense_evidence": protocol["dense_evidence"],
        "reserve_groups_present": [],
        "rows": manifest_rows,
    }
    _write_json_new(output / "dense_clip_manifest.json", manifest)

    pass_a_order = packet_order(sample, protocol["sample"]["pass_order_salts"]["pass_a"])
    pass_b_order = packet_order(sample, protocol["sample"]["pass_order_salts"]["pass_b"])
    packet_a = {
        "schema_version": "aea-visible-action-blinded-packet-v2",
        "protocol_id": protocol["protocol_id"],
        "pass": "pass_a",
        "role": protocol["annotation_schema"]["pass_role"],
        "codebook": str((output / "annotation_codebook.md").resolve()),
        "blinding": ["example_id", "event_group", "location", "prior_v1_membership", "v1_label", "other_pass", "outcomes"],
        "rows": _packet_rows(pass_a_order, output, "pass_a"),
    }
    packet_b = {
        **{key: value for key, value in packet_a.items() if key != "rows"},
        "pass": "pass_b",
        "rows": _packet_rows(pass_b_order, output, "pass_b"),
    }
    _write_json_new(output / "annotation_packet_pass_a.json", packet_a)
    _write_json_new(output / "annotation_packet_pass_b.json", packet_b)

    sequences = sorted({str(row["sequence_id"]) for row in sample})
    preflight.update({
        "schema_version": "aea-visible-action-dense-preflight-v2",
        "protocol_id": protocol["protocol_id"],
        "written_before_media_access": True,
        "sample_size": len(sample),
        "sample_digest": manifest["sample_digest"],
        "planned_development_sequences": sequences,
        "planned_development_event_groups": sorted({str(row["event_group"]) for row in sample}),
        "reserve_event_groups_planned": [],
        "rgb_queries_planned": len(sample) * frame_count,
        "imu_queries_planned": 0,
        "signed_manifest_loaded": False,
    })
    _write_json_new(output / "dense_preflight_receipt.json", preflight)

    sample_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in sample:
        sample_by_sequence.setdefault(str(row["sequence_id"]), []).append(row)
    opened_sequences = []
    frame_time_errors = []
    capture_time_rows = []
    for sequence_id in sequences:
        # This path is constructed only after exact development/reserve preflight.
        vrs_path = raw_root / sequence_id / "recording.vrs"
        if not vrs_path.is_file():
            raise FileNotFoundError(f"missing selected development VRS: {sequence_id}")
        provider = RGBOnlyProjectAriaProvider(vrs_path)
        opened_sequences.append(sequence_id)
        for row in sorted(sample_by_sequence[sequence_id], key=lambda item: str(item["blind_id"])):
            blind_id = str(row["blind_id"])
            directory = dense_root / blind_id
            frames_dir = directory / "frames"
            frames_dir.mkdir(parents=True, exist_ok=False)
            query_times = _dense_times(row["window"], frame_count)
            images: list[Image.Image] = []
            captures: list[int] = []
            for index, query_time in enumerate(query_times):
                pixels, capture_ns = provider.image_at(int(query_time))
                image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB")
                image.save(frames_dir / f"{index:03d}.jpg", quality=88, optimize=True)
                images.append(image)
                captures.append(capture_ns)
            errors = np.abs(np.asarray(captures, dtype=np.float64) - query_times) / 1e6
            maximum_error = float(errors.max())
            if maximum_error > 100.0:
                raise ValueError(f"{blind_id} maximum RGB timestamp error exceeds 100 ms")
            frame_time_errors.append(maximum_error)
            capture_time_rows.append({
                "blind_id": blind_id,
                "query_device_time_ns": [int(value) for value in query_times],
                "capture_device_time_ns": captures,
                "maximum_absolute_error_ms": maximum_error,
                "unique_capture_count": len(set(captures)),
            })
            # Directory already exists, so contact-sheet helper receives dedicated child dirs.
            _contact_sheet(images, list(range(31)), directory / "full" / "contact_sheet.jpg", 7,
                           "Frames 00-30; ASR anchor is centered between red frames 15 and 16")
            (directory / "full" / "contact_sheet.jpg").replace(directory / "contact_sheet.jpg")
            (directory / "full").rmdir()
            _contact_sheet(images, list(range(16)), directory / "detail_a" / "detail_00_15.jpg", 4,
                           "Frames 00-15; red frame 15 borders the centered ASR anchor")
            (directory / "detail_a" / "detail_00_15.jpg").replace(directory / "detail_00_15.jpg")
            (directory / "detail_a").rmdir()
            _contact_sheet(images, list(range(16, 31)), directory / "detail_b" / "detail_16_30.jpg", 4,
                           "Frames 16-30; red frame 16 borders the centered ASR anchor")
            (directory / "detail_b" / "detail_16_30.jpg").replace(directory / "detail_16_30.jpg")
            (directory / "detail_b").rmdir()
        del provider

    access_receipt = {
        "schema_version": "aea-visible-action-reserve-access-receipt-v2",
        "protocol_id": protocol["protocol_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dense_manifest_sha256": sha256_file(output / "dense_clip_manifest.json"),
        "development_rgb_vrs_files_opened": len(opened_sequences),
        "development_rgb_sequences_opened": opened_sequences,
        "development_rgb_frames_queried": len(sample) * frame_count,
        "development_imu_arrays_opened": 0,
        "reserve_rgb_files_opened": 0,
        "reserve_imu_arrays_opened": 0,
        "reserve_event_groups_opened": [],
        "signed_manifest_loaded": False,
        "signed_urls_loaded_printed_or_copied": False,
        "rgb_api_only": True,
        "imu_stream_resolved_or_queried": False,
        "frame_timing": {
            "maximum_absolute_error_ms": max(frame_time_errors),
            "mean_per_window_maximum_absolute_error_ms": float(np.mean(frame_time_errors)),
            "all_windows_have_31_unique_captures": all(row["unique_capture_count"] == 31 for row in capture_time_rows),
            "rows": capture_time_rows,
        },
    }
    _write_json_new(output / "reserve_access_receipt.json", access_receipt)
    return access_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=Path("output/aea_dev_learnability_v1/development_examples.jsonl"))
    parser.add_argument("--partition", type=Path, default=Path("output/aea_dev_learnability_v1/partition_manifest.json"))
    parser.add_argument("--prior-prelabel", type=Path, default=Path("output/aea_dev_learnability_v1/audit_manifest_prelabel.json"))
    parser.add_argument("--protocol", type=Path, default=Path("output/aea_visible_action_v2/preregistered_protocol.json"))
    parser.add_argument("--amendment", type=Path, default=Path("output/aea_visible_action_v2/preregistered_protocol_amendment_1.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/aea_raw"))
    parser.add_argument("--out", type=Path, default=Path("output/aea_visible_action_v2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(
        args.development, args.partition, args.prior_prelabel,
        args.protocol, args.amendment, args.raw_root, args.out,
    )
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "development_rgb_frames_queried": result["development_rgb_frames_queried"],
        "reserve_rgb_files_opened": result["reserve_rgb_files_opened"],
        "reserve_imu_arrays_opened": result["reserve_imu_arrays_opened"],
    }, indent=2))


if __name__ == "__main__":
    main()
