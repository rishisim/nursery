from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from babyworld_lite.aea.preprocess import IMU_CHANNELS, SCHEMA_VERSION


SMOKE_ACTIONS = ("get", "put", "cook", "grab")
SMOKE_OBJECTS = ("coffee", "dish", "egg", "cup")


def build_aea_smoke_fixture(root: str | Path) -> Path:
    """Create a synthetic schema fixture; never present it as an AEA finding."""
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260713)
    colors = {
        "get": (220, 80, 70),
        "put": (70, 150, 220),
        "cook": (240, 180, 60),
        "grab": (80, 190, 110),
    }
    example_number = 0
    for location in (1, 2, 3):
        for sequence in (1, 2, 3, 4):
            sequence_id = f"loc{location}_script1_seq{sequence}_rec1"
            wearer_group = f"loc{location}_script1_rec1"
            event_group = f"loc{location}_script1_seq{sequence}"
            for repeat in (0, 1):
                for action_index, action in enumerate(SMOKE_ACTIONS):
                    obj = SMOKE_OBJECTS[(action_index + sequence + repeat) % len(SMOKE_OBJECTS)]
                    example_id = f"{sequence_id}_w{repeat * 4 + action_index:04d}"
                    directory = output / "windows" / sequence_id / example_id
                    frames_dir = directory / "frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    frame_paths = []
                    for frame_index in range(3):
                        image = Image.new("RGB", (28, 28), colors[action])
                        draw = ImageDraw.Draw(image)
                        draw.rectangle(
                            (4 + frame_index, 8, 12 + frame_index, 16),
                            fill=(255, 255, 255),
                        )
                        path = frames_dir / f"{frame_index:03d}.jpg"
                        image.save(path)
                        frame_paths.append(str(path.relative_to(output)))
                    time = np.linspace(0, 2 * np.pi, 30, endpoint=False)
                    imu = np.stack([
                        np.sin(time + action_index),
                        np.cos(time + action_index),
                        np.sin(2 * time + action_index),
                        np.cos(2 * time + action_index),
                        np.sin(3 * time + action_index),
                        np.cos(3 * time + action_index),
                    ], axis=1).astype(np.float32)
                    imu += rng.normal(0, 0.01, size=imu.shape).astype(np.float32)
                    imu_path = directory / "imu.npy"
                    np.save(imu_path, imu, allow_pickle=False)
                    rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "example_id": example_id,
                        "sequence_id": sequence_id,
                        "event_group": event_group,
                        "wearer_session_group": wearer_group,
                        "location": location,
                        "script": 1,
                        "sequence": sequence,
                        "recording": 1,
                        "window": {
                            "start_device_time_ns": example_number * 10_000_000_000,
                            "end_device_time_ns": example_number * 10_000_000_000 + 6_000_000_000,
                            "anchor_device_time_ns": example_number * 10_000_000_000 + 3_000_000_000,
                            "duration_seconds": 6.0,
                        },
                        "model_inputs": {
                            "frame_paths": frame_paths,
                            "frame_capture_device_time_ns": [0, 2_000_000_000, 4_000_000_000],
                            "transcript": f"please {action} the {obj} now",
                            "imu_path": str(imu_path.relative_to(output)),
                            "imu_channels": list(IMU_CHANNELS),
                            "imu_stream": "fixture-six-axis",
                            "imu_rate_hz": 5.0,
                        },
                        "evaluation_targets": {"action_verb": action, "object_noun": obj},
                        "annotation_provenance": {
                            "action_source": "synthetic smoke fixture",
                            "action_surface": action,
                            "action_asr_confidence": 1.0,
                            "object_source": "synthetic smoke fixture",
                            "object_surface": obj,
                            "object_asr_confidence": 1.0,
                            "object_gap_seconds": 0.1,
                            "human_action_annotation": False,
                        },
                        "quality": {
                            "raw_sample_count": 30,
                            "raw_rate_hz": 5.0,
                            "resampled_sample_count": 30,
                            "resampled_rate_hz": 5.0,
                            "coverage_fraction": 1.0,
                            "maximum_raw_gap_ms": 200.0,
                            "maximum_frame_time_error_ms": 0.0,
                        },
                    })
                    example_number += 1
    examples = output / "examples.jsonl"
    examples.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "FIXTURE_STATUS.txt").write_text(
        "Synthetic AEA-schema smoke fixture. Infrastructure validation only; not real AEA data.\n"
    )
    return examples
