"""Deterministic non-outcome construction fixtures.

All numbers in this module are arbitrary software-test values.  They are not
measurements, ecological estimates, priors, or acquisition targets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .policy import CONSTRUCTION_PROFILE, CONTRACT_VERSION, SCHEMA_VERSION, canonical_digest, canonical_json_bytes
from .schema import canonical_jsonl, manifest_for_files


NONSCIENTIFIC_CONSTRUCTION_FIXTURE_SETTINGS = {
    "profile_label": CONSTRUCTION_PROFILE,
    "episode_count": 2,
    "frame_count": 4,
    "frame_width_px": 64,
    "frame_height_px": 64,
    "frame_interval_ns": 100_000_000,
    "purpose": "SOFTWARE_CONSTRUCTION_VALIDATION_ONLY",
    "ecological_claim_allowed": False,
    "acquisition_claim_allowed": False,
}


def _frame_payload(episode_index: int, frame_index: int) -> Image.Image:
    settings = NONSCIENTIFIC_CONSTRUCTION_FIXTURE_SETTINGS
    image = Image.new("RGB", (settings["frame_width_px"], settings["frame_height_px"]), (235, 238, 240))
    draw = ImageDraw.Draw(image)
    # Geometry and colors are arbitrary renderer fixtures; no labels or sensor
    # markers are rendered into the model-visible pixels.
    x = 10 + frame_index * 6
    y = 20 + episode_index * 12
    draw.ellipse((x, y, x + 16, y + 16), fill=(44, 108, 180), outline=(20, 45, 70))
    draw.rectangle((42, 36, 56, 50), fill=(217, 128, 56), outline=(70, 40, 20))
    return image


def _sample(timestamp_ns: int, episode_index: int, frame_index: int) -> dict[str, Any]:
    phase = float(episode_index * 10 + frame_index)
    return {
        "imu": {
            "timestamp_ns": timestamp_ns,
            "acceleration_m_s2": [0.01 * phase, 0.02 * phase, 9.8],
            "angular_velocity_rad_s": [0.001 * phase, 0.002 * phase, 0.003 * phase],
            "valid": True,
        },
        "proprio": {
            "timestamp_ns": timestamp_ns,
            "joint_position_rad": [0.01 * phase, -0.01 * phase],
            "joint_velocity_rad_s": [0.1, -0.1],
            "end_effector_position_m": [0.01 * frame_index, 0.02 * episode_index, 0.3],
            "end_effector_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "valid": True,
        },
        "touch": {
            "timestamp_ns": timestamp_ns,
            "contact_binary": [float(frame_index >= 2)],
            "normal_force_n": [0.2 * max(0, frame_index - 1)],
            "pressure_pa": [1.5 * max(0, frame_index - 1)],
            "slip_velocity_m_s": [0.01 * max(0, frame_index - 2)],
            "vibration_m_s2": [0.02 * max(0, frame_index - 1)],
            "valid": True,
        },
        "motor": {
            "timestamp_ns": timestamp_ns,
            "values": [0.1 * frame_index, -0.05 * frame_index],
            "valid": True,
        },
        "object": {
            "timestamp_ns": timestamp_ns,
            "position_m": [0.02 * frame_index, 0.0, 0.1],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity_m_s": [0.02, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.01 * frame_index],
            "visible_fraction": 1.0,
        },
    }


def build_construction_fixture(destination: str | Path) -> dict[str, Any]:
    """Write a deterministic fixture bundle; never train or score a learner."""

    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("construction fixture destination must be absent or empty")
    visible_root = root / "model_visible"
    instrumentation_root = root / "instrumentation"
    oracle_root = root / "hidden_oracle"
    for directory in (visible_root, instrumentation_root, oracle_root):
        directory.mkdir(parents=True, exist_ok=True)

    settings = NONSCIENTIFIC_CONSTRUCTION_FIXTURE_SETTINGS
    visible_records: list[dict[str, Any]] = []
    instrumentation_records: list[dict[str, Any]] = []
    oracle_records: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    for episode_index in range(settings["episode_count"]):
        episode_id = f"fx{episode_index:04d}"
        episode_ids.append(episode_id)
        frame_records: list[dict[str, Any]] = []
        samples = []
        for frame_index in range(settings["frame_count"]):
            timestamp_ns = frame_index * settings["frame_interval_ns"]
            frame_rel = Path("frames") / episode_id / f"{frame_index:04d}.png"
            frame_path = visible_root / frame_rel
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            _frame_payload(episode_index, frame_index).save(frame_path, format="PNG", optimize=False)
            payload = frame_path.read_bytes()
            frame_records.append(
                {
                    "timestamp_ns": timestamp_ns,
                    "path": frame_rel.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "width_px": settings["frame_width_px"],
                    "height_px": settings["frame_height_px"],
                    "channels": 3,
                    "color_space": "sRGB",
                    "dtype": "uint8",
                    "encoding": "PNG",
                }
            )
            samples.append(_sample(timestamp_ns, episode_index, frame_index))

        duration_ns = settings["frame_count"] * settings["frame_interval_ns"]
        visible_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_version": CONTRACT_VERSION,
                "profile_label": CONSTRUCTION_PROFILE,
                "episode_id": episode_id,
                "study_instance_id": "construction-fixture-instance-v1",
                "split": "train" if episode_index == 0 else "test",
                "timebase": "MONOTONIC_NANOSECONDS",
                "duration_ns": duration_ns,
                "frame_stream": frame_records,
                "utterances": [
                    {
                        "utterance_id": f"u{episode_index:04d}",
                        "onset_ns": settings["frame_interval_ns"] // 2,
                        "offset_ns": settings["frame_interval_ns"] * 2,
                        "text": "look at it move",
                        "speaker_role": "CAREGIVER",
                        "language_tag": "en",
                        "valid": True,
                    }
                ],
                "head_imu": {
                    "coordinate_frame": "HEAD_RIGHT_HANDED_XYZ",
                    "samples": [sample["imu"] for sample in samples],
                },
                "proprioception": {
                    "joint_names": ["joint_00", "joint_01"],
                    "coordinate_frame": "BODY_RIGHT_HANDED_XYZ",
                    "samples": [sample["proprio"] for sample in samples],
                },
                "contact_touch": {
                    "sensor_names": ["sensor_00"],
                    "coordinate_frame": "END_EFFECTOR_RIGHT_HANDED_XYZ",
                    "samples": [sample["touch"] for sample in samples],
                },
                "continuous_motor": {
                    "channel_names": ["control_00", "control_01"],
                    "units": ["normalized", "normalized"],
                    "samples": [sample["motor"] for sample in samples],
                },
            }
        )
        instrumentation_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_version": CONTRACT_VERSION,
                "profile_label": CONSTRUCTION_PROFILE,
                "episode_id": episode_id,
                "object_state_trajectories": [
                    {"object_id": f"o{episode_index:04d}", "samples": [sample["object"] for sample in samples]}
                ],
            }
        )
        oracle_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "contract_version": CONTRACT_VERSION,
                "profile_label": CONSTRUCTION_PROFILE,
                "episode_id": episode_id,
                "events": [
                    {
                        "event_id": f"e{episode_index:04d}",
                        "event_type": "fixture_motion_event",
                        "onset_ns": settings["frame_interval_ns"],
                        "offset_ns": settings["frame_interval_ns"] * 3,
                        "participant_object_ids": [f"o{episode_index:04d}"],
                        "causal_parent_event_ids": [],
                    }
                ],
                "utterance_event_truth": [
                    {
                        "utterance_id": f"u{episode_index:04d}",
                        "target_event_id": f"e{episode_index:04d}" if episode_index == 0 else None,
                        "null_reason": None if episode_index == 0 else "CONSTRUCTION_NULL_CASE",
                        "candidate_event_ids": [f"e{episode_index:04d}"],
                    }
                ],
                "counterfactual_truth": {"fixture_only": True, "scientific_interpretation": False},
            }
        )

    (visible_root / "episodes.jsonl").write_bytes(canonical_jsonl(visible_records))
    (instrumentation_root / "object_states.jsonl").write_bytes(canonical_jsonl(instrumentation_records))
    (oracle_root / "events.jsonl").write_bytes(canonical_jsonl(oracle_records))
    (root / "fixture_settings.json").write_bytes(canonical_json_bytes(settings) + b"\n")

    paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    )
    file_manifest = manifest_for_files(root, paths)
    manifest_core = {
        "bundle_version": "child-only-fixture-bundle-v1",
        "contract_version": CONTRACT_VERSION,
        "profile_label": CONSTRUCTION_PROFILE,
        "roots": {
            "model_visible": "model_visible",
            "instrumentation": "instrumentation",
            "hidden_oracle": "hidden_oracle",
        },
        "file_manifest": file_manifest,
        "episode_ids": episode_ids,
    }
    manifest = {**manifest_core, "canonical_digest": canonical_digest(manifest_core)}
    (root / "bundle_manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest
