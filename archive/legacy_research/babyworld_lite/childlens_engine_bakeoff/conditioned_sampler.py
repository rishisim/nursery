"""Deterministic ChildLens-aggregate-conditioned episode population sampler."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .spec_kernel import CalibrationPolicy, compile_episode_intent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSampler:
    root: Path
    config: dict[str, Any]
    calibration: dict[str, Any]
    protocol: dict[str, Any]
    assets: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, config_path: Path) -> "FrozenSampler":
        root = config_path.parent.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        calibration_path = root / config["calibration"]["path"]
        if sha256_file(calibration_path) != config["calibration"]["sha256"]:
            raise ValueError("frozen ChildLens calibration hash mismatch")
        protocol_path = root / config["kernel_protocol"]
        assets_path = root / config["asset_registry"]
        return cls(
            root=root,
            config=config,
            calibration=json.loads(calibration_path.read_text(encoding="utf-8")),
            protocol=json.loads(protocol_path.read_text(encoding="utf-8")),
            assets=json.loads(assets_path.read_text(encoding="utf-8")),
        )

    def sample(self) -> list[dict[str, Any]]:
        seeds = self.config["qualification"]["seeds"]
        scenes = self.config["sampling"]["scenes"]
        targets = self.config["sampling"]["targets"]
        schedules = _speech_schedules(len(seeds), self.config)
        episodes = []
        calibration_policy = CalibrationPolicy.load(
            self.root / self.config["calibration"]["path"]
        )
        for index, (seed, schedule) in enumerate(zip(seeds, schedules)):
            rng = np.random.default_rng(seed)
            scene = scenes[index % len(scenes)]
            target_id = targets[(index // len(scenes)) % len(targets)]
            target = self.assets[target_id]
            target_text = (
                "die Tasse" if target_id == "yellow_cup_authored" else "der Ball"
            )
            phase_durations = _phase_durations(rng, self.config)
            transport_range = self.config["sampling"]["transport_delta_range_m"][
                scene["id"]
            ]
            intent = {
                "schema": "EpisodeIntent",
                "mode": "calibrated",
                "activity_family": "object_centered_reach_manipulate",
                "scene": scene,
                "target": {"asset_id": target_id},
                "phases": self.protocol["supported_phases"],
                "duration_s": 30.0,
                "speech": {
                    "name_count": len(schedule),
                    "text": target_text,
                },
                "controller": {
                    **self.config["sampling"]["controller_base"],
                    "transport_delta_m": float(rng.uniform(*transport_range)),
                    "target_anchor_from_hand_m": self.config["sampling"][
                        "target_anchor_from_hand_m"
                    ][scene["id"]],
                },
                "camera": {
                    **self.config["sampling"]["camera_base"],
                    "position_offset_m": [
                        round(0.23 + float(rng.uniform(-0.012, 0.012)), 6),
                        round(0.11 + float(rng.uniform(-0.010, 0.010)), 6),
                        round(0.17 + float(rng.uniform(-0.008, 0.008)), 6),
                    ],
                },
                "seed": int(seed),
                "conditioned_sample": {
                    "calibration_sha256": self.config["calibration"]["sha256"],
                    "phase_durations_s": phase_durations,
                    "speech_events": [
                        {
                            "start_s": event["start_s"],
                            "duration_s": event["duration_s"],
                            "semantic_constraint": "name_target",
                            "text": target_text,
                        }
                        for event in schedule
                    ],
                },
            }
            resolved = compile_episode_intent(
                intent,
                self.protocol,
                calibration_policy,
                asset_registry=self.assets,
            )
            episodes.append(
                {
                    "episode_id": f"clcb_{index:03d}_{seed}",
                    "seed": seed,
                    "intent": intent,
                    "resolved_spec": resolved,
                    "sampled_measurements": measure_schedule(resolved),
                }
            )
        return episodes


def _phase_durations(rng: np.random.Generator, config: dict[str, Any]) -> list[float]:
    base = np.asarray(config["sampling"]["phase_duration_weights"], dtype=float)
    jitter = rng.uniform(0.92, 1.08, len(base))
    values = 30.0 * base * jitter / np.sum(base * jitter)
    rounded = np.round(values, 6)
    rounded[-1] += 30.0 - float(rounded.sum())
    return rounded.tolist()


def _speech_schedules(count: int, config: dict[str, Any]) -> list[list[dict[str, float]]]:
    """Construct a bounded population satisfying the frozen interval constraints."""
    policy = config["sampling"]["speech_schedule"]
    bout_counts = list(policy["bout_counts"])
    if len(bout_counts) != count:
        raise ValueError("speech bout count schedule must match seed count")
    schedules = []
    for index, bouts in enumerate(bout_counts):
        duration = float(policy["duration_cycle_s"][index % len(policy["duration_cycle_s"])])
        gap = float(policy["gap_cycle_s"][index % len(policy["gap_cycle_s"])])
        occupied = bouts * duration + max(0, bouts - 1) * gap
        if occupied > 29.0:
            raise ValueError("sampled speech schedule exceeds episode support")
        start = (30.0 - occupied) / 2.0
        schedules.append(
            [
                {
                    "start_s": round(start + j * (duration + gap), 6),
                    "duration_s": duration,
                }
                for j in range(bouts)
            ]
        )
    return schedules


def measure_schedule(spec: dict[str, Any]) -> dict[str, float]:
    events = spec["speech_events"]["value"]
    duration = float(spec["duration_s"]["value"])
    bout_durations = [float(event["duration_s"]) for event in events]
    gaps = [
        float(events[index + 1]["start_s"])
        - float(events[index]["start_s"])
        - float(events[index]["duration_s"])
        for index in range(len(events) - 1)
    ]
    phases = spec["phase_timestamps"]["value"]
    candidate_phases = {
        "near_miss",
        "touch_push",
        "post_contact_grasp",
        "lift_place",
        "release_drop",
    }
    return {
        "speech_bout_duration_seconds": float(np.mean(bout_durations)),
        "speech_seconds_per_observation_minute": float(
            sum(bout_durations) * 60.0 / duration
        ),
        "speech_bout_density_per_minute": float(len(events) * 60.0 / duration),
        "speech_gap_seconds": float(np.mean(gaps)) if gaps else 0.0,
        "candidate_event_density_per_minute": float(
            (
                sum(phase["phase"] in candidate_phases for phase in phases)
                + int(len(events) >= 3)
            )
            * 60.0
            / duration
        ),
        "activity_duration_seconds": duration,
        "speech_activity_overlap_fraction": 1.0,
    }


def qualify_sampled_population(
    episodes: list[dict[str, Any]], config: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    targets = calibration["direct_targets"]
    feature_names = config["qualification"]["direct_schedule_features"]
    rows = []
    for feature in feature_names:
        values = np.asarray(
            [episode["sampled_measurements"][feature] for episode in episodes],
            dtype=float,
        )
        target = targets[feature]
        interval = target["interval_90"]
        estimate = float(values.mean())
        rows.append(
            {
                "feature": feature,
                "target_mean": target["mean"],
                "target_interval_90": interval,
                "generated_mean": estimate,
                "generated_p10_p50_p90": np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
                "deviation": estimate - float(target["mean"]),
                "passed": bool(interval[0] <= estimate <= interval[1]),
            }
        )
    vectors = np.asarray(
        [
            [episode["sampled_measurements"][feature] for feature in feature_names]
            for episode in episodes
        ]
    )
    unique_vectors = len({tuple(row) for row in vectors})
    return {
        "rows": rows,
        "all_direct_schedule_features_pass": all(row["passed"] for row in rows),
        "joint_coverage": {
            "unique_feature_vectors": unique_vectors,
            "minimum_required": config["qualification"]["joint_minimum_unique_vectors"],
            "passed": unique_vectors
            >= config["qualification"]["joint_minimum_unique_vectors"],
        },
    }
