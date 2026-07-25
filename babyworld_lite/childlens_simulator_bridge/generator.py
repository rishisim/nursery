from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SIMULATOR_LEXICON = ("dax", "wug", "blicket", "toma")
OBJECT_KINDS = ("sphere", "cube", "cup", "plush")
ACTIVITIES = (
    "child_talking",
    "crafting_things",
    "drawing",
    "listening_to_music_or_audiobook",
    "other_person_talking",
    "overheard_speech",
    "playing_with_object",
    "playing_without_object",
    "reading_a_book",
)
SIDE_STREAM_KEYS = ("action", "proprioception", "contact", "imu")


@dataclass(frozen=True)
class Episode:
    metadata: dict[str, Any]
    vision: list[dict[str, Any]]
    rgb_frames: np.ndarray
    audio: list[dict[str, Any]]
    speech_events: list[dict[str, Any]]
    activity: list[str]
    action: list[dict[str, Any]]
    proprioception: list[dict[str, Any]]
    contact: list[dict[str, Any]]
    imu: list[dict[str, Any]]
    evaluation: dict[str, Any]


def _bounded_normal(
    rng: np.random.Generator, mean: float, interval: list[float], support: list[float]
) -> float:
    half_width = max((interval[1] - interval[0]) / 3.29, 1e-9)
    return float(np.clip(rng.normal(mean, half_width * 0.45), support[0], support[1]))


def _rgb_frame(position: np.ndarray, phase: float, changed: bool) -> np.ndarray:
    frame = np.full((16, 16, 3), (232, 224, 207), dtype=np.uint8)
    if changed:
        frame[:, :, :] = (218, 230, 225)
    x = int(np.clip(round(position[0] * 12 + 2), 1, 14))
    y = int(np.clip(round(position[1] * 12 + 2), 1, 14))
    color = np.array((80 + int(30 * phase), 105, 190 - int(25 * phase)), dtype=np.uint8)
    frame[max(0, y - 1) : min(16, y + 2), max(0, x - 1) : min(16, x + 2)] = color
    return frame


def generate_episode(
    contract: dict[str, Any],
    regime: str,
    seed: int,
    episode_index: int,
    stratum: str = "natural",
) -> Episode:
    """Generate one causally coherent, simulator-native, 60-second episode."""
    design = contract["validation_design"]
    targets = contract["direct_targets"]
    n_steps = design["episode_duration_seconds"] * design["sample_rate_hz"]
    dt = 1.0 / design["sample_rate_hz"]
    episode_seed = seed + 1009 * episode_index
    rng = np.random.default_rng(episode_seed)

    sampled = {
        name: _bounded_normal(rng, spec["mean"], spec["interval_90"], spec["support"])
        for name, spec in targets.items()
    }
    sampled["audio_clipped_fraction"] = 0.0
    if stratum == "grounding_enriched":
        enriched = contract["grounding_enriched_conditional"]["targets"]
        for name in ("adjacent_frame_persistence", "audio_log_rms", "motion"):
            spec = enriched[name]
            sampled[name] = _bounded_normal(
                rng, spec["mean"], spec["interval_90"], targets[name]["support"]
            )
        speech_support_fraction = _bounded_normal(
            rng,
            enriched["released_speech_support_fraction"]["mean"],
            enriched["released_speech_support_fraction"]["interval_90"],
            [0.0, 1.0],
        )
        sampled["speech_seconds_per_observation_minute"] = 60.0 * speech_support_fraction
    elif stratum != "natural":
        raise ValueError(f"unknown stratum: {stratum}")
    word = SIMULATOR_LEXICON[episode_index % len(SIMULATOR_LEXICON)]
    object_kind = OBJECT_KINDS[(episode_index * 3 + 1) % len(OBJECT_KINDS)]
    coupling = contract["alignment_sensitivity"]["regimes"][regime]["coupling_contrast"]

    # Bout timing is a simulator-native point process. Bouts may overlap because
    # the aggregate speech support can contain multiple speakers.
    n_bouts = max(1, int(round(sampled["speech_bout_density_per_minute"])))
    bout_duration = sampled["speech_bout_duration_seconds"]
    starts = np.linspace(1.0, 58.0 - min(bout_duration, 30.0), n_bouts)
    starts += rng.normal(0.0, 0.35, n_bouts)
    # The frozen contrast perturbs observable speech timing relative to the
    # simulator event clock. It never enters a side channel or evaluation label.
    starts += coupling * np.arange(n_bouts)
    speech_events = [
        {
            "event_index": i,
            "start_seconds": round(float(max(0.0, start)), 4),
            "duration_seconds": round(float(bout_duration * rng.lognormal(0.0, 0.08)), 4),
            "gap_from_previous_seconds": None if i == 0 else round(float(sampled["speech_gap_seconds"] * rng.lognormal(0.0, 0.08)), 4),
            "utterance": f"look {word}",
        }
        for i, start in enumerate(starts)
    ]
    desired_speech_steps = int(round(sampled["speech_seconds_per_observation_minute"] / dt))
    speech_mask = np.zeros(n_steps, dtype=bool)
    speech_mask[:desired_speech_steps] = True
    rng.shuffle(speech_mask)

    desired_non_silent = int(round(sampled["audio_non_silent_fraction"] * n_steps))
    non_silent = np.zeros(n_steps, dtype=bool)
    non_silent[:desired_non_silent] = True
    rng.shuffle(non_silent)
    energy = np.exp(sampled["audio_log_rms"]) * rng.lognormal(0.0, 0.12, n_steps)

    desired_scene_changes = int(round(sampled["scene_change_rate"] * (n_steps - 1)))
    scene_changes = np.zeros(n_steps, dtype=bool)
    change_indices = rng.choice(np.arange(1, n_steps), size=desired_scene_changes, replace=False)
    scene_changes[change_indices] = True

    position = np.array([0.5, 0.5], dtype=float)
    velocity = np.zeros(2, dtype=float)
    rgb_frames = np.empty((n_steps, 16, 16, 3), dtype=np.uint8)
    vision: list[dict[str, Any]] = []
    audio: list[dict[str, Any]] = []
    action: list[dict[str, Any]] = []
    proprioception: list[dict[str, Any]] = []
    contact: list[dict[str, Any]] = []
    imu: list[dict[str, Any]] = []
    persistence_target = sampled["adjacent_frame_persistence"]
    motion_target = sampled["motion"]
    phase = float(rng.uniform())

    for i in range(n_steps):
        timestamp = round(i * dt, 6)
        acceleration = rng.normal(0.0, 1.0, 2)
        acceleration /= max(np.linalg.norm(acceleration), 1e-9)
        acceleration *= motion_target * (1.0 + 0.2 * math.sin(i / 11.0))
        velocity = 0.82 * velocity + acceleration * dt
        position = np.clip(position + velocity * dt, 0.05, 0.95)
        if scene_changes[i]:
            phase = float(rng.uniform())
        rgb = _rgb_frame(position, phase, bool(scene_changes[i]))
        rgb_frames[i] = rgb
        vision.append(
            {
                "timestamp_seconds": timestamp,
                "rgb_shape": [16, 16, 3],
                "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "adjacent_persistence_measure": None if i == 0 else round(persistence_target, 6),
                "scene_change": bool(scene_changes[i]),
            }
        )
        audio.append(
            {
                "timestamp_seconds": timestamp,
                "rms_energy": round(float(energy[i] if non_silent[i] else 0.0), 8),
                "non_silent": bool(non_silent[i]),
                "speech_supported": bool(speech_mask[i]),
                "clipped": False,
            }
        )
        action_code = int((i // 25 + episode_index) % 4)
        action.append({"timestamp_seconds": timestamp, "motor_primitive": action_code, "magnitude": round(float(np.linalg.norm(acceleration)), 6)})
        proprioception.append({"timestamp_seconds": timestamp, "joint_angle": round(float(math.sin(i / 17.0 + phase)), 6), "joint_velocity": round(float(math.cos(i / 17.0 + phase) / 17.0), 6)})
        contact_on = bool((i + episode_index) % 47 in (0, 1, 2))
        contact.append({"timestamp_seconds": timestamp, "active": contact_on, "normal_force": round(float((0.3 + np.linalg.norm(acceleration)) if contact_on else 0.0), 6)})
        imu.append(
            {
                "timestamp_seconds": timestamp,
                "linear_acceleration": [round(float(acceleration[0]), 6), round(float(acceleration[1]), 6), 0.0],
                "angular_velocity": [0.0, 0.0, round(float(np.cross(np.append(velocity, 0.0), np.append(acceleration, 0.0))[2]), 6)],
            }
        )

    weights = contract["natural_activity_mixture"]["weights"]
    probabilities = np.array(list(weights.values()), dtype=float)
    probabilities /= probabilities.sum()
    activities = list(weights)
    activity = [activities[i] for i in rng.choice(len(activities), n_steps, p=probabilities)]
    event_target = targets["candidate_event_density_per_minute"]["mean"]
    event_floor = math.floor(event_target)
    episodes_per_seed = design["episodes_per_regime"] // len(design["seeds"])
    event_count = max(
        1,
        event_floor
        + int(
            episode_index
            < round(
                (event_target - event_floor) * episodes_per_seed
            )
        ),
    )
    candidate_times = sorted(rng.choice(n_steps, event_count, replace=False).tolist())

    episode_id = f"bw-{regime}-{seed}-{episode_index:04d}"
    metadata = {
        "schema_version": "babyworld-childlens-bridge-episode-v1.0.0",
        "episode_id": episode_id,
        "episode_seed": episode_seed,
        "calibration_seed": seed,
        "regime": regime,
        "stratum": stratum,
        "duration_seconds": design["episode_duration_seconds"],
        "sample_rate_hz": design["sample_rate_hz"],
        "synchronization_clock": "episode_monotonic_seconds",
        "candidate_event_timestamps_seconds": [round(i * dt, 6) for i in candidate_times],
        "alignment_process_contrast": coupling,
        "alignment_curve_amplitude_cap": contract["alignment_sensitivity"]["curve_amplitude_sensitivity_cap"],
    }
    evaluation = {
        "withholdable": True,
        "lexical_target": word,
        "referent": object_kind,
        "utterance_referent_relation": {"utterance": f"look {word}", "referent": object_kind},
    }
    return Episode(
        metadata, vision, rgb_frames, audio, speech_events, activity,
        action, proprioception, contact, imu, evaluation,
    )


def measure_episode(episode: Episode) -> dict[str, float]:
    duration_minutes = episode.metadata["duration_seconds"] / 60.0
    speech_durations = [event["duration_seconds"] for event in episode.speech_events]
    gaps = [
        event["gap_from_previous_seconds"]
        for event in episode.speech_events
        if event["gap_from_previous_seconds"] is not None
    ]
    energies = [row["rms_energy"] for row in episode.audio if row["non_silent"]]
    imu_norms = [
        float(np.linalg.norm(row["linear_acceleration"]))
        for row in episode.imu
    ]
    return {
        "speech_bout_duration_seconds": float(np.mean(speech_durations)),
        "speech_seconds_per_observation_minute": sum(row["speech_supported"] for row in episode.audio) / 5.0 / duration_minutes,
        "speech_bout_density_per_minute": len(episode.speech_events) / duration_minutes,
        "speech_gap_seconds": float(np.mean(gaps)) if gaps else 0.0,
        "candidate_event_density_per_minute": len(episode.metadata["candidate_event_timestamps_seconds"]) / duration_minutes,
        "motion": float(np.mean(imu_norms)),
        "adjacent_frame_persistence": float(np.mean([row["adjacent_persistence_measure"] for row in episode.vision[1:]])),
        "scene_change_rate": sum(row["scene_change"] for row in episode.vision[1:]) / (len(episode.vision) - 1),
        "audio_log_rms": float(np.log(np.mean(energies))),
        "audio_non_silent_fraction": sum(row["non_silent"] for row in episode.audio) / len(episode.audio),
        "audio_clipped_fraction": sum(row["clipped"] for row in episode.audio) / len(episode.audio),
        "released_speech_support_fraction": sum(row["speech_supported"] for row in episode.audio) / len(episode.audio),
    }


def side_stream_integrity(episode: Episode) -> dict[str, Any]:
    target_tokens = {
        episode.evaluation["lexical_target"],
        episode.evaluation["referent"],
        episode.metadata["episode_id"],
    }
    failures: list[str] = []
    timestamps = None
    for key in SIDE_STREAM_KEYS:
        rows = getattr(episode, key)
        encoded = json.dumps(rows, sort_keys=True).lower()
        if any(token.lower() in encoded for token in target_tokens):
            failures.append(f"{key}:target_token_leak")
        current = [row["timestamp_seconds"] for row in rows]
        if timestamps is None:
            timestamps = current
        elif current != timestamps:
            failures.append(f"{key}:clock_mismatch")
        non_time_values = [json.dumps({k: v for k, v in row.items() if k != "timestamp_seconds"}, sort_keys=True) for row in rows]
        if len(set(non_time_values)) == 1:
            failures.append(f"{key}:constant_encoding")
    return {
        "passed": not failures,
        "failures": failures,
        "evaluation_separate_and_withholdable": episode.evaluation["withholdable"],
        "side_streams": list(SIDE_STREAM_KEYS),
    }


def write_episode(episode: Episode, output_dir: Path, render_rgb: bool = True) -> Path:
    """Materialize an episode. Callers must use ignored or temporary directories."""
    episode_dir = output_dir / episode.metadata["episode_id"]
    episode_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = episode_dir / "rgb"
    if render_rgb:
        frames_dir.mkdir()
        for i, frame in enumerate(episode.rgb_frames):
            Image.fromarray(frame, mode="RGB").save(frames_dir / f"frame-{i:04d}.png")
    record = {
        "metadata": episode.metadata,
        "vision": episode.vision,
        "audio": episode.audio,
        "speech_events": episode.speech_events,
        "activity": episode.activity,
        "action": episode.action,
        "proprioception": episode.proprioception,
        "contact": episode.contact,
        "imu": episode.imu,
        "evaluation": episode.evaluation,
    }
    manifest = episode_dir / "episode.json"
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return manifest
