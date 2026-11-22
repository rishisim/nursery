"""Canonical spec compiler for the bounded MIMo–MolmoSpaces prototype kernel."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class UnsupportedEpisodeIntent(ValueError):
    """Raised when an intent is outside the preregistered kernel support."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibrationPolicy:
    source_path: Path
    source_sha256: str
    speech_bout_duration_s: float
    speech_gap_s: float
    speech_density_per_minute: float
    motion_mean: float

    @classmethod
    def load(cls, path: Path) -> "CalibrationPolicy":
        raw = path.read_bytes()
        record = json.loads(raw)
        aggregate = record["direct_targets"]
        return cls(
            source_path=path,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            speech_bout_duration_s=float(
                aggregate["speech_bout_duration_seconds"]["mean"]
            ),
            speech_gap_s=float(aggregate["speech_gap_seconds"]["mean"]),
            speech_density_per_minute=float(
                aggregate["speech_bout_density_per_minute"]["mean"]
            ),
            motion_mean=float(aggregate["motion"]["mean"]),
        )


SUPPORTED_PHASES = {
    "look_settle",
    "reach",
    "near_miss",
    "touch_push",
    "post_contact_grasp",
    "lift_place",
    "release_drop",
    "object_naming",
}
SUPPORTED_SCENES = {"FloorPlan1", "FloorPlan201"}


def _field(value: Any, provenance: str) -> dict[str, Any]:
    return {"value": value, "provenance": provenance}


def compile_episode_intent(
    intent: dict[str, Any],
    protocol: dict[str, Any],
    calibration: CalibrationPolicy,
    *,
    asset_registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compile a semantic intent into a deterministic, provenance-rich spec."""
    if intent.get("activity_family") != "object_centered_reach_manipulate":
        raise UnsupportedEpisodeIntent(
            f"unsupported activity_family: {intent.get('activity_family')!r}"
        )
    mode = intent.get("mode")
    if mode not in protocol["modes"]:
        raise UnsupportedEpisodeIntent(f"unsupported mode: {mode!r}")
    phases = intent.get("phases")
    if not isinstance(phases, list) or not phases:
        raise UnsupportedEpisodeIntent("phases must be a non-empty list")
    unsupported = sorted(set(phases) - SUPPORTED_PHASES)
    if unsupported:
        raise UnsupportedEpisodeIntent(f"unsupported phases: {unsupported}")
    scene_id = intent.get("scene", {}).get("id")
    if scene_id not in SUPPORTED_SCENES:
        raise UnsupportedEpisodeIntent(f"unsupported scene id: {scene_id!r}")
    object_id = intent.get("target", {}).get("asset_id")
    if object_id not in asset_registry:
        raise UnsupportedEpisodeIntent(f"unknown target asset_id: {object_id!r}")
    conditioned_sample = intent.get("conditioned_sample")
    if mode == "calibrated" and "speech_timing" in intent:
        raise UnsupportedEpisodeIntent(
            "calibrated intent cannot override ChildLens-derived speech timing"
        )
    if conditioned_sample is not None:
        if mode != "calibrated":
            raise UnsupportedEpisodeIntent("conditioned_sample requires calibrated mode")
        if conditioned_sample.get("calibration_sha256") != calibration.source_sha256:
            raise UnsupportedEpisodeIntent("conditioned_sample calibration hash mismatch")

    seed = int(intent.get("seed", protocol["seed"]))
    duration_s = float(intent.get("duration_s", 30.0))
    if duration_s <= 0 or duration_s > 30.0:
        raise UnsupportedEpisodeIntent("duration_s must be in (0, 30]")
    if conditioned_sample is None:
        phase_durations = [duration_s / len(phases)] * len(phases)
    else:
        phase_durations = [
            float(item) for item in conditioned_sample["phase_durations_s"]
        ]
        if len(phase_durations) != len(phases) or any(
            item <= 0 for item in phase_durations
        ):
            raise UnsupportedEpisodeIntent("invalid conditioned phase durations")
        if abs(sum(phase_durations) - duration_s) > 1e-5:
            raise UnsupportedEpisodeIntent("conditioned phase durations must sum to duration")
    phase_edges = np.cumsum([0.0, *phase_durations])
    phase_timestamps = [
        {
            "phase": phase,
            "start_s": round(float(phase_edges[index]), 9),
            "end_s": round(float(phase_edges[index + 1]), 9),
        }
        for index, phase in enumerate(phases)
    ]
    exact_names = int(intent.get("speech", {}).get("name_count", 1))
    if exact_names < 0 or exact_names > 4:
        raise UnsupportedEpisodeIntent("speech.name_count must be between 0 and 4")
    if conditioned_sample is not None:
        sampled_events = conditioned_sample.get("speech_events", [])
        if len(sampled_events) != exact_names:
            raise UnsupportedEpisodeIntent("conditioned speech event count mismatch")
        speech_events = []
        for event in sampled_events:
            start_s = float(event["start_s"])
            event_duration_s = float(event["duration_s"])
            if start_s < 0 or event_duration_s <= 0 or start_s + event_duration_s > duration_s:
                raise UnsupportedEpisodeIntent("conditioned speech event outside episode")
            speech_events.append(
                {
                    "start_s": round(start_s, 9),
                    "duration_s": round(event_duration_s, 9),
                    "semantic_constraint": "name_target",
                    "text": event.get("text", intent.get("speech", {}).get("text", "die Tasse")),
                    "timing_provenance": "childlens_conditioned_sampler",
                    "status": "local_synthetic_timing_interface_not_human_validation",
                }
            )
        speech_provenance = "childlens_conditioned_sampler"
    elif mode == "calibrated":
        speech_gap = calibration.speech_gap_s
        speech_start = max(0.5, phase_timestamps[0]["end_s"])
        speech_provenance = "childlens_calibration"
    else:
        requested = intent.get("speech_timing", {})
        speech_gap = float(requested.get("gap_s", calibration.speech_gap_s))
        speech_start = float(requested.get("start_s", 1.0))
        speech_provenance = (
            "user_intent" if requested else "childlens_calibration"
        )
    if conditioned_sample is None:
        speech_events = [
            {
                "start_s": round(speech_start + index * speech_gap, 9),
                "duration_s": None,
                "semantic_constraint": "name_target",
                "text": intent.get("speech", {}).get("text", "die Tasse"),
                "timing_provenance": speech_provenance,
                "status": "local_synthetic_timing_interface_not_human_validation",
            }
            for index in range(exact_names)
        ]
    resolved = {
        "schema": "ResolvedEpisodeSpec",
        "protocol_id": protocol["protocol_id"],
        "mode": _field(mode, "user_intent"),
        "canonical_scientific_batch_admissible": _field(
            mode == "calibrated", "derived"
        ),
        "activity_family": _field(intent["activity_family"], "user_intent"),
        "scene": {
            "id": _field(scene_id, "user_intent"),
            "version": _field(intent["scene"]["version"], "asset_sampler"),
            "asset_sha256": _field(
                intent["scene"]["asset_sha256"], "asset_sampler"
            ),
        },
        "target": {
            "asset_id": _field(object_id, "user_intent"),
            "asset_sha256": _field(
                asset_registry[object_id]["sha256"], "asset_sampler"
            ),
            "scale_m": _field(asset_registry[object_id]["scale_m"], "asset_sampler"),
            "geometry": _field(
                asset_registry[object_id]["geometry"], "asset_sampler"
            ),
            "rgba": _field(asset_registry[object_id]["rgba"], "asset_sampler"),
        },
        "embodiment": {
            "id": _field("MIMo-v2-24-month", "controller"),
            "age_months": _field(24, "controller"),
        },
        "duration_s": _field(duration_s, "user_intent"),
        "phase_timestamps": _field(phase_timestamps, "derived"),
        "controller": _field(intent.get("controller", {}), "controller"),
        "camera": _field(intent.get("camera", {}), "user_intent"),
        "speech_events": _field(speech_events, speech_provenance),
        "speech_semantic_constraint": _field(
            {
                "name_count": exact_names,
                "calibrated_mode_status": (
                    "permitted_conditional_semantic_constraint; timing remains calibrated"
                    if mode == "calibrated"
                    else "demonstration_only"
                ),
            },
            "user_intent",
        ),
        "seed": _field(seed, "user_intent"),
        "units": _field(protocol["units"], "derived"),
        "coordinate_frames": _field(protocol["coordinate_frames"], "derived"),
        "calibration": {
            "record_sha256": _field(
                calibration.source_sha256, "childlens_calibration"
            ),
            "speech_bout_duration_s": _field(
                calibration.speech_bout_duration_s, "childlens_calibration"
            ),
            "speech_gap_s": _field(
                calibration.speech_gap_s, "childlens_calibration"
            ),
            "speech_density_per_minute": _field(
                calibration.speech_density_per_minute, "childlens_calibration"
            ),
            "motion_mean": _field(calibration.motion_mean, "childlens_calibration"),
        },
    }
    resolved["spec_sha256"] = content_hash(resolved)
    return resolved
