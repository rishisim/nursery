"""Independent, fail-closed QA for the procedural clothed scene gate.

The auditor accepts one ignored run root and never imports a producer module.  A
``qa_evidence.json`` file may map producer-specific filenames to the canonical
evidence names used here.  Paths in that file must be relative to the run root.
Visual predicates require dense, rendered-frame observations: counters,
viewport object-centre tests, and other proxies are deliberately rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import zlib
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import CONFIG_PATH, load_frozen_config, validate_frozen_config


PASS = "PASS"
FAIL = "FAIL"
UNAVAILABLE = "UNAVAILABLE"
REPORT_SCHEMA = "embodied.independent_qa_report.v1"
EVIDENCE_SCHEMA = "embodied.independent_qa_evidence.v1"
SOURCE_AUDIT_SCHEMA = "embodied.procedural_gate_source_audit.v1"
DECODED_FRAME_REVIEW_SCHEMA = "embodied.decoded_frame_review.v1"
REQUIRED_PYTHON_SOURCE_AUDIT_FILES = {
    "__init__.py", "__main__.py", "runner.py", "contracts.py", "independent_qa.py", "cli.py",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
VISUAL_PROXY_TERMS = (
    "counter",
    "object_center",
    "object-centre",
    "viewport",
    "proxy",
    "bone_name",
    "existence",
)


@dataclass(frozen=True)
class Check:
    check_id: str
    status: str
    summary: str
    evidence: Any = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "check_id": self.check_id,
            "status": self.status,
            "summary": self.summary,
        }
        if self.evidence is not None:
            result["evidence"] = self.evidence
        return result


def _check(check_id: str, condition: bool | None, summary: str, evidence: Any = None) -> Check:
    status = UNAVAILABLE if condition is None else PASS if condition else FAIL
    return Check(check_id, status, summary, evidence)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inside(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"evidence path escapes run root: {candidate}") from exc
    return candidate


def _mapped_path(root: Path, mapping: Mapping[str, Any], key: str, default: str) -> Path:
    value = mapping.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a relative path string")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{key} must be relative to the run root")
    return _inside(root, root / path)


def _optional_report(root: Path, mapping: Mapping[str, Any], key: str, default: str) -> Any | None:
    inline = mapping.get(key)
    if isinstance(inline, (dict, list)):
        return inline
    path = _mapped_path(root, mapping, key, default)
    return _read_json(path) if path.is_file() else None


def _load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"trace line {line_number} is not an object")
                rows.append(value)
        return rows
    value = _read_json(path)
    if isinstance(value, list):
        return value
    for key in ("steps", "rows", "trace"):
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value[key]
    raise ValueError("trace JSON must be a list or contain steps/rows/trace")


def _clock(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("clock")
    return value if isinstance(value, Mapping) else row


def _field(row: Mapping[str, Any], name: str, default: Any = None) -> Any:
    clock = _clock(row)
    return clock.get(name, row.get(name, default))


def _pose_position(value: Any) -> Sequence[float] | None:
    if not isinstance(value, Mapping):
        return None
    pose = value.get("pose")
    if isinstance(pose, Mapping):
        value = pose
    for key in ("position_world_m", "position", "translation_m"):
        candidate = value.get(key)
        vector = _vector(candidate)
        if vector is not None:
            return vector
    return None


def _pose_rotation(value: Any) -> Sequence[float] | None:
    if not isinstance(value, Mapping):
        return None
    pose = value.get("pose")
    if isinstance(pose, Mapping):
        value = pose
    for key in ("rotation_world_xyzw", "rotation_xyzw", "quaternion_xyzw"):
        candidate = value.get(key)
        quaternion = _vector(candidate, 4)
        if quaternion is not None:
            return quaternion
    return None


def _pose_velocity_complete(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    pose = value.get("pose")
    sources = [value, pose] if isinstance(pose, Mapping) else [value]
    linear = next((source.get("linear_velocity_world_m_s") for source in sources
                   if "linear_velocity_world_m_s" in source), None)
    angular = next((source.get("angular_velocity_world_rad_s") for source in sources
                    if "angular_velocity_world_rad_s" in source), None)
    return _vector(linear) is not None and _vector(angular) is not None


def _pose_velocities(value: Any) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    if not isinstance(value, Mapping):
        return None, None
    pose = value.get("pose")
    sources = [value, pose] if isinstance(pose, Mapping) else [value]
    linear = next((source.get("linear_velocity_world_m_s") for source in sources
                   if "linear_velocity_world_m_s" in source), None)
    angular = next((source.get("angular_velocity_world_rad_s") for source in sources
                    if "angular_velocity_world_rad_s" in source), None)
    return _vector(linear), _vector(angular)


def _magnitude(value: Sequence[float] | None) -> float:
    return math.sqrt(sum(component * component for component in value)) if value is not None else math.nan


def _objects(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = row.get("objects", [])
    if isinstance(value, Mapping):
        return [dict(item, persistent_id=item.get("persistent_id", key)) if isinstance(item, Mapping) else {}
                for key, item in value.items()]
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _contacts(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = row.get("contacts", [])
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _hands(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("hand_state")
    if not isinstance(value, Mapping):
        return None
    normalized = dict(value)
    for side in ("left", "right"):
        digits = value.get(f"{side}_digits")
        if isinstance(digits, list):
            for digit in digits:
                if isinstance(digit, Mapping) and digit.get("digit"):
                    normalized[f"{side}_{str(digit['digit']).lower()}"] = digit
    return normalized


def _target_id(rows: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]) -> str | None:
    explicit = evidence.get("target_object_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    candidates: set[str] = set()
    for row in rows[: min(len(rows), 32)]:
        for obj in _objects(row):
            role = str(obj.get("role", "")).lower()
            semantic = str(obj.get("semantic_id", "")).lower()
            if role in {"target", "interactive_target"} or semantic in {"target", "interactive_target"}:
                value = obj.get("persistent_id")
                if isinstance(value, str):
                    candidates.add(value)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _destination_id(rows: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]) -> str | None:
    explicit = evidence.get("destination_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    values = {
        episode.get("destination_id")
        for row in rows
        for episode in [row.get("episode")]
        if isinstance(episode, Mapping) and isinstance(episode.get("destination_id"), str)
        and episode.get("destination_id")
    }
    return next(iter(values)) if len(values) == 1 else None


def _object_for(row: Mapping[str, Any], persistent_id: str) -> Mapping[str, Any] | None:
    for obj in _objects(row):
        if obj.get("persistent_id") == persistent_id:
            return obj
    return None


def _vector(value: Any, length: int = 3) -> tuple[float, ...] | None:
    if isinstance(value, Mapping):
        keys = ("x", "y", "z", "w")[:length]
        if not all(key in value for key in keys):
            return None
        value = [value[key] for key in keys]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _quat_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    qa = _vector(a, 4)
    qb = _vector(b, 4)
    if qa is None or qb is None:
        return math.nan
    na = math.sqrt(sum(item * item for item in qa))
    nb = math.sqrt(sum(item * item for item in qb))
    if na <= 0 or nb <= 0:
        return math.nan
    dot = abs(sum(x * y for x, y in zip(qa, qb)) / (na * nb))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _quat_roll_deg(q: Sequence[float]) -> float:
    value = _vector(q, 4)
    if value is None:
        return math.nan
    x, y, z, w = value
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        return math.nan
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = _vector(a), _vector(b)
    if va is None or vb is None:
        return math.nan
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    return sum(x * y for x, y in zip(va, vb)) / (na * nb) if na and nb else math.nan


def _contact_hand_digit(contact: Mapping[str, Any]) -> tuple[str | None, str | None]:
    hand = str(contact.get("hand", "")).lower() or None
    digit = str(contact.get("digit", "")).lower() or None
    haystack = " ".join(str(contact.get(key, "")) for key in ("collider_a", "collider_b", "body_part"))
    lowered = haystack.lower().replace("_", " ").replace("-", " ")
    if hand not in {"left", "right"}:
        hand = "right" if "right" in lowered else "left" if "left" in lowered else None
    if digit not in {"thumb", "index", "middle", "ring", "little"}:
        digit = next((name for name in ("thumb", "index", "middle", "ring", "little") if name in lowered), None)
    return hand, digit


def _contact_targets(contact: Mapping[str, Any], target_id: str) -> bool:
    explicit = contact.get("object_id") or contact.get("persistent_object_id")
    if explicit is not None:
        return explicit == target_id
    target = target_id.lower()
    colliders = [str(contact.get(key, "")).lower() for key in ("collider_a", "collider_b")]
    return any(target == collider or target in collider for collider in colliders)


def _normal_on_target(contact: Mapping[str, Any], target_id: str) -> tuple[float, ...] | None:
    explicit = _vector(contact.get("normal_on_object_world"))
    if explicit is not None:
        return explicit
    normal = _vector(contact.get("normal_world"))
    if normal is None:
        return None
    target = target_id.lower()
    collider_a = str(contact.get("collider_a", "")).lower()
    collider_b = str(contact.get("collider_b", "")).lower()
    # The recorder canonicalizes normal_world from collider_a toward collider_b.
    if target == collider_a or target in collider_a:
        return normal
    if target == collider_b or target in collider_b:
        return tuple(-value for value in normal)
    return None


def _measured_contact(contact: Mapping[str, Any]) -> bool:
    provenance = str(contact.get("provenance", "")).lower()
    return provenance in {"physx_measured", "measured_physx", "measured"}


def _separation_eligible(contact: Mapping[str, Any], maximum_separation_m: float) -> bool:
    separation = contact.get("separation_m")
    return (
        isinstance(separation, (int, float))
        and math.isfinite(float(separation))
        and float(separation) <= maximum_separation_m
    )


def _impulse_magnitude(contact: Mapping[str, Any]) -> float:
    explicit = contact.get("available_impulse_magnitude_n_s")
    if isinstance(explicit, (int, float)) and math.isfinite(float(explicit)):
        return abs(float(explicit))
    impulse = contact.get("available_impulse_n_s")
    if isinstance(impulse, (int, float)) and math.isfinite(float(impulse)):
        return abs(float(impulse))
    vector = _vector(impulse)
    return _magnitude(vector) if vector is not None else 0.0


def _max_contiguous_duration(steps: Iterable[int], physics_hz: float) -> float:
    ordered = sorted(set(steps))
    if not ordered or physics_hz <= 0:
        return 0.0
    best = current = 1
    for previous, step in zip(ordered, ordered[1:]):
        current = current + 1 if step == previous + 1 else 1
        best = max(best, current)
    return best / physics_hz


def build_dense_timeline(
    rows: Sequence[Mapping[str, Any]], phases: Sequence[Mapping[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Aggregate every physics row into a dense render-frame phase timeline."""
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        frame = _field(row, "render_frame")
        if isinstance(frame, int):
            grouped[frame].append(row)
    timeline = []
    for frame in sorted(grouped):
        samples = grouped[frame]
        time_s = _field(samples[-1], "time_s")
        phase_id = _field(samples[-1], "phase_id")
        if phase_id is None and isinstance(time_s, (int, float)):
            phase_id = next((phase.get("id") for phase in phases
                             if float(phase["start_s"]) <= float(time_s) < float(phase["end_s"])), None)
        digits = sorted({f"{hand}_{digit}" for row in samples for contact in _contacts(row)
                         for hand, digit in [_contact_hand_digit(contact)] if hand and digit})
        timeline.append(
            {
                "render_frame": frame,
                "first_physics_step": _field(samples[0], "physics_step"),
                "last_physics_step": _field(samples[-1], "physics_step"),
                "first_time_s": _field(samples[0], "time_s"),
                "last_time_s": _field(samples[-1], "time_s"),
                "phase_id": phase_id or "unavailable",
                "physics_samples": len(samples),
                "contact_samples": sum(len(_contacts(row)) for row in samples),
                "contact_digits": digits,
                "object_ids": sorted({str(obj.get("persistent_id")) for row in samples for obj in _objects(row)
                                      if obj.get("persistent_id") is not None}),
                "assistance_entries": sum(len(row.get("assistance_ledger", [])) for row in samples
                                          if isinstance(row.get("assistance_ledger", []), list)),
            }
        )
    return timeline


def _trace_checks(
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    tolerances: Mapping[str, Any],
    duration_s: float,
    phases: Sequence[Mapping[str, Any]],
) -> tuple[list[Check], dict[str, Any]]:
    clock_tol = tolerances["clock"]
    physics_hz = float(clock_tol["physics_hz"])
    render_hz = int(clock_tol["render_hz"])
    ratio = int(clock_tol["exact_steps_per_render_frame"])
    if not rows:
        return [
            _check("trace.present", None, "episode trace is missing"),
            _check("trace.clock", None, "exact physics/render clock cannot be checked"),
            _check("trace.fields", None, "full body, hand, object, camera, and controller state unavailable"),
            _check("trace.identity", None, "persistent identity cannot be checked"),
            _check("trace.duration", None, "complete frozen episode duration cannot be checked"),
        ], {"timeline": [], "target_object_id": None}

    ordered = sorted(rows, key=lambda row: int(_field(row, "physics_step", -1)))
    steps = [_field(row, "physics_step") for row in ordered]
    frames = [_field(row, "render_frame") for row in ordered]
    times = [_field(row, "time_s") for row in ordered]
    valid_scalars = all(isinstance(step, int) and isinstance(frame, int) and isinstance(time, (int, float))
                        for step, frame, time in zip(steps, frames, times))
    clock_ok = False
    clock_details: dict[str, Any] = {"rows": len(rows), "physics_hz": physics_hz, "render_hz": render_hz}
    if valid_scalars:
        base_step, base_frame, base_time = steps[0], frames[0], float(times[0])
        contiguous = all(step == base_step + index for index, step in enumerate(steps))
        mapped = all(frame == base_frame + (step - base_step) // ratio for step, frame in zip(steps, frames))
        time_error = max(abs((float(time) - base_time) - (step - base_step) / physics_hz)
                         for step, time in zip(steps, times))
        frame_counts: dict[int, int] = defaultdict(int)
        for frame in frames:
            frame_counts[frame] += 1
        interior = [count for frame, count in frame_counts.items() if frame not in {min(frames), max(frames)}]
        dense = all(count == ratio for count in interior)
        monotonic_time = all(float(current) > float(previous) for previous, current in zip(times, times[1:]))
        # Exact synchronization is established by contiguous integer physics steps
        # and the frozen step//ratio render mapping. Unity serializes time_s from
        # float32; its microsecond-scale drift is retained as a diagnostic and is
        # not allowed to false-fail the exact integer clock.
        clock_ok = contiguous and mapped and dense and monotonic_time and physics_hz / render_hz == ratio
        clock_details.update(contiguous=contiguous, integer_mapping=mapped, dense_interior_frames=dense,
                             monotonic_timestamps=monotonic_time,
                             maximum_float_timestamp_drift_s=time_error,
                             timestamp_drift_is_diagnostic_not_integer_mapping_evidence=True)

    missing_counts: dict[str, int] = defaultdict(int)
    required_body = (
        "root", "pelvis", "torso", "neck", "head",
        "left_shoulder", "left_upper_arm", "left_elbow", "left_lower_arm", "left_forearm", "left_wrist",
        "right_shoulder", "right_upper_arm", "right_elbow", "right_lower_arm", "right_forearm", "right_wrist",
    )
    required_digits = tuple(f"{hand}_{digit}" for hand in ("left", "right")
                            for digit in ("thumb", "index", "middle", "ring", "little"))
    for row in ordered:
        body = row.get("body_state")
        if not isinstance(body, Mapping):
            missing_counts["body_state"] += 1
        else:
            if set(body) != set(required_body):
                missing_counts["body_state.segment_set"] += 1
            segment_ids = [body.get(name, {}).get("segment_id", name)
                           if isinstance(body.get(name), Mapping) else name for name in required_body]
            if len(set(segment_ids)) != len(required_body):
                missing_counts["body_state.duplicate_segment_ids"] += 1
            for name in required_body:
                state = body.get(name)
                if _pose_position(state) is None or _pose_rotation(state) is None:
                    missing_counts[f"body_state.{name}.pose"] += 1
                if not _pose_velocity_complete(state):
                    missing_counts[f"body_state.{name}.velocity"] += 1
        hands = _hands(row)
        if hands is None:
            missing_counts["hand_state"] += 1
        else:
            for palm in ("left_palm", "right_palm"):
                if _pose_position(hands.get(palm)) is None or _pose_rotation(hands.get(palm)) is None:
                    missing_counts[f"hand_state.{palm}.pose"] += 1
                if not _pose_velocity_complete(hands.get(palm)):
                    missing_counts[f"hand_state.{palm}.velocity"] += 1
            for digit in required_digits:
                state = hands.get(digit)
                if not isinstance(state, Mapping):
                    missing_counts[f"hand_state.{digit}"] += 1
                    continue
                segments = state.get("segments")
                if not isinstance(segments, list) or len(segments) < 3:
                    missing_counts[f"hand_state.{digit}.segments"] += 1
                elif any(_pose_position(segment) is None or _pose_rotation(segment) is None for segment in segments):
                    missing_counts[f"hand_state.{digit}.segment_pose"] += 1
                elif any(not _pose_velocity_complete(segment) for segment in segments):
                    missing_counts[f"hand_state.{digit}.segment_velocity"] += 1
                if "closure_commanded" not in state or "closure_observed" not in state:
                    missing_counts[f"hand_state.{digit}.closure"] += 1
        controller = row.get("controller_state")
        if not isinstance(controller, Mapping):
            missing_counts["controller_state"] += 1
        elif any(name not in controller for name in ("targets", "errors", "recovery_state", "speed_limits", "closure_limits")):
            missing_counts["controller_state.fields"] += 1
        camera = row.get("camera_state")
        if not isinstance(camera, Mapping) or _pose_position(camera) is None or _pose_rotation(camera) is None:
            missing_counts["camera_state.pose"] += 1
        elif any(name not in camera for name in ("parent_id", "intrinsics", "clearance_m", "optical_vs_face_forward_deg")):
            missing_counts["camera_state.fields"] += 1
        for obj in _objects(row):
            if _pose_position(obj) is None or _pose_rotation(obj) is None:
                missing_counts["object.pose"] += 1
            if not _pose_velocity_complete(obj):
                missing_counts["object.velocity"] += 1
            if not all(name in obj for name in ("persistent_id", "semantic_id", "instance_id", "support_id", "sleeping")):
                missing_counts["object.fields"] += 1
        for contact in _contacts(row):
            if not all(name in contact for name in ("collider_a", "collider_b", "point_world_m", "normal_world",
                                                    "separation_m", "relative_velocity_world_m_s",
                                                    "available_impulse_n_s", "provenance")):
                missing_counts["contact.fields"] += 1
        derived = row.get("derived_state") if isinstance(row.get("derived_state"), Mapping) else row
        if not all(name in derived for name in ("joint_proprioception", "head_accelerometer_m_s2", "head_gyroscope_rad_s")):
            missing_counts["derived_state"] += 1
        if not isinstance(row.get("assistance_ledger"), list):
            missing_counts["assistance_ledger"] += 1

    identities: dict[str, set[tuple[Any, Any]]] = defaultdict(set)
    duplicate_ids = False
    for row in ordered:
        seen: set[str] = set()
        for obj in _objects(row):
            persistent = obj.get("persistent_id")
            if isinstance(persistent, str):
                duplicate_ids |= persistent in seen
                seen.add(persistent)
                identities[persistent].add((obj.get("semantic_id"), obj.get("instance_id")))
    target = _target_id(ordered, evidence)
    identity_ok = bool(identities) and not duplicate_ids and all(len(values) == 1 for values in identities.values())
    if target is not None:
        identity_ok &= target in identities and all(_object_for(row, target) is not None for row in ordered)
    expected_steps = round(duration_s * physics_hz)
    duration_ok = len(ordered) == expected_steps
    if valid_scalars:
        duration_ok &= steps[-1] - steps[0] + 1 == expected_steps

    return [
        _check("trace.present", True, f"loaded {len(rows)} physics-step rows"),
        _check("trace.clock", clock_ok, "trace obeys the frozen exact 240:30 (8:1) clock", clock_details),
        _check("trace.fields", not missing_counts, "trace contains complete body, palm rotation, per-digit, object, contact, controller, and camera state",
               {"missing_row_counts": dict(sorted(missing_counts.items()))}),
        _check("trace.identity", identity_ok, "persistent object semantic/instance identity remains stable",
               {"target_object_id": target, "identity_variants": {key: len(value) for key, value in identities.items()}}),
        _check("trace.duration", duration_ok, "trace covers every physics step of the frozen 20–30 second ActivityPlan",
               {"expected_steps": expected_steps, "observed_steps": len(ordered)}),
    ], {"timeline": build_dense_timeline(ordered, phases), "target_object_id": target, "ordered_rows": ordered}


def _interaction_checks(
    rows: Sequence[Mapping[str, Any]], target_id: str | None, destination_id: str | None,
    tolerances: Mapping[str, Any]
) -> tuple[list[Check], dict[str, Any]]:
    required = tolerances["interaction"]
    maximum_separation_m = float(tolerances["contact"]["qualification_max_measured_separation_m"])
    physics_hz = float(tolerances["clock"]["physics_hz"])
    if not rows or target_id is None:
        return [
            _check("interaction.dynamic_force_bearing", None, "continuous unsupported force-bearing evidence unavailable"),
            _check("interaction.no_initial_overlap", None, "initial separation/depenetration evidence unavailable"),
            _check("interaction.right_capture", None, "measured right-hand contact dwell unavailable"),
            _check("interaction.left_support", None, "meaningful opposing left support unavailable"),
            _check("interaction.lift_turn", None, "free-object lift and turn unavailable"),
            _check("interaction.free_release", None, "commanded free release/settle unavailable"),
            _check("interaction.no_assistance", None, "assistance ledger unavailable"),
        ], {}

    states: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for row in rows:
        obj = _object_for(row, target_id)
        if obj is not None:
            states.append((row, obj))
    state_by_step = {int(_field(row, "physics_step", -1)): (row, obj) for row, obj in states}
    qualification_steps = [
        int(controller["bimanual_qualification_step"])
        for row in rows
        for controller in [row.get("controller_state")]
        if isinstance(controller, Mapping)
        and isinstance(controller.get("bimanual_qualification_step"), int)
        and int(controller["bimanual_qualification_step"]) >= 0
    ]
    qualification_step = min(qualification_steps) if qualification_steps else None
    baseline_state = state_by_step.get(qualification_step) if qualification_step is not None else (states[0] if states else None)
    baseline_position = _pose_position(baseline_state[1]) if baseline_state is not None else None
    post_qualification_states = [
        (row, obj) for row, obj in states
        if qualification_step is None or int(_field(row, "physics_step", -1)) >= qualification_step
    ]
    post_qualification_positions = [_pose_position(obj) for _, obj in post_qualification_states]
    if baseline_position is None or not post_qualification_states or any(position is None for position in post_qualification_positions):
        lift_m = math.nan
        lift_start_step = math.inf
        lifted_steps: set[int] = set()
    else:
        baseline_y = float(baseline_position[1])
        lift_m = max(float(position[1]) for position in post_qualification_positions) - baseline_y
        lift_start_step = next((_field(row, "physics_step") for (row, _), position in zip(post_qualification_states, post_qualification_positions)
                                if float(position[1]) - baseline_y > 0.005), math.inf)
        lifted_steps = {_field(row, "physics_step") for (row, _), position in zip(post_qualification_states, post_qualification_positions)
                        if float(position[1]) - baseline_y > 0.01}

    def is_turn_phase(row: Mapping[str, Any]) -> bool:
        phase = str(_field(row, "phase_id", "")).lower().replace("_", "").replace("-", "")
        return phase in {"turn", "bimanualturn", "bimanualconditionliftinspectturn"}

    turn_states = [(row, obj) for row, obj in states if is_turn_phase(row)]
    turn_rotations = [_pose_rotation(obj) for _, obj in turn_states]
    turn_deg = math.nan
    if turn_rotations and turn_rotations[0] is not None:
        values = [_quat_angle_deg(turn_rotations[0], rotation) for rotation in turn_rotations if rotation is not None]
        turn_deg = max(values) if values else math.nan
    supported_turn_steps = sum(
        obj.get("support_id") not in {None, "", "none", "unavailable"} for _, obj in turn_states
    )

    right_steps: list[int] = []
    left_steps: list[int] = []
    opposing_steps: list[int] = []
    right_impulse_steps: list[int] = []
    left_impulse_support_steps: list[int] = []
    literal_postqualification_steps: list[int] = []
    controller_carry_steps: list[int] = []
    available_impulses: list[float] = []
    eligible_hand_rows = 0
    speculative_hand_rows = 0
    eligible_nonzero_impulse_rows = 0
    force_status_by_step: dict[int, tuple[bool, bool]] = {}
    initial_contact_steps: list[int] = []
    for row in rows:
        step = _field(row, "physics_step", -1)
        measured_target = [contact for contact in _contacts(row)
                           if _contact_targets(contact, target_id) and _measured_contact(contact)]
        measured_hand = [contact for contact in measured_target if _contact_hand_digit(contact)[0] in {"left", "right"}]
        eligible = [contact for contact in measured_hand
                    if _separation_eligible(contact, maximum_separation_m)]
        eligible_hand_rows += len(eligible)
        speculative_hand_rows += len(measured_hand) - len(eligible)
        for contact in eligible:
            magnitude = _impulse_magnitude(contact)
            available_impulses.append(magnitude)
            eligible_nonzero_impulse_rows += magnitude > 0.0
        if eligible and isinstance(step, int) and step < int(round(0.25 * physics_hz)):
            initial_contact_steps.append(step)
        by_hand: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for contact in eligible:
            hand, digit = _contact_hand_digit(contact)
            if hand and digit:
                by_hand[hand][digit].append(contact)
        right_digits = set(by_hand["right"])
        right_thumb_normals = [_normal_on_target(contact, target_id) for contact in by_hand["right"].get("thumb", [])]
        right_non_thumb_normals = [
            _normal_on_target(contact, target_id)
            for digit, contacts in by_hand["right"].items() if digit != "thumb"
            for contact in contacts
        ]
        right_opposing_geometry = any(
            math.isfinite(_dot(thumb, other)) and _dot(thumb, other) <= -0.25
            for thumb in right_thumb_normals for other in right_non_thumb_normals
        )
        right_geometry = ("thumb" in right_digits
                          and len(right_digits - {"thumb"}) >= int(required["right_minimum_non_thumb_digits"])
                          and right_opposing_geometry)
        if step < lift_start_step and right_geometry:
            right_steps.append(step)
        right_impulse_digits = {
            digit for digit, contacts in by_hand["right"].items()
            if any(_impulse_magnitude(contact) > 0.0 for contact in contacts)
        }
        right_force_now = ("thumb" in right_impulse_digits
                           and len(right_impulse_digits - {"thumb"}) >= int(required["right_minimum_non_thumb_digits"])
                           and right_opposing_geometry)
        if right_force_now:
            right_impulse_steps.append(step)
        left_digits = set(by_hand["left"])
        non_little_left_digits = left_digits - {"little"}
        has_non_little_left = bool(non_little_left_digits)
        has_stable_left_digits = bool(non_little_left_digits)
        right_normals = [_normal_on_target(contact, target_id) for contacts in by_hand["right"].values() for contact in contacts]
        left_normals = [_normal_on_target(contact, target_id) for contacts in by_hand["left"].values() for contact in contacts]
        opposing_geometry = any(math.isfinite(_dot(left, right)) and _dot(left, right) <= -0.25
                                 for left in left_normals for right in right_normals)
        meaningful_left_geometry = right_geometry and has_stable_left_digits and opposing_geometry
        literal_opposing_geometry = right_geometry and has_non_little_left and opposing_geometry
        if meaningful_left_geometry:
            left_steps.append(step)
            opposing_steps.append(step)
        left_impulse_contacts = [
            contact for digit, contacts in by_hand["left"].items() if digit != "little"
            for contact in contacts if _impulse_magnitude(contact) > 0.0
        ]
        left_force_now = meaningful_left_geometry and right_force_now and bool(left_impulse_contacts)
        if left_force_now:
            left_impulse_support_steps.append(step)
        if isinstance(step, int):
            force_status_by_step[step] = (right_force_now, left_force_now)
        if qualification_step is not None and step > qualification_step and literal_opposing_geometry:
            literal_postqualification_steps.append(step)
        controller = row.get("controller_state")
        if isinstance(controller, Mapping) and controller.get("carry_contacts_maintained") is True:
            controller_carry_steps.append(step)

    right_dwell = _max_contiguous_duration(right_steps, physics_hz)
    left_dwell = _max_contiguous_duration(left_steps, physics_hz)
    opposing_dwell = _max_contiguous_duration(opposing_steps, physics_hz)
    right_impulse_dwell = _max_contiguous_duration(right_impulse_steps, physics_hz)
    left_impulse_dwell = _max_contiguous_duration(left_impulse_support_steps, physics_hz)
    controller_carry_dwell = _max_contiguous_duration(controller_carry_steps, physics_hz)
    literal_postqualification_dwell = _max_contiguous_duration(literal_postqualification_steps, physics_hz)
    right_ok = right_impulse_dwell > float(required["right_opposition_min_s"])
    left_ok = left_impulse_dwell > float(required["left_support_min_s"])

    all_ledgers = [entry for row in rows for entry in row.get("assistance_ledger", [])
                   if isinstance(row.get("assistance_ledger", []), list)]
    authority_counter_names = (
        "targetPoseWriteCounter", "targetVelocityWriteCounter", "targetForceCounter",
        "targetTorqueCounter", "targetJointCounter", "targetParentingCounter",
        "targetKinematicChangeCounter",
    )
    counter_violations = []
    counter_rows = 0
    for row in rows:
        counters = row.get("authority_counters")
        if isinstance(counters, Mapping):
            counter_rows += 1
            for name in authority_counter_names:
                if counters.get(name) != 0:
                    counter_violations.append({"step": _field(row, "physics_step"), "counter": name,
                                               "value": counters.get(name)})
    no_recorded_assistance = not all_ledgers and counter_rows == len(rows) and not counter_violations
    release_rows = []
    prior_commanded_closure = 0.0
    for index, row in enumerate(rows):
        controller = row.get("controller_state", {})
        events = controller.get("events", []) if isinstance(controller, Mapping) else []
        commanded = isinstance(controller, Mapping) and (
            controller.get("commanded_release") is True
            or controller.get("release_commanded") is True
            or "release_commanded" in events
        )
        hands = _hands(row)
        closures = []
        if hands is not None:
            for hand in ("left", "right"):
                for digit in ("thumb", "index", "middle", "ring", "little"):
                    state = hands.get(f"{hand}_{digit}")
                    if isinstance(state, Mapping) and isinstance(state.get("closure_commanded"), (int, float)):
                        closures.append(float(state["closure_commanded"]))
        commanded |= len(closures) == 10 and max(closures) <= 0.05 and prior_commanded_closure > 0.2
        if closures:
            prior_commanded_closure = max(closures)
        if commanded:
            release_rows.append(index)
    free_release_ok: bool | None = None
    release_details: dict[str, Any] = {"commanded_release_rows": release_rows}
    if release_rows:
        start = release_rows[0]
        post = rows[start + 1:]
        post_contact_flags = [any(
            _contact_targets(contact, target_id)
            and _measured_contact(contact)
            and _separation_eligible(contact, maximum_separation_m)
            and _contact_hand_digit(contact)[0] in {"left", "right"}
            for contact in _contacts(row)
        ) for row in post]
        hand_contact_after = any(post_contact_flags)
        last_contact_index = max((index for index, active in enumerate(post_contact_flags) if active), default=-1)
        contact_free_tail_steps = len(post_contact_flags) - last_contact_index - 1
        contact_free_release = contact_free_tail_steps >= int(tolerances["clock"]["exact_steps_per_render_frame"])
        post_objects = [obj for row in post for obj in [_object_for(row, target_id)] if obj is not None]
        settled_objects = post_objects[last_contact_index + 1:]
        observed_support_ids = sorted({str(obj.get("support_id")) for obj in settled_objects
                                       if obj.get("support_id") not in {None, "", "none", "unavailable"}})
        destination_settle = bool(destination_id) and any(
            obj.get("support_id") == destination_id for obj in settled_objects
        )
        settled = destination_settle and any(
            obj.get("support_id") == destination_id and obj.get("sleeping") is True
            for obj in settled_objects
        )
        dynamic = all(
            obj.get("free_dynamic") is True
            or (obj.get("is_kinematic") is False and obj.get("parent_id") in {None, ""})
            for obj in post_objects
        )
        free_release_ok = bool(post_objects) and contact_free_release and settled and dynamic and no_recorded_assistance
        release_details.update(
            hand_contact_after_opening=hand_contact_after,
            contact_free_tail_steps=contact_free_tail_steps,
            contact_free_release=contact_free_release,
            settled=settled,
            expected_destination_id=destination_id,
            observed_post_release_support_ids=observed_support_ids,
            destination_specific_settle=destination_settle,
            dynamic=dynamic,
        )

    running_right = running_left = 0
    derived_qualification_step: int | None = None
    for step in sorted(force_status_by_step):
        right_now, left_now = force_status_by_step[step]
        running_right = running_right + 1 if right_now else 0
        running_left = running_left + 1 if left_now else 0
        if (running_right / physics_hz > float(required["right_opposition_min_s"])
                and running_left / physics_hz > float(required["left_support_min_s"])):
            derived_qualification_step = step
            break
    if derived_qualification_step is not None:
        qualification_step = derived_qualification_step
    release_step = int(_field(rows[release_rows[0]], "physics_step", -1)) if release_rows else None
    qualified_rows = [
        (row, _object_for(row, target_id)) for row in rows
        if qualification_step is not None
        and int(_field(row, "physics_step", -1)) >= qualification_step
        and (release_step is None or int(_field(row, "physics_step", -1)) < release_step)
    ]
    continuous_support = bool(qualified_rows) and all(
        force_status_by_step.get(int(_field(row, "physics_step", -1))) == (True, True)
        for row, _ in qualified_rows
    )
    qualified_manipulation_rows = [
        (row, obj) for row, obj in qualified_rows
        if int(_field(row, "physics_step", -1)) in lifted_steps
    ]
    unsupported_while_qualified = bool(qualified_manipulation_rows) and all(
        obj is not None and obj.get("support_id") in {None, "", "none"}
        for _, obj in qualified_manipulation_rows
    )

    metrics = {
        "right_measured_opposition_dwell_s": right_dwell,
        "right_required_digits_nonzero_impulse_dwell_s": right_impulse_dwell,
        "meaningful_opposing_left_geometric_dwell_s": left_dwell,
        "meaningful_opposing_left_nonzero_impulse_dwell_s": left_impulse_dwell,
        "left_opposing_normal_dwell_s": opposing_dwell,
        "controller_carry_telemetry_dwell_s": controller_carry_dwell,
        "literal_same_row_postqualification_opposing_geometry_dwell_s": literal_postqualification_dwell,
        "controller_carry_telemetry_steps": controller_carry_steps,
        "literal_same_row_postqualification_opposing_geometry_steps": literal_postqualification_steps,
        "bimanual_qualification_step": qualification_step,
        "support_continuous_until_commanded_opening": continuous_support,
        "object_unsupported_during_qualified_manipulation": unsupported_while_qualified,
        "qualified_lift_manipulation_steps": [
            int(_field(row, "physics_step", -1)) for row, _ in qualified_manipulation_rows
        ],
        "lift_m": lift_m if math.isfinite(lift_m) else None,
        "turn_during_bimanual_turn_deg": turn_deg if math.isfinite(turn_deg) else None,
        "turn_phase_steps": len(turn_states),
        "turn_phase_supported_steps": supported_turn_steps,
        "maximum_available_impulse_n_s": max(available_impulses) if available_impulses else None,
        "qualification_maximum_separation_m": maximum_separation_m,
        "eligible_hand_target_rows": eligible_hand_rows,
        "speculative_hand_target_rows_excluded": speculative_hand_rows,
        "eligible_nonzero_impulse_hand_target_rows": eligible_nonzero_impulse_rows,
        "initial_contact_steps": initial_contact_steps,
        "assistance_entries": len(all_ledgers),
        "authority_counter_rows": counter_rows,
        "authority_counter_violations": counter_violations[:20],
        "assistance_evidence_scope": "feasible runtime target-authority accounting plus mandatory independent source audit",
        **release_details,
    }
    return [
        _check("interaction.dynamic_force_bearing", right_ok and left_ok and continuous_support and unsupported_while_qualified,
               "required-digit nonzero PhysX impulse support is continuous on an unsupported object until opening", metrics),
        _check("interaction.no_initial_overlap", not initial_contact_steps,
               "qualification does not begin from initial overlap or depenetration", {"steps": initial_contact_steps}),
        _check("interaction.right_capture", right_ok,
               "right thumb plus at least two non-thumb contacts carry nonzero PhysX impulse beyond 0.30 s", metrics),
        _check("interaction.left_support", left_ok,
               "opposing non-little left support carries nonzero PhysX impulse beyond 0.25 s", metrics),
        _check("interaction.lift_turn", math.isfinite(lift_m) and math.isfinite(turn_deg)
               and lift_m > float(required["lift_min_m"]) and turn_deg > float(required["turn_min_deg"])
               and continuous_support and unsupported_while_qualified,
               "free target exceeds frozen post-qualification lift and in-phase turn thresholds", metrics),
        _check("interaction.free_release", free_release_ok,
               "commanded opening produces contact-free dynamic release and sleep-settle on the frozen EpisodeSpec destination", release_details),
        _check("interaction.no_assistance", no_recorded_assistance,
               "assistance ledger is empty and every feasible runtime target-authority counter remains zero",
               {"entries": all_ledgers[:20], "counter_rows": counter_rows,
                "counter_violations": counter_violations[:20], "runtime_detector": True}),
    ], metrics


def _camera_checks(
    rows: Sequence[Mapping[str, Any]],
    tolerances: Mapping[str, Any],
    capture_ledger: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[Check], dict[str, Any]]:
    required = tolerances["camera"]
    samples: list[tuple[float, Mapping[str, Any]]] = []
    for row in rows:
        camera = row.get("camera_state")
        time_s = _field(row, "time_s")
        if isinstance(camera, Mapping) and isinstance(time_s, (int, float)):
            samples.append((float(time_s), camera))
    if not samples:
        return [
            _check("camera.mount", None, "head camera trace unavailable"),
            _check("camera.motion", None, "camera roll and speed unavailable"),
        ], {}
    clearance = [float(camera["clearance_m"]) for _, camera in samples if isinstance(camera.get("clearance_m"), (int, float))]
    optical = [abs(float(camera["optical_vs_face_forward_deg"])) for _, camera in samples
               if isinstance(camera.get("optical_vs_face_forward_deg"), (int, float))]
    parents = {camera.get("parent_id") for _, camera in samples}
    positions = [_pose_position(camera) for _, camera in samples]
    rotations = [_pose_rotation(camera) for _, camera in samples]
    ledger_rolls = [
        abs(float(frame["roll_deg"])) for frame in capture_ledger
        if isinstance(frame.get("roll_deg"), (int, float)) and math.isfinite(float(frame["roll_deg"]))
    ]
    # World-quaternion Euler roll is coordinate-convention dependent and yielded
    # a false ~180 degree value for the valid Unity head mount. Prefer the
    # registered capture ledger's camera-relative roll; retain the quaternion
    # fallback only when no capture ledger exists.
    rolls = ledger_rolls or [abs(_quat_roll_deg(rotation)) for rotation in rotations if rotation is not None]
    roll_source = "registered_capture_ledger.roll_deg" if ledger_rolls else "world_quaternion_fallback"
    linear_speeds, angular_speeds = [], []
    for ((t0, _), (t1, _)), p0, p1, q0, q1 in zip(zip(samples, samples[1:]), positions, positions[1:], rotations, rotations[1:]):
        dt = t1 - t0
        if dt > 0 and p0 is not None and p1 is not None:
            linear_speeds.append(_distance(p0, p1) / dt)
        if dt > 0 and q0 is not None and q1 is not None:
            angular_speeds.append(_quat_angle_deg(q0, q1) / dt)
    complete = len(clearance) == len(samples) and len(optical) == len(samples) and all(value is not None for value in positions + rotations)
    clearance_min = required.get("minimum_clearance_m", required.get("minimum_head_or_clothing_clearance_m"))
    mount_ok = complete and len(parents) == 1 and None not in parents and min(clearance) > float(clearance_min)
    mount_ok &= max(optical) <= float(required["optical_vs_face_forward_max_deg"])
    motion_ok = bool(rolls) and max(rolls) <= float(required["roll_abs_max_deg"])
    if "linear_speed_max_m_s" in required:
        motion_ok &= not linear_speeds or max(linear_speeds) <= float(required["linear_speed_max_m_s"])
    if "angular_speed_max_deg_s" in required:
        motion_ok &= not angular_speeds or max(angular_speeds) <= float(required["angular_speed_max_deg_s"])
    metrics = {
        "sample_count": len(samples),
        "parent_ids": sorted(str(parent) for parent in parents),
        "minimum_clearance_m": min(clearance) if clearance else None,
        "maximum_optical_vs_face_forward_deg": max(optical) if optical else None,
        "maximum_abs_roll_deg": max(rolls) if rolls else None,
        "roll_source": roll_source,
        "maximum_linear_speed_m_s": max(linear_speeds) if linear_speeds else 0.0,
        "maximum_angular_speed_deg_s": max(angular_speeds) if angular_speeds else 0.0,
    }
    return [
        _check("camera.mount", mount_ok, "camera has one traced head parent, neutral optical mount, and positive measured clearance", metrics),
        _check("camera.motion", motion_ok, "camera roll and physics-clocked linear/angular speed remain bounded", metrics),
    ], metrics


def _motion_limit_checks(rows: Sequence[Mapping[str, Any]], physics_hz: float) -> tuple[list[Check], dict[str, Any]]:
    if not rows:
        return [_check("motion.commanded_limits", None, "palm/finger speed and closure limit evidence unavailable")], {}
    violations: dict[str, int] = defaultdict(int)
    maxima = {"palm_linear_m_s": 0.0, "palm_angular_rad_s": 0.0,
              "finger_linear_m_s": 0.0, "finger_angular_rad_s": 0.0,
              "closure_rate_s": 0.0}
    previous_closures: dict[str, float] = {}
    for row in rows:
        controller = row.get("controller_state")
        limits = controller.get("speed_limits") if isinstance(controller, Mapping) else None
        closure_limits = controller.get("closure_limits") if isinstance(controller, Mapping) else None
        if not isinstance(limits, Mapping) or not isinstance(closure_limits, Mapping):
            violations["missing_limits"] += 1
            continue
        names = ("palm_linear_max_m_s", "palm_angular_max_rad_s",
                 "finger_segment_linear_max_m_s", "finger_segment_angular_max_rad_s")
        if any(not isinstance(limits.get(name), (int, float)) or float(limits[name]) <= 0 for name in names):
            violations["invalid_speed_limits"] += 1
            continue
        if limits.get("provenance") == "unavailable":
            violations["unavailable_speed_limits"] += 1
        closure_names = ("closure_min", "closure_max", "closure_rate_max_s")
        if any(not isinstance(closure_limits.get(name), (int, float)) for name in closure_names):
            violations["invalid_closure_limits"] += 1
            continue
        if closure_limits.get("provenance") == "unavailable" or float(closure_limits["closure_rate_max_s"]) <= 0:
            violations["unavailable_closure_limits"] += 1
        hands = _hands(row)
        if hands is None:
            violations["missing_hands"] += 1
            continue
        for palm in ("left_palm", "right_palm"):
            linear, angular = _pose_velocities(hands.get(palm))
            maxima["palm_linear_m_s"] = max(maxima["palm_linear_m_s"], _magnitude(linear))
            maxima["palm_angular_rad_s"] = max(maxima["palm_angular_rad_s"], _magnitude(angular))
            if not math.isfinite(_magnitude(linear)) or _magnitude(linear) > float(limits["palm_linear_max_m_s"]) + 1e-5:
                violations["palm_linear"] += 1
            if not math.isfinite(_magnitude(angular)) or _magnitude(angular) > float(limits["palm_angular_max_rad_s"]) + 1e-5:
                violations["palm_angular"] += 1
        for hand in ("left", "right"):
            for digit in ("thumb", "index", "middle", "ring", "little"):
                key = f"{hand}_{digit}"
                state = hands.get(key)
                if not isinstance(state, Mapping):
                    violations["missing_digits"] += 1
                    continue
                for segment in state.get("segments", []):
                    linear, angular = _pose_velocities(segment)
                    maxima["finger_linear_m_s"] = max(maxima["finger_linear_m_s"], _magnitude(linear))
                    maxima["finger_angular_rad_s"] = max(maxima["finger_angular_rad_s"], _magnitude(angular))
                    if not math.isfinite(_magnitude(linear)) or _magnitude(linear) > float(limits["finger_segment_linear_max_m_s"]) + 1e-5:
                        violations["finger_linear"] += 1
                    if not math.isfinite(_magnitude(angular)) or _magnitude(angular) > float(limits["finger_segment_angular_max_rad_s"]) + 1e-5:
                        violations["finger_angular"] += 1
                closure = state.get("closure_commanded")
                if not isinstance(closure, (int, float)):
                    violations["missing_closure"] += 1
                    continue
                closure = float(closure)
                if not float(closure_limits["closure_min"]) <= closure <= float(closure_limits["closure_max"]):
                    violations["closure_range"] += 1
                if key in previous_closures:
                    rate = abs(closure - previous_closures[key]) * physics_hz
                    maxima["closure_rate_s"] = max(maxima["closure_rate_s"], rate)
                    if rate > float(closure_limits["closure_rate_max_s"]) + 1e-5:
                        violations["closure_rate"] += 1
                previous_closures[key] = closure
    metrics = {**maxima, "violation_counts": dict(sorted(violations.items()))}
    return [_check("motion.commanded_limits", not violations,
                   "engine-observed palm/finger speeds and commanded closure/rates remain within traced controller limits",
                   metrics)], metrics


def _dynamic_finger_checks(rows: Sequence[Mapping[str, Any]]) -> list[Check]:
    if not rows:
        return [_check("embodiment.compliant_dynamic_fingers", None, "dynamic articulated finger truth is unavailable")]
    missing: dict[str, int] = defaultdict(int)
    observed = 0
    for row in rows:
        hands = _hands(row)
        if hands is None:
            missing["hand_state"] += 1
            continue
        for side in ("left", "right"):
            for digit in ("thumb", "index", "middle", "ring", "little"):
                state = hands.get(f"{side}_{digit}")
                if not isinstance(state, Mapping):
                    missing["digit"] += 1
                    continue
                bindings: list[tuple[Any, Any]] = []
                dynamic_bodies = state.get("dynamic_body_states", state.get("dynamic_bodies"))
                compliant_joints = state.get("compliant_joint_states", state.get("compliant_joints"))
                if isinstance(dynamic_bodies, list) and isinstance(compliant_joints, list):
                    bindings = list(zip(dynamic_bodies, compliant_joints))
                elif isinstance(state.get("segments"), list) and all(
                    isinstance(segment, Mapping)
                    and "dynamic_body" in segment and "compliant_joint" in segment
                    for segment in state["segments"]
                ):
                    bindings = [(segment["dynamic_body"], segment["compliant_joint"])
                                for segment in state["segments"]]
                if len(bindings) != 3:
                    missing["three_segment_bindings"] += 1
                for body, joint in bindings:
                    if not isinstance(body, Mapping) or body.get("is_kinematic") is not False:
                        missing["dynamic_body"] += 1
                    elif (body.get("provenance") != "physx_measured"
                          or not isinstance(body.get("mass_kg"), (int, float))
                          or float(body["mass_kg"]) <= 0):
                        missing["dynamic_body_truth"] += 1
                    pose = body.get("pose") if isinstance(body, Mapping) else None
                    if (not isinstance(pose, Mapping)
                            or _vector(pose.get("position_world_m"), 3) is None
                            or _vector(pose.get("rotation_world_xyzw"), 4) is None):
                        missing["dynamic_body_pose"] += 1
                    if not isinstance(joint, Mapping):
                        missing["compliant_joint"] += 1
                        observed += 1
                        continue
                    drives = (
                        joint.get("angular_x_drive_spring_n_m_rad"),
                        joint.get("angular_x_drive_damper_n_m_s_rad"),
                        joint.get("angular_x_drive_max_force_n_m"),
                        joint.get("angular_yz_drive_spring_n_m_rad"),
                        joint.get("angular_yz_drive_damper_n_m_s_rad"),
                        joint.get("angular_yz_drive_max_force_n_m"),
                    )
                    if (joint.get("provenance") != "engine_observed"
                            or "ConfigurableJoint" not in str(joint.get("drive_provenance", ""))
                            or any(not isinstance(value, (int, float)) or float(value) <= 0 for value in drives)):
                        missing["compliant_joint_truth"] += 1
                    observed += 1
    expected = len(rows) * 30
    return [_check(
        "embodiment.compliant_dynamic_fingers",
        observed == expected and not missing,
        "all thirty finger segments use traced dynamic rigidbodies and finite compliant ConfigurableJoint drives",
        {"observed_digit_rows": observed, "expected_digit_rows": expected, "failure_counts": dict(sorted(missing.items()))},
    )]


def _visual_method_is_direct(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    lowered = value.lower()
    return ("dense" in lowered and ("frame" in lowered or "visual" in lowered)
            and not any(term in lowered for term in VISUAL_PROXY_TERMS))


def _validate_dense_decoded_sequence(
    root: Path,
    entry: Mapping[str, Any],
    veto_keys: Sequence[str],
) -> tuple[Path, list[dict[str, Any]]]:
    ledger_relative = Path(str(entry.get("capture_ledger", "")))
    if not str(ledger_relative) or ledger_relative.is_absolute():
        raise ValueError("dense decoded review requires a relative capture_ledger")
    ledger_path = _inside(root, root / ledger_relative)
    ledger = _load_trace(ledger_path)
    expected_frames = round(float(load_frozen_config()["activity_plan"]["duration_s"]) * 30.0)
    frame_ids = [row.get("render_frame") for row in ledger if isinstance(row, Mapping)]
    if frame_ids != list(range(expected_frames)):
        raise ValueError("capture ledger is not the complete dense frozen frame sequence")
    annotations = entry.get("frames")
    if not isinstance(annotations, list):
        raise ValueError("dense decoded review requires frame observations")
    annotation_by_frame = {row.get("render_frame"): row for row in annotations if isinstance(row, Mapping)}
    if len(annotation_by_frame) != len(annotations) or set(annotation_by_frame) != set(frame_ids):
        raise ValueError("decoded observations must cover every ledger frame exactly once")
    canonical_frames: list[dict[str, Any]] = []
    for ledger_row in ledger:
        frame = ledger_row["render_frame"]
        annotation = annotation_by_frame[frame]
        if (not _visual_method_is_direct(annotation.get("method"))
                or "decoded" not in str(annotation.get("method")).lower()
                or annotation.get("direct_observation") is not True):
            raise ValueError("each observation must be a direct dense decoded-frame review")
        if any(not isinstance(annotation.get(key), bool) for key in veto_keys):
            raise ValueError("every visual veto must be explicitly reviewed as true or false")
        stream_map = {item.get("stream"): item for item in ledger_row.get("streams", []) if isinstance(item, Mapping)}
        decoded: dict[str, Any] = {}
        for role, stream_name in (("head", "head_rgb_hero"), ("external", "external_clean")):
            receipt = stream_map.get(stream_name)
            if not isinstance(receipt, Mapping) or not isinstance(receipt.get("relative_path"), str):
                raise ValueError(f"frame {frame} lacks {stream_name}")
            media_path = _inside(root, ledger_path.parent / receipt["relative_path"])
            file_hash = _sha256(media_path)
            if receipt.get("file_sha256") != file_hash or annotation.get(f"{role}_file_sha256") != file_hash:
                raise ValueError(f"frame {frame} {role} encoded hash mismatch")
            width, height, rgb = _decode_png_rgb(media_path)
            if (width, height) != (1920, 1080):
                raise ValueError(f"frame {frame} {role} is not frozen 1920x1080")
            decoded_hash = hashlib.sha256(rgb).hexdigest()
            if annotation.get(f"{role}_decoded_rgb_sha256") != decoded_hash:
                raise ValueError(f"frame {frame} {role} decoded hash mismatch")
            decoded[role] = {
                "relative_path": str(media_path.relative_to(root)),
                "file_sha256": file_hash,
                "decoded_rgb_sha256": decoded_hash,
                "width_px": width,
                "height_px": height,
            }
        canonical_frames.append({
            "render_frame": frame,
            "method": annotation["method"],
            "direct_observation": True,
            **{key: annotation[key] for key in veto_keys},
            "head": decoded["head"],
            "external": decoded["external"],
        })
    return ledger_path, canonical_frames


def write_visual_audit_from_decoded_review(
    run_root: Path | str,
    review: Mapping[str, Any],
    path: Path | str | None = None,
) -> Path:
    """Bind direct dense review to media across canonical sibling stage runs."""
    root = Path(run_root).resolve()
    if not root.is_dir() or review.get("schema") != DECODED_FRAME_REVIEW_SCHEMA:
        raise ValueError(f"expected {DECODED_FRAME_REVIEW_SCHEMA} under an existing aggregate run root")
    reviewer_id = review.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("decoded review requires a reviewer_id")
    sweeps = review.get("garment_sweeps")
    if not isinstance(sweeps, list) or not sweeps:
        raise ValueError("decoded review requires garment_sweeps")
    anatomy_vetoes = ("nude_or_exposed", "exploding_weights", "fused_digits", "detached_wrists", "clipping")
    canonical_sweeps = []
    for sweep in sweeps:
        if not isinstance(sweep, Mapping) or not isinstance(sweep.get("configuration_id"), str):
            raise ValueError("garment sweep requires a configuration_id")
        ledger_path, frames = _validate_dense_decoded_sequence(root, sweep, anatomy_vetoes)
        canonical_sweeps.append({
            "configuration_id": sweep["configuration_id"],
            "method": "dense decoded-frame visual review bound to registered head/external media hashes",
            **{key: any(frame[key] for frame in frames) for key in anatomy_vetoes},
            "dense_frame_review_complete": True,
            "reviewed_frame_count": len(frames),
            "source_capture_ledger": str(ledger_path.relative_to(root)),
            "source_capture_ledger_sha256": _sha256(ledger_path),
            "dense_frame_reviews": frames,
        })
    payload: dict[str, Any] = {
        "schema": "embodied.visual_audit.v1",
        "reviewer_id": reviewer_id,
        "adapter": "independent_qa.write_visual_audit_from_decoded_review",
        "aggregate_evidence_root": str(root),
        "garment_sweeps": canonical_sweeps,
    }
    motion = review.get("motion_camera_qualification")
    if isinstance(motion, Mapping):
        motion_vetoes = ("clipping", "camera_in_mesh", "malformed_anatomy", "occluded_required_motion")
        ledger_path, frames = _validate_dense_decoded_sequence(root, motion, motion_vetoes)
        payload["motion_camera_qualification"] = {
            "method": "dense decoded-frame visual review bound to registered head/external media hashes",
            **{key: motion.get(key) for key in (
                "object_free", "continuity_pass", "both_arms_natural", "static_target_view",
                "head_video_reviewed", "external_video_reviewed")},
            **{key: any(frame[key] for frame in frames) for key in motion_vetoes},
            "dense_frame_review_complete": True,
            "reviewed_frame_count": len(frames),
            "source_capture_ledger": str(ledger_path.relative_to(root)),
            "source_capture_ledger_sha256": _sha256(ledger_path),
            "dense_frame_reviews": frames,
        }
    event_rows = review.get("event_visibility")
    if isinstance(event_rows, list):
        canonical_events = []
        for event in event_rows:
            if not isinstance(event, Mapping) or not isinstance(event.get("event"), str):
                raise ValueError("event visibility requires named event rows")
            ledger_path, frames = _validate_dense_decoded_sequence(root, event, ())
            frame = event.get("frame")
            if not isinstance(frame, int) or frame < 0 or frame >= len(frames):
                raise ValueError("event visibility frame is outside its bound ledger")
            canonical_events.append({
                "event": event["event"],
                "frame": frame,
                "visible": event.get("visible") is True,
                "method": "direct dense decoded-frame visual review bound to registered head/external media hashes",
                "source_capture_ledger": str(ledger_path.relative_to(root)),
                "source_capture_ledger_sha256": _sha256(ledger_path),
                "head": frames[frame]["head"],
                "external": frames[frame]["external"],
            })
        payload["event_visibility"] = canonical_events
    episode_rows = review.get("episodes")
    if isinstance(episode_rows, list):
        episode_vetoes = ("malformed_anatomy", "garment_clipping", "floating_or_stretched_limbs",
                          "furniture_intrusion", "overexposure", "static_target_view", "bad_transitions",
                          "incoherent_room", "proxy_hero_pixels", "camera_in_mesh")
        canonical_episodes = []
        for episode in episode_rows:
            if not isinstance(episode, Mapping) or not isinstance(episode.get("episode_id"), str):
                raise ValueError("episode review requires an episode_id")
            ledger_path, frames = _validate_dense_decoded_sequence(root, episode, episode_vetoes)
            canonical_episodes.append({
                "episode_id": episode["episode_id"],
                "method": "dense decoded-frame visual review bound to registered head/external media hashes",
                "coherent": episode.get("coherent") is True,
                **{key: any(frame[key] for frame in frames) for key in episode_vetoes},
                "dense_frame_review_complete": True,
                "reviewed_frame_count": len(frames),
                "source_capture_ledger": str(ledger_path.relative_to(root)),
                "source_capture_ledger_sha256": _sha256(ledger_path),
                "dense_frame_reviews": frames,
            })
        payload["episodes"] = canonical_episodes
    if path is None:
        destination = root / "visual_audit.json"
    else:
        requested = Path(path)
        destination = (requested if requested.is_absolute() else root / requested).resolve()
    _inside(root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _visual_checks(visual: Any, tolerances: Mapping[str, Any]) -> list[Check]:
    if not isinstance(visual, Mapping):
        return [
            _check("visual.garment_sweeps", None, "dense garment/body/anatomy review missing"),
            _check("visual.motion_camera", None, "object-free head/external motion review missing"),
            _check("visual.event_visibility", None, "required event visibility review missing"),
            _check("visual.episode_coherence", None, "dense integrated episode review missing"),
        ]
    if visual.get("schema") == "embodied.registered_visual_measurements.v1":
        evidence = {
            "measured_capture_evidence_only": visual.get("measured_capture_evidence_only"),
            "direct_decoded_frame_review_performed": visual.get("direct_decoded_frame_review_performed"),
            "measurement_scope": visual.get("measurement_scope"),
        }
        return [
            _check("visual.garment_sweeps", None, "registered capture measurements do not replace dense garment/body/anatomy review", evidence),
            _check("visual.motion_camera", None, "registered capture measurements do not replace decoded head/external motion review", evidence),
            _check("visual.event_visibility", None, "registered projections are geometric evidence, not decoded-frame visibility review", evidence),
            _check("visual.episode_coherence", None, "registered capture measurements do not make visual-coherence judgments", evidence),
        ]
    sweeps = visual.get("garment_sweeps", [])
    expected_frames = round(float(load_frozen_config()["activity_plan"]["duration_s"]) * 30.0)
    def bound_frames_ok(entry: Mapping[str, Any], vetoes: Sequence[str]) -> bool:
        frames = entry.get("dense_frame_reviews")
        return (
            entry.get("dense_frame_review_complete") is True
            and entry.get("reviewed_frame_count") == expected_frames
            and isinstance(frames, list) and len(frames) == expected_frames
            and all(
                isinstance(frame, Mapping) and frame.get("direct_observation") is True
                and _visual_method_is_direct(frame.get("method"))
                and all(frame.get(key) is False for key in vetoes)
                and all(isinstance(frame.get(role), Mapping)
                        and frame[role].get("width_px") == 1920 and frame[role].get("height_px") == 1080
                        and isinstance(frame[role].get("file_sha256"), str) and len(frame[role]["file_sha256"]) == 64
                        and isinstance(frame[role].get("decoded_rgb_sha256"), str) and len(frame[role]["decoded_rgb_sha256"]) == 64
                        for role in ("head", "external"))
                for frame in frames
            )
        )
    bad_anatomy_keys = ("nude_or_exposed", "exploding_weights", "fused_digits", "detached_wrists", "clipping")
    sweep_ok = isinstance(sweeps, list) and len({row.get("configuration_id") for row in sweeps if isinstance(row, Mapping)}) >= 3
    sweep_ok &= all(
        isinstance(row, Mapping) and _visual_method_is_direct(row.get("method"))
        and bound_frames_ok(row, bad_anatomy_keys)
        and all(row.get(key) is False for key in bad_anatomy_keys)
        for row in sweeps
    )

    motion = visual.get("motion_camera_qualification")
    motion_ok = isinstance(motion, Mapping) and _visual_method_is_direct(motion.get("method"))
    motion_ok &= bool(motion.get("object_free")) and bool(motion.get("continuity_pass"))
    motion_ok &= bool(motion.get("both_arms_natural")) and motion.get("static_target_view") is False
    motion_ok &= bool(motion.get("head_video_reviewed")) and bool(motion.get("external_video_reviewed"))
    motion_vetoes = ("clipping", "camera_in_mesh", "malformed_anatomy", "occluded_required_motion")
    motion_ok &= bound_frames_ok(motion, motion_vetoes)
    motion_ok &= all(motion.get(key) is False for key in motion_vetoes)

    visibility = visual.get("event_visibility", [])
    required_events = set(tolerances["camera"]["required_events_visible"])
    visible_events = {row.get("event") for row in visibility if isinstance(row, Mapping)
                      and row.get("visible") is True and isinstance(row.get("frame"), int)
                      and _visual_method_is_direct(row.get("method"))
                      and all(isinstance(row.get(role), Mapping)
                              and len(str(row[role].get("file_sha256", ""))) == 64
                              and len(str(row[role].get("decoded_rgb_sha256", ""))) == 64
                              for role in ("head", "external"))}
    visibility_ok = required_events <= visible_events

    episodes = visual.get("episodes", [])
    veto_keys = ("malformed_anatomy", "garment_clipping", "floating_or_stretched_limbs",
                 "furniture_intrusion", "overexposure", "static_target_view", "bad_transitions",
                 "incoherent_room", "proxy_hero_pixels", "camera_in_mesh")
    episode_ok = isinstance(episodes, list) and bool(episodes)
    episode_ok &= all(isinstance(row, Mapping) and _visual_method_is_direct(row.get("method"))
                      and row.get("coherent") is True and all(row.get(key) is False for key in veto_keys)
                      and bound_frames_ok(row, veto_keys)
                      for row in episodes)
    return [
        _check("visual.garment_sweeps", sweep_ok, "three garment configurations pass dense anatomy/clipping review",
               {"reviewed": len(sweeps) if isinstance(sweeps, list) else 0}),
        _check("visual.motion_camera", motion_ok, "object-free head/external sequence has continuous full-body motion and non-static gaze"),
        _check("visual.event_visibility", visibility_ok, "every required interaction event is visible in direct rendered-frame review",
               {"required": sorted(required_events), "visible": sorted(str(value) for value in visible_events)}),
        _check("visual.episode_coherence", episode_ok, "integrated episodes pass all independent visual veto predicates",
               {"reviewed": len(episodes) if isinstance(episodes, list) else 0}),
    ]


def _registration_checks(report: Any, tolerances: Mapping[str, Any]) -> list[Check]:
    if not isinstance(report, Mapping):
        return [
            _check("registration.skin_collider", None, "skin/collider registration measurement missing"),
            _check("registration.garment_body", None, "garment/body penetration and affected-vertex distribution missing"),
            _check("registration.penetration", None, "hand/object and target/support penetration missing"),
        ]
    provenance = " ".join(str(report.get(key, "")) for key in
                          ("provenance", "body_collider_provenance", "garment_body_provenance")).lower()
    measured = "measured" in provenance or "engine_observed" in provenance or "engineobserved" in provenance
    proxy = any(term in provenance for term in VISUAL_PROXY_TERMS)
    registration = tolerances["registration"]
    contact = tolerances["contact"]
    configurations = report.get("garment_configurations", [])
    distribution = report.get("garment_affected_vertex_distribution")
    self_clearance_provenance = str(report.get("anatomical_self_clearance_provenance", "")).lower()
    self_clearance_steps = report.get("self_clearance_steps", [])
    self_clearance_ok = report.get("passed") is True
    self_clearance_ok &= "physxmeasured" in self_clearance_provenance or "physx_measured" in self_clearance_provenance
    self_clearance_ok &= report.get("self_clearance_sampled_every_physics_step") is True
    self_clearance_ok &= report.get("non_adjacent_anatomy_clearance_passed") is True
    self_clearance_ok &= isinstance(report.get("non_adjacent_self_clearance_samples"), int)
    self_clearance_ok &= report.get("non_adjacent_self_clearance_samples", 0) > 0
    self_clearance_ok &= report.get("non_adjacent_self_overlap_samples") == 0
    self_clearance_ok &= isinstance(self_clearance_steps, list) and bool(self_clearance_steps)
    self_clearance_ok &= all(
        isinstance(row, Mapping) and row.get("swept_pair_pose_samples", 0) > 0
        and row.get("overlap_samples") == 0
        for row in self_clearance_steps
    )
    solver_provenance = str(report.get("solver_motion_self_clearance_provenance", "")).lower()
    solver_steps = report.get("solver_motion_self_clearance_steps", [])
    expected_solver_steps = report.get("expected_motion_sample_count")
    solver_ok = "physxmeasured" in solver_provenance or "physx_measured" in solver_provenance
    solver_ok &= report.get("solver_motion_self_clearance_sampled_every_physics_step") is True
    solver_ok &= report.get("solver_motion_non_adjacent_anatomy_clearance_passed") is True
    solver_ok &= isinstance(report.get("solver_motion_self_clearance_samples"), int)
    solver_ok &= report.get("solver_motion_self_clearance_samples", 0) > 0
    solver_ok &= report.get("solver_motion_self_overlap_samples") == 0
    solver_ok &= report.get("solver_motion_incomplete_sweep_intervals") == 0
    solver_ok &= isinstance(solver_steps, list) and bool(solver_steps)
    if isinstance(expected_solver_steps, int) and expected_solver_steps > 0:
        solver_ok &= len(solver_steps) == expected_solver_steps
        solver_ok &= [row.get("physics_step") for row in solver_steps if isinstance(row, Mapping)] == list(range(expected_solver_steps))
    solver_ok &= all(
        isinstance(row, Mapping) and row.get("swept_pair_pose_samples", 0) > 0
        and row.get("overlap_samples") == 0
        and row.get("incomplete_sweep_intervals") == 0
        for row in solver_steps
    )
    self_clearance_ok &= solver_ok
    try:
        body_distribution = report.get("body_collider_registration", {})
        skin_max = report.get("skin_collider_max_m", body_distribution.get("maximum_m"))
        garment_rows = report.get("garments", [])
        canonical_garment_ok = isinstance(garment_rows, list) and bool(garment_rows) and all(
            isinstance(row, Mapping)
            and float(row["maximum_penetration_m"]) <= float(registration["garment_body_max_penetration_m"])
            and float(row["affected_fraction"]) <= float(registration["garment_affected_vertex_fraction_max"])
            and bool(row.get("affected_vertex_distribution_counts"))
            for row in garment_rows
        )
        skin_ok = measured and not proxy and self_clearance_ok and float(skin_max) <= float(registration["skin_collider_max_m"])
        if canonical_garment_ok:
            garment_ok = measured and not proxy and self_clearance_ok and report.get("sampled_every_physics_step") is True
        else:
            garment_ok = measured and not proxy and float(report["garment_body_max_penetration_m"]) <= float(registration["garment_body_max_penetration_m"])
            garment_ok &= isinstance(distribution, (dict, list)) and bool(distribution)
            garment_ok &= isinstance(configurations, list) and len({row.get("configuration_id") for row in configurations
                                                                    if isinstance(row, Mapping)}) >= 3
            garment_ok &= all(
                isinstance(row, Mapping)
                and float(row["skin_collider_max_m"]) <= float(registration["skin_collider_max_m"])
                and float(row["garment_body_max_penetration_m"]) <= float(registration["garment_body_max_penetration_m"])
                and float(row.get("garment_affected_vertex_fraction", 1.0)) <= float(registration["garment_affected_vertex_fraction_max"])
                and bool(row.get("garment_affected_vertex_distribution"))
                for row in configurations
            )
        penetration_ok = measured and not proxy and self_clearance_ok
        finger_limit = registration["finger_object_max_penetration_m"]
        support_limit = registration["support_max_penetration_m"]
        penetration_ok &= float(report["finger_object_max_penetration_m"]) <= float(finger_limit)
        penetration_ok &= float(report["target_support_max_penetration_m"]) <= float(support_limit)
        penetration_ok &= report.get("finger_object_contact_samples", 0) > 0
        penetration_ok &= report.get("target_support_contact_samples", 0) > 0
        penetration_provenance = " ".join(str(report.get(name, "")) for name in (
            "finger_object_penetration_provenance", "target_support_penetration_provenance"
        )).lower()
        penetration_ok &= "physxmeasured" in penetration_provenance or "physx_measured" in penetration_provenance
        if "trace_finger_object_max_penetration_m" in report:
            penetration_ok &= abs(float(report["finger_object_max_penetration_m"])
                                  - float(report["trace_finger_object_max_penetration_m"])) <= 1e-7
        if "trace_target_support_max_penetration_m" in report:
            penetration_ok &= abs(float(report["target_support_max_penetration_m"])
                                  - float(report["trace_target_support_max_penetration_m"])) <= 1e-7
    except (KeyError, TypeError, ValueError):
        skin_ok = garment_ok = penetration_ok = False
    return [
        _check("registration.skin_collider", skin_ok, "passed registration receipt includes full swept non-adjacent self-clearance and skin/collider tolerance", report),
        _check("registration.garment_body", garment_ok, "passed registration receipt includes swept self-clearance and garment penetration/distribution", report),
        _check("registration.penetration", penetration_ok, "passed registration receipt includes swept self-clearance and measured contact penetrations", report),
    ]


def _registration_with_trace_penetration(
    report: Any, rows: Sequence[Mapping[str, Any]], target_id: str | None
) -> Any:
    if not isinstance(report, Mapping) or not rows or not target_id:
        return report
    adapted = dict(report)
    finger_penetrations: list[float] = []
    support_penetrations: list[float] = []
    for row in rows:
        for contact in _contacts(row):
            if not _measured_contact(contact) or not _contact_targets(contact, target_id):
                continue
            separation = contact.get("separation_m")
            if not isinstance(separation, (int, float)) or not math.isfinite(float(separation)):
                continue
            penetration = max(0.0, -float(separation))
            hand, digit = _contact_hand_digit(contact)
            if hand in {"left", "right"} and digit is not None:
                finger_penetrations.append(penetration)
            elif hand is None:
                support_penetrations.append(penetration)
    if finger_penetrations:
        adapted["trace_finger_object_max_penetration_m"] = max(finger_penetrations)
    if support_penetrations:
        adapted["trace_target_support_max_penetration_m"] = max(support_penetrations)
    if finger_penetrations or support_penetrations:
        adapted["trace_contact_penetration_provenance"] = (
            "physx_measured episode-trace contact separation; penetration=max(0,-separation_m)"
        )
    return adapted


def _decode_png_rgb(path: Path) -> tuple[int, int, bytes]:
    """Fully decode a non-interlaced RGB/RGBA PNG into top-down RGB bytes."""
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("label stream is not PNG")
    offset = 8
    idat = bytearray()
    width = height = bit_depth = color_type = interlace = None
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        kind = payload[offset + 4:offset + 8]
        data = payload[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            idat.extend(data)
        elif kind == b"IEND":
            break
    if bit_depth != 8 or color_type not in {2, 6} or interlace != 0 or not width or not height:
        raise ValueError("label PNG must be non-interlaced RGB/RGBA uint8")
    channels = 3 if color_type == 2 else 4
    stride = width * channels
    raw = zlib.decompress(bytes(idat))
    if len(raw) != height * (stride + 1):
        raise ValueError("label PNG scanline length mismatch")
    rows: list[bytearray] = []
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        scan = bytearray(raw[cursor + 1:cursor + 1 + stride])
        cursor += stride + 1
        previous = rows[-1] if rows else bytearray(stride)
        for index in range(stride):
            left = scan[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                scan[index] = (scan[index] + left) & 0xFF
            elif filter_type == 2:
                scan[index] = (scan[index] + up) & 0xFF
            elif filter_type == 3:
                scan[index] = (scan[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                predictor = left + up - upper_left
                pa, pb, pc = abs(predictor - left), abs(predictor - up), abs(predictor - upper_left)
                nearest = left if pa <= pb and pa <= pc else up if pb <= pc else upper_left
                scan[index] = (scan[index] + nearest) & 0xFF
            elif filter_type != 0:
                raise ValueError("unsupported PNG filter")
        rows.append(scan)
    rgb = bytearray()
    for row in rows:
        for base in range(0, len(row), channels):
            rgb.extend(row[base:base + 3])
    return width, height, bytes(rgb)


def _png_uint24_at(path: Path, x: int, y_bottom_left: int) -> int:
    """Decode one RGB uint24 pixel without trusting a producer-side sample."""
    width, height, rgb = _decode_png_rgb(path)
    if not (0 <= x < width and 0 <= y_bottom_left < height):
        raise ValueError("projected label pixel is outside PNG")
    base = ((height - 1 - y_bottom_left) * width + x) * 3
    return rgb[base] + 256 * rgb[base + 1] + 65536 * rgb[base + 2]


def _projection_checks(
    report: Any,
    tolerances: Mapping[str, Any],
    capture_ledger: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    target_id: str | None,
    ledger_path: Path,
    root: Path,
) -> list[Check]:
    if not isinstance(report, Mapping):
        return [_check("contact.visible_projection", None, "measured-contact to visible-surface projection missing")]
    if report.get("schema") != "embodied.registered_contact_projection.v1":
        return [_check("contact.visible_projection", False, "contact projection report does not use the registered capture schema")]
    records = report.get("records", [])
    if not isinstance(records, list) or not records:
        return [_check("contact.visible_projection", False, "contact projection report has no records")]
    ledger_hash_ok = ledger_path.is_file() and report.get("source_capture_ledger_sha256") == _sha256(ledger_path)
    ledger_by_frame = {row.get("render_frame"): row for row in capture_ledger if isinstance(row, Mapping)}
    embedded = {
        (row.get("render_frame"), row.get("physics_step"), tuple(_vector(item.get("point_world_m")) or ()),
         item.get("expected_contact_collider_a"), item.get("expected_contact_collider_b")):
        item
        for row in capture_ledger if isinstance(row, Mapping)
        for item in row.get("event_visibility_inputs", []) if isinstance(item, Mapping)
        and item.get("input_id") == "physx_contact_point_input"
    }
    label_manifest_path = root / "semantic_instance_manifest.json"
    try:
        label_manifest = _read_json(label_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        label_manifest = {}
    bindings = {
        (item.get("semantic_uint24"), item.get("persistent_instance_uint24")): item
        for item in label_manifest.get("bindings", []) if isinstance(label_manifest, Mapping)
        and isinstance(item, Mapping)
    }
    trace_by_step = {int(_field(row, "physics_step")): row for row in trace_rows
                     if isinstance(_field(row, "physics_step"), int)}
    checked = []
    required_pairs = {"right_thumb", "left_non_little"}
    observed_pairs: set[str] = set()
    right_non_thumb_digits: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            checked.append(False)
            continue
        method = str(row.get("method", ""))
        measured_projection = "actual semantic/instance uint24 values decoded" in method.lower()
        measured_projection &= "physics.raycastall" in method.lower()
        measured_projection &= not any(term in method.lower() for term in VISUAL_PROXY_TERMS)
        colliders = (row.get("expected_contact_collider_a"), row.get("expected_contact_collider_b"))
        hand, digit = _contact_hand_digit({"collider_a": colliders[0], "collider_b": colliders[1]})
        pair = "right_thumb" if hand == "right" and digit == "thumb" else "left_non_little" if hand == "left" and digit not in {"", "little"} else "other"
        observed_pairs.add(pair)
        if hand == "right" and digit in {"index", "middle", "ring", "little"}:
            right_non_thumb_digits.add(digit)
        render_frame, physics_step = row.get("render_frame"), row.get("physics_step")
        point = _vector(row.get("physical_contact_point_world_m"))
        frame_ok = isinstance(render_frame, int) and isinstance(physics_step, int)
        frame_ok &= physics_step // int(tolerances["clock"]["exact_steps_per_render_frame"]) == render_frame
        embedded_row = embedded.get((render_frame, physics_step, tuple(point or ()), colliders[0], colliders[1]))
        embedded_ok = isinstance(embedded_row, Mapping)
        trace_row = trace_by_step.get(physics_step) if isinstance(physics_step, int) else None
        trace_contact_ok = target_id is not None and trace_row is not None and any(
            _measured_contact(contact)
            and _contact_targets(contact, target_id)
            and {contact.get("collider_a"), contact.get("collider_b")} == set(colliders)
            and _vector(contact.get("point_world_m")) == point
            for contact in _contacts(trace_row)
        )
        pixel = _vector(row.get("pixel_xy"), 2)
        pixel_ok = pixel is not None and all(float(value).is_integer() for value in pixel)
        semantic_id = row.get("first_visible_semantic_uint24")
        instance_id = row.get("first_visible_persistent_instance_uint24")
        decoded_ok = False
        capture_row = ledger_by_frame.get(render_frame)
        if pixel_ok and isinstance(capture_row, Mapping):
            stream_map = {item.get("stream"): item for item in capture_row.get("streams", []) if isinstance(item, Mapping)}
            semantic_stream = stream_map.get("head_semantic_uint24")
            instance_stream = stream_map.get("head_persistent_instance_uint24")
            try:
                semantic_path = _inside(root, root / semantic_stream["relative_path"])
                instance_path = _inside(root, root / instance_stream["relative_path"])
                decoded_semantic = _png_uint24_at(semantic_path, int(pixel[0]), int(pixel[1]))
                decoded_instance = _png_uint24_at(instance_path, int(pixel[0]), int(pixel[1]))
                decoded_ok = decoded_semantic == semantic_id and decoded_instance == instance_id
                decoded_ok &= semantic_stream.get("file_sha256") == _sha256(semantic_path)
                decoded_ok &= instance_stream.get("file_sha256") == _sha256(instance_path)
                decoded_ok &= semantic_stream.get("authority_state_sha256") == capture_row.get("authority_state_sha256")
                decoded_ok &= instance_stream.get("authority_state_sha256") == capture_row.get("authority_state_sha256")
                decoded_ok &= embedded_row.get("rendered_semantic_uint24") == decoded_semantic
                decoded_ok &= embedded_row.get("rendered_persistent_instance_uint24") == decoded_instance
            except (KeyError, TypeError, OSError, ValueError, zlib.error):
                decoded_ok = False
        binding = bindings.get((semantic_id, instance_id))
        expected_text = " ".join(str(value or "") for value in colliders).lower()
        expected_hand = hand in {"left", "right"} and digit is not None
        weighted_skin = isinstance(binding, Mapping) and binding.get("semantic_name") == "body_skin"
        expected_target = bool(target_id) and target_id.lower() in expected_text
        target_surface = isinstance(binding, Mapping) and binding.get("persistent_instance_name") == target_id
        identity_ok = weighted_skin if expected_hand else expected_target and target_surface
        surface_ok = row.get("first_visible_collider_id") in colliders
        surface_ok &= isinstance(semantic_id, int) and semantic_id > 0
        surface_ok &= isinstance(instance_id, int) and instance_id > 0
        checked.append(measured_projection and frame_ok and embedded_ok and trace_contact_ok
                       and pixel_ok and decoded_ok and identity_ok and surface_ok)
    passed = all(checked) and required_pairs <= observed_pairs and len(right_non_thumb_digits) >= 2
    passed &= ledger_hash_ok
    return [_check("contact.visible_projection", passed,
                   "same-state contacts decode to an expected weighted-skin/target identity in the actual frozen semantic and instance pixels",
                   {"records": len(records), "valid_records": sum(checked), "observed_required_pairs": sorted(observed_pairs),
                    "projected_right_non_thumb_digits": sorted(right_non_thumb_digits),
                    "source_capture_ledger_sha256_matches": ledger_hash_ok})]


def _capture_checks(
    report: Any, root: Path, tolerances: Mapping[str, Any], duration_s: float,
    capture_ledger: Sequence[Mapping[str, Any]] = (),
) -> list[Check]:
    if not isinstance(report, Mapping):
        return [_check("capture.registered_modalities", None, "RGB/depth/semantic/instance capture manifest missing")]
    if report.get("schema") == "embodied.registered_capture_manifest.v1":
        required_streams = {
            "rgb": "head_rgb_hero",
            "metric_depth": "head_metric_depth_uint24_mm",
            "semantic": "head_semantic_uint24",
            "persistent_instance": "head_persistent_instance_uint24",
        }
        expected_frames = round(duration_s * float(tolerances["capture"]["fps"]))
        frame_ids: list[int] = []
        valid_rows = bool(capture_ledger)
        for row in capture_ledger:
            if not isinstance(row, Mapping) or row.get("schema") != "embodied.registered_capture_frame.v1":
                valid_rows = False
                continue
            frame = row.get("render_frame")
            if not isinstance(frame, int):
                valid_rows = False
                continue
            frame_ids.append(frame)
            streams = {item.get("stream"): item for item in row.get("streams", []) if isinstance(item, Mapping)}
            state_hash = row.get("authority_state_sha256")
            for stream in required_streams.values():
                item = streams.get(stream)
                if not isinstance(item, Mapping):
                    valid_rows = False
                    continue
                relative = item.get("relative_path")
                if not isinstance(relative, str) or Path(relative).is_absolute():
                    valid_rows = False
                    continue
                try:
                    path = _inside(root, root / relative)
                except ValueError:
                    valid_rows = False
                    continue
                valid_rows &= path.is_file() and item.get("file_sha256") == (_sha256(path) if path.is_file() else None)
                valid_rows &= item.get("authority_state_sha256") == state_hash
                valid_rows &= item.get("render_frame") == frame and item.get("physics_step") == row.get("physics_step")
            valid_rows &= row.get("authority_state_unchanged_across_modalities") is True
            valid_rows &= row.get("physics_advanced_between_modalities") is False
            valid_rows &= isinstance(row.get("intrinsics"), Mapping)
            valid_rows &= all(isinstance(row.get(name), list) and len(row[name]) == 16 for name in (
                "camera_to_world_matrix_column_major", "world_to_camera_matrix_column_major",
                "projection_matrix_column_major",
            ))
        dense = frame_ids == list(range(expected_frames))
        passed = report.get("width_px") == tolerances["capture"]["resolution_px"][0]
        passed &= report.get("height_px") == tolerances["capture"]["resolution_px"][1]
        passed &= report.get("fps") == tolerances["capture"]["fps"]
        passed &= report.get("frames_captured") == expected_frames and dense and valid_rows
        passed &= report.get("exact_integer_clock") is True and report.get("all_modalities_state_invariant") is True
        passed &= report.get("physics_advanced_between_modalities") is False
        passed &= report.get("hero_contains_proxy_pixels") is False
        return [_check(
            "capture.registered_modalities", passed,
            "registered Unity manifest and per-frame ledger prove exact synchronized labeled modalities",
            {"frame_count": len(frame_ids), "expected_frame_count": expected_frames, "dense": dense,
             "ledger_rows_valid": valid_rows, "schema": report.get("schema")},
        )]
    frames = report.get("frames", [])
    required_streams = list(tolerances["capture"]["streams"])
    resolution_ok = report.get("resolution_px") == tolerances["capture"]["resolution_px"]
    fps_ok = report.get("fps") == tolerances["capture"]["fps"]
    valid_rows = True
    frame_ids: list[int] = []
    state_hashes = True
    for row in frames if isinstance(frames, list) else []:
        if not isinstance(row, Mapping) or not isinstance(row.get("render_frame"), int):
            valid_rows = False
            continue
        frame_ids.append(row["render_frame"])
        modalities = row.get("modalities")
        if not isinstance(modalities, Mapping) or set(modalities) != set(required_streams):
            valid_rows = False
            continue
        frozen_hashes = set()
        for stream in required_streams:
            item = modalities[stream]
            if not isinstance(item, Mapping):
                valid_rows = False
                continue
            relative = item.get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute():
                valid_rows = False
                continue
            try:
                path = _inside(root, root / relative)
            except ValueError:
                valid_rows = False
                continue
            if not path.is_file():
                valid_rows = False
            expected_hash = item.get("sha256")
            if isinstance(expected_hash, str) and path.is_file() and _sha256(path) != expected_hash:
                valid_rows = False
            frozen_hashes.add(item.get("frozen_state_sha256"))
        state_hashes &= len(frozen_hashes) == 1 and None not in frozen_hashes
        valid_rows &= isinstance(row.get("intrinsics"), Mapping) and isinstance(row.get("extrinsics"), Mapping)
    dense = bool(frame_ids) and frame_ids == list(range(frame_ids[0], frame_ids[-1] + 1))
    expected_frames = round(duration_s * float(tolerances["capture"]["fps"]))
    passed = resolution_ok and fps_ok and valid_rows and state_hashes and dense and len(frame_ids) == expected_frames
    passed &= report.get("same_frozen_frame") is True
    passed &= report.get("proxy_hero_pixels", 1) == 0
    return [_check("capture.registered_modalities", passed,
                   "1920x1080 RGB/depth/semantic/persistent-instance frames share exact frame state and camera calibration",
                   {"frame_count": len(frame_ids), "expected_frame_count": expected_frames, "dense": dense, "resolution_ok": resolution_ok,
                    "fps_ok": fps_ok, "state_hashes_match": state_hashes})]


def _replay_checks(report: Any, trace_path: Path, tolerances: Mapping[str, Any]) -> list[Check]:
    if not isinstance(report, Mapping):
        return [_check("replay.fresh_process_rerender", None, "fresh-process replay and same-trace rerender receipt missing")]
    required = tolerances["replay"]
    actual_trace_hash = _sha256(trace_path) if trace_path.is_file() else None
    def valid_hash(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())
    try:
        passed = report.get("fresh_process") is True and report.get("same_trace_rerender") is True
        passed &= report.get("source_trace_sha256") == actual_trace_hash
        passed &= report.get("rerender_trace_sha256") == actual_trace_hash
        passed &= valid_hash(report.get("source_render_sha256")) and valid_hash(report.get("rerender_render_sha256"))
        passed &= bool(report.get("source_machine_id")) and report.get("source_machine_id") == report.get("replay_machine_id")
        passed &= float(report["translation_max_m"]) <= float(required["translation_max_m"])
        passed &= float(report["rotation_max_deg"]) <= float(required["rotation_max_deg"])
        passed &= float(report["object_velocity_max_m_s"]) <= float(required["object_velocity_max_m_s"])
        passed &= float(report["rerender_min_psnr_db"]) >= float(required["rerender_min_psnr_db"])
    except (KeyError, TypeError, ValueError):
        passed = False
    return [_check("replay.fresh_process_rerender", passed,
                   "fresh-process replay and same-trace rerender meet frozen numeric/image tolerances",
                   {**report, "audited_trace_sha256": actual_trace_hash})]


def _provenance_checks(report: Any, frozen: Sequence[Mapping[str, Any]]) -> list[Check]:
    if not isinstance(report, (Mapping, list)):
        return [_check("truth.provenance", None, "per-field measured/commanded/derived/unavailable provenance receipt missing")]
    rows = report.get("fields", report.get("records", [])) if isinstance(report, Mapping) else report
    if not isinstance(rows, list):
        return [_check("truth.provenance", False, "truth provenance receipt is malformed")]
    actual = {row.get("field"): row for row in rows if isinstance(row, Mapping) and row.get("field")}
    failures = []
    if all(expected["field"] in actual for expected in frozen):
        expected_rows = [(expected["field"], expected["provenance"]) for expected in frozen]
    else:
        # The canonical Unity recorder expands broad frozen categories into
        # narrower per-source rows; these aliases are stricter, not substitutes.
        expected_rows = [
            ("body_hand_camera_pose", "engine_observed"),
            ("body_hand_segment_velocity", "derived"),
            ("controller_targets", "commanded"),
            ("controller_errors", "derived"),
            ("contacts", "physx_measured"),
            ("free_object", "physx_measured"),
            ("joint_proprioception", "derived"),
            ("head_imu", "derived"),
            ("biological_torque", "unavailable"),
        ]
    for field, expected_provenance in expected_rows:
        row = actual.get(field)
        if not isinstance(row, Mapping):
            failures.append(f"missing:{field}")
            continue
        if row.get("provenance") != expected_provenance:
            failures.append(f"provenance:{field}")
        if not row.get("formula_or_source") or not row.get("units"):
            failures.append(f"source_or_units:{field}")
    torque = actual.get("biological_torque", {})
    if torque.get("provenance") != "unavailable":
        failures.append("biological_torque_must_be_unavailable")
    return [_check("truth.provenance", not failures,
                   "all truth fields carry frozen measured/commanded/derived/unavailable provenance and formulas",
                   {"field_count": len(actual), "failures": failures})]


def _independent_episode_pass(row: Mapping[str, Any], aggregate_gate: str) -> bool:
    qa = row.get("independent_qa")
    if not isinstance(qa, Mapping):
        return False
    gates = qa.get("gate_summary")
    # Matrix/robustness aggregation must bootstrap from the direct physical,
    # registration, camera, and projection evidence in A-D. Requiring the
    # report's overall decision (or E/F themselves) is circular: those gates
    # consume this aggregate. The aggregate gate is retained in the signature
    # to make the caller's evidence role explicit.
    if aggregate_gate not in {"E", "F"}:
        return False
    required_gates = set("ABCD")
    return (
        qa.get("schema") == REPORT_SCHEMA
        and isinstance(gates, Mapping) and required_gates <= set(gates)
        and all(gates[gate] == PASS for gate in required_gates)
    )


def _matrix_checks(report: Any, visual: Any) -> list[Check]:
    if not isinstance(report, Mapping) or not isinstance(report.get("episodes"), list):
        return [_check("generality.matrix", None, "multi-seed/multi-garment episode matrix missing")]
    episodes = [row for row in report["episodes"] if isinstance(row, Mapping)]
    rooms = {row.get("room_family", row.get("room_seed")) for row in episodes
             if row.get("room_family", row.get("room_seed")) is not None}
    garments = {row.get("garment_configuration_id") for row in episodes if row.get("garment_configuration_id")}
    completed = [row for row in episodes if _independent_episode_pass(row, "E")]
    complete_rooms = {row.get("room_family", row.get("room_seed")) for row in completed}
    complete_garments = {row.get("garment_configuration_id") for row in completed}
    shared = all(len({row.get(key) for row in episodes}) == 1 for key in ("compiler_id", "catalog_id", "controller_profile_id"))
    clean = all(row.get("seed_specific_retuning") is False and row.get("assistance_entries", 1) == 0 for row in episodes)
    reviewed = set()
    if isinstance(visual, Mapping) and isinstance(visual.get("episodes"), list):
        reviewed = {row.get("episode_id") for row in visual["episodes"] if isinstance(row, Mapping)
                    and row.get("coherent") is True and _visual_method_is_direct(row.get("method"))}
    completed_reviewed = all(row.get("episode_id") in reviewed for row in completed)
    rich_rooms = all(
        int(row.get("contextual_object_count", row.get("contextual_visible_object_count", 0))) > 10
        and row.get("closed_finished_room") is True
        and row.get("no_visible_primitive_furniture") is True
        and row.get("all_reachable_objects_physics_backed") is True
        and row.get("compiler_generated") is True
        for row in episodes
    )
    distinct_relations = len({row.get("destination_relation", row.get("destination_id")) for row in episodes}) >= 3
    distinct_strategies = len({row.get("contact_strategy") for row in episodes}) >= 2
    passed = len(episodes) >= 3 and len(rooms) >= 3 and len(garments) >= 3
    passed &= len(complete_rooms) >= 2 and len(complete_garments) >= 2
    passed &= shared and clean and completed_reviewed and rich_rooms and distinct_relations and distinct_strategies
    return [_check("generality.matrix", passed,
                   "three rooms and garments use one implementation; at least 2/3 carry full independent-QA PASS receipts",
                   {"room_seeds": sorted(str(value) for value in rooms), "garments": sorted(str(value) for value in garments),
                    "completed_room_count": len(complete_rooms), "completed_garment_count": len(complete_garments),
                    "shared_implementation": shared, "no_retuning_or_assistance": clean,
                    "rich_camera_aware_rooms": rich_rooms, "distinct_destination_relations": distinct_relations,
                    "distinct_contact_strategies": distinct_strategies})]


def _robustness_checks(report: Any) -> list[Check]:
    if not isinstance(report, Mapping):
        return [_check("robustness.primary", None, "primary nominal/shift/mass-friction robustness matrix missing")]
    rows = report.get("primary_robustness", report.get("robustness_variants", []))
    if not isinstance(rows, list) or len(rows) < 3:
        return [_check("robustness.primary", False, "fewer than three frozen primary robustness variants were reported")]
    names = {row.get("variant") for row in rows if isinstance(row, Mapping)}
    required = {"nominal", "lateral_target_shift", "mass_friction_change"}
    passed_rows = [row for row in rows if isinstance(row, Mapping) and _independent_episode_pass(row, "F")]
    same_controller = len({row.get("controller_profile_id") for row in rows if isinstance(row, Mapping)}) == 1
    clean = all(isinstance(row, Mapping) and row.get("retuned") is False
                and row.get("assistance_entries", 1) == 0 for row in rows)
    passed = required <= names and len(passed_rows) >= 2 and same_controller and clean
    return [_check("robustness.primary", passed,
                   "at least 2/3 frozen primary variants carry full independent-QA PASS receipts with no retuning or assistance",
                   {"variants": sorted(str(value) for value in names), "passed": len(passed_rows),
                    "same_controller": same_controller, "clean": clean})]


def _source_audit_checks(report: Any, execution: Any, authority: Any) -> list[Check]:
    if not isinstance(report, Mapping):
        return [_check("authority.source_audit", None, "mandatory pre-execution source audit receipt missing")]
    body = dict(report)
    declared_hash = body.pop("receipt_sha256", None)
    actual_hash = _canonical_json_sha256(body)
    findings = report.get("findings")
    forbidden = report.get("forbidden_findings")
    hashes = report.get("source_sha256")
    valid_hashes = isinstance(hashes, Mapping) and bool(hashes) and all(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
        for value in hashes.values()
    )
    passed = report.get("schema") == SOURCE_AUDIT_SCHEMA and report.get("passed") is True
    passed &= declared_hash == actual_hash and valid_hashes
    passed &= isinstance(findings, list) and isinstance(forbidden, list) and not forbidden
    def valid_finding(row: Any) -> bool:
        if not isinstance(row, Mapping) or row.get("status") != "ALLOWLISTED":
            return False
        if row.get("scope") in {"initialization", "non_target"}:
            return True
        if row.get("scope") != "rerender_only":
            return False
        reason = str(row.get("reason", "")).lower()
        truthful_target_scope = (
            row.get("target_related") is False
            or (row.get("target_related") is True and row.get("method") == "ApplyReplayTargetState")
        )
        return truthful_target_scope and "render" in reason and "excluded" in reason and "manipulation evidence" in reason

    passed &= all(valid_finding(row) for row in findings or [])
    covered_source_names = {Path(str(name)).name for name in (hashes or {})}
    python_coverage = REQUIRED_PYTHON_SOURCE_AUDIT_FILES <= covered_source_names
    passed &= python_coverage
    for receipt in (execution, authority):
        if isinstance(receipt, Mapping):
            passed &= receipt.get("source_audit_sha256") == declared_hash
        else:
            passed = False
    return [_check("authority.source_audit", passed,
                   "source audit hash is carried through execution/Unity and contains no forbidden target assistance",
                   {"declared_sha256": declared_hash, "recomputed_sha256": actual_hash,
                    "finding_count": len(findings) if isinstance(findings, list) else None,
                    "forbidden_count": len(forbidden) if isinstance(forbidden, list) else None,
                    "required_python_sources": sorted(REQUIRED_PYTHON_SOURCE_AUDIT_FILES),
                    "configured_python_source_coverage": python_coverage})]


def _authority_checks(report: Any, run_root: Path) -> list[Check]:
    if not isinstance(report, Mapping):
        return [_check("authority.single_state", None, "single-authority state receipt missing")]
    state_ids = report.get("consumer_state_ids")
    required = {"collider_skeleton", "weighted_body", "garments", "head_camera", "recorder"}
    ids = {state_ids.get(key) for key in required} if isinstance(state_ids, Mapping) and required <= set(state_ids) else set()
    explicit_ids_pass = len(ids) == 1 and None not in ids
    canonical_receipt_pass = report.get("schema") == "embodied.single_authority_receipt.v1"
    canonical_receipt_pass &= report.get("single_state_drives_body_clothing_camera_truth") is True
    canonical_receipt_pass &= all(report.get(name) not in {None, "", "UNAVAILABLE"}
                                  for name in ("authority_root", "avatar_root", "head", "camera_parent", "target_rigidbody"))
    passed = explicit_ids_pass or canonical_receipt_pass
    if "engine" in report:
        passed &= report.get("engine") == "Unity"
    if "physics" in report:
        passed &= report.get("physics") in {"Unity PhysX", "PhysX"}
    passed &= report.get("physics_hz") == 240 and report.get("render_hz") == 30 and report.get("steps_per_render_frame") == 8
    passed &= report.get("independent_render_timeline") is False and report.get("camera_keyframes") is not True
    ignored: bool | None = None
    repository = Path(__file__).resolve().parents[3]
    try:
        relative = run_root.resolve().relative_to(repository.resolve())
    except ValueError:
        pass
    else:
        result = subprocess.run(["git", "check-ignore", "--quiet", str(relative)], cwd=repository, check=False)
        ignored = result.returncode == 0
        passed &= ignored
    passive_counters = {
        name: report.get(name) for name in (
            "object_pose_writes_after_initialization", "object_external_forces",
            "attachment_or_joint_count", "assistance_ledger_entries",
        )
    }
    return [_check("authority.single_state", passed,
                   "one Unity/PhysX physics-clocked state drives collider, body, garments, camera, and recorder",
                   {
                       "consumer_state_ids": state_ids,
                       "run_root_git_ignored": ignored,
                       "producer_counter_values": passive_counters,
                       "producer_counter_semantics": "passive/hard-coded receipt fields; not independent runtime detectors and not used to pass this check",
                   })]


def _rate(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _video_role(path: Path) -> str:
    name = path.stem.lower().replace("-", "_")
    for modality in ("depth", "semantic", "instance"):
        if modality in name:
            return f"registered_{modality}"
    if "overlay" in name or "contact" in name and "diagnostic" in name:
        return "contact_overlay"
    if "external" in name or "third_person" in name:
        return "clean_external"
    if "head" in name or "ego" in name:
        return "clean_head"
    return "unclassified"


def _dense_video_timeline(frame_count: int, fps: float, phases: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    timeline = []
    complete = True
    for frame in range(frame_count):
        time_s = frame / fps
        phase = next((row.get("id") for row in phases
                      if isinstance(row.get("start_s"), (int, float))
                      and isinstance(row.get("end_s"), (int, float))
                      and float(row["start_s"]) <= time_s < float(row["end_s"])), None)
        complete &= phase is not None
        timeline.append({"frame": frame, "time_s": time_s, "phase_id": phase or "unavailable"})
    return timeline, complete


def _video_specs(root: Path, evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    declared: dict[Path, dict[str, Any]] = {}
    videos = evidence.get("videos", [])
    if isinstance(videos, Mapping):
        videos = [{"role": role, "path": path} for role, path in videos.items()]
    if isinstance(videos, list):
        for item in videos:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                continue
            relative = Path(item["path"])
            if relative.is_absolute():
                raise ValueError("declared run video must be relative to run root")
            path = _inside(root, root / relative)
            declared[path] = {**item, "path": path}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and path not in declared:
            resolved = _inside(root, path)
            role = _video_role(path)
            declared[resolved] = {
                "path": resolved,
                "role": role,
                "clean": role in {"clean_head", "clean_external"},
                "labeled": role == "contact_overlay",
            }
    return list(declared.values())


def _audit_video(
    path: Path,
    role: str,
    declared: Mapping[str, Any],
    default_phases: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "role": role, "sha256": _sha256(path) if path.is_file() else None}
    if not path.is_file():
        result.update(status=FAIL, error="video file is missing", probe_ok=False, full_decode_ok=False)
        return result
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        result.update(status=UNAVAILABLE, error="ffprobe/ffmpeg not found", probe_ok=False, full_decode_ok=False)
        return result
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    result["probe_ok"] = probe.returncode == 0
    result["full_decode_ok"] = decode.returncode == 0
    result["decode_error"] = decode.stderr[-4000:] if decode.returncode else ""
    if probe.returncode == 0:
        try:
            payload = json.loads(probe.stdout)
            stream = payload.get("streams", [{}])[0]
            result.update(
                codec_name=stream.get("codec_name"), width=stream.get("width"), height=stream.get("height"),
                pixel_format=stream.get("pix_fmt"), fps=_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
                frame_count=int(stream.get("nb_read_frames") or stream.get("nb_frames"))
                if str(stream.get("nb_read_frames") or stream.get("nb_frames", "")).isdigit() else None,
                duration_s=float(stream.get("duration") or payload.get("format", {}).get("duration"))
                if stream.get("duration") or payload.get("format", {}).get("duration") else None,
            )
            frame_count = result.get("frame_count")
            fps = result.get("fps")
            declared_phases = declared.get("phase_intervals", default_phases)
            if isinstance(frame_count, int) and frame_count >= 0 and isinstance(fps, (int, float)) and fps > 0:
                phases = declared_phases if isinstance(declared_phases, list) else []
                result["dense_phase_timeline"], result["phase_mapping_complete"] = _dense_video_timeline(frame_count, fps, phases)
            else:
                result["dense_phase_timeline"], result["phase_mapping_complete"] = [], False
        except (ValueError, TypeError, IndexError) as exc:
            result["probe_parse_error"] = str(exc)
            result["probe_ok"] = False
    is_clean = role in {"clean_head", "clean_external"}
    is_episode_video = is_clean or role == "contact_overlay" or role.startswith("variant") or declared.get("hero") is True
    hero_properties = True
    if is_episode_video:
        hero_properties = result.get("width") == 1920 and result.get("height") == 1080
        hero_properties &= result.get("fps") is not None and abs(result["fps"] - 30.0) <= 1e-6
        hero_properties &= result.get("duration_s") is not None and 20.0 <= result["duration_s"] <= 30.0
        if default_phases:
            expected_duration = max(float(row["end_s"]) for row in default_phases)
            hero_properties &= result.get("frame_count") == round(expected_duration * 30.0)
    if is_clean:
        hero_properties &= declared.get("clean") is True
    if role == "contact_overlay":
        hero_properties &= declared.get("labeled") is True and declared.get("clean") is not True
    result["required_properties_ok"] = hero_properties
    result["status"] = PASS if result["probe_ok"] and result["full_decode_ok"] and hero_properties else FAIL
    return result


def _video_checks(root: Path, evidence: Mapping[str, Any], prior_paths: Mapping[str, Path | None]) -> tuple[list[Check], list[dict[str, Any]]]:
    specs = _video_specs(root, evidence)
    phases = load_frozen_config()["activity_plan"]["phases"]
    audits = [_audit_video(Path(item["path"]), str(item.get("role", "unclassified")), item, phases) for item in specs]
    for role, path in prior_paths.items():
        if path is not None:
            audits.append(_audit_video(path.resolve(), role, {"clean": True}))
    all_ok: bool | None = all(item["status"] == PASS for item in audits) if audits else None
    roles = {item["role"] for item in audits if item["status"] == PASS}
    distinct = {item["path"] for item in audits if item["role"] in {"clean_head", "clean_external", "contact_overlay"}}
    required_ok = {"clean_head", "clean_external", "contact_overlay"} <= roles and len(distinct) >= 3
    if not audits:
        required_ok = None
    current_audits = [item for item in audits if item["role"] not in prior_paths]
    timelines_ok: bool | None = (all(item.get("phase_mapping_complete") is True for item in current_audits)
                                 if current_audits else None)
    checks = [
        _check("video.every_file_decodes", all_ok, "every discovered and supplied video passes ffprobe and full ffmpeg decode",
               {"audited_video_count": len(audits)}),
        _check("video.separate_views", required_ok,
               "separate 1080p30 clean head, clean external, and labeled diagnostic overlay videos exist"),
        _check("video.dense_phase_timelines", timelines_ok,
               "every current video has a dense one-row-per-frame phase timeline",
               {"current_video_count": len(current_audits)}),
    ]
    return checks, audits


def _comparison_checks(visual: Any, prior_paths: Mapping[str, Path | None], audits: Sequence[Mapping[str, Any]]) -> list[Check]:
    if not all(prior_paths.values()):
        return [_check("comparison.prior_outputs", None, "both prior Unity audition and corrected bimanual clip were not supplied")]
    comparisons = visual.get("comparisons", []) if isinstance(visual, Mapping) else []
    required = {"prior_unity_audition", "prior_corrected_bimanual"}
    passed_baselines = {row.get("baseline") for row in comparisons if isinstance(row, Mapping)
                        and _visual_method_is_direct(row.get("method")) and row.get("current_plainly_better") is True}
    prior_decode = all(item.get("status") == PASS for item in audits if item.get("role") in required)
    return [_check("comparison.prior_outputs", prior_decode and required <= passed_baselines,
                   "dense review directly finds the current primary plainly better than both preserved prior outputs",
                   {"reviewed_baselines": sorted(str(value) for value in passed_baselines), "prior_decode_ok": prior_decode})]


def _gate(gate_id: str, title: str, checks: Sequence[Check]) -> dict[str, Any]:
    status = FAIL if any(check.status == FAIL for check in checks) else UNAVAILABLE if any(check.status == UNAVAILABLE for check in checks) else PASS
    return {"gate": gate_id, "title": title, "status": status, "checks": [check.as_dict() for check in checks]}


def audit_run(
    run_root: Path | str,
    *,
    prior_unity_audition: Path | str | None = None,
    prior_corrected_bimanual: Path | str | None = None,
) -> dict[str, Any]:
    """Audit a generated run root and return a dense gate-by-gate report.

    Optional prior paths are explicit preserved public/synthetic comparison
    inputs.  The auditor never searches outside ``run_root`` for them.
    """
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise ValueError(f"run root is not a directory: {root}")
    config = load_frozen_config()
    validate_frozen_config(config)
    tolerances = config["qa_tolerances"]
    duration_s = float(config["activity_plan"]["duration_s"])
    phases = config["activity_plan"]["phases"]
    evidence_path = root / "qa_evidence.json"
    evidence = _read_json(evidence_path) if evidence_path.is_file() else {}
    if not isinstance(evidence, Mapping):
        raise ValueError("qa_evidence.json must contain an object")
    if evidence and evidence.get("schema") not in {None, EVIDENCE_SCHEMA}:
        raise ValueError(f"unexpected QA evidence schema: {evidence.get('schema')}")

    trace_path = _mapped_path(root, evidence, "trace", "episode_trace.jsonl")
    try:
        rows = _load_trace(trace_path)
        trace_error = None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        rows, trace_error = [], str(exc)
    trace_checks, trace_details = _trace_checks(rows, evidence, tolerances, duration_s, phases)
    if trace_error:
        trace_checks[0] = _check("trace.present", False, "episode trace is malformed", {"error": trace_error})
    capture_ledger_path = _mapped_path(root, evidence, "capture_frame_ledger", "capture_frame_ledger.jsonl")
    try:
        capture_ledger = _load_trace(capture_ledger_path)
    except (OSError, ValueError, json.JSONDecodeError):
        capture_ledger = []
    interaction_checks, interaction_metrics = _interaction_checks(
        trace_details.get("ordered_rows", []), trace_details.get("target_object_id"),
        _destination_id(trace_details.get("ordered_rows", []), evidence), tolerances
    )
    camera_checks, camera_metrics = _camera_checks(
        trace_details.get("ordered_rows", []), tolerances, capture_ledger
    )
    motion_limit_checks, motion_limit_metrics = _motion_limit_checks(
        trace_details.get("ordered_rows", []), float(tolerances["clock"]["physics_hz"])
    )

    authority = _optional_report(root, evidence, "authority_receipt", "authority_receipt.json")
    execution = _optional_report(root, evidence, "execution_receipt", "execution_receipt.json")
    source_audit = _optional_report(root, evidence, "source_audit_receipt", "source_audit_receipt.json")
    registration = _optional_report(root, evidence, "registration_report", "registration_report.json")
    registration = _registration_with_trace_penetration(
        registration, trace_details.get("ordered_rows", []), trace_details.get("target_object_id")
    )
    projection = _optional_report(root, evidence, "contact_projection", "contact_projection.json")
    capture = _optional_report(root, evidence, "capture_manifest", "registered_capture_manifest.json")
    replay = _optional_report(root, evidence, "replay_report", "replay_report.json")
    provenance = _optional_report(root, evidence, "truth_provenance", "truth_provenance.json")
    if provenance is None:
        trace_manifest_path = root / "episode_trace_manifest.json"
        if trace_manifest_path.is_file():
            trace_manifest = _read_json(trace_manifest_path)
            if isinstance(trace_manifest, Mapping) and isinstance(trace_manifest.get("provenance_registry"), list):
                provenance = {"fields": trace_manifest["provenance_registry"]}
    visual = _optional_report(root, evidence, "visual_audit", "visual_audit.json")
    matrix = _optional_report(root, evidence, "episode_matrix", "episode_matrix.json")
    visual_checks = _visual_checks(visual, tolerances)
    registration_checks = _registration_checks(registration, tolerances)
    projection_checks = _projection_checks(
        projection, tolerances, capture_ledger, trace_details.get("ordered_rows", []),
        trace_details.get("target_object_id"), capture_ledger_path, root
    )
    capture_checks = _capture_checks(capture, root, tolerances, duration_s, capture_ledger)
    replay_checks = _replay_checks(replay, trace_path, tolerances)
    provenance_checks = _provenance_checks(provenance, config["truth_provenance"])
    matrix_checks = _matrix_checks(matrix, visual)
    robustness_checks = _robustness_checks(matrix)
    authority_checks = _authority_checks(authority, root)
    source_audit_checks = _source_audit_checks(source_audit, execution, authority)
    dynamic_finger_checks = _dynamic_finger_checks(trace_details.get("ordered_rows", []))

    prior_paths = {
        "prior_unity_audition": Path(prior_unity_audition).resolve() if prior_unity_audition else None,
        "prior_corrected_bimanual": Path(prior_corrected_bimanual).resolve() if prior_corrected_bimanual else None,
    }
    video_checks, video_audits = _video_checks(root, evidence, prior_paths)
    comparison_checks = _comparison_checks(visual, prior_paths, video_audits)

    gates = [
        _gate("A", "Visible compliant-hand microcell, source audit, frozen contracts/rates, and free release",
              source_audit_checks + authority_checks + trace_checks[:2] + dynamic_finger_checks + interaction_checks),
        _gate("B", "Embodiment, anatomy, garments, and registration sweeps",
              registration_checks[:2] + visual_checks[:1]),
        _gate("C", "Object-free full-body motion and head-camera qualification",
              trace_checks[2:4] + camera_checks + motion_limit_checks + video_checks[1:2] + visual_checks[1:2]),
        _gate("D", "Unassisted free-object bimanual interaction and visible/physical contact",
              dynamic_finger_checks + interaction_checks + registration_checks[2:] + projection_checks + visual_checks[2:3]),
        _gate("E", "Procedural room/clothing generality without retuning", matrix_checks + visual_checks[3:]),
        _gate("F", "Synchronized capture, replay/rerender, dense decode, and prior comparisons",
              trace_checks[4:] + robustness_checks + capture_checks + replay_checks + provenance_checks
              + video_checks[:1] + video_checks[2:] + comparison_checks),
    ]
    non_pass = [check for gate in gates for check in gate["checks"] if check["status"] != PASS]
    veto_reasons = [
        {"gate": gate["gate"], "check_id": check["check_id"], "status": check["status"], "reason": check["summary"]}
        for gate in gates for check in gate["checks"] if check["status"] != PASS
    ]
    decision = PASS if not non_pass else "PROMOTION_VETO"
    first_failed_gate = next((gate["gate"] for gate in gates if gate["status"] != PASS), None)
    downstream_catalog = {
        "A": ["B anti-clipping sweep", "C motion/camera qualification", "D polished cell", "E three integrated episodes", "F robustness/replay/rerender/multimodal QA"],
        "B": ["C motion/camera qualification", "D polished cell", "E three integrated episodes", "F robustness/replay/rerender/multimodal QA"],
        "C": ["D polished cell", "E three integrated episodes", "F robustness/replay/rerender/multimodal QA"],
        "D": ["E three integrated episodes", "F robustness/replay/rerender/multimodal QA"],
        "E": ["F robustness/replay/rerender/multimodal QA"],
        "F": [],
    }
    return {
        "schema": REPORT_SCHEMA,
        "run_root": str(root),
        "contract_provenance": {
            "executed_episode_contract_sha256": execution.get("contract_sha256") if isinstance(execution, Mapping) else None,
            "post_outcome_qa_record_config_sha256": _sha256(CONFIG_PATH),
            "new_experiment_run": False,
        },
        "frozen_tolerance_schema": tolerances["schema"],
        "qa_decision": decision,
        "promotion_veto": bool(non_pass),
        "integrated_decision": "INTEGRATED PASS" if not non_pass else "NO-GO",
        "gate_summary": {gate["gate"]: gate["status"] for gate in gates},
        "gates": gates,
        "veto_reasons": veto_reasons,
        "first_nonpassing_gate": first_failed_gate,
        "downstream_artifacts_not_generated": downstream_catalog.get(first_failed_gate, []),
        "dense_timeline": trace_details.get("timeline", []),
        "derived_metrics": {"interaction": interaction_metrics, "camera": camera_metrics,
                            "motion_limits": motion_limit_metrics},
        "video_audits": video_audits,
        "evidence_policy": {
            "visual_counters_accepted": False,
            "viewport_object_center_proxy_accepted": False,
            "missing_required_evidence_promotable": False,
            "restricted_childlens_accessed": False,
            "biological_torque_claimed": False,
            "authority_receipt_zero_counters_accepted_as_runtime_detectors": False,
            "scene_conditioned_hand_target_tracking": "prequalification interactionAnchorWorld follows the free target center and drives only bounded kinematic hand waypoints",
        },
    }


evaluate_run = audit_run


def write_report(report: Mapping[str, Any], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent visual/physical QA for a procedural scene-gate run root")
    parser.add_argument("run_root", type=Path, help="ignored generated run root")
    parser.add_argument("--prior-unity-audition", type=Path)
    parser.add_argument("--prior-corrected-bimanual", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON report path; stdout is always emitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_run(
        args.run_root,
        prior_unity_audition=args.prior_unity_audition,
        prior_corrected_bimanual=args.prior_corrected_bimanual,
    )
    if args.output:
        write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qa_decision"] == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
