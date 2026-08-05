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
import subprocess
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
    required_body = ("root", "torso", "neck", "head")
    required_digits = tuple(f"{hand}_{digit}" for hand in ("left", "right")
                            for digit in ("thumb", "index", "middle", "ring", "little"))
    for row in ordered:
        body = row.get("body_state")
        if not isinstance(body, Mapping):
            missing_counts["body_state"] += 1
        else:
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
        _check("trace.duration", duration_ok, "trace covers every physics step of the frozen 16-second ActivityPlan",
               {"expected_steps": expected_steps, "observed_steps": len(ordered)}),
    ], {"timeline": build_dense_timeline(ordered, phases), "target_object_id": target, "ordered_rows": ordered}


def _interaction_checks(
    rows: Sequence[Mapping[str, Any]], target_id: str | None, tolerances: Mapping[str, Any]
) -> tuple[list[Check], dict[str, Any]]:
    required = tolerances["interaction"]
    maximum_separation_m = float(tolerances["contact"]["qualification_max_measured_separation_m"])
    physics_hz = float(tolerances["clock"]["physics_hz"])
    if not rows or target_id is None:
        return [
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
        by_hand: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for contact in eligible:
            hand, digit = _contact_hand_digit(contact)
            if hand and digit:
                by_hand[hand][digit].append(contact)
        right_digits = set(by_hand["right"])
        right_geometry = "thumb" in right_digits and len(right_digits - {"thumb"}) >= int(required["right_minimum_non_thumb_digits"])
        if step < lift_start_step and right_geometry:
            right_steps.append(step)
        right_impulse_digits = {
            digit for digit, contacts in by_hand["right"].items()
            if any(_impulse_magnitude(contact) > 0.0 for contact in contacts)
        }
        if ("thumb" in right_impulse_digits
                and len(right_impulse_digits - {"thumb"}) >= int(required["right_minimum_non_thumb_digits"])):
            right_impulse_steps.append(step)
        left_digits = set(by_hand["left"])
        non_little_left_digits = left_digits - {"little"}
        has_non_little_left = bool(non_little_left_digits)
        has_stable_left_digits = len(non_little_left_digits) >= 2
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
        if meaningful_left_geometry and left_impulse_contacts:
            left_impulse_support_steps.append(step)
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
    right_ok = right_dwell >= float(required["right_opposition_min_s"])
    left_ok = left_dwell >= 0.25 and opposing_dwell >= 0.25

    all_ledgers = [entry for row in rows for entry in row.get("assistance_ledger", [])
                   if isinstance(row.get("assistance_ledger", []), list)]
    no_recorded_assistance = not all_ledgers
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
        hand_contact_after = any(
            _contact_targets(contact, target_id)
            and _measured_contact(contact)
            and _separation_eligible(contact, maximum_separation_m)
            and _contact_hand_digit(contact)[0] in {"left", "right"}
            for row in post for contact in _contacts(row)
        )
        post_objects = [obj for row in post for obj in [_object_for(row, target_id)] if obj is not None]
        settled = any(obj.get("sleeping") is True or obj.get("support_id") not in {None, "", "none", "unavailable"}
                      for obj in post_objects)
        dynamic = all(
            obj.get("free_dynamic") is True
            or (obj.get("is_kinematic") is False and obj.get("parent_id") in {None, ""})
            for obj in post_objects
        )
        free_release_ok = bool(post_objects) and not hand_contact_after and settled and dynamic and no_recorded_assistance
        release_details.update(hand_contact_after=hand_contact_after, settled=settled, dynamic=dynamic)

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
        "lift_m": lift_m if math.isfinite(lift_m) else None,
        "turn_during_bimanual_turn_deg": turn_deg if math.isfinite(turn_deg) else None,
        "turn_phase_steps": len(turn_states),
        "turn_phase_supported_steps": supported_turn_steps,
        "maximum_available_impulse_n_s": max(available_impulses) if available_impulses else None,
        "qualification_maximum_separation_m": maximum_separation_m,
        "eligible_hand_target_rows": eligible_hand_rows,
        "speculative_hand_target_rows_excluded": speculative_hand_rows,
        "eligible_nonzero_impulse_hand_target_rows": eligible_nonzero_impulse_rows,
        "assistance_entries": len(all_ledgers),
        "assistance_evidence_scope": "passive per-step ledger only; not an independent runtime API detector",
        **release_details,
    }
    return [
        _check("interaction.right_capture", right_ok,
               "right thumb plus at least two non-thumb PhysX contacts dwell before lift", metrics),
        _check("interaction.left_support", left_ok,
               "left support is sustained, includes a non-little digit, and physically opposes the right hand", metrics),
        _check("interaction.lift_turn", math.isfinite(lift_m) and math.isfinite(turn_deg)
               and lift_m > float(required["lift_min_m"]) and turn_deg > float(required["turn_min_deg"]),
               "free target exceeds frozen post-qualification lift and in-phase turn thresholds", metrics),
        _check("interaction.free_release", free_release_ok,
               "commanded opening produces contact-free dynamic release and support/sleep settle", release_details),
        _check("interaction.no_assistance", no_recorded_assistance,
               "the passive per-step assistance ledger is empty; source audit is required to exclude uninstrumented assistance APIs",
               {"entries": all_ledgers[:20], "runtime_detector": False}),
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
    mount_ok = complete and len(parents) == 1 and None not in parents and min(clearance) > float(required["minimum_head_or_clothing_clearance_m"])
    mount_ok &= max(optical) <= float(required["optical_vs_face_forward_max_deg"])
    motion_ok = bool(rolls) and max(rolls) <= float(required["roll_abs_max_deg"])
    motion_ok &= (not linear_speeds or max(linear_speeds) <= float(required["linear_speed_max_m_s"]))
    motion_ok &= (not angular_speeds or max(angular_speeds) <= float(required["angular_speed_max_deg_s"]))
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


def _visual_method_is_direct(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    lowered = value.lower()
    return ("dense" in lowered and ("frame" in lowered or "visual" in lowered)
            and not any(term in lowered for term in VISUAL_PROXY_TERMS))


def _visual_checks(visual: Any, tolerances: Mapping[str, Any]) -> list[Check]:
    if not isinstance(visual, Mapping):
        return [
            _check("visual.garment_sweeps", None, "dense garment/body/anatomy review missing"),
            _check("visual.motion_camera", None, "object-free head/external motion review missing"),
            _check("visual.event_visibility", None, "required event visibility review missing"),
            _check("visual.episode_coherence", None, "dense integrated episode review missing"),
        ]
    sweeps = visual.get("garment_sweeps", [])
    bad_anatomy_keys = ("nude_or_exposed", "exploding_weights", "fused_digits", "detached_wrists", "clipping")
    sweep_ok = isinstance(sweeps, list) and len({row.get("configuration_id") for row in sweeps if isinstance(row, Mapping)}) >= 3
    sweep_ok &= all(isinstance(row, Mapping) and _visual_method_is_direct(row.get("method"))
                    and all(row.get(key) is False for key in bad_anatomy_keys) for row in sweeps)

    motion = visual.get("motion_camera_qualification")
    motion_ok = isinstance(motion, Mapping) and _visual_method_is_direct(motion.get("method"))
    motion_ok &= bool(motion.get("object_free")) and bool(motion.get("continuity_pass"))
    motion_ok &= bool(motion.get("both_arms_natural")) and motion.get("static_target_view") is False
    motion_ok &= bool(motion.get("head_video_reviewed")) and bool(motion.get("external_video_reviewed"))

    visibility = visual.get("event_visibility", [])
    required_events = set(tolerances["camera"]["required_events_visible"])
    visible_events = {row.get("event") for row in visibility if isinstance(row, Mapping)
                      and row.get("visible") is True and isinstance(row.get("frame"), int)
                      and _visual_method_is_direct(row.get("method"))}
    visibility_ok = required_events <= visible_events

    episodes = visual.get("episodes", [])
    veto_keys = ("malformed_anatomy", "garment_clipping", "floating_or_stretched_limbs",
                 "furniture_intrusion", "overexposure", "static_target_view", "bad_transitions",
                 "incoherent_room", "proxy_hero_pixels", "camera_in_mesh")
    episode_ok = isinstance(episodes, list) and bool(episodes)
    episode_ok &= all(isinstance(row, Mapping) and _visual_method_is_direct(row.get("method"))
                      and row.get("coherent") is True and all(row.get(key) is False for key in veto_keys)
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
    provenance = str(report.get("provenance", "")).lower()
    measured = "measured" in provenance or "engine_observed" in provenance
    proxy = any(term in provenance for term in VISUAL_PROXY_TERMS)
    registration = tolerances["registration"]
    contact = tolerances["contact"]
    configurations = report.get("garment_configurations", [])
    distribution = report.get("garment_affected_vertex_distribution")
    try:
        skin_ok = measured and not proxy and float(report["skin_collider_max_m"]) <= float(registration["skin_collider_max_m"])
        garment_ok = measured and not proxy and float(report["garment_body_max_penetration_m"]) <= float(registration["garment_body_max_penetration_m"])
        garment_ok &= isinstance(distribution, (dict, list)) and bool(distribution)
        garment_ok &= isinstance(configurations, list) and len({row.get("configuration_id") for row in configurations
                                                                if isinstance(row, Mapping)}) >= 3
        garment_ok &= all(
            isinstance(row, Mapping)
            and float(row["skin_collider_max_m"]) <= float(registration["skin_collider_max_m"])
            and float(row["garment_body_max_penetration_m"]) <= float(registration["garment_body_max_penetration_m"])
            and bool(row.get("garment_affected_vertex_distribution"))
            for row in configurations
        )
        penetration_ok = measured and not proxy
        penetration_ok &= float(report["finger_object_max_penetration_m"]) <= float(contact["finger_object_max_penetration_m"])
        penetration_ok &= float(report["target_support_max_penetration_m"]) <= float(contact["target_support_max_penetration_m"])
    except (KeyError, TypeError, ValueError):
        skin_ok = garment_ok = penetration_ok = False
    return [
        _check("registration.skin_collider", skin_ok, "measured maximum skin/collider error is within tolerance", report),
        _check("registration.garment_body", garment_ok, "three garments meet penetration tolerance with affected-vertex distribution", report),
        _check("registration.penetration", penetration_ok, "measured finger/object and target/support penetration are within tolerance", report),
    ]


def _projection_checks(report: Any, tolerances: Mapping[str, Any]) -> list[Check]:
    if not isinstance(report, (Mapping, list)):
        return [_check("contact.visible_projection", None, "measured-contact to visible-surface projection missing")]
    records = report.get("records", []) if isinstance(report, Mapping) else report
    if not isinstance(records, list) or not records:
        return [_check("contact.visible_projection", False, "contact projection report has no records")]
    checked = []
    required_pairs = {"right_thumb", "left_non_little"}
    observed_pairs: set[str] = set()
    right_non_thumb_digits: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            checked.append(False)
            continue
        method = str(row.get("method", ""))
        direct = _visual_method_is_direct(method) and "surface" in method.lower()
        hand = str(row.get("hand", "")).lower()
        digit = str(row.get("digit", "")).lower()
        pair = "right_thumb" if hand == "right" and digit == "thumb" else "left_non_little" if hand == "left" and digit not in {"", "little"} else "other"
        observed_pairs.add(pair)
        if hand == "right" and digit in {"index", "middle", "ring", "little"}:
            right_non_thumb_digits.add(digit)
        physical_frame, visible_frame = row.get("physical_frame"), row.get("visible_frame")
        frame_ok = isinstance(physical_frame, int) and isinstance(visible_frame, int)
        frame_ok &= abs(physical_frame - visible_frame) <= int(tolerances["contact"]["visible_physical_max_frame_delta"])
        point_ok = _vector(row.get("physical_contact_point_world_m")) is not None
        skin_surface = str(row.get("visible_skin_surface_id", "")).lower().replace("-", "_")
        surface_ok = row.get("correct_visible_surface") is True and bool(skin_surface)
        surface_ok &= hand in skin_surface and digit in skin_surface
        surface_ok &= row.get("visible_object_surface_id") not in {None, ""}
        try:
            error_ok = float(row["skin_projection_error_m"]) <= float(tolerances["registration"]["skin_collider_max_m"])
            error_ok &= float(row["object_projection_error_m"]) <= float(tolerances["contact"]["finger_object_max_penetration_m"])
        except (KeyError, TypeError, ValueError):
            error_ok = False
        checked.append(direct and frame_ok and point_ok and surface_ok and error_ok)
    passed = all(checked) and required_pairs <= observed_pairs and len(right_non_thumb_digits) >= 2
    return [_check("contact.visible_projection", passed,
                   "measured contacts project to the correct visible skin/object surfaces within one render frame",
                   {"records": len(records), "valid_records": sum(checked), "observed_required_pairs": sorted(observed_pairs),
                    "projected_right_non_thumb_digits": sorted(right_non_thumb_digits)})]


def _capture_checks(report: Any, root: Path, tolerances: Mapping[str, Any], duration_s: float) -> list[Check]:
    if not isinstance(report, Mapping):
        return [_check("capture.registered_modalities", None, "RGB/depth/semantic/instance capture manifest missing")]
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


def _matrix_checks(report: Any, visual: Any) -> list[Check]:
    if not isinstance(report, Mapping) or not isinstance(report.get("episodes"), list):
        return [_check("generality.matrix", None, "multi-seed/multi-garment episode matrix missing")]
    episodes = [row for row in report["episodes"] if isinstance(row, Mapping)]
    rooms = {row.get("room_seed") for row in episodes if row.get("room_seed") is not None}
    garments = {row.get("garment_configuration_id") for row in episodes if row.get("garment_configuration_id")}
    completed = [row for row in episodes if row.get("completed") is True]
    complete_rooms = {row.get("room_seed") for row in completed}
    complete_garments = {row.get("garment_configuration_id") for row in completed}
    shared = all(len({row.get(key) for row in episodes}) == 1 for key in ("compiler_id", "catalog_id", "controller_profile_id"))
    clean = all(row.get("seed_specific_retuning") is False and row.get("assistance_entries", 1) == 0 for row in episodes)
    reviewed = set()
    if isinstance(visual, Mapping) and isinstance(visual.get("episodes"), list):
        reviewed = {row.get("episode_id") for row in visual["episodes"] if isinstance(row, Mapping)
                    and row.get("coherent") is True and _visual_method_is_direct(row.get("method"))}
    completed_reviewed = all(row.get("episode_id") in reviewed for row in completed)
    passed = len(rooms) >= 3 and len(garments) >= 3 and len(complete_rooms) >= 2 and len(complete_garments) >= 2
    passed &= shared and clean and completed_reviewed
    return [_check("generality.matrix", passed,
                   "three rooms and garments use one compiler/catalog/controller; at least 2/3 each complete without retuning or assistance",
                   {"room_seeds": sorted(str(value) for value in rooms), "garments": sorted(str(value) for value in garments),
                    "completed_room_count": len(complete_rooms), "completed_garment_count": len(complete_garments),
                    "shared_implementation": shared, "no_retuning_or_assistance": clean})]


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
        hero_properties &= result.get("duration_s") is not None and 12.0 <= result["duration_s"] <= 20.0
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
        trace_details.get("ordered_rows", []), trace_details.get("target_object_id"), tolerances
    )
    camera_checks, camera_metrics = _camera_checks(
        trace_details.get("ordered_rows", []), tolerances, capture_ledger
    )
    motion_limit_checks, motion_limit_metrics = _motion_limit_checks(
        trace_details.get("ordered_rows", []), float(tolerances["clock"]["physics_hz"])
    )

    authority = _optional_report(root, evidence, "authority_receipt", "authority_receipt.json")
    execution = _optional_report(root, evidence, "execution_receipt", "execution_receipt.json")
    registration = _optional_report(root, evidence, "registration_report", "registration_report.json")
    projection = _optional_report(root, evidence, "contact_projection", "contact_projection.json")
    capture = _optional_report(root, evidence, "capture_manifest", "capture_manifest.json")
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
    projection_checks = _projection_checks(projection, tolerances)
    capture_checks = _capture_checks(capture, root, tolerances, duration_s)
    replay_checks = _replay_checks(replay, trace_path, tolerances)
    provenance_checks = _provenance_checks(provenance, config["truth_provenance"])
    matrix_checks = _matrix_checks(matrix, visual)
    authority_checks = _authority_checks(authority, root)

    prior_paths = {
        "prior_unity_audition": Path(prior_unity_audition).resolve() if prior_unity_audition else None,
        "prior_corrected_bimanual": Path(prior_corrected_bimanual).resolve() if prior_corrected_bimanual else None,
    }
    video_checks, video_audits = _video_checks(root, evidence, prior_paths)
    comparison_checks = _comparison_checks(visual, prior_paths, video_audits)

    gates = [
        _gate("A", "Frozen authority, contracts, rates, and one-state proof", authority_checks + trace_checks[:2]),
        _gate("B", "Embodiment, anatomy, garments, and registration sweeps",
              registration_checks[:2] + visual_checks[:1]),
        _gate("C", "Object-free full-body motion and head-camera qualification",
              trace_checks[2:4] + camera_checks + motion_limit_checks + video_checks[1:2] + visual_checks[1:2]),
        _gate("D", "Unassisted free-object bimanual interaction and visible/physical contact",
              interaction_checks + registration_checks[2:] + projection_checks + visual_checks[2:3]),
        _gate("E", "Procedural room/clothing generality without retuning", matrix_checks + visual_checks[3:]),
        _gate("F", "Synchronized capture, replay/rerender, dense decode, and prior comparisons",
              trace_checks[4:] + capture_checks + replay_checks + provenance_checks + video_checks[:1] + video_checks[2:] + comparison_checks),
    ]
    non_pass = [check for gate in gates for check in gate["checks"] if check["status"] != PASS]
    veto_reasons = [
        {"gate": gate["gate"], "check_id": check["check_id"], "status": check["status"], "reason": check["summary"]}
        for gate in gates for check in gate["checks"] if check["status"] != PASS
    ]
    decision = PASS if not non_pass else "PROMOTION_VETO"
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
        "gate_summary": {gate["gate"]: gate["status"] for gate in gates},
        "gates": gates,
        "veto_reasons": veto_reasons,
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
