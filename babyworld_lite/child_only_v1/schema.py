"""Strict episode schemas with physically separate visible, instrumentation, and oracle roots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .policy import (
    CONSTRUCTION_PROFILE,
    CONTRACT_VERSION,
    SCHEMA_VERSION,
    PolicyViolation,
    canonical_digest,
    canonical_json_bytes,
    reject_forbidden_source_markers,
)


VISIBLE_KEYS = {
    "schema_version",
    "contract_version",
    "profile_label",
    "episode_id",
    "study_instance_id",
    "split",
    "timebase",
    "duration_ns",
    "frame_stream",
    "utterances",
    "head_imu",
    "proprioception",
    "contact_touch",
    "continuous_motor",
}
INSTRUMENTATION_KEYS = {
    "schema_version",
    "contract_version",
    "profile_label",
    "episode_id",
    "object_state_trajectories",
}
ORACLE_KEYS = {
    "schema_version",
    "contract_version",
    "profile_label",
    "episode_id",
    "events",
    "utterance_event_truth",
    "counterfactual_truth",
}
FORBIDDEN_VISIBLE_KEY_PARTS = {
    "action_label",
    "causal",
    "class_label",
    "counterfactual",
    "detector",
    "donor",
    "event_truth",
    "hidden",
    "object_category",
    "oracle",
    "scoring_target",
}


class SchemaViolation(ValueError):
    """Raised when an episode or bundle violates the closed schema."""


def _exact(record: Mapping[str, Any], keys: set[str], context: str) -> None:
    if set(record) != keys:
        raise SchemaViolation(
            f"{context} keys must be exactly {sorted(keys)}; got {sorted(record)}"
        )


def _finite_number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise SchemaViolation(f"{context} must be a finite number")
    return float(value)


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SchemaViolation(f"{context} must be an integer >= {minimum}")
    return value


def _vector(value: Any, length: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise SchemaViolation(f"{context} must contain {length} values")
    return [_finite_number(item, context) for item in value]


def _validate_header(record: Mapping[str, Any], context: str) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise SchemaViolation(f"{context} has the wrong schema version")
    if record.get("contract_version") != CONTRACT_VERSION:
        raise SchemaViolation(f"{context} has the wrong contract version")
    if record.get("profile_label") != CONSTRUCTION_PROFILE:
        raise SchemaViolation(f"{context} fixture lacks the mandatory construction label")
    episode_id = record.get("episode_id")
    if not isinstance(episode_id, str) or not re_safe_id(episode_id):
        raise SchemaViolation(f"{context} episode_id must be opaque and path-safe")


def re_safe_id(value: str) -> bool:
    return bool(value) and all(ch.isalnum() or ch in "-_" for ch in value)


def _validate_timestamps(samples: Sequence[Mapping[str, Any]], duration_ns: int, context: str) -> None:
    timestamps: list[int] = []
    for sample in samples:
        timestamp = _integer(sample.get("timestamp_ns"), f"{context}.timestamp_ns")
        if timestamp > duration_ns:
            raise SchemaViolation(f"{context} timestamp exceeds episode duration")
        timestamps.append(timestamp)
    if timestamps != sorted(set(timestamps)):
        raise SchemaViolation(f"{context} timestamps must be strictly increasing")


def _safe_relative_path(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise SchemaViolation(f"{context} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SchemaViolation(f"{context} escapes its declared root")
    if any(part.startswith(".") for part in path.parts):
        raise SchemaViolation(f"{context} may not contain hidden or traversal components")
    return path


def validate_visible_episode(record: Mapping[str, Any], visible_root: str | Path) -> None:
    """Validate a model-visible episode without opening any oracle artifact."""

    _exact(record, VISIBLE_KEYS, "visible episode")
    _validate_header(record, "visible episode")
    reject_forbidden_source_markers(record)
    serialized_keys = {str(key).lower() for key in _walk_keys(record)}
    if serialized_keys.intersection(FORBIDDEN_VISIBLE_KEY_PARTS):
        raise SchemaViolation("visible episode contains a hidden/oracle/detector key")
    if record["split"] not in {"train", "validation", "test"}:
        raise SchemaViolation("split must be train, validation, or test")
    if record["timebase"] != "MONOTONIC_NANOSECONDS":
        raise SchemaViolation("v1 timebase must be monotonic nanoseconds")
    duration_ns = _integer(record["duration_ns"], "duration_ns", minimum=1)
    root = Path(visible_root).resolve()

    frames = record["frame_stream"]
    if not isinstance(frames, list) or not frames:
        raise SchemaViolation("frame_stream must be nonempty")
    _validate_timestamps(frames, duration_ns, "frame_stream")
    for frame in frames:
        _exact(
            frame,
            {"timestamp_ns", "path", "sha256", "width_px", "height_px", "channels", "color_space", "dtype", "encoding"},
            "frame",
        )
        relpath = _safe_relative_path(frame["path"], "frame.path")
        if any(part.lower() in {"oracle", "hidden_oracle", "instrumentation"} for part in relpath.parts):
            raise SchemaViolation("visible frame path cannot name a hidden artifact root")
        resolved = (root / Path(*relpath.parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise SchemaViolation("frame path escapes visible root") from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise SchemaViolation("frame path must resolve to a regular in-root file")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != frame["sha256"]:
            raise SchemaViolation("frame content digest mismatch")
        if frame["channels"] != 3 or frame["color_space"] != "sRGB" or frame["dtype"] != "uint8":
            raise SchemaViolation("v1 frames must be uint8 three-channel sRGB")
        if frame["encoding"] not in {"PNG", "JPEG"}:
            raise SchemaViolation("v1 frame encoding must be PNG or JPEG")
        with Image.open(resolved) as image:
            image.load()
            if image.mode != "RGB" or image.size != (frame["width_px"], frame["height_px"]):
                raise SchemaViolation("frame metadata differs from actual RGB image")

    utterances = record["utterances"]
    if not isinstance(utterances, list):
        raise SchemaViolation("utterances must be a list")
    seen_utterances: set[str] = set()
    for utterance in utterances:
        _exact(
            utterance,
            {"utterance_id", "onset_ns", "offset_ns", "text", "speaker_role", "language_tag", "valid"},
            "utterance",
        )
        utterance_id = utterance["utterance_id"]
        if not isinstance(utterance_id, str) or utterance_id in seen_utterances:
            raise SchemaViolation("utterance IDs must be unique strings")
        seen_utterances.add(utterance_id)
        onset = _integer(utterance["onset_ns"], "utterance.onset_ns")
        offset = _integer(utterance["offset_ns"], "utterance.offset_ns")
        if onset >= offset or offset > duration_ns:
            raise SchemaViolation("utterance interval must be nonempty and inside the episode")
        if not isinstance(utterance["text"], str) or not utterance["text"].strip():
            raise SchemaViolation("utterance text must be nonempty")
        if utterance["speaker_role"] not in {"CAREGIVER", "CHILD", "OTHER"}:
            raise SchemaViolation("unknown speaker role")
        if not isinstance(utterance["valid"], bool):
            raise SchemaViolation("utterance validity must be boolean")

    imu = record["head_imu"]
    _exact(imu, {"coordinate_frame", "samples"}, "head_imu")
    if imu["coordinate_frame"] != "HEAD_RIGHT_HANDED_XYZ":
        raise SchemaViolation("head IMU coordinate frame must be declared")
    if not isinstance(imu["samples"], list) or not imu["samples"]:
        raise SchemaViolation("head IMU samples must be nonempty")
    _validate_timestamps(imu["samples"], duration_ns, "head_imu")
    for sample in imu["samples"]:
        _exact(sample, {"timestamp_ns", "acceleration_m_s2", "angular_velocity_rad_s", "valid"}, "IMU sample")
        _vector(sample["acceleration_m_s2"], 3, "acceleration_m_s2")
        _vector(sample["angular_velocity_rad_s"], 3, "angular_velocity_rad_s")
        if not isinstance(sample["valid"], bool):
            raise SchemaViolation("IMU validity must be boolean")

    proprio = record["proprioception"]
    _exact(proprio, {"joint_names", "coordinate_frame", "samples"}, "proprioception")
    joint_names = proprio["joint_names"]
    if not isinstance(joint_names, list) or not joint_names or len(set(joint_names)) != len(joint_names):
        raise SchemaViolation("proprioception joint names must be unique and nonempty")
    if not isinstance(proprio["samples"], list) or not proprio["samples"]:
        raise SchemaViolation("proprioception samples must be nonempty")
    _validate_timestamps(proprio["samples"], duration_ns, "proprioception")
    for sample in proprio["samples"]:
        _exact(
            sample,
            {"timestamp_ns", "joint_position_rad", "joint_velocity_rad_s", "end_effector_position_m", "end_effector_quaternion_xyzw", "valid"},
            "proprioception sample",
        )
        _vector(sample["joint_position_rad"], len(joint_names), "joint_position_rad")
        _vector(sample["joint_velocity_rad_s"], len(joint_names), "joint_velocity_rad_s")
        _vector(sample["end_effector_position_m"], 3, "end_effector_position_m")
        quat = _vector(sample["end_effector_quaternion_xyzw"], 4, "end_effector_quaternion_xyzw")
        norm = math.sqrt(sum(value * value for value in quat))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise SchemaViolation("end-effector quaternion must be normalized")
        if not isinstance(sample["valid"], bool):
            raise SchemaViolation("proprioception validity must be boolean")

    touch = record["contact_touch"]
    _exact(touch, {"sensor_names", "coordinate_frame", "samples"}, "contact_touch")
    sensor_names = touch["sensor_names"]
    if not isinstance(sensor_names, list) or not sensor_names or len(set(sensor_names)) != len(sensor_names):
        raise SchemaViolation("touch sensor names must be unique and nonempty")
    if any(any(token in str(name).lower() for token in ("action", "event", "class")) for name in sensor_names):
        raise SchemaViolation("touch channel names cannot encode labels")
    if not isinstance(touch["samples"], list) or not touch["samples"]:
        raise SchemaViolation("touch samples must be nonempty")
    _validate_timestamps(touch["samples"], duration_ns, "contact_touch")
    for sample in touch["samples"]:
        _exact(
            sample,
            {"timestamp_ns", "contact_binary", "normal_force_n", "pressure_pa", "slip_velocity_m_s", "vibration_m_s2", "valid"},
            "touch sample",
        )
        for key in ("contact_binary", "normal_force_n", "pressure_pa", "slip_velocity_m_s", "vibration_m_s2"):
            values = _vector(sample[key], len(sensor_names), key)
            if key == "contact_binary" and any(value not in {0.0, 1.0} for value in values):
                raise SchemaViolation("contact_binary must contain only zero or one")
        if not isinstance(sample["valid"], bool):
            raise SchemaViolation("touch validity must be boolean")

    motor = record["continuous_motor"]
    _exact(motor, {"channel_names", "units", "samples"}, "continuous_motor")
    channel_names = motor["channel_names"]
    units = motor["units"]
    if not isinstance(channel_names, list) or not channel_names or len(channel_names) != len(units):
        raise SchemaViolation("motor channels require one declared unit each")
    if any(not isinstance(unit, str) or not unit for unit in units):
        raise SchemaViolation("motor units must be nonempty strings")
    if any(any(token in str(name).lower() for token in ("action", "event", "label")) for name in channel_names):
        raise SchemaViolation("motor channel names cannot encode categorical labels")
    if not isinstance(motor["samples"], list) or not motor["samples"]:
        raise SchemaViolation("motor samples must be nonempty")
    _validate_timestamps(motor["samples"], duration_ns, "continuous_motor")
    for sample in motor["samples"]:
        _exact(sample, {"timestamp_ns", "values", "valid"}, "motor sample")
        _vector(sample["values"], len(channel_names), "motor values")
        if not isinstance(sample["valid"], bool):
            raise SchemaViolation("motor validity must be boolean")


def validate_instrumentation_episode(record: Mapping[str, Any]) -> None:
    _exact(record, INSTRUMENTATION_KEYS, "instrumentation episode")
    _validate_header(record, "instrumentation episode")
    trajectories = record["object_state_trajectories"]
    if not isinstance(trajectories, list) or not trajectories:
        raise SchemaViolation("object-state trajectories must be nonempty")
    seen: set[str] = set()
    for trajectory in trajectories:
        _exact(trajectory, {"object_id", "samples"}, "object trajectory")
        object_id = trajectory["object_id"]
        if not isinstance(object_id, str) or object_id in seen:
            raise SchemaViolation("object IDs must be unique opaque strings")
        seen.add(object_id)
        samples = trajectory["samples"]
        if not isinstance(samples, list) or not samples:
            raise SchemaViolation("object trajectory samples must be nonempty")
        timestamps: list[int] = []
        for sample in samples:
            _exact(
                sample,
                {"timestamp_ns", "position_m", "quaternion_xyzw", "linear_velocity_m_s", "angular_velocity_rad_s", "visible_fraction"},
                "object state sample",
            )
            timestamps.append(_integer(sample["timestamp_ns"], "object timestamp"))
            _vector(sample["position_m"], 3, "object position")
            quat = _vector(sample["quaternion_xyzw"], 4, "object quaternion")
            if not math.isclose(math.sqrt(sum(value * value for value in quat)), 1.0, abs_tol=1e-4):
                raise SchemaViolation("object quaternion must be normalized")
            _vector(sample["linear_velocity_m_s"], 3, "object linear velocity")
            _vector(sample["angular_velocity_rad_s"], 3, "object angular velocity")
            visible = _finite_number(sample["visible_fraction"], "visible_fraction")
            if not 0.0 <= visible <= 1.0:
                raise SchemaViolation("visible_fraction must be in [0, 1]")
        if timestamps != sorted(set(timestamps)):
            raise SchemaViolation("object timestamps must be strictly increasing")


def validate_oracle_episode(record: Mapping[str, Any]) -> None:
    _exact(record, ORACLE_KEYS, "oracle episode")
    _validate_header(record, "oracle episode")
    events = record["events"]
    if not isinstance(events, list):
        raise SchemaViolation("oracle events must be a list")
    event_ids: set[str] = set()
    for event in events:
        _exact(event, {"event_id", "event_type", "onset_ns", "offset_ns", "participant_object_ids", "causal_parent_event_ids"}, "oracle event")
        if event["event_id"] in event_ids:
            raise SchemaViolation("oracle event IDs must be unique")
        event_ids.add(event["event_id"])
        if _integer(event["onset_ns"], "event onset") >= _integer(event["offset_ns"], "event offset"):
            raise SchemaViolation("oracle event interval must be nonempty")
    truth = record["utterance_event_truth"]
    if not isinstance(truth, list):
        raise SchemaViolation("utterance-event truth must be a list")
    for link in truth:
        _exact(link, {"utterance_id", "target_event_id", "null_reason", "candidate_event_ids"}, "utterance-event link")
        target = link["target_event_id"]
        if target is not None and target not in event_ids:
            raise SchemaViolation("utterance target references an unknown event")
        if target is None and not link["null_reason"]:
            raise SchemaViolation("null utterance-event truth requires a reason")


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SchemaViolation(f"blank JSONL line at {path}:{line_number}")
            records.append(json.loads(line))
    return records


def manifest_for_files(root: str | Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    base = Path(root).resolve()
    unique = sorted(set(relative_paths))
    if len(unique) != len(relative_paths):
        raise SchemaViolation("manifest cannot contain duplicate paths")
    files: list[dict[str, Any]] = []
    for rel in unique:
        relpath = _safe_relative_path(rel, "manifest path")
        path = (base / Path(*relpath.parts)).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise SchemaViolation("manifest path escapes root") from exc
        if not path.is_file() or path.is_symlink():
            raise SchemaViolation("manifest entries must be regular in-root files")
        payload = path.read_bytes()
        files.append({"path": relpath.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
    core = {"manifest_version": "child-only-file-manifest-v1", "files": files}
    return {**core, "manifest_digest": canonical_digest(core)}


def validate_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Validate all three roots while keeping learner-visible loading separate."""

    root = Path(bundle_root).resolve()
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise SchemaViolation("bundle_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _exact(
        manifest,
        {"bundle_version", "contract_version", "profile_label", "roots", "file_manifest", "episode_ids", "canonical_digest"},
        "bundle manifest",
    )
    if manifest["bundle_version"] != "child-only-fixture-bundle-v1" or manifest["contract_version"] != CONTRACT_VERSION:
        raise SchemaViolation("wrong bundle version")
    if manifest["profile_label"] != CONSTRUCTION_PROFILE:
        raise SchemaViolation("bundle lacks construction label")
    roots = manifest["roots"]
    _exact(roots, {"model_visible", "instrumentation", "hidden_oracle"}, "bundle roots")
    root_paths = {name: (root / rel).resolve() for name, rel in roots.items()}
    if len(set(root_paths.values())) != 3:
        raise SchemaViolation("visible, instrumentation, and oracle roots must be distinct")
    for name, path in root_paths.items():
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SchemaViolation(f"{name} root escapes bundle") from exc
        if not path.is_dir() or path.is_symlink():
            raise SchemaViolation(f"{name} root must be a regular directory")

    visible_file = root_paths["model_visible"] / "episodes.jsonl"
    instrumentation_file = root_paths["instrumentation"] / "object_states.jsonl"
    oracle_file = root_paths["hidden_oracle"] / "events.jsonl"
    if len({visible_file.resolve(), instrumentation_file.resolve(), oracle_file.resolve()}) != 3:
        raise SchemaViolation("visible and hidden records must be physically separate files")
    visible_records = _load_jsonl(visible_file)
    instrumentation_records = _load_jsonl(instrumentation_file)
    oracle_records = _load_jsonl(oracle_file)
    for record in visible_records:
        validate_visible_episode(record, root_paths["model_visible"])
    for record in instrumentation_records:
        validate_instrumentation_episode(record)
    for record in oracle_records:
        validate_oracle_episode(record)
    episode_sets = [
        {record["episode_id"] for record in records}
        for records in (visible_records, instrumentation_records, oracle_records)
    ]
    expected_ids = list(manifest["episode_ids"])
    if len(expected_ids) != len(set(expected_ids)) or any(ids != set(expected_ids) for ids in episode_sets):
        raise SchemaViolation("bundle roots do not contain the exact same unique episode inventory")

    listed = manifest["file_manifest"]
    expected_file_manifest = manifest_for_files(root, [item["path"] for item in listed["files"]])
    if listed != expected_file_manifest:
        raise SchemaViolation("bundle file manifest digest or inventory mismatch")
    observed_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    listed_files = [item["path"] for item in listed["files"]]
    if observed_files != listed_files:
        raise SchemaViolation("bundle contains missing or unexpected files")
    core = {key: manifest[key] for key in manifest if key != "canonical_digest"}
    if manifest["canonical_digest"] != canonical_digest(core):
        raise SchemaViolation("bundle canonical digest mismatch")
    return {
        "valid": True,
        "episode_count": len(expected_ids),
        "bundle_digest": manifest["canonical_digest"],
        "profile_label": CONSTRUCTION_PROFILE,
    }


def canonical_jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted(records, key=lambda record: str(record.get("episode_id", "")))
    return b"".join(canonical_json_bytes(record) + b"\n" for record in ordered)
