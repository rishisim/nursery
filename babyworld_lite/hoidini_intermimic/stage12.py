"""Stage 1 task contracts and the robot-free EmbodiedGenV2 scene boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET


TASK_SCHEMA_VERSION = "hoidini-intermimic-task-contract-1.1.0"
MANIFEST_SCHEMA_VERSION = "hoidini-intermimic-scene-manifest-1.0.0"
INSPECTION_SCHEMA_VERSION = "hoidini-intermimic-layout-inspection-1.0.0"
BINDING_SCHEMA_VERSION = "hoidini-intermimic-shared-scene-binding-1.0.0"
UPSTREAM_CATEGORIES = {
    "background",
    "context",
    "manipulated_objs",
    "distractor_objs",
}
ROBOT_CATEGORIES = {"robot"}
KNOWN_ROBOT_NAMES = {"franka", "piper", "ur5"}
EMBODIEDGEN_RELEASE = "v2.0.1"
EMBODIEDGEN_COMMIT = "9b333554254af196bace88c1a171a3bf047fa09c"
NON_BACKGROUND_CATEGORIES = UPSTREAM_CATEGORIES - {"background"}
HUMAN_CONTACT_PARTICIPANTS = {
    "human_hand_left_or_right",
    "human_hand_left",
    "human_hand_right",
    "human_hands",
}
SEED_NAMES = {
    "master",
    "embodiedgen_image",
    "embodiedgen_asset",
    "embodiedgen_layout",
    "interactmove",
    "intermimic",
    "render",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ContractError(ValueError):
    """Raised when a Stage 1 or Stage 2 boundary contract is invalid."""


def _fail(message: str) -> None:
    raise ContractError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"JSON root must be an object: {path}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _number_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        _fail(f"{label} must contain {length} numbers")
    return [_finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _quaternion_xyzw(value: Any, label: str) -> list[float]:
    quat = _number_vector(value, 4, label)
    norm = math.sqrt(sum(item * item for item in quat))
    if norm <= 1e-12:
        _fail(f"{label} cannot be a zero quaternion")
    return [item / norm for item in quat]


def _required_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty string")
    return value


def _required_sha256(value: Any, label: str) -> str:
    digest = _required_string(value, label)
    if SHA256_PATTERN.fullmatch(digest) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _required_revision(value: Any, label: str) -> str:
    revision = _required_string(value, label)
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision) is None:
        _fail(f"{label} must be a lowercase 40- or 64-character revision")
    return revision


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_relation_rows(value: Any, label: str, bindings: dict[str, Any]) -> None:
    if not isinstance(value, list) or not value:
        _fail(f"{label} must be a non-empty list")
    relation_ids: set[str] = set()
    for index, relation_value in enumerate(value):
        relation = _required_mapping(relation_value, f"{label}[{index}]")
        relation_id = _required_string(relation.get("relation_id"), f"{label}[{index}].relation_id")
        if relation_id in relation_ids:
            _fail(f"duplicate relation ID in {label}: {relation_id}")
        relation_ids.add(relation_id)
        if relation.get("child_role") not in bindings or relation.get("parent_role") not in bindings:
            _fail(f"{label}[{index}] must reference declared scene roles")
        if relation.get("spatial_relation") not in {"ON", "INSIDE", "FLOOR", "IN"}:
            _fail(f"{label}[{index}].spatial_relation is invalid")
        _required_string(
            relation.get("activity_object_description"),
            f"{label}[{index}].activity_object_description",
        )


def validate_task_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the explicit Stage 1 contract and return it unchanged."""

    if contract.get("schema_version") != TASK_SCHEMA_VERSION:
        _fail(f"schema_version must be {TASK_SCHEMA_VERSION!r}")
    _required_string(contract.get("task_id"), "task_id")
    _required_string(contract.get("scene_id"), "scene_id")
    _required_string(contract.get("activity_prompt"), "activity_prompt")
    duration = _finite_number(contract.get("duration_seconds"), "duration_seconds")
    if duration <= 0:
        _fail("duration_seconds must be positive")
    frame_rate = contract.get("frame_rate_hz")
    frame_count = contract.get("frame_count")
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, int) or frame_rate <= 0:
        _fail("frame_rate_hz must be a positive integer")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        _fail("frame_count must be a positive integer")
    if not math.isclose(duration * frame_rate, frame_count, abs_tol=1e-9):
        _fail("duration_seconds, frame_rate_hz, and frame_count disagree")
    if contract.get("sampling") != "half_open_[0,duration)":
        _fail("sampling must be 'half_open_[0,duration)'")

    coordinates = _required_mapping(contract.get("coordinate_system"), "coordinate_system")
    expected_coordinates = {
        "handedness": "right",
        "up_axis": "+Z",
        "linear_unit": "meter",
        "mass_unit": "kilogram",
        "angle_unit": "radian",
        "time_unit": "second",
        "quaternion_order": "xyzw",
        "transform_semantics": "T_parent_child",
        "gravity_m_s2": [0.0, 0.0, -9.81],
    }
    if coordinates != expected_coordinates:
        _fail(f"coordinate_system must equal {expected_coordinates}")

    shared = _required_mapping(contract.get("shared_scene_source"), "shared_scene_source")
    canonical_root = Path(_required_string(shared.get("canonical_root"), "shared_scene_source.canonical_root"))
    if not canonical_root.is_absolute() or ".superseded" in canonical_root.parts:
        _fail("shared_scene_source.canonical_root must be an absolute non-superseded path")
    for name in (
        "activity_spec_sha256",
        "handoff_sha256",
        "generation_receipt_sha256",
        "layout_sha256",
        "reference_mesh_sha256",
    ):
        _required_sha256(shared.get(name), f"shared_scene_source.{name}")
    for name in ("handoff_path", "generation_receipt_path", "layout_path"):
        authoritative_path = Path(_required_string(shared.get(name), f"shared_scene_source.{name}"))
        if not authoritative_path.is_absolute():
            _fail(f"shared_scene_source.{name} must be absolute")
        if ".superseded" in authoritative_path.parts:
            _fail(f"shared_scene_source.{name} must not refer to a superseded path")
    if shared.get("embodiedgen_commit") != EMBODIEDGEN_COMMIT:
        _fail(f"shared_scene_source.embodiedgen_commit must be {EMBODIEDGEN_COMMIT}")
    inventory = _required_mapping(shared.get("inventory"), "shared_scene_source.inventory")
    for name in ("file_count", "total_bytes"):
        value = inventory.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail(f"shared_scene_source.inventory.{name} must be a positive integer")
    _required_sha256(inventory.get("canonical_sha256"), "shared_scene_source.inventory.canonical_sha256")
    backend = _required_mapping(shared.get("object_asset_backend"), "shared_scene_source.object_asset_backend")
    if backend.get("name") != "SAM3D":
        _fail("shared_scene_source.object_asset_backend.name must identify the accepted SAM3D package")
    _required_revision(backend.get("source_commit"), "shared_scene_source.object_asset_backend.source_commit")
    _required_revision(backend.get("checkpoint_revision"), "shared_scene_source.object_asset_backend.checkpoint_revision")
    bundle = _required_mapping(shared.get("scene_bundle"), "shared_scene_source.scene_bundle")
    bundle_path = Path(_required_string(bundle.get("path"), "shared_scene_source.scene_bundle.path"))
    if not bundle_path.is_absolute():
        _fail("shared_scene_source.scene_bundle.path must be absolute")
    if ".superseded" in bundle_path.parts:
        _fail("shared_scene_source.scene_bundle.path must not refer to a superseded path")
    _required_sha256(bundle.get("file_sha256"), "shared_scene_source.scene_bundle.file_sha256")
    _required_sha256(bundle.get("semantic_sha256"), "shared_scene_source.scene_bundle.semantic_sha256")
    _required_string(bundle.get("bundle_id"), "shared_scene_source.scene_bundle.bundle_id")
    if shared.get("robot_actor_inserted") is not False:
        _fail("shared_scene_source.robot_actor_inserted must be false")
    if shared.get("source_assets_copied") is not False:
        _fail("shared_scene_source.source_assets_copied must be false")
    forbidden_root = Path(
        _required_string(
            shared.get("superseded_scene_root_forbidden"),
            "shared_scene_source.superseded_scene_root_forbidden",
        )
    )
    if not forbidden_root.is_absolute() or forbidden_root == canonical_root:
        _fail("superseded_scene_root_forbidden must be a distinct absolute path")

    seeds = _required_mapping(contract.get("seeds"), "seeds")
    if set(seeds) != SEED_NAMES:
        _fail(f"seeds must contain exactly {sorted(SEED_NAMES)}")
    for name in sorted(SEED_NAMES):
        value = seeds.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"seeds.{name} must be a non-negative integer")

    hand_use = _required_mapping(contract.get("hand_use"), "hand_use")
    if hand_use != {"mode": "one_hand", "hands": ["right"]}:
        _fail("hand_use must declare the canonical one-hand right protocol")

    bindings = _required_mapping(contract.get("scene_bindings"), "scene_bindings")
    if set(bindings) != {"target", "support", "goal"}:
        _fail("scene_bindings must contain exactly target, support, and goal")
    roles_by_source: dict[str, list[str]] = {}
    source_by_instance_id: dict[str, str] = {}
    for role, expected_body_mode in (
        ("target", "dynamic"),
        ("support", "static"),
        ("goal", "static"),
    ):
        binding = _required_mapping(bindings.get(role), f"scene_bindings.{role}")
        name = _required_string(binding.get("source_name"), f"scene_bindings.{role}.source_name")
        instance_id = _required_string(
            binding.get("canonical_instance_id"),
            f"scene_bindings.{role}.canonical_instance_id",
        )
        if re.fullmatch(r"egv2_[0-9a-f]{24}", instance_id) is None:
            _fail(f"scene_bindings.{role}.canonical_instance_id is invalid")
        prior_source = source_by_instance_id.get(instance_id)
        if prior_source is not None and prior_source != name:
            _fail(
                f"canonical instance ID is reused across distinct sources: "
                f"{prior_source!r} and {name!r}"
            )
        source_by_instance_id[instance_id] = name
        same_as = binding.get("same_instance_as")
        prior_roles = roles_by_source.setdefault(name, [])
        if prior_roles:
            if same_as not in prior_roles:
                _fail(f"scene binding source name is reused without a valid same_instance_as alias: {name}")
            original = bindings[same_as]
            if (
                original.get("canonical_instance_id") != instance_id
                or original.get("body_mode") != binding.get("body_mode")
            ):
                _fail(f"scene_bindings.{role} alias does not match scene_bindings.{same_as}")
        elif same_as is not None:
            _fail(f"scene_bindings.{role}.same_instance_as must point to an earlier role")
        prior_roles.append(role)
        categories = binding.get("allowed_upstream_categories")
        if not isinstance(categories, list) or not categories:
            _fail(f"scene_bindings.{role}.allowed_upstream_categories must be a non-empty list")
        if any(category not in NON_BACKGROUND_CATEGORIES for category in categories):
            _fail(f"scene_bindings.{role} contains an unsupported upstream category")
        if binding.get("body_mode") != expected_body_mode:
            _fail(f"scene_bindings.{role}.body_mode must be {expected_body_mode!r}")

    human = _required_mapping(contract.get("initial_human_state"), "initial_human_state")
    if human.get("body_model") != "adult_neutral_smplx_beta0":
        _fail("initial_human_state.body_model must be adult_neutral_smplx_beta0")
    expected_human_fields = {
        "source_body_model": "SMPL-X",
        "morphology": "adult",
        "gender": "neutral",
        "source_initial_stance": "standing_neutral",
    }
    for name, expected in expected_human_fields.items():
        if human.get(name) != expected:
            _fail(f"initial_human_state.{name} must be {expected!r}")
    betas = _number_vector(human.get("betas"), 10, "initial_human_state.betas")
    if any(value != 0.0 for value in betas):
        _fail("initial_human_state.betas must contain ten zero values")
    placement = _required_mapping(human.get("placement"), "initial_human_state.placement")
    if placement.get("method") != "target_relative":
        _fail("initial_human_state.placement.method must be target_relative")
    if placement.get("reference_role") not in bindings:
        _fail("initial_human_state.placement.reference_role must name a scene binding")
    _number_vector(placement.get("translation_offset_world_m"), 3, "initial_human_state.placement.translation_offset_world_m")
    _quaternion_xyzw(placement.get("orientation_xyzw"), "initial_human_state.placement.orientation_xyzw")
    if not isinstance(placement.get("face_reference"), bool):
        _fail("initial_human_state.placement.face_reference must be boolean")
    _required_string(human.get("joint_pose"), "initial_human_state.joint_pose")

    initial_objects = _required_mapping(contract.get("initial_object_states"), "initial_object_states")
    for role in bindings:
        state = _required_mapping(initial_objects.get(role), f"initial_object_states.{role}")
        if state.get("pose_source") != "embodiedgen_layout":
            _fail(f"initial_object_states.{role}.pose_source must be embodiedgen_layout")

    _validate_relation_rows(contract.get("required_initial_relations"), "required_initial_relations", bindings)
    _validate_relation_rows(contract.get("required_final_relations"), "required_final_relations", bindings)

    phases = contract.get("action_phases")
    if not isinstance(phases, list) or not phases:
        _fail("action_phases must be a non-empty list")
    cursor = 0.0
    frame_cursor = 0
    phase_names: set[str] = set()
    for index, phase_value in enumerate(phases):
        phase = _required_mapping(phase_value, f"action_phases[{index}]")
        name = _required_string(phase.get("id"), f"action_phases[{index}].id")
        if name in phase_names:
            _fail(f"duplicate action phase name: {name}")
        phase_names.add(name)
        start = _finite_number(phase.get("start_seconds"), f"action_phases[{index}].start_seconds")
        end = _finite_number(phase.get("end_seconds"), f"action_phases[{index}].end_seconds")
        start_frame = phase.get("start_frame")
        end_frame = phase.get("end_frame_exclusive")
        if (
            isinstance(start_frame, bool)
            or not isinstance(start_frame, int)
            or isinstance(end_frame, bool)
            or not isinstance(end_frame, int)
        ):
            _fail("action phase frame bounds must be integers")
        if not math.isclose(start, cursor, abs_tol=1e-9) or end <= start:
            _fail("action_phases must be contiguous, ordered, and have positive duration")
        if start_frame != frame_cursor or end_frame <= start_frame:
            _fail("action_phases frame bounds must be contiguous, ordered, and positive")
        if not math.isclose(start, start_frame / frame_rate, abs_tol=1e-9) or not math.isclose(
            end, end_frame / frame_rate, abs_tol=1e-9
        ):
            _fail("action phase seconds and frame bounds disagree")
        cursor = end
        frame_cursor = end_frame
        contact_hands = phase.get("expected_target_contact_hands")
        if not isinstance(contact_hands, list) or any(hand not in {"left", "right"} for hand in contact_hands):
            _fail(f"action_phases[{index}].expected_target_contact_hands is invalid")
        _required_string(phase.get("intent"), f"action_phases[{index}].intent")
    if not math.isclose(cursor, duration, abs_tol=1e-9) or frame_cursor != frame_count:
        _fail("action_phases must cover exactly duration_seconds")

    windows = contract.get("contact_windows")
    if not isinstance(windows, list) or not windows:
        _fail("contact_windows must be a non-empty list")
    contact_ids: set[str] = set()
    windows_by_id: dict[str, dict[str, Any]] = {}
    allowed_participants = set(bindings) | HUMAN_CONTACT_PARTICIPANTS
    for index, window_value in enumerate(windows):
        window = _required_mapping(window_value, f"contact_windows[{index}]")
        contact_id = _required_string(window.get("contact_id"), f"contact_windows[{index}].contact_id")
        if contact_id in contact_ids:
            _fail(f"duplicate contact ID: {contact_id}")
        contact_ids.add(contact_id)
        windows_by_id[contact_id] = window
        start = _finite_number(window.get("start_seconds"), f"contact_windows[{index}].start_seconds")
        end = _finite_number(window.get("end_seconds"), f"contact_windows[{index}].end_seconds")
        start_frame = window.get("start_frame")
        end_frame = window.get("end_frame_exclusive")
        if (
            isinstance(start_frame, bool)
            or not isinstance(start_frame, int)
            or isinstance(end_frame, bool)
            or not isinstance(end_frame, int)
            or start_frame < 0
            or end_frame <= start_frame
            or end_frame > frame_count
        ):
            _fail(f"contact_windows[{index}] has invalid frame bounds")
        if not math.isclose(start, start_frame / frame_rate, abs_tol=1e-9) or not math.isclose(
            end, end_frame / frame_rate, abs_tol=1e-9
        ):
            _fail(f"contact_windows[{index}] seconds and frame bounds disagree")
        if start < 0 or end <= start or end > duration:
            _fail(f"contact_windows[{index}] lies outside the task duration")
        if window.get("requirement") not in {"required", "allowed", "forbidden"}:
            _fail(f"contact_windows[{index}].requirement is invalid")
        participants = window.get("participants")
        if not isinstance(participants, list) or len(participants) < 2:
            _fail(f"contact_windows[{index}].participants must name at least two participants")
        if any(participant not in allowed_participants for participant in participants):
            _fail(f"contact_windows[{index}] contains an unknown participant")
    right_contact_phases = [
        phase for phase in phases if phase["expected_target_contact_hands"] == ["right"]
    ]
    if not right_contact_phases:
        _fail("action_phases must contain a right-hand target-contact interval")
    right_start = right_contact_phases[0]["start_frame"]
    right_end = right_contact_phases[-1]["end_frame_exclusive"]
    expected_windows = {
        "mug_on_table_initially": (0, right_start, "required", ["target", "support"]),
        "right_hand_grasps_mug": (right_start, right_end, "required", ["human_hand_right", "target"]),
        "mug_on_table_at_end": (right_end, frame_count, "required", ["target", "goal"]),
        "hands_clear_after_release": (right_end, frame_count, "forbidden", ["human_hands", "target"]),
    }
    if set(windows_by_id) != set(expected_windows):
        _fail("contact_windows must contain exactly the canonical contact-window IDs")
    for contact_id, (start_frame, end_frame, requirement, participants) in expected_windows.items():
        window = windows_by_id[contact_id]
        expected = {
            "start_frame": start_frame,
            "end_frame_exclusive": end_frame,
            "requirement": requirement,
            "participants": participants,
        }
        for name, value in expected.items():
            _expect_equal(window.get(name), value, f"contact_windows.{contact_id}.{name}")

    metrics = contract.get("acceptance_metrics")
    if not isinstance(metrics, list) or not metrics:
        _fail("acceptance_metrics must be a non-empty list")
    metric_ids: set[str] = set()
    for index, metric_value in enumerate(metrics):
        metric = _required_mapping(metric_value, f"acceptance_metrics[{index}]")
        metric_id = _required_string(metric.get("id"), f"acceptance_metrics[{index}].id")
        if metric_id in metric_ids:
            _fail(f"duplicate acceptance metric ID: {metric_id}")
        metric_ids.add(metric_id)
        _required_string(metric.get("metric"), f"acceptance_metrics[{index}].metric")
        if metric.get("operator") not in {"eq", "gte", "lte"}:
            _fail(f"acceptance_metrics[{index}].operator is invalid")
        if metric.get("evaluation_window") != "full_episode":
            _fail(f"acceptance_metrics[{index}].evaluation_window must be full_episode")
        if metric.get("threshold") is None:
            _fail(f"acceptance_metrics[{index}].threshold is required")
    diagnostics = contract.get("non_acceptance_diagnostics")
    if not isinstance(diagnostics, list):
        _fail("non_acceptance_diagnostics must be a list")
    for index, diagnostic_value in enumerate(diagnostics):
        diagnostic = _required_mapping(diagnostic_value, f"non_acceptance_diagnostics[{index}]")
        _required_string(diagnostic.get("id"), f"non_acceptance_diagnostics[{index}].id")
        _required_string(diagnostic.get("description"), f"non_acceptance_diagnostics[{index}].description")
    return contract


def load_task_contract(path: Path) -> dict[str, Any]:
    return validate_task_contract(_load_json(path))


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        _fail(f"cannot form a stable ID from source name {value!r}")
    return slug


def _safe_path(root: Path, raw_path: str, label: str) -> Path:
    _required_string(raw_path, label)
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        _fail(f"{label} escapes the scene root: {raw_path}")
    if not candidate.exists():
        _fail(f"{label} does not exist: {raw_path}")
    return candidate


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_floats(raw: str | None, length: int, default: list[float], label: str) -> list[float]:
    if raw is None:
        return list(default)
    parts = raw.split()
    if len(parts) != length:
        _fail(f"{label} must contain {length} numbers")
    return [_float_text(part, label) for part in parts]


def _float_text(raw: str, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be a number") from exc
    return _finite_number(value, label)


def _parse_origin(parent: ET.Element | None, label: str) -> dict[str, list[float]]:
    origin = parent.find("origin") if parent is not None else None
    return {
        "translation_m": _parse_floats(origin.get("xyz") if origin is not None else None, 3, [0.0, 0.0, 0.0], f"{label}.xyz"),
        "rotation_rpy_rad": _parse_floats(origin.get("rpy") if origin is not None else None, 3, [0.0, 0.0, 0.0], f"{label}.rpy"),
    }


def _mesh_records(scene_root: Path, urdf_path: Path, link: ET.Element, kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, node in enumerate(link.findall(kind)):
        mesh = node.find("geometry/mesh")
        if mesh is None:
            _fail(f"{urdf_path}: {kind}[{index}] must use a mesh geometry")
        filename = mesh.get("filename")
        if filename is None:
            _fail(f"{urdf_path}: {kind}[{index}] mesh has no filename")
        mesh_path = _safe_path(urdf_path.parent, filename, f"{kind} mesh")
        if not mesh_path.is_file():
            _fail(f"{kind} mesh must be a file: {filename}")
        try:
            relative = _relative_path(scene_root, mesh_path)
        except ValueError:
            _fail(f"{kind} mesh escapes the scene root: {filename}")
        records.append(
            {
                "path": relative,
                "sha256": _sha256(mesh_path),
                "scale": _parse_floats(mesh.get("scale"), 3, [1.0, 1.0, 1.0], f"{kind}[{index}].scale"),
                "local_transform": _parse_origin(node, f"{kind}[{index}].origin"),
            }
        )
    if not records:
        _fail(f"{urdf_path}: at least one {kind} mesh is required")
    return records


def _optional_float(element: ET.Element | None, label: str) -> float | None:
    if element is None:
        return None
    raw = element.get("value") if element.get("value") is not None else element.text
    if raw is None:
        _fail(f"{label} is empty")
    return _float_text(raw, label)


def _physics_value(root: ET.Element, link: ET.Element, tag: str) -> float | None:
    element = link.find(f"collision/gazebo/{tag}")
    if element is None:
        element = link.find(f"gazebo/{tag}")
    if element is None:
        link_name = link.get("name")
        for gazebo in root.findall("gazebo"):
            if gazebo.get("reference") in {None, link_name}:
                element = gazebo.find(tag)
                if element is not None:
                    break
    return _optional_float(element, tag)


def _inertia_record(link: ET.Element) -> dict[str, Any]:
    inertial = link.find("inertial")
    inertia = inertial.find("inertia") if inertial is not None else None
    origin = _parse_origin(inertial, "inertial.origin") if inertial is not None else None
    if inertia is None:
        return {
            "available": False,
            "source": "missing",
            "center_of_mass_m": origin["translation_m"] if origin else None,
            "matrix_kg_m2": None,
            "raw_upstream_matrix_kg_m2": None,
        }
    names = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    values = {name: _float_text(inertia.get(name, "nan"), f"inertia.{name}") for name in names}
    matrix = [
        [values["ixx"], values["ixy"], values["ixz"]],
        [values["ixy"], values["iyy"], values["iyz"]],
        [values["ixz"], values["iyz"], values["izz"]],
    ]
    is_template_placeholder = (
        values["ixx"] == values["iyy"] == values["izz"] == 1.0
        and values["ixy"] == values["ixz"] == values["iyz"] == 0.0
        and link.find("extra_info") is not None
    )
    determinant = (
        values["ixx"] * (values["iyy"] * values["izz"] - values["iyz"] ** 2)
        - values["ixy"] * (values["ixy"] * values["izz"] - values["iyz"] * values["ixz"])
        + values["ixz"] * (values["ixy"] * values["iyz"] - values["iyy"] * values["ixz"])
    )
    positive_definite = (
        values["ixx"] > 0
        and values["ixx"] * values["iyy"] - values["ixy"] ** 2 > 0
        and determinant > 0
    )
    if not positive_definite:
        _fail("URDF inertia matrix must be positive definite")
    half_trace = (values["ixx"] + values["iyy"] + values["izz"]) / 2.0
    second_moment = [
        [half_trace - values["ixx"], -values["ixy"], -values["ixz"]],
        [-values["ixy"], half_trace - values["iyy"], -values["iyz"]],
        [-values["ixz"], -values["iyz"], half_trace - values["izz"]],
    ]
    tolerance = 1e-12
    principal_two = (
        second_moment[0][0] * second_moment[1][1] - second_moment[0][1] ** 2,
        second_moment[0][0] * second_moment[2][2] - second_moment[0][2] ** 2,
        second_moment[1][1] * second_moment[2][2] - second_moment[1][2] ** 2,
    )
    second_moment_det = (
        second_moment[0][0] * (second_moment[1][1] * second_moment[2][2] - second_moment[1][2] ** 2)
        - second_moment[0][1] * (second_moment[0][1] * second_moment[2][2] - second_moment[1][2] * second_moment[0][2])
        + second_moment[0][2] * (second_moment[0][1] * second_moment[1][2] - second_moment[1][1] * second_moment[0][2])
    )
    if (
        any(second_moment[index][index] < -tolerance for index in range(3))
        or any(value < -tolerance for value in principal_two)
        or second_moment_det < -tolerance
    ):
        _fail("URDF inertia violates rigid-body principal-moment triangle inequalities")
    return {
        "available": not is_template_placeholder,
        "source": "urdf_inertial" if not is_template_placeholder else "upstream_template_placeholder",
        "center_of_mass_m": origin["translation_m"] if origin else [0.0, 0.0, 0.0],
        "matrix_kg_m2": matrix if not is_template_placeholder else None,
        "raw_upstream_matrix_kg_m2": matrix,
    }


def _parse_urdf(scene_root: Path, urdf_path: Path, source_name: str) -> dict[str, Any]:
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ContractError(f"cannot parse URDF {urdf_path}: {exc}") from exc
    links = root.findall("link")
    if len(links) != 1:
        _fail(f"Stage 2 supports exactly one URDF link per rigid object: {urdf_path}")
    link = links[0]
    robot_name = root.get("name")
    link_name = link.get("name")
    expected_identity = _safe_slug(source_name)
    if _safe_slug(robot_name or "") != expected_identity or _safe_slug(link_name or "") != expected_identity:
        _fail(f"URDF robot/link identity does not match source object {source_name!r}: {urdf_path}")
    mass = _optional_float(link.find("inertial/mass"), "mass")
    if mass is not None and mass <= 0:
        _fail(f"mass must be positive: {urdf_path}")
    static_friction = _physics_value(root, link, "mu1")
    dynamic_friction = _physics_value(root, link, "mu2")
    for label, value in (("static friction", static_friction), ("dynamic friction", dynamic_friction)):
        if value is not None and value < 0:
            _fail(f"{label} cannot be negative: {urdf_path}")
    restitution = _physics_value(root, link, "restitution")
    if restitution is None:
        restitution = _physics_value(root, link, "restitution_coefficient")
    if restitution is not None and not 0 <= restitution <= 1:
        _fail(f"restitution must be between zero and one: {urdf_path}")
    extra_info = link.find("extra_info")
    min_mass = _optional_float(extra_info.find("min_mass") if extra_info is not None else None, "extra_info.min_mass")
    max_mass = _optional_float(extra_info.find("max_mass") if extra_info is not None else None, "extra_info.max_mass")
    real_height = _optional_float(extra_info.find("real_height") if extra_info is not None else None, "extra_info.real_height")
    if min_mass is not None and max_mass is not None and min_mass > max_mass:
        _fail(f"extra_info mass range is reversed: {urdf_path}")
    has_embodiedgen_estimate = min_mass is not None and max_mass is not None
    return {
        "urdf": {
            "path": _relative_path(scene_root, urdf_path),
            "sha256": _sha256(urdf_path),
            "link_name": link.get("name"),
        },
        "visual_meshes": _mesh_records(scene_root, urdf_path, link, "visual"),
        "collision_meshes": _mesh_records(scene_root, urdf_path, link, "collision"),
        "physical_properties": {
            "mass": {
                "available": mass is not None,
                "value_kg": mass,
                "source": "embodiedgen_vlm_estimate_midpoint_in_urdf" if mass is not None and has_embodiedgen_estimate else ("urdf_inertial" if mass is not None else "missing"),
                "upstream_estimate_range_kg": [min_mass, max_mass] if has_embodiedgen_estimate else None,
            },
            "inertia": _inertia_record(link),
            "friction": {
                "available": static_friction is not None and dynamic_friction is not None,
                "static": static_friction,
                "dynamic": dynamic_friction,
                "source": "embodiedgen_vlm_estimate_urdf_collision_gazebo" if static_friction is not None or dynamic_friction is not None else "missing",
                "estimation_surface_pair": "object_material_relative_to_rubber" if static_friction is not None or dynamic_friction is not None else None,
            },
            "restitution": {
                "available": restitution is not None,
                "value": restitution,
                "source": "urdf" if restitution is not None else "missing_not_embodiedgen_default",
            },
            "upstream_real_height_m": real_height,
        },
    }


def _resolve_urdf(scene_root: Path, source_name: str, asset_reference: Any) -> Path:
    reference = _required_string(asset_reference, f"assets.{source_name}")
    asset_path = _safe_path(scene_root, reference, f"assets.{source_name}")
    if asset_path.is_file():
        if asset_path.suffix.lower() != ".urdf":
            _fail(f"asset file must be a URDF: {asset_path}")
        return asset_path
    expected = asset_path / f"{source_name.replace(' ', '_')}.urdf"
    if not expected.is_file():
        _fail(f"expected EmbodiedGen URDF is missing: {_relative_path(scene_root, expected)}")
    return expected.resolve()


def _tree_edges(layout: dict[str, Any]) -> dict[str, tuple[str, str]]:
    edges: dict[str, tuple[str, str]] = {}
    for parent, raw_children in layout["tree"].items():
        _required_string(parent, "tree parent")
        if not isinstance(raw_children, list):
            _fail(f"tree.{parent} must be a list")
        for index, raw_edge in enumerate(raw_children):
            if not isinstance(raw_edge, list) or len(raw_edge) != 2:
                _fail(f"tree.{parent}[{index}] must be [child, spatial_relation]")
            child = _required_string(raw_edge[0], f"tree.{parent}[{index}].child")
            relation = raw_edge[1]
            if relation not in {"ON", "INSIDE", "FLOOR", "IN"}:
                _fail(f"tree.{parent}[{index}] has an unsupported spatial relation")
            if child in edges:
                _fail(f"tree child appears under multiple parents: {child}")
            edges[child] = (parent, relation)
    return edges


def _validate_layout(layout: dict[str, Any]) -> dict[str, tuple[str, str]]:
    required = ("tree", "relation", "objs_desc", "objs_mapping", "assets", "quality", "position")
    missing = [name for name in required if name not in layout]
    if missing:
        _fail(f"layout.json is missing fields: {', '.join(missing)}")
    for name in required:
        if not isinstance(layout[name], dict):
            _fail(f"layout.{name} must be an object")
    mappings = layout["objs_mapping"]
    slug_to_name: dict[str, str] = {}
    for source_name, category in mappings.items():
        _required_string(source_name, "objs_mapping source name")
        if category not in UPSTREAM_CATEGORIES | ROBOT_CATEGORIES:
            _fail(f"unsupported EmbodiedGen category {category!r} for {source_name!r}")
        slug = _safe_slug(source_name)
        if slug in slug_to_name and slug_to_name[slug] != source_name:
            _fail(f"ambiguous normalized source names: {slug_to_name[slug]!r} and {source_name!r}")
        slug_to_name[slug] = source_name
    relation = layout["relation"]
    background = _required_string(relation.get("background"), "relation.background")
    context = _required_string(relation.get("context"), "relation.context")
    manipulated = relation.get("manipulated_objs")
    distractors = relation.get("distractor_objs")
    if not isinstance(manipulated, list) or not all(isinstance(name, str) for name in manipulated):
        _fail("relation.manipulated_objs must be a list of names")
    if not isinstance(distractors, list) or not all(isinstance(name, str) for name in distractors):
        _fail("relation.distractor_objs must be a list of names")
    declared = [background, context, *manipulated, *distractors]
    if len(declared) != len(set(declared)):
        _fail("relation object categories must not overlap")
    expected_categories = {
        background: "background",
        context: "context",
        **{name: "manipulated_objs" for name in manipulated},
        **{name: "distractor_objs" for name in distractors},
    }
    non_robot_mappings = {name: category for name, category in mappings.items() if category not in ROBOT_CATEGORIES}
    if non_robot_mappings != expected_categories:
        _fail("relation categories and objs_mapping disagree")
    for source_name in non_robot_mappings:
        if source_name not in layout["position"]:
            _fail(f"position is missing for {source_name!r}")
        if source_name not in layout["assets"]:
            _fail(f"asset mapping is missing for {source_name!r}")
    return _tree_edges(layout)


def _pose(layout: dict[str, Any], source_name: str) -> dict[str, list[float]]:
    raw = layout["position"].get(source_name)
    values = _number_vector(raw, 7, f"position.{source_name}")
    return {
        "translation_m": values[:3],
        "orientation_xyzw": _quaternion_xyzw(values[3:], f"position.{source_name}.quaternion"),
    }


def resolve_scene_manifest(
    task_contract_path: Path,
    layout_path: Path,
) -> dict[str, Any]:
    """Inspect a task/layout boundary; this alone is not a completed Stage 2 binding."""

    task_contract_path = task_contract_path.resolve()
    layout_path = layout_path.resolve()
    task = load_task_contract(task_contract_path)
    layout = _load_json(layout_path)
    tree_edges = _validate_layout(layout)
    scene_root = layout_path.parent

    relation = layout["relation"]
    robot_names = set(KNOWN_ROBOT_NAMES)
    if isinstance(relation.get("robot"), str):
        robot_names.add(relation["robot"])
    robot_names.update(name for name, category in layout["objs_mapping"].items() if category in ROBOT_CATEGORIES)
    stripped_robot_names = sorted(
        name
        for name in robot_names
        if name in layout["position"] or name in layout["assets"] or name in layout["objs_mapping"] or name == relation.get("robot")
    )

    bindings = task["scene_bindings"]
    roles_by_source: dict[str, list[str]] = {}
    for role, binding in bindings.items():
        roles_by_source.setdefault(binding["source_name"], []).append(role)
    for role, binding in bindings.items():
        source_name = binding["source_name"]
        category = layout["objs_mapping"].get(source_name)
        if category is None:
            _fail(f"task {role} object is absent from layout.objs_mapping: {source_name}")
        if category not in binding["allowed_upstream_categories"]:
            _fail(f"task {role} object {source_name!r} has disallowed category {category!r}")

    resolved_initial_relations: list[dict[str, str]] = []
    for required_relation in task["required_initial_relations"]:
        child_role = required_relation["child_role"]
        parent_role = required_relation["parent_role"]
        child_name = bindings[child_role]["source_name"]
        parent_name = bindings[parent_role]["source_name"]
        actual = tree_edges.get(child_name)
        expected = (parent_name, required_relation["spatial_relation"])
        if actual != expected:
            _fail(
                f"required initial relation {required_relation['relation_id']!r} is not present: "
                f"expected {child_name} {expected[1]} {parent_name}, got {actual}"
            )
        resolved_initial_relations.append(
            {
                "relation_id": required_relation["relation_id"],
                "child_role": child_role,
                "parent_role": parent_role,
                "spatial_relation": expected[1],
            }
        )

    scene_id = _safe_slug(_required_string(task.get("scene_id"), "scene_id"))
    instances: list[dict[str, Any]] = []
    instance_id_by_source: dict[str, str] = {}
    background_name = relation.get("background")
    for source_name in sorted(layout["objs_mapping"]):
        category = layout["objs_mapping"][source_name]
        if category == "background" or category in ROBOT_CATEGORIES or source_name in robot_names:
            continue
        if source_name not in layout["assets"]:
            _fail(f"asset mapping is missing for {source_name!r}")
        instance_id = f"scene/{scene_id}/instance/{_safe_slug(source_name)}"
        instance_id_by_source[source_name] = instance_id
        roles = roles_by_source.get(source_name, [])
        if roles:
            body_modes = {bindings[role]["body_mode"] for role in roles}
            if len(body_modes) != 1:
                _fail(f"aliased task roles disagree on body mode for {source_name!r}")
            body_mode = body_modes.pop()
            body_mode_provenance = ",".join(
                f"task_contract.scene_bindings.{role}.body_mode" for role in roles
            )
        else:
            body_mode = "static" if category == "context" else "dynamic"
            body_mode_provenance = f"local_policy_from_upstream_category:{category}"
        urdf_path = _resolve_urdf(scene_root, source_name, layout["assets"][source_name])
        instances.append(
            {
                "instance_id": instance_id,
                "source_name": source_name,
                "upstream_category": category,
                "scientific_roles": roles,
                "body_mode": body_mode,
                "body_mode_provenance": body_mode_provenance,
                "upstream_quality": layout["quality"].get(source_name),
                "world_pose": _pose(layout, source_name),
                **_parse_urdf(scene_root, urdf_path, source_name),
            }
        )

    background: dict[str, Any] | None = None
    if isinstance(background_name, str):
        if layout["objs_mapping"].get(background_name) != "background":
            _fail("relation.background must map to the background category")
        asset_reference = layout["assets"].get(background_name)
        if asset_reference is None:
            _fail(f"background asset mapping is missing for {background_name!r}")
        background_dir = _safe_path(scene_root, asset_reference, f"assets.{background_name}")
        visual_references = []
        for filename, representation in (("gs_model.ply", "gaussian_splat"), ("mesh_model.ply", "color_mesh")):
            path = background_dir / filename
            if path.is_file():
                visual_references.append(
                    {"path": _relative_path(scene_root, path), "sha256": _sha256(path), "representation": representation}
                )
        if not visual_references:
            _fail("EmbodiedGen background must contain gs_model.ply or mesh_model.ply")
        background = {
            "source_name": background_name,
            "world_pose": _pose(layout, background_name),
            "visual_references": visual_references,
            "collision": {
                "available": False,
                "path": None,
                "reason": "EmbodiedGen does not designate its background mesh as collision geometry",
            },
        }

    role_bindings = {role: instance_id_by_source[binding["source_name"]] for role, binding in bindings.items()}
    placement = task["initial_human_state"]["placement"]
    reference_role = placement["reference_role"]
    reference_name = bindings[reference_role]["source_name"]
    reference_translation = _pose(layout, reference_name)["translation_m"]
    offset = placement["translation_offset_world_m"]
    human_translation = [reference_translation[index] + float(offset[index]) for index in range(3)]

    return {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "boundary_status": "diagnostic_only_not_stage2",
        "scene_id": scene_id,
        "task": {
            "task_id": task["task_id"],
            "contract_path": task_contract_path.name,
            "contract_sha256": _sha256(task_contract_path),
        },
        "source": {
            "system": "EmbodiedGenV2",
            "assumed_release": EMBODIEDGEN_RELEASE,
            "assumed_commit": EMBODIEDGEN_COMMIT,
            "layout_pose_contract": "right-handed Z-up meters with XYZW quaternions",
            "layout_path": layout_path.name,
            "layout_sha256": _sha256(layout_path),
            "generation_execution_verified": False,
            "validation_scope": "artifact_boundary_only",
        },
        "coordinate_system": task["coordinate_system"],
        "robot_policy": {
            "resolved_robot_instances": 0,
            "stripped_upstream_robot_names": stripped_robot_names,
            "rule": "upstream robot metadata is ignored; no robot asset or instance is preserved",
        },
        "role_bindings": role_bindings,
        "required_initial_relations": [
            {
                **relation,
                "child_instance_id": role_bindings[relation["child_role"]],
                "parent_instance_id": role_bindings[relation["parent_role"]],
            }
            for relation in resolved_initial_relations
        ],
        "initial_human_state": {
            "body_model": task["initial_human_state"]["body_model"],
            "joint_pose": task["initial_human_state"]["joint_pose"],
            "world_pose": {
                "translation_m": human_translation,
                "orientation_xyzw": _quaternion_xyzw(placement["orientation_xyzw"], "initial human orientation"),
            },
            "placement_provenance": {
                "reference_instance_id": role_bindings[reference_role],
                "translation_offset_world_m": offset,
                "reachability_checked": False,
            },
        },
        "background": background,
        "instances": instances,
        "limitations": [
            "A naked layout resolution does not verify scene generation; shared-scene binding must verify the handoff and inventory.",
            "Background visual meshes are not treated as collision proxies without an explicit validated designation.",
            "Missing restitution and upstream placeholder inertia remain unavailable rather than receiving invented values.",
            "Human reachability is not proactively evaluated at Stages 1-2.",
        ],
    }


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _expect_vector_close(actual: Any, expected: Any, label: str) -> None:
    if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
        _fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")
    if any(not math.isclose(float(a), float(b), abs_tol=1e-9) for a, b in zip(actual, expected)):
        _fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _expect_quaternion_equivalent(actual: Any, expected: Any, label: str) -> None:
    actual_quat = _quaternion_xyzw(actual, f"{label}.actual")
    expected_quat = _quaternion_xyzw(expected, f"{label}.expected")
    dot = sum(a * b for a, b in zip(actual_quat, expected_quat))
    if not math.isclose(abs(dot), 1.0, abs_tol=1e-9):
        _fail(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _verified_inventory(scene_root: Path, handoff_inventory: dict[str, Any]) -> dict[str, Any]:
    rows = handoff_inventory.get("files")
    if not isinstance(rows, list) or not rows:
        _fail("handoff.inventory.files must be a non-empty list")
    expected_paths: list[str] = []
    expected_bytes = 0
    for index, row_value in enumerate(rows):
        row = _required_mapping(row_value, f"handoff.inventory.files[{index}]")
        raw_path = _required_string(row.get("path"), f"handoff.inventory.files[{index}].path")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts or ".superseded" in relative.parts:
            _fail(f"handoff inventory path is unsafe: {raw_path}")
        if raw_path != relative.as_posix():
            _fail(f"handoff inventory path is not canonical POSIX text: {raw_path}")
        expected_paths.append(raw_path)
        size = row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail(f"handoff.inventory.files[{index}].bytes must be a non-negative integer")
        expected_bytes += size
        _required_sha256(row.get("sha256"), f"handoff.inventory.files[{index}].sha256")
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
        _fail("handoff inventory paths must be sorted and unique")

    actual_paths: list[str] = []
    for path in scene_root.rglob("*"):
        if path.is_symlink():
            _fail(f"canonical scene contains a symlink: {path.relative_to(scene_root).as_posix()}")
        if path.is_file():
            actual_paths.append(_relative_path(scene_root, path))
    actual_paths.sort()
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected_paths))
        _fail(f"canonical scene inventory paths drifted: missing={missing}, extra={extra}")

    for index, row in enumerate(rows):
        path = _safe_path(scene_root, row["path"], f"handoff.inventory.files[{index}].path")
        if not path.is_file():
            _fail(f"inventory member is not a regular file: {row['path']}")
        actual_size = path.stat().st_size
        if actual_size != row["bytes"]:
            _fail(f"inventory byte count drifted for {row['path']}")
        if _sha256(path) != row["sha256"]:
            _fail(f"inventory SHA-256 drifted for {row['path']}")

    _expect_equal(handoff_inventory.get("file_count"), len(rows), "handoff.inventory.file_count")
    _expect_equal(handoff_inventory.get("total_bytes"), expected_bytes, "handoff.inventory.total_bytes")
    canonical_digest = _canonical_json_sha256(rows)
    _expect_equal(
        handoff_inventory.get("canonical_sha256"),
        canonical_digest,
        "handoff.inventory.canonical_sha256",
    )
    return {
        "file_count": len(rows),
        "total_bytes": expected_bytes,
        "canonical_sha256": canonical_digest,
    }


def _bundle_instances_by_source(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    instances = bundle.get("instances")
    if not isinstance(instances, list) or not instances:
        _fail("scene bundle instances must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    sources_by_id: dict[str, str] = {}
    for index, instance_value in enumerate(instances):
        instance = _required_mapping(instance_value, f"scene_bundle.instances[{index}]")
        source = _required_string(instance.get("source_node_key"), f"scene_bundle.instances[{index}].source_node_key")
        if source in result:
            _fail(f"scene bundle repeats source_node_key {source!r}")
        instance_id = _required_string(instance.get("instance_id"), f"scene_bundle.instances[{index}].instance_id")
        if re.fullmatch(r"egv2_[0-9a-f]{24}", instance_id) is None:
            _fail(f"scene bundle instance ID is invalid: {instance_id}")
        expected_id = "egv2_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
        _expect_equal(instance_id, expected_id, f"scene bundle canonical ID for {source}")
        prior_source = sources_by_id.get(instance_id)
        if prior_source is not None:
            _fail(f"scene bundle instance ID is reused by {prior_source!r} and {source!r}")
        sources_by_id[instance_id] = source
        result[source] = instance
    return result


def bind_shared_scene(
    task_contract_path: Path,
    handoff_path: Path | None = None,
) -> dict[str, Any]:
    """Verify and bind the exact durable shared scene without copying its assets."""

    task_contract_path = task_contract_path.resolve()
    task = load_task_contract(task_contract_path)
    shared = task["shared_scene_source"]
    expected_handoff_path = Path(shared["handoff_path"]).resolve()
    if handoff_path is None:
        handoff_path = expected_handoff_path
    else:
        handoff_path = handoff_path.resolve()
    _expect_equal(str(handoff_path), str(expected_handoff_path), "shared handoff path")
    if not handoff_path.is_file():
        _fail(f"shared handoff does not exist: {handoff_path}")
    _expect_equal(_sha256(handoff_path), shared["handoff_sha256"], "shared handoff SHA-256")
    handoff = _load_json(handoff_path)
    _expect_equal(handoff.get("schema"), "NurserySharedEmbodiedGenSceneHandoff", "handoff.schema")
    _expect_equal(handoff.get("schema_version"), 1, "handoff.schema_version")
    _expect_equal(handoff.get("status"), "passed", "handoff.status")

    scene_root = Path(shared["canonical_root"]).resolve()
    _expect_equal(handoff.get("canonical_scene_root"), str(scene_root), "handoff.canonical_scene_root")
    if not scene_root.is_dir():
        _fail(f"canonical shared scene root does not exist: {scene_root}")
    forbidden_root = Path(shared["superseded_scene_root_forbidden"]).resolve()
    if scene_root == forbidden_root or forbidden_root in scene_root.parents:
        _fail("canonical scene resolves inside the forbidden superseded root")

    verified_inventory = _verified_inventory(
        scene_root,
        _required_mapping(handoff.get("inventory"), "handoff.inventory"),
    )
    for name, expected in shared["inventory"].items():
        _expect_equal(verified_inventory.get(name), expected, f"shared_scene_source.inventory.{name}")

    activity_path = scene_root / "metadata" / "activity.json"
    activity = _load_json(activity_path)
    _expect_equal(activity.get("schema"), "InteractMoveInterMimicActivitySpec", "activity.schema")
    _expect_equal(activity.get("schema_version"), 1, "activity.schema_version")
    activity_digest = _canonical_json_sha256(activity)
    _expect_equal(activity_digest, shared["activity_spec_sha256"], "ActivitySpec semantic SHA-256")

    handoff_activity = _required_mapping(handoff.get("activity"), "handoff.activity")
    expected_activity = {
        "activity_id": task["task_id"],
        "activity_spec_sha256": shared["activity_spec_sha256"],
        "duration_s": task["duration_seconds"],
        "fps": task["frame_rate_hz"],
        "frame_count": task["frame_count"],
        "prompt": task["activity_prompt"],
        "sampling": task["sampling"],
        "seeds": task["seeds"],
    }
    _expect_equal(handoff_activity, expected_activity, "handoff.activity")
    _expect_equal(activity.get("activity_id"), task["task_id"], "ActivitySpec activity_id")
    _expect_equal(activity.get("prompt"), task["activity_prompt"], "ActivitySpec prompt")
    _expect_equal(activity.get("timing"), {
        "duration_s": task["duration_seconds"],
        "fps": task["frame_rate_hz"],
        "frame_count": task["frame_count"],
        "sampling": task["sampling"],
    }, "ActivitySpec timing")
    _expect_equal(activity.get("seeds"), task["seeds"], "ActivitySpec seeds")
    expected_actor = {
        "body_model": task["initial_human_state"]["source_body_model"],
        "morphology": task["initial_human_state"]["morphology"],
        "gender": task["initial_human_state"]["gender"],
        "betas": task["initial_human_state"]["betas"],
        "initial_stance": task["initial_human_state"]["source_initial_stance"],
        "initial_placement": {
            "method": task["initial_human_state"]["placement"]["method"],
            "target_relative_position_m": task["initial_human_state"]["placement"]["translation_offset_world_m"],
            "face_target": task["initial_human_state"]["placement"]["face_reference"],
            "selected_by": "nursery",
        },
    }
    _expect_equal(activity.get("actor"), expected_actor, "ActivitySpec actor")
    _expect_equal(activity.get("hand_use"), task["hand_use"], "ActivitySpec hand_use")
    _expect_equal(activity.get("success_criteria"), task["acceptance_metrics"], "ActivitySpec success criteria")
    activity_relations = _required_mapping(activity.get("relations"), "ActivitySpec relations")
    for activity_label, task_label in (("start", "required_initial_relations"), ("final", "required_final_relations")):
        source_relations = activity_relations.get(activity_label)
        task_relations = task[task_label]
        if not isinstance(source_relations, list) or len(source_relations) != len(task_relations):
            _fail(f"ActivitySpec {activity_label} relation count differs from the task contract")
        for index, (source_relation, task_relation) in enumerate(zip(source_relations, task_relations)):
            source_relation = _required_mapping(
                source_relation,
                f"ActivitySpec relations.{activity_label}[{index}]",
            )
            expected_relation = {
                "subject": task_relation["child_role"],
                "predicate": task_relation["spatial_relation"].lower(),
                "object_description": task_relation["activity_object_description"],
            }
            for name, expected in expected_relation.items():
                _expect_equal(
                    source_relation.get(name),
                    expected,
                    f"ActivitySpec relations.{activity_label}[{index}].{name}",
                )
    activity_phases = activity.get("phases")
    if not isinstance(activity_phases, list) or len(activity_phases) != len(task["action_phases"]):
        _fail("ActivitySpec phase count differs from the task contract")
    for index, (source_phase, task_phase) in enumerate(zip(activity_phases, task["action_phases"])):
        for name in ("id", "start_frame", "end_frame_exclusive", "expected_target_contact_hands"):
            _expect_equal(source_phase.get(name), task_phase.get(name), f"ActivitySpec phases[{index}].{name}")

    generation = _required_mapping(handoff.get("generation"), "handoff.generation")
    expected_generation_pairs = {
        "embodiedgen_commit": shared["embodiedgen_commit"],
        "image_to_3d_backend": shared["object_asset_backend"]["name"],
        "sam3d_source_commit": shared["object_asset_backend"]["source_commit"],
        "sam3d_checkpoint_revision": shared["object_asset_backend"]["checkpoint_revision"],
        "layout_path": shared["layout_path"],
        "layout_sha256": shared["layout_sha256"],
        "receipt_path": shared["generation_receipt_path"],
        "receipt_sha256": shared["generation_receipt_sha256"],
        "robot_actor_inserted": False,
        "native_robot_metadata_only": True,
        "upstream_sim_cli_invoked": False,
        "accepted_assets_regenerated": False,
    }
    for name, expected in expected_generation_pairs.items():
        _expect_equal(generation.get(name), expected, f"handoff.generation.{name}")

    layout_path = Path(shared["layout_path"]).resolve()
    receipt_path = Path(shared["generation_receipt_path"]).resolve()
    try:
        layout_path.relative_to(scene_root)
        receipt_path.relative_to(scene_root)
    except ValueError:
        _fail("layout and generation receipt must live inside the canonical scene root")
    _expect_equal(_sha256(layout_path), shared["layout_sha256"], "layout SHA-256")
    _expect_equal(_sha256(receipt_path), shared["generation_receipt_sha256"], "generation receipt SHA-256")
    generation_receipt = _load_json(receipt_path)
    _expect_equal(generation_receipt.get("schema"), "InteractMoveInterMimicFreshGenerationReceipt", "generation receipt schema")
    _expect_equal(generation_receipt.get("status"), "passed", "generation receipt status")
    _expect_equal(generation_receipt.get("prompt"), task["activity_prompt"], "generation receipt prompt")
    _expect_equal(generation_receipt.get("seeds"), {
        "image": task["seeds"]["embodiedgen_image"],
        "asset": task["seeds"]["embodiedgen_asset"],
        "layout": task["seeds"]["embodiedgen_layout"],
    }, "generation receipt seeds")
    models = _required_mapping(generation_receipt.get("models"), "generation receipt models")
    _expect_equal(models.get("image_to_3d_backend"), "SAM3D", "generation receipt backend")
    sam3d = _required_mapping(models.get("sam3d"), "generation receipt models.sam3d")
    _expect_equal(sam3d.get("source_commit"), shared["object_asset_backend"]["source_commit"], "SAM3D source commit")
    _expect_equal(sam3d.get("checkpoint_revision"), shared["object_asset_backend"]["checkpoint_revision"], "SAM3D checkpoint revision")
    robot_policy = _required_mapping(generation_receipt.get("robot_policy"), "generation receipt robot_policy")
    _expect_equal(robot_policy.get("render_insert_robot"), False, "generation receipt render_insert_robot")
    _expect_equal(robot_policy.get("robot_actor_loaded"), False, "generation receipt robot_actor_loaded")
    _expect_equal(robot_policy.get("upstream_sim_cli_invoked"), False, "generation receipt upstream_sim_cli_invoked")

    bundle_spec = shared["scene_bundle"]
    bundle_path = Path(bundle_spec["path"]).resolve()
    if not bundle_path.is_file():
        _fail(f"scene bundle does not exist: {bundle_path}")
    _expect_equal(_sha256(bundle_path), bundle_spec["file_sha256"], "scene bundle file SHA-256")
    bundle = _load_json(bundle_path)
    _expect_equal(_canonical_json_sha256(bundle), bundle_spec["semantic_sha256"], "scene bundle semantic SHA-256")
    _expect_equal(bundle.get("schema"), "InteractMoveInterMimicSceneBundle", "scene bundle schema")
    _expect_equal(bundle.get("schema_version"), 1, "scene bundle schema_version")
    _expect_equal(bundle.get("bundle_id"), bundle_spec["bundle_id"], "scene bundle ID")
    handoff_bundle = _required_mapping(handoff.get("scene_bundle"), "handoff.scene_bundle")
    for name in ("path", "file_sha256", "semantic_sha256", "bundle_id"):
        _expect_equal(handoff_bundle.get(name), bundle_spec[name], f"handoff.scene_bundle.{name}")
    _expect_equal(handoff_bundle.get("validation_status"), "passed", "handoff scene bundle validation status")
    capabilities = _required_mapping(bundle.get("capabilities"), "scene bundle capabilities")
    for name in (
        "complete_scene_ready",
        "floor_contact_ready",
        "interactmove_scene_input_ready",
        "reference_mesh_ready",
        "room_collision_ready",
        "target_category_exact_match",
        "visual_background_ready",
    ):
        _expect_equal(capabilities.get(name), True, f"scene bundle capabilities.{name}")
    _expect_equal(capabilities.get("gaussian_render_ready"), False, "scene bundle capabilities.gaussian_render_ready")
    _expect_equal(capabilities.get("dynamic_settle_ready"), False, "scene bundle capabilities.dynamic_settle_ready")
    _expect_equal(capabilities.get("physics_material_complete"), False, "scene bundle capabilities.physics_material_complete")
    settling = _required_mapping(bundle.get("settling"), "scene bundle settling")
    _expect_equal(settling.get("status"), "not_run", "scene bundle settling status")

    resolved = _required_mapping(handoff.get("resolved_instances"), "handoff.resolved_instances")
    bundle_instances = _bundle_instances_by_source(bundle)
    layout = _load_json(layout_path)
    layout_edges = _validate_layout(layout)
    expected_scene_id = "egscene_" + shared["layout_sha256"][:24]
    _expect_equal(bundle.get("scene_id"), expected_scene_id, "scene bundle scene ID")
    expected_bundle_sources = {
        source
        for source, category in layout["objs_mapping"].items()
        if category not in ROBOT_CATEGORIES and source not in KNOWN_ROBOT_NAMES
    }
    _expect_equal(set(bundle_instances), expected_bundle_sources, "scene bundle instance source set")
    relation = layout["relation"]
    expected_role_by_category = {
        "background": "background",
        "context": "support",
        "manipulated_objs": "target",
        "distractor_objs": "distractor",
    }
    expected_body_by_category = {
        "background": "static",
        "context": "static",
        "manipulated_objs": "dynamic",
        "distractor_objs": "dynamic",
    }
    id_by_source = {source: instance["instance_id"] for source, instance in bundle_instances.items()}
    for source, instance in bundle_instances.items():
        category = layout["objs_mapping"][source]
        _expect_equal(instance.get("role"), expected_role_by_category[category], f"scene bundle {source} role")
        _expect_equal(instance.get("native_role"), category, f"scene bundle {source} native role")
        _expect_equal(instance.get("body_type"), expected_body_by_category[category], f"scene bundle {source} body type")
        expected_category = None if category == "background" else source
        _expect_equal(instance.get("category"), expected_category, f"scene bundle {source} category")
        edge = layout_edges.get(source)
        if category == "background":
            _expect_equal(instance.get("parent_instance_id"), None, f"scene bundle {source} parent")
            _expect_equal(instance.get("spatial_relation"), None, f"scene bundle {source} spatial relation")
        else:
            if edge is None:
                _fail(f"layout tree has no parent relation for {source!r}")
            parent_source, spatial_relation = edge
            _expect_equal(instance.get("parent_instance_id"), id_by_source[parent_source], f"scene bundle {source} parent")
            _expect_equal(instance.get("spatial_relation"), spatial_relation, f"scene bundle {source} spatial relation")
    native_robot_name = relation.get("robot")
    _required_string(native_robot_name, "layout.relation.robot")
    if native_robot_name in bundle_instances:
        _fail("scene bundle contains a robot instance")
    bundle_source = _required_mapping(bundle.get("source"), "scene bundle source")
    _expect_equal(bundle_source.get("format"), "EmbodiedGenV2", "scene bundle source format")
    _expect_equal(bundle_source.get("upstream_commit"), shared["embodiedgen_commit"], "scene bundle upstream commit")
    source_layout = _required_mapping(bundle_source.get("layout"), "scene bundle source.layout")
    _expect_equal(source_layout.get("path"), layout_path.name, "scene bundle source layout path")
    _expect_equal(source_layout.get("sha256"), shared["layout_sha256"], "scene bundle source layout SHA-256")
    ignored_robot = _required_mapping(
        bundle_source.get("ignored_native_robot_pose"),
        "scene bundle source.ignored_native_robot_pose",
    )
    _expect_equal(ignored_robot.get("source_node_key"), native_robot_name, "ignored native robot source")
    _expect_equal(ignored_robot.get("authority"), "ignored_not_humanoid_authority", "ignored native robot authority")
    ignored_robot_pose = _required_mapping(ignored_robot.get("pose"), "ignored native robot pose")
    layout_robot_pose = _pose(layout, native_robot_name)
    _expect_vector_close(
        ignored_robot_pose.get("translation_m"),
        layout_robot_pose["translation_m"],
        "ignored native robot translation",
    )
    _expect_quaternion_equivalent(
        ignored_robot_pose.get("rotation_xyzw"),
        layout_robot_pose["orientation_xyzw"],
        "ignored native robot orientation",
    )
    for role in ("target", "support"):
        binding = task["scene_bindings"][role]
        resolved_role = _required_mapping(resolved.get(role), f"handoff.resolved_instances.{role}")
        _expect_equal(resolved_role.get("source_node_key"), binding["source_name"], f"handoff resolved {role} source")
        _expect_equal(resolved_role.get("instance_id"), binding["canonical_instance_id"], f"handoff resolved {role} ID")
        bundle_instance = bundle_instances.get(binding["source_name"])
        if bundle_instance is None:
            _fail(f"scene bundle is missing the {role} instance")
        _expect_equal(bundle_instance.get("instance_id"), binding["canonical_instance_id"], f"scene bundle {role} ID")
    target_binding = task["scene_bindings"]["target"]
    target_receipt = _required_mapping(
        bundle.get("target_resolution_receipt"),
        "scene bundle target_resolution_receipt",
    )
    expected_target_receipt = {
        "chosen_source_node_key": target_binding["source_name"],
        "chosen_instance_id": target_binding["canonical_instance_id"],
        "source_category": target_binding["source_name"],
        "requested_category": target_binding["source_name"],
        "category_exact_match": True,
        "layout_sha256": shared["layout_sha256"],
        "method": "sole_manipulated_instance",
    }
    for name, expected in expected_target_receipt.items():
        _expect_equal(target_receipt.get(name), expected, f"scene bundle target receipt {name}")
    _expect_equal(
        task["scene_bindings"]["goal"]["canonical_instance_id"],
        task["scene_bindings"]["support"]["canonical_instance_id"],
        "goal/support canonical alias",
    )

    conventions = _required_mapping(bundle.get("conventions"), "scene bundle conventions")
    expected_conventions = {
        "handedness": "right",
        "world_up": "+Z",
        "units": {"length": "m", "mass": "kg", "angle": "rad", "time": "s"},
        "quaternion_order": "xyzw",
        "transform_semantics": "T_parent_child",
        "gravity_m_s2": [0.0, 0.0, -9.81],
    }
    _expect_equal(conventions, expected_conventions, "scene bundle conventions")
    _expect_equal(handoff.get("coordinate_convention"), expected_conventions, "handoff coordinate convention")

    boundaries = _required_mapping(handoff.get("scientific_boundaries"), "handoff.scientific_boundaries")
    _expect_equal(boundaries.get("interactmove_motion_generation"), "not_run", "InteractMove motion boundary")
    _expect_equal(boundaries.get("intermimic_execution"), "not_run", "InterMimic boundary")
    _expect_equal(boundaries.get("scene_bundle_validation"), "passed", "scene bundle boundary")

    manifest = resolve_scene_manifest(task_contract_path, layout_path)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest.pop("boundary_status", None)
    manifest["scene_id"] = expected_scene_id
    manifest["task"]["activity_id"] = task["task_id"]
    canonical_ids = {source: instance["instance_id"] for source, instance in bundle_instances.items()}
    for instance in manifest["instances"]:
        source_name = instance["source_name"]
        if source_name not in canonical_ids:
            _fail(f"scene bundle is missing canonical identity for {source_name!r}")
        bundle_instance = bundle_instances[source_name]
        _expect_equal(bundle_instance.get("body_type"), instance["body_mode"], f"scene bundle {source_name} body type")
        bundle_pose = _required_mapping(
            bundle_instance.get("initial_pose_world"),
            f"scene bundle {source_name} initial_pose_world",
        )
        _expect_vector_close(
            bundle_pose.get("translation_m"),
            instance["world_pose"]["translation_m"],
            f"scene bundle {source_name} translation",
        )
        _expect_quaternion_equivalent(
            bundle_pose.get("rotation_xyzw"),
            instance["world_pose"]["orientation_xyzw"],
            f"scene bundle {source_name} orientation",
        )
        instance["instance_id"] = canonical_ids[source_name]
        instance["instance_id_provenance"] = "validated_shared_scene_bundle"
    background_source = relation["background"]
    background_bundle_instance = bundle_instances[background_source]
    background_pose = _required_mapping(
        background_bundle_instance.get("initial_pose_world"),
        "scene bundle background initial_pose_world",
    )
    _expect_vector_close(
        background_pose.get("translation_m"),
        manifest["background"]["world_pose"]["translation_m"],
        "scene bundle background translation",
    )
    _expect_quaternion_equivalent(
        background_pose.get("rotation_xyzw"),
        manifest["background"]["world_pose"]["orientation_xyzw"],
        "scene bundle background orientation",
    )
    manifest["role_bindings"] = {
        role: binding["canonical_instance_id"] for role, binding in task["scene_bindings"].items()
    }
    reference_role = task["initial_human_state"]["placement"]["reference_role"]
    manifest["initial_human_state"]["placement_provenance"]["reference_instance_id"] = (
        manifest["role_bindings"][reference_role]
    )
    for relation in manifest["required_initial_relations"]:
        relation["child_instance_id"] = manifest["role_bindings"][relation["child_role"]]
        relation["parent_instance_id"] = manifest["role_bindings"][relation["parent_role"]]
    target_bundle_instance = bundle_instances[task["scene_bindings"]["target"]["source_name"]]
    _expect_equal(target_bundle_instance.get("parent_instance_id"), manifest["role_bindings"]["support"], "target parent instance")
    _expect_equal(target_bundle_instance.get("spatial_relation"), "ON", "target initial spatial relation")
    manifest["required_final_relations"] = [
        {
            **relation,
            "child_instance_id": manifest["role_bindings"][relation["child_role"]],
            "parent_instance_id": manifest["role_bindings"][relation["parent_role"]],
            "evaluation_status": "intended_not_yet_evaluated",
        }
        for relation in task["required_final_relations"]
    ]

    humanoid = _required_mapping(bundle.get("humanoid_initial_pose"), "scene bundle humanoid_initial_pose")
    _expect_equal(humanoid.get("authority"), "stage1_target_relative_placement_not_native_robot_pose", "humanoid pose authority")
    pose_world = _required_mapping(humanoid.get("pose_world"), "scene bundle humanoid pose_world")
    computed_pose = manifest["initial_human_state"]["world_pose"]
    _expect_equal(computed_pose.get("translation_m"), pose_world.get("translation_m"), "computed humanoid translation")
    _expect_equal(computed_pose.get("orientation_xyzw"), pose_world.get("rotation_xyzw"), "computed humanoid orientation")
    manifest["initial_human_state"]["world_pose"] = {
        "translation_m": pose_world["translation_m"],
        "orientation_xyzw": pose_world["rotation_xyzw"],
    }
    manifest["initial_human_state"]["placement_provenance"].update({
        "authority": humanoid["authority"],
        "activity_spec_sha256": shared["activity_spec_sha256"],
        "face_reference": task["initial_human_state"]["placement"]["face_reference"],
    })

    environment = _required_mapping(bundle.get("environment"), "scene bundle environment")
    _expect_equal(
        environment.get("background_instance_id"),
        background_bundle_instance["instance_id"],
        "scene bundle environment background instance ID",
    )
    reference_mesh = _required_mapping(environment.get("reference_mesh_report"), "scene bundle reference_mesh_report")
    _expect_equal(reference_mesh.get("sha256"), shared["reference_mesh_sha256"], "reference mesh SHA-256")
    reference_mesh_path = _safe_path(scene_root, reference_mesh.get("path"), "scene bundle reference mesh")
    _expect_equal(_sha256(reference_mesh_path), shared["reference_mesh_sha256"], "reference mesh file SHA-256")
    geometry_manifest_path = _safe_path(
        scene_root,
        environment.get("geometry_manifest"),
        "scene bundle geometry manifest",
    )
    if not geometry_manifest_path.is_file():
        _fail("scene bundle geometry manifest must be a file")
    geometry_manifest = _load_json(geometry_manifest_path)
    _expect_equal(geometry_manifest.get("schema"), "InteractMoveEnvironmentGeometry", "environment geometry schema")
    _expect_equal(geometry_manifest.get("schema_version"), 1, "environment geometry schema_version")
    collision_geometry = environment.get("collision_geometry")
    if not isinstance(collision_geometry, list) or len(collision_geometry) != 5:
        _fail("scene bundle must designate exactly one floor and four wall collision geometries")
    floors = [row for row in collision_geometry if isinstance(row, dict) and row.get("role") == "floor"]
    walls = [row for row in collision_geometry if isinstance(row, dict) and row.get("role") == "walls"]
    if len(floors) != 1 or len(walls) != 4:
        _fail("scene bundle collision geometry must contain one floor and four walls")
    floor = floors[0]
    _expect_equal(floor.get("collision_id"), "floor", "floor collision ID")
    _expect_equal(floor.get("frame"), "world", "floor collision frame")
    _expect_equal(floor.get("geometry_type"), "plane", "floor collision geometry type")
    _expect_equal(floor.get("normal"), [0.0, 0.0, 1.0], "floor collision normal")
    _expect_equal(floor.get("z_m"), 0.0, "floor collision height")
    expected_wall_ids = {"wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max"}
    _expect_equal({wall.get("collision_id") for wall in walls}, expected_wall_ids, "wall collision IDs")
    geometry_rows = geometry_manifest.get("collision_geometry")
    if (
        not isinstance(geometry_rows, list)
        or len(geometry_rows) != 5
        or any(not isinstance(row, dict) for row in geometry_rows)
    ):
        _fail("environment geometry collision_geometry must contain exactly five object rows")
    geometry_by_id = {
        row.get("collision_id"): row for row in geometry_rows if isinstance(row, dict)
    }
    if len(geometry_by_id) != len(geometry_rows):
        _fail("environment geometry collision IDs must be unique")
    _expect_equal(set(geometry_by_id), {"floor", *expected_wall_ids}, "environment geometry collision IDs")
    _expect_equal(geometry_by_id["floor"], floor, "environment geometry floor")
    wall_paths: set[str] = set()
    for index, wall in enumerate(walls):
        _expect_equal(wall.get("geometry_type"), "mesh", f"wall collision {index} geometry type")
        _expect_equal(wall.get("frame"), "world", f"wall collision {index} frame")
        _expect_equal(wall.get("units"), "m", f"wall collision {index} units")
        wall_mesh = _required_mapping(wall.get("mesh"), f"wall collision {index} mesh")
        wall_relative_path = _required_string(wall_mesh.get("path"), f"wall collision {index} mesh.path")
        if wall_relative_path in wall_paths:
            _fail("scene bundle repeats a wall collision mesh path")
        wall_paths.add(wall_relative_path)
        _expect_equal(wall_mesh.get("format"), "obj", f"wall collision {index} format")
        _expect_equal(wall_mesh.get("scale_xyz"), [1.0, 1.0, 1.0], f"wall collision {index} scale")
        _expect_equal(wall_mesh.get("finite_vertices"), True, f"wall collision {index} finite vertices")
        for count_name in ("vertex_count", "face_count"):
            count = wall_mesh.get(count_name)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                _fail(f"wall collision {index} {count_name} must be positive")
        wall_path = _safe_path(scene_root, wall_mesh.get("path"), f"wall collision {index} path")
        _expect_equal(_sha256(wall_path), wall_mesh.get("sha256"), f"wall collision {index} SHA-256")
        geometry_row = _required_mapping(
            geometry_by_id[wall["collision_id"]],
            f"environment geometry {wall['collision_id']}",
        )
        _expect_equal(geometry_row.get("frame"), "world", f"environment geometry {wall['collision_id']} frame")
        _expect_equal(geometry_row.get("role"), "walls", f"environment geometry {wall['collision_id']} role")
        _expect_equal(geometry_row.get("geometry_type"), "mesh", f"environment geometry {wall['collision_id']} type")
        _expect_equal(geometry_row.get("units"), "m", f"environment geometry {wall['collision_id']} units")
        _expect_equal(geometry_row.get("scale_xyz"), [1.0, 1.0, 1.0], f"environment geometry {wall['collision_id']} scale")
        try:
            expected_geometry_path = Path(wall_relative_path).relative_to("background").as_posix()
        except ValueError:
            _fail(f"wall collision path must live below background/: {wall_relative_path}")
        _expect_equal(
            geometry_row.get("path"),
            expected_geometry_path,
            f"environment geometry {wall['collision_id']} path",
        )
    manifest["background"]["instance_id"] = environment.get("background_instance_id")
    manifest["background"]["reference_mesh"] = reference_mesh
    manifest["background"]["collision"] = {
        "available": True,
        "authority": "validated_shared_scene_bundle",
        "geometry_manifest": environment.get("geometry_manifest"),
        "geometries": collision_geometry,
    }
    manifest["source"].update({
        "assumed_commit": shared["embodiedgen_commit"],
        "layout_path": str(layout_path),
        "generation_receipt_path": str(receipt_path),
        "generation_receipt_sha256": shared["generation_receipt_sha256"],
        "generation_execution_verified": True,
        "validation_scope": "shared_scene_handoff_full_inventory_and_bundle",
    })
    manifest["robot_policy"].update({
        "shared_scene_robot_actor_inserted": False,
        "native_robot_metadata_only": True,
    })
    manifest["limitations"] = [
        "Missing restitution and upstream placeholder inertia remain unavailable rather than receiving invented values.",
        "The scene-only SAPIEN canary does not validate humanoid control, intended contact, URDF mass application, or object settling.",
        "Human accessibility was not proactively evaluated at Stages 1-2, by contract.",
        "The shared humanoid record fixes world placement metadata and a neutral-stance label, not complete SMPL-X tensors or root/pelvis translation semantics.",
    ]

    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "status": "passed",
        "implementation": {
            "stage12_path": str(Path(__file__).resolve()),
            "stage12_sha256": _sha256(Path(__file__).resolve()),
        },
        "task": {
            "task_id": task["task_id"],
            "activity_prompt": task["activity_prompt"],
            "activity_spec_sha256": shared["activity_spec_sha256"],
            "contract_path": str(task_contract_path),
            "contract_sha256": _sha256(task_contract_path),
            "duration_seconds": task["duration_seconds"],
            "frame_rate_hz": task["frame_rate_hz"],
            "frame_count": task["frame_count"],
            "sampling": task["sampling"],
            "seeds": task["seeds"],
        },
        "shared_scene": {
            "scene_id": expected_scene_id,
            "canonical_root": str(scene_root),
            "source_assets_copied": False,
            "handoff_path": str(handoff_path),
            "handoff_sha256": shared["handoff_sha256"],
            "layout_path": str(layout_path),
            "layout_sha256": shared["layout_sha256"],
            "generation_receipt_path": str(receipt_path),
            "generation_receipt_sha256": shared["generation_receipt_sha256"],
            "inventory": verified_inventory,
            "scene_bundle": bundle_spec,
            "object_asset_backend": shared["object_asset_backend"],
            "reference_mesh_sha256": shared["reference_mesh_sha256"],
            "superseded_scene_root_rejected": str(forbidden_root),
        },
        "scene_manifest": manifest,
        "physics_readiness": {
            "complete_scene_ready": capabilities["complete_scene_ready"],
            "room_collision_ready": capabilities["room_collision_ready"],
            "physics_material_complete": capabilities["physics_material_complete"],
            "dynamic_settle_ready": capabilities["dynamic_settle_ready"],
            "settling_status": settling["status"],
            "scene_bundle_reported_blockers": capabilities.get("blockers", []),
            "additional_unresolved_physics_inputs": [
                "table collision restitution is missing even though the SceneBundle blocker list names only dynamic objects",
                "the scene-only canary did not apply URDF mass",
                "upstream placeholder inertia remains unresolved where reported by object asset parsing",
            ],
            "scene_only_canary_is_not_stage3_validation": True,
        },
        "stage_status": {
            "stage1_activity_contract_complete": True,
            "stage2_shared_scene_binding_complete": True,
            "hoidini_motion_generated": False,
            "intermimic_executed": False,
            "video_rendered": False,
        },
    }
