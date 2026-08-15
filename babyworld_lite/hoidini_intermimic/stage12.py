"""Stage 1 task contracts and the robot-free EmbodiedGenV2 scene boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET


TASK_SCHEMA_VERSION = "hoidini-intermimic-task-contract-1.0.0"
MANIFEST_SCHEMA_VERSION = "hoidini-intermimic-scene-manifest-1.0.0"
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
    "human_hands",
}
MEASUREMENT_TYPES = {
    "identity_match",
    "support_separation",
    "goal_region",
    "stable_support_duration",
    "unexpected_support_loss",
    "humanoid_fall",
    "maximum_penetration_depth",
    "non_target_displacement",
}


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


def validate_task_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate the explicit Stage 1 contract and return it unchanged."""

    if contract.get("schema_version") != TASK_SCHEMA_VERSION:
        _fail(f"schema_version must be {TASK_SCHEMA_VERSION!r}")
    _required_string(contract.get("task_id"), "task_id")
    _required_string(contract.get("activity_prompt"), "activity_prompt")
    duration = _finite_number(contract.get("duration_seconds"), "duration_seconds")
    if duration <= 0:
        _fail("duration_seconds must be positive")

    coordinates = _required_mapping(contract.get("coordinate_system"), "coordinate_system")
    expected_coordinates = {
        "handedness": "right",
        "up_axis": "+Z",
        "linear_unit": "meter",
        "angle_unit": "radian",
        "quaternion_order": "xyzw",
    }
    if coordinates != expected_coordinates:
        _fail(f"coordinate_system must equal {expected_coordinates}")

    seeds = _required_mapping(contract.get("seeds"), "seeds")
    for name in ("scene", "layout", "motion", "physics", "render"):
        value = seeds.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"seeds.{name} must be a non-negative integer")

    bindings = _required_mapping(contract.get("scene_bindings"), "scene_bindings")
    source_names: set[str] = set()
    for role, expected_body_mode in (
        ("target", "dynamic"),
        ("support", "static"),
        ("goal", "static"),
    ):
        binding = _required_mapping(bindings.get(role), f"scene_bindings.{role}")
        name = _required_string(binding.get("source_name"), f"scene_bindings.{role}.source_name")
        if name in source_names:
            _fail(f"scene binding source name is reused: {name}")
        source_names.add(name)
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
    placement = _required_mapping(human.get("placement"), "initial_human_state.placement")
    if placement.get("reference_role") not in bindings:
        _fail("initial_human_state.placement.reference_role must name a scene binding")
    _number_vector(placement.get("translation_offset_world_m"), 3, "initial_human_state.placement.translation_offset_world_m")
    _quaternion_xyzw(placement.get("orientation_xyzw"), "initial_human_state.placement.orientation_xyzw")
    _required_string(human.get("joint_pose"), "initial_human_state.joint_pose")

    initial_objects = _required_mapping(contract.get("initial_object_states"), "initial_object_states")
    for role in bindings:
        state = _required_mapping(initial_objects.get(role), f"initial_object_states.{role}")
        if state.get("pose_source") != "embodiedgen_layout":
            _fail(f"initial_object_states.{role}.pose_source must be embodiedgen_layout")

    initial_relations = contract.get("required_initial_relations")
    if not isinstance(initial_relations, list) or not initial_relations:
        _fail("required_initial_relations must be a non-empty list")
    relation_ids: set[str] = set()
    for index, relation_value in enumerate(initial_relations):
        relation = _required_mapping(relation_value, f"required_initial_relations[{index}]")
        relation_id = _required_string(relation.get("relation_id"), f"required_initial_relations[{index}].relation_id")
        if relation_id in relation_ids:
            _fail(f"duplicate required relation ID: {relation_id}")
        relation_ids.add(relation_id)
        if relation.get("child_role") not in bindings or relation.get("parent_role") not in bindings:
            _fail(f"required_initial_relations[{index}] must reference declared scene roles")
        if relation.get("spatial_relation") not in {"ON", "INSIDE", "FLOOR", "IN"}:
            _fail(f"required_initial_relations[{index}].spatial_relation is invalid")

    phases = contract.get("action_phases")
    if not isinstance(phases, list) or not phases:
        _fail("action_phases must be a non-empty list")
    cursor = 0.0
    phase_names: set[str] = set()
    for index, phase_value in enumerate(phases):
        phase = _required_mapping(phase_value, f"action_phases[{index}]")
        name = _required_string(phase.get("name"), f"action_phases[{index}].name")
        if name in phase_names:
            _fail(f"duplicate action phase name: {name}")
        phase_names.add(name)
        start = _finite_number(phase.get("start_seconds"), f"action_phases[{index}].start_seconds")
        end = _finite_number(phase.get("end_seconds"), f"action_phases[{index}].end_seconds")
        if not math.isclose(start, cursor, abs_tol=1e-9) or end <= start:
            _fail("action_phases must be contiguous, ordered, and have positive duration")
        cursor = end
        _required_string(phase.get("intent"), f"action_phases[{index}].intent")
    if not math.isclose(cursor, duration, abs_tol=1e-9):
        _fail("action_phases must cover exactly duration_seconds")

    windows = contract.get("contact_windows")
    if not isinstance(windows, list) or not windows:
        _fail("contact_windows must be a non-empty list")
    contact_ids: set[str] = set()
    allowed_participants = set(bindings) | HUMAN_CONTACT_PARTICIPANTS
    for index, window_value in enumerate(windows):
        window = _required_mapping(window_value, f"contact_windows[{index}]")
        contact_id = _required_string(window.get("contact_id"), f"contact_windows[{index}].contact_id")
        if contact_id in contact_ids:
            _fail(f"duplicate contact ID: {contact_id}")
        contact_ids.add(contact_id)
        start = _finite_number(window.get("start_seconds"), f"contact_windows[{index}].start_seconds")
        end = _finite_number(window.get("end_seconds"), f"contact_windows[{index}].end_seconds")
        if start < 0 or end <= start or end > duration:
            _fail(f"contact_windows[{index}] lies outside the task duration")
        if window.get("requirement") not in {"required", "allowed", "forbidden"}:
            _fail(f"contact_windows[{index}].requirement is invalid")
        participants = window.get("participants")
        if not isinstance(participants, list) or len(participants) < 2:
            _fail(f"contact_windows[{index}].participants must name at least two participants")
        if any(participant not in allowed_participants for participant in participants):
            _fail(f"contact_windows[{index}] contains an unknown participant")

    criteria = contract.get("success_criteria")
    failures = contract.get("forbidden_failures")
    if not isinstance(criteria, list) or not criteria:
        _fail("success_criteria must be a non-empty list")
    if not isinstance(failures, list) or not failures:
        _fail("forbidden_failures must be a non-empty list")
    criterion_ids: set[str] = set()
    for label, rows in (("success_criteria", criteria), ("forbidden_failures", failures)):
        for index, row_value in enumerate(rows):
            row = _required_mapping(row_value, f"{label}[{index}]")
            criterion_id = _required_string(row.get("criterion_id"), f"{label}[{index}].criterion_id")
            if criterion_id in criterion_ids:
                _fail(f"duplicate criterion ID: {criterion_id}")
            criterion_ids.add(criterion_id)
            _required_string(row.get("plain_language"), f"{label}[{index}].plain_language")
            measurement = _required_mapping(row.get("measurement"), f"{label}[{index}].measurement")
            if measurement.get("type") not in MEASUREMENT_TYPES:
                _fail(f"{label}[{index}].measurement.type is unsupported")
            if "phase" in measurement and measurement["phase"] not in phase_names:
                _fail(f"{label}[{index}].measurement.phase is unknown")
            if "role" in measurement and measurement["role"] not in bindings:
                _fail(f"{label}[{index}].measurement.role is unknown")
            if "exclude_roles" in measurement:
                excluded = measurement["exclude_roles"]
                if not isinstance(excluded, list) or any(role not in bindings for role in excluded):
                    _fail(f"{label}[{index}].measurement.exclude_roles contains an unknown role")
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
    """Resolve a validated task and EmbodiedGen layout into a robot-free manifest."""

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
    role_by_source = {binding["source_name"]: role for role, binding in bindings.items()}
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
        role = role_by_source.get(source_name)
        if role is not None:
            body_mode = bindings[role]["body_mode"]
            body_mode_provenance = f"task_contract.scene_bindings.{role}.body_mode"
        else:
            body_mode = "static" if category == "context" else "dynamic"
            body_mode_provenance = f"local_policy_from_upstream_category:{category}"
        urdf_path = _resolve_urdf(scene_root, source_name, layout["assets"][source_name])
        instances.append(
            {
                "instance_id": instance_id,
                "source_name": source_name,
                "upstream_category": category,
                "scientific_roles": [role] if role is not None else [],
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
        if len(visual_references) != 2:
            _fail("EmbodiedGen background must contain gs_model.ply and mesh_model.ply")
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
        "schema_version": MANIFEST_SCHEMA_VERSION,
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
            "No real EmbodiedGen scene-generation run is claimed by this boundary compiler.",
            "Background visual meshes are not treated as collision proxies without an explicit validated designation.",
            "Missing restitution and upstream placeholder inertia remain unavailable rather than receiving invented values.",
            "Human reachability is not proactively evaluated at Stages 1-2.",
        ],
    }
