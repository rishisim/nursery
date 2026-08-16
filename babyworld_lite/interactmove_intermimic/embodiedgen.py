"""Strict Stage 2 EmbodiedGen ingestion and Stage 3 scene compilation."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any

from .activity import (
    activity_spec_sha256,
    build_embodiedgen_request,
    validate_actor_contract,
    validate_activity_spec,
    validate_target_request_contract,
)
from .common import (
    ContractError,
    canonicalize_quaternion_xyzw,
    content_sha256,
    multiply_transform_matrices,
    posix_relative_path,
    quaternion_xyzw_from_rpy,
    read_json,
    require_exact_keys,
    require_finite_number,
    require_mapping,
    require_nonempty_string,
    require_vector,
    safe_relative_file,
    sha256_file,
    transform_matrix,
)


SCENE_BUNDLE_SCHEMA = "InteractMoveInterMimicSceneBundle"
SCENE_BUNDLE_SCHEMA_VERSION = 1
NATIVE_PROFILE = "embodiedgen-v2.0.1-sapien-rh-zup-m-xyzw"
EMBODIEDGEN_UPSTREAM_VERSION = "v2.0.1"
EMBODIEDGEN_UPSTREAM_COMMIT = "9b333554254af196bace88c1a171a3bf047fa09c"
_LAYOUT_REQUIRED = {
    "tree",
    "relation",
    "objs_desc",
    "objs_mapping",
    "assets",
    "position",
}
_LAYOUT_OPTIONAL = {"quality"}
_ROLE_KEYS = ("background", "context", "manipulated_objs", "distractor_objs")
_SPATIAL_RELATIONS = {"ON", "INSIDE", "IN", "FLOOR"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL = {
    "schema",
    "schema_version",
    "protocol_id",
    "bundle_id",
    "scene_id",
    "activity_binding",
    "source",
    "conventions",
    "files",
    "assets",
    "instances",
    "environment",
    "target_request",
    "target_resolution_receipt",
    "humanoid_initial_pose",
    "capabilities",
    "validation",
    "settling",
}
_CONVENTIONS = {
    "units": {"length": "m", "mass": "kg", "angle": "rad", "time": "s"},
    "handedness": "right",
    "world_up": "+Z",
    "quaternion_order": "xyzw",
    "transform_semantics": "T_parent_child",
    "gravity_m_s2": [0.0, 0.0, -9.81],
}
_REQUEST_KEYS = {
    "schema",
    "schema_version",
    "protocol_id",
    "activity_id",
    "activity_spec_sha256",
    "prompt",
    "upstream",
    "native_profile",
    "background_list",
    "seeds",
    "retry_limits",
    "render_insert_robot",
    "native_robot_pose_role",
    "expected_outputs",
}


def _instance_id(node: str) -> str:
    return "egv2_" + hashlib.sha256(node.encode("utf-8")).hexdigest()[:24]


def _finite_vector_text(
    text: str | None, *, length: int, default: str | None, where: str
) -> tuple[list[float], bool]:
    used_default = text is None
    if text is None:
        if default is None:
            raise ContractError(f"{where} is required")
        text = default
    parts = text.split()
    if len(parts) != length:
        raise ContractError(f"{where} must contain {length} finite numbers")
    return ([_finite_text(part, where=f"{where}[{i}]") for i, part in enumerate(parts)], used_default)


def _finite_text(text: str, *, where: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{where} must be a finite number") from exc
    return require_finite_number(value, where=where)


def _file_record(path: Path, root: Path, roles: set[str]) -> dict[str, Any]:
    return {
        "path": posix_relative_path(root, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "roles": sorted(roles),
    }


class _Files:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._records: dict[str, dict[str, Any]] = {}

    def add(self, path: Path, role: str) -> dict[str, Any]:
        relative = posix_relative_path(self.root, path)
        if relative in self._records:
            record = self._records[relative]
            roles = set(record["roles"])
            roles.add(role)
            record["roles"] = sorted(roles)
            return record
        record = _file_record(path, self.root, {role})
        if record["bytes"] == 0:
            raise ContractError(f"consumed file must not be empty: {relative}")
        self._records[relative] = record
        return record

    def values(self) -> list[dict[str, Any]]:
        return [self._records[key] for key in sorted(self._records)]


def _declared_child_path(
    root: Path, directory: Any, leaf: str, *, where: str
) -> Path:
    directory_text = require_nonempty_string(directory, where=where)
    if any(mark in directory_text for mark in ("\\", "\x00", "://")):
        raise ContractError(f"{where} must be a relative POSIX directory")
    pure = PurePosixPath(directory_text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError(f"{where} must be a traversal-free relative POSIX directory")
    return safe_relative_file(root, (pure / leaf).as_posix(), where=where)


def _urdf_reference(root: Path, urdf_relative: str, reference: Any, *, where: str) -> Path:
    text = require_nonempty_string(reference, where=where)
    if any(mark in text for mark in ("\\", "\x00", "://")) or text.startswith("package:"):
        raise ContractError(f"{where} must be a relative POSIX path")
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError(f"{where} must be a traversal-free relative POSIX path")
    relative = PurePosixPath(urdf_relative).parent / pure
    return safe_relative_file(root, relative.as_posix(), where=where)


def _mesh_dependencies(mesh_path: Path, root: Path, files: _Files) -> list[str]:
    """Hash OBJ material libraries and the texture files those libraries name."""

    if mesh_path.suffix.lower() != ".obj":
        return []
    try:
        lines = mesh_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ContractError(f"OBJ is not valid UTF-8: {posix_relative_path(root, mesh_path)}") from exc
    mesh_relative = PurePosixPath(posix_relative_path(root, mesh_path))
    material_paths: list[Path] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped.startswith("mtllib "):
            continue
        try:
            references = shlex.split(stripped)[1:]
        except ValueError as exc:
            raise ContractError(f"malformed OBJ mtllib at {mesh_relative}:{line_number}") from exc
        for reference in references:
            material_paths.append(
                _urdf_reference(
                    root,
                    mesh_relative.as_posix(),
                    reference,
                    where=f"OBJ mtllib {mesh_relative}:{line_number}",
                )
            )
    hashes: set[str] = set()
    map_keys = {"map_ka", "map_kd", "map_ks", "map_ke", "map_bump", "bump", "disp", "decal", "norm"}
    for material_path in material_paths:
        material_record = files.add(material_path, "visual_material")
        hashes.add(material_record["sha256"])
        try:
            material_lines = material_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ContractError(
                f"MTL is not valid UTF-8: {material_record['path']}"
            ) from exc
        material_relative = PurePosixPath(material_record["path"])
        for line_number, line in enumerate(material_lines, start=1):
            try:
                tokens = shlex.split(line.strip())
            except ValueError as exc:
                raise ContractError(
                    f"malformed MTL reference at {material_relative}:{line_number}"
                ) from exc
            if len(tokens) < 2 or tokens[0].lower() not in map_keys:
                continue
            texture_path = _urdf_reference(
                root,
                material_relative.as_posix(),
                tokens[-1],
                where=f"MTL texture {material_relative}:{line_number}",
            )
            texture_record = files.add(texture_path, "visual_texture")
            hashes.add(texture_record["sha256"])
    return sorted(hashes)


def _pose_from_layout(raw: Any, *, where: str) -> dict[str, Any]:
    pose = require_vector(raw, length=7, where=where)
    quaternion, changed = canonicalize_quaternion_xyzw(
        pose[3:], where=f"{where}[3:7]", maximum_norm_error=1e-3
    )
    translation = pose[:3]
    return {
        "parent_frame": "world",
        "child_frame": None,
        "translation_m": translation,
        "rotation_xyzw": quaternion,
        "matrix_4x4": transform_matrix(translation, quaternion),
        "quaternion_canonicalized": changed,
        "provenance": {
            "source": "layout.json",
            "source_pointer": where,
            "source_order": "[x,y,z,qx,qy,qz,qw]",
            "operation": "normalize_and_choose_nonnegative_w",
        },
    }


def _geometry_origin(element: ET.Element, *, where: str) -> dict[str, Any]:
    origin = element.find("./origin")
    xyz, xyz_default = _finite_vector_text(
        None if origin is None else origin.get("xyz"),
        length=3,
        default="0 0 0",
        where=f"{where}.origin.xyz",
    )
    rpy, rpy_default = _finite_vector_text(
        None if origin is None else origin.get("rpy"),
        length=3,
        default="0 0 0",
        where=f"{where}.origin.rpy",
    )
    quaternion = quaternion_xyzw_from_rpy(rpy)
    return {
        "translation_m": xyz,
        "rpy_rad": rpy,
        "rotation_xyzw": quaternion,
        "matrix_4x4": transform_matrix(xyz, quaternion),
        "used_urdf_default": xyz_default or rpy_default,
        "source_pointer": where,
    }


def _mesh_geometry(
    element: ET.Element,
    *,
    root: Path,
    urdf_relative: str,
    files: _Files,
    role: str,
    where: str,
) -> dict[str, Any]:
    geometry = element.find("./geometry")
    if geometry is None:
        raise ContractError(f"{where}.geometry is required")
    children = list(geometry)
    if len(children) != 1 or children[0].tag != "mesh":
        raise ContractError(f"{where} must contain exactly one mesh geometry")
    mesh = children[0]
    mesh_path = _urdf_reference(
        root, urdf_relative, mesh.get("filename"), where=f"{where}.mesh.filename"
    )
    scale, used_default = _finite_vector_text(
        mesh.get("scale"), length=3, default="1 1 1", where=f"{where}.mesh.scale"
    )
    if any(value <= 0.0 for value in scale):
        raise ContractError(f"{where}.mesh.scale entries must be > 0")
    record = files.add(mesh_path, role)
    dependency_hashes = _mesh_dependencies(mesh_path, root, files) if role == "visual_mesh" else []
    return {
        "file": record["path"],
        "sha256": record["sha256"],
        "dependency_sha256": dependency_hashes,
        "scale_xyz": scale,
        "scale_used_urdf_default": used_default,
        "T_link_geometry": _geometry_origin(element, where=where),
    }


def _optional_scalar_text(element: ET.Element | None, path: str, *, where: str) -> float | None:
    if element is None:
        return None
    found = element.find(path)
    if found is None or found.text is None:
        return None
    return _finite_text(found.text.strip(), where=where)


_INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


def _validate_inertia_values(value: Any, *, where: str) -> dict[str, float]:
    inertia = require_mapping(value, where=where)
    require_exact_keys(inertia, required=set(_INERTIA_KEYS), where=where)
    values = {
        name: require_finite_number(inertia[name], where=f"{where}.{name}")
        for name in _INERTIA_KEYS
    }
    ixx, ixy, ixz = values["ixx"], values["ixy"], values["ixz"]
    iyy, iyz, izz = values["iyy"], values["iyz"], values["izz"]
    determinant = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    if ixx <= 0.0 or ixx * iyy - ixy * ixy <= 0.0 or determinant <= 0.0:
        raise ContractError(f"{where} must be symmetric positive definite")

    half_trace = 0.5 * (ixx + iyy + izz)
    jxx, jyy, jzz = half_trace - ixx, half_trace - iyy, half_trace - izz
    jxy, jxz, jyz = -ixy, -ixz, -iyz
    tolerance = 1e-12 * max(1.0, abs(half_trace))
    principal_minors = (
        jxx,
        jyy,
        jzz,
        jxx * jyy - jxy * jxy,
        jxx * jzz - jxz * jxz,
        jyy * jzz - jyz * jyz,
        jxx * (jyy * jzz - jyz * jyz)
        - jxy * (jxy * jzz - jyz * jxz)
        + jxz * (jxy * jyz - jyy * jxz),
    )
    if any(minor < -tolerance for minor in principal_minors):
        raise ContractError(f"{where} violates principal-moment triangle inequalities")
    return values


def _inertial(link: ET.Element, *, where: str) -> dict[str, Any]:
    inertial = link.find("./inertial")
    if inertial is None:
        return {
            "mass_kg": None,
            "origin": None,
            "inertia_kg_m2": None,
            "source_fields": {"mass_kg": None, "inertia_kg_m2": None},
        }
    mass_element = inertial.find("./mass")
    inertia_element = inertial.find("./inertia")
    mass = None
    if mass_element is not None:
        if mass_element.get("value") is None:
            raise ContractError(f"{where}.mass is missing value")
        mass = _finite_text(mass_element.get("value", ""), where=f"{where}.mass")
        if mass <= 0.0:
            raise ContractError(f"{where}.mass must be > 0")
    values = None
    if inertia_element is not None:
        if any(inertia_element.get(name) is None for name in _INERTIA_KEYS):
            raise ContractError(f"{where}.inertia must provide all six components")
        values = _validate_inertia_values(
            {
                name: _finite_text(
                    inertia_element.get(name, ""),
                    where=f"{where}.inertia.{name}",
                )
                for name in _INERTIA_KEYS
            },
            where=f"{where}.inertia",
        )
    return {
        "mass_kg": mass,
        "origin": _geometry_origin(inertial, where=f"{where}.inertial"),
        "inertia_kg_m2": values,
        "source_fields": {
            "mass_kg": (
                None if mass_element is None else "link/inertial/mass@value"
            ),
            "inertia_kg_m2": (
                None if inertia_element is None else "link/inertial/inertia@*"
            ),
        },
    }


def _collision_material(collision: ET.Element, *, where: str) -> dict[str, Any]:
    gazebo = collision.find("./gazebo")
    mu1 = _optional_scalar_text(gazebo, "./mu1", where=f"{where}.mu1")
    mu2 = _optional_scalar_text(gazebo, "./mu2", where=f"{where}.mu2")
    if (mu1 is None) != (mu2 is None):
        raise ContractError(f"{where} has partial friction; mu1 and mu2 must occur together")
    if mu1 is not None and (mu1 < 0.0 or mu2 is None or mu2 < 0.0):
        raise ContractError(f"{where} friction values must be >= 0")
    return {
        "source_mu1": mu1,
        "source_mu2": mu2,
        "source_restitution": None,
        "static_friction": mu1,
        "dynamic_friction": mu2,
        "restitution": None,
        "source_fields": {
            "static_friction": None if mu1 is None else "collision/gazebo/mu1",
            "dynamic_friction": None if mu2 is None else "collision/gazebo/mu2",
            "restitution": None,
        },
        "interpretation": "EmbodiedGen_mu1_static_mu2_dynamic",
    }


def _joint_record(joint: ET.Element, *, where: str) -> dict[str, Any]:
    name = require_nonempty_string(joint.get("name"), where=f"{where}.name")
    kind = require_nonempty_string(joint.get("type"), where=f"{where}.type")
    parent = joint.find("./parent")
    child = joint.find("./child")
    if parent is None or child is None:
        raise ContractError(f"{where} must declare parent and child links")
    parent_name = require_nonempty_string(parent.get("link"), where=f"{where}.parent")
    child_name = require_nonempty_string(child.get("link"), where=f"{where}.child")
    axis_element = joint.find("./axis")
    axis = None
    if axis_element is not None:
        axis, _ = _finite_vector_text(
            axis_element.get("xyz"), length=3, default="1 0 0", where=f"{where}.axis"
        )
    limit_element = joint.find("./limit")
    limit = None
    if limit_element is not None:
        limit = {}
        for key in ("lower", "upper", "effort", "velocity"):
            raw = limit_element.get(key)
            limit[key] = (
                None
                if raw is None
                else _finite_text(raw, where=f"{where}.limit.{key}")
            )
    return {
        "name": name,
        "type": kind,
        "parent_link": parent_name,
        "child_link": child_name,
        "origin": _geometry_origin(joint, where=where),
        "axis": axis,
        "limit": limit,
    }


def _validate_joint_topology(
    link_names: set[str], joints: list[dict[str, Any]], *, node: str
) -> None:
    """Require a unique rooted link tree before retaining articulated metadata."""

    if len(link_names) == 1:
        if joints:
            raise ContractError(f"single-link URDF asset {node!r} cannot declare joints")
        return
    if not joints:
        raise ContractError(
            f"multi-link URDF asset {node!r} requires joints that form one rooted tree"
        )
    joint_names: set[str] = set()
    child_to_parent: dict[str, str] = {}
    for joint in joints:
        name = joint["name"]
        parent = joint["parent_link"]
        child = joint["child_link"]
        if name in joint_names:
            raise ContractError(f"duplicate URDF joint name in asset {node!r}: {name!r}")
        joint_names.add(name)
        if parent not in link_names or child not in link_names:
            raise ContractError(
                f"URDF joint references an unknown link in asset {node!r}"
            )
        if parent == child:
            raise ContractError(f"URDF joint in asset {node!r} cannot be self-referential")
        if child in child_to_parent:
            raise ContractError(
                f"URDF link {child!r} has multiple parents in asset {node!r}"
            )
        child_to_parent[child] = parent
    roots = link_names - set(child_to_parent)
    if len(roots) != 1 or len(joints) != len(link_names) - 1:
        raise ContractError(f"URDF asset {node!r} must form exactly one rooted link tree")
    root = next(iter(roots))
    for link in link_names:
        seen: set[str] = set()
        cursor = link
        while cursor != root:
            if cursor in seen or cursor not in child_to_parent:
                raise ContractError(
                    f"URDF asset {node!r} link topology is cyclic or disconnected"
                )
            seen.add(cursor)
            cursor = child_to_parent[cursor]


def _normalize_affordance(payload: dict[str, Any], *, where: str) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    affordances = result.get("affordances")
    if not isinstance(affordances, list):
        raise ContractError(f"{where}.affordances must be an array")
    seen_ids: set[int] = set()
    for index, item in enumerate(affordances):
        if not isinstance(item, dict):
            raise ContractError(f"{where}.affordances[{index}] must be an object")
        part_id = item.get("id")
        if isinstance(part_id, bool) or not isinstance(part_id, int) or part_id < 0:
            raise ContractError(f"{where}.affordances[{index}].id must be a nonnegative integer")
        if part_id in seen_ids:
            raise ContractError(f"{where} has duplicate affordance id {part_id}")
        seen_ids.add(part_id)
        groups = item.get("grasp_group", {})
        if not isinstance(groups, dict):
            raise ContractError(f"{where}.affordances[{index}].grasp_group must be an object")
        for grasp_name, grasp in groups.items():
            if not isinstance(grasp_name, str) or not isinstance(grasp, dict):
                raise ContractError(f"{where} grasp records must be named objects")
            position = require_vector(grasp.get("position"), length=3, where=f"{where}.{grasp_name}.position")
            orientation = require_mapping(grasp.get("orientation"), where=f"{where}.{grasp_name}.orientation")
            require_exact_keys(orientation, required={"w", "xyz"}, where=f"{where}.{grasp_name}.orientation")
            xyz = require_vector(orientation["xyz"], length=3, where=f"{where}.{grasp_name}.orientation.xyz")
            w = require_finite_number(orientation["w"], where=f"{where}.{grasp_name}.orientation.w")
            xyzw, changed = canonicalize_quaternion_xyzw(
                [*xyz, w], where=f"{where}.{grasp_name}.orientation", maximum_norm_error=1e-3
            )
            grasp["position"] = position
            grasp["orientation_source_wxyz"] = {"w": w, "xyz": xyz}
            grasp["orientation_xyzw"] = xyzw
            grasp["orientation_canonicalized"] = changed
            del grasp["orientation"]
    return result


def _parse_asset(
    root: Path,
    node: str,
    asset_directory: str,
    files: _Files,
    warnings: list[str],
) -> tuple[str, dict[str, Any], list[str]]:
    if any(mark in node for mark in ("/", "\\", "\x00")):
        raise ContractError(f"physical node key cannot be used as an asset filename: {node!r}")
    urdf_path = _declared_child_path(
        root, asset_directory, f"{node.replace(' ', '_')}.urdf", where=f"assets[{node!r}]"
    )
    urdf_record = files.add(urdf_path, "urdf")
    raw = urdf_path.read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ContractError(f"URDF must not contain DTD/entity declarations: {urdf_record['path']}")
    try:
        root_xml = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ContractError(f"invalid URDF XML: {urdf_record['path']}: {exc}") from exc
    if root_xml.tag != "robot":
        raise ContractError(f"URDF root must be <robot>: {urdf_record['path']}")
    links_xml = root_xml.findall("./link")
    if not links_xml:
        raise ContractError(f"URDF must contain at least one link: {urdf_record['path']}")
    links: list[dict[str, Any]] = []
    blockers: list[str] = []
    consumed_hashes = {urdf_record["sha256"]}
    visual_count = collision_count = 0
    link_names: set[str] = set()
    categories: set[str] = set()
    for link_index, link in enumerate(links_xml):
        where = f"URDF {node!r}.links[{link_index}]"
        link_name = require_nonempty_string(link.get("name"), where=f"{where}.name")
        if link_name in link_names:
            raise ContractError(f"duplicate URDF link name: {link_name!r}")
        link_names.add(link_name)
        visuals = [
            _mesh_geometry(
                visual,
                root=root,
                urdf_relative=urdf_record["path"],
                files=files,
                role="visual_mesh",
                where=f"{where}.visuals[{index}]",
            )
            for index, visual in enumerate(link.findall("./visual"))
        ]
        collisions = []
        for index, collision in enumerate(link.findall("./collision")):
            collision_where = f"{where}.collisions[{index}]"
            geometry = _mesh_geometry(
                collision,
                root=root,
                urdf_relative=urdf_record["path"],
                files=files,
                role="collision_mesh",
                where=collision_where,
            )
            geometry["material"] = _collision_material(collision, where=f"{collision_where}.material")
            collisions.append(geometry)
        visual_count += len(visuals)
        collision_count += len(collisions)
        inertial = _inertial(link, where=where)
        category_element = link.find("./extra_info/category")
        category = None
        if category_element is not None and category_element.text is not None:
            category = require_nonempty_string(
                category_element.text, where=f"{where}.extra_info.category"
            )
            categories.add(category)
        links.append(
            {
                "name": link_name,
                "category": category,
                "visuals": visuals,
                "collisions": collisions,
                "inertial": inertial,
            }
        )
        if inertial["mass_kg"] is None:
            blockers.append(f"{node}:{link_name}:missing_mass")
        if inertial["inertia_kg_m2"] is None:
            blockers.append(f"{node}:{link_name}:missing_inertia")
        for collision_index, collision in enumerate(collisions):
            material = collision["material"]
            if material["source_mu1"] is None:
                blockers.append(f"{node}:{link_name}:collision_{collision_index}:missing_friction")
            blockers.append(f"{node}:{link_name}:collision_{collision_index}:missing_restitution")
        for geometry in [*visuals, *collisions]:
            consumed_hashes.add(geometry["sha256"])
            consumed_hashes.update(geometry["dependency_sha256"])
    if visual_count == 0 or collision_count == 0:
        raise ContractError(f"non-background asset {node!r} requires visual and collision meshes")
    if not categories:
        warnings.append(f"asset {node!r} has no source semantic category")
    elif len(categories) > 1:
        warnings.append(
            f"asset {node!r} has multiple link categories: {sorted(categories)}"
        )
    joints = [
        _joint_record(joint, where=f"URDF {node!r}.joints[{index}]")
        for index, joint in enumerate(root_xml.findall("./joint"))
    ]
    _validate_joint_topology(link_names, joints, node=node)

    affordance = None
    custom = root_xml.find("./custom_data/affordance")
    annot_text = None if custom is None else custom.findtext("./affordance_annot")
    seg_mesh_xml = None if custom is None else custom.find("./visual_seg/geometry/mesh")
    has_annotation = bool(annot_text and annot_text.strip())
    has_segmentation = bool(seg_mesh_xml is not None and seg_mesh_xml.get("filename"))
    if has_annotation != has_segmentation:
        raise ContractError(
            f"asset {node!r} has a partial affordance declaration; annotation and segmentation are required together"
        )
    if has_annotation and has_segmentation:
        annot_path = _urdf_reference(root, urdf_record["path"], annot_text.strip(), where=f"{node}.affordance_annot")
        seg_path = _urdf_reference(root, urdf_record["path"], seg_mesh_xml.get("filename"), where=f"{node}.visual_seg")
        annot_record = files.add(annot_path, "affordance_annotation")
        seg_record = files.add(seg_path, "affordance_segmentation")
        consumed_hashes.update((annot_record["sha256"], seg_record["sha256"]))
        affordance = {
            "annotation_file": annot_record["path"],
            "annotation_sha256": annot_record["sha256"],
            "segmentation_file": seg_record["path"],
            "segmentation_sha256": seg_record["sha256"],
            "semantics": _normalize_affordance(read_json(annot_path), where=f"affordance {node!r}"),
            "grasp_frame": "asset_root_link_after_visual_origin_and_scale",
            "simulation_validation": "source_asserted_not_revalidated",
        }
    else:
        warnings.append(f"asset {node!r} has no complete optional affordance annotation")
    asset_id = "asset_" + content_sha256(sorted(consumed_hashes))[:24]
    return asset_id, {
        "asset_id": asset_id,
        "kind": "urdf",
        "source_node_key": node,
        "category": next(iter(categories)) if len(categories) == 1 else None,
        "link_categories": sorted(categories),
        "source_urdf": urdf_record["path"],
        "source_urdf_sha256": urdf_record["sha256"],
        "links": links,
        "joints": joints,
        "affordance": affordance,
        "consumed_file_sha256": sorted(consumed_hashes),
    }, blockers


def _layout_graph(layout: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str | None], str | None]:
    relation = require_mapping(layout["relation"], where="layout.relation")
    require_exact_keys(
        relation,
        required=set(_ROLE_KEYS),
        optional={"robot", "task", "task_desc"},
        where="layout.relation",
    )
    for metadata_key in ("task", "task_desc"):
        if metadata_key in relation:
            require_nonempty_string(
                relation[metadata_key],
                where=f"layout.relation.{metadata_key}",
            )
    background = require_nonempty_string(relation["background"], where="layout.relation.background")
    context = require_nonempty_string(relation["context"], where="layout.relation.context")
    manipulated = relation["manipulated_objs"]
    distractors = relation["distractor_objs"]
    if not isinstance(manipulated, list) or not isinstance(distractors, list):
        raise ContractError("layout manipulated_objs and distractor_objs must be arrays")
    role_members = {
        "background": [background],
        "context": [context],
        "manipulated_objs": [require_nonempty_string(v, where="manipulated node") for v in manipulated],
        "distractor_objs": [require_nonempty_string(v, where="distractor node") for v in distractors],
    }
    flat = [node for key in _ROLE_KEYS for node in role_members[key]]
    if len(flat) != len(set(flat)):
        raise ContractError("physical nodes must partition native relation roles exactly once")
    mapping = require_mapping(layout["objs_mapping"], where="layout.objs_mapping")
    if set(mapping) != set(flat):
        raise ContractError("objs_mapping node set must equal the physical role partition")
    native_role: dict[str, str] = {}
    for key, nodes in role_members.items():
        for node in nodes:
            if mapping[node] != key:
                raise ContractError(f"objs_mapping role disagrees with relation for {node!r}")
            native_role[node] = key

    robot = relation.get("robot")
    if robot is not None:
        robot = require_nonempty_string(robot, where="layout.relation.robot")
        if robot in native_role:
            raise ContractError("native robot must not be a physical node")
    tree = require_mapping(layout["tree"], where="layout.tree")
    allowed = set(flat) | ({robot} if robot is not None else set())
    parent: dict[str, str | None] = {node: None for node in allowed}
    edge_relation: dict[str, str] = {}
    observed = set(tree)
    for parent_node, children in tree.items():
        if parent_node not in allowed:
            raise ContractError(f"layout.tree contains unknown parent {parent_node!r}")
        if not isinstance(children, list):
            raise ContractError(f"layout.tree[{parent_node!r}] must be an array")
        for index, edge in enumerate(children):
            if not isinstance(edge, list) or len(edge) != 2:
                raise ContractError(f"layout tree edge {parent_node}[{index}] must be [child, relation]")
            child = require_nonempty_string(edge[0], where="tree child")
            spatial = edge[1]
            if child not in allowed or spatial not in _SPATIAL_RELATIONS:
                raise ContractError(f"invalid tree edge {parent_node!r} -> {child!r}")
            if child == parent_node or parent[child] is not None:
                raise ContractError(f"tree node {child!r} must have exactly one parent")
            parent[child] = parent_node
            edge_relation[child] = spatial
            observed.add(child)
    if observed != allowed:
        raise ContractError(f"layout tree node coverage mismatch: {sorted(allowed - observed)}")
    roots = [node for node, value in parent.items() if value is None]
    if roots != [background]:
        raise ContractError("layout tree must have the background as its sole root")
    for node in allowed:
        seen: set[str] = set()
        cursor: str | None = node
        while cursor is not None:
            if cursor in seen:
                raise ContractError("layout tree must be acyclic")
            seen.add(cursor)
            cursor = parent[cursor]
    if parent[context] != background or edge_relation.get(context) != "FLOOR":
        raise ContractError("context must be FLOOR child of background")
    for node in role_members["manipulated_objs"] + role_members["distractor_objs"]:
        placement = (parent[node], edge_relation.get(node))
        if placement not in {(context, "ON"), (context, "INSIDE"), (background, "FLOOR")}:
            raise ContractError(
                f"object {node!r} must be ON/INSIDE context or FLOOR background"
            )
    if robot is not None and (parent[robot] != background or edge_relation.get(robot) != "IN"):
        raise ContractError("native robot must be an IN child of background")
    return native_role, parent, robot


def _target_receipt(
    activity_spec: Mapping[str, Any],
    manipulated: list[str],
    ids: Mapping[str, str],
    layout_sha256: str,
) -> dict[str, Any]:
    request = activity_spec["target_request"]
    rule = request["resolution_rule"]
    method = rule["method"]
    if method == "sole_manipulated_instance":
        if len(manipulated) != 1:
            raise ContractError("sole_manipulated_instance requires exactly one manipulated node")
        chosen = manipulated[0]
    else:
        expected = rule["expected_source_node_key"]
        matches = [node for node in manipulated if node == expected]
        if len(matches) != 1:
            raise ContractError("exact target source key must match exactly one manipulated node")
        chosen = matches[0]
    return {
        "target_request_sha256": content_sha256(request),
        "layout_sha256": layout_sha256,
        "candidate_source_node_keys": sorted(manipulated),
        "chosen_source_node_key": chosen,
        "chosen_instance_id": ids[chosen],
        "method": method,
    }


def _humanoid_pose(activity_spec: Mapping[str, Any], target_pose: Mapping[str, Any]) -> dict[str, Any]:
    placement = activity_spec["actor"]["initial_placement"]
    offset = placement["target_relative_position_m"]
    target = target_pose["translation_m"]
    position = [target[index] + float(offset[index]) for index in range(3)]
    dx, dy = target[0] - position[0], target[1] - position[1]
    if dx == 0.0 and dy == 0.0:
        raise ContractError("humanoid target-relative position must permit a defined facing yaw")
    yaw = math.atan2(dy, dx)
    quaternion = quaternion_xyzw_from_rpy([0.0, 0.0, yaw])
    return {
        "selected_by": "nursery",
        "authority": "stage1_target_relative_placement_not_native_robot_pose",
        "pose_world": {
            "translation_m": position,
            "rotation_xyzw": quaternion,
            "matrix_4x4": transform_matrix(position, quaternion),
        },
        "yaw_rad": yaw,
        "provenance": {
            "activity_spec_sha256": activity_spec_sha256(activity_spec),
            "target_relative_position_m": list(offset),
            "face_target": True,
            "accessibility_gate": "not_performed_by_contract",
        },
    }


def _rigid_geometry_world_transforms(
    asset: Mapping[str, Any], world_instance: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve rigid root-link geometry transforms; flag articulated assets."""

    if asset.get("kind") != "urdf":
        return [], []
    if asset.get("joints"):
        return [], [
            f"{asset['asset_id']}:articulated_link_world_transforms_not_resolved"
        ]
    world_matrix = world_instance["matrix_4x4"]
    records: list[dict[str, Any]] = []
    for link in asset["links"]:
        for geometry_type in ("visuals", "collisions"):
            for index, geometry in enumerate(link[geometry_type]):
                records.append(
                    {
                        "link_name": link["name"],
                        "geometry_type": geometry_type[:-1],
                        "geometry_index": index,
                        "file": geometry["file"],
                        "scale_xyz": geometry["scale_xyz"],
                        "T_world_geometry": multiply_transform_matrices(
                            world_matrix,
                            geometry["T_link_geometry"]["matrix_4x4"],
                        ),
                        "composition": "T_world_instance @ T_link_geometry",
                    }
                )
    return records, []


def _validate_embodiedgen_request(
    value: Mapping[str, Any],
    activity_spec: Mapping[str, Any],
    *,
    upstream_version: str,
    upstream_commit: str,
    native_profile: str,
) -> str:
    request = require_mapping(value, where="embodiedgen_request")
    require_exact_keys(request, required=_REQUEST_KEYS, where="embodiedgen_request")
    rebuilt = build_embodiedgen_request(
        activity_spec,
        upstream_version=upstream_version,
        upstream_commit=upstream_commit,
        background_list=request["background_list"],
    )
    if request != rebuilt:
        raise ContractError("embodiedgen_request does not equal the canonical Stage 1 projection")
    if request["schema"] != "EmbodiedGenRequest" or request["schema_version"] != 1:
        raise ContractError("embodiedgen_request schema/version mismatch")
    if request["protocol_id"] != activity_spec["protocol_id"]:
        raise ContractError("embodiedgen_request protocol_id mismatch")
    if request["activity_id"] != activity_spec["activity_id"]:
        raise ContractError("embodiedgen_request activity_id mismatch")
    if request["activity_spec_sha256"] != activity_spec_sha256(activity_spec):
        raise ContractError("embodiedgen_request activity hash mismatch")
    if request["prompt"] != activity_spec["prompt"]:
        raise ContractError("embodiedgen_request prompt mismatch")
    upstream = require_mapping(request["upstream"], where="embodiedgen_request.upstream")
    require_exact_keys(upstream, required={"version", "commit"}, where="embodiedgen_request.upstream")
    if upstream != {"version": upstream_version, "commit": upstream_commit}:
        raise ContractError("embodiedgen_request upstream mismatch")
    if request["native_profile"] != native_profile:
        raise ContractError("embodiedgen_request native_profile mismatch")
    if request["render_insert_robot"] is not False:
        raise ContractError("embodiedgen_request.render_insert_robot must be false")
    if request["native_robot_pose_role"] != (
        "may_be_present_for_embodiedgen_layout_placement_but_is_not_"
        "humanoid_authority"
    ):
        raise ContractError("embodiedgen_request native robot role mismatch")
    # The Stage 1 builder validates the remaining exact request subcontracts.
    if not isinstance(request["background_list"], list) or not request["background_list"]:
        raise ContractError("embodiedgen_request.background_list must be non-empty")
    if not isinstance(request["seeds"], Mapping) or not isinstance(request["retry_limits"], Mapping):
        raise ContractError("embodiedgen_request seeds/retry_limits must be objects")
    if not isinstance(request["expected_outputs"], list):
        raise ContractError("embodiedgen_request.expected_outputs must be an array")
    return content_sha256(request)


def compile_scene_bundle(
    scene_root: Path,
    layout_relative_path: str,
    activity_spec: Mapping[str, Any],
    *,
    upstream_version: str,
    upstream_commit: str,
    native_profile: str = NATIVE_PROFILE,
    embodiedgen_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly ingest one native layout and compile a deterministic SceneBundle."""

    validate_activity_spec(activity_spec)
    if native_profile != NATIVE_PROFILE:
        raise ContractError(f"native_profile must be exactly {NATIVE_PROFILE!r}")
    upstream_version = require_nonempty_string(upstream_version, where="upstream_version")
    upstream_commit = require_nonempty_string(upstream_commit, where="upstream_commit")
    if upstream_version != EMBODIEDGEN_UPSTREAM_VERSION:
        raise ContractError(
            f"upstream_version must be exactly {EMBODIEDGEN_UPSTREAM_VERSION!r}"
        )
    if upstream_commit != EMBODIEDGEN_UPSTREAM_COMMIT:
        raise ContractError(
            f"upstream_commit must be exactly {EMBODIEDGEN_UPSTREAM_COMMIT!r}"
        )
    request_sha256 = (
        None
        if embodiedgen_request is None
        else _validate_embodiedgen_request(
            embodiedgen_request,
            activity_spec,
            upstream_version=upstream_version,
            upstream_commit=upstream_commit,
            native_profile=native_profile,
        )
    )
    root = Path(scene_root)
    layout_path = safe_relative_file(root, layout_relative_path, where="layout_relative_path")
    layout = read_json(layout_path)
    require_exact_keys(layout, required=_LAYOUT_REQUIRED, optional=_LAYOUT_OPTIONAL, where="layout")
    layout_sha = sha256_file(layout_path)
    files = _Files(root)
    files.add(layout_path, "native_layout")
    native_roles, parents, robot = _layout_graph(layout)
    physical_nodes = set(native_roles)
    descriptions = require_mapping(layout["objs_desc"], where="layout.objs_desc")
    assets_map = require_mapping(layout["assets"], where="layout.assets")
    positions = require_mapping(layout["position"], where="layout.position")
    if set(descriptions) != physical_nodes or set(assets_map) != physical_nodes:
        raise ContractError("every and only physical nodes require descriptions and assets")
    allowed_positions = physical_nodes | ({robot} if robot is not None else set())
    if set(positions) != allowed_positions:
        raise ContractError("position keys must be physical nodes plus the optional native robot")
    poses: dict[str, dict[str, Any]] = {}
    ids: dict[str, str] = {}
    if len({_instance_id(node) for node in physical_nodes}) != len(physical_nodes):
        raise ContractError("deterministic instance ID collision")
    for node in sorted(physical_nodes):
        require_nonempty_string(descriptions[node], where=f"objs_desc[{node!r}]")
        ids[node] = _instance_id(node)
        poses[node] = _pose_from_layout(positions[node], where=f"position[{node!r}]")
        poses[node]["child_frame"] = f"instance/{ids[node]}"
    ignored_robot_pose = None
    if robot is not None:
        ignored_robot_pose = {
            "source_node_key": robot,
            "pose": _pose_from_layout(positions[robot], where=f"position[{robot!r}]"),
            "authority": "ignored_not_humanoid_authority",
        }

    manipulated = sorted(node for node, role in native_roles.items() if role == "manipulated_objs")
    receipt = _target_receipt(activity_spec, manipulated, ids, layout_sha)
    target_node = receipt["chosen_source_node_key"]
    warnings: list[str] = []
    deferred = [
        "background_room_collision_readiness",
        "collision_mesh_import_and_decomposition",
        "collision_visual_alignment",
        "initial_interpenetration_and_support_contacts",
        "mesh_content_watertightness_and_convexity",
        "physics_settling_and_final_velocities",
        "source_asserted_affordance_grasp_revalidation",
    ]
    asset_records: dict[str, dict[str, Any]] = {}
    node_assets: dict[str, str] = {}
    physics_blockers: list[str] = []
    for node in sorted(physical_nodes - {next(n for n, r in native_roles.items() if r == "background")}):
        asset_id, asset, blockers = _parse_asset(
            root, node, assets_map[node], files, warnings
        )
        if asset_id in asset_records and asset_records[asset_id] != asset:
            raise ContractError(f"content-identical asset collision has inconsistent records: {asset_id}")
        asset_records[asset_id] = asset
        node_assets[node] = asset_id
        if native_roles[node] in {"manipulated_objs", "distractor_objs"}:
            physics_blockers.extend(blockers)

    background_node = next(node for node, role in native_roles.items() if role == "background")
    background_relative = PurePosixPath(
        require_nonempty_string(assets_map[background_node], where="background asset")
    )
    reference_record = None
    reference_relative = (background_relative / "mesh_model.ply").as_posix()
    reference_candidate = root.joinpath(*PurePosixPath(reference_relative).parts)
    if reference_candidate.exists() or reference_candidate.is_symlink():
        reference_path = safe_relative_file(
            root, reference_relative, where="background mesh_model.ply"
        )
        reference_record = files.add(reference_path, "background_reference_mesh")
    else:
        warnings.append(
            "background has no mesh_model.ply reference mesh; Gaussian geometry "
            "must not be used as collision"
        )
    gs_record = None
    gs_relative = (background_relative / "gs_model.ply").as_posix()
    candidate = root.joinpath(*PurePosixPath(gs_relative).parts)
    if candidate.exists() or candidate.is_symlink():
        gs_path = safe_relative_file(root, gs_relative, where="background gs_model.ply")
        gs_record = files.add(gs_path, "background_gaussian_render")
    else:
        warnings.append("background has no optional gs_model.ply Gaussian representation")
    if reference_record is None and gs_record is None:
        raise ContractError(
            "background requires at least one of mesh_model.ply or gs_model.ply"
        )
    bg_hashes = (
        [] if reference_record is None else [reference_record["sha256"]]
    ) + ([] if gs_record is None else [gs_record["sha256"]])
    background_asset_id = "asset_" + content_sha256(sorted(bg_hashes))[:24]
    asset_records[background_asset_id] = {
        "asset_id": background_asset_id,
        "kind": "background_reference",
        "source_node_key": background_node,
        "reference_mesh": (
            None if reference_record is None else reference_record["path"]
        ),
        "reference_mesh_sha256": (
            None if reference_record is None else reference_record["sha256"]
        ),
        "gaussian_model": None if gs_record is None else gs_record["path"],
        "gaussian_model_sha256": None if gs_record is None else gs_record["sha256"],
        "collision_ready": False,
        "consumed_file_sha256": sorted(bg_hashes),
    }
    node_assets[background_node] = background_asset_id

    target_source_category = asset_records[node_assets[target_node]].get("category")
    target_category_match = (
        isinstance(target_source_category, str)
        and target_source_category.casefold()
        == activity_spec["target_request"]["category"].casefold()
    )
    receipt["source_category"] = target_source_category
    receipt["requested_category"] = activity_spec["target_request"]["category"]
    receipt["category_exact_match"] = target_category_match
    target_blockers = []
    if target_source_category is None:
        target_blockers.append("target:source_category_missing")
    elif not target_category_match:
        target_blockers.append("target:source_category_mismatch")

    instance_roles = {
        "background": "background",
        "context": "support",
        "distractor_objs": "distractor",
    }
    instances = []
    scene_conversion_blockers: list[str] = []
    for node in sorted(physical_nodes, key=lambda item: ids[item]):
        native_role = native_roles[node]
        role = (
            "target"
            if node == target_node
            else "other_manipulated"
            if native_role == "manipulated_objs"
            else instance_roles[native_role]
        )
        parent = parents[node]
        spatial = None
        if parent is not None:
            for child, relation_name in require_mapping(layout["tree"], where="layout.tree")[parent]:
                if child == node:
                    spatial = relation_name
                    break
        geometry_world_transforms, geometry_blockers = (
            _rigid_geometry_world_transforms(
                asset_records[node_assets[node]], poses[node]
            )
        )
        scene_conversion_blockers.extend(geometry_blockers)
        instances.append(
            {
                "instance_id": ids[node],
                "source_node_key": node,
                "description": descriptions[node],
                "category": asset_records[node_assets[node]].get("category"),
                "role": role,
                "native_role": native_role,
                "body_type": "static" if native_role in {"background", "context"} else "dynamic",
                "asset_id": node_assets[node],
                "parent_instance_id": None if parent is None or parent == robot else ids[parent],
                "spatial_relation": spatial,
                "initial_pose_world": poses[node],
                "geometry_world_transforms": geometry_world_transforms,
            }
        )
    if reference_record is not None:
        warnings.append(
            "background mesh_model.ply is reference/render geometry, not "
            "validated collision geometry"
        )
    blockers = sorted(
        set(
            physics_blockers
            + target_blockers
            + scene_conversion_blockers
            + ["environment:background_room_collision_not_ready"]
            + (
                ["environment:background_reference_mesh_not_ready"]
                if reference_record is None
                else []
            )
            + ([] if request_sha256 is not None else ["embodiedgen_request_not_bound"])
        )
    )
    physics_complete = not physics_blockers
    payload: dict[str, Any] = {
        "schema": SCENE_BUNDLE_SCHEMA,
        "schema_version": SCENE_BUNDLE_SCHEMA_VERSION,
        "protocol_id": activity_spec["protocol_id"],
        "bundle_id": "",
        "scene_id": "egscene_" + layout_sha[:24],
        "activity_binding": {
            "activity_id": activity_spec["activity_id"],
            "activity_spec_sha256": activity_spec_sha256(activity_spec),
            "embodiedgen_request_sha256": request_sha256,
            "actor": copy.deepcopy(activity_spec["actor"]),
        },
        "source": {
            "format": "EmbodiedGenV2",
            "upstream_version": upstream_version,
            "upstream_commit": upstream_commit,
            "native_profile": native_profile,
            "embodiedgen_request_sha256": request_sha256,
            "layout": {"path": posix_relative_path(root, layout_path), "sha256": layout_sha},
            "ignored_native_robot_pose": ignored_robot_pose,
            "native_quality": copy.deepcopy(layout.get("quality")),
        },
        "conventions": copy.deepcopy(_CONVENTIONS),
        "files": files.values(),
        "assets": {key: asset_records[key] for key in sorted(asset_records)},
        "instances": instances,
        "environment": {
            "background_instance_id": ids[background_node],
            "reference_mesh": (
                None if reference_record is None else reference_record["path"]
            ),
            "gaussian_model": None if gs_record is None else gs_record["path"],
            "ground_plane": {"z_m": 0.0, "provenance": "nursery_frozen_scene_policy"},
            "background_geometry_role": "render_and_reference_only_not_collision",
        },
        "target_request": copy.deepcopy(activity_spec["target_request"]),
        "target_resolution_receipt": receipt,
        "humanoid_initial_pose": _humanoid_pose(activity_spec, poses[target_node]),
        "capabilities": {
            "interactmove_scene_input_ready": (
                request_sha256 is not None
                and target_category_match
                and not scene_conversion_blockers
                and reference_record is not None
            ),
            "target_category_exact_match": target_category_match,
            "physics_material_complete": physics_complete,
            "dynamic_settle_ready": physics_complete and request_sha256 is not None,
            "gaussian_render_ready": gs_record is not None,
            "reference_mesh_ready": reference_record is not None,
            "floor_contact_ready": True,
            "room_collision_ready": False,
            "blockers": blockers,
        },
        "validation": {
            "offline_conversion": "passed",
            "warnings": sorted(set(warnings)),
            "deferred_simulator_checks": sorted(deferred),
            "mesh_content_validation_performed": False,
            "humanoid_accessibility_check_performed": False,
        },
        "settling": {"status": "not_run", "receipt": None},
    }
    payload["bundle_id"] = "scene_" + content_sha256({**payload, "bundle_id": ""})[:32]
    validate_scene_bundle(payload)
    return payload


def _require_sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _validate_matrix_4x4(value: Any, *, where: str) -> list[list[float]]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise ContractError(f"{where} must be a 4x4 numeric matrix")
    return [
        require_vector(row, length=4, where=f"{where}[{index}]")
        for index, row in enumerate(value)
    ]


def _validate_pose_record(
    value: Any,
    *,
    where: str,
    expected_child_frame: str | None,
) -> Mapping[str, Any]:
    pose = require_mapping(value, where=where)
    require_exact_keys(
        pose,
        required={
            "parent_frame",
            "child_frame",
            "translation_m",
            "rotation_xyzw",
            "matrix_4x4",
            "quaternion_canonicalized",
            "provenance",
        },
        where=where,
    )
    if pose["parent_frame"] != "world" or pose["child_frame"] != expected_child_frame:
        raise ContractError(f"{where} has invalid frame labels")
    translation = require_vector(
        pose["translation_m"], length=3, where=f"{where}.translation_m"
    )
    quaternion, _ = canonicalize_quaternion_xyzw(
        pose["rotation_xyzw"],
        where=f"{where}.rotation_xyzw",
        maximum_norm_error=1e-9,
    )
    if quaternion != pose["rotation_xyzw"]:
        raise ContractError(f"{where}.rotation_xyzw is not canonical")
    matrix = _validate_matrix_4x4(pose["matrix_4x4"], where=f"{where}.matrix_4x4")
    if matrix != transform_matrix(translation, quaternion):
        raise ContractError(f"{where}.matrix_4x4 disagrees with its pose")
    if not isinstance(pose["quaternion_canonicalized"], bool):
        raise ContractError(f"{where}.quaternion_canonicalized must be Boolean")
    provenance = require_mapping(pose["provenance"], where=f"{where}.provenance")
    require_exact_keys(
        provenance,
        required={"source", "source_pointer", "source_order", "operation"},
        where=f"{where}.provenance",
    )
    if provenance["source"] != "layout.json" or provenance["source_order"] != (
        "[x,y,z,qx,qy,qz,qw]"
    ):
        raise ContractError(f"{where}.provenance does not identify the native layout pose")
    require_nonempty_string(
        provenance["source_pointer"], where=f"{where}.provenance.source_pointer"
    )
    require_nonempty_string(
        provenance["operation"], where=f"{where}.provenance.operation"
    )
    return pose


def _validate_local_origin(value: Any, *, where: str) -> Mapping[str, Any]:
    origin = require_mapping(value, where=where)
    require_exact_keys(
        origin,
        required={
            "translation_m",
            "rpy_rad",
            "rotation_xyzw",
            "matrix_4x4",
            "used_urdf_default",
            "source_pointer",
        },
        where=where,
    )
    translation = require_vector(
        origin["translation_m"], length=3, where=f"{where}.translation_m"
    )
    rpy = require_vector(origin["rpy_rad"], length=3, where=f"{where}.rpy_rad")
    quaternion = quaternion_xyzw_from_rpy(rpy)
    if origin["rotation_xyzw"] != quaternion:
        raise ContractError(f"{where}.rotation_xyzw disagrees with rpy_rad")
    matrix = _validate_matrix_4x4(origin["matrix_4x4"], where=f"{where}.matrix_4x4")
    if matrix != transform_matrix(translation, quaternion):
        raise ContractError(f"{where}.matrix_4x4 disagrees with its origin")
    if not isinstance(origin["used_urdf_default"], bool):
        raise ContractError(f"{where}.used_urdf_default must be Boolean")
    require_nonempty_string(origin["source_pointer"], where=f"{where}.source_pointer")
    return origin


def validate_scene_bundle(value: Mapping[str, Any]) -> None:
    """Validate deterministic Stage 3 invariants without claiming simulation QA."""

    bundle = require_mapping(value, where="scene bundle")
    require_exact_keys(bundle, required=_TOP_LEVEL, where="scene bundle")
    if bundle["schema"] != SCENE_BUNDLE_SCHEMA or bundle["schema_version"] != 1:
        raise ContractError("invalid SceneBundle schema/version")
    if bundle["protocol_id"] != "interactmove_intermimic":
        raise ContractError("SceneBundle protocol_id must be 'interactmove_intermimic'")
    if bundle["conventions"] != _CONVENTIONS:
        raise ContractError("SceneBundle conventions do not match the frozen contract")
    expected = "scene_" + content_sha256({**bundle, "bundle_id": ""})[:32]
    if bundle["bundle_id"] != expected:
        raise ContractError("SceneBundle bundle_id does not match canonical content")
    source = require_mapping(bundle["source"], where="source")
    require_exact_keys(
        source,
        required={
            "format",
            "upstream_version",
            "upstream_commit",
            "native_profile",
            "embodiedgen_request_sha256",
            "layout",
            "ignored_native_robot_pose",
            "native_quality",
        },
        where="source",
    )
    if source["format"] != "EmbodiedGenV2":
        raise ContractError("SceneBundle source.format must be 'EmbodiedGenV2'")
    if source["upstream_version"] != EMBODIEDGEN_UPSTREAM_VERSION:
        raise ContractError("SceneBundle has unsupported EmbodiedGen version")
    if source["upstream_commit"] != EMBODIEDGEN_UPSTREAM_COMMIT:
        raise ContractError("SceneBundle has unsupported EmbodiedGen commit")
    if source["native_profile"] != NATIVE_PROFILE:
        raise ContractError("SceneBundle has unsupported native profile")
    ignored = source["ignored_native_robot_pose"]
    if ignored is not None:
        ignored = require_mapping(ignored, where="source.ignored_native_robot_pose")
        require_exact_keys(
            ignored,
            required={"source_node_key", "pose", "authority"},
            where="source.ignored_native_robot_pose",
        )
        require_nonempty_string(
            ignored["source_node_key"],
            where="source.ignored_native_robot_pose.source_node_key",
        )
        if ignored["authority"] != "ignored_not_humanoid_authority":
            raise ContractError("native robot pose must not have humanoid authority")
        _validate_pose_record(
            ignored["pose"],
            where="source.ignored_native_robot_pose.pose",
            expected_child_frame=None,
        )
    activity = require_mapping(bundle["activity_binding"], where="activity_binding")
    require_exact_keys(
        activity,
        required={
            "activity_id",
            "activity_spec_sha256",
            "embodiedgen_request_sha256",
            "actor",
        },
        where="activity_binding",
    )
    require_nonempty_string(activity["activity_id"], where="activity_binding.activity_id")
    _require_sha256(
        activity["activity_spec_sha256"],
        where="activity_binding.activity_spec_sha256",
    )
    actor = require_mapping(activity["actor"], where="activity_binding.actor")
    validate_actor_contract(actor)
    files = bundle["files"]
    if not isinstance(files, list):
        raise ContractError("files must be an array")
    file_paths: list[str] = []
    file_hashes: set[str] = set()
    file_by_path: dict[str, Mapping[str, Any]] = {}
    for index, record_raw in enumerate(files):
        record = require_mapping(record_raw, where=f"files[{index}]")
        require_exact_keys(
            record,
            required={"path", "bytes", "sha256", "roles"},
            where=f"files[{index}]",
        )
        path = require_nonempty_string(record["path"], where=f"files[{index}].path")
        pure_path = PurePosixPath(path)
        if (
            pure_path.is_absolute()
            or pure_path.as_posix() != path
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or "\\" in path
            or ":" in path
        ):
            raise ContractError(f"files[{index}].path must be a safe relative POSIX path")
        digest = _require_sha256(record["sha256"], where=f"files[{index}].sha256")
        byte_count = record["bytes"]
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ContractError(f"files[{index}].bytes must be a positive integer")
        roles = record["roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or roles != sorted(set(roles))
            or any(not isinstance(role, str) or not role for role in roles)
        ):
            raise ContractError(f"files[{index}].roles must be sorted unique names")
        file_paths.append(path)
        file_hashes.add(digest)
        file_by_path[path] = record
    if file_paths != sorted(set(file_paths)):
        raise ContractError("file paths must be sorted and unique")
    file_set = set(file_paths)
    assets = require_mapping(bundle["assets"], where="assets")
    for asset_id, asset_raw in assets.items():
        if not isinstance(asset_id, str) or not asset_id.startswith("asset_"):
            raise ContractError("asset IDs must start with 'asset_'")
        asset = require_mapping(asset_raw, where=f"assets[{asset_id}]")
        if asset.get("asset_id") != asset_id:
            raise ContractError(f"asset key/id mismatch: {asset_id}")
        consumed = asset.get("consumed_file_sha256")
        if (
            not isinstance(consumed, list)
            or consumed != sorted(set(consumed))
            or any(not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in consumed)
            or not set(consumed).issubset(file_hashes)
        ):
            raise ContractError(f"asset {asset_id} has invalid consumed file hashes")
        if asset_id != "asset_" + content_sha256(consumed)[:24]:
            raise ContractError(f"asset {asset_id} does not match its consumed content hashes")
        if asset.get("kind") == "background_reference":
            require_exact_keys(
                asset,
                required={
                    "asset_id",
                    "kind",
                    "source_node_key",
                    "reference_mesh",
                    "reference_mesh_sha256",
                    "gaussian_model",
                    "gaussian_model_sha256",
                    "collision_ready",
                    "consumed_file_sha256",
                },
                where=f"asset {asset_id}",
            )
            reference = asset["reference_mesh"]
            reference_sha = asset["reference_mesh_sha256"]
            if reference is None:
                if reference_sha is not None:
                    raise ContractError(
                        f"asset {asset_id} has a reference hash without a file"
                    )
            elif (
                reference not in file_by_path
                or reference_sha != file_by_path[reference]["sha256"]
            ):
                raise ContractError(f"asset {asset_id} has an invalid reference mesh binding")
            gaussian = asset["gaussian_model"]
            gaussian_sha = asset["gaussian_model_sha256"]
            if gaussian is None:
                if gaussian_sha is not None:
                    raise ContractError(f"asset {asset_id} has a Gaussian hash without a file")
            elif gaussian not in file_by_path or gaussian_sha != file_by_path[gaussian]["sha256"]:
                raise ContractError(f"asset {asset_id} has an invalid Gaussian file binding")
            if reference is None and gaussian is None:
                raise ContractError(
                    f"asset {asset_id} has no background render representation"
                )
            if asset["collision_ready"] is not False:
                raise ContractError("background reference geometry cannot claim collision readiness")
            require_nonempty_string(
                asset["source_node_key"],
                where=f"asset {asset_id}.source_node_key",
            )
        elif asset.get("kind") == "urdf":
            require_exact_keys(
                asset,
                required={
                    "asset_id",
                    "kind",
                    "source_node_key",
                    "category",
                    "link_categories",
                    "source_urdf",
                    "source_urdf_sha256",
                    "links",
                    "joints",
                    "affordance",
                    "consumed_file_sha256",
                },
                where=f"asset {asset_id}",
            )
            source_urdf = asset["source_urdf"]
            if source_urdf not in file_by_path or asset["source_urdf_sha256"] != file_by_path[source_urdf]["sha256"]:
                raise ContractError(f"asset {asset_id} has an invalid URDF binding")
            require_nonempty_string(
                asset["source_node_key"],
                where=f"asset {asset_id}.source_node_key",
            )
            links_raw = asset["links"]
            if not isinstance(links_raw, list) or not links_raw:
                raise ContractError(f"asset {asset_id}.links must be a non-empty array")
            serialized_link_names: set[str] = set()
            serialized_categories: set[str] = set()
            for link_index, link_raw in enumerate(links_raw):
                link = require_mapping(link_raw, where=f"asset {asset_id}.links[{link_index}]")
                require_exact_keys(
                    link,
                    required={"name", "category", "visuals", "collisions", "inertial"},
                    where=f"asset {asset_id}.links[{link_index}]",
                )
                link_name = require_nonempty_string(
                    link["name"], where=f"asset {asset_id}.links[{link_index}].name"
                )
                if link_name in serialized_link_names:
                    raise ContractError(f"asset {asset_id} has duplicate link names")
                serialized_link_names.add(link_name)
                if link["category"] is not None:
                    serialized_categories.add(
                        require_nonempty_string(
                            link["category"],
                            where=f"asset {asset_id}.links[{link_index}].category",
                        )
                    )
                for geometry_type in ("visuals", "collisions"):
                    geometries = link[geometry_type]
                    if not isinstance(geometries, list):
                        raise ContractError(f"asset {asset_id}.{geometry_type} must be an array")
                    for geometry_index, geometry_raw in enumerate(geometries):
                        geometry = require_mapping(
                            geometry_raw,
                            where=f"asset {asset_id}.{geometry_type}[{geometry_index}]",
                        )
                        required_geometry = {
                            "file",
                            "sha256",
                            "dependency_sha256",
                            "scale_xyz",
                            "scale_used_urdf_default",
                            "T_link_geometry",
                        }
                        if geometry_type == "collisions":
                            required_geometry.add("material")
                        require_exact_keys(
                            geometry,
                            required=required_geometry,
                            where=f"asset {asset_id}.{geometry_type}[{geometry_index}]",
                        )
                        geometry_file = geometry["file"]
                        if geometry_file not in file_by_path or geometry["sha256"] != file_by_path[geometry_file]["sha256"]:
                            raise ContractError(f"asset {asset_id} geometry has an invalid file binding")
                        dependencies = geometry["dependency_sha256"]
                        if (
                            not isinstance(dependencies, list)
                            or dependencies != sorted(set(dependencies))
                            or not set(dependencies).issubset(file_hashes)
                        ):
                            raise ContractError(f"asset {asset_id} geometry dependencies are invalid")
                        require_vector(
                            geometry["scale_xyz"],
                            length=3,
                            positive=True,
                            where=f"asset {asset_id} geometry scale_xyz",
                        )
                        if not isinstance(geometry["scale_used_urdf_default"], bool):
                            raise ContractError(f"asset {asset_id} geometry scale provenance must be Boolean")
                        _validate_local_origin(
                            geometry["T_link_geometry"],
                            where=f"asset {asset_id} geometry T_link_geometry",
                        )
                        if geometry_type == "collisions":
                            material = require_mapping(
                                geometry["material"],
                                where=f"asset {asset_id} collision material",
                            )
                            require_exact_keys(
                                material,
                                required={
                                    "source_mu1",
                                    "source_mu2",
                                    "source_restitution",
                                    "static_friction",
                                    "dynamic_friction",
                                    "restitution",
                                    "source_fields",
                                    "interpretation",
                                },
                                where=f"asset {asset_id} collision material",
                            )
                            mu1 = material["source_mu1"]
                            mu2 = material["source_mu2"]
                            if (mu1 is None) != (mu2 is None):
                                raise ContractError(
                                    f"asset {asset_id} collision friction must be paired"
                                )
                            if mu1 is not None:
                                mu1 = require_finite_number(
                                    mu1, where=f"asset {asset_id} collision source_mu1"
                                )
                                mu2 = require_finite_number(
                                    mu2, where=f"asset {asset_id} collision source_mu2"
                                )
                                if mu1 < 0.0 or mu2 < 0.0:
                                    raise ContractError(
                                        f"asset {asset_id} collision friction must be nonnegative"
                                    )
                            static_friction = material["static_friction"]
                            dynamic_friction = material["dynamic_friction"]
                            if static_friction is not None:
                                static_friction = require_finite_number(
                                    static_friction,
                                    where=f"asset {asset_id} collision static_friction",
                                )
                            if dynamic_friction is not None:
                                dynamic_friction = require_finite_number(
                                    dynamic_friction,
                                    where=f"asset {asset_id} collision dynamic_friction",
                                )
                            if mu1 != static_friction or mu2 != dynamic_friction:
                                raise ContractError(f"asset {asset_id} collision friction fields disagree")
                            if material["source_restitution"] is not None or material["restitution"] is not None:
                                raise ContractError(f"asset {asset_id} cannot synthesize source restitution")
                            source_fields = require_mapping(
                                material["source_fields"],
                                where=f"asset {asset_id} collision material source_fields",
                            )
                            require_exact_keys(
                                source_fields,
                                required={
                                    "static_friction",
                                    "dynamic_friction",
                                    "restitution",
                                },
                                where=f"asset {asset_id} collision material source_fields",
                            )
                            expected_material_sources = {
                                "static_friction": (
                                    None if mu1 is None else "collision/gazebo/mu1"
                                ),
                                "dynamic_friction": (
                                    None if mu2 is None else "collision/gazebo/mu2"
                                ),
                                "restitution": None,
                            }
                            if source_fields != expected_material_sources:
                                raise ContractError(
                                    f"asset {asset_id} collision material provenance is inconsistent"
                                )
                            if material["interpretation"] != "EmbodiedGen_mu1_static_mu2_dynamic":
                                raise ContractError(
                                    f"asset {asset_id} collision material interpretation is invalid"
                                )
                inertial = require_mapping(
                    link["inertial"], where=f"asset {asset_id}.links[{link_index}].inertial"
                )
                require_exact_keys(
                    inertial,
                    required={"mass_kg", "origin", "inertia_kg_m2", "source_fields"},
                    where=f"asset {asset_id}.links[{link_index}].inertial",
                )
                if inertial["mass_kg"] is not None and require_finite_number(
                    inertial["mass_kg"], where=f"asset {asset_id} inertial mass"
                ) <= 0.0:
                    raise ContractError(f"asset {asset_id} inertial mass must be positive")
                if inertial["origin"] is not None:
                    _validate_local_origin(
                        inertial["origin"], where=f"asset {asset_id} inertial origin"
                    )
                if inertial["inertia_kg_m2"] is not None:
                    _validate_inertia_values(
                        inertial["inertia_kg_m2"],
                        where=f"asset {asset_id} inertia_kg_m2",
                    )
                inertial_sources = require_mapping(
                    inertial["source_fields"],
                    where=f"asset {asset_id} inertial source_fields",
                )
                require_exact_keys(
                    inertial_sources,
                    required={"mass_kg", "inertia_kg_m2"},
                    where=f"asset {asset_id} inertial source_fields",
                )
                expected_inertial_sources = {
                    "mass_kg": (
                        None
                        if inertial["mass_kg"] is None
                        else "link/inertial/mass@value"
                    ),
                    "inertia_kg_m2": (
                        None
                        if inertial["inertia_kg_m2"] is None
                        else "link/inertial/inertia@*"
                    ),
                }
                if inertial_sources != expected_inertial_sources:
                    raise ContractError(
                        f"asset {asset_id} inertial provenance is inconsistent"
                    )
            expected_link_categories = sorted(serialized_categories)
            if asset["link_categories"] != expected_link_categories:
                raise ContractError(
                    f"asset {asset_id}.link_categories disagrees with its links"
                )
            expected_asset_category = (
                expected_link_categories[0]
                if len(expected_link_categories) == 1
                else None
            )
            if asset["category"] != expected_asset_category:
                raise ContractError(
                    f"asset {asset_id}.category disagrees with its link categories"
                )
            joints = asset["joints"]
            if not isinstance(joints, list):
                raise ContractError(f"asset {asset_id}.joints must be an array")
            for joint_index, joint_raw in enumerate(joints):
                joint = require_mapping(
                    joint_raw, where=f"asset {asset_id}.joints[{joint_index}]"
                )
                require_exact_keys(
                    joint,
                    required={
                        "name",
                        "type",
                        "parent_link",
                        "child_link",
                        "origin",
                        "axis",
                        "limit",
                    },
                    where=f"asset {asset_id}.joints[{joint_index}]",
                )
                for name in ("name", "type", "parent_link", "child_link"):
                    require_nonempty_string(
                        joint[name],
                        where=f"asset {asset_id}.joints[{joint_index}].{name}",
                    )
                _validate_local_origin(
                    joint["origin"],
                    where=f"asset {asset_id}.joints[{joint_index}].origin",
                )
                if joint["axis"] is not None:
                    require_vector(
                        joint["axis"],
                        length=3,
                        where=f"asset {asset_id}.joints[{joint_index}].axis",
                    )
                if joint["limit"] is not None:
                    limit = require_mapping(
                        joint["limit"],
                        where=f"asset {asset_id}.joints[{joint_index}].limit",
                    )
                    require_exact_keys(
                        limit,
                        required={"lower", "upper", "effort", "velocity"},
                        where=f"asset {asset_id}.joints[{joint_index}].limit",
                    )
                    for name, number in limit.items():
                        if number is not None:
                            require_finite_number(
                                number,
                                where=f"asset {asset_id}.joints[{joint_index}].limit.{name}",
                            )
            _validate_joint_topology(
                serialized_link_names, joints, node=asset["source_node_key"]
            )
            affordance = asset.get("affordance")
            if affordance is not None:
                affordance = require_mapping(affordance, where=f"asset {asset_id}.affordance")
                require_exact_keys(
                    affordance,
                    required={
                        "annotation_file",
                        "annotation_sha256",
                        "segmentation_file",
                        "segmentation_sha256",
                        "semantics",
                        "grasp_frame",
                        "simulation_validation",
                    },
                    where=f"asset {asset_id}.affordance",
                )
                for key in ("annotation_file", "segmentation_file"):
                    if affordance.get(key) not in file_set:
                        raise ContractError(f"asset {asset_id} affordance references an unknown file")
                if affordance["annotation_sha256"] != file_by_path[affordance["annotation_file"]]["sha256"] or affordance["segmentation_sha256"] != file_by_path[affordance["segmentation_file"]]["sha256"]:
                    raise ContractError(f"asset {asset_id} affordance hashes disagree with the manifest")
        else:
            raise ContractError(f"asset {asset_id} has unsupported kind")
    instances = bundle["instances"]
    if not isinstance(instances, list) or not instances:
        raise ContractError("instances must be a non-empty array")
    instance_ids: list[str] = []
    source_keys: list[str] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(instances):
        instance = require_mapping(raw, where=f"instances[{index}]")
        require_exact_keys(
            instance,
            required={
                "instance_id",
                "source_node_key",
                "description",
                "category",
                "role",
                "native_role",
                "body_type",
                "asset_id",
                "parent_instance_id",
                "spatial_relation",
                "initial_pose_world",
                "geometry_world_transforms",
            },
            where=f"instances[{index}]",
        )
        instance_id = require_nonempty_string(instance["instance_id"], where="instance_id")
        source_key = require_nonempty_string(instance["source_node_key"], where="source_node_key")
        require_nonempty_string(instance["description"], where=f"instance {instance_id} description")
        if instance["asset_id"] not in assets:
            raise ContractError(f"instance {instance_id} references an unknown asset")
        if instance_id != _instance_id(source_key):
            raise ContractError(f"instance {instance_id} is not the stable ID of its source key")
        pose = _validate_pose_record(
            instance["initial_pose_world"],
            where=f"instance {instance_id} initial_pose_world",
            expected_child_frame=f"instance/{instance_id}",
        )
        asset = assets[instance["asset_id"]]
        if asset["source_node_key"] != source_key:
            raise ContractError(
                f"instance {instance_id} source key disagrees with its asset"
            )
        if instance["category"] != asset.get("category"):
            raise ContractError(
                f"instance {instance_id} category disagrees with its asset"
            )
        expected_geometry_world, _ = _rigid_geometry_world_transforms(asset, pose)
        geometry_world = instance["geometry_world_transforms"]
        if not isinstance(geometry_world, list):
            raise ContractError(
                f"instance {instance_id} geometry_world_transforms must be an array"
            )
        for geometry_index, geometry_raw in enumerate(geometry_world):
            geometry = require_mapping(
                geometry_raw,
                where=(
                    f"instance {instance_id} geometry_world_transforms"
                    f"[{geometry_index}]"
                ),
            )
            if geometry.get("file") not in file_set:
                raise ContractError(
                    f"instance {instance_id} world geometry references an unknown file"
                )
            matrix = geometry.get("T_world_geometry")
            if (
                not isinstance(matrix, list)
                or len(matrix) != 4
                or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
            ):
                raise ContractError(
                    f"instance {instance_id} world geometry transform must be 4x4"
                )
            for row_index, row in enumerate(matrix):
                require_vector(
                    row,
                    length=4,
                    where=(
                        f"instance {instance_id} geometry matrix row "
                        f"{row_index}"
                    ),
                )
        if geometry_world != expected_geometry_world:
            raise ContractError(
                f"instance {instance_id} world geometry transforms disagree with its asset and pose"
            )
        instance_ids.append(instance_id)
        source_keys.append(source_key)
        by_id[instance_id] = instance
    if len(instance_ids) != len(set(instance_ids)) or len(source_keys) != len(set(source_keys)):
        raise ContractError("instance IDs and source keys must be unique")
    for instance in by_id.values():
        parent_id = instance.get("parent_instance_id")
        if parent_id is not None and parent_id not in by_id:
            raise ContractError("instance parent references an unknown instance")
    receipt = require_mapping(bundle["target_resolution_receipt"], where="target receipt")
    require_exact_keys(
        receipt,
        required={
            "target_request_sha256",
            "layout_sha256",
            "candidate_source_node_keys",
            "chosen_source_node_key",
            "chosen_instance_id",
            "method",
            "source_category",
            "requested_category",
            "category_exact_match",
        },
        where="target receipt",
    )
    target_id = receipt["chosen_instance_id"]
    target_key = receipt["chosen_source_node_key"]
    if target_id not in by_id or by_id[target_id].get("source_node_key") != target_key:
        raise ContractError("target receipt does not identify a continuous instance")
    if by_id[target_id].get("role") != "target":
        raise ContractError("target receipt must resolve to the target-role instance")
    target_request = require_mapping(bundle["target_request"], where="target_request")
    require_exact_keys(
        target_request,
        required={"description", "category", "prompt_span", "resolution_rule"},
        where="target_request",
    )
    validate_target_request_contract(target_request)
    rule = require_mapping(
        target_request["resolution_rule"], where="target_request.resolution_rule"
    )
    require_exact_keys(
        rule,
        required={"method", "expected_source_node_key"},
        where="target_request.resolution_rule",
    )
    if receipt["target_request_sha256"] != content_sha256(target_request):
        raise ContractError("target receipt is not bound to target_request")
    layout_source = require_mapping(source.get("layout"), where="source.layout")
    require_exact_keys(
        layout_source, required={"path", "sha256"}, where="source.layout"
    )
    layout_path = require_nonempty_string(layout_source["path"], where="source.layout.path")
    layout_digest = _require_sha256(layout_source["sha256"], where="source.layout.sha256")
    if layout_path not in file_by_path or file_by_path[layout_path]["sha256"] != layout_digest:
        raise ContractError("source layout is not bound to the file manifest")
    if receipt["layout_sha256"] != layout_digest:
        raise ContractError("target receipt is not bound to the source layout")
    if bundle["scene_id"] != "egscene_" + layout_digest[:24]:
        raise ContractError("scene_id is not bound to the source layout")
    manipulated_keys = sorted(
        item["source_node_key"]
        for item in instances
        if item["native_role"] == "manipulated_objs"
    )
    if receipt["candidate_source_node_keys"] != manipulated_keys:
        raise ContractError("target receipt candidate set disagrees with scene instances")
    if target_key not in manipulated_keys or by_id[target_id]["native_role"] != "manipulated_objs":
        raise ContractError("target receipt must choose one manipulated candidate")
    if receipt["method"] != rule["method"]:
        raise ContractError("target receipt resolution method disagrees with target_request")
    if rule["method"] == "sole_manipulated_instance":
        if len(manipulated_keys) != 1 or target_key != manipulated_keys[0]:
            raise ContractError(
                "sole_manipulated_instance must choose the sole manipulated candidate"
            )
    elif target_key != rule["expected_source_node_key"]:
        raise ContractError(
            "exact_source_node_key receipt does not choose the precommitted key"
        )
    if receipt["requested_category"] != target_request["category"]:
        raise ContractError("target receipt requested category disagrees with target_request")
    if receipt["source_category"] != by_id[target_id]["category"]:
        raise ContractError("target receipt source category disagrees with target instance")
    expected_category_match = (
        isinstance(receipt["source_category"], str)
        and receipt["source_category"].casefold()
        == target_request["category"].casefold()
    )
    if receipt["category_exact_match"] is not expected_category_match:
        raise ContractError("target receipt category_exact_match is inconsistent")
    for instance in instances:
        native_role = instance["native_role"]
        if native_role not in _ROLE_KEYS:
            raise ContractError("instance has an unknown native role")
        expected_body = (
            "static" if native_role in {"background", "context"} else "dynamic"
        )
        if instance["body_type"] != expected_body:
            raise ContractError("instance body_type disagrees with native role")
        expected_role = (
            "target"
            if instance["instance_id"] == target_id
            else "other_manipulated"
            if native_role == "manipulated_objs"
            else {
                "background": "background",
                "context": "support",
                "distractor_objs": "distractor",
            }[native_role]
        )
        if instance["role"] != expected_role:
            raise ContractError("instance role disagrees with target and native role")
    environment = require_mapping(bundle["environment"], where="environment")
    require_exact_keys(
        environment,
        required={
            "background_instance_id",
            "reference_mesh",
            "gaussian_model",
            "ground_plane",
            "background_geometry_role",
        },
        where="environment",
    )
    background_id = environment["background_instance_id"]
    if background_id not in by_id or by_id[background_id].get("role") != "background":
        raise ContractError("environment background_instance_id is invalid")
    background_asset = assets[by_id[background_id]["asset_id"]]
    if environment["reference_mesh"] != background_asset["reference_mesh"] or environment["gaussian_model"] != background_asset["gaussian_model"]:
        raise ContractError("environment render files disagree with the background asset")
    ground = require_mapping(environment["ground_plane"], where="environment.ground_plane")
    require_exact_keys(
        ground, required={"z_m", "provenance"}, where="environment.ground_plane"
    )
    if require_finite_number(ground["z_m"], where="environment.ground_plane.z_m") != 0.0 or ground["provenance"] != "nursery_frozen_scene_policy":
        raise ContractError("environment ground plane does not match the frozen policy")
    if environment["background_geometry_role"] != "render_and_reference_only_not_collision":
        raise ContractError("background geometry cannot be promoted to collision implicitly")
    request_hash = activity.get("embodiedgen_request_sha256")
    if request_hash != source.get("embodiedgen_request_sha256"):
        raise ContractError("EmbodiedGen request bindings disagree")
    capabilities = require_mapping(bundle["capabilities"], where="capabilities")
    require_exact_keys(
        capabilities,
        required={
            "interactmove_scene_input_ready",
            "target_category_exact_match",
            "physics_material_complete",
            "dynamic_settle_ready",
            "gaussian_render_ready",
            "reference_mesh_ready",
            "floor_contact_ready",
            "room_collision_ready",
            "blockers",
        },
        where="capabilities",
    )
    for key in (
        "interactmove_scene_input_ready",
        "target_category_exact_match",
        "physics_material_complete",
        "dynamic_settle_ready",
        "gaussian_render_ready",
        "reference_mesh_ready",
        "floor_contact_ready",
        "room_collision_ready",
    ):
        if not isinstance(capabilities[key], bool):
            raise ContractError(f"capabilities.{key} must be Boolean")
    if capabilities["gaussian_render_ready"] is not (
        environment["gaussian_model"] is not None
    ):
        raise ContractError("gaussian_render_ready disagrees with the environment")
    if capabilities["reference_mesh_ready"] is not (
        environment["reference_mesh"] is not None
    ):
        raise ContractError("reference_mesh_ready disagrees with the environment")
    if capabilities["floor_contact_ready"] is not True:
        raise ContractError("base SceneBundle floor capability is invalid")
    if capabilities["room_collision_ready"] is not False:
        raise ContractError("base SceneBundle cannot claim room collision readiness")
    blockers = capabilities["blockers"]
    if not isinstance(blockers, list) or blockers != sorted(set(blockers)):
        raise ContractError("capability blockers must be sorted and unique")
    if request_hash is not None and (
        not isinstance(request_hash, str) or _SHA256.fullmatch(request_hash) is None
    ):
        raise ContractError("EmbodiedGen request hash is invalid")
    expected_blockers = {"environment:background_room_collision_not_ready"}
    expected_physics_blockers: set[str] = set()
    expected_articulation_blockers: set[str] = set()
    if request_hash is None:
        expected_blockers.add("embodiedgen_request_not_bound")
    if environment["reference_mesh"] is None:
        expected_blockers.add("environment:background_reference_mesh_not_ready")
    if receipt["source_category"] is None:
        expected_blockers.add("target:source_category_missing")
    elif not expected_category_match:
        expected_blockers.add("target:source_category_mismatch")
    for instance in instances:
        asset = assets[instance["asset_id"]]
        if asset["kind"] != "urdf":
            continue
        if asset["joints"]:
            expected_articulation_blockers.add(
                f"{asset['asset_id']}:articulated_link_world_transforms_not_resolved"
            )
        if instance["native_role"] not in {"manipulated_objs", "distractor_objs"}:
            continue
        node = instance["source_node_key"]
        for link in asset["links"]:
            link_name = link["name"]
            inertial = link["inertial"]
            if inertial["mass_kg"] is None:
                expected_physics_blockers.add(f"{node}:{link_name}:missing_mass")
            if inertial["inertia_kg_m2"] is None:
                expected_physics_blockers.add(f"{node}:{link_name}:missing_inertia")
            for collision_index, collision in enumerate(link["collisions"]):
                material = collision["material"]
                if material["source_mu1"] is None:
                    expected_physics_blockers.add(
                        f"{node}:{link_name}:collision_{collision_index}:missing_friction"
                    )
                if material["source_restitution"] is None:
                    expected_physics_blockers.add(
                        f"{node}:{link_name}:collision_{collision_index}:missing_restitution"
                    )
    expected_blockers.update(expected_physics_blockers)
    expected_blockers.update(expected_articulation_blockers)
    if blockers != sorted(expected_blockers):
        raise ContractError("capability blockers disagree with serialized scene state")
    expected_physics_complete = not expected_physics_blockers
    if capabilities["physics_material_complete"] is not expected_physics_complete:
        raise ContractError("physics_material_complete is inconsistent with blockers")
    if capabilities["dynamic_settle_ready"] is not (
        expected_physics_complete and request_hash is not None
    ):
        raise ContractError("dynamic_settle_ready is inconsistent")
    category_match = receipt.get("category_exact_match")
    if not isinstance(category_match, bool):
        raise ContractError("target receipt category_exact_match must be Boolean")
    if capabilities.get("target_category_exact_match") is not category_match:
        raise ContractError("target category capability and receipt disagree")
    expected_interactmove_ready = (
        request_hash is not None
        and category_match
        and not expected_articulation_blockers
        and environment["reference_mesh"] is not None
    )
    if capabilities.get("interactmove_scene_input_ready") is not expected_interactmove_ready:
        raise ContractError("interactmove_scene_input_ready is inconsistent")
    humanoid = require_mapping(bundle["humanoid_initial_pose"], where="humanoid_initial_pose")
    require_exact_keys(
        humanoid,
        required={"selected_by", "authority", "pose_world", "yaw_rad", "provenance"},
        where="humanoid_initial_pose",
    )
    target_pose = by_id[target_id]["initial_pose_world"]
    offset = actor["initial_placement"]["target_relative_position_m"]
    target_translation = target_pose["translation_m"]
    expected_position = [
        target_translation[index] + offset[index] for index in range(3)
    ]
    dx = target_translation[0] - expected_position[0]
    dy = target_translation[1] - expected_position[1]
    expected_yaw = math.atan2(dy, dx)
    expected_quaternion = quaternion_xyzw_from_rpy([0.0, 0.0, expected_yaw])
    expected_humanoid = {
        "selected_by": "nursery",
        "authority": "stage1_target_relative_placement_not_native_robot_pose",
        "pose_world": {
            "translation_m": expected_position,
            "rotation_xyzw": expected_quaternion,
            "matrix_4x4": transform_matrix(expected_position, expected_quaternion),
        },
        "yaw_rad": expected_yaw,
        "provenance": {
            "activity_spec_sha256": activity["activity_spec_sha256"],
            "target_relative_position_m": list(offset),
            "face_target": True,
            "accessibility_gate": "not_performed_by_contract",
        },
    }
    if humanoid != expected_humanoid:
        raise ContractError(
            "humanoid_initial_pose disagrees with the bound actor and target pose"
        )
    validation = require_mapping(bundle["validation"], where="validation")
    require_exact_keys(
        validation,
        required={
            "offline_conversion",
            "warnings",
            "deferred_simulator_checks",
            "mesh_content_validation_performed",
            "humanoid_accessibility_check_performed",
        },
        where="validation",
    )
    if validation["offline_conversion"] != "passed":
        raise ContractError("base SceneBundle offline conversion must have passed")
    for key in ("warnings", "deferred_simulator_checks"):
        items = validation[key]
        if (
            not isinstance(items, list)
            or items != sorted(set(items))
            or any(not isinstance(item, str) or not item for item in items)
        ):
            raise ContractError(f"validation.{key} must be sorted and unique")
    if validation["mesh_content_validation_performed"] is not False:
        raise ContractError("base SceneBundle cannot claim mesh-content validation")
    if validation.get("humanoid_accessibility_check_performed") is not False:
        raise ContractError("the contract forbids a humanoid accessibility gate")
    settling = require_mapping(bundle["settling"], where="settling")
    if settling != {"status": "not_run", "receipt": None}:
        raise ContractError("base SceneBundle must have an unattached settling receipt")


__all__ = [
    "NATIVE_PROFILE",
    "SCENE_BUNDLE_SCHEMA",
    "SCENE_BUNDLE_SCHEMA_VERSION",
    "compile_scene_bundle",
    "validate_scene_bundle",
]
