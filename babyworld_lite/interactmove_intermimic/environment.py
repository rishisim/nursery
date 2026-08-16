"""Explicit mesh-backed room contract for InteractMove scene ingestion."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Callable, Mapping

from .common import (
    ContractError,
    read_json,
    require_exact_keys,
    require_finite_number,
    require_mapping,
    require_nonempty_string,
    require_vector,
    safe_relative_file,
)


ENVIRONMENT_GEOMETRY_SCHEMA = "InteractMoveEnvironmentGeometry"
ENVIRONMENT_GEOMETRY_SCHEMA_VERSION = 1
ENVIRONMENT_GEOMETRY_FILENAME = "environment_geometry.json"
_PLY_SCALARS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _relative_child(background: PurePosixPath, value: Any, *, where: str) -> str:
    text = require_nonempty_string(value, where=where)
    if "\\" in text or "\x00" in text or ":" in text:
        raise ContractError(f"{where} must be a relative POSIX path")
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ContractError(f"{where} must be a traversal-free relative POSIX path")
    return (background / relative).as_posix()


def _mesh_report(
    *,
    relative_path: str,
    vertex_count: int,
    face_count: int,
    vertices: list[tuple[float, float, float]],
    scale_xyz: list[float],
) -> dict[str, Any]:
    if vertex_count < 4 or face_count < 2 or len(vertices) != vertex_count:
        raise ContractError(
            f"mesh {relative_path!r} must contain at least 4 vertices and 2 faces"
        )
    scaled = [
        tuple(vertex[index] * scale_xyz[index] for index in range(3))
        for vertex in vertices
    ]
    if any(not math.isfinite(item) for vertex in scaled for item in vertex):
        raise ContractError(f"mesh {relative_path!r} contains non-finite vertices")
    minimum = [min(vertex[index] for vertex in scaled) for index in range(3)]
    maximum = [max(vertex[index] for vertex in scaled) for index in range(3)]
    extents = [maximum[index] - minimum[index] for index in range(3)]
    if any(extent <= 0.0 for extent in extents):
        raise ContractError(f"mesh {relative_path!r} has a non-positive 3D extent")
    return {
        "path": relative_path,
        "format": PurePosixPath(relative_path).suffix.lower().lstrip("."),
        "vertex_count": vertex_count,
        "face_count": face_count,
        "scale_xyz": scale_xyz,
        "aabb_min_m": minimum,
        "aabb_max_m": maximum,
        "extents_m": extents,
        "finite_vertices": True,
    }


def _inspect_obj(path: Path, relative_path: str, scale_xyz: list[float]) -> dict[str, Any]:
    vertices: list[tuple[float, float, float]] = []
    face_count = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ContractError(f"OBJ mesh is not UTF-8: {relative_path}") from exc
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v":
            if len(parts) < 4:
                raise ContractError(f"OBJ vertex is malformed at {relative_path}:{line_number}")
            try:
                vertex = tuple(float(parts[index]) for index in range(1, 4))
            except ValueError as exc:
                raise ContractError(
                    f"OBJ vertex is not numeric at {relative_path}:{line_number}"
                ) from exc
            vertices.append(vertex)
        elif parts[0] == "f":
            if len(parts) < 4:
                raise ContractError(f"OBJ face is malformed at {relative_path}:{line_number}")
            face_count += max(1, len(parts) - 3)
    return _mesh_report(
        relative_path=relative_path,
        vertex_count=len(vertices),
        face_count=face_count,
        vertices=vertices,
        scale_xyz=scale_xyz,
    )


def _ply_header(path: Path, relative_path: str) -> tuple[list[str], int]:
    data = path.read_bytes()
    end = data.find(b"end_header\n")
    marker_size = len(b"end_header\n")
    if end < 0:
        end = data.find(b"end_header\r\n")
        marker_size = len(b"end_header\r\n")
    if end < 0:
        raise ContractError(f"PLY mesh has no end_header marker: {relative_path}")
    try:
        header = data[:end].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ContractError(f"PLY header is not ASCII: {relative_path}") from exc
    if not header or header[0].strip() != "ply":
        raise ContractError(f"PLY mesh has no magic header: {relative_path}")
    return header, end + marker_size


def _inspect_ply(path: Path, relative_path: str, scale_xyz: list[float]) -> dict[str, Any]:
    header, payload_offset = _ply_header(path, relative_path)
    data = path.read_bytes()
    mesh_format = None
    vertex_count = None
    face_count = 0
    current_element = None
    vertex_properties: list[tuple[str, str]] = []
    for raw in header[1:]:
        parts = raw.strip().split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "format" and len(parts) == 3:
            mesh_format = parts[1]
        elif parts[0] == "element" and len(parts) == 3:
            current_element = parts[1]
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise ContractError(f"PLY element count is invalid: {relative_path}") from exc
            if count < 0:
                raise ContractError(f"PLY element count is negative: {relative_path}")
            if current_element == "vertex":
                vertex_count = count
            elif current_element == "face":
                face_count = count
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ContractError(f"PLY vertex properties must be scalar: {relative_path}")
            if parts[1] not in _PLY_SCALARS:
                raise ContractError(f"unsupported PLY scalar type {parts[1]!r}: {relative_path}")
            vertex_properties.append((parts[1], parts[2]))
    if mesh_format not in {"ascii", "binary_little_endian"}:
        raise ContractError(f"unsupported PLY format {mesh_format!r}: {relative_path}")
    if vertex_count is None:
        raise ContractError(f"PLY mesh has no vertex element: {relative_path}")
    names = [name for _, name in vertex_properties]
    if any(axis not in names for axis in ("x", "y", "z")):
        raise ContractError(f"PLY vertices require x/y/z properties: {relative_path}")
    indices = [names.index(axis) for axis in ("x", "y", "z")]
    vertices: list[tuple[float, float, float]] = []
    if mesh_format == "ascii":
        try:
            payload_lines = data[payload_offset:].decode("ascii").splitlines()
        except UnicodeDecodeError as exc:
            raise ContractError(f"ASCII PLY payload is not ASCII: {relative_path}") from exc
        if len(payload_lines) < vertex_count:
            raise ContractError(f"PLY vertex payload is truncated: {relative_path}")
        for row in payload_lines[:vertex_count]:
            parts = row.split()
            if len(parts) < len(vertex_properties):
                raise ContractError(f"PLY vertex payload is malformed: {relative_path}")
            try:
                values = [float(parts[index]) for index in indices]
            except ValueError as exc:
                raise ContractError(f"PLY vertex payload is not numeric: {relative_path}") from exc
            vertices.append(tuple(values))
    else:
        format_string = "<" + "".join(_PLY_SCALARS[kind] for kind, _ in vertex_properties)
        row_size = struct.calcsize(format_string)
        end = payload_offset + row_size * vertex_count
        if end > len(data):
            raise ContractError(f"binary PLY vertex payload is truncated: {relative_path}")
        for offset in range(payload_offset, end, row_size):
            values = struct.unpack_from(format_string, data, offset)
            vertices.append(tuple(float(values[index]) for index in indices))
    return _mesh_report(
        relative_path=relative_path,
        vertex_count=vertex_count,
        face_count=face_count,
        vertices=vertices,
        scale_xyz=scale_xyz,
    )


def inspect_mesh(path: Path, relative_path: str, scale_xyz: list[float]) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _inspect_obj(path, relative_path, scale_xyz)
    if suffix == ".ply":
        return _inspect_ply(path, relative_path, scale_xyz)
    raise ContractError(f"environment mesh format must be OBJ or PLY: {relative_path}")


def validate_wall_collision_envelope(
    reports: list[Mapping[str, Any]],
) -> None:
    """Validate room scale across one or more explicit wall collision meshes."""
    if not reports:
        raise ContractError("room collision requires at least one walls mesh")
    for report in reports:
        if report["extents_m"][2] < 1.8:
            raise ContractError("each walls collision mesh must have room-scale height")
    minimum = [
        min(report["aabb_min_m"][axis] for report in reports) for axis in range(3)
    ]
    maximum = [
        max(report["aabb_max_m"][axis] for report in reports) for axis in range(3)
    ]
    extents = [maximum[axis] - minimum[axis] for axis in range(3)]
    if min(extents[:2]) < 2.0 or extents[2] < 1.8:
        raise ContractError("walls collision meshes do not enclose a room-scale volume")


def load_environment_geometry(
    scene_root: Path,
    background_relative: PurePosixPath,
    native_profile: str,
    add_file: Callable[[Path, str], Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Load and validate an explicit room manifest; never infer collision from render mesh."""

    manifest_relative = (background_relative / ENVIRONMENT_GEOMETRY_FILENAME).as_posix()
    candidate = scene_root.joinpath(*PurePosixPath(manifest_relative).parts)
    if not candidate.exists() and not candidate.is_symlink():
        return None
    manifest_path = safe_relative_file(
        scene_root, manifest_relative, where="background environment geometry manifest"
    )
    manifest_record = add_file(manifest_path, "background_environment_geometry_manifest")
    manifest = read_json(manifest_path)
    require_exact_keys(
        manifest,
        required={
            "schema",
            "schema_version",
            "native_profile",
            "reference_mesh",
            "collision_geometry",
            "render_mode",
            "provenance",
        },
        where="environment geometry manifest",
    )
    if manifest["schema"] != ENVIRONMENT_GEOMETRY_SCHEMA:
        raise ContractError("environment geometry manifest schema mismatch")
    if manifest["schema_version"] != ENVIRONMENT_GEOMETRY_SCHEMA_VERSION:
        raise ContractError("environment geometry manifest version mismatch")
    if manifest["native_profile"] != native_profile:
        raise ContractError("environment geometry native profile mismatch")
    if manifest["render_mode"] != "raster_reference_mesh":
        raise ContractError("environment render_mode must be raster_reference_mesh")
    provenance = require_mapping(manifest["provenance"], where="environment provenance")
    require_exact_keys(
        provenance,
        required={
            "source",
            "embodiedgen_commit",
            "source_job_id",
            "source_artifact_path",
            "source_artifact_sha256",
            "creation_method",
        },
        where="environment provenance",
    )
    require_nonempty_string(provenance["source"], where="environment provenance.source")
    require_nonempty_string(
        provenance["embodiedgen_commit"], where="environment provenance.embodiedgen_commit"
    )
    if provenance["embodiedgen_commit"] != "9b333554254af196bace88c1a171a3bf047fa09c":
        raise ContractError("environment provenance EmbodiedGen commit mismatch")
    require_nonempty_string(
        provenance["source_job_id"], where="environment provenance.source_job_id"
    )
    digest = require_nonempty_string(
        provenance["source_artifact_sha256"],
        where="environment provenance.source_artifact_sha256",
    )
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractError("environment provenance source_artifact_sha256 is invalid")
    source_artifact_relative = _relative_child(
        background_relative,
        provenance["source_artifact_path"],
        where="environment provenance.source_artifact_path",
    )
    source_artifact_path = safe_relative_file(
        scene_root,
        source_artifact_relative,
        where="environment provenance source artifact",
    )
    source_artifact_record = add_file(
        source_artifact_path, "background_environment_source_receipt"
    )
    if source_artifact_record["sha256"] != digest:
        raise ContractError("environment provenance source artifact hash mismatch")
    require_nonempty_string(
        provenance["creation_method"], where="environment provenance.creation_method"
    )

    reference = require_mapping(manifest["reference_mesh"], where="environment reference_mesh")
    require_exact_keys(
        reference,
        required={"path", "frame", "units", "scale_xyz", "purposes"},
        where="environment reference_mesh",
    )
    if reference["frame"] != "world" or reference["units"] != "m":
        raise ContractError("environment reference mesh must be world-frame meters")
    scale = require_vector(
        reference["scale_xyz"], length=3, positive=True, where="reference_mesh.scale_xyz"
    )
    purposes = reference["purposes"]
    if purposes != ["interactmove_pointcloud", "raster_render"]:
        raise ContractError("reference_mesh.purposes must freeze point-cloud and raster use")
    reference_relative = _relative_child(
        background_relative, reference["path"], where="reference_mesh.path"
    )
    reference_path = safe_relative_file(
        scene_root, reference_relative, where="environment reference mesh"
    )
    reference_file = add_file(reference_path, "background_reference_mesh")
    reference_report = inspect_mesh(reference_path, reference_relative, scale)
    horizontal = sorted(reference_report["extents_m"][:2])
    if horizontal[0] < 2.0 or horizontal[1] > 30.0:
        raise ContractError("reference mesh has implausible dining-room horizontal scale")
    if not 1.8 <= reference_report["extents_m"][2] <= 10.0:
        raise ContractError("reference mesh has implausible dining-room height")
    if abs(reference_report["aabb_min_m"][2]) > 0.10:
        raise ContractError("reference mesh floor must be within 0.10 m of world z=0")
    reference_report["sha256"] = reference_file["sha256"]

    collision_raw = manifest["collision_geometry"]
    if not isinstance(collision_raw, list) or not collision_raw:
        raise ContractError("collision_geometry must be a non-empty array")
    collisions: list[dict[str, Any]] = []
    ids: set[str] = set()
    floor_count = 0
    walls_count = 0
    wall_reports: list[Mapping[str, Any]] = []
    consumed = {
        manifest_record["sha256"],
        reference_file["sha256"],
        source_artifact_record["sha256"],
    }
    for index, raw in enumerate(collision_raw):
        item = require_mapping(raw, where=f"collision_geometry[{index}]")
        common = {"collision_id", "role", "geometry_type", "frame"}
        kind = item.get("geometry_type")
        if kind == "plane":
            require_exact_keys(
                item,
                required=common | {"z_m", "normal"},
                where=f"collision_geometry[{index}]",
            )
        elif kind == "mesh":
            require_exact_keys(
                item,
                required=common | {"path", "units", "scale_xyz"},
                where=f"collision_geometry[{index}]",
            )
        else:
            raise ContractError(f"collision_geometry[{index}] has unsupported geometry_type")
        collision_id = require_nonempty_string(
            item["collision_id"], where=f"collision_geometry[{index}].collision_id"
        )
        if collision_id in ids:
            raise ContractError("collision geometry IDs must be unique")
        ids.add(collision_id)
        role = require_nonempty_string(item["role"], where=f"collision_geometry[{index}].role")
        if role not in {"floor", "walls", "environment_structure"}:
            raise ContractError(f"collision_geometry[{index}] has unsupported role")
        if item["frame"] != "world":
            raise ContractError("environment collision geometry must be in the world frame")
        if kind == "plane":
            if role != "floor":
                raise ContractError("only floor collision may use an infinite plane")
            z_m = require_finite_number(item["z_m"], where=f"collision_geometry[{index}].z_m")
            normal = require_vector(item["normal"], length=3, where=f"collision_geometry[{index}].normal")
            if z_m != 0.0 or normal != [0.0, 0.0, 1.0]:
                raise ContractError("floor collision must be the frozen +Z plane at z=0")
            floor_count += 1
            collisions.append(
                {
                    "collision_id": collision_id,
                    "role": role,
                    "geometry_type": kind,
                    "frame": "world",
                    "z_m": z_m,
                    "normal": normal,
                }
            )
        else:
            if item["units"] != "m":
                raise ContractError("collision mesh units must be meters")
            collision_scale = require_vector(
                item["scale_xyz"],
                length=3,
                positive=True,
                where=f"collision_geometry[{index}].scale_xyz",
            )
            relative = _relative_child(
                background_relative, item["path"], where=f"collision_geometry[{index}].path"
            )
            if relative == reference_relative:
                raise ContractError("reference mesh must not be reused implicitly as collision")
            path = safe_relative_file(scene_root, relative, where="environment collision mesh")
            record = add_file(path, "background_collision_mesh")
            report = inspect_mesh(path, relative, collision_scale)
            report["sha256"] = record["sha256"]
            consumed.add(record["sha256"])
            if role == "walls":
                walls_count += 1
                wall_reports.append(report)
            collisions.append(
                {
                    "collision_id": collision_id,
                    "role": role,
                    "geometry_type": kind,
                    "frame": "world",
                    "units": "m",
                    "mesh": report,
                }
            )
    if floor_count != 1 or walls_count < 1:
        raise ContractError("room collision requires exactly one floor plane and at least one walls mesh")
    validate_wall_collision_envelope(wall_reports)
    return {
        "manifest": manifest_relative,
        "manifest_sha256": manifest_record["sha256"],
        "render_mode": "raster_reference_mesh",
        "reference_mesh": reference_report,
        "collision_geometry": collisions,
        "provenance": dict(provenance),
        "consumed_file_sha256": sorted(consumed),
    }


def validate_environment_envelope(
    environment: Mapping[str, Any], instance_positions: Mapping[str, list[float]]
) -> None:
    """Require every non-background instance origin to lie inside the room AABB."""

    reference = require_mapping(environment["reference_mesh"], where="reference mesh report")
    minimum = require_vector(reference["aabb_min_m"], length=3, where="reference aabb_min_m")
    maximum = require_vector(reference["aabb_max_m"], length=3, where="reference aabb_max_m")
    for source_node_key, position in instance_positions.items():
        xyz = require_vector(position, length=3, where=f"instance position {source_node_key!r}")
        if not (
            minimum[0] + 0.05 <= xyz[0] <= maximum[0] - 0.05
            and minimum[1] + 0.05 <= xyz[1] <= maximum[1] - 0.05
            and minimum[2] - 0.05 <= xyz[2] <= maximum[2] + 0.05
        ):
            raise ContractError(
                f"instance {source_node_key!r} lies outside the validated room envelope"
            )
