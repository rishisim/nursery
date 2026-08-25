"""Prepare an EmbodiedGen layout for one deterministic HOIDiNi interaction.

The result preserves the complete EmbodiedGen scene, while the active HOIDiNi
condition contains exactly one manipulated object, one horizontal support, and
one prompt. This module deliberately does not import HOIDiNi or GRAB and does
not create, encode, retarget, or infer human motion.

Only OBJ geometry is accepted. EmbodiedGen generated assets include OBJ visual
and collision meshes; rejecting other formats keeps this boundary explicit and
dependency-light.
"""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class BridgeValidationError(ValueError):
    """Raised when an EmbodiedGen layout cannot be prepared safely."""


@dataclass(frozen=True)
class Pose:
    """A world pose in metres with an XYZW quaternion."""

    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    @property
    def matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = quaternion_xyzw_to_matrix(self.quaternion_xyzw)
        result[:3, 3] = self.translation
        return result

    @property
    def axis_angle(self) -> tuple[float, float, float]:
        return tuple(quaternion_xyzw_to_axis_angle(self.quaternion_xyzw))

    def to_list(self) -> list[float]:
        return [*self.translation, *self.quaternion_xyzw]


@dataclass(frozen=True)
class GeometryReference:
    """One URDF geometry reference and its mesh-to-asset transform."""

    kind: str
    link: str
    urdf_path: Path
    source: str
    mesh_path: Path
    scale: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    mesh_to_asset: np.ndarray

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "link": self.link,
            "urdf_path": str(self.urdf_path),
            "source": self.source,
            "mesh_path": str(self.mesh_path),
            "scale": list(self.scale),
            "origin_xyz": list(self.origin_xyz),
            "origin_rpy": list(self.origin_rpy),
            "mesh_to_asset": self.mesh_to_asset.tolist(),
        }


@dataclass(frozen=True)
class Relationship:
    parent: str
    child: str
    relation: str

    def to_manifest(self) -> dict[str, str]:
        return {
            "parent": self.parent,
            "child": self.child,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class SceneNode:
    """One fully preserved EmbodiedGen scene node."""

    name: str
    role: str
    pose: Pose
    asset_reference: str | None
    asset_path: Path | None
    visual_geometry: tuple[GeometryReference, ...]
    collision_geometry: tuple[GeometryReference, ...]
    parent_relationships: tuple[Relationship, ...]
    child_relationships: tuple[Relationship, ...]
    description: Any = None
    quality: Any = None
    affordance_metadata: Mapping[str, Any] | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "pose": self.pose.to_list(),
            "asset_reference": self.asset_reference,
            "asset_path": str(self.asset_path) if self.asset_path else None,
            "visual_geometry": [item.to_manifest() for item in self.visual_geometry],
            "collision_geometry": [
                item.to_manifest() for item in self.collision_geometry
            ],
            "parent_relationships": [
                item.to_manifest() for item in self.parent_relationships
            ],
            "child_relationships": [
                item.to_manifest() for item in self.child_relationships
            ],
            "description": copy.deepcopy(self.description),
            "quality": copy.deepcopy(self.quality),
            "affordance_metadata": copy.deepcopy(self.affordance_metadata),
        }


@dataclass(frozen=True)
class TriangleMesh:
    """A triangular mesh in one documented coordinate frame."""

    vertices: np.ndarray
    faces: np.ndarray

    def transformed(self, transform: np.ndarray) -> "TriangleMesh":
        transform = _validate_transform(transform, "mesh transform")
        vertices_h = np.column_stack(
            [self.vertices, np.ones(len(self.vertices), dtype=np.float64)]
        )
        vertices = (transform @ vertices_h.T).T[:, :3]
        return TriangleMesh(vertices=vertices, faces=self.faces.copy())

    def to_manifest(self) -> dict[str, Any]:
        return {"vertices": self.vertices.tolist(), "faces": self.faces.tolist()}


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray
    normals: np.ndarray
    seed: int

    def to_manifest(self) -> dict[str, Any]:
        return {
            "count": len(self.points),
            "seed": self.seed,
            "points": self.points.tolist(),
            "normals": self.normals.tolist(),
        }


@dataclass(frozen=True)
class HumanPrefixRequirement:
    """The intentionally unfilled human-motion boundary."""

    required: bool = True
    provided: bool = False
    frames: int = 15
    fps: int = 20
    body_model: str = "SMPL-X"
    joints: int = 52
    message: str = (
        "A caller-supplied 15-frame SMPL-X human prefix at 20 fps is still "
        "required before HOIDiNi inference. This bridge does not generate, "
        "retarget, encode, or infer that prefix."
    )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "provided": self.provided,
            "frames": self.frames,
            "fps": self.fps,
            "body_model": self.body_model,
            "joints": self.joints,
            "message": self.message,
        }


@dataclass(frozen=True)
class ActiveInteraction:
    """The interaction-focused subset intended to condition HOIDiNi."""

    target_name: str
    support_name: str
    prompt: str
    canonical_target_mesh: TriangleMesh
    canonical_target_pose: Pose
    point_cloud: PointCloud
    support_corners_world: np.ndarray
    canonical_mesh_source: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "support_name": self.support_name,
            "prompt": self.prompt,
            "canonical_target_pose": self.canonical_target_pose.to_list(),
            "canonical_mesh_source": self.canonical_mesh_source,
            "canonical_target_mesh": self.canonical_target_mesh.to_manifest(),
            "point_cloud": self.point_cloud.to_manifest(),
            "support_corners_world": self.support_corners_world.tolist(),
        }


@dataclass(frozen=True)
class PreparedSceneBundle:
    """Canonical in-memory result of deterministic scene preparation."""

    schema: str
    layout_path: Path
    raw_layout: Mapping[str, Any]
    units: str
    up_axis: str
    embodiedgen_to_hoidini_world: np.ndarray
    nodes: Mapping[str, SceneNode]
    relationships: tuple[Relationship, ...]
    active_interaction: ActiveInteraction
    human_prefix: HumanPrefixRequirement

    @property
    def ready_for_hoidini_inference(self) -> bool:
        return not self.human_prefix.required or self.human_prefix.provided

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible deterministic preparation manifest."""

        return {
            "schema": self.schema,
            "layout_path": str(self.layout_path),
            "units": self.units,
            "up_axis": self.up_axis,
            "embodiedgen_to_hoidini_world": (
                self.embodiedgen_to_hoidini_world.tolist()
            ),
            "raw_layout": copy.deepcopy(self.raw_layout),
            "relationships": [item.to_manifest() for item in self.relationships],
            "nodes": {
                name: self.nodes[name].to_manifest() for name in sorted(self.nodes)
            },
            "active_interaction": self.active_interaction.to_manifest(),
            "human_prefix": self.human_prefix.to_manifest(),
            "ready_for_hoidini_inference": self.ready_for_hoidini_inference,
        }


def prepare_scene(
    layout_path: str | Path,
    *,
    target: str | None = None,
    prompt: str | None = None,
    point_count: int = 512,
    seed: int = 0,
) -> PreparedSceneBundle:
    """Prepare one complete scene and one active interaction without inference."""

    layout_path = Path(layout_path).expanduser().resolve()
    if not layout_path.is_file():
        raise BridgeValidationError(f"Layout file does not exist: {layout_path}")
    if point_count <= 0:
        raise BridgeValidationError("point_count must be a positive integer")
    try:
        raw_layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeValidationError(f"Cannot read layout JSON: {exc}") from exc
    if not isinstance(raw_layout, dict):
        raise BridgeValidationError("layout.json root must be an object")

    layout = _validate_layout(raw_layout)
    relationships = _parse_relationships(layout["tree"])
    active_target = _select_target(layout["relation"], target)
    roles = _roles_for_all_nodes(layout, relationships)
    nodes = _prepare_nodes(layout_path, layout, relationships, roles)

    target_node = nodes[active_target]
    support_name = _select_support(active_target, relationships, roles)
    support_node = nodes[support_name]
    target_mesh, target_source = _node_mesh(target_node, prefer="visual")
    canonical_mesh, canonical_pose = _canonicalize_target_mesh(
        target_mesh, target_node.pose
    )
    point_cloud = sample_point_cloud(
        canonical_mesh, point_count=point_count, seed=seed
    )
    support_mesh, _ = _node_mesh(support_node, prefer="visual")
    support_world = support_mesh.transformed(support_node.pose.matrix)
    support_corners = derive_support_corners(support_world)

    selected_prompt = prompt if prompt is not None else layout["relation"].get("task_desc")
    if not isinstance(selected_prompt, str) or not selected_prompt.strip():
        raise BridgeValidationError(
            "A non-empty prompt override or relation.task_desc is required"
        )
    active = ActiveInteraction(
        target_name=active_target,
        support_name=support_name,
        prompt=selected_prompt.strip(),
        canonical_target_mesh=canonical_mesh,
        canonical_target_pose=canonical_pose,
        point_cloud=point_cloud,
        support_corners_world=support_corners,
        canonical_mesh_source=target_source,
    )
    return PreparedSceneBundle(
        schema="nursery.embodiedgen_hoidini/1",
        layout_path=layout_path,
        raw_layout=copy.deepcopy(raw_layout),
        units="metres",
        up_axis="+Z",
        embodiedgen_to_hoidini_world=np.eye(4, dtype=np.float64),
        nodes=nodes,
        relationships=relationships,
        active_interaction=active,
        human_prefix=HumanPrefixRequirement(),
    )


def validate_embodiedgen_pose(values: Sequence[Any], *, name: str = "pose") -> Pose:
    """Validate and normalize ``[x, y, z, qx, qy, qz, qw]``."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise BridgeValidationError(f"{name} must be a seven-value sequence")
    if len(values) != 7:
        raise BridgeValidationError(f"{name} must contain exactly seven values")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BridgeValidationError(f"{name} must contain numeric values") from exc
    if not np.isfinite(numeric).all():
        raise BridgeValidationError(f"{name} contains non-finite values")
    quat = numeric[3:]
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise BridgeValidationError(f"{name} quaternion has zero length")
    if abs(norm - 1.0) > 1e-3:
        raise BridgeValidationError(
            f"{name} quaternion norm {norm:.6f} is not within 0.001 of one"
        )
    quat /= norm
    return Pose(tuple(numeric[:3]), tuple(quat))


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an XYZW quaternion to a rotation matrix."""

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise BridgeValidationError("quaternion must have four finite values")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise BridgeValidationError("quaternion has zero length")
    x, y, z, w = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_xyzw_to_axis_angle(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an XYZW quaternion to an axis-angle rotation vector."""

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise BridgeValidationError("quaternion must have four finite values")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise BridgeValidationError("quaternion has zero length")
    quat /= norm
    if quat[3] < 0:
        quat = -quat
    vector = quat[:3]
    sin_half = float(np.linalg.norm(vector))
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sin_half, float(np.clip(quat[3], -1.0, 1.0)))
    return vector / sin_half * angle


def sample_point_cloud(
    mesh: TriangleMesh, *, point_count: int, seed: int
) -> PointCloud:
    """Sample points and face normals deterministically by triangle area."""

    _validate_mesh(mesh, "point-cloud source")
    if point_count <= 0:
        raise BridgeValidationError("point_count must be positive")
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_area = np.linalg.norm(cross, axis=1)
    if not np.isfinite(double_area).all() or float(double_area.sum()) <= 1e-12:
        raise BridgeValidationError("point-cloud source has no non-degenerate faces")
    probabilities = double_area / double_area.sum()
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(mesh.faces), size=point_count, p=probabilities)
    chosen = triangles[face_ids]
    uv = rng.random((point_count, 2))
    sqrt_u = np.sqrt(uv[:, 0])
    points = (
        (1.0 - sqrt_u)[:, None] * chosen[:, 0]
        + (sqrt_u * (1.0 - uv[:, 1]))[:, None] * chosen[:, 1]
        + (sqrt_u * uv[:, 1])[:, None] * chosen[:, 2]
    )
    normals = cross[face_ids] / double_area[face_ids, None]
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise BridgeValidationError("sampled point cloud contains non-finite values")
    return PointCloud(points=points, normals=normals, seed=int(seed))


def derive_support_corners(
    mesh_world: TriangleMesh,
    *,
    horizontal_tolerance_degrees: float = 5.0,
    plane_tolerance: float = 1e-6,
) -> np.ndarray:
    """Extract one unambiguous horizontal quadrilateral top surface."""

    _validate_mesh(mesh_world, "support mesh")
    triangles = mesh_world.vertices[mesh_world.faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(cross, axis=1)
    valid = lengths > 1e-12
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / lengths[valid, None]
    flat = np.ptp(triangles[:, :, 2], axis=1) <= plane_tolerance
    horizontal = np.abs(normals[:, 2]) >= math.cos(
        math.radians(horizontal_tolerance_degrees)
    )
    candidates = np.flatnonzero(valid & flat & horizontal)
    if not len(candidates):
        raise BridgeValidationError(
            "Support geometry has no horizontal triangular surface"
        )
    heights = triangles[candidates, :, 2].mean(axis=1)
    top_height = float(heights.max())
    top_faces = candidates[np.abs(heights - top_height) <= plane_tolerance]
    if len(
        _face_components(
            mesh_world.vertices,
            mesh_world.faces[top_faces],
            tolerance=plane_tolerance,
        )
    ) != 1:
        raise BridgeValidationError(
            "Support geometry has multiple disconnected top surfaces"
        )
    vertex_ids = np.unique(mesh_world.faces[top_faces].reshape(-1))
    hull = _convex_hull_2d(
        mesh_world.vertices[vertex_ids, :2], tolerance=plane_tolerance
    )
    if len(hull) != 4:
        raise BridgeValidationError(
            "Support top is ambiguous or unsupported: expected four outer corners, "
            f"found {len(hull)}"
        )
    corners = np.column_stack([hull, np.full(4, top_height, dtype=np.float64)])
    signed_area = _signed_area_2d(corners[:, :2])
    if abs(signed_area) <= plane_tolerance:
        raise BridgeValidationError("Support corners form a degenerate polygon")
    if signed_area < 0:
        corners = corners[::-1]
    normal_z = float(
        np.cross(corners[1] - corners[0], corners[2] - corners[1])[2]
    )
    if normal_z <= 0:
        raise BridgeValidationError("Support corners do not have an upward normal")
    return corners


def _validate_layout(raw: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("tree", "relation", "objs_mapping", "assets", "position"):
        if key not in raw:
            raise BridgeValidationError(f"layout.json is missing required field {key!r}")
        if not isinstance(raw[key], dict):
            raise BridgeValidationError(f"layout field {key!r} must be an object")
    return dict(raw)


def _parse_relationships(tree: Mapping[str, Any]) -> tuple[Relationship, ...]:
    relationships: list[Relationship] = []
    for parent, children in tree.items():
        if not isinstance(parent, str) or not isinstance(children, list):
            raise BridgeValidationError("layout tree must map node names to child lists")
        for entry in children:
            if not isinstance(entry, list) or len(entry) != 2:
                raise BridgeValidationError(
                    f"Tree child under {parent!r} must be [child, relation]"
                )
            child, relation = entry
            if not isinstance(child, str) or not isinstance(relation, str):
                raise BridgeValidationError("Tree child and relation must be strings")
            relationships.append(Relationship(parent, child, relation))
    return tuple(relationships)


def _roles_for_all_nodes(
    layout: Mapping[str, Any], relationships: Sequence[Relationship]
) -> dict[str, str]:
    roles = dict(layout["objs_mapping"])
    relation = layout["relation"]
    for key, role in (("robot", "robot"), ("background", "background"), ("context", "context")):
        name = relation.get(key)
        if isinstance(name, str):
            roles.setdefault(name, role)
    for name in relation.get("manipulated_objs", []):
        roles.setdefault(name, "manipulated_objs")
    for name in relation.get("distractor_objs", []):
        roles.setdefault(name, "distractor_objs")
    all_nodes = set(layout["position"]) | set(layout["assets"])
    for relationship in relationships:
        all_nodes.update((relationship.parent, relationship.child))
    missing = sorted(node for node in all_nodes if node not in roles)
    if missing:
        raise BridgeValidationError(f"Scene nodes have no role: {missing}")
    missing_pose = sorted(node for node in all_nodes if node not in layout["position"])
    if missing_pose:
        raise BridgeValidationError(f"Scene nodes have no world pose: {missing_pose}")
    return {node: str(roles[node]) for node in sorted(all_nodes)}


def _select_target(relation: Mapping[str, Any], requested: str | None) -> str:
    manipulated = relation.get("manipulated_objs")
    if not isinstance(manipulated, list) or not all(
        isinstance(name, str) for name in manipulated
    ):
        raise BridgeValidationError("relation.manipulated_objs must be a string list")
    if not manipulated:
        raise BridgeValidationError("Layout has no manipulated object")
    if requested is None:
        if len(manipulated) != 1:
            raise BridgeValidationError(
                "Layout has multiple manipulated objects; provide an explicit target: "
                + ", ".join(manipulated)
            )
        return manipulated[0]
    if requested in manipulated:
        return requested
    matches = [name for name in manipulated if name.casefold() == requested.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise BridgeValidationError(
        f"Requested target {requested!r} is not an unambiguous manipulated object"
    )


def _select_support(
    target: str,
    relationships: Sequence[Relationship],
    roles: Mapping[str, str],
) -> str:
    parents = [item for item in relationships if item.child == target]
    if len(parents) != 1:
        raise BridgeValidationError(
            f"Target {target!r} must have exactly one parent relationship"
        )
    relationship = parents[0]
    if relationship.relation.upper() != "ON":
        raise BridgeValidationError(
            f"Target {target!r} uses unsupported support relation "
            f"{relationship.relation!r}; only ON is supported"
        )
    if roles.get(relationship.parent) != "context":
        raise BridgeValidationError(
            f"Target support {relationship.parent!r} is not the context node"
        )
    return relationship.parent


def _prepare_nodes(
    layout_path: Path,
    layout: Mapping[str, Any],
    relationships: Sequence[Relationship],
    roles: Mapping[str, str],
) -> dict[str, SceneNode]:
    result: dict[str, SceneNode] = {}
    descriptions = layout.get("objs_desc", {})
    quality = layout.get("quality", {})
    for name, role in roles.items():
        pose = validate_embodiedgen_pose(layout["position"][name], name=f"pose[{name}]")
        asset_reference = layout["assets"].get(name)
        asset_path: Path | None = None
        visual: tuple[GeometryReference, ...] = ()
        collision: tuple[GeometryReference, ...] = ()
        affordance: Mapping[str, Any] | None = None
        if asset_reference is not None:
            if not isinstance(asset_reference, str) or not asset_reference:
                raise BridgeValidationError(f"Asset reference for {name!r} is invalid")
            asset_path = _resolve_path(layout_path.parent, asset_reference)
            if not asset_path.exists():
                raise BridgeValidationError(
                    f"Asset reference for {name!r} does not exist: {asset_path}"
                )
            urdf_path = _resolve_urdf(asset_path, name)
            if urdf_path is not None:
                visual, collision, affordance = _parse_urdf(urdf_path)
        result[name] = SceneNode(
            name=name,
            role=role,
            pose=pose,
            asset_reference=asset_reference,
            asset_path=asset_path,
            visual_geometry=visual,
            collision_geometry=collision,
            parent_relationships=tuple(
                item for item in relationships if item.child == name
            ),
            child_relationships=tuple(
                item for item in relationships if item.parent == name
            ),
            description=copy.deepcopy(descriptions.get(name)),
            quality=copy.deepcopy(quality.get(name)),
            affordance_metadata=affordance,
        )
    return result


def _resolve_path(parent: Path, source: str) -> Path:
    if source.startswith("file://"):
        source = source[7:]
    if source.startswith("package://"):
        raise BridgeValidationError(f"package:// asset URI is unsupported: {source}")
    path = Path(source).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def _resolve_urdf(asset_path: Path, node_name: str) -> Path | None:
    if asset_path.is_file():
        return asset_path if asset_path.suffix.casefold() == ".urdf" else None
    candidates = []
    for candidate_name in (node_name, node_name.replace(" ", "_")):
        candidate = asset_path / f"{candidate_name}.urdf"
        if candidate.is_file():
            candidates.append(candidate)
    candidates.extend(asset_path.glob("*.urdf"))
    unique = sorted({item.resolve() for item in candidates})
    if len(unique) > 1:
        raise BridgeValidationError(
            f"Asset directory has ambiguous URDF files for {node_name!r}: {unique}"
        )
    return unique[0] if unique else None


def _parse_urdf(
    urdf_path: Path,
) -> tuple[
    tuple[GeometryReference, ...],
    tuple[GeometryReference, ...],
    Mapping[str, Any] | None,
]:
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise BridgeValidationError(f"Cannot parse URDF {urdf_path}: {exc}") from exc
    link_transforms = _urdf_link_transforms(root, urdf_path)
    geometry: dict[str, list[GeometryReference]] = {"visual": [], "collision": []}
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            raise BridgeValidationError(f"URDF link has no name: {urdf_path}")
        for kind in geometry:
            for element in link.findall(kind):
                mesh_element = element.find("./geometry/mesh")
                if mesh_element is None or not mesh_element.get("filename"):
                    continue
                source = str(mesh_element.get("filename"))
                mesh_path = _resolve_path(urdf_path.parent, source)
                if not mesh_path.is_file():
                    raise BridgeValidationError(
                        f"URDF {kind} mesh does not exist: {mesh_path}"
                    )
                scale = _parse_vector(
                    mesh_element.get("scale"), 3, (1.0, 1.0, 1.0), "mesh scale"
                )
                if any(value <= 0 for value in scale):
                    raise BridgeValidationError(f"Mesh scale must be positive: {scale}")
                origin = element.find("origin")
                xyz = _parse_vector(
                    origin.get("xyz") if origin is not None else None,
                    3,
                    (0.0, 0.0, 0.0),
                    "origin xyz",
                )
                rpy = _parse_vector(
                    origin.get("rpy") if origin is not None else None,
                    3,
                    (0.0, 0.0, 0.0),
                    "origin rpy",
                )
                mesh_to_asset = (
                    link_transforms[link_name]
                    @ _origin_matrix(xyz, rpy)
                    @ np.diag([*scale, 1.0])
                )
                geometry[kind].append(
                    GeometryReference(
                        kind=kind,
                        link=link_name,
                        urdf_path=urdf_path,
                        source=source,
                        mesh_path=mesh_path,
                        scale=scale,
                        origin_xyz=xyz,
                        origin_rpy=rpy,
                        mesh_to_asset=mesh_to_asset,
                    )
                )
    affordance = _read_affordance_metadata(root, urdf_path)
    return tuple(geometry["visual"]), tuple(geometry["collision"]), affordance


def _urdf_link_transforms(root: ET.Element, urdf_path: Path) -> dict[str, np.ndarray]:
    link_names = {
        link.get("name") for link in root.findall("link") if link.get("name")
    }
    parents: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise BridgeValidationError(f"URDF joint lacks parent or child: {urdf_path}")
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            raise BridgeValidationError(f"URDF joint has empty link name: {urdf_path}")
        if joint.get("type", "fixed") != "fixed":
            raise BridgeValidationError(
                "Non-fixed URDF joint is unsupported for deterministic preparation: "
                f"{joint.get('name', '<unnamed>')}"
            )
        origin = joint.find("origin")
        xyz = _parse_vector(
            origin.get("xyz") if origin is not None else None,
            3,
            (0.0, 0.0, 0.0),
            "joint origin xyz",
        )
        rpy = _parse_vector(
            origin.get("rpy") if origin is not None else None,
            3,
            (0.0, 0.0, 0.0),
            "joint origin rpy",
        )
        if child_name in parents:
            raise BridgeValidationError(f"URDF link has multiple parents: {child_name}")
        parents[child_name] = (parent_name, _origin_matrix(xyz, rpy))

    cache: dict[str, np.ndarray] = {}

    def transform(link: str, active: set[str]) -> np.ndarray:
        if link in cache:
            return cache[link]
        if link in active:
            raise BridgeValidationError(f"URDF joint cycle includes {link!r}")
        active.add(link)
        if link in parents:
            parent_name, local = parents[link]
            if parent_name not in link_names:
                raise BridgeValidationError(f"URDF joint references unknown link {parent_name}")
            value = transform(parent_name, active) @ local
        else:
            value = np.eye(4, dtype=np.float64)
        active.remove(link)
        cache[link] = value
        return value

    return {name: transform(name, set()) for name in sorted(link_names)}


def _read_affordance_metadata(
    root: ET.Element, urdf_path: Path
) -> Mapping[str, Any] | None:
    element = root.find(".//custom_data/affordance/affordance_annot")
    if element is None or not element.text or not element.text.strip():
        return None
    path = _resolve_path(urdf_path.parent, element.text.strip())
    if not path.is_file():
        raise BridgeValidationError(f"Affordance annotation does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeValidationError(f"Cannot read affordance metadata {path}: {exc}") from exc
    return {
        "source_path": str(path),
        "frame": "URDF asset/link frame; robot gripper grasps are metadata only",
        "payload": payload,
    }


def _node_mesh(node: SceneNode, *, prefer: str) -> tuple[TriangleMesh, str]:
    if prefer not in {"visual", "collision"}:
        raise ValueError(prefer)
    preferred = node.visual_geometry if prefer == "visual" else node.collision_geometry
    fallback = node.collision_geometry if prefer == "visual" else node.visual_geometry
    references = preferred or fallback
    source_kind = prefer if preferred else ("collision" if prefer == "visual" else "visual")
    if not references:
        raise BridgeValidationError(
            f"Scene node {node.name!r} has no URDF visual or collision mesh"
        )
    meshes = [
        _load_obj(reference.mesh_path).transformed(reference.mesh_to_asset)
        for reference in references
    ]
    return _combine_meshes(meshes), source_kind


def _canonicalize_target_mesh(mesh: TriangleMesh, pose: Pose) -> tuple[TriangleMesh, Pose]:
    _validate_mesh(mesh, "target mesh")
    center = (mesh.vertices.min(axis=0) + mesh.vertices.max(axis=0)) / 2.0
    canonical = TriangleMesh(vertices=mesh.vertices - center, faces=mesh.faces.copy())
    rotation = quaternion_xyzw_to_matrix(pose.quaternion_xyzw)
    compensated_translation = np.asarray(pose.translation) + rotation @ center
    return canonical, Pose(tuple(compensated_translation), pose.quaternion_xyzw)


def _load_obj(path: Path) -> TriangleMesh:
    if path.suffix.casefold() != ".obj":
        raise BridgeValidationError(
            f"Unsupported mesh format {path.suffix!r}; deterministic bridge accepts OBJ"
        )
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BridgeValidationError(f"Cannot read OBJ {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, 1):
        fields = raw_line.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise BridgeValidationError(f"Invalid OBJ vertex at {path}:{line_number}")
            try:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            except ValueError as exc:
                raise BridgeValidationError(
                    f"Invalid OBJ vertex at {path}:{line_number}"
                ) from exc
        elif fields[0] == "f":
            if len(fields) < 4:
                raise BridgeValidationError(f"Invalid OBJ face at {path}:{line_number}")
            indices: list[int] = []
            for field in fields[1:]:
                try:
                    raw_index = int(field.split("/", 1)[0])
                except ValueError as exc:
                    raise BridgeValidationError(
                        f"Invalid OBJ face index at {path}:{line_number}"
                    ) from exc
                index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                if index < 0 or index >= len(vertices):
                    raise BridgeValidationError(
                        f"OBJ face index is out of range at {path}:{line_number}"
                    )
                indices.append(index)
            for offset in range(1, len(indices) - 1):
                faces.append([indices[0], indices[offset], indices[offset + 1]])
    mesh = TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )
    _validate_mesh(mesh, str(path))
    return mesh


def _combine_meshes(meshes: Sequence[TriangleMesh]) -> TriangleMesh:
    if not meshes:
        raise BridgeValidationError("Cannot combine an empty mesh list")
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        _validate_mesh(mesh, "combined mesh input")
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return TriangleMesh(vertices=np.vstack(vertices), faces=np.vstack(faces))


def _validate_mesh(mesh: TriangleMesh, name: str) -> None:
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1:] != (3,):
        raise BridgeValidationError(f"{name} vertices must have shape [N, 3]")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1:] != (3,):
        raise BridgeValidationError(f"{name} faces must have shape [M, 3]")
    if not len(mesh.vertices) or not len(mesh.faces):
        raise BridgeValidationError(f"{name} is empty")
    if not np.isfinite(mesh.vertices).all():
        raise BridgeValidationError(f"{name} contains non-finite vertices")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise BridgeValidationError(f"{name} has invalid face indices")


def _validate_transform(transform: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise BridgeValidationError(f"{name} must be a finite 4x4 matrix")
    return value


def _parse_vector(
    source: str | None,
    length: int,
    default: tuple[float, ...],
    name: str,
) -> tuple[float, ...]:
    if source is None or not source.strip():
        return default
    try:
        result = tuple(float(item) for item in source.split())
    except ValueError as exc:
        raise BridgeValidationError(f"{name} contains non-numeric values") from exc
    if len(result) != length or not np.isfinite(result).all():
        raise BridgeValidationError(f"{name} must have {length} finite values")
    return result


def _origin_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = xyz
    return result


def _face_components(
    vertices: np.ndarray, faces: np.ndarray, *, tolerance: float
) -> list[set[int]]:
    """Find face components while treating duplicate coincident vertices as shared."""

    quantized = np.round(vertices / tolerance).astype(np.int64)
    vertex_to_faces: dict[tuple[int, int, int], list[int]] = {}
    for face_id, face in enumerate(faces):
        for vertex in face:
            key = tuple(quantized[int(vertex)])
            vertex_to_faces.setdefault(key, []).append(face_id)
    remaining = set(range(len(faces)))
    components: list[set[int]] = []
    while remaining:
        component: set[int] = set()
        stack = [remaining.pop()]
        while stack:
            face_id = stack.pop()
            component.add(face_id)
            neighbours = {
                neighbour
                for vertex in faces[face_id]
                for neighbour in vertex_to_faces[tuple(quantized[int(vertex)])]
            }
            new = neighbours & remaining
            remaining -= new
            stack.extend(new)
        components.append(component)
    return components


def _convex_hull_2d(points: np.ndarray, *, tolerance: float) -> np.ndarray:
    rounded = np.unique(np.round(points / tolerance).astype(np.int64), axis=0)
    unique = rounded.astype(np.float64) * tolerance
    if len(unique) < 3:
        return unique
    ordered = sorted(map(tuple, unique.tolist()))

    def cross(
        origin: tuple[float, float],
        point_a: tuple[float, float],
        point_b: tuple[float, float],
    ) -> float:
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (
            point_a[1] - origin[1]
        ) * (point_b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _signed_area_2d(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
