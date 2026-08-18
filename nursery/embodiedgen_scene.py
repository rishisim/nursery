"""Minimal activity-to-static-room adapter for EmbodiedGen V2.

The module owns only orchestration and small JSON/CSV manifests. Room creation,
asset retrieval and generation, and collision-aware placement remain native
EmbodiedGen operations. Native multi-stage planning is adapted to the OpenAI
Responses API through EmbodiedGen's existing GPTclient protocol.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/embodiedgen_scene.json"
OPENAI_MODEL = "gpt-5.6-luna"
SUPPORTED_RELATIONS = frozenset({"in_room", "on", "beside", "inside"})
ROOM_CLI_TYPES = {
    "Bedroom": "bedroom",
    "LivingRoom": "livingRoom",
    "Kitchen": "kitchen",
    "Bathroom": "bathroom",
    "DiningRoom": "diningRoom",
    "Office": "office",
}
REQUIRED_CONFIG_KEYS = frozenset(
    {
        "embodiedgen_root",
        "output_root",
        "dataset_index",
        "room_complexity",
        "room_types",
        "room_seeds",
        "distractor_count",
        "generate_usd",
    }
)


ScenePipelineError = RuntimeError
_OPENAI_RESPONSES_CLIENT: Any | None = None


class _ResponsesGPTClient:
    """EmbodiedGen GPTclient protocol backed by one lazy Responses client."""

    model_name = OPENAI_MODEL

    def query(
        self,
        text_prompt: str,
        image_base64: Sequence[Any] | None = None,
        system_role: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> str:
        if image_base64:
            raise ScenePipelineError("Nursery scene planning supports text input only")
        request: dict[str, Any] = {
            "model": self.model_name,
            "input": text_prompt,
            "reasoning": {"effort": "low"},
        }
        if system_role:
            request["instructions"] = system_role
        if params:
            max_output_tokens = (
                params.get("max_output_tokens")
                or params.get("max_completion_tokens")
                or params.get("max_tokens")
            )
            if max_output_tokens is not None:
                request["max_output_tokens"] = int(max_output_tokens)
            if isinstance(params.get("reasoning"), Mapping):
                request["reasoning"] = dict(params["reasoning"])

        try:
            response = _openai_responses_client().responses.create(**request)
        except Exception as exc:
            raise ScenePipelineError(f"OpenAI Responses request failed: {exc}") from exc
        output = str(getattr(response, "output_text", "")).strip()
        if not output:
            raise ScenePipelineError("OpenAI Responses request returned no text")
        return output


_RESPONSES_GPT_CLIENT = _ResponsesGPTClient()


_BLENDER_PREVIEW_SCRIPT = r'''import sys
from pathlib import Path

import bpy
from mathutils import Vector


output_path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 1280
scene.render.resolution_y = 960
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(output_path)
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "RANDOM"
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True

hidden_shell_collections = (
    "placeholders:room_shells",
    "placeholders:room_meshes",
    "placeholders:portal_cutters",
    "unique_assets:room_ceiling",
    "unique_assets:room_exterior",
    "unique_assets:room_wall",
)
for collection_name in hidden_shell_collections:
    collection = bpy.data.collections.get(collection_name)
    if collection is not None:
        collection.hide_render = True

floor_collection = bpy.data.collections.get("unique_assets:room_floor")
if floor_collection is None:
    raise RuntimeError("scene has no EmbodiedGen room-floor collection")
visible_floors = [obj for obj in floor_collection.objects if not obj.hide_render]
if not visible_floors:
    raise RuntimeError("scene has no visible EmbodiedGen room floor")
room_floor = max(visible_floors, key=lambda obj: obj.dimensions.x * obj.dimensions.y)
floor_collection.hide_render = False
for obj in floor_collection.objects:
    obj.hide_render = obj != room_floor

corners = [room_floor.matrix_world @ Vector(corner) for corner in room_floor.bound_box]
minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
center = (minimum + maximum) / 2
room_span = max(maximum.x - minimum.x, maximum.y - minimum.y)
if room_span <= 0:
    raise RuntimeError("room floor has invalid bounds")

camera_data = bpy.data.cameras.new("nursery_preview_camera")
camera = bpy.data.objects.new("nursery_preview_camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (
    center.x - room_span,
    center.y - room_span,
    maximum.z + 0.9 * room_span,
)
camera.data.type = "ORTHO"
camera.data.ortho_scale = 1.25 * room_span
target = center + Vector((0, 0, 0.4))
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera

output_path.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.render.render(write_still=True)
'''


_BLENDER_URDF_PREVIEW_SCRIPT = r'''import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Euler, Matrix, Vector


arguments = sys.argv[sys.argv.index("--") + 1 :]
urdf_path = Path(arguments[0]).resolve()
output_path = Path(arguments[1]).resolve()
root = ET.parse(urdf_path).getroot()


def values(element, attribute, default):
    if element is None or not element.get(attribute):
        return default
    return tuple(float(value) for value in element.get(attribute).split())


def origin_matrix(element):
    origin = element.find("origin") if element is not None else None
    xyz = values(origin, "xyz", (0.0, 0.0, 0.0))
    rpy = values(origin, "rpy", (0.0, 0.0, 0.0))
    return Matrix.Translation(Vector(xyz)) @ Euler(rpy, "XYZ").to_matrix().to_4x4()


joints = {}
for joint in root.findall("joint"):
    parent = joint.find("parent")
    child = joint.find("child")
    if parent is not None and child is not None:
        joints[child.get("link")] = (parent.get("link"), origin_matrix(joint))

link_transforms = {}


def link_transform(link_name, active=None):
    if link_name in link_transforms:
        return link_transforms[link_name]
    active = set() if active is None else active
    if link_name in active:
        raise RuntimeError(f"joint cycle at {link_name}")
    active.add(link_name)
    if link_name not in joints:
        transform = Matrix.Identity(4)
    else:
        parent_name, joint_origin = joints[link_name]
        transform = link_transform(parent_name, active) @ joint_origin
    active.remove(link_name)
    link_transforms[link_name] = transform
    return transform


scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 1280
scene.render.resolution_y = 960
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(output_path)
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "RANDOM"
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True

rendered_objects = []
floor_objects = []
for link in root.findall("link"):
    link_name = link.get("name", "")
    normalized_name = link_name.casefold()
    if normalized_name.endswith(("_ceiling", "_exterior", "_wall")):
        continue
    for visual in link.findall("visual"):
        mesh = visual.find("./geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            continue
        filename = mesh.get("filename")
        if filename.startswith("file://"):
            filename = filename[7:]
        if filename.startswith("package://"):
            raise RuntimeError(f"package URI is unsupported: {filename}")
        mesh_path = Path(filename).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        mesh_path = mesh_path.resolve()
        if mesh_path.suffix.casefold() != ".obj":
            raise RuntimeError(f"preview requires OBJ visual meshes: {mesh_path}")
        before = set(bpy.data.objects)
        bpy.ops.wm.obj_import(filepath=str(mesh_path), forward_axis="Y", up_axis="Z")
        imported = [obj for obj in bpy.data.objects if obj not in before]
        scale = values(mesh, "scale", (1.0, 1.0, 1.0))
        mesh_scale = Matrix.Diagonal((*scale, 1.0))
        transform = link_transform(link_name) @ origin_matrix(visual) @ mesh_scale
        for obj in imported:
            obj.matrix_world = transform @ obj.matrix_world
            obj.name = f"{link_name}:{obj.name}"
            rendered_objects.append(obj)
            if normalized_name.endswith("_floor"):
                floor_objects.append(obj)

if not rendered_objects:
    raise RuntimeError("URDF has no renderable visual OBJ meshes")
bpy.context.view_layer.update()
framing_objects = floor_objects or rendered_objects
corners = [obj.matrix_world @ Vector(corner) for obj in framing_objects for corner in obj.bound_box]
minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
center = (minimum + maximum) / 2
scene_span = max(maximum.x - minimum.x, maximum.y - minimum.y)
if scene_span <= 0:
    raise RuntimeError("URDF visual meshes have invalid bounds")

camera_data = bpy.data.cameras.new("nursery_preview_camera")
camera = bpy.data.objects.new("nursery_preview_camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (
    center.x - scene_span,
    center.y - scene_span,
    maximum.z + 0.9 * scene_span,
)
camera.data.type = "ORTHO"
camera.data.ortho_scale = 1.25 * scene_span
target = center + Vector((0, 0, 0.4))
camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
scene.camera = camera

output_path.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.render.render(write_still=True)
'''


def _openai_responses_client() -> Any:
    global _OPENAI_RESPONSES_CLIENT
    if _OPENAI_RESPONSES_CLIENT is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ScenePipelineError(
                "The EmbodiedGen environment must provide the OpenAI Python SDK"
            ) from exc
        # OpenAI() reads OPENAI_API_KEY through the SDK's standard env contract.
        _OPENAI_RESPONSES_CLIENT = OpenAI()
    return _OPENAI_RESPONSES_CLIENT


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_config(config_path: str | os.PathLike[str] | None) -> dict[str, Any]:
    path = Path(config_path or DEFAULT_CONFIG).expanduser().resolve()
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ScenePipelineError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ScenePipelineError(f"Invalid JSON configuration: {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ScenePipelineError(f"Configuration must be a JSON object: {path}")
    missing = sorted(REQUIRED_CONFIG_KEYS - config.keys())
    unknown = sorted(config.keys() - REQUIRED_CONFIG_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ScenePipelineError(f"Invalid configuration ({'; '.join(details)})")

    project_root = path.parent.parent
    resolved = dict(config)
    resolved["config_path"] = str(path)
    resolved["project_root"] = str(project_root)
    for key in ("embodiedgen_root", "output_root", "dataset_index"):
        value = Path(str(config[key])).expanduser()
        resolved[key] = str(value if value.is_absolute() else (project_root / value).resolve())

    room_types = config["room_types"]
    room_seeds = config["room_seeds"]
    if not isinstance(room_types, list) or not room_types or not all(
        isinstance(value, str) and value for value in room_types
    ):
        raise ScenePipelineError("room_types must be a non-empty list of strings")
    if len(room_types) != len(set(room_types)):
        raise ScenePipelineError("room_types must be unique")
    unsupported_room_types = sorted(set(room_types) - set(ROOM_CLI_TYPES))
    if unsupported_room_types:
        raise ScenePipelineError(
            f"room_types are unsupported by room-cli: {unsupported_room_types}"
        )
    if not isinstance(room_seeds, list) or not room_seeds or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in room_seeds
    ):
        raise ScenePipelineError("room_seeds must be a non-empty list of integers")
    if len(room_seeds) != len(set(room_seeds)):
        raise ScenePipelineError("room_seeds must be unique")
    if config["room_complexity"] not in {"minimalist", "simple", "medium", "detail"}:
        raise ScenePipelineError("room_complexity is not supported by EmbodiedGen V2")
    if not isinstance(config["distractor_count"], int) or config["distractor_count"] < 0:
        raise ScenePipelineError("distractor_count must be a non-negative integer")
    if not isinstance(config["generate_usd"], bool):
        raise ScenePipelineError("generate_usd must be a boolean")
    return resolved


def _existing_writable_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir() or not os.access(candidate, os.W_OK):
        raise ScenePipelineError(f"Output location is not writable: {path}")
    return candidate


def _validate_native_setup(config: Mapping[str, Any], *, require_room_cli: bool) -> None:
    embodiedgen_root = Path(config["embodiedgen_root"])
    required_sources = (
        embodiedgen_root / "embodied_gen/scripts/room_gen/gen_room.py",
        embodiedgen_root / "embodied_gen/skills/spatial-computing/cli/main.py",
        embodiedgen_root
        / "embodied_gen/skills/asset-retrieval/scripts/retrieve_asset.py",
    )
    missing_sources = [str(path) for path in required_sources if not path.is_file()]
    if missing_sources:
        raise ScenePipelineError(
            "EmbodiedGen V2 source is incomplete or absent; missing: "
            + ", ".join(missing_sources)
        )
    if require_room_cli and shutil.which("room-cli") is None:
        raise ScenePipelineError(
            "room-cli is unavailable; install EmbodiedGen's room profile first"
        )
    dataset_index = Path(config["dataset_index"])
    if not dataset_index.is_file():
        raise ScenePipelineError(f"EmbodiedGen asset index not found: {dataset_index}")
    _existing_writable_parent(Path(config["output_root"]))


def _run_native(
    command: Sequence[str], *, cwd: Path, operation: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ScenePipelineError(f"Could not start {operation}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ScenePipelineError(
            f"{operation} failed with exit code {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    return result


def _resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
    if filename.startswith("file://"):
        filename = filename[7:]
    if filename.startswith("package://"):
        raise ScenePipelineError(
            f"Cannot validate package URI without a package map: {filename}"
        )
    mesh = Path(filename).expanduser()
    return (mesh if mesh.is_absolute() else urdf_path.parent / mesh).resolve()


def _parse_urdf(urdf_path: Path) -> ET.Element:
    try:
        return ET.parse(urdf_path).getroot()
    except FileNotFoundError as exc:
        raise ScenePipelineError(f"URDF not found: {urdf_path}") from exc
    except ET.ParseError as exc:
        raise ScenePipelineError(f"Invalid URDF XML: {urdf_path}: {exc}") from exc


def _validate_urdf_meshes(urdf_path: Path, *, require_floor: bool) -> list[str]:
    root = _parse_urdf(urdf_path)
    link_names = [element.get("name", "") for element in root.findall(".//link")]
    missing = []
    for mesh_element in root.findall(".//mesh"):
        filename = mesh_element.get("filename")
        if filename and not _resolve_mesh_path(filename, urdf_path).is_file():
            missing.append(filename)
    if missing:
        raise ScenePipelineError(
            f"URDF references missing meshes ({urdf_path}): {', '.join(missing[:8])}"
        )
    if require_floor and not any("floor" in name.casefold() for name in link_names):
        raise ScenePipelineError(f"URDF has no floor link: {urdf_path}")
    return link_names


def _parse_native_list(output: str, label: str) -> list[str]:
    match = re.search(rf"{re.escape(label)}:\s*(\[[^\n]*\])", output)
    if match is None:
        raise ScenePipelineError(f"Native spatial inspection omitted {label}")
    try:
        values = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        raise ScenePipelineError(f"Could not parse native {label}") from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ScenePipelineError(f"Native {label} is not a string list")
    return values


def _inspect_room(urdf_path: Path, embodiedgen_root: Path) -> tuple[list[str], list[str]]:
    result = _run_native(
        (
            sys.executable,
            "-m",
            "embodied_gen.skills.spatial-computing.cli.main",
            "--urdf_path",
            str(urdf_path),
            "--list_instances",
        ),
        cwd=embodiedgen_root,
        operation=f"Inspecting room {urdf_path.parent.parent.parent.name}",
    )
    combined = f"{result.stdout}\n{result.stderr}"
    return (
        _parse_native_list(combined, "instance_names"),
        _parse_native_list(combined, "room_names"),
    )


def _room_record(
    *,
    room_id: str,
    room_type: str,
    seed: int,
    complexity: str,
    room_bank: Path,
    embodiedgen_root: Path,
) -> dict[str, Any]:
    urdf_path = room_bank / room_id / "urdf/export_scene/scene.urdf"
    _validate_urdf_meshes(urdf_path, require_floor=True)
    instances, rooms = _inspect_room(urdf_path, embodiedgen_root)
    if not rooms or not any("floor" in name.casefold() for name in rooms):
        raise ScenePipelineError(f"Native inspection found no floor room: {urdf_path}")
    return {
        "room_id": room_id,
        "room_type": room_type,
        "seed": seed,
        "complexity": complexity,
        "urdf_path": urdf_path.relative_to(room_bank).as_posix(),
        "instances": instances,
        "rooms": rooms,
    }


def prepare_room_bank(
    config_path: str | os.PathLike[str] | None = None,
    *,
    room_types: Sequence[str] | None = None,
    room_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Generate missing native rooms and write the validated local bank index."""

    config = _load_config(config_path)
    _validate_native_setup(config, require_room_cli=True)
    selected_room_types = list(config["room_types"] if room_types is None else room_types)
    selected_room_seeds = list(config["room_seeds"] if room_seeds is None else room_seeds)
    if not selected_room_types or not selected_room_seeds:
        raise ScenePipelineError("Room-bank filters cannot be empty")
    unknown_types = sorted(set(selected_room_types) - set(config["room_types"]))
    unknown_seeds = sorted(set(selected_room_seeds) - set(config["room_seeds"]))
    if unknown_types or unknown_seeds:
        raise ScenePipelineError(
            "Room-bank filters must be present in the canonical config"
            + (f"; unknown room types: {unknown_types}" if unknown_types else "")
            + (f"; unknown seeds: {unknown_seeds}" if unknown_seeds else "")
        )
    output_root = Path(config["output_root"])
    room_bank = output_root / "room_bank"
    embodiedgen_root = Path(config["embodiedgen_root"])
    room_bank.mkdir(parents=True, exist_ok=True)

    records_by_id: dict[str, dict[str, Any]] = {}
    existing_bank_path = room_bank / "bank.json"
    if existing_bank_path.is_file():
        try:
            existing_bank = json.loads(existing_bank_path.read_text())
        except json.JSONDecodeError as exc:
            raise ScenePipelineError(f"Invalid existing room bank: {existing_bank_path}") from exc
        for record in existing_bank.get("rooms", []):
            if isinstance(record, dict) and isinstance(record.get("room_id"), str):
                records_by_id[record["room_id"]] = record
    generated = 0
    skipped = 0
    for room_type in selected_room_types:
        for seed in selected_room_seeds:
            room_id = f"{room_type}_seed{seed}"
            scene_urdf = room_bank / room_id / "urdf/export_scene/scene.urdf"
            valid_existing = False
            if scene_urdf.is_file():
                try:
                    _validate_urdf_meshes(scene_urdf, require_floor=True)
                    valid_existing = True
                except ScenePipelineError:
                    valid_existing = False
            if not valid_existing:
                command = [
                    "room-cli",
                    "-m",
                    "embodied_gen.scripts.room_gen.gen_room",
                    "--output-root",
                    str(room_bank),
                    "--room-type",
                    ROOM_CLI_TYPES[room_type],
                    "--seed",
                    str(seed),
                    "--complexity",
                    config["room_complexity"],
                    "--urdf",
                    "--usd" if config["generate_usd"] else "--no-usd",
                ]
                _run_native(
                    command,
                    cwd=embodiedgen_root,
                    operation=f"Generating {room_id}",
                )
                generated += 1
            else:
                skipped += 1
            records_by_id[room_id] = _room_record(
                room_id=room_id,
                room_type=room_type,
                seed=seed,
                complexity=config["room_complexity"],
                room_bank=room_bank,
                embodiedgen_root=embodiedgen_root,
            )

    configured_order = {
        f"{room_type}_seed{seed}": index
        for index, (room_type, seed) in enumerate(
            (room_type, seed)
            for room_type in config["room_types"]
            for seed in config["room_seeds"]
        )
    }
    records = []
    for room_id in sorted(configured_order, key=configured_order.get):
        record = records_by_id.get(room_id)
        if record is None:
            continue
        retained_urdf = room_bank / str(record.get("urdf_path", ""))
        try:
            _validate_urdf_meshes(retained_urdf, require_floor=True)
        except ScenePipelineError:
            continue
        records.append(record)
    room_ids = [record["room_id"] for record in records]
    if len(room_ids) != len(set(room_ids)):
        raise ScenePipelineError("Room bank contains duplicate room IDs")
    bank = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rooms": records,
    }
    _write_json(room_bank / "bank.json", bank)
    return {
        "bank_path": str(room_bank / "bank.json"),
        "room_count": len(records),
        "generated": generated,
        "skipped": skipped,
    }


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    room = plan.get("room")
    if not isinstance(room, str) or not room:
        raise ScenePipelineError("Activity plan requires a non-empty room")
    validated: dict[str, Any] = {"room": room}
    for group in ("objects", "distractors"):
        entries = plan.get(group, [])
        if not isinstance(entries, list):
            raise ScenePipelineError(f"Activity plan {group} must be a list")
        normalized_entries = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ScenePipelineError(f"Every {group} entry must be an object")
            name = entry.get("name")
            relation = entry.get("relation")
            target = entry.get("target")
            if not isinstance(name, str) or not name:
                raise ScenePipelineError(f"Every {group} entry requires a name")
            if relation not in SUPPORTED_RELATIONS:
                raise ScenePipelineError(
                    f"Unsupported relation {relation!r}; expected one of "
                    + ", ".join(sorted(SUPPORTED_RELATIONS))
                )
            if relation != "in_room" and (not isinstance(target, str) or not target):
                raise ScenePipelineError(f"Relation {relation!r} requires a target")
            normalized_entries.append(
                {"name": name, "relation": relation, "target": target or room}
            )
        validated[group] = normalized_entries
    return validated


def _import_native_layout(embodiedgen_root: Path) -> Any:
    if not (embodiedgen_root / "embodied_gen/models/layout.py").is_file():
        raise ScenePipelineError(
            f"EmbodiedGen V2 layout module not found under {embodiedgen_root}"
        )
    root_text = str(embodiedgen_root)
    added_to_path = root_text not in sys.path
    if added_to_path:
        sys.path.insert(0, root_text)
    try:
        return importlib.import_module("embodied_gen.models.layout")
    except Exception as exc:
        raise ScenePipelineError(f"Could not import native EmbodiedGen layout: {exc}") from exc
    finally:
        if added_to_path:
            sys.path.remove(root_text)


def _native_layout_info(activity: str, embodiedgen_root: Path) -> Any:
    layout = _import_native_layout(embodiedgen_root)
    disassemble_prompt = layout.LAYOUT_DISASSEMBLE_PROMPT + """

Nursery scene-only compatibility (these rules override robot-specific rules above):
- Plan a static activity scene, not a robotic task; never add a robot or human asset.
- Keep the native output keys for LayoutInfo compatibility, but set "robot" to null.
- Treat activity objects as fixed visual/collision meshes; do not request soft-body physics.
- Preserve an explicit on, beside, inside, or in-room relation from the task description.
"""
    disassembler = layout.LayoutDesigner(
        gpt_client=_RESPONSES_GPT_CLIENT,
        system_prompt=disassemble_prompt,
    )
    hierarchy_prompt = layout.LAYOUT_HIERARCHY_PROMPT + """

Nursery scene-only compatibility:
- A child may be "BESIDE" a parent when the task explicitly says beside or next to.
- Ignore the robot field and do not emit a robot or human node in the layout tree.
- Honor explicit object relations from the task instead of robot-manipulation defaults.
- Do not add coordinates, cameras, human poses, or custom physics parameters.
"""
    grapher = layout.LayoutDesigner(
        gpt_client=_RESPONSES_GPT_CLIENT,
        system_prompt=hierarchy_prompt,
    )
    describer = layout.LayoutDesigner(
        gpt_client=_RESPONSES_GPT_CLIENT,
        system_prompt=layout.LAYOUT_DESCRIBER_PROMPT,
    )
    try:
        relation = disassembler(activity)
        tree = grapher(relation)
        object_mapping = layout.Scene3DItemEnum.object_mapping(relation)
        description_prompt = f'{relation["task_desc"]} {object_mapping}'
        descriptions = describer(description_prompt)
        return layout.LayoutInfo(tree, relation, descriptions, object_mapping)
    except ScenePipelineError:
        raise
    except Exception as exc:
        raise ScenePipelineError(f"Native EmbodiedGen planning failed: {exc}") from exc


def _canonical_room_type(value: str) -> str:
    aliases = {
        "bedroom": "Bedroom",
        "living room": "LivingRoom",
        "livingroom": "LivingRoom",
        "kitchen": "Kitchen",
        "bathroom": "Bathroom",
        "dining room": "DiningRoom",
        "diningroom": "DiningRoom",
        "office": "Office",
    }
    normalized = _normalized_name(value)
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ScenePipelineError(
            f"Native planner selected unsupported room type {value!r}"
        ) from exc


def _explicit_activity_relation(
    activity: str, object_name: str
) -> tuple[str, str] | None:
    name_pattern = re.escape(_normalized_name(object_name)).replace(r"\ ", r"\s+")
    match = re.search(
        rf"\b(?:a|an|the)?\s*{name_pattern}\s+"
        rf"(on|beside|inside)\s+(?:a|an|the)\s+"
        rf"([a-z0-9][a-z0-9 _-]*?)(?=\s*,|\s+and\b|[.;]|$)",
        activity.casefold().replace("_", " "),
    )
    if match is None:
        return None
    return match.group(1), _normalized_name(match.group(2))


def _plan_from_layout_info(layout_info: Any, *, activity: str = "") -> dict[str, Any]:
    relation = layout_info.relation
    tree = layout_info.tree
    if not isinstance(relation, Mapping) or not isinstance(tree, Mapping):
        raise ScenePipelineError("Native LayoutInfo has invalid relation or tree data")
    room_value = relation.get("background")
    context = relation.get("context")
    manipulated = relation.get("manipulated_objs", [])
    distractors = relation.get("distractor_objs", [])
    if not isinstance(room_value, str) or not isinstance(context, str):
        raise ScenePipelineError("Native LayoutInfo omitted background or context")
    if not isinstance(manipulated, list) or not isinstance(distractors, list):
        raise ScenePipelineError("Native LayoutInfo object groups must be lists")

    edges: dict[str, tuple[str, str]] = {}
    for parent, children in tree.items():
        if not isinstance(parent, str) or not isinstance(children, list):
            continue
        for edge in children:
            if (
                isinstance(edge, (list, tuple))
                and len(edge) == 2
                and isinstance(edge[0], str)
                and isinstance(edge[1], str)
            ):
                edges[edge[0]] = (parent, edge[1].rsplit(".", 1)[-1].upper())

    relation_names = {
        "ON": "on",
        "INSIDE": "inside",
        "FLOOR": "in_room",
        "IN": "in_room",
        "BESIDE": "beside",
    }

    def convert(name: Any) -> dict[str, str]:
        if not isinstance(name, str) or name not in edges:
            raise ScenePipelineError(f"Native LayoutInfo omitted placement for {name!r}")
        parent, native_relation = edges[name]
        if native_relation not in relation_names:
            raise ScenePipelineError(
                f"Native LayoutInfo used unsupported relation {native_relation!r}"
            )
        local_relation = relation_names[native_relation]
        target = room_value if local_relation == "in_room" else parent
        return {"name": name, "relation": local_relation, "target": target}

    explicit_objects = []
    other_objects = []
    other_distractors = []
    seen: set[str] = set()
    for group, destination in (
        (manipulated, other_objects),
        (distractors, other_distractors),
    ):
        for name in group:
            normalized = _normalized_name(str(name))
            if normalized in seen:
                continue
            seen.add(normalized)
            entry = convert(name)
            explicit = _explicit_activity_relation(activity, str(name))
            if explicit is not None:
                entry["relation"], entry["target"] = explicit
                explicit_objects.append(entry)
            else:
                destination.append(entry)

    target_names = {
        _normalized_name(entry["target"])
        for entry in explicit_objects + other_objects + other_distractors
        if entry["relation"] != "in_room"
    }
    ignored_names = {"person", "human", "robot", "franka", "ur5", "piper"}

    def keep(entry: Mapping[str, str]) -> bool:
        normalized = _normalized_name(entry["name"])
        return normalized not in target_names and normalized not in ignored_names

    return _validate_plan(
        {
            "room": _canonical_room_type(room_value),
            "objects": explicit_objects + [entry for entry in other_objects if keep(entry)],
            "distractors": [entry for entry in other_distractors if keep(entry)],
        }
    )


def plan_activity(
    activity: str,
    *,
    config_path: str | os.PathLike[str] | None = None,
    planner: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a small plan through native EmbodiedGen multi-stage planning."""

    if not isinstance(activity, str) or not activity.strip():
        raise ScenePipelineError("activity must be a non-empty string")
    if planner is not None:
        return _validate_plan(planner(activity.strip()))
    config = _load_config(config_path)
    layout_info = _native_layout_info(
        activity.strip(), Path(config["embodiedgen_root"])
    )
    plan = _plan_from_layout_info(layout_info, activity=activity.strip())
    plan["distractors"] = plan["distractors"][: config["distractor_count"]]
    return plan


def select_room(
    plan: Mapping[str, Any], bank: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    """Select an exact room type deterministically from the prepared bank."""

    rooms = bank.get("rooms")
    if not isinstance(rooms, list):
        raise ScenePipelineError("bank.json has no rooms list")
    candidates = sorted(
        (
            room
            for room in rooms
            if isinstance(room, dict) and room.get("room_type") == plan.get("room")
        ),
        key=lambda room: str(room.get("room_id", "")),
    )
    if not candidates:
        raise ScenePipelineError(f"No prepared room has type {plan.get('room')!r}")
    return dict(candidates[seed % len(candidates)])


def _normalized_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " ")))


def _resolve_exactish_name(query: str, names: Sequence[str]) -> str | None:
    normalized_query = _normalized_name(query)
    normalized = [(name, _normalized_name(name)) for name in names]
    exact = [name for name, value in normalized if value == normalized_query]
    if exact:
        return sorted(exact)[0]
    contained = [
        name
        for name, value in normalized
        if normalized_query and normalized_query in value.split()
    ]
    if contained:
        return sorted(contained, key=lambda name: (len(name), name))[0]
    phrase = [name for name, value in normalized if normalized_query in value]
    return sorted(phrase, key=lambda name: (len(name), name))[0] if phrase else None


def _asset_candidates(
    query: str,
    *,
    index_file: Path,
    dataset_root: Path,
    embodiedgen_root: Path,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if not index_file.is_file():
        return []
    script = embodiedgen_root / "embodied_gen/skills/asset-retrieval/scripts/retrieve_asset.py"
    try:
        result = _run_native(
            (
                sys.executable,
                str(script),
                query,
                "--dataset-root",
                str(dataset_root),
                "--index-file",
                str(index_file),
                "--top-k",
                str(top_k),
                "--format",
                "json",
            ),
            cwd=embodiedgen_root,
            operation=f"Retrieving asset {query!r}",
        )
    except ScenePipelineError as exc:
        if "No matching assets found" in str(exc):
            return []
        raise
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ScenePipelineError(f"Asset retrieval returned invalid JSON for {query!r}") from exc
    if not isinstance(payload, list):
        raise ScenePipelineError(f"Asset retrieval returned an invalid result for {query!r}")
    return [candidate for candidate in payload if isinstance(candidate, dict)]


def _deterministic_asset_choice(
    query: str, candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    normalized_query = _normalized_name(query)
    for candidate in candidates:
        categories = (
            candidate.get("category", ""),
            candidate.get("secondary_category", ""),
            candidate.get("primary_category", ""),
        )
        if any(_normalized_name(str(value)) == normalized_query for value in categories):
            return dict(candidate)
    return None


def _json_object_from_response(text: str, *, operation: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ScenePipelineError(f"{operation} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ScenePipelineError(f"{operation} must return one JSON object")
    return payload


def _select_ambiguous_assets(
    ambiguous: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any] | None]:
    """Select every ambiguous retrieval result in one shared Responses call."""

    choices: dict[str, list[dict[str, Any]]] = {}
    for name, candidates in ambiguous.items():
        choices[name] = [
            {
                "index": index,
                "primary_category": candidate.get("primary_category"),
                "secondary_category": candidate.get("secondary_category"),
                "category": candidate.get("category"),
                "description": candidate.get("description"),
            }
            for index, candidate in enumerate(candidates)
        ]
    prompt = (
        "Select the most suitable static 3D asset for each requested object. "
        "Return only one JSON object mapping every object name to its candidate "
        "index, or null when none is suitable. Do not omit keys.\n\n"
        + json.dumps(choices, ensure_ascii=False)
    )
    payload = _json_object_from_response(
        _RESPONSES_GPT_CLIENT.query(prompt), operation="Batched asset selection"
    )
    if set(payload) != set(ambiguous):
        raise ScenePipelineError(
            "Batched asset selection must return exactly the requested object names"
        )

    selected: dict[str, Mapping[str, Any] | None] = {}
    for name, candidates in ambiguous.items():
        index = payload[name]
        if index is None:
            selected[name] = None
        elif (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(candidates)
        ):
            selected[name] = dict(candidates[index])
        else:
            raise ScenePipelineError(
                f"Batched asset selection returned an invalid index for {name!r}"
            )
    return selected


def _safe_asset_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_") or "asset"
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{slug}_{digest}"


def _append_generated_asset(index_file: Path, *, name: str, urdf_path: Path) -> None:
    fields = [
        "uuid",
        "primary_category",
        "secondary_category",
        "category",
        "description",
        "generate_time",
        "urdf_path",
    ]
    index_file.parent.mkdir(parents=True, exist_ok=True)
    existing_paths: set[str] = set()
    if index_file.is_file():
        with index_file.open(newline="", encoding="utf-8") as stream:
            existing_paths = {row.get("urdf_path", "") for row in csv.DictReader(stream)}
    relative_path = urdf_path.relative_to(index_file.parent).as_posix()
    if relative_path in existing_paths:
        return
    write_header = not index_file.exists() or index_file.stat().st_size == 0
    with index_file.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "uuid": hashlib.sha256(str(urdf_path).encode()).hexdigest()[:16],
                "primary_category": "generated",
                "secondary_category": "generated",
                "category": name,
                "description": name,
                "generate_time": datetime.now(timezone.utc).isoformat(),
                "urdf_path": relative_path,
            }
        )


def _generate_assets(
    names: Sequence[str], *, output_root: Path, embodiedgen_root: Path
) -> dict[str, dict[str, Any]]:
    if not names:
        return {}
    asset_names = [_safe_asset_name(name) for name in names]
    command = ["text3d-cli", "--prompts", *names, "--asset_names", *asset_names]
    command.extend(["--output_root", str(output_root)])
    _run_native(command, cwd=embodiedgen_root, operation="Generating missing assets")
    local_index = output_root / "local_assets.csv"
    generated: dict[str, dict[str, Any]] = {}
    for name, asset_name in zip(names, asset_names, strict=True):
        urdf_path = output_root / "asset3d" / asset_name / "result" / f"{asset_name}.urdf"
        if not urdf_path.is_file():
            raise ScenePipelineError(f"Text-to-3D did not produce expected URDF: {urdf_path}")
        _append_generated_asset(local_index, name=name, urdf_path=urdf_path)
        generated[name] = {"urdf_path": str(urdf_path), "source": "generated"}
    return generated


def resolve_assets(
    plan: Mapping[str, Any],
    room: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    candidate_selector: Callable[
        [Mapping[str, Sequence[Mapping[str, Any]]]],
        Mapping[str, Mapping[str, Any] | None],
    ]
    | None = None,
) -> dict[str, Any]:
    """Reuse room instances, retrieve exact assets, and generate only misses."""

    instances = room.get("instances", [])
    if not isinstance(instances, list):
        raise ScenePipelineError("Selected room has no instance list")
    requested = list(plan.get("objects", [])) + list(plan.get("distractors", []))
    resolved: dict[str, dict[str, Any]] = {}
    missing = []
    for entry in requested:
        name = str(entry["name"])
        existing = _resolve_exactish_name(name, instances)
        if existing is not None:
            resolved[name] = {"source": "room", "instance": existing}
        elif name not in missing:
            missing.append(name)

    embodiedgen_root = Path(config["embodiedgen_root"])
    official_index = Path(config["dataset_index"])
    generated_root = Path(config["output_root"]) / "generated_assets"
    local_index = generated_root / "local_assets.csv"
    unresolved = []
    ambiguous: dict[str, list[dict[str, Any]]] = {}
    for name in missing:
        candidates = _asset_candidates(
            name,
            index_file=local_index,
            dataset_root=generated_root,
            embodiedgen_root=embodiedgen_root,
        )
        candidates.extend(
            _asset_candidates(
                name,
                index_file=official_index,
                dataset_root=official_index.parent,
                embodiedgen_root=embodiedgen_root,
            )
        )
        choice = _deterministic_asset_choice(name, candidates)
        if choice is None:
            if candidates:
                ambiguous[name] = candidates
            else:
                unresolved.append(name)
        else:
            urdf_path = Path(str(choice["urdf_path"]))
            if not urdf_path.is_file():
                raise ScenePipelineError(f"Retrieved asset URDF does not exist: {urdf_path}")
            resolved[name] = {
                "source": "retrieved",
                "urdf_path": str(urdf_path),
                "candidate": choice,
            }
    if ambiguous:
        selections = (candidate_selector or _select_ambiguous_assets)(ambiguous)
        if not isinstance(selections, Mapping):
            raise ScenePipelineError("Asset candidate selector must return a mapping")
        for name, candidates in ambiguous.items():
            selection = selections.get(name)
            if selection is None:
                unresolved.append(name)
                continue
            if not isinstance(selection, Mapping):
                raise ScenePipelineError(f"Invalid asset selection for {name!r}")
            selected_path = str(selection.get("urdf_path", ""))
            matched = next(
                (
                    candidate
                    for candidate in candidates
                    if str(candidate.get("urdf_path", "")) == selected_path
                ),
                None,
            )
            if matched is None:
                raise ScenePipelineError(
                    f"Asset selector returned a candidate not offered for {name!r}"
                )
            urdf_path = Path(selected_path)
            if not urdf_path.is_file():
                raise ScenePipelineError(f"Retrieved asset URDF does not exist: {urdf_path}")
            resolved[name] = {
                "source": "retrieved",
                "urdf_path": str(urdf_path),
                "candidate": matched,
            }
    resolved.update(
        _generate_assets(
            unresolved,
            output_root=generated_root,
            embodiedgen_root=embodiedgen_root,
        )
    )
    return resolved


def _asset_visual_obj(urdf_path: Path) -> Path:
    root = _parse_urdf(urdf_path)
    visual_paths = []
    for mesh in root.findall(".//visual//mesh"):
        filename = mesh.get("filename")
        if filename:
            path = _resolve_mesh_path(filename, urdf_path)
            if path.suffix.casefold() == ".obj":
                visual_paths.append(path)
    if not visual_paths:
        raise ScenePipelineError(f"Asset URDF has no visual OBJ: {urdf_path}")
    visual = visual_paths[0]
    collision = visual.with_name(f"{visual.stem}_collision.obj")
    if not visual.is_file() or not collision.is_file():
        raise ScenePipelineError(
            f"Asset requires sibling visual/collision OBJ files: {visual}, {collision}"
        )
    return visual


def _create_workspace(scene_dir: Path, room_urdf: Path) -> Path:
    workspace = scene_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copy2(room_urdf, workspace / "scene.urdf")
    for source in room_urdf.parent.iterdir():
        if source.name == "scene.urdf":
            continue
        destination = workspace / source.name
        try:
            destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        except OSError:
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
    return workspace


def _new_scene_id(activity: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{activity}".encode()).hexdigest()[:10]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{digest}"


def _placement_config(
    entry: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
    room: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any] | None:
    if asset.get("source") == "room":
        return None
    relation = str(entry["relation"])
    if relation == "inside":
        # TODO: Enable only when native FloorplanManager provides true
        # collision-aware container-volume placement.
        raise ScenePipelineError(
            "EmbodiedGen V2 FloorplanManager has no collision-aware inside-container "
            "primitive; use on, beside, or in_room until that native capability exists"
        )
    rooms = room.get("rooms", [])
    instances = room.get("instances", [])
    exact_room = _resolve_exactish_name(str(room["room_type"]), rooms)
    if exact_room is None and rooms:
        exact_room = sorted(rooms)[0]
    target = str(entry.get("target", ""))
    placement: dict[str, Any] = {
        "asset_path": str(_asset_visual_obj(Path(str(asset["urdf_path"])))),
        "instance_key": f"{_safe_asset_name(str(entry['name']))}_{ordinal}",
    }
    if exact_room:
        placement["in_room"] = exact_room
    if relation in {"on", "beside"}:
        resolved_target = _resolve_exactish_name(target, instances)
        if resolved_target is None:
            raise ScenePipelineError(
                f"Could not resolve placement target {target!r} in {room['room_id']}"
            )
        placement[f"{relation}_instance"] = resolved_target
        if relation == "on":
            placement["place_strategy"] = "top"
    return placement


def _validate_placement_targets(
    plan: Mapping[str, Any], room: Mapping[str, Any]
) -> None:
    instances = room.get("instances", [])
    rooms = room.get("rooms", [])
    if not isinstance(instances, list) or not isinstance(rooms, list):
        raise ScenePipelineError("Selected room is missing native instance metadata")
    if not rooms:
        raise ScenePipelineError(f"Selected room has no native floor room: {room['room_id']}")
    for entry in list(plan.get("objects", [])) + list(plan.get("distractors", [])):
        relation = str(entry["relation"])
        if relation == "inside":
            # TODO: Enable only when native FloorplanManager provides true
            # collision-aware container-volume placement.
            raise ScenePipelineError(
                "EmbodiedGen V2 FloorplanManager has no collision-aware inside-container "
                "primitive; use on, beside, or in_room until that native capability exists"
            )
        if relation in {"on", "beside"} and _resolve_exactish_name(
            str(entry["target"]), instances
        ) is None:
            raise ScenePipelineError(
                f"Could not resolve placement target {entry['target']!r} in "
                f"{room['room_id']}"
            )


def compose_scene(
    activity: str,
    plan: Mapping[str, Any],
    room: Mapping[str, Any],
    assets: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create an isolated workspace and batch-place all non-room assets."""

    output_root = Path(config["output_root"])
    room_bank = output_root / "room_bank"
    room_urdf = room_bank / str(room["urdf_path"])
    original_digest = hashlib.sha256(room_urdf.read_bytes()).hexdigest()
    scene_dir = output_root / "scenes" / _new_scene_id(activity, seed)
    workspace = _create_workspace(scene_dir, room_urdf)
    _write_json(scene_dir / "activity.json", {"activity": activity, "seed": seed})
    _write_json(scene_dir / "plan.json", plan)
    _write_json(scene_dir / "selected_room.json", room)

    requested = list(plan.get("objects", [])) + list(plan.get("distractors", []))
    placements = []
    placed_names = []
    for ordinal, entry in enumerate(requested, 1):
        placement = _placement_config(
            entry,
            asset=assets[str(entry["name"])],
            room=room,
            ordinal=ordinal,
        )
        if placement is not None:
            placements.append(placement)
            placed_names.append(str(entry["name"]))

    scene_urdf = workspace / "scene.urdf"
    updated_urdf = workspace / "scene_updated.urdf"
    if placements:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="batch_insert_",
            dir=scene_dir,
            encoding="utf-8",
            delete=False,
        ) as stream:
            json.dump(placements, stream, indent=2)
            batch_path = Path(stream.name)
        try:
            _run_native(
                (
                    sys.executable,
                    "-m",
                    "embodied_gen.skills.spatial-computing.cli.main",
                    "--urdf_path",
                    str(scene_urdf),
                    "--batch_insert_config",
                    str(batch_path),
                ),
                cwd=Path(config["embodiedgen_root"]),
                operation="Batch-composing scene",
            )
        finally:
            batch_path.unlink(missing_ok=True)
    else:
        shutil.copy2(scene_urdf, updated_urdf)

    if not updated_urdf.is_file():
        raise ScenePipelineError(f"Native placement did not create {updated_urdf}")
    _validate_urdf_meshes(updated_urdf, require_floor=True)
    if hashlib.sha256(room_urdf.read_bytes()).hexdigest() != original_digest:
        raise ScenePipelineError(f"Immutable room bank entry changed: {room['room_id']}")

    retrieved_count = sum(asset.get("source") == "retrieved" for asset in assets.values())
    generated_count = sum(asset.get("source") == "generated" for asset in assets.values())
    requested_names = [str(entry["name"]) for entry in requested]
    retrieved_names = [
        name for name, asset in assets.items() if asset.get("source") == "retrieved"
    ]
    generated_names = [
        name for name, asset in assets.items() if asset.get("source") == "generated"
    ]
    manifest = {
        "activity": activity,
        "seed": seed,
        "room_id": room["room_id"],
        "scene_urdf": str(updated_urdf),
        "assets_retrieved": retrieved_count,
        "assets_generated": generated_count,
        "placements_requested": len(placements),
        "placements_succeeded": len(placements),
        "requested_objects": requested_names,
        "retrieved_objects": retrieved_names,
        "generated_objects": generated_names,
        "placed_objects": placed_names,
    }
    _write_json(scene_dir / "scene_manifest.json", manifest)
    return manifest


def build_scene(
    activity: str,
    *,
    seed: int = 0,
    config_path: str | os.PathLike[str] | None = None,
    plan: Mapping[str, Any] | None = None,
    planner: Callable[[str], Mapping[str, Any]] | None = None,
    candidate_selector: Callable[
        [Mapping[str, Sequence[Mapping[str, Any]]]],
        Mapping[str, Mapping[str, Any] | None],
    ]
    | None = None,
) -> dict[str, Any]:
    """Build one static activity scene from a prepared immutable room bank."""

    config = _load_config(config_path)
    _validate_native_setup(config, require_room_cli=False)
    validated_plan = _validate_plan(plan) if plan is not None else plan_activity(
        activity, config_path=config["config_path"], planner=planner
    )
    if len(validated_plan["distractors"]) > config["distractor_count"]:
        raise ScenePipelineError(
            f"Activity plan exceeds distractor_count={config['distractor_count']}"
        )
    bank_path = Path(config["output_root"]) / "room_bank/bank.json"
    try:
        bank = json.loads(bank_path.read_text())
    except FileNotFoundError as exc:
        raise ScenePipelineError(
            f"Room bank not prepared: {bank_path}; run prepare-bank first"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ScenePipelineError(f"Invalid room bank JSON: {bank_path}") from exc
    room = select_room(validated_plan, bank, seed=seed)
    _validate_placement_targets(validated_plan, room)
    assets = resolve_assets(
        validated_plan,
        room,
        config=config,
        candidate_selector=candidate_selector,
    )
    return compose_scene(
        activity,
        validated_plan,
        room,
        assets,
        seed=seed,
        config=config,
    )


def _resolve_preview_source(
    scene: str | os.PathLike[str], config: Mapping[str, Any]
) -> tuple[str, Path]:
    requested = Path(scene).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend(
            (
                Path(config["project_root"]) / requested,
                Path(config["output_root"]) / "room_bank" / requested,
            )
        )
    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.suffix.casefold() in {".blend", ".urdf"}:
            source_candidates = (candidate,)
        else:
            source_candidates = (
                candidate / "workspace/scene_updated.urdf",
                candidate / "scene_updated.urdf",
                candidate / "blender/scene.blend",
                candidate / "scene.blend",
                candidate / "urdf/export_scene/scene.urdf",
            )
        for source_path in source_candidates:
            checked.append(source_path)
            if source_path.is_file():
                return source_path.suffix.casefold()[1:], source_path
    raise ScenePipelineError(
        "No EmbodiedGen scene.blend or scene_updated.urdf found; checked: "
        + ", ".join(str(path) for path in checked)
    )


def preview_scene(
    scene: str | os.PathLike[str],
    *,
    config_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Render a fast isometric preview of a room-bank or composed scene."""

    config = _load_config(config_path)
    source_type, source_path = _resolve_preview_source(scene, config)
    if source_type == "urdf":
        _validate_urdf_meshes(source_path, require_floor=True)
    embodiedgen_root = Path(config["embodiedgen_root"])
    bundled_blender = embodiedgen_root / "thirdparty/infinigen/blender/blender"
    blender_command = (
        str(bundled_blender)
        if bundled_blender.is_file()
        else shutil.which("blender")
    )
    if blender_command is None:
        raise ScenePipelineError(
            "Blender is unavailable; install EmbodiedGen's room profile first"
        )

    if source_path.parent.name == "blender":
        scene_id = source_path.parent.parent.name
    elif source_path.parent.name == "workspace":
        scene_id = source_path.parent.parent.name
    elif source_path.name == "scene.urdf" and source_path.parent.name == "export_scene":
        scene_id = source_path.parents[2].name
    else:
        scene_id = source_path.stem
    if output_path is None:
        preview_path = Path(config["output_root"]) / "previews" / f"{scene_id}.png"
    else:
        requested_output = Path(output_path).expanduser()
        preview_path = (
            requested_output
            if requested_output.is_absolute()
            else Path(config["project_root"]) / requested_output
        )
    preview_path = preview_path.resolve()
    if preview_path.suffix.casefold() != ".png":
        raise ScenePipelineError("Preview output must use a .png extension")
    _existing_writable_parent(preview_path.parent)

    with tempfile.TemporaryDirectory(prefix="nursery-embodied-preview-") as temp_dir:
        script_path = Path(temp_dir) / "render_preview.py"
        if source_type == "blend":
            script_path.write_text(_BLENDER_PREVIEW_SCRIPT)
            command = [
                blender_command,
                "-b",
                str(source_path),
                "-P",
                str(script_path),
                "--",
                str(preview_path),
            ]
        else:
            script_path.write_text(_BLENDER_URDF_PREVIEW_SCRIPT)
            command = [
                blender_command,
                "-b",
                "--factory-startup",
                "-P",
                str(script_path),
                "--",
                str(source_path),
                str(preview_path),
            ]
        _run_native(
            command,
            cwd=embodiedgen_root,
            operation=f"Previewing {scene_id}",
        )
    if not preview_path.is_file() or preview_path.stat().st_size == 0:
        raise ScenePipelineError(f"Blender did not produce a preview: {preview_path}")
    return {
        "scene_id": scene_id,
        f"scene_{source_type}": str(source_path),
        "preview_path": str(preview_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nursery.embodiedgen_scene",
        description="Prepare EmbodiedGen rooms and compose static Nursery scenes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-bank", help="Prepare the reusable room bank")
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare.add_argument(
        "--room-type",
        action="append",
        dest="room_types",
        help="Configured room type to prepare; repeat for a smoke-test subset",
    )
    prepare.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="room_seeds",
        help="Configured seed to prepare; repeat for a smoke-test subset",
    )
    build = subparsers.add_parser("build", help="Build one isolated activity scene")
    build.add_argument("--activity", required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--config", default=str(DEFAULT_CONFIG))
    build.add_argument(
        "--plan",
        type=Path,
        help="Use an explicit plan JSON instead of native EmbodiedGen planning",
    )
    preview = subparsers.add_parser(
        "preview", help="Render a room-bank or composed-scene preview"
    )
    preview.add_argument(
        "--scene",
        required=True,
        help="Room ID, scene directory, scene.blend, or scene_updated.urdf",
    )
    preview.add_argument("--config", default=str(DEFAULT_CONFIG))
    preview.add_argument(
        "--output",
        type=Path,
        help="PNG path (default: outputs/embodiedgen_scene/previews/<scene-id>.png)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preparation or activity-to-scene command."""

    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare-bank":
            result = prepare_room_bank(
                args.config,
                room_types=args.room_types,
                room_seeds=args.room_seeds,
            )
        elif args.command == "build":
            explicit_plan = None
            if args.plan is not None:
                try:
                    explicit_plan = json.loads(args.plan.read_text())
                except (FileNotFoundError, json.JSONDecodeError) as exc:
                    raise ScenePipelineError(f"Could not read plan JSON: {args.plan}") from exc
            result = build_scene(
                args.activity,
                seed=args.seed,
                config_path=args.config,
                plan=explicit_plan,
            )
        else:
            result = preview_scene(
                args.scene,
                config_path=args.config,
                output_path=args.output,
            )
    except ScenePipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
