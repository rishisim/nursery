"""Blender-side renderer for the prospective asset-rich fidelity pilot."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore


EPISODES = {
    "playroom_duck_push": {
        "asset": "rubber_duck_toy",
        "label": "Dax",
        "noun": "Ente",
        "action": "schieben",
        "room": "playroom",
        "color": (0.22, 0.48, 0.72, 1.0),
        "start": (-0.55, 0.05, 0.62),
        "end": (0.55, 0.05, 0.62),
    },
    "tabletop_apple_lift": {
        "asset": "food_apple_01",
        "label": "Koba",
        "noun": "Apfel",
        "action": "heben",
        "room": "tabletop",
        "color": (0.52, 0.34, 0.18, 1.0),
        "start": (-0.35, 0.1, 0.78),
        "end": (-0.05, 0.08, 1.25),
    },
    "livingroom_ball_roll": {
        "asset": "baseball_01",
        "label": "Mipa",
        "noun": "Ball",
        "action": "rollen",
        "room": "living_room",
        "color": (0.28, 0.52, 0.32, 1.0),
        "start": (-0.65, 0.02, 0.61),
        "end": (0.65, 0.02, 0.61),
    },
}


def material(name: str, color: tuple[float, float, float, float], roughness: float = 0.55):
    value = bpy.data.materials.new(name)
    value.diffuse_color = color
    value.use_nodes = True
    principled = value.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = roughness
    return value


def cube(name: str, location, scale, mat):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def import_target(path: Path, location):
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = list(bpy.context.selected_objects)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {path}")
    root = bpy.data.objects.new("target_root", None)
    bpy.context.collection.objects.link(root)
    for obj in imported:
        if obj.parent is None:
            obj.parent = root
    bounds = []
    for obj in meshes:
        bounds.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    extent = max(
        max(v.x for v in bounds) - min(v.x for v in bounds),
        max(v.y for v in bounds) - min(v.y for v in bounds),
        max(v.z for v in bounds) - min(v.z for v in bounds),
    )
    root.scale = (0.28 / extent,) * 3
    root.location = location
    for obj in meshes:
        obj["semantic_class"] = "target"
        obj.pass_index = 1
    return root, meshes


def create_hand(start, end):
    skin = material("neutral_skin", (0.62, 0.35, 0.22, 1.0), 0.48)
    forearm = cube("visible_forearm", (start[0] - 0.55, start[1] - 0.25, start[2] + 0.08),
                   (0.34, 0.09, 0.09), skin)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12,
                                       location=(start[0] - 0.18, start[1] - 0.16, start[2] + 0.05),
                                       scale=(0.19, 0.13, 0.07))
    palm = bpy.context.object
    palm.name = "visible_hand"
    palm.data.materials.append(skin)
    palm.pass_index = 2
    for i in range(4):
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.022, depth=0.18,
                                           location=(start[0] - 0.02 + i * 0.045,
                                                     start[1] - 0.08, start[2] + 0.025),
                                           rotation=(math.pi / 2, 0, 0))
        finger = bpy.context.object
        finger.name = f"visible_finger_{i}"
        finger.data.materials.append(skin)
        finger.parent = palm
    forearm.parent = palm
    palm.location = (0, 0, 0)
    palm.keyframe_insert(data_path="location", frame=1)
    contact_delta = Vector(start) - Vector((start[0] - 0.18, start[1] - 0.16, start[2] + 0.05))
    palm.location = contact_delta
    palm.keyframe_insert(data_path="location", frame=180)
    palm.keyframe_insert(data_path="location", frame=300)
    palm.location = contact_delta + Vector(end) - Vector(start)
    palm.keyframe_insert(data_path="location", frame=450)
    palm.location += Vector((0.0, -0.25, 0.18))
    palm.keyframe_insert(data_path="location", frame=600)
    return palm


def render(episode_id: str, asset_root: Path, output_root: Path):
    spec = EPISODES[episode_id]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 360
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = 600
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("warm_indoor_world")
    scene.world.color = (0.035, 0.045, 0.065)

    floor_mat = material("floor", spec["color"], 0.62)
    wall_mat = material("walls", (0.55, 0.51, 0.44, 1.0), 0.75)
    rug_mat = material("rug", (0.16, 0.08, 0.04, 1.0), 0.95)
    cube("floor", (0, 0, 0.42), (2.5, 2.3, 0.08), floor_mat)
    cube("back_wall", (0, 1.8, 1.55), (2.5, 0.08, 1.2), wall_mat)
    cube("side_wall", (-2.25, 0, 1.55), (0.08, 1.8, 1.2), wall_mat)
    cube("rug", (0, 0.15, 0.515), (1.45, 0.9, 0.012), rug_mat)
    if spec["room"] == "tabletop":
        cube("table", (0, 0.1, 0.62), (1.35, 0.8, 0.12),
             material("table_wood", (0.20, 0.075, 0.025, 1.0), 0.42))
    else:
        cube("sofa", (0.75, 1.28, 0.82), (0.9, 0.35, 0.28),
             material("sofa_fabric", (0.10, 0.16, 0.24, 1.0), 0.9))

    target, meshes = import_target(asset_root / spec["asset"] / "model.gltf", spec["start"])
    target.keyframe_insert(data_path="location", frame=1)
    target.keyframe_insert(data_path="location", frame=300)
    target.location = spec["end"]
    target.keyframe_insert(data_path="location", frame=450)
    target.keyframe_insert(data_path="location", frame=600)
    create_hand(spec["start"], spec["end"])

    distractor_materials = [
        material("distractor_blue", (0.05, 0.2, 0.65, 1.0), 0.35),
        material("distractor_red", (0.65, 0.07, 0.04, 1.0), 0.5),
        material("distractor_green", (0.06, 0.42, 0.13, 1.0), 0.7),
    ]
    for idx, (x, y) in enumerate(((-0.9, 0.75), (0.75, 0.62), (1.2, -0.15))):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.11 + idx * 0.025,
                                             location=(x, y, 0.66))
        bpy.context.object.name = f"distractor_{idx}"
        bpy.context.object.data.materials.append(distractor_materials[idx])
        bpy.context.object.pass_index = 3 + idx

    bpy.ops.object.light_add(type="AREA", location=(-0.4, -0.6, 2.8))
    bpy.context.object.data.energy = 900
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 2.5
    bpy.ops.object.light_add(type="AREA", location=(1.5, 0.8, 1.8))
    bpy.context.object.data.energy = 350
    bpy.context.object.data.color = (1.0, 0.55, 0.32)
    bpy.context.object.data.size = 1.2

    bpy.ops.object.camera_add(location=(0.05, -1.55, 0.96))
    camera = bpy.context.object
    camera.name = "chest_camera_not_head_pose"
    camera.data.type = "PERSP"
    camera.data.sensor_width = 36
    camera.data.lens = 36 / (2 * math.tan(math.radians(140 / 2)))
    look_at(camera, (0, 0.2, 0.74))
    camera.keyframe_insert(data_path="location", frame=1)
    camera.location.x = -0.08
    camera.location.z = 1.01
    camera.keyframe_insert(data_path="location", frame=300)
    camera.location.x = 0.12
    camera.location.z = 0.93
    camera.keyframe_insert(data_path="location", frame=600)
    scene.camera = camera

    for obj in scene.objects:
        if obj.animation_data and obj.animation_data.action:
            for curve in obj.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "BEZIER"

    destination = output_root / episode_id
    destination.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(destination / "rgb_no_audio.mp4")
    started = time.monotonic()
    bpy.ops.render.render(animation=True)
    elapsed = time.monotonic() - started
    event = {
        "episode_id": episode_id,
        "duration_seconds": 20.0,
        "fps": 30,
        "resolution": [640, 360],
        "camera_mount": "chest-height egocentric; not head pose",
        "horizontal_fov_degrees": 140,
        "target": {"asset_id": spec["asset"], "nonce_label": spec["label"], "familiar_noun": spec["noun"]},
        "events": [
            {"start_seconds": 0.0, "end_seconds": 6.0, "action": "observe"},
            {"start_seconds": 6.0, "end_seconds": 10.0, "action": "touch"},
            {"start_seconds": 10.0, "end_seconds": 15.0, "action": spec["action"]},
            {"start_seconds": 15.0, "end_seconds": 20.0, "action": "observe_result"},
        ],
        "render_elapsed_seconds": elapsed,
        "render_engine": "BLENDER_EEVEE_NEXT",
    }
    (destination / "event_graph.json").write_text(json.dumps(event, indent=2) + "\n")
    scene.frame_set(330)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(destination / "representative.png")
    bpy.ops.render.render(write_still=True)
    print(json.dumps(event))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    render(args[0], Path(args[1]), Path(args[2]))
