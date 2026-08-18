"""Blender appearance replay driven exactly by exported MuJoCo transforms.

Run under Blender's bundled Python:
  blender -b --python blender_replay.py -- telemetry.npz body_names.json out mode
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def _arguments() -> tuple[Path, Path, Path, str]:
    values = sys.argv[sys.argv.index("--") + 1 :]
    return Path(values[0]), Path(values[1]), Path(values[2]), values[3]


def _material(name: str, rgba: tuple[float, float, float, float], roughness=0.5):
    material = bpy.data.materials.new(name)
    material.diffuse_color = rgba
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Roughness"].default_value = roughness
    return material


def _cube(name, location, scale, material, bevel=0.03):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    modifier = obj.modifiers.new("soft_edges", "BEVEL")
    modifier.width = bevel
    modifier.segments = 3
    obj.data.materials.append(material)
    return obj


def _uv(name, location, scale, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def _animate_transform(obj, poses: np.ndarray) -> None:
    for frame_index, pose in enumerate(poses, start=1):
        obj.location = pose[:3]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Quaternion(pose[3:7])
        obj.keyframe_insert("location", frame=frame_index)
        obj.keyframe_insert("rotation_quaternion", frame=frame_index)
    for curve in obj.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def _save_depth_png(depth: np.ndarray, path: Path) -> None:
    finite = np.isfinite(depth)
    visual = np.zeros_like(depth, dtype=np.float32)
    if finite.any():
        lo, hi = np.percentile(depth[finite], [2, 98])
        visual[finite] = 1.0 - np.clip((depth[finite] - lo) / max(hi - lo, 1e-6), 0, 1)
    height, width = visual.shape
    image = bpy.data.images.new(path.stem, width=width, height=height, alpha=False)
    rgba = np.repeat(visual[::-1, :, None], 4, axis=2)
    rgba[..., 3] = 1.0
    image.pixels = rgba.ravel()
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def main() -> None:
    telemetry_path, body_names_path, output_dir, mode = _arguments()
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = np.load(telemetry_path)
    body_names = json.loads(body_names_path.read_text(encoding="utf-8"))
    body_pose = telemetry["body_pose"]
    camera_pose = telemetry["camera_pose"]
    target_index = body_names.index("mimo_audition_ball")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = body_pose.shape[0]
    scene.world = bpy.data.worlds.new("replay_world")
    scene.world.color = (0.025, 0.035, 0.05)

    skin = _material("skin_proxy", (0.72, 0.42, 0.25, 1), 0.7)
    red = _material("ball", (0.7, 0.025, 0.02, 1), 0.28)
    blue = _material("cup", (0.03, 0.18, 0.75, 1), 0.3)
    wood = _material("wood", (0.22, 0.07, 0.025, 1), 0.75)
    wall = _material("wall", (0.38, 0.52, 0.62, 1), 0.85)
    rug = _material("rug", (0.68, 0.3, 0.09, 1), 0.9)
    fabric = _material("fabric", (0.08, 0.28, 0.24, 1), 0.95)
    yellow = _material("toy_yellow", (0.9, 0.55, 0.03, 1), 0.5)

    # License-auditable visual proxies: static, deterministic, and never used for physics.
    _cube("floor", (0.3, 0.72, 0.43), (1.8, 1.8, 0.03), wood)
    _cube("back_wall", (0.65, 1.25, 1.1), (1.8, 0.04, 0.7), wall)
    _cube("side_wall", (-0.55, 0.55, 1.1), (0.04, 1.2, 0.7), wall)
    _cube("workspace", (0.30, 0.70, 0.50), (0.34, 0.30, 0.025), rug)
    _cube("sofa_base", (0.80, 0.92, 0.58), (0.35, 0.16, 0.12), fabric, 0.07)
    _cube("sofa_back", (0.92, 1.05, 0.79), (0.35, 0.07, 0.25), fabric, 0.07)
    _cube("shelf", (-0.12, 1.10, 0.78), (0.24, 0.08, 0.35), wood)
    for z in (0.58, 0.78, 0.98):
        _cube(f"shelf_board_{z}", (-0.12, 1.02, z), (0.25, 0.12, 0.018), wood)
    for index, location in enumerate(((0.02, 1.0, 0.63), (-0.12, 1.0, 0.83), (0.0, 1.0, 1.03))):
        _cube(f"book_{index}", location, (0.035, 0.06, 0.09), yellow, 0.01)
    _uv("distractor_ball", (0.47, 0.88, 0.56), (0.035,) * 3, yellow)
    _cube("toy_block", (0.55, 0.75, 0.55), (0.035,) * 3, wall, 0.01)

    # Exact MIMo joint/body replay, rendered as a smooth five-finger skin proxy.
    replay_objects: dict[str, bpy.types.Object] = {}
    hand_names = [
        name
        for name in body_names
        if name == "mimo_right_hand"
        or any(
            token in name
            for token in ("right_ff", "right_mf", "right_rf", "right_lf", "right_th")
        )
    ]
    for name in hand_names:
        index = body_names.index(name)
        if name == "mimo_right_hand":
            scale = (0.042, 0.032, 0.018)
        elif name.endswith(("knuckle", "base", "metacarpal")):
            scale = (0.013, 0.011, 0.011)
        else:
            scale = (0.011, 0.009, 0.009)
        obj = _uv(f"visual_proxy_{name}", body_pose[0, index, :3], scale, skin)
        _animate_transform(obj, body_pose[:, index])
        replay_objects[name] = obj

    target_poses = body_pose[:, target_index]
    if mode == "lift":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48, radius=0.04, depth=0.12, location=target_poses[0, :3]
        )
        target = bpy.context.object
        target.data.materials.append(blue)
    else:
        target = _uv("replay_target", target_poses[0, :3], (0.05,) * 3, red)
    target.name = "replay_target"
    _animate_transform(target, target_poses)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "exact_mujoco_chest_camera"
    camera.data.type = "PERSP"
    camera.data.angle_y = math.radians(90)
    scene.camera = camera
    for frame_index, pose in enumerate(camera_pose, start=1):
        camera.location = pose[:3]
        matrix = Matrix(np.asarray(pose[3:12]).reshape(3, 3).tolist()).to_4x4()
        camera.rotation_mode = "QUATERNION"
        camera.rotation_quaternion = matrix.to_quaternion()
        camera.keyframe_insert("location", frame=frame_index)
        camera.keyframe_insert("rotation_quaternion", frame=frame_index)

    bpy.ops.object.light_add(type="AREA", location=(0.25, 0.55, 1.35))
    bpy.context.object.data.energy = 650
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 1.2
    bpy.ops.object.light_add(type="AREA", location=(0.9, 0.6, 0.9))
    bpy.context.object.data.energy = 350
    bpy.context.object.data.size = 0.8

    scene.view_layers["ViewLayer"].use_pass_z = True
    video_path = output_dir / f"blender_exact_replay_{mode}.mp4"
    scene.render.filepath = str(video_path)
    if not video_path.exists():
        bpy.ops.render.render(animation=True)

    keyframes = [46, 71, 106, 151, 211]
    original_format = scene.render.image_settings.file_format
    scene.render.image_settings.file_format = "PNG"
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True
    tree = scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    normalize_depth = tree.nodes.new("CompositorNodeNormalize")
    depth_output = tree.nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = str(output_dir)
    depth_output.format.file_format = "PNG"
    tree.links.new(render_layers.outputs["Depth"], normalize_depth.inputs[0])
    tree.links.new(normalize_depth.outputs[0], depth_output.inputs[0])
    segmentation_material = _material("segmentation_white", (1, 1, 1, 1), 1.0)
    segmentation_material.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (1, 1, 1, 1)
    segmentation_material.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.0
    for frame in keyframes:
        scene.frame_set(frame)
        depth_output.file_slots[0].path = f"replay_depth_{frame - 1:03d}_"
        scene.render.filepath = str(output_dir / f"replay_rgb_{frame - 1:03d}.png")
        bpy.ops.render.render(write_still=True)
        mesh_objects = [obj for obj in scene.objects if obj.type == "MESH"]
        visibility = {obj.name: obj.hide_render for obj in mesh_objects}
        for obj in mesh_objects:
            obj.hide_render = obj != target
        original_materials = list(target.data.materials)
        target.data.materials.clear()
        target.data.materials.append(segmentation_material)
        scene.render.filepath = str(
            output_dir / f"replay_segmentation_{frame - 1:03d}.png"
        )
        bpy.ops.render.render(write_still=True)
        target.data.materials.clear()
        for material in original_materials:
            target.data.materials.append(material)
        for obj in mesh_objects:
            obj.hide_render = visibility[obj.name]
    scene.render.image_settings.file_format = original_format

    # Transform equivalence is validated from Blender F-curves after keyframe creation.
    max_location_error = 0.0
    for name, obj in replay_objects.items():
        index = body_names.index(name)
        curves = {curve.data_path: curve for curve in obj.animation_data.action.fcurves}
        location_curves = [c for c in obj.animation_data.action.fcurves if c.data_path == "location"]
        for frame in (1, 71, 151, 240):
            reconstructed = np.array([curve.evaluate(frame) for curve in sorted(location_curves, key=lambda c: c.array_index)])
            max_location_error = max(
                max_location_error,
                float(np.max(np.abs(reconstructed - body_pose[frame - 1, index, :3]))),
            )
    receipt = {
        "source": "MuJoCo causal telemetry; Blender appearance replay only",
        "mode": mode,
        "frames": int(body_pose.shape[0]),
        "mimo_body_transform_sha256": hashlib.sha256(body_pose.tobytes()).hexdigest(),
        "camera_transform_sha256": hashlib.sha256(camera_pose.tobytes()).hexdigest(),
        "target_transform_sha256": hashlib.sha256(target_poses.tobytes()).hexdigest(),
        "maximum_replayed_body_location_error_m": max_location_error,
        "visual_proxy": "smooth articulated five-finger proxy driven by MIMo body transforms",
        "scene_proxy": "deterministic furnished-room visual proxies; no physics rerun",
        "video": video_path.name,
    }
    (output_dir / "blender_replay_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
