"""Render the continuous MPFB right hand/forearm from an EpisodeTrace.

Run inside Blender 4.2+::

  Blender mpfb_age_minimum_rigged.blend --background --python \
    mpfb_overlay_renderer.py -- --trace episode_trace.npz \
    --body-names body_names.json --output-dir /ignored/overlay

The renderer is deliberately appearance-only: it reads the causal trace and
never modifies target or physics state.  MuJoCo and Blender both use a
right-handed Z-up world, while quaternion input is MuJoCo's ``w, x, y, z``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix, Quaternion, Vector


MPFB_CHAINS = {
    "thumb": ("finger1-1.R", "finger1-2.R", "finger1-3.R"),
    "index": ("finger2-1.R", "finger2-2.R", "finger2-3.R"),
    "middle": ("finger3-1.R", "finger3-2.R", "finger3-3.R"),
    "ring": ("finger4-1.R", "finger4-2.R", "finger4-3.R"),
    "little": ("finger5-1.R", "finger5-2.R", "finger5-3.R"),
}
MIMO_BASES = {
    "thumb": "right_thbase",
    "index": "right_ffknuckle",
    "middle": "right_mfknuckle",
    "ring": "right_rfknuckle",
    "little": "right_lfknuckle",
}
MIMO_CHAINS = {
    "thumb": ("right_thbase", "right_thhub", "right_thdistal"),
    "index": ("right_ffknuckle", "right_ffmiddle", "right_ffdistal"),
    "middle": ("right_mfknuckle", "right_mfmiddle", "right_mfdistal"),
    "ring": ("right_rfknuckle", "right_rfmiddle", "right_rfdistal"),
    # MPFB has three phalanges; MIMo additionally exposes a metacarpal.
    "little": ("right_lfknuckle", "right_lfmiddle", "right_lfdistal"),
}
ALIGNMENT_RMS_GATE_M = 0.006
ALIGNMENT_MAX_GATE_M = 0.006
MIMO_DISTAL_SURFACE_OFFSET_M = {
    "thumb": 0.02201,
    "index": 0.01633,
    "middle": 0.01800,
    "ring": 0.01663,
    "little": 0.01477,
}
DISTAL_SKIN_THICKNESS_SCALE = 1.0
CONTACT_SKIN_EXTENSION_M = 0.005
CONTACT_SKIN_MAX_ORIGIN_DISTANCE_M = 0.030
PHASE_POSE = {
    "look": "neutral",
    "reorient": "neutral",
    "approach": "neutral",
    "reach_past_distractor": "open",
    "fingertip_contact": "contact",
    "grasp": "grasp",
    "lift": "grasp",
    "inspect_rotate": "grasp",
    "head_turn_maintain_contact": "grasp",
    "release": "open",
    "settle": "neutral",
}
FLEXION_DEGREES = {
    "neutral": (0.0, 0.0, 0.0),
    "open": (-8.0, -5.0, -3.0),
    "contact": (18.0, 24.0, 18.0),
    "grasp": (48.0, 62.0, 55.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _armature() -> bpy.types.Object:
    objects = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(objects) != 1:
        raise RuntimeError(f"expected one armature, found {[obj.name for obj in objects]}")
    return objects[0]


def _mesh_for_armature(armature: bpy.types.Object) -> bpy.types.Object:
    meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for modifier in obj.modifiers
        )
    ]
    if not meshes:
        raise RuntimeError("no weighted mesh attached to MPFB armature")
    return max(meshes, key=lambda obj: len(obj.data.vertices))


def _crop_to_forearm(mesh: bpy.types.Object, armature: bpy.types.Object) -> int:
    """Keep the weighted right arm plus a small topology transition band."""
    bones = armature.data.bones
    wanted_bones = {
        bone.name
        for bone in bones
        if bone.name == "wrist.R"
        or bone.name.startswith("finger") and bone.name.endswith(".R")
        or bone.name == "lowerarm02.R"
    }
    distal_groups = {
        group.index
        for group in mesh.vertex_groups
        if group.name in wanted_bones and group.name != "lowerarm02.R"
    }
    lowerarm_group = mesh.vertex_groups["lowerarm02.R"].index
    keep = {
        vertex.index
        for vertex in mesh.data.vertices
        if any(
            (
                assignment.group in distal_groups
                and assignment.weight > 1e-5
            )
            or (
                assignment.group == lowerarm_group
                and assignment.weight >= 0.55
            )
            for assignment in vertex.groups
        )
    }
    neighbors: dict[int, set[int]] = {
        vertex.index: set() for vertex in mesh.data.vertices
    }
    for edge in mesh.data.edges:
        left, right = edge.vertices
        neighbors[left].add(right)
        neighbors[right].add(left)
    for _ in range(1):
        keep |= {neighbor for index in keep for neighbor in neighbors[index]}
    remove_indices = [
        vertex.index
        for vertex in mesh.data.vertices
        if vertex.index not in keep
    ]
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for index in remove_indices:
        mesh.data.vertices[index].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")
    return len(mesh.data.vertices)


def _rest_points(armature: bpy.types.Object) -> dict[str, Vector]:
    bones = armature.data.bones
    points = {"wrist": bones["wrist.R"].head_local.copy()}
    for finger, chain in MPFB_CHAINS.items():
        points[finger] = bones[chain[0]].head_local.copy()
    return points


def _world_to_source(
    point: np.ndarray,
    rotation_row: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> Vector:
    return Vector(((point - translation) @ rotation_row.T / scale).tolist())


def _calibrate_finger_rest_pose(
    armature: bpy.types.Object,
    reference_body_pose: np.ndarray,
    body_index: dict[str, int],
    rotation_row: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> dict[str, Any]:
    """Place MPFB finger rest bones on neutral MIMo chain origins.

    The global wrist/palm fit establishes a proper world transform.  Each
    finger chain is then calibrated independently in armature-local space.
    This preserves the explicit body correspondence instead of asking one
    uniform hand-shape fit to absorb different finger splay and proportions.
    """
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    wrist = armature.data.edit_bones["wrist.R"]
    wrist_target = _world_to_source(
        reference_body_pose[body_index["right_hand"], :3],
        rotation_row,
        scale,
        translation,
    )
    wrist_tail_offset = wrist.tail - wrist.head
    wrist.head = wrist_target
    wrist.tail = wrist_target + wrist_tail_offset
    chain_receipts: dict[str, Any] = {}
    for finger, mpfb_chain in MPFB_CHAINS.items():
        mimo_chain = MIMO_CHAINS[finger]
        origins_world = [
            reference_body_pose[body_index[name], :3] for name in mimo_chain
        ]
        origins_local = [
            _world_to_source(point, rotation_row, scale, translation)
            for point in origins_world
        ]
        if len(origins_local) < 2:
            raise RuntimeError(f"insufficient MIMo chain points for {finger}")
        terminal_vector = origins_local[-1] - origins_local[-2]
        # MIMo exposes the distal phalanx origin but no terminal body origin.
        # Continue the final measured segment by one segment length so the
        # visible fingertip reaches the physical distal capsule surface.
        terminal_length = max(terminal_vector.length, 0.004 / scale)
        terminal_direction = terminal_vector.normalized()
        tails = origins_local[1:] + [
            origins_local[-1] + terminal_direction * terminal_length
        ]
        lengths = []
        for bone_name, head, tail in zip(mpfb_chain, origins_local, tails):
            bone = armature.data.edit_bones[bone_name]
            bone.use_connect = False
            # The MakeHuman hierarchy includes intermediate palm controls that
            # do not exist in the MIMo chain.  Detach the deform phalanges so
            # each authoritative segment can receive an absolute pose matrix.
            bone.parent = None
            bone.head = head
            bone.tail = tail
            bone.roll = 0.0
            lengths.append(float((tail - head).length * scale))
        chain_receipts[finger] = {
            "mpfb_bones": list(mpfb_chain),
            "mimo_bodies": list(mimo_chain),
            "segment_lengths_m": lengths,
        }
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    return chain_receipts


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Proper-rotation Umeyama fit for row-vector points."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = float(singular.sum() / np.square(source_centered).sum())
    translation = target_mean - source_mean @ rotation * scale
    mapped = source @ rotation * scale + translation
    return rotation, scale, translation, mapped


def _object_matrix(rotation_row: np.ndarray, scale: float, translation: np.ndarray) -> Matrix:
    matrix = Matrix.Identity(4)
    # Blender columns transform column vectors; transpose the row-vector fit.
    for row in range(3):
        for column in range(3):
            matrix[row][column] = float(rotation_row.T[row, column] * scale)
        matrix[row][3] = float(translation[row])
    return matrix


def _pose_fingers(armature: bpy.types.Object, pose_name: str) -> None:
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "XYZ"
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)
    flexion = FLEXION_DEGREES[pose_name]
    for finger, chain in MPFB_CHAINS.items():
        multiplier = 0.75 if finger == "thumb" else 1.0
        for index, bone_name in enumerate(chain):
            armature.pose.bones[bone_name].rotation_euler.x = math.radians(
                flexion[index] * multiplier
            )


def _pose_mimo_chains(
    armature: bpy.types.Object,
    body_pose: np.ndarray,
    body_index: dict[str, int],
    contact_position: np.ndarray,
) -> dict[str, Any]:
    """Pose the calibrated MPFB bones on the authoritative MIMo chain.

    Edit-bone calibration establishes one neutral bind pose.  Episode frames
    then use pose matrices, which deforms the weighted MPFB skin while leaving
    its thickness and bind topology intact.
    """
    for pose_bone in armature.pose.bones:
        pose_bone.matrix_basis = Matrix.Identity(4)
    world_to_armature = armature.matrix_world.inverted()
    selected_contact_finger = None
    contact_distance = None
    if np.isfinite(contact_position).all():
        distal_distances = {
            finger: float(
                np.linalg.norm(
                    body_pose[body_index[MIMO_CHAINS[finger][-1]], :3]
                    - contact_position
                )
            )
            for finger in MIMO_CHAINS
        }
        candidate = min(distal_distances, key=distal_distances.get)
        if distal_distances[candidate] <= CONTACT_SKIN_MAX_ORIGIN_DISTANCE_M:
            selected_contact_finger = candidate
            contact_distance = distal_distances[candidate]
    for finger, mpfb_chain in MPFB_CHAINS.items():
        mimo_chain = MIMO_CHAINS[finger]
        origins = [
            world_to_armature
            @ Vector(body_pose[body_index[name], :3].tolist())
            for name in mimo_chain
        ]
        distal_pose = body_pose[body_index[mimo_chain[-1]]]
        distal_rotation = Quaternion(tuple(float(value) for value in distal_pose[3:7]))
        distal_surface_world = Vector(distal_pose[:3].tolist()) + (
            distal_rotation
            @ Vector((0.0, 0.0, MIMO_DISTAL_SURFACE_OFFSET_M[finger]))
        )
        if finger == selected_contact_finger:
            origin_world = Vector(distal_pose[:3].tolist())
            contact_world = Vector(contact_position.tolist())
            contact_direction = (contact_world - origin_world).normalized()
            distal_surface_world = contact_world + (
                contact_direction * CONTACT_SKIN_EXTENSION_M
            )
        tails = origins[1:] + [world_to_armature @ distal_surface_world]
        for bone_name, head, tail in zip(mpfb_chain, origins, tails):
            direction = tail - head
            length = max(direction.length, 1e-8)
            rest_bone = armature.data.bones[bone_name]
            rest_direction = rest_bone.vector.normalized()
            delta_rotation = rest_direction.rotation_difference(
                direction.normalized()
            ).to_matrix()
            desired_rotation = (
                delta_rotation @ rest_bone.matrix_local.to_3x3().normalized()
            ).to_4x4()
            rest_length = max(rest_bone.length, 1e-8)
            transverse_scale = (
                DISTAL_SKIN_THICKNESS_SCALE
                if bone_name == mpfb_chain[-1]
                else 1.0
            )
            scale = Matrix.Diagonal(
                (
                    transverse_scale,
                    length / rest_length,
                    transverse_scale,
                    1.0,
                )
            )
            armature.pose.bones[bone_name].matrix = (
                Matrix.Translation(head) @ desired_rotation @ scale
            )
    bpy.context.view_layer.update()
    return {
        "active": selected_contact_finger is not None,
        "finger": selected_contact_finger,
        "distal_origin_to_contact_m": contact_distance,
        "extension_beyond_contact_m": (
            CONTACT_SKIN_EXTENSION_M
            if selected_contact_finger is not None
            else 0.0
        ),
    }


def _camera_from_trace(pose: np.ndarray) -> tuple[Vector, Matrix]:
    location = Vector(pose[:3])
    rotation = Matrix(np.asarray(pose[3:12], dtype=float).reshape(3, 3).tolist())
    return location, rotation


def _configure_scene(
    width: int, height: int, *, render_samples: int
) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_layers[0].use_pass_z = True
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -1.25
    scene.eevee.taa_render_samples = render_samples
    scene.world.color = (0.02, 0.02, 0.02)
    for obj in list(bpy.data.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    camera_data = bpy.data.cameras.new("EpisodeTraceCamera")
    camera = bpy.data.objects.new("EpisodeTraceCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.data.type = "PERSP"
    camera.data.angle_y = math.radians(90.0)
    camera.data.clip_start = 0.005
    scene.camera = camera
    lamp_data = bpy.data.lights.new("OverlayKey", "AREA")
    lamp_data.energy = 80.0
    lamp_data.size = 0.4
    lamp = bpy.data.objects.new("OverlayKey", lamp_data)
    bpy.context.collection.objects.link(lamp)
    return camera


def _apply_skin_material(mesh: bpy.types.Object) -> bpy.types.Material:
    """Use an explicit local material so missing MPFB textures cannot turn gray."""
    material = bpy.data.materials.new("MPFBProvisionalSkinMaterial")
    material.diffuse_color = (0.46, 0.24, 0.15, 1.0)
    material.roughness = 0.68
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.46, 0.24, 0.15, 1.0)
        principled.inputs["Roughness"].default_value = 0.68
        if "Subsurface Weight" in principled.inputs:
            principled.inputs["Subsurface Weight"].default_value = 0.035
    mesh.data.materials.clear()
    mesh.data.materials.append(material)
    for polygon in mesh.data.polygons:
        polygon.material_index = 0
    return material


def _sleeve_cuff(armature: bpy.types.Object) -> bpy.types.Object:
    lower = armature.data.bones["lowerarm02.R"]
    wrist = armature.data.bones["wrist.R"].head_local
    axis = lower.head_local - wrist
    center = wrist + axis * 0.68
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.022, depth=0.032)
    cuff = bpy.context.object
    cuff.name = "LowerForearmSleeveCuff"
    cuff.rotation_mode = "QUATERNION"
    cuff.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(
        axis.normalized()
    )
    cuff.location = center
    cuff.parent = armature
    cuff.matrix_parent_inverse = Matrix.Identity(4)
    material = bpy.data.materials.new("SleeveCuffMaterial")
    material.diffuse_color = (0.18, 0.28, 0.48, 1.0)
    material.roughness = 0.72
    cuff.data.materials.append(material)
    return cuff


def _target_visual(spec: dict[str, Any]) -> bpy.types.Object:
    target = spec["scene_family"]["target"]
    geometry = target["geometry"]
    rgba = tuple(float(value) for value in target["rgba"])
    scale = np.asarray(target["scale_m"], dtype=float)
    material = bpy.data.materials.new("ResolvedTargetMaterial")
    material.diffuse_color = rgba
    root = bpy.data.objects.new("ResolvedTarget", None)
    bpy.context.collection.objects.link(root)
    if geometry == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
        visual = bpy.context.object
        visual.name = "ResolvedTargetSphere"
        visual.scale = Vector((scale / 2.0).tolist())
        visual.data.materials.append(material)
        visual.parent = root
    elif geometry in {
        "cup",
        "mug",
        "cylinder_with_handle",
        "cylinder_with_three_capsule_handle",
    }:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=float(scale[0] / 2.0),
            depth=float(scale[2]),
        )
        visual = bpy.context.object
        visual.name = "ResolvedTargetCupBody"
        visual.data.materials.append(material)
        visual.parent = root
        handle_radius = 0.007
        capsule_specs = (
            ((0.0, -0.045, 0.025), (handle_radius, 0.020, handle_radius)),
            ((0.0, -0.065, 0.0), (handle_radius, handle_radius, 0.032)),
            ((0.0, -0.045, -0.025), (handle_radius, 0.020, handle_radius)),
        )
        for index, (location, dimensions) in enumerate(capsule_specs):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
            capsule = bpy.context.object
            capsule.name = f"ResolvedTargetHandleCapsule{index + 1}"
            capsule.location = Vector(tuple(float(x) for x in location))
            capsule.scale = Vector(tuple(float(x) for x in dimensions))
            capsule.data.materials.append(material)
            capsule.parent = root
    else:
        raise ValueError(f"unsupported resolved target geometry for overlay: {geometry}")
    return root


def _set_render_hidden(root: bpy.types.Object, hidden: bool) -> None:
    root.hide_render = hidden
    for child in root.children_recursive:
        child.hide_render = hidden


def _distractor_visual() -> bpy.types.Object:
    material = bpy.data.materials.new("ReachDistractorMaterial")
    material.diffuse_color = (0.08, 0.28, 0.72, 1.0)
    material.roughness = 0.62
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.035)
    visual = bpy.context.object
    visual.name = "ReachDistractor"
    visual.data.materials.append(material)
    return visual


def _set_pose(object_: bpy.types.Object, pose: np.ndarray) -> None:
    object_.location = Vector(pose[:3])
    object_.rotation_mode = "QUATERNION"
    object_.rotation_quaternion = Quaternion(
        (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
    )


def _save_depth(output: Path, width: int, height: int) -> dict[str, Any]:
    """Save the Render Result, including its metric Z pass, as multilayer EXR."""
    del width, height
    scene = bpy.context.scene
    prior_format = scene.render.image_settings.file_format
    prior_mode = scene.render.image_settings.color_mode
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_mode = "RGBA"
    bpy.data.images["Render Result"].save_render(str(output), scene=scene)
    scene.render.image_settings.file_format = prior_format
    scene.render.image_settings.color_mode = prior_mode
    return {
        "file": output.name,
        "sha256": _sha256(output),
        "format": "OpenEXR Multilayer",
        "pass": "ViewLayer.Depth.Z",
        "units": "metre",
    }


def _render_depth_array(work_path: Path) -> np.ndarray:
    """Return the current metric Z pass, using one overwritten ignored EXR."""
    scene = bpy.context.scene
    prior_format = scene.render.image_settings.file_format
    prior_mode = scene.render.image_settings.color_mode
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_mode = "RGBA"
    bpy.data.images["Render Result"].save_render(str(work_path), scene=scene)
    scene.render.image_settings.file_format = prior_format
    scene.render.image_settings.color_mode = prior_mode
    image_input = oiio.ImageInput.open(str(work_path))
    if image_input is None:
        raise RuntimeError(f"OpenImageIO could not open {work_path}")
    try:
        spec = image_input.spec()
        channel_names = list(spec.channelnames)
        if "ViewLayer.Depth.Z" not in channel_names:
            raise RuntimeError(f"missing Depth.Z channel in {channel_names}")
        pixels = np.asarray(image_input.read_image(), dtype=np.float32)
        return pixels[..., channel_names.index("ViewLayer.Depth.Z")].copy()
    finally:
        image_input.close()


def _inspection_indices(
    trace: dict[str, np.ndarray], *, all_frames: bool, fps: int
) -> list[int]:
    if all_frames:
        truth_hz = int(round(1.0 / np.median(np.diff(trace["time_s"]))))
        if truth_hz % fps:
            raise ValueError("trace truth rate must be divisible by requested fps")
        return list(range(0, len(trace["time_s"]), truth_hz // fps))
    phases = trace["phase"].astype(str)
    result = [0]
    near_miss = _near_miss_closest_sample(trace)
    for pose_name in ("open", "contact", "grasp"):
        if pose_name == "open":
            result.append(near_miss["nearest_frame_index"])
            continue
        matches = [
            index
            for index, phase in enumerate(phases)
            if PHASE_POSE.get(phase) == pose_name
        ]
        if matches:
            result.append(matches[len(matches) // 2])
    result.append(len(phases) - 1)
    return list(dict.fromkeys(result))


def _near_miss_closest_sample(
    trace: dict[str, np.ndarray],
) -> dict[str, Any]:
    frame_mask = trace["phase"].astype(str) == "reach_past_distractor"
    frame_indices = np.flatnonzero(
        frame_mask & np.isfinite(trace["near_miss_clearance_m"])
    )
    if not frame_indices.size:
        raise ValueError("trace has no sampled near-miss clearance")
    local = int(np.argmin(trace["near_miss_clearance_m"][frame_indices]))
    nearest_frame = int(frame_indices[local])
    time_s = float(trace["time_s"][nearest_frame])
    return {
        "time_s": time_s,
        "signed_separation_m": float(
            trace["near_miss_clearance_m"][nearest_frame]
        ),
        "nearest_frame_index": nearest_frame,
        "nearest_frame_time_s": time_s,
    }


def _render_alpha_mask(
    output: Path,
    visible_meshes: set[bpy.types.Object],
    width: int,
    height: int,
) -> np.ndarray:
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    prior = {obj: obj.hide_render for obj in mesh_objects}
    for obj in mesh_objects:
        obj.hide_render = obj not in visible_meshes
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    saved = bpy.data.images.load(str(output), check_existing=False)
    pixels = np.asarray(saved.pixels[:], dtype=np.float32).reshape(
        height, width, 4
    )
    bpy.data.images.remove(saved)
    for obj, hidden in prior.items():
        obj.hide_render = hidden
    # Blender image-buffer rows are bottom-up; saved PNG/projected pixels use
    # the conventional top-left image origin.
    return np.flipud(pixels[:, :, 3] > 0.05)


def _mask_gap(
    hand_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    metres_per_pixel: float,
    threshold_pixels: float,
) -> dict[str, Any]:
    hand_yx = np.argwhere(hand_mask)
    target_yx = np.argwhere(target_mask)
    if not hand_yx.size or not target_yx.size:
        return {
            "passed": False,
            "reason": "hand_or_target_mask_empty",
            "gap_pixels": None,
            "gap_m": None,
        }
    if np.any(hand_mask & target_mask):
        gap_pixels = 0.0
    else:
        minimum_squared = math.inf
        for start in range(0, len(hand_yx), 512):
            delta = (
                hand_yx[start : start + 512, None, :]
                - target_yx[None, :, :]
            )
            minimum_squared = min(
                minimum_squared,
                float(np.min(np.sum(delta * delta, axis=2))),
            )
        gap_pixels = math.sqrt(minimum_squared)
    gap_m = gap_pixels * metres_per_pixel
    return {
        "gap_pixels": gap_pixels,
        "metres_per_pixel": metres_per_pixel,
        "approximate_planar_gap_m": gap_m,
        "screen_space_nonoverlap_threshold_pixels": threshold_pixels,
        "passed": bool(gap_pixels >= threshold_pixels),
    }


def _project_world_point(
    point: np.ndarray,
    camera: bpy.types.Object,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    camera_point = camera.matrix_world.inverted() @ Vector(point.tolist())
    depth = -float(camera_point.z)
    if depth <= 0.0:
        return np.asarray([np.nan, np.nan])
    focal = height / (2.0 * math.tan(math.radians(90.0) / 2.0))
    return np.asarray(
        [
            width / 2.0 + focal * float(camera_point.x) / depth,
            height / 2.0 - focal * float(camera_point.y) / depth,
        ]
    )


def _nearest_mask_distance(pixel_xy: np.ndarray, mask: np.ndarray) -> float:
    if not np.isfinite(pixel_xy).all() or not np.any(mask):
        return float("inf")
    coordinates_yx = np.argwhere(mask)
    differences = coordinates_yx[:, ::-1] - pixel_xy
    return float(np.sqrt(np.min(np.sum(differences * differences, axis=1))))


def run(
    trace_path: Path,
    body_names_path: Path,
    spec_path: Path,
    output_dir: Path,
    *,
    width: int,
    height: int,
    all_frames: bool,
    render_samples: int,
    fps: int,
) -> dict[str, Any]:
    trace_archive = np.load(trace_path, allow_pickle=False)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    required = {
        "body_pose",
        "camera_pose",
        "target_pose",
        "reach_distractor_pose",
        "touch_contact_position",
        "phase",
        "time_s",
    }
    missing = sorted(required - set(trace_archive.files))
    if missing:
        raise ValueError(f"EpisodeTrace missing required streams: {missing}")
    trace = {key: trace_archive[key] for key in trace_archive.files}
    body_names = json.loads(body_names_path.read_text(encoding="utf-8"))
    if len(body_names) != trace["body_pose"].shape[1]:
        raise ValueError("body_names length does not match body_pose")
    body_index = {
        name.removeprefix("kernel_"): index for index, name in enumerate(body_names)
    }
    needed = {
        "right_hand",
        *(name for chain in MIMO_CHAINS.values() for name in chain),
    }
    absent = sorted(needed - set(body_index))
    if absent:
        raise ValueError(f"EpisodeTrace body mapping missing {absent}")

    output_dir.mkdir(parents=True, exist_ok=True)
    armature = _armature()
    mesh = _mesh_for_armature(armature)
    for modifier in mesh.modifiers:
        if modifier.type == "MASK":
            modifier.show_render = False
    for obj in bpy.data.objects:
        if obj not in {armature, mesh}:
            obj.hide_render = True
    retained_vertices = _crop_to_forearm(mesh, armature)
    skin_material = _apply_skin_material(mesh)
    camera = _configure_scene(
        width, height, render_samples=render_samples
    )
    cuff = _sleeve_cuff(armature)
    target_visual = _target_visual(spec)
    distractor_visual = _distractor_visual()
    if all_frames:
        # The MuJoCo background already contains the authoritative target and
        # distractor.  The all-frame overlay replaces only the visible skin.
        _set_render_hidden(target_visual, True)
        distractor_visual.hide_render = True
    rest = _rest_points(armature)
    source = np.asarray(
        [list(rest["wrist"])] + [list(rest[finger]) for finger in MIMO_BASES],
        dtype=np.float64,
    )
    reference_body_pose = trace["body_pose"][0]
    reference_target = np.asarray(
        [reference_body_pose[body_index["right_hand"], :3]]
        + [
            reference_body_pose[body_index[mimo_name], :3]
            for mimo_name in MIMO_BASES.values()
        ],
        dtype=np.float64,
    )
    initial_rotation, initial_scale, initial_translation, initial_mapped = (
        _similarity(source, reference_target)
    )
    initial_errors = np.linalg.norm(initial_mapped - reference_target, axis=1)
    chain_calibration = _calibrate_finger_rest_pose(
        armature,
        reference_body_pose,
        body_index,
        initial_rotation,
        initial_scale,
        initial_translation,
    )
    rest = _rest_points(armature)
    source = np.asarray(
        [list(rest["wrist"])] + [list(rest[finger]) for finger in MIMO_BASES],
        dtype=np.float64,
    )
    alignment_labels = ["right_hand"] + [
        name for chain in MIMO_CHAINS.values() for name in chain
    ]
    near_miss_closest = _near_miss_closest_sample(trace)
    rendered_near_miss_gap = None
    projected_contact_records = []
    frame_records = []
    render_indices = _inspection_indices(trace, all_frames=all_frames, fps=fps)
    depth_memmap = None
    depth_work_path = output_dir / "overlay_depth_work.exr"
    if all_frames:
        depth_memmap = np.lib.format.open_memmap(
            output_dir / "overlay_depth_m.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(render_indices), height, width),
        )
    for output_index, frame_index in enumerate(render_indices):
        body_pose = trace["body_pose"][frame_index]
        target = np.asarray(
            [body_pose[body_index["right_hand"], :3]]
            + [
                body_pose[body_index[mimo_name], :3]
                for mimo_name in MIMO_BASES.values()
            ],
            dtype=np.float64,
        )
        rotation, scale, translation, mapped = _similarity(source, target)
        alignment_target = np.asarray(
            [
                body_pose[body_index[name], :3]
                for name in alignment_labels
            ],
            dtype=np.float64,
        )
        armature.matrix_world = _object_matrix(rotation, scale, translation)
        pose_name = PHASE_POSE.get(str(trace["phase"][frame_index]), "neutral")
        skin_contact_calibration = _pose_mimo_chains(
            armature,
            body_pose,
            body_index,
            trace["touch_contact_position"][frame_index],
        )
        wrist_world = armature.matrix_world @ rest["wrist"]
        chain_heads_world = [
            armature.matrix_world @ armature.pose.bones[bone_name].head
            for chain in MPFB_CHAINS.values()
            for bone_name in chain
        ]
        alignment_mapped = np.asarray(
            [list(wrist_world)] + [list(point) for point in chain_heads_world],
            dtype=np.float64,
        )
        mapped = np.asarray(
            [list(wrist_world)]
            + [
                list(
                    armature.matrix_world
                    @ armature.pose.bones[MPFB_CHAINS[finger][0]].head
                )
                for finger in MIMO_BASES
            ],
            dtype=np.float64,
        )
        _set_pose(target_visual, trace["target_pose"][frame_index])
        distractor_visual.location = Vector(
            trace["reach_distractor_pose"][frame_index, :3]
        )
        camera_location, camera_rotation = _camera_from_trace(
            trace["camera_pose"][frame_index]
        )
        camera.location = camera_location
        # MuJoCo camera matrix columns are right, up, backward.  Blender's local
        # camera axes are right, up, backward too.
        camera.matrix_world = Matrix.Translation(camera_location) @ camera_rotation.to_4x4()
        camera_local_landmarks = np.asarray(
            [
                list(
                    camera.matrix_world.inverted()
                    @ Vector(point.tolist())
                )
                for point in mapped
            ]
        )
        key = bpy.data.objects["OverlayKey"]
        key.location = camera_location + camera_rotation @ Vector((0.3, 0.2, -0.2))
        key.rotation_euler = (
            Vector(trace["target_pose"][frame_index, :3]) - key.location
        ).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        output = output_dir / (
            f"overlay_{output_index:04d}.png"
            if all_frames
            else f"overlay_{frame_index:04d}_{pose_name}.png"
        )
        bpy.context.scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        depth_record = None
        if all_frames:
            depth_memmap[output_index] = _render_depth_array(depth_work_path)
        else:
            depth_record = _save_depth(
                output_dir / f"overlay_{frame_index:04d}_{pose_name}_depth_z.exr",
                width,
                height,
            )
        mask_record = None
        if (
            not all_frames
            and frame_index == near_miss_closest["nearest_frame_index"]
        ):
            hand_mask_path = (
                output_dir / f"overlay_{frame_index:04d}_hand_mask.png"
            )
            distractor_mask_path = (
                output_dir / f"overlay_{frame_index:04d}_distractor_mask.png"
            )
            hand_mask = _render_alpha_mask(
                hand_mask_path, {mesh, cuff}, width, height
            )
            distractor_mask = _render_alpha_mask(
                distractor_mask_path, {distractor_visual}, width, height
            )
            camera_inverse = camera.matrix_world.inverted()
            hand_depth = -float(
                (camera_inverse @ Vector(body_pose[body_index["right_hand"], :3]))[
                    2
                ]
            )
            distractor_depth = -float(
                (
                    camera_inverse
                    @ Vector(trace["reach_distractor_pose"][frame_index, :3])
                )[2]
            )
            metres_per_pixel = (
                2.0
                * min(hand_depth, distractor_depth)
                * math.tan(math.radians(90.0) / 2.0)
                / height
            )
            mask_record = _mask_gap(
                hand_mask,
                distractor_mask,
                metres_per_pixel=metres_per_pixel,
                # The frozen 0.008 m near-miss threshold is a 3-D geom-distance
                # gate and is enforced by physics QA.  Its screen-space visual
                # counterpart is non-overlap, not a perspective-scaled 3-D
                # distance.  Require at least one full pixel of separation.
                threshold_pixels=1.0,
            )
            mask_record.update(
                {
                    "hand_mask": hand_mask_path.name,
                    "hand_mask_sha256": _sha256(hand_mask_path),
                    "distractor_mask": distractor_mask_path.name,
                    "distractor_mask_sha256": _sha256(distractor_mask_path),
                    "hand_camera_depth_m": hand_depth,
                    "distractor_camera_depth_m": distractor_depth,
                    "camera_vertical_fov_deg": 90.0,
                }
            )
            rendered_near_miss_gap = mask_record
        contact_record = None
        contact_position = trace["touch_contact_position"][frame_index]
        if not all_frames and np.isfinite(contact_position).all():
            hand_mask_path = output_dir / f"overlay_{frame_index:04d}_contact_hand_mask.png"
            target_mask_path = output_dir / f"overlay_{frame_index:04d}_contact_target_mask.png"
            hand_mask = _render_alpha_mask(
                hand_mask_path, {mesh, cuff}, width, height
            )
            target_meshes = {
                child
                for child in target_visual.children_recursive
                if child.type == "MESH"
            }
            target_mask = _render_alpha_mask(
                target_mask_path, target_meshes, width, height
            )
            projected = _project_world_point(
                contact_position, camera, width=width, height=height
            )
            target_distance = _nearest_mask_distance(projected, target_mask)
            hand_distance = _nearest_mask_distance(projected, hand_mask)
            contact_record = {
                "frame_index": frame_index,
                "time_s": float(trace["time_s"][frame_index]),
                "projected_pixel_xy": projected.tolist(),
                "target_distance_px": target_distance,
                "hand_distance_px": hand_distance,
                "visible_union_distance_px": min(
                    target_distance, hand_distance
                ),
                "hand_mask": hand_mask_path.name,
                "target_mask": target_mask_path.name,
            }
            projected_contact_records.append(contact_record)
        errors = np.linalg.norm(alignment_mapped - alignment_target, axis=1)
        frame_records.append(
            {
                "frame_index": frame_index,
                "time_s": float(trace["time_s"][frame_index]),
                "phase": str(trace["phase"][frame_index]),
                "pose": pose_name,
                "uniform_scale_m_per_blender_unit": scale,
                "landmark_rms_error_m": float(np.sqrt(np.mean(errors**2))),
                "landmark_max_error_m": float(errors.max()),
                "alignment_landmarks": alignment_labels,
                "alignment_errors_m": {
                    label: float(error)
                    for label, error in zip(alignment_labels, errors)
                },
                "camera_local_landmarks_m": camera_local_landmarks.tolist(),
                "landmarks_in_vertical_90deg_frustum": int(
                    np.sum(
                        (camera_local_landmarks[:, 2] < 0.0)
                        & (
                            np.abs(camera_local_landmarks[:, 1])
                            <= -camera_local_landmarks[:, 2]
                        )
                        & (
                            np.abs(camera_local_landmarks[:, 0])
                            <= -camera_local_landmarks[:, 2] * width / height
                        )
                    )
                ),
                "frame": output.name,
                "frame_sha256": _sha256(output),
                "depth": depth_record,
                "rendered_skin_distractor_gap": mask_record,
                "projected_contact": contact_record,
                "contact_skin_calibration": skin_contact_calibration,
            }
        )
    if depth_memmap is not None:
        depth_memmap.flush()
        del depth_memmap
        if depth_work_path.exists():
            depth_work_path.unlink()
    receipt = {
        "schema_version": "mpfb_episode_trace_overlay.v1",
        "status": "pending_gate_evaluation",
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "body_names_sha256": _sha256(body_names_path),
        "resolved_spec": str(spec_path),
        "resolved_spec_sha256": _sha256(spec_path),
        "blend": bpy.data.filepath,
        "blend_sha256": _sha256(Path(bpy.data.filepath)),
        "blender_version": bpy.app.version_string,
        "render_mode": "all_frames" if all_frames else "inspection_frames",
        "fps": fps,
        "render_samples": render_samples,
        "near_miss_closest_sample": near_miss_closest,
        "rendered_near_miss_gap": rendered_near_miss_gap,
        "projected_contact_records": projected_contact_records,
        "coordinate_contract": "MuJoCo and Blender right-handed Z-up; camera right/up/backward matrix",
        "mapping": {
            "global_fit": ["right_hand", *MIMO_BASES.values()],
            "neutral_reference_frame": 0,
            "per_chain_rest_calibration": chain_calibration,
            "frame_pose_source": (
                "actual MIMo per-frame chain body origins; semantic pose labels "
                "do not override physics alignment"
            ),
            "phase_pose": PHASE_POSE,
            "mimo_distal_surface_offset_m": MIMO_DISTAL_SURFACE_OFFSET_M,
            "distal_skin_thickness_scale": DISTAL_SKIN_THICKNESS_SCALE,
            "contact_skin_calibration": {
                "source": "simulator-truth contact position",
                "maximum_distal_origin_distance_m": CONTACT_SKIN_MAX_ORIGIN_DISTANCE_M,
                "extension_beyond_contact_m": CONTACT_SKIN_EXTENSION_M,
                "authority": "appearance deformation only; no physics or trace mutation",
            },
            "finger_chains": MPFB_CHAINS,
            "flexion_degrees": FLEXION_DEGREES,
        },
        "continuous_right_hand_forearm": {
            "mesh": mesh.name,
            "retained_vertices": retained_vertices,
            "five_finger_chains": len(MPFB_CHAINS),
            "material": skin_material.name,
            "claim_boundary": (
                "provisional MPFB appearance proxy; not age-matched, infant-"
                "calibrated, human-validated, or infant-trained"
            ),
        },
        "resolved_target_visual": {
            "geometry": spec["scene_family"]["target"]["geometry"],
            "rgba": spec["scene_family"]["target"]["rgba"],
            "scale_m": spec["scene_family"]["target"]["scale_m"],
            "cup_handle": "three visible rounded capsule elements",
        },
        "frames": frame_records,
        "maximum_landmark_error_m": max(
            record["landmark_max_error_m"] for record in frame_records
        ),
        "alignment_gate": {
            "rms_threshold_m": ALIGNMENT_RMS_GATE_M,
            "maximum_threshold_m": ALIGNMENT_MAX_GATE_M,
            "pre_chain_calibration_reference_rms_m": float(
                np.sqrt(np.mean(initial_errors**2))
            ),
            "pre_chain_calibration_reference_maximum_m": float(
                initial_errors.max()
            ),
            "observed_maximum_frame_rms_m": max(
                record["landmark_rms_error_m"] for record in frame_records
            ),
            "observed_maximum_landmark_error_m": max(
                record["landmark_max_error_m"] for record in frame_records
            ),
        },
    }
    receipt["alignment_gate"]["passed"] = bool(
        receipt["alignment_gate"]["observed_maximum_frame_rms_m"]
        <= ALIGNMENT_RMS_GATE_M
        and receipt["alignment_gate"]["observed_maximum_landmark_error_m"]
        <= ALIGNMENT_MAX_GATE_M
    )
    near_miss_gate_passed = bool(
        all_frames
        or (
            rendered_near_miss_gap is not None
            and rendered_near_miss_gap["passed"]
        )
    )
    contact_gate_passed = bool(
        all_frames
        or (
            projected_contact_records
            and max(
                record["visible_union_distance_px"]
                for record in projected_contact_records
            )
            <= float(spec["frozen_gates"]["visible_contact_alignment_px_max"])
        )
    )
    receipt["appearance_qualification"] = {
        "alignment_passed": receipt["alignment_gate"]["passed"],
        "rendered_near_miss_gap_passed": near_miss_gate_passed,
        "projected_contact_alignment_passed": contact_gate_passed,
        "all_frame_depth_stream_present": bool(
            not all_frames or (output_dir / "overlay_depth_m.npy").is_file()
        ),
        "physics_and_camera_authority": "read-only EpisodeTrace replay",
    }
    receipt["appearance_qualification"]["passed"] = bool(
        all(receipt["appearance_qualification"].values())
    )
    receipt["status"] = (
        "appearance_qualified"
        if receipt["appearance_qualification"]["passed"]
        else "appearance_qualification_failed"
    )
    receipt_path = output_dir / "overlay_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--body-names", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--render-samples", type=int, default=16)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(arguments)
    print(
        json.dumps(
            run(
                args.trace,
                args.body_names,
                args.spec,
                args.output_dir,
                width=args.width,
                height=args.height,
                all_frames=args.all_frames,
                render_samples=args.render_samples,
                fps=args.fps,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
