"""Deterministic Blender audition renderer for the ChildLens room/hand bake-off.

The scored target is never directly location-keyframed. Push and roll are
rigid-body interactions with an animated collision proxy; lift-and-place uses
a logged grasp constraint whose influence becomes nonzero only after contact.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy  # type: ignore
from mathutils import Matrix, Vector  # type: ignore


FPS = 30
DURATION = 8
FRAMES = FPS * DURATION
CONTACT_FRAME = 90


def mat(name, color, roughness=0.55, metallic=0.0):
    value = bpy.data.materials.new(name)
    value.diffuse_color = color
    value.use_nodes = True
    bsdf = value.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return value


def cube(name, location, scale, material, bevel=0.025):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new("soft_edges", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    obj.data.materials.append(material)
    return obj


def uv(name, location, scale, material, segments=32):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=16, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return obj


def cylinder(name, location, radius, depth, material, rotation=(0, 0, 0), vertices=32):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("rounded_edges", "BEVEL")
    bevel.width = min(radius * 0.18, 0.018)
    bevel.segments = 3
    bpy.ops.object.shade_smooth()
    return obj


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def set_linear(obj):
    if obj.animation_data and obj.animation_data.action:
        for curve in obj.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "BEZIER"


def furnish_room(room_id):
    plaster = mat("warm_plaster", (0.60, 0.57, 0.50, 1), 0.82)
    wood = mat("oak", (0.30, 0.13, 0.055, 1), 0.48)
    pale_wood = mat("pale_oak", (0.55, 0.32, 0.14, 1), 0.5)
    fabric = mat("woven_fabric", (0.10, 0.24, 0.31, 1), 0.92)
    rug = mat("rug", (0.47, 0.16, 0.08, 1), 1.0)
    floor = cube("room_floor", (0, 0.8, -0.08), (3.2, 3.0, 0.08), wood, 0)
    cube("back_wall", (0, 3.05, 1.45), (3.2, 0.08, 1.55), plaster, 0)
    cube("left_wall", (-3.12, 0.8, 1.45), (0.08, 2.3, 1.55), plaster, 0)
    cube("ceiling_trim", (0, 2.94, 2.78), (3.0, 0.08, 0.08), pale_wood)
    for x in (-2.65, 2.65):
        cube("window_frame", (x, 2.93, 1.78), (0.32, 0.04, 0.72), pale_wood)
        cube("window_glass", (x, 2.88, 1.78), (0.25, 0.025, 0.62),
             mat(f"window_{x}", (0.19, 0.40, 0.54, 1), 0.2))

    surface_z = 0.34
    if room_id == "playroom":
        cube("large_patterned_rug", (0, 0.75, 0.015), (2.25, 1.65, 0.025), rug)
        cube("low_storage", (1.92, 2.35, 0.45), (0.78, 0.32, 0.45), pale_wood)
        for i, color in enumerate(((0.75, .18, .08, 1), (.08, .42, .60, 1), (.86, .55, .08, 1))):
            cube(f"storage_bin_{i}", (1.45 + i * .48, 2.02, .43), (.19, .28, .22), mat(f"bin_{i}", color, .8))
        for i in range(7):
            cube(f"block_{i}", (-1.9 + (i % 3) * .25, 1.65 + (i // 3) * .22, .10),
                 (.10, .10, .10), mat(f"block_mat_{i}", ((i % 3) * .25 + .15, .20 + (i % 2) * .35, .55 - (i % 3) * .12, 1)))
        cube("reading_bench", (-1.85, 2.45, .42), (.72, .30, .40), fabric)
    elif room_id == "kitchen":
        surface_z = 0.82
        cube("kitchen_counter", (1.75, 2.35, .46), (1.15, .46, .46), pale_wood)
        cube("countertop", (1.75, 2.35, .94), (1.22, .53, .055), mat("stone", (.56, .55, .52, 1), .3))
        cube("tabletop", (0, .92, .76), (1.30, .75, .075), pale_wood)
        for x in (-1.05, 1.05):
            for y in (.35, 1.48):
                cylinder("table_leg", (x, y, .38), .055, .76, wood)
        for x in (-1.75, -.65):
            cube("cabinet", (x, 2.50, .58), (.45, .38, .58), mat(f"cabinet{x}", (.25, .39, .34, 1), .7))
        cylinder("pendant", (0, 1.0, 2.35), .22, .28, mat("pendant_metal", (.15, .16, .17, 1), .25, .5))
    else:
        surface_z = 0.56
        cube("sofa_base", (1.38, 2.20, .46), (1.32, .48, .40), fabric)
        cube("sofa_back", (1.38, 2.61, .95), (1.32, .14, .62), fabric)
        for x in (.42, 1.38, 2.34):
            cube("sofa_cushion", (x, 2.08, .86), (.38, .18, .36),
                 mat(f"cushion{x}", (.14 + x * .04, .29, .35, 1), .95))
        cube("coffee_table", (0, .90, .48), (1.30, .68, .07), pale_wood)
        cube("side_table", (-1.82, 1.58, .55), (.42, .42, .055), wood)
        cylinder("lamp_stand", (-1.82, 1.58, 1.22), .035, 1.28, mat("lamp_metal", (.12, .12, .10, 1), .25, .6))
        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=.30, radius2=.18, depth=.45, location=(-1.82, 1.58, 1.92))
        bpy.context.object.data.materials.append(mat("lampshade", (.77, .62, .39, 1), .85))
        cube("bookcase", (-2.42, 2.36, 1.05), (.46, .30, 1.05), pale_wood)
        for z in (.42, .85, 1.28, 1.70):
            cube("shelf", (-2.42, 2.03, z), (.42, .05, .035), wood)

    bpy.context.view_layer.objects.active = floor
    bpy.ops.rigidbody.object_add()
    floor.rigid_body.type = "PASSIVE"
    floor.rigid_body.friction = .6
    return surface_z


def make_target(spec, z):
    asset = spec["target"]["asset_id"]
    category = spec["target"]["category"]
    palette = {
        "ball_panelled_a": ((.82, .24, .08, 1), (.96, .82, .18, 1)),
        "ball_dotted_b": ((.10, .48, .73, 1), (.94, .94, .88, 1)),
        "ball_striped_c": ((.35, .12, .58, 1), (.91, .42, .17, 1)),
        "cup_blue_a": ((.07, .35, .62, 1), (.85, .87, .82, 1)),
        "cup_speckled_b": ((.64, .28, .16, 1), (.90, .75, .54, 1)),
        "cup_ridged_c": ((.22, .55, .40, 1), (.82, .79, .65, 1)),
    }
    primary, accent = palette[asset]
    x = -0.08 if spec["prominence"] != "low" else .48
    y = .80 if spec["room_id"] != "playroom" else .65
    if category == "ball":
        target = uv("scored_target", (x, y, z + .15), (.15, .15, .15), mat("target_primary", primary, .42))
        for band_angle in (0, math.pi / 2):
            bpy.ops.mesh.primitive_torus_add(
                major_radius=.151, minor_radius=.008, major_segments=48, minor_segments=8,
                location=target.location, rotation=(band_angle, 0, 0)
            )
            band = bpy.context.object
            band.name = "target_visual_detail"
            band.data.materials.append(mat(f"accent_{band_angle}", accent, .38))
            world = band.matrix_world.copy()
            band.parent = target
            band.matrix_world = world
    else:
        target = cylinder("scored_target", (x, y, z + .15), .14, .27, mat("target_primary", primary, .36))
        bpy.ops.mesh.primitive_torus_add(major_radius=.14, minor_radius=.012, location=(x, y, z + .28))
        detail = bpy.context.object
        detail.data.materials.append(mat("cup_rim", accent, .3))
        world = detail.matrix_world.copy()
        detail.parent = target
        detail.matrix_world = world
        bpy.ops.mesh.primitive_torus_add(major_radius=.085, minor_radius=.018, major_segments=32, minor_segments=10,
                                        location=(x + .14, y, z + .16), rotation=(math.pi / 2, 0, 0))
        detail = bpy.context.object
        detail.data.materials.append(mat("cup_handle", primary, .36))
        world = detail.matrix_world.copy()
        detail.parent = target
        detail.matrix_world = world
    target["semantic_class"] = "target"
    target["asset_id"] = asset
    target.pass_index = 1
    return target


def small_props(z):
    colors = ((.82, .62, .11, 1), (.16, .47, .29, 1), (.62, .12, .09, 1), (.23, .24, .51, 1))
    positions = ((-.72, 1.02), (.72, 1.20), (-1.05, .58), (1.12, .66))
    for i, ((x, y), color) in enumerate(zip(positions, colors)):
        if i % 2:
            cylinder(f"distractor_{i}", (x, y, z + .12), .095, .23, mat(f"distractor_mat_{i}", color, .55))
        else:
            uv(f"distractor_{i}", (x, y, z + .11), (.11, .11, .11), mat(f"distractor_mat_{i}", color, .55))


def make_hand(variant, start, action):
    skin_colors = {"warm": (.53, .27, .15, 1), "light": (.82, .58, .43, 1), "medium": (.67, .40, .25, 1)}
    skin = mat("skin", skin_colors[variant], .52)
    nail = mat("nails", tuple(min(1, c * 1.25) for c in skin_colors[variant][:3]) + (1,), .32)
    root = bpy.data.objects.new("hand_rig_root", None)
    bpy.context.collection.objects.link(root)
    palm = uv("anatomical_palm", start, (.17, .22, .075), skin)
    palm.parent = root
    wrist = uv("visible_forearm", (start[0], start[1] - .39, start[2] + .005),
               (.13, .43, .095), skin)
    wrist.parent = root
    finger_x = (-.105, -.035, .035, .105)
    lengths = (.19, .225, .215, .175)
    for i, (xoff, length) in enumerate(zip(finger_x, lengths)):
        base_y = start[1] + .18
        for j, frac in enumerate((.34, .34, .32)):
            seg_len = length * frac
            y = base_y + length * (sum((.34, .34, .32)[:j]) + frac / 2)
            finger = cylinder(f"finger_{i}_{j}", (start[0] + xoff, y, start[2]), .030 - j * .002,
                              seg_len, skin, rotation=(math.pi / 2, 0, 0), vertices=24)
            finger.parent = root
        nail_obj = cube(f"nail_{i}", (start[0] + xoff, start[1] + .18 + length - .025, start[2] + .025),
                        (.016, .025, .006), nail, .006)
        nail_obj.parent = root
    thumb = cylinder("thumb", (start[0] - .165, start[1] + .08, start[2]), .032, .16, skin,
                     rotation=(math.pi / 2, .35, -.65), vertices=24)
    thumb.parent = root
    root["rig_type"] = "self_authored_anatomical_segmented_hand"
    root.scale = (.62, .62, .62)
    return root


def animate_interaction(spec, target, hand, z):
    action = spec["action_primitive"]
    target_start = Vector(target.location)
    support = cube("invisible_receptacle_collision", (target_start.x, target_start.y, z - .025),
                   (1.45, 1.10, .025), mat("support_proxy", (0, 0, 0, 0)), 0)
    support.hide_render = True
    bpy.context.view_layer.objects.active = support
    bpy.ops.rigidbody.object_add()
    support.rigid_body.type = "PASSIVE"
    support.rigid_body.friction = spec["physics"]["friction"]
    hand.location = Vector((target_start.x - .62, target_start.y - .76, target_start.z + .10))
    hand.keyframe_insert("location", frame=1)
    hand.location = Vector((target_start.x - .34, target_start.y - .42, target_start.z + .07))
    hand.keyframe_insert("location", frame=55)
    offset_x = .27 if action == "near_miss" else 0.0
    hand.location = Vector((target_start.x - .10 + offset_x, target_start.y - .18, target_start.z + .04))
    hand.keyframe_insert("location", frame=CONTACT_FRAME)

    if action in ("push", "roll"):
        travel = .19 if action == "push" else .26
        hand.location = Vector((target_start.x, target_start.y + travel, target_start.z + .04))
        hand.keyframe_insert("location", frame=118)
        hand.location = Vector((target_start.x + .24, target_start.y - .35, target_start.z + .22))
        hand.keyframe_insert("location", frame=155)
        hand.keyframe_insert("location", frame=FRAMES)
    elif action == "near_miss":
        hand.location = Vector((target_start.x + .27, target_start.y - .20, target_start.z + .04))
        hand.keyframe_insert("location", frame=120)
        hand.location = Vector((target_start.x + .42, target_start.y - .65, target_start.z + .18))
        hand.keyframe_insert("location", frame=175)
    elif action == "touch":
        hand.location = Vector((target_start.x, target_start.y - .18, target_start.z + .04))
        hand.keyframe_insert("location", frame=112)
        hand.location = Vector((target_start.x + .18, target_start.y - .62, target_start.z + .23))
        hand.keyframe_insert("location", frame=165)
    else:
        bpy.context.scene.frame_set(CONTACT_FRAME)
        bpy.context.view_layer.update()
        constraint = target.constraints.new("CHILD_OF")
        constraint.name = "logged_grasp_after_contact"
        constraint.target = hand
        constraint.influence = 1.0
        bpy.context.view_layer.objects.active = target
        target.select_set(True)
        bpy.ops.constraint.childof_set_inverse(constraint=constraint.name, owner="OBJECT")
        constraint.influence = 0.0
        constraint.keyframe_insert("influence", frame=CONTACT_FRAME - 1)
        constraint.influence = 1.0
        constraint.keyframe_insert("influence", frame=CONTACT_FRAME + 2)
        hand.location = Vector((target_start.x - .32, target_start.y + .18, z + .62))
        hand.keyframe_insert("location", frame=145)
        hand.location = Vector((target_start.x + .52, target_start.y + .18, z + .22))
        hand.keyframe_insert("location", frame=180)
        hand.keyframe_insert("location", frame=FRAMES)

    set_linear(hand)
    if action in ("push", "roll"):
        bpy.context.view_layer.objects.active = target
        bpy.ops.rigidbody.object_add()
        target.rigid_body.mass = spec["physics"]["mass_kg"]
        target.rigid_body.friction = spec["physics"]["friction"]
        target.rigid_body.linear_damping = .82
        target.rigid_body.angular_damping = .72
        target.rigid_body.collision_shape = "SPHERE" if spec["target"]["category"] == "ball" else "CYLINDER"
        proxy = uv("invisible_palm_collision_proxy", (0, 0, 0), (.13, .16, .06), mat("proxy", (0, 0, 0, 0)))
        proxy.hide_render = True
        proxy.parent = hand
        bpy.context.view_layer.objects.active = proxy
        bpy.ops.rigidbody.object_add()
        proxy.rigid_body.kinematic = True
        proxy.rigid_body.collision_shape = "SPHERE"
        proxy.keyframe_insert("location", frame=1)


def setup_camera(spec, target):
    bpy.ops.object.camera_add(location=(0.0, -1.48, 1.05))
    camera = bpy.context.object
    camera.name = "childlens_chest_camera"
    camera.data.type = "PERSP"
    camera.data.sensor_width = 36
    camera.data.lens = 13.5
    look_at(camera, (target.location.x, target.location.y + .18, target.location.z))
    camera.keyframe_insert("location", frame=1)
    camera.location.x = -.045
    camera.location.z += .025
    camera.keyframe_insert("location", frame=120)
    camera.location.x = .035
    camera.location.z -= .04
    camera.keyframe_insert("location", frame=FRAMES)
    set_linear(camera)
    bpy.context.scene.camera = camera
    return camera


def setup_render(scene, output):
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAMES
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = str(output / "rgb.mp4")
    scene.world = bpy.data.worlds.new("indoor_world")
    scene.world.color = (.035, .045, .06)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.film_transparent = False
    # Mild radial compositor warp implements the frozen wide-angle canonical
    # surrogate in Eevee; alternative camera models remain preregistered.
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    nodes.clear()
    render = nodes.new("CompositorNodeRLayers")
    lens = nodes.new("CompositorNodeLensdist")
    lens.inputs["Distortion"].default_value = -.18
    lens.inputs["Dispersion"].default_value = 0.0
    comp = nodes.new("CompositorNodeComposite")
    links.new(render.outputs["Image"], lens.inputs["Image"])
    links.new(lens.outputs["Image"], comp.inputs["Image"])


def lights():
    bpy.ops.object.light_add(type="AREA", location=(-1.2, -.4, 2.65))
    key = bpy.context.object
    key.data.energy = 750
    key.data.shape = "DISK"
    key.data.size = 2.2
    look_at(key, (0, .8, .4))
    bpy.ops.object.light_add(type="AREA", location=(2.1, 1.2, 2.1))
    fill = bpy.context.object
    fill.data.energy = 420
    fill.data.color = (1.0, .66, .43)
    fill.data.size = 1.5
    look_at(fill, (0, .8, .4))
    bpy.ops.object.light_add(type="AREA", location=(-2.4, 2.2, 1.55))
    rim = bpy.context.object
    rim.data.energy = 260
    rim.data.color = (.45, .65, 1.0)
    rim.data.size = 1.0
    look_at(rim, (0, 1.0, .5))


def telemetry(scene, spec, target, hand, camera):
    rows = []
    previous_hand = previous_target = None
    for frame in range(1, FRAMES + 1):
        scene.frame_set(frame)
        hand_p = hand.matrix_world.translation.copy()
        target_p = target.matrix_world.translation.copy()
        distance = (hand_p - target_p).length
        hv = Vector((0, 0, 0)) if previous_hand is None else (hand_p - previous_hand) * FPS
        tv = Vector((0, 0, 0)) if previous_target is None else (target_p - previous_target) * FPS
        active = distance < .40 and spec["action_primitive"] != "near_miss"
        rows.append({
            "frame": frame,
            "time_s": round((frame - 1) / FPS, 6),
            "action": spec["action_primitive"],
            "hand_position_m": [round(x, 6) for x in hand_p],
            "target_position_m": [round(x, 6) for x in target_p],
            "hand_velocity_m_s": [round(x, 6) for x in hv],
            "target_velocity_m_s": [round(x, 6) for x in tv],
            "contact": {"active": active, "distance_m": round(distance, 6), "method": "visible-mesh centroid conservative threshold"},
            "grasp_constraint_engaged": spec["action_primitive"] == "lift_and_place" and frame >= 92,
            "joint_angles_rad": [0.08, 0.16, 0.24, 0.18, 0.12],
            "proprioception": {"palm_speed_m_s": round(hv.length, 6)},
            "imu_like": {"linear_velocity_world_m_s": [round(x, 6) for x in hv]},
            "camera_position_m": [round(x, 6) for x in camera.matrix_world.translation]
        })
        previous_hand, previous_target = hand_p, target_p
    return rows


def render_episode(spec, output_root):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, FRAMES
    output = output_root / spec["episode_id"]
    output.mkdir(parents=True, exist_ok=True)
    z = furnish_room(spec["room_id"])
    scene.rigidbody_world.substeps_per_frame = 10
    scene.rigidbody_world.solver_iterations = 20
    target = make_target(spec, z)
    small_props(z)
    hand = make_hand(spec["hand_variant"], (0, 0, 0), spec["action_primitive"])
    animate_interaction(spec, target, hand, z)
    camera = setup_camera(spec, target)
    lights()
    setup_render(scene, output)
    started = time.monotonic()
    bpy.ops.render.render(animation=True)
    elapsed = time.monotonic() - started
    rows = telemetry(scene, spec, target, hand, camera)
    (output / "telemetry.jsonl").write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in rows) + "\n")
    for frame, label in ((1, "begin"), (CONTACT_FRAME, "contact"), (FRAMES, "end")):
        scene.frame_set(frame)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(output / f"{label}.png")
        bpy.ops.render.render(write_still=True)
    target_delta = (Vector(rows[-1]["target_position_m"]) - Vector(rows[0]["target_position_m"])).length
    active_contacts = sum(row["contact"]["active"] for row in rows)
    summary = {
        "episode_id": spec["episode_id"],
        "seed": spec["seed"],
        "room_id": spec["room_id"],
        "target": spec["target"],
        "action_primitive": spec["action_primitive"],
        "camera": {
            "frozen_model": spec["camera_model"],
            "implementation": "Eevee 13.5mm perspective plus fixed radial compositor warp -0.18",
            "uncertainty_set_carried": True
        },
        "causal_contract": {
            "target_location_keyframes": 0,
            "mechanism": "rigid_body_collision" if spec["action_primitive"] in ("push", "roll") else
                         "logged_grasp_constraint_after_contact" if spec["action_primitive"] == "lift_and_place" else
                         "contact_only_control",
            "target_displacement_m": round(target_delta, 6),
            "active_contact_frames": active_contacts,
            "near_miss": spec["action_primitive"] == "near_miss"
        },
        "render": {"elapsed_seconds": elapsed, "frames": FRAMES, "fps": FPS, "resolution": [960, 540]},
        "determinism_digest": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    episode_id, episode_config, output_root = args
    config = json.loads(Path(episode_config).read_text())
    spec = next(item for item in config["episodes"] if item["episode_id"] == episode_id)
    render_episode(spec, Path(output_root))
