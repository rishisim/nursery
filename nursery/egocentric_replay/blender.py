"""Render replay jobs inside Blender after the shared URDF helper prelude."""

import json
import math
import time

import numpy as np


def mesh_object(name, vertices, faces, color, instance=0):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1)
    material.use_nodes = True
    material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (*color, 1)
    obj.data.materials.append(material)
    obj.pass_index = instance
    return obj


def load_ply(path, color, instance=0):
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=path)
    objects = [obj for obj in bpy.data.objects if obj not in before]
    if len(objects) != 1:
        raise RuntimeError("Expected a single GRAB mesh")
    obj = objects[0]
    material = bpy.data.materials.new(obj.name)
    material.diffuse_color = (*color, 1)
    material.use_nodes = True
    material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (*color, 1)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj.pass_index = instance
    return obj


def remove_object(obj):
    mesh = obj.data
    materials = list(mesh.materials)
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    for material in materials:
        if material.users == 0 and not material.name.startswith("instance_"):
            bpy.data.materials.remove(material)


def bounds(obj):
    corners = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    return corners.min(0), corners.max(0)


def import_room():
    floors, obstacles = [], []
    for link in root.findall("link"):
        name = link.get("name", "")
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename").removeprefix("file://")
            path = Path(filename)
            if not path.is_absolute():
                path = urdf_path.parent / path
            before = set(bpy.data.objects)
            bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
            scale = values(mesh, "scale", (1, 1, 1))
            transform = link_transform(name) @ origin_matrix(visual) @ Matrix.Diagonal((*scale, 1))
            for obj in set(bpy.data.objects) - before:
                obj.matrix_world = transform @ obj.matrix_world
                obj.pass_index = 0
                if name.casefold().endswith("_floor"):
                    floors.append(obj)
                elif not name.casefold().endswith("_exterior"):
                    obstacles.append(obj)
    bpy.context.view_layer.update()
    if not floors:
        raise RuntimeError("Room has no floor")
    return floors, obstacles


def table_surface(table, x, y):
    inverse = table.matrix_world.inverted()
    top = bounds(table)[1][2]
    origin = inverse @ Vector((x, y, top + 1))
    direction = inverse.to_3x3() @ Vector((0, 0, -1))
    hit, point, normal, _ = table.ray_cast(origin, direction)
    if not hit:
        return None
    return table.matrix_world @ point, (table.matrix_world.to_3x3() @ normal).normalized()


def rest_position(obj, point, normal):
    obj.rotation_euler = Vector((0, 0, 1)).rotation_difference(normal).to_euler()
    corners = np.array(obj.bound_box)
    low, high = corners.min(0), corners.max(0)
    anchor = Vector(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]))
    return point - obj.rotation_euler.to_matrix() @ anchor


def open_area(vertices, objects, square):
    angle = math.radians(square["square_orientation_degrees"])
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    points = [vertices.reshape(-1, 3)]
    points.extend(np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
                  for obj in objects)
    local = np.concatenate(points) @ rotation
    low, high = local.min(0), local.max(0)
    if np.any((high - low)[:2] > 2.5):
        raise RuntimeError("Recorded interaction does not fit certified 2.5m square")
    center = np.mean(square["square_vertices_m"], axis=0)
    return np.array([*center, square["floor_height_m"]]) - rotation @ np.array([
        *((high + low)[:2] / 2), low[2]])


def render_pair(image, mask):
    image.parent.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(image)
    scene.render.engine = "BLENDER_WORKBENCH"
    bpy.ops.render.render(write_still=True)
    original = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        original.append((obj, list(obj.data.materials)))
        material = bpy.data.materials.get(f"instance_{obj.pass_index}")
        if material is None:
            material = bpy.data.materials.new(f"instance_{obj.pass_index}")
            material.diffuse_color = (obj.pass_index / 255, 0, 0, 1)
        obj.data.materials.clear()
        obj.data.materials.append(material)
    previous = scene.view_settings.view_transform
    world = scene.world.node_tree.nodes.get("Background") if scene.world.use_nodes else None
    strength = world.inputs["Strength"].default_value if world else None
    if world:
        world.inputs["Strength"].default_value = 0
    scene.view_settings.view_transform = "Raw"
    shading = scene.display.shading
    display = (shading.light, shading.color_type, shading.show_shadows, shading.show_cavity,
               scene.display.render_aa)
    shading.light, shading.color_type = "FLAT", "MATERIAL"
    shading.show_shadows = shading.show_cavity = False
    scene.display.render_aa = "OFF"
    scene.render.filter_size = 0.01
    scene.render.dither_intensity = 0
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(mask)
    bpy.ops.render.render(write_still=True)
    for obj, materials in original:
        obj.data.materials.clear()
        for material in materials:
            obj.data.materials.append(material)
    shading.light, shading.color_type, shading.show_shadows, shading.show_cavity, scene.display.render_aa = display
    scene.view_settings.view_transform = previous
    if world:
        world.inputs["Strength"].default_value = strength


def _place_distractors(table, entries, count, occupied, rng):
    low, high = bounds(table)
    distractors = []
    categories = []
    # The shuffled category order is uniform; infeasible props are rejected.
    for entry in entries:
        obj = load_ply(entry["mesh"], entry["color"], len(distractors) + 2)
        bpy.context.view_layer.update()
        placed = False
        for _ in range(256):
            x, y = rng.uniform(low[:2], high[:2])
            surface = table_surface(table, x, y)
            if surface is None:
                continue
            point, normal = surface
            pos = np.array(rest_position(obj, point, normal))
            obj.location = (0, 0, 0)
            bpy.context.view_layer.update()
            a, b = bounds(obj)
            lower, upper = a + pos, b + pos
            if np.any(lower[:2] < low[:2] + 0.02) or np.any(upper[:2] > high[:2] - 0.02):
                continue
            if np.any(np.all(upper > occupied[:, 0] - 0.04, axis=1) &
                      np.all(lower < occupied[:, 1] + 0.04, axis=1)):
                continue
            if any(np.all(upper > bounds(other)[0] - 0.02) and
                   np.all(lower < bounds(other)[1] + 0.02) for other in distractors):
                continue
            obj.location = pos
            bpy.context.view_layer.update()
            placed = True
            break
        if not placed:
            remove_object(obj)
            continue
        distractors.append(obj)
        categories.append(entry["category"])
        if len(distractors) == count:
            break
    return distractors, categories


def main():
    global bpy, Matrix, Vector, scene, config

    import bpy
    from mathutils import Matrix, Vector

    job = json.loads(Path(arguments[2]).read_text())
    config, output = job["config"], Path(job["output"])
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 1
    scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.quality = 90
    scene.world.color = (0.35, 0.35, 0.35)
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    floors, obstacles = import_room()
    light_data = bpy.data.lights.new("room_light", "AREA")
    light_data.energy, light_data.size = 150, 5
    light = bpy.data.objects.new("room_light", light_data)
    scene.collection.objects.link(light)
    floor_low, floor_high = bounds(floors[0])
    light.location = ((floor_low[0] + floor_high[0]) / 2,
                      (floor_low[1] + floor_high[1]) / 2, floor_high[2] + 2.2)
    light.rotation_euler = (0, 0, 0)
    camera_data = bpy.data.cameras.new("head_camera")
    camera = bpy.data.objects.new("head_camera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.angle = math.radians(config["horizontal_fov_deg"])
    camera_data.clip_start = 0.02
    scene.camera = camera
    started = time.monotonic()
    image_seconds = []
    realized = {}
    clip_counts = []
    rng = np.random.default_rng(job.get("seed", 0))
    for probe in job.get("probes", []):
        table = load_ply(probe["table_mesh"], [0.55, 0.4, 0.25])
        table.matrix_world = Matrix(probe["table_pose"])
        active = load_ply(probe["mesh"], probe["color"], 1)
        bpy.context.view_layer.update()
        a, b = bounds(table)
        lower, upper = bounds(active)
        center_x, center_y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        surface = table_surface(table, center_x, center_y)
        if surface is None:
            raise RuntimeError("No tabletop surface under probe")
        point, normal = surface
        active.location = rest_position(active, point, normal)
        bpy.context.view_layer.update()
        floor_anchor = np.array([center_x, center_y, 0.0])
        shift = open_area(np.array([[a, b, floor_anchor]]), [table, active], job["clear_square"])
        translation = Matrix.Translation(Vector(shift))
        for obj in (table, active):
            obj.matrix_world = translation @ obj.matrix_world
        bpy.context.view_layer.update()
        low, high = bounds(active)
        target = Vector((low + high) / 2)
        extent = float(np.max(high - low))
        radius = max(0.04, extent * 1.8)
        camera.location = target + Vector((radius * math.cos(probe["yaw"]),
                                           radius * math.sin(probe["yaw"]),
                                           max(0.04, extent * 3 * probe["height"])))
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.image_settings.file_format = "JPEG"
        scene.render.filepath = str(output / probe["image"])
        (output / probe["image"]).parent.mkdir(parents=True, exist_ok=True)
        before_render = time.monotonic()
        bpy.ops.render.render(write_still=True)
        image_seconds.append(time.monotonic() - before_render)
        for obj in (table, active):
            remove_object(obj)
    for segment in job["segments"]:
        arrays = np.load(segment["arrays"])
        table = load_ply(segment["table_mesh"], [0.55, 0.4, 0.25])
        active = load_ply(segment["object_mesh"], segment["color"], 1)
        table.matrix_world = Matrix(arrays["table"][0].tolist())
        active.matrix_world = Matrix(arrays["object"][0].tolist())
        bpy.context.view_layer.update()
        occupied = np.asarray(segment["trajectory_bounds"] + segment["hand_bounds"])
        distractors, categories = _place_distractors(
            table, segment["distractors"], segment["distractor_count"], occupied, rng)
        clip_counts.append({"clip": segment["clip"], "segment_index": segment["segment_index"],
                            "target": segment["distractor_count"], "realized": len(categories)})
        realized.update({frame["image"]: categories for frame in segment["frames"]})
        actor = mesh_object("actor", arrays["vertices"][0], arrays["faces"], [0.7, 0.5, 0.35])
        extent_points = np.concatenate([arrays["vertices"].reshape(-1, 3),
                                        np.array(segment["trajectory_bounds"]).reshape(-1, 3)])
        shift = open_area(extent_points[None], [table, active, *distractors], job["clear_square"])
        translation = Matrix.Translation(Vector(shift))
        relative_distractors = [table.matrix_world.inverted() @ obj.matrix_world for obj in distractors]
        for index, frame in enumerate(segment["frames"]):
            actor.data.vertices.foreach_set("co", (arrays["vertices"][index] + shift).ravel())
            actor.data.update()
            table.matrix_world = translation @ Matrix(arrays["table"][index].tolist())
            active.matrix_world = translation @ Matrix(arrays["object"][index].tolist())
            for obj, relative in zip(distractors, relative_distractors):
                obj.matrix_world = table.matrix_world @ relative
            camera.matrix_world = translation @ Matrix(arrays["camera"][index].tolist())
            before_render = time.monotonic()
            render_pair(output / frame["image"], output / frame["mask"])
            image_seconds.append(time.monotonic() - before_render)
        for obj in [actor, table, active, *distractors]:
            remove_object(obj)
    stats_path = output / f"render_stats_{job.get('job_id', 'job')}.json"
    previous = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    image_seconds = previous.get("image_seconds", []) + image_seconds
    stats_path.write_text(json.dumps({"seconds": previous.get("seconds", 0) + time.monotonic() - started,
                                                          "engine": "BLENDER_WORKBENCH", "id_engine": "BLENDER_WORKBENCH", "threads": 1, "job_id": job.get("job_id", "job"),
                                                          "distractors": {**previous.get("distractors", {}), **realized},
                                                          "clip_counts": previous.get("clip_counts", []) + clip_counts,
                                                          "images": len(image_seconds),
                                                          "image_seconds": image_seconds,
                                                          "blender_version": bpy.app.version_string}))


if __name__ == "__main__":
    main()
