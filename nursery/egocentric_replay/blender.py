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


def open_area(vertices, objects, floors, obstacles):
    low, high = vertices.min(axis=(0, 1)), vertices.max(axis=(0, 1))
    for obj in objects:
        a, b = bounds(obj)
        low, high = np.minimum(low, a), np.maximum(high, b)
    half = (high - low) / 2 + [0.10, 0.10, 0]
    center = (high + low) / 2
    for floor in floors:
        a, b = bounds(floor)
        xs = sorted(np.linspace(a[0] + half[0], b[0] - half[0], 9),
                    key=lambda value: abs(value - (a[0] + b[0]) / 2))
        ys = sorted(np.linspace(a[1] + half[1], b[1] - half[1], 9),
                    key=lambda value: abs(value - (a[1] + b[1]) / 2))
        for x in xs:
            for y in ys:
                if x - half[0] < a[0] or x + half[0] > b[0]:
                    continue
                if y - half[1] < a[1] or y + half[1] > b[1]:
                    continue
                inverse = floor.matrix_world.inverted()
                supported = True
                for dx in (-half[0], 0, half[0]):
                    for dy in (-half[1], 0, half[1]):
                        origin = inverse @ Vector((x + dx, y + dy, b[2] + 0.1))
                        direction = inverse.to_3x3() @ Vector((0, 0, -1))
                        hit, point, normal, _ = floor.ray_cast(origin, direction)
                        if not hit or (floor.matrix_world.to_3x3() @ normal).normalized().z < 0.95:
                            supported = False
                if not supported:
                    continue
                shift = np.array([x - center[0], y - center[1], b[2] - low[2]])
                lower, upper = low + shift, high + shift
                if all(not (np.all(upper > bounds(obj)[0]) and np.all(lower < bounds(obj)[1]))
                       for obj in obstacles):
                    return shift
    raise RuntimeError("No conservative open floor area fits the recorded interaction")


def render_pair(image, mask):
    image.parent.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(image)
    scene.cycles.samples = config["samples"]
    bpy.ops.render.render(write_still=True)
    original = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        original.append((obj, list(obj.data.materials)))
        material = bpy.data.materials.get(f"instance_{obj.pass_index}")
        if material is None:
            material = bpy.data.materials.new(f"instance_{obj.pass_index}")
            material.use_nodes = True
            nodes = material.node_tree.nodes
            nodes.clear()
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (obj.pass_index / 255, 0, 0, 1)
            out = nodes.new("ShaderNodeOutputMaterial")
            material.node_tree.links.new(emission.outputs[0], out.inputs[0])
        obj.data.materials.clear()
        obj.data.materials.append(material)
    previous = scene.view_settings.view_transform
    world = scene.world.node_tree.nodes.get("Background") if scene.world.use_nodes else None
    strength = world.inputs["Strength"].default_value if world else None
    if world:
        world.inputs["Strength"].default_value = 0
    scene.view_settings.view_transform = "Raw"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.filter_size = 0.01
    scene.render.dither_intensity = 0
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(mask)
    bpy.ops.render.render(write_still=True)
    for obj, materials in original:
        obj.data.materials.clear()
        for material in materials:
            obj.data.materials.append(material)
    scene.view_settings.view_transform = previous
    if world:
        world.inputs["Strength"].default_value = strength


def main():
    global bpy, Matrix, Vector, scene, config

    import bpy
    from mathutils import Matrix, Vector

    job = json.loads(Path(arguments[2]).read_text())
    config, output = job["config"], Path(job["output"])
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = config["device"]
    if config["device"] == "GPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL" if sys.platform == "darwin" else "CUDA"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type != "CPU"
        if not any(device.use for device in preferences.devices):
            raise RuntimeError("No Cycles GPU available")
    scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.quality = 90
    scene.world.color = (0.35, 0.35, 0.35)
    scene.cycles.use_denoising = False
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
        shift = open_area(np.array([[a, b, floor_anchor]]), [table, active], floors, obstacles)
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
        scene.cycles.samples = config["samples"]
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
        low, high = bounds(table)
        distractors = []
        occupied = segment["trajectory_bounds"]
        for index, entry in enumerate(segment["distractors"], 2):
            obj = load_ply(entry["mesh"], entry["color"], index)
            a, b = bounds(obj)
            placed = False
            for x in np.linspace(low[0], high[0], 14):
                for y in np.linspace(low[1], high[1], 14):
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
                    if np.all(upper > np.array(occupied[0]) - 0.04) and np.all(lower < np.array(occupied[1]) + 0.04):
                        continue
                    if any(np.all(upper > bounds(other)[0] - 0.02) and
                           np.all(lower < bounds(other)[1] + 0.02) for other in distractors):
                        continue
                    obj.location = pos
                    bpy.context.view_layer.update()
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                raise RuntimeError("Cannot place distractors away from recorded trajectory")
            distractors.append(obj)
        actor = mesh_object("actor", arrays["vertices"][0], arrays["faces"], [0.7, 0.5, 0.35])
        extent_points = np.concatenate([arrays["vertices"].reshape(-1, 3),
                                        np.array(segment["trajectory_bounds"])])
        shift = open_area(extent_points[None], [table, active, *distractors], floors, obstacles)
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
    (output / "render_stats.json").write_text(json.dumps({"seconds": time.monotonic() - started,
                                                          "engine": "CYCLES", "device": config["device"],
                                                          "image_seconds": image_seconds,
                                                          "blender_version": bpy.app.version_string}))


if __name__ == "__main__":
    main()
