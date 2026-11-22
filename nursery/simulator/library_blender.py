"""Build one Blender asset library per word: normalized, decimated, textured, with thumbnails.

Run inside Blender: blender -b --factory-startup -P library_blender.py -- job.json
"""

import json
import math
from pathlib import Path
import random
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector


MAX_FACES = 40000
MAX_TEXTURE = 512


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in new if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh in GLB")
    bpy.context.view_layer.update()
    for obj in meshes:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
    for obj in new:
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        # Shared mesh data cannot be transformed per object; make each single-user first.
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    for modifier in list(obj.modifiers):
        obj.modifiers.remove(modifier)
    if obj.data.shape_keys:
        obj.shape_key_clear()
    return obj


def normalize(obj, size, rotation):
    obj.matrix_world = Matrix.Rotation(math.radians(rotation[0]), 4, "X") @ \
        Matrix.Rotation(math.radians(rotation[1]), 4, "Y") @ obj.matrix_world
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    coords = [v.co for v in obj.data.vertices]
    low = Vector([min(c[i] for c in coords) for i in range(3)])
    high = Vector([max(c[i] for c in coords) for i in range(3)])
    extent = max(high - low)
    if extent <= 0:
        raise RuntimeError("Degenerate mesh")
    scale = size / extent
    offset = Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))
    obj.data.transform(Matrix.Diagonal((scale, scale, scale, 1)) @ Matrix.Translation(-offset))
    obj.data.update()
    return scale


def decimate(obj):
    faces = len(obj.data.polygons)
    if faces > MAX_FACES:
        modifier = obj.modifiers.new("decimate", "DECIMATE")
        modifier.ratio = MAX_FACES / faces
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    return len(obj.data.polygons)


def shrink_textures(obj):
    textured = False
    for material in obj.data.materials:
        if material is None or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                image = node.image
                textured = True
                width, height = image.size
                if max(width, height) > MAX_TEXTURE:
                    factor = MAX_TEXTURE / max(width, height)
                    image.scale(max(1, int(width * factor)), max(1, int(height * factor)))
                image.pack()
    return textured


def material(name, color, image=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    mat.diffuse_color = (*color, 1)
    if image is not None:
        node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(image))
        node.image.pack()
        mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _hsv(rng, s=(0.5, 0.9), v=(0.5, 0.95)):
    import colorsys
    return colorsys.hsv_to_rgb(rng.random(), rng.uniform(*s), rng.uniform(*v))


def _single_material(obj):
    # Boolean cuts add the cutter's empty slots; keep one slot so the color applies everywhere.
    obj.data.materials.clear()
    for polygon in obj.data.polygons:
        polygon.material_index = 0


def procedural(word, seed, image):
    rng = random.Random(seed)
    if word == "balloon":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, radius=0.5)
        obj = bpy.context.active_object
        for v in obj.data.vertices:
            z = v.co.z
            v.co.z *= rng.uniform(1.15, 1.3)
            if z < 0:
                v.co.x *= 1 + z * 0.35
                v.co.y *= 1 + z * 0.35
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.05, radius2=0.015, depth=0.08,
                                        location=(0, 0, -0.66))
        knot = bpy.context.active_object
        color = _hsv(rng, (0.6, 1.0), (0.6, 1.0))
        for part in (obj, knot):
            part.data.materials.append(material(f"balloon_{seed}", color))
    elif word == "button":
        holes = rng.choice((2, 4, 4))
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.5, depth=rng.uniform(0.1, 0.18))
        obj = bpy.context.active_object
        for index in range(holes):
            angle = 2 * math.pi * index / holes + math.pi / 4
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.07, depth=0.5,
                                                location=(0.17 * math.cos(angle), 0.17 * math.sin(angle), 0))
            cutter = bpy.context.active_object
            modifier = obj.modifiers.new("hole", "BOOLEAN")
            modifier.object, modifier.operation = cutter, "DIFFERENCE"
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.data.objects.remove(cutter, do_unlink=True)
        _single_material(obj)
        bpy.ops.mesh.primitive_torus_add(major_radius=0.46, minor_radius=0.05, location=(0, 0, 0.04))
        rim = bpy.context.active_object
        color = _hsv(rng, (0.0, 0.9), (0.2, 0.95))
        for part in (obj, rim):
            part.data.materials.append(material(f"button_{seed}", color))
    elif word == "tape":
        width = rng.uniform(0.25, 0.5)
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.5, depth=width)
        obj = bpy.context.active_object
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=rng.uniform(0.32, 0.4), depth=width * 2)
        cutter = bpy.context.active_object
        modifier = obj.modifiers.new("core", "BOOLEAN")
        modifier.object, modifier.operation = cutter, "DIFFERENCE"
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
        _single_material(obj)
        palette = [(0.55, 0.56, 0.58), (0.85, 0.75, 0.52), (0.08, 0.08, 0.08), (0.75, 0.1, 0.1),
                   (0.1, 0.3, 0.75), (0.95, 0.85, 0.1), (0.1, 0.55, 0.2), (0.9, 0.9, 0.88)]
        obj.data.materials.append(material(f"tape_{seed}", rng.choice(palette)))
    elif word == "bagel":
        bpy.ops.mesh.primitive_torus_add(major_segments=64, minor_segments=24,
                                         major_radius=0.32, minor_radius=rng.uniform(0.16, 0.2))
        obj = bpy.context.active_object
        for v in obj.data.vertices:
            v.co.z *= rng.uniform(0.7, 0.85)
            v.co *= 1 + 0.03 * math.sin(7 * math.atan2(v.co.y, v.co.x) + seed)
        top = _hsv(rng, (0.55, 0.75), (0.45, 0.65))
        top = (top[0], top[1] * 0.75, top[2] * 0.45) if rng.random() < 0.2 else top
        obj.data.materials.append(material(f"bagel_top_{seed}", (0.55, 0.33, 0.12)))
        obj.data.materials.append(material(f"bagel_side_{seed}", (0.85, 0.66, 0.4)))
        for polygon in obj.data.polygons:
            polygon.material_index = 0 if polygon.normal.z > 0.35 else 1
        if rng.random() < 0.5:
            seeds = (0.95, 0.9, 0.8) if rng.random() < 0.6 else (0.05, 0.05, 0.06)
            mat = material(f"bagel_seed_{seed}", seeds)
            for _ in range(80):
                angle, around = rng.uniform(0, 2 * math.pi), rng.uniform(-0.9, 0.9)
                radius = 0.32 + 0.17 * math.sin(around)
                bpy.ops.mesh.primitive_uv_sphere_add(segments=6, ring_count=4, radius=0.018,
                                                     location=(radius * math.cos(angle), radius * math.sin(angle),
                                                               0.13 * math.cos(around)))
                speck = bpy.context.active_object
                speck.scale.z = 0.4
                speck.data.materials.append(mat)
    elif word == "bill":
        bpy.ops.mesh.primitive_plane_add(size=1)
        obj = bpy.context.active_object
        obj.scale = (1, 0.425, 1)
        bpy.ops.object.transform_apply(scale=True)
        solid = obj.modifiers.new("solid", "SOLIDIFY")
        solid.thickness = 0.004
        bpy.ops.object.modifier_apply(modifier=solid.name)
        obj.data.materials.append(material(f"bill_{seed}", (1, 1, 1), image))
        for v in obj.data.vertices:
            v.co.z += 0.01 * math.sin(v.co.x * 3 + seed)
    else:
        raise RuntimeError(f"No procedural recipe for {word}")
    bpy.ops.object.select_all(action="DESELECT")
    for part in bpy.context.scene.objects:
        part.select_set(part.type == "MESH")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(40))
    return obj


def thumbnail(obj, path):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x = scene.render.resolution_y = 160
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("world")
    scene.world.color = (0.6, 0.6, 0.6)
    camera = bpy.data.objects.get("thumb_camera")
    if camera is None:
        camera = bpy.data.objects.new("thumb_camera", bpy.data.cameras.new("thumb_camera"))
        scene.collection.objects.link(camera)
    scene.camera = camera
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector()) / 8
    radius = max((c - center).length for c in corners)
    camera.data.angle = math.radians(40)
    camera.location = center + Vector((1.0, -1.6, 0.9)).normalized() * radius * 3.2
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.clip_start, camera.data.clip_end = radius * 0.01, radius * 20
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    output = Path(job["output"])
    results = []
    clear_scene()
    collection = bpy.data.collections.new(job["word"])
    for entry in job["assets"]:
        name = f"a_{entry['id']}"
        record = {"id": entry["id"], "name": name}
        try:
            existing = set(bpy.data.objects)
            if entry["source"] == "objaverse":
                obj = import_glb(entry["path"])
            else:
                obj = procedural(job["word"], entry["seed"], entry.get("image"))
            scale = normalize(obj, entry["size_m"], entry.get("rotation", (0, 0)))
            faces = decimate(obj)
            textured = shrink_textures(obj)
            if entry["source"] == "procedural":
                textured = True
            obj.name, obj.data.name = name, name
            for other in set(bpy.data.objects) - existing - {obj}:
                bpy.data.objects.remove(other, do_unlink=True)
            for users in list(obj.users_collection):
                users.objects.unlink(obj)
            collection.objects.link(obj)
            # Thumbnails render from a temporary scene link only.
            bpy.context.scene.collection.objects.link(obj)
            thumbnail(obj, output / "thumbs" / f"{entry['id']}.png")
            bpy.context.scene.collection.objects.unlink(obj)
            mesh = obj.data
            mesh.calc_loop_triangles()
            vertices = [list(v.co) for v in mesh.vertices]
            triangles = [list(t.vertices) for t in mesh.loop_triangles]
            (output / "geometry").mkdir(parents=True, exist_ok=True)
            (output / "geometry" / f"{entry['id']}.json").write_text(json.dumps({"vertices": vertices, "faces": triangles}))
            dims = [max(v[i] for v in vertices) - min(v[i] for v in vertices) for i in range(3)]
            record.update(status="ok", faces=faces, textured=textured, source_scale=scale,
                          extents_m=dims, materials=len(mesh.materials))
        except Exception as error:
            record.update(status="failed", error=str(error)[:300])
            for other in set(bpy.data.objects) - existing:
                bpy.data.objects.remove(other, do_unlink=True)
        results.append(record)
    for obj in list(bpy.context.scene.collection.objects):
        bpy.context.scene.collection.objects.unlink(obj)
    bpy.context.scene.collection.children.link(collection)
    for image in bpy.data.images:
        if image.users and not image.packed_file and image.source == "FILE":
            image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / f"{job['word']}.blend"), compress=True)
    (output / f"{job['word']}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
