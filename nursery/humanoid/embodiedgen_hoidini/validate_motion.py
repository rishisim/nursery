"""Validate and optionally preview one saved scene-conditioned mug-lift run."""

import argparse
import json
from pathlib import Path

import numpy as np

from .collision import build_static_collision_scene, validate_trajectories
from .prepare_scene import prepare_scene


def evaluate_scene_collisions(
    bundle,
    body_vertices,
    object_vertices,
    *,
    subdivisions=2,
    tolerance_m=0.02,
):
    """Return compact collision metrics without storing per-vertex results."""
    active = bundle.active_interaction
    obstacles = build_static_collision_scene(bundle)
    samples = validate_trajectories(
        body_vertices,
        object_vertices,
        obstacles,
        object_faces=active.canonical_target_mesh.faces,
        object_template=active.canonical_target_mesh,
        object_name=active.target_name,
        subdivisions=subdivisions,
    )
    body_room = [
        row
        for row in samples
        if row.moving_geometry == "body" and row.obstacle != active.target_name
    ]
    object_room = [
        row
        for row in samples
        if row.moving_geometry == "object"
        and row.obstacle != active.support_name
    ]
    body_object = [
        row
        for row in samples
        if row.moving_geometry == "body" and row.obstacle == active.target_name
    ]

    def maximum(rows):
        return max((row.maximum_penetration_m for row in rows), default=0.0)

    body_room_depth = maximum(body_room)
    object_room_depth = maximum(object_room)
    body_object_depth = maximum(body_object)
    return {
        "static_collision_meshes": len(obstacles),
        "enclosing_boundary_meshes": sum(
            item.enclosing_boundary for item in obstacles
        ),
        "collision_subdivisions_per_frame": subdivisions,
        "collision_tolerance_m": tolerance_m,
        "maximum_body_room_penetration_m": body_room_depth,
        "maximum_object_room_penetration_m": object_room_depth,
        "maximum_body_object_penetration_m": body_object_depth,
        "checks": {
            "body_not_penetrating_room": body_room_depth <= tolerance_m,
            "object_not_penetrating_room": object_room_depth <= tolerance_m,
            "body_not_penetrating_target": body_object_depth <= tolerance_m,
        },
        "limitations": [
            "Open meshes are unsigned clearance surfaces, not solid volumes",
            "Linear subframes reduce tunnelling but are not continuous collision detection",
            "Target-support contact is assessed by the existing support-specific check",
        ],
    }


def validate_motion(output, *, render=False, collision_subdivisions=2):
    import torch
    from hoidini.amasstools.geometry import axis_angle_to_matrix
    from hoidini.closd.diffusion_planner.utils import dist_util
    from hoidini.datasets.grab.grab_utils import get_MANO_SMPLX_vertex_ids
    from hoidini.datasets.smpldata import SmplData, SmplModelsFK
    from hoidini.object_contact_prediction.cpdm_dno_conds import Table
    from hoidini.skeletons.mano_anchors.mano_contact_anchors import get_contact_anchors_info

    output = Path(output).resolve()
    receipt = json.loads((output / "input.json").read_text())
    args = receipt["motion_run"]
    bundle = prepare_scene(args["scene"], target=args["target"], prompt=args["prompt"], seed=args["seed"])
    active = bundle.active_interaction
    with np.load(output / "motion.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(output / "prefix.npz", allow_pickle=False) as archive:
        prefix = {key: archive[key] for key in archive.files}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Motion contains non-finite values")
    if not all(len(v) == args["frames"] for v in arrays.values()):
        raise ValueError("Saved motion arrays have inconsistent frame counts")
    dist_util.setup_dist(0)
    device = dist_util.dev()
    motion = SmplData(**{key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in arrays.items()})
    with torch.no_grad():
        body = SmplModelsFK.create("smplx", len(motion), device=device).smpldata_to_smpl_output(motion)
    hand_map = get_MANO_SMPLX_vertex_ids()
    anchor_map = get_contact_anchors_info()[0]
    indices = np.concatenate([np.asarray(hand_map[hand + "_hand"])[anchor_map] for hand in ("left", "right")])
    anchors = body.vertices[:, indices]
    rotation = axis_angle_to_matrix(motion.poses_obj)
    vertices = torch.tensor(active.canonical_target_mesh.vertices, dtype=torch.float32, device=device)
    min_hand_distance = []
    object_bottom = []
    object_vertices_world = []
    table = Table(torch.tensor(active.support_corners_world[None], dtype=torch.float32, device=device))
    table_penetration = []
    for frame in range(len(motion)):
        world_vertices = vertices @ rotation[frame].T + motion.trans_obj[frame]
        object_vertices_world.append(world_vertices)
        distance = torch.cdist(anchors[frame][None], world_vertices[None]).amin()
        min_hand_distance.append(float(distance))
        object_bottom.append(float(world_vertices[:, 2].min()))
        table_penetration.append(float(table.loss_below_surface(world_vertices[None, None]).max()))
    lift = arrays["trans_obj"][:, 2] - arrays["trans_obj"][0, 2]
    touching_lift = (lift >= 0.03) & (np.asarray(min_hand_distance) <= 0.03)
    prefix_error = max(float(np.abs(arrays[key][:15] - prefix[key]).max()) for key in ["joints", "trans", "trans_obj"])
    initial_error = float(np.abs(arrays["trans_obj"][0] - active.canonical_target_pose.translation).max())
    table_z = float(active.support_corners_world[:, 2].mean())
    collision_report = evaluate_scene_collisions(
        bundle,
        body.vertices.detach().cpu().numpy(),
        torch.stack(object_vertices_world).detach().cpu().numpy(),
        subdivisions=collision_subdivisions,
    )
    report = {
        "frames": len(motion), "fps": 20,
        "finite": True, "source_prefix_frames": 15,
        "prefix_position_error_m": prefix_error,
        "initial_object_position_error_m": initial_error,
        "maximum_object_lift_m": float(lift.max()),
        "minimum_hand_object_vertex_distance_m": min(min_hand_distance),
        "lifted_frames_with_hand_within_3cm": int(touching_lift.sum()),
        "minimum_object_bottom_minus_table_height_m": min(object_bottom) - table_z,
        "maximum_object_penetration_within_table_footprint_m": max(table_penetration),
        "minimum_body_vertex_height_m": float(body.vertices[:, :, 2].min()),
        "minimum_prefix_body_vertex_height_m": float(body.vertices[:15, :, 2].min()),
        "scene_collisions": collision_report,
        "checks": {
            "prefix_preserved": prefix_error < 0.001,
            "starts_at_scene_object": initial_error < 0.001,
            "mug_lifted_with_nearby_hand": int(touching_lift.sum()) >= 5,
            "object_not_penetrating_table": max(table_penetration) <= 0.02,
            "body_not_below_floor": float(body.vertices[:, :, 2].min()) >= -0.05,
            **collision_report["checks"],
        },
        "limitations": [
            "Kinematic motion, not physical execution or validated forces",
            "Distances use hand anchors and mesh vertices, not measured contact forces",
            "HOIDiNi is conditioned on the selected object and desk; other room obstacles are not optimization constraints",
        ],
    }
    report["passed"] = all(report["checks"].values())
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if render:
        _render(output, bundle, motion.to("cpu"), np.asarray(min_hand_distance), lift)
    return report


def _render(output, bundle, motion, distances, lift):
    import bpy
    from mathutils import Matrix, Vector
    from hoidini.blender_utils.visualize_hoi_animation import AnimationSetup, visualize_hoi_animation

    active = bundle.active_interaction
    # Preview the final object trajectory, without auxiliary hand-relative
    # object hypotheses that the upstream debugging viewer also displays.
    motion.poses_obj_from_lhand = None
    motion.poses_obj_from_rhand = None
    visualize_hoi_animation([motion], object_path_or_name=str(output / "object.obj"),
                            text=active.prompt, anim_setup=AnimationSetup.MESH_ALL,
                            reset_blender=True, save=False)
    for name, node in bundle.nodes.items():
        if name == active.target_name:
            continue
        for geometry in node.visual_geometry:
            bpy.ops.wm.obj_import(filepath=str(geometry.mesh_path), forward_axis="Y", up_axis="Z")
            for obj in list(bpy.context.selected_objects):
                obj.matrix_world = Matrix((node.pose.matrix @ geometry.mesh_to_asset).tolist())
                # Keep the complete room in the blend, but open the walls for QA.
                if any(part in name.lower() for part in ("wall", "ceiling", "exterior")):
                    obj.hide_render = True
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.render.resolution_x = 960
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = 20
    scene.frame_end = len(motion) - 1
    positions = np.concatenate([motion.joints.cpu().numpy().reshape(-1, 3),
                                motion.trans_obj.cpu().numpy()])
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    center = Vector((lower + upper) / 2)
    span = max(float((upper - lower).max()), 2.0)
    bpy.ops.object.camera_add(location=center + Vector((1.2, -1.6, 0.9)) * span)
    camera = bpy.context.object
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 38
    scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=center + Vector((0, 0, 3)))
    bpy.context.object.data.energy = 700
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 5
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Room preview world")
    scene.world.color = (0.5, 0.5, 0.5)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "motion.blend"))
    best = int(np.argmax(np.where(distances <= 0.03, lift, -1)))
    for frame in sorted({0, 14, best, len(motion) - 1}):
        scene.frame_set(frame)
        scene.render.filepath = str(output / f"frame_{frame:03d}.png")
        bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--collision-subdivisions", type=int, default=2)
    args = parser.parse_args()
    report = validate_motion(
        args.output,
        render=args.render,
        collision_subdivisions=args.collision_subdivisions,
    )
    if args.render:
        # This installed bpy build segfaults during interpreter teardown.
        # All saves have returned; preserve the validation exit status and
        # bypass teardown only after successful completion of rendering.
        import os
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if report["passed"] else 1)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
