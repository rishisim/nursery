"""Bounded Blender physics adapter for ChildLens media-grounding readiness.

The adapter emits engineering evidence only.  It never launches cue-lift arms.
All clocks are integer ticks, contact is computed from evaluated primitive
collision geometry, and proprioception/IMU are derived from evaluated poses.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class BlenderEpisodeSpec:
    seed: int = 2026072501
    clock_hz: int = 24_000
    physics_hz: int = 240
    render_fps: int = 24
    duration_seconds: float = 2.0
    width: int = 160
    height: int = 120

    def __post_init__(self) -> None:
        if self.clock_hz % self.physics_hz or self.clock_hz % self.render_fps:
            raise ValueError("physics and render rates must divide the master clock")
        if self.physics_hz % self.render_fps:
            raise ValueError("physics_hz must be divisible by render_fps")

    @property
    def physics_steps(self) -> int:
        return round(self.duration_seconds * self.physics_hz)

    @property
    def ticks_per_physics_step(self) -> int:
        return self.clock_hz // self.physics_hz


def sphere_box_signed_distance(
    sphere_center: Sequence[float],
    radius: float,
    box_center: Sequence[float],
    half_extents: Sequence[float],
    box_quaternion_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Exact signed separation for an oriented sphere/box pair.

    Negative separation means penetration.  The closest point is on/in the box;
    this bounded fixture deliberately uses axis-aligned evaluated collision
    geometry so the result is exact rather than a height or time proxy.
    """
    world_delta = [float(sphere_center[i]) - float(box_center[i]) for i in range(3)]
    w, x, y, z = (float(v) for v in box_quaternion_wxyz)
    # Rotate world delta into box coordinates with the quaternion conjugate.
    rotation = (
        (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
        (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
        (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)),
    )
    delta = [sum(rotation[i][j] * world_delta[j] for j in range(3)) for i in range(3)]
    closest_local = [
        min(max(delta[i], -float(half_extents[i])), float(half_extents[i]))
        for i in range(3)
    ]
    offset = [delta[i] - closest_local[i] for i in range(3)]
    outside = math.sqrt(sum(v * v for v in offset))
    if outside > 0:
        local_normal = [v / outside for v in offset]
        signed = outside - radius
    else:
        face_gaps = [float(half_extents[i]) - abs(delta[i]) for i in range(3)]
        axis = min(range(3), key=face_gaps.__getitem__)
        local_normal = [0.0, 0.0, 0.0]
        local_normal[axis] = 1.0 if delta[axis] >= 0 else -1.0
        signed = -(radius + face_gaps[axis])
    # Convert box-local point/normal back to world coordinates (R, not R^-1).
    point = [
        float(box_center[i]) + sum(rotation[j][i] * closest_local[j] for j in range(3))
        for i in range(3)
    ]
    normal = [sum(rotation[j][i] * local_normal[j] for j in range(3)) for i in range(3)]
    return {
        "active": signed <= 0.0,
        "signed_separation_m": signed,
        "point_world_m": point,
        "normal_box_to_sphere": normal,
        "method": "exact_oriented_sphere_box_sdf",
    }


def unwrap_angles(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    result = [float(values[0])]
    for value in values[1:]:
        candidate = float(value)
        while candidate - result[-1] > math.pi:
            candidate -= 2 * math.pi
        while candidate - result[-1] < -math.pi:
            candidate += 2 * math.pi
        result.append(candidate)
    return result


def finite_difference(values: Sequence[Sequence[float]], dt: float) -> list[list[float]]:
    if len(values) < 3 or dt <= 0:
        raise ValueError("at least three samples and positive dt are required")
    output: list[list[float]] = []
    for index in range(len(values)):
        lo, hi = ((0, 1) if index == 0 else
                  ((len(values) - 2, len(values) - 1)
                   if index == len(values) - 1 else (index - 1, index + 1)))
        scale = 1.0 / ((hi - lo) * dt)
        output.append([(values[hi][j] - values[lo][j]) * scale
                       for j in range(len(values[index]))])
    return output


def derive_pose_signals(samples: list[dict[str, Any]], dt: float) -> dict[str, Any]:
    positions = [row["position_m"] for row in samples]
    rotations = unwrap_angles([row["joint_angle_rad"] for row in samples])
    velocity = finite_difference(positions, dt)
    acceleration = finite_difference(velocity, dt)
    qdot = [row[0] for row in finite_difference([[q] for q in rotations], dt)]
    angular_velocity = [[0.0, 0.0, value] for value in qdot]
    return {
        "joint_position_rad": rotations,
        "joint_velocity_rad_s": qdot,
        "linear_acceleration_world_m_s2": acceleration,
        "angular_velocity_world_rad_s": angular_velocity,
        "method": {
            "joint": "evaluated_relative_transform_z_angle_unwrapped",
            "velocity": "timestamped_central_difference_one_sided_boundaries",
            "smoothing": "none",
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def run_blender_episode(output_dir: Path, spec: BlenderEpisodeSpec) -> Path:
    """Run inside Blender and emit state, contact fixtures, RGB/depth/seg, MP4."""
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = spec.width
    scene.render.resolution_y = spec.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = spec.render_fps
    scene.frame_start = 1
    scene.frame_end = spec.physics_steps
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("bounded_world")
    scene.world = bpy.data.worlds.new("childlens_evidence_world")
    scene.world.color = (0.04, 0.04, 0.04)

    def cube(name: str, location: tuple[float, float, float],
             scale: tuple[float, float, float], passive: bool) -> Any:
        bpy.ops.mesh.primitive_cube_add(location=location)
        obj = bpy.context.object
        obj.name = name
        obj.scale = scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = "PASSIVE" if passive else "ACTIVE"
        obj.rigid_body.collision_shape = "BOX"
        obj.rigid_body.use_deactivation = False
        return obj

    base = cube("articulated_base", (0, 0, 1.2), (0.12, 0.12, 0.12), True)
    link = cube("articulated_link", (0.0, -0.65, 1.2), (0.08, 0.65, 0.08), False)
    link.rotation_euler.z = -0.65
    link.rigid_body.mass = 0.8
    # A bounded commanded kinematic articulation is used because Blender's
    # public Python API doesn't expose stable per-step motor impulses.  The
    # rigid body remains in Bullet and constrained; measured q comes only from
    # the evaluated transform, never from these command targets.
    link.rigid_body.kinematic = True
    link.rotation_euler.z = -0.65
    link.keyframe_insert(data_path="rotation_euler", frame=1)
    link.rotation_euler.z = 0.65
    link.keyframe_insert(data_path="rotation_euler", frame=spec.physics_steps)
    for curve in link.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 1.2))
    hinge = bpy.context.object
    hinge.name = "evaluated_hinge"
    bpy.ops.rigidbody.constraint_add()
    constraint = hinge.rigid_body_constraint
    constraint.type = "HINGE"
    constraint.object1 = base
    constraint.object2 = link
    constraint.use_motor_ang = True
    constraint.motor_ang_target_velocity = 1.4
    constraint.motor_ang_max_impulse = 30.0

    contact_sphere_center = (0.42, -0.44, 1.2)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16,
                                       radius=0.13, location=contact_sphere_center)
    target = bpy.context.object
    target.name = "contact_target"
    bpy.ops.rigidbody.object_add()
    target.rigid_body.type = "PASSIVE"
    target.rigid_body.collision_shape = "SPHERE"
    target.pass_index = 2
    link.pass_index = 1
    base.pass_index = 3
    link.color = (1.0, 0.0, 0.0, 1.0)
    target.color = (0.0, 1.0, 0.0, 1.0)
    base.color = (0.0, 0.0, 1.0, 1.0)

    bpy.ops.object.light_add(type="AREA", location=(1.5, -2.0, 3.5))
    bpy.context.object.data.energy = 850
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 4
    bpy.ops.object.camera_add(location=(2.7, -4.2, 2.5))
    camera = bpy.context.object
    scene.camera = camera
    direction = Vector((0, -0.3, 1.15)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_object_index = True
    state: list[dict[str, Any]] = []
    render_every = spec.physics_hz // spec.render_fps
    for step in range(spec.physics_steps):
        # Blender frames are the physics ticks for the evidence bake.
        scene.render.fps = spec.physics_hz
        scene.frame_set(step + 1)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_link = link.evaluated_get(depsgraph)
        evaluated_base = base.evaluated_get(depsgraph)
        matrix = evaluated_link.matrix_world.copy()
        link_center = list(matrix.translation)
        # IMU is mounted near the distal end, so rotation produces physical
        # translational acceleration as well as angular velocity.
        position = list(matrix @ Vector((0.0, -0.5, 0.0)))
        relative = evaluated_base.matrix_world.inverted() @ matrix
        angle = float(relative.to_euler("XYZ").z)
        contact = sphere_box_signed_distance(
            contact_sphere_center, 0.13, link_center, (0.08, 0.65, 0.08),
            tuple(float(x) for x in matrix.to_quaternion()),
        )
        row = {
            "tick": step * spec.ticks_per_physics_step,
            "timestamp_seconds": step / spec.physics_hz,
            "position_m": [float(x) for x in position],
            "quaternion_wxyz": [float(x) for x in matrix.to_quaternion()],
            "joint_angle_rad": angle,
            "contact": contact,
        }
        state.append(row)
        if step % render_every == 0:
            render_index = step // render_every
            scene.render.filepath = str(frames_dir / f"multilayer_{render_index:04d}.exr")
            scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
            bpy.ops.render.render(write_still=True)
            scene.render.filepath = str(frames_dir / f"rgb_{render_index:04d}.png")
            scene.render.image_settings.file_format = "PNG"
            bpy.ops.render.render(write_still=True)
            # An independent object-color render makes segmentation directly
            # decodable instead of relying on self-declared metadata.
            scene.render.engine = "BLENDER_WORKBENCH"
            scene.display.shading.light = "FLAT"
            scene.display.shading.color_type = "OBJECT"
            scene.display.shading.show_shadows = False
            scene.render.filepath = str(frames_dir / f"segmentation_{render_index:04d}.png")
            bpy.ops.render.render(write_still=True)
            scene.render.engine = "BLENDER_EEVEE_NEXT"

    signals = derive_pose_signals(state, 1.0 / spec.physics_hz)
    positive = any(row["contact"]["active"] for row in state)
    negative_fixture = sphere_box_signed_distance((3, 3, 3), 0.13, (0, 0, 0), (0.08, 0.65, 0.08))
    static_samples = [{"position_m": [0.0, 0.0, 0.0], "joint_angle_rad": 0.0}
                      for _ in range(5)]
    static = derive_pose_signals(static_samples, 1.0 / spec.physics_hz)
    state_path = output_dir / "physical_state.json"
    action_commands = [
        {"tick": 0, "kind": "hinge_angle_command", "target_rad": -0.65},
        {"tick": (spec.physics_steps - 1) * spec.ticks_per_physics_step,
         "kind": "hinge_angle_command", "target_rad": 0.65},
    ]
    state_path.write_text(json.dumps({"spec": asdict(spec),
                                      "action_commands": action_commands,
                                      "samples": state, "signals": signals},
                                     indent=2) + "\n")
    ffmpeg = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(spec.render_fps),
         "-i", str(frames_dir / "rgb_%04d.png"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(output_dir / "episode.mp4")],
        capture_output=True, text=True, check=False,
    )
    controls = {
        "positive_contact_observed": positive,
        "negative_no_contact": not negative_fixture["active"],
        "joint_changed_rad": max(signals["joint_position_rad"]) - min(signals["joint_position_rad"]),
        "dynamic_gyro_peak_rad_s": max(abs(v[2]) for v in signals["angular_velocity_world_rad_s"]),
        "dynamic_acceleration_peak_m_s2": max(abs(x) for v in signals["linear_acceleration_world_m_s2"] for x in v),
        "static_joint_velocity_peak_rad_s": max(abs(x) for x in static["joint_velocity_rad_s"]),
        "static_gyro_peak_rad_s": max(abs(x) for v in static["angular_velocity_world_rad_s"] for x in v),
        "static_acceleration_peak_m_s2": max(abs(x) for v in static["linear_acceleration_world_m_s2"] for x in v),
    }
    controls["passed"] = (
        controls["positive_contact_observed"]
        and controls["negative_no_contact"]
        and controls["joint_changed_rad"] > 0.5
        and controls["dynamic_gyro_peak_rad_s"] > 0.1
        and controls["dynamic_acceleration_peak_m_s2"] > 0.01
        and controls["static_joint_velocity_peak_rad_s"] < 1e-9
        and controls["static_gyro_peak_rad_s"] < 1e-9
        and controls["static_acceleration_peak_m_s2"] < 1e-9
    )
    artifacts = [state_path, output_dir / "episode.mp4", *sorted(frames_dir.glob("*"))]
    receipt = {
        "schema_version": "childlens-blender-adaptive-repair-1.0.0",
        "purpose": "environment readiness only; cue-lift arms not run",
        "engine": {"name": "Blender", "version": bpy.app.version_string},
        "clock": {"master_hz": spec.clock_hz, "physics_hz": spec.physics_hz,
                  "render_fps": spec.render_fps},
        "contact_method": "exact evaluated oriented primitive sphere-box signed distance",
        "controls": controls,
        "channels": {
            "rgb_png": len(list(frames_dir.glob("rgb_*.png"))),
            "depth_multilayer_exr": len(list(frames_dir.glob("multilayer_*.exr"))),
            "object_color_segmentation_png": len(list(frames_dir.glob("segmentation_*.png"))),
            "physical_samples": len(state),
            "action_commands": len(action_commands),
            "playable_mp4": ffmpeg.returncode == 0 and (output_dir / "episode.mp4").stat().st_size > 0,
        },
        "artifact_hashes": {path.relative_to(output_dir).as_posix(): _sha256(path)
                            for path in artifacts if path.is_file()},
    }
    receipt_path = output_dir / "blender_replay_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if not controls["passed"]:
        raise RuntimeError(f"Blender physical controls failed: {controls}")
    return receipt_path


def main() -> int:
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    # Blender keeps its own flags in sys.argv and passes script flags after --.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    receipt = run_blender_episode(args.output, BlenderEpisodeSpec(duration_seconds=args.duration))
    print(receipt)
    return 0


if __name__ == "__main__":
    import os
    import traceback
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        # Blender returns zero for an uncaught Python exception; force a
        # fail-closed process status for doctors and automation.
        os._exit(1)
