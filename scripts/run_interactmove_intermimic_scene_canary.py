#!/usr/bin/env python3
"""Run a robot-free SAPIEN import/render canary for an EmbodiedGen layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence
from xml.sax.saxutils import escape


EMBODIEDGEN_COMMIT = "9b333554254af196bace88c1a171a3bf047fa09c"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a robot-free EmbodiedGen SAPIEN scene canary. This checks "
            "scene import, dynamics, and RGB rendering; it is not an "
            "InterMimic or humanoid rollout."
        )
    )
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--generation-receipt", type=Path)
    parser.add_argument("--scene-bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--sim-hz", type=int, default=200)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--camera-radius-m", type=float, default=1.4)
    parser.add_argument("--camera-height-m", type=float, default=1.2)
    parser.add_argument("--target-height-m", type=float, default=0.8)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(array: Any) -> str:
    import numpy as np

    normalized = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(normalized).cast("B")).hexdigest()


def _sample_steps(frame_count: int, sim_hz: int, video_fps: int) -> list[int]:
    return [(frame * sim_hz) // video_fps for frame in range(frame_count)]


def _pose(entity: Any) -> tuple[Any, Any]:
    import numpy as np

    pose = entity.get_pose()
    return np.asarray(pose.p, dtype=np.float64), np.asarray(
        pose.q, dtype=np.float64
    )


def _velocity(entity: Any) -> tuple[Any, Any]:
    import numpy as np

    for component in entity.components:
        if hasattr(component, "linear_velocity") and hasattr(
            component, "angular_velocity"
        ):
            return np.asarray(component.linear_velocity, dtype=np.float64), np.asarray(
                component.angular_velocity, dtype=np.float64
            )
    zeros = np.zeros(3, dtype=np.float64)
    return zeros.copy(), zeros.copy()


def _capture(actors: dict[str, Any], names: Sequence[str]) -> tuple[Any, Any]:
    import numpy as np

    poses = np.empty((len(names), 7), dtype=np.float64)
    velocities = np.empty((len(names), 6), dtype=np.float64)
    for index, name in enumerate(names):
        position, quaternion_wxyz = _pose(actors[name])
        linear, angular = _velocity(actors[name])
        poses[index, :3] = position
        poses[index, 3:] = quaternion_wxyz
        velocities[index, :3] = linear
        velocities[index, 3:] = angular
    return poses, velocities


def _version(module: Any) -> str | None:
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


def _verify_scene_bundle_files(scene_root: Path, scene_bundle: dict[str, Any]) -> None:
    root = scene_root.resolve(strict=True)
    for record in scene_bundle["files"]:
        relative = Path(*record["path"].split("/"))
        candidate = root / relative
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"SceneBundle file has a symlink component: {record['path']}")
        resolved = candidate.resolve(strict=True)
        if root not in resolved.parents or not resolved.is_file():
            raise ValueError(f"SceneBundle file escapes the scene root: {record['path']}")
        if resolved.stat().st_size != record["bytes"] or _sha256(resolved) != record["sha256"]:
            raise ValueError(f"SceneBundle file hash/size mismatch: {record['path']}")


def _write_environment_urdf(
    output_dir: Path,
    scene_root: Path,
    scene_bundle: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    environment = scene_bundle["environment"]
    if environment["render_mode"] != "raster_reference_mesh":
        raise ValueError("SceneBundle does not declare a raster room render")
    reference = environment["reference_mesh_report"]
    if reference is None:
        raise ValueError("SceneBundle lacks a validated room reference mesh")
    visual_path = (scene_root / Path(*reference["path"].split("/"))).resolve(strict=True)
    visual_scale = " ".join(str(value) for value in reference["scale_xyz"])
    collision_xml: list[str] = []
    collision_meshes: list[dict[str, Any]] = []
    floor_declared = False
    for collision in environment["collision_geometry"]:
        if collision["geometry_type"] == "plane":
            floor_declared = collision["role"] == "floor" and collision["z_m"] == 0.0
            continue
        mesh = collision["mesh"]
        path = (scene_root / Path(*mesh["path"].split("/"))).resolve(strict=True)
        scale = " ".join(str(value) for value in mesh["scale_xyz"])
        collision_xml.append(
            "<collision name=\"{}\"><geometry><mesh filename=\"{}\" scale=\"{}\"/>"
            "</geometry></collision>".format(
                escape(collision["collision_id"]), escape(str(path)), escape(scale)
            )
        )
        collision_meshes.append(
            {
                "collision_id": collision["collision_id"],
                "role": collision["role"],
                "path": mesh["path"],
                "sha256": mesh["sha256"],
            }
        )
    if not floor_declared or not any(item["role"] == "walls" for item in collision_meshes):
        raise ValueError("SceneBundle lacks explicit floor and walls collision")
    urdf = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<robot name=\"nursery_room\"><link name=\"room\">"
        f"<visual name=\"room_reference\"><geometry><mesh filename=\"{escape(str(visual_path))}\" "
        f"scale=\"{escape(visual_scale)}\"/></geometry></visual>"
        + "".join(collision_xml)
        + "</link></robot>\n"
    )
    urdf_path = output_dir / "environment_runtime.urdf"
    urdf_path.write_text(urdf, encoding="utf-8")
    return urdf_path, {
        "visual_reference": {
            "path": reference["path"],
            "sha256": reference["sha256"],
        },
        "floor_collision": "SAPIEN z=0 ground plane bound to explicit SceneBundle plane",
        "collision_meshes": collision_meshes,
    }


def _contact_metrics(scene: Any) -> dict[str, Any]:
    if not hasattr(scene, "get_contacts"):
        raise RuntimeError("SAPIEN scene does not expose contact inspection")
    contacts = scene.get_contacts()
    point_count = 0
    max_penetration = 0.0
    pair_records: list[dict[str, Any]] = []
    for contact in contacts:
        body_names = []
        for body in getattr(contact, "bodies", []):
            entity = getattr(body, "entity", None)
            name = getattr(entity, "name", None) or getattr(body, "name", None)
            body_names.append("<unnamed>" if name is None else str(name))
        pair_point_count = 0
        pair_max_penetration = 0.0
        for point in getattr(contact, "points", []):
            separation = float(point.separation)
            if not math.isfinite(separation):
                raise RuntimeError("SAPIEN contact separation is non-finite")
            point_count += 1
            penetration = max(0.0, -separation)
            max_penetration = max(max_penetration, penetration)
            pair_point_count += 1
            pair_max_penetration = max(pair_max_penetration, penetration)
        pair_records.append(
            {
                "bodies": body_names,
                "contact_point_count": pair_point_count,
                "max_penetration_m": pair_max_penetration,
            }
        )
    pair_records.sort(
        key=lambda item: (-item["max_penetration_m"], item["bodies"])
    )
    return {
        "contact_pair_count": len(contacts),
        "contact_point_count": point_count,
        "max_penetration_m": max_penetration,
        "contact_pairs_by_penetration": pair_records,
    }


def _load_fixed_room(loader: Any, environment_urdf_path: Path):
    articulations, entities = loader.load_multiple(str(environment_urdf_path))
    loaded = list(articulations) + list(entities)
    if not loaded:
        raise RuntimeError("SAPIEN failed to import the explicit room geometry")
    return articulations, entities, loaded


def run(args: argparse.Namespace) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np
    import sapien

    from embodied_gen.utils.enum import LayoutInfo, Scene3DItemEnum
    from embodied_gen.utils.simulation import (
        SapienSceneManager,
        load_assets_from_layout_file,
        render_images,
    )

    layout_path = args.layout.resolve(strict=True)
    if not layout_path.is_file():
        raise ValueError("--layout must be a regular file")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("--output-dir must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_receipt = None
    generation_receipt_path = None
    if args.generation_receipt is not None:
        generation_receipt_path = args.generation_receipt.resolve(strict=True)
        generation_receipt = json.loads(
            generation_receipt_path.read_text(encoding="utf-8")
        )
        if generation_receipt.get("status") != "passed":
            raise ValueError("--generation-receipt must have status=passed")
        policy = generation_receipt.get("robot_policy", {})
        if policy.get("upstream_sim_cli_invoked") is not False:
            raise ValueError("generation receipt must prove sim_cli was omitted")
        if policy.get("robot_actor_loaded") is not False:
            raise ValueError("generation receipt must prove robot-free generation")
        layout_receipt = generation_receipt.get("layout", {})
        if layout_receipt.get("sha256") != _sha256(layout_path):
            raise ValueError("generation receipt layout hash mismatch")
    scene_bundle = None
    scene_bundle_path = None
    if args.scene_bundle is not None:
        from babyworld_lite.interactmove_intermimic.embodiedgen import (
            validate_scene_bundle,
        )

        scene_bundle_path = args.scene_bundle.resolve(strict=True)
        scene_bundle = json.loads(scene_bundle_path.read_text(encoding="utf-8"))
        validate_scene_bundle(scene_bundle)
        if scene_bundle["source"]["layout"]["sha256"] != _sha256(layout_path):
            raise ValueError("SceneBundle layout hash mismatch")
        for capability in (
            "visual_background_ready",
            "room_collision_ready",
            "complete_scene_ready",
            "interactmove_scene_input_ready",
        ):
            if scene_bundle["capabilities"].get(capability) is not True:
                raise ValueError(f"SceneBundle capability gate failed: {capability}")
        _verify_scene_bundle_files(layout_path.parent, scene_bundle)
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        raise ValueError("--duration-s must be finite and positive")
    for name in ("sim_hz", "video_fps", "width", "height"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    total_steps_float = args.duration_s * args.sim_hz
    total_steps = round(total_steps_float)
    if not math.isclose(total_steps_float, total_steps, abs_tol=1e-9):
        raise ValueError("duration-s * sim-hz must be an integer")
    frame_count_float = args.duration_s * args.video_fps
    frame_count = round(frame_count_float)
    if not math.isclose(frame_count_float, frame_count, abs_tol=1e-9):
        raise ValueError("duration-s * video-fps must be an integer")

    layout_raw = json.loads(layout_path.read_text(encoding="utf-8"))
    layout = LayoutInfo.from_dict(layout_raw)
    native_robot = layout.relation.get(Scene3DItemEnum.ROBOT.value)
    manager = SapienSceneManager(args.sim_hz, ray_tracing=False)
    environment_import = None
    room_articulations: list[Any] = []
    room_entities: list[Any] = []
    room_objects: list[Any] = []
    environment_urdf_path = None
    if scene_bundle is not None:
        environment_urdf_path, environment_import = _write_environment_urdf(
            output_dir, layout_path.parent, scene_bundle
        )
        loader = manager.scene.create_urdf_loader()
        loader.fix_root_link = True
        room_articulations, room_entities, room_objects = _load_fixed_room(
            loader, environment_urdf_path
        )
    manager.initialize_circular_cameras(
        num_cameras=1,
        radius=args.camera_radius_m,
        height=args.camera_height_m,
        target_pt=[0.0, 0.0, args.target_height_m],
        image_hw=(args.height, args.width),
        fovy_deg=75.0,
    )
    actors = load_assets_from_layout_file(manager.scene, str(layout_path), 0.004)
    if native_robot in actors:
        raise RuntimeError("native robot was unexpectedly loaded as a scene actor")
    names = sorted(actors)
    if not names:
        raise RuntimeError("scene import produced no actors")

    poses = np.empty((total_steps + 1, len(names), 7), dtype=np.float64)
    velocities = np.empty((total_steps + 1, len(names), 6), dtype=np.float64)
    poses[0], velocities[0] = _capture(actors, names)

    sample_steps = _sample_steps(frame_count, args.sim_hz, args.video_fps)
    video_path = output_dir / "scene_canary.mp4"
    writer = imageio.get_writer(
        video_path,
        fps=args.video_fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    next_frame = 0
    try:
        for step in range(total_steps + 1):
            while next_frame < frame_count and sample_steps[next_frame] == step:
                manager.scene.update_render()
                camera = manager.cameras[0]
                camera.take_picture()
                frame = np.asarray(render_images(camera, ["Color"])["Color"])
                writer.append_data(frame)
                next_frame += 1
            if step == total_steps:
                break
            manager.scene.step()
            poses[step + 1], velocities[step + 1] = _capture(actors, names)
    finally:
        writer.close()
    if next_frame != frame_count:
        raise RuntimeError(
            f"rendered {next_frame} frames but expected {frame_count}"
        )

    trajectory_path = output_dir / "rigid_body_rollout.npz"
    np.savez_compressed(
        trajectory_path,
        names=np.asarray(names, dtype="U"),
        times_s=np.arange(total_steps + 1, dtype=np.float64) / args.sim_hz,
        poses_world_wxyz=poses,
        velocities_world=velocities,
    )
    context_name = layout.relation.get(Scene3DItemEnum.CONTEXT.value)
    static_drift_m = None
    if context_name in names:
        context_index = names.index(context_name)
        static_drift_m = float(
            np.linalg.norm(
                poses[-1, context_index, :3] - poses[0, context_index, :3]
            )
        )
    max_final_linear_speed = float(
        np.linalg.norm(velocities[-1, :, :3], axis=1).max(initial=0.0)
    )
    max_final_angular_speed = float(
        np.linalg.norm(velocities[-1, :, 3:], axis=1).max(initial=0.0)
    )
    contact_metrics = _contact_metrics(manager.scene)
    settling_thresholds = {
        "max_final_linear_speed_m_s": 0.01,
        "max_final_angular_speed_rad_s": 0.01,
        "max_static_translation_drift_m": 1e-6,
        "max_penetration_m": 0.02,
    }
    settling_passed = (
        static_drift_m is not None
        and static_drift_m <= settling_thresholds["max_static_translation_drift_m"]
        and max_final_linear_speed <= settling_thresholds["max_final_linear_speed_m_s"]
        and max_final_angular_speed <= settling_thresholds["max_final_angular_speed_rad_s"]
        and contact_metrics["max_penetration_m"] <= settling_thresholds["max_penetration_m"]
        and bool(room_objects)
    )

    try:
        source_commit = subprocess.run(
            ["git", "-C", os.environ["EMBODIEDGEN_SOURCE_ROOT"], "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (KeyError, OSError, subprocess.CalledProcessError):
        source_commit = None
    receipt = {
        "schema": "InteractMoveInterMimicSceneCanaryReceipt",
        "schema_version": 1,
        "status": "passed" if settling_passed else "failed",
        "scientific_admission": "canary_only_not_stage3_settling_receipt",
        "boundaries": {
            "fresh_embodiedgen_scene_generation": (
                "passed" if generation_receipt is not None else "not_bound"
            ),
            "scene_bundle_validation": (
                "passed" if scene_bundle is not None else "not_bound"
            ),
            "scene_only_sapien_physics_render": "passed" if settling_passed else "failed",
            "interactmove_motion_generation": "not_run",
            "intermimic_execution": "not_run",
        },
        "source": {
            "layout_path": str(layout_path),
            "layout_sha256": _sha256(layout_path),
            "embodiedgen_expected_commit": EMBODIEDGEN_COMMIT,
            "embodiedgen_actual_commit": source_commit,
            "generation_receipt": (
                None
                if generation_receipt_path is None
                else {
                    "path": str(generation_receipt_path),
                    "sha256": _sha256(generation_receipt_path),
                }
            ),
            "scene_bundle": (
                None
                if scene_bundle_path is None
                else {
                    "path": str(scene_bundle_path),
                    "sha256": _sha256(scene_bundle_path),
                    "bundle_id": scene_bundle["bundle_id"],
                    "capabilities": scene_bundle["capabilities"],
                }
            ),
        },
        "simulator": {
            "name": "SAPIEN",
            "version": _version(sapien),
            "native_robot_declared": native_robot,
            "native_robot_loaded": False,
            "humanoid_loaded": False,
            "explicit_room_loaded": bool(room_objects),
            "explicit_room_articulation_count": len(room_articulations),
            "explicit_room_entity_count": len(room_entities),
        },
        "timing": {
            "duration_s": args.duration_s,
            "sim_hz": args.sim_hz,
            "physics_steps": total_steps,
            "video_fps": args.video_fps,
            "video_frames": frame_count,
            "sampling": "half_open_[0,duration)",
        },
        "actors": names,
        "observations": {
            "context_static_translation_drift_m": static_drift_m,
            "max_final_linear_speed_m_s": max_final_linear_speed,
            "max_final_angular_speed_rad_s": max_final_angular_speed,
            **contact_metrics,
        },
        "settling_check": {
            "passed": settling_passed,
            "thresholds": settling_thresholds,
            "scope": "scene_objects_only_no_humanoid",
        },
        "environment_import": environment_import,
        "physics_semantics": {
            "importer": "EmbodiedGen load_assets_from_layout_file",
            "urdf_mass_applied": False,
            "friction": "upstream clips source mu1/mu2 before SAPIEN material creation",
            "restitution": "upstream hard-coded 0.05",
            "background_physics": "explicit fixed room collision imported",
            "floor": "SAPIEN z=0 ground plane explicitly bound by SceneBundle",
            "contact_trace_recorded": False,
            "contact_summary_recorded": True,
        },
        "outputs": {
            "video": {
                "path": video_path.name,
                "bytes": video_path.stat().st_size,
                "sha256": _sha256(video_path),
            },
            "trajectory": {
                "path": trajectory_path.name,
                "bytes": trajectory_path.stat().st_size,
                "sha256": _sha256(trajectory_path),
                "semantic_arrays": {
                    "poses_world_wxyz": {
                        "dtype": poses.dtype.str,
                        "shape": list(poses.shape),
                        "sha256": _array_digest(poses),
                    },
                    "velocities_world": {
                        "dtype": velocities.dtype.str,
                        "shape": list(velocities.shape),
                        "sha256": _array_digest(velocities),
                    },
                },
            },
        },
        "limitations": [
            "no_humanoid_or_InterMimic_controller",
            "no_contact_trace_only_contact_summary",
            "raster_room_mesh_rendered_Gaussian_not_composited",
            "upstream_importer_does_not_apply_URDF_mass",
        ]
        + (
            ["fresh_prompt_generation_receipt_not_bound"]
            if generation_receipt is None
            else []
        ),
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    args = _parser().parse_args()
    receipt = run(args)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
