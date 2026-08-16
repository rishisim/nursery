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
        "status": "completed",
        "scientific_admission": "canary_only_not_stage3_settling_receipt",
        "source": {
            "layout_path": str(layout_path),
            "layout_sha256": _sha256(layout_path),
            "embodiedgen_expected_commit": EMBODIEDGEN_COMMIT,
            "embodiedgen_actual_commit": source_commit,
        },
        "simulator": {
            "name": "SAPIEN",
            "version": _version(sapien),
            "native_robot_declared": native_robot,
            "native_robot_loaded": False,
            "humanoid_loaded": False,
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
        },
        "physics_semantics": {
            "importer": "EmbodiedGen load_assets_from_layout_file",
            "urdf_mass_applied": False,
            "friction": "upstream clips source mu1/mu2 before SAPIEN material creation",
            "restitution": "upstream hard-coded 0.05",
            "background_physics": "not_loaded",
            "floor": "SAPIEN z=0 ground plane",
            "contact_trace_recorded": False,
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
            "no_contact_trace",
            "no_Gaussian_background_compositing",
            "upstream_importer_does_not_apply_URDF_mass",
            "not_a_fresh_prompt_generated_scene",
        ],
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
