"""Render a bounded pinned MolmoSpaces iTHOR scene with classic MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(scene_path: Path, output_dir: Path, *, seed: int = 20260725) -> dict:
    del seed  # The stock scene has a frozen initial state and no randomized policy.
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-0.3, -0.5, 0.9]
    camera.elevation = -5
    camera.distance = 2.0
    frames: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    camera_state: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    segmentation_frames: list[np.ndarray] = []
    sample_indices = {0, 60, 119}
    started = time.perf_counter()
    for frame_index in range(120):
        for _ in range(2):
            mujoco.mj_step(model, data)
        camera.azimuth = -10 + 55 * frame_index / 119
        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        camera_state.append(
            np.asarray([*camera.lookat, camera.azimuth, camera.elevation, camera.distance])
        )
        if frame_index in sample_indices:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera)
            depth_frames.append(renderer.render().copy())
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera)
            segmentation_frames.append(renderer.render().copy())
            renderer.disable_segmentation_rendering()
    elapsed = time.perf_counter() - started
    renderer.close()

    video_path = output_dir / "stock_molmospaces_ithor_floorplan1_classic.mp4"
    imageio.mimwrite(video_path, frames, fps=30, codec="libx264", quality=8)
    telemetry_path = output_dir / "stock_molmospaces_telemetry.npz"
    np.savez_compressed(
        telemetry_path,
        frame=np.arange(120, dtype=np.int32),
        time_s=np.arange(1, 121, dtype=np.float64) * model.opt.timestep * 2,
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        camera=np.asarray(camera_state),
        depth=np.asarray(depth_frames),
        segmentation=np.asarray(segmentation_frames),
        keyframe=np.asarray(sorted(sample_indices), dtype=np.int32),
    )
    for index, frame_index in enumerate(sorted(sample_indices)):
        depth = depth_frames[index]
        normalized = (255 * (depth - depth.min()) / max(np.ptp(depth), 1e-9)).astype(np.uint8)
        imageio.imwrite(output_dir / f"depth_{frame_index:03d}.png", normalized)
        segmentation = segmentation_frames[index][..., 0]
        colored = ((segmentation.astype(np.int64) * 2654435761) % 255).astype(np.uint8)
        imageio.imwrite(output_dir / f"segmentation_{frame_index:03d}.png", colored)

    receipt = {
        "stage": "stock_molmospaces",
        "renderer": "classic_mujoco",
        "scene": "iTHOR FloorPlan1",
        "scene_version": "20251217_with_occupancy",
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "model_counts": {
            "bodies": model.nbody,
            "geometries": model.ngeom,
            "meshes": model.nmesh,
            "textures": model.ntex,
            "joints": model.njnt,
        },
        "frames": len(frames),
        "simulation_seconds": 120 * model.opt.timestep * 2,
        "wall_seconds": elapsed,
        "frames_per_wall_second": len(frames) / elapsed,
        "depth_keyframes": len(depth_frames),
        "segmentation_keyframes": len(segmentation_frames),
        "video": {"name": video_path.name, "sha256": _sha256(video_path)},
        "telemetry": {"name": telemetry_path.name, "sha256": _sha256(telemetry_path)},
    }
    (output_dir / "stock_molmospaces_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.scene_path, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
