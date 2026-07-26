"""Render a deterministic physics trace in the actual furnished MuJoCo scene."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .physics_kernel import KernelModel


def _scene_option(kernel: KernelModel, *, show_collision_hand: bool) -> mujoco.MjvOption:
    model = kernel.model
    option = mujoco.MjvOption()
    kernel_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (model.body(int(model.geom_bodyid[geom_id])).name or "").startswith(
            "kernel_"
        )
    ]
    model.geom_group[kernel_geom_ids] = 5
    if show_collision_hand:
        model.geom_group[list(kernel.hand_geom_ids)] = 0
        model.geom_group[kernel.target_geom_id] = 0
        model.geom_group[model.geom("kernel_support_geom").id] = 0
    option.geomgroup[5] = 0
    return option


def render_trace(
    kernel: KernelModel,
    trace: dict[str, np.ndarray],
    output_dir: Path,
    *,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    show_collision_hand: bool = True,
) -> dict[str, Any]:
    """Render RGB plus synchronized metric depth and segmentation to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model, data = kernel.model, kernel.data
    option = _scene_option(kernel, show_collision_hand=show_collision_hand)
    renderer = mujoco.Renderer(model, width=width, height=height)
    prefix = (
        "actual_furnished_native_diagnostic"
        if show_collision_hand
        else "actual_furnished_background"
    )
    video_path = output_dir / f"{prefix}.mp4"
    streams_path = output_dir / (
        "render_streams.h5"
        if show_collision_hand
        else "background_render_streams.h5"
    )
    writer = imageio.get_writer(
        video_path, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    frame_count = len(trace["time_s"])
    target_area = np.zeros(frame_count, dtype=np.float64)
    phase_change_frames = (
        np.flatnonzero(trace["phase"][1:] != trace["phase"][:-1]) + 1
    )
    contact_frames = np.flatnonzero(trace["touch_contact_count"] > 0)
    grasp_change_frames = (
        np.flatnonzero(trace["grasp_active"][1:] != trace["grasp_active"][:-1])
        + 1
    )
    inspection_frames = {
        0,
        frame_count - 1,
        *phase_change_frames.tolist(),
        *contact_frames[:1].tolist(),
        *grasp_change_frames.tolist(),
    }
    inspection_paths: list[Path] = []
    started = time.perf_counter()
    with h5py.File(streams_path, "w") as h5:
        h5.attrs["clock"] = "episode_seconds"
        h5.attrs["target_geom_id"] = int(kernel.target_geom_id)
        h5.create_dataset("time_s", data=trace["time_s"])
        depth_dataset = h5.create_dataset(
            "depth_m",
            shape=(frame_count, height, width),
            dtype=np.float16,
            chunks=(1, height, width),
            compression="gzip",
            compression_opts=2,
        )
        segmentation_dataset = h5.create_dataset(
            "segmentation",
            shape=(frame_count, height, width, 2),
            dtype=np.int32,
            chunks=(1, height, width, 2),
            compression="gzip",
            compression_opts=2,
        )
        for frame_index in range(frame_count):
            data.qpos[:] = trace["qpos"][frame_index]
            data.qvel[:] = trace["qvel"][frame_index]
            data.time = float(trace["time_s"][frame_index])
            mujoco.mj_forward(model, data)
            renderer.update_scene(
                data, camera="kernel_chest_camera", scene_option=option
            )
            rgb = renderer.render().copy()
            writer.append_data(rgb)
            if frame_index in inspection_frames:
                path = output_dir / f"inspection_{frame_index:04d}.png"
                imageio.imwrite(path, rgb)
                inspection_paths.append(path)
            renderer.enable_depth_rendering()
            renderer.update_scene(
                data, camera="kernel_chest_camera", scene_option=option
            )
            depth_dataset[frame_index] = renderer.render().astype(np.float16)
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(
                data, camera="kernel_chest_camera", scene_option=option
            )
            segmentation = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            segmentation_dataset[frame_index] = segmentation
            target_mask = (
                (segmentation[..., 0] == kernel.target_geom_id)
                & (
                    segmentation[..., 1]
                    == int(mujoco.mjtObj.mjOBJ_GEOM)
                )
            )
            target_area[frame_index] = float(target_mask.mean())
            if show_collision_hand and frame_index in inspection_frames:
                imageio.imwrite(
                    output_dir / f"replay_segmentation_{frame_index:04d}.png",
                    (target_mask.astype(np.uint8) * 255),
                )
        h5.create_dataset("target_area_fraction", data=target_area)
    writer.close()
    renderer.close()
    sheet_path = output_dir / (
        "inspection_sheet.png"
        if show_collision_hand
        else "background_inspection_sheet.png"
    )
    make_inspection_sheet(inspection_paths, sheet_path)
    return {
        "frames": frame_count,
        "width": width,
        "height": height,
        "wall_seconds": time.perf_counter() - started,
        "minimum_target_area_fraction": float(target_area.min()),
        "maximum_target_area_fraction": float(target_area.max()),
        "visible_frame_fraction": float(np.mean(target_area > 0)),
        "video": video_path.name,
        "streams": streams_path.name,
        "inspection_sheet": sheet_path.name,
        "appearance_status": (
            "actual_furnished_scene_with_native_collision_hand_diagnostic_only"
            if show_collision_hand
            else "actual_furnished_background_layer"
        ),
    }


def make_inspection_sheet(frame_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in sorted(frame_paths)]
    if not images:
        raise ValueError("at least one inspection frame is required")
    thumb_width = 320
    thumb_height = round(images[0].height * thumb_width / images[0].width)
    label_height = 28
    columns = 3
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (image, path) in enumerate(zip(images, sorted(frame_paths))):
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(image.resize((thumb_width, thumb_height)), (x, y))
        draw.text((x + 6, y + thumb_height + 5), path.stem, fill="black")
    sheet.save(output_path)
