"""Deterministically replay an embodied trace without mutating its camera."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .physics_kernel import KernelModel, _rotation_angle


def _scene_option(
    *, collision_diagnostic: bool = False, appearance: bool = True
) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    # Group 3 is the static physical authority for the right arm/hand. Group 2
    # is the co-articulated native appearance copy. MolmoSpaces and the frozen
    # authored clutter use group 4 for collision meshes. Both physical layers
    # remain hidden in authoritative RGB and are exposed only in the external
    # diagnostic view.
    option.geomgroup[2] = int(appearance)
    option.geomgroup[3] = int(collision_diagnostic)
    option.geomgroup[4] = int(collision_diagnostic)
    return option


def _mask(segmentation: np.ndarray, geom_ids: set[int]) -> np.ndarray:
    return (
        np.isin(segmentation[..., 0], tuple(geom_ids))
        & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
    )


def _project_world_point(
    point: np.ndarray,
    camera_pose: np.ndarray,
    *,
    width: int,
    height: int,
    vertical_fov_degrees: float,
) -> np.ndarray:
    rotation = camera_pose[3:12].reshape(3, 3)
    camera_point = rotation.T @ (point - camera_pose[:3])
    depth = -float(camera_point[2])
    if depth <= 0:
        return np.asarray([np.nan, np.nan])
    focal = height / (2.0 * np.tan(np.deg2rad(vertical_fov_degrees) / 2.0))
    return np.asarray(
        [
            width / 2.0 + focal * camera_point[0] / depth,
            height / 2.0 - focal * camera_point[1] / depth,
        ]
    )


def _nearest_mask_distance(pixel_xy: np.ndarray, mask: np.ndarray) -> float:
    if not np.isfinite(pixel_xy).all() or not np.any(mask):
        return float("inf")
    coordinates_yx = np.argwhere(mask)
    differences = coordinates_yx[:, ::-1] - pixel_xy
    return float(np.sqrt(np.min(np.sum(differences * differences, axis=1))))


def _contact_surface_points(kernel: KernelModel) -> list[np.ndarray]:
    """Return the two physical surface points for each hand/target contact."""
    hand_ids = set(kernel.hand_geom_ids)
    target_ids = set(kernel.target_geom_ids)
    points = []
    for contact_index in range(kernel.data.ncon):
        contact = kernel.data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if not pair.intersection(hand_ids) or not pair.intersection(target_ids):
            continue
        midpoint = np.asarray(contact.pos, dtype=np.float64)
        normal = np.asarray(contact.frame[:3], dtype=np.float64)
        half_separation = 0.5 * float(contact.dist) * normal
        points.extend((midpoint - half_separation, midpoint + half_separation))
    return points


def _project_contact_candidates(
    points: list[np.ndarray],
    camera_pose: np.ndarray,
    target_mask: np.ndarray,
    hand_mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    width: int,
    height: int,
    vertical_fov_degrees: float,
    occlusion_tolerance_m: float,
) -> dict[str, Any]:
    """Project physical surface points and retain only non-occluded ones."""
    rows = []
    rotation = camera_pose[3:12].reshape(3, 3)
    for point in points:
        projected = _project_world_point(
            point,
            camera_pose,
            width=width,
            height=height,
            vertical_fov_degrees=vertical_fov_degrees,
        )
        in_frame = bool(
            np.isfinite(projected).all()
            and 0.0 <= projected[0] < width
            and 0.0 <= projected[1] < height
        )
        camera_point = rotation.T @ (point - camera_pose[:3])
        contact_depth = -float(camera_point[2])
        rendered_depth = None
        occluded = False
        if in_frame:
            pixel_x = int(np.clip(round(float(projected[0])), 0, width - 1))
            pixel_y = int(np.clip(round(float(projected[1])), 0, height - 1))
            rendered_depth = float(depth_m[pixel_y, pixel_x])
            occluded = bool(
                rendered_depth < contact_depth - occlusion_tolerance_m
            )
        target_distance = _nearest_mask_distance(projected, target_mask)
        hand_distance = _nearest_mask_distance(projected, hand_mask)
        rows.append(
            {
                "world_xyz": point.tolist(),
                "projected_pixel_xy": projected.tolist(),
                "in_frame": in_frame,
                "contact_depth_m": contact_depth,
                "rendered_depth_m": rendered_depth,
                "occluded": occluded,
                "target_distance_px": target_distance,
                "hand_distance_px": hand_distance,
                "visible_union_distance_px": min(
                    target_distance, hand_distance
                ),
            }
        )
    visible = [
        row
        for row in rows
        if row["in_frame"]
        and not row["occluded"]
        and np.isfinite(row["visible_union_distance_px"])
    ]
    selected = min(
        visible,
        key=lambda row: row["visible_union_distance_px"],
        default=None,
    )
    return {
        "candidates": rows,
        "selected": selected,
        "in_frame": any(row["in_frame"] for row in rows),
        "occluded": bool(rows) and not visible,
    }


def render_trace(
    kernel: KernelModel,
    trace: dict[str, np.ndarray],
    output_dir: Path,
    *,
    truth_hz: int = 60,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    vertical_fov_degrees: float = 90.0,
    contact_occlusion_tolerance_m: float = 0.006,
) -> dict[str, Any]:
    """Write authoritative RGB/depth/segmentation plus an external QA view."""
    if truth_hz % fps:
        raise ValueError("truth_hz must be an integer multiple of render fps")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, data = kernel.model, kernel.data
    authoritative_option = _scene_option(
        collision_diagnostic=False, appearance=True
    )
    external_option = _scene_option(
        collision_diagnostic=True, appearance=True
    )
    renderer = mujoco.Renderer(model, width=width, height=height)
    baseline_path = output_dir / "baseline_rgb.mp4"
    external_path = output_dir / "external_qa.mp4"
    streams_path = output_dir / "render_streams.h5"
    baseline_writer = imageio.get_writer(
        baseline_path, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    external_writer = imageio.get_writer(
        external_path, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    stride = truth_hz // fps
    truth_indices = np.arange(0, len(trace["time_s"]), stride, dtype=np.int64)
    frame_count = len(truth_indices)
    target_ids = set(kernel.target_geom_ids)
    collision_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_group[geom_id]) in {3, 4}
    }
    appearance_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if (model.geom(geom_id).name or "").startswith("kernel_visual:")
    }
    clutter_visual_ids = set(kernel.clutter_visual_geom_ids)
    target_area = np.zeros(frame_count, dtype=np.float64)
    clutter_area = np.zeros(frame_count, dtype=np.float64)
    collision_proxy_pixels = 0
    skin_artifact_pixels = 0
    maximum_replay_translation_error = 0.0
    maximum_replay_rotation_error = 0.0
    projected_contact_errors: list[dict[str, Any]] = []
    phase_change_truth = set(
        (np.flatnonzero(trace["phase"][1:] != trace["phase"][:-1]) + 1).tolist()
    )
    first_contact = np.flatnonzero(trace["touch_contact_count"] > 0)
    inspection_truth = {0, len(trace["time_s"]) - 1, *phase_change_truth}
    if len(first_contact):
        inspection_truth.add(int(first_contact[0]))
    assist_changes = np.flatnonzero(
        trace["assist_active"][1:] != trace["assist_active"][:-1]
    ) + 1
    inspection_truth.update(assist_changes.tolist())
    inspection_paths: list[Path] = []
    started = time.perf_counter()
    try:
        with h5py.File(streams_path, "w") as h5:
            h5.attrs["clock"] = "episode_seconds"
            h5.attrs["camera"] = "kernel_head_camera"
            h5.attrs["camera_authority"] = "articulated head plus immutable mount"
            h5.create_dataset("truth_index", data=truth_indices)
            h5.create_dataset("time_s", data=trace["time_s"][truth_indices])
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
            for frame_index, truth_index in enumerate(truth_indices):
                data.qpos[:] = trace["qpos"][truth_index]
                data.qvel[:] = trace["qvel"][truth_index]
                data.time = float(trace["time_s"][truth_index])
                mujoco.mj_forward(model, data)
                derived_camera = np.concatenate(
                    [data.cam_xpos[kernel.camera_id], data.cam_xmat[kernel.camera_id]]
                )
                recorded_camera = trace["camera_pose"][truth_index]
                maximum_replay_translation_error = max(
                    maximum_replay_translation_error,
                    float(np.linalg.norm(derived_camera[:3] - recorded_camera[:3])),
                )
                maximum_replay_rotation_error = max(
                    maximum_replay_rotation_error,
                    _rotation_angle(
                        derived_camera[3:12].reshape(3, 3),
                        recorded_camera[3:12].reshape(3, 3),
                    ),
                )

                renderer.update_scene(
                    data,
                    camera="kernel_head_camera",
                    scene_option=authoritative_option,
                )
                rgb = renderer.render().copy()
                baseline_writer.append_data(rgb)
                renderer.update_scene(
                    data,
                    camera="kernel_external_qa",
                    scene_option=external_option,
                )
                external_writer.append_data(renderer.render().copy())

                renderer.enable_depth_rendering()
                renderer.update_scene(
                    data,
                    camera="kernel_head_camera",
                    scene_option=authoritative_option,
                )
                depth_m = renderer.render().copy()
                depth_dataset[frame_index] = depth_m.astype(np.float16)
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                renderer.update_scene(
                    data,
                    camera="kernel_head_camera",
                    scene_option=authoritative_option,
                )
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                segmentation_dataset[frame_index] = segmentation
                target_mask = _mask(segmentation, target_ids)
                appearance_mask = _mask(segmentation, appearance_ids)
                collision_mask = _mask(segmentation, collision_ids)
                clutter_mask = _mask(segmentation, clutter_visual_ids)
                target_area[frame_index] = float(target_mask.mean())
                clutter_area[frame_index] = float(clutter_mask.mean())
                collision_proxy_pixels += int(collision_mask.sum())
                magenta = (
                    (rgb[..., 0] >= 180)
                    & (rgb[..., 1] <= 120)
                    & (rgb[..., 2] >= 150)
                )
                skin_artifact_pixels += int((magenta & appearance_mask).sum())

                contact_position = trace["touch_contact_position"][truth_index]
                if np.isfinite(contact_position).all():
                    surface_points = _contact_surface_points(kernel)
                    if not surface_points:
                        surface_points = [contact_position]
                    projection = _project_contact_candidates(
                        surface_points,
                        recorded_camera,
                        target_mask,
                        appearance_mask,
                        depth_m,
                        width=width,
                        height=height,
                        vertical_fov_degrees=vertical_fov_degrees,
                        occlusion_tolerance_m=contact_occlusion_tolerance_m,
                    )
                    selected = projection["selected"]
                    projected_contact_errors.append(
                        {
                            "truth_index": int(truth_index),
                            "time_s": float(trace["time_s"][truth_index]),
                            "in_frame": projection["in_frame"],
                            "occluded": projection["occluded"],
                            "surface_candidates": projection["candidates"],
                            "projected_pixel_xy": (
                                selected["projected_pixel_xy"]
                                if selected
                                else None
                            ),
                            "target_distance_px": (
                                selected["target_distance_px"]
                                if selected
                                else None
                            ),
                            "hand_distance_px": (
                                selected["hand_distance_px"]
                                if selected
                                else None
                            ),
                            "visible_union_distance_px": (
                                selected["visible_union_distance_px"]
                                if selected
                                else None
                            ),
                        }
                    )
                if any(abs(int(truth_index) - item) <= stride for item in inspection_truth):
                    path = output_dir / f"inspection_{frame_index:04d}.png"
                    imageio.imwrite(path, rgb)
                    inspection_paths.append(path)
            h5.create_dataset("target_area_fraction", data=target_area)
            h5.create_dataset("clutter_area_fraction", data=clutter_area)
    finally:
        baseline_writer.close()
        external_writer.close()
        renderer.close()
    sheet_path = output_dir / "inspection_sheet.png"
    make_inspection_sheet(inspection_paths, sheet_path)
    finite_contact_errors = [
        row["visible_union_distance_px"]
        for row in projected_contact_errors
        if row["in_frame"]
        and not row["occluded"]
        and row["visible_union_distance_px"] is not None
        and np.isfinite(row["visible_union_distance_px"])
    ]
    contact_distances_m = np.abs(
        trace["touch_minimum_distance_m"][truth_indices]
    )
    contact_distances_m = contact_distances_m[
        np.isfinite(contact_distances_m)
    ]
    first_contact_truth = (
        int(first_contact[0]) if len(first_contact) else None
    )
    release_changes = np.flatnonzero(
        trace["assist_active"][:-1] & ~trace["assist_active"][1:]
    ) + 1
    release_truth = int(release_changes[0]) if len(release_changes) else None

    def event_target_visible(truth_index: int | None) -> bool:
        if truth_index is None:
            return False
        frame_index = int(np.argmin(np.abs(truth_indices - truth_index)))
        return bool(target_area[frame_index] > 0)
    return {
        "schema": "EmbodiedRenderQA",
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "wall_seconds": time.perf_counter() - started,
        "minimum_target_area_fraction": float(target_area.min()),
        "maximum_target_area_fraction": float(target_area.max()),
        "visible_frame_fraction": float(np.mean(target_area > 0)),
        "authored_clutter_visual_count": len(clutter_visual_ids),
        "clutter_visible_frame_fraction": float(np.mean(clutter_area > 0)),
        "maximum_clutter_area_fraction": float(clutter_area.max()),
        "collision_proxy_pixels": collision_proxy_pixels,
        "skin_artifact_pixels": skin_artifact_pixels,
        "maximum_replay_camera_translation_error_m": maximum_replay_translation_error,
        "maximum_replay_camera_rotation_error_rad": maximum_replay_rotation_error,
        "projected_contact_observations": len(projected_contact_errors),
        "projected_visible_contact_observations": len(finite_contact_errors),
        "projected_occluded_contact_observations": sum(
            row["occluded"] for row in projected_contact_errors
        ),
        "maximum_projected_contact_error_px": (
            max(finite_contact_errors) if finite_contact_errors else None
        ),
        "maximum_3d_contact_surface_distance_m": (
            float(contact_distances_m.max())
            if len(contact_distances_m)
            else None
        ),
        "contact_projection_details": projected_contact_errors,
        "contact_projection_method": (
            "MuJoCo hand/target contact surface endpoints; depth-occluded "
            "points excluded before applying unchanged spatial/pixel gates"
        ),
        "contact_occlusion_tolerance_m": contact_occlusion_tolerance_m,
        "rgb_truth_contact_frame_offset": 0,
        "rgb_truth_release_frame_offset": 0,
        "contact_event_target_visible": event_target_visible(first_contact_truth),
        "release_event_target_visible": event_target_visible(release_truth),
        "video": baseline_path.name,
        "external_qa_video": external_path.name,
        "streams": streams_path.name,
        "inspection_sheet": sheet_path.name,
        "appearance_status": (
            "authoritative deterministic native MIMo co-articulated appearance "
            "layer; physical collision layer remains enabled and hidden"
        ),
        "native_appearance_geom_group": 2,
        "physical_collision_geom_group": 3,
        "native_material": "skin",
    }


def make_inspection_sheet(frame_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in sorted(set(frame_paths))]
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
    for index, (source, path) in enumerate(zip(images, sorted(set(frame_paths)))):
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(source.resize((thumb_width, thumb_height)), (x, y))
        draw.text((x + 6, y + thumb_height + 5), path.stem, fill="black")
    sheet.save(output_path)
