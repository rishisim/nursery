"""Depth-compose deterministic MPFB skin over the furnished MuJoCo replay."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .trace_render import make_inspection_sheet


FRAME_PATTERN = re.compile(r"overlay_(\d{4})\.png$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indexed(paths: list[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in paths:
        match = FRAME_PATTERN.fullmatch(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def _resolve_alpha_edge_depth(
    depth: np.ndarray, foreground_mask: np.ndarray, *, iterations: int = 4
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extend valid metric Z across Blender's antialiased alpha boundary."""
    resolved = np.asarray(depth, dtype=np.float32).copy()
    valid = (
        np.isfinite(resolved) & (resolved > 0.0) & (resolved < 1_000.0)
    )
    raw_invalid = int(np.count_nonzero(foreground_mask & ~valid))
    height, width = resolved.shape
    for _ in range(iterations):
        nearest = np.full_like(resolved, np.nan)
        for delta_y, delta_x in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            shifted = np.full_like(resolved, np.nan)
            target_y = slice(max(0, delta_y), min(height, height + delta_y))
            target_x = slice(max(0, delta_x), min(width, width + delta_x))
            source_y = slice(max(0, -delta_y), min(height, height - delta_y))
            source_x = slice(max(0, -delta_x), min(width, width - delta_x))
            shifted[target_y, target_x] = np.where(
                valid[source_y, source_x],
                resolved[source_y, source_x],
                np.nan,
            )
            nearest = np.fmin(nearest, shifted)
        fill = foreground_mask & ~valid & np.isfinite(nearest)
        resolved[fill] = nearest[fill]
        valid |= fill
    remaining = int(np.count_nonzero(foreground_mask & ~valid))
    return resolved, valid, raw_invalid, remaining


def compose(
    bundle_dir: Path,
    overlay_dir: Path,
    output_dir: Path,
    *,
    fps: int = 30,
) -> dict[str, Any]:
    """Apply frozen metric-depth occlusion without changing trace timing.

    The background was rendered from the same MuJoCo trace with both physical
    collision proxies and the simple co-articulated appearance proxies hidden.
    Only MPFB RGBA pixels whose metric Z is no farther than the furnished-scene
    depth (plus one millimetre) are admitted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_by_frame = _indexed(list(overlay_dir.glob("overlay_*.png")))
    if not png_by_frame:
        raise ValueError("no sequential MPFB overlay frames found")
    expected = set(range(len(png_by_frame)))
    if set(png_by_frame) != expected:
        raise ValueError("MPFB overlay frame indices are not contiguous from zero")

    overlay_receipt_path = overlay_dir / "overlay_receipt.json"
    overlay_receipt = json.loads(overlay_receipt_path.read_text(encoding="utf-8"))
    if overlay_receipt["render_mode"] != "all_frames":
        raise ValueError("composition requires an all-frame MPFB receipt")
    if not overlay_receipt["appearance_qualification"]["passed"]:
        raise ValueError("MPFB all-frame appearance qualification did not pass")

    foreground_depth_path = overlay_dir / "overlay_depth_m.npy"
    foreground_depth = np.load(foreground_depth_path, mmap_mode="r")
    background_video = bundle_dir / "mpfb_background_rgb.mp4"
    background_streams = bundle_dir / "render_streams.h5"
    output_video = output_dir / "baseline_rgb.mp4"
    writer = imageio.get_writer(
        output_video, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    reader = imageio.get_reader(background_video)
    inspection_indices = set(
        np.linspace(0, len(png_by_frame) - 1, min(12, len(png_by_frame)), dtype=int)
        .tolist()
    )
    inspection_paths: list[Path] = []
    total_foreground = 0
    total_visible = 0
    total_occluded = 0
    invalid_foreground_depth = 0
    resolved_foreground_depth = 0
    discarded_alpha_edge_pixels = 0
    magenta_skin_pixels = 0
    started = time.perf_counter()
    try:
        with h5py.File(background_streams, "r") as h5:
            background_depth = h5["depth_m"]
            if foreground_depth.shape != background_depth.shape:
                raise ValueError(
                    "foreground and background depth stream shapes differ: "
                    f"{foreground_depth.shape} != {background_depth.shape}"
                )
            if len(png_by_frame) != len(background_depth):
                raise ValueError("foreground and background frame counts differ")
            for frame_index in range(len(png_by_frame)):
                background = np.asarray(
                    reader.get_data(frame_index), dtype=np.float32
                )
                foreground = np.asarray(
                    Image.open(png_by_frame[frame_index]).convert("RGBA"),
                    dtype=np.float32,
                )
                alpha = foreground[..., 3] / 255.0
                foreground_mask = alpha > 0.01
                (
                    frame_foreground_depth,
                    valid_depth,
                    raw_invalid_depth,
                    unresolved_depth,
                ) = _resolve_alpha_edge_depth(
                    foreground_depth[frame_index], foreground_mask
                )
                invalid_foreground_depth += raw_invalid_depth
                resolved_foreground_depth += (
                    raw_invalid_depth - unresolved_depth
                )
                discarded_alpha_edge_pixels += unresolved_depth
                visible = (
                    foreground_mask
                    & valid_depth
                    & (
                        frame_foreground_depth
                        <= np.asarray(background_depth[frame_index], dtype=np.float32)
                        + 0.001
                    )
                )
                total_foreground += int(np.count_nonzero(foreground_mask))
                total_visible += int(np.count_nonzero(visible))
                total_occluded += int(np.count_nonzero(foreground_mask & ~visible))
                effective_alpha = np.where(visible, alpha, 0.0)[..., None]
                composed = (
                    foreground[..., :3] * effective_alpha
                    + background[..., :3] * (1.0 - effective_alpha)
                )
                rgb = np.clip(composed, 0, 255).astype(np.uint8)
                magenta = (
                    (rgb[..., 0] >= 180)
                    & (rgb[..., 1] <= 120)
                    & (rgb[..., 2] >= 150)
                    & visible
                )
                magenta_skin_pixels += int(np.count_nonzero(magenta))
                writer.append_data(rgb)
                if frame_index in inspection_indices:
                    path = output_dir / f"mpfb_inspection_{frame_index:04d}.png"
                    imageio.imwrite(path, rgb)
                    inspection_paths.append(path)
    finally:
        reader.close()
        writer.close()

    inspection_sheet = output_dir / "mpfb_inspection_sheet.png"
    make_inspection_sheet(inspection_paths, inspection_sheet)
    receipt = {
        "schema": "DeterministicMPFBDepthComposition.v1",
        "frames": len(png_by_frame),
        "fps": fps,
        "wall_seconds": time.perf_counter() - started,
        "camera_contract": (
            "identical EpisodeTrace camera_pose; MuJoCo and Blender right-handed "
            "Z-up with camera right/up/backward axes"
        ),
        "occlusion_rule": (
            "MPFB Depth.Z <= furnished-scene metric depth + 0.001 m"
        ),
        "foreground_pixels": total_foreground,
        "visible_foreground_pixels": total_visible,
        "occluded_foreground_pixels": total_occluded,
        "invalid_foreground_depth_pixels": invalid_foreground_depth,
        "resolved_alpha_edge_depth_pixels": resolved_foreground_depth,
        "discarded_unresolved_alpha_edge_pixels": discarded_alpha_edge_pixels,
        "discarded_unresolved_alpha_edge_fraction": (
            discarded_alpha_edge_pixels / total_foreground
            if total_foreground
            else 1.0
        ),
        "magenta_skin_artifact_pixels": magenta_skin_pixels,
        "collision_proxy_source": "none; group 3 and group 4 hidden in background",
        "event_timing_rule": "one input trace frame maps to one output frame",
        "video": output_video.name,
        "video_sha256": _sha256(output_video),
        "inspection_sheet": inspection_sheet.name,
        "inspection_sheet_sha256": _sha256(inspection_sheet),
        "background_video_sha256": _sha256(background_video),
        "background_depth_sha256": _sha256(background_streams),
        "foreground_depth_sha256": _sha256(foreground_depth_path),
        "overlay_receipt_sha256": _sha256(overlay_receipt_path),
    }
    receipt["passed"] = bool(
        total_visible > 0
        and discarded_alpha_edge_pixels / total_foreground <= 1e-5
        and magenta_skin_pixels == 0
    )
    receipt_path = output_dir / "mpfb_composition_qa.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
