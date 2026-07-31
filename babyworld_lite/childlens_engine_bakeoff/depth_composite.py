"""Depth-compose MPFB foreground replay over the actual furnished scene."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .trace_render import make_inspection_sheet


FRAME_PATTERN = re.compile(r"overlay_(\d{4})_.*\.png$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indexed(paths: list[Path]) -> dict[int, Path]:
    result = {}
    for path in paths:
        match = FRAME_PATTERN.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def compose(
    bundle_dir: Path,
    overlay_dir: Path,
    output_dir: Path,
    *,
    fps: int = 30,
) -> dict:
    """Use the two metric Z streams to resolve every foreground pixel."""
    try:
        import OpenEXR
    except ImportError as error:
        raise RuntimeError("OpenEXR is required for metric depth composition") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    png_by_frame = _indexed(list(overlay_dir.glob("overlay_*.png")))
    exr_by_frame = {
        int(path.name.split("_")[1]): path
        for path in overlay_dir.glob("overlay_*_depth_z.exr")
    }
    if set(png_by_frame) != set(exr_by_frame):
        raise ValueError("RGBA and foreground depth frame sets differ")
    background_video = bundle_dir / "actual_furnished_background.mp4"
    background_streams = bundle_dir / "background_render_streams.h5"
    reader = imageio.get_reader(background_video)
    output_video = output_dir / "canonical_composed_silent.mp4"
    writer = imageio.get_writer(
        output_video, fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    inspection_indices = {
        0,
        113,
        226,
        262,
        338,
        369,
        451,
        563,
        675,
        716,
        788,
        900,
    }
    inspection_paths = []
    total_foreground = 0
    total_occluded = 0
    invalid_foreground_depth = 0
    started = time.perf_counter()
    with h5py.File(background_streams, "r") as h5:
        background_depth = h5["depth_m"]
        if len(png_by_frame) != len(background_depth):
            raise ValueError("foreground and background frame counts differ")
        for frame_index in sorted(png_by_frame):
            background = np.asarray(reader.get_data(frame_index), dtype=np.float32)
            foreground = np.asarray(
                Image.open(png_by_frame[frame_index]).convert("RGBA"),
                dtype=np.float32,
            )
            alpha = foreground[..., 3] / 255.0
            foreground_mask = alpha > 0.0
            foreground_depth = OpenEXR.File(
                str(exr_by_frame[frame_index])
            ).channels()["ViewLayer.Depth.Z"].pixels
            valid_depth = np.isfinite(foreground_depth) & (foreground_depth > 0)
            visible = (
                foreground_mask
                & valid_depth
                & (
                    foreground_depth
                    <= np.asarray(background_depth[frame_index], dtype=np.float32)
                    + 0.001
                )
            )
            total_foreground += int(np.count_nonzero(foreground_mask))
            total_occluded += int(np.count_nonzero(foreground_mask & ~visible))
            invalid_foreground_depth += int(
                np.count_nonzero(foreground_mask & ~valid_depth)
            )
            effective_alpha = np.where(visible, alpha, 0.0)[..., None]
            composed = (
                foreground[..., :3] * effective_alpha
                + background[..., :3] * (1.0 - effective_alpha)
            )
            rgb = np.clip(composed, 0, 255).astype(np.uint8)
            writer.append_data(rgb)
            if frame_index in inspection_indices:
                path = output_dir / f"composed_inspection_{frame_index:04d}.png"
                imageio.imwrite(path, rgb)
                inspection_paths.append(path)
    reader.close()
    writer.close()
    inspection_sheet = output_dir / "composed_inspection_sheet.png"
    make_inspection_sheet(inspection_paths, inspection_sheet)
    output_with_audio = output_dir / "canonical_composed_episode.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_video),
            "-i",
            str(bundle_dir / "speech_waveform.wav"),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-t",
            "30",
            str(output_with_audio),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    receipt = {
        "schema": "DepthCompositionReceipt.v1",
        "frames": len(png_by_frame),
        "fps": fps,
        "wall_seconds": time.perf_counter() - started,
        "camera_contract": (
            "identical EpisodeTrace camera_pose; MuJoCo and Blender right-handed "
            "Z-up with camera right/up/backward axes"
        ),
        "occlusion_rule": (
            "foreground Depth.Z <= furnished-scene metric depth + 0.001 m"
        ),
        "foreground_pixels": total_foreground,
        "occluded_foreground_pixels": total_occluded,
        "invalid_foreground_depth_pixels": invalid_foreground_depth,
        "silent_video": output_video.name,
        "silent_video_sha256": _sha256(output_video),
        "episode_video": output_with_audio.name,
        "episode_video_sha256": _sha256(output_with_audio),
        "inspection_sheet": inspection_sheet.name,
        "inspection_sheet_sha256": _sha256(inspection_sheet),
        "background_video_sha256": _sha256(background_video),
        "background_depth_sha256": _sha256(background_streams),
        "overlay_receipt_sha256": _sha256(overlay_dir / "overlay_receipt.json"),
    }
    (output_dir / "depth_composition_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt
