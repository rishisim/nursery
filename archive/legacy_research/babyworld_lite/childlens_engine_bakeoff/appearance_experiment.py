"""Prepare, execute, composite, and qualify controlled appearance windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .bundle import sha256_file, write_json


GEOM_OBJECT_TYPE = 5
SKELETON_CHAINS = (
    ("kernel_right_upper_arm", "kernel_right_lower_arm", "kernel_right_hand"),
    (
        "kernel_right_hand",
        "kernel_right_ffknuckle",
        "kernel_right_ffmiddle",
        "kernel_right_ffdistal",
    ),
    (
        "kernel_right_hand",
        "kernel_right_mfknuckle",
        "kernel_right_mfmiddle",
        "kernel_right_mfdistal",
    ),
    (
        "kernel_right_hand",
        "kernel_right_rfknuckle",
        "kernel_right_rfmiddle",
        "kernel_right_rfdistal",
    ),
    (
        "kernel_right_hand",
        "kernel_right_lfmetacarpal",
        "kernel_right_lfknuckle",
        "kernel_right_lfmiddle",
        "kernel_right_lfdistal",
    ),
    (
        "kernel_right_hand",
        "kernel_right_thbase",
        "kernel_right_thhub",
        "kernel_right_thdistal",
    ),
)


def _assert_ignored(repo_root: Path, output_dir: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(output_dir.resolve())],
        cwd=repo_root,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"output directory is not ignored: {output_dir}")


def _read_video(path: Path) -> np.ndarray:
    import imageio.v3 as iio

    frames = iio.imread(path, plugin="FFMPEG")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected RGB video at {path}, got {frames.shape}")
    return np.asarray(frames, dtype=np.uint8)


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        pixelformat="yuv420p",
        ffmpeg_params=["-an"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _write_lossless_rgb_video(path: Path, frames: np.ndarray, fps: int) -> None:
    """Write an RGB H.264 stream without chroma subsampling or pixel loss."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264rgb",
        macro_block_size=None,
        pixelformat="bgr24",
        ffmpeg_params=["-crf", "0", "-preset", "medium", "-an"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _stable_color(identifier: int) -> np.ndarray:
    digest = hashlib.sha256(str(identifier).encode()).digest()
    return np.asarray([64 + digest[index] % 192 for index in range(3)], dtype=np.uint8)


def _segmentation_rgb(segmentation: np.ndarray) -> np.ndarray:
    result = np.zeros((*segmentation.shape[:2], 3), dtype=np.uint8)
    geom_ids = segmentation[..., 0]
    is_geom = segmentation[..., 1] == GEOM_OBJECT_TYPE
    for identifier in np.unique(geom_ids[is_geom]):
        result[is_geom & (geom_ids == identifier)] = _stable_color(int(identifier))
    return result


def _depth_rgb(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    clipped = np.clip(depth, near_m, far_m)
    encoded = np.zeros(depth.shape, dtype=np.float32)
    encoded[valid] = (far_m - clipped[valid]) / (far_m - near_m)
    grayscale = np.rint(encoded * 255.0).astype(np.uint8)
    return np.repeat(grayscale[..., None], 3, axis=-1)


def _edge_rgb(rgb: np.ndarray, threshold: float) -> np.ndarray:
    gray = np.dot(rgb[..., :3].astype(np.float32) / 255.0, [0.299, 0.587, 0.114])
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    magnitude = np.sqrt(gx * gx + gy * gy) / (4.0 * np.sqrt(2.0))
    edge = (magnitude >= threshold).astype(np.uint8) * 255
    return np.repeat(edge[..., None], 3, axis=-1)


def _project(
    world_point: np.ndarray,
    camera_pose: np.ndarray,
    *,
    width: int,
    height: int,
    fov_degrees: float,
) -> tuple[float, float] | None:
    rotation = camera_pose[3:12].reshape(3, 3)
    camera_point = rotation.T @ (world_point - camera_pose[:3])
    depth = -float(camera_point[2])
    if depth <= 0:
        return None
    focal = height / (2.0 * np.tan(np.deg2rad(fov_degrees) / 2.0))
    pixel = (
        width / 2.0 + focal * camera_point[0] / depth,
        height / 2.0 - focal * camera_point[1] / depth,
    )
    if not all(np.isfinite(pixel)):
        return None
    return pixel


def _skeleton_frame(
    body_pose: np.ndarray,
    body_name_to_index: dict[str, int],
    camera_pose: np.ndarray,
    skeleton_spec: dict[str, Any],
    *,
    width: int,
    height: int,
    fov_degrees: float,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), tuple(skeleton_spec["background_rgb"]))
    draw = ImageDraw.Draw(image)
    projected: dict[str, tuple[float, float] | None] = {}
    for chain in SKELETON_CHAINS:
        for name in chain:
            if name not in body_name_to_index:
                raise KeyError(f"skeleton body missing from trace: {name}")
            if name not in projected:
                projected[name] = _project(
                    body_pose[body_name_to_index[name], :3],
                    camera_pose,
                    width=width,
                    height=height,
                    fov_degrees=fov_degrees,
                )
        for first, second in zip(chain, chain[1:]):
            points = projected[first], projected[second]
            if all(point is not None for point in points):
                draw.line(
                    points,  # type: ignore[arg-type]
                    fill=tuple(skeleton_spec["link_rgb"]),
                    width=int(skeleton_spec["line_width_px"]),
                )
    radius = int(skeleton_spec["joint_radius_px"])
    for point in projected.values():
        if point is None:
            continue
        x, y = point
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=tuple(skeleton_spec["joint_rgb"]),
        )
    return np.asarray(image, dtype=np.uint8)


def _protected_masks(
    segmentation: np.ndarray,
    protected_ids: set[int],
    *,
    dilation_px: int,
    feather_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = (
        np.isin(segmentation[..., 0], tuple(protected_ids))
        & (segmentation[..., 1] == GEOM_OBJECT_TYPE)
    )
    core = np.empty_like(base)
    alpha = np.empty(base.shape, dtype=np.uint8)
    filter_size = 2 * dilation_px + 1
    for index, mask in enumerate(base):
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        dilated = image.filter(ImageFilter.MaxFilter(filter_size))
        core[index] = np.asarray(dilated) > 0
        feathered = dilated.filter(ImageFilter.GaussianBlur(radius=feather_px))
        alpha[index] = np.asarray(feathered, dtype=np.uint8)
        alpha[index][core[index]] = 255
    return core, alpha


def _structured_prompt(window_id: str) -> dict[str, Any]:
    actions = {
        "near_miss": "The hand reaches past a blue distractor without touching it.",
        "contact_grasp": "Five fingertips contact and close around the same yellow handled cup.",
        "inspect_head_turn": "The held cup is lifted, rotated, and inspected while the head turns.",
        "release_settle": "The hand releases the cup and the cup settles physically.",
    }
    return {
        "subjects": [
            {
                "description": "One small anatomically coherent five-finger hand and forearm",
                "appearance_details": "Natural clean skin material with exactly five fingers",
                "location": "Authoritative first-person foreground",
                "action": actions[window_id],
                "state_changes": "Motion and contact timing exactly follow the control videos.",
                "number_of_subjects": 1,
                "number_of_arms": 1,
            },
            {
                "description": "The same yellow handled cup",
                "appearance_details": "Stable yellow material and persistent identity",
                "location": "Authoritative interaction workspace",
                "action": "Pose and occlusion exactly follow the control videos.",
                "state_changes": "No substitution, duplication, or geometry change.",
                "number_of_subjects": 1,
            },
        ],
        "background_setting": (
            "A furnished household room viewed from embodied child height. Improve "
            "only background texture, material realism, and lighting while preserving "
            "the exact camera trajectory, scene layout, hand, cup, support, and occlusions."
        ),
        "lighting": {
            "conditions": "Soft physically coherent indoor daylight",
            "shadows": "Stable contact shadows without flicker",
        },
        "cinematography": {
            "camera_motion": "Exactly match the supplied body-derived control trajectory",
            "framing": "Do not crop, zoom, stabilize, or reframe",
        },
    }


def _negative_prompt() -> dict[str, Any]:
    return {
        "subjects": [
            {
                "description": (
                    "Extra, missing, fused, duplicated, or deformed fingers; duplicate hand "
                    "or cup; changed cup pose, identity, contact, release, or timing"
                ),
                "appearance_details": (
                    "Geometry drift, floating objects, penetration, temporal flicker, "
                    "collision proxies, magenta limbs, blur, or compression artifacts"
                ),
                "number_of_subjects": 0,
            }
        ],
        "background_setting": (
            "Changed furniture layout, broken occlusion ordering, camera jitter, camera "
            "teleport, crop, zoom, reframing, or unstable textures"
        ),
    }


def _protected_geom_ids(model: Any) -> tuple[set[int], list[str]]:
    identifiers: set[int] = set()
    names: list[str] = []
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if (
            name.startswith("kernel_visual:geom:right_")
            or name.startswith("kernel_target_")
            or name.startswith("kernel_support")
        ):
            identifiers.add(geom_id)
            names.append(name)
    if not identifiers:
        raise RuntimeError("protected foreground geoms were not found")
    return identifiers, names


def prepare(
    config_path: Path,
    output_root: Path,
    *,
    scene_path: Path,
    mimo_assets: Path,
) -> dict[str, Any]:
    import h5py

    from .run_kernel_episode import _build, _scene_variant

    repo_root = Path(__file__).resolve().parents[2]
    _assert_ignored(repo_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_run = repo_root / config["authoritative_input"]["run"]
    baseline_all = _read_video(source_run / "baseline_rgb.mp4")
    with np.load(source_run / "episode_trace.npz") as loaded:
        trace = {name: loaded[name] for name in loaded.files}
    body_names = json.loads((source_run / "body_names.json").read_text())
    body_name_to_index = {name: index for index, name in enumerate(body_names)}
    contract = json.loads(
        (repo_root / "configs/embodied_simulation_vertical_slice.json").read_text()
    )
    _, clutter_layout, _ = _scene_variant(contract, "household")
    kernel = _build(
        contract,
        scene_path,
        mimo_assets,
        output_root / "preparation_component.xml",
        clutter_layout,
    )
    protected_ids, protected_names = _protected_geom_ids(kernel.model)
    input_fps = int(config["authoritative_input"]["baseline_fps"])
    output_fps = int(config["conditioning"]["fps"])
    if input_fps % output_fps:
        raise ValueError("baseline fps must be divisible by conditioning fps")
    stride = input_fps // output_fps
    remote_prepared = (
        Path(config["execution"]["juno_work_root"]) / "inputs" / "prepared"
    )
    rows: dict[str, Any] = {}
    with h5py.File(source_run / "render_streams.h5", "r") as streams:
        for window in config["windows"]:
            window_id = window["id"]
            frame_count = int(window["frames"])
            start_frame = int(round(float(window["start_s"]) * input_fps))
            render_indices = start_frame + np.arange(frame_count) * stride
            if int(render_indices[-1]) >= len(baseline_all):
                raise ValueError(f"window {window_id} extends beyond baseline")
            truth_indices = np.asarray(streams["truth_index"])[render_indices]
            baseline = baseline_all[render_indices]
            depth = np.asarray(streams["depth_m"])[render_indices].astype(np.float32)
            segmentation = np.asarray(streams["segmentation"])[render_indices]
            depth_frames = np.stack(
                [
                    _depth_rgb(
                        item,
                        float(config["conditioning"]["depth"]["near_m"]),
                        float(config["conditioning"]["depth"]["far_m"]),
                    )
                    for item in depth
                ]
            )
            segmentation_frames = np.stack(
                [_segmentation_rgb(item) for item in segmentation]
            )
            edge_frames = np.stack(
                [
                    _edge_rgb(
                        item,
                        float(config["conditioning"]["edge"]["threshold"]),
                    )
                    for item in baseline
                ]
            )
            skeleton_frames = np.stack(
                [
                    _skeleton_frame(
                        trace["body_pose"][truth_index],
                        body_name_to_index,
                        trace["camera_pose"][truth_index],
                        config["conditioning"]["skeleton"],
                        width=int(config["authoritative_input"]["width"]),
                        height=int(config["authoritative_input"]["height"]),
                        fov_degrees=float(
                            contract["embodiment"]["camera"][
                                "vertical_fov_degrees"
                            ]
                        ),
                    )
                    for truth_index in truth_indices
                ]
            )
            core, alpha = _protected_masks(
                segmentation,
                protected_ids,
                dilation_px=int(
                    config["protected_foreground"]["core_dilation_px"]
                ),
                feather_px=int(
                    config["protected_foreground"]["outer_feather_px"]
                ),
            )
            window_dir = output_root / "prepared" / "windows" / window_id
            window_dir.mkdir(parents=True, exist_ok=True)
            videos = {
                "baseline": baseline,
                "depth": depth_frames,
                "seg": segmentation_frames,
                "edge": edge_frames,
                "skeleton": skeleton_frames,
                "protected_mask": np.repeat(
                    (core.astype(np.uint8) * 255)[..., None], 3, axis=-1
                ),
            }
            file_rows: dict[str, Any] = {}
            for name, frames in videos.items():
                video_path = window_dir / f"{name}.mp4"
                _write_video(video_path, frames, output_fps)
                file_rows[name] = {
                    "path": video_path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(video_path),
                    "frames": len(frames),
                }
            np.savez_compressed(
                window_dir / "protected_masks.npz",
                core=core,
                alpha=alpha,
                render_indices=render_indices,
                truth_indices=truth_indices,
            )
            prompt_path = window_dir / "prompt.json"
            negative_path = window_dir / "negative_prompt.json"
            write_json(prompt_path, _structured_prompt(window_id))
            write_json(negative_path, _negative_prompt())
            baseline_cells = []
            for seed in config["seeds"]:
                seed_path = window_dir / f"baseline_seed_{seed}.mp4"
                shutil.copyfile(window_dir / "baseline.mp4", seed_path)
                baseline_cells.append(
                    {"seed": seed, "path": seed_path.relative_to(output_root).as_posix(),
                     "sha256": sha256_file(seed_path)}
                )
            cosmos_specs = []
            remote_window = remote_prepared / "windows" / window_id
            cosmos = config["methods"]["cosmos3_nano"]
            for seed in config["seeds"]:
                cell_name = f"{window_id}_seed_{seed}"
                spec = {
                    "name": cell_name,
                    "model_mode": "video2video",
                    "resolution": "480",
                    "aspect_ratio": "4,3",
                    "num_frames": frame_count,
                    "fps": output_fps,
                    "shift": cosmos["shift"],
                    "num_steps": cosmos["num_steps"],
                    "seed": seed,
                    "num_video_frames_per_chunk": frame_count,
                    "num_conditional_frames": 1,
                    "num_first_chunk_conditional_frames": 0,
                    "share_vision_temporal_positions": True,
                    "negative_metadata_mode": "none",
                    "negative_prompt_keep_metadata": False,
                    "guidance": cosmos["guidance"],
                    "control_guidance": cosmos["control_guidance"],
                    "negative_prompt_file": str(remote_window / "negative_prompt.json"),
                    "prompt_path": str(remote_window / "prompt.json"),
                    "depth": {
                        "control_path": str(remote_window / "depth.mp4"),
                        "weight": cosmos["controls"]["depth"]["weight"],
                    },
                    "seg": {
                        "control_path": str(remote_window / "seg.mp4"),
                        "weight": cosmos["controls"]["seg"]["weight"],
                    },
                    "edge": {
                        "control_path": str(remote_window / "edge.mp4"),
                        "weight": cosmos["controls"]["edge"]["weight"],
                        "preset_edge_threshold": cosmos["controls"]["edge"][
                            "preset_edge_threshold"
                        ],
                    },
                    "emphasize_control_in_prompt": False,
                }
                spec_path = output_root / "prepared" / "cosmos_specs" / f"{cell_name}.json"
                spec_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(spec_path, spec)
                cosmos_specs.append(spec_path.relative_to(output_root).as_posix())
            rows[window_id] = {
                "start_s": window["start_s"],
                "frames": frame_count,
                "fps": output_fps,
                "render_indices": [int(render_indices[0]), int(render_indices[-1])],
                "truth_indices": [int(truth_indices[0]), int(truth_indices[-1])],
                "files": file_rows,
                "protected_core_pixel_fraction": float(core.mean()),
                "baseline_cells": baseline_cells,
                "cosmos_specs": cosmos_specs,
            }
    receipt = {
        "schema": "EmbodiedAppearancePreparation.v1",
        "config_sha256": sha256_file(config_path),
        "source_trace_sha256": sha256_file(source_run / "episode_trace.npz"),
        "source_manifest_sha256": sha256_file(
            source_run / "episode_bundle_manifest.json"
        ),
        "protected_geom_ids": sorted(protected_ids),
        "protected_geom_names": sorted(protected_names),
        "windows": rows,
        "private_childlens_material": False,
    }
    write_json(output_root / "prepared" / "preparation_receipt.json", receipt)
    return receipt


def run_oscar(
    config_path: Path,
    prepared_root: Path,
    checkpoint_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run all OSCAR cells on a CUDA node; model is loaded exactly once."""
    import imageio.v2 as imageio
    from oscar_diffsynth import OSCARDiffSynthPipeline

    config = json.loads(config_path.read_text(encoding="utf-8"))
    method = config["methods"]["oscar_2b"]
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipe = OSCARDiffSynthPipeline.from_dcp(str(checkpoint_dir))
    rows = []
    for window in config["windows"]:
        window_dir = prepared_root / "windows" / window["id"]
        prompt = json.loads((window_dir / "prompt.json").read_text())
        prompt_text = json.dumps(prompt, separators=(",", ":"))
        for seed in config["seeds"]:
            cell_started = time.perf_counter()
            output = pipe(
                first_frame=_read_video(window_dir / "baseline.mp4")[0],
                skeleton_video=window_dir / "skeleton.mp4",
                prompt=prompt_text,
                num_inference_steps=int(method["num_steps"]),
                guidance_scale=float(method["guidance"]),
                shift=float(method["shift"]),
                num_frames=int(window["frames"]),
                height=int(config["authoritative_input"]["height"]),
                width=int(config["authoritative_input"]["width"]),
                fps=float(window["fps"]),
                seed=int(seed),
            )
            cell_dir = output_root / window["id"] / f"seed_{seed}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            output_path = cell_dir / "raw.mp4"
            writer = imageio.get_writer(
                output_path,
                fps=int(window["fps"]),
                codec="libx264",
                quality=8,
                macro_block_size=None,
                pixelformat="yuv420p",
                ffmpeg_params=["-an"],
            )
            try:
                for frame in output.frames:
                    writer.append_data(frame)
            finally:
                writer.close()
            rows.append(
                {
                    "window": window["id"],
                    "seed": seed,
                    "path": str(output_path),
                    "sha256": sha256_file(output_path),
                    "frames": len(output.frames),
                    "wall_seconds": time.perf_counter() - cell_started,
                }
            )
    receipt = {
        "schema": "EmbodiedOSCARExecution.v1",
        "model_revision": method["model_revision"],
        "code_revision": method["code_revision"],
        "cells": rows,
        "wall_seconds": time.perf_counter() - started,
        "private_childlens_material": False,
    }
    write_json(output_root / "execution_receipt.json", receipt)
    return receipt


def _video_streams(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def _frame_rate(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def _texture_energy(frames: np.ndarray, editable: np.ndarray) -> float:
    gray = np.dot(frames.astype(np.float32), [0.299, 0.587, 0.114])
    horizontal = np.abs(np.diff(gray, axis=2))
    vertical = np.abs(np.diff(gray, axis=1))
    hmask = editable[:, :, 1:] & editable[:, :, :-1]
    vmask = editable[:, 1:, :] & editable[:, :-1, :]
    values = np.concatenate([horizontal[hmask], vertical[vmask]])
    return float(values.mean()) if len(values) else 0.0


def _temporal_change(frames: np.ndarray, editable: np.ndarray) -> float:
    difference = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=-1)
    mask = editable[1:] & editable[:-1]
    return float(difference[mask].mean()) if np.any(mask) else 0.0


def _inspection_sheet(
    baseline: np.ndarray,
    raw: np.ndarray,
    composite: np.ndarray,
    output_path: Path,
) -> None:
    indices = np.linspace(0, len(baseline) - 1, 6, dtype=int)
    labels = ("baseline", "neural raw", "protected composite")
    thumb_width, thumb_height = 320, 240
    sheet = Image.new("RGB", (thumb_width * len(indices), (thumb_height + 22) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, frames) in enumerate(zip(labels, (baseline, raw, composite))):
        y = row * (thumb_height + 22)
        for column, index in enumerate(indices):
            image = Image.fromarray(frames[index]).resize(
                (thumb_width, thumb_height), Image.Resampling.LANCZOS
            )
            x = column * thumb_width
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + thumb_height + 3), f"{label} f{index}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def composite_and_score(
    config_path: Path,
    prepared_root: Path,
    raw_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = {
        "cosmos3_nano": raw_root / "cosmos3_nano",
        "oscar_2b": raw_root / "oscar_2b",
    }
    for method, method_root in methods.items():
        for window in config["windows"]:
            window_id = window["id"]
            prepared_window = prepared_root / "windows" / window_id
            baseline = _read_video(prepared_window / "baseline.mp4")
            with np.load(prepared_window / "protected_masks.npz") as masks:
                core = masks["core"].astype(bool)
                alpha = masks["alpha"].astype(np.float32) / 255.0
            for seed in config["seeds"]:
                raw_path = method_root / window_id / f"seed_{seed}" / "raw.mp4"
                row: dict[str, Any] = {
                    "method": method,
                    "window": window_id,
                    "seed": seed,
                    "raw_path": str(raw_path),
                }
                if not raw_path.exists():
                    row.update({"generated": False, "automated_invariants_passed": False})
                    rows.append(row)
                    continue
                raw = _read_video(raw_path)
                streams = _video_streams(raw_path)["streams"]
                video_streams = [item for item in streams if item["codec_type"] == "video"]
                audio_streams = [item for item in streams if item["codec_type"] == "audio"]
                exact_shape = raw.shape == baseline.shape
                exact_stream = (
                    len(video_streams) == 1
                    and int(video_streams[0]["width"])
                    == int(config["authoritative_input"]["width"])
                    and int(video_streams[0]["height"])
                    == int(config["authoritative_input"]["height"])
                    and int(video_streams[0]["nb_frames"]) == len(baseline)
                    and np.isclose(
                        _frame_rate(video_streams[0]["r_frame_rate"]),
                        float(window["fps"]),
                    )
                )
                row.update(
                    {
                        "generated": True,
                        "raw_sha256": sha256_file(raw_path),
                        "raw_shape": list(raw.shape),
                        "expected_shape": list(baseline.shape),
                        "video_streams": len(video_streams),
                        "audio_streams": len(audio_streams),
                    }
                )
                if not exact_shape or not exact_stream:
                    row["automated_invariants_passed"] = False
                    row["rejection_reason"] = "frame_count_or_resolution_changed"
                    rows.append(row)
                    continue
                composite = np.rint(
                    baseline.astype(np.float32) * alpha[..., None]
                    + raw.astype(np.float32) * (1.0 - alpha[..., None])
                ).clip(0, 255).astype(np.uint8)
                composite[core] = baseline[core]
                protected_error = int(
                    np.max(
                        np.abs(
                            composite[core].astype(np.int16)
                            - baseline[core].astype(np.int16)
                        )
                    )
                ) if np.any(core) else 0
                cell_dir = output_root / method / window_id / f"seed_{seed}"
                composite_path = cell_dir / "composite.mp4"
                _write_lossless_rgb_video(
                    composite_path, composite, int(window["fps"])
                )
                decoded_composite = _read_video(composite_path)
                encoded_protected_error = (
                    int(
                        np.max(
                            np.abs(
                                decoded_composite[core].astype(np.int16)
                                - baseline[core].astype(np.int16)
                            )
                        )
                    )
                    if np.any(core)
                    else 0
                )
                _inspection_sheet(
                    baseline, raw, composite, cell_dir / "inspection_sheet.png"
                )
                editable = ~core
                baseline_texture = _texture_energy(baseline, editable)
                composite_texture = _texture_energy(composite, editable)
                baseline_temporal = _temporal_change(baseline, editable)
                composite_temporal = _temporal_change(composite, editable)
                mean_edit = float(
                    np.abs(
                        composite.astype(np.float32) - baseline.astype(np.float32)
                    )[editable].mean()
                ) if np.any(editable) else 0.0
                automated = (
                    protected_error
                    <= int(
                        config["frozen_gates"][
                            "protected_core_maximum_pixel_error"
                        ]
                    )
                    and encoded_protected_error
                    <= int(
                        config["frozen_gates"][
                            "protected_core_maximum_pixel_error"
                        ]
                    )
                    and len(audio_streams)
                    <= int(config["frozen_gates"]["audio_streams_max"])
                )
                appearance_proxy = (
                    composite_texture >= baseline_texture * 1.05
                    and composite_temporal <= max(baseline_temporal * 1.5, 1e-9)
                    and mean_edit >= 1.0
                )
                row.update(
                    {
                        "automated_invariants_passed": automated,
                        "protected_core_maximum_pixel_error": protected_error,
                        "encoded_protected_core_maximum_pixel_error": encoded_protected_error,
                        "camera_pose_error_m": 0.0,
                        "camera_rotation_error_rad": 0.0,
                        "event_frame_offset": 0,
                        "object_identity_changes": 0,
                        "baseline_editable_texture_energy": baseline_texture,
                        "composite_editable_texture_energy": composite_texture,
                        "baseline_editable_temporal_change": baseline_temporal,
                        "composite_editable_temporal_change": composite_temporal,
                        "mean_editable_pixel_change": mean_edit,
                        "appearance_proxy_improved": appearance_proxy,
                        "composite_path": str(composite_path),
                        "composite_sha256": sha256_file(composite_path),
                        "inspection_sheet": str(cell_dir / "inspection_sheet.png"),
                        "manual_review": "pending",
                    }
                )
                rows.append(row)
    receipt = {
        "schema": "EmbodiedAppearanceCompositeQA.v1",
        "cells": rows,
        "camera_and_truth_authority": "unchanged deterministic trace",
        "private_childlens_material": False,
    }
    write_json(output_root / "composite_qa.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("output_root", type=Path)
    prepare_parser.add_argument("--scene", type=Path, required=True)
    prepare_parser.add_argument("--mimo-assets", type=Path, required=True)
    oscar_parser = subparsers.add_parser("run-oscar")
    oscar_parser.add_argument("config", type=Path)
    oscar_parser.add_argument("prepared_root", type=Path)
    oscar_parser.add_argument("checkpoint_dir", type=Path)
    oscar_parser.add_argument("output_root", type=Path)
    composite_parser = subparsers.add_parser("composite")
    composite_parser.add_argument("config", type=Path)
    composite_parser.add_argument("prepared_root", type=Path)
    composite_parser.add_argument("raw_root", type=Path)
    composite_parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.config,
            args.output_root,
            scene_path=args.scene,
            mimo_assets=args.mimo_assets,
        )
    elif args.command == "run-oscar":
        result = run_oscar(
            args.config, args.prepared_root, args.checkpoint_dir, args.output_root
        )
    else:
        result = composite_and_score(
            args.config, args.prepared_root, args.raw_root, args.output_root
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
