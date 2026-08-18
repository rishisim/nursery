"""Score frozen ChildLens camera uncertainty models on target masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _coordinates(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx, cy = (width - 1) / 2, (height - 1) / 2
    dx, dy = xx - cx, yy - cy
    return dx, dy, np.hypot(dx, dy)


def _remap(mask: np.ndarray, model: str) -> np.ndarray:
    height, width = mask.shape
    dx, dy, radius = _coordinates(height, width)
    diagonal_radius = np.hypot(width / 2, height / 2)
    safe_radius = np.maximum(radius, 1e-8)
    if model == "rectilinear_brown_mild":
        normalized = radius / diagonal_radius
        scale = 1 - 0.16 * normalized**2 + 0.035 * normalized**4
        source_radius = radius * scale
    else:
        theta_max = np.deg2rad(70.0)
        if model == "equisolid_140_diagonal":
            focal = diagonal_radius / (2 * np.sin(theta_max / 2))
            theta = 2 * np.arcsin(np.clip(radius / (2 * focal), 0, 0.999999))
        elif model == "polynomial_fisheye_mild":
            coefficient_at_max = 0.78 * theta_max - 0.035 * theta_max**3
            focal = diagonal_radius / coefficient_at_max
            theta = radius / max(focal * 0.78, 1e-8)
            for _ in range(6):
                value = focal * (0.78 * theta - 0.035 * theta**3) - radius
                derivative = focal * (0.78 - 0.105 * theta**2)
                theta -= value / np.maximum(derivative, 1e-8)
        else:
            raise ValueError(model)
        rectilinear_focal = (height / 2) / np.tan(np.deg2rad(45.0))
        source_radius = rectilinear_focal * np.tan(theta)
    map_x = (width - 1) / 2 + dx * source_radius / safe_radius
    map_y = (height - 1) / 2 + dy * source_radius / safe_radius
    return cv2.remap(
        mask.astype(np.uint8),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def run(mask_dir: Path, output_path: Path) -> dict:
    models = (
        "rectilinear_brown_mild",
        "equisolid_140_diagonal",
        "polynomial_fisheye_mild",
    )
    rows = []
    for path in sorted(mask_dir.glob("replay_segmentation_*.png")):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        mask = rgb.mean(axis=2) > 0.5
        for model in models:
            transformed = _remap(mask, model)
            rows.append(
                {
                    "frame": int(path.stem.rsplit("_", 1)[1]),
                    "camera_model": model,
                    "target_area_fraction": float(transformed.mean()),
                    "passes_0_015": bool(transformed.mean() >= 0.015),
                }
            )
    result = {
        "frozen_source": "configs/childlens_room_hand_camera_bakeoff.json",
        "threshold": 0.015,
        "rows": rows,
        "minimum_area_fraction": min(row["target_area_fraction"] for row in rows),
        "all_pass": all(row["passes_0_015"] for row in rows),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.mask_dir, args.output_path), indent=2))


if __name__ == "__main__":
    main()
