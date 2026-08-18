"""Run the pinned MIMo-v2 showroom preflight and retain shared-clock evidence."""

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


def run(output_dir: Path, *, steps: int = 240, seed: int = 20260725) -> dict:
    from mimoEnv.envs.dummy import DEMO_XML, MIMoV2DummyEnv

    output_dir.mkdir(parents=True, exist_ok=True)
    env = MIMoV2DummyEnv(
        model_path=DEMO_XML,
        render_mode="rgb_array",
        show_sensors=False,
        width=640,
        height=480,
        age=24,
    )
    observation, _ = env.reset(seed=seed)
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    frames: list[np.ndarray] = []
    eye_frames: list[np.ndarray] = []
    frame_steps: list[int] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    proprioception: list[np.ndarray] = []
    touch: list[np.ndarray] = []
    vestibular: list[np.ndarray] = []
    started = time.perf_counter()
    for step in range(steps):
        observation, _, terminated, truncated, _ = env.step(action)
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
        proprioception.append(observation["observation"].copy())
        touch.append(observation["touch"].copy())
        vestibular.append(observation["vestibular"].copy())
        if step % 2 == 0:
            frames.append(env.render())
            eye_frames.append(observation["eye_left"].copy())
            frame_steps.append(step)
        if terminated or truncated:
            raise RuntimeError(f"stock MIMo episode ended unexpectedly at step {step}")
    elapsed = time.perf_counter() - started
    env.close()

    video_path = output_dir / "stock_mimo_v2_showroom.mp4"
    combined_frames = [
        np.concatenate(
            [
                frame,
                np.repeat(np.repeat(eye, 2, axis=0), 2, axis=1)[:480, :480],
            ],
            axis=1,
        )
        for frame, eye in zip(frames, eye_frames)
    ]
    imageio.mimwrite(video_path, combined_frames, fps=30, codec="libx264", quality=8)
    telemetry_path = output_dir / "stock_mimo_v2_telemetry.npz"
    np.savez_compressed(
        telemetry_path,
        step=np.arange(steps, dtype=np.int32),
        time_s=np.arange(1, steps + 1, dtype=np.float64) * 0.01,
        action=np.repeat(action[None, :], steps, axis=0),
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        proprioception=np.asarray(proprioception),
        touch=np.asarray(touch),
        vestibular=np.asarray(vestibular),
        video_frame_step=np.asarray(frame_steps, dtype=np.int32),
    )
    touch_array = np.asarray(touch)
    receipt = {
        "stage": "stock_mimo_v2",
        "seed": seed,
        "age_months": 24,
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "mujoco": mujoco.__version__,
        "steps": steps,
        "simulation_seconds": steps * 0.01,
        "wall_seconds": elapsed,
        "simulation_steps_per_wall_second": steps / elapsed,
        "action_shape": list(action.shape),
        "qpos_shape": list(np.asarray(qpos).shape),
        "qvel_shape": list(np.asarray(qvel).shape),
        "proprioception_shape": list(np.asarray(proprioception).shape),
        "touch_shape": list(touch_array.shape),
        "touch_nonzero_samples": int(np.count_nonzero(touch_array)),
        "vestibular_shape": list(np.asarray(vestibular).shape),
        "video_frames": len(frames),
        "video": {"name": video_path.name, "sha256": _sha256(video_path)},
        "telemetry": {"name": telemetry_path.name, "sha256": _sha256(telemetry_path)},
    }
    receipt_path = output_dir / "stock_mimo_v2_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, steps=args.steps, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
