"""Deterministic TDW evidence episode for the adaptive ChildLens repair.

This module emits engineering-readiness evidence only.  It does not implement
or launch a cue-lift comparison.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int = 2026072501
    frames: int = 48
    physics_hz: int = 60
    clock_hz: int = 24_000
    port: int = 0
    launch_arch: str = "x86_64"

    @property
    def ticks_per_frame(self) -> int:
        if self.clock_hz % self.physics_hz:
            raise ValueError("clock_hz must be divisible by physics_hz")
        return self.clock_hz // self.physics_hz


def derive_imu(samples: list[dict[str, Any]], dt: float) -> list[dict[str, Any]]:
    """Derive acceleration/angular velocity from timestamped physical poses.

    Central differences are used away from the boundaries; one-sided
    differences are used at the boundaries.  No smoothing is applied.
    """
    if dt <= 0 or len(samples) < 3:
        raise ValueError("derive_imu requires at least three samples and dt > 0")
    velocities = [s["velocity"] for s in samples]
    angular = [s["angular_velocity"] for s in samples]
    result: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        lo, hi = (0, 1) if i == 0 else ((len(samples) - 2, len(samples) - 1) if i == len(samples) - 1 else (i - 1, i + 1))
        span = (hi - lo) * dt
        result.append(
            {
                "tick": sample["tick"],
                "linear_acceleration_m_s2": [
                    (velocities[hi][axis] - velocities[lo][axis]) / span for axis in range(3)
                ],
                "angular_velocity_rad_s": list(angular[i]),
                "method": "finite_difference_physics_velocity",
            }
        )
    return result


def _v(value: Any) -> list[float]:
    return [float(x) for x in value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_port(requested: int) -> int:
    if requested:
        return requested
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def run_episode(build: Path, output_dir: Path, spec: EpisodeSpec = EpisodeSpec()) -> Path:
    """Launch the official build and emit a normalized replay receipt."""
    # Imports stay local so the repository doctor can inspect this module even
    # when the isolated TDW environment is not active.
    from tdw.add_ons.collision_manager import CollisionManager
    from tdw.add_ons.image_capture import ImageCapture
    from tdw.add_ons.object_manager import ObjectManager
    from tdw.add_ons.robot import Robot
    from tdw.add_ons.third_person_camera import ThirdPersonCamera
    from tdw.controller import Controller
    from tdw.tdw_utils import TDWUtils

    build = build.resolve()
    if not build.is_file():
        raise FileNotFoundError(build)
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "frames"
    log_path = output_dir / "tdw_player.log"
    port = _resolve_port(spec.port)
    build_command = [str(build), f"-port {port}", "-batchmode", "-logFile", str(log_path)]
    if sys.platform == "darwin" and spec.launch_arch == "x86_64":
        build_command = ["/usr/bin/arch", "-x86_64", *build_command]
    process = subprocess.Popen(
        build_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    # Unity's socket client is reliable once the player has completed its
    # platform initialization; matching TDW's own launcher cadence avoids an
    # Apple-Silicon startup race observed when binding immediately.
    time.sleep(5)
    controller = None
    try:
        controller = Controller(port=port, check_version=False, launch_build=False)
        falling_id, static_id, robot_id = 10, 20, 30
        camera = ThirdPersonCamera(
            avatar_id="ego", position={"x": 2.2, "y": 1.7, "z": -3.2},
            look_at={"x": 0, "y": 0.7, "z": 0},
        )
        capture = ImageCapture(
            path=frames_dir, avatar_ids=["ego"], png=True,
            pass_masks=["_img", "_id", "_depth"],
        )
        objects = ObjectManager(transforms=True, rigidbodies=True, bounds=False)
        collisions = CollisionManager(enter=True, stay=True, exit=True)
        robot = Robot(name="ur5", robot_id=robot_id, position={"x": -1.0, "y": 0, "z": 0})
        controller.add_ons.extend([camera, capture, objects, collisions, robot])
        commands = [
            {"$type": "set_screen_size", "width": 160, "height": 120},
            TDWUtils.create_empty_room(6, 6),
            *controller.get_add_physics_object(
                "iron_box", falling_id, position={"x": 0, "y": 1.2, "z": 0},
                default_physics_values=False, mass=1, bounciness=0,
            ),
            *controller.get_add_physics_object(
                "iron_box", static_id, position={"x": 1.2, "y": 2.4, "z": 0},
                default_physics_values=False, kinematic=True, gravity=False,
            ),
        ]
        controller.communicate(commands)
        movable = [
            (jid, joint) for jid, joint in robot.static.joints.items()
            if getattr(joint.joint_type, "name", "") != "fixed_joint"
        ]
        if not movable:
            raise RuntimeError("TDW robot exposed no movable joint")
        joint_id = movable[0][0]
        samples: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        positive_contacts: list[dict[str, Any]] = []
        forbidden_negative_contacts: list[dict[str, Any]] = []
        initial_angle = None
        for frame in range(spec.frames):
            tick = frame * spec.ticks_per_frame
            if frame == 4:
                current = float(robot.dynamic.joints[joint_id].angles[0])
                initial_angle = current
                target = current + 20.0
                robot.set_joint_targets({joint_id: target})
                actions.append({"tick": tick, "type": "set_joint_target", "joint_id": joint_id, "target": target})
            controller.communicate([])
            transform = objects.transforms[falling_id]
            rigidbody = objects.rigidbodies[falling_id]
            joint = robot.dynamic.joints[joint_id]
            samples.append(
                {
                    "frame": frame,
                    "tick": tick,
                    "timestamp_seconds": tick / spec.clock_hz,
                    "transform": {"position": _v(transform.position), "rotation": _v(transform.rotation)},
                    "velocity": _v(rigidbody.velocity),
                    "angular_velocity": _v(rigidbody.angular_velocity),
                    "sleeping": bool(rigidbody.sleeping),
                    "joint": {"id": joint_id, "angles_degrees": _v(joint.angles), "moving": bool(joint.moving)},
                }
            )
            for pair, event in collisions.obj_collisions.items():
                row = {"tick": tick, "kind": "object", "ids": [pair.int1, pair.int2], "state": event.state}
                (forbidden_negative_contacts if static_id in row["ids"] else positive_contacts).append(row)
            for object_id, event in collisions.env_collisions.items():
                row = {"tick": tick, "kind": "environment", "ids": [object_id], "state": event.state}
                (forbidden_negative_contacts if object_id == static_id else positive_contacts).append(row)
        controller.communicate({"$type": "terminate"})
        controller = None
        process.wait(timeout=15)
        images = sorted(p for p in frames_dir.rglob("*") if p.is_file())
        pass_counts = {mask: sum(mask in p.name for p in images) for mask in ("img", "id", "depth")}
        imu = derive_imu(samples, 1 / spec.physics_hz)
        final_angle = float(samples[-1]["joint"]["angles_degrees"][0])
        receipt = {
            "schema_version": "childlens-tdw-adaptive-repair-v1.0.0",
            "purpose": "environment readiness only; cue-lift arms not run",
            "spec": asdict(spec) | {"resolved_port": port},
            "engine": {"name": "TDW", "release": "1.13.0", "build": str(build)},
            "channels": {
                "images": [{"path": str(p.relative_to(output_dir)), "sha256": _sha256(p)} for p in images],
                "pass_counts": pass_counts,
                "physical_samples": samples,
                "actions": actions,
                "proprioception": [s["joint"] | {"tick": s["tick"]} for s in samples],
                "imu": imu,
                "contacts": positive_contacts,
            },
            "controls": {
                "positive_collision_observed": bool(positive_contacts),
                "negative_no_collision": not forbidden_negative_contacts,
                "negative_object_id": static_id,
                "forbidden_negative_contacts": forbidden_negative_contacts,
                "joint_changed_degrees": None if initial_angle is None else final_angle - initial_angle,
                "static_pre_action_joint_quiet": len({
                    tuple(s["joint"]["angles_degrees"]) for s in samples[:4]
                }) == 1,
            },
        }
        receipt_path = output_dir / "replay_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt_path
    finally:
        if controller is not None:
            try:
                controller.communicate({"$type": "terminate"})
            except Exception:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
