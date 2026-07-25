"""Evidence probe for the MolmoSpaces high-level custom-robot attachment route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco

from molmo_spaces.robots.abstract import Robot

from .hybrid import _component_xml


class MIMoMolmoSpacesProbeRobot(Robot):
    """Minimal adapter used only to exercise Robot.add_robot_to_scene."""

    @staticmethod
    def robot_model_root_name() -> str:
        return "mimo_location"

    @property
    def namespace(self):  # pragma: no cover - never instantiated in this probe
        return "mimo_"

    @property
    def robot_view(self):  # pragma: no cover
        raise NotImplementedError

    @property
    def kinematics(self):  # pragma: no cover
        raise NotImplementedError

    @property
    def parallel_kinematics(self):  # pragma: no cover
        raise NotImplementedError

    @property
    def controllers(self):  # pragma: no cover
        return {}

    def reset(self):  # pragma: no cover
        raise NotImplementedError


def run(scene_path: Path, mimo_assets: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    component_path = output_dir / "route_a_mimo_component.xml"
    component_path.write_text(_component_xml(mimo_assets, "push"), encoding="utf-8")
    scene_spec = mujoco.MjSpec.from_file(str(scene_path.resolve()))
    config = SimpleNamespace(get_robot_xml_path=lambda: component_path)
    MIMoMolmoSpacesProbeRobot.add_robot_to_scene(
        config, scene_spec, "mimo_", [0.0, 0.6, 0.55], [1.0, 0.0, 0.0, 0.0]
    )
    model = scene_spec.compile()
    result = {
        "route": "A_high_level_MolmoSpaces_Robot.add_robot_to_scene",
        "compiled": True,
        "full_five_finger_body_present": (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mimo_right_hand") >= 0
        ),
        "actuator_count": int(model.nu),
        "sensor_count": int(model.nsensor),
        "required_mimo_sensor_present": (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, "mimo_vestibular_acc"
            )
            >= 0
        ),
        "disposition": (
            "schema_attachment_succeeds_but_runtime_RobotView_kinematics_and_controller_"
            "contracts_require_a_bespoke_full_humanoid_adapter"
        ),
        "fallback": "route_B_direct_MjSpec_attach_used_for_bounded_exercised_integration",
    }
    (output_dir / "route_a_probe_receipt.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_path", type=Path)
    parser.add_argument("mimo_assets", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.scene_path, args.mimo_assets, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
