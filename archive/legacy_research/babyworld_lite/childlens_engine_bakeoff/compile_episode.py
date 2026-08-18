"""Compile bounded episode intents into deterministic resolved specifications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .bundle import write_json
from .spec_kernel import CalibrationPolicy, compile_episode_intent


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compose_embodied_prompt_plan(
    episode: dict[str, Any], vertical_contract: dict[str, Any]
) -> dict[str, Any]:
    """Resolve a prompt to the frozen bounded action/language template.

    The prompt selects a repository-authored template.  This compiler never
    emits joint, body, camera, or object trajectories; those remain controller
    and physics outputs.
    """
    allowed = set(episode["planner"]["allowed_actions"])
    schedule = episode["activity"]["vertical_slice"]
    if not schedule:
        raise ValueError("continuous episode schedule must not be empty")
    if schedule[0]["start_s"] != 0.0:
        raise ValueError("continuous episode schedule must start at zero")
    if schedule[-1]["end_s"] != episode["activity"]["duration_s"]:
        raise ValueError("continuous episode schedule must end at duration_s")
    if any(left["end_s"] != right["start_s"] for left, right in zip(schedule, schedule[1:])):
        raise ValueError("continuous episode schedule must be contiguous")
    invalid_actions = sorted({row["action"] for row in schedule} - allowed)
    if invalid_actions:
        raise ValueError(f"planner emitted actions outside bounded vocabulary: {invalid_actions}")

    target_request = episode["planner"]["target_reference"]
    target = vertical_contract["scene_family"]["target"]
    if target_request["required_persistent_id"] != target["persistent_id"]:
        raise ValueError("requested scene object does not resolve to the frozen target")
    scene_variants = {
        row["id"]: row for row in vertical_contract["scene_family"]["variants"]
    }
    variant_id = episode["scene_variant"]
    if variant_id not in scene_variants:
        raise ValueError(f"unknown furnished scene variant: {variant_id}")

    actions = [
        {
            "index": index,
            "action": row["action"],
            "behavior_phase": row["phase"],
            "start_s": row["start_s"],
            "end_s": row["end_s"],
            "target_persistent_id": target["persistent_id"],
        }
        for index, row in enumerate(schedule)
    ]
    plan = {
        "schema": "EmbodiedBoundedActivityLanguagePlan.v1",
        "prompt": episode["prompt"],
        "duration_s": episode["activity"]["duration_s"],
        "scene": {
            "source_scene": vertical_contract["scene_family"]["source_scene"],
            "variant": variant_id,
            "distractor_count": scene_variants[variant_id]["distractor_count"],
        },
        "scene_resolution": {
            "requested_label": target_request["requested_label"],
            "resolved_persistent_id": target["persistent_id"],
            "resolved_geometry_sha256": target["sha256"],
            "status": "resolved_before_execution",
            "repairs": [],
        },
        "bounded_action_vocabulary": sorted(allowed),
        "actions": actions,
        "utterances": episode["language"]["utterances"],
        "trajectory_authority": (
            "controller, IK, collision constraints, and MuJoCo physics only"
        ),
        "planner_emitted_joint_trajectories": False,
        "planner_emitted_camera_trajectory": False,
        "planner_emitted_object_trajectory": False,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def compile_embodied_file(
    episode_path: Path, vertical_contract_path: Path, output_path: Path
) -> dict[str, Any]:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    vertical_contract = json.loads(
        vertical_contract_path.read_text(encoding="utf-8")
    )
    plan = compose_embodied_prompt_plan(episode, vertical_contract)
    write_json(output_path, plan)
    return plan


def compile_file(
    intent_path: Path,
    protocol_path: Path,
    calibration_path: Path,
    asset_registry_path: Path,
    output_path: Path,
) -> dict:
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assets = json.loads(asset_registry_path.read_text(encoding="utf-8"))
    calibration = CalibrationPolicy.load(calibration_path)
    resolved = compile_episode_intent(
        intent, protocol, calibration, asset_registry=assets
    )
    write_json(output_path, resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("intent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/childlens_mimo_molmospaces_spec_kernel.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("configs/childlens_simulator_bridge.json"),
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("configs/childlens_mimo_molmospaces_asset_registry.json"),
    )
    args = parser.parse_args()
    resolved = compile_file(
        args.intent, args.protocol, args.calibration, args.assets, args.output
    )
    print(json.dumps({"spec_sha256": resolved["spec_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
