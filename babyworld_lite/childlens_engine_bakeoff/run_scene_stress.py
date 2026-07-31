"""Run the frozen embodied vertical slice across the three scene variants."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from .bundle import write_json
from .run_kernel_episode import _assert_ignored, run


VARIANTS = ("sparse", "household", "messy")


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    physics = result["physics_qa"]
    collision = physics["collision_policy"]
    render = result["render_qa"]
    return {
        "passed": result["passed"],
        "distractor_count": result["distractor_count"],
        "truth_samples": result["truth_samples"],
        "camera_mount_max_translation_error_m": physics["camera_mount"][
            "maximum_translation_error_m"
        ],
        "camera_mount_max_rotation_error_rad": physics["camera_mount"][
            "maximum_rotation_error_rad"
        ],
        "collision_policy_sha256": collision["final_sha256"],
        "collision_policy_unchanged": collision["unchanged"],
        "all_relevant_collisions_enabled": collision["all_relevant_enabled"],
        "minimum_relevant_contact_distance_m": collision[
            "minimum_relevant_contact_distance_m"
        ],
        "persistent_penetration_frames": collision[
            "persistent_penetration_frames"
        ],
        "near_miss_clearance_m": physics["near_miss"]["minimum_clearance_m"],
        "near_miss_contact_substeps": physics["near_miss"]["contact_substeps"],
        "first_contact_time_s": physics["contact"]["first_contact_time_s"],
        "maximum_distinct_finger_contacts": physics["contact"][
            "maximum_distinct_finger_contacts"
        ],
        "grasp_assist_engaged": physics["grasp"]["assist_engaged"],
        "grasp_assist_engagement_time_s": physics["grasp"][
            "assist_engagement_time_s"
        ],
        "grasp_assist_pose_jump_m": physics["grasp"]["assist_pose_jump_m"],
        "grasp_assist_pose_jump_degrees": physics["grasp"][
            "assist_pose_jump_degrees"
        ],
        "maximum_timestamp_error_s": physics["synchronization"][
            "maximum_timestamp_error_s"
        ],
        "object_identity_changes": physics["object_identity"]["changes"],
        "determinism_all_pass": result["determinism_qa"]["all_pass"],
        "determinism_maximum_numeric_absolute_error": result[
            "determinism_qa"
        ]["maximum_numeric_absolute_error"],
        "projected_contact_error_px": render[
            "maximum_projected_contact_error_px"
        ],
        "contact_surface_distance_m": render[
            "maximum_3d_contact_surface_distance_m"
        ],
        "collision_proxy_pixels": render["collision_proxy_pixels"],
        "skin_artifact_pixels": render["skin_artifact_pixels"],
        "authored_clutter_visual_count": render[
            "authored_clutter_visual_count"
        ],
        "clutter_visible_frame_fraction": render[
            "clutter_visible_frame_fraction"
        ],
        "maximum_clutter_area_fraction": render[
            "maximum_clutter_area_fraction"
        ],
        "qualification_checks": {
            "physics": result["qualification"]["physics"],
            "render": result["qualification"]["render"],
            "appearance": result["qualification"]["appearance"],
        },
        "artifact_dir": result["output_dir"],
        "manifest": result["manifest"],
    }


def run_stress(
    contract_path: Path,
    scene_path: Path,
    mimo_assets: Path,
    output_root: Path,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    _assert_ignored(repo_root, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        try:
            result = run(
                contract_path,
                scene_path,
                mimo_assets,
                output_root / variant,
                scene_variant=variant,
                render=True,
            )
            scenes[variant] = _compact(result)
        except Exception as error:  # Preserve bounded evidence for later repair.
            failure = {
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "artifact_dir": str(output_root / variant),
            }
            write_json(output_root / variant / "failure.json", failure)
            scenes[variant] = failure

    required = ("sparse", "household")
    passed = all(scenes[name].get("passed", False) for name in required)
    report = {
        "schema": "EmbodiedSceneStressQualification",
        "question": (
            "Does the unchanged frozen vertical slice satisfy the Phase 1 hard "
            "gates in sparse and household clutter, with messy retained as a "
            "bounded diagnostic?"
        ),
        "event_semantics_changed": False,
        "required_variants": list(required),
        "diagnostic_variants": ["messy"],
        "scenes": scenes,
        "required_variants_passed": passed,
        "phase_2_gate": "PASS" if passed else "REPAIR_REQUIRED",
        "regeneration_command": [
            "python",
            "-m",
            "babyworld_lite.childlens_engine_bakeoff.run_scene_stress",
            str(contract_path),
            str(scene_path),
            str(mimo_assets),
            str(output_root),
        ],
        "private_childlens_material": False,
    }
    write_json(output_root / "phase_2_stress_qa.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("mimo_assets", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = run_stress(
        args.contract, args.scene, args.mimo_assets, args.output_root
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
