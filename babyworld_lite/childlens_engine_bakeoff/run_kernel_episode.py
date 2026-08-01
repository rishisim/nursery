"""Run and qualify the canonical embodied vertical slice."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .bundle import build_manifest, sha256_file, validate_shared_clock, write_json
from .determinism import write_receipt as write_determinism_receipt
from .physics_kernel import build_kernel_model, run_physics_trace
from .trace_render import render_trace


def _assert_ignored(repo_root: Path, output_dir: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(output_dir.resolve())],
        cwd=repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"output directory is not ignored: {output_dir}")


def _target_definition(contract: dict[str, Any]) -> dict[str, Any]:
    target = contract["scene_family"]["target"]
    return {"geometry": target["geometry"], "rgba": target["rgba"]}


def _scene_variant(
    contract: dict[str, Any], scene_variant: str | None
) -> tuple[str, list[dict[str, Any]], int]:
    """Resolve one frozen scene-family member without changing event semantics."""
    if scene_variant is None:
        return "phase_1_base", [], 1
    variants = {
        item["id"]: item for item in contract["scene_family"]["variants"]
    }
    if scene_variant not in variants:
        raise ValueError(
            f"unknown scene variant {scene_variant!r}; expected one of "
            f"{sorted(variants)}"
        )
    count = int(variants[scene_variant]["distractor_count"])
    placements = contract["scene_family"]["clutter_layout"]["placements"]
    selected = placements[: count - 1]
    if len(selected) != count - 1:
        raise ValueError(
            f"scene variant {scene_variant!r} requires {count - 1} authored "
            f"clutter placements, but only {len(selected)} are available"
        )
    return scene_variant, selected, count


def _build(
    contract: dict[str, Any],
    scene_path: Path,
    mimo_assets: Path,
    component_path: Path,
    clutter_layout: list[dict[str, Any]],
):
    placement = contract["scene_family"]["qualified_placement"]
    mount = contract["embodiment"]["camera_mount"]
    return build_kernel_model(
        scene_path,
        mimo_assets,
        component_path,
        root_xy=tuple(placement["root_xy_m"]),
        target_definition=_target_definition(contract),
        support_definition=contract["scene_family"].get("support"),
        clutter_layout=clutter_layout,
        target_offset_from_root=tuple(
            placement["target_offset_from_root_m"]
        ),
        support_offset_from_root=tuple(
            placement["support_offset_from_root_m"]
        ),
        reach_distractor_offset_from_root=tuple(
            placement.get(
                "reach_distractor_offset_from_root_m", [0.23, 0.02, 0.10]
            )
        ),
        camera_mount_position=tuple(mount["translation_head_m"]),
        camera_mount_quaternion=tuple(mount["quaternion_head_wxyz"]),
    )


def _physics_gate(contract: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    gate = contract["frozen_gates"]
    checks = {
        "immutable_head_camera_mount": (
            receipt["camera_mount"]["maximum_translation_error_m"]
            <= gate["camera_mount_translation_error_m_max"]
            and receipt["camera_mount"]["maximum_rotation_error_rad"]
            <= gate["camera_mount_rotation_error_rad_max"]
        ),
        "static_relevant_collisions": (
            receipt["collision_policy"]["unchanged"]
            and receipt["collision_policy"]["all_relevant_enabled"]
        ),
        "penetration": (
            receipt["collision_policy"]["minimum_relevant_contact_distance_m"]
            >= -gate["penetration_depth_m_max"]
            and receipt["collision_policy"]["persistent_penetration_frames"]
            <= gate["persistent_penetration_frames_max"]
        ),
        "true_near_miss": (
            receipt["near_miss"]["minimum_clearance_m"]
            >= gate["near_miss_clearance_m_min"]
            and receipt["near_miss"]["contact_substeps"]
            <= gate["near_miss_contact_count_max"]
        ),
        "contact_before_gated_assist": (
            receipt["contact"]["first_contact_time_s"] is not None
            and receipt["grasp"]["multipoint_contact_evidence_duration_s"]
            >= contract["grasp"]["minimum_contact_duration_s_for_assist"]
            and receipt["contact"]["maximum_distinct_finger_contacts"]
            >= contract["grasp"]["minimum_distinct_finger_contacts_for_assist"]
            and receipt["grasp"]["assist_pose_jump_m"]
            <= contract["grasp"]["assist_pose_jump_tolerance_m"]
            and receipt["grasp"]["assist_pose_jump_degrees"]
            <= contract["grasp"]["assist_pose_jump_tolerance_degrees"]
        ),
        "lift_rotate_head_turn": (
            receipt["manipulation"]["maximum_lift_m"]
            >= gate["minimum_lift_m"]
            and receipt["manipulation"]["maximum_object_rotation_degrees"]
            >= gate["minimum_rotation_degrees"]
            and receipt["manipulation"]["maximum_head_turn_degrees"]
            >= gate["head_turn_degrees_min"]
            and receipt["manipulation"]["head_turn_contact_retention_fraction"]
            >= gate["contact_retention_fraction_during_head_turn_min"]
        ),
        "release_and_settle": (
            receipt["manipulation"]["released"]
            and receipt["manipulation"]["final_settled_window_maximum_speed_m_s"]
            <= gate["release_settle_speed_m_s_max"]
            and receipt["manipulation"]["final_settled_duration_s"]
            >= gate["release_settle_duration_s_min"]
        ),
        "imu_consistency": (
            receipt["imu_consistency"]["accelerometer_rmse_m_s2"]
            <= gate["imu_linear_acceleration_rmse_m_s2_max"]
            and receipt["imu_consistency"]["gyroscope_rmse_rad_s"]
            <= gate["imu_angular_velocity_rmse_rad_s_max"]
        ),
        "shared_clock": (
            receipt["synchronization"]["all_frame_stream_lengths_equal"]
            and receipt["synchronization"]["maximum_timestamp_error_s"]
            <= gate["stream_timestamp_error_s_max"]
        ),
        "object_identity": (
            receipt["object_identity"]["changes"]
            <= gate["object_identity_changes_max"]
        ),
        "physical_authority": (
            not receipt["direct_target_transform_after_initialization"]
            and not receipt["independent_camera_control"]
            and not receipt["hand_mocap_or_weld"]
        ),
        "collision_checked_locomotion_assist": (
            receipt["locomotion_assist"]["explicitly_labeled"]
            and receipt["locomotion_assist"]["collision_checked"]
            and receipt["locomotion_assist"]["maximum_root_speed_m_s"]
            <= contract["embodiment"]["locomotion_assist"]["maximum_root_speed_m_s"]
            and receipt["locomotion_assist"]["maximum_root_acceleration_m_s2"]
            <= contract["embodiment"]["locomotion_assist"]["maximum_root_acceleration_m_s2"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _render_gate(contract: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    gate = contract["frozen_gates"]
    checks = {
        "collision_proxies_hidden": (
            receipt["collision_proxy_pixels"] <= gate["collision_proxy_pixels_max"]
        ),
        "skin_artifacts_absent": (
            receipt["skin_artifact_pixels"] <= gate["skin_artifact_pixels_max"]
        ),
        "contact_projection": (
            receipt["maximum_projected_contact_error_px"] is not None
            and receipt["maximum_projected_contact_error_px"]
            <= gate["visible_contact_alignment_px_max"]
            and receipt["maximum_3d_contact_surface_distance_m"] is not None
            and receipt["maximum_3d_contact_surface_distance_m"]
            <= gate["visible_contact_alignment_m_max"]
        ),
        "event_frames": (
            abs(receipt["rgb_truth_contact_frame_offset"])
            <= gate["rgb_truth_event_frame_offset_max"]
            and abs(receipt["rgb_truth_release_frame_offset"])
            <= gate["rgb_truth_event_frame_offset_max"]
            and receipt["contact_event_target_visible"]
            and receipt["release_event_target_visible"]
        ),
        "camera_replay": (
            receipt["maximum_replay_camera_translation_error_m"]
            <= gate["replay_numeric_absolute_error_max"]
            and receipt["maximum_replay_camera_rotation_error_rad"]
            <= gate["replay_numeric_absolute_error_max"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _run_blender_overlay(
    *,
    repo_root: Path,
    blender_path: Path,
    blend_path: Path,
    trace_path: Path,
    body_names_path: Path,
    spec_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    fps: int,
    all_frames: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(blender_path),
        str(blend_path),
        "--background",
        "--python",
        str(Path(__file__).with_name("mpfb_overlay_renderer.py")),
        "--",
        "--trace",
        str(trace_path),
        "--body-names",
        str(body_names_path),
        "--spec",
        str(spec_path),
        "--output-dir",
        str(output_dir),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--render-samples",
        "4" if all_frames else "16",
    ]
    if all_frames:
        command.append("--all-frames")
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (output_dir / "blender.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Blender MPFB replay failed ({result.returncode}):\n"
            + result.stdout[-4000:]
        )
    return json.loads((output_dir / "overlay_receipt.json").read_text())


def _appearance_gate(
    contract: dict[str, Any], render_receipt: dict[str, Any]
) -> dict[str, Any]:
    gate = contract["frozen_gates"]
    checks = {
        "native_mimo_appearance_authoritative": (
            render_receipt["appearance_status"].startswith(
                "authoritative deterministic native MIMo"
            )
        ),
        "physical_collision_layer_hidden": (
            render_receipt["physical_collision_geom_group"] == 3
            and render_receipt["collision_proxy_pixels"]
            <= gate["collision_proxy_pixels_max"]
        ),
        "coarticulated_native_layer_visible": (
            render_receipt["native_appearance_geom_group"] == 2
            and render_receipt["native_material"] == "skin"
        ),
        "clean_native_material": (
            render_receipt["skin_artifact_pixels"]
            <= gate["skin_artifact_pixels_max"]
        ),
    }
    return {
        "authoritative_baseline": "baseline_rgb.mp4",
        "policy": "native MIMo appearance; MPFB diagnostic only",
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(
    contract_path: Path,
    scene_path: Path,
    mimo_assets: Path,
    output_dir: Path,
    *,
    render: bool = True,
    scene_variant: str | None = None,
    blender_path: Path | None = None,
    mpfb_blend_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    _assert_ignored(repo_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    variant_id, clutter_layout, distractor_count = _scene_variant(
        contract, scene_variant
    )
    copied_contract = output_dir / "vertical_slice_contract.json"
    shutil.copyfile(contract_path, copied_contract)

    kernel = _build(
        contract,
        scene_path,
        mimo_assets,
        output_dir / "kernel_component.xml",
        clutter_layout,
    )
    truth_hz = int(contract["rates_hz"]["truth"])
    physics_hz = int(contract["rates_hz"]["physics"])
    trace, physics_qa = run_physics_trace(
        kernel, contract, truth_hz=truth_hz, physics_hz=physics_hz
    )
    physics_qa["scene_variant"] = {
        "id": variant_id,
        "distractor_count": distractor_count,
        "fixed_reach_distractor_count": 1,
        "authored_clutter_count": len(clutter_layout),
        "physical_clutter_geom_count": len(kernel.clutter_geom_ids),
        "visual_clutter_geom_count": len(kernel.clutter_visual_geom_ids),
        "event_semantics_unchanged": True,
    }
    trace_path = output_dir / "episode_trace.npz"
    np.savez_compressed(trace_path, **trace)
    write_json(output_dir / "body_names.json", list(kernel.mimo_body_names))
    write_json(output_dir / "physics_qa.json", physics_qa)
    clock_qa = validate_shared_clock(
        trace["time_s"], {key: value for key, value in trace.items() if key != "time_s"}
    )
    write_json(output_dir / "shared_clock_qa.json", clock_qa)

    replay_kernel = _build(
        contract,
        scene_path,
        mimo_assets,
        output_dir / "replay_component.xml",
        clutter_layout,
    )
    replay_trace, _ = run_physics_trace(
        replay_kernel, contract, truth_hz=truth_hz, physics_hz=physics_hz
    )
    replay_path = output_dir / "episode_trace_replay.npz"
    np.savez_compressed(replay_path, **replay_trace)
    determinism_qa = write_determinism_receipt(
        trace_path,
        replay_path,
        output_dir / "determinism_qa.json",
        atol=float(contract["frozen_gates"]["replay_numeric_absolute_error_max"]),
    )

    render_qa = None
    appearance_qa = None
    mpfb_diagnostic = None
    if render:
        if (blender_path is None) != (mpfb_blend_path is None):
            raise ValueError(
                "provide both --blender and --mpfb-blend for optional MPFB diagnostics"
            )
        resolution = contract["embodiment"]["camera"]["resolution_px"]
        render_qa = render_trace(
            kernel,
            trace,
            output_dir,
            truth_hz=truth_hz,
            fps=int(contract["rates_hz"]["render"]),
            width=int(resolution[0]),
            height=int(resolution[1]),
            vertical_fov_degrees=float(
                contract["embodiment"]["camera"]["vertical_fov_degrees"]
            ),
        )
        render_qa["authoritative_video"] = "baseline_rgb.mp4"
        appearance_qa = _appearance_gate(contract, render_qa)
        write_json(output_dir / "appearance_qa.json", appearance_qa)
        if blender_path is not None and mpfb_blend_path is not None:
            receipt = _run_blender_overlay(
                repo_root=repo_root,
                blender_path=blender_path,
                blend_path=mpfb_blend_path,
                trace_path=trace_path,
                body_names_path=output_dir / "body_names.json",
                spec_path=copied_contract,
                output_dir=output_dir / "mpfb_diagnostic",
                width=int(resolution[0]),
                height=int(resolution[1]),
                fps=int(contract["rates_hz"]["render"]),
                all_frames=False,
            )
            mpfb_diagnostic = {
                "role": "diagnostic_only_not_phase_1_gate",
                "status": receipt["status"],
                "visibly_better_than_native": False,
                "selected_as_authoritative": False,
                "receipt": "mpfb_diagnostic/overlay_receipt.json",
                "maximum_landmark_error_m": receipt[
                    "maximum_landmark_error_m"
                ],
                "appearance_qualification": receipt[
                    "appearance_qualification"
                ],
            }
            write_json(output_dir / "mpfb_diagnostic_qa.json", mpfb_diagnostic)
        render_qa["mpfb_role"] = "diagnostic_only_not_phase_1_gate"
        write_json(output_dir / "render_qa.json", render_qa)

    qualification = {
        "schema": "EmbodiedVerticalSliceQualification",
        "scientific_scope": contract["scientific_scope"],
        "physics": _physics_gate(contract, physics_qa),
        "determinism": determinism_qa,
        "render": _render_gate(contract, render_qa) if render_qa else None,
        "appearance": appearance_qa,
    }
    qualification["passed"] = bool(
        qualification["physics"]["passed"]
        and determinism_qa["all_pass"]
        and (
            not render
            or (
                qualification["render"]["passed"]
                and qualification["appearance"]["passed"]
            )
        )
    )
    write_json(output_dir / "qualification.json", qualification)

    files = [
        "vertical_slice_contract.json",
        "kernel_component.xml",
        "replay_component.xml",
        "embodied_mimo_model.xml",
        "episode_trace.npz",
        "episode_trace_replay.npz",
        "body_names.json",
        "physics_qa.json",
        "shared_clock_qa.json",
        "determinism_qa.json",
        "qualification.json",
    ]
    if render:
        files.extend(
            [
                "baseline_rgb.mp4",
                "external_qa.mp4",
                "render_streams.h5",
                "render_qa.json",
                "inspection_sheet.png",
                "appearance_qa.json",
            ]
        )
        if mpfb_diagnostic is not None:
            files.extend(
                [
                    "mpfb_diagnostic_qa.json",
                    "mpfb_diagnostic/overlay_receipt.json",
                ]
            )
    regeneration_command = [
        "python",
        "-m",
        "babyworld_lite.childlens_engine_bakeoff.run_kernel_episode",
        str(contract_path),
        str(scene_path),
        str(mimo_assets),
        str(output_dir),
    ]
    if scene_variant is not None:
        regeneration_command.extend(["--scene-variant", variant_id])
    if blender_path is not None and mpfb_blend_path is not None:
        regeneration_command.extend(
            [
                "--blender",
                str(blender_path),
                "--mpfb-blend",
                str(mpfb_blend_path),
            ]
        )
    manifest = build_manifest(
        output_dir,
        files,
        spec_sha256=sha256_file(contract_path),
        provenance={
            **contract["provenance"],
            "scene_sha256": sha256_file(scene_path),
            "scene_variant": variant_id,
            "distractor_count": distractor_count,
            "mujoco_runtime": mujoco.__version__,
            "blender_executable_sha256": (
                sha256_file(blender_path) if blender_path else None
            ),
            "mpfb_generated_blend_sha256": (
                sha256_file(mpfb_blend_path) if mpfb_blend_path else None
            ),
            "private_childlens_material": False,
            "empirical_source": "ChildLens only; no restricted media accessed",
        },
        regeneration_command=regeneration_command,
    )
    write_json(output_dir / "episode_bundle_manifest.json", manifest)
    result = {
        "passed": qualification["passed"],
        "output_dir": str(output_dir),
        "scene_variant": variant_id,
        "distractor_count": distractor_count,
        "truth_samples": int(len(trace["time_s"])),
        "physics_qa": physics_qa,
        "determinism_qa": determinism_qa,
        "render_qa": render_qa,
        "appearance_qa": appearance_qa,
        "mpfb_diagnostic": mpfb_diagnostic,
        "qualification": qualification,
        "manifest": "episode_bundle_manifest.json",
    }
    write_json(output_dir / "qa_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("mimo_assets", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--scene-variant", choices=("sparse", "household", "messy"))
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--mpfb-blend", type=Path)
    args = parser.parse_args()
    result = run(
        args.contract,
        args.scene,
        args.mimo_assets,
        args.output_dir,
        render=not args.no_render,
        scene_variant=args.scene_variant,
        blender_path=args.blender,
        mpfb_blend_path=args.mpfb_blend,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
