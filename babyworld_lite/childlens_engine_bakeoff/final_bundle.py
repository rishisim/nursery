"""Assemble and verify the accepted Phase 6 embodied episode bundle."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from .bundle import build_manifest, sha256_file, write_json
from .run_kernel_episode import _assert_ignored


TRACE_SEMANTICS = {
    "time_s": "authoritative shared truth clock in episode seconds",
    "phase": "repository-authored controller phase active at the truth sample",
    "behavior_action": "bounded action-vocabulary label active at the truth sample",
    "qpos": "MuJoCo generalized position vector in embodied_mimo_model.xml order",
    "qvel": "MuJoCo generalized velocity vector in embodied_mimo_model.xml order",
    "action": "bounded root, arm, wrist, and finger controller target vector",
    "touch_wrench": "summed hand-target contact wrench [Fx,Fy,Fz,Tx,Ty,Tz]",
    "touch_contact_count": "number of simultaneous physical hand-target contacts",
    "distinct_finger_contacts": "number of distinct finger bodies contacting the target",
    "assist_active": "contact-gated grasp assist disclosure flag",
    "locomotion_assist_active": "collision-checked root-translation locomotion assist flag",
    "vestibular_accelerometer": "MuJoCo head vestibular-site linear acceleration",
    "vestibular_gyroscope": "MuJoCo head vestibular-site angular velocity",
    "vestibular_kinematic_accelerometer": "independent kinematic head-site linear acceleration check",
    "vestibular_kinematic_gyroscope": "independent kinematic head-site angular velocity check",
    "vestibular_position": "world position of the articulated head vestibular site",
    "target_pose": "persistent target world pose [x,y,z,qw,qx,qy,qz]",
    "hand_pose": "right-hand world pose [x,y,z,qw,qx,qy,qz]",
    "head_pose": "articulated head world pose [x,y,z,qw,qx,qy,qz]",
    "camera_pose": "head-camera world pose [x,y,z,row-major 3x3 rotation]",
    "body_pose": "MIMo body world poses [x,y,z,qw,qx,qy,qz] in body_names.json order",
    "reach_distractor_pose": "world position of the authored near-miss distractor",
    "touch_contact_position": "first physical hand-target MuJoCo contact point in world meters; NaN when absent",
    "touch_minimum_distance_m": "minimum signed physical hand-target contact distance",
    "support_contact_count": "number of target-support contacts",
    "support_contact_force_n": "summed target-support contact force magnitude",
    "minimum_relevant_contact_distance_m": "minimum signed distance among frozen relevant collision pairs",
    "minimum_body_environment_contact_distance_m": "minimum signed MIMo-body/environment contact distance",
    "near_miss_clearance_m": "signed hand/distractor clearance during the near-miss phase; NaN otherwise",
    "camera_mount_translation_error_m": "translation residual for T_world_camera = T_world_head * T_head_camera",
    "camera_mount_rotation_error_rad": "rotation residual for T_world_camera = T_world_head * T_head_camera",
    "world_reset": "hidden world-reset flag; must remain false",
}

RENDER_SEMANTICS = {
    "time_s": "authoritative render clock in episode seconds",
    "truth_index": "index into episode_trace.npz for the rendered frame",
    "depth_m": "authoritative head-camera metric depth in meters",
    "segmentation": "MuJoCo segmentation pair [geom_id, object_type_id] per pixel",
    "target_area_fraction": "fraction of pixels assigned to the persistent target",
    "clutter_area_fraction": "fraction of pixels assigned to authored clutter appearance geoms",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _verify_bound_path(repo_root: Path, binding: dict[str, str]) -> Path:
    path = repo_root / binding["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise ValueError(
            f"bound file changed: {binding['path']} expected "
            f"{binding['sha256']} got {actual}"
        )
    return path


def validate_manifest(bundle_dir: Path, manifest_name: str) -> dict[str, Any]:
    manifest = _read_json(bundle_dir / manifest_name)
    failures: list[str] = []
    for row in manifest["files"]:
        path = bundle_dir / row["path"]
        if not path.is_file():
            failures.append(f"missing:{row['path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            failures.append(f"bytes:{row['path']}")
        if sha256_file(path) != row["sha256"]:
            failures.append(f"sha256:{row['path']}")
    return {
        "passed": not failures,
        "file_count": len(manifest["files"]),
        "failures": failures,
    }


def _shape(value: Iterable[int]) -> list[int]:
    return [int(item) for item in value]


def describe_truth_bundle(
    trace_path: Path,
    render_path: Path,
    body_names_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    trace_streams: dict[str, Any] = {}
    with np.load(trace_path, allow_pickle=False) as trace:
        missing_trace = sorted(
            set(config["truth_contract"]["required_trace_streams"])
            - set(trace.files)
        )
        if missing_trace:
            raise ValueError(f"required trace streams missing: {missing_trace}")
        for name in trace.files:
            values = trace[name]
            trace_streams[name] = {
                "shape": _shape(values.shape),
                "dtype": str(values.dtype),
                "semantics": TRACE_SEMANTICS[name],
            }

    render_streams: dict[str, Any] = {}
    with h5py.File(render_path, "r") as streams:
        missing_render = sorted(
            set(config["truth_contract"]["required_render_streams"])
            - set(streams.keys())
        )
        if missing_render:
            raise ValueError(f"required render streams missing: {missing_render}")
        for name in sorted(streams.keys()):
            values = streams[name]
            render_streams[name] = {
                "shape": _shape(values.shape),
                "dtype": str(values.dtype),
                "compression": values.compression,
                "semantics": RENDER_SEMANTICS[name],
            }
        render_attributes = {
            str(name): (
                value.decode("utf-8") if isinstance(value, bytes) else value
            )
            for name, value in streams.attrs.items()
        }

    return {
        "schema": "EmbodiedSimulatorTruthSchema.v1",
        "clock": {
            "units": "episode_seconds",
            "physics_hz": config["truth_contract"]["physics_hz"],
            "truth_hz": config["truth_contract"]["truth_hz"],
            "render_hz": config["truth_contract"]["render_hz"],
            "render_to_truth_mapping": "render_streams.h5:truth_index",
        },
        "coordinate_conventions": {
            "world": "MuJoCo world coordinates; distances in meters",
            "poses": config["truth_contract"]["pose_convention"],
            "camera_mount": config["truth_contract"]["camera_mount_source"],
            "camera_equation": (
                "T_world_camera(t) = T_world_head(t) * T_head_camera"
            ),
        },
        "trace_file": trace_path.name,
        "trace_streams": trace_streams,
        "body_names": _read_json(body_names_path),
        "render_file": render_path.name,
        "render_attributes": render_attributes,
        "render_streams": render_streams,
        "rgb": {
            "authoritative": "baseline_rgb.mp4",
            "accepted_audio_mux": "accepted_episode.mp4",
            "external_qa": "external_qa.mp4 (diagnostic only)",
            "neural_appearance": None,
        },
        "speech": {
            "waveform": "speech.wav",
            "transcript": "transcript.txt",
            "utterance_alignment": "speech_alignment.json",
            "clock": "episode_seconds",
        },
        "persistent_object_identity": (
            "cell_vertical_contract.json:scene_family.target.persistent_id"
        ),
        "assist_disclosure": ["assist_active", "locomotion_assist_active"],
    }


def compose_preview(
    frames: list[np.ndarray],
    samples: list[dict[str, Any]],
    *,
    columns: int,
    thumbnail_px: tuple[int, int],
) -> Image.Image:
    if len(frames) != len(samples) or not frames:
        raise ValueError("preview frames and samples must be non-empty and aligned")
    thumb_width, thumb_height = thumbnail_px
    label_height = 30
    rows = int(math.ceil(len(frames) / columns))
    sheet = Image.new(
        "RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (frame, sample) in enumerate(zip(frames, samples)):
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_width, thumb_height), "black")
        x_pad = (thumb_width - image.width) // 2
        y_pad = (thumb_height - image.height) // 2
        canvas.paste(image, (x_pad, y_pad))
        column = index % columns
        row = index // columns
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        sheet.paste(canvas, (x, y))
        label = f"{float(sample['time_s']):05.1f}s  {sample['label']}"
        draw.text((x + 7, y + thumb_height + 7), label, fill="white")
    return sheet


def make_preview(
    video_path: Path, preview: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    metadata_row = iio.immeta(video_path, plugin="FFMPEG")
    fps = float(metadata_row["fps"])
    samples = preview["samples"]
    indices = [int(round(float(row["time_s"]) * fps)) for row in samples]
    by_index: dict[int, np.ndarray] = {}
    wanted = set(indices)
    for index, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
        if index in wanted:
            by_index[index] = np.asarray(frame)
        if index >= max(indices):
            break
    missing = sorted(wanted - set(by_index))
    if missing:
        raise ValueError(f"preview frames absent from video: {missing}")
    frames = [by_index[index] for index in indices]
    sheet = compose_preview(
        frames,
        samples,
        columns=int(preview["columns"]),
        thumbnail_px=tuple(preview["thumbnail_px"]),
    )
    sheet.save(output_path, format="PNG", optimize=True)
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "sample_count": len(samples),
        "fps": fps,
        "frame_indices": indices,
        "samples": samples,
    }


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _source_records(source_run: Path, batch_dir: Path) -> dict[str, Any]:
    return {
        "batch": _read_json(batch_dir / "batch_aggregate.json"),
        "batch_qualification": _read_json(
            batch_dir / "batch_qualification.json"
        ),
        "visual_distribution": _read_json(
            batch_dir / "visual_distribution_qa.json"
        ),
        "cell": _read_json(source_run / "cell_result.json"),
        "vertical": _read_json(source_run / "cell_vertical_contract.json"),
        "continuous": _read_json(
            source_run / "continuous_qualification.json"
        ),
        "qualification": _read_json(source_run / "qualification.json"),
        "physics": _read_json(source_run / "physics_qa.json"),
        "render": _read_json(source_run / "render_qa.json"),
        "appearance": _read_json(source_run / "appearance_qa.json"),
        "determinism": _read_json(source_run / "determinism_qa.json"),
        "shared_clock": _read_json(source_run / "shared_clock_qa.json"),
        "speech": _read_json(source_run / "speech_qa.json"),
        "cross_modal": _read_json(source_run / "cross_modal_qa.json"),
        "mux": _read_json(source_run / "mux_qa.json"),
        "source_manifest": _read_json(
            source_run / "continuous_episode_manifest.json"
        ),
    }


def acceptance_checks(
    config: dict[str, Any],
    records: dict[str, Any],
    trace: dict[str, np.ndarray],
    render_summary: dict[str, Any],
    *,
    bound_inputs_valid: bool,
    source_manifest_valid: bool,
    expected_artifacts_valid: bool,
) -> dict[str, bool]:
    truth = config["truth_contract"]
    acceptance = config["acceptance"]
    vertical_gates = records["vertical"]["frozen_gates"]
    batch_q = records["batch_qualification"]
    cell = records["cell"]
    physics = records["physics"]
    render = records["render"]
    qualification = records["qualification"]
    camera = physics["camera_mount"]
    collision = physics["collision_policy"]
    manipulation = physics["manipulation"]
    continuous = physics["continuous_episode"]
    near_miss = physics["near_miss"]
    imu = physics["imu_consistency"]
    required_trace = set(truth["required_trace_streams"])
    required_render = set(truth["required_render_streams"])
    render_shapes = render_summary["streams"]
    frame_count = int(truth["render_frames"])
    height, width = int(truth["resolution_px"][1]), int(
        truth["resolution_px"][0]
    )
    checks = {
        "bound_inputs_and_source_hashes": (
            bound_inputs_valid
            and source_manifest_valid
            and expected_artifacts_valid
        ),
        "frozen_phase5_selection": (
            batch_q["passed"]
            and batch_q["selected_cell_id"]
            == config["source_selection"]["cell_id"]
            and cell["cell_id"] == config["source_selection"]["cell_id"]
            and cell["passed"]
        ),
        "generalization_operating_envelope": (
            batch_q["total_episode_pass_count"]
            >= acceptance["required_batch_passes"]
            and batch_q["required_scene_pass_count"]
            >= acceptance["required_scene_passes"]
            and batch_q["bounded_failure_count"]
            <= acceptance["maximum_bounded_batch_failures"]
        ),
        "continuous_qualified_trace": (
            records["continuous"]["passed"]
            and continuous["single_physics_trace"]
            and continuous["hidden_world_reset_count"]
            <= acceptance["hidden_world_reset_count_max"]
            and not np.asarray(trace["world_reset"]).any()
        ),
        "truth_clock_and_streams": (
            records["shared_clock"]["passed"]
            and required_trace <= set(trace)
            and len(trace["time_s"]) == truth["truth_samples"]
            and abs(float(trace["time_s"][0])) <= 1e-15
            and abs(float(trace["time_s"][-1]) - truth["duration_s"])
            <= records["vertical"]["frozen_gates"][
                "stream_timestamp_error_s_max"
            ]
        ),
        "immutable_embodied_head_camera": (
            records["vertical"]["embodiment"]["camera_mount"]["immutable"]
            and not physics["independent_camera_control"]
            and camera["immutable"]
            and camera["maximum_translation_error_m"]
            <= vertical_gates["camera_mount_translation_error_m_max"]
            and camera["maximum_rotation_error_rad"]
            <= vertical_gates["camera_mount_rotation_error_rad_max"]
        ),
        "physical_authority_and_static_collisions": (
            not physics["hand_mocap_or_weld"]
            and not physics["direct_target_transform_after_initialization"]
            and collision["all_relevant_enabled"]
            and collision["unchanged"]
            and collision["initial_sha256"] == collision["final_sha256"]
            and physics["grasp"]["collisions_remained_enabled"]
        ),
        "penetration_gate": (
            collision["persistent_penetration_frames"]
            <= vertical_gates["persistent_penetration_frames_max"]
            and collision["minimum_relevant_contact_distance_m"]
            >= -vertical_gates["penetration_depth_m_max"]
        ),
        "near_miss_gate": (
            near_miss["contact_substeps"]
            <= vertical_gates["near_miss_contact_count_max"]
            and near_miss["minimum_clearance_m"]
            >= vertical_gates["near_miss_clearance_m_min"]
        ),
        "contact_release_and_projection": (
            render["rgb_truth_contact_frame_offset"]
            <= vertical_gates["rgb_truth_event_frame_offset_max"]
            and render["rgb_truth_release_frame_offset"]
            <= vertical_gates["rgb_truth_event_frame_offset_max"]
            and render["maximum_projected_contact_error_px"]
            <= vertical_gates["visible_contact_alignment_px_max"]
            and render["maximum_3d_contact_surface_distance_m"]
            <= vertical_gates["visible_contact_alignment_m_max"]
            and physics["contact"]["maximum_distinct_finger_contacts"] >= 2
            and manipulation["released"]
        ),
        "object_identity_and_manipulation": (
            physics["object_identity"]["changes"]
            <= vertical_gates["object_identity_changes_max"]
            and physics["object_identity"]["persistent_id"]
            == config["source_selection"]["target_persistent_id"]
            and manipulation["maximum_lift_m"]
            >= vertical_gates["minimum_lift_m"]
            and manipulation["maximum_object_rotation_degrees"]
            >= vertical_gates["minimum_rotation_degrees"]
        ),
        "imu_matches_head_kinematics": (
            imu["accelerometer_rmse_m_s2"]
            <= vertical_gates["imu_linear_acceleration_rmse_m_s2_max"]
            and imu["gyroscope_rmse_rad_s"]
            <= vertical_gates["imu_angular_velocity_rmse_rad_s_max"]
        ),
        "exact_deterministic_replay": (
            records["determinism"]["all_pass"]
            and records["determinism"]["maximum_numeric_absolute_error"] == 0.0
            and records["determinism"]["first_trace_sha256"]
            == records["determinism"]["second_trace_sha256"]
            == config["source_selection"]["expected_artifacts"][
                "episode_trace.npz"
            ]
        ),
        "render_streams_synchronized": (
            required_render <= set(render_shapes)
            and render_shapes["depth_m"]["shape"]
            == [frame_count, height, width]
            and render_shapes["segmentation"]["shape"]
            == [frame_count, height, width, 2]
            and render_summary["clock_matches_trace"]
        ),
        "authoritative_native_appearance": (
            qualification["appearance"]["passed"]
            and records["appearance"]["passed"]
            and config["authoritative_appearance"]["neural_appearance_selected"]
            is False
            and render["collision_proxy_pixels"]
            <= vertical_gates["collision_proxy_pixels_max"]
            and render["skin_artifact_pixels"]
            <= vertical_gates["skin_artifact_pixels_max"]
        ),
        "assist_disclosure": (
            cell["unflagged_assist_frames"]
            <= acceptance["unflagged_assist_frames_max"]
            and int(np.asarray(trace["assist_active"]).sum())
            == cell["assist_active_truth_frames"]
            and physics["locomotion_assist"]["explicitly_labeled"]
            and physics["locomotion_assist"]["collision_checked"]
        ),
        "authoritative_speech_and_alignment": (
            records["speech"]["duration_s"] == truth["duration_s"]
            and records["speech"]["clipped_samples"] == 0
            and records["speech"]["neural_render_audio_used"] is False
            and records["cross_modal"]["all_utterances_event_aligned"]
            and records["cross_modal"][
                "target_visible_at_all_utterance_starts"
            ]
            and records["mux"]["video_streams"] == 1
            and records["mux"]["audio_streams"] == 1
            and records["mux"]["neural_render_audio_used"] is False
        ),
        "qualification_receipts_pass": (
            qualification["passed"]
            and qualification["physics"]["passed"]
            and qualification["render"]["passed"]
            and qualification["appearance"]["passed"]
            and qualification["determinism"]["all_pass"]
        ),
        "governance_boundaries_preserved": (
            config["privacy_and_governance"][
                "restricted_childlens_access_permitted"
            ]
            is False
            and config["privacy_and_governance"][
                "restricted_childlens_remote_transfer_permitted"
            ]
            is False
            and config["privacy_and_governance"]["public_synthetic_inputs_only"]
            and acceptance["frozen_thresholds_changed"] is False
        ),
    }
    return {name: bool(value) for name, value in checks.items()}


def gate_from_checks(checks: dict[str, bool]) -> dict[str, Any]:
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failures,
        "gate_decision": "PASS" if not failures else "STOP",
        "failed_checks": failures,
        "check_count": len(checks),
    }


def _render_summary(
    render_path: Path, trace: dict[str, np.ndarray]
) -> dict[str, Any]:
    streams: dict[str, Any] = {}
    with h5py.File(render_path, "r") as h5:
        for name in h5:
            streams[name] = {
                "shape": _shape(h5[name].shape),
                "dtype": str(h5[name].dtype),
            }
        indices = h5["truth_index"][:]
        times = h5["time_s"][:]
    return {
        "streams": streams,
        "clock_matches_trace": bool(
            np.array_equal(times, np.asarray(trace["time_s"])[indices])
        ),
        "truth_index_min": int(indices.min()),
        "truth_index_max": int(indices.max()),
    }


def _external_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _environment_provenance(
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
    records: dict[str, Any],
) -> dict[str, Any]:
    source_provenance = records["source_manifest"]["provenance"]
    external = {
        "MIMo": {
            "path": ".external/mimo-pin",
            "commit": _external_revision(repo_root / ".external/mimo-pin"),
        },
        "MolmoSpaces": {
            "path": ".external/molmospaces-pin",
            "commit": _external_revision(
                repo_root / ".external/molmospaces-pin"
            ),
        },
        "MPFB_diagnostic_only": {
            "path": ".external/mpfb-pin",
            "commit": _external_revision(repo_root / ".external/mpfb-pin"),
        },
    }
    expected_revisions = {
        "MIMo": source_provenance["MIMo"]["commit"],
        "MolmoSpaces": source_provenance["MolmoSpaces"]["commit"],
        "MPFB_diagnostic_only": source_provenance["MPFB"]["commit"],
    }
    if any(
        external[name]["commit"] != expected_revisions[name]
        for name in expected_revisions
    ):
        raise ValueError("local public source revision differs from bound provenance")
    scene = (
        repo_root
        / ".external/molmospaces-cache/scenes/ithor/20251217_with_occupancy/FloorPlan201_physics.xml"
    )
    if sha256_file(scene) != source_provenance["scene_sha256"]:
        raise ValueError("local furnished scene differs from bound provenance")
    return {
        "schema": "EmbodiedAssetDependencyProvenance.v1",
        "frozen_contract_sha256": sha256_file(config_path),
        "repository": {
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "branch", "--show-current"),
            "remote": _git(repo_root, "remote", "get-url", "origin"),
        },
        "execution_environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "mujoco": _package_version("mujoco"),
            "h5py": _package_version("h5py"),
            "imageio": _package_version("imageio"),
            "Pillow": _package_version("Pillow"),
            "cuda_used": False,
        },
        "source_provenance_and_licenses": source_provenance,
        "verified_external_source_revisions": external,
        "verified_scene": {
            "path": str(scene.relative_to(repo_root)),
            "sha256": sha256_file(scene),
        },
        "bound_inputs": config["source_selection"]["bindings"],
        "seeds": records["vertical"]["seeds"],
        "appearance": config["authoritative_appearance"],
        "privacy_and_governance": config["privacy_and_governance"],
    }


def _replay_instructions(
    provenance: dict[str, Any], config: dict[str, Any]
) -> str:
    commit = provenance["repository"]["commit"]
    expected = config["source_selection"]["expected_artifacts"]
    return f"""# Exact replay instructions

This bundle was assembled from repository commit `{commit}` on branch
`{provenance['repository']['branch']}`. Use only public/synthetic inputs. Do not
access, decode, copy, move, or upload restricted ChildLens media.

1. Check out commit `{commit}` and recreate the package versions and public
   source revisions in `asset_dependency_provenance.json`.
2. Verify the furnished scene and public source hashes recorded there.
3. Confirm the disposable destination is ignored:

   `git check-ignore runs/embodied_simulation/phase_6/replay`

4. From the repository root, run:

   `.venv-embodied/bin/python -m babyworld_lite.childlens_engine_bakeoff.run_continuous_episode runs/embodied_simulation/phase_6/final/cell_episode_contract.json runs/embodied_simulation/phase_6/final/cell_vertical_contract.json .external/molmospaces-cache/scenes/ithor/20251217_with_occupancy/FloorPlan201_physics.xml .external/mimo-pin/mimoEnv/assets runs/embodied_simulation/phase_6/replay`

5. Verify exact replay outputs:

   - `episode_trace.npz`: `{expected['episode_trace.npz']}`
   - `episode_trace_replay.npz`: `{expected['episode_trace_replay.npz']}`
   - `baseline_rgb.mp4`: `{expected['baseline_rgb.mp4']}`
   - `accepted_episode.mp4`: `{expected['accepted_episode.mp4']}`
   - `render_streams.h5`: `{expected['render_streams.h5']}`
   - `speech.wav`: `{expected['speech.wav']}`

The physics trace is byte-identical across the recorded replay. If a different
renderer, codec, speech executable, OS, or dependency build changes media bytes,
that environment is outside this exact-replay receipt; do not reinterpret the
frozen scientific thresholds.
"""


def _decision_report(
    config: dict[str, Any],
    records: dict[str, Any],
    checks: dict[str, bool],
    trace: dict[str, np.ndarray],
) -> dict[str, Any]:
    gate = gate_from_checks(checks)
    cell = records["cell"]
    physics = records["physics"]
    visual = cell["visual_distribution"]
    return {
        "schema": "EmbodiedSimulationFinalDecision.v1",
        "phase": 6,
        "gate_decision": gate["gate_decision"],
        "selected_cell_id": cell["cell_id"],
        "selection_policy": config["source_selection"]["policy"],
        "accepted_artifacts": {
            "episode_with_authoritative_speech": "accepted_episode.mp4",
            "authoritative_rgb": "baseline_rgb.mp4",
            "truth_trace": "episode_trace.npz",
            "depth_and_segmentation": "render_streams.h5",
            "qa_preview": "qa_preview.png",
            "neural_appearance": None,
        },
        "engineering_qualification": {
            "status": "PASS" if gate["passed"] else "FAIL",
            "continuous_trace": True,
            "duration_s": physics["duration_s"],
            "exact_replay_maximum_numeric_error": records["determinism"][
                "maximum_numeric_absolute_error"
            ],
            "camera_mount_max_translation_error_m": physics["camera_mount"][
                "maximum_translation_error_m"
            ],
            "camera_mount_max_rotation_error_rad": physics["camera_mount"][
                "maximum_rotation_error_rad"
            ],
            "persistent_penetration_frames": physics["collision_policy"][
                "persistent_penetration_frames"
            ],
            "object_identity_changes": physics["object_identity"]["changes"],
            "batch_passes": records["batch_qualification"][
                "total_episode_pass_count"
            ],
            "batch_total": records["batch"]["episode_count"],
        },
        "assistance_disclosure": {
            "grasp_assist_intervals": physics["grasp"]["assist_intervals"],
            "grasp_assist_truth_frames": int(
                np.asarray(trace["assist_active"]).sum()
            ),
            "grasp_assist_duration_s": cell["assist_duration_s"],
            "unflagged_assist_frames": cell["unflagged_assist_frames"],
            "locomotion_assist_truth_frames": int(
                np.asarray(trace["locomotion_assist_active"]).sum()
            ),
            "interpretation": (
                "This accepted engineering trace is assist-heavy and is not an "
                "unassisted dexterity result."
            ),
        },
        "hardware_privacy_status": {
            "status": "STOP_HARDWARE_PRIVACY_NO_LOCAL_CUDA",
            "restricted_childlens_accessed": False,
            "restricted_childlens_transferred": False,
        },
        "measurement_validity": {
            "status": "NOT_ESTABLISHED",
            "childlens_scoring_performed": False,
            "synthetic_instrument_checks_substitute_for_childlens_validity": False,
        },
        "naturalistic_calibration": {
            "status": "TARGET_INTERVALS_NOT_MET",
            "empirical_source": "ChildLens aggregate intervals only",
            "age_scope": "3-5 years; provisional young-child bridge only",
            "motion": visual["motion"],
            "adjacent_frame_persistence": visual[
                "adjacent_frame_persistence"
            ],
            "scene_change_rate": visual["scene_change_rate"],
            "infant_calibration": False,
            "human_validation": False,
        },
        "appearance_validity": {
            "status": "DETERMINISTIC_BASELINE_ONLY",
            "authoritative_rgb": "native MIMo co-articulated appearance",
            "neural_result_accepted": False,
            "mpfb_role": "diagnostic only",
        },
        "causal_evidence": {
            "status": "NOT_PRODUCED",
            "learner_comparison_run": False,
            "causal_claim": False,
        },
        "operating_envelope": {
            "passes": records["batch_qualification"][
                "total_episode_pass_count"
            ],
            "episodes": records["batch"]["episode_count"],
            "bounded_failures": records["batch_qualification"][
                "bounded_failure_count"
            ],
            "known_boundary": (
                "red_ball_authored at seed 20260731 fails the second retrieve "
                "lift reproducibly across sparse, household, and messy scenes"
            ),
        },
        "failed_acceptance_checks": gate["failed_checks"],
    }


def assemble(
    config_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    config = _read_json(config_path)
    expected_output = repo_root / config["bundle"]["output_root"]
    output_dir = (output_dir or expected_output).resolve()
    if output_dir != expected_output.resolve():
        raise ValueError("Phase 6 output must use the frozen ignored output root")
    _assert_ignored(repo_root, output_dir)

    for binding in config["source_selection"]["bindings"].values():
        _verify_bound_path(repo_root, binding)
    source_run = repo_root / config["source_selection"]["source_run"]
    batch_dir = repo_root / "runs/embodied_simulation/phase_5/batch"
    source_manifest_receipt = validate_manifest(
        source_run, "continuous_episode_manifest.json"
    )
    if not source_manifest_receipt["passed"]:
        raise ValueError(
            f"selected source manifest failed: {source_manifest_receipt['failures']}"
        )
    for name, expected in config["source_selection"][
        "expected_artifacts"
    ].items():
        actual = sha256_file(source_run / name)
        if actual != expected:
            raise ValueError(
                f"selected artifact changed: {name} expected {expected} got {actual}"
            )

    records = _source_records(source_run, batch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in config["bundle"]["copy_selected_files"]:
        source = source_run / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)
    for source_name, destination_name in config["bundle"][
        "copy_bound_records"
    ].items():
        shutil.copy2(repo_root / source_name, output_dir / destination_name)
    shutil.copy2(config_path, output_dir / "final_bundle_contract.json")

    truth_schema = describe_truth_bundle(
        output_dir / "episode_trace.npz",
        output_dir / "render_streams.h5",
        output_dir / "body_names.json",
        config,
    )
    write_json(output_dir / "truth_schema.json", truth_schema)
    preview = make_preview(
        output_dir / "baseline_rgb.mp4",
        config["preview"],
        output_dir / "qa_preview.png",
    )
    provenance = _environment_provenance(
        repo_root, config_path, config, records
    )
    write_json(output_dir / "asset_dependency_provenance.json", provenance)
    (output_dir / "replay_instructions.md").write_text(
        _replay_instructions(provenance, config), encoding="utf-8"
    )

    with np.load(output_dir / "episode_trace.npz", allow_pickle=False) as loaded:
        trace = {name: loaded[name] for name in loaded.files}
    render_summary = _render_summary(
        output_dir / "render_streams.h5", trace
    )
    checks = acceptance_checks(
        config,
        records,
        trace,
        render_summary,
        bound_inputs_valid=True,
        source_manifest_valid=source_manifest_receipt["passed"],
        expected_artifacts_valid=True,
    )
    gate = gate_from_checks(checks)
    final_qa = {
        "schema": "EmbodiedSimulationFinalAcceptanceQA.v1",
        **gate,
        "checks": checks,
        "selected_cell_id": config["source_selection"]["cell_id"],
        "source_manifest": source_manifest_receipt,
        "trace_sha256": sha256_file(output_dir / "episode_trace.npz"),
        "replay_trace_sha256": sha256_file(
            output_dir / "episode_trace_replay.npz"
        ),
        "accepted_episode_sha256": sha256_file(
            output_dir / "accepted_episode.mp4"
        ),
        "baseline_rgb_sha256": sha256_file(
            output_dir / "baseline_rgb.mp4"
        ),
        "render_streams_sha256": sha256_file(
            output_dir / "render_streams.h5"
        ),
        "render_summary": render_summary,
        "preview": preview,
        "frozen_thresholds_changed": False,
        "restricted_childlens_accessed": False,
    }
    write_json(output_dir / "final_acceptance_qa.json", final_qa)
    decision = _decision_report(config, records, checks, trace)
    write_json(output_dir / "final_decision_report.json", decision)
    if not gate["passed"]:
        raise ValueError(f"Phase 6 acceptance gate failed: {gate['failed_checks']}")

    manifest_files = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "final_bundle_manifest.json"
    )
    expected_names = set(config["bundle"]["copy_selected_files"])
    expected_names.update(config["bundle"]["copy_bound_records"].values())
    expected_names.update(config["bundle"]["generated_files"])
    expected_names.remove("final_bundle_manifest.json")
    if set(manifest_files) != expected_names:
        raise ValueError(
            "final output contains missing or unexpected files: "
            f"missing={sorted(expected_names - set(manifest_files))}, "
            f"unexpected={sorted(set(manifest_files) - expected_names)}"
        )
    manifest = build_manifest(
        output_dir,
        manifest_files,
        spec_sha256=sha256_file(config_path),
        provenance={
            "repository_commit": provenance["repository"]["commit"],
            "selected_cell_id": config["source_selection"]["cell_id"],
            "source_continuous_manifest_sha256": config["source_selection"][
                "bindings"
            ]["selected_continuous_manifest"]["sha256"],
            "authoritative_rgb": "baseline_rgb.mp4",
            "neural_appearance_selected": False,
            "restricted_childlens_accessed": False,
        },
        regeneration_command=[
            ".venv-embodied/bin/python",
            "-m",
            "babyworld_lite.childlens_engine_bakeoff.final_bundle",
            str(config_path.relative_to(repo_root)),
        ],
    )
    manifest["schema"] = "EmbodiedSimulationFinalBundleManifest.v1"
    write_json(output_dir / "final_bundle_manifest.json", manifest)
    manifest_receipt = validate_manifest(output_dir, "final_bundle_manifest.json")
    if not manifest_receipt["passed"]:
        raise ValueError(
            f"final bundle manifest failed: {manifest_receipt['failures']}"
        )
    return {
        "passed": True,
        "gate_decision": "PASS",
        "output_dir": str(output_dir.relative_to(repo_root)),
        "file_count": manifest_receipt["file_count"],
        "bundle_manifest_sha256": sha256_file(
            output_dir / "final_bundle_manifest.json"
        ),
        "accepted_episode_sha256": final_qa["accepted_episode_sha256"],
        "trace_sha256": final_qa["trace_sha256"],
        "preview_sha256": preview["sha256"],
        "acceptance_check_count": gate["check_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("configs/embodied_simulation_final_bundle.json"),
    )
    args = parser.parse_args()
    print(json.dumps(assemble(args.config), indent=2))


if __name__ == "__main__":
    main()
