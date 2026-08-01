"""Compile and qualify one continuous prompt-to-episode candidate."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .bundle import build_manifest, sha256_file, write_json
from .compile_episode import compose_embodied_prompt_plan
from .run_kernel_episode import _assert_ignored, run as run_kernel_episode
from .speech_audio import generate_authoritative_speech


def _execution_spec(
    episode: dict[str, Any],
    vertical: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    resolved = copy.deepcopy(vertical)
    resolved["schema"] = "EmbodiedSimulationContinuousExecutionSpec.v1"
    resolved["activity"] = copy.deepcopy(episode["activity"])
    resolved["controller"] = copy.deepcopy(episode["controller"])
    resolved["language"] = copy.deepcopy(episode["language"])
    resolved["continuous_episode"] = {
        "prompt": episode["prompt"],
        "scene_variant": episode["scene_variant"],
        "plan_sha256": plan["plan_sha256"],
        "selection_policy": episode["selection_policy"],
        "privacy": episode["privacy"],
        "vertical_slice_hard_gates_changed": False,
    }
    return resolved


def _mux_authoritative_audio(
    video_path: Path, audio_path: Path, output_path: Path
) -> dict[str, Any]:
    ffmpeg = Path("/opt/homebrew/bin/ffmpeg")
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-metadata:s:a:0",
            "title=authoritative separately generated speech",
            str(output_path),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "/opt/homebrew/bin/ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,nb_frames",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    result = json.loads(probe.stdout)
    streams = result["streams"]
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "duration_s": float(result["format"]["duration"]),
        "video_streams": sum(row["codec_type"] == "video" for row in streams),
        "audio_streams": sum(row["codec_type"] == "audio" for row in streams),
        "video_codec": next(
            row["codec_name"] for row in streams if row["codec_type"] == "video"
        ),
        "audio_codec": next(
            row["codec_name"] for row in streams if row["codec_type"] == "audio"
        ),
        "video_frames": int(
            next(
                row["nb_frames"]
                for row in streams
                if row["codec_type"] == "video"
            )
        ),
        "neural_render_audio_used": False,
    }


def _cross_modal_qa(
    episode: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    trace = np.load(output_dir / "episode_trace.npz")
    speech = json.loads(
        (output_dir / "speech_alignment.json").read_text(encoding="utf-8")
    )
    with h5py.File(output_dir / "render_streams.h5", "r") as streams:
        render_times = streams["time_s"][:]
        target_area = streams["target_area_fraction"][:]
    rows = []
    for utterance in speech["utterances"]:
        truth_index = int(
            np.argmin(np.abs(trace["time_s"] - float(utterance["start_s"])))
        )
        frame_index = int(
            np.argmin(np.abs(render_times - float(utterance["start_s"])))
        )
        observed_phase = str(trace["phase"][truth_index])
        expected = utterance["event"]
        event_match = (
            observed_phase == expected
            or (
                expected == "shake_bang_transfer"
                and observed_phase in {"shake", "bang", "transfer"}
            )
        )
        rows.append(
            {
                "utterance_id": utterance["id"],
                "start_s": utterance["start_s"],
                "expected_event": expected,
                "observed_behavior_phase": observed_phase,
                "event_match": event_match,
                "target_area_fraction": float(target_area[frame_index]),
                "target_visible": bool(target_area[frame_index] > 0.0),
            }
        )
    receipt = {
        "schema": "ContinuousEpisodeCrossModalQA.v1",
        "clock": "episode_seconds",
        "utterance_rows": rows,
        "all_utterances_event_aligned": all(row["event_match"] for row in rows),
        "target_visible_at_all_utterance_starts": all(
            row["target_visible"] for row in rows
        ),
        "authoritative_speech_source": "speech.wav",
        "authoritative_video_source": "baseline_rgb.mp4",
        "neural_audio_removed_or_absent": True,
    }
    write_json(output_dir / "cross_modal_qa.json", receipt)
    return receipt


def _continuous_gate(
    episode: dict[str, Any],
    plan: dict[str, Any],
    kernel_result: dict[str, Any],
    speech_qa: dict[str, Any] | None,
    cross_modal_qa: dict[str, Any] | None,
    mux_qa: dict[str, Any] | None,
    *,
    render: bool,
    speech: bool,
) -> dict[str, Any]:
    thresholds = episode["qualification"]
    physics = kernel_result["physics_qa"]
    continuous = physics["continuous_episode"]
    allowed = set(episode["planner"]["allowed_actions"])
    checks = {
        "bounded_plan": (
            not plan["planner_emitted_joint_trajectories"]
            and not plan["planner_emitted_camera_trajectory"]
            and not plan["planner_emitted_object_trajectory"]
            and {row["action"] for row in plan["actions"]} <= allowed
        ),
        "scene_resolved_before_execution": (
            plan["scene_resolution"]["status"] == "resolved_before_execution"
        ),
        "continuous_duration_and_clock": (
            abs(
                physics["duration_s"]
                - float(episode["activity"]["duration_s"])
            )
            <= thresholds["duration_absolute_error_s_max"]
            and kernel_result["truth_samples"] == thresholds["truth_samples"]
            and continuous["single_physics_trace"]
            and continuous["hidden_world_reset_count"]
            <= thresholds["hidden_world_reset_count_max"]
            and continuous["maximum_target_truth_step_m"]
            <= thresholds["maximum_truth_discontinuity_m"]
        ),
        "vertical_slice_hard_gates": kernel_result["qualification"]["physics"][
            "passed"
        ],
        "deterministic_replay": kernel_result["determinism_qa"]["all_pass"],
        "shake_completed": continuous["shake_vertical_range_m"]
        >= thresholds["minimum_shake_vertical_range_m"],
        "bang_completed": continuous["bang_support_contact_samples"]
        >= thresholds["minimum_bang_support_contact_samples"],
        "transfer_completed": continuous["transfer_lateral_range_m"]
        >= thresholds["minimum_transfer_lateral_range_m"],
        "release_settle_completed": (
            physics["manipulation"]["released"]
            and physics["manipulation"]["final_settled_duration_s"]
            >= 0.75
        ),
        "retrieve_completed": (
            continuous["retrieve_contact_samples"] > 0
            and continuous["retrieve_lift_m"]
            >= thresholds["minimum_retrieve_lift_m"]
            and continuous["assist_interval_count"] >= 2
        ),
        "object_identity": physics["object_identity"]["changes"] == 0,
    }
    if render:
        checks["authoritative_render"] = bool(
            kernel_result["qualification"]["render"]["passed"]
            and kernel_result["qualification"]["appearance"]["passed"]
            and kernel_result["render_qa"]["frames"]
            == thresholds["render_frames"]
        )
    if speech:
        checks["authoritative_speech"] = bool(
            speech_qa
            and thresholds["utterance_count_min"]
            <= speech_qa["utterance_count"]
            <= thresholds["utterance_count_max"]
            and speech_qa["duration_s"] == episode["activity"]["duration_s"]
            and speech_qa["clipped_samples"] == 0
            and not speech_qa["neural_render_audio_used"]
        )
    if render and speech:
        checks["cross_modal_and_mux"] = bool(
            cross_modal_qa
            and cross_modal_qa["all_utterances_event_aligned"]
            and cross_modal_qa["target_visible_at_all_utterance_starts"]
            and mux_qa
            and mux_qa["video_streams"] == 1
            and mux_qa["audio_streams"] == 1
            and not mux_qa["neural_render_audio_used"]
        )
    return {
        "schema": "ContinuousEpisodeQualification.v1",
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(
    episode_path: Path,
    vertical_contract_path: Path,
    scene_path: Path,
    mimo_assets: Path,
    output_dir: Path,
    *,
    render: bool = True,
    speech: bool = True,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    _assert_ignored(repo_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    vertical = json.loads(vertical_contract_path.read_text(encoding="utf-8"))
    plan = compose_embodied_prompt_plan(episode, vertical)
    plan_path = output_dir / "activity_language_plan.json"
    write_json(plan_path, plan)
    resolved = _execution_spec(episode, vertical, plan)
    resolved_path = output_dir / "resolved_episode_contract.json"
    write_json(resolved_path, resolved)

    kernel_result = run_kernel_episode(
        resolved_path,
        scene_path,
        mimo_assets,
        output_dir,
        render=render,
        scene_variant=episode["scene_variant"],
    )
    speech_qa = (
        generate_authoritative_speech(episode, output_dir) if speech else None
    )
    cross_modal_qa = (
        _cross_modal_qa(episode, output_dir) if render and speech else None
    )
    mux_qa = None
    if render and speech:
        mux_qa = _mux_authoritative_audio(
            output_dir / "baseline_rgb.mp4",
            output_dir / "speech.wav",
            output_dir / "accepted_episode.mp4",
        )
        write_json(output_dir / "mux_qa.json", mux_qa)

    qualification = _continuous_gate(
        episode,
        plan,
        kernel_result,
        speech_qa,
        cross_modal_qa,
        mux_qa,
        render=render,
        speech=speech,
    )
    write_json(output_dir / "continuous_qualification.json", qualification)
    files = [
        "activity_language_plan.json",
        "resolved_episode_contract.json",
        "vertical_slice_contract.json",
        "episode_trace.npz",
        "episode_trace_replay.npz",
        "body_names.json",
        "physics_qa.json",
        "shared_clock_qa.json",
        "determinism_qa.json",
        "qualification.json",
        "qa_report.json",
        "episode_bundle_manifest.json",
        "continuous_qualification.json",
    ]
    if render:
        files.extend(
            [
                "baseline_rgb.mp4",
                "external_qa.mp4",
                "render_streams.h5",
                "render_qa.json",
                "appearance_qa.json",
                "inspection_sheet.png",
            ]
        )
    if speech:
        files.extend(
            [
                "speech.wav",
                "transcript.txt",
                "speech_alignment.json",
                "speech_qa.json",
            ]
        )
    if render and speech:
        files.extend(
            ["accepted_episode.mp4", "cross_modal_qa.json", "mux_qa.json"]
        )
    manifest = build_manifest(
        output_dir,
        files,
        spec_sha256=sha256_file(episode_path),
        provenance={
            **vertical["provenance"],
            "vertical_contract_sha256": sha256_file(vertical_contract_path),
            "scene_sha256": sha256_file(scene_path),
            "scene_variant": episode["scene_variant"],
            "speech_engine": episode["language"]["speech"],
            "neural_appearance_selected": False,
            "private_childlens_material": False,
            "empirical_source": "ChildLens only; no restricted media accessed",
        },
        regeneration_command=[
            "python",
            "-m",
            "babyworld_lite.childlens_engine_bakeoff.run_continuous_episode",
            str(episode_path),
            str(vertical_contract_path),
            str(scene_path),
            str(mimo_assets),
            str(output_dir),
        ],
    )
    write_json(output_dir / "continuous_episode_manifest.json", manifest)
    result = {
        "passed": qualification["passed"],
        "output_dir": str(output_dir),
        "plan_sha256": plan["plan_sha256"],
        "kernel": kernel_result,
        "speech_qa": speech_qa,
        "cross_modal_qa": cross_modal_qa,
        "mux_qa": mux_qa,
        "qualification": qualification,
        "manifest": "continuous_episode_manifest.json",
    }
    write_json(output_dir / "continuous_qa_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("vertical_contract", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("mimo_assets", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-speech", action="store_true")
    args = parser.parse_args()
    result = run(
        args.episode,
        args.vertical_contract,
        args.scene,
        args.mimo_assets,
        args.output_dir,
        render=not args.no_render,
        speech=not args.no_speech,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
