"""Run and package one ResolvedEpisodeSpec through the bounded prototype kernel."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .bundle import build_manifest, validate_shared_clock, write_json
from .camera_sensitivity import run as run_camera_sensitivity
from .physics_kernel import build_kernel_model, run_physics_trace, value
from .staging import select_free_candidate
from .trace_render import render_trace


def _target_definition(spec: dict) -> dict:
    return {
        "geometry": spec["target"]["geometry"]["value"],
        "rgba": spec["target"]["rgba"]["value"],
    }


def synthesize_speech(spec: dict, output_dir: Path) -> tuple[Path, Path]:
    events = value(spec, "speech_events")
    event_path = output_dir / "speech_events.json"
    write_json(
        event_path,
        {
            "clock": "episode_seconds",
            "events": events,
            "claim_boundary": (
                "local synthetic waveform/timing interface only; not human "
                "validation, natural German ground truth, or language quality evidence"
            ),
        },
    )
    if not events:
        raise ValueError("resolved episode has no speech events")
    utterance_paths = []
    for index, event in enumerate(events):
        utterance = output_dir / f"speech_utterance_{index:02d}.aiff"
        subprocess.run(
            ["say", "-v", "Anna", "-o", str(utterance), event["text"]],
            check=True,
        )
        utterance_paths.append(utterance)
    waveform = output_dir / "speech_waveform.wav"
    command = ["ffmpeg", "-y"]
    for utterance in utterance_paths:
        command.extend(["-i", str(utterance)])
    filters = []
    mix_inputs = []
    for index, event in enumerate(events):
        delay_ms = int(round(float(event["start_s"]) * 1000))
        filters.append(f"[{index}:a]adelay={delay_ms}|{delay_ms}[a{index}]")
        mix_inputs.append(f"[a{index}]")
    filters.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(events)}:normalize=0,apad=pad_dur="
        + str(value(spec, "duration_s"))
        + "[out]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            str(value(spec, "duration_s")),
            "-ar",
            "48000",
            "-ac",
            "1",
            str(waveform),
        ]
    )
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return waveform, event_path


def run(
    spec_path: Path,
    intent_path: Path,
    scene_path: Path,
    map_path: Path,
    mimo_assets: Path,
    output_dir: Path,
    *,
    render: bool = True,
) -> dict:
    from molmo_spaces.utils.scene_maps import iTHORMap

    output_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    if scene_path.stem.removesuffix("_physics") != spec["scene"]["id"]["value"]:
        raise ValueError("scene path does not match ResolvedEpisodeSpec scene id")
    shutil.copyfile(spec_path, output_dir / "resolved_episode_spec.json")
    write_json(output_dir / "episode_intent.json", intent)
    scene_map = iTHORMap.load(str(map_path), agent_radius=0.35)
    candidate = select_free_candidate(
        scene_map.occupancy, scene_map.map_to_world, anchor_xy_m=(0.0, 0.0)
    )
    staging_receipt = {
        **candidate.__dict__,
        "agent_clearance_radius_m": 0.35,
        "map_path": map_path.name,
    }
    kernel = build_kernel_model(
        scene_path,
        mimo_assets,
        output_dir / "kernel_component.xml",
        root_xy=(candidate.world_x_m, candidate.world_y_m),
        target_definition=_target_definition(spec),
    )
    trace, physics_qa = run_physics_trace(kernel, spec)
    staging_receipt["clearance_checks"] = {
        "root_free_with_0_35_m_radius": bool(
            scene_map.check_collision(
                np.asarray([candidate.world_x_m, candidate.world_y_m, 0.0])
            )
        ),
        "camera_xy_free_with_0_35_m_radius": bool(
            scene_map.check_collision(
                np.asarray(
                    [
                        trace["camera_pose"][0, 0],
                        trace["camera_pose"][0, 1],
                        0.0,
                    ]
                )
            )
        ),
        "target_stage_contract": (
            "target is deliberately staged on the deterministic kernel support; "
            "agent-radius occupancy is not applicable to a supported object"
        ),
        "target_initial_center_finite": bool(
            np.isfinite(trace["target_pose"][0, :3]).all()
        ),
    }
    write_json(output_dir / "staging_receipt.json", staging_receipt)
    trace_path = output_dir / "episode_trace.npz"
    np.savez_compressed(trace_path, **trace)
    write_json(output_dir / "body_names.json", list(kernel.mimo_body_names))
    write_json(output_dir / "physics_qa.json", physics_qa)
    frame_streams = {
        key: trace[key]
        for key in (
            "qpos",
            "qvel",
            "action",
            "touch_wrench",
            "touch_contact_count",
            "grasp_active",
            "vestibular_accelerometer",
            "vestibular_gyroscope",
            "target_pose",
            "hand_pose",
            "camera_pose",
            "body_pose",
            "phase",
        )
    }
    clock_qa = validate_shared_clock(trace["time_s"], frame_streams)
    write_json(output_dir / "shared_clock_qa.json", clock_qa)
    render_qa = None
    if render:
        render_qa = render_trace(kernel, trace, output_dir)
        render_qa["camera_sensitivity"] = run_camera_sensitivity(
            output_dir, output_dir / "camera_sensitivity_qa.json"
        )
        write_json(output_dir / "render_qa.json", render_qa)
    waveform, _ = synthesize_speech(spec, output_dir)
    final_video = output_dir / "episode_with_speech.mp4"
    if render:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(output_dir / "actual_furnished_native_diagnostic.mp4"),
                "-i",
                str(waveform),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-t",
                str(value(spec, "duration_s")),
                str(final_video),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    relative_files = [
        "episode_intent.json",
        "resolved_episode_spec.json",
        "staging_receipt.json",
        "kernel_component.xml",
        "episode_trace.npz",
        "body_names.json",
        "physics_qa.json",
        "shared_clock_qa.json",
        "speech_events.json",
        "speech_waveform.wav",
    ]
    if render:
        relative_files.extend(
            [
                "actual_furnished_native_diagnostic.mp4",
                "episode_with_speech.mp4",
                "render_streams.h5",
                "render_qa.json",
                "camera_sensitivity_qa.json",
                "inspection_sheet.png",
            ]
        )
        relative_files.extend(
            path.name
            for path in sorted(output_dir.glob("replay_segmentation_*.png"))
        )
    manifest = build_manifest(
        output_dir,
        relative_files,
        spec_sha256=spec["spec_sha256"],
        provenance={
            "MIMo": {
                "version": "2.0.0",
                "commit": "040b0ae4914cbfb26afdf830aa81775b90922f3f",
            },
            "MolmoSpaces": {
                "version": "0.2.0",
                "commit": "c2f1b583f087e1d3994e1377574843b759d9d0f8",
            },
            "private_childlens_material": False,
            "empirical_source": "ChildLens only",
        },
        regeneration_command=[
            "python3",
            "-m",
            "babyworld_lite.childlens_engine_bakeoff.run_kernel_episode",
            str(spec_path),
            str(intent_path),
            str(scene_path),
            str(map_path),
            str(mimo_assets),
            str(output_dir),
        ],
    )
    write_json(output_dir / "episode_bundle_manifest.json", manifest)
    result = {
        "spec_sha256": spec["spec_sha256"],
        "physics_qa": physics_qa,
        "shared_clock_qa": clock_qa,
        "render_qa": render_qa,
        "manifest": "episode_bundle_manifest.json",
    }
    write_json(output_dir / "qa_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("intent", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("scene_map", type=Path)
    parser.add_argument("mimo_assets", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    result = run(
        args.spec,
        args.intent,
        args.scene,
        args.scene_map,
        args.mimo_assets,
        args.output_dir,
        render=not args.no_render,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
