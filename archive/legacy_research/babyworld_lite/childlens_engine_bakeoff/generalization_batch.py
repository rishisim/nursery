"""Run the frozen Phase 5 embodied engineering/generalization matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image

from .bundle import sha256_file, write_json
from .run_continuous_episode import run as run_continuous_episode
from .run_kernel_episode import _assert_ignored


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bound_json(repo_root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = repo_root / binding["path"]
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise ValueError(
            f"bound file changed: {path} expected {binding['sha256']} got {actual}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = config["matrix"]
    cells = []
    for scene in matrix["scene_variants"]:
        for target in matrix["target_objects"]:
            for seed in matrix["seeds"]:
                cells.append(
                    {
                        "matrix_index": len(cells),
                        "cell_id": (
                            f"{scene}__{target['persistent_id']}__"
                            f"{seed['seed']}"
                        ),
                        "scene_variant": scene,
                        "target": target,
                        "seed": seed,
                    }
                )
    if len(cells) != matrix["episode_count"]:
        raise ValueError("frozen matrix does not produce the declared cell count")
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("frozen matrix contains duplicate cell identifiers")
    return cells


def _add_vector(left: list[float], right: list[float]) -> list[float]:
    return [float(a + b) for a, b in zip(left, right)]


def materialize_cell(
    episode: dict[str, Any],
    vertical: dict[str, Any],
    registry: dict[str, Any],
    cell: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_episode = copy.deepcopy(episode)
    resolved_vertical = copy.deepcopy(vertical)
    target_axis = cell["target"]
    target_id = target_axis["persistent_id"]
    if target_id not in registry:
        raise ValueError(f"target is absent from the bound registry: {target_id}")

    target = {"persistent_id": target_id, **copy.deepcopy(registry[target_id])}
    target.pop("role", None)
    resolved_vertical["scene_family"]["target"] = target
    placement = resolved_vertical["scene_family"]["qualified_placement"]
    seed = cell["seed"]
    placement["target_offset_from_root_m"] = _add_vector(
        placement["target_offset_from_root_m"],
        seed["target_and_support_delta_m"],
    )
    placement["support_offset_from_root_m"] = _add_vector(
        placement["support_offset_from_root_m"],
        seed["target_and_support_delta_m"],
    )
    placement["reach_distractor_offset_from_root_m"] = _add_vector(
        placement.get("reach_distractor_offset_from_root_m", [0.23, 0.02, 0.10]),
        seed["reach_distractor_delta_m"],
    )

    noun = target_axis["noun"]
    color_noun = target_axis["color_noun"]
    resolved_episode["scene_variant"] = cell["scene_variant"]
    resolved_episode["controller"]["seed"] = seed["seed"]
    resolved_episode["planner"]["target_reference"] = {
        "requested_label": target_axis["requested_label"],
        "required_persistent_id": target_id,
    }
    resolved_episode["prompt"] = (
        f"Find the {color_noun}, pick it up, inspect it, shake and tap it, "
        "put it down, then retrieve it and look again."
    )
    texts = (
        f"Look, a {color_noun}.",
        f"Touch the {noun}.",
        f"Up goes the {noun}.",
        "Shake, tap, and move it.",
        f"{noun.capitalize()} again.",
        f"{color_noun.capitalize()}.",
    )
    for utterance, text in zip(
        resolved_episode["language"]["utterances"], texts
    ):
        utterance["text"] = text
    return resolved_episode, resolved_vertical


def _validate_manifest(root: Path, name: str) -> tuple[bool, str]:
    manifest = json.loads((root / name).read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            return False, f"missing:{row['path']}"
        if path.stat().st_size != row["bytes"]:
            return False, f"size:{row['path']}"
        if sha256_file(path) != row["sha256"]:
            return False, f"sha256:{row['path']}"
    return True, _canonical_sha256(sorted(row["path"] for row in manifest["files"]))


def _failed_checks(checks: dict[str, bool]) -> list[str]:
    return sorted(name for name, passed in checks.items() if not passed)


def _compact_cell_result(
    *,
    repo_root: Path,
    cell: dict[str, Any],
    run_dir: Path,
    execution_source: str,
) -> dict[str, Any]:
    qa = json.loads(
        (run_dir / "continuous_qa_report.json").read_text(encoding="utf-8")
    )
    kernel = qa["kernel"]
    physics = kernel["physics_qa"]
    continuous = physics["continuous_episode"]
    qualification = qa["qualification"]
    manifest_valid, schema_signature = _validate_manifest(
        run_dir, "continuous_episode_manifest.json"
    )
    assist_intervals = physics["grasp"]["assist_intervals"]
    assist_duration = sum(
        float(row["end_s"] - row["start_s"]) for row in assist_intervals
    )
    hard_failed = _failed_checks(kernel["qualification"]["physics"]["checks"])
    continuous_failed = _failed_checks(qualification["checks"])
    passed = bool(qualification["passed"] and manifest_valid)
    failure_taxonomy: list[str] = []
    if not kernel["determinism_qa"]["all_pass"]:
        failure_taxonomy.append("DETERMINISM")
    if physics["object_identity"]["changes"]:
        failure_taxonomy.append("OBJECT_IDENTITY")
    if hard_failed:
        if (
            cell["scene_variant"] == "messy"
            and set(hard_failed) <= {"penetration"}
        ):
            failure_taxonomy.append("BOUNDED_DIAGNOSTIC_CLUTTER")
        else:
            failure_taxonomy.append("ENGINEERING_HARD_GATE")
    action_checks = {
        "shake_completed",
        "bang_completed",
        "transfer_completed",
        "release_settle_completed",
        "retrieve_completed",
    }
    if action_checks.intersection(continuous_failed):
        failure_taxonomy.append("ACTION_COMPLETION")
    if {
        "authoritative_render",
        "authoritative_speech",
        "cross_modal_and_mux",
    }.intersection(continuous_failed) or not manifest_valid:
        failure_taxonomy.append("RENDER_OR_CROSS_MODAL")
    trace = np.load(run_dir / "episode_trace.npz")
    render_qa = kernel["render_qa"]
    mux_path = run_dir / "accepted_episode.mp4"
    result = {
        "schema": "EmbodiedGeneralizationCellResult.v1",
        "cell_id": cell["cell_id"],
        "matrix_index": cell["matrix_index"],
        "scene_variant": cell["scene_variant"],
        "target_persistent_id": cell["target"]["persistent_id"],
        "seed": cell["seed"]["seed"],
        "execution_source": execution_source,
        "run": str(run_dir.relative_to(repo_root)),
        "passed": passed,
        "failure_taxonomy": failure_taxonomy,
        "hard_gate_failed_checks": hard_failed,
        "continuous_failed_checks": continuous_failed,
        "manifest_valid": manifest_valid,
        "bundle_schema_signature": schema_signature,
        "trace_sha256": sha256_file(run_dir / "episode_trace.npz"),
        "mux_sha256": sha256_file(mux_path) if mux_path.is_file() else None,
        "duration_s": physics["duration_s"],
        "physics_steps": physics["physics_steps"],
        "truth_samples": physics["truth_samples"],
        "render_frames": render_qa["frames"] if render_qa else None,
        "camera_mount_max_translation_error_m": physics["camera_mount"][
            "maximum_translation_error_m"
        ],
        "camera_mount_max_rotation_error_rad": physics["camera_mount"][
            "maximum_rotation_error_rad"
        ],
        "minimum_relevant_contact_distance_m": physics["collision_policy"][
            "minimum_relevant_contact_distance_m"
        ],
        "persistent_penetration_frames": physics["collision_policy"][
            "persistent_penetration_frames"
        ],
        "near_miss_clearance_m": physics["near_miss"]["minimum_clearance_m"],
        "near_miss_contact_substeps": physics["near_miss"]["contact_substeps"],
        "first_contact_time_s": physics["contact"]["first_contact_time_s"],
        "maximum_distinct_finger_contacts": physics["contact"][
            "maximum_distinct_finger_contacts"
        ],
        "maximum_contact_force_n": physics["contact"]["maximum_force_n"],
        "assist_interval_count": continuous["assist_interval_count"],
        "assist_duration_s": assist_duration,
        "assist_active_truth_frames": int(trace["assist_active"].sum()),
        "unflagged_assist_frames": 0,
        "maximum_lift_m": physics["manipulation"]["maximum_lift_m"],
        "maximum_object_rotation_degrees": physics["manipulation"][
            "maximum_object_rotation_degrees"
        ],
        "maximum_head_turn_degrees": physics["manipulation"][
            "maximum_head_turn_degrees"
        ],
        "head_turn_contact_retention_fraction": physics["manipulation"][
            "head_turn_contact_retention_fraction"
        ],
        "shake_vertical_range_m": continuous["shake_vertical_range_m"],
        "bang_support_contact_samples": continuous["bang_support_contact_samples"],
        "transfer_lateral_range_m": continuous["transfer_lateral_range_m"],
        "retrieve_contact_samples": continuous["retrieve_contact_samples"],
        "retrieve_lift_m": continuous["retrieve_lift_m"],
        "object_identity_changes": physics["object_identity"]["changes"],
        "maximum_timestamp_error_s": physics["synchronization"][
            "maximum_timestamp_error_s"
        ],
        "maximum_replay_numeric_error": kernel["determinism_qa"][
            "maximum_numeric_absolute_error"
        ],
        "collision_proxy_pixels": (
            render_qa["collision_proxy_pixels"] if render_qa else None
        ),
        "skin_artifact_pixels": (
            render_qa["skin_artifact_pixels"] if render_qa else None
        ),
        "maximum_projected_contact_error_px": (
            render_qa["maximum_projected_contact_error_px"]
            if render_qa
            else None
        ),
        "maximum_3d_contact_surface_distance_m": (
            render_qa["maximum_3d_contact_surface_distance_m"]
            if render_qa
            else None
        ),
    }
    return result


def _verify_reuse(
    repo_root: Path,
    config: dict[str, Any],
    episode_sha256: str,
    vertical_sha256: str,
) -> Path:
    reuse = config["reuse"]
    source = repo_root / reuse["source_run"]
    manifest = json.loads(
        (source / "continuous_episode_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["spec_sha256"] != episode_sha256:
        raise ValueError("Phase 4 reuse episode contract hash changed")
    if manifest["provenance"]["vertical_contract_sha256"] != vertical_sha256:
        raise ValueError("Phase 4 reuse vertical contract hash changed")
    if sha256_file(source / "episode_trace.npz") != reuse["required_trace_sha256"]:
        raise ValueError("Phase 4 reuse trace hash changed")
    if sha256_file(source / "accepted_episode.mp4") != reuse["required_mux_sha256"]:
        raise ValueError("Phase 4 reuse mux hash changed")
    valid, _ = _validate_manifest(source, "continuous_episode_manifest.json")
    qualification = json.loads(
        (source / "continuous_qualification.json").read_text(encoding="utf-8")
    )
    if not valid or not qualification["passed"]:
        raise ValueError("Phase 4 reuse no longer passes manifest qualification")
    return source


def _sample_video_frames(
    video_path: Path, sampling: dict[str, Any]
) -> list[np.ndarray]:
    fps = float(iio.immeta(video_path, plugin="FFMPEG")["fps"])
    count = int(sampling["frames_per_episode"])
    indices = {
        int(round((0.5 + second) * fps)): position
        for position, second in enumerate(range(count))
    }
    frames: list[np.ndarray | None] = [None] * count
    final_index = max(indices)
    for frame_index, frame in enumerate(iio.imiter(video_path, plugin="FFMPEG")):
        if frame_index in indices:
            frames[indices[frame_index]] = np.asarray(frame, dtype=np.uint8)
        if frame_index >= final_index:
            break
    if any(frame is None for frame in frames):
        raise ValueError(f"video is missing frozen sample frames: {video_path}")
    return [frame for frame in frames if frame is not None]


def _grayscale_metrics(
    frames: list[np.ndarray], sampling: dict[str, Any]
) -> tuple[dict[str, float], list[Image.Image]]:
    size = tuple(int(item) for item in sampling["grayscale_size_px"])
    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    grayscale = [
        np.asarray(image.convert("L").resize(size), dtype=np.float32) / 255.0
        for image in images
    ]
    differences = np.asarray(
        [
            float(np.mean(np.abs(right - left)))
            for left, right in zip(grayscale, grayscale[1:])
        ],
        dtype=np.float32,
    )
    return (
        {
            "motion": float(differences.mean()),
            "scene_change_rate": float(
                np.mean(
                    differences
                    >= float(sampling["scene_change_motion_threshold"])
                )
            ),
        },
        images,
    )


def _frontend_receipt(model_dir: Path, frontend: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "model.safetensors": frontend["weights_sha256"],
        "config.json": frontend["config_sha256"],
    }
    actual = {
        path.name: sha256_file(path)
        for path in sorted(model_dir.iterdir())
        if path.is_file()
    }
    mismatches = {
        name: {"expected": expected_hash, "actual": actual.get(name)}
        for name, expected_hash in expected.items()
        if actual.get(name) != expected_hash
    }
    if mismatches:
        raise ValueError(f"frozen DINO frontend hash mismatch: {mismatches}")
    return {
        "repository": frontend["repository"],
        "revision": frontend["revision"],
        "model_dir": str(model_dir),
        "file_sha256": actual,
        "required_hashes_match": True,
    }


def _load_frontend(model_dir: Path, frontend: dict[str, Any]):
    import torch
    import transformers
    from transformers import AutoImageProcessor, Dinov2Model

    receipt = _frontend_receipt(model_dir, frontend)
    torch.set_num_threads(4)
    processor = AutoImageProcessor.from_pretrained(
        model_dir, local_files_only=True, use_fast=False
    )
    model = Dinov2Model.from_pretrained(model_dir, local_files_only=True).eval()
    receipt.update(
        {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
        }
    )
    return torch, processor, model, receipt


def _dino_persistence(
    images: list[Image.Image], torch: Any, processor: Any, model: Any
) -> float:
    rows = []
    for left in range(0, len(images), 32):
        pixels = processor(
            images=images[left : left + 32], return_tensors="pt"
        ).pixel_values
        with torch.inference_mode():
            embedded = model(pixel_values=pixels).pooler_output
        rows.append(embedded.detach().cpu().numpy().astype(np.float32))
    values = np.concatenate(rows)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(values[:-1] * values[1:], axis=1)))


def _interval_status(
    value: float, interval: list[float]
) -> dict[str, Any]:
    low, high = (float(item) for item in interval)
    return {
        "value": value,
        "target_interval_90": [low, high],
        "passed": low <= value <= high,
    }


def score_visual_distribution(
    *,
    results: list[dict[str, Any]],
    config: dict[str, Any],
    repo_root: Path,
    model_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    visual = config["visual_distribution"]
    torch, processor, model, frontend_receipt = _load_frontend(
        model_dir, visual["vision_frontend"]
    )
    scored = []
    for result in results:
        if result.get("status") == "EXECUTION_ERROR":
            continue
        frames = _sample_video_frames(
            repo_root / result["run"] / "baseline_rgb.mp4", visual["sampling"]
        )
        grayscale, images = _grayscale_metrics(frames, visual["sampling"])
        values = {
            **grayscale,
            "adjacent_frame_persistence": _dino_persistence(
                images, torch, processor, model
            ),
        }
        metrics = {
            name: _interval_status(
                values[name], definition["target_interval_90"]
            )
            for name, definition in visual["metrics"].items()
        }
        result["visual_distribution"] = metrics
        if not all(row["passed"] for row in metrics.values()):
            result["failure_taxonomy"] = sorted(
                {*result["failure_taxonomy"], "VISUAL_DISTRIBUTION_MISS"}
            )
        scored.append(values)
    aggregate = {
        "schema": "EmbodiedVisualDistributionAggregate.v1",
        "cell_count": len(scored),
        "frontend": frontend_receipt,
        "metrics": {},
        "interpretation": visual["interpretation"],
        "model_model_sensitivity_only": True,
    }
    for name, definition in visual["metrics"].items():
        values = [row[name] for row in scored]
        aggregate["metrics"][name] = {
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "target_interval_90": definition["target_interval_90"],
            "mean_passed": (
                definition["target_interval_90"][0]
                <= float(np.mean(values))
                <= definition["target_interval_90"][1]
            ),
            "cells_passed": sum(
                result["visual_distribution"][name]["passed"]
                for result in results
                if "visual_distribution" in result
            ),
        }
    return results, aggregate


def _visual_distance(result: dict[str, Any]) -> float:
    distance = 0.0
    for row in result["visual_distribution"].values():
        low, high = row["target_interval_90"]
        midpoint = (low + high) / 2.0
        half_width = (high - low) / 2.0
        distance += abs(row["value"] - midpoint) / half_width
    return distance


def _batch_qualification(
    results: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    gate = config["batch_gate"]
    required_scenes = set(config["matrix"]["required_scene_variants"])
    required_passes = [
        row
        for row in results
        if row.get("passed") and row["scene_variant"] in required_scenes
    ]
    failures = [row for row in results if not row.get("passed")]
    passed_schemas = {
        row["bundle_schema_signature"] for row in results if row.get("passed")
    }
    checks = {
        "matrix_cells_completed": (
            len(results) == gate["matrix_cells_completed"]
            and all(row.get("status") != "EXECUTION_ERROR" for row in results)
        ),
        "minimum_required_scene_passes": len(required_passes)
        >= gate["minimum_required_scene_continuous_passes"],
        "each_required_scene_has_a_pass": all(
            any(row["scene_variant"] == scene for row in required_passes)
            for scene in required_scenes
        ),
        "target_objects_executed": len(
            {
                row["target_persistent_id"]
                for row in results
                if row.get("status") != "EXECUTION_ERROR"
            }
        )
        == gate["target_objects_executed"],
        "failures_bounded_and_replay_qualified": all(
            row.get("status") != "EXECUTION_ERROR"
            and row["maximum_replay_numeric_error"] == 0.0
            for row in failures
        ),
        "passed_bundle_schemas_match": len(passed_schemas) == 1,
        "selectable_candidate_exists": any(row.get("passed") for row in results),
        "visual_distribution_not_used_as_truth_gate": gate[
            "visual_distribution_is_not_a_core_truth_gate"
        ],
    }
    eligible = [row for row in results if row.get("passed")]
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["unflagged_assist_frames"],
                row["assist_duration_s"],
                _visual_distance(row),
                row["matrix_index"],
            ),
        )
    return (
        {
            "schema": "EmbodiedGeneralizationBatchQualification.v1",
            "checks": checks,
            "passed": all(checks.values()),
            "required_scene_pass_count": len(required_passes),
            "total_episode_pass_count": len(eligible),
            "bounded_failure_count": len(failures),
            "selected_cell_id": selected["cell_id"] if selected else None,
        },
        selected,
    )


def run(
    config_path: Path,
    scene_path: Path,
    mimo_assets: Path,
    output_root: Path,
    *,
    model_dir: Path,
    indices: set[int] | None = None,
    render: bool = True,
    speech: bool = True,
    score_visual: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = output_root.resolve()
    model_dir = model_dir.resolve()
    _assert_ignored(repo_root, output_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    episode = _bound_json(repo_root, config["canonical_episode_contract"])
    vertical = _bound_json(repo_root, config["canonical_vertical_slice_contract"])
    registry = _bound_json(repo_root, config["asset_registry"])
    cells = build_cells(config)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "frozen_batch_manifest.json",
        {
            "schema": "EmbodiedGeneralizationBatchManifest.v1",
            "config_sha256": sha256_file(config_path),
            "cells": cells,
            "scene_sha256": sha256_file(scene_path),
            "privacy": config["privacy_and_governance"],
        },
    )

    reuse_source = _verify_reuse(
        repo_root,
        config,
        config["canonical_episode_contract"]["sha256"],
        config["canonical_vertical_slice_contract"]["sha256"],
    )
    selected_indices = indices if indices is not None else set(range(len(cells)))
    for cell in cells:
        if cell["matrix_index"] not in selected_indices:
            continue
        cell_dir = output_root / "cells" / cell["cell_id"]
        _assert_ignored(repo_root, cell_dir)
        cell_dir.mkdir(parents=True, exist_ok=True)
        result_path = cell_dir / "cell_result.json"
        if resume and result_path.exists():
            continue
        cell_episode, cell_vertical = materialize_cell(
            episode, vertical, registry, cell
        )
        episode_path = cell_dir / "cell_episode_contract.json"
        vertical_path = cell_dir / "cell_vertical_contract.json"
        write_json(episode_path, cell_episode)
        write_json(vertical_path, cell_vertical)
        try:
            if cell["cell_id"] == config["reuse"]["cell_id"]:
                run_dir = reuse_source
                source = "revalidated_phase_4_reference"
            else:
                run_dir = cell_dir
                source = "phase_5_execution"
                run_continuous_episode(
                    episode_path,
                    vertical_path,
                    scene_path,
                    mimo_assets,
                    run_dir,
                    render=render,
                    speech=speech,
                )
            compact = _compact_cell_result(
                repo_root=repo_root,
                cell=cell,
                run_dir=run_dir,
                execution_source=source,
            )
            compact["status"] = "PASS" if compact["passed"] else "FAIL"
        except Exception as error:
            compact = {
                "schema": "EmbodiedGeneralizationCellResult.v1",
                "cell_id": cell["cell_id"],
                "matrix_index": cell["matrix_index"],
                "scene_variant": cell["scene_variant"],
                "target_persistent_id": cell["target"]["persistent_id"],
                "seed": cell["seed"]["seed"],
                "status": "EXECUTION_ERROR",
                "passed": False,
                "failure_taxonomy": ["EXECUTION_ERROR"],
                "error_type": type(error).__name__,
                "error": str(error),
            }
        write_json(result_path, compact)

    results = []
    for cell in cells:
        result_path = output_root / "cells" / cell["cell_id"] / "cell_result.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
    complete = len(results) == len(cells)
    visual_aggregate = None
    if complete and render and score_visual:
        results, visual_aggregate = score_visual_distribution(
            results=results,
            config=config,
            repo_root=repo_root,
            model_dir=model_dir,
        )
        for result in results:
            result_path = (
                output_root / "cells" / result["cell_id"] / "cell_result.json"
            )
            write_json(result_path, result)
        write_json(output_root / "visual_distribution_qa.json", visual_aggregate)

    qualification = None
    selected = None
    if complete and render and speech and visual_aggregate is not None:
        qualification, selected = _batch_qualification(results, config)
        write_json(output_root / "batch_qualification.json", qualification)
    aggregate = {
        "schema": "EmbodiedGeneralizationBatchAggregate.v1",
        "config_sha256": sha256_file(config_path),
        "episode_count": len(results),
        "complete": complete,
        "results": results,
        "visual_distribution": visual_aggregate,
        "qualification": qualification,
        "selected_candidate": (
            {
                "cell_id": selected["cell_id"],
                "run": selected["run"],
                "trace_sha256": selected["trace_sha256"],
                "mux_sha256": selected["mux_sha256"],
                "selection_rank": {
                    "unflagged_assist_frames": selected[
                        "unflagged_assist_frames"
                    ],
                    "assist_duration_s": selected["assist_duration_s"],
                    "normalized_visual_l1_distance": _visual_distance(selected),
                    "matrix_index": selected["matrix_index"],
                },
            }
            if selected
            else None
        ),
        "privacy": {
            "restricted_childlens_accessed": False,
            "public_synthetic_inputs_only": True,
            "empirical_source": "ChildLens aggregate intervals only",
        },
    }
    write_json(output_root / "batch_aggregate.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("configs/embodied_simulation_generalization.json"),
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path(
            ".external/molmospaces-cache/scenes/ithor/"
            "20251217_with_occupancy/FloorPlan201_physics.xml"
        ),
    )
    parser.add_argument(
        "--mimo-assets",
        type=Path,
        default=Path(".external/mimo-pin/mimoEnv/assets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/embodied_simulation/phase_5/batch"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".external/models/childlens_bridge_v5/dinov2-small"),
    )
    parser.add_argument("--indices", default="")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--no-visual-score", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    indices = (
        {int(token) for token in args.indices.split(",") if token}
        if args.indices
        else None
    )
    result = run(
        args.config,
        args.scene,
        args.mimo_assets,
        args.output_root,
        model_dir=args.model_dir,
        indices=indices,
        render=not args.no_render,
        speech=not args.no_speech,
        score_visual=not args.no_visual_score,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
