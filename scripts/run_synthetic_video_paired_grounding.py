#!/usr/bin/env python3
"""Paired qualitative and CLIP-grounded lexical preservation test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


class GroundingError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def private_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(canonical(value) + b"\n")
    os.chmod(path, 0o600)


def extract_frames(media: Path, root: Path, timestamps: list[float]) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths = []
    for ordinal, timestamp in enumerate(timestamps):
        target = root / f"{ordinal:02d}.jpg"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{timestamp:.6f}", "-i", str(media), "-frames:v", "1", "-vf", "scale='min(768,iw)':-2", "-q:v", "2", "-y", str(target)],
            check=True,
        )
        if not target.is_file() or not target.stat().st_size:
            raise GroundingError("E_FRAME_DECODE")
        os.chmod(target, 0o600)
        paths.append(target)
    return paths


def aggregate_arm_metrics(rows: list[dict], probes: dict, arms: tuple[str, ...]) -> dict:
    grouped: dict[tuple[str, str], list[list[float]]] = {}
    for row in rows:
        key = (row["arm"], row["probe"])
        scores = [float(value) for value in row["scores"]]
        if len(scores) != 1 + len(probes[row["probe"]]["distractors"]):
            raise GroundingError("E_SCORE_SHAPE")
        grouped.setdefault(key, []).append(scores)
    metrics = {}
    for probe in probes:
        metrics[probe] = {}
        for arm in arms:
            values = grouped.get((arm, probe), [])
            if len(values) != 10:
                raise GroundingError("E_SCORE_COUNT")
            correct = [int(row[0] == max(row)) for row in values]
            margins = [row[0] - max(row[1:]) for row in values]
            metrics[probe][arm] = {
                "frame_count": len(values),
                "top1_accuracy": sum(correct) / len(correct),
                "mean_target_probability": sum(row[0] for row in values) / len(values),
                "mean_target_minus_best_distractor_margin": sum(margins) / len(margins),
            }
    return metrics


def metric_delta(later: dict, earlier: dict) -> dict:
    return {
        key: later[key] - earlier[key]
        for key in ("top1_accuracy", "mean_target_probability", "mean_target_minus_best_distractor_margin")
    }


def aggregate(rows: list[dict], probes: dict, arms: tuple[str, str]) -> dict:
    metrics = aggregate_arm_metrics(rows, probes, arms)
    for probe in probes:
        real, synthetic = metrics[probe][arms[0]], metrics[probe][arms[1]]
        metrics[probe]["synthetic_minus_real"] = metric_delta(synthetic, real)
    return metrics


def aggregate_three_arm(rows: list[dict], probes: dict, arms: tuple[str, str, str]) -> dict:
    metrics = aggregate_arm_metrics(rows, probes, arms)
    real_arm, hailuo_arm, ltx_arm = arms
    for probe in probes:
        values = metrics[probe]
        values["deltas"] = {
            "hailuo_minus_real": metric_delta(values[hailuo_arm], values[real_arm]),
            "ltx_minus_real": metric_delta(values[ltx_arm], values[real_arm]),
            "ltx_minus_hailuo": metric_delta(values[ltx_arm], values[hailuo_arm]),
        }
    return metrics


def qualitative(existing: dict) -> dict:
    checks = existing["checks"]
    return {
        "setting": checks["setting_broadly_retained"],
        "activity": checks["activity_broadly_retained"],
        "primary_objects": checks["primary_public_category_objects_retained"],
        "attribute": "covered_by_grounded_adjective_probe",
        "hand_action": checks["intended_hand_action_beat_retained_when_present"],
        "viewpoint": checks["first_person_camera_retained"],
        "continuity": checks["no_cuts_or_severe_object_identity_drift"],
        "artifacts": checks["no_caption_logo_watermark_physics_or_anatomy_defect"],
        "speech_and_referent_timing": checks["speech_and_referent_timing_within_0_75_seconds"],
        "null_meaning": existing["null_check_meaning"],
    }


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    preregistration = json.loads(args.config.read_text())
    three_arm = args.ltx_root is not None
    config_key = "public_three_arm_grounded_lexical_test" if three_arm else "public_single_pair_grounded_lexical_test"
    config = preregistration[config_key]
    frozen_status = "FROZEN_BEFORE_THREE_ARM_FRAME_EXTRACTION_OR_SCORE" if three_arm else "FROZEN_BEFORE_CLIP_EVALUATOR_DOWNLOAD_OR_SCORE"
    if config["status"] != frozen_status:
        raise GroundingError("E_PROTOCOL_NOT_FROZEN")
    ltx_result = None
    if three_arm:
        episode_plan = json.loads((args.prototype_root / "episode_plan.json").read_text())
        ltx_result = json.loads(args.ltx_result.read_text())
        commitment = config["shared_source_prompt_commitment_sha256"]
        if episode_plan["commitment_sha256"] != commitment or ltx_result["prompt_commitment_sha256"] != commitment:
            raise GroundingError("E_SHARED_PROMPT_COMMITMENT")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    timestamps = [float(value) for value in config["frames"]["timestamps_seconds"]]
    real_frames = extract_frames(args.prototype_root / "source_10s.mp4", root / "frames/real", timestamps)
    hailuo_frames = extract_frames(args.prototype_root / "synthetic_10s.mp4", root / "frames/hailuo", timestamps)
    probes = preregistration["public_single_pair_grounded_lexical_test"]["probes"]
    arms = [
        {"name": "public_real_source", "frames": [str(path) for path in real_frames]},
        {"name": "public_hailuo_derived" if three_arm else "public_synthetic_derived", "frames": [str(path) for path in hailuo_frames]},
    ]
    if three_arm:
        ltx_frames = extract_frames(args.ltx_root / "synthetic_10s.mp4", root / "frames/ltx", timestamps)
        arms.append({"name": "public_ltx_derived", "frames": [str(path) for path in ltx_frames]})
    manifest = {
        "schema_version": 1,
        "evaluator": config["evaluator"],
        "probes": probes,
        "arms": arms,
    }
    manifest_path, scores_path = root / "scoring_manifest.json", root / "frame_scores.json"
    private_write(manifest_path, manifest)
    environment = dict(os.environ)
    environment["TRANSFORMERS_JS_ROOT"] = str(args.transformers_js_root.resolve())
    environment["HF_HUB_CACHE"] = str((root / "model_cache").resolve())
    subprocess.run(["node", str(args.scorer), str(manifest_path), str(scores_path)], check=True, env=environment)
    scores = json.loads(scores_path.read_text())
    arm_names = tuple(arm["name"] for arm in arms)
    metrics = aggregate_three_arm(scores["rows"], probes, arm_names) if three_arm else aggregate(scores["rows"], probes, arm_names)
    existing = json.loads((args.prototype_root / "exploratory_comparison.json").read_text())
    qualitative_result = qualitative(existing)
    if three_arm:
        ltx_qualitative = ltx_result["one_time_qualitative_check"]
        qualitative_result = {
            "hailuo_vs_real": qualitative_result,
            "ltx_vs_real": {
                "setting": ltx_qualitative["setting_broadly_retained"],
                "activity": ltx_qualitative["garment_care_activity_broadly_retained"],
                "primary_objects": ltx_qualitative["primary_public_category_garment_retained"],
                "attribute": "covered_by_grounded_adjective_probe",
                "hand_action": ltx_qualitative["two_hand_contact_and_manipulation_retained"],
                "viewpoint": ltx_qualitative["first_person_camera_retained"],
                "continuity": ltx_qualitative["single_shot_continuity_retained"],
                "artifacts": ltx_qualitative["no_obvious_caption_logo_watermark_impossible_physics_floating_object_or_severe_anatomy_defect_in_sampled_frames"],
                "speech_and_referent_timing": None,
                "null_meaning": "not_applicable_because_source_speech_was_unsupported",
            },
        }
    result = {
        "schema_version": 1,
        "status": "EXPLORATORY_COMPLETE",
        "question": config["question"],
        "arms": arm_names,
        "qualitative": qualitative_result,
        "grounded_lexical_metrics": metrics,
        "human_response_KL": None,
        "human_response_KL_reason": "no participant response distribution for this custom pair",
        "lexical_acquisition": None,
        "lexical_acquisition_reason": preregistration["public_single_pair_grounded_lexical_test"]["acquisition_status"],
        "omnibus_score": None,
        "claim_limits": config["claim_limits"],
        "wall_seconds": time.monotonic() - started,
    }
    result["commitment_sha256"] = digest({key: value for key, value in result.items() if key != "wall_seconds"})
    private_write(root / ("three_arm_grounded_lexical_result.json" if three_arm else "paired_grounded_lexical_result.json"), result)
    real_key, synthetic_key = arm_names[0], arm_names[1]
    compact = {
        "status": result["status"],
        "arm_count": len(arm_names),
        "pair_count": 3 if three_arm else 1,
        "frame_count_per_arm": 10,
        "probe_count": len(metrics),
        "noun_real_top1": metrics["noun"][real_key]["top1_accuracy"],
        "noun_hailuo_top1" if three_arm else "noun_synthetic_top1": metrics["noun"][synthetic_key]["top1_accuracy"],
        "adjective_real_top1": metrics["adjective"][real_key]["top1_accuracy"],
        "adjective_hailuo_top1" if three_arm else "adjective_synthetic_top1": metrics["adjective"][synthetic_key]["top1_accuracy"],
        "action_real_top1": metrics["action_guardrail"][real_key]["top1_accuracy"],
        "action_hailuo_top1" if three_arm else "action_synthetic_top1": metrics["action_guardrail"][synthetic_key]["top1_accuracy"],
        "viewpoint_real_top1": metrics["viewpoint_guardrail"][real_key]["top1_accuracy"],
        "viewpoint_hailuo_top1" if three_arm else "viewpoint_synthetic_top1": metrics["viewpoint_guardrail"][synthetic_key]["top1_accuracy"],
        "human_response_KL_created": False,
        "lexical_acquisition_measured": False,
        "omnibus_score_created": False,
        "wall_seconds": result["wall_seconds"],
        "commitment_sha256": result["commitment_sha256"],
    }
    if three_arm:
        compact.update({
            "noun_ltx_top1": metrics["noun"][arm_names[2]]["top1_accuracy"],
            "adjective_ltx_top1": metrics["adjective"][arm_names[2]]["top1_accuracy"],
            "action_ltx_top1": metrics["action_guardrail"][arm_names[2]]["top1_accuracy"],
            "viewpoint_ltx_top1": metrics["viewpoint_guardrail"][arm_names[2]]["top1_accuracy"],
            "hailuo_qualitative_mismatch_count": sum(value is False for key, value in result["qualitative"]["hailuo_vs_real"].items() if key != "null_meaning"),
            "ltx_qualitative_mismatch_count": sum(value is False for key, value in result["qualitative"]["ltx_vs_real"].items() if key != "null_meaning"),
        })
    else:
        compact["qualitative_mismatch_count"] = sum(value is False for key, value in result["qualitative"].items() if key != "null_meaning")
    private_write(root / "compact_result.json", compact)
    print(json.dumps(compact, sort_keys=True))
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_video_preregistration.json"))
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--ltx-root", type=Path)
    parser.add_argument("--ltx-result", type=Path, default=Path("results/synthetic_video_public_single_clip_ltx_api.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transformers-js-root", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, default=Path("scripts/synthetic_video_clip_scores.mjs"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
