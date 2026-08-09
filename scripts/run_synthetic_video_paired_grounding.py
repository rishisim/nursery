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


def aggregate(rows: list[dict], probes: dict, arms: tuple[str, str]) -> dict:
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
        real, synthetic = metrics[probe][arms[0]], metrics[probe][arms[1]]
        metrics[probe]["synthetic_minus_real"] = {
            key: synthetic[key] - real[key]
            for key in ("top1_accuracy", "mean_target_probability", "mean_target_minus_best_distractor_margin")
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
    config = json.loads(args.config.read_text())["public_single_pair_grounded_lexical_test"]
    if config["status"] != "FROZEN_BEFORE_CLIP_EVALUATOR_DOWNLOAD_OR_SCORE":
        raise GroundingError("E_PROTOCOL_NOT_FROZEN")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    timestamps = [float(value) for value in config["frames"]["timestamps_seconds"]]
    real_frames = extract_frames(args.prototype_root / "source_10s.mp4", root / "frames/real", timestamps)
    synthetic_frames = extract_frames(args.prototype_root / "synthetic_10s.mp4", root / "frames/synthetic", timestamps)
    manifest = {
        "schema_version": 1,
        "evaluator": config["evaluator"],
        "probes": config["probes"],
        "arms": [
            {"name": "public_real_source", "frames": [str(path) for path in real_frames]},
            {"name": "public_synthetic_derived", "frames": [str(path) for path in synthetic_frames]},
        ],
    }
    manifest_path, scores_path = root / "scoring_manifest.json", root / "frame_scores.json"
    private_write(manifest_path, manifest)
    environment = dict(os.environ)
    environment["TRANSFORMERS_JS_ROOT"] = str(args.transformers_js_root.resolve())
    environment["HF_HUB_CACHE"] = str((root / "model_cache").resolve())
    subprocess.run(["node", str(args.scorer), str(manifest_path), str(scores_path)], check=True, env=environment)
    scores = json.loads(scores_path.read_text())
    metrics = aggregate(scores["rows"], config["probes"], ("public_real_source", "public_synthetic_derived"))
    existing = json.loads((args.prototype_root / "exploratory_comparison.json").read_text())
    result = {
        "schema_version": 1,
        "status": "EXPLORATORY_COMPLETE",
        "question": config["question"],
        "qualitative": qualitative(existing),
        "grounded_lexical_metrics": metrics,
        "human_response_KL": None,
        "human_response_KL_reason": "no participant response distribution for this custom pair",
        "lexical_acquisition": None,
        "lexical_acquisition_reason": config["acquisition_status"],
        "omnibus_score": None,
        "claim_limits": config["claim_limits"],
        "wall_seconds": time.monotonic() - started,
    }
    result["commitment_sha256"] = digest({key: value for key, value in result.items() if key != "wall_seconds"})
    private_write(root / "paired_grounded_lexical_result.json", result)
    compact = {
        "status": result["status"],
        "pair_count": 1,
        "frame_count_per_arm": 10,
        "probe_count": len(metrics),
        "noun_real_top1": metrics["noun"]["public_real_source"]["top1_accuracy"],
        "noun_synthetic_top1": metrics["noun"]["public_synthetic_derived"]["top1_accuracy"],
        "adjective_real_top1": metrics["adjective"]["public_real_source"]["top1_accuracy"],
        "adjective_synthetic_top1": metrics["adjective"]["public_synthetic_derived"]["top1_accuracy"],
        "action_real_top1": metrics["action_guardrail"]["public_real_source"]["top1_accuracy"],
        "action_synthetic_top1": metrics["action_guardrail"]["public_synthetic_derived"]["top1_accuracy"],
        "viewpoint_real_top1": metrics["viewpoint_guardrail"]["public_real_source"]["top1_accuracy"],
        "viewpoint_synthetic_top1": metrics["viewpoint_guardrail"]["public_synthetic_derived"]["top1_accuracy"],
        "qualitative_mismatch_count": sum(value is False for key, value in result["qualitative"].items() if key != "null_meaning"),
        "human_response_KL_created": False,
        "lexical_acquisition_measured": False,
        "omnibus_score_created": False,
        "wall_seconds": result["wall_seconds"],
        "commitment_sha256": result["commitment_sha256"],
    }
    private_write(root / "compact_result.json", compact)
    print(json.dumps(compact, sort_keys=True))
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic_video_preregistration.json"))
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transformers-js-root", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, default=Path("scripts/synthetic_video_clip_scores.mjs"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
