#!/usr/bin/env python3
"""Run one frozen public LTX-2.3 episode plan without exposing its prompt in argv."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


PROMPT_COMMITMENT = "63e20f20c29cf6d88116172ec245a10b788753581fc4643e773104f7eb6a71cb"
SOURCE_REVISION = "9377758131b1ffde4b7f766804590a6617bf2ab9"
WEIGHTS_REVISION = "4229404625088d21c4f112eb640fb04a0900ee25"
GEMMA_REVISION = "68f7ee4fbd59087436ada77ed2d62f373fdd4482"


class JunoLTXError(RuntimeError):
    """Fail closed on a frozen-run contract violation."""


def private_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def run(args: argparse.Namespace) -> dict:
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    if plan.get("commitment_sha256") != PROMPT_COMMITMENT:
        raise JunoLTXError("E_PROMPT_COMMITMENT")
    if not isinstance(plan.get("prompt"), str) or not plan["prompt"].strip():
        raise JunoLTXError("E_PROMPT_MISSING")
    if args.source_revision != SOURCE_REVISION:
        raise JunoLTXError("E_SOURCE_REVISION")
    if args.weights_revision != WEIGHTS_REVISION or args.gemma_revision != GEMMA_REVISION:
        raise JunoLTXError("E_WEIGHTS_REVISION")
    for path in (args.checkpoint, args.upsampler, args.gemma_root):
        if not path.exists():
            raise JunoLTXError("E_MODEL_ASSET_MISSING")
    if args.output.exists():
        raise JunoLTXError("E_OUTPUT_ALREADY_EXISTS")

    # Import only after the cheap frozen-contract checks have passed.
    import torch
    from ltx_pipelines.distilled import main as official_distilled_main

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise JunoLTXError("E_ONE_CUDA_GPU_REQUIRED")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    sys.argv = [
        "ltx_pipelines.distilled",
        "--distilled-checkpoint-path",
        str(args.checkpoint),
        "--spatial-upsampler-path",
        str(args.upsampler),
        "--gemma-root",
        str(args.gemma_root),
        "--seed",
        str(args.seed),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.num_frames),
        "--frame-rate",
        str(args.frame_rate),
        "--quantization",
        "fp8-cast",
        "--output-path",
        str(args.output),
        "--prompt",
        plan["prompt"],
    ]
    official_distilled_main()
    elapsed = time.monotonic() - started
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise JunoLTXError("E_OUTPUT_MISSING")
    record = {
        "schema_version": 1,
        "status": "JUNO_LTX_GENERATION_COMPLETE",
        "public_only": True,
        "prompt_commitment_sha256": PROMPT_COMMITMENT,
        "manual_prompt_edit": False,
        "source_revision": SOURCE_REVISION,
        "weights_revision": WEIGHTS_REVISION,
        "gemma_revision": GEMMA_REVISION,
        "pipeline": "official_ltx_pipelines_distilled_two_stage",
        "prompt_enhancement": False,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "quantization": "fp8-cast",
        "offload": "none",
        "gpu_count": 1,
        "gpu_model": torch.cuda.get_device_name(0),
        "generation_wall_seconds": elapsed,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "native_audio_disposition": "discard_before_matched_local_audio_mux",
        "accepted_clip_count": 0,
    }
    private_write(args.record, record)
    print(json.dumps({"status": record["status"], "generation_wall_seconds": elapsed}, sort_keys=True))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upsampler", type=Path, required=True)
    parser.add_argument("--gemma-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--weights-revision", default=WEIGHTS_REVISION)
    parser.add_argument("--gemma-revision", default=GEMMA_REVISION)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=1152)
    parser.add_argument("--num-frames", type=int, default=241)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
