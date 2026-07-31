#!/usr/bin/env python3
"""Public/dummy 2-node x 2-GPU FLUX topology qualification for Phase 4."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path


def main() -> None:
    import torch
    import torch.distributed as dist
    from diffusers import Flux2KleinPipeline

    required_env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_DISABLED": "true",
    }
    if any(os.environ.get(key) != value for key, value in required_env.items()):
        raise RuntimeError("offline/telemetry environment contract is incomplete")
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    if world_size != 2:
        raise RuntimeError(f"expected exactly two ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    started = time.monotonic()
    model_root = Path(os.environ["PHASE4_FLUX_MODEL_ROOT"])
    output_root = Path(os.environ["PHASE4_PREFLIGHT_OUTPUT_ROOT"])
    output_root.mkdir(parents=True, exist_ok=True)
    pipe = Flux2KleinPipeline.from_pretrained(
        model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(f"cuda:{local_rank}")
    generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(4100 + rank)
    image = pipe(
        prompt="A red wooden cube next to a blue wooden sphere on a plain table, no people.",
        height=256,
        width=256,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=generator,
    ).images[0]
    image_path = output_root / f"rank-{rank}.png"
    image.save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    finite = torch.tensor([1 if image.getbbox() is not None else 0], device=local_rank)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    evidence = {
        "rank": rank,
        "node_hash": hashlib.sha256(socket.gethostname().encode()).hexdigest(),
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "cuda_capability": list(torch.cuda.get_device_capability(local_rank)),
        "image_sha256": digest,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(local_rank),
        "wall_seconds": time.monotonic() - started,
        "finite_output": bool(finite.item()),
    }
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(evidence, gathered, dst=0)
    dist.barrier()
    if rank == 0:
        node_count = len({item["node_hash"] for item in gathered})
        result = {
            "schema_version": 1,
            "gate": "phase4_flux_public_dummy_final_topology",
            "status": "PASS" if node_count == 1 and all(item["finite_output"] for item in gathered) else "FAIL",
            "public_dummy_only": True,
            "world_size": world_size,
            "node_count": node_count,
            "gpus_per_node": 2,
            "model_revision": os.environ["PHASE4_FLUX_MODEL_REVISION"],
            "telemetry_disabled": True,
            "local_files_only": True,
            "ranks": gathered,
        }
        payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (output_root / "result.json").write_bytes(payload)
        print(json.dumps({
            "status": result["status"], "world_size": world_size,
            "node_count": node_count, "gpus_per_node": 2,
            "result_sha256": hashlib.sha256(payload).hexdigest(),
        }, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
