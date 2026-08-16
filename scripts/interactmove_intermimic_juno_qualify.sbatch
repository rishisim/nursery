#!/usr/bin/env bash
#SBATCH --job-name=imim-qualify
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/juno/dal503972/interactmove_intermimic/logs/qualification-%j.out
#SBATCH --error=/scratch/juno/dal503972/interactmove_intermimic/logs/qualification-%j.err

set -euo pipefail
umask 077

readonly SCRATCH_ROOT=/scratch/juno/dal503972/interactmove_intermimic
readonly DURABLE_ROOT=/work/dal503972/interactmove_intermimic
readonly CONTAINER=/work/dal503972/phase4_public/models/pytorch-2.8.0-cu126.sif
readonly RUN_ROOT="${SCRATCH_ROOT}/runs/qualification/${SLURM_JOB_ID}"
readonly RECEIPT="${RUN_ROOT}/qualification_receipt.json"
readonly DURABLE_RECEIPT="${DURABLE_ROOT}/compact_records/juno_qualification_${SLURM_JOB_ID}.json"

install -d -m 700 "${RUN_ROOT}" "${DURABLE_ROOT}/compact_records"
module purge
module load cuda/12.4

export IMIM_QUALIFICATION_CONTAINER="${CONTAINER}"
export IMIM_QUALIFICATION_RECEIPT="${RECEIPT}"
export IMIM_QUALIFICATION_DURABLE_RECEIPT="${DURABLE_RECEIPT}"

/usr/bin/python3 - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any


def run(command: list[str], *, timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "exit_code": None,
            "stderr": str(exc),
            "stdout": "",
        }
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stderr": completed.stderr.strip(),
        "stdout": completed.stdout.strip(),
    }


container = os.environ["IMIM_QUALIFICATION_CONTAINER"]
receipt_path = Path(os.environ["IMIM_QUALIFICATION_RECEIPT"])
durable_receipt_path = Path(os.environ["IMIM_QUALIFICATION_DURABLE_RECEIPT"])
torch_probe = r"""
import json
import torch

device = torch.device('cuda:0')
left = torch.arange(4096, dtype=torch.float32, device=device).reshape(64, 64)
right = torch.eye(64, dtype=torch.float32, device=device)
result = left @ right
torch.cuda.synchronize()
print(json.dumps({
    'cuda_available': torch.cuda.is_available(),
    'cuda_device_count': torch.cuda.device_count(),
    'device_name': torch.cuda.get_device_name(0),
    'device_capability': list(torch.cuda.get_device_capability(0)),
    'matmul_exact': bool(torch.equal(result, left)),
    'pytorch_version': torch.__version__,
    'pytorch_cuda_version': torch.version.cuda,
}, sort_keys=True))
"""

checks = {
    "git_embodiedgen_head": run(
        [
            "git",
            "ls-remote",
            "https://github.com/HorizonRobotics/EmbodiedGen.git",
            "HEAD",
        ]
    ),
    "huggingface_head": run(
        ["curl", "-LIs", "--max-time", "15", "https://huggingface.co"]
    ),
    "nvidia_smi": run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    ),
    "singularity_version": run(["singularity", "--version"]),
    "torch_cuda": run(
        ["singularity", "exec", "--nv", container, "python3", "-c", torch_probe],
        timeout=120,
    ),
}
required = ("git_embodiedgen_head", "huggingface_head", "nvidia_smi", "torch_cuda")
status = "passed" if all(checks[name]["exit_code"] == 0 for name in required) else "failed"
receipt = {
    "schema": "InteractMoveInterMimicJunoQualification",
    "schema_version": 1,
    "status": status,
    "slurm": {
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "visible_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
    },
    "host": {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    },
    "paths": {
        "container": container,
        "container_bytes": Path(container).stat().st_size if Path(container).is_file() else None,
        "durable_root": "/work/dal503972/interactmove_intermimic",
        "scratch_root": "/scratch/juno/dal503972/interactmove_intermimic",
    },
    "credential_presence": {
        name: bool(os.environ.get(name))
        for name in (
            "API_KEY",
            "API_VERSION",
            "AZURE_OPENAI_API_KEY",
            "DASHSCOPE_API_KEY",
            "ENDPOINT",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "MODEL_NAME",
            "OPENAI_API_KEY",
        )
    },
    "executables": {
        name: shutil.which(name)
        for name in ("curl", "git", "nvidia-smi", "singularity")
    },
    "checks": checks,
}
encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
receipt_path.write_text(encoded, encoding="utf-8")
durable_receipt_path.write_text(encoded, encoding="utf-8")
print(json.dumps({"durable_receipt": str(durable_receipt_path), "status": status}, sort_keys=True))
raise SystemExit(0 if status == "passed" else 1)
PY
