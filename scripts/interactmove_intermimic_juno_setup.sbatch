#!/usr/bin/env bash
#SBATCH --job-name=imim-setup
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/juno/dal503972/interactmove_intermimic/logs/setup-%j.out
#SBATCH --error=/scratch/juno/dal503972/interactmove_intermimic/logs/setup-%j.err

set -euo pipefail
umask 077

readonly EMBODIEDGEN_COMMIT=9b333554254af196bace88c1a171a3bf047fa09c
readonly SCRATCH_ROOT=/scratch/juno/dal503972/interactmove_intermimic
readonly DURABLE_ROOT=/work/dal503972/interactmove_intermimic
readonly SOURCE_ROOT="${SCRATCH_ROOT}/source/EmbodiedGen"
readonly ENV_ROOT="${SCRATCH_ROOT}/environments/embodiedgen-v2.0.1-py310-cu126"
readonly RUN_ROOT="${SCRATCH_ROOT}/runs/setup/${SLURM_JOB_ID}"
readonly BASIC_MARKER="${ENV_ROOT}/.embodiedgen-basic-${EMBODIEDGEN_COMMIT}"
readonly MANIFEST="${DURABLE_ROOT}/manifests/embodiedgen_environment_${SLURM_JOB_ID}.txt"
readonly RECEIPT="${DURABLE_ROOT}/compact_records/embodiedgen_setup_${SLURM_JOB_ID}.json"

install -d -m 700 \
    "${RUN_ROOT}" \
    "${SCRATCH_ROOT}/caches/conda" \
    "${SCRATCH_ROOT}/caches/huggingface" \
    "${SCRATCH_ROOT}/caches/pip" \
    "${SCRATCH_ROOT}/downloads" \
    "${SCRATCH_ROOT}/environments" \
    "${SCRATCH_ROOT}/source" \
    "${DURABLE_ROOT}/compact_records" \
    "${DURABLE_ROOT}/manifests"

export CONDA_PKGS_DIRS="${SCRATCH_ROOT}/caches/conda"
export HF_HOME="${SCRATCH_ROOT}/caches/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/caches/pip"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export EMBODIEDGEN_TORCH_CUDA_ARCH_LIST=9.0
export TORCH_CUDA_ARCH_LIST=9.0
export TCNN_CUDA_ARCHITECTURES=90
export IMIM_SETUP_MANIFEST="${MANIFEST}"
export IMIM_SETUP_RECEIPT="${RECEIPT}"
export IMIM_SETUP_SOURCE="${SOURCE_ROOT}"
export IMIM_SETUP_ENV="${ENV_ROOT}"
export IMIM_SETUP_STATUS=failed
export IMIM_SETUP_STARTED_UTC
IMIM_SETUP_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

write_receipt() {
    /usr/bin/python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


def output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


manifest = Path(os.environ["IMIM_SETUP_MANIFEST"])
source = Path(os.environ["IMIM_SETUP_SOURCE"])
environment = Path(os.environ["IMIM_SETUP_ENV"])
manifest_digest = None
if manifest.is_file():
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
receipt = {
    "schema": "InteractMoveInterMimicEmbodiedGenSetup",
    "schema_version": 1,
    "status": os.environ["IMIM_SETUP_STATUS"],
    "started_utc": os.environ["IMIM_SETUP_STARTED_UTC"],
    "finished_utc": output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
    "slurm": {
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "visible_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
    },
    "source": {
        "path": str(source),
        "expected_commit": "9b333554254af196bace88c1a171a3bf047fa09c",
        "actual_commit": output(["git", "-C", str(source), "rev-parse", "HEAD"]),
    },
    "environment": {
        "path": str(environment),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_digest,
        "python": output([str(environment / "bin/python"), "--version"]),
    },
}
path = Path(os.environ["IMIM_SETUP_RECEIPT"])
path.write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}
trap write_receipt EXIT

source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    conda create --prefix "${ENV_ROOT}" python=3.10.13 -y
fi
set +u
conda activate "${ENV_ROOT}"
set -u

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
    git clone https://github.com/HorizonRobotics/EmbodiedGen.git "${SOURCE_ROOT}"
fi
if [[ -n "$(git -C "${SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to modify a dirty EmbodiedGen source checkout: ${SOURCE_ROOT}" >&2
    exit 1
fi
git -C "${SOURCE_ROOT}" fetch origin "${EMBODIEDGEN_COMMIT}"
git -C "${SOURCE_ROOT}" checkout --detach "${EMBODIEDGEN_COMMIT}"

cd "${SOURCE_ROOT}"
if [[ ! -f "${ENV_ROOT}/etc/conda/activate.d/cuda126.sh" ]]; then
    bash install.sh cu126
fi
set +u
conda deactivate
conda activate "${ENV_ROOT}"
set -u
export EMBODIEDGEN_TORCH_CUDA_ARCH_LIST=9.0
export TORCH_CUDA_ARCH_LIST=9.0
export TCNN_CUDA_ARCHITECTURES=90
# EmbodiedGen's installer uses the PyTorch CUDA index as its only index.  That
# index does not carry ordinary build dependencies, so resolve those from PyPI
# first and install the pinned CUDA wheels with PyPI as an explicit fallback.
if [[ ! -f "${BASIC_MARKER}" ]] && ! (
    python -c 'import torch; import sapien; import mani_skill; import coacd; import open3d; import trimesh'
    layout-cli --help >/dev/null
    sim-cli --help >/dev/null
); then
    python -m pip install --upgrade \
        'pip>=25.1,<26' \
        'setuptools<81' \
        wheel
    python -m pip install \
        'flit_core>=3.11,<4' \
        'jinja2>=3.1' \
        'typing_extensions>=4.10'
    python -m pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple
    bash install.sh basic
fi
printf '%s\n' "${EMBODIEDGEN_COMMIT}" > "${BASIC_MARKER}"
if ! python -c 'import flash_attn; assert flash_attn.__version__ == "2.8.2"'; then
    FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install \
        --no-cache-dir \
        --no-build-isolation \
        --no-binary flash-attn \
        --force-reinstall \
        --no-deps \
        flash-attn==2.8.2
fi
set +u
conda deactivate
conda activate "${ENV_ROOT}"
set -u

python - <<'PY'
import json

import torch
import sapien
import mani_skill
import coacd
import open3d
import trimesh

assert torch.cuda.is_available()
tensor = torch.eye(32, device="cuda")
assert bool(torch.equal(tensor @ tensor, tensor))
print(json.dumps({
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "cuda_device": torch.cuda.get_device_name(0),
    "pytorch": torch.__version__,
    "pytorch_cuda": torch.version.cuda,
    "sapien": getattr(sapien, "__version__", None),
}, sort_keys=True))
PY
layout-cli --help >/dev/null
sim-cli --help >/dev/null

{
    printf 'embodiedgen_commit=%s\n' "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
    printf 'torch_cuda_arch_list=%s\n' "${TORCH_CUDA_ARCH_LIST}"
    conda list --explicit
    printf '\n[pip-freeze]\n'
    python -m pip freeze --all
} > "${MANIFEST}"

export IMIM_SETUP_STATUS=passed
