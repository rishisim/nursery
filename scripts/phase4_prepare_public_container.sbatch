#!/bin/bash
#SBATCH --job-name=phase4-public-container
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
local_tmp="/tmp/phase4-container-${SLURM_JOB_ID}"
mkdir -p "$local_tmp"
cleanup() {
  find "$local_tmp" -mindepth 1 -delete 2>/dev/null || true
  rmdir "$local_tmp" 2>/dev/null || true
}
trap cleanup EXIT
export SINGULARITY_TMPDIR="$local_tmp"
export TMPDIR="$local_tmp"
container_path="${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif"
if [[ ! -f "$container_path" ]]; then
  singularity pull --disable-cache "$container_path" \
    docker://pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
fi
local_pydeps="${local_tmp}/pydeps"
mkdir -p "$local_pydeps"
singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" \
  "$container_path" \
  python -m pip install --disable-pip-version-check --no-deps \
  --target "$local_pydeps" \
  diffusers==0.39.0 transformers==4.57.6 accelerate==1.14.0 \
  tokenizers==0.22.2 regex==2024.11.6 sentencepiece protobuf \
  huggingface-hub==0.36.2 safetensors \
  httpx==0.28.1 httpcore==1.0.9 anyio==4.9.0 certifi==2025.4.26 \
  idna==3.10 h11==0.16.0 sniffio==1.3.1
PYTHONPATH="$local_pydeps" \
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" \
  "$container_path" \
  python -c 'import torch,diffusers,transformers; from diffusers import Flux2KleinPipeline; print("CONTAINER_ENVIRONMENT_READY")'
tar -C "$local_pydeps" -cf "${local_tmp}/pydeps.tar" .
cp "${local_tmp}/pydeps.tar" "${PHASE4_PUBLIC_ROOT}/models/phase4-pydeps.tar"
