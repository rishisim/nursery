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
singularity pull --disable-cache \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  docker://pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
find "${PHASE4_PUBLIC_ROOT}/pydeps" -mindepth 1 -delete 2>/dev/null || true
singularity exec "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python -m pip install --disable-pip-version-check \
  --target "${PHASE4_PUBLIC_ROOT}/pydeps" \
  diffusers==0.39.0 transformers==4.57.6 accelerate==1.14.0 \
  sentencepiece protobuf huggingface-hub==0.36.2 safetensors
PYTHONPATH="${PHASE4_PUBLIC_ROOT}/pydeps" \
  singularity exec "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python -c 'import torch,diffusers,transformers; from diffusers import Flux2KleinPipeline; print("CONTAINER_ENVIRONMENT_READY")'
