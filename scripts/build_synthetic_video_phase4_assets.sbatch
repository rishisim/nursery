#!/bin/bash
#SBATCH --job-name=phase4-common-assets
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
tmp="/tmp/phase4-build-${SLURM_JOB_ID}"
mkdir -p "$tmp/pydeps"
trap 'find "$tmp" -mindepth 1 -delete 2>/dev/null || true; rmdir "$tmp" 2>/dev/null || true' EXIT
tar -C "$tmp/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
export PYTHONPATH="$tmp/pydeps:${PHASE4_EGOBABY_ROOT}"
export PATH="$tmp/pydeps/bin:$PATH"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29543
srun --kill-on-bad-exit=1 singularity exec --nv \
  --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${PHASE4_RESTRICTED_ROOT}:${PHASE4_RESTRICTED_ROOT},${PHASE4_EGOBABY_ROOT}:${PHASE4_EGOBABY_ROOT}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python "${PHASE4_PUBLIC_ROOT}/source/build_synthetic_video_phase4_assets.py"
