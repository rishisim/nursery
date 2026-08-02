#!/bin/bash
#SBATCH --job-name=lean-real-prepare
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
tmp="/tmp/lean-real-${SLURM_JOB_ID}"
mkdir -p "$tmp/pydeps" "$tmp/bin"
trap 'find "$tmp" -mindepth 1 -delete 2>/dev/null || true; rmdir "$tmp" 2>/dev/null || true' EXIT
tar -C "$tmp/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/source:$tmp/pydeps"
ffmpeg_exe=$(singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
ln -s "$ffmpeg_exe" "$tmp/bin/ffmpeg"
export PATH="$tmp/bin:$PATH" SINGULARITYENV_PREPEND_PATH="$tmp/bin"
singularity exec --nv \
  --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${PHASE4_RESTRICTED_ROOT}:${PHASE4_RESTRICTED_ROOT}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_lean_pilot.py" prepare \
    --root "${PHASE4_RESTRICTED_ROOT}/real_only_proof" --public "${PHASE4_PUBLIC_ROOT}" \
    --credited-hours "${CREDITED_HOURS:-1}"
