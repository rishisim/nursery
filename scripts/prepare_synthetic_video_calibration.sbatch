#!/bin/bash
#SBATCH --job-name=synthetic-calibration-prepare
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/extractor-repair"
mkdir -p "$output_root/logs"
exec >"$output_root/logs/prepare-${SLURM_JOB_ID}.out" 2>"$output_root/logs/prepare-${SLURM_JOB_ID}.err"
scratch="${TMPDIR:-/tmp}/synthetic-calibration-prepare-${SLURM_JOB_ID}"
mkdir -p "$scratch/pydeps"
trap 'find "$scratch" -mindepth 1 -delete 2>/dev/null || true; rmdir "$scratch" 2>/dev/null || true' EXIT
tar -C "$scratch/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/source:$scratch/pydeps"
singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" prepare-public \
  --public-root "${PHASE4_PUBLIC_ROOT}" \
  --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
