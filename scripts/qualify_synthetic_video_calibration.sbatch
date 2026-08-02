#!/bin/bash
#SBATCH --job-name=synthetic-calibration-qualify
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HOME="${PHASE4_PUBLIC_ROOT}/models/pe-hf-home"
output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/extractor-repair"
mkdir -p "$output_root/logs"
exec >"$output_root/logs/qualify-${SLURM_JOB_ID}.out" 2>"$output_root/logs/qualify-${SLURM_JOB_ID}.err"
scratch="${TMPDIR:-/tmp}/synthetic-calibration-qualify-${SLURM_JOB_ID}"
mkdir -p "$scratch/pydeps"
trap 'find "$scratch" -mindepth 1 -delete 2>/dev/null || true; rmdir "$scratch" 2>/dev/null || true' EXIT
tar -C "$scratch/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/calibration-repair-pydeps:${PHASE4_PUBLIC_ROOT}/source:${PHASE4_PUBLIC_ROOT}/models/evaluator-pydeps:$scratch/pydeps:${PHASE4_EGOBABY_ROOT}"
singularity exec --nv --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${PHASE4_EGOBABY_ROOT}:${PHASE4_EGOBABY_ROOT},${scratch}:${scratch}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" qualify-public \
  --public-root "${PHASE4_PUBLIC_ROOT}" \
  --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
