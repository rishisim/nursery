#!/bin/bash
#SBATCH --job-name=lean-real-learner
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HOME="${PHASE4_PUBLIC_ROOT}/models/learner-hf-home"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29631 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
export CC="${PHASE4_PUBLIC_ROOT}/models/triton-compiler/bin/x86_64-conda-linux-gnu-cc"
export PATH="${PHASE4_PUBLIC_ROOT}/models/triton-compiler/bin:$PATH"
log_root="${PHASE4_RESTRICTED_ROOT}/real_only_proof/logs"
mkdir -p "$log_root"
exec >"${log_root}/learner-${SLURM_JOB_ID}.out" 2>"${log_root}/learner-${SLURM_JOB_ID}.err"
tmp="/tmp/lean-learner-${SLURM_JOB_ID}"
mkdir -p "$tmp/pydeps"
trap 'find "$tmp" -mindepth 1 -delete 2>/dev/null || true; rmdir "$tmp" 2>/dev/null || true' EXIT
tar -C "$tmp/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/source:${PHASE4_PUBLIC_ROOT}/models/learner-pydeps:${PHASE4_PUBLIC_ROOT}/models/evaluator-pydeps:$tmp/pydeps:${PHASE4_EGOBABY_ROOT}:${PHASE4_EGOBABY_ROOT}/apps/baselines/dinov2/third_party"
singularity exec --nv --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${PHASE4_RESTRICTED_ROOT}:${PHASE4_RESTRICTED_ROOT},${PHASE4_EGOBABY_ROOT}:${PHASE4_EGOBABY_ROOT}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_lean_learner.py" \
  --upstream "${PHASE4_EGOBABY_ROOT}" --root "${PHASE4_RESTRICTED_ROOT}/real_only_proof" --public "${PHASE4_PUBLIC_ROOT}" --restricted "${PHASE4_RESTRICTED_ROOT}" --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
