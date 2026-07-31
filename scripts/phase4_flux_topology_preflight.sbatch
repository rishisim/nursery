#!/bin/bash
#SBATCH --job-name=phase4-flux-public-preflight
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HOME="${PHASE4_PUBLIC_ROOT}/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/pydeps"
master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
export MASTER_ADDR="$master_addr"
export MASTER_PORT=29541
srun --kill-on-bad-exit=1 singularity exec --nv \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python "${PHASE4_PUBLIC_ROOT}/source/phase4_flux_topology_preflight.py"
