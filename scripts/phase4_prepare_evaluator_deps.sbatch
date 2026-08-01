#!/bin/bash
#SBATCH --job-name=phase4-eval-public
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1
target="${PHASE4_PUBLIC_ROOT}/models/evaluator-pydeps"
mkdir -p "$target"
find "$target" -mindepth 1 -delete
singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python -m pip install --no-input --no-deps --target "$target" \
    hydra-core==1.3.2 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 \
    submitit==1.5.4 posix_ipc==1.3.2 cloudpickle==3.1.2 \
    einops==0.8.2 timm==1.0.26 ftfy==6.3.1 wcwidth==0.8.2 \
    'https://github.com/facebookresearch/stopes/archive/e582066bc542caddbef12aa4b008610bbafc4ba1.tar.gz'
singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" \
  --env PYTHONPATH="$target:${PHASE4_EGOBABY_ROOT}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
  python -c 'from evaluation.multimodal.machine_devbench.base import MachineDevBenchEvalModule; print("EVALUATOR_IMPORT_PASS")'
find "$target" -type f -exec chmod 600 {} +
find "$target" -type d -exec chmod 700 {} +
