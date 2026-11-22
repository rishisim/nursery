#!/usr/bin/env bash
#SBATCH --job-name=nursery-step9
#SBATCH --partition=h200,h100,a30
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:30:00
#SBATCH --array=0-84%8
#SBATCH --no-requeue

set -euo pipefail
umask 077
project=/work/dal503972/nursery-step7
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Prepare Step 9 in the private runtime before submitting this array.
"$project/step7-python" -m scripts.step9_acquisition_curves run --task "${SLURM_ARRAY_TASK_ID:?}"
