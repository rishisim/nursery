#!/usr/bin/env bash
#SBATCH --job-name=nursery-temporal-init
#SBATCH --partition=h200,h100,a30
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%8
#SBATCH --no-requeue

set -euo pipefail
umask 077
project=/work/dal503972/nursery-step7
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
modes=(zero per_word)
arms=(video video_gaze)
task=${SLURM_ARRAY_TASK_ID:?}
# Prepare both modes with prepare-temporal-init in the private Step 7 runtime first.
"$project/step7-python" -m scripts.step7_word_types run-temporal-init \
    --temporal-weight-init "${modes[task / 4]}" --arm "${arms[task / 2 % 2]}" --seed "$((task % 2))"
