#!/usr/bin/env bash
#SBATCH --job-name=nursery-noun-drop
#SBATCH --partition=h200,h100,a30
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue

set -euo pipefail
umask 077
project=/work/dal503972/nursery-step7
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cells=(A A B B)
task=${SLURM_ARRAY_TASK_ID:?}
# Existing private Step 7 runtime binds project/repo at the frozen Mac repo path.
# Reuse its world16/train_cache, step7_compact, gaze.parquet, frame_partitions.json
# and clean_856/frames.parquet. Each cell freeze contains its own speech and panels.
# Only synthetic GRAB inputs are used. Preserve symlinks when syncing the freezes.
"$project/step7-python" -m scripts.step7_word_types run-cell --cell "${cells[task]}" --seed "$((task % 2))"
