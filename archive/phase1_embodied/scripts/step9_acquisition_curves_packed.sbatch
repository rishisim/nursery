#!/usr/bin/env bash
# Run three Step 9 queue tasks at once on one GPU (H100/H200 only; A30 memory is too small).
# Usage: sbatch --array=0-N%4 step9_acquisition_curves_packed.sbatch FIRST_TASK
#SBATCH --job-name=nursery-step9-packed
#SBATCH --partition=h200,h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 077
project=/work/dal503972/nursery-step7
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
first=${1:?first queue task}
pids=()
for slot in 0 1 2; do
    task=$((first + 3 * SLURM_ARRAY_TASK_ID + slot))
    [ "$task" -le 84 ] || break
    "$project/step7-python" -m scripts.step9_acquisition_curves run --task "$task" &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
