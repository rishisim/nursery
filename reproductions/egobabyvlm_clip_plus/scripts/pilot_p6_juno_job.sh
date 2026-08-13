#!/bin/bash -l
#SBATCH --job-name=p6-bert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail

REPRO_DIR="${REPRO_DIR:?Set REPRO_DIR to the checked-out reproduction directory}"
python3 "$REPRO_DIR/storage.py" check-juno
DURABLE_ROOT="$(python3 "$REPRO_DIR/storage.py" root durable)"
SCRATCH_ROOT="$(python3 "$REPRO_DIR/storage.py" root scratch)"
exec pixi run -e dev python "$REPRO_DIR/scripts/pilot_p6.py" prepare-tokenizer \
  --config "$REPRO_DIR/configs/pilot_p6.json" \
  --scratch-root "$SCRATCH_ROOT" --durable-root "$DURABLE_ROOT"
