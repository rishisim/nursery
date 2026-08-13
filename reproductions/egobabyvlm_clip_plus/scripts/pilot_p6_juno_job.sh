#!/bin/bash -l
#SBATCH --job-name=p6-bert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=true

REPRO_DIR="${REPRO_DIR:?Set REPRO_DIR to the checked-out reproduction directory}"
python3 "$REPRO_DIR/storage.py" check-juno
DURABLE_ROOT="$(python3 "$REPRO_DIR/storage.py" root durable)"
SCRATCH_ROOT="$(python3 "$REPRO_DIR/storage.py" root scratch)"
PYTHON_BIN="${P6_PYTHON_BIN:?Set P6_PYTHON_BIN to the verified P2 environment Python}"
PHASE="${P6_PHASE:?Set P6_PHASE to health, resume, or finalize}"
COMMON=(--config "$REPRO_DIR/configs/pilot_p6.json" --scratch-root "$SCRATCH_ROOT" --durable-root "$DURABLE_ROOT")
case "$PHASE" in
  health) exec "$PYTHON_BIN" "$REPRO_DIR/scripts/pilot_p6.py" train "${COMMON[@]}" --target-step 50 ;;
  resume) exec "$PYTHON_BIN" "$REPRO_DIR/scripts/pilot_p6.py" train "${COMMON[@]}" --resume-step 50 --target-step 100 ;;
  finalize) exec "$PYTHON_BIN" "$REPRO_DIR/scripts/pilot_p6.py" finalize "${COMMON[@]}" ;;
  *) echo "Unknown P6_PHASE" >&2; exit 2 ;;
esac
