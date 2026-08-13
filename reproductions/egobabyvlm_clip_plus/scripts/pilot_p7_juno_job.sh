#!/bin/bash -l
#SBATCH --job-name=p7-clip-plus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
umask 077
REPRO_DIR="${REPRO_DIR:?}"; P7_PHASE="${P7_PHASE:?}"
python3 "$REPRO_DIR/storage.py" check-juno
DURABLE_ROOT="$(python3 "$REPRO_DIR/storage.py" root durable)"; SCRATCH_ROOT="$(python3 "$REPRO_DIR/storage.py" root scratch)"
SOURCE_ROOT="$SCRATCH_ROOT/caches/p2_source/egobabyvlm"; PIXI="$SCRATCH_ROOT/caches/p2_tools/pixi-0.70.0/pixi"
RUN_ROOT="$SCRATCH_ROOT/runs/pilot_p7/p7-7b9e2c41"; RECORD_ROOT="$DURABLE_ROOT/run_records/pilot_p0/p0-4cc3af23/pilot_p7"; LOG_ROOT="$DURABLE_ROOT/logs/pilot_p7/p7-7b9e2c41"
mkdir -p "$RUN_ROOT" "$RECORD_ROOT" "$LOG_ROOT"; chmod 700 "$RUN_ROOT" "$RECORD_ROOT" "$LOG_ROOT"
export PIXI_HOME="$SCRATCH_ROOT/caches/p2_pixi" PIXI_CACHE_DIR="$SCRATCH_ROOT/caches/p2_runtime_cache/pixi" XDG_CACHE_HOME="$SCRATCH_ROOT/caches/p2_runtime_cache/xdg" HF_HOME="$SCRATCH_ROOT/caches/p2_runtime_cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 30000)) RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
args=("$P7_PHASE" --config "$REPRO_DIR/configs/pilot_p7.json" --source-root "$SOURCE_ROOT" --p3-root "$SCRATCH_ROOT/runs/pilot_p3/p3-3d96f71c" --p5-checkpoint "$DURABLE_ROOT/checkpoints/pilot_p5/p5-91c43e2a/model_final.rank_0.pth" --p5-training-dir "$SCRATCH_ROOT/runs/pilot_p5/p5-91c43e2a/training" --p6-checkpoint "$DURABLE_ROOT/checkpoints/pilot_p6/p6-6d31a4e7/step_100.pt" --tokenizer "$DURABLE_ROOT/checkpoints/pilot_p6/p6-6d31a4e7/tokenizer.json" --vocab "$DURABLE_ROOT/checkpoints/pilot_p6/p6-6d31a4e7/vocab.txt" --scratch-root "$SCRATCH_ROOT" --durable-root "$DURABLE_ROOT" --run-root "$RUN_ROOT" --record-root "$RECORD_ROOT" --aggregate-output "$DURABLE_ROOT/aggregate_results/pilot_p7_p7-7b9e2c41.json")
target=65; [[ "$P7_PHASE" = resume ]] && target=130
[[ "$P7_PHASE" != finalize ]] && args+=(--target "$target")
"$PIXI" run --manifest-path "$SOURCE_ROOT/pixi.toml" --environment default -- python "$REPRO_DIR/scripts/pilot_p7.py" "${args[@]}"
