#!/bin/bash -l
#SBATCH --job-name=p8-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
umask 077
REPRO_DIR="${REPRO_DIR:?}"; P8_PHASE="${P8_PHASE:?}"
python3 "$REPRO_DIR/storage.py" check-juno
D="$(python3 "$REPRO_DIR/storage.py" root durable)"; S="$(python3 "$REPRO_DIR/storage.py" root scratch)"
SRC="$S/caches/p2_source/egobabyvlm"; PIXI="$S/caches/p2_tools/pixi-0.70.0/pixi"; REC="$D/run_records/pilot_p0/p0-4cc3af23/pilot_p8"
mkdir -p "$REC" "$D/logs/pilot_p8/p8-4a20f18d"; chmod 700 "$REC" "$D/logs/pilot_p8/p8-4a20f18d"
export PIXI_HOME="$S/caches/p2_pixi" PIXI_CACHE_DIR="$S/caches/p2_runtime_cache/pixi" XDG_CACHE_HOME="$S/caches/p2_runtime_cache/xdg" HF_HOME="$S/caches/p2_runtime_cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true
args=("$P8_PHASE" --config "$REPRO_DIR/configs/pilot_p8.json" --scratch-root "$S" --durable-root "$D" --source-root "$SRC" --checkpoint "$D/checkpoints/pilot_p7/p7-7b9e2c41/step_130.pt" --p7-config "$D/checkpoints/pilot_p7/p7-7b9e2c41/pilot_p7.json" --p5-training-dir "$S/runs/pilot_p5/p5-91c43e2a/training" --p6-checkpoint "$D/checkpoints/pilot_p6/p6-6d31a4e7/step_100.pt" --tokenizer "$D/checkpoints/pilot_p6/p6-6d31a4e7/tokenizer.json" --vocab "$D/checkpoints/pilot_p6/p6-6d31a4e7/vocab.txt" --p7-completion "$D/run_records/pilot_p0/p0-4cc3af23/pilot_p7/completion.json" --p7-audit "$D/run_records/pilot_p0/p0-4cc3af23/pilot_p7/hardening_audit.json" --p7-inventory "$D/run_records/pilot_p0/p0-4cc3af23/pilot_p7/retained_inventory.json" --p4-selection "$REPRO_DIR/pilots/juno_sample/p4_selection.json" --record-root "$REC")
[[ "$P8_PHASE" = evaluate ]] && args+=(--qualification "$REC/qualification.json")
"$PIXI" run --manifest-path "$SRC/pixi.toml" --environment default -- python "$REPRO_DIR/scripts/pilot_p8.py" "${args[@]}"
