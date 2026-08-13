#!/bin/bash
# Canonical Pilot P5 GPU segment. Required roots are storage.py-resolved.
set -euo pipefail
umask 077
: "${P5_SCRATCH_ROOT:?}" "${P5_DURABLE_ROOT:?}" "${P5_CODE_ROOT:?}" "${P5_FRAMES_ROOT:?}" "${P5_STOP_STEP:?}"
case "$P5_CODE_ROOT" in "$P5_SCRATCH_ROOT"/runs/pilot_p5/*) ;; *) exit 2;; esac
case "$P5_FRAMES_ROOT" in "$P5_SCRATCH_ROOT"/runs/pilot_p3/p3-3d96f71c|"$P5_SCRATCH_ROOT"/runs/pilot_p3/p3-3d96f71c/*) ;; *) exit 2;; esac
source_root="$P5_SCRATCH_ROOT/caches/p2_source/egobabyvlm"
pixi="$P5_SCRATCH_ROOT/caches/p2_tools/pixi-0.70.0/pixi"
run_root="$P5_SCRATCH_ROOT/runs/pilot_p5/p5-91c43e2a"
durable="$P5_DURABLE_ROOT/run_records/pilot_p0/p0-4cc3af23/pilot_p5"
mkdir -p "$run_root" "$durable" "$P5_DURABLE_ROOT/logs/pilot_p5/p5-91c43e2a"
chmod 700 "$run_root" "$durable" "$P5_DURABLE_ROOT/logs/pilot_p5/p5-91c43e2a"
test "$(git -C "$source_root" rev-parse HEAD)" = 224621caf0628270b6115845ac75a65b984234a3
test "$(sha256sum "$source_root/pixi.lock"|cut -d' ' -f1)" = 360c372997ea7f9ba89abcf87796e57ebcd2fdbc9734489666120073752e7244
export PIXI_HOME="$P5_SCRATCH_ROOT/caches/p2_pixi" PIXI_CACHE_DIR="$P5_SCRATCH_ROOT/caches/p2_runtime_cache/pixi" XDG_CACHE_HOME="$P5_SCRATCH_ROOT/caches/p2_runtime_cache/xdg" HF_HOME="$P5_SCRATCH_ROOT/caches/p2_runtime_cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true
args=(--config "$P5_CODE_ROOT/configs/pilot_p5.json" --source-root "$source_root" --frames-root "$P5_FRAMES_ROOT" --scratch-root "$P5_SCRATCH_ROOT" --durable-root "$P5_DURABLE_ROOT" --run-root "$run_root" --detail "$durable/detail.json" --completion "$durable/completion.json" --stop-step "$P5_STOP_STEP")
[[ "$P5_STOP_STEP" = 100 ]] && args+=(--resume)
"$pixi" run --manifest-path "$source_root/pixi.toml" --environment default -- python "$P5_CODE_ROOT/scripts/pilot_p5.py" "${args[@]}"
