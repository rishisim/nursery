#!/bin/bash
set -euo pipefail
umask 077
: "${P4_SCRATCH_ROOT:?storage.py-resolved scratch root required}"
: "${P4_DURABLE_ROOT:?storage.py-resolved durable root required}"
: "${P4_CODE_ROOT:?submitted code root required}"
ns="$P4_SCRATCH_ROOT/evaluation_data/pilot_p4_calibration_only"
source_root="$P4_SCRATCH_ROOT/caches/p2_source/egobabyvlm"
pixi="$P4_SCRATCH_ROOT/caches/p2_tools/pixi-0.70.0/pixi"
case "$ns" in "$P4_SCRATCH_ROOT/evaluation_data/pilot_p4_calibration_only") ;; *) exit 2;; esac
export HOME="$ns/runtime_home" XDG_CACHE_HOME="$ns/runtime_home/.cache"
export HF_HOME="$ns/caches/huggingface" TORCH_HOME="$ns/caches/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true
mkdir -p "$HOME/.cache/clip" "$ns/outputs"
ln -sfn "$ns/models/ViT-B-16.pt" "$HOME/.cache/clip/ViT-B-16.pt"
"$pixi" run --manifest-path "$source_root/pixi.toml" --environment default -- \
  python "$P4_CODE_ROOT/scripts/pilot_p4.py" run \
  --config "$P4_CODE_ROOT/configs/pilot_p4.json" --scratch "$P4_SCRATCH_ROOT" \
  --durable "$P4_DURABLE_ROOT" --source "$source_root" \
  --archive "$ns/archive/MachineDevBench.tar" --data-root "$ns/extracted/MachineDevBench" \
  --output "$ns/outputs" --run-id p4-3a56d9c1
