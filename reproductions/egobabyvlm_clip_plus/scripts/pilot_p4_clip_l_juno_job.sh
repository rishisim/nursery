#!/bin/bash
set -euo pipefail
umask 077
: "${P4_SCRATCH_ROOT:?storage.py-resolved scratch root required}"
: "${P4_DURABLE_ROOT:?storage.py-resolved durable root required}"
: "${P4_CODE_ROOT:?submitted code root required}"
base="$P4_SCRATCH_ROOT/evaluation_data/pilot_p4_calibration_only"
ns="$base/clip_l_diagnostic"
source_root="$P4_SCRATCH_ROOT/caches/p2_source/egobabyvlm"
pixi="$P4_SCRATCH_ROOT/caches/p2_tools/pixi-0.70.0/pixi"
export HOME="$ns/runtime_home" XDG_CACHE_HOME="$ns/runtime_home/.cache"
export HF_HOME="$ns/caches/huggingface" TORCH_HOME="$ns/caches/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true
mkdir -p "$HOME" "$ns/outputs"
"$pixi" run --manifest-path "$source_root/pixi.toml" --environment default -- \
  python "$P4_CODE_ROOT/scripts/pilot_p4_clip_l_diagnostic.py" \
  --config "$P4_CODE_ROOT/configs/pilot_p4_clip_l_diagnostic.json" \
  --scratch "$P4_SCRATCH_ROOT" --durable "$P4_DURABLE_ROOT" --source "$source_root" \
  --archive "$base/archive/MachineDevBench.tar" --data-root "$base/extracted/MachineDevBench" \
  --output "$ns/outputs" --run-id p4l-6c16b754
