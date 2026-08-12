#!/bin/bash
# Canonical Pilot P2 GPU job. Submit only after storage.py check-juno passes.
# Required exported variables: P2_SCRATCH_ROOT, P2_DURABLE_ROOT, P2_RUN_ROOT.
set -euo pipefail
umask 077

: "${P2_SCRATCH_ROOT:?storage.py-resolved scratch root required}"
: "${P2_DURABLE_ROOT:?storage.py-resolved durable root required}"
: "${P2_RUN_ROOT:?governed P2 run root required}"

case "$P2_RUN_ROOT" in
  "$P2_SCRATCH_ROOT"/runs/pilot_p2/*) ;;
  *) echo "P2 run root is outside governed scratch" >&2; exit 2 ;;
esac

cache_root="$P2_SCRATCH_ROOT/caches"
source_root="$cache_root/p2_source/egobabyvlm"
pixi="$cache_root/p2_tools/pixi-0.70.0/pixi"
detail="$P2_DURABLE_ROOT/run_records/pilot_p2/p2-7e49c8a1.json"

test -x "$pixi"
test "$(git -C "$source_root" rev-parse HEAD)" = "224621caf0628270b6115845ac75a65b984234a3"
test "$(sha256sum "$source_root/pixi.lock" | cut -d' ' -f1)" = \
  "360c372997ea7f9ba89abcf87796e57ebcd2fdbc9734489666120073752e7244"

export PIXI_HOME="$cache_root/p2_pixi"
export PIXI_CACHE_DIR="$cache_root/p2_runtime_cache/pixi"
export XDG_CACHE_HOME="$cache_root/p2_runtime_cache/xdg"
export HF_HOME="$cache_root/p2_runtime_cache/huggingface"

"$pixi" install --manifest-path "$source_root/pixi.toml" --locked --environment default

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true

"$pixi" run --manifest-path "$source_root/pixi.toml" --environment default -- \
  python "$P2_RUN_ROOT/scripts/pilot_p2_preflight.py" \
    --config "$P2_RUN_ROOT/configs/pilot_p2.json" \
    --source-root "$source_root" \
    --scratch-root "$P2_SCRATCH_ROOT" \
    --durable-root "$P2_DURABLE_ROOT" \
    --output "$detail"

chmod 600 "$detail"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  chmod 600 "$P2_DURABLE_ROOT/logs/pilot_p2/p2-7e49c8a1-$SLURM_JOB_ID.log"
fi
