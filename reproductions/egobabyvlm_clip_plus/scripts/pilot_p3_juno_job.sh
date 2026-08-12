#!/bin/bash
# Canonical Pilot P3 GPU job. Submit only after storage.py check-juno passes.
set -euo pipefail
umask 077

: "${P3_SCRATCH_ROOT:?storage.py-resolved scratch root required}"
: "${P3_DURABLE_ROOT:?storage.py-resolved durable root required}"
: "${P3_LEDGER:?owner-only governed two-record ledger required}"
: "${P3_MEDIA_ROOT:?governed raw media root required}"
: "${P3_CODE_ROOT:?submitted canonical P3 code root required}"
: "${P3_SLOT:=all}"
: "${P3_LOG_DIR:?owner-only durable log directory required}"

P3_LOG_PATH="$P3_LOG_DIR/${SLURM_JOB_NAME:-pilot_p3}-${SLURM_JOB_ID:-manual}.log"
chmod 600 "$P3_LOG_PATH"

case "$P3_MEDIA_ROOT" in "$P3_SCRATCH_ROOT"/*) ;; *) exit 2 ;; esac
case "$P3_CODE_ROOT" in "$P3_SCRATCH_ROOT"/runs/pilot_p3/*) ;; *) exit 2 ;; esac
case "$P3_LEDGER" in "$P3_DURABLE_ROOT"/*) ;; *) exit 2 ;; esac
test "$(stat -c '%a' "$P3_LEDGER")" = 600

cache_root="$P3_SCRATCH_ROOT/caches"
source_root="$cache_root/p2_source/egobabyvlm"
pixi="$cache_root/p2_tools/pixi-0.70.0/pixi"
preprocess_root="$cache_root/p2_preprocessing_only_not_learned_initialization"
vtc_root="$preprocess_root/VTC"
vtc_python="$preprocess_root/vtc_runtime/.venv/bin/python"

test "$(git -C "$source_root" rev-parse HEAD)" = 224621caf0628270b6115845ac75a65b984234a3
test "$(git -C "$vtc_root" rev-parse HEAD)" = 9308411fb47f4290dfcd84338fc84bfb7f91227f
test "$(sha256sum "$vtc_root/uv.lock" | cut -d' ' -f1)" = cf1525880c3a2e3dd502409ebd62ec6c4269d0d2da9c9e4b446d764ea09baa4e
test "$(sha256sum "$preprocess_root/vtc_2_2/model/config.toml" | cut -d' ' -f1)" = e2110e032aa6ce6ebfb8c86c39aa7e3c013ad9a77ff67938ad30d9c36ff9b07f
test "$(sha256sum "$preprocess_root/vtc_2_2/model/best.ckpt" | cut -d' ' -f1)" = 5c3446d037f9c6746cbe19bb77a77973354d3443556f9e2278e83cd4fe7db5a2
test -x "$pixi"; test -x "$vtc_python"

export PIXI_HOME="$cache_root/p2_pixi"
export PIXI_CACHE_DIR="$cache_root/p2_runtime_cache/pixi"
export XDG_CACHE_HOME="$cache_root/p2_runtime_cache/xdg"
export HF_HOME="$cache_root/p2_runtime_cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true
ffmpeg=$("$pixi" run --manifest-path "$source_root/pixi.toml" --environment default -- which ffmpeg)
ffprobe=$("$pixi" run --manifest-path "$source_root/pixi.toml" --environment default -- which ffprobe)
test -x "$ffmpeg"; test -x "$ffprobe"

args=(
  --config "$P3_CODE_ROOT/configs/pilot_p3.json"
  --ledger "$P3_LEDGER"
  --media-root "$P3_MEDIA_ROOT"
  --scratch-root "$P3_SCRATCH_ROOT"
  --durable-root "$P3_DURABLE_ROOT"
  --slot "$P3_SLOT"
  --ffmpeg "$ffmpeg"
  --ffprobe "$ffprobe"
  --vtc-python "$vtc_python"
  --vtc-script "$vtc_root/scripts/infer.py"
  --vtc-runner "$P3_CODE_ROOT/scripts/pilot_p3_vtc_runner.py"
  --vtc-root "$vtc_root"
  --vtc-config "$preprocess_root/vtc_2_2/model/config.toml"
  --vtc-checkpoint "$preprocess_root/vtc_2_2/model/best.ckpt"
  --vtc-thresholds "$vtc_root/thresholds/f1.toml"
  --upstream-root "$source_root"
  --whisper-model "$preprocess_root/whisper_large_v2"
  --alignment-model "WAV2VEC2_ASR_BASE_960H"
  --alignment-dir "$preprocess_root"
  --log "$P3_LOG_PATH"
)
if [[ -n "${P3_STOP_AFTER:-}" ]]; then args+=(--stop-after "$P3_STOP_AFTER"); fi

"$pixi" run --manifest-path "$source_root/pixi.toml" --environment default -- \
  python "$P3_CODE_ROOT/scripts/pilot_p3.py" "${args[@]}"
