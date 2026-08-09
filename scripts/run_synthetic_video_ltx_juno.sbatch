#!/bin/bash
#SBATCH --job-name=public-ltx-juno
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077

public_root="${SYNTHETIC_VIDEO_PUBLIC_ROOT:?}"
run_root="${SYNTHETIC_VIDEO_LTX_JUNO_RUN_ROOT:?}"
source_root="${public_root}/models/ltx-2.3/source"
model_root="${public_root}/models/ltx-2.3/weights"
container="${public_root}/models/pytorch-2.8.0-cu126.sif"
input_plan="${run_root}/input/episode_plan.json"
runner="${run_root}/input/run_synthetic_video_ltx_juno.py"

[[ -d "$public_root" && -d "$run_root" && -d "$source_root/.git" ]] || exit 65
[[ -f "$container" && -f "$input_plan" && -f "$runner" ]] || exit 65
[[ "$(git -C "$source_root" rev-parse HEAD)" == "9377758131b1ffde4b7f766804590a6617bf2ab9" ]] || exit 66
[[ -f "${model_root}/ltx-2.3-22b-distilled-1.1.safetensors" ]] || exit 65
[[ -f "${model_root}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" ]] || exit 65
[[ -f "${model_root}/gemma-3-12b/config.json" ]] || exit 65
[[ ! -e "${run_root}/native_generated.mp4" ]] || exit 66

install -d -m 700 "${run_root}/logs"
exec >"${run_root}/logs/job.out" 2>"${run_root}/logs/job.err"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 UV_OFFLINE=1
export TMPDIR="${run_root}/tmp" UV_CACHE_DIR="${public_root}/cache/uv-ltx-2.3"
install -d -m 700 "$TMPDIR" "$UV_CACHE_DIR"

singularity exec --nv \
  --bind "${public_root}:${public_root}" \
  "$container" \
  "${source_root}/.venv/bin/python" "$runner" \
    --plan "$input_plan" \
    --checkpoint "${model_root}/ltx-2.3-22b-distilled-1.1.safetensors" \
    --upsampler "${model_root}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    --gemma-root "${model_root}/gemma-3-12b" \
    --output "${run_root}/native_generated.mp4" \
    --record "${run_root}/generation_record.json"
