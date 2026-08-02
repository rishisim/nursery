#!/bin/bash
#SBATCH --job-name=synthetic-calibration-qualify
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
candidate_id="${PHASE4_ACTIVITY_CANDIDATE:-legacy-repair}"
activity_run_mode="${PHASE4_ACTIVITY_RUN_MODE:-development}"
export HF_HOME="${PHASE4_PUBLIC_ROOT}/models/pe-hf-home"
if [[ "$candidate_id" == "legacy-repair" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/extractor-repair"
else
  case "$candidate_id" in
    egohod_egovideo_l_zero_shot|videoprism_lvt_l_zero_shot|vjepa2_vitl_public_probe) ;;
    *) exit 64 ;;
  esac
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/extractor-redesign/activity-checkpoint-selection"
fi
mkdir -p "$output_root/logs"
exec >"$output_root/logs/qualify-${SLURM_JOB_ID}.out" 2>"$output_root/logs/qualify-${SLURM_JOB_ID}.err"
scratch="${TMPDIR:-/tmp}/synthetic-calibration-qualify-${SLURM_JOB_ID}"
mkdir -p "$scratch/pydeps"
trap 'find "$scratch" -mindepth 1 -delete 2>/dev/null || true; rmdir "$scratch" 2>/dev/null || true' EXIT
if [[ "$candidate_id" == "legacy-repair" ]]; then
  tar -C "$scratch/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/calibration-repair-pydeps:${PHASE4_PUBLIC_ROOT}/source:${PHASE4_PUBLIC_ROOT}/models/evaluator-pydeps:$scratch/pydeps:${PHASE4_EGOBABY_ROOT}"
  singularity exec --nv --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${PHASE4_EGOBABY_ROOT}:${PHASE4_EGOBABY_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" qualify-public \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
else
  case "$activity_run_mode" in
    sizing|development) ;;
    *) exit 64 ;;
  esac
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/activity-pydeps/${candidate_id}:${PHASE4_PUBLIC_ROOT}/source:$scratch/pydeps"
  export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
  if [[ "$activity_run_mode" == "sizing" ]]; then
    singularity exec --nv --net --network none \
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
      "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
      python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" activity-size \
      --public-root "${PHASE4_PUBLIC_ROOT}" \
      --manifest "${PHASE4_PUBLIC_ROOT}/public/manifests/charades-selection-manifest.json" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --candidate-id "$candidate_id" --device cuda
  else
    singularity exec --nv --net --network none \
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
      "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
      python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" activity-candidate \
      --public-root "${PHASE4_PUBLIC_ROOT}" \
      --manifest "${PHASE4_PUBLIC_ROOT}/public/manifests/charades-selection-manifest.json" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --candidate-id "$candidate_id" --partition development --device cuda
  fi
fi
