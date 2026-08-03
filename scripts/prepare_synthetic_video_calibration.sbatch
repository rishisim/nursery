#!/bin/bash
#SBATCH --job-name=synthetic-calibration-prepare
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
prep_mode="${PHASE4_CALIBRATION_PREP_MODE:-legacy-repair}"
if [[ "$prep_mode" == "activity-selection" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/extractor-redesign/activity-checkpoint-selection"
elif [[ "$prep_mode" == "mechanistic-tuple" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/mechanistic-tuples"
elif [[ "$prep_mode" == "mechanistic-tuple-runtime" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/mechanistic-tuples"
elif [[ "$prep_mode" == "mechanistic-tuple-fixtures" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/mechanistic-tuples"
elif [[ "$prep_mode" == "mechanistic-tuple-fixture-feasibility" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/mechanistic-tuples"
elif [[ "$prep_mode" == "mechanistic-tuple-no-hand-review-prepare" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/mechanistic-tuples"
elif [[ "$prep_mode" == "mechanistic-tuple-no-hand-review-seal" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/mechanistic-tuples"
else
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/extractor-repair"
fi
mkdir -p "$output_root/logs"
exec >"$output_root/logs/prepare-${SLURM_JOB_ID}.out" 2>"$output_root/logs/prepare-${SLURM_JOB_ID}.err"
scratch="${TMPDIR:-/tmp}/synthetic-calibration-prepare-${SLURM_JOB_ID}"
mkdir -p "$scratch/pydeps"
trap 'find "$scratch" -mindepth 1 -delete 2>/dev/null || true; rmdir "$scratch" 2>/dev/null || true' EXIT
if [[ "$prep_mode" == "legacy-repair" ]]; then
  tar -C "$scratch/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
fi
export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/source:$scratch/pydeps"
if [[ "$prep_mode" == "activity-selection" ]]; then
  for repository in EgoHOD videoprism vjepa2 CLIP; do
    git -C "${PHASE4_PUBLIC_ROOT}/models/activity-code/${repository}" diff --quiet
    git -C "${PHASE4_PUBLIC_ROOT}/models/activity-code/${repository}" diff --cached --quiet
  done
  export PHASE4_ACTIVITY_CODE_CLEAN=1
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" activity-prepare \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --manifest "${PHASE4_PUBLIC_ROOT}/public/manifests/charades-selection-manifest.json" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
elif [[ "$prep_mode" == "mechanistic-tuple" ]]; then
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-prepare \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
elif [[ "$prep_mode" == "mechanistic-tuple-runtime" ]]; then
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-runtime-prepare \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
elif [[ "$prep_mode" == "mechanistic-tuple-fixtures" ]]; then
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source"
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-fixtures-prepare \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --audio-seed-root "${PHASE4_PUBLIC_ROOT}/public/mechanistic-training-tuple-fixtures/audio-seed" \
    --no-hand-review-root "${PHASE4_PUBLIC_ROOT}/public/mechanistic-training-tuple-fixtures/no-hand-review" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
elif [[ "$prep_mode" == "mechanistic-tuple-fixture-feasibility" ]]; then
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source"
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-fixtures-feasibility \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --recipe active-visor-hos \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
elif [[ "$prep_mode" == "mechanistic-tuple-no-hand-review-prepare" ]]; then
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source"
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-no-hand-review-prepare \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
elif [[ "$prep_mode" == "mechanistic-tuple-no-hand-review-seal" ]]; then
  if [[ "${PHASE4_AUTHORIZED_APPLICANT_ATTESTED:-0}" != "1" ]] || \
     [[ "${PHASE4_BLIND_TO_EGOHOS_OUTPUT_ATTESTED:-0}" != "1" ]] || \
     [[ "${PHASE4_EGOHOS_INFERENCE_NOT_STARTED_ATTESTED:-0}" != "1" ]]; then
    echo '{"status":"BLOCKED_ATTESTATIONS_REQUIRED"}'
    exit 64
  fi
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source"
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-no-hand-review-seal \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
    --authorized-applicant-attested \
    --blind-to-egohos-output-attested \
    --egohos-inference-not-started-attested
else
  singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" prepare-public \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
fi
