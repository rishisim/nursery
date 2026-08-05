#!/bin/bash
#SBATCH --job-name=synthetic-calibration-qualify
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# The static A30 directives remain the default for later scientific
# development/holdout. Health attempt 11 uses one prospectively frozen
# command-line topology selected immediately before submission.

set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
candidate_id="${PHASE4_ACTIVITY_CANDIDATE:-legacy-repair}"
activity_run_mode="${PHASE4_ACTIVITY_RUN_MODE:-development}"
tuple_run_mode="${PHASE4_TUPLE_RUN_MODE:-none}"
export HF_HOME="${PHASE4_PUBLIC_ROOT}/models/pe-hf-home"
if [[ "$tuple_run_mode" == "health" || "$tuple_run_mode" == "development" || "$tuple_run_mode" == "holdout" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/mechanistic-tuples/construct-aligned-engineering-health"
elif [[ "$tuple_run_mode" == "sizing" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/mechanistic-tuples"
elif [[ "$candidate_id" == "legacy-repair" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/extractor-repair"
else
  case "$candidate_id" in
    egohod_egovideo_l_zero_shot|videoprism_lvt_l_zero_shot|vjepa2_vitl_public_probe) ;;
    *) exit 64 ;;
  esac
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/extractor-redesign/activity-checkpoint-selection"
fi
install -d -m 700 "$output_root" "$output_root/logs"
exec >"$output_root/logs/qualify-${SLURM_JOB_ID}.out" 2>"$output_root/logs/qualify-${SLURM_JOB_ID}.err"
if [[ "$tuple_run_mode" == "health" ]]; then
  health_attempt="${PHASE4_HEALTH_ATTEMPT:-}"
  case "$health_attempt" in
    11) ;;
    *) exit 64 ;;
  esac
  health_gpu_type="${PHASE4_HEALTH_GPU_TYPE:-}"
  health_partition="${PHASE4_HEALTH_PARTITION:-}"
  health_gres="${PHASE4_HEALTH_GRES:-}"
  health_wall_minutes="${PHASE4_HEALTH_WALL_MINUTES:-}"
  case "${health_gpu_type}|${health_partition}|${health_gres}|${health_wall_minutes}" in
    NVIDIA_A30_24GB\|a30\|gpu:nvidia_a30:1\|60|\
    NVIDIA_H100_NVL\|h100\|gpu:nvidia_h100_nvl:1\|60|\
    NVIDIA_H100_NVL_3G_47GB_MIG\|h100\|gpu:nvidia_h100_nvl_3g.47gb:1\|60|\
    NVIDIA_H200_NVL\|h200\|gpu:nvidia_h200_nvl:1\|60) ;;
    *) exit 64 ;;
  esac
  export PHASE4_HEALTH_SUBMISSION_STARTED_EPOCH="$(date +%s)"
  health_attempt_root="$output_root/health/attempt-$(printf '%02d' "$health_attempt")"
  install -d -m 700 "$health_attempt_root"
  wrapper_marker="$health_attempt_root/wrapper-started.json"
  [[ ! -e "$wrapper_marker" ]] || exit 65
  printf '{"schema_version":1,"attempt":%d,"submission_started_epoch":%s,"GPU_type":"%s","GPU_count":1,"CPU_count":8,"memory_GiB":32}\n' \
    "$health_attempt" "$PHASE4_HEALTH_SUBMISSION_STARTED_EPOCH" "$health_gpu_type" >"$wrapper_marker"
  chmod 600 "$wrapper_marker"
fi
if [[ "$tuple_run_mode" == "health" || "$tuple_run_mode" == "development" || "$tuple_run_mode" == "holdout" ]]; then
  container_image="${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif"
  container_sha256="f274f1ac3726376b762b557ff9a07203b2d42aac3157a7a354b998e589c35792"
  container_bytes=3731320832
  [[ -L "$container_image" ]] || exit 65
  [[ -f "$container_image" ]] || exit 65
  [[ "$(stat -Lc %s "$container_image")" -eq "$container_bytes" ]] || exit 65
  printf '%s  %s\n' "$container_sha256" "$container_image" | sha256sum --check --status
  case "$SLURM_JOB_ID" in ''|*[!0-9]*) exit 66 ;; esac
  if [[ "$tuple_run_mode" == "health" ]]; then
    container_attestation="$health_attempt_root/container-attestation.json"
    attestation_attempt="$health_attempt"
  else
    container_attestation_root="$output_root/container-attestations"
    install -d -m 700 "$container_attestation_root"
    container_attestation="$container_attestation_root/${tuple_run_mode}-${SLURM_JOB_ID}.json"
    attestation_attempt=null
  fi
  [[ ! -e "$container_attestation" ]] || exit 65
  printf '{"artifact_family":"BASE_CONTAINER","attempt":%s,"bytes":3731320832,"host_entry_is_symlink":true,"job_id":%s,"predicate_count":4,"predicate_pass_count":4,"resolved_target_regular_file":true,"run_mode":"%s","schema_version":1,"sha256":"%s","source":"WRAPPER_HOST_BEFORE_CONTAINER"}\n' \
    "$attestation_attempt" "$SLURM_JOB_ID" "$tuple_run_mode" "$container_sha256" >"$container_attestation"
  chmod 600 "$container_attestation"
fi
scratch="${TMPDIR:-/tmp}/synthetic-calibration-qualify-${SLURM_JOB_ID}"
install -d -m 700 "$scratch" "$scratch/pydeps"
export PYTHONPYCACHEPREFIX="$scratch/pycache"
trap 'find "$scratch" -mindepth 1 -delete 2>/dev/null || true; rmdir "$scratch" 2>/dev/null || true' EXIT

require_health_topology() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || exit 66
  [[ "${WORLD_SIZE:-1}" == "1" ]] || exit 66
  [[ "${LOCAL_WORLD_SIZE:-1}" == "1" ]] || exit 66

  local job_record
  job_record="$(scontrol show job --oneliner "$SLURM_JOB_ID" 2>/dev/null)" || exit 66
  case " $job_record " in *" Partition=${health_partition} "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" NumNodes=1 "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" NumCPUs=8 "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" NumTasks=1 "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" TimeLimit=01:00:00 "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" MinMemoryCPU=4G "*) ;; *) exit 66 ;; esac
  local gres_suffix="${health_gres#gpu:}"
  [[ " $job_record " == *" TresPerNode=gres/gpu:${gres_suffix} "* || \
     " $job_record " == *" TresPerNode=gres:gpu:${gres_suffix} "* ]] || exit 66

  case "${SLURM_JOB_ID:-}" in
    ''|*[!0-9]*) exit 66 ;;
  esac
  local topology_attestation="$health_attempt_root/topology-attestation.json"
  [[ ! -e "$topology_attestation" ]] || exit 65
  printf '{"CPU_count":8,"GRES":"%s","attempt":%s,"job_id":%s,"local_world_size":1,"memory_per_CPU_GiB":4,"node_count":1,"partition":"%s","predicate_count":7,"predicate_pass_count":7,"schema_version":1,"source":"WRAPPER_SCONTROL_BEFORE_CONTAINER","task_count":1,"time_limit_minutes":60,"world_size":1}\n' \
    "$health_gres" "$health_attempt" "$SLURM_JOB_ID" "$health_partition" >"$topology_attestation"
  chmod 600 "$topology_attestation"

}

if [[ "$tuple_run_mode" == "sizing" ]]; then
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source:${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/code/egobabyvlm:${PHASE4_PUBLIC_ROOT}/models/activity-pydeps/egohod_egovideo_l_zero_shot"
  singularity exec --nv --net --network none \
    --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-size \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --scratch-root "$scratch" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
    --device cuda
elif [[ "$tuple_run_mode" == "health" || "$tuple_run_mode" == "development" || "$tuple_run_mode" == "holdout" ]]; then
  language_archive="${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
  language_archive_sha256="97ef52ecaa8c99db017e598d8a63d0d2170affef14ef46e7df7a656abd3a1a07"
  language_archive_bytes=542423040
  if [[ ! -f "$language_archive" || "$(stat -c %s "$language_archive")" -ne "$language_archive_bytes" ]]; then
    exit 65
  fi
  printf '%s  %s\n' "$language_archive_sha256" "$language_archive" | sha256sum --check --status
  tar -C "$scratch/pydeps" -xf "$language_archive"
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source:${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/code/egobabyvlm:${PHASE4_PUBLIC_ROOT}/models/activity-pydeps/egohod_egovideo_l_zero_shot:$scratch/pydeps"
  if [[ "$tuple_run_mode" == "health" ]]; then
    require_health_topology
    singularity exec --nv --net --network none \
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
      "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
      python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-health \
      --public-root "${PHASE4_PUBLIC_ROOT}" \
      --scratch-root "$scratch" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --container-attestation "$container_attestation" \
      --attempt "$health_attempt" \
      --device cuda
  else
    singularity exec --nv --net --network none \
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
      "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
      python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-qualify \
      --public-root "${PHASE4_PUBLIC_ROOT}" \
      --scratch-root "$scratch" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --container-attestation "$container_attestation" \
      --partition "$tuple_run_mode" \
      --device cuda
  fi
elif [[ "$candidate_id" == "legacy-repair" ]]; then
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
