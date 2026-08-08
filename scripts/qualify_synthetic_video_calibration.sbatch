#!/bin/bash
#SBATCH --job-name=synthetic-calibration-qualify
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# The static A30 directives remain the default for later scientific
# development/holdout. Health attempt 20 uses one prospectively frozen
# command-line topology selected immediately before submission.

set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
execution_backend="${PHASE4_EXECUTION_BACKEND:-slurm}"
if [[ "$execution_backend" == "hf_jobs_public_only" || "$execution_backend" == "vastai_public_only" ]]; then
  execution_id="${PHASE4_PUBLIC_EXECUTION_ID:-}"
  [[ "$execution_id" =~ ^readiness-(health|development|holdout)-[a-z0-9-]+$ ]] || exit 66
  public_input_root="${PHASE4_PUBLIC_INPUT_ROOT:-}"
  public_output_root="${PHASE4_PUBLIC_OUTPUT_ROOT:-}"
  [[ "$public_input_root" == /* && "$public_output_root" == /* ]] || exit 65
  [[ -d "$public_input_root" && -d "$public_output_root" && -w "$public_output_root" ]] || exit 65
  public_input_root="$(readlink -f "$public_input_root")"
  public_output_root="$(readlink -f "$public_output_root")"
  [[ "$public_input_root" != "$public_output_root" ]] || exit 66
  [[ "$public_input_root" != "$public_output_root/"* ]] || exit 66
  [[ "$public_output_root" != "$public_input_root/"* ]] || exit 66
  [[ -d "$public_input_root/models" && -d "$public_input_root/public" && -d "$public_input_root/source" ]] || exit 65
  scratch="${TMPDIR:-/tmp}/synthetic-calibration-qualify-${execution_id}"
  merged_public_root="$scratch/public-root"
  [[ ! -e "$merged_public_root" ]] || exit 65
  install -d -m 700 "$scratch" "$merged_public_root"
  ln -s "$public_input_root/models" "$merged_public_root/models"
  ln -s "$public_input_root/public" "$merged_public_root/public"
  ln -s "$public_input_root/source" "$merged_public_root/source"
  ln -s "$public_output_root" "$merged_public_root/runs"
  export PHASE4_PUBLIC_ROOT="$merged_public_root"
fi
candidate_id="${PHASE4_ACTIVITY_CANDIDATE:-legacy-repair}"
activity_run_mode="${PHASE4_ACTIVITY_RUN_MODE:-development}"
tuple_run_mode="${PHASE4_TUPLE_RUN_MODE:-none}"
readiness_run_mode="${PHASE4_READINESS_RUN_MODE:-none}"
case "$readiness_run_mode" in none|health|development|holdout) ;; *) exit 64 ;; esac
if [[ "$readiness_run_mode" != "none" && "$tuple_run_mode" != "none" ]]; then
  exit 64
fi
export HF_HOME="${PHASE4_PUBLIC_ROOT}/models/pe-hf-home"
if [[ "$readiness_run_mode" != "none" ]]; then
  output_root="${PHASE4_PUBLIC_ROOT}/runs/synthetic-video-calibration/public-only-readiness"
elif [[ "$tuple_run_mode" == "health" || "$tuple_run_mode" == "development" || "$tuple_run_mode" == "holdout" ]]; then
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

# Rented public-only backends run this same canonical wrapper inside the
# immutable OCI source image used to build the pinned SIF.
if [[ "$execution_backend" == "hf_jobs_public_only" || "$execution_backend" == "vastai_public_only" ]]; then
  [[ "$readiness_run_mode" != "none" ]] || exit 64
  [[ "${WORLD_SIZE:-1}" == "1" && "${LOCAL_WORLD_SIZE:-1}" == "1" ]] || exit 66
  [[ "${PHASE4_PUBLIC_ROOT:-}" == /* ]] || exit 65
  if [[ "$readiness_run_mode" == "health" ]]; then
    [[ "${PHASE4_PUBLIC_WALL_MINUTES:-}" == "5" ]] || exit 66
  else
    [[ "${PHASE4_PUBLIC_WALL_MINUTES:-}" == "15" ]] || exit 66
  fi
  [[ "${PHASE4_RUNTIME_IMAGE_DIGEST:-}" == "sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69" ]] || exit 66
  [[ "$(nproc)" -ge 8 ]] || exit 66
  memory_kib="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
  [[ "$memory_kib" -ge 33554432 ]] || exit 66
  if [[ "$execution_backend" == "vastai_public_only" ]]; then
    # Symlink mode bits are not mutable access controls on Linux; validate the
    # actual files and directories while allowing links within the sealed tree.
    [[ -z "$(find "$public_input_root" ! -type l -perm /222 -print -quit)" ]] || exit 66
    mapfile -t provider_gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    [[ "${#provider_gpu_names[@]}" -eq 1 && "${provider_gpu_names[0]}" == NVIDIA\ L4* ]] || exit 66
    provider_name="VAST_AI"
    provider_flavor="on_demand_verified"
    provider_memory_gib=32
  else
    provider_name="HUGGING_FACE_JOBS"
    provider_flavor="l40sx1"
    provider_memory_gib=62
  fi

  install -d -m 700 "$output_root" "$output_root/logs"
  exec >"$output_root/logs/${execution_id}.out" 2>"$output_root/logs/${execution_id}.err"
  install -d -m 700 "$scratch" "$scratch/pydeps" "$scratch/network-guard" "$scratch/bin"
  trap 'find "$scratch" -mindepth 1 -delete 2>/dev/null || true; rmdir "$scratch" 2>/dev/null || true' EXIT
  export PYTHONPYCACHEPREFIX="$scratch/pycache"

  container_image="${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif"
  [[ -f "$container_image" && "$(stat -c %s "$container_image")" -eq 3731320832 ]] || exit 65
  printf '%s  %s\n' f274f1ac3726376b762b557ff9a07203b2d42aac3157a7a354b998e589c35792 "$container_image" | sha256sum --check --status
  fixture_target="${PHASE4_PUBLIC_ROOT}/public/mechanistic-training-tuple-fixtures"
  source_record="$fixture_target/visor-hos-source-feasibility.json"
  no_hand_seal="$fixture_target/no-hand-review/verified-no-hand-seal.json"
  fixture_manifest="$fixture_target/fixture-manifest.json"
  [[ -f "$source_record" && "$(stat -c %s "$source_record")" -eq 838802 ]] || exit 65
  [[ -f "$no_hand_seal" && "$(stat -c %s "$no_hand_seal")" -eq 19579 ]] || exit 65
  [[ -f "$fixture_manifest" && "$(stat -c %s "$fixture_manifest")" -eq 1426457 ]] || exit 65
  printf '%s  %s\n' 64176cf282f1038174b12277560cb5f7aa8d61aa027599ad5c44b53e066f9f9e "$source_record" | sha256sum --check --status
  printf '%s  %s\n' 362ab55a026110e7035fb1b1e24bc959023a353849e8f4c246f66b59e041743e "$no_hand_seal" | sha256sum --check --status
  printf '%s  %s\n' 740eebfce87cc8cf543fefcb8015a3ec75b934acc9a39c8d81adb5a7dd331371 "$fixture_manifest" | sha256sum --check --status

  language_archive="${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
  [[ -f "$language_archive" && "$(stat -c %s "$language_archive")" -eq 542423040 ]] || exit 65
  printf '%s  %s\n' 97ef52ecaa8c99db017e598d8a63d0d2170affef14ef46e7df7a656abd3a1a07 "$language_archive" | sha256sum --check --status
  tar -C "$scratch/pydeps" -xf "$language_archive"
  printf '%s\n' \
    'import socket' \
    '_original_socket = socket.socket' \
    'class _OfflineSocket(_original_socket):' \
    '    def connect(self, address):' \
    '        if self.family in {socket.AF_INET, socket.AF_INET6}:' \
    '            raise OSError("E_PUBLIC_READINESS_EXTERNAL_NETWORK_DISABLED")' \
    '        return super().connect(address)' \
    '    def connect_ex(self, address):' \
    '        if self.family in {socket.AF_INET, socket.AF_INET6}:' \
    '            return 101' \
    '        return super().connect_ex(address)' \
    'socket.socket = _OfflineSocket' \
    'def _blocked(*_args, **_kwargs):' \
    '    raise OSError("E_PUBLIC_READINESS_EXTERNAL_NETWORK_DISABLED")' \
    'socket.create_connection = _blocked' \
    >"$scratch/network-guard/sitecustomize.py"
  export PYTHONPATH="$scratch/network-guard:${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source:${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/code/egobabyvlm:${PHASE4_PUBLIC_ROOT}/models/activity-pydeps/egohod_egovideo_l_zero_shot:$scratch/pydeps"
  ffmpeg_exe="$(env PYTHONPATH="$scratch/pydeps" python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  [[ -x "$ffmpeg_exe" && "$(stat -c %s "$ffmpeg_exe")" -eq 79826272 ]] || exit 65
  printf '%s  %s\n' e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99 "$ffmpeg_exe" | sha256sum --check --status
  ln -s "$ffmpeg_exe" "$scratch/bin/ffmpeg"
  export PATH="$scratch/bin:$PATH"

  attestation_root="$output_root/attestations"
  install -d -m 700 "$attestation_root"
  container_attestation="$attestation_root/${execution_id}-container.json"
  fixture_bind_attestation="$attestation_root/${execution_id}-fixture.json"
  topology_attestation="$attestation_root/${execution_id}-topology.json"
  [[ ! -e "$container_attestation" && ! -e "$fixture_bind_attestation" && ! -e "$topology_attestation" ]] || exit 65
  printf '{"artifact_family":"BASE_CONTAINER_OCI_WITH_PRESERVED_SIF_PROVENANCE","execution_id":"%s","legacy_SIF_bytes":3731320832,"legacy_SIF_sha256":"f274f1ac3726376b762b557ff9a07203b2d42aac3157a7a354b998e589c35792","predicate_count":4,"predicate_pass_count":4,"run_mode":"readiness-%s","runtime_image_manifest_sha256":"dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69","schema_version":1,"source":"CANONICAL_WRAPPER_OCI_RUNTIME"}\n' "$execution_id" "$readiness_run_mode" >"$container_attestation"
  printf '{"active_fixture_root_verified":true,"artifact_family":"SEALED_PUBLIC_FIXTURE_READ_ONLY_VOLUME","execution_id":"%s","fixture_manifest_bytes":1426457,"fixture_manifest_file_sha256":"740eebfce87cc8cf543fefcb8015a3ec75b934acc9a39c8d81adb5a7dd331371","path_field_count":0,"predicate_count":9,"predicate_pass_count":9,"read_only":true,"run_mode":"readiness-%s","schema_version":1,"source":"CANONICAL_WRAPPER_READ_ONLY_PUBLIC_VOLUME","source_record_bytes":838802,"source_record_file_sha256":"64176cf282f1038174b12277560cb5f7aa8d61aa027599ad5c44b53e066f9f9e","verified_no_hand_seal_bytes":19579,"verified_no_hand_seal_file_sha256":"362ab55a026110e7035fb1b1e24bc959023a353849e8f4c246f66b59e041743e"}\n' "$execution_id" "$readiness_run_mode" >"$fixture_bind_attestation"
  printf '{"CPU_count":8,"GPU_count":1,"execution_id":"%s","local_world_size":1,"memory_GiB":%s,"node_count":1,"provider":"%s","provider_flavor":"%s","route_id":"learner_effective_public_only_readiness","run_mode":"%s","schema_version":1,"source":"CANONICAL_WRAPPER_PROVIDER_ATTESTATION","task_count":1,"time_limit_minutes":%s,"world_size":1}\n' "$execution_id" "$provider_memory_gib" "$provider_name" "$provider_flavor" "$readiness_run_mode" "$PHASE4_PUBLIC_WALL_MINUTES" >"$topology_attestation"
  chmod 600 "$container_attestation" "$fixture_bind_attestation" "$topology_attestation"

  python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" readiness-prepare \
    --public-root "${PHASE4_PUBLIC_ROOT}" \
    --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
  if [[ "$readiness_run_mode" == "health" ]]; then
    [[ "${PHASE4_READINESS_ATTEMPT:-}" == "2" ]] || exit 64
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" readiness-health \
      --public-root "${PHASE4_PUBLIC_ROOT}" --scratch-root "$scratch" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --container-attestation "$container_attestation" \
      --fixture-bind-attestation "$fixture_bind_attestation" \
      --topology-attestation "$topology_attestation" --attempt 2 --device cuda
  else
    python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" readiness-qualify \
      --public-root "${PHASE4_PUBLIC_ROOT}" --scratch-root "$scratch" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --container-attestation "$container_attestation" \
      --fixture-bind-attestation "$fixture_bind_attestation" \
      --topology-attestation "$topology_attestation" \
      --partition "$readiness_run_mode" --device cuda
  fi
  exit 0
fi

install -d -m 700 "$output_root" "$output_root/logs"
exec >"$output_root/logs/qualify-${SLURM_JOB_ID}.out" 2>"$output_root/logs/qualify-${SLURM_JOB_ID}.err"
if [[ "$tuple_run_mode" == "health" ]]; then
  health_attempt="${PHASE4_HEALTH_ATTEMPT:-}"
  case "$health_attempt" in
    20) ;;
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
fixture_source_root=""
fixture_target=""
fixture_binds=""
if [[ "$readiness_run_mode" != "none" || "$tuple_run_mode" == "health" || "$tuple_run_mode" == "development" || "$tuple_run_mode" == "holdout" ]]; then
  fixture_source_root="${PHASE4_FIXTURE_SOURCE_ROOT:-}"
  fixture_target="${PHASE4_PUBLIC_ROOT}/public/mechanistic-training-tuple-fixtures"
  [[ "$fixture_source_root" == /* ]] || exit 65
  [[ -d "$fixture_source_root" ]] || exit 65
  [[ -d "$fixture_target" ]] || exit 65
  [[ "$(readlink -f "$fixture_source_root")" != "$(readlink -f "$fixture_target")" ]] || exit 65
  source_record="$fixture_source_root/visor-hos-source-feasibility.json"
  no_hand_seal="$fixture_source_root/no-hand-review/verified-no-hand-seal.json"
  fixture_manifest="$fixture_source_root/fixture-manifest.json"
  [[ -f "$source_record" && "$(stat -c %s "$source_record")" -eq 838802 ]] || exit 65
  [[ -f "$no_hand_seal" && "$(stat -c %s "$no_hand_seal")" -eq 19579 ]] || exit 65
  [[ -f "$fixture_manifest" && "$(stat -c %s "$fixture_manifest")" -eq 1426457 ]] || exit 65
  printf '%s  %s\n' 64176cf282f1038174b12277560cb5f7aa8d61aa027599ad5c44b53e066f9f9e "$source_record" | sha256sum --check --status
  printf '%s  %s\n' 362ab55a026110e7035fb1b1e24bc959023a353849e8f4c246f66b59e041743e "$no_hand_seal" | sha256sum --check --status
  printf '%s  %s\n' 740eebfce87cc8cf543fefcb8015a3ec75b934acc9a39c8d81adb5a7dd331371 "$fixture_manifest" | sha256sum --check --status
  fixture_binds="${fixture_source_root}:${fixture_source_root}:ro,${fixture_source_root}:${fixture_target}:ro"
fi
if [[ "$readiness_run_mode" != "none" || "$tuple_run_mode" == "health" || "$tuple_run_mode" == "development" || "$tuple_run_mode" == "holdout" ]]; then
  container_image="${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif"
  container_sha256="f274f1ac3726376b762b557ff9a07203b2d42aac3157a7a354b998e589c35792"
  container_bytes=3731320832
  [[ -L "$container_image" ]] || exit 65
  [[ -f "$container_image" ]] || exit 65
  [[ "$(stat -Lc %s "$container_image")" -eq "$container_bytes" ]] || exit 65
  printf '%s  %s\n' "$container_sha256" "$container_image" | sha256sum --check --status
  case "$SLURM_JOB_ID" in ''|*[!0-9]*) exit 66 ;; esac
  if [[ "$readiness_run_mode" != "none" ]]; then
    container_attestation_root="$output_root/attestations"
    install -d -m 700 "$container_attestation_root"
    container_attestation="$container_attestation_root/${readiness_run_mode}-${SLURM_JOB_ID}-container.json"
    [[ ! -e "$container_attestation" ]] || exit 65
    printf '{"artifact_family":"BASE_CONTAINER","bytes":3731320832,"host_entry_is_symlink":true,"job_id":%s,"predicate_count":4,"predicate_pass_count":4,"resolved_target_regular_file":true,"run_mode":"readiness-%s","schema_version":1,"sha256":"%s","source":"WRAPPER_HOST_BEFORE_CONTAINER"}\n' \
      "$SLURM_JOB_ID" "$readiness_run_mode" "$container_sha256" >"$container_attestation"
    chmod 600 "$container_attestation"
  elif [[ "$tuple_run_mode" == "health" ]]; then
    container_attestation="$health_attempt_root/container-attestation.json"
    attestation_attempt="$health_attempt"
  else
    container_attestation_root="$output_root/container-attestations"
    install -d -m 700 "$container_attestation_root"
    container_attestation="$container_attestation_root/${tuple_run_mode}-${SLURM_JOB_ID}.json"
    attestation_attempt=null
  fi
  if [[ "$readiness_run_mode" == "none" ]]; then
    [[ ! -e "$container_attestation" ]] || exit 65
    printf '{"artifact_family":"BASE_CONTAINER","attempt":%s,"bytes":3731320832,"host_entry_is_symlink":true,"job_id":%s,"predicate_count":4,"predicate_pass_count":4,"resolved_target_regular_file":true,"run_mode":"%s","schema_version":1,"sha256":"%s","source":"WRAPPER_HOST_BEFORE_CONTAINER"}\n' \
      "$attestation_attempt" "$SLURM_JOB_ID" "$tuple_run_mode" "$container_sha256" >"$container_attestation"
    chmod 600 "$container_attestation"
  fi
  if [[ "$readiness_run_mode" != "none" ]]; then
    fixture_bind_attestation="$container_attestation_root/${readiness_run_mode}-${SLURM_JOB_ID}-fixture-bind.json"
    [[ ! -e "$fixture_bind_attestation" ]] || exit 65
    printf '{"artifact_family":"SEALED_PUBLIC_FIXTURE_BIND","fixture_manifest_bytes":1426457,"fixture_manifest_file_sha256":"740eebfce87cc8cf543fefcb8015a3ec75b934acc9a39c8d81adb5a7dd331371","job_id":%s,"path_field_count":0,"predicate_count":9,"predicate_pass_count":9,"read_only":true,"run_mode":"readiness-%s","schema_version":1,"source":"WRAPPER_HOST_BEFORE_CONTAINER","source_bound_at_original_absolute_alias":true,"source_bound_over_active_fixture_target":true,"source_record_bytes":838802,"source_record_file_sha256":"64176cf282f1038174b12277560cb5f7aa8d61aa027599ad5c44b53e066f9f9e","verified_no_hand_seal_bytes":19579,"verified_no_hand_seal_file_sha256":"362ab55a026110e7035fb1b1e24bc959023a353849e8f4c246f66b59e041743e"}\n' \
      "$SLURM_JOB_ID" "$readiness_run_mode" >"$fixture_bind_attestation"
    chmod 600 "$fixture_bind_attestation"
  elif [[ "$tuple_run_mode" == "health" ]]; then
    fixture_bind_attestation="$health_attempt_root/fixture-bind-attestation.json"
  else
    fixture_bind_attestation="$container_attestation_root/${tuple_run_mode}-${SLURM_JOB_ID}-fixture-bind.json"
  fi
  if [[ "$readiness_run_mode" == "none" ]]; then
    [[ ! -e "$fixture_bind_attestation" ]] || exit 65
    printf '{"artifact_family":"SEALED_PUBLIC_FIXTURE_BIND","attempt":%s,"fixture_manifest_bytes":1426457,"fixture_manifest_file_sha256":"740eebfce87cc8cf543fefcb8015a3ec75b934acc9a39c8d81adb5a7dd331371","job_id":%s,"path_field_count":0,"predicate_count":9,"predicate_pass_count":9,"read_only":true,"run_mode":"%s","schema_version":1,"source":"WRAPPER_HOST_BEFORE_CONTAINER","source_bound_at_original_absolute_alias":true,"source_bound_over_active_fixture_target":true,"source_record_bytes":838802,"source_record_file_sha256":"64176cf282f1038174b12277560cb5f7aa8d61aa027599ad5c44b53e066f9f9e","verified_no_hand_seal_bytes":19579,"verified_no_hand_seal_file_sha256":"362ab55a026110e7035fb1b1e24bc959023a353849e8f4c246f66b59e041743e"}\n' \
      "$attestation_attempt" "$SLURM_JOB_ID" "$tuple_run_mode" >"$fixture_bind_attestation"
    chmod 600 "$fixture_bind_attestation"
  fi
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
  local scheduler_gres_ok=false
  if [[ " $job_record " == *" TresPerNode=gres/gpu:${gres_suffix} "* || \
        " $job_record " == *" TresPerNode=gres:gpu:${gres_suffix} "* ]]; then
    scheduler_gres_ok=true
  elif [[ "$health_attempt" == "20" && "$health_gpu_type" == "NVIDIA_A30_24GB" && \
          "$health_partition" == "a30" && "$health_gres" == "gpu:nvidia_a30:1" && \
          " $job_record " == *" TresPerNode=gres/gpu:1 "* ]]; then
    # Slurm may serialize an A30-only partition's one-GPU request generically.
    # The in-container runner still blocks on the exact A30 name and 23--25 GiB.
    scheduler_gres_ok=true
  fi
  [[ "$scheduler_gres_ok" == "true" ]] || exit 66

  case "${SLURM_JOB_ID:-}" in
    ''|*[!0-9]*) exit 66 ;;
  esac
  local topology_attestation="$health_attempt_root/topology-attestation.json"
  [[ ! -e "$topology_attestation" ]] || exit 65
  printf '{"CPU_count":8,"GRES":"%s","attempt":%s,"job_id":%s,"local_world_size":1,"memory_per_CPU_GiB":4,"node_count":1,"partition":"%s","predicate_count":7,"predicate_pass_count":7,"schema_version":1,"source":"WRAPPER_SCONTROL_BEFORE_CONTAINER","task_count":1,"time_limit_minutes":60,"world_size":1}\n' \
    "$health_gres" "$health_attempt" "$SLURM_JOB_ID" "$health_partition" >"$topology_attestation"
  chmod 600 "$topology_attestation"

}

require_readiness_topology() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || exit 66
  [[ "${WORLD_SIZE:-1}" == "1" ]] || exit 66
  [[ "${LOCAL_WORLD_SIZE:-1}" == "1" ]] || exit 66
  local readiness_gpu_type="${PHASE4_READINESS_GPU_TYPE:-}"
  local readiness_partition="${PHASE4_READINESS_PARTITION:-}"
  local readiness_gres="${PHASE4_READINESS_GRES:-}"
  local expected_wall
  if [[ "$readiness_run_mode" == "health" ]]; then expected_wall=5; else expected_wall=15; fi
  case "${readiness_gpu_type}|${readiness_partition}|${readiness_gres}" in
    NVIDIA_A30_24GB\|a30\|gpu:nvidia_a30:1|\
    NVIDIA_H100_NVL\|h100\|gpu:nvidia_h100_nvl:1|\
    NVIDIA_H100_NVL_3G_47GB_MIG\|h100\|gpu:nvidia_h100_nvl_3g.47gb:1|\
    NVIDIA_H200_NVL\|h200\|gpu:nvidia_h200_nvl:1) ;;
    *) exit 64 ;;
  esac
  local job_record
  job_record="$(scontrol show job --oneliner "$SLURM_JOB_ID" 2>/dev/null)" || exit 66
  case " $job_record " in *" Partition=${readiness_partition} "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" NumNodes=1 "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" NumCPUs=8 "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" NumTasks=1 "*) ;; *) exit 66 ;; esac
  local expected_time
  expected_time="$(printf '%02d:%02d:00' "$((expected_wall / 60))" "$((expected_wall % 60))")"
  case " $job_record " in *" TimeLimit=${expected_time} "*) ;; *) exit 66 ;; esac
  case " $job_record " in *" MinMemoryCPU=4G "*) ;; *) exit 66 ;; esac
  local gres_suffix="${readiness_gres#gpu:}"
  local scheduler_gres_ok=false
  if [[ " $job_record " == *" TresPerNode=gres/gpu:${gres_suffix} " || \
        " $job_record " == *" TresPerNode=gres:gpu:${gres_suffix} " ]]; then
    scheduler_gres_ok=true
  elif [[ "$readiness_gpu_type" == "NVIDIA_A30_24GB" && \
          "$readiness_partition" == "a30" && \
          " $job_record " == *" TresPerNode=gres/gpu:1 " ]]; then
    scheduler_gres_ok=true
  fi
  [[ "$scheduler_gres_ok" == "true" ]] || exit 66
  topology_attestation="$output_root/attestations/${readiness_run_mode}-${SLURM_JOB_ID}-topology.json"
  [[ ! -e "$topology_attestation" ]] || exit 65
  printf '{"CPU_count":8,"GPU_count":1,"GRES":"%s","job_id":%s,"local_world_size":1,"memory_GiB":32,"node_count":1,"partition":"%s","route_id":"learner_effective_public_only_readiness","run_mode":"%s","schema_version":1,"source":"WRAPPER_SCONTROL_BEFORE_CONTAINER","task_count":1,"time_limit_minutes":%d,"world_size":1}\n' \
    "$readiness_gres" "$SLURM_JOB_ID" "$readiness_partition" "$readiness_run_mode" "$expected_wall" >"$topology_attestation"
  chmod 600 "$topology_attestation"
}

if [[ "$readiness_run_mode" != "none" ]]; then
  language_archive="${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
  language_archive_sha256="97ef52ecaa8c99db017e598d8a63d0d2170affef14ef46e7df7a656abd3a1a07"
  language_archive_bytes=542423040
  [[ -f "$language_archive" && "$(stat -c %s "$language_archive")" -eq "$language_archive_bytes" ]] || exit 65
  printf '%s  %s\n' "$language_archive_sha256" "$language_archive" | sha256sum --check --status
  tar -C "$scratch/pydeps" -xf "$language_archive"
  export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/runtime-pydeps:${PHASE4_PUBLIC_ROOT}/source:${PHASE4_PUBLIC_ROOT}/models/mechanistic-tuples/code/egobabyvlm:${PHASE4_PUBLIC_ROOT}/models/activity-pydeps/egohod_egovideo_l_zero_shot:$scratch/pydeps"
  install -d -m 700 "$scratch/bin"
  ffmpeg_exe=$(singularity exec --net --network none \
    --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    env PYTHONPATH="$scratch/pydeps" \
    python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
  [[ -x "$ffmpeg_exe" && "$(stat -c %s "$ffmpeg_exe")" -eq 79826272 ]] || exit 65
  printf '%s  %s\n' e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99 "$ffmpeg_exe" | sha256sum --check --status
  ln -s "$ffmpeg_exe" "$scratch/bin/ffmpeg"
  export PATH="$scratch/bin:$PATH"
  export SINGULARITYENV_PREPEND_PATH="$scratch/bin"
  require_readiness_topology
  if [[ "$readiness_run_mode" == "health" ]]; then
    readiness_attempt="${PHASE4_READINESS_ATTEMPT:-}"
    case "$readiness_attempt" in 1|2) ;; *) exit 64 ;; esac
    singularity exec --nv --net --network none \
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${fixture_binds},${scratch}:${scratch}" \
      "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
      python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" readiness-prepare \
      --public-root "${PHASE4_PUBLIC_ROOT}" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json"
    singularity exec --nv --net --network none \
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${fixture_binds},${scratch}:${scratch}" \
      "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
      python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" readiness-health \
      --public-root "${PHASE4_PUBLIC_ROOT}" \
      --scratch-root "$scratch" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --container-attestation "$container_attestation" \
      --fixture-bind-attestation "$fixture_bind_attestation" \
      --topology-attestation "$topology_attestation" \
      --attempt "$readiness_attempt" --device cuda
  else
    singularity exec --nv --net --network none \
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${fixture_binds},${scratch}:${scratch}" \
      "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
      python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" readiness-qualify \
      --public-root "${PHASE4_PUBLIC_ROOT}" \
      --scratch-root "$scratch" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --container-attestation "$container_attestation" \
      --fixture-bind-attestation "$fixture_bind_attestation" \
      --topology-attestation "$topology_attestation" \
      --partition "$readiness_run_mode" --device cuda
  fi
elif [[ "$tuple_run_mode" == "sizing" ]]; then
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
  install -d -m 700 "$scratch/bin"
  ffmpeg_exe=$(singularity exec --net --network none \
    --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${scratch}:${scratch}" \
    "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
    env PYTHONPATH="$scratch/pydeps" \
    python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
  [[ -x "$ffmpeg_exe" && "$(stat -c %s "$ffmpeg_exe")" -eq 79826272 ]] || exit 65
  printf '%s  %s\n' e7e7fb30477f717e6f55f9180a70386c62677ef8a4d4d1a5d948f4098aa3eb99 "$ffmpeg_exe" | sha256sum --check --status
  ln -s "$ffmpeg_exe" "$scratch/bin/ffmpeg"
  export PATH="$scratch/bin:$PATH"
  export SINGULARITYENV_PREPEND_PATH="$scratch/bin"
  if [[ "$tuple_run_mode" == "health" ]]; then
    require_health_topology
    singularity exec --nv --net --network none \
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${fixture_binds},${scratch}:${scratch}" \
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
      --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT},${fixture_binds},${scratch}:${scratch}" \
      "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" \
      python "${PHASE4_PUBLIC_ROOT}/source/run_synthetic_video_calibration.py" tuple-qualify \
      --public-root "${PHASE4_PUBLIC_ROOT}" \
      --scratch-root "$scratch" \
      --config "${PHASE4_PUBLIC_ROOT}/source/synthetic_video_real_only_proof.json" \
      --container-attestation "$container_attestation" \
      --fixture-bind-attestation "$fixture_bind_attestation" \
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
