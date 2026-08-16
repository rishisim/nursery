#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=hoi-stage2-qualify

set -u

DURABLE_ROOT=/work/dal503972/hoidini_intermimic
SCRATCH_ROOT=/scratch/juno/dal503972/hoidini_intermimic
PYTORCH_CONTAINER=/work/dal503972/phase4_public/models/pytorch-2.8.0-cu126.sif

record_command() {
  local key=$1
  shift
  printf '%s=' "$key"
  if command -v "$1" >/dev/null 2>&1; then
    command -v "$1"
  else
    echo missing
  fi
}

echo "record_type=hoidini_intermimic_juno_qualification"
echo "recorded_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "slurm_job_id=${SLURM_JOB_ID:-missing}"
echo "hostname=$(hostname)"
echo "kernel=$(uname -srmo)"
if test -f /etc/os-release; then
  . /etc/os-release
  echo "os_id=${ID:-unknown}"
  echo "os_version=${VERSION_ID:-unknown}"
fi

echo "durable_root_present=$(test -d "$DURABLE_ROOT" && echo yes || echo no)"
echo "scratch_root_present=$(test -d "$SCRATCH_ROOT" && echo yes || echo no)"
df -h "$DURABLE_ROOT" "$SCRATCH_ROOT" 2>/dev/null || true

record_command python3_path python3
record_command module_path module
record_command singularity_path singularity
record_command apptainer_path apptainer
record_command conda_path conda
record_command git_path git
record_command curl_path curl

echo "hf_token_env_present=$(test -n "${HF_TOKEN:-}" && echo yes || echo no)"
echo "huggingface_token_env_present=$(test -n "${HUGGINGFACE_HUB_TOKEN:-}" && echo yes || echo no)"
echo "openai_key_env_present=$(test -n "${OPENAI_API_KEY:-}" && echo yes || echo no)"
echo "hf_token_file_present=$(test -f "$HOME/.cache/huggingface/token" && echo yes || echo no)"
echo "gpt_config_present=$(test -f "$DURABLE_ROOT/frozen_configs/gpt_config.yaml" && echo yes || echo no)"

printf 'huggingface_http_status='
curl -L -sS -o /dev/null -w '%{http_code}\n' --max-time 20 https://huggingface.co/ || echo request_failed

module load cuda/12.6 >/dev/null 2>&1 || true
echo "cuda_home=${CUDA_HOME:-missing}"
record_command nvcc_path nvcc
nvcc --version 2>/dev/null | tail -1 || true
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || true

if test -f "$PYTORCH_CONTAINER" && command -v singularity >/dev/null 2>&1; then
  echo "pytorch_container_present=yes"
  singularity exec --nv "$PYTORCH_CONTAINER" python3 - <<'PY'
import json
import platform
import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    payload.update(
        {
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "tiny_cuda_sum": float((tensor * tensor).sum().item()),
        }
    )
print("pytorch_cuda=" + json.dumps(payload, sort_keys=True))
PY
  echo "pytorch_cuda_exit=$?"
else
  echo "pytorch_container_present=no"
  echo "pytorch_cuda_exit=unavailable"
fi

module load miniconda/24.11.1 >/dev/null 2>&1 || true
record_command conda_after_module_path conda
conda --version 2>/dev/null || true

echo "qualification_job_complete=yes"
