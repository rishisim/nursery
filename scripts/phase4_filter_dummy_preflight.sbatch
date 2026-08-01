#!/bin/bash
#SBATCH --job-name=phase4-filter-dummy
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true
export HF_HOME="${PHASE4_PUBLIC_ROOT}/models/pe-hf-home"
tmp="/tmp/phase4-filter-dummy-${SLURM_JOB_ID}"
mkdir -p "$tmp/pydeps"
trap 'find "$tmp" -mindepth 1 -delete 2>/dev/null || true; rmdir "$tmp" 2>/dev/null || true' EXIT
tar -C "$tmp/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
export PYTHONPATH="${PHASE4_PUBLIC_ROOT}/models/evaluator-pydeps:$tmp/pydeps:${PHASE4_EGOBABY_ROOT}:${PHASE4_PUBLIC_ROOT}/source"
singularity exec --nv \
  --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" python - <<'PY'
import json, os, torch
from PIL import Image
from build_synthetic_video_phase4_assets import load_frozen_pe_engine
from apps.benchmark_creation.utils.vision_scoring import score_image_text_matrix
from pathlib import Path
cfg=json.loads(Path(os.environ["PHASE4_CONFIG"]).read_text())
engine=load_frozen_pe_engine(Path(os.environ["PHASE4_PUBLIC_ROOT"]),cfg)
matrix=score_image_text_matrix(engine,[Image.new("RGB",(64,64),"red"),Image.new("RGB",(64,64),"blue")],["a red square","a blue square"],"cuda")
assert tuple(matrix.shape)==(2,2) and torch.isfinite(matrix).all()
print(json.dumps({"status":"PASS","shape":[2,2],"gpu_count":1},sort_keys=True))
PY
