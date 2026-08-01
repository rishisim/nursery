#!/bin/bash
#SBATCH --job-name=phase4-public-language-models
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
tmp="/tmp/phase4-language-${SLURM_JOB_ID}"
mkdir -p "$tmp/pydeps"
trap 'find "$tmp" -mindepth 1 -delete 2>/dev/null || true; rmdir "$tmp" 2>/dev/null || true' EXIT
container="${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif"
tar -C "$tmp/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-pydeps.tar"
singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" "$container" \
  python -m pip install --disable-pip-version-check --no-deps --target "$tmp/pydeps" \
  numpy==2.2.6 \
  openai-whisper==20250625 tiktoken==0.11.0 more-itertools==10.7.0 \
  numba==0.61.2 llvmlite==0.44.0 nltk==3.9.1 imageio-ffmpeg==0.6.0
export PYTHONPATH="$tmp/pydeps"
export HF_HOME="${PHASE4_PUBLIC_ROOT}/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export PHASE4_PUBLIC_ROOT
singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" "$container" \
  python - <<'PY'
import os
from pathlib import Path
import whisper
from huggingface_hub import snapshot_download
import nltk
root=Path(os.environ['PHASE4_PUBLIC_ROOT'])/'models'
whisper.load_model('small', device='cpu', download_root=str(root/'whisper'))
snapshot_download(
    repo_id='Helsinki-NLP/opus-mt-de-en',
    revision='1a922f3b32a8e809e17a47d4b32142d8105924e5',
    local_dir=str(root/'opus-mt-de-en'),
)
for package in ('wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng'):
    nltk.download(package, download_dir=str(root/'nltk_data'), quiet=True, raise_on_error=True)
PY
tar -C "$tmp/pydeps" -cf "$tmp/language-pydeps.tar" .
cp "$tmp/language-pydeps.tar" "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" "$container" \
  python - <<'PY'
import hashlib, os
from pathlib import Path
import whisper
import nltk
from transformers import MarianMTModel, MarianTokenizer
root=Path(os.environ['PHASE4_PUBLIC_ROOT'])/'models'
nltk.data.path[:] = [str(root/'nltk_data')]
p=root/'whisper'/'small.pt'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794'
whisper.load_model('small', device='cpu', download_root=str(root/'whisper'))
MarianTokenizer.from_pretrained(root/'opus-mt-de-en', local_files_only=True)
MarianMTModel.from_pretrained(root/'opus-mt-de-en', local_files_only=True)
from nltk.corpus import wordnet
assert wordnet.synsets('dog')
print('LANGUAGE_ENVIRONMENT_READY')
PY
