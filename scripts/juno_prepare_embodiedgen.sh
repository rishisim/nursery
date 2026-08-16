#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_nvl_3g.47gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=hoi-eg-install

set -euo pipefail

SCRATCH_ROOT=/scratch/juno/dal503972/hoidini_intermimic
SOURCE_ROOT="$SCRATCH_ROOT/source/EmbodiedGen"
ENV_PREFIX="$SCRATCH_ROOT/environments/embodiedgen"
PATCH_PATH="$SCRATCH_ROOT/jobs/embodiedgen_v2.0.1_robot_free.patch"
RESPONSES_PATCH_PATH="$SCRATCH_ROOT/jobs/embodiedgen_v2.0.1_responses_api.patch"
EXPECTED_COMMIT=9b333554254af196bace88c1a171a3bf047fa09c
OPENAI_SDK_VERSION=3.1.0

export HF_HOME="$SCRATCH_ROOT/caches/huggingface"
export MODELSCOPE_CACHE="$SCRATCH_ROOT/caches/modelscope"
export TORCH_HOME="$SCRATCH_ROOT/caches/torch"
export PIP_CACHE_DIR="$SCRATCH_ROOT/caches/pip"
export CONDA_PKGS_DIRS="$SCRATCH_ROOT/caches/conda-pkgs"
export PYTHONNOUSERSITE=1
export PIP_EXTRA_INDEX_URL=https://pypi.org/simple
export TORCH_CUDA_ARCH_LIST=9.0

module load cuda/12.6
module load miniconda/24.11.1
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$SCRATCH_ROOT/source" "$SCRATCH_ROOT/environments" \
  "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR"
mkdir -p "$CONDA_PKGS_DIRS"

if test ! -d "$SOURCE_ROOT/.git"; then
  git clone https://github.com/HorizonRobotics/EmbodiedGen.git "$SOURCE_ROOT"
fi

git -C "$SOURCE_ROOT" fetch --tags --force origin v2.0.1
git -C "$SOURCE_ROOT" checkout --detach v2.0.1
actual_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
test "$actual_commit" = "$EXPECTED_COMMIT"

for patch_path in "$PATCH_PATH" "$RESPONSES_PATCH_PATH"; do
  if ! git -C "$SOURCE_ROOT" apply --unidiff-zero --reverse --check "$patch_path" >/dev/null 2>&1; then
    git -C "$SOURCE_ROOT" apply --unidiff-zero --check "$patch_path"
    git -C "$SOURCE_ROOT" apply --unidiff-zero "$patch_path"
  fi
done

if test ! -s "$ENV_PREFIX/conda-meta/history"; then
  test ! -e "$ENV_PREFIX" || {
    echo "incomplete_environment_prefix=$ENV_PREFIX" >&2
    exit 2
  }
  conda create --prefix "$ENV_PREFIX" python=3.10.13 -y
fi
conda activate "$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
test "$(command -v python)" = "$ENV_PREFIX/bin/python"
test "$(python -c 'import platform; print(platform.python_version())')" = "3.10.13"
conda install --prefix "$ENV_PREFIX" -c conda-forge gcc_linux-64=13 gxx_linux-64=13 -y
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
test -x "$CC"
test -x "$CXX"

cd "$SOURCE_ROOT"
bash install.sh basic
python -m pip install --upgrade "openai==$OPENAI_SDK_VERSION"

python - <<'PY'
import openai
from openai import OpenAI

assert openai.__version__ == "3.1.0"
assert hasattr(OpenAI(api_key="not-a-real-key"), "responses")
print("openai_responses_sdk=passed")
PY

python - <<'PY'
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
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    payload.update({
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "tiny_cuda_sum": float((x * x).sum().item()),
    })
print("install_receipt=" + json.dumps(payload, sort_keys=True))
PY
python -c 'import embodied_gen, sapien, torch, trimesh; print("import_smoke=ok")'
command -v layout-cli
git diff --check
git status --short
echo "environment_preparation_complete=yes"
