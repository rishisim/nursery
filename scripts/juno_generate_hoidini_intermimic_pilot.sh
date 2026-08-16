#!/usr/bin/env bash
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=hoi-eg-pilot

set -euo pipefail

DURABLE_ROOT=/work/dal503972/hoidini_intermimic
SCRATCH_ROOT=/scratch/juno/dal503972/hoidini_intermimic
SOURCE_ROOT="$SCRATCH_ROOT/source/EmbodiedGen"
ENV_PREFIX="$SCRATCH_ROOT/environments/embodiedgen"
GPT_CONFIG="$DURABLE_ROOT/frozen_configs/gpt_config.yaml"
BACKGROUND_ROOT="$DURABLE_ROOT/scene_packages/backgrounds"
BACKGROUND_LIST="$BACKGROUND_ROOT/example_gen_scenes/scene_part_list.txt"
OUTPUT_ROOT="$DURABLE_ROOT/scene_packages/pilot"
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"

test -s "$GPT_CONFIG" || { echo "blocked=gpt_vlm_config_missing"; exit 20; }
test -s "$HF_TOKEN_FILE" || { echo "blocked=huggingface_token_missing"; exit 21; }
hf_token=$(tr -d '\r\n' < "$HF_TOKEN_FILE")
sd_status=$(curl -L -sS -o /dev/null -w '%{http_code}' --max-time 30 \
  -H "Authorization: Bearer $hf_token" \
  https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/resolve/main/model_index.json)
unset hf_token
test "$sd_status" = 200 || { echo "blocked=sd35_medium_http_$sd_status"; exit 22; }

module load cuda/12.6
module load miniconda/24.11.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME="$SCRATCH_ROOT/caches/huggingface"
export MODELSCOPE_CACHE="$SCRATCH_ROOT/caches/modelscope"
export TORCH_HOME="$SCRATCH_ROOT/caches/torch"
export TORCH_CUDA_ARCH_LIST=9.0

python - "$GPT_CONFIG" <<'PY'
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
agent_type = config.get("agent_type")
agent = config.get(agent_type)
if not isinstance(agent, dict) or not agent.get("api_key"):
    raise SystemExit("blocked=usable_private_gpt_config_missing")
if agent.get("model_name") != "gpt-5.6-luna":
    raise SystemExit("blocked=openai_model_must_be_gpt-5.6-luna")
print("openai_model=gpt-5.6-luna")
PY

python - <<'PY'
import openai
from openai import OpenAI

assert openai.__version__ == "3.1.0"
assert hasattr(OpenAI(api_key="not-a-real-key"), "responses")
print("openai_responses_sdk=passed")
PY
grep -q 'self.client.responses.create' "$SOURCE_ROOT/embodied_gen/utils/gpt_clients.py"

python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (9, 0)
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
assert float((x * x).sum().item()) == 14.0
print("installed_environment_cuda_smoke=passed")
PY

install -m 600 "$GPT_CONFIG" "$SOURCE_ROOT/embodied_gen/utils/gpt_config.yaml"

if test ! -s "$BACKGROUND_LIST"; then
  mkdir -p "$BACKGROUND_ROOT"
  hf download HorizonRobotics/EmbodiedGenData \
    --repo-type dataset \
    --local-dir "$BACKGROUND_ROOT" \
    --include 'example_gen_scenes/scene_00[01][0-9]/**' \
      'example_gen_scenes/scene_part_list.txt'
fi
test -s "$BACKGROUND_LIST"

if test -e "$OUTPUT_ROOT"; then
  test -d "$OUTPUT_ROOT" && test -z "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" || {
    echo "blocked=pilot_output_already_nonempty"
    exit 23
  }
else
  mkdir -p "$OUTPUT_ROOT"
fi

cd "$SOURCE_ROOT"
CUDA_VISIBLE_DEVICES=0 layout-cli \
  --task_descs 'Pick up the mug from the table, carry it to the tray, and put it down.' \
  --bg_list "$BACKGROUND_LIST" \
  --output_root "$OUTPUT_ROOT" \
  --seed_img 101 \
  --seed_3d 101 \
  --seed_layout 102 \
  --n_image_retry 4 \
  --n_asset_retry 3 \
  --n_pipe_retry 2

echo "real_generation_complete=yes"
