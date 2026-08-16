#!/usr/bin/env bash
#SBATCH --job-name=imim-generate
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/juno/dal503972/interactmove_intermimic/logs/generate-%j.out
#SBATCH --error=/scratch/juno/dal503972/interactmove_intermimic/logs/generate-%j.err

set -euo pipefail
umask 077

readonly EMBODIEDGEN_COMMIT=9b333554254af196bace88c1a171a3bf047fa09c
readonly DATASET_COMMIT=58258b50a0fc95034f2f3cc03b332ec6f72b91fd
readonly TRELLIS_SOURCE_COMMIT=55a8e8164b195bbf927e0978f00e76c835e6011f
readonly TRELLIS_FLEXICUBES_COMMIT=f97beb0dd3c6c68f3ab5696b6dcaf9af69f0514e
readonly TRELLIS_REVISION=25e0d31ffbebe4b5a97464dd851910efc3002d96
readonly DINOV2_COMMIT=7764ea0f912e53c92e82eb78a2a1631e92725fc8
readonly ACTIVITY_ID=red-mug-mouth-return
readonly SCRATCH_ROOT=/scratch/juno/dal503972/interactmove_intermimic
readonly DURABLE_ROOT=/work/dal503972/interactmove_intermimic
readonly SOURCE_ROOT="${SCRATCH_ROOT}/source/EmbodiedGen"
readonly NURSERY_ROOT="${SCRATCH_ROOT}/source/nursery"
readonly ENV_ROOT="${SCRATCH_ROOT}/environments/embodiedgen-v2.0.1-py310-cu126"
readonly GPT_CONFIG="${DURABLE_ROOT}/frozen_configs/gpt_config.yaml"
readonly DATASET_ROOT="${SCRATCH_ROOT}/downloads/EmbodiedGenData-${DATASET_COMMIT}"
readonly CATALOG_ROOT="${SCRATCH_ROOT}/downloads/background_catalog"
readonly RUN_ROOT="${SCRATCH_ROOT}/runs/fresh_generation/${SLURM_JOB_ID}"
readonly GENERATED_ROOT="${RUN_ROOT}/scene"
readonly GENERATED_RECEIPT="${GENERATED_ROOT}/generation_receipt.json"
readonly DURABLE_SCENE="${DURABLE_ROOT}/scene_packages/${ACTIVITY_ID}"
readonly COMPACT_RECEIPT="${DURABLE_ROOT}/compact_records/fresh_embodiedgen_${SLURM_JOB_ID}.json"
readonly MANIFEST="${DURABLE_ROOT}/manifests/${ACTIVITY_ID}_${SLURM_JOB_ID}.sha256"
readonly TRELLIS_ROOT="${DURABLE_ROOT}/checkpoints/TRELLIS-image-large-${TRELLIS_REVISION}"
readonly TRELLIS_MANIFEST="${DURABLE_ROOT}/manifests/trellis_${TRELLIS_REVISION}.sha256"
readonly TORCH_HOME="${DURABLE_ROOT}/checkpoints/torch-hub"
readonly DINOV2_MANIFEST="${DURABLE_ROOT}/manifests/dinov2_${DINOV2_COMMIT}.sha256"
readonly SETUP_JOB_ID="${IMIM_SETUP_JOB_ID:?submit with the admitted TRELLIS setup job}"
readonly SETUP_RECEIPT="${DURABLE_ROOT}/compact_records/embodiedgen_setup_${SETUP_JOB_ID}.json"
readonly RESUME_ROOT="${SCRATCH_ROOT}/runs/fresh_generation/328320/scene"
readonly RESUME_LAYOUT="${SCRATCH_ROOT}/runs/layout_recovery/328320/layout_draft.json"
readonly RESUME_SCENE_TREE="${RESUME_ROOT}/scene_tree.jpg"
readonly RESUME_IMAGES="${RESUME_ROOT}/images"
readonly RESUME_RECEIPT="${RESUME_ROOT}/generation_receipt.json"
readonly ASSET_RESUME_ROOT="${SCRATCH_ROOT}/runs/fresh_generation/328381/scene"
readonly ASSET_RESUME_RECEIPT="${DURABLE_ROOT}/compact_records/fresh_embodiedgen_328381.json"
readonly NURSERY_COMMIT="${IMIM_NURSERY_COMMIT:?submit with the exact nursery commit}"

install -d -m 700 \
    "${RUN_ROOT}" \
    "${SCRATCH_ROOT}/caches/huggingface" \
    "${SCRATCH_ROOT}/caches/modelscope" \
    "${SCRATCH_ROOT}/caches/rembg" \
    "${SCRATCH_ROOT}/downloads" \
    "${SCRATCH_ROOT}/logs" \
    "${DURABLE_ROOT}/compact_records" \
    "${DURABLE_ROOT}/manifests" \
    "${DURABLE_ROOT}/scene_packages"

retain_receipt() {
    if [[ -f "${GENERATED_RECEIPT}" ]]; then
        cp "${GENERATED_RECEIPT}" "${COMPACT_RECEIPT}.tmp"
        mv "${COMPACT_RECEIPT}.tmp" "${COMPACT_RECEIPT}"
    fi
}
trap retain_receipt EXIT

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    echo "EmbodiedGen environment is missing: ${ENV_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${GPT_CONFIG}" || -L "${GPT_CONFIG}" ]]; then
    echo "Private GPT config is missing or unsafe" >&2
    exit 1
fi
if [[ "$(stat -c %a "${GPT_CONFIG}")" != 600 ]]; then
    echo "Private GPT config must have mode 600" >&2
    exit 1
fi
if [[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" != "${EMBODIEDGEN_COMMIT}" ]]; then
    echo "EmbodiedGen checkout does not match the frozen commit" >&2
    exit 1
fi
git -C "${SOURCE_ROOT}" diff --quiet
git -C "${SOURCE_ROOT}" diff --cached --quiet
if [[ "$(git -C "${SOURCE_ROOT}/thirdparty/TRELLIS" rev-parse HEAD)" != "${TRELLIS_SOURCE_COMMIT}" ]]; then
    echo "TRELLIS source checkout does not match the frozen commit" >&2
    exit 1
fi
if [[ "$(git -C "${SOURCE_ROOT}/thirdparty/TRELLIS/trellis/representations/mesh/flexicubes" rev-parse HEAD)" != "${TRELLIS_FLEXICUBES_COMMIT}" ]]; then
    echo "TRELLIS FlexiCubes checkout does not match the frozen commit" >&2
    exit 1
fi
if [[ "$(git -C "${NURSERY_ROOT}" rev-parse HEAD)" != "${NURSERY_COMMIT}" ]]; then
    echo "Nursery checkout does not match IMIM_NURSERY_COMMIT" >&2
    exit 1
fi
git -C "${NURSERY_ROOT}" diff --quiet
git -C "${NURSERY_ROOT}" diff --cached --quiet
if [[ -e "${DURABLE_SCENE}" ]]; then
    echo "Refusing to overwrite durable scene: ${DURABLE_SCENE}" >&2
    exit 1
fi
if [[ -e "${COMPACT_RECEIPT}" || -e "${MANIFEST}" ]]; then
    echo "Refusing to overwrite job-bound durable records" >&2
    exit 1
fi
if [[ ! -d "${TRELLIS_ROOT}" || -L "${TRELLIS_ROOT}" || ! -s "${TRELLIS_MANIFEST}" ]]; then
    echo "Pinned TRELLIS checkpoint package or manifest is missing" >&2
    exit 1
fi
(
    cd "${TRELLIS_ROOT}"
    sha256sum --check "${TRELLIS_MANIFEST}"
)
if [[ ! -s "${DINOV2_MANIFEST}" ]]; then
    echo "Pinned DINOv2 manifest is missing" >&2
    exit 1
fi
(
    cd "${TORCH_HOME}"
    sha256sum --check "${DINOV2_MANIFEST}"
)
if [[ ! -f "${SETUP_RECEIPT}" || -L "${SETUP_RECEIPT}" ]]; then
    echo "Admitted TRELLIS setup receipt is missing or unsafe" >&2
    exit 1
fi
"${ENV_ROOT}/bin/python" - "${SETUP_RECEIPT}" <<'PY'
from pathlib import Path
import json
import sys

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("status") != "passed":
    raise SystemExit("TRELLIS setup receipt did not pass")
if receipt.get("trellis", {}).get("checkpoint_revision") != (
    "25e0d31ffbebe4b5a97464dd851910efc3002d96"
):
    raise SystemExit("TRELLIS setup receipt revision mismatch")
if receipt.get("dinov2", {}).get("commit") != (
    "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
):
    raise SystemExit("DINOv2 setup receipt commit mismatch")
PY
for resume_file in "${RESUME_LAYOUT}" "${RESUME_SCENE_TREE}" "${RESUME_RECEIPT}" \
    "${RESUME_IMAGES}/table.png" "${RESUME_IMAGES}/table_raw.png"; do
    if [[ ! -f "${resume_file}" || -L "${resume_file}" ]]; then
        echo "Required resume artifact is missing or unsafe: ${resume_file}" >&2
        exit 1
    fi
done
if [[ ! -d "${ASSET_RESUME_ROOT}" || -L "${ASSET_RESUME_ROOT}" || \
      ! -f "${ASSET_RESUME_RECEIPT}" || -L "${ASSET_RESUME_RECEIPT}" ]]; then
    echo "Hash-bound job-328381 TRELLIS asset resume package is missing or unsafe" >&2
    exit 1
fi
if [[ "$(sha256sum "${ASSET_RESUME_RECEIPT}" | cut -d' ' -f1)" != "0842642b0e10a5ddee297d2188accf3ac3ac95042764551f9c3ce3dd90408d23" ]]; then
    echo "Job-328381 TRELLIS asset receipt hash mismatch" >&2
    exit 1
fi
if [[ "$(sha256sum "${RESUME_LAYOUT}" | cut -d' ' -f1)" != "f3a2125ffce069c251b4db636d5ec1d2da4397ed79414b0ab3b5b8a5caaf7c19" ]] || \
   [[ "$(sha256sum "${RESUME_SCENE_TREE}" | cut -d' ' -f1)" != "9e13a26bb74bbda2d35ee65e37fdba14d1876d0be59a5608df48725256782eb2" ]] || \
   [[ "$(sha256sum "${RESUME_RECEIPT}" | cut -d' ' -f1)" != "f6d352776d94d52c849bfaa62443ccfaa337a3ec9168293d405c79b6b8cc05c3" ]] || \
   [[ "$(sha256sum "${RESUME_IMAGES}/table.png" | cut -d' ' -f1)" != "317b980184bba52c5acb92de0e69d7992ba01dec7746fe5711dca246a80367ef" ]] || \
   [[ "$(sha256sum "${RESUME_IMAGES}/table_raw.png" | cut -d' ' -f1)" != "21291ddb429eb8a31fc367a6e390f53033ffe8efff3a09ad89676ac3c6853cf2" ]]; then
    echo "One or more resume artifacts do not match the frozen job-328320 hashes" >&2
    exit 1
fi
install -d -m 700 "${RUN_ROOT}/microsoft"
ln -s "${TRELLIS_ROOT}" "${RUN_ROOT}/microsoft/TRELLIS-image-large"
if [[ "$(readlink -f "${RUN_ROOT}/microsoft/TRELLIS-image-large")" != "${TRELLIS_ROOT}" ]]; then
    echo "TRELLIS run-local checkpoint binding failed" >&2
    exit 1
fi

readonly BACKGROUND_SOURCE="${DATASET_ROOT}/example_gen_scenes/scene_0010"
if [[ ! -s "${BACKGROUND_SOURCE}/prompt.txt" || ! -s "${BACKGROUND_SOURCE}/gs_model.ply" ]]; then
    echo "Pinned scene_0010 background package is incomplete" >&2
    exit 1
fi
if [[ ! -d "${CATALOG_ROOT}" ]]; then
    catalog_tmp="${CATALOG_ROOT}.tmp-${SLURM_JOB_ID}"
    if [[ -e "${catalog_tmp}" ]]; then
        echo "Refusing stale background catalog staging path" >&2
        exit 1
    fi
    mkdir -m 700 "${catalog_tmp}"
    cp -aL "${BACKGROUND_SOURCE}" "${catalog_tmp}/scene_0010"
    "${ENV_ROOT}/bin/python" - "${catalog_tmp}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
description = (root / "scene_0010" / "prompt.txt").read_text(
    encoding="utf-8"
).strip().replace("\n", " ")
if not description or ":" in description:
    raise SystemExit("background prompt is empty or contains a colon")
(root / "scene_list.txt").write_text(
    f"scene_0010: {description}\n", encoding="utf-8"
)
PY
    mv "${catalog_tmp}" "${CATALOG_ROOT}"
fi

export HF_HOME="${SCRATCH_ROOT}/caches/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME
export MODELSCOPE_CACHE="${SCRATCH_ROOT}/caches/modelscope"
export U2NET_HOME="${SCRATCH_ROOT}/caches/rembg"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export TEXT_MODEL=sd35
export EMBODIEDGEN_SOURCE_ROOT="${SOURCE_ROOT}"
export PYTHONPATH="${NURSERY_ROOT}:${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_TOKEN
HF_TOKEN=$(<"${HOME}/.cache/huggingface/token")

source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
set +u
conda activate "${ENV_ROOT}"
set -u

python - <<'PY'
import torch
assert torch.cuda.is_available()
value = torch.eye(16, device="cuda")
assert bool(torch.equal(value @ value, value))
print("installed_environment_cuda_smoke=passed")
PY

readonly ACTIVITY="${NURSERY_ROOT}/configs/interactmove_intermimic_activity.json"
readonly PROTOCOL="${NURSERY_ROOT}/configs/interactmove_intermimic.json"
readonly REQUEST="${RUN_ROOT}/embodiedgen_request.json"
python -m babyworld_lite.interactmove_intermimic validate-activity \
    "${ACTIVITY}" --print-digest
python -m babyworld_lite.interactmove_intermimic prepare-embodiedgen-request \
    "${ACTIVITY}" \
    --protocol "${PROTOCOL}" \
    --background-list '["scene_0010"]' \
    --output "${REQUEST}"

cd "${RUN_ROOT}"
python "${NURSERY_ROOT}/scripts/run_interactmove_intermimic_embodiedgen_generation.py" \
    --activity "${ACTIVITY}" \
    --request "${REQUEST}" \
    --protocol "${PROTOCOL}" \
    --gpt-config "${GPT_CONFIG}" \
    --background-catalog "${CATALOG_ROOT}/scene_list.txt" \
    --source-root "${SOURCE_ROOT}" \
    --resume-layout "${RESUME_LAYOUT}" \
    --resume-scene-tree "${RESUME_SCENE_TREE}" \
    --resume-images "${RESUME_IMAGES}" \
    --resume-receipt "${RESUME_RECEIPT}" \
    --resume-assets "${ASSET_RESUME_ROOT}" \
    --resume-asset-receipt "${ASSET_RESUME_RECEIPT}" \
    --output-dir "${GENERATED_ROOT}" \
    --receipt "${GENERATED_RECEIPT}"

python - "${GENERATED_RECEIPT}" <<'PY'
from pathlib import Path
import json, sys
receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("status") != "passed":
    raise SystemExit("fresh generation receipt did not pass")
if receipt.get("robot_policy", {}).get("upstream_sim_cli_invoked") is not False:
    raise SystemExit("generation receipt does not prove sim_cli omission")
if receipt.get("robot_policy", {}).get("robot_actor_loaded") is not False:
    raise SystemExit("generation receipt does not prove robot-free execution")
PY

scene_tmp="${DURABLE_SCENE}.tmp-${SLURM_JOB_ID}"
if [[ -e "${scene_tmp}" ]]; then
    echo "Refusing stale durable scene staging path: ${scene_tmp}" >&2
    exit 1
fi
mkdir -m 700 "${scene_tmp}"
cp -a "${GENERATED_ROOT}/." "${scene_tmp}/"
mkdir -m 700 "${scene_tmp}/metadata"
cp "${ACTIVITY}" "${scene_tmp}/metadata/activity.json"
cp "${REQUEST}" "${scene_tmp}/metadata/embodiedgen_request.json"
cp "${PROTOCOL}" "${scene_tmp}/metadata/protocol.json"
mv "${scene_tmp}" "${DURABLE_SCENE}"

manifest_tmp="${MANIFEST}.tmp-${SLURM_JOB_ID}"
(
    cd "${DURABLE_SCENE}"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${manifest_tmp}"
mv "${manifest_tmp}" "${MANIFEST}"
retain_receipt
trap - EXIT

echo "fresh_generation_status=passed"
echo "durable_scene=${DURABLE_SCENE}"
echo "compact_receipt=${COMPACT_RECEIPT}"
echo "manifest=${MANIFEST}"
