#!/usr/bin/env bash
#SBATCH --job-name=imim-scene
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/juno/dal503972/interactmove_intermimic/logs/scene-%j.out
#SBATCH --error=/scratch/juno/dal503972/interactmove_intermimic/logs/scene-%j.err

set -euo pipefail
umask 077

readonly EMBODIEDGEN_COMMIT=9b333554254af196bace88c1a171a3bf047fa09c
readonly ACTIVITY_ID=red-mug-mouth-return
readonly SCRATCH_ROOT=/scratch/juno/dal503972/interactmove_intermimic
readonly DURABLE_ROOT=/work/dal503972/interactmove_intermimic
readonly ENV_ROOT="${SCRATCH_ROOT}/environments/embodiedgen-v2.0.1-py310-cu126"
readonly SOURCE_ROOT="${SCRATCH_ROOT}/source/EmbodiedGen"
readonly NURSERY_ROOT="${SCRATCH_ROOT}/source/nursery"
readonly SCENE_ROOT="${DURABLE_ROOT}/scene_packages/${ACTIVITY_ID}"
readonly RUN_ROOT="${SCRATCH_ROOT}/runs/scene_canary/${SLURM_JOB_ID}"
readonly RESULT_ROOT="${DURABLE_ROOT}/simulation_canaries/${ACTIVITY_ID}"
readonly RUNNER="${NURSERY_ROOT}/scripts/run_interactmove_intermimic_scene_canary.py"
readonly NURSERY_COMMIT="${IMIM_NURSERY_COMMIT:?submit with the exact nursery commit}"
readonly ACTIVITY="${SCENE_ROOT}/metadata/activity.json"
readonly REQUEST="${SCENE_ROOT}/metadata/embodiedgen_request.json"
readonly PROTOCOL="${SCENE_ROOT}/metadata/protocol.json"
readonly GENERATION_RECEIPT="${SCENE_ROOT}/generation_receipt.json"
readonly BUNDLE="${RUN_ROOT}/scene_bundle.json"
readonly COMPACT_BUNDLE="${DURABLE_ROOT}/compact_records/scene_bundle_${ACTIVITY_ID}_${SLURM_JOB_ID}.json"
readonly COMPACT_RECEIPT="${DURABLE_ROOT}/compact_records/fresh_scene_canary_${SLURM_JOB_ID}.json"

install -d -m 700 \
    "${RUN_ROOT}" \
    "${DURABLE_ROOT}/compact_records" \
    "${DURABLE_ROOT}/simulation_canaries"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    echo "EmbodiedGen environment is missing: ${ENV_ROOT}" >&2
    exit 1
fi
if [[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" != "${EMBODIEDGEN_COMMIT}" ]]; then
    echo "EmbodiedGen checkout does not match the frozen commit" >&2
    exit 1
fi
git -C "${SOURCE_ROOT}" diff --quiet
git -C "${SOURCE_ROOT}" diff --cached --quiet
if [[ "$(git -C "${NURSERY_ROOT}" rev-parse HEAD)" != "${NURSERY_COMMIT}" ]]; then
    echo "Nursery checkout does not match IMIM_NURSERY_COMMIT" >&2
    exit 1
fi
git -C "${NURSERY_ROOT}" diff --quiet
git -C "${NURSERY_ROOT}" diff --cached --quiet
for required in \
    "${SCENE_ROOT}/layout.json" \
    "${ACTIVITY}" \
    "${REQUEST}" \
    "${PROTOCOL}" \
    "${GENERATION_RECEIPT}" \
    "${RUNNER}"; do
    if [[ ! -f "${required}" || -L "${required}" ]]; then
        echo "Required regular file is missing or unsafe: ${required}" >&2
        exit 1
    fi
done
if [[ -e "${RESULT_ROOT}" || -e "${COMPACT_BUNDLE}" || -e "${COMPACT_RECEIPT}" ]]; then
    echo "Refusing to overwrite canonical fresh-scene result" >&2
    exit 1
fi

export EMBODIEDGEN_SOURCE_ROOT="${SOURCE_ROOT}"
export PYTHONPATH="${NURSERY_ROOT}:${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
set +u
conda activate "${ENV_ROOT}"
set -u

python -m babyworld_lite.interactmove_intermimic compile-scene-bundle \
    "${ACTIVITY}" \
    --request "${REQUEST}" \
    --scene-root "${SCENE_ROOT}" \
    --layout layout.json \
    --protocol "${PROTOCOL}" \
    --output "${BUNDLE}"
python -m babyworld_lite.interactmove_intermimic validate-scene-bundle \
    "${BUNDLE}" --print-digest

python "${RUNNER}" \
    --layout "${SCENE_ROOT}/layout.json" \
    --generation-receipt "${GENERATION_RECEIPT}" \
    --scene-bundle "${BUNDLE}" \
    --output-dir "${RUN_ROOT}/result" \
    --duration-s 10 \
    --sim-hz 200 \
    --video-fps 30 \
    --width 512 \
    --height 512

result_tmp="${RESULT_ROOT}.tmp-${SLURM_JOB_ID}"
if [[ -e "${result_tmp}" ]]; then
    echo "Refusing stale result staging path: ${result_tmp}" >&2
    exit 1
fi
mkdir -m 700 "${result_tmp}"
cp -a "${RUN_ROOT}/result/." "${result_tmp}/"
cp "${BUNDLE}" "${result_tmp}/scene_bundle.json"
mv "${result_tmp}" "${RESULT_ROOT}"
cp "${BUNDLE}" "${COMPACT_BUNDLE}.tmp"
mv "${COMPACT_BUNDLE}.tmp" "${COMPACT_BUNDLE}"
cp "${RESULT_ROOT}/receipt.json" "${COMPACT_RECEIPT}.tmp"
mv "${COMPACT_RECEIPT}.tmp" "${COMPACT_RECEIPT}"

echo "scene_bundle=${COMPACT_BUNDLE}"
echo "simulation_result=${RESULT_ROOT}"
echo "simulation_receipt=${COMPACT_RECEIPT}"
