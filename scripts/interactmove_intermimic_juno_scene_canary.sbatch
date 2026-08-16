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

readonly DATASET_COMMIT=58258b50a0fc95034f2f3cc03b332ec6f72b91fd
readonly SCRATCH_ROOT=/scratch/juno/dal503972/interactmove_intermimic
readonly DURABLE_ROOT=/work/dal503972/interactmove_intermimic
readonly ENV_ROOT="${SCRATCH_ROOT}/environments/embodiedgen-v2.0.1-py310-cu126"
readonly SOURCE_ROOT="${SCRATCH_ROOT}/source/EmbodiedGen"
readonly DOWNLOAD_ROOT="${SCRATCH_ROOT}/downloads/EmbodiedGenData-${DATASET_COMMIT}"
readonly SCENE_ROOT="${DURABLE_ROOT}/scene_packages/public_example_task_0000-${DATASET_COMMIT}"
readonly RESULT_ROOT="${DURABLE_ROOT}/simulation_canaries/official_task_0000"
readonly RUN_ROOT="${SCRATCH_ROOT}/runs/scene_canary/${SLURM_JOB_ID}"
readonly MANIFEST="${DURABLE_ROOT}/manifests/public_example_task_0000-${DATASET_COMMIT}.sha256"
readonly RUNNER="${SCRATCH_ROOT}/remote_scripts/run_interactmove_intermimic_scene_canary.py"

install -d -m 700 \
    "${RUN_ROOT}" \
    "${SCRATCH_ROOT}/caches/huggingface" \
    "${SCRATCH_ROOT}/downloads" \
    "${DURABLE_ROOT}/compact_records" \
    "${DURABLE_ROOT}/manifests" \
    "${DURABLE_ROOT}/scene_packages" \
    "${DURABLE_ROOT}/simulation_canaries"

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    echo "EmbodiedGen environment is missing: ${ENV_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${RUNNER}" ]]; then
    echo "Canary runner is missing: ${RUNNER}" >&2
    exit 1
fi
if [[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" != "9b333554254af196bace88c1a171a3bf047fa09c" ]]; then
    echo "EmbodiedGen checkout does not match the frozen commit" >&2
    exit 1
fi

export HF_HOME="${SCRATCH_ROOT}/caches/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export EMBODIEDGEN_SOURCE_ROOT="${SOURCE_ROOT}"
export PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
set +u
conda activate "${ENV_ROOT}"
set -u

if [[ ! -d "${SCENE_ROOT}" ]]; then
    python - "${DOWNLOAD_ROOT}" <<'PY'
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
from urllib.parse import quote

import requests

root = Path(sys.argv[1])
token_file = Path.home() / ".cache" / "huggingface" / "token"
token = token_file.read_text(encoding="utf-8").strip() if token_file.is_file() else None
headers = {} if token is None else {"Authorization": f"Bearer {token}"}
revision = "58258b50a0fc95034f2f3cc03b332ec6f72b91fd"
prefix = "example_layouts/task_0000"
api_url = (
    "https://huggingface.co/api/datasets/HorizonRobotics/EmbodiedGenData/"
    f"tree/{revision}/{prefix}?recursive=true&expand=false&limit=1000"
)
files = []
while api_url:
    response = requests.get(api_url, headers=headers, timeout=60)
    response.raise_for_status()
    files.extend(row for row in response.json() if row.get("type") == "file")
    api_url = response.links.get("next", {}).get("url")
if not files:
    raise RuntimeError("official task_0000 tree contained no files")

def download(row):
    relative = row["path"]
    if not relative.startswith(prefix + "/"):
        raise RuntimeError(f"unsafe dataset path: {relative}")
    destination = root.joinpath(*relative.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = row.get("size")
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    url = (
        "https://huggingface.co/datasets/HorizonRobotics/EmbodiedGenData/"
        f"resolve/{revision}/{quote(relative, safe='/')}"
    )
    part = destination.with_name(destination.name + f".part-{os.getpid()}")
    with requests.get(url, headers=headers, stream=True, timeout=300) as result:
        result.raise_for_status()
        with part.open("wb") as stream:
            for chunk in result.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if expected_size is not None and part.stat().st_size != expected_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"size mismatch for {relative}")
    os.replace(part, destination)

with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(download, files))
print(f"downloaded_or_verified={len(files)}")
PY
    scene_source="${DOWNLOAD_ROOT}/example_layouts/task_0000"
    if [[ ! -f "${scene_source}/layout.json" ]]; then
        echo "Pinned public scene download has no layout.json" >&2
        exit 1
    fi
    scene_tmp="${SCENE_ROOT}.tmp-${SLURM_JOB_ID}"
    if [[ -e "${scene_tmp}" ]]; then
        echo "Refusing stale scene staging path: ${scene_tmp}" >&2
        exit 1
    fi
    mkdir -m 700 "${scene_tmp}"
    cp -aL "${scene_source}/." "${scene_tmp}/"
    mv "${scene_tmp}" "${SCENE_ROOT}"
fi

if [[ ! -f "${MANIFEST}" ]]; then
    manifest_tmp="${MANIFEST}.tmp-${SLURM_JOB_ID}"
    (
        cd "${SCENE_ROOT}"
        find . -type f -print0 | sort -z | xargs -0 sha256sum
    ) > "${manifest_tmp}"
    mv "${manifest_tmp}" "${MANIFEST}"
fi

if [[ -e "${RESULT_ROOT}" ]]; then
    echo "Refusing to overwrite canonical canary result: ${RESULT_ROOT}" >&2
    exit 1
fi

python "${RUNNER}" \
    --layout "${SCENE_ROOT}/layout.json" \
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
mv "${result_tmp}" "${RESULT_ROOT}"
cp "${RESULT_ROOT}/receipt.json" \
    "${DURABLE_ROOT}/compact_records/scene_canary_${SLURM_JOB_ID}.json"
