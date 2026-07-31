#!/usr/bin/env bash
# Run the frozen Cosmos 3 Nano and OSCAR-2B Phase 3 cells on one Juno GPU.

set -euo pipefail

work_root="${1:-/work/dal503972/embodied_phase3}"
if [[ "${work_root}" != /work/dal503972/embodied_phase3 ]]; then
    echo "refusing unexpected work root: ${work_root}" >&2
    exit 2
fi

umask 077
run_root="${work_root}/runs/appearance"
input_root="${work_root}/inputs"
mkdir -p "${run_root}/raw/cosmos3_nano" "${run_root}/raw/oscar_2b"

export HF_HOME="${work_root}/cache/huggingface"
export HF_HUB_OFFLINE=1
export COSMOS_TRAINING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

driver_record="${run_root}/gpu_preflight.txt"
{
    date -u +%Y-%m-%dT%H:%M:%SZ
    hostname
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
    nvidia-smi
} >"${driver_record}" 2>&1

cosmos_status=0
(
    cd "${work_root}/source/cosmos-framework"
    env LD_LIBRARY_PATH= \
        "${work_root}/envs/cosmos/bin/python" \
        -m cosmos_framework.scripts.inference \
        --parallelism-preset=latency \
        -i "${input_root}/prepared/cosmos_specs/*.json" \
        -o "${run_root}/cosmos_official" \
        --checkpoint-path "${work_root}/models/cosmos3_nano" \
        --no-guardrails
) || cosmos_status=$?

if [[ "${cosmos_status}" -eq 0 ]]; then
    "${work_root}/envs/cosmos/bin/python" - "${work_root}" <<'PY'
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
official = root / "runs/appearance/cosmos_official"
normalized = root / "runs/appearance/raw/cosmos3_nano"
for sample_dir in sorted(path for path in official.iterdir() if path.is_dir()):
    name = sample_dir.name
    window, seed = name.rsplit("_seed_", 1)
    source = sample_dir / "vision.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = normalized / window / f"seed_{seed}" / "raw.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
PY
fi

oscar_status=0
export COSMOS_REASON_PATH="${work_root}/models/cosmos_reason1_7b"
export WAN_VAE_PATH="${work_root}/models/wan2_1_vae/Wan2.1_VAE.pth"
export PYTHONPATH="${work_root}/source/oscar-public:${input_root}/repo"
oscar_site_packages="$(${work_root}/envs/oscar/bin/python - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
oscar_cuda_libraries="$(
    find "${oscar_site_packages}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print \
        | sort \
        | paste -sd: -
)"
env LD_LIBRARY_PATH="${oscar_cuda_libraries}:${oscar_site_packages}/torch/lib" \
    "${work_root}/envs/oscar/bin/torchrun" \
    --standalone \
    --nproc-per-node=1 \
    -m babyworld_lite.childlens_engine_bakeoff.appearance_experiment \
    run-oscar \
    "${input_root}/embodied_simulation_appearance.json" \
    "${input_root}/prepared" \
    "${work_root}/models/oscar_2b" \
    "${run_root}/raw/oscar_2b" || oscar_status=$?

"${work_root}/envs/cosmos/bin/python" - \
    "${run_root}" "${cosmos_status}" "${oscar_status}" <<'PY'
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root = Path(sys.argv[1])
receipt = {
    "schema": "EmbodiedAppearanceJunoExecution.v1",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cosmos3_nano_exit_code": int(sys.argv[2]),
    "oscar_2b_exit_code": int(sys.argv[3]),
    "public_synthetic_only": True,
    "neural_audio_authoritative": False,
}
(run_root / "execution_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY

exit 0
