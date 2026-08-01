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

cosmos_runtime_checkpoint="${run_root}/runtime/cosmos3_nano"
"${work_root}/envs/cosmos/bin/python" - "${work_root}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
source = root / "models/cosmos3_nano"
runtime = root / "runs/appearance/runtime/cosmos3_nano"
runtime.mkdir(parents=True, exist_ok=True)
for child in source.iterdir():
    if child.name in {".cache", "config.json"}:
        continue
    link = runtime / child.name
    if link.is_symlink() and link.resolve() == child.resolve():
        continue
    link.unlink(missing_ok=True)
    link.symlink_to(child, target_is_directory=child.is_dir())

source_config = source / "config.json"
config = json.loads(source_config.read_text())
sound = config["model"]["config"]["sound_tokenizer"]
sound["from_checkpoint"] = True
runtime_config = runtime / "config.json"
runtime_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
receipt = {
    "schema": "EmbodiedCosmosRuntimeConfig.v1",
    "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
    "runtime_config_sha256": hashlib.sha256(runtime_config.read_bytes()).hexdigest(),
    "sound_tokenizer_source": "checkpoint-bundled public asset",
    "sound_generation_enabled_in_samples": False,
}
(runtime.parent / "runtime_config_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY

cosmos_status=0
cosmos_site_packages="$(${work_root}/envs/cosmos/bin/python - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)"
cosmos_cuda_libraries="$(
    find "${cosmos_site_packages}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print \
        | sort \
        | paste -sd: -
)"
cosmos_av_libraries="${cosmos_site_packages}/av.libs"
(
    cd "${work_root}/source/cosmos-framework"
    env LD_LIBRARY_PATH="${cosmos_av_libraries}:${cosmos_cuda_libraries}:${cosmos_site_packages}/torch/lib" \
        "${work_root}/envs/cosmos/bin/python" \
        -m cosmos_framework.scripts.inference \
        --parallelism-preset=latency \
        -i "${input_root}/prepared/cosmos_specs/*.json" \
        -o "${run_root}/cosmos_official" \
        --checkpoint-path "${cosmos_runtime_checkpoint}" \
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
