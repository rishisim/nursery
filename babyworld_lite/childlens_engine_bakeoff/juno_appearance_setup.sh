#!/usr/bin/env bash
# Prepare the isolated public/synthetic Phase 3 environment on Juno.

set -euo pipefail

work_root="${1:-/work/dal503972/embodied_phase3}"
if [[ "${work_root}" != /work/dal503972/embodied_phase3 ]]; then
    echo "refusing unexpected work root: ${work_root}" >&2
    exit 2
fi

umask 077
mkdir -p \
    "${work_root}/bin" \
    "${work_root}/cache/huggingface" \
    "${work_root}/cache/uv" \
    "${work_root}/envs" \
    "${work_root}/inputs" \
    "${work_root}/logs" \
    "${work_root}/models" \
    "${work_root}/runs/appearance" \
    "${work_root}/source"
chmod 700 "${work_root}"

uv_bin="${work_root}/bin/uv"
if [[ ! -x "${uv_bin}" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${work_root}/bin" sh
fi
"${uv_bin}" --version

export HF_HOME="${work_root}/cache/huggingface"
export UV_CACHE_DIR="${work_root}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${work_root}/python"

verify_checkout() {
    local repository="$1"
    local revision="$2"
    local destination="$3"
    if [[ ! -d "${destination}/.git" ]]; then
        git clone --filter=blob:none "${repository}" "${destination}"
    fi
    git -C "${destination}" fetch --filter=blob:none origin "${revision}"
    git -C "${destination}" checkout --detach "${revision}"
    local actual_revision
    actual_revision="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual_revision}" != "${revision}" ]]; then
        echo "revision mismatch for ${destination}: ${actual_revision}" >&2
        exit 3
    fi
}

verify_checkout \
    https://github.com/NVIDIA/cosmos.git \
    404b9bf2144640834c63ae7d9e7269e0f4ea02cb \
    "${work_root}/source/cosmos"
verify_checkout \
    https://github.com/NVIDIA/cosmos-framework.git \
    5e67049cd94acb667786f1e6dd0dab821cb90c97 \
    "${work_root}/source/cosmos-framework"
verify_checkout \
    https://github.com/wuzy2115/oscar-public.git \
    4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb \
    "${work_root}/source/oscar-public"

cosmos_env="${work_root}/envs/cosmos"
(
    cd "${work_root}/source/cosmos-framework"
    env LD_LIBRARY_PATH= UV_PROJECT_ENVIRONMENT="${cosmos_env}" \
        "${uv_bin}" sync \
        --python 3.13 \
        --no-dev \
        --group=cu128
)

oscar_env="${work_root}/envs/oscar"
if [[ ! -x "${oscar_env}/bin/python" ]]; then
    env LD_LIBRARY_PATH= "${uv_bin}" venv --python 3.13 "${oscar_env}"
fi
env LD_LIBRARY_PATH= "${uv_bin}" pip install \
    --python "${oscar_env}/bin/python" \
    torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
env LD_LIBRARY_PATH= "${uv_bin}" pip install \
    --python "${oscar_env}/bin/python" \
    -r "${work_root}/source/oscar-public/requirements_minimal.txt"
env LD_LIBRARY_PATH= "${uv_bin}" pip install \
    --python "${oscar_env}/bin/python" \
    'transformer-engine==2.12.0+cu128.torch210' \
    --find-links https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/transformer-engine

"${cosmos_env}/bin/python" - "${work_root}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(sys.argv[1])
models = (
    (
        "nvidia/Cosmos3-Nano",
        "411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
        root / "models/cosmos3_nano",
    ),
    (
        "zywu2115/OSCAR-2B",
        "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6",
        root / "models/oscar_2b",
    ),
    (
        "nvidia/Cosmos-Reason1-7B",
        "375e24000b24baed78f4618d3dd779e47cd96323",
        root / "models/cosmos_reason1_7b",
    ),
)
for repository, revision, destination in models:
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=destination,
    )
PY

"${cosmos_env}/bin/python" - "${work_root}" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
receipt = {
    "schema": "EmbodiedAppearanceJunoSetup.v1",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": platform.node(),
    "python": platform.python_version(),
    "public_synthetic_only": True,
    "source_revisions": {
        "cosmos": "404b9bf2144640834c63ae7d9e7269e0f4ea02cb",
        "cosmos_framework": "5e67049cd94acb667786f1e6dd0dab821cb90c97",
        "oscar_public": "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb",
    },
    "model_revisions": {
        "cosmos3_nano": "411f42a8fdfb8c5b2583cb8786e0938f49796eaa",
        "oscar_2b": "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6",
        "cosmos_reason1_7b": "375e24000b24baed78f4618d3dd779e47cd96323",
    },
}
path = root / "runs/appearance/setup_receipt.json"
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

du -sh "${work_root}/envs" "${work_root}/models" "${work_root}/source"
