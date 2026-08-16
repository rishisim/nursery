#!/usr/bin/env bash
#SBATCH --job-name=imim-setup
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/juno/dal503972/interactmove_intermimic/logs/setup-%j.out
#SBATCH --error=/scratch/juno/dal503972/interactmove_intermimic/logs/setup-%j.err

set -euo pipefail
umask 077

readonly EMBODIEDGEN_COMMIT=9b333554254af196bace88c1a171a3bf047fa09c
readonly OPENAI_SDK_VERSION=3.1.0
readonly TRELLIS_SOURCE_COMMIT=55a8e8164b195bbf927e0978f00e76c835e6011f
readonly TRELLIS_FLEXICUBES_COMMIT=f97beb0dd3c6c68f3ab5696b6dcaf9af69f0514e
readonly TRELLIS_REPOSITORY=microsoft/TRELLIS-image-large
readonly TRELLIS_REVISION=25e0d31ffbebe4b5a97464dd851910efc3002d96
readonly DINOV2_REPOSITORY=https://github.com/facebookresearch/dinov2.git
readonly DINOV2_COMMIT=7764ea0f912e53c92e82eb78a2a1631e92725fc8
readonly SCRATCH_ROOT=/scratch/juno/dal503972/interactmove_intermimic
readonly DURABLE_ROOT=/work/dal503972/interactmove_intermimic
readonly SOURCE_ROOT="${SCRATCH_ROOT}/source/EmbodiedGen"
readonly ENV_ROOT="${SCRATCH_ROOT}/environments/embodiedgen-v2.0.1-py310-cu126"
readonly RUN_ROOT="${SCRATCH_ROOT}/runs/setup/${SLURM_JOB_ID}"
readonly BASIC_MARKER="${ENV_ROOT}/.embodiedgen-basic-${EMBODIEDGEN_COMMIT}"
readonly MANIFEST="${DURABLE_ROOT}/manifests/embodiedgen_environment_${SLURM_JOB_ID}.txt"
readonly RECEIPT="${DURABLE_ROOT}/compact_records/embodiedgen_setup_${SLURM_JOB_ID}.json"
readonly TRELLIS_ROOT="${DURABLE_ROOT}/checkpoints/TRELLIS-image-large-${TRELLIS_REVISION}"
readonly TRELLIS_STAGING="${TRELLIS_ROOT}.staging"
readonly TRELLIS_MANIFEST="${DURABLE_ROOT}/manifests/trellis_${TRELLIS_REVISION}.sha256"
readonly TORCH_HOME="${DURABLE_ROOT}/checkpoints/torch-hub"
readonly DINOV2_SOURCE="${TORCH_HOME}/hub/facebookresearch_dinov2_main"
readonly DINOV2_MANIFEST="${DURABLE_ROOT}/manifests/dinov2_${DINOV2_COMMIT}.sha256"
readonly RESUME_IMAGE="${SCRATCH_ROOT}/runs/fresh_generation/328320/scene/images/table.png"

install -d -m 700 \
    "${RUN_ROOT}" \
    "${SCRATCH_ROOT}/caches/conda" \
    "${SCRATCH_ROOT}/caches/huggingface" \
    "${SCRATCH_ROOT}/caches/pip" \
    "${SCRATCH_ROOT}/downloads" \
    "${SCRATCH_ROOT}/environments" \
    "${SCRATCH_ROOT}/source" \
    "${DURABLE_ROOT}/checkpoints" \
    "${DURABLE_ROOT}/compact_records" \
    "${DURABLE_ROOT}/manifests"

export CONDA_PKGS_DIRS="${SCRATCH_ROOT}/caches/conda"
export HF_HOME="${SCRATCH_ROOT}/caches/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME
export PIP_CACHE_DIR="${SCRATCH_ROOT}/caches/pip"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export EMBODIEDGEN_TORCH_CUDA_ARCH_LIST=9.0
export TORCH_CUDA_ARCH_LIST=9.0
export TCNN_CUDA_ARCHITECTURES=90
export IMIM_SETUP_MANIFEST="${MANIFEST}"
export IMIM_SETUP_RECEIPT="${RECEIPT}"
export IMIM_SETUP_SOURCE="${SOURCE_ROOT}"
export IMIM_SETUP_ENV="${ENV_ROOT}"
export IMIM_SETUP_TRELLIS_ROOT="${TRELLIS_ROOT}"
export IMIM_SETUP_TRELLIS_MANIFEST="${TRELLIS_MANIFEST}"
export IMIM_SETUP_DINOV2_SOURCE="${DINOV2_SOURCE}"
export IMIM_SETUP_DINOV2_MANIFEST="${DINOV2_MANIFEST}"
export IMIM_SETUP_STATUS=failed
export IMIM_SETUP_STARTED_UTC
IMIM_SETUP_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

write_receipt() {
    /usr/bin/python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


def output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


manifest = Path(os.environ["IMIM_SETUP_MANIFEST"])
source = Path(os.environ["IMIM_SETUP_SOURCE"])
environment = Path(os.environ["IMIM_SETUP_ENV"])
trellis_root = Path(os.environ["IMIM_SETUP_TRELLIS_ROOT"])
trellis_manifest = Path(os.environ["IMIM_SETUP_TRELLIS_MANIFEST"])
dinov2_source = Path(os.environ["IMIM_SETUP_DINOV2_SOURCE"])
dinov2_manifest = Path(os.environ["IMIM_SETUP_DINOV2_MANIFEST"])
manifest_digest = None
if manifest.is_file():
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
receipt = {
    "schema": "InteractMoveInterMimicEmbodiedGenSetup",
    "schema_version": 1,
    "status": os.environ["IMIM_SETUP_STATUS"],
    "started_utc": os.environ["IMIM_SETUP_STARTED_UTC"],
    "finished_utc": output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
    "slurm": {
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "visible_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
    },
    "source": {
        "path": str(source),
        "expected_commit": "9b333554254af196bace88c1a171a3bf047fa09c",
        "actual_commit": output(["git", "-C", str(source), "rev-parse", "HEAD"]),
    },
    "environment": {
        "path": str(environment),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_digest,
        "python": output([str(environment / "bin/python"), "--version"]),
    },
    "trellis": {
        "source_repository": "microsoft/TRELLIS",
        "source_commit": "55a8e8164b195bbf927e0978f00e76c835e6011f",
        "source_license": "MIT",
        "checkpoint_repository": "microsoft/TRELLIS-image-large",
        "checkpoint_revision": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
        "checkpoint_license": "MIT",
        "checkpoint_path": str(trellis_root),
        "manifest_path": str(trellis_manifest),
        "manifest_sha256": (
            hashlib.sha256(trellis_manifest.read_bytes()).hexdigest()
            if trellis_manifest.is_file()
            else None
        ),
    },
    "dinov2": {
        "repository": "facebookresearch/dinov2",
        "commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        "license": "Apache-2.0",
        "source_path": str(dinov2_source),
        "manifest_path": str(dinov2_manifest),
        "manifest_sha256": (
            hashlib.sha256(dinov2_manifest.read_bytes()).hexdigest()
            if dinov2_manifest.is_file()
            else None
        ),
    },
    "sam3d_comparison": {
        "access_status": "pending_not_admitted",
        "blocks_trellis_run": False,
    },
}
path = Path(os.environ["IMIM_SETUP_RECEIPT"])
path.write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}
trap write_receipt EXIT

source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    conda create --prefix "${ENV_ROOT}" python=3.10.13 -y
fi
set +u
conda activate "${ENV_ROOT}"
set -u

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
    git clone https://github.com/HorizonRobotics/EmbodiedGen.git "${SOURCE_ROOT}"
fi
if [[ -n "$(git -C "${SOURCE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to modify a dirty EmbodiedGen source checkout: ${SOURCE_ROOT}" >&2
    exit 1
fi
git -C "${SOURCE_ROOT}" fetch origin "${EMBODIEDGEN_COMMIT}"
git -C "${SOURCE_ROOT}" checkout --detach "${EMBODIEDGEN_COMMIT}"
git -C "${SOURCE_ROOT}" submodule update --init --recursive thirdparty/TRELLIS
if [[ "$(git -C "${SOURCE_ROOT}/thirdparty/TRELLIS" rev-parse HEAD)" != "${TRELLIS_SOURCE_COMMIT}" ]]; then
    echo "TRELLIS source checkout does not match the frozen submodule commit" >&2
    exit 1
fi
if [[ "$(git -C "${SOURCE_ROOT}/thirdparty/TRELLIS/trellis/representations/mesh/flexicubes" rev-parse HEAD)" != "${TRELLIS_FLEXICUBES_COMMIT}" ]]; then
    echo "TRELLIS FlexiCubes checkout does not match the frozen submodule commit" >&2
    exit 1
fi

cd "${SOURCE_ROOT}"
if [[ ! -f "${ENV_ROOT}/etc/conda/activate.d/cuda126.sh" ]]; then
    bash install.sh cu126
fi
set +u
conda deactivate
conda activate "${ENV_ROOT}"
set -u
export EMBODIEDGEN_TORCH_CUDA_ARCH_LIST=9.0
export TORCH_CUDA_ARCH_LIST=9.0
export TCNN_CUDA_ARCHITECTURES=90
# EmbodiedGen's installer uses the PyTorch CUDA index as its only index.  That
# index does not carry ordinary build dependencies, so resolve those from PyPI
# first and install the pinned CUDA wheels with PyPI as an explicit fallback.
if [[ ! -f "${BASIC_MARKER}" ]] && ! (
    python -c 'import torch; import sapien; import mani_skill; import coacd; import open3d; import trimesh'
    layout-cli --help >/dev/null
    sim-cli --help >/dev/null
); then
    python -m pip install --upgrade \
        'pip>=25.1,<26' \
        'setuptools<81' \
        wheel
    python -m pip install \
        'flit_core>=3.11,<4' \
        'jinja2>=3.1' \
        'typing_extensions>=4.10'
    python -m pip install \
        torch==2.8.0 \
        torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple
    bash install.sh basic
fi
printf '%s\n' "${EMBODIEDGEN_COMMIT}" > "${BASIC_MARKER}"
if ! python -c 'import flash_attn; assert flash_attn.__version__ == "2.8.2"'; then
    FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install \
        --no-cache-dir \
        --no-build-isolation \
        --no-binary flash-attn \
        --force-reinstall \
        --no-deps \
        flash-attn==2.8.2
fi
set +u
conda deactivate
conda activate "${ENV_ROOT}"
set -u
python -m pip install "openai==${OPENAI_SDK_VERSION}"
python - <<'PY'
import openai
from importlib.metadata import PackageNotFoundError, distribution
from openai import OpenAI
from packaging.requirements import Requirement

assert openai.__version__ == "3.1.0"
assert hasattr(OpenAI(api_key="not-a-real-key"), "responses")
pending = ["openai"]
checked = set()
while pending:
    name = pending.pop()
    package = distribution(name)
    canonical_name = package.metadata["Name"].lower().replace("_", "-")
    if canonical_name in checked:
        continue
    checked.add(canonical_name)
    for raw_requirement in package.requires or []:
        requirement = Requirement(raw_requirement)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            dependency = distribution(requirement.name)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"missing OpenAI dependency: {requirement.name}"
            ) from exc
        if requirement.specifier and dependency.version not in requirement.specifier:
            raise RuntimeError(
                f"OpenAI dependency mismatch: {requirement}, got {dependency.version}"
            )
        pending.append(requirement.name)
print("openai_dependency_closure=passed")
PY
set +e
python -m pip check > "${RUN_ROOT}/pip-check.txt" 2>&1
GLOBAL_PIP_CHECK_EXIT_CODE=$?
set -e
if [[ "${GLOBAL_PIP_CHECK_EXIT_CODE}" -ne 0 ]]; then
    echo "Global pip check has unrelated pre-existing conflicts:" >&2
    cat "${RUN_ROOT}/pip-check.txt" >&2
fi

readonly HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
if [[ ! -f "${HF_TOKEN_PATH}" || -L "${HF_TOKEN_PATH}" ]]; then
    echo "Hugging Face token is missing or unsafe" >&2
    exit 1
fi
export HF_TOKEN
HF_TOKEN=$(<"${HF_TOKEN_PATH}")
export IMIM_TRELLIS_REPOSITORY="${TRELLIS_REPOSITORY}"
export IMIM_TRELLIS_REVISION="${TRELLIS_REVISION}"
export IMIM_TRELLIS_ROOT="${TRELLIS_ROOT}"
export IMIM_TRELLIS_STAGING="${TRELLIS_STAGING}"
export IMIM_TRELLIS_MANIFEST="${TRELLIS_MANIFEST}"
python - <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download


EXPECTED = {
    ".gitattributes": 1519,
    "README.md": 464,
    "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json": 843,
    "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors": 171450952,
    "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json": 372,
    "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors": 181903412,
    "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json": 396,
    "ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors": 171450488,
    "ckpts/slat_enc_swin8_B_64l8_fp16.json": 321,
    "ckpts/slat_enc_swin8_B_64l8_fp16.safetensors": 173242816,
    "ckpts/slat_flow_img_dit_L_64l8p2_fp16.json": 442,
    "ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors": 1203755136,
    "ckpts/ss_dec_conv3d_16l8_fp16.json": 245,
    "ckpts/ss_dec_conv3d_16l8_fp16.safetensors": 147591972,
    "ckpts/ss_enc_conv3d_16l8_fp16.json": 244,
    "ckpts/ss_enc_conv3d_16l8_fp16.safetensors": 119068016,
    "ckpts/ss_flow_img_dit_L_16l8_fp16.json": 385,
    "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors": 1130770840,
    "pipeline.json": 1987,
}


def validate(root: Path) -> list[str]:
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }
    if actual != set(EXPECTED):
        missing = sorted(set(EXPECTED) - actual)
        extra = sorted(actual - set(EXPECTED))
        raise RuntimeError(f"TRELLIS checkpoint file mismatch: missing={missing}, extra={extra}")
    records = []
    for relative, expected_size in sorted(EXPECTED.items()):
        path = root / relative
        if path.is_symlink() or path.stat().st_size != expected_size:
            raise RuntimeError(f"TRELLIS checkpoint size or path mismatch: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        records.append(f"{digest.hexdigest()}  {relative}")
    return records


repository = os.environ["IMIM_TRELLIS_REPOSITORY"]
revision = os.environ["IMIM_TRELLIS_REVISION"]
root = Path(os.environ["IMIM_TRELLIS_ROOT"])
staging = Path(os.environ["IMIM_TRELLIS_STAGING"])
manifest = Path(os.environ["IMIM_TRELLIS_MANIFEST"])
if root.exists() and staging.exists():
    raise RuntimeError("TRELLIS final and staging directories both exist")
if not root.exists():
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=staging,
        max_workers=4,
        token=os.environ["HF_TOKEN"],
    )
    validate(staging)
    staging.rename(root)
records = validate(root)
payload = "\n".join(records) + "\n"
temporary = manifest.with_name(manifest.name + ".tmp")
temporary.write_text(payload, encoding="utf-8")
os.chmod(temporary, 0o600)
if manifest.exists():
    if manifest.read_bytes() != temporary.read_bytes():
        raise RuntimeError("existing TRELLIS manifest does not match checkpoint bytes")
    temporary.unlink()
else:
    temporary.replace(manifest)
print(f"trellis_checkpoint_files={len(records)}")
print(f"trellis_checkpoint_bytes={sum(EXPECTED.values())}")
PY
unset HF_TOKEN

install -d -m 700 "${TORCH_HOME}/hub" "${TORCH_HOME}/hub/checkpoints"
if [[ ! -d "${DINOV2_SOURCE}/.git" ]]; then
    git clone "${DINOV2_REPOSITORY}" "${DINOV2_SOURCE}"
fi
if [[ -n "$(git -C "${DINOV2_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing a dirty DINOv2 source checkout" >&2
    exit 1
fi
git -C "${DINOV2_SOURCE}" fetch origin "${DINOV2_COMMIT}"
git -C "${DINOV2_SOURCE}" checkout --detach "${DINOV2_COMMIT}"
if [[ "$(git -C "${DINOV2_SOURCE}" rev-parse HEAD)" != "${DINOV2_COMMIT}" ]]; then
    echo "DINOv2 source checkout does not match the frozen commit" >&2
    exit 1
fi

python - <<'PY'
import json

import torch
import sapien
import mani_skill
import coacd
import open3d
import trimesh

assert torch.cuda.is_available()
tensor = torch.eye(32, device="cuda")
assert bool(torch.equal(tensor @ tensor, tensor))
print(json.dumps({
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "cuda_device": torch.cuda.get_device_name(0),
    "pytorch": torch.__version__,
    "pytorch_cuda": torch.version.cuda,
    "sapien": getattr(sapien, "__version__", None),
}, sort_keys=True))
PY

if [[ ! -f "${RESUME_IMAGE}" || -L "${RESUME_IMAGE}" ]]; then
    echo "Hash-bound TRELLIS qualification image is missing or unsafe" >&2
    exit 1
fi
if [[ "$(sha256sum "${RESUME_IMAGE}" | cut -d' ' -f1)" != "317b980184bba52c5acb92de0e69d7992ba01dec7746fe5711dca246a80367ef" ]]; then
    echo "Hash-bound TRELLIS qualification image does not match job 328320" >&2
    exit 1
fi
install -d -m 700 "${RUN_ROOT}/microsoft"
ln -s "${TRELLIS_ROOT}" "${RUN_ROOT}/microsoft/TRELLIS-image-large"
if [[ "$(readlink -f "${RUN_ROOT}/microsoft/TRELLIS-image-large")" != "${TRELLIS_ROOT}" ]]; then
    echo "TRELLIS checkpoint binding failed" >&2
    exit 1
fi
(
    cd "${RUN_ROOT}"
    PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python - "${RESUME_IMAGE}" <<'PY'
import gc
import json
import sys

import torch
from PIL import Image

from embodied_gen.data.utils import trellis_preprocess
from thirdparty.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained(
    "microsoft/TRELLIS-image-large"
)
pipeline.cuda()
image = trellis_preprocess(Image.open(sys.argv[1]))
outputs = pipeline.run(
    image,
    preprocess_image=False,
    seed=2026081502,
    sparse_structure_sampler_params={"steps": 3},
    slat_sampler_params={"steps": 3},
    formats=["mesh", "gaussian"],
)
assert len(outputs["mesh"]) == 1
assert len(outputs["gaussian"]) == 1
assert outputs["mesh"][0].vertices.shape[0] > 0
assert outputs["mesh"][0].faces.shape[0] > 0
print(json.dumps({
    "backend": "TRELLIS",
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "cuda_device": torch.cuda.get_device_name(0),
    "mesh_faces": int(outputs["mesh"][0].faces.shape[0]),
    "mesh_vertices": int(outputs["mesh"][0].vertices.shape[0]),
    "smoke_sampler_steps": 3,
}, sort_keys=True))
pipeline.cpu()
del outputs, pipeline
gc.collect()
torch.cuda.empty_cache()
PY
)

if [[ ! -s "${TORCH_HOME}/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth" ]]; then
    echo "Pinned DINOv2 checkpoint was not materialized by the TRELLIS smoke test" >&2
    exit 1
fi
dinov2_manifest_tmp="${DINOV2_MANIFEST}.tmp"
(
    cd "${TORCH_HOME}"
    sha256sum hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth > "${dinov2_manifest_tmp}"
)
chmod 600 "${dinov2_manifest_tmp}"
if [[ -e "${DINOV2_MANIFEST}" ]]; then
    if ! cmp -s "${dinov2_manifest_tmp}" "${DINOV2_MANIFEST}"; then
        echo "Existing DINOv2 manifest does not match checkpoint bytes" >&2
        exit 1
    fi
    rm "${dinov2_manifest_tmp}"
else
    mv "${dinov2_manifest_tmp}" "${DINOV2_MANIFEST}"
fi
(
    cd "${TORCH_HOME}"
    sha256sum --check "${DINOV2_MANIFEST}"
)
layout-cli --help >/dev/null
sim-cli --help >/dev/null

{
    printf 'embodiedgen_commit=%s\n' "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
    printf 'torch_cuda_arch_list=%s\n' "${TORCH_CUDA_ARCH_LIST}"
    printf 'openai_sdk=%s\n' "$(python -c 'import openai; print(openai.__version__)')"
    printf 'openai_dependency_closure=passed\n'
    printf 'global_pip_check_exit_code=%s\n' "${GLOBAL_PIP_CHECK_EXIT_CODE}"
    if [[ -s "${RUN_ROOT}/pip-check.txt" ]]; then
        printf '\n[global-pip-check]\n'
        cat "${RUN_ROOT}/pip-check.txt"
    fi
    conda list --explicit
    printf '\n[pip-freeze]\n'
    python -m pip freeze --all
} > "${MANIFEST}"

export IMIM_SETUP_STATUS=passed
