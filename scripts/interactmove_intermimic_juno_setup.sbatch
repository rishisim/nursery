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
readonly SAM3D_SOURCE_REPOSITORY=HochCC/sam-3d-objects
readonly SAM3D_SOURCE_COMMIT=01417d16fb5cc762a60f370c1bf7f59d603ddfaf
readonly SAM3D_CHECKPOINT_REPOSITORY=facebook/sam-3d-objects
readonly SAM3D_CHECKPOINT_REVISION=2e73555018d2741ccd486e56c24fac41155a1dc6
readonly SCRATCH_ROOT=/scratch/juno/dal503972/interactmove_intermimic
readonly DURABLE_ROOT=/work/dal503972/interactmove_intermimic
readonly SOURCE_ROOT="${SCRATCH_ROOT}/source/EmbodiedGen"
readonly ENV_ROOT="${SCRATCH_ROOT}/environments/embodiedgen-v2.0.1-py310-cu126"
readonly RUN_ROOT="${SCRATCH_ROOT}/runs/setup/${SLURM_JOB_ID}"
readonly BASIC_MARKER="${ENV_ROOT}/.embodiedgen-basic-${EMBODIEDGEN_COMMIT}"
readonly MANIFEST="${DURABLE_ROOT}/manifests/embodiedgen_environment_${SLURM_JOB_ID}.txt"
readonly RECEIPT="${DURABLE_ROOT}/compact_records/embodiedgen_setup_${SLURM_JOB_ID}.json"
readonly SAM3D_ROOT="${DURABLE_ROOT}/checkpoints/sam-3d-objects-${SAM3D_CHECKPOINT_REVISION}"
readonly SAM3D_STAGING="${SAM3D_ROOT}.staging"
readonly SAM3D_MANIFEST="${DURABLE_ROOT}/manifests/sam3d_${SAM3D_CHECKPOINT_REVISION}.sha256"
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
export PIP_CACHE_DIR="${SCRATCH_ROOT}/caches/pip"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"
export EMBODIEDGEN_TORCH_CUDA_ARCH_LIST=9.0
export TORCH_CUDA_ARCH_LIST=9.0
export TCNN_CUDA_ARCHITECTURES=90
export IMIM_SETUP_MANIFEST="${MANIFEST}"
export IMIM_SETUP_RECEIPT="${RECEIPT}"
export IMIM_SETUP_SOURCE="${SOURCE_ROOT}"
export IMIM_SETUP_ENV="${ENV_ROOT}"
export IMIM_SETUP_SAM3D_ROOT="${SAM3D_ROOT}"
export IMIM_SETUP_SAM3D_MANIFEST="${SAM3D_MANIFEST}"
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
sam3d_root = Path(os.environ["IMIM_SETUP_SAM3D_ROOT"])
sam3d_manifest = Path(os.environ["IMIM_SETUP_SAM3D_MANIFEST"])
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
    "sam3d": {
        "source_repository": "HochCC/sam-3d-objects",
        "source_commit": "01417d16fb5cc762a60f370c1bf7f59d603ddfaf",
        "actual_source_commit": output(
            ["git", "-C", str(source / "thirdparty" / "sam3d"), "rev-parse", "HEAD"]
        ),
        "source_license": "SAM License",
        "checkpoint_repository": "facebook/sam-3d-objects",
        "checkpoint_revision": "2e73555018d2741ccd486e56c24fac41155a1dc6",
        "checkpoint_license": "SAM License",
        "access_status": "authenticated_admitted",
        "checkpoint_path": str(sam3d_root),
        "manifest_path": str(sam3d_manifest),
        "manifest_sha256": (
            hashlib.sha256(sam3d_manifest.read_bytes()).hexdigest()
            if sam3d_manifest.is_file()
            else None
        ),
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
git -C "${SOURCE_ROOT}" submodule update --init --recursive thirdparty/sam3d
if [[ "$(git -C "${SOURCE_ROOT}/thirdparty/sam3d" rev-parse HEAD)" != "${SAM3D_SOURCE_COMMIT}" ]]; then
    echo "SAM3D source checkout does not match the frozen submodule commit" >&2
    exit 1
fi
git -C "${SOURCE_ROOT}/thirdparty/sam3d" diff --quiet
git -C "${SOURCE_ROOT}/thirdparty/sam3d" diff --cached --quiet

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
export IMIM_SAM3D_REPOSITORY="${SAM3D_CHECKPOINT_REPOSITORY}"
export IMIM_SAM3D_REVISION="${SAM3D_CHECKPOINT_REVISION}"
export IMIM_SAM3D_ROOT="${SAM3D_ROOT}"
export IMIM_SAM3D_STAGING="${SAM3D_STAGING}"
export IMIM_SAM3D_MANIFEST="${SAM3D_MANIFEST}"
python - <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download


EXPECTED = {
    "LICENSE": 8204,
    "README.md": 7708,
    "checkpoints/pipeline.yaml": 3548,
    "checkpoints/slat_decoder_gs.ckpt": 171476155,
    "checkpoints/slat_decoder_gs.yaml": 576,
    "checkpoints/slat_decoder_gs_4.ckpt": 170269801,
    "checkpoints/slat_decoder_gs_4.yaml": 575,
    "checkpoints/slat_decoder_mesh.ckpt": 363726862,
    "checkpoints/slat_decoder_mesh.pt": 363728714,
    "checkpoints/slat_decoder_mesh.yaml": 300,
    "checkpoints/slat_encoder.ckpt": 173263986,
    "checkpoints/slat_encoder.yaml": 268,
    "checkpoints/slat_generator.ckpt": 4906537684,
    "checkpoints/slat_generator.yaml": 1986,
    "checkpoints/ss_decoder.ckpt": 147609242,
    "checkpoints/ss_decoder.yaml": 244,
    "checkpoints/ss_encoder.ckpt": 119085402,
    "checkpoints/ss_encoder.safetensors": 0,
    "checkpoints/ss_encoder.yaml": 231,
    "checkpoints/ss_generator.ckpt": 6690136964,
    "checkpoints/ss_generator.yaml": 5076,
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
        raise RuntimeError(f"SAM3D checkpoint file mismatch: missing={missing}, extra={extra}")
    records = []
    for relative, expected_size in sorted(EXPECTED.items()):
        path = root / relative
        if path.is_symlink() or path.stat().st_size != expected_size:
            raise RuntimeError(f"SAM3D checkpoint size or path mismatch: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        records.append(f"{digest.hexdigest()}  {relative}")
    return records


repository = os.environ["IMIM_SAM3D_REPOSITORY"]
revision = os.environ["IMIM_SAM3D_REVISION"]
root = Path(os.environ["IMIM_SAM3D_ROOT"])
staging = Path(os.environ["IMIM_SAM3D_STAGING"])
manifest = Path(os.environ["IMIM_SAM3D_MANIFEST"])
if root.exists() and staging.exists():
    raise RuntimeError("SAM3D final and staging directories both exist")
if not root.exists():
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        revision=revision,
        local_dir=staging,
        allow_patterns=["checkpoints/*", "LICENSE", "README.md"],
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
        raise RuntimeError("existing SAM3D manifest does not match checkpoint bytes")
    temporary.unlink()
else:
    temporary.replace(manifest)
print(f"sam3d_checkpoint_files={len(records)}")
print(f"sam3d_checkpoint_bytes={sum(EXPECTED.values())}")
PY
unset HF_TOKEN

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
    echo "Hash-bound SAM3D qualification image is missing or unsafe" >&2
    exit 1
fi
if [[ "$(sha256sum "${RESUME_IMAGE}" | cut -d' ' -f1)" != "317b980184bba52c5acb92de0e69d7992ba01dec7746fe5711dca246a80367ef" ]]; then
    echo "Hash-bound SAM3D qualification image does not match job 328320" >&2
    exit 1
fi
install -d -m 700 "${RUN_ROOT}/weights"
ln -s "${SAM3D_ROOT}" "${RUN_ROOT}/weights/sam-3d-objects"
if [[ "$(readlink -f "${RUN_ROOT}/weights/sam-3d-objects")" != "${SAM3D_ROOT}" ]]; then
    echo "SAM3D checkpoint binding failed" >&2
    exit 1
fi
(
    cd "${RUN_ROOT}"
    PYTHONPATH="${SOURCE_ROOT}:${SOURCE_ROOT}/thirdparty/sam3d${PYTHONPATH:+:${PYTHONPATH}}" python - "${RESUME_IMAGE}" <<'PY'
import gc
import json
import sys

import torch
from PIL import Image

from embodied_gen.models.sam3d import Sam3dInference

pipeline = Sam3dInference(local_dir="weights/sam-3d-objects")
image = Image.open(sys.argv[1]).convert("RGBA")
outputs = pipeline.run(
    image,
    seed=2026081502,
    stage1_inference_steps=1,
    stage2_inference_steps=1,
)
assert len(outputs["mesh"]) == 1
assert len(outputs["gaussian"]) == 1
mesh = outputs["mesh"][0]
assert mesh.vertices.shape[0] > 0
assert mesh.faces.shape[0] > 0
print(json.dumps({
    "backend": "SAM3D",
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "cuda_device": torch.cuda.get_device_name(0),
    "mesh_faces": int(mesh.faces.shape[0]),
    "mesh_vertices": int(mesh.vertices.shape[0]),
    "smoke_stage1_steps": 1,
    "smoke_stage2_steps": 1,
}, sort_keys=True))
del outputs, pipeline
gc.collect()
torch.cuda.empty_cache()
PY
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
