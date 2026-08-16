#!/usr/bin/env bash
#SBATCH --job-name=imim-room
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/juno/dal503972/interactmove_intermimic/logs/room-%j.out
#SBATCH --error=/scratch/juno/dal503972/interactmove_intermimic/logs/room-%j.err

set -euo pipefail
umask 077

readonly EMBODIEDGEN_COMMIT=9b333554254af196bace88c1a171a3bf047fa09c
readonly PANO2ROOM_COMMIT=bbf93ae57086ed700edc6ee445852d4457a9d704
readonly PANO_SOURCE_COMMIT=84a1198dbae40d0ecc7276e54d957f1789566610
readonly PANO_MODEL_REVISION=d20770f5ec0000a2aaa579a1209d11dc92ddb08a
readonly BIG_LAMA_REVISION=05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9
readonly SD2_INPAINT_REVISION=5f74973cbb64c8568780732c17f43eb269d63a0d
readonly REALESRGAN_REVISION=a64fcdebeea17287d830736cd0853df1093b97ab
readonly OMNIDATA_COMMIT=152cf1465313b68cdf5d47bb09568a9357c57086
readonly ACTIVITY_ID=red-mug-mouth-return
readonly SCRATCH_ROOT=/scratch/juno/dal503972/interactmove_intermimic
readonly DURABLE_ROOT=/work/dal503972/interactmove_intermimic
readonly SOURCE_ROOT="${SCRATCH_ROOT}/source/EmbodiedGen"
readonly NURSERY_ROOT="${SCRATCH_ROOT}/source/nursery"
readonly ENV_ROOT="${SCRATCH_ROOT}/environments/embodiedgen-v2.0.1-py310-cu126"
readonly CURRENT_SCENE="${DURABLE_ROOT}/scene_packages/${ACTIVITY_ID}"
readonly CURRENT_RESULT="${DURABLE_ROOT}/simulation_canaries/${ACTIVITY_ID}"
readonly CURRENT_HANDOFF="${DURABLE_ROOT}/compact_records/shared_scene_${ACTIVITY_ID}.json"
readonly RUN_ROOT="${SCRATCH_ROOT}/runs/room_repair/${SLURM_JOB_ID}"
readonly CANDIDATE_SCENE="${RUN_ROOT}/candidate_scene"
readonly GENERATION_WORK="${RUN_ROOT}/scene3d_generation"
readonly BUNDLE="${RUN_ROOT}/scene_bundle.json"
readonly RESULT="${RUN_ROOT}/simulation_result"
readonly HANDOFF="${RUN_ROOT}/shared_scene.json"
readonly ROOM_RUNNER="${NURSERY_ROOT}/scripts/run_interactmove_intermimic_room_repair.py"
readonly CANARY_RUNNER="${NURSERY_ROOT}/scripts/run_interactmove_intermimic_scene_canary.py"
readonly NURSERY_COMMIT="${IMIM_NURSERY_COMMIT:?submit with the exact nursery commit}"
readonly EXPECTED_SCENE_RECEIPT_SHA256=beaeccc61e87e74636171d48b963d01563e107c906f166e8490b32e808f9d3ac
readonly EXPECTED_RESULT_RECEIPT_SHA256=747082c1bb42e15c786405ac4fdeaee95a85ab9e1eb6c8b3aab7f0bfc60f8e89
readonly EXPECTED_HANDOFF_SHA256=9436c6cb6a267d9352cb5a2785374ccde302c0394657cc29c23cb76f0dd69466
readonly SUPERSEDED_ROOT="${DURABLE_ROOT}/.superseded/${SLURM_JOB_ID}"
readonly STAGED_SCENE="${DURABLE_ROOT}/scene_packages/.room-staging-${SLURM_JOB_ID}"
readonly STAGED_RESULT="${DURABLE_ROOT}/simulation_canaries/.room-staging-${SLURM_JOB_ID}"
readonly COMPACT_BUNDLE="${DURABLE_ROOT}/compact_records/scene_bundle_${ACTIVITY_ID}_${SLURM_JOB_ID}.json"
readonly COMPACT_RECEIPT="${DURABLE_ROOT}/compact_records/fresh_scene_canary_${SLURM_JOB_ID}.json"
readonly COMPACT_GENERATION="${DURABLE_ROOT}/compact_records/fresh_embodiedgen_room_repair_${SLURM_JOB_ID}.json"
readonly MODEL_MANIFEST="${DURABLE_ROOT}/manifests/room_scene3d_models_${SLURM_JOB_ID}.sha256"
readonly MODEL_RECEIPT="${DURABLE_ROOT}/compact_records/room_scene3d_models_${SLURM_JOB_ID}.json"
readonly CHECKPOINT_ROOT="${DURABLE_ROOT}/checkpoints/room_scene3d"
readonly PANO_MODEL_ROOT="${CHECKPOINT_ROOT}/pano-${PANO_MODEL_REVISION}"
readonly BIG_LAMA_ROOT="${CHECKPOINT_ROOT}/big-lama-${BIG_LAMA_REVISION}"
readonly SD2_ROOT="${CHECKPOINT_ROOT}/sd2-inpaint-${SD2_INPAINT_REVISION}"
readonly REALESRGAN_ROOT="${CHECKPOINT_ROOT}/roboassetgen-${REALESRGAN_REVISION}"
readonly OMNIDATA_ROOT="${SCRATCH_ROOT}/source/omnidata_models-${OMNIDATA_COMMIT}"
readonly ROOM_PROMPT='Simple empty dining room interior with warm neutral walls, a level wood floor, visible corners and four walls, and no furniture or loose objects; designed as context for the activity: Pick up the red mug from the table, bring it to your mouth, and place it back on the table.'
readonly ROOM_SEED=2026081501

install -d -m 700 \
    "${RUN_ROOT}" \
    "${SCRATCH_ROOT}/caches/huggingface" \
    "${SCRATCH_ROOT}/caches/torch_room" \
    "${SCRATCH_ROOT}/logs" \
    "${DURABLE_ROOT}/compact_records" \
    "${DURABLE_ROOT}/manifests" \
    "${CHECKPOINT_ROOT}"

for path in "${RUN_ROOT}" "${STAGED_SCENE}" "${STAGED_RESULT}" \
    "${SUPERSEDED_ROOT}" "${COMPACT_BUNDLE}" "${COMPACT_RECEIPT}" \
    "${COMPACT_GENERATION}" "${MODEL_MANIFEST}" "${MODEL_RECEIPT}"; do
    if [[ "${path}" != "${RUN_ROOT}" && -e "${path}" ]]; then
        echo "Refusing pre-existing job-bound path: ${path}" >&2
        exit 1
    fi
done
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
    echo "Pinned EmbodiedGen environment is missing" >&2
    exit 1
fi
if [[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" != "${EMBODIEDGEN_COMMIT}" ]]; then
    echo "EmbodiedGen source commit mismatch" >&2
    exit 1
fi
if [[ "$(git -C "${NURSERY_ROOT}" rev-parse HEAD)" != "${NURSERY_COMMIT}" ]]; then
    echo "Nursery source commit mismatch" >&2
    exit 1
fi
git -C "${NURSERY_ROOT}" diff --quiet
git -C "${NURSERY_ROOT}" diff --cached --quiet
for bound in \
    "${CURRENT_SCENE}/generation_receipt.json:${EXPECTED_SCENE_RECEIPT_SHA256}" \
    "${CURRENT_RESULT}/receipt.json:${EXPECTED_RESULT_RECEIPT_SHA256}" \
    "${CURRENT_HANDOFF}:${EXPECTED_HANDOFF_SHA256}"; do
    file=${bound%%:*}
    expected=${bound##*:}
    if [[ ! -f "${file}" || -L "${file}" || \
          "$(sha256sum "${file}" | cut -d' ' -f1)" != "${expected}" ]]; then
        echo "Current incomplete predecessor is missing or changed: ${file}" >&2
        exit 1
    fi
done

git -C "${SOURCE_ROOT}" submodule update --init --recursive thirdparty/pano2room
if [[ "$(git -C "${SOURCE_ROOT}/thirdparty/pano2room" rev-parse HEAD)" != "${PANO2ROOM_COMMIT}" ]]; then
    echo "Pano2Room submodule commit mismatch" >&2
    exit 1
fi

export HF_HOME="${SCRATCH_ROOT}/caches/huggingface"
export TORCH_HOME="${SCRATCH_ROOT}/caches/torch_room"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/caches/pip"
source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
set +u
conda activate "${ENV_ROOT}"
set -u

python -m pip install --disable-pip-version-check --no-deps \
    "txt2panoimg@git+https://github.com/HochCC/SD-T2I-360PanoImage@${PANO_SOURCE_COMMIT}"
python -m pip install --disable-pip-version-check \
    kornia==0.7.0 \
    h5py==3.10.0 \
    albumentations==0.5.2 \
    webdataset==0.2.57 \
    icecream==2.1.3 \
    pyequilib==0.3.0 \
    lpips==0.1.4 \
    pymeshlab==2022.2.post4 \
    py360convert==1.0.4

if [[ ! -d "${OMNIDATA_ROOT}" ]]; then
    git clone https://github.com/alexsax/omnidata_models "${OMNIDATA_ROOT}.staging-${SLURM_JOB_ID}"
    git -C "${OMNIDATA_ROOT}.staging-${SLURM_JOB_ID}" checkout "${OMNIDATA_COMMIT}"
    mv "${OMNIDATA_ROOT}.staging-${SLURM_JOB_ID}" "${OMNIDATA_ROOT}"
fi
if [[ "$(git -C "${OMNIDATA_ROOT}" rev-parse HEAD)" != "${OMNIDATA_COMMIT}" ]]; then
    echo "Omnidata source commit mismatch" >&2
    exit 1
fi

export IMIM_PANO_MODEL_ROOT="${PANO_MODEL_ROOT}"
export IMIM_BIG_LAMA_ROOT="${BIG_LAMA_ROOT}"
export IMIM_SD2_ROOT="${SD2_ROOT}"
export IMIM_REALESRGAN_ROOT="${REALESRGAN_ROOT}"
python - <<'PY'
import os
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

specs = [
    (
        "archerfmy0831/sd-t2i-360panoimage",
        "d20770f5ec0000a2aaa579a1209d11dc92ddb08a",
        Path(os.environ["IMIM_PANO_MODEL_ROOT"]),
        None,
    ),
    (
        "smartywu/big-lama",
        "05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9",
        Path(os.environ["IMIM_BIG_LAMA_ROOT"]),
        ["big-lama.zip"],
    ),
    (
        "sd2-community/stable-diffusion-2-inpainting",
        "5f74973cbb64c8568780732c17f43eb269d63a0d",
        Path(os.environ["IMIM_SD2_ROOT"]),
        None,
    ),
    (
        "xinjjj/RoboAssetGen",
        "a64fcdebeea17287d830736cd0853df1093b97ab",
        Path(os.environ["IMIM_REALESRGAN_ROOT"]),
        ["super_resolution/*"],
    ),
]
for repo, revision, root, allow_patterns in specs:
    revision_file = root / ".nursery_revision"
    if root.exists():
        if not revision_file.is_file() or revision_file.read_text().strip() != revision:
            raise RuntimeError(f"existing checkpoint root has wrong revision: {root}")
        continue
    staging = root.with_name(root.name + ".staging-" + os.environ["SLURM_JOB_ID"])
    if staging.exists():
        raise RuntimeError(f"stale checkpoint staging root: {staging}")
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=staging,
        allow_patterns=allow_patterns,
    )
    revision_file = staging / ".nursery_revision"
    revision_file.write_text(revision + "\n", encoding="utf-8")
    staging.rename(root)
PY

for required in \
    "${PANO_MODEL_ROOT}/.nursery_revision" \
    "${BIG_LAMA_ROOT}/big-lama.zip" \
    "${SD2_ROOT}/model_index.json" \
    "${REALESRGAN_ROOT}/super_resolution/RealESRGAN_x4plus.pth"; do
    if [[ ! -s "${required}" || -L "${required}" ]]; then
        echo "Pinned room checkpoint file is missing or unsafe: ${required}" >&2
        exit 1
    fi
done

export IMIM_MODEL_MANIFEST="${MODEL_MANIFEST}"
export IMIM_MODEL_RECEIPT="${MODEL_RECEIPT}"
export IMIM_ENV_ROOT="${ENV_ROOT}"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess

roots = {
    "pano": Path(os.environ["IMIM_PANO_MODEL_ROOT"]),
    "big_lama": Path(os.environ["IMIM_BIG_LAMA_ROOT"]),
    "sd2_inpaint": Path(os.environ["IMIM_SD2_ROOT"]),
    "realesrgan": Path(os.environ["IMIM_REALESRGAN_ROOT"]),
}
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

lines = []
summary = {}
for name, root in roots.items():
    records = []
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = sha256_file(path)
        size = path.stat().st_size
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "bytes": size, "sha256": digest})
        lines.append(f"{digest}  {name}/{relative}")
        total += size
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    summary[name] = {
        "root": str(root),
        "file_count": len(records),
        "total_bytes": total,
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }
manifest = Path(os.environ["IMIM_MODEL_MANIFEST"])
manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
freeze = subprocess.run(
    [str(Path(os.environ["IMIM_ENV_ROOT"]) / "bin" / "python"), "-m", "pip", "freeze"],
    check=True,
    capture_output=True,
).stdout
receipt = {
    "schema": "InteractMoveRoomModelReceipt",
    "schema_version": 1,
    "status": "passed",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "models": summary,
    "full_manifest": {
        "path": str(manifest),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    },
    "environment_freeze_sha256": hashlib.sha256(freeze).hexdigest(),
}
Path(os.environ["IMIM_MODEL_RECEIPT"]).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

python - <<'PY'
import sys, types
import torchvision.transforms.functional as TF
module = types.ModuleType("torchvision.transforms.functional_tensor")
module.rgb_to_grayscale = TF.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = module
for name in (
    "txt2panoimg", "equilib", "kornia", "albumentations", "webdataset",
    "icecream", "lpips", "pymeshlab", "basicsr", "realesrgan",
):
    __import__(name)
print("room dependency imports passed")
PY

cp -a "${CURRENT_SCENE}" "${CANDIDATE_SCENE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${NURSERY_ROOT}:${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${SOURCE_ROOT}"
python "${ROOM_RUNNER}" \
    --candidate-scene "${CANDIDATE_SCENE}" \
    --generation-work-dir "${GENERATION_WORK}" \
    --embodiedgen-source "${SOURCE_ROOT}" \
    --pano-model "${PANO_MODEL_ROOT}" \
    --big-lama-zip "${BIG_LAMA_ROOT}/big-lama.zip" \
    --sd2-inpaint-model "${SD2_ROOT}" \
    --realesrgan-model "${REALESRGAN_ROOT}/super_resolution/RealESRGAN_x4plus.pth" \
    --omnidata-source "${OMNIDATA_ROOT}" \
    --room-prompt "${ROOM_PROMPT}" \
    --seed "${ROOM_SEED}" \
    --job-id "${SLURM_JOB_ID}"

export IMIM_CANDIDATE_SCENE="${CANDIDATE_SCENE}"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

scene = Path(os.environ["IMIM_CANDIDATE_SCENE"])
path = scene / "generation_receipt.json"
receipt = json.loads(path.read_text(encoding="utf-8"))
model_receipt = Path(os.environ["IMIM_MODEL_RECEIPT"])
receipt["models"]["room_scene3d"]["model_receipt"] = {
    "path": str(model_receipt),
    "sha256": hashlib.sha256(model_receipt.read_bytes()).hexdigest(),
}
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cd "${NURSERY_ROOT}"
python -m babyworld_lite.interactmove_intermimic compile-scene-bundle \
    "${CANDIDATE_SCENE}/metadata/activity.json" \
    --request "${CANDIDATE_SCENE}/metadata/embodiedgen_request.json" \
    --scene-root "${CANDIDATE_SCENE}" \
    --layout layout.json \
    --protocol "${CANDIDATE_SCENE}/metadata/protocol.json" \
    --output "${BUNDLE}"
python -m babyworld_lite.interactmove_intermimic validate-scene-bundle \
    "${BUNDLE}" --print-digest

python "${CANARY_RUNNER}" \
    --layout "${CANDIDATE_SCENE}/layout.json" \
    --generation-receipt "${CANDIDATE_SCENE}/generation_receipt.json" \
    --scene-bundle "${BUNDLE}" \
    --output-dir "${RESULT}" \
    --duration-s 10 \
    --sim-hz 200 \
    --video-fps 30 \
    --width 512 \
    --height 512

export IMIM_BUNDLE="${BUNDLE}"
export IMIM_RESULT="${RESULT}"
export IMIM_HANDOFF="${HANDOFF}"
export IMIM_CANONICAL_SCENE="${CURRENT_SCENE}"
export IMIM_COMPACT_BUNDLE="${COMPACT_BUNDLE}"
export IMIM_COMPACT_RECEIPT="${COMPACT_RECEIPT}"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical_sha(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()

scene = Path(os.environ["IMIM_CANDIDATE_SCENE"])
bundle_path = Path(os.environ["IMIM_BUNDLE"])
result = Path(os.environ["IMIM_RESULT"])
handoff_path = Path(os.environ["IMIM_HANDOFF"])
bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
receipt = json.loads((result / "receipt.json").read_text(encoding="utf-8"))
generation = json.loads((scene / "generation_receipt.json").read_text(encoding="utf-8"))
activity = json.loads((scene / "metadata/activity.json").read_text(encoding="utf-8"))
request = json.loads((scene / "metadata/embodiedgen_request.json").read_text(encoding="utf-8"))
required = (
    "visual_background_ready", "room_collision_ready", "complete_scene_ready",
    "reference_mesh_ready", "interactmove_scene_input_ready",
)
if receipt["status"] != "passed" or any(bundle["capabilities"][key] is not True for key in required):
    raise RuntimeError("repaired scene did not pass every complete-scene gate")
inventory = []
for path in sorted(item for item in scene.rglob("*") if item.is_file()):
    inventory.append({
        "path": path.relative_to(scene).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    })
target = next(item for item in bundle["instances"] if item["role"] == "target")
support = next(item for item in bundle["instances"] if item["role"] == "support")
canonical_scene = Path(os.environ["IMIM_CANONICAL_SCENE"])
job_id = os.environ["SLURM_JOB_ID"]
handoff = {
    "schema": "NurserySharedEmbodiedGenSceneHandoff",
    "schema_version": 1,
    "status": "passed",
    "canonical_scene_root": str(canonical_scene),
    "activity": {
        "activity_id": activity["activity_id"],
        "activity_spec_sha256": request["activity_spec_sha256"],
        "prompt": activity["prompt"],
        "seeds": activity["seeds"],
        **activity["timing"],
    },
    "generation": {
        "embodiedgen_commit": "9b333554254af196bace88c1a171a3bf047fa09c",
        "image_to_3d_backend": "SAM3D",
        "sam3d_source_commit": "01417d16fb5cc762a60f370c1bf7f59d603ddfaf",
        "sam3d_checkpoint_revision": "2e73555018d2741ccd486e56c24fac41155a1dc6",
        "accepted_assets_regenerated": False,
        "preserved_object_generation_job_id": "328538",
        "room_generation_job_id": job_id,
        "room_prompt": generation["room_completion"]["room_prompt"],
        "room_seed": generation["room_completion"]["seed"],
        "room_model_pins": generation["models"]["room_scene3d"],
        "layout_path": str(canonical_scene / "layout.json"),
        "layout_sha256": generation["layout"]["sha256"],
        "receipt_path": str(canonical_scene / "generation_receipt.json"),
        "receipt_sha256": sha(scene / "generation_receipt.json"),
        "robot_actor_inserted": False,
        "upstream_sim_cli_invoked": False,
        "native_robot_metadata_only": True,
    },
    "inventory": {
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(item["bytes"] for item in inventory),
        "canonical_sha256": canonical_sha(inventory),
    },
    "resolved_instances": {
        "target": {
            "source_node_key": target["source_node_key"],
            "instance_id": target["instance_id"],
            "category": target["category"],
        },
        "support": {
            "source_node_key": support["source_node_key"],
            "instance_id": support["instance_id"],
            "category": support["category"],
        },
    },
    "coordinate_convention": bundle["conventions"],
    "scene_bundle": {
        "path": os.environ["IMIM_COMPACT_BUNDLE"],
        "file_sha256": sha(bundle_path),
        "semantic_sha256": canonical_sha(bundle),
        "bundle_id": bundle["bundle_id"],
        "slurm_job_id": job_id,
        "validation_status": "passed",
        "capabilities": bundle["capabilities"],
    },
    "scene_only_simulation": {
        "status": receipt["status"],
        "slurm_job_id": job_id,
        "duration_s": receipt["timing"]["duration_s"],
        "video_fps": receipt["timing"]["video_fps"],
        "video_frames": receipt["timing"]["video_frames"],
        "video_path": str(Path("/work/dal503972/interactmove_intermimic/simulation_canaries") / activity["activity_id"] / "scene_canary.mp4"),
        "video_sha256": receipt["outputs"]["video"]["sha256"],
        "receipt_path": os.environ["IMIM_COMPACT_RECEIPT"],
        "receipt_sha256": sha(result / "receipt.json"),
        "settling_check": receipt["settling_check"],
    },
    "completeness": {
        "visual_background": "passed_raster_room_mesh_rendered",
        "physics_collision": "passed_explicit_floor_and_walls_imported",
        "interactmove_pointcloud": "passed_validated_finite_metric_reference_mesh",
        "interactmove_motion_generation": "not_run",
        "intermimic_execution": "not_run",
    },
    "scientific_boundaries": {
        "fresh_embodiedgen_scene_generation": "passed_objects_plus_room_background",
        "scene_bundle_validation": "passed",
        "scene_only_sapien_physics_render": "passed_canary_only",
        "interactmove_motion_generation": "not_run",
        "intermimic_execution": "not_run",
    },
}
handoff_path.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cp -a "${CANDIDATE_SCENE}" "${STAGED_SCENE}"
cp -a "${RESULT}" "${STAGED_RESULT}"
if [[ "$(sha256sum "${STAGED_SCENE}/generation_receipt.json" | cut -d' ' -f1)" != \
      "$(sha256sum "${CANDIDATE_SCENE}/generation_receipt.json" | cut -d' ' -f1)" ]]; then
    echo "Durable scene staging verification failed" >&2
    exit 1
fi
if [[ "$(sha256sum "${STAGED_RESULT}/receipt.json" | cut -d' ' -f1)" != \
      "$(sha256sum "${RESULT}/receipt.json" | cut -d' ' -f1)" ]]; then
    echo "Durable result staging verification failed" >&2
    exit 1
fi

install -d -m 700 "${SUPERSEDED_ROOT}"
mv "${CURRENT_SCENE}" "${SUPERSEDED_ROOT}/incomplete_scene"
mv "${CURRENT_RESULT}" "${SUPERSEDED_ROOT}/incomplete_scene_canary"
mv "${CURRENT_HANDOFF}" "${SUPERSEDED_ROOT}/shared_scene_incomplete.json"
mv "${STAGED_SCENE}" "${CURRENT_SCENE}"
mv "${STAGED_RESULT}" "${CURRENT_RESULT}"
cp "${BUNDLE}" "${COMPACT_BUNDLE}.tmp"
mv "${COMPACT_BUNDLE}.tmp" "${COMPACT_BUNDLE}"
cp "${CURRENT_RESULT}/receipt.json" "${COMPACT_RECEIPT}.tmp"
mv "${COMPACT_RECEIPT}.tmp" "${COMPACT_RECEIPT}"
cp "${CURRENT_SCENE}/generation_receipt.json" "${COMPACT_GENERATION}.tmp"
mv "${COMPACT_GENERATION}.tmp" "${COMPACT_GENERATION}"
cp "${HANDOFF}" "${CURRENT_HANDOFF}.tmp"
mv "${CURRENT_HANDOFF}.tmp" "${CURRENT_HANDOFF}"

echo "canonical_scene=${CURRENT_SCENE}"
echo "scene_bundle=${COMPACT_BUNDLE}"
echo "simulation_result=${CURRENT_RESULT}"
echo "shared_handoff=${CURRENT_HANDOFF}"
echo "superseded_incomplete=${SUPERSEDED_ROOT}"
