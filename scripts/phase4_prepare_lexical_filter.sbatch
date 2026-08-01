#!/bin/bash
#SBATCH --job-name=phase4-filter-public
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_DISABLED=true PYTHONNOUSERSITE=1
tmp="/tmp/phase4-filter-public-${SLURM_JOB_ID}"
mkdir -p "$tmp/pydeps"
trap 'find "$tmp" -mindepth 1 -delete 2>/dev/null || true; rmdir "$tmp" 2>/dev/null || true' EXIT
tar -C "$tmp/pydeps" -xf "${PHASE4_PUBLIC_ROOT}/models/phase4-language-pydeps.tar"
export PYTHONPATH="$tmp/pydeps"
export HF_HOME="${PHASE4_PUBLIC_ROOT}/models/pe-hf-home"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
export SINGULARITYENV_HF_HOME="$HF_HOME"
singularity exec --bind "${PHASE4_PUBLIC_ROOT}:${PHASE4_PUBLIC_ROOT}" \
  "${PHASE4_PUBLIC_ROOT}/models/pytorch-2.8.0-cu126.sif" python - <<'PY'
from huggingface_hub import hf_hub_download
import hashlib, json, os
from pathlib import Path

root = Path(os.environ["PHASE4_PUBLIC_ROOT"])
revision = "bafb0f76541d399057e980a25947f67acec76575"
path = Path(hf_hub_download("facebook/PE-Core-L14-336", "PE-Core-L14-336.pt", revision=revision))
readme = Path(hf_hub_download("facebook/PE-Core-L14-336", "README.md", revision=revision))
h = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        h.update(block)
os.environ["HF_HUB_OFFLINE"] = "1"
offline = Path(hf_hub_download("facebook/PE-Core-L14-336", "PE-Core-L14-336.pt", revision=revision, local_files_only=True))
record = {"status":"PASS","repository":"facebook/PE-Core-L14-336","revision":revision,"license":"Apache-2.0","weights_sha256":h.hexdigest(),"bytes":path.stat().st_size,"offline_reload":offline==path,"readme_present":readme.is_file()}
target = root / "models/pe-filter-manifest.json"
target.write_text(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n")
target.chmod(0o600)
print(json.dumps({"status":"PASS","offline_reload":record["offline_reload"],"bytes":record["bytes"]},sort_keys=True))
PY
