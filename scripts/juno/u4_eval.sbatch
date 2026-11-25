#!/bin/bash
#SBATCH -J u4-eval
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=176G
#SBATCH -t 12:00:00
#SBATCH -o outputs/runs/u4-%j.out
set -euo pipefail

# Submit from /work/dal503972/nursery_u1. Inputs are U3's simulated GRAB caches.
: "${SLURM_JOB_ID:?Submit this job with sbatch}"
cd -- /work/dal503972/nursery_u1
export TRITON_CACHE_DIR=/work/dal503972/.cache/triton
export XDG_CACHE_HOME=/work/dal503972/.cache
export TORCHINDUCTOR_CACHE_DIR=/work/dal503972/.cache/inductor
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

root=outputs/runs/unified_u3_curves
out=outputs/runs/unified_u4_eval
arms=(V+G+B V+G+B_shuffled_body V+G V CLIP-Global BabyLLaVA-Nursery)
seeds=(0 1 2 3 4)
checkpoints=()
embedding_args=()
only=${ONLY:-}
case "$only" in
    ""|prompted) ;;
    *) echo "Unsupported ONLY=$only" >&2; exit 2 ;;
esac
if [[ ${DRY_RUN:-0} == 1 ]]; then
    out=outputs/runs/unified_u4_eval_dryrun
    checkpoints=("$root/V+G+B/n630/seed0/best.pt")
else
    for arm in "${arms[@]}"; do
        for seed in "${seeds[@]}"; do
            checkpoints+=("$root/$arm/n630/seed$seed/best.pt")
        done
    done
    if [[ -z $only ]]; then
        # All five n630 seeds already export embeddings in the scoring pass.
        # Use seed 0 at smaller counts to bound repeated sensory data setup.
        for arm in V+G+B V CLIP-Global; do
            for n in 63 158 315; do
                embedding_args+=(--embedding-checkpoint "$root/$arm/n$n/seed0/best.pt")
            done
        done
    fi
fi

args=(--device cuda --out-dir "$out")
if [[ $only == prompted ]]; then
    args+=(--only prompted)
    if [[ -n ${TEXT_ONLY_SUMMARY:-} ]]; then
        args+=(--text-only-summary "$TEXT_ONLY_SUMMARY")
    fi
else
    args+=(--text-only-summary outputs/runs/unified_u4_text_only/u4_eval/summary.json)
fi

# Reuse U3's cache copy, mount detection, fallback and cleanup in the evaluator process.
exec .external/cue-env/bin/python -B -u - "${args[@]}" \
    ${embedding_args[@]+"${embedding_args[@]}"} "${checkpoints[@]}" <<'PY'
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from scripts.juno import stage_caches as staging

for arg in sys.argv[1:]:
    if arg.endswith("/best.pt"):
        staging.readable_file(Path(arg))

from nursery.unified import evaluate, u4_eval

study = staging.STUDY.resolve(strict=True)
protocol = json.loads((study / "protocol.json").read_text())
cache_dirs = (
    (Path(protocol["inputs"]) / "train_cache", Path("inputs/train_cache")),
    (Path(protocol["gaze"]["closeup_cache"]), Path("closeup_cache")),
)
caches = []
for source, relative in cache_dirs:
    for name in ("emb.f16.npy", "index.parquet"):
        path = (source / name).resolve(strict=True)
        caches.append((path, relative / name, staging.readable_file(path)))
required = (sum(size for _, _, size in caches) * 11 + 9) // 10

signal.signal(signal.SIGINT, staging.stop)
signal.signal(signal.SIGTERM, staging.stop)
with ExitStack() as cleanup:
    try:
        free = staging.local_free()
        if free < required:
            raise OSError(f"{staging.LOCAL} has {free:,} bytes free; need {required:,}")
        destination = cleanup.enter_context(tempfile.TemporaryDirectory(
            prefix=f"nursery-u4-{os.environ['SLURM_JOB_ID']}-", dir=staging.LOCAL,
        ))
        evaluate.STUDY = staging.stage(study, protocol, caches, Path(destination))
    except (OSError, subprocess.CalledProcessError) as error:
        cleanup.close()
        print(f"WARNING: staging failed ({error}); using original /work cache paths.", file=sys.stderr)
    u4_eval.main()
PY
