#!/bin/bash
#SBATCH -J u1-local
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -a 0-2
#SBATCH -o outputs/runs/u1-%A_%a.out
set -euo pipefail

# Submit from the Juno repo root. Array indices 0–2 match diag4.
# Select index 3 explicitly to include the video-only arm.
cd -- "${SLURM_SUBMIT_DIR:-$PWD}"
export TRITON_CACHE_DIR=/work/dal503972/.cache/triton
export XDG_CACHE_HOME=/work/dal503972/.cache
export TORCHINDUCTOR_CACHE_DIR=/work/dal503972/.cache/inductor

dry_run=${STAGE_DRY_RUN:-0}
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
    shift
fi
if [[ -z ${SLURM_JOB_ID:-} && $dry_run != 1 ]]; then
    echo "Run with sbatch, or use --dry-run on the login node." >&2
    exit 2
fi
if [[ $# != 1 || ! $1 =~ ^[[:alnum:]_][[:alnum:]_.-]*$ ]]; then
    echo "Usage: u1_array.sbatch [--dry-run] RUN_NAME" >&2
    exit 2
fi

arms=("V+G|V+G|" "V+G+B|V+G+B|" "V+G+B_shuffled_body|V+G+B|--shuffle-body" "V|V|")
task=${SLURM_ARRAY_TASK_ID:-0}
if [[ ! $task =~ ^[0-3]$ ]]; then
    echo "Expected a U1 array index from 0 through 3." >&2
    exit 2
fi
IFS="|" read -r arm condition flag <<< "${arms[$task]}"
train_args=(--condition "$condition" --n-clips "${N_CLIPS:-630}"
            --seed "${SEED:-0}" --epochs "${EPOCHS:-35}" --device cuda
            --out-dir "outputs/runs/$1/$arm/seed${SEED:-0}")
if [[ $dry_run == 1 ]]; then
    train_args+=(--dry-run)
fi
if [[ -n $flag ]]; then
    train_args+=("$flag")
fi

child=
stop_child() {
    if [[ -n $child ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" || true
    fi
}
trap stop_child EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

.external/cue-env/bin/python -B -u -m scripts.juno.stage_caches "${train_args[@]}" &
child=$!
status=0
wait "$child" || status=$?
child=
exit "$status"
