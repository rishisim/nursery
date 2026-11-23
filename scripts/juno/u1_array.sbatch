#!/bin/bash
#SBATCH -J u1-local
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:30:00
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

arms=("V+G|V+G|" "V+G+B|V+G+B|" "V+G+B_shuffled_body|V+G+B|--shuffle-body" "V|V|"
      "V+G+B_lambda_gen_0|V+G+B|--lambda-gen 0.0" "V+G+B_no_mhsa|V+G+B|--no-mhsa")
# With SEEDS_PER_ARM=n, index i runs arm i/n with seed i%n (parity: --array=0-17%4, n=3).
task=${SLURM_ARRAY_TASK_ID:-0}
seed=${SEED:-0}
if [[ -n ${SEEDS_PER_ARM:-} ]]; then
    seed=$((task % SEEDS_PER_ARM))
    task=$((task / SEEDS_PER_ARM))
fi
if [[ ! $task =~ ^[0-5]$ ]]; then
    echo "Expected a U1 arm index from 0 through 5." >&2
    exit 2
fi
IFS="|" read -r arm condition flag <<< "${arms[$task]}"
out_dir="outputs/runs/$1/$arm/seed$seed"
if [[ -f $out_dir/metrics.json ]]; then
    echo "Skipping completed run $out_dir"
    exit 0
fi
train_args=(--condition "$condition" --n-clips "${N_CLIPS:-630}"
            --seed "$seed" --epochs "${EPOCHS:-35}" --device cuda
            --out-dir "$out_dir")
if [[ $dry_run == 1 ]]; then
    train_args+=(--dry-run)
fi
if [[ -n $flag ]]; then
    read -r -a flag_args <<< "$flag"
    train_args+=("${flag_args[@]}")
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
