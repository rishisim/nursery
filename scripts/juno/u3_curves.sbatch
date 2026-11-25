#!/bin/bash
#SBATCH -J u3-curves
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -a 0-134%8
#SBATCH -o outputs/runs/u3-%A_%a.out
set -euo pipefail

# Submit from the Juno repo root.
cd -- "${SLURM_SUBMIT_DIR:-$PWD}"
export TRITON_CACHE_DIR=/work/dal503972/.cache/triton
export XDG_CACHE_HOME=/work/dal503972/.cache
export TORCHINDUCTOR_CACHE_DIR=/work/dal503972/.cache/inductor

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
    shift
fi
if [[ $# != 0 || ( -z ${SLURM_JOB_ID:-} && $dry_run != 1 ) ]]; then
    echo "Submit with sbatch, or use --dry-run on the login node." >&2
    exit 2
fi

task=${SLURM_ARRAY_TASK_ID:-0}
if [[ ! $task =~ ^(0|[1-9][0-9]{0,2})$ ]] || (( task > 134 )); then
    echo "Expected a U3 task index from 0 through 134." >&2
    exit 2
fi

# Tasks 0–124 use arm order below, then ascending clip count, then seed fastest.
# Tasks 125–134 use the shuffled-body control at 158 and 630 clips, seed fastest.
seed=$((task % 5))
if (( task < 125 )); then
    arms=(V+G+B V+G V CLIP-Global BabyLLaVA-Nursery)
    clip_counts=(63 158 315 473 630)
    arm=${arms[task / 25]}
    n_clips=${clip_counts[task / 5 % 5]}
    condition=$arm
    flags=()
else
    clip_counts=(158 630)
    arm=V+G+B_shuffled_body
    n_clips=${clip_counts[(task - 125) / 5]}
    condition=V+G+B
    flags=(--shuffle-body)
fi

out_dir="${RUN_ROOT:-outputs/runs/unified_u3_curves}/$arm/n${n_clips}/seed${seed}"
if [[ -f $out_dir/metrics.json ]]; then
    echo "Skipping completed run $out_dir"
    exit 0
fi
u2="$PWD/outputs/runs/unified_u2_data/frozen"
test -r "$u2/manifest.parquet"
train_args=(--condition "$condition" --n-clips "$n_clips" --seed "$seed" --epochs "${EPOCHS:-60}"
            --device cuda --u2 "$u2" --out-dir "$out_dir" ${flags[@]+"${flags[@]}"})
printf 'U3 arm %s:' "$arm"
printf ' %q' "${train_args[@]}"
printf '\n'

python=.external/cue-env/bin/python
if [[ $dry_run == 1 ]]; then
    train_args+=(--dry-run)
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

"$python" -B -u -m scripts.juno.stage_caches "${train_args[@]}" &
child=$!
status=0
wait "$child" || status=$?
child=
exit "$status"
