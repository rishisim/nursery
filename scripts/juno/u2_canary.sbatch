#!/bin/bash
#SBATCH -J u2-canary
#SBATCH -p h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -a 0-2
#SBATCH -o outputs/runs/u2-%A_%a.out
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

case ${SLURM_ARRAY_TASK_ID:-0} in
    0) arm=V+G+B; condition=V+G+B; flags=() ;;
    1) arm=V+G+B_bow_only; condition=V+G+B; flags=(--bow-only) ;;
    2) arm=text_only; condition="Text-Only LM"; flags=() ;;
    *) echo "Expected a U2 arm index from 0 through 2." >&2; exit 2 ;;
esac

out_dir="${RUN_ROOT:-outputs/runs/unified_u2_canary}/$arm/seed0"
if [[ -f $out_dir/metrics.json ]]; then
    echo "Skipping completed run $out_dir"
    exit 0
fi
u2="$PWD/outputs/runs/unified_u2_data/frozen"
test -r "$u2/manifest.parquet"
train_args=(--condition "$condition" --n-clips 630 --seed 0 --epochs 40
            --device cuda --u2 "$u2" --out-dir "$out_dir" "${flags[@]}")
printf 'U2 arm %s:' "$arm"
printf ' %q' "${train_args[@]}"
printf '\n'

python=.external/cue-env/bin/python
if [[ $arm == text_only ]]; then
    if [[ $dry_run == 1 ]]; then
        test -r outputs/runs/step9_acquisition_curves/frozen/protocol.json
        "$python" -B -m nursery.unified.train --help >/dev/null
        echo "Dry run: text-only training uses the shared speech manifest without feature caches."
        exit 0
    fi
    exec "$python" -B -u -m nursery.unified.train "${train_args[@]}"
fi
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
