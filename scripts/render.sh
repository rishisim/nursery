#!/bin/bash
# Resumable full render: probes + one session per schedule index. Re-run to resume.
# Usage: scripts/render.sh <work_dir> [config.json]   (merged dataset lands in <work_dir>/episodes)
# Env: WORKERS=10 SEED=7 SESSIONS=107 (per subject) ROOMS=outputs/embodiedgen_scene/room_bank
# Internal: scripts/render.sh --one <work_dir> <config> <index>
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$REPO/.external/egocentric-env/bin/python
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
abspath() { (mkdir -p "$(dirname "$1")" && cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); }
SEED=${SEED:-7}; SESSIONS=${SESSIONS:-107}; ROOMS=${ROOMS:-$REPO/outputs/embodiedgen_scene/room_bank}
if [ "${1:-}" != "--one" ]; then
  work=$(abspath "${1:?usage: render.sh <work_dir> [config.json]}"); config=$(abspath "${2:-$REPO/nursery/simulator/config.json}")
fi
cd "$REPO" || exit 1

if [ "${1:-}" = "--one" ]; then
  work=$2; config=$3; idx=$4
  subject=s$((idx / SESSIONS + 1)); vid=$(printf "%s_%04d" "$subject" $((idx % SESSIONS)))
  dir=$work/sessions/$vid
  [ -e "$dir/DONE" ] || [ -e "$dir/FAILED" ] && exit 0
  rm -rf "$dir"
  PYTHONPATH=. "$PY" -m nursery.simulator build --config "$config" --rooms "$ROOMS" --output "$dir" \
    --sessions-per-subject "$SESSIONS" --seed "$SEED" --session-index "$idx" --workers 1 > "$work/logs/$vid.log" 2>&1
  code=$?
  if [ -f "$dir/frames.parquet" ]; then echo $code > "$dir/DONE"; else mkdir -p "$dir"; echo $code > "$dir/FAILED"; fi
  exit 0
fi

mkdir -p "$work/logs" "$work/sessions"
date +%s > "$work/start_time"
if [ ! -e "$work/probes_run/DONE" ]; then
  rm -rf "$work/probes_run"
  PYTHONPATH=. "$PY" -m nursery.simulator build --config "$config" --rooms "$ROOMS" --output "$work/probes_run" \
    --sessions-per-subject 1 --seed "$SEED" --subjects s1 --workers 3 > "$work/logs/probes_run.log" 2>&1
  [ -f "$work/probes_run/probes.parquet" ] && touch "$work/probes_run/DONE"
fi
total=$(( SESSIONS * $("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['subjects']))" "$config") ))
export SEED SESSIONS ROOMS
seq 0 $((total - 1)) | xargs -P "${WORKERS:-10}" -I{} bash "$SELF" --one "$work" "$config" {}
date > "$work/ALL_DONE"
"$REPO/.external/cue-env/bin/python" "$REPO/scripts/merge_render.py" "$work" "$SESSIONS"
