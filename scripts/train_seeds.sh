#!/bin/bash
# Train several seeds, 2 at a time. Skips seeds that already have metrics.json.
# Usage: scripts/train_seeds.sh <run_dir> <name> "<extra train args>" <seed>...
# Example: scripts/train_seeds.sh outputs/runs/clean_video_only none "--condition none" 0 1 2 3 4   (env EPOCHS=40)
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$REPO/.external/cue-env/bin/python
RUN=$(cd "$1" && pwd); name=$2; args=$3; shift 3
cd "$REPO"; export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
run() { [ -f "$RUN/${name}_s$1/metrics.json" ] || "$PY" -m nursery.learner train --run "$RUN" --name "$name" --seed "$1" --epochs "${EPOCHS:-40}" $args > "$RUN/${name}_s$1.log" 2>&1; }
while [ $# -gt 0 ]; do
  run "$1" &
  [ $# -gt 1 ] && { run "$2" & shift; }
  shift; wait
done
