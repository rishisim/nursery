#!/usr/bin/env bash

set -euo pipefail

DURABLE_ROOT=/work/dal503972/hoidini_intermimic
SOURCE_ROOT=/scratch/juno/dal503972/hoidini_intermimic/source/nursery
TASK_CONTRACT="$SOURCE_ROOT/configs/hoidini_intermimic_pilot.json"
STAGE12_SOURCE="$SOURCE_ROOT/babyworld_lite/hoidini_intermimic/stage12.py"
HANDOFF=/work/dal503972/interactmove_intermimic/compact_records/shared_scene_red-mug-mouth-return.json
OUTPUT="$DURABLE_ROOT/compact_records/shared_scene_binding_red-mug-mouth-return.json"
EXPECTED_TASK_SHA256=3b91be82a86ca9a5490cf48c5ba94b48af91a7ca511ee0b05fcf6bed7d9c4396
EXPECTED_STAGE12_SHA256=fe3ef936dea1d2c8df673ee2100d297496d613b9d921e0ab7dde868a86faa3f5

test -f "$TASK_CONTRACT" || { echo "blocked=task_contract_missing"; exit 20; }
test -f "$STAGE12_SOURCE" || { echo "blocked=stage12_source_missing"; exit 23; }
test -f "$HANDOFF" || { echo "blocked=canonical_shared_scene_handoff_missing"; exit 21; }
test ! -e /work/dal503972/hoidini_intermimic/scene_packages/red-mug-mouth-return || {
  echo "blocked=unexpected_asset_copy_in_hoidini_namespace"
  exit 22
}
actual_task_sha256=$(sha256sum "$TASK_CONTRACT" | awk '{print $1}')
actual_stage12_sha256=$(sha256sum "$STAGE12_SOURCE" | awk '{print $1}')
test "$actual_task_sha256" = "$EXPECTED_TASK_SHA256" || {
  echo "blocked=task_contract_hash_mismatch"
  exit 24
}
test "$actual_stage12_sha256" = "$EXPECTED_STAGE12_SHA256" || {
  echo "blocked=stage12_source_hash_mismatch"
  exit 25
}
mkdir -p "$DURABLE_ROOT/compact_records"

PYTHONPATH="$SOURCE_ROOT" python3 "$SOURCE_ROOT/scripts/prepare_hoidini_intermimic_scene.py" \
  bind-shared-scene \
  --task-contract "$TASK_CONTRACT" \
  --handoff "$HANDOFF" \
  --output "$OUTPUT"

python3 - "$OUTPUT" "$EXPECTED_TASK_SHA256" "$EXPECTED_STAGE12_SHA256" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
expected_task_sha256 = sys.argv[2]
expected_stage12_sha256 = sys.argv[3]
record = json.load(open(path, encoding="utf-8"))
assert record["status"] == "passed"
assert record["task"]["contract_sha256"] == expected_task_sha256
assert record["implementation"]["stage12_sha256"] == expected_stage12_sha256
assert record["shared_scene"]["source_assets_copied"] is False
assert record["shared_scene"]["object_asset_backend"]["name"] == "SAM3D"
assert record["stage_status"]["stage2_shared_scene_binding_complete"] is True
assert record["stage_status"]["hoidini_motion_generated"] is False
assert record["stage_status"]["intermimic_executed"] is False
assert record["stage_status"]["video_rendered"] is False
print("binding_sha256=" + hashlib.sha256(open(path, "rb").read()).hexdigest())
PY
