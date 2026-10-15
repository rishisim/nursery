#!/bin/bash
# Speech -> embed -> video-only train -> summary.
# Usage: scripts/pipeline.sh <episode_root> <output_dir> [seed]   (env EPOCHS overrides the config epochs)
set -euo pipefail
EP=$(cd "$1" && pwd); mkdir -p "$2"; OUT=$(cd "$2" && pwd); SEED=${3:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$REPO/.external/cue-env/bin/python
cd "$REPO"; LOG=$OUT/pipeline.log
exec > >(while IFS= read -r l; do printf '%s %s\n' "$(date '+%F %T')" "$l"; done | tee -a "$LOG") 2>&1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
step() { echo "== $* =="; }
t0=$(date +%s)
[ -f "$OUT/manifest.parquet" ] || { step speech; "$PY" -m nursery.speech transcripts --episodes "$EP" --output "$OUT" --seed "$SEED"; }
step embed; t1=$(date +%s)
"$PY" -m nursery.learner embed --episodes "$EP" --output "$OUT"
t2=$(date +%s); echo "EMBED_SECONDS $((t2-t1))"
step train
"$PY" -m nursery.learner train --run "$OUT" --condition none --seed "$SEED" ${EPOCHS:+--epochs "$EPOCHS"}
t3=$(date +%s); echo "TRAIN_SECONDS $((t3-t2))"
step summary
"$PY" - "$OUT" "$SEED" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
out, seed = Path(sys.argv[1]), sys.argv[2]
m = json.loads((out / f"none_s{seed}" / "metrics.json").read_text())
h = m["history"]
print(f"pairs {m['n_pairs']}  vocab {m['vocab']}  device {m['device']}  selected_ep {m['selected_epoch']} (by dev)")
print("probe counts:", {s: f"{c['n_scored']}/{c['n_total']} scored, {c['n_oov']} oov" for s, c in m["probe_counts"].items()})
print(f"test acc (selected) {m['reported_test_acc']:.1f}%  best {m['best_test_acc']:.1f}%  final {m['final_test_acc']:.1f}%")
if "reported_konkle_acc" in m:
    print(f"Konkle test-60 macro acc (selected) {m['reported_konkle_acc']:.1f}%  best {m['best_konkle_acc']:.1f}%  (4AFC, chance 25%, OOV at chance)")
print("group accuracy at selected epoch (% / n words):")
for k, g in m["groups"].items():
    print(f"  {k:18s} {g['acc']:6.1f}  n={g['words']}")
print("per-epoch  ep  loss  dev  test  konkle")
for r in h:
    print(f"  {r['ep']:3d} {r['train_loss']:.4f} {r['dev']:.3f} {r['acc']:.3f} {r.get('konkle', float('nan')):.3f}")
flags = []
for r in h:
    for k in ("dev", "acc", "konkle"):
        if r.get(k, 0) > 0.95:
            flags.append(f"{k} {r[k]:.3f} at ep {r['ep']}")
if h[0]["train_loss"] < 0.05:
    flags.append(f"epoch-0 train loss {h[0]['train_loss']:.4f} (train acc ~1.0 suspected)")
print("SATURATION:", "; ".join(flags) if flags else "none (all probe/Konkle acc <= 0.95, epoch-0 loss not ~0)")
fr = sum(len(pd.read_parquet(out / f"{n}_frames.parquet")) for n in ("train", "probes", "konkle"))
print(f"frames embedded: {fr}")
PY
echo "TOTAL_SECONDS $(( $(date +%s) - t0 ))  log: $LOG"
