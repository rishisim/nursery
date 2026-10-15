"""Summarize metrics.json files: python scripts/summarize.py 'outputs/runs/clean_video_only/none_s*/metrics.json' ..."""
import json,glob,sys,statistics as st
for pat in sys.argv[1:]:
    rows=[json.load(open(f)) for f in sorted(glob.glob(pat))]
    if not rows: print(pat,'none'); continue
    t=[m['reported_test_acc'] for m in rows]; k=[m['reported_konkle_acc'] for m in rows]
    g=lambda key: st.mean(m['groups'][key]['acc'] for m in rows)
    sd=lambda x: st.stdev(x) if len(x)>1 else 0
    print(f"{pat.split('/')[-3]}/{pat.split('/')[-2]} n={len(rows)} test {st.mean(t):.1f}±{sd(t):.1f} (held {g('test_held'):.1f} seen {g('test_seen_only'):.1f} extras {g('test_extras'):.1f}) konkle {st.mean(k):.1f}±{sd(k):.1f} (held {g('konkle_held'):.1f} seen {g('konkle_seen_only'):.1f}) ep {[m['selected_epoch'] for m in rows]}")
