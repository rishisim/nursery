"""Merge per-session render dirs plus the probe run into <work_dir>/episodes (hardlinks).

Usage: python scripts/merge_render.py <work_dir> [sessions_per_subject]
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

root = Path(sys.argv[1]); out = root / "episodes"
per_subject = int(sys.argv[2]) if len(sys.argv) > 2 else 107
out.mkdir(exist_ok=True)
for sub in ("frames", "body", "probes"):
    (out / sub).mkdir(exist_ok=True)

def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    os.link(src, dst)

def link_tree(src, dst):
    for path in src.rglob("*"):
        if path.is_file():
            link(path, dst / path.relative_to(src))

done = sorted(p.parent for p in (root / "sessions").glob("*/DONE"))
frames, sessions, statuses, receipts = [], [], {}, []
for d in done:
    vid = d.name
    manifest = json.loads((d / "manifest.json").read_text())
    statuses[vid] = (manifest["status"], manifest["statistics"]["held_visible_fraction"], manifest["statistics"]["mean_visible_categories"])
    frames.append(pq.read_table(d / "frames.parquet")); sessions.append(pq.read_table(d / "sessions.parquet"))
    link_tree(d / "frames" / vid, out / "frames" / vid)
    link(d / "body" / f"{vid}.npz", out / "body" / f"{vid}.npz")
    link(d / f"render_stats_{vid}.json", out / f"render_stats_{vid}.json")
    receipts.append(json.loads((d / f"render_stats_{vid}.json").read_text()))
probe = root / "probes_run"
link_tree(probe / "probes", out / "probes")
for path in probe.glob("render_stats_probe_*.json"):
    link(path, out / path.name); receipts.append(json.loads(path.read_text()))
probes = pq.read_table(probe / "probes.parquet")
frames = pa.concat_tables(frames); sessions = pa.concat_tables(sessions)
for name, t in (("frames", frames), ("sessions", sessions), ("probes", probes)):
    pq.write_table(t, out / f"{name}.parquet")
(out / "render_stats.json").write_text(json.dumps({
    "seconds": sum(r["seconds"] for r in receipts), "image_seconds": [s for r in receipts for s in r["image_seconds"]],
    "blender_version": receipts[0]["blender_version"], "jobs": receipts}))
rows = frames.to_pylist()
contacted = [r for r in rows if r["held_category"] is not None]
vis = lambda r: [o for o in json.loads(r["objects"]) if o["visible_px"] > 0]
held = sum(any(o["category"] == r["held_category"] for o in vis(r)) for r in contacted) / len(contacted)
mean_vis = float(np.mean([len(vis(r)) for r in rows]))
first = json.loads((done[0] / "manifest.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
statistics = {"frames": len(rows), "sessions": len(sessions), "probes": len(probes),
              "contact_fraction": len(contacted) / len(rows), "mean_visible_categories": mean_vis,
              "held_visible_fraction": held,
              "render_seconds_sum": sum(r["seconds"] for r in receipts)}
status = "complete" if held >= 0.7 and mean_vis >= 2 else "referential_ambiguity_failed"
manifest = {**{k: first[k] for k in ("schema_version", "config", "seed", "renderer", "rooms", "hand_vertex_ids_sha256",
            "body_models_sha256", "software", "code_commit")},
            "status": status, "sessions_per_subject": per_subject, "merged_from": "per-session runs (session-index) + probe run",
            "session_gate": {k: {"status": s, "held_visible": h, "mean_visible": m} for k, (s, h, m) in statuses.items() if s != "complete"},
            "tables_sha256": {n: sha(out / f"{n}.parquet") for n in ("frames", "sessions", "probes")},
            "statistics": statistics}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(statistics, indent=1), status, "gate-failing sessions:", len(manifest["session_gate"]))
