#!/usr/bin/env python3
"""Create/verify the owner-only P9 retained-artifact inventory and projections."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pilot_p9.json"
PILOT = ROOT / "pilots" / "juno_sample"


def require(value: object, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def storage_module():
    spec = importlib.util.spec_from_file_location("egobaby_storage", ROOT / "storage.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def projections(config: dict, aggregates: dict[str, dict]) -> dict:
    p3 = aggregates["P3"]
    hours = p3["aggregate_total_source_hours"]
    stage_seconds_per_hour = p3["wall_seconds_per_video_hour"]
    output_bytes_per_hour = sum(p3["storage_bytes"].values()) / hours
    scenarios = []
    for duration in config["duration_scenarios_hours"]:
        base_wall_hours = sum(stage_seconds_per_hour.values()) * duration / 3600
        scenarios.append({
            "source_video_hours": duration,
            "preprocessing_wall_hours": {
                name: round(base_wall_hours * multiplier, 6)
                for name, multiplier in config["preprocessing_uncertainty_multipliers"].items()
            },
            "base_stage_wall_hours": {
                name: round(seconds * duration / 3600, 6)
                for name, seconds in stage_seconds_per_hour.items()
            },
            "base_audio_plus_frames_bytes": round(output_bytes_per_hour * duration),
        })
    sizes = {
        "dino": aggregates["P5"]["resources"]["checkpoint_bytes"],
        "bert": aggregates["P6"]["resources"]["checkpoint_bytes"],
        "clip_plus": aggregates["P7"]["resources"]["checkpoint_bytes"],
    }
    retention = []
    for count in config["checkpoint_retention_counts_per_seed"]:
        retention.append({
            "checkpoints_per_component_per_seed": count,
            "three_seed_bytes": config["scientific_seed_count"] * count * sum(sizes.values()),
        })
    return {
        "measured_pilot_source_video_hours": hours,
        "p3_stage_wall_seconds_per_source_video_hour": stage_seconds_per_hour,
        "p3_audio_plus_frames_bytes_per_source_video_hour": round(output_bytes_per_hour),
        "duration_scenarios": scenarios,
        "measured_checkpoint_sizes_bytes": sizes,
        "checkpoint_retention_scenarios": retention,
        "training_cost_formula": "three_seed_cost = 3 * declared_full_run_updates * measured_cost_per_update_at_frozen_batch_world_size_semantics",
        "training_video_hour_ratio_extrapolation_allowed": False,
    }


def load_aggregates() -> dict[str, dict]:
    return {
        stage: json.loads((PILOT / filename).read_text())
        for stage, filename in {
            "P1": "p1_aggregate.json", "P2": "p2_aggregate.json",
            "P3": "p3_aggregate.json", "P4B": "p4_aggregate.json",
            "P4": "p4_clip_l_diagnostic_aggregate.json", "P5": "p5_aggregate.json",
            "P6": "p6_aggregate.json", "P7": "p7_aggregate.json", "P8": "p8_aggregate.json",
        }.items()
    }


def remote_program() -> str:
    return r'''import hashlib,json,os,pathlib,stat,sys
durable=pathlib.Path(sys.argv[1]); scratch=pathlib.Path(sys.argv[2]); out=pathlib.Path(sys.argv[3])
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
rows=[]
for p in sorted(durable.rglob('*')):
 if not p.is_file() or p==out or out.parent in p.parents: continue
 rel=str(p.relative_to(durable)); phase='P0-P8' if ('pilot_' in rel or 'p0-' in rel or 'p8-' in rel) else 'pre-pilot-or-storage'
 rows.append({'relative_path':rel,'bytes':p.stat().st_size,'sha256':digest(p),'mode':format(stat.S_IMODE(p.stat().st_mode),'04o'),'scope':phase})
bad_modes=[r['relative_path'] for r in rows if r['mode']!='0600']
required={
 'P1':['run_records/p0-4cc3af23/pilot_p1_audit.private.json'],
 'P2':['run_records/pilot_p2/p2-7e49c8a1.json'],
 'P3':['run_records/pilot_p0/p0-4cc3af23/pilot_p3/completion.json','run_records/pilot_p0/p0-4cc3af23/pilot_p3/checksum_inventory.json'],
 'P4':['run_records/pilot_p4/score_complete.json','run_records/pilot_p4/clip_l_diagnostic/clip_l_score_complete.json'],
 'P5':['checkpoints/pilot_p5/p5-91c43e2a/model_final.rank_0.pth','run_records/pilot_p0/p0-4cc3af23/pilot_p5/audit_attestation.json'],
 'P6':['checkpoints/pilot_p6/p6-6d31a4e7/step_100.pt','run_records/pilot_p0/p0-4cc3af23/pilot_p6/durable_inventory.json'],
 'P7':['checkpoints/pilot_p7/p7-7b9e2c41/step_130.pt','run_records/pilot_p0/p0-4cc3af23/pilot_p7/retained_inventory.json'],
 'P8':['run_records/pilot_p0/p0-4cc3af23/pilot_p8/attempt_ledger.json','run_records/pilot_p0/p0-4cc3af23/pilot_p8/integrity_audit.json','run_records/pilot_p0/p0-4cc3af23/pilot_p8/inventory.json']}
missing={k:[x for x in v if not (durable/x).is_file()] for k,v in required.items()}; missing={k:v for k,v in missing.items() if v}
scratch_families={}
for top in sorted(scratch.iterdir()):
 if top.is_dir():
  files=[p for p in top.rglob('*') if p.is_file()]
  scratch_families[top.name]={'file_count':len(files),'total_bytes':sum(p.stat().st_size for p in files)}
out.parent.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(out.parent,0o700)
payload={'schema_version':1,'scope':'complete_retained_pilot_artifact_inventory_not_full_dataset_inventory','entries':rows,'entry_count':len(rows),'total_bytes':sum(r['bytes'] for r in rows),'bad_file_modes':bad_modes,'required_family_missing':missing,'scratch_family_aggregates':scratch_families,'scratch_is_backup':False,'backup_or_snapshot_evidence':'not_demonstrated_by_storage_or_mount_checks'}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.chmod(out,0o600)
print(json.dumps({'entry_count':len(rows),'total_bytes':payload['total_bytes'],'inventory_sha256':digest(out),'bad_file_mode_count':len(bad_modes),'required_family_missing_count':sum(map(len,missing.values())),'scratch_family_aggregates':scratch_families,'backup_or_snapshot_evidence':payload['backup_or_snapshot_evidence']}))'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-juno", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    aggregates = load_aggregates()
    result = {"projections": projections(config, aggregates)}
    if args.audit_juno:
        storage = storage_module()
        storage.check_tier("durable"); storage.check_tier("scratch")
        durable = storage.storage_root("durable"); scratch = storage.storage_root("scratch")
        output = durable / config["durable_namespace"] / config["detailed_inventory_name"]
        completed = storage.ssh_run("python3", "-c", remote_program(), str(durable), str(scratch), str(output))
        audit = json.loads(completed.stdout)
        require(audit["bad_file_mode_count"] == 0, "durable inventory contains non-0600 files")
        require(audit["required_family_missing_count"] == 0, "required retained artifact is missing")
        result["governed_audit"] = audit
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
