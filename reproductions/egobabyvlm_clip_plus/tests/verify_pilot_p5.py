#!/usr/bin/env python3
"""Dependency-free permanent verification for Pilot P5."""
import importlib.util, json, math, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; CONFIG=ROOT/"configs/pilot_p5.json"; AGG=ROOT/"pilots/juno_sample/p5_aggregate.json"
def require(v,m):
    if not v: raise AssertionError(m)
def fails(fn,m):
    try: fn()
    except Exception: return
    raise AssertionError(m)
def module():
    s=importlib.util.spec_from_file_location("pilot_p5",ROOT/"scripts/pilot_p5.py"); x=importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
def main():
    p=module(); c=json.loads(CONFIG.read_text()); p.validate_config(c)
    bad=json.loads(json.dumps(c)); bad["model"]["arch"]="vit_small"; fails(lambda:p.validate_config(bad),"smaller architecture accepted")
    bad=json.loads(json.dumps(c)); bad["objectives"]["ibot"]["loss_weight"]=0; fails(lambda:p.validate_config(bad),"disabled iBOT accepted")
    bad=json.loads(json.dumps(c)); bad["model"]["pretrained_weights"]="weights.pth"; fails(lambda:p.validate_config(bad),"pretrained initialization accepted")
    bad=json.loads(json.dumps(c)); bad["storage"]["scratch_namespace"]="pilot_p4_calibration_only"; fails(lambda:p.validate_config(bad),"P4 namespace accepted")
    state={k:True for k in c["checkpoint"]["required_state"]}; state.update(optimizer_step=10,config_sha256=p.canonical_hash(c)); p.checkpoint_schema(state,c,10)
    del state["teacher"]; fails(lambda:p.checkpoint_schema(state,c,10),"corrupt checkpoint accepted")
    state["teacher"]=True; state["optimizer_step"]=9; fails(lambda:p.checkpoint_schema(state,c,10),"step mismatch accepted")
    with tempfile.TemporaryDirectory() as n:
        f=Path(n)/"m.jsonl"; f.write_text(json.dumps({"iteration":9,"total_loss":1.0,**{k:0.1 for k in p.COMPONENTS}})+"\n")
        p.finite_metrics(f,10); f.write_text(json.dumps({"total_loss":math.inf,**{k:0.1 for k in p.COMPONENTS}})+"\n"); fails(lambda:p.finite_metrics(f,10),"nonfinite loss accepted")
    require(c["execution"]["authorized_completed_runs"]==c["execution"]["authorized_reproducible_minimum_oom_records"]==1,"single run/OOM policy changed")
    require(c["input"]["privacy_safe_total_frame_count"]==684 and c["inventory_status"]=="incomplete_inventory","input aggregate changed")
    if AGG.exists():
        text=AGG.read_text(); a=json.loads(text)
        require(a["status"] in {"p5_complete","p5_incomplete_minimum_valid_oom"},"invalid P5 state")
        require(a["classification"]==p.LABELS and a["scientific_status_effect"]=="none","classification changed")
        require(a["p0_p1_p2_p3_p4_status_preserved"] and a["next_stage"]=="P6" and not a["next_stage_started"],"pilot lifecycle changed")
        require(not any(x in text.lower() for x in ("/work/","/scratch/","participant","session","frame_","image_","caption_")),"aggregate leaks governed detail")
    print("Pilot P5 verification passed")
if __name__=="__main__": main()
