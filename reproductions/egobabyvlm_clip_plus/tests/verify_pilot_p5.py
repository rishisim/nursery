#!/usr/bin/env python3
"""Dependency-free permanent verification for Pilot P5."""
import importlib.util, json, math, tempfile
from types import SimpleNamespace
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
        f=Path(n)/"m.jsonl"; row=lambda i:{"iteration":i,"total_loss":1.0,**{k:0.1 for k in p.COMPONENTS}}
        f.write_text("\n".join(json.dumps(row(i)) for i in range(0,100,10))+"\n")
        rows,coverage=p.finite_metrics(f,100); require(coverage["row_count"]==10 and coverage["final_logged_iteration"]==90,"metric coverage not recorded")
        f.write_text(json.dumps({**row(0),"total_loss":math.inf})+"\n"); fails(lambda:p.finite_metrics(f,10),"nonfinite loss accepted")
        f.write_text(json.dumps({k:v for k,v in row(0).items() if k!="ibot_loss"})+"\n"); fails(lambda:p.finite_metrics(f,10),"missing component accepted")
        f.write_text("\n".join(json.dumps(row(i)) for i in (0,20,10))+"\n"); fails(lambda:p.finite_metrics(f,30),"unordered rows accepted")
        f.write_text("\n".join(json.dumps(row(i)) for i in (0,10))+"\n"); fails(lambda:p.finite_metrics(f,30),"incomplete coverage accepted")
        f.write_text("\n".join(json.dumps(row(i)) for i in (0,10))+"\n"); fails(lambda:p.finite_metrics(f,10),"post-target row accepted")
    segments=[{"step":10,"resume":False},{"step":100,"resume":True}]
    resume=p.validate_resume_evidence(segments,"No checkpoint found. Initializing model from scratch\nStarting training from iteration 0","Loading optimizer from /governed/model_0000009.rank_0.pth\nStarting training from iteration 10","health-111.log","resume-222.log")
    require(resume["separate_process_job_proven"],"separate process evidence absent")
    fails(lambda:p.validate_resume_evidence(segments,"Starting training from iteration 0","Starting training from iteration 10","health-111.log","resume-111.log"),"same job accepted")
    p.validate_resource_evidence(3846,7164,"pinned_upstream_allocator_max_mem_log_mib","slurm_sacct_gres_gpumem_inmax_mib","max mem: 3846","gres/gpumem=7164M")
    fails(lambda:p.validate_resource_evidence(None,7164,"pinned_upstream_allocator_max_mem_log_mib","slurm_sacct_gres_gpumem_inmax_mib","max mem: 3846","gres/gpumem=7164M"),"missing measurement accepted")
    fails(lambda:p.validate_resource_evidence(3846,7164,None,"slurm_sacct_gres_gpumem_inmax_mib","max mem: 3846","gres/gpumem=7164M"),"unattributed measurement accepted")
    fake_state={k:True for k in c["checkpoint"]["required_state"]}; fake_state.update(optimizer_step=100,config_sha256=p.canonical_hash(c),scaler={"scale":1})
    official={"model":{"student.x":1,"teacher.x":2},"optimizer":{"state":1},"iteration":99,"p5_state":fake_state}
    evidence=p.validate_official_checkpoint_payload(official,c,100); require(not evidence["official_scaler_checkpointable"] and evidence["supplemental_scaler_present"],"official/supplemental state conflated")
    bad=dict(official); bad.pop("optimizer"); fails(lambda:p.validate_official_checkpoint_payload(bad,c,100),"missing official optimizer accepted")
    with tempfile.TemporaryDirectory() as n:
        d=Path(n); durable=d/"durable"; scratch=d/"scratch"; record=durable/"run_records"/c["storage"]["durable_namespace"]
        args=SimpleNamespace(durable_root=durable,scratch_root=scratch,run_root=scratch/"runs"/c["storage"]["scratch_namespace"],detail=record/"detail.json",completion=record/"completion.json",inventory=record/"checksum_inventory.json",promoted_checkpoint=durable/"checkpoints/pilot_p5"/c["engineering_run_id"]/"model_final.rank_0.pth",promoted_log_dir=record/"logs",aggregate_output=durable/"aggregate_results"/f"pilot_p5_{c['engineering_run_id']}.json",important_log=[durable/"logs/pilot_p5"/c["engineering_run_id"]/"health-111.log",durable/"logs/pilot_p5"/c["engineering_run_id"]/"resume-222.log"])
        p.validate_finalize_paths(c,args); args.aggregate_output=d/"outside.json"; fails(lambda:p.validate_finalize_paths(c,args),"arbitrary aggregate path accepted")
    require(c["execution"]["authorized_completed_runs"]==c["execution"]["authorized_reproducible_minimum_oom_records"]==1,"single run/OOM policy changed")
    require(c["input"]["privacy_safe_total_frame_count"]==684 and c["inventory_status"]=="incomplete_inventory","input aggregate changed")
    if AGG.exists():
        text=AGG.read_text(); a=json.loads(text)
        require(a["status"] in {"p5_complete","p5_incomplete_minimum_valid_oom"},"invalid P5 state")
        require(a["classification"]==p.LABELS and a["scientific_status_effect"]=="none","classification changed")
        require(a["p0_p1_p2_p3_p4_status_preserved"] and a["next_stage"]=="P6" and not a["next_stage_started"],"pilot lifecycle changed")
        require(a["metric_log_coverage"]["iterations"]==list(range(0,100,10)),"tracked metric coverage incomplete")
        require(a["execution"]["sample_continuity"].startswith("not_proven"),"sample continuity overstated")
        require(a["verification"]["official_model_restore_proven"] and a["verification"]["official_optimizer_restore_proven"] and not a["verification"]["official_scaler_restore_proven"],"official checkpoint claims inaccurate")
        require(a["resources"]["allocated_source"] and a["resources"]["reserved_source"] and a["resources"]["allocator_log_sha256"] and a["resources"]["scheduler_record_sha256"],"resource attribution missing")
        require(not any(x in text.lower() for x in ("/work/","/scratch/","participant","session","frame_","image_","caption_")),"aggregate leaks governed detail")
    print("Pilot P5 verification passed")
if __name__=="__main__": main()
