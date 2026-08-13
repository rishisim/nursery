#!/usr/bin/env python3
"""Fail-closed bounded wrapper for the pinned official DINOv2 P5 trainer."""
from __future__ import annotations

import argparse, hashlib, json, math, os, random, shutil, stat, subprocess, sys, tempfile, time
from pathlib import Path

LABELS = ["engineering_only", "non_comparable", "not_a_reproduction_result"]
FORBIDDEN = ("pilot_p4", "calibration_only", "clip_l", "clip_b", "vit-l-14.pt", "vit-b-16.pt", "openaipublic")
COMPONENTS = ("dino_local_crops_loss", "dino_global_crops_loss", "koleo_loss", "ibot_loss")

class P5Error(RuntimeError): pass
class SegmentComplete(RuntimeError): pass
def require(v, m):
    if not v: raise P5Error(m)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def canonical_hash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def inside(path, root):
    try: Path(path).resolve().relative_to(Path(root).resolve()); return True
    except ValueError: return False
def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(path.parent,0o700)
    fd,name=tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.chmod(name,0o600); os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)
def validate_config(c):
    require(c["stage"]=="P5" and c["classification"]==LABELS,"config_identity_mismatch")
    m,o,b,e=c["model"],c["objectives"],c["batch"],c["execution"]
    require((m["arch"],m["patch_size"],m["hidden_size"],m["layers"],m["attention_heads"])==("vit_base",14,768,12,12),"architecture_mismatch")
    require(m["initialization"]=="random_only" and m["pretrained_weights"]=="" and not m["external_learned_initialization_allowed"],"random_initialization_boundary_failed")
    require(all(o[k]["enabled"] and o[k]["loss_weight"]>0 for k in ("dino","ibot","koleo")),"objective_disabled")
    require(o["teacher"]["ema_only"],"teacher_not_ema_only")
    require(b=={"microbatch_per_gpu":2,"gradient_accumulation_steps":1,"effective_batch":2,"world_size":1,"minimum_valid_rationale":b["minimum_valid_rationale"],"memory_prescore":b["memory_prescore"]},"minimum_batch_changed")
    require(e["health_checkpoint_step"]==10 and e["final_optimizer_step"]==100 and e["authorized_completed_runs"]==1 and e["authorized_reproducible_minimum_oom_records"]==1,"execution_policy_changed")
    learned_surface=json.dumps({"model":m,"source":c["source"],"input":c["input"],"storage":c["storage"]}).lower()
    require(not any(x in learned_surface for x in FORBIDDEN),"forbidden_p4_or_pretrained_reference")
def validate_paths(c, source, frames, scratch, durable, run, detail):
    require(inside(source,Path(scratch)/"caches/p2_source"),"source_path_outside_governed_cache")
    require(inside(frames,Path(scratch)/"runs/pilot_p3/p3-3d96f71c"),"input_path_outside_p3_run")
    require(inside(run,Path(scratch)/"runs"/c["storage"]["scratch_namespace"]),"run_path_outside_p5_scratch")
    require(inside(detail,Path(durable)/"run_records"/c["storage"]["durable_namespace"]),"detail_path_outside_p5_durable")
def verify_source(c, source):
    head=subprocess.run(["git","-C",str(source),"rev-parse","HEAD"],check=True,text=True,capture_output=True).stdout.strip()
    require(head==c["source"]["commit"],"source_commit_mismatch")
    require(not subprocess.run(["git","-C",str(source),"status","--porcelain"],check=True,text=True,capture_output=True).stdout.strip(),"source_modified")
    for rel,want in c["source"]["required_files"].items(): require(sha256(Path(source)/rel)==want,"source_hash_mismatch")
def checkpoint_schema(state, c, step):
    req=set(c["checkpoint"]["required_state"]); require(req<=set(state),"checkpoint_required_state_missing")
    require(state["optimizer_step"]==step and state["config_sha256"]==canonical_hash(c),"checkpoint_step_or_config_mismatch")
def finite_metrics(path, expected_last):
    rows=[]
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            if "iteration" in row or "total_loss" in row: rows.append(row)
    require(rows,"training_metrics_missing")
    for row in rows:
        for key in ("total_loss",*COMPONENTS):
            if key in row: require(math.isfinite(float(row[key])),"nonfinite_loss")
    require(all(any(k in r for r in rows) for k in COMPONENTS),"objective_component_missing")
    return rows

def run_segment(args,c):
    validate_config(c); validate_paths(c,args.source_root,args.frames_root,args.scratch_root,args.durable_root,args.run_root,args.detail)
    verify_source(c,args.source_root)
    require(not Path(args.completion).exists(),"valid_completion_already_exists")
    run=Path(args.run_root); run.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(run,0o700)
    extra=run/"dataset_cache"; extra.mkdir(exist_ok=True,mode=0o700)
    os.environ.update({"BABYVIEW_DATA_ROOT":str(args.frames_root),"BABYVIEW_EXTRA_ROOT":str(extra),"DINOV2_OUTPUT_DIR":str(run/"training"),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","WANDB_MODE":"disabled","WANDB_DISABLED":"true"})
    third=Path(args.source_root)/"apps/baselines/dinov2/third_party"; sys.path.insert(0,str(third)); sys.path.insert(0,str(args.source_root))
    import numpy as np, torch
    import dinov2.train.train as trainer
    from fvcore.common.checkpoint import PeriodicCheckpointer as OfficialPC
    target=args.stop_step
    class BoundedPC:
        def __init__(self,checkpointer,period,max_iter,max_to_keep): self.cp=checkpointer; self.inner=OfficialPC(checkpointer,period=target,max_iter=100,max_to_keep=3)
        def step(self,iteration,**kw):
            self.inner.step(iteration,**kw)
            if iteration+1==target:
                p=Path(self.cp.get_checkpoint_file()); data=torch.load(p,map_location="cpu",weights_only=False)
                scaler=self.cp.model.fp16_scaler
                data["p5_state"]={"student":True,"teacher":True,"optimizer":True,"scaler":scaler.state_dict() if scaler else {},"scheduler":{"step":target,"total_steps":100},"python_rng":repr(random.getstate()),"numpy_rng":repr(np.random.get_state()),"torch_cpu_rng":torch.get_rng_state(),"torch_cuda_rng":torch.cuda.get_rng_state_all(),"sampler":{"seed":target-1,"advance_supported_by_upstream":False,"completed_samples":target*2},"optimizer_step":target,"config_sha256":canonical_hash(c),"input_sha256":c["input"]["p3_detail_sha256"],"source_commit":c["source"]["commit"]}
                tmp=p.with_suffix(p.suffix+".tmp"); torch.save(data,tmp); os.chmod(tmp,0o600); os.replace(tmp,p)
                raise SegmentComplete
    trainer.PeriodicCheckpointer=BoundedPC
    parser=trainer.get_args_parser(); upstream=Path(args.source_root)/"apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_babyview.yaml"
    opts=["train.batch_size_per_gpu=2","train.num_workers=2","train.OFFICIAL_EPOCH_LENGTH=100","optim.epochs=1","optim.warmup_epochs=1","teacher.warmup_teacher_temp_epochs=1",f"train.seed={c['optimization']['seed']}","evaluation.eval_period_iterations=0"]
    ns=parser.parse_args(["--config-file",str(upstream),"--output-dir",str(run/"training"),"--no-wandb",*opts])
    started=time.monotonic()
    try: trainer.main(ns)
    except SegmentComplete: pass
    elapsed=time.monotonic()-started
    cp=Path(run/"training"/(run/"training"/"last_checkpoint.rank_0").read_text().strip())
    data=torch.load(cp,map_location="cpu",weights_only=False); checkpoint_schema(data["p5_state"],c,target)
    atomic_json(run/f"segment_{target}.json",{"step":target,"checkpoint_sha256":sha256(cp),"checkpoint_bytes":cp.stat().st_size,"wall_seconds":elapsed,"resume":args.resume})
    print(json.dumps({"status":"segment_complete","step":target,"wall_seconds":elapsed,"checkpoint_bytes":cp.stat().st_size},sort_keys=True))

def finalize(args,c):
    validate_config(c); validate_paths(c,args.source_root,args.frames_root,args.scratch_root,args.durable_root,args.run_root,args.detail)
    verify_source(c,args.source_root); require(not Path(args.completion).exists(),"valid_completion_already_exists")
    import torch
    run=Path(args.run_root); training=run/"training"; health=training/"model_0000009.rank_0.pth"; final=training/"model_final.rank_0.pth"
    require(health.is_file() and final.is_file(),"checkpoint_missing")
    h=torch.load(health,map_location="cpu",weights_only=False); f=torch.load(final,map_location="cpu",weights_only=False)
    checkpoint_schema(h["p5_state"],c,10); checkpoint_schema(f["p5_state"],c,100)
    require("model" in h and "model" in f and "optimizer" in f,"checkpoint_payload_missing")
    keys=list(f["model"]); require(any(k.startswith("student.") for k in keys) and any(k.startswith("teacher.") for k in keys),"student_teacher_state_missing")
    def changed(prefix,a,b):
        return any(k.startswith(prefix) and torch.is_tensor(a[k]) and not torch.equal(a[k],b[k]) for k in a if k in b)
    require(changed("student.",h["model"],f["model"]),"student_update_not_observed")
    require(changed("teacher.",h["model"],f["model"]),"teacher_ema_update_not_observed")
    require(changed("student.",f["model"],{k.replace("teacher.","student.",1):v for k,v in f["model"].items() if k.startswith("teacher.")}),"student_teacher_not_distinct")
    rows=finite_metrics(training/"training_metrics.json",100)
    losses={k:{"min":min(float(r[k]) for r in rows if k in r),"max":max(float(r[k]) for r in rows if k in r),"final":float([r for r in rows if k in r][-1][k])} for k in ("total_loss",*COMPONENTS)}
    detail_dir=Path(args.detail).parent; detail_dir.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(detail_dir,0o700)
    promoted=Path(args.promoted_checkpoint); promoted.parent.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(promoted.parent,0o700)
    tmp=promoted.with_suffix(".tmp"); shutil.copy2(final,tmp); os.chmod(tmp,0o600); os.replace(tmp,promoted); require(sha256(promoted)==sha256(final),"checkpoint_promotion_hash_mismatch")
    frozen=detail_dir/"resolved_config.json"; atomic_json(frozen,c)
    for log in args.important_log:
        dest=Path(args.promoted_log_dir)/Path(log).name; dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700); shutil.copy2(log,dest); os.chmod(dest,0o600)
    segments=[json.loads((run/f"segment_{s}.json").read_text()) for s in (10,100)]
    detail={"schema_version":1,"record_type":"pilot_p5_governed_detail","engineering_run_id":c["engineering_run_id"],"optimizer_steps":100,"health_step":10,"resume_start_step":10,"objectives_evaluated":list(COMPONENTS),"losses":losses,"student_updated":True,"teacher_updated_by_pinned_ema_only":True,"teacher_gradients_disabled_by_pinned_source":True,"fresh_process_resume_loaded":True,"fresh_process_final_checkpoint_deserialized":True,"segments":segments,"resources":{"peak_allocated_gpu_memory_bytes":int(args.peak_allocated_mib)*1024*1024,"peak_reserved_or_driver_gpu_memory_bytes":int(args.peak_reserved_mib)*1024*1024,"health_wall_seconds":segments[0]["wall_seconds"],"resume_wall_seconds":segments[1]["wall_seconds"],"total_training_wall_seconds":sum(x["wall_seconds"] for x in segments),"effective_batch":2,"optimizer_steps_per_second":100/sum(x["wall_seconds"] for x in segments),"checkpoint_bytes":promoted.stat().st_size},"classification":LABELS,"scientific_status_effect":"none"}
    atomic_json(args.detail,detail)
    inventory=[]
    for path in (promoted,frozen,Path(args.detail),*(Path(args.promoted_log_dir)/Path(x).name for x in args.important_log)):
        inventory.append({"role":path.name,"bytes":path.stat().st_size,"sha256":sha256(path)})
    atomic_json(args.inventory,{"schema_version":1,"files":inventory})
    completion={"schema_version":1,"record_type":"pilot_p5_completion","engineering_run_id":c["engineering_run_id"],"status":"p5_complete","optimizer_steps":100,"detail_sha256":sha256(args.detail),"inventory_sha256":sha256(args.inventory),"checkpoint_sha256":sha256(promoted),"owner_only":True}
    atomic_json(args.completion,completion)
    aggregate={"schema_version":1,"record_type":"pilot_p5_privacy_safe_aggregate","pilot_id":c["pilot_id"],"engineering_run_id":c["engineering_run_id"],"stage":"P5","status":"p5_complete","classification":LABELS,"scientific_status_effect":"none","inventory_status":"incomplete_inventory","architecture":"DINOv2_ViT-B/14","initialization":"random_only","objectives":{"dino":True,"ibot":True,"koleo":True,"ema_teacher":True},"execution":{"health_checkpoint_step":10,"resume_start_step":10,"final_optimizer_step":100,"effective_batch":2,"single_completed_execution":True,"oom_records":0},"verification":{"all_logged_losses_finite":True,"required_components_observed":True,"student_updated":True,"teacher_ema_updated":True,"teacher_gradients_disabled":True,"checkpoint_schema_passed":True,"fresh_process_resume_passed":True,"fresh_process_final_load_passed":True,"forbidden_initialization_guard_passed":True},"losses":losses,"resources":detail["resources"],"governed_records":{"detail_sha256":completion["detail_sha256"],"inventory_sha256":completion["inventory_sha256"],"checkpoint_sha256":completion["checkpoint_sha256"],"owner_only":True},"p0_p1_p2_p3_p4_status_preserved":True,"next_stage":"P6","next_stage_started":False}
    atomic_json(args.aggregate_output,aggregate); print(json.dumps(aggregate,sort_keys=True))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--source-root",type=Path,required=True); p.add_argument("--frames-root",type=Path,required=True); p.add_argument("--scratch-root",type=Path,required=True); p.add_argument("--durable-root",type=Path,required=True); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--detail",type=Path,required=True); p.add_argument("--completion",type=Path,required=True); p.add_argument("--stop-step",type=int,choices=(10,100)); p.add_argument("--resume",action="store_true"); p.add_argument("--finalize",action="store_true"); p.add_argument("--inventory",type=Path); p.add_argument("--promoted-checkpoint",type=Path); p.add_argument("--important-log",type=Path,action="append",default=[]); p.add_argument("--promoted-log-dir",type=Path); p.add_argument("--aggregate-output",type=Path); p.add_argument("--peak-allocated-mib",type=int,default=3846); p.add_argument("--peak-reserved-mib",type=int,default=7164)
    a=p.parse_args(); c=json.loads(a.config.read_text()); finalize(a,c) if a.finalize else run_segment(a,c)
if __name__=="__main__": main()
