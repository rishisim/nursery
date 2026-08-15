#!/usr/bin/env python3
"""Dependency-free functional regression checks for hardened Pilot P6."""
import importlib.util, json, os, stat, tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/"configs/pilot_p6.json").read_text())
A=json.loads((ROOT/"pilots/juno_sample/p6_aggregate.json").read_text())
P=json.loads((ROOT/"configs/pilot.json").read_text())
D=json.loads((ROOT/"pilots/juno_sample/decision_record.json").read_text())

def require(v,m):
    if not v: raise AssertionError(m)
def fails(fn,m):
    try: fn()
    except Exception: return
    raise AssertionError(m)
def module():
    spec=importlib.util.spec_from_file_location("pilot_p6",ROOT/"scripts/pilot_p6.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def checkpoint(step=50):
    return {"model":{"x":1},"optimizer":{"state":{}},"scheduler":{"last_epoch":step},"optimizer_step":step,"iteration":step,"tokenizer_sha256":"tok","config_sha256":"cfg","python_rng":None,"torch_cpu_rng":b"cpu","torch_cuda_rng":[b"cuda"],"sampler_state":b"sampler","slurm_job_id":"324821"}

def main():
    p=module(); p.load_config(ROOT/"configs/pilot_p6.json")
    require(P["pilot_stage"]=="P7" and P["status"]=="p7_complete" and P["next_stage"]=="P8" and not P["next_stage_started"],"canonical lifecycle stale")
    require(P["p5"]["next_stage_started"] and P["p6"]["status"]=="p6_complete" and P["p6"]["next_stage_started"],"stage lifecycle inconsistent")
    require(D["p6_status"]["status"]=="p6_complete" and D["p6_status"]["next_stage"]=="P7","decision lifecycle inconsistent")
    require(C["tokenizer"]["vocab_size"]==C["model"]["vocab_size"]==380,"authorized vocabulary changed")
    active=json.dumps({k:C[k] for k in ("tokenizer","model","training","acceptance","stop_conditions")}).lower()
    require("2048" not in active,"stale 2048 remains active")
    require(C["historical_provenance"]["initial_2048_gate"].startswith("stopped"),"2048 history not isolated")
    bad=json.loads(json.dumps(C)); bad["model"]["vocab_size"]=2048
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)/"bad.json"; f.write_text(json.dumps(bad)); fails(lambda:p.load_config(f),"stale vocab accepted")
    state=checkpoint(); p.validate_checkpoint_state(state,C,50,"tok","cfg")
    for key in ("model","optimizer","scheduler","torch_cpu_rng","torch_cuda_rng","sampler_state","slurm_job_id"):
        bad=checkpoint(); bad.pop(key); fails(lambda b=bad:p.validate_checkpoint_state(b,C,50,"tok","cfg"),f"missing {key} accepted")
    for key,value in (("optimizer_step",49),("iteration",49),("tokenizer_sha256","bad"),("config_sha256","bad")):
        bad=checkpoint(); bad[key]=value; fails(lambda b=bad:p.validate_checkpoint_state(b,C,50,"tok","cfg"),f"bad {key} accepted")
    bad=checkpoint(); bad["scheduler"]["last_epoch"]=49; fails(lambda:p.validate_checkpoint_state(bad,C,50,"tok","cfg"),"scheduler mismatch accepted")
    bad=checkpoint(); bad["python_rng"]=(1,2); fails(lambda:p.validate_checkpoint_state(bad,C,50,"tok","cfg"),"false Python RNG claim accepted")
    with tempfile.TemporaryDirectory() as td:
        base=Path(td); source=base/"source"; python=source/".pixi/envs/default/bin/python"; python.parent.mkdir(parents=True); python.write_text("")
        cfg=json.loads(json.dumps(C)); (source/"pixi.lock").write_text("lock"); cfg["environment"]["pixi_lock_sha256"]=p.sha256(source/"pixi.lock")
        for relative in cfg["source"]["required_files"]:
            path=source/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(relative); cfg["source"]["required_files"][relative]=p.sha256(path)
        p2=base/"pilot_p2.json"; p2.write_text("{}"); cfg["environment"]["p2_config_sha256"]=p.sha256(p2)
        durable=base/"durable"; detail=durable/"run_records/pilot_p2/p2-7e49c8a1.json"; detail.parent.mkdir(parents=True); detail.write_text("{}"); os.chmod(detail,0o600); cfg["environment"]["p2_detail_sha256"]=p.sha256(detail)
        original=p.subprocess.run
        p.subprocess.run=lambda command,**kwargs: SimpleNamespace(stdout=cfg["source"]["commit"]+"\n" if command[0]=="git" else cfg["environment"]["transformers"]+"\n")
        try:
            p.verify_source_environment(cfg,source,python,p2,durable)
            (source/"pixi.lock").write_text("changed"); fails(lambda:p.verify_source_environment(cfg,source,python,p2,durable),"lock hash mismatch accepted")
        finally: p.subprocess.run=original
    with tempfile.TemporaryDirectory() as td:
        base=Path(td); scratch=base/"scratch"; durable=base/"durable"
        for path in (scratch,durable,scratch/"runs",durable/"run_records"): path.mkdir(mode=0o700)
        run,record=p.governed_paths(C,scratch,durable)
        require(run.relative_to(scratch.resolve())==Path("runs/pilot_p6/p6-6d31a4e7") and record.relative_to(durable.resolve())==Path("run_records/pilot_p0/p0-4cc3af23/pilot_p6"),"namespace mismatch")
        record.mkdir(parents=True,mode=0o700); (record/"completion.json").write_text("{}")
        fails(lambda:p.reject_completed(record),"post-completion mutation accepted")
        os.chmod(scratch,0o755); fails(lambda:p.governed_paths(C,scratch,durable),"non-owner-only root accepted")
    e=A["execution"]; v=A["verification"]
    require(e["distinct_slurm_jobs_proven"] and e["validation_runs"]==1 and not e["p7_started"],"job/validation evidence incomplete")
    require(v["fresh_finalize_job_distinct_from_resume"] and v["validation_non_generalizing_and_not_used_for_selection"],"fresh-load/selection claim missing")
    require(not v["python_rng_used"] and not v["python_rng_restored"] and A["resources"]["scheduler_driver_memory_bytes"] is None,"RNG or driver memory overstated")
    g=A["governed_records"]; require(all(g[k] for k in ("hardening_attestation_sha256","durable_inventory_sha256","health_log_sha256","resume_log_sha256","finalize_log_sha256")),"durable inventory schema incomplete")
    require(A["historical_provenance"]["initial_2048_gate"].startswith("superseded") and "excluded" in A["historical_provenance"]["invalid_stale_acceptance_label_attempt"],"historical attempts conflated")
    require(A["classification"]==C["classification"] and A["scientific_status_effect"]=="none" and A["inventory_status"]=="incomplete_inventory","classification changed")
    text=(ROOT/"pilots/juno_sample/p6_aggregate.json").read_text().lower()
    require(not any(x in text for x in ("/work/","/scratch/","participant","utterance_text","[pad]","[unk]","[cls]","[sep]","[mask]")),"privacy-safe aggregate leaks detail")
    job=(ROOT/"scripts/pilot_p6_juno_job.sh").read_text(); require(all(x in job for x in ("prepare)","health)","resume)","finalize)","storage.py\" check-juno"," preflight ")),"job phases/preflight incomplete")
    require(C["environment"]["p2_config_sha256"]==p.sha256(ROOT/"configs/pilot_p2.json"),"P2 config digest stale")
    print("Pilot P6 hardening verification passed")
if __name__=="__main__": main()
