#!/usr/bin/env python3
"""Fail-closed P8 adapter qualification and single Machine-DevBench score."""
from __future__ import annotations
import argparse, hashlib, json, math, os, stat, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
from typing import Any

LABELS=["engineering_only","non_comparable","not_a_reproduction_result"]
class P8Error(RuntimeError): pass
def require(v:Any,m:str)->None:
    if not v: raise P8Error(m)
def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def atomic_json(p:Path,v:dict[str,Any])->None:
    p.parent.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(p.parent,0o700)
    fd,n=tempfile.mkstemp(prefix="."+p.name+".",dir=p.parent)
    try:
        with os.fdopen(fd,"w") as f: json.dump(v,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.chmod(n,0o600); os.replace(n,p)
    finally:
        if os.path.exists(n): os.unlink(n)
def owner(p:Path,d=False)->None:
    require(p.exists(),"governed_path_missing:"+p.name); require(stat.S_IMODE(p.stat().st_mode)==(0o700 if d else 0o600),"owner_mode:"+p.name)
def state_hash(torch,state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def load_config(p:Path)->dict:
    c=json.loads(p.read_text()); require(c["stage"]=="P8" and c["classification"]==LABELS,"config_identity"); return c

class P7CheckpointAdapter:
    """Canonical P7 towers exposed through the released evaluator protocol."""
    def __init__(self, checkpoint:str, p7_config:str, source_root:str, p5_training_dir:str, p6_checkpoint:str, tokenizer:str, vocab:str):
        import torch
        from pilot_p7 import config as p7_config_load, imports, make_model
        class A: pass
        a=A(); a.p6_checkpoint=Path(p6_checkpoint); a.vocab=Path(vocab); a.p5_training_dir=Path(p5_training_dir); a.source_root=Path(source_root)
        c=p7_config_load(Path(p7_config)); lib=imports(a.source_root); model,tok,_=make_model(c,a,lib)
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False); result=model.load_state_dict(payload["model"],strict=True)
        require(not result.missing_keys and not result.unexpected_keys,"strict_state_coverage")
        self.model=model.eval(); self.tokenizer=tok; self.device=next(model.parameters()).device
        T=lib[4]; self.pil_transform=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.loaded={"model_state_sha256":state_hash(torch,payload["model"]),"parameter_keys":len(payload["model"]),"temperature":float(model.logit_scale.detach().cpu()),"global_update":payload["global_update"]}
    def eval(self): self.model.eval(); return self
    def to(self,device): self.model.to(device); self.device=device; return self
    def extract_features(self,batch):
        import torch
        images=torch.stack([self.pil_transform(x.convert("RGB")) for x in batch["image"]]).to(self.device)
        enc=self.tokenizer(batch["text"],padding=True,truncation=True,max_length=64,return_tensors="pt").to(self.device)
        vi=torch.nn.functional.normalize(self.model.vision(images),dim=-1)
        hidden=self.model.bert(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"]).last_hidden_state[:,0]
        tx=torch.nn.functional.normalize(self.model.text_projection(hidden),dim=-1)
        return {"image_features":vi,"text_features":tx}
    def compute_similarity(self,image_features,text_features,normalize=True):
        import torch
        if normalize: image_features=torch.nn.functional.normalize(image_features,dim=-1); text_features=torch.nn.functional.normalize(text_features,dim=-1)
        return image_features@text_features.T*self.model.logit_scale.exp()

def common(c,a):
    owner(a.scratch_root,True); owner(a.durable_root,True); owner(a.checkpoint); owner(a.p7_config); owner(a.p6_checkpoint); owner(a.tokenizer); owner(a.vocab)
    require(sha256(a.checkpoint)==c["selected_checkpoint"]["sha256"],"checkpoint_hash")
    for p,k in ((a.p7_config,"p7_executed_config_sha256"),(a.p6_checkpoint,"p6_checkpoint_sha256"),(a.tokenizer,"p6_tokenizer_sha256"),(a.vocab,"p6_vocab_sha256")): require(sha256(p)==c["prerequisites"][k],k)
    require(subprocess.run(["git","-C",str(a.source_root),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()==c["source"]["commit"],"source_commit")
    require(sha256(a.source_root/"pixi.lock")==c["source"]["pixi_lock_sha256"],"environment_hash")
    for rel,want in c["source"]["evaluator_hashes"].items(): require(sha256(a.source_root/rel)==want,"evaluator_hash:"+rel)
    for p,k in ((a.p7_completion,"p7_completion_sha256"),(a.p7_audit,"p7_hardening_audit_sha256"),(a.p7_inventory,"p7_retained_inventory_sha256")): owner(p); require(sha256(p)==c["prerequisites"][k],k)
    require(sha256(a.p4_selection)==c["evaluator"]["p4_selection_sha256"],"p4_selection_hash")

def adapter(c,a):
    return P7CheckpointAdapter(*map(str,(a.checkpoint,a.p7_config,a.source_root,a.p5_training_dir,a.p6_checkpoint,a.tokenizer,a.vocab)))
def qualify(c,a):
    common(c,a); import torch
    from PIL import Image
    model=adapter(c,a); img=Image.new("RGB",(224,224),(17,31,47))
    with torch.no_grad(): f=model.extract_features({"image":[img,img],"text":["a synthetic test","another synthetic test"]}); s=model.compute_similarity(f["image_features"],f["text_features"])
    require(tuple(f["image_features"].shape)==(2,512) and tuple(f["text_features"].shape)==(2,512),"feature_dimensions")
    require(torch.isfinite(s).all() and torch.allclose(f["image_features"].norm(dim=-1),torch.ones(2,device=model.device),atol=1e-5),"forward_contract")
    ledger=a.record_root/"attempt_ledger.json"; require(not ledger.exists(),"ledger_already_exists")
    q={"schema_version":1,"status":"qualified","benchmark_accessed":False,"checkpoint_sha256":sha256(a.checkpoint),"adapter":model.loaded,"dimensions":{"vision":768,"text":768,"projection":512},"normalization":True,"text_pooling":"CLS","tokenizer_size":len(model.tokenizer),"synthetic_forward_finite":True,"aggregation_regression":"verified_by_tracked_P4_fixture","classification":LABELS}
    atomic_json(a.record_root/"qualification.json",q); atomic_json(ledger,{"schema_version":1,"run_id":c["engineering_run_id"],"state":"not_started","authorized_attempts":1,"scoring_started_count":0,"scoring_complete_count":0,"benchmark_accessed":False}); print(json.dumps(q,sort_keys=True))
def evaluate(c,a):
    common(c,a); ledger=a.record_root/"attempt_ledger.json"; owner(ledger); old=json.loads(ledger.read_text()); require(old["state"]=="not_started" and old["scoring_started_count"]==0,"attempt_consumed")
    require(sha256(a.qualification)==sha256(a.record_root/"qualification.json"),"qualification_record")
    job=os.environ.get("SLURM_JOB_ID",""); require(job.isdigit(),"slurm_job_required")
    atomic_json(ledger,{**old,"state":"scoring_started","scoring_started_count":1,"benchmark_accessed":True,"slurm_job_id":job,"started_unix":time.time()})
    began=time.monotonic(); base=a.scratch_root/"evaluation_data"/c["storage"]["scratch_namespace"]; archive=base/"archive"/c["archive"]["name"]; extract=base/"extracted"
    archive.parent.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(archive.parent,0o700)
    if not archive.exists(): urllib.request.urlretrieve("https://github.com/facebookresearch/egobabyvlm/releases/download/Eval-Data/MachineDevBench.tar",archive); os.chmod(archive,0o600)
    require(archive.stat().st_size==c["archive"]["size_bytes"] and sha256(archive)==c["archive"]["sha256"],"archive_hash")
    from pilot_p4 import safe_extract
    inv=safe_extract(archive,extract,c); data=extract/"MachineDevBench"
    sys.path[:0]=[str(a.source_root),str(Path(__file__).parent)]
    from omegaconf import OmegaConf
    from evaluation.multimodal.machine_devbench.base import MachineDevBenchEvalModule
    from evaluation.multimodal.machine_devbench.metrics import ResultAggregator
    pooled=ResultAggregator(); raw={}; per={}; seen=set(); tasks=c["inventory"]["tasks"]
    kwargs={"checkpoint":str(a.checkpoint),"p7_config":str(a.p7_config),"source_root":str(a.source_root),"p5_training_dir":str(a.p5_training_dir),"p6_checkpoint":str(a.p6_checkpoint),"tokenizer":str(a.tokenizer),"vocab":str(a.vocab)}
    for style in c["inventory"]["styles"]:
        cfg=OmegaConf.create({"_target_":"evaluation.multimodal.machine_devbench.base.MachineDevBenchEvalModule","name":"machine_devbench","output_dir":str(base/"outputs"),"data_root":str(data),"style":style,"tasks":tasks,"batch_size":1,"num_workers":c["execution"]["num_workers"],"seed":c["execution"]["seed"],"model":{"_target_":"pilot_p8.P7CheckpointAdapter","name":"p7-pilot","kwargs":kwargs}})
        out=MachineDevBenchEvalModule(cfg).run(); per[style]=out["results"]; raw[style]=out["raw_records"]
        for i,r in enumerate(out["raw_records"]):
            key=(style,r["task_name"],i); require(key not in seen,"duplicate_example"); seen.add(key); pooled.add(r["task_name"],r["prediction"],r["target"],r["metadata"])
    total=pooled.compute(); require(set(total["by_task"])==set(tasks) and all(total["by_task"][t]["n_trials"]>0 for t in tasks),"coverage")
    vals={t:round(total["by_task"][t]["accuracy"]*100,6) for t in tasks}; vals.update({"lexical":round(total["by_task_type"]["lexical"]["accuracy"]*100,6),"grammatical":round(total["by_task_type"]["grammatical"]["accuracy"]*100,6),"overall":round(total["overall"]["accuracy"]*100,6)}); require(all(math.isfinite(x) for x in vals.values()),"finite_scores")
    detail={"schema_version":1,"record_type":"pilot_p8_governed_result","run_id":c["engineering_run_id"],"classification":LABELS,"results":total,"per_style":per,"raw_predictions":raw,"coverage":{"task_count":10,"manifest_count":20,"trial_count":len(seen),"duplicate_count":0},"runtime_seconds":time.monotonic()-began,"slurm_job_id":job,"archive_inventory":inv}
    atomic_json(a.record_root/"result.json",detail); result_hash=sha256(a.record_root/"result.json")
    atomic_json(ledger,{**json.loads(ledger.read_text()),"state":"scoring_complete","scoring_complete_count":1,"completed_unix":time.time(),"result_sha256":result_hash})
    aggregate={"schema_version":1,"record_type":"pilot_p8_privacy_safe_aggregate","stage":"P8","status":"p8_complete","classification":LABELS,"results_percent":vals,"coverage":detail["coverage"],"runtime_seconds":detail["runtime_seconds"],"memory":{"driver_bytes":None},"execution":{"attempt_count":1,"benchmark_accessed":True,"slurm_job_id":job},"selected_checkpoint":c["selected_checkpoint"],"governed_records":{"result_sha256":result_hash,"ledger_sha256":sha256(ledger),"owner_only":True},"limitations":["Two-video pilot and 380-token vocabulary are non-generalizing and non-comparable.","P4 CLIP-L calibrated only the evaluator and aggregation; it did not initialize, tune, alter, ensemble with, or score P7.","The full reproduction must recalibrate the evaluator without leakage."],"scientific_status_effect":"none","next_stage":"P9","next_stage_started":False}
    atomic_json(a.durable_root/"aggregate_results"/(c["engineering_run_id"]+".json"),aggregate); print(json.dumps(aggregate,sort_keys=True))
def main():
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=("qualify","evaluate"));
    for n in ("config","scratch_root","durable_root","source_root","checkpoint","p7_config","p5_training_dir","p6_checkpoint","tokenizer","vocab","p7_completion","p7_audit","p7_inventory","p4_selection","record_root","qualification"): p.add_argument("--"+n.replace("_","-"),dest=n,type=Path,required=n!="qualification")
    a=p.parse_args(); c=load_config(a.config); a.record_root.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(a.record_root,0o700)
    (qualify if a.phase=="qualify" else evaluate)(c,a)
if __name__=="__main__": main()
