#!/usr/bin/env python3
"""Fail-closed Pilot P7: one exact, resumable multimodal CLIP+ cycle."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, stat, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any

LABELS=["engineering_only","non_comparable","not_a_reproduction_result"]
RUN_ID="p7-7b9e2c41"
class P7Error(RuntimeError): pass
def require(v: Any,m: str)->None:
    if not v: raise P7Error(m)
def sha256(p: Path)->str:
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
def owner(p:Path,directory=True)->None:
    require(p.is_absolute() and p.exists(),"governed_path_missing")
    require(stat.S_IMODE(p.stat().st_mode)==(0o700 if directory else 0o600),"governed_mode_failure")
def state_hash(torch:Any,state:dict[str,Any])->str:
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());
        if torch.is_tensor(v): h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def config(path:Path)->dict[str,Any]:
    c=json.loads(path.read_text()); require(c["stage"]=="P7" and c["engineering_run_id"]==RUN_ID and c["classification"]==LABELS,"config_identity")
    s=c["schedule"]; modes=s["cycle_modes"]*s["repetitions"]
    require(len(modes)==130 and {x:modes.count(x) for x in set(modes)}==s["exact_counts"],"schedule_not_exact")
    require(s["checkpoint_after_global_update"]==65 and {x:modes[:65].count(x) for x in set(modes)}==s["checkpoint_counts"],"boundary_not_exact")
    require(c["batch"]["microbatch"]==2 and c["batch"]["distinct_examples_required"],"contrastive_negative_semantics")
    learned=json.dumps({"source":c["source"],"prerequisites":c["prerequisites"],"model":c["model"]}).lower(); require("machine-devbench" not in learned and "clip-l" not in learned and "clip-b" not in learned,"forbidden_calibration_or_evaluation_reference")
    return c
def paths(c,args):
    scratch=args.scratch_root.resolve(); durable=args.durable_root.resolve(); owner(scratch); owner(durable)
    run=scratch/"runs"/c["storage"]["scratch_namespace"]; rec=durable/"run_records"/c["storage"]["durable_namespace"]
    require(run==args.run_root.resolve() and rec==args.record_root.resolve(),"namespace_mismatch")
    return run,rec
def verify(c,args,run,rec,reject_completed=True):
    if reject_completed: require(not (rec/"completion.json").exists(),"P7_already_complete")
    require(subprocess.run(["git","-C",str(args.source_root),"rev-parse","HEAD"],text=True,capture_output=True,check=True).stdout.strip()==c["source"]["commit"],"source_commit")
    require(not subprocess.run(["git","-C",str(args.source_root),"status","--porcelain"],text=True,capture_output=True,check=True).stdout.strip(),"source_dirty")
    require(sha256(args.source_root/"pixi.lock")==c["source"]["pixi_lock_sha256"],"lock_hash")
    for rel,want in c["source"]["required_files"].items(): require(sha256(args.source_root/rel)==want,"source_file_hash")
    checks=[(args.p5_checkpoint,c["prerequisites"]["p5_checkpoint_sha256"]),(args.p6_checkpoint,c["prerequisites"]["p6_checkpoint_sha256"]),(args.tokenizer,c["prerequisites"]["p6_tokenizer_sha256"]),(args.vocab,c["prerequisites"]["p6_vocab_sha256"])]
    for p,w in checks: owner(p.resolve(),False); require(sha256(p)==w,"prerequisite_hash")
    require(args.p5_training_dir.resolve().is_dir() and sha256(args.p5_training_dir/"model_final.rank_0.pth")==c["prerequisites"]["p5_checkpoint_sha256"],"P5_resume_source_hash")
    manifests=sorted(args.p3_root.glob("*/*/manifests/paired.json")); require(len(manifests)==2,"pair_manifests")
    by_count={len(json.loads(p.read_text())):p for p in manifests}; require(set(by_count)=={1,80},"pair_counts")
    require(sha256(by_count[80])==c["prerequisites"]["training_pair_manifest_sha256"] and sha256(by_count[1])==c["prerequisites"]["validation_pair_manifest_sha256"],"pair_hash")
    run.mkdir(parents=True,exist_ok=True,mode=0o700); rec.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(run,0o700); os.chmod(rec,0o700)
    return by_count[80]
def imports(source):
    sys.path[:0]=[str(source),str(source/"apps/baselines/dinov2/third_party")]
    import numpy as np, torch
    from PIL import Image
    from omegaconf import OmegaConf
    from torchvision import transforms
    from transformers import BertConfig,BertForMaskedLM,BertTokenizerFast
    from apps.baselines.clip.modeling.dinov2_ssl import DINOv2SSL
    from apps.baselines.clip.modeling.vision_encoder import CustomDINOv2VisionEncoder
    return np,torch,Image,OmegaConf,transforms,BertConfig,BertForMaskedLM,BertTokenizerFast,DINOv2SSL,CustomDINOv2VisionEncoder
def make_model(c,args,lib):
    np,torch,Image,OmegaConf,T,BertConfig,BertForMaskedLM,BertTokenizerFast,DINOv2SSL,Vision=lib
    p6=torch.load(args.p6_checkpoint,map_location="cpu",weights_only=False); require(p6["optimizer_step"]==100,"P6_step")
    bc=BertConfig(vocab_size=380,hidden_size=768,num_hidden_layers=12,num_attention_heads=12,intermediate_size=3072,hidden_act="gelu",hidden_dropout_prob=.1,attention_probs_dropout_prob=.1,max_position_embeddings=512,type_vocab_size=2,initializer_range=.02,layer_norm_eps=1e-12,position_embedding_type="absolute",pad_token_id=0)
    bert=BertForMaskedLM(bc); bert.load_state_dict(p6["model"],strict=True)
    tok=BertTokenizerFast(vocab_file=str(args.vocab),do_lower_case=False,pad_token="[PAD]",unk_token="[UNK]",cls_token="[CLS]",sep_token="[SEP]",mask_token="[MASK]"); require(len(tok)==380,"tokenizer_size")
    dcfg=args.source_root/"apps/baselines/dinov2/third_party/dinov2/configs/train/vitb14_babyview.yaml"
    overrides={"train":{"batch_size_per_gpu":2,"OFFICIAL_EPOCH_LENGTH":10,"num_workers":0},"optim":{"epochs":1,"warmup_epochs":1},"teacher":{"warmup_teacher_temp_epochs":1}}
    ssl=DINOv2SSL(dcfg,device="cuda",overrides=overrides,pretrained_dir=args.p5_training_dir)
    vision=Vision(dinov2_config=OmegaConf.to_container(ssl.cfg,resolve=False),embedding_dim=512,freeze=False); vision.backbone.load_state_dict(ssl.teacher_backbone_state_dict(),strict=True)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.vision=vision; self.bert=bert.bert; self.mlm_head=bert.cls; self.vision_projection=vision.projection
            self.text_projection=torch.nn.Linear(768,512); self.logit_scale=torch.nn.Parameter(torch.tensor(-math.log(.07)))
        def contrastive(self,images,encoded):
            vi=torch.nn.functional.normalize(self.vision(images),dim=-1); out=self.bert(input_ids=encoded["input_ids"],attention_mask=encoded["attention_mask"])
            tx=torch.nn.functional.normalize(self.text_projection(out.last_hidden_state[:,0]),dim=-1); logits=vi@tx.T*self.logit_scale.exp(); target=torch.arange(len(images),device=images.device)
            return (torch.nn.functional.cross_entropy(logits,target)+torch.nn.functional.cross_entropy(logits.T,target))/2
        def mlm(self,b):
            hidden=self.bert(input_ids=b["input_ids"],attention_mask=b["attention_mask"]).last_hidden_state
            return torch.nn.functional.cross_entropy(self.mlm_head(hidden).view(-1,380),b["labels"].view(-1),ignore_index=-100)
    model=Model().cuda(); return model,tok,ssl
def paired_data(manifest,Image,T):
    rows=json.loads(manifest.read_text()); frame_root=manifest.parent.parent/"frames"
    out=[]
    for row in rows:
        require(row["frame_indices"],"pair_without_frame"); idx=int(row["frame_indices"][0]); p=frame_root/f"frame_{idx:08d}.jpg"; require(p.is_file(),"paired_frame_missing"); out.append((p,row["text"]))
    tf=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]); return out,tf
def encode_batch(torch,tok,texts,device): return tok(texts,padding=True,truncation=True,max_length=64,return_tensors="pt").to(device)
def mlm_batch(torch,tok,text,index,generator,device):
    b=encode_batch(torch,tok,[text],device); ids=b["input_ids"].cpu().clone(); labels=ids.clone(); special=torch.tensor([tok.get_special_tokens_mask(x.tolist(),already_has_special_tokens=True) for x in labels],dtype=torch.bool)
    mask=torch.bernoulli(torch.full(labels.shape,.15).masked_fill(special,0),generator=generator).bool()
    if not mask.any(): mask[tuple((~special).nonzero()[0].tolist())]=True
    labels[~mask]=-100; rep=torch.bernoulli(torch.full(labels.shape,.8),generator=generator).bool()&mask; ids[rep]=tok.mask_token_id
    rnd=torch.bernoulli(torch.full(labels.shape,.5),generator=generator).bool()&mask&~rep; words=torch.randint(380,labels.shape,generator=generator); ids[rnd]=words[rnd]
    return {"input_ids":ids.to(device),"attention_mask":b["attention_mask"],"labels":labels.to(device)}
def snapshot(torch,model,ssl):
    return {"vision":state_hash(torch,model.vision.backbone.state_dict()),"vision_projection":state_hash(torch,model.vision.projection.state_dict()),"text":state_hash(torch,model.bert.state_dict()),"text_projection":state_hash(torch,model.text_projection.state_dict()),"mlm_head":state_hash(torch,model.mlm_head.state_dict()),"mlm_bias":hashlib.sha256(model.mlm_head.predictions.bias.detach().cpu().numpy().tobytes()).hexdigest(),"temperature":hashlib.sha256(model.logit_scale.detach().cpu().numpy().tobytes()).hexdigest(),"ssl_student":state_hash(torch,ssl.model.student.state_dict()),"ssl_teacher":state_hash(torch,ssl.model.teacher.state_dict())}
def save(torch,p,payload):
    p.parent.mkdir(parents=True,exist_ok=True,mode=0o700); t=p.with_suffix(".tmp"); torch.save(payload,t); os.chmod(t,0o600); os.replace(t,p)
def validate_checkpoint(torch,st,c,args,manifest,executed_config,expected_step):
    required={"model","ssl","contrastive_optimizer","mlm_optimizer","contrastive_scheduler","mlm_scheduler","global_update","counts","next_mode","history","python_rng","numpy_rng","torch_cpu_rng","torch_cuda_rng","sampler_rng","tokenizer_sha256","manifest_sha256","p5_sha256","p6_sha256","config_sha256","source_commit","environment_sha256","slurm_job_id"}
    require(required<=set(st),"checkpoint_schema_missing:"+",".join(sorted(required-set(st))))
    require(isinstance(st["model"],dict) and st["model"] and isinstance(st["ssl"],dict) and {"student","teacher","optimizer","scaler"}<=set(st["ssl"]),"model_or_SSL_schema")
    require(all(isinstance(st[k],dict) and st[k] for k in ("contrastive_optimizer","mlm_optimizer","contrastive_scheduler","mlm_scheduler")),"optimizer_or_scheduler_schema")
    require(st["python_rng"] is not None and st["numpy_rng"] is not None and torch.is_tensor(st["torch_cpu_rng"]) and isinstance(st["torch_cuda_rng"],list) and st["torch_cuda_rng"] and torch.is_tensor(st["sampler_rng"]),"RNG_or_sampler_schema")
    modes=c["schedule"]["cycle_modes"]*c["schedule"]["repetitions"]; expected_history=modes[:expected_step]; expected_counts={k:expected_history.count(k) for k in c["schedule"]["exact_counts"]}
    require(st["global_update"]==expected_step and st["counts"]==expected_counts and st["history"]==expected_history and st["next_mode"]==(modes[expected_step] if expected_step<len(modes) else None),"checkpoint_counter_history_or_next_mode")
    expected={"tokenizer_sha256":sha256(args.tokenizer),"manifest_sha256":sha256(manifest),"p5_sha256":sha256(args.p5_checkpoint),"p6_sha256":sha256(args.p6_checkpoint),"config_sha256":sha256(executed_config),"source_commit":c["source"]["commit"],"environment_sha256":sha256(args.source_root/"pixi.lock")}
    for key,want in expected.items(): require(st[key]==want,"checkpoint_saved_hash_mismatch:"+key)
    require(str(st["slurm_job_id"]).isdigit(),"checkpoint_originating_slurm_id")
    require(st["contrastive_scheduler"].get("last_epoch")==expected_counts["contrastive"] and st["mlm_scheduler"].get("last_epoch")==expected_counts["mlm"],"checkpoint_scheduler_counters")
    return {"step":expected_step,"counts":expected_counts,"originating_slurm_job_id":st["slurm_job_id"],"all_saved_hashes_verified":True,"all_applicable_state_schema_verified":True}
def train(args,c,run,manifest,lib):
    np,torch,Image,OmegaConf,T,*_=lib; model,tok,ssl=make_model(c,args,lib); data,tf=paired_data(manifest,Image,T); require(len(data)>=2,"no_genuine_negatives")
    modes=c["schedule"]["cycle_modes"]*10; target=args.target; resume=args.resume is not None
    contrast_params=list(model.vision.parameters())+list(model.bert.parameters())+list(model.text_projection.parameters())+[model.logit_scale]
    mlm_params=list({id(p):p for p in list(model.bert.parameters())+list(model.mlm_head.parameters())}.values())
    co=torch.optim.AdamW(contrast_params,lr=c["optimization"]["contrastive_learning_rate"],weight_decay=.01); mo=torch.optim.AdamW(mlm_params,lr=c["optimization"]["mlm_learning_rate"],weight_decay=.01)
    cs=torch.optim.lr_scheduler.LambdaLR(co,lambda n:min((n+1)/2,max(0,(100-n)/98))); ms=torch.optim.lr_scheduler.LambdaLR(mo,lambda n:min((n+1)/2,max(0,(20-n)/18)))
    gen=torch.Generator().manual_seed(c["optimization"]["seed"]); start=0; counts={"contrastive":0,"mlm":0,"dinov2":0}; history=[]
    if resume:
        executed=args.executed_config or (args.durable_root/"checkpoints/pilot_p7"/RUN_ID/"pilot_p7.json"); owner(executed.resolve(),False)
        st=torch.load(args.resume,map_location="cpu",weights_only=False); validate_checkpoint(torch,st,c,args,manifest,executed,65)
        model.load_state_dict(st["model"]); ssl.load_state_dict(st["ssl"]); co.load_state_dict(st["contrastive_optimizer"]); mo.load_state_dict(st["mlm_optimizer"]); cs.load_state_dict(st["contrastive_scheduler"]); ms.load_state_dict(st["mlm_scheduler"])
        random.setstate(st["python_rng"]); np.random.set_state(st["numpy_rng"]); torch.set_rng_state(st["torch_cpu_rng"]); torch.cuda.set_rng_state_all(st["torch_cuda_rng"]); gen.set_state(st["sampler_rng"]); start=st["global_update"]; counts=st["counts"]; history=st["history"]
    initial=snapshot(torch,model,ssl); p5_teacher_hash=state_hash(torch,ssl.teacher_backbone_state_dict()); require(initial["vision"]==p5_teacher_hash,"P5_teacher_not_copied_to_vision")
    losses={"contrastive":[],"mlm":[],"dinov2":[]}; assertions=[]; torch.cuda.reset_peak_memory_stats(); began=time.monotonic()
    for pos in range(start,target):
        mode=modes[pos]; before=snapshot(torch,model,ssl); i0=(2*counts[mode])%len(data); ids=[i0,(i0+1)%len(data)]; require(ids[0]!=ids[1],"fake_negative")
        images=torch.stack([tf(Image.open(data[i][0]).convert("RGB")) for i in ids]).cuda(); texts=[data[i][1] for i in ids]
        if mode=="contrastive":
            co.zero_grad(set_to_none=True); loss=model.contrastive(images,encode_batch(torch,tok,texts,"cuda")); require(torch.isfinite(loss),"nonfinite_contrastive"); loss.backward(); require(model.logit_scale.grad is not None and torch.isfinite(model.logit_scale.grad) and model.vision.projection.weight.grad is not None and model.text_projection.weight.grad is not None,"contrastive_gradient_reachability"); torch.nn.utils.clip_grad_norm_(contrast_params,3); co.step(); cs.step()
        elif mode=="mlm":
            mo.zero_grad(set_to_none=True); loss=model.mlm(mlm_batch(torch,tok,texts[0],counts[mode],gen,"cuda")); require(torch.isfinite(loss),"nonfinite_mlm"); loss.backward(); torch.nn.utils.clip_grad_norm_(mlm_params,3); mo.step(); ms.step()
        else:
            result=ssl.step(ssl.prepare_batch(images),counts[mode]); require(all(math.isfinite(float(v)) for v in result.values()),"nonfinite_dino"); loss=torch.tensor(result["total_loss"]); model.vision.backbone.load_state_dict(ssl.teacher_backbone_state_dict(),strict=True)
        after=snapshot(torch,model,ssl); changed={k:before[k]!=after[k] for k in before};
        if mode=="contrastive": require(not any(changed[k] for k in ("mlm_bias","ssl_student","ssl_teacher")),"contrastive_forbidden_update_contract")
        elif mode=="mlm": require(changed["text"] and changed["mlm_head"] and changed["mlm_bias"] and not any(changed[k] for k in ("vision","vision_projection","text_projection","temperature","ssl_student","ssl_teacher")),"mlm_component_update_contract")
        else: require(changed["ssl_teacher"] and changed["vision"] and not any(changed[k] for k in ("vision_projection","text","text_projection","mlm_head","mlm_bias","temperature")),f"dino_component_update_contract:{changed}")
        counts[mode]+=1; losses[mode].append(float(loss)); history.append(mode); assertions.append(changed)
    final_snapshot=snapshot(torch,model,ssl); require(final_snapshot["ssl_student"]!=initial["ssl_student"],"DINO_student_did_not_update_across_segment"); require(all(final_snapshot[k]!=initial[k] for k in ("vision_projection","text_projection","temperature","text","mlm_bias","ssl_teacher")),"intended_components_did_not_update_across_segment")
    job=os.environ.get("SLURM_JOB_ID",""); require(job.isdigit(),"slurm_job_required"); elapsed=time.monotonic()-began
    payload={"model":model.state_dict(),"ssl":ssl.state_dict(),"contrastive_optimizer":co.state_dict(),"mlm_optimizer":mo.state_dict(),"contrastive_scheduler":cs.state_dict(),"mlm_scheduler":ms.state_dict(),"global_update":target,"counts":counts,"next_mode":modes[target] if target<len(modes) else None,"history":history,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_cpu_rng":torch.get_rng_state(),"torch_cuda_rng":torch.cuda.get_rng_state_all(),"sampler_rng":gen.get_state(),"tokenizer_sha256":sha256(args.tokenizer),"manifest_sha256":sha256(manifest),"p5_sha256":sha256(args.p5_checkpoint),"p6_sha256":sha256(args.p6_checkpoint),"config_sha256":sha256(args.config),"source_commit":c["source"]["commit"],"environment_sha256":c["source"]["pixi_lock_sha256"],"slurm_job_id":job}
    cp=run/"checkpoints"/f"step_{target}.pt"; save(torch,cp,payload)
    atomic_json(run/f"segment_{target}.json",{"job_id":job,"start":start,"end":target,"counts":counts,"losses":losses,"all_losses_finite":True,"component_assertions_passed":True,"initial_p5_teacher_vision_hash":p5_teacher_hash,"peak_allocated_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"wall_seconds":elapsed,"checkpoint_sha256":sha256(cp),"checkpoint_bytes":cp.stat().st_size})
def finalize(args,c,run,rec,manifest,lib):
    np,torch,*_=lib; executed=args.executed_config or (args.durable_root/"checkpoints/pilot_p7"/RUN_ID/"pilot_p7.json"); owner(executed.resolve(),False); st=torch.load(run/"checkpoints/step_130.pt",map_location="cpu",weights_only=False); validate_checkpoint(torch,st,c,args,manifest,executed,130)
    require(st["history"]==c["schedule"]["cycle_modes"]*10,"mode_transition_history")
    seg65=json.loads((run/"segment_65.json").read_text()); seg130=json.loads((run/"segment_130.json").read_text()); final_job=os.environ.get("SLURM_JOB_ID",""); jobs=[seg65["job_id"],seg130["job_id"],final_job]; require(len(set(jobs))==3 and all(x.isdigit() for x in jobs),"three_distinct_jobs")
    model,tok,ssl=make_model(c,args,lib); model.load_state_dict(st["model"]); ssl.load_state_dict(st["ssl"]); data,tf=paired_data(manifest,lib[2],lib[4]); model.eval()
    with torch.no_grad(): smoke=model.contrastive(torch.stack([tf(lib[2].open(data[i][0]).convert("RGB")) for i in (0,1)]).cuda(),encode_batch(torch,tok,[data[0][1],data[1][1]],"cuda"))
    require(torch.isfinite(smoke),"fresh_load_smoke")
    promoted=args.durable_root/"checkpoints/pilot_p7"/RUN_ID; promoted.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(promoted,0o700)
    for src in (run/"checkpoints/step_65.pt",run/"checkpoints/step_130.pt",args.config): shutil.copy2(src,promoted/src.name); os.chmod(promoted/src.name,0o600); require(sha256(src)==sha256(promoted/src.name),"promotion_hash")
    all_losses={m:seg65["losses"][m]+seg130["losses"][m] for m in c["schedule"]["exact_counts"]}; wall=seg65["wall_seconds"]+seg130["wall_seconds"]
    detail={"schema_version":1,"record_type":"pilot_p7_governed_detail","status":"p7_complete","run_id":RUN_ID,"slurm_job_ids":{"health":jobs[0],"resume":jobs[1],"finalize":jobs[2]},"schedule":c["schedule"],"counts":st["counts"],"losses":all_losses,"resume":{"exact_boundary":65,"next_mode":"contrastive","no_repeated_or_skipped_transitions":True,"all_checkpoint_state_restored":True},"verification":{"P5_teacher_copied_to_vision_at_segment_start":True,"all_losses_finite":True,"component_update_and_forbidden_update_assertions":True,"scheduler_completion":{"contrastive":st["contrastive_scheduler"]["last_epoch"],"mlm":st["mlm_scheduler"]["last_epoch"]},"fresh_load_smoke":True},"resources":{"peak_allocated_gpu_memory_bytes":max(seg65["peak_allocated_gpu_memory_bytes"],seg130["peak_allocated_gpu_memory_bytes"]),"scheduler_driver_memory_bytes":None,"training_wall_seconds":wall,"updates_per_second":130/wall,"checkpoint_bytes":seg130["checkpoint_bytes"]},"classification":LABELS,"scientific_status_effect":"none"}
    atomic_json(rec/"detail.json",detail); inv={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in promoted.iterdir()}; atomic_json(rec/"inventory.json",{"schema_version":1,"files":inv})
    completion={"schema_version":1,"status":"p7_complete","run_id":RUN_ID,"detail_sha256":sha256(rec/"detail.json"),"inventory_sha256":sha256(rec/"inventory.json"),"final_checkpoint_sha256":sha256(promoted/"step_130.pt"),"owner_only":True}; atomic_json(rec/"completion.json",completion)
    aggregate={"schema_version":1,"record_type":"pilot_p7_privacy_safe_aggregate","pilot_id":"juno_sample","engineering_run_id":RUN_ID,"stage":"P7","status":"p7_complete","classification":LABELS,"scientific_status_effect":"none","inventory_status":"incomplete_inventory","question_answer":"yes_full_multimodal_control_flow_passed_with_pilot_towers","schedule":{"counts":st["counts"],"total":130,"checkpoint_boundary":65},"execution":{"batch":2,"genuine_contrastive_negatives":True,"slurm_job_ids":detail["slurm_job_ids"],"p8_started":False},"losses":{m:{"count":len(v),"min":min(v),"max":max(v),"final":v[-1]} for m,v in all_losses.items()},"verification":detail["verification"],"resume":detail["resume"],"resources":detail["resources"],"governed_records":completion,"limitations":["Engineering-only two-video pilot; no generalization or performance conclusion.","Machine-DevBench was not accessed and P4 CLIP-L was not used; full reproduction evaluator calibration remains required.","Driver-reported memory was unavailable and remains null."],"p0_p1_p2_p3_p4_p5_p6_status_preserved":True,"next_stage":"P8","next_stage_started":False}
    atomic_json(args.aggregate_output,aggregate)
def audit_existing(args,c,run,rec,manifest,lib):
    np,torch,Image,OmegaConf,T,*_=lib; require((rec/"completion.json").is_file(),"completed_record_missing")
    promoted=args.durable_root/"checkpoints/pilot_p7"/RUN_ID; executed=promoted/"pilot_p7.json"; owner(executed,False)
    states={}; validations={}
    for step in (65,130):
        cp=promoted/f"step_{step}.pt"; owner(cp,False); states[step]=torch.load(cp,map_location="cpu",weights_only=False); validations[str(step)]=validate_checkpoint(torch,states[step],c,args,manifest,executed,step)
    require(states[65]["slurm_job_id"]=="324893" and states[130]["slurm_job_id"]=="324896","canonical_checkpoint_job_ids")
    model,tok,ssl=make_model(c,args,lib); st=states[130]
    contrast_params=list(model.vision.parameters())+list(model.bert.parameters())+list(model.text_projection.parameters())+[model.logit_scale]
    mlm_params=list({id(p):p for p in list(model.bert.parameters())+list(model.mlm_head.parameters())}.values())
    co=torch.optim.AdamW(contrast_params,lr=c["optimization"]["contrastive_learning_rate"],weight_decay=.01); mo=torch.optim.AdamW(mlm_params,lr=c["optimization"]["mlm_learning_rate"],weight_decay=.01)
    cs=torch.optim.lr_scheduler.LambdaLR(co,lambda n:min((n+1)/2,max(0,(100-n)/98))); ms=torch.optim.lr_scheduler.LambdaLR(mo,lambda n:min((n+1)/2,max(0,(20-n)/18)))
    model.load_state_dict(st["model"],strict=True); ssl.load_state_dict(st["ssl"]); co.load_state_dict(st["contrastive_optimizer"]); mo.load_state_dict(st["mlm_optimizer"]); cs.load_state_dict(st["contrastive_scheduler"]); ms.load_state_dict(st["mlm_scheduler"])
    random.setstate(st["python_rng"]); np.random.set_state(st["numpy_rng"]); torch.set_rng_state(st["torch_cpu_rng"]); torch.cuda.set_rng_state_all(st["torch_cuda_rng"]); gen=torch.Generator(); gen.set_state(st["sampler_rng"])
    require(cs.last_epoch==100 and ms.last_epoch==20 and len(co.state)>0 and len(mo.state)>0 and len(ssl.optimizer.state)>0,"restored_state_not_operational")
    data,tf=paired_data(manifest,Image,T); before=snapshot(torch,model,ssl); model.eval(); co.zero_grad(set_to_none=True)
    images=torch.stack([tf(Image.open(data[i][0]).convert("RGB")) for i in (0,1)]).cuda(); encoded=encode_batch(torch,tok,[data[0][1],data[1][1]],"cuda"); loss=model.contrastive(images,encoded); require(torch.isfinite(loss),"audit_nonfinite_loss"); loss.backward()
    named={"vision_backbone":next(p for p in model.vision.backbone.parameters() if p.requires_grad),"bert_backbone":next(p for p in model.bert.parameters() if p.requires_grad),"vision_projection":model.vision.projection.weight,"text_projection":model.text_projection.weight,"temperature":model.logit_scale}
    gradients={k:{"present":p.grad is not None,"finite":bool(p.grad is not None and torch.isfinite(p.grad).all()),"nonzero":bool(p.grad is not None and torch.count_nonzero(p.grad)>0)} for k,p in named.items()}; require(all(v["present"] and v["finite"] and v["nonzero"] for v in gradients.values()),"backbone_or_projection_gradient_gate")
    after=snapshot(torch,model,ssl); require(before==after,"gradient_audit_mutated_model_or_auxiliary_state")
    audit_job=os.environ.get("SLURM_JOB_ID",""); require(audit_job.isdigit() and audit_job not in {"324893","324896","324912"},"distinct_audit_job")
    evidence=rec/"evidence"; evidence.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(evidence,0o700)
    for name in ("segment_65.json","segment_130.json"):
        shutil.copy2(run/name,evidence/name); os.chmod(evidence/name,0o600)
    att={"schema_version":1,"record_type":"pilot_p7_preserved_artifact_hardening_audit","training_updates_executed":0,"canonical_jobs":{"health":"324893","resume":"324896","finalize":"324912","audit":audit_job},"checkpoint_validations":validations,"fresh_restore":{"model":True,"ssl_student_teacher_ema":True,"contrastive_optimizer":True,"mlm_optimizer":True,"contrastive_scheduler":True,"mlm_scheduler":True,"torch_cpu_cuda_rng":True,"python_numpy_rng":True,"sampler":True,"claim":"all serialized applicable state passed schema, provenance, counter, and operational load validation; only RNG values were set, not replay-compared"},"contrastive_gradient_audit":{"loss_finite":True,"gradients":gradients,"optimizer_step":False,"model_or_auxiliary_state_mutated":False},"owner_only":True}
    atomic_json(rec/"hardening_audit.json",att)
    print(json.dumps({"audit_job_id":audit_job,"hardening_audit_sha256":sha256(rec/"hardening_audit.json"),"gradient_evidence":gradients},sort_keys=True))
def inventory_existing(args,c,run,rec):
    att=json.loads((rec/"hardening_audit.json").read_text()); audit_job=str(att["canonical_jobs"]["audit"]); require(audit_job.isdigit(),"audit_job_missing")
    promoted=args.durable_root/"checkpoints/pilot_p7"/RUN_ID
    evidence=rec/"evidence"; accounting=subprocess.run(["sacct","-j","324893,324896,324912,"+audit_job,"--noheader","-o","JobID,State,Elapsed,ExitCode,TRESUsageInMax","-P"],check=True,text=True,capture_output=True).stdout
    for job in ("324893","324896","324912",audit_job): require(f"{job}|COMPLETED|" in accounting,"completed_slurm_accounting_missing:"+job)
    (evidence/"slurm_accounting.txt").write_text(accounting); os.chmod(evidence/"slurm_accounting.txt",0o600)
    logroot=args.durable_root/"logs/pilot_p7"/RUN_ID; required_logs={"canonical_health":"health-324893.log","canonical_resume":"resume-324896.log","canonical_finalize":"finalize-324912.log","audit":f"audit-{audit_job}.log"}; failed=["health-324887.log","health-324888.log","health-324889.log","health-324890.log","health-324892.log","resume-324894.log","audit-326917.log"]
    entries=[]
    def add(role,p,excluded=False): owner(p,False); entries.append({"role":role,"opaque_name":p.name,"sha256":sha256(p),"bytes":p.stat().st_size,"excluded_diagnostic_provenance":excluded})
    for p in sorted(promoted.iterdir()): add("promoted_checkpoint_or_executed_config",p)
    for p in (rec/"detail.json",rec/"completion.json",rec/"hardening_audit.json",evidence/"segment_65.json",evidence/"segment_130.json",evidence/"slurm_accounting.txt"): add("governed_record_or_evidence",p)
    for role,name in required_logs.items(): add("important_slurm_log:"+role,logroot/name)
    for name in failed: add("excluded_invalid_diagnostic_log",logroot/name,True)
    inventory={"schema_version":2,"record_type":"pilot_p7_complete_retained_artifact_inventory","scope":"all retained P7 promoted checkpoints/config, governed records, promoted segment evidence, canonical/audit Slurm logs and accounting, and excluded invalid diagnostic logs","p7_retained_artifact_inventory_status":"complete","global_full_dataset_inventory_status":"incomplete_and_out_of_scope","entries":entries,"entry_count":len(entries),"owner_only_verified":True}
    atomic_json(rec/"retained_inventory.json",inventory)
    print(json.dumps({"audit_job_id":audit_job,"hardening_audit_sha256":sha256(rec/"hardening_audit.json"),"retained_inventory_sha256":sha256(rec/"retained_inventory.json"),"entry_count":len(entries)},sort_keys=True))
def main():
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=("health","resume","finalize","audit-existing","inventory-existing")); p.add_argument("--target",type=int,choices=(65,130)); p.add_argument("--executed-config",type=Path)
    for n in ("config","source_root","p3_root","p5_checkpoint","p5_training_dir","p6_checkpoint","tokenizer","vocab","scratch_root","durable_root","run_root","record_root","aggregate_output"): p.add_argument("--"+n.replace("_","-"),dest=n,type=Path,required=True)
    a=p.parse_args(); a.resume=None; c=config(a.config); run,rec=paths(c,a); manifest=verify(c,a,run,rec,reject_completed=a.phase not in ("audit-existing","inventory-existing")); lib=None if a.phase=="inventory-existing" else imports(a.source_root)
    if a.phase=="health": train(a,c,run,manifest,lib)
    elif a.phase=="resume": a.resume=run/"checkpoints/step_65.pt"; train(a,c,run,manifest,lib)
    elif a.phase=="finalize": finalize(a,c,run,rec,manifest,lib)
    elif a.phase=="audit-existing": audit_existing(a,c,run,rec,manifest,lib)
    else: inventory_existing(a,c,run,rec)
if __name__=="__main__": main()
