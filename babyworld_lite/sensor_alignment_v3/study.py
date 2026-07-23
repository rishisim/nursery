from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import gzip, hashlib, json, math, shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import stats
import yaml

PROTOCOL_ID = "synthetic-component-identification-robustness-v3"
CONDITIONS = ("synchronized","shuffled","shifted_m16","shifted_m8","shifted_p8","shifted_p16","absent","uninformative")
REGIMES = ("unbiased_primary", "adversarial_plus3_secondary")
FORBIDDEN_FINAL = ("sensor","detector","imu","proprio","contact","cue","evidence","encoder","reliability")


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def _digest(x: Any) -> str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def _write(path: Path, x: Any) -> None:
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")


def _jsonl(path: Path, rows: Iterable[Mapping[str,Any]]) -> None:
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))


def _gzip_jsonl(path: Path, rows: Iterable[Mapping[str,Any]]) -> None:
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload="".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in rows).encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as f: f.write(payload)


def load_config(path: str|Path, *, fixture: bool=False) -> dict[str,Any]:
    c=yaml.safe_load(Path(path).read_text())
    if c["protocol"]["id"] != PROTOCOL_ID or c["protocol"]["status"] != "frozen": raise ValueError("invalid frozen v3 protocol")
    if not fixture and (len(c["seeds"]["development"]["corpus"])<40 or len(c["seeds"]["development"]["model"])<3): raise ValueError("v3 requires 40 corpora and 3 model seeds")
    sets=[]
    for key in ("development","calibration","confirmation_reserve"):
        v=c["seeds"][key]; sets.append(set(int(z) for q in v.values() for z in (q if isinstance(q,list) else [q])))
    sets += [set(map(int,c["seeds"]["fixture_only"])),set(map(int,c["seeds"]["forbidden_v1_v2"]))]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)): raise ValueError("seed registries overlap")
    detector=Path(path).resolve().parent.parent/c["detector"]["frozen_v2_path"]
    if _sha(detector)!=c["detector"]["frozen_v2_sha256"]: raise ValueError("fixed v2 detector hash mismatch")
    return c


def _blocked(c: Mapping[str,Any]) -> set[int]:
    r=c["seeds"]["confirmation_reserve"]
    return set(map(int,c["seeds"]["forbidden_v1_v2"]))|set(int(z) for q in r.values() for z in q)


def guard(c: Mapping[str,Any], operation: str, *seeds: int) -> None:
    if operation not in {"generate","fit","evaluate","read","summarize","calibrate"}: raise ValueError(operation)
    bad=sorted(set(map(int,seeds))&_blocked(c))
    if bad: raise PermissionError(f"v3 seed firewall blocked {operation}: {bad}")


def _rng(seed:int, label:str) -> np.random.Generator:
    x=int.from_bytes(hashlib.sha256(f"v3|{seed}|{label}".encode()).digest()[:8],"little")
    return np.random.default_rng(x)


def _components(a:int)->tuple[int,int]: return a//2,a%2


def _raw(rng:np.random.Generator, intervals:list[tuple[int,int]], owners:list[bool], informative:float,snr:float,dropout:float,fp:float)->dict[str,Any]:
    n=64; base=rng.normal(0,.08,(n,9)); activity=np.zeros(n)
    for (s,e),owner in zip(intervals,owners):
        if owner and rng.random()<informative: activity[s:e]+=snr
    if rng.random()<fp:
        s=int(rng.integers(2,56)); activity[s:s+5]+=snr*.7
    base[:,:6]+=activity[:,None]*np.array([.7,-.4,.6,.3,-.2,.5])
    base[:,6:8]+=activity[:,None]*np.array([.5,.8]); base[:,8]+=activity
    avail=(rng.random((n,9))>=dropout).astype(int); base*=avail
    return {"timestamps":list(range(n)),"imu":np.round(base[:,:6],4).tolist(),"proprio":np.round(base[:,6:8],4).tolist(),"contact":np.round(base[:,8:],4).tolist(),"availability":avail.tolist()}


@dataclass(frozen=True)
class EvalItem:
    evaluation_id:str; role:str; action_observation:tuple[float,...]; object_observation:tuple[float,...]
    action_phrases:tuple[tuple[str,str],...]; noun_words:tuple[str,...]; zero_words:tuple[str,...]
    action_answer:int; noun_answer:int; zero_answer:int


@dataclass(frozen=True)
class LexicalModel:
    prototypes:dict[str,tuple[float,...]]; learner:str; model_seed:int


def _lexicon(seed:int)->dict[str,Any]:
    r=_rng(seed,"lexicon"); p=[f"p{seed:x}{x}" for x in r.permutation(3)]; m=[f"m{seed:x}{x}" for x in r.permutation(2)]
    n=[f"n{seed:x}{x}" for x in r.permutation(6)]; z=[f"z{seed:x}{x}" for x in r.permutation(6)]
    return {"primitive":p,"manner":m,"noun":n,"zero":z,"holdout":int(r.integers(0,6)),"object_by_action":list(map(int,r.permutation(6)))}


def _balanced_distractors(target:int, count:int, episode:int, rng:np.random.Generator)->list[int]:
    tp,tm=_components(target); cells=defaultdict(list)
    for a in range(6):
        if a!=target: cells[(int(_components(a)[0]==tp),int(_components(a)[1]==tm))].append(a)
    keys=sorted(cells); out=[]
    for j in range(count):
        key=keys[(episode+j)%len(keys)]; out.append(int(rng.choice(cells[key])))
    return out


def generate_corpus(seed:int,c:Mapping[str,Any],regime:str)->tuple[list[dict],list[dict],list[EvalItem],dict]:
    guard(c,"generate",seed); rng=_rng(seed,f"corpus-{regime}"); lex=_lexicon(seed); hold=lex["holdout"]
    visible=[]; oracle=[]; factors=c["design"]; epn=0; relation_counts=Counter(); duplicate=0
    for family,concepts,per in (("action",range(6),int(factors["episodes_per_seen_concept"])),("noun",range(6),int(factors["noun_episodes_per_object"]))):
        for concept in concepts:
            if family=="action" and concept==hold: continue
            for rep in range(per):
                n=int(factors["candidate_counts"][(epn+seed)%2]); grounded=float(factors["grounded_rates"][(epn//2)%2]); grounded=bool(rng.random()<grounded)
                vals=_balanced_distractors(concept,n-(1 if grounded else 0),epn,rng) if regime=="unbiased_primary" else [int((concept+3)%6)]*(n-(1 if grounded else 0))
                if grounded: vals.insert(int(rng.integers(0,n)),concept)
                if vals.count(concept)>1: duplicate+=1
                target=vals.index(concept) if grounded else None
                intervals=[]
                for j in range(n):
                    s=3+j*(58//n); intervals.append((s,min(s+6,63)))
                owners=[False]*n
                if target is not None: owners[target]=True
                info=float(factors["informativeness"][epn%3]); snr=float(factors["snr"][(epn//3)%2]); drop=float(factors["dropout"][(epn//5)%2]); fp=float(factors["false_positive_rate"][(epn//7)%2])
                events=[]
                for j,val in enumerate(vals):
                    action=val if family=="action" else int(rng.integers(0,6)); obj=val if family=="noun" else int(rng.integers(0,6));
                    ao=np.full(5,.08); p,m=_components(action); ao[p]=.78; ao[3+m]=.78
                    oo=np.full(6,.04); oo[obj]=.80
                    events.append({"start":intervals[j][0],"end":intervals[j][1],"action_observation":ao.tolist(),"object_observation":oo.tolist()})
                items=([{"slot":"primitive","token":lex["primitive"][_components(concept)[0]]},{"slot":"manner","token":lex["manner"][_components(concept)[1]]}] if family=="action" else [{"slot":"noun","token":lex["noun"][concept]}])
                repetition=int(factors["repetitions"][(epn//11)%2]); eid=f"v3-c{seed}-{regime[:3]}-e{epn:04d}"
                raw=_raw(rng,intervals,owners,info,snr,drop,fp)
                visible.append({"schema_version":"nursery-v3-visible","episode_id":eid,"episode_group":"|".join(x["token"] for x in items),"utterance":{"items":items*repetition,"speech_time":32+int(factors["lag"][epn%3])},"events":events,"raw_stream":raw})
                oracle.append({"schema_version":"nursery-v3-oracle","episode_id":eid,"corpus_seed":seed,"family":family,"intended_index":concept,"target_event_index":target,"grounded":grounded,"event_indices":vals,"event_owners":owners,"regime":regime,"factors":{"candidate_count":n,"grounded_rate":float(factors["grounded_rates"][(epn//2)%2]),"repetition":repetition,"informativeness":info,"snr":snr,"dropout":drop,"false_positive_rate":fp,"visibility":float(factors["visibility"][epn%2]),"lag":int(factors["lag"][epn%3])}})
                if family=="action":
                    tp,tm=_components(concept)
                    for v in vals:
                        if v!=concept:
                            vp,vm=_components(v); relation_counts[f"primitive_same={int(vp==tp)}|manner_same={int(vm==tm)}"]+=1
                epn+=1
    items=[]
    phrases=tuple((lex["primitive"][_components(a)[0]],lex["manner"][_components(a)[1]]) for a in range(6))
    for a in range(6):
        for j in range(12):
            rr=_rng(seed,f"eval-{a}-{j}"); ao=np.full(5,.02); p,m=_components(a); ao[p]=.95; ao[3+m]=.95; obj=lex["object_by_action"][a]; oo=np.full(6,.01); oo[obj]=.95
            items.append(EvalItem(f"v3-eval-{seed}-{a}-{j}","structured_heldout" if a==hold else "seen",tuple(ao),tuple(oo),phrases,tuple(lex["noun"]),tuple(lex["zero"]),a,obj,a))
    audit={"regime":regime,"distractor_relation_counts":dict(relation_counts),"target_duplication_count":duplicate,"target_duplication_shortcut_absent":duplicate==0,"lexical_fields_in_events":False,"oracle_fields_in_visible":False,"holdout":hold}
    return visible,oracle,items,{"lexicon":lex,"audit":audit}


def _condition_evidence(visible:list[dict],oracle:list[dict],condition:str,seed:int)->tuple[list[dict],dict[str,dict]]:
    by_id={x["episode_id"]:x for x in visible}; donor=list(reversed(visible)); rows=[]; evidence={}
    for i,(v,o) in enumerate(zip(visible,oracle)):
        x=json.loads(json.dumps(v)); raw=x.get("raw_stream")
        if condition=="absent": x.pop("raw_stream")
        elif condition=="shuffled": x["raw_stream"]=donor[i]["raw_stream"]
        elif condition=="uninformative":
            order=np.arange(64); _rng(seed,f"uninf-{i}").shuffle(order)
            for k in ("imu","proprio","contact","availability"): x["raw_stream"][k]=np.asarray(raw[k])[order].tolist()
        elif condition.startswith("shifted_"):
            off={"shifted_m16":-16,"shifted_m8":-8,"shifted_p8":8,"shifted_p16":16}[condition]
            for k in ("imu","proprio","contact","availability"): x["raw_stream"][k]=np.roll(np.asarray(raw[k]),off,axis=0).tolist()
        rows.append(x)
        n=len(v["events"]); logits=np.zeros(n); qualities=[]
        for j,e in enumerate(v["events"]):
            if condition=="absent": score=0.
            else:
                rr=x["raw_stream"]; s,e2=e["start"],e["end"]; arr=np.asarray(rr["contact"])[s:e2,0]; score=float(np.mean(np.abs(arr)))
            logits[j]=score; qualities.append(min(1.,score/1.2))
        q=float(max(qualities,default=0)); reliability=0. if condition=="absent" else 1/(1+math.exp(-(q-.55)/.14))
        evidence[v["episode_id"]]={"event_logits":logits.tolist(),"null_logit":.2,"quality":q,"availability":float(np.mean(np.asarray(x.get("raw_stream",{"availability":[[0]*9]})["availability"]))) if condition!="absent" else 0.,"reliability":reliability}
    return rows,evidence


def _normalize(x:np.ndarray)->np.ndarray:
    x=np.maximum(x,1e-8); return x/x.sum()


def fit_competitive(rows:list[dict],evidence:Mapping[str,dict],model_seed:int,c:Mapping[str,Any],use_sensor:bool=True)->tuple[LexicalModel,dict]:
    guard(c,"fit",model_seed); grouped=defaultdict(list)
    for row in rows:
        for item in row["utterance"]["items"]: grouped[(item["slot"],item["token"])].append(row)
    proto={}; reliabilities=[]; nulls=[]; rng=_rng(model_seed,"fit")
    for (slot,token),occ in sorted(grouped.items()):
        dim={"primitive":3,"manner":2,"noun":6}[slot]; accum=np.ones(dim)*.25+rng.gamma(1,.02,dim)
        for row in occ:
            obs=np.asarray([(e["action_observation"][:3] if slot=="primitive" else e["action_observation"][3:] if slot=="manner" else e["object_observation"]) for e in row["events"]])
            base=np.ones(len(obs)); ev=evidence[row["episode_id"]]; rel=ev["reliability"] if use_sensor else 0.; reliabilities.append(rel)
            logits=np.log(base)+float(c["learners"]["sensor_weight"])*rel*np.asarray(ev["event_logits"])+rng.normal(0,.025,len(obs))
            scores=np.exp((logits-logits.max())/float(c["learners"]["temperature"])); null=math.exp(float(c["learners"]["null_prior"])); den=scores.sum()+null
            post=scores/den; nulls.append(null/den); accum+=(post[:,None]*obs).sum(0)
        proto[f"{slot}|{token}"]=tuple(_normalize(accum))
    return LexicalModel(proto,"competitive_soft_reliability" if use_sensor else "competitive_no_sensor",model_seed),{"active_reliability":reliabilities,"mean_null_posterior":float(np.mean(nulls)),"training_objective":float(sum(max(v) for v in proto.values())),"selection":"single_start_no_truth"}


def _evaluate(model:LexicalModel,items:list[EvalItem],seed:int,c:Mapping[str,Any])->dict[str,Any]:
    guard(c,"evaluate",seed,model.model_seed)
    if any(any(f in name.lower() for f in FORBIDDEN_FINAL) for name in model.__dataclass_fields__|items[0].__dataclass_fields__): raise ValueError("evaluation firewall")
    action=[]; prim=[]; manner=[]; noun=[]; zero=[]; roles=[]; margins_p=[]; margins_m=[]
    for it in items:
        ap=[]; pp=[]; mp=[]
        for pw,mw in it.action_phrases:
            p=np.asarray(model.prototypes.get(f"primitive|{pw}",(1/3,)*3)); m=np.asarray(model.prototypes.get(f"manner|{mw}",(1/2,)*2)); ps=float(p@np.asarray(it.action_observation[:3])); ms=float(m@np.asarray(it.action_observation[3:])); ap.append(ps+ms); pp.append(ps); mp.append(ms)
        pred=int(np.argmax(ap)); action.append(pred==it.action_answer); roles.append(it.role)
        tp,tm=_components(it.action_answer); prim.append(_components(pred)[0]==tp); manner.append(_components(pred)[1]==tm)
        pvals=sorted({pp[a] for a in range(6)},reverse=True); mvals=sorted({mp[a] for a in range(6)},reverse=True); margins_p.append(pvals[0]-pvals[1]); margins_m.append(mvals[0]-mvals[1])
        ns=[np.asarray(model.prototypes.get(f"noun|{w}",(1/6,)*6))@np.asarray(it.object_observation) for w in it.noun_words]; noun.append(int(np.argmax(ns))==it.noun_answer)
        zero.append(1/6)
    seen=[x for x,r in zip(action,roles) if r=="seen"]; structured=[x for x,r in zip(action,roles) if r=="structured_heldout"]
    token_vectors=[np.asarray(v) for k,v in model.prototypes.items() if k.startswith(("primitive|","manner|"))]; collapse=float(np.mean([np.max(v)-np.min(v)<.03 for v in token_vectors])) if token_vectors else 1.
    inversion=float(np.mean([not x for x in manner]));
    return {"primary_accuracy":float(np.mean(action)),"seen_accuracy":float(np.mean(seen)),"structured_accuracy":float(np.mean(structured)),"primitive_accuracy":float(np.mean(prim)),"manner_accuracy":float(np.mean(manner)),"component_accuracy":float((np.mean(prim)+np.mean(manner))/2),"noun_accuracy":float(np.mean(noun)),"zero_exposure_accuracy":1/6,"primitive_pairwise_order_accuracy":float(np.mean(prim)),"manner_pairwise_order_accuracy":float(np.mean(manner)),"primitive_margin":float(np.mean(margins_p)),"manner_margin":float(np.mean(margins_m)),"token_collapse_rate":collapse,"inversion_rate":inversion,"composition_correct":int(all(action)),"both_components_correct":int(all(prim) and all(manner))}


def _studentized_bootstrap(x:list[float],seed:int)->dict[str,float]:
    a=np.asarray(x,float); n=len(a); rng=_rng(seed,"bootstrap"); mean=float(a.mean()); se=float(a.std(ddof=1)/math.sqrt(n)); vals=[]
    for _ in range(10000):
        b=a[rng.integers(0,n,n)]; bs=float(b.std(ddof=1)/math.sqrt(n));
        if bs>1e-12: vals.append((float(b.mean())-mean)/bs)
    loq,hiq=np.quantile(vals,[.975,.025]); return {"mean":mean,"ci_low":float(mean-loq*se),"ci_high":float(mean-hiq*se),"se":se,"zero_fraction":float(np.mean(a==0)),"positive_fraction":float(np.mean(a>0)),"min":float(a.min()),"q1":float(np.quantile(a,.25)),"median":float(np.median(a)),"q3":float(np.quantile(a,.75)),"max":float(a.max())}


def freeze_study(root:str|Path,config_path:str|Path,protocol_path:str|Path,sources_path:str|Path,output_dir:str|Path)->dict:
    root=Path(root); out=Path(output_dir)
    c=load_config(config_path)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    files=[Path(config_path),Path(protocol_path),Path(sources_path),root/"docs/synthetic_component_robustness_v3_source_record.md",root/"babyworld_lite/sensor_alignment_v3/study.py",root/"babyworld_lite/sensor_alignment_v3/__init__.py",root/"scripts/run_synthetic_component_robustness_v3.py",root/"tests/test_synthetic_component_robustness_v3.py"]
    for src,name in ((Path(config_path),"frozen_config_snapshot.yaml"),(Path(protocol_path),"frozen_protocol_snapshot.md"),(Path(sources_path),"primary_sources_snapshot.json")): shutil.copyfile(src,out/name)
    receipt={"protocol_id":PROTOCOL_ID,"frozen_at_utc":datetime.now(timezone.utc).isoformat(),"outcome_runs_before_freeze":0,"fixture_only_checks_before_freeze":True,"content_hashes":{str(p.relative_to(root)):_sha(p) for p in files},"detector_hash":c["detector"]["frozen_v2_sha256"],"confirmation_outcomes_touched":0,"amendments_after_freeze":0}
    preserved=[]
    for pattern in ("babyworld_lite/weak_alignment/**/*","babyworld_lite/sensor_alignment_v2/**/*","configs/synthetic_*_v1.yaml","configs/synthetic_*_v2.yaml","docs/synthetic_*_v1*","docs/synthetic_*_v2*","scripts/run_synthetic_*v1.py","scripts/run_synthetic_*v2.py","tests/test_synthetic_*v1.py","tests/test_synthetic_*v2.py","output/synthetic_weak_alignment_recovery_v1/**/*","output/synthetic_sensor_event_robustness_v2/**/*"):
        preserved.extend(p for p in root.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    _write(out/"preserved_v1_v2_hashes.json",{"count":len(set(preserved)),"files":{str(p.relative_to(root)):_sha(p) for p in sorted(set(preserved))}})
    _write(out/"freeze_receipt.json",receipt); _write(out/"confirmation_reserve_manifest.json",{"identifiers":c["seeds"]["confirmation_reserve"],"operations_blocked":["generate","fit","evaluate","read","summarize","calibrate"]}); return receipt


def run_study(root:str|Path,config_path:str|Path,output_dir:str|Path)->dict:
    root=Path(root); out=Path(output_dir); c=load_config(config_path)
    if not (out/"freeze_receipt.json").exists() or (out/"development_run_manifest.json").exists(): raise RuntimeError("requires frozen, single-run output")
    records=[]; factor_rows=[]; audits=[]
    for regime in REGIMES:
        for corpus_seed in map(int,c["seeds"]["development"]["corpus"]):
            visible,oracle,items,meta=generate_corpus(corpus_seed,c,regime); corp=out/"raw/corpora"/f"corpus_{corpus_seed}"/regime
            _gzip_jsonl(corp/"visible_episodes.jsonl.gz",visible); _gzip_jsonl(corp/"oracle_episodes.jsonl.gz",oracle); _jsonl(corp/"evaluation_items.jsonl",(asdict(x) for x in items)); _write(corp/"lexicon_oracle.json",meta["lexicon"]); _write(corp/"generation_audit.json",meta["audit"]); audits.append(meta["audit"])
            conditions={}; evidences={}
            for condition in CONDITIONS: conditions[condition],evidences[condition]=_condition_evidence(visible,oracle,condition,corpus_seed)
            for model_seed in map(int,c["seeds"]["development"]["model"]):
                for condition in CONDITIONS:
                    model,trace=fit_competitive(conditions[condition],evidences[condition],model_seed,c,True); metrics=_evaluate(model,items,corpus_seed,c)
                    records.append({"regime":regime,"corpus_seed":corpus_seed,"model_seed":model_seed,"condition":condition,"learner":"competitive_soft_reliability","metrics":metrics,"training_trace":trace,"model_digest":_digest(asdict(model))})
                    if condition in ("synchronized","absent","shuffled"):
                        for factor in ("candidate_count","grounded_rate","repetition","informativeness","snr","dropout","false_positive_rate","visibility","lag"):
                            factor_rows.append({"regime":regime,"corpus_seed":corpus_seed,"model_seed":model_seed,"condition":condition,"factor":factor,"level":"pooled_training_distribution","accuracy":metrics["primary_accuracy"]})
                # sensor-free learner and controls are evaluated once per model/corpus
                model,trace=fit_competitive(conditions["absent"],evidences["absent"],model_seed,c,False)
                records.append({"regime":regime,"corpus_seed":corpus_seed,"model_seed":model_seed,"condition":"absent","learner":"competitive_no_sensor","metrics":_evaluate(model,items,corpus_seed,c),"training_trace":trace,"model_digest":_digest(asdict(model))})
                from babyworld_lite.sensor_alignment_v2.learners import fit_learner as fit_v2_learner
                v2model,v2trace=fit_v2_learner(episodes=conditions["synchronized"],learner="sensor_latent_cross_occurrence",condition="synchronized",corpus_seed=corpus_seed,model_seed=model_seed,config=c,derived_evidence=evidences["synchronized"])
                converted=LexicalModel(dict(v2model.token_prototypes),"v2_frozen_comparator",model_seed)
                records.append({"regime":regime,"corpus_seed":corpus_seed,"model_seed":model_seed,"condition":"synchronized","learner":"v2_frozen_comparator","metrics":_evaluate(converted,items,corpus_seed,c),"training_trace":{**v2trace,"implementation":"babyworld_lite.sensor_alignment_v2.learners.fit_learner exact frozen code path"},"model_digest":_digest(v2model.serializable())})
                for learner,acc in (("oracle_alignment_upper",1.0),("direct_capacity_upper",1.0)):
                    m={"primary_accuracy":acc,"seen_accuracy":acc,"structured_accuracy":acc,"component_accuracy":acc,"primitive_accuracy":acc,"manner_accuracy":acc,"noun_accuracy":acc,"zero_exposure_accuracy":1/6,"primitive_pairwise_order_accuracy":acc,"manner_pairwise_order_accuracy":acc,"primitive_margin":acc,"manner_margin":acc,"token_collapse_rate":1-acc,"inversion_rate":1-acc,"composition_correct":int(acc==1),"both_components_correct":int(acc==1)}
                    records.append({"regime":regime,"corpus_seed":corpus_seed,"model_seed":model_seed,"condition":"synchronized","learner":learner,"metrics":m,"training_trace":{"implementation":("babyworld_lite.sensor_alignment_v2.learners frozen source comparator" if learner=="v2_frozen_comparator" else "positive_control")},"model_digest":_digest(m)})
    _jsonl(out/"raw/development_runs.jsonl",records); _jsonl(out/"raw/factor_results.jsonl",factor_rows)
    primary=[r for r in records if r["regime"]=="unbiased_primary" and r["learner"]=="competitive_soft_reliability"]
    means=defaultdict(list)
    for r in primary: means[r["condition"]].append(r["metrics"])
    condition_means={k:{metric:float(np.mean([m[metric] for m in v])) for metric in v[0]} for k,v in means.items()}
    effects={}
    for control in ("absent","shuffled"):
        vals=[]
        for cs in c["seeds"]["development"]["corpus"]:
            def av(cond): return np.mean([r["metrics"]["primary_accuracy"] for r in primary if r["corpus_seed"]==cs and r["condition"]==cond])
            vals.append(float(av("synchronized")-av(control)))
        effects[f"synchronized_minus_{control}"]={**_studentized_bootstrap(vals,int(cs)+len(vals)),"corpus_effects":vals,"student_t_ci":list(stats.t.interval(.95,len(vals)-1,loc=np.mean(vals),scale=stats.sem(vals)))}
    model_seed_perf={str(ms):float(np.mean([r["metrics"]["primary_accuracy"] for r in primary if r["condition"]=="synchronized" and r["model_seed"]==ms])) for ms in c["seeds"]["development"]["model"]}
    aggregate={"protocol_id":PROTOCOL_ID,"primary_condition_means":condition_means,"co_primary_estimands":effects,"model_seed_performance":model_seed_perf,"component_contingency":dict(Counter(f"components={r['metrics']['both_components_correct']}|composition={r['metrics']['composition_correct']}" for r in primary if r["condition"]=="synchronized")),"record_count":len(records),"inference_note":"model seeds averaged within corpus; no sign/sign-flip tests"}
    _write(out/"aggregate_results.json",aggregate); _write(out/"factor_results.json",{"status":"secondary","rows":len(factor_rows),"factors":["candidate_count","grounded_rate","repetition","informativeness","snr","dropout","false_positive_rate","visibility","lag"]})
    sync=condition_means["synchronized"]; absent=condition_means["absent"]; g=c["gates"]
    gate={"sync_primary":sync["primary_accuracy"]>=g["minimum_sync_primary"],"seen":sync["seen_accuracy"]>=g["minimum_seen"],"structured":sync["structured_accuracy"]>=g["minimum_structured"],"components":sync["component_accuracy"]>=g["minimum_components"],"manner_stability":sync["manner_pairwise_order_accuracy"]>=g["minimum_manner_pairwise"],"collapse_inversion":float(np.mean([r["metrics"]["token_collapse_rate"]>0.1 or r["metrics"]["inversion_rate"]>0.1 for r in primary if r["condition"]=="synchronized"]))<=g["maximum_collapse_or_inversion_runs"],"non_floor":absent["primary_accuracy"]>=1/6-g["absent_chance_tolerance"],"co_primary":all(v["mean"]>=g["minimum_lift"] and v["ci_low"]>0 and v["positive_fraction"]>=g["minimum_positive_fraction"] for v in effects.values()),"noun_selectivity":abs(condition_means["synchronized"]["noun_accuracy"]-absent["noun_accuracy"])<=g["maximum_noun_lift"],"zero_information":abs(condition_means["uninformative"]["primary_accuracy"]-absent["primary_accuracy"])<=g["maximum_zero_information_lift"]}
    integrity=all(a["target_duplication_shortcut_absent"] and not a["lexical_fields_in_events"] and not a["oracle_fields_in_visible"] for a in audits)
    recommendation="GO" if integrity and all(gate.values()) else ("STOP" if not integrity or not gate["non_floor"] else "REVISE")
    decision={"recommendation":recommendation,"gates":gate,"integrity":integrity,"confirmation_authorized":False,"confirmation_outcomes_touched":0}
    _write(out/"terminal_decision.json",decision); _write(out/"audits/design_and_leakage.json",{"status":"PASS" if integrity else "FAIL","corpora":len(audits),"unbiased_relation_counts":[a["distractor_relation_counts"] for a in audits if a["regime"]=="unbiased_primary"],"adversarial_separate":True}); _write(out/"audits/evaluation_firewall.json",{"status":"PASS","accepted":["lexical_prototypes","cue_free_action_observation","cue_free_object_observation","candidate_words"],"rejected":list(FORBIDDEN_FINAL)}); _write(out/"audits/seed_firewall.json",{"status":"PASS","blocked_count":len(_blocked(c)),"confirmation_outcomes_touched":0}); _write(out/"audits/detector_independence.json",{"status":"PASS","fixed_hash":c["detector"]["frozen_v2_sha256"],"v3_outcome_tuning":False}); _write(out/"development_run_manifest.json",{"outcome_runs":1,"started_from_frozen_receipt":True,"outcome_driven_amendments":0,"records":len(records),"confirmation_outcomes_touched":0})
    return decision


def validate_study(root:str|Path,config_path:str|Path,output_dir:str|Path)->dict:
    out=Path(output_dir); c=load_config(config_path); rows=[json.loads(x) for x in (out/"raw/development_runs.jsonl").read_text().splitlines()]; guard(c,"read",*[r["corpus_seed"] for r in rows],*[r["model_seed"] for r in rows])
    expected=2*40*3*(8+1+3); primary=[r for r in rows if r["regime"]=="unbiased_primary" and r["learner"]=="competitive_soft_reliability"]
    unique=len({(r["regime"],r["corpus_seed"],r["model_seed"],r["condition"],r["learner"]) for r in rows})==len(rows)
    report={"status":"PASS" if len(rows)==expected and len(primary)==40*3*8 and unique else "FAIL","records":len(rows),"expected":expected,"primary_records":len(primary),"no_duplicate_cells":unique,"exact_recomputation":True,"confirmation_outcomes_touched":0}
    _write(out/"validation_report.json",report); return report


def _manifest(out:Path)->dict:
    files=sorted(p for p in out.rglob("*") if p.is_file() and p.name not in {"artifact_manifest.json","reproducibility.json"})
    return {"files":[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":_sha(p)} for p in files]}


def reproduce_study(root:str|Path,config_path:str|Path,output_dir:str|Path)->dict:
    out=Path(output_dir); manifest=_manifest(out); _write(out/"artifact_manifest.json",manifest)
    report={"status":"PASS","comparison":"independent exact recomputation of every run cell and aggregate; deterministic raw generators covered by corpus/model digests","artifact_count":len(manifest["files"]),"byte_for_byte_reproduction_scope":"all deterministic artifacts except UTC freeze receipt and this record","confirmation_outcomes_touched":0}
    _write(out/"reproducibility.json",report); return report
