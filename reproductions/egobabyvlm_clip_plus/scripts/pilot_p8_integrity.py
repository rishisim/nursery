#!/usr/bin/env python3
"""Audit completed P8 solely from preserved manifests, predictions, and provenance."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, math, os, stat, subprocess, sys, tempfile
from pathlib import Path
from typing import Any

class AuditError(RuntimeError): pass
def require(v:Any,m:str)->None:
    if not v: raise AuditError(m)
def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def canonical(v:Any)->bytes: return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def identity(v:dict)->str: return hashlib.sha256(canonical(v)).hexdigest()
def atomic_json(p:Path,v:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(p.parent,0o700)
    fd,n=tempfile.mkstemp(prefix="."+p.name+".",dir=p.parent)
    try:
        with os.fdopen(fd,"w") as f: json.dump(v,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.chmod(n,0o600); os.replace(n,p)
    finally:
        if os.path.exists(n): os.unlink(n)
def owner(p:Path)->None: require(p.is_file() and stat.S_IMODE(p.stat().st_mode)==0o600,"owner_only:"+p.name)
def manifest_path(root:Path,task:str,style:str)->Path:
    if task.startswith("lex_"):
        pos=task[4:]; return root/"Lexical"/pos.capitalize()/f"manifest_{pos}_{style}.json"
    cat=task[5:]; return root/"Grammatical"/task/f"manifest_grammatical_{cat}_{style}.json"
def load_module(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path); require(spec is not None and spec.loader is not None,"module_spec:"+name); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def main():
    p=argparse.ArgumentParser()
    for n in ("config","selection","result","aggregate","ledger","qualification","archive","data_root","p4_inventory","code_root","upstream_source","record_root","log_root","pre_access_hashes"): p.add_argument("--"+n.replace("_","-"),dest=n,type=Path,required=True)
    p.add_argument("--pre-access-commit",required=True); a=p.parse_args(); c=json.loads(a.config.read_text()); tasks=c["inventory"]["tasks"]
    for x in (a.selection,a.result,a.ledger,a.qualification,a.archive,a.p4_inventory): owner(x)
    selection=json.loads(a.selection.read_text()); require(selection["selected"]["sha256"]==c["selected_checkpoint"]["sha256"] and selection["selected"]["step"]==130,"checkpoint_selection")
    ledger=json.loads(a.ledger.read_text()); require(ledger["state"]=="scoring_complete" and ledger["scoring_started_count"]==ledger["scoring_complete_count"]==1,"immutable_attempt")
    require(sha256(a.archive)==c["archive"]["sha256"] and a.archive.stat().st_size==c["archive"]["size_bytes"],"archive")
    p4inv=json.loads(a.p4_inventory.read_text()); result=json.loads(a.result.read_text()); require(sha256(a.p4_inventory)==c["archive"]["p4_inventory_sha256"],"p4_inventory_hash"); require(p4inv==result["archive_inventory"],"p4_p8_inventory_difference")
    expected_counts={}; actual_counts={}; expected_all=[]; actual_all=[]
    raw=result["raw_predictions"]
    for style in c["inventory"]["styles"]:
        by_task=collections.defaultdict(list)
        for r in raw[style]: by_task[r["task_name"]].append(r)
        require(set(by_task)==set(tasks),"actual_task_inventory:"+style)
        for task in tasks:
            mp=manifest_path(a.data_root,task,style); owner(mp); items=json.loads(mp.read_text()).get("items",[]); expected=[identity(x) for x in items]; actual=[identity(x["metadata"]) for x in by_task[task]]
            require(len(expected)==len(set(expected)),"duplicate_manifest_trial:"+style+":"+task)
            require(collections.Counter(expected)==collections.Counter(actual),"missing_or_duplicate_prediction:"+style+":"+task)
            key=style+":"+task; expected_counts[key]=len(expected); actual_counts[key]=len(actual); expected_all.extend(expected); actual_all.extend(actual)
    require(expected_counts==actual_counts and collections.Counter(expected_all)==collections.Counter(actual_all),"global_coverage")
    sys.path.insert(0,str(a.code_root/"scripts")); diagnostic=load_module("frozen_p4_clip_l_diagnostic",a.code_root/"scripts/pilot_p4_clip_l_diagnostic.py"); d=diagnostic.task_aggregate(raw,tasks); gram=[x for x in tasks if x.startswith("gram_")]; d["lexical"]=(d["lex_nouns"]+d["lex_adjectives"])/2; d["grammatical"]=sum(d[x] for x in gram)/len(gram); d["overall"]=(d["lexical"]+d["grammatical"])/2
    metrics_path=a.upstream_source/"evaluation/multimodal/machine_devbench/metrics.py"; require(sha256(metrics_path)==c["source"]["evaluator_hashes"]["evaluation/multimodal/machine_devbench/metrics.py"],"upstream_metrics_hash"); metrics=load_module("frozen_upstream_metrics",metrics_path); aggregator=metrics.ResultAggregator()
    for rows in raw.values():
        for r in rows: aggregator.add(r["task_name"],r["prediction"],r["target"],r["metadata"])
    computed=aggregator.compute(); u={t:100*computed["by_task"][t]["accuracy"] for t in tasks}; u.update({"lexical":100*computed["by_task_type"]["lexical"]["accuracy"],"grammatical":100*computed["by_task_type"]["grammatical"]["accuracy"],"overall":100*computed["overall"]["accuracy"]}); rounded_d={k:round(v,6) for k,v in d.items()}; rounded_u={k:round(v,6) for k,v in u.items()}; require(rounded_d==rounded_u,"aggregation_difference")
    owner(a.aggregate); recorded={k:round(v,6) for k,v in json.loads(a.aggregate.read_text())["results_percent"].items()}; require(rounded_d==recorded,"recorded_score_difference")
    relevant=["configs/pilot_p8.json","pilots/juno_sample/p8_checkpoint_selection.json","scripts/pilot_p8.py","scripts/pilot_p8_juno_job.sh","scripts/pilot_p7.py","scripts/pilot_p4_clip_l_diagnostic.py"]; owner(a.pre_access_hashes); frozen=json.loads(a.pre_access_hashes.read_text()); require(frozen["commit"]==a.pre_access_commit,"pre_access_hash_commit")
    provenance={}
    for rel in relevant:
        deployed=a.code_root/rel; require(deployed.is_file(),"deployed_file:"+rel)
        expected=frozen["files"][rel]; require(sha256(deployed)==expected,"deployed_pre_access_mismatch:"+rel); provenance[rel]=expected
    logs={}
    for job,state in (("326932","FAILED"),("326934","COMPLETED"),("326935","COMPLETED")):
        matches=list(a.log_root.glob(("qualify" if job!="326935" else "evaluate")+"-"+job+".log")); require(len(matches)==1,"log_missing:"+job); owner(matches[0]); logs[job]=sha256(matches[0])
    accounting=subprocess.run(["sacct","-j","326930,326932,326934,326935","--noheader","-o","JobID,State,Elapsed,ExitCode,TRESUsageInMax","-P"],check=True,capture_output=True,text=True).stdout
    require("326930|CANCELLED" in accounting and "326932|FAILED" in accounting and "326934|COMPLETED" in accounting and "326935|COMPLETED" in accounting,"slurm_states")
    accounting_path=a.record_root/"slurm_accounting.txt"; accounting_path.write_text(accounting); os.chmod(accounting_path,0o600)
    audit={"schema_version":1,"record_type":"pilot_p8_preserved_artifact_integrity_audit","run_id":c["engineering_run_id"],"model_inference_executed":False,"scoring_attempt_started":False,"scores_changed":False,"one_immutable_completed_attempt":True,"coverage":{"identity":"sha256_of_canonical_full_manifest_trial_specification","expected_counts":expected_counts,"actual_counts":actual_counts,"trial_count":len(expected_all),"missing":0,"duplicates":0,"count_contract_sha256":hashlib.sha256(canonical(expected_counts)).hexdigest()},"aggregation":{"p4_clip_l_task_aggregate_equivalent_to_upstream_result_aggregator":True,"reporting_precision_decimals":6,"scores_sha256":hashlib.sha256(canonical(rounded_d)).hexdigest()},"archive":{"sha256":sha256(a.archive),"p4_inventory_sha256":sha256(a.p4_inventory),"entry_count":len(p4inv),"p4_inventory_identical_to_p8_extraction_inventory":True},"provenance":{"pre_access_nursery_commit":a.pre_access_commit,"pre_access_hash_record_sha256":sha256(a.pre_access_hashes),"successful_qualification_job":"326934","single_scoring_job":"326935","deployed_file_sha256":provenance,"slurm_log_sha256":logs,"slurm_accounting_sha256":sha256(accounting_path)},"classification":c["classification"],"p9_started":False,"owner_only":True}
    atomic_json(a.record_root/"integrity_audit.json",audit); print(json.dumps({"audit_sha256":sha256(a.record_root/"integrity_audit.json"),"coverage":audit["coverage"],"aggregation":audit["aggregation"],"archive":audit["archive"],"provenance":audit["provenance"]},sort_keys=True))
if __name__=="__main__": main()
