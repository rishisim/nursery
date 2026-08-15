#!/usr/bin/env python3
import ast, importlib.util, json, py_compile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def req(v,m):
    if not v: raise AssertionError(m)
def main():
    c=json.loads((R/"configs/pilot_p8.json").read_text()); s=json.loads((R/"pilots/juno_sample/p8_checkpoint_selection.json").read_text()); p=json.loads((R/"configs/pilot.json").read_text()); d=json.loads((R/"pilots/juno_sample/decision_record.json").read_text())
    req(c["stage"]=="P8" and c["status"]=="frozen_before_benchmark_access","freeze")
    req(len(s["eligible_checkpoints"])==1 and s["selected"]["step"]==130 and not s["validation_loss_selection_claimed"],"selection")
    req(s["selected"]["sha256"]=="287dc2f08565c5a012d9768e0c06de08804fa73fc82ec498f3ec42726c9cb431","checkpoint")
    req(c["execution"]["authorized_score_runs"]==1 and not c["execution"]["retry_after_started"],"one run")
    req(len(c["inventory"]["tasks"])==10 and c["inventory"]["manifest_count"]==20,"inventory")
    req(c["evaluator"]["clip_l_must_not_initialize_tune_alter_ensemble_or_score_p7"] and c["evaluator"]["full_reproduction_recalibration_required"],"CLIP-L boundary")
    src=(R/"scripts/pilot_p8.py").read_text(); req(src.index('"state":"scoring_started"')<src.index("safe_extract(archive"),"ledger before benchmark read")
    req("load_state_dict(payload[\"model\"],strict=True)" in src and "synthetic test" in src,"adapter qualification")
    req("duplicate_trial_identity" in src and 'enumerate(out["raw_records"])' not in src,"stable trial identity")
    spec=importlib.util.spec_from_file_location("p4",R/"scripts/pilot_p4.py"); p4=importlib.util.module_from_spec(spec); spec.loader.exec_module(p4)
    req(p4.merge_bin("[1,2)")=="[1,4)" and p4.merge_bin("[2,4)")=="[1,4)" and p4.merge_bin("[4,8)")=="[4,8)","P4 bin semantics")
    req(p4.prediction_lexical(1,1)==1 and p4.prediction_grammatical([[1,0],[0,1]])==0 and p4.prediction_grammatical([[1,1],[1,1]])==1,"P4 scoring semantics")
    py_compile.compile(str(R/"scripts/pilot_p8.py"),doraise=True); py_compile.compile(str(R/"scripts/pilot_p8_integrity.py"),doraise=True); ast.parse((R/"scripts/pilot_p8_integrity.py").read_text())
    audit_src=(R/"scripts/pilot_p8_integrity.py").read_text(); req("torch" not in audit_src and "model_inference_executed\":False" in audit_src and "scoring_attempt_started\":False" in audit_src,"preserved-only audit")
    aggregate=R/"pilots/juno_sample/p8_aggregate.json"
    if aggregate.exists():
        a=json.loads(aggregate.read_text()); req(p["pilot_stage"]=="P8" and p["status"]=="p8_complete" and p["next_stage"]=="P9" and not p["next_stage_started"],"lifecycle")
        req(a["execution"]["attempt_count"]==a["execution"]["scoring_started_count"]==a["execution"]["scoring_complete_count"]==1,"attempt ledger")
        req(a["coverage"]["trial_count"]==3721 and a["coverage"]["missing_trials"]==0 and a["coverage"]["duplicate_trial_identities"]==0 and a["coverage"]["independent_manifest_prediction_multiset_match"],"complete independent coverage")
        req(a["aggregation_equivalence"]["all_ten_tasks_and_three_aggregates_identical"] and a["aggregation_equivalence"]["reporting_precision_decimals"]==6,"aggregation equivalence")
        req(a["pre_access_provenance"]["nursery_commit"]=="c935e9b78f8bc8c59e2a589da2e1824dc8baf577","pre-access commit")
        req(a["governed_records"]["integrity_audit_sha256"]==c["post_run_integrity"]["audit_record_sha256"] and a["governed_records"]["complete_inventory_entry_count"]==11527,"governed hardening")
        req(d["status"]=="p8_complete" and d["p7_status"]["next_stage_started"] and d["p8_status"]["status"]=="p8_complete" and d["p8_status"]["attempt_ledger"]["scoring_complete"]==1,"decision lifecycle")
        req(set(a["results_percent"])==set(c["inventory"]["tasks"]+['lexical','grammatical','overall']),"all aggregates")
        req(a["next_stage"]=="P9" and not a["next_stage_started"],"P9 not started")
    print("Pilot P8 frozen pre-access verification passed")
if __name__=="__main__": main()
