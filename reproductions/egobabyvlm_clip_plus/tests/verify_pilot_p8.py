#!/usr/bin/env python3
import importlib.util, json, py_compile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def req(v,m):
    if not v: raise AssertionError(m)
def main():
    c=json.loads((R/"configs/pilot_p8.json").read_text()); s=json.loads((R/"pilots/juno_sample/p8_checkpoint_selection.json").read_text())
    req(c["stage"]=="P8" and c["status"]=="frozen_before_benchmark_access","freeze")
    req(len(s["eligible_checkpoints"])==1 and s["selected"]["step"]==130 and not s["validation_loss_selection_claimed"],"selection")
    req(s["selected"]["sha256"]=="287dc2f08565c5a012d9768e0c06de08804fa73fc82ec498f3ec42726c9cb431","checkpoint")
    req(c["execution"]["authorized_score_runs"]==1 and not c["execution"]["retry_after_started"],"one run")
    req(len(c["inventory"]["tasks"])==10 and c["inventory"]["manifest_count"]==20,"inventory")
    req(c["evaluator"]["clip_l_must_not_initialize_tune_alter_ensemble_or_score_p7"] and c["evaluator"]["full_reproduction_recalibration_required"],"CLIP-L boundary")
    src=(R/"scripts/pilot_p8.py").read_text(); req(src.index('"state":"scoring_started"')<src.index("safe_extract(archive"),"ledger before benchmark read")
    req("load_state_dict(payload[\"model\"],strict=True)" in src and "synthetic test" in src,"adapter qualification")
    spec=importlib.util.spec_from_file_location("p4",R/"scripts/pilot_p4.py"); p4=importlib.util.module_from_spec(spec); spec.loader.exec_module(p4)
    req(p4.merge_bin("[1,2)")=="[1,4)" and p4.merge_bin("[2,4)")=="[1,4)" and p4.merge_bin("[4,8)")=="[4,8)","P4 bin semantics")
    req(p4.prediction_lexical(1,1)==1 and p4.prediction_grammatical([[1,0],[0,1]])==0 and p4.prediction_grammatical([[1,1],[1,1]])==1,"P4 scoring semantics")
    py_compile.compile(str(R/"scripts/pilot_p8.py"),doraise=True)
    print("Pilot P8 frozen pre-access verification passed")
if __name__=="__main__": main()
