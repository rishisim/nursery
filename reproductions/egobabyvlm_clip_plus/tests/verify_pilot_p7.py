#!/usr/bin/env python3
import json,py_compile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def req(v,m):
    if not v: raise AssertionError(m)
def main():
    c=json.loads((R/"configs/pilot_p7.json").read_text()); a=json.loads((R/"pilots/juno_sample/p7_aggregate.json").read_text()); modes=c["schedule"]["cycle_modes"]*10
    req(len(modes)==130 and {k:modes.count(k) for k in c["schedule"]["exact_counts"]}==c["schedule"]["exact_counts"],"schedule")
    req({k:modes[:65].count(k) for k in c["schedule"]["checkpoint_counts"]}==c["schedule"]["checkpoint_counts"] and modes[65]=="contrastive","boundary")
    req(a["status"]=="p7_complete" and a["schedule"]["counts"]==c["schedule"]["exact_counts"] and a["schedule"]["total"]==130,"counts")
    req(all(a["losses"][m]["count"]==n for m,n in c["schedule"]["exact_counts"].items()),"loss coverage")
    req(a["execution"]["batch"]==2 and a["execution"]["genuine_contrastive_negatives"] and len(set(a["execution"]["slurm_job_ids"].values()))==3,"execution")
    req(all(a["verification"][k] for k in ("P5_teacher_copied_to_vision_at_segment_start","all_losses_finite","component_update_and_forbidden_update_assertions","fresh_load_smoke")),"gates")
    req(a["verification"]["scheduler_completion"]=={"contrastive":100,"mlm":20} and a["resources"]["scheduler_driver_memory_bytes"] is None,"resources")
    req(a["next_stage"]=="P8" and not a["next_stage_started"] and not a["execution"]["p8_started"],"P8")
    text=(R/"pilots/juno_sample/p7_aggregate.json").read_text().lower(); req(not any(x in text for x in ("/work/","/scratch/","utterance","participant","frame_")),"privacy")
    py_compile.compile(str(R/"scripts/pilot_p7.py"),doraise=True)
    req(c["prerequisites"]["p5_checkpoint_sha256"]==json.loads((R/"pilots/juno_sample/p5_aggregate.json").read_text())["governed_records"]["checkpoint_sha256"],"P5")
    req(c["prerequisites"]["p6_checkpoint_sha256"]==json.loads((R/"pilots/juno_sample/p6_aggregate.json").read_text())["governed_records"]["final_checkpoint_sha256"],"P6")
    print("Pilot P7 verification passed")
if __name__=="__main__": main()
