#!/usr/bin/env python3
"""Dependency-free verification for the final pilot promotion decision."""
import importlib.util, json, py_compile
from pathlib import Path

R=Path(__file__).resolve().parents[1]
def req(v,m):
    if not v: raise AssertionError(m)
def main():
    spec=json.loads((R/"spec.json").read_text()); p=json.loads((R/"configs/pilot.json").read_text())
    c=json.loads((R/"configs/pilot_p9.json").read_text()); d=json.loads((R/"pilots/juno_sample/p9_decision.json").read_text())
    req(spec["phase"]["status"]=="blocked_by_exact_usable_subset_evidence","scientific status changed")
    req(all(x["status"]=="unresolved" for x in spec["unresolved_reproduction_variables"]),"scientific variable resolved")
    req(p["pilot_stage"]=="P9" and p["status"]=="p9_complete" and not p["next_stage_started"],"P9 lifecycle")
    req(p["next_stage"]==d["next_action"]=="mandatory_full_reproduction_reset_and_readiness_gates","invented P10")
    req(d["three_axis_decision"]=={"pilot_engineering_completion":"complete","infrastructure_pipeline_scale_readiness":"conditional_go","scientific_full_reproduction_readiness":"no_go_blocked"},"three-axis decision")
    req(set(d["gate_reconciliation"])=={f"P{i}" for i in range(9)},"P0-P8 reconciliation incomplete")
    req(d["governed_storage_audit"]["required_family_missing_count"]==0 and d["governed_storage_audit"]["scratch_sole_irreplaceable_copy_found"] is False,"retained artifact gate")
    req(d["governed_storage_audit"]["backup_or_snapshot_evidence"]=="not_demonstrated","backup overclaim")
    req([x["duration_hours"] for x in d["preprocessing_projection"]["scenarios"]]==c["duration_scenarios_hours"],"duration scenarios")
    req("380-token tokenizer" in d["artifact_classification"]["C"] and "all P7 CLIP+ weights" in d["artifact_classification"]["C"],"pilot artifacts promoted scientifically")
    req("cannot guide" in d["P8_non_claim"] and "resource estimates" in d["P8_non_claim"],"P8 non-claim incomplete")
    req(len(d["mandatory_reset"])==6 and any("30522" in x for x in d["mandatory_reset"]),"mandatory reset incomplete")
    req(c["mode"]=="evidence_reconciliation_only" and all(x in c["forbidden_actions"] for x in ("training","new_media_preprocessing","Machine-DevBench_rescoring","scratch_or_durable_cleanup")),"P9 scope")
    py_compile.compile(str(R/"scripts/pilot_p9.py"),doraise=True)
    module_spec=importlib.util.spec_from_file_location("p9",R/"scripts/pilot_p9.py"); m=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(m)
    calc=m.projections(c,m.load_aggregates()); req(calc["duration_scenarios"][1]["source_video_hours"]==868.4602,"projection calculation")
    req(calc["training_video_hour_ratio_extrapolation_allowed"] is False,"training ratio extrapolation enabled")
    print("Pilot P9 verification passed")
if __name__=="__main__": main()
