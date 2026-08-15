#!/usr/bin/env python3
import json,py_compile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def req(v,m):
    if not v: raise AssertionError(m)
def main():
    c=json.loads((R/"configs/pilot_p7.json").read_text()); a=json.loads((R/"pilots/juno_sample/p7_aggregate.json").read_text()); p=json.loads((R/"configs/pilot.json").read_text()); modes=c["schedule"]["cycle_modes"]*10
    req(p["pilot_stage"] in {"P7","P8","P9"} and p["status"] in {"p7_complete","p8_complete","p9_complete"} and p["next_stage"] in {"P8","P9","mandatory_full_reproduction_reset_and_readiness_gates"} and not p["next_stage_started"],"canonical lifecycle")
    req(p["p6"]["next_stage_started"] and p["p7"]["canonical_training_config"]=="configs/pilot_p7.json" and p["p7"]["aggregate_record"].endswith("p7_aggregate.json"),"P6/P7 lifecycle metadata")
    req(len(modes)==130 and {k:modes.count(k) for k in c["schedule"]["exact_counts"]}==c["schedule"]["exact_counts"],"schedule")
    req({k:modes[:65].count(k) for k in c["schedule"]["checkpoint_counts"]}==c["schedule"]["checkpoint_counts"] and modes[65]=="contrastive","boundary")
    req(a["status"]=="p7_complete" and a["schedule"]["counts"]==c["schedule"]["exact_counts"] and a["schedule"]["total"]==130,"counts")
    req(all(a["losses"][m]["count"]==n for m,n in c["schedule"]["exact_counts"].items()),"loss coverage")
    jobs=a["execution"]["slurm_job_ids"]; req(a["execution"]["batch"]==2 and a["execution"]["genuine_contrastive_negatives"] and len(set(jobs.values()))==4 and jobs["preserved_artifact_audit"]=="326922","execution")
    req(all(a["verification"][k] for k in ("P5_teacher_copied_to_vision_at_segment_start","all_losses_finite","component_update_and_forbidden_update_assertions","fresh_load_all_applicable_state_validation")),"gates")
    inv=a["inventory"]; req(inv["p7_retained_artifact_inventory_status"]=="complete_after_preserved_artifact_hardening_audit" and inv["entry_count"]==20 and inv["global_full_dataset_inventory_status"].startswith("incomplete_inventory_out_of_scope"),"retained inventory scope")
    req(a["governed_records"]["retained_inventory_sha256"]=="9a8abc3c7942e96184fb7a7207f782639f6d153a108f1d060d8cd458dff69914" and a["governed_records"]["hardening_audit_sha256"]=="6a73aa3d41b28b8986a632588769afdac7daf91dc96c0a426b88354e6757d14f","audit records")
    grads=a["verification"]["preserved_artifact_gradient_audit"]; req(not grads["optimizer_step"] and grads["model_and_auxiliary_state_unchanged"] and all(grads["finite_present_nonzero_gradients"].values()),"backbone gradient evidence")
    req(a["resume"]["all_saved_hashes_counters_history_and_next_mode_verified_before_state_restoration"] and a["resume"]["all_serialized_applicable_state_passed_schema_provenance_counter_and_operational_load_validation"] and a["resume"]["rng_values_set_not_replay_compared"],"honest checkpoint validation claim")
    req("all_checkpoint_state_restored" not in a["resume"],"unsupported exact restoration claim")
    req(a["verification"]["scheduler_completion"]=={"contrastive":100,"mlm":20} and a["resources"]["scheduler_driver_memory_bytes"] is None,"resources")
    req(a["next_stage"]=="P8" and not a["next_stage_started"] and not a["execution"]["p8_started"],"P8")
    text=(R/"pilots/juno_sample/p7_aggregate.json").read_text().lower(); req(not any(x in text for x in ("/work/","/scratch/","utterance","participant","frame_")),"privacy")
    py_compile.compile(str(R/"scripts/pilot_p7.py"),doraise=True)
    source=(R/"scripts/pilot_p7.py").read_text(); validator=source[source.index("def validate_checkpoint"):source.index("def train")]
    for field in ("tokenizer_sha256","manifest_sha256","p5_sha256","p6_sha256","config_sha256","source_commit","environment_sha256","global_update","counts","next_mode","history","model","ssl","contrastive_optimizer","mlm_optimizer","contrastive_scheduler","mlm_scheduler","torch_cpu_rng","torch_cuda_rng","python_rng","numpy_rng","sampler_rng","slurm_job_id"): req(field in validator,"checkpoint validator missing "+field)
    req(source.index("validate_checkpoint(torch,st,c,args,manifest,executed,65)") < source.index("model.load_state_dict(st[\"model\"]"),"resume validates after restoration")
    req("vision_backbone" in source and "bert_backbone" in source and "optimizer_step\":False" in source and "before==after" in source,"backbone gradient audit absent")
    wrapper=(R/"scripts/pilot_p7_juno_job.sh").read_text(); req("EXECUTED_CONFIG" in wrapper and "afda6484fb346780d6f6ad856e0753b7234a7f26e121d001bd17dd7aefe6d818" in wrapper,"executed config provenance")
    req(c["prerequisites"]["p5_checkpoint_sha256"]==json.loads((R/"pilots/juno_sample/p5_aggregate.json").read_text())["governed_records"]["checkpoint_sha256"],"P5")
    req(c["prerequisites"]["p6_checkpoint_sha256"]==json.loads((R/"pilots/juno_sample/p6_aggregate.json").read_text())["governed_records"]["final_checkpoint_sha256"],"P6")
    print("Pilot P7 verification passed")
if __name__=="__main__": main()
