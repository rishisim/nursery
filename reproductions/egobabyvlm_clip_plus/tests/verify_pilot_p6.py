#!/usr/bin/env python3
"""Dependency-free verification of the frozen and stopped Pilot P6 state."""
import ast, json, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/pilot_p6.json"
AGG=ROOT/"pilots/juno_sample/p6_aggregate.json"
SCRIPT=ROOT/"scripts/pilot_p6.py"

def require(value, message):
    if not value: raise AssertionError(message)

def main():
    c=json.loads(CONFIG.read_text()); a=json.loads(AGG.read_text()); source=SCRIPT.read_text(); ast.parse(source)
    require(c["model"]|{} and (c["model"]["hidden_size"],c["model"]["num_hidden_layers"],c["model"]["num_attention_heads"],c["model"]["intermediate_size"])==(768,12,12,3072),"not exact BERT-base")
    require(c["model"]["initialization"]=="random_only" and not c["model"]["pretrained_or_checkpoint_initialization"],"random initialization not frozen")
    t=c["tokenizer"]; require(t["algorithm"]=="WordPiece" and t["vocab_size"]==2048 and t["corpus"]=="training_split_only","tokenizer contract changed")
    require(len(t["special_tokens"])==len(set(t["special_tokens"]))==5 and sorted(t["special_token_ids"].values())==list(range(5)),"special-token contract invalid")
    split=c["input"]["recording_level_split"]; require(split["disjoint_recordings"] and not split["fallback_used"] and split["train_slot"]!=split["validation_slot"],"recording split invalid")
    require(split["train"]["utterances"]==98 and split["validation"]["utterances"]==1,"split counts changed")
    lower=json.dumps(c).lower(); require(all(x in lower for x in ("p4 calibration","p5 checkpoint","external pretrained")),"forbidden guards missing")
    require(c["training"]["health_checkpoint_step"]==50 and c["training"]["final_optimizer_step"]==100,"step contract changed")
    require(c["execution"]["authorized_completed_runs"]==c["execution"]["authorized_reproducible_minimum_oom_records"]==1,"run/OOM policy changed")
    required=set(c["checkpoint"]["required_state"]); require({"model","optimizer","scheduler","optimizer_step","tokenizer_sha256","config_sha256","python_rng","sampler_state"}<=required,"checkpoint state incomplete")
    require("tokenizer.train_from_iterator" in source and "texts," in source,"train-only tokenizer adapter absent")
    require("find_manifest" in source and "manifest_sha256" in source and "completion.exists" in source,"hash/idempotence gates absent")
    require(a["status"]=="p6_stopped_natural_vocabulary_below_target" and a["tokenizer"]["observed_unique_vocab_size"]==381,"stop evidence invalid")
    require(a["tokenizer"]["target_vocab_size"]==2048 and not a["tokenizer"]["external_tokens"] and not a["tokenizer"]["fabricated_tokens"],"vocabulary failure misstated")
    e=a["execution"]; require(e["juno_guard_passed"] and e["tokenizer_gate_executed_once"] and not e["bert_constructed"] and e["optimizer_steps"]==0,"execution state false")
    require(not e["validation_executed"] and not e["p4_loaded"] and not e["p5_or_dino_loaded"] and not e["p7_started"],"scope leakage claimed")
    require(a["p0_p1_p2_p3_p4_p5_status_preserved"] and a["inventory_status"]=="incomplete_inventory","prior lifecycle changed")
    text=AGG.read_text().lower(); require(not any(x in text for x in ("/work/","/scratch/","participant","session","utterance_text","[pad]","[unk]","[cls]","[sep]","[mask]")),"aggregate leaks governed detail or token strings")
    for path in ROOT.glob("pilots/juno_sample/p[1-5]_aggregate.json"): json.loads(path.read_text())
    with tempfile.TemporaryDirectory() as directory:
        require(Path(directory).is_dir(),"framework temporary directory unavailable")
    print("Pilot P6 verification passed (stopped at exact natural-vocabulary gate)")

if __name__=="__main__": main()
