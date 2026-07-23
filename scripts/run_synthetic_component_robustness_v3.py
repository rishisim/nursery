#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from babyworld_lite.sensor_alignment_v3.study import freeze_study, run_study, validate_study, reproduce_study
CONFIG=ROOT/"configs/synthetic_component_robustness_v3.yaml"
PROTOCOL=ROOT/"docs/synthetic_component_robustness_v3_protocol.md"
SOURCES=ROOT/"docs/synthetic_component_robustness_v3_primary_sources.json"
OUTPUT=ROOT/"output/synthetic_component_robustness_v3"

def main():
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=["freeze","run","validate","reproduce","all"]); a=p.parse_args()
    result={}
    if a.phase in ("freeze","all"): result["freeze"]=freeze_study(ROOT,CONFIG,PROTOCOL,SOURCES,OUTPUT)
    if a.phase in ("run","all"): result["run"]=run_study(ROOT,CONFIG,OUTPUT)
    if a.phase in ("validate","all"): result["validate"]=validate_study(ROOT,CONFIG,OUTPUT)
    if a.phase in ("reproduce","all"): result["reproduce"]=reproduce_study(ROOT,CONFIG,OUTPUT)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
