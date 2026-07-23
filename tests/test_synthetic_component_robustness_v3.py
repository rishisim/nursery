from dataclasses import fields
from pathlib import Path
import hashlib, json, copy
import pytest
from babyworld_lite.sensor_alignment_v3 import study as v3

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/synthetic_component_robustness_v3.yaml"

def fixture_config():
    c=v3.load_config(CFG); c=copy.deepcopy(c); c["seeds"]["development"]={"corpus":[88001],"model":[881,883,887]}; return c

def test_seed_firewall_all_operations():
    c=v3.load_config(CFG)
    for op in ("generate","fit","evaluate","read","summarize","calibrate"):
        with pytest.raises(PermissionError): v3.guard(c,op,97103)
        with pytest.raises(PermissionError): v3.guard(c,op,12011)

def test_registries_are_fresh_and_model_replicates_stochastic():
    c=v3.load_config(CFG); assert len(c["seeds"]["development"]["corpus"])==40; assert len(c["seeds"]["development"]["model"])>=3
    assert len(set(c["seeds"]["development"]["model"]))==len(c["seeds"]["development"]["model"])

def test_fixed_detector_hash_and_independence():
    c=v3.load_config(CFG); p=ROOT/c["detector"]["frozen_v2_path"]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==c["detector"]["frozen_v2_sha256"]

def test_unbiased_distractors_no_duplication_or_leakage():
    c=fixture_config(); visible,oracle,items,meta=v3.generate_corpus(88001,c,"unbiased_primary")
    assert meta["audit"]["target_duplication_shortcut_absent"]
    assert not meta["audit"]["lexical_fields_in_events"] and not meta["audit"]["oracle_fields_in_visible"]
    counts=meta["audit"]["distractor_relation_counts"]; assert len(counts)>=3
    assert all("intended_index" not in e for r in visible for e in r["events"])

def test_adversarial_regime_is_separate():
    c=fixture_config(); _,oracle,_,meta=v3.generate_corpus(88001,c,"adversarial_plus3_secondary")
    assert meta["audit"]["regime"]=="adversarial_plus3_secondary"
    assert all(v==(o["intended_index"]+3)%6 for o in oracle if o["family"]=="action" for v in o["event_indices"] if v!=o["intended_index"])

def test_raw_visible_oracle_schema_separation():
    c=fixture_config(); visible,oracle,_,_=v3.generate_corpus(88001,c,"unbiased_primary")
    assert "raw_stream" in visible[0] and "target_event_index" not in visible[0]
    assert "target_event_index" in oracle[0] and "raw_stream" not in oracle[0]
    raw=visible[0]["raw_stream"]; assert len(raw["imu"][0])==6 and len(raw["proprio"][0])==2 and len(raw["contact"][0])==1

def test_paired_condition_identity_and_absence():
    c=fixture_config(); v,o,_,_=v3.generate_corpus(88001,c,"unbiased_primary")
    sync,_=v3._condition_evidence(v,o,"synchronized",88001); absent,_=v3._condition_evidence(v,o,"absent",88001)
    assert "raw_stream" not in absent[0]
    assert {k:x for k,x in sync[0].items() if k!="raw_stream"}==absent[0]

def test_competitive_behavior_reliability_and_null():
    c=fixture_config(); v,o,_,_=v3.generate_corpus(88001,c,"unbiased_primary"); rows,ev=v3._condition_evidence(v,o,"synchronized",88001)
    m,t=v3.fit_competitive(rows,ev,881,c,True); assert m.prototypes and t["mean_null_posterior"]>0
    assert min(t["active_reliability"])>=0 and max(t["active_reliability"])<=1

def test_evaluation_firewall_fields():
    names=set(v3.LexicalModel.__dataclass_fields__)|set(v3.EvalItem.__dataclass_fields__)
    assert not any(any(x in n.lower() for x in v3.FORBIDDEN_FINAL) for n in names)

def test_preservation_v1_v2_namespaces_are_not_imported_for_writes():
    src=(ROOT/"babyworld_lite/sensor_alignment_v3/study.py").read_text()
    assert "output/synthetic_sensor_event_robustness_v2/detector/frozen_detector.json" not in src
    assert "rmtree" not in src and "unlink(" not in src
