import json
from pathlib import Path


def test_episode_and_clock_are_single_continuous_authority():
    episode = json.loads(Path("configs/embodied_simulation_episode.json").read_text())
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    assert episode["activity"]["duration_s"] == 56.0
    assert episode["activity"]["continuous_trace_required"]
    assert not episode["activity"]["hidden_world_resets_permitted"]
    assert gate["authority"]["physics_hz"] == 240
    assert gate["authority"]["render_hz"] == 30
    assert gate["authority"]["steps_per_frame"] == 8
    assert gate["assistance_ledger"] == []


def test_hybrid_source_contains_no_object_assistance_api():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/unity_native_gate/"
        "HybridBimanualCellBuilder.cs"
    ).read_text()
    assert "AddForce" not in source
    assert "AddTorque" not in source
    assert "FixedJoint" not in source
    assert "SetPositionAndRotation" not in source
    assert "target.position =" not in source
    assert "blueCup.position =" not in source
    assert "assistance_ledger_entries=0" in source


def test_unavailable_truth_cannot_satisfy_an_integrated_pass():
    gate = json.loads(Path("configs/embodied_simulation_bimanual_gate.json").read_text())
    provenance = gate["field_provenance"]
    assert provenance["metric_depth"].startswith("UNAVAILABLE")
    assert provenance["semantic_and_instance"].startswith("UNAVAILABLE")
    assert provenance["head_imu"].startswith("UNAVAILABLE")
    decision = Path("docs/embodied_simulation_bimanual_gate_decision.md").read_text()
    assert "Decision: **NO-GO" in decision
