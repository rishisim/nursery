import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIG = json.loads(
    (ROOT / "configs/embodied_simulation_reference_hand_microcell.json").read_text()
)
SOURCE = (
    ROOT
    / "runs/embodied_simulation/reference_hand_microcell/project/Assets/ReferenceHand/ReferenceHandMicrocell.cs"
).read_text()


def test_reference_hand_config_is_pinned_and_bounded():
    assert CONFIG["package"]["commit"] == "833d82e7333a5f37ebc0844d02431acf74f35d24"
    assert CONFIG["unity"]["editor"] == "6000.0.80f1"
    assert CONFIG["clock"] == {
        "physics_hz": 240,
        "render_hz": 30,
        "steps_per_render_frame": 8,
        "mapping": "FixedUpdate command -> assigned SyntheticLeapProvider fixed event -> Physics.Simulate(1/240) -> post-step record",
        "independent_animation_timeline": False,
        "independent_device_or_service_clock": False,
    }
    assert CONFIG["hand"]["grab_helper_components"] == 0
    assert CONFIG["hand"]["grab_helper_object_instances"] == 0
    assert CONFIG["qualification"]["point_impulses_are_pair_aggregated"] is False
    assert CONFIG["attempts"]["maximum_physical_attempts"] == 2


def test_reference_hand_source_contains_no_object_assistance_operations():
    assert "GrabHelperObject" not in SOURCE
    assert "AddForce(" not in SOURCE
    assert "AddTorque(" not in SOURCE
    assert "MovePosition(" not in SOURCE
    assert "MoveRotation(" not in SOURCE
    assert "TeleportRoot" not in SOURCE
    assert "target.isKinematic = false" in SOURCE
    assert "provider.EmitFixedFrame(hand);" in SOURCE
    assert "Physics.Simulate(Dt);" in SOURCE
    assert "ContactPairPoint.impulse" in SOURCE


def test_reference_hand_decision_stops_downstream_work_after_two_failures():
    decision = (ROOT / "docs/embodied_simulation/reference_hand_microcell_decision.md").read_text()
    assert "Decision: **NO-GO" in decision
    assert "No third physical attempt was run" in decision
    assert "two additional frozen-shape runs" in decision
