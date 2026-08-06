import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CONFIG = json.loads(
    (ROOT / "configs/embodied_simulation_reference_hand_microcell.json").read_text()
)
SOURCE = (
    ROOT
    / "babyworld_lite/childlens_engine_bakeoff/reference_hand_microcell/ReferenceHandQualification.cs"
).read_text()
RUNNER = (
    ROOT / "babyworld_lite/childlens_engine_bakeoff/reference_hand_microcell/runner.py"
).read_text()


def test_reference_hand_config_is_pinned_and_bounded():
    assert CONFIG["package"]["commit"] == "833d82e7333a5f37ebc0844d02431acf74f35d24"
    assert CONFIG["unity"]["editor"] == "6000.0.80f1"
    assert CONFIG["status"] == "HAND_QUALIFICATION_NO_GO"
    assert CONFIG["eligible_physical_attempts_consumed"] == 0
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
    assert CONFIG["qualification"]["controllable_dof_count"] == 20
    assert CONFIG["qualification"]["controllable_pass_count"] == 20
    assert CONFIG["qualification"]["dof_sweep_pass_count"] == 20
    assert CONFIG["qualification"]["max_visible_registration_error_m"] == 0.04622013
    assert CONFIG["qualification"]["active_right_hand_renderer_count"] == 0
    assert CONFIG["qualification"]["right_hand_binder_count"] == 1
    assert CONFIG["physical_execution"]["attempts_consumed"] == 0


def test_reference_hand_source_contains_no_object_assistance_operations():
    assert "GrabHelperObject" not in SOURCE
    assert "MakeFinger" not in SOURCE
    assert "PoseCommand" not in SOURCE
    assert "AddForce(" not in SOURCE
    assert "AddTorque(" not in SOURCE
    assert "MovePosition(" not in SOURCE
    assert "MoveRotation(" not in SOURCE
    assert "TeleportRoot" not in SOURCE
    assert "dof_start_indices[ArticulationBody.index]" in SOURCE
    assert "axisIndex < 0 || axisIndex > 1" in SOURCE
    assert "SweepRampSteps" in SOURCE
    assert "steady_state" in SOURCE
    assert "post_step_articulation_capsule_fk" in SOURCE
    assert "CurrentFixedFrame" not in SOURCE
    assert "slerpDrive" not in SOURCE
    assert "provider.EmitFixedFrame(command);" in SOURCE
    assert "Physics.Simulate(Dt);" in SOURCE


def test_reference_hand_decision_stops_downstream_work_after_two_failures():
    decision = (ROOT / "docs/embodied_simulation/reference_hand_microcell_decision.md").read_text()
    assert "HAND-QUALIFICATION NO-GO" in decision
    assert "eligible physical attempts consumed: 0" in decision
    assert "no ContactPose seed was consumed" in decision
    assert "zero active right-hand renderers" in decision
    assert "ArticulationBody has no `slerpDrive`" in decision


def test_support_and_release_definitions_require_measured_events():
    definitions = CONFIG["metric_definitions"]
    assert "measured support ContactEvent" in definitions["target_is_supported"]
    assert "support ContactEvent is absent" in definitions["unsupported_lift"]
    assert "previously qualified support window" in definitions["free_release_and_settling"]
    assert definitions["phase_presence_or_package_grab_flag_is_not_evidence"] is True


def test_fresh_checkout_runner_stages_tracked_sources_into_ignored_project():
    assert "shutil.copy2(SOURCE_ROOT / name" in RUNNER
    assert "ReferenceHandQualification.cs" in RUNNER
    assert "ReferenceHandMicrocellBuilder.cs" in RUNNER
    assert "shutil.rmtree(OUTPUT_ROOT)" in RUNNER
    assert "runs/embodied_simulation/reference_hand_microcell/project" not in SOURCE
