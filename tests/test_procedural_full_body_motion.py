import re
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "babyworld_lite"
    / "childlens_engine_bakeoff"
    / "procedural_scene_gate"
    / "FullBodyBimanualMotion.cs"
)


def source() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def test_motion_module_implements_the_frozen_interface_and_uses_one_clock():
    text = source()
    assert "namespace ProceduralSceneGate" in text
    assert "public sealed class FullBodyBimanualMotion : IFullBodyMotionModule" in text
    assert "public void Bind(GateContext context)" in text
    assert "public void ApplyCommand(GateContext context, int physicsStep)" in text
    assert "context.PhysicsStep != physicsStep" in text
    assert "FrozenGate.PhysicsHz" in text
    assert "physicsStep / (float)FrozenGate.PhysicsHz" in text
    for independent_clock in ("Time.time", "Time.deltaTime", "Time.fixedTime", "Update()", "LateUpdate()"):
        assert independent_clock not in text


def test_root_head_both_arms_palms_and_full_rotations_are_bound():
    text = source()
    for binding in ("root", "torso", "neck", "head"):
        assert f'AddBinding("{binding}"' in text
    for bone in ("upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist"):
        assert f'"{bone}"' in text
    assert 'static readonly string[] SideCodes = { "L", "R" }' in text
    assert 'RequireAny(byName, "wrist.L")' in text
    assert 'RequireAny(byName, "wrist.R")' in text
    assert "Quaternion desiredLocal" in text
    assert "rotationWorldXyzw = value.rotation" in text
    assert "angularVelocityWorldRadps" in text


def test_every_digit_and_each_of_three_segments_has_an_individual_command():
    text = source()
    assert "FrozenGate.Digits" in text
    assert "digitClosureCommanded" in text
    assert "ApplyFingerRotation(\"L\", digitIndex, nextLeft)" in text
    assert "ApplyFingerRotation(\"R\", digitIndex, nextRight)" in text
    assert re.search(r"for \(int segment = 1; segment <= 3; segment\+\+\)", text)
    assert '"_segment" + segment' in text
    for digit in ("thumb", "index", "middle", "ring", "little"):
        assert digit in Path(
            SOURCE_PATH.parent / "GateContracts.cs"
        ).read_text(encoding="utf-8")


def test_reusable_sequence_and_qualification_stages_are_explicit():
    text = source()
    for primitive in (
        "Idle",
        "Scan",
        "Gaze",
        "TorsoReorientation",
        "RightReach",
        "Preshape",
        "Touch",
        "ContactAwareClosure",
        "LeftJoin",
        "OpposingSupport",
        "BimanualTurn",
        "Lower",
        "CommandedOpen",
        "Release",
        "Withdraw",
        "FinalGaze",
    ):
        assert primitive in text
    assert "RequiredOppositionSeconds = 0.25f" in text
    assert 'result.rightDigits.Contains("thumb") && rightNonThumb >= 2' in text
    assert "leftSupportDwellSteps >= requiredSteps" in text
    assert "!rightOppositionQualified || !meaningfulLeftSupportQualified" in text


def test_stage_selector_supports_sweeps_object_free_motion_and_integrated_runs():
    text = source()
    assert 'GetEnvironmentVariable("PROCEDURAL_GATE_STAGE")' in text
    for stage in ("garment_sweep", "motion_camera", "bimanual_cell", "integrated"):
        assert f'"{stage}"' in text
    assert "ApplyGarmentSweep(physicsStep)" in text
    assert "MotionCameraPrimitiveAt(physicsStep)" in text
    assert "context.TargetBody == null" in text


def test_contact_response_is_bounded_and_only_reads_physx_truth():
    text = source()
    assert "MaximumPalmLinearSpeedMps = 0.72f" in text
    assert "MaximumPalmAngularSpeedDegps = 110f" in text
    assert "MaximumFingerAngularSpeedDegps = 180f" in text
    assert "MaximumClosurePerSecond = 0.6f" in text
    assert "MaximumAcceptedContactImpulseNs = 0.08f" in text
    assert "MaximumAcceptedFingerPenetrationM = 0.0022f" in text
    assert "MaximumQualifiedContactSeparationM = 0.0005f" in text
    assert "contact.separationM > MaximumQualifiedContactSeparationM" in text
    assert "contact.availableImpulseNs.magnitude" in text
    assert "-contact.separationM" in text
    assert "YieldingForImpulse" in text
    assert "YieldingForPenetration" in text


def test_controller_truth_exposes_commands_observations_errors_and_recovery():
    text = source()
    assert "public MotionControllerSnapshot ControllerSnapshot" in text
    assert "FullBodyMotionTelemetryProvider : MonoBehaviour, IPhysicsTruthControllerTelemetryProvider" in text
    assert "context.AuthorityRoot.AddComponent<FullBodyMotionTelemetryProvider>()" in text
    assert "ControllerTelemetrySnapshot SampleAfterPhysicsStep(int physicsStep)" in text
    assert "digit_closure_commands = ControllerSnapshot.digits.Select" in text
    assert "MaximumFingerSegmentLinearSpeedMps" in text
    assert 'provenance = "commanded_controller_limit"' in text
    assert "PoseState commanded" in text
    assert "PoseState engineObserved" in text
    assert "positionErrorWorldM" in text
    assert "rotationErrorDeg" in text
    assert "recoveryState" in text
    for qualification_field in (
        "rightOppositionQualified",
        "meaningfulLeftSupportQualified",
        "carryContactsMaintained",
        "rightOppositionDwellSteps",
        "leftSupportDwellSteps",
        "bimanualQualificationStep",
    ):
        assert qualification_field in text
    assert "evidence.rightOpposed && evidence.leftMeaningful" in text
    assert "TruthSource.Commanded" in text
    assert "TruthSource.EngineObserved" in text
    assert "public void ObserveAfterPhysicsStep(GateContext context)" in text
    assert "biological torque unavailable" in text


def test_free_target_has_no_assistance_or_pose_actuation_path():
    text = source()
    forbidden = (
        "SetParent(",
        "AddForce(",
        "AddTorque(",
        "MovePosition(",
        "MoveRotation(",
        "FixedJoint",
        "ConfigurableJoint",
        "isKinematic = true",
        "TargetBody.position =",
        "TargetBody.rotation =",
        "TargetBody.linearVelocity =",
        "TargetBody.angularVelocity =",
    )
    for token in forbidden:
        assert token not in text
    assert "context.TargetBody.worldCenterOfMass" in text
    assert "context.TargetBody.GetComponent<Collider>()" in text
