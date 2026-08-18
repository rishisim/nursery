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


def method_body(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    raise AssertionError(f"unclosed method: {signature}")


def test_motion_module_implements_the_frozen_interface_and_uses_one_clock():
    text = source()
    assert "namespace ProceduralSceneGate" in text
    assert "public sealed class FullBodyBimanualMotion : IFullBodyMotionModule" in text
    assert "public void Bind(GateContext context)" in text
    assert "public void ApplyCommand(GateContext context, int physicsStep)" in text
    assert "public void SynchronizeCompletedPhysicsState(GateContext context, int physicsStep)" in text
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
    assert "FrozenGate.RightForceOppositionSeconds" in text
    assert "FrozenGate.LeftForceSupportSeconds" in text
    assert 'result.rightForceDigits.Contains("thumb")' in text
    assert "rightForceNonThumb >= 2" in text
    assert "leftHasNonLittleForceDigit" in text
    assert "leftSupportDwellSteps >= requiredLeftSteps" in text
    assert "!rightOppositionQualified || !meaningfulLeftSupportQualified" in text


def test_integrated_motion_consumes_the_contiguous_frozen_24_second_plan():
    text = source()
    required_phases = (
        "scan",
        "body_reorientation",
        "anticipatory_right_reach",
        "visible_touch_and_force_capture",
        "left_assistance",
        "unsupported_lift_inspect_turn",
        "lower_place_commanded_open",
        "free_release_settle",
        "both_hand_withdrawal",
        "final_gaze_transition",
    )
    phase_initializer = text[text.index("FrozenActivityPhaseIds") : text.index("};", text.index("FrozenActivityPhaseIds"))]
    assert [value.strip('"') for value in re.findall(r'"([a-z_]+)"', phase_initializer)] == list(required_phases)

    bind_plan = method_body(text, "void BindFrozenActivityPlan(GateContext context)")
    assert "context.ActivityPhasesSeconds.TryGetValue(phaseId, out Vector2 span)" in bind_plan
    assert "Mathf.Approximately(span.x, priorEnd)" in bind_plan
    assert "Mathf.Approximately(priorEnd, FrozenGate.DurationSeconds)" in bind_plan

    primitive = method_body(text, "MotionPrimitive PrimitiveAt(int physicsStep)")
    assert all(f'PhaseEndStep("{phase}")' in primitive for phase in required_phases[:-1])
    assert not re.search(r"StepAt\([0-9]", primitive)
    assert 'SubphaseBoundaryStep("lower_place_commanded_open", 0.50f)' in primitive
    assert "MotionPrimitive.Release" in primitive
    assert "MotionPrimitive.Withdraw" in primitive
    assert primitive.rstrip().endswith("return MotionPrimitive.FinalGaze;")
    for stale_boundary in ("13.0f", "14.4f", "15.8f"):
        assert stale_boundary not in text


def test_destination_contact_and_gaze_labels_have_runtime_geometric_effects():
    text = source()
    binding = method_body(text, "void BindActivitySemantics(GateContext context)")
    assert "destinationId = context.DestinationId" in binding
    assert "finalGazeZoneId = context.FinalGazeZone" in binding
    assert "context.Destinations.TryGetValue(destinationId, out destination)" in binding
    assert "context.Destinations.TryGetValue(finalGazeZoneId, out finalGazeTarget)" in binding
    assert "throw new InvalidOperationException" in binding

    placement = method_body(text, "static Vector3 ResolvePlacementAnchor(")
    assert "destinationTransform.GetComponentsInChildren<Collider>(true)" in placement
    assert "supportSurface.bounds.max.y + targetHalfHeight" in placement
    waypoints = method_body(text, "void DesiredPalmWaypoints(")
    turn_branch = waypoints[waypoints.index("primitive == MotionPrimitive.BimanualTurn") :]
    lower_branch = turn_branch[turn_branch.index("primitive == MotionPrimitive.Lower") :]
    assert "Vector3.Lerp(liftedSource, liftedDestination, transfer)" in turn_branch
    assert "placementAnchorWorld + interactionUpWorld * UnsupportedLiftHeightM" in turn_branch
    assert "Vector3.Lerp(" in lower_branch and "placementAnchorWorld" in lower_branch
    assert lower_branch.index("placementAnchorWorld") < lower_branch.index("primitive == MotionPrimitive.Withdraw")

    gaze = method_body(text, "void ApplyRootAndGaze(int physicsStep, MotionPrimitive primitive)")
    assert "finalGazeTarget.position - head.position" in gaze
    assert "Quaternion.LookRotation(toActualZone.normalized, interactionUpWorld)" in gaze
    assert "Quaternion.Inverse(head.parent.rotation)" in gaze
    assert "HeadCamera" not in gaze


def test_contact_strategies_select_distinct_waypoints_and_force_bearing_digits():
    text = source()
    palm_bias = method_body(text, "Vector3 ContactStrategyPalmBias(string side)")
    closure_scale = method_body(text, "float StrategyDigitClosureScale(string digit)")
    strategy_ids = (
        "radial_thumb_index_middle",
        "cup_body_thumb_middle_ring",
        "face_opposition_thumb_index_middle",
    )
    assert all(strategy in palm_bias and strategy in closure_scale for strategy in strategy_ids)

    bias_returns = re.findall(r"return ([^;]+);", palm_bias)
    assert len(set(bias_returns[:3])) == 3
    assert 'digit == "index" || digit == "middle" ? 1f' in closure_scale
    assert 'digit == "middle" || digit == "ring" ? 1f' in closure_scale
    assert 'digit == "index" ? 0.68f' in closure_scale

    finger_commands = method_body(text, "void ApplyFingerRotations(")
    assert "float targetClosure = closure * StrategyDigitClosureScale(digit);" in finger_commands
    assert "boundedContactClosureLatch" in finger_commands


def test_radial_repair_latches_only_complete_geometry_and_force_bearing_digits():
    text = source()
    command = method_body(text, "public void ApplyCommand(GateContext context, int physicsStep)")
    fingers = method_body(text, "void ApplyFingerRotations(")
    rotations = method_body(text, "void ApplyFingerRotation(string side, int digitIndex, float closure)")
    assert 'evidence.rightDigits.Contains("thumb")' in command
    assert "measuredRightNonThumbDigits >= 2" in command
    assert "evidence.rightForceDigits.Contains(digit)" in fingers
    assert "evidence.leftForceDigits.Contains(digit)" in fingers
    assert "measuredNonThumbDigits >= 1 ? 0.82f : 0.04f" in fingers
    assert "abduction -= middleFlexionPlaneCorrection" in rotations


def test_turn_release_and_impulse_dwell_remain_strict():
    text = source()
    assert "const float CommandedObjectTurnDeg = 35f;" in text
    assert "Quaternion.AngleAxis(CommandedObjectTurnDeg, interactionUpWorld)" in text
    assert "placementAnchorWorld" in method_body(text, "void DesiredPalmWaypoints(")
    assert int(0.30 * 240) + 1 == 73
    assert int(0.25 * 240) + 1 == 61
    assert "Mathf.FloorToInt(FrozenGate.RightForceOppositionSeconds * FrozenGate.PhysicsHz) + 1" in text
    assert "Mathf.FloorToInt(FrozenGate.LeftForceSupportSeconds * FrozenGate.PhysicsHz) + 1" in text
    assert "RecordSupportContinuityFailure" in text


def test_stage_selector_supports_sweeps_object_free_motion_and_integrated_runs():
    text = source()
    assert 'GetEnvironmentVariable("PROCEDURAL_GATE_STAGE")' in text
    for stage in ("clipping", "motion_camera", "microcell", "polished_cell", "integrated"):
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
    assert "MaximumAcceptedFingerPenetrationM = FrozenGate.FingerObjectPenetrationMaxM" in text
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
    assert "commanded_compliant_joint_drive" in text
    assert "biological torque unavailable" in text


def test_finger_commands_drive_dynamic_joints_and_skin_follows_completed_physx():
    text = source()
    assert "BindCompliantFingerAuthority(context)" in text
    assert "body == null || body.isKinematic" in text
    assert "context.FingerJoints.TryGetValue" in text
    assert "SetCompliantFingerDrive" in text
    assert "joint.targetRotation = JointSpaceTargetRotation" in text
    assert "body.WakeUp()" in text
    assert "body.transform.TransformPoint(authorityBonePositionInBody[key])" in text
    assert "body.rotation * authorityBoneRotationInBody[key]" in text
    assert "bone.SetPositionAndRotation(position, rotation)" in text
    assert "the prior completed PhysX finger state was not synchronized" in text
    assert "SetLocalRotationBounded(bone, desired, MaximumFingerAngularSpeedDegps)" not in text


def test_qualification_requires_simultaneous_physx_impulse_opposition_and_continuity():
    text = source()
    assert "contact.provenance == TruthSource.PhysXMeasured" in text
    assert "impulse > MinimumQualifiedImpulseNs" in text
    assert "contact.separationM >= -MaximumAcceptedFingerPenetrationM" in text
    assert "rightThumbForceNormals" in text
    assert "rightNonThumbForceNormals" in text
    assert "Vector3.Dot(thumbNormal, fingerNormal) <= -0.25f" in text
    assert "Mathf.FloorToInt(FrozenGate.RightForceOppositionSeconds * FrozenGate.PhysicsHz) + 1" in text
    assert "Mathf.FloorToInt(FrozenGate.LeftForceSupportSeconds * FrozenGate.PhysicsHz) + 1" in text
    assert "RecordSupportContinuityFailure" in text
    assert "supportContinuityViolated" in text
    assert "beforeCommandedOpening" in text
    assert "initialOverlapDisqualified" in text


def test_prequalification_tracking_and_recovery_are_disclosed_without_object_authority():
    text = source()
    assert "bounded_prequalification_hand_target_tracking" in text
    assert "reads measured worldCenterOfMass only" in text
    assert "context.AuthorityAudit.recoveryCounter++" in text
    assert "context.RecoveryLedger.Add" in text
    assert "never modifies the free object" in text


def test_free_target_has_no_assistance_or_pose_actuation_path():
    text = source()
    forbidden = (
        "SetParent(",
        "AddForce(",
        "AddTorque(",
        "MovePosition(",
        "MoveRotation(",
        "FixedJoint",
        "isKinematic = true",
        "TargetBody.position =",
        "TargetBody.rotation =",
        "TargetBody.linearVelocity =",
        "TargetBody.angularVelocity =",
    )
    for token in forbidden:
        assert token not in text
    assert "context.TargetBody.worldCenterOfMass" in text
    assert "context.TargetBody.GetComponentInChildren<Collider>()" in text
    assert "context.TargetJoint" not in text
