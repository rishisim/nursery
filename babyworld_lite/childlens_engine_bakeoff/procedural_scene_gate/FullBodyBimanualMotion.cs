#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace ProceduralSceneGate
{
    public enum MotionPrimitive
    {
        Idle,
        GarmentSweep,
        Scan,
        Gaze,
        TorsoReorientation,
        RightReach,
        Preshape,
        Touch,
        ContactAwareClosure,
        LeftJoin,
        OpposingSupport,
        BimanualTurn,
        Lower,
        CommandedOpen,
        Release,
        Withdraw,
        FinalGaze
    }

    public enum MotionRecoveryState
    {
        Nominal,
        SeekingContact,
        HoldingForRightOpposition,
        HoldingForLeftSupport,
        YieldingForPenetration,
        YieldingForImpulse
    }

    [Serializable]
    public sealed class MotionChannelState
    {
        public string binding;
        public PoseState commanded;
        public PoseState engineObserved;
        public Vector3 positionErrorWorldM;
        public float rotationErrorDeg;
        public TruthSource commandedProvenance = TruthSource.Commanded;
        public TruthSource observedProvenance = TruthSource.EngineObserved;
    }

    [Serializable]
    public sealed class MotionDigitState
    {
        public string hand;
        public string digit;
        public string[] segmentBindings;
        public float closureCommanded;
        public float closureObserved;
        public float closureError;
    }

    [Serializable]
    public sealed class MotionControllerSnapshot
    {
        public const string Schema = "embodied.full_body_motion_controller.v1";
        public int physicsStep;
        public float physicsTimeSeconds;
        public string activePrimitive;
        public string recoveryState;
        public MotionChannelState[] channels;
        public MotionDigitState[] digits;
        public Vector3 leftPalmWaypointWorldM;
        public Vector3 rightPalmWaypointWorldM;
        public float leftClosureCommanded;
        public float rightClosureCommanded;
        public bool rightOppositionQualified;
        public bool meaningfulLeftSupportQualified;
        public int rightOppositionDwellSteps;
        public int leftSupportDwellSteps;
        public float maximumPalmLinearSpeedMps;
        public float maximumPalmAngularSpeedDegps;
        public float maximumFingerAngularSpeedDegps;
        public float maximumClosurePerSecond;
        public float maximumAcceptedContactImpulseNs;
        public float maximumAcceptedFingerPenetrationM;
        public string controlClaim = "kinematic engineering control; biological torque unavailable";
    }

    public sealed class FullBodyMotionTelemetryProvider : MonoBehaviour, IPhysicsTruthControllerTelemetryProvider
    {
        FullBodyBimanualMotion owner;

        public void Configure(FullBodyBimanualMotion motion)
        {
            owner = motion ?? throw new ArgumentNullException(nameof(motion));
        }

        public ControllerTelemetrySnapshot SampleAfterPhysicsStep(int physicsStep)
        {
            if (owner == null)
                throw new InvalidOperationException("full-body motion telemetry provider is not configured");
            return owner.SampleAfterPhysicsStep(physicsStep);
        }
    }

    /// <summary>
    /// Physics-step-clocked engineering controller for the single weighted avatar.
    /// It only commands the bound embodiment. The free target is read as measured
    /// context and is never assigned, attached, forced, or repaired by this module.
    /// </summary>
    public sealed class FullBodyBimanualMotion : IFullBodyMotionModule
    {
        public const float MaximumPalmLinearSpeedMps = 0.72f;
        public const float MaximumPalmAngularSpeedDegps = 110f;
        public const float MaximumArmJointAngularSpeedDegps = 150f;
        public const float MaximumFingerAngularSpeedDegps = 180f;
        public const float MaximumFingerSegmentLinearSpeedMps = 0.45f;
        public const float MaximumClosurePerSecond = 0.6f;
        public const float MaximumAcceptedContactImpulseNs = 0.08f;
        public const float MaximumAcceptedFingerPenetrationM = 0.0022f;
        public const float MaximumQualifiedContactSeparationM = 0.0005f;
        public const float RequiredOppositionSeconds = 0.25f;

        const float RootLinearSpeedMps = 0.10f;
        const float BodyAngularSpeedDegps = 50f;
        const float ContactShellM = 0.018f;
        const float GripClampM = 0f;
        const int ArmSolverPasses = 4;

        static readonly string[] SideCodes = { "L", "R" };
        static readonly string[] ArmBoneStems =
        {
            "upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist"
        };

        readonly Dictionary<string, Transform> bindings = new Dictionary<string, Transform>();
        readonly Dictionary<Transform, Quaternion> restLocalRotations = new Dictionary<Transform, Quaternion>();
        readonly Dictionary<string, Vector3> restPointsInAvatar = new Dictionary<string, Vector3>();
        readonly Dictionary<string, PoseState> priorCommanded = new Dictionary<string, PoseState>();
        readonly Dictionary<string, PoseState> commanded = new Dictionary<string, PoseState>();
        readonly Dictionary<string, PoseState> observed = new Dictionary<string, PoseState>();
        readonly Dictionary<string, float> digitClosureObserved = new Dictionary<string, float>();
        readonly Dictionary<string, float> digitClosureCommanded = new Dictionary<string, float>();
        readonly Dictionary<string, float> boundedContactClosureLatch = new Dictionary<string, float>();
        readonly Dictionary<string, float> qualifiedContactClosureBase = new Dictionary<string, float>();
        readonly Dictionary<string, float> qualifiedPreloadClosureTarget = new Dictionary<string, float>();

        Transform avatar;
        Transform root;
        Transform torso;
        Transform neck;
        Transform head;
        Transform leftPalm;
        Transform rightPalm;
        Vector3 rootRestPosition;
        Quaternion rootRestRotation;
        Vector3 interactionAnchorWorld;
        Vector3 interactionRightWorld;
        Vector3 interactionUpWorld;
        Vector3 interactionForwardWorld;
        Vector3 previousLeftWaypoint;
        Vector3 previousRightWaypoint;
        Vector3 latchedLeftContactWaypoint;
        Vector3 latchedRightContactWaypoint;
        Vector3 qualifiedSqueezeAxisWorld;
        float targetContactRadiusM;
        float leftClosure;
        float rightClosure;
        int rightOppositionDwellSteps;
        int leftSupportDwellSteps;
        int bimanualQualificationStep = -1;
        bool rightOppositionQualified;
        bool meaningfulLeftSupportQualified;
        bool carryContactsMaintained;
        bool leftContactWaypointLatched;
        bool rightContactWaypointLatched;
        bool bound;
        OperatingStage operatingStage;
        GateContext boundContext;

        public MotionControllerSnapshot ControllerSnapshot { get; private set; }

        public void Bind(GateContext context)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (context.AvatarRoot == null) throw new InvalidOperationException("AvatarRoot is required before motion binding");
            if (context.AuthorityRoot == null) throw new InvalidOperationException("AuthorityRoot is required before motion binding");
            operatingStage = ReadOperatingStage();
            if (context.TargetBody == null &&
                (operatingStage == OperatingStage.BimanualCell || operatingStage == OperatingStage.Integrated))
                throw new InvalidOperationException("a free PhysX target is required for bimanual and integrated motion stages");
            boundedContactClosureLatch.Clear();
            qualifiedContactClosureBase.Clear();
            qualifiedPreloadClosureTarget.Clear();

            avatar = context.AvatarRoot.transform;
            var byName = avatar.GetComponentsInChildren<Transform>(true)
                .GroupBy(value => value.name)
                .ToDictionary(group => group.Key, group => group.First());

            root = avatar;
            torso = context.Torso != null ? context.Torso : RequireAny(byName, "spine03", "spine02");
            neck = context.Neck != null ? context.Neck : RequireAny(byName, "neck01", "neck");
            head = context.Head != null ? context.Head : RequireAny(byName, "head");
            leftPalm = context.LeftPalm != null ? context.LeftPalm : RequireAny(byName, "wrist.L");
            rightPalm = context.RightPalm != null ? context.RightPalm : RequireAny(byName, "wrist.R");

            AddBinding("root", root);
            AddBinding("torso", torso);
            AddBinding("neck", neck);
            AddBinding("head", head);
            foreach (string side in SideCodes)
            {
                string hand = side == "L" ? "left" : "right";
                foreach (string stem in ArmBoneStems)
                {
                    Transform bone = RequireAny(byName, stem + "." + side);
                    AddBinding(hand + "_" + stem, bone);
                }
                for (int digitIndex = 0; digitIndex < FrozenGate.Digits.Length; digitIndex++)
                {
                    string digit = FrozenGate.Digits[digitIndex];
                    string fingerKey = hand + "_" + digit;
                    Transform[] segments = ResolveFingerSegments(context, byName, fingerKey, digitIndex + 1, side);
                    for (int segmentIndex = 0; segmentIndex < 3; segmentIndex++)
                        AddBinding(fingerKey + "_segment" + (segmentIndex + 1), segments[segmentIndex]);
                }
            }

            context.Torso = torso;
            context.Neck = neck;
            context.Head = head;
            context.LeftPalm = leftPalm;
            context.RightPalm = rightPalm;
            boundContext = context;

            rootRestPosition = root.position;
            rootRestRotation = root.rotation;
            foreach (var row in bindings)
            {
                if (!restLocalRotations.ContainsKey(row.Value)) restLocalRotations.Add(row.Value, row.Value.localRotation);
                restPointsInAvatar[row.Key] = avatar.InverseTransformPoint(row.Value.position);
            }

            interactionRightWorld = rootRestRotation * Vector3.right;
            interactionUpWorld = rootRestRotation * Vector3.up;
            interactionForwardWorld = rootRestRotation * Vector3.forward;
            interactionAnchorWorld = context.TargetBody != null
                ? context.TargetBody.worldCenterOfMass
                : rootRestPosition + interactionUpWorld * 0.62f + interactionForwardWorld * 0.38f + interactionRightWorld * 0.18f;
            Collider targetCollider = context.TargetBody == null ? null : context.TargetBody.GetComponent<Collider>();
            targetContactRadiusM = targetCollider == null
                ? 0.045f
                : Mathf.Max(targetCollider.bounds.extents.x, Mathf.Max(targetCollider.bounds.extents.y, targetCollider.bounds.extents.z));
            targetContactRadiusM += ContactShellM;

            previousLeftWaypoint = leftPalm.position;
            previousRightWaypoint = rightPalm.position;
            ControllerSnapshot = NewSnapshot(0, MotionPrimitive.Idle, MotionRecoveryState.Nominal);
            CaptureCommanded(0);
            ObserveAfterPhysicsStep(context);
            FullBodyMotionTelemetryProvider provider = context.AuthorityRoot.GetComponent<FullBodyMotionTelemetryProvider>();
            if (provider == null) provider = context.AuthorityRoot.AddComponent<FullBodyMotionTelemetryProvider>();
            provider.Configure(this);
            bound = true;
        }

        public void ApplyCommand(GateContext context, int physicsStep)
        {
            RequireBound(context, physicsStep);
            if (physicsStep > FrozenGate.PhysicsHz * 20)
                throw new ArgumentOutOfRangeException(nameof(physicsStep), "motion commands are frozen to a maximum 20 second episode");

            if (context.TargetBody != null && (!rightOppositionQualified || !meaningfulLeftSupportQualified))
                interactionAnchorWorld = context.TargetBody.worldCenterOfMass;

            ContactEvidence evidence = ReadMeasuredContactEvidence(context, physicsStep);
            UpdateQualification(evidence);
            if (rightOppositionQualified && meaningfulLeftSupportQualified && bimanualQualificationStep < 0)
            {
                bimanualQualificationStep = physicsStep;
                if (context.TargetBody != null)
                    interactionAnchorWorld = context.TargetBody.worldCenterOfMass;
                Vector3 rightMeanNormal = MeanDirection(evidence.rightNormals);
                Vector3 leftMeanNormal = MeanDirection(evidence.leftNormals);
                qualifiedSqueezeAxisWorld = (rightMeanNormal - leftMeanNormal).normalized;
                if (qualifiedSqueezeAxisWorld.sqrMagnitude < 0.99f)
                    throw new InvalidOperationException("qualified bimanual contact normals did not define a squeeze axis");
                foreach (string digit in evidence.rightDigits.Distinct())
                {
                    string key = "right_" + digit;
                    if (!boundedContactClosureLatch.TryGetValue(key, out float closure))
                        digitClosureCommanded.TryGetValue(key, out closure);
                    qualifiedContactClosureBase[key] = closure;
                    qualifiedPreloadClosureTarget[key] = Mathf.Clamp01(closure + QualifiedPreloadClosureDelta(key));
                    boundedContactClosureLatch[key] = qualifiedPreloadClosureTarget[key];
                }
                foreach (string digit in evidence.leftDigits.Distinct())
                {
                    string key = "left_" + digit;
                    if (!boundedContactClosureLatch.TryGetValue(key, out float closure))
                        digitClosureCommanded.TryGetValue(key, out closure);
                    qualifiedContactClosureBase[key] = closure;
                    qualifiedPreloadClosureTarget[key] = Mathf.Clamp01(closure + QualifiedPreloadClosureDelta(key));
                    boundedContactClosureLatch[key] = qualifiedPreloadClosureTarget[key];
                }
            }
            MotionPrimitive primitive = operatingStage == OperatingStage.MotionCamera
                ? MotionCameraPrimitiveAt(physicsStep)
                : operatingStage == OperatingStage.GarmentSweep
                    ? MotionPrimitive.GarmentSweep
                    : PrimitiveAt(physicsStep);
            int measuredRightNonThumbDigits = evidence.rightDigits.Count(value => value != "thumb");
            if (measuredRightNonThumbDigits >= 1 &&
                (!rightContactWaypointLatched || rightOppositionDwellSteps == 1) &&
                Between(primitive, MotionPrimitive.Touch, MotionPrimitive.BimanualTurn))
            {
                rightContactWaypointLatched = true;
                latchedRightContactWaypoint = rightPalm.position;
            }
            bool measuredLeftSupportSurface = evidence.leftPalm ||
                evidence.leftDigits.Any(value => value != "little");
            if (evidence.leftAny && measuredLeftSupportSurface &&
                !leftContactWaypointLatched &&
                Between(primitive, MotionPrimitive.LeftJoin, MotionPrimitive.BimanualTurn))
            {
                leftContactWaypointLatched = true;
                latchedLeftContactWaypoint = leftPalm.position;
            }
            MotionRecoveryState recovery = RecoveryFor(primitive, evidence);
            carryContactsMaintained = bimanualQualificationStep >= 0 && evidence.rightOpposed && evidence.leftMeaningful;
            RegulateQualifiedGrip(evidence, physicsStep);
            context.AnatomicalColliderVelocityDriveCommanded =
                rightOppositionQualified && meaningfulLeftSupportQualified &&
                (primitive == MotionPrimitive.BimanualTurn || primitive == MotionPrimitive.Lower);

            if (operatingStage == OperatingStage.GarmentSweep)
            {
                ApplyGarmentSweep(physicsStep);
                ControllerSnapshot = NewSnapshot(physicsStep, primitive, MotionRecoveryState.Nominal);
                CaptureCommanded(physicsStep);
                PopulateSnapshotChannels();
                return;
            }

            ApplyRootAndGaze(physicsStep, primitive);
            DesiredPalmWaypoints(physicsStep, primitive, out Vector3 desiredLeft, out Vector3 desiredRight);
            float maximumPalmStepM = MaximumPalmLinearSpeedMps / FrozenGate.PhysicsHz;
            previousLeftWaypoint = Vector3.MoveTowards(previousLeftWaypoint, desiredLeft, maximumPalmStepM);
            previousRightWaypoint = Vector3.MoveTowards(previousRightWaypoint, desiredRight, maximumPalmStepM);

            SolveArm("L", previousLeftWaypoint, DesiredPalmRotation("L", physicsStep, primitive));
            SolveArm("R", previousRightWaypoint, DesiredPalmRotation("R", physicsStep, primitive));

            float desiredRightClosure = DesiredClosure("R", physicsStep, primitive, evidence, recovery);
            float desiredLeftClosure = DesiredClosure("L", physicsStep, primitive, evidence, recovery);
            float closureStep = MaximumClosurePerSecond / FrozenGate.PhysicsHz;
            rightClosure = Mathf.MoveTowards(rightClosure, desiredRightClosure, closureStep);
            leftClosure = Mathf.MoveTowards(leftClosure, desiredLeftClosure, closureStep);
            ApplyFingerRotations("R", rightClosure, primitive, evidence);
            ApplyFingerRotations("L", leftClosure, primitive, evidence);

            ControllerSnapshot = NewSnapshot(physicsStep, primitive, recovery);
            CaptureCommanded(physicsStep);
            PopulateSnapshotChannels();
        }

        /// <summary>
        /// Called by the truth recorder after the same Physics.Simulate step. If a
        /// recorder does not call it, the next ApplyCommand refreshes observations
        /// before issuing the following step's command.
        /// </summary>
        public void ObserveAfterPhysicsStep(GateContext context)
        {
            if (ControllerSnapshot == null || bindings.Count == 0) return;
            int physicsStep = context == null ? ControllerSnapshot.physicsStep : context.PhysicsStep;
            foreach (var row in bindings)
                observed[row.Key] = PoseFor(row.Value, physicsStep, observed);
            ObserveDigitClosures();
            PopulateSnapshotChannels();
        }

        public ControllerTelemetrySnapshot SampleAfterPhysicsStep(int physicsStep)
        {
            if (!bound || boundContext == null) throw new InvalidOperationException("motion telemetry requested before Bind");
            if (ControllerSnapshot == null || ControllerSnapshot.physicsStep != physicsStep)
                throw new InvalidOperationException("motion telemetry clock does not match the just-completed physics step");
            if (boundContext.PhysicsStep != physicsStep)
                throw new InvalidOperationException("GateContext.PhysicsStep changed before controller telemetry sampling");

            ObserveAfterPhysicsStep(boundContext);
            ControllerTargetTruth[] targets = ControllerSnapshot.channels.Select(channel => new ControllerTargetTruth
            {
                control_id = channel.binding,
                position_world_m = channel.commanded.positionWorldM,
                rotation_world_xyzw = channel.commanded.rotationWorldXyzw,
                provenance = "commanded",
                formula_or_source = "FullBodyBimanualMotion command held for this PhysicsStep"
            }).ToArray();
            var errors = ControllerSnapshot.channels.Select(channel => new ControllerErrorTruth
            {
                control_id = channel.binding,
                position_error_m = channel.positionErrorWorldM.magnitude,
                rotation_error_deg = channel.rotationErrorDeg,
                closure_error = 0f,
                provenance = "derived",
                formula_or_source = "commanded target minus post-Physics.Simulate engine-observed pose"
            }).ToList();
            errors.AddRange(ControllerSnapshot.digits.Select(digit => new ControllerErrorTruth
            {
                control_id = digit.hand + "_" + digit.digit + "_closure",
                position_error_m = 0f,
                rotation_error_deg = 0f,
                closure_error = digit.closureError,
                provenance = "derived",
                formula_or_source = "commanded closure minus post-Physics.Simulate rotation-derived closure"
            }));

            return new ControllerTelemetrySnapshot
            {
                physics_step = physicsStep,
                phase_id = ControllerSnapshot.activePrimitive,
                targets = targets,
                errors = errors.ToArray(),
                digit_closure_commands = ControllerSnapshot.digits.Select(digit => new DigitClosureCommandTruth
                {
                    hand = digit.hand,
                    digit = digit.digit,
                    closure_commanded = digit.closureCommanded,
                    provenance = "commanded"
                }).ToArray(),
                recovery_state = ControllerSnapshot.recoveryState,
                recovery_state_provenance = "engine_observed",
                right_opposition_qualified = rightOppositionQualified,
                meaningful_left_support_qualified = meaningfulLeftSupportQualified,
                carry_contacts_maintained = carryContactsMaintained,
                right_opposition_dwell_steps = rightOppositionDwellSteps,
                left_support_dwell_steps = leftSupportDwellSteps,
                bimanual_qualification_step = bimanualQualificationStep,
                speed_limits = new SpeedLimitsTruth
                {
                    palm_linear_max_m_s = MaximumPalmLinearSpeedMps,
                    palm_angular_max_rad_s = MaximumPalmAngularSpeedDegps * Mathf.Deg2Rad,
                    finger_segment_linear_max_m_s = MaximumFingerSegmentLinearSpeedMps,
                    finger_segment_angular_max_rad_s = MaximumFingerAngularSpeedDegps * Mathf.Deg2Rad,
                    provenance = "commanded_controller_limit"
                },
                closure_limits = new ClosureLimitsTruth
                {
                    closure_min = 0f,
                    closure_max = 1f,
                    closure_rate_max_s = MaximumClosurePerSecond,
                    finger_object_penetration_stop_m = MaximumAcceptedFingerPenetrationM,
                    provenance = "commanded_controller_limit"
                }
            };
        }

        static Transform RequireAny(Dictionary<string, Transform> byName, params string[] names)
        {
            foreach (string name in names)
                if (byName.TryGetValue(name, out Transform result)) return result;
            throw new InvalidOperationException("weighted avatar is missing required bone: " + string.Join(" or ", names));
        }

        void AddBinding(string name, Transform value)
        {
            if (value == null) throw new InvalidOperationException("null weighted bone binding: " + name);
            bindings[name] = value;
        }

        static Transform[] ResolveFingerSegments(
            GateContext context,
            Dictionary<string, Transform> byName,
            string fingerKey,
            int digitIndex,
            string side)
        {
            if (context.FingerSegments.TryGetValue(fingerKey, out Transform[] configured) &&
                configured != null && configured.Length == 3 && configured.All(value => value != null))
                return configured;

            var result = new Transform[3];
            for (int segment = 1; segment <= 3; segment++)
                result[segment - 1] = RequireAny(byName, "finger" + digitIndex + "-" + segment + "." + side);
            context.FingerSegments[fingerKey] = result;
            return result;
        }

        void RequireBound(GateContext context, int physicsStep)
        {
            if (!bound) throw new InvalidOperationException("Bind must complete before ApplyCommand");
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (context.AvatarRoot == null || context.AvatarRoot.transform != avatar)
                throw new InvalidOperationException("the motion module cannot switch authority roots after Bind");
            if (context.PhysicsStep != physicsStep)
                throw new InvalidOperationException("ApplyCommand must use GateContext.PhysicsStep as its only clock");
        }

        static int StepAt(float seconds)
        {
            return Mathf.RoundToInt(seconds * FrozenGate.PhysicsHz);
        }

        static float PhaseProgress(int physicsStep, float startSeconds, float endSeconds)
        {
            int start = StepAt(startSeconds);
            int end = StepAt(endSeconds);
            return Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(start, end, physicsStep));
        }

        static MotionPrimitive PrimitiveAt(int physicsStep)
        {
            if (physicsStep < StepAt(1.8f)) return MotionPrimitive.Scan;
            if (physicsStep < StepAt(3.0f)) return MotionPrimitive.TorsoReorientation;
            if (physicsStep < StepAt(4.2f)) return MotionPrimitive.RightReach;
            if (physicsStep < StepAt(5.0f)) return MotionPrimitive.Preshape;
            if (physicsStep < StepAt(6.2f)) return MotionPrimitive.Touch;
            if (physicsStep < StepAt(6.6f)) return MotionPrimitive.ContactAwareClosure;
            if (physicsStep < StepAt(7.55f)) return MotionPrimitive.LeftJoin;
            if (physicsStep < StepAt(8.0f)) return MotionPrimitive.OpposingSupport;
            if (physicsStep < StepAt(11.3f)) return MotionPrimitive.BimanualTurn;
            if (physicsStep < StepAt(12.2f)) return MotionPrimitive.Lower;
            if (physicsStep < StepAt(12.8f)) return MotionPrimitive.CommandedOpen;
            if (physicsStep < StepAt(13.0f)) return MotionPrimitive.Release;
            if (physicsStep < StepAt(14.4f)) return MotionPrimitive.Withdraw;
            return MotionPrimitive.FinalGaze;
        }

        static MotionPrimitive MotionCameraPrimitiveAt(int physicsStep)
        {
            if (physicsStep < StepAt(1.8f)) return MotionPrimitive.Scan;
            if (physicsStep < StepAt(3.0f)) return MotionPrimitive.TorsoReorientation;
            if (physicsStep < StepAt(6.0f)) return MotionPrimitive.RightReach;
            if (physicsStep < StepAt(12.5f)) return MotionPrimitive.Gaze;
            if (physicsStep < StepAt(14.4f)) return MotionPrimitive.Withdraw;
            return MotionPrimitive.FinalGaze;
        }

        void ApplyGarmentSweep(int physicsStep)
        {
            float t = physicsStep / (float)FrozenGate.PhysicsHz;
            MotionPrimitive bodyPrimitive = physicsStep < StepAt(4f)
                ? MotionPrimitive.Scan
                : physicsStep < StepAt(12f) ? MotionPrimitive.TorsoReorientation : MotionPrimitive.FinalGaze;
            ApplyRootAndGaze(physicsStep, bodyPrimitive);

            Vector3 leftRest = RestPoint("left_wrist");
            Vector3 rightRest = RestPoint("right_wrist");
            float lift = 0.055f * Mathf.Sin(t * 1.35f);
            float reach = 0.075f * Mathf.Sin(t * 0.95f + 0.45f);
            Vector3 leftGoal = leftRest + interactionUpWorld * lift + interactionForwardWorld * reach;
            Vector3 rightGoal = rightRest - interactionUpWorld * lift + interactionForwardWorld * reach;
            float maximumPalmStepM = MaximumPalmLinearSpeedMps / FrozenGate.PhysicsHz;
            previousLeftWaypoint = Vector3.MoveTowards(previousLeftWaypoint, leftGoal, maximumPalmStepM);
            previousRightWaypoint = Vector3.MoveTowards(previousRightWaypoint, rightGoal, maximumPalmStepM);
            SolveArm("L", previousLeftWaypoint, restLocalRotations[leftPalm] * Quaternion.Euler(10f * Mathf.Sin(t), -14f, 8f));
            SolveArm("R", previousRightWaypoint, restLocalRotations[rightPalm] * Quaternion.Euler(-10f * Mathf.Sin(t), 14f, -8f));

            float closureStep = MaximumClosurePerSecond / FrozenGate.PhysicsHz;
            float leftTotal = 0f;
            float rightTotal = 0f;
            for (int digitIndex = 0; digitIndex < FrozenGate.Digits.Length; digitIndex++)
            {
                string digit = FrozenGate.Digits[digitIndex];
                float leftTarget = 0.48f + 0.42f * Mathf.Sin(t * 1.25f + digitIndex * 0.63f);
                float rightTarget = 0.48f + 0.42f * Mathf.Sin(t * 1.25f + digitIndex * 0.63f + 1.1f);
                string leftKey = "left_" + digit;
                string rightKey = "right_" + digit;
                digitClosureCommanded.TryGetValue(leftKey, out float priorLeft);
                digitClosureCommanded.TryGetValue(rightKey, out float priorRight);
                float nextLeft = Mathf.MoveTowards(priorLeft, Mathf.Clamp01(leftTarget), closureStep);
                float nextRight = Mathf.MoveTowards(priorRight, Mathf.Clamp01(rightTarget), closureStep);
                ApplyFingerRotation("L", digitIndex, nextLeft);
                ApplyFingerRotation("R", digitIndex, nextRight);
                leftTotal += nextLeft;
                rightTotal += nextRight;
            }
            leftClosure = leftTotal / FrozenGate.Digits.Length;
            rightClosure = rightTotal / FrozenGate.Digits.Length;
        }

        void ApplyRootAndGaze(int physicsStep, MotionPrimitive primitive)
        {
            float timeSeconds = physicsStep / (float)FrozenGate.PhysicsHz;
            float reorient = PhaseProgress(physicsStep, 1.8f, 3.0f);
            float unwind = PhaseProgress(physicsStep, 13.0f, 15.8f);
            float bodyAmount = reorient * (1f - unwind);
            float scan = primitive == MotionPrimitive.Scan ? Mathf.Sin(timeSeconds * 2.25f) : 0f;
            float inspectActivation = primitive == MotionPrimitive.BimanualTurn &&
                                      rightOppositionQualified && meaningfulLeftSupportQualified
                ? QualifiedPhaseProgress(physicsStep, 2.0f, 2.4f)
                : 0f;
            float inspect = inspectActivation * Mathf.Sin(timeSeconds * 1.7f);
            bool taskView = primitive == MotionPrimitive.Gaze || Between(primitive, MotionPrimitive.RightReach, MotionPrimitive.Release);

            Vector3 desiredRootPosition = rootRestPosition + interactionRightWorld * (0.008f * bodyAmount);
            Quaternion desiredRootRotation = rootRestRotation * Quaternion.Euler(0f, 11f * bodyAmount, -1.5f * bodyAmount);
            root.position = Vector3.MoveTowards(root.position, desiredRootPosition, RootLinearSpeedMps / FrozenGate.PhysicsHz);
            root.rotation = Quaternion.RotateTowards(root.rotation, desiredRootRotation, BodyAngularSpeedDegps / FrozenGate.PhysicsHz);

            float torsoPitch = taskView ? 10f : 0f;
            float torsoYaw = 7f * bodyAmount + 2f * inspect;
            SetLocalRotationBounded(torso, restLocalRotations[torso] * Quaternion.Euler(torsoPitch, torsoYaw, -1.5f * bodyAmount), BodyAngularSpeedDegps);

            float neckPitch = primitive == MotionPrimitive.Gaze || Between(primitive, MotionPrimitive.Preshape, MotionPrimitive.Release) ? 13f : 0f;
            float neckYaw = primitive == MotionPrimitive.Scan ? 4f * scan : primitive == MotionPrimitive.FinalGaze ? -7f * unwind : 2f * inspect;
            SetLocalRotationBounded(neck, restLocalRotations[neck] * Quaternion.Euler(neckPitch, neckYaw, 0f), BodyAngularSpeedDegps);

            float headPitch = taskView ? 44f : 2f;
            float headYaw = primitive == MotionPrimitive.Scan ? 11f * scan : primitive == MotionPrimitive.FinalGaze ? -15f * unwind : 4f * inspect;
            float headRoll = primitive == MotionPrimitive.Scan ? -1.8f * scan : 0f;
            SetLocalRotationBounded(head, restLocalRotations[head] * Quaternion.Euler(headPitch, headYaw, headRoll), BodyAngularSpeedDegps);
        }

        void DesiredPalmWaypoints(int physicsStep, MotionPrimitive primitive, out Vector3 left, out Vector3 right)
        {
            Vector3 leftRest = RestPoint("left_wrist");
            Vector3 rightRest = RestPoint("right_wrist");
            Vector3 rightContact = GraspCenteredPalmWaypoint("R", interactionAnchorWorld);
            Vector3 leftContact = GraspCenteredPalmWaypoint("L", interactionAnchorWorld);
            Vector3 rightStandOff = rightContact + interactionRightWorld * 0.065f;
            Vector3 leftStandOff = leftContact - interactionRightWorld * 0.055f;
            left = leftRest;
            right = rightRest;

            if (primitive == MotionPrimitive.RightReach || primitive == MotionPrimitive.Preshape)
            {
                float reach = PhaseProgress(physicsStep, 3.0f, 4.2f);
                right = Vector3.Lerp(rightRest, rightStandOff, reach);
                return;
            }

            if (primitive == MotionPrimitive.Gaze)
            {
                right = rightContact;
                left = leftRest;
                return;
            }

            if (primitive == MotionPrimitive.Touch)
            {
                float touch = PhaseProgress(physicsStep, 5.0f, 6.2f);
                right = rightContactWaypointLatched
                    ? latchedRightContactWaypoint
                    : Vector3.Lerp(rightStandOff, rightContact, touch);
                return;
            }

            if (primitive == MotionPrimitive.ContactAwareClosure)
            {
                right = rightContactWaypointLatched ? latchedRightContactWaypoint : rightContact;
                return;
            }

            if (primitive == MotionPrimitive.LeftJoin || primitive == MotionPrimitive.OpposingSupport)
            {
                if (!rightOppositionQualified)
                {
                    right = rightContactWaypointLatched ? latchedRightContactWaypoint : rightContact;
                    left = leftRest;
                    return;
                }
                float join = PhaseProgress(physicsStep, 6.6f, 7.8f);
                right = rightContactWaypointLatched ? latchedRightContactWaypoint : rightContact;
                if (primitive == MotionPrimitive.LeftJoin)
                {
                    left = Vector3.Lerp(leftRest, leftStandOff, join);
                }
                else
                {
                    float support = PhaseProgress(physicsStep, 7.55f, 8.0f);
                    left = leftContactWaypointLatched
                        ? latchedLeftContactWaypoint
                        : Vector3.Lerp(leftStandOff, leftContact, support);
                }
                return;
            }

            if (primitive == MotionPrimitive.BimanualTurn)
            {
                if (!rightOppositionQualified)
                {
                    right = rightContactWaypointLatched ? latchedRightContactWaypoint : rightContact;
                    left = leftRest;
                    return;
                }
                if (!meaningfulLeftSupportQualified)
                {
                    right = rightContactWaypointLatched ? latchedRightContactWaypoint : rightContact;
                    left = leftContactWaypointLatched ? latchedLeftContactWaypoint : leftContact;
                    return;
                }
                if (!carryContactsMaintained)
                {
                    right = latchedRightContactWaypoint;
                    left = latchedLeftContactWaypoint;
                    return;
                }
                float lift = QualifiedPhaseProgress(physicsStep, 1.0f, 3.0f);
                float turn = 28f * QualifiedPhaseProgress(physicsStep, 2.1f, 3.0f);
                Vector3 center = interactionAnchorWorld + interactionUpWorld * (0.105f * lift);
                Quaternion turnRotation = Quaternion.AngleAxis(turn, interactionUpWorld);
                float clamp = GripClampM * QualifiedPhaseProgress(physicsStep, 0f, 0.35f);
                Vector3 rightOffset = latchedRightContactWaypoint - interactionAnchorWorld + qualifiedSqueezeAxisWorld * clamp;
                Vector3 leftOffset = latchedLeftContactWaypoint - interactionAnchorWorld - qualifiedSqueezeAxisWorld * clamp;
                right = center + turnRotation * rightOffset;
                left = center + turnRotation * leftOffset;
                return;
            }

            if (primitive == MotionPrimitive.Lower || primitive == MotionPrimitive.CommandedOpen || primitive == MotionPrimitive.Release)
            {
                if (!rightOppositionQualified)
                {
                    right = rightContactWaypointLatched ? latchedRightContactWaypoint : rightContact;
                    left = leftRest;
                    return;
                }
                if (!meaningfulLeftSupportQualified)
                {
                    right = rightContactWaypointLatched ? latchedRightContactWaypoint : rightContact;
                    left = leftContactWaypointLatched ? latchedLeftContactWaypoint : leftContact;
                    return;
                }
                if (!carryContactsMaintained)
                {
                    right = latchedRightContactWaypoint;
                    left = latchedLeftContactWaypoint;
                    return;
                }
                float lower = PhaseProgress(physicsStep, 11.3f, 12.2f);
                float turn = 28f;
                Vector3 center = interactionAnchorWorld + interactionUpWorld * (0.105f * (1f - lower));
                Quaternion turnRotation = Quaternion.AngleAxis(turn, interactionUpWorld);
                Vector3 rightOffset = latchedRightContactWaypoint - interactionAnchorWorld + qualifiedSqueezeAxisWorld * GripClampM;
                Vector3 leftOffset = latchedLeftContactWaypoint - interactionAnchorWorld - qualifiedSqueezeAxisWorld * GripClampM;
                right = center + turnRotation * rightOffset;
                left = center + turnRotation * leftOffset;
                return;
            }

            if (primitive == MotionPrimitive.Withdraw || primitive == MotionPrimitive.FinalGaze)
            {
                float withdraw = PhaseProgress(physicsStep, 13.0f, 14.3f);
                Quaternion turnRotation = Quaternion.AngleAxis(28f, interactionUpWorld);
                Vector3 rightReleased = rightContactWaypointLatched
                    ? interactionAnchorWorld + turnRotation * (latchedRightContactWaypoint - interactionAnchorWorld)
                    : rightContact;
                Vector3 leftReleased = leftContactWaypointLatched
                    ? interactionAnchorWorld + turnRotation * (latchedLeftContactWaypoint - interactionAnchorWorld)
                    : leftContact;
                right = Vector3.Lerp(rightReleased, rightRest, withdraw);
                left = Vector3.Lerp(leftReleased, leftRest, withdraw);
            }
        }

        Quaternion DesiredPalmRotation(string side, int physicsStep, MotionPrimitive primitive)
        {
            Transform palm = side == "L" ? leftPalm : rightPalm;
            float handed = side == "L" ? -1f : 1f;
            float preshape = Between(primitive, MotionPrimitive.Preshape, MotionPrimitive.Release) ? 1f : 0f;
            float turn = Between(primitive, MotionPrimitive.BimanualTurn, MotionPrimitive.Withdraw)
                ? 28f * QualifiedPhaseProgress(physicsStep, 2.1f, 3.0f)
                : 0f;
            float anatomicalRoll = side == "R" ? 15f : 9f;
            return restLocalRotations[palm] * Quaternion.Euler(8f * preshape, handed * (18f * preshape + turn), anatomicalRoll * preshape);
        }

        void SolveArm(string side, Vector3 waypoint, Quaternion desiredPalmLocalRotation)
        {
            string hand = side == "L" ? "left" : "right";
            Transform palm = side == "L" ? leftPalm : rightPalm;
            string[] chainNames =
            {
                hand + "_upperarm01", hand + "_upperarm02", hand + "_lowerarm01", hand + "_lowerarm02"
            };
            float jointBudgetPerPass = MaximumArmJointAngularSpeedDegps / FrozenGate.PhysicsHz / ArmSolverPasses;
            for (int pass = 0; pass < ArmSolverPasses; pass++)
            {
                for (int chainIndex = chainNames.Length - 1; chainIndex >= 0; chainIndex--)
                {
                    Transform joint = bindings[chainNames[chainIndex]];
                    Vector3 currentVector = palm.position - joint.position;
                    Vector3 desiredVector = waypoint - joint.position;
                    if (currentVector.sqrMagnitude < 1e-10f || desiredVector.sqrMagnitude < 1e-10f) continue;
                    float denominator = currentVector.magnitude * desiredVector.magnitude;
                    Vector3 axis = Vector3.Cross(currentVector, desiredVector);
                    float axisMagnitude = axis.magnitude;
                    float angleDeg = Mathf.Atan2(axisMagnitude, Vector3.Dot(currentVector, desiredVector)) * Mathf.Rad2Deg;
                    if (denominator < 1e-10f || axisMagnitude < 1e-10f || angleDeg < 1e-7f) continue;
                    float boundedAngleDeg = Mathf.Min(angleDeg, jointBudgetPerPass);
                    Quaternion worldDelta = Quaternion.AngleAxis(boundedAngleDeg, axis / axisMagnitude);
                    Quaternion desiredWorld = worldDelta * joint.rotation;
                    Quaternion desiredLocal = Quaternion.Inverse(joint.parent.rotation) * desiredWorld;
                    joint.localRotation = desiredLocal;
                }
            }
            SetLocalRotationBounded(palm, desiredPalmLocalRotation, MaximumPalmAngularSpeedDegps);
        }

        void SetLocalRotationBounded(Transform value, Quaternion desired, float speedDegps)
        {
            value.localRotation = Quaternion.RotateTowards(value.localRotation, desired, speedDegps / FrozenGate.PhysicsHz);
        }

        float DesiredClosure(
            string side,
            int physicsStep,
            MotionPrimitive primitive,
            ContactEvidence evidence,
            MotionRecoveryState recovery)
        {
            if (recovery == MotionRecoveryState.YieldingForImpulse || recovery == MotionRecoveryState.YieldingForPenetration)
                return Mathf.Max(0f, (side == "R" ? rightClosure : leftClosure) - 0.12f);
            if (primitive == MotionPrimitive.CommandedOpen)
            {
                float current = side == "R" ? rightClosure : leftClosure;
                return Mathf.Min(current, 0.82f * (1f - PhaseProgress(physicsStep, 12.2f, 12.8f)));
            }
            if (primitive == MotionPrimitive.Release || primitive == MotionPrimitive.Withdraw || primitive == MotionPrimitive.FinalGaze)
                return 0f;

            if (side == "R")
            {
                if (primitive == MotionPrimitive.Preshape) return 0.16f;
                if (primitive == MotionPrimitive.Touch) return 0.16f;
                if (Between(primitive, MotionPrimitive.ContactAwareClosure, MotionPrimitive.Lower))
                    return evidence.rightAny ? 0.82f : 0.48f;
                return 0.08f;
            }

            if (primitive == MotionPrimitive.LeftJoin) return 0.34f;
            if (Between(primitive, MotionPrimitive.OpposingSupport, MotionPrimitive.Lower))
                return 0.46f;
            return 0.06f;
        }

        void ApplyFingerRotations(
            string side,
            float closure,
            MotionPrimitive primitive,
            ContactEvidence evidence)
        {
            for (int digitIndex = 0; digitIndex < FrozenGate.Digits.Length; digitIndex++)
            {
                string hand = side == "L" ? "left" : "right";
                string digit = FrozenGate.Digits[digitIndex];
                float targetClosure = closure;
                bool rightInteraction = side == "R" &&
                    Between(primitive, MotionPrimitive.Preshape, MotionPrimitive.Lower);
                bool leftInteraction = side == "L" && rightOppositionQualified &&
                    Between(primitive, MotionPrimitive.LeftJoin, MotionPrimitive.Lower);
                if (rightInteraction && digit == "thumb")
                {
                    int measuredNonThumbDigits = evidence.rightDigits.Count(value => value != "thumb");
                    targetClosure = Mathf.Min(targetClosure, measuredNonThumbDigits >= 1 ? 0.60f : 0.04f);
                }
                string key = hand + "_" + digit;
                digitClosureCommanded.TryGetValue(key, out float priorClosure);
                bool boundedContact = evidence.maximumImpulseNs <= MaximumAcceptedContactImpulseNs &&
                                      evidence.maximumPenetrationM <= MaximumAcceptedFingerPenetrationM;
                bool digitInMeasuredContact = side == "R"
                    ? evidence.rightDigits.Contains(digit)
                    : evidence.leftDigits.Contains(digit);
                bool interactionContactLatchActive = rightInteraction || leftInteraction;
                if (interactionContactLatchActive && digitInMeasuredContact && boundedContact &&
                    !boundedContactClosureLatch.ContainsKey(key))
                    boundedContactClosureLatch[key] = priorClosure;
                if (!boundedContact && bimanualQualificationStep < 0)
                    boundedContactClosureLatch.Remove(key);
                else if (interactionContactLatchActive && bimanualQualificationStep < 0 && !digitInMeasuredContact)
                    boundedContactClosureLatch.Remove(key);
                if (interactionContactLatchActive && boundedContactClosureLatch.TryGetValue(key, out float latchedClosure))
                    targetClosure = latchedClosure;
                float boundedClosure = Mathf.MoveTowards(
                    priorClosure,
                    targetClosure,
                    MaximumClosurePerSecond / FrozenGate.PhysicsHz);
                ApplyFingerRotation(side, digitIndex, boundedClosure);
            }
        }

        void ApplyFingerRotation(string side, int digitIndex, float closure)
        {
            string hand = side == "L" ? "left" : "right";
            float handed = side == "L" ? -1f : 1f;
            string digit = FrozenGate.Digits[digitIndex];
            digitClosureCommanded[hand + "_" + digit] = closure;
            for (int segment = 1; segment <= 3; segment++)
            {
                Transform bone = bindings[hand + "_" + digit + "_segment" + segment];
                float curlMaximum = digit == "thumb"
                    ? (segment == 1 ? 24f : 34f)
                    : (segment == 1 ? 30f : segment == 2 ? 48f : 42f);
                float abduction = segment == 1 ? handed * (digitIndex - 2f) * 1.8f * (1f - closure) : 0f;
                if (digit == "middle" && segment == 1)
                {
                    float middleFlexionPlaneCorrection = Mathf.Min(side == "R" ? 30f : 40f, 100f * closure);
                    abduction += side == "R"
                        ? middleFlexionPlaneCorrection
                        : -middleFlexionPlaneCorrection;
                }
                if (side == "L" && digit == "index" && segment == 1)
                    abduction -= Mathf.Min(2.1f, 5f * closure);
                float thumbOppositionMaximum = side == "L" ? 25f : 20f;
                float opposition = digit == "thumb"
                    ? side == "R"
                        ? 14f * (1f - closure) - thumbOppositionMaximum * closure
                        : -handed * thumbOppositionMaximum * closure
                    : 0f;
                Quaternion desired = restLocalRotations[bone] * Quaternion.Euler(handed * curlMaximum * closure, opposition, abduction);
                SetLocalRotationBounded(bone, desired, MaximumFingerAngularSpeedDegps);
            }
        }

        ContactEvidence ReadMeasuredContactEvidence(GateContext context, int physicsStep)
        {
            var result = new ContactEvidence();
            if (context.TargetBody == null) return result;
            int measuredStep = Mathf.Max(0, physicsStep - 1);
            string targetName = context.TargetBody.name;
            foreach (ContactTruth contact in context.Contacts.Where(value => value.physicsStep == measuredStep))
            {
                if (!ContainsIdentity(contact.colliderA, targetName) && !ContainsIdentity(contact.colliderB, targetName)) continue;
                if (contact.separationM > MaximumQualifiedContactSeparationM) continue;
                string handCollider = ContainsIdentity(contact.colliderA, targetName) ? contact.colliderB : contact.colliderA;
                string side = SideFor(handCollider);
                string digit = DigitFor(handCollider);
                if (side == "R")
                {
                    result.rightAny = true;
                    if (digit != null) result.rightDigits.Add(digit);
                    result.rightNormals.Add(contact.normalWorld.normalized);
                }
                else if (side == "L")
                {
                    result.leftAny = true;
                    if (digit != null) result.leftDigits.Add(digit);
                    if (ContainsIgnoreCase(handCollider, "palm") || ContainsIgnoreCase(handCollider, "wrist")) result.leftPalm = true;
                    result.leftNormals.Add(contact.normalWorld.normalized);
                }
                if (digit != null && side != null)
                {
                    string key = (side == "R" ? "right_" : "left_") + digit;
                    if (!result.minimumSeparationByDigitM.TryGetValue(key, out float priorSeparation) ||
                        contact.separationM < priorSeparation)
                        result.minimumSeparationByDigitM[key] = contact.separationM;
                    float impulse = contact.availableImpulseNs.magnitude;
                    if (!result.maximumImpulseByDigitNs.TryGetValue(key, out float priorImpulse) || impulse > priorImpulse)
                        result.maximumImpulseByDigitNs[key] = impulse;
                }
                result.maximumImpulseNs = Mathf.Max(result.maximumImpulseNs, contact.availableImpulseNs.magnitude);
                result.maximumPenetrationM = Mathf.Max(result.maximumPenetrationM, Mathf.Max(0f, -contact.separationM));
            }
            int rightNonThumb = result.rightDigits.Count(value => value != "thumb");
            result.rightOpposed = result.rightDigits.Contains("thumb") && rightNonThumb >= 2;
            bool leftNotLittleOnly = result.leftPalm || result.leftDigits.Contains("thumb") ||
                                     result.leftDigits.Count(value => value != "little") >= 2;
            bool normalsOppose = result.rightNormals.Any(rightNormal =>
                result.leftNormals.Any(leftNormal => Vector3.Dot(rightNormal, leftNormal) <= -0.25f));
            result.leftMeaningful = result.leftAny && leftNotLittleOnly && normalsOppose;
            return result;
        }

        void UpdateQualification(ContactEvidence evidence)
        {
            rightOppositionDwellSteps = evidence.rightOpposed ? rightOppositionDwellSteps + 1 : 0;
            leftSupportDwellSteps = evidence.leftMeaningful && evidence.rightOpposed && rightOppositionQualified
                ? leftSupportDwellSteps + 1
                : 0;
            int requiredSteps = Mathf.CeilToInt(RequiredOppositionSeconds * FrozenGate.PhysicsHz);
            if (rightOppositionDwellSteps >= requiredSteps) rightOppositionQualified = true;
            if (leftSupportDwellSteps >= requiredSteps) meaningfulLeftSupportQualified = true;
        }

        static MotionRecoveryState BoundedRecoveryFor(MotionPrimitive primitive, ContactEvidence evidence)
        {
            if (evidence.maximumImpulseNs > MaximumAcceptedContactImpulseNs) return MotionRecoveryState.YieldingForImpulse;
            if (evidence.maximumPenetrationM > MaximumAcceptedFingerPenetrationM) return MotionRecoveryState.YieldingForPenetration;
            if (primitive == MotionPrimitive.ContactAwareClosure && !evidence.rightAny) return MotionRecoveryState.SeekingContact;
            return MotionRecoveryState.Nominal;
        }

        MotionRecoveryState RecoveryFor(MotionPrimitive primitive, ContactEvidence evidence, bool includeQualification = true)
        {
            MotionRecoveryState bounded = BoundedRecoveryFor(primitive, evidence);
            if (bounded != MotionRecoveryState.Nominal) return bounded;
            if (includeQualification && primitive == MotionPrimitive.OpposingSupport && !rightOppositionQualified)
                return MotionRecoveryState.HoldingForRightOpposition;
            if (includeQualification && primitive == MotionPrimitive.BimanualTurn && !meaningfulLeftSupportQualified)
                return MotionRecoveryState.HoldingForLeftSupport;
            return MotionRecoveryState.Nominal;
        }

        Vector3 RestPoint(string binding)
        {
            return avatar.TransformPoint(restPointsInAvatar[binding]);
        }

        float QualifiedPhaseProgress(int physicsStep, float startAfterQualificationS, float endAfterQualificationS)
        {
            if (bimanualQualificationStep < 0) return 0f;
            int start = bimanualQualificationStep + StepAt(startAfterQualificationS);
            int end = bimanualQualificationStep + StepAt(endAfterQualificationS);
            return Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(start, end, physicsStep));
        }

        Vector3 GraspCenteredPalmWaypoint(string side, Vector3 objectCenterWorld)
        {
            string hand = side == "L" ? "left" : "right";
            Transform palm = side == "L" ? leftPalm : rightPalm;
            Vector3 thumbTip = bindings[hand + "_thumb_segment3"].position;
            Vector3 indexTip = bindings[hand + "_index_segment3"].position;
            Vector3 middleTip = bindings[hand + "_middle_segment3"].position;
            Vector3 opposedFingerCenter = 0.5f * (indexTip + middleTip);
            Vector3 apertureCenter = 0.5f * (thumbTip + opposedFingerCenter);
            return objectCenterWorld + (palm.position - apertureCenter);
        }

        void CaptureCommanded(int physicsStep)
        {
            priorCommanded.Clear();
            foreach (var row in commanded) priorCommanded[row.Key] = row.Value;
            commanded.Clear();
            foreach (var row in bindings)
                commanded[row.Key] = PoseFor(row.Value, physicsStep, priorCommanded);
        }

        PoseState PoseFor(Transform value, int physicsStep, Dictionary<string, PoseState> previousByBinding)
        {
            string binding = bindings.First(row => row.Value == value).Key;
            PoseState result = new PoseState
            {
                positionWorldM = value.position,
                rotationWorldXyzw = value.rotation,
                linearVelocityWorldMps = Vector3.zero,
                angularVelocityWorldRadps = Vector3.zero
            };
            if (previousByBinding.TryGetValue(binding, out PoseState prior))
            {
                float dt = 1f / FrozenGate.PhysicsHz;
                result.linearVelocityWorldMps = (result.positionWorldM - prior.positionWorldM) / dt;
                result.angularVelocityWorldRadps = AngularVelocity(prior.rotationWorldXyzw, result.rotationWorldXyzw, dt);
            }
            return result;
        }

        static Vector3 AngularVelocity(Quaternion from, Quaternion to, float dt)
        {
            Quaternion delta = to * Quaternion.Inverse(from);
            delta.ToAngleAxis(out float angleDeg, out Vector3 axis);
            if (angleDeg > 180f) angleDeg -= 360f;
            if (Mathf.Abs(angleDeg) < 1e-5f || !float.IsFinite(axis.x) || !float.IsFinite(axis.y) || !float.IsFinite(axis.z))
                return Vector3.zero;
            return axis.normalized * (angleDeg * Mathf.Deg2Rad / dt);
        }

        MotionControllerSnapshot NewSnapshot(int physicsStep, MotionPrimitive primitive, MotionRecoveryState recovery)
        {
            return new MotionControllerSnapshot
            {
                physicsStep = physicsStep,
                physicsTimeSeconds = physicsStep / (float)FrozenGate.PhysicsHz,
                activePrimitive = primitive.ToString(),
                recoveryState = recovery.ToString(),
                channels = Array.Empty<MotionChannelState>(),
                digits = Array.Empty<MotionDigitState>(),
                leftPalmWaypointWorldM = previousLeftWaypoint,
                rightPalmWaypointWorldM = previousRightWaypoint,
                leftClosureCommanded = leftClosure,
                rightClosureCommanded = rightClosure,
                rightOppositionQualified = rightOppositionQualified,
                meaningfulLeftSupportQualified = meaningfulLeftSupportQualified,
                rightOppositionDwellSteps = rightOppositionDwellSteps,
                leftSupportDwellSteps = leftSupportDwellSteps,
                maximumPalmLinearSpeedMps = MaximumPalmLinearSpeedMps,
                maximumPalmAngularSpeedDegps = MaximumPalmAngularSpeedDegps,
                maximumFingerAngularSpeedDegps = MaximumFingerAngularSpeedDegps,
                maximumClosurePerSecond = MaximumClosurePerSecond,
                maximumAcceptedContactImpulseNs = MaximumAcceptedContactImpulseNs,
                maximumAcceptedFingerPenetrationM = MaximumAcceptedFingerPenetrationM
            };
        }

        void PopulateSnapshotChannels()
        {
            if (ControllerSnapshot == null) return;
            ControllerSnapshot.channels = bindings.Select(row =>
            {
                commanded.TryGetValue(row.Key, out PoseState commandedPose);
                observed.TryGetValue(row.Key, out PoseState observedPose);
                return new MotionChannelState
                {
                    binding = row.Key,
                    commanded = commandedPose,
                    engineObserved = observedPose,
                    positionErrorWorldM = commandedPose.positionWorldM - observedPose.positionWorldM,
                    rotationErrorDeg = Quaternion.Angle(commandedPose.rotationWorldXyzw, observedPose.rotationWorldXyzw)
                };
            }).ToArray();

            var digits = new List<MotionDigitState>();
            foreach (string hand in new[] { "left", "right" })
            {
                foreach (string digit in FrozenGate.Digits)
                {
                    string key = hand + "_" + digit;
                    digitClosureObserved.TryGetValue(key, out float closureObservation);
                    digitClosureCommanded.TryGetValue(key, out float closureCommand);
                    digits.Add(new MotionDigitState
                    {
                        hand = hand,
                        digit = digit,
                        segmentBindings = Enumerable.Range(1, 3).Select(segment => key + "_segment" + segment).ToArray(),
                        closureCommanded = closureCommand,
                        closureObserved = closureObservation,
                        closureError = closureCommand - closureObservation
                    });
                }
            }
            ControllerSnapshot.digits = digits.ToArray();
            ControllerSnapshot.leftPalmWaypointWorldM = previousLeftWaypoint;
            ControllerSnapshot.rightPalmWaypointWorldM = previousRightWaypoint;
        }

        void ObserveDigitClosures()
        {
            foreach (string hand in new[] { "left", "right" })
            foreach (string digit in FrozenGate.Digits)
            {
                string key = hand + "_" + digit;
                float angleSum = 0f;
                for (int segment = 1; segment <= 3; segment++)
                {
                    Transform bone = bindings[key + "_segment" + segment];
                    angleSum += Quaternion.Angle(restLocalRotations[bone], bone.localRotation);
                }
                digitClosureObserved[key] = Mathf.Clamp01(angleSum / 120f);
            }
        }

        static bool ContainsIdentity(string colliderName, string identity)
        {
            return !string.IsNullOrEmpty(colliderName) && !string.IsNullOrEmpty(identity) &&
                   colliderName.IndexOf(identity, StringComparison.OrdinalIgnoreCase) >= 0;
        }

        static bool ContainsIgnoreCase(string value, string token)
        {
            return !string.IsNullOrEmpty(value) && value.IndexOf(token, StringComparison.OrdinalIgnoreCase) >= 0;
        }

        static string SideFor(string colliderName)
        {
            if (ContainsIgnoreCase(colliderName, "left") || ContainsIgnoreCase(colliderName, ".L")) return "L";
            if (ContainsIgnoreCase(colliderName, "right") || ContainsIgnoreCase(colliderName, ".R")) return "R";
            return null;
        }

        static string DigitFor(string colliderName)
        {
            for (int index = 0; index < FrozenGate.Digits.Length; index++)
            {
                string digit = FrozenGate.Digits[index];
                if (ContainsIgnoreCase(colliderName, digit) || ContainsIgnoreCase(colliderName, "finger" + (index + 1) + "-"))
                    return digit;
            }
            return null;
        }

        static Vector3 MeanDirection(IEnumerable<Vector3> directions)
        {
            Vector3 sum = Vector3.zero;
            int count = 0;
            foreach (Vector3 direction in directions)
            {
                sum += direction.normalized;
                count++;
            }
            return count == 0 || sum.sqrMagnitude < 1e-8f ? Vector3.zero : sum.normalized;
        }

        static float QualifiedPreloadClosureDelta(string key)
        {
            return 0f;
        }

        void RegulateQualifiedGrip(ContactEvidence evidence, int physicsStep)
        {
            if (bimanualQualificationStep < 0) return;
            if (physicsStep <= bimanualQualificationStep + StepAt(1.0f))
            {
                if (!carryContactsMaintained) return;
                foreach (string key in new[] { "right_thumb", "right_index", "right_middle" })
                {
                    if (!boundedContactClosureLatch.TryGetValue(key, out float current) ||
                        !qualifiedContactClosureBase.TryGetValue(key, out float apertureBase)) continue;
                    bool hasSeparation = evidence.minimumSeparationByDigitM.TryGetValue(key, out float separation);
                    evidence.maximumImpulseByDigitNs.TryGetValue(key, out float impulse);
                    bool requiresPhysicalContact = !hasSeparation || separation > -0.00005f || impulse < 0.002f;
                    if (!requiresPhysicalContact) continue;
                    float cap = key == "right_thumb" ? 0.020f : key == "right_index" ? 0.100f : 0.030f;
                    float conditioned = Mathf.Min(apertureBase + cap, current + 0.0004f);
                    boundedContactClosureLatch[key] = conditioned;
                    qualifiedPreloadClosureTarget[key] = conditioned;
                }
            }
            bool relax = evidence.maximumImpulseNs > MaximumAcceptedContactImpulseNs ||
                         evidence.maximumPenetrationM > MaximumAcceptedFingerPenetrationM;
            bool restore = evidence.maximumImpulseNs < 0.04f &&
                           evidence.maximumPenetrationM < 0.0015f;
            if (!relax && !restore) return;
            float maximumStep = relax ? 0.00015f : 0.00005f;
            foreach (string key in qualifiedPreloadClosureTarget.Keys.ToArray())
            {
                if (!boundedContactClosureLatch.TryGetValue(key, out float current)) continue;
                float target = relax
                    ? qualifiedContactClosureBase[key]
                    : qualifiedPreloadClosureTarget[key];
                boundedContactClosureLatch[key] = Mathf.MoveTowards(current, target, maximumStep);
            }
        }

        static bool Between(MotionPrimitive value, MotionPrimitive first, MotionPrimitive last)
        {
            return (int)value >= (int)first && (int)value <= (int)last;
        }

        static OperatingStage ReadOperatingStage()
        {
            string raw = Environment.GetEnvironmentVariable("PROCEDURAL_GATE_STAGE");
            if (string.IsNullOrWhiteSpace(raw) || raw == "integrated") return OperatingStage.Integrated;
            if (raw == "garment_sweep") return OperatingStage.GarmentSweep;
            if (raw == "motion_camera") return OperatingStage.MotionCamera;
            if (raw == "bimanual_cell") return OperatingStage.BimanualCell;
            throw new InvalidOperationException("unsupported PROCEDURAL_GATE_STAGE: " + raw);
        }

        enum OperatingStage
        {
            GarmentSweep,
            MotionCamera,
            BimanualCell,
            Integrated
        }

        sealed class ContactEvidence
        {
            public readonly HashSet<string> rightDigits = new HashSet<string>();
            public readonly HashSet<string> leftDigits = new HashSet<string>();
            public readonly List<Vector3> rightNormals = new List<Vector3>();
            public readonly List<Vector3> leftNormals = new List<Vector3>();
            public readonly Dictionary<string, float> minimumSeparationByDigitM = new Dictionary<string, float>();
            public readonly Dictionary<string, float> maximumImpulseByDigitNs = new Dictionary<string, float>();
            public bool rightAny;
            public bool leftAny;
            public bool leftPalm;
            public bool rightOpposed;
            public bool leftMeaningful;
            public float maximumImpulseNs;
            public float maximumPenetrationM;
        }
    }
}
#endif
