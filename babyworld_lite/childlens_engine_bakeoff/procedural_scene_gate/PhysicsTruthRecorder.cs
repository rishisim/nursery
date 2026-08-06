#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Unity.Collections;
using UnityEngine;

namespace ProceduralSceneGate
{
    /// <summary>
    /// The motion module implements this read-only bridge so the post-PhysX trace can
    /// retain the command that produced the observed engine state.  The provider must
    /// return the command held during the just-completed physics step; the recorder
    /// never reconstructs commands from observed transforms.
    /// </summary>
    public interface IPhysicsTruthControllerTelemetryProvider
    {
        ControllerTelemetrySnapshot SampleAfterPhysicsStep(int physicsStep);
    }

    [Serializable]
    public sealed class ControllerTelemetrySnapshot
    {
        public int physics_step;
        public string phase_id;
        public ControllerTargetTruth[] targets = Array.Empty<ControllerTargetTruth>();
        public ControllerErrorTruth[] errors = Array.Empty<ControllerErrorTruth>();
        public DigitClosureCommandTruth[] digit_closure_commands = Array.Empty<DigitClosureCommandTruth>();
        public string recovery_state;
        public string recovery_state_provenance = "engine_observed";
        public bool right_opposition_qualified;
        public bool meaningful_left_support_qualified;
        public bool carry_contacts_maintained;
        public int right_opposition_dwell_steps;
        public int left_support_dwell_steps;
        public int bimanual_qualification_step = -1;
        public SpeedLimitsTruth speed_limits = new SpeedLimitsTruth();
        public ClosureLimitsTruth closure_limits = new ClosureLimitsTruth();
    }

    [Serializable]
    public sealed class ControllerTargetTruth
    {
        public string control_id;
        public Vector3 position_world_m;
        public Quaternion rotation_world_xyzw = Quaternion.identity;
        public string provenance = "commanded";
        public string formula_or_source = "motion controller target held for this physics step";
    }

    [Serializable]
    public sealed class ControllerErrorTruth
    {
        public string control_id;
        public float position_error_m;
        public float rotation_error_deg;
        public float closure_error;
        public string provenance = "derived";
        public string formula_or_source = "commanded target minus post-Physics.Simulate engine-observed state";
    }

    [Serializable]
    public sealed class DigitClosureCommandTruth
    {
        public string hand;
        public string digit;
        public float closure_commanded;
        public string provenance = "commanded";
    }

    [Serializable]
    public sealed class SpeedLimitsTruth
    {
        public float palm_linear_max_m_s;
        public float palm_angular_max_rad_s;
        public float finger_segment_linear_max_m_s;
        public float finger_segment_angular_max_rad_s;
        public string provenance = "unavailable";
    }

    [Serializable]
    public sealed class ClosureLimitsTruth
    {
        public float closure_min;
        public float closure_max;
        public float closure_rate_max_s;
        public float finger_object_penetration_stop_m;
        public string provenance = "unavailable";
    }

    /// <summary>Stable catalog identity supplied by SceneCompiler when available.</summary>
    public sealed class PhysicsTruthObjectIdentity : MonoBehaviour
    {
        public string persistent_id;
        public string semantic_id;
        public string instance_id;
    }

    /// <summary>
    /// Deterministic post-step recorder for the single Unity/PhysX authority state.
    /// RecordAfterPhysicsStep is called exactly once immediately after each manual
    /// Physics.Simulate(1/240) call.  This class observes state only.
    /// </summary>
    public sealed class PhysicsTruthRecorder : IPhysicsTruthModule
    {
        public const string TraceFileName = "episode_trace.jsonl";
        public const string ManifestFileName = "episode_trace_manifest.json";
        public const string ClockReceiptFileName = "episode_trace_clock_receipt.json";
        public const string HashReceiptFileName = "episode_trace_hash_receipt.json";
        public const string InteractionSummaryFileName = "interaction_summary.json";
        public const float ObservedClosureNormalizationDeg = 90f;
        private const float Dt = 1f / FrozenGate.PhysicsHz;
        // GateContext stores seconds as a Unity float; near 16 s, adjacent
        // representable values introduce about 1.02 us of quantization. The
        // integer physics/render indices remain the exact clock authority.
        private const float ClockToleranceSeconds = 2e-6f;
        private const float OutOfPhysicsPositionToleranceM = 1e-6f;
        private const float OutOfPhysicsRotationToleranceDeg = 1e-4f;
        private const float OutOfPhysicsLinearVelocityToleranceMps = 1e-6f;
        private const float OutOfPhysicsAngularVelocityToleranceRadps = 1e-6f;

        private readonly Dictionary<string, PriorPose> priorPoses = new Dictionary<string, PriorPose>();
        private readonly Dictionary<string, Quaternion> priorRelativeRotations = new Dictionary<string, Quaternion>();
        private readonly Dictionary<string, Quaternion[]> bindDigitLocalRotations = new Dictionary<string, Quaternion[]>();
        private readonly Dictionary<string, PendingContact> pendingContacts = new Dictionary<string, PendingContact>();
        private readonly HashSet<string> avatarColliderIds = new HashSet<string>();
        private readonly HashSet<string> targetColliderIds = new HashSet<string>();
        private readonly Dictionary<string, string> externalColliderPersistentIds = new Dictionary<string, string>();
        private readonly Dictionary<string, Transform> bodySegments = new Dictionary<string, Transform>();
        private readonly List<PhysicsTruthContactProbe> installedProbes = new List<PhysicsTruthContactProbe>();

        private GateContext boundContext;
        private StreamWriter traceWriter;
        private IPhysicsTruthControllerTelemetryProvider controllerProvider;
        private Collider[] avatarColliders = Array.Empty<Collider>();
        private string tracePath;
        private int rowCount;
        private int firstPhysicsStep = -1;
        private int lastPhysicsStep = -1;
        private int firstRenderFrame = -1;
        private int lastRenderFrame = -1;
        private float firstTimeSeconds;
        private float lastTimeSeconds;
        private float maximumClockErrorSeconds;
        private bool contiguousPhysicsSteps = true;
        private bool contiguousTimes = true;
        private bool exactRenderMapping = true;
        private int unavailableControllerSteps;
        private int controllerClockMismatchSteps;
        private int maximumContactsInStep;
        private int totalContactRows;
        private int maximumAssistanceEntries;
        private int unavailableCameraClearanceSteps;
        private int nonFreeObjectSteps;
        private int cameraParentMismatchSteps;
        private int unavailableDynamicFingerBindings;
        private bool authorityBindingsVerified;
        private TargetAuthoritySnapshot lastCompletedTargetState;
        private TargetAuthoritySnapshot beforePhysicsTargetState;
        private bool hasLastCompletedTargetState;
        private bool hasBeforePhysicsTargetState;
        private int rightMeasuredImpulseDwellSteps;
        private int leftMeasuredImpulseDwellSteps;
        private bool measuredRightQualified;
        private bool measuredLeftQualified;
        private bool objectUnsupportedDuringQualifiedManipulation = true;
        private bool supportContinuousUntilCommandedOpening = true;
        private bool qualificationEverActive;
        private float qualificationTargetCenterY;
        private Quaternion qualificationTargetRotation;
        private bool hasQualificationBaseline;
        private int measuredBimanualQualificationStep = -1;
        private float maximumQualifiedLiftM;
        private float maximumQualifiedTurnDeg;
        private bool initialFingerOverlapDisqualified;
        private float fingerObjectMaximumPenetrationM;
        private float targetSupportMaximumPenetrationM;
        private bool releaseEverCommanded;
        private bool releaseAtRequiredDestination;
        private string lastObservedDestinationSupportId = "";
        private Vector3 priorHeadPosition;
        private Vector3 priorPriorHeadPosition;
        private Quaternion priorHeadRotation;
        private bool hasPriorHead;
        private bool hasPriorPriorHead;
        private bool completed;

        public void Bind(GateContext context)
        {
            if (boundContext != null)
                throw new InvalidOperationException("PhysicsTruthRecorder is already bound");
            ValidateBindings(context);
            boundContext = context;
            Directory.CreateDirectory(context.OutputRoot);
            tracePath = Path.Combine(context.OutputRoot, TraceFileName);
            traceWriter = new StreamWriter(
                new FileStream(tracePath, FileMode.Create, FileAccess.Write, FileShare.Read),
                new UTF8Encoding(false)
            );

            controllerProvider = context.AuthorityRoot
                .GetComponentsInChildren<MonoBehaviour>(true)
                .OfType<IPhysicsTruthControllerTelemetryProvider>()
                .SingleOrDefault();
            // Collider ownership is an explicit embodiment contract. Dynamic palm/finger
            // collision drivers are intentionally reparented outside AvatarRoot, so a
            // hierarchy scan would misclassify those force-producing colliders as an
            // external support for the manipulated object.
            avatarColliders = context.AvatarColliders
                .Where(collider => collider != null)
                .Distinct()
                .ToArray();
            if (avatarColliders.Length == 0)
                throw new InvalidOperationException("GateContext.AvatarColliders must enumerate every anatomical collider");
            foreach (Collider collider in avatarColliders)
                avatarColliderIds.Add(StableTransformId(collider.transform));
            foreach (Collider collider in context.TargetBody.GetComponentsInChildren<Collider>(true))
                targetColliderIds.Add(StableTransformId(collider.transform));
            IndexExternalColliderIdentities(context);
            BindBodySegments(context);
            if (!IsSha256(context.AuthorityAudit.sourceAuditSha256))
                throw new InvalidOperationException("a frozen source-audit SHA-256 must be supplied before truth binding");
            authorityBindingsVerified = IsChildOrSelf(context.AvatarRoot.transform, context.AuthorityRoot.transform)
                                        && IsChildOrSelf(context.HeadCameraMount, context.Head)
                                        && IsChildOrSelf(context.HeadCamera.transform, context.HeadCameraMount);

            foreach (string side in new[] { "left", "right" })
            foreach (string digit in FrozenGate.Digits)
            {
                string key = side + "_" + digit;
                bindDigitLocalRotations[key] = context.FingerSegments[key]
                    .Select(segment => segment.localRotation)
                    .ToArray();
                for (int segment = 1; segment <= 3; segment++)
                {
                    string authorityKey = key + "_segment" + segment;
                    if (!context.FingerBodies.ContainsKey(authorityKey) || context.FingerBodies[authorityKey] == null
                        || !context.FingerJoints.ContainsKey(authorityKey) || context.FingerJoints[authorityKey] == null)
                        unavailableDynamicFingerBindings++;
                }
            }

            lastCompletedTargetState = SampleTargetAuthorityState(context.TargetBody);
            hasLastCompletedTargetState = true;
            if (lastCompletedTargetState.jointIds.Length > 0)
                RegisterAuthorityViolation(
                    context,
                    "target_joint",
                    lastCompletedTargetState.jointIds.Length,
                    "target was bound with a Joint or as a Joint.connectedBody"
                );
            if (!string.IsNullOrWhiteSpace(lastCompletedTargetState.parentId))
                RegisterAuthorityViolation(context, "target_parenting", 1, "target was bound with a parent");
            if (lastCompletedTargetState.isKinematic)
                RegisterAuthorityViolation(context, "target_kinematic", 1, "target was bound as kinematic");

            IEnumerable<Collider> observedColliders = avatarColliders
                .Concat(context.TargetBody.GetComponentsInChildren<Collider>(true))
                .Where(collider => collider != null)
                .Distinct();
            foreach (Collider collider in observedColliders)
            {
                collider.providesContacts = true;
                PhysicsTruthContactProbe probe = collider.gameObject.GetComponent<PhysicsTruthContactProbe>();
                if (probe == null)
                    probe = collider.gameObject.AddComponent<PhysicsTruthContactProbe>();
                probe.Configure(this);
                installedProbes.Add(probe);
            }
            Physics.ContactEvent += ObserveContactEvent;
        }

        /// <summary>
        /// Read-only boundary sample taken after all controller commands and
        /// Physics.SyncTransforms, immediately before the sole Physics.Simulate call.
        /// A difference from the prior completed state occurred outside PhysX.
        /// </summary>
        public void RecordBeforePhysicsStep(GateContext context)
        {
            RequireActiveContext(context);
            TargetAuthoritySnapshot current = SampleTargetAuthorityState(context.TargetBody);
            if (hasLastCompletedTargetState)
                AuditOutOfPhysicsChanges(context, lastCompletedTargetState, current);
            beforePhysicsTargetState = current;
            hasBeforePhysicsTargetState = true;
        }

        public void RecordAfterPhysicsStep(GateContext context)
        {
            RequireActiveContext(context);
            CompleteTargetAuthorityAudit(context);
            ValidateAndAccumulateClock(context);
            ControllerTelemetrySnapshot telemetry = SampleControllerTelemetry(context.PhysicsStep);
            ContactTruthRow[] contacts = ConsumeContacts(context);
            ControllerStateTruth controllerState = BuildControllerState(telemetry, context.PhysicsStep);

            ObjectTruth freeObject = SampleFreeObject(context, contacts);
            ForceBearingQualificationTruth forceQualification = SampleForceBearingQualification(
                context,
                telemetry,
                contacts,
                freeObject
            );
            EpisodeTraceRow row = new EpisodeTraceRow
            {
                schema = FrozenGate.TraceSchema,
                episode = new EpisodeContextTruth
                {
                    episode_id = context.EpisodeId,
                    cell_id = context.CellId,
                    room_family = context.RoomFamily,
                    garment_configuration_id = context.GarmentConfigurationId,
                    target_id = context.TargetId,
                    destination_id = context.DestinationId,
                    contact_strategy = context.ContactStrategy,
                    final_gaze_zone = context.FinalGazeZone
                },
                clock = new ClockTruth
                {
                    physics_step = context.PhysicsStep,
                    render_frame = context.RenderFrame,
                    time_s = context.TimeSeconds,
                    dt_s = Dt,
                    physics_hz = FrozenGate.PhysicsHz,
                    render_hz = FrozenGate.RenderHz,
                    steps_per_render_frame = FrozenGate.StepsPerFrame,
                    sample_phase = "post_physics_simulate",
                    phase_id = telemetry.phase_id ?? "unavailable"
                },
                body_state = new BodyStateTruth
                {
                    root = SampleTransformPose(bodySegments["root"], "body/root"),
                    pelvis = SampleTransformPose(bodySegments["pelvis"], "body/pelvis"),
                    torso = SampleTransformPose(bodySegments["torso"], "body/torso"),
                    neck = SampleTransformPose(bodySegments["neck"], "body/neck"),
                    head = SampleTransformPose(bodySegments["head"], "body/head"),
                    left_shoulder = SampleTransformPose(bodySegments["left_shoulder"], "body/left/shoulder"),
                    left_upper_arm = SampleTransformPose(bodySegments["left_upper_arm"], "body/left/upper_arm"),
                    left_elbow = SampleTransformPose(bodySegments["left_elbow"], "body/left/elbow"),
                    left_lower_arm = SampleTransformPose(bodySegments["left_lower_arm"], "body/left/lower_arm"),
                    left_forearm = SampleTransformPose(bodySegments["left_forearm"], "body/left/forearm"),
                    left_wrist = SampleTransformPose(bodySegments["left_wrist"], "body/left/wrist"),
                    left_palm = SampleTransformPose(bodySegments["left_palm"], "body/left/palm"),
                    right_shoulder = SampleTransformPose(bodySegments["right_shoulder"], "body/right/shoulder"),
                    right_upper_arm = SampleTransformPose(bodySegments["right_upper_arm"], "body/right/upper_arm"),
                    right_elbow = SampleTransformPose(bodySegments["right_elbow"], "body/right/elbow"),
                    right_lower_arm = SampleTransformPose(bodySegments["right_lower_arm"], "body/right/lower_arm"),
                    right_forearm = SampleTransformPose(bodySegments["right_forearm"], "body/right/forearm"),
                    right_wrist = SampleTransformPose(bodySegments["right_wrist"], "body/right/wrist"),
                    right_palm = SampleTransformPose(bodySegments["right_palm"], "body/right/palm")
                },
                controller_state = controllerState,
                contacts = contacts,
                objects = new[] { freeObject },
                camera_state = SampleCamera(context),
                assistance_ledger = context.AssistanceLedger.ToArray(),
                recovery_ledger = context.RecoveryLedger.ToArray(),
                authority_counters = CopyAuthorityAudit(context.AuthorityAudit),
                force_bearing_qualification = forceQualification
            };
            row.hand_state = SampleHands(context, telemetry);
            row.derived_state = SampleDerivedState(context);

            traceWriter.WriteLine(JsonUtility.ToJson(row, false));
            rowCount++;
            maximumContactsInStep = Math.Max(maximumContactsInStep, contacts.Length);
            totalContactRows += contacts.Length;
            maximumAssistanceEntries = Math.Max(maximumAssistanceEntries, context.AssistanceLedger.Count);
        }

        public void Complete(GateContext context)
        {
            RequireActiveContext(context);
            Physics.ContactEvent -= ObserveContactEvent;
            traceWriter.Flush();
            traceWriter.Dispose();
            traceWriter = null;

            int expectedRowCount = Mathf.RoundToInt(FrozenGate.DurationSeconds * FrozenGate.PhysicsHz);
            bool clockValid = rowCount == expectedRowCount && firstPhysicsStep == 0
                              && lastPhysicsStep == expectedRowCount - 1
                              && contiguousPhysicsSteps && contiguousTimes && exactRenderMapping
                              && maximumClockErrorSeconds <= ClockToleranceSeconds;
            ClockReceipt receipt = new ClockReceipt
            {
                schema = "embodied.episode_trace.clock_receipt.v1",
                episode_id = context.EpisodeId,
                physics_hz = FrozenGate.PhysicsHz,
                render_hz = FrozenGate.RenderHz,
                exact_steps_per_render_frame = FrozenGate.StepsPerFrame,
                expected_row_count = expectedRowCount,
                row_count = rowCount,
                first_physics_step = firstPhysicsStep,
                last_physics_step = lastPhysicsStep,
                first_render_frame = firstRenderFrame,
                last_render_frame = lastRenderFrame,
                first_time_s = firstTimeSeconds,
                last_time_s = lastTimeSeconds,
                contiguous_physics_steps = contiguousPhysicsSteps,
                contiguous_time_samples = contiguousTimes,
                exact_integer_render_mapping = exactRenderMapping,
                maximum_clock_error_s = maximumClockErrorSeconds,
                sample_phase = "post_physics_simulate",
                passed = clockValid
            };
            string clockPath = Path.Combine(context.OutputRoot, ClockReceiptFileName);
            WriteJson(clockPath, receipt);

            string traceSha256 = Sha256(tracePath);
            string clockSha256 = Sha256(clockPath);
            InteractionSummary interactionSummary = new InteractionSummary
            {
                schema = "embodied.physics_truth.interaction_summary.v1",
                episode_id = context.EpisodeId,
                target_id = context.TargetId,
                required_destination_id = context.DestinationId,
                observed_destination_support_id = lastObservedDestinationSupportId,
                finger_object_max_penetration_m = fingerObjectMaximumPenetrationM,
                target_support_max_penetration_m = targetSupportMaximumPenetrationM,
                finger_object_penetration_limit_m = FrozenGate.FingerObjectPenetrationMaxM,
                target_support_penetration_limit_m = FrozenGate.SupportPenetrationMaxM,
                finger_object_penetration_passed = fingerObjectMaximumPenetrationM <= FrozenGate.FingerObjectPenetrationMaxM,
                target_support_penetration_passed = targetSupportMaximumPenetrationM <= FrozenGate.SupportPenetrationMaxM,
                right_measured_impulse_qualified = measuredRightQualified,
                left_measured_impulse_qualified = measuredLeftQualified,
                lift_over_0_10_m = maximumQualifiedLiftM > 0.10f,
                turn_over_30_deg = maximumQualifiedTurnDeg > 30f,
                object_unsupported_during_qualified_manipulation = objectUnsupportedDuringQualifiedManipulation,
                release_command_ever_observed = releaseEverCommanded,
                release_at_required_destination = releaseAtRequiredDestination,
                provenance = "derived_from_post_Physics.Simulate_contact_rows"
            };
            string interactionSummaryPath = Path.Combine(context.OutputRoot, InteractionSummaryFileName);
            WriteJson(interactionSummaryPath, interactionSummary);
            TraceManifest manifest = new TraceManifest
            {
                schema = "embodied.episode_trace.manifest.v1",
                episode_id = context.EpisodeId,
                trace_schema = FrozenGate.TraceSchema,
                trace_file = TraceFileName,
                trace_sha256 = traceSha256,
                clock_receipt_file = ClockReceiptFileName,
                clock_receipt_sha256 = clockSha256,
                interaction_summary_file = InteractionSummaryFileName,
                interaction_summary_sha256 = Sha256(interactionSummaryPath),
                finger_object_max_penetration_m = fingerObjectMaximumPenetrationM,
                target_support_max_penetration_m = targetSupportMaximumPenetrationM,
                required_destination_id = context.DestinationId,
                observed_destination_support_id = lastObservedDestinationSupportId,
                object_unsupported_during_qualified_manipulation = objectUnsupportedDuringQualifiedManipulation,
                release_at_required_destination = releaseAtRequiredDestination,
                row_count = rowCount,
                controller_provider_bound = controllerProvider != null,
                unavailable_controller_steps = unavailableControllerSteps,
                controller_clock_mismatch_steps = controllerClockMismatchSteps,
                maximum_contacts_in_step = maximumContactsInStep,
                total_contact_rows = totalContactRows,
                maximum_assistance_entries = maximumAssistanceEntries,
                avatar_collider_count = avatarColliders.Length,
                unavailable_camera_clearance_steps = unavailableCameraClearanceSteps,
                non_free_object_steps = nonFreeObjectSteps,
                camera_parent_mismatch_steps = cameraParentMismatchSteps,
                unavailable_dynamic_finger_bindings = unavailableDynamicFingerBindings,
                assistance_ledger_empty = maximumAssistanceEntries == 0,
                recovery_ledger_entries = context.RecoveryLedger.Count,
                authority_counters = CopyAuthorityAudit(context.AuthorityAudit),
                source_audit_sha256 = context.AuthorityAudit.sourceAuditSha256,
                source_audit_present = IsSha256(context.AuthorityAudit.sourceAuditSha256),
                runtime_detection_coverage = "boundary-sampled target parenting, Joint graph, isKinematic, and between-step Rigidbody/Transform pose and velocity discontinuities",
                force_torque_detection_boundary = "counters require instrumented call sites; mandatory source audit covers uninstrumented calls",
                zero_counters_are_independent_proof = false,
                clock_passed = clockValid,
                one_authority_state = authorityBindingsVerified,
                object_state_authority = "Unity Rigidbody/PhysX sampled after Physics.Simulate",
                embodiment_state_authority = "Unity Transform state sampled after Physics.Simulate",
                biological_torque = "unavailable_not_claimed",
                trace_complete = clockValid && controllerProvider != null
                                 && unavailableControllerSteps == 0
                                 && controllerClockMismatchSteps == 0
                                 && avatarColliders.Length > 0
                                 && unavailableCameraClearanceSteps == 0
                                 && nonFreeObjectSteps == 0
                                 && cameraParentMismatchSteps == 0
                                 && authorityBindingsVerified
                                 && unavailableDynamicFingerBindings == 0
                                 && maximumAssistanceEntries == 0
                                 && AuthorityAuditPassed(context.AuthorityAudit)
                                 && IsSha256(context.AuthorityAudit.sourceAuditSha256),
                provenance_registry = ProvenanceRegistry()
            };
            string manifestPath = Path.Combine(context.OutputRoot, ManifestFileName);
            WriteJson(manifestPath, manifest);

            HashReceipt hashReceipt = new HashReceipt
            {
                schema = "embodied.episode_trace.hash_receipt.v1",
                episode_id = context.EpisodeId,
                algorithm = "sha256",
                trace_file = TraceFileName,
                trace_sha256 = traceSha256,
                manifest_file = ManifestFileName,
                manifest_sha256 = Sha256(manifestPath),
                clock_receipt_file = ClockReceiptFileName,
                clock_receipt_sha256 = clockSha256,
                interaction_summary_file = InteractionSummaryFileName,
                interaction_summary_sha256 = Sha256(interactionSummaryPath)
            };
            WriteJson(Path.Combine(context.OutputRoot, HashReceiptFileName), hashReceipt);
            completed = true;
        }

        internal void ObserveCollision(Collision collision)
        {
            if (boundContext == null || completed || collision == null)
                return;
            int count = collision.contactCount;
            if (count <= 0)
                return;

            ContactPoint[] points = collision.contacts;
            for (int index = 0; index < points.Length; index++)
            {
                ContactPoint point = points[index];
                Collider first = point.thisCollider;
                Collider second = point.otherCollider;
                if (first == null || second == null)
                    continue;
                string firstId = StableTransformId(first.transform);
                string secondId = StableTransformId(second.transform);
                if (!targetColliderIds.Contains(firstId) && !targetColliderIds.Contains(secondId))
                    continue;
                Collider colliderA = first;
                Collider colliderB = second;
                string colliderAId = firstId;
                string colliderBId = secondId;
                if (StringComparer.Ordinal.Compare(colliderAId, colliderBId) > 0)
                {
                    colliderA = second;
                    colliderB = first;
                    colliderAId = secondId;
                    colliderBId = firstId;
                }

                Vector3 normal = point.normal;
                Vector3 centerDirection = colliderB.bounds.center - colliderA.bounds.center;
                if (Vector3.Dot(normal, centerDirection) < 0f)
                    normal = -normal;
                Vector3 canonicalImpulse = collision.impulse;
                if (Vector3.Dot(canonicalImpulse, normal) < 0f)
                    canonicalImpulse = -canonicalImpulse;
                string key = ContactKey(colliderAId, colliderBId, point.point);
                pendingContacts[key] = new PendingContact
                {
                    colliderA = colliderA,
                    colliderB = colliderB,
                    colliderAId = colliderAId,
                    colliderBId = colliderBId,
                    pointWorldM = point.point,
                    normalWorld = normal.normalized,
                    separationM = point.separation,
                    relativeVelocityWorldMps = PointVelocity(colliderA, point.point) - PointVelocity(colliderB, point.point),
                    collisionPairImpulseNs = canonicalImpulse,
                    impulseScope = "Collision.impulse pair aggregate repeated per contact; do not sum per-point rows",
                    source = "Unity collision callback ContactPoint and Collision state from the completed PhysX step"
                };
            }
        }

        private void ObserveContactEvent(PhysicsScene scene, NativeArray<ContactPairHeader>.ReadOnly headers)
        {
            if (boundContext == null || completed || scene != Physics.defaultPhysicsScene)
                return;
            for (int headerIndex = 0; headerIndex < headers.Length; headerIndex++)
            {
                ContactPairHeader header = headers[headerIndex];
                for (int pairIndex = 0; pairIndex < header.pairCount; pairIndex++)
                {
                    ContactPair pair = header.GetContactPair(pairIndex);
                    Collider first = pair.collider;
                    Collider second = pair.otherCollider;
                    if (first == null || second == null || pair.contactCount <= 0)
                        continue;
                    string firstId = StableTransformId(first.transform);
                    string secondId = StableTransformId(second.transform);
                    if (!targetColliderIds.Contains(firstId) && !targetColliderIds.Contains(secondId))
                        continue;
                    Collider colliderA = first;
                    Collider colliderB = second;
                    string colliderAId = firstId;
                    string colliderBId = secondId;
                    if (StringComparer.Ordinal.Compare(colliderAId, colliderBId) > 0)
                    {
                        colliderA = second;
                        colliderB = first;
                        colliderAId = secondId;
                        colliderBId = firstId;
                    }
                    for (int contactIndex = 0; contactIndex < pair.contactCount; contactIndex++)
                    {
                        ContactPairPoint point = pair.GetContactPoint(contactIndex);
                        Vector3 normal = point.normal;
                        Vector3 centerDirection = colliderB.bounds.center - colliderA.bounds.center;
                        if (Vector3.Dot(normal, centerDirection) < 0f)
                            normal = -normal;
                        Vector3 impulse = point.impulse;
                        if (Vector3.Dot(impulse, normal) < 0f)
                            impulse = -impulse;
                        string key = ContactKey(colliderAId, colliderBId, point.position);
                        pendingContacts[key] = new PendingContact
                        {
                            colliderA = colliderA,
                            colliderB = colliderB,
                            colliderAId = colliderAId,
                            colliderBId = colliderBId,
                            pointWorldM = point.position,
                            normalWorld = normal.normalized,
                            separationM = point.separation,
                            relativeVelocityWorldMps = PointVelocity(colliderA, point.position) - PointVelocity(colliderB, point.position),
                            collisionPairImpulseNs = impulse,
                            impulseScope = "ContactPairPoint.impulse for this PhysX contact point",
                            source = "Unity Physics.ContactEvent ContactPairPoint from the completed manual simulation step"
                        };
                    }
                }
            }
        }

        private static void ValidateBindings(GateContext context)
        {
            if (context == null)
                throw new ArgumentNullException(nameof(context));
            if (string.IsNullOrWhiteSpace(context.EpisodeId))
                throw new InvalidOperationException("GateContext.EpisodeId is required");
            if (string.IsNullOrWhiteSpace(context.OutputRoot))
                throw new InvalidOperationException("GateContext.OutputRoot is required");
            if (context.AuthorityRoot == null || context.AvatarRoot == null || context.Torso == null
                || context.Neck == null || context.Head == null || context.LeftPalm == null
                || context.RightPalm == null || context.TargetBody == null
                || context.HeadCameraMount == null || context.HeadCamera == null)
                throw new InvalidOperationException("truth recorder requires authority/avatar/body/palms/object/head-camera bindings");
            foreach (string side in new[] { "left", "right" })
            foreach (string digit in FrozenGate.Digits)
            {
                string key = side + "_" + digit;
                if (!context.FingerSegments.TryGetValue(key, out Transform[] segments)
                    || segments == null || segments.Length != 3 || segments.Any(segment => segment == null))
                    throw new InvalidOperationException(key + " must bind exactly three non-null anatomical segments");
            }
        }

        private void RequireActiveContext(GateContext context)
        {
            if (boundContext == null || !ReferenceEquals(boundContext, context))
                throw new InvalidOperationException("PhysicsTruthRecorder called with a context other than its bound authority context");
            if (completed)
                throw new InvalidOperationException("PhysicsTruthRecorder is already complete");
        }

        private void CompleteTargetAuthorityAudit(GateContext context)
        {
            TargetAuthoritySnapshot completedState = SampleTargetAuthorityState(context.TargetBody);
            if (!hasBeforePhysicsTargetState)
            {
                RegisterAuthorityViolation(
                    context,
                    "audit_boundary_missing",
                    1,
                    "RecordBeforePhysicsStep was not called for the completed manual physics step"
                );
            }
            else
            {
                AuditStructuralChanges(context, beforePhysicsTargetState, completedState);
            }
            lastCompletedTargetState = completedState;
            hasLastCompletedTargetState = true;
            hasBeforePhysicsTargetState = false;
        }

        private void AuditOutOfPhysicsChanges(
            GateContext context,
            TargetAuthoritySnapshot completed,
            TargetAuthoritySnapshot beforePhysics
        )
        {
            AuditStructuralChanges(context, completed, beforePhysics);
            if (Vector3.Distance(completed.rigidbodyPosition, beforePhysics.rigidbodyPosition)
                    > OutOfPhysicsPositionToleranceM
                || Quaternion.Angle(completed.rigidbodyRotation, beforePhysics.rigidbodyRotation)
                    > OutOfPhysicsRotationToleranceDeg
                || Vector3.Distance(completed.transformPosition, beforePhysics.transformPosition)
                    > OutOfPhysicsPositionToleranceM
                || Quaternion.Angle(completed.transformRotation, beforePhysics.transformRotation)
                    > OutOfPhysicsRotationToleranceDeg)
            {
                RegisterAuthorityViolation(
                    context,
                    "target_pose_write",
                    1,
                    "target Rigidbody/Transform pose changed between completed PhysX state and next pre-sim boundary"
                );
            }
            if (Vector3.Distance(completed.linearVelocity, beforePhysics.linearVelocity)
                    > OutOfPhysicsLinearVelocityToleranceMps
                || Vector3.Distance(completed.angularVelocity, beforePhysics.angularVelocity)
                    > OutOfPhysicsAngularVelocityToleranceRadps)
            {
                RegisterAuthorityViolation(
                    context,
                    "target_velocity_write",
                    1,
                    "target Rigidbody velocity changed between completed PhysX state and next pre-sim boundary"
                );
            }
        }

        private void AuditStructuralChanges(
            GateContext context,
            TargetAuthoritySnapshot previous,
            TargetAuthoritySnapshot current
        )
        {
            if (!string.Equals(previous.parentId, current.parentId, StringComparison.Ordinal))
                RegisterAuthorityViolation(context, "target_parenting", 1, "target parent changed at runtime");
            if (previous.isKinematic != current.isKinematic)
                RegisterAuthorityViolation(context, "target_kinematic", 1, "target Rigidbody.isKinematic changed at runtime");
            int changedJoints = previous.jointIds.Except(current.jointIds, StringComparer.Ordinal).Count()
                                + current.jointIds.Except(previous.jointIds, StringComparer.Ordinal).Count();
            if (changedJoints > 0)
                RegisterAuthorityViolation(context, "target_joint", changedJoints, "target Joint graph changed at runtime");
        }

        private static TargetAuthoritySnapshot SampleTargetAuthorityState(Rigidbody body)
        {
            string[] jointIds = UnityEngine.Object.FindObjectsByType<Joint>(FindObjectsSortMode.None)
                .Where(joint => joint != null
                                && (joint.GetComponent<Rigidbody>() == body || joint.connectedBody == body))
                .Select(joint => StableTransformId(joint.transform) + "::" + joint.GetType().Name)
                .OrderBy(value => value, StringComparer.Ordinal)
                .ToArray();
            return new TargetAuthoritySnapshot
            {
                rigidbodyPosition = body.position,
                rigidbodyRotation = body.rotation,
                transformPosition = body.transform.position,
                transformRotation = body.transform.rotation,
                linearVelocity = body.linearVelocity,
                angularVelocity = body.angularVelocity,
                isKinematic = body.isKinematic,
                parentId = body.transform.parent == null ? "" : StableTransformId(body.transform.parent),
                jointIds = jointIds
            };
        }

        private static void RegisterAuthorityViolation(
            GateContext context,
            string category,
            int count,
            string detail
        )
        {
            if (count <= 0) return;
            AuthorityAuditState audit = context.AuthorityAudit;
            switch (category)
            {
                case "target_pose_write": audit.targetPoseWriteCounter += count; break;
                case "target_velocity_write": audit.targetVelocityWriteCounter += count; break;
                case "target_joint": audit.targetJointCounter += count; break;
                case "target_parenting": audit.targetParentingCounter += count; break;
                case "target_kinematic": audit.targetKinematicChangeCounter += count; break;
            }
            string ledgerEntry = "step=" + context.PhysicsStep + " category=" + category
                                 + " target=" + context.TargetId + " detail=" + detail;
            if (!context.AssistanceLedger.Contains(ledgerEntry))
                context.AssistanceLedger.Add(ledgerEntry);
        }

        private static AuthorityAuditState CopyAuthorityAudit(AuthorityAuditState source)
        {
            return new AuthorityAuditState
            {
                targetPoseWriteCounter = source.targetPoseWriteCounter,
                targetVelocityWriteCounter = source.targetVelocityWriteCounter,
                targetForceCounter = source.targetForceCounter,
                targetTorqueCounter = source.targetTorqueCounter,
                targetJointCounter = source.targetJointCounter,
                targetParentingCounter = source.targetParentingCounter,
                targetKinematicChangeCounter = source.targetKinematicChangeCounter,
                recoveryCounter = source.recoveryCounter,
                sourceAuditSha256 = source.sourceAuditSha256
            };
        }

        private static bool AuthorityAuditPassed(AuthorityAuditState audit)
        {
            return audit.targetPoseWriteCounter == 0
                   && audit.targetVelocityWriteCounter == 0
                   && audit.targetForceCounter == 0
                   && audit.targetTorqueCounter == 0
                   && audit.targetJointCounter == 0
                   && audit.targetParentingCounter == 0
                   && audit.targetKinematicChangeCounter == 0;
        }

        private static bool IsSha256(string value)
        {
            return !string.IsNullOrWhiteSpace(value)
                   && value.Length == 64
                   && value.All(Uri.IsHexDigit);
        }

        private void ValidateAndAccumulateClock(GateContext context)
        {
            int expectedRenderFrame = context.PhysicsStep / FrozenGate.StepsPerFrame;
            float expectedTimeSeconds = (context.PhysicsStep + 1) * Dt;
            maximumClockErrorSeconds = Mathf.Max(
                maximumClockErrorSeconds,
                Mathf.Abs(context.TimeSeconds - expectedTimeSeconds)
            );
            exactRenderMapping &= context.RenderFrame == expectedRenderFrame;
            if (rowCount == 0)
            {
                firstPhysicsStep = context.PhysicsStep;
                firstRenderFrame = context.RenderFrame;
                firstTimeSeconds = context.TimeSeconds;
            }
            else
            {
                contiguousPhysicsSteps &= context.PhysicsStep == lastPhysicsStep + 1;
                float deltaError = Mathf.Abs((context.TimeSeconds - lastTimeSeconds) - Dt);
                maximumClockErrorSeconds = Mathf.Max(maximumClockErrorSeconds, deltaError);
                contiguousTimes &= deltaError <= ClockToleranceSeconds;
            }
            lastPhysicsStep = context.PhysicsStep;
            lastRenderFrame = context.RenderFrame;
            lastTimeSeconds = context.TimeSeconds;
        }

        private ControllerTelemetrySnapshot SampleControllerTelemetry(int physicsStep)
        {
            if (controllerProvider == null)
            {
                unavailableControllerSteps++;
                return UnavailableControllerTelemetry(physicsStep);
            }
            ControllerTelemetrySnapshot snapshot = controllerProvider.SampleAfterPhysicsStep(physicsStep);
            if (snapshot == null)
            {
                unavailableControllerSteps++;
                return UnavailableControllerTelemetry(physicsStep);
            }
            snapshot.targets = snapshot.targets ?? Array.Empty<ControllerTargetTruth>();
            snapshot.errors = snapshot.errors ?? Array.Empty<ControllerErrorTruth>();
            snapshot.digit_closure_commands = snapshot.digit_closure_commands ?? Array.Empty<DigitClosureCommandTruth>();
            snapshot.speed_limits = snapshot.speed_limits ?? new SpeedLimitsTruth();
            snapshot.closure_limits = snapshot.closure_limits ?? new ClosureLimitsTruth();
            if (snapshot.physics_step != physicsStep)
                controllerClockMismatchSteps++;
            if (snapshot.targets.Length == 0 || snapshot.digit_closure_commands.Length != 10
                || snapshot.speed_limits.provenance == "unavailable"
                || snapshot.closure_limits.provenance == "unavailable")
                unavailableControllerSteps++;
            return snapshot;
        }

        private static ControllerTelemetrySnapshot UnavailableControllerTelemetry(int physicsStep)
        {
            return new ControllerTelemetrySnapshot
            {
                physics_step = physicsStep,
                phase_id = "unavailable",
                recovery_state = "unavailable",
                recovery_state_provenance = "unavailable"
            };
        }

        private static ControllerStateTruth BuildControllerState(ControllerTelemetrySnapshot telemetry, int physicsStep)
        {
            bool releaseCommanded = string.Equals(telemetry.phase_id, "CommandedOpen", StringComparison.OrdinalIgnoreCase)
                                    || string.Equals(telemetry.phase_id, "Release", StringComparison.OrdinalIgnoreCase);
            return new ControllerStateTruth
            {
                telemetry_physics_step = telemetry.physics_step,
                expected_physics_step = physicsStep,
                phase_id = telemetry.phase_id ?? "unavailable",
                targets = telemetry.targets,
                errors = telemetry.errors,
                recovery_state = telemetry.recovery_state ?? "unavailable",
                recovery_state_provenance = telemetry.recovery_state_provenance ?? "unavailable",
                right_opposition_qualified = telemetry.right_opposition_qualified,
                meaningful_left_support_qualified = telemetry.meaningful_left_support_qualified,
                carry_contacts_maintained = telemetry.carry_contacts_maintained,
                right_opposition_dwell_steps = telemetry.right_opposition_dwell_steps,
                left_support_dwell_steps = telemetry.left_support_dwell_steps,
                bimanual_qualification_step = telemetry.bimanual_qualification_step,
                speed_limits = telemetry.speed_limits,
                closure_limits = telemetry.closure_limits,
                release_commanded = releaseCommanded,
                events = releaseCommanded ? new[] { "release_commanded" } : Array.Empty<string>(),
                provenance = telemetry.phase_id == "unavailable" ? "unavailable" : "commanded_and_engine_observed"
            };
        }

        private BodyHandsTruth SampleHands(GateContext context, ControllerTelemetrySnapshot telemetry)
        {
            return new BodyHandsTruth
            {
                left_palm = SampleTransformPose(bodySegments["left_palm"], "hands/left/palm"),
                right_palm = SampleTransformPose(bodySegments["right_palm"], "hands/right/palm"),
                left_thumb = SampleDigit(context, telemetry, "left", "thumb"),
                left_index = SampleDigit(context, telemetry, "left", "index"),
                left_middle = SampleDigit(context, telemetry, "left", "middle"),
                left_ring = SampleDigit(context, telemetry, "left", "ring"),
                left_little = SampleDigit(context, telemetry, "left", "little"),
                right_thumb = SampleDigit(context, telemetry, "right", "thumb"),
                right_index = SampleDigit(context, telemetry, "right", "index"),
                right_middle = SampleDigit(context, telemetry, "right", "middle"),
                right_ring = SampleDigit(context, telemetry, "right", "ring"),
                right_little = SampleDigit(context, telemetry, "right", "little")
            };
        }

        private DigitTruth SampleDigit(
            GateContext context,
            ControllerTelemetrySnapshot telemetry,
            string hand,
            string digit
        )
        {
            string key = hand + "_" + digit;
            Transform[] segments = context.FingerSegments[key];
            DigitClosureCommandTruth command = telemetry.digit_closure_commands.FirstOrDefault(
                item => item != null && item.hand == hand && item.digit == digit
            );
            float observedDegrees = 0f;
            for (int index = 0; index < segments.Length; index++)
                observedDegrees += Quaternion.Angle(bindDigitLocalRotations[key][index], segments[index].localRotation);
            observedDegrees /= segments.Length;
            return new DigitTruth
            {
                hand = hand,
                digit = digit,
                segments = segments.Select((segment, index) => new FingerSegmentTruth
                {
                    segment_index = index,
                    segment_id = StableTransformId(segment),
                    pose = SampleTransformPose(segment, "hands/" + key + "/segment_" + index)
                }).ToArray(),
                closure_commanded = command == null ? 0f : command.closure_commanded,
                closure_commanded_provenance = command == null ? "unavailable" : "commanded",
                closure_observed = Mathf.Clamp01(observedDegrees / ObservedClosureNormalizationDeg),
                closure_observed_provenance = "derived",
                closure_observed_formula = "mean Quaternion.Angle(bind_local_rotation,current_local_rotation) / 90 deg",
                dynamic_body_states = Enumerable.Range(1, 3)
                    .Select(index => SampleFingerBody(context, key + "_segment" + index))
                    .ToArray(),
                compliant_joint_states = Enumerable.Range(1, 3)
                    .Select(index => SampleFingerJoint(context, key + "_segment" + index))
                    .ToArray()
            };
        }

        private static DynamicFingerBodyTruth SampleFingerBody(GateContext context, string key)
        {
            if (!context.FingerBodies.TryGetValue(key, out Rigidbody body) || body == null)
                return new DynamicFingerBodyTruth
                {
                    body_id = "unavailable",
                    provenance = "unavailable",
                    physical_authority = "unavailable"
                };
            return new DynamicFingerBodyTruth
            {
                body_id = StableTransformId(body.transform),
                pose = new PoseTruth
                {
                    position_world_m = body.position,
                    rotation_world_xyzw = body.rotation,
                    linear_velocity_world_m_s = body.linearVelocity,
                    angular_velocity_world_rad_s = body.angularVelocity,
                    position_provenance = "engine_observed",
                    rotation_provenance = "engine_observed",
                    linear_velocity_provenance = "physx_measured",
                    angular_velocity_provenance = "physx_measured",
                    velocity_formula_or_source = "Unity Rigidbody state sampled after Physics.Simulate"
                },
                is_kinematic = body.isKinematic,
                mass_kg = body.mass,
                linear_velocity_world_m_s = body.linearVelocity,
                angular_velocity_world_rad_s = body.angularVelocity,
                sleeping = body.IsSleeping(),
                collision_detection_mode = body.collisionDetectionMode.ToString(),
                provenance = "physx_measured",
                physical_authority = body.isKinematic
                    ? "Unity Rigidbody kinematic (not compliant dynamic authority)"
                    : "Unity dynamic Rigidbody/PhysX"
            };
        }

        private static CompliantFingerJointTruth SampleFingerJoint(GateContext context, string key)
        {
            if (!context.FingerJoints.TryGetValue(key, out ConfigurableJoint joint) || joint == null)
                return new CompliantFingerJointTruth
                {
                    joint_id = "unavailable",
                    provenance = "unavailable",
                    drive_provenance = "unavailable"
                };
            return new CompliantFingerJointTruth
            {
                joint_id = StableTransformId(joint.transform),
                connected_body_id = joint.connectedBody == null
                    ? "world"
                    : StableTransformId(joint.connectedBody.transform),
                target_rotation_xyzw = joint.targetRotation,
                target_angular_velocity_rad_s = joint.targetAngularVelocity,
                angular_x_drive_spring_n_m_rad = joint.angularXDrive.positionSpring,
                angular_x_drive_damper_n_m_s_rad = joint.angularXDrive.positionDamper,
                angular_x_drive_max_force_n_m = joint.angularXDrive.maximumForce,
                angular_yz_drive_spring_n_m_rad = joint.angularYZDrive.positionSpring,
                angular_yz_drive_damper_n_m_s_rad = joint.angularYZDrive.positionDamper,
                angular_yz_drive_max_force_n_m = joint.angularYZDrive.maximumForce,
                provenance = "engine_observed",
                drive_provenance = "commanded ConfigurableJoint drive consumed by PhysX"
            };
        }

        private PoseTruth SampleTransformPose(Transform transform, string stateKey)
        {
            Vector3 position = transform.position;
            Quaternion rotation = transform.rotation;
            PoseTruth sample = new PoseTruth
            {
                position_world_m = position,
                rotation_world_xyzw = rotation,
                position_provenance = "engine_observed",
                rotation_provenance = "engine_observed",
                velocity_formula_or_source = "first finite difference of consecutive post-Physics.Simulate poses at 240 Hz"
            };
            if (priorPoses.TryGetValue(stateKey, out PriorPose prior))
            {
                sample.linear_velocity_world_m_s = (position - prior.position) / Dt;
                sample.angular_velocity_world_rad_s = QuaternionAngularVelocityWorld(prior.rotation, rotation, Dt);
                sample.linear_velocity_provenance = "derived";
                sample.angular_velocity_provenance = "derived";
            }
            else
            {
                sample.linear_velocity_world_m_s = Vector3.zero;
                sample.angular_velocity_world_rad_s = Vector3.zero;
                sample.linear_velocity_provenance = "unavailable";
                sample.angular_velocity_provenance = "unavailable";
            }
            priorPoses[stateKey] = new PriorPose { position = position, rotation = rotation };
            return sample;
        }

        private ContactTruthRow[] ConsumeContacts(GateContext context)
        {
            ContactTruth[] sharedContacts = context.Contacts
                .Where(value => value.physicsStep == context.PhysicsStep)
                .ToArray();
            PendingContact[] measuredContacts = pendingContacts.Values.ToArray();
            foreach (PendingContact contact in measuredContacts)
            {
                context.Contacts.Add(new ContactTruth
                {
                    physicsStep = context.PhysicsStep,
                    colliderA = contact.colliderAId,
                    colliderB = contact.colliderBId,
                    pointWorldM = contact.pointWorldM,
                    normalWorld = contact.normalWorld,
                    separationM = contact.separationM,
                    relativeVelocityWorldMps = contact.relativeVelocityWorldMps,
                    availableImpulseNs = contact.collisionPairImpulseNs,
                    provenance = TruthSource.PhysXMeasured
                });
            }
            List<ContactTruthRow> rows = measuredContacts.Select(contact => new ContactTruthRow
                {
                    physics_step = context.PhysicsStep,
                    collider_a = contact.colliderAId,
                    collider_b = contact.colliderBId,
                    point_world_m = contact.pointWorldM,
                    normal_world = contact.normalWorld,
                    separation_m = contact.separationM,
                    relative_velocity_world_m_s = contact.relativeVelocityWorldMps,
                    available_impulse_n_s = contact.collisionPairImpulseNs,
                    available_impulse_magnitude_n_s = contact.collisionPairImpulseNs.magnitude,
                    nonzero_impulse_observed = contact.collisionPairImpulseNs.sqrMagnitude > 0f,
                    mean_force_equivalent_n = contact.collisionPairImpulseNs.magnitude / Dt,
                    force_evidence_semantics = contact.collisionPairImpulseNs.sqrMagnitude > 0f
                        ? "nonzero PhysX point impulse observed; mean-force equivalent is derived as impulse/dt"
                        : "zero PhysX point impulse observed; separation eligibility is reported independently",
                    qualification_separation_eligible = contact.separationM <= FullBodyBimanualMotion.MaximumQualifiedContactSeparationM,
                    available_impulse_scope = contact.impulseScope,
                    persistent_object_id = ContactPersistentObjectId(context, contact.colliderAId, contact.colliderBId),
                    hand = ContactHand(contact.colliderAId, contact.colliderBId),
                    digit = ContactDigit(contact.colliderAId, contact.colliderBId),
                    normal_on_object_world = ContactNormalOnObject(contact.colliderAId, contact.colliderBId, contact.normalWorld),
                    provenance = "physx_measured",
                    formula_or_source = contact.source
                })
                .ToList();
            foreach (ContactTruth contact in sharedContacts)
            {
                rows.Add(new ContactTruthRow
                {
                    physics_step = context.PhysicsStep,
                    collider_a = contact.colliderA,
                    collider_b = contact.colliderB,
                    point_world_m = contact.pointWorldM,
                    normal_world = contact.normalWorld,
                    separation_m = contact.separationM,
                    relative_velocity_world_m_s = contact.relativeVelocityWorldMps,
                    available_impulse_n_s = contact.availableImpulseNs,
                    available_impulse_magnitude_n_s = contact.availableImpulseNs.magnitude,
                    nonzero_impulse_observed = contact.availableImpulseNs.sqrMagnitude > 0f,
                    mean_force_equivalent_n = contact.availableImpulseNs.magnitude / Dt,
                    force_evidence_semantics = contact.availableImpulseNs.sqrMagnitude > 0f
                        ? "nonzero shared PhysX impulse observed; mean-force equivalent is derived as impulse/dt"
                        : "zero shared PhysX impulse observed; separation eligibility is reported independently",
                    qualification_separation_eligible = contact.separationM <= FullBodyBimanualMotion.MaximumQualifiedContactSeparationM,
                    available_impulse_scope = "shared GateContext ContactTruth",
                    persistent_object_id = ContactPersistentObjectId(context, contact.colliderA, contact.colliderB),
                    hand = ContactHand(contact.colliderA, contact.colliderB),
                    digit = ContactDigit(contact.colliderA, contact.colliderB),
                    normal_on_object_world = ContactNormalOnObject(contact.colliderA, contact.colliderB, contact.normalWorld),
                    provenance = TruthLabel(contact.provenance),
                    formula_or_source = "shared frozen GateContext contact instrumentation"
                });
            }
            pendingContacts.Clear();
            return rows
                .OrderBy(row => row.collider_a, StringComparer.Ordinal)
                .ThenBy(row => row.collider_b, StringComparer.Ordinal)
                .ThenBy(row => row.point_world_m.x)
                .ThenBy(row => row.point_world_m.y)
                .ThenBy(row => row.point_world_m.z)
                .ToArray();
        }

        private void IndexExternalColliderIdentities(GateContext context)
        {
            externalColliderPersistentIds.Clear();
            foreach (SceneIdentity identity in UnityEngine.Object.FindObjectsByType<SceneIdentity>(FindObjectsSortMode.None))
            {
                if (identity == null || string.IsNullOrWhiteSpace(identity.persistent_id))
                    continue;
                foreach (Collider collider in identity.GetComponentsInChildren<Collider>(true))
                    if (collider != null)
                        externalColliderPersistentIds[StableTransformId(collider.transform)] = identity.persistent_id;
            }
            foreach (KeyValuePair<string, Transform> destination in context.Destinations)
            {
                if (destination.Value == null
                    || !string.Equals(destination.Key, context.DestinationId, StringComparison.Ordinal))
                    continue;
                foreach (Collider collider in destination.Value.GetComponentsInChildren<Collider>(true))
                    if (collider != null)
                        externalColliderPersistentIds[StableTransformId(collider.transform)] = context.DestinationId;
            }
        }

        private void BindBodySegments(GateContext context)
        {
            bodySegments.Clear();
            foreach (string semanticId in new[]
                     {
                         "root", "pelvis", "torso", "neck", "head",
                         "left_shoulder", "left_upper_arm", "left_elbow", "left_lower_arm", "left_forearm", "left_wrist", "left_palm",
                         "right_shoulder", "right_upper_arm", "right_elbow", "right_lower_arm", "right_forearm", "right_wrist", "right_palm"
                     })
                BindBodySegment(context, semanticId);
            string[] stableIds = bodySegments.Values.Select(StableTransformId).ToArray();
            if (stableIds.Distinct(StringComparer.Ordinal).Count() != stableIds.Length)
                throw new InvalidOperationException("frozen body truth segments must map one-to-one to distinct Transforms");
        }

        private void BindBodySegment(GateContext context, string semanticId)
        {
            Transform segment = context.BodySegments
                .Where(pair => string.Equals(pair.Key, semanticId, StringComparison.OrdinalIgnoreCase))
                .Select(pair => pair.Value)
                .FirstOrDefault(value => value != null);
            if (segment == null)
                throw new InvalidOperationException("required body truth segment is unavailable: " + semanticId);
            bodySegments[semanticId] = segment;
        }

        private ObjectTruth SampleFreeObject(GateContext context, ContactTruthRow[] contacts)
        {
            Rigidbody body = context.TargetBody;
            bool active = body.gameObject.activeInHierarchy;
            if (active && body.isKinematic)
                nonFreeObjectSteps++;
            PhysicsTruthObjectIdentity identity = body.GetComponent<PhysicsTruthObjectIdentity>();
            string stableName = body.gameObject.name;
            string persistentId = identity != null && !string.IsNullOrWhiteSpace(identity.persistent_id)
                ? identity.persistent_id : stableName;
            string semanticId = identity != null && !string.IsNullOrWhiteSpace(identity.semantic_id)
                ? identity.semantic_id : "interactive_target";
            string instanceId = identity != null && !string.IsNullOrWhiteSpace(identity.instance_id)
                ? identity.instance_id : context.EpisodeId + "::" + stableName;
            ContactTruthRow[] targetContacts = contacts
                .Where(contact => targetColliderIds.Contains(contact.collider_a) || targetColliderIds.Contains(contact.collider_b))
                .ToArray();
            foreach (ContactTruthRow contact in targetContacts)
            {
                string pairedCollider = PairedTargetCollider(contact);
                float penetrationM = Mathf.Max(0f, -contact.separation_m);
                if (avatarColliderIds.Contains(pairedCollider))
                    fingerObjectMaximumPenetrationM = Mathf.Max(fingerObjectMaximumPenetrationM, penetrationM);
                else
                    targetSupportMaximumPenetrationM = Mathf.Max(targetSupportMaximumPenetrationM, penetrationM);
            }
            string supportColliderId = targetContacts
                .OrderBy(contact => contact.separation_m)
                .Select(PairedTargetCollider)
                .FirstOrDefault(candidate => !avatarColliderIds.Contains(candidate));
            string supportId = ResolveExternalSupportId(supportColliderId);
            return new ObjectTruth
            {
                active = active,
                availability_provenance = active ? "physx_measured" : "unavailable",
                persistent_id = persistentId,
                semantic_id = semanticId,
                instance_id = instanceId,
                identity_provenance = identity == null ? "derived_from_stable_scene_name" : "engine_observed",
                pose = new PoseTruth
                {
                    position_world_m = body.position,
                    rotation_world_xyzw = body.rotation,
                    linear_velocity_world_m_s = active ? body.linearVelocity : Vector3.zero,
                    angular_velocity_world_rad_s = active ? body.angularVelocity : Vector3.zero,
                    position_provenance = active ? "physx_measured" : "unavailable",
                    rotation_provenance = active ? "physx_measured" : "unavailable",
                    linear_velocity_provenance = active ? "physx_measured" : "unavailable",
                    angular_velocity_provenance = active ? "physx_measured" : "unavailable",
                    velocity_formula_or_source = active
                        ? "Unity Rigidbody state sampled after Physics.Simulate"
                        : "free object disabled for this object-free qualification stage"
                },
                linear_velocity_world_m_s = active ? body.linearVelocity : Vector3.zero,
                angular_velocity_world_rad_s = active ? body.angularVelocity : Vector3.zero,
                support_id = active ? supportId ?? "" : "",
                support_provenance = !active ? "unavailable" : supportId == null ? "physx_measured_no_active_support_contact" : "derived",
                support_formula_or_source = "persistent SceneIdentity for a non-avatar collider paired with the free object in current PhysX contacts",
                sleeping = active && body.IsSleeping(),
                sleeping_provenance = active ? "physx_measured" : "unavailable",
                is_kinematic = body.isKinematic,
                parent_id = body.transform.parent == null ? "" : StableTransformId(body.transform.parent),
                free_dynamic = active && !body.isKinematic,
                physical_authority = active ? "Unity Rigidbody/PhysX" : "unavailable_in_object_free_stage"
            };
        }

        private string PairedTargetCollider(ContactTruthRow contact)
        {
            if (contact == null)
                return null;
            if (targetColliderIds.Contains(contact.collider_a))
                return contact.collider_b;
            if (targetColliderIds.Contains(contact.collider_b))
                return contact.collider_a;
            return null;
        }

        private string ResolveExternalSupportId(string colliderId)
        {
            if (string.IsNullOrWhiteSpace(colliderId))
                return null;
            return externalColliderPersistentIds.TryGetValue(colliderId, out string persistentId)
                ? persistentId
                : colliderId;
        }

        private ForceBearingQualificationTruth SampleForceBearingQualification(
            GateContext context,
            ControllerTelemetrySnapshot telemetry,
            ContactTruthRow[] contacts,
            ObjectTruth freeObject
        )
        {
            if (context.PhysicsStep == 0 && contacts.Any(contact =>
                    !string.IsNullOrWhiteSpace(contact.hand)
                    && !string.IsNullOrWhiteSpace(contact.persistent_object_id)
                    && contact.separation_m < 0f))
                initialFingerOverlapDisqualified = true;
            ContactTruthRow[] eligible = contacts.Where(contact =>
                    contact.provenance == "physx_measured"
                    && contact.qualification_separation_eligible
                    && contact.separation_m >= -FrozenGate.FingerObjectPenetrationMaxM
                    && contact.available_impulse_magnitude_n_s > FullBodyBimanualMotion.MinimumQualifiedImpulseNs
                    && !string.IsNullOrWhiteSpace(contact.persistent_object_id)
                    && (contact.hand == "left" || contact.hand == "right")
                )
                .ToArray();
            ContactTruthRow[] right = eligible.Where(contact => contact.hand == "right").ToArray();
            ContactTruthRow[] thumb = right.Where(contact => contact.digit == "thumb").ToArray();
            ContactTruthRow[] nonThumb = right.Where(contact =>
                    contact.digit == "index" || contact.digit == "middle"
                    || contact.digit == "ring" || contact.digit == "little"
                )
                .ToArray();
            string[] rightNonThumbDigits = nonThumb.Select(contact => contact.digit)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(value => value, StringComparer.Ordinal)
                .ToArray();
            bool rightOpposingNow = !initialFingerOverlapDisqualified
                                    && thumb.Length > 0 && rightNonThumbDigits.Length >= 2
                                    && HasOpposingNormals(thumb, nonThumb);
            rightMeasuredImpulseDwellSteps = rightOpposingNow ? rightMeasuredImpulseDwellSteps + 1 : 0;
            if (rightMeasuredImpulseDwellSteps > FrozenGate.RightForceOppositionSeconds * FrozenGate.PhysicsHz)
                measuredRightQualified = true;

            ContactTruthRow[] left = eligible.Where(contact => contact.hand == "left" && contact.digit != "little").ToArray();
            string[] leftDigits = left.Select(contact => contact.digit)
                .Where(value => !string.IsNullOrWhiteSpace(value) && value != "palm")
                .Distinct(StringComparer.Ordinal)
                .OrderBy(value => value, StringComparer.Ordinal)
                .ToArray();
            bool leftSupportingNow = rightOpposingNow && leftDigits.Length > 0
                                     && HasOpposingNormals(left, right);
            leftMeasuredImpulseDwellSteps = leftSupportingNow ? leftMeasuredImpulseDwellSteps + 1 : 0;
            if (leftMeasuredImpulseDwellSteps > FrozenGate.LeftForceSupportSeconds * FrozenGate.PhysicsHz)
                measuredLeftQualified = true;

            bool fullyQualified = measuredRightQualified && measuredLeftQualified;
            qualificationEverActive |= fullyQualified;
            if (fullyQualified)
            {
                if (!hasQualificationBaseline)
                {
                    qualificationTargetCenterY = context.TargetBody.worldCenterOfMass.y;
                    qualificationTargetRotation = context.TargetBody.rotation;
                    hasQualificationBaseline = true;
                    measuredBimanualQualificationStep = context.PhysicsStep;
                }
                maximumQualifiedLiftM = Mathf.Max(
                    maximumQualifiedLiftM,
                    context.TargetBody.worldCenterOfMass.y - qualificationTargetCenterY
                );
                maximumQualifiedTurnDeg = Mathf.Max(
                    maximumQualifiedTurnDeg,
                    Quaternion.Angle(qualificationTargetRotation, context.TargetBody.rotation)
                );
            }
            bool openingCommanded = string.Equals(telemetry.phase_id, "CommandedOpen", StringComparison.OrdinalIgnoreCase)
                                    || string.Equals(telemetry.phase_id, "Release", StringComparison.OrdinalIgnoreCase);
            releaseEverCommanded |= openingCommanded;
            bool currentSupportIsRequiredDestination = !string.IsNullOrWhiteSpace(context.DestinationId)
                                                       && string.Equals(
                                                           freeObject.support_id,
                                                           context.DestinationId,
                                                           StringComparison.Ordinal
                                                       );
            bool objectHasAvatarContact = contacts
                .Where(contact => targetColliderIds.Contains(contact.collider_a)
                                  || targetColliderIds.Contains(contact.collider_b))
                .Select(PairedTargetCollider)
                .Any(colliderId => avatarColliderIds.Contains(colliderId));
            bool freeReleaseAtRequiredDestinationCurrent = releaseEverCommanded
                                                           && currentSupportIsRequiredDestination
                                                           && !objectHasAvatarContact
                                                           && freeObject.free_dynamic;
            if (currentSupportIsRequiredDestination)
                lastObservedDestinationSupportId = freeObject.support_id;
            if (freeReleaseAtRequiredDestinationCurrent)
                releaseAtRequiredDestination = true;
            if (fullyQualified && !openingCommanded && (!rightOpposingNow || !leftSupportingNow))
                supportContinuousUntilCommandedOpening = false;
            string normalizedPhase = (telemetry.phase_id ?? "").Replace("_", "").Replace("-", "").ToLowerInvariant();
            bool manipulationPhase = normalizedPhase.Contains("lift") || normalizedPhase.Contains("turn")
                                     || normalizedPhase.Contains("inspect") || normalizedPhase.Contains("transfer");
            bool qualifiedLiftWindow = fullyQualified && maximumQualifiedLiftM > 0.005f
                                       && manipulationPhase && !openingCommanded;
            if (qualifiedLiftWindow && !string.IsNullOrWhiteSpace(freeObject.support_id))
                objectUnsupportedDuringQualifiedManipulation = false;

            return new ForceBearingQualificationTruth
            {
                right_current_measured_impulse_opposition = rightOpposingNow,
                right_current_thumb_present = thumb.Length > 0,
                right_current_non_thumb_digits = rightNonThumbDigits,
                right_measured_impulse_dwell_steps = rightMeasuredImpulseDwellSteps,
                right_measured_impulse_dwell_s = rightMeasuredImpulseDwellSteps * Dt,
                right_measured_impulse_qualified = measuredRightQualified,
                right_required_strictly_greater_s = FrozenGate.RightForceOppositionSeconds,
                left_current_measured_impulse_support = leftSupportingNow,
                left_current_non_little_digits = leftDigits,
                left_measured_impulse_dwell_steps = leftMeasuredImpulseDwellSteps,
                left_measured_impulse_dwell_s = leftMeasuredImpulseDwellSteps * Dt,
                left_measured_impulse_qualified = measuredLeftQualified,
                left_required_strictly_greater_s = FrozenGate.LeftForceSupportSeconds,
                object_unsupported_current = string.IsNullOrWhiteSpace(freeObject.support_id),
                object_unsupported_during_qualified_manipulation = objectUnsupportedDuringQualifiedManipulation,
                required_destination_id = context.DestinationId,
                observed_support_id = freeObject.support_id,
                current_support_is_required_destination = currentSupportIsRequiredDestination,
                release_command_ever_observed = releaseEverCommanded,
                free_release_at_required_destination_current = freeReleaseAtRequiredDestinationCurrent,
                release_at_required_destination = releaseAtRequiredDestination,
                finger_object_max_penetration_m = fingerObjectMaximumPenetrationM,
                target_support_max_penetration_m = targetSupportMaximumPenetrationM,
                support_continuous_until_commanded_opening = supportContinuousUntilCommandedOpening,
                qualification_ever_active = qualificationEverActive,
                measured_bimanual_qualification_step = measuredBimanualQualificationStep,
                initial_overlap_disqualified = initialFingerOverlapDisqualified,
                maximum_qualified_lift_m = maximumQualifiedLiftM,
                lift_over_0_10_m = maximumQualifiedLiftM > 0.10f,
                maximum_qualified_turn_deg = maximumQualifiedTurnDeg,
                turn_over_30_deg = maximumQualifiedTurnDeg > 30f,
                impulse_semantics = "current-step PhysX ContactPairPoint impulse; geometry and impulse are independently required",
                maximum_eligible_separation_m = FullBodyBimanualMotion.MaximumQualifiedContactSeparationM,
                maximum_eligible_penetration_m = FrozenGate.FingerObjectPenetrationMaxM,
                minimum_nonzero_impulse_n_s = FullBodyBimanualMotion.MinimumQualifiedImpulseNs,
                maximum_opposing_normal_dot = -0.25f,
                provenance = "derived_from_physx_measured_contacts"
            };
        }

        private static bool HasOpposingNormals(ContactTruthRow[] first, ContactTruthRow[] second)
        {
            return first.Any(a => a.normal_on_object_world.sqrMagnitude > 0f
                                  && second.Any(b => b.normal_on_object_world.sqrMagnitude > 0f
                                                     && Vector3.Dot(
                                                         a.normal_on_object_world.normalized,
                                                         b.normal_on_object_world.normalized
                                                     ) <= -0.25f));
        }

        private CameraTruth SampleCamera(GateContext context)
        {
            Camera camera = context.HeadCamera;
            Transform cameraTransform = camera.transform;
            float fy = FrozenGate.Height / (2f * Mathf.Tan(0.5f * camera.fieldOfView * Mathf.Deg2Rad));
            float fx = fy;
            Collider[] clearanceColliders = avatarColliders
                .Where(collider => collider != null && collider.enabled && collider.gameObject.activeInHierarchy)
                .ToArray();
            if (clearanceColliders.Length == 0)
                unavailableCameraClearanceSteps++;
            if (!IsChildOrSelf(cameraTransform, context.HeadCameraMount)
                || !IsChildOrSelf(context.HeadCameraMount, context.Head))
                cameraParentMismatchSteps++;
            float clearance = clearanceColliders.Length == 0
                ? 0f
                : clearanceColliders.Min(collider => Vector3.Distance(
                    cameraTransform.position,
                    collider.ClosestPoint(cameraTransform.position)
                ));
            return new CameraTruth
            {
                parent_id = cameraTransform.parent == null ? "none" : StableTransformId(cameraTransform.parent),
                mount_id = StableTransformId(context.HeadCameraMount),
                mount_pose = SampleTransformPose(context.HeadCameraMount, "camera/head_mount"),
                position_world_m = cameraTransform.position,
                rotation_world_xyzw = cameraTransform.rotation,
                pose_provenance = "engine_observed",
                intrinsics = new CameraIntrinsicsTruth
                {
                    width_px = FrozenGate.Width,
                    height_px = FrozenGate.Height,
                    fx_px = fx,
                    fy_px = fy,
                    cx_px = FrozenGate.Width * 0.5f,
                    cy_px = FrozenGate.Height * 0.5f,
                    near_m = camera.nearClipPlane,
                    far_m = camera.farClipPlane,
                    provenance = "derived",
                    formula_or_source = "pinhole intrinsics from frozen raster size and Unity vertical field of view"
                },
                world_to_camera_extrinsics = MatrixToArray(camera.worldToCameraMatrix),
                extrinsics_provenance = "engine_observed",
                clearance_m = clearance,
                clearance_provenance = clearanceColliders.Length == 0 ? "unavailable" : "derived",
                clearance_formula_or_source = "minimum distance from optical center to avatar collider ClosestPoint",
                optical_vs_face_forward_deg = Vector3.Angle(context.Head.forward, cameraTransform.forward),
                optical_alignment_provenance = "derived"
            };
        }

        private DerivedStateTruth SampleDerivedState(GateContext context)
        {
            List<JointProprioceptionTruth> joints = new List<JointProprioceptionTruth>
            {
                SampleRelativeJoint("root", context.AuthorityRoot.transform, bodySegments["root"]),
                SampleRelativeJoint("pelvis", bodySegments["root"], bodySegments["pelvis"]),
                SampleRelativeJoint("torso", bodySegments["pelvis"], bodySegments["torso"]),
                SampleRelativeJoint("neck", bodySegments["torso"], bodySegments["neck"]),
                SampleRelativeJoint("head", bodySegments["neck"], bodySegments["head"]),
                SampleRelativeJoint("left_shoulder", bodySegments["torso"], bodySegments["left_shoulder"]),
                SampleRelativeJoint("left_upper_arm", bodySegments["left_shoulder"], bodySegments["left_upper_arm"]),
                SampleRelativeJoint("left_elbow", bodySegments["left_upper_arm"], bodySegments["left_elbow"]),
                SampleRelativeJoint("left_lower_arm", bodySegments["left_elbow"], bodySegments["left_lower_arm"]),
                SampleRelativeJoint("left_forearm", bodySegments["left_lower_arm"], bodySegments["left_forearm"]),
                SampleRelativeJoint("left_wrist", bodySegments["left_forearm"], bodySegments["left_wrist"]),
                SampleRelativeJoint("left_palm", bodySegments["left_wrist"], bodySegments["left_palm"]),
                SampleRelativeJoint("right_shoulder", bodySegments["torso"], bodySegments["right_shoulder"]),
                SampleRelativeJoint("right_upper_arm", bodySegments["right_shoulder"], bodySegments["right_upper_arm"]),
                SampleRelativeJoint("right_elbow", bodySegments["right_upper_arm"], bodySegments["right_elbow"]),
                SampleRelativeJoint("right_lower_arm", bodySegments["right_elbow"], bodySegments["right_lower_arm"]),
                SampleRelativeJoint("right_forearm", bodySegments["right_lower_arm"], bodySegments["right_forearm"]),
                SampleRelativeJoint("right_wrist", bodySegments["right_forearm"], bodySegments["right_wrist"]),
                SampleRelativeJoint("right_palm", bodySegments["right_wrist"], bodySegments["right_palm"])
            };
            foreach (string side in new[] { "left", "right" })
            foreach (string digit in FrozenGate.Digits)
            {
                Transform[] segments = context.FingerSegments[side + "_" + digit];
                for (int index = 0; index < segments.Length; index++)
                {
                    Transform parent = index == 0 ? bodySegments[side + "_palm"] : segments[index - 1];
                    joints.Add(SampleRelativeJoint(side + "_" + digit + "_" + index, parent, segments[index]));
                }
            }

            Vector3 headPosition = bodySegments["head"].position;
            Quaternion headRotation = bodySegments["head"].rotation;
            DerivedVectorTruth gyroscope;
            if (hasPriorHead)
            {
                Vector3 worldOmega = QuaternionAngularVelocityWorld(priorHeadRotation, headRotation, Dt);
                gyroscope = new DerivedVectorTruth
                {
                    value = Quaternion.Inverse(headRotation) * worldOmega,
                    provenance = "derived",
                    formula_or_source = "head-local quaternion finite difference: R_head^-1 * log(R_t * R_(t-1)^-1) / dt",
                    units = "rad/s"
                };
            }
            else
            {
                gyroscope = UnavailableVector("one prior head rotation is required", "rad/s");
            }

            DerivedVectorTruth accelerometer;
            if (hasPriorPriorHead)
            {
                Vector3 worldAcceleration = (headPosition - 2f * priorHeadPosition + priorPriorHeadPosition) / (Dt * Dt);
                accelerometer = new DerivedVectorTruth
                {
                    value = Quaternion.Inverse(headRotation) * (worldAcceleration - Physics.gravity),
                    provenance = "derived",
                    formula_or_source = "R_head^-1 * ((p_t - 2*p_(t-1) + p_(t-2))/dt^2 - Physics.gravity)",
                    units = "m/s2"
                };
            }
            else
            {
                accelerometer = UnavailableVector("two prior head positions are required", "m/s2");
            }

            if (hasPriorHead)
            {
                priorPriorHeadPosition = priorHeadPosition;
                hasPriorPriorHead = true;
            }
            priorHeadPosition = headPosition;
            priorHeadRotation = headRotation;
            hasPriorHead = true;
            return new DerivedStateTruth
            {
                joint_proprioception = joints.ToArray(),
                head_accelerometer_m_s2 = accelerometer,
                head_gyroscope_rad_s = gyroscope,
                biological_torque = new UnavailableTruth
                {
                    provenance = "unavailable",
                    formula_or_source = "not measured and not claimed",
                    units = "unavailable"
                }
            };
        }

        private JointProprioceptionTruth SampleRelativeJoint(string jointId, Transform parent, Transform child)
        {
            Quaternion relativeRotation = Quaternion.Inverse(parent.rotation) * child.rotation;
            Vector3 rotationVector = QuaternionRotationVector(relativeRotation);
            string provenance = "unavailable";
            Vector3 relativeAngularVelocity = Vector3.zero;
            if (priorRelativeRotations.TryGetValue(jointId, out Quaternion prior))
            {
                relativeAngularVelocity = QuaternionAngularVelocityWorld(prior, relativeRotation, Dt);
                provenance = "derived";
            }
            priorRelativeRotations[jointId] = relativeRotation;
            return new JointProprioceptionTruth
            {
                joint_id = jointId,
                parent_id = StableTransformId(parent),
                child_id = StableTransformId(child),
                relative_rotation_xyzw = relativeRotation,
                relative_rotation_vector_rad = rotationVector,
                relative_angular_velocity_rad_s = relativeAngularVelocity,
                position_provenance = "derived",
                velocity_provenance = provenance,
                formula_or_source = "parent^-1 * child rotation; angular velocity from consecutive relative rotations at 240 Hz"
            };
        }

        private static Vector3 QuaternionAngularVelocityWorld(Quaternion previous, Quaternion current, float dt)
        {
            Quaternion delta = current * Quaternion.Inverse(previous);
            delta.ToAngleAxis(out float angleDegrees, out Vector3 axis);
            if (angleDegrees > 180f)
                angleDegrees -= 360f;
            if (axis.sqrMagnitude < 1e-12f || Mathf.Abs(angleDegrees) < 1e-8f)
                return Vector3.zero;
            return axis.normalized * (angleDegrees * Mathf.Deg2Rad / dt);
        }

        private static Vector3 QuaternionRotationVector(Quaternion rotation)
        {
            rotation.ToAngleAxis(out float angleDegrees, out Vector3 axis);
            if (angleDegrees > 180f)
                angleDegrees -= 360f;
            if (axis.sqrMagnitude < 1e-12f || Mathf.Abs(angleDegrees) < 1e-8f)
                return Vector3.zero;
            return axis.normalized * (angleDegrees * Mathf.Deg2Rad);
        }

        private static DerivedVectorTruth UnavailableVector(string reason, string units)
        {
            return new DerivedVectorTruth
            {
                value = Vector3.zero,
                provenance = "unavailable",
                formula_or_source = reason,
                units = units
            };
        }

        private static string StableTransformId(Transform transform)
        {
            List<string> parts = new List<string>();
            Transform cursor = transform;
            while (cursor != null)
            {
                parts.Add(cursor.name + "[" + cursor.GetSiblingIndex() + "]");
                cursor = cursor.parent;
            }
            parts.Reverse();
            return string.Join("/", parts);
        }

        private static bool IsChildOrSelf(Transform child, Transform expectedAncestor)
        {
            return child != null && expectedAncestor != null
                   && (child == expectedAncestor || child.IsChildOf(expectedAncestor));
        }

        private static string ContactKey(string colliderA, string colliderB, Vector3 point)
        {
            return colliderA + "|" + colliderB + "|"
                   + Mathf.RoundToInt(point.x * 10000f) + "|"
                   + Mathf.RoundToInt(point.y * 10000f) + "|"
                   + Mathf.RoundToInt(point.z * 10000f);
        }

        private string ContactPersistentObjectId(GateContext context, string colliderA, string colliderB)
        {
            if (!targetColliderIds.Contains(colliderA) && !targetColliderIds.Contains(colliderB))
                return "";
            PhysicsTruthObjectIdentity identity = context.TargetBody.GetComponent<PhysicsTruthObjectIdentity>();
            return identity != null && !string.IsNullOrWhiteSpace(identity.persistent_id)
                ? identity.persistent_id
                : context.TargetBody.gameObject.name;
        }

        private Vector3 ContactNormalOnObject(
            string colliderA,
            string colliderB,
            Vector3 canonicalNormalAtoB
        )
        {
            if (targetColliderIds.Contains(colliderA))
                return canonicalNormalAtoB;
            if (targetColliderIds.Contains(colliderB))
                return -canonicalNormalAtoB;
            return Vector3.zero;
        }

        private static string ContactHand(string colliderA, string colliderB)
        {
            string value = (colliderA + " " + colliderB).ToLowerInvariant();
            if (value.Contains(".l[") || value.Contains("collider_left_") || value.Contains("_left_")) return "left";
            if (value.Contains(".r[") || value.Contains("collider_right_") || value.Contains("_right_")) return "right";
            return "";
        }

        private static string ContactDigit(string colliderA, string colliderB)
        {
            string value = (colliderA + " " + colliderB).ToLowerInvariant();
            if (value.Contains("thumb") || value.Contains("finger1")) return "thumb";
            if (value.Contains("index") || value.Contains("finger2")) return "index";
            if (value.Contains("middle") || value.Contains("finger3")) return "middle";
            if (value.Contains("ring") || value.Contains("finger4")) return "ring";
            if (value.Contains("little") || value.Contains("pinky") || value.Contains("finger5")) return "little";
            if (value.Contains("palm") || value.Contains("wrist")) return "palm";
            return "";
        }

        private static Vector3 PointVelocity(Collider collider, Vector3 pointWorldM)
        {
            Rigidbody body = collider.attachedRigidbody;
            return body == null ? Vector3.zero : body.GetPointVelocity(pointWorldM);
        }

        private static float[] MatrixToArray(Matrix4x4 matrix)
        {
            float[] values = new float[16];
            for (int row = 0; row < 4; row++)
            for (int column = 0; column < 4; column++)
                values[row * 4 + column] = matrix[row, column];
            return values;
        }

        private static string TruthLabel(TruthSource source)
        {
            switch (source)
            {
                case TruthSource.Commanded: return "commanded";
                case TruthSource.EngineObserved: return "engine_observed";
                case TruthSource.PhysXMeasured: return "physx_measured";
                case TruthSource.Derived: return "derived";
                default: return "unavailable";
            }
        }

        private static TruthProvenanceRow[] ProvenanceRegistry()
        {
            return new[]
            {
                new TruthProvenanceRow("body_hand_camera_pose", "engine_observed", "Unity Transform after Physics.Simulate", "m and quaternion_xyzw"),
                new TruthProvenanceRow("body_hand_segment_velocity", "derived", "finite difference of consecutive post-step poses", "m/s and rad/s"),
                new TruthProvenanceRow("controller_targets", "commanded", "motion controller telemetry held during the completed step", "SI plus quaternion_xyzw"),
                new TruthProvenanceRow("controller_errors", "derived", "commanded target minus engine-observed state", "m, deg, unit closure"),
                new TruthProvenanceRow("contacts", "physx_measured", "Unity collision callback ContactPoint and Collision", "m, m/s, N*s"),
                new TruthProvenanceRow("free_object", "physx_measured", "Unity Rigidbody after Physics.Simulate", "m, m/s, rad/s"),
                new TruthProvenanceRow("dynamic_finger_bodies", "physx_measured", "thirty Unity dynamic Rigidbodies after Physics.Simulate", "kg, m/s, rad/s"),
                new TruthProvenanceRow("compliant_finger_joints", "engine_observed", "ConfigurableJoint targets and drives consumed by PhysX", "quaternion_xyzw, rad/s, N*m"),
                new TruthProvenanceRow("force_bearing_qualification", "derived", "strict dwell over eligible current-step separation, per-point impulse, digit identity, and opposing normals", "s, m, N*s"),
                new TruthProvenanceRow("authority_counters", "engine_observed_and_source_audited", "runtime boundary snapshots plus mandatory hashed canonical source audit", "event counts"),
                new TruthProvenanceRow("joint_proprioception", "derived", "relative rotations and finite differences", "rad and rad/s"),
                new TruthProvenanceRow("head_imu", "derived", "head pose finite differences in head-local coordinates", "m/s2 and rad/s"),
                new TruthProvenanceRow("biological_torque", "unavailable", "not measured and not claimed", "unavailable")
            };
        }

        private static void WriteJson(string path, object value)
        {
            File.WriteAllText(path, JsonUtility.ToJson(value, true) + "\n", new UTF8Encoding(false));
        }

        private static string Sha256(string path)
        {
            using (SHA256 hash = SHA256.Create())
            using (FileStream stream = File.OpenRead(path))
                return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        private struct PriorPose
        {
            public Vector3 position;
            public Quaternion rotation;
        }

        private struct TargetAuthoritySnapshot
        {
            public Vector3 rigidbodyPosition;
            public Quaternion rigidbodyRotation;
            public Vector3 transformPosition;
            public Quaternion transformRotation;
            public Vector3 linearVelocity;
            public Vector3 angularVelocity;
            public bool isKinematic;
            public string parentId;
            public string[] jointIds;
        }

        private sealed class PendingContact
        {
            public Collider colliderA;
            public Collider colliderB;
            public string colliderAId;
            public string colliderBId;
            public Vector3 pointWorldM;
            public Vector3 normalWorld;
            public float separationM;
            public Vector3 relativeVelocityWorldMps;
            public Vector3 collisionPairImpulseNs;
            public string impulseScope;
            public string source;
        }
    }

    [DisallowMultipleComponent]
    public sealed class PhysicsTruthContactProbe : MonoBehaviour
    {
        private PhysicsTruthRecorder owner;

        internal void Configure(PhysicsTruthRecorder recorder)
        {
            owner = recorder;
        }

        private void OnCollisionEnter(Collision collision)
        {
            owner?.ObserveCollision(collision);
        }

        private void OnCollisionStay(Collision collision)
        {
            owner?.ObserveCollision(collision);
        }
    }

    [Serializable]
    public sealed class EpisodeTraceRow
    {
        public string schema;
        public EpisodeContextTruth episode;
        public ClockTruth clock;
        public BodyStateTruth body_state;
        public BodyHandsTruth hand_state;
        public ControllerStateTruth controller_state;
        public ContactTruthRow[] contacts;
        public ObjectTruth[] objects;
        public CameraTruth camera_state;
        public DerivedStateTruth derived_state;
        public string[] assistance_ledger;
        public AuthorityLedgerEntry[] recovery_ledger;
        public AuthorityAuditState authority_counters;
        public ForceBearingQualificationTruth force_bearing_qualification;
    }

    [Serializable]
    public sealed class EpisodeContextTruth
    {
        public string episode_id;
        public string cell_id;
        public string room_family;
        public string garment_configuration_id;
        public string target_id;
        public string destination_id;
        public string contact_strategy;
        public string final_gaze_zone;
    }

    [Serializable]
    public sealed class ClockTruth
    {
        public int physics_step;
        public int render_frame;
        public float time_s;
        public float dt_s;
        public int physics_hz;
        public int render_hz;
        public int steps_per_render_frame;
        public string sample_phase;
        public string phase_id;
    }

    [Serializable]
    public sealed class PoseTruth
    {
        public Vector3 position_world_m;
        public Quaternion rotation_world_xyzw = Quaternion.identity;
        public Vector3 linear_velocity_world_m_s;
        public Vector3 angular_velocity_world_rad_s;
        public string position_provenance;
        public string rotation_provenance;
        public string linear_velocity_provenance;
        public string angular_velocity_provenance;
        public string velocity_formula_or_source;
    }

    [Serializable]
    public sealed class BodyStateTruth
    {
        public PoseTruth root;
        public PoseTruth pelvis;
        public PoseTruth torso;
        public PoseTruth neck;
        public PoseTruth head;
        public PoseTruth left_shoulder;
        public PoseTruth left_upper_arm;
        public PoseTruth left_elbow;
        public PoseTruth left_lower_arm;
        public PoseTruth left_forearm;
        public PoseTruth left_wrist;
        public PoseTruth left_palm;
        public PoseTruth right_shoulder;
        public PoseTruth right_upper_arm;
        public PoseTruth right_elbow;
        public PoseTruth right_lower_arm;
        public PoseTruth right_forearm;
        public PoseTruth right_wrist;
        public PoseTruth right_palm;
    }

    [Serializable]
    public sealed class BodyHandsTruth
    {
        public PoseTruth left_palm;
        public PoseTruth right_palm;
        public DigitTruth left_thumb;
        public DigitTruth left_index;
        public DigitTruth left_middle;
        public DigitTruth left_ring;
        public DigitTruth left_little;
        public DigitTruth right_thumb;
        public DigitTruth right_index;
        public DigitTruth right_middle;
        public DigitTruth right_ring;
        public DigitTruth right_little;
    }

    [Serializable]
    public sealed class DigitTruth
    {
        public string hand;
        public string digit;
        public FingerSegmentTruth[] segments;
        public float closure_commanded;
        public string closure_commanded_provenance;
        public float closure_observed;
        public string closure_observed_provenance;
        public string closure_observed_formula;
        public DynamicFingerBodyTruth[] dynamic_body_states;
        public CompliantFingerJointTruth[] compliant_joint_states;
    }

    [Serializable]
    public sealed class DynamicFingerBodyTruth
    {
        public string body_id;
        public PoseTruth pose;
        public bool is_kinematic;
        public float mass_kg;
        public Vector3 linear_velocity_world_m_s;
        public Vector3 angular_velocity_world_rad_s;
        public bool sleeping;
        public string collision_detection_mode;
        public string provenance;
        public string physical_authority;
    }

    [Serializable]
    public sealed class CompliantFingerJointTruth
    {
        public string joint_id;
        public string connected_body_id;
        public Quaternion target_rotation_xyzw = Quaternion.identity;
        public Vector3 target_angular_velocity_rad_s;
        public float angular_x_drive_spring_n_m_rad;
        public float angular_x_drive_damper_n_m_s_rad;
        public float angular_x_drive_max_force_n_m;
        public float angular_yz_drive_spring_n_m_rad;
        public float angular_yz_drive_damper_n_m_s_rad;
        public float angular_yz_drive_max_force_n_m;
        public string provenance;
        public string drive_provenance;
    }

    [Serializable]
    public sealed class FingerSegmentTruth
    {
        public int segment_index;
        public string segment_id;
        public PoseTruth pose;
    }

    [Serializable]
    public sealed class ControllerStateTruth
    {
        public int telemetry_physics_step;
        public int expected_physics_step;
        public string phase_id;
        public ControllerTargetTruth[] targets;
        public ControllerErrorTruth[] errors;
        public string recovery_state;
        public string recovery_state_provenance;
        public bool right_opposition_qualified;
        public bool meaningful_left_support_qualified;
        public bool carry_contacts_maintained;
        public int right_opposition_dwell_steps;
        public int left_support_dwell_steps;
        public int bimanual_qualification_step;
        public SpeedLimitsTruth speed_limits;
        public ClosureLimitsTruth closure_limits;
        public bool release_commanded;
        public string[] events;
        public string provenance;
    }

    [Serializable]
    public sealed class ContactTruthRow
    {
        public int physics_step;
        public string collider_a;
        public string collider_b;
        public Vector3 point_world_m;
        public Vector3 normal_world;
        public float separation_m;
        public Vector3 relative_velocity_world_m_s;
        public Vector3 available_impulse_n_s;
        public float available_impulse_magnitude_n_s;
        public bool nonzero_impulse_observed;
        public float mean_force_equivalent_n;
        public string force_evidence_semantics;
        public bool qualification_separation_eligible;
        public string available_impulse_scope;
        public string persistent_object_id;
        public string hand;
        public string digit;
        public Vector3 normal_on_object_world;
        public string provenance;
        public string formula_or_source;
    }

    [Serializable]
    public sealed class ForceBearingQualificationTruth
    {
        public bool right_current_measured_impulse_opposition;
        public bool right_current_thumb_present;
        public string[] right_current_non_thumb_digits;
        public int right_measured_impulse_dwell_steps;
        public float right_measured_impulse_dwell_s;
        public bool right_measured_impulse_qualified;
        public float right_required_strictly_greater_s;
        public bool left_current_measured_impulse_support;
        public string[] left_current_non_little_digits;
        public int left_measured_impulse_dwell_steps;
        public float left_measured_impulse_dwell_s;
        public bool left_measured_impulse_qualified;
        public float left_required_strictly_greater_s;
        public bool object_unsupported_current;
        public bool object_unsupported_during_qualified_manipulation;
        public string required_destination_id;
        public string observed_support_id;
        public bool current_support_is_required_destination;
        public bool release_command_ever_observed;
        public bool free_release_at_required_destination_current;
        public bool release_at_required_destination;
        public float finger_object_max_penetration_m;
        public float target_support_max_penetration_m;
        public bool support_continuous_until_commanded_opening;
        public bool qualification_ever_active;
        public int measured_bimanual_qualification_step;
        public bool initial_overlap_disqualified;
        public float maximum_qualified_lift_m;
        public bool lift_over_0_10_m;
        public float maximum_qualified_turn_deg;
        public bool turn_over_30_deg;
        public string impulse_semantics;
        public float maximum_eligible_separation_m;
        public float maximum_eligible_penetration_m;
        public float minimum_nonzero_impulse_n_s;
        public float maximum_opposing_normal_dot;
        public string provenance;
    }

    [Serializable]
    public sealed class ObjectTruth
    {
        public bool active;
        public string availability_provenance;
        public string persistent_id;
        public string semantic_id;
        public string instance_id;
        public string identity_provenance;
        public PoseTruth pose;
        public Vector3 linear_velocity_world_m_s;
        public Vector3 angular_velocity_world_rad_s;
        public string support_id;
        public string support_provenance;
        public string support_formula_or_source;
        public bool sleeping;
        public string sleeping_provenance;
        public bool is_kinematic;
        public string parent_id;
        public bool free_dynamic;
        public string physical_authority;
    }

    [Serializable]
    public sealed class CameraIntrinsicsTruth
    {
        public int width_px;
        public int height_px;
        public float fx_px;
        public float fy_px;
        public float cx_px;
        public float cy_px;
        public float near_m;
        public float far_m;
        public string provenance;
        public string formula_or_source;
    }

    [Serializable]
    public sealed class CameraTruth
    {
        public string parent_id;
        public string mount_id;
        public PoseTruth mount_pose;
        public Vector3 position_world_m;
        public Quaternion rotation_world_xyzw = Quaternion.identity;
        public string pose_provenance;
        public CameraIntrinsicsTruth intrinsics;
        public float[] world_to_camera_extrinsics;
        public string extrinsics_provenance;
        public float clearance_m;
        public string clearance_provenance;
        public string clearance_formula_or_source;
        public float optical_vs_face_forward_deg;
        public string optical_alignment_provenance;
    }

    [Serializable]
    public sealed class JointProprioceptionTruth
    {
        public string joint_id;
        public string parent_id;
        public string child_id;
        public Quaternion relative_rotation_xyzw = Quaternion.identity;
        public Vector3 relative_rotation_vector_rad;
        public Vector3 relative_angular_velocity_rad_s;
        public string position_provenance;
        public string velocity_provenance;
        public string formula_or_source;
    }

    [Serializable]
    public sealed class DerivedVectorTruth
    {
        public Vector3 value;
        public string provenance;
        public string formula_or_source;
        public string units;
    }

    [Serializable]
    public sealed class UnavailableTruth
    {
        public string provenance;
        public string formula_or_source;
        public string units;
    }

    [Serializable]
    public sealed class DerivedStateTruth
    {
        public JointProprioceptionTruth[] joint_proprioception;
        public DerivedVectorTruth head_accelerometer_m_s2;
        public DerivedVectorTruth head_gyroscope_rad_s;
        public UnavailableTruth biological_torque;
    }

    [Serializable]
    public sealed class TruthProvenanceRow
    {
        public string field;
        public string provenance;
        public string formula_or_source;
        public string units;

        public TruthProvenanceRow(string field, string provenance, string formulaOrSource, string units)
        {
            this.field = field;
            this.provenance = provenance;
            formula_or_source = formulaOrSource;
            this.units = units;
        }
    }

    [Serializable]
    public sealed class ClockReceipt
    {
        public string schema;
        public string episode_id;
        public int physics_hz;
        public int render_hz;
        public int exact_steps_per_render_frame;
        public int expected_row_count;
        public int row_count;
        public int first_physics_step;
        public int last_physics_step;
        public int first_render_frame;
        public int last_render_frame;
        public float first_time_s;
        public float last_time_s;
        public bool contiguous_physics_steps;
        public bool contiguous_time_samples;
        public bool exact_integer_render_mapping;
        public float maximum_clock_error_s;
        public string sample_phase;
        public bool passed;
    }

    [Serializable]
    public sealed class TraceManifest
    {
        public string schema;
        public string episode_id;
        public string trace_schema;
        public string trace_file;
        public string trace_sha256;
        public string clock_receipt_file;
        public string clock_receipt_sha256;
        public string interaction_summary_file;
        public string interaction_summary_sha256;
        public float finger_object_max_penetration_m;
        public float target_support_max_penetration_m;
        public string required_destination_id;
        public string observed_destination_support_id;
        public bool object_unsupported_during_qualified_manipulation;
        public bool release_at_required_destination;
        public int row_count;
        public bool controller_provider_bound;
        public int unavailable_controller_steps;
        public int controller_clock_mismatch_steps;
        public int maximum_contacts_in_step;
        public int total_contact_rows;
        public int maximum_assistance_entries;
        public int avatar_collider_count;
        public int unavailable_camera_clearance_steps;
        public int non_free_object_steps;
        public int camera_parent_mismatch_steps;
        public int unavailable_dynamic_finger_bindings;
        public bool assistance_ledger_empty;
        public int recovery_ledger_entries;
        public AuthorityAuditState authority_counters;
        public string source_audit_sha256;
        public bool source_audit_present;
        public string runtime_detection_coverage;
        public string force_torque_detection_boundary;
        public bool zero_counters_are_independent_proof;
        public bool clock_passed;
        public bool one_authority_state;
        public string object_state_authority;
        public string embodiment_state_authority;
        public string biological_torque;
        public bool trace_complete;
        public TruthProvenanceRow[] provenance_registry;
    }

    [Serializable]
    public sealed class InteractionSummary
    {
        public string schema;
        public string episode_id;
        public string target_id;
        public string required_destination_id;
        public string observed_destination_support_id;
        public float finger_object_max_penetration_m;
        public float target_support_max_penetration_m;
        public float finger_object_penetration_limit_m;
        public float target_support_penetration_limit_m;
        public bool finger_object_penetration_passed;
        public bool target_support_penetration_passed;
        public bool right_measured_impulse_qualified;
        public bool left_measured_impulse_qualified;
        public bool lift_over_0_10_m;
        public bool turn_over_30_deg;
        public bool object_unsupported_during_qualified_manipulation;
        public bool release_command_ever_observed;
        public bool release_at_required_destination;
        public string provenance;
    }

    [Serializable]
    public sealed class HashReceipt
    {
        public string schema;
        public string episode_id;
        public string algorithm;
        public string trace_file;
        public string trace_sha256;
        public string manifest_file;
        public string manifest_sha256;
        public string clock_receipt_file;
        public string clock_receipt_sha256;
        public string interaction_summary_file;
        public string interaction_summary_sha256;
    }
}
#endif
