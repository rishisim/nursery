#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace ProceduralSceneGate
{
    /// <summary>
    /// Synchronous Built-in-pipeline capture from the post-PhysX authority state.
    /// This class never advances physics and never changes a body, object, garment,
    /// head, or camera transform after Bind.  Replacement shaders are used only for
    /// registered non-RGB streams; QA proxy drawing is isolated to external/overlay.
    /// </summary>
    public sealed class RegisteredCapture : IRegisteredCaptureModule
    {
        public const string LedgerFileName = "capture_frame_ledger.jsonl";
        public const string ManifestFileName = "registered_capture_manifest.json";
        public const string LabelManifestFileName = "semantic_instance_manifest.json";
        public const string FovQualificationFileName = "camera_fov_qualification.json";
        public const string ContactProjectionFileName = "contact_projection.json";
        public const string VisualMeasurementsFileName = "visual_measurements.json";

        private static readonly float[] VerticalFovCandidatesDeg = { 60f, 68f, 75f };
        private const float NearClipM = 0.01f;
        private const float FarClipM = 20f;
        private const float MinimumClearanceM = 0.001f;
        private const float MinimumMilestoneViewportMargin = 0.035f;
        private const int SweptClearanceIntervals = 8;
        private const float MaximumOpticalAngleDeg = 15f;
        private const float MaximumRollDeg = 12f;
        private const float MaximumLinearSpeedMps = 1.5f;
        private const float MaximumAngularSpeedDegps = 120f;
        private const int QaOverlayLayer = 31;

        private readonly List<CaptureLabelBinding> labelBindings = new List<CaptureLabelBinding>();
        private readonly List<float> clearances = new List<float>();
        private readonly List<float> opticalAngles = new List<float>();
        private readonly List<float> absoluteRolls = new List<float>();
        private readonly List<float> linearSpeeds = new List<float>();
        private readonly List<float> angularSpeeds = new List<float>();
        private readonly List<RegisteredContactProjectionRecord> contactProjectionRecords =
            new List<RegisteredContactProjectionRecord>();

        private GateContext boundContext;
        private string captureMode;
        private string stage;
        private StreamWriter ledger;
        private Shader metricDepthShader;
        private Shader semanticInstanceShader;
        private RegisteredCaptureQaOverlay qaOverlay;
        private int framesCaptured;
        private int firstFrame = -1;
        private int lastFrame = -1;
        private bool exactClock = true;
        private bool contiguousFrames = true;
        private bool allModalitiesStateInvariant = true;
        private bool completed;
        private Vector3 previousCameraPosition;
        private Quaternion previousCameraRotation;
        private bool hasPreviousCameraPose;
        private Vector3 frozenExternalPosition;
        private Quaternion frozenExternalRotation;
        private Vector3 frozenHeadMountLocalPosition;
        private Quaternion frozenHeadMountLocalRotation;
        private Vector3 previousClearancePosition;
        private Quaternion previousClearanceRotation;
        private bool hasPreviousClearancePose;
        private FovQualificationReceipt fovQualification;

        public void Bind(GateContext context)
        {
            if (boundContext != null)
                throw new InvalidOperationException("RegisteredCapture is already bound");
            ValidateBaseBindings(context);
            boundContext = context;
            captureMode = ReadCaptureMode();
            stage = Environment.GetEnvironmentVariable("PROCEDURAL_GATE_STAGE") ?? "integrated";

            ConfigureAuthoritativeHeadCamera(context);
            ConfigureFixedExternalCamera(context);
            ConfigureRendererLabels(context);
            ConfigureQaOverlay(context);

            ValidateAuthoritativeCameraChain(context);
            CameraClearanceResult bindClearance = ProspectivelyFreezeHeadMountClearance(context);
            fovQualification = ProspectivelyFreezeFieldOfView(context);
            Directory.CreateDirectory(context.OutputRoot);
            File.WriteAllText(
                Path.Combine(context.OutputRoot, FovQualificationFileName),
                JsonUtility.ToJson(fovQualification, true) + "\n",
                new UTF8Encoding(false)
            );
            if (!fovQualification.qualification_pass)
                throw new InvalidOperationException("prospective FOV/event-view qualification rejected before capture: " + fovQualification.failure_reason);
            context.HeadCamera.fieldOfView = fovQualification.selected_vertical_fov_deg;
            frozenHeadMountLocalPosition = context.HeadCameraMount.localPosition;
            frozenHeadMountLocalRotation = context.HeadCameraMount.localRotation;

            float opticalAngle = Vector3.Angle(context.Head.forward, context.HeadCamera.transform.forward);
            if (opticalAngle > MaximumOpticalAngleDeg + 1e-4f)
                throw new InvalidOperationException("head optical mount exceeds the frozen 15 degree neutral-axis limit");
            if (bindClearance.inside_body_mesh || bindClearance.minimum_surface_distance_m < MinimumClearanceM)
                throw new InvalidOperationException("head optical origin/near plane lacks positive avatar/clothing clearance");

            if (captureMode == "none")
                return;
            CreateOutputDirectories(context.OutputRoot, captureMode == "all");
            ledger = new StreamWriter(
                new FileStream(Path.Combine(context.OutputRoot, LedgerFileName), FileMode.Create, FileAccess.Write, FileShare.Read),
                new UTF8Encoding(false)
            );
        }

        public void CaptureFrozenFrame(GateContext context)
        {
            RequireActiveContext(context);
            ValidateCaptureClock(context);
            if (captureMode == "none")
                return;

            int physicsStepAtEntry = context.PhysicsStep;
            int renderFrameAtEntry = context.RenderFrame;
            string authorityState = AuthorityStateSha256(context);
            PoseFingerprint headCameraPose = PoseFingerprint.From(context.HeadCamera.transform);
            AssertExternalCameraStillFixed();
            AssertFixedHeadMountAndChain(context);

            // Clearance is a prospective render reject, not a diagnostic sampled
            // after pixels have already been emitted. The sweep covers the prior
            // registered pose through this frozen pose, including the near plane.
            CameraClearanceResult clearance = MeasureSweptCameraClearance(context);
            if (clearance.inside_body_mesh || clearance.minimum_surface_distance_m < MinimumClearanceM)
                throw new InvalidOperationException("camera swept origin/near plane lacks positive head/hair/garment/furniture clearance before render");
            EventVisibilityInput[] visibility = EventVisibilityInputs(context);

            var receipts = new List<CaptureStreamReceipt>();
            string frameName = "frame_" + context.RenderFrame.ToString("D4", CultureInfo.InvariantCulture) + ".png";
            int heroMask = context.HeadCamera.cullingMask & ~(1 << QaOverlayLayer);
            receipts.Add(CaptureRgb(
                context.HeadCamera,
                Path.Combine(context.OutputRoot, "head", "rgb", frameName),
                "head_rgb_hero",
                heroMask,
                authorityState,
                context
            ));
            AssertFrozenState(context, physicsStepAtEntry, renderFrameAtEntry, authorityState, headCameraPose);

            if (captureMode == "all")
            {
                receipts.Add(CaptureReplacement(
                    context.HeadCamera,
                    Path.Combine(context.OutputRoot, "head", "depth", frameName),
                    "head_metric_depth_uint24_mm",
                    metricDepthShader,
                    0f,
                    heroMask,
                    authorityState,
                    context,
                    null
                ));
                AssertFrozenState(context, physicsStepAtEntry, renderFrameAtEntry, authorityState, headCameraPose);
                receipts.Add(CaptureReplacement(
                    context.HeadCamera,
                    Path.Combine(context.OutputRoot, "head", "semantic", frameName),
                    "head_semantic_uint24",
                    semanticInstanceShader,
                    0f,
                    heroMask,
                    authorityState,
                    context,
                    visibility
                ));
                AssertFrozenState(context, physicsStepAtEntry, renderFrameAtEntry, authorityState, headCameraPose);
                receipts.Add(CaptureReplacement(
                    context.HeadCamera,
                    Path.Combine(context.OutputRoot, "head", "instance", frameName),
                    "head_persistent_instance_uint24",
                    semanticInstanceShader,
                    1f,
                    heroMask,
                    authorityState,
                    context,
                    visibility
                ));
                AssertFrozenState(context, physicsStepAtEntry, renderFrameAtEntry, authorityState, headCameraPose);
                FinalizeRenderedLabelSamples(visibility, context, labelBindings);
                foreach (EventVisibilityInput input in visibility.Where(value =>
                             StringComparer.Ordinal.Equals(value.input_id, "physx_contact_point_input")))
                {
                    contactProjectionRecords.Add(new RegisteredContactProjectionRecord
                    {
                        render_frame = context.RenderFrame,
                        physics_step = context.PhysicsStep,
                        physical_contact_point_world_m = input.point_world_m,
                        expected_contact_collider_a = input.expected_contact_collider_a,
                        expected_contact_collider_b = input.expected_contact_collider_b,
                        pixel_xy = input.rendered_label_pixel_xy,
                        first_visible_collider_id = input.first_visible_collider_id,
                        first_visible_renderer_path = input.first_visible_renderer_path,
                        first_visible_semantic_uint24 = input.first_visible_semantic_uint24,
                        first_visible_persistent_instance_uint24 = input.first_visible_persistent_instance_uint24,
                        contact_projects_to_expected_visible_surface = input.contact_projects_to_expected_visible_surface,
                        contact_visible_in_registered_frame = input.contact_visible_in_registered_frame,
                        method = input.contact_projection_method,
                        provenance = input.provenance
                    });
                }
            }

            int externalMask = context.ExternalCamera.cullingMask & ~(1 << QaOverlayLayer);
            qaOverlay.capture_enabled = false;
            receipts.Add(CaptureRgb(
                context.ExternalCamera,
                Path.Combine(context.OutputRoot, "external", "clean", frameName),
                "external_clean",
                externalMask,
                authorityState,
                context
            ));
            AssertFrozenState(context, physicsStepAtEntry, renderFrameAtEntry, authorityState, headCameraPose);

            qaOverlay.PrepareFrame(context);
            qaOverlay.capture_enabled = true;
            receipts.Add(CaptureRgb(
                context.ExternalCamera,
                Path.Combine(context.OutputRoot, "external", "overlay", frameName),
                "external_collider_contact_overlay_QA_ONLY",
                externalMask,
                authorityState,
                context
            ));
            qaOverlay.capture_enabled = false;
            AssertFrozenState(context, physicsStepAtEntry, renderFrameAtEntry, authorityState, headCameraPose);

            CameraMotionSample motion = CameraMotion(context.HeadCamera.transform);
            float opticalVsFaceForward = Vector3.Angle(context.Head.forward, context.HeadCamera.transform.forward);
            float roll = CameraRollDeg(context.HeadCamera.transform, context.Head);
            CameraIntrinsics intrinsics = Intrinsics(context.HeadCamera);

            clearances.Add(clearance.minimum_surface_distance_m);
            opticalAngles.Add(opticalVsFaceForward);
            absoluteRolls.Add(Mathf.Abs(roll));
            linearSpeeds.Add(motion.linear_speed_m_s);
            angularSpeeds.Add(motion.angular_speed_deg_s);

            var row = new CaptureFrameLedgerRow
            {
                schema = "embodied.registered_capture_frame.v1",
                episode_id = context.EpisodeId,
                stage = stage,
                capture_mode = captureMode,
                render_frame = context.RenderFrame,
                physics_step = context.PhysicsStep,
                time_s = context.TimeSeconds,
                sample_phase = "same synchronous post-physics-step frozen state",
                physics_hz = FrozenGate.PhysicsHz,
                render_hz = FrozenGate.RenderHz,
                steps_per_render_frame = FrozenGate.StepsPerFrame,
                width_px = FrozenGate.Width,
                height_px = FrozenGate.Height,
                authority_state_sha256 = authorityState,
                authority_state_unchanged_across_modalities = true,
                physics_advanced_between_modalities = false,
                streams = receipts.ToArray(),
                camera_parent_id = StableTransformPath(context.HeadCamera.transform.parent),
                camera_position_world_m = context.HeadCamera.transform.position,
                camera_rotation_world_xyzw = context.HeadCamera.transform.rotation,
                camera_to_world_matrix_column_major = MatrixValues(context.HeadCamera.cameraToWorldMatrix),
                world_to_camera_matrix_column_major = MatrixValues(context.HeadCamera.worldToCameraMatrix),
                projection_matrix_column_major = MatrixValues(context.HeadCamera.projectionMatrix),
                intrinsics = intrinsics,
                camera_state_provenance = "engine_observed after the eighth PhysX step; fixed local mount under authoritative head",
                intrinsics_provenance = "derived from frozen Unity Camera vertical FOV and exact 1920x1080 raster",
                clearance_m = clearance.minimum_surface_distance_m,
                clearance_inside_body_mesh = clearance.inside_body_mesh,
                swept_clearance_m = clearance.minimum_surface_distance_m,
                swept_clearance_samples = clearance.sweep_samples,
                swept_clearance_scene_collider_m = clearance.minimum_scene_collider_distance_m,
                clearance_method = clearance.method,
                optical_vs_face_forward_deg = opticalVsFaceForward,
                roll_deg = roll,
                linear_speed_m_s = motion.linear_speed_m_s,
                angular_speed_deg_s = motion.angular_speed_deg_s,
                optical_within_tolerance = opticalVsFaceForward <= MaximumOpticalAngleDeg,
                clearance_within_tolerance = !clearance.inside_body_mesh && clearance.minimum_surface_distance_m >= MinimumClearanceM,
                roll_within_tolerance = Mathf.Abs(roll) <= MaximumRollDeg,
                linear_speed_within_tolerance = motion.linear_speed_m_s <= MaximumLinearSpeedMps,
                angular_speed_within_tolerance = motion.angular_speed_deg_s <= MaximumAngularSpeedDegps,
                event_visibility_inputs = visibility,
                event_visibility_interpretation = "geometric frustum inputs only; never promoted to visual evidence without dense decoded-video review",
                overlay_true_context_contact_count = qaOverlay.true_contact_count,
                overlay_qa_penetration_proxy_count = qaOverlay.qa_penetration_proxy_count,
                overlay_is_hero = false,
                overlay_contains_labeled_qa_proxies = true
            };
            ledger.WriteLine(JsonUtility.ToJson(row, false));
            ledger.Flush();

            if (firstFrame < 0) firstFrame = context.RenderFrame;
            lastFrame = context.RenderFrame;
            framesCaptured++;
        }

        public void Complete(GateContext context)
        {
            RequireActiveContext(context);
            ledger?.Flush();
            ledger?.Dispose();
            ledger = null;

            Directory.CreateDirectory(context.OutputRoot);
            WriteLabelManifest(context);
            var manifest = new RegisteredCaptureManifest
            {
                schema = "embodied.registered_capture_manifest.v1",
                episode_id = context.EpisodeId,
                stage = stage,
                capture_mode = captureMode,
                unity_version = Application.unityVersion,
                rendering_pipeline = "Unity Built-in Render Pipeline synchronous Camera.Render",
                width_px = FrozenGate.Width,
                height_px = FrozenGate.Height,
                fps = FrozenGate.RenderHz,
                physics_hz = FrozenGate.PhysicsHz,
                steps_per_render_frame = FrozenGate.StepsPerFrame,
                frames_captured = framesCaptured,
                first_frame = firstFrame,
                last_frame = lastFrame,
                expected_full_episode_frames = (int)(FrozenGate.DurationSeconds * FrozenGate.RenderHz),
                contiguous_frames = contiguousFrames,
                exact_integer_clock = exactClock,
                all_modalities_state_invariant = allModalitiesStateInvariant,
                physics_advanced_between_modalities = false,
                head_mount_parent = context.HeadCameraMount && context.HeadCameraMount.parent
                    ? StableTransformPath(context.HeadCameraMount.parent) : "UNAVAILABLE",
                head_mount_local_position_m = context.HeadCameraMount ? context.HeadCameraMount.localPosition : Vector3.zero,
                head_mount_local_rotation_xyzw = context.HeadCameraMount ? context.HeadCameraMount.localRotation : Quaternion.identity,
                head_mount_fixed_after_bind = true,
                target_lock = false,
                dependent_mount = false,
                independent_camera_animation = false,
                render_only_camera_keyframes = false,
                tabletop_view_source = "authoritative root/torso/neck/head motion through one fixed head-local optical transform",
                rgb_encoding = "PNG RGB24 sRGB hero; zero QA overlay layer pixels",
                metric_depth_encoding = "PNG RGB24 linear, little-endian uint24 millimetres: mm=R+256*G+65536*B; zero is background",
                semantic_encoding = "PNG RGB24 linear little-endian uint24 semantic ID; zero is background",
                instance_encoding = "PNG RGB24 linear little-endian uint24 persistent instance ID; zero is background",
                label_manifest = LabelManifestFileName,
                frame_ledger = captureMode == "none" ? "UNAVAILABLE_CAPTURE_MODE_NONE" : LedgerFileName,
                stream_roots = StreamRoots(captureMode),
                fov_qualification = fovQualification,
                fov_qualification_file = FovQualificationFileName,
                fov_qualification_sha256 = FileSha256(Path.Combine(context.OutputRoot, FovQualificationFileName)),
                selected_vertical_fov_deg = fovQualification != null ? fovQualification.selected_vertical_fov_deg : -1f,
                fov_freeze_rule = "prospectively audition 60/68/75 degrees on one immutable bind-time milestone geometry receipt; freeze the narrowest candidate with every required event inside the safe viewport and no body/support/furniture occluder",
                minimum_clearance_m = MinOrUnavailable(clearances),
                maximum_optical_vs_face_forward_deg = MaxOrUnavailable(opticalAngles),
                maximum_abs_roll_deg = MaxOrUnavailable(absoluteRolls),
                maximum_linear_speed_m_s = MaxOrUnavailable(linearSpeeds),
                maximum_angular_speed_deg_s = MaxOrUnavailable(angularSpeeds),
                visual_evidence_rule = "frame counters, hashes, frustum tests, and overlay proxies are QA inputs, not visual PASS evidence",
                hero_contains_proxy_pixels = false,
                external_overlay_is_separate_qa_only = true,
                provenance = "camera state engine-observed; intrinsics and motion derived; rendered pixels Unity-observed; overlay proxy rows explicitly QA-only"
            };
            File.WriteAllText(
                Path.Combine(context.OutputRoot, ManifestFileName),
                JsonUtility.ToJson(manifest, true) + "\n",
                new UTF8Encoding(false)
            );

            if (captureMode != "none")
            {
                WriteMeasuredCaptureAdapters(context);
            }

            qaOverlay?.DisposeMaterial();
            completed = true;
        }

        private void WriteMeasuredCaptureAdapters(GateContext context)
        {
            string ledgerPath = Path.Combine(context.OutputRoot, LedgerFileName);
            var projection = new RegisteredContactProjectionReport
            {
                schema = "embodied.registered_contact_projection.v1",
                episode_id = context.EpisodeId,
                source_capture_ledger = LedgerFileName,
                source_capture_ledger_sha256 = FileSha256(ledgerPath),
                records = contactProjectionRecords.ToArray(),
                provenance = "direct adapter of same-state PhysX ContactTruth projection rows embedded in the registered capture ledger"
            };
            string projectionPath = Path.Combine(context.OutputRoot, ContactProjectionFileName);
            File.WriteAllText(
                projectionPath,
                JsonUtility.ToJson(projection, true) + "\n",
                new UTF8Encoding(false)
            );

            // This receipt intentionally contains measurements only.  It cannot
            // stand in for the independent dense decoded-frame review required by
            // the visual gates, so the auditor keeps those checks UNAVAILABLE.
            var visualMeasurements = new RegisteredVisualMeasurements
            {
                schema = "embodied.registered_visual_measurements.v1",
                episode_id = context.EpisodeId,
                source_capture_ledger = LedgerFileName,
                source_capture_ledger_sha256 = FileSha256(ledgerPath),
                registered_contact_projection = ContactProjectionFileName,
                registered_contact_projection_sha256 = FileSha256(projectionPath),
                measured_capture_evidence_only = true,
                direct_decoded_frame_review_performed = false,
                registered_projection_record_count = contactProjectionRecords.Count,
                measurement_scope = "camera calibration/motion/clearance, rendered stream hashes, and same-state first-visible-surface projections only",
                provenance = "Unity-observed registered capture receipts; no anatomy, clipping, coherence, or aesthetic judgment is inferred"
            };
            File.WriteAllText(
                Path.Combine(context.OutputRoot, VisualMeasurementsFileName),
                JsonUtility.ToJson(visualMeasurements, true) + "\n",
                new UTF8Encoding(false)
            );

        }

        private static void ValidateBaseBindings(GateContext context)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (context.AuthorityRoot == null || context.AvatarRoot == null || context.Head == null
                || context.LeftPalm == null || context.RightPalm == null)
                throw new InvalidOperationException("capture requires authority, avatar, head, and both palm bindings");
            if (string.IsNullOrWhiteSpace(context.OutputRoot))
                throw new InvalidOperationException("capture requires GateContext.OutputRoot");
            if (FrozenGate.Width != 1920 || FrozenGate.Height != 1080 || FrozenGate.RenderHz != 30
                || FrozenGate.PhysicsHz != 240 || FrozenGate.StepsPerFrame != 8)
                throw new InvalidOperationException("registered capture constants differ from the frozen 240:30, 1920x1080 contract");
        }

        private void ConfigureAuthoritativeHeadCamera(GateContext context)
        {
            Camera camera = context.HeadCamera;
            Transform mount = context.HeadCameraMount;
            if (camera == null)
            {
                GameObject cameraObject = mount != null ? mount.gameObject : new GameObject("AUTHORITATIVE_FIXED_CHILD_HEAD_OPTICAL_MOUNT");
                camera = cameraObject.GetComponent<Camera>();
                // UnityEngine.Object has fake-null semantics, which null-coalescing does
                // not honor reliably for missing native components.
                if (!camera) camera = cameraObject.AddComponent<Camera>();
            }
            mount = camera.transform;
            mount.name = "AUTHORITATIVE_FIXED_CHILD_HEAD_OPTICAL_MOUNT";
            mount.SetParent(context.Head, false);
            mount.localRotation = Quaternion.identity;
            mount.position = ComputeHeadSurfaceMountWorld(context, 0.032f);

            camera.enabled = false; // only explicit synchronous Camera.Render is permitted
            // ProspectivelyFreezeFieldOfView replaces this widest temporary
            // preflight value with exactly one frozen candidate before capture.
            camera.fieldOfView = VerticalFovCandidatesDeg[VerticalFovCandidatesDeg.Length - 1];
            camera.aspect = FrozenGate.Width / (float)FrozenGate.Height;
            camera.nearClipPlane = NearClipM;
            camera.farClipPlane = FarClipM;
            camera.allowHDR = false;
            camera.allowMSAA = false;
            camera.clearFlags = CameraClearFlags.Skybox;
            camera.cullingMask &= ~(1 << QaOverlayLayer);
            context.HeadCameraMount = mount;
            context.HeadCamera = camera;

            metricDepthShader = Shader.Find("ProceduralSceneGate/MetricDepthUint24Millimetres");
            semanticInstanceShader = Shader.Find("ProceduralSceneGate/SemanticInstanceUint24");
            if (metricDepthShader == null || semanticInstanceShader == null)
                throw new InvalidOperationException("registered capture replacement shaders are missing");
        }

        private static void ValidateAuthoritativeCameraChain(GateContext context)
        {
            if (!context.Torso || !context.Neck || !context.Head)
                throw new InvalidOperationException("camera requires the frozen root/pelvis-to-torso-to-neck-to-head chain");
            if (!IsChildOrSelf(context.AvatarRoot.transform, context.AuthorityRoot.transform)
                || !IsChildOrSelf(context.Torso, context.AvatarRoot.transform)
                || !IsChildOrSelf(context.Neck, context.Torso)
                || !IsChildOrSelf(context.Head, context.Neck)
                || context.HeadCameraMount.parent != context.Head
                || context.HeadCamera.transform != context.HeadCameraMount)
                throw new InvalidOperationException("camera authority must be avatar root -> torso -> neck -> head -> one fixed measured optical mount");
        }

        private void AssertFixedHeadMountAndChain(GateContext context)
        {
            ValidateAuthoritativeCameraChain(context);
            if (Vector3.Distance(context.HeadCameraMount.localPosition, frozenHeadMountLocalPosition) > 1e-7f
                || Quaternion.Angle(context.HeadCameraMount.localRotation, frozenHeadMountLocalRotation) > 1e-5f)
                throw new InvalidOperationException("head optical mount changed after its bind-time measured local pose was frozen");
            if (Mathf.Abs(context.HeadCamera.fieldOfView - fovQualification.selected_vertical_fov_deg) > 1e-5f)
                throw new InvalidOperationException("head camera FOV changed after prospective candidate freeze");
        }

        private FovQualificationReceipt ProspectivelyFreezeFieldOfView(GateContext context)
        {
            MilestoneViewAnchor[] anchors = BuildMilestoneViewAnchors(context);
            if (anchors.Length == 0)
                throw new InvalidOperationException("FOV preflight has no required milestone geometry");

            Vector3 origin = context.HeadCamera.transform.position;
            Vector3 taskCentroid = Vector3.zero;
            foreach (MilestoneViewAnchor anchor in anchors) taskCentroid += anchor.point_world_m;
            taskCentroid /= anchors.Length;
            Vector3 predictedTaskAxis = (taskCentroid - origin).normalized;
            if (predictedTaskAxis.sqrMagnitude < 1e-8f)
                throw new InvalidOperationException("FOV preflight task axis is degenerate");
            Vector3 predictedUp = Vector3.ProjectOnPlane(context.Head.up, predictedTaskAxis).normalized;
            if (predictedUp.sqrMagnitude < 1e-8f) predictedUp = Vector3.up;
            Quaternion predictedTaskView = Quaternion.LookRotation(predictedTaskAxis, predictedUp);

            string geometrySha256 = MilestoneGeometrySha256(context, anchors, origin, predictedTaskView);
            var candidates = new List<FovCandidateReceipt>();
            foreach (float candidate in VerticalFovCandidatesDeg)
            {
                MilestoneCandidateResult[] results = anchors
                    .Select(anchor => EvaluateMilestoneCandidate(context, anchor, origin, predictedTaskView, candidate))
                    .ToArray();
                candidates.Add(new FovCandidateReceipt
                {
                    vertical_fov_deg = candidate,
                    same_prerun_geometry_sha256 = geometrySha256,
                    same_prerun_geometry_trace_sha256 = geometrySha256,
                    milestone_results = results,
                    all_required_events_inspectable = results.All(value => value.inside_safe_viewport && !value.occluded),
                    provenance = "derived prospectively from one immutable bind-time milestone geometry set; virtual evaluation pose is never applied to the authoritative fixed head mount"
                });
            }
            FovCandidateReceipt selected = candidates.FirstOrDefault(value => value.all_required_events_inspectable);
            string failures = selected == null
                ? string.Join(", ", candidates.Select(value =>
                    value.vertical_fov_deg.ToString("0", CultureInfo.InvariantCulture) + "deg="
                    + string.Join("/", value.milestone_results.Where(row => !row.inside_safe_viewport || row.occluded)
                        .Select(row => row.event_id + (row.occluded ? ":occluded_by_" + row.first_blocker_id : ":outside_safe_viewport")))))
                : null;
            return new FovQualificationReceipt
            {
                schema = "embodied.camera_fov_qualification.v1",
                auditioned_vertical_fov_deg = VerticalFovCandidatesDeg.ToArray(),
                selected_vertical_fov_deg = selected != null ? selected.vertical_fov_deg : -1f,
                selection_rule = "narrowest candidate with every required milestone inside a 3.5% safe viewport margin and no measured body/support/furniture ray occluder",
                same_prerun_geometry_sha256 = geometrySha256,
                same_prerun_geometry_trace_sha256 = geometrySha256,
                prerun_trace_definition = "ordered touch/capture/left-assistance/minimum-lift/turn/placement/free-release/bilateral-withdrawal milestone anchors derived once before physics execution",
                predicted_task_view_axis_world = predictedTaskAxis,
                virtual_pose_applied_to_camera = false,
                camera_mount_changed_during_audition = false,
                candidates = candidates.ToArray(),
                geometric_prediction_is_visual_pass_evidence = false,
                qualification_pass = selected != null,
                failure_reason = failures
            };
        }

        private static MilestoneViewAnchor[] BuildMilestoneViewAnchors(GateContext context)
        {
            var result = new List<MilestoneViewAnchor>();
            bool targetAvailable = context.TargetBody != null;
            Bounds targetBounds = targetAvailable ? CombinedColliderBounds(context.TargetBody.gameObject) : new Bounds();
            if (targetAvailable)
            {
                Vector3 center = targetBounds.center;
                Vector3 lateral = context.Head.right.normalized * Mathf.Max(targetBounds.extents.x, 0.035f);
                result.Add(new MilestoneViewAnchor("touch", center, "target collider bounds center at bind"));
                result.Add(new MilestoneViewAnchor("capture", center + lateral * 0.45f, "predicted right opposition side of bind-time target bounds"));
                result.Add(new MilestoneViewAnchor("left_assistance", center - lateral * 0.45f, "predicted left load-sharing side of bind-time target bounds"));
                result.Add(new MilestoneViewAnchor("lift", center + Vector3.up * 0.10f, "frozen minimum qualified lift displacement above bind-time target"));
                result.Add(new MilestoneViewAnchor("turn", center + Vector3.up * 0.10f + lateral * 0.35f, "predicted visible turned-target extent at minimum lift"));
            }

            SceneIdentity destination = UnityEngine.Object.FindObjectsByType<SceneIdentity>(FindObjectsSortMode.None)
                .Where(value => value != null)
                .FirstOrDefault(value => !string.IsNullOrWhiteSpace(context.DestinationId)
                    && (StringComparer.Ordinal.Equals(value.persistent_id, context.DestinationId)
                        || value.name.IndexOf(context.DestinationId, StringComparison.OrdinalIgnoreCase) >= 0));
            Vector3 placement;
            string placementSource;
            if (destination != null)
            {
                placement = CombinedRendererOrColliderBounds(destination.gameObject).center;
                placementSource = "SceneIdentity matching frozen destination_id";
            }
            else if (targetAvailable)
            {
                Collider support = FindInitialSupportBelow(targetBounds, context.TargetBody.gameObject);
                placement = support != null
                    ? new Vector3(targetBounds.center.x, support.bounds.max.y + targetBounds.extents.y, targetBounds.center.z)
                    : targetBounds.center;
                placementSource = support != null
                    ? "bind-time support top fallback; destination SceneIdentity unavailable"
                    : "bind-time target center fallback; destination SceneIdentity and support unavailable";
            }
            else
            {
                placement = context.RightPalm.position;
                placementSource = "contact-free measured right-hand workspace";
            }
            result.Add(new MilestoneViewAnchor("placement", placement, placementSource));
            result.Add(new MilestoneViewAnchor("free_release", placement + Vector3.up * 0.015f, placementSource + "; visible free-settle volume"));
            result.Add(new MilestoneViewAnchor("withdrawal", Vector3.Lerp(placement, context.RightPalm.position, 0.55f), "mid-sweep from placement to bind-time measured right-hand rest"));
            result.Add(new MilestoneViewAnchor("withdrawal_left", Vector3.Lerp(placement, context.LeftPalm.position, 0.55f), "mid-sweep from placement to bind-time measured left-hand rest"));
            return result.ToArray();
        }

        private static MilestoneCandidateResult EvaluateMilestoneCandidate(
            GateContext context,
            MilestoneViewAnchor anchor,
            Vector3 origin,
            Quaternion virtualRotation,
            float verticalFovDeg)
        {
            Vector3 cameraLocal = Quaternion.Inverse(virtualRotation) * (anchor.point_world_m - origin);
            float tanHalfVertical = Mathf.Tan(0.5f * verticalFovDeg * Mathf.Deg2Rad);
            float tanHalfHorizontal = tanHalfVertical * (FrozenGate.Width / (float)FrozenGate.Height);
            float x = cameraLocal.z > 1e-6f ? 0.5f + 0.5f * cameraLocal.x / (cameraLocal.z * tanHalfHorizontal) : -1f;
            float y = cameraLocal.z > 1e-6f ? 0.5f + 0.5f * cameraLocal.y / (cameraLocal.z * tanHalfVertical) : -1f;
            bool safe = cameraLocal.z >= NearClipM
                && x >= MinimumMilestoneViewportMargin && x <= 1f - MinimumMilestoneViewportMargin
                && y >= MinimumMilestoneViewportMargin && y <= 1f - MinimumMilestoneViewportMargin;
            OcclusionResult occlusion = ProspectivelyMeasureOcclusion(context, origin, anchor.point_world_m);
            return new MilestoneCandidateResult
            {
                event_id = anchor.event_id,
                point_world_m = anchor.point_world_m,
                predicted_viewport_xyz = new Vector3(x, y, cameraLocal.z),
                inside_safe_viewport = safe,
                occluded = occlusion.occluded,
                first_blocker_id = occlusion.first_blocker_id,
                anchor_provenance = anchor.provenance
            };
        }

        private static Vector3 ComputeHeadSurfaceMountWorld(GateContext context, float forwardClearanceM)
        {
            var headWeightedWorld = new List<Vector3>();
            foreach (SkinnedMeshRenderer renderer in context.AvatarRoot.GetComponentsInChildren<SkinnedMeshRenderer>(true))
            {
                int headBoneIndex = Array.IndexOf(renderer.bones, context.Head);
                Mesh shared = renderer.sharedMesh;
                if (headBoneIndex < 0 || shared == null) continue;
                BoneWeight[] weights = shared.boneWeights;
                if (weights == null || weights.Length == 0) continue;
                var baked = new Mesh();
                try
                {
                    renderer.BakeMesh(baked, true);
                    Vector3[] vertices = baked.vertices;
                    for (int index = 0; index < vertices.Length && index < weights.Length; index++)
                    {
                        BoneWeight weight = weights[index];
                        float headWeight = 0f;
                        if (weight.boneIndex0 == headBoneIndex) headWeight += weight.weight0;
                        if (weight.boneIndex1 == headBoneIndex) headWeight += weight.weight1;
                        if (weight.boneIndex2 == headBoneIndex) headWeight += weight.weight2;
                        if (weight.boneIndex3 == headBoneIndex) headWeight += weight.weight3;
                        if (headWeight >= 0.25f)
                            headWeightedWorld.Add(renderer.transform.TransformPoint(vertices[index]));
                    }
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(baked);
                }
            }

            if (headWeightedWorld.Count == 0)
                return context.Head.TransformPoint(new Vector3(0f, 0.015f, 0.12f));
            Vector3 centroid = Vector3.zero;
            foreach (Vector3 point in headWeightedWorld) centroid += point;
            centroid /= headWeightedWorld.Count;
            Vector3 faceForward = context.Head.forward.normalized;
            float forwardSurface = headWeightedWorld.Max(point => Vector3.Dot(point, faceForward));
            return centroid + faceForward * (forwardSurface - Vector3.Dot(centroid, faceForward) + forwardClearanceM);
        }

        private CameraClearanceResult ProspectivelyFreezeHeadMountClearance(GateContext context)
        {
            float[] forwardClearanceCandidatesM = { 0.032f, 0.045f, 0.060f, 0.080f };
            CameraClearanceResult last = null;
            foreach (float candidate in forwardClearanceCandidatesM)
            {
                context.HeadCameraMount.position = ComputeHeadSurfaceMountWorld(context, candidate);
                hasPreviousClearancePose = false;
                last = MeasureSweptCameraClearance(context);
                if (!last.inside_body_mesh && last.minimum_surface_distance_m >= MinimumClearanceM)
                    return last;
            }
            throw new InvalidOperationException(
                "no bounded fixed head-mount candidate has positive avatar/clothing/collider clearance; last minimum="
                + (last == null ? "unavailable" : last.minimum_surface_distance_m.ToString("R", CultureInfo.InvariantCulture))
                + "; avatar_minimum=" + (last == null ? "unavailable" : last.minimum_avatar_surface_distance_m.ToString("R", CultureInfo.InvariantCulture))
                + "; scene_minimum=" + (last == null ? "unavailable" : last.minimum_scene_collider_distance_m.ToString("R", CultureInfo.InvariantCulture))
                + "; avatar_renderer=" + (last == null ? "unavailable" : last.minimum_avatar_renderer_id)
                + "; scene_collider=" + (last == null ? "unavailable" : last.minimum_scene_collider_id)
            );
        }

        private static void ConfigureFixedExternalCamera(GateContext context)
        {
            Camera camera = context.ExternalCamera;
            if (camera == null)
                camera = new GameObject("FIXED_AT_BIND_EXTERNAL_QA_CAMERA").AddComponent<Camera>();
            camera.name = "FIXED_AT_BIND_EXTERNAL_QA_CAMERA";
            camera.transform.SetParent(context.AuthorityRoot.transform, true);
            Vector3 subject = context.TargetBody && context.TargetBody.gameObject.activeInHierarchy
                ? Vector3.Lerp(context.Head.position, context.TargetBody.worldCenterOfMass, 0.45f)
                : Vector3.Lerp(context.Head.position, context.RightPalm.position, 0.35f);
            Vector3 position = context.Head.position
                + context.Head.right * 1.55f
                + context.Head.forward * 1.75f
                + Vector3.up * 0.22f;
            camera.transform.SetPositionAndRotation(
                position,
                Quaternion.LookRotation((subject - position).normalized, Vector3.up)
            );
            camera.enabled = false;
            camera.fieldOfView = 48f;
            camera.aspect = FrozenGate.Width / (float)FrozenGate.Height;
            camera.nearClipPlane = 0.03f;
            camera.farClipPlane = FarClipM;
            camera.allowHDR = false;
            camera.allowMSAA = true;
            camera.cullingMask &= ~(1 << QaOverlayLayer);
            context.ExternalCamera = camera;
        }

        private void ConfigureRendererLabels(GateContext context)
        {
            labelBindings.Clear();
            var persistentInstanceIds = new Dictionary<string, uint>(StringComparer.Ordinal);
            var usedInstanceIds = new Dictionary<uint, string>();
            foreach (Renderer renderer in UnityEngine.Object.FindObjectsByType<Renderer>(FindObjectsSortMode.None)
                         .Where(value => value != null && value.enabled && value.gameObject.layer != QaOverlayLayer)
                         .OrderBy(value => StableTransformPath(value.transform), StringComparer.Ordinal))
            {
                SceneIdentity sceneIdentity = renderer.GetComponentInParent<SceneIdentity>(true);
                PhysicsTruthObjectIdentity identity = renderer.GetComponentInParent<PhysicsTruthObjectIdentity>(true);
                string semanticName = sceneIdentity != null && !string.IsNullOrWhiteSpace(sceneIdentity.semantic_class)
                    ? sceneIdentity.semantic_class
                    : identity != null && !string.IsNullOrWhiteSpace(identity.semantic_id)
                        ? identity.semantic_id : InferredSemanticName(renderer, context);
                string persistentName = sceneIdentity != null && !string.IsNullOrWhiteSpace(sceneIdentity.persistent_id)
                    ? sceneIdentity.persistent_id
                    : identity != null && !string.IsNullOrWhiteSpace(identity.persistent_id)
                        ? identity.persistent_id
                        : context.EpisodeId + "::" + StableTransformPath(renderer.transform);
                uint semanticId = sceneIdentity != null
                    ? CheckedUint24(sceneIdentity.semantic_id, "SceneIdentity.semantic_id")
                    : identity != null && TryUint24(identity.semantic_id, out uint truthSemantic)
                        ? truthSemantic : SemanticCode(semanticName);
                string sourceInstanceId = sceneIdentity != null ? sceneIdentity.instance_id.ToString(CultureInfo.InvariantCulture)
                    : identity != null ? identity.instance_id : "UNAVAILABLE";
                uint preferredInstanceId = sceneIdentity != null && sceneIdentity.instance_id > 0 && sceneIdentity.instance_id <= 0x00FFFFFF
                    ? (uint)sceneIdentity.instance_id
                    : identity != null && TryUint24(identity.instance_id, out uint truthInstance)
                        ? truthInstance : StableUint24("instance::" + persistentName);
                uint instanceId = ResolvePersistentInstanceId(persistentName, preferredInstanceId, persistentInstanceIds, usedInstanceIds);
                var properties = new MaterialPropertyBlock();
                renderer.GetPropertyBlock(properties);
                properties.SetFloat("_SemanticId", semanticId);
                properties.SetFloat("_InstanceId", instanceId);
                renderer.SetPropertyBlock(properties);
                labelBindings.Add(new CaptureLabelBinding
                {
                    renderer_path = StableTransformPath(renderer.transform),
                    semantic_name = semanticName,
                    semantic_uint24 = (int)semanticId,
                    persistent_instance_name = persistentName,
                    persistent_instance_uint24 = (int)instanceId,
                    source_instance_id = sourceInstanceId,
                    identity_provenance = sceneIdentity != null ? "SceneCompiler/SceneIdentity, deterministically mapped to collision-free uint24 when needed"
                        : identity != null ? "PhysicsTruthObjectIdentity exact uint24 or deterministic string mapping"
                        : "deterministic hierarchy fallback"
                });
            }
        }

        private void ConfigureQaOverlay(GateContext context)
        {
            qaOverlay = context.ExternalCamera.gameObject.GetComponent<RegisteredCaptureQaOverlay>();
            if (qaOverlay == null) qaOverlay = context.ExternalCamera.gameObject.AddComponent<RegisteredCaptureQaOverlay>();
            qaOverlay.Configure();
            qaOverlay.capture_enabled = false;
            frozenExternalPosition = context.ExternalCamera.transform.position;
            frozenExternalRotation = context.ExternalCamera.transform.rotation;
        }

        private CaptureStreamReceipt CaptureRgb(
            Camera camera,
            string path,
            string stream,
            int cullingMask,
            string authorityState,
            GateContext context)
        {
            return RenderToPng(camera, path, stream, null, 0f, cullingMask, false, authorityState, context);
        }

        private CaptureStreamReceipt CaptureReplacement(
            Camera camera,
            string path,
            string stream,
            Shader shader,
            float labelMode,
            int cullingMask,
            string authorityState,
            GateContext context,
            EventVisibilityInput[] labelSamples)
        {
            return RenderToPng(camera, path, stream, shader, labelMode, cullingMask, true, authorityState, context, labelSamples);
        }

        private CaptureStreamReceipt RenderToPng(
            Camera camera,
            string path,
            string stream,
            Shader replacement,
            float labelMode,
            int cullingMask,
            bool linear,
            string authorityState,
            GateContext context,
            EventVisibilityInput[] labelSamples = null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            cullingMask = cullingMask & ~(1 << QaOverlayLayer);
            var renderTexture = new RenderTexture(
                FrozenGate.Width,
                FrozenGate.Height,
                24,
                RenderTextureFormat.ARGB32,
                linear ? RenderTextureReadWrite.Linear : RenderTextureReadWrite.sRGB
            )
            {
                antiAliasing = 1,
                name = "RegisteredCapture_" + stream
            };
            var texture = new Texture2D(FrozenGate.Width, FrozenGate.Height, TextureFormat.RGB24, false, linear);
            RenderTexture priorActive = RenderTexture.active;
            RenderTexture priorTarget = camera.targetTexture;
            int priorMask = camera.cullingMask;
            CameraClearFlags priorClearFlags = camera.clearFlags;
            Color priorBackground = camera.backgroundColor;
            bool priorHdr = camera.allowHDR;
            bool priorMsaa = camera.allowMSAA;
            try
            {
                camera.targetTexture = renderTexture;
                camera.cullingMask = cullingMask;
                camera.allowHDR = false;
                camera.allowMSAA = false;
                if (replacement != null)
                {
                    Shader.SetGlobalFloat("_RegisteredLabelMode", labelMode);
                    camera.clearFlags = CameraClearFlags.SolidColor;
                    camera.backgroundColor = Color.black;
                    camera.SetReplacementShader(replacement, "");
                }
                camera.Render();
                RenderTexture.active = renderTexture;
                if (stream == "external_collider_contact_overlay_QA_ONLY")
                    qaOverlay.DrawNow(camera);
                texture.ReadPixels(new Rect(0, 0, FrozenGate.Width, FrozenGate.Height), 0, 0, false);
                texture.Apply(false, false);
                if (labelSamples != null && (stream == "head_semantic_uint24" || stream == "head_persistent_instance_uint24"))
                {
                    foreach (EventVisibilityInput input in labelSamples.Where(value => value.available && value.geometric_frustum_input))
                    {
                        int x = Mathf.Clamp(Mathf.FloorToInt(input.pixel_xy.x), 0, FrozenGate.Width - 1);
                        int y = Mathf.Clamp(Mathf.FloorToInt(input.pixel_xy.y), 0, FrozenGate.Height - 1);
                        Color32 pixel = texture.GetPixel(x, y);
                        int uint24 = pixel.r + 256 * pixel.g + 65536 * pixel.b;
                        input.rendered_label_pixel_xy = new Vector2(x, y);
                        if (stream == "head_semantic_uint24") input.rendered_semantic_uint24 = uint24;
                        else input.rendered_persistent_instance_uint24 = uint24;
                    }
                }
                File.WriteAllBytes(path, texture.EncodeToPNG());
            }
            finally
            {
                if (replacement != null) camera.ResetReplacementShader();
                Shader.SetGlobalFloat("_RegisteredLabelMode", 0f);
                camera.targetTexture = priorTarget;
                camera.cullingMask = priorMask;
                camera.clearFlags = priorClearFlags;
                camera.backgroundColor = priorBackground;
                camera.allowHDR = priorHdr;
                camera.allowMSAA = priorMsaa;
                RenderTexture.active = priorActive;
                renderTexture.Release();
                UnityEngine.Object.DestroyImmediate(renderTexture);
                UnityEngine.Object.DestroyImmediate(texture);
            }
            return new CaptureStreamReceipt
            {
                stream = stream,
                relative_path = RelativeTo(context.OutputRoot, path),
                file_sha256 = FileSha256(path),
                authority_state_sha256 = authorityState,
                width_px = FrozenGate.Width,
                height_px = FrozenGate.Height,
                render_frame = context.RenderFrame,
                physics_step = context.PhysicsStep,
                hero = stream == "head_rgb_hero" || stream == "external_clean",
                qa_only = stream.Contains("QA_ONLY"),
                provenance = "Unity Camera.Render from synchronous post-PhysX state"
            };
        }

        private void AssertFrozenState(
            GateContext context,
            int physicsStep,
            int renderFrame,
            string authorityState,
            PoseFingerprint headCameraPose)
        {
            bool unchanged = context.PhysicsStep == physicsStep
                && context.RenderFrame == renderFrame
                && authorityState == AuthorityStateSha256(context)
                && headCameraPose.Equals(PoseFingerprint.From(context.HeadCamera.transform));
            allModalitiesStateInvariant &= unchanged;
            if (!unchanged)
                throw new InvalidOperationException("authority/body/object/head-camera state changed between registered modalities");
        }

        private void ValidateCaptureClock(GateContext context)
        {
            bool mapped = context.PhysicsStep + 1 == (context.RenderFrame + 1) * FrozenGate.StepsPerFrame;
            mapped &= Mathf.Abs(context.TimeSeconds - (context.PhysicsStep + 1) / (float)FrozenGate.PhysicsHz) <= 1e-5f;
            exactClock &= mapped;
            if (!mapped) throw new InvalidOperationException("capture was not called on the frozen eighth post-PhysX step");
            if (framesCaptured > 0)
            {
                bool contiguous = context.RenderFrame == lastFrame + 1;
                contiguousFrames &= contiguous;
                if (!contiguous) throw new InvalidOperationException("registered render frames are not contiguous");
            }
        }

        private void AssertExternalCameraStillFixed()
        {
            if (Vector3.Distance(boundContext.ExternalCamera.transform.position, frozenExternalPosition) > 1e-7f
                || Quaternion.Angle(boundContext.ExternalCamera.transform.rotation, frozenExternalRotation) > 1e-5f)
                throw new InvalidOperationException("external QA camera moved after its one bind-time pose was frozen");
        }

        private CameraMotionSample CameraMotion(Transform cameraTransform)
        {
            var result = new CameraMotionSample();
            if (hasPreviousCameraPose)
            {
                float dt = 1f / FrozenGate.RenderHz;
                result.linear_speed_m_s = Vector3.Distance(previousCameraPosition, cameraTransform.position) / dt;
                result.angular_speed_deg_s = Quaternion.Angle(previousCameraRotation, cameraTransform.rotation) / dt;
                result.provenance = "derived finite difference of consecutive registered post-PhysX camera poses";
            }
            else
            {
                result.provenance = "unavailable_first_frame";
            }
            previousCameraPosition = cameraTransform.position;
            previousCameraRotation = cameraTransform.rotation;
            hasPreviousCameraPose = true;
            return result;
        }

        private static CameraIntrinsics Intrinsics(Camera camera)
        {
            float fy = 0.5f * FrozenGate.Height / Mathf.Tan(0.5f * camera.fieldOfView * Mathf.Deg2Rad);
            return new CameraIntrinsics
            {
                fx_px = fy,
                fy_px = fy,
                cx_px = FrozenGate.Width * 0.5f,
                cy_px = FrozenGate.Height * 0.5f,
                vertical_fov_deg = camera.fieldOfView,
                near_clip_m = camera.nearClipPlane,
                far_clip_m = camera.farClipPlane,
                skew_px = 0f,
                pixel_origin = "bottom_left_Unity_render_target"
            };
        }

        private static EventVisibilityInput[] EventVisibilityInputs(GateContext context)
        {
            var result = new List<EventVisibilityInput>();
            AddVisibilityInput(result, context.HeadCamera, "right_palm", context.RightPalm ? context.RightPalm.position : Vector3.zero, context.RightPalm, null, null);
            AddVisibilityInput(result, context.HeadCamera, "left_palm", context.LeftPalm ? context.LeftPalm.position : Vector3.zero, context.LeftPalm, null, null);
            bool targetActive = context.TargetBody && context.TargetBody.gameObject.activeInHierarchy;
            AddVisibilityInput(
                result,
                context.HeadCamera,
                "target_center_of_mass",
                targetActive ? context.TargetBody.worldCenterOfMass : Vector3.zero,
                targetActive,
                null,
                null
            );
            foreach (ContactTruth contact in context.Contacts.Where(value => value.physicsStep == context.PhysicsStep))
                AddVisibilityInput(
                    result,
                    context.HeadCamera,
                    "physx_contact_point_input",
                    contact.pointWorldM,
                    true,
                    contact.colliderA,
                    contact.colliderB
                );
            return result.ToArray();
        }

        private static void AddVisibilityInput(
            List<EventVisibilityInput> rows,
            Camera camera,
            string id,
            Vector3 world,
            bool available,
            string expectedColliderA,
            string expectedColliderB)
        {
            Vector3 viewport = available ? camera.WorldToViewportPoint(world) : new Vector3(-1f, -1f, -1f);
            ContactProjectionResult projection = available
                ? ProjectToFirstVisibleSurface(camera, world, expectedColliderA, expectedColliderB)
                : ContactProjectionResult.Unavailable();
            bool inFrustum = available && viewport.z > camera.nearClipPlane
                && viewport.x >= 0f && viewport.x <= 1f && viewport.y >= 0f && viewport.y <= 1f;
            rows.Add(new EventVisibilityInput
            {
                input_id = id,
                available = available,
                point_world_m = world,
                viewport_xyz = viewport,
                pixel_xy = new Vector2(viewport.x * FrozenGate.Width, viewport.y * FrozenGate.Height),
                geometric_frustum_input = inFrustum,
                ray_occluded = projection.occluded,
                first_visible_collider_id = projection.first_visible_collider_id,
                first_visible_renderer_path = projection.first_visible_renderer_path,
                first_visible_semantic_uint24 = 0,
                first_visible_persistent_instance_uint24 = 0,
                expected_contact_collider_a = expectedColliderA,
                expected_contact_collider_b = expectedColliderB,
                contact_projects_to_expected_visible_surface = false,
                contact_visible_in_registered_frame = false,
                contact_projection_method = "actual semantic/instance uint24 values decoded at the projected contact pixel after both frozen replacement-shader renders; Physics.RaycastAll supplies only the expected collider surface check",
                provenance = "rendered-pixel measured labels plus same-state PhysX contact and ray surface; producer booleans and renderer ancestry cannot establish label identity"
            });
        }

        private static ContactProjectionResult ProjectToFirstVisibleSurface(
            Camera camera,
            Vector3 world,
            string expectedColliderA,
            string expectedColliderB)
        {
            Vector3 origin = camera.transform.position;
            Vector3 delta = world - origin;
            float distance = delta.magnitude;
            if (distance <= 1e-7f) return ContactProjectionResult.Unavailable();
            RaycastHit[] hits = Physics.RaycastAll(origin, delta / distance, distance + 0.003f, camera.cullingMask, QueryTriggerInteraction.Ignore)
                .OrderBy(value => value.distance)
                .ToArray();
            if (hits.Length == 0)
            {
                return new ContactProjectionResult
                {
                    occluded = false,
                    expected_surface_match = string.IsNullOrEmpty(expectedColliderA) && string.IsNullOrEmpty(expectedColliderB),
                    first_visible_collider_id = "NO_RENDERED_PHYSICS_SURFACE",
                    first_visible_renderer_path = "UNAVAILABLE"
                };
            }

            Collider collider = hits[0].collider;
            string colliderId = StableTransformPath(collider.transform);
            bool expected = string.IsNullOrEmpty(expectedColliderA) && string.IsNullOrEmpty(expectedColliderB)
                || StringComparer.Ordinal.Equals(colliderId, expectedColliderA)
                || StringComparer.Ordinal.Equals(colliderId, expectedColliderB);
            return new ContactProjectionResult
            {
                occluded = !expected && (!string.IsNullOrEmpty(expectedColliderA) || !string.IsNullOrEmpty(expectedColliderB)),
                expected_surface_match = expected,
                first_visible_collider_id = colliderId,
                first_visible_renderer_path = "RESOLVED_FROM_RENDERED_LABEL_PIXEL_NOT_COLLIDER_ANCESTRY"
            };
        }

        private static void FinalizeRenderedLabelSamples(
            IEnumerable<EventVisibilityInput> inputs,
            GateContext context,
            IReadOnlyList<CaptureLabelBinding> bindings)
        {
            foreach (EventVisibilityInput input in inputs)
            {
                CaptureLabelBinding sampled = bindings.FirstOrDefault(value =>
                    value.semantic_uint24 == input.rendered_semantic_uint24
                    && value.persistent_instance_uint24 == input.rendered_persistent_instance_uint24);
                bool sampledIdentity = sampled != null && input.rendered_semantic_uint24 > 0
                    && input.rendered_persistent_instance_uint24 > 0;
                bool expectedIdentity = sampledIdentity && SampledBindingMatchesExpectedContact(
                    sampled, input.expected_contact_collider_a, input.expected_contact_collider_b, context);
                input.first_visible_semantic_uint24 = input.rendered_semantic_uint24;
                input.first_visible_persistent_instance_uint24 = input.rendered_persistent_instance_uint24;
                input.first_visible_renderer_path = sampled != null ? sampled.renderer_path : "UNRESOLVED_RENDERED_LABEL_PAIR";
                input.rendered_label_sample_complete = sampledIdentity;
                input.contact_projects_to_expected_visible_surface = expectedIdentity
                    && !input.ray_occluded && input.first_visible_collider_id != "NO_RENDERED_PHYSICS_SURFACE";
                input.contact_visible_in_registered_frame = input.geometric_frustum_input
                    && input.contact_projects_to_expected_visible_surface;
            }
        }

        private static bool SampledBindingMatchesExpectedContact(
            CaptureLabelBinding binding,
            string expectedColliderA,
            string expectedColliderB,
            GateContext context)
        {
            string expected = ((expectedColliderA ?? "") + " " + (expectedColliderB ?? "")).ToLowerInvariant();
            bool expectsHand = expected.Contains("left") || expected.Contains("right")
                || expected.Contains("thumb") || expected.Contains("index") || expected.Contains("middle")
                || expected.Contains("ring") || expected.Contains("little") || expected.Contains("pinky")
                || expected.Contains("finger") || expected.Contains("palm");
            bool weightedSkin = StringComparer.Ordinal.Equals(binding.semantic_name, "body_skin")
                && binding.renderer_path.StartsWith(StableTransformPath(context.AvatarRoot.transform), StringComparison.Ordinal);
            bool expectsTarget = !string.IsNullOrWhiteSpace(context.TargetId)
                && expected.IndexOf(context.TargetId, StringComparison.OrdinalIgnoreCase) >= 0;
            bool targetIdentity = expectsTarget
                && StringComparer.Ordinal.Equals(binding.persistent_instance_name, context.TargetId);
            return expectsHand ? weightedSkin : targetIdentity;
        }

        private static float CameraRollDeg(Transform camera, Transform head)
        {
            Vector3 forward = camera.forward.normalized;
            Vector3 referenceUp = Vector3.ProjectOnPlane(Vector3.up, forward).normalized;
            if (referenceUp.sqrMagnitude < 1e-8f)
                referenceUp = Vector3.ProjectOnPlane(head.up, forward).normalized;
            return Vector3.SignedAngle(referenceUp, Vector3.ProjectOnPlane(camera.up, forward), forward);
        }

        private CameraClearanceResult MeasureSweptCameraClearance(GateContext context)
        {
            Camera camera = context.HeadCamera;
            Vector3 currentPosition = camera.transform.position;
            Quaternion currentRotation = camera.transform.rotation;
            Vector3 startPosition = hasPreviousClearancePose ? previousClearancePosition : currentPosition;
            Quaternion startRotation = hasPreviousClearancePose ? previousClearanceRotation : currentRotation;
            var probes = new List<Vector3>();
            for (int interval = 0; interval <= SweptClearanceIntervals; interval++)
            {
                float t = interval / (float)SweptClearanceIntervals;
                Vector3 position = Vector3.Lerp(startPosition, currentPosition, t);
                Quaternion rotation = Quaternion.Slerp(startRotation, currentRotation, t);
                probes.AddRange(CameraClearanceProbes(camera, position, rotation));
            }

            float minimum = float.PositiveInfinity;
            float sceneColliderMinimum = float.PositiveInfinity;
            string minimumAvatarRendererId = "NONE";
            string minimumSceneColliderId = "NONE";
            bool insideBody = false;
            int trianglesTested = 0;
            foreach (Renderer renderer in context.AvatarRoot.GetComponentsInChildren<Renderer>(true)
                         .Where(value => value != null && value.enabled))
            {
                MeshWorldSnapshot snapshot = MeshWorldSnapshot.From(renderer);
                if (snapshot == null) continue;
                try
                {
                    trianglesTested += snapshot.triangles.Length / 3;
                    foreach (Vector3 probe in probes)
                    {
                        float distance = snapshot.DistanceToSurface(probe);
                        if (distance < minimum)
                        {
                            minimum = distance;
                            minimumAvatarRendererId = StableTransformPath(renderer.transform);
                        }
                    }
                    foreach (Vector3 probe in probes)
                        insideBody |= snapshot.ContainsByOddEvenRay(probe);
                }
                finally
                {
                    snapshot.Dispose();
                }
            }
            if (float.IsPositiveInfinity(minimum))
                throw new InvalidOperationException("camera clearance unavailable: avatar has no readable rendered triangles");

            foreach (Collider collider in UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsSortMode.None)
                         .Where(value => value != null && value.enabled && value.gameObject.activeInHierarchy
                             && !value.isTrigger
                             && !context.AvatarColliders.Contains(value)
                             && !value.transform.IsChildOf(context.AvatarRoot.transform)))
            {
                foreach (Vector3 probe in probes)
                {
                    Vector3 nearest = collider.ClosestPoint(probe);
                    float distance = Vector3.Distance(probe, nearest);
                    // A non-convex MeshCollider can return the query point even
                    // when it lies demonstrably outside the collider's world
                    // AABB. The AABB distance is a conservative lower bound on
                    // distance to every collider surface.
                    float boundsLowerBound = Mathf.Sqrt(collider.bounds.SqrDistance(probe));
                    distance = Mathf.Max(distance, boundsLowerBound);
                    if (distance < sceneColliderMinimum)
                    {
                        sceneColliderMinimum = distance;
                        minimumSceneColliderId = StableTransformPath(collider.transform);
                    }
                    if (distance <= 1e-7f) insideBody = true;
                }
            }
            float combinedMinimum = float.IsPositiveInfinity(sceneColliderMinimum)
                ? minimum : Mathf.Min(minimum, sceneColliderMinimum);
            if (float.IsPositiveInfinity(sceneColliderMinimum)) sceneColliderMinimum = -1f;
            previousClearancePosition = currentPosition;
            previousClearanceRotation = currentRotation;
            hasPreviousClearancePose = true;
            return new CameraClearanceResult
            {
                minimum_surface_distance_m = combinedMinimum,
                minimum_avatar_surface_distance_m = minimum,
                minimum_scene_collider_distance_m = sceneColliderMinimum,
                minimum_avatar_renderer_id = minimumAvatarRendererId,
                minimum_scene_collider_id = minimumSceneColliderId,
                inside_body_mesh = insideBody,
                triangles_tested = trianglesTested,
                sweep_samples = probes.Count,
                sweep_intervals = SweptClearanceIntervals,
                method = "swept bind/prior-to-current optical origin plus near-plane center/corners; exact point-to-rendered-triangle distance and odd-even containment for skin/hair/garments, plus Collider.ClosestPoint for furniture/support/objects"
            };
        }

        private static Vector3[] CameraClearanceProbes(Camera camera, Vector3 position, Quaternion rotation)
        {
            Vector3[] frustum = new Vector3[4];
            camera.CalculateFrustumCorners(
                new Rect(0f, 0f, 1f, 1f),
                camera.nearClipPlane,
                Camera.MonoOrStereoscopicEye.Mono,
                frustum
            );
            var probes = new Vector3[6];
            probes[0] = position;
            probes[1] = position + rotation * new Vector3(0f, 0f, camera.nearClipPlane);
            for (int index = 0; index < frustum.Length; index++) probes[index + 2] = position + rotation * frustum[index];
            return probes;
        }

        private static OcclusionResult ProspectivelyMeasureOcclusion(GateContext context, Vector3 origin, Vector3 endpoint)
        {
            Vector3 delta = endpoint - origin;
            float distance = delta.magnitude;
            if (distance <= 1e-6f) return new OcclusionResult(false, "NONE");
            foreach (RaycastHit hit in Physics.RaycastAll(origin, delta / distance, distance, ~0, QueryTriggerInteraction.Ignore)
                         .OrderBy(value => value.distance))
            {
                if (context.TargetBody && hit.transform.IsChildOf(context.TargetBody.transform))
                    continue; // the free target occupies each predicted manipulation endpoint
                Bounds expanded = hit.collider.bounds;
                expanded.Expand(0.006f);
                if (expanded.Contains(endpoint))
                    continue; // endpoint hand/destination geometry is not an occluder
                if (hit.distance < distance - 0.003f)
                    return new OcclusionResult(true, StableTransformPath(hit.collider.transform));
            }
            return new OcclusionResult(false, "NONE");
        }

        private static Collider FindInitialSupportBelow(Bounds targetBounds, GameObject target)
        {
            return UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsSortMode.None)
                .Where(value => value != null && value.enabled && value.gameObject.activeInHierarchy
                    && value.gameObject != target && !value.transform.IsChildOf(target.transform)
                    && value.bounds.max.y <= targetBounds.min.y + 0.01f
                    && Mathf.Abs(value.bounds.center.x - targetBounds.center.x) <= value.bounds.extents.x + targetBounds.extents.x
                    && Mathf.Abs(value.bounds.center.z - targetBounds.center.z) <= value.bounds.extents.z + targetBounds.extents.z)
                .OrderBy(value => Mathf.Abs(targetBounds.min.y - value.bounds.max.y))
                .FirstOrDefault();
        }

        private static Bounds CombinedColliderBounds(GameObject root)
        {
            Collider[] colliders = root.GetComponentsInChildren<Collider>(true);
            if (colliders.Length == 0)
                throw new InvalidOperationException("milestone object has no collider bounds: " + root.name);
            Bounds bounds = colliders[0].bounds;
            for (int index = 1; index < colliders.Length; index++) bounds.Encapsulate(colliders[index].bounds);
            return bounds;
        }

        private static Bounds CombinedRendererOrColliderBounds(GameObject root)
        {
            Renderer[] renderers = root.GetComponentsInChildren<Renderer>(true);
            if (renderers.Length > 0)
            {
                Bounds bounds = renderers[0].bounds;
                for (int index = 1; index < renderers.Length; index++) bounds.Encapsulate(renderers[index].bounds);
                return bounds;
            }
            return CombinedColliderBounds(root);
        }

        private static Renderer ClosestRenderer(Transform colliderTransform)
        {
            Renderer renderer = colliderTransform.GetComponent<Renderer>();
            if (renderer) return renderer;
            renderer = colliderTransform.GetComponentInChildren<Renderer>(true);
            if (renderer) return renderer;
            for (Transform cursor = colliderTransform.parent; cursor != null; cursor = cursor.parent)
            {
                renderer = cursor.GetComponent<Renderer>();
                if (renderer) return renderer;
                renderer = cursor.GetComponentInChildren<Renderer>(true);
                if (renderer) return renderer;
            }
            return null;
        }

        private static bool IsChildOrSelf(Transform child, Transform ancestor)
        {
            return child != null && ancestor != null && (child == ancestor || child.IsChildOf(ancestor));
        }

        private static string MilestoneGeometrySha256(
            GateContext context,
            IEnumerable<MilestoneViewAnchor> anchors,
            Vector3 origin,
            Quaternion virtualRotation)
        {
            var builder = new StringBuilder();
            builder.Append(context.EpisodeId).Append('|').Append(context.RoomFamily).Append('|');
            AppendVector(builder, origin);
            AppendQuaternion(builder, virtualRotation);
            foreach (MilestoneViewAnchor anchor in anchors.OrderBy(value => value.event_id, StringComparer.Ordinal))
            {
                builder.Append(anchor.event_id).Append('|').Append(anchor.provenance).Append('|');
                AppendVector(builder, anchor.point_world_m);
            }
            using (SHA256 sha = SHA256.Create())
                return Hex(sha.ComputeHash(Encoding.UTF8.GetBytes(builder.ToString())));
        }

        private static string AuthorityStateSha256(GateContext context)
        {
            var transforms = new Dictionary<int, Transform>();
            foreach (GameObject root in new[] { context.AuthorityRoot, context.AvatarRoot, context.TargetBody ? context.TargetBody.gameObject : null })
            {
                if (root == null) continue;
                foreach (Transform transform in root.GetComponentsInChildren<Transform>(true))
                    transforms[transform.GetInstanceID()] = transform;
            }
            var builder = new StringBuilder();
            foreach (Transform transform in transforms.Values.OrderBy(StableTransformPath, StringComparer.Ordinal))
            {
                builder.Append(StableTransformPath(transform)).Append('|');
                AppendVector(builder, transform.position);
                AppendQuaternion(builder, transform.rotation);
                builder.Append(transform.gameObject.activeSelf ? '1' : '0').Append(';');
            }
            foreach (Rigidbody body in transforms.Values.Select(value => value.GetComponent<Rigidbody>())
                         .Where(value => value != null)
                         .Distinct()
                         .OrderBy(value => StableTransformPath(value.transform), StringComparer.Ordinal))
            {
                builder.Append("RB|").Append(StableTransformPath(body.transform)).Append('|');
                AppendVector(builder, body.linearVelocity);
                AppendVector(builder, body.angularVelocity);
                builder.Append(body.IsSleeping() ? '1' : '0').Append(';');
            }
            using (SHA256 sha = SHA256.Create())
                return Hex(sha.ComputeHash(Encoding.UTF8.GetBytes(builder.ToString())));
        }

        private static void AppendVector(StringBuilder builder, Vector3 value)
        {
            AppendFloat(builder, value.x); AppendFloat(builder, value.y); AppendFloat(builder, value.z);
        }

        private static void AppendQuaternion(StringBuilder builder, Quaternion value)
        {
            AppendFloat(builder, value.x); AppendFloat(builder, value.y); AppendFloat(builder, value.z); AppendFloat(builder, value.w);
        }

        private static void AppendFloat(StringBuilder builder, float value)
        {
            builder.Append(value.ToString("R", CultureInfo.InvariantCulture)).Append(',');
        }

        private static string ReadCaptureMode()
        {
            string value = Environment.GetEnvironmentVariable("PROCEDURAL_GATE_CAPTURE_MODE") ?? "all";
            if (value != "all" && value != "qualification" && value != "none")
                throw new InvalidOperationException("PROCEDURAL_GATE_CAPTURE_MODE must be all, qualification, or none");
            return value;
        }

        private static void CreateOutputDirectories(string root, bool allModalities)
        {
            Directory.CreateDirectory(Path.Combine(root, "head", "rgb"));
            Directory.CreateDirectory(Path.Combine(root, "external", "clean"));
            Directory.CreateDirectory(Path.Combine(root, "external", "overlay"));
            if (!allModalities) return;
            Directory.CreateDirectory(Path.Combine(root, "head", "depth"));
            Directory.CreateDirectory(Path.Combine(root, "head", "semantic"));
            Directory.CreateDirectory(Path.Combine(root, "head", "instance"));
        }

        private static string[] StreamRoots(string mode)
        {
            if (mode == "none") return Array.Empty<string>();
            var roots = new List<string> { "head/rgb", "external/clean", "external/overlay" };
            if (mode == "all") roots.InsertRange(1, new[] { "head/depth", "head/semantic", "head/instance" });
            return roots.ToArray();
        }

        private void WriteLabelManifest(GateContext context)
        {
            var manifest = new CaptureLabelManifest
            {
                schema = "embodied.semantic_instance_manifest.v1",
                episode_id = context.EpisodeId,
                background_uint24 = 0,
                encoding = "little-endian RGB uint24: id=R+256*G+65536*B",
                persistent_across_frames = true,
                labels_assigned_before_first_physics_step = true,
                bindings = labelBindings.ToArray()
            };
            File.WriteAllText(
                Path.Combine(context.OutputRoot, LabelManifestFileName),
                JsonUtility.ToJson(manifest, true) + "\n",
                new UTF8Encoding(false)
            );
        }

        private void RequireActiveContext(GateContext context)
        {
            if (boundContext == null || !ReferenceEquals(context, boundContext))
                throw new InvalidOperationException("RegisteredCapture called with a context other than its bound authority context");
            if (completed) throw new InvalidOperationException("RegisteredCapture is already complete");
        }

        private static string InferredSemanticName(Renderer renderer, GateContext context)
        {
            string name = renderer.name.ToLowerInvariant();
            if (context.TargetBody && renderer.transform.IsChildOf(context.TargetBody.transform)) return "interactive_target";
            if (renderer.transform.IsChildOf(context.AvatarRoot.transform))
            {
                string[] garmentTokens = { "garment", "shirt", "top", "sleeve", "vest", "trouser", "overall", "sock", "cloth" };
                return garmentTokens.Any(name.Contains) ? "garment" : "body_skin";
            }
            if (name.Contains("table")) return "support_table";
            if (name.Contains("sofa")) return "sofa";
            if (name.Contains("chair")) return "chair";
            if (name.Contains("book")) return "books_or_storage";
            if (name.Contains("rug")) return "rug";
            if (name.Contains("lamp")) return "lamp";
            if (name.Contains("plant")) return "plant";
            return "scene_static";
        }

        private static uint SemanticCode(string name)
        {
            switch (name)
            {
                case "body_skin": return 100;
                case "garment": return 101;
                case "interactive_target": return 41;
                case "support_table": return 20;
                case "sofa": return 21;
                case "chair": return 23;
                case "books_or_storage": return 22;
                case "rug": return 24;
                case "lamp": return 25;
                case "plant": return 26;
                case "scene_static": return 10;
                default: return StableUint24("semantic::" + name);
            }
        }

        private static uint StableUint24(string value)
        {
            using (SHA256 sha = SHA256.Create())
            {
                byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(value ?? "UNAVAILABLE"));
                uint result = (uint)(digest[0] | digest[1] << 8 | digest[2] << 16) & 0x00FFFFFFu;
                return result == 0u ? 1u : result;
            }
        }

        private static bool TryUint24(string value, out uint result)
        {
            return uint.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out result)
                && result > 0u && result <= 0x00FFFFFFu;
        }

        private static uint CheckedUint24(int value, string field)
        {
            if (value <= 0 || value > 0x00FFFFFF)
                throw new InvalidOperationException(field + " must be a non-background uint24 value");
            return (uint)value;
        }

        private static uint ResolvePersistentInstanceId(
            string persistentName,
            uint preferred,
            Dictionary<string, uint> assigned,
            Dictionary<uint, string> used)
        {
            if (assigned.TryGetValue(persistentName, out uint existing)) return existing;
            uint candidate = preferred == 0u ? StableUint24("instance::" + persistentName) : preferred;
            while (used.TryGetValue(candidate, out string owner) && owner != persistentName)
            {
                candidate = candidate == 0x00FFFFFFu ? 1u : candidate + 1u;
            }
            assigned[persistentName] = candidate;
            used[candidate] = persistentName;
            return candidate;
        }

        private static string StableTransformPath(Transform transform)
        {
            if (transform == null) return "UNAVAILABLE";
            var parts = new Stack<string>();
            for (Transform current = transform; current != null; current = current.parent)
                parts.Push(current.name + "[" + current.GetSiblingIndex().ToString(CultureInfo.InvariantCulture) + "]");
            return string.Join("/", parts.ToArray());
        }

        private static float[] MatrixValues(Matrix4x4 matrix)
        {
            var values = new float[16];
            for (int index = 0; index < 16; index++) values[index] = matrix[index];
            return values;
        }

        private static string RelativeTo(string root, string path)
        {
            Uri rootUri = new Uri(Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar);
            return Uri.UnescapeDataString(rootUri.MakeRelativeUri(new Uri(Path.GetFullPath(path))).ToString()).Replace('/', Path.DirectorySeparatorChar);
        }

        private static string FileSha256(string path)
        {
            using (SHA256 sha = SHA256.Create())
            using (FileStream stream = File.OpenRead(path))
                return Hex(sha.ComputeHash(stream));
        }

        private static string Hex(byte[] bytes)
        {
            var builder = new StringBuilder(bytes.Length * 2);
            foreach (byte value in bytes) builder.Append(value.ToString("x2", CultureInfo.InvariantCulture));
            return builder.ToString();
        }

        private static float MinOrUnavailable(List<float> values) => values.Count == 0 ? -1f : values.Min();
        private static float MaxOrUnavailable(List<float> values) => values.Count == 0 ? -1f : values.Max();

        private struct PoseFingerprint : IEquatable<PoseFingerprint>
        {
            public Vector3 position;
            public Quaternion rotation;
            public Vector3 localPosition;
            public Quaternion localRotation;

            public static PoseFingerprint From(Transform transform) => new PoseFingerprint
            {
                position = transform.position,
                rotation = transform.rotation,
                localPosition = transform.localPosition,
                localRotation = transform.localRotation
            };

            public bool Equals(PoseFingerprint other)
            {
                return position == other.position && rotation == other.rotation
                    && localPosition == other.localPosition && localRotation == other.localRotation;
            }
        }

        private readonly struct MilestoneViewAnchor
        {
            public readonly string event_id;
            public readonly Vector3 point_world_m;
            public readonly string provenance;

            public MilestoneViewAnchor(string eventId, Vector3 pointWorldM, string source)
            {
                event_id = eventId;
                point_world_m = pointWorldM;
                provenance = source;
            }
        }

        private readonly struct OcclusionResult
        {
            public readonly bool occluded;
            public readonly string first_blocker_id;

            public OcclusionResult(bool isOccluded, string blocker)
            {
                occluded = isOccluded;
                first_blocker_id = blocker;
            }
        }

        private struct ContactProjectionResult
        {
            public bool occluded;
            public bool expected_surface_match;
            public string first_visible_collider_id;
            public string first_visible_renderer_path;
            public int semantic_uint24;
            public int persistent_instance_uint24;

            public static ContactProjectionResult Unavailable()
            {
                return new ContactProjectionResult
                {
                    occluded = false,
                    expected_surface_match = false,
                    first_visible_collider_id = "UNAVAILABLE",
                    first_visible_renderer_path = "UNAVAILABLE"
                };
            }
        }

        private sealed class MeshWorldSnapshot : IDisposable
        {
            public readonly Vector3[] vertices;
            public readonly int[] triangles;
            private readonly Mesh ownedMesh;

            private MeshWorldSnapshot(Vector3[] vertices, int[] triangles, Mesh ownedMesh)
            {
                this.vertices = vertices;
                this.triangles = triangles;
                this.ownedMesh = ownedMesh;
            }

            public static MeshWorldSnapshot From(Renderer renderer)
            {
                Mesh mesh = null;
                Mesh owned = null;
                if (renderer is SkinnedMeshRenderer skinned)
                {
                    owned = new Mesh();
                    skinned.BakeMesh(owned, true);
                    mesh = owned;
                }
                else
                {
                    MeshFilter filter = renderer.GetComponent<MeshFilter>();
                    if (filter != null) mesh = filter.sharedMesh;
                }
                if (mesh == null)
                {
                    if (owned != null) UnityEngine.Object.DestroyImmediate(owned);
                    return null;
                }
                try
                {
                    Vector3[] local = mesh.vertices;
                    int[] indices = mesh.triangles;
                    var world = new Vector3[local.Length];
                    for (int index = 0; index < local.Length; index++)
                        world[index] = renderer.transform.TransformPoint(local[index]);
                    return new MeshWorldSnapshot(world, indices, owned);
                }
                catch
                {
                    if (owned != null) UnityEngine.Object.DestroyImmediate(owned);
                    return null;
                }
            }

            public float DistanceToSurface(Vector3 point)
            {
                float minimumSquared = float.PositiveInfinity;
                for (int index = 0; index + 2 < triangles.Length; index += 3)
                {
                    Vector3 a = vertices[triangles[index]];
                    Vector3 b = vertices[triangles[index + 1]];
                    Vector3 c = vertices[triangles[index + 2]];
                    minimumSquared = Mathf.Min(minimumSquared, PointTriangleSquaredDistance(point, a, b, c));
                }
                return Mathf.Sqrt(minimumSquared);
            }

            public bool ContainsByOddEvenRay(Vector3 point)
            {
                Vector3 direction = new Vector3(0.937f, 0.277f, 0.211f).normalized;
                int intersections = 0;
                for (int index = 0; index + 2 < triangles.Length; index += 3)
                    if (RayTriangle(point, direction, vertices[triangles[index]], vertices[triangles[index + 1]], vertices[triangles[index + 2]]))
                        intersections++;
                return (intersections & 1) == 1;
            }

            public void Dispose()
            {
                if (ownedMesh != null) UnityEngine.Object.DestroyImmediate(ownedMesh);
            }

            private static bool RayTriangle(Vector3 origin, Vector3 direction, Vector3 a, Vector3 b, Vector3 c)
            {
                Vector3 edge1 = b - a;
                Vector3 edge2 = c - a;
                Vector3 p = Vector3.Cross(direction, edge2);
                float determinant = Vector3.Dot(edge1, p);
                if (Mathf.Abs(determinant) < 1e-9f) return false;
                float inverse = 1f / determinant;
                Vector3 t = origin - a;
                float u = Vector3.Dot(t, p) * inverse;
                if (u < 0f || u > 1f) return false;
                Vector3 q = Vector3.Cross(t, edge1);
                float v = Vector3.Dot(direction, q) * inverse;
                if (v < 0f || u + v > 1f) return false;
                return Vector3.Dot(edge2, q) * inverse > 1e-6f;
            }

            private static float PointTriangleSquaredDistance(Vector3 point, Vector3 a, Vector3 b, Vector3 c)
            {
                Vector3 ab = b - a;
                Vector3 ac = c - a;
                Vector3 ap = point - a;
                float d1 = Vector3.Dot(ab, ap);
                float d2 = Vector3.Dot(ac, ap);
                if (d1 <= 0f && d2 <= 0f) return ap.sqrMagnitude;
                Vector3 bp = point - b;
                float d3 = Vector3.Dot(ab, bp);
                float d4 = Vector3.Dot(ac, bp);
                if (d3 >= 0f && d4 <= d3) return bp.sqrMagnitude;
                float vc = d1 * d4 - d3 * d2;
                if (vc <= 0f && d1 >= 0f && d3 <= 0f)
                {
                    float v = d1 / (d1 - d3);
                    return (point - (a + v * ab)).sqrMagnitude;
                }
                Vector3 cp = point - c;
                float d5 = Vector3.Dot(ab, cp);
                float d6 = Vector3.Dot(ac, cp);
                if (d6 >= 0f && d5 <= d6) return cp.sqrMagnitude;
                float vb = d5 * d2 - d1 * d6;
                if (vb <= 0f && d2 >= 0f && d6 <= 0f)
                {
                    float w = d2 / (d2 - d6);
                    return (point - (a + w * ac)).sqrMagnitude;
                }
                float va = d3 * d6 - d5 * d4;
                if (va <= 0f && d4 - d3 >= 0f && d5 - d6 >= 0f)
                {
                    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                    return (point - (b + w * (c - b))).sqrMagnitude;
                }
                float denominator = 1f / (va + vb + vc);
                float insideV = vb * denominator;
                float insideW = vc * denominator;
                return (point - (a + ab * insideV + ac * insideW)).sqrMagnitude;
            }
        }
    }

    /// <summary>
    /// Built-in-pipeline post-render overlay.  It is enabled for external/overlay
    /// only. Green boxes are Collider.bounds; red marks are shared measured-contact
    /// inputs; amber marks are explicitly labeled ComputePenetration QA proxies.
    /// </summary>
    public sealed class RegisteredCaptureQaOverlay : MonoBehaviour
    {
        public bool capture_enabled;
        public int true_contact_count;
        public int qa_penetration_proxy_count;
        private Material lineMaterial;
        private Bounds[] colliderBounds = Array.Empty<Bounds>();
        private OverlayContact[] contacts = Array.Empty<OverlayContact>();

        public void Configure()
        {
            Shader shader = Shader.Find("Hidden/Internal-Colored");
            if (shader == null) throw new InvalidOperationException("Built-in QA line shader is unavailable");
            lineMaterial = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
            lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
            lineMaterial.SetInt("_ZWrite", 0);
        }

        public void PrepareFrame(GateContext context)
        {
            colliderBounds = UnityEngine.Object.FindObjectsByType<Collider>(FindObjectsSortMode.None)
                .Where(value => value != null && value.enabled && value.gameObject.activeInHierarchy)
                .Select(value => value.bounds)
                .ToArray();
            var rows = context.Contacts
                .Where(value => value.physicsStep == context.PhysicsStep)
                .Select(value => new OverlayContact
                {
                    point = value.pointWorldM,
                    normal = value.normalWorld,
                    color = new Color(1f, 0.1f, 0.1f, 0.95f),
                    provenance = "shared PhysX-measured ContactTruth input"
                })
                .ToList();
            true_contact_count = rows.Count;

            if (context.TargetBody && context.TargetBody.gameObject.activeInHierarchy)
            {
                Collider[] target = context.TargetBody.GetComponentsInChildren<Collider>(true);
                Collider[] avatar = context.AvatarRoot.GetComponentsInChildren<Collider>(true);
                foreach (Collider first in avatar.Where(value => value.enabled))
                foreach (Collider second in target.Where(value => value.enabled))
                {
                    if (!Physics.ComputePenetration(
                            first, first.transform.position, first.transform.rotation,
                            second, second.transform.position, second.transform.rotation,
                            out Vector3 direction, out float distance)) continue;
                    Vector3 firstPoint = first.ClosestPoint(second.bounds.center);
                    Vector3 secondPoint = second.ClosestPoint(first.bounds.center);
                    rows.Add(new OverlayContact
                    {
                        point = 0.5f * (firstPoint + secondPoint),
                        normal = direction.normalized * Mathf.Max(distance, 0.01f),
                        color = new Color(1f, 0.58f, 0.05f, 0.95f),
                        provenance = "QA-only Physics.ComputePenetration overlap proxy; not contact truth"
                    });
                }
            }
            qa_penetration_proxy_count = rows.Count - true_contact_count;
            contacts = rows.ToArray();
        }

        public void DrawNow(Camera camera)
        {
            if (!capture_enabled || lineMaterial == null) return;
            lineMaterial.SetPass(0);
            GL.PushMatrix();
            GL.LoadProjectionMatrix(camera.projectionMatrix);
            GL.modelview = camera.worldToCameraMatrix;
            GL.Begin(GL.LINES);
            GL.Color(new Color(0.1f, 1f, 0.3f, 0.72f));
            foreach (Bounds bounds in colliderBounds) DrawBounds(bounds);
            foreach (OverlayContact contact in contacts)
            {
                GL.Color(contact.color);
                DrawCross(contact.point, 0.008f);
                GL.Vertex(contact.point);
                GL.Vertex(contact.point + contact.normal * 0.04f);
            }
            GL.End();
            GL.PopMatrix();
        }

        public void DisposeMaterial()
        {
            if (lineMaterial != null) DestroyImmediate(lineMaterial);
            lineMaterial = null;
        }

        private static void DrawCross(Vector3 point, float radius)
        {
            GL.Vertex(point - Vector3.right * radius); GL.Vertex(point + Vector3.right * radius);
            GL.Vertex(point - Vector3.up * radius); GL.Vertex(point + Vector3.up * radius);
            GL.Vertex(point - Vector3.forward * radius); GL.Vertex(point + Vector3.forward * radius);
        }

        private static void DrawBounds(Bounds bounds)
        {
            Vector3 min = bounds.min;
            Vector3 max = bounds.max;
            Vector3[] p =
            {
                new Vector3(min.x, min.y, min.z), new Vector3(max.x, min.y, min.z),
                new Vector3(max.x, max.y, min.z), new Vector3(min.x, max.y, min.z),
                new Vector3(min.x, min.y, max.z), new Vector3(max.x, min.y, max.z),
                new Vector3(max.x, max.y, max.z), new Vector3(min.x, max.y, max.z)
            };
            int[] edges = { 0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7 };
            for (int index = 0; index < edges.Length; index += 2)
            {
                GL.Vertex(p[edges[index]]);
                GL.Vertex(p[edges[index + 1]]);
            }
        }

        private struct OverlayContact
        {
            public Vector3 point;
            public Vector3 normal;
            public Color color;
            public string provenance;
        }
    }

    [Serializable] public sealed class CaptureStreamReceipt
    {
        public string stream;
        public string relative_path;
        public string file_sha256;
        public string authority_state_sha256;
        public int width_px;
        public int height_px;
        public int render_frame;
        public int physics_step;
        public bool hero;
        public bool qa_only;
        public string provenance;
    }

    [Serializable] public sealed class CameraIntrinsics
    {
        public float fx_px;
        public float fy_px;
        public float cx_px;
        public float cy_px;
        public float skew_px;
        public float vertical_fov_deg;
        public float near_clip_m;
        public float far_clip_m;
        public string pixel_origin;
    }

    [Serializable] public sealed class EventVisibilityInput
    {
        public string input_id;
        public bool available;
        public Vector3 point_world_m;
        public Vector3 viewport_xyz;
        public Vector2 pixel_xy;
        public Vector2 rendered_label_pixel_xy;
        public int rendered_semantic_uint24;
        public int rendered_persistent_instance_uint24;
        public bool rendered_label_sample_complete;
        public bool geometric_frustum_input;
        public bool ray_occluded;
        public string first_visible_collider_id;
        public string first_visible_renderer_path;
        public int first_visible_semantic_uint24;
        public int first_visible_persistent_instance_uint24;
        public string expected_contact_collider_a;
        public string expected_contact_collider_b;
        public bool contact_projects_to_expected_visible_surface;
        public bool contact_visible_in_registered_frame;
        public string contact_projection_method;
        public string provenance;
    }

    [Serializable] public sealed class CaptureFrameLedgerRow
    {
        public string schema;
        public string episode_id;
        public string stage;
        public string capture_mode;
        public int render_frame;
        public int physics_step;
        public float time_s;
        public string sample_phase;
        public int physics_hz;
        public int render_hz;
        public int steps_per_render_frame;
        public int width_px;
        public int height_px;
        public string authority_state_sha256;
        public bool authority_state_unchanged_across_modalities;
        public bool physics_advanced_between_modalities;
        public CaptureStreamReceipt[] streams;
        public string camera_parent_id;
        public Vector3 camera_position_world_m;
        public Quaternion camera_rotation_world_xyzw;
        public float[] camera_to_world_matrix_column_major;
        public float[] world_to_camera_matrix_column_major;
        public float[] projection_matrix_column_major;
        public CameraIntrinsics intrinsics;
        public string camera_state_provenance;
        public string intrinsics_provenance;
        public float clearance_m;
        public bool clearance_inside_body_mesh;
        public float swept_clearance_m;
        public int swept_clearance_samples;
        public float swept_clearance_scene_collider_m;
        public string clearance_method;
        public float optical_vs_face_forward_deg;
        public float roll_deg;
        public float linear_speed_m_s;
        public float angular_speed_deg_s;
        public bool optical_within_tolerance;
        public bool clearance_within_tolerance;
        public bool roll_within_tolerance;
        public bool linear_speed_within_tolerance;
        public bool angular_speed_within_tolerance;
        public EventVisibilityInput[] event_visibility_inputs;
        public string event_visibility_interpretation;
        public int overlay_true_context_contact_count;
        public int overlay_qa_penetration_proxy_count;
        public bool overlay_is_hero;
        public bool overlay_contains_labeled_qa_proxies;
    }

    [Serializable] public sealed class RegisteredContactProjectionRecord
    {
        public int render_frame;
        public int physics_step;
        public Vector3 physical_contact_point_world_m;
        public string expected_contact_collider_a;
        public string expected_contact_collider_b;
        public Vector2 pixel_xy;
        public string first_visible_collider_id;
        public string first_visible_renderer_path;
        public int first_visible_semantic_uint24;
        public int first_visible_persistent_instance_uint24;
        public bool contact_projects_to_expected_visible_surface;
        public bool contact_visible_in_registered_frame;
        public string method;
        public string provenance;
    }

    [Serializable] public sealed class RegisteredContactProjectionReport
    {
        public string schema;
        public string episode_id;
        public string source_capture_ledger;
        public string source_capture_ledger_sha256;
        public RegisteredContactProjectionRecord[] records;
        public string provenance;
    }

    [Serializable] public sealed class RegisteredVisualMeasurements
    {
        public string schema;
        public string episode_id;
        public string source_capture_ledger;
        public string source_capture_ledger_sha256;
        public string registered_contact_projection;
        public string registered_contact_projection_sha256;
        public bool measured_capture_evidence_only;
        public bool direct_decoded_frame_review_performed;
        public int registered_projection_record_count;
        public string measurement_scope;
        public string provenance;
    }

    [Serializable] public sealed class CaptureLabelBinding
    {
        public string renderer_path;
        public string semantic_name;
        public int semantic_uint24;
        public string persistent_instance_name;
        public int persistent_instance_uint24;
        public string source_instance_id;
        public string identity_provenance;
    }

    [Serializable] public sealed class CaptureLabelManifest
    {
        public string schema;
        public string episode_id;
        public int background_uint24;
        public string encoding;
        public bool persistent_across_frames;
        public bool labels_assigned_before_first_physics_step;
        public CaptureLabelBinding[] bindings;
    }

    [Serializable] public sealed class RegisteredCaptureManifest
    {
        public string schema;
        public string episode_id;
        public string stage;
        public string capture_mode;
        public string unity_version;
        public string rendering_pipeline;
        public int width_px;
        public int height_px;
        public int fps;
        public int physics_hz;
        public int steps_per_render_frame;
        public int frames_captured;
        public int first_frame;
        public int last_frame;
        public int expected_full_episode_frames;
        public bool contiguous_frames;
        public bool exact_integer_clock;
        public bool all_modalities_state_invariant;
        public bool physics_advanced_between_modalities;
        public string head_mount_parent;
        public Vector3 head_mount_local_position_m;
        public Quaternion head_mount_local_rotation_xyzw;
        public bool head_mount_fixed_after_bind;
        public bool target_lock;
        public bool dependent_mount;
        public bool independent_camera_animation;
        public bool render_only_camera_keyframes;
        public string tabletop_view_source;
        public string rgb_encoding;
        public string metric_depth_encoding;
        public string semantic_encoding;
        public string instance_encoding;
        public string label_manifest;
        public string frame_ledger;
        public string[] stream_roots;
        public FovQualificationReceipt fov_qualification;
        public string fov_qualification_file;
        public string fov_qualification_sha256;
        public float selected_vertical_fov_deg;
        public string fov_freeze_rule;
        public float minimum_clearance_m;
        public float maximum_optical_vs_face_forward_deg;
        public float maximum_abs_roll_deg;
        public float maximum_linear_speed_m_s;
        public float maximum_angular_speed_deg_s;
        public string visual_evidence_rule;
        public bool hero_contains_proxy_pixels;
        public bool external_overlay_is_separate_qa_only;
        public string provenance;
    }

    [Serializable] public sealed class CameraClearanceResult
    {
        public float minimum_surface_distance_m;
        public float minimum_avatar_surface_distance_m;
        public float minimum_scene_collider_distance_m;
        public string minimum_avatar_renderer_id;
        public string minimum_scene_collider_id;
        public bool inside_body_mesh;
        public int triangles_tested;
        public int sweep_samples;
        public int sweep_intervals;
        public string method;
    }

    [Serializable] public sealed class FovQualificationReceipt
    {
        public string schema;
        public float[] auditioned_vertical_fov_deg;
        public float selected_vertical_fov_deg;
        public string selection_rule;
        public string same_prerun_geometry_sha256;
        public string same_prerun_geometry_trace_sha256;
        public string prerun_trace_definition;
        public Vector3 predicted_task_view_axis_world;
        public bool virtual_pose_applied_to_camera;
        public bool camera_mount_changed_during_audition;
        public FovCandidateReceipt[] candidates;
        public bool geometric_prediction_is_visual_pass_evidence;
        public bool qualification_pass;
        public string failure_reason;
    }

    [Serializable] public sealed class FovCandidateReceipt
    {
        public float vertical_fov_deg;
        public string same_prerun_geometry_sha256;
        public string same_prerun_geometry_trace_sha256;
        public bool all_required_events_inspectable;
        public MilestoneCandidateResult[] milestone_results;
        public string provenance;
    }

    [Serializable] public sealed class MilestoneCandidateResult
    {
        public string event_id;
        public Vector3 point_world_m;
        public Vector3 predicted_viewport_xyz;
        public bool inside_safe_viewport;
        public bool occluded;
        public string first_blocker_id;
        public string anchor_provenance;
    }

    [Serializable] public sealed class CameraMotionSample
    {
        public float linear_speed_m_s;
        public float angular_speed_deg_s;
        public string provenance;
    }
}
#endif
