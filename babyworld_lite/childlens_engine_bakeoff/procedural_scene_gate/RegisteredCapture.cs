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

        private const float VerticalFovDeg = 68f;
        private const float NearClipM = 0.01f;
        private const float FarClipM = 20f;
        private const float MinimumClearanceM = 0.001f;
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

            CameraClearanceResult bindClearance = MeasureCameraClearance(context);
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
                    context
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
                    context
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
                    context
                ));
                AssertFrozenState(context, physicsStepAtEntry, renderFrameAtEntry, authorityState, headCameraPose);
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

            CameraClearanceResult clearance = MeasureCameraClearance(context);
            CameraMotionSample motion = CameraMotion(context.HeadCamera.transform);
            float opticalVsFaceForward = Vector3.Angle(context.Head.forward, context.HeadCamera.transform.forward);
            float roll = CameraRollDeg(context.HeadCamera.transform, context.Head);
            CameraIntrinsics intrinsics = Intrinsics(context.HeadCamera);
            EventVisibilityInput[] visibility = EventVisibilityInputs(context);

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

            qaOverlay?.DisposeMaterial();
            completed = true;
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
            camera.fieldOfView = VerticalFovDeg;
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
            GateContext context)
        {
            return RenderToPng(camera, path, stream, shader, labelMode, cullingMask, true, authorityState, context);
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
            GateContext context)
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
            AddVisibilityInput(result, context.HeadCamera, "right_palm", context.RightPalm ? context.RightPalm.position : Vector3.zero, context.RightPalm);
            AddVisibilityInput(result, context.HeadCamera, "left_palm", context.LeftPalm ? context.LeftPalm.position : Vector3.zero, context.LeftPalm);
            bool targetActive = context.TargetBody && context.TargetBody.gameObject.activeInHierarchy;
            AddVisibilityInput(
                result,
                context.HeadCamera,
                "target_center_of_mass",
                targetActive ? context.TargetBody.worldCenterOfMass : Vector3.zero,
                targetActive
            );
            foreach (ContactTruth contact in context.Contacts.Where(value => value.physicsStep == context.PhysicsStep))
                AddVisibilityInput(result, context.HeadCamera, "physx_contact_point_input", contact.pointWorldM, true);
            return result.ToArray();
        }

        private static void AddVisibilityInput(List<EventVisibilityInput> rows, Camera camera, string id, Vector3 world, bool available)
        {
            Vector3 viewport = available ? camera.WorldToViewportPoint(world) : new Vector3(float.NaN, float.NaN, float.NaN);
            rows.Add(new EventVisibilityInput
            {
                input_id = id,
                available = available,
                point_world_m = world,
                viewport_xyz = viewport,
                geometric_frustum_input = available && viewport.z > camera.nearClipPlane
                    && viewport.x >= 0f && viewport.x <= 1f && viewport.y >= 0f && viewport.y <= 1f,
                provenance = "derived projection input; occlusion, anatomy, clipping, and visual coherence require decoded-frame review"
            });
        }

        private static float CameraRollDeg(Transform camera, Transform head)
        {
            Vector3 forward = camera.forward.normalized;
            Vector3 referenceUp = Vector3.ProjectOnPlane(Vector3.up, forward).normalized;
            if (referenceUp.sqrMagnitude < 1e-8f)
                referenceUp = Vector3.ProjectOnPlane(head.up, forward).normalized;
            return Vector3.SignedAngle(referenceUp, Vector3.ProjectOnPlane(camera.up, forward), forward);
        }

        private static CameraClearanceResult MeasureCameraClearance(GateContext context)
        {
            Vector3[] frustum = new Vector3[4];
            context.HeadCamera.CalculateFrustumCorners(
                new Rect(0f, 0f, 1f, 1f),
                context.HeadCamera.nearClipPlane,
                Camera.MonoOrStereoscopicEye.Mono,
                frustum
            );
            var probes = new List<Vector3>
            {
                context.HeadCamera.transform.position,
                context.HeadCamera.transform.TransformPoint(new Vector3(0f, 0f, context.HeadCamera.nearClipPlane))
            };
            probes.AddRange(frustum.Select(context.HeadCamera.transform.TransformPoint));

            float minimum = float.PositiveInfinity;
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
                        minimum = Mathf.Min(minimum, snapshot.DistanceToSurface(probe));
                    if (InferredSemanticName(renderer, context) == "body_skin")
                        insideBody |= snapshot.ContainsByOddEvenRay(probes[0]);
                }
                finally
                {
                    snapshot.Dispose();
                }
            }
            if (float.IsPositiveInfinity(minimum))
                throw new InvalidOperationException("camera clearance unavailable: avatar has no readable rendered triangles");
            return new CameraClearanceResult
            {
                minimum_surface_distance_m = minimum,
                inside_body_mesh = insideBody,
                triangles_tested = trianglesTested,
                method = "minimum exact point-to-rendered-triangle distance from optical origin and near-plane center/corners; odd-even ray rejects origin inside weighted body"
            };
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
        public bool geometric_frustum_input;
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
        public bool inside_body_mesh;
        public int triangles_tested;
        public string method;
    }

    [Serializable] public sealed class CameraMotionSample
    {
        public float linear_speed_m_s;
        public float angular_speed_deg_s;
        public string provenance;
    }
}
#endif
