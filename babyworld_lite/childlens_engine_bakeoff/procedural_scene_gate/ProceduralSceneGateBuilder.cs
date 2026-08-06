#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace ProceduralSceneGate
{
    public static class ProceduralSceneGateBuilder
    {
        private const float Dt = 1f / FrozenGate.PhysicsHz;
        private const int TotalSteps = (int)(FrozenGate.DurationSeconds * FrozenGate.PhysicsHz);

        [MenuItem("BabyWorld/Run Procedural Clothed Scene Gate")]
        public static void Run()
        {
            string output = RequiredEnvironment("PROCEDURAL_GATE_OUTPUT");
            Directory.CreateDirectory(output);
            ClearScene();
            ConfigurePhysics();
            string stage = Environment.GetEnvironmentVariable("PROCEDURAL_GATE_STAGE") ?? "integrated";
            CompiledEpisodeContract contract = LoadAndVerifyCompiledContract();
            var context = BuildAuthoritativeContext(contract, output, stage);
            context.AuthorityAudit.sourceAuditSha256 = RequiredSha256("PROCEDURAL_GATE_SOURCE_AUDIT_SHA256");
            EpisodeTraceRow[] replayRows = LoadReplayTraceIfRequested(context);
            ISceneCompilerModule scene = new ProceduralSceneCompiler();
            var embodiment = new EmbodimentGarments();
            IFullBodyMotionModule motion = new FullBodyBimanualMotion();
            var truth = new PhysicsTruthRecorder();
            IRegisteredCaptureModule capture = new RegisteredCapture();

            embodiment.Build(context, "Assets/Avatar/child.fbx");
            embodiment.ApplyGarmentConfiguration(context, context.GarmentConfigurationId);
            scene.Build(context, "Assets/Furniture");
            if (stage == "rerender")
            {
                RunRenderOnlyReplay(context, replayRows, embodiment, capture);
                WriteAuthorityReceipt(context, stage);
                AssetDatabase.SaveAssets();
                EditorApplication.Exit(0);
                return;
            }
            if ((stage == "clipping" || stage == "motion_camera") && context.TargetBody)
                context.TargetBody.gameObject.SetActive(false);
            motion.Bind(context);
            capture.Bind(context);
            truth.Bind(context);
            Physics.SyncTransforms();

            for (int step = 0; step < TotalSteps; step++)
            {
                context.PhysicsStep = step;
                context.TimeSeconds = step * Dt;
                motion.ApplyCommand(context, step);
                embodiment.UpdateRegisteredCollidersBeforePhysics(context);
                Physics.SyncTransforms();
                truth.RecordBeforePhysicsStep(context);
                Physics.Simulate(Dt);
                motion.SynchronizeCompletedPhysicsState(context, step);
                Physics.SyncTransforms();
                context.TimeSeconds = (step + 1) * Dt;
                embodiment.SampleRegistrationAtPhysicsStep(context);
                context.RenderFrame = step / FrozenGate.StepsPerFrame;
                truth.RecordAfterPhysicsStep(context);
                if (replayRows != null)
                    CompareReplayObjectState(context, replayRows[step]);
                if ((step + 1) % FrozenGate.StepsPerFrame == 0)
                {
                    capture.CaptureFrozenFrame(context);
                }
            }

            truth.Complete(context);
            capture.Complete(context);
            string registrationPath = Path.Combine(output, "registration_report.json");
            embodiment.MeasureRegistration(context, registrationPath);
            bool registrationPassed = RegistrationReportPassed(registrationPath);
            if (replayRows != null)
                WriteReplayConsumptionReceipt(context, false, replayRows.Length);
            WriteAuthorityReceipt(context, stage);
            AssetDatabase.SaveAssets();
            EditorApplication.Exit(AuthorityAuditPassed(context) && registrationPassed ? 0 : 3);
        }

        private static CompiledEpisodeContract LoadAndVerifyCompiledContract()
        {
            string path = Path.GetFullPath(RequiredEnvironment("PROCEDURAL_GATE_CONTRACT"));
            if (!File.Exists(path)) throw new FileNotFoundException("compiled contract is missing", path);
            string raw = File.ReadAllText(path, Encoding.UTF8);
            string expectedFileSha = RequiredSha256("PROCEDURAL_GATE_CONTRACT_FILE_SHA256");
            if (!string.Equals(Sha256(path), expectedFileSha, StringComparison.Ordinal))
                throw new InvalidOperationException("compiled contract file SHA-256 does not match the runner seal");
            CompiledEpisodeContract contract = JsonUtility.FromJson<CompiledEpisodeContract>(raw);
            if (contract == null || contract.schema != "embodied.compiled_episode_contract.v2")
                throw new InvalidOperationException("compiled contract has the wrong or missing schema");
            string expectedContractSha = RequiredSha256("PROCEDURAL_GATE_CONTRACT_SHA256");
            if (!string.Equals(contract.contract_sha256, expectedContractSha, StringComparison.Ordinal))
                throw new InvalidOperationException("compiled contract semantic SHA-256 does not match the runner seal");
            return contract;
        }

        private static GateContext BuildAuthoritativeContext(
            CompiledEpisodeContract contract,
            string output,
            string stage
        )
        {
            if (contract.episode_spec == null || contract.avatar_spec == null || contract.scene_spec == null
                || contract.scene_spec.target == null || contract.scene_spec.reachability == null
                || contract.scene_spec.instances == null || contract.scene_spec.zones == null
                || contract.activity_plan == null || contract.activity_plan.phases == null
                || contract.authority == null || contract.qa_tolerances == null)
                throw new InvalidOperationException("compiled contract omits an authoritative runtime section");
            RequireExactEnvironment("PROCEDURAL_GATE_EPISODE_ID", contract.episode_id);
            RequireExactEnvironment("PROCEDURAL_GATE_CELL_ID", contract.episode_spec.cell_id);
            RequireExactEnvironment("PROCEDURAL_GATE_TARGET_ID", contract.episode_spec.target_id);
            RequireExactEnvironment("PROCEDURAL_GATE_DESTINATION_ID", contract.episode_spec.destination_id);
            RequireExactEnvironment("PROCEDURAL_GATE_CONTACT_STRATEGY", contract.episode_spec.contact_strategy);
            RequireExactEnvironment("PROCEDURAL_GATE_FINAL_GAZE_ZONE", contract.episode_spec.final_gaze_zone);
            RequireExactEnvironment("PROCEDURAL_GATE_ROOM_FAMILY", contract.scene_spec.room_family);
            RequireExactEnvironment("PROCEDURAL_GATE_GARMENT_CONFIG", contract.avatar_spec.garment_configuration_id);
            if (RequiredInt("PROCEDURAL_GATE_SEED") != contract.scene_spec.seed)
                throw new InvalidOperationException("PROCEDURAL_GATE_SEED does not exactly match the compiled contract");
            if (contract.episode_spec.episode_id != contract.episode_id
                || contract.episode_spec.schema != "embodied.episode_spec.v2"
                || contract.avatar_spec.schema != "embodied.avatar_spec.v1"
                || contract.scene_spec.schema != "embodied.scene_spec.v1"
                || contract.activity_plan.schema != "embodied.activity_plan.v1"
                || contract.episode_spec.seed != contract.scene_spec.seed
                || contract.episode_spec.room_family != contract.scene_spec.room_family
                || contract.episode_spec.target_id != contract.scene_spec.target.persistent_id
                || contract.episode_spec.destination_id != contract.scene_spec.destination_id
                || contract.episode_spec.garment_configuration_id != contract.avatar_spec.garment_configuration_id)
                throw new InvalidOperationException("compiled contract identity fields disagree internally");
            if (contract.scene_spec.target.dimensions_m == null || contract.scene_spec.target.dimensions_m.Length != 3
                || contract.scene_spec.envelope_m == null || contract.scene_spec.envelope_m.Length != 3
                || contract.scene_spec.reachability.compiled_midpoint_band_m == null
                || contract.scene_spec.reachability.compiled_midpoint_band_m.Length != 2
                || string.IsNullOrWhiteSpace(contract.scene_spec.material_variant)
                || contract.scene_spec.instances.Length < 2
                || contract.scene_spec.minimum_contextual_objects <= 10
                || contract.scene_spec.minimum_contextual_objects > contract.scene_spec.instances.Length
                || contract.scene_spec.support_relations == null
                || contract.scene_spec.support_relations.Length != 2
                || contract.scene_spec.sightlines == null
                || !contract.scene_spec.sightlines.target_visible_at_required_events
                || contract.scene_spec.stabilization_s <= 0f
                || !contract.scene_spec.no_visible_primitive_furniture
                || contract.scene_spec.zones.Any(string.IsNullOrWhiteSpace)
                || contract.scene_spec.instances.Any(value => value == null
                    || string.IsNullOrWhiteSpace(value.persistent_id)
                    || string.IsNullOrWhiteSpace(value.asset_id)
                    || value.asset_dimensions_m == null
                    || value.asset_dimensions_m.Length != 3
                    || string.IsNullOrWhiteSpace(value.semantic_class)
                    || string.IsNullOrWhiteSpace(value.collision_source)))
                throw new InvalidOperationException("compiled target dimensions/reach band are malformed");
            if (Mathf.Abs(contract.activity_plan.duration_s - FrozenGate.DurationSeconds) > 1e-6f)
                throw new InvalidOperationException("compiled ActivityPlan duration differs from the frozen runtime clock");
            ValidateAuthorityAndTolerances(contract);

            var context = new GateContext
            {
                EpisodeId = contract.episode_id,
                OutputRoot = output,
                Seed = contract.scene_spec.seed,
                RoomFamily = contract.scene_spec.room_family,
                GarmentConfigurationId = contract.avatar_spec.garment_configuration_id,
                CellId = contract.episode_spec.cell_id,
                TargetId = contract.episode_spec.target_id,
                DestinationId = contract.episode_spec.destination_id,
                ContactStrategy = contract.episode_spec.contact_strategy,
                FinalGazeZone = contract.episode_spec.final_gaze_zone,
                CompiledContractPath = Path.GetFullPath(RequiredEnvironment("PROCEDURAL_GATE_CONTRACT")),
                CompiledContractSha256 = contract.contract_sha256,
                RobustnessVariant = stage == "robustness"
                    ? Environment.GetEnvironmentVariable("PROCEDURAL_GATE_VARIANT") ?? "nominal"
                    : "nominal",
                ReplayTracePath = Environment.GetEnvironmentVariable("PROCEDURAL_GATE_REPLAY_TRACE") ?? "",
                TargetMidpointReachM = contract.scene_spec.reachability.compiled_requested_midpoint_m,
                TargetReachBandMinM = contract.scene_spec.reachability.compiled_midpoint_band_m[0],
                TargetReachBandMaxM = contract.scene_spec.reachability.compiled_midpoint_band_m[1],
                TargetLateralBiasM = contract.scene_spec.reachability.lateral_bias_toward_right_shoulder_m,
                TargetDimensionsM = new Vector3(
                    contract.scene_spec.target.dimensions_m[0],
                    contract.scene_spec.target.dimensions_m[1],
                    contract.scene_spec.target.dimensions_m[2]
                ),
                TargetMassKg = contract.scene_spec.target.mass_kg,
                TargetStaticFriction = contract.scene_spec.target.static_friction,
                TargetDynamicFriction = contract.scene_spec.target.dynamic_friction,
                TargetGeometry = contract.scene_spec.target.geometry,
                TargetSemanticId = contract.scene_spec.target.semantic_id,
                TargetInstanceId = contract.scene_spec.target.instance_id,
                SceneEnvelopeM = new Vector3(
                    contract.scene_spec.envelope_m[0],
                    contract.scene_spec.envelope_m[1],
                    contract.scene_spec.envelope_m[2]
                ),
                SceneMaterialVariant = contract.scene_spec.material_variant,
                SceneZoneIds = contract.scene_spec.zones.ToArray(),
                ExpectedSceneAssetIds = contract.scene_spec.instances.Select(value => value.asset_id).ToArray(),
                ExpectedSceneInstances = contract.scene_spec.instances.Select(value => new SceneInstanceAuthority
                {
                    PersistentId = value.persistent_id,
                    AssetId = value.asset_id,
                    AssetDimensionsM = new Vector3(
                        value.asset_dimensions_m[0], value.asset_dimensions_m[1], value.asset_dimensions_m[2]),
                    SemanticClass = value.semantic_class,
                    CollisionSource = value.collision_source,
                    Interactive = value.interactive,
                    MassKg = value.mass_kg,
                    StaticFriction = value.static_friction,
                    DynamicFriction = value.dynamic_friction
                }).ToArray(),
                ExpectedSupportRelations = contract.scene_spec.support_relations.Select(value => new SceneSupportAuthority
                {
                    ChildId = value.child_id,
                    SupportId = value.support_id,
                    DestinationId = value.destination_id
                }).ToArray(),
                ExpectedTargetVisibleAtRequiredEvents = contract.scene_spec.sightlines.target_visible_at_required_events,
                ExpectedFinalGazeZone = contract.scene_spec.sightlines.final_gaze_zone,
                SceneStabilizationSeconds = contract.scene_spec.stabilization_s,
                RequireNoVisiblePrimitiveFurniture = contract.scene_spec.no_visible_primitive_furniture,
                MinimumContextualObjects = contract.scene_spec.minimum_contextual_objects,
                AuthorityRoot = new GameObject("AUTHORITATIVE_PHYSICS_CLOCKED_STATE")
            };
            PopulateActivityPhases(context, contract.activity_plan.phases);
            if (stage == "robustness") ApplyFrozenRobustnessVariant(context, contract.robustness_variants);
            return context;
        }

        private static void ValidateAuthorityAndTolerances(CompiledEpisodeContract contract)
        {
            AuthorityContract authority = contract.authority;
            QaTolerancesContract tolerances = contract.qa_tolerances;
            if (authority.physics_hz != FrozenGate.PhysicsHz || authority.render_hz != FrozenGate.RenderHz
                || authority.steps_per_render_frame != FrozenGate.StepsPerFrame
                || authority.biological_torque_claim_permitted)
                throw new InvalidOperationException("compiled authority does not match the frozen runtime authority");
            if (tolerances.schema != "embodied.qa_tolerances.v1" || tolerances.clock == null
                || tolerances.registration == null || tolerances.contact == null
                || tolerances.interaction == null || tolerances.camera == null
                || tolerances.capture == null || tolerances.replay == null)
                throw new InvalidOperationException("compiled QA tolerance schema is missing required sections");
            if (tolerances.clock.physics_hz != FrozenGate.PhysicsHz
                || tolerances.clock.render_hz != FrozenGate.RenderHz
                || tolerances.clock.exact_steps_per_render_frame != FrozenGate.StepsPerFrame
                || tolerances.capture.fps != FrozenGate.RenderHz
                || tolerances.capture.resolution_px == null || tolerances.capture.resolution_px.Length != 2
                || tolerances.capture.resolution_px[0] != FrozenGate.Width
                || tolerances.capture.resolution_px[1] != FrozenGate.Height
                || !tolerances.capture.same_frozen_frame_required
                || !tolerances.capture.contact_projection_required
                || !tolerances.capture.zero_proxy_hero_pixels)
                throw new InvalidOperationException("compiled clock/capture tolerances differ from the frozen runtime");
            if (Mathf.Abs(tolerances.registration.skin_collider_max_m - 0.005f) > 1e-7f
                || Mathf.Abs(tolerances.registration.garment_body_max_penetration_m - 0.002f) > 1e-7f
                || Mathf.Abs(tolerances.registration.garment_affected_vertex_fraction_max - 0.001f) > 1e-7f
                || Mathf.Abs(tolerances.registration.finger_object_max_penetration_m - FrozenGate.FingerObjectPenetrationMaxM) > 1e-7f
                || Mathf.Abs(tolerances.registration.support_max_penetration_m - FrozenGate.SupportPenetrationMaxM) > 1e-7f
                || !tolerances.registration.positive_camera_head_hair_garment_clearance_every_frame)
                throw new InvalidOperationException("compiled registration tolerances differ from the frozen contract");
            if (Mathf.Abs(tolerances.contact.qualification_max_measured_separation_m - 0.0005f) > 1e-7f
                || !tolerances.contact.simultaneous_nonzero_physx_impulse_required
                || !tolerances.contact.correct_visible_surface_projection_required
                || tolerances.contact.visible_physical_max_frame_delta != 1)
                throw new InvalidOperationException("compiled contact tolerances differ from the frozen contract");
            if (Mathf.Abs(tolerances.interaction.right_opposition_min_s - FrozenGate.RightForceOppositionSeconds) > 1e-7f
                || Mathf.Abs(tolerances.interaction.left_support_min_s - FrozenGate.LeftForceSupportSeconds) > 1e-7f
                || Mathf.Abs(tolerances.interaction.lift_min_m - 0.10f) > 1e-7f
                || Mathf.Abs(tolerances.interaction.turn_min_deg - 30f) > 1e-7f
                || !tolerances.interaction.free_release_required || !tolerances.interaction.unsupported_required
                || !tolerances.interaction.support_continuous_until_commanded_open
                || !tolerances.interaction.settling_required)
                throw new InvalidOperationException("compiled interaction tolerances differ from the frozen contract");
            if (tolerances.camera.fov_candidates_deg == null
                || !tolerances.camera.fov_candidates_deg.SequenceEqual(new[] { 60, 68, 75 })
                || Mathf.Abs(tolerances.camera.optical_vs_face_forward_max_deg - 15f) > 1e-7f
                || Mathf.Abs(tolerances.camera.minimum_clearance_m - 0.001f) > 1e-7f
                || Mathf.Abs(tolerances.camera.roll_abs_max_deg - 12f) > 1e-7f)
                throw new InvalidOperationException("compiled camera tolerances differ from the frozen contract");
            if (!tolerances.replay.fresh_process_required || !tolerances.replay.same_trace_rerender_required
                || Mathf.Abs(tolerances.replay.translation_max_m - 0.0005f) > 1e-7f
                || Mathf.Abs(tolerances.replay.rotation_max_deg - 0.05f) > 1e-7f
                || Mathf.Abs(tolerances.replay.object_velocity_max_m_s - 0.002f) > 1e-7f
                || Mathf.Abs(tolerances.replay.rerender_min_psnr_db - 60f) > 1e-7f)
                throw new InvalidOperationException("compiled replay tolerances differ from the frozen contract");
        }

        private static void PopulateActivityPhases(GateContext context, ActivityPhaseContract[] phases)
        {
            float cursor = 0f;
            foreach (ActivityPhaseContract phase in phases)
            {
                if (phase == null || string.IsNullOrWhiteSpace(phase.id) || context.ActivityPhasesSeconds.ContainsKey(phase.id)
                    || Mathf.Abs(phase.start_s - cursor) > 1e-6f || phase.end_s <= phase.start_s)
                    throw new InvalidOperationException("compiled ActivityPlan phases are malformed or non-contiguous");
                context.ActivityPhasesSeconds.Add(phase.id, new Vector2(phase.start_s, phase.end_s));
                cursor = phase.end_s;
            }
            if (Mathf.Abs(cursor - FrozenGate.DurationSeconds) > 1e-6f)
                throw new InvalidOperationException("compiled ActivityPlan does not cover the frozen duration");
        }

        private static void ApplyFrozenRobustnessVariant(GateContext context, RobustnessVariantContract[] variants)
        {
            if (variants == null) throw new InvalidOperationException("compiled robustness variants are missing");
            RobustnessVariantContract variant = variants.SingleOrDefault(value => value != null && value.id == context.RobustnessVariant);
            if (variant == null) throw new InvalidOperationException("unknown robustness variant: " + context.RobustnessVariant);
            if (variant.mass_scale <= 0f || variant.static_friction_scale <= 0f || variant.dynamic_friction_scale <= 0f)
                throw new InvalidOperationException("compiled robustness scale must be positive");
            context.TargetLateralBiasM += variant.target_lateral_shift_m;
            context.TargetMassKg *= variant.mass_scale;
            context.TargetStaticFriction *= variant.static_friction_scale;
            context.TargetDynamicFriction *= variant.dynamic_friction_scale;
        }

        private static float replayTranslationMaxM;
        private static float replayRotationMaxDeg;
        private static float replayObjectVelocityMaxMps;
        private static readonly Dictionary<string, FingerBodyReplayBinding> replayFingerBodyBindings =
            new Dictionary<string, FingerBodyReplayBinding>(StringComparer.Ordinal);

        private static EpisodeTraceRow[] LoadReplayTraceIfRequested(GateContext context)
        {
            string stage = Environment.GetEnvironmentVariable("PROCEDURAL_GATE_STAGE") ?? "integrated";
            bool required = stage == "replay" || stage == "rerender";
            if (string.IsNullOrWhiteSpace(context.ReplayTracePath))
            {
                if (required) throw new InvalidOperationException(stage + " requires PROCEDURAL_GATE_REPLAY_TRACE");
                return null;
            }
            string path = Path.GetFullPath(context.ReplayTracePath);
            if (!File.Exists(path)) throw new FileNotFoundException("replay trace is missing", path);
            string expectedSha = RequiredSha256("PROCEDURAL_GATE_REPLAY_TRACE_SHA256");
            if (!string.Equals(Sha256(path), expectedSha, StringComparison.Ordinal))
                throw new InvalidOperationException("replay trace SHA-256 does not match the runner seal");
            string[] lines = File.ReadAllLines(path, Encoding.UTF8);
            if (lines.Length != TotalSteps || lines.Any(string.IsNullOrWhiteSpace))
                throw new InvalidOperationException("replay trace must contain exactly one nonempty row per frozen physics step");
            var rows = new EpisodeTraceRow[TotalSteps];
            for (int step = 0; step < rows.Length; step++)
            {
                try { rows[step] = JsonUtility.FromJson<EpisodeTraceRow>(lines[step]); }
                catch (Exception error) { throw new InvalidOperationException("malformed replay trace JSON at row " + step, error); }
                ValidateReplayRow(context, rows[step], step, stage == "rerender");
            }
            context.ReplayTracePath = path;
            replayTranslationMaxM = 0f;
            replayRotationMaxDeg = 0f;
            replayObjectVelocityMaxMps = 0f;
            return rows;
        }

        private static void ValidateReplayRow(GateContext context, EpisodeTraceRow row, int step, bool renderOnly)
        {
            if (row == null || row.schema != FrozenGate.TraceSchema || row.episode == null || row.clock == null
                || row.objects == null || row.objects.Length != 1 || row.objects[0] == null || !row.objects[0].active
                || row.objects[0].pose == null)
                throw new InvalidOperationException("replay trace row is missing required authority state at step " + step);
            if (row.episode.episode_id != context.EpisodeId || row.episode.cell_id != context.CellId
                || row.episode.room_family != context.RoomFamily
                || row.episode.garment_configuration_id != context.GarmentConfigurationId
                || row.episode.target_id != context.TargetId || row.episode.destination_id != context.DestinationId
                || row.episode.contact_strategy != context.ContactStrategy
                || row.episode.final_gaze_zone != context.FinalGazeZone
                || row.objects[0].persistent_id != context.TargetId)
                throw new InvalidOperationException("replay trace labels differ from the compiled contract at step " + step);
            float expectedTime = (step + 1) * Dt;
            if (row.clock.physics_step != step || row.clock.render_frame != step / FrozenGate.StepsPerFrame
                || row.clock.physics_hz != FrozenGate.PhysicsHz || row.clock.render_hz != FrozenGate.RenderHz
                || row.clock.steps_per_render_frame != FrozenGate.StepsPerFrame
                || Mathf.Abs(row.clock.time_s - expectedTime) > 2e-6f)
                throw new InvalidOperationException("replay trace clock is malformed at step " + step);
            if (renderOnly && (row.body_state == null || row.hand_state == null || row.camera_state == null))
                throw new InvalidOperationException("render-only replay trace omits body/hand/camera state at step " + step);
        }

        private static void CompareReplayObjectState(GateContext context, EpisodeTraceRow row)
        {
            ObjectTruth expected = row.objects[0];
            replayTranslationMaxM = Mathf.Max(
                replayTranslationMaxM,
                Vector3.Distance(context.TargetBody.position, expected.pose.position_world_m)
            );
            replayRotationMaxDeg = Mathf.Max(
                replayRotationMaxDeg,
                Quaternion.Angle(context.TargetBody.rotation, expected.pose.rotation_world_xyzw)
            );
            replayObjectVelocityMaxMps = Mathf.Max(
                replayObjectVelocityMaxMps,
                Vector3.Distance(context.TargetBody.linearVelocity, expected.linear_velocity_world_m_s)
            );
        }

        private static void RunRenderOnlyReplay(
            GateContext context,
            EpisodeTraceRow[] rows,
            EmbodimentGarments embodiment,
            IRegisteredCaptureModule capture
        )
        {
            if (rows == null || rows.Length != TotalSteps)
                throw new InvalidOperationException("render-only rerender requires a complete validated source trace");
            PrepareReplayFingerBodyBindings(context);
            capture.Bind(context);
            for (int step = 0; step < rows.Length; step++)
            {
                EpisodeTraceRow row = rows[step];
                context.PhysicsStep = step;
                context.TimeSeconds = row.clock.time_s;
                context.RenderFrame = row.clock.render_frame;
                ApplyRenderOnlyReplayState(context, row, true);
                embodiment.UpdateRegisteredCollidersBeforePhysics(context);
                Physics.SyncTransforms();
                // Consume the paired collider-clearance snapshot so the render-only
                // loop cannot carry a stale prospective audit into the next row.
                // This receipt is not emitted or eligible as physics evidence.
                embodiment.SampleRegistrationAtPhysicsStep(context);
                if ((step + 1) % FrozenGate.StepsPerFrame == 0)
                    capture.CaptureFrozenFrame(context);
            }
            capture.Complete(context);
            WriteReplayConsumptionReceipt(context, true, rows.Length);
        }

        private static void ApplyRenderOnlyReplayState(GateContext context, EpisodeTraceRow row, bool renderOnly)
        {
            if (!renderOnly) throw new InvalidOperationException("trace-driven pose writes are forbidden in physics execution");
            ApplyReplayBodySegment(context, "root", row.body_state.root);
            ApplyReplayBodySegment(context, "pelvis", row.body_state.pelvis);
            ApplyReplayBodySegment(context, "torso", row.body_state.torso);
            ApplyReplayBodySegment(context, "neck", row.body_state.neck);
            ApplyReplayBodySegment(context, "head", row.body_state.head);
            ApplyReplayBodySegment(context, "left_shoulder", row.body_state.left_shoulder);
            ApplyReplayBodySegment(context, "left_upper_arm", row.body_state.left_upper_arm);
            ApplyReplayBodySegment(context, "left_elbow", row.body_state.left_elbow);
            ApplyReplayBodySegment(context, "left_lower_arm", row.body_state.left_lower_arm);
            ApplyReplayBodySegment(context, "left_forearm", row.body_state.left_forearm);
            ApplyReplayBodySegment(context, "left_wrist", row.body_state.left_wrist);
            ApplyReplayBodySegment(context, "right_shoulder", row.body_state.right_shoulder);
            ApplyReplayBodySegment(context, "right_upper_arm", row.body_state.right_upper_arm);
            ApplyReplayBodySegment(context, "right_elbow", row.body_state.right_elbow);
            ApplyReplayBodySegment(context, "right_lower_arm", row.body_state.right_lower_arm);
            ApplyReplayBodySegment(context, "right_forearm", row.body_state.right_forearm);
            ApplyReplayBodySegment(context, "right_wrist", row.body_state.right_wrist);
            ApplyReplayPose(context.LeftPalm, row.hand_state.left_palm);
            ApplyReplayPose(context.RightPalm, row.hand_state.right_palm);
            ApplyReplayDigit(context, "left_thumb", row.hand_state.left_thumb);
            ApplyReplayDigit(context, "left_index", row.hand_state.left_index);
            ApplyReplayDigit(context, "left_middle", row.hand_state.left_middle);
            ApplyReplayDigit(context, "left_ring", row.hand_state.left_ring);
            ApplyReplayDigit(context, "left_little", row.hand_state.left_little);
            ApplyReplayDigit(context, "right_thumb", row.hand_state.right_thumb);
            ApplyReplayDigit(context, "right_index", row.hand_state.right_index);
            ApplyReplayDigit(context, "right_middle", row.hand_state.right_middle);
            ApplyReplayDigit(context, "right_ring", row.hand_state.right_ring);
            ApplyReplayDigit(context, "right_little", row.hand_state.right_little);
            ApplyReplayTargetState(context.TargetBody, row.objects[0], renderOnly);
            context.AuthorityAudit.targetPoseWriteCounter += 2;
            context.AuthorityAudit.targetVelocityWriteCounter += 2;
            context.Contacts.Clear();
            foreach (ContactTruthRow contact in row.contacts ?? Array.Empty<ContactTruthRow>())
            {
                context.Contacts.Add(new ContactTruth
                {
                    physicsStep = row.clock.physics_step,
                    colliderA = contact.collider_a,
                    colliderB = contact.collider_b,
                    pointWorldM = contact.point_world_m,
                    normalWorld = contact.normal_world,
                    separationM = contact.separation_m,
                    relativeVelocityWorldMps = contact.relative_velocity_world_m_s,
                    availableImpulseNs = contact.available_impulse_n_s,
                    provenance = TruthSource.PhysXMeasured
                });
            }
        }

        private static void ApplyReplayPose(Transform target, PoseTruth pose)
        {
            if (target == null || pose == null) throw new InvalidOperationException("render-only replay pose binding is missing");
            target.SetPositionAndRotation(pose.position_world_m, pose.rotation_world_xyzw);
        }

        private static void ApplyReplayBodySegment(GateContext context, string key, PoseTruth pose)
        {
            if (!context.BodySegments.TryGetValue(key, out Transform target))
                throw new InvalidOperationException("render-only replay body binding is missing: " + key);
            ApplyReplayPose(target, pose);
        }

        private static void ApplyReplayDigit(GateContext context, string key, DigitTruth digit)
        {
            if (digit == null || digit.segments == null || digit.segments.Length != 3
                || digit.dynamic_body_states == null || digit.dynamic_body_states.Length != 3
                || !context.FingerSegments.TryGetValue(key, out Transform[] segments) || segments.Length != 3)
                throw new InvalidOperationException("render-only replay digit binding is malformed: " + key);
            for (int index = 0; index < 3; index++)
            {
                PoseTruth pose = digit.segments[index].pose;
                ApplyReplayPose(segments[index], pose);
                string bodyKey = key + "_segment" + (index + 1);
                if (!replayFingerBodyBindings.TryGetValue(bodyKey, out FingerBodyReplayBinding binding)
                    || binding == null || binding.bone != segments[index])
                    throw new InvalidOperationException("render-only replay finger-body offset binding is missing: " + bodyKey);
                ApplyReplayFingerBody(binding, digit.dynamic_body_states[index]);
            }
        }

        private static void PrepareReplayFingerBodyBindings(GateContext context)
        {
            replayFingerBodyBindings.Clear();
            foreach (KeyValuePair<string, Rigidbody> pair in context.FingerBodies)
            {
                if (pair.Value == null || !context.FingerAuthorityBones.TryGetValue(pair.Key, out Transform bone)
                    || bone == null)
                    throw new InvalidOperationException("render-only replay finger body/bone binding is incomplete: " + pair.Key);
                Vector3 positionInBone = bone.InverseTransformPoint(pair.Value.position);
                Quaternion rotationInBone = Quaternion.Inverse(bone.rotation) * pair.Value.rotation;
                if (positionInBone.sqrMagnitude <= 1e-12f && Quaternion.Angle(rotationInBone, Quaternion.identity) <= 1e-5f)
                    throw new InvalidOperationException("render-only replay requires the measured nonzero bone/body offset: " + pair.Key);
                replayFingerBodyBindings.Add(pair.Key, new FingerBodyReplayBinding
                {
                    body = pair.Value,
                    bone = bone,
                    positionInBone = positionInBone,
                    rotationInBone = rotationInBone
                });
            }
            if (replayFingerBodyBindings.Count != 30)
                throw new InvalidOperationException("render-only replay requires all thirty dynamic finger-body offsets");
        }

        private static void ApplyReplayFingerBody(
            FingerBodyReplayBinding binding,
            DynamicFingerBodyTruth state
        )
        {
            if (state == null || state.provenance != "physx_measured" || state.is_kinematic
                || state.body_id != StableTransformId(binding.body.transform) || state.pose == null)
                throw new InvalidOperationException("render-only replay dynamic finger-body truth is unavailable or non-dynamic");
            binding.body.position = state.pose.position_world_m;
            binding.body.rotation = state.pose.rotation_world_xyzw;
            binding.body.linearVelocity = state.linear_velocity_world_m_s;
            binding.body.angularVelocity = state.angular_velocity_world_rad_s;
        }

        private static string StableTransformId(Transform transform)
        {
            var parts = new List<string>();
            for (Transform cursor = transform; cursor != null; cursor = cursor.parent)
                parts.Add(cursor.name + "[" + cursor.GetSiblingIndex() + "]");
            parts.Reverse();
            return string.Join("/", parts);
        }

        private static void ApplyReplayTargetState(Rigidbody body, ObjectTruth state, bool renderOnly)
        {
            if (!renderOnly) throw new InvalidOperationException("target replay writes are forbidden during physics execution");
            body.position = state.pose.position_world_m;
            body.rotation = state.pose.rotation_world_xyzw;
            body.linearVelocity = state.linear_velocity_world_m_s;
            body.angularVelocity = state.angular_velocity_world_rad_s;
        }

        private static void WriteReplayConsumptionReceipt(GateContext context, bool renderOnly, int rows)
        {
            var receipt = new ReplayConsumptionReceipt
            {
                schema = "embodied.unity_replay_consumption.v1",
                episode_id = context.EpisodeId,
                contract_sha256 = context.CompiledContractSha256,
                source_trace = context.ReplayTracePath,
                source_trace_sha256 = Sha256(context.ReplayTracePath),
                rows_consumed = rows,
                render_only = renderOnly,
                physics_simulated = !renderOnly,
                trace_driven_target_writes = renderOnly,
                eligible_manipulation_evidence = !renderOnly,
                replayed_state_provenance = renderOnly ? "source_trace_render_replay_not_manipulation_truth" : "fresh_process_physx_compared_to_source_trace",
                translation_max_m = replayTranslationMaxM,
                rotation_max_deg = replayRotationMaxDeg,
                object_velocity_max_m_s = replayObjectVelocityMaxMps
            };
            File.WriteAllText(
                Path.Combine(context.OutputRoot, "replay_consumption_receipt.json"),
                JsonUtility.ToJson(receipt, true) + "\n",
                new UTF8Encoding(false)
            );
        }

        private static void ConfigurePhysics()
        {
            Physics.simulationMode = SimulationMode.Script;
            Physics.gravity = new Vector3(0f, -9.81f, 0f);
            Physics.defaultSolverIterations = 24;
            Physics.defaultSolverVelocityIterations = 12;
            Physics.reuseCollisionCallbacks = false;
        }

        private static void ClearScene()
        {
            foreach (GameObject gameObject in UnityEngine.Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None))
                UnityEngine.Object.DestroyImmediate(gameObject);
        }

        private static void WriteAuthorityReceipt(GateContext context, string stage)
        {
            var receipt = new AuthorityReceipt
            {
                schema = "embodied.single_authority_receipt.v1",
                episode_id = context.EpisodeId,
                stage = stage,
                physics_hz = FrozenGate.PhysicsHz,
                render_hz = FrozenGate.RenderHz,
                steps_per_render_frame = FrozenGate.StepsPerFrame,
                physics_steps = stage == "rerender" ? 0 : TotalSteps,
                render_frames = TotalSteps / FrozenGate.StepsPerFrame,
                authority_root = context.AuthorityRoot ? context.AuthorityRoot.name : "UNAVAILABLE",
                avatar_root = context.AvatarRoot ? context.AvatarRoot.name : "UNAVAILABLE",
                torso = context.Torso ? context.Torso.name : "UNAVAILABLE",
                neck = context.Neck ? context.Neck.name : "UNAVAILABLE",
                head = context.Head ? context.Head.name : "UNAVAILABLE",
                camera_parent = context.HeadCameraMount && context.HeadCameraMount.parent ? context.HeadCameraMount.parent.name : "UNAVAILABLE",
                target_rigidbody = context.TargetBody ? context.TargetBody.name : "UNAVAILABLE",
                object_pose_writes_after_initialization = context.AuthorityAudit.targetPoseWriteCounter,
                object_velocity_writes_after_initialization = context.AuthorityAudit.targetVelocityWriteCounter,
                object_external_forces = context.AuthorityAudit.targetForceCounter,
                object_external_torques = context.AuthorityAudit.targetTorqueCounter,
                attachment_or_joint_count = context.AuthorityAudit.targetParentingCounter + context.AuthorityAudit.targetJointCounter,
                object_parenting_changes = context.AuthorityAudit.targetParentingCounter,
                object_joint_changes = context.AuthorityAudit.targetJointCounter,
                object_kinematic_changes = context.AuthorityAudit.targetKinematicChangeCounter,
                recovery_ledger_entries = context.RecoveryLedger.Count,
                assistance_ledger_entries = context.AssistanceLedger.Count,
                source_audit_sha256 = context.AuthorityAudit.sourceAuditSha256,
                runtime_accounting_passed = stage != "rerender" && AuthorityAuditPassed(context),
                runtime_detection_coverage = "boundary-sampled target parenting, Joint graph, isKinematic, and between-step Rigidbody/Transform pose and velocity discontinuities",
                force_torque_detection_boundary = "counters require instrumented call sites; mandatory source-audit SHA-256 remains required",
                zero_counters_are_independent_proof = false,
                independent_render_timeline = false,
                single_state_drives_body_clothing_camera_truth = context.AvatarRoot && context.Head && context.HeadCameraMount && context.TargetBody,
                render_only_trace_playback = stage == "rerender",
                eligible_manipulation_evidence = stage != "rerender",
                disclosure = "Kinematic embodiment commands and engine-observed poses are engineering control; the free target and contacts are PhysX-measured. No biological torque is claimed."
            };
            File.WriteAllText(
                Path.Combine(context.OutputRoot, "authority_receipt.json"),
                JsonUtility.ToJson(receipt, true)
            );
        }

        private static string RequiredEnvironment(string name)
        {
            string value = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(value)) throw new Exception(name + " is required");
            return value;
        }

        private static int RequiredInt(string name)
        {
            if (!int.TryParse(RequiredEnvironment(name), out int value)) throw new Exception(name + " must be an integer");
            return value;
        }

        private static void RequireExactEnvironment(string name, string expected)
        {
            string actual = RequiredEnvironment(name);
            if (!string.Equals(actual, expected, StringComparison.Ordinal))
                throw new InvalidOperationException(name + " does not exactly match the compiled contract");
        }

        private static string RequiredSha256(string name)
        {
            string value = RequiredEnvironment(name).Trim().ToLowerInvariant();
            if (value.Length != 64 || value.Any(character => !Uri.IsHexDigit(character)))
                throw new Exception(name + " must be a 64-character hexadecimal SHA-256");
            return value;
        }

        private static string Sha256(string path)
        {
            using (SHA256 hash = SHA256.Create())
            using (FileStream stream = File.OpenRead(path))
                return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }

        private static bool RegistrationReportPassed(string path)
        {
            if (!File.Exists(path)) return false;
            try
            {
                RegistrationPassReceipt receipt = JsonUtility.FromJson<RegistrationPassReceipt>(
                    File.ReadAllText(path, Encoding.UTF8)
                );
                return receipt != null && receipt.schema == "embodied.embodiment_registration.v2"
                    && receipt.self_clearance_sampled_every_physics_step
                    && receipt.non_adjacent_anatomy_clearance_passed
                    && receipt.passed;
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static bool AuthorityAuditPassed(GateContext context)
        {
            AuthorityAuditState audit = context.AuthorityAudit;
            return audit.targetPoseWriteCounter == 0
                   && audit.targetVelocityWriteCounter == 0
                   && audit.targetForceCounter == 0
                   && audit.targetTorqueCounter == 0
                   && audit.targetJointCounter == 0
                   && audit.targetParentingCounter == 0
                   && audit.targetKinematicChangeCounter == 0
                   && context.AssistanceLedger.Count == 0;
        }

        [Serializable]
        private sealed class AuthorityReceipt
        {
            public string schema;
            public string episode_id;
            public string stage;
            public int physics_hz;
            public int render_hz;
            public int steps_per_render_frame;
            public int physics_steps;
            public int render_frames;
            public string authority_root;
            public string avatar_root;
            public string torso;
            public string neck;
            public string head;
            public string camera_parent;
            public string target_rigidbody;
            public int object_pose_writes_after_initialization;
            public int object_velocity_writes_after_initialization;
            public int object_external_forces;
            public int object_external_torques;
            public int attachment_or_joint_count;
            public int object_parenting_changes;
            public int object_joint_changes;
            public int object_kinematic_changes;
            public int recovery_ledger_entries;
            public int assistance_ledger_entries;
            public string source_audit_sha256;
            public bool runtime_accounting_passed;
            public string runtime_detection_coverage;
            public string force_torque_detection_boundary;
            public bool zero_counters_are_independent_proof;
            public bool independent_render_timeline;
            public bool single_state_drives_body_clothing_camera_truth;
            public bool render_only_trace_playback;
            public bool eligible_manipulation_evidence;
            public string disclosure;
        }

        [Serializable]
        private sealed class CompiledEpisodeContract
        {
            public string schema;
            public string episode_id;
            public string contract_sha256;
            public EpisodeSpecContract episode_spec;
            public AvatarSpecContract avatar_spec;
            public SceneSpecContract scene_spec;
            public ActivityPlanContract activity_plan;
            public AuthorityContract authority;
            public QaTolerancesContract qa_tolerances;
            public RobustnessVariantContract[] robustness_variants;
        }

        [Serializable]
        private sealed class EpisodeSpecContract
        {
            public string schema;
            public string episode_id;
            public string cell_id;
            public int seed;
            public string room_family;
            public string garment_configuration_id;
            public string target_id;
            public string destination_id;
            public string contact_strategy;
            public string final_gaze_zone;
        }

        [Serializable]
        private sealed class AvatarSpecContract
        {
            public string schema;
            public string garment_configuration_id;
        }

        [Serializable]
        private sealed class SceneSpecContract
        {
            public string schema;
            public int seed;
            public string room_family;
            public string material_variant;
            public string destination_id;
            public float[] envelope_m;
            public int minimum_contextual_objects;
            public string[] zones;
            public SceneInstanceContract[] instances;
            public TargetContract target;
            public ReachabilityContract reachability;
            public SceneSupportContract[] support_relations;
            public SightlinesContract sightlines;
            public float stabilization_s;
            public bool no_visible_primitive_furniture;
        }

        [Serializable]
        private sealed class SceneInstanceContract
        {
            public string persistent_id;
            public string asset_id;
            public float[] asset_dimensions_m;
            public string semantic_class;
            public string collision_source;
            public bool interactive;
            public float mass_kg;
            public float static_friction;
            public float dynamic_friction;
        }

        [Serializable]
        private sealed class SceneSupportContract
        {
            public string child_id;
            public string support_id;
            public string destination_id;
        }

        [Serializable]
        private sealed class SightlinesContract
        {
            public bool target_visible_at_required_events;
            public string final_gaze_zone;
        }

        [Serializable]
        private sealed class TargetContract
        {
            public string persistent_id;
            public int semantic_id;
            public int instance_id;
            public string geometry;
            public float[] dimensions_m;
            public float mass_kg;
            public float static_friction;
            public float dynamic_friction;
        }

        [Serializable]
        private sealed class ReachabilityContract
        {
            public float compiled_requested_midpoint_m;
            public float[] compiled_midpoint_band_m;
            public float lateral_bias_toward_right_shoulder_m;
        }

        [Serializable]
        private sealed class ActivityPlanContract
        {
            public string schema;
            public float duration_s;
            public ActivityPhaseContract[] phases;
        }

        [Serializable]
        private sealed class ActivityPhaseContract
        {
            public string id;
            public float start_s;
            public float end_s;
        }

        [Serializable]
        private sealed class AuthorityContract
        {
            public int physics_hz;
            public int render_hz;
            public int steps_per_render_frame;
            public bool biological_torque_claim_permitted;
        }

        [Serializable]
        private sealed class QaTolerancesContract
        {
            public string schema;
            public QaClockContract clock;
            public QaRegistrationContract registration;
            public QaContactContract contact;
            public QaInteractionContract interaction;
            public QaCameraContract camera;
            public QaCaptureContract capture;
            public QaReplayContract replay;
        }

        [Serializable]
        private sealed class QaClockContract
        {
            public int physics_hz;
            public int render_hz;
            public int exact_steps_per_render_frame;
        }

        [Serializable]
        private sealed class QaRegistrationContract
        {
            public float skin_collider_max_m;
            public float garment_body_max_penetration_m;
            public float garment_affected_vertex_fraction_max;
            public float finger_object_max_penetration_m;
            public float support_max_penetration_m;
            public bool positive_camera_head_hair_garment_clearance_every_frame;
        }

        [Serializable]
        private sealed class QaInteractionContract
        {
            public float right_opposition_min_s;
            public float left_support_min_s;
            public float lift_min_m;
            public float turn_min_deg;
            public bool free_release_required;
            public bool unsupported_required;
            public bool support_continuous_until_commanded_open;
            public bool settling_required;
        }

        [Serializable]
        private sealed class QaContactContract
        {
            public float qualification_max_measured_separation_m;
            public bool simultaneous_nonzero_physx_impulse_required;
            public bool correct_visible_surface_projection_required;
            public int visible_physical_max_frame_delta;
        }

        [Serializable]
        private sealed class QaCameraContract
        {
            public int[] fov_candidates_deg;
            public float optical_vs_face_forward_max_deg;
            public float minimum_clearance_m;
            public float roll_abs_max_deg;
        }

        [Serializable]
        private sealed class QaCaptureContract
        {
            public int fps;
            public int[] resolution_px;
            public bool same_frozen_frame_required;
            public bool contact_projection_required;
            public bool zero_proxy_hero_pixels;
        }

        [Serializable]
        private sealed class QaReplayContract
        {
            public bool fresh_process_required;
            public bool same_trace_rerender_required;
            public float translation_max_m;
            public float rotation_max_deg;
            public float object_velocity_max_m_s;
            public float rerender_min_psnr_db;
        }

        [Serializable]
        private sealed class RobustnessVariantContract
        {
            public string id;
            public float target_lateral_shift_m;
            public float mass_scale;
            public float static_friction_scale;
            public float dynamic_friction_scale;
        }

        [Serializable]
        private sealed class ReplayConsumptionReceipt
        {
            public string schema;
            public string episode_id;
            public string contract_sha256;
            public string source_trace;
            public string source_trace_sha256;
            public int rows_consumed;
            public bool render_only;
            public bool physics_simulated;
            public bool trace_driven_target_writes;
            public bool eligible_manipulation_evidence;
            public string replayed_state_provenance;
            public float translation_max_m;
            public float rotation_max_deg;
            public float object_velocity_max_m_s;
        }

        [Serializable]
        private sealed class RegistrationPassReceipt
        {
            public string schema;
            public bool self_clearance_sampled_every_physics_step;
            public bool non_adjacent_anatomy_clearance_passed;
            public bool passed;
        }

        private sealed class FingerBodyReplayBinding
        {
            public Rigidbody body;
            public Transform bone;
            public Vector3 positionInBone;
            public Quaternion rotationInBone;
        }
    }
}
#endif
