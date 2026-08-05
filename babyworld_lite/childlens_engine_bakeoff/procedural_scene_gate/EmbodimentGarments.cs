#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace ProceduralSceneGate
{
    /// <summary>
    /// Builds the one visible MPFB body and catalog-selected garments.  This
    /// module never advances the skeleton: every renderer is bound to the exact
    /// Transform instances owned by the authoritative avatar hierarchy.
    /// </summary>
    public sealed class EmbodimentGarments : IEmbodimentGarmentModule
    {
        const string FrozenConfigName = "embodied_simulation_procedural_scene_gate.json";
        const string ExpectedAvatarId = "mpfb_child_cc0";
        const float AvatarScale = 1.9f;
        const float SkinColliderToleranceM = .0075f;
        const float GarmentBodyToleranceM = .004f;
        const float AffectedEpsilonM = 1e-5f;
        const int AnatomicalColliderLayer = 29;

        static readonly string[] Digits = { "thumb", "index", "middle", "ring", "little" };
        static readonly string[] MpfbDigits = { "finger1", "finger2", "finger3", "finger4", "finger5" };
        static readonly float[] GarmentPenetrationEdgesM = { .001f, .002f, .003f, .004f, .006f };
        static readonly float[] SkinColliderErrorEdgesM = { .001f, .003f, .005f, .0075f, .010f, .020f };

        GateContext boundContext;
        FrozenDocument frozen;
        AvatarSpec activeAvatarSpec;
        SkinnedMeshRenderer bodyRenderer;
        Mesh bodyBake;
        GameObject garmentRoot;
        readonly Dictionary<string, Transform> bones = new Dictionary<string, Transform>(StringComparer.Ordinal);
        readonly List<GarmentBinding> garments = new List<GarmentBinding>();
        readonly List<Collider> anatomicalColliders = new List<Collider>();
        readonly List<FittedColliderBinding> fittedColliderBindings = new List<FittedColliderBinding>();
        readonly List<KinematicColliderBinding> kinematicColliderBindings = new List<KinematicColliderBinding>();
        readonly Dictionary<int, List<Collider>> fittedCollidersByVertex = new Dictionary<int, List<Collider>>();
        readonly HashSet<Transform> fittedEnvelopeBones = new HashSet<Transform>();
        readonly Dictionary<Transform, Collider[]> collidersByDominantBone = new Dictionary<Transform, Collider[]>();
        Collider[][] registrationCandidatesByVertex;
        readonly List<RegistrationStep> registrationSteps = new List<RegistrationStep>();
        readonly Distribution skinColliderErrors = new Distribution(SkinColliderErrorEdgesM);
        readonly Distribution exactFittedSkinColliderErrors = new Distribution(SkinColliderErrorEdgesM);
        readonly Distribution fallbackSkinColliderErrors = new Distribution(SkinColliderErrorEdgesM);
        readonly Dictionary<string, Distribution> skinColliderErrorsByBone = new Dictionary<string, Distribution>(StringComparer.Ordinal);
        int lastSampledStep = -1;
        int auditedMotionSampleCount;
        int registrationEligibleVertexCount;
        int exactFittedRegistrationVertexCount;
        int fallbackRegistrationVertexCount;
        bool samplesContiguousFromZero = true;
        int moduleCreatedProxyRenderers;

        public void Build(GateContext context, string avatarAssetPath)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (context.AuthorityRoot == null) throw new InvalidOperationException("GateContext.AuthorityRoot must exist before embodiment build");
            if (string.IsNullOrWhiteSpace(avatarAssetPath)) throw new ArgumentException("avatar asset path is required", nameof(avatarAssetPath));
            if (context.AvatarRoot != null) throw new InvalidOperationException("the canonical gate permits one AvatarRoot");

            frozen = LoadFrozenDocument();
            ValidateFrozenCatalog(frozen);
            ValidateAvatarAsset(avatarAssetPath, frozen.assets.avatar);

            var sourcePrefab = AssetDatabase.LoadAssetAtPath<GameObject>(avatarAssetPath);
            if (sourcePrefab == null) throw new FileNotFoundException("Unity could not load the frozen MPFB avatar", avatarAssetPath);
            var avatar = (GameObject)PrefabUtility.InstantiatePrefab(sourcePrefab);
            if (avatar == null) throw new InvalidOperationException("Unity could not instantiate the frozen MPFB avatar");
            avatar.name = "CC0_WEIGHTED_MPFB_CHILD";
            avatar.transform.SetParent(context.AuthorityRoot.transform, false);
            avatar.transform.localPosition = Vector3.zero;
            // The curated room families place the reachable support in -Z.
            // Freeze the avatar's one initialization-time room orientation so
            // the anatomical head-forward axis faces the task zone.
            avatar.transform.localRotation = Quaternion.Euler(0f, 180f, 0f);
            avatar.transform.localScale = Vector3.one * AvatarScale;

            foreach (var animator in avatar.GetComponentsInChildren<Animator>(true)) animator.enabled = false;
            foreach (var animation in avatar.GetComponentsInChildren<Animation>(true)) animation.enabled = false;

            var weighted = avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            if (weighted.Length != 1) throw new InvalidOperationException("the frozen MPFB source must contain exactly one weighted body renderer");
            bodyRenderer = weighted[0];
            bodyRenderer.name = "MPFB_WEIGHTED_BODY_SKIN";
            bodyRenderer.enabled = true;
            bodyRenderer.updateWhenOffscreen = true;
            bodyRenderer.forceMatrixRecalculationPerRender = true;
            bodyRenderer.localBounds = new Bounds(Vector3.zero, Vector3.one * 4f);
            bodyRenderer.sharedMaterial = SkinMaterial();
            if (bodyRenderer.sharedMesh == null || bodyRenderer.sharedMesh.vertexCount == 0)
                throw new InvalidOperationException("the MPFB body has no weighted mesh");
            if (bodyRenderer.sharedMesh.boneWeights.Length != bodyRenderer.sharedMesh.vertexCount)
                throw new InvalidOperationException("the MPFB source must expose four-weight skinning for every body vertex");

            bones.Clear();
            foreach (var group in avatar.GetComponentsInChildren<Transform>(true).GroupBy(x => x.name)) {
                if (group.Count() != 1) throw new InvalidOperationException("duplicate MPFB bone name: " + group.Key);
                bones.Add(group.Key, group.Single());
            }
            BindAnatomy(context);
            VerifyWeightedFiveFingerHands();
            bodyBake = new Mesh { name = "MPFB_BODY_REGISTRATION_BAKE" };
            bodyRenderer.BakeMesh(bodyBake, true);
            BuildAnatomicalColliders();
            BindKinematicColliderDrivers(avatar.transform);

            garmentRoot = new GameObject("CATALOG_DRIVEN_SKINNED_GARMENTS");
            garmentRoot.transform.SetParent(bodyRenderer.transform.parent, false);
            garmentRoot.transform.localPosition = bodyRenderer.transform.localPosition;
            garmentRoot.transform.localRotation = bodyRenderer.transform.localRotation;
            garmentRoot.transform.localScale = bodyRenderer.transform.localScale;
            context.AvatarRoot = avatar;
            boundContext = context;
            moduleCreatedProxyRenderers = 0;
        }

        public void ApplyGarmentConfiguration(GateContext context, string configurationId)
        {
            RequireBoundContext(context);
            if (context.PhysicsStep != 0 && activeAvatarSpec != null)
                throw new InvalidOperationException("garments are frozen before the episode; mid-episode changes are forbidden");
            if (string.IsNullOrWhiteSpace(configurationId)) throw new ArgumentException("garment configuration id is required", nameof(configurationId));

            var matches = frozen.avatar_specs.Where(x => x.garment_configuration_id == configurationId).ToArray();
            if (matches.Length != 1) throw new InvalidOperationException("configuration must resolve to one frozen AvatarSpec: " + configurationId);
            var selected = matches[0];
            if (selected.garments == null || selected.garments.Length == 0)
                throw new InvalidOperationException("frozen AvatarSpec contains no garments: " + configurationId);

            ClearGeneratedGarments();
            foreach (var garment in selected.garments.OrderBy(x => x.layer).ThenBy(x => x.id, StringComparer.Ordinal))
                garments.Add(BuildGarment(garment));
            if (garments.Count != selected.garments.Length)
                throw new InvalidOperationException("not every catalog garment was instantiated");

            activeAvatarSpec = selected;
            context.GarmentConfigurationId = selected.garment_configuration_id;
            ResetRegistrationAccumulators();
        }

        /// <summary>
        /// Integration hook: call exactly once after each authoritative 240 Hz
        /// physics step and after the body followers have been updated.  It is
        /// observation-only and writes no Transform, drive, or Rigidbody state.
        /// </summary>
        public void SampleRegistrationAtPhysicsStep(GateContext context)
        {
            RequireBoundContext(context);
            if (activeAvatarSpec == null) throw new InvalidOperationException("apply a frozen garment configuration before sampling");
            if (context.PhysicsStep == lastSampledStep) return;
            if (lastSampledStep < 0) samplesContiguousFromZero = context.PhysicsStep == 0;
            else samplesContiguousFromZero &= context.PhysicsStep == lastSampledStep + 1;

            bool auditThisStep = context.PhysicsStep == 0 || (context.PhysicsStep + 1) % FrozenGate.StepsPerFrame == 0;
            if (!auditThisStep) {
                registrationSteps.Add(new RegistrationStep {
                    physics_step = context.PhysicsStep,
                    time_s = context.TimeSeconds,
                    audited = false,
                    body_collider = new BodyColliderStep { provenance = TruthSource.Unavailable.ToString(), samples = 0 },
                    garments = Array.Empty<GarmentStep>(),
                });
                lastSampledStep = context.PhysicsStep;
                return;
            }

            bodyRenderer.BakeMesh(bodyBake, true);
            var row = new RegistrationStep {
                physics_step = context.PhysicsStep,
                time_s = context.TimeSeconds,
                audited = true,
                body_collider = SampleBodyColliderRegistration(context),
                garments = garments.Select(SampleGarmentRegistration).ToArray(),
            };
            registrationSteps.Add(row);
            auditedMotionSampleCount++;
            lastSampledStep = context.PhysicsStep;
        }

        /// <summary>
        /// Refit the collider surface from the freshly commanded weighted skin
        /// before the same step is submitted to PhysX. This is a follower of the
        /// sole skeleton state, not an independent animation or object repair.
        /// </summary>
        public void UpdateRegisteredCollidersBeforePhysics(GateContext context)
        {
            RequireBoundContext(context);
            bodyRenderer.BakeMesh(bodyBake, true);
            Vector3[] vertices = bodyBake.vertices;
            foreach (FittedColliderBinding binding in fittedColliderBindings) {
                Vector3[] points = binding.vertexIndices
                    .Where(index => index >= 0 && index < vertices.Length)
                    .Select(index => bodyRenderer.transform.TransformPoint(vertices[index]))
                    .ToArray();
                if (points.Length == 0) continue;
                Vector3 center = points.Aggregate(Vector3.zero, (sum, point) => sum + point) / points.Length;
                float[] distances = points.Select(point => Vector3.Distance(point, center)).OrderBy(value => value).ToArray();
                float radiusWorld = Mathf.Max(.0015f, (distances[0] + distances[distances.Length - 1]) * .5f);
                binding.collider.transform.position = center;
                binding.collider.transform.rotation = binding.bone.rotation;
                binding.collider.center = Vector3.zero;
                binding.collider.radius = radiusWorld / UniformScale(binding.collider.transform);
            }
            foreach (KinematicColliderBinding binding in kinematicColliderBindings) {
                Vector3 desiredPosition = binding.bone.TransformPoint(binding.positionInBone);
                Quaternion desiredRotation = binding.bone.rotation * binding.rotationInBone;
                if (context.AnatomicalColliderVelocityDriveCommanded) {
                    binding.body.MovePosition(desiredPosition);
                    binding.body.MoveRotation(desiredRotation);
                } else {
                    binding.body.position = desiredPosition;
                    binding.body.rotation = desiredRotation;
                }
            }
        }

        public void MeasureRegistration(GateContext context, string reportPath)
        {
            RequireBoundContext(context);
            if (string.IsNullOrWhiteSpace(reportPath)) throw new ArgumentException("registration report path is required", nameof(reportPath));
            if (lastSampledStep != context.PhysicsStep) SampleRegistrationAtPhysicsStep(context);

            int expectedSamples = Mathf.RoundToInt(FrozenGate.DurationSeconds * FrozenGate.PhysicsHz);
            bool completeMotion = samplesContiguousFromZero && registrationSteps.Count == expectedSamples &&
                                  registrationSteps[0].physics_step == 0 &&
                                  registrationSteps[registrationSteps.Count - 1].physics_step == expectedSamples - 1;
            bool colliderAvailable = skinColliderErrors.count > 0;
            var garmentReceipts = garments.Select(x => x.Receipt()).ToArray();
            bool garmentAvailable = garmentReceipts.Length > 0 && garmentReceipts.All(x => x.samples > 0);
            bool bodyPass = colliderAvailable && skinColliderErrors.maximum_m <= SkinColliderToleranceM;
            bool garmentPass = garmentAvailable && garmentReceipts.All(x => x.maximum_penetration_m <= GarmentBodyToleranceM);

            var report = new EmbodimentRegistrationReport {
                schema = "embodied.embodiment_registration.v1",
                avatar_spec_schema = activeAvatarSpec.schema,
                embodiment_manifest_schema = "embodied.embodiment_manifest.v1",
                avatar_id = activeAvatarSpec.avatar_id,
                garment_configuration_id = activeAvatarSpec.garment_configuration_id,
                source_asset_id = frozen.assets.avatar.asset_id,
                source_asset_sha256 = frozen.assets.avatar.sha256,
                source_license = frozen.assets.avatar.license,
                construction_license = frozen.assets.procedural_garments.license,
                construction_method = frozen.assets.procedural_garments.construction,
                authority_root = context.AuthorityRoot.name,
                avatar_root = context.AvatarRoot.name,
                body_renderer = bodyRenderer.name,
                body_bones = bodyRenderer.bones.Select(x => x.name).ToArray(),
                anatomical_collider_ids = anatomicalColliders.Select(x => x.name).OrderBy(x => x, StringComparer.Ordinal).ToArray(),
                anatomical_collider_count = anatomicalColliders.Count,
                kinematic_collider_body_count = anatomicalColliders.Count(x => x.attachedRigidbody != null && x.attachedRigidbody.isKinematic),
                registration_eligible_vertex_count = registrationEligibleVertexCount,
                exact_fitted_registration_vertex_count = exactFittedRegistrationVertexCount,
                fallback_registration_vertex_count = fallbackRegistrationVertexCount,
                registration_candidate_coverage_by_bone = RegistrationCandidateCoverage(),
                both_palms_and_all_finger_segments_have_colliders = CompleteHandColliderSet(),
                hand_topology = activeAvatarSpec.hand_topology,
                garment_renderers = garments.Select(x => x.renderer.name).ToArray(),
                garment_layers = garments.Select(x => x.spec.layer).ToArray(),
                garment_fit_offsets_m = garments.Select(x => x.spec.fit_offset_m).ToArray(),
                garment_materials = garments.Select(x => x.spec.material).ToArray(),
                body_collider_provenance = colliderAvailable ? TruthSource.EngineObserved.ToString() : TruthSource.Unavailable.ToString(),
                body_collider_unavailable_reason = colliderAvailable ? null : "no supported avatar collider could be matched to a weighted MPFB bone",
                body_collider_registration = skinColliderErrors.Receipt(),
                exact_fitted_body_collider_registration = exactFittedSkinColliderErrors.Receipt(),
                fallback_body_collider_registration = fallbackSkinColliderErrors.Receipt(),
                body_collider_registration_by_bone = skinColliderErrorsByBone
                    .OrderBy(row => row.Key, StringComparer.Ordinal)
                    .Select(row => new BoneRegistrationReceipt { bone = row.Key, distribution = row.Value.Receipt() })
                    .ToArray(),
                garment_body_provenance = garmentAvailable ? TruthSource.EngineObserved.ToString() : TruthSource.Unavailable.ToString(),
                garment_body_method = "corresponding weighted surface vertices; signed clearance along the baked MPFB body normal after each authoritative physics step",
                garment_body_full_triangle_intersection_provenance = TruthSource.Unavailable.ToString(),
                garment_body_full_triangle_intersection_unavailable_reason = "the canonical gate does not fabricate a robust skinned triangle-triangle intersection result; signed correspondence penetration is reported",
                garment_self_intersection_provenance = TruthSource.Unavailable.ToString(),
                garment_self_intersection_unavailable_reason = "no validated robust skinned self-intersection solver is present",
                garments = garmentReceipts,
                motion_sample_count = registrationSteps.Count,
                audited_motion_sample_count = auditedMotionSampleCount,
                registration_audit_stride_physics_steps = FrozenGate.StepsPerFrame,
                expected_motion_sample_count = expectedSamples,
                first_physics_step = registrationSteps.Count == 0 ? -1 : registrationSteps[0].physics_step,
                last_physics_step = registrationSteps.Count == 0 ? -1 : registrationSteps[registrationSteps.Count - 1].physics_step,
                sampled_every_physics_step = completeMotion,
                registration_steps = registrationSteps.ToArray(),
                module_created_proxy_renderers = moduleCreatedProxyRenderers,
                independently_advanced_animation = false,
                body_collider_tolerance_m = SkinColliderToleranceM,
                garment_body_tolerance_m = GarmentBodyToleranceM,
                passed = completeMotion && bodyPass && garmentPass && CompleteHandColliderSet() && moduleCreatedProxyRenderers == 0,
            };

            string directory = Path.GetDirectoryName(Path.GetFullPath(reportPath));
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
            File.WriteAllText(reportPath, JsonUtility.ToJson(report, true));
        }

        GarmentBinding BuildGarment(GarmentSpec spec)
        {
            ValidateGarmentSpec(spec);
            Mesh source = bodyRenderer.sharedMesh;
            Vector3[] sourceVertices = source.vertices;
            Vector3[] normals = source.normals;
            if (normals == null || normals.Length != sourceVertices.Length)
                throw new InvalidOperationException("the frozen MPFB source requires bind-pose normals for fitted garments");
            BoneWeight[] weights = source.boneWeights;

            bool[] eligible = new bool[source.vertexCount];
            for (int i = 0; i < eligible.Length; i++) eligible[i] = GarmentWeight(spec.id, weights[i]) >= .35f;
            var triangles = new List<int>();
            for (int submesh = 0; submesh < source.subMeshCount; submesh++) {
                int[] sourceTriangles = source.GetTriangles(submesh);
                for (int i = 0; i + 2 < sourceTriangles.Length; i += 3) {
                    int a = sourceTriangles[i], b = sourceTriangles[i + 1], c = sourceTriangles[i + 2];
                    if ((eligible[a] ? 1 : 0) + (eligible[b] ? 1 : 0) + (eligible[c] ? 1 : 0) < 2) continue;
                    triangles.Add(a); triangles.Add(b); triangles.Add(c);
                }
            }
            if (triangles.Count == 0) throw new InvalidOperationException("catalog garment selected no weighted source triangles: " + spec.id);

            var vertices = (Vector3[])sourceVertices.Clone();
            for (int i = 0; i < vertices.Length; i++) {
                Vector3 worldNormal = bodyRenderer.transform.TransformDirection(normals[i]).normalized;
                vertices[i] += bodyRenderer.transform.InverseTransformVector(worldNormal * spec.fit_offset_m);
            }
            var mesh = new Mesh {
                name = "BIND_DERIVED_" + spec.id,
                indexFormat = source.indexFormat,
                vertices = vertices,
                normals = (Vector3[])normals.Clone(),
                tangents = source.tangents,
                colors = source.colors,
                uv = source.uv,
                uv2 = source.uv2,
                uv3 = source.uv3,
                uv4 = source.uv4,
                boneWeights = source.boneWeights,
                bindposes = source.bindposes,
                bounds = source.bounds,
                subMeshCount = 1,
            };
            mesh.SetTriangles(triangles, 0, true);

            var go = new GameObject("GARMENT_" + spec.id + "_LAYER_" + spec.layer);
            go.transform.SetParent(garmentRoot.transform, false);
            var renderer = go.AddComponent<SkinnedMeshRenderer>();
            renderer.name = "SKINNED_" + spec.id;
            renderer.sharedMesh = mesh;
            renderer.bones = bodyRenderer.bones;
            renderer.rootBone = bodyRenderer.rootBone;
            renderer.updateWhenOffscreen = true;
            renderer.forceMatrixRecalculationPerRender = true;
            renderer.localBounds = bodyRenderer.localBounds;
            renderer.sortingOrder = spec.layer;
            renderer.sharedMaterial = GarmentMaterial(spec);

            if (renderer.sharedMesh.bindposes.Length != source.bindposes.Length ||
                renderer.sharedMesh.boneWeights.Length != source.boneWeights.Length ||
                !renderer.bones.SequenceEqual(bodyRenderer.bones))
                throw new InvalidOperationException("garment did not preserve the exact MPFB skin binding: " + spec.id);

            return new GarmentBinding {
                spec = spec,
                renderer = renderer,
                baked = new Mesh { name = spec.id + "_REGISTRATION_BAKE" },
                usedVertexIndices = triangles.Distinct().OrderBy(x => x).ToArray(),
                penetration = new Distribution(GarmentPenetrationEdgesM),
                minimumClearanceM = float.PositiveInfinity,
            };
        }

        BodyColliderStep SampleBodyColliderRegistration(GateContext context)
        {
            BoneWeight[] weights = bodyRenderer.sharedMesh.boneWeights;
            Vector3[] vertices = bodyBake.vertices;
            int observed = 0;
            float maximum = 0;
            for (int i = 0; i < vertices.Length && i < weights.Length; i++) {
                int index = DominantBone(weights[i], out float weight);
                if (weight < .60f || index < 0 || index >= bodyRenderer.bones.Length) continue;
                Transform bone = bodyRenderer.bones[index];
                Collider[] candidates = registrationCandidatesByVertex != null && i < registrationCandidatesByVertex.Length
                    ? registrationCandidatesByVertex[i] : null;
                if (candidates == null || candidates.Length == 0) continue;
                Vector3 world = bodyRenderer.transform.TransformPoint(vertices[i]);
                float error = candidates.Select(x => SurfaceDistance(x, world)).Where(float.IsFinite).DefaultIfEmpty(float.NaN).Min();
                if (!float.IsFinite(error)) continue;
                skinColliderErrors.Observe(error, error > SkinColliderToleranceM);
                if (fittedCollidersByVertex.ContainsKey(i)) exactFittedSkinColliderErrors.Observe(error, error > SkinColliderToleranceM);
                else fallbackSkinColliderErrors.Observe(error, error > SkinColliderToleranceM);
                if (!skinColliderErrorsByBone.TryGetValue(bone.name, out Distribution byBone)) {
                    byBone = new Distribution(SkinColliderErrorEdgesM);
                    skinColliderErrorsByBone.Add(bone.name, byBone);
                }
                byBone.Observe(error, error > SkinColliderToleranceM);
                maximum = Mathf.Max(maximum, error);
                observed++;
            }
            return new BodyColliderStep {
                provenance = observed > 0 ? TruthSource.EngineObserved.ToString() : TruthSource.Unavailable.ToString(),
                samples = observed,
                maximum_error_m = observed > 0 ? maximum : 0,
            };
        }

        GarmentStep SampleGarmentRegistration(GarmentBinding garment)
        {
            garment.renderer.BakeMesh(garment.baked, true);
            Vector3[] bodyVertices = bodyBake.vertices;
            Vector3[] bodyNormals = bodyBake.normals;
            Vector3[] garmentVertices = garment.baked.vertices;
            int affected = 0;
            float maximum = 0;
            float minimumClearance = float.PositiveInfinity;
            foreach (int i in garment.usedVertexIndices) {
                if (i >= bodyVertices.Length || i >= bodyNormals.Length || i >= garmentVertices.Length) continue;
                Vector3 bodyWorld = bodyRenderer.transform.TransformPoint(bodyVertices[i]);
                Vector3 normalWorld = bodyRenderer.transform.TransformDirection(bodyNormals[i]).normalized;
                Vector3 garmentWorld = garment.renderer.transform.TransformPoint(garmentVertices[i]);
                float signedClearance = Vector3.Dot(garmentWorld - bodyWorld, normalWorld);
                float penetration = Mathf.Max(0, -signedClearance);
                garment.penetration.Observe(penetration, penetration > AffectedEpsilonM);
                garment.minimumClearanceM = Mathf.Min(garment.minimumClearanceM, signedClearance);
                minimumClearance = Mathf.Min(minimumClearance, signedClearance);
                maximum = Mathf.Max(maximum, penetration);
                if (penetration > AffectedEpsilonM) affected++;
            }
            int samples = garment.usedVertexIndices.Length;
            return new GarmentStep {
                garment_id = garment.spec.id,
                provenance = samples > 0 ? TruthSource.EngineObserved.ToString() : TruthSource.Unavailable.ToString(),
                samples = samples,
                affected_vertices = affected,
                affected_fraction = samples > 0 ? affected / (float)samples : 0,
                minimum_signed_clearance_m = float.IsFinite(minimumClearance) ? minimumClearance : 0,
                maximum_penetration_m = maximum,
            };
        }

        void BindAnatomy(GateContext context)
        {
            context.Torso = FindFirstRequired("spine03", "spine02", "spine01");
            context.Neck = FindFirstRequired("neck", "neck01");
            context.Head = FindRequired("head");
            context.LeftPalm = FindRequired("wrist.L");
            context.RightPalm = FindRequired("wrist.R");
            context.FingerSegments.Clear();
            for (int side = 0; side < 2; side++) {
                string suffix = side == 0 ? ".L" : ".R";
                string semanticSide = side == 0 ? "left_" : "right_";
                for (int digit = 0; digit < Digits.Length; digit++) {
                    var segments = new Transform[3];
                    for (int segment = 1; segment <= 3; segment++)
                        segments[segment - 1] = FindRequired(MpfbDigits[digit] + "-" + segment + suffix);
                    context.FingerSegments.Add(semanticSide + Digits[digit], segments);
                }
            }
        }

        void VerifyWeightedFiveFingerHands()
        {
            var rendererBones = bodyRenderer.bones;
            var weights = bodyRenderer.sharedMesh.boneWeights;
            var required = new List<Transform> { FindRequired("wrist.L"), FindRequired("wrist.R") };
            foreach (string suffix in new[] { ".L", ".R" })
                for (int digit = 0; digit < MpfbDigits.Length; digit++)
                    for (int segment = 1; segment <= 3; segment++)
                        required.Add(FindRequired(MpfbDigits[digit] + "-" + segment + suffix));
            if (required.Distinct().Count() != 32) throw new InvalidOperationException("the two hand hierarchies contain fused Transform instances");
            foreach (Transform requiredBone in required) {
                int index = Array.IndexOf(rendererBones, requiredBone);
                if (index < 0 || !weights.Any(x => BoneWeightFor(x, index) > 0))
                    throw new InvalidOperationException("anatomical hand bone has no MPFB skin weights: " + requiredBone.name);
            }
        }

        void BuildAnatomicalColliders()
        {
            anatomicalColliders.Clear();
            fittedColliderBindings.Clear();
            fittedCollidersByVertex.Clear();
            fittedEnvelopeBones.Clear();
            Physics.IgnoreLayerCollision(AnatomicalColliderLayer, AnatomicalColliderLayer, true);
            PhysicsMaterial handMaterial = ColliderMaterial("MPFB_HAND_HIGH_FRICTION", 1f, 1f);
            PhysicsMaterial bodyMaterial = ColliderMaterial("MPFB_BODY_MATERIAL", .65f, .75f);

            foreach (string suffix in new[] { ".L", ".R" }) {
                string side = suffix == ".L" ? "left" : "right";
                BuildPalmBox(side, suffix, handMaterial);
                AddBoneCapsule(side + "_upper_arm_1", FindRequired("upperarm01" + suffix), FindRequired("upperarm02" + suffix), .018f, .060f, bodyMaterial);
                AddBoneCapsule(side + "_upper_arm_2", FindRequired("upperarm02" + suffix), FindRequired("lowerarm01" + suffix), .018f, .055f, bodyMaterial);
                AddBoneCapsule(side + "_forearm_1", FindRequired("lowerarm01" + suffix), FindRequired("lowerarm02" + suffix), .014f, .050f, bodyMaterial);
                AddBoneCapsule(side + "_forearm_2", FindRequired("lowerarm02" + suffix), FindRequired("wrist" + suffix), .012f, .045f, bodyMaterial);
                foreach (string armBone in new[] { "upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist" })
                    AddFittedEnvelopeSpheres(FindRequired(armBone + suffix), bodyMaterial);

                for (int digit = 0; digit < Digits.Length; digit++) {
                    for (int segment = 1; segment <= 3; segment++) {
                        Transform bone = FindRequired(MpfbDigits[digit] + "-" + segment + suffix);
                        AddFittedEnvelopeSpheres(bone, handMaterial);
                        Transform endpoint;
                        if (segment < 3) endpoint = FindRequired(MpfbDigits[digit] + "-" + (segment + 1) + suffix);
                        else {
                            Transform previous = FindRequired(MpfbDigits[digit] + "-2" + suffix);
                            endpoint = null;
                            Vector3 distal = bone.position + (bone.position - previous.position) * .68f;
                            AddBoneCapsule(side + "_" + Digits[digit] + "_segment_" + segment, bone, distal, .0038f, digit == 0 ? .012f : .0095f, handMaterial);
                            continue;
                        }
                        AddBoneCapsule(side + "_" + Digits[digit] + "_segment_" + segment, bone, endpoint, .0038f, digit == 0 ? .012f : .0095f, handMaterial);
                    }
                }
            }

            // These fitted envelopes make collider/skin registration cover the
            // rest of the clothed body without introducing a second rig.
            foreach (Transform bone in bodyRenderer.bones.Distinct()) {
                AddFittedEnvelopeSpheres(bone, bodyMaterial);
            }
            AddMissingRegistrationEnvelopeSpheres(bodyMaterial);

            if (!CompleteHandColliderSet())
                throw new InvalidOperationException("anatomical collider build did not cover both palms and all thirty finger segments");

            // Registration is sampled densely. Resolve the semantic bone-to-
            // collider relationship once instead of repeating name parsing and
            // LINQ allocations for every weighted vertex at every physics step.
            collidersByDominantBone.Clear();
            foreach (Transform bone in bodyRenderer.bones.Distinct()) {
                Collider[] candidates = anatomicalColliders.Where(x => x && x.enabled && !x.isTrigger && ColliderMatchesBone(x, bone)).ToArray();
                if (candidates.Length > 0) collidersByDominantBone.Add(bone, candidates);
            }
            BuildVertexRegistrationCandidateCache();
        }

        void BindKinematicColliderDrivers(Transform avatarRoot)
        {
            kinematicColliderBindings.Clear();
            var root = new GameObject("PHYSICS_ANATOMICAL_COLLIDER_DRIVERS");
            root.transform.SetParent(avatarRoot, false);
            foreach (Collider collider in anatomicalColliders.Where(value => value && !value.isTrigger)) {
                Transform bone = collider.transform.parent;
                Vector3 positionInBone = bone.InverseTransformPoint(collider.transform.position);
                Quaternion rotationInBone = Quaternion.Inverse(bone.rotation) * collider.transform.rotation;
                Rigidbody body = collider.attachedRigidbody;
                collider.transform.SetParent(root.transform, true);
                kinematicColliderBindings.Add(new KinematicColliderBinding {
                    body = body,
                    bone = bone,
                    positionInBone = positionInBone,
                    rotationInBone = rotationInBone,
                });
            }
        }

        void BuildPalmBox(string side, string suffix, PhysicsMaterial material)
        {
            Transform wrist = FindRequired("wrist" + suffix);
            var indices = new List<int> { Array.IndexOf(bodyRenderer.bones, wrist) };
            for (int digit = 0; digit < MpfbDigits.Length; digit++)
                indices.Add(Array.IndexOf(bodyRenderer.bones, FindRequired(MpfbDigits[digit] + "-1" + suffix)));
            Vector3[] points = WeightedPoints(indices.Where(x => x >= 0).ToArray(), .18f).Select(wrist.InverseTransformPoint).ToArray();
            if (points.Length < 8) throw new InvalidOperationException("insufficient weighted MPFB palm vertices: " + side);
            Bounds bounds = BoundsFor(points);
            bounds.Expand(.002f / UniformScale(wrist));
            var go = ColliderObject("COLLIDER_" + side + "_palm", wrist, wrist.TransformPoint(bounds.center), wrist.rotation);
            var collider = go.AddComponent<BoxCollider>();
            collider.center = Vector3.zero;
            collider.size = new Vector3(Mathf.Max(bounds.size.x, .020f / UniformScale(wrist)), Mathf.Max(bounds.size.y, .020f / UniformScale(wrist)), Mathf.Max(bounds.size.z, .020f / UniformScale(wrist)));
            collider.material = material;
            anatomicalColliders.Add(collider);
        }

        void AddBoneCapsule(string semanticId, Transform bone, Transform endpoint, float minimumRadiusM, float maximumRadiusM, PhysicsMaterial material)
            => AddBoneCapsule(semanticId, bone, endpoint.position, minimumRadiusM, maximumRadiusM, material);

        void AddBoneCapsule(string semanticId, Transform bone, Vector3 endpointWorld, float minimumRadiusM, float maximumRadiusM, PhysicsMaterial material)
        {
            Vector3 start = bone.position;
            Vector3 delta = endpointWorld - start;
            if (delta.sqrMagnitude < 1e-8f) throw new InvalidOperationException("zero-length anatomical segment: " + semanticId);
            Vector3 midpoint = (start + endpointWorld) * .5f;
            Quaternion worldRotation = Quaternion.FromToRotation(Vector3.up, delta.normalized);
            int boneIndex = Array.IndexOf(bodyRenderer.bones, bone);
            Vector3[] skinPoints = boneIndex < 0 ? Array.Empty<Vector3>() : WeightedPoints(new[] { boneIndex }, .15f);
            float radiusWorld = EstimateRadialRadius(skinPoints, start, endpointWorld, minimumRadiusM, maximumRadiusM);
            var go = ColliderObject("COLLIDER_" + semanticId, bone, midpoint, worldRotation);
            float scale = UniformScale(go.transform);
            var collider = go.AddComponent<CapsuleCollider>();
            collider.direction = 1;
            collider.center = Vector3.zero;
            collider.radius = radiusWorld / scale;
            collider.height = Mathf.Max(2f * collider.radius, delta.magnitude / scale);
            collider.material = material;
            anatomicalColliders.Add(collider);
        }

        void AddFittedEnvelopeSpheres(Transform bone, PhysicsMaterial material)
        {
            if (!fittedEnvelopeBones.Add(bone)) return;
            int[] boneIndices = bodyRenderer.bones
                .Select((candidate, index) => new { candidate, index })
                .Where(row => row.candidate == bone)
                .Select(row => row.index)
                .ToArray();
            if (boneIndices.Length == 0) return;
            Mesh mesh = bodyRenderer.sharedMesh;
            BoneWeight[] weights = mesh.boneWeights;
            Vector3[] vertices = bodyBake != null && bodyBake.vertexCount == weights.Length
                ? bodyBake.vertices
                : mesh.vertices;
            var indexed = new List<IndexedLocalPoint>();
            for (int index = 0; index < weights.Length && index < vertices.Length; index++) {
                if (boneIndices.Sum(boneIndex => BoneWeightFor(weights[index], boneIndex)) < .55f) continue;
                Vector3 world = bodyRenderer.transform.TransformPoint(vertices[index]);
                indexed.Add(new IndexedLocalPoint { vertexIndex = index, local = bone.InverseTransformPoint(world) });
            }
            if (indexed.Count < 6) return;
            float cellSize = .006f / UniformScale(bone);
            Dictionary<Vector3Int, List<IndexedLocalPoint>> cells = indexed
                .GroupBy(point => new Vector3Int(
                        Mathf.FloorToInt(point.local.x / cellSize),
                        Mathf.FloorToInt(point.local.y / cellSize),
                        Mathf.FloorToInt(point.local.z / cellSize)))
                .ToDictionary(group => group.Key, group => group.ToList());
            int group = 0;
            foreach (List<IndexedLocalPoint> points in cells.OrderBy(row => row.Key.x).ThenBy(row => row.Key.y).ThenBy(row => row.Key.z).Select(row => row.Value)) {
                CreateFittedEnvelopeSphere(bone, points, material, "envelope_" + group);
                group++;
            }
        }

        void AddMissingRegistrationEnvelopeSpheres(PhysicsMaterial material)
        {
            BoneWeight[] weights = bodyRenderer.sharedMesh.boneWeights;
            Vector3[] vertices = bodyBake.vertices;
            var missing = new List<MissingFittedPoint>();
            for (int index = 0; index < weights.Length && index < vertices.Length; index++) {
                int boneIndex = DominantBone(weights[index], out float weight);
                if (weight < .60f || boneIndex < 0 || boneIndex >= bodyRenderer.bones.Length || fittedCollidersByVertex.ContainsKey(index)) continue;
                Transform bone = bodyRenderer.bones[boneIndex];
                Vector3 world = bodyRenderer.transform.TransformPoint(vertices[index]);
                missing.Add(new MissingFittedPoint { bone = bone, point = new IndexedLocalPoint { vertexIndex = index, local = bone.InverseTransformPoint(world) } });
            }
            foreach (IGrouping<Transform, MissingFittedPoint> boneGroup in missing.GroupBy(row => row.bone)) {
                Transform bone = boneGroup.Key;
                float cellSize = .006f / UniformScale(bone);
                var cells = boneGroup.GroupBy(row => new Vector3Int(
                    Mathf.FloorToInt(row.point.local.x / cellSize),
                    Mathf.FloorToInt(row.point.local.y / cellSize),
                    Mathf.FloorToInt(row.point.local.z / cellSize)));
                int group = 0;
                foreach (var cell in cells.OrderBy(row => row.Key.x).ThenBy(row => row.Key.y).ThenBy(row => row.Key.z)) {
                    CreateFittedEnvelopeSphere(bone, cell.Select(row => row.point).ToList(), material, "registration_repair_" + group);
                    group++;
                }
            }
        }

        void CreateFittedEnvelopeSphere(Transform bone, List<IndexedLocalPoint> points, PhysicsMaterial material, string suffix)
        {
            Vector3[] sample = points.Select(point => point.local).ToArray();
            Vector3 center = sample.Aggregate(Vector3.zero, (sum, point) => sum + point) / sample.Length;
            float[] distances = sample.Select(point => Vector3.Distance(point, center)).OrderBy(x => x).ToArray();
            float radius = Mathf.Max(.0015f / UniformScale(bone), (distances[0] + distances[distances.Length - 1]) * .5f);
            var go = ColliderObject("COLLIDER_" + SemanticBoneName(bone.name) + "_" + suffix, bone, bone.TransformPoint(center), bone.rotation);
            var collider = go.AddComponent<SphereCollider>();
            collider.center = Vector3.zero;
            collider.radius = radius;
            // Dense fitted spheres are query geometry for skin-registration and
            // labeled overlays.  Force-producing contact is owned once by the
            // anatomical palm boxes and segment capsules created above.
            collider.isTrigger = true;
            collider.material = material;
            anatomicalColliders.Add(collider);
            fittedColliderBindings.Add(new FittedColliderBinding {
                collider = collider,
                bone = bone,
                vertexIndices = points.Select(point => point.vertexIndex).ToArray(),
            });
            foreach (IndexedLocalPoint point in points) {
                if (!fittedCollidersByVertex.TryGetValue(point.vertexIndex, out List<Collider> vertexColliders)) {
                    vertexColliders = new List<Collider>();
                    fittedCollidersByVertex.Add(point.vertexIndex, vertexColliders);
                }
                vertexColliders.Add(collider);
            }
        }

        void BuildVertexRegistrationCandidateCache()
        {
            Mesh mesh = bodyRenderer.sharedMesh;
            Vector3[] vertices = mesh.vertices;
            BoneWeight[] weights = mesh.boneWeights;
            registrationCandidatesByVertex = new Collider[vertices.Length][];
            registrationEligibleVertexCount = 0;
            exactFittedRegistrationVertexCount = 0;
            fallbackRegistrationVertexCount = 0;
            for (int i = 0; i < vertices.Length && i < weights.Length; i++) {
                int boneIndex = DominantBone(weights[i], out float weight);
                if (weight < .60f || boneIndex < 0 || boneIndex >= bodyRenderer.bones.Length) continue;
                registrationEligibleVertexCount++;
                Transform bone = bodyRenderer.bones[boneIndex];
                if (fittedCollidersByVertex.TryGetValue(i, out List<Collider> exactFitted) && exactFitted.Count > 0) {
                    registrationCandidatesByVertex[i] = exactFitted.ToArray();
                    exactFittedRegistrationVertexCount++;
                    continue;
                }
                if (!collidersByDominantBone.TryGetValue(bone, out Collider[] candidates) || candidates.Length == 0) continue;
                Vector3 world = bodyRenderer.transform.TransformPoint(vertices[i]);
                registrationCandidatesByVertex[i] = candidates
                    .OrderBy(collider => SurfaceDistance(collider, world))
                    .Take(3)
                    .ToArray();
                fallbackRegistrationVertexCount++;
            }
        }

        CandidateCoverageReceipt[] RegistrationCandidateCoverage()
        {
            BoneWeight[] weights = bodyRenderer.sharedMesh.boneWeights;
            return Enumerable.Range(0, weights.Length)
                .Select(index => new { index, boneIndex = DominantBone(weights[index], out float weight), weight })
                .Where(row => row.weight >= .60f && row.boneIndex >= 0 && row.boneIndex < bodyRenderer.bones.Length)
                .GroupBy(row => bodyRenderer.bones[row.boneIndex].name)
                .OrderBy(group => group.Key, StringComparer.Ordinal)
                .Select(group => new CandidateCoverageReceipt {
                    bone = group.Key,
                    eligible = group.Count(),
                    exact_fitted = group.Count(row => fittedCollidersByVertex.ContainsKey(row.index)),
                    fallback = group.Count(row => !fittedCollidersByVertex.ContainsKey(row.index) && registrationCandidatesByVertex[row.index] != null),
                    unavailable = group.Count(row => registrationCandidatesByVertex[row.index] == null),
                })
                .ToArray();
        }

        Vector3[] WeightedPoints(int[] boneIndices, float minimumCombinedWeight)
        {
            Mesh mesh = bodyRenderer.sharedMesh;
            BoneWeight[] weights = mesh.boneWeights;
            Vector3[] vertices = mesh.vertices;
            var selected = new List<Vector3>();
            for (int i = 0; i < weights.Length && i < vertices.Length; i++) {
                float combined = boneIndices.Sum(index => BoneWeightFor(weights[i], index));
                if (combined >= minimumCombinedWeight) selected.Add(bodyRenderer.transform.TransformPoint(vertices[i]));
            }
            return selected.ToArray();
        }

        static float EstimateRadialRadius(Vector3[] points, Vector3 start, Vector3 end, float minimum, float maximum)
        {
            if (points == null || points.Length == 0) return minimum;
            float[] radial = points.Select(point => DistanceToSegment(point, start, end)).OrderBy(x => x).ToArray();
            return Mathf.Clamp(Quantile(radial, .72f), minimum, maximum);
        }

        static Bounds BoundsFor(Vector3[] points)
        {
            if (points == null || points.Length == 0) throw new ArgumentException("points are required");
            var bounds = new Bounds(points[0], Vector3.zero);
            foreach (Vector3 point in points) bounds.Encapsulate(point);
            return bounds;
        }

        static float Quantile(float[] sorted, float q)
        {
            if (sorted == null || sorted.Length == 0) return 0;
            return sorted[Mathf.Clamp(Mathf.RoundToInt((sorted.Length - 1) * q), 0, sorted.Length - 1)];
        }

        static float UniformScale(Transform value)
        {
            Vector3 scale = Abs(value.lossyScale);
            return Mathf.Max(1e-6f, Mathf.Max(scale.x, Mathf.Max(scale.y, scale.z)));
        }

        static GameObject ColliderObject(string name, Transform parent, Vector3 worldPosition, Quaternion worldRotation)
        {
            var go = new GameObject(name);
            go.layer = AnatomicalColliderLayer;
            go.transform.SetPositionAndRotation(worldPosition, worldRotation);
            go.transform.SetParent(parent, true);
            var body = go.AddComponent<Rigidbody>();
            body.isKinematic = true;
            body.useGravity = false;
            body.interpolation = RigidbodyInterpolation.None;
            body.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
            return go;
        }

        static PhysicsMaterial ColliderMaterial(string name, float dynamicFriction, float staticFriction)
        {
            return new PhysicsMaterial {
                name = name,
                dynamicFriction = dynamicFriction,
                staticFriction = staticFriction,
                frictionCombine = PhysicsMaterialCombine.Maximum,
                bounceCombine = PhysicsMaterialCombine.Minimum,
                bounciness = 0,
            };
        }

        bool CompleteHandColliderSet()
        {
            var names = new HashSet<string>(anatomicalColliders.Select(x => x.name), StringComparer.Ordinal);
            foreach (string side in new[] { "left", "right" }) {
                if (!names.Contains("COLLIDER_" + side + "_palm")) return false;
                foreach (string digit in Digits)
                    for (int segment = 1; segment <= 3; segment++)
                        if (!names.Contains("COLLIDER_" + side + "_" + digit + "_segment_" + segment)) return false;
            }
            return true;
        }

        float GarmentWeight(string garmentId, BoneWeight weight)
        {
            float total = 0;
            AddIfRegion(ref total, garmentId, weight.boneIndex0, weight.weight0);
            AddIfRegion(ref total, garmentId, weight.boneIndex1, weight.weight1);
            AddIfRegion(ref total, garmentId, weight.boneIndex2, weight.weight2);
            AddIfRegion(ref total, garmentId, weight.boneIndex3, weight.weight3);
            return total;
        }

        void AddIfRegion(ref float total, string garmentId, int boneIndex, float weight)
        {
            if (weight <= 0 || boneIndex < 0 || boneIndex >= bodyRenderer.bones.Length) return;
            string name = bodyRenderer.bones[boneIndex].name.ToLowerInvariant();
            bool torso = ContainsAny(name, "root", "spine", "pelvis", "hip", "clavicle", "shoulder", "breast");
            bool upperArm = ContainsAny(name, "upperarm", "shoulder");
            bool lowerArm = ContainsAny(name, "lowerarm", "forearm");
            bool leg = ContainsAny(name, "upperleg", "lowerleg", "thigh", "shin", "calf", "knee", "pelvis", "hip");
            bool foot = ContainsAny(name, "foot", "toe", "ankle");
            bool include;
            switch (garmentId) {
                case "long_sleeve_top": include = torso || upperArm || lowerArm; break;
                case "short_sleeve_top": include = torso || upperArm; break;
                case "contrast_vest": include = torso && !upperArm && !lowerArm; break;
                case "soft_trousers": include = leg && !foot; break;
                case "ankle_socks": include = foot || name.Contains("lowerleg"); break;
                case "play_overall": include = torso || leg; break;
                default: throw new InvalidOperationException("unsupported garment construction id in frozen catalog: " + garmentId);
            }
            if (include) total += weight;
        }

        static bool ContainsAny(string value, params string[] fragments) => fragments.Any(value.Contains);

        static int DominantBone(BoneWeight weight, out float maximum)
        {
            int index = weight.boneIndex0; maximum = weight.weight0;
            if (weight.weight1 > maximum) { maximum = weight.weight1; index = weight.boneIndex1; }
            if (weight.weight2 > maximum) { maximum = weight.weight2; index = weight.boneIndex2; }
            if (weight.weight3 > maximum) { maximum = weight.weight3; index = weight.boneIndex3; }
            return index;
        }

        static float BoneWeightFor(BoneWeight weight, int boneIndex)
        {
            float total = 0;
            if (weight.boneIndex0 == boneIndex) total += weight.weight0;
            if (weight.boneIndex1 == boneIndex) total += weight.weight1;
            if (weight.boneIndex2 == boneIndex) total += weight.weight2;
            if (weight.boneIndex3 == boneIndex) total += weight.weight3;
            return total;
        }

        static bool ColliderMatchesBone(Collider collider, Transform bone)
        {
            if (collider.transform == bone || collider.transform.parent == bone) return true;
            if (collider.attachedArticulationBody != null && collider.attachedArticulationBody.transform == bone) return true;
            if (collider.attachedRigidbody != null && collider.attachedRigidbody.transform == bone) return true;
            string colliderName = SemanticBoneName(collider.transform.name);
            string boneName = SemanticBoneName(bone.name);
            return colliderName == boneName || (boneName.Length >= 5 && colliderName.Contains(boneName));
        }

        static string SemanticBoneName(string value)
        {
            string result = value.ToLowerInvariant().Replace("physical", "").Replace("collider", "").Replace("body", "").Replace("right", "").Replace("left", "");
            result = result.Replace("finger1", "thumb").Replace("finger2", "index").Replace("finger3", "middle").Replace("finger4", "ring").Replace("finger5", "little");
            result = new string(result.Where(char.IsLetterOrDigit).ToArray());
            if ((result.EndsWith("r") || result.EndsWith("l")) && result.Length > 1) result = result.Substring(0, result.Length - 1);
            return result;
        }

        static float SurfaceDistance(Collider collider, Vector3 worldPoint)
        {
            if (collider is SphereCollider sphere) {
                Vector3 center = sphere.transform.TransformPoint(sphere.center);
                Vector3 scale = Abs(sphere.transform.lossyScale);
                float radius = sphere.radius * Mathf.Max(scale.x, Mathf.Max(scale.y, scale.z));
                return Mathf.Abs(Vector3.Distance(worldPoint, center) - radius);
            }
            if (collider is CapsuleCollider capsule) {
                Vector3 scale = Abs(capsule.transform.lossyScale);
                Vector3 axis = capsule.direction == 0 ? Vector3.right : capsule.direction == 1 ? Vector3.up : Vector3.forward;
                float axialScale = capsule.direction == 0 ? scale.x : capsule.direction == 1 ? scale.y : scale.z;
                float radialScale = capsule.direction == 0 ? Mathf.Max(scale.y, scale.z) : capsule.direction == 1 ? Mathf.Max(scale.x, scale.z) : Mathf.Max(scale.x, scale.y);
                float radius = capsule.radius * radialScale;
                float halfSegment = Mathf.Max(0, capsule.height * axialScale * .5f - radius);
                Vector3 center = capsule.transform.TransformPoint(capsule.center);
                Vector3 worldAxis = capsule.transform.TransformDirection(axis).normalized;
                Vector3 a = center - worldAxis * halfSegment, b = center + worldAxis * halfSegment;
                return Mathf.Abs(DistanceToSegment(worldPoint, a, b) - radius);
            }
            if (collider is BoxCollider box) {
                Vector3 local = box.transform.InverseTransformPoint(worldPoint) - box.center;
                Vector3 half = box.size * .5f;
                Vector3 outside = new Vector3(Mathf.Max(Mathf.Abs(local.x) - half.x, 0), Mathf.Max(Mathf.Abs(local.y) - half.y, 0), Mathf.Max(Mathf.Abs(local.z) - half.z, 0));
                if (outside.sqrMagnitude > 0) return Vector3.Distance(worldPoint, box.ClosestPoint(worldPoint));
                Vector3 scale = Abs(box.transform.lossyScale);
                return Mathf.Min((half.x - Mathf.Abs(local.x)) * scale.x, Mathf.Min((half.y - Mathf.Abs(local.y)) * scale.y, (half.z - Mathf.Abs(local.z)) * scale.z));
            }
            Vector3 closest = collider.ClosestPoint(worldPoint);
            float distance = Vector3.Distance(worldPoint, closest);
            return distance > 1e-8f ? distance : float.NaN;
        }

        static Vector3 Abs(Vector3 value) => new Vector3(Mathf.Abs(value.x), Mathf.Abs(value.y), Mathf.Abs(value.z));
        static float DistanceToSegment(Vector3 p, Vector3 a, Vector3 b)
        {
            Vector3 ab = b - a;
            float t = ab.sqrMagnitude > 1e-12f ? Mathf.Clamp01(Vector3.Dot(p - a, ab) / ab.sqrMagnitude) : 0;
            return Vector3.Distance(p, a + ab * t);
        }

        Transform FindRequired(string name)
        {
            if (!bones.TryGetValue(name, out Transform result)) throw new InvalidOperationException("missing required MPFB bone: " + name);
            return result;
        }

        Transform FindFirstRequired(params string[] names)
        {
            foreach (string name in names) if (bones.TryGetValue(name, out Transform result)) return result;
            throw new InvalidOperationException("missing required MPFB bone; expected one of: " + string.Join(", ", names));
        }

        void RequireBoundContext(GateContext context)
        {
            if (!ReferenceEquals(context, boundContext) || bodyRenderer == null || context.AvatarRoot == null)
                throw new InvalidOperationException("embodiment module is not bound to this GateContext");
        }

        void ClearGeneratedGarments()
        {
            foreach (var garment in garments) {
                if (garment.baked != null) UnityEngine.Object.DestroyImmediate(garment.baked);
                if (garment.renderer != null && garment.renderer.sharedMesh != null) UnityEngine.Object.DestroyImmediate(garment.renderer.sharedMesh);
                if (garment.renderer != null) UnityEngine.Object.DestroyImmediate(garment.renderer.gameObject);
            }
            garments.Clear();
        }

        void ResetRegistrationAccumulators()
        {
            registrationSteps.Clear();
            skinColliderErrors.Clear();
            exactFittedSkinColliderErrors.Clear();
            fallbackSkinColliderErrors.Clear();
            skinColliderErrorsByBone.Clear();
            foreach (var garment in garments) { garment.penetration.Clear(); garment.minimumClearanceM = float.PositiveInfinity; }
            lastSampledStep = -1;
            auditedMotionSampleCount = 0;
            samplesContiguousFromZero = true;
        }

        static FrozenDocument LoadFrozenDocument()
        {
            string environment = Environment.GetEnvironmentVariable("PROCEDURAL_SCENE_GATE_CONFIG");
            var candidates = new List<string>();
            if (!string.IsNullOrWhiteSpace(environment)) candidates.Add(environment);
            candidates.Add(Path.Combine(Environment.CurrentDirectory, "configs", FrozenConfigName));
            candidates.Add(Path.Combine(Application.dataPath, "Config", FrozenConfigName));
            DirectoryInfo cursor = new DirectoryInfo(Environment.CurrentDirectory);
            for (int i = 0; i < 8 && cursor != null; i++, cursor = cursor.Parent)
                candidates.Add(Path.Combine(cursor.FullName, "configs", FrozenConfigName));
            string path = candidates.Select(Path.GetFullPath).Distinct().FirstOrDefault(File.Exists);
            if (path == null) throw new FileNotFoundException("set PROCEDURAL_SCENE_GATE_CONFIG to the frozen repository config", FrozenConfigName);
            var document = JsonUtility.FromJson<FrozenDocument>(File.ReadAllText(path));
            if (document == null) throw new InvalidOperationException("Unity could not parse the frozen procedural scene gate config");
            return document;
        }

        static void ValidateFrozenCatalog(FrozenDocument document)
        {
            if (document.schema != FrozenGate.ConfigSchema) throw new InvalidOperationException("wrong frozen config schema: " + document.schema);
            if (document.assets == null || document.assets.avatar == null || document.assets.procedural_garments == null)
                throw new InvalidOperationException("frozen asset and garment provenance are required");
            if (document.assets.avatar.asset_id != ExpectedAvatarId || document.assets.avatar.license != "CC0" || !document.assets.avatar.public_synthetic)
                throw new InvalidOperationException("avatar must be the frozen public CC0 MPFB asset");
            if (document.avatar_specs == null || document.avatar_specs.Length < 3 || document.avatar_specs.Select(x => x.garment_configuration_id).Distinct().Count() < 3)
                throw new InvalidOperationException("at least three frozen garment configurations are required");
            foreach (var spec in document.avatar_specs) {
                if (spec.schema != "embodied.avatar_spec.v1" || spec.avatar_id != ExpectedAvatarId)
                    throw new InvalidOperationException("garment configuration is not attached to the frozen AvatarSpec");
                if (spec.hand_topology == null || spec.hand_topology.hands != 2 || spec.hand_topology.digits_per_hand != 5 || spec.hand_topology.segments_per_digit != 3)
                    throw new InvalidOperationException("every AvatarSpec must preserve two five-finger three-segment hands");
            }
        }

        static void ValidateAvatarAsset(string assetPath, AvatarAsset asset)
        {
            string fullPath = Path.GetFullPath(assetPath);
            if (!File.Exists(fullPath)) throw new FileNotFoundException("frozen avatar asset does not exist", assetPath);
            string actual = Sha256(fullPath);
            if (!string.Equals(actual, asset.sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("MPFB source hash differs from the frozen AvatarSpec: " + actual);
        }

        static void ValidateGarmentSpec(GarmentSpec spec)
        {
            if (spec == null || string.IsNullOrWhiteSpace(spec.id) || string.IsNullOrWhiteSpace(spec.material))
                throw new InvalidOperationException("catalog garment id and material are required");
            if (spec.layer < 0 || spec.layer > 31) throw new InvalidOperationException("garment layer is outside frozen supported range: " + spec.layer);
            if (spec.fit_offset_m <= 0 || spec.fit_offset_m > .02f) throw new InvalidOperationException("garment fit offset is invalid: " + spec.fit_offset_m);
            if (spec.color_rgba == null || spec.color_rgba.Length != 4) throw new InvalidOperationException("garment RGBA color is required");
        }

        static Material SkinMaterial()
        {
            var material = new Material(RequiredLitShader()) { name = "MPFB_CHILD_SKIN_MATERIAL", color = new Color(.74f, .49f, .36f, 1) };
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", .18f);
            return material;
        }

        static Material GarmentMaterial(GarmentSpec spec)
        {
            var color = new Color(spec.color_rgba[0], spec.color_rgba[1], spec.color_rgba[2], spec.color_rgba[3]);
            var material = new Material(RequiredLitShader()) { name = "GARMENT_MATERIAL_" + spec.material, color = color };
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", spec.material == "denim" ? .08f : .16f);
            return material;
        }

        static Shader RequiredLitShader()
        {
            Shader shader = Shader.Find("Standard");
            if (shader == null) shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) throw new InvalidOperationException("a Unity lit shader is required for hero embodiment pixels");
            return shader;
        }

        static string Sha256(string path)
        {
            using (var hash = SHA256.Create())
                return BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(path))).Replace("-", "").ToLowerInvariant();
        }

        sealed class GarmentBinding
        {
            public GarmentSpec spec;
            public SkinnedMeshRenderer renderer;
            public Mesh baked;
            public int[] usedVertexIndices;
            public Distribution penetration;
            public float minimumClearanceM;

            public GarmentRegistrationReceipt Receipt() => new GarmentRegistrationReceipt {
                garment_id = spec.id,
                layer = spec.layer,
                fit_offset_m = spec.fit_offset_m,
                material = spec.material,
                color_rgba = spec.color_rgba,
                samples = penetration.count,
                affected_vertices = penetration.affected,
                affected_fraction = penetration.count > 0 ? penetration.affected / (float)penetration.count : 0,
                affected_vertex_distribution_edges_m = penetration.edges,
                affected_vertex_distribution_counts = penetration.bins,
                penetration_p50_upper_bound_m = penetration.QuantileUpperBound(.50f),
                penetration_p95_upper_bound_m = penetration.QuantileUpperBound(.95f),
                maximum_penetration_m = penetration.maximum_m,
                minimum_signed_clearance_m = float.IsFinite(minimumClearanceM) ? minimumClearanceM : 0,
                passed = penetration.count > 0 && penetration.maximum_m <= GarmentBodyToleranceM,
            };
        }

        sealed class FittedColliderBinding
        {
            public SphereCollider collider;
            public Transform bone;
            public int[] vertexIndices;
        }

        sealed class KinematicColliderBinding
        {
            public Rigidbody body;
            public Transform bone;
            public Vector3 positionInBone;
            public Quaternion rotationInBone;
        }

        struct IndexedLocalPoint
        {
            public int vertexIndex;
            public Vector3 local;
        }

        struct MissingFittedPoint
        {
            public Transform bone;
            public IndexedLocalPoint point;
        }

        sealed class Distribution
        {
            public readonly float[] edges;
            public readonly long[] bins;
            public long count, affected;
            public float maximum_m;

            public Distribution(float[] sourceEdges) { edges = (float[])sourceEdges.Clone(); bins = new long[edges.Length + 1]; }
            public void Observe(float value, bool isAffected) {
                if (!float.IsFinite(value) || value < 0) return;
                count++; if (isAffected) affected++; maximum_m = Mathf.Max(maximum_m, value);
                int bin = 0; while (bin < edges.Length && value > edges[bin]) bin++; bins[bin]++;
            }
            public void Clear() { Array.Clear(bins, 0, bins.Length); count = 0; affected = 0; maximum_m = 0; }
            public float QuantileUpperBound(float q) {
                if (count == 0) return 0;
                long target = Math.Max(1, (long)Math.Ceiling(count * q)), cumulative = 0;
                for (int i = 0; i < bins.Length; i++) { cumulative += bins[i]; if (cumulative >= target) return i < edges.Length ? edges[i] : maximum_m; }
                return maximum_m;
            }
            public DistributionReceipt Receipt() => new DistributionReceipt {
                samples = count,
                out_of_tolerance_samples = affected,
                histogram_edges_m = edges,
                histogram_counts = bins,
                p50_upper_bound_m = QuantileUpperBound(.50f),
                p95_upper_bound_m = QuantileUpperBound(.95f),
                maximum_m = maximum_m,
            };
        }

        [Serializable] sealed class FrozenDocument { public string schema; public FrozenAssets assets; public AvatarSpec[] avatar_specs; }
        [Serializable] sealed class FrozenAssets { public AvatarAsset avatar; public ProceduralGarmentAsset procedural_garments; }
        [Serializable] sealed class AvatarAsset { public string asset_id, sha256, license; public bool public_synthetic; }
        [Serializable] sealed class ProceduralGarmentAsset { public string license, construction; public bool real_time_cloth_required; }
        [Serializable] sealed class AvatarSpec { public string schema, avatar_id, source_asset, skeleton, garment_configuration_id; public HandTopology hand_topology; public GarmentSpec[] garments; }
        [Serializable] public sealed class HandTopology { public int hands, digits_per_hand, segments_per_digit; }
        [Serializable] sealed class GarmentSpec { public string id, material; public int layer; public float fit_offset_m; public float[] color_rgba; }

        [Serializable] sealed class EmbodimentRegistrationReport
        {
            public string schema, avatar_spec_schema, embodiment_manifest_schema, avatar_id, garment_configuration_id;
            public string source_asset_id, source_asset_sha256, source_license, construction_license, construction_method;
            public string authority_root, avatar_root, body_renderer;
            public string[] body_bones, anatomical_collider_ids, garment_renderers, garment_materials;
            public HandTopology hand_topology;
            public int anatomical_collider_count, kinematic_collider_body_count;
            public int registration_eligible_vertex_count, exact_fitted_registration_vertex_count, fallback_registration_vertex_count;
            public int[] garment_layers;
            public float[] garment_fit_offsets_m;
            public string body_collider_provenance, body_collider_unavailable_reason;
            public DistributionReceipt body_collider_registration;
            public DistributionReceipt exact_fitted_body_collider_registration, fallback_body_collider_registration;
            public BoneRegistrationReceipt[] body_collider_registration_by_bone;
            public CandidateCoverageReceipt[] registration_candidate_coverage_by_bone;
            public string garment_body_provenance, garment_body_method;
            public string garment_body_full_triangle_intersection_provenance, garment_body_full_triangle_intersection_unavailable_reason;
            public string garment_self_intersection_provenance, garment_self_intersection_unavailable_reason;
            public GarmentRegistrationReceipt[] garments;
            public int motion_sample_count, audited_motion_sample_count, registration_audit_stride_physics_steps, expected_motion_sample_count, first_physics_step, last_physics_step, module_created_proxy_renderers;
            public bool both_palms_and_all_finger_segments_have_colliders, sampled_every_physics_step, independently_advanced_animation, passed;
            public float body_collider_tolerance_m, garment_body_tolerance_m;
            public RegistrationStep[] registration_steps;
        }
        [Serializable] sealed class CandidateCoverageReceipt { public string bone; public int eligible, exact_fitted, fallback, unavailable; }

        [Serializable] sealed class DistributionReceipt
        {
            public long samples, out_of_tolerance_samples;
            public float[] histogram_edges_m;
            public long[] histogram_counts;
            public float p50_upper_bound_m, p95_upper_bound_m, maximum_m;
        }

        [Serializable] sealed class BoneRegistrationReceipt
        {
            public string bone;
            public DistributionReceipt distribution;
        }

        [Serializable] sealed class GarmentRegistrationReceipt
        {
            public string garment_id, material;
            public int layer;
            public float fit_offset_m;
            public float[] color_rgba;
            public long samples, affected_vertices;
            public float affected_fraction;
            public float[] affected_vertex_distribution_edges_m;
            public long[] affected_vertex_distribution_counts;
            public float penetration_p50_upper_bound_m, penetration_p95_upper_bound_m, maximum_penetration_m, minimum_signed_clearance_m;
            public bool passed;
        }

        [Serializable] sealed class RegistrationStep { public int physics_step; public float time_s; public bool audited; public BodyColliderStep body_collider; public GarmentStep[] garments; }
        [Serializable] sealed class BodyColliderStep { public string provenance; public int samples; public float maximum_error_m; }
        [Serializable] sealed class GarmentStep
        {
            public string garment_id, provenance;
            public int samples, affected_vertices;
            public float affected_fraction, minimum_signed_clearance_m, maximum_penetration_m;
        }
    }
}
#endif
