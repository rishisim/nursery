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
    /// Compiles every frozen room through one catalog-driven path.  The compiler is
    /// allowed to initialize the free target once; after its Rigidbody is created no
    /// code in this module retains a target Transform or writes an object trajectory.
    /// </summary>
    public sealed class ProceduralSceneCompiler : ISceneCompilerModule
    {
        private const string SceneSchema = "embodied.scene_spec.v1";
        private const string ReceiptSchema = "embodied.scene_compiler_validation.v1";
        private const string CatalogId = "kenney_furniture_kit_cc0_curated";
        private const string CatalogLicense = "CC0";
        private const string CatalogArchiveSha256 = "68afa4e6dc8a53942379fb47f1e84ec735d46b77bb1c1ceb968e245693dde067";
        private const string TargetPersistentId = "target_001";
        private const int TargetSemanticId = 41;
        private const int TargetInstanceId = 41001;
        private const string TargetGeometrySpecSha256 = "9d9963e46446fb05187c285062e7562af142a82906b5adf3f491f1d70060e5c5";
        private const float TargetDiameterM = 0.055f;
        private const float TargetMassKg = 0.12f;
        private const float TargetStaticFriction = 1.0f;
        private const float TargetDynamicFriction = 0.9f;
        private const float FurnitureStaticFriction = 0.75f;
        private const float FurnitureDynamicFriction = 0.65f;
        private const float SupportGapM = 0.001f;
        private const float TargetMidpointReachCenterM = 0.36f;
        private const float TargetSeedReachOffsetM = 0.020f;
        private const float TargetLateralBiasTowardRightShoulderM = 0.025f;

        private static readonly CatalogMember[] Catalog =
        {
            new CatalogMember("tableCoffee", "8c81c31a74aadc1e89e334eb1ac43380f41c14f47e5dd9a6b029ebf74850adda", new Vector3(1.15f, 0.62f, 0.86f), "support_table", "mesh_or_compound_box"),
            new CatalogMember("loungeSofaLong", "bbc91d1b01537dbc847685560265b6fcd72a21333c8aab7f8906a5addf8dbd20", new Vector3(1.65f, 0.92f, 0.78f), "sofa", "compound_box"),
            new CatalogMember("bookcaseOpen", "31beea66f34e64660c870e39c52ed2e81984939c88f4bd763c7dce56e9ef76a9", new Vector3(0.72f, 1.64f, 0.35f), "storage", "compound_box"),
            new CatalogMember("chairCushion", "afc684f28ab2e4ee3fe9cd59a92f03103e4f62c74ae785029fdd672fc6914cdd", new Vector3(0.55f, 1.24f, 0.55f), "chair", "compound_box"),
            new CatalogMember("rugRectangle", "0cdd8cea357bc267c9f121b4e7d32513c08f1b179522d95c6632bf2306d38f79", new Vector3(1.75f, 0.015f, 1.20f), "rug", "box"),
            new CatalogMember("lampSquareFloor", "c4b35656f587cacf2ccf57d1c0216541923890e878cec0d0f58ff71cd4a71bfc", new Vector3(0.42f, 1.35f, 0.42f), "lamp", "compound_box"),
            new CatalogMember("pottedPlant", "cf66928b943e30ef5c1f06145419ca2cda534249bb4d77831ae723532c4f8de3", new Vector3(0.55f, 1.00f, 0.55f), "plant", "compound_box"),
            new CatalogMember("books", "91fd64f5fd821b5046724ffa18974e8964b85bfbd4a0d89687d2283966394f5f", new Vector3(0.32f, 0.14f, 0.24f), "books", "box")
        };

        // These are family constraints, not seed-specific scenes.  Every placement
        // still passes through the same normalization, collision, identity, and
        // validation implementation below.
        private static readonly RoomProfile[] Profiles =
        {
            new RoomProfile(
                "warm_playroom", "warm_oak", new Vector3(4.4f, 2.55f, 4.8f),
                new[] { "window_scan", "low_table", "toy_shelf", "rug" },
                new Color(0.78f, 0.74f, 0.64f), new Color(0.42f, 0.27f, 0.15f), new Color(0.92f, 0.82f, 0.64f),
                new[]
                {
                    new Placement("bookcaseOpen", -1.55f, 3.55f, 2f, false, "toy_shelf"),
                    new Placement("rugRectangle", 0.05f, 1.75f, 0f, true, "rug"),
                    new Placement("chairCushion", 1.25f, 2.45f, 212f, false, "reading_corner"),
                    new Placement("books", -1.54f, 3.36f, 8f, false, "shelf_clutter")
                }),
            new RoomProfile(
                "sage_living_corner", "painted_sage", new Vector3(4.8f, 2.55f, 5.2f),
                new[] { "sofa", "low_table", "bookcase", "window_scan" },
                new Color(0.65f, 0.73f, 0.66f), new Color(0.35f, 0.27f, 0.20f), new Color(0.75f, 0.84f, 0.66f),
                new[]
                {
                    new Placement("loungeSofaLong", 0.75f, 4.15f, 2f, false, "sofa"),
                    new Placement("rugRectangle", 0.05f, 1.85f, 90f, true, "rug"),
                    new Placement("lampSquareFloor", 1.82f, 3.88f, 185f, false, "distant_lamp"),
                    new Placement("pottedPlant", -1.82f, 3.08f, 178f, false, "distant_plant")
                }),
            new RoomProfile(
                "birch_art_room", "natural_birch", new Vector3(4.2f, 2.55f, 4.6f),
                new[] { "art_shelf", "low_table", "reading_chair", "window_scan" },
                new Color(0.83f, 0.82f, 0.73f), new Color(0.58f, 0.45f, 0.29f), new Color(0.88f, 0.88f, 0.78f),
                new[]
                {
                    new Placement("bookcaseOpen", -1.48f, 3.45f, 1f, false, "art_shelf"),
                    new Placement("chairCushion", 1.25f, 2.38f, 202f, false, "reading_chair"),
                    new Placement("rugRectangle", -0.08f, 1.72f, 0f, true, "rug"),
                    new Placement("pottedPlant", 1.60f, 3.54f, 183f, false, "distant_plant")
                })
        };

        private readonly List<CompiledInstance> _instances = new List<CompiledInstance>();
        private readonly Dictionary<string, Material> _materials = new Dictionary<string, Material>();
        private RoomProfile _profile;
        private GateContext _context;
        private Transform _sceneRoot;
        private string _furnitureAssetRoot;
        private System.Random _random;

        public void Build(GateContext context, string furnitureAssetRoot)
        {
            ValidateInputs(context, furnitureAssetRoot);
            _context = context;
            _furnitureAssetRoot = furnitureAssetRoot.TrimEnd('/', '\\');
            _profile = Profiles.Single(profile => profile.RoomFamily == context.RoomFamily);
            _random = new System.Random(context.Seed);
            _instances.Clear();

            var rootObject = new GameObject("COMPILED_ROOM_" + context.RoomFamily);
            rootObject.transform.SetParent(context.AuthorityRoot.transform, false);
            _sceneRoot = rootObject.transform;

            ConfigureEnvironment();
            BuildRoomSurfaces();
            BuildZones();

            float lateralJitter = NextSigned(0.018f);
            float depthJitter = NextSigned(0.018f);
            var table = PlaceCatalogInstance(
                FindCatalog("tableCoffee"),
                new Vector2(0.04f + lateralJitter, 0.52f + depthJitter),
                NextSigned(2.5f),
                true,
                "primary_support",
                0);

            float desiredReachM = TargetMidpointReachCenterM + TargetSeedReachOffsetM * (PositiveModulo(context.Seed, 3) - 1);
            Transform rightShoulder = FindRightShoulder(context);
            Transform leftShoulder = FindLeftShoulder(context);
            Vector3 shoulderMidpoint = 0.5f * (leftShoulder.position + rightShoulder.position);
            Vector3 targetPosition = SolveTargetPosition(shoulderMidpoint, table.Bounds.max.y, desiredReachM);
            Vector3 towardRightShoulder = Vector3.ProjectOnPlane(rightShoulder.position - shoulderMidpoint, Vector3.up).normalized;
            targetPosition += towardRightShoulder * TargetLateralBiasTowardRightShoulderM;
            MoveStaticSupportUnderTarget(table, targetPosition);
            targetPosition.y = table.Bounds.max.y + TargetDiameterM * 0.5f + SupportGapM;

            foreach (Placement placement in _profile.Placements)
            {
                float localX = placement.LocalX + NextSigned(0.025f);
                float localDepth = placement.LocalDepth + NextSigned(0.025f);
                float yaw = placement.YawDeg + NextSigned(3.0f);
                PlaceCatalogInstance(
                    FindCatalog(placement.AssetId),
                    new Vector2(localX, localDepth),
                    yaw,
                    placement.Reachable,
                    placement.Role,
                    _instances.Count);
            }

            Physics.SyncTransforms();
            CreateFreeTarget(targetPosition, table.PersistentId);
            Physics.SyncTransforms();

            SceneSpecReceipt sceneSpec = CompileSceneSpec(table, rightShoulder, leftShoulder, desiredReachM);
            SceneValidationReceipt validation = ValidateCompiledScene(sceneSpec, table, rightShoulder, leftShoulder);
            WriteReceipts(sceneSpec, validation);
        }

        private static void ValidateInputs(GateContext context, string furnitureAssetRoot)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (!context.AuthorityRoot) throw new InvalidOperationException("GateContext.AuthorityRoot must exist before scene compilation");
            if (!context.AvatarRoot || !context.RightPalm || !context.Head)
                throw new InvalidOperationException("embodiment must bind AvatarRoot, RightPalm, and Head before target initialization");
            if (context.TargetBody) throw new InvalidOperationException("the free target was already initialized");
            if (context.Seed == 0) throw new InvalidOperationException("a nonzero frozen scene seed is required");
            if (!Profiles.Any(profile => profile.RoomFamily == context.RoomFamily))
                throw new InvalidOperationException("unknown frozen room family: " + context.RoomFamily);
            if (string.IsNullOrWhiteSpace(context.OutputRoot)) throw new InvalidOperationException("GateContext.OutputRoot is required for scene receipts");
            if (string.IsNullOrWhiteSpace(furnitureAssetRoot) || !furnitureAssetRoot.Replace('\\', '/').StartsWith("Assets/", StringComparison.Ordinal))
                throw new InvalidOperationException("furnitureAssetRoot must be a Unity project Assets path");
            if (context.AssistanceLedger.Count != 0) throw new InvalidOperationException("assistance ledger was nonempty before scene compilation");
        }

        private void ConfigureEnvironment()
        {
            RenderSettings.fog = false;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = Color.Lerp(_profile.WallColor, Color.white, 0.30f);
            RenderSettings.ambientEquatorColor = Color.Lerp(_profile.WallColor, _profile.FloorColor, 0.45f);
            RenderSettings.ambientGroundColor = Color.Lerp(_profile.FloorColor, Color.black, 0.45f);

            var key = new GameObject("window_key_light");
            key.transform.SetParent(_sceneRoot, false);
            key.transform.localRotation = Quaternion.Euler(44f, -31f, 0f);
            Light keyLight = key.AddComponent<Light>();
            keyLight.type = LightType.Directional;
            keyLight.intensity = 1.18f;
            keyLight.color = new Color(1.0f, 0.91f, 0.78f);
            keyLight.shadows = LightShadows.Soft;

            var fill = new GameObject("room_bounce_light");
            fill.transform.SetParent(_sceneRoot, false);
            fill.transform.localPosition = new Vector3(1.35f, 2.12f, -1.28f);
            Light fillLight = fill.AddComponent<Light>();
            fillLight.type = LightType.Point;
            fillLight.range = 5.8f;
            fillLight.intensity = 2.3f;
            fillLight.color = Color.Lerp(_profile.AccentColor, Color.white, 0.35f);
            fillLight.shadows = LightShadows.Soft;
        }

        private void BuildRoomSurfaces()
        {
            float width = _profile.EnvelopeM.x;
            float height = _profile.EnvelopeM.y;
            float depth = _profile.EnvelopeM.z;
            float roomCenterDepth = depth * 0.5f - 0.55f;
            float backDepth = depth - 0.55f;

            CreateRoomSurface("floor_surface", new Vector3(0f, -0.04f, -roomCenterDepth), new Vector3(width, 0.08f, depth), _profile.FloorColor, "floor");
            CreateRoomSurface("back_wall_surface", new Vector3(0f, height * 0.5f, -backDepth), new Vector3(width, height, 0.08f), _profile.WallColor, "wall");
            CreateRoomSurface("left_wall_surface", new Vector3(-width * 0.5f, height * 0.5f, -roomCenterDepth), new Vector3(0.08f, height, depth), _profile.WallColor, "wall");
            CreateRoomSurface("right_wall_surface", new Vector3(width * 0.5f, height * 0.5f, -roomCenterDepth), new Vector3(0.08f, height, depth), _profile.WallColor, "wall");
            CreateRoomSurface("back_baseboard", new Vector3(0f, 0.09f, -backDepth + 0.055f), new Vector3(width - 0.10f, 0.18f, 0.045f), Color.Lerp(_profile.WallColor, Color.white, 0.68f), "trim");
            CreateRoomSurface("window_scan_surface", new Vector3(0.72f, 1.52f, -backDepth + 0.055f), new Vector3(1.12f, 0.86f, 0.025f), new Color(0.54f, 0.75f, 0.90f), "window");
        }

        private void CreateRoomSurface(string name, Vector3 localPosition, Vector3 size, Color color, string semanticClass)
        {
            var surface = new GameObject(name);
            surface.transform.SetParent(_sceneRoot, false);
            surface.transform.localPosition = localPosition;
            surface.transform.localScale = size;
            Mesh mesh = UnitBoxMesh();
            surface.AddComponent<MeshFilter>().sharedMesh = mesh;
            surface.AddComponent<MeshRenderer>().sharedMaterial = MaterialFor("surface_" + semanticClass, color, 0.12f);
            var collider = surface.AddComponent<BoxCollider>();
            collider.sharedMaterial = MakePhysicsMaterial("room_surface", FurnitureStaticFriction, FurnitureDynamicFriction);
            SceneIdentity identity = surface.AddComponent<SceneIdentity>();
            identity.persistent_id = _context.RoomFamily + "_" + name;
            identity.semantic_id = SemanticId(semanticClass);
            identity.instance_id = StableInstanceId(_context.Seed, identity.persistent_id);
            identity.semantic_class = semanticClass;
            identity.interactive = false;
            identity.physics_role = "static_room_surface";
            identity.license = "repository-authored";
        }

        private void BuildZones()
        {
            float backDepth = _profile.EnvelopeM.z - 0.62f;
            for (int index = 0; index < _profile.ZoneIds.Length; index++)
            {
                string zoneId = _profile.ZoneIds[index];
                var zoneObject = new GameObject("ZONE_" + zoneId);
                zoneObject.transform.SetParent(_sceneRoot, false);
                Vector3 localCenter;
                Vector3 extents;
                switch (zoneId)
                {
                    case "low_table":
                        localCenter = new Vector3(0f, 0.45f, -0.62f);
                        extents = new Vector3(1.35f, 0.90f, 1.20f);
                        break;
                    case "window_scan":
                        localCenter = new Vector3(0.72f, 1.52f, -backDepth);
                        extents = new Vector3(1.12f, 0.86f, 0.30f);
                        break;
                    case "rug":
                        localCenter = new Vector3(0f, 0.05f, -1.75f);
                        extents = new Vector3(1.90f, 0.10f, 1.35f);
                        break;
                    default:
                        localCenter = new Vector3(-1.35f, 0.82f, -3.25f);
                        extents = new Vector3(0.85f, 1.65f, 0.75f);
                        break;
                }
                zoneObject.transform.localPosition = localCenter;
                SceneZone zone = zoneObject.AddComponent<SceneZone>();
                zone.zone_id = zoneId;
                zone.center_world_m = zoneObject.transform.position;
                zone.extents_m = extents;
                zone.semantic_purpose = ZonePurpose(zoneId);
            }
        }

        private CompiledInstance PlaceCatalogInstance(CatalogMember member, Vector2 floorCoordinate, float yawDeg, bool reachable, string role, int index)
        {
            string assetPath = _furnitureAssetRoot + "/" + member.AssetId + ".obj";
            VerifyCatalogSource(assetPath, member.Sha256);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
            if (!prefab) throw new InvalidOperationException("missing frozen furniture asset: " + assetPath);
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            if (!instance) throw new InvalidOperationException("could not instantiate frozen furniture asset: " + assetPath);

            instance.name = _context.RoomFamily + "_" + member.AssetId + "_" + index.ToString("D2");
            instance.transform.SetParent(_sceneRoot, false);
            instance.transform.localPosition = Vector3.zero;
            instance.transform.localRotation = Quaternion.identity;
            instance.transform.localScale = Vector3.one;
            ScaleToCatalogDimensions(instance, member.DimensionsM);
            instance.transform.localRotation = Quaternion.Euler(0f, yawDeg, 0f);

            Bounds current = BoundsOf(instance);
            Vector3 desiredFloorCenter = _sceneRoot.TransformPoint(new Vector3(floorCoordinate.x, 0f, -floorCoordinate.y));
            instance.transform.position += new Vector3(
                desiredFloorCenter.x - current.center.x,
                desiredFloorCenter.y - current.min.y,
                desiredFloorCenter.z - current.center.z);

            Collider[] colliders = AddCatalogColliders(instance, member.CollisionSource);
            foreach (Collider collider in colliders)
                collider.sharedMaterial = MakePhysicsMaterial("furniture", FurnitureStaticFriction, FurnitureDynamicFriction);
            foreach (Renderer renderer in instance.GetComponentsInChildren<Renderer>(true))
                renderer.sharedMaterial = FurnitureMaterial(member.SemanticClass);

            string persistentId = instance.name;
            SceneIdentity identity = instance.AddComponent<SceneIdentity>();
            identity.persistent_id = persistentId;
            identity.semantic_id = SemanticId(member.SemanticClass);
            identity.instance_id = StableInstanceId(_context.Seed, persistentId);
            identity.semantic_class = member.SemanticClass;
            identity.interactive = false;
            identity.physics_role = reachable ? "reachable_static_physx_collider" : "distant_noninteractive_static_physx_collider";
            identity.support_id = role == "primary_support" ? persistentId : string.Empty;
            identity.license = CatalogLicense;
            identity.source_sha256 = member.Sha256;

            current = BoundsOf(instance);
            var compiled = new CompiledInstance
            {
                persistent_id = persistentId,
                asset_id = member.AssetId,
                asset_path = assetPath,
                source_sha256 = member.Sha256,
                license = CatalogLicense,
                catalog_dimensions_m = member.DimensionsM,
                compiled_bounds_center_world_m = current.center,
                compiled_bounds_size_m = current.size,
                rotation_world_xyzw = instance.transform.rotation,
                semantic_class = member.SemanticClass,
                semantic_id = identity.semantic_id,
                instance_id = identity.instance_id,
                material_variant = _profile.MaterialVariant,
                collision_source = member.CollisionSource,
                collider_count = colliders.Length,
                interactive = false,
                reachable = reachable,
                role = role,
                visible_geometry_source = "verified_imported_kenney_mesh",
                game_object = instance,
                Bounds = current
            };
            _instances.Add(compiled);
            return compiled;
        }

        private void MoveStaticSupportUnderTarget(CompiledInstance table, Vector3 targetPosition)
        {
            Vector3 currentCenter = table.Bounds.center;
            Vector3 roomForward = _sceneRoot.TransformDirection(Vector3.back);
            Vector3 desiredCenter = targetPosition + roomForward * 0.14f;
            table.game_object.transform.position += new Vector3(desiredCenter.x - currentCenter.x, 0f, desiredCenter.z - currentCenter.z);
            table.Bounds = BoundsOf(table.game_object);
            table.compiled_bounds_center_world_m = table.Bounds.center;
            table.compiled_bounds_size_m = table.Bounds.size;
        }

        private Vector3 SolveTargetPosition(Vector3 shoulderMidpoint, float supportTopY, float desiredReachM)
        {
            Vector3 target = shoulderMidpoint;
            Vector3 roomForward = _sceneRoot.TransformDirection(Vector3.back);
            float targetY = supportTopY + TargetDiameterM * 0.5f + SupportGapM;
            float vertical = targetY - shoulderMidpoint.y;
            float forwardSquared = desiredReachM * desiredReachM - vertical * vertical;
            if (forwardSquared <= 0.01f)
                throw new InvalidOperationException("catalog support height cannot satisfy the frozen bimanual shoulder-midpoint reach corridor");
            target += roomForward * Mathf.Sqrt(forwardSquared);
            target.y = targetY;
            return target;
        }

        private void CreateFreeTarget(Vector3 initialPosition, string supportId)
        {
            var target = new GameObject(TargetPersistentId);
            target.transform.SetPositionAndRotation(
                initialPosition,
                _sceneRoot.rotation * Quaternion.Euler(0f, NextSigned(18f), 0f));
            target.transform.localScale = Vector3.one * TargetDiameterM;

            Mesh targetMesh = RoundedGraspToyMesh();
            target.AddComponent<MeshFilter>().sharedMesh = targetMesh;
            target.AddComponent<MeshRenderer>().sharedMaterial = MaterialFor("interactive_target", new Color(0.92f, 0.24f, 0.10f), 0.24f);
            var targetCollider = target.AddComponent<SphereCollider>();
            targetCollider.radius = 0.5f;
            targetCollider.sharedMaterial = MakePhysicsMaterial("interactive_target", TargetStaticFriction, TargetDynamicFriction);

            SceneIdentity identity = target.AddComponent<SceneIdentity>();
            identity.persistent_id = TargetPersistentId;
            identity.semantic_id = TargetSemanticId;
            identity.instance_id = TargetInstanceId;
            identity.semantic_class = "interactive_target";
            identity.interactive = true;
            identity.physics_role = "free_non_kinematic_physx_rigidbody";
            identity.support_id = supportId;
            identity.license = "repository-authored";
            identity.source_sha256 = TargetGeometrySpecSha256;

            PhysicsTruthObjectIdentity truthIdentity = target.AddComponent<PhysicsTruthObjectIdentity>();
            truthIdentity.persistent_id = TargetPersistentId;
            truthIdentity.semantic_id = TargetSemanticId.ToString();
            truthIdentity.instance_id = TargetInstanceId.ToString();

            Rigidbody body = target.AddComponent<Rigidbody>();
            body.mass = TargetMassKg;
            body.useGravity = true;
            body.isKinematic = false;
            body.interpolation = RigidbodyInterpolation.None;
            body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            body.maxAngularVelocity = 40f;
            body.sleepThreshold = 0.002f;
            body.solverIterations = 24;
            body.solverVelocityIterations = 12;
            _context.TargetBody = body;
        }

        private SceneSpecReceipt CompileSceneSpec(CompiledInstance table, Transform rightShoulder, Transform leftShoulder, float desiredReachM)
        {
            SceneZone[] zoneComponents = _sceneRoot.GetComponentsInChildren<SceneZone>(true);
            var zones = zoneComponents.Select(zone => new ZoneReceipt
            {
                zone_id = zone.zone_id,
                center_world_m = zone.center_world_m,
                extents_m = zone.extents_m,
                semantic_purpose = zone.semantic_purpose
            }).ToArray();
            Vector3 targetPosition = _context.TargetBody.position;
            Vector3 sightlineOrigin = _context.Head.position;
            return new SceneSpecReceipt
            {
                schema = SceneSchema,
                seed = _context.Seed,
                room_family = _context.RoomFamily,
                material_variant = _profile.MaterialVariant,
                envelope_m = _profile.EnvelopeM,
                zones = zones,
                instances = _instances.Select(instance => instance.WithoutRuntimeReferences()).ToArray(),
                target = new TargetReceipt
                {
                    persistent_id = TargetPersistentId,
                    semantic_id = TargetSemanticId,
                    instance_id = TargetInstanceId,
                    geometry = "repository_authored_rounded_grasp_toy",
                    geometry_spec_sha256 = TargetGeometrySpecSha256,
                    license = "repository-authored",
                    dimensions_m = Vector3.one * TargetDiameterM,
                    mass_kg = TargetMassKg,
                    static_friction = TargetStaticFriction,
                    dynamic_friction = TargetDynamicFriction,
                    collision_source = "analytic_sphere_matching_render_envelope",
                    collision_policy = "free_non_kinematic_physx_rigidbody",
                    support_id = table.PersistentId,
                    initial_position_world_m = targetPosition,
                    initial_rotation_world_xyzw = _context.TargetBody.rotation,
                    post_initialization_transform_writes = 0
                },
                support_relations = new[]
                {
                    new SupportRelation
                    {
                        child_id = TargetPersistentId,
                        support_id = table.PersistentId,
                        initial_separation_m = TargetBounds().min.y - table.Bounds.max.y
                    }
                },
                reachability = new ReachabilityReceipt
                {
                    anchor_id = rightShoulder.name,
                    anchor_provenance = "engine_observed_weighted_avatar_transform_at_scene_initialization",
                    target_from_right_shoulder_m = Vector3.Distance(rightShoulder.position, targetPosition),
                    target_from_left_shoulder_m = Vector3.Distance(leftShoulder.position, targetPosition),
                    compiled_requested_m = desiredReachM,
                    compiled_midpoint_band_m = new Vector2(0.34f, 0.38f),
                    lateral_bias_toward_right_shoulder_m = TargetLateralBiasTowardRightShoulderM,
                    aperture_aware = true,
                    seed_specific_retuning = false,
                    compiled_limit_m = new Vector2(0.34f, 0.48f),
                    corridor_radius_m = 0.055f
                },
                sightlines = new SightlineReceipt
                {
                    origin_id = _context.Head.name,
                    origin_world_m = sightlineOrigin,
                    target_world_m = targetPosition,
                    target_visible_from_head_geometry = HasClearSightline(sightlineOrigin, _context.TargetBody.gameObject),
                    final_gaze_zone = "window_scan"
                },
                stabilization_s = 1.0f,
                no_visible_primitive_furniture = true,
                catalog_id = CatalogId,
                catalog_archive_sha256 = CatalogArchiveSha256,
                catalog_license = CatalogLicense
            };
        }

        private SceneValidationReceipt ValidateCompiledScene(SceneSpecReceipt spec, CompiledInstance table, Transform rightShoulder, Transform leftShoulder)
        {
            GameObject target = _context.TargetBody.gameObject;
            bool targetUnparented = target.transform.parent == null;
            bool targetFree = !_context.TargetBody.isKinematic && _context.TargetBody.useGravity;
            bool noJoints = target.GetComponentsInChildren<Joint>(true).Length == 0;
            bool targetColliderValid = target.GetComponentsInChildren<Collider>(true).Any(collider => collider.enabled);
            bool reachableColliders = _instances.Where(instance => instance.reachable)
                .All(instance => instance.game_object.GetComponentsInChildren<Collider>(true).Any(collider => collider.enabled));
            bool importedHeroFurniture = _instances.All(instance => instance.visible_geometry_source == "verified_imported_kenney_mesh");
            bool hashesVerified = _instances.All(instance => instance.source_sha256 == FindCatalog(instance.asset_id).Sha256);
            float rightReach = Vector3.Distance(rightShoulder.position, _context.TargetBody.position);
            float leftReach = Vector3.Distance(leftShoulder.position, _context.TargetBody.position);
            float supportSeparation = TargetBounds().min.y - table.Bounds.max.y;
            bool reachPass = rightReach >= 0.34f && rightReach <= 0.48f && leftReach >= 0.34f && leftReach <= 0.48f;
            bool supportPass = supportSeparation >= 0f && supportSeparation <= 0.002f;
            bool sightlinePass = spec.sightlines.target_visible_from_head_geometry;
            bool assistanceClean = _context.AssistanceLedger.Count == 0;
            bool passed = targetUnparented && targetFree && noJoints && targetColliderValid && reachableColliders &&
                          importedHeroFurniture && hashesVerified && reachPass && supportPass && sightlinePass && assistanceClean;
            if (!passed)
            {
                throw new InvalidOperationException(
                    "compiled SceneSpec failed static/runtime scene validation: " +
                    string.Join(",", new[]
                    {
                        "target_unparented=" + targetUnparented,
                        "target_free=" + targetFree,
                        "no_joints=" + noJoints,
                        "target_collider=" + targetColliderValid,
                        "reachable_colliders=" + reachableColliders,
                        "imported_furniture=" + importedHeroFurniture,
                        "hashes=" + hashesVerified,
                        "reach=" + reachPass,
                        "support=" + supportPass,
                        "sightline=" + sightlinePass,
                        "assistance=" + assistanceClean
                    }));
            }
            return new SceneValidationReceipt
            {
                schema = ReceiptSchema,
                episode_id = _context.EpisodeId,
                seed = _context.Seed,
                room_family = _context.RoomFamily,
                material_variant = _profile.MaterialVariant,
                status = "COMPILED_CONTRACT_VALIDATED_ONLY",
                visual_pass_claimed = false,
                physical_episode_pass_claimed = false,
                source_hashes_verified = hashesVerified,
                imported_visible_furniture_only = importedHeroFurniture,
                reachable_elements_have_physx_colliders = reachableColliders,
                distant_decor_explicit_noninteractive = _instances.Where(instance => !instance.reachable).All(instance => !instance.interactive),
                target_unparented = targetUnparented,
                target_non_kinematic = targetFree,
                target_joint_count = target.GetComponentsInChildren<Joint>(true).Length,
                target_post_initialization_transform_writes = 0,
                target_external_force_or_spring_commands = 0,
                target_support_initial_separation_m = supportSeparation,
                measured_right_shoulder_target_distance_m = rightReach,
                measured_left_shoulder_target_distance_m = leftReach,
                head_target_sightline_clear = sightlinePass,
                assistance_ledger_entries_at_compile = _context.AssistanceLedger.Count,
                no_seed_specific_controller_or_scene_code = true,
                deterministic_selection_inputs = "GateContext.Seed + GateContext.RoomFamily + frozen catalog/profile data",
                disclosure = "Scene compilation receipt only. It is not visual evidence, contact evidence, interaction evidence, or a PASS decision."
            };
        }

        private void WriteReceipts(SceneSpecReceipt spec, SceneValidationReceipt validation)
        {
            Directory.CreateDirectory(_context.OutputRoot);
            File.WriteAllText(Path.Combine(_context.OutputRoot, "scene_spec.json"), JsonUtility.ToJson(spec, true));
            File.WriteAllText(Path.Combine(_context.OutputRoot, "scene_compiler_validation.json"), JsonUtility.ToJson(validation, true));
        }

        private bool HasClearSightline(Vector3 origin, GameObject target)
        {
            Vector3 destination = target.GetComponent<Collider>().bounds.center;
            Vector3 direction = destination - origin;
            RaycastHit[] hits = Physics.RaycastAll(origin, direction.normalized, direction.magnitude + TargetDiameterM, ~0, QueryTriggerInteraction.Ignore);
            foreach (RaycastHit hit in hits.OrderBy(hit => hit.distance))
            {
                if (_context.AvatarRoot && (hit.transform == _context.AvatarRoot.transform || hit.transform.IsChildOf(_context.AvatarRoot.transform)))
                    continue;
                return hit.transform == target.transform || hit.transform.IsChildOf(target.transform);
            }
            return false;
        }

        private static Transform FindRightShoulder(GateContext context)
        {
            Transform shoulder = context.AvatarRoot.GetComponentsInChildren<Transform>(true)
                .FirstOrDefault(transform => transform.name == "upperarm01.R");
            if (!shoulder)
            {
                for (Transform cursor = context.RightPalm; cursor; cursor = cursor.parent)
                {
                    if (cursor.name.IndexOf("upperarm", StringComparison.OrdinalIgnoreCase) >= 0)
                        shoulder = cursor;
                }
            }
            if (!shoulder) throw new InvalidOperationException("could not locate the bound right shoulder for target initialization");
            return shoulder;
        }

        private static Transform FindLeftShoulder(GateContext context)
        {
            Transform shoulder = context.AvatarRoot.GetComponentsInChildren<Transform>(true)
                .FirstOrDefault(transform => transform.name == "upperarm01.L");
            if (!shoulder)
            {
                for (Transform cursor = context.LeftPalm; cursor; cursor = cursor.parent)
                {
                    if (cursor.name.IndexOf("upperarm", StringComparison.OrdinalIgnoreCase) >= 0)
                        shoulder = cursor;
                }
            }
            if (!shoulder) throw new InvalidOperationException("could not locate the bound left shoulder for target initialization");
            return shoulder;
        }

        private Collider[] AddCatalogColliders(GameObject instance, string collisionSource)
        {
            var colliders = new List<Collider>();
            if (collisionSource == "box")
            {
                Bounds localBounds = LocalGeometryBounds(instance.transform);
                BoxCollider box = instance.AddComponent<BoxCollider>();
                box.center = localBounds.center;
                box.size = localBounds.size;
                colliders.Add(box);
            }
            else if (collisionSource == "mesh_or_compound_box")
            {
                foreach (MeshFilter filter in instance.GetComponentsInChildren<MeshFilter>(true))
                {
                    if (!filter.sharedMesh) continue;
                    MeshCollider meshCollider = filter.gameObject.AddComponent<MeshCollider>();
                    meshCollider.sharedMesh = filter.sharedMesh;
                    meshCollider.convex = false;
                    colliders.Add(meshCollider);
                }
            }
            else
            {
                foreach (MeshFilter filter in instance.GetComponentsInChildren<MeshFilter>(true))
                {
                    if (!filter.sharedMesh) continue;
                    BoxCollider box = filter.gameObject.AddComponent<BoxCollider>();
                    box.center = filter.sharedMesh.bounds.center;
                    box.size = filter.sharedMesh.bounds.size;
                    colliders.Add(box);
                }
            }
            if (colliders.Count == 0) throw new InvalidOperationException("no collider source geometry found for " + instance.name);
            return colliders.ToArray();
        }

        private void ScaleToCatalogDimensions(GameObject instance, Vector3 dimensionsM)
        {
            Bounds local = LocalGeometryBounds(instance.transform);
            if (local.size.x <= 0f || local.size.y <= 0f || local.size.z <= 0f)
                throw new InvalidOperationException("invalid imported Renderer bounds for " + instance.name);
            instance.transform.localScale = new Vector3(
                dimensionsM.x / local.size.x,
                dimensionsM.y / local.size.y,
                dimensionsM.z / local.size.z);
        }

        private static Bounds LocalGeometryBounds(Transform root)
        {
            bool initialized = false;
            Bounds bounds = new Bounds();
            foreach (MeshFilter filter in root.GetComponentsInChildren<MeshFilter>(true))
            {
                if (!filter.sharedMesh) continue;
                Bounds meshBounds = filter.sharedMesh.bounds;
                foreach (Vector3 corner in BoundsCorners(meshBounds))
                {
                    Vector3 local = root.InverseTransformPoint(filter.transform.TransformPoint(corner));
                    if (!initialized)
                    {
                        bounds = new Bounds(local, Vector3.zero);
                        initialized = true;
                    }
                    else bounds.Encapsulate(local);
                }
            }
            if (!initialized) throw new InvalidOperationException("catalog object has no MeshFilter geometry: " + root.name);
            return bounds;
        }

        private static IEnumerable<Vector3> BoundsCorners(Bounds bounds)
        {
            Vector3 min = bounds.min;
            Vector3 max = bounds.max;
            yield return new Vector3(min.x, min.y, min.z);
            yield return new Vector3(min.x, min.y, max.z);
            yield return new Vector3(min.x, max.y, min.z);
            yield return new Vector3(min.x, max.y, max.z);
            yield return new Vector3(max.x, min.y, min.z);
            yield return new Vector3(max.x, min.y, max.z);
            yield return new Vector3(max.x, max.y, min.z);
            yield return new Vector3(max.x, max.y, max.z);
        }

        private static Bounds BoundsOf(GameObject gameObject)
        {
            Renderer[] renderers = gameObject.GetComponentsInChildren<Renderer>(true);
            if (renderers.Length == 0) throw new InvalidOperationException("object has no visible Renderer geometry: " + gameObject.name);
            Bounds result = renderers[0].bounds;
            for (int index = 1; index < renderers.Length; index++) result.Encapsulate(renderers[index].bounds);
            return result;
        }

        private Bounds TargetBounds()
        {
            return _context.TargetBody.GetComponent<Collider>().bounds;
        }

        private Material FurnitureMaterial(string semanticClass)
        {
            Color baseColor;
            switch (_profile.MaterialVariant)
            {
                case "painted_sage": baseColor = new Color(0.31f, 0.56f, 0.42f); break;
                case "natural_birch": baseColor = new Color(0.76f, 0.64f, 0.44f); break;
                default: baseColor = new Color(0.55f, 0.34f, 0.18f); break;
            }
            if (semanticClass == "rug") baseColor = Color.Lerp(_profile.AccentColor, Color.white, 0.12f);
            else if (semanticClass == "plant") baseColor = new Color(0.20f, 0.48f, 0.25f);
            else if (semanticClass == "lamp") baseColor = Color.Lerp(_profile.AccentColor, Color.white, 0.35f);
            else if (semanticClass == "books") baseColor = new Color(0.74f, 0.29f, 0.17f);
            return MaterialFor(_profile.MaterialVariant + "_" + semanticClass, baseColor, semanticClass == "rug" ? 0.05f : 0.20f);
        }

        private Material MaterialFor(string key, Color color, float glossiness)
        {
            if (_materials.TryGetValue(key, out Material cached)) return cached;
            Shader shader = Shader.Find("Standard");
            if (!shader) shader = Shader.Find("Universal Render Pipeline/Lit");
            if (!shader) throw new InvalidOperationException("no supported Unity lit shader is available");
            var material = new Material(shader) { name = "Runtime_" + key, color = color };
            if (material.HasProperty("_Glossiness")) material.SetFloat("_Glossiness", glossiness);
            if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", glossiness);
            _materials[key] = material;
            return material;
        }

#if UNITY_6000_0_OR_NEWER
        private static PhysicsMaterial MakePhysicsMaterial(string role, float staticFriction, float dynamicFriction)
        {
            var material = new PhysicsMaterial("Runtime_" + role + "_physics");
#else
        private static PhysicMaterial MakePhysicsMaterial(string role, float staticFriction, float dynamicFriction)
        {
            var material = new PhysicMaterial("Runtime_" + role + "_physics");
#endif
            material.staticFriction = staticFriction;
            material.dynamicFriction = dynamicFriction;
            material.bounciness = 0.02f;
            material.frictionCombine = PhysicsMaterialCombine.Maximum;
            material.bounceCombine = PhysicsMaterialCombine.Minimum;
            return material;
        }

        private static Mesh UnitBoxMesh()
        {
            var mesh = new Mesh { name = "AuthoredRoomSurfaceUnitBox" };
            Vector3[] vertices =
            {
                new Vector3(-.5f,-.5f,-.5f), new Vector3(.5f,-.5f,-.5f), new Vector3(.5f,.5f,-.5f), new Vector3(-.5f,.5f,-.5f),
                new Vector3(-.5f,-.5f,.5f), new Vector3(-.5f,.5f,.5f), new Vector3(.5f,.5f,.5f), new Vector3(.5f,-.5f,.5f),
                new Vector3(-.5f,-.5f,-.5f), new Vector3(-.5f,.5f,-.5f), new Vector3(-.5f,.5f,.5f), new Vector3(-.5f,-.5f,.5f),
                new Vector3(.5f,-.5f,-.5f), new Vector3(.5f,-.5f,.5f), new Vector3(.5f,.5f,.5f), new Vector3(.5f,.5f,-.5f),
                new Vector3(-.5f,.5f,-.5f), new Vector3(.5f,.5f,-.5f), new Vector3(.5f,.5f,.5f), new Vector3(-.5f,.5f,.5f),
                new Vector3(-.5f,-.5f,-.5f), new Vector3(-.5f,-.5f,.5f), new Vector3(.5f,-.5f,.5f), new Vector3(.5f,-.5f,-.5f)
            };
            int[] triangles =
            {
                0,2,1,0,3,2,4,6,5,4,7,6,8,10,9,8,11,10,
                12,14,13,12,15,14,16,18,17,16,19,18,20,22,21,20,23,22
            };
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static Mesh RoundedGraspToyMesh()
        {
            const int longitudeSegments = 48;
            const int latitudeSegments = 24;
            var vertices = new List<Vector3> { Vector3.up * 0.5f };
            var uvs = new List<Vector2> { new Vector2(0.5f, 1f) };
            for (int latitude = 1; latitude < latitudeSegments; latitude++)
            {
                float phi = Mathf.PI * latitude / latitudeSegments;
                float y = Mathf.Cos(phi) * 0.5f;
                float ringRadius = Mathf.Sin(phi) * 0.5f;
                for (int longitude = 0; longitude < longitudeSegments; longitude++)
                {
                    float theta = Mathf.PI * 2f * longitude / longitudeSegments;
                    vertices.Add(new Vector3(Mathf.Cos(theta) * ringRadius, y, Mathf.Sin(theta) * ringRadius));
                    uvs.Add(new Vector2(longitude / (float)longitudeSegments, 1f - latitude / (float)latitudeSegments));
                }
            }
            int bottomIndex = vertices.Count;
            vertices.Add(Vector3.down * 0.5f);
            uvs.Add(new Vector2(0.5f, 0f));
            var triangles = new List<int>();
            for (int longitude = 0; longitude < longitudeSegments; longitude++)
            {
                int next = (longitude + 1) % longitudeSegments;
                triangles.Add(0); triangles.Add(1 + longitude); triangles.Add(1 + next);
            }
            for (int latitude = 0; latitude < latitudeSegments - 2; latitude++)
            {
                int ring = 1 + latitude * longitudeSegments;
                int nextRing = ring + longitudeSegments;
                for (int longitude = 0; longitude < longitudeSegments; longitude++)
                {
                    int next = (longitude + 1) % longitudeSegments;
                    triangles.Add(ring + longitude); triangles.Add(nextRing + longitude); triangles.Add(nextRing + next);
                    triangles.Add(ring + longitude); triangles.Add(nextRing + next); triangles.Add(ring + next);
                }
            }
            int finalRing = 1 + (latitudeSegments - 2) * longitudeSegments;
            for (int longitude = 0; longitude < longitudeSegments; longitude++)
            {
                int next = (longitude + 1) % longitudeSegments;
                triangles.Add(finalRing + longitude); triangles.Add(bottomIndex); triangles.Add(finalRing + next);
            }
            var mesh = new Mesh { name = "RepositoryAuthoredRoundedGraspToy" };
            mesh.SetVertices(vertices);
            mesh.SetUVs(0, uvs);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static CatalogMember FindCatalog(string assetId)
        {
            CatalogMember member = Catalog.SingleOrDefault(candidate => candidate.AssetId == assetId);
            if (member == null) throw new InvalidOperationException("room profile references an unfrozen catalog member: " + assetId);
            return member;
        }

        private static void VerifyCatalogSource(string assetPath, string frozenSha256)
        {
            string normalized = assetPath.Replace('\\', '/');
            string absolutePath = Path.Combine(Application.dataPath, normalized.Substring("Assets/".Length));
            if (!File.Exists(absolutePath)) throw new FileNotFoundException("missing frozen catalog source", absolutePath);
            using (SHA256 hash = SHA256.Create())
            using (FileStream stream = File.OpenRead(absolutePath))
            {
                string actual = BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant();
                if (!string.Equals(actual, frozenSha256, StringComparison.Ordinal))
                    throw new InvalidOperationException("catalog source hash mismatch for " + assetPath + ": " + actual);
            }
        }

        private static int SemanticId(string semanticClass)
        {
            switch (semanticClass)
            {
                case "floor": return 1;
                case "wall": return 2;
                case "window": return 3;
                case "trim": return 4;
                case "support_table": return 20;
                case "sofa": return 21;
                case "storage": return 22;
                case "chair": return 23;
                case "rug": return 24;
                case "lamp": return 25;
                case "plant": return 26;
                case "books": return 27;
                default: return 10;
            }
        }

        private static int StableInstanceId(int seed, string persistentId)
        {
            unchecked
            {
                uint hash = 2166136261;
                string text = seed + ":" + persistentId;
                foreach (char character in text)
                {
                    hash ^= character;
                    hash *= 16777619;
                }
                return 100000 + (int)(hash % 900000000);
            }
        }

        private static int PositiveModulo(int value, int divisor)
        {
            int remainder = value % divisor;
            return remainder < 0 ? remainder + divisor : remainder;
        }

        private float NextSigned(float magnitude)
        {
            return ((float)_random.NextDouble() * 2f - 1f) * magnitude;
        }

        private static string ZonePurpose(string zoneId)
        {
            if (zoneId == "low_table") return "reachable support and interaction corridor";
            if (zoneId == "window_scan") return "initial scan and final gaze sightline";
            if (zoneId == "rug") return "embodiment stance and furniture spacing";
            return "noninteractive contextual furnishing zone";
        }

        private sealed class CatalogMember
        {
            public readonly string AssetId;
            public readonly string Sha256;
            public readonly Vector3 DimensionsM;
            public readonly string SemanticClass;
            public readonly string CollisionSource;

            public CatalogMember(string assetId, string sha256, Vector3 dimensionsM, string semanticClass, string collisionSource)
            {
                AssetId = assetId;
                Sha256 = sha256;
                DimensionsM = dimensionsM;
                SemanticClass = semanticClass;
                CollisionSource = collisionSource;
            }
        }

        private sealed class Placement
        {
            public readonly string AssetId;
            public readonly float LocalX;
            public readonly float LocalDepth;
            public readonly float YawDeg;
            public readonly bool Reachable;
            public readonly string Role;

            public Placement(string assetId, float localX, float localDepth, float yawDeg, bool reachable, string role)
            {
                AssetId = assetId;
                LocalX = localX;
                LocalDepth = localDepth;
                YawDeg = yawDeg;
                Reachable = reachable;
                Role = role;
            }
        }

        private sealed class RoomProfile
        {
            public readonly string RoomFamily;
            public readonly string MaterialVariant;
            public readonly Vector3 EnvelopeM;
            public readonly string[] ZoneIds;
            public readonly Color WallColor;
            public readonly Color FloorColor;
            public readonly Color AccentColor;
            public readonly Placement[] Placements;

            public RoomProfile(string roomFamily, string materialVariant, Vector3 envelopeM, string[] zoneIds, Color wallColor, Color floorColor, Color accentColor, Placement[] placements)
            {
                RoomFamily = roomFamily;
                MaterialVariant = materialVariant;
                EnvelopeM = envelopeM;
                ZoneIds = zoneIds;
                WallColor = wallColor;
                FloorColor = floorColor;
                AccentColor = accentColor;
                Placements = placements;
            }
        }
    }

    public sealed class SceneIdentity : MonoBehaviour
    {
        public string persistent_id;
        public int semantic_id;
        public int instance_id;
        public string semantic_class;
        public bool interactive;
        public string physics_role;
        public string support_id;
        public string license;
        public string source_sha256;
    }

    public sealed class SceneZone : MonoBehaviour
    {
        public string zone_id;
        public Vector3 center_world_m;
        public Vector3 extents_m;
        public string semantic_purpose;
    }

    [Serializable]
    public sealed class SceneSpecReceipt
    {
        public string schema;
        public int seed;
        public string room_family;
        public string material_variant;
        public Vector3 envelope_m;
        public ZoneReceipt[] zones;
        public CompiledInstance[] instances;
        public TargetReceipt target;
        public SupportRelation[] support_relations;
        public ReachabilityReceipt reachability;
        public SightlineReceipt sightlines;
        public float stabilization_s;
        public bool no_visible_primitive_furniture;
        public string catalog_id;
        public string catalog_archive_sha256;
        public string catalog_license;
    }

    [Serializable]
    public sealed class CompiledInstance
    {
        public string persistent_id;
        public string asset_id;
        public string asset_path;
        public string source_sha256;
        public string license;
        public Vector3 catalog_dimensions_m;
        public Vector3 compiled_bounds_center_world_m;
        public Vector3 compiled_bounds_size_m;
        public Quaternion rotation_world_xyzw;
        public string semantic_class;
        public int semantic_id;
        public int instance_id;
        public string material_variant;
        public string collision_source;
        public int collider_count;
        public bool interactive;
        public bool reachable;
        public string role;
        public string visible_geometry_source;
        [NonSerialized] public GameObject game_object;
        [NonSerialized] public Bounds Bounds;

        public string PersistentId { get { return persistent_id; } }

        public CompiledInstance WithoutRuntimeReferences()
        {
            return new CompiledInstance
            {
                persistent_id = persistent_id,
                asset_id = asset_id,
                asset_path = asset_path,
                source_sha256 = source_sha256,
                license = license,
                catalog_dimensions_m = catalog_dimensions_m,
                compiled_bounds_center_world_m = compiled_bounds_center_world_m,
                compiled_bounds_size_m = compiled_bounds_size_m,
                rotation_world_xyzw = rotation_world_xyzw,
                semantic_class = semantic_class,
                semantic_id = semantic_id,
                instance_id = instance_id,
                material_variant = material_variant,
                collision_source = collision_source,
                collider_count = collider_count,
                interactive = interactive,
                reachable = reachable,
                role = role,
                visible_geometry_source = visible_geometry_source
            };
        }
    }

    [Serializable]
    public sealed class ZoneReceipt
    {
        public string zone_id;
        public Vector3 center_world_m;
        public Vector3 extents_m;
        public string semantic_purpose;
    }

    [Serializable]
    public sealed class TargetReceipt
    {
        public string persistent_id;
        public int semantic_id;
        public int instance_id;
        public string geometry;
        public string geometry_spec_sha256;
        public string license;
        public Vector3 dimensions_m;
        public float mass_kg;
        public float static_friction;
        public float dynamic_friction;
        public string collision_source;
        public string collision_policy;
        public string support_id;
        public Vector3 initial_position_world_m;
        public Quaternion initial_rotation_world_xyzw;
        public int post_initialization_transform_writes;
    }

    [Serializable]
    public sealed class SupportRelation
    {
        public string child_id;
        public string support_id;
        public float initial_separation_m;
    }

    [Serializable]
    public sealed class ReachabilityReceipt
    {
        public string anchor_id;
        public string anchor_provenance;
        public float target_from_right_shoulder_m;
        public float target_from_left_shoulder_m;
        public float compiled_requested_m;
        public Vector2 compiled_midpoint_band_m;
        public float lateral_bias_toward_right_shoulder_m;
        public bool aperture_aware;
        public bool seed_specific_retuning;
        public Vector2 compiled_limit_m;
        public float corridor_radius_m;
    }

    [Serializable]
    public sealed class SightlineReceipt
    {
        public string origin_id;
        public Vector3 origin_world_m;
        public Vector3 target_world_m;
        public bool target_visible_from_head_geometry;
        public string final_gaze_zone;
    }

    [Serializable]
    public sealed class SceneValidationReceipt
    {
        public string schema;
        public string episode_id;
        public int seed;
        public string room_family;
        public string material_variant;
        public string status;
        public bool visual_pass_claimed;
        public bool physical_episode_pass_claimed;
        public bool source_hashes_verified;
        public bool imported_visible_furniture_only;
        public bool reachable_elements_have_physx_colliders;
        public bool distant_decor_explicit_noninteractive;
        public bool target_unparented;
        public bool target_non_kinematic;
        public int target_joint_count;
        public int target_post_initialization_transform_writes;
        public int target_external_force_or_spring_commands;
        public float target_support_initial_separation_m;
        public float measured_right_shoulder_target_distance_m;
        public float measured_left_shoulder_target_distance_m;
        public bool head_target_sightline_clear;
        public int assistance_ledger_entries_at_compile;
        public bool no_seed_specific_controller_or_scene_code;
        public string deterministic_selection_inputs;
        public string disclosure;
    }
}
#endif
