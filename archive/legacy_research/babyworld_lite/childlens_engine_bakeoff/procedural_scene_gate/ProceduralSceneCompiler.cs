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
        private const float FurnitureStaticFriction = 0.75f;
        private const float FurnitureDynamicFriction = 0.65f;
        private const float SupportGapM = 0.001f;
        private const float ProspectiveViewHalfAngleDeg = 37.5f;

        private static readonly TargetDefinition[] Targets =
        {
            new TargetDefinition(
                "red_toy_001", 41, 41001, "rounded_toy",
                "9d9963e46446fb05187c285062e7562af142a82906b5adf3f491f1d70060e5c5",
                new Vector3(0.055f, 0.055f, 0.055f), 0.09f, 1.0f, 0.9f,
                "analytic_sphere", "radial_thumb_index_middle", new Color(0.92f, 0.18f, 0.10f)),
            new TargetDefinition(
                "blue_cup_001", 42, 42001, "handled_cup",
                "f79ab83b8850ec62bfe84c07b81793e42bfed6ae8158b06736e4467f05f96885",
                new Vector3(0.065f, 0.075f, 0.065f), 0.08f, 0.9f, 0.8f,
                "compound_analytic", "cup_body_thumb_middle_ring", new Color(0.10f, 0.38f, 0.88f)),
            new TargetDefinition(
                "yellow_block_001", 43, 43001, "beveled_block",
                "c60dba9e25789265ef8aecc27f670fe0ca8988feefc9c7c9ea4a9bf74de14e06",
                new Vector3(0.060f, 0.050f, 0.060f), 0.10f, 0.95f, 0.85f,
                "analytic_box", "face_opposition_thumb_index_middle", new Color(0.96f, 0.72f, 0.08f))
        };

        private static readonly CatalogMember[] Catalog =
        {
            new CatalogMember("tableCoffee", "8c81c31a74aadc1e89e334eb1ac43380f41c14f47e5dd9a6b029ebf74850adda", new Vector3(1.15f, .62f, .86f), "support_table", "mesh_or_compound_box"),
            new CatalogMember("loungeSofaLong", "bbc91d1b01537dbc847685560265b6fcd72a21333c8aab7f8906a5addf8dbd20", new Vector3(1.65f, .92f, .78f), "sofa", "compound_box"),
            new CatalogMember("bookcaseOpen", "31beea66f34e64660c870e39c52ed2e81984939c88f4bd763c7dce56e9ef76a9", new Vector3(.72f, 1.64f, .35f), "storage", "compound_box"),
            new CatalogMember("chairCushion", "afc684f28ab2e4ee3fe9cd59a92f03103e4f62c74ae785029fdd672fc6914cdd", new Vector3(.55f, 1.24f, .55f), "chair", "compound_box"),
            new CatalogMember("rugRectangle", "0cdd8cea357bc267c9f121b4e7d32513c08f1b179522d95c6632bf2306d38f79", new Vector3(1.75f, .015f, 1.20f), "rug", "box"),
            new CatalogMember("lampSquareFloor", "c4b35656f587cacf2ccf57d1c0216541923890e878cec0d0f58ff71cd4a71bfc", new Vector3(.42f, 1.35f, .42f), "lamp", "compound_box"),
            new CatalogMember("pottedPlant", "cf66928b943e30ef5c1f06145419ca2cda534249bb4d77831ae723532c4f8de3", new Vector3(.55f, 1.00f, .55f), "plant", "compound_box"),
            new CatalogMember("books", "91fd64f5fd821b5046724ffa18974e8964b85bfbd4a0d89687d2283966394f5f", new Vector3(.32f, .14f, .24f), "books", "box")
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
                    new Placement("chairCushion", 1.10f, 2.35f, 212f, false, "reading_corner"),
                    new Placement("books", -1.54f, 3.05f, 8f, false, "shelf_clutter"),
                    new Placement("books", -0.85f, 3.98f, -9f, false, "toy_shelf_books"),
                    new Placement("pottedPlant", 1.72f, 3.80f, 179f, false, "window_plant"),
                    new Placement("lampSquareFloor", 1.85f, 3.10f, 184f, false, "reading_lamp"),
                    new Placement("chairCushion", -1.35f, 2.30f, 148f, false, "caregiver_chair"),
                    new Placement("books", 0.92f, 3.38f, 14f, false, "picture_books"),
                    new Placement("rugRectangle", -0.15f, 3.42f, 0f, false, "back_play_mat")
                }),
            new RoomProfile(
                "sage_living_corner", "painted_sage", new Vector3(4.8f, 2.55f, 5.2f),
                new[] { "sofa", "low_table", "bookcase", "window_scan" },
                new Color(0.65f, 0.73f, 0.66f), new Color(0.35f, 0.27f, 0.20f), new Color(0.75f, 0.84f, 0.66f),
                new[]
                {
                    new Placement("loungeSofaLong", 0.65f, 4.02f, 2f, false, "sofa"),
                    new Placement("rugRectangle", 0.05f, 1.85f, 90f, true, "rug"),
                    new Placement("lampSquareFloor", 1.92f, 3.48f, 185f, false, "distant_lamp"),
                    new Placement("pottedPlant", -1.92f, 3.20f, 178f, false, "distant_plant"),
                    new Placement("books", -1.58f, 4.18f, -6f, false, "bookcase"),
                    new Placement("chairCushion", -1.35f, 2.24f, 154f, false, "reading_chair"),
                    new Placement("books", 1.18f, 3.08f, 11f, false, "side_books"),
                    new Placement("pottedPlant", 1.94f, 2.55f, 183f, false, "window_plant"),
                    new Placement("rugRectangle", -0.20f, 3.50f, 0f, false, "sofa_rug"),
                    new Placement("books", -0.92f, 3.72f, 7f, false, "family_books")
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
                    new Placement("pottedPlant", 1.68f, 3.46f, 183f, false, "distant_plant"),
                    new Placement("books", -0.82f, 3.05f, 6f, false, "art_books"),
                    new Placement("lampSquareFloor", -1.75f, 2.48f, 176f, false, "task_lamp"),
                    new Placement("books", 0.10f, 3.05f, -12f, false, "portfolio_stack"),
                    new Placement("chairCushion", 0.82f, 3.46f, 214f, false, "observer_chair"),
                    new Placement("rugRectangle", 0.10f, 3.28f, 0f, false, "gallery_rug"),
                    new Placement("books", -0.78f, 3.68f, 9f, false, "paper_stack")
                })
        };

        private readonly List<CompiledInstance> _instances = new List<CompiledInstance>();
        private readonly Dictionary<string, Material> _materials = new Dictionary<string, Material>();
        private RoomProfile _profile;
        private GateContext _context;
        private Transform _sceneRoot;
        private string _furnitureAssetRoot;
        private System.Random _random;
        private TargetDefinition _target;
        private CompiledInstance _destination;

        public void Build(GateContext context, string furnitureAssetRoot)
        {
            ValidateInputs(context, furnitureAssetRoot);
            _context = context;
            _furnitureAssetRoot = furnitureAssetRoot.TrimEnd('/', '\\');
            _profile = Profiles.Single(profile => profile.RoomFamily == context.RoomFamily);
            _target = ResolveTarget(context);
            _random = new System.Random(context.Seed);
            _instances.Clear();
            context.Destinations.Clear();

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

            float desiredReachM = context.TargetMidpointReachM;
            Transform rightShoulder = FindRightShoulder(context);
            Transform leftShoulder = FindLeftShoulder(context);
            Vector3 shoulderMidpoint = 0.5f * (leftShoulder.position + rightShoulder.position);
            Vector3 targetPosition = SolveTargetPosition(shoulderMidpoint, table.Bounds.max.y, desiredReachM, _target.DimensionsM.y);
            Vector3 towardRightShoulder = Vector3.ProjectOnPlane(rightShoulder.position - shoulderMidpoint, Vector3.up).normalized;
            targetPosition += towardRightShoulder * context.TargetLateralBiasM;
            MoveStaticSupportUnderTarget(table, targetPosition);
            targetPosition.y = table.Bounds.max.y + _target.DimensionsM.y * 0.5f + SupportGapM;

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

            BindGazeZoneTransforms(table);
            Physics.SyncTransforms();
            _destination = CreateActivityContext(table, targetPosition);
            Physics.SyncTransforms();
            RejectOverlappingFurnishings();
            RejectObjectsOutsideFinishedRoom();
            RejectFurnitureInInteractionCorridor(table, targetPosition, _destination.Bounds.center);
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
            if (string.IsNullOrWhiteSpace(context.TargetId) || string.IsNullOrWhiteSpace(context.DestinationId) ||
                string.IsNullOrWhiteSpace(context.ContactStrategy) || string.IsNullOrWhiteSpace(context.FinalGazeZone))
                throw new InvalidOperationException("GateContext must freeze TargetId, DestinationId, ContactStrategy, and FinalGazeZone");
            if (string.IsNullOrWhiteSpace(context.CompiledContractPath) || string.IsNullOrWhiteSpace(context.CompiledContractSha256))
                throw new InvalidOperationException("GateContext must be populated from the authoritative compiled contract");
            if (context.TargetReachBandMinM <= 0f || context.TargetReachBandMaxM < context.TargetReachBandMinM ||
                context.TargetMidpointReachM < context.TargetReachBandMinM || context.TargetMidpointReachM > context.TargetReachBandMaxM)
                throw new InvalidOperationException("compiled-contract target reach is unset or outside its authoritative band");
            if (context.TargetLateralBiasM < 0f || context.TargetDimensionsM.x <= 0f || context.TargetDimensionsM.y <= 0f ||
                context.TargetDimensionsM.z <= 0f || context.TargetMassKg <= 0f || context.TargetStaticFriction < 0f ||
                context.TargetDynamicFriction < 0f || string.IsNullOrWhiteSpace(context.TargetGeometry) ||
                context.TargetSemanticId <= 0 || context.TargetInstanceId <= 0)
                throw new InvalidOperationException("compiled-contract target geometry, identity, or physical parameters are unset");
            RoomProfile profile = Profiles.Single(candidate => candidate.RoomFamily == context.RoomFamily);
            if (!Approximately(context.SceneEnvelopeM, profile.EnvelopeM))
                throw new InvalidOperationException("compiled-contract scene envelope does not match the approved room profile");
            if (!string.Equals(context.SceneMaterialVariant, profile.MaterialVariant, StringComparison.Ordinal))
                throw new InvalidOperationException("compiled-contract material variant does not match the approved room profile");
            if (context.SceneZoneIds == null || !context.SceneZoneIds.SequenceEqual(profile.ZoneIds, StringComparer.Ordinal))
                throw new InvalidOperationException("compiled-contract scene zones do not match the approved room profile");
            string[] profileAssets = new[] { "tableCoffee" }.Concat(profile.Placements.Select(placement => placement.AssetId)).ToArray();
            if (context.ExpectedSceneAssetIds == null || !SameMultiset(context.ExpectedSceneAssetIds, profileAssets))
                throw new InvalidOperationException("compiled-contract scene asset identities do not match the approved room profile");
            if (context.ExpectedSceneInstances == null || context.ExpectedSceneInstances.Length != profileAssets.Length)
                throw new InvalidOperationException("compiled-contract scene instance authorities are incomplete");
            for (int index = 0; index < profileAssets.Length; index++)
            {
                SceneInstanceAuthority authority = context.ExpectedSceneInstances[index];
                CatalogMember catalog = FindCatalog(profileAssets[index]);
                string expectedPersistentId = context.RoomFamily + "_" + profileAssets[index] + "_" + index.ToString("D2");
                if (authority == null || authority.AssetId != profileAssets[index]
                    || authority.PersistentId != expectedPersistentId
                    || !Approximately(authority.AssetDimensionsM, catalog.DimensionsM)
                    || authority.SemanticClass != catalog.SemanticClass
                    || authority.CollisionSource != catalog.CollisionSource
                    || authority.Interactive || authority.MassKg != 0f
                    || authority.StaticFriction < 0f || authority.DynamicFriction < 0f)
                    throw new InvalidOperationException("compiled-contract scene instance authority diverges at index " + index);
            }
            if (context.ExpectedSupportRelations == null || context.ExpectedSupportRelations.Length != 2
                || context.ExpectedSupportRelations[0].ChildId != context.TargetId
                || context.ExpectedSupportRelations[0].SupportId != context.ExpectedSceneInstances[0].PersistentId
                || !string.IsNullOrEmpty(context.ExpectedSupportRelations[0].DestinationId)
                || context.ExpectedSupportRelations[1].ChildId != context.TargetId
                || context.ExpectedSupportRelations[1].DestinationId != context.DestinationId
                || !string.IsNullOrEmpty(context.ExpectedSupportRelations[1].SupportId))
                throw new InvalidOperationException("compiled-contract support relations are incomplete or divergent");
            if (!context.ExpectedTargetVisibleAtRequiredEvents
                || context.ExpectedFinalGazeZone != context.FinalGazeZone
                || context.SceneStabilizationSeconds <= 0f
                || !context.RequireNoVisiblePrimitiveFurniture)
                throw new InvalidOperationException("compiled-contract sightline/stabilization/visible-furniture policy is incomplete");
            if (context.MinimumContextualObjects <= 0 ||
                context.MinimumContextualObjects > context.ExpectedSceneAssetIds.Length ||
                context.MinimumContextualObjects <= 10)
                throw new InvalidOperationException("compiled-contract contextual-object minimum is inconsistent with its scene asset inventory");
            if (string.IsNullOrWhiteSpace(context.OutputRoot)) throw new InvalidOperationException("GateContext.OutputRoot is required for scene receipts");
            if (string.IsNullOrWhiteSpace(furnitureAssetRoot) || !furnitureAssetRoot.Replace('\\', '/').StartsWith("Assets/", StringComparison.Ordinal))
                throw new InvalidOperationException("furnitureAssetRoot must be a Unity project Assets path");
            if (context.AssistanceLedger.Count != 0) throw new InvalidOperationException("assistance ledger was nonempty before scene compilation");
        }

        private static bool SameMultiset(IEnumerable<string> left, IEnumerable<string> right)
        {
            return left.OrderBy(value => value, StringComparer.Ordinal)
                .SequenceEqual(right.OrderBy(value => value, StringComparer.Ordinal), StringComparer.Ordinal);
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
            float width = _context.SceneEnvelopeM.x;
            float height = _context.SceneEnvelopeM.y;
            float depth = _context.SceneEnvelopeM.z;
            float roomCenterDepth = depth * 0.5f - 0.55f;
            float backDepth = depth - 0.55f;

            CreateRoomSurface("floor_surface", new Vector3(0f, -0.04f, -roomCenterDepth), new Vector3(width, 0.08f, depth), _profile.FloorColor, "floor");
            CreateRoomSurface("back_wall_surface", new Vector3(0f, height * 0.5f, -backDepth), new Vector3(width, height, 0.08f), _profile.WallColor, "wall");
            CreateRoomSurface("left_wall_surface", new Vector3(-width * 0.5f, height * 0.5f, -roomCenterDepth), new Vector3(0.08f, height, depth), _profile.WallColor, "wall");
            CreateRoomSurface("right_wall_surface", new Vector3(width * 0.5f, height * 0.5f, -roomCenterDepth), new Vector3(0.08f, height, depth), _profile.WallColor, "wall");
            CreateRoomSurface("front_wall_surface", new Vector3(0f, height * 0.5f, 0.55f), new Vector3(width, height, 0.08f), _profile.WallColor, "wall");
            CreateRoomSurface("ceiling_surface", new Vector3(0f, height + 0.04f, -roomCenterDepth), new Vector3(width, 0.08f, depth), Color.Lerp(_profile.WallColor, Color.white, 0.22f), "ceiling");
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
            float backDepth = _context.SceneEnvelopeM.z - 0.62f;
            for (int index = 0; index < _context.SceneZoneIds.Length; index++)
            {
                string zoneId = _context.SceneZoneIds[index];
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

        private void BindGazeZoneTransforms(CompiledInstance table)
        {
            foreach (SceneZone zone in _sceneRoot.GetComponentsInChildren<SceneZone>(true))
            {
                GameObject backing;
                if (zone.zone_id == "low_table") backing = table.game_object;
                else if (zone.zone_id == "window_scan") backing = RequireSceneChild("window_scan_surface");
                else
                {
                    CompiledInstance instance = _instances.FirstOrDefault(candidate => candidate.role == zone.zone_id);
                    if (instance == null)
                        throw new InvalidOperationException("gaze zone has no visible physics-backed scene object: " + zone.zone_id);
                    backing = instance.game_object;
                }
                Bounds bounds = BoundsOf(backing);
                Transform anchor = CreateBackedAnchor("GAZE_ANCHOR_" + zone.zone_id, backing.transform, bounds.center);
                zone.transform.position = anchor.position;
                zone.center_world_m = anchor.position;
                zone.extents_m = bounds.size;
                RegisterContextDestination(zone.zone_id, anchor);
            }

            GameObject backWall = RequireSceneChild("back_wall_surface");
            Bounds backWallBounds = BoundsOf(backWall);
            Vector3 lookAwayWorld = _sceneRoot.TransformPoint(new Vector3(-0.72f, 1.30f, 0f));
            lookAwayWorld.z = backWallBounds.max.z;
            RegisterContextDestination(
                "look_away",
                CreateBackedAnchor("GAZE_ANCHOR_look_away", backWall.transform, lookAwayWorld));
        }

        private GameObject RequireSceneChild(string name)
        {
            Transform child = _sceneRoot.Find(name);
            if (!child) throw new InvalidOperationException("compiled scene is missing required physical object: " + name);
            return child.gameObject;
        }

        private static Transform CreateBackedAnchor(string name, Transform backing, Vector3 worldPosition)
        {
            if (!backing.GetComponentsInChildren<Renderer>(true).Any(renderer => renderer.enabled) ||
                !backing.GetComponentsInChildren<Collider>(true).Any(collider => collider.enabled))
                throw new InvalidOperationException("destination/gaze anchor backing must be visible and physics-backed: " + backing.name);
            var anchor = new GameObject(name);
            anchor.transform.SetPositionAndRotation(worldPosition, backing.rotation);
            anchor.transform.SetParent(backing, true);
            return anchor.transform;
        }

        private void RegisterContextDestination(string id, Transform transform)
        {
            if (string.IsNullOrWhiteSpace(id) || !transform)
                throw new InvalidOperationException("cannot register an empty GateContext destination transform");
            if (_context.Destinations.ContainsKey(id))
                throw new InvalidOperationException("duplicate GateContext destination transform: " + id);
            _context.Destinations.Add(id, transform);
        }

        private CompiledInstance PlaceCatalogInstance(CatalogMember member, Vector2 floorCoordinate, float yawDeg, bool reachable, string role, int index)
        {
            SceneInstanceAuthority authority = _context.ExpectedSceneInstances[index];
            if (authority.AssetId != member.AssetId)
                throw new InvalidOperationException("compiled instance ordering diverged before placement at index " + index);
            string assetPath = _furnitureAssetRoot + "/" + member.AssetId + ".obj";
            VerifyCatalogSource(assetPath, member.Sha256);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
            if (!prefab) throw new InvalidOperationException("missing frozen furniture asset: " + assetPath);
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            if (!instance) throw new InvalidOperationException("could not instantiate frozen furniture asset: " + assetPath);

            instance.name = authority.PersistentId;
            instance.transform.SetParent(_sceneRoot, false);
            instance.transform.localPosition = Vector3.zero;
            instance.transform.localRotation = Quaternion.identity;
            instance.transform.localScale = Vector3.one;
            ScaleToCatalogDimensions(instance, authority.AssetDimensionsM);
            instance.transform.localRotation = Quaternion.Euler(0f, yawDeg, 0f);

            Bounds current = BoundsOf(instance);
            Vector3 desiredFloorCenter = _sceneRoot.TransformPoint(new Vector3(floorCoordinate.x, 0f, -floorCoordinate.y));
            instance.transform.position += new Vector3(
                desiredFloorCenter.x - current.center.x,
                desiredFloorCenter.y - current.min.y,
                desiredFloorCenter.z - current.center.z);

            Collider[] colliders = AddCatalogColliders(instance, authority.CollisionSource);
            foreach (Collider collider in colliders)
                collider.sharedMaterial = MakePhysicsMaterial(
                    "furniture_" + index, authority.StaticFriction, authority.DynamicFriction);
            foreach (Renderer renderer in instance.GetComponentsInChildren<Renderer>(true))
                renderer.sharedMaterial = FurnitureMaterial(authority.SemanticClass);

            string persistentId = instance.name;
            SceneIdentity identity = instance.AddComponent<SceneIdentity>();
            identity.persistent_id = persistentId;
            identity.semantic_id = SemanticId(authority.SemanticClass);
            identity.instance_id = StableInstanceId(_context.Seed, persistentId);
            identity.semantic_class = authority.SemanticClass;
            identity.interactive = authority.Interactive;
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
                catalog_dimensions_m = authority.AssetDimensionsM,
                compiled_bounds_center_world_m = current.center,
                compiled_bounds_size_m = current.size,
                rotation_world_xyzw = instance.transform.rotation,
                semantic_class = authority.SemanticClass,
                semantic_id = identity.semantic_id,
                instance_id = identity.instance_id,
                material_variant = _context.SceneMaterialVariant,
                collision_source = authority.CollisionSource,
                collider_count = colliders.Length,
                interactive = authority.Interactive,
                mass_kg = authority.MassKg,
                static_friction = authority.StaticFriction,
                dynamic_friction = authority.DynamicFriction,
                reachable = reachable,
                role = role,
                visible_geometry_source = "verified_imported_kenney_mesh",
                physics_backed = true,
                provenance = "hash-verified Kenney CC0 catalog mesh with deterministic PhysX collider",
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

        private Vector3 SolveTargetPosition(Vector3 shoulderMidpoint, float supportTopY, float desiredReachM, float targetHeightM)
        {
            Vector3 target = shoulderMidpoint;
            Vector3 roomForward = _sceneRoot.TransformDirection(Vector3.back);
            float targetY = supportTopY + targetHeightM * 0.5f + SupportGapM;
            float vertical = targetY - shoulderMidpoint.y;
            float forwardSquared = desiredReachM * desiredReachM - vertical * vertical;
            if (forwardSquared <= 0.01f)
                throw new InvalidOperationException("catalog support height cannot satisfy the frozen bimanual shoulder-midpoint reach corridor");
            target += roomForward * Mathf.Sqrt(forwardSquared);
            target.y = targetY;
            return target;
        }

        private static TargetDefinition ResolveTarget(GateContext context)
        {
            TargetDefinition template = Targets.SingleOrDefault(candidate => candidate.PersistentId == context.TargetId);
            if (template == null) throw new InvalidOperationException("unknown frozen target: " + context.TargetId);
            if (!string.Equals(template.ContactStrategy, context.ContactStrategy, StringComparison.Ordinal))
                throw new InvalidOperationException("target/contact strategy mismatch: " + context.TargetId + "/" + context.ContactStrategy);
            string requiredDestination = template.PersistentId == "red_toy_001" ? "shallow_bin" :
                template.PersistentId == "blue_cup_001" ? "tray" : "marked_support";
            if (!string.Equals(requiredDestination, context.DestinationId, StringComparison.Ordinal))
                throw new InvalidOperationException("target/destination mismatch: " + context.TargetId + "/" + context.DestinationId);
            string requiredRoom = template.PersistentId == "red_toy_001" ? "warm_playroom" :
                template.PersistentId == "blue_cup_001" ? "sage_living_corner" : "birch_art_room";
            if (!string.Equals(requiredRoom, context.RoomFamily, StringComparison.Ordinal))
                throw new InvalidOperationException("target/room-family mismatch: " + context.TargetId + "/" + context.RoomFamily);
            float expectedMassScale = context.RobustnessVariant == "mass_friction_change" ? 1.25f : 1f;
            float expectedFrictionScale = context.RobustnessVariant == "mass_friction_change" ? .8f : 1f;
            if (context.RobustnessVariant != "nominal" && context.RobustnessVariant != "lateral_target_shift" &&
                context.RobustnessVariant != "mass_friction_change")
                throw new InvalidOperationException("target uses an unknown prospectively frozen robustness variant: " + context.RobustnessVariant);
            bool targetContractMatchesTemplate = context.TargetSemanticId == template.SemanticId &&
                context.TargetInstanceId == template.InstanceId && context.TargetGeometry == template.Geometry &&
                Approximately(context.TargetDimensionsM, template.DimensionsM) &&
                Mathf.Approximately(context.TargetMassKg, template.MassKg * expectedMassScale) &&
                Mathf.Approximately(context.TargetStaticFriction, template.StaticFriction * expectedFrictionScale) &&
                Mathf.Approximately(context.TargetDynamicFriction, template.DynamicFriction * expectedFrictionScale);
            if (!targetContractMatchesTemplate)
                throw new InvalidOperationException("authoritative compiled-contract target does not match the approved asset template: " + context.TargetId);
            return new TargetDefinition(
                context.TargetId, context.TargetSemanticId, context.TargetInstanceId, context.TargetGeometry,
                template.GeometrySpecSha256, context.TargetDimensionsM, context.TargetMassKg,
                context.TargetStaticFriction, context.TargetDynamicFriction, template.CollisionSource,
                context.ContactStrategy, template.Color);
        }

        private static bool Approximately(Vector3 left, Vector3 right)
        {
            return Mathf.Approximately(left.x, right.x) && Mathf.Approximately(left.y, right.y) &&
                Mathf.Approximately(left.z, right.z);
        }

        private CompiledInstance CreateActivityContext(CompiledInstance table, Vector3 targetPosition)
        {
            Vector3 right = _sceneRoot.TransformDirection(Vector3.right);
            Vector3 forward = _sceneRoot.TransformDirection(Vector3.back);
            float supportTop = table.Bounds.max.y;
            if (_context.TargetId == "red_toy_001")
            {
                CreateAuthoredProp(
                    "picture_book", SupportedCenter(targetPosition + right * 0.145f + forward * 0.015f, supportTop, 0.012f),
                    new Vector3(0.135f, 0.012f, 0.105f), new Color(0.18f, 0.55f, 0.82f),
                    "picture_book", "beside_picture_book", table.PersistentId, UnitBoxMesh(), false);
                return CreateDestination(
                    "shallow_bin", targetPosition - right * 0.255f + forward * 0.025f,
                    new Vector3(0.185f, 0.050f, 0.155f), new Color(0.92f, 0.62f, 0.18f),
                    0.045f, table.PersistentId, supportTop);
            }
            if (_context.TargetId == "blue_cup_001")
            {
                CreateAuthoredProp(
                    "reading_book", SupportedCenter(targetPosition + right * 0.150f + forward * 0.010f, supportTop, 0.014f),
                    new Vector3(0.145f, 0.014f, 0.110f), new Color(0.68f, 0.22f, 0.24f),
                    "book", "beside_book", table.PersistentId, UnitBoxMesh(), false);
                return CreateDestination(
                    "tray", targetPosition - right * 0.260f + forward * 0.030f,
                    new Vector3(0.205f, 0.026f, 0.165f), new Color(0.86f, 0.79f, 0.62f),
                    0.020f, table.PersistentId, supportTop);
            }

            Color[] craftColors =
            {
                new Color(0.88f, 0.20f, 0.28f), new Color(0.12f, 0.55f, 0.74f),
                new Color(0.32f, 0.68f, 0.30f), new Color(0.72f, 0.28f, 0.76f)
            };
            for (int index = 0; index < craftColors.Length; index++)
            {
                float side = index % 2 == 0 ? 1f : -1f;
                float lateral = side > 0f ? 0.125f + 0.025f * index : 0.100f;
                float depth = index < 2 ? 0.085f : -0.075f;
                CreateAuthoredProp(
                    "craft_stick_" + index.ToString("D2"),
                    SupportedCenter(targetPosition + right * side * lateral + forward * depth, supportTop, 0.010f),
                    new Vector3(0.010f, 0.010f, 0.105f), craftColors[index],
                    "craft_object", "among_craft_objects", table.PersistentId, UnitBoxMesh(), false);
            }
            return CreateDestination(
                "marked_support", targetPosition - right * 0.255f + forward * 0.020f,
                new Vector3(0.155f, 0.010f, 0.145f), new Color(0.18f, 0.68f, 0.48f),
                0f, table.PersistentId, supportTop);
        }

        private static Vector3 SupportedCenter(Vector3 position, float supportTop, float height)
        {
            position.y = supportTop + height * 0.5f + SupportGapM;
            return position;
        }

        private CompiledInstance CreateDestination(
            string destinationId,
            Vector3 desiredCenter,
            Vector3 dimensions,
            Color color,
            float wallHeightM,
            string supportId,
            float supportTop)
        {
            if (!string.Equals(destinationId, _context.DestinationId, StringComparison.Ordinal))
                throw new InvalidOperationException("activity destination did not match GateContext.DestinationId");
            desiredCenter.y = supportTop + dimensions.y * 0.5f + SupportGapM;
            bool openContainer = wallHeightM > 0f;
            Mesh mesh = openContainer ? OpenContainerMesh() : UnitBoxMesh();
            CompiledInstance destination = CreateAuthoredProp(
                destinationId, desiredCenter, dimensions, color, "activity_destination",
                "free_release_destination", supportId, mesh, openContainer);
            destination.game_object.name = destinationId;
            RegisterContextDestination(destinationId, destination.game_object.transform);
            return destination;
        }

        private CompiledInstance CreateAuthoredProp(
            string persistentSuffix,
            Vector3 centerWorld,
            Vector3 dimensions,
            Color color,
            string semanticClass,
            string role,
            string supportId,
            Mesh mesh,
            bool compoundContainer)
        {
            string persistentId = persistentSuffix == _context.DestinationId
                ? persistentSuffix
                : _context.RoomFamily + "_" + persistentSuffix;
            var instance = new GameObject(persistentId);
            instance.transform.SetParent(_sceneRoot, true);
            instance.transform.SetPositionAndRotation(centerWorld, _sceneRoot.rotation);
            instance.transform.localScale = dimensions;
            instance.AddComponent<MeshFilter>().sharedMesh = mesh;
            instance.AddComponent<MeshRenderer>().sharedMaterial = MaterialFor(
                _context.SceneMaterialVariant + "_" + semanticClass + "_" + persistentSuffix, color, 0.16f);

            var colliders = new List<Collider>();
            if (compoundContainer)
            {
                const float wall = 0.08f;
                const float baseHeight = 0.16f;
                colliders.Add(AddBoxCollider(instance, new Vector3(0f, -0.5f + baseHeight * 0.5f, 0f), new Vector3(1f, baseHeight, 1f)));
                colliders.Add(AddBoxCollider(instance, new Vector3(-0.5f + wall * 0.5f, 0f, 0f), new Vector3(wall, 1f, 1f)));
                colliders.Add(AddBoxCollider(instance, new Vector3(0.5f - wall * 0.5f, 0f, 0f), new Vector3(wall, 1f, 1f)));
                colliders.Add(AddBoxCollider(instance, new Vector3(0f, 0f, -0.5f + wall * 0.5f), new Vector3(1f, 1f, wall)));
                colliders.Add(AddBoxCollider(instance, new Vector3(0f, 0f, 0.5f - wall * 0.5f), new Vector3(1f, 1f, wall)));
            }
            else colliders.Add(AddBoxCollider(instance, Vector3.zero, Vector3.one));
            foreach (Collider collider in colliders)
                collider.sharedMaterial = MakePhysicsMaterial("reachable_context", FurnitureStaticFriction, FurnitureDynamicFriction);

            SceneIdentity identity = instance.AddComponent<SceneIdentity>();
            identity.persistent_id = persistentId;
            identity.semantic_id = SemanticId(semanticClass);
            identity.instance_id = StableInstanceId(_context.Seed, persistentId);
            identity.semantic_class = semanticClass;
            identity.interactive = false;
            identity.physics_role = "reachable_static_physx_collider";
            identity.support_id = supportId;
            identity.license = "repository-authored";
            identity.source_sha256 = Sha256Hex("authored-mesh:" + semanticClass + ":" + dimensions.ToString("R"));

            Bounds bounds = BoundsOf(instance);
            var compiled = new CompiledInstance
            {
                persistent_id = persistentId,
                asset_id = "repository_authored_" + semanticClass,
                asset_path = "ProceduralSceneCompiler.cs",
                source_sha256 = identity.source_sha256,
                license = "repository-authored",
                catalog_dimensions_m = dimensions,
                compiled_bounds_center_world_m = bounds.center,
                compiled_bounds_size_m = bounds.size,
                rotation_world_xyzw = instance.transform.rotation,
                semantic_class = semanticClass,
                semantic_id = identity.semantic_id,
                instance_id = identity.instance_id,
                material_variant = _context.SceneMaterialVariant,
                collision_source = compoundContainer ? "compound_analytic_static" : "analytic_box_static",
                collider_count = colliders.Count,
                interactive = false,
                reachable = true,
                role = role,
                visible_geometry_source = "repository_authored_context_mesh",
                physics_backed = true,
                provenance = "deterministic SceneCompiler authored mesh and PhysX collider",
                game_object = instance,
                Bounds = bounds
            };
            _instances.Add(compiled);
            return compiled;
        }

        private static BoxCollider AddBoxCollider(GameObject instance, Vector3 center, Vector3 size)
        {
            BoxCollider collider = instance.AddComponent<BoxCollider>();
            collider.center = center;
            collider.size = size;
            return collider;
        }

        private void RejectFurnitureInInteractionCorridor(CompiledInstance table, Vector3 target, Vector3 destination)
        {
            Bounds corridor = new Bounds(0.5f * (target + destination), Vector3.zero);
            corridor.Encapsulate(target);
            corridor.Encapsulate(destination);
            corridor.Expand(new Vector3(0.30f, 0.42f, 0.30f));
            foreach (CompiledInstance instance in _instances)
            {
                if (instance == table || instance.reachable || instance.semantic_class == "rug") continue;
                if (corridor.Intersects(instance.Bounds))
                    throw new InvalidOperationException("prospective swept interaction corridor blocked by " + instance.persistent_id);
            }
        }

        private void RejectOverlappingFurnishings()
        {
            CompiledInstance[] furniture = _instances
                .Where(instance => instance.license == CatalogLicense && instance.semantic_class != "rug")
                .ToArray();
            for (int first = 0; first < furniture.Length; first++)
            {
                for (int second = first + 1; second < furniture.Length; second++)
                {
                    if (furniture[first].Bounds.Intersects(furniture[second].Bounds))
                        throw new InvalidOperationException(
                            "prospective furnishing overlap: " + furniture[first].persistent_id + "/" + furniture[second].persistent_id);
                }
            }
            CompiledInstance[] activityContext = _instances
                .Where(instance => instance.visible_geometry_source == "repository_authored_context_mesh")
                .ToArray();
            for (int first = 0; first < activityContext.Length; first++)
            {
                for (int second = first + 1; second < activityContext.Length; second++)
                {
                    if (activityContext[first].Bounds.Intersects(activityContext[second].Bounds))
                        throw new InvalidOperationException(
                            "prospective activity-context overlap: " + activityContext[first].persistent_id + "/" + activityContext[second].persistent_id);
                }
            }
        }

        private void RejectObjectsOutsideFinishedRoom()
        {
            float halfWidth = _context.SceneEnvelopeM.x * 0.5f - 0.045f;
            float backInterior = -(_context.SceneEnvelopeM.z - 0.55f) + 0.045f;
            float frontInterior = 0.55f - 0.045f;
            foreach (CompiledInstance instance in _instances)
            {
                foreach (Vector3 cornerWorld in BoundsCorners(instance.Bounds))
                {
                    Vector3 corner = _sceneRoot.InverseTransformPoint(cornerWorld);
                    if (corner.x < -halfWidth || corner.x > halfWidth ||
                        corner.z < backInterior || corner.z > frontInterior ||
                        corner.y < -0.002f || corner.y > _context.SceneEnvelopeM.y)
                        throw new InvalidOperationException("prospective finished-room envelope violation: " + instance.persistent_id);
                }
            }
        }

        private void CreateFreeTarget(Vector3 initialPosition, string supportId)
        {
            var target = new GameObject(_target.PersistentId);
            target.transform.SetPositionAndRotation(
                initialPosition,
                _sceneRoot.rotation * Quaternion.Euler(0f, NextSigned(18f), 0f));
            target.transform.localScale = _target.DimensionsM;

            Mesh targetMesh = _target.Geometry == "rounded_toy" ? RoundedGraspToyMesh() :
                _target.Geometry == "handled_cup" ? HandledCupMesh() : BeveledBlockMesh();
            target.AddComponent<MeshFilter>().sharedMesh = targetMesh;
            target.AddComponent<MeshRenderer>().sharedMaterial = MaterialFor("interactive_target_" + _target.PersistentId, _target.Color, 0.24f);
            Collider[] targetColliders = AddTargetColliders(target, _target);
            foreach (Collider targetCollider in targetColliders)
                targetCollider.sharedMaterial = MakePhysicsMaterial("interactive_target", _target.StaticFriction, _target.DynamicFriction);

            SceneIdentity identity = target.AddComponent<SceneIdentity>();
            identity.persistent_id = _target.PersistentId;
            identity.semantic_id = _target.SemanticId;
            identity.instance_id = _target.InstanceId;
            identity.semantic_class = "interactive_target";
            identity.interactive = true;
            identity.physics_role = "free_non_kinematic_physx_rigidbody";
            identity.support_id = supportId;
            identity.license = "repository-authored";
            identity.source_sha256 = _target.GeometrySpecSha256;

            PhysicsTruthObjectIdentity truthIdentity = target.AddComponent<PhysicsTruthObjectIdentity>();
            truthIdentity.persistent_id = _target.PersistentId;
            truthIdentity.semantic_id = _target.SemanticId.ToString();
            truthIdentity.instance_id = _target.InstanceId.ToString();

            Rigidbody body = target.AddComponent<Rigidbody>();
            body.mass = _target.MassKg;
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
            Transform destinationAnchor = RequireContextDestination(_context.DestinationId);
            Transform finalGazeAnchor = RequireContextDestination(_context.FinalGazeZone);
            Vector3 finalGazePoint = finalGazeAnchor.position;
            return new SceneSpecReceipt
            {
                schema = SceneSchema,
                authoritative_contract_sha256 = _context.CompiledContractSha256,
                seed = _context.Seed,
                cell_id = _context.CellId,
                room_family = _context.RoomFamily,
                material_variant = _context.SceneMaterialVariant,
                target_id = _context.TargetId,
                destination_id = _context.DestinationId,
                contact_strategy = _context.ContactStrategy,
                final_gaze_zone = _context.FinalGazeZone,
                destination_transform_name = destinationAnchor.name,
                final_gaze_transform_name = finalGazeAnchor.name,
                expected_scene_asset_ids = _context.ExpectedSceneAssetIds.ToArray(),
                minimum_contextual_objects = _context.MinimumContextualObjects,
                envelope_m = _context.SceneEnvelopeM,
                zones = zones,
                instances = _instances.Select(instance => instance.WithoutRuntimeReferences()).ToArray(),
                target = new TargetReceipt
                {
                    persistent_id = _target.PersistentId,
                    semantic_id = _target.SemanticId,
                    instance_id = _target.InstanceId,
                    geometry = _target.Geometry,
                    geometry_spec_sha256 = _target.GeometrySpecSha256,
                    license = "repository-authored",
                    dimensions_m = _target.DimensionsM,
                    mass_kg = _target.MassKg,
                    static_friction = _target.StaticFriction,
                    dynamic_friction = _target.DynamicFriction,
                    collision_source = _target.CollisionSource,
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
                        child_id = _context.ExpectedSupportRelations[0].ChildId,
                        support_id = _context.ExpectedSupportRelations[0].SupportId,
                        relation = "initial_physical_support",
                        initial_separation_m = TargetBounds().min.y - table.Bounds.max.y
                    },
                    new SupportRelation
                    {
                        child_id = _context.ExpectedSupportRelations[1].ChildId,
                        destination_id = _context.ExpectedSupportRelations[1].DestinationId,
                        support_id = table.PersistentId,
                        relation = "intended_free_release_destination",
                        initial_separation_m = Vector3.Distance(targetPosition, destinationAnchor.position)
                    },
                    new SupportRelation
                    {
                        child_id = _context.DestinationId,
                        destination_id = _context.DestinationId,
                        support_id = table.PersistentId,
                        relation = "destination_physical_support",
                        initial_separation_m = _destination.Bounds.min.y - table.Bounds.max.y
                    }
                },
                reachability = new ReachabilityReceipt
                {
                    anchor_id = rightShoulder.name,
                    anchor_provenance = "engine_observed_weighted_avatar_transform_at_scene_initialization",
                    target_from_right_shoulder_m = Vector3.Distance(rightShoulder.position, targetPosition),
                    target_from_left_shoulder_m = Vector3.Distance(leftShoulder.position, targetPosition),
                    compiled_requested_m = desiredReachM,
                    compiled_midpoint_band_m = new Vector2(_context.TargetReachBandMinM, _context.TargetReachBandMaxM),
                    lateral_bias_toward_right_shoulder_m = _context.TargetLateralBiasM,
                    aperture_aware = true,
                    seed_specific_retuning = false,
                    compiled_limit_m = new Vector2(_context.TargetReachBandMinM, _context.TargetReachBandMaxM),
                    corridor_radius_m = 0.060f
                },
                sightlines = new SightlineReceipt
                {
                    origin_id = _context.Head.name,
                    origin_world_m = sightlineOrigin,
                    target_world_m = targetPosition,
                    target_visible_from_head_geometry = HasClearSightline(sightlineOrigin, _context.TargetBody.gameObject),
                    final_gaze_zone = _context.FinalGazeZone,
                    final_gaze_world_m = finalGazePoint,
                    final_gaze_zone_compiled = _context.Destinations.ContainsKey(_context.FinalGazeZone),
                    final_gaze_visible_from_head_geometry = HasClearSightlineToAnchor(sightlineOrigin, finalGazeAnchor),
                    destination_world_m = destinationAnchor.position,
                    destination_visible_from_head_geometry = HasClearSightlineToObject(sightlineOrigin, _destination.game_object),
                    prospective_view_half_angle_deg = ProspectiveViewHalfAngleDeg
                },
                stabilization_s = _context.SceneStabilizationSeconds,
                no_visible_primitive_furniture = _context.RequireNoVisiblePrimitiveFurniture,
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
            bool importedHeroFurniture = _instances
                .Where(instance => instance.license == CatalogLicense)
                .All(instance => instance.visible_geometry_source == "verified_imported_kenney_mesh");
            bool hashesVerified = _instances
                .Where(instance => instance.license == CatalogLicense)
                .All(instance => instance.source_sha256 == FindCatalog(instance.asset_id).Sha256);
            string[] actualCatalogAssetIds = _instances.Where(instance => instance.license == CatalogLicense)
                .Select(instance => instance.asset_id).ToArray();
            int contextualVisibleObjectCount = actualCatalogAssetIds.Length;
            bool contextualObjectCountPass = contextualVisibleObjectCount >= _context.MinimumContextualObjects;
            string[] actualZoneIds = _sceneRoot.GetComponentsInChildren<SceneZone>(true)
                .Select(zone => zone.zone_id).ToArray();
            bool compiledInstanceAuthorityPass = _context.ExpectedSceneInstances.Length == actualCatalogAssetIds.Length &&
                _context.ExpectedSceneInstances.Select((authority, index) => new { authority, index }).All(row =>
                    row.authority.PersistentId == _instances[row.index].persistent_id &&
                    row.authority.AssetId == _instances[row.index].asset_id &&
                    Approximately(row.authority.AssetDimensionsM, _instances[row.index].catalog_dimensions_m) &&
                    row.authority.SemanticClass == _instances[row.index].semantic_class &&
                    row.authority.CollisionSource == _instances[row.index].collision_source &&
                    row.authority.Interactive == _instances[row.index].interactive &&
                    Mathf.Abs(row.authority.MassKg - _instances[row.index].mass_kg) <= 1e-6f &&
                    Mathf.Abs(row.authority.StaticFriction - _instances[row.index].static_friction) <= 1e-6f &&
                    Mathf.Abs(row.authority.DynamicFriction - _instances[row.index].dynamic_friction) <= 1e-6f);
            bool compiledRelationAuthorityPass = spec.support_relations.Length >= 2 &&
                spec.support_relations[0].child_id == _context.ExpectedSupportRelations[0].ChildId &&
                spec.support_relations[0].support_id == _context.ExpectedSupportRelations[0].SupportId &&
                spec.support_relations[1].child_id == _context.ExpectedSupportRelations[1].ChildId &&
                spec.support_relations[1].destination_id == _context.ExpectedSupportRelations[1].DestinationId;
            bool compiledSceneContractPass = SameMultiset(actualCatalogAssetIds, _context.ExpectedSceneAssetIds) &&
                SameMultiset(actualZoneIds, _context.SceneZoneIds) &&
                _instances.All(instance => instance.material_variant == _context.SceneMaterialVariant) &&
                Approximately(spec.envelope_m, _context.SceneEnvelopeM) &&
                spec.minimum_contextual_objects == _context.MinimumContextualObjects &&
                compiledInstanceAuthorityPass && compiledRelationAuthorityPass &&
                spec.sightlines.final_gaze_zone == _context.ExpectedFinalGazeZone &&
                _context.ExpectedTargetVisibleAtRequiredEvents &&
                Mathf.Abs(spec.stabilization_s - _context.SceneStabilizationSeconds) <= 1e-6f &&
                spec.no_visible_primitive_furniture == _context.RequireNoVisiblePrimitiveFurniture &&
                spec.authoritative_contract_sha256 == _context.CompiledContractSha256;
            bool persistentIdentitiesUnique = _instances.Select(instance => instance.persistent_id).Distinct().Count() == _instances.Count;
            bool allVisibleContextPhysicsBacked = _instances.All(instance => instance.physics_backed);
            float rightReach = Vector3.Distance(rightShoulder.position, _context.TargetBody.position);
            float leftReach = Vector3.Distance(leftShoulder.position, _context.TargetBody.position);
            float midpointReach = Vector3.Distance(
                0.5f * (rightShoulder.position + leftShoulder.position),
                _context.TargetBody.position);
            float supportSeparation = TargetBounds().min.y - table.Bounds.max.y;
            bool reachPass = midpointReach >= _context.TargetReachBandMinM &&
                midpointReach <= _context.TargetReachBandMaxM;
            bool supportPass = supportSeparation >= 0f && supportSeparation <= 0.002f;
            bool sightlinePass = spec.sightlines.target_visible_from_head_geometry &&
                spec.sightlines.destination_visible_from_head_geometry;
            bool cameraConePass = IsInProspectiveTaskView(_context.Head, spec.sightlines.target_world_m) &&
                IsInProspectiveTaskView(_context.Head, spec.sightlines.destination_world_m);
            bool finalGazePass = spec.sightlines.final_gaze_zone_compiled &&
                spec.sightlines.final_gaze_visible_from_head_geometry;
            bool authoritativeMappingsPass = RequireContextDestination(_context.DestinationId) == _destination.game_object.transform &&
                RequireContextDestination(_context.DestinationId).GetComponentsInChildren<Collider>(true).Any(collider => collider.enabled) &&
                _context.Destinations.ContainsKey(_context.FinalGazeZone) &&
                _context.Destinations.All(mapping => IsBackedByVisiblePhysics(mapping.Value));
            bool assistanceClean = _context.AssistanceLedger.Count == 0;
            bool passed = targetUnparented && targetFree && noJoints && targetColliderValid && reachableColliders &&
                          importedHeroFurniture && hashesVerified && contextualObjectCountPass && persistentIdentitiesUnique &&
                          compiledSceneContractPass && allVisibleContextPhysicsBacked && reachPass && supportPass && sightlinePass && cameraConePass &&
                          finalGazePass && authoritativeMappingsPass && assistanceClean;
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
                        "contextual_count=" + contextualObjectCountPass,
                        "compiled_scene_contract=" + compiledSceneContractPass,
                        "identity_unique=" + persistentIdentitiesUnique,
                        "physics_backed=" + allVisibleContextPhysicsBacked,
                        "reach=" + reachPass,
                        "support=" + supportPass,
                        "sightline=" + sightlinePass,
                        "camera_cone=" + cameraConePass,
                        "final_gaze=" + finalGazePass,
                        "authoritative_mappings=" + authoritativeMappingsPass,
                        "assistance=" + assistanceClean
                    }));
            }
            return new SceneValidationReceipt
            {
                schema = ReceiptSchema,
                episode_id = _context.EpisodeId,
                seed = _context.Seed,
                room_family = _context.RoomFamily,
                material_variant = _context.SceneMaterialVariant,
                status = "COMPILED_CONTRACT_VALIDATED_ONLY",
                visual_pass_claimed = false,
                physical_episode_pass_claimed = false,
                source_hashes_verified = hashesVerified,
                imported_visible_furniture_only = importedHeroFurniture,
                reachable_elements_have_physx_colliders = reachableColliders,
                distant_decor_explicit_noninteractive = _instances.Where(instance => !instance.reachable).All(instance => !instance.interactive),
                contextual_visible_object_count = contextualVisibleObjectCount,
                contextual_visible_object_minimum = _context.MinimumContextualObjects,
                compiled_scene_contract_authority_validated = compiledSceneContractPass,
                persistent_identities_unique = persistentIdentitiesUnique,
                all_visible_context_physics_backed = allVisibleContextPhysicsBacked,
                camera_aware_target_and_destination_sightlines = sightlinePass && cameraConePass,
                final_gaze_zone_compiled = finalGazePass,
                authoritative_destination_and_gaze_mappings = authoritativeMappingsPass,
                deterministic_prospective_rejection_only = true,
                target_unparented = targetUnparented,
                target_non_kinematic = targetFree,
                target_joint_count = target.GetComponentsInChildren<Joint>(true).Length,
                target_post_initialization_transform_writes = 0,
                target_external_force_or_spring_commands = 0,
                target_support_initial_separation_m = supportSeparation,
                measured_right_shoulder_target_distance_m = rightReach,
                measured_left_shoulder_target_distance_m = leftReach,
                measured_shoulder_midpoint_target_distance_m = midpointReach,
                head_target_sightline_clear = sightlinePass,
                assistance_ledger_entries_at_compile = _context.AssistanceLedger.Count,
                no_seed_specific_controller_or_scene_code = true,
                deterministic_selection_inputs = "authoritative compiled GateContext contract + GateContext.Seed + frozen catalog/profile data",
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
            return HasClearSightlineToObject(origin, target);
        }

        private bool HasClearSightlineToObject(Vector3 origin, GameObject target)
        {
            Collider targetCollider = target.GetComponents<Collider>().First(collider => collider.enabled);
            Vector3 destination = targetCollider.bounds.center;
            Vector3 direction = destination - origin;
            float targetExtent = targetCollider.bounds.extents.magnitude;
            RaycastHit[] hits = Physics.RaycastAll(origin, direction.normalized, direction.magnitude + targetExtent, ~0, QueryTriggerInteraction.Ignore);
            foreach (RaycastHit hit in hits.OrderBy(hit => hit.distance))
            {
                if (_context.AvatarColliders.Contains(hit.collider) ||
                    (_context.AvatarRoot && (hit.transform == _context.AvatarRoot.transform || hit.transform.IsChildOf(_context.AvatarRoot.transform))))
                    continue;
                return hit.transform == target.transform || hit.transform.IsChildOf(target.transform);
            }
            return false;
        }

        private bool HasClearSightlineToAnchor(Vector3 origin, Transform anchor)
        {
            Transform backing = VisiblePhysicsBacking(anchor);
            if (!backing) return false;
            Vector3 direction = anchor.position - origin;
            RaycastHit[] hits = Physics.RaycastAll(
                origin, direction.normalized, direction.magnitude + 0.01f, ~0, QueryTriggerInteraction.Ignore);
            foreach (RaycastHit hit in hits.OrderBy(hit => hit.distance))
            {
                if (_context.AvatarColliders.Contains(hit.collider) ||
                    (_context.AvatarRoot && (hit.transform == _context.AvatarRoot.transform || hit.transform.IsChildOf(_context.AvatarRoot.transform))))
                    continue;
                return hit.transform == backing || hit.transform.IsChildOf(backing);
            }
            return false;
        }

        private Transform RequireContextDestination(string id)
        {
            if (!_context.Destinations.TryGetValue(id, out Transform transform) || !transform)
                throw new InvalidOperationException("GateContext has no compiled transform for contract label: " + id);
            return transform;
        }

        private static bool IsBackedByVisiblePhysics(Transform anchor)
        {
            return VisiblePhysicsBacking(anchor) != null;
        }

        private static Transform VisiblePhysicsBacking(Transform anchor)
        {
            for (Transform cursor = anchor; cursor; cursor = cursor.parent)
            {
                bool visible = cursor.GetComponentsInChildren<Renderer>(true).Any(renderer => renderer.enabled);
                bool physical = cursor.GetComponentsInChildren<Collider>(true).Any(collider => collider.enabled);
                if (visible && physical) return cursor;
            }
            return null;
        }

        private static bool IsInProspectiveTaskView(Transform head, Vector3 point)
        {
            Vector3 toPoint = (point - head.position).normalized;
            Vector3 plannedTaskForward = (head.rotation * Quaternion.Euler(44f, 0f, 0f)) * Vector3.forward;
            return Vector3.Angle(plannedTaskForward, toPoint) <= ProspectiveViewHalfAngleDeg;
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
            switch (_context.SceneMaterialVariant)
            {
                case "painted_sage": baseColor = new Color(0.31f, 0.56f, 0.42f); break;
                case "natural_birch": baseColor = new Color(0.76f, 0.64f, 0.44f); break;
                default: baseColor = new Color(0.55f, 0.34f, 0.18f); break;
            }
            if (semanticClass == "rug") baseColor = Color.Lerp(_profile.AccentColor, Color.white, 0.12f);
            else if (semanticClass == "plant") baseColor = new Color(0.20f, 0.48f, 0.25f);
            else if (semanticClass == "lamp") baseColor = Color.Lerp(_profile.AccentColor, Color.white, 0.35f);
            else if (semanticClass == "books") baseColor = new Color(0.74f, 0.29f, 0.17f);
            return MaterialFor(_context.SceneMaterialVariant + "_" + semanticClass, baseColor, semanticClass == "rug" ? 0.05f : 0.20f);
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

        private static Collider[] AddTargetColliders(GameObject target, TargetDefinition definition)
        {
            if (definition.CollisionSource == "analytic_sphere")
            {
                SphereCollider sphere = target.AddComponent<SphereCollider>();
                sphere.radius = 0.5f;
                return new Collider[] { sphere };
            }
            if (definition.CollisionSource == "compound_analytic")
            {
                CapsuleCollider body = target.AddComponent<CapsuleCollider>();
                body.direction = 1;
                body.center = new Vector3(-0.08f, -0.03f, 0f);
                body.radius = 0.33f;
                body.height = 0.94f;
                BoxCollider handle = target.AddComponent<BoxCollider>();
                handle.center = new Vector3(0.41f, 0.03f, 0f);
                handle.size = new Vector3(0.18f, 0.54f, 0.20f);
                return new Collider[] { body, handle };
            }
            BoxCollider box = target.AddComponent<BoxCollider>();
            box.size = new Vector3(0.96f, 0.96f, 0.96f);
            return new Collider[] { box };
        }

        private static Mesh OpenContainerMesh()
        {
            var vertices = new List<Vector3>();
            var triangles = new List<int>();
            const float wall = 0.08f;
            const float baseHeight = 0.16f;
            float top = 0.5f;
            float wallCenterY = 0.5f * (-0.5f + top);
            float wallSizeY = top + 0.5f;
            AppendBox(vertices, triangles, new Vector3(0f, -0.5f + baseHeight * 0.5f, 0f), new Vector3(1f, baseHeight, 1f));
            AppendBox(vertices, triangles, new Vector3(-0.5f + wall * 0.5f, wallCenterY, 0f), new Vector3(wall, wallSizeY, 1f));
            AppendBox(vertices, triangles, new Vector3(0.5f - wall * 0.5f, wallCenterY, 0f), new Vector3(wall, wallSizeY, 1f));
            AppendBox(vertices, triangles, new Vector3(0f, wallCenterY, -0.5f + wall * 0.5f), new Vector3(1f, wallSizeY, wall));
            AppendBox(vertices, triangles, new Vector3(0f, wallCenterY, 0.5f - wall * 0.5f), new Vector3(1f, wallSizeY, wall));
            var mesh = new Mesh { name = "RepositoryAuthoredOpenActivityDestination" };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static Mesh HandledCupMesh()
        {
            const int segments = 32;
            const float radius = 0.34f;
            var vertices = new List<Vector3>();
            var triangles = new List<int>();
            for (int segment = 0; segment < segments; segment++)
            {
                float a0 = 2f * Mathf.PI * segment / segments;
                float a1 = 2f * Mathf.PI * (segment + 1) / segments;
                Vector3 bottom0 = new Vector3(Mathf.Cos(a0) * radius - 0.08f, -0.5f, Mathf.Sin(a0) * radius);
                Vector3 bottom1 = new Vector3(Mathf.Cos(a1) * radius - 0.08f, -0.5f, Mathf.Sin(a1) * radius);
                Vector3 top1 = new Vector3(Mathf.Cos(a1) * radius - 0.08f, 0.5f, Mathf.Sin(a1) * radius);
                Vector3 top0 = new Vector3(Mathf.Cos(a0) * radius - 0.08f, 0.5f, Mathf.Sin(a0) * radius);
                AppendQuad(vertices, triangles, bottom0, bottom1, top1, top0);
                AppendTriangle(vertices, triangles, new Vector3(-0.08f, -0.5f, 0f), bottom1, bottom0);
            }
            AppendBox(vertices, triangles, new Vector3(0.39f, 0.35f, 0f), new Vector3(0.22f, 0.16f, 0.16f));
            AppendBox(vertices, triangles, new Vector3(0.46f, 0.02f, 0f), new Vector3(0.08f, 0.58f, 0.16f));
            AppendBox(vertices, triangles, new Vector3(0.39f, -0.31f, 0f), new Vector3(0.22f, 0.16f, 0.16f));
            var mesh = new Mesh { name = "RepositoryAuthoredHandledCup" };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static Mesh BeveledBlockMesh()
        {
            const int sides = 8;
            var vertices = new List<Vector3>();
            var triangles = new List<int>();
            for (int side = 0; side < sides; side++)
            {
                float a0 = Mathf.PI * 0.25f * side + Mathf.PI * 0.125f;
                float a1 = Mathf.PI * 0.25f * (side + 1) + Mathf.PI * 0.125f;
                Vector3 lower0 = new Vector3(Mathf.Cos(a0) * 0.52f, -0.5f, Mathf.Sin(a0) * 0.52f);
                Vector3 lower1 = new Vector3(Mathf.Cos(a1) * 0.52f, -0.5f, Mathf.Sin(a1) * 0.52f);
                Vector3 upper1 = new Vector3(lower1.x, 0.5f, lower1.z);
                Vector3 upper0 = new Vector3(lower0.x, 0.5f, lower0.z);
                AppendQuad(vertices, triangles, lower0, lower1, upper1, upper0);
                AppendTriangle(vertices, triangles, Vector3.down * 0.5f, lower1, lower0);
                AppendTriangle(vertices, triangles, Vector3.up * 0.5f, upper0, upper1);
            }
            var mesh = new Mesh { name = "RepositoryAuthoredBeveledBlock" };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static void AppendTriangle(List<Vector3> vertices, List<int> triangles, Vector3 a, Vector3 b, Vector3 c)
        {
            int start = vertices.Count;
            vertices.Add(a); vertices.Add(b); vertices.Add(c);
            triangles.Add(start); triangles.Add(start + 1); triangles.Add(start + 2);
        }

        private static void AppendQuad(List<Vector3> vertices, List<int> triangles, Vector3 a, Vector3 b, Vector3 c, Vector3 d)
        {
            int start = vertices.Count;
            vertices.Add(a); vertices.Add(b); vertices.Add(c); vertices.Add(d);
            triangles.Add(start); triangles.Add(start + 1); triangles.Add(start + 2);
            triangles.Add(start); triangles.Add(start + 2); triangles.Add(start + 3);
        }

        private static void AppendBox(List<Vector3> vertices, List<int> triangles, Vector3 center, Vector3 size)
        {
            Vector3 e = size * 0.5f;
            Vector3 p000 = center + new Vector3(-e.x, -e.y, -e.z);
            Vector3 p001 = center + new Vector3(-e.x, -e.y, e.z);
            Vector3 p010 = center + new Vector3(-e.x, e.y, -e.z);
            Vector3 p011 = center + new Vector3(-e.x, e.y, e.z);
            Vector3 p100 = center + new Vector3(e.x, -e.y, -e.z);
            Vector3 p101 = center + new Vector3(e.x, -e.y, e.z);
            Vector3 p110 = center + new Vector3(e.x, e.y, -e.z);
            Vector3 p111 = center + new Vector3(e.x, e.y, e.z);
            AppendQuad(vertices, triangles, p000, p100, p110, p010);
            AppendQuad(vertices, triangles, p101, p001, p011, p111);
            AppendQuad(vertices, triangles, p001, p000, p010, p011);
            AppendQuad(vertices, triangles, p100, p101, p111, p110);
            AppendQuad(vertices, triangles, p010, p110, p111, p011);
            AppendQuad(vertices, triangles, p001, p101, p100, p000);
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
                case "picture_book": return 31;
                case "book": return 32;
                case "craft_object": return 33;
                case "activity_destination": return 34;
                default: return 10;
            }
        }

        private static string Sha256Hex(string value)
        {
            using (SHA256 hash = SHA256.Create())
            {
                byte[] digest = hash.ComputeHash(System.Text.Encoding.UTF8.GetBytes(value));
                return BitConverter.ToString(digest).Replace("-", string.Empty).ToLowerInvariant();
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

        private sealed class TargetDefinition
        {
            public readonly string PersistentId;
            public readonly int SemanticId;
            public readonly int InstanceId;
            public readonly string Geometry;
            public readonly string GeometrySpecSha256;
            public readonly Vector3 DimensionsM;
            public readonly float MassKg;
            public readonly float StaticFriction;
            public readonly float DynamicFriction;
            public readonly string CollisionSource;
            public readonly string ContactStrategy;
            public readonly Color Color;

            public TargetDefinition(
                string persistentId, int semanticId, int instanceId, string geometry,
                string geometrySpecSha256, Vector3 dimensionsM, float massKg,
                float staticFriction, float dynamicFriction, string collisionSource,
                string contactStrategy, Color color)
            {
                PersistentId = persistentId;
                SemanticId = semanticId;
                InstanceId = instanceId;
                Geometry = geometry;
                GeometrySpecSha256 = geometrySpecSha256;
                DimensionsM = dimensionsM;
                MassKg = massKg;
                StaticFriction = staticFriction;
                DynamicFriction = dynamicFriction;
                CollisionSource = collisionSource;
                ContactStrategy = contactStrategy;
                Color = color;
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
        public string authoritative_contract_sha256;
        public int seed;
        public string cell_id;
        public string room_family;
        public string material_variant;
        public string target_id;
        public string destination_id;
        public string contact_strategy;
        public string final_gaze_zone;
        public string destination_transform_name;
        public string final_gaze_transform_name;
        public string[] expected_scene_asset_ids;
        public int minimum_contextual_objects;
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
        public float mass_kg;
        public float static_friction;
        public float dynamic_friction;
        public bool reachable;
        public string role;
        public string visible_geometry_source;
        public bool physics_backed;
        public string provenance;
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
                mass_kg = mass_kg,
                static_friction = static_friction,
                dynamic_friction = dynamic_friction,
                reachable = reachable,
                role = role,
                visible_geometry_source = visible_geometry_source,
                physics_backed = physics_backed,
                provenance = provenance
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
        public string destination_id;
        public string relation;
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
        public Vector3 final_gaze_world_m;
        public bool final_gaze_zone_compiled;
        public bool final_gaze_visible_from_head_geometry;
        public Vector3 destination_world_m;
        public bool destination_visible_from_head_geometry;
        public float prospective_view_half_angle_deg;
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
        public int contextual_visible_object_count;
        public int contextual_visible_object_minimum;
        public bool compiled_scene_contract_authority_validated;
        public bool persistent_identities_unique;
        public bool all_visible_context_physics_backed;
        public bool camera_aware_target_and_destination_sightlines;
        public bool final_gaze_zone_compiled;
        public bool authoritative_destination_and_gaze_mappings;
        public bool deterministic_prospective_rejection_only;
        public bool target_unparented;
        public bool target_non_kinematic;
        public int target_joint_count;
        public int target_post_initialization_transform_writes;
        public int target_external_force_or_spring_commands;
        public float target_support_initial_separation_m;
        public float measured_right_shoulder_target_distance_m;
        public float measured_left_shoulder_target_distance_m;
        public float measured_shoulder_midpoint_target_distance_m;
        public bool head_target_sightline_clear;
        public int assistance_ledger_entries_at_compile;
        public bool no_seed_specific_controller_or_scene_code;
        public string deterministic_selection_inputs;
        public string disclosure;
    }
}
#endif
