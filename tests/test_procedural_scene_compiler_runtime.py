import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT
    / "babyworld_lite"
    / "childlens_engine_bakeoff"
    / "procedural_scene_gate"
    / "ProceduralSceneCompiler.cs"
)
CONFIG_PATH = ROOT / "configs" / "embodied_simulation_procedural_scene_gate.json"


def _source() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _method(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


def test_one_deterministic_compiler_covers_the_three_frozen_cells_without_bespoke_seeds():
    source = _source()
    config = _config()
    assert "public sealed class ProceduralSceneCompiler : ISceneCompilerModule" in source
    assert "public void Build(GateContext context, string furnitureAssetRoot)" in source
    assert source.count("new RoomProfile(") == 3
    assert "new System.Random(context.Seed)" in source
    assert "float desiredReachM = context.TargetMidpointReachM" in source
    assert "seed_specific_retuning = false" in source
    for cell in config["episode_cells"]:
        assert f'"{cell["room_family"]}"' in source
        assert f'"{cell["material_variant"]}"' in source
        assert f'"{cell["target_id"]}"' in source
        assert f'"{cell["destination_id"]}"' in source
        assert f'"{cell["contact_strategy"]}"' in source
        assert str(cell["seed"]) not in source


def test_each_materially_distinct_profile_has_ten_contextual_catalog_placements_plus_support():
    source = _source()
    profile_block = _method(source, "private static readonly RoomProfile[] Profiles", "private readonly List<CompiledInstance>")
    assert profile_block.count("new Placement(") == 30
    for family in ("warm_playroom", "sage_living_corner", "birch_art_room"):
        start = profile_block.index(f'"{family}"')
        next_starts = [profile_block.find(f'"{other}"', start + 1) for other in (
            "warm_playroom", "sage_living_corner", "birch_art_room"
        )]
        ends = [value for value in next_starts if value >= 0]
        end = min(ends) if ends else len(profile_block)
        assert profile_block[start:end].count("new Placement(") == 10
    assert "contextual_visible_object_minimum = _context.MinimumContextualObjects" in source
    assert "contextual_visible_object_count = contextualVisibleObjectCount" in source
    assert "contextualVisibleObjectCount >= _context.MinimumContextualObjects" in source
    assert "contextualVisibleObjectCount = actualCatalogAssetIds.Length" in source
    assert 'CreateRoomSurface("front_wall_surface"' in source
    assert 'CreateRoomSurface("ceiling_surface"' in source


def test_catalog_furniture_is_hash_verified_imported_and_never_a_visible_primitive():
    source = _source()
    catalog = _config()["assets"]["furniture_catalog"]
    assert catalog["asset_id"] in source
    assert catalog["archive_sha256"] in source
    assert 'private const string CatalogLicense = "CC0"' in source
    for member in catalog["members"]:
        assert f'"{member["id"]}"' in source
        assert f'"{member["sha256"]}"' in source
        assert f'"{member["semantic_class"]}"' in source
        assert f'"{member["collision_source"]}"' in source
    assert "AssetDatabase.LoadAssetAtPath<GameObject>(assetPath)" in source
    assert "PrefabUtility.InstantiatePrefab(prefab)" in source
    assert "VerifyCatalogSource(assetPath, member.Sha256)" in source
    assert "GameObject.CreatePrimitive" not in source
    assert 'visible_geometry_source = "verified_imported_kenney_mesh"' in source
    assert 'license = "repository-authored"' in source
    assert 'visible_geometry_source = "repository_authored_context_mesh"' in source


def test_activity_context_and_destination_relations_are_cell_specific_and_physics_backed():
    source = _source()
    for token in (
        "picture_book",
        "beside_picture_book",
        "shallow_bin",
        "reading_book",
        "beside_book",
        "tray",
        "craft_stick_",
        "among_craft_objects",
        "marked_support",
        "free_release_destination",
    ):
        assert f'"{token}"' in source
    assert "CreateActivityContext(table, targetPosition)" in source
    assert "CreateDestination(" in source
    assert "AddBoxCollider(instance" in source
    assert 'identity.physics_role = "reachable_static_physx_collider"' in source
    assert "physics_backed = true" in source
    assert "allVisibleContextPhysicsBacked" in source
    assert 'relation = "initial_physical_support"' in source
    assert 'relation = "intended_free_release_destination"' in source
    assert 'relation = "destination_physical_support"' in source
    assert "child_id = _context.DestinationId" in source
    assert "destination_id = _context.DestinationId" in source


def test_a_b_c_destinations_have_distinct_physical_release_geometries_not_interchangeable_labels():
    source = _source()
    activity = _method(source, "private CompiledInstance CreateActivityContext", "private static Vector3 SupportedCenter")
    assert '"shallow_bin", targetPosition - right * 0.255f + forward * 0.025f' in activity
    assert "new Vector3(0.185f, 0.050f, 0.155f)" in activity
    assert '"tray", targetPosition - right * 0.260f + forward * 0.030f' in activity
    assert "new Vector3(0.205f, 0.026f, 0.165f)" in activity
    assert '"marked_support", targetPosition - right * 0.255f + forward * 0.020f' in activity
    assert "new Vector3(0.155f, 0.010f, 0.145f)" in activity
    assert activity.count("return CreateDestination(") == 3
    destination = _method(source, "private CompiledInstance CreateDestination", "private CompiledInstance CreateAuthoredProp")
    assert "bool openContainer = wallHeightM > 0f" in destination
    assert "openContainer ? OpenContainerMesh() : UnitBoxMesh()" in destination
    assert "RegisterContextDestination(destinationId, destination.game_object.transform)" in destination
    assert "DESTINATION_RELEASE_ANCHOR" not in destination


def test_destination_dictionary_exposes_the_actual_collider_bearing_semantic_object():
    source = _source()
    destination = _method(source, "private CompiledInstance CreateDestination", "private CompiledInstance CreateAuthoredProp")
    assert "CreateAuthoredProp(" in destination
    assert '"activity_destination"' in destination
    assert '"free_release_destination"' in destination
    assert "RegisterContextDestination(destinationId, destination.game_object.transform)" in destination
    validation = _method(source, "private SceneValidationReceipt ValidateCompiledScene", "private void WriteReceipts")
    assert "RequireContextDestination(_context.DestinationId) == _destination.game_object.transform" in validation
    assert "RequireContextDestination(_context.DestinationId).GetComponentsInChildren<Collider>(true)" in validation
    assert "Any(collider => collider.enabled)" in validation


def test_contract_labels_are_bound_to_real_visible_physics_transforms_for_motion_consumers():
    source = _source()
    build = _method(source, "public void Build(GateContext context", "private static void ValidateInputs")
    assert build.index("BindGazeZoneTransforms(table)") < build.index("CreateActivityContext(table, targetPosition)")
    assert "context.Destinations.Clear()" in build
    gaze = _method(source, "private void BindGazeZoneTransforms", "private GameObject RequireSceneChild")
    assert 'zone.zone_id == "low_table"' in gaze
    assert 'zone.zone_id == "window_scan"' in gaze
    assert "candidate.role == zone.zone_id" in gaze
    assert 'RegisterContextDestination(zone.zone_id, anchor)' in gaze
    assert 'RegisterContextDestination(\n                "look_away"' in gaze
    assert 'new Placement("books", -1.58f, 4.18f, -6f, false, "bookcase")' in source
    backing = _method(source, "private static Transform CreateBackedAnchor", "private void RegisterContextDestination")
    assert "GetComponentsInChildren<Renderer>" in backing
    assert "GetComponentsInChildren<Collider>" in backing
    assert "anchor.transform.SetParent(backing, true)" in backing
    assert "RequireContextDestination(_context.DestinationId)" in source
    assert "RequireContextDestination(_context.FinalGazeZone)" in source
    assert "authoritative_destination_and_gaze_mappings" in source


def test_compiled_contract_is_authoritative_for_reach_and_target_physics():
    source = _source()
    validate = _method(source, "private static void ValidateInputs", "private void ConfigureEnvironment")
    assert "CompiledContractPath" in validate
    assert "CompiledContractSha256" in validate
    assert "TargetMidpointReachM < context.TargetReachBandMinM" in validate
    assert "TargetMidpointReachM > context.TargetReachBandMaxM" in validate
    assert "TargetDimensionsM.x <= 0f" in validate
    resolve = _method(source, "private static TargetDefinition ResolveTarget", "private static bool Approximately")
    for field in (
        "TargetSemanticId",
        "TargetInstanceId",
        "TargetGeometry",
        "TargetDimensionsM",
        "TargetMassKg",
        "TargetStaticFriction",
        "TargetDynamicFriction",
        "ContactStrategy",
    ):
        assert f"context.{field}" in resolve
    assert "TargetMidpointReachCenterM" not in source
    assert "TargetSeedReachOffsetM" not in source
    assert "TargetLateralBiasTowardRightShoulderM" not in source
    assert "compiled_midpoint_band_m = new Vector2(_context.TargetReachBandMinM, _context.TargetReachBandMaxM)" in source
    assert "lateral_bias_toward_right_shoulder_m = _context.TargetLateralBiasM" in source


def test_compiled_scene_spec_fields_are_behavioral_inputs_or_hard_failures():
    source = _source()
    validate = _method(source, "private static void ValidateInputs", "private static bool SameMultiset")
    for field in (
        "SceneEnvelopeM",
        "SceneMaterialVariant",
        "SceneZoneIds",
        "ExpectedSceneAssetIds",
        "MinimumContextualObjects",
    ):
        assert f"context.{field}" in validate
    assert "Approximately(context.SceneEnvelopeM, profile.EnvelopeM)" in validate
    assert "string.Equals(context.SceneMaterialVariant, profile.MaterialVariant" in validate
    assert "context.SceneZoneIds.SequenceEqual(profile.ZoneIds" in validate
    assert "SameMultiset(context.ExpectedSceneAssetIds, profileAssets)" in validate
    assert "context.MinimumContextualObjects <= 10" in validate
    assert "float width = _context.SceneEnvelopeM.x" in source
    assert "float height = _context.SceneEnvelopeM.y" in source
    assert "float depth = _context.SceneEnvelopeM.z" in source
    assert "for (int index = 0; index < _context.SceneZoneIds.Length; index++)" in source
    assert "string zoneId = _context.SceneZoneIds[index]" in source
    assert "switch (_context.SceneMaterialVariant)" in source
    assert "envelope_m = _context.SceneEnvelopeM" in source
    assert "material_variant = _context.SceneMaterialVariant" in source


def test_runtime_scene_receipt_revalidates_actual_assets_zones_material_envelope_and_seal():
    source = _source()
    validation = _method(source, "private SceneValidationReceipt ValidateCompiledScene", "private void WriteReceipts")
    assert "actualCatalogAssetIds" in validation
    assert "actualZoneIds" in validation
    assert "SameMultiset(actualCatalogAssetIds, _context.ExpectedSceneAssetIds)" in validation
    assert "SameMultiset(actualZoneIds, _context.SceneZoneIds)" in validation
    assert "instance.material_variant == _context.SceneMaterialVariant" in validation
    assert "Approximately(spec.envelope_m, _context.SceneEnvelopeM)" in validation
    assert "spec.minimum_contextual_objects == _context.MinimumContextualObjects" in validation
    assert "compiledInstanceAuthorityPass" in validation
    assert "compiledRelationAuthorityPass" in validation
    assert "spec.authoritative_contract_sha256 == _context.CompiledContractSha256" in validation
    assert "compiledSceneContractPass" in validation
    assert "compiled_scene_contract_authority_validated = compiledSceneContractPass" in validation
    assert "IsInProspectiveTaskView(_context.Head, spec.sightlines.target_world_m)" in validation
    assert "head.rotation * Quaternion.Euler(44f, 0f, 0f)" in source


def test_compiled_instance_physics_and_identity_fields_drive_runtime_instances():
    source = _source()
    placement = _method(source, "private CompiledInstance PlaceCatalogInstance", "private void MoveStaticSupportUnderTarget")
    assert "_context.ExpectedSceneInstances[index]" in placement
    assert "instance.name = authority.PersistentId" in placement
    assert "ScaleToCatalogDimensions(instance, authority.AssetDimensionsM)" in placement
    assert "AddCatalogColliders(instance, authority.CollisionSource)" in placement
    assert "authority.StaticFriction, authority.DynamicFriction" in placement
    assert "authority.SemanticClass" in placement


def test_profile_catalog_multisets_match_each_compiled_episode_scene_inventory():
    source = _source()
    config = _config()
    profile_block = _method(source, "private static readonly RoomProfile[] Profiles", "private readonly List<CompiledInstance>")
    contracts_module = __import__(
        "babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate.contracts",
        fromlist=["compile_contract_matrix"],
    )
    by_room = {
        contract["scene_spec"]["room_family"]: sorted(
            instance["asset_id"] for instance in contract["scene_spec"]["instances"]
        )
        for contract in contracts_module.compile_contract_matrix(config)
    }
    families = ("warm_playroom", "sage_living_corner", "birch_art_room")
    for index, family in enumerate(families):
        start = profile_block.index(f'"{family}"')
        end = profile_block.index(f'"{families[index + 1]}"', start) if index + 1 < len(families) else len(profile_block)
        placements = re.findall(r'new Placement\("([^"]+)"', profile_block[start:end])
        assert sorted(["tableCoffee", *placements]) == by_room[family]


def test_all_three_targets_match_the_frozen_geometry_identity_and_contact_records():
    source = _source()
    for target in _config()["assets"]["interactive_targets"]:
        assert f'"{target["persistent_id"]}"' in source
        assert f'{target["semantic_id"]}, {target["instance_id"]}' in source
        assert f'"{target["geometry"]}"' in source
        assert f'"{target["geometry_spec_sha256"]}"' in source
        assert f'{target["mass_kg"]:.2f}f' in source
        assert f'"{target["collision_source"]}"' in source
    assert "RoundedGraspToyMesh()" in source
    assert "HandledCupMesh()" in source
    assert "BeveledBlockMesh()" in source
    assert "SphereCollider sphere" in source
    assert "CapsuleCollider body" in source
    assert "BoxCollider handle" in source
    assert "BoxCollider box" in source
    assert "target/contact strategy mismatch" in source
    assert "target/destination mismatch" in source
    assert "target/room-family mismatch" in source
    assert "authoritative compiled-contract target does not match the approved asset template" in source


def test_target_is_initialized_once_then_left_as_a_free_physx_body():
    source = _source()
    target_method = _method(source, "private void CreateFreeTarget", "private SceneSpecReceipt CompileSceneSpec")
    assert "target.transform.SetPositionAndRotation" in target_method
    assert "target.transform.SetParent" not in target_method
    assert "body.isKinematic = false" in target_method
    assert "body.useGravity = true" in target_method
    assert "body.mass = _target.MassKg" in target_method
    assert "PhysicsTruthObjectIdentity" in target_method
    assert "truthIdentity.persistent_id = _target.PersistentId" in target_method
    assert target_method.index("target.transform.SetPositionAndRotation") < target_method.index("target.AddComponent<Rigidbody>()")
    after_rigidbody = target_method[target_method.index("target.AddComponent<Rigidbody>()") :]
    assert "target.transform" not in after_rigidbody
    assert "target.AddComponent<Joint>" not in target_method


def test_no_assistance_controller_or_object_trajectory_mechanism_exists():
    source = _source()
    for forbidden in (
        "FixedJoint",
        "ConfigurableJoint",
        ".AddForce(",
        ".AddTorque(",
        ".MovePosition(",
        ".MoveRotation(",
        "isKinematic = true",
        "void Update(",
        "void FixedUpdate(",
        "OnCollisionStay(",
    ):
        assert forbidden not in source
    assert "post_initialization_transform_writes = 0" in source
    assert "target_external_force_or_spring_commands = 0" in source
    assert "target_unparented" in source
    assert "target_joint_count" in source


def test_camera_aware_swept_volume_and_receipts_prospectively_reject_invalid_layouts():
    source = _source()
    assert "RejectFurnitureInInteractionCorridor" in source
    assert "prospective swept interaction corridor blocked" in source
    assert "RejectOverlappingFurnishings" in source
    assert "RejectObjectsOutsideFinishedRoom" in source
    assert "prospective finished-room envelope violation" in source
    assert "HasClearSightlineToObject" in source
    assert "IsInProspectiveTaskView" in source
    assert "destination_visible_from_head_geometry" in source
    assert "HasClearSightlineToAnchor" in source
    assert "final_gaze_zone_compiled" in source
    assert "final_gaze_visible_from_head_geometry" in source
    assert "camera_aware_target_and_destination_sightlines" in source
    assert "deterministic_prospective_rejection_only = true" in source
    assert "persistentIdentitiesUnique" in source
    assert "source_hashes_verified" in source
    assert 'private const string SceneSchema = "embodied.scene_spec.v1"' in source
    assert 'status = "COMPILED_CONTRACT_VALIDATED_ONLY"' in source
    assert "visual_pass_claimed = false" in source
    assert "physical_episode_pass_claimed = false" in source
    assert "It is not visual evidence" in source
    assert re.search(r"supportSeparation >= 0f && supportSeparation <= 0\.002f", source)
    assert "midpointReach >= _context.TargetReachBandMinM" in source
    assert "midpointReach <= _context.TargetReachBandMaxM" in source
