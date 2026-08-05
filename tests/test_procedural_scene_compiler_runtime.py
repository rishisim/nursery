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


def _method(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


def test_compiler_implements_the_frozen_interface_and_three_profile_path():
    source = _source()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert "public sealed class ProceduralSceneCompiler : ISceneCompilerModule" in source
    assert "public void Build(GateContext context, string furnitureAssetRoot)" in source
    assert source.count("new RoomProfile(") == 3
    for scene in config["scene_matrix"]:
        assert f'"{scene["room_family"]}"' in source
        assert f'"{scene["material_variant"]}"' in source
        assert str(scene["seed"]) not in source, "frozen seed values must not select bespoke scene code"
    assert "new System.Random(context.Seed)" in source
    assert "PositiveModulo(context.Seed, 3)" in source
    assert "TargetMidpointReachCenterM = 0.36f" in source
    assert "TargetSeedReachOffsetM = 0.020f" in source
    assert "TargetLateralBiasTowardRightShoulderM = 0.025f" in source
    assert "towardRightShoulder" in source


def test_runtime_catalog_mirrors_frozen_hashes_dimensions_and_import_convention():
    source = _source()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    catalog = config["assets"]["furniture_catalog"]
    assert catalog["asset_id"] in source
    assert catalog["archive_sha256"] in source
    assert 'private const string CatalogLicense = "CC0"' in source
    for member in catalog["members"]:
        assert f'"{member["id"]}"' in source
        assert f'"{member["sha256"]}"' in source
        assert f'"{member["semantic_class"]}"' in source
        assert f'"{member["collision_source"]}"' in source
    assert 'AssetDatabase.LoadAssetAtPath<GameObject>(assetPath)' in source
    assert 'PrefabUtility.InstantiatePrefab(prefab)' in source
    assert 'VerifyCatalogSource(assetPath, member.Sha256)' in source
    assert "GameObject.CreatePrimitive" not in source
    assert 'visible_geometry_source = "verified_imported_kenney_mesh"' in source


def test_reachable_colliders_and_distant_noninteractive_roles_are_explicit():
    source = _source()
    assert "AddCatalogColliders(instance, member.CollisionSource)" in source
    assert '"reachable_static_physx_collider"' in source
    assert '"distant_noninteractive_static_physx_collider"' in source
    assert "identity.interactive = false" in source
    assert "reachable_elements_have_physx_colliders" in source
    assert "distant_decor_explicit_noninteractive" in source
    assert 'new Placement("rugRectangle"' in source
    assert 'new Placement("pottedPlant"' in source
    assert 'new Placement("books"' in source


def test_target_exactly_matches_frozen_free_physx_contract():
    source = _source()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = config["assets"]["interactive_target"]
    target_method = _method(source, "private void CreateFreeTarget", "private SceneSpecReceipt CompileSceneSpec")
    assert f'private const string TargetPersistentId = "{frozen["persistent_id"]}"' in source
    assert f"private const int TargetSemanticId = {frozen['semantic_id']};" in source
    assert f"private const int TargetInstanceId = {frozen['instance_id']};" in source
    assert frozen["geometry_spec_sha256"] in source
    assert "private const float TargetDiameterM = 0.055f;" in source
    assert "private const float TargetMassKg = 0.12f;" in source
    assert "private const float TargetStaticFriction = 1.0f;" in source
    assert "private const float TargetDynamicFriction = 0.9f;" in source
    assert "target.transform.SetParent" not in target_method
    assert "body.isKinematic = false" in target_method
    assert "body.useGravity = true" in target_method
    assert "body.mass = TargetMassKg" in target_method
    assert frozen["collision_source"] == "analytic_sphere_matching_render_envelope"
    assert "target.AddComponent<SphereCollider>()" in target_method
    assert "targetCollider.radius = 0.5f" in target_method
    assert "MeshCollider" not in target_method
    assert "PhysicsTruthObjectIdentity" in target_method
    assert "truthIdentity.persistent_id = TargetPersistentId" in target_method
    assert "truthIdentity.semantic_id = TargetSemanticId.ToString()" in target_method
    assert "truthIdentity.instance_id = TargetInstanceId.ToString()" in target_method
    assert target_method.index("target.transform.SetPositionAndRotation") < target_method.index("target.AddComponent<Rigidbody>()")
    after_rigidbody = target_method[target_method.index("target.AddComponent<Rigidbody>()") :]
    assert "target.transform" not in after_rigidbody


def test_no_assistance_mechanism_or_target_control_loop_exists():
    source = _source()
    forbidden = (
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
    )
    for token in forbidden:
        assert token not in source
    assert "post_initialization_transform_writes = 0" in source
    assert "target_post_initialization_transform_writes = 0" in source
    assert "target_external_force_or_spring_commands = 0" in source
    assert "target_unparented" in source
    assert "target_joint_count" in source


def test_scene_receipts_cover_zones_support_reach_sightline_and_scientific_scope():
    source = _source()
    assert 'private const string SceneSchema = "embodied.scene_spec.v1"' in source
    assert '"scene_spec.json"' in source
    assert '"scene_compiler_validation.json"' in source
    assert "support_relations" in source
    assert "target_from_right_shoulder_m" in source
    assert "target_from_left_shoulder_m" in source
    assert "compiled_midpoint_band_m" in source
    assert "aperture_aware = true" in source
    assert "seed_specific_retuning = false" in source
    assert "engine_observed_weighted_avatar_transform_at_scene_initialization" in source
    assert "HasClearSightline" in source
    assert "target_visible_from_head_geometry" in source
    assert "no_visible_primitive_furniture = true" in source
    assert 'status = "COMPILED_CONTRACT_VALIDATED_ONLY"' in source
    assert "visual_pass_claimed = false" in source
    assert "physical_episode_pass_claimed = false" in source
    assert "It is not visual evidence" in source
    assert re.search(r"supportSeparation >= 0f && supportSeparation <= 0\.002f", source)
    assert re.search(r"rightReach >= 0\.34f && rightReach <= 0\.48f", source)
    assert re.search(r"leftReach >= 0\.34f && leftReach <= 0\.48f", source)
