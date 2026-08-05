import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/EmbodimentGarments.cs"
CONFIG = ROOT / "configs/embodied_simulation_procedural_scene_gate.json"


def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_exact_integration_class_implements_frozen_interface():
    text = source_text()
    assert "namespace ProceduralSceneGate" in text
    assert "public sealed class EmbodimentGarments : IEmbodimentGarmentModule" in text
    assert "void Build(GateContext context, string avatarAssetPath)" in text
    assert "void ApplyGarmentConfiguration(GateContext context, string configurationId)" in text
    assert "void MeasureRegistration(GateContext context, string reportPath)" in text


def test_one_cc0_weighted_mpfb_body_and_complete_anatomical_hands_are_required():
    text = source_text()
    assert "PrefabUtility.InstantiatePrefab" in text
    assert "weighted.Length != 1" in text
    assert 'document.assets.avatar.license != "CC0"' in text
    assert "VerifyWeightedFiveFingerHands" in text
    assert 'new[] { ".L", ".R" }' in text
    assert "segment <= 3" in text
    assert "required.Distinct().Count() != 32" in text
    assert "BoneWeightFor(x, index) > 0" in text
    assert "BuildAnatomicalColliders" in text
    assert "CompleteHandColliderSet" in text


def test_bone_bound_kinematic_colliders_cover_palms_arms_and_every_digit_segment():
    text = source_text()
    assert "BuildPalmBox" in text
    assert "AddBoneCapsule" in text
    assert '"COLLIDER_" + side + "_palm"' in text
    assert 'side + "_" + Digits[digit] + "_segment_" + segment' in text
    assert "collider.transform.parent == bone" in text
    assert "body.isKinematic = true" in text
    assert "CollisionDetectionMode.ContinuousSpeculative" in text
    assert "AddFittedEnvelopeSpheres" in text
    assert "both_palms_and_all_finger_segments_have_colliders" in text


def test_garments_are_loaded_from_frozen_catalog_not_variant_branches():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    text = source_text()
    ids = {row["garment_configuration_id"] for row in config["avatar_specs"]}
    assert ids == {"sunset_play", "forest_layer", "sky_overall"}
    for configuration_id in ids:
        assert configuration_id not in text
    assert "LoadFrozenDocument" in text
    assert "PROCEDURAL_SCENE_GATE_CONFIG" in text
    assert "frozen.avatar_specs.Where" in text
    assert "OrderBy(x => x.layer)" in text
    assert "spec.fit_offset_m" in text
    assert "spec.color_rgba" in text
    assert "spec.material" in text


def test_bind_derived_garments_preserve_the_authoritative_skin_binding():
    text = source_text()
    assert "Vector3[] sourceVertices = source.vertices" in text
    assert "boneWeights = source.boneWeights" in text
    assert "bindposes = source.bindposes" in text
    assert "renderer.bones = bodyRenderer.bones" in text
    assert "renderer.rootBone = bodyRenderer.rootBone" in text
    assert "renderer.bones.SequenceEqual(bodyRenderer.bones)" in text
    assert "bodyRenderer.transform.InverseTransformVector(worldNormal * spec.fit_offset_m)" in text


def test_registration_is_observation_only_and_dense_over_the_physics_clock():
    text = source_text()
    assert "SampleRegistrationAtPhysicsStep" in text
    assert "bodyRenderer.BakeMesh(bodyBake, true)" in text
    assert "garment.renderer.BakeMesh(garment.baked, true)" in text
    assert "context.PhysicsStep == lastSampledStep" in text
    assert "context.PhysicsStep == lastSampledStep + 1" in text
    assert "FrozenGate.DurationSeconds * FrozenGate.PhysicsHz" in text
    assert "sampled_every_physics_step = completeMotion" in text
    assert "affected_vertex_distribution_counts" in text
    assert "body_collider_registration" in text
    assert "collider.transform.parent == bone" in text


def test_unavailable_registration_fields_are_explicit_not_fabricated():
    text = source_text()
    assert "TruthSource.Unavailable.ToString()" in text
    assert "garment_body_full_triangle_intersection_provenance" in text
    assert "garment_self_intersection_provenance" in text
    assert "no validated robust skinned self-intersection solver is present" in text
    assert "passed = completeMotion && bodyPass && garmentPass" in text


def test_module_never_creates_a_second_motion_or_object_assistance_path():
    text = source_text()
    forbidden = [
        "FixedJoint",
        "AddForce",
        "TargetBody.MovePosition",
        "TargetBody.MoveRotation",
        "Physics.Simulate",
        "AnimationCurve",
        "targetBody.transform",
        "TargetBody.transform.position",
        "TargetBody.transform.rotation",
    ]
    for token in forbidden:
        assert token not in text
    assert "animator.enabled = false" in text
    assert "animation.enabled = false" in text
    assert "independently_advanced_animation = false" in text
    assert "moduleCreatedProxyRenderers = 0" in text
