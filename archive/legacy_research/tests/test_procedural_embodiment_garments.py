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


def test_hybrid_colliders_keep_palms_kinematic_and_make_all_finger_segments_dynamic():
    text = source_text()
    assert "BuildPalmBox" in text
    assert "AddBoneCapsule" in text
    assert '"COLLIDER_" + side + "_palm"' in text
    assert 'side + "_" + Digits[digit] + "_segment_" + segment' in text
    assert "collider.transform.parent == bone" in text
    assert "body.isKinematic = true" in text
    assert "CollisionDetectionMode.ContinuousSpeculative" in text
    assert "BindHybridColliderDrivers(context)" in text
    assert "ConfigureDynamicFingerBody" in text
    assert "body.isKinematic = false" in text
    assert "CollisionDetectionMode.ContinuousDynamic" in text
    assert "context.FingerBodies.Add(segmentKey, body)" in text
    assert "context.FingerAuthorityBones.Add(segmentKey, authorityBones[segment - 1])" in text
    assert "context.FingerBodies.Count != 30" in text
    assert "AddFittedEnvelopeSpheres" in text
    assert "both_palms_and_all_finger_segments_have_colliders" in text


def test_each_dynamic_chain_uses_bounded_compliant_physx_joints_connected_to_the_palm():
    text = source_text()
    assert "ConfigureCompliantFingerJoint" in text
    assert "joint.connectedBody = connectedBody" in text
    assert "joint.xMotion = ConfigurableJointMotion.Locked" in text
    assert "joint.angularXMotion = ConfigurableJointMotion.Limited" in text
    assert "joint.rotationDriveMode = RotationDriveMode.Slerp" in text
    assert "joint.slerpDrive = new JointDrive" in text
    assert "maximumForce = 1.15f" in text
    assert "joint.enableCollision = false" in text
    assert "dynamic_finger_body_count = context.FingerBodies.Count" in text
    assert "compliant_finger_joint_count = context.FingerJoints.Count" in text


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


def test_frozen_registration_bounds_and_affected_vertex_limit_are_part_of_pass_logic():
    text = source_text()
    assert "const float SkinColliderToleranceM = .005f;" in text
    assert "const float GarmentBodyToleranceM = .002f;" in text
    assert "const float GarmentAffectedVertexFractionMax = .001f;" in text
    assert "x.affected_fraction <= GarmentAffectedVertexFractionMax" in text
    assert "affectedUniqueVertexIndices.Add(i)" in text
    assert "affectedUniqueVertexIndices.Count / (float)usedVertexIndices.Length <= GarmentAffectedVertexFractionMax" in text
    assert "affected_unique_vertex_numerator = affectedUniqueVertexIndices.Count" in text
    assert "affected_unique_vertex_denominator = usedVertexIndices.Length" in text
    assert "unique garment vertices penetrated at any audited pose / unique garment vertices" in text
    assert "penetration.affected / (float)penetration.count" not in text
    assert "garment_affected_vertex_fraction_max = GarmentAffectedVertexFractionMax" in text


def test_all_embodiment_colliders_are_registered_independently_of_avatar_hierarchy():
    text = source_text()
    assert "context.AvatarColliders.Clear()" in text
    assert "context.AvatarColliders.Add(collider)" in text
    assert "context.AvatarColliders.Add(queryCollider)" in text
    assert "context.AvatarColliders.Count != anatomicalColliders.Count" in text
    assert "every force/query embodiment collider must be registered independently of hierarchy" in text
    assert "context.AvatarColliders.Count == anatomicalColliders.Count" in text
    assert "registered_avatar_collider_count = context.AvatarColliders.Count" in text
    assert "avatar_collider_registry_complete = context.AvatarColliders.Count == anatomicalColliders.Count" in text


def test_body_segment_registry_covers_required_full_upper_body_proprioception():
    text = source_text()
    required = {
        "root", "pelvis", "torso", "neck", "head",
        "left_shoulder", "left_upper_arm", "left_elbow", "left_lower_arm",
        "left_forearm", "left_wrist", "left_palm",
        "right_shoulder", "right_upper_arm", "right_elbow", "right_lower_arm",
        "right_forearm", "right_wrist", "right_palm",
    }
    assert "context.BodySegments.Clear()" in text
    assert "body_segment_ids = context.BodySegments.Keys.OrderBy" in text
    for segment in required:
        assert f'"{segment}"' in text or f'side + "_{segment.split("_", 1)[-1]}"' in text
    assert 'context.BodySegments.Add(side + "_lower_arm", lowerArm)' in text
    assert 'context.BodySegments.Add(side + "_forearm", forearmBody.transform)' in text
    assert 'context.BodySegments.Add(side + "_palm", palmBody.transform)' in text


def test_self_collision_uses_selective_adjacent_exclusions_and_runtime_swept_evidence():
    text = source_text()
    assert "Physics.IgnoreLayerCollision" not in text
    assert "Physics.GetIgnoreLayerCollision(AnatomicalColliderLayer, AnatomicalColliderLayer)" in text
    assert "only explicit adjacent links may be excluded" in text
    assert "ConfigureSelectiveAdjacentCollisionExclusions" in text
    assert "AreSameOrAdjacentAnatomicalLinks" in text
    assert "Physics.IgnoreCollision(a, b, true)" in text
    assert "ignoredAdjacentColliderPairs.Contains(ColliderPairKey(a, b))" in text
    assert "MeasureProspectiveNonAdjacentSelfClearance" in text
    assert "SelfSweepMinimumSubsteps = 8" in text
    assert "SelfSweepMaximumSubsteps = 32" in text
    assert "SelfSweepMaximumSurfaceMotionPerSubstepM = .0005f" in text
    assert "a.bounds.extents.magnitude * rotationADeg * Mathf.Deg2Rad" in text
    assert "b.bounds.extents.magnitude * rotationBDeg * Mathf.Deg2Rad" in text
    assert "SelfSweepMaximumRotationPerSubstepDeg = 1f" in text
    assert "requiredSubsteps" in text
    assert "sweepSample <= substeps" in text
    assert "ConservativeSweptBoundingSphereClearance" in text
    assert "broadphase_certified_separated_pairs = broadphaseCertifiedPairs" in text
    assert "var startPoses = anatomicalColliders" in text
    assert "PredictEndOfStepPose" in text
    assert "start.position + body.linearVelocity * dt" in text
    assert "MeasureProspectiveNonAdjacentSelfClearance(context, startPoses, prospectivePoses)" in text
    assert "Physics.ComputePenetration" in text
    assert "collider_a = a.name" in text
    assert "collider_b = b.name" in text
    assert "incompleteSelfSweepIntervals == 0" in text
    assert "self_sweep_motion_bounds_respected" in text
    assert "sweep_coverage_complete = evaluatedPairs > 0 && incompleteIntervals == 0" in text
    assert "selfClearanceSteps.Count == expectedSamples" in text
    assert "nonAdjacentSelfOverlapSamples == 0" in text
    assert "self_clearance_sampled_every_physics_step = completeSelfClearance" in text
    assert "non_adjacent_anatomy_clearance_passed = selfClearancePass" in text
    assert "completeMotion && bodyPass && garmentPass && selfClearancePass && solverSelfClearancePass && penetrationPass" in text


def test_registration_report_carries_physx_measured_interaction_penetration_receipts():
    text = source_text()
    assert "SampleInteractionPenetration(context, context.PhysicsStep - 1)" in text
    assert "SampleInteractionPenetration(context, context.PhysicsStep)" in text
    assert "interactionPenetrationSampledSteps.Add(sampledPhysicsStep)" in text
    assert "value.provenance == TruthSource.PhysXMeasured" in text
    assert "finger_object_max_penetration_m = fingerObjectMaxPenetrationM" in text
    assert "target_support_max_penetration_m = targetSupportMaxPenetrationM" in text
    assert "finger_object_penetration_provenance" in text
    assert "target_support_penetration_provenance" in text
    assert "fingerSamples > 0 && supportSamples > 0" in text
    assert "FrozenGate.FingerObjectPenetrationMaxM" in text
    assert "FrozenGate.SupportPenetrationMaxM" in text


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
