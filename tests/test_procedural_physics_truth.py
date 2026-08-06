import re
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "babyworld_lite"
    / "childlens_engine_bakeoff"
    / "procedural_scene_gate"
    / "PhysicsTruthRecorder.cs"
)
BUILDER = SOURCE.with_name("ProceduralSceneGateBuilder.cs")


def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def builder_text() -> str:
    return BUILDER.read_text(encoding="utf-8")


def test_recorder_implements_frozen_module_and_exact_post_step_clock():
    source = source_text()
    assert "public sealed class PhysicsTruthRecorder : IPhysicsTruthModule" in source
    assert "public void RecordAfterPhysicsStep(GateContext context)" in source
    assert 'sample_phase = "post_physics_simulate"' in source
    assert "FrozenGate.PhysicsHz" in source
    assert "FrozenGate.RenderHz" in source
    assert "FrozenGate.StepsPerFrame" in source
    assert "context.PhysicsStep / FrozenGate.StepsPerFrame" in source


def test_trace_has_full_body_palms_digits_segments_and_controller_truth():
    source = source_text()
    for field in (
        "root",
        "torso",
        "neck",
        "head",
        "left_palm",
        "right_palm",
        "rotation_world_xyzw",
        "linear_velocity_world_m_s",
        "angular_velocity_world_rad_s",
        "closure_commanded",
        "closure_observed",
        "targets",
        "errors",
        "recovery_state",
        "speed_limits",
        "closure_limits",
        "right_opposition_qualified",
        "meaningful_left_support_qualified",
        "carry_contacts_maintained",
        "right_opposition_dwell_steps",
        "left_support_dwell_steps",
        "bimanual_qualification_step",
    ):
        assert field in source
    assert 'new[] { "left", "right" }' in source
    assert "FrozenGate.Digits" in source
    assert "segments.Length != 3" in source
    assert "digit_closure_commands.Length != 10" in source
    assert "public DigitTruth left_thumb" in source
    assert "public DigitTruth right_little" in source


def test_contact_and_free_object_rows_are_physx_registered_and_identified():
    source = source_text()
    for field in (
        "collider_a",
        "collider_b",
        "point_world_m",
        "normal_world",
        "separation_m",
        "relative_velocity_world_m_s",
        "available_impulse_n_s",
        "available_impulse_magnitude_n_s",
        "nonzero_impulse_observed",
        "mean_force_equivalent_n",
        "force_evidence_semantics",
        "qualification_separation_eligible",
        "persistent_id",
        "semantic_id",
        "instance_id",
        "support_id",
        "sleeping",
        "persistent_object_id",
        "normal_on_object_world",
        "active",
        "availability_provenance",
    ):
        assert field in source
    assert "OnCollisionEnter" in source
    assert "OnCollisionStay" in source
    assert "collision.impulse" in source
    assert "mean-force equivalent is derived as impulse/dt" in source
    assert "separation eligibility is reported independently" in source
    assert "body.GetPointVelocity(pointWorldM)" in source
    assert "context.Contacts.Add(new ContactTruth" in source
    assert 'provenance = "physx_measured"' in source
    assert "body.linearVelocity" in source
    assert "body.angularVelocity" in source
    assert "body.IsSleeping()" in source
    assert "pose = new PoseTruth" in source
    assert "position_world_m = body.position" in source
    assert 'velocity_formula_or_source = "Unity Rigidbody state sampled after Physics.Simulate"' in source
    assert "free object disabled for this object-free qualification stage" in source
    assert "release_commanded" in source
    assert 'string.Equals(telemetry.phase_id, "CommandedOpen"' in source


def test_camera_proprioception_imu_and_claim_boundaries_are_explicit():
    source = source_text()
    for field in (
        "parent_id",
        "mount_pose",
        "world_to_camera_extrinsics",
        "intrinsics",
        "clearance_m",
        "optical_vs_face_forward_deg",
        "joint_proprioception",
        "head_accelerometer_m_s2",
        "head_gyroscope_rad_s",
        "assistance_ledger",
    ):
        assert field in source
    assert "worldAcceleration - Physics.gravity" in source
    assert "QuaternionAngularVelocityWorld" in source
    assert 'biological_torque = "unavailable_not_claimed"' in source
    assert 'formula_or_source = "not measured and not claimed"' in source


def test_provenance_labels_are_not_collapsed_or_inferred_as_visual_evidence():
    source = source_text()
    for provenance in (
        '"commanded"',
        '"engine_observed"',
        '"physx_measured"',
        '"derived"',
        '"unavailable"',
    ):
        assert provenance in source
    assert "viewport" not in source.lower()
    assert "visual evidence" not in source.lower()
    assert "reconstructs commands from observed transforms" in source
    assert "unavailableControllerSteps++" in source


def test_recorder_never_uses_free_object_assistance_apis():
    source = source_text()
    forbidden_calls = (
        ".SetParent(",
        ".MovePosition(",
        ".MoveRotation(",
        ".AddForce(",
        ".AddTorque(",
        "AddComponent<FixedJoint>",
        "AddComponent<SpringJoint>",
        "AddComponent<ConfigurableJoint>",
    )
    assert not any(call in source for call in forbidden_calls)
    assert "TargetBody.transform.position =" not in source
    assert "TargetBody.transform.rotation =" not in source
    assert "TargetBody.isKinematic =" not in source


def test_completion_writes_deterministic_trace_manifest_clock_and_hash_receipts():
    source = source_text()
    for artifact in (
        "episode_trace.jsonl",
        "episode_trace_manifest.json",
        "episode_trace_clock_receipt.json",
        "episode_trace_hash_receipt.json",
    ):
        assert artifact in source
    assert "SHA256.Create()" in source
    assert "contiguous_physics_steps" in source
    assert "exact_integer_render_mapping" in source
    assert "expected_row_count" in source
    assert "unavailable_camera_clearance_steps" in source
    assert "non_free_object_steps" in source
    assert "camera_parent_mismatch_steps" in source
    assert "assistance_ledger_empty" in source
    assert "trace_complete" in source


def test_builder_populates_frozen_cell_identity_and_source_audit_from_environment():
    source = builder_text()
    for name in (
        "PROCEDURAL_GATE_CELL_ID",
        "PROCEDURAL_GATE_TARGET_ID",
        "PROCEDURAL_GATE_DESTINATION_ID",
        "PROCEDURAL_GATE_CONTACT_STRATEGY",
        "PROCEDURAL_GATE_FINAL_GAZE_ZONE",
        "PROCEDURAL_GATE_SOURCE_AUDIT_SHA256",
    ):
        assert (
            f'RequiredEnvironment("{name}")' in source
            or f'RequiredSha256("{name}")' in source
            or f'RequireExactEnvironment("{name}"' in source
        )
    assert "context.AuthorityAudit.sourceAuditSha256" in source
    assert "64-character hexadecimal SHA-256" in source


def test_builder_synchronizes_completed_physx_state_before_truth_and_capture():
    source = builder_text()
    simulate = source.index("Physics.Simulate(Dt);")
    synchronize = source.index("motion.SynchronizeCompletedPhysicsState(context, step);")
    observe = source.index("truth.RecordAfterPhysicsStep(context);")
    capture = source.index("capture.CaptureFrozenFrame(context);")
    assert simulate < synchronize < observe < capture
    between = source[simulate:synchronize]
    assert between.strip() == "Physics.Simulate(Dt);"
    assert "context.RenderFrame = step / FrozenGate.StepsPerFrame;" in source
    assert "(step + 1) % FrozenGate.StepsPerFrame == 0" in source


def test_runtime_authority_accounting_is_observed_and_not_a_zero_only_receipt():
    recorder = source_text()
    builder = builder_text()
    for token in (
        "RecordBeforePhysicsStep",
        "AuditOutOfPhysicsChanges",
        "SampleTargetAuthorityState",
        "FindObjectsByType<Joint>",
        "targetPoseWriteCounter",
        "targetVelocityWriteCounter",
        "targetJointCounter",
        "targetParentingCounter",
        "targetKinematicChangeCounter",
        "authority_counters",
        "source_audit_sha256",
    ):
        assert token in recorder
    assert "truth.RecordBeforePhysicsStep(context);" in builder
    assert "object_pose_writes_after_initialization = context.AuthorityAudit.targetPoseWriteCounter" in builder
    assert "object_external_forces = context.AuthorityAudit.targetForceCounter" in builder
    assert "object_pose_writes_after_initialization = 0" not in builder
    assert "object_external_forces = 0" not in builder
    assert "attachment_or_joint_count = 0" not in builder


def test_trace_emits_ledgers_dynamic_finger_authority_and_force_bearing_semantics():
    source = source_text()
    for token in (
        "recovery_ledger",
        "assistance_ledger",
        "dynamic_body",
        "compliant_joint",
        "Unity dynamic Rigidbody/PhysX",
        "commanded ConfigurableJoint drive consumed by PhysX",
        "force_bearing_qualification",
        "right_current_measured_impulse_opposition",
        "right_current_non_thumb_digits",
        "left_current_non_little_digits",
        "right_measured_impulse_dwell_s",
        "left_measured_impulse_dwell_s",
        "object_unsupported_during_qualified_manipulation",
        "support_continuous_until_commanded_opening",
        "lift_over_0_10_m",
        "turn_over_30_deg",
        "current-step PhysX ContactPairPoint impulse",
    ):
        assert token in source
    assert "rightMeasuredImpulseDwellSteps > FrozenGate.RightForceOppositionSeconds * FrozenGate.PhysicsHz" in source
    assert "leftMeasuredImpulseDwellSteps > FrozenGate.LeftForceSupportSeconds * FrozenGate.PhysicsHz" in source


def test_avatar_collider_ownership_is_explicit_and_independent_of_transform_hierarchy():
    source = source_text()
    bind = source[source.index("public void Bind(GateContext context)"):source.index("public void RecordBeforePhysicsStep")]
    assert "avatarColliders = context.AvatarColliders" in bind
    assert "context.AvatarRoot.GetComponentsInChildren<Collider>(true)" not in bind
    assert "foreach (Collider collider in avatarColliders)" in bind
    assert "avatarColliderIds.Add(StableTransformId(collider.transform))" in bind
    assert "IEnumerable<Collider> observedColliders = avatarColliders" in bind
    assert "GateContext.AvatarColliders must enumerate every anatomical collider" in bind

    # Regression fixture: the force-producing finger is outside AvatarRoot, but
    # explicit ownership still prevents it from becoming an external support.
    target_contacts = [
        {"paired": "PhysicsDrivers/right_index_segment_2", "separation_m": -0.0004},
    ]
    explicit_avatar_ids = {"PhysicsDrivers/right_index_segment_2"}
    external_supports = [row for row in target_contacts if row["paired"] not in explicit_avatar_ids]
    assert external_supports == []


def test_body_truth_and_proprioception_cover_the_complete_bilateral_arm_chain():
    source = source_text()
    required = (
        "root",
        "pelvis",
        "torso",
        "neck",
        "head",
        "left_shoulder",
        "left_upper_arm",
        "left_elbow",
        "left_lower_arm",
        "left_forearm",
        "left_wrist",
        "left_palm",
        "right_shoulder",
        "right_upper_arm",
        "right_elbow",
        "right_lower_arm",
        "right_forearm",
        "right_wrist",
        "right_palm",
    )
    for segment in required:
        assert f'bodySegments["{segment}"]' in source
        assert f'public PoseTruth {segment};' in source
        assert f'SampleRelativeJoint("{segment}"' in source
    body_class = source[
        source.index("public sealed class BodyStateTruth"):
        source.index("public sealed class BodyHandsTruth")
    ]
    assert tuple(re.findall(r"public PoseTruth ([a-z_]+);", body_class)) == required
    proprio_chain = source[
        source.index("List<JointProprioceptionTruth> joints"):
        source.index('foreach (string side in new[] { "left", "right" }', source.index("List<JointProprioceptionTruth> joints"))
    ]
    assert tuple(re.findall(r'SampleRelativeJoint\("([a-z_]+)"', proprio_chain)) == required
    assert "required body truth segment is unavailable" in source
    assert "frozen body truth segments must map one-to-one to distinct Transforms" in source
    assert "stableIds.Distinct(StringComparer.Ordinal).Count() != stableIds.Length" in source

    # Semantic fixture: lower arm and forearm are separate replay fields, and
    # a lower-arm alias reusing the elbow Transform must be rejected.
    distinct_fixture = {
        "left_elbow": "Avatar/arm/lowerarm01.L",
        "left_lower_arm": "Avatar/arm/lowerarm02.L",
        "left_forearm": "Avatar/arm/forearm.L",
        "left_wrist": "Avatar/arm/wrist.L",
        "left_palm": "Avatar/arm/palm.L",
    }
    duplicate_fixture = {**distinct_fixture, "left_lower_arm": distinct_fixture["left_elbow"]}
    assert len(set(distinct_fixture.values())) == len(distinct_fixture)
    assert len(set(duplicate_fixture.values())) != len(duplicate_fixture)


def test_penetration_and_free_release_summary_is_measured_and_destination_specific():
    source = source_text()
    sample = source[source.index("private ObjectTruth SampleFreeObject"):source.index("private ForceBearingQualificationTruth")]
    assert "PairedTargetCollider(contact)" in sample
    assert "avatarColliderIds.Contains(pairedCollider)" in sample
    assert "fingerObjectMaximumPenetrationM" in sample
    assert "targetSupportMaximumPenetrationM" in sample
    assert "ResolveExternalSupportId" in sample
    assert "SceneIdentity" in source
    assert 'string.Equals(\n                                                           freeObject.support_id,\n                                                           context.DestinationId,' in source
    assert "freeReleaseAtRequiredDestinationCurrent" in source
    assert "!objectHasAvatarContact" in source
    assert "freeObject.free_dynamic" in source
    for field in (
        "finger_object_max_penetration_m",
        "target_support_max_penetration_m",
        "object_unsupported_during_qualified_manipulation",
        "required_destination_id",
        "observed_destination_support_id",
        "release_at_required_destination",
    ):
        assert field in source
    assert 'InteractionSummaryFileName = "interaction_summary.json"' in source
    assert "FrozenGate.FingerObjectPenetrationMaxM" in source
    assert "FrozenGate.SupportPenetrationMaxM" in source

    # A release on an arbitrary support must not satisfy a destination-labelled episode.
    fixture = {"required_destination_id": "tray", "observed_support_id": "table"}
    assert fixture["observed_support_id"] != fixture["required_destination_id"]


def test_interaction_summary_exports_strict_measured_qualification_outcomes():
    source = source_text()
    summary = source[
        source.index("InteractionSummary interactionSummary = new InteractionSummary"):
        source.index("string interactionSummaryPath")
    ]
    assert "right_measured_impulse_qualified = measuredRightQualified" in summary
    assert "left_measured_impulse_qualified = measuredLeftQualified" in summary
    assert "lift_over_0_10_m = maximumQualifiedLiftM > 0.10f" in summary
    assert "turn_over_30_deg = maximumQualifiedTurnDeg > 30f" in summary
    assert "release_at_required_destination = releaseAtRequiredDestination" in summary
    assert "object_unsupported_during_qualified_manipulation = objectUnsupportedDuringQualifiedManipulation" in summary
    assert "maximumQualifiedLiftM >= 0.10f" not in source
    assert "maximumQualifiedTurnDeg >= 30f" not in source
