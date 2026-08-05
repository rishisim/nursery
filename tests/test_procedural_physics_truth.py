from pathlib import Path


SOURCE = (
    Path(__file__).parents[1]
    / "babyworld_lite"
    / "childlens_engine_bakeoff"
    / "procedural_scene_gate"
    / "PhysicsTruthRecorder.cs"
)


def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


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
    assert "free object disabled for this object-free qualification stage" in source
    assert "release_commanded" in source


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
