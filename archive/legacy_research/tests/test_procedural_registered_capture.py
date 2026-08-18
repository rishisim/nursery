from pathlib import Path


ROOT = Path("babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate")
SOURCE = (ROOT / "RegisteredCapture.cs").read_text(encoding="utf-8")
DEPTH = (ROOT / "MetricDepth.shader").read_text(encoding="utf-8")
LABELS = (ROOT / "SemanticInstance.shader").read_text(encoding="utf-8")


def test_registered_capture_implements_frozen_interface_and_exact_clock():
    assert "public sealed class RegisteredCapture : IRegisteredCaptureModule" in SOURCE
    assert "context.PhysicsStep + 1 == (context.RenderFrame + 1) * FrozenGate.StepsPerFrame" in SOURCE
    assert "FrozenGate.Width != 1920" in SOURCE
    assert "FrozenGate.Height != 1080" in SOURCE
    assert "FrozenGate.RenderHz != 30" in SOURCE
    assert "FrozenGate.PhysicsHz != 240" in SOURCE
    assert "Physics.Simulate" not in SOURCE


def test_head_camera_is_one_fixed_neutral_head_child_without_target_lock():
    assert "mount.SetParent(context.Head, false)" in SOURCE
    assert "mount.localRotation = Quaternion.identity" in SOURCE
    assert "Vector3.Angle(context.Head.forward, context.HeadCamera.transform.forward)" in SOURCE
    assert "MaximumOpticalAngleDeg = 15f" in SOURCE
    assert "head_mount_fixed_after_bind = true" in SOURCE
    assert "target_lock = false" in SOURCE
    assert "independent_camera_animation = false" in SOURCE
    assert "tabletop_view_source" in SOURCE
    assert "void Update(" not in SOURCE
    assert "void LateUpdate(" not in SOURCE


def test_fov_is_prospectively_auditioned_once_and_narrowest_qualified_is_frozen():
    assert "VerticalFovCandidatesDeg = { 60f, 68f, 75f }" in SOURCE
    assert "ProspectivelyFreezeFieldOfView(context)" in SOURCE
    assert "same_prerun_geometry_sha256" in SOURCE
    assert "same_prerun_geometry_trace_sha256" in SOURCE
    assert "FirstOrDefault(value => value.all_required_events_inspectable)" in SOURCE
    assert "MinimumMilestoneViewportMargin = 0.035f" in SOURCE
    assert "virtual_pose_applied_to_camera = false" in SOURCE
    assert "camera_mount_changed_during_audition = false" in SOURCE
    assert "camera_fov_qualification.json" in SOURCE
    assert "prospective FOV/event-view qualification rejected before capture" in SOURCE
    assert "context.HeadCamera.fieldOfView = fovQualification.selected_vertical_fov_deg" in SOURCE


def test_camera_chain_and_mount_are_single_authority_and_immutable():
    assert "ValidateAuthoritativeCameraChain(context)" in SOURCE
    assert "IsChildOrSelf(context.Neck, context.Torso)" in SOURCE
    assert "IsChildOrSelf(context.Head, context.Neck)" in SOURCE
    assert "context.HeadCameraMount.parent != context.Head" in SOURCE
    assert "frozenHeadMountLocalPosition" in SOURCE
    assert "frozenHeadMountLocalRotation" in SOURCE
    assert "head optical mount changed after its bind-time measured local pose was frozen" in SOURCE


def test_clearance_uses_rendered_geometry_and_near_plane_not_a_counter():
    assert "ProspectivelyFreezeHeadMountClearance(context)" in SOURCE
    assert "float[] forwardClearanceCandidatesM = { 0.032f, 0.045f, 0.060f, 0.080f }" in SOURCE
    assert "hasPreviousClearancePose = false" in SOURCE
    assert "&& !value.isTrigger" in SOURCE
    assert "&& !context.AvatarColliders.Contains(value)" in SOURCE
    assert "CalculateFrustumCorners" in SOURCE
    assert "PointTriangleSquaredDistance" in SOURCE
    assert "ContainsByOddEvenRay" in SOURCE
    assert "minimum_surface_distance_m" in SOURCE
    assert "inside_body_mesh" in SOURCE
    assert "MinimumClearanceM = 0.001f" in SOURCE


def test_camera_clearance_is_swept_and_rejected_before_any_pixels_are_written():
    assert "SweptClearanceIntervals = 8" in SOURCE
    assert "MeasureSweptCameraClearance(context)" in SOURCE
    assert "CameraClearanceProbes(camera, position, rotation)" in SOURCE
    assert "Quaternion.Slerp(startRotation, currentRotation, t)" in SOURCE
    assert "collider.ClosestPoint" in SOURCE
    assert "collider.bounds.SqrDistance(probe)" in SOURCE
    assert "distance = Mathf.Max(distance, boundsLowerBound)" in SOURCE
    assert SOURCE.index("CameraClearanceResult clearance = MeasureSweptCameraClearance(context)") < SOURCE.index("var receipts = new List<CaptureStreamReceipt>()")
    assert "camera swept origin/near plane lacks positive head/hair/garment/furniture clearance before render" in SOURCE
    assert "swept_clearance_samples" in SOURCE
    assert "minimum_scene_collider_distance_m" in SOURCE


def test_modalities_share_state_and_runner_frame_roots():
    for path in (
        '"head", "rgb"',
        '"head", "depth"',
        '"head", "semantic"',
        '"head", "instance"',
        '"external", "clean"',
        '"external", "overlay"',
    ):
        assert path in SOURCE
    assert '"frame_" + context.RenderFrame.ToString("D4"' in SOURCE
    assert "AuthorityStateSha256(context)" in SOURCE
    assert SOURCE.count("AssertFrozenState(context") >= 6
    assert "physics_advanced_between_modalities = false" in SOURCE
    assert 'GetEnvironmentVariable("PROCEDURAL_GATE_CAPTURE_MODE")' in SOURCE
    assert 'value != "all" && value != "qualification" && value != "none"' in SOURCE


def test_intrinsics_extrinsics_and_provenance_are_in_frame_ledger():
    for field in (
        "camera_position_world_m",
        "camera_rotation_world_xyzw",
        "camera_to_world_matrix_column_major",
        "world_to_camera_matrix_column_major",
        "projection_matrix_column_major",
        "intrinsics",
        "camera_state_provenance",
        "intrinsics_provenance",
        "authority_state_sha256",
    ):
        assert field in SOURCE
    assert "capture_frame_ledger.jsonl" in SOURCE
    assert "file_sha256" in SOURCE


def test_metric_depth_is_lossless_uint24_millimetres():
    assert 'Shader "ProceduralSceneGate/MetricDepthUint24Millimetres"' in DEPTH
    assert "-UnityObjectToViewPos(input.vertex).z" in DEPTH
    assert "input.view_depth_m * 1000.0" in DEPTH
    assert "16777215.0" in DEPTH
    assert "R + 256*G + 65536*B" in DEPTH
    assert "RenderTextureReadWrite.Linear" in SOURCE
    assert "TextureFormat.RGB24" in SOURCE


def test_semantic_and_persistent_instance_ids_are_exact_registered_uint24():
    assert 'Shader "ProceduralSceneGate/SemanticInstanceUint24"' in LABELS
    assert "_SemanticId" in LABELS
    assert "_InstanceId" in LABELS
    assert "_RegisteredLabelMode" in LABELS
    assert "16777215.0" in LABELS
    assert "MaterialPropertyBlock" in SOURCE
    assert "ResolvePersistentInstanceId" in SOURCE
    assert "deterministically mapped to collision-free uint24" in SOURCE
    assert "PhysicsTruthObjectIdentity" in SOURCE
    assert "persistent_across_frames = true" in SOURCE
    assert "semantic_instance_manifest.json" in SOURCE


def test_overlay_is_separate_labeled_qa_and_never_in_hero_masks():
    assert "QaOverlayLayer = 31" in SOURCE
    assert SOURCE.count("& ~(1 << QaOverlayLayer)") >= 3
    assert "external_collider_contact_overlay_QA_ONLY" in SOURCE
    assert "qaOverlay.capture_enabled = false" in SOURCE
    assert "qaOverlay.capture_enabled = true" in SOURCE
    assert "Physics.ComputePenetration" in SOURCE
    assert "QA-only Physics.ComputePenetration overlap proxy; not contact truth" in SOURCE
    assert "overlay_is_hero = false" in SOURCE
    assert "hero_contains_proxy_pixels = false" in SOURCE


def test_visibility_inputs_are_explicitly_not_visual_evidence():
    assert "event_visibility_inputs" in SOURCE
    assert "never promoted to visual evidence" in SOURCE
    assert "frame counters, hashes, frustum tests, and overlay proxies are QA inputs" in SOURCE


def test_contact_projection_is_registered_to_first_visible_labeled_surface():
    assert "ProjectToFirstVisibleSurface" in SOURCE
    assert "Physics.RaycastAll" in SOURCE
    assert "contact.colliderA" in SOURCE
    assert "contact.colliderB" in SOURCE
    assert "contact_projects_to_expected_visible_surface" in SOURCE
    assert "contact_visible_in_registered_frame" in SOURCE
    assert "first_visible_semantic_uint24" in SOURCE
    assert "first_visible_persistent_instance_uint24" in SOURCE
    assert "pixel_xy" in SOURCE
    assert "actual semantic/instance uint24 values decoded at the projected contact pixel" in SOURCE


def test_measured_capture_adapters_emit_one_authoritative_projection_schema_without_visual_theater():
    assert 'ContactProjectionFileName = "contact_projection.json"' in SOURCE
    assert 'schema = "embodied.registered_contact_projection.v1"' in SOURCE
    assert "records = contactProjectionRecords.ToArray()" in SOURCE
    assert "physical_contact_point_world_m = input.point_world_m" in SOURCE
    assert "expected_contact_collider_a = input.expected_contact_collider_a" in SOURCE
    assert "source_capture_ledger_sha256 = FileSha256(ledgerPath)" in SOURCE
    assert 'VisualMeasurementsFileName = "visual_measurements.json"' in SOURCE
    assert 'schema = "embodied.registered_visual_measurements.v1"' in SOURCE
    assert "measured_capture_evidence_only = true" in SOURCE
    assert "direct_decoded_frame_review_performed = false" in SOURCE
    assert 'QaEvidenceFileName = "qa_evidence.json"' not in SOURCE


def test_contact_ids_are_sampled_from_actual_frozen_label_pixels_not_renderer_ancestry():
    assert "texture.GetPixel(x, y)" in SOURCE
    assert "input.rendered_semantic_uint24 = uint24" in SOURCE
    assert "input.rendered_persistent_instance_uint24 = uint24" in SOURCE
    assert "FinalizeRenderedLabelSamples(visibility, context, labelBindings)" in SOURCE
    assert "SampledBindingMatchesExpectedContact" in SOURCE
    assert 'StringComparer.Ordinal.Equals(binding.semantic_name, "body_skin")' in SOURCE
    assert "producer booleans and renderer ancestry cannot establish label identity" in SOURCE
    projection_body = SOURCE[SOURCE.index("private static ContactProjectionResult ProjectToFirstVisibleSurface"):SOURCE.index("private static void FinalizeRenderedLabelSamples")]
    assert "ClosestRenderer" not in projection_body
    assert "semantic_uint24 = binding" not in projection_body


def test_fov_preflight_covers_every_required_manipulation_milestone_and_occlusion():
    for event in (
        '"touch"',
        '"capture"',
        '"left_assistance"',
        '"lift"',
        '"turn"',
        '"placement"',
        '"free_release"',
        '"withdrawal"',
    ):
        assert f"MilestoneViewAnchor({event}" in SOURCE
    assert "ProspectivelyMeasureOcclusion" in SOURCE
    assert "first_blocker_id" in SOURCE
    assert "body/support/furniture ray occluder" in SOURCE
