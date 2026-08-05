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


def test_clearance_uses_rendered_geometry_and_near_plane_not_a_counter():
    assert "CalculateFrustumCorners" in SOURCE
    assert "PointTriangleSquaredDistance" in SOURCE
    assert "ContainsByOddEvenRay" in SOURCE
    assert "minimum_surface_distance_m" in SOURCE
    assert "inside_body_mesh" in SOURCE
    assert "MinimumClearanceM = 0.001f" in SOURCE


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
