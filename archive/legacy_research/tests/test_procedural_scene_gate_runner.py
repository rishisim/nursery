import json
import shutil
from pathlib import Path

import pytest

from babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate.runner import (
    AVATAR_SHA256,
    EDITOR_SOURCE_FILES,
    PYTHON_SOURCE_FILES,
    SHADER_SOURCE_FILES,
    STAGE_ORDER,
    _compare_object_traces,
    _actual_run_summary,
    _require_backend_environment_contract,
    _require_stage_prerequisites,
    audit_canonical_sources,
    contract_by_episode_id,
    discover_public_asset_project,
    discover_unity_editor,
)


def test_verified_local_unity_and_public_assets_are_discoverable():
    unity = discover_unity_editor()
    assert unity.is_file()
    project = discover_public_asset_project()
    assert project.joinpath("Assets/Avatar/child.fbx").is_file()
    assert len(AVATAR_SHA256) == 64


def test_runner_uses_one_canonical_unity_orchestrator():
    assert EDITOR_SOURCE_FILES.count("ProceduralSceneGateBuilder.cs") == 1
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/ProceduralSceneGateBuilder.cs"
    ).read_text()
    assert "Physics.Simulate(Dt)" in source
    assert "CaptureFrozenFrame(context)" in source
    assert source.index("Physics.Simulate(Dt)") < source.index("CaptureFrozenFrame(context)")
    assert "object_pose_writes_after_initialization = context.AuthorityAudit.targetPoseWriteCounter" in source
    assert "source_audit_sha256 = context.AuthorityAudit.sourceAuditSha256" in source
    assert "independent_render_timeline = false" in source


def test_contract_lookup_rejects_nonfrozen_episode():
    contract = contract_by_episode_id("A_playroom_red_toy")
    assert contract["scene_spec"]["room_family"] == "warm_playroom"
    with pytest.raises(ValueError):
        contract_by_episode_id("ad_hoc_seed_specific_episode")


def test_cli_exposes_one_entry_point_for_ordered_stage_runs():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/cli.py"
    ).read_text()
    assert "run-episode" in source
    assert tuple(STAGE_ORDER) == (
        "microcell", "clipping", "motion_camera", "polished_cell", "integrated",
        "robustness", "replay", "rerender", "qa",
    )
    assert "run-robustness" in source
    assert "replay" in source
    assert "rerender" in source


def test_source_audit_is_hashed_and_fail_closed_on_target_assistance(tmp_path):
    module_root = Path("babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate")
    for name in EDITOR_SOURCE_FILES + SHADER_SOURCE_FILES + PYTHON_SOURCE_FILES:
        shutil.copy2(module_root / name, tmp_path / name)
    receipt = audit_canonical_sources(tmp_path)
    assert receipt["passed"]
    assert len(receipt["receipt_sha256"]) == 64
    builder = tmp_path / "ProceduralSceneGateBuilder.cs"
    builder.write_text(
        builder.read_text().replace(
            "Physics.Simulate(Dt);",
            "context.TargetBody.AddForce(Vector3.up);\n                Physics.Simulate(Dt);",
        )
    )
    with pytest.raises(RuntimeError, match="forbidden target assistance"):
        audit_canonical_sources(tmp_path)


def test_runner_passes_frozen_cell_identity_and_source_audit_to_unity():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/runner.py"
    ).read_text()
    for name in (
        "PROCEDURAL_GATE_CELL_ID", "PROCEDURAL_GATE_TARGET_ID", "PROCEDURAL_GATE_DESTINATION_ID",
        "PROCEDURAL_GATE_CONTACT_STRATEGY", "PROCEDURAL_GATE_FINAL_GAZE_ZONE",
        "PROCEDURAL_GATE_SOURCE_AUDIT_SHA256",
    ):
        assert name in source


def test_unity_consumes_compiled_contract_as_runtime_authority_not_env_labels():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/ProceduralSceneGateBuilder.cs"
    ).read_text()
    assert "LoadAndVerifyCompiledContract()" in source
    assert "PROCEDURAL_GATE_CONTRACT_FILE_SHA256" in source
    assert "PROCEDURAL_GATE_CONTRACT_SHA256" in source
    assert "RequireExactEnvironment(\"PROCEDURAL_GATE_EPISODE_ID\", contract.episode_id)" in source
    assert "EpisodeId = contract.episode_id" in source
    assert "TargetMidpointReachM = contract.scene_spec.reachability.compiled_requested_midpoint_m" in source
    assert "PopulateActivityPhases(context, contract.activity_plan.phases)" in source
    assert "ApplyFrozenRobustnessVariant(context, contract.robustness_variants)" in source
    assert "context.TargetMassKg *= variant.mass_scale" in source


def test_backend_guard_rejects_label_only_environment_theater(tmp_path):
    builder = tmp_path / "ProceduralSceneGateBuilder.cs"
    builder.write_text(
        "// PROCEDURAL_GATE_CONTRACT LoadAndVerifyCompiledContract()\n"
        "// BuildAuthoritativeContext(contract, output, stage) PROCEDURAL_GATE_CONTRACT_FILE_SHA256\n"
        '// RequireExactEnvironment("PROCEDURAL_GATE_EPISODE_ID"\n'
    )
    with pytest.raises(RuntimeError, match="semantically consume"):
        _require_backend_environment_contract("PROCEDURAL_GATE_CONTRACT", tmp_path)
    _require_backend_environment_contract(
        "PROCEDURAL_GATE_CONTRACT",
        Path("babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate"),
    )


def _trace_row(step, x, velocity_x=0.0):
    return {
        "clock": {"physics_step": step},
        "objects": [
            {
                "persistent_id": "red_toy_001",
                "pose": {
                    "position_world_m": {"x": x, "y": 0.1, "z": -0.2},
                    "rotation_world_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "linear_velocity_world_m_s": {"x": velocity_x, "y": 0.0, "z": 0.0},
            }
        ],
    }


def test_fresh_replay_metrics_are_computed_from_actual_trace_rows(tmp_path):
    source = tmp_path / "source.jsonl"
    replay = tmp_path / "replay.jsonl"
    source.write_text("\n".join(json.dumps(_trace_row(step, 0.1 * step)) for step in range(2)) + "\n")
    replay.write_text(
        "\n".join(json.dumps(_trace_row(step, 0.1 * step + 0.00025, 0.001)) for step in range(2)) + "\n"
    )
    metrics = _compare_object_traces(source, replay)
    assert metrics["rows"] == 2
    assert metrics["translation_max_m"] == pytest.approx(0.00025)
    assert metrics["rotation_max_deg"] == pytest.approx(0.0)
    assert metrics["object_velocity_max_m_s"] == pytest.approx(0.001)

    replay.write_text(json.dumps(_trace_row(9, 0.0)) + "\n")
    with pytest.raises(ValueError, match="clock is malformed|different row counts"):
        _compare_object_traces(source, replay)


def test_rerender_is_explicitly_nonphysics_and_excluded_from_manipulation_evidence():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/ProceduralSceneGateBuilder.cs"
    ).read_text()
    branch = source[source.index('if (stage == "rerender")'):source.index('if ((stage == "clipping"')]
    assert "RunRenderOnlyReplay" in branch
    assert "Physics.Simulate" not in branch
    assert "truth.Bind" not in branch
    assert "eligible_manipulation_evidence = !renderOnly" in source
    assert "trace-driven pose writes are forbidden in physics execution" in source


def test_replay_path_labels_never_become_robustness_variants():
    runner = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/runner.py"
    ).read_text()
    builder = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/ProceduralSceneGateBuilder.cs"
    ).read_text()
    assert 'if variant and stage == "robustness":' in runner
    assert 'environment["PROCEDURAL_GATE_VARIANT"] = variant' in runner
    assert 'RobustnessVariant = stage == "robustness"' in builder
    assert ': "nominal"' in builder


def test_builder_populates_scene_contract_and_registration_controls_exit_status():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/ProceduralSceneGateBuilder.cs"
    ).read_text()
    for marker in (
        "SceneEnvelopeM = new Vector3(",
        "SceneMaterialVariant = contract.scene_spec.material_variant",
        "SceneZoneIds = contract.scene_spec.zones.ToArray()",
        "ExpectedSceneAssetIds = contract.scene_spec.instances.Select(value => value.asset_id).ToArray()",
        "ExpectedSceneInstances = contract.scene_spec.instances.Select(value => new SceneInstanceAuthority",
        "ExpectedSupportRelations = contract.scene_spec.support_relations.Select(value => new SceneSupportAuthority",
        "SceneStabilizationSeconds = contract.scene_spec.stabilization_s",
        "MinimumContextualObjects = contract.scene_spec.minimum_contextual_objects",
        "ValidateAuthorityAndTolerances(contract)",
    ):
        assert marker in source
    assert "bool registrationPassed = RegistrationReportPassed(registrationPath)" in source
    assert "AuthorityAuditPassed(context) && registrationPassed" in source
    assert 'receipt.schema == "embodied.embodiment_registration.v2"' in source
    assert "receipt.self_clearance_sampled_every_physics_step" in source


def test_rerender_restores_dynamic_finger_bodies_with_bind_offsets_and_body_truth():
    source = Path(
        "babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate/ProceduralSceneGateBuilder.cs"
    ).read_text()
    assert 'string bodyKey = key + "_segment" + (index + 1)' in source
    assert "PrepareReplayFingerBodyBindings(context)" in source
    assert "bone.InverseTransformPoint(pair.Value.position)" in source
    assert "Quaternion.Inverse(bone.rotation) * pair.Value.rotation" in source
    assert "requires the measured nonzero bone/body offset" in source
    assert "digit.dynamic_body_states == null || digit.dynamic_body_states.Length != 3" in source
    assert "ApplyReplayFingerBody(binding, digit.dynamic_body_states[index])" in source
    finger_body = source[source.index("private static void ApplyReplayFingerBody"):]
    assert "state.body_id != StableTransformId(binding.body.transform)" in finger_body
    assert "binding.body.position = state.pose.position_world_m" in finger_body
    assert "binding.body.rotation = state.pose.rotation_world_xyzw" in finger_body
    assert "binding.body.linearVelocity = state.linear_velocity_world_m_s" in finger_body
    assert "state.pose == null" in finger_body
    rerender_loop = source[source.index("private static void RunRenderOnlyReplay"):source.index("private static void ApplyRenderOnlyReplayState")]
    assert rerender_loop.index("embodiment.UpdateRegisteredCollidersBeforePhysics(context)") \
        < rerender_loop.index("embodiment.SampleRegistrationAtPhysicsStep(context)")


def test_source_audit_hashes_python_and_keeps_rerender_writes_method_scoped(tmp_path):
    module_root = Path("babyworld_lite/childlens_engine_bakeoff/procedural_scene_gate")
    for name in EDITOR_SOURCE_FILES + SHADER_SOURCE_FILES + PYTHON_SOURCE_FILES:
        shutil.copy2(module_root / name, tmp_path / name)
    receipt = audit_canonical_sources(tmp_path)
    assert set(PYTHON_SOURCE_FILES) <= set(receipt["source_sha256"])
    builder = tmp_path / "ProceduralSceneGateBuilder.cs"
    builder.write_text(
        builder.read_text().replace(
            "Physics.gravity = new Vector3(0f, -9.81f, 0f);",
            "body.position = pose.position_world_m;\n            Physics.gravity = new Vector3(0f, -9.81f, 0f);",
        )
    )
    with pytest.raises(RuntimeError, match="forbidden target assistance"):
        audit_canonical_sources(tmp_path)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_completed_evidence(run_root):
    _write_json(run_root / "authority_receipt.json", {
        "runtime_accounting_passed": True, "assistance_ledger_entries": 0,
    })
    _write_json(run_root / "episode_trace_manifest.json", {"trace_complete": True})
    _write_json(run_root / "scene_compiler_validation.json", {
        "target_unparented": True,
        "target_non_kinematic": True,
        "all_visible_context_physics_backed": True,
        "camera_aware_target_and_destination_sightlines": True,
        "reachable_elements_have_physx_colliders": True,
        "deterministic_prospective_rejection_only": True,
        "contextual_visible_object_count": 11,
    })
    _write_json(run_root / "scene_spec.json", {
        "catalog_id": "canonical_catalog", "no_visible_primitive_furniture": True,
    })
    _write_json(run_root / "interaction_summary.json", {
        "schema": "embodied.physics_truth.interaction_summary.v1",
        "right_measured_impulse_qualified": True,
        "left_measured_impulse_qualified": True,
        "lift_over_0_10_m": True,
        "turn_over_30_deg": True,
        "release_at_required_destination": True,
        "object_unsupported_during_qualified_manipulation": True,
        "finger_object_penetration_passed": True,
        "target_support_penetration_passed": True,
    })
    _write_json(run_root / "registration_report.json", {
        "schema": "embodied.embodiment_registration.v2",
        "passed": True,
        "self_clearance_sampled_every_physics_step": True,
        "non_adjacent_anatomy_clearance_passed": True,
    })
    _write_json(run_root / "registered_capture_manifest.json", {
        "schema": "embodied.registered_capture_manifest.v1",
        "frames_captured": 720,
        "contiguous_frames": True,
        "exact_integer_clock": True,
        "all_modalities_state_invariant": True,
        "hero_contains_proxy_pixels": False,
    })
    _write_json(run_root / "contact_projection.json", {
        "schema": "embodied.registered_contact_projection.v1",
        "records": [{
            "contact_projects_to_expected_visible_surface": True,
            "contact_visible_in_registered_frame": True,
        }],
    })
    _write_json(run_root / "independent_qa_report.json", {
        "schema": "embodied.independent_qa_report.v1",
        "qa_decision": "PROMOTION_VETO",
        "promotion_veto": True,
        "gate_summary": {**{gate: "PASS" for gate in "ABCD"}, "E": "FAIL", "F": "UNAVAILABLE"},
    })
    final_row = {
        "force_bearing_qualification": {
            "right_measured_impulse_qualified": True,
            "left_measured_impulse_qualified": True,
            "lift_over_0_10_m": True,
            "turn_over_30_deg": True,
        }
    }
    (run_root / "episode_trace.jsonl").write_text(json.dumps(final_row) + "\n")


def test_episode_completion_requires_all_actual_physical_capture_and_qa_evidence(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_completed_evidence(run_root)
    receipt = {
        "episode_id": "A_playroom_red_toy",
        "run_root": str(run_root),
        "returncode": 0,
        "stage": "integrated",
    }
    summary = _actual_run_summary(receipt)
    assert summary["completed"] is True
    assert summary["interaction_passed"] is True
    assert summary["registration_passed"] is True
    assert summary["contact_projection_passed"] is True
    assert summary["independent_qa_passed"] is True
    assert summary["automated_visual_review_claimed"] is False

    interaction = json.loads((run_root / "interaction_summary.json").read_text())
    interaction["release_at_required_destination"] = False
    _write_json(run_root / "interaction_summary.json", interaction)
    assert _actual_run_summary(receipt)["completed"] is False


def test_ordered_stage_prerequisites_require_cumulative_gate_pass_receipts(tmp_path):
    episode_id = "A_playroom_red_toy"
    contract_sha = contract_by_episode_id(episode_id)["contract_sha256"]
    source_audit_sha = audit_canonical_sources()["receipt_sha256"]
    with pytest.raises(RuntimeError, match="ordered gate prerequisite"):
        _require_stage_prerequisites(episode_id, tmp_path, "clipping")
    a_root = tmp_path / "episodes" / episode_id / "A_microcell"
    _write_json(a_root / "independent_qa_report.json", {
        "schema": "embodied.independent_qa_report.v1",
        "run_root": str(a_root),
        "contract_provenance": {"executed_episode_contract_sha256": contract_sha},
        "gate_summary": {"A": "PASS", "B": "UNAVAILABLE"},
    })
    _write_json(a_root / "execution_receipt.json", {
        "schema": "embodied.unity_episode_execution.v1",
        "episode_id": episode_id,
        "stage": "microcell",
        "returncode": 0,
        "run_root": str(a_root),
        "contract_sha256": contract_sha,
        "source_audit_sha256": "0" * 64,
    })
    with pytest.raises(RuntimeError, match="ordered gate prerequisite"):
        _require_stage_prerequisites(episode_id, tmp_path, "clipping")
    a_execution = json.loads((a_root / "execution_receipt.json").read_text())
    a_execution["source_audit_sha256"] = source_audit_sha
    _write_json(a_root / "execution_receipt.json", a_execution)
    _require_stage_prerequisites(episode_id, tmp_path, "clipping")
    with pytest.raises(RuntimeError, match="gate B"):
        _require_stage_prerequisites(episode_id, tmp_path, "motion_camera")
    b_root = tmp_path / "episodes" / episode_id / "B_clipping"
    _write_json(b_root / "independent_qa_report.json", {
        "schema": "embodied.independent_qa_report.v1",
        "run_root": str(b_root),
        "contract_provenance": {"executed_episode_contract_sha256": contract_sha},
        "gate_summary": {"A": "FAIL", "B": "PASS"},
    })
    _write_json(b_root / "execution_receipt.json", {
        "schema": "embodied.unity_episode_execution.v1",
        "episode_id": episode_id,
        "stage": "clipping",
        "returncode": 0,
        "run_root": str(b_root),
        "contract_sha256": contract_sha,
        "source_audit_sha256": source_audit_sha,
    })
    _require_stage_prerequisites(episode_id, tmp_path, "motion_camera")
