from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from babyworld_lite.hoidini_intermimic.stage12 import (
    ContractError,
    bind_shared_scene,
    load_task_contract,
    resolve_scene_manifest,
    validate_task_contract,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "configs" / "hoidini_intermimic_pilot.json"
JUNO_QUALIFICATION = ROOT / "configs" / "hoidini_intermimic_juno_qualification.json"
JUNO_RECORD = ROOT / "docs" / "hoidini_intermimic_juno_qualification.json"
JUNO_PREPARE = ROOT / "scripts" / "juno_prepare_embodiedgen.sh"
JUNO_BIND = ROOT / "scripts" / "juno_bind_hoidini_intermimic_shared_scene.sh"
RESPONSES_PATCH = ROOT / "scripts" / "embodiedgen_v2.0.1_responses_api.patch"
STAGE12 = ROOT / "babyworld_lite" / "hoidini_intermimic" / "stage12.py"
BINDING_RECORD = ROOT / "docs" / "hoidini_intermimic_shared_scene_binding.json"
FIXTURE = ROOT / "tests" / "fixtures" / "hoidini_intermimic"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_shared_scene(tmp_path: Path) -> tuple[Path, Path, Path]:
    scene_root = tmp_path / "canonical_scene"
    shutil.copytree(FIXTURE, scene_root)
    (scene_root / "background" / "gs_model.ply").unlink()
    wall_paths = []
    for name in ("wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max"):
        wall_path = scene_root / "background" / "collision" / f"{name}.obj"
        wall_path.parent.mkdir(parents=True, exist_ok=True)
        wall_path.write_text("v 0 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n", encoding="utf-8")
        wall_paths.append(wall_path)
    floor_collision = {
        "collision_id": "floor",
        "role": "floor",
        "frame": "world",
        "geometry_type": "plane",
        "normal": [0.0, 0.0, 1.0],
        "z_m": 0.0,
    }
    bundle_wall_collisions = [
        {
            "collision_id": path.stem,
            "role": "walls",
            "frame": "world",
            "geometry_type": "mesh",
            "units": "m",
            "mesh": {
                "path": path.relative_to(scene_root).as_posix(),
                "sha256": _sha256(path),
                "format": "obj",
                "scale_xyz": [1.0, 1.0, 1.0],
                "finite_vertices": True,
                "vertex_count": 3,
                "face_count": 1,
            },
        }
        for path in wall_paths
    ]
    geometry_wall_collisions = [
        {
            "collision_id": path.stem,
            "role": "walls",
            "frame": "world",
            "geometry_type": "mesh",
            "units": "m",
            "path": path.relative_to(scene_root / "background").as_posix(),
            "scale_xyz": [1.0, 1.0, 1.0],
        }
        for path in wall_paths
    ]
    _write_json(
        scene_root / "background" / "environment_geometry.json",
        {
            "schema": "InteractMoveEnvironmentGeometry",
            "schema_version": 1,
            "collision_geometry": [floor_collision, *geometry_wall_collisions],
        },
    )

    task = json.loads(TASK.read_text(encoding="utf-8"))
    activity = {
        "schema": "InteractMoveInterMimicActivitySpec",
        "schema_version": 1,
        "protocol_id": "interactmove_intermimic",
        "activity_id": task["task_id"],
        "prompt": task["activity_prompt"],
        "actor": {
            "body_model": "SMPL-X",
            "morphology": "adult",
            "gender": "neutral",
            "betas": [0.0] * 10,
            "initial_stance": "standing_neutral",
            "initial_placement": {
                "method": "target_relative",
                "target_relative_position_m": [-1.0, 0.0, 0.0],
                "face_target": True,
                "selected_by": "nursery",
            },
        },
        "hand_use": task["hand_use"],
        "relations": {
            "start": [{"subject": "target", "predicate": "on", "object_description": "the table"}],
            "final": [{"subject": "target", "predicate": "on", "object_description": "the table"}],
        },
        "timing": {
            "duration_s": 10.0,
            "fps": 30,
            "frame_count": 300,
            "sampling": "half_open_[0,duration)",
        },
        "seeds": task["seeds"],
        "phases": [
            {
                "id": phase["id"],
                "description": phase["intent"],
                "start_frame": phase["start_frame"],
                "end_frame_exclusive": phase["end_frame_exclusive"],
                "expected_target_contact_hands": phase["expected_target_contact_hands"],
            }
            for phase in task["action_phases"]
        ],
        "success_criteria": task["acceptance_metrics"],
    }
    activity_path = scene_root / "metadata" / "activity.json"
    _write_json(activity_path, activity)

    layout_path = scene_root / "layout.json"
    generation_receipt = {
        "schema": "InteractMoveInterMimicFreshGenerationReceipt",
        "schema_version": 1,
        "status": "passed",
        "prompt": task["activity_prompt"],
        "seeds": {
            "image": task["seeds"]["embodiedgen_image"],
            "asset": task["seeds"]["embodiedgen_asset"],
            "layout": task["seeds"]["embodiedgen_layout"],
        },
        "models": {
            "image_to_3d_backend": "SAM3D",
            "sam3d": {
                "source_commit": task["shared_scene_source"]["object_asset_backend"]["source_commit"],
                "checkpoint_revision": task["shared_scene_source"]["object_asset_backend"]["checkpoint_revision"],
            },
        },
        "robot_policy": {
            "render_insert_robot": False,
            "robot_actor_loaded": False,
            "upstream_sim_cli_invoked": False,
        },
    }
    receipt_path = scene_root / "generation_receipt.json"
    _write_json(receipt_path, generation_receipt)

    reference_mesh = scene_root / "background" / "mesh_model.ply"
    target_id = task["scene_bindings"]["target"]["canonical_instance_id"]
    support_id = task["scene_bindings"]["support"]["canonical_instance_id"]
    canonical_id = lambda source: "egv2_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
    background_id = canonical_id("kitchen")
    tray_id = canonical_id("tray")
    bundle = {
        "schema": "InteractMoveInterMimicSceneBundle",
        "schema_version": 1,
        "bundle_id": "scene_fixture0000000000000000000000",
        "scene_id": "egscene_" + _sha256(layout_path)[:24],
        "conventions": {
            "handedness": "right",
            "world_up": "+Z",
            "units": {"length": "m", "mass": "kg", "angle": "rad", "time": "s"},
            "quaternion_order": "xyzw",
            "transform_semantics": "T_parent_child",
            "gravity_m_s2": [0.0, 0.0, -9.81],
        },
        "instances": [
            {
                "source_node_key": "red mug",
                "instance_id": target_id,
                "role": "target",
                "native_role": "manipulated_objs",
                "category": "red mug",
                "body_type": "dynamic",
                "parent_instance_id": support_id,
                "spatial_relation": "ON",
                "initial_pose_world": {
                    "translation_m": [0.1149, 0.1831, 0.8242],
                    "rotation_xyzw": [0.0, 0.0, 0.27769142224444826, 0.9606703253519678],
                },
            },
            {
                "source_node_key": "table",
                "instance_id": support_id,
                "role": "support",
                "native_role": "context",
                "category": "table",
                "body_type": "static",
                "parent_instance_id": background_id,
                "spatial_relation": "FLOOR",
                "initial_pose_world": {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
            {
                "source_node_key": "tray",
                "instance_id": tray_id,
                "role": "distractor",
                "native_role": "distractor_objs",
                "category": "tray",
                "body_type": "dynamic",
                "parent_instance_id": support_id,
                "spatial_relation": "ON",
                "initial_pose_world": {"translation_m": [0.3, 0.0, 0.81], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
            {
                "source_node_key": "kitchen",
                "instance_id": background_id,
                "role": "background",
                "native_role": "background",
                "category": None,
                "body_type": "static",
                "parent_instance_id": None,
                "spatial_relation": None,
                "initial_pose_world": {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            },
        ],
        "source": {
            "format": "EmbodiedGenV2",
            "upstream_commit": task["shared_scene_source"]["embodiedgen_commit"],
            "layout": {"path": "layout.json", "sha256": _sha256(layout_path)},
            "ignored_native_robot_pose": {
                "source_node_key": "franka",
                "authority": "ignored_not_humanoid_authority",
                "pose": {
                    "translation_m": [0.0, -0.8, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        },
        "target_resolution_receipt": {
            "chosen_source_node_key": "red mug",
            "chosen_instance_id": target_id,
            "source_category": "red mug",
            "requested_category": "red mug",
            "category_exact_match": True,
            "layout_sha256": _sha256(layout_path),
            "method": "sole_manipulated_instance",
        },
        "environment": {
            "background_instance_id": background_id,
            "geometry_manifest": "background/environment_geometry.json",
            "reference_mesh_report": {
                "path": "background/mesh_model.ply",
                "sha256": _sha256(reference_mesh),
                "format": "ply",
                "scale_xyz": [1.0, 1.0, 1.0],
            },
            "collision_geometry": [floor_collision, *bundle_wall_collisions],
        },
        "humanoid_initial_pose": {
            "authority": "stage1_target_relative_placement_not_native_robot_pose",
            "pose_world": {
                "translation_m": [-0.8851, 0.1831, 0.8242],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "capabilities": {
            "complete_scene_ready": True,
            "floor_contact_ready": True,
            "interactmove_scene_input_ready": True,
            "reference_mesh_ready": True,
            "room_collision_ready": True,
            "target_category_exact_match": True,
            "visual_background_ready": True,
            "gaussian_render_ready": False,
            "dynamic_settle_ready": False,
            "physics_material_complete": False,
            "blockers": ["red mug:red_mug:collision_0:missing_restitution"],
        },
        "settling": {"status": "not_run", "receipt": None},
    }
    compact_root = tmp_path / "compact"
    bundle_path = compact_root / "scene_bundle.json"
    _write_json(bundle_path, bundle)

    inventory_rows = []
    for path in sorted(path for path in scene_root.rglob("*") if path.is_file()):
        inventory_rows.append(
            {"path": path.relative_to(scene_root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    inventory = {
        "files": inventory_rows,
        "file_count": len(inventory_rows),
        "total_bytes": sum(row["bytes"] for row in inventory_rows),
        "canonical_sha256": _canonical_sha256(inventory_rows),
    }
    activity_sha = _canonical_sha256(activity)
    handoff = {
        "schema": "NurserySharedEmbodiedGenSceneHandoff",
        "schema_version": 1,
        "status": "passed",
        "canonical_scene_root": str(scene_root),
        "activity": {
            "activity_id": task["task_id"],
            "activity_spec_sha256": activity_sha,
            "duration_s": 10.0,
            "fps": 30,
            "frame_count": 300,
            "prompt": task["activity_prompt"],
            "sampling": "half_open_[0,duration)",
            "seeds": task["seeds"],
        },
        "generation": {
            "accepted_assets_regenerated": False,
            "embodiedgen_commit": task["shared_scene_source"]["embodiedgen_commit"],
            "image_to_3d_backend": "SAM3D",
            "layout_path": str(layout_path),
            "layout_sha256": _sha256(layout_path),
            "native_robot_metadata_only": True,
            "receipt_path": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "robot_actor_inserted": False,
            "sam3d_checkpoint_revision": task["shared_scene_source"]["object_asset_backend"]["checkpoint_revision"],
            "sam3d_source_commit": task["shared_scene_source"]["object_asset_backend"]["source_commit"],
            "upstream_sim_cli_invoked": False,
        },
        "inventory": inventory,
        "resolved_instances": {
            "target": {"source_node_key": "red mug", "instance_id": target_id},
            "support": {"source_node_key": "table", "instance_id": support_id},
        },
        "scene_bundle": {
            "path": str(bundle_path),
            "file_sha256": _sha256(bundle_path),
            "semantic_sha256": _canonical_sha256(bundle),
            "bundle_id": bundle["bundle_id"],
            "validation_status": "passed",
        },
        "coordinate_convention": bundle["conventions"],
        "scientific_boundaries": {
            "interactmove_motion_generation": "not_run",
            "intermimic_execution": "not_run",
            "scene_bundle_validation": "passed",
        },
    }
    handoff_path = compact_root / "handoff.json"
    _write_json(handoff_path, handoff)

    shared = task["shared_scene_source"]
    shared.update(
        {
            "canonical_root": str(scene_root),
            "activity_spec_sha256": activity_sha,
            "handoff_path": str(handoff_path),
            "handoff_sha256": _sha256(handoff_path),
            "generation_receipt_path": str(receipt_path),
            "generation_receipt_sha256": _sha256(receipt_path),
            "layout_path": str(layout_path),
            "layout_sha256": _sha256(layout_path),
            "inventory": {name: inventory[name] for name in ("file_count", "total_bytes", "canonical_sha256")},
            "scene_bundle": handoff["scene_bundle"],
            "reference_mesh_sha256": _sha256(reference_mesh),
            "superseded_scene_root_forbidden": str(tmp_path / ".superseded" / "old"),
        }
    )
    task_path = tmp_path / "task.json"
    _write_json(task_path, task)
    return task_path, handoff_path, scene_root


def _reseal_bundle(task_path: Path, handoff_path: Path, mutate) -> None:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    bundle_path = Path(handoff["scene_bundle"]["path"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    mutate(bundle)
    _write_json(bundle_path, bundle)
    bundle_fields = {
        "file_sha256": _sha256(bundle_path),
        "semantic_sha256": _canonical_sha256(bundle),
    }
    handoff["scene_bundle"].update(bundle_fields)
    _write_json(handoff_path, handoff)
    task["shared_scene_source"]["scene_bundle"].update(bundle_fields)
    task["shared_scene_source"]["handoff_sha256"] = _sha256(handoff_path)
    _write_json(task_path, task)


def test_pilot_contract_is_exact_shared_red_mug_activity() -> None:
    contract = load_task_contract(TASK)
    assert contract["activity_prompt"] == "Pick up the red mug from the table, bring it to your mouth, and place it back on the table."
    assert (contract["duration_seconds"], contract["frame_rate_hz"], contract["frame_count"]) == (10.0, 30, 300)
    assert contract["scene_bindings"]["target"]["source_name"] == "red mug"
    assert contract["scene_bindings"]["goal"]["same_instance_as"] == "support"
    assert contract["scene_bindings"]["goal"]["canonical_instance_id"] == contract["scene_bindings"]["support"]["canonical_instance_id"]
    assert [phase["id"] for phase in contract["action_phases"]] == ["approach", "reach", "grasp-lift", "bring-to-mouth", "return", "release-settle"]
    assert contract["action_phases"][0]["start_frame"] == 0
    assert contract["action_phases"][-1]["end_frame_exclusive"] == 300
    assert set(contract["seeds"]) == {"master", "embodiedgen_image", "embodiedgen_asset", "embodiedgen_layout", "interactmove", "intermimic", "render"}
    assert [row["metric"] for row in contract["acceptance_metrics"]] == ["target_instance_correct", "phase_order_fraction", "intended_contact_fraction", "stable_grasp_fraction", "max_penetration_m", "balance_success", "final_relation_satisfied"]


def test_task_contract_rejects_unacknowledged_or_mismatched_alias() -> None:
    contract = json.loads(TASK.read_text(encoding="utf-8"))
    del contract["scene_bindings"]["goal"]["same_instance_as"]
    with pytest.raises(ContractError, match="reused without"):
        validate_task_contract(contract)
    contract = json.loads(TASK.read_text(encoding="utf-8"))
    contract["scene_bindings"]["goal"]["canonical_instance_id"] = "egv2_aaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ContractError, match="alias does not match"):
        validate_task_contract(contract)


def test_task_contract_rejects_a_phase_gap() -> None:
    contract = json.loads(TASK.read_text(encoding="utf-8"))
    contract["action_phases"][1]["start_frame"] = 61
    with pytest.raises(ContractError, match="frame bounds"):
        validate_task_contract(contract)


def test_task_contract_rejects_unknown_contact_participant() -> None:
    contract = json.loads(TASK.read_text(encoding="utf-8"))
    contract["contact_windows"][0]["participants"][0] = "targte"
    with pytest.raises(ContractError, match="unknown participant"):
        validate_task_contract(contract)


def test_task_contract_rejects_contact_or_superseded_path_drift() -> None:
    contract = json.loads(TASK.read_text(encoding="utf-8"))
    contract["contact_windows"][1]["start_frame"] = 101
    contract["contact_windows"][1]["start_seconds"] = 101 / 30
    with pytest.raises(ContractError, match="right_hand_grasps_mug.start_frame"):
        validate_task_contract(contract)

    contract = json.loads(TASK.read_text(encoding="utf-8"))
    contract["shared_scene_source"]["scene_bundle"]["path"] = "/work/example/.superseded/bundle.json"
    with pytest.raises(ContractError, match="must not refer to a superseded path"):
        validate_task_contract(contract)


def test_resolver_builds_robot_free_manifest_with_alias_and_physics_provenance() -> None:
    manifest = resolve_scene_manifest(TASK, FIXTURE / "layout.json")
    assert manifest["robot_policy"]["resolved_robot_instances"] == 0
    assert manifest["robot_policy"]["stripped_upstream_robot_names"] == ["franka"]
    assert len(manifest["instances"]) == 3
    assert manifest["role_bindings"]["support"] == manifest["role_bindings"]["goal"]
    assert manifest["background"]["collision"]["available"] is False
    assert len(manifest["background"]["visual_references"]) == 2
    instances = {instance["source_name"]: instance for instance in manifest["instances"]}
    assert instances["red mug"]["scientific_roles"] == ["target"]
    assert instances["table"]["scientific_roles"] == ["support", "goal"]
    assert instances["red mug"]["body_mode"] == "dynamic"
    assert instances["table"]["body_mode"] == "static"
    assert instances["red mug"]["physical_properties"]["mass"]["value_kg"] == 0.35
    assert instances["red mug"]["physical_properties"]["friction"]["static"] == 0.6
    assert instances["red mug"]["physical_properties"]["restitution"]["available"] is False
    assert instances["red mug"]["physical_properties"]["inertia"]["source"] == "upstream_template_placeholder"
    assert instances["red mug"]["visual_meshes"][0]["local_transform"]["translation_m"] == [0.01, 0.0, 0.0]
    assert instances["red mug"]["world_pose"]["orientation_xyzw"] == pytest.approx([0.0, 0.0, 0.27769142224444826, 0.9606703253519678])
    assert manifest["initial_human_state"]["world_pose"]["translation_m"] == [-0.8851, 0.1831, 0.8242]


def test_resolver_is_deterministic_and_accepts_mesh_only_background(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    (scene / "background" / "gs_model.ply").unlink()
    first = resolve_scene_manifest(TASK, scene / "layout.json")
    second = resolve_scene_manifest(TASK, scene / "layout.json")
    assert first == second
    assert [row["representation"] for row in first["background"]["visual_references"]] == ["color_mesh"]


def test_resolver_rejects_asset_path_escape(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    layout_path = scene / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["assets"]["red mug"] = "../../outside"
    _write_json(layout_path, layout)
    with pytest.raises(ContractError, match="escapes the scene root"):
        resolve_scene_manifest(TASK, layout_path)


def test_resolver_rejects_missing_declared_collision_mesh(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    (scene / "asset3d" / "red_mug" / "result" / "mesh" / "red_mug_collision.obj").unlink()
    with pytest.raises(ContractError, match="does not exist"):
        resolve_scene_manifest(TASK, scene / "layout.json")


def test_resolver_rejects_relation_mapping_contradiction(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    layout_path = scene / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["relation"]["manipulated_objs"] = ["cup"]
    _write_json(layout_path, layout)
    with pytest.raises(ContractError, match="categories and objs_mapping disagree"):
        resolve_scene_manifest(TASK, layout_path)


def test_resolver_rejects_wrong_initial_support_relation(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    layout_path = scene / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["tree"]["table"] = [["tray", "ON"]]
    layout["tree"]["kitchen"].append(["red mug", "FLOOR"])
    _write_json(layout_path, layout)
    with pytest.raises(ContractError, match="target_starts_on_support"):
        resolve_scene_manifest(TASK, layout_path)


def test_resolver_rejects_miswired_urdf_identity(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    urdf = scene / "asset3d" / "red_mug" / "result" / "red_mug.urdf"
    urdf.write_text(urdf.read_text(encoding="utf-8").replace('robot name="red_mug"', 'robot name="bowl"'), encoding="utf-8")
    with pytest.raises(ContractError, match="identity does not match"):
        resolve_scene_manifest(TASK, scene / "layout.json")


def test_shared_scene_binder_preserves_canonical_ids_and_scientific_boundaries(tmp_path: Path) -> None:
    task_path, handoff_path, scene_root = _synthetic_shared_scene(tmp_path)
    binding = bind_shared_scene(task_path, handoff_path)
    assert binding["status"] == "passed"
    assert len(binding["implementation"]["stage12_sha256"]) == 64
    assert binding["shared_scene"]["canonical_root"] == str(scene_root)
    assert binding["shared_scene"]["source_assets_copied"] is False
    assert binding["shared_scene"]["object_asset_backend"]["name"] == "SAM3D"
    assert binding["scene_manifest"]["role_bindings"]["support"] == binding["scene_manifest"]["role_bindings"]["goal"]
    assert binding["scene_manifest"]["role_bindings"]["target"] == "egv2_028a54da21c1691ae18ea225"
    assert binding["scene_manifest"]["initial_human_state"]["world_pose"]["translation_m"] == [-0.8851, 0.1831, 0.8242]
    assert binding["scene_manifest"]["initial_human_state"]["placement_provenance"]["reference_instance_id"] == "egv2_028a54da21c1691ae18ea225"
    assert binding["scene_manifest"]["background"]["collision"]["available"] is True
    assert binding["physics_readiness"]["dynamic_settle_ready"] is False
    assert binding["stage_status"] == {
        "stage1_activity_contract_complete": True,
        "stage2_shared_scene_binding_complete": True,
        "hoidini_motion_generated": False,
        "intermimic_executed": False,
        "video_rendered": False,
    }


def test_shared_scene_binder_rejects_inventory_drift(tmp_path: Path) -> None:
    task_path, handoff_path, scene_root = _synthetic_shared_scene(tmp_path)
    with (scene_root / "background" / "mesh_model.ply").open("a", encoding="utf-8") as handle:
        handle.write("drift\n")
    with pytest.raises(ContractError, match="byte count drifted"):
        bind_shared_scene(task_path, handoff_path)


def test_shared_scene_binder_rejects_same_size_or_path_inventory_drift(tmp_path: Path) -> None:
    task_path, handoff_path, scene_root = _synthetic_shared_scene(tmp_path)
    mesh = scene_root / "asset3d" / "red_mug" / "result" / "mesh" / "red_mug.obj"
    original = mesh.read_text(encoding="utf-8")
    mesh.write_text(original.replace("-0.04", "-0.03", 1), encoding="utf-8")
    assert mesh.stat().st_size == len(original.encode("utf-8"))
    with pytest.raises(ContractError, match="SHA-256 drifted"):
        bind_shared_scene(task_path, handoff_path)

    task_path, handoff_path, scene_root = _synthetic_shared_scene(tmp_path / "extra")
    (scene_root / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ContractError, match="inventory paths drifted"):
        bind_shared_scene(task_path, handoff_path)

    task_path, handoff_path, scene_root = _synthetic_shared_scene(tmp_path / "symlink")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (scene_root / "escape-link").symlink_to(outside)
    with pytest.raises(ContractError, match="contains a symlink"):
        bind_shared_scene(task_path, handoff_path)


def test_shared_scene_binder_rejects_relation_drift(tmp_path: Path) -> None:
    task_path, handoff_path, _ = _synthetic_shared_scene(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["required_final_relations"][0]["spatial_relation"] = "INSIDE"
    _write_json(task_path, task)
    with pytest.raises(ContractError, match="relations.final.*predicate"):
        bind_shared_scene(task_path, handoff_path)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda bundle: bundle["instances"][0].update({"role": "distractor"}), "red mug role"),
        (lambda bundle: bundle["environment"].update({"background_instance_id": bundle["instances"][0]["instance_id"]}), "background instance ID"),
        (
            lambda bundle: bundle["instances"].append(
                {
                    "source_node_key": "franka",
                    "instance_id": "egv2_" + hashlib.sha256(b"franka").hexdigest()[:24],
                    "role": "robot",
                    "native_role": "robot",
                    "category": "franka",
                    "body_type": "dynamic",
                }
            ),
            "instance source set",
        ),
    ],
)
def test_shared_scene_binder_rejects_bundle_identity_or_robot_drift(tmp_path: Path, mutate, error: str) -> None:
    task_path, handoff_path, _ = _synthetic_shared_scene(tmp_path)
    _reseal_bundle(task_path, handoff_path, mutate)
    with pytest.raises(ContractError, match=error):
        bind_shared_scene(task_path, handoff_path)


def test_cli_separates_diagnostic_inspection_from_stage2_binding(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "prepare_hoidini_intermimic_scene.py"
    rejected = subprocess.run(
        [sys.executable, str(script), "resolve-scene"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "invalid choice" in rejected.stderr

    output = tmp_path / "inspection.json"
    inspected = subprocess.run(
        [
            sys.executable,
            str(script),
            "inspect-layout",
            "--task-contract",
            str(TASK),
            "--layout",
            str(FIXTURE / "layout.json"),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert inspected.returncode == 0, inspected.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "hoidini-intermimic-layout-inspection-1.0.0"
    assert payload["boundary_status"] == "diagnostic_only_not_stage2"


def test_juno_qualification_and_bind_launcher_are_scientifically_bounded() -> None:
    qualification = json.loads(JUNO_QUALIFICATION.read_text(encoding="utf-8"))
    assert qualification["embodiedgen"]["commit"] == "9b333554254af196bace88c1a171a3bf047fa09c"
    assert qualification["embodiedgen"]["openai_model"] == "gpt-5.6-luna"
    assert qualification["embodiedgen"]["openai_api"] == "responses"
    assert qualification["embodiedgen"]["openai_sdk"] == "3.1.0"
    assert qualification["shared_scene"]["independent_generation_allowed"] is False
    assert qualification["task_contract"]["sha256"] == _sha256(TASK)
    assert qualification["stage12_source"]["sha256"] == _sha256(STAGE12)
    bind_script = JUNO_BIND.read_text(encoding="utf-8")
    assert "bind-shared-scene" in bind_script
    assert "unexpected_asset_copy_in_hoidini_namespace" in bind_script
    assert "hoidini_motion_generated" in bind_script
    assert "NURSERY_SOURCE_ROOT" not in bind_script
    assert f"EXPECTED_TASK_SHA256={_sha256(TASK)}" in bind_script
    assert f"EXPECTED_STAGE12_SHA256={_sha256(STAGE12)}" in bind_script
    assert "task_contract_hash_mismatch" in bind_script
    assert "stage12_source_hash_mismatch" in bind_script


def test_juno_prepare_pins_responses_sdk_and_applies_vendor_patch() -> None:
    prepare = JUNO_PREPARE.read_text(encoding="utf-8")
    patch = RESPONSES_PATCH.read_text(encoding="utf-8")
    assert "OPENAI_SDK_VERSION=3.1.0" in prepare
    assert '"$RESPONSES_PATCH_PATH"' in prepare
    assert 'self.client.responses.create(**kwargs)' in patch
    assert '"type": "input_text"' in patch
    assert '"type": "input_image"' in patch
    assert '"store": False' in patch
    assert "_image_to_png_data_url" in patch
    assert '"reasoning": {"effort": "low"}' in patch
    assert "response.output_text" in patch
    assert _sha256(RESPONSES_PATCH) == "11caee81f7f52fe90fcff23a85cbd6da8a5a5bb02c01ab1355bc2d7594518f16"


def test_juno_empirical_record_does_not_overclaim_motion_or_physics() -> None:
    record = json.loads(JUNO_RECORD.read_text(encoding="utf-8"))
    binding = json.loads(BINDING_RECORD.read_text(encoding="utf-8"))
    assert record["source"]["task_contract_sha256"] == _sha256(TASK)
    assert record["source"]["stage12_source_sha256"] == _sha256(STAGE12)
    assert record["source"]["binding_receipt_sha256"] == _sha256(BINDING_RECORD)
    assert binding["task"]["contract_sha256"] == _sha256(TASK)
    assert binding["implementation"]["stage12_sha256"] == _sha256(STAGE12)
    assert record["scientific_scope"]["physics_validated"] is False
    assert record["scientific_scope"]["reachability_checked"] is False
    assert record["scientific_scope"]["hoidini_executed"] is False
    assert record["scientific_scope"]["intermimic_executed"] is False
    assert record["scientific_scope"]["video_rendered"] is False
