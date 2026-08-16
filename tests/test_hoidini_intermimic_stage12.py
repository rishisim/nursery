from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from babyworld_lite.hoidini_intermimic.stage12 import (
    ContractError,
    load_task_contract,
    resolve_scene_manifest,
    validate_task_contract,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "configs" / "hoidini_intermimic_pilot.json"
JUNO_QUALIFICATION = ROOT / "configs" / "hoidini_intermimic_juno_qualification.json"
JUNO_RECORD = ROOT / "docs" / "hoidini_intermimic_juno_qualification.json"
FIXTURE = ROOT / "tests" / "fixtures" / "hoidini_intermimic"


def test_pilot_contract_is_explicit_and_covers_ten_seconds() -> None:
    contract = load_task_contract(TASK)

    assert contract["activity_prompt"] == (
        "Pick up the mug from the table, carry it to the tray, and put it down."
    )
    assert contract["duration_seconds"] == 10.0
    assert contract["scene_bindings"]["target"]["source_name"] == "mug"
    assert contract["scene_bindings"]["support"]["source_name"] == "table"
    assert contract["scene_bindings"]["goal"]["source_name"] == "tray"
    assert contract["action_phases"][0]["start_seconds"] == 0.0
    assert contract["action_phases"][-1]["end_seconds"] == 10.0
    assert set(contract["seeds"]) == {"scene", "layout", "motion", "physics", "render"}


def test_juno_qualification_is_pinned_and_scientifically_bounded() -> None:
    qualification = json.loads(JUNO_QUALIFICATION.read_text(encoding="utf-8"))

    assert qualification["embodiedgen"]["release"] == "v2.0.1"
    assert qualification["embodiedgen"]["commit"] == (
        "9b333554254af196bace88c1a171a3bf047fa09c"
    )
    assert qualification["juno"]["durable_root"] == (
        "/work/dal503972/hoidini_intermimic"
    )
    assert qualification["juno"]["scratch_root"] == (
        "/scratch/juno/dal503972/hoidini_intermimic"
    )
    assert "A qualification job is not scene generation." in qualification[
        "scientific_exclusions"
    ]


def test_juno_empirical_record_does_not_overclaim_stage2() -> None:
    record = json.loads(JUNO_RECORD.read_text(encoding="utf-8"))

    assert record["decision"] in {
        "QUALIFIED_STAGE12_BOUNDARY",
        "REJECTED_STAGE12_BOUNDARY",
        "INCONCLUSIVE_ENVIRONMENT",
    }
    if record["decision"] != "QUALIFIED_STAGE12_BOUNDARY":
        assert record["real_scene_generation"]["executed"] is False
        assert record["stage2_empirical_gate_complete"] is False
    assert record["scientific_scope"]["physics_validated"] is False
    assert record["scientific_scope"]["reachability_checked"] is False
    assert record["scientific_scope"]["hoidini_executed"] is False
    assert record["scientific_scope"]["intermimic_executed"] is False


def test_task_contract_rejects_a_phase_gap() -> None:
    contract = json.loads(TASK.read_text(encoding="utf-8"))
    contract["action_phases"][1]["start_seconds"] = 1.1

    with pytest.raises(ContractError, match="contiguous"):
        validate_task_contract(contract)


def test_task_contract_rejects_unknown_contact_participant() -> None:
    contract = json.loads(TASK.read_text(encoding="utf-8"))
    contract["contact_windows"][0]["participants"][0] = "targte"

    with pytest.raises(ContractError, match="unknown participant"):
        validate_task_contract(contract)


def test_resolver_builds_robot_free_manifest_with_physics_provenance() -> None:
    manifest = resolve_scene_manifest(TASK, FIXTURE / "layout.json")

    assert manifest["robot_policy"]["resolved_robot_instances"] == 0
    assert manifest["robot_policy"]["stripped_upstream_robot_names"] == ["franka"]
    assert len(manifest["instances"]) == 3
    assert all(instance["source_name"] != "franka" for instance in manifest["instances"])
    assert set(manifest["role_bindings"]) == {"target", "support", "goal"}
    assert manifest["background"]["collision"]["available"] is False
    assert len(manifest["background"]["visual_references"]) == 2

    instances = {instance["source_name"]: instance for instance in manifest["instances"]}
    assert instances["mug"]["body_mode"] == "dynamic"
    assert instances["table"]["body_mode"] == "static"
    assert instances["tray"]["body_mode"] == "static"
    assert instances["mug"]["physical_properties"]["mass"]["value_kg"] == 0.35
    assert instances["mug"]["physical_properties"]["mass"]["source"] == (
        "embodiedgen_vlm_estimate_midpoint_in_urdf"
    )
    assert instances["mug"]["physical_properties"]["mass"]["upstream_estimate_range_kg"] == [0.3, 0.4]
    assert instances["mug"]["physical_properties"]["friction"]["static"] == 0.6
    assert instances["mug"]["physical_properties"]["restitution"] == {
        "available": False,
        "value": None,
        "source": "missing_not_embodiedgen_default",
    }
    mug_inertia = instances["mug"]["physical_properties"]["inertia"]
    assert mug_inertia["available"] is False
    assert mug_inertia["source"] == "upstream_template_placeholder"
    assert mug_inertia["raw_upstream_matrix_kg_m2"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert instances["table"]["physical_properties"]["inertia"]["available"] is True
    assert instances["tray"]["physical_properties"]["inertia"]["available"] is False
    assert instances["mug"]["visual_meshes"][0]["local_transform"]["translation_m"] == [0.01, 0.0, 0.0]
    assert instances["mug"]["world_pose"]["orientation_xyzw"] == pytest.approx(
        [0.0, 0.0, 2 ** -0.5, 2 ** -0.5]
    )
    assert {
        relation["relation_id"] for relation in manifest["required_initial_relations"]
    } == {"target_starts_on_support", "goal_starts_on_support"}
    assert manifest["initial_human_state"]["world_pose"]["translation_m"] == [0.0, -1.0, 0.0]
    assert manifest["initial_human_state"]["placement_provenance"]["reachability_checked"] is False


def test_resolver_is_deterministic() -> None:
    first = resolve_scene_manifest(TASK, FIXTURE / "layout.json")
    second = resolve_scene_manifest(TASK, FIXTURE / "layout.json")
    assert first == second


def test_resolver_rejects_asset_path_escape(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    layout_path = scene / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["assets"]["mug"] = "../../outside"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    with pytest.raises(ContractError, match="escapes the scene root"):
        resolve_scene_manifest(TASK, layout_path)


def test_resolver_rejects_missing_declared_collision_mesh(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    missing = scene / "asset3d" / "mug" / "result" / "mesh" / "mug_collision.obj"
    missing.unlink()

    with pytest.raises(ContractError, match="does not exist"):
        resolve_scene_manifest(TASK, scene / "layout.json")


def test_resolver_rejects_relation_mapping_contradiction(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    layout_path = scene / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["relation"]["manipulated_objs"] = ["cup"]
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    with pytest.raises(ContractError, match="categories and objs_mapping disagree"):
        resolve_scene_manifest(TASK, layout_path)


def test_resolver_rejects_wrong_initial_support_relation(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    layout_path = scene / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["tree"]["table"] = [["tray", "ON"]]
    layout["tree"]["kitchen"].append(["mug", "FLOOR"])
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    with pytest.raises(ContractError, match="target_starts_on_support"):
        resolve_scene_manifest(TASK, layout_path)


def test_resolver_rejects_miswired_urdf_identity(tmp_path: Path) -> None:
    scene = tmp_path / "scene"
    shutil.copytree(FIXTURE, scene)
    urdf = scene / "asset3d" / "mug" / "result" / "mug.urdf"
    urdf.write_text(urdf.read_text(encoding="utf-8").replace('robot name="mug"', 'robot name="bowl"'), encoding="utf-8")

    with pytest.raises(ContractError, match="identity does not match"):
        resolve_scene_manifest(TASK, scene / "layout.json")
