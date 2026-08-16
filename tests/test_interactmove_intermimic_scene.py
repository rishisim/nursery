from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from babyworld_lite.interactmove_intermimic.activity import (
    build_embodiedgen_request,
)
from babyworld_lite.interactmove_intermimic.common import (
    ContractError,
    canonicalize_quaternion_xyzw,
    content_sha256,
)
from babyworld_lite.interactmove_intermimic.embodiedgen import (
    compile_scene_bundle,
    validate_scene_bundle,
)


UPSTREAM_VERSION = "v2.0.1"
UPSTREAM_COMMIT = "9b333554254af196bace88c1a171a3bf047fa09c"
METRICS = (
    "target_instance_correct",
    "phase_order_fraction",
    "intended_contact_fraction",
    "stable_grasp_fraction",
    "max_penetration_m",
    "balance_success",
    "final_relation_satisfied",
)


def _activity_spec() -> dict:
    boolean_metrics = {
        "target_instance_correct",
        "balance_success",
        "final_relation_satisfied",
    }
    return {
        "schema": "InteractMoveInterMimicActivitySpec",
        "schema_version": 1,
        "protocol_id": "interactmove_intermimic",
        "activity_id": "lift-blue-bowl",
        "prompt": "Lift the blue bowl from the table with both hands.",
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
        "target_request": {
            "description": "the blue bowl on the table",
            "category": "bowl",
            "prompt_span": "blue bowl",
            "resolution_rule": {
                "method": "sole_manipulated_instance",
                "expected_source_node_key": None,
            },
        },
        "relations": {
            "start": [
                {
                    "subject": "target",
                    "predicate": "on",
                    "object_description": "the table",
                    "description": "the target starts on the table",
                }
            ],
            "final": [
                {
                    "subject": "target",
                    "predicate": "held_by",
                    "object_description": "the adult actor",
                    "description": "the target finishes held by the actor",
                }
            ],
        },
        "phases": [
            {
                "id": "approach",
                "description": "approach the target",
                "start_frame": 0,
                "end_frame_exclusive": 100,
                "expected_target_contact_hands": [],
            },
            {
                "id": "grasp-lift",
                "description": "grasp and lift the target",
                "start_frame": 100,
                "end_frame_exclusive": 300,
                "expected_target_contact_hands": ["left", "right"],
            },
        ],
        "timing": {
            "duration_s": 10.0,
            "fps": 30,
            "frame_count": 300,
            "sampling": "half_open_[0,duration)",
        },
        "hand_use": {"mode": "two_hand", "hands": ["left", "right"]},
        "seeds": {
            "master": 10,
            "embodiedgen_image": 11,
            "embodiedgen_asset": 12,
            "embodiedgen_layout": 13,
            "interactmove": 14,
            "intermimic": 15,
            "render": 16,
        },
        "success_criteria": [
            {
                "id": f"criterion-{index}",
                "metric": metric,
                "operator": (
                    "eq"
                    if metric in boolean_metrics
                    else "lte"
                    if metric == "max_penetration_m"
                    else "gte"
                ),
                "threshold": (
                    True
                    if metric in boolean_metrics
                    else 0.02
                    if metric == "max_penetration_m"
                    else 0.9
                ),
                "unit": "m" if metric == "max_penetration_m" else None,
                "evaluation_window": "full_episode",
            }
            for index, metric in enumerate(METRICS)
        ],
    }


def _urdf(name: str, *, collision: bool = True) -> str:
    category = "bowl" if name == "blue_bowl" else name
    collision_xml = (
        """
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="mesh/collision.obj" scale="1 1 1"/></geometry>
      <gazebo><mu1>0.70</mu1><mu2>0.55</mu2></gazebo>
    </collision>"""
        if collision
        else ""
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="{name}">
  <link name="{name}">
    <visual>
      <origin xyz="0.01 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="mesh/visual.obj" scale="1 1 1"/></geometry>
    </visual>
    {collision_xml}
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.4"/>
    </inertial>
    <extra_info><category>{category}</category></extra_info>
  </link>
</robot>
"""


def _layout() -> dict:
    return {
        "tree": {
            "nursery": [["table", "FLOOR"], ["franka", "IN"]],
            "table": [["blue bowl", "ON"]],
        },
        "relation": {
            "background": "nursery",
            "context": "table",
            "manipulated_objs": ["blue bowl"],
            "distractor_objs": [],
            "robot": "franka",
        },
        "objs_desc": {
            "nursery": "a small nursery room",
            "table": "a low wooden table",
            "blue bowl": "a blue bowl",
        },
        "objs_mapping": {
            "nursery": "background",
            "table": "context",
            "blue bowl": "manipulated_objs",
        },
        "assets": {
            "nursery": "background",
            "table": "asset3d/table/result",
            "blue bowl": "asset3d/blue_bowl/result",
        },
        "quality": {"blue bowl": "source_canary_not_simulator_validation"},
        "position": {
            "nursery": [0, 0, 0, 0, 0, 0, 1],
            "table": [1, 2, 0, 0, 0, 0, 1],
            "blue bowl": [1, 2, 0.8, 0, 0, 0, 0.9995],
            "franka": [-2, 0, 0, 0, 0, 0, 1],
        },
    }


def _write_canary(tmp_path: Path, *, collision: bool = True) -> Path:
    root = tmp_path / "scene"
    root.mkdir(parents=True)
    (root / "layout.json").write_text(json.dumps(_layout()), encoding="utf-8")
    background = root / "background"
    background.mkdir()
    room_vertices = [
        (-4, -4, 0), (4, -4, 0), (4, 4, 0), (-4, 4, 0),
        (-4, -4, 3), (4, -4, 3), (4, 4, 3), (-4, 4, 3),
    ]
    room_faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    ply = [
        "ply", "format ascii 1.0", "element vertex 8",
        "property float x", "property float y", "property float z",
        "element face 12", "property list uchar int vertex_indices", "end_header",
    ]
    ply.extend(" ".join(str(value) for value in vertex) for vertex in room_vertices)
    ply.extend("3 " + " ".join(str(value) for value in face) for face in room_faces)
    (background / "mesh_model.ply").write_text("\n".join(ply) + "\n", encoding="ascii")
    collision_root = background / "collision"
    collision_root.mkdir()
    obj = [
        *("v " + " ".join(str(value) for value in vertex) for vertex in room_vertices),
        *("f " + " ".join(str(index + 1) for index in face) for face in room_faces),
    ]
    (collision_root / "walls.obj").write_text("\n".join(obj) + "\n", encoding="utf-8")
    (background / "source_inventory.json").write_bytes(b"{}\n")
    (background / "environment_geometry.json").write_text(
        json.dumps(
            {
                "schema": "InteractMoveEnvironmentGeometry",
                "schema_version": 1,
                "native_profile": "embodiedgen-v2.0.1-sapien-rh-zup-m-xyzw",
                "reference_mesh": {
                    "path": "mesh_model.ply",
                    "frame": "world",
                    "units": "m",
                    "scale_xyz": [1.0, 1.0, 1.0],
                    "purposes": ["interactmove_pointcloud", "raster_render"],
                },
                "collision_geometry": [
                    {
                        "collision_id": "floor",
                        "role": "floor",
                        "geometry_type": "plane",
                        "frame": "world",
                        "z_m": 0.0,
                        "normal": [0.0, 0.0, 1.0],
                    },
                    {
                        "collision_id": "walls",
                        "role": "walls",
                        "geometry_type": "mesh",
                        "frame": "world",
                        "path": "collision/walls.obj",
                        "units": "m",
                        "scale_xyz": [1.0, 1.0, 1.0],
                    },
                ],
                "render_mode": "raster_reference_mesh",
                "provenance": {
                    "source": "EmbodiedGenV2 canary",
                    "embodiedgen_commit": UPSTREAM_COMMIT,
                    "source_job_id": "pytest-canary",
                    "source_artifact_path": "source_inventory.json",
                    "source_artifact_sha256": hashlib.sha256(b"{}\n").hexdigest(),
                    "creation_method": "inline deterministic room fixture",
                },
            }
        ),
        encoding="utf-8",
    )
    for directory, name in (
        ("asset3d/table/result", "table"),
        ("asset3d/blue_bowl/result", "blue_bowl"),
    ):
        asset = root / directory
        mesh = asset / "mesh"
        mesh.mkdir(parents=True)
        (mesh / "visual.obj").write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
        )
        (mesh / "collision.obj").write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
        )
        has_collision = collision or name != "blue_bowl"
        (asset / f"{name}.urdf").write_text(
            _urdf(name, collision=has_collision), encoding="utf-8"
        )
    return root


def _request(spec: dict) -> dict:
    return build_embodiedgen_request(
        spec,
        upstream_version=UPSTREAM_VERSION,
        upstream_commit=UPSTREAM_COMMIT,
        background_list=["nursery"],
    )


def _compile(root: Path, spec: dict, *, request: dict | None) -> dict:
    return compile_scene_bundle(
        root,
        "layout.json",
        spec,
        upstream_version=UPSTREAM_VERSION,
        upstream_commit=UPSTREAM_COMMIT,
        embodiedgen_request=request,
    )


def _instance(bundle: dict, source_key: str) -> dict:
    return next(
        item for item in bundle["instances"] if item["source_node_key"] == source_key
    )


def test_deterministic_scene_compile_and_validate(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    request = _request(spec)

    first = _compile(root, spec, request=request)
    second = _compile(root, deepcopy(spec), request=deepcopy(request))
    assert first == second
    validate_scene_bundle(first)

    expected_target_id = "egv2_" + hashlib.sha256(b"blue bowl").hexdigest()[:24]
    receipt = first["target_resolution_receipt"]
    assert receipt["candidate_source_node_keys"] == ["blue bowl"]
    assert receipt["chosen_source_node_key"] == "blue bowl"
    assert receipt["chosen_instance_id"] == expected_target_id
    assert receipt["method"] == "sole_manipulated_instance"
    assert receipt["source_category"] == "bowl"
    assert receipt["category_exact_match"] is True
    assert receipt["layout_sha256"] == first["source"]["layout"]["sha256"]

    ignored_robot = first["source"]["ignored_native_robot_pose"]
    assert ignored_robot["source_node_key"] == "franka"
    assert ignored_robot["authority"] == "ignored_not_humanoid_authority"
    assert all(item["source_node_key"] != "franka" for item in first["instances"])

    humanoid = first["humanoid_initial_pose"]
    assert humanoid["selected_by"] == "nursery"
    assert humanoid["pose_world"]["translation_m"] == [0.0, 2.0, 0.8]
    assert humanoid["pose_world"]["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert humanoid["provenance"]["accessibility_gate"] == "not_performed_by_contract"
    assert first["validation"]["humanoid_accessibility_check_performed"] is False

    assert first["conventions"]["handedness"] == "right"
    assert first["conventions"]["world_up"] == "+Z"
    assert first["conventions"]["quaternion_order"] == "xyzw"
    target = _instance(first, "blue bowl")
    assert target["initial_pose_world"]["quaternion_canonicalized"] is True
    assert target["initial_pose_world"]["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert target["initial_pose_world"]["matrix_4x4"][2][3] == 0.8
    visual_world = next(
        item
        for item in target["geometry_world_transforms"]
        if item["geometry_type"] == "visual"
    )
    assert visual_world["composition"] == (
        "T_world_instance @ T_link_geometry"
    )
    assert visual_world["T_world_geometry"][0][3] == pytest.approx(1.01)
    assert visual_world["T_world_geometry"][2][3] == pytest.approx(0.8)

    assert (_instance(first, "nursery")["role"], _instance(first, "nursery")["body_type"]) == ("background", "static")
    assert (_instance(first, "table")["role"], _instance(first, "table")["body_type"]) == ("support", "static")
    assert (target["role"], target["body_type"]) == ("target", "dynamic")
    assert target["category"] == "bowl"


def test_official_relation_task_metadata_is_admitted_strictly(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    layout = _layout()
    layout["relation"]["task"] = "pick and place"
    layout["relation"]["task_desc"] = _activity_spec()["prompt"]
    (root / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    spec = _activity_spec()

    bundle = _compile(root, spec, request=_request(spec))

    validate_scene_bundle(bundle)


def test_real_layout_quaternion_canonicalization_is_idempotent() -> None:
    first, changed = canonicalize_quaternion_xyzw(
        [0.0, 0.4783, 0.0, -0.8782], where="official background pose"
    )
    second, changed_again = canonicalize_quaternion_xyzw(
        first, where="canonical background pose", maximum_norm_error=1e-9
    )

    assert changed is True
    assert second == first
    assert changed_again is False


def test_manifest_physics_capabilities_and_request_binding(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    request = _request(spec)
    bundle = _compile(root, spec, request=request)

    paths = [record["path"] for record in bundle["files"]]
    assert paths == sorted(set(paths))
    assert all(record["bytes"] > 0 for record in bundle["files"])
    assert bundle["source"]["layout"]["path"] == "layout.json"
    assert bundle["target_resolution_receipt"]["layout_sha256"] == bundle["source"]["layout"]["sha256"]
    assert bundle["activity_binding"]["embodiedgen_request_sha256"] == content_sha256(request)
    assert bundle["source"]["embodiedgen_request_sha256"] == content_sha256(request)

    target_asset = bundle["assets"][_instance(bundle, "blue bowl")["asset_id"]]
    inertial = target_asset["links"][0]["inertial"]
    material = target_asset["links"][0]["collisions"][0]["material"]
    assert inertial["mass_kg"] == 0.4
    assert inertial["inertia_kg_m2"] is None
    assert material["source_mu1"] == 0.7
    assert material["source_mu2"] == 0.55
    assert material["source_restitution"] is None
    blockers = bundle["capabilities"]["blockers"]
    assert any(item.endswith(":missing_inertia") for item in blockers)
    assert any(item.endswith(":missing_restitution") for item in blockers)
    assert not any(item.endswith(":missing_mass") for item in blockers)
    assert not any(item.endswith(":missing_friction") for item in blockers)
    assert bundle["capabilities"]["physics_material_complete"] is False
    assert bundle["capabilities"]["dynamic_settle_ready"] is False
    assert bundle["capabilities"]["interactmove_scene_input_ready"] is True
    assert bundle["capabilities"]["target_category_exact_match"] is True
    assert bundle["capabilities"]["floor_contact_ready"] is True
    assert bundle["capabilities"]["room_collision_ready"] is True
    assert bundle["capabilities"]["visual_background_ready"] is True
    assert bundle["capabilities"]["complete_scene_ready"] is True
    assert bundle["environment"]["background_geometry_role"] == "validated_render_reference_with_explicit_collision"
    assert bundle["validation"]["mesh_content_validation_performed"] is True
    assert bundle["settling"] == {"status": "not_run", "receipt": None}


def test_gaussian_only_official_background_is_lossless_but_not_pointcloud_ready(
    tmp_path: Path,
) -> None:
    root = _write_canary(tmp_path)
    (root / "background" / "environment_geometry.json").unlink()
    (root / "background" / "mesh_model.ply").unlink()
    (root / "background" / "gs_model.ply").write_bytes(b"ply\ngaussian-canary\n")
    spec = _activity_spec()

    bundle = _compile(root, spec, request=_request(spec))
    validate_scene_bundle(bundle)

    assert bundle["environment"]["reference_mesh"] is None
    assert bundle["environment"]["gaussian_model"] == "background/gs_model.ply"
    assert bundle["capabilities"]["gaussian_render_ready"] is True
    assert bundle["capabilities"]["reference_mesh_ready"] is False
    assert bundle["capabilities"]["interactmove_scene_input_ready"] is False
    assert (
        "environment:background_reference_mesh_not_ready"
        in bundle["capabilities"]["blockers"]
    )


def test_background_requires_mesh_or_gaussian_representation(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    (root / "background" / "environment_geometry.json").unlink()
    (root / "background" / "mesh_model.ply").unlink()
    spec = _activity_spec()

    with pytest.raises(ContractError, match="at least one"):
        _compile(root, spec, request=_request(spec))


def test_unmanifested_reference_mesh_remains_fail_closed(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    (root / "background" / "environment_geometry.json").unlink()
    spec = _activity_spec()

    bundle = _compile(root, spec, request=_request(spec))

    assert bundle["environment"]["reference_mesh"] == "background/mesh_model.ply"
    assert bundle["capabilities"]["reference_mesh_ready"] is False
    assert bundle["capabilities"]["room_collision_ready"] is False
    assert bundle["capabilities"]["interactmove_scene_input_ready"] is False
    assert "environment:background_geometry_manifest_not_ready" in bundle["capabilities"]["blockers"]


def test_environment_manifest_requires_explicit_walls_collision(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    manifest_path = root / "background" / "environment_geometry.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collision_geometry"] = manifest["collision_geometry"][:1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = _activity_spec()

    with pytest.raises(ContractError, match="floor plane and at least one walls"):
        _compile(root, spec, request=_request(spec))


def test_environment_reference_mesh_rejects_nonfinite_vertex(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    mesh_path = root / "background" / "mesh_model.ply"
    text = mesh_path.read_text(encoding="ascii").replace("-4 -4 0", "nan -4 0", 1)
    mesh_path.write_text(text, encoding="ascii")
    spec = _activity_spec()

    with pytest.raises(ContractError, match="non-finite vertices"):
        _compile(root, spec, request=_request(spec))


def test_bundle_mutation_is_rejected_even_with_rehashed_bundle_id(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    bundle = _compile(root, spec, request=_request(spec))
    target = _instance(bundle, "blue bowl")
    target["role"] = "distractor"
    bundle["bundle_id"] = "scene_" + content_sha256(
        {**bundle, "bundle_id": ""}
    )[:32]
    with pytest.raises(ContractError, match="target-role"):
        validate_scene_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda bundle: bundle["humanoid_initial_pose"]["pose_world"].update(
                {"translation_m": [999.0, 999.0, 999.0]}
            ),
            "humanoid_initial_pose disagrees",
        ),
        (
            lambda bundle: _instance(bundle, "blue bowl")[
                "initial_pose_world"
            ].update({"matrix_4x4": [[0.0] * 4 for _ in range(4)]}),
            "matrix_4x4 disagrees",
        ),
    ],
    ids=["humanoid-pose", "instance-matrix"],
)
def test_rehashed_derived_pose_tampering_is_rejected(
    tmp_path: Path, mutate, message: str
) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    bundle = _compile(root, spec, request=_request(spec))
    mutate(bundle)
    bundle["bundle_id"] = "scene_" + content_sha256(
        {**bundle, "bundle_id": ""}
    )[:32]
    with pytest.raises(ContractError, match=message):
        validate_scene_bundle(bundle)


def test_malformed_urdf_number_is_a_contract_error(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    urdf = root / "asset3d/blue_bowl/result/blue_bowl.urdf"
    urdf.write_text(
        urdf.read_text(encoding="utf-8").replace(
            'scale="1 1 1"', 'scale="wat 1 1"', 1
        ),
        encoding="utf-8",
    )
    spec = _activity_spec()
    with pytest.raises(ContractError, match="finite number"):
        _compile(root, spec, request=_request(spec))


def test_disconnected_multilink_urdf_is_rejected(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    urdf = root / "asset3d/blue_bowl/result/blue_bowl.urdf"
    payload = urdf.read_text(encoding="utf-8").replace(
        "</robot>", '<link name="detached"/>\n</robot>'
    )
    urdf.write_text(payload, encoding="utf-8")
    spec = _activity_spec()
    with pytest.raises(ContractError, match="requires joints"):
        _compile(root, spec, request=_request(spec))


def test_rehashed_target_rule_bypass_is_rejected(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    bundle = _compile(root, spec, request=_request(spec))
    bundle["target_request"]["resolution_rule"] = {
        "method": "exact_source_node_key",
        "expected_source_node_key": "not-the-target",
    }
    receipt = bundle["target_resolution_receipt"]
    receipt["method"] = "exact_source_node_key"
    receipt["target_request_sha256"] = content_sha256(bundle["target_request"])
    bundle["bundle_id"] = "scene_" + content_sha256(
        {**bundle, "bundle_id": ""}
    )[:32]
    with pytest.raises(ContractError, match="precommitted key"):
        validate_scene_bundle(bundle)


def test_bad_rehashed_target_category_type_is_a_contract_error(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    bundle = _compile(root, spec, request=_request(spec))
    bundle["target_request"]["category"] = []
    bundle["target_resolution_receipt"]["target_request_sha256"] = content_sha256(
        bundle["target_request"]
    )
    bundle["bundle_id"] = "scene_" + content_sha256(
        {**bundle, "bundle_id": ""}
    )[:32]
    with pytest.raises(ContractError, match="category must be a non-empty string"):
        validate_scene_bundle(bundle)


def test_readiness_cannot_be_forged_by_removing_blockers(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    bundle = _compile(root, spec, request=_request(spec))
    bundle["capabilities"]["blockers"] = [
        blocker
        for blocker in bundle["capabilities"]["blockers"]
        if ":missing_" not in blocker
    ]
    bundle["capabilities"]["physics_material_complete"] = True
    bundle["capabilities"]["dynamic_settle_ready"] = True
    bundle["bundle_id"] = "scene_" + content_sha256(
        {**bundle, "bundle_id": ""}
    )[:32]
    with pytest.raises(ContractError, match="blockers disagree"):
        validate_scene_bundle(bundle)


def test_instance_category_must_remain_bound_to_asset_links(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    bundle = _compile(root, spec, request=_request(spec))
    target = _instance(bundle, "blue bowl")
    asset = bundle["assets"][target["asset_id"]]
    asset["category"] = "knife"
    asset["link_categories"] = ["knife"]
    asset["links"][0]["category"] = "knife"
    bundle["bundle_id"] = "scene_" + content_sha256(
        {**bundle, "bundle_id": ""}
    )[:32]
    with pytest.raises(ContractError, match="category disagrees with its asset"):
        validate_scene_bundle(bundle)


@pytest.mark.parametrize(
    "mutation", ["negative-friction", "boolean-friction", "invalid-inertia"]
)
def test_rehashed_invalid_physics_values_cannot_claim_readiness(
    tmp_path: Path, mutation: str
) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    bundle = _compile(root, spec, request=_request(spec))
    target = _instance(bundle, "blue bowl")
    link = bundle["assets"][target["asset_id"]]["links"][0]
    if mutation == "negative-friction":
        material = link["collisions"][0]["material"]
        material["source_mu1"] = material["static_friction"] = -1.0
        material["source_mu2"] = material["dynamic_friction"] = -2.0
        message = "friction must be nonnegative"
    elif mutation == "boolean-friction":
        material = link["collisions"][0]["material"]
        material["source_mu1"] = material["static_friction"] = 1.0
        material["source_mu2"] = 1.0
        material["dynamic_friction"] = True
        message = "dynamic_friction must be a finite number"
    else:
        link["inertial"]["inertia_kg_m2"] = {
            "ixx": -1.0,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyy": -1.0,
            "iyz": 0.0,
            "izz": -1.0,
        }
        link["inertial"]["source_fields"][
            "inertia_kg_m2"
        ] = "link/inertial/inertia@*"
        bundle["capabilities"]["blockers"] = [
            blocker
            for blocker in bundle["capabilities"]["blockers"]
            if not blocker.endswith(":missing_inertia")
        ]
        message = "positive definite"
    bundle["bundle_id"] = "scene_" + content_sha256(
        {**bundle, "bundle_id": ""}
    )[:32]
    with pytest.raises(ContractError, match=message):
        validate_scene_bundle(bundle)


def test_stale_request_and_ambiguous_sole_target_are_rejected(tmp_path: Path) -> None:
    root = _write_canary(tmp_path)
    spec = _activity_spec()
    stale = _request(spec)
    stale["activity_spec_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="canonical Stage 1 projection"):
        _compile(root, spec, request=stale)

    layout_path = root / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["relation"]["manipulated_objs"].append("red cup")
    layout["objs_mapping"]["red cup"] = "manipulated_objs"
    layout["objs_desc"]["red cup"] = "a red cup"
    layout["assets"]["red cup"] = "asset3d/red_cup/result"
    layout["position"]["red cup"] = [1.2, 2, 0.8, 0, 0, 0, 1]
    layout["tree"]["table"].append(["red cup", "ON"])
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    with pytest.raises(ContractError, match="exactly one manipulated"):
        _compile(root, spec, request=_request(spec))


def test_invalid_graph_unsafe_traversal_and_missing_collision_are_rejected(
    tmp_path: Path,
) -> None:
    unsafe_root = _write_canary(tmp_path / "unsafe")
    spec = _activity_spec()
    with pytest.raises(ContractError, match="relative POSIX path"):
        compile_scene_bundle(
            unsafe_root,
            "../layout.json",
            spec,
            upstream_version=UPSTREAM_VERSION,
            upstream_commit=UPSTREAM_COMMIT,
        )

    graph_root = _write_canary(tmp_path / "graph")
    layout_path = graph_root / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["tree"]["nursery"].append(["blue bowl", "FLOOR"])
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    with pytest.raises(ContractError, match="exactly one parent"):
        _compile(graph_root, spec, request=_request(spec))

    collision_root = _write_canary(tmp_path / "collision", collision=False)
    with pytest.raises(ContractError, match="requires visual and collision meshes"):
        _compile(collision_root, spec, request=_request(spec))


def test_compile_without_request_records_blocker_and_is_not_settle_ready(
    tmp_path: Path,
) -> None:
    root = _write_canary(tmp_path)
    bundle = _compile(root, _activity_spec(), request=None)
    assert bundle["activity_binding"]["embodiedgen_request_sha256"] is None
    assert "embodiedgen_request_not_bound" in bundle["capabilities"]["blockers"]
    assert bundle["capabilities"]["dynamic_settle_ready"] is False
    assert bundle["capabilities"]["interactmove_scene_input_ready"] is False
    validate_scene_bundle(bundle)
