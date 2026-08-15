from __future__ import annotations

from copy import deepcopy
import json

import pytest

from babyworld_lite.interactmove_intermimic.activity import (
    activity_spec_sha256,
    build_embodiedgen_request,
    load_activity_spec,
    validate_activity_spec,
)
from babyworld_lite.interactmove_intermimic.common import (
    ContractError,
    canonicalize_quaternion_xyzw,
)


REQUIRED_METRICS = (
    "target_instance_correct",
    "phase_order_fraction",
    "intended_contact_fraction",
    "stable_grasp_fraction",
    "max_penetration_m",
    "balance_success",
    "final_relation_satisfied",
)


def valid_activity_spec() -> dict:
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
                "end_frame_exclusive": 75,
                "expected_target_contact_hands": [],
            },
            {
                "id": "grasp-and-lift",
                "description": "grasp and lift the target",
                "start_frame": 75,
                "end_frame_exclusive": 250,
                "expected_target_contact_hands": ["left", "right"],
            },
            {
                "id": "settle",
                "description": "settle in the final state",
                "start_frame": 250,
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
            for index, metric in enumerate(REQUIRED_METRICS)
        ],
    }


def test_valid_spec_has_deterministic_digest_and_embodiedgen_request() -> None:
    spec = valid_activity_spec()
    validate_activity_spec(spec)

    digest = activity_spec_sha256(spec)
    assert digest == activity_spec_sha256(deepcopy(spec))
    changed = deepcopy(spec)
    changed["prompt"] += " Please move carefully."
    assert activity_spec_sha256(changed) != digest

    request = build_embodiedgen_request(
        spec,
        upstream_version="v2.0.1",
        upstream_commit="9b333554",
        background_list=["kitchen", "studio"],
    )
    assert request["activity_spec_sha256"] == digest
    assert request["prompt"] == spec["prompt"]
    assert request["upstream"] == {
        "version": "v2.0.1",
        "commit": "9b333554",
    }
    assert request["seeds"] == {"image": 11, "asset": 12, "layout": 13}
    assert request["retry_limits"] == {"image": 4, "asset": 3, "pipeline": 2}
    assert request["render_insert_robot"] is False
    assert request["native_profile"] == (
        "embodiedgen-v2.0.1-sapien-rh-zup-m-xyzw"
    )
    assert request["native_robot_pose_role"] == (
        "may_be_present_for_embodiedgen_layout_placement_but_is_not_"
        "humanoid_authority"
    )
    required = {
        item["path"] for item in request["expected_outputs"] if item["required"]
    }
    optional = {
        item["path"] for item in request["expected_outputs"] if not item["required"]
    }
    assert "layout.json" in required
    assert "background/mesh_model.ply" in required
    assert "background/gs_model.ply" in optional


def test_load_accepts_valid_json_and_rejects_duplicate_keys(tmp_path) -> None:
    spec = valid_activity_spec()
    path = tmp_path / "activity.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert load_activity_spec(path) == spec

    duplicate = json.dumps(spec).replace(
        '"schema":', '"schema": "duplicate", "schema":', 1
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate JSON key"):
        load_activity_spec(path)


def _add_top_level_key(spec: dict) -> None:
    spec["unexpected"] = True


def _add_nested_key(spec: dict) -> None:
    spec["actor"]["unexpected"] = True


def _remove_required_key(spec: dict) -> None:
    del spec["target_request"]


@pytest.mark.parametrize(
    "mutation",
    [_add_top_level_key, _add_nested_key, _remove_required_key],
    ids=["extra-top-level", "extra-nested", "missing-required"],
)
def test_closed_contract_rejects_extra_and_missing_keys(mutation) -> None:
    spec = valid_activity_spec()
    mutation(spec)
    with pytest.raises(ContractError):
        validate_activity_spec(spec)


def test_prompt_span_must_remain_an_exact_verbatim_substring() -> None:
    spec = valid_activity_spec()
    spec["target_request"]["prompt_span"] = "Blue Bowl"
    with pytest.raises(ContractError, match="exact substring"):
        validate_activity_spec(spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("morphology", "child", "morphology"),
        ("body_model", "SMPL-H", "body_model"),
        ("betas", [0.0] * 9 + [float("nan")], "finite number"),
    ],
)
def test_actor_must_be_finite_adult_smplx(field, value, message) -> None:
    spec = valid_activity_spec()
    spec["actor"][field] = value
    with pytest.raises(ContractError, match=message):
        validate_activity_spec(spec)


@pytest.mark.parametrize(
    ("phase_index", "field", "value"),
    [
        (1, "start_frame", 76),
        (2, "end_frame_exclusive", 299),
    ],
    ids=["gap", "incomplete-coverage"],
)
def test_phases_must_contiguously_cover_all_300_frames(
    phase_index, field, value
) -> None:
    spec = valid_activity_spec()
    spec["phases"][phase_index][field] = value
    with pytest.raises(ContractError, match="contiguous|cover"):
        validate_activity_spec(spec)


def test_phase_contact_union_must_match_declared_hand_use() -> None:
    spec = valid_activity_spec()
    spec["phases"][1]["expected_target_contact_hands"] = ["right"]
    spec["phases"][2]["expected_target_contact_hands"] = ["right"]
    with pytest.raises(ContractError, match="union"):
        validate_activity_spec(spec)


@pytest.mark.parametrize("seed", [True, 2**32])
def test_seeds_reject_boolean_and_values_outside_uint32(seed) -> None:
    spec = valid_activity_spec()
    spec["seeds"]["intermimic"] = seed
    with pytest.raises(ContractError):
        validate_activity_spec(spec)


def test_every_required_success_metric_must_be_present() -> None:
    spec = valid_activity_spec()
    spec["success_criteria"] = [
        item
        for item in spec["success_criteria"]
        if item["metric"] != "stable_grasp_fraction"
    ]
    with pytest.raises(ContractError, match="stable_grasp_fraction"):
        validate_activity_spec(spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [("fps", 24), ("frame_count", 299)],
)
def test_stage1_clock_is_locked_to_30_fps_and_300_frames(field, value) -> None:
    spec = valid_activity_spec()
    spec["timing"][field] = value
    with pytest.raises(ContractError, match="exactly"):
        validate_activity_spec(spec)


def test_both_exact_target_resolution_rule_variants_are_accepted() -> None:
    sole = valid_activity_spec()
    validate_activity_spec(sole)

    exact = valid_activity_spec()
    exact["target_request"]["resolution_rule"] = {
        "method": "exact_source_node_key",
        "expected_source_node_key": "bowl_source_07",
    }
    validate_activity_spec(exact)


@pytest.mark.parametrize(
    "rule",
    [
        {
            "method": "sole_manipulated_instance",
            "expected_source_node_key": "bowl_source_07",
        },
        {"method": "exact_source_node_key", "expected_source_node_key": None},
        {"method": "nearest_instance", "expected_source_node_key": None},
    ],
    ids=["sole-with-key", "exact-without-key", "unknown-method"],
)
def test_target_resolution_rule_rejects_every_other_shape(rule) -> None:
    spec = valid_activity_spec()
    spec["target_request"]["resolution_rule"] = rule
    with pytest.raises(ContractError):
        validate_activity_spec(spec)


def test_actor_placement_requires_horizontal_target_offset() -> None:
    spec = valid_activity_spec()
    spec["actor"]["initial_placement"]["target_relative_position_m"] = [
        0.0,
        -0.0,
        1.0,
    ]
    with pytest.raises(ContractError, match="nonzero horizontal offset"):
        validate_activity_spec(spec)


def test_half_turn_quaternion_sign_and_signed_zero_are_canonical() -> None:
    positive, _ = canonicalize_quaternion_xyzw(
        [1.0, -0.0, 0.0, 0.0], where="positive half turn"
    )
    negative, _ = canonicalize_quaternion_xyzw(
        [-1.0, 0.0, -0.0, -0.0], where="negative half turn"
    )
    assert positive == negative == [1.0, 0.0, 0.0, 0.0]
    assert all(str(value) != "-0.0" for value in negative)
