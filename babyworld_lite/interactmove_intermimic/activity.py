"""Strict Stage 1 activity contract for the InteractMove--InterMimic pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

from .common import (
    ContractError,
    content_sha256,
    read_json,
    require_exact_keys,
    require_finite_number,
    require_integer,
    require_mapping,
    require_nonempty_string,
    require_vector,
)


ACTIVITY_SCHEMA = "InteractMoveInterMimicActivitySpec"
ACTIVITY_SCHEMA_VERSION = 1
PROTOCOL_ID = "interactmove_intermimic"
EMBODIEDGEN_REQUEST_SCHEMA = "EmbodiedGenRequest"
EMBODIEDGEN_NATIVE_PROFILE = "embodiedgen-v2.0.1-sapien-rh-zup-m-xyzw"

_TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "protocol_id",
    "activity_id",
    "prompt",
    "actor",
    "target_request",
    "relations",
    "phases",
    "timing",
    "hand_use",
    "seeds",
    "success_criteria",
}
_RELATION_PREDICATES = {
    "on",
    "inside",
    "near",
    "held_by",
    "contacting",
    "not_contacting",
}
_SUCCESS_METRICS = {
    "target_instance_correct",
    "phase_order_fraction",
    "intended_contact_fraction",
    "stable_grasp_fraction",
    "max_penetration_m",
    "balance_success",
    "final_relation_satisfied",
}
_BOOLEAN_SUCCESS_METRICS = {
    "target_instance_correct",
    "balance_success",
    "final_relation_satisfied",
}
_FRACTION_SUCCESS_METRICS = {
    "phase_order_fraction",
    "intended_contact_fraction",
    "stable_grasp_fraction",
}
_CANONICAL_HAND_LISTS = ([], ["left"], ["right"], ["left", "right"])
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?")
_UINT32_MAX = 2**32 - 1


def _require_slug(value: Any, *, where: str) -> str:
    text = require_nonempty_string(value, where=where)
    if _SLUG.fullmatch(text) is None:
        raise ContractError(
            f"{where} must be a lowercase path-safe slug containing only "
            "letters, digits, underscores, and hyphens"
        )
    return text


def _require_list(value: Any, *, where: str, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty" if nonempty else "an"
        raise ContractError(f"{where} must be {qualifier} array")
    return value


def _require_canonical_hands(value: Any, *, where: str) -> list[str]:
    hands = _require_list(value, where=where)
    if hands not in _CANONICAL_HAND_LISTS:
        raise ContractError(
            f"{where} must be one of [], ['left'], ['right'], or "
            "['left', 'right']"
        )
    return hands


def _validate_actor(value: Any) -> None:
    actor = require_mapping(value, where="actor")
    require_exact_keys(
        actor,
        required={
            "body_model",
            "morphology",
            "gender",
            "betas",
            "initial_stance",
            "initial_placement",
        },
        where="actor",
    )
    if actor["body_model"] != "SMPL-X":
        raise ContractError("actor.body_model must be 'SMPL-X'")
    if actor["morphology"] != "adult":
        raise ContractError("actor.morphology must be 'adult'")
    if not isinstance(actor["gender"], str) or actor["gender"] not in {
        "neutral",
        "female",
        "male",
    }:
        raise ContractError("actor.gender must be neutral, female, or male")
    require_vector(actor["betas"], length=10, where="actor.betas")
    require_nonempty_string(actor["initial_stance"], where="actor.initial_stance")

    placement = require_mapping(
        actor["initial_placement"], where="actor.initial_placement"
    )
    require_exact_keys(
        placement,
        required={
            "method",
            "target_relative_position_m",
            "face_target",
            "selected_by",
        },
        where="actor.initial_placement",
    )
    if placement["method"] != "target_relative":
        raise ContractError(
            "actor.initial_placement.method must be 'target_relative'"
        )
    target_offset = require_vector(
        placement["target_relative_position_m"],
        length=3,
        where="actor.initial_placement.target_relative_position_m",
    )
    if target_offset[0] == 0.0 and target_offset[1] == 0.0:
        raise ContractError(
            "actor.initial_placement.target_relative_position_m must have a "
            "nonzero horizontal offset so facing yaw is defined"
        )
    if placement["face_target"] is not True:
        raise ContractError("actor.initial_placement.face_target must be true")
    if placement["selected_by"] != "nursery":
        raise ContractError("actor.initial_placement.selected_by must be 'nursery'")


def validate_actor_contract(value: Any) -> None:
    """Validate the actor subset embedded in later-stage receipts."""

    _validate_actor(value)


def validate_target_request_contract(
    value: Any, *, prompt: str | None = None
) -> None:
    """Validate the precommitted target request, with prompt binding if available."""

    target = require_mapping(value, where="target_request")
    require_exact_keys(
        target,
        required={"description", "category", "prompt_span", "resolution_rule"},
        where="target_request",
    )
    require_nonempty_string(target["description"], where="target_request.description")
    require_nonempty_string(target["category"], where="target_request.category")
    prompt_span = require_nonempty_string(
        target["prompt_span"], where="target_request.prompt_span"
    )
    if prompt is not None and prompt_span not in prompt:
        raise ContractError(
            "target_request.prompt_span must be an exact substring of prompt"
        )

    rule = require_mapping(
        target["resolution_rule"], where="target_request.resolution_rule"
    )
    require_exact_keys(
        rule,
        required={"method", "expected_source_node_key"},
        where="target_request.resolution_rule",
    )
    method = rule["method"]
    expected_key = rule["expected_source_node_key"]
    if method == "sole_manipulated_instance":
        if expected_key is not None:
            raise ContractError(
                "sole_manipulated_instance requires expected_source_node_key=null"
            )
    elif method == "exact_source_node_key":
        require_nonempty_string(
            expected_key,
            where="target_request.resolution_rule.expected_source_node_key",
        )
    else:
        raise ContractError(
            "target_request.resolution_rule.method must be "
            "sole_manipulated_instance or exact_source_node_key"
        )


def _validate_relations(value: Any) -> None:
    relations = require_mapping(value, where="relations")
    require_exact_keys(
        relations, required={"start", "final"}, where="relations"
    )
    for state in ("start", "final"):
        items = _require_list(
            relations[state], where=f"relations.{state}", nonempty=True
        )
        for index, raw_relation in enumerate(items):
            where = f"relations.{state}[{index}]"
            relation = require_mapping(raw_relation, where=where)
            require_exact_keys(
                relation,
                required={"subject", "predicate", "object_description", "description"},
                where=where,
            )
            if relation["subject"] != "target":
                raise ContractError(f"{where}.subject must be 'target'")
            if (
                not isinstance(relation["predicate"], str)
                or relation["predicate"] not in _RELATION_PREDICATES
            ):
                raise ContractError(
                    f"{where}.predicate must be one of "
                    f"{sorted(_RELATION_PREDICATES)}"
                )
            require_nonempty_string(
                relation["object_description"],
                where=f"{where}.object_description",
            )
            require_nonempty_string(
                relation["description"], where=f"{where}.description"
            )


def _validate_timing(value: Any) -> tuple[int, int]:
    timing = require_mapping(value, where="timing")
    require_exact_keys(
        timing,
        required={"duration_s", "fps", "frame_count", "sampling"},
        where="timing",
    )
    duration_s = require_finite_number(timing["duration_s"], where="timing.duration_s")
    if duration_s != 10.0:
        raise ContractError("timing.duration_s must be exactly 10.0")
    fps = require_integer(timing["fps"], where="timing.fps", minimum=1)
    if fps != 30:
        raise ContractError("timing.fps must be exactly 30")
    frame_count = require_integer(
        timing["frame_count"], where="timing.frame_count", minimum=1
    )
    if frame_count != 300:
        raise ContractError("timing.frame_count must be exactly 300")
    if timing["sampling"] != "half_open_[0,duration)":
        raise ContractError(
            "timing.sampling must be 'half_open_[0,duration)'"
        )
    return fps, frame_count


def _validate_phases(value: Any, *, frame_count: int) -> set[str]:
    phases = _require_list(value, where="phases", nonempty=True)
    seen_ids: set[str] = set()
    contact_hands: set[str] = set()
    expected_start = 0
    for index, raw_phase in enumerate(phases):
        where = f"phases[{index}]"
        phase = require_mapping(raw_phase, where=where)
        require_exact_keys(
            phase,
            required={
                "id",
                "description",
                "start_frame",
                "end_frame_exclusive",
                "expected_target_contact_hands",
            },
            where=where,
        )
        phase_id = _require_slug(phase["id"], where=f"{where}.id")
        if phase_id in seen_ids:
            raise ContractError(f"duplicate phase id: {phase_id!r}")
        seen_ids.add(phase_id)
        require_nonempty_string(phase["description"], where=f"{where}.description")
        start = require_integer(
            phase["start_frame"], where=f"{where}.start_frame"
        )
        end = require_integer(
            phase["end_frame_exclusive"],
            where=f"{where}.end_frame_exclusive",
            minimum=1,
        )
        if start != expected_start:
            raise ContractError(
                f"{where}.start_frame must be {expected_start} for contiguous coverage"
            )
        if end <= start:
            raise ContractError(f"{where} must span at least one frame")
        if end > frame_count:
            raise ContractError(f"{where}.end_frame_exclusive exceeds frame_count")
        expected_start = end
        hands = _require_canonical_hands(
            phase["expected_target_contact_hands"],
            where=f"{where}.expected_target_contact_hands",
        )
        contact_hands.update(hands)
    if expected_start != frame_count:
        raise ContractError("phases must contiguously cover [0, frame_count)")
    return contact_hands


def _validate_hand_use(value: Any, *, phase_contact_hands: set[str]) -> None:
    hand_use = require_mapping(value, where="hand_use")
    require_exact_keys(hand_use, required={"mode", "hands"}, where="hand_use")
    mode = hand_use["mode"]
    hands = _require_canonical_hands(hand_use["hands"], where="hand_use.hands")
    if mode == "one_hand":
        if hands not in (["left"], ["right"]):
            raise ContractError("one_hand mode requires exactly one declared hand")
    elif mode == "two_hand":
        if hands != ["left", "right"]:
            raise ContractError(
                "two_hand mode requires canonical hands ['left', 'right']"
            )
    else:
        raise ContractError("hand_use.mode must be one_hand or two_hand")
    if set(hands) != phase_contact_hands:
        raise ContractError(
            "union of phase expected_target_contact_hands must equal hand_use.hands"
        )


def _validate_seeds(value: Any) -> None:
    seeds = require_mapping(value, where="seeds")
    names = {
        "master",
        "embodiedgen_image",
        "embodiedgen_asset",
        "embodiedgen_layout",
        "interactmove",
        "intermimic",
        "render",
    }
    require_exact_keys(seeds, required=names, where="seeds")
    for name in sorted(names):
        seed = require_integer(seeds[name], where=f"seeds.{name}")
        if seed > _UINT32_MAX:
            raise ContractError(f"seeds.{name} must be <= {_UINT32_MAX}")


def _validate_success_criteria(value: Any) -> None:
    criteria = _require_list(value, where="success_criteria", nonempty=True)
    seen_ids: set[str] = set()
    observed_metrics: set[str] = set()
    for index, raw_criterion in enumerate(criteria):
        where = f"success_criteria[{index}]"
        criterion = require_mapping(raw_criterion, where=where)
        require_exact_keys(
            criterion,
            required={
                "id",
                "metric",
                "operator",
                "threshold",
                "unit",
                "evaluation_window",
            },
            where=where,
        )
        criterion_id = _require_slug(criterion["id"], where=f"{where}.id")
        if criterion_id in seen_ids:
            raise ContractError(f"duplicate success criterion id: {criterion_id!r}")
        seen_ids.add(criterion_id)
        metric = criterion["metric"]
        if not isinstance(metric, str) or metric not in _SUCCESS_METRICS:
            raise ContractError(
                f"{where}.metric must be one of {sorted(_SUCCESS_METRICS)}"
            )
        observed_metrics.add(metric)
        operator = criterion["operator"]
        if not isinstance(operator, str) or operator not in {"eq", "gte", "lte"}:
            raise ContractError(f"{where}.operator must be eq, gte, or lte")
        threshold = criterion["threshold"]
        if not isinstance(threshold, bool):
            threshold = require_finite_number(
                threshold, where=f"{where}.threshold"
            )
        unit = criterion["unit"]
        if unit is not None and not isinstance(unit, str):
            raise ContractError(f"{where}.unit must be a string or null")
        if metric in _BOOLEAN_SUCCESS_METRICS:
            if operator != "eq" or threshold is not True or unit is not None:
                raise ContractError(
                    f"{where} Boolean success metrics require operator='eq', "
                    "threshold=true, and unit=null"
                )
        elif metric in _FRACTION_SUCCESS_METRICS:
            if (
                operator != "gte"
                or isinstance(threshold, bool)
                or not 0.0 <= threshold <= 1.0
                or unit is not None
            ):
                raise ContractError(
                    f"{where} fraction metrics require operator='gte', a "
                    "numeric threshold in [0,1], and unit=null"
                )
        elif metric == "max_penetration_m":
            if (
                operator != "lte"
                or isinstance(threshold, bool)
                or threshold < 0.0
                or unit != "m"
            ):
                raise ContractError(
                    f"{where} max_penetration_m requires operator='lte', a "
                    "non-negative numeric threshold, and unit='m'"
                )
        require_nonempty_string(
            criterion["evaluation_window"],
            where=f"{where}.evaluation_window",
        )
    missing = sorted(_SUCCESS_METRICS - observed_metrics)
    if missing:
        raise ContractError(f"success_criteria is missing required metrics: {missing}")


def validate_activity_spec(value: Mapping[str, Any]) -> None:
    """Fail closed unless ``value`` satisfies the complete Stage 1 contract."""

    spec = require_mapping(value, where="activity spec")
    require_exact_keys(spec, required=_TOP_LEVEL_KEYS, where="activity spec")
    if spec["schema"] != ACTIVITY_SCHEMA:
        raise ContractError(f"schema must be {ACTIVITY_SCHEMA!r}")
    if (
        isinstance(spec["schema_version"], bool)
        or spec["schema_version"] != ACTIVITY_SCHEMA_VERSION
    ):
        raise ContractError("schema_version must be integer 1")
    if spec["protocol_id"] != PROTOCOL_ID:
        raise ContractError(f"protocol_id must be {PROTOCOL_ID!r}")
    _require_slug(spec["activity_id"], where="activity_id")
    prompt = require_nonempty_string(spec["prompt"], where="prompt")
    _validate_actor(spec["actor"])
    validate_target_request_contract(spec["target_request"], prompt=prompt)
    _validate_relations(spec["relations"])
    _, frame_count = _validate_timing(spec["timing"])
    phase_hands = _validate_phases(spec["phases"], frame_count=frame_count)
    _validate_hand_use(spec["hand_use"], phase_contact_hands=phase_hands)
    _validate_seeds(spec["seeds"])
    _validate_success_criteria(spec["success_criteria"])


def load_activity_spec(path: Path) -> dict[str, Any]:
    """Load strict JSON from ``path`` and validate it as a Stage 1 activity."""

    value = read_json(path)
    validate_activity_spec(value)
    return value


def activity_spec_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical semantic digest of a valid Stage 1 activity."""

    validate_activity_spec(value)
    return content_sha256(value)


def build_embodiedgen_request(
    value: Mapping[str, Any],
    *,
    upstream_version: str,
    upstream_commit: str,
    background_list: Sequence[str],
) -> dict[str, Any]:
    """Project a valid activity into a reproducible EmbodiedGen request."""

    validate_activity_spec(value)
    version = require_nonempty_string(
        upstream_version, where="upstream_version"
    )
    commit = require_nonempty_string(upstream_commit, where="upstream_commit")
    if isinstance(background_list, (str, bytes)) or not isinstance(
        background_list, Sequence
    ):
        raise ContractError("background_list must be an array of strings")
    backgrounds = [
        require_nonempty_string(item, where=f"background_list[{index}]")
        for index, item in enumerate(background_list)
    ]
    if not backgrounds:
        raise ContractError("background_list must be non-empty")
    if len(backgrounds) != len(set(backgrounds)):
        raise ContractError("background_list entries must be unique")

    seeds = value["seeds"]
    return {
        "schema": EMBODIEDGEN_REQUEST_SCHEMA,
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "activity_id": value["activity_id"],
        "activity_spec_sha256": activity_spec_sha256(value),
        "prompt": value["prompt"],
        "upstream": {"version": version, "commit": commit},
        "native_profile": EMBODIEDGEN_NATIVE_PROFILE,
        "background_list": backgrounds,
        "seeds": {
            "image": seeds["embodiedgen_image"],
            "asset": seeds["embodiedgen_asset"],
            "layout": seeds["embodiedgen_layout"],
        },
        "retry_limits": {"image": 4, "asset": 3, "pipeline": 2},
        "render_insert_robot": False,
        "native_robot_pose_role": (
            "may_be_present_for_embodiedgen_layout_placement_but_is_not_"
            "humanoid_authority"
        ),
        "expected_outputs": [
            {"path": "layout.json", "required": True},
            {"path": "asset3d/**/result/*.urdf", "required": True},
            {
                "path": "each_urdf_referenced_visual_and_collision_meshes",
                "required": True,
            },
            {
                "path": (
                    "at_least_one_of_background/mesh_model.ply_or_"
                    "background/gs_model.ply"
                ),
                "required": True,
            },
            {"path": "background/mesh_model.ply", "required": False},
            {"path": "background/gs_model.ply", "required": False},
            {"path": "affordance_annot.json", "required": False},
            {"path": "mesh_part_seg.glb", "required": False},
        ],
    }


__all__ = [
    "ACTIVITY_SCHEMA",
    "ACTIVITY_SCHEMA_VERSION",
    "EMBODIEDGEN_NATIVE_PROFILE",
    "EMBODIEDGEN_REQUEST_SCHEMA",
    "PROTOCOL_ID",
    "activity_spec_sha256",
    "build_embodiedgen_request",
    "load_activity_spec",
    "validate_actor_contract",
    "validate_activity_spec",
    "validate_target_request_contract",
]
