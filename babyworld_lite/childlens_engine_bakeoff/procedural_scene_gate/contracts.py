from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "embodied_simulation_procedural_scene_gate.json"
OUTPUT_ROOT = REPOSITORY_ROOT / "runs" / "embodied_simulation" / "procedural_scene_gate"
SCHEMA = "embodied.procedural_scene_gate.v1"
REQUIRED_SCHEMAS = {
    "AvatarSpec": "embodied.avatar_spec.v1",
    "EmbodimentManifest": "embodied.embodiment_manifest.v1",
    "SceneSpec": "embodied.scene_spec.v1",
    "ActivityPlan": "embodied.activity_plan.v1",
    "EpisodeTrace": "embodied.episode_trace.v1",
    "TruthProvenance": "embodied.truth_provenance.v1",
    "QATolerances": "embodied.qa_tolerances.v1",
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    authority = config["authority"]
    if authority["physics_hz"] != 240 or authority["render_hz"] != 30:
        raise ValueError("the frozen gate requires 240 Hz physics and 30 Hz rendering")
    if authority["physics_hz"] // authority["render_hz"] != authority["steps_per_render_frame"]:
        raise ValueError("physics/render clocks must have the frozen exact integer mapping")
    implementation = config["implementation_freeze"]
    if not implementation["same_controller_all_seeds_and_garments"] or implementation["seed_specific_retuning"]:
        raise ValueError("the frozen implementation must be shared without seed-specific retuning")
    module_root = Path(__file__).resolve().parent
    for name, expected_sha256 in implementation["source_sha256"].items():
        source = module_root / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"canonical implementation source hash changed: {name}")
    registry = config["schema_registry"]
    for name, schema in REQUIRED_SCHEMAS.items():
        if registry.get(name, {}).get("schema") != schema:
            raise ValueError(f"missing or changed frozen schema: {name}")
    avatar_specs = config["avatar_specs"]
    if len(avatar_specs) < 3 or len({row["garment_configuration_id"] for row in avatar_specs}) < 3:
        raise ValueError("at least three distinct clothing configurations are frozen")
    if any(row["hand_topology"] != {"hands": 2, "digits_per_hand": 5, "segments_per_digit": 3} for row in avatar_specs):
        raise ValueError("every AvatarSpec must bind two five-finger three-segment hands")
    scenes = config["scene_matrix"]
    if len(scenes) < 3 or len({row["room_family"] for row in scenes}) < 3:
        raise ValueError("at least three materially distinct room families are frozen")
    compiler = config["scene_compiler"]
    if compiler["target_midpoint_reach_band_m"] != [0.34, 0.38]:
        raise ValueError("the aperture-aware target reach band changed")
    if compiler["target_midpoint_reach_center_m"] != 0.36 or compiler["seed_modulo_offset_m"] != 0.02:
        raise ValueError("the deterministic aperture-aware reach mapping changed")
    if compiler["target_lateral_bias_toward_right_shoulder_m"] != 0.025:
        raise ValueError("the shared bimanual lateral reach bias changed")
    if not compiler["same_band_all_room_seeds"] or compiler["seed_specific_retuning"]:
        raise ValueError("all scenes must use one reach band without retuning")
    plan = config["activity_plan"]
    if not 12.0 <= plan["duration_s"] <= 20.0 or not plan["no_seed_specific_retuning"]:
        raise ValueError("ActivityPlan duration or retuning rule changed")
    phases = plan["phases"]
    if phases[0]["start_s"] != 0.0 or phases[-1]["end_s"] != plan["duration_s"]:
        raise ValueError("ActivityPlan must cover the complete episode")
    for previous, current in zip(phases, phases[1:]):
        if previous["end_s"] != current["start_s"]:
            raise ValueError("ActivityPlan phases must be contiguous")
    trace = config["trace_contract"]
    if trace["assistance_ledger"] != []:
        raise ValueError("the assistance ledger must freeze empty")
    streams = config["qa_tolerances"]["capture"]["streams"]
    if streams != ["rgb", "metric_depth", "semantic", "persistent_instance"]:
        raise ValueError("registered capture streams changed")
    if config["privacy_and_claims"]["restricted_childlens_access_permitted"]:
        raise ValueError("restricted ChildLens access must remain prohibited")


def assert_output_root_ignored(output_root: Path, repository_root: Path = REPOSITORY_ROOT) -> None:
    relative = output_root.resolve().relative_to(repository_root.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(relative)],
        cwd=repository_root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"generated output root is not ignored: {relative}")


def _scene_spec(
    scene: dict[str, Any], assets: dict[str, Any], compiler: dict[str, Any]
) -> dict[str, Any]:
    seed = scene["seed"]
    offset_index = seed % 3 - 1
    requested_midpoint_reach = round(
        compiler["target_midpoint_reach_center_m"]
        + compiler["seed_modulo_offset_m"] * offset_index,
        6,
    )
    layouts = {
        "warm_playroom": {
            "envelope_m": [4.4, 2.55, 4.8],
            "instances": ["tableCoffee", "bookcaseOpen", "rugRectangle", "chairCushion", "books"],
        },
        "sage_living_corner": {
            "envelope_m": [4.8, 2.55, 5.2],
            "instances": ["tableCoffee", "loungeSofaLong", "rugRectangle", "lampSquareFloor", "pottedPlant"],
        },
        "birch_art_room": {
            "envelope_m": [4.2, 2.55, 4.6],
            "instances": ["tableCoffee", "bookcaseOpen", "chairCushion", "rugRectangle", "pottedPlant"],
        },
    }
    layout = layouts[scene["room_family"]]
    instances = []
    for index, asset_id in enumerate(layout["instances"]):
        member = next(row for row in assets["furniture_catalog"]["members"] if row["id"] == asset_id)
        instances.append(
            {
                "persistent_id": f"{scene['room_family']}_{asset_id}_{index:02d}",
                "asset_id": asset_id,
                "asset_dimensions_m": member["dimensions_m"],
                "semantic_class": member["semantic_class"],
                "collision_source": member["collision_source"],
                "interactive": False,
                "mass_kg": None,
                "static_friction": 0.75,
                "dynamic_friction": 0.65,
            }
        )
    return {
        "schema": REQUIRED_SCHEMAS["SceneSpec"],
        "seed": seed,
        "room_family": scene["room_family"],
        "material_variant": scene["material_variant"],
        "envelope_m": layout["envelope_m"],
        "zones": scene["zones"],
        "instances": instances,
        "target": deepcopy(assets["interactive_target"]),
        "support_relations": [
            {"child_id": "target_001", "support_id": f"{scene['room_family']}_tableCoffee_00"}
        ],
        "reachability": {
            "aperture_aware": True,
            "compiled_requested_midpoint_m": requested_midpoint_reach,
            "compiled_midpoint_band_m": compiler["target_midpoint_reach_band_m"],
            "lateral_bias_toward_right_shoulder_m": compiler[
                "target_lateral_bias_toward_right_shoulder_m"
            ],
            "measured_bilateral_shoulder_limit_m": compiler[
                "measured_bilateral_shoulder_reach_limit_m"
            ],
            "deterministic_mapping": compiler["deterministic_mapping"],
            "seed_specific_retuning": False,
        },
        "sightlines": {"target_visible_at_required_events": True, "final_gaze_zone": "window_scan"},
        "stabilization_s": 1.0,
        "no_visible_primitive_furniture": True,
    }


def compile_contract_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_frozen_config(config)
    result = []
    for scene_index, scene in enumerate(config["scene_matrix"]):
        for garment_index, avatar in enumerate(config["avatar_specs"]):
            scene_spec = _scene_spec(scene, config["assets"], config["scene_compiler"])
            activity = deepcopy(config["activity_plan"])
            avatar_spec = deepcopy(avatar)
            contract = {
                "schema": "embodied.compiled_episode_contract.v1",
                "episode_id": f"{scene['room_family']}__{avatar['garment_configuration_id']}",
                "matrix_indices": {"scene": scene_index, "garment": garment_index},
                "avatar_spec": avatar_spec,
                "scene_spec": scene_spec,
                "activity_plan": activity,
                "trace_contract": deepcopy(config["trace_contract"]),
                "truth_provenance": deepcopy(config["truth_provenance"]),
                "qa_tolerances": deepcopy(config["qa_tolerances"]),
                "forbidden_assistance": list(config["forbidden_assistance"]),
                "authority": deepcopy(config["authority"]),
            }
            contract["contract_sha256"] = _canonical_sha256(contract)
            result.append(contract)
    return result


def write_frozen_bundle(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    config = load_frozen_config()
    validate_frozen_config(config)
    assert_output_root_ignored(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    matrix = compile_contract_matrix(config)
    frozen = deepcopy(config)
    frozen["config_sha256"] = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    frozen["compiled_episode_count"] = len(matrix)
    (output_root / "frozen_contract.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for contract in matrix:
        episode_root = output_root / "episodes" / contract["episode_id"]
        episode_root.mkdir(parents=True, exist_ok=True)
        (episode_root / "compiled_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    receipt = {
        "schema": "embodied.contract_freeze_receipt.v1",
        "config_sha256": frozen["config_sha256"],
        "compiled_episode_count": len(matrix),
        "episode_contract_sha256": {row["episode_id"]: row["contract_sha256"] for row in matrix},
        "output_root": str(output_root),
        "output_root_ignored": True,
    }
    (output_root / "contract_freeze_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
