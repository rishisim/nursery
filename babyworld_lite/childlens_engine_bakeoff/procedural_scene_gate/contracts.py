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
SCHEMA = "embodied.integrated_dexterous_scene_gate.v2"
REQUIRED_SCHEMAS = {
    "EpisodeSpec": "embodied.episode_spec.v2",
    "AvatarSpec": "embodied.avatar_spec.v1",
    "EmbodimentManifest": "embodied.embodiment_manifest.v1",
    "SceneSpec": "embodied.scene_spec.v1",
    "ActivityPlan": "embodied.activity_plan.v1",
    "EpisodeTrace": "embodied.episode_trace.v2",
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
    sealed_hashes = implementation.get("sealed_source_sha256", {})
    if implementation.get("status") == "sealed_for_execution":
        if not sealed_hashes:
            raise ValueError("execution freeze requires sealed canonical source hashes")
        for name, expected_sha256 in sealed_hashes.items():
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
    scenes = config["episode_cells"]
    if len(scenes) < 3 or len({row["room_family"] for row in scenes}) < 3:
        raise ValueError("at least three materially distinct room families are frozen")
    compiler = config["scene_compiler"]
    if compiler["target_midpoint_reach_band_m"] != [0.33, 0.38]:
        raise ValueError("the prospective aperture-aware target reach band changed")
    if not compiler["same_band_all_room_seeds"] or compiler["seed_specific_retuning"]:
        raise ValueError("all scenes must use one reach band without retuning")
    plan = config["activity_plan"]
    if not 20.0 <= plan["duration_s"] <= 30.0 or not plan["no_seed_specific_retuning"]:
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
    interaction = config["qa_tolerances"]["interaction"]
    if interaction["right_opposition_min_s"] != 0.30 or interaction["left_support_min_s"] != 0.25:
        raise ValueError("force-bearing dwell thresholds changed")
    if interaction["lift_min_m"] != 0.10 or interaction["turn_min_deg"] != 30.0:
        raise ValueError("integrated manipulation thresholds changed")
    registration = config["qa_tolerances"]["registration"]
    if registration["skin_collider_max_m"] != 0.005 or registration["garment_body_max_penetration_m"] != 0.002:
        raise ValueError("anti-clipping registration thresholds changed")


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
            "instances": ["tableCoffee", "bookcaseOpen", "rugRectangle", "chairCushion", "books", "books", "pottedPlant", "lampSquareFloor", "chairCushion", "books", "rugRectangle"],
        },
        "sage_living_corner": {
            "envelope_m": [4.8, 2.55, 5.2],
            "instances": ["tableCoffee", "loungeSofaLong", "rugRectangle", "lampSquareFloor", "pottedPlant", "books", "chairCushion", "books", "pottedPlant", "rugRectangle", "books"],
        },
        "birch_art_room": {
            "envelope_m": [4.2, 2.55, 4.6],
            "instances": ["tableCoffee", "bookcaseOpen", "chairCushion", "rugRectangle", "pottedPlant", "books", "lampSquareFloor", "books", "chairCushion", "rugRectangle", "books"],
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
        "target": deepcopy(next(row for row in assets["interactive_targets"] if row["persistent_id"] == scene["target_id"])),
        "support_relations": [
            {"child_id": scene["target_id"], "support_id": f"{scene['room_family']}_tableCoffee_00"},
            {"child_id": scene["target_id"], "destination_id": scene["destination_id"]},
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
        "sightlines": {"target_visible_at_required_events": True, "final_gaze_zone": scene["final_gaze_zone"]},
        "episode_cell": scene["cell_id"],
        "destination_id": scene["destination_id"],
        "stabilization_s": 1.0,
        "no_visible_primitive_furniture": True,
    }


def compile_contract_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_frozen_config(config)
    result = []
    avatars = {row["garment_configuration_id"]: row for row in config["avatar_specs"]}
    for scene_index, scene in enumerate(config["episode_cells"]):
            avatar = avatars[scene["garment_configuration_id"]]
            scene_spec = _scene_spec(scene, config["assets"], config["scene_compiler"])
            activity = deepcopy(config["activity_plan"])
            avatar_spec = deepcopy(avatar)
            episode_spec = deepcopy(scene)
            episode_spec["schema"] = REQUIRED_SCHEMAS["EpisodeSpec"]
            contract = {
                "schema": "embodied.compiled_episode_contract.v2",
                "episode_id": scene["episode_id"],
                "matrix_indices": {"scene": scene_index, "garment": scene_index},
                "episode_spec": episode_spec,
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
