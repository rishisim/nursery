from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA = "embodied.unity_native.episode_spec.v1"
ROOMS = {"playroom", "living_area", "art_corner"}
ACTIONS = ("scan", "reorient", "reach", "touch", "grasp", "lift", "inspect", "rotate", "transfer", "place", "release", "withdraw", "gaze_away")
COLORS = ("red", "blue", "yellow", "green", "orange")
ASSETS = {
    "avatar": {"id": "mpfb_child_cc0", "sha256": "b766981d9d3504cea220c0d72ad8aa56cbd80453e910fc76dc8c8814fbd980de", "license": "CC0"},
    "furniture": {"id": "kenney_furniture_kit_cc0", "sha256": "68afa4e6dc8a53942379fb47f1e84ec735d46b77bb1c1ceb968e245693dde067", "license": "CC0"},
    "synthetic": {"id": "repository_parametric_primitives", "sha256": hashlib.sha256(b"unity-native-gate-v1-primitives").hexdigest(), "license": "repository-authored"},
}
ROOM_CATALOG = {
    "playroom": {"zones": ["low_play_table", "rug_play_zone", "toy_shelf"], "furniture": ["tableCoffee", "rugRectangle", "bookcaseOpen", "chairCushion"], "lighting": "warm_window"},
    "living_area": {"zones": ["coffee_table", "sofa_zone", "window_sightline"], "furniture": ["loungeSofaLong", "tableCoffee", "lampSquareFloor", "pottedPlant"], "lighting": "soft_daylight"},
    "art_corner": {"zones": ["craft_table", "shallow_bin_zone", "display_shelf"], "furniture": ["tableCoffee", "bookcaseOpen", "chairCushion", "rugRectangle"], "lighting": "north_light"},
}


@dataclass(frozen=True)
class EpisodeSpec:
    schema: str
    prompt: str
    seed: int
    room_family: str
    duration_s: float
    profile: str
    target: dict
    distractors: list[dict]
    phases: list[dict]
    contact_regions: dict
    success_predicates: list[dict]
    event_language_intents: list[dict]
    controller_profile_id: str
    camera_profile_id: str
    assets: dict
    provenance: dict
    backend: str = "unity_physx"


def _tokens(prompt: str) -> set[str]:
    return set(re.findall(r"[a-z]+", prompt.lower()))


def compile_episode(prompt: str, seed: int, room_family: str, duration_s: float, profile: str) -> EpisodeSpec:
    if room_family not in ROOMS:
        raise ValueError(f"room_family must be one of {sorted(ROOMS)}")
    if not 12 <= duration_s <= 18:
        raise ValueError("duration_s must be within the frozen 12-18 second gate")
    words = _tokens(prompt)
    color = next((x for x in COLORS if x in words), "red")
    if "cup" in words: geometry, identity, strategy = "cup", "cup", "thumb_finger_wrap"
    elif "block" in words or "cube" in words: geometry, identity, strategy = "rounded_cube", "block", "opposed_pads"
    else: geometry, identity, strategy = "rounded_toy", "toy", "multi_digit_enclosure"
    if "tray" in words: destination = "tray"
    elif "bin" in words: destination = "shallow_bin"
    else: destination = "source_support"
    requested = [a for a in ACTIONS if a in words]
    spine = ["scan", "reorient", "reach", "touch", "grasp", "lift"]
    spine += ["inspect", "rotate"] if ("inspect" in words or "turn" in words or "look" in words) else []
    spine += ["transfer"] if destination != "source_support" else []
    spine += ["place", "release", "withdraw"]
    if "away" in words or "window" in words: spine += ["gaze_away"]
    ordered = list(dict.fromkeys(spine + requested))
    phase_s = duration_s / len(ordered)
    phases = [{"id": action, "start_s": round(i * phase_s, 6), "end_s": round((i + 1) * phase_s, 6), "action": action} for i, action in enumerate(ordered)]
    phases[-1]["end_s"] = duration_s
    rng = random.Random(seed)
    distractor_colors = [c for c in COLORS if c != color]
    rng.shuffle(distractor_colors)
    target = {"persistent_id": f"{color}_{identity}_001", "semantic_id": 41, "instance_id": 41001, "identity": identity, "color": color, "geometry": geometry, "contact_strategy": strategy, "destination": destination, "dynamic": True}
    distractors = [{"persistent_id": f"{c}_distractor_{i+1:03d}", "semantic_id": 42 + i, "instance_id": 42001 + i, "color": c, "interactive": False} for i, c in enumerate(distractor_colors[:3])]
    return EpisodeSpec(SCHEMA, prompt, seed, room_family, duration_s, profile, target, distractors, phases,
        {"required_digits": ["thumb", "index", "middle"], "strategy": strategy, "target_regions": ["opposed_side_a", "opposed_side_b"]},
        [{"id": "multi_digit_capture", "minimum_digits": 3, "dwell_s": .2}, {"id": "lift", "minimum_m": .08}, {"id": "release", "destination": destination, "free_body": True}],
        [{"intent": "acknowledge_target", "after": "reorient"}, {"intent": "acknowledge_release", "after": "release"}],
        "anatomical_impedance_v1", "child_head_neutral_fov68_v1", ASSETS,
        {"compiler": "deterministic_local_bounded_v1", "cloud_required": False, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "seed": seed})


def compile_scene(episode: EpisodeSpec) -> dict:
    rng = random.Random(episode.seed ^ 0x51CE)
    room = ROOM_CATALOG[episode.room_family]
    width, depth = round(rng.uniform(3.8, 4.8), 3), round(rng.uniform(4.2, 5.4), 3)
    lateral = round(rng.uniform(-.09, .09), 4)
    support = {"persistent_id": "primary_support", "asset": "tableCoffee", "transform": {"position_m": [lateral, .31, .65], "rotation_y_deg": 0, "scale_m": [1.15, .62, .86]}, "interactive": False, "collision_policy": "static_support"}
    furniture_slots = [(-1.25, .46, 1.78, [1.65, .92, .78]), (1.55, .82, 1.75, [.72, 1.64, .35]), (-1.55, .62, -.55, [.55, 1.24, .55])]
    instances = [support]
    for i, name in enumerate([x for x in room["furniture"] if x != "tableCoffee"]):
        x, y, z, scale = furniture_slots[i]
        instances.append({"persistent_id": f"furniture_{i+1:03d}", "asset": name, "transform": {"position_m": [round(x + lateral * .3, 4), y, z], "rotation_y_deg": round((i - 1) * 9 + rng.uniform(-3, 3), 3), "scale_m": scale}, "interactive": False, "collision_policy": "static_environment"})
    target_pos = [round(lateral + .38, 4), .6475, .36]
    destination_kind = episode.target["destination"]
    destination_pos = [round(lateral - .30, 4), .6325, .68]
    distractor_poses = [[round(lateral + .10, 4), .65, .52], [round(lateral - .18, 4), .65, .40], [round(lateral + .20, 4), .65, .68]]
    target = {**episode.target, "transform": {"position_m": target_pos, "rotation_y_deg": 0, "scale_m": [.055, .055, .055]}, "support_id": "primary_support", "collision_policy": "free_dynamic"}
    distractors = [{**row, "transform": {"position_m": distractor_poses[i], "rotation_y_deg": round(rng.uniform(-25, 25), 3), "scale_m": [.05, .06, .05]}, "support_id": "primary_support", "collision_policy": "static_clutter"} for i, row in enumerate(episode.distractors)]
    destination = {"persistent_id": "destination_001", "kind": destination_kind, "transform": {"position_m": destination_pos, "rotation_y_deg": 0, "scale_m": [.24, .025, .19]}, "support_id": "primary_support", "interactive": destination_kind != "source_support", "collision_policy": "static_receptacle"}
    corridor = {"start_m": [0, 0, -.15], "end_m": [lateral, 0, .18], "radius_m": .22}
    blocking = [x for x in instances if x["persistent_id"] != "primary_support"]
    eye = [0, 1.13, .13]
    direction = [target_pos[i] - eye[i] for i in range(3)]
    sight_distance = sum(v * v for v in direction) ** .5
    sightline = {"origin_m": eye, "target_m": target_pos, "distance_m": round(sight_distance, 6), "checked_instance_ids": [x["persistent_id"] for x in blocking]}
    scene = {"schema": "embodied.unity_native.scene_spec.v1", "episode_schema": episode.schema, "seed": episode.seed,
        "room": {"family": episode.room_family, "envelope_m": [width, 2.55, depth], "zones": room["zones"], "lighting": room["lighting"], "root_transform": {"position_m": [0, 0, 0], "rotation_y_deg": 0}, "openings": [{"persistent_id": "window_001", "type": "window", "wall": "front_positive_z", "transform": {"position_m": [1.1, 1.45, depth * .5], "rotation_y_deg": 180, "scale_m": [1.1, .9, .05]}, "sightline_required": True}]},
        "avatar_spawn": {"position_m": [0, 0, -.15], "rotation_y_deg": 0, "face_forward": "+Z"},
        "embodiment_reach_profile": json.loads((Path(__file__).parents[3] / "configs" / "embodied_simulation_unity_native_gate.json").read_text())["embodiment_reach_profile"],
        "instances": instances, "target": target, "destination": destination, "distractors": distractors,
        "support_relations": [{"child_id": target["persistent_id"], "support_id": "primary_support"}, {"child_id": destination["persistent_id"], "support_id": "primary_support"}] + [{"child_id": x["persistent_id"], "support_id": "primary_support"} for x in distractors],
        "approach_corridor": corridor,
        "compiled_validation": {"event_sightline": sightline, "stabilization_required_s": 1.0},
        "variation": {"wall_hue": round(rng.random(), 6), "material_variant": rng.randrange(4), "layout_variant": rng.randrange(1000000), "lateral_offset_m": lateral}, "assets": episode.assets}
    return validate_scene_geometry(scene)


def _world_aabb(item: dict) -> tuple[list[float], list[float]]:
    transform = item["transform"]
    p, s = transform["position_m"], transform["scale_m"]
    angle = abs(__import__("math").radians(transform.get("rotation_y_deg", 0)))
    c, sn = abs(__import__("math").cos(angle)), abs(__import__("math").sin(angle))
    extent = [(c * s[0] + sn * s[2]) * .5, s[1] * .5, (sn * s[0] + c * s[2]) * .5]
    return ([p[i] - extent[i] for i in range(3)], [p[i] + extent[i] for i in range(3)])


def _overlap(a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]) -> bool:
    return all(a[0][i] < b[1][i] and b[0][i] < a[1][i] for i in range(3))


def _segment_aabb(origin: list[float], target: list[float], box: tuple[list[float], list[float]]) -> tuple[bool, float]:
    t0, t1 = 0.0, 1.0
    direction = [target[i] - origin[i] for i in range(3)]
    for i in range(3):
        if abs(direction[i]) < 1e-12:
            if origin[i] < box[0][i] or origin[i] > box[1][i]:
                break
            continue
        lo, hi = (box[0][i] - origin[i]) / direction[i], (box[1][i] - origin[i]) / direction[i]
        if lo > hi:
            lo, hi = hi, lo
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 > t1:
            break
    else:
        if t0 <= t1:
            return True, 0.0
    minimum = float("inf")
    for n in range(101):
        point = [origin[i] + direction[i] * n / 100 for i in range(3)]
        delta = [max(box[0][i] - point[i], 0, point[i] - box[1][i]) for i in range(3)]
        minimum = min(minimum, sum(x * x for x in delta) ** .5)
    return False, minimum


def validate_scene_geometry(scene: dict) -> dict:
    support = next(x for x in scene["instances"] if x["persistent_id"] == "primary_support")
    structural = [x for x in scene["instances"] if x["persistent_id"] != "primary_support"]
    tabletop = [scene["target"], scene["destination"], *scene["distractors"]]
    structural_pairs, tabletop_pairs = [], []
    for group, receipt in ((scene["instances"], structural_pairs), (tabletop, tabletop_pairs)):
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if _overlap(_world_aabb(a), _world_aabb(b)):
                    raise ValueError(f"unintended overlap: {a['persistent_id']} / {b['persistent_id']}")
                receipt.append([a["persistent_id"], b["persistent_id"]])
    support_box = _world_aabb(support)
    relations = {x["child_id"]: x for x in scene["support_relations"]}
    support_receipts = []
    for child in tabletop:
        box = _world_aabb(child)
        gap = box[0][1] - support_box[1][1]
        contained = box[0][0] >= support_box[0][0] and box[1][0] <= support_box[1][0] and box[0][2] >= support_box[0][2] and box[1][2] <= support_box[1][2]
        relations[child["persistent_id"]]["initial_gap_m"] = round(gap, 6)
        support_receipts.append({"child_id": child["persistent_id"], "contained_xz": contained, "computed_gap_m": round(gap, 6)})
        if not contained or abs(gap) > .001:
            raise ValueError(f"invalid support relation: {child['persistent_id']}")
    sight = scene["compiled_validation"]["event_sightline"]
    obstacle_receipts = []
    for obstacle in structural:
        hit, clearance = _segment_aabb(sight["origin_m"], sight["target_m"], _world_aabb(obstacle))
        obstacle_receipts.append({"instance_id": obstacle["persistent_id"], "intersects": hit, "minimum_sampled_clearance_m": round(clearance, 6)})
        if hit:
            raise ValueError(f"blocked event sightline: {obstacle['persistent_id']}")
    sight["obstacle_receipts"] = obstacle_receipts
    sight["unoccluded_by_compiled_aabbs"] = True
    corridor = scene["approach_corridor"]
    corridor_receipts = []
    for obstacle in structural:
        hit, clearance = _segment_aabb(corridor["start_m"], corridor["end_m"], _world_aabb(obstacle))
        capsule_clear = not hit and clearance >= corridor["radius_m"]
        corridor_receipts.append({"instance_id": obstacle["persistent_id"], "centerline_intersects": hit,
            "centerline_clearance_m": round(clearance, 6), "required_radius_m": corridor["radius_m"], "capsule_clear": capsule_clear})
        if not capsule_clear:
            raise ValueError(f"blocked approach corridor: {obstacle['persistent_id']}")
    avatar = scene["avatar_spawn"]
    avatar_radius = .22
    avatar_receipts = []
    for item in scene["instances"]:
        box = _world_aabb(item)
        point = avatar["position_m"]
        delta = [max(box[0][i] - point[i], 0, point[i] - box[1][i]) for i in range(3)]
        clearance = sum(x * x for x in delta) ** .5
        clear = clearance >= avatar_radius
        avatar_receipts.append({"instance_id": item["persistent_id"], "clearance_m": round(clearance, 6), "required_radius_m": avatar_radius, "clear": clear})
        if not clear:
            raise ValueError(f"avatar spawn intersects clearance volume: {item['persistent_id']}")
    validation = scene["compiled_validation"]
    minimum_approach = min(x["centerline_clearance_m"] for x in corridor_receipts)
    validation.update({"structural_aabb_non_overlap": True, "structural_checked_non_overlap_pairs": structural_pairs,
        "tabletop_aabb_non_overlap": True, "tabletop_checked_non_overlap_pairs": tabletop_pairs,
        "support_relations_consistent": True, "support_receipts": support_receipts,
        "approach_volume_clear": True, "minimum_approach_clearance_m": minimum_approach,
        "approach_obstacle_receipts": corridor_receipts, "avatar_spawn_clear": True,
        "avatar_spawn_receipts": avatar_receipts,
        "collision_free_initialization_scopes": ["structural_world_aabbs", "tabletop_world_aabbs", "support_relations", "avatar_spawn_sphere", "approach_capsule"],
        "collision_free_initialization_scopes_pass": True})
    validate_reachability(scene)
    return scene


def validate_reachability(scene: dict) -> dict:
    profile = scene["embodiment_reach_profile"]
    target = scene["target"]["transform"]["position_m"]
    shoulder, fingertip = profile["rest_shoulder_m"], profile["nominal_driven_fingertip_centroid_m"]
    distance = lambda a, b: sum((a[i] - b[i]) ** 2 for i in range(3)) ** .5
    shoulder_distance, fingertip_distance = distance(shoulder, target), distance(fingertip, target)
    checks = {
        "shoulder_radius_pass": profile["shoulder_target_radius_m"][0] <= shoulder_distance <= profile["shoulder_target_radius_m"][1],
        "right_lateral_limit_pass": profile["target_lateral_x_m"][0] <= target[0] <= profile["target_lateral_x_m"][1],
        "vertical_limit_pass": profile["target_vertical_y_m"][0] <= target[1] <= profile["target_vertical_y_m"][1],
        "forward_limit_pass": profile["target_forward_z_m"][0] <= target[2] <= profile["target_forward_z_m"][1],
        "nominal_fingertip_distance_pass": fingertip_distance <= profile["nominal_fingertip_target_max_m"],
    }
    receipt = {"profile_id": profile["profile_id"], "shoulder_to_target_m": round(shoulder_distance, 6), "nominal_fingertip_centroid_to_target_m": round(fingertip_distance, 6), **checks, "approach_direction": profile["approach_direction"], "compiled_reachability_pass": all(checks.values())}
    scene["compiled_validation"]["reachability"] = receipt
    if not receipt["compiled_reachability_pass"]:
        raise ValueError("target outside embodiment reach envelope")
    return scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile a bounded prompt into Unity-native EpisodeSpec and SceneSpec")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--room-family", required=True, choices=sorted(ROOMS))
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--profile", default="nominal")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    episode = compile_episode(args.prompt, args.seed, args.room_family, args.duration, args.profile)
    scene = compile_scene(episode)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "episode_spec.json").write_text(json.dumps(asdict(episode), indent=2) + "\n")
    (args.output / "scene_spec.json").write_text(json.dumps(scene, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
