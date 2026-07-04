from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = 224
FPS = 12
N_FRAMES = 28

COLORS = {
    "red": (215, 54, 60),
    "blue": (55, 110, 215),
    "green": (74, 167, 96),
    "yellow": (235, 190, 63),
    "purple": (145, 87, 190),
}

MATERIALS = {
    # hardness, mass, friction, bounciness, tactile_noise
    "foam": {"hardness": 0.25, "mass": 0.6, "friction": 0.82, "bounciness": 0.08, "tactile_noise": 0.10},
    "wood": {"hardness": 0.70, "mass": 1.5, "friction": 0.55, "bounciness": 0.16, "tactile_noise": 0.04},
    "plastic": {"hardness": 0.55, "mass": 1.0, "friction": 0.38, "bounciness": 0.38, "tactile_noise": 0.06},
    "metal": {"hardness": 0.92, "mass": 2.1, "friction": 0.30, "bounciness": 0.20, "tactile_noise": 0.03},
}

SHAPES = {
    "ball": {"rollability": 1.0, "graspability": 0.45, "topple_threshold": 999.0},
    "block": {"rollability": 0.05, "graspability": 0.62, "topple_threshold": 999.0},
    "cup": {"rollability": 0.10, "graspability": 0.80, "topple_threshold": 0.78},
    "plush": {"rollability": 0.03, "graspability": 0.75, "topple_threshold": 999.0},
}

ACTIONS = ["tap", "push", "grasp", "poke"]


@dataclass
class ObjectSpec:
    shape: str
    color_name: str
    material: str
    x: float
    y: float
    radius: float
    mass: float
    friction: float
    bounciness: float
    hardness: float
    rollability: float
    graspability: float
    topple_threshold: float


@dataclass
class Episode:
    episode_id: int
    seed: int
    action: str
    utterance_pre: str
    utterance_post: str
    object: ObjectSpec
    hidden_goal: str
    event_label: str
    frames: List[Dict[str, Any]]
    tactile_summary: Dict[str, float]
    causal_graph: Dict[str, Any]
    counterfactuals: Dict[str, str]

    def to_jsonable(self) -> Dict[str, Any]:
        d = asdict(self)
        # keep numbers compact for JSONL readability
        for fr in d["frames"]:
            for key in ["hand", "object"]:
                fr[key] = {k: round(float(v), 4) if isinstance(v, (float, int)) else v for k, v in fr[key].items()}
            fr["touch"] = {k: round(float(v), 4) if isinstance(v, (float, int)) else v for k, v in fr["touch"].items()}
        return d


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def sample_object(r: random.Random) -> ObjectSpec:
    shape = r.choice(list(SHAPES.keys()))
    color_name = r.choice(list(COLORS.keys()))
    material = r.choice(list(MATERIALS.keys()))
    mat = MATERIALS[material]
    shp = SHAPES[shape]
    radius = r.uniform(15, 26) if shape != "cup" else r.uniform(18, 24)
    x = r.uniform(70, 154)
    y = r.uniform(64, 118)
    return ObjectSpec(
        shape=shape,
        color_name=color_name,
        material=material,
        x=x,
        y=y,
        radius=radius,
        mass=mat["mass"] * r.uniform(0.85, 1.18),
        friction=min(0.95, max(0.05, mat["friction"] + r.uniform(-0.08, 0.08))),
        bounciness=min(0.9, max(0.0, mat["bounciness"] + r.uniform(-0.06, 0.06))),
        hardness=min(1.0, max(0.05, mat["hardness"] + r.uniform(-0.06, 0.06))),
        rollability=shp["rollability"],
        graspability=shp["graspability"],
        topple_threshold=shp["topple_threshold"],
    )


def infer_event(action: str, obj: ObjectSpec, impulse: float, contact: bool) -> str:
    """Rule-based ground-truth causal label. This is intentionally transparent."""
    if not contact:
        return "missed"
    effective_force = impulse / max(0.2, obj.mass)
    if action == "grasp":
        success_score = obj.graspability + 0.25 * obj.friction - 0.10 * obj.mass
        return "grasp_success" if success_score > 0.55 else "grasp_fail"
    if obj.shape == "cup" and action in {"push", "poke"} and effective_force > obj.topple_threshold:
        return "topples"
    if obj.shape == "ball" and action in {"tap", "push", "poke"}:
        if effective_force * obj.rollability > 0.55:
            return "rolls_far"
        return "small_move"
    if effective_force > obj.friction * 0.75:
        return "slides"
    if obj.bounciness > 0.33 and action == "tap":
        return "bounces"
    return "stays"


def post_utterance(obj: ObjectSpec, event: str) -> str:
    templates = {
        "rolls_far": f"The {obj.color_name} {obj.shape} rolled away!",
        "small_move": f"The {obj.color_name} {obj.shape} moved a little.",
        "slides": f"The {obj.color_name} {obj.shape} slid on the floor.",
        "topples": f"Oops, the {obj.color_name} cup tipped over.",
        "bounces": f"The {obj.color_name} {obj.shape} bounced.",
        "stays": f"The {obj.color_name} {obj.shape} stayed there.",
        "missed": f"You missed the {obj.color_name} {obj.shape}.",
        "grasp_success": f"You picked up the {obj.color_name} {obj.shape}.",
        "grasp_fail": f"The {obj.color_name} {obj.shape} slipped away.",
    }
    return templates[event]


def simulate_episode(episode_id: int, seed: int) -> Episode:
    r = _rng(seed)
    obj = sample_object(r)
    action = r.choice(ACTIONS)
    # Hidden action parameter; tactile/contact exposes it later.
    intended_impulse = r.uniform(0.35, 1.55)
    miss_chance = 0.07 if action != "poke" else 0.12
    contact_happens = r.random() > miss_chance
    event = infer_event(action, obj, intended_impulse, contact_happens)

    utterance_pre = r.choice([
        f"Can you {action} the {obj.color_name} {obj.shape}?",
        f"Look at the {obj.color_name} {obj.shape}. Try to {action} it.",
        f"What happens if you {action} the {obj.shape}?",
    ])
    utterance_post = post_utterance(obj, event)

    frames: List[Dict[str, Any]] = []
    hand_start = np.array([CANVAS * 0.50 + r.uniform(-14, 14), CANVAS - 18.0])
    target = np.array([obj.x + r.uniform(-3, 3), obj.y + r.uniform(-3, 3)])
    obj_pos = np.array([obj.x, obj.y], dtype=float)
    obj_vel = np.array([0.0, 0.0], dtype=float)
    angle = 0.0
    contact_frame = 9 + r.randint(-1, 2)
    # action direction roughly upward/right/left from child's point of view
    direction = np.array([r.uniform(-0.75, 0.75), r.uniform(-0.35, 0.55)])
    norm = np.linalg.norm(direction) or 1.0
    direction = direction / norm
    previous_hand = hand_start.copy()

    first_contact_force = 0.0
    max_force = 0.0
    contact_count = 0
    slip_estimate = 0.0
    vibration_energy = 0.0

    for t in range(N_FRAMES):
        # Move hand from start to target, then follow through.
        if t < contact_frame:
            alpha = t / max(1, contact_frame)
            hand = (1 - alpha) * hand_start + alpha * target
        else:
            follow = min(1.0, (t - contact_frame) / max(1, N_FRAMES - contact_frame - 1))
            if action == "grasp":
                hand = target + np.array([0.0, -35.0 * follow])
            elif action == "tap":
                hand = target + direction * (10 * math.sin(follow * math.pi))
            else:
                hand = target + direction * (45 * follow)

        hand_vel = hand - previous_hand
        previous_hand = hand.copy()
        distance = float(np.linalg.norm(hand - obj_pos))
        is_contact = bool(contact_happens and t >= contact_frame and distance <= obj.radius * 1.25)
        contact_force = 0.0
        normal_x, normal_y = 0.0, 0.0
        if is_contact:
            contact_count += 1
            force_noise = r.uniform(0.85, 1.15)
            contact_force = intended_impulse * obj.hardness * force_noise
            max_force = max(max_force, contact_force)
            if first_contact_force == 0.0:
                first_contact_force = contact_force
            normal = obj_pos - hand
            n_norm = np.linalg.norm(normal) or 1.0
            normal = normal / n_norm
            normal_x, normal_y = float(normal[0]), float(normal[1])
            # tactile proxies: slip and vibration are not labels, but causal clues
            slip_estimate += float(max(0.0, 1.0 - obj.friction) * np.linalg.norm(hand_vel) / 20.0)
            vibration_energy += float(obj.bounciness * contact_force)

            if t == contact_frame or (action == "push" and t < contact_frame + 4):
                if event in {"rolls_far", "small_move", "slides", "topples", "bounces"}:
                    scalar = intended_impulse / max(0.2, obj.mass)
                    if event == "rolls_far":
                        scalar *= 6.0
                    elif event == "slides":
                        scalar *= 3.2
                    elif event == "bounces":
                        scalar *= 4.0
                    elif event == "small_move":
                        scalar *= 1.5
                    elif event == "topples":
                        scalar *= 2.0
                    obj_vel += direction * scalar

        if event == "grasp_success" and t > contact_frame:
            # object moves with the hand
            obj_pos = 0.65 * obj_pos + 0.35 * (hand + np.array([0.0, 8.0]))
            obj_vel *= 0.25
        else:
            # simple frictional dynamics
            obj_pos += obj_vel
            obj_vel *= max(0.0, 1.0 - obj.friction * 0.09)
            if obj_pos[0] < obj.radius or obj_pos[0] > CANVAS - obj.radius:
                obj_vel[0] *= -obj.bounciness
                obj_pos[0] = min(max(obj_pos[0], obj.radius), CANVAS - obj.radius)
            if obj_pos[1] < obj.radius or obj_pos[1] > CANVAS - obj.radius:
                obj_vel[1] *= -obj.bounciness
                obj_pos[1] = min(max(obj_pos[1], obj.radius), CANVAS - obj.radius)

        if event == "topples" and t > contact_frame:
            angle = min(math.pi / 2, angle + 0.16)
        elif obj.shape == "ball" and np.linalg.norm(obj_vel) > 0.1:
            angle += float(np.linalg.norm(obj_vel) * 0.05)

        frames.append({
            "t": t,
            "hand": {"x": float(hand[0]), "y": float(hand[1]), "vx": float(hand_vel[0]), "vy": float(hand_vel[1])},
            "object": {"x": float(obj_pos[0]), "y": float(obj_pos[1]), "vx": float(obj_vel[0]), "vy": float(obj_vel[1]), "angle": float(angle)},
            "touch": {"contact": float(is_contact), "force": float(contact_force), "normal_x": normal_x, "normal_y": normal_y},
        })

    tactile_summary = {
        "first_contact_force": first_contact_force,
        "max_force": max_force,
        "contact_count": float(contact_count),
        "slip_estimate": slip_estimate,
        "vibration_energy": vibration_energy,
        "hardness_proxy": max_force / max(0.1, intended_impulse),
    }
    causal_graph = {
        "nodes": [
            {"id": "action", "value": action},
            {"id": "object", "value": obj.shape},
            {"id": "material", "value": obj.material},
            {"id": "contact", "value": bool(contact_happens)},
            {"id": "impulse", "value": round(intended_impulse, 3)},
            {"id": "event", "value": event},
        ],
        "edges": [
            ["action", "contact"], ["contact", "event"], ["impulse", "event"],
            ["object", "event"], ["material", "event"],
        ],
        "formula_hint": "event = f(action, contact, impulse/mass, shape affordances, friction, bounciness)",
    }
    # cheap counterfactuals from same initial state, alternate actions
    counterfactuals = {}
    for alt in ACTIONS:
        if alt != action:
            counterfactuals[f"if_action_were_{alt}"] = infer_event(alt, obj, intended_impulse, contact_happens)

    return Episode(
        episode_id=episode_id,
        seed=seed,
        action=action,
        utterance_pre=utterance_pre,
        utterance_post=utterance_post,
        object=obj,
        hidden_goal="learn_action_effect",
        event_label=event,
        frames=frames,
        tactile_summary=tactile_summary,
        causal_graph=causal_graph,
        counterfactuals=counterfactuals,
    )


def _draw_object(draw: ImageDraw.ImageDraw, obj: ObjectSpec, x: float, y: float, angle: float) -> None:
    color = COLORS[obj.color_name]
    r = obj.radius
    if obj.shape == "ball":
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(30, 30, 30), width=2)
        # rolling stripe
        draw.line([x, y-r*0.65, x, y+r*0.65], fill=(255,255,255), width=2)
    elif obj.shape == "block":
        draw.rounded_rectangle([x-r, y-r, x+r, y+r], radius=4, fill=color, outline=(30,30,30), width=2)
    elif obj.shape == "cup":
        # rotate is approximated by a wider tilted bounding shape when toppling
        if angle > 0.5:
            draw.rounded_rectangle([x-r*1.25, y-r*0.55, x+r*1.25, y+r*0.55], radius=5, fill=color, outline=(30,30,30), width=2)
            draw.arc([x-r*1.25, y-r*0.55, x+r*1.25, y+r*0.55], 0, 360, fill=(255,255,255), width=2)
        else:
            draw.rounded_rectangle([x-r*0.75, y-r*1.2, x+r*0.75, y+r*1.2], radius=5, fill=color, outline=(30,30,30), width=2)
            draw.arc([x-r*0.75, y-r*1.2, x+r*0.75, y-r*0.65], 0, 360, fill=(255,255,255), width=2)
    elif obj.shape == "plush":
        draw.ellipse([x-r*1.15, y-r*0.8, x+r*1.15, y+r*0.8], fill=color, outline=(30,30,30), width=2)
        draw.ellipse([x-r*0.8, y-r*1.1, x-r*0.35, y-r*0.55], fill=color, outline=(30,30,30), width=1)
        draw.ellipse([x+r*0.35, y-r*1.1, x+r*0.8, y-r*0.55], fill=color, outline=(30,30,30), width=1)


def render_frame(ep: Episode, frame_index: int) -> Image.Image:
    fr = ep.frames[frame_index]
    img = Image.new("RGB", (CANVAS, CANVAS), (245, 239, 225))
    draw = ImageDraw.Draw(img)
    # simple table/floor cues
    draw.rectangle([0, 150, CANVAS, CANVAS], fill=(232, 221, 204))
    for x in range(0, CANVAS, 28):
        draw.line([x, 150, x+20, CANVAS], fill=(219, 207, 190), width=1)

    _draw_object(draw, ep.object, fr["object"]["x"], fr["object"]["y"], fr["object"]["angle"])

    hx, hy = fr["hand"]["x"], fr["hand"]["y"]
    draw.ellipse([hx-12, hy-10, hx+12, hy+10], fill=(238, 174, 135), outline=(115, 72, 54), width=2)
    draw.line([hx, hy+10, hx, CANVAS+15], fill=(238, 174, 135), width=9)
    if fr["touch"]["contact"] > 0:
        # starburst contact marker
        for k in range(8):
            ang = k * math.pi / 4
            draw.line([hx, hy, hx + math.cos(ang)*16, hy + math.sin(ang)*16], fill=(255, 120, 40), width=2)
    # overlay modality caption
    text = f"{ep.action} -> {ep.event_label} | f={fr['touch']['force']:.2f}"
    draw.rectangle([0, 0, CANVAS, 18], fill=(255,255,255))
    draw.text((4, 3), text, fill=(30,30,30))
    return img


def render_gif(ep: Episode, out_path: Path, every: int = 1) -> None:
    frames = [render_frame(ep, i) for i in range(0, len(ep.frames), every)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000/FPS), loop=0)


def write_episode(ep: Episode, out_dir: Path, render: bool = False) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = ep.to_jsonable()
    meta_path = out_dir / f"episode_{ep.episode_id:05d}.json"
    meta_path.write_text(json.dumps(record, indent=2))
    if render:
        gif_path = out_dir / f"episode_{ep.episode_id:05d}.gif"
        render_gif(ep, gif_path)
        record["gif_path"] = str(gif_path)
    return record


def flatten_for_manifest(ep: Episode) -> Dict[str, Any]:
    obj = ep.object
    fr0 = ep.frames[0]
    fr10 = ep.frames[min(12, len(ep.frames)-1)]
    last = ep.frames[-1]
    return {
        "episode_id": ep.episode_id,
        "seed": ep.seed,
        "shape": obj.shape,
        "color": obj.color_name,
        "material": obj.material,
        "action": ep.action,
        "event_label": ep.event_label,
        "utterance_pre": ep.utterance_pre,
        "utterance_post": ep.utterance_post,
        "mass": obj.mass,
        "friction": obj.friction,
        "bounciness": obj.bounciness,
        "hardness": obj.hardness,
        "rollability": obj.rollability,
        "graspability": obj.graspability,
        "initial_obj_x": fr0["object"]["x"],
        "initial_obj_y": fr0["object"]["y"],
        "early_obj_vx": fr10["object"]["vx"],
        "early_obj_vy": fr10["object"]["vy"],
        "final_obj_x": last["object"]["x"],
        "final_obj_y": last["object"]["y"],
        **ep.tactile_summary,
    }
