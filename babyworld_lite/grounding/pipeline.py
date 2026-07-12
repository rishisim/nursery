from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw
import yaml

from babyworld_lite.grounding.config import load_grounding_config, validate_grounding_config
from babyworld_lite.sim import CANVAS, FPS, Episode, ObjectSpec, _draw_object, sample_object, simulate_episode


RAW_FRAME_POLICY = "scene_pixels_only_no_text_or_contact_marker"
MODEL_INPUT_ALLOWLIST = {
    "top": {"frame_paths", "utterances", "motor"},
    "utterance": {"text", "onset_frame"},
    "motor": {"t", "x", "y", "vx", "vy", "available"},
}
FORBIDDEN_MODEL_KEY_PARTS = {
    "action", "causal", "event", "force", "goal", "hidden", "impulse", "label",
    "material", "oracle", "shape", "target", "touch",
}


@dataclass
class BaseEpisode:
    episode: Episode
    split: str
    distractors: list[ObjectSpec]
    episode_hash: str
    activity_frames: int
    target_visible: bool
    target_occlusion_fraction: float
    camera_motion_amplitude_px: float


def _weighted_choice(rng: random.Random, weights: Mapping[Any, Any]) -> str:
    labels = [str(key) for key in weights]
    values = [float(value) for value in weights.values()]
    return rng.choices(labels, weights=values, k=1)[0]


def _holdouts(config: Mapping[str, Any], key: str) -> list[tuple[str, str]]:
    return [(str(item["shape"]), str(item["action"])) for item in config["splits"][key]]


def _split_for(shape: str, action: str, config: Mapping[str, Any]) -> str:
    composition = (shape, action)
    if composition in _holdouts(config, "test_compositions"):
        return "test"
    if composition in _holdouts(config, "validation_compositions"):
        return "validation"
    return "train"


def _matching_episode(
    episode_id: int,
    rng: random.Random,
    shape: str,
    action: str,
    material: str,
) -> Episode:
    # Rejection sampling keeps the existing physics implementation authoritative
    # while allowing the wrapper to impose configurable marginal distributions.
    start = rng.randrange(1, 2**31 - 1)
    for offset in range(20_000):
        candidate = simulate_episode(episode_id, start + offset * 1009)
        if (
            candidate.object.shape == shape
            and candidate.action == action
            and candidate.object.material == material
        ):
            return candidate
    raise RuntimeError(f"could not sample requested episode {shape}/{action}/{material}")


def _sample_distractors(
    rng: random.Random,
    target: ObjectSpec,
    count: int,
) -> list[ObjectSpec]:
    distractors: list[ObjectSpec] = []
    for _ in range(count):
        obj = sample_object(rng)
        for _attempt in range(100):
            obj.x = rng.uniform(28, CANVAS - 28)
            obj.y = rng.uniform(45, 132)
            others = [target, *distractors]
            if all((obj.x - other.x) ** 2 + (obj.y - other.y) ** 2 > (obj.radius + other.radius + 8) ** 2 for other in others):
                break
        distractors.append(obj)
    return distractors


def _episode_hash(ep: Episode) -> str:
    payload = {
        "seed": ep.seed,
        "object": asdict(ep.object),
        "frames": ep.frames,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _composition_schedule(config: Mapping[str, Any], rng: random.Random) -> list[tuple[str, str]]:
    n = int(config["generation"]["base_episodes"])
    minimum = int(config["splits"]["minimum_base_episodes_per_holdout"])
    forced: list[tuple[str, str]] = []
    for composition in [*_holdouts(config, "validation_compositions"), *_holdouts(config, "test_compositions")]:
        forced.extend([composition] * minimum)
    distributions = config["distributions"]
    while len(forced) < n:
        forced.append((
            _weighted_choice(rng, distributions["target_shapes"]),
            _weighted_choice(rng, distributions["actions"]),
        ))
    rng.shuffle(forced)
    return forced


def _build_bases(config: Mapping[str, Any]) -> list[BaseEpisode]:
    rng = random.Random(int(config["generation"]["seed"]))
    distributions = config["distributions"]
    calibration = config["calibration_targets"]
    bases: list[BaseEpisode] = []
    for episode_id, (shape, action) in enumerate(_composition_schedule(config, rng)):
        material = _weighted_choice(rng, distributions["materials"])
        episode = _matching_episode(episode_id, rng, shape, action, material)
        distractor_count = int(_weighted_choice(rng, distributions["distractor_count"]))
        bases.append(BaseEpisode(
            episode=episode,
            split=_split_for(shape, action, config),
            distractors=_sample_distractors(rng, episode.object, distractor_count),
            episode_hash=_episode_hash(episode),
            activity_frames=int(_weighted_choice(rng, calibration["activity_window_frames"])),
            target_visible=_weighted_choice(rng, calibration["target_visibility"]) == "visible",
            target_occlusion_fraction=float(_weighted_choice(rng, calibration["target_occlusion_fraction"])),
            camera_motion_amplitude_px=float(_weighted_choice(rng, calibration["camera_motion_amplitude_px"])),
        ))
    if len({base.split for base in bases}) != 3:
        raise ValueError("configuration/sample did not produce train, validation, and test bases")
    return bases


def _within_split_derangement(bases: Sequence[BaseEpisode]) -> list[int]:
    """Derange whole examples inside each split, preferring different actions."""
    mapping = [0] * len(bases)
    for split in ("train", "validation", "test"):
        group = [index for index, base in enumerate(bases) if base.split == split]
        if len(group) < 2:
            raise ValueError(f"{split} needs at least two base episodes for no-self cue shuffling")
        ordered = sorted(group, key=lambda i: bases[i].episode.action)
        counts = Counter(bases[index].episode.action for index in ordered)
        shift = max(counts.values()) if max(counts.values()) * 2 <= len(group) else 1
        donors = ordered[shift:] + ordered[:shift]
        for target, donor in zip(ordered, donors):
            mapping[target] = donor
        if any(target == mapping[target] for target in group):
            # A one-step rotation is always a no-self derangement for len >= 2.
            donors = ordered[1:] + ordered[:1]
            for target, donor in zip(ordered, donors):
                mapping[target] = donor
    if any(target == donor or bases[target].split != bases[donor].split for target, donor in enumerate(mapping)):
        raise AssertionError("failed to construct split-local no-self derangement")
    return mapping


def _contact_frame(ep: Episode) -> int:
    for frame in ep.frames:
        if float(frame["touch"]["contact"]) > 0:
            return int(frame["t"])
    return len(ep.frames) // 2


def _fit_utterance_length(text: str, target_words: int) -> str:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    words = words[:target_words]
    fillers = ["right", "now", "with", "me", "please", "over", "there"]
    while len(words) < target_words:
        words.append(fillers[(len(words) - 1) % len(fillers)])
    return " ".join(words) + "."


def _sample_utterance_count(base: BaseEpisode, config: Mapping[str, Any], rng: random.Random) -> int:
    calibration = config["calibration_targets"]
    target_rate = float(calibration["utterance_rate_per_minute"])
    expected_count = target_rate * (base.activity_frames / FPS) / 60.0
    adjusted_weights = {
        str(count): float(weight) * math.exp(-abs(int(count) - expected_count))
        for count, weight in calibration["utterances_per_activity_window"].items()
    }
    return int(_weighted_choice(rng, adjusted_weights))


def _language_packet(base: BaseEpisode, config: Mapping[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    ep = base.episode
    contact = _contact_frame(ep)
    calibration = config["calibration_targets"]
    count = _sample_utterance_count(base, config, rng)
    candidates = [
        ep.utterance_pre,
        "Can you do that to it?",
        rng.choice(list(config["weak_alignment"]["irrelevant_utterances"])),
        ep.utterance_post,
    ]
    packet: list[dict[str, Any]] = []
    previous_onset = contact
    for index in range(count):
        target_length = int(_weighted_choice(rng, calibration["utterance_length_words"]))
        if index == 0:
            onset = contact
        else:
            interval = int(_weighted_choice(rng, calibration["inter_utterance_interval_frames"]))
            onset = previous_onset - interval
        previous_onset = onset
        packet.append({
            "text": _fit_utterance_length(candidates[index % len(candidates)], target_length),
            "onset_frame": onset,
        })
    return packet


def _schedule_packet(
    packet: Sequence[Mapping[str, Any]],
    alignment: str,
    n_frames: int,
    config: Mapping[str, Any],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], str]:
    output = copy.deepcopy(list(packet))
    contact = int(packet[0]["onset_frame"])
    # Keep the same text inventory in every arm while changing timing/pairing.
    if alignment == "shuffled":
        for index, utterance in enumerate(output[1:], start=1):
            distance = max(1, abs(int(utterance["onset_frame"]) - contact))
            utterance["onset_frame"] = n_frames + distance if index % 2 else -distance
        return output, alignment
    if alignment == "strong":
        return output, alignment
    mixture = config["weak_alignment"]["mixture"]
    kind = _weighted_choice(rng, mixture)
    minimum = int(config["weak_alignment"]["min_delay_frames"])
    maximum = int(config["weak_alignment"]["max_delay_frames"])
    delay = rng.randint(minimum, maximum)
    if kind == "current":
        pass
    elif kind == "delayed":
        output[0]["onset_frame"] = n_frames + delay
    elif kind == "ambiguous":
        output[0]["onset_frame"] = n_frames + delay
        if len(output) > 1:
            output[1]["onset_frame"] = contact
    elif kind == "irrelevant":
        output[0]["onset_frame"] = n_frames + delay
        if len(output) > 2:
            output[2]["onset_frame"] = contact
    elif kind == "silent":
        # Text inventory is preserved, but the event-centered window is silent.
        output[0]["onset_frame"] = n_frames + delay
        for index, utterance in enumerate(output[1:], start=1):
            utterance["onset_frame"] = n_frames + delay + 3 * index
    return output, kind


def _raw_motor(ep: Episode, length: int | None = None) -> list[dict[str, Any]]:
    frames = ep.frames[:length] if length is not None else ep.frames
    return [
        {
            "t": int(frame["t"]),
            "x": round(float(frame["hand"]["x"]), 5),
            "y": round(float(frame["hand"]["y"]), 5),
            "vx": round(float(frame["hand"]["vx"]), 5),
            "vy": round(float(frame["hand"]["vy"]), 5),
            "available": 1,
        }
        for frame in frames
    ]


def _null_motor(length: int) -> list[dict[str, Any]]:
    return [{"t": t, "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "available": 0} for t in range(length)]


def _fit_motor_length(motor: Sequence[Mapping[str, Any]], length: int) -> list[dict[str, Any]]:
    output = [dict(sample) for sample in motor[:length]]
    if len(output) < length:
        output.extend(_null_motor(length - len(output)))
    for t, sample in enumerate(output):
        sample["t"] = t
    return output


def _shift_motor(motor: Sequence[Mapping[str, Any]], shift: int) -> list[dict[str, Any]]:
    result = _null_motor(len(motor))
    for t in range(len(motor)):
        source = t - shift
        if 0 <= source < len(motor):
            result[t] = dict(motor[source])
            result[t]["t"] = t
    return result


def validate_model_inputs(model_inputs: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    extra_top = set(model_inputs) - MODEL_INPUT_ALLOWLIST["top"]
    violations.extend(f"model_inputs.{key}" for key in sorted(extra_top))
    for i, utterance in enumerate(model_inputs.get("utterances", [])):
        for key in set(utterance) - MODEL_INPUT_ALLOWLIST["utterance"]:
            violations.append(f"model_inputs.utterances[{i}].{key}")
    for i, sample in enumerate(model_inputs.get("motor", [])):
        for key in set(sample) - MODEL_INPUT_ALLOWLIST["motor"]:
            violations.append(f"model_inputs.motor[{i}].{key}")
    return violations


def observable_leakage_paths(record: Mapping[str, Any]) -> list[str]:
    violations = validate_model_inputs(record.get("model_inputs", {}))

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(part in lowered for part in FORBIDDEN_MODEL_KEY_PARTS):
                    violations.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(record.get("model_inputs", {}), "model_inputs")
    return sorted(set(violations))


def _result_word(event: str) -> str:
    return {
        "rolls_far": "roll", "small_move": "move", "slides": "slide", "topples": "tip",
        "bounces": "bounce", "stays": "stay", "missed": "miss",
        "grasp_success": "pick up", "grasp_fail": "slip",
    }[event]


def _frame_paths(base: BaseEpisode, config: Mapping[str, Any]) -> list[str]:
    stride = int(config["generation"]["frame_stride"])
    return [f"frames/episode_{base.episode.episode_id:05d}/frame_{i:04d}.png" for i in range(0, base.activity_frames, stride)]


def _prepare(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[BaseEpisode]]:
    validate_grounding_config(config)
    bases = _build_bases(config)
    rng = random.Random(int(config["generation"]["seed"]) + 99173)
    language_donors = _within_split_derangement(bases)
    motor_donors = _within_split_derangement(bases)
    packets = [_language_packet(base, config, rng) for base in bases]
    examples: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    profile = {"name": config["profile"]["name"], "calibration_status": config["profile"]["calibration_status"]}
    for index, base in enumerate(bases):
        ep = base.episode
        for alignment in config["conditions"]["language_alignment"]:
            language_source = language_donors[index] if alignment == "shuffled" else index
            packet, weak_kind = _schedule_packet(
                packets[language_source], alignment, base.activity_frames, config, rng
            )
            for motor_condition in config["conditions"]["motor_cues"]:
                motor_source = index
                if motor_condition == "null":
                    motor = _null_motor(base.activity_frames)
                elif motor_condition == "synchronized":
                    motor = _raw_motor(ep, base.activity_frames)
                elif motor_condition == "shuffled":
                    motor_source = motor_donors[index]
                    motor = _fit_motor_length(
                        _raw_motor(bases[motor_source].episode, bases[motor_source].activity_frames),
                        base.activity_frames,
                    )
                elif motor_condition == "time_shifted":
                    motor = _shift_motor(
                        _raw_motor(ep, base.activity_frames),
                        int(config["conditions"]["motor_time_shift_frames"]),
                    )
                else:  # pragma: no cover - config validation prevents this
                    raise KeyError(motor_condition)
                example_id = f"ep{ep.episode_id:05d}-{alignment}-{motor_condition}"
                example = {
                    "schema_version": "grounding-v0",
                    "example_id": example_id,
                    "base_episode_id": ep.episode_id,
                    "episode_hash": base.episode_hash,
                    "split": base.split,
                    "profile": profile,
                    "alignment_condition": alignment,
                    "motor_condition": motor_condition,
                    "model_inputs": {
                        "frame_paths": _frame_paths(base, config),
                        "utterances": packet,
                        "motor": motor,
                    },
                    "evaluation_targets": {
                        "action_verb": ep.action,
                        "object_noun": ep.object.shape,
                        "color_adjective": ep.object.color_name,
                        "result_verb": _result_word(ep.event_label),
                    },
                }
                examples.append(example)
                oracle.append({
                    "example_id": example_id,
                    "base_episode_id": ep.episode_id,
                    "language_source_episode_id": bases[language_source].episode.episode_id,
                    "motor_source_episode_id": bases[motor_source].episode.episode_id if motor_condition != "null" else None,
                    "weak_alignment_kind": weak_kind,
                    "action": ep.action,
                    "event_label": ep.event_label,
                    "object": asdict(ep.object),
                    "hidden_goal": ep.hidden_goal,
                    "causal_graph": ep.causal_graph,
                    "counterfactuals": ep.counterfactuals,
                    "calibration_realization": {
                        "activity_window_frames": base.activity_frames,
                        "target_visible": base.target_visible,
                        "target_occlusion_fraction": base.target_occlusion_fraction,
                        "camera_motion_amplitude_px": base.camera_motion_amplitude_px,
                        "distractor_count": len(base.distractors),
                        "utterance_count": len(packets[index]),
                    },
                })
    return examples, oracle, bases


def build_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build model/evaluation records; oracle state is intentionally not returned."""
    examples, _oracle, _bases = _prepare(config)
    return examples


def render_raw_frame(base: BaseEpisode | Episode, frame_index: int, distractors: Sequence[ObjectSpec] = ()) -> Image.Image:
    ep = base.episode if isinstance(base, BaseEpisode) else base
    if isinstance(base, BaseEpisode):
        distractors = base.distractors
        visible = base.target_visible
        occlusion = base.target_occlusion_fraction
        amplitude = base.camera_motion_amplitude_px
    else:
        visible = True
        occlusion = 0.0
        amplitude = 0.0
    frame = ep.frames[frame_index]
    image = Image.new("RGB", (CANVAS, CANVAS), (245, 239, 225))
    draw = ImageDraw.Draw(image)
    phase = 2 * math.pi * frame_index / max(1, len(ep.frames) - 1)
    camera_x = amplitude * math.sin(phase)
    camera_y = amplitude * 0.5 * math.cos(phase)
    draw.rectangle([0, 150 + camera_y, CANVAS, CANVAS], fill=(232, 221, 204))
    for x in range(0, CANVAS, 28):
        draw.line([x + camera_x, 150 + camera_y, x + 20 + camera_x, CANVAS], fill=(219, 207, 190), width=1)
    for distractor in distractors:
        _draw_object(draw, distractor, distractor.x + camera_x, distractor.y + camera_y, 0.0)
    target_x = frame["object"]["x"] + camera_x
    target_y = frame["object"]["y"] + camera_y
    if visible:
        _draw_object(draw, ep.object, target_x, target_y, frame["object"]["angle"])
        if occlusion > 0:
            width = 2 * ep.object.radius * occlusion
            draw.rectangle(
                [
                    target_x + ep.object.radius - width,
                    target_y - ep.object.radius * 1.25,
                    target_x + ep.object.radius + 2,
                    target_y + ep.object.radius * 1.25,
                ],
                fill=(205, 197, 184),
            )
    hx, hy = frame["hand"]["x"] + camera_x, frame["hand"]["y"] + camera_y
    draw.ellipse([hx - 12, hy - 10, hx + 12, hy + 10], fill=(238, 174, 135), outline=(115, 72, 54), width=2)
    draw.line([hx, hy + 10, hx, CANVAS + 15], fill=(238, 174, 135), width=9)
    # Deliberately no caption and no artificial contact starburst.
    return image


def generate_grounding_dataset(config_or_path: Mapping[str, Any] | str | Path, out_dir: str | Path) -> dict[str, Any]:
    config = load_grounding_config(config_or_path) if isinstance(config_or_path, (str, Path)) else copy.deepcopy(dict(config_or_path))
    validate_grounding_config(config)
    examples, oracle, bases = _prepare(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if bool(config["generation"]["render_frames"]):
        stride = int(config["generation"]["frame_stride"])
        for base in bases:
            frame_dir = out / "frames" / f"episode_{base.episode.episode_id:05d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            for index in range(0, base.activity_frames, stride):
                render_raw_frame(base, index).save(frame_dir / f"frame_{index:04d}.png")
    (out / "examples.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in examples))
    (out / "oracle.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in oracle))
    (out / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    from babyworld_lite.grounding.audit import audit_records
    audit = audit_records(examples, oracle, config)
    (out / "audit_summary.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["valid"]:
        raise RuntimeError(f"generated dataset failed audit: {audit['failures']}")
    return {"output_dir": str(out), "examples": len(examples), "base_episodes": len(bases), "audit": audit}
