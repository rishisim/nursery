from __future__ import annotations

from collections import OrderedDict
from math import hypot, sqrt
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional


FeatureGroups = Dict[str, Dict[str, Any]]

ARM_MODALITIES = OrderedDict(
    [
        ("vision", ("vision",)),
        ("vision_proprio", ("vision", "proprio")),
        ("vision_proprio_touch", ("vision", "proprio", "touch")),
        ("oracle_full_state", ("vision", "proprio", "touch", "oracle")),
    ]
)


def _as_record(episode: Any) -> Mapping[str, Any]:
    if hasattr(episode, "to_jsonable"):
        return episode.to_jsonable()
    if isinstance(episode, Mapping):
        return episode
    raise TypeError("episode must be a BabyWorld-Lite Episode or JSON-like mapping")


def _clamp_k(frames: Iterable[Mapping[str, Any]], k: int) -> int:
    frame_count = len(list(frames)) if not hasattr(frames, "__len__") else len(frames)  # type: ignore[arg-type]
    if frame_count <= 0:
        raise ValueError("episode has no frames")
    return max(0, min(int(k), frame_count - 1))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def _object_dict(record: Mapping[str, Any]) -> Mapping[str, Any]:
    obj = record["object"]
    if isinstance(obj, Mapping):
        return obj
    # Dataclasses are normally converted by _as_record, but this keeps the helper tolerant.
    return obj.__dict__


def prediction_frame(episode: Any) -> int:
    """Return the observable contact frame used as prediction time.

    Episodes with actual contact use the first tactile contact. Missed episodes
    use the first frame where the hand geometrically reaches the object radius,
    falling back to the closest approach if no such frame exists.
    """

    record = _as_record(episode)
    frames = record["frames"]
    for index, frame in enumerate(frames):
        if _float(frame["touch"].get("contact")) > 0.0:
            return index

    obj = _object_dict(record)
    radius = _float(obj.get("radius"))
    threshold = radius * 1.25
    closest_index = 0
    closest_distance = float("inf")
    for index, frame in enumerate(frames):
        hand = frame["hand"]
        state = frame["object"]
        distance = hypot(_float(hand.get("x")) - _float(state.get("x")), _float(hand.get("y")) - _float(state.get("y")))
        if distance < closest_distance:
            closest_distance = distance
            closest_index = index
        if distance <= threshold:
            return index
    return closest_index


def hidden_impulse(episode: Any) -> float:
    """Extract the simulator's hidden impulse for the positive leakage audit only."""

    record = _as_record(episode)
    for node in record.get("causal_graph", {}).get("nodes", []):
        if node.get("id") == "impulse":
            return _float(node.get("value"))
    return 0.0


def regression_targets(episode: Any, k: Optional[int] = None) -> Dict[str, float]:
    record = _as_record(episode)
    frames = record["frames"]
    pred_k = prediction_frame(record) if k is None else _clamp_k(frames, k)
    state_k = frames[pred_k]["object"]
    state_t = frames[-1]["object"]
    dx = _float(state_t.get("x")) - _float(state_k.get("x"))
    dy = _float(state_t.get("y")) - _float(state_k.get("y"))
    d_angle = _float(state_t.get("angle")) - _float(state_k.get("angle"))
    return {
        "target_delta_x": dx,
        "target_delta_y": dy,
        "target_displacement": hypot(dx, dy),
        "target_final_x": _float(state_t.get("x")),
        "target_final_y": _float(state_t.get("y")),
        "target_topple_angle": abs(d_angle),
    }


def extract_windowed_feature_groups(episode: Any, k: int, max_frames: Optional[int] = None) -> FeatureGroups:
    """Emit modality feature groups computable from frames [0, k] only.

    This deliberately ignores manifest columns and episode-wide tactile_summary
    values, including hardness_proxy.
    """

    record = _as_record(episode)
    frames = record["frames"]
    pred_k = _clamp_k(frames, k)
    frame_k = frames[pred_k]
    window = frames[: pred_k + 1]
    obj = _object_dict(record)
    max_frame_count = max_frames or len(frames)

    # The simulator appends frame k after applying contact dynamics. At the first
    # contact frame, use the immediately preceding visual object state so vision
    # does not see the consequence at the exact prediction instant.
    vision_state_index = pred_k
    if pred_k > 0 and _float(frame_k["touch"].get("contact")) > 0.0:
        prior_contacts = [_float(frame["touch"].get("contact")) > 0.0 for frame in frames[:pred_k]]
        if not any(prior_contacts):
            vision_state_index = pred_k - 1
    object_k = frames[vision_state_index]["object"]
    vision: Dict[str, Any] = {
        "vision_shape": obj.get("shape"),
        "vision_color": obj.get("color_name", obj.get("color")),
        "vision_radius": _float(obj.get("radius")),
        "vision_obj_x_k": _float(object_k.get("x")),
        "vision_obj_y_k": _float(object_k.get("y")),
        "vision_obj_vx_k": _float(object_k.get("vx")),
        "vision_obj_vy_k": _float(object_k.get("vy")),
        "vision_k": float(pred_k),
        "vision_object_state_frame": float(vision_state_index),
    }

    hand_xs: list[float] = []
    hand_ys: list[float] = []
    hand_speeds: list[float] = []
    proprio: Dict[str, Any] = {
        "proprio_action": record.get("action"),
        "proprio_k": float(pred_k),
    }
    previous_x: Optional[float] = None
    previous_y: Optional[float] = None
    path_length = 0.0
    for index, frame in enumerate(window):
        hand = frame["hand"]
        x = _float(hand.get("x"))
        y = _float(hand.get("y"))
        vx = _float(hand.get("vx"))
        vy = _float(hand.get("vy"))
        hand_xs.append(x)
        hand_ys.append(y)
        hand_speeds.append(hypot(vx, vy))
        if previous_x is not None and previous_y is not None:
            path_length += hypot(x - previous_x, y - previous_y)
        previous_x = x
        previous_y = y

    hand_k = frame_k["hand"]
    proprio.update(
        {
            "proprio_hand_x_k": _float(hand_k.get("x")),
            "proprio_hand_y_k": _float(hand_k.get("y")),
            "proprio_hand_vx_k": _float(hand_k.get("vx")),
            "proprio_hand_vy_k": _float(hand_k.get("vy")),
            "proprio_hand_speed_k": hypot(_float(hand_k.get("vx")), _float(hand_k.get("vy"))),
            "proprio_path_length": path_length,
            "proprio_mean_hand_x": _mean(hand_xs),
            "proprio_mean_hand_y": _mean(hand_ys),
            "proprio_std_hand_x": _std(hand_xs),
            "proprio_std_hand_y": _std(hand_ys),
            "proprio_mean_speed": _mean(hand_speeds),
            "proprio_max_speed": max(hand_speeds) if hand_speeds else 0.0,
        }
    )

    for index in range(max_frame_count):
        observed = index <= pred_k and index < len(frames)
        if observed:
            hand = frames[index]["hand"]
            proprio[f"proprio_hand_x_t{index:02d}"] = _float(hand.get("x"))
            proprio[f"proprio_hand_y_t{index:02d}"] = _float(hand.get("y"))
            proprio[f"proprio_hand_vx_t{index:02d}"] = _float(hand.get("vx"))
            proprio[f"proprio_hand_vy_t{index:02d}"] = _float(hand.get("vy"))
            proprio[f"proprio_hand_observed_t{index:02d}"] = 1.0
        else:
            proprio[f"proprio_hand_x_t{index:02d}"] = 0.0
            proprio[f"proprio_hand_y_t{index:02d}"] = 0.0
            proprio[f"proprio_hand_vx_t{index:02d}"] = 0.0
            proprio[f"proprio_hand_vy_t{index:02d}"] = 0.0
            proprio[f"proprio_hand_observed_t{index:02d}"] = 0.0

    forces: list[float] = []
    contact_forces: list[float] = []
    normal_xs: list[float] = []
    normal_ys: list[float] = []
    contact_count = 0.0
    slip_estimate = 0.0
    vibration_accum = 0.0
    previous_force: Optional[float] = None
    first_contact_force = 0.0

    for frame in window:
        touch_frame = frame["touch"]
        force = _float(touch_frame.get("force"))
        contact = _float(touch_frame.get("contact")) > 0.0
        forces.append(force)
        if previous_force is not None:
            vibration_accum += (force - previous_force) ** 2
        previous_force = force
        if not contact:
            continue
        contact_count += 1.0
        contact_forces.append(force)
        if first_contact_force == 0.0 and force > 0.0:
            first_contact_force = force
        nx = _float(touch_frame.get("normal_x"))
        ny = _float(touch_frame.get("normal_y"))
        normal_xs.append(nx)
        normal_ys.append(ny)

        hand = frame["hand"]
        vx = _float(hand.get("vx"))
        vy = _float(hand.get("vy"))
        speed = hypot(vx, vy)
        normal_mag = hypot(nx, ny)
        if normal_mag > 0.0:
            normal_speed = (vx * nx + vy * ny) / normal_mag
            tangential_speed = sqrt(max(0.0, speed**2 - normal_speed**2))
        else:
            tangential_speed = speed
        slip_estimate += tangential_speed / 20.0

    touch_k = frame_k["touch"]
    touch: Dict[str, Any] = {
        "touch_contact_at_k": _float(touch_k.get("contact")),
        "touch_force_at_k": _float(touch_k.get("force")),
        "touch_normal_x_at_k": _float(touch_k.get("normal_x")),
        "touch_normal_y_at_k": _float(touch_k.get("normal_y")),
        "touch_first_contact_force": first_contact_force,
        "touch_max_force": max(forces) if forces else 0.0,
        "touch_mean_force": _mean(forces),
        "touch_mean_contact_force": _mean(contact_forces),
        "touch_contact_count": contact_count,
        "touch_mean_normal_x": _mean(normal_xs),
        "touch_mean_normal_y": _mean(normal_ys),
        "touch_slip_estimate": slip_estimate,
        "touch_vibration_energy": sqrt(vibration_accum),
    }

    oracle: Dict[str, Any] = {
        "oracle_mass": _float(obj.get("mass")),
        "oracle_friction": _float(obj.get("friction")),
        "oracle_bounciness": _float(obj.get("bounciness")),
        "oracle_hardness": _float(obj.get("hardness")),
    }

    return {
        "vision": vision,
        "proprio": proprio,
        "touch": touch,
        "oracle": oracle,
    }


def flatten_feature_groups(groups: Mapping[str, Mapping[str, Any]], modalities: Iterable[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for modality in modalities:
        row.update(groups[modality])
    return row


def mutate_future_frames_for_test(record: MutableMapping[str, Any], k: int) -> None:
    """Helper used by tests to make future leakage obvious."""

    for frame in record["frames"][k + 1 :]:
        frame["hand"] = {"x": 999.0, "y": -999.0, "vx": 123.0, "vy": -456.0}
        frame["object"] = {"x": -888.0, "y": 777.0, "vx": 42.0, "vy": -42.0, "angle": 99.0}
        frame["touch"] = {"contact": 1.0, "force": 999.0, "normal_x": 0.5, "normal_y": -0.5}
    record["tactile_summary"] = {
        "first_contact_force": 999.0,
        "max_force": 999.0,
        "contact_count": 999.0,
        "slip_estimate": 999.0,
        "vibration_energy": 999.0,
        "hardness_proxy": 999.0,
    }
