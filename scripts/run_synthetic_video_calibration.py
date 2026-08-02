#!/usr/bin/env python3
"""Governed bounded C calibration and aggregate-conditioned episode planning."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
from typing import Any

from nursery_egobaby_preflight.contract import compact_aggregate_json


TERMINAL_FIELDS = frozenset(
    {
        "status",
        "axis_count",
        "joint_count",
        "sampled_frame_count",
        "grounding_event_count",
        "missing_axis_count",
        "suppressed_cell_count",
        "episode_plan_count",
        "calibration_commitment_sha256",
        "episode_plan_commitment_sha256",
    }
)
TERMINAL_HASH_FIELDS = frozenset(
    {"calibration_commitment_sha256", "episode_plan_commitment_sha256"}
)
ACTIVITY_AXIS = "activity_context_mixture"
VISUAL_AXIS = "egocentric_visual_regime"
SCENE_AXIS = "scene_complexity"
HAND_AXIS = "hand_object_action_structure"
TEMPORAL_AXIS = "temporal_continuity_and_recurrence"
LANGUAGE_AXIS = "language_environment"
GROUNDING_AXIS = "audiovisual_grounding_opportunity"
DIVERSITY_AXIS = "diversity_and_heterogeneity"
AXES = (
    ACTIVITY_AXIS,
    VISUAL_AXIS,
    SCENE_AXIS,
    HAND_AXIS,
    TEMPORAL_AXIS,
    LANGUAGE_AXIS,
    GROUNDING_AXIS,
    DIVERSITY_AXIS,
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical(value) + b"\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def bucket(value: float, edges: list[float]) -> str:
    if not math.isfinite(value):
        raise ValueError("bucket value must be finite")
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return f"bin_{index}"
    if value == edges[-1]:
        return f"bin_{len(edges) - 2}"
    raise ValueError("bucket value is outside frozen edges")


def union_duration(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def classify_speech_act(text: str, rules: dict[str, Any]) -> str:
    normalized = " ".join(text.lower().strip().split())
    stripped = normalized.rstrip(".?!")
    first = stripped.split(maxsplit=1)[0] if stripped else ""
    if any(stripped.startswith(prefix) for prefix in rules["naming_prefixes"]):
        return "naming"
    if "?" in normalized or first in rules["question_mark_or_public_initial_words"]:
        return "question"
    if first in rules["directive_first_words"]:
        return "directive"
    return "other"


def _allocated_labels(
    proportions: dict[str, float], count: int, seed: int, namespace: str
) -> list[str]:
    usable = {key: max(0.0, float(value)) for key, value in proportions.items()}
    total = sum(usable.values())
    if not usable or total <= 0:
        raise ValueError(f"no allocation support for {namespace}")
    exact = {key: value / total * count for key, value in usable.items()}
    allocated = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = count - sum(allocated.values())
    order = sorted(
        usable,
        key=lambda key: (-(exact[key] - allocated[key]), digest([seed, namespace, key])),
    )
    for key in order[:remainder]:
        allocated[key] += 1
    tagged = [
        (key, ordinal)
        for key in sorted(allocated)
        for ordinal in range(allocated[key])
    ]
    tagged.sort(key=lambda pair: digest([seed, namespace, pair[0], pair[1]]))
    return [key for key, _ in tagged]


def _duration(ffmpeg: str, source: Path) -> float:
    completed = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-i", str(source)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    match = re.search(
        rb"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr
    )
    if match is None:
        raise RuntimeError("E_MEDIA_DURATION")
    return int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])


def _decode_frame(ffmpeg: str, source: Path, timestamp: float, size: int):
    from PIL import Image

    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, timestamp):.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            f"scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    if completed.returncode or not completed.stdout:
        return None
    try:
        with Image.open(BytesIO(completed.stdout)) as image:
            return image.convert("RGB").copy()
    except Exception:
        return None


def _image_metrics(image, previous) -> dict[str, float | None]:
    import numpy as np

    array = np.asarray(image, dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_strength = float((dx.mean() + dy.mean()) / 2)
    edge_fraction = float(
        ((dx[:, :-1] + dy[:-1, :]) / 2 > 0.12).mean()
    )
    motion = None
    if previous is not None:
        prior = np.asarray(previous, dtype=np.float32).mean(axis=2) / 255.0
        motion = float(np.abs(gray - prior).mean())
    return {
        "brightness": float(gray.mean()),
        "blur_edge_strength": edge_strength,
        "clutter_edge_fraction": edge_fraction,
        "motion_mean_absolute_luma": motion,
    }


def _load_vision(public: Path, cfg: dict[str, Any], device: str):
    import torch
    from apps.alignment_scoring.third_party.perception_models.core.vision_encoder import (
        pe,
        transforms,
    )

    frozen = cfg["calibration_C"]["extractor"]["vision_model"]
    candidates = list(
        (public / "models/pe-hf-home/hub").glob(
            f"models--facebook--PE-Core-L14-336/snapshots/{frozen['revision']}/PE-Core-L14-336.pt"
        )
    )
    if len(candidates) != 1 or file_digest(candidates[0]) != frozen["weights_sha256"]:
        raise RuntimeError("E_FROZEN_VISION_MODEL")
    model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=False)
    model.load_ckpt(str(candidates[0]))
    model = model.to(device).eval()
    transform = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    prompt_groups = cfg["calibration_C"]["extractor"]["regular_frame_prompt_groups"]
    flattened: list[str] = []
    slices: dict[str, tuple[int, int, list[str]]] = {}
    for group, prompts in prompt_groups.items():
        start = len(flattened)
        labels = list(prompts)
        flattened.extend(prompts[label] for label in labels)
        slices[group] = (start, len(flattened), labels)
    with torch.inference_mode():
        text = model.encode_text(tokenizer(flattened).to(device), normalize=True).cpu()
    return model, transform, text, slices


def _vision_batch(model, transform, text, slices, images, margin: float, device: str):
    import torch

    with torch.inference_mode():
        batch = torch.stack([transform(image) for image in images]).to(device)
        features = model.encode_image(batch, normalize=True).cpu()
        scores = features @ text.T
    labels: list[dict[str, str | None]] = []
    for row in scores:
        values: dict[str, str | None] = {}
        for group, (start, stop, names) in slices.items():
            part = row[start:stop]
            order = torch.argsort(part, descending=True)
            top = int(order[0])
            gap = float(part[top] - part[int(order[1])]) if len(order) > 1 else 1.0
            values[group] = names[top] if gap >= margin else None
        labels.append(values)
    return features, labels


def _bootstrap_categorical(
    observations: list[tuple[str, str | None]], replicates: int, seed: int
) -> dict[str, Any]:
    import numpy as np

    present = [(child, value) for child, value in observations if value is not None]
    counts = Counter(value for _, value in present)
    support = sum(counts.values())
    proportions = {
        label: count / support for label, count in sorted(counts.items())
    } if support else {}
    by_child: dict[str, list[str]] = defaultdict(list)
    for child, value in present:
        by_child[child].append(value)
    children = sorted(by_child)
    intervals: dict[str, list[float]] = {}
    if children and counts:
        rng = np.random.default_rng(seed)
        draws: dict[str, list[float]] = {label: [] for label in counts}
        for _ in range(replicates):
            chosen = rng.choice(children, size=len(children), replace=True)
            sampled = Counter(
                value for child in chosen for value in by_child[str(child)]
            )
            denominator = sum(sampled.values())
            for label in draws:
                draws[label].append(sampled[label] / denominator if denominator else 0.0)
        intervals = {
            label: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for label, values in draws.items()
        }
    return {
        "support": support,
        "missing": len(observations) - support,
        "counts": dict(sorted(counts.items())),
        "proportions": proportions,
        "child_cluster_bootstrap_95ci": intervals,
    }


def _add(observations, axis: str, feature: str, child: str, value: str | None):
    observations[axis][feature].append((child, value))


def _summarize(
    observations: dict[str, dict[str, list[tuple[str, str | None]]]], cfg: dict[str, Any]
) -> tuple[dict[str, Any], int, int]:
    uncertainty = cfg["calibration_C"]["extractor"]["uncertainty"]
    missing_cfg = cfg["calibration_C"]["extractor"]["missingness"]
    sparse_min = cfg["calibration_C"]["extractor"]["disclosure_review"][
        "minimum_cell_count_for_external_numeric_export"
    ]
    targets: dict[str, Any] = {}
    missing_axes = 0
    suppressed = 0
    for axis in AXES:
        features = {}
        axis_observed = axis_total = 0
        for feature, values in sorted(observations[axis].items()):
            summary = _bootstrap_categorical(
                values, uncertainty["replicates"], uncertainty["seed"]
            )
            features[feature] = summary
            axis_observed += summary["support"]
            axis_total += summary["support"] + summary["missing"]
            suppressed += sum(
                count < sparse_min for count in summary["counts"].values()
            )
        missing_fraction = (
            1.0 - axis_observed / axis_total if axis_total else 1.0
        )
        status = "MEASURED"
        feature_support = [feature["support"] for feature in features.values()]
        insufficient_features = sum(
            support < missing_cfg["minimum_observations_per_feature"]
            for support in feature_support
        )
        if not feature_support or max(feature_support) < missing_cfg["minimum_observations_per_feature"]:
            status = "INSUFFICIENT_SUPPORT"
        elif missing_fraction > missing_cfg["maximum_axis_missing_fraction"]:
            status = "HIGH_MISSINGNESS"
        elif insufficient_features:
            status = "MEASURED_WITH_DECLARED_PARTIAL_FEATURES"
        if status in {"INSUFFICIENT_SUPPORT", "HIGH_MISSINGNESS"}:
            missing_axes += 1
        targets[axis] = {
            "status": status,
            "model_derived": axis in {ACTIVITY_AXIS, SCENE_AXIS, HAND_AXIS, GROUNDING_AXIS},
            "missing_fraction": missing_fraction,
            "insufficient_feature_count": insufficient_features,
            "features": features,
        }
    return targets, missing_axes, suppressed


def _proportions(targets: dict[str, Any], axis: str, feature: str, fallback):
    values = targets[axis]["features"].get(feature, {}).get("proportions", {})
    values = {key: value for key, value in values.items() if key != "ABSTAIN"}
    return values or fallback


def _speech_line(kind: str, noun: dict[str, str]) -> str:
    if kind == "naming":
        return f"Schau mal, das ist {noun['article_nom']} {noun['de']}."
    if kind == "directive":
        return f"Bitte nimm {noun['article_acc']} {noun['de']} und halte den Gegenstand gut fest."
    if kind == "question":
        return f"Wo ist {noun['article_nom']} {noun['de']}? Zeig mir den Gegenstand."
    return f"Wir schauen jetzt gemeinsam auf {noun['article_acc']} {noun['de']}."


def _build_episode_plans(
    cfg: dict[str, Any], targets: dict[str, Any], calibration_commitment: str
) -> dict[str, Any]:
    plan_cfg = cfg["calibration_C"]["episode_plan"]
    count = plan_cfg["candidate_plan_count"]
    seed = plan_cfg["plan_seed"]
    activity = _allocated_labels(
        _proportions(targets, ACTIVITY_AXIS, "activity", {"other": 1.0}),
        count,
        seed,
        "activity",
    )
    hand = _allocated_labels(
        _proportions(targets, HAND_AXIS, "hand_action", {"manipulate": 1.0}),
        count,
        seed,
        "hand_action",
    )
    framing = _allocated_labels(
        _proportions(targets, VISUAL_AXIS, "framing", {"centered": 1.0}),
        count,
        seed,
        "framing",
    )
    occlusion = _allocated_labels(
        _proportions(targets, SCENE_AXIS, "occlusion", {"clear": 1.0}),
        count,
        seed,
        "occlusion",
    )
    distractors = _allocated_labels(
        _proportions(targets, SCENE_AXIS, "distractors", {"few": 1.0}),
        count,
        seed,
        "distractors",
    )
    acts = _allocated_labels(
        _proportions(targets, LANGUAGE_AXIS, "speech_act", {"naming": 1.0}),
        count * plan_cfg["utterances_per_plan"],
        seed,
        "speech_act",
    )
    contexts = plan_cfg["public_contexts"]
    actions = plan_cfg["public_actions"]
    lexicon = plan_cfg["public_lexicon"]
    adjectives = plan_cfg["public_adjectives"]
    rows = []
    for index in range(count):
        noun = lexicon[index % len(lexicon)]
        adjective = adjectives[(index // len(lexicon)) % len(adjectives)]
        context = contexts[(index // (len(lexicon) * len(adjectives))) % len(contexts)]
        action = actions[index % len(actions)]
        visual_prompt = (
            "Photorealistic naturalistic home-video footage in one continuous uncut "
            "first-person egocentric child-height head-camera shot. "
            f"An ordinary {context} during {activity[index].replace('_', ' ')}. "
            f"A {adjective['en']} {noun['en']} remains the same persistent object. "
            f"The child-view action is {action}; hand-action regime {hand[index].replace('_', ' ')}. "
            f"Framing is {framing[index]}, occlusion is {occlusion[index]}, and there are {distractors[index]} distractors. "
            "Use plausible head motion, ordinary lighting, stable object identity, realistic contact and chronological action order. "
            "No cuts, captions, text, logos, mirrors, identifiable faces, unsafe behavior, dialogue, lyrics, or music. "
            "Quiet room ambience only; generated audio will be discarded."
        )
        utterance_kinds = acts[index * 2 : index * 2 + 2]
        utterances = [_speech_line(kind, noun) for kind in utterance_kinds]
        rows.append(
            {
                "plan_key": digest([seed, "episode", index]),
                "seed": int(digest([seed, "ltx", index])[:8], 16) % 2147483647,
                "duration_seconds": plan_cfg["clip_seconds"],
                "visual_prompt_en": visual_prompt,
                "utterances_de": utterances,
                "utterance_kinds": utterance_kinds,
                "utterance_onsets_seconds": plan_cfg["utterance_onsets_seconds"],
                "public_provenance": {
                    "wordnet": noun["wordnet"],
                    "public_pool_only": True,
                },
                "target_labels": {
                    "activity": activity[index],
                    "hand_action": hand[index],
                    "framing": framing[index],
                    "occlusion": occlusion[index],
                    "distractors": distractors[index],
                },
            }
        )
    value = {
        "schema_version": 1,
        "status": "FROZEN",
        "calibration_commitment_sha256": calibration_commitment,
        "plan_seed": seed,
        "selection": plan_cfg["selection"],
        "evaluation_steering": False,
        "plans": rows,
    }
    value["episode_plan_commitment_sha256"] = digest(value)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    import imageio_ffmpeg
    import numpy as np

    cfg = json.loads(args.config.read_text())
    extractor = cfg["calibration_C"]["extractor"]
    stage = json.loads((args.restricted_root / "restricted_stage_manifest.json").read_text())
    checkpoint = json.loads(
        (args.restricted_root / "asr/restricted_calibration.json").read_text()
    )
    rows = stage.get("calibration")
    items = checkpoint.get("items")
    if not isinstance(rows, list) or not isinstance(items, dict) or len(rows) != len(items):
        raise RuntimeError("E_CALIBRATION_INPUT")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    model, transform, text_features, slices = _load_vision(
        args.public_root, cfg, args.device
    )
    observations: dict[str, dict[str, list[tuple[str, str | None]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    restricted_features = []
    all_features = []
    all_recording_indices = []
    sampled_frames = grounding_events = 0
    recording_index = 0
    bins = extractor["fixed_numeric_bins"]
    margin = extractor["vision_abstention_top1_margin_min"]
    scratch = args.scratch_root
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    for row in sorted(rows, key=lambda value: value["asset_key"]):
        asset_key = row["asset_key"]
        item = items.get(asset_key)
        if not isinstance(item, dict):
            raise RuntimeError("E_CALIBRATION_CHECKPOINT")
        source = args.restricted_root / row["file"]
        local = scratch / f"{digest([asset_key, 'media'])}{source.suffix.lower()}"
        shutil.copy2(source, local)
        try:
            duration = _duration(ffmpeg, local)
            regular_times = []
            timestamp = extractor["frame_sample_interval_seconds"] / 2
            while timestamp < duration and len(regular_times) < extractor["max_regular_frames_per_recording"]:
                regular_times.append(timestamp)
                timestamp += extractor["frame_sample_interval_seconds"]
            images = []
            decoded_times = []
            metrics = []
            previous = None
            for timestamp in regular_times:
                image = _decode_frame(ffmpeg, local, timestamp, extractor["decoded_image_size"])
                if image is None:
                    continue
                images.append(image)
                decoded_times.append(timestamp)
                metrics.append(_image_metrics(image, previous))
                previous = image
            if images:
                features, labels = _vision_batch(
                    model, transform, text_features, slices, images, margin, args.device
                )
            else:
                features = np.empty((0, 1), dtype=np.float32)
                labels = []
            for index, (timestamp, metric, label) in enumerate(zip(decoded_times, metrics, labels, strict=True)):
                child = row["child_key"]
                _add(observations, ACTIVITY_AXIS, "activity", child, label["activity"])
                _add(observations, VISUAL_AXIS, "framing", child, label["framing"])
                _add(observations, SCENE_AXIS, "occlusion", child, label["occlusion"])
                _add(observations, SCENE_AXIS, "distractors", child, label["distractors"])
                _add(observations, HAND_AXIS, "hand_action", child, label["hand_action"])
                _add(observations, GROUNDING_AXIS, "regular_referent", child, label["referent"])
                for feature in ("brightness", "blur_edge_strength", "clutter_edge_fraction"):
                    axis = SCENE_AXIS if feature == "clutter_edge_fraction" else VISUAL_AXIS
                    _add(observations, axis, feature, child, bucket(float(metric[feature]), bins[feature]))
                motion = metric["motion_mean_absolute_luma"]
                motion_bin = bucket(float(motion), bins["motion_mean_absolute_luma"]) if motion is not None else None
                _add(observations, VISUAL_AXIS, "motion", child, motion_bin)
                transition = None
                if motion is not None:
                    transition = "idle" if motion < 0.03 else "transition" if motion < 0.25 else "shot_discontinuity"
                _add(observations, TEMPORAL_AXIS, "idle_transition", child, transition)
                speech_here = any(
                    segment.get("status") == "ACCEPT" and segment["start"] <= timestamp <= segment["end"]
                    for segment in item.get("segments", [])
                )
                null_label = f"{'speech' if speech_here else 'silence'}|{label['referent'] or 'abstain'}"
                _add(observations, GROUNDING_AXIS, "speech_referent_null", child, null_label)
                restricted_features.append({
                    "asset_key": asset_key,
                    "child_key": child,
                    "session_key": row["session_key"],
                    "timestamp": timestamp,
                    "metrics": metric,
                    "labels": label,
                })
                sampled_frames += 1
            if len(features):
                features_np = features.numpy()
                all_features.extend(features_np)
                all_recording_indices.extend([recording_index] * len(features_np))
                for index in range(1, len(features_np)):
                    similarity = float(features_np[index] @ features_np[index - 1])
                    _add(observations, TEMPORAL_AXIS, "adjacent_similarity", row["child_key"], bucket(similarity, bins["adjacent_feature_similarity"]))
                for index in range(2, len(features_np)):
                    recurrence = float(np.max(features_np[index] @ features_np[: index - 1].T))
                    _add(observations, TEMPORAL_AXIS, "recurrence", row["child_key"], bucket(recurrence, bins["adjacent_feature_similarity"]))
            accepted = [segment for segment in item.get("segments", []) if segment.get("status") == "ACCEPT"]
            intervals = [(max(0.0, float(s["start"])), min(duration, float(s["end"]))) for s in accepted]
            speech_fraction = union_duration(intervals) / duration if duration > 0 else 0.0
            _add(observations, LANGUAGE_AXIS, "speech_fraction", row["child_key"], bucket(speech_fraction, bins["speech_fraction"]))
            normalized_counts = Counter(" ".join(str(s.get("en", "")).lower().split()) for s in accepted)
            previous_end = None
            speech_rows = []
            for index, segment in enumerate(accepted):
                start, stop = float(segment["start"]), float(segment["end"])
                text = str(segment.get("en", ""))
                act = classify_speech_act(text, extractor["language_rules"])
                _add(observations, LANGUAGE_AXIS, "utterance_duration", row["child_key"], bucket(max(0.0, stop - start), bins["utterance_duration_seconds"]))
                _add(observations, LANGUAGE_AXIS, "utterance_word_count", row["child_key"], bucket(float(len(text.split())), bins["utterance_word_count"]))
                _add(observations, LANGUAGE_AXIS, "speech_act", row["child_key"], act)
                repeated = "repeated" if normalized_counts[" ".join(text.lower().split())] > 1 else "unique"
                _add(observations, LANGUAGE_AXIS, "repetition", row["child_key"], repeated)
                if previous_end is not None:
                    _add(observations, LANGUAGE_AXIS, "pause_proxy", row["child_key"], bucket(max(0.0, start - previous_end), bins["pause_seconds"]))
                previous_end = max(previous_end or stop, stop)
                speech_rows.append((index, segment, act))
            ordered_events = sorted(speech_rows, key=lambda value: digest([asset_key, value[0], 42]))[: extractor["max_naming_events_per_recording"]]
            grounding_images = []
            grounding_meta = []
            grounding_metrics = []
            for event_ordinal, (index, segment, act) in enumerate(ordered_events):
                midpoint = (float(segment["start"]) + float(segment["end"])) / 2
                for position, offset in zip(("before", "during", "after"), extractor["naming_offsets_seconds"], strict=True):
                    image = _decode_frame(ffmpeg, local, min(max(0.0, midpoint + offset), max(0.0, duration - 0.01)), extractor["decoded_image_size"])
                    if image is not None:
                        grounding_images.append(image)
                        grounding_meta.append((event_ordinal, position, act))
                        grounding_metrics.append(_image_metrics(image, None))
            if grounding_images:
                _, grounding_labels = _vision_batch(model, transform, text_features, slices, grounding_images, margin, args.device)
                grouped: dict[int, list[tuple[str, str, dict[str, str | None], dict[str, float | None]]]] = defaultdict(list)
                for meta, label, metric in zip(grounding_meta, grounding_labels, grounding_metrics, strict=True):
                    grouped[meta[0]].append((meta[1], meta[2], label, metric))
                for event_index, group in grouped.items():
                    for position, act, label, metric in group:
                        if act == "naming":
                            _add(observations, GROUNDING_AXIS, f"naming_referent_{position}", row["child_key"], label["referent"])
                        if position == "during":
                            _add(observations, GROUNDING_AXIS, "naming_by_referent_visibility", row["child_key"], f"{act == 'naming'}|{label['referent'] or 'abstain'}")
                            _add(observations, GROUNDING_AXIS, "naming_by_hand_action", row["child_key"], f"{act == 'naming'}|{label['hand_action'] or 'abstain'}")
                            clutter = bucket(float(metric["clutter_edge_fraction"]), bins["clutter_edge_fraction"])
                            _add(observations, SCENE_AXIS, "clutter_by_occlusion", row["child_key"], f"{clutter}|{label['occlusion'] or 'abstain'}")
                            grounding_events += 1
            _add(observations, LANGUAGE_AXIS, "overlap", row["child_key"], "present" if sum(stop-start for start,stop in intervals) - union_duration(intervals) > 0.01 else "absent")
        finally:
            local.unlink(missing_ok=True)
        recording_index += 1
    if all_features:
        feature_matrix = np.asarray(all_features, dtype=np.float32)
        record_ids = np.asarray(all_recording_indices)
        near = np.zeros(len(feature_matrix), dtype=bool)
        for start in range(0, len(feature_matrix), 256):
            scores = feature_matrix[start : start + 256] @ feature_matrix.T
            same = record_ids[start : start + 256, None] == record_ids[None, :]
            scores[same] = -1.0
            near[start : start + len(scores)] = scores.max(axis=1) >= 0.98
        for value, row in zip(near.tolist(), restricted_features, strict=True):
            _add(observations, DIVERSITY_AXIS, "cross_recording_near_duplicate", row["child_key"], "near_duplicate" if value else "unique")
        for row in restricted_features:
            _add(observations, DIVERSITY_AXIS, "activity_long_tail", row["child_key"], row["labels"]["activity"])
        activity_counts = Counter(
            row["labels"]["activity"]
            for row in restricted_features
            if row["labels"]["activity"] is not None
        )
        activity_total = sum(activity_counts.values())
        global_activity = {
            label: count / activity_total for label, count in activity_counts.items()
        } if activity_total else {}
        by_child = defaultdict(list)
        by_session = defaultdict(list)
        session_child = {}
        for row in restricted_features:
            label = row["labels"]["activity"]
            if label is None:
                continue
            by_child[row["child_key"]].append(label)
            by_session[row["session_key"]].append(label)
            session_child[row["session_key"]] = row["child_key"]
            tail = "long_tail" if global_activity.get(label, 0.0) < 0.05 else "common"
            _add(observations, DIVERSITY_AXIS, "long_tail_coverage", row["child_key"], tail)
        dispersion_edges = [0.0, 0.1, 0.2, 0.4, 1.000001]
        for child, labels in by_child.items():
            local_counts = Counter(labels)
            local_total = sum(local_counts.values())
            tv = 0.5 * sum(
                abs(local_counts[label] / local_total - global_activity.get(label, 0.0))
                for label in set(local_counts) | set(global_activity)
            )
            _add(observations, DIVERSITY_AXIS, "between_child_dispersion", child, bucket(tv, dispersion_edges))
        for session, labels in by_session.items():
            local_counts = Counter(labels)
            local_total = sum(local_counts.values())
            tv = 0.5 * sum(
                abs(local_counts[label] / local_total - global_activity.get(label, 0.0))
                for label in set(local_counts) | set(global_activity)
            )
            _add(observations, DIVERSITY_AXIS, "between_session_dispersion", session_child[session], bucket(tv, dispersion_edges))
    for row in restricted_features:
        motion = row["metrics"]["motion_mean_absolute_luma"]
        motion_label = bucket(float(motion), bins["motion_mean_absolute_luma"]) if motion is not None else "missing"
        blur_label = bucket(float(row["metrics"]["blur_edge_strength"]), bins["blur_edge_strength"])
        _add(observations, VISUAL_AXIS, "motion_by_blur", row["child_key"], f"{motion_label}|{blur_label}")
    targets, missing_axes, suppressed = _summarize(observations, cfg)
    measurement = {
        "schema_version": 1,
        "status": "PASS" if missing_axes == 0 else "PASS_WITH_DECLARED_MISSINGNESS",
        "source": "development_set_C_only",
        "extractor_contract": extractor,
        "measured_axes": list(AXES),
        "unmeasured": ["human_activity_labels", "speaker_diarization", "human_referent_ground_truth", "exact_object_counts", "full_distributional_fidelity"],
        "targets": targets,
        "joint_features": ["naming_by_referent_visibility", "naming_by_hand_action", "clutter_by_occlusion", "motion_by_blur"],
        "sampled_frame_count": sampled_frames,
        "grounding_event_count": grounding_events,
        "ffmpeg_sha256": file_digest(Path(ffmpeg)),
        "row_level_features_retained_governed_only": True,
        "external_target_values_exported": False,
        "omnibus_score": None,
    }
    measurement["calibration_commitment_sha256"] = digest(measurement)
    plan = _build_episode_plans(cfg, targets, measurement["calibration_commitment_sha256"])
    output = args.restricted_root / "synthetic_one_hour/calibration"
    write_private(output / "restricted_calibration_features.json", {"schema_version": 1, "rows": restricted_features})
    write_private(output / "restricted_calibration_targets.json", measurement)
    write_private(output / "restricted_episode_plans.json", plan)
    compact = {
        "status": "PASS" if missing_axes == 0 else "PASS_WITH_DECLARED_MISSINGNESS",
        "axis_count": len(AXES),
        "joint_count": 4,
        "sampled_frame_count": sampled_frames,
        "grounding_event_count": grounding_events,
        "missing_axis_count": missing_axes,
        "suppressed_cell_count": suppressed,
        "episode_plan_count": len(plan["plans"]),
        "calibration_commitment_sha256": measurement["calibration_commitment_sha256"],
        "episode_plan_commitment_sha256": plan["episode_plan_commitment_sha256"],
    }
    write_private(output / "compact_calibration_result.json", compact)
    return compact


def report(args: argparse.Namespace) -> dict[str, Any]:
    path = args.restricted_root / "synthetic_one_hour/calibration/compact_calibration_result.json"
    value = json.loads(path.read_text())
    print(
        compact_aggregate_json(
            value,
            allowed_fields=TERMINAL_FIELDS,
            sha256_fields=TERMINAL_HASH_FIELDS,
        )
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--restricted-root", type=Path, required=True)
    run_parser.add_argument("--public-root", type=Path, required=True)
    run_parser.add_argument("--scratch-root", type=Path, required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--restricted-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        value = run(args)
        print(
            compact_aggregate_json(
                value,
                allowed_fields=TERMINAL_FIELDS,
                sha256_fields=TERMINAL_HASH_FIELDS,
            )
        )
    else:
        report(args)


if __name__ == "__main__":
    main()
