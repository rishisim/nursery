from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from babyworld_lite.grounding.pipeline import render_raw_frame, validate_model_inputs
from babyworld_lite.sim import ACTIONS, CANVAS, Episode, ObjectSpec


ARM_NAMES = ("null", "synchronized", "shuffled", "time_shifted")
MOTOR_INPUT_KEYS = ("x", "y", "vx", "vy")
MODEL_INPUT_ALLOWLIST = ("rendered_rgb_frames", "utterance_text", "hand_motor_trajectory")


@dataclass(frozen=True)
class ExampleMetadata:
    episode_id: int
    shape: str
    color: str
    action: str


class EpisodeAdapter(Protocol):
    """Boundary that a future corpus adapter must implement.

    Labels are exposed only as metadata for splitting/evaluation. ``video`` and
    ``motor`` are the model inputs and are intentionally label-free.
    """

    def metadata(self, record: Mapping[str, Any]) -> ExampleMetadata: ...

    def text(self, record: Mapping[str, Any]) -> str: ...

    def video(
        self, record: Mapping[str, Any], frame_count: int, image_size: int
    ) -> torch.Tensor: ...

    def motor(self, record: Mapping[str, Any], frame_count: int) -> torch.Tensor: ...


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_episode(record: Mapping[str, Any]) -> Episode:
    obj = ObjectSpec(**record["object"])
    return Episode(
        episode_id=int(record["episode_id"]),
        seed=int(record.get("seed", 0)),
        action=str(record["action"]),
        utterance_pre=str(record["utterance_pre"]),
        utterance_post=str(record["utterance_post"]),
        object=obj,
        hidden_goal=str(record.get("hidden_goal", "")),
        event_label=str(record["event_label"]),
        frames=list(record["frames"]),
        tactile_summary=dict(record.get("tactile_summary", {})),
        causal_graph=dict(record.get("causal_graph", {})),
        counterfactuals=dict(record.get("counterfactuals", {})),
    )


def _sample_indices(length: int, count: int) -> np.ndarray:
    if length < 1:
        raise ValueError("an episode must contain at least one frame")
    if count < 1:
        raise ValueError("frame_count must be positive")
    return np.linspace(0, length - 1, count).round().astype(np.int64)


class BabyWorldEpisodeAdapter:
    """Leak-free adapter for the original ``episodes.jsonl`` schema."""

    def metadata(self, record: Mapping[str, Any]) -> ExampleMetadata:
        obj = record["object"]
        return ExampleMetadata(
            episode_id=int(record["episode_id"]),
            shape=str(obj["shape"]),
            color=str(obj["color_name"]),
            action=str(record["action"]),
        )

    def text(self, record: Mapping[str, Any]) -> str:
        return str(record["utterance_pre"])

    def video(
        self, record: Mapping[str, Any], frame_count: int, image_size: int
    ) -> torch.Tensor:
        episode = _as_episode(record)
        indices = _sample_indices(len(episode.frames), frame_count)
        images: list[torch.Tensor] = []
        for index in indices:
            # Grounding's renderer is actual scene RGB, without the simulator's
            # explanatory action/event/force caption or contact starburst.
            image = render_raw_frame(episode, int(index)).resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.uint8).copy()
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(images)

    def motor(self, record: Mapping[str, Any], frame_count: int) -> torch.Tensor:
        frames = record["frames"]
        indices = _sample_indices(len(frames), frame_count)
        values = [
            [float(frames[i]["hand"][key]) / CANVAS for key in MOTOR_INPUT_KEYS]
            for i in indices
        ]
        return torch.tensor(values, dtype=torch.float32)


class GroundingRecordAdapter:
    """Model-visible consumer for calibration-ready ``examples.jsonl`` rows.

    Only ``model_inputs`` is used to construct tensors. Evaluation targets are
    consulted solely by ``metadata`` for splitting/scoring, and oracle state is
    kept in a physically separate file.
    """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)

    @staticmethod
    def _inputs(record: Mapping[str, Any]) -> Mapping[str, Any]:
        violations = validate_model_inputs(record.get("model_inputs", {}))
        if violations:
            raise ValueError(f"model input schema violations: {violations}")
        return record["model_inputs"]

    def metadata(self, record: Mapping[str, Any]) -> ExampleMetadata:
        targets = record["evaluation_targets"]
        return ExampleMetadata(
            episode_id=int(record["base_episode_id"]),
            shape=str(targets["object_noun"]),
            color=str(targets["color_adjective"]),
            action=str(targets["action_verb"]),
        )

    def text(self, record: Mapping[str, Any]) -> str:
        inputs = self._inputs(record)
        window_frames = len(inputs["motor"])
        observed = sorted(
            (
                item for item in inputs["utterances"]
                if 0 <= int(item["onset_frame"]) < window_frames
            ),
            key=lambda item: int(item["onset_frame"]),
        )
        return " ".join(str(item["text"]) for item in observed)

    def video(
        self, record: Mapping[str, Any], frame_count: int, image_size: int
    ) -> torch.Tensor:
        paths = list(self._inputs(record)["frame_paths"])
        indices = _sample_indices(len(paths), frame_count)
        frames: list[torch.Tensor] = []
        for index in indices:
            with Image.open(self.dataset_root / paths[int(index)]) as image:
                resized = image.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
                array = np.asarray(resized, dtype=np.uint8).copy()
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(frames)

    def motor(self, record: Mapping[str, Any], frame_count: int) -> torch.Tensor:
        sequence = list(self._inputs(record)["motor"])
        indices = _sample_indices(len(sequence), frame_count)
        values = []
        for index in indices:
            sample = sequence[int(index)]
            available = float(sample["available"])
            values.append([
                float(sample[key]) / CANVAS * available for key in MOTOR_INPUT_KEYS
            ])
        return torch.tensor(values, dtype=torch.float32)


def load_pilot_source(
    path: str | Path, alignment_condition: str = "weak"
) -> tuple[list[dict[str, Any]], EpisodeAdapter, str]:
    records = load_jsonl(path)
    if not records:
        raise ValueError("no pilot records loaded")
    if records[0].get("schema_version") == "grounding-v0":
        selected = [
            record for record in records
            if record["alignment_condition"] == alignment_condition
            and record["motor_condition"] == "synchronized"
        ]
        if not selected:
            raise ValueError(
                f"no synchronized {alignment_condition!r} grounding records found"
            )
        return selected, GroundingRecordAdapter(Path(path).parent), "grounding-v0"
    return records, BabyWorldEpisodeAdapter(), "babyworld-legacy"


class WordTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, vocabulary: Mapping[str, int]):
        self.vocabulary = dict(vocabulary)

    @staticmethod
    def words(text: str) -> list[str]:
        return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

    @classmethod
    def fit(cls, texts: Sequence[str]) -> "WordTokenizer":
        words = sorted({word for text in texts for word in cls.words(text)})
        return cls({cls.PAD: 0, cls.UNK: 1, **{word: i + 2 for i, word in enumerate(words)}})

    def encode(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.vocabulary.get(word, 1) for word in self.words(text)][:max_length]
        mask = [1] * len(ids)
        ids.extend([0] * (max_length - len(ids)))
        mask.extend([0] * (max_length - len(mask)))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.vocabulary)


def held_out_composition_indices(
    records: Sequence[Mapping[str, Any]],
    adapter: EpisodeAdapter,
    compositions: Sequence[tuple[str, str]],
) -> tuple[list[int], list[int]]:
    held_out = set(compositions)
    if not held_out:
        raise ValueError("at least one held-out composition is required")
    train: list[int] = []
    test: list[int] = []
    metadata = [adapter.metadata(record) for record in records]
    for index, item in enumerate(metadata):
        (test if (item.shape, item.action) in held_out else train).append(index)
    if not train or not test:
        raise ValueError("held-out composition split produced an empty partition")
    train_set = set(train)
    for shape, action in held_out:
        if not any(item.shape == shape and item.action != action for i, item in enumerate(metadata) if i in train_set):
            raise ValueError(f"training lacks separate support for shape {shape!r}")
        if not any(item.action == action and item.shape != shape for i, item in enumerate(metadata) if i in train):
            raise ValueError(f"training lacks separate support for action {action!r}")
    return train, test


def preassigned_split_indices(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[int], list[int]]:
    train = [index for index, record in enumerate(records) if record.get("split") == "train"]
    test = [index for index, record in enumerate(records) if record.get("split") == "test"]
    if not train or not test:
        raise ValueError("preassigned grounding split produced an empty partition")
    train_compositions = {
        (records[index]["evaluation_targets"]["object_noun"], records[index]["evaluation_targets"]["action_verb"])
        for index in train
    }
    test_compositions = {
        (records[index]["evaluation_targets"]["object_noun"], records[index]["evaluation_targets"]["action_verb"])
        for index in test
    }
    if train_compositions & test_compositions:
        raise ValueError("preassigned train/test composition overlap")
    train_shapes = {shape for shape, _action in train_compositions}
    train_actions = {action for _shape, action in train_compositions}
    for shape, action in test_compositions:
        if shape not in train_shapes or action not in train_actions:
            raise ValueError(
                "preassigned composition split lacks separate training support for "
                f"{shape} x {action}"
            )
    return train, test


def deranged_index_map(indices: Sequence[int], seed: int) -> dict[int, int]:
    if len(indices) < 2:
        raise ValueError("at least two examples are required for shuffled motor cues")
    rng = np.random.default_rng(seed)
    order = np.asarray(indices, dtype=np.int64)[rng.permutation(len(indices))]
    donors = np.roll(order, 1)
    mapping = {int(target): int(donor) for target, donor in zip(order, donors)}
    if any(target == donor for target, donor in mapping.items()):
        raise AssertionError("motor derangement contains a self-match")
    return mapping


class GroundingCorpus:
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        adapter: EpisodeAdapter,
        frame_count: int,
        image_size: int,
    ):
        self.records = list(records)
        self.adapter = adapter
        self.frame_count = frame_count
        self.image_size = image_size
        self._video_cache: dict[int, torch.Tensor] = {}
        self._motor_cache: dict[int, torch.Tensor] = {}

    def metadata(self, index: int) -> ExampleMetadata:
        return self.adapter.metadata(self.records[index])

    def text(self, index: int) -> str:
        return self.adapter.text(self.records[index])

    def video(self, index: int) -> torch.Tensor:
        if index not in self._video_cache:
            self._video_cache[index] = self.adapter.video(
                self.records[index], self.frame_count, self.image_size
            )
        return self._video_cache[index]

    def motor(self, index: int) -> torch.Tensor:
        if index not in self._motor_cache:
            self._motor_cache[index] = self.adapter.motor(
                self.records[index], self.frame_count
            )
        return self._motor_cache[index]


class ArmDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        corpus: GroundingCorpus,
        indices: Sequence[int],
        tokenizer: WordTokenizer,
        arm: str,
        max_text_length: int,
        manipulation_seed: int,
        time_shift: int = 2,
    ):
        if arm not in ARM_NAMES:
            raise KeyError(f"unknown arm: {arm}")
        self.corpus = corpus
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.arm = arm
        self.max_text_length = max_text_length
        self.time_shift = time_shift
        self.donors = deranged_index_map(self.indices, manipulation_seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _motor(self, index: int) -> torch.Tensor:
        source = self.donors[index] if self.arm == "shuffled" else index
        motor = self.corpus.motor(source).clone()
        if self.arm == "null":
            motor.zero_()
        elif self.arm == "time_shifted":
            shift = max(1, min(abs(self.time_shift), len(motor) - 1))
            shifted = torch.zeros_like(motor)
            if self.time_shift > 0:
                shifted[shift:] = motor[:-shift]
            else:
                shifted[:-shift] = motor[shift:]
            motor = shifted
        return motor

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = self.indices[position]
        metadata = self.corpus.metadata(index)
        tokens, mask = self.tokenizer.encode(
            self.corpus.text(index), self.max_text_length
        )
        return {
            "video": self.corpus.video(index),
            "tokens": tokens,
            "text_mask": mask,
            "motor": self._motor(index),
            "episode_index": index,
            "metadata": metadata,
            "text": self.corpus.text(index),
        }


class MotorFreeEvaluationDataset(Dataset[dict[str, Any]]):
    """Primary-test dataset whose code path never constructs a motor tensor."""

    def __init__(
        self,
        corpus: GroundingCorpus,
        indices: Sequence[int],
        tokenizer: WordTokenizer,
        max_text_length: int,
    ):
        self.corpus = corpus
        self.indices = list(indices)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = self.indices[position]
        tokens, mask = self.tokenizer.encode(
            self.corpus.text(index), self.max_text_length
        )
        return {
            "video": self.corpus.video(index),
            "tokens": tokens,
            "text_mask": mask,
            "episode_index": index,
            "metadata": self.corpus.metadata(index),
            "text": self.corpus.text(index),
        }


def collate_examples(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "video": torch.stack([item["video"] for item in items]),
        "tokens": torch.stack([item["tokens"] for item in items]),
        "text_mask": torch.stack([item["text_mask"] for item in items]),
        "motor": torch.stack([item["motor"] for item in items]),
        "episode_index": [int(item["episode_index"]) for item in items],
        "metadata": [item["metadata"] for item in items],
        "text": [str(item["text"]) for item in items],
    }


def collate_motor_free_examples(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "video": torch.stack([item["video"] for item in items]),
        "tokens": torch.stack([item["tokens"] for item in items]),
        "text_mask": torch.stack([item["text_mask"] for item in items]),
        "episode_index": [int(item["episode_index"]) for item in items],
        "metadata": [item["metadata"] for item in items],
        "text": [str(item["text"]) for item in items],
    }


def action_prompt(metadata: ExampleMetadata, action: str) -> str:
    description = " ".join(part for part in (metadata.color, metadata.shape) if part)
    # Evaluation candidates are balanced minimal pairs: only the action word
    # changes while the noun, adjective, length, and syntax remain fixed.
    return f"The action is {action} the {description}."


def validate_action_inventory(records: Sequence[Mapping[str, Any]], adapter: EpisodeAdapter) -> tuple[str, ...]:
    present = {adapter.metadata(record).action for record in records}
    ordered = tuple(action for action in ACTIONS if action in present)
    if len(ordered) < 2:
        raise ValueError("action grounding requires at least two actions")
    return ordered
