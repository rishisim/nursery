from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from babyworld_lite.grounding.pilot_data import WordTokenizer
from babyworld_lite.grounding.pilot_model import GroundingModel


CHECKPOINT_SCHEMA_VERSION = "nursery-grounding-checkpoint-v1"


def save_grounding_checkpoint(
    path: str | Path,
    model: GroundingModel,
    tokenizer: WordTokenizer,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    """Save the complete state needed by an external evaluation adapter."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "tokenizer_vocabulary": dict(tokenizer.vocabulary),
        "model_config": {
            "vocabulary_size": len(tokenizer),
            "hidden_dim": int(config["hidden_dim"]),
            "embedding_dim": int(config["embedding_dim"]),
            "motor_dim": 4,
        },
        "input_config": {
            "frame_count": int(config["frame_count"]),
            "image_size": int(config["image_size"]),
            "max_text_length": int(config["max_text_length"]),
        },
        "metadata": dict(metadata),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_grounding_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[GroundingModel, WordTokenizer, dict[str, Any]]:
    """Load a Nursery learner without requiring its original training corpus."""
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema: {payload.get('schema_version')!r}"
        )
    tokenizer = WordTokenizer(payload["tokenizer_vocabulary"])
    model = GroundingModel(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, payload
