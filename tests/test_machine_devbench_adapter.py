from __future__ import annotations

from dataclasses import asdict

from PIL import Image
import torch

from babyworld_lite.grounding.machine_devbench_adapter import (
    NurseryMachineDevBenchExtractor,
)
from babyworld_lite.grounding.pilot_checkpoint import save_grounding_checkpoint
from babyworld_lite.grounding.pilot_data import WordTokenizer
from babyworld_lite.grounding.pilot_experiment import PilotConfig
from babyworld_lite.grounding.pilot_model import GroundingModel


def test_nursery_adapter_implements_multimodal_feature_protocol(tmp_path) -> None:
    config = PilotConfig(
        frame_count=2,
        image_size=20,
        max_text_length=8,
        hidden_dim=8,
        embedding_dim=6,
    )
    tokenizer = WordTokenizer.fit(["the red ball is pushed", "touch the cup"])
    model = GroundingModel(len(tokenizer), hidden_dim=8, embedding_dim=6)
    checkpoint = save_grounding_checkpoint(
        tmp_path / "learner.pt",
        model,
        tokenizer,
        asdict(config),
        {"arm": "synchronized", "seed": 1},
    )
    extractor = NurseryMachineDevBenchExtractor(checkpoint, device="cpu")
    batch = {
        "image": [Image.new("RGB", (24, 28), "red"), Image.new("RGB", (24, 28), "blue")],
        "text": ["the red ball", "the blue cup"],
    }
    features = extractor.extract_features(batch)
    assert features["image_features"].shape == (2, 6)
    assert features["text_features"].shape == (2, 6)
    assert extractor.feature_dim == 6
    similarity = extractor.compute_similarity(
        features["image_features"], features["text_features"], normalize=True
    )
    assert similarity.shape == (2, 2)
    assert torch.isfinite(similarity).all()
