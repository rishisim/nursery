from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VideoEncoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal = nn.GRU(64, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = video.shape
        pixels = video.reshape(batch * frames, channels, height, width).float() / 255.0
        frame_features = self.frame_encoder(pixels).flatten(1).reshape(batch, frames, -1)
        _, hidden = self.temporal(frame_features)
        return F.normalize(self.projection(hidden[-1]), dim=-1)


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.recurrent = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, hidden = self.recurrent(values)
        return F.normalize(self.projection(hidden[-1]), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_dim, padding_idx=0)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        outputs, _ = self.recurrent(embedded)
        lengths = mask.sum(dim=1).clamp(min=1) - 1
        final = outputs[torch.arange(len(outputs), device=outputs.device), lengths]
        return F.normalize(self.projection(final), dim=-1)


class GroundingModel(nn.Module):
    """The same three-encoder architecture is used in every motor arm."""

    def __init__(
        self,
        vocabulary_size: int,
        hidden_dim: int = 64,
        embedding_dim: int = 48,
        motor_dim: int = 4,
    ):
        super().__init__()
        self.video = VideoEncoder(hidden_dim, embedding_dim)
        self.text = TextEncoder(vocabulary_size, hidden_dim, embedding_dim)
        self.motor = SequenceEncoder(motor_dim, hidden_dim, embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.video(video)

    def encode_text(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.text(tokens, mask)

    def encode_motor(self, motor: torch.Tensor) -> torch.Tensor:
        return self.motor(motor)


def symmetric_contrastive_loss(
    left: torch.Tensor, right: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    logits = logit_scale.exp().clamp(max=100) * left @ right.T
    labels = torch.arange(len(left), device=left.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def grounding_loss(
    model: GroundingModel,
    video: torch.Tensor,
    tokens: torch.Tensor,
    text_mask: torch.Tensor,
    motor: torch.Tensor,
    motor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    video_embedding = model.encode_video(video)
    text_embedding = model.encode_text(tokens, text_mask)
    motor_embedding = model.encode_motor(motor)
    video_text = symmetric_contrastive_loss(
        video_embedding, text_embedding, model.logit_scale
    )
    motor_text = symmetric_contrastive_loss(
        motor_embedding, text_embedding, model.logit_scale
    )
    motor_video = symmetric_contrastive_loss(
        motor_embedding, video_embedding, model.logit_scale
    )
    total = video_text + motor_weight * (motor_text + motor_video)
    return total, {
        "loss": float(total.detach()),
        "video_text_loss": float(video_text.detach()),
        "motor_text_loss": float(motor_text.detach()),
        "motor_video_loss": float(motor_video.detach()),
    }
