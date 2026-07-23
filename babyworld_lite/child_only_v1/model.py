"""Compact from-scratch temporal CLIP+-style construction scaffold.

The primary architecture follows the public EgoBabyVLM CLIP+ interface shape:
a BERT-like text tower and ViT-like vision tower projected into a shared space.
No external weights or tokenizers are loaded.  Raw side streams and the
event/null aligner exist only behind the training API.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import re
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .policy import CHILD_CORPORA, CONSTRUCTION_PROFILE, PolicyViolation, canonical_digest


@dataclass(frozen=True)
class TemporalCLIPConfig:
    image_size: int = 64
    patch_size: int = 8
    max_frames: int = 8
    max_tokens: int = 32
    vocabulary_size: int = 512
    text_width: int = 64
    text_layers: int = 2
    text_heads: int = 4
    vision_width: int = 64
    spatial_layers: int = 2
    temporal_layers: int = 2
    vision_heads: int = 4
    side_input_dim: int = 18
    side_width: int = 48
    embedding_dim: int = 48
    dropout: float = 0.1
    initialization: str = "SCRATCH"
    pretrained_source: None = None

    def validate(self) -> None:
        if self.initialization != "SCRATCH" or self.pretrained_source is not None:
            raise PolicyViolation("scientific learner must initialize from scratch")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.text_width % self.text_heads or self.vision_width % self.vision_heads:
            raise ValueError("tower widths must be divisible by attention head counts")
        for key, value in asdict(self).items():
            if key not in {"dropout", "initialization", "pretrained_source"} and int(value) <= 0:
                raise ValueError(f"{key} must be positive")

    @property
    def architecture_digest(self) -> str:
        return canonical_digest(asdict(self))


class FreshCorpusTokenizer:
    """A deterministic, corpus-instance-bound tokenizer trained from raw text.

    This compact construction implementation uses a closed word/punctuation
    vocabulary.  A post-access adapter may replace the tokenizer algorithm only
    through a new frozen protocol version; it may never reuse another corpus's
    tokenizer state.
    """

    RESERVED = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

    def __init__(
        self,
        vocabulary: Mapping[str, int],
        *,
        source_family: str,
        corpus_instance_id: str,
        tokenizer_artifact_id: str,
        profile_label: str | None,
    ) -> None:
        self.vocabulary = dict(vocabulary)
        self.source_family = source_family
        self.corpus_instance_id = corpus_instance_id
        self.tokenizer_artifact_id = tokenizer_artifact_id
        self.profile_label = profile_label
        if tuple(token for token, _ in sorted(self.vocabulary.items(), key=lambda item: item[1])[:5]) != self.RESERVED:
            raise PolicyViolation("tokenizer reserved-token inventory is invalid")

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        *,
        source_family: str,
        corpus_instance_id: str,
        tokenizer_artifact_id: str,
        max_vocabulary_size: int,
        profile_label: str | None = None,
    ) -> "FreshCorpusTokenizer":
        if source_family == "SYNTHETIC_FIXTURE":
            if profile_label != CONSTRUCTION_PROFILE:
                raise PolicyViolation("fixture tokenizer requires the construction label")
        elif source_family in CHILD_CORPORA:
            if profile_label is not None or not corpus_instance_id:
                raise PolicyViolation("scientific tokenizer must bind exactly one corpus instance")
        else:
            raise PolicyViolation("tokenizer source must be one selected child corpus or a construction fixture")
        if max_vocabulary_size < len(cls.RESERVED):
            raise ValueError("max_vocabulary_size is smaller than reserved-token inventory")
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("tokenizer training requires nonempty raw-text records")
        counts = Counter(token for text in texts for token in cls._pieces(text))
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        vocabulary = {token: index for index, token in enumerate(cls.RESERVED)}
        for token in ordered[: max_vocabulary_size - len(cls.RESERVED)]:
            vocabulary[token] = len(vocabulary)
        return cls(
            vocabulary,
            source_family=source_family,
            corpus_instance_id=corpus_instance_id,
            tokenizer_artifact_id=tokenizer_artifact_id,
            profile_label=profile_label,
        )

    @staticmethod
    def _pieces(text: str) -> list[str]:
        return re.findall(r"[\w']+|[^\w\s]", text.lower(), flags=re.UNICODE)

    def encode(self, texts: Sequence[str], max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        if max_tokens < 2:
            raise ValueError("max_tokens must leave room for CLS and SEP")
        rows: list[list[int]] = []
        for text in texts:
            pieces = self._pieces(text)[: max_tokens - 2]
            ids = [self.vocabulary["[CLS]"]]
            ids.extend(self.vocabulary.get(piece, self.vocabulary["[UNK]"]) for piece in pieces)
            ids.append(self.vocabulary["[SEP]"])
            rows.append(ids)
        width = min(max(len(row) for row in rows), max_tokens)
        tokens = torch.full((len(rows), width), self.vocabulary["[PAD]"], dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for index, row in enumerate(rows):
            row = row[:width]
            tokens[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = True
        return tokens, mask

    def receipt(self) -> dict[str, Any]:
        core = {
            "tokenizer_version": "fresh-corpus-tokenizer-v1",
            "source_family": self.source_family,
            "corpus_instance_id": self.corpus_instance_id,
            "tokenizer_artifact_id": self.tokenizer_artifact_id,
            "profile_label": self.profile_label,
            "vocabulary": self.vocabulary,
        }
        return {**core, "digest": canonical_digest(core)}


class TinyBertTextTower(nn.Module):
    """Bidirectional Transformer encoder with BERT-like CLS pooling."""

    def __init__(self, config: TemporalCLIPConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.text_width, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.empty(1, config.max_tokens, config.text_width))
        self.type_embedding = nn.Embedding(1, config.text_width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.text_width,
            nhead=config.text_heads,
            dim_feedforward=config.text_width * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.text_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.text_width)
        self.projection = nn.Linear(config.text_width, config.embedding_dim, bias=False)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or attention_mask.shape != tokens.shape:
            raise ValueError("text tokens and attention mask must have shape (B, L)")
        if tokens.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("token sequence exceeds configured maximum")
        hidden = self.token_embedding(tokens) + self.position_embedding[:, : tokens.shape[1]]
        hidden = hidden + self.type_embedding.weight[0].view(1, 1, -1)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
        hidden = self.norm(hidden)
        return F.normalize(self.projection(hidden[:, 0]), dim=-1), hidden


class CompactTemporalViTTower(nn.Module):
    """Small random-init spatial ViT plus temporal Transformer aggregator."""

    def __init__(self, config: TemporalCLIPConfig) -> None:
        super().__init__()
        self.config = config
        patches = (config.image_size // config.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, config.vision_width, config.patch_size, stride=config.patch_size)
        self.spatial_cls = nn.Parameter(torch.empty(1, 1, config.vision_width))
        self.spatial_position = nn.Parameter(torch.empty(1, patches + 1, config.vision_width))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=config.vision_width,
            nhead=config.vision_heads,
            dim_feedforward=config.vision_width * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=config.spatial_layers, enable_nested_tensor=False)
        self.temporal_cls = nn.Parameter(torch.empty(1, 1, config.vision_width))
        self.temporal_position = nn.Parameter(torch.empty(1, config.max_frames + 1, config.vision_width))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.vision_width,
            nhead=config.vision_heads,
            dim_feedforward=config.vision_width * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=config.temporal_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.vision_width)
        self.projection = nn.Linear(config.vision_width, config.embedding_dim, bias=False)
        for parameter in (self.spatial_cls, self.spatial_position, self.temporal_cls, self.temporal_position):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError("video must have shape (B, T, C, H, W)")
        batch, frames, channels, height, width = video.shape
        if channels != 3 or height != self.config.image_size or width != self.config.image_size:
            raise ValueError("video dimensions differ from configured RGB input")
        if frames > self.config.max_frames:
            raise ValueError("video exceeds configured maximum frame count")
        pixels = video.float()
        if video.dtype == torch.uint8:
            pixels = pixels / 255.0
        patches = self.patch_embed(pixels.reshape(batch * frames, channels, height, width))
        patches = patches.flatten(2).transpose(1, 2)
        cls = self.spatial_cls.expand(batch * frames, -1, -1)
        spatial = torch.cat((cls, patches), dim=1) + self.spatial_position[:, : patches.shape[1] + 1]
        spatial = self.spatial_encoder(spatial)[:, 0].reshape(batch, frames, -1)
        temporal_cls = self.temporal_cls.expand(batch, -1, -1)
        temporal = torch.cat((temporal_cls, spatial), dim=1) + self.temporal_position[:, : frames + 1]
        temporal = self.norm(self.temporal_encoder(temporal))
        frame_states = temporal[:, 1:]
        return F.normalize(self.projection(temporal[:, 0]), dim=-1), frame_states


class RawSideTemporalEncoder(nn.Module):
    """Small temporal encoder for raw IMU/proprioception/contact/motor values."""

    def __init__(self, config: TemporalCLIPConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.side_input_dim)
        self.recurrent = nn.GRU(config.side_input_dim, config.side_width, batch_first=True)
        self.projection = nn.Linear(config.side_width, config.vision_width)

    def forward(self, raw_side: torch.Tensor) -> torch.Tensor:
        if raw_side.ndim != 3:
            raise ValueError("raw side stream must have shape (B, T, S)")
        outputs, _ = self.recurrent(self.input_norm(raw_side.float()))
        return self.projection(outputs)


class TrainingOnlyEventNullAligner(nn.Module):
    """Scores temporally extended candidate events plus an explicit null event."""

    def __init__(self, config: TemporalCLIPConfig) -> None:
        super().__init__()
        self.event_projection = nn.Linear(config.vision_width * 2, config.embedding_dim)
        self.null_event = nn.Parameter(torch.empty(config.embedding_dim))
        nn.init.normal_(self.null_event, std=0.02)

    def forward(
        self,
        text_embedding: torch.Tensor,
        frame_states: torch.Tensor,
        side_states: torch.Tensor,
        candidate_event_mask: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_event_mask.ndim != 3:
            raise ValueError("candidate event mask must have shape (B, E, T)")
        if frame_states.shape != side_states.shape:
            raise ValueError("vision and side temporal states must be aligned")
        if candidate_event_mask.shape[0] != frame_states.shape[0] or candidate_event_mask.shape[2] != frame_states.shape[1]:
            raise ValueError("candidate masks must share batch/time dimensions")
        if candidate_valid.shape != candidate_event_mask.shape[:2]:
            raise ValueError("candidate_valid must have shape (B, E)")
        weights = candidate_event_mask.float()
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        visual_events = torch.einsum("bet,btd->bed", weights, frame_states) / denominator
        side_events = torch.einsum("bet,btd->bed", weights, side_states) / denominator
        event_keys = F.normalize(self.event_projection(torch.cat((visual_events, side_events), dim=-1)), dim=-1)
        event_logits = torch.einsum("bd,bed->be", text_embedding, event_keys)
        event_logits = event_logits.masked_fill(~candidate_valid.bool(), torch.finfo(event_logits.dtype).min)
        null_key = F.normalize(self.null_event, dim=0)
        null_logit = torch.einsum("bd,d->b", text_embedding, null_key).unsqueeze(1)
        return torch.cat((event_logits, null_logit), dim=1)


class TemporalCLIPPlusTrainingModel(nn.Module):
    """Full training scaffold; all causal arms instantiate this exact class."""

    def __init__(self, config: TemporalCLIPConfig, *, initialization_receipt: Mapping[str, Any]) -> None:
        super().__init__()
        config.validate()
        if initialization_receipt.get("model_initialization") != "SCRATCH" or initialization_receipt.get("parent_checkpoint") is not None:
            raise PolicyViolation("model initialization receipt is not scratch-only")
        self.config = config
        self.initialization_receipt = dict(initialization_receipt)
        self.text_tower = TinyBertTextTower(config)
        self.vision_tower = CompactTemporalViTTower(config)
        self.side_encoder = RawSideTemporalEncoder(config)
        self.training_event_null_aligner = TrainingOnlyEventNullAligner(config)
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())

    def training_forward(
        self,
        *,
        video: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        raw_side: torch.Tensor,
        candidate_event_mask: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return construction tensors; this method does not select or score outcomes."""

        video_embedding, frame_states = self.vision_tower(video)
        text_embedding, _ = self.text_tower(tokens, attention_mask)
        side_states = self.side_encoder(raw_side)
        alignment_logits = self.training_event_null_aligner(
            text_embedding,
            frame_states,
            side_states,
            candidate_event_mask,
            candidate_valid,
        )
        contrastive_logits = self.logit_scale.exp().clamp(max=100) * video_embedding @ text_embedding.T
        return {
            "video_embedding": video_embedding,
            "text_embedding": text_embedding,
            "contrastive_logits": contrastive_logits,
            "training_alignment_logits": alignment_logits,
        }

    def export_primary_evaluation_model(self) -> "VisionTextPrimaryEvaluationModel":
        """Deep-copy only vision/text/similarity state into a narrow module."""

        exported = VisionTextPrimaryEvaluationModel(
            copy.deepcopy(self.vision_tower),
            copy.deepcopy(self.text_tower),
            self.logit_scale.detach().clone(),
        )
        forbidden = ("side", "aligner", "event", "oracle", "detector", "corpus")
        if any(any(token in key.lower() for token in forbidden) for key in exported.state_dict()):
            raise PolicyViolation("evaluation export contains forbidden training-only state")
        return exported


@dataclass(frozen=True)
class VisionTextEvalBatch:
    video: torch.Tensor
    tokens: torch.Tensor
    attention_mask: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VisionTextEvalBatch":
        expected = {"video", "tokens", "attention_mask"}
        if set(value) != expected:
            extras = sorted(set(value) - expected)
            missing = sorted(expected - set(value))
            raise PolicyViolation(f"primary evaluation accepts only vision/text tensors; extras={extras}, missing={missing}")
        if not all(isinstance(value[key], torch.Tensor) for key in expected):
            raise PolicyViolation("primary evaluation inputs must already be tensors")
        return cls(video=value["video"], tokens=value["tokens"], attention_mask=value["attention_mask"])


class VisionTextPrimaryEvaluationModel(nn.Module):
    """Evaluation-only projection containing no side/oracle/corpus components."""

    def __init__(self, vision_tower: nn.Module, text_tower: nn.Module, logit_scale: torch.Tensor) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.text_tower = text_tower
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=False)

    def forward(self, batch: VisionTextEvalBatch) -> dict[str, torch.Tensor]:
        if not isinstance(batch, VisionTextEvalBatch):
            raise PolicyViolation("primary evaluation requires a VisionTextEvalBatch")
        vision, _ = self.vision_tower(batch.video)
        text, _ = self.text_tower(batch.tokens, batch.attention_mask)
        return {
            "vision_features": vision,
            "text_features": text,
            "similarity": self.compute_similarity(vision, text),
        }

    @property
    def feature_dim(self) -> int:
        return int(self.vision_tower.projection.out_features)

    def extract_video_features(self, video: torch.Tensor) -> torch.Tensor:
        return self.vision_tower(video)[0]

    def extract_text_features(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_tower(tokens, attention_mask)[0]

    def compute_similarity(self, vision: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100) * F.normalize(vision, dim=-1) @ F.normalize(text, dim=-1).T


def symmetric_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Architectural scaffold utility; construction validation does not call it."""

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("paired contrastive logits must be square")
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
