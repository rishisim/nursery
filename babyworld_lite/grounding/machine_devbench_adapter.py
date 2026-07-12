from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from babyworld_lite.grounding.pilot_checkpoint import load_grounding_checkpoint


def resolve_eval_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NurseryMachineDevBenchExtractor(nn.Module):
    """Expose a Nursery grounding learner through Machine-DevBench's protocol.

    Machine-DevBench presents static images. Nursery's visual encoder was trained
    on short clips, so the adapter repeats each benchmark image across the
    checkpoint's configured frame count. This is a transparent, deterministic
    compatibility rule; it is not a claim that the provisional Nursery learner
    is in-distribution for the benchmark.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.eval_device = resolve_eval_device(device)
        model, tokenizer, payload = load_grounding_checkpoint(
            self.checkpoint_path, self.eval_device
        )
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_metadata = dict(payload["metadata"])
        self.frame_count = int(payload["input_config"]["frame_count"])
        self.image_size = int(payload["input_config"]["image_size"])
        self.max_text_length = int(payload["input_config"]["max_text_length"])
        self._feature_dim = int(payload["model_config"]["embedding_dim"])

    def _pil_batch(self, images: list[Image.Image]) -> torch.Tensor:
        arrays = []
        for image in images:
            resized = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            arrays.append(np.asarray(resized, dtype=np.uint8).copy())
        return torch.stack(
            [torch.from_numpy(array).permute(2, 0, 1) for array in arrays]
        )

    def _tensor_batch(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("image tensor must have shape (B, 3, H, W)")
        values = images.detach()
        if values.shape[-2:] != (self.image_size, self.image_size):
            values = F.interpolate(
                values.float(),
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            )
        if values.dtype != torch.uint8:
            if float(values.max()) <= 1.0:
                values = values * 255.0
            values = values.round().clamp(0, 255).to(torch.uint8)
        return values.cpu()

    def extract_image_features(
        self, images: torch.Tensor | list[Image.Image]
    ) -> torch.Tensor:
        image_batch = (
            self._pil_batch(images)
            if isinstance(images, list)
            else self._tensor_batch(images)
        )
        video = image_batch[:, None].repeat(1, self.frame_count, 1, 1, 1)
        with torch.inference_mode():
            return self.model.encode_video(video.to(self.eval_device))

    def extract_text_features(
        self, text: list[str] | torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(text, list):
            encoded = [
                self.tokenizer.encode(value, self.max_text_length) for value in text
            ]
            tokens = torch.stack([value[0] for value in encoded])
            mask = torch.stack([value[1] for value in encoded])
        elif isinstance(text, Mapping):
            tokens = text["input_ids"]
            mask = text.get("attention_mask", tokens.ne(0))
        else:
            tokens = text
            mask = tokens.ne(0)
        with torch.inference_mode():
            return self.model.encode_text(
                tokens.to(self.eval_device), mask.to(self.eval_device)
            )

    def extract_video_features(self, video: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model.encode_video(video.to(self.eval_device))

    def extract_features(
        self, inputs: Mapping[str, Any]
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if "image" in inputs:
            outputs["image_features"] = self.extract_image_features(inputs["image"])
        if "text" in inputs:
            outputs["text_features"] = self.extract_text_features(inputs["text"])
        if "video" in inputs:
            outputs["video_features"] = self.extract_video_features(inputs["video"])
        return outputs

    def compute_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        if normalize:
            a = F.normalize(a, p=2, dim=-1)
            b = F.normalize(b, p=2, dim=-1)
        return self.model.logit_scale.exp().clamp(max=100) * a @ b.T

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
