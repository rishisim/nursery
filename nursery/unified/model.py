"""Joint contrastive and prefix-conditioned generative embodied learner."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nursery.learner.losses import weighted_nce

from .gate import ModalityGate
from .language import CausalLanguage
from .sensory import BodyStream, SensoryFusion, StaticVisionGaze, TemporalVideoStream


CONDITIONS = ("V", "V+G", "V+G+B")


class UnifiedEmbodiedLearner(nn.Module):
    def __init__(self, vocab_size, *, input_dim=768, grid=16, body_dims=None,
                 condition="V", lambda_gen=0.2, mhsa=True):
        super().__init__()
        if condition not in CONDITIONS or not math.isfinite(lambda_gen) or lambda_gen < 0:
            raise ValueError("Unknown condition or invalid generative loss weight")
        if condition == "V+G+B" and body_dims is None:
            raise ValueError("V+G+B requires body group dimensions")
        self.condition, self.lambda_gen = condition, lambda_gen
        self.static = StaticVisionGaze(input_dim, grid)
        self.temporal = TemporalVideoStream(input_dim)
        self.fusion = SensoryFusion() if mhsa else None
        self.language = CausalLanguage(vocab_size)
        self.gate = ModalityGate()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.body = BodyStream(body_dims) if body_dims is not None else None

    def enc_text(self, words, counts):
        return self.language.encode(words, counts)

    def temporal_frames(self, tokens, ctx):
        """Pool each frozen frame for the Phase 1 temporal window adapter."""
        mean = tokens.mean(dim=-2)
        if self.condition == "V" or "gaze_point" not in ctx:
            return mean
        # Match Phase 1 steering: CLS keeps its share, and the close-up
        # receives half the frame mass when gaze is valid.
        valid = ctx["gaze_valid"] & ctx["same"] & ctx["keep"][:, None]
        point = ctx["gaze_point"].masked_fill(~valid[..., None], 0)
        prior = self.static.frame_prior(point, valid)
        pooled = torch.einsum("bfk,bfkd->bfd", prior.exp(), tokens)
        if "closeup" in ctx:
            pooled = (pooled + ctx["closeup"].masked_fill(~valid[..., None], 0)) * 0.5
        return torch.where(valid[..., None], pooled, mean)

    def encode_sensory(self, video, frame_valid, temporal=None, gaze=None, body=None,
                       *, remove_extras=False):
        if remove_extras or self.condition == "V":
            gaze = None
        if remove_extras or self.condition != "V+G+B":
            body = None
        static, regions, log_prior = self.static(video, frame_valid, gaze)
        no_input = torch.zeros(len(video), device=video.device, dtype=torch.bool)
        motion, motion_active = torch.zeros_like(static), no_input
        if temporal is not None:
            same = temporal["same"]
            if self.training:
                keep = torch.rand(len(video), device=video.device) >= 0.5
                same = same & keep[:, None]
            motion = self.temporal(temporal["frames"], same)
            motion_active = same.any(dim=1)
        embodiment, body_active = torch.zeros_like(static), no_input
        if body is not None:
            embodiment, body_active = self.body(body)
        active = torch.stack((torch.ones_like(no_input), motion_active, body_active), dim=1)
        tokens = torch.stack((static, motion, embodiment), dim=1).masked_fill(~active[..., None], 0)
        fused = self.fusion(tokens, active) if self.fusion is not None else tokens
        return fused, active, regions, log_prior, fused[:, 0] - static

    def score(self, sensory, text):
        fused, active, regions, log_prior, residual = sensory
        static = self.static.score(regions, log_prior, text, residual)
        motion, body = (F.normalize(fused[:, 1:], dim=-1) @ text.T).unbind(1)
        gate = self.gate(text, active[:, None, :])
        return (gate * torch.stack((static, motion, body), dim=-1)).sum(-1)

    def forward(self, words, counts, video, frame_valid, temporal=None, gaze=None, body=None):
        # InfoNCE queries use the unconditional text pass. The sensory prefix
        # only enters the separate LM loss, with the same text encoder weights.
        text = self.enc_text(words, counts)
        sensory = self.encode_sensory(video, frame_valid, temporal, gaze, body)
        logits = self.logit_scale.clamp(max=math.log(100)).exp() * self.score(sensory, text)
        nce = weighted_nce(logits, logits.new_ones(len(logits)))
        loss = nce
        if self.lambda_gen:
            fused, active, *_ = sensory
            loss = loss + self.lambda_gen * self.language.lm_loss(words, counts, fused, active)
        return {"loss": loss, "logits": logits, "text": text}
