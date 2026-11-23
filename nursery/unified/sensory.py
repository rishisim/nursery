"""Aligned static, temporal and body streams over frozen simulator features."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nursery.learner.body import BodyEncoder
from nursery.learner.interaction import centre_clip_valid
from nursery.learner.losses import soft_pool
from nursery.learner.video_only import TemporalVideoEncoder


class StaticVisionGaze(nn.Module):
    """Soft-MIL over DINO CLS, patches and optional shared-projection close-ups."""

    tau = 0.1
    sigma = 0.15

    def __init__(self, input_dim, grid):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Dropout(0.1),
                                     nn.Linear(input_dim, 512))
        axis = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("patch_centers", torch.stack((xx, yy), -1).flatten(0, 1))
        self.regions = grid ** 2 + 1

    def frame_prior(self, point, valid):
        in_crop = valid & ((point >= 0) & (point <= 1)).all(-1)
        distance = (point[:, :, None] - self.patch_centers).square().sum(-1)
        patch_prior = torch.log_softmax(-distance / (2 * self.sigma ** 2), -1)
        # CLS retains its uniform share, while gaze redistributes the patch mass.
        uniform = -math.log(self.regions)
        patch_prior = torch.where(
            in_crop[..., None], patch_prior + math.log1p(-1 / self.regions), uniform
        )
        cls_prior = torch.full_like(patch_prior[..., :1], uniform)
        return torch.cat((cls_prior, patch_prior), -1)

    def forward(self, video, frame_valid, gaze=None):
        # Keep the same token layout with or without gaze for exact ablations.
        if gaze is None:
            point = video.new_zeros((*frame_valid.shape, 2))
            closeup = video.new_zeros((*frame_valid.shape, video.shape[-1]))
            valid = torch.zeros_like(frame_valid)
        else:
            valid = frame_valid & gaze["gaze_valid"] & gaze["keep"][:, None]
            point = gaze["gaze_point"].masked_fill(~valid[..., None], 0)
            closeup = gaze["closeup"].masked_fill(~valid[..., None], 0)

        frame_mass = -frame_valid.sum(1, keepdim=True).log()
        sharp_mass = frame_mass - math.log(2)
        main_mass = torch.where(valid, sharp_mass, frame_mass)
        main_prior = self.frame_prior(point, valid) + main_mass[..., None]
        main_prior = main_prior.masked_fill(~frame_valid[..., None], -torch.inf)
        sharp_prior = sharp_mass.expand_as(valid).masked_fill(~valid, -torch.inf)
        log_prior = torch.cat((main_prior, sharp_prior[..., None]), -1).flatten(1)

        video = video.masked_fill(~frame_valid[..., None, None], 0)
        tokens = torch.cat((video, closeup.unsqueeze(-2)), dim=-2).flatten(1, 2)
        regions = F.normalize(self.project(tokens), dim=-1)
        # A query-independent summary is the token supplied to fusion and the LM.
        summary = torch.bmm(log_prior.exp().unsqueeze(1), regions).squeeze(1)
        return summary, regions, log_prior

    def score(self, regions, log_prior, text):
        return soft_pool(regions @ text.T, log_prior, self.tau)


class TemporalVideoStream(nn.Module):
    """Center readout of a 128-d BiGRU over an 11-second clip-bounded window."""

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = TemporalVideoEncoder(input_dim)
        self.project = nn.Linear(128, 512)

    def forward(self, frames, same):
        return self.project(self.encoder(frames, same))


class BodyStream(nn.Module):
    """Per-group MLPs and a clip-reset 128-d BiGRU, read at the center."""

    def __init__(self, dims):
        super().__init__()
        names = ("contact", "fingers", "arm", "load", "skin")
        if set(dims) != set(names):
            raise ValueError("Body requires contact, fingers, arm, load and skin")
        self.encoder = BodyEncoder({name: dims[name] for name in names})
        self.project = nn.Linear(128, 512)

    def forward(self, body):
        same = centre_clip_valid(body) & body["keep"].bool()[:, None]
        groups = {name: values.masked_fill(~same[..., None], 0)
                  for name, values in body["groups"].items()}
        states = self.encoder(groups, (~same).float(), body["seg"].masked_fill(~same, -2))
        return self.project(states[:, 5]), same[:, 5]


class SensoryFusion(nn.Module):
    """Update each active summary through one zero-initialized residual."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.attention = nn.MultiheadAttention(512, 4, batch_first=True)
        self.out_proj = nn.Linear(512, 512)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, tokens, active):
        tokens = tokens.masked_fill(~active[..., None], 0)
        x = self.norm(tokens)
        attended, _ = self.attention(x, x, x, key_padding_mask=~active, need_weights=False)
        return (tokens + self.out_proj(attended)).masked_fill(~active[..., None], 0)
