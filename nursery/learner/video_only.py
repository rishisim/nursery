"""Video-only word learner; also the unchanged visual base of the additive learner."""

from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import _upstream_module
from .losses import distill_kl, soft_pool, weighted_nce


class TemporalVideoEncoder(nn.Module):
    """Project each frame, then read a one-layer bidirectional GRU at the center."""

    def __init__(self, emb_dim):
        super().__init__()
        self.project = nn.Linear(emb_dim, 128)
        self.forward_cell = nn.GRUCell(128, 64)
        self.backward_cell = nn.GRUCell(128, 64)

    def forward(self, frames, same):
        if frames.ndim != 3 or frames.shape[1] != 11 or same.shape != frames.shape[:2]:
            raise ValueError("Temporal video requires 11 frame vectors and an aligned mask")
        x = self.project(frames.masked_fill(~same[..., None], 0))

        def read(cell, steps):
            state = x.new_zeros(len(x), cell.hidden_size)
            for step in steps:
                state = torch.where(same[:, step, None], cell(x[:, step], state), state)
            return state

        # Each direction needs only the steps up to its center readout.
        forward = read(self.forward_cell, range(6))
        backward = read(self.backward_cell, range(10, 4, -1))
        return torch.cat((forward, backward), dim=-1)


@lru_cache(maxsize=None)
def _video_class():
    region = _upstream_module("train_region_mil")

    class VideoOnly(region.RegionMIL):
        def __init__(self, vocab, emb_dim, tau_p=0.1, temporal_video=False,
                     temporal_weight_init="one", **kwargs):
            if temporal_weight_init not in ("one", "zero", "per_word") or (
                    not temporal_video and temporal_weight_init != "one"):
                raise ValueError("Temporal weight mode requires temporal video and one, zero or per_word")
            super().__init__(vocab, emb_dim=emb_dim, **kwargs)
            self.tau_p = tau_p
            self.encoder = None
            self.temporal_video = TemporalVideoEncoder(emb_dim) if temporal_video else None
            if temporal_video:
                self.temporal_weight = nn.Parameter(
                    torch.zeros(vocab) if temporal_weight_init == "per_word"
                    else torch.tensor(float(temporal_weight_init == "one")))

        def scale(self):
            return self.logit_scale.clamp(max=np.log(100)).exp()

        def pair_scores(self, v, t, n, ctx=None):
            R, T = self.enc_regions(v), self.enc_text(t, n)
            if self.temporal_video is not None and self.temporal_weight.ndim == 1 and ctx is not None:
                ctx = dict(ctx, temporal_tokens=t)
            return R, T, self.score_encoded(R, T, ctx)

        def score_encoded(self, R, T, ctx=None):
            """Unscaled experience–language scores shared by training and evaluation."""
            similarities = torch.einsum("brd,md->brm", R, T)
            uniform = torch.full(similarities.shape[:2], -np.log(similarities.shape[1]),
                                 device=similarities.device, dtype=similarities.dtype)
            return self.add_temporal(soft_pool(similarities, uniform, self.tau_p), T, ctx)

        def temporal_frames(self, tokens, ctx):
            return tokens.mean(dim=-2)

        def temporal_score(self, T, ctx, tokens=None):
            latent = self.temporal_video(ctx["frames"], ctx["same"])
            # Share a 128-d slice of the existing word embeddings.
            if self.temporal_weight.ndim == 0:
                word = F.normalize(T[:, :latent.shape[-1]], dim=-1)
            else:
                # Weight each token's contribution before pooling an utterance.
                # Use the unweighted norm so a single query gets exactly its word's gate.
                embeddings = self.word(tokens)[..., :latent.shape[-1]]
                embeddings = embeddings.masked_fill(tokens[..., None] == 0, 0)
                weighted = (embeddings * self.temporal_weight[tokens][..., None]).sum(dim=1)
                word = weighted / embeddings.sum(dim=1).norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return F.normalize(latent, dim=-1) @ word.t()

        def add_temporal(self, score, T, ctx):
            if self.temporal_video is None or ctx is None or "temporal" not in ctx:
                return score
            temporal = self.temporal_score(T, ctx["temporal"], ctx.get("temporal_tokens"))
            if self.temporal_weight.ndim == 0:
                temporal = self.temporal_weight * temporal
            return score + temporal

        def losses(self, v, t, n, w, ctx=None, weights=None, target=None, temp=2.0):
            _, _, scores = self.pair_scores(v, t, n, ctx if self.temporal_video is not None else None)
            logits = self.scale() * scores
            out = {"match": weighted_nce(logits, w)}
            if (weights or {}).get("distill", 0) > 0 and target is not None:
                out["distill"] = distill_kl(logits, target.detach(), w, temp)
            return out

        def forward_loss(self, v, t, n, w, ctx=None, weights=None, target=None, temp=2.0):
            terms = self.losses(v, t, n, w, ctx, weights, target, temp)
            return terms["match"] + sum((weights or {})[k] * terms[k] for k in terms if k != "match")

    return VideoOnly


def VideoOnlyMIL(vocab, emb_dim, tau_p=0.1, pool="soft", *, temporal_video=False,
                 temporal_weight_init="one"):
    """Construct the baseline without any body modules; preserve historical max pooling."""
    if pool == "max":
        if temporal_video or temporal_weight_init != "one":
            raise ValueError("Temporal video requires soft pooling")
        return _upstream_module("train_frame_mil").RegionMIL(vocab, emb_dim=emb_dim)
    if pool != "soft":
        raise ValueError("pool must be max or soft")
    return _video_class()(vocab, emb_dim, tau_p, temporal_video=temporal_video,
                          temporal_weight_init=temporal_weight_init)
