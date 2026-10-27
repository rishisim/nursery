"""Video-only word learner; also the unchanged visual base of the additive learner."""

from functools import lru_cache

import numpy as np
import torch

from . import _upstream_module
from .losses import distill_kl, soft_pool, weighted_nce


@lru_cache(maxsize=None)
def _video_class():
    region = _upstream_module("train_region_mil")

    class VideoOnly(region.RegionMIL):
        def __init__(self, vocab, emb_dim, tau_p=0.1, **kwargs):
            super().__init__(vocab, emb_dim=emb_dim, **kwargs)
            self.tau_p = tau_p
            self.encoder = None

        def scale(self):
            return self.logit_scale.clamp(max=np.log(100)).exp()

        def pair_scores(self, v, t, n, ctx=None):
            R, T = self.enc_regions(v), self.enc_text(t, n)
            similarities = torch.einsum("brd,md->brm", R, T)
            uniform = torch.full(similarities.shape[:2], -np.log(similarities.shape[1]),
                                 device=similarities.device, dtype=similarities.dtype)
            return R, T, soft_pool(similarities, uniform, self.tau_p)

        def losses(self, v, t, n, w, ctx=None, weights=None, target=None, temp=2.0):
            _, _, scores = self.pair_scores(v, t, n)
            logits = self.scale() * scores
            out = {"match": weighted_nce(logits, w)}
            if (weights or {}).get("distill", 0) > 0 and target is not None:
                out["distill"] = distill_kl(logits, target.detach(), w, temp)
            return out

        def forward_loss(self, v, t, n, w, ctx=None, weights=None, target=None, temp=2.0):
            terms = self.losses(v, t, n, w, ctx, weights, target, temp)
            return terms["match"] + sum((weights or {})[k] * terms[k] for k in terms if k != "match")

    return VideoOnly


def VideoOnlyMIL(vocab, emb_dim, tau_p=0.1, pool="soft"):
    """Construct the baseline without any body modules; preserve historical max pooling."""
    if pool == "max":
        return _upstream_module("train_frame_mil").RegionMIL(vocab, emb_dim=emb_dim)
    if pool != "soft":
        raise ValueError("pool must be max or soft")
    return _video_class()(vocab, emb_dim, tau_p)
