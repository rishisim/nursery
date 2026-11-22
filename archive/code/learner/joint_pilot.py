"""Historical width-128 joint pilot, NOT the proposed acquisition-curve architecture.

Kept runnable only through the explicitly historical --arch joint option.
"""

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nursery.learner import _upstream_module
from nursery.learner.body import BodyEncoder
from nursery.learner.losses import REGIONS, soft_pool, weighted_nce


JOINT = {"width": 128, "layers": 2, "heads": 4, "ff": 512, "dropout": 0.0}  # bounded pilot settings


@lru_cache(maxsize=None)
def _joint_class():
    region = _upstream_module("train_region_mil")

    class JointMIL(region.RegionMIL):
        """Historical joint visual-body learner (`--arch joint`): the window's projected region tokens and the
        body encoder's per-second states go through one shared transformer; the contextualised visual tokens are
        scored against the words with soft-MIL. No body tokens = the video-only control."""

        joint = True

        def __init__(self, vocab, emb_dim, tau_p=0.1, dims=None, half=5, **settings):
            cfg = {**JOINT, **settings}
            width = cfg["width"]
            super().__init__(vocab, dim=width, emb_dim=emb_dim)
            self.settings, self.tau_p, self.half = cfg, tau_p, half
            # common parameters: built in the default RNG stream, in this order, with or without a body
            self.modality = nn.Parameter(torch.randn(2, width) * 0.02)  # 0 visual, 1 body
            self.position = nn.Parameter(torch.randn(2 * half + 1, width) * 0.02)  # seconds from the centre frame
            layer = nn.TransformerEncoderLayer(width, cfg["heads"], cfg["ff"], cfg["dropout"], activation="gelu",
                                               batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(width)
            self.encoder = self.body_proj = None
            if dims is not None:
                with torch.random.fork_rng(devices=[]):  # keep the default RNG stream identical to the control
                    torch.manual_seed(0)
                    self.encoder = BodyEncoder(dims)
                    self.body_proj = nn.Linear(2 * self.encoder.hidden, width)

        def scale(self):
            return self.logit_scale.clamp(max=np.log(100)).exp()

        def param_counts(self):
            count = lambda *modules: sum(p.numel() for m in modules if m is not None for p in m.parameters())
            body = count(self.encoder, self.body_proj)
            total = sum(p.numel() for p in self.parameters())
            return {"total": total, "common": total - body, "body_only": body}

        def enc_regions(self, v):
            """Plain probe input [K,17,768]: one frame at offset 0, no body."""
            if v.shape[1] != REGIONS:
                raise ValueError("enc_regions takes single frames [K,17,D]; use represent for windows")
            ones = torch.ones(len(v), 1, dtype=torch.bool, device=v.device)
            return self.represent(v, torch.zeros_like(ones, dtype=torch.long), ones)[0]

        def body_keep(self, ctx):
            """[B,K] body tokens that may be attended: pair keeps its body, step exists, same clip as the centre."""
            return (ctx["keep"] > 0).unsqueeze(1) & (ctx["missing"] == 0) & ctx["same"]

        def represent(self, v, offset, valid, ctx=None):
            """v [B,F*17,D] window regions, offset [B,F] seconds from the centre frame, valid [B,F] (False = repeated
            padding frame). A ctx with body inputs adds the body tokens. -> contextualised visual tokens R
            [B,F*17,width] (unit norm) and their validity [B,F*17]."""
            n = v.shape[1]
            x = (self.vproj(v - self.center_mu) + self.modality[0]
                 + self.position[offset + self.half].repeat_interleave(REGIONS, 1))
            ok = valid.repeat_interleave(REGIONS, 1)
            pad = ~ok
            if self.encoder is not None and ctx is not None and "groups" in ctx:
                body = self.body_proj(self.states(ctx)) + self.modality[1] + self.position
                x, pad = torch.cat([x, body], 1), torch.cat([pad, ~self.body_keep(ctx)], 1)
            out = self.final_norm(self.transformer(x, src_key_padding_mask=pad))[:, :n]
            return F.normalize(out, dim=-1), ok

        def states(self, ctx):
            return self.encoder(ctx["groups"], ctx["missing"], ctx["seg"])

        def score(self, R, valid, T):
            """The one score, in training and every evaluation: soft-MIL (tau_p) over the valid visual tokens with a
            uniform prior. R [B,N,D], valid [B,N], T [M,D] -> [B,M]."""
            s = torch.einsum("brd,md->brm", R, T)
            log_prior = torch.where(valid, -torch.log(valid.sum(1, keepdim=True).to(R.dtype)),
                                    torch.full_like(R[..., 0], -np.inf))
            return soft_pool(s, log_prior, self.tau_p)

        def pair_scores(self, v, t, n, ctx):
            """Training score matrix [B,B]: image window i (with its body, if any) against word bag j."""
            R, ok = self.represent(v, ctx["vis_offset"], ctx["vis_valid"], ctx)
            return self.score(R, ok, self.enc_text(t, n))

        def losses(self, v, t, n, w, ctx=None, weights=None, target=None, temp=2.0):
            return {"match": weighted_nce(self.scale() * self.pair_scores(v, t, n, ctx), w)}

        def forward_loss(self, v, t, n, w, ctx=None, weights=None):
            return self.losses(v, t, n, w, ctx)["match"]

    return JointMIL


def JointMIL(vocab, emb_dim, tau_p=0.1, dims=None, **settings):
    """Build the joint model (subclass of the pinned upstream RegionMIL, imported lazily)."""
    return _joint_class()(vocab, emb_dim, tau_p, dims, **settings)
