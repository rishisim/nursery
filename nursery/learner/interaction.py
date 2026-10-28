"""Additive word matching with a zero-initialized visual–body interaction."""

from functools import lru_cache

import torch
import torch.nn as nn

from .additive import _model_class as _additive_class


def centre_clip_valid(ctx):
    """Observed steps in the contiguous source clip containing the centre."""
    seg = ctx["seg"]
    centre = seg.shape[1] // 2
    same = (seg == seg[:, centre:centre + 1]) & (seg >= 0)
    left = same[:, :centre + 1].flip(1).cumprod(1).flip(1).bool()
    right = same[:, centre + 1:].cumprod(1).bool()
    return torch.cat((left, right), 1) & (ctx["missing"] == 0)


@lru_cache(maxsize=None)
def _model_class():
    class JointInteraction(_additive_class()):
        def __init__(self, vocab, emb_dim, tau_p=0.1, dims=None, oracle_classes=0, **kwargs):
            super().__init__(vocab, emb_dim, tau_p, dims, oracle_classes, **kwargs)
            self.interaction_enabled = True
            if self.encoder is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(0)
                    self.interaction_query = nn.Linear(self.latent, 64)
                    self.interaction_key = nn.Linear(self.word.embedding_dim, 64)
                    self.interaction_value = nn.Linear(self.word.embedding_dim, 64)
                    self.interaction_out = nn.Linear(64, self.word.embedding_dim, bias=False)
                    nn.init.zeros_(self.interaction_out.weight)

        def interaction_similarity(self, R, T, ctx):
            valid = centre_clip_valid(ctx)
            visual_valid = ctx.get("visual_valid", torch.ones(R.shape[:2], dtype=torch.bool, device=R.device))
            visual_valid = visual_valid.bool()
            query = self.interaction_query(self.states(ctx))
            # Zero invalid values too: they must not contribute to the attended content.
            visual = R.masked_fill(~visual_valid.unsqueeze(-1), 0)
            key, value = self.interaction_key(visual), self.interaction_value(visual)
            logits = query @ key.transpose(1, 2) / 8.0
            logits = logits.masked_fill(~visual_valid.unsqueeze(1), -torch.inf)
            # An empty visual row yields zero attention, without a NaN softmax.
            logits = torch.where(visual_valid.any(1)[:, None, None], logits, torch.zeros_like(logits))
            attention = logits.softmax(-1) * visual_valid.unsqueeze(1)
            joint = (attention @ value) * query
            pooled = (joint * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
            return self.interaction_out(pooled) @ T.t()

        def score_encoded(self, R, T, ctx=None):
            """Unscaled situation/word scores shared by training and evaluation."""
            if self.encoder is None or ctx is None:
                return super().score_encoded(R, T, ctx)
            keep = ctx["keep"].bool() & centre_clip_valid(ctx).any(1)
            ctx = dict(ctx, keep=keep)
            scores = super().score_encoded(R, T, ctx)
            if self.interaction_enabled and keep.any():
                scores = scores + keep.unsqueeze(1) * self.interaction_similarity(R, T, ctx)
            return scores

        def pair_scores(self, v, t, n, ctx=None):
            R, T = self.enc_regions(v), self.enc_text(t, n)
            return R, T, self.score_encoded(R, T, ctx)

    return JointInteraction


def JointInteractionMIL(vocab, emb_dim, tau_p=0.1, dims=None, oracle_classes=0):
    """Build the joint learner while retaining the additive model's direct pathways."""
    return _model_class()(vocab, emb_dim, tau_p, dims, oracle_classes)
