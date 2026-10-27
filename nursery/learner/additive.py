"""Additive language learner: the video-only base plus learned body–word evidence.

Optional historical diagnostic losses stay explicit; no joint interaction is added here.
"""

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .body import N_PATCH, BodyEncoder
from .losses import REGIONS, distill_kl, soft_pool, weighted_nce
from .video_only import _video_class


@lru_cache(maxsize=None)
def _model_class():
    class Additive(_video_class()):
        """RegionMIL (picture-only evaluation, unchanged) trained with soft-MIL pooling plus optional body losses.

        `dims` maps encoder groups to widths (None: no body modules). `oracle_classes` > 0 switches the
        vision -> body targets from body features to the held-object identity.
        """

        def __init__(self, vocab, emb_dim, tau_p=0.1, dims=None, oracle_classes=0, **kwargs):
            super().__init__(vocab, emb_dim=emb_dim, **kwargs)
            self.tau_p, self.oracle_classes = tau_p, oracle_classes
            self.encoder = None
            if dims is not None:
                dim = self.word.embedding_dim
                with torch.random.fork_rng(devices=[]):  # keep the default RNG stream identical to the baseline
                    torch.manual_seed(0)
                    self.encoder = BodyEncoder(dims)
                    self.latent = 2 * self.encoder.hidden
                    self.region_in = nn.Linear(dim, 32)
                    out = oracle_classes + 1 if oracle_classes else 2 * N_PATCH + 4
                    self.region_head = nn.Sequential(nn.Linear(REGIONS * 32, 128), nn.ReLU(), nn.Linear(128, out))
                    self.decoder = nn.Linear(self.latent, sum(dims.values()))
                    self.fuse = nn.Linear(self.latent, dim)
                    self.beta = nn.Parameter(torch.tensor(1.0))

        # ---- body-side helpers (training and the body-present test) ----

        def states(self, ctx, mask=None):
            groups, missing = ctx["groups"], ctx["missing"]
            if mask is not None:
                groups = {k: g * (~mask).unsqueeze(-1) for k, g in groups.items()}
                missing = torch.maximum(missing, mask.float())
            return self.encoder(groups, missing, ctx["seg"])

        def latent_of(self, ctx):
            return self.states(ctx)[:, ctx["seg"].shape[1] // 2]

        def body_similarity(self, z, T):
            """Cosine between the projected body latent [B,*] and word embeddings T [M,D] -> [B,M]."""
            return F.normalize(self.fuse(z), dim=-1) @ T.t()

        # ---- losses ----

        def predict_loss(self, R, ctx):
            """Vision -> body: the 17 projected regions of each window frame predict that second's body targets."""
            B = R.shape[0]
            frames = R.view(B, -1, REGIONS, R.shape[-1])
            h = F.relu(self.region_in(frames)).flatten(2)
            out = self.region_head(h)  # [B,F,out]
            valid = ctx["frame_valid"]
            target = ctx["frame_targets"]
            if self.oracle_classes:
                label = torch.where(target["identity"] >= 0, target["identity"], torch.full_like(target["identity"], self.oracle_classes))
                loss = F.cross_entropy(out[valid], label[valid])
                return loss
            contact, change, aperture, view = out.split([N_PATCH, N_PATCH, 2, 2], dim=-1)
            nv = ctx["frame_next_valid"]
            parts = [F.binary_cross_entropy_with_logits(contact[valid], target["contact"][valid]),
                     F.binary_cross_entropy_with_logits(view[valid], target["in_view"][valid]),
                     F.mse_loss(aperture[valid], target["aperture"][valid])]
            parts.append(F.mse_loss(change[nv], target["change"][nv]) if nv.any() else out.sum() * 0)
            return sum(parts)

        def episode_loss(self, T, w, ctx, scale):
            """Utterances matched (soft-MIL) against all regions of their episode; weak alignment, not the named object."""
            sel = ctx.get("bag_sel")
            if sel is None or len(sel) < 2:
                return None
            images = ctx["bag_images"]
            Rb = self.enc_regions(images.flatten(1, 2))  # [Be, E*17, D]
            valid = ctx["bag_valid"].repeat_interleave(REGIONS, dim=1)
            log_prior = torch.full(valid.shape, -np.inf, device=Rb.device, dtype=Rb.dtype)
            log_prior = torch.where(valid, -torch.log(valid.sum(1, keepdim=True).to(Rb.dtype)), log_prior)
            s = torch.einsum("brd,md->brm", Rb, T[sel])
            return weighted_nce(scale * soft_pool(s, log_prior, self.tau_p), w[sel])

        def consistency_loss(self, w, ctx, scale):
            """Regions of two frames of one episode match each other (soft-MIL over region pairs) against other episodes."""
            sel = ctx.get("bag_sel")
            if sel is None or len(sel) < 2:
                return None
            Rb = self.enc_regions(ctx["bag_images"].flatten(1, 2)).view(len(sel), -1, REGIONS, self.word.embedding_dim)
            rows = torch.arange(len(sel), device=Rb.device)
            a, b = Rb[rows, ctx["bag_ab"][:, 0]], Rb[rows, ctx["bag_ab"][:, 1]]
            s = torch.einsum("ird,jqd->ijrq", a, b).flatten(2)
            logits = scale * self.tau_p * torch.logsumexp(s / self.tau_p, dim=2)
            episode = ctx["bag_episode"]
            same = (episode[:, None] == episode[None, :]) & ~torch.eye(len(sel), dtype=torch.bool, device=Rb.device)
            return weighted_nce(logits.masked_fill(same, -1e4), w[sel])

        def mask_loss(self, ctx):
            """Hide some body seconds, predict them from the rest of their clip."""
            states = self.states(ctx, ctx["mask"])
            target = torch.cat([ctx["groups"][k] for k in self.encoder.names], -1)
            chosen = ctx["mask"]
            if not chosen.any():
                return states.sum() * 0
            return F.mse_loss(self.decoder(states)[chosen], target[chosen])

        def pair_scores(self, v, t, n, ctx=None):
            """Training score matrix [B,B] (image i against word bag j): soft-MIL picture score (uniform region prior,
            tau_p) plus, with a body and kept pairs, beta * body-word cosine (late fusion). Returns R, T, scores."""
            R, T, pic = super().pair_scores(v, t, n)
            if self.encoder is not None and ctx is not None and ctx["keep"].any():
                # Body pathway: the body latent adds a similarity term to the score (late fusion), so the language
                # loss depends on the body; gradients reach the regions, words and every body sub-encoder.
                z = self.latent_of(ctx)
                pic = pic + ctx["keep"].unsqueeze(1) * self.beta * self.body_similarity(z, T)
            return R, T, pic

        @torch.no_grad()
        def teacher_logits(self, v, t, n, ctx):
            """Detached scaled combined (picture + body) logits [B,B], exactly as in training; the body is always kept."""
            return self.scale() * self.pair_scores(v, t, n, dict(ctx, keep=torch.ones_like(ctx["keep"])))[2]

        def losses(self, v, t, n, w, ctx=None, weights=None, target=None, temp=2.0):
            """Dict of loss terms. 'match' is the picture-only soft-MIL term (always on); an auxiliary term is computed
            only when its weight is positive. With a body, 'match' includes the body pathway. 'distill' (weights key,
            needs `target`, detached teacher logits) pulls the batch's scaled logits toward the teacher's."""
            scale = self.scale()
            R, T, pic = self.pair_scores(v, t, n, ctx)
            out = {"match": weighted_nce(scale * pic, w)}
            weights = weights or {}
            if weights.get("distill", 0) > 0 and target is not None:
                out["distill"] = distill_kl(scale * pic, target.detach(), w, temp)
            if weights.get("pred", 0) > 0:
                out["pred"] = self.predict_loss(R, ctx)
            if weights.get("episode", 0) > 0:
                out["episode"] = self.episode_loss(T, w, ctx, scale)
            if weights.get("consist", 0) > 0:
                out["consist"] = self.consistency_loss(w, ctx, scale)
            if weights.get("mask", 0) > 0:
                out["mask"] = self.mask_loss(ctx)
            if weights.get("bodymatch", 0) > 0:
                if self.encoder is None or ctx is None:
                    raise ValueError("The bodymatch loss needs a body")
                # body-only word matching on every pair of the batch (no body dropout)
                out["bodymatch"] = weighted_nce(scale * self.body_similarity(self.latent_of(ctx), T), w)
            return {k: v for k, v in out.items() if v is not None}

        def forward_loss(self, v, t, n, w, ctx=None, weights=None, target=None, temp=2.0):
            terms = self.losses(v, t, n, w, ctx, weights, target, temp)
            return terms["match"] + sum((weights or {})[k] * terms[k] for k in terms if k != "match")

    return Additive


def AdditiveMIL(vocab, emb_dim, tau_p=0.1, dims=None, oracle_classes=0):
    """Build the model (the class subclasses the pinned upstream RegionMIL, imported lazily)."""
    return _model_class()(vocab, emb_dim, tau_p, dims, oracle_classes)
