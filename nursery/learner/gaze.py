"""Two gaze readouts over the unchanged visual word learner.

Inputs contain only crop-coordinate gaze points and frozen close-up image features.
The readout choice and its hyperparameters await the development-only pilot.
"""

from functools import lru_cache
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import distill_kl, soft_pool, weighted_nce
from .video_only import _video_class


def compact_closeup(tokens):
    """Mean frozen patch features, excluding CLS: [..., 1+G*G, D] -> [..., D].

    Average before the trainable projection; never inflate a compact cache back
    to 257 tokens. Normalization happens after projection, as for video.
    """
    if tokens.ndim < 2 or tokens.shape[-2] < 2:
        raise ValueError("Close-up tokens must include CLS and a square patch grid")
    grid = math.isqrt(tokens.shape[-2] - 1)
    if grid * grid != tokens.shape[-2] - 1:
        raise ValueError("Close-up tokens must include CLS and a square patch grid")
    return tokens[..., 1:, :].float().mean(dim=-2)


@lru_cache(maxsize=None)
def _gaze_class():
    class Gaze(_video_class()):
        def __init__(self, vocab, emb_dim, tau_p=0.1, option="additive", point_only=False,
                     num_regions=257, gaze_weight=1.0, steering_sigma=0.15):
            if option not in ("additive", "steering"):
                raise ValueError("Gaze option must be additive or steering")
            grid = math.isqrt(num_regions - 1) if num_regions > 1 else 0
            if grid < 1 or grid * grid + 1 != num_regions:
                raise ValueError("Video tokens must include CLS and a square patch grid")
            if not math.isfinite(steering_sigma) or steering_sigma <= 0:
                raise ValueError("steering_sigma must be positive and finite")
            if not math.isfinite(gaze_weight) or gaze_weight < 0:
                raise ValueError("gaze_weight must be non-negative and finite")
            if option == "steering" and gaze_weight != 1.0:
                raise ValueError("gaze_weight initializes the additive branch only")
            super().__init__(vocab, emb_dim, tau_p)
            self.option, self.point_only = option, point_only
            self.num_regions, self.steering_sigma = num_regions, steering_sigma
            # Do not consume the baseline's initialization/dropout RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                if option == "additive":
                    self.beta = nn.Parameter(torch.tensor(float(gaze_weight)))
                    width = 2 if point_only else 2 + emb_dim
                    self.gaze_encoder = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                                      nn.Linear(128, self.word.embedding_dim))
            axis = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            self.register_buffer("patch_centers", torch.stack((xx, yy), -1).reshape(-1, 2), persistent=False)

        def _inputs(self, R, ctx):
            point, valid = ctx["gaze_point"], ctx["gaze_valid"]
            if point.ndim != 3 or point.shape[-1] != 2 or valid.shape != point.shape[:2]:
                raise ValueError("Gaze points and validity must have shapes [B,F,2] and [B,F]")
            if R.shape[0] != len(point) or R.shape[1] != point.shape[1] * self.num_regions:
                raise ValueError("Gaze frames must match the flattened video window")
            if valid.dtype != torch.bool:
                raise ValueError("Gaze validity must be boolean")
            keep = ctx.get("keep", torch.ones(len(R), device=R.device, dtype=torch.bool))
            if keep.shape != (len(R),) or keep.dtype != torch.bool:
                raise ValueError("Gaze keep must be a boolean [B] mask")
            valid = valid & keep[:, None]
            if not torch.isfinite(point[valid]).all():
                raise ValueError("Valid gaze points must be finite")
            point = torch.where(valid[..., None], point, torch.zeros_like(point))
            closeup = None
            if not self.point_only:
                closeup = ctx["closeup"]
                if closeup.shape != (*valid.shape, self.center_mu.numel()):
                    raise ValueError("Close-up must be compact [B,F,encoder_dim]")
                if not torch.isfinite(closeup[valid]).all():
                    raise ValueError("Valid close-up features must be finite")
                closeup = torch.where(valid[..., None], closeup, torch.zeros_like(closeup))
            return point, valid, closeup

        def score_encoded(self, R, T, ctx=None):
            """Same scoring for training and tests; ctx=None removes gaze exactly.

            Additive: a separate point/close-up branch adds word evidence.
            Steering: a Gaussian spatial prior selects visual patches; one
            compact close-up token per frame shares the visual projection.
            Invalid/padded frames retain the original uniform video prior.
            """
            picture = super().score_encoded(R, T)
            if ctx is None:
                return picture
            point, valid, closeup = self._inputs(R, ctx)
            active = valid.any(dim=1)
            if not active.any():
                return picture
            if self.option == "additive":
                values = point if self.point_only else torch.cat((point, closeup - self.center_mu), -1)
                states = self.gaze_encoder(values)
                pooled = (states * valid[..., None]).sum(1) / valid.sum(1).clamp(min=1)[:, None]
                score = F.normalize(pooled, dim=-1) @ T.t()
                return picture + active[:, None] * self.beta * score

            # Each frame keeps equal mass; CLS keeps its 1/num_regions mass.
            # A point outside DINO's center crop cannot select a visible patch.
            in_crop = valid & ((point >= 0) & (point <= 1)).all(-1)
            distance = (point[:, :, None, :] - self.patch_centers).square().sum(-1)
            patch_prior = torch.log_softmax(-distance / (2 * self.steering_sigma ** 2), -1)
            patch_prior = torch.where(in_crop[..., None], patch_prior,
                                      torch.full_like(patch_prior, -math.log(len(self.patch_centers))))
            cls = torch.full_like(patch_prior[..., :1], -math.log(self.num_regions))
            prior = torch.cat((cls, patch_prior + math.log(1 - 1 / self.num_regions)), -1)
            prior = prior - math.log(point.shape[1])
            similarities = torch.einsum("brd,md->brm", R, T)
            if closeup is not None:
                # Equal mass for the selected sharp view and that frame's bank;
                # absent close-ups contribute no candidate and no reweighting.
                prior = prior - valid[..., None].to(prior.dtype) * math.log(2)
                sharp_prior = torch.where(valid, torch.full_like(valid, -math.log(2 * point.shape[1]), dtype=prior.dtype),
                                          torch.full_like(valid, -math.inf, dtype=prior.dtype))
                sharp = self.enc_regions(closeup)
                similarities = torch.cat((similarities, torch.einsum("bfd,md->bfm", sharp, T)), dim=1)
                prior = torch.cat((prior.flatten(1), sharp_prior), dim=1)
            else:
                prior = prior.flatten(1)
            steered = soft_pool(similarities, prior, self.tau_p)
            # Exact baseline, including its floating-point arithmetic, for
            # removed/all-missing rows in mixed evaluation minibatches.
            return torch.where(active[:, None], steered, picture)

        def losses(self, v, t, n, w, ctx=None, weights=None, target=None, temp=2.0):
            _, _, scores = self.pair_scores(v, t, n, ctx)
            logits = self.scale() * scores
            out = {"match": weighted_nce(logits, w)}
            if (weights or {}).get("distill", 0) > 0 and target is not None:
                out["distill"] = distill_kl(logits, target.detach(), w, temp)
            return out

    return Gaze


def GazeMIL(vocab, emb_dim, tau_p=0.1, option="additive", point_only=False,
            num_regions=257, gaze_weight=1.0, steering_sigma=0.15):
    """Build an unselected gaze option; no labels or speech metadata are inputs.

    gaze_weight initializes the learned additive scalar; steering uses a
    normalized spatial prior and does not have a separate additive weight.
    """
    return _gaze_class()(vocab, emb_dim, tau_p, option, point_only, num_regions,
                         gaze_weight, steering_sigma)
