"""Shared word-matching, pooling and retained diagnostic losses."""

import torch
import torch.nn.functional as F

LOSSES = ("pred", "episode", "consist", "mask", "bodymatch")
REGIONS = 17


def soft_pool(s, log_prior, tau):
    """Soft-MIL score tau * logsumexp_r(log_prior_r + s_r / tau). s [B,R,M], log_prior [B,R] -> [B,M]."""
    return tau * torch.logsumexp(log_prior.unsqueeze(-1) + s / tau, dim=1)


def weighted_nce(logits, w):
    """Symmetric weighted InfoNCE over a square [B,B] logit matrix (positives on the diagonal)."""
    lab = torch.arange(len(logits), device=logits.device)
    li = F.cross_entropy(logits, lab, reduction="none")
    lt = F.cross_entropy(logits.t(), lab, reduction="none")
    return (0.5 * w * (li + lt)).sum() / w.sum().clamp(min=1e-6)


def distill_kl(logits, target, w, temp):
    """Hinton distillation KL(teacher || student) at temperature `temp`, scaled by temp^2, in both matching directions
    (image -> candidate words, word -> candidate images) like the symmetric language loss. Weighted by pair weight."""
    def kl(s, t):
        return F.kl_div(F.log_softmax(s / temp, -1), F.softmax(t / temp, -1), reduction="none").sum(-1)

    return temp ** 2 * (0.5 * w * (kl(logits, target) + kl(logits.t(), target.t()))).sum() / w.sum().clamp(min=1e-6)
