"""The learner: upstream RegionMIL (frozen DINOv2 regions -> linear projection -> word vectors) plus training-only
body losses (docs/learner.md). Picture-only evaluation never touches the body modules."""

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _upstream_module
from .body import N_PATCH, BodyEncoder

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


class BodyData:
    """The body store, pair metadata and episode bags on one device; builds the per-batch context.

    pairs: ps/pt session row and centre frame per training pair [N]; win_frames [N,Fmax] frames of the pair's
    image window (-1 pad, same order as upstream FramePairs); bags [S,T,E] frames of each second's episode (-1 none),
    bag_episode [S,T] its episode id;
    cache/row_of: the region cache in RAM and its row per (session, frame), to fetch bag images.
    """

    def __init__(self, store, ps, pt, win_frames, bags, bag_episode, cache, row_of, device, half=5, mask_prob=0.0, dropout=0.5):
        self.store, self.half, self.mask_prob, self.dropout, self.device = store, half, mask_prob, dropout, device
        t = lambda a, dtype=None: torch.as_tensor(a, dtype=dtype, device=device)
        self.groups = {k: t(v) for k, v in store["groups"].items()}
        self.targets = {k: t(v) for k, v in store["targets"].items()}
        self.clip, self.present, self.next_valid = t(store["clip"]), t(store["present"]), t(store["next_valid"])
        self.ps, self.pt, self.win_frames = t(ps), t(pt), t(win_frames)
        self.bags, self.bag_episode, self.cache, self.row_of = bags, bag_episode, cache, row_of
        self.offsets = torch.arange(-half, half + 1, device=device)
        self.dims = {k: v.shape[-1] for k, v in store["groups"].items()}

    def window(self, s, t):
        """Window inputs for sessions s and centre frames t (device long tensors)."""
        steps = t[:, None] + self.offsets
        inside = (steps >= 0) & (steps < self.clip.shape[1])
        steps = steps.clamp(0, self.clip.shape[1] - 1)
        seg = torch.where(inside & self.present[s[:, None], steps], self.clip[s[:, None], steps],
                          torch.full_like(steps, -2))
        groups = {k: g[s[:, None], steps] for k, g in self.groups.items()}
        return groups, (seg == -2).float(), seg

    def batch(self, idx, generator, training=True, bag_frames=9):
        """Context for pair indices `idx` (numpy/long): window inputs, vision targets, bags, masked-modelling mask."""
        idx = torch.as_tensor(idx, device=self.device)
        s, t = self.ps[idx], self.pt[idx]
        groups, missing, seg = self.window(s, t)
        half = self.half
        same = (seg == seg[:, half:half + 1]) & (seg >= 0)
        ctx = {"groups": groups, "missing": missing, "seg": seg, "same": same}
        # body dropout: a dropped pair is scored from the picture alone
        ctx["keep"] = ((torch.rand(len(s), generator=generator) >= self.dropout).float().to(self.device)
                       if training else torch.ones(len(s), device=self.device))
        if training and self.mask_prob > 0:
            pick = torch.rand(seg.shape, generator=generator).to(self.device) < self.mask_prob
            ctx["mask"] = pick & same
        wf = self.win_frames[idx]
        valid = wf >= 0
        fr = wf.clamp(min=0)
        ctx["frame_valid"] = valid
        ctx["frame_targets"] = {k: v[s[:, None], fr] for k, v in self.targets.items()}
        ctx["frame_next_valid"] = self.next_valid[s[:, None], fr] & valid
        pick = []
        if self.bags is not None:
            bag = torch.as_tensor(self.bags[s.cpu().numpy(), t.cpu().numpy()])  # [B,E] frames
            pick = torch.nonzero((bag >= 0).sum(1) >= 2).squeeze(1)
        if len(pick):
            frames = bag[pick].numpy()
            rows = self.row_of[s.cpu().numpy()[pick.numpy()][:, None], np.maximum(frames, 0)]
            images = torch.from_numpy(self.cache[rows.reshape(-1)].astype(np.float32)).to(self.device)
            ctx["bag_sel"] = pick.to(self.device)
            ctx["bag_images"] = images.view(len(pick), frames.shape[1], REGIONS, -1)
            ctx["bag_valid"] = torch.from_numpy(frames >= 0).to(self.device)
            ctx["bag_episode"] = torch.as_tensor(self.bag_episode[s.cpu().numpy()[pick.numpy()], t.cpu().numpy()[pick.numpy()]]).to(self.device)
            # two distinct frames of each bag for the consistency loss
            n = ctx["bag_valid"].sum(1)
            a = (torch.rand(len(pick), generator=generator) * n.cpu()).long()
            b = (a + 1 + (torch.rand(len(pick), generator=generator) * (n.cpu() - 1)).long()) % n.cpu()
            ctx["bag_ab"] = torch.stack([a, b], 1).to(self.device)
        return ctx


@lru_cache(maxsize=None)
def _model_class():
    region = _upstream_module("train_region_mil")

    class LearnerMIL(region.RegionMIL):
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

        def scale(self):
            return self.logit_scale.clamp(max=np.log(100)).exp()

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
            R, T = self.enc_regions(v), self.enc_text(t, n)
            s = torch.einsum("brd,md->brm", R, T)
            uniform = torch.full(s.shape[:2], -np.log(s.shape[1]), device=s.device, dtype=s.dtype)
            pic = soft_pool(s, uniform, self.tau_p)
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

    return LearnerMIL


def LearnerMIL(vocab, emb_dim, tau_p=0.1, dims=None, oracle_classes=0):
    """Build the model (the class subclasses the pinned upstream RegionMIL, imported lazily)."""
    return _model_class()(vocab, emb_dim, tau_p, dims, oracle_classes)


JOINT = {"width": 128, "layers": 2, "heads": 4, "ff": 512, "dropout": 0.0}  # bounded pilot settings


@lru_cache(maxsize=None)
def _joint_class():
    region = _upstream_module("train_region_mil")

    class JointMIL(region.RegionMIL):
        """Joint visual-body learner (`--arch joint`, docs/learner.md): the window's projected region tokens and the
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
