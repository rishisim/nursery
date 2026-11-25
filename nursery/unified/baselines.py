"""Video-only U3 baselines on the shared frozen DINO features."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nursery.learner.losses import weighted_nce
from .language import CausalLanguage


class CLIPGlobal(nn.Module):
    condition = "CLIP-Global"
    bow_only = False

    def __init__(self, vocab_size, *, input_dim=768, max_words=32, lambda_gen=0.2):
        super().__init__()
        if not math.isfinite(lambda_gen) or lambda_gen < 0:
            raise ValueError("Invalid generative loss weight")
        self.visual = nn.Sequential(nn.LayerNorm(input_dim), nn.Dropout(0.1), nn.Linear(input_dim, 512))
        self.language = CausalLanguage(vocab_size, max_words)
        # EOS retrieval bypasses the unified learner's lexical residual.
        self.language.ctx_proj.requires_grad_(False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.lambda_gen = lambda_gen

    def encode_sensory(self, video, frame_valid):
        cls = self.visual(video[:, :, 0].masked_fill(~frame_valid[..., None], 0))
        pooled = cls.masked_fill(~frame_valid[..., None], 0).sum(1)
        return F.normalize(pooled / frame_valid.sum(1, keepdim=True).clamp_min(1), dim=-1)

    def enc_text(self, words, counts):
        return self.language.encode_eos(words, counts)

    def score(self, sensory, text):
        return sensory @ text.T

    def forward(self, words, counts, video, frame_valid):
        visual = self.encode_sensory(video, frame_valid)
        text = self.enc_text(words, counts)
        logits = self.logit_scale.clamp(max=math.log(100)).exp() * self.score(visual, text)
        active = torch.ones_like(counts[:, None], dtype=torch.bool)
        loss = weighted_nce(logits, logits.new_ones(len(words)))
        loss = loss + self.lambda_gen * self.language.lm_loss(words, counts, visual[:, None], active)
        return dict(loss=loss, logits=logits, text=text)


class BabyLLaVANursery(nn.Module):
    condition = "BabyLLaVA-Nursery"
    bow_only = False

    def __init__(self, vocab_size, *, input_dim=768, max_words=32, max_frames=16, p_uncond=0.25):
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 256),
                                    nn.GELU(), nn.Linear(256, 256))
        self.prefix_position = nn.Parameter(torch.zeros(16 + max_frames, 256))
        self.language = CausalLanguage(vocab_size, max_words)
        self.p_uncond = p_uncond

    def encode_sensory(self, video, frame_valid):
        batch, frames, _, dim = video.shape
        if frames + 16 > len(self.prefix_position):
            raise ValueError("Video exceeds the visual prefix position table")
        weights = frame_valid.float() / frame_valid.sum(1, keepdim=True).clamp_min(1)
        patches = torch.einsum("bf,bfpd->bpd", weights.to(video.dtype), video[:, :, 1:])
        grid = patches.transpose(1, 2).reshape(batch, dim, 16, 16)
        spatial = F.avg_pool2d(grid, 4, 4).flatten(2).transpose(1, 2)
        cls = video[:, :, 0].masked_fill(~frame_valid[..., None], 0)
        prefix = self.visual(torch.cat((spatial, cls), dim=1))
        return prefix + self.prefix_position[:16 + frames], F.pad(frame_valid, (16, 0), value=True)

    def _predictions(self, words, counts, sensory, *, p_uncond=0.0):
        tokens = self.language._tokens(words, counts)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        prefix, active = sensory
        if self.training and p_uncond > 0:
            keep = torch.rand(len(words), device=words.device) >= p_uncond
            embedding = self.language.word
            hidden = embedding.weight.new_empty((*inputs.shape, embedding.embedding_dim))
            if keep.any():
                hidden[keep] = self.language.hidden_with_prefix(inputs[keep], prefix[keep], active[keep])
            if (~keep).any():
                hidden[~keep] = self.language._hidden(inputs[~keep])
        else:
            hidden = self.language.hidden_with_prefix(inputs, prefix, active)
        return self.language.lm_head(hidden), targets

    def forward(self, words, counts, video, frame_valid):
        prefix = self.encode_sensory(video, frame_valid)
        logits, targets = self._predictions(words, counts, prefix, p_uncond=self.p_uncond)
        return dict(loss=F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0),
                    logits=logits)

    def score_sentences(self, prefix, words, counts, n_choices, *, pmi=True):
        expanded = tuple(t.repeat_interleave(n_choices, dim=0) for t in prefix)
        logits, targets = self._predictions(words, counts, expanded)
        logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        cond = logp.masked_fill(targets.eq(0), 0).sum(1)
        scores = cond - self.language.sentence_logprob(words, counts) if pmi else cond / (counts + 1)
        return scores.reshape(-1, n_choices)

    def score_words(self, prefix, query_ids, prompt_ids):
        prompt = query_ids.new_tensor([self.language.bos, *prompt_ids]).expand(len(prefix[0]), -1)
        hidden = self.language.hidden_with_prefix(prompt, *prefix)
        return self.language.lm_head(hidden[:, -1]).log_softmax(-1)[:, query_ids]


BASELINE_CONDITIONS = (CLIPGlobal.condition, BabyLLaVANursery.condition)
BASELINE_MODELS = {cls.condition: cls for cls in (CLIPGlobal, BabyLLaVANursery)}
