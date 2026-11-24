"""Joint contrastive and prefix-conditioned generative embodied learner."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from nursery.learner.losses import soft_pool, weighted_nce

from .language import CausalLanguage
from .sensory import BodyStream, StaticVisionGaze, TemporalVideoStream


CONDITIONS = ("V", "V+G", "V+G+B")


class UnifiedEmbodiedLearner(nn.Module):
    def __init__(self, vocab_size, *, input_dim=768, grid=16, body_dims=None,
                 condition="V", lambda_gen=0.2, aux_weight=0.25, tau_m=0.1,
                 max_words=32, bow_only=False):
        super().__init__()
        if condition not in CONDITIONS or not math.isfinite(lambda_gen) or lambda_gen < 0:
            raise ValueError("Unknown condition or invalid generative loss weight")
        if condition == "V+G+B" and body_dims is None:
            raise ValueError("V+G+B requires body group dimensions")
        if not math.isfinite(aux_weight) or aux_weight < 0:
            raise ValueError("Invalid auxiliary loss weight")
        if not math.isfinite(tau_m) or tau_m <= 0:
            raise ValueError("Expected a positive modality pooling temperature")
        self.condition, self.lambda_gen = condition, lambda_gen
        self.aux_weight, self.tau_m = aux_weight, tau_m
        self.static = StaticVisionGaze(input_dim, grid)
        self.temporal = TemporalVideoStream(input_dim)
        self.language = CausalLanguage(vocab_size, max_words)
        self.bow_only = bow_only
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.body = BodyStream(body_dims) if body_dims is not None else None

    def enc_text(self, words, counts):
        if self.bow_only:
            ordered = words.sort(dim=1, descending=True).values
            vectors, content = self.language.encode_tokens(ordered, counts, bow_only=True)
            return F.normalize(vectors.masked_fill(~content[..., None], 0).sum(1), dim=-1)
        return self.language.encode(words, counts)

    def temporal_frames(self, tokens, ctx):
        """Pool each frozen frame for the Phase 1 temporal window adapter."""
        mean = tokens.mean(dim=-2)
        if self.condition == "V" or "gaze_point" not in ctx:
            return mean
        # Match Phase 1 steering: CLS keeps its share, and the close-up
        # receives half the frame mass when gaze is valid.
        valid = ctx["gaze_valid"] & ctx["same"] & ctx["keep"][:, None]
        point = ctx["gaze_point"].masked_fill(~valid[..., None], 0)
        prior = self.static.frame_prior(point, valid)
        pooled = torch.einsum("bfk,bfkd->bfd", prior.exp(), tokens)
        if "closeup" in ctx:
            pooled = (pooled + ctx["closeup"].masked_fill(~valid[..., None], 0)) * 0.5
        return torch.where(valid[..., None], pooled, mean)

    def encode_sensory(self, video, frame_valid, temporal=None, gaze=None, body=None,
                       *, remove_extras=False):
        if remove_extras or self.condition == "V":
            gaze = None
        if remove_extras or self.condition != "V+G+B":
            body = None
        static, regions, log_prior = self.static(video, frame_valid, gaze)
        no_input = torch.zeros(len(video), device=video.device, dtype=torch.bool)
        motion, motion_active = torch.zeros_like(static), no_input
        if temporal is not None:
            same = temporal["same"]
            motion = self.temporal(temporal["frames"], same)
            motion_active = same.any(dim=1)
        embodiment, body_active = torch.zeros_like(static), no_input
        if body is not None:
            embodiment, body_active = self.body(body)
        active = torch.stack((torch.ones_like(no_input), motion_active, body_active), dim=1)
        tokens = torch.stack((F.normalize(static, dim=-1), F.normalize(motion, dim=-1),
                              F.normalize(embodiment, dim=-1)), dim=1)
        tokens = tokens.masked_fill(~active[..., None], 0)
        return tokens, active, regions, log_prior

    def score(self, sensory, text):
        return self._scores(sensory, text)[0]

    def _token_streams(self, sensory, flat_h):
        tokens, active, regions, log_prior = sensory
        static = self.static.score(regions, log_prior, flat_h)
        motion, body = (tokens[:, 1:] @ flat_h.T).unbind(1)
        streams = torch.stack((static, motion, body), dim=-1)
        active_prior = -active.sum(-1, keepdim=True).float().log()
        return streams, active_prior.expand_as(active).masked_fill(~active, -torch.inf)

    def _scores(self, sensory, text):
        per_word = isinstance(text, tuple)
        if per_word:
            tokens, content = text
            query_ids, token_ids = content.nonzero(as_tuple=True)
            flat_h = tokens[query_ids, token_ids]
        else:
            flat_h = text
        streams, active_prior = self._token_streams(sensory, flat_h)
        # soft_pool reduces axis 1, so streams are the regions and tokens
        # are the independent queries.
        pooled = soft_pool(streams.transpose(1, 2), active_prior, self.tau_m)
        if not per_word:
            return pooled, streams

        lengths = content.sum(1).clamp_min(1)
        scores = pooled.new_zeros(pooled.shape[0], len(content))
        per_stream = streams.new_zeros(streams.shape[0], len(content), 3)
        return (scores.index_add(1, query_ids, pooled) / lengths,
                per_stream.index_add(1, query_ids, streams) / lengths[:, None])

    def posterior_routing(self, sensory, text):
        """Posterior stream probabilities for each word vector and clip."""
        streams, active_prior = self._token_streams(sensory, text)
        return (streams / self.tau_m + active_prior[:, None, :]).softmax(dim=-1)

    def forward(self, words, counts, video, frame_valid, temporal=None, gaze=None, body=None):
        # InfoNCE queries use the unconditional text pass. The sensory prefix
        # only enters the separate LM loss, with the same text encoder weights.
        if self.bow_only:
            # Keep floating-point reduction order identical for permutations.
            words_for_retrieval = words.sort(dim=1, descending=True).values
            text = self.language.encode_tokens(words_for_retrieval, counts, bow_only=True)
        else:
            text = self.language.encode_tokens(words, counts)
        sensory = self.encode_sensory(video, frame_valid, temporal, gaze, body)
        tokens, active, *_ = sensory
        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        scores, streams = self._scores(sensory, text)
        logits = scale * scores
        loss = weighted_nce(logits, logits.new_ones(len(logits)))
        if self.lambda_gen:
            loss = loss + self.lambda_gen * self.language.lm_loss(words, counts, tokens, active)
        if self.training and self.aux_weight:
            terms = []
            for stream, valid in zip(streams.unbind(-1), active.unbind(-1)):
                # Keep matched pairs on the diagonal in both NCE directions.
                stream_logits = scale * stream[valid][:, valid]
                if len(stream_logits):
                    terms.append(weighted_nce(stream_logits, logits.new_ones(len(stream_logits))))
            if terms:
                loss = loss + self.aux_weight * torch.stack(terms).mean()
        return {"loss": loss, "logits": logits, "text": text}
