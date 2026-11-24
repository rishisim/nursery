"""One causal text transformer shared by retrieval and prefix-conditioned LM."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalLanguage(nn.Module):
    def __init__(self, vocab_size, max_words=32):
        super().__init__()
        self.bos, self.eos = vocab_size, vocab_size + 1
        self.word = nn.Embedding(vocab_size + 2, 256, padding_idx=0)
        self.position = nn.Embedding(max_words + 2, 256)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(256, 4, dim_feedforward=1024, dropout=0.1,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(256)
        self.project = nn.Linear(256, 512)
        self.prefix = nn.Linear(512, 256)
        self.prefix_position = nn.Parameter(torch.zeros(3, 256))
        self.lm_head = nn.Linear(256, vocab_size + 2)
        self.ctx_proj = nn.Linear(256, 256)
        nn.init.zeros_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)

    def _tokens(self, words, counts):
        words = F.pad(words, (1, 1))
        words[:, 0] = self.bos
        return words.scatter(1, counts[:, None] + 1, self.eos)

    def _hidden(self, words, sensory=None, active=None):
        positions = torch.arange(words.shape[1], device=words.device)
        x = self.word(words) + self.position(positions)
        padding = words.eq(0)
        prefix_len = 0
        if sensory is not None:
            prefix_len = sensory.shape[1]
            prefix = self.prefix(sensory.masked_fill(~active[..., None], 0))
            x = torch.cat((prefix + self.prefix_position[:prefix_len], x), dim=1)
            padding = torch.cat((~active, padding), dim=1)
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        # Sensory tokens can see each other. Only text is autoregressive.
        causal[:prefix_len, :prefix_len] = False
        for layer in self.layers:
            x = layer(x, src_mask=causal, src_key_padding_mask=padding)
        return self.norm(x[:, prefix_len:])

    def encode_tokens(self, words, counts, bow_only: bool = False):
        tokens = self._tokens(words, counts)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        content = (positions > 0) & (positions <= counts[:, None]) & tokens.ne(0)
        h = self.word(tokens)
        if not bow_only:
            h = h + self.ctx_proj(self._hidden(tokens))
        return F.normalize(self.project(h), dim=-1), content

    def encode(self, words, counts):
        tokens = self._tokens(words, counts)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        content = (positions > 0) & (positions <= counts[:, None]) & tokens.ne(0)
        length = content.sum(1, keepdim=True).clamp_min(1)
        mask = ~content[..., None]
        lexical_mean = self.word(tokens).masked_fill(mask, 0).sum(1) / length
        ctx_mean = self._hidden(tokens).masked_fill(mask, 0).sum(1) / length
        pooled = lexical_mean + self.ctx_proj(ctx_mean)
        return F.normalize(self.project(pooled), dim=-1)

    def lm_loss(self, words, counts, sensory=None, active=None, p_uncond: float = 0.25):
        tokens = self._tokens(words, counts)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        if self.training and sensory is not None and p_uncond:
            keep = torch.rand(len(words), device=words.device) >= p_uncond
            hidden = self.word.weight.new_empty((*inputs.shape, self.word.embedding_dim))
            if keep.any():
                hidden[keep] = self._hidden(inputs[keep], sensory[keep], active[keep])
            if (~keep).any():
                hidden[~keep] = self._hidden(inputs[~keep])
        else:
            hidden = self._hidden(inputs, sensory, active)
        logits = self.lm_head(hidden)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)

    def sentence_logprob(self, words, counts):
        """Unconditional log P(words, EOS | BOS) for each sentence."""
        tokens = self._tokens(words, counts)
        logits = self.lm_head(self._hidden(tokens[:, :-1]))
        targets = tokens[:, 1:]
        logprob = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        return logprob.masked_fill(targets.eq(0), 0).sum(1)
