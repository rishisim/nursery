"""Utterance-conditioned routing on the active sensory simplex."""

import torch
from torch import nn


class ModalityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("b_init", torch.tensor([1.5, -1.5, -0.5]))

    def forward(self, text, active):
        logits = self.mlp(text) + self.b_init
        return logits.masked_fill(~active, -torch.inf).softmax(dim=-1)
