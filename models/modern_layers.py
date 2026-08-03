# -*- coding: utf-8 -*-
"""Shared layers for the newer multimodal sentiment models.

The project targets PyTorch 1.8, so attention is kept sequence-first instead
of relying on the newer ``batch_first`` argument.
"""
import torch
from torch import nn
import torch.nn.functional as F


def opt_value(opt, name, default):
    """Read an optional model setting while preserving old config files."""
    return getattr(opt, name, default)


class InputEmbedding(nn.Module):
    """Turn the repository's token ids into vectors when required."""
    def __init__(self, opt):
        super(InputEmbedding, self).__init__()
        self.enabled = bool(getattr(opt, 'embedding_enabled', False))
        if self.enabled:
            matrix = torch.tensor(opt.lookup_table, dtype=torch.float)
            trainable = bool(getattr(opt, 'embedding_trainable', False))
            self.embedding = nn.Embedding.from_pretrained(matrix, freeze=not trainable)

    def forward(self, modalities):
        if not self.enabled:
            return modalities
        return [self.embedding(x) if x.dim() == 2 else x for x in modalities]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_mult=4):
        super(SelfAttentionBlock, self).__init__()
        if dim % heads != 0:
            raise ValueError('model dimension {} must be divisible by {} heads'.format(dim, heads))
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(self, x):
        q = self.norm1(x).transpose(0, 1)
        attended, _ = self.attn(q, q, q, need_weights=False)
        x = x + attended.transpose(0, 1)
        return x + self.ff(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_mult=4):
        super(CrossAttentionBlock, self).__init__()
        if dim % heads != 0:
            raise ValueError('model dimension {} must be divisible by {} heads'.format(dim, heads))
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(self, query, context):
        q = self.query_norm(query).transpose(0, 1)
        kv = self.context_norm(context).transpose(0, 1)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        query = query + attended.transpose(0, 1)
        return query + self.ff(self.out_norm(query))


class TokenProjector(nn.Module):
    """Learned queries compress an arbitrary sequence into a fixed token set."""
    def __init__(self, input_dim, dim, token_len, heads, dropout):
        super(TokenProjector, self).__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.tokens = nn.Parameter(torch.randn(1, token_len, dim) * 0.02)
        self.cross = CrossAttentionBlock(dim, heads, dropout)

    def forward(self, x):
        x = self.proj(x)
        tokens = self.tokens.expand(x.size(0), -1, -1)
        return self.cross(tokens, x)


class AttentivePool(nn.Module):
    def __init__(self, dim):
        super(AttentivePool, self).__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class KANLinear(nn.Module):
    """A dependency-free first-order B-spline KAN layer.

    Every input/output edge owns a learnable univariate piecewise-linear
    function over a fixed grid, plus the standard KAN residual base function.
    """
    def __init__(self, in_features, out_features, grid_size=8):
        super(KANLinear, self).__init__()
        if grid_size < 2:
            raise ValueError('KAN grid_size must be at least 2')
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('grid', torch.linspace(-1.0, 1.0, grid_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        bounded = torch.tanh(x)
        step = 2.0 / float(self.grid_size - 1)
        basis = F.relu(1.0 - torch.abs(bounded.unsqueeze(-1) - self.grid) / step)
        spline = torch.einsum('...ig,oig->...o', basis, self.spline_weight)
        base = F.linear(F.silu(x), self.base_weight, self.bias)
        return base + spline


class KAN(nn.Module):
    def __init__(self, dims, grid_size=8, dropout=0.0, final_activation=False):
        super(KAN, self).__init__()
        layers = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(KANLinear(in_dim, out_dim, grid_size))
            if index < len(dims) - 2 or final_activation:
                layers.extend([nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
