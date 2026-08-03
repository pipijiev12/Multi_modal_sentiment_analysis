# -*- coding: utf-8 -*-
"""M3SA: modulating unimodal and cross-modal dynamics (EMNLP 2021)."""
import torch
from torch import nn
import torch.nn.functional as F

from .modern_layers import (AttentivePool, InputEmbedding, SelfAttentionBlock,
                            opt_value)


class UnimodalEncoder(nn.Module):
    def __init__(self, input_dim, dim, heads, layers, dropout):
        super(UnimodalEncoder, self).__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([
            SelfAttentionBlock(dim, heads, dropout) for _ in range(layers)
        ])
        self.pool = AttentivePool(dim)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.pool(x)


class M3SA(nn.Module):
    """Soft-filter M3SA with the paper's modulation loss and baseline embeddings."""
    def __init__(self, opt):
        super(M3SA, self).__init__()
        if len(opt.input_dims) != 3:
            raise ValueError('M3SA requires textual, visual and acoustic modalities')
        self.input_embed = InputEmbedding(opt)
        dim = int(opt_value(opt, 'm3sa_dim', 128))
        heads = int(opt_value(opt, 'm3sa_heads', 4))
        layers = int(opt_value(opt, 'm3sa_layers', 2))
        dropout = float(opt_value(opt, 'm3sa_dropout', 0.1))
        self.auxiliary_weight = float(opt_value(opt, 'm3sa_aux_weight', 1.0))
        self.filter_scale = float(opt_value(opt, 'm3sa_filter_scale', 10.0))
        self.filter_penalty_weight = float(opt_value(opt, 'm3sa_filter_penalty', 0.01))

        self.encoders = nn.ModuleList([
            UnimodalEncoder(input_dim, dim, heads, layers, dropout)
            for input_dim in opt.input_dims
        ])
        self.multimodal_shift = nn.Linear(dim * 3, dim)
        self.filters = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 2), nn.ReLU()) for _ in range(3)
        ])
        self.baselines = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim)) for _ in range(3)
        ])
        self.fusion_norm = nn.LayerNorm(dim)
        # The classifier is deliberately shared by unimodal and multimodal paths.
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, opt.output_dim))
        self._unimodal_outputs = None
        self._filter_penalty = None

    def forward(self, in_modalities):
        modalities = self.input_embed(in_modalities)
        encoded = [encoder(x) for encoder, x in zip(self.encoders, modalities)]
        common = self.multimodal_shift(torch.cat(encoded, dim=1))
        filtered = []
        penalties = []
        for index, feature in enumerate(encoded):
            shift = F.relu(common - feature)
            assignment = torch.softmax(self.filter_scale * self.filters[index](shift), dim=1)
            gate = assignment[:, :1]
            penalties.append(1.0 - (assignment[:, 0] - assignment[:, 1]).pow(2))
            baseline = self.baselines[index].expand_as(feature)
            filtered.append(gate * feature + (1.0 - gate) * baseline)

        self._unimodal_outputs = [self.classifier(feature) for feature in encoded]
        self._filter_penalty = torch.stack(penalties, dim=1).mean()
        fused = self.fusion_norm(filtered[0] + filtered[1] + filtered[2])
        return self.classifier(fused)

    def auxiliary_loss(self, targets):
        """Harmonic-mean cross-modal modulation loss from equations 11--13."""
        if self._unimodal_outputs is None or self.auxiliary_weight == 0:
            return targets.new_tensor(0.0)
        losses = []
        for prediction in self._unimodal_outputs:
            prediction = prediction.reshape_as(targets)
            per_element = torch.abs(prediction - targets)
            losses.append(per_element.reshape(per_element.size(0), -1).mean(dim=1))
        stacked = torch.stack(losses, dim=1).clamp_min(1e-6)
        harmonic = 3.0 / torch.sum(1.0 / stacked, dim=1)
        # Each modality is weighted using the losses of the other modalities.
        weights = torch.stack([
            harmonic * stacked[:, 1] * stacked[:, 2],
            harmonic * stacked[:, 0] * stacked[:, 2],
            harmonic * stacked[:, 0] * stacked[:, 1]
        ], dim=1).detach()
        modulated = torch.sum(stacked * weights, dim=1).mean()
        penalty = targets.new_tensor(0.0) if self._filter_penalty is None else self._filter_penalty
        return (self.auxiliary_weight * modulated +
                self.filter_penalty_weight * penalty)
